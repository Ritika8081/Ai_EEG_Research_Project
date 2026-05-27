"""
Part 3 follow-up — did the depthwise spatial filter actually use the
contrast channels, or did it ignore them?

The bipolar / surface Laplacian comparison is the obvious thing a
reviewer will throw at Part 3, so I wanted a way to check whether
the model leans on the contrasts at all. If it doesn't, my "model
gets to choose between raw and contrast representations" framing is
just dressing on a bipolar montage and I should drop it.

The test: every depthwise filter has a weight vector over all 74
channels (64 raw + 10 contrasts). I split each filter's weight into
the raw part and the contrast part, take the L2-squared share on
the contrasts, and compare it to the random baseline 10/74 ≈ 13.5 %
(what you'd get if all channels mattered equally). If the trained
filters concentrate above the baseline — some far above — the model
did pick up the prior. If they sit at the baseline, I retract the
framing.

The depthwise weight tensor is shape (F1·D, 1, n_channels, 1) =
(16, 1, 74, 1). First 64 channels are raw EEG, last 10 are the
hemispheric contrasts.

Outputs:
    results/part3_filter_analysis.json   per-filter shares + summary
    results/figures/part3_filter_weights.png   bars vs random baseline
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import TRAIN_SUBJECTS, load_data, per_trial_zscore
from src.eegnet import EEGNet
from src.part3_idea import augment_with_contrasts
from src.seed import seed_all
from src.train import TrainCfg, make_loader, train_model

OUT_JSON = "results/part3_filter_analysis.json"
OUT_FIG = "results/figures/part3_filter_weights.png"
N_CONTRAST = 10  # last 10 channels of the 74-channel augmented input
SEED = 1337


def per_filter_contrast_share(model: EEGNet) -> np.ndarray:
    """Per-filter share of L2-squared weight on the contrast channels.

    I use L2-squared rather than L2 because what actually drives a
    filter's output is the variance contribution of each input
    channel, not the raw norm. Squared weights are the right thing
    to share-out."""
    w = model.depthwise.weight.detach().cpu().numpy()  # (F1*D, 1, n_chan, 1)
    w = w[:, 0, :, 0]  # (F1*D, n_chan)
    raw = w[:, :-N_CONTRAST]
    contrast = w[:, -N_CONTRAST:]
    raw_sq = (raw ** 2).sum(axis=1)
    contrast_sq = (contrast ** 2).sum(axis=1)
    return contrast_sq / (raw_sq + contrast_sq + 1e-12)


def main() -> None:
    seed_all(SEED)
    print(f"[part3 filter analysis] seed={SEED}")
    print("Loading EEGBCI training subjects 1-8 (cached after first run)...")
    train_set = load_data(TRAIN_SUBJECTS)

    X_aug = augment_with_contrasts(train_set.X, train_set.ch_names)
    X = per_trial_zscore(X_aug)
    n_chan = X.shape[1]
    assert n_chan == 64 + N_CONTRAST, f"Expected 74 channels, got {n_chan}"
    print(f"  augmented input: {X.shape}  (64 raw + {N_CONTRAST} contrasts)")

    # Snapshot init-time weight distribution as a sanity check on the
    # random baseline. EEGNet's depthwise init is roughly Kaiming-uniform,
    # so the per-filter L2² share should sit near N_CONTRAST / n_chan.
    model_init = EEGNet(n_channels=n_chan, n_times=X.shape[2])
    init_share = per_filter_contrast_share(model_init)
    random_baseline = N_CONTRAST / float(n_chan)

    # Now train EEGNet on the augmented input from scratch.
    cfg = TrainCfg(epochs=60)
    model = EEGNet(n_channels=n_chan, n_times=X.shape[2])
    loader = make_loader(X, train_set.y, train_set.subjects, batch=cfg.batch, shuffle=True)
    print("Training 60 epochs (this matches the Part 3 run in main.py)...")
    train_model(model, loader, None, cfg)
    trained_share = per_filter_contrast_share(model)

    n_filters = len(trained_share)
    n_above = int((trained_share > random_baseline).sum())
    fold_above_random = float(np.mean(trained_share) / random_baseline)

    print(f"\nRandom-baseline contrast share (10/74):  {random_baseline:.3f}")
    print(f"Init-time mean per-filter share:         {float(np.mean(init_share)):.3f}")
    print(f"Trained mean per-filter share:           {float(np.mean(trained_share)):.3f}")
    print(f"Filters above random baseline:           {n_above} / {n_filters}")
    print(f"Mean fold-above-random:                  {fold_above_random:.2f}x")
    print("Per-filter trained shares:")
    for k, s in enumerate(trained_share):
        marker = "  <-- concentrates on contrasts" if s > 0.40 else ""
        print(f"  filter {k:2d}: {s:.3f}{marker}")

    # Bar plot vs the random baseline. The key visual: if bars sit at
    # the dashed line, the contrasts were not used. If they spread
    # above it (some far above), the model chose to lean on them.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    idx = np.arange(n_filters)
    bars = ax.bar(idx, trained_share, color="steelblue", label="trained filters")
    for k, bar in enumerate(bars):
        if trained_share[k] > 0.40:
            bar.set_color("crimson")
    ax.axhline(
        random_baseline,
        color="black",
        linestyle="--",
        label=f"random baseline (10/74 = {random_baseline:.3f})",
    )
    ax.axhline(
        float(np.mean(trained_share)),
        color="darkorange",
        linestyle=":",
        label=f"trained mean = {float(np.mean(trained_share)):.3f}",
    )
    ax.set_xlabel("depthwise spatial filter index (F1·D = 16 filters)")
    ax.set_ylabel("share of weight variance on contrast channels")
    ax.set_title(
        "Per-filter weight share on hemispheric-contrast channels\n"
        "(if contrasts were redundant, bars would sit at the dashed line)"
    )
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0, max(1.0, float(trained_share.max()) * 1.1))
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=140)
    plt.close()

    out = {
        "seed": SEED,
        "n_filters": n_filters,
        "n_contrast_channels": N_CONTRAST,
        "n_total_channels": n_chan,
        "random_baseline": random_baseline,
        "init_mean_share": float(np.mean(init_share)),
        "init_share_per_filter": [float(x) for x in init_share],
        "trained_mean_share": float(np.mean(trained_share)),
        "trained_share_per_filter": [float(x) for x in trained_share],
        "filters_above_baseline": n_above,
        "max_share": float(trained_share.max()),
        "min_share": float(trained_share.min()),
        "mean_fold_above_random": fold_above_random,
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON} and {OUT_FIG}")


if __name__ == "__main__":
    main()
