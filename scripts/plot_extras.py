"""
Generate the two extra figures requested in review:

1. Per-subject test-accuracy bar chart (S9 vs S10 broken out).
2. Topomap of (a) baseline EEGNet depthwise spatial filters, and
   (b) the CCSF-learned filters at each electrode position.

Usage: python scripts/plot_extras.py
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import TEST_SUBJECTS, TRAIN_SUBJECTS, EpochSet, load_data, per_trial_zscore
from src.eegnet import EEGNet
from src.part2_model import TopoNet
from src.seed import seed_all
from src.train import TrainCfg, evaluate, make_loader, train_model

mne.set_log_level("ERROR")

FIG = os.path.join("results", "figures")
os.makedirs(FIG, exist_ok=True)


def _split_train(set_, val_subjects):
    train_mask = np.isin(set_.subjects, [s for s in np.unique(set_.subjects) if s not in val_subjects])
    val_mask = ~train_mask
    def pick(mask):
        return EpochSet(set_.X[mask], set_.y[mask], set_.subjects[mask], set_.ch_names, set_.pos)
    return pick(train_mask), pick(val_mask)


def per_subject_accuracy(train_set, test_set, model_factory, label, seed=1337, epochs=60):
    seed_all(seed)
    train, val = _split_train(train_set, val_subjects=[max(np.unique(train_set.subjects))])
    Xtr = per_trial_zscore(train.X)
    Xte = per_trial_zscore(test_set.X)
    cfg = TrainCfg(epochs=epochs)
    m = model_factory(Xtr.shape[1], Xtr.shape[2], train.pos)
    tl = make_loader(Xtr, train.y, train.subjects, cfg.batch, shuffle=True)
    train_model(m, tl, None, cfg)
    # Per-subject test accuracy.
    out = {}
    m.eval()
    with torch.no_grad():
        for s in np.unique(test_set.subjects):
            mask = test_set.subjects == s
            Xs = per_trial_zscore(test_set.X[mask])
            xt = torch.from_numpy(Xs).float().to(cfg.device)
            yt = torch.from_numpy(test_set.y[mask]).long().to(cfg.device)
            logits = m(xt)
            out[int(s)] = float((logits.argmax(1) == yt).float().mean().item())
    return out, m


def plot_per_subject(results, out_path):
    """results: dict like {label: {S9: acc, S10: acc}}."""
    labels = list(results.keys())
    subjects = sorted(next(iter(results.values())).keys())
    width = 0.8 / len(labels)
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, lab in enumerate(labels):
        vals = [results[lab][s] for s in subjects]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subjects])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Per-subject test accuracy (cross-subject, 8-subj train pool)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eegnet_topomaps(model: EEGNet, ch_names, pos, out_path):
    # EEGNet's depthwise.weight has shape (F2, F1/group=1, C, 1) so per filter
    # the spatial profile is weight[k, 0, :, 0].
    W = model.depthwise.weight.detach().cpu().numpy().squeeze()  # (F2, C)
    n = W.shape[0]
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    info = mne.create_info(ch_names=list(ch_names), sfreq=160.0, ch_types="eeg")
    mont = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(mont, on_missing="ignore")
    for k in range(n_rows * n_cols):
        ax = axes.ravel()[k]
        if k >= n:
            ax.axis("off")
            continue
        v = W[k]
        mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r", contours=0)
        ax.set_title(f"filter {k}", fontsize=8)
    fig.suptitle("EEGNet baseline — depthwise spatial filter topomaps", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ccsf_topomaps(model: TopoNet, ch_names, pos, out_path):
    # CCSF outputs the same shape (F2, C). Compute via the module.
    W = model.spatial.compute_W().detach().cpu().numpy().T  # (F2, C)
    n = W.shape[0]
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    info = mne.create_info(ch_names=list(ch_names), sfreq=160.0, ch_types="eeg")
    mont = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(mont, on_missing="ignore")
    for k in range(n_rows * n_cols):
        ax = axes.ravel()[k]
        if k >= n:
            ax.axis("off")
            continue
        v = W[k]
        mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r", contours=0)
        ax.set_title(f"filter {k}", fontsize=8)
    fig.suptitle("TopoNet — CCSF spatial filter topomaps", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    print(f"Loading data... (epochs={args.epochs})")
    train_set = load_data(TRAIN_SUBJECTS)
    test_set = load_data(TEST_SUBJECTS)

    # Per-subject test for baseline EEGNet vs CCSF-only (the surviving Part-2 half).
    print("Training baseline EEGNet (1 seed) for per-subject + topomap...")
    base_acc, base_model = per_subject_accuracy(
        train_set, test_set,
        lambda c, t, pos: EEGNet(n_channels=c, n_times=t),
        "EEGNet", epochs=args.epochs,
    )
    print(f"  {base_acc}")

    print("Training CCSF-only (1 seed) for per-subject + topomap...")
    ccsf_acc, ccsf_model = per_subject_accuracy(
        train_set, test_set,
        lambda c, t, pos: TopoNet(pos=pos, n_channels=c, n_times=t, use_whitening=False),
        "CCSF-only", epochs=args.epochs,
    )
    print(f"  {ccsf_acc}")

    plot_per_subject(
        {"EEGNet": base_acc, "CCSF only": ccsf_acc},
        os.path.join(FIG, "per_subject_accuracy.png"),
    )
    plot_eegnet_topomaps(base_model, train_set.ch_names, train_set.pos,
                         os.path.join(FIG, "topomap_eegnet.png"))
    plot_ccsf_topomaps(ccsf_model, train_set.ch_names, train_set.pos,
                       os.path.join(FIG, "topomap_ccsf.png"))
    print("Saved per-subject bar + 2 topomap figures.")


if __name__ == "__main__":
    main()
