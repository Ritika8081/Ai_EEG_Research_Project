"""
Part 2 ablation: isolate the contributions of (a) per-trial whitening and
(b) Coordinate-Conditioned Spatial Filters.

Configurations tested (cross-subject, 3 seeds each):
- EEGNet (no change)            <- already in metrics.json
- EEGNet + whitening only       <- Part 2 minus CCSF
- TopoNet, no whitening (CCSF only) <- Part 2 minus whitening
- TopoNet (both)                <- already in metrics.json

We run only the two missing rows here, dump them into
results/ablation_part2.json. The other two rows are pulled from the
existing metrics.json so the slide table is one coherent comparison.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import TEST_SUBJECTS, TRAIN_SUBJECTS, EpochSet, load_data, per_trial_zscore
from src.eegnet import EEGNet
from src.part2_model import TopoNet, _whiten_per_trial
from src.seed import seed_all
from src.train import TrainCfg, evaluate, extract_features, make_loader, train_model
from src.eval import cluster_diagnostics

SEEDS = [1337, 2024, 7]


class WhitenedEEGNet(EEGNet):
    """EEGNet but every input is per-trial whitened first.
    Isolates the contribution of whitening (no CCSF)."""

    def __init__(self, *args, whiten_eps: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.whiten_eps = whiten_eps

    def forward(self, x):
        # x: (B, C, T) or (B, 1, C, T)
        if x.dim() == 4:
            x = x.squeeze(1)
        x = _whiten_per_trial(x, eps=self.whiten_eps)
        return super().forward(x)

    def features(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = _whiten_per_trial(x, eps=self.whiten_eps)
        return super().features(x)


def _split_train(set_: EpochSet, val_subjects):
    train_mask = np.isin(set_.subjects, [s for s in np.unique(set_.subjects) if s not in val_subjects])
    val_mask = ~train_mask
    def pick(mask):
        return EpochSet(set_.X[mask], set_.y[mask], set_.subjects[mask], set_.ch_names, set_.pos)
    return pick(train_mask), pick(val_mask)


def run(model_factory, train_set, test_set, label: str, epochs: int = 60):
    accs, val_accs = [], []
    last_model, last_loader = None, None
    for s in SEEDS:
        seed_all(s)
        train, val = _split_train(train_set, val_subjects=[max(np.unique(train_set.subjects))])
        Xtr = per_trial_zscore(train.X)
        Xva = per_trial_zscore(val.X)
        Xte = per_trial_zscore(test_set.X)
        cfg = TrainCfg(epochs=epochs)
        model = model_factory(Xtr.shape[1], Xtr.shape[2], train.pos)
        tl = make_loader(Xtr, train.y, train.subjects, cfg.batch, shuffle=True)
        vl = make_loader(Xva, val.y, val.subjects, cfg.batch, shuffle=False)
        te = make_loader(Xte, test_set.y, test_set.subjects, cfg.batch, shuffle=False)
        train_model(model, tl, vl, cfg)
        val_accs.append(evaluate(model, vl, cfg.device))
        accs.append(evaluate(model, te, cfg.device))
        last_model, last_loader = model, te
        print(f"  [{label}] seed={s}  val={val_accs[-1]:.3f}  test={accs[-1]:.3f}")
    feats, ys, subs = extract_features(last_model, last_loader, "cuda" if torch.cuda.is_available() else "cpu")
    diag = cluster_diagnostics(feats, ys, subs)
    arr = np.asarray(accs)
    return {
        "label": label,
        "best": float(arr.max()), "worst": float(arr.min()),
        "mean": float(arr.mean()), "std": float(arr.std()),
        **diag,
    }


def main():
    print("Loading data...")
    train_set = load_data(TRAIN_SUBJECTS)
    test_set = load_data(TEST_SUBJECTS)

    reduced_mask = np.isin(train_set.subjects, [1, 2, 3])
    reduced_set = EpochSet(
        train_set.X[reduced_mask], train_set.y[reduced_mask],
        train_set.subjects[reduced_mask], train_set.ch_names, train_set.pos,
    )

    out = {}

    # Whitening only on top of EEGNet.
    def whiten_only(c, t, pos):
        return WhitenedEEGNet(n_channels=c, n_times=t)

    # CCSF only — TopoNet with whitening disabled.
    def ccsf_only(c, t, pos):
        return TopoNet(pos=pos, n_channels=c, n_times=t, use_whitening=False)

    print("\n--- Ablation: full pool (8 subj) ---")
    out["whiten_only_full"] = run(whiten_only, train_set, test_set, "whiten_only_full")
    out["ccsf_only_full"]   = run(ccsf_only,   train_set, test_set, "ccsf_only_full")

    print("\n--- Ablation: reduced pool (3 subj) ---")
    out["whiten_only_reduced"] = run(whiten_only, reduced_set, test_set, "whiten_only_reduced")
    out["ccsf_only_reduced"]   = run(ccsf_only,   reduced_set, test_set, "ccsf_only_reduced")

    with open("results/ablation_part2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSummary:")
    for k, v in out.items():
        print(f"  {k:30s}  best={v['best']:.3f}  worst={v['worst']:.3f}  "
              f"sil_cls={v['silhouette_class']:.3f}  sil_subj={v['silhouette_subject']:.3f}")


if __name__ == "__main__":
    main()
