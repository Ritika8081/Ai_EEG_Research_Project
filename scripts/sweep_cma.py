"""
CMA hyperparameter sweep on the reduced (3-subject) training pool.

In `report/part3_idea.md` I hypothesised that CMA's failure to reduce
subject-silhouette might be a configuration issue. The two knobs are
sigma (RBF width, in chord-distance units) and theta_max (rotation angle
ceiling, degrees). We sweep over a small grid and report which
configurations actually lower sil_subject vs baseline (0.062 on reduced
pool, 0.084 on full).

We use 2 seeds and 40 epochs per config to keep wall-time bounded.
This is a diagnostic, not the headline result.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import TEST_SUBJECTS, TRAIN_SUBJECTS, EpochSet, load_data, per_trial_zscore
from src.eegnet import EEGNet
from src.part3_idea import CMAAugmenter, build_cma_bank
from src.seed import seed_all
from src.train import TrainCfg, evaluate, extract_features, make_loader, train_model
from src.eval import cluster_diagnostics

SEEDS = [1337, 2024]
SIGMAS = [0.08, 0.18, 0.30]
THETAS = [2.0, 4.0, 8.0]


def _split_train(set_, val_subjects):
    train_mask = np.isin(set_.subjects, [s for s in np.unique(set_.subjects) if s not in val_subjects])
    val_mask = ~train_mask
    def pick(mask):
        return EpochSet(set_.X[mask], set_.y[mask], set_.subjects[mask], set_.ch_names, set_.pos)
    return pick(train_mask), pick(val_mask)


def one_config(train_set, test_set, sigma, theta_max, epochs):
    bank = build_cma_bank(train_set.pos, n_aug=24, theta_max_deg=theta_max, sigma=sigma)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aug = CMAAugmenter(bank, p_apply=0.8).to(device)
    accs = []
    last_model, last_loader = None, None
    for s in SEEDS:
        seed_all(s)
        train, val = _split_train(train_set, val_subjects=[max(np.unique(train_set.subjects))])
        Xtr = per_trial_zscore(train.X)
        Xte = per_trial_zscore(test_set.X)
        cfg = TrainCfg(epochs=epochs, augment_fn=aug)
        m = EEGNet(n_channels=Xtr.shape[1], n_times=Xtr.shape[2])
        tl = make_loader(Xtr, train.y, train.subjects, cfg.batch, shuffle=True)
        te = make_loader(Xte, test_set.y, test_set.subjects, cfg.batch, shuffle=False)
        train_model(m, tl, None, cfg)
        accs.append(evaluate(m, te, cfg.device))
        last_model, last_loader = m, te
    feats, ys, subs = extract_features(last_model, last_loader, "cuda" if torch.cuda.is_available() else "cpu")
    diag = cluster_diagnostics(feats, ys, subs)
    arr = np.asarray(accs)
    return {
        "best": float(arr.max()), "worst": float(arr.min()), "mean": float(arr.mean()),
        **diag,
    }


def main(epochs: int = 40):
    print("Loading data...")
    train_set = load_data(TRAIN_SUBJECTS)
    test_set = load_data(TEST_SUBJECTS)
    reduced_mask = np.isin(train_set.subjects, [1, 2, 3])
    reduced_set = EpochSet(
        train_set.X[reduced_mask], train_set.y[reduced_mask],
        train_set.subjects[reduced_mask], train_set.ch_names, train_set.pos,
    )

    grid = {}
    for sigma in SIGMAS:
        for theta in THETAS:
            tag = f"sigma{sigma}_theta{theta}"
            print(f"  {tag}")
            grid[tag] = one_config(reduced_set, test_set, sigma, theta, epochs)
            print(f"    best={grid[tag]['best']:.3f}  sil_subj={grid[tag]['silhouette_subject']:.3f}")

    with open("results/sweep_cma.json", "w", encoding="utf-8") as f:
        json.dump(grid, f, indent=2)

    print("\nReduced-pool baseline (no CMA): sil_subj=0.062, best=0.656")
    print("Configurations that improved subject-decoupling vs baseline:")
    for k, v in grid.items():
        if v["silhouette_subject"] < 0.062:
            print(f"  {k}: sil_subj={v['silhouette_subject']:.3f}  best={v['best']:.3f}")


if __name__ == "__main__":
    main()
