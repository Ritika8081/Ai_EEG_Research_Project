"""
Test the hypothesis from `report/part3_idea.md`:

    CMA's representation-level benefit decays with training time
    because the model can invert a fixed linear augmentation. A
    schedule that tapers p_apply across training should retain
    sil(subject) gains at 60 epochs.

Setup: reduced 3-subject pool (same as the rest of Part 3). 3 seeds.
Constant p_apply=0.8 (the headline config, known to *raise* sil_subj
to 0.090 at 60 epochs) vs curriculum p_apply from 0.9 → 0.1.
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

SEEDS = [1337, 2024, 7]


def _split_train(set_, val_subjects):
    train_mask = np.isin(set_.subjects, [s for s in np.unique(set_.subjects) if s not in val_subjects])
    val_mask = ~train_mask
    def pick(mask):
        return EpochSet(set_.X[mask], set_.y[mask], set_.subjects[mask], set_.ch_names, set_.pos)
    return pick(train_mask), pick(val_mask)


def run(train_set, test_set, label, curriculum: bool, epochs: int = 60):
    accs = []
    last_model, last_loader = None, None
    for s in SEEDS:
        seed_all(s)
        train, val = _split_train(train_set, val_subjects=[max(np.unique(train_set.subjects))])
        Xtr = per_trial_zscore(train.X)
        Xte = per_trial_zscore(test_set.X)
        bank = build_cma_bank(train_set.pos, n_aug=24, theta_max_deg=4.0, sigma=0.18)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Approx total batches: epochs * (n_train // batch). batch=64.
        total_steps = epochs * max(1, len(train.y) // 64)
        aug = CMAAugmenter(
            bank, p_apply=0.8, curriculum=curriculum,
            p_start=0.9, p_end=0.1, total_steps=total_steps,
        ).to(device)
        cfg = TrainCfg(epochs=epochs, augment_fn=aug)
        m = EEGNet(n_channels=Xtr.shape[1], n_times=Xtr.shape[2])
        tl = make_loader(Xtr, train.y, train.subjects, cfg.batch, shuffle=True)
        te = make_loader(Xte, test_set.y, test_set.subjects, cfg.batch, shuffle=False)
        train_model(m, tl, None, cfg)
        accs.append(evaluate(m, te, cfg.device))
        last_model, last_loader = m, te
        print(f"  [{label}] seed={s}  test={accs[-1]:.3f}")
    feats, ys, subs = extract_features(last_model, last_loader, "cuda" if torch.cuda.is_available() else "cpu")
    diag = cluster_diagnostics(feats, ys, subs)
    arr = np.asarray(accs)
    return {
        "best": float(arr.max()), "worst": float(arr.min()), "mean": float(arr.mean()),
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
    out["constant_p_3subj"] = run(reduced_set, test_set, "constant_p_3subj", curriculum=False)
    out["curriculum_p_3subj"] = run(reduced_set, test_set, "curriculum_p_3subj", curriculum=True)
    out["curriculum_p_8subj"] = run(train_set, test_set, "curriculum_p_8subj", curriculum=True)

    with open("results/curriculum_cma.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSummary:")
    for k, v in out.items():
        print(
            f"  {k:25s}  best={v['best']:.3f}  worst={v['worst']:.3f}  "
            f"sil_cls={v['silhouette_class']:.3f}  sil_subj={v['silhouette_subject']:.3f}"
        )


if __name__ == "__main__":
    main()
