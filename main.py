"""
Reproducer for all three parts.

Run:    python main.py

Artefacts:
    results/figures/*.png   embeddings + training curves
    results/metrics.json    all reported numbers
    results/log.txt         best/worst run summary

Random seeds are fixed in src/seed.py. We run each experiment with
N_REPEATS different seeds and report both the BEST and WORST run, never
the mean alone — per the assignment's intellectual-honesty constraint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import TEST_SUBJECTS, TRAIN_SUBJECTS, EpochSet, load_data, per_trial_zscore
from src.eegnet import EEGNet
from src.eval import cluster_diagnostics, plot_embeddings
from src.part2_model import TopoNet
from src.part3_idea import augment_with_contrasts
from src.seed import seed_all
from src.train import TrainCfg, evaluate, extract_features, make_loader, train_model

RESULTS = "results"
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

# Multiple seeds per experiment so we can report best AND worst — assignment
# explicitly forbids cherry-picking a single favourable run.
SEEDS = [1337, 2024, 7, 42, 99]
N_SEEDS = 3  # use first N for the heavier experiments to keep runtime bounded


def _split_train(set_: EpochSet, val_subjects: List[int]) -> Tuple[EpochSet, EpochSet]:
    """Pull out a small subject-held-out validation split from the train pool."""
    train_mask = np.isin(set_.subjects, [s for s in np.unique(set_.subjects) if s not in val_subjects])
    val_mask = ~train_mask
    def pick(mask):
        return EpochSet(set_.X[mask], set_.y[mask], set_.subjects[mask], set_.ch_names, set_.pos)
    return pick(train_mask), pick(val_mask)


def _runs_summary(accs: List[float]) -> Dict[str, float]:
    a = np.asarray(accs, dtype=np.float64)
    return {"best": float(a.max()), "worst": float(a.min()), "mean": float(a.mean()), "std": float(a.std())}


def run_baseline(train_set: EpochSet, test_set: EpochSet, label: str, epochs: int = 60) -> Dict[str, float]:
    """Train EEGNet on train_set, evaluate on test_set. Returns best/worst over seeds."""
    print(f"\n[baseline] {label}  train={len(train_set.y)}  test={len(test_set.y)}")
    accs_test, accs_val = [], []
    last_model, last_loader_test = None, None
    for seed in SEEDS[:N_SEEDS]:
        seed_all(seed)
        train, val = _split_train(train_set, val_subjects=[max(np.unique(train_set.subjects))])
        Xtr = per_trial_zscore(train.X)
        Xva = per_trial_zscore(val.X)
        Xte = per_trial_zscore(test_set.X)
        n_chan, n_times = Xtr.shape[1], Xtr.shape[2]
        cfg = TrainCfg(epochs=epochs)
        model = EEGNet(n_channels=n_chan, n_times=n_times)
        tl = make_loader(Xtr, train.y, train.subjects, batch=cfg.batch, shuffle=True)
        vl = make_loader(Xva, val.y, val.subjects, batch=cfg.batch, shuffle=False)
        te = make_loader(Xte, test_set.y, test_set.subjects, batch=cfg.batch, shuffle=False)
        train_model(model, tl, vl, cfg)
        accs_val.append(evaluate(model, vl, cfg.device))
        accs_test.append(evaluate(model, te, cfg.device))
        last_model, last_loader_test = model, te
        print(f"  seed={seed}  val={accs_val[-1]:.3f}  test={accs_test[-1]:.3f}")

    # Save embedding plot using last seed's model.
    feats, ys, subs = extract_features(last_model, last_loader_test, "cuda" if torch.cuda.is_available() else "cpu")
    diag = cluster_diagnostics(feats, ys, subs)
    plot_embeddings(feats, ys, subs, title=f"EEGNet — {label}", out_path=os.path.join(FIG, f"baseline_{label}.png"))

    out = {"label": label, **_runs_summary(accs_test), "val": _runs_summary(accs_val), **diag}
    return out


def run_part2(train_set: EpochSet, test_set: EpochSet, label: str, epochs: int = 60) -> Dict[str, float]:
    print(f"\n[part2 TopoNet] {label}  train={len(train_set.y)}  test={len(test_set.y)}")
    accs_test, accs_val = [], []
    last_model, last_loader = None, None
    for seed in SEEDS[:N_SEEDS]:
        seed_all(seed)
        train, val = _split_train(train_set, val_subjects=[max(np.unique(train_set.subjects))])
        Xtr = per_trial_zscore(train.X)
        Xva = per_trial_zscore(val.X)
        Xte = per_trial_zscore(test_set.X)
        cfg = TrainCfg(epochs=epochs)
        model = TopoNet(pos=train.pos, n_channels=Xtr.shape[1], n_times=Xtr.shape[2])
        tl = make_loader(Xtr, train.y, train.subjects, batch=cfg.batch, shuffle=True)
        vl = make_loader(Xva, val.y, val.subjects, batch=cfg.batch, shuffle=False)
        te = make_loader(Xte, test_set.y, test_set.subjects, batch=cfg.batch, shuffle=False)
        train_model(model, tl, vl, cfg)
        accs_val.append(evaluate(model, vl, cfg.device))
        accs_test.append(evaluate(model, te, cfg.device))
        last_model, last_loader = model, te
        print(f"  seed={seed}  val={accs_val[-1]:.3f}  test={accs_test[-1]:.3f}")
    feats, ys, subs = extract_features(last_model, last_loader, "cuda" if torch.cuda.is_available() else "cpu")
    diag = cluster_diagnostics(feats, ys, subs)
    plot_embeddings(feats, ys, subs, title=f"TopoNet — {label}", out_path=os.path.join(FIG, f"part2_{label}.png"))
    return {"label": label, **_runs_summary(accs_test), "val": _runs_summary(accs_val), **diag}


def run_part3(train_set: EpochSet, test_set: EpochSet, label: str, epochs: int = 60) -> Dict[str, float]:
    """Part 3: append ten pairwise contrasts between symmetric left-right
    motor-cortex electrodes (plus one lateral-vs-midline contrast) to
    the raw 64-channel input. EEGNet then sees 74 channels and can use
    either representation, with the contrasts giving it a ready-made
    lateralization feature it would otherwise have to learn."""
    print(f"\n[part3 hemi-contrasts] {label}  train={len(train_set.y)}  test={len(test_set.y)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_full = augment_with_contrasts(train_set.X, train_set.ch_names)
    Xte_full = augment_with_contrasts(test_set.X, test_set.ch_names)
    print(f"  channels augmented 64 -> {Xtr_full.shape[1]} (64 raw + 10 contrasts)")
    train_mod = EpochSet(Xtr_full, train_set.y, train_set.subjects, train_set.ch_names, train_set.pos)
    test_mod = EpochSet(Xte_full, test_set.y, test_set.subjects, test_set.ch_names, test_set.pos)
    accs_test, accs_val = [], []
    last_model, last_loader = None, None
    for seed in SEEDS[:N_SEEDS]:
        seed_all(seed)
        train, val = _split_train(train_mod, val_subjects=[max(np.unique(train_mod.subjects))])
        Xtr = per_trial_zscore(train.X)
        Xva = per_trial_zscore(val.X)
        Xte = per_trial_zscore(test_mod.X)
        cfg = TrainCfg(epochs=epochs)
        model = EEGNet(n_channels=Xtr.shape[1], n_times=Xtr.shape[2])
        tl = make_loader(Xtr, train.y, train.subjects, batch=cfg.batch, shuffle=True)
        vl = make_loader(Xva, val.y, val.subjects, batch=cfg.batch, shuffle=False)
        te = make_loader(Xte, test_mod.y, test_mod.subjects, batch=cfg.batch, shuffle=False)
        train_model(model, tl, vl, cfg)
        accs_val.append(evaluate(model, vl, cfg.device))
        accs_test.append(evaluate(model, te, cfg.device))
        last_model, last_loader = model, te
        print(f"  seed={seed}  val={accs_val[-1]:.3f}  test={accs_test[-1]:.3f}")
    feats, ys, subs = extract_features(last_model, last_loader, device)
    diag = cluster_diagnostics(feats, ys, subs)
    plot_embeddings(feats, ys, subs, title=f"EEGNet on contrasts — {label}", out_path=os.path.join(FIG, f"part3_{label}.png"))
    return {"label": label, **_runs_summary(accs_test), "val": _runs_summary(accs_val), **diag}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--quick", action="store_true", help="Tiny epoch budget for smoke-tests")
    args = parser.parse_args()
    epochs = 10 if args.quick else args.epochs

    t0 = time.time()
    print("Loading EEGBCI data (this may download on first run)...")
    train_set = load_data(TRAIN_SUBJECTS)
    test_set = load_data(TEST_SUBJECTS)
    print(f"Loaded train={train_set.X.shape}  test={test_set.X.shape}  in {time.time()-t0:.1f}s")

    # Restrict to the 3-subject failure case (subjects 1, 2, 3) for the
    # "reduced training data" experiment.
    reduced_mask = np.isin(train_set.subjects, [1, 2, 3])
    reduced_set = EpochSet(
        train_set.X[reduced_mask], train_set.y[reduced_mask], train_set.subjects[reduced_mask],
        train_set.ch_names, train_set.pos,
    )

    metrics: Dict[str, dict] = {}
    metrics["baseline_full_8subj"] = run_baseline(train_set, test_set, label="full_8subj", epochs=epochs)
    metrics["baseline_reduced_3subj"] = run_baseline(reduced_set, test_set, label="reduced_3subj", epochs=epochs)
    metrics["part2_full_8subj"] = run_part2(train_set, test_set, label="full_8subj", epochs=epochs)
    metrics["part2_reduced_3subj"] = run_part2(reduced_set, test_set, label="reduced_3subj", epochs=epochs)
    metrics["part3_full_8subj"] = run_part3(train_set, test_set, label="full_8subj", epochs=epochs)
    metrics["part3_reduced_3subj"] = run_part3(reduced_set, test_set, label="reduced_3subj", epochs=epochs)

    out_path = os.path.join(RESULTS, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Human-readable summary. ASCII-only header so Windows console codepages
    # (cp1252, cp437) don't mangle the em-dash when stdout is redirected.
    lines = ["RESULTS - best/worst over seeds (test accuracy, cross-subject)"]
    for k, v in metrics.items():
        lines.append(
            f"  {k:32s}  best={v['best']:.3f}  worst={v['worst']:.3f}  "
            f"sil_class={v['silhouette_class']:.3f}  sil_subj={v['silhouette_subject']:.3f}"
        )
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(RESULTS, "log.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
