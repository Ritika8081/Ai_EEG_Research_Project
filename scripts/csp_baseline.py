"""
Classical CSP + LDA reference baseline.

Common Spatial Patterns (CSP) is THE classical BCI baseline for motor
imagery. It explicitly extracts subject-specific spatial filters that
maximise variance ratio between classes — i.e. it is *built* to exploit
the same covariance structure EEGNet learns implicitly. So:

- If CSP+LDA badly underperforms EEGNet, the deep model is learning
  something beyond CSP — useful information for the defence.
- If CSP+LDA matches or beats EEGNet on this subset, the deep model
  isn't actually using its capacity — also useful information.

Either outcome strengthens the failure-analysis narrative.

Usage: python scripts/csp_baseline.py
"""
import json
import os
import sys
import warnings

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import TEST_SUBJECTS, TRAIN_SUBJECTS, EpochSet, load_data

warnings.filterwarnings("ignore")

SEEDS = [1337, 2024, 7]


def make_pipeline(n_components: int = 6):
    csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return Pipeline([("csp", csp), ("lda", lda)])


def run_eval(train_set: EpochSet, test_set: EpochSet, label: str):
    accs = []
    for s in SEEDS:
        # CSP is deterministic given inputs; seed only affects ties.
        np.random.seed(s)
        clf = make_pipeline()
        clf.fit(train_set.X.astype(np.float64), train_set.y)
        acc = clf.score(test_set.X.astype(np.float64), test_set.y)
        accs.append(float(acc))
        print(f"  [{label}] seed={s}  test={acc:.3f}")
    arr = np.asarray(accs)
    return {"best": float(arr.max()), "worst": float(arr.min()), "mean": float(arr.mean())}


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
    print("CSP+LDA, full 8-subj pool, cross-subject test:")
    out["csp_full_8subj"] = run_eval(train_set, test_set, "csp_full_8subj")
    print("CSP+LDA, reduced 3-subj pool, cross-subject test:")
    out["csp_reduced_3subj"] = run_eval(reduced_set, test_set, "csp_reduced_3subj")

    with open("results/csp_baseline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSummary:")
    for k, v in out.items():
        print(f"  {k:25s}  best={v['best']:.3f}  worst={v['worst']:.3f}  mean={v['mean']:.3f}")


if __name__ == "__main__":
    main()
