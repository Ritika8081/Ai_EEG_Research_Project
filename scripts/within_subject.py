"""
Optional within-subject reference baseline.

The main assignment's cross-subject setup is the headline. This script
runs leave-one-trial-out-style within-subject training as a *reference
point* so the cross-subject gap can be quantified — useful for the
defence slide.

Each subject: 80/20 random split of their own trials, EEGNet trained from
scratch. Reported as mean ± std over subjects.

Usage:  python scripts/within_subject.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import TRAIN_SUBJECTS, TEST_SUBJECTS, load_data, per_trial_zscore
from src.eegnet import EEGNet
from src.seed import seed_all
from src.train import TrainCfg, evaluate, make_loader, train_model

SUBJECTS = TRAIN_SUBJECTS + TEST_SUBJECTS


def main(seed: int = 1337, epochs: int = 60) -> None:
    accs = []
    for s in SUBJECTS:
        seed_all(seed)
        ep = load_data([s])
        X = per_trial_zscore(ep.X)
        n = len(ep.y)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        cut = int(0.8 * n)
        tr, te = idx[:cut], idx[cut:]
        cfg = TrainCfg(epochs=epochs)
        m = EEGNet(n_channels=X.shape[1], n_times=X.shape[2])
        tl = make_loader(X[tr], ep.y[tr], None, batch=cfg.batch, shuffle=True)
        tel = make_loader(X[te], ep.y[te], None, batch=cfg.batch, shuffle=False)
        train_model(m, tl, None, cfg)
        a = evaluate(m, tel, cfg.device)
        accs.append(a)
        print(f"  S{s:03d}  test_within={a:.3f}")
    arr = np.asarray(accs)
    print(f"\nWithin-subject EEGNet: mean={arr.mean():.3f}  std={arr.std():.3f}  best={arr.max():.3f}  worst={arr.min():.3f}")
    print("Compare against cross-subject values in results/metrics.json — the gap *is* the failure.")


if __name__ == "__main__":
    main()
