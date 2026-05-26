"""Class-conditional plots of the two key contrast channels.

Sanity check that the engineered feature has class-discriminative
structure before we ask the network to use it. Two panels:

- C3 - C4 averaged over trials, one curve per class
- (C3+C4)/2 - Cz averaged over trials, one curve per class

Saves to results/figures/part3_contrasts.png.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from src.data import TRAIN_SUBJECTS, load_data
from src.part3_idea import HEMI_PAIRS, LATERAL_MIDLINE

OUT = "results/figures/part3_contrasts.png"


def main() -> None:
    ep = load_data(TRAIN_SUBJECTS)
    name_to_idx = {n: i for i, n in enumerate(ep.ch_names)}
    sfreq = 160.0
    t_axis = np.arange(ep.X.shape[-1]) / sfreq + 0.5  # tmin = 0.5s post-cue

    c3 = ep.X[:, name_to_idx["C3"]]
    c4 = ep.X[:, name_to_idx["C4"]]
    cz = ep.X[:, name_to_idx["Cz"]]

    diff_lr = c3 - c4
    diff_lm = 0.5 * (c3 + c4) - cz

    mask0 = ep.y == 0
    mask1 = ep.y == 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, sig, title in [
        (axes[0], diff_lr, "C3 − C4 (left vs right hemisphere)"),
        (axes[1], diff_lm, "(C3 + C4)/2 − Cz (lateral vs midline)"),
    ]:
        m0 = sig[mask0].mean(axis=0)
        m1 = sig[mask1].mean(axis=0)
        s0 = sig[mask0].std(axis=0) / np.sqrt(mask0.sum())
        s1 = sig[mask1].std(axis=0) / np.sqrt(mask1.sum())
        ax.plot(t_axis, m0, color="C0", label="class T1")
        ax.fill_between(t_axis, m0 - s0, m0 + s0, color="C0", alpha=0.2)
        ax.plot(t_axis, m1, color="C3", label="class T2")
        ax.fill_between(t_axis, m1 - s1, m1 + s1, color="C3", alpha=0.2)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("time (s, post-cue)")
        ax.set_ylabel("contrast (µV)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"Class-conditional contrast channels (train pool, N={mask0.sum()} T1, {mask1.sum()} T2)",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Saved {OUT}")

    # Quick numeric summary so we can quote it in the report.
    for name, sig in [("C3-C4", diff_lr), ("(C3+C4)/2 - Cz", diff_lm)]:
        m0 = sig[mask0].mean()
        m1 = sig[mask1].mean()
        # Approximate Cohen's d on the trial means.
        pooled = np.sqrt(0.5 * (sig[mask0].mean(axis=-1).std() ** 2 + sig[mask1].mean(axis=-1).std() ** 2))
        d = (m1 - m0) / (pooled + 1e-12)
        print(f"  {name:18s}  mean T1={m0:+.3e}  T2={m1:+.3e}  Cohen-d~={d:+.3f}")


if __name__ == "__main__":
    main()
