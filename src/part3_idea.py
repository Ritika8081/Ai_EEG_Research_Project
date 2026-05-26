"""
Part 3 — Pairwise electrode contrasts instead of raw channels.

The half-page rationale is in `report/part3_idea.md`. Headline idea:
EEGNet's spatial filters learn subject-specific electrode patterns
(Part 1 failure mode). Instead of giving EEGNet 64 raw channels, give
it ten pairwise contrasts between symmetric left-right motor-cortex
electrodes plus one lateral-vs-midline contrast. Anything common to a
left-right pair (overall arousal, electrode contact, average-reference
shift) cancels in the difference; what remains should be closer to the
lateralized motor signal.

The implementation entry point is `compute_hemispheric_contrasts` at
the bottom of this file.

This file also keeps the earlier CMA (Cortical-Manifold Augmentation)
implementation from an earlier exploration step. The scripts in
scripts/ still reference these classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level("ERROR")


def _random_so3(theta_max_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Random rotation matrix with rotation angle <= theta_max_deg (uniform axis)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    theta = np.deg2rad(rng.uniform(0.0, theta_max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def build_cma_bank(pos: np.ndarray, n_aug: int = 32, theta_max_deg: float = 4.0, sigma: float = 0.15, seed: int = 1337) -> torch.Tensor:
    """Returns a tensor of interpolation matrices, shape (n_aug+1, C, C)."""
    rng = np.random.default_rng(seed)
    pos = pos.astype(np.float32)
    norms = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-12
    p_unit = pos / norms  # (C, 3)
    C = p_unit.shape[0]
    mats: List[np.ndarray] = [np.eye(C, dtype=np.float32)]
    for _ in range(n_aug):
        R = _random_so3(theta_max_deg, rng).astype(np.float32)
        p_rot = p_unit @ R.T
        cos_d = p_unit @ p_rot.T
        K = np.exp(-(1.0 - cos_d) / (sigma ** 2))
        K = K / (K.sum(axis=1, keepdims=True) + 1e-8)
        mats.append(K.astype(np.float32))
    return torch.from_numpy(np.stack(mats, axis=0))


class CMAAugmenter:
    """Sample one interpolation matrix per batch and apply it."""

    def __init__(self, bank: torch.Tensor, p_apply: float = 0.8, seed: int = 1337,
                 curriculum: bool = False, p_start: float = 0.9, p_end: float = 0.1,
                 total_steps: int = 1000):
        self.bank = bank
        self.p_apply = p_apply
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)
        self.curriculum = curriculum
        self.p_start = p_start
        self.p_end = p_end
        self.total_steps = total_steps
        self._step = 0

    def to(self, device: str) -> "CMAAugmenter":
        self.bank = self.bank.to(device)
        return self

    def _current_p(self) -> float:
        if not self.curriculum:
            return self.p_apply
        frac = min(1.0, self._step / max(1, self.total_steps))
        return self.p_start + (self.p_end - self.p_start) * frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p = self._current_p()
        self._step += 1
        if torch.rand((), generator=self.gen).item() > p:
            return x
        idx = torch.randint(0, self.bank.shape[0], (1,), generator=self.gen).item()
        A = self.bank[idx]
        return torch.einsum("ij,bjt->bit", A, x)


# ---------------------------------------------------------------------------
# Hemispheric contrasts — the headline Part 3 idea.
# ---------------------------------------------------------------------------

# Left-right symmetric pairs near motor cortex. The PhysioNet 64-channel
# montage uses the standard 10-10 names after eegbci.standardize().
HEMI_PAIRS = [
    ("C3", "C4"),
    ("C1", "C2"),
    ("C5", "C6"),
    ("FC3", "FC4"),
    ("FC1", "FC2"),
    ("FC5", "FC6"),
    ("CP3", "CP4"),
    ("CP1", "CP2"),
    ("CP5", "CP6"),
]
# Lateral hand area vs midline foot area (covers the fists-vs-feet
# interpretation of the task in case the brief's "left vs right fist"
# label is incorrect).
LATERAL_MIDLINE = (("C3", "C4"), "Cz")


def compute_hemispheric_contrasts(X: np.ndarray, ch_names: Sequence[str]) -> np.ndarray:
    """Return only the ten pairwise electrode contrasts.

    Output shape: (n_trials, 10, n_times). Nine left-right differences
    plus one lateral-vs-midline contrast. Trials where the needed
    electrodes are missing produce a zero channel (should not happen
    on the standard EEGBCI 64-channel cap).
    """
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    n_trials, _, n_times = X.shape
    out = np.zeros((n_trials, len(HEMI_PAIRS) + 1, n_times), dtype=np.float32)
    for k, (left, right) in enumerate(HEMI_PAIRS):
        if left in name_to_idx and right in name_to_idx:
            out[:, k] = X[:, name_to_idx[left]] - X[:, name_to_idx[right]]
    (lat_chs, mid_ch) = LATERAL_MIDLINE
    if all(c in name_to_idx for c in lat_chs) and mid_ch in name_to_idx:
        lat_avg = sum(X[:, name_to_idx[c]] for c in lat_chs) / float(len(lat_chs))
        out[:, -1] = (lat_avg - X[:, name_to_idx[mid_ch]]).astype(np.float32)
    return out


def augment_with_contrasts(X: np.ndarray, ch_names: Sequence[str]) -> np.ndarray:
    """Append the ten hemispheric contrast channels to the raw 64 channels.

    Output shape: (n_trials, 74, n_times). Keeping the raw channels
    preserves any non-motor task signal; adding contrasts gives the
    depthwise spatial filter a ready-made lateralization feature it
    would otherwise have to learn from scratch.
    """
    contrasts = compute_hemispheric_contrasts(X, ch_names)
    return np.concatenate([X.astype(np.float32), contrasts], axis=1)
