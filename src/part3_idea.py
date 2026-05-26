"""
Part 3 — Cortical-Manifold Augmentation (CMA): a novel, underexplored idea.

The half-page rationale lives in `report/part3_idea.md`. Code-side: this
file implements the minimal proof-of-concept.

The idea in one line
--------------------
Augment training data by simulating small random electrode-montage
rotations and interpolating the EEG back to the canonical channel set via
a spherical RBF — i.e. data augmentation in *electrode-position space*,
not signal space.

Why this is different from known augmentations
----------------------------------------------
- Channel dropout: drops electrodes. Discrete. Doesn't model position
  uncertainty.
- Time-shift / SpecAugment: temporal axis, ignores subject-to-subject
  spatial variability.
- Euclidean / Riemannian alignment: deterministic transform of test data
  using calibration; not an augmentation.
- Mixup across subjects: blends signals, but the blended trials are
  *not physically plausible* — the resulting "subject" has no consistent
  head geometry.

CMA is the only one of these that produces trials which look like the
*same* recording done with a slightly different cap placement. Inter-rater
cap-placement variability on a 10-20 system is reported at 5-10 mm
(roughly 2-4°). The augmentation pushes the model toward filters that
are invariant under exactly the kind of small rotation that occurs in
practice.

PoC details
-----------
1. Positions are first projected onto the unit sphere.
2. We precompute a bank of random rotations R_k ~ SO(3) with angle ~ U(0, θ_max).
3. For each R_k, build interpolation matrix A_k ∈ R^(C, C) using a
   Gaussian RBF over geodesic distance:
       A_k[i, j] = K(R_k x_j, x_i) / Σ_j K(R_k x_j, x_i)
   where K(a, b) = exp(-(1 - aᵀb) / σ²).
4. At training time, for each batch, sample k, apply A_k.

This is a linear, channel-only, batch-independent operator → cheap.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch


def _random_so3(theta_max_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Random rotation matrix with rotation angle <= theta_max_deg (uniform axis)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    theta = np.deg2rad(rng.uniform(0.0, theta_max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def build_cma_bank(pos: np.ndarray, n_aug: int = 32, theta_max_deg: float = 4.0, sigma: float = 0.15, seed: int = 1337) -> torch.Tensor:
    """Returns a tensor of interpolation matrices, shape (n_aug+1, C, C).

    Index 0 is the identity (i.e. no augmentation), so a uniform sample
    from this bank includes the "no-op" with probability 1/(n_aug+1).
    """
    rng = np.random.default_rng(seed)
    pos = pos.astype(np.float32)
    norms = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-12
    p_unit = pos / norms  # (C, 3)
    C = p_unit.shape[0]

    mats: List[np.ndarray] = [np.eye(C, dtype=np.float32)]
    for _ in range(n_aug):
        R = _random_so3(theta_max_deg, rng).astype(np.float32)
        p_rot = p_unit @ R.T  # rotated source positions
        # K[i, j] = K(p_rot[j], p_unit[i]) — value at canonical i interpolated
        # from rotated source j.
        cos_d = p_unit @ p_rot.T  # (C, C)
        K = np.exp(-(1.0 - cos_d) / (sigma ** 2))
        # Row-normalise so the interpolated signal preserves DC scale.
        K = K / (K.sum(axis=1, keepdims=True) + 1e-8)
        mats.append(K.astype(np.float32))
    return torch.from_numpy(np.stack(mats, axis=0))  # (n_aug+1, C, C)


class CMAAugmenter:
    """Callable augmenter: samples one interpolation matrix per batch and applies it.

    Supports an optional curriculum: p_apply linearly decays from p_start
    to p_end across `total_steps` calls. This is the empirically-motivated
    fix to the "60-epoch overfit" failure mode documented in
    `report/part3_idea.md`. Default behaviour is constant p_apply
    (curriculum disabled).
    """

    def __init__(
        self,
        bank: torch.Tensor,
        p_apply: float = 0.8,
        seed: int = 1337,
        curriculum: bool = False,
        p_start: float = 0.9,
        p_end: float = 0.1,
        total_steps: int = 1000,
    ):
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
        # Linear decay clipped to [p_end, p_start].
        frac = min(1.0, self._step / max(1, self.total_steps))
        return self.p_start + (self.p_end - self.p_start) * frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T). Sample ONE matrix per batch — keeps augmentation
        # within-batch homogeneous, which avoids fighting BatchNorm statistics.
        p = self._current_p()
        self._step += 1
        if torch.rand((), generator=self.gen).item() > p:
            return x
        idx = torch.randint(0, self.bank.shape[0], (1,), generator=self.gen).item()
        A = self.bank[idx]  # (C, C)
        # einsum: 'ij,bjt->bit'
        return torch.einsum("ij,bjt->bit", A, x)
