"""
Part 2 — TopoNet: my proposed improvement.

Root cause being addressed
--------------------------
EEGNet's depthwise spatial conv has one scalar weight per electrode-index
per filter (shape: (F1*D, C)). The semantics of "electrode index c" depends
entirely on which physical position c maps to. Across subjects, the
mapping is approximately stable (10-05 montage), but the *covariance
structure* at each electrode varies — cortical-source-to-scalp projection
geometry differs per subject. So a per-index weight is forced to encode a
training-set-mean spatial profile, which is biased toward dominant
subjects.

Two changes, both targeting subject-specific spatial overfit at its
mechanistic source:

1. Per-trial covariance whitening (Riemannian recentering to identity).
   For each trial X (C × T):  X' = (X Xᵀ / T + εI)^(-1/2) X
   After whitening, the spatial covariance of every trial is approximately
   the identity. The depthwise filter cannot exploit subject-specific
   covariance structure because it has been algebraically removed.
   This is closely related to Euclidean Alignment but applied per-trial
   (not per-subject mean) so it requires no calibration data and
   no test-subject statistics.

2. Coordinate-Conditioned Spatial Filters (CCSF).
   Replace the per-electrode weight matrix with a small MLP
       w_k(c) = MLP(pos_c, e_k)
   where pos_c ∈ R^3 is the 3D scalp position of electrode c and e_k is a
   learned filter embedding for the k-th spatial filter. The filter is now
   a smooth function on the scalp manifold. Two implications:
   (a) Filters are differentiable in electrode position → small inter-subject
       montage shifts do not break them.
   (b) The number of parameters is independent of channel count, which
       removes a degree of freedom that previously soaked up subject noise.

The rest of the architecture is identical to EEGNet so the comparison is
clean.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _whiten_per_trial(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """X: (B, C, T) -> whitened so that X X^T / T ~ I per trial."""
    B, C, T = x.shape
    cov = torch.bmm(x, x.transpose(1, 2)) / T  # (B, C, C)
    cov = cov + eps * torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
    # Symmetric eigendecomp. eigh is stable for SPD matrices and differentiable.
    # We use it without_grad: whitening is intended as preprocessing, not as
    # a learned operator, so gradients flowing through eigh would only add
    # noise and instability.
    with torch.no_grad():
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp_min(1e-6)
        inv_sqrt = evecs @ torch.diag_embed(evals.rsqrt()) @ evecs.transpose(1, 2)
    return torch.bmm(inv_sqrt, x)


class CoordinateSpatialFilter(nn.Module):
    """w_k(c) = MLP([pos_c, e_k]) — coordinate-conditioned spatial filter bank."""

    def __init__(self, n_filters: int, pos: np.ndarray, hidden: int = 32, emb_dim: int = 8):
        super().__init__()
        self.n_filters = n_filters
        # Register positions as a buffer (non-trainable), normalised to unit sphere.
        pos = pos.astype(np.float32)
        radii = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-8
        pos = pos / radii  # all electrodes sit on ~unit sphere
        self.register_buffer("pos", torch.from_numpy(pos))  # (C, 3)
        self.filter_emb = nn.Parameter(torch.randn(n_filters, emb_dim) * 0.1)
        self.mlp = nn.Sequential(
            nn.Linear(3 + emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def compute_W(self) -> torch.Tensor:
        # Returns (C, F) — one weight per (electrode, filter) pair.
        C = self.pos.shape[0]
        F = self.n_filters
        pos_e = self.pos.unsqueeze(1).expand(C, F, 3)
        emb_e = self.filter_emb.unsqueeze(0).expand(C, F, self.filter_emb.shape[1])
        inp = torch.cat([pos_e, emb_e], dim=-1).reshape(C * F, -1)
        w = self.mlp(inp).reshape(C, F)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F_in, C, T) where the depthwise grouping expects F_out = F_in*D.
        # We multiplex F_in into F_out via filter_emb, so callers should set
        # n_filters = F_in * D, and we apply per filter k that maps to its
        # temporal group g(k) = k // D.
        W = self.compute_W()  # (C, F_out)
        B, Fin, C, T = x.shape
        D = self.n_filters // Fin
        # For each output filter k -> input group g = k // D.
        # Expand x along the F_out dim by repeating group blocks.
        x_rep = x.repeat_interleave(D, dim=1)  # (B, F_out, C, T)
        # Now mix electrodes: (B, F_out, C, T) -> (B, F_out, 1, T) via W
        # einsum: 'b k c t, c k -> b k t'
        y = torch.einsum("bkct,ck->bkt", x_rep, W).unsqueeze(2)  # (B, F_out, 1, T)
        return y


class TopoNet(nn.Module):
    def __init__(
        self,
        pos: np.ndarray,
        n_channels: int = 64,
        n_times: int = 321,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        kern_len: int = 80,
        dropout: float = 0.5,
        whiten_eps: float = 1e-3,
        use_whitening: bool = True,
    ):
        super().__init__()
        F2 = F1 * D
        self.use_whitening = use_whitening
        self.whiten_eps = whiten_eps
        self.temporal = nn.Conv2d(
            1, F1, kernel_size=(1, kern_len), padding=(0, kern_len // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = CoordinateSpatialFilter(n_filters=F2, pos=pos)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.sep_depth = nn.Conv2d(
            F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False
        )
        self.sep_point = nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        t_after = n_times // 4 // 8
        self.flat_dim = F2 * max(1, t_after)
        self.fc = nn.Linear(self.flat_dim, n_classes)
        self.last_features: Optional[torch.Tensor] = None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        if self.use_whitening:
            x = _whiten_per_trial(x, eps=self.whiten_eps)
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.temporal(x)
        x = self.bn1(x)
        x = self.spatial(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        if h.shape[1] != self.flat_dim:
            self.flat_dim = h.shape[1]
            self.fc = nn.Linear(self.flat_dim, self.fc.out_features).to(h.device)
        self.last_features = h
        return self.fc(h)
