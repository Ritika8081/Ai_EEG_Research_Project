"""
EEGNet-8,2 — Lawhern et al., 2018, faithful PyTorch port.

Notes on faithfulness:
- F1=8, D=2, F2=F1*D=16.
- Temporal kernel length is set to sfreq/2 (=80 at 160 Hz) per the paper.
- DepthwiseConv2D over electrodes uses max_norm=1.0 — enforced via projection
  after each optim step (see train.py).
- Final Dense uses max_norm=0.25 (also projected in train.py).
- ELU activations, AvgPool, Dropout 0.5. Padding="same" on temporal conv.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 321,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        kern_len: int = 80,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        F2 = F1 * D
        # Block 1: temporal conv then depthwise spatial conv.
        self.temporal = nn.Conv2d(
            1, F1, kernel_size=(1, kern_len), padding=(0, kern_len // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        # Depthwise over electrodes — collapses the channel dimension per
        # temporal-filter group. This is the layer Part 1 argues is the
        # principal failure surface for cross-subject transfer.
        self.depthwise = nn.Conv2d(
            F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv = depthwise + pointwise.
        self.sep_depth = nn.Conv2d(
            F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False
        )
        self.sep_point = nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier — input dim depends on n_times.
        t_after = n_times
        t_after = (t_after + (kern_len % 2 == 0)) // 1  # temporal "same"
        t_after = t_after // 4  # pool1
        t_after = t_after // 8  # pool2
        self.flat_dim = F2 * max(1, t_after)
        self.fc = nn.Linear(self.flat_dim, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.bn1(x)
        x = self.depthwise(x)
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
        # If we computed flat_dim wrong (e.g. odd kernel padding), lazily fix
        # the head — keeps the constructor robust to small length changes.
        if h.shape[1] != self.flat_dim:
            self.flat_dim = h.shape[1]
            self.fc = nn.Linear(self.flat_dim, self.fc.out_features).to(h.device)
        return self.fc(h)
