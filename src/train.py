"""
Training + evaluation utilities shared across all parts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainCfg:
    epochs: int = 80
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    label_smoothing: float = 0.0
    max_norm_depthwise: float = 1.0
    max_norm_fc: float = 0.25
    # Optional auxiliary loss hook (used by Part 2 adversarial head, ignored otherwise).
    aux_loss_fn: Optional[Callable[[torch.Tensor, Dict], torch.Tensor]] = None
    # Optional augmentation callable applied on the device-tensor batch.
    augment_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


@dataclass
class RunHistory:
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)


def _project_max_norm(weight: torch.Tensor, max_val: float, dim) -> None:
    """Renormalise rows of `weight` so per-filter L2 norm <= max_val.
    `dim` is either an int (axis to norm over) or a tuple — for tuples we
    flatten and compute vector norm, since torch.norm(dim=tuple) is
    matrix_norm only."""
    with torch.no_grad():
        if isinstance(dim, (tuple, list)):
            # Norm per slice along dim 0. Flatten the trailing dims.
            w = weight.reshape(weight.shape[0], -1)
            norm = w.norm(dim=1, keepdim=True).clamp_min(1e-8)
            scale = norm.clamp_max(max_val) / norm
            w.mul_(scale)
        else:
            norm = weight.norm(dim=dim, keepdim=True).clamp_min(1e-8)
            weight.mul_(norm.clamp_max(max_val) / norm)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    subjects: Optional[np.ndarray],
    batch: int,
    shuffle: bool,
) -> DataLoader:
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    if subjects is not None:
        st = torch.from_numpy(subjects).long()
        ds = TensorDataset(Xt, yt, st)
    else:
        ds = TensorDataset(Xt, yt)
    # num_workers=0 on Windows + small data is fastest; multiprocess overhead
    # dwarfs the per-epoch cost here.
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, drop_last=False)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    cfg: TrainCfg,
    extra_state: Optional[Dict] = None,
) -> RunHistory:
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    hist = RunHistory()
    extra_state = extra_state or {}

    for epoch in range(cfg.epochs):
        model.train()
        n, tot, correct, loss_sum = 0, 0, 0, 0.0
        for batch in train_loader:
            if len(batch) == 3:
                x, y, s = batch
            else:
                x, y = batch
                s = None
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            if s is not None:
                s = s.to(cfg.device, non_blocking=True)

            if cfg.augment_fn is not None:
                x = cfg.augment_fn(x)

            logits = model(x)
            loss = ce(logits, y)
            if cfg.aux_loss_fn is not None:
                aux = cfg.aux_loss_fn(
                    model.last_features if hasattr(model, "last_features") else None,
                    {"subjects": s, "epoch": epoch, **extra_state},
                )
                if aux is not None:
                    loss = loss + aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # max_norm projections — EEGNet relies on these for regularisation.
            if hasattr(model, "depthwise"):
                _project_max_norm(model.depthwise.weight, cfg.max_norm_depthwise, dim=(1, 2, 3))
            if hasattr(model, "fc"):
                _project_max_norm(model.fc.weight, cfg.max_norm_fc, dim=1)

            tot += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
        hist.train_acc.append(correct / max(n, 1))
        hist.train_loss.append(loss_sum / max(n, 1))
        if val_loader is not None:
            va = evaluate(model, val_loader, cfg.device)
            hist.val_acc.append(va)
    return hist


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    n, c = 0, 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        c += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return c / max(n, 1)


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    feats, ys, subs = [], [], []
    for batch in loader:
        if len(batch) == 3:
            x, y, s = batch
            subs.append(s.numpy())
        else:
            x, y = batch
        x = x.to(device)
        h = model.features(x).cpu().numpy()
        feats.append(h)
        ys.append(y.numpy())
    feats = np.concatenate(feats, 0)
    ys = np.concatenate(ys, 0)
    subs = np.concatenate(subs, 0) if subs else np.zeros(len(ys), dtype=np.int64)
    return feats, ys, subs
