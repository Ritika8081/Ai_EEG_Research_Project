"""Quick sanity check — runs all three models for 2 epochs on subjects 1-2 only."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.data import load_data, per_trial_zscore
from src.eegnet import EEGNet
from src.part2_model import TopoNet
from src.part3_idea import build_cma_bank, CMAAugmenter
from src.seed import seed_all
from src.train import TrainCfg, make_loader, train_model, evaluate

seed_all(1337)
print("Loading subjects 1-3...")
t0 = time.time()
ep = load_data([1, 2, 3])
print(f"  loaded in {time.time()-t0:.1f}s, X={ep.X.shape}")

X = per_trial_zscore(ep.X)
tl = make_loader(X[:60], ep.y[:60], ep.subjects[:60], batch=32, shuffle=True)
vl = make_loader(X[60:], ep.y[60:], ep.subjects[60:], batch=32, shuffle=False)

cfg = TrainCfg(epochs=2)

print("\nEEGNet baseline...")
m = EEGNet(n_channels=X.shape[1], n_times=X.shape[2])
train_model(m, tl, vl, cfg)
print(f"  val_acc={evaluate(m, vl, cfg.device):.3f}")

print("\nTopoNet (Part 2)...")
m = TopoNet(pos=ep.pos, n_channels=X.shape[1], n_times=X.shape[2])
train_model(m, tl, vl, cfg)
print(f"  val_acc={evaluate(m, vl, cfg.device):.3f}")

print("\nEEGNet + CMA (Part 3)...")
bank = build_cma_bank(ep.pos, n_aug=8, theta_max_deg=4.0)
aug = CMAAugmenter(bank, p_apply=0.8).to(cfg.device)
cfg2 = TrainCfg(epochs=2, augment_fn=aug)
m = EEGNet(n_channels=X.shape[1], n_times=X.shape[2])
train_model(m, tl, vl, cfg2)
print(f"  val_acc={evaluate(m, vl, cfg2.device):.3f}")

print("\nSmoke test OK")
