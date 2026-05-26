"""
PhysioNet EEGBCI loader for motor-imagery (runs 6, 10, 14: imagined both-fists vs both-feet).

Design notes
------------
- We deliberately do NOT use any subject-wise normalisation that depends on
  test-set statistics. Cross-subject evaluation in Part 1 must reflect what
  a real deployment would see: subject 9/10 statistics are *not* known at
  training time.
- Bandpass 4-38 Hz keeps mu/beta motor rhythms and drops slow drift + line
  noise. Sample rate is 160 Hz; we keep it.
- Epoch window 0.5-2.5 s after cue. 0-0.5 s is mostly motor-prep, weak
  signal; 2.5 s onwards drifts into the next cue in this paradigm.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# Frozen constants — also documented in README.
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 10]
RUNS = [6, 10, 14]
SFREQ = 160.0
TMIN, TMAX = 0.5, 2.5  # seconds, relative to cue onset
L_FREQ, H_FREQ = 4.0, 38.0
# PhysioNet runs 6, 10, 14 are Task 4 (imagined motor imagery):
#   T1 = both fists, T2 = both feet
EVENT_ID = {"T1": 0, "T2": 1}


@dataclass
class EpochSet:
    X: np.ndarray  # (n_trials, n_channels, n_times) float32
    y: np.ndarray  # (n_trials,) int64
    subjects: np.ndarray  # (n_trials,) int64
    ch_names: List[str]
    pos: np.ndarray  # (n_channels, 3) float32 — used by Part 2/3


def _load_raw_one_subject(subj: int):
    # update_path=True silently configures MNE's cache dir; otherwise MNE
    # blocks on an interactive prompt the first time it runs.
    fnames = eegbci.load_data(subj, RUNS, update_path=True, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in fnames]
    raw = concatenate_raws(raws)
    # MNE's eegbci channel names have trailing dots ("Fc5..") — normalise.
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    return raw


def _epoch(raw) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose=False)
    # Map MNE event codes back to {0,1}
    events[:, 2] = np.where(events[:, 2] == 2, 0, 1)
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"fists": 0, "feet": 1},
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )
    X = epochs.get_data(copy=False).astype(np.float32)  # (n, c, t)
    y = epochs.events[:, 2].astype(np.int64)
    pos = np.array(
        [epochs.info["chs"][i]["loc"][:3] for i in range(len(epochs.ch_names))],
        dtype=np.float32,
    )
    return X, y, epochs.ch_names, pos


def load_data(subjects: Sequence[int], cache_dir: str = "results/cache") -> EpochSet:
    """Load epochs for a list of subjects. Caches per-subject arrays to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    Xs, ys, subs, ref_names, ref_pos = [], [], [], None, None
    for s in subjects:
        cache = os.path.join(cache_dir, f"S{s:03d}.npz")
        if os.path.exists(cache):
            data = np.load(cache, allow_pickle=True)
            X, y, ch_names, pos = data["X"], data["y"], list(data["ch_names"]), data["pos"]
        else:
            raw = _load_raw_one_subject(s)
            X, y, ch_names, pos = _epoch(raw)
            np.savez_compressed(cache, X=X, y=y, ch_names=np.array(ch_names), pos=pos)
        if ref_names is None:
            ref_names, ref_pos = ch_names, pos
        else:
            # Sanity: order should match (all subjects use the same montage).
            assert ch_names == ref_names, f"Channel mismatch for subject {s}"
        Xs.append(X)
        ys.append(y)
        subs.append(np.full(len(y), s, dtype=np.int64))
    return EpochSet(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0),
        subjects=np.concatenate(subs, axis=0),
        ch_names=ref_names,
        pos=ref_pos,
    )


def per_trial_zscore(X: np.ndarray) -> np.ndarray:
    """Per-trial, per-channel z-score. Removes trial-level DC + scale, which
    are dominated by impedance/skin variation, not neural content."""
    m = X.mean(axis=-1, keepdims=True)
    s = X.std(axis=-1, keepdims=True) + 1e-6
    return (X - m) / s
