"""
Evaluation diagnostics: t-SNE/UMAP embeddings, subject/class cluster scores.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:
    import umap

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


def _safe_sil(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score robust to single-class slices."""
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def cluster_diagnostics(features: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Dict[str, float]:
    """How much is the embedding organised by SUBJECT vs by CLASS?

    A model that has truly learned class structure should have:
        sil_class >> sil_subject
    If the opposite holds, the network has effectively learned a
    subject-identifier rather than a task-feature.
    """
    return {
        "silhouette_class": _safe_sil(features, y),
        "silhouette_subject": _safe_sil(features, subjects),
    }


def project_2d(features: np.ndarray, method: str = "umap", seed: int = 1337) -> Tuple[np.ndarray, str]:
    if method == "umap" and HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(features), "UMAP"
    # tSNE fallback. perplexity is auto-clamped because the test split is small.
    perp = max(5, min(30, len(features) // 4))
    return TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca").fit_transform(features), "t-SNE"


def plot_embeddings(
    features: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    title: str,
    out_path: str,
    method: str = "umap",
    seed: int = 1337,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    emb, name = project_2d(features, method=method, seed=seed)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    # Left: coloured by class.
    for cls in np.unique(y):
        m = y == cls
        ax[0].scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.7, label=f"class {cls}")
    ax[0].set_title(f"{title} — by class")
    ax[0].legend(fontsize=8)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # Right: coloured by subject.
    for s in np.unique(subjects):
        m = subjects == s
        ax[1].scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.7, label=f"S{int(s)}")
    ax[1].set_title(f"{title} — by subject")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.suptitle(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
