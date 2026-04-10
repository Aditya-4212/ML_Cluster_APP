# ============================================================
# utils/metrics.py
# Clustering evaluation metrics — safe, noise-aware wrappers
# ============================================================

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def _valid_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask excluding DBSCAN noise points (label == -1)."""
    return np.array(labels) != -1


def compute_all_metrics(X: np.ndarray, labels) -> dict:
    """
    Returns a dict with all available clustering metrics.
    Handles DBSCAN noise points (label = -1) gracefully.
    """
    m: dict = {}
    arr = np.array(labels)
    valid = _valid_mask(arr)
    X_valid   = X[valid]
    lbl_valid = arr[valid]

    m["Clusters"]  = int(len(set(arr)) - (1 if -1 in arr else 0))
    m["Noise pts"] = int((arr == -1).sum())

    if valid.sum() >= 2 and len(set(lbl_valid)) >= 2:
        try:
            m["Silhouette ↑"] = round(silhouette_score(X_valid, lbl_valid), 4)
        except Exception:
            pass
        try:
            m["Davies-Bouldin ↓"] = round(davies_bouldin_score(X_valid, lbl_valid), 4)
        except Exception:
            pass
        try:
            m["Calinski-Harabasz ↑"] = round(calinski_harabasz_score(X_valid, lbl_valid), 1)
        except Exception:
            pass

    return m


def safe_silhouette(X: np.ndarray, labels) -> float | None:
    """Returns silhouette score or None if not computable."""
    try:
        arr   = np.array(labels)
        valid = _valid_mask(arr)
        if valid.sum() < 2 or len(set(arr[valid])) < 2:
            return None
        return round(silhouette_score(X[valid], arr[valid]), 4)
    except Exception:
        return None


def safe_davies_bouldin(X: np.ndarray, labels) -> float | None:
    """Returns Davies-Bouldin score or None if not computable."""
    try:
        arr   = np.array(labels)
        valid = _valid_mask(arr)
        if valid.sum() < 2 or len(set(arr[valid])) < 2:
            return None
        return round(davies_bouldin_score(X[valid], arr[valid]), 4)
    except Exception:
        return None
