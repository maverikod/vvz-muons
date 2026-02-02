"""
Numerical metrics and spectrum files from L, W, eigenvalues, eigenvectors — Step 7.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

LAMBDA_THRESHOLD = 1e-12


def compute_metrics(
    L: np.ndarray,
    W: np.ndarray,
    eigenvalues: np.ndarray,
    eigvec_first10: np.ndarray,
    N_events: int,
    features_count: int,
    d: int,
    mode: str,
    bins: int,
) -> dict[str, Any]:
    """
    Compute numerical metrics from L, W, eigenvalues, and first eigenvectors.

    Args:
        L: Laplacian (d, d).
        W: Connectivity matrix (d, d).
        eigenvalues: 1D, sorted ascending.
        eigvec_first10: (d, n_vecs) first eigenvectors.
        N_events: Number of events (rows of O).
        features_count: Number of branches (features).
        d: Observable dimension (columns of O).
        mode: "quantile" or "zscore".
        bins: Number of bins (quantile mode).

    Returns:
        Dict with N_events, features_count, mode, bins, d, density_W, trace_L,
        lambda_min_nonzero, Neff, PR_k (list of up to 10 floats).
    """
    density_W = float(np.count_nonzero(W)) / (d * d) if d > 0 else 0.0
    trace_L = float(np.trace(L))

    lambda_use = eigenvalues[eigenvalues > LAMBDA_THRESHOLD]
    if len(lambda_use) == 0:
        lambda_min_nonzero = float("nan")
        Neff = float("nan")
    else:
        lambda_min_nonzero = float(np.min(lambda_use))
        s1 = np.sum(lambda_use)
        s2 = np.sum(lambda_use**2)
        Neff = float(s1 * s1 / s2) if s2 > 0 else float("nan")

    n_vecs = eigvec_first10.shape[1]
    if n_vecs == 0:
        pr_list: list[float] = []
    else:
        v2 = eigvec_first10**2
        v4 = eigvec_first10**4
        sum_v2 = np.sum(v2, axis=0)
        sum_v4 = np.sum(v4, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.where(sum_v4 > 0, (sum_v2**2) / sum_v4, np.nan)
        pr_list = [float(pr[k]) for k in range(min(10, n_vecs))]

    return {
        "N_events": N_events,
        "features_count": features_count,
        "mode": mode,
        "bins": bins,
        "d": d,
        "density_W": density_W,
        "trace_L": trace_L,
        "lambda_min_nonzero": lambda_min_nonzero,
        "Neff": Neff,
        "PR_k": pr_list,
    }


def _nan_to_none(obj: Any) -> Any:
    """Replace float nan with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def write_metrics_json(metrics: dict[str, Any], path: Path) -> None:
    """Write metrics dict to metrics.json (numerical only). NaN -> null."""
    with open(path, "w") as f:
        json.dump(_nan_to_none(metrics), f, indent=2)


def write_spectrum_csv(
    eigenvalues: np.ndarray,
    eigvec_first10: np.ndarray,
    path: Path,
) -> None:
    """
    Write spectrum.csv with columns k, lambda_k, PR_k.

    Rows: one per mode (0 to len(eigenvalues)-1). PR_k only for first
    min(10, n_eigvecs) modes; rest empty.
    """
    n_eig = len(eigenvalues)
    n_vecs = eigvec_first10.shape[1]

    if n_vecs > 0:
        v2 = eigvec_first10**2
        v4 = eigvec_first10**4
        sum_v2 = np.sum(v2, axis=0)
        sum_v4 = np.sum(v4, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr_all = np.where(sum_v4 > 0, (sum_v2**2) / sum_v4, np.nan)
    else:
        pr_all = np.array([])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "lambda_k", "PR_k"])
        for k in range(n_eig):
            lambda_k = float(eigenvalues[k])
            if k < len(pr_all):
                pr_k = pr_all[k]
                pr_str = "" if np.isnan(pr_k) else f"{float(pr_k)}"
            else:
                pr_str = ""
            writer.writerow([k, lambda_k, pr_str])
