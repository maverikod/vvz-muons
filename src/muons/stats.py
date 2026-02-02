"""
Chunked first-pass branch statistics (min/max/mean/std/nan_rate, median) — Step 3.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from typing import Any

import numpy as np

import awkward as ak  # type: ignore[import-untyped]

MEDIAN_SAMPLE_SIZE = 200_000


def compute_branch_stats(
    tree: Any,
    branches: list[str],
    chunk: int = 200_000,
    max_events: int = 0,
) -> list[dict[str, Any]]:
    """
    Compute per-branch stats in one chunked pass; median from first 200k events.

    Args:
        tree: TTree or RNTuple (must have .num_entries, .arrays()).
        branches: Branch names (order preserved).
        chunk: Chunk size for iteration.
        max_events: Max events to process (0 = all).

    Returns:
        List of dicts with keys: branch, min, max, mean, std, nan_rate, median, n.
        One dict per branch in same order as branches.
    """
    n_entries = tree.num_entries
    n_total = min(n_entries, max_events) if max_events > 0 else n_entries

    # Accumulators per branch: min, max, n, n_valid, sum_x, sum_x2, nan_count
    acc: dict[str, dict[str, Any]] = {b: _empty_acc() for b in branches}

    start = 0
    while start < n_total:
        stop = min(start + chunk, n_total)
        arr = tree.arrays(branches, entry_start=start, entry_stop=stop)
        for b in branches:
            col = arr[b]
            npy = ak.to_numpy(col).flatten() if col.ndim > 1 else ak.to_numpy(col)
            _update_acc(acc[b], npy)
        start = stop

    median_sample = _median_sample(tree, branches, n_total)
    return _to_rows(branches, acc, median_sample, n_total)


def _empty_acc() -> dict[str, Any]:
    """Return empty accumulator for one branch."""
    return {
        "min": np.nan,
        "max": np.nan,
        "n": 0,
        "n_valid": 0,
        "sum_x": 0.0,
        "sum_x2": 0.0,
        "nan_count": 0,
    }


def _update_acc(acc: dict[str, Any], npy: np.ndarray) -> None:
    """Update accumulator with chunk array (vectorized)."""
    n = len(npy)
    if n == 0:
        return
    valid = np.isfinite(npy)
    n_valid = int(np.sum(valid))
    acc["nan_count"] += n - n_valid
    acc["n"] += n
    if n_valid > 0:
        x = npy[valid]
        chunk_min = float(np.min(x))
        chunk_max = float(np.max(x))
        sum_x = float(np.sum(x))
        sum_x2 = float(np.sum(x * x))
        if acc["n_valid"] == 0:
            acc["min"] = chunk_min
            acc["max"] = chunk_max
            acc["sum_x"] = sum_x
            acc["sum_x2"] = sum_x2
        else:
            acc["min"] = min(acc["min"], chunk_min)
            acc["max"] = max(acc["max"], chunk_max)
            acc["sum_x"] += sum_x
            acc["sum_x2"] += sum_x2
        acc["n_valid"] += n_valid


def _median_sample(tree: Any, branches: list[str], n_total: int) -> dict[str, float]:
    """Compute median per branch from first min(200k, n_total) events."""
    n_sample = min(MEDIAN_SAMPLE_SIZE, n_total)
    if n_sample == 0:
        return {b: np.nan for b in branches}
    arr = tree.arrays(branches, entry_stop=n_sample)
    out: dict[str, float] = {}
    for b in branches:
        col = arr[b]
        npy = ak.to_numpy(col).flatten() if col.ndim > 1 else ak.to_numpy(col)
        out[b] = float(np.nanmedian(npy))
    return out


def _to_rows(
    branches: list[str],
    acc: dict[str, dict[str, Any]],
    median_sample: dict[str, float],
    n_total: int,
) -> list[dict[str, Any]]:
    """Convert accumulators and median to branch_stats rows."""
    rows: list[dict[str, Any]] = []
    for b in branches:
        a = acc[b]
        n = a["n"]
        n_valid = a["n_valid"]
        nan_rate = (a["nan_count"] / n) if n > 0 else 0.0
        if n_valid == 0:
            mean = np.nan
            std = np.nan
        else:
            mean = a["sum_x"] / n_valid
            var_raw = (a["sum_x2"] / n_valid) - (mean * mean)
            var_raw = max(0.0, var_raw)
            if n_valid > 1:
                std = np.sqrt(var_raw * n_valid / (n_valid - 1))  # Sample std
            else:
                std = 0.0
        rows.append(
            {
                "branch": b,
                "min": a["min"],
                "max": a["max"],
                "mean": mean,
                "std": std,
                "nan_rate": nan_rate,
                "median": median_sample[b],
                "n": n_total,
            }
        )
    return rows
