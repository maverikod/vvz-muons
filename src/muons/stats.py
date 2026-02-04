"""
Chunked first-pass branch statistics (min/max/mean/std/nan_rate, median) — Step 3.
Supports extended features (scalar + jagged aggregates) per addontspc.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

import awkward as ak  # type: ignore[import-untyped]

from muons.jagged_aggs import build_chunk_matrix

MEDIAN_SAMPLE_SIZE = 200_000


def compute_branch_stats(
    tree: Any,
    branches: list[str],
    chunk: int = 200_000,
    max_events: int = 0,
    scalar_branches: list[str] | None = None,
    jagged_specs: list[tuple[str, list[str]]] | None = None,
) -> list[dict[str, Any]]:
    """
    Compute per-feature stats in one chunked pass; median from first 200k events.

    When scalar_branches and jagged_specs are None, branches is the list of
    scalar branch names (legacy). When provided, branches is the full
    feature_names list (scalar + derived) and chunks are built via
    build_chunk_matrix (scalar + jagged aggregates).

    Returns:
        List of dicts with keys: branch, min, max, mean, std, nan_rate, median, n.
        One dict per feature in same order as branches.
    """
    n_entries = tree.num_entries
    n_total = min(n_entries, max_events) if max_events > 0 else n_entries

    acc: dict[str, dict[str, Any]] = {b: _empty_acc() for b in branches}

    if jagged_specs:
        _compute_stats_extended(
            tree, branches, scalar_branches or [], jagged_specs, chunk, n_total, acc
        )
    else:
        start = 0
        last_pct = -1
        while start < n_total:
            stop = min(start + chunk, n_total)
            arr = tree.arrays(branches, entry_start=start, entry_stop=stop)
            for b in branches:
                col = arr[b]
                npy = ak.to_numpy(col).flatten() if col.ndim > 1 else ak.to_numpy(col)
                _update_acc(acc[b], npy)
            pct = int(100 * stop / n_total) if n_total > 0 else 100
            if pct >= last_pct + 5 or stop == n_total:
                logger.info("Branch stats: %d / %d events (%.0f%%)", stop, n_total, 100.0 * stop / n_total)
                last_pct = pct
            start = stop

    median_sample = _median_sample(tree, branches, n_total, scalar_branches, jagged_specs)
    return _to_rows(branches, acc, median_sample, n_total)


def _compute_stats_extended(
    tree: Any,
    feature_names: list[str],
    scalar_branches: list[str],
    jagged_specs: list[tuple[str, list[str]]],
    chunk: int,
    n_total: int,
    acc: dict[str, dict[str, Any]],
) -> None:
    """Chunked pass: load scalar + jagged, build matrix, update acc per column."""
    load_branches = list(scalar_branches) + [b for b, _ in jagged_specs]
    start = 0
    last_pct = -1
    while start < n_total:
        stop = min(start + chunk, n_total)
        arr = tree.arrays(load_branches, entry_start=start, entry_stop=stop)
        mat = build_chunk_matrix(arr, scalar_branches, jagged_specs, feature_names)
        for j, name in enumerate(feature_names):
            _update_acc(acc[name], mat[:, j])
        pct = int(100 * stop / n_total) if n_total > 0 else 100
        if pct >= last_pct + 5 or stop == n_total:
            logger.info("Branch stats: %d / %d events (%.0f%%)", stop, n_total, 100.0 * stop / n_total)
            last_pct = pct
        start = stop


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


def _median_sample(
    tree: Any,
    branches: list[str],
    n_total: int,
    scalar_branches: list[str] | None = None,
    jagged_specs: list[tuple[str, list[str]]] | None = None,
) -> dict[str, float]:
    """Compute median per feature from first min(200k, n_total) events."""
    n_sample = min(MEDIAN_SAMPLE_SIZE, n_total)
    if n_sample == 0:
        return {b: np.nan for b in branches}
    if jagged_specs and scalar_branches is not None:
        load_branches = list(scalar_branches) + [b for b, _ in jagged_specs]
        arr = tree.arrays(load_branches, entry_stop=n_sample)
        mat = build_chunk_matrix(arr, scalar_branches, jagged_specs, branches)
        return {name: float(np.nanmedian(mat[:, j])) for j, name in enumerate(branches)}
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
