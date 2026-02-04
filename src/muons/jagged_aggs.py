"""
Jagged-array aggregates: one scalar per event (addontspc §E).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from typing import Any

import awkward as ak  # type: ignore[import-untyped]
import numpy as np


def _quantile_per_event(flat: Any, q: float) -> np.ndarray:
    """
    Per-event quantile over jagged lists; vectorized via pad-to-rectangular + nanquantile(axis=1).
    """
    max_len = int(ak.max(ak.num(flat, axis=1)))
    if max_len <= 0:
        return np.full(len(flat), np.nan, dtype=np.float64)
    padded = ak.pad_none(flat, target=max_len, clip=True)
    filled = ak.fill_none(padded, np.nan)
    rect = np.asarray(ak.to_numpy(filled), dtype=np.float64)
    return np.nanquantile(rect, q, axis=1).astype(np.float64)


# Allowed aggregator names per addontspc
JAGGED_AGG_NAMES = (
    "len",
    "sum",
    "mean",
    "std",
    "min",
    "max",
    "q25",
    "median",
    "q75",
    "l2",
)


def compute_agg(column: Any, agg: str) -> np.ndarray:
    """
    Compute one aggregate per event for a jagged numeric column.

    Empty array -> len=0, all other aggs=NaN. Population std: sqrt(mean((a-mean)^2)).

    Args:
        column: Awkward array (variable-length lists of numbers).
        agg: One of len, sum, mean, std, min, max, q25, median, q75, l2.

    Returns:
        1D float array, length = len(column). NaN where agg undefined (e.g. empty).
    """
    if agg == "len":
        return np.asarray(ak.num(column, axis=1), dtype=np.float64)
    n_ev = len(column)
    out = np.full(n_ev, np.nan, dtype=np.float64)
    lens = ak.num(column, axis=1)
    mask = lens > 0
    if not np.any(mask):
        return out
    flat = column[mask]
    if agg == "sum":
        vals = ak.sum(flat, axis=1)
    elif agg == "mean":
        vals = ak.mean(flat, axis=1)
    elif agg == "std":
        # Population std: sqrt(mean((a-mean)^2))
        mean = ak.mean(flat, axis=1)
        diff = flat - ak.broadcast_arrays(mean, flat)[0]
        vals = np.sqrt(ak.mean(diff**2, axis=1))
    elif agg == "min":
        vals = ak.min(flat, axis=1)
    elif agg == "max":
        vals = ak.max(flat, axis=1)
    elif agg in ("q25", "median", "q75"):
        q = 0.25 if agg == "q25" else 0.5 if agg == "median" else 0.75
        vals = _quantile_per_event(flat, q)
        out[mask] = np.asarray(vals, dtype=np.float64)
        return out
    elif agg == "l2":
        vals = np.sqrt(ak.sum(flat**2, axis=1))
    else:
        raise ValueError(f"Unknown aggregator: {agg}")
    out[mask] = np.asarray(ak.to_numpy(vals), dtype=np.float64)
    return out


def build_chunk_matrix(
    arr: Any,
    scalar_branches: list[str],
    jagged_specs: list[tuple[str, list[str]]],
    feature_names: list[str],
) -> np.ndarray:
    """
    Build (chunk_size x n_features) matrix: scalar columns then jagged agg columns.

    Args:
        arr: Awkward array from tree.arrays(...) for one chunk.
        scalar_branches: List of scalar branch names.
        jagged_specs: List of (branch_name, [agg1, agg2, ...]).
        feature_names: Full ordered list of feature names (must match columns).

    Returns:
        (n_events, n_features) float64. NaN for undefined (e.g. empty-array aggs).
    """
    cols: list[np.ndarray] = []
    for b in scalar_branches:
        col = arr[b]
        npy = (
            ak.to_numpy(col).flatten()
            if hasattr(col, "ndim") and col.ndim > 1
            else np.asarray(ak.to_numpy(col)).flatten()
        )
        cols.append(np.asarray(npy, dtype=np.float64))
    for branch, aggs in jagged_specs:
        col = arr[branch]
        for a in aggs:
            cols.append(compute_agg(col, a))
    return np.column_stack(cols)
