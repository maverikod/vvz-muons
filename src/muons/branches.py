"""
Branch (feature) selection: config list or auto from scan chunk (Step 2).
Supports jagged branches via aggregates (addontspc).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from typing import Any

import numpy as np

import awkward as ak  # type: ignore[import-untyped]

from muons.jagged_aggs import JAGGED_AGG_NAMES

# Constants from techspec §3 Step 2
MAX_SCAN_EVENTS = 20_000
MAX_BRANCHES = 64
NAN_RATE_THRESHOLD = 0.2
# addontspc
MAX_SCALAR_BRANCHES_DEFAULT = 32
MAX_JAGGED_BRANCHES_DEFAULT = 8
JAGGED_EMPTY_RATE_THRESHOLD = 0.5
DEFAULT_JAGGED_AGGS = ["len", "mean", "std", "min", "max", "l2"]


def select_branches(
    tree: Any,
    branches_config: list[str] | None,
    max_scan: int = MAX_SCAN_EVENTS,
) -> list[str]:
    """
    Determine branch list: from config (with existence check) or auto from scan.
    Legacy: returns only scalar branch names (no jagged).
    """
    if branches_config is not None and len(branches_config) > 0:
        return _branches_from_config(tree, branches_config)
    return _branches_auto(tree, max_scan)


def select_features(
    tree: Any,
    config: dict[str, Any],
    max_scan: int = MAX_SCAN_EVENTS,
) -> tuple[list[str], list[str], list[tuple[str, list[str]]]]:
    """
    Determine feature list: scalars + jagged aggregates when allow_jagged=true.

    Returns:
        (feature_names, scalar_branches, jagged_specs).
        feature_names = scalar_branches + [f"{b}__{a}" for (b, aggs) in jagged_specs for a in aggs].
    """
    max_scan = config.get("max_scan", max_scan)
    max_scalar = config.get("max_scalar_branches", MAX_SCALAR_BRANCHES_DEFAULT)
    branches_cfg = config.get("branches")

    if branches_cfg is not None and len(branches_cfg) > 0:
        scalar_branches = _branches_from_config(tree, branches_cfg)[:max_scalar]
    else:
        scalar_branches = _branches_auto(tree, max_scan)[:max_scalar]

    jagged_specs: list[tuple[str, list[str]]] = []
    if config.get("allow_jagged"):
        aggs = config.get("jagged_aggs") or list(DEFAULT_JAGGED_AGGS)
        aggs = [a for a in aggs if a in JAGGED_AGG_NAMES]
        if not aggs:
            aggs = list(DEFAULT_JAGGED_AGGS)
        max_jagged = config.get("max_jagged_branches", MAX_JAGGED_BRANCHES_DEFAULT)
        jagged_list = config.get("jagged_branches")
        if jagged_list is None:
            jagged_list = _jagged_branches_auto(tree, max_scan, max_jagged)
        else:
            available = set(_tree_keys(tree))
            for b in jagged_list:
                if b not in available:
                    raise ValueError(f"Jagged branch '{b}' not found in tree")
            jagged_list = jagged_list[:max_jagged]
        jagged_specs = [(b, list(aggs)) for b in jagged_list]

    feature_names = list(scalar_branches) + [f"{b}__{a}" for b, aggs in jagged_specs for a in aggs]
    return feature_names, scalar_branches, jagged_specs


def _tree_keys(tree: Any) -> list[str]:
    """Return list of branch names from tree (TTree or RNTuple)."""
    if hasattr(tree, "keys"):
        keys = tree.keys()
        return list(keys) if not isinstance(keys, list) else keys
    raise ValueError("Tree has no keys() method")


def _branches_from_config(tree: Any, branches_config: list[str]) -> list[str]:
    """Use config branch list; verify each exists. Return in config order."""
    available = set(_tree_keys(tree))
    for name in branches_config:
        if name not in available:
            raise ValueError(f"Branch '{name}' not found in tree")
    return list(branches_config)


def _branches_auto(tree: Any, max_scan: int) -> list[str]:
    """
    Auto-select branches: load up to max_scan events, filter scalar/numeric,
    nan_rate <= 0.2, std > 0; sort by nan_rate then name; cap at MAX_BRANCHES.
    """
    arr = tree.arrays(entry_stop=max_scan)
    fields = list(arr.fields)
    candidates: list[tuple[float, str]] = []  # (nan_rate, name) for sorting

    for name in fields:
        col = arr[name]
        # Scalar: one value per event (1D, no jagged)
        if col.ndim != 1:
            continue
        npy = ak.to_numpy(col)
        # Numeric: int/uint/float only
        if npy.dtype.kind not in ("i", "u", "f"):
            continue
        nan_rate = float(np.isnan(npy).mean())
        if nan_rate > NAN_RATE_THRESHOLD:
            continue
        std_b = np.nanstd(npy)
        if not (np.isfinite(std_b) and std_b > 0):
            continue
        candidates.append((nan_rate, name))

    if not candidates:
        raise ValueError(
            "Auto-selection: no branch passed filters (scalar, numeric, "
            "nan_rate <= 0.2, std > 0)."
        )

    candidates.sort(key=lambda x: (x[0], x[1]))
    selected = [name for _, name in candidates[:MAX_BRANCHES]]
    return selected


def _jagged_branches_auto(tree: Any, max_scan: int, max_jagged: int) -> list[str]:
    """
    Auto-select jagged branches: numeric arrays, empty rate <= 50%, std > 0.
    Priority: lower empty rate, then higher variance; cap at max_jagged.
    """
    arr = tree.arrays(entry_stop=max_scan)
    fields = list(arr.fields)
    candidates: list[tuple[float, float, str]] = []  # (empty_rate, -std, name)

    for name in fields:
        col = arr[name]
        try:
            lens = ak.num(col, axis=1)
        except Exception:
            continue
        n_ev = len(lens)
        if n_ev == 0:
            continue
        empty_rate = float(ak.sum(lens == 0) / n_ev)
        if empty_rate > JAGGED_EMPTY_RATE_THRESHOLD:
            continue
        flat = ak.flatten(col)
        npy = ak.to_numpy(flat)
        if npy.dtype.kind not in ("i", "u", "f"):
            continue
        std_all = np.nanstd(npy)
        if not (np.isfinite(std_all) and std_all > 0):
            continue
        candidates.append((empty_rate, -std_all, name))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return [name for _, _, name in candidates[:max_jagged]]
