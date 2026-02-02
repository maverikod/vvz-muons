"""
Branch (feature) selection: config list or auto from scan chunk (Step 2).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from typing import Any

import numpy as np

import awkward as ak  # type: ignore[import-untyped]

# Constants from techspec §3 Step 2
MAX_SCAN_EVENTS = 20_000
MAX_BRANCHES = 64
NAN_RATE_THRESHOLD = 0.2


def select_branches(
    tree: Any,
    branches_config: list[str] | None,
    max_scan: int = MAX_SCAN_EVENTS,
) -> list[str]:
    """
    Determine branch list: from config (with existence check) or auto from scan.

    Args:
        tree: TTree or RNTuple (must have .keys() and .arrays(entry_stop=...)).
        branches_config: Optional list of branch names from YAML; if non-empty,
            use it after validating each branch exists.
        max_scan: Max events to load for auto-selection (only when branches_config
            is None or empty).

    Returns:
        Ordered list of branch names (length <= MAX_BRANCHES for auto).

    Raises:
        ValueError: If a config branch is missing, or if auto-selection finds
            no branch passing filters.
    """
    if branches_config is not None and len(branches_config) > 0:
        return _branches_from_config(tree, branches_config)
    return _branches_auto(tree, max_scan)


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
