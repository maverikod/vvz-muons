"""
Build observable matrix O: quantile (one-hot CSR) or zscore (dense memmap) — Step 4.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import awkward as ak  # type: ignore[import-untyped]
import numpy as np
from scipy.sparse import csr_matrix, save_npz  # type: ignore[import-untyped]

from muons.jagged_aggs import build_chunk_matrix

EDGE_SAMPLE_SIZE = 200_000


def build_quantile_O(
    tree: Any,
    branches: list[str],
    branch_stats_list: list[dict[str, Any]],
    out_path: Path,
    bins: int = 16,
    chunk: int = 200_000,
    max_events: int = 0,
    scalar_branches: list[str] | None = None,
    jagged_specs: list[tuple[str, list[str]]] | None = None,
) -> tuple[Path, Path]:
    """
    Build O as sparse CSR (events x sum(bins)) with one-hot per feature; chunked.

    When scalar_branches/jagged_specs are set, branches is the full feature list
    and chunk data is built via build_chunk_matrix.

    Returns:
        (path to bin_definitions.csv, path to O_matrix.npz).
    """
    stats_by_branch = {r["branch"]: r for r in branch_stats_list}
    n_entries = tree.num_entries
    n_total = min(n_entries, max_events) if max_events > 0 else n_entries

    edges_per_branch = _quantile_edges_from_sample(
        tree,
        branches,
        stats_by_branch,
        bins,
        n_total,
        scalar_branches,
        jagged_specs,
    )
    bin_def_path = out_path / "bin_definitions.csv"
    _write_bin_definitions(bin_def_path, branches, bins, edges_per_branch)

    d = len(branches) * bins
    offsets = [i * bins for i in range(len(branches))]

    if n_total == 0:
        o_mat = csr_matrix((0, d), dtype=np.float64)
        npz_path = out_path / "O_matrix.npz"
        _save_csr(npz_path, o_mat)
        return bin_def_path, npz_path

    row_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    load_branches = (
        list(scalar_branches) + [b for b, _ in jagged_specs]
        if jagged_specs and scalar_branches is not None
        else branches
    )
    start = 0
    last_pct = -1
    while start < n_total:
        stop = min(start + chunk, n_total)
        arr = tree.arrays(load_branches, entry_start=start, entry_stop=stop)
        pct = int(100 * stop / n_total) if n_total > 0 else 100
        if pct >= last_pct + 5 or stop == n_total:
            logger.info("Building O (quantile): %d / %d events (%.0f%%)", stop, n_total, 100.0 * stop / n_total)
            last_pct = pct
        if jagged_specs and scalar_branches is not None:
            mat = build_chunk_matrix(arr, scalar_branches, jagged_specs, branches)
        else:
            mat = np.column_stack(
                [
                    np.asarray(
                        (
                            ak.to_numpy(arr[b]).flatten()
                            if hasattr(arr[b], "ndim") and arr[b].ndim > 1
                            else ak.to_numpy(arr[b])
                        ),
                        dtype=np.float64,
                    )
                    for b in branches
                ]
            )
        n_chunk = stop - start
        rows = np.arange(n_chunk, dtype=np.int64) + start
        cols = np.empty((n_chunk, len(branches)), dtype=np.int64)
        for j, b in enumerate(branches):
            median_b = float(stats_by_branch[b]["median"])
            vals = np.where(np.isfinite(mat[:, j]), mat[:, j], median_b).astype(np.float64)
            edges_b = edges_per_branch[b]
            bin_idx = np.searchsorted(edges_b, vals, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, bins - 1)
            cols[:, j] = offsets[j] + bin_idx
        data = np.ones(n_chunk * len(branches), dtype=np.float64)
        row_flat = np.repeat(rows, len(branches))
        col_flat = cols.ravel()
        row_list.append(row_flat)
        col_list.append(col_flat)
        data_list.append(data)
        start = stop

    logger.info("Building O (quantile): assembling sparse matrix...")
    rows_all = np.concatenate(row_list)
    cols_all = np.concatenate(col_list)
    data_all = np.concatenate(data_list)
    o_mat = csr_matrix((data_all, (rows_all, cols_all)), shape=(n_total, d), dtype=np.float64)
    npz_path = out_path / "O_matrix.npz"
    _save_csr(npz_path, o_mat)
    return bin_def_path, npz_path


def build_zscore_O(
    tree: Any,
    branches: list[str],
    branch_stats_list: list[dict[str, Any]],
    out_path: Path,
    chunk: int = 200_000,
    max_events: int = 0,
    scalar_branches: list[str] | None = None,
    jagged_specs: list[tuple[str, list[str]]] | None = None,
) -> tuple[Path, Path]:
    """
    Build O as dense float64 (events x len(branches)); chunked write to memmap.
    When scalar_branches/jagged_specs set, chunk data from build_chunk_matrix.
    """
    stats_by_branch = {r["branch"]: r for r in branch_stats_list}
    n_entries = tree.num_entries
    n_total = min(n_entries, max_events) if max_events > 0 else n_entries
    d = len(branches)

    params: dict[str, dict[str, float]] = {}
    for b in branches:
        r = stats_by_branch[b]
        params[b] = {
            "mean": float(r["mean"]),
            "std": float(r["std"]),
            "median": float(r["median"]),
        }
    json_path = out_path / "zscore_params.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)

    npy_path = out_path / "O_matrix.npy"
    o_mat = np.memmap(npy_path, dtype=np.float64, mode="w+", shape=(n_total, d))

    if n_total == 0:
        del o_mat
        return json_path, npy_path

    load_branches = (
        list(scalar_branches) + [b for b, _ in jagged_specs]
        if jagged_specs and scalar_branches is not None
        else branches
    )
    start = 0
    last_pct = -1
    while start < n_total:
        stop = min(start + chunk, n_total)
        arr = tree.arrays(load_branches, entry_start=start, entry_stop=stop)
        pct = int(100 * stop / n_total) if n_total > 0 else 100
        if pct >= last_pct + 5 or stop == n_total:
            logger.info("Building O (zscore): %d / %d events (%.0f%%)", stop, n_total, 100.0 * stop / n_total)
            last_pct = pct
        if jagged_specs and scalar_branches is not None:
            mat = build_chunk_matrix(arr, scalar_branches, jagged_specs, branches)
        else:
            mat = np.column_stack(
                [
                    np.asarray(
                        (
                            ak.to_numpy(arr[b]).flatten()
                            if hasattr(arr[b], "ndim") and arr[b].ndim > 1
                            else ak.to_numpy(arr[b])
                        ),
                        dtype=np.float64,
                    )
                    for b in branches
                ]
            )
        chunk_rows = stop - start
        O_chunk = np.empty((chunk_rows, d), dtype=np.float64)
        for j, b in enumerate(branches):
            median_b = float(stats_by_branch[b]["median"])
            mean_b = float(stats_by_branch[b]["mean"])
            std_b = float(stats_by_branch[b]["std"])
            x = np.where(np.isfinite(mat[:, j]), mat[:, j], median_b).astype(np.float64)
            if std_b > 0:
                O_chunk[:, j] = (x - mean_b) / std_b
            else:
                O_chunk[:, j] = 0.0
        o_mat[start:stop, :] = O_chunk
        start = stop

    o_mat.flush()
    del o_mat
    return json_path, npy_path


def _quantile_edges_from_sample(
    tree: Any,
    branches: list[str],
    stats_by_branch: dict[str, dict[str, Any]],
    bins: int,
    n_total: int,
    scalar_branches: list[str] | None = None,
    jagged_specs: list[tuple[str, list[str]]] | None = None,
) -> dict[str, np.ndarray]:
    """Compute (bins+1) edges per feature from first min(200k, n_total) events."""
    n_sample = min(EDGE_SAMPLE_SIZE, n_total)
    if n_sample == 0:
        edges_empty = np.concatenate([[-np.inf], np.zeros(bins - 1, dtype=np.float64), [np.inf]])
        return {b: edges_empty.copy() for b in branches}
    if jagged_specs and scalar_branches is not None:
        load_branches = list(scalar_branches) + [b for b, _ in jagged_specs]
        arr = tree.arrays(load_branches, entry_stop=n_sample)
        mat = build_chunk_matrix(arr, scalar_branches, jagged_specs, branches)
    else:
        arr = tree.arrays(branches, entry_stop=n_sample)
        mat = np.column_stack(
            [
                np.asarray(
                    (
                        ak.to_numpy(arr[b]).flatten()
                        if hasattr(arr[b], "ndim") and arr[b].ndim > 1
                        else ak.to_numpy(arr[b])
                    ),
                    dtype=np.float64,
                )
                for b in branches
            ]
        )
    edges_per_branch: dict[str, np.ndarray] = {}
    for j, b in enumerate(branches):
        median_b = float(stats_by_branch[b]["median"])
        vals = np.where(np.isfinite(mat[:, j]), mat[:, j], median_b).astype(np.float64)
        q = np.linspace(0, 1, bins + 1)
        quantiles = np.nanquantile(vals, q)
        edges = np.concatenate([[-np.inf], quantiles[1:-1], [np.inf]])
        edges_per_branch[b] = edges.astype(np.float64)
    return edges_per_branch


def _write_bin_definitions(
    path: Path,
    branches: list[str],
    bins: int,
    edges_per_branch: dict[str, np.ndarray],
) -> None:
    """Write bin_definitions.csv: branch, bin_id, left_edge, right_edge."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["branch", "bin_id", "left_edge", "right_edge"])
        for b in branches:
            edges = edges_per_branch[b]
            for k in range(bins):
                left = edges[k]
                right = edges[k + 1]
                writer.writerow([b, k, left, right])


def _save_csr(path: Path, o_mat: csr_matrix) -> None:
    """Save CSR matrix to O_matrix.npz (scipy.sparse format)."""
    save_npz(path, o_mat)
