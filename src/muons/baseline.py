"""
Baseline control: column-shuffle O, repeat Steps 5–7 — Step 8.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, load_npz  # type: ignore[import-untyped]

OMatrix = Union[csr_matrix, np.ndarray]

BASELINE_DEFAULT_SEED = 0


def shuffle_O_columns(o_mat: OMatrix, seed: int | None = BASELINE_DEFAULT_SEED) -> OMatrix:
    """
    Shuffle each column of O independently to destroy correlations, keep marginals.

    Args:
        o_mat: Observable matrix (N, d), sparse CSR or dense.
        seed: RNG seed for reproducibility (default 0).

    Returns:
        O_shuffled: Same shape and type as o_mat.
    """
    rng = np.random.default_rng(seed)
    if isinstance(o_mat, csr_matrix):
        return _shuffle_csr_columns(o_mat, rng)
    return _shuffle_dense_columns(np.asarray(o_mat), rng)


def _shuffle_dense_columns(o_mat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each column of dense O in place (copy)."""
    o_shuffled = np.empty_like(o_mat)
    d = o_mat.shape[1]
    for j in range(d):
        o_shuffled[:, j] = rng.permutation(o_mat[:, j])
    return o_shuffled


def _shuffle_csr_columns(o_mat: csr_matrix, rng: np.random.Generator) -> csr_matrix:
    """Shuffle row indices within each column of CSR O; data unchanged."""
    o_coo = o_mat.tocoo()
    row = np.asarray(o_coo.row, dtype=np.intp)
    col = np.asarray(o_coo.col, dtype=np.intp)
    data = np.asarray(o_coo.data, dtype=np.float64)
    n = o_mat.shape[0]
    d = o_mat.shape[1]
    for j in range(d):
        mask = col == j
        if not np.any(mask):
            continue
        rows_j = row[mask]
        row[mask] = rng.permutation(rows_j)
    return csr_matrix((data, (row, col)), shape=(n, d), dtype=np.float64)


def load_O_for_baseline(out_path: Path, mode: str) -> OMatrix:
    """
    Load full O from Step 4 output for baseline (column-shuffle). O in memory.

    Args:
        out_path: Directory with O_matrix.npz or O_matrix.npy.
        mode: "quantile" or "zscore".

    Returns:
        Observable matrix: Sparse CSR (quantile) or dense ndarray (zscore).
    """
    if mode == "quantile":
        return load_npz(out_path / "O_matrix.npz")
    o_mmap = np.load(out_path / "O_matrix.npy", mmap_mode="r")
    return np.asarray(o_mmap, dtype=np.float64, order="C")
