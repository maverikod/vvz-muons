"""
Correlation matrix C and connectivity matrix W from observable matrix O — Step 5.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, load_npz  # type: ignore[import-untyped]

# Type for O: sparse CSR or dense array
OMatrix = Union[csr_matrix, np.ndarray]


def compute_correlation(o_mat: OMatrix) -> np.ndarray:
    """
    Compute Pearson correlation matrix C between columns of O.

    Args:
        o_mat: Observable matrix, shape (N, d). Sparse CSR (quantile) or dense (zscore).

    Returns:
        C: Correlation matrix (d, d), float64. Symmetric, diag 1 (or 0 if zero variance).
    """
    if hasattr(o_mat, "shape"):
        n_rows, d = o_mat.shape
    else:
        n_rows, d = o_mat.shape[0], o_mat.shape[1]
    if d == 0 or n_rows == 0:
        return np.zeros((d, d), dtype=np.float64)

    if isinstance(o_mat, csr_matrix):
        return _correlation_sparse(o_mat, n_rows, d)
    return _correlation_dense(np.asarray(o_mat), n_rows, d)


def _correlation_sparse(o_mat: csr_matrix, N: int, d: int) -> np.ndarray:
    """C from sparse O: Cov = (O.T @ O)/N, then normalize to correlation."""
    gram = (o_mat.T @ o_mat).toarray().astype(np.float64)
    gram /= N
    # Cov[i,j] = gram[i,j] - mean_i*mean_j; mean_i = gram[i,i] for one-hot
    means = np.diag(gram).copy()
    cov = gram - np.outer(means, means)
    sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return _cov_to_corr(cov, sigma, d)


def _correlation_dense(o_mat: np.ndarray, N: int, d: int) -> np.ndarray:
    """C from dense O: center, then Cov = (O_c.T @ O_c)/(N-1)."""
    o_centered = o_mat - np.nanmean(o_mat, axis=0)
    o_centered = np.where(np.isfinite(o_centered), o_centered, 0.0)
    ddof = 1 if N > 1 else 0
    cov = (o_centered.T @ o_centered) / max(N - ddof, 1)
    sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return _cov_to_corr(cov, sigma, d)


def _cov_to_corr(cov: np.ndarray, sigma: np.ndarray, d: int) -> np.ndarray:
    """Convert covariance to correlation; handle zero variance."""
    C = np.zeros((d, d), dtype=np.float64)
    np.fill_diagonal(C, 1.0)
    ok = sigma > 1e-14
    if not np.any(ok):
        return C
    outer = np.outer(sigma, sigma)
    np.place(outer, outer <= 0, 1.0)
    C = cov / outer
    np.fill_diagonal(C, 1.0)
    C[~ok, :] = 0.0
    C[:, ~ok] = 0.0
    return C


def compute_correlation_from_files(
    out_path: Path,
    mode: str,
    branches: list[str],
    chunk: int = 200_000,
) -> np.ndarray:
    """
    Compute C by loading O from Step 4 output (quantile npz or zscore chunked).

    Args:
        out_path: Directory containing O_matrix.npz or O_matrix.npy.
        mode: "quantile" or "zscore".
        branches: Branch names (used for quantile; zscore infers d from O shape).
        chunk: Row chunk size for zscore dense read.

    Returns:
        C: Correlation matrix (d, d).
    """
    if mode == "quantile":
        npz_path = out_path / "O_matrix.npz"
        o_mat = load_npz(npz_path)
        return compute_correlation(o_mat)

    # zscore: O_matrix.npy is already z-scored; chunked Gram to avoid loading full O
    npy_path = out_path / "O_matrix.npy"
    O_mmap = np.load(npy_path, mmap_mode="r")
    N, d = O_mmap.shape
    if N == 0 or d == 0:
        return np.zeros((d, d), dtype=np.float64)
    gram = np.zeros((d, d), dtype=np.float64)
    start = 0
    while start < N:
        stop = min(start + chunk, N)
        chunk_arr = np.array(O_mmap[start:stop], dtype=np.float64)
        gram += chunk_arr.T @ chunk_arr
        start = stop
    cov = gram / max(N - 1, 1)
    sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return _cov_to_corr(cov, sigma, d)


def build_W(C: np.ndarray, tau: float = 0.1, topk: int = 0) -> np.ndarray:
    """
    Build connectivity matrix W from correlation C: W = max(0,C), diag=0, then sparsify.

    Args:
        C: Correlation matrix (d, d).
        tau: Zero out W[i,j] where W[i,j] < tau (default 0.1).
        topk: Keep top-k edges per row (0 = off), then symmetrize.

    Returns:
        W: (d, d) non-negative, diag 0.
    """
    d = C.shape[0]
    W = np.maximum(C, 0.0)
    np.fill_diagonal(W, 0.0)

    if tau > 0:
        W[W < tau] = 0.0

    if topk > 0 and d > 1:
        k = min(topk, d - 1)
        W_copy = W.copy()
        np.fill_diagonal(W_copy, -np.inf)
        topk_idx = np.argpartition(W_copy, -k, axis=1)[:, -k:]
        row_ar = np.arange(d, dtype=np.intp)[:, None]
        W_out = np.zeros_like(W)
        W_out[row_ar, topk_idx] = W[row_ar, topk_idx]
        W = np.maximum(W_out, W_out.T)
        np.fill_diagonal(W, 0.0)

    return np.asarray(W, dtype=np.float64)


def save_corr_npz(C: np.ndarray, W: np.ndarray, path: Path) -> None:
    """Save C and W to corr.npz (float64 arrays)."""
    np.savez_compressed(path, C=C.astype(np.float64), W=W.astype(np.float64))
