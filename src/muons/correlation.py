"""
Correlation matrix C and connectivity matrix W from observable matrix O — Step 5.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)
from scipy.sparse import csr_matrix, load_npz  # type: ignore[import-untyped]

from muons.backend import get_backend, to_numpy

# Type for O: sparse CSR or dense array
OMatrix = Union[csr_matrix, np.ndarray]


def compute_correlation(o_mat: OMatrix) -> np.ndarray:
    """
    Compute Pearson correlation matrix C between columns of O.

    Uses GPU (CuPy) when available; CPU only when no CUDA device is present.

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

    # Estimate GPU memory: data + Gram/cov (float64 = 8 bytes)
    if isinstance(o_mat, csr_matrix):
        req_b = o_mat.data.nbytes + o_mat.indices.nbytes + o_mat.indptr.nbytes + 2 * d * d * 8
    else:
        req_b = 2 * n_rows * d * 8 + 2 * d * d * 8  # O + centered + Gram + cov
    xp, _ = get_backend(required_bytes=req_b)
    if isinstance(o_mat, csr_matrix):
        return _correlation_sparse(o_mat, n_rows, d, xp)
    return _correlation_dense(np.asarray(o_mat), n_rows, d, xp)


def _correlation_sparse(o_mat: csr_matrix, N: int, d: int, xp: Any) -> np.ndarray:
    """C from sparse O: Cov = (O.T @ O)/N, then normalize to correlation."""
    if xp is np:
        gram = (o_mat.T @ o_mat).toarray().astype(np.float64)
    else:
        o_gpu = xp.sparse.csr_matrix(
            (xp.asarray(o_mat.data), xp.asarray(o_mat.indices), xp.asarray(o_mat.indptr)),
            shape=o_mat.shape,
        )
        gram = (o_gpu.T @ o_gpu).toarray().astype(xp.float64)
    gram /= N
    means = xp.diag(gram).copy()
    cov = gram - xp.outer(means, means)
    sigma = xp.sqrt(xp.maximum(xp.diag(cov), 0.0))
    C = _cov_to_corr(cov, sigma, d, xp)
    return to_numpy(xp, C)


def _correlation_dense(o_mat: np.ndarray, N: int, d: int, xp: Any) -> np.ndarray:
    """C from dense O: center, then Cov = (O_c.T @ O_c)/(N-1)."""
    o_x = xp.asarray(o_mat, dtype=xp.float64)
    o_centered = o_x - xp.nanmean(o_x, axis=0)
    o_centered = xp.where(xp.isfinite(o_centered), o_centered, 0.0)
    ddof = 1 if N > 1 else 0
    cov = (o_centered.T @ o_centered) / max(N - ddof, 1)
    sigma = xp.sqrt(xp.maximum(xp.diag(cov), 0.0))
    C = _cov_to_corr(cov, sigma, d, xp)
    return to_numpy(xp, C)


def _cov_to_corr(cov: Any, sigma: Any, d: int, xp: Any) -> Any:
    """Convert covariance to correlation; handle zero variance. Returns xp-array."""
    C = xp.zeros((d, d), dtype=xp.float64)
    xp.fill_diagonal(C, 1.0)
    ok = sigma > 1e-14
    if not xp.any(ok):
        return C
    outer = xp.outer(sigma, sigma)
    outer = xp.where(outer <= 0, 1.0, outer)
    C = cov / outer
    xp.fill_diagonal(C, 1.0)
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

    # zscore: O_matrix.npy is already z-scored; chunked Gram (GPU when available)
    npy_path = out_path / "O_matrix.npy"
    O_mmap = np.load(npy_path, mmap_mode="r")
    N, d = O_mmap.shape
    if N == 0 or d == 0:
        return np.zeros((d, d), dtype=np.float64)
    req_b = chunk * d * 8 + 2 * d * d * 8  # chunk + Gram + cov
    xp, _ = get_backend(required_bytes=req_b)
    gram = xp.zeros((d, d), dtype=xp.float64)
    start = 0
    last_pct = -1
    while start < N:
        stop = min(start + chunk, N)
        chunk_arr = np.array(O_mmap[start:stop], dtype=np.float64)
        chunk_x = xp.asarray(chunk_arr)
        gram += chunk_x.T @ chunk_x
        pct = int(100 * stop / N) if N > 0 else 100
        if pct >= last_pct + 5 or stop == N:
            logger.info("Correlation (zscore): %d / %d events (%.0f%%)", stop, N, 100.0 * stop / N)
            last_pct = pct
        start = stop
    cov = gram / max(N - 1, 1)
    sigma = xp.sqrt(xp.maximum(xp.diag(cov), 0.0))
    C = _cov_to_corr(cov, sigma, d, xp)
    return to_numpy(xp, C)


def build_W(C: np.ndarray, tau: float = 0.1, topk: int = 0) -> np.ndarray:
    """
    Build connectivity matrix W from correlation C: W = max(0,C), diag=0, then sparsify.

    Uses GPU when available. Sparsification order per techspec §3 Step 5:
    first topk (keep top-k per row, symmetrize), then tau (zero out W < tau).

    Args:
        C: Correlation matrix (d, d).
        tau: Zero out W[i,j] where W[i,j] < tau (default 0.1).
        topk: Keep top-k edges per row (0 = off), then symmetrize.

    Returns:
        W: (d, d) non-negative, diag 0.
    """
    d = C.shape[0]
    req_b = 3 * d * d * 8  # C, W, W_copy for topk
    xp, _ = get_backend(required_bytes=req_b)
    W = xp.maximum(xp.asarray(C, dtype=xp.float64), 0.0)
    xp.fill_diagonal(W, 0.0)

    if topk > 0 and d > 1:
        k = min(topk, d - 1)
        inf_val = xp.float64("-inf")
        W_copy = W.copy()
        xp.fill_diagonal(W_copy, inf_val)
        topk_idx = xp.argpartition(W_copy, -k, axis=1)[:, -k:]
        row_ar = xp.arange(d, dtype=xp.intp)[:, None]
        W_out = xp.zeros_like(W)
        W_out[row_ar, topk_idx] = W[row_ar, topk_idx]
        W = xp.maximum(W_out, W_out.T)
        xp.fill_diagonal(W, 0.0)

    if tau > 0:
        W[W < tau] = 0.0

    return to_numpy(xp, W)


def save_corr_npz(C: np.ndarray, W: np.ndarray, path: Path) -> None:
    """Save C and W to corr.npz (float64 arrays)."""
    np.savez_compressed(path, C=C.astype(np.float64), W=W.astype(np.float64))
