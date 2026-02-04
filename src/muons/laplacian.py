"""
Laplacian L from W and eigenvalue spectrum — Step 6.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import eigsh  # type: ignore[import-untyped]

from muons.backend import get_backend, to_numpy

DENSE_EIGH_THRESHOLD = 500


def build_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Build graph Laplacian L = D - W with D = diag(row_sums(W)).

    Uses GPU when available. Returns numpy array.

    Args:
        W: Connectivity matrix (d, d), non-negative, diag 0.

    Returns:
        L: Laplacian (d, d), symmetric, row-sum 0.
    """
    d = W.shape[0]
    req_b = 2 * d * d * 8  # W, L (float64)
    xp, _ = get_backend(required_bytes=req_b)
    W_x = xp.asarray(W, dtype=xp.float64)
    d_vec = xp.asarray(W_x.sum(axis=1), dtype=xp.float64).flatten()
    D = xp.diag(d_vec)
    L_x = D - W_x
    return to_numpy(xp, L_x)


def compute_spectrum(
    W: np.ndarray,
    k_eigs: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build L from W and compute smallest eigenvalues and first 10 eigenvectors.

    Uses GPU (CuPy) when available; CPU only when no CUDA device is present.

    Args:
        W: Connectivity matrix (d, d).
        k_eigs: Number of smallest eigenvalues when d > 500.

    Returns:
        L: Laplacian (d, d), dense.
        eigenvalues: 1D array, sorted ascending (length d if d<=500 else k).
        eigvec_first10: (d, 10) first eigenvectors (or (d, k) if k < 10).
    """
    d = W.shape[0]
    if d == 0:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
        )

    k = min(max(k_eigs, 10), d)
    req_b = 3 * d * d * 8 + k * d * 8  # W, L, eigenvectors
    xp, use_gpu = get_backend(required_bytes=req_b)
    W_x = xp.asarray(W, dtype=xp.float64)
    d_vec = xp.asarray(W_x.sum(axis=1), dtype=xp.float64).flatten()
    D = xp.diag(d_vec)
    L_x = D - W_x

    if d <= DENSE_EIGH_THRESHOLD:
        w, v = _eigen_dense(L_x, d, xp)
    else:
        w, v = _eigen_sparse(L_x, k_eigs, d, xp)

    n_vecs = min(10, v.shape[1])
    eigvec_first10 = to_numpy(xp, v[:, :n_vecs])
    eigenvalues = to_numpy(xp, w)
    L_np = to_numpy(xp, L_x)

    return L_np, eigenvalues, eigvec_first10


def _eigen_dense(L: Any, d: int, xp: Any) -> tuple[Any, Any]:
    """Full eigendecomposition via eigh; eigenvalues ascending."""
    w, v = xp.linalg.eigh(L)
    order = xp.argsort(w)
    return w[order].astype(xp.float64), v[:, order].astype(xp.float64)


def _eigen_sparse(L: Any, k_eigs: int, d: int, xp: Any) -> tuple[Any, Any]:
    """Smallest k eigenvalues via eigsh; k >= 10 for eigvec_first10."""
    k = min(max(k_eigs, 10), d)
    if xp is np:
        L_sparse = csr_matrix(L)
        w, v = eigsh(L_sparse, k=k, which="SM")
        order = np.argsort(w)
        return w[order].astype(np.float64), v[:, order].astype(np.float64)
    # GPU: cupyx.scipy.sparse.linalg.eigsh (lazy import)
    from cupyx.scipy.sparse import linalg as cp_eigsh  # type: ignore[import-untyped]

    L_sparse = xp.sparse.csr_matrix(L)
    w, v = cp_eigsh.eigsh(L_sparse, k=k, which="SA")
    order = xp.argsort(w)
    return w[order].astype(xp.float64), v[:, order].astype(xp.float64)


def save_laplacian_npz(
    L: np.ndarray,
    eigenvalues: np.ndarray,
    eigvec_first10: np.ndarray,
    path: Path,
) -> None:
    """
    Save L, eigenvalues, and first 10 eigenvectors to laplacian.npz.

    Uses key 'lambda' for eigenvalues (techspec). NumPy allows 'lambda' as
    array key in npz.

    Args:
        L: Laplacian (d, d).
        eigenvalues: 1D array.
        eigvec_first10: (d, 10) or (d, k).
        path: Output file path.
    """
    # "lambda" is reserved in Python; pass via dict for npz key
    kwargs: dict[str, np.ndarray] = {
        "L": L.astype(np.float64),
        "eigvec_first10": eigvec_first10.astype(np.float64),
        "lambda": eigenvalues.astype(np.float64),
    }
    np.savez_compressed(path, **kwargs)  # type: ignore[arg-type]
