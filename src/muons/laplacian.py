"""
Laplacian L from W and eigenvalue spectrum — Step 6.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import eigsh  # type: ignore[import-untyped]

DENSE_EIGH_THRESHOLD = 500


def build_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Build graph Laplacian L = D - W with D = diag(row_sums(W)).

    Args:
        W: Connectivity matrix (d, d), non-negative, diag 0.

    Returns:
        L: Laplacian (d, d), symmetric, row-sum 0.
    """
    d_vec = np.asarray(W.sum(axis=1), dtype=np.float64).flatten()
    D = np.diag(d_vec)
    return np.asarray(D - W, dtype=np.float64)


def compute_spectrum(
    W: np.ndarray,
    k_eigs: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build L from W and compute smallest eigenvalues and first 10 eigenvectors.

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

    L = build_laplacian(W)

    if d <= DENSE_EIGH_THRESHOLD:
        eigenvalues, eigenvectors = _eigen_dense(L, d)
    else:
        eigenvalues, eigenvectors = _eigen_sparse(L, k_eigs, d)

    n_vecs = min(10, eigenvectors.shape[1])
    eigvec_first10 = np.asarray(eigenvectors[:, :n_vecs], dtype=np.float64)

    return L, eigenvalues, eigvec_first10


def _eigen_dense(L: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Full eigendecomposition via eigh; eigenvalues ascending."""
    w, v = np.linalg.eigh(L)
    order = np.argsort(w)
    return w[order].astype(np.float64), v[:, order].astype(np.float64)


def _eigen_sparse(
    L: np.ndarray,
    k_eigs: int,
    d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Smallest k eigenvalues via eigsh; k >= 10 for eigvec_first10."""
    k = min(max(k_eigs, 10), d)
    L_sparse = csr_matrix(L)
    w, v = eigsh(L_sparse, k=k, which="SM")
    order = np.argsort(w)
    return w[order].astype(np.float64), v[:, order].astype(np.float64)


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
