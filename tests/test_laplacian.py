"""
Tests for muons.laplacian: L, spectrum, laplacian.npz (Step 6).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from muons.laplacian import build_laplacian, compute_spectrum, save_laplacian_npz


def test_build_laplacian_small() -> None:
    """L = D - W; row sums of L are 0; symmetric."""
    d = 4
    W = np.array(
        [
            [0, 1, 0.5, 0],
            [1, 0, 0.2, 0.3],
            [0.5, 0.2, 0, 0.1],
            [0, 0.3, 0.1, 0],
        ],
        dtype=np.float64,
    )
    L = build_laplacian(W)
    assert L.shape == (d, d)
    np.testing.assert_allclose(L, L.T)
    np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(L), W.sum(axis=1))
    np.testing.assert_allclose(L + W, np.diag(W.sum(axis=1)))


def test_build_laplacian_empty() -> None:
    """Empty W gives empty L."""
    W = np.zeros((0, 0), dtype=np.float64)
    L = build_laplacian(W)
    assert L.shape == (0, 0)


def test_compute_spectrum_small() -> None:
    """compute_spectrum returns L, eigenvalues (ascending), eigvec_first10."""
    d = 5
    W = np.maximum(np.random.default_rng(42).random((d, d)) * 0.5, 0.0)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    L, eigenvalues, eigvec_first10 = compute_spectrum(W, k_eigs=3)
    assert L.shape == (d, d)
    assert eigenvalues.shape == (d,)
    assert np.all(np.diff(eigenvalues) >= -1e-10)
    assert eigvec_first10.shape == (d, min(10, d))
    np.testing.assert_allclose(
        L @ eigvec_first10[:, 0], eigenvalues[0] * eigvec_first10[:, 0], atol=1e-8
    )


def test_compute_spectrum_empty() -> None:
    """Empty W returns zero-sized arrays."""
    W = np.zeros((0, 0), dtype=np.float64)
    L, eigenvalues, eigvec_first10 = compute_spectrum(W, k_eigs=10)
    assert L.shape == (0, 0)
    assert eigenvalues.shape == (0,)
    assert eigvec_first10.shape == (0, 0)


def test_save_laplacian_npz() -> None:
    """save_laplacian_npz writes L, lambda, eigvec_first10; load matches."""
    d = 3
    L = np.eye(d) - 0.1 * np.ones((d, d))
    eigenvalues = np.array([0.0, 0.3, 0.6], dtype=np.float64)
    eigvec_first10 = np.eye(d)[:, :3].astype(np.float64)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "laplacian.npz"
        save_laplacian_npz(L, eigenvalues, eigvec_first10, path)
        assert path.exists()
        data = np.load(path)
        assert "L" in data and "lambda" in data and "eigvec_first10" in data
        np.testing.assert_allclose(data["L"], L)
        np.testing.assert_allclose(data["lambda"], eigenvalues)
        np.testing.assert_allclose(data["eigvec_first10"], eigvec_first10)
