"""
Tests for muons.correlation: C, W, corr.npz (Step 5).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from muons.correlation import (
    build_W,
    compute_correlation,
    compute_correlation_from_files,
    save_corr_npz,
)


def test_compute_correlation_dense_identity() -> None:
    """Dense O with unit columns gives C = I (up to numerical)."""
    n, d = 100, 5
    rng = np.random.default_rng(42)
    o_mat = rng.standard_normal((n, d))
    C = compute_correlation(o_mat)
    assert C.shape == (d, d)
    np.testing.assert_allclose(np.diag(C), 1.0)
    assert np.allclose(C, C.T)
    off_diag = C[~np.eye(d, dtype=bool)]
    assert np.all(np.abs(off_diag) <= 1.0 + 1e-10)


def test_compute_correlation_sparse_small() -> None:
    """Sparse one-hot O: C is (d,d), symmetric, diag 1."""
    n, d = 50, 10
    rows = np.arange(n)
    cols = np.minimum(np.arange(n) % d, d - 1)
    data = np.ones(n)
    o_mat = csr_matrix((data, (rows, cols)), shape=(n, d))
    C = compute_correlation(o_mat)
    assert C.shape == (d, d)
    np.testing.assert_allclose(np.diag(C), 1.0)
    assert np.allclose(C, C.T)


def test_compute_correlation_empty() -> None:
    """O with N=0 returns (d,d) zero C."""
    o_mat = np.zeros((0, 3))
    C = compute_correlation(o_mat)
    assert C.shape == (3, 3)
    np.testing.assert_array_equal(C, 0.0)


def test_build_W_basic() -> None:
    """W = max(0,C), diag 0; tau zeros small entries."""
    d = 4
    C = np.array(
        [
            [1.0, 0.5, -0.2, 0.8],
            [0.5, 1.0, 0.3, -0.1],
            [-0.2, 0.3, 1.0, 0.4],
            [0.8, -0.1, 0.4, 1.0],
        ]
    )
    W = build_W(C, tau=0.0, topk=0)
    assert W.shape == (d, d)
    np.testing.assert_array_equal(np.diag(W), 0.0)
    assert np.all(W >= 0)
    assert np.allclose(W, W.T)
    np.testing.assert_allclose(W[0, 1], 0.5)
    np.testing.assert_allclose(W[0, 2], 0.0)

    W_tau = build_W(C, tau=0.35, topk=0)
    assert W_tau[0, 1] == 0.5
    assert W_tau[0, 3] == 0.8
    assert W_tau[0, 2] == 0.0
    assert W_tau[1, 2] == 0.0


def test_build_W_topk() -> None:
    """Topk keeps top-k per row then symmetrize; W symmetric, diag 0."""
    d = 4
    C = np.array(
        [
            [1.0, 0.9, 0.1, 0.5],
            [0.9, 1.0, 0.2, 0.3],
            [0.1, 0.2, 1.0, 0.8],
            [0.5, 0.3, 0.8, 1.0],
        ]
    )
    W = build_W(C, tau=0.0, topk=2)
    assert W.shape == (d, d)
    np.testing.assert_array_equal(np.diag(W), 0.0)
    assert np.allclose(W, W.T)
    assert np.all(W >= 0)


def test_save_corr_npz() -> None:
    """save_corr_npz writes C and W; load matches."""
    d = 3
    C = np.eye(d) + 0.1 * np.ones((d, d))
    W = build_W(C, tau=0.0, topk=0)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "corr.npz"
        save_corr_npz(C, W, path)
        assert path.exists()
        data = np.load(path)
        assert "C" in data and "W" in data
        np.testing.assert_allclose(data["C"], C)
        np.testing.assert_allclose(data["W"], W)


def test_compute_correlation_from_files_quantile() -> None:
    """compute_correlation_from_files(quantile) loads npz and returns C."""
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        n, d = 60, 8
        o_mat = csr_matrix((np.ones(n), (np.arange(n), np.arange(n) % d)), shape=(n, d))
        npz_path = out / "O_matrix.npz"
        save_npz(npz_path, o_mat)
        C = compute_correlation_from_files(out, "quantile", ["x"] * d)
        assert C.shape == (d, d)
        np.testing.assert_allclose(np.diag(C), 1.0)
