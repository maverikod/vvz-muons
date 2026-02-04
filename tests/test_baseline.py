"""
Tests for muons.baseline: shuffle_O_columns, load_O_for_baseline (Step 8).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from muons.baseline import load_O_for_baseline, shuffle_O_columns


def test_shuffle_dense_reproducible() -> None:
    """Same seed gives same shuffle; different seed gives different (shape unchanged)."""
    rng = np.random.default_rng(1)
    o_mat = rng.standard_normal((20, 5))
    o1 = shuffle_O_columns(o_mat, seed=42)
    o2 = shuffle_O_columns(o_mat, seed=42)
    o3 = shuffle_O_columns(o_mat, seed=99)
    assert o1.shape == o_mat.shape
    np.testing.assert_array_equal(o1, o2)
    assert not np.allclose(o1, o3)


def test_shuffle_dense_marginals_preserved() -> None:
    """Column-wise: each column has same values (permuted), so marginals preserved."""
    o_mat = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64)
    o_sh = shuffle_O_columns(o_mat, seed=0)
    assert o_sh.shape == o_mat.shape
    for j in range(o_mat.shape[1]):
        np.testing.assert_array_equal(np.sort(o_sh[:, j]), np.sort(o_mat[:, j]))


def test_shuffle_csr_shape_and_type() -> None:
    """CSR in -> CSR out; shape and nnz preserved."""
    n, d = 10, 4
    row = np.arange(n)
    col = np.arange(n) % d
    data = np.ones(n)
    o_mat = csr_matrix((data, (row, col)), shape=(n, d))
    o_sh = shuffle_O_columns(o_mat, seed=7)
    assert isinstance(o_sh, csr_matrix)
    assert o_sh.shape == o_mat.shape
    assert o_sh.nnz == o_mat.nnz


def test_load_O_for_baseline_quantile() -> None:
    """load_O_for_baseline(quantile) loads O_matrix.npz."""
    n, d = 30, 6
    o_mat = csr_matrix((np.ones(n), (np.arange(n), np.arange(n) % d)), shape=(n, d))
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        save_npz(out / "O_matrix.npz", o_mat)
        loaded = load_O_for_baseline(out, "quantile")
        assert isinstance(loaded, csr_matrix)
        np.testing.assert_allclose(loaded.toarray(), o_mat.toarray())


def test_load_O_for_baseline_zscore() -> None:
    """load_O_for_baseline(zscore) loads O_matrix.npy as dense array."""
    n, d = 20, 3
    o_mat = np.random.default_rng(11).standard_normal((n, d)).astype(np.float64)
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        np.save(out / "O_matrix.npy", o_mat)
        loaded = load_O_for_baseline(out, "zscore")
        assert isinstance(loaded, np.ndarray)
        np.testing.assert_allclose(loaded, o_mat)
