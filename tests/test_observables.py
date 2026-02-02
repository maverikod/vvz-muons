"""
Tests for muons.observables: build_quantile_O, build_zscore_O (Step 4).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot
from scipy.sparse import load_npz

from muons.io import open_root, select_tree
from muons.branches import select_branches
from muons.stats import compute_branch_stats
from muons.observables import build_quantile_O, build_zscore_O


def test_build_quantile_O_shape_and_files() -> None:
    """Quantile O: bin_definitions.csv and O_matrix.npz written; shape (n, d)."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        out = Path(tmp) / "out"
        out.mkdir()
        n = 200
        x = np.linspace(0.0, 10.0, n)
        with uproot.create(path) as f:
            f["tree"] = {"a": x}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            branch_list = select_branches(tree, ["a"])
            stats = compute_branch_stats(tree, branch_list, chunk=100)
        bin_def_path, npz_path = build_quantile_O(tree, branch_list, stats, out, bins=4, chunk=50)
        assert bin_def_path.exists()
        assert npz_path.exists()
        o_mat = load_npz(npz_path)
        assert o_mat.shape == (n, 4)
        assert o_mat.format == "csr"
        with open(bin_def_path) as f:
            lines = f.readlines()
        assert lines[0].strip() == "branch,bin_id,left_edge,right_edge"
        assert len(lines) == 1 + 4


def test_build_zscore_O_shape_and_files() -> None:
    """Zscore O: zscore_params.json and O_matrix.npy written; shape (n, d)."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        out = Path(tmp) / "out"
        out.mkdir()
        n = 150
        x = np.random.default_rng(1).standard_normal(n)
        with uproot.create(path) as f:
            f["tree"] = {"b": x}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            branch_list = select_branches(tree, ["b"])
            stats = compute_branch_stats(tree, branch_list, chunk=80)
        json_path, npy_path = build_zscore_O(tree, branch_list, stats, out, chunk=50)
        assert json_path.exists()
        assert npy_path.exists()
        o_mat = np.memmap(npy_path, dtype=np.float64, mode="r", shape=(n, 1))
        o_np = np.asarray(o_mat)
        np.testing.assert_allclose(o_np.mean(axis=0), 0.0, atol=0.1)
        np.testing.assert_allclose(o_np.std(axis=0), 1.0, atol=0.1)


def test_build_quantile_O_empty_tree() -> None:
    """Quantile O with max_events=0 and empty tree writes (0, d) matrix."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.root"
        out = Path(tmp) / "out"
        out.mkdir()
        with uproot.create(path) as f:
            f["tree"] = {"a": np.array([], dtype=float)}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            stats = [
                {
                    "branch": "a",
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "nan_rate": 0.0,
                    "median": np.nan,
                    "n": 0,
                }
            ]
        bin_def_path, npz_path = build_quantile_O(tree, ["a"], stats, out, bins=4)
        o_mat = load_npz(npz_path)
        assert o_mat.shape == (0, 4)


def test_build_zscore_O_max_events() -> None:
    """Zscore O respects max_events."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        out = Path(tmp) / "out"
        out.mkdir()
        n = 100
        x = np.arange(n, dtype=float)
        with uproot.create(path) as f:
            f["tree"] = {"c": x}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            branch_list = select_branches(tree, ["c"])
            stats = compute_branch_stats(tree, branch_list, max_events=40)
        _, npy_path = build_zscore_O(tree, branch_list, stats, out, chunk=20, max_events=40)
        o_mat = np.memmap(npy_path, dtype=np.float64, mode="r", shape=(40, 1))
        assert o_mat.shape == (40, 1)
