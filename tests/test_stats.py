"""
Tests for muons.stats: compute_branch_stats (Step 3).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot

from muons.io import open_root, select_tree
from muons.branches import select_branches
from muons.stats import compute_branch_stats


def test_compute_branch_stats_single_chunk() -> None:
    """Stats over small tree match numpy (one chunk)."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        n = 100
        x = np.linspace(0.0, 10.0, n)
        with uproot.create(path) as f:
            f["tree"] = {"a": x}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            branch_list = select_branches(tree, ["a"])
            rows = compute_branch_stats(tree, branch_list, chunk=200, max_events=0)
        assert len(rows) == 1
        r = rows[0]
        assert r["branch"] == "a"
        assert r["n"] == n
        np.testing.assert_allclose(r["min"], 0.0)
        np.testing.assert_allclose(r["max"], 10.0)
        np.testing.assert_allclose(r["mean"], np.mean(x))
        np.testing.assert_allclose(r["std"], np.std(x, ddof=1))
        np.testing.assert_allclose(r["nan_rate"], 0.0)
        np.testing.assert_allclose(r["median"], np.median(x))


def test_compute_branch_stats_chunked() -> None:
    """Stats over two chunks merge correctly."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        n = 500
        x = np.random.default_rng(42).uniform(0, 1, n)
        with uproot.create(path) as f:
            f["tree"] = {"b": x}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            rows = compute_branch_stats(tree, ["b"], chunk=200, max_events=0)
        assert len(rows) == 1
        r = rows[0]
        assert r["n"] == n
        np.testing.assert_allclose(r["min"], np.min(x))
        np.testing.assert_allclose(r["max"], np.max(x))
        np.testing.assert_allclose(r["mean"], np.mean(x), rtol=1e-10)
        np.testing.assert_allclose(r["std"], np.std(x, ddof=1), rtol=1e-9)


def test_compute_branch_stats_max_events() -> None:
    """max_events limits number of events processed."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        n = 100
        with uproot.create(path) as f:
            f["tree"] = {"c": np.arange(n, dtype=float)}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            rows = compute_branch_stats(tree, ["c"], chunk=50, max_events=30)
        assert len(rows) == 1
        assert rows[0]["n"] == 30
        np.testing.assert_allclose(rows[0]["mean"], 14.5)  # 0..29 mean


def test_compute_branch_stats_nan_rate() -> None:
    """nan_rate reflects fraction of non-finite values."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        x = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        with uproot.create(path) as f:
            f["tree"] = {"d": x}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            rows = compute_branch_stats(tree, ["d"], chunk=10)
        assert len(rows) == 1
        np.testing.assert_allclose(rows[0]["nan_rate"], 0.4)
        np.testing.assert_allclose(rows[0]["mean"], 2.333333, rtol=1e-5)
