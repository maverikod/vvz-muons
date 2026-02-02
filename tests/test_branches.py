"""
Tests for muons.branches: select_branches (Step 2).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import uproot

from muons.io import open_root, select_tree
from muons.branches import select_branches


def test_select_branches_from_config() -> None:
    """When config provides branches, they are validated and returned in order."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        with uproot.create(path) as f:
            f["tree"] = {
                "a": np.array([1.0, 2.0, 3.0]),
                "b": np.array([4.0, 5.0, 6.0]),
            }
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            out = select_branches(tree, ["b", "a"])
        assert out == ["b", "a"]


def test_select_branches_config_missing_raises() -> None:
    """When config branch is not in tree, ValueError is raised."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        with uproot.create(path) as f:
            f["tree"] = {"a": np.array([1.0, 2.0])}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            with pytest.raises(ValueError, match="Branch 'x' not found"):
                select_branches(tree, ["a", "x"])


def test_select_branches_auto() -> None:
    """Auto-selection returns scalar numeric branches with nan_rate<=0.2, std>0."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        n = 100
        with uproot.create(path) as f:
            f["tree"] = {
                "good": np.linspace(0.0, 1.0, n),
                "zero_std": np.ones(n),
                "high_nan": np.array([np.nan] * 50 + list(range(50)), dtype=float),
            }
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            out = select_branches(tree, None, max_scan=n)
        assert "good" in out
        assert "zero_std" not in out  # std > 0 filter
        assert "high_nan" not in out  # nan_rate <= 0.2 filter


def test_select_branches_empty_config_uses_auto() -> None:
    """Empty or None branches_config triggers auto-selection."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        with uproot.create(path) as f:
            f["tree"] = {"x": np.array([1.0, 2.0, 3.0])}
        with open_root(path) as file_handle:
            _, tree = select_tree(file_handle, "tree")
            out_none = select_branches(tree, None)
            out_empty = select_branches(tree, [])
        assert out_none == ["x"]
        assert out_empty == ["x"]
