"""
Tests for muons.io: open_root, select_tree (Step 1).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import uproot

from muons.io import open_root, select_tree


def test_open_root_file_not_found() -> None:
    """open_root with nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="ROOT file not found"):
        open_root("/nonexistent/path/file.root")


def test_open_root_and_select_tree_max_entries() -> None:
    """select_tree with tree_name=None selects TTree with max num_entries."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        with uproot.create(path) as f:
            f["small"] = {"x": np.array([1.0, 2.0])}
            f["large"] = {"y": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        with open_root(path) as file_handle:
            name, tree = select_tree(file_handle, None)
        assert name == "large"
        assert tree.num_entries == 5


def test_select_tree_by_name() -> None:
    """select_tree with tree_name set returns that tree."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        with uproot.create(path) as f:
            f["small"] = {"x": np.array([1.0, 2.0])}
            f["large"] = {"y": np.array([1.0, 2.0, 3.0])}
        with open_root(path) as file_handle:
            name, tree = select_tree(file_handle, "small")
        assert name == "small"
        assert tree.num_entries == 2


def test_select_tree_not_found_raises() -> None:
    """select_tree with unknown tree_name raises ValueError."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.root"
        with uproot.create(path) as f:
            f["only"] = {"x": np.array([1.0])}
        with open_root(path) as file_handle:
            with pytest.raises(ValueError, match="Tree 'missing' not found"):
                select_tree(file_handle, "missing")
