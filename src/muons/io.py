"""
ROOT file and TTree I/O: open file, select tree (Step 1).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import uproot  # type: ignore[import-untyped]


def open_root(path: str | Path) -> uproot.ReadOnlyDirectory:
    """
    Open a ROOT file for reading. Does not load file contents into memory.

    Args:
        path: Path to the ROOT file (filesystem or URL).

    Returns:
        Opened file handle (ReadOnlyDirectory). Use as context manager to close.

    Raises:
        FileNotFoundError: If path does not exist (local files).
        OSError: If file is not a valid ROOT file or cannot be read.
    """
    path = Path(path) if isinstance(path, str) else path
    if not path.exists() and not str(path).startswith(("http://", "https://")):
        raise FileNotFoundError(f"ROOT file not found: {path}")
    return uproot.open(path)


def select_tree(file_handle: Any, tree_name: str | None) -> tuple[str, Any]:
    """
    Resolve which TTree to use: by name if given, else the one with max num_entries.

    Args:
        file_handle: Opened ROOT file (e.g. from open_root()).
        tree_name: Optional tree name from config. If set, must exist and be a TTree.

    Returns:
        (tree_name, tree) so downstream steps use tree for chunked iteration.
        tree is TTree or RNTuple (tree-like: has num_entries and arrays()).

    Raises:
        ValueError: If tree_name is given but not found or not a TTree;
            or if file has no TTrees.
    """

    def _is_tree_like(obj: Any) -> bool:
        """Accept TTree or RNTuple (uproot 5 creates RNTuple by default)."""
        return hasattr(obj, "num_entries") and (
            isinstance(obj, uproot.TTree) or hasattr(obj, "arrays")
        )

    def _logical_name(key: str) -> str:
        """Strip ROOT cycle suffix (e.g. 'tree;1' -> 'tree')."""
        return key.split(";")[0]

    trees: list[tuple[str, Any]] = []
    for key in file_handle.keys(recursive=False):
        obj = file_handle[key]
        if _is_tree_like(obj):
            trees.append((_logical_name(key), obj))

    if not trees:
        raise ValueError("No TTrees found in ROOT file")

    if tree_name is not None:
        for name, tree in trees:
            if name == tree_name:
                return (name, tree)
        raise ValueError(f"Tree '{tree_name}' not found in ROOT file")

    # Select TTree with maximum num_entries; tie-break: first in iteration order.
    best_name, best_tree = max(trees, key=lambda p: p[1].num_entries)
    return (best_name, best_tree)
