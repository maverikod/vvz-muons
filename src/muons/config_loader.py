"""
Load optional YAML config (tree, branches, mode, bins, chunk, etc.).
Jagged-branch options follow docs/addontspc.md.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_config(path: str | Path | None) -> dict[str, Any]:
    """
    Load pipeline config from a YAML file.

    Args:
        path: Path to YAML file, or None for empty config.

    Returns:
        Dict with keys: tree, branches, mode, bins, chunk, max_events,
        tau, topk, k_eigs, baseline, seed. Addontspc (jagged) keys when used:
        allow_jagged, jagged_branches, jagged_aggs, max_jagged_branches,
        max_scalar_branches, max_scan (see docs/addontspc.md).
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return dict(data) if data else {}
