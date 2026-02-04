"""
Manifest: SHA256 of input ROOT, library versions, effective params, runtime (techspec §2).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def input_sha256(path: str | Path, chunk_size: int = 65536) -> str:
    """
    Compute SHA256 of input file in chunks (no full load).

    Args:
        path: Path to ROOT file (or any file).
        chunk_size: Read chunk size in bytes.

    Returns:
        Hex digest of SHA256.
    """
    path = Path(path)
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_library_versions() -> dict[str, str]:
    """Return dict of library name -> version string for techspec manifest."""
    result: dict[str, str] = {}
    libs = [
        ("uproot", "uproot"),
        ("awkward", "awkward"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
        ("cupy", "cupy"),
    ]
    for attr, mod_name in libs:
        try:
            mod = __import__(mod_name)
            result[attr] = getattr(mod, "__version__", "?")
        except Exception:
            result[attr] = "?"
    return result


def write_manifest_json(
    out_path: Path,
    input_root: str | Path,
    effective_params: dict[str, Any],
    runtime_seconds: float,
) -> None:
    """
    Write manifest.json: SHA256 of input ROOT, datetime, params, library versions, runtime.

    Args:
        out_path: Path to manifest.json (file path, not directory).
        input_root: Path to input ROOT file (for SHA256).
        effective_params: Dict with tree, branches (list or count), chunk, mode, bins, etc.
        runtime_seconds: Pipeline runtime in seconds.
    """
    sha = input_sha256(input_root)
    lib_versions = get_library_versions()
    # Keep branches as count in manifest to avoid huge JSON
    params = dict(effective_params)
    if "branches" in params and isinstance(params["branches"], list):
        params["branch_count"] = len(params["branches"])
        params.pop("branches", None)

    payload: dict[str, Any] = {
        "input_sha256": sha,
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "effective_params": params,
        "library_versions": lib_versions,
        "runtime_seconds": round(runtime_seconds, 3),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
