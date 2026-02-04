"""
Compute backend: prefer CUDA (CuPy) when GPU is available and memory is not over 80%.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Do not use GPU if used memory is at or above this fraction (0..1).
GPU_MEMORY_THRESHOLD = 0.80
# Safety factor: require free >= required_bytes * this (for temp allocations).
GPU_REQUIRED_MARGIN = 1.2

_cupy_available = False
_cupy_xp: Any = None

try:
    import cupy as cp  # type: ignore[import-untyped]

    # Avoid loading CUDA context on import; check device when first needed.
    if cp.cuda.is_available():
        _cupy_available = True
        _cupy_xp = cp
except Exception:
    pass


def get_gpu_memory_usage() -> Tuple[float, int, int] | None:
    """
    Return (used_fraction, total_bytes, free_bytes) for GPU 0, or None if unavailable.

    used_fraction is in [0, 1]. Requires CuPy and an active device context.
    """
    if not _cupy_available or _cupy_xp is None:
        return None
    try:
        _cupy_xp.cuda.Device(0).use()
        free_b, total_b = _cupy_xp.cuda.runtime.memGetInfo()
        if total_b <= 0:
            return None
        used_b = total_b - free_b
        used_frac = used_b / total_b
        return (used_frac, total_b, free_b)
    except Exception:
        return None


def get_backend(required_bytes: int | None = None) -> Tuple[Any, bool]:
    """
    Return (array module, use_gpu).

    Use GPU only when CuPy is available, device works, GPU memory used < 80%,
    and (if required_bytes is set) free memory >= required_bytes * GPU_REQUIRED_MARGIN.
    Otherwise use CPU. Logs the choice on every call.
    """
    if _cupy_available and _cupy_xp is not None:
        try:
            _cupy_xp.cuda.Device(0).use()
            mem = get_gpu_memory_usage()
            if mem is not None:
                used_frac, total_b, free_b = mem
                if used_frac >= GPU_MEMORY_THRESHOLD:
                    logger.warning(
                        "GPU memory usage %.0f%% (>= %.0f%%) — using CPU.",
                        used_frac * 100,
                        GPU_MEMORY_THRESHOLD * 100,
                    )
                    return np, False
                need_b = (
                    int(required_bytes * GPU_REQUIRED_MARGIN)
                    if required_bytes is not None and required_bytes > 0
                    else 0
                )
                if need_b > 0 and free_b < need_b:
                    logger.warning(
                        "GPU free memory %.0f MiB < required %.0f MiB — using CPU.",
                        free_b / (1024 * 1024),
                        need_b / (1024 * 1024),
                    )
                    return np, False
                logger.info(
                    "Compute backend: CUDA (CuPy) — GPU %.0f%% used, %.0f MiB free. Using GPU.",
                    used_frac * 100,
                    free_b / (1024 * 1024),
                )
                return _cupy_xp, True
            logger.warning("GPU memory check failed — using CPU.")
            return np, False
        except Exception as e:
            logger.warning("CuPy import ok but CUDA device failed: %s — falling back to CPU.", e)
            return np, False
    logger.info(
        "Compute backend: CPU (NumPy/SciPy) — CuPy not available or CUDA device failed. "
        "Install cupy-cuda12x (or matching CUDA) for GPU acceleration."
    )
    return np, False


def to_numpy(xp: Any, arr: Any) -> np.ndarray:
    """Convert array to numpy on host. No-op if already numpy."""
    if xp is np or arr is None:
        return np.asarray(arr, dtype=np.float64)
    if hasattr(xp, "asnumpy"):
        return np.asarray(xp.asnumpy(arr), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)
