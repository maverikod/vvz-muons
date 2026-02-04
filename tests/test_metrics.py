"""
Tests for muons.metrics: compute_metrics, metrics.json, spectrum.csv (Step 7).

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from muons.metrics import (
    compute_metrics,
    write_metrics_json,
    write_report_md,
    write_spectrum_csv,
)


def test_compute_metrics_basic() -> None:
    """compute_metrics returns dict with N_events, d, density_W, trace_L, Neff, PR_k."""
    d = 4
    L = np.eye(d) * 2 - np.ones((d, d)) * 0.2
    W = np.maximum(np.eye(d) * 0 + 0.2, 0.0)
    np.fill_diagonal(W, 0.0)
    eigenvalues = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    eigvec = np.eye(d).astype(np.float64)
    m = compute_metrics(
        L,
        W,
        eigenvalues,
        eigvec,
        N_events=100,
        features_count=4,
        d=d,
        mode="quantile",
        bins=16,
    )
    assert m["N_events"] == 100
    assert m["d"] == d
    assert m["features_count"] == 4
    assert m["mode"] == "quantile"
    assert m["bins"] == 16
    assert "density_W" in m and "trace_L" in m
    assert "lambda_min_nonzero" in m and "Neff" in m
    assert "PR_k" in m and len(m["PR_k"]) <= 10


def test_compute_metrics_all_zero_eigenvalues() -> None:
    """When no eigenvalue > 1e-12, lambda_min_nonzero and Neff are nan."""
    d = 2
    L = np.zeros((d, d))
    W = np.zeros((d, d))
    eigenvalues = np.array([0.0, 0.0], dtype=np.float64)
    eigvec = np.eye(d).astype(np.float64)
    m = compute_metrics(
        L,
        W,
        eigenvalues,
        eigvec,
        N_events=0,
        features_count=2,
        d=d,
        mode="zscore",
        bins=0,
    )
    assert np.isnan(m["lambda_min_nonzero"])
    assert np.isnan(m["Neff"])


def test_write_metrics_json() -> None:
    """write_metrics_json writes JSON; NaN becomes null."""
    metrics = {"N_events": 10, "Neff": float("nan"), "PR_k": [1.0, 2.0]}
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.json"
        write_metrics_json(metrics, path)
        assert path.exists()
        import json

        with open(path) as f:
            data = json.load(f)
        assert data["N_events"] == 10
        assert data["Neff"] is None
        assert data["PR_k"] == [1.0, 2.0]


def test_write_spectrum_csv() -> None:
    """write_spectrum_csv writes k, lambda_k, PR_k."""
    eigenvalues = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    eigvec = np.eye(3).astype(np.float64)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectrum.csv"
        write_spectrum_csv(eigenvalues, eigvec, path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert lines[0] == "k,lambda_k,PR_k"
        assert len(lines) == 4  # header + 3 rows


def test_write_report_md() -> None:
    """write_report_md writes header and baseline section when keys present."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.md"
        write_report_md(path, {"N_events": 5, "d": 3}, "tree1", 3, "quantile")
        assert path.exists()
        text = path.read_text()
        assert "Pipeline report" in text
        assert "tree1" in text
        write_report_md(
            path,
            {"baseline_Neff": 2.0, "delta_Neff": 1.0, "corr_fro_ratio": 1.5},
            "t", 1, "zscore",
        )
        assert "Baseline" in path.read_text()
