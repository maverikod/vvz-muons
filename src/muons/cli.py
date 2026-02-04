"""
CLI entry point for process_root: streaming ROOT pipeline.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from muons.baseline import load_O_for_baseline, shuffle_O_columns
from muons.branches import select_branches, select_features
from muons.config_loader import load_config
from muons.correlation import (
    build_W,
    compute_correlation,
    compute_correlation_from_files,
    save_corr_npz,
)
from muons.io import open_root, select_tree
from muons.laplacian import compute_spectrum, save_laplacian_npz
from muons.metrics import (
    compute_metrics,
    write_metrics_json,
    write_report_md,
    write_spectrum_csv,
)
from muons.observables import build_quantile_O, build_zscore_O
from muons.stats import compute_branch_stats
from muons.manifest import write_manifest_json
from muons.backend import get_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for process_root CLI (see docs/techspec.md)."""
    parser = argparse.ArgumentParser(
        description="Streaming ROOT: TTree -> correlation/Laplacian/spectrum.",
    )
    parser.add_argument("--input", required=True, help="Path to input ROOT file")
    parser.add_argument(
        "--out",
        default="data/out",
        help="Output base path; each run creates a subdir YYYY-MM-DDThh_mm_ss (default: data/out)",
    )
    parser.add_argument("--config", help="Optional YAML config path")
    parser.add_argument("--tree", help="TTree name (overrides config)")
    parser.add_argument("--mode", choices=("quantile", "zscore"), default="quantile")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=200_000)
    parser.add_argument("--threshold-tau", "--tau", type=float, default=0.1, dest="tau")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--k-eigs", type=int, default=200)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Each run writes to its own directory (techspec §2, project_structure): no overwriting.
    run_start = datetime.now()
    timestamp_dir = run_start.strftime("%Y-%m-%dT%H_%M_%S")
    out_base = Path(args.out)
    out_path = out_base / timestamp_dir
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Run started at %s — output: %s", run_start.isoformat(), out_path)

    config = load_config(args.config) if args.config else {}
    tree_name_cfg = config.get("tree")
    tree_name: str | None = args.tree if args.tree is not None else tree_name_cfg

    _write_run_parameters(out_path, args, config, run_start)

    # Resolve backend once so user sees GPU/CPU message at start (no required_bytes yet).
    xp_backend, use_gpu = get_backend()
    if use_gpu:
        logger.info("Heavy linear algebra (C, W, L, eig) will run on GPU (CuPy).")
    else:
        logger.info("Heavy linear algebra will run on CPU (NumPy/SciPy).")

    t0 = time.perf_counter()

    use_jagged = config.get("allow_jagged", False)

    logger.info("Opening ROOT file: %s", args.input)
    with open_root(args.input) as file_handle:
        selected_name, tree = select_tree(file_handle, tree_name)
        logger.info("Tree: %s", selected_name)
        if use_jagged:
            feature_names, scalar_branches, jagged_specs = select_features(tree, config)
            branch_list = feature_names
            logger.info("Features (jagged): %d", len(branch_list))
        else:
            branch_list = select_branches(tree, config.get("branches"))
            scalar_branches = None
            jagged_specs = None
            logger.info("Branches: %d", len(branch_list))

        features_path = out_path / "features_used.json"
        with open(features_path, "w") as f:
            json.dump({"tree": selected_name, "branches": branch_list}, f, indent=2)

        if jagged_specs:
            derived_path = out_path / "derived_features.json"
            with open(derived_path, "w") as f:
                json.dump(
                    {
                        "jagged_branches": [b for b, _ in jagged_specs],
                        "jagged_aggs": list({a for _, aggs in jagged_specs for a in aggs}),
                        "feature_names": branch_list,
                    },
                    f,
                    indent=2,
                )

        logger.info("Computing branch stats (chunk=%d)...", args.chunk)
        branch_stats_list = compute_branch_stats(
            tree,
            branch_list,
            chunk=args.chunk,
            max_events=args.max_events or 0,
            scalar_branches=scalar_branches,
            jagged_specs=jagged_specs,
        )
        logger.info("Branch stats done (%d branches).", len(branch_stats_list))

        branch_stats_path = out_path / "branch_stats.csv"
        _write_branch_stats_csv(branch_stats_path, branch_stats_list)

        if args.mode == "quantile":
            logger.info("Building O matrix (quantile, bins=%d)...", args.bins)
            _bin_def_path, _o_npz_path = build_quantile_O(
                tree,
                branch_list,
                branch_stats_list,
                out_path,
                bins=args.bins,
                chunk=args.chunk,
                max_events=args.max_events or 0,
                scalar_branches=scalar_branches,
                jagged_specs=jagged_specs,
            )
            logger.info("O matrix (quantile) written.")
        else:
            logger.info("Building O matrix (zscore)...")
            _zscore_json_path, _o_npy_path = build_zscore_O(
                tree,
                branch_list,
                branch_stats_list,
                out_path,
                chunk=args.chunk,
                max_events=args.max_events or 0,
                scalar_branches=scalar_branches,
                jagged_specs=jagged_specs,
            )
            logger.info("O matrix (zscore) written.")

    logger.info("Computing correlation matrix C...")
    C = compute_correlation_from_files(out_path, args.mode, branch_list, chunk=args.chunk)
    logger.info("Building W and saving corr.npz...")
    W = build_W(C, tau=args.tau, topk=args.topk)
    save_corr_npz(C, W, out_path / "corr.npz")

    logger.info("Computing Laplacian and spectrum (k_eigs=%d)...", args.k_eigs)
    L, eigenvalues, eigvec_first10 = compute_spectrum(W, k_eigs=args.k_eigs)
    save_laplacian_npz(L, eigenvalues, eigvec_first10, out_path / "laplacian.npz")

    d = W.shape[0]
    N_events = branch_stats_list[0]["n"] if branch_stats_list else 0
    metrics_dict = compute_metrics(
        L,
        W,
        eigenvalues,
        eigvec_first10,
        N_events=N_events,
        features_count=len(branch_list),
        d=d,
        mode=args.mode,
        bins=args.bins,
    )
    if args.baseline:
        logger.info("Baseline: shuffling O and recomputing C, W, spectrum...")
        O_baseline = load_O_for_baseline(out_path, args.mode)
        O_shuffled = shuffle_O_columns(O_baseline, args.seed)
        C0 = compute_correlation(O_shuffled)
        W0 = build_W(C0, tau=args.tau, topk=args.topk)
        L0, eigenvalues0, eigvec0 = compute_spectrum(W0, k_eigs=args.k_eigs)
        metrics0 = compute_metrics(
            L0,
            W0,
            eigenvalues0,
            eigvec0,
            N_events=N_events,
            features_count=len(branch_list),
            d=d,
            mode=args.mode,
            bins=args.bins,
        )
        baseline_Neff = metrics0["Neff"]
        delta_Neff = (
            metrics_dict["Neff"] - baseline_Neff
            if not np.isnan(metrics_dict["Neff"])
            else float("nan")
        )
        fro_C = np.sqrt(np.sum(C * C))
        fro_C0 = np.sqrt(np.sum(C0 * C0))
        corr_fro_ratio = float(fro_C / fro_C0) if fro_C0 > 0 else float("nan")
        metrics_dict["baseline_Neff"] = baseline_Neff
        metrics_dict["delta_Neff"] = delta_Neff
        metrics_dict["corr_fro_ratio"] = corr_fro_ratio

    logger.info("Writing metrics, spectrum, report...")
    write_metrics_json(metrics_dict, out_path / "metrics.json")
    write_spectrum_csv(eigenvalues, eigvec_first10, out_path / "spectrum.csv")
    write_report_md(
        out_path / "report.md",
        metrics_dict,
        selected_name,
        len(branch_list),
        args.mode,
    )

    runtime_seconds = time.perf_counter() - t0
    logger.info("Writing manifest (runtime %.1f s).", runtime_seconds)
    effective_params = {
        "tree": selected_name,
        "branches": branch_list,
        "chunk": args.chunk,
        "mode": args.mode,
        "bins": args.bins,
        "max_events": args.max_events or 0,
        "tau": args.tau,
        "topk": args.topk,
        "k_eigs": args.k_eigs,
        "baseline": args.baseline,
        "seed": args.seed,
    }
    write_manifest_json(
        out_path / "manifest.json",
        args.input,
        effective_params,
        runtime_seconds,
    )

    msg = (
        f"Steps 2–8 done. Tree: {selected_name}, features: {len(branch_list)}, "
        f"mode={args.mode}. Wrote manifest.json, features_used, branch_stats, O matrix, "
        "corr.npz, laplacian.npz, metrics.json, spectrum.csv, report.md."
    )
    if jagged_specs:
        msg += " derived_features.json (jagged aggregates)."
    if args.baseline:
        msg += " Baseline: baseline_Neff, delta_Neff, corr_fro_ratio in metrics and report."
    logger.info("Done. %s", msg)
    sys.exit(0)


def _write_run_parameters(
    out_path: Path,
    args: argparse.Namespace,
    config: dict,
    run_start: datetime,
) -> None:
    """Write run_parameters.json with start time, argv, CLI args, and config."""
    params: dict = {
        "started_at": run_start.isoformat(),
        "argv": list(sys.argv),
        "input": args.input,
        "out_base": str(args.out),
        "out_run_dir": str(out_path),
        "tree": getattr(args, "tree", None),
        "config_path": args.config,
        "mode": args.mode,
        "bins": args.bins,
        "chunk": args.chunk,
        "tau": args.tau,
        "topk": args.topk,
        "k_eigs": args.k_eigs,
        "max_events": args.max_events or 0,
        "baseline": args.baseline,
        "seed": args.seed,
    }
    # Config may contain non-JSON values; keep only JSON-serializable
    config_ser: dict = {}
    for k, v in config.items():
        try:
            json.dumps(v)
            config_ser[k] = v
        except (TypeError, ValueError):
            config_ser[k] = str(v)
    params["config"] = config_ser
    with open(out_path / "run_parameters.json", "w") as f:
        json.dump(params, f, indent=2)


def _write_branch_stats_csv(path: Path, rows: list[dict]) -> None:
    """Write branch_stats.csv from list of dicts (branch, min, max, mean, std, etc.)."""
    if not rows:
        return
    fieldnames = ["branch", "min", "max", "mean", "std", "nan_rate", "median", "n"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
