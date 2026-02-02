"""
CLI entry point for process_root: streaming ROOT pipeline.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from muons.branches import select_branches
from muons.config_loader import load_config
from muons.correlation import build_W, compute_correlation_from_files, save_corr_npz
from muons.io import open_root, select_tree
from muons.laplacian import compute_spectrum, save_laplacian_npz
from muons.observables import build_quantile_O, build_zscore_O
from muons.stats import compute_branch_stats


def main() -> None:
    """Entry point for process_root CLI (see docs/techspec.md)."""
    parser = argparse.ArgumentParser(
        description="Streaming ROOT: TTree -> correlation/Laplacian/spectrum.",
    )
    parser.add_argument("--input", required=True, help="Path to input ROOT file")
    parser.add_argument("--out", default="out", help="Output directory")
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

    config = load_config(args.config) if args.config else {}
    tree_name_cfg = config.get("tree")
    tree_name: str | None = args.tree if args.tree is not None else tree_name_cfg

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    with open_root(args.input) as file_handle:
        selected_name, tree = select_tree(file_handle, tree_name)
        branch_list = select_branches(tree, config.get("branches"))

        features_path = out_path / "features_used.json"
        with open(features_path, "w") as f:
            json.dump({"tree": selected_name, "branches": branch_list}, f, indent=2)

        branch_stats_list = compute_branch_stats(
            tree, branch_list, chunk=args.chunk, max_events=args.max_events or 0
        )

    branch_stats_path = out_path / "branch_stats.csv"
    _write_branch_stats_csv(branch_stats_path, branch_stats_list)

    if args.mode == "quantile":
        _bin_def_path, _o_npz_path = build_quantile_O(
            tree,
            branch_list,
            branch_stats_list,
            out_path,
            bins=args.bins,
            chunk=args.chunk,
            max_events=args.max_events or 0,
        )
    else:
        _zscore_json_path, _o_npy_path = build_zscore_O(
            tree,
            branch_list,
            branch_stats_list,
            out_path,
            chunk=args.chunk,
            max_events=args.max_events or 0,
        )

    C = compute_correlation_from_files(out_path, args.mode, branch_list, chunk=args.chunk)
    W = build_W(C, tau=args.tau, topk=args.topk)
    save_corr_npz(C, W, out_path / "corr.npz")

    L, eigenvalues, eigvec_first10 = compute_spectrum(W, k_eigs=args.k_eigs)
    save_laplacian_npz(L, eigenvalues, eigvec_first10, out_path / "laplacian.npz")

    msg = (
        f"Steps 2–6 done. Tree: {selected_name}, branches: {len(branch_list)}, "
        f"mode={args.mode}. Wrote {features_path}, {branch_stats_path}, O matrix, "
        "corr.npz, laplacian.npz. Steps 7–8 not implemented."
    )
    print(msg, file=sys.stderr)
    sys.exit(0)


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
