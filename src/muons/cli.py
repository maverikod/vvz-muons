"""
CLI entry point for process_root: streaming ROOT pipeline.

Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com
"""

import argparse
import sys

from muons.config_loader import load_config
from muons.io import open_root, select_tree


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

    with open_root(args.input) as file_handle:
        selected_name, _tree = select_tree(file_handle, tree_name)
        # Step 1 done. Pipeline steps 2–8 not yet implemented.
        print(
            f"Not implemented. Input: {args.input}, out: {args.out}, tree: {selected_name}. "
            "See docs/techspec.md",
            file=sys.stderr,
        )
    sys.exit(1)


if __name__ == "__main__":
    main()
