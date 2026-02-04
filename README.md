# muons-processor

Streaming ROOT file processor: chunked pipeline from TTree to correlation matrix, Laplacian and spectrum. No full load into memory (suitable for 2–16 GB RAM).

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

## Requirements

- Python 3.10+
- ROOT file (path to your `.root` file; not included in the repo)

## Clone and install (from Git)

```bash
git clone <repository-url> muons-processor
cd muons-processor
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
# optional GPU support (CUDA 12.x):
pip install -e ".[cuda]"
```

## Install (from PyPI, when published)

```bash
pip install muons-processor
```

**GPU (CUDA) on Ubuntu:** use CuPy for correlation and Laplacian (CPU only when no GPU is present):

```bash
pip install "muons-processor[cuda]"
```

Requires NVIDIA driver and CUDA 12.x (e.g. `cupy-cuda12x`). If CuPy is not installed, the pipeline runs on CPU.

From source:

```bash
pip install -e .
# with GPU support:
pip install -e ".[cuda]"
```

## Usage (CLI)

**Recommended run** (jagged branches + baseline; config is in the repo):

```bash
python -m muons.cli --input "/path/to/Run2012BC_DoubleMuParked_Muons.root" --config config_jagged.yaml --out data/out
```

Replace `/path/to/Run2012BC_DoubleMuParked_Muons.root` with your ROOT file path. The file `config_jagged.yaml` is part of the repository (quantile mode, baseline, full jagged aggregates; see `docs/addontspc.md`).

Minimal run without config:

```bash
python -m muons.cli --input /path/to/your/file.root --out data/out
```

Main options: `--mode quantile|zscore`, `--bins`, `--chunk`, `--max-events`, `--tau`, `--topk`, `--k-eigs`, `--baseline`. Config can set `tree`, `branches`, `allow_jagged`, `jagged_aggs`, etc. (see `docs/techspec.md` and `docs/addontspc.md`).

## Output

Each run writes to a **timestamped subdirectory** `data/out/YYYY-MM-DDThh_mm_ss/` (or `--out` base path). No overwriting between runs.

- `run_parameters.json`, `manifest.json`, `features_used.json`
- `branch_stats.csv`, (optional) `bin_definitions.csv`
- `O_matrix.npz` or `O_matrix.npy` + `zscore_params.json`
- `corr.npz`, `laplacian.npz`
- `metrics.json`, `spectrum.csv`, `report.md`

With `allow_jagged: true` in config: `derived_features.json` as well.

## Project layout (Git-friendly)

- `src/muons/` — package source
- `tests/` — tests
- `docs/` — technical spec, plan, addendum (jagged)
- `data/in/` — placeholder for input ROOT files (not committed)
- `data/out/` — pipeline output root; run dirs are timestamped (not committed)
- `config_jagged.yaml` — **tracked in Git**; config for jagged branches and full aggregates (used by the recommended CLI command above).

**.gitignore** excludes: `.venv/`, `data/out/*`, `*.root`, `out_MAX*/`, `runs_summary.csv`, `final_conclusions.md`, `code_analysis/`, `.cursor/`, build and cache dirs. Only code, docs, config examples, and `.gitkeep` under `data/` are tracked.

## Git

- **Commit template:** `git config commit.template .gitmessage` (from repo root).
- **Publish:** no secrets or machine-specific paths in repo; put your ROOT file path in CLI or local config only.

## License

MIT
