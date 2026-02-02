# muons-processor

Streaming ROOT file processor: chunked pipeline from TTree to correlation matrix, Laplacian and spectrum. No full load into memory (suitable for 2–16 GB RAM).

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

## Requirements

- Python 3.10+
- ROOT file (e.g. `Run2012BC_DoubleMuParked_Muons.root`)

## Install

```bash
pip install muons-processor
```

From source:

```bash
pip install -e .
```

## Usage (CLI)

```bash
process_root \
  --input /path/to/file.root \
  --out out \
  --mode quantile \
  --bins 16 \
  --chunk 200000 \
  --threshold-tau 0.1 \
  --topk 0 \
  --k-eigs 200
```

Optional config via YAML: `tree`, `branches`, `mode`, `bins`, `chunk`, `max_events`, `tau`, `topk`, `k_eigs`, `baseline`, `seed`.

## Output (in `out/`)

- `manifest.json`, `features_used.json`
- `branch_stats.csv`, (optional) `bin_definitions.csv`
- `O_matrix.npz` or `O_matrix.npy` + `zscore_params.json`
- `corr.npz`, `laplacian.npz`
- `metrics.json`, `spectrum.csv`, `report.md`

## Git

- **Commit template:** `git config commit.template .gitmessage` (use from repo root).
- **Ignore:** `.gitignore` and `.gitattributes` are set for Python, build, and data files.

## License

MIT
