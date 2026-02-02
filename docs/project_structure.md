# Project structure

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Directory layout and purpose of main folders.

---

## Root layout

```
muons/
├── docs/           # Documentation
├── data/           # All data (input and output)
│   ├── in/         # Input data (e.g. ROOT files)
│   └── out/        # Output data (pipeline results)
├── src/
│   └── muons/      # Package source
├── tests/          # Tests
├── pyproject.toml
├── README.md
└── ...
```

---

## Directories

| Directory   | Purpose |
|------------|---------|
| **docs**   | Project documentation: technical spec, analysis, structure, accents, rules. |
| **data**   | Data root. Do not commit large or generated files; use `.gitignore` for `*.root`, `out/`, etc. |
| **data/in**  | Input data only (e.g. ROOT files). CLI `--input` may point here or elsewhere. |
| **data/out** | Default or recommended output directory for pipeline results (manifest, CSV, NPZ, report). All outputs from section 2 of the techspec go under a single output dir; typically `data/out` or a path passed via `--out`. |

---

## Output directory contents (under `data/out` or `--out`)

As per techspec, the pipeline writes:

- **Manifests:** `manifest.json`, `features_used.json`
- **Tables:** `branch_stats.csv`, `bin_definitions.csv` (if mode=quantile)
- **Matrices:** `O_matrix.npz` or `O_matrix.npy` + `zscore_params.json`, `corr.npz`, `laplacian.npz`
- **Metrics/report:** `metrics.json`, `spectrum.csv`, `report.md`

Input files stay in `data/in` (or any path given by `--input`).
