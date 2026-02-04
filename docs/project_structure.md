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
| **data/out** | Base path for pipeline results (default or `--out`). **Each run writes to its own subdirectory** named by run start time: `data/out/YYYY-MM-DDThh_mm_ss/`. No overwriting between runs. |

---

## Output directory contents (one run = one timestamped subdir)

Each run creates `<out_base>/YYYY-MM-DDThh_mm_ss/`. As per techspec §2, the pipeline writes there:

- **Run params:** `run_parameters.json` (argv, config, start time)
- **Manifests:** `manifest.json`, `features_used.json`
- **Tables:** `branch_stats.csv`, `bin_definitions.csv` (if mode=quantile)
- **Matrices:** `O_matrix.npz` or `O_matrix.npy` + `zscore_params.json`, `corr.npz`, `laplacian.npz`
- **Metrics/report:** `metrics.json`, `spectrum.csv`, `report.md`
- **Jagged (addontspc):** `derived_features.json` when `allow_jagged: true`

Input files stay in `data/in` (or any path given by `--input`).
