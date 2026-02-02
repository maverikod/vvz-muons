# Step-by-step verification (implementation vs techspec & plan)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Summary of pipeline step verification. **Full detail for each step** (detailed algorithm, data types, file formats, edge cases, accents, implementation status) is in the **individual step files** under [docs/plan/](plan/).

---

## Pre-conditions (techspec §0, §1)

| Item | Requirement | Status |
|------|-------------|--------|
| No full ROOT load | Never load entire ROOT file into memory | ✅ Chunked reads only (tree.arrays with entry_start/stop) |
| Chunked only | All processing streaming; chunk size configurable (default 200000) | ✅ Chunk loop in stats, observables, correlation (zscore) |
| Numerical output only | CSV/NPZ/JSON + short report; no physics text, no synthetic data, no plots by default | ✅ Step 1–8 outputs + report.md |
| Input | `input_root` required; YAML optional (tree, branches, mode, bins, chunk, max_events, tau, topk, k_eigs, baseline, seed) | ✅ CLI has `--input`, `--config`, `--tree`; config for tree, branches |
| RAM | 2–16 GB without OOM | ❌ Not testable |

**Current code:** `src/muons/cli.py` (argparse, config load, Step 1–8, features_used.json, branch_stats.csv, O matrix, corr.npz, laplacian.npz, metrics.json, spectrum.csv, report.md; baseline when --baseline), `muons.io`, `muons.config_loader`, `muons.branches`, `muons.stats`, `muons.observables`, `muons.correlation`, `muons.laplacian`, `muons.metrics`, `muons.baseline`.

---

## Steps (detail in plan files)

| Step | Plan file | Goal | Implemented? |
|------|-----------|------|--------------|
|  | [step_01_open_root_select_tree.md](plan/step_01_open_root_select_tree.md) | Open ROOT, select TTree (uproot.open; config tree or max num_entries) | ✅ Yes |
|  | [step_02_select_branches.md](plan/step_02_select_branches.md) | Select branches: config or auto (20k scan; scalar, numeric, nan_rate≤0.2, std>0; cap 64); features_used.json | ✅ Yes |
|  | [step_03_first_pass_stats.md](plan/step_03_first_pass_stats.md) | Chunked stats (min/max/mean/std/nan_rate), median from 200k; branch_stats.csv | ✅ Yes |
|  | [step_04_build_O_matrix.md](plan/step_04_build_O_matrix.md) | Build O: quantile (bin edges, one-hot CSR, bin_definitions, O_matrix.npz) or zscore (memmap, zscore_params, O_matrix.npy) | ✅ Yes |
|  | [step_05_correlation_W.md](plan/step_05_correlation_W.md) | C (Pearson; O.T@O/N for sparse), W=max(0,C) diag=0, topk/tau; corr.npz | ✅ Yes |
|  | [step_06_laplacian_spectrum.md](plan/step_06_laplacian_spectrum.md) | L=D−W, eigenvalues (eigh d≤500 / eigsh k_eigs), laplacian.npz (L, lambda, eigvec_first10) | ✅ Yes |
|  | [step_07_metrics.md](plan/step_07_metrics.md) | Metrics (N_events, d, density_W, trace_L, lambda_min_nonzero, Neff, PR_k); metrics.json, spectrum.csv | ✅ Yes |
|  | [step_08_baseline.md](plan/step_08_baseline.md) | baseline: column-shuffle O, repeat 5–7; baseline_Neff, delta_Neff, corr_fro_ratio in metrics and report | ✅ Yes |

Each plan file contains: **Detailed algorithm** (sub-steps, formulas), **data types and shapes**, **file format** (columns/keys), **edge cases and validation**, **accents** (vectorization, CUDA, parallelization), **implementation status** table, **suggested module**, and **step completion checklist**.

---

## Cross-step: Manifest and report

| Item | Requirement | Status |
|------|-------------|--------|
| **manifest.json** | SHA256 of input ROOT, date/time, effective params, library versions, runtime | ❌ No code |
| **report.md** | Short human-readable summary; when baseline=true include baseline_Neff, delta_Neff, corr_fro_ratio | ❌ No code |

**Suggested module:** `muons.manifest` (SHA256, versions, write_manifest_json). Report: from metrics dict in cli or muons.metrics.

---

## Summary

- **CLI:** Argparse, YAML config load, `--tree`, `--out` (default data/out), `--config`, `--baseline`; pipeline runs Step 1–8.
- **Steps 1–8:** Implemented (see table above). Full specification and detail are in [docs/plan/](plan/) per step.
- **report.md:** Written (short summary; baseline section when --baseline). **manifest.json:** Not implemented.
