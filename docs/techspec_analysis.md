# Technical specification analysis (techspec.md)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Analysis of `docs/techspec.md`: requirements, algorithm, outputs, and implementation mapping.

---

## 1. Purpose and constraints

| Item | Description |
|------|-------------|
| **Goal** | Process large ROOT file (`Run2012BC_DoubleMuParked_Muons.root`, ~2 GB) without loading it fully into memory. |
| **Method** | Chunked (streaming) only. No assumptions about physics meaning of branches. |
| **Output** | Strictly numerical: CSV/NPZ/JSON + short report. |
| **Hard rule** | Must not read entire ROOT into memory. |

---

## 2. Inputs

| Input | Type | Required | Notes |
|-------|------|----------|--------|
| `input_root` | path | Yes | Path to ROOT file. |
| YAML config | file | No | Optional overrides. |

**YAML options:** `tree`, `branches`, `mode` (quantile/zscore), `bins` (default 16), `chunk` (default 200000), `max_events` (0 = all), `tau` (default 0.1), `topk` (0 = off), `k_eigs`, `baseline` (bool), `seed` (int).

---

## 3. Outputs (directory `out/`)

### A) Manifests / metadata

| File | Content |
|------|---------|
| `manifest.json` | SHA256 of input ROOT, datetime, effective params (tree, branches, chunk, mode…), library versions, runtime. |
| `features_used.json` | Chosen tree + ordered list of branches. |

### B) Tables

| File | Content |
|------|---------|
| `branch_stats.csv` | Per branch: min, max, mean, std, nan_rate, median, n. |
| `bin_definitions.csv` | Only if `mode=quantile`: per branch: bin_id, left_edge, right_edge. |

### C) Matrices and spectrum

| File | When | Content |
|------|------|---------|
| `O_matrix.npz` | mode=quantile | Sparse CSR: events × observables (one-hot). |
| `O_matrix.npy` + `zscore_params.json` | mode=zscore | Dense (memmap/ndarray) + mean/std/median. |
| `corr.npz` | Always | `C` (correlation), `W` (connectivity). |
| `laplacian.npz` | Always | `L` (Laplacian), `lambda` (eigenvalues sorted), `eigvec_first10`. |

### D) Metrics and report

| File | Content |
|------|---------|
| `metrics.json` | N_events, features_count, mode, bins, d, density_W, trace_L, lambda_min_nonzero, Neff, PR_k (first 10), baseline fields if baseline=true. |
| `spectrum.csv` | k, lambda_k, PR_k (if available). |
| `report.md` | Short human-readable summary. |

---

## 4. Algorithm (8 steps) — implementation mapping

| Step | Description | Implementation notes |
|------|-------------|------------------------|
| **1** | Open ROOT via uproot; choose tree (config or max num_entries). | `uproot.open()`; iterate keys, pick by `num_entries`. |
| **2** | Choose branches: from config or auto (scan first 20k events; scalar, numeric, nan_rate ≤ 0.2, std > 0; cap 64). Write `features_used.json`. | Branch scan chunked; filter dtypes; compute nan_rate, std. |
| **3** | First pass (chunked): per-branch min/max/mean/std/nan; median from first 200k for imputation. Write `branch_stats.csv`. | Online or chunked stats; optional sample for median. |
| **4** | Build observable matrix O: **A** quantile bins (200k sample → bin edges; NaN→median; one-hot) → CSR; **B** zscore (NaN→median; z=(x-mean)/std) → memmap. Write bin_definitions / zscore_params + O. | Two code paths; chunked write for O. |
| **5** | Correlation C (Pearson on columns of O); W = max(0,C), diag=0; optional topk per row + tau threshold. Write `corr.npz`. | For one-hot: C via O.T@O/N; sparse-friendly. |
| **6** | D = diag(rowsum(W)); L = D − W; eigenvalues: full eigh if d≤500 else eigsh for k_eigs smallest. Write `laplacian.npz`. | scipy.sparse.linalg.eigsh for large d. |
| **7** | Compute metrics (N_events, d, density_W, trace_L, lambda_min_nonzero, Neff, PR_k). Write `metrics.json`, `spectrum.csv`. | Neff = (sum λ)²/sum λ² over λ > 1e-12; PR(v) = (sum v²)²/sum v⁴. |
| **8** | If baseline: shuffle each column of O independently; repeat steps 5–7; add baseline_Neff, delta_Neff, corr_fro_ratio to metrics and report. | Column-wise shuffle; same pipeline on O_shuffled. |

---

## 5. Acceptance criteria

1. Runs on 2–16 GB RAM without OOM.
2. All files from section 2 are produced.
3. `manifest.json` includes SHA256 of input ROOT and library versions.
4. Baseline (when enabled) is computed and present in `metrics.json` and `report.md`.

---

## 6. Tech stack and CLI

- **Python:** 3.10+.
- **Libraries:** uproot, awkward, numpy, pandas, scipy, pyyaml, tqdm.
- **CLI:** single script `process_root` (in this project: entry point `process_root` → `muons.cli:main`).

**Explicitly out of scope:** Physical interpretation in text; tuning for “expected” results; synthetic data; plots by default.

---

## 7. Suggested module layout (for production code)

- **`muons.io`** — open ROOT, select tree/branches, chunked iteration.
- **`muons.stats`** — chunked min/max/mean/std/nan_rate, median sample.
- **`muons.observables`** — quantile binning + one-hot CSR, or zscore + memmap.
- **`muons.correlation`** — C, W, sparsification (topk, tau).
- **`muons.laplacian`** — L, eigenvalues/eigenvectors (eigh / eigsh).
- **`muons.metrics`** — Neff, PR_k, trace_L, etc.; baseline delta.
- **`muons.manifest`** — SHA256, versions, effective config, timing → manifest.json.
- **`muons.cli`** — argparse + config loading + orchestration (already stubbed).

File size rule: keep each module under ~350–400 lines; use facades if needed.

---

## 8. Optional “strict” extension (from spec end)

Possible hardening: fixed branch list (from `tree.keys()`), fixed binning scheme and exact output filenames so the executor makes no ad-hoc decisions.
