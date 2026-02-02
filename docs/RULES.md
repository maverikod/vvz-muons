# Project rules

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Rules derived from the technical specification (techspec.md), project structure (project_structure.md), and implementation accents (accents.md). All code and design must comply.

---

## 1. Data and layout

| Rule | Source |
|------|--------|
| **docs** — all project documentation lives under `docs/`. | Structure |
| **data** — all data (input and output) under `data/`. | Structure |
| **data/in** — input data only (e.g. ROOT files). CLI `--input` may point here or elsewhere. | Structure |
| **data/out** — default/recommended output directory for pipeline results. All outputs from techspec section 2 go into a single output dir (e.g. `data/out` or `--out`). | Structure |

---

## 2. Techspec constraints (algorithm and memory)

| Rule | Source |
|------|--------|
| **No full ROOT load.** Never load the entire ROOT file into memory. | Techspec §0 |
| **Chunked only.** All processing over ROOT data must be streaming (chunked). Chunk size configurable (default 200000). | Techspec §0, §3 |
| **Strict numerical output.** Output is CSV/NPZ/JSON + short report only. No physics interpretation in text, no synthetic data, no plots by default. | Techspec §0, §6 |
| **RAM budget.** Pipeline must run on 2–16 GB RAM without OOM. | Techspec §4 |
| **All output files.** Pipeline must produce all files listed in techspec §2 (manifest, features_used, branch_stats, bin_definitions when quantile, O_matrix, corr, laplacian, metrics, spectrum, report). | Techspec §2, §4 |
| **Manifest.** `manifest.json` must include SHA256 of input ROOT and library versions. | Techspec §4 |
| **Baseline.** When `baseline=true`, compute baseline (column-shuffle), repeat steps 5–7, and add baseline_Neff, delta_Neff, corr_fro_ratio to metrics and report. | Techspec §3 Step 8, §4 |

---

## 3. Implementation accents (canonical errors)

| Rule | Source |
|------|--------|
| **Vectorization.** Use vectorized operations (NumPy, Awkward, SciPy) over arrays; avoid Python loops over events/features where equivalent vectorized code exists. **Canonical error:** event-level Python loops for numeric transforms. | Accents |
| **CUDA.** Use GPU (CUDA) for heavy linear algebra (correlation, Laplacian, eigenvalues) when a CUDA device is available and problem size justifies it. Fallback to multi-threaded CPU. **Canonical error:** using only CPU when GPU is available and supported. | Accents |
| **Parallelization.** Parallelize at every level where possible and safe: chunk processing, per-branch stats, linear algebra backends (BLAS/LAPACK threads, GPU). **Canonical error:** not using parallelism when it is possible (e.g. single-threaded chunk loop within memory budget). | Accents |

Parallelization and GPU use must respect the 2–16 GB RAM limit and chunked semantics (no full-file load).

---

## 4. Code and repository

| Rule | Description |
|------|-------------|
| **File size.** Source files ≤ 350–400 lines. Split into facade + smaller modules if larger. | User rules |
| **One class per file.** One class per file, except exceptions/enums/errors. | User rules |
| **Docstrings.** Every code file header must include: Author: Vasiliy Zdanovskiy, email: vasilyvz@gmail.com. | User rules |
| **Language.** Code, comments, docstrings, tests: English only. Documentation: English unless requested otherwise. | User rules |
| **Production code.** No `pass` or hardcoded placeholders in production code; abstract methods use `NotImplemented`. | User rules |
| **Imports.** All imports at top of file except when implementing lazy loading. | User rules |
| **Linting.** After production code: run black, flake8, mypy and fix all issues. | User rules |
| **Commit.** After each logical block of file changes: commit. Push only on request. | User rules |

---

## 5. Pipeline behaviour (summary)

- **Input:** `input_root` (required), optional YAML (tree, branches, mode, bins, chunk, max_events, tau, topk, k_eigs, baseline, seed).
- **Output dir:** `data/out` or path from `--out`; all result files written there.
- **Steps:** Open ROOT → choose tree → choose branches (config or auto) → chunked stats → build O (quantile or zscore) → C, W → L, eigenvalues/eigenvectors → metrics, spectrum, report; if baseline, column-shuffle and repeat 5–7.
- **Accents:** Vectorize; use CUDA when available; parallelize chunks, branches, and LA backends within memory and streaming constraints.

Violations of techspec constraints (§0, §2, §4) or of accents (vectorization, CUDA, parallelization when possible) are treated as **canonical errors** and must be fixed before release.
