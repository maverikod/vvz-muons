# Step 7 — Metrics (numerical only)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 7.

---

## Goal

Compute numerical metrics from **L**, eigenvalues, and eigenvectors; write metrics and spectrum files.

## Inputs

- **L**, `lambda`, and first eigenvectors from Step 6.
- **W** from Step 5 (for density_W).
- Config: `mode`, `bins`, and pipeline metadata (N_events, features_count, d).

## Algorithm

1. **Metrics to compute:**
   - `N_events` — number of events.
   - `features_count` — number of branches (features).
   - `mode`, `bins`.
   - `d` — dimension of observable space (columns of O).
   - `density_W` = nnz(W) / d².
   - `trace_L` — trace of L.
   - `lambda_min_nonzero` — smallest eigenvalue > 1e-12.
   - `lambda_use` = lambda[lambda > 1e-12].
   - `Neff` = (sum(lambda_use))² / sum(lambda_use²).
   - For the first 10 modes (if eigenvectors available): `PR_k` = (sum(v²))² / sum(v⁴) (participation ratio).

2. **Files:**
   - `metrics.json` — all above metrics (and any other numerical pipeline parameters).
   - `spectrum.csv` — columns: k, lambda_k, PR_k (for first 10 if available).

## Outputs

- **File:** `metrics.json`.
- **File:** `spectrum.csv`.

## Constraints (techspec §6)

- Output is strictly numerical; no physics interpretation or synthetic data in these files.

## Implementation notes

- PR(v) = (sum(v²))² / sum(v⁴) in vectorized form (one line per eigenvector).
- Ensure spectrum.csv has consistent column names (k, lambda_k, PR_k) and PR_k empty or N/A where not computed.
