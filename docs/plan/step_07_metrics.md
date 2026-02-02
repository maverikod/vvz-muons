# Step 7 — Metrics (numerical only)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 7.

## References (self-contained)

| Doc | Link | Description |
|-----|------|-------------|
| **Techspec** | [../techspec.md](../techspec.md) §3 Step 7 | Algorithm and constraints |
| **Structure** | [../project_structure.md](../project_structure.md) | Layout, output dirs |
| **Accents** | [../accents.md](../accents.md) | Vectorization, CUDA, parallelization |
| **Rules** | [../RULES.md](../RULES.md) | Project rules |
| **Plan index** | [README.md](README.md) | All steps |
| **Verification** | [../step_by_step_verification.md](../step_by_step_verification.md) | Implementation status |
| **Previous** | [step_05_correlation_W.md](step_05_correlation_W.md), [step_06_laplacian_spectrum.md](step_06_laplacian_spectrum.md) | Steps 5–6 (W, L, lambda, eigvec) |
| **Next** | [step_08_baseline.md](step_08_baseline.md) | Step 8 — Baseline (merges into metrics/report) |

---

## Goal

Compute numerical metrics from **L**, eigenvalues, and eigenvectors; write metrics and spectrum files.

## Inputs

- **L**, `lambda`, and first eigenvectors from [Step 6](step_06_laplacian_spectrum.md).
- **W** from [Step 5](step_05_correlation_W.md) (for density_W).
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

---

## Detailed algorithm

1. **density_W:** nnz = number of non-zero elements of W (or np.count_nonzero(W)); density_W = nnz / (d * d).
2. **trace_L:** trace_L = np.trace(L) or sum(diag(L)); for Laplacian equals sum of row-degrees = 2 * sum(W) for symmetric W (or sum of d_vec).
3. **lambda_min_nonzero:** lambda_positive = lambda[lambda > 1e-12]; if len(lambda_positive) == 0 then lambda_min_nonzero = np.nan or omit; else lambda_min_nonzero = np.min(lambda_positive).
4. **lambda_use:** lambda_use = lambda[lambda > 1e-12].
5. **Neff:** Neff = (np.sum(lambda_use))**2 / np.sum(lambda_use**2). If lambda_use empty, Neff = np.nan or 0; document.
6. **PR_k:** For k in 0..min(9, n_eigvecs-1): v = eigvec_first10[:, k]; PR_k = (np.sum(v**2))**2 / np.sum(v**4). If v is zero vector, PR_k = nan. Store as list of 10 values (or fewer if fewer eigenvectors).
7. **metrics.json:** Write JSON with keys: N_events, features_count, mode, bins, d, density_W, trace_L, lambda_min_nonzero, Neff, PR_k (list). Add baseline_Neff, delta_Neff, corr_fro_ratio only when baseline was run ([Step 8](step_08_baseline.md)).
8. **spectrum.csv:** Rows k = 0, 1, ... (one per eigenvalue/eigenvector). Columns: k (int), lambda_k (float), PR_k (float or empty). For k >= number of computed eigenvalues, lambda_k can be empty; for k >= 10 or no eigenvectors, PR_k empty or N/A.

## File formats

**metrics.json:** Keys (all scalar or list): N_events, features_count, mode, bins, d, density_W, trace_L, lambda_min_nonzero, Neff, PR_k (list of up to 10 floats). Optional when baseline: baseline_Neff, delta_Neff, corr_fro_ratio.

**spectrum.csv:** Header: k, lambda_k, PR_k. Rows: one per mode (0 to k_max-1). PR_k blank or N/A where not computed.

## Data types and shapes

| Metric | Type |
|--------|------|
| N_events, features_count, d | int |
| mode, bins | str, int |
| density_W, trace_L, lambda_min_nonzero, Neff | float |
| PR_k | list of float (length ≤ 10) |
| spectrum | table (k, lambda_k, PR_k) |

## Edge cases and validation

- **No eigenvalues > 1e-12:** lambda_min_nonzero = nan; Neff = nan or 0.
- **Fewer than 10 eigenvectors:** PR_k list length < 10; spectrum.csv PR_k empty for k >= n_eigvecs.
- **Zero eigenvector:** PR(v) = 0/0 → nan; store nan.

## Accents (vectorization, CUDA, parallelization)

- **Vectorization:** PR_k for all 10 modes in one go: V = eigvec_first10; v2 = V**2; v4 = V**4; PR = (v2.sum(axis=0))**2 / (v4.sum(axis=0)). No loop over k.
- **CUDA:** N/A for this step (metrics are lightweight).
- **Parallelization:** N/A unless computing many metrics in parallel.

## Implementation status

| What | Exists in code? |
|------|------------------|
| density_W, trace_L, lambda_min_nonzero | Yes (`muons.metrics.compute_metrics`) |
| Neff formula | Yes |
| PR_k for first 10 (vectorized) | Yes |
| Write metrics.json | Yes (`write_metrics_json`) |
| Write spectrum.csv | Yes (`write_spectrum_csv`) |

**Module:** `muons.metrics`. Functions: `compute_metrics(...)`, `write_metrics_json(...)`, `write_spectrum_csv(...)`.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [ ] **Vectorization / CUDA review:** Confirm no part of the algorithm is implemented with Python loops where vectorized operations (NumPy, Awkward, SciPy) or CUDA would be applicable.
- [ ] **Code mapper:** Run `code_mapper -r <path_to_step_7_code>` (path to the module(s) implementing this step, not the project root).
