# Step 5 — Correlation and connectivity matrix (C and W)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 5.

## References (self-contained)

| Doc | Link | Description |
|-----|------|-------------|
| **Techspec** | [../techspec.md](../techspec.md) §3 Step 5 | Algorithm and constraints |
| **Structure** | [../project_structure.md](../project_structure.md) | Layout, output dirs |
| **Accents** | [../accents.md](../accents.md) | Vectorization, CUDA, parallelization |
| **Rules** | [../RULES.md](../RULES.md) | Project rules |
| **Plan index** | [README.md](README.md) | All steps |
| **Verification** | [../step_by_step_verification.md](../step_by_step_verification.md) | Implementation status |
| **Previous** | [step_04_build_O_matrix.md](step_04_build_O_matrix.md) | Step 4 — Build O matrix |
| **Next** | [step_06_laplacian_spectrum.md](step_06_laplacian_spectrum.md), [step_07_metrics.md](step_07_metrics.md) | Step 6 (L, spectrum); Step 7 (metrics use W) |

---

## Goal

Compute the correlation matrix **C** between columns of **O**, then build the connectivity matrix **W** with optional sparsification.

## Inputs

- Matrix **O** from [Step 4](step_04_build_O_matrix.md) (CSR if quantile, dense/memmap if zscore).
- **Config:** `tau` (default 0.1), `topk` (0 = off).

## Algorithm

1. **Correlation C:** Pearson correlation between columns of **O**.
   - If **O** is sparse one-hot: covariance via **O.T @ O / N**, then normalize to correlations.
2. **Connectivity W:**
   - W = max(0, C).
   - Set diagonal to 0.
3. **Sparsification:**
   - If `topk > 0`: keep top-k edges per row, then symmetrize.
   - If `tau > 0`: set W[i,j] = 0 where W[i,j] < tau.

## Outputs

- **File:** `corr.npz` — arrays `C` (correlation matrix) and `W` (connectivity matrix). Used by [Step 6](step_06_laplacian_spectrum.md), [Step 7](step_07_metrics.md).

## Constraints (techspec §0, accents)

- Vectorized linear algebra; prefer GPU (CUDA) for large d when available; fallback multi-threaded CPU.
- Respect 2–16 GB RAM; avoid materializing full dense W if d is very large and sparse W is sufficient.

## Implementation notes

- For sparse O: **O.T @ O** is efficient as CSR; then scale to correlation (divide by stds).
- For dense O: use vectorized correlation (e.g. centered O then O.T @ O / (N-1), or scipy/numpy correlation).
- Symmetrize after topk: e.g. W = (W + W.T) / 2 or max(W, W.T) depending on semantics.

---

## Detailed algorithm

1. **Correlation C (Pearson):**
   - Sparse one-hot: Cov = (O.T @ O) / N (N = number of events). C[i,j] = Cov[i,j] / (sigma_i * sigma_j) where sigma_i = sqrt(Cov[i,i]); diagonal C[i,i] = 1. Use column stds from sqrt(diag(Cov)) or from O column norms.
   - Dense O: center columns O_c = O - O.mean(axis=0); Cov = (O_c.T @ O_c) / (N-1); C = correlation from Cov (divide by outer(sigma, sigma)).
2. **Connectivity W:** W = np.maximum(C, 0); np.fill_diagonal(W, 0) (or W -= diag(diag(W))).
3. **Sparsification (order may vary):**
   - If tau > 0: W[W < tau] = 0.
   - If topk > 0: For each row i, keep only the largest topk values (excluding diagonal); set rest to 0. Symmetrize: W = max(W, W.T) or W = (W + W.T)/2 so that edge (i,j) exists if either (i,j) or (j,i) was kept.
4. **Save:** np.savez("corr.npz", C=C, W=W) (or savez_compressed).

## File format: `corr.npz`

- Arrays: `C` (float, shape (d, d)), `W` (float, shape (d, d)). Same dtype (e.g. float64).

## Data types and shapes

| Item | Type | Notes |
|------|------|--------|
| O | sparse CSR or dense (N, d) | N = N_events, d = columns. |
| C, W | (d, d) float | Symmetric; W non-negative, diag(W)=0. |

## Edge cases and validation

- **d = 0 or N = 0:** Skip or return empty C, W; document.
- **Column with zero variance (dense):** sigma_i = 0 → C[i,:] and C[:,i] undefined; use 0 or 1 on diagonal, 0 off-diagonal.
- **topk >= d:** Effectively no topk sparsification.
- **tau > max(W):** W becomes zero matrix; L will be zero (handle in Step 6).

## Accents (vectorization, CUDA, parallelization)

- **Vectorization:** O.T @ O in one call; C and W from array ops (no Python loop over i,j). Topk: np.argpartition or sort per row; vectorized masking.
- **CUDA:** Use GPU for O.T @ O and C/W construction when d large and GPU available (cuBLAS, CuPy, etc.). Fallback: multi-threaded BLAS.
- **Parallelization:** BLAS/LAPACK threads for matrix multiply; or parallel over blocks of rows for topk.

## Implementation status

| What | Exists in code? |
|------|------------------|
| Correlation C (sparse/dense) | Yes (`muons.correlation`) |
| W = max(0,C), diag=0 | Yes |
| topk per row + symmetrize | Yes |
| tau threshold | Yes |
| Save corr.npz | Yes |
| CUDA backend | No (CPU vectorized; optional later) |

**Suggested module:** `muons.correlation`. Functions: `compute_correlation(O)`, `build_W(C, tau, topk)`, `save_corr_npz(C, W, path)`.

---

## Step completion checklist

- [x] **Tests:** Run tests for the code written in this step.
- [x] **Vectorization / CUDA review:** Algorithm uses vectorized O.T @ O, argpartition for topk; no event-level Python loops.
- [x] **Code mapper:** Run `code_mapper -r src/muons` (correlation.py included).
