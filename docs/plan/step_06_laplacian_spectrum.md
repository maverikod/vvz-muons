# Step 6 — Laplacian and spectrum

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 6.

## References (self-contained)

| Doc | Link | Description |
|-----|------|-------------|
| **Techspec** | [../techspec.md](../techspec.md) §3 Step 6 | Algorithm and constraints |
| **Structure** | [../project_structure.md](../project_structure.md) | Layout, output dirs |
| **Accents** | [../accents.md](../accents.md) | Vectorization, CUDA, parallelization |
| **Rules** | [../RULES.md](../RULES.md) | Project rules |
| **Plan index** | [README.md](README.md) | All steps |
| **Verification** | [../step_by_step_verification.md](../step_by_step_verification.md) | Implementation status |
| **Previous** | [step_05_correlation_W.md](step_05_correlation_W.md) | Step 5 — Correlation, W |
| **Next** | [step_07_metrics.md](step_07_metrics.md) | Step 7 — Metrics, spectrum |

---

## Goal

Build the graph Laplacian **L** from **W** and compute the smallest eigenvalues and (optionally) first eigenvectors.

## Inputs

- Matrix **W** from [Step 5](step_05_correlation_W.md).
- **Config:** `k_eigs` — number of smallest eigenvalues to compute when d is large.

## Algorithm

1. **Degree matrix:** D = diag(sum(W, axis=1)) (row sums of W).
2. **Laplacian:** L = D - W.
3. **Eigenvalues (and eigenvectors):**
   - If d ≤ 500: full eigendecomposition (e.g. `scipy.linalg.eigh` or `numpy.linalg.eigh`).
   - If d > 500: use `scipy.sparse.linalg.eigsh` for the `k_eigs` smallest eigenvalues (and eigenvectors if needed).
4. Sort eigenvalues in ascending order.

## Outputs

- **File:** `laplacian.npz` — arrays:
  - `L` — Laplacian matrix,
  - `lambda` — eigenvalues (sorted ascending),
  - `eigvec_first10` — first 10 eigenvectors (if computed). Used by [Step 7](step_07_metrics.md).

## Constraints (techspec §0, accents)

- Use GPU (CUDA) for heavy linear algebra when available and problem size justifies it; otherwise multi-threaded CPU (BLAS/LAPACK).
- Keep within 2–16 GB RAM; for large d use sparse eigensolver and avoid full dense L if possible.

## Implementation notes

- For sparse W, keep L sparse and use `scipy.sparse.linalg.eigsh` with which='SM'.
- Eigenvectors: request at least 10 when using eigsh so that `eigvec_first10` is available for metrics and spectrum.

---

## Detailed algorithm

1. **Degree vector:** d_vec = np.asarray(W.sum(axis=1)).flatten() (or W.sum(axis=0) depending on layout). Shape (d,).
2. **Laplacian L:** L = D - W where D = diag(d_vec). If W is sparse, keep L sparse: L = scipy.sparse.diags(d_vec) - W. If W dense: L = np.diag(d_vec) - W.
3. **Eigenvalue problem:**
   - If d ≤ 500: Use dense eigh: `w, v = np.linalg.eigh(L)` or scipy.linalg.eigh(L). Full spectrum; sort w ascending; take first 10 columns of v as eigvec_first10.
   - If d > 500: Use `scipy.sparse.linalg.eigsh(L, k=max(k_eigs, 10), which='SM')` to get smallest eigenvalues and eigenvectors. Sort returned eigenvalues ascending; first 10 eigenvectors → eigvec_first10. If k_eigs < 10, still request at least 10 for eigvec_first10.
4. **Sort:** Ensure lambda is sorted ascending (eigsh may return in different order).
5. **Save:** np.savez("laplacian.npz", L=L, lambda=lambda_sorted, eigvec_first10=eigvec_first10). Note: `lambda` is reserved in Python; use key "lambda" in npz or "eigenvalues".

## File format: `laplacian.npz`

- `L`: Laplacian matrix, shape (d, d), sparse or dense.
- `lambda` (or `eigenvalues`): 1D float array, length k (k_eigs or d), sorted ascending.
- `eigvec_first10`: 2D float, shape (d, 10) — first 10 eigenvectors (columns) corresponding to smallest eigenvalues. If fewer than 10 computed, shape (d, k).

## Data types and shapes

| Item | Type | Notes |
|------|------|--------|
| D | diag(d_vec) | (d, d) |
| L | (d, d) | Symmetric; row-sum 0. |
| lambda | (k,) | k = min(d, k_eigs) or d. |
| eigvec_first10 | (d, 10) | Columns unit (or normalized). |

## Edge cases and validation

- **W all zeros:** L = 0; eigenvalues all 0; eigenvectors arbitrary (e.g. identity columns). Document.
- **Disconnected graph:** L has multiple zero eigenvalues; eigsh which='SM' returns smallest including zeros.
- **k_eigs > d:** Request min(k_eigs, d) eigenvalues.
- **Sparse L:** Use eigsh; do not convert to dense if d large (memory).

## Accents (vectorization, CUDA, parallelization)

- **Vectorization:** D and L construction from array ops. No Python loop over matrix elements.
- **CUDA:** Use GPU eigensolver (e.g. cuSOLVER) when available and d justifies it; fallback scipy eigh/eigsh with multi-threaded BLAS.
- **Parallelization:** BLAS/LAPACK threads in eigh/eigsh; or parallel runs for different k if needed.

## Implementation status

| What | Exists in code? |
|------|------------------|
| D = diag(rowsum(W)), L = D - W | Yes (`muons.laplacian.build_laplacian`) |
| eigh for d≤500, eigsh for d>500 | Yes (`_eigen_dense`, `_eigen_sparse`) |
| Request ≥10 eigenvectors | Yes (`compute_spectrum`, k=max(k_eigs, 10)) |
| Sort eigenvalues ascending | Yes |
| Save laplacian.npz | Yes (`save_laplacian_npz`) |
| CUDA backend | No (CPU vectorized; optional later) |

**Module:** `muons.laplacian`. Functions: `build_laplacian(W)`, `compute_spectrum(W, k_eigs)`, `save_laplacian_npz`.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [x] **Vectorization / CUDA review:** No Python loops over matrix elements; eigh/eigsh used.
- [ ] **Code mapper:** Run `code_mapper -r src/muons` (laplacian.py included).
