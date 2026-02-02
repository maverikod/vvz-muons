# Step 5 — Correlation and connectivity matrix (C and W)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 5.

---

## Goal

Compute the correlation matrix **C** between columns of **O**, then build the connectivity matrix **W** with optional sparsification.

## Inputs

- Matrix **O** from Step 4 (CSR if quantile, dense/memmap if zscore).
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

- **File:** `corr.npz` — arrays `C` (correlation matrix) and `W` (connectivity matrix).

## Constraints (techspec §0, accents)

- Vectorized linear algebra; prefer GPU (CUDA) for large d when available; fallback multi-threaded CPU.
- Respect 2–16 GB RAM; avoid materializing full dense W if d is very large and sparse W is sufficient.

## Implementation notes

- For sparse O: **O.T @ O** is efficient as CSR; then scale to correlation (divide by stds).
- For dense O: use vectorized correlation (e.g. centered O then O.T @ O / (N-1), or scipy/numpy correlation).
- Symmetrize after topk: e.g. W = (W + W.T) / 2 or max(W, W.T) depending on semantics.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [ ] **Vectorization / CUDA review:** Confirm no part of the algorithm is implemented with Python loops where vectorized operations (NumPy, Awkward, SciPy) or CUDA would be applicable.
- [ ] **Code mapper:** Run `code_mapper -r <path_to_step_5_code>` (path to the module(s) implementing this step, not the project root).
