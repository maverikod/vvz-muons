# Step 6 — Laplacian and spectrum

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 6.

---

## Goal

Build the graph Laplacian **L** from **W** and compute the smallest eigenvalues and (optionally) first eigenvectors.

## Inputs

- Matrix **W** from Step 5.
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
  - `eigvec_first10` — first 10 eigenvectors (if computed).

## Constraints (techspec §0, accents)

- Use GPU (CUDA) for heavy linear algebra when available and problem size justifies it; otherwise multi-threaded CPU (BLAS/LAPACK).
- Keep within 2–16 GB RAM; for large d use sparse eigensolver and avoid full dense L if possible.

## Implementation notes

- For sparse W, keep L sparse and use `scipy.sparse.linalg.eigsh` with which='SM'.
- Eigenvectors: request at least 10 when using eigsh so that `eigvec_first10` is available for metrics and spectrum.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [ ] **Vectorization / CUDA review:** Confirm no part of the algorithm is implemented with Python loops where vectorized operations (NumPy, Awkward, SciPy) or CUDA would be applicable.
- [ ] **Code mapper:** Run `code_mapper -r <path_to_step_6_code>` (path to the module(s) implementing this step, not the project root).
