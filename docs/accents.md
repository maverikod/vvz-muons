# Project accents (implementation priorities)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Non‑negotiable implementation priorities. Ignoring them when applicable is treated as a **canonical error**.

---

## 1. Vectorization

- **Rule:** Use vectorized operations (NumPy, Awkward, SciPy) over arrays; avoid Python-level loops over events or features where equivalent vectorized code exists.
- **Applies to:** Chunk loading, branch statistics (min/max/mean/std/nan_rate), bin assignment, one-hot encoding, z-score, correlation (e.g. `O.T @ O`), Laplacian construction, metric formulas (Neff, PR_k).
- **Wrong:** `for i in range(n_events): ...` for numeric transforms that can be expressed as array ops.
- **Right:** Operate on full chunks or arrays (e.g. `np.nanmean(chunk, axis=0)`, `scipy.sparse.csr_matrix` from batch indices).

---

## 2. CUDA (GPU)

- **Rule:** Use GPU (CUDA) for compute-heavy linear algebra and dense/sparse operations when the runtime has a CUDA-capable device and the problem size justifies it.
- **Applies to:** Correlation matrix (`C`), matrix `W`, Laplacian `L`, eigenvalue/singular value stages (e.g. cuSOLVER/cuSPARSE or libraries that dispatch to GPU), large matrix products (e.g. `O.T @ O` for zscore mode).
- **Fallback:** If no GPU or CUDA unavailable, use optimized CPU (multi-threaded BLAS/LAPACK, SciPy sparse). Implementation must detect capability and choose backend.
- **Wrong:** Using only CPU when GPU is available and the operation is supported on GPU.

---

## 3. Parallelization everywhere possible

- **Rule:** Parallelize at every level where it is possible and safe. **Not using parallelism when it is possible is a canonical error.**
- **Applies to:**
  - **Chunk-level:** Process multiple chunks in parallel (e.g. thread pool or process pool), respecting memory limits (2–16 GB RAM).
  - **Branch/feature-level:** Compute per-branch statistics or per-branch bin edges in parallel (e.g. `concurrent.futures` or vectorized over columns).
  - **Linear algebra:** Use multi-threaded BLAS/LAPACK (NumPy/SciPy default or env-set MKL/OpenBLAS threads); for eigenvalue solvers use threaded backends.
  - **Baseline vs main run:** If independent, run baseline and main pipeline in parallel where meaningful (or queue as parallel jobs).
- **Constraints:** Must preserve chunked, out-of-core semantics (no full ROOT load); total memory must stay within 2–16 GB; avoid oversubscription (cap worker count / thread count).
- **Wrong:** Single-threaded chunk loop when multiple chunks can be processed concurrently within memory budget; single-threaded correlation/eigen when multi-threaded or GPU backend exists.

---

## Summary

| Accent           | Requirement |
|------------------|-------------|
| **Vectorization**| Prefer array/vectorized ops; no Python event loops for numeric work. |
| **CUDA**         | Use GPU for heavy linear algebra when available; fallback to CPU. |
| **Parallelization** | Parallelize chunks, branches, and LA backends; skipping it when possible = canonical error. |

All three must be considered together with techspec constraints: streaming only, no full-file load, fixed RAM budget.
