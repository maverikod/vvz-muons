# Backend (CUDA vs CPU) and vectorization

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Verification that the pipeline uses CUDA when possible and vectorized operations throughout.

---

## 1. CUDA usage in code

- **`muons/backend.py`**: `get_backend(required_bytes=None)` returns `(xp, use_gpu)`. If CuPy is installed, GPU memory &lt; 80%, and (when `required_bytes` is set) free memory ≥ required × 1.2, returns CuPy; otherwise NumPy (CPU).
- **`correlation.py`**: `compute_correlation`, `compute_correlation_from_files` (zscore path), `build_W` call `get_backend(required_bytes=...)` and use `xp.asarray(...)`, `xp` matrix ops (`O.T @ O`, `_cov_to_corr`, etc.). When `xp` is CuPy, data is on GPU and ops run on GPU.
- **`laplacian.py`**: `build_laplacian`, `compute_spectrum` call `get_backend(required_bytes=...)`, then `W_x = xp.asarray(W)`, `L_x = D - W_x`, and `xp.linalg.eigh(L)` or `cupyx.scipy.sparse.linalg.eigsh(L)`. So C, W, L and eigendecomposition run on GPU when backend is CuPy.

**Why GPU might not be used (low GPU load):**

1. **CuPy not installed** — optional dependency: `pip install -e ".[cuda]"`. Without it, backend is always CPU.
2. **GPU memory ≥ 80%** — e.g. nvidia-smi shows 1929/2048 MiB ≈ 94%. Then `get_backend()` logs "GPU memory usage 94% (>= 80%) — using CPU." and returns CPU.
3. **Required memory &gt; free** — for a given step we require `free ≥ required_bytes × 1.2`. If not, we use CPU and log "GPU free memory X MiB < required Y MiB — using CPU."

At run start the pipeline logs either "Compute backend: CUDA (CuPy) — ..." or the reason for CPU. A second line says "Heavy linear algebra ... will run on GPU" or "on CPU".

---

## 2. Vectorization

- **Correlation / Laplacian / W**: All operations are matrix/array ops (`@`, `diag`, `outer`, `eigh`, `argpartition`, etc.). No Python loops over events or features for numeric work.
- **Observables**: `np.column_stack`, `np.searchsorted`, `np.where`, `np.concatenate` over chunks; bin indices and one-hot built in vectorized form.
- **Stats**: Per-branch stats use `np.nanmin`, `np.sum`, etc. over arrays; median from vectorized sample. Chunk loop is over chunks, not over events.
- **Jagged aggregates**: `compute_agg` uses Awkward vectorized ops (`ak.sum(flat, axis=1)`, `ak.mean`, etc.) and `_quantile_per_event` uses `ak.pad_none` + `np.nanquantile(..., axis=1)` (no per-event Python loop).
- **Baseline shuffle**: Dense shuffle uses `np.column_stack([rng.permutation(n) for _ in range(d)])` and `np.take_along_axis` (loop is over d columns, not N rows). CSR shuffle loops over columns to permute row indices; data handling is array-based.

So vectorization is used; the only loops are over branches/chunks/columns (small dimension), not over events.

---

## 3. Batching (chunking)

- ROOT is read in chunks (`chunk` size, default 200_000).
- Branch stats and O-matrix construction process data chunk by chunk.
- Zscore correlation from file accumulates Gram matrix from chunks (`compute_correlation_from_files`); each chunk is sent to `xp` (GPU if available) and `gram += chunk_x.T @ chunk_x`.

---

## 4. Quick check

- Run: `python -m muons.cli --input /path/to/file.root --out data/out --max-events 10000`
- In the first lines of output you should see either "Compute backend: CUDA (CuPy) — ..." and "Heavy linear algebra ... will run on GPU", or "GPU memory usage ... — using CPU" and "Heavy linear algebra will run on CPU".
- If you expect GPU but see CPU: install CuPy (`pip install -e ".[cuda]"`) and/or free GPU memory (e.g. close other processes using the GPU) so that usage &lt; 80%.
