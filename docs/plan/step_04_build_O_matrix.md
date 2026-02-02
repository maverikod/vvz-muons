# Step 4 — Build observable matrix O (Π_obs)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 4.

---

## Goal

Build the observable matrix **O** (events × features) in either quantile (sparse one-hot) or zscore (dense) mode.

## Inputs

- Tree, branch list, and `branch_stats` (including medians) from Steps 1–3.
- **Config:** `mode` (`quantile` | `zscore`), `bins` (default 16), `chunk`, `max_events`.

## Algorithm

### Mode A: `quantile`

1. Using the first 200k events, compute quantile bin edges per branch for `bins` bins.
2. For each event (in chunks):
   - Replace NaN/inf with branch median.
   - Map value to bin index; encode as one-hot (one 1 per branch).
3. **O** = CSR matrix of shape (N_events × d), d = sum of bins over all branches.

**Outputs:**

- `bin_definitions.csv` — per branch: bin_id, left_edge, right_edge.
- `O_matrix.npz` — sparse CSR matrix O.

### Mode B: `zscore`

1. For each branch: replace NaN/inf with median; then z-score = (x - mean) / std (from branch_stats).
2. Store **O** in a memmap (`O_matrix.npy`), writing in chunks.

**Outputs:**

- `zscore_params.json` — mean, std, median per branch.
- `O_matrix.npy` — dense O (memmap or ndarray).

## Constraints (techspec §0, §4)

- Chunked processing; no full file load. Vectorized operations only.
- Respect 2–16 GB RAM; for zscore use memmap and chunked writes.

## Implementation notes

- Quantile: use `np.quantile` or equivalent on the 200k sample for edges; one-hot via sparse constructor (e.g. `scipy.sparse.csr_matrix` from indices).
- Zscore: stream chunks, compute z-score, write slice to memmap; avoid holding full O in RAM at once.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [ ] **Vectorization / CUDA review:** Confirm no part of the algorithm is implemented with Python loops where vectorized operations (NumPy, Awkward, SciPy) or CUDA would be applicable.
- [ ] **Code mapper:** Run `code_mapper -r <path_to_step_4_code>` (path to the module(s) implementing this step, not the project root).
