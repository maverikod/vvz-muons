# Step 4 — Build observable matrix O (Π_obs)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 4.

## References (self-contained)

| Doc | Link | Description |
|-----|------|-------------|
| **Techspec** | [../techspec.md](../techspec.md) §3 Step 4 | Algorithm and constraints |
| **Structure** | [../project_structure.md](../project_structure.md) | Layout, output dirs |
| **Accents** | [../accents.md](../accents.md) | Vectorization, CUDA, parallelization |
| **Rules** | [../RULES.md](../RULES.md) | Project rules |
| **Plan index** | [README.md](README.md) | All steps |
| **Verification** | [../step_by_step_verification.md](../step_by_step_verification.md) | Implementation status |
| **Previous** | [step_01_open_root_select_tree.md](step_01_open_root_select_tree.md), [step_02_select_branches.md](step_02_select_branches.md), [step_03_first_pass_stats.md](step_03_first_pass_stats.md) | Steps 1–3 |
| **Next** | [step_05_correlation_W.md](step_05_correlation_W.md), [step_08_baseline.md](step_08_baseline.md) | Step 5 (C, W); Step 8 (baseline uses O) |

---

## Goal

Build the observable matrix **O** (events × features) in either quantile (sparse one-hot) or zscore (dense) mode.

## Inputs

- Tree, branch list, and `branch_stats` (including medians) from [Step 1](step_01_open_root_select_tree.md), [Step 2](step_02_select_branches.md), [Step 3](step_03_first_pass_stats.md).
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
- `O_matrix.npz` — sparse CSR matrix O. Used by [Step 5](step_05_correlation_W.md), [Step 8](step_08_baseline.md).

### Mode B: `zscore`

1. For each branch: replace NaN/inf with median; then z-score = (x - mean) / std (from branch_stats).
2. Store **O** in a memmap (`O_matrix.npy`), writing in chunks.

**Outputs:**

- `zscore_params.json` — mean, std, median per branch.
- `O_matrix.npy` — dense O (memmap or ndarray). Used by [Step 5](step_05_correlation_W.md), [Step 8](step_08_baseline.md).

## Constraints (techspec §0, §4)

- Chunked processing; no full file load. Vectorized operations only.
- Respect 2–16 GB RAM; for zscore use memmap and chunked writes.

## Implementation notes

- Quantile: use `np.quantile` or equivalent on the 200k sample for edges; one-hot via sparse constructor (e.g. `scipy.sparse.csr_matrix` from indices).
- Zscore: stream chunks, compute z-score, write slice to memmap; avoid holding full O in RAM at once.

---

## Detailed algorithm

### Mode A: quantile

1. **Sample for edges:** Load first 200k events: `sample = tree.arrays(branches, entry_stop=min(200_000, N))`. For each branch b: replace NaN/inf with median_b from branch_stats; then `edges_b = np.quantile(values_b, np.linspace(0, 1, bins+1)[1:-1])` or `np.percentile(..., [100*k/bins for k in 1..bins-1])` to get bins-1 inner edges; prepend -inf, append +inf → `bins+1` boundaries.
2. **bin_definitions.csv:** For each branch, for each bin index 0..bins-1: row (branch, bin_id, left_edge, right_edge) from edges[b] and edges[b+1].
3. **Chunked O construction:** For each chunk of events:
   - For each branch: values = chunk[b]; values[~np.isfinite(values)] = median_b; bin_idx = np.searchsorted(edges_b, values, side='right') - 1; clip to [0, bins-1].
   - One-hot: column index for branch b, bin k is `offset_b + k` where offset_b = sum(bins for previous branches). For each event, one column index per branch → d indices per event. Build CSR: row = event index (global), col = offset_b + bin_idx_b, data = 1.
4. **Merge or stream:** Either append CSR chunks vertically (vstack) or build incrementally; final shape (N_events, d), d = sum(bins) over branches.
5. **Save:** `scipy.sparse.save_npz("O_matrix.npz", O)` (CSR).

### Mode B: zscore

1. **Params:** zscore_params = {branch: {"mean": mean_b, "std": std_b, "median": median_b} for branch in branches} from branch_stats. Write `zscore_params.json`.
2. **Memmap:** Create `O = np.memmap("O_matrix.npy", dtype=float, mode='w+', shape=(N_events, d))` with d = len(branches).
3. **Chunked write:** For each chunk: load chunk arrays; for each branch, x = chunk[b]; x[~np.isfinite(x)] = median_b; O_chunk[:, j] = (x - mean_b) / std_b (avoid div by zero: std_b = 0 → 0 or skip branch). Write O[start:stop, :] = O_chunk.
4. **Flush and close** memmap.

## File formats

**bin_definitions.csv** (quantile only):

| Column | Type |
|--------|------|
| branch | str |
| bin_id | int (0..bins-1) |
| left_edge | float |
| right_edge | float |

**zscore_params.json** (zscore only):

```json
{"<branch>": {"mean": float, "std": float, "median": float}, ...}
```

**O_matrix.npz:** Single key (e.g. the array name); CSR matrix. **O_matrix.npy:** Dense float64, shape (N_events, d).

## Data types and shapes

| Item | Quantile | Zscore |
|------|----------|--------|
| O shape | (N_events, d), d = sum(bins) | (N_events, len(branches)) |
| O dtype | sparse float (1.0) | float64 |
| Bin edges | (bins+1,) per branch | — |

## Edge cases and validation

- **std_b = 0 in zscore:** Use 0 for z-score column or skip (avoid inf). Document.
- **All NaN branch in quantile:** All values → median; all in one bin.
- **N_events = 0:** Empty matrix; still write files with correct shape (0, d).

## Accents (vectorization, CUDA, parallelization)

- **Vectorization:** Bin assignment with np.searchsorted; one-hot from indices (no loop over events). Z-score: (x - mean) / std over full column.
- **CUDA:** Optional for very large O construction (quantile one-hot or zscore write).
- **Parallelization:** Chunk-level parallel build of O (then concatenate); or parallel over branches for edge computation.

## Implementation status

| What | Exists in code? |
|------|------------------|
| Quantile edges from 200k sample | Yes (`_quantile_edges_from_sample`) |
| Chunked quantile → one-hot CSR | Yes (`build_quantile_O`) |
| Zscore memmap chunked write | Yes (`build_zscore_O`) |
| bin_definitions.csv, O_matrix.npz, zscore_params.json, O_matrix.npy | Yes |

**Module:** `muons.observables`. Functions: `build_quantile_O(...)`, `build_zscore_O(...)`; helpers `_quantile_edges_from_sample`, `_write_bin_definitions`, `_save_csr`.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step (`tests/test_observables.py`).
- [ ] **Vectorization / CUDA review:** Bin assignment with np.searchsorted; z-score over columns; no event-level Python loops.
- [ ] **Code mapper:** Run `code_mapper -r src/muons -o code_analysis`.
