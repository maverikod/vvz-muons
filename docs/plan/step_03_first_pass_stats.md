# Step 3 — First pass of statistics (chunked)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 3.

## References (self-contained)

| Doc | Link | Description |
|-----|------|-------------|
| **Techspec** | [../techspec.md](../techspec.md) §3 Step 3 | Algorithm and constraints |
| **Structure** | [../project_structure.md](../project_structure.md) | Layout, output dirs |
| **Accents** | [../accents.md](../accents.md) | Vectorization, CUDA, parallelization |
| **Rules** | [../RULES.md](../RULES.md) | Project rules |
| **Plan index** | [README.md](README.md) | All steps |
| **Verification** | [../step_by_step_verification.md](../step_by_step_verification.md) | Implementation status |
| **Previous** | [step_01_open_root_select_tree.md](step_01_open_root_select_tree.md), [step_02_select_branches.md](step_02_select_branches.md) | Steps 1–2 |
| **Next** | [step_04_build_O_matrix.md](step_04_build_O_matrix.md) | Step 4 — Build O matrix |

---

## Goal

Compute per-branch statistics in one chunked pass over the tree, and estimate medians for imputation.

## Inputs

- Selected tree and branch list from [Step 1](step_01_open_root_select_tree.md), [Step 2](step_02_select_branches.md).
- **Config:** `chunk` (default 200_000), `max_events` (0 = all).

## Algorithm

1. Iterate over the tree in chunks of size `chunk`.
2. For each branch compute:
   - min, max, mean, std,
   - fraction of non-finite (nan/inf) — “nan_rate”.
3. Separately estimate **median** from a sample of the first 200k events (for NaN/imputation in [Step 4](step_04_build_O_matrix.md)).

## Outputs

- **File:** `branch_stats.csv` — columns per branch: min, max, mean, std, nan_rate, median, n (or equivalent). Used by [Step 4](step_04_build_O_matrix.md).

## Constraints (techspec §0, §4)

- Chunked only; no full ROOT load. RAM budget 2–16 GB.
- Vectorized operations (NumPy/Awkward); no event-level Python loops.

## Implementation notes

- Accumulate min/max/mean/std in a single pass (Welford or chunk-wise then merge).
- Median: use first 200k events only to stay within memory; store in branch_stats for [Step 4](step_04_build_O_matrix.md).

---

## Detailed algorithm

1. **Chunked iteration:** For `start` in 0, chunk, 2*chunk, ... until `max_events` (if > 0) or end of tree:
   - Load `tree.arrays(branches, entry_start=start, entry_stop=min(start+chunk, N))` where N = min(num_entries, max_events) if max_events else num_entries.
2. **Per-chunk stats (vectorized):** For each branch array in chunk:
   - min_b = np.nanmin(chunk_b), max_b = np.nanmax(chunk_b).
   - mean_b: Welford update or chunk mean then merge: n_tot, mean_tot = merge(n1, mean1, n2, mean2).
   - std_b: merge chunk variances (parallel to Welford) or recompute from merged mean.
   - nan_rate_b: count of ~np.isfinite(chunk_b) / len(chunk_b); merge as weighted average over chunks.
3. **Merge across chunks:** Global min = min of chunk mins; global max = max of chunk maxes; mean and std merged by standard formulas (n1*mean1 + n2*mean2)/(n1+n2), combined variance formula).
4. **Median sample:** Load one chunk of first 200k events: `sample = tree.arrays(branches, entry_stop=min(200_000, num_entries))`. Per branch: `median_b = np.nanmedian(ak.to_numpy(ak.flatten(sample[b])))`.
5. **n:** Total number of events processed (sum of chunk sizes).

## File format: `branch_stats.csv`

| Column | Type | Description |
|--------|------|-------------|
| branch | str | Branch name (index or first column). |
| min | float | Global minimum (nan-free). |
| max | float | Global maximum. |
| mean | float | Global mean. |
| std | float | Global std. |
| nan_rate | float | Fraction of non-finite values in [0,1]. |
| median | float | From first 200k events. |
| n | int | Number of events (rows) used. |

One row per branch; same order as `features_used.json`.

## Data types and shapes

| Item | Type | Notes |
|------|------|--------|
| chunk | int | Default 200_000. |
| Per-branch stats | 7 scalars (min, max, mean, std, nan_rate, median, n) | float except n. |
| branch_stats table | (n_branches, 8) | Rows = branches. |

## Edge cases and validation

- **Branch all NaN:** mean/std undefined; use np.nan or 0; median from sample may be NaN (Step 4 must handle).
- **max_events < 200k:** Median sample uses min(200k, max_events) events.
- **Empty tree:** n=0; min/max/mean/std/median can be NaN or omitted; document.

## Accents (vectorization, CUDA, parallelization)

- **Vectorization:** All per-branch stats from array ops (np.nanmin, np.nanmax, np.nanmean, np.nanstd, np.nanmedian over chunk columns). No event-level loop.
- **CUDA:** Optional for very large chunk stats.
- **Parallelization:** Chunk loop can run in parallel (e.g. process K chunks, then merge); or parallel over branches within a chunk. Must respect 2–16 GB RAM (limit concurrent chunks).

## Implementation status

| What | Exists in code? |
|------|------------------|
| Chunked iteration over tree | Yes (`compute_branch_stats` loop) |
| Merge min/max/mean/std/nan_rate over chunks | Yes (accumulators, `_update_acc`, `_to_rows`) |
| Median from first 200k | Yes (`_median_sample`) |
| Write branch_stats.csv | Yes (cli `_write_branch_stats_csv`) |

**Module:** `muons.stats`. `compute_branch_stats(tree, branches, chunk, max_events)` returns list of dicts; CSV written in cli.

---

## Step completion checklist

- [x] **Tests:** Run tests for the code written in this step (tests/test_stats.py).
- [x] **Vectorization / CUDA review:** Stats use NumPy over chunk columns (no event-level loops).
- [ ] **Code mapper:** Run `code_mapper -r src/muons/stats.py` when tool is available.
