# Step 3 — First pass of statistics (chunked)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 3.

---

## Goal

Compute per-branch statistics in one chunked pass over the tree, and estimate medians for imputation.

## Inputs

- Selected tree and branch list from Steps 1–2.
- **Config:** `chunk` (default 200_000), `max_events` (0 = all).

## Algorithm

1. Iterate over the tree in chunks of size `chunk`.
2. For each branch compute:
   - min, max, mean, std,
   - fraction of non-finite (nan/inf) — “nan_rate”.
3. Separately estimate **median** from a sample of the first 200k events (for NaN/imputation in Step 4).

## Outputs

- **File:** `branch_stats.csv` — columns per branch: min, max, mean, std, nan_rate, median, n (or equivalent).

## Constraints (techspec §0, §4)

- Chunked only; no full ROOT load. RAM budget 2–16 GB.
- Vectorized operations (NumPy/Awkward); no event-level Python loops.

## Implementation notes

- Accumulate min/max/mean/std in a single pass (Welford or chunk-wise then merge).
- Median: use first 200k events only to stay within memory; store in branch_stats for Step 4.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [ ] **Vectorization / CUDA review:** Confirm no part of the algorithm is implemented with Python loops where vectorized operations (NumPy, Awkward, SciPy) or CUDA would be applicable.
- [ ] **Code mapper:** Run `code_mapper -r <path_to_step_3_code>` (path to the module(s) implementing this step, not the project root).
