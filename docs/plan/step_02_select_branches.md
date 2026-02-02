# Step 2 — Select branches (features)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 2.

---

## Goal

Determine the list of branches (features) to use: either from config or by automatic selection.

## Inputs

- Selected tree from Step 1.
- **Optional (YAML):** `branches` — explicit list of branch names.

## Algorithm

**If `branches` is set:**

- Use that list; verify each branch exists in the tree.

**Otherwise (auto-selection):**

1. Scan the first `max_scan = 20_000` events.
2. Keep only branches that are:
   - scalar (one value per event),
   - numeric (int/float),
   - `nan_rate <= 0.2`,
   - `std > 0`.
3. Limit to 64 branches (first 64 by order or by smallest `nan_rate`).

## Outputs

- **File:** `features_used.json` — selected tree name + ordered list of branch names.
- In-memory: branch list for Steps 3–4.

## Constraints (techspec §0)

- No full-file load; scan only the first 20k events for auto-selection.
- Vectorized stats (no event-level Python loops).

## Implementation notes

- Use Awkward/NumPy for nan_rate and std over the scan chunk.
- Branch order must be stable for reproducibility and downstream matrices.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [ ] **Vectorization / CUDA review:** Confirm no part of the algorithm is implemented with Python loops where vectorized operations (NumPy, Awkward, SciPy) or CUDA would be applicable.
- [ ] **Code mapper:** Run `code_mapper -r <path_to_step_2_code>` (path to the module(s) implementing this step, not the project root).
