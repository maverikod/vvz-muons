# Step 2 — Select branches (features)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 2.

## References (self-contained)

| Doc | Link | Description |
|-----|------|-------------|
| **Techspec** | [../techspec.md](../techspec.md) §3 Step 2 | Algorithm and constraints |
| **Structure** | [../project_structure.md](../project_structure.md) | Layout, output dirs |
| **Accents** | [../accents.md](../accents.md) | Vectorization, CUDA, parallelization |
| **Rules** | [../RULES.md](../RULES.md) | Project rules |
| **Plan index** | [README.md](README.md) | All steps |
| **Verification** | [../step_by_step_verification.md](../step_by_step_verification.md) | Implementation status |
| **Previous** | [step_01_open_root_select_tree.md](step_01_open_root_select_tree.md) | Step 1 — Open ROOT, select tree |
| **Next** | [step_03_first_pass_stats.md](step_03_first_pass_stats.md) | Step 3 — First pass stats |

---

## Goal

Determine the list of branches (features) to use: either from config or by automatic selection.

## Inputs

- Selected tree from [Step 1](step_01_open_root_select_tree.md).
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
- In-memory: branch list for [Step 3](step_03_first_pass_stats.md), [Step 4](step_04_build_O_matrix.md).

## Constraints (techspec §0)

- No full-file load; scan only the first 20k events for auto-selection.
- Vectorized stats (no event-level Python loops).

## Implementation notes

- Use Awkward/NumPy for nan_rate and std over the scan chunk.
- Branch order must be stable for reproducibility and downstream matrices.

---

## Detailed algorithm

**If `branches` is set in config:**

1. For each name in `branches`, check `name in tree.keys()` (or equivalent). If any missing → fail with "branch X not found".
2. Return list in the same order as in config.

**Else (auto-selection):**

1. **Load scan chunk:** `arr = tree.arrays(entry_stop=max_scan)` with `max_scan = 20_000`. Use all tree keys or a safe subset (only scalar branches).
2. **Filter branches:** For each branch (column):
   - **Scalar:** `ak.num(arr[b], axis=1) == 1` or equivalent (one value per event).
   - **Numeric:** dtype in (int, uint, float); reject strings, objects.
   - **nan_rate:** `nan_rate_b = np.isnan(ak.to_numpy(ak.flatten(arr[b]))).mean()` (or vectorized over column). Keep only if `nan_rate_b <= 0.2`.
   - **std:** `std_b = np.nanstd(ak.to_numpy(ak.flatten(arr[b])))`. Keep only if `std_b > 0`.
3. **Order and cap:** Sort remaining by nan_rate ascending (then by branch name for tie-break). Take first 64. Return ordered list of branch names.
4. **Persistence:** Same list is later written to `features_used.json` (in output step).

## File format: `features_used.json`

Written in output phase; content defined here:

```json
{
  "tree": "<selected tree name>",
  "branches": ["branch1", "branch2", ...]
}
```

- `branches`: ordered list of strings; length ≤ 64.

## Data types and shapes

| Item | Type | Notes |
|------|------|--------|
| Scan chunk | Awkward array | shape (n_events,) per branch; n_events ≤ 20_000. |
| nan_rate | float per branch | in [0, 1]. |
| std | float per branch | ≥ 0; filter std > 0. |
| Return | `list[str]` | length ≤ 64, stable order. |

## Edge cases and validation

- **Config branches: empty list:** Treat as "no branches specified" → run auto-selection.
- **Auto-selection: no branch passes filters:** Fail or return empty list; document.
- **Auto-selection: < 64 branches:** Return all that pass.
- **Branch with all NaN:** nan_rate = 1 → excluded. std = 0 → excluded.

## Accents (vectorization, CUDA, parallelization)

- **Vectorization:** Compute nan_rate and std with NumPy/Awkward over full column (no `for i in range(n_events)`).
- **CUDA:** Not required for 20k scan; optional if scaling to much larger scan.
- **Parallelization:** Per-branch stats can be computed in parallel (e.g. ThreadPoolExecutor over branches) within memory; or vectorized over columns in one go.

## Implementation status

| What | Exists in code? |
|------|------------------|
| Config `branches` read | Yes (config.get("branches") in cli) |
| Branch existence check | Yes (`_branches_from_config`) |
| Load up to 20k events | Yes (`tree.arrays(entry_stop=max_scan)`) |
| Scalar/numeric/nan_rate/std filter | Yes (`_branches_auto`) |
| Cap 64, sort by nan_rate | Yes (MAX_BRANCHES, sort by (nan_rate, name)) |
| Write features_used.json | Yes (cli writes to `--out`/features_used.json) |

**Module:** `muons.branches`. Function: `select_branches(tree, branches_config, max_scan=20_000) -> list[str]`.

---

## Step completion checklist

- [x] **Tests:** Run tests for the code written in this step (tests/test_branches.py).
- [x] **Vectorization / CUDA review:** Algorithm uses NumPy/Awkward over columns (no event-level Python loops).
- [ ] **Code mapper:** Run `code_mapper -r src/muons/branches.py` (and cli if desired) when tool is available.
