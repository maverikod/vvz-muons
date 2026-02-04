# Step 1 — Open ROOT and select tree

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 1.

## References (self-contained)

| Doc | Link | Description |
|-----|------|-------------|
| **Techspec** | [../techspec.md](../techspec.md) §3 Step 1 | Algorithm and constraints |
| **Structure** | [../project_structure.md](../project_structure.md) | Layout, output dirs |
| **Accents** | [../accents.md](../accents.md) | Vectorization, CUDA, parallelization |
| **Rules** | [../RULES.md](../RULES.md) | Project rules |
| **Plan index** | [README.md](README.md) | All steps |
| **Verification** | [../step_by_step_verification.md](../step_by_step_verification.md) | Implementation status |
| **Next** | [step_02_select_branches.md](step_02_select_branches.md) | Step 2 — Select branches |

---

## Goal

Open the input ROOT file and determine which TTree to use for the pipeline.

## Inputs

- **Required:** `input_root` — path to the ROOT file.
- **Optional (YAML):** `tree` — tree name if known.

## Algorithm

1. Open the file via `uproot.open(input_root)`.
2. If `tree` is set in config — use that tree name.
3. Otherwise select the TTree with the maximum `num_entries`.

## Outputs

- In-memory: opened uproot handle and selected tree (name and reference).
- Used by: [Step 2](step_02_select_branches.md) (branch selection), and all subsequent chunked reads.

## Constraints (techspec §0)

- Do not load the entire ROOT file into memory.
- Chunked/streaming access only.

## Implementation notes

- Use uproot’s lazy/iterative APIs; avoid `.arrays()` over the full file.
- Persist chosen tree name for `manifest.json` and `features_used.json`.

---

## Detailed algorithm

1. **Open:** Call `uproot.open(input_root)` → returns file-like handle (e.g. `ReadOnlyDirectory`). Do not call `.arrays()` or load any TTree fully.
2. **List keys:** Get iterable of keys (e.g. `file.keys()` or `file.keys(filter_class=uproot.TTree)`). Only TTrees have `num_entries`.
3. **Resolve tree name:**
   - If config has `tree: "name"` → use `"name"`, then `file[name]` to get tree reference.
   - Else: for each key `k` that is a TTree, get `num_entries`; select `argmax(num_entries)`. Tie-break: first in iteration order.
4. **Return:** `(tree_name: str, tree: TTree)` so that all later steps use `tree` for chunked iteration.

## Data types and shapes

| Item | Type | Notes |
|------|------|--------|
| `input_root` | `str` (path) | Must exist and be readable ROOT file. |
| `tree` (config) | `str \| None` | If set, must be a key in the file that is a TTree. |
| Return: tree_name | `str` | Exact name as in file. |
| Return: tree | uproot.TTree | Supports `tree.keys()`, `tree.arrays(branches, entry_start, entry_stop)`. |

## Edge cases and validation

- **File not found / not ROOT:** Fail with clear error; do not load.
- **No TTrees in file:** Fail or return None; document behaviour.
- **Config `tree` given but missing in file:** Fail with "tree X not found".
- **Multiple TTrees with same max num_entries:** Pick first by key order (deterministic).

## Accents (vectorization, CUDA, parallelization)

- **Vectorization:** N/A (no numeric arrays in this step).
- **CUDA:** N/A.
- **Parallelization:** N/A; single open + selection. Optional: parallel discovery of num_entries per key only if many keys (usually unnecessary).

## Implementation status

| What | Exists in code? |
|------|------------------|
| `uproot.open(path)` | Yes (`muons.io.open_root`) |
| Config `tree` read (YAML) | Yes (`muons.config_loader.load_config`; CLI `--tree`, `--config`) |
| Tree selection by max `num_entries` | Yes (`muons.io.select_tree`) |
| Return (name, tree) for downstream | Yes |

**Suggested module:** `muons.io`. Functions: `open_root(path: str)`, `select_tree(file, tree_name: str \| None) -> tuple[str, TTree]`. Implemented.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step (`tests/test_io.py`).
- [ ] **Vectorization / CUDA review:** N/A for this step (no numeric arrays).
- [ ] **Code mapper:** Run `code_mapper -r src/muons -o code_analysis`.
