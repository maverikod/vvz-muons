# Step 1 — Open ROOT and select tree

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 1.

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
- Used by: Step 2 (branch selection), and all subsequent chunked reads.

## Constraints (techspec §0)

- Do not load the entire ROOT file into memory.
- Chunked/streaming access only.

## Implementation notes

- Use uproot’s lazy/iterative APIs; avoid `.arrays()` over the full file.
- Persist chosen tree name for `manifest.json` and `features_used.json`.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [ ] **Vectorization / CUDA review:** Confirm no part of the algorithm is implemented with Python loops where vectorized operations (NumPy, Awkward, SciPy) or CUDA would be applicable.
- [ ] **Code mapper:** Run `code_mapper -r <path_to_step_1_code>` (path to the module(s) implementing this step, not the project root).
