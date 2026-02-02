# Step 8 — Baseline control (“without correlations”)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 8.

---

## Goal

If `baseline=true`, destroy correlations while preserving marginals by column-wise shuffling of **O**, then repeat Steps 5–7 and add baseline metrics.

## Inputs

- Matrix **O** from Step 4 (same as used for main pipeline).
- **Config:** `baseline` (true/false), `seed` for reproducibility.

## Algorithm

**Only if `baseline=true`:**

1. **Shuffle:** For each column of **O**, shuffle values independently (use `seed` for RNG). Marginals (column distributions) are preserved; correlations are destroyed.
2. **Repeat Steps 5–7** on the shuffled **O** to obtain:
   - C0, W0, L0, lambda0, Neff0, spectrum0, etc.
3. **Add to `metrics.json`:**
   - `baseline_Neff` = Neff0.
   - `delta_Neff` = Neff - Neff0.
   - `corr_fro_ratio` = ‖C‖_F / ‖C0‖_F (Frobenius norm ratio).
4. **Report:** Include baseline_Neff, delta_Neff, and corr_fro_ratio in `report.md`.

## Outputs

- Updated `metrics.json` with baseline_Neff, delta_Neff, corr_fro_ratio.
- Updated `report.md` mentioning baseline results.

## Constraints (techspec §3 Step 8, §4)

- Baseline must be computed when baseline=true; acceptance criteria require it in metrics and report.
- Shuffling must be reproducible (fixed seed); vectorized (e.g. numpy RandomGenerator per column).

## Implementation notes

- For sparse O (CSR): shuffle indices per column without changing structure, or rebuild O from shuffled bin indices in a vectorized way.
- For dense O: shuffle each column in place or build a copy with shuffled columns; avoid Python per-element loops.
- Do not overwrite main C, W, L, metrics; compute baseline in separate variables and merge only the three extra fields into metrics and report.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [ ] **Vectorization / CUDA review:** Confirm no part of the algorithm is implemented with Python loops where vectorized operations (NumPy, Awkward, SciPy) or CUDA would be applicable.
- [ ] **Code mapper:** Run `code_mapper -r <path_to_step_8_code>` (path to the module(s) implementing this step, not the project root).
