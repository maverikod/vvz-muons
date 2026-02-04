# Step 8 — Baseline control (“without correlations”)

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Implementation plan for techspec §3 Step 8.

## References (self-contained)

| Doc | Link | Description |
|-----|------|-------------|
| **Techspec** | [../techspec.md](../techspec.md) §3 Step 8 | Algorithm and constraints |
| **Structure** | [../project_structure.md](../project_structure.md) | Layout, output dirs |
| **Accents** | [../accents.md](../accents.md) | Vectorization, CUDA, parallelization |
| **Rules** | [../RULES.md](../RULES.md) | Project rules |
| **Plan index** | [README.md](README.md) | All steps |
| **Verification** | [../step_by_step_verification.md](../step_by_step_verification.md) | Implementation status |
| **Previous** | [step_04_build_O_matrix.md](step_04_build_O_matrix.md) (O), [step_05_correlation_W.md](step_05_correlation_W.md), [step_06_laplacian_spectrum.md](step_06_laplacian_spectrum.md), [step_07_metrics.md](step_07_metrics.md) | Steps 4–7 (repeat 5–7 on O_shuffled) |

---

## Goal

If `baseline=true`, destroy correlations while preserving marginals by column-wise shuffling of **O**, then repeat [Step 5](step_05_correlation_W.md), [Step 6](step_06_laplacian_spectrum.md), [Step 7](step_07_metrics.md) and add baseline metrics.

## Inputs

- Matrix **O** from [Step 4](step_04_build_O_matrix.md) (same as used for main pipeline).
- **Config:** `baseline` (true/false), `seed` for reproducibility.

## Algorithm

**Only if `baseline=true`:**

1. **Shuffle:** For each column of **O**, shuffle values independently (use `seed` for RNG). Marginals (column distributions) are preserved; correlations are destroyed.
2. **Repeat [Step 5](step_05_correlation_W.md)–[Step 7](step_07_metrics.md)** on the shuffled **O** to obtain:
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
- One-time load of O into memory for column-shuffle is allowed; for very large N×d, RAM may exceed 2–16 GB (document in techspec).

## Implementation notes

- For sparse O (CSR): shuffle indices per column without changing structure, or rebuild O from shuffled bin indices in a vectorized way.
- For dense O: shuffle each column in place or build a copy with shuffled columns; avoid Python per-element loops.
- Do not overwrite main C, W, L, metrics; compute baseline in separate variables and merge only the three extra fields into metrics and report.

---

## Detailed algorithm

1. **If baseline != true:** Skip entire step; do not add baseline_Neff, delta_Neff, corr_fro_ratio.
2. **RNG:** rng = np.random.default_rng(seed). Use seed from config (fixed for reproducibility).
3. **Shuffle O to O_shuffled:**
   - **Dense O (N, d):** For each column j: O_shuffled[:, j] = rng.permutation(O[:, j]) or O_shuffled = O.copy(); for j in range(d): O_shuffled[:, j] = rng.permutation(O[:, j]). Vectorized: O_shuffled = np.column_stack([rng.permutation(O[:, j]) for j in range(d)]) or apply permutation indices per column.
   - **Sparse O (CSR):** Option A: convert to COO; shuffle data within each column (by column index); rebuild CSR. Option B: for each column, get row indices and data; shuffle row indices; reassign. Ensure one 1 per row per "branch block" if one-hot (shuffle within column only).
4. **Repeat [Step 5](step_05_correlation_W.md)–[Step 7](step_07_metrics.md)** on O_shuffled: compute C0, W0, L0, lambda0, Neff0, and optionally spectrum0. Do not overwrite C, W, L, metrics, spectrum from main run.
5. **Frobenius norms:** ||C||_F = np.sqrt(np.sum(C**2)); ||C0||_F = np.sqrt(np.sum(C0**2)). corr_fro_ratio = ||C||_F / ||C0||_F. If ||C0||_F == 0, use nan or omit.
6. **Merge into metrics.json:** Add keys baseline_Neff = Neff0, delta_Neff = Neff - Neff0, corr_fro_ratio. Write updated metrics.json (or append when writing in Step 7).
7. **report.md:** Include a short section or line: baseline_Neff, delta_Neff, corr_fro_ratio (human-readable).

## File format updates

- **metrics.json:** Append (or merge when writing): "baseline_Neff": float, "delta_Neff": float, "corr_fro_ratio": float.
- **report.md:** Text line(s) with baseline_Neff, delta_Neff, corr_fro_ratio.

## Data types and shapes

| Item | Type |
|------|------|
| O_shuffled | Same shape and type as O |
| baseline_Neff, delta_Neff, corr_fro_ratio | float |

## Edge cases and validation

- **seed None:** Use default seed (e.g. 0) or document "non-reproducible".
- **O all zeros:** C0 = 0; corr_fro_ratio = inf or nan; document.
- **Neff0 > Neff:** delta_Neff negative (valid; baseline can have higher Neff).

## Accents (vectorization, CUDA, parallelization)

- **Vectorization:** Shuffle per column with one call per column (rng.permutation) or batched; no element-level loop. PR and Neff reuse same vectorized code as Step 7.
- **CUDA:** Steps 5–6 on O_shuffled can use same GPU backend as main pipeline.
- **Parallelization:** Baseline Steps 5–7 are independent of main run; could run in parallel with main 5–7 if memory allows (optional; usually sequential is simpler).

## Implementation status

| What | Exists in code? |
|------|------------------|
| Column-wise shuffle (dense/sparse) | Yes (`muons.baseline.shuffle_O_columns`) |
| Repeat 5–7 on O_shuffled | Yes (cli: load_O_for_baseline, C0, W0, L0, metrics0) |
| baseline_Neff, delta_Neff, corr_fro_ratio | Yes (merged into metrics_dict, write_metrics_json) |
| Merge into metrics.json and report.md | Yes (`write_report_md` with baseline section) |

**Module:** `muons.baseline`. `shuffle_O_columns`, `load_O_for_baseline`. Orchestrator (cli) calls baseline when args.baseline.

---

## Step completion checklist

- [ ] **Tests:** Run tests for the code written in this step.
- [ ] **Vectorization / CUDA review:** Confirm no part of the algorithm is implemented with Python loops where vectorized operations (NumPy, Awkward, SciPy) or CUDA would be applicable.
- [ ] **Code mapper:** Run `code_mapper -r <path_to_step_8_code>` (path to the module(s) implementing this step, not the project root).
