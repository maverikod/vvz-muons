# Implementation plan — pipeline steps

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Step-by-step implementation plan derived from [techspec.md](../techspec.md) §3. One file per step.

Each step file ends with a **step completion checklist**: testing, vectorization/CUDA review, and running `code_mapper -r <path_to_step_code>` (path to the code of that step, not the project root).

---

## Steps

| Step | File | Summary |
|------|------|---------|
| 1 | [step_01_open_root_select_tree.md](step_01_open_root_select_tree.md) | Open ROOT file and select TTree |
| 2 | [step_02_select_branches.md](step_02_select_branches.md) | Select branches (features): config or auto |
| 3 | [step_03_first_pass_stats.md](step_03_first_pass_stats.md) | Chunked first pass: branch stats and median |
| 4 | [step_04_build_O_matrix.md](step_04_build_O_matrix.md) | Build observable matrix O (quantile or zscore) |
| 5 | [step_05_correlation_W.md](step_05_correlation_W.md) | Correlation C and connectivity W |
| 6 | [step_06_laplacian_spectrum.md](step_06_laplacian_spectrum.md) | Laplacian L and eigenvalue spectrum |
| 7 | [step_07_metrics.md](step_07_metrics.md) | Numerical metrics and spectrum.csv |
| 8 | [step_08_baseline.md](step_08_baseline.md) | Baseline control (column-shuffle, repeat 5–7) |

---

## References

- **Techspec:** [docs/techspec.md](../techspec.md) — full algorithm and outputs.
- **Structure:** [docs/project_structure.md](../project_structure.md) — layout and output paths.
- **Accents:** [docs/accents.md](../accents.md) — vectorization, CUDA, parallelization.
