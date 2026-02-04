# Implementation plan — pipeline steps

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Step-by-step implementation plan derived from [techspec.md](../techspec.md) §3. One file per step.

Each step file ends with a **step completion checklist**: testing, vectorization/CUDA review, and running `code_mapper -r <path_to_step_code>` (path to the code of that step, not the project root).

---

## Steps

| Step | File | Summary |
|------|------|---------|
|   | [step_01_open_root_select_tree.md](step_01_open_root_select_tree.md) | Open ROOT file and select TTree |
|   | [step_02_select_branches.md](step_02_select_branches.md) | Select branches (features): config or auto |
|   | [step_03_first_pass_stats.md](step_03_first_pass_stats.md) | Chunked first pass: branch stats and median |
|   | [step_04_build_O_matrix.md](step_04_build_O_matrix.md) | Build observable matrix O (quantile or zscore) |
|   | [step_05_correlation_W.md](step_05_correlation_W.md) | Correlation C and connectivity W |
|   | [step_06_laplacian_spectrum.md](step_06_laplacian_spectrum.md) | Laplacian L and eigenvalue spectrum |
|   | [step_07_metrics.md](step_07_metrics.md) | Numerical metrics and spectrum.csv |
|   | [step_08_baseline.md](step_08_baseline.md) | Baseline control (column-shuffle, repeat 5–7) |

---

## References (self-contained)

| Doc | Link | Description |
|-----|------|-------------|
| **Techspec** | [../techspec.md](../techspec.md) | Full algorithm (§3) and outputs (§2). |
| **Structure** | [../project_structure.md](../project_structure.md) | Layout and output paths. |
| **Accents** | [../accents.md](../accents.md) | Vectorization, CUDA, parallelization. |
| **Rules** | [../RULES.md](../RULES.md) | Project rules. |
| **Verification** | [../step_by_step_verification.md](../step_by_step_verification.md) | Implementation status and step index. |

Each step file contains its own **References** section with links to these docs and to previous/next steps.
