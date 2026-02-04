# ТЗ: Maximum experimental analysis runs

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Reference for the "Максимальные прогоны экспериментального анализа" specification.

---

## Summary

- **Runs:** RUN-MAX-1 (bins=32), RUN-MAX-2 (bins=16), RUN-MAX-3 (bins=8); all `max_events=0`.
- **Output dirs:** `out_MAX1/`, `out_MAX2/`, `out_MAX3/`.
- **Config:** `config_jagged.yaml` (mode=quantile, baseline=true, tau=0.05, topk=50, k_eigs=500, chunk=200000, allow_jagged=true, full jagged_aggs).
- **After runs:** `runs_summary.csv`, `final_conclusions.md` (facts only).

## How to run

From project root (with `.venv` activated):

```bash
python run_max_runs.py --input /path/to/Run2012BC_DoubleMuParked_Muons.root --config config_jagged.yaml
```

Optional: `--stop-disk-gb 20` (default: stop if free disk < 20 GB).

## Required artifacts per run (§5)

In each `out_MAX*/`: metrics.json, spectrum.csv, laplacian.npz, corr.npz, O_matrix.npz, features_used.json, derived_features.json, branch_stats.csv, bin_definitions.csv, report.md, run.log, manifest.json.

## Success criteria (§7)

1. All three runs completed or stopped only due to disk.
2. Non-degenerate Laplacian (λ_min_nonzero > 0).
3. Neff ≪ d in all runs.
4. Baseline gives substantially smaller Neff.
5. Results consistent across bins 32/16/8.
