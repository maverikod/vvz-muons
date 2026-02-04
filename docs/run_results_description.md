# Pipeline run: results and numerical conclusions

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Description of the results and numerical conclusions from the pipeline run on real data (Run2012BC_DoubleMuParked_Muons.root).

---

## 1. Run setup

| Item | Value |
|------|--------|
| **Input file** | Run2012BC_DoubleMuParked_Muons.root |
| **Input SHA256** | f8a9d40dba9ee7131a4110f93b4e14a29f99255cbff777018848477f2f61b00b |
| **Tree** | Events |
| **Mode** | quantile |
| **Bins** | 16 |
| **Chunk size** | 200,000 |
| **Max events** | 0 (all) |
| **tau** | 0.1 |
| **topk** | 0 |
| **k_eigs** | 200 |
| **Baseline** | true |
| **Runtime** | 89.57 s |

---

## 2. Data summary

| Quantity | Value |
|----------|--------|
| **N_events** | 61,540,413 |
| **Features (branches) selected** | 1 |
| **Branch name** | nMuon |
| **Observable dimension d** | 16 (one-hot: 1 branch × 16 bins) |

### Branch statistics (nMuon)

| Statistic | Value |
|-----------|--------|
| min | 0.0 |
| max | 50.0 |
| mean | 2.426 |
| std | 1.187 |
| nan_rate | 0.0 |
| median | 2.0 |
| n | 61,540,413 |

---

## 3. Numerical results

### 3.1 Correlation and connectivity

| Metric | Value |
|--------|--------|
| **density_W** | 0.0 |
| **trace_L** | 0.0 |

The connectivity matrix **W** is the zero matrix: after applying W = max(0, C), diagonal set to 0, and threshold tau = 0.1, no entries remain above tau. Thus the graph has no edges.

### 3.2 Spectrum

| Metric | Value |
|--------|--------|
| **lambda_min_nonzero** | — (all eigenvalues 0) |
| **Neff** | — (undefined: no positive eigenvalues) |
| **PR_k** (first 10 modes) | 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 |

**Eigenvalues:** λ₀ … λ₁₅ = 0 (all 16 eigenvalues are zero). The Laplacian L = D − W is the zero matrix because W = 0 and thus D = 0.

### 3.3 Baseline

| Metric | Value |
|--------|--------|
| **baseline_Neff** | — |
| **delta_Neff** | — |
| **corr_fro_ratio** | 1.0 |

With W = 0, the baseline run also yields zero connectivity; the ratio ‖C‖_F / ‖C₀‖_F = 1.0 reflects that the correlation structure is unchanged in the baseline (both sides have the same Frobenius norm).

---

## 4. Conclusions

1. **Single scalar branch:** The Events tree has 6 branches; only **nMuon** is scalar (one value per event). The other branches (Muon_pt, Muon_eta, Muon_phi, Muon_mass, Muon_charge) are jagged arrays and are excluded by the current auto-selection (scalar, numeric, nan_rate ≤ 0.2, std > 0). So the observable space is one feature (nMuon) discretized into 16 bins → d = 16.

2. **Zero connectivity:** With a single one-hot encoded branch, the 16×16 correlation matrix C has a structure that, after zeroing the diagonal and applying tau = 0.1, leaves no entries ≥ tau. Hence W = 0, L = 0, and all eigenvalues are zero. This is a degenerate case: no “graph” of observables to analyse.

3. **Effective dimension:** Neff = (Σλ)² / Σλ² is undefined when all λ = 0 (no positive eigenvalues). PR_k = 1.0 for all reported modes is consistent with flat/zero eigenvectors when L = 0.

4. **Runtime and stability:** The pipeline processed 61.5M events in ~90 s in chunked mode with baseline. No OOM; output files (manifest, features_used, branch_stats, bin_definitions, O_matrix, corr, laplacian, metrics, spectrum, report) were produced as specified.

5. **Recommendation for richer results:** To obtain non-trivial correlation and Laplacian spectrum on this or similar data, use either (a) a ROOT file with more scalar branches, or (b) a config that supplies derived scalar features (e.g. first Muon_pt, sum of Muon_pt, etc.) so that d > 16 and the correlation matrix has off-diagonal structure above tau.

---

*Run date (UTC): 2026-02-02. Output directory: data/out/.*
