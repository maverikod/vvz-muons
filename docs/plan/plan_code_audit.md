# Plan vs techspec vs code — audit report

**Author:** Vasiliy Zdanovskiy  
**email:** vasilyvz@gmail.com

Сверка документации плана (step_01–step_08), ТЗ (docs/techspec.md) и реализации в `src/muons/`. Выявлены недописанный код, недоделанные алгоритмы и отклонения от плана/ТЗ.

---

## 1. Недописанный код и плейсхолдеры

| Проверка | Результат |
|----------|-----------|
| `pass` в продакшн-коде | **Нет** — не найдено |
| `NotImplemented` в продакшн-коде | **Нет** — не найдено |
| Жёсткие плейсхолдеры / TODO / FIXME | **Нет** — только комментарии (e.g. laplacian.py про ключ `lambda`) |

**Итог:** Недописанного кода и плейсхолдеров в продакшн-модулях не обнаружено.

---

## 2. Соответствие плану по шагам

| Step | План (docs/plan) | Код | Отклонения |
|------|-------------------|-----|------------|
| 1 — Open ROOT, select tree | io.open_root, select_tree; config `tree`; max num_entries | `muons.io` | Нет |
| 2 — Select branches | config or auto (20k scan; scalar, numeric, nan_rate≤0.2, std>0; cap 64); features_used.json | `muons.branches`; cli пишет features_used.json | Нет |
| 3 — First pass stats | Chunked min/max/mean/std/nan_rate; median from 200k; branch_stats.csv | `muons.stats`; _write_branch_stats_csv в cli | Нет |
| 4 — Build O | Quantile: edges 200k, one-hot CSR, bin_definitions, O_matrix.npz. Zscore: memmap, zscore_params, O_matrix.npy | `muons.observables` | Нет |
| 5 — C, W | C (Pearson; sparse O.T@O/N); W=max(0,C) diag=0; topk then tau; corr.npz | `muons.correlation` | Нет (порядок topk → tau соблюдён) |
| 6 — Laplacian, spectrum | D=diag(rowsum(W)), L=D−W; eigh d≤500 / eigsh k_eigs; laplacian.npz (L, lambda, eigvec_first10) | `muons.laplacian` | Нет |
| 7 — Metrics | density_W, trace_L, lambda_min_nonzero, Neff, PR_k; metrics.json, spectrum.csv | `muons.metrics` | Нет |
| 8 — Baseline | Column-shuffle O; repeat 5–7; baseline_Neff, delta_Neff, corr_fro_ratio в metrics и report | `muons.baseline` + cli | Нет |

Все шаги плана реализованы; явных недоделанных алгоритмов по шагам нет.

---

## 3. Отклонения от ТЗ (techspec)

### 3.1 Исправлено в ходе аудита

- **manifest.json, версии библиотек:** ТЗ §5 требует список библиотек: uproot, awkward, numpy, **pandas**, scipy, pyyaml, tqdm. В `get_library_versions()` не было pandas — **добавлено** (pandas в зависимостях проекта и в ТЗ).

### 3.2 Порядок разреживания W (устранено)

- **ТЗ §3 Шаг 5:** порядок обязателен: сначала topk, затем tau.
- **План step_05:** в разделе «Detailed algorithm» исправлено: «Sparsification (fixed order per techspec §3 Step 5)» — сначала topk, затем tau (раньше было наоборот).
- **ТЗ:** добавлена явная формулировка «порядок обязателен: сначала topk, затем tau».
- **Код** (`correlation.build_W`): уже соблюдал этот порядок.

### 3.3 Baseline и память (документировано)

- В **techspec** §3 Шаг 8 добавлено примечание: допускается однократная загрузка O в память для baseline; при очень больших N×d возможен выход за 2–16 GB RAM.
- В **план step_08** в Constraints добавлено: one-time load of O for column-shuffle is allowed; for very large N×d, RAM may exceed 2–16 GB.

### 3.4 Отклонения, оставшиеся на усмотрение

| Пункт | ТЗ/Правила | Факт в коде | Рекомендация |
|-------|-------------|-------------|--------------|
| **CUDA** | Accents: использовать GPU для тяжёлой линейной алгебры при наличии устройства | Только CPU (NumPy/SciPy) в correlation и laplacian | В плане шагов 5–6 помечено «CUDA backend: No (optional later)». Либо добавить опциональный CUDA-бэкенд, либо зафиксировать в правилах/плане отложенный CUDA. |

---

## 4. Выходы по ТЗ §2 — наличие в коде

| Файл | Требование | Реализация |
|------|------------|------------|
| manifest.json | SHA256 ROOT, дата/время, параметры, версии библиотек, время | `muons.manifest.write_manifest_json`; cli вызывает после шагов 1–8 |
| features_used.json | tree + branches | cli после select_branches |
| branch_stats.csv | min/max/mean/std/nan_rate/median/n | stats + _write_branch_stats_csv |
| bin_definitions.csv | при mode=quantile | observables._write_bin_definitions |
| O_matrix.npz / O_matrix.npy + zscore_params.json | по режиму | observables.build_quantile_O / build_zscore_O |
| corr.npz | C, W | correlation.save_corr_npz |
| laplacian.npz | L, lambda, eigvec_first10 | laplacian.save_laplacian_npz |
| metrics.json | N_events, features_count, mode, bins, d, density_W, trace_L, lambda_min_nonzero, Neff, PR_k; при baseline — baseline_Neff, delta_Neff, corr_fro_ratio | metrics.compute_metrics, write_metrics_json; cli добавляет baseline поля |
| spectrum.csv | k, lambda_k, PR_k | metrics.write_spectrum_csv |
| report.md | краткий отчёт; при baseline — baseline_Neff, delta_Neff, corr_fro_ratio | metrics.write_report_md |

Все обязательные выходы ТЗ §2 создаются.

---

## 5. Чеклисты плана (step completion checklist)

В каждом step_XX файле есть чеклисты: тесты, векторность/CUDA, code_mapper.

- **Тесты:** Есть тесты для всех шагов: test_io, test_branches, test_stats, test_observables, test_correlation, test_laplacian, test_metrics, test_baseline.
- **Векторизация/CUDA:** В коде — векторные операции (NumPy/SciPy), без поэлементных циклов по событиям; CUDA не реализован (см. п. 3.2).
- **Code mapper:** По правилам проекта — запускать после блоков изменений; в чеклистах не отмечено как выполненное (ручной шаг).

---

## 6. Краткий итог

- **Недописанный код:** не найден (нет pass/NotImplemented/плейсхолдеров в продакшн-коде).
- **Недоделанные алгоритмы:** по шагам 1–8 плана все описанные алгоритмы реализованы.
- **Отклонения от плана:** существенных нет; порядок разреживания (topk → tau) соответствует ТЗ.
- **Отклонения от ТЗ:** добавлен pandas в список версий в manifest; порядок разреживания (topk → tau) зафиксирован в плане и ТЗ; допущение по памяти для baseline добавлено в ТЗ и план step_08. Оставшееся — CUDA (опционально позже).

*Дата сверки: 2025-02-02. Обновлено: порядок разреживания и baseline-память устранены/задокументированы.*
