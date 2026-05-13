# Phase 4 - Performance Findings

## Finding PERF-001: `kbet` Built Pandas Series Once Per Cell

- Severity: MEDIUM
- Status: FIXED. Historical observation describes the 1.0.0 audit baseline.
- Locations: `src/spVIPESmulti/metrics.py:L251` to
  `src/spVIPESmulti/metrics.py:L308`
- Evidence: `audits/SCVI_EXTENSION_AUDIT/perf/metrics_integration_report_cprofile.log`
  profiled `integration_report` on 800 cells x 16 dims. `kbet` consumed
  0.780 s of 0.970 s total. The profile shows 801 calls each to pandas
  `value_counts` and `reindex`.
- Observation: inside the per-cell loop, `kbet` constructs
  `pd.Series(groups[idx[i]])`, then calls `value_counts(...).reindex(...)`.
- Risk: this creates high Python and pandas overhead for every cell. Runtime
  scales poorly with cell count before the algorithmic nearest-neighbor cost is
  the only dominant term.
- Expected win: constant-factor, likely several-fold for metric reports on
  thousands of cells.
- Implementation risk: LOW. Encode group labels once with `np.unique` and
  `return_inverse=True`, then accumulate neighbor counts with NumPy.
- Native requirement: none. Try vectorized NumPy first.
- Resolution: `kbet` now uses `_neighbor_label_counts`, which encodes labels
  once and accumulates per-neighbourhood counts with NumPy. Parity coverage is
  in `tests/test_utils.py::TestKbet`, including imbalanced string labels and
  single-group inputs.
- Benchmark: `audits/SCVI_EXTENSION_AUDIT/perf/kbet_vectorized_pytest_benchmark.log`
  records 800 cells x 16 dims with mean runtime 6.99 ms.

## Finding PERF-002: LISI Metrics Used Python Loops Over Cells

- Severity: LOW
- Status: FIXED. Historical observation describes the 1.0.0 audit baseline.
- Locations: `src/spVIPESmulti/metrics.py:L192` to
  `src/spVIPESmulti/metrics.py:L224`
- Evidence: the same cProfile run shows `ilisi` called twice and consuming
  0.138 s combined on 800 cells. The loop at lines 220-223 calls `np.unique`
  once per cell.
- Observation: group or label codes can be encoded once and counted per
  neighborhood without constructing per-cell unique arrays.
- Risk: metric reporting slows for large validation sweeps.
- Expected win: constant-factor.
- Implementation risk: LOW-MEDIUM. Needs parity tests for label ordering,
  single-category inputs, and small `k`.
- Native requirement: none. Vectorized NumPy or scikit-learn output reuse is
  preferred before any native path.
- Resolution: `ilisi` and `clisi` now share `_neighbor_label_counts` with
  `kbet`. Parity coverage is in `tests/test_utils.py::TestIlisi`, including
  imbalanced string labels and single-category inputs.

## Finding PERF-003: Posterior Mean Defaults to 5000 Monte Carlo Samples

- Severity: LOW
- Status: DEFERRED. This is not a 1.0.1 release blocker.
- Locations: `src/spVIPESmulti/model/spvipesmulti.py:L427`,
  `src/spVIPESmulti/model/spvipesmulti.py:L1484` to
  `src/spVIPESmulti/model/spvipesmulti.py:L1488`, and
  `src/spVIPESmulti/model/spvipesmulti.py:L1496` to
  `src/spVIPESmulti/model/spvipesmulti.py:L1501`
- Evidence: source inspection. The default `mc_samples=5000` is used when
  `normalized=True` and `give_mean=True`; samples are materialized with shape
  `(mc_samples, batch, latent_dim)`.
- Observation: common posterior extraction can allocate large temporary tensors
  even for moderate batches.
- Risk: memory spikes and avoidable latency during embedding extraction.
- Expected win: constant-factor memory and runtime reduction if a smaller
  default or analytic approximation is valid.
- Implementation risk: MEDIUM because changing the default alters numerical
  outputs.
- Native requirement: none.
- Next step: profile real posterior extraction workflows before changing
  defaults or adding approximations.
