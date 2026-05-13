# Phase 4 - Performance Findings

## Finding PERF-001: `kbet` Builds Pandas Series Once Per Cell

- Severity: MEDIUM
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

## Finding PERF-002: LISI Metrics Use Python Loops Over Cells

- Severity: LOW
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

## Finding PERF-003: Posterior Mean Defaults to 5000 Monte Carlo Samples

- Severity: LOW
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
