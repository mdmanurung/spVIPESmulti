# Phase 5 - Native Acceleration Candidates

## Summary

No GOOD native-extension candidate is recommended yet. The measured bottleneck
is interpreter and pandas overhead in metrics code, but it has a lower-risk
NumPy rewrite path. There is no current evidence that Cython, custom CUDA,
Triton, or pybind11 would be worth the packaging and maintenance cost.

## Candidate NAT-001: Vectorize kBET Before Considering Native Code

- Location: `src/spVIPESmulti/metrics.py:L251` to
  `src/spVIPESmulti/metrics.py:L308`
- Why it qualifies for optimization: cProfile shows `kbet` dominates
  `integration_report` for 800 cells, with 801 pandas `value_counts` calls.
- Why current Python is interpreter-bound: the inner loop creates pandas Series
  objects and reindexes them once per cell.
- Proposed target: pure NumPy first, not native. Encode group labels once and
  compute neighbor count matrices by integer code.
- Expected win: constant-factor estimate. No measured speedup until a parity
  benchmark exists.
- FFI boundary design: not applicable until a vectorized baseline is measured.
- Packaging impact: none for the pure-Python rewrite.
- Risk: LOW, but parity tests must cover one group, imbalanced groups, small
  `k`, and categorical/string labels.
- Pure-Python fallback: yes, and it should be attempted first.

## Candidate NAT-002: `torch.compile` for Module Inference/Loss

- Location: `src/spVIPESmulti/module/spVIPESmultimodule.py:L782` to
  `src/spVIPESmulti/module/spVIPESmultimodule.py:L1889`
- Why it is not yet GOOD: no Lightning profiler run shows module forward or
  loss dominating wall time. The module also has dictionaries keyed by group,
  data-dependent branches for multimodal mode, and distribution objects that
  may cause graph breaks.
- Proposed target if future profiling justifies it: `torch.compile` around
  narrow tensor-only helpers or submodules, not the whole scvi module first.
- Expected win: unknown. Do not claim a speedup without a benchmark.
- Packaging impact: none for `torch.compile`.
- Risk: MEDIUM because dynamic dicts and distributions can graph-break or
  change numerical behavior.
- Pure-Python fallback: keep existing eager path as default until parity and
  performance tests pass.

## Rejected Native Paths

- Numba: not suitable for torch tensors or AnnData/dict-of-tensor boundaries.
- Cython/pybind11: no isolated stable numerical kernel currently justifies a
  compiled extension.
- Custom CUDA/Triton: no evidence that a fused GPU kernel dominates runtime.
