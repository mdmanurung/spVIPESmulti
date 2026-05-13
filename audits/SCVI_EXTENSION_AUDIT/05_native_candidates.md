# Phase 5 - Native Acceleration Candidates

## Summary

No GOOD native-extension candidate is recommended. The measured metrics
bottleneck had a lower-risk NumPy rewrite path, and that path is now in place.
There is no current evidence that Cython, custom CUDA, Triton, or pybind11
would be worth the packaging and maintenance cost.

## Candidate NAT-001: Vectorize kBET Before Considering Native Code

- Status: CLOSED. The accepted resolution is the pure NumPy path, not a native
  implementation.
- Location: `src/spVIPESmulti/metrics.py:L251` to
  `src/spVIPESmulti/metrics.py:L308`
- Why it qualified for optimization: cProfile showed `kbet` dominated
  `integration_report` for 800 cells, with 801 pandas `value_counts` calls.
- Resolution: `kbet` now encodes group labels once and computes neighbour
  count matrices through NumPy accumulation. No FFI boundary is needed.
- Expected win: constant-factor reduction in Python/pandas overhead.
- FFI boundary design: not applicable until a vectorized baseline is measured.
- Packaging impact: none for the pure-Python rewrite.
- Risk: LOW. Parity tests cover one group, imbalanced groups, small `k`, and
  string labels.
- Pure-Python fallback: the NumPy implementation is the maintained path.

## Candidate NAT-002: `torch.compile` for Module Inference/Loss

- Status: DEFERRED. This is not a 1.0.1 release blocker.
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
