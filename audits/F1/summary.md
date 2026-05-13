# F1 Conditional Orthogonality Instrumentation Audit

Run ID: `f1_kang_overhead_20260510`
Verdict: **pass**

## Overhead Gate

- Gate: orthogonality metric wall-time overhead \<= +5%
- Disabled mean wall time: 0.5078 sec
- Enabled mean wall time: 0.5001 sec
- Overhead: -1.5164%

## Notes

- Dataset: local Kang IFNB H5AD, with megakaryocytes removed.
- Gene subset: top genes by variance on the fixed benchmark subset.
- Training compared identical model settings with only `compute_orthogonality_metric` toggled.
- Targeted validation: `pytest tests/test_disentangle_metrics.py tests/test_multimodal_disentangle.py -q` passed separately.
