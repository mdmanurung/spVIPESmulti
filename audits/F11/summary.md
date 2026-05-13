# F11 Nonlinear Dependence Diagnostics Audit

- run_id: `f11_nonlinear_20260513T125425Z`
- verdict: `iterate`
- reason: hsic_rbf CV >0.30 or missing
- hidden_nonlinear_signal: `True`

## Gates

- Complete seed coverage: `[0, 1, 2]`
- Core cross-seed CV threshold: `<= 0.30`
- Cross-seed CV: `{'hsic_rbf': 0.31162060823563753, 'partial_corr_mean_abs': 0.2429299330814246, 'partial_corr_adjusted_mean_abs': 0.2419203643883585}`

## Mean Metrics

- hsic_rbf: `0.006043283712981735`
- hsic_null_p95: `0.00018395639512197638`
- partial_corr_mean_abs: `0.15398895551460592`
- partial_corr_adjusted_mean_abs: `0.15413145879194656`
- orthogonality_within_stratum: `0.07397293361524741`

## Implementation Validation

- `conda run -n spvm python -m pytest tests/test_nonlinear_dependence_metrics.py -q` -> `12 passed`
- `conda run -n spvm python -m pytest tests/test_utils.py -q` -> `61 passed`
- `conda run -n spvm python -m pytest tests/ -q` -> `283 passed, 2 skipped`
