# Phase 6 - Regression Safety Net

## Regression Tests

Audit regression files:

- `tests/test_audit_dataloaders.py`
  - `test_dataloaders_all_exports_are_importable`: guards INT-001.
- `tests/test_audit_model_spvipesmulti.py`
  - `test_get_latent_representation_indices_subset_loader`: guards INT-002.
  - `test_get_latent_representation_uses_validated_adata_manager`: guards
    INT-003.
- `tests/test_audit_module_spvipesmultimodule.py`
  - `test_negative_binomial_loss_uses_raw_count_targets`: guards INT-005.
  - `test_multimodal_negative_binomial_loss_uses_raw_count_targets`: guards
    INT-005 in multimodal mode.
  - `test_single_modal_all_zero_library_is_finite`: guards INT-004.

These tests now pass normally against the fixed implementation. The only
remaining skipped audit test is the native-candidate placeholder for NAT-001,
which is intentionally not part of the 1.0.1 correctness release.

## Test Data and Reproducibility

- No data files were added.
- Tests use tiny in-memory AnnData or tensor fixtures.
- Tests are CPU-only and deterministic.
- No downloaded datasets are used.

## Verification

Logs are under `audits/SCVI_EXTENSION_AUDIT/logs/`.

- Audit regression slice:
  - `python -m pytest -p no:cacheprovider tests/test_audit_*.py -q`: PASS,
    6 passed, 1 skipped, 1 warning.
- Full suite after fixes:
  - `python -m pytest -p no:cacheprovider -q`: PASS, 294 passed, 3 skipped,
    185 warnings.
- Lower-bound compatibility:
  - Disposable environment with `scvi-tools==1.2.2.post2`:
    `python -m pytest -p no:cacheprovider tests/test_audit_*.py -q`: PASS,
    6 passed, 1 skipped, 1 warning.
  - `scvi-tools==1.0.0` and `scvi-tools==1.1.6.post2` failed import under
    current dependency resolution before spVIPESmulti code was reached.
- Quality gates after cleanup:
  - `ruff format --check .`: PASS.
  - `ruff check .`: PASS.
  - `git ls-files '*.md' -z | xargs -0 mdformat --check`: PASS.
  - `git ls-files '*.md' -z | xargs -0 markdownlint`: PASS.
  - `python -m mypy src/spVIPESmulti`: PASS.
  - `LD_LIBRARY_PATH=<spvm>/lib:$LD_LIBRARY_PATH python -m sphinx -W -b html docs audits/SCVI_EXTENSION_AUDIT/docs_build_html`:
    PASS.
  - `python -m build`: PASS.
- PERF-001 benchmark:
  - `SPVIPES_RUN_BENCHMARKS=1 python -m pytest tests/bench/test_audit_metrics_benchmark.py --benchmark-only -q`:
    PASS, mean 6.99 ms on the 800 cells x 16 dims fixture.

## Baseline Comparison

- Before additive audit tests: 285 collected; 284 passed, 1 skipped, 213
  warnings.
- Initial safety-net state: 292 collected; 284 passed, 3 skipped, 5 xfailed,
  213 warnings.
- Fixed state: 297 tests run; 294 passed, 3 skipped, 185 warnings.
- The INT-001 through INT-005 audit tests have moved from expected-failing
  safety guards to passing regression coverage.
