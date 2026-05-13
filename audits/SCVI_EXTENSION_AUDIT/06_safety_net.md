# Phase 6 - Regression Safety Net

## Added Tests

New additive files:

- `tests/test_audit_dataloaders.py`
  - `test_dataloaders_all_exports_are_importable`: strict `xfail` for INT-001.
- `tests/test_audit_model_spvipesmulti.py`
  - `test_get_latent_representation_indices_subset_loader`: strict `xfail` for
    INT-002.
  - `test_get_latent_representation_uses_validated_adata_manager`: strict
    `xfail` for INT-003.
- `tests/test_audit_module_spvipesmultimodule.py`
  - `test_negative_binomial_loss_uses_raw_count_targets`: strict `xfail` for
    INT-005.
  - `test_single_modal_all_zero_library_is_finite`: strict `xfail` for INT-004.

These tests are intentionally written as expected-failing tests against desired
contract behavior. When fixes are approved, each corresponding `xfail` should be
removed in the same patch as the fix.

## Test Data and Reproducibility

- No data files were added.
- Tests use tiny in-memory AnnData or tensor fixtures.
- Tests are CPU-only and deterministic.
- No downloaded datasets are used.

## Verification

Logs are under `audits/SCVI_EXTENSION_AUDIT/logs/`.

- New files only:
  - `ruff format --check tests/test_audit_*.py`: PASS.
  - `ruff check tests/test_audit_*.py`: PASS.
  - `mdformat --check audits/SCVI_EXTENSION_AUDIT/*.md`: PASS.
  - `markdownlint audits/SCVI_EXTENSION_AUDIT/*.md`: PASS.
  - `python -m pytest -p no:cacheprovider tests/test_audit_*.py -q`: PASS as
    expected-failing safety net, `5 xfailed`.
- Full suite with additive audit tests:
  - `python -m pytest -p no:cacheprovider --collect-only -q`: PASS,
    292 tests collected.
  - `python -m pytest -p no:cacheprovider -q`: PASS,
    284 passed, 3 skipped, 5 xfailed, 213 warnings.

## Baseline Comparison

- Before additive audit tests: 285 collected; 284 passed, 1 skipped,
  213 warnings.
- After additive audit tests: 292 collected; 284 passed, 3 skipped, 5 xfailed,
  213 warnings.
- The functional pass count is unchanged; the new tests add expected-failing
  guards for fix approval.
