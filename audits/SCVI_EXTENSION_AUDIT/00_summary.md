# spVIPESmulti scvi-tools Extension Audit Summary

## Version Manifest

- Python 3.11.15
- spVIPESmulti 1.0.0 audit baseline; 1.0.1 release target
- scvi-tools 1.4.2
- torch 2.11.0+cu128
- lightning 2.6.1
- anndata 0.12.14
- numpy 2.4.4
- scipy 1.17.1
- pytest 9.0.3

All findings were opened against the 1.0.0 audit baseline. The correctness
fixes are now implemented and guarded by passing regression tests for the
1.0.1 release target.

## Fixed Findings

1. INT-005: NB reconstruction loss now uses raw count targets.
1. INT-002: `get_latent_representation(indices=...)` now subsets per-group
   loader indices.
1. INT-003: posterior calls with a provided `adata` now use that validated
   AnnData manager.
1. INT-004: single-modal all-zero cells now produce finite library tensors.
1. INT-001: `spVIPESmulti.dataloaders.__all__` now contains only importable
   names.

## Completed Fixes

1. INT-001 was fixed by aligning `dataloaders.__all__` with real imports.
1. INT-002 and INT-003 were fixed together in posterior loading.
1. INT-004 was fixed with shared finite observed-library handling.
1. INT-005 was fixed by separating encoder log transformation from NB
   reconstruction targets.
1. PERF-001 and PERF-002 are fixed by the shared NumPy neighbour-count helper
   used by `kbet`, `ilisi`, and `clisi`.

## Native-port Roadmap

- Do not start native ports yet.
- `kbet` and LISI metrics now use the NumPy path; keep native extensions out
  of scope unless a fresh benchmark shows a remaining isolated numerical
  kernel.
- Consider `torch.compile` only after Lightning profiling shows module
  inference/loss dominates runtime and after graph-break risks are isolated.

## Changelog

### Fixed

- `AnnDataLoader` exports: align `spVIPESmulti.dataloaders.__all__` with
  importable names.
- `spVIPESmulti.get_latent_representation`: honor `indices` and provided
  `adata` manager during posterior extraction.
- `spVIPESmultimodule.inference`: clamp single-modal observed library sizes
  before taking logs.
- `spVIPESmultimodule.loss`: evaluate negative-binomial reconstruction
  likelihoods on raw count targets.

### Changed

- `spVIPESmulti.get_latent_representation`: document the returned dictionary
  schema, shapes, and ordering.

## Version-bump Implications

- INT-001, INT-002, INT-003, and INT-004 are bug fixes and fit a patch release.
- INT-005 may change trained-model behavior and benchmark values. Treat it as a
  bug fix scientifically, but consider a minor release note if users rely on
  exact historical losses.
- Removing `log_variational_generative` from the public constructor would need a
  deprecation cycle; changing its default may require a minor release note.

## Compatibility Window

- Proposed fixes should remain compatible with scvi-tools `>=1.0,<2` unless
  the posterior-manager fix relies on APIs introduced after 1.0. Verify against
  the lower bound before release.
- Current audit evidence is checked only against scvi-tools 1.4.2.

## Not Audited

- GPU-only behavior and CUDA profiling were not audited.
- MuData variants were not audited beyond source inventory.
- Full notebook execution was not audited.
- Sphinx notebook execution remains disabled; docs now build with the conda
  environment library path used to resolve `GLIBCXX_3.4.29`.

## Safety Net Confirmation

- Additive audit tests are under `tests/test_audit_*.py`.
- The audit tests for INT-001 through INT-005 are now normal passing regression
  tests, not `xfail` placeholders.
- Audit test slice passes: 6 passed, 1 skipped, 1 warning.
- Full pytest passes: 294 passed, 3 skipped, 185 warnings.
- Quality gates now pass: ruff, mypy, mdformat on tracked Markdown,
  markdownlint on tracked Markdown, Sphinx, build, and pytest.

Audit correctness fixes are complete for the 1.0.1 release target. pytest
status: PASS. PERF-001 and PERF-002 are closed by the current vectorized
metrics implementation. PERF-003 and NAT-002 remain deferred performance
follow-ups, not 1.0.1 release blockers.
