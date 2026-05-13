# spVIPESmulti scvi-tools Extension Audit Summary

## Version Manifest

- Python 3.11.15
- spVIPESmulti 1.0.0
- scvi-tools 1.4.2
- torch 2.11.0+cu128
- lightning 2.6.1
- anndata 0.12.14
- numpy 2.4.4
- scipy 1.17.1
- pytest 9.0.3

All findings are relative to this manifest and the current dirty worktree.

## Top Findings

1. INT-005: NB reconstruction loss uses `log1p` targets by default. This is the
   highest scientific-risk finding because NB likelihoods should evaluate raw
   count targets.
1. INT-002: `get_latent_representation(indices=...)` silently ignores
   `indices`.
1. INT-003: posterior calls with a provided `adata` validate it but still pass
   `self.adata_manager` to the loader.
1. INT-004: single-modal all-zero cells produce `-inf` library tensors.
1. INT-001: `spVIPESmulti.dataloaders.__all__` contains non-importable names.

## Dependency-ordered Fix Plan

1. Fix INT-001 independently by aligning `dataloaders.__all__` with real
   imports or adding real compatibility bindings.
1. Fix INT-002 and INT-003 together in posterior loading. Both touch
   `get_latent_representation` loader construction and returned ordering.
1. Fix INT-004 by introducing a shared clamped library helper for single-modal
   and multimodal inference.
1. Fix INT-005 after deciding whether `log_variational_generative` should be
   deprecated or repurposed. Use raw NB targets first; keep encoder log
   transformation separate.
1. Only after correctness fixes, optimize PERF-001 and PERF-002 with vectorized
   NumPy parity tests.

## Native-port Roadmap

- Do not start native ports yet.
- First optimize `kbet` with NumPy and benchmark it.
- Consider `torch.compile` only after Lightning profiling shows module
  inference/loss dominates runtime and after graph-break risks are isolated.

## Changelog Draft

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
- Sphinx build could not pass due environment `GLIBCXX_3.4.29`/`libzmq`
  incompatibility.

## Safety Net Confirmation

- Additive audit tests are under `tests/test_audit_*.py`.
- New-test Ruff format and Ruff lint checks pass.
- Audit Markdown `mdformat --check` and `markdownlint` checks pass.
- Full pytest with audit tests passes: 284 passed, 3 skipped, 5 xfailed,
  213 warnings.
- Baseline status remains: ruff FAIL, mypy FAIL, mdformat FAIL, markdownlint
  FAIL, Sphinx FAIL, build PASS, pytest PASS.

Audit complete. Safety net is in place. pytest, ruff, mypy, and markdown lint
status: pytest PASS; new-file ruff PASS; audit Markdown lint PASS; repo-wide
ruff/mypy/Markdown lint WARN from baseline failures. Awaiting approval to
proceed with fixes - please specify which findings by ID to address and in what
order, and whether to attempt any Phase 5 native ports.
