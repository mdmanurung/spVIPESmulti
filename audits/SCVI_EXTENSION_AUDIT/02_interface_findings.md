# Phase 2 - Interface Compatibility Findings

## Finding INT-001: `dataloaders.__all__` Advertises Missing Exports

- Severity: MEDIUM
- Locations: `src/spVIPESmulti/dataloaders/__init__.py:L1` to
  `src/spVIPESmulti/dataloaders/__init__.py:L4`
- scvi-tools version contract checked against: 1.4.2
- Observation: the module imports `AnnDataLoader` and `ConcatDataLoader`, but
  `__all__` also lists `AnnTorchDataset`, `SupervisedConcatDataLoader`, and
  `ClassDataLoader`. Runtime verification shows those three names are absent.
- Risk: `from spVIPESmulti.dataloaders import AnnTorchDataset` fails with
  `ImportError`; docs and autosummary can advertise non-importable API.
- Confidence: HIGH, confirmed dynamically in `spvm`.
- Suggested fix: either bind the advertised names intentionally, or remove them
  from `__all__` and docs.

## Finding INT-002: `get_latent_representation(indices=...)` Is Ignored

- Severity: HIGH
- Locations: `src/spVIPESmulti/model/spvipesmulti.py:L424` to
  `src/spVIPESmulti/model/spvipesmulti.py:L483`
- scvi-tools version contract checked against: 1.4.2
- Observation: the public method accepts and documents `indices`, but no body
  path reads it. The `ConcatDataLoader` receives `group_indices_list` only.
- Risk: callers asking for a subset silently receive embeddings for all
  registered group cells, which can corrupt downstream analyses expecting row
  alignment to the requested subset.
- Confidence: HIGH, confirmed by source inspection and pinned by
  `tests/test_audit_model_spvipesmulti.py`.
- Suggested fix: intersect `indices` with each resolved group list before
  constructing `ConcatDataLoader`, and document returned ordering.

## Finding INT-003: Posterior Calls With New `adata` Keep Using Original Manager

- Severity: HIGH
- Locations: `src/spVIPESmulti/model/spvipesmulti.py:L460` to
  `src/spVIPESmulti/model/spvipesmulti.py:L477`
- scvi-tools version contract checked against: 1.4.2
- Observation: `_validate_anndata(adata)` can transfer or validate a manager for
  a provided AnnData, but `get_latent_representation` then constructs
  `ConcatDataLoader(self.adata_manager, ...)`, not the manager associated with
  the validated AnnData.
- Risk: posterior APIs can appear to accept compatible new AnnData but load
  tensors from the original registered object or stale manager state.
- Confidence: MEDIUM-HIGH. scvi-tools 1.4.2 manager-transfer behavior was
  inspected dynamically; the exact runtime manifestation depends on copied or
  transferred AnnData inputs.
- Suggested fix: after validation, retrieve the manager for that AnnData via
  `self.get_anndata_manager(adata, required=True)` and pass that manager to the
  loader.

## Finding INT-004: Single-modal All-zero Cells Produce `-inf` Library Values

- Severity: HIGH
- Locations: `src/spVIPESmulti/module/spVIPESmultimodule.py:L798` to
  `src/spVIPESmulti/module/spVIPESmultimodule.py:L801`
- scvi-tools version contract checked against: 1.4.2
- Observation: single-modal inference computes `torch.log(xs.sum(1))` without a
  lower clamp. Dynamic verification with an all-zero cell produced
  `tensor([[-inf], [-inf]])`. The multimodal path clamps with `min=1e-6`.
- Risk: all-zero cells produce non-finite library tensors that flow into the
  decoder, making posterior outputs or losses non-finite.
- Confidence: HIGH, confirmed dynamically and pinned by
  `tests/test_audit_module_spvipesmultimodule.py`.
- Suggested fix: match the multimodal path by clamping the library sum before
  `log`, and add an explicit all-zero-cell regression test.

## Finding INT-005: NB Reconstruction Uses `log1p` Targets by Default

- Severity: HIGH
- Locations: `src/spVIPESmulti/module/spVIPESmultimodule.py:L289` to
  `src/spVIPESmulti/module/spVIPESmultimodule.py:L290`,
  `src/spVIPESmulti/module/spVIPESmultimodule.py:L1707` to
  `src/spVIPESmulti/module/spVIPESmultimodule.py:L1718`, and
  `src/spVIPESmulti/module/spVIPESmultimodule.py:L1810` to
  `src/spVIPESmulti/module/spVIPESmultimodule.py:L1819`
- scvi-tools version contract checked against: 1.4.2
- Observation: `log_variational_generative` defaults to `True`. When enabled,
  both single-modal and NB multimodal losses transform observations with
  `torch.log(1 + x)` before passing them to `NegativeBinomialMixture.log_prob`.
- Risk: scvi-tools NB likelihoods model raw non-negative integer-like counts.
  Feeding log-transformed fractional targets changes the reconstruction
  objective and triggers support warnings in the test suite.
- Confidence: HIGH. The full pytest baseline emits distribution support
  warnings at `spVIPESmultimodule.py:L1819`; the target behavior is pinned by
  `tests/test_audit_module_spvipesmultimodule.py`.
- Suggested fix: keep log transformation for encoder inputs only; use raw count
  tensors as NB reconstruction targets unless a non-count likelihood is selected.
