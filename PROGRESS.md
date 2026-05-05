# PROGRESS.md

Purpose: dated execution ledger of what has been implemented, validated, and decided.

How to use:
- Add concise milestone entries with verification evidence.
- Keep implementation detail here, not in PLAN.md or HANDOFF.md.
- Omit "Next action" footers for completed entries; active pointer lives in HANDOFF.md only.

---

## 2026-05-05 (unequal batch-size loss fix)

### U1: Robust loss aggregation for unequal per-group minibatch sizes
Status: completed

What changed:
- Fixed single-modal `loss()` aggregation in `spVIPESmultimodule` so group losses are reduced to scalars before cross-group accumulation.
- Fixed multimodal `_loss_multimodal()` aggregation similarly for per-modality losses and shared PoE KL terms.
- Updated `LossOutput` payload construction to remain shape-safe with unequal per-group sizes by:
  - storing scalar means in `reconstruction_loss` and `kl_local` dictionaries,
  - providing explicit `n_obs_minibatch`.
- Added focused regression tests that directly exercise unequal per-group batch lengths in both single-modal and multimodal loss paths.

Files:
- `src/spVIPESmulti/module/spVIPESmultimodule.py`
- `tests/test_regression_fixes.py`

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_regression_fixes.py -k "UnequalGroupBatchLossAggregation or unequal" -q` passed (`2 passed`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_lightning_trainer_compat.py tests/test_multimodal_disentangle.py tests/test_multigroup_multimodal.py -q` passed (`23 passed`).

## 2026-05-05 (documentation synchronization)

### D1: README/API/docs alignment with repository state
Status: completed

What changed:
  - added `sample_key` in `setup_anndata(...)` usage.
  - added optional sample-aware posterior aggregation and `differential_abundance(...)` usage in the basic workflow.
  - quick-reference table now includes `embed`, `get_shared_posterior`, `get_aggregated_posterior`, and `differential_abundance`.
  - `setup_anndata(...)` signature/table now documents `sample_key`.
  - `train(...)` docs now correctly show `group_indices_list` as optional (auto-inferred fallback).
  - `get_latent_representation(...)` docs now state `batch_size=None` defaults to `scvi.settings.batch_size`.
  - added dedicated sections for `embed`, `get_shared_posterior`, `get_aggregated_posterior`, and `differential_abundance`.
  - removed `notebooks/dialogue_multigroup_vignette` and `notebooks/iri_days_vignette` from `docs/index.md`.

Files:

Verification:

### Final pass: vignette index accuracy (malaria notebook discoverability)
Status: completed

What changed:
- Added `docs/notebooks/malaria_bcells.ipynb` to the Sphinx toctree in `docs/index.md`.
- Added a corresponding tutorial bullet to `README.md` under "Documentation & Tutorials".

Verification:
- Confirmed `malaria_bcells` is referenced in both index locations (`README.md` and `docs/index.md`).
- Checked diagnostics for modified files: no errors.

Next action:
- Keep build troubleshooting deferred; content indexing is now aligned with tracked notebook files.

## 2026-05-05 (audit session)

### Audit-driven bug fixes (8 issues)
Status: completed

Triggered by independent deep-code audit of the full codebase. All items verified with `pytest -q` → `174 passed, 1 skipped`.

#### B1 — `normalized=True` crash in `get_latent_representation`
- `_process_batches` only populated `latent_shared[g]` in the `if not normalized` branch; when `normalized=True` the list stayed empty, causing `torch.cat([])` in `_format_results`.
- Added the symmetric `else` branch for shared PoE latent (mirrors existing private latent handling).
- File: `src/spVIPESmulti/model/spvipesmulti.py`

#### Q1 — Gaussian likelihood correctness (Option B: per-feature heteroscedastic scale)
- `build_likelihood` was using `px_rate_shared` as the mean (ignoring the private/shared mixture) and a hardcoded `scale=0.1`.
- Added `self.log_scale_gaussian: nn.ParameterDict` in `spVIPESmultimodule.__init__`, one `Parameter(zeros(n_features))` per `(group, modality)` with likelihood `"gaussian"`.
- `_generative_multimodal` looks up the parameter and passes it as `log_scale` to `build_likelihood`.
- `build_likelihood` signature updated: `px_scale` (required for Gaussian — the mixed mean) and `log_scale` (required for Gaussian — per-feature log std). Scale is `exp(log_scale).clamp(min=1e-4).expand_as(mean)`.
- Updated `test_gaussian_likelihood` in `tests/test_multigroup_multimodal.py` to pass the new required args.
- Files: `src/spVIPESmulti/module/utils.py`, `src/spVIPESmulti/module/spVIPESmultimodule.py`, `tests/test_multigroup_multimodal.py`

#### Q2 — `get_loadings` multimodal support (Option B: full implementation)
- `module.get_loadings(dataset, type_latent)` now accepts `dataset` as `int` (single-modal) or `(group, modality)` tuple (multimodal) — decoder lookup works for both key shapes.
- `model.get_loadings()` detects `is_multimodal`; for multimodal it iterates `self.module.decoders` keys and returns dict keyed by `((group, modality), latent_type)` with `var_names` from `groups_modality_var_indices`; for single-modal keeps the existing `(i, latent_type)` key scheme.
- Files: `src/spVIPESmulti/module/spVIPESmultimodule.py`, `src/spVIPESmulti/model/spvipesmulti.py`

#### Q3 — Remove `cudnn.benchmark = True` global side effect (Option A)
- Deleted the module-level `torch.backends.cudnn.benchmark = True` line (line 18 of spVIPESmultimodule.py). It mutated global PyTorch state at import time, invisible to users.
- File: `src/spVIPESmulti/module/spVIPESmultimodule.py`

#### B4 — NF prior silently ignored for multimodal private KL
- `_loss_multimodal` always used standard Normal KL for per-modality private latents regardless of `use_nf_prior` / `nf_target`.
- Added `_nf_kl(qz_mod_private, z_mod_private, "private")` branch, mirroring single-modal `loss()` logic.
- File: `src/spVIPESmulti/module/spVIPESmultimodule.py`

#### B5 — Jeffreys integration loss silently ignored in single-modal
- `loss()` never called `_compute_jeffreys_integ_loss`; `use_jeffreys_integ=True` on a single-modal model had no effect.
- Added the `if self.use_jeffreys_integ:` block at the end of `loss()`, matching `_loss_multimodal`.
- File: `src/spVIPESmulti/module/spVIPESmultimodule.py`

#### B8 — `setup_anndata` used `print()` instead of `logger.info()`
- Replaced all 7 `print()` calls (including emoji) with `logger.info()` using `%s` formatting.
- File: `src/spVIPESmulti/model/spvipesmulti.py`

Verification:
- `pytest -q` → `174 passed, 1 skipped, 109 warnings` (57s)
- No new test failures introduced.

Known coverage gaps not yet addressed (test-only work, no bug):
- No test for `normalized=True` path in `get_latent_representation`.
- No test for `get_loadings` on a multimodal model.
- No test for `use_jeffreys_integ=True` on single-modal model.

## 2026-05-05 (hardening follow-up)

### H1: Regression coverage + likelihood support hardening
Status: completed

What changed:
- Added integration regression coverage for:
  - `get_latent_representation(normalized=True)` shape/completeness path.
  - `get_loadings()` on multimodal models with tuple-keyed `(group, modality)` decoders.
  - Single-modal `use_jeffreys_integ=True` path to verify Jeffreys contribution affects loss.
- Added optional strict likelihood support validation in module loss paths:
  - New module flag `strict_likelihood_support` (default `False` for backward compatibility).
  - Added `_validate_likelihood_observations(...)` checks before `log_prob` evaluation.
  - Baseline validation enforces finite values and non-negative observations for NB.
  - Strict mode additionally enforces integer-like NB counts when targets are not log-transformed.
- Added targeted regression test to ensure strict mode rejects fractional NB counts.

Files:
- `src/spVIPESmulti/module/spVIPESmultimodule.py`
- `tests/test_api_boilerplate_reduction.py`
- `tests/test_regression_fixes.py`

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_api_boilerplate_reduction.py tests/test_regression_fixes.py -q` passed (`25 passed`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -q` passed (`177 passed, 2 skipped`).

Notes:
- Existing warning-only support mismatches in synthetic/integration tests remain warning-level by default; strict enforcement is opt-in via `strict_likelihood_support=True`.

### H2: Public docs pass for strict likelihood validation + multimodal loadings
Status: completed

What changed:
- Documented `strict_likelihood_support` in README model-constructor examples and behavior notes.
- Added `strict_likelihood_support` to API constructor parameter table.
- Corrected outdated API note that claimed `get_loadings()` was single-modal only; docs now describe multimodal tuple-key output shape.

Files:
- `README.md`
- `docs/api.md`

Verification:
- Manual docs consistency check against current implementation in `spVIPESmulti.model.spvipesmulti.spVIPESmulti.__init__` and `spVIPESmulti.model.spvipesmulti.spVIPESmulti.get_loadings` completed.

---

## 2026-05-05

### R4: Public evaluation API second slice (held-out validation metrics)
Status: completed

What changed:
- Enabled validation execution by default in `train()` whenever a validation split exists by setting `check_val_every_n_epoch=1` unless the caller already overrides it.
- Fixed `PatchedTrainRunner` to pass the multi-group splitter as a Lightning datamodule, preserving validation dataloaders.
- Updated `MultiGroupDataSplitter` to expose aggregate `n_train` / `n_val` counts and to use evaluation-safe validation/test loaders (`shuffle=False`, `drop_last=False`).
- Updated `get_latent_representation()` to honor the documented default batch size when `batch_size=None`.
- Extended `model.evaluate(...)` to expose `held_out_metrics` from training history when available, including `held_out_nll` as an alias of `reconstruction_loss_validation`.
- Added focused tests for held-out metric extraction and for validation history logging during training.

Verification:
- Discriminating live check: one-epoch CPU training now reports non-empty validation batches and populates validation history keys (`elbo_validation`, `reconstruction_loss_validation`, `validation_loss`, ...).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_evaluate.py tests/test_lightning_trainer_compat.py -q` passed (`23 passed`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -v` passed (`173 passed, 2 skipped`).

Notes:
- `held_out_metrics` are sourced from training history on the model's registered AnnData, not recomputed for arbitrary external AnnData objects.
- This keeps the implementation minimal and consistent with the current training/evaluation architecture.

### R4: Public evaluation API (diagnostics-first)
Status: completed

What changed:
- Added `model.evaluate(...)` to `src/spVIPESmulti/model/spvipesmulti.py`.
- The API computes public, script-free diagnostics for the shared latent and optional private latents using the existing `integration_report(...)` metrics stack.
- `evaluate(...)` accepts either a precomputed shared embedding (`z_shared_key`) or falls back to `get_latent_representation(...)` when no embedding is present.
- Added clear metadata and informational warnings to make fallback paths explicit.
- Kept held-out NLL out of scope for this pass because validation-loss plumbing is still absent in the training path.
- Added focused unit coverage in `tests/test_evaluate.py` for return schema, embedding fallback, label handling, private-latent rows, and finite shared metrics.

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_evaluate.py -q` passed (`19 passed`, warnings only).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_evaluate.py tests/test_enrichment.py -q` passed (`26 passed, 1 skipped`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -v` passed (`170 passed, 2 skipped`).

Notes:
- The repeated auto-inference warning seen in unit tests is expected because each dummy-model fixture is fresh and independently exercises the public fallback path.
- Held-out NLL remains a follow-up item for the R4 second slice, not a regression in this implementation.

### Inference-path speedup: remove duplicate _split_tensors_by_group call
Status: completed

What changed:
- In `_process_batches` (model/spvipesmulti.py), removed the redundant `per_group = self.module._split_tensors_by_group(tensors_by_group)` call that preceded `_get_inference_input`.
- `_get_inference_input` internally calls `_split_tensors_by_group` and already exposes `global_indices` in its return dict. The batch loop now reads `inference_inputs["global_indices"][g]` and `poe_log_z.shape[0]` for the fallback arange.
- Saves one full tensor split per batch iteration in every `get_latent_representation` / `embed` / `get_shared_posterior` call chain.
- `_process_all_cells_with_cycling` is confirmed dead code (not called from any path); left as-is.

Verification:
- `pytest tests/test_api_boilerplate_reduction.py tests/test_regression_fixes.py tests/test_differential_abundance.py tests/test_lightning_trainer_compat.py tests/test_nf_prior.py -v -q` passed (`32 passed`).

### Full regression + smoke validation
Status: completed

What changed:
- Executed repository-wide regression suite and smoke scenarios as required by handoff.
- Confirmed all MrVI DA additions remain stable under full-suite execution.

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -v` passed (`151 passed, 2 skipped, 64 warnings`, `59.38s`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python scripts/smoke_vignettes.py` passed (`7/7` cases).

Notes:
- Smoke script ran on CUDA (`NVIDIA L40S`) and completed all configured cases.
- Remaining warnings are pre-existing environment/model warnings; no new regression failures observed.

## 2026-05-04

### R3: MrVI-style differential abundance
Status: completed

What changed:
- Added optional sample registration in setup path:
  - `spVIPESmulti.setup_anndata(..., sample_key=...)` now registers categorical obs field `"sample"`.
- Added shared-posterior plumbing without touching training/loss:
  - `_process_batches(...)` now collects per-group shared posterior `loc` and `scale` from `poe_stats`.
  - `_format_results(...)` now returns original + reordered shared posterior arrays.
- Added public DA APIs on model class:
  - `get_shared_posterior(...)`
  - `get_aggregated_posterior(...)`
  - `differential_abundance(...)`
- Added sample-aware aggregation and fallback behavior:
  - If `sample_key` is absent, DA aggregation falls back to one synthetic sample per group and emits an informational warning.
- Added alignment precondition warning in `differential_abundance(...)` when both:
  - `disentangle_group_shared_weight == 0`
  - `use_jeffreys_integ == False`
- Added focused tests:
  - New file `tests/test_differential_abundance.py`.
  - Covers sign behavior under synthetic shift, output size, sample-subset filtering, fallback warning, and alignment warning.

Verification:
- `python -m pytest tests/test_differential_abundance.py -v` passed (`5 passed`).
- `python -m pytest tests/test_regression_fixes.py -q` passed (`17 passed`).
- Confirmed no static errors in modified model file via editor diagnostics.

Notes:
- Full `pytest -v` and `scripts/smoke_vignettes.py` were not re-run in this pass to keep turnaround focused on R3 API delivery and targeted regression checks.

### P2: Second-pass doc compression (MrVI spec focus)
Status: completed

What changed:
- Condensed the MrVI DA section in ImplementationPlan.md into a tighter execution contract.
- Removed repeated narrative while preserving all locked decisions and checklist items.
- Kept PLAN/PROGRESS synchronization and added explicit update stamps.

Verification:
- Manual consistency check across PLAN.md, PROGRESS.md, and ImplementationPlan.md completed.
- Confirmed no feature scope or decision changes; only wording/structure compression.

### M1: Consolidated implementation milestone
Status: completed

What changed:
- R1 and R2 shipped:
  - Auto-inferred group indices in train and latent extraction paths.
  - Added one-call embedding API with transactional key writes.
- Enrichment and interpretation QoL shipped:
  - Added enrichment scoring and summarization APIs.
  - Added network validation helper and interpretation report/plots.
  - Added optional enrichment dependency wiring and dedicated tests.
  - Added real decoupler-backed integration coverage.
- Test hardening and docs updates:
  - Hardened CUDA-sensitive tests for mixed driver environments.
  - Added enrichment quickstart and updated README/API docs.
- Planning system consolidation:
  - Consolidated planning/spec sources into PLAN.md + ImplementationPlan.md.
  - Removed obsolete/duplicate planning artifacts.

Verification (high signal):
- `pytest tests/test_utils.py -q` passed.
- `pytest tests/test_api_boilerplate_reduction.py -q` passed.
- `pytest tests/test_enrichment.py -q` passed.
- `pytest -m integration -k enrichment -v` passed.
- `CUDA_VISIBLE_DEVICES='' python scripts/smoke_vignettes.py --epochs 1 --cells_per_group 50 --n_hvg 200` passed (`7/7`).
- Full suite status varied by local CUDA driver state; CPU-safe targeted tests are green.

### P3: Third-pass template normalization
Status: completed

What changed:
- Applied a strict short-template style across HANDOFF.md, PLAN.md, PROGRESS.md, ImplementationPlan.md, and CLAUDE.md continuity section.
- Standardized wording for purpose, usage/read-order, active target, and update stamps.
- Reduced repeated phrasing without changing active scope or decisions.

Verification:
- Manual cross-file consistency check completed.
- Confirmed active target remains R3 MrVI DA and deferred backlog semantics are unchanged.
