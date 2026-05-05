# PLAN.md

Purpose: canonical active queue plus deferred backlog.

Status legend: `todo` | `in-progress` | `done` | `blocked`

---

## Current Iteration

No active item. Activate a deferred item or propose a new roadmap candidate, then add it here before coding.

## Blockers / Decisions Needed

None.

---

## Deferred Backlog

Rules: every item needs deferral reason and reactivation trigger. Move to Current Iteration before coding.

### P5. Counterfactual cross-group augmentation

Status: Deferred
Source: CellDISECT (Megas et al., 2025)
Deferral reason: extra encoder pass and private bank maintenance; high training-cost increase.
Reactivation trigger: DA stabilization plus acceptable compute budget.
Notes: add gated weight (default off), reuse direct `z_shared` + `z_private` decoder path.

### P6. Multi-covariate generalization

Status: Deferred
Source: CellDISECT (Megas et al., 2025)
Deferral reason: broad metadata and architecture refactor across data/model/loss.
Reactivation trigger: after single-covariate stability and API simplification.
Notes: promote `groups_key` to multi-key design and nested covariate metadata in `adata.uns`.

### P7. Reference-group decoder masking

Status: Deferred
Source: Multi-ContrastiveVAE (Wang et al., 2024)
Deferral reason: asymmetric behavior and collapse risk if misconfigured.
Reactivation trigger: explicit treatment-vs-control use case.
Notes: add optional `reference_group`; force shared-only decode for reference group.

### Reactivation Checklist

- [ ] Item moved to Current Iteration with explicit scope.
- [ ] User-facing API and backward-compatibility story defined.
- [ ] Tests and smoke validation commands defined before coding.
- [ ] Success metrics and stop criteria explicit.

## Last Updated

- 2026-05-05: Added new shared_15 ranked decoupler extension analysis in the malaria B-cell notebook using DoRothEA TF and PROGENy pathway resources with ULM + MLM and consensus scoring; validated compute, focused TF summary, and final consensus barplots.
- 2026-05-05: Added new shared_15 ranked-program analyses to the malaria B-cell notebook: curated B-cell marker GSEA/ULM, Hallmark GSEA/ULM, and a separate CollecTRI TF-scoring block using ULM on the full shared_15 loading vector; reran stale failed cells so the new outputs are synchronized.
- 2026-05-05: Refined malaria B-cell shared_15 enrichment notebook cells to strip the `Negative_` prefix from loadings-derived gene names and reran enrichment against ImmuneSigDB only using the top 100 positive genes; results are weaker than the broader MSigDB screen and mostly monocyte/DC/NK reference signatures.
- 2026-05-05: Added shared-latent celltype-specificity analysis cells to the malaria B-cell notebook, including one-vs-rest per-dimension AUROC/effect-size ranking, per-dimension best-celltype summaries, and Atypical-focused plots; validated Atypical as strongest on `shared_15` (AUC `0.904`).
- 2026-05-05: Added notebook-friendly model persistence utility `scripts/save_spvipesmulti_model.py` (CLI + importable helper) to save in-memory trained `spVIPESmulti` objects with scvi-version-tolerant save kwargs.
- 2026-05-05: Unequal-batch loss aggregation fix completed: single-modal and multimodal loss paths now aggregate per-group/per-modality terms as scalars and provide shape-safe `LossOutput` bookkeeping; regression coverage added for unequal group batch lengths (`2 passed`) plus related training/multimodal suites (`23 passed`).
- 2026-05-05: Documentation synchronization pass completed: updated README and docs/api for current public APIs (`sample_key`, `embed`, posterior/DA helpers, auto-inferred group indices), removed stale notebook links from docs index, and added an Unreleased changelog note.
- 2026-05-05: Docs hardening pass completed: documented `strict_likelihood_support` in README/API and corrected stale multimodal `get_loadings()` API note.
- 2026-05-05: Hardening follow-up completed: added regression tests for normalized latent path, multimodal get_loadings, and single-modal Jeffreys integration; added optional strict likelihood support validation; full suite green (`177 passed, 2 skipped`).
- 2026-05-05: All roadmap items R1–R4 complete. No active item; Current Iteration cleared.
- 2026-05-05: Audit-driven bug-fix session complete (8 bugs fixed, 174/174 tests pass). See PROGRESS.md for detail.
