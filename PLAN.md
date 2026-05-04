# PLAN.md

Purpose: canonical planning document for active queue + deferred backlog.

How to use:
- Keep active queue short and execution-oriented.
- Update statuses before and after each coding block.
- Keep deferred ideas in the Deferred Backlog section below.
- Link to design detail docs instead of duplicating full specs.

Status legend:
- `todo`: approved but not started
- `in-progress`: currently being worked on
- `done`: completed and recorded in PROGRESS.md
- `blocked`: waiting on decision or dependency

---

## Current Iteration

### 1) MrVI-style differential abundance
Status: `todo`
Design doc: `FeaturePlanMrvi.md`
Goal:
- Add DA APIs and tests based on shared-posterior aggregation.

Planned work:
- Surface shared posterior `loc/scale` in latent extraction flow.
- Register optional sample metadata key for DA.
- Implement `get_shared_posterior`, `get_aggregated_posterior`, and `differential_abundance`.
- Add DA unit tests and update dependencies if needed.

Exit criteria:
- DA tests pass and smoke tests show no regression.

### 2) API boilerplate reduction (group index inference)
Status: `todo`
Roadmap source: `ImplementationPlan.md` (R1)
Goal:
- Remove manual `group_indices_list` setup from normal user workflows.

Exit criteria:
- Public APIs can infer group indices from registered anndata metadata.

### 3) One-call embedding API
Status: `todo`
Roadmap source: `ImplementationPlan.md` (R2)
Goal:
- Add a single-call embedding path that computes and stores shared/private latents.

Exit criteria:
- A documented `embed(...)` flow writes standard `obsm` keys.

---

## Blockers / Decisions Needed

- Confirm final DA return object (`xarray.Dataset` vs DataFrame + metadata).

---

## Deferred Backlog

Purpose: valid ideas intentionally not in active execution.

Backlog rules:
- Add items here only when deferred by an explicit decision.
- Each item must include deferral reason and re-activation trigger.
- Move an item to Current Iteration before coding starts.
- Keep completed history out of this section (record completion in `PROGRESS.md`).

### P5. Counterfactual cross-group augmentation

Status: Deferred
Source: CellDISECT (Megas et al., 2025)
Deferral reason:
- Requires extra encoder pass and private-posterior bank maintenance.
- Expected training cost increase is substantial.

Re-activation trigger:
- After current API streamlining and DA work stabilize.
- When compute budget for heavier training passes is acceptable.

Implementation notes:
- Add gated weight (default off).
- Reuse decoder path that already accepts `z_shared` + `z_private` directly.

### P6. Multi-covariate generalization

Status: Deferred
Source: CellDISECT (Megas et al., 2025)
Deferral reason:
- Requires broad metadata and architecture refactor across data, model, and loss.

Re-activation trigger:
- Only after current single-covariate stability and API simplification land.

Implementation notes:
- Promote single `groups_key` to multiple covariate keys.
- Introduce nested covariate metadata in `adata.uns`.

### P7. Reference-group decoder masking

Status: Deferred
Source: Multi-ContrastiveVAE (Wang et al., 2024)
Deferral reason:
- Asymmetric by design and not suitable for all use cases.
- Risk of reference-dominated representation collapse if misconfigured.

Re-activation trigger:
- A concrete treatment-vs-control use case where asymmetry is desired.

Implementation notes:
- Add optional `reference_group` parameter.
- Override decoder mixing for reference group to shared-only reconstruction.

### Reactivation Checklist (for any deferred item)

- [ ] Item has an owning implementation plan in Current Iteration.
- [ ] User-facing API and backward-compatibility story is explicit.
- [ ] Tests and smoke validation commands are defined before coding.
- [ ] Success metrics and stop criteria are explicit.

## Last Updated

- 2026-05-04: Merged deferred backlog from `PLANS.md` into this canonical plan doc.
- 2026-05-04: Initialized active plan tracker and aligned with roadmap/feature docs.
