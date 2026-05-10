# PLAN.md

Purpose: canonical active queue plus deferred backlog.

Status legend: `todo` | `in-progress` | `done` | `blocked`

---

## Current Iteration

### R1. Next-phase feature roadmap (consolidated)
Status: in-progress | Priority: HIGH

Scope:
- Single-source planning lives in `FEATURE_ROADMAP.md` (supersedes the previous
  `DISENTANGLE_SECOND_PASS_ACTION_PLAN.md`, `COUNTERFACTUAL_DESIGN.md`,
  `COUNTERFACTUAL_AUDIT.md`, and `audits/SECOND_PASS_AUDIT_PLAN.md`).
- Features F1–F7 are ordered by architectural risk; each ships with a TDD plan
  and a quantitative go/no-go benchmark (§2 of the roadmap).

Immediate next slice:
- F1 — conditional orthogonality instrumentation (metrics only, no arch change).
- After F1 passes its overhead gate: F2 (counterfactual MVP) and F3+F4 (loss
  terms + covariate heads) can proceed in parallel.

Success criteria:
- Per-feature "Pass" rules in `FEATURE_ROADMAP.md` §2.
- Artifacts under `audits/<feature_id>/` plus a PROGRESS.md entry per feature.

Parallel external work (not owned in this session):
- N5 malaria B-cell latent-retuning pilot sweep (see HANDOFF.md).

## Blockers / Decisions Needed
None.

---

## Deferred Backlog

Rules: every item needs deferral reason and reactivation trigger. Move to Current Iteration before coding.
Full implementation specs for all items live in ImplementationPlan.md.

### P-PERF-1. Vectorize `_label_based_poe` reassembly
Status: **done** (2026-05-08)
→ See PROGRESS.md for implementation detail.

---

### P-PERF-2. Low-rank mixer in `LinearDecoderSPVIPE`
Status: Deferred | Priority: MEDIUM
Deferral reason: architecture change; requires ablation to confirm quality is preserved.
Reactivation trigger: after P-PERF-1 done and profiled.
→ Full spec: ImplementationPlan.md §P-PERF-2.

---

### P-PERF-3. `torch.compile` (blocked on P-PERF-1)
Status: Deferred — blocked | Priority: LOW-MEDIUM
Deferral reason: graph-breaks on `.item()` loop until P-PERF-1 is done.
Reactivation trigger: P-PERF-1 complete and validated.
→ Full spec: ImplementationPlan.md §P-PERF-3.

---

### P-PERF-4. SiLU activation in encoder
Status: Deferred | Priority: LOW
Deferral reason: minor change, no urgency.
Reactivation trigger: any encoder-touching session.
→ Full spec: ImplementationPlan.md §P-PERF-4.

---

### N5-D. Fix adversarial overreach on z_private
Status: Deferred | Priority: MEDIUM
Deferral reason: defer until pilot winner confirmed (Phase 4).
Reactivation trigger: after Phase 3 retrain (v4 model).
→ Full spec: ImplementationPlan.md §N5-D.

---

### N5-E. Class-weighted CE for minority cell types
Status: Deferred | Priority: MEDIUM
Deferral reason: module surgery; defer until pilot results confirm direction.
Reactivation trigger: after Phase 3 retrain (v4 model).
→ Full spec: ImplementationPlan.md §N5-E.

---

### P6. Multi-covariate generalization
Status: Deferred | Priority: LOW
Deferral reason: broad metadata and architecture refactor across data/model/loss.
Reactivation trigger: after single-covariate stability and API simplification.
→ Full spec: ImplementationPlan.md §P6.

---

### Roadmap items F1–F7
All deferred backlog entries previously tracked here as A2/D2/F1 are now consolidated
into `FEATURE_ROADMAP.md` features F1–F7. Activate by promoting the relevant
feature into the Current Iteration block above and starting with its TDD plan.

### Reactivation Checklist

- [ ] Item moved to Current Iteration with explicit scope.
- [ ] User-facing API and backward-compatibility story defined.
- [ ] Tests and smoke validation commands defined before coding.
- [ ] Success metrics and stop criteria explicit.
