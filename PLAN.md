# PLAN.md

Purpose: canonical active queue plus deferred backlog.

Status legend: `todo` | `in-progress` | `done` | `blocked`

---

## Current Iteration

No active package-code item.
→ See PROGRESS.md for L1 (keyed layers, 2026-05-07) and M2 (multimodal alignment hardening, 2026-05-07).

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

### Reactivation Checklist

- [ ] Item moved to Current Iteration with explicit scope.
- [ ] User-facing API and backward-compatibility story defined.
- [ ] Tests and smoke validation commands defined before coding.
- [ ] Success metrics and stop criteria explicit.
