# PLAN.md

Purpose: canonical active queue plus deferred backlog.

Status legend: `todo` | `in-progress` | `done` | `blocked`

---

## Current Iteration

### DOC-TUTORIAL-1: Modernize Tutorial.ipynb (simulated data vignette)
Status: **in-progress** (2026-05-08)
- 45-cell rewrite complete; px_scale KeyError fixed (added `px_scale` to generative output).
- Re-execution running as smoke test (PID 1176119, log: /tmp/tutorial_rebuild.log).
- Result: pending execution completion.

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

### P-PERF-3. `torch.compile`
Status: **cancelled** (2026-05-08) — dropped from backlog per user decision.

---

### P-PERF-4. SiLU activation in encoder
Status: Deferred | Priority: LOW
Deferral reason: minor change, no urgency.
Reactivation trigger: any encoder-touching session.
→ Full spec: ImplementationPlan.md §P-PERF-4.

---

### N5-D. Fix adversarial overreach on z_private
Status: **done** (2026-05-08)
→ See PROGRESS.md.

---

### N5-E. Class-weighted CE for minority cell types
Status: **done** (pre-existing implementation confirmed 2026-05-08)
→ Weights computed at model init, registered as buffer, threaded into CE calls.

---

### P6. Multi-covariate generalization
Status: **cancelled** (2026-05-08) — dropped from backlog per user decision.

### Reactivation Checklist

- [ ] Item moved to Current Iteration with explicit scope.
- [ ] User-facing API and backward-compatibility story defined.
- [ ] Tests and smoke validation commands defined before coding.
- [ ] Success metrics and stop criteria explicit.
