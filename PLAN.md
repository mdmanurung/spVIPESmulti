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
- Features F1–F7 are now ordered by scientific readiness: measure leakage, strengthen
  covariate disentanglement, then promote safe counterfactuals. Each feature ships with
  a TDD plan and a quantitative go/no-go benchmark (§2 of the roadmap).
- Optional extension tracks F8–F14 are now included in the same roadmap:
  - F8: optional SysVI-style shared-latent VampPrior
  - F9: optional SysVI-style latent cycle-consistency regularizer
  - F10: CellDISECT-aligned Kang benchmark + disentanglement/counterfactual metric pack
  - F11: nonlinear dependence diagnostics (HSIC / MI / partial correlation)
  - F12: conditional decoder / MMD alignment track
  - F13: artifact/QC latent track
  - F14: causal / coupled-VAE research track

Immediate next slice:
- F1 — conditional orthogonality instrumentation is closed.
  - Completed in code: module-level helpers, training-time kwargs wiring,
    optional metric logging in extra_metrics, and green F1/multimodal tests.
  - Closeout passed the Kang IFN overhead gate (`-1.5164%` wall-time overhead vs
    disabled) and wrote `audits/F1/metrics.csv`, `audits/F1/summary.md`, and
    `audits/F1/recommendation.json`.
- Run F4-lite next (condition/donor registration,
  default-off donor/batch covariate heads, scheduled GRL scaling, explicit covariate
  key semantics, and external latent probes).
  - Implemented: registration, default-off heads/losses, guards, scheduled GRL
    scaling, unit/integration tests, and the F4 probe harness.
  - Remaining for F4 promotion: run the full roadmap 3-seed Kang probe matrix and
    decide whether any F4 heads should move beyond opt-in.
- F2 starts after F4-lite has a passing probe/audit baseline and must ship as a safe
  counterfactual API: centroid shifts first, OOD/realism filtering on by default, and
  arbitrary latent replacement treated as a diagnostic helper only.
- Activate/harden F10a (internal CellDISECT-style metric helpers and artifact schema)
  alongside F4/F2; add F10b external CellDISECT runner only after F10a is green.
- Defer F8-F14 implementation until F10 baseline artifacts exist. F11-F14 are concept
  tracks with first slices only; architecture changes in F12-F14 require separate design
  docs before implementation.

Success criteria:
- Per-feature "Pass" rules in `FEATURE_ROADMAP.md` §2.
- Artifacts under `audits/<feature_id>/` plus a PROGRESS.md entry per feature.
- `audits/kang_ifnb/` is the general benchmark lane with append-only
  `metrics.csv` rows and short per-run notes for disentanglement comparisons.

Benchmark tooling status:
- Implemented `scripts/benchmark_kang_ifnb.py` to append one row per
  `(model, seed)` to `audits/kang_ifnb/metrics.csv`.
- Supports `spvipesmulti`, optional original `spVIPES` adapter, and optional
  `contrastiveVAE` adapter; unavailable baselines are logged as rows with
  explanatory `notes` so the audit trail remains complete.
- Next execution step: run the full F4 3-seed Kang probe matrix with
  `scripts/benchmark_f4_covariate_probes.py` and summarize deltas vs baseline.
- Environment note: if `pertpy` download is blocked/corrupted, run with
  `--kang-h5ad-path /absolute/path/to/kang_2018.h5ad`.

Execution note (2026-05-10):
- F1 implementation and closeout are complete. The overhead audit used
  `docs/notebooks/data/kang_2018.h5ad` with megakaryocyte exclusion and passed the
  `<= +5%` gate.
- F4-lite implementation is complete for the code/test/probe-harness slice. A smoke
  probe audit wrote `audits/F4/`; promotion remains blocked on the full 3-seed probe
  benchmark.
- Roadmap sequencing changed after scientific/architecture audit: F4-lite now precedes
  F2, F3 follows F4, and F11-F14 capture nonlinear diagnostics, conditional/MMD,
  artifact-latent, and causal/coupled-VAE extension ideas as deferred tracks.

Parallel external work (not owned in this session):
- N5 malaria B-cell latent-retuning pilot sweep (see HANDOFF.md).

## Blockers / Decisions Needed
- Confirm the Kang IFN technical-batch column for F4 `batch_key`; if absent, F4 benchmark
  must skip `batch-shared` rather than substituting a biological label.

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

### Roadmap items F1–F14
All deferred backlog entries previously tracked here as A2/D2/F1 are now consolidated
into `FEATURE_ROADMAP.md` features F1–F14. Activate by promoting the relevant
feature into the Current Iteration block above and starting with its TDD plan.

### Reactivation Checklist

- [ ] Item moved to Current Iteration with explicit scope.
- [ ] User-facing API and backward-compatibility story defined.
- [ ] Tests and smoke validation commands defined before coding.
- [ ] Success metrics and stop criteria explicit.
