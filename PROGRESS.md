# PROGRESS.md

Purpose: dated execution ledger of what has been implemented, validated, and decided.

How to use:
- Append new entries; do not rewrite history.
- Include file-level change summary and verification commands.
- If a task is incomplete, include clear next action.

---

## 2026-05-04

### Consolidated PLAN.md + PLANS.md into one canonical planning source
Status: completed

What changed:
- Merged deferred backlog content (P5/P6/P7 + intake/reactivation rules) into `PLAN.md`.
- Kept `PLANS.md` as a lightweight compatibility redirect to avoid breaking older references.
- Updated planning references in `CLAUDE.md` and `ImplementationPlan.md` to point to `PLAN.md`.
- Updated `scripts/validate_disentanglement.py` header text to reference planning checklist in `PLAN.md`.

Verification:
- Checked for `PLANS.md` references and updated active documentation pointers.
- Verified no code-path behavior changes in `src/`.

Next action:
- Continue feature work from `PLAN.md` Current Iteration and keep deferred items in `PLAN.md` Deferred Backlog.

### Documentation system refactor for implementation continuity
Status: completed

What changed:
- Re-scoped `ImplementationPlan.md` into roadmap-only candidates.
- Re-scoped `FeaturePlanMrvi.md` into a feature-specific implementation spec with checklists.
- Re-scoped `PLANS.md` into a strict deferred-only backlog.
- Updated `CLAUDE.md` with a documentation map and fresh-session startup order.
- Added `PLAN.md` as active execution queue.
- Added `PROGRESS.md` as dated implementation ledger.
- Added `HANDOFF.md` as next-session baton pass.

Verification:
- Documentation edits reviewed for non-overlapping responsibilities.
- No code-path changes made in `src/`.

Next action:
- Start implementation of MrVI DA from `FeaturePlanMrvi.md` and update this log with concrete code/test results.

## 2026-05-05

### Second-pass docs/vignette/README consistency review and cleanup
Status: completed

What changed:
- Removed non-existent notebook entries from `docs/index.md` toctree:
	- dropped `notebooks/dialogue_multigroup_vignette`
	- dropped `notebooks/iri_days_vignette`
- Updated `scripts/smoke_vignettes.py` vignette mapping and header rationale to reflect only notebooks currently tracked in `docs/notebooks`.
- Removed `toc.not_readable` and `toc.not_included` from `docs/conf.py` `suppress_warnings` so missing notebook/toctree issues are surfaced instead of silently ignored.
- Added `src/` to the Sphinx import path in `docs/conf.py` and switched API autodoc targets in `docs/api.md` to the public class path (`spVIPESmulti.model.spVIPESmulti`) for more robust documentation references.

Verification:
- Searched for stale notebook references in the updated paths (`README.md`, `docs/`, `scripts/smoke_vignettes.py`) and confirmed no remaining hits for `dialogue_multigroup_vignette` / `iri_days_vignette`.
- Checked editor diagnostics for changed files:
	- `docs/index.md`: no errors
	- `docs/conf.py`: no errors
	- `scripts/smoke_vignettes.py`: no errors
- Ran `make -C docs html`: still fails in this environment during autosummary import resolution (`no module named spVIPESmulti.model`), indicating a remaining docs-build environment/import-path issue beyond the stale toctree cleanup.

Next action:
- Optional follow-up: either (a) generate and commit the missing IRI/dialogue notebooks if they are still intended deliverables, or (b) keep docs/smoke mapping aligned to only committed notebooks.

### Follow-up: content-accuracy-only cleanup (build deferred)
Status: completed

What changed:
- Updated `scripts/smoke_vignettes.py` prose and mapping labels to remove OT-cluster/OT-paired claims not represented by the script's executed `CASES`.
- Kept focus on documentation/reporting correctness only; no additional docs build troubleshooting performed in this follow-up pass.

Verification:
- Re-searched key prose files for OT strategy wording used by the outdated mapping and confirmed no remaining hits in `scripts/smoke_vignettes.py`, `README.md`, or `docs/*.md`.
- Checked diagnostics for `scripts/smoke_vignettes.py`: no errors.

Next action:
- Defer docs build/import-path stabilization to a later dedicated pass, per current instruction.
