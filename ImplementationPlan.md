# ImplementationPlan.md

Purpose: medium-horizon roadmap specs for future candidates only.

Related docs:
- PLAN.md: active execution queue and deferred backlog.
- PROGRESS.md: dated execution history.
- HANDOFF.md: next-session bootstrap.

---

## Active Roadmap Candidates

None. All planned items (R1–R4) are complete. See PLAN.md Deferred Backlog for the candidate pool.

---

## Prioritization Rules

- Prefer S/M items that reduce repeated notebook boilerplate.
- Only one M/L feature should be active at a time.
- Any new public API must include tests and one example update.

## Verification Baseline

- `pytest -v`
- `python scripts/smoke_vignettes.py`
- `python scripts/validate_disentanglement.py` for disentanglement-adjacent changes

## Last Updated

- 2026-05-05: R3 (MrVI DA) and R4 (public evaluation API) removed — both complete. Implementation history in PROGRESS.md.
