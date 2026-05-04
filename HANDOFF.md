# HANDOFF.md

Purpose: short baton pass for the next fresh Copilot session.

Read this first, then read `PLAN.md` and `PROGRESS.md`.

---

## Current State (2026-05-04)

- Documentation roles were intentionally separated to avoid overlap.
- No model code changes were made yet for MrVI DA.
- Next implementation target is MrVI-style differential abundance.

## What To Do Next

1. Open `FeaturePlanMrvi.md` and confirm/lock the DA return object.
2. Implement posterior plumbing and DA APIs in `src/spVIPESmulti/model/spvipesmulti.py`.
3. Add tests in `tests/test_differential_abundance.py`.
4. Run validation commands and log outcomes in `PROGRESS.md`.
5. Update `PLAN.md` statuses.

## Critical Constraints

- Warn users when DA is run without shared-latent alignment settings.
- Avoid modifying training/loss logic for this feature.
- Keep backward compatibility for existing setup/inference paths.

## Validation Commands

- `pytest tests/test_differential_abundance.py -v`
- `pytest -v`
- `python scripts/smoke_vignettes.py`

## If You Are Out Of Time

- Leave `PLAN.md` and `PROGRESS.md` updated with exact stop point.
- Record any open questions in `FeaturePlanMrvi.md` under Open Questions.
