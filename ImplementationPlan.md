# ImplementationPlan.md

Purpose: medium-horizon implementation roadmap (what to build next and why).
This file is not a session log and not a deferred-ideas archive.

How to use this file:
- Keep 3 to 8 active candidates only.
- Each item must be implementation-ready enough to estimate effort.
- Move completed execution details to PROGRESS.md.
- Move long-term or speculative work to PLAN.md (Deferred Backlog section).

Related docs:
- PLAN.md: current execution queue for the next session(s)
- PROGRESS.md: dated execution history
- FeaturePlanMrvi.md: detailed design for the MrVI DA port
- PLAN.md (Deferred Backlog): deferred/iceboxed architecture ideas
- HANDOFF.md: short baton pass for the next fresh Copilot session

---

## Active Roadmap Candidates

### R1. Auto-infer group indices in public APIs (Priority: High, Effort: S)
Problem:
Repeated boilerplate for `group_indices_list` appears in tutorials and can drift
from registered anndata metadata.

Scope:
- Make `group_indices_list` optional in `train` and latent extraction flows.
- Add one helper in `utils` for explicit retrieval.

Likely files:
- `src/spVIPESmulti/model/spvipesmulti.py`
- `src/spVIPESmulti/model/base/training_mixin.py`
- `src/spVIPESmulti/utils.py`

Exit criteria:
- No tutorial requires manual `group_indices_list` construction.
- Existing explicit calls remain backward compatible.

### R2. One-call embedding API (Priority: High, Effort: S)
Problem:
Users repeatedly call latent extraction and storage as two separate steps.

Scope:
- Add `model.embed(...)` wrapper that computes and stores shared/private embeddings.

Likely files:
- `src/spVIPESmulti/model/spvipesmulti.py`
- `src/spVIPESmulti/utils.py`

Exit criteria:
- One documented call writes standard `obsm` keys.
- Existing API behavior is unchanged.

### R3. MrVI-style differential abundance (Priority: High, Effort: M)
Problem:
No label-free DA API currently exists.

Scope:
- Follow design in `FeaturePlanMrvi.md`.
- Introduce DA API, tests, and dependency updates.

Likely files:
- `src/spVIPESmulti/model/spvipesmulti.py`
- `pyproject.toml`
- `tests/test_differential_abundance.py`

Exit criteria:
- DA method returns stable output format with tests passing.
- Alignment-precondition warning is emitted when required.

### R4. Public evaluation API (Priority: Medium, Effort: S)
Problem:
Held-out evaluation is script-only and not part of model API.

Scope:
- Add `model.evaluate(...)` with held-out NLL and key diagnostics.

Likely files:
- `src/spVIPESmulti/model/spvipesmulti.py`
- `src/spVIPESmulti/metrics.py`

Exit criteria:
- Users can compute evaluation metrics without custom scripts.
- Unit tests validate return schema and finite outputs.

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

- 2026-05-04: Re-scoped this document to roadmap-only role.
