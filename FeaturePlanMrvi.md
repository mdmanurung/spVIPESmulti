# FeaturePlanMrvi.md

Purpose: single-feature implementation specification for MrVI-style
differential abundance (DA) in spVIPESmulti.

How to use this file:
- Keep it specific to this feature only.
- Track implementation state with checkboxes and date stamps.
- Record design decisions here; record execution history in PROGRESS.md.

---

## Feature Summary

- Feature: label-free differential abundance in shared latent space.
- Target API: model methods on `spVIPESmulti` plus tests.
- Design basis: MrVI-style posterior aggregation and log-density ratio scoring.
- Current state: Not started.

## Why This Feature Exists

MrVI computes DA by aggregating per-cell posterior distributions into
sample-level and then group-level mixtures. The equivalent latent in this
project is post-PoE `z_shared`, allowing a label-free DA workflow based on
posterior geometry rather than explicit cell-type labels.

## Non-Negotiable Precondition

spVIPESmulti uses per-group shared encoders. Without an active alignment
mechanism (`disentangle_group_shared_weight > 0` or `use_jeffreys_integ=True`),
encoder-specific placement can bias DA. The DA API must emit a warning when
both are off.

## Scope

In scope:
- Expose shared posterior parameters (`loc`, `scale`) from existing inference flow.
- Add sample-aware posterior aggregation and DA scoring APIs.
- Add tests for correctness, filtering behavior, and precondition warning.
- Add required dependency updates if needed by return format.

Out of scope:
- Differential expression.
- Outlier cell-sample pair detection.
- Local sample distances.
- Any training or loss-function change.

---

## Implementation Checklist

### A. Data and posterior plumbing
- [ ] Extend model batch processing to collect shared posterior `loc` and `scale`.
- [ ] Extend result formatting to expose both original and reordered arrays.
- [ ] Add optional `sample_key` registration in `setup_anndata`.

### B. Public API
- [ ] `get_shared_posterior(...)`
- [ ] `get_aggregated_posterior(...)`
- [ ] `differential_abundance(...)`
- [ ] Add precondition warning for weak/no alignment configuration.

### C. Dependencies and typing
- [ ] Add runtime dependency if required by output object.
- [ ] Ensure method docstrings include expected dimensions and semantics.

### D. Tests
- [ ] New test module for DA behavior.
- [ ] Verify sign behavior under synthetic covariate shift.
- [ ] Verify output size matches cell count.
- [ ] Verify sample-subset filtering.
- [ ] Verify alignment precondition warning.

### E. Validation
- [ ] `pytest tests/test_differential_abundance.py -v`
- [ ] `python scripts/smoke_vignettes.py`
- [ ] Optional qualitative check in CINEMA-OT vignette.

---

## Planned File Touches

- `src/spVIPESmulti/model/spvipesmulti.py`
- `src/spVIPESmulti/__init__.py` (no new re-export expected)
- `tests/test_differential_abundance.py` (new)
- `pyproject.toml` (if dependency additions are needed)

## Design Decisions

- Unit of comparison: per-sample aggregation, then per-group average.
- DA result shape: per-cell scores indexed by comparison covariate.
- API location: methods on the model class, not a separate module.

## Open Questions

- Confirm final output object (`xarray.Dataset` vs DataFrame + metadata).
- Confirm naming convention for sample registry key in setup manager.

## Last Updated

- 2026-05-04: Reformatted into feature-specific execution spec with checklist.
