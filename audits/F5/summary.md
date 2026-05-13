# F5 Counterfactual Protocols Audit

- decision: `implemented`
- scope: donor/condition-aware protocol helpers layered on existing F2 encode/decode APIs
- artifact type: implementation and unit/integration validation; no quantitative Kang F5 benchmark run yet

## Implemented Protocols

| protocol | public helper | behavior |
|---|---|---|
| P1 | `private_swap_unmatched` | swaps source `z_private` into target cells while preserving target `z_shared` |
| P2 | `private_swap_label_matched` | swaps `z_private` only from source cells with matching registered `label_key` |
| P3 | `private_swap_stratified` | prefers label+donor/timepoint matches and records deterministic fallback counts |
| condition shift | `donor_condition_shift` | applies condition centroid shifts using within-donor deltas when available |

## Validation

- `conda run -n spvm python -m pytest tests/test_counterfactual_protocols.py -q`
  - `7 passed`
- `conda run -n spvm python -m pytest tests/test_counterfactual_basics.py tests/test_counterfactual_integration.py tests/test_counterfactual_diagnostics.py -q`
  - `18 passed`
- `conda run -n spvm python -m pytest tests/ -q`
  - `271 passed, 2 skipped`

## Notes

- The protocol layer keeps F2 outputs as associative decoder predictions, not causal claims.
- Multimodal counterfactual expansion, learned perturbation vectors, and cycle-consistency losses remain deferred.
- A shared F2 utility bug was fixed so encoded metadata reports global AnnData observation indices for every group.
