# F3 Orthogonality Loss Audit

- run_id: `f3_orthogonality_20260513T120104Z`
- recommendation_json_verdict: `reject`
- decision: `archive`
- recommended_weight: `None`
- reason: no tested F3 weight passed all published gates
- default policy: keep `orthogonality_weight=0.0` in all presets

## Gate Summary

| weight | ortho reduction | recon worse | iLISI worse | kBET worse | cLISI worse | kNN purity worse | max failing gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.01 | 14.97% | 0.00% | 0.01% | 0.49% | 0.00% | 0.00% | orthogonality reduction \<20%; orthogonality CV 0.376 |
| 0.05 | 18.79% | 0.29% | 0.00% | 0.49% | 0.00% | 0.00% | orthogonality reduction \<20% |
| 0.10 | 41.99% | 0.00% | 0.00% | 0.00% | 3.21% | 2.50% | orthogonality CV 0.476 |
| 0.20 | -4.29% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | orthogonality reduction \<20%; orthogonality CV 0.320 |

## Decision

The real 3-seed Kang audit completed with all 15 rows present and `notes=ok`.
No tested nonzero weight satisfied both the reduction and cross-seed stability
requirements. F3 remains implemented for manual experiments but archived as
experimental/default-off evidence; do not promote a nonzero default.
