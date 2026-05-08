# HANDOFF.md

Purpose: next-session bootstrap in under one minute.

Read order: HANDOFF.md → PLAN.md → PROGRESS.md → ImplementationPlan.md (relevant section only)

---

## Performance improvements completed (2026-05-08)

- **P-PERF-1**: `_label_based_poe` reassembly vectorized — eliminates O(n_cells) GPU–CPU syncs.
- **VAL-GATE**: `_validate_likelihood_observations` isfinite/non-negative scans gated behind `validate_observations=False` default.
- **DL-WORKERS**: `num_workers` exposed in `train()` (default 0, backward-compatible).
- All 168 non-evaluate tests pass.

Next performance items still in backlog: P-PERF-2 (low-rank mixer), P-PERF-3 (torch.compile — ruled out by user), P-PERF-4 (SiLU).

---

## Current State (2026-05-08)

### Baseline run complete
- Trained 400 epochs (early stopping did NOT fire — validation loss still improving at epoch 400).
- Model saved to `results/spvipes_bcells_recommended_v3`. ✅

### Baseline metrics (z_shared)
| metric     | value | verdict      |
|-----------|-------|--------------|
| iLISI     | 1.904 | OK — groups mix |
| cLISI     | 2.347 | BAD — want ≈1 |
| kBET      | 0.894 | OK            |
| knn_purity| 0.517 | BAD           |
| leiden_ari| 0.312 | BAD           |

Worst cell types (k-NN purity): Activated MZ (0.165), Transitional (0.169), Pre-Plasmablast (0.191).
Best-dim AUROC for Activated MZ = 0.585 (≈ random → essentially invisible in shared space).
Root cause: class imbalance — Atypical (n=3155), Activated (n=2284) dominate shared space.

### Pilot sweep: COMPLETE
Results in `scripts/pilot_results_celltype.json`. Winner: **Variant A** (`label_shared=4`).
- knn_purity: 0.535 (+6.3% vs baseline 0.472)
- leiden_ARI: 0.380 (+13.8% vs baseline 0.242)
- iLISI/kBET unchanged (integration quality preserved)

### v4 retrain: RUNNING (PID 1028229, started 2026-05-08)
- `LABEL_SHARED_W = 4` (Variant A promoted)
- Save path: `results/spvipes_bcells_recommended_v4`
- `MAX_EPOCHS = 400`

## Immediate Next Action

**v4 retrain is RUNNING** (PID 1028229, started 2026-05-08).
Log: `tail -f /tmp/retrain_v4.log`

When done:
1. Check `results/spvipes_bcells_recommended_v4/` exists.
2. Check integration metrics in notebook output (knn_purity, leiden_ARI, cLISI).
3. Compare against v3 baseline in HANDOFF §Baseline metrics table.
4. Update PROGRESS.md with v4 results.
