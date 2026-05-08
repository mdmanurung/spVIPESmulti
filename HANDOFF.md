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

### Pilot sweep: RUNNING (PID 665287, started 2026-05-07 23:32)
Results → `scripts/pilot_results_celltype.{json,md}` when done.

→ Root-cause analysis and Phase 4 follow-up items: see PLAN.md §N5-D and §N5-E.

## Immediate Next Action

1. Wait for pilot to finish or check its output:
   ```bash
   tail -50 /tmp/pilot_run.log
   cat scripts/pilot_results_celltype.md  # once it exists
   ```

2. Promote winning variant using these pre-drafted notebook cell 10 changes:

   **Variant A** (label_shared ↑): change ONE line in cell 10:
   ```python
   LABEL_SHARED_W = 4   # was 2
   ```

   **Variant B** (Jeffreys ↓): add ONE kwarg to `spVIPESmulti(...)` call in cell 10:
   ```python
   jeffreys_integ_weight=0.2,   # was 0.5
   ```
   Also update notebook markdown table row for `jeffreys_integ_weight`.

   **Variant C** (HVG union): replace HVG cell (cell 6) body:
   ```python
   from spVIPESmulti.utils import highly_variable_genes_union
   highly_variable_genes_union(adata, group_key="antigen_specific", n_top_genes=3000)
   adata = adata[:, adata.var["highly_variable"]].copy()
   print(f"HVG union: {adata.n_vars} genes")
   ```

3. After promoting: bump save path to `results/spvipes_bcells_recommended_v4`, set
   `MAX_EPOCHS=400`, and run full retrain via nbconvert.
