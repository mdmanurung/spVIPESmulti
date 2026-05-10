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

**Master plan:** `FEATURE_ROADMAP.md` consolidates the previous disentanglement and
counterfactual planning docs. Features are ordered by scientific readiness, each with
a TDD plan and a quantitative go/no-go benchmark.

**F1 status — closed.**

- Code path, tests, and training kwargs are complete.
- Targeted validation: `pytest tests/test_disentangle_metrics.py tests/test_multimodal_disentangle.py -q` -> `14 passed`.
- Kang overhead gate passed: disabled mean `0.5078 sec`, enabled mean `0.5001 sec`,
  overhead `-1.5164%`.
- Artifacts are under `audits/F1/`.

**F4-lite status — implementation/probe harness landed.**

- Added `condition_key`/`donor_key` registration and default-off donor/batch heads.
- Added covariate GRL scaling from the existing scvi `kl_weight` warmup and targeted coverage in `tests/test_covariate_heads.py`.
- Targeted validation: `pytest tests/test_covariate_heads.py tests/test_multimodal_disentangle.py tests/test_regression_fixes.py tests/test_multigroup_multimodal.py -q` -> `64 passed`.
- Added `scripts/benchmark_f4_covariate_probes.py`; smoke audit wrote `audits/F4/`.
- Kang default mapping uses `condition_key="label"` and `donor_key="replicate"`.
  No technical `batch_key` is known, so the standalone batch-shared rows are skipped
  unless a real technical-batch column is provided; the combined `full_bio` probe still
  runs the available donor heads.

**Next action — F4 promotion audit.**

Run the full 3-seed F4 probe matrix on Kang and summarize baseline deltas:

```bash
python scripts/benchmark_f4_covariate_probes.py \
  --run-id f4_kang_probes_<date> \
  --kang-h5ad-path docs/notebooks/data/kang_2018.h5ad \
  --seeds 0,1,2 \
  --max-epochs 40
```

### Previous Context
- Pilot sweep: See PLAN.md for status
- Performance optimizations (P-PERF-1 done, P-PERF-2–4 deferred)
- Disentanglement roadmap: N5-D, N5-E (Phase 4 items)

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
