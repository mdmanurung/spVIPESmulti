# HANDOFF.md

Purpose: next-session bootstrap in under one minute.

Read order: HANDOFF.md → PLAN.md → PROGRESS.md → ImplementationPlan.md (relevant section only)

---

## Environment Guard (2026-05-13)

- Use the Jupyter kernel `Python (spvm)` for notebooks in this repo. It points at
  `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/spvm/bin/python`
  and sets `PYTHONNOUSERSITE=1`.
- Avoid the default `Python 3 (ipykernel)` kernel here; it points at a global Python
  3.10 interpreter and can import `scvi`/`lightning`/`torchmetrics`/`torchvision` from
  `~/.local`, causing `RuntimeError: operator torchvision::nms does not exist`.
- Repo backstops are now in place:
  - `src/sitecustomize.py` removes user-site paths when the editable `src` directory is
    on `sys.path` and propagates `PYTHONNOUSERSITE=1` to child Python processes.
  - `spVIPESmulti._siteguard` runs before package-level heavy imports and raises a
    clear restart instruction if risky packages were already loaded from user-site.
- Escape hatch for intentional debugging only: `SPVIPESMULTI_ALLOW_USER_SITE=1`.

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

**F4-lite status — implemented and audited.**

- Added `condition_key`/`donor_key` registration and default-off donor/batch heads.
- Added covariate GRL scaling from the existing scvi `kl_weight` warmup and targeted coverage in `tests/test_covariate_heads.py`.
- Targeted validation: `pytest tests/test_covariate_heads.py tests/test_multimodal_disentangle.py tests/test_regression_fixes.py tests/test_multigroup_multimodal.py -q` -> `64 passed`.
- Added `scripts/benchmark_f4_covariate_probes.py`; the full 3-seed Kang probe audit
  wrote `audits/F4/` and rejected preset promotion.
- Updated `docs/notebooks/kang_ifn_commit_old.ipynb` to use the implemented F1/F4-lite
  APIs: condition/donor registration, opt-in donor heads, orthogonality metric logging,
  reordered latent extraction, and notebook-local held-out probe reporting.
- Notebook validation: JSON parses, code cells syntax-parse except the intentional
  IPython help cell `?model.train`, and `pytest tests/test_covariate_heads.py -q`
  -> `12 passed`.
- Kang default mapping uses `condition_key="label"` and `donor_key="replicate"`.
  No technical `batch_key` is known, so the standalone batch-shared rows are skipped
  unless a real technical-batch column is provided; the combined `full_bio` probe still
  runs the available donor heads.
- Current recommendation: keep F4 heads and `minimal_safe_bio` / `full_bio` available
  for opt-in/manual experiments only; do not present those presets as recommended.

**F2/F10a status — implemented and validated.**

- `spVIPESmulti.interventions` now provides the additive, single-modal
  counterfactual API: deterministic posterior-mean encoding, centroid-shift latent
  edits, direct decoder rollout, OOD/realism flags, diagnostics, and explicit
  multimodal rejection.
- F10a internal CellDISECT-style metric helpers live under
  `spVIPESmulti.interventions.metrics` and cover Pearson, delta-Pearson, top-DE
  cosine, Wasserstein, CAG, MIG-proxy scores, and skipped-baseline artifact rows.
- Targeted validation: `pytest tests/test_celldisect_metric_parity.py tests/test_counterfactual_basics.py tests/test_counterfactual_integration.py tests/test_counterfactual_diagnostics.py -q`
  -> `24 passed`.

**F10b status — implemented and smoke-audited.**

- Added `scripts/benchmark_kang_celldisect_parity.py` as an optional audit harness
  that writes `audits/F10/metrics.csv`, `summary.md`, and `recommendation.json`.
- External CellDISECT is optional; unavailable packages are recorded as explicit
  skipped rows.
- Targeted validation: `pytest tests/test_celldisect_metric_parity.py tests/test_celldisect_parity_runner.py -q`
  -> `10 passed`.
- Kang smoke wrote `audits/F10/` plus
  `audits/kang_ifnb/f10b_smoke_20260513_f10b.md`; verdict is informational because
  `spVIPESmulti` rows are available and external CellDISECT is not installed.

**F3 status — implemented, default-off, smoke-audited.**

- Added `orthogonality_weight` to presets, model/module constructors, and loss
  aggregation; every existing preset keeps `orthogonality_weight=0.0`.
- Added differentiable single-modal and multimodal shared/private orthogonality loss
  plus `orthogonality_loss` logging when the weight is enabled. F1 metrics remain
  independently controlled by `compute_orthogonality_metric`.
- Added `tests/test_orthogonality_loss.py`, `tests/test_f3_benchmark.py`, and
  `scripts/benchmark_f3_orthogonality.py`.
- Fixed the recurring `spvm`/pytest import crash by normalizing inherited
  `CONDA_PREFIX` in the repo guards and avoiding duplicate `torchvision::nms`
  registration in `tests/conftest.py`.
- Validation: `pytest tests -q` -> `258 passed, 2 skipped`.
- Smoke audit wrote `audits/F3/smoke/`; verdict was `reject` for the tiny
  1-seed/2-epoch run, so F3 remains experimental/default-off.

**Next action — decide whether to run a real F3 multi-seed Kang audit.**

Use `scripts/benchmark_f3_orthogonality.py` with at least 3 seeds before recommending
any nonzero `orthogonality_weight`. Outputs and F10 artifacts remain audit evidence
only; no causal claims.

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
