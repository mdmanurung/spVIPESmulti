# PLAN.md

Purpose: canonical active queue plus deferred backlog. This file should only describe work that is still active, pending, or blocked; completed work is tracked in PROGRESS.md and archived plans.

Status legend: `todo` | `in-progress` | `done` | `blocked`

---

## Current Iteration

### AUDIT-REMEDIATION: 2026-05-08 audit findings
Status: **done** (2026-05-09)
- All W-001..W-056 items implemented. See PROGRESS.md §2026-05-09 for full change log.
- Test result: **199 passed, 4 skipped, 0 failed** (`pytest tests/ -q`)
- Breaking changes: W-053 (LayerNorm replaces BatchNorm1d in Encoder; checkpoint-breaking), W-040 (kbet() returns rejection rate, not exp(-mean_chi2)).
- Remaining Q-### audit questions are intentionally deferred and are not part of the active queue unless explicitly reactivated.

### DOC-TUTORIAL-1: Modernize Tutorial.ipynb (simulated data vignette)
Status: **done** (2026-05-09)
- All 7 smoke test cases **PASS** on CPU (CUDA disabled; 22.5s total). Validates core API compatibility.
- Focused regression tests also pass on CPU: `tests/test_lightning_trainer_compat.py` (3/3) and `tests/test_utils.py -k PlotLatentDimensionStatsCompatibility` (3/3).
- End-to-end notebook execution completed successfully via CPU-only nbconvert:
	- `CUDA_VISIBLE_DEVICES='' SCVI_DISABLE_CUDA=true python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 docs/notebooks/Tutorial.ipynb --output-dir /tmp`
	- Output written: `/tmp/Tutorial.ipynb`
	- Wall-clock runtime: `real 173m58.818s`
- Technical note: HPC driver (v12.0.80) is incompatible with PyTorch cu130; CPU execution is the validated fallback path.

### DOC-LEGACY-1: Legacy spVIPES reproduction vignette (Phase 1 — qualitative)
Status: **in-progress** (execution active on 2026-05-09)
- Builder: `scripts/build_legacy_reproduction_notebook.py` → `docs/notebooks/legacy_spVIPES_reproduction.ipynb` (31 cells).
- Scope: 2-group RNA-only label-based PoE on Splatter sim ([Zenodo 10070301](https://zenodo.org/records/10070301)) with all post-spVIPES additions disabled (`disentangle_preset="off"`, `use_nf_prior=False`, `use_jeffreys_integ=False`, `group_loss_weights=None`).
- Config mirrors original tutorial: `n_dim_shared=10`, `n_dim_private=7`, `n_hidden=128`, `dropout=0.1`, `batch_size=128`, `train_size=1.0`, `max_epochs=400`, no early stopping, no KL warmup.
- Notebook training cell re-synced with the builder/config (`train_size=1.0`, `n_epochs_kl_warmup=0`) after a drifted in-notebook edit.
- API substitution: original `transport_plan_key='transport_plan'` (OT-paired) → `label_key='Celltypes'` (label-based) — closest supervised analogue.
- Training path now has a Lightning 2.6.x no-validation compatibility fix in `PatchedTrainRunner`, which unblocks the `train_size=1.0` notebook execution branch.
- Data: notebook downloads `splatter_simulation-2.h5ad` (~1 GB) from Zenodo on first run, caches in `data/`.
- Model cached to `results/spvipes_legacy_reproduction/`.
- Acceptance: shared UMAP separates `Celltypes`; per-group private UMAPs separate `Gene_programs` within each dataset.
- Active execution command:
	- `CUDA_VISIBLE_DEVICES='' SCVI_DISABLE_CUDA=true python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 docs/notebooks/legacy_spVIPES_reproduction.ipynb --output-dir /tmp`
- Runtime note: nbconvert emitted a non-fatal `MissingIDFieldWarning` about cell id normalization; execution continues.

### DOC-LEGACY-2: Quantitative parity benchmark (Phase 2 — deferred)
Status: **deferred**
- Goal: report side-by-side metrics (ARI / NMI / silhouette on `Celltypes` for shared; on `Gene_programs` for each private latent) between original `spVIPES` and `spVIPESmulti` legacy mode.
- Reactivation trigger: Phase 1 vignette executes cleanly **and** there is demand for a quantitative claim of equivalence in the README/paper.
- Blocker for activation: requires a working `nrclaudio/spVIPES` install (Python 3.9 env, scvi-tools 0.x) or a frozen pickle of its outputs on the same Splatter file.
- Not active until DOC-LEGACY-1 is complete and the parity question is explicitly reopened.

## Blockers / Decisions Needed

- No blocking decision currently required.
- Active execution path is base Python 3.10 + CPU-only notebook runs.
- Remaining dependency is only wall-clock time until `legacy_spVIPES_reproduction.ipynb` completes.

---

## Deferred Backlog

Rules: every item needs deferral reason and reactivation trigger. Move to Current Iteration before coding.
Full implementation specs for all items live in ImplementationPlan.md.

### P-PERF-1. Vectorize `_label_based_poe` reassembly
Status: **done** (2026-05-08)
→ See PROGRESS.md for implementation detail.

---

### P-PERF-2. Low-rank mixer in `LinearDecoderSPVIPE`
Status: **done** (2026-05-08) — default `use_low_rank_mixer=True`, `rank=4`. Ablation (`scripts/ablate_low_rank_mixer.json`) shows rank=4 outperforms full mixer on knn_purity (+6%), leiden_ARI (+31%), cLISI (-6.5%) at 60× fewer mixer params (45K vs 2.7M). See PROGRESS.md.

---

### P-PERF-3. `torch.compile`
Status: **cancelled** (2026-05-08) — dropped from backlog per user decision.

---

### P-PERF-4. SiLU activation in encoder
Status: **done** (2026-05-08) — `Encoder.encoder_activation` default is `"silu"`, configurable to `"relu"`/`"leakyrelu"`. See `src/spVIPESmulti/nn/networks.py` line 66.

---

### N5-D. Fix adversarial overreach on z_private
Status: **done** (2026-05-08)
→ See PROGRESS.md.

---

### N5-E. Class-weighted CE for minority cell types
Status: **done** (pre-existing implementation confirmed 2026-05-08)
→ Weights computed at model init, registered as buffer, threaded into CE calls.

---

### P6. Multi-covariate generalization
Status: **cancelled** (2026-05-08) — dropped from backlog per user decision.

### Reactivation Checklist

- [ ] Item moved to Current Iteration with explicit scope.
- [ ] User-facing API and backward-compatibility story defined.
- [ ] Tests and smoke validation commands defined before coding.
- [ ] Success metrics and stop criteria explicit.
