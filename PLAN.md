# PLAN.md

Purpose: canonical active queue plus deferred backlog. This file should only describe work that is still active, pending, or blocked; completed work is tracked in PROGRESS.md and archived plans.

Status legend: `todo` | `in-progress` | `done` | `blocked`

---

## Current Iteration

<<<<<<< HEAD
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
=======
No active package-code item.
→ See PROGRESS.md for L1 (keyed layers, 2026-05-07) and M2 (multimodal alignment hardening, 2026-05-07).

Parallel external work (not owned in this session):
- N5 malaria B-cell latent-retuning pilot sweep (see HANDOFF.md).
>>>>>>> 128dc0d (notebook fixes)

## Blockers / Decisions Needed

- No blocking decision currently required.
- Active execution path is base Python 3.10 + CPU-only notebook runs.
- Remaining dependency is only wall-clock time until `legacy_spVIPES_reproduction.ipynb` completes.

---

## Deferred Backlog

Rules: every item needs deferral reason and reactivation trigger. Move to Current Iteration before coding.
Full implementation specs for all items live in ImplementationPlan.md.

### P-PERF-1. Vectorize `_label_based_poe` reassembly
<<<<<<< HEAD
Status: **done** (2026-05-08)
→ See PROGRESS.md for implementation detail.
=======
Status: Deferred | Priority: HIGH
Deferral reason: needs regression coverage before touching hot-path forward code.
Reactivation trigger: any training-speed work session.
→ Full spec: ImplementationPlan.md §P-PERF-1.
>>>>>>> 128dc0d (notebook fixes)

---

### P-PERF-2. Low-rank mixer in `LinearDecoderSPVIPE`
<<<<<<< HEAD
Status: **done** (2026-05-08) — default `use_low_rank_mixer=True`, `rank=4`. Ablation (`scripts/ablate_low_rank_mixer.json`) shows rank=4 outperforms full mixer on knn_purity (+6%), leiden_ARI (+31%), cLISI (-6.5%) at 60× fewer mixer params (45K vs 2.7M). See PROGRESS.md.

---

### P-PERF-3. `torch.compile`
Status: **cancelled** (2026-05-08) — dropped from backlog per user decision.
=======
Status: Deferred | Priority: MEDIUM
Deferral reason: architecture change; requires ablation to confirm quality is preserved.
Reactivation trigger: after P-PERF-1 done and profiled.
→ Full spec: ImplementationPlan.md §P-PERF-2.

---

### P-PERF-3. `torch.compile` (blocked on P-PERF-1)
Status: Deferred — blocked | Priority: LOW-MEDIUM
Deferral reason: graph-breaks on `.item()` loop until P-PERF-1 is done.
Reactivation trigger: P-PERF-1 complete and validated.
→ Full spec: ImplementationPlan.md §P-PERF-3.
>>>>>>> 128dc0d (notebook fixes)

---

### P-PERF-4. SiLU activation in encoder
<<<<<<< HEAD
Status: **done** (2026-05-08) — `Encoder.encoder_activation` default is `"silu"`, configurable to `"relu"`/`"leakyrelu"`. See `src/spVIPESmulti/nn/networks.py` line 66.
=======
Status: Deferred | Priority: LOW
Deferral reason: minor change, no urgency.
Reactivation trigger: any encoder-touching session.
→ Full spec: ImplementationPlan.md §P-PERF-4.
>>>>>>> 128dc0d (notebook fixes)

---

### N5-D. Fix adversarial overreach on z_private
<<<<<<< HEAD
Status: **done** (2026-05-08)
→ See PROGRESS.md.
=======
Status: Deferred | Priority: MEDIUM
Deferral reason: defer until pilot winner confirmed (Phase 4).
Reactivation trigger: after Phase 3 retrain (v4 model).
→ Full spec: ImplementationPlan.md §N5-D.
>>>>>>> 128dc0d (notebook fixes)

---

### N5-E. Class-weighted CE for minority cell types
<<<<<<< HEAD
Status: **done** (pre-existing implementation confirmed 2026-05-08)
→ Weights computed at model init, registered as buffer, threaded into CE calls.
=======
Status: Deferred | Priority: MEDIUM
Deferral reason: module surgery; defer until pilot results confirm direction.
Reactivation trigger: after Phase 3 retrain (v4 model).
→ Full spec: ImplementationPlan.md §N5-E.
>>>>>>> 128dc0d (notebook fixes)

---

### P6. Multi-covariate generalization
<<<<<<< HEAD
Status: **cancelled** (2026-05-08) — dropped from backlog per user decision.
=======
Status: Deferred | Priority: LOW
Deferral reason: broad metadata and architecture refactor across data/model/loss.
Reactivation trigger: after single-covariate stability and API simplification.
→ Full spec: ImplementationPlan.md §P6.
>>>>>>> 128dc0d (notebook fixes)

### Reactivation Checklist

- [ ] Item moved to Current Iteration with explicit scope.
- [ ] User-facing API and backward-compatibility story defined.
- [ ] Tests and smoke validation commands defined before coding.
- [ ] Success metrics and stop criteria explicit.
