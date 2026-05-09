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
Status: **blocked** (2026-05-09)
- 45-cell rewrite complete; px_scale KeyError fixed (added `px_scale` to generative output).
- Bug fix: `traversal.py` `cat_args` condition `n_batch > 1` → `n_batch > 0` (FCLayers expects 1 cat arg even for single-batch; `n_cat_list=[1]` set at init when `n_batch > 0`).
- Compatibility fix: `PatchedTrainRunner` now bypasses Lightning's datamodule path when `n_val==0`, using explicit `train_dataloaders` so `train_size=1.0` no longer hits `MultiGroupDataSplitter.val_dataloader() -> None` under Lightning 2.6.x.
- Independent hardening: `spVIPESmulti.pl.plot_latent_dimension_stats` now accepts both the new `is_collapsed` and legacy `is_vanished` columns, preventing notebook failures when old latent-stats tables are encountered.
- Re-execution was attempted with the configured environment via:
	- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m jupyter nbconvert --to notebook --execute --inplace docs/notebooks/Tutorial.ipynb > tutorial_rerun.log 2>&1`
- Terminal exited before completion; `tutorial_rerun.log` currently contains only the nbconvert start line.
- Result: notebook completion status could not be confirmed in the current environment.
- Completion criterion: run to clean completion and confirm no execution traceback in `tutorial_rerun.log`.

### DOC-LEGACY-1: Legacy spVIPES reproduction vignette (Phase 1 — qualitative)
Status: **blocked** (notebook generated 2026-05-09; end-to-end execution blocked by notebook runner environment)
- Builder: `scripts/build_legacy_reproduction_notebook.py` → `docs/notebooks/legacy_spVIPES_reproduction.ipynb` (31 cells).
- Scope: 2-group RNA-only label-based PoE on Splatter sim ([Zenodo 10070301](https://zenodo.org/records/10070301)) with all post-spVIPES additions disabled (`disentangle_preset="off"`, `use_nf_prior=False`, `use_jeffreys_integ=False`, `group_loss_weights=None`).
- Config mirrors original tutorial: `n_dim_shared=10`, `n_dim_private=7`, `n_hidden=128`, `dropout=0.1`, `batch_size=128`, `train_size=1.0`, `max_epochs=400`, no early stopping, no KL warmup.
- Notebook training cell re-synced with the builder/config (`train_size=1.0`, `n_epochs_kl_warmup=0`) after a drifted in-notebook edit.
- API substitution: original `transport_plan_key='transport_plan'` (OT-paired) → `label_key='Celltypes'` (label-based) — closest supervised analogue.
- Training path now has a Lightning 2.6.x no-validation compatibility fix in `PatchedTrainRunner`, which should unblock the `train_size=1.0` notebook execution branch once rerun.
- Data: notebook downloads `splatter_simulation-2.h5ad` (~1 GB) from Zenodo on first run, caches in `data/`.
- Model cached to `results/spvipes_legacy_reproduction/`.
- Acceptance: shared UMAP separates `Celltypes`; per-group private UMAPs separate `Gene_programs` within each dataset.
- Execution attempts/results:
	- `python -m jupyter nbconvert ... > legacy_repro_rerun.log` failed with a traceback rooted in `importlib.metadata` entry-point discovery under the configured Python 3.13 environment.
	- `python -m nbconvert ... > legacy_repro_rerun_v2.log` started conversion but did not reach a confirmed terminal completion state before terminal exit.

### DOC-LEGACY-2: Quantitative parity benchmark (Phase 2 — deferred)
Status: **deferred**
- Goal: report side-by-side metrics (ARI / NMI / silhouette on `Celltypes` for shared; on `Gene_programs` for each private latent) between original `spVIPES` and `spVIPESmulti` legacy mode.
- Reactivation trigger: Phase 1 vignette executes cleanly **and** there is demand for a quantitative claim of equivalence in the README/paper.
- Blocker for activation: requires a working `nrclaudio/spVIPES` install (Python 3.9 env, scvi-tools 0.x) or a frozen pickle of its outputs on the same Splatter file.
- Not active until DOC-LEGACY-1 is complete and the parity question is explicitly reopened.

## Blockers / Decisions Needed

- Notebook smoke execution is currently blocked by instability in the configured Python 3.13 Jupyter/nbconvert environment (`legacy_repro_rerun.log` traceback, plus `pip` warning about invalid distribution `~orch`).
- Decision needed: confirm the execution environment to use for notebook smoke reruns (repair current env vs. run nbconvert from a known-good Python/Jupyter environment).

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
