# PROGRESS.md

Purpose: dated execution ledger of what has been implemented, validated, and decided.

How to use:
- Add concise milestone entries with verification evidence.
- Keep implementation detail here, not in PLAN.md or HANDOFF.md.
- Omit "Next action" footers for completed entries; active pointer lives in HANDOFF.md only.

---

## 2026-05-10 (roadmap consistency patch: F1 naming + F2/F10 sequencing)

### User-directed consistency updates applied
Status: completed (code + docs + tests alignment)

What changed:
- Standardized F1 canonical metric naming on `orthogonality_within_stratum`
  and `orthogonality_worst_stratum` in module logging.
- Removed legacy alias emission (`orthogonality_loss`, `orthogonality_worst`) to
  avoid reporting ambiguity.
- Updated multimodal orthogonality helper to return excluded-strata counts and
  propagate them into `orthogonality_excluded_strata`.
- Updated roadmap and plan sequencing per decision to keep F2 first and delay
  F10 activation until after F2 baseline APIs are in place.
- Clarified `audits/kang_ifnb/` as the general benchmark lane, with optional
  F10-specific parity mirrors under `audits/F10/`.
- Strengthened `tests/test_disentangle_metrics.py` to assert canonical
  orthogonality metrics in training history when enabled and absence when
  disabled.

Files:
- `src/spVIPESmulti/module/spVIPESmultimodule.py`
- `tests/test_disentangle_metrics.py`
- `FEATURE_ROADMAP.md`
- `PLAN.md`
- `audits/kang_ifnb/README.md`
- `PROGRESS.md`


## 2026-05-10 (Kang IFNB benchmark runner implementation)

### Append-only benchmark script for spVIPESmulti vs external baselines
Status: completed (implementation + syntax validation)

What changed:
- Added `scripts/benchmark_kang_ifnb.py` with a reproducible benchmark flow that:
  - loads Kang IFNB from `pertpy` (`pt.data.kang_2018()`),
  - removes megakaryocytes,
  - applies batch-aware HVG selection (`batch_key="label"`),
  - runs per-seed benchmark jobs,
  - appends one row per `(model, seed)` into `audits/kang_ifnb/metrics.csv`.
- Implemented model adapters for:
  - `spvipesmulti` (full train + latent extraction + metric computation),
  - `spvipes_original` (best-effort API-compatible adapter; logs unavailability
    or incompatibility as a row),
  - `contrastivevae` via `scvi.external.ContrastiveVI` (logs unavailability as a row).
- Added robust audit behavior: missing/unsupported baselines do not silently skip;
  they are recorded in `notes` so comparisons are traceable.

Metrics populated per row:
- integration: `iLISI`, `cLISI`, `kBET` (acceptance), `knn_purity`, `leiden_ari`
- separation/disentanglement: `silhouette_group`, `silhouette_label`,
  `orthogonality_within_stratum`, `orthogonality_worst_stratum`,
  `orthogonality_excluded_strata`
- auxiliary: `cycle_consistency_l2`, `identity_preservation`, training wall time,
  reconstruction and KL summary fields when available.

Documentation update:
- Extended `audits/kang_ifnb/README.md` with runnable command examples and behavior
  notes for missing baselines.

Verification:
- `python3 -m py_compile scripts/benchmark_kang_ifnb.py` -> passed.
- Editor diagnostics: no errors in `scripts/benchmark_kang_ifnb.py`.
- Dry-run execution attempted with `--models spvipesmulti --max-epochs 1` but
  blocked by upstream Kang dataset download in this environment (zero-byte
  `kang_2018.h5ad` via pertpy cache).

Follow-up hardening:
- Added `--kang-h5ad-path` fallback in `scripts/benchmark_kang_ifnb.py` to run
  against a local Kang copy when pertpy download is unavailable.

Files:
- `scripts/benchmark_kang_ifnb.py`
- `audits/kang_ifnb/README.md`
- `PLAN.md`


## 2026-05-10 (roadmap expansion: SysVI + CellDISECT)

### Feature roadmap broadened with external method-informed tracks
Status: completed (planning artifact update)

What changed:
- Extended `FEATURE_ROADMAP.md` with three new optional tracks grounded in external
  references:
  - `F8` optional SysVI-style shared-latent VampPrior track.
  - `F9` optional SysVI-style latent cycle-consistency regularizer track.
  - `F10` CellDISECT-aligned Kang benchmark and metric parity track.
- Added explicit CellDISECT anchor references in the reproducibility defaults, so
  Kang evaluations now include both integration and counterfactual comparisons
  against public CellDISECT protocol/material.
- Added a new shared-infrastructure subsection that captures the Kang parity setup
  extracted from CellDISECT tutorial/repro scripts:
  - standard covariates (`cell_type`, `condition`),
  - common training hyperparameters,
  - leave-one-cell-type-out and hard split families,
  - per-cell-type artifact outputs (`pearson`, `delta_pearson`, `emd`).
- Expanded the roadmap metric suite with CellDISECT-aligned measures:
  - Pearson(mean), delta-Pearson,
  - top-DE metrics,
  - Wasserstein distance,
  - CAG and MIG-style disentanglement diagnostics,
  - optional fairness probes.
- Added targeted validation command slots for the new tracks:
  - `tests/test_vampprior_shared.py`
  - `tests/test_latent_cycle_loss.py`
  - `tests/test_celldisect_metric_parity.py`
- Updated risk register with new risk/mitigation entries for VampPrior stability,
  cycle-loss over-correction, and external benchmark mismatch.

Planning synchronization:
- Updated `PLAN.md` current iteration scope to include F8-F10.
- Updated immediate sequencing: close F1 overhead gate, then activate F10 audit
  harness before implementing F8/F9.

Files:
- `FEATURE_ROADMAP.md`
- `PLAN.md`

Evidence basis used for roadmap expansion:
- SysVI module/docs (`scvi-tools`) for optional VampPrior and standardized latent
  cycle-consistency loss behavior.
- CellDISECT tutorial + reproducibility Kang scripts for counterfactual metrics,
  split strategy, and disentanglement evaluation concepts.


## 2026-05-10 (Kang IFNB benchmark audit scaffold)

### Audit folder and metric schema
Status: completed (tracking scaffold)

What changed:
- Added `audits/kang_ifnb/` as the append-only home for Kang IFNB benchmark runs.
- Documented the required baseline comparisons against current `spVIPESmulti`,
  original `spVIPES`, and `contrastiveVAE`.
- Created `audits/kang_ifnb/metrics.csv` with a stable column schema for future
  benchmark rows, including disentanglement-specific metrics:
  - `orthogonality_within_stratum`
  - `orthogonality_worst_stratum`
  - `orthogonality_excluded_strata`
  - `cycle_consistency_l2`
  - `target_decoder_realism`
  - `identity_preservation`

Files:
- `audits/kang_ifnb/README.md`
- `audits/kang_ifnb/metrics.csv`
- `FEATURE_ROADMAP.md`
- `PLAN.md`


## 2026-05-10 (F1 conditional orthogonality instrumentation)

### F1: Helper implementation, train wiring, and test validation
Status: completed (implementation + unit/integration test scope)

What changed:
- Implemented module-level orthogonality helpers in
  `src/spVIPESmulti/module/spVIPESmultimodule.py`:
  - `_within_stratum_corr_norm(...)`
  - `_within_stratum_corr_norm_multimodal(...)`
- Added optional orthogonality metric integration in `_compute_disentangle_losses(...)`
  with logging keys:
  - `orthogonality_loss`
  - `orthogonality_worst`
  - `orthogonality_excluded_strata`
- Added orthogonality configuration to module constructor:
  - `compute_orthogonality_metric`
  - `orthogonality_groupby_keys`
  - `orthogonality_min_cells_per_stratum`
- Wired training-time kwargs handling in
  `src/spVIPESmulti/model/base/training_mixin.py` so these arguments are consumed
  by model train flow instead of being forwarded to Lightning Trainer.

Test hardening/fixes:
- Updated `tests/test_disentangle_metrics.py` fixture and execution path:
  - fixed Poisson data generation (`np.random.poisson`),
  - ensured required `indices` obs field exists,
  - used canonical `prepare_adatas(...)` flow to populate expected uns metadata,
  - forced CPU execution for training tests (`accelerator="cpu", devices=1`),
  - disabled CUDA visibility in test module to avoid RNG-state CUDA init on
    incompatible driver environments,
  - normalized helper unpacking to match helper return shape.

Verification:
- `pytest tests/test_disentangle_metrics.py -q` -> `9 passed`.
- Static diagnostics show no file-level errors in modified files.

Files:
- `src/spVIPESmulti/module/spVIPESmultimodule.py`
- `src/spVIPESmulti/model/base/training_mixin.py`
- `tests/test_disentangle_metrics.py`
- `PLAN.md`

Follow-up required to close F1 feature gate:
- Run Kang IFN benchmark path with megakaryocyte exclusion and collect overhead
  delta vs baseline (hard gate: <= +5% wall time).
- Write artifacts under `audits/F1/` (`metrics.csv`, `summary.md`, `recommendation.json`).


## 2026-05-10 (second-pass disentanglement execution planning)

### D2: Actionable implementation playbook authored
Status: completed (planning artifact)

What changed:
- Converted the second-pass disentanglement audit into an execution-ready markdown plan with:
  - phased scope (minimal first merge vs ambitious follow-up),
  - exact work packages (WP1-WP7),
  - target files/modules for each change,
  - validation commands and acceptance criteria,
  - risk register and immediate next coding slice.
- Activated the plan in `PLAN.md` as current-iteration item `D2`.
- Repaired `PLAN.md` `A2` section so scope/success criteria are explicit again.

Files:
- `DISENTANGLE_SECOND_PASS_ACTION_PLAN.md`
- `PLAN.md`

Execution notes:
- Recommended first implementation step remains WP1 + WP2 (API/registry + presets),
  followed by module-side losses and orthogonality.

## 2026-05-08 (training performance — Tier 1)

## 2026-05-10 (second-pass audit planning)

### A2: Actionable second-pass audit plan authored and activated
Status: completed (planning); execution in progress

  - conditional orthogonality enforcement (shared vs private latent dependence),
  - donor/individual-aware condition counterfactual diagnostics.
- Added explicit promotion/reject gates, run matrix, required artifacts, and immediate execution step.
- Activated this workstream in `PLAN.md` as current iteration item `A2`.

Files:
- `audits/SECOND_PASS_AUDIT_PLAN.md`
- `PLAN.md`

Execution notes:
Status: completed
What changed:
- Replaced the per-cell Python loop (O(n_cells) GPU→CPU syncs via `.item()`) with a
  vectorized boolean-mask scatter: O(n_labels) on-GPU tensor ops.
- At batch_size=2048 × 5 groups this eliminates ~10,000+ GPU–CPU syncs per training step.
- No change to output values; semantics are identical.

- `src/spVIPESmulti/module/spVIPESmultimodule.py` (reassembly block in `_label_based_poe`)

### VAL-GATE: Gate `_validate_likelihood_observations` behind `validate_observations` flag
Status: completed
What changed:
- Added `validate_observations: bool = False` constructor parameter to `spVIPESmultimodule`.
  by default.
- `strict_likelihood_support` check is preserved as an unconditional opt-in — it only runs
  when `strict_likelihood_support=True` (already explicit) regardless of `validate_observations`.
- Users can enable observation validation for debugging:
  `spVIPESmulti(adata, validate_observations=True)` (via `**model_kwargs` passthrough).
Files:
- `src/spVIPESmulti/module/spVIPESmultimodule.py`

### DL-WORKERS: Expose `num_workers` in `train()`
Status: completed
What changed:
- Added `num_workers: int = 0` parameter to `MultiGroupTrainingMixin.train()`.
- Threaded through `MultiGroupDataSplitter` → `ConcatDataLoader` → `AnnDataLoader`.
- Default 0 is fully backward-compatible. Users on multi-core HPC nodes can set e.g.
  `model.train(..., num_workers=4)` to overlap data loading with GPU compute.


Verification:
- `pytest tests/ -q --ignore=tests/test_evaluate.py` → 168 passed, 1 skipped, 0 failures.

## 2026-05-07 (keyed layers support in data preparation)
### L1: Keyed per-group and per-modality layer selection

What changed:
- Implemented keyed `layers` support in `src/spVIPESmulti/data/prepare_adatas.py` for both
  preparation entry points:
  - `prepare_adatas(adatas, layers={group: layer_name_or_None})`
  - `prepare_multimodal_adatas(adatas, modality_likelihoods=..., layers={group: {modality: layer_name_or_None}})`
- Added shared validation helpers so requested layers are applied by group/modality before
  concatenation.
- The implementation is partial-mapping-friendly:
  - omitted groups/modalities fall back to the input object's existing `adata.X`,
  - explicit `None` also falls back to `adata.X`.
- Added clear validation errors for:
  - unknown group keys in `layers`,
  - unknown modality keys in nested multimodal `layers`,
  - requested layer names absent from `adata.layers`.
- Updated API docs to replace the old "reserved" placeholder description with the shipped keyed
  mapping API.

Files:
- `src/spVIPESmulti/data/prepare_adatas.py`
- `tests/test_multigroup_multimodal.py`
- `docs/api.md`

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_multigroup_multimodal.py -k "group_specific_layers or missing_group_layer or multimodal_group_modality_layers or missing_multimodal_layer" -q` passed (`4 passed`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_multigroup_multimodal.py tests/test_regression_fixes.py -q` passed (`45 passed`).

## 2026-05-07 (multimodal prep alignment hardening)

### M2: Validate multimodal within-group cell alignment before concat
Status: completed

What changed:
- Fixed `prepare_multimodal_adatas(...)` in `src/spVIPESmulti/data/prepare_adatas.py` so it no
  longer silently returns zero cells when modalities within the same group have different
  `obs_names`.
- Added explicit within-group modality validation before the axis=1 concat:
  - if modalities contain the same cells in a different order, they are realigned to the first
    modality's `obs_names`,
  - if modalities do not contain the same cells, the function now raises `ValueError` instead of
    dropping cells via an implicit inner join.
- Corrected the multimodal prefix-overlap regression test so it actually tests prefix overlap with
  aligned cells, rather than accidentally triggering the cell-alignment bug.
- Added a dedicated regression test asserting that mismatched multimodal `obs_names` raise a clear
  error.

Why it mattered:
- The previously reported `test_multimodal_overlapping_prefixes` failure was a stale diagnosis.
  Prefix-overlap bookkeeping for multimodal groups was already correct; the real defect was silent
  cell loss during the per-group multimodal concat when RNA/protein modalities used different
  `obs_names`.

Files:
- `src/spVIPESmulti/data/prepare_adatas.py`
- `tests/test_regression_fixes.py`

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_regression_fixes.py -k "overlapping_prefixes or mismatched_obs_names_raise" -q` passed (`3 passed`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_regression_fixes.py tests/test_multigroup_multimodal.py -q` passed (`41 passed`).

## 2026-05-07 (malaria B-cell latent retuning — Phase 1+2 setup)

### N5: Notebook instrumentation and pilot sweep scaffold
Status: baseline complete; pilot sweep running (PID 665287, started 2026-05-07 23:32)

What changed:
- `docs/notebooks/malaria_bcells_recommended.ipynb` (23 cells total, up from 19):
  - Fixed save path `results/spvipes_bcells_recommended_v1` → `results/spvipes_bcells_recommended_v3`.
  - Added **training-history summary cell** (cell 12, after train): prints first/final/drop/epoch
    count for reconstruction_loss_{train,validation}, elbo_train, kl_local_train; flags NaN.
  - Added **markdown header cell** (cell 17, after compute_shared_umap).
  - Added **integration-report cell** (cell 18): calls `spVIPESmulti.metrics.integration_report`
    with z_shared, antigen_specific groups, cluster_label cell-types, and per-group private
    latents; prints ilisi/kbet/clisi/knn_purity/leiden_ari/silhouette table.
  - Added **failure-mode audit cell** (cell 19): per-cell-type k-NN purity (k=20) sorted table,
    one-vs-rest AUROC per shared dimension per cell type, antigen×cluster_label crosstab.
- `scripts/pilot_celltype_separation.py` (new): runs 4 conditions (baseline + 3 variants) at
  150-epoch budget, outputs `scripts/pilot_results_celltype.{json,md}`.
  - Variant A: `disentangle_label_shared_weight=4.0` (↑ from 2.0)
  - Variant B: `jeffreys_integ_weight=0.2` (↓ from 0.5)
  - Variant C: `highly_variable_genes_union(group_key="antigen_specific", n_top_genes=3000)`

Context:
- Motivation: visible shared UMAP shows moderate cell-type separation but no quantitative
  evidence existed. Pilot targets the three most tractable root causes identified from
  training-config analysis: insufficient label supervision, over-strong integration pressure,
  and gene-set coverage gaps from global HVG selection.
- Acceptance gate: knn_purity and/or leiden_ari up, clisi down, ilisi/kbet stable (≤10% drop).

Files:
- `docs/notebooks/malaria_bcells_recommended.ipynb`
- `scripts/pilot_celltype_separation.py`

Verification (pending):
- Baseline complete: 400 epochs trained, early stopping did NOT fire, model saved to `results/spvipes_bcells_recommended_v3`.
- Pilot sweep running; results will appear in `scripts/pilot_results_celltype.{json,md}`.
- Root-cause analysis and Phase 4 follow-up specs: ImplementationPlan.md §N5-D and §N5-E.

---

## 2026-05-08 (performance audit)

### P-PERF: Third-pass code audit — training-speed bottlenecks identified
Status: completed (specs written; implementation deferred)

Findings:
- `_label_based_poe` reassembly loop issues ~16,384 GPU-CPU syncs per step (batch=2048, 8 groups). Fully vectorizable. → §P-PERF-1.
- `LinearDecoderSPVIPE` `mixture` layer is 296×1000 (296K params × 8 decoders). Low-rank factorization possible. → §P-PERF-2.
- `torch.compile` blocked on P-PERF-1 graph-break (`.item()` in hot path). → §P-PERF-3.
- `Encoder` uses `nn.ReLU()`; SiLU would improve convergence speed. → §P-PERF-4.

Files:
- `ImplementationPlan.md` (specs added)
- `PLAN.md` (stubs added to deferred backlog)

---

## 2026-05-05 (malaria notebook DoRothEA/PROGENy MLM + consensus)

### N4: Add MLM + consensus scoring for shared_15 with DoRothEA and PROGENy
Status: completed

What changed:
- Added a new analysis block in `docs/notebooks/malaria_bcells.ipynb` to score the ranked shared_15 loading vector with both `decoupler.mt.ulm` and `decoupler.mt.mlm`, then aggregate method evidence with `decoupler.mt.consensus`.
- Added reusable helpers for:
  - target-symbol harmonization and network filtering/deduplication (`prepare_network_for_ranked_input`),
  - method execution + consensus aggregation (`run_ulm_mlm_consensus`),
  - consistent summary-table construction (`summarize_method_outputs`).
- Ran the new method stack on:
  - DoRothEA TF network (`dc.op.dorothea(organism="human")`, confidence A/B/C),
  - PROGENy pathway network (`dc.op.progeny(organism="human")`).
- Added a focused DoRothEA TF readout table for `TBX21`, `BATF`, `IRF4`, `PAX5`, `RFX5`, `PRDM1`, and `BCL6`.
- Added a final dual-panel barplot cell showing consensus scores for top positive/negative DoRothEA TFs and PROGENy pathways.
- Removed temporary notebook probe/introspection cells used to debug decoupler `consensus` payload conventions.

Implementation notes:
- `decoupler.mt.consensus` with dict input expects keys containing `"score_"`; passing `{"score_ulm": ..., "score_mlm": ...}` resolved prior `list index out of range` failure.
- Network rows are deduplicated on `source,target` before scoring to prevent repeated-edge assertion failures.

Key results:
- DoRothEA retained `340` TFs (`9267` edges) after overlap filtering.
- Top positive DoRothEA consensus TFs: `SIX2`, `SOX10`, `TBX21`, `SPI1`, `LEF1`.
- Top negative DoRothEA consensus TFs (from the consensus plot): `PRDM14`, `ELF1`, `LYL1`, `BACH1`, `NR5A2`.
- Focus TF table indicates directional signal with `TBX21` positive and `PRDM1`/`RFX5` negative, but consensus-adjusted significance remains weak (high `consensus_padj`).
- PROGENy retained `14` pathways (`10732` edges).
- Top positive PROGENy consensus pathways: `Hypoxia`, `p53`; strongest negatives include `MAPK`, `Estrogen`, `PI3K`.

Files:
- `docs/notebooks/malaria_bcells.ipynb`

Verification:
- Executed the main DoRothEA/PROGENy ULM+MLM+consensus compute cell successfully in the live kernel.
- Executed the focused DoRothEA TF summary cell successfully.
- Executed the final consensus barplot cell successfully.

## 2026-05-05 (malaria notebook ranked programs + TF scoring)

### N3: Ranked shared_15 marker/pathway and CollecTRI TF analysis
Status: completed

What changed:
- Added a new ranked-program analysis block in `docs/notebooks/malaria_bcells.ipynb` that treats the full shared_15 loading vector as the input statistic vector.
- Added curated B-cell state programs (`atypical_b_cell`, `activated_b_cell`, `naive_memory_b_cell`, `plasma_b_cell`) and scored them with both `decoupler.mt.gsea` and `decoupler.mt.ulm`.
- Added Hallmark scoring with both `gsea` and `ulm` using `dc.op.hallmark(organism="human")`.
- Added a separate CollecTRI transcription-factor scoring block using `dc.op.collectri(organism="human")` and `decoupler.mt.ulm`, following the decoupler TF-scoring tutorial pattern but applied to the ranked shared_15 loading vector.
- Fixed the earlier Hallmark duplicate-edge failure by deduplicating `source`/`target` pairs before decoupler scoring and reran the stale failed cell successfully.

Key results:
- Curated B-cell marker analysis points strongly to the `atypical_b_cell` program for shared_15:
  - GSEA `2.137239`
  - ULM `6.809558`
- Hallmark results are moderate and led by `UV_RESPONSE_DN`, `XENOBIOTIC_METABOLISM`, `ANGIOGENESIS`, `APICAL_JUNCTION`, `IL2_STAT5_SIGNALING`, and `HEDGEHOG_SIGNALING`.
- CollecTRI TF scoring did not yield significant TFs after multiple-testing correction (`0` TFs at `padj < 0.05`), but the top positive scores include `TBX21`, `STAT3`, `GATA3`, and `MAF`, while selected B-cell TFs such as `PAX5`, `RFX5`, and `PRDM1` score negative.

Files:
- `docs/notebooks/malaria_bcells.ipynb`

Verification:
- Executed the new CollecTRI TF-scoring cell successfully in the live `scvi-test` kernel.
- Executed the focused TF summary cell and TF barplot cell successfully.
- Re-executed the ranked marker/Hallmark compute cell successfully after deduplicating Hallmark edges.
- Re-executed the ranked marker/Hallmark summary and plot cells successfully.

## 2026-05-05 (malaria notebook ImmuneSigDB follow-up)

### N2: ImmuneSigDB-only enrichment for shared_15
Status: completed

What changed:
- Updated the shared_15 enrichment notebook cell in `docs/notebooks/malaria_bcells.ipynb` to normalize loading-derived genes by explicitly removing the `Negative_` prefix before matching to external gene sets.
- Narrowed the enrichment resource from the mixed MSigDB subset to `immunesigdb` only, as a more appropriate immune-focused follow-up.
- Kept the enrichment query at the top 100 positive shared_15 loading genes.
- Simplified the follow-up bar plot cell to a single-color ImmuneSigDB view.

Key result:
- The ImmuneSigDB-only run produced weaker corrected signal than the broader MSigDB screen; top terms cluster around monocyte, dendritic-cell, and NK-related reference signatures.
- Top ranked terms include `GSE29618_MONOCYTE_VS_PDC_UP`, `GSE21774_CD62L_POS_CD56_BRIGHT_VS_CD62L_NEG_CD56_DIM_NK_CELL_UP`, `GSE30083_SP3_VS_SP4_THYMOCYTE_DN`, and `GSE39916_B_CELL_SPLEEN_VS_PLASMA_CELL_BONE_MARROW_DN`.
- The best hits all remained above `0.2` FDR, so this should be treated as directional annotation rather than strong pathway-level evidence.

Files:
- `docs/notebooks/malaria_bcells.ipynb`

Verification:
- Re-executed the shared_15 enrichment cell successfully in the live `scvi-test` kernel after the `Negative_` prefix normalization.
- Re-executed the ImmuneSigDB bar plot cell successfully in the live `scvi-test` kernel.

## 2026-05-05 (malaria notebook latent specificity)

### N1: Shared-latent celltype-specificity analysis for malaria B cells
Status: completed

What changed:
- Added a notebook analysis cell in `docs/notebooks/malaria_bcells.ipynb` that scores each shared latent dimension against each `cluster_label` using one-vs-rest AUROC and Cohen's $d$.
- Added two complementary summaries:
  - best shared dimension per cell type,
  - best-matching cell type for each shared dimension.
- Added an Atypical-focused ranking table and a follow-up box/strip plot cell for the top four Atypical-associated shared dimensions.
- Kept the workflow notebook-local and reused the existing stored shared embedding (`X_spVIPESmulti_shared`) without changing library code.

Key result:
- `Atypical` is most strongly distinguished by `shared_15` with specificity AUC `0.903941` and Cohen's $d = 1.897346`.
- Secondary Atypical-associated shared dimensions are `shared_10`, `shared_4` (low in Atypical), `shared_0`, and `shared_8`.

Files:
- `docs/notebooks/malaria_bcells.ipynb`

Verification:
- Executed the notebook analysis cell successfully in the live `scvi-test` kernel.
- Executed the Atypical top-dimension plot cell successfully in the live `scvi-test` kernel.
- Confirmed `cluster_label` contains `Atypical` and that the notebook stores the shared embedding under `X_spVIPESmulti_shared`.

## 2026-05-05 (model persistence utility)

### S1: Add script to save trained spVIPESmulti model
Status: completed

What changed:
- Added `scripts/save_spvipesmulti_model.py` as a notebook-friendly persistence helper.
- Exposed an importable function:
  - `save_spvipesmulti_model(model, output_dir, overwrite=False, save_anndata=True)`
- Added CLI mode intended for notebook `%run` usage:
  - resolves model from IPython namespace via `--model-var` (default `model_spv`),
  - saves to `--output-dir`,
  - supports `--overwrite` and `--no-save-anndata`.
- Implemented scvi-version-tolerant save dispatch by inspecting `model.save` signature and only forwarding supported kwargs (`overwrite`, `save_anndata`).

Files:
- `scripts/save_spvipesmulti_model.py`

Verification:
- Script syntax and argument wiring validated by static inspection.
- Runtime save path is delegated to `model.save(...)` from the active trained model object, matching scvi BaseModel save behavior.

## 2026-05-05 (unequal batch-size loss fix)

### U1: Robust loss aggregation for unequal per-group minibatch sizes
Status: completed

What changed:
- Fixed single-modal `loss()` aggregation in `spVIPESmultimodule` so group losses are reduced to scalars before cross-group accumulation.
- Fixed multimodal `_loss_multimodal()` aggregation similarly for per-modality losses and shared PoE KL terms.
- Updated `LossOutput` payload construction to remain shape-safe with unequal per-group sizes by:
  - storing scalar means in `reconstruction_loss` and `kl_local` dictionaries,
  - providing explicit `n_obs_minibatch`.
- Added focused regression tests that directly exercise unequal per-group batch lengths in both single-modal and multimodal loss paths.

Files:
- `src/spVIPESmulti/module/spVIPESmultimodule.py`
- `tests/test_regression_fixes.py`

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_regression_fixes.py -k "UnequalGroupBatchLossAggregation or unequal" -q` passed (`2 passed`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_lightning_trainer_compat.py tests/test_multimodal_disentangle.py tests/test_multigroup_multimodal.py -q` passed (`23 passed`).

## 2026-05-05 (documentation synchronization)

### D1: README/API/docs alignment with repository state
Status: completed

What changed:
  - added `sample_key` in `setup_anndata(...)` usage.
  - added optional sample-aware posterior aggregation and `differential_abundance(...)` usage in the basic workflow.
  - quick-reference table now includes `embed`, `get_shared_posterior`, `get_aggregated_posterior`, and `differential_abundance`.
  - `setup_anndata(...)` signature/table now documents `sample_key`.
  - `train(...)` docs now correctly show `group_indices_list` as optional (auto-inferred fallback).
  - `get_latent_representation(...)` docs now state `batch_size=None` defaults to `scvi.settings.batch_size`.
  - added dedicated sections for `embed`, `get_shared_posterior`, `get_aggregated_posterior`, and `differential_abundance`.
  - removed `notebooks/dialogue_multigroup_vignette` and `notebooks/iri_days_vignette` from `docs/index.md`.

Files:

Verification:

### Final pass: vignette index accuracy (malaria notebook discoverability)
Status: completed

What changed:
- Added `docs/notebooks/malaria_bcells.ipynb` to the Sphinx toctree in `docs/index.md`.
- Added a corresponding tutorial bullet to `README.md` under "Documentation & Tutorials".

Verification:
- Confirmed `malaria_bcells` is referenced in both index locations (`README.md` and `docs/index.md`).
- Checked diagnostics for modified files: no errors.

Next action:
- Keep build troubleshooting deferred; content indexing is now aligned with tracked notebook files.

## 2026-05-05 (audit session)

### Audit-driven bug fixes (8 issues)
Status: completed

Triggered by independent deep-code audit of the full codebase. All items verified with `pytest -q` → `174 passed, 1 skipped`.

#### B1 — `normalized=True` crash in `get_latent_representation`
- `_process_batches` only populated `latent_shared[g]` in the `if not normalized` branch; when `normalized=True` the list stayed empty, causing `torch.cat([])` in `_format_results`.
- Added the symmetric `else` branch for shared PoE latent (mirrors existing private latent handling).
- File: `src/spVIPESmulti/model/spvipesmulti.py`

#### Q1 — Gaussian likelihood correctness (Option B: per-feature heteroscedastic scale)
- `build_likelihood` was using `px_rate_shared` as the mean (ignoring the private/shared mixture) and a hardcoded `scale=0.1`.
- Added `self.log_scale_gaussian: nn.ParameterDict` in `spVIPESmultimodule.__init__`, one `Parameter(zeros(n_features))` per `(group, modality)` with likelihood `"gaussian"`.
- `_generative_multimodal` looks up the parameter and passes it as `log_scale` to `build_likelihood`.
- `build_likelihood` signature updated: `px_scale` (required for Gaussian — the mixed mean) and `log_scale` (required for Gaussian — per-feature log std). Scale is `exp(log_scale).clamp(min=1e-4).expand_as(mean)`.
- Updated `test_gaussian_likelihood` in `tests/test_multigroup_multimodal.py` to pass the new required args.
- Files: `src/spVIPESmulti/module/utils.py`, `src/spVIPESmulti/module/spVIPESmultimodule.py`, `tests/test_multigroup_multimodal.py`

#### Q2 — `get_loadings` multimodal support (Option B: full implementation)
- `module.get_loadings(dataset, type_latent)` now accepts `dataset` as `int` (single-modal) or `(group, modality)` tuple (multimodal) — decoder lookup works for both key shapes.
- `model.get_loadings()` detects `is_multimodal`; for multimodal it iterates `self.module.decoders` keys and returns dict keyed by `((group, modality), latent_type)` with `var_names` from `groups_modality_var_indices`; for single-modal keeps the existing `(i, latent_type)` key scheme.
- Files: `src/spVIPESmulti/module/spVIPESmultimodule.py`, `src/spVIPESmulti/model/spvipesmulti.py`

#### Q3 — Remove `cudnn.benchmark = True` global side effect (Option A)
- Deleted the module-level `torch.backends.cudnn.benchmark = True` line (line 18 of spVIPESmultimodule.py). It mutated global PyTorch state at import time, invisible to users.
- File: `src/spVIPESmulti/module/spVIPESmultimodule.py`

#### B4 — NF prior silently ignored for multimodal private KL
- `_loss_multimodal` always used standard Normal KL for per-modality private latents regardless of `use_nf_prior` / `nf_target`.
- Added `_nf_kl(qz_mod_private, z_mod_private, "private")` branch, mirroring single-modal `loss()` logic.
- File: `src/spVIPESmulti/module/spVIPESmultimodule.py`

#### B5 — Jeffreys integration loss silently ignored in single-modal
- `loss()` never called `_compute_jeffreys_integ_loss`; `use_jeffreys_integ=True` on a single-modal model had no effect.
- Added the `if self.use_jeffreys_integ:` block at the end of `loss()`, matching `_loss_multimodal`.
- File: `src/spVIPESmulti/module/spVIPESmultimodule.py`

#### B8 — `setup_anndata` used `print()` instead of `logger.info()`
- Replaced all 7 `print()` calls (including emoji) with `logger.info()` using `%s` formatting.
- File: `src/spVIPESmulti/model/spvipesmulti.py`

Verification:
- `pytest -q` → `174 passed, 1 skipped, 109 warnings` (57s)
- No new test failures introduced.

Known coverage gaps not yet addressed (test-only work, no bug):
- No test for `normalized=True` path in `get_latent_representation`.
- No test for `get_loadings` on a multimodal model.
- No test for `use_jeffreys_integ=True` on single-modal model.

## 2026-05-05 (hardening follow-up)

### H1: Regression coverage + likelihood support hardening
Status: completed

What changed:
- Added integration regression coverage for:
  - `get_latent_representation(normalized=True)` shape/completeness path.
  - `get_loadings()` on multimodal models with tuple-keyed `(group, modality)` decoders.
  - Single-modal `use_jeffreys_integ=True` path to verify Jeffreys contribution affects loss.
- Added optional strict likelihood support validation in module loss paths:
  - New module flag `strict_likelihood_support` (default `False` for backward compatibility).
  - Added `_validate_likelihood_observations(...)` checks before `log_prob` evaluation.
  - Baseline validation enforces finite values and non-negative observations for NB.
  - Strict mode additionally enforces integer-like NB counts when targets are not log-transformed.
- Added targeted regression test to ensure strict mode rejects fractional NB counts.

Files:
- `src/spVIPESmulti/module/spVIPESmultimodule.py`
- `tests/test_api_boilerplate_reduction.py`
- `tests/test_regression_fixes.py`

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_api_boilerplate_reduction.py tests/test_regression_fixes.py -q` passed (`25 passed`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -q` passed (`177 passed, 2 skipped`).

Notes:
- Existing warning-only support mismatches in synthetic/integration tests remain warning-level by default; strict enforcement is opt-in via `strict_likelihood_support=True`.

### H2: Public docs pass for strict likelihood validation + multimodal loadings
Status: completed

What changed:
- Documented `strict_likelihood_support` in README model-constructor examples and behavior notes.
- Added `strict_likelihood_support` to API constructor parameter table.
- Corrected outdated API note that claimed `get_loadings()` was single-modal only; docs now describe multimodal tuple-key output shape.

Files:
- `README.md`
- `docs/api.md`

Verification:
- Manual docs consistency check against current implementation in `spVIPESmulti.model.spvipesmulti.spVIPESmulti.__init__` and `spVIPESmulti.model.spvipesmulti.spVIPESmulti.get_loadings` completed.

---

## 2026-05-05

### R4: Public evaluation API second slice (held-out validation metrics)
Status: completed

What changed:
- Enabled validation execution by default in `train()` whenever a validation split exists by setting `check_val_every_n_epoch=1` unless the caller already overrides it.
- Fixed `PatchedTrainRunner` to pass the multi-group splitter as a Lightning datamodule, preserving validation dataloaders.
- Updated `MultiGroupDataSplitter` to expose aggregate `n_train` / `n_val` counts and to use evaluation-safe validation/test loaders (`shuffle=False`, `drop_last=False`).
- Updated `get_latent_representation()` to honor the documented default batch size when `batch_size=None`.
- Extended `model.evaluate(...)` to expose `held_out_metrics` from training history when available, including `held_out_nll` as an alias of `reconstruction_loss_validation`.
- Added focused tests for held-out metric extraction and for validation history logging during training.

Verification:
- Discriminating live check: one-epoch CPU training now reports non-empty validation batches and populates validation history keys (`elbo_validation`, `reconstruction_loss_validation`, `validation_loss`, ...).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_evaluate.py tests/test_lightning_trainer_compat.py -q` passed (`23 passed`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -v` passed (`173 passed, 2 skipped`).

Notes:
- `held_out_metrics` are sourced from training history on the model's registered AnnData, not recomputed for arbitrary external AnnData objects.
- This keeps the implementation minimal and consistent with the current training/evaluation architecture.

### R4: Public evaluation API (diagnostics-first)
Status: completed

What changed:
- Added `model.evaluate(...)` to `src/spVIPESmulti/model/spvipesmulti.py`.
- The API computes public, script-free diagnostics for the shared latent and optional private latents using the existing `integration_report(...)` metrics stack.
- `evaluate(...)` accepts either a precomputed shared embedding (`z_shared_key`) or falls back to `get_latent_representation(...)` when no embedding is present.
- Added clear metadata and informational warnings to make fallback paths explicit.
- Kept held-out NLL out of scope for this pass because validation-loss plumbing is still absent in the training path.
- Added focused unit coverage in `tests/test_evaluate.py` for return schema, embedding fallback, label handling, private-latent rows, and finite shared metrics.

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_evaluate.py -q` passed (`19 passed`, warnings only).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest tests/test_evaluate.py tests/test_enrichment.py -q` passed (`26 passed, 1 skipped`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -v` passed (`170 passed, 2 skipped`).

Notes:
- The repeated auto-inference warning seen in unit tests is expected because each dummy-model fixture is fresh and independently exercises the public fallback path.
- Held-out NLL remains a follow-up item for the R4 second slice, not a regression in this implementation.

### Inference-path speedup: remove duplicate _split_tensors_by_group call
Status: completed

What changed:
- In `_process_batches` (model/spvipesmulti.py), removed the redundant `per_group = self.module._split_tensors_by_group(tensors_by_group)` call that preceded `_get_inference_input`.
- `_get_inference_input` internally calls `_split_tensors_by_group` and already exposes `global_indices` in its return dict. The batch loop now reads `inference_inputs["global_indices"][g]` and `poe_log_z.shape[0]` for the fallback arange.
- Saves one full tensor split per batch iteration in every `get_latent_representation` / `embed` / `get_shared_posterior` call chain.
- `_process_all_cells_with_cycling` is confirmed dead code (not called from any path); left as-is.

Verification:
- `pytest tests/test_api_boilerplate_reduction.py tests/test_regression_fixes.py tests/test_differential_abundance.py tests/test_lightning_trainer_compat.py tests/test_nf_prior.py -v -q` passed (`32 passed`).

### Full regression + smoke validation
Status: completed

What changed:
- Executed repository-wide regression suite and smoke scenarios as required by handoff.
- Confirmed all MrVI DA additions remain stable under full-suite execution.

Verification:
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -v` passed (`151 passed, 2 skipped, 64 warnings`, `59.38s`).
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python scripts/smoke_vignettes.py` passed (`7/7` cases).

Notes:
- Smoke script ran on CUDA (`NVIDIA L40S`) and completed all configured cases.
- Remaining warnings are pre-existing environment/model warnings; no new regression failures observed.

## 2026-05-04

### R3: MrVI-style differential abundance
Status: completed

What changed:
- Added optional sample registration in setup path:
  - `spVIPESmulti.setup_anndata(..., sample_key=...)` now registers categorical obs field `"sample"`.
- Added shared-posterior plumbing without touching training/loss:
  - `_process_batches(...)` now collects per-group shared posterior `loc` and `scale` from `poe_stats`.
  - `_format_results(...)` now returns original + reordered shared posterior arrays.
- Added public DA APIs on model class:
  - `get_shared_posterior(...)`
  - `get_aggregated_posterior(...)`
  - `differential_abundance(...)`
- Added sample-aware aggregation and fallback behavior:
  - If `sample_key` is absent, DA aggregation falls back to one synthetic sample per group and emits an informational warning.
- Added alignment precondition warning in `differential_abundance(...)` when both:
  - `disentangle_group_shared_weight == 0`
  - `use_jeffreys_integ == False`
- Added focused tests:
  - New file `tests/test_differential_abundance.py`.
  - Covers sign behavior under synthetic shift, output size, sample-subset filtering, fallback warning, and alignment warning.

Verification:
- `python -m pytest tests/test_differential_abundance.py -v` passed (`5 passed`).
- `python -m pytest tests/test_regression_fixes.py -q` passed (`17 passed`).
- Confirmed no static errors in modified model file via editor diagnostics.

Notes:
- Full `pytest -v` and `scripts/smoke_vignettes.py` were not re-run in this pass to keep turnaround focused on R3 API delivery and targeted regression checks.

### P2: Second-pass doc compression (MrVI spec focus)
Status: completed

What changed:
- Condensed the MrVI DA section in ImplementationPlan.md into a tighter execution contract.
- Removed repeated narrative while preserving all locked decisions and checklist items.
- Kept PLAN/PROGRESS synchronization and added explicit update stamps.

Verification:
- Manual consistency check across PLAN.md, PROGRESS.md, and ImplementationPlan.md completed.
- Confirmed no feature scope or decision changes; only wording/structure compression.

### M1: Consolidated implementation milestone
Status: completed

What changed:
- R1 and R2 shipped:
  - Auto-inferred group indices in train and latent extraction paths.
  - Added one-call embedding API with transactional key writes.
- Enrichment and interpretation QoL shipped:
  - Added enrichment scoring and summarization APIs.
  - Added network validation helper and interpretation report/plots.
  - Added optional enrichment dependency wiring and dedicated tests.
  - Added real decoupler-backed integration coverage.
- Test hardening and docs updates:
  - Hardened CUDA-sensitive tests for mixed driver environments.
  - Added enrichment quickstart and updated README/API docs.
- Planning system consolidation:
  - Consolidated planning/spec sources into PLAN.md + ImplementationPlan.md.
  - Removed obsolete/duplicate planning artifacts.

Verification (high signal):
- `pytest tests/test_utils.py -q` passed.
- `pytest tests/test_api_boilerplate_reduction.py -q` passed.
- `pytest tests/test_enrichment.py -q` passed.
- `pytest -m integration -k enrichment -v` passed.
- `CUDA_VISIBLE_DEVICES='' python scripts/smoke_vignettes.py --epochs 1 --cells_per_group 50 --n_hvg 200` passed (`7/7`).
- Full suite status varied by local CUDA driver state; CPU-safe targeted tests are green.

### P3: Third-pass template normalization
Status: completed

What changed:
- Applied a strict short-template style across HANDOFF.md, PLAN.md, PROGRESS.md, ImplementationPlan.md, and CLAUDE.md continuity section.
- Standardized wording for purpose, usage/read-order, active target, and update stamps.
- Reduced repeated phrasing without changing active scope or decisions.

Verification:
- Manual cross-file consistency check completed.
- Confirmed active target remains R3 MrVI DA and deferred backlog semantics are unchanged.
