# spVIPESmulti — Next-Phase Feature Roadmap

Date: 2026-05-10
Owner: GitHub Copilot (consolidated plan)
Status: Active
Supersedes: `DISENTANGLE_SECOND_PASS_ACTION_PLAN.md`, `COUNTERFACTUAL_DESIGN.md`,
`COUNTERFACTUAL_AUDIT.md`, `audits/SECOND_PASS_AUDIT_PLAN.md` (all merged here).

---

## 0. How to read this document

Single source of truth for the next batch of feature work. Every feature ships with:

1. **Background & motivation** — what problem it solves and why now.
2. **Scope & non-goals** — explicit boundaries.
3. **TDD implementation plan** — failing tests first, then implementation, then validation.
4. **Quantitative go/no-go benchmark** — concrete numerical gates that decide whether
   to keep, iterate, or revert the feature. Decisions are data-driven, not subjective.

Features are ordered by **architectural risk (low → high)** so that low-risk, high-value
work lands first and informs the higher-risk follow-ups.

F1-F7 below keep that strict ordering. F8-F10 are optional extension tracks added after
the original sequence and can be scheduled opportunistically once F1 closeout is complete.

| # | Feature | Arch risk | Phase | Depends on |
|---|---|---|---|---|
| F1 | Conditional orthogonality instrumentation | **None** (metrics only) | Phase 1 | — |
| F2 | Counterfactual latent editing module (MVP) | **None** (new external module) | Phase 1 | F1 |
| F3 | Optional shared–private orthogonality loss | Low (default-off loss term) | Phase 2 | F1 |
| F4 | Condition/donor/batch covariate heads + losses | Low (default-off losses) | Phase 2 | F1 |
| F5 | Donor/condition-aware counterfactual protocols | Low (extends F2) | Phase 2 | F2, F4 |
| F6 | Graph-informed prototype regularizer | Medium (new regularizer) | Phase 3 | F4 |
| F7 | Counterfactual consistency loss + perturbation vectors | Medium (training + Phase 2 cf API) | Phase 3 | F2, F4 |
| F8 | Optional SysVI-style VampPrior for shared latent | Low (prior swap, default-off) | Phase 3 | F1 |
| F9 | Optional SysVI-style latent cycle-consistency regularizer | Medium (new training path, default-off) | Phase 3 | F4 |
| F10 | CellDISECT-aligned Kang benchmark + metrics pack | None (evaluation only) | Phase 1.5 | F2 |

Cross-cutting hard constraints (apply to every feature):

- No rewrites to encoders, decoders, PoE strategy, or latent dimensionality flow.
- All new losses must be **opt-in** (default weights = 0.0); defaults must be backward compatible.
- Single-modal and multimodal paths must remain feature-parity.
- Existing presets and tests must keep passing unchanged.

---

## 1. Shared infrastructure

### 1.1 Existing code anchors (do not duplicate)

- Disentanglement preset plumbing: `src/spVIPESmulti/model/_disentangle_presets.py`,
  `src/spVIPESmulti/model/spvipesmulti.py`.
- AnnData field registration & registry access: `src/spVIPESmulti/model/spvipesmulti.py`
  (note: `sample_key` is already registered but not yet used as a supervision signal).
- Disentanglement heads + loss accumulation: `src/spVIPESmulti/module/spVIPESmultimodule.py`
  (`_compute_disentangle_losses`, `loss`, `_loss_multimodal`, `generative`,
  `_generative_multimodal`, `_label_based_poe`).
- Latent extraction & embedding: `src/spVIPESmulti/model/spvipesmulti.py`
  (`get_latent_representation`, `embed`).
- Test entry points to extend: `tests/test_multimodal_disentangle.py`,
  `tests/test_regression_fixes.py`, `tests/test_multigroup_multimodal.py`.

### 1.2 Reproducibility & evaluation defaults (apply to every benchmark)

- Freeze one dataset and preprocessing path per benchmark. Default: Kang IFN dataset
  (ctrl vs. stim, immune cells from PertPy) via `docs/notebooks/kang_ifn_commit_old.ipynb`
  for benchmarking. This provides a standard two-condition, multi-cell-type baseline
  with known IFN response signatures.
  
  **Data preprocessing note:** Before analysis, remove megakaryocytes from the Kang IFN
  dataset (filter `adata = adata[adata.obs["cell_type"] != "Megakaryocytes"]` or equivalent).
  Megakaryocytes are a small, transcriptionally distinct population that can distort
  integration metrics; their removal aligns with standard immune-cell analysis pipelines.

- Every new feature benchmark must be compared against two external anchors in addition
  to the current `spVIPESmulti` baseline:
  - original `spVIPES` from <https://github.com/nrclaudio/spVIPES>
  - `contrastiveVAE` from scvi-tools
    (<https://github.com/scverse/scvi-tools/blob/612157b04320cf13b72e3e500707371b05811f54/src/scvi/external/contrastivevi/_model.py#L49>)
  - `CellDISECT` on Kang, aligned with the public tutorial + reproducibility scripts
    (<https://celldisect.readthedocs.io/en/latest/tutorials/CellDISECT_Counterfactual.html>,
    <https://github.com/stathismegas/CellDISECT_reproducibility/tree/main/reproduce_benchmarks/kang>)

  Use the same Kang IFN preprocessing, seeds, and train/validation split across all
  anchors so differences are attributable to the model or feature change rather
  than the benchmark setup.

- Fixed seed set: **3 seeds minimum**. Identical train/val split, batch size, and
  early-stopping settings across variants.
- Report mean ± std across seeds. Reject any variant with coefficient of variation > 0.2
  on the core metrics defined in §1.3.
- Save artifacts under `audits/<feature_id>/` as tidy CSV plus a Markdown summary and a
  `recommendation.json` with the go/no-go verdict.

- For Kang IFN runs, use `audits/kang_ifnb/` as the general benchmark lane.
  Keep one append-only `metrics.csv` plus one short Markdown note per run so feature
  regressions, disentanglement shifts, and baseline comparisons stay easy to diff.
  F10-specific CellDISECT parity artifacts can additionally be mirrored under
  `audits/F10/`.

### 1.3 Standard metric set (reused across F1, F3, F4, F6)

Core integration metrics on `z_shared`:

- `iLISI` (group mixing, higher better)
- `cLISI` (label purity, higher better)
- `kBET` (acceptance rate)
- `kNN purity` per cell type
- `Leiden ARI` against label
- silhouette by group / by label

Latent-quality metrics:

- Reconstruction loss (per cell)
- KL shared / KL private
- **Orthogonality**: mean Frobenius norm of within-stratum `corr(z_shared, z_private)`
  (introduced by F1)
- **Worst-stratum orthogonality**: max per-stratum norm

Counterfactual metrics (introduced by F2, F5):

- Cycle consistency (X→Y→X latent and expression L2 distance)
- Realism under target decoder (reconstruction proxy)
- Identity preservation (donor / cell-type classifier agreement)

CellDISECT-aligned metrics (introduced by F10):

- Counterfactual Pearson(mean): corr(mean(x_pred), mean(x_true))
- Counterfactual delta-Pearson: corr((x_pred-x_ctrl), (x_true-x_ctrl))
- Top-DE Pearson and Top-DE cosine (rank by |delta_true|)
- Wasserstein distance over gene marginals (top-20 DE and all HVGs)
- Disentanglement classifier gap (CAG): acc(S_i|Z_i) - acc(S_i|Z_{-i})
- MI-based scores: maxMIG / concatMIG / minMIG
- Optional fairness probes: demographic parity and equalized odds on Z_{-i}

### 1.4 CellDISECT reference protocol on Kang (for benchmark parity)

Reference observations from public CellDISECT material to be mirrored in F10:

- Data: `kang_normalized_hvg.h5ad`, with raw counts in `layers['counts']`.
- Core covariates: `cats = ['cell_type', 'condition']`.
- Common CellDISECT settings (public tutorial/repro):
  - `n_latent_shared=32`, `n_latent_attribute=32`, `n_hidden=128`, `n_layers=2`
  - `recon_weight=20`, `cf_weight=0.8`, `beta=0.003`, `clf_weight=0.05`,
    `adv_clf_weight=0.014`, `adv_period=5`, `n_cf=1`
- Kang split strategy to include in parity runs:
  - leave-one-cell-type-out splits (`split_CD14 Mono`, `split_CD4 T`, ...)
  - harder multi-cell-type held-out splits (`split_CD14Mono_CD4T`, etc.)
- Counterfactual target used in tutorial benchmark:
  - CD14 Mono control→stimulated (`x_ctrl`, `x_true`, `x_pred`) with Pearson and
    delta-Pearson reported for top DE and all genes.

Artifacts for this protocol should be written under `audits/F10/` and mirrored with
append-only summaries in `audits/kang_ifnb/`.

---

## 2. Feature specifications

---

### F1 — Conditional orthogonality instrumentation (metrics only)

**Background.** The model already enforces shared/private separation through GRL +
supervised heads, but we have no quantitative readout of *residual* dependence between
`z_shared` and `z_private`, especially **conditional on biological strata** (donor,
timepoint, cell type). Without that readout we cannot evaluate any of the downstream
features (F2–F7) objectively.

**Scope.**

- Compute within-stratum correlation norm between `z_shared` and `z_private` and log it
  in `extra_metrics` for both single-modal and multimodal paths.
- No loss change. No model API change beyond optional kwargs to the metric.
- Stratification keys: any registered categorical (default `("sample",)`; falls back to
  the global batch if none registered).

**Non-goals.** Adding a penalty (that is F3). Adding new heads (that is F4).

**TDD plan.**

1. **Test first** (`tests/test_disentangle_metrics.py`, NEW):
   - `test_orthogonality_metric_present_when_enabled`: train a 2-step model, assert
     `extra_metrics["orthogonality_within_stratum"]` is finite and ≥ 0.
   - `test_orthogonality_zero_for_independent_inputs`: feed synthetic independent
     `z_shared`, `z_private` to the helper; assert value is low (e.g. < 0.1).
   - `test_orthogonality_one_for_perfect_copy`: feed `z_private = z_shared`; assert
     value > 0.8.
   - `test_min_cells_per_stratum_filter`: strata below `orthogonality_min_cells_per_stratum`
     must be excluded and counted in `extra_metrics["orthogonality_excluded_strata"]`.
   - Multimodal parity: same checks under multimodal path.
2. **Implement** helper in `src/spVIPESmulti/module/spVIPESmultimodule.py`:
   - `_within_stratum_corr_norm(z_shared, z_private, stratum_ids, min_cells)` returning
     mean and worst-stratum Frobenius norm.
   - Wire into both `loss(...)` and `_loss_multimodal(...)` behind kwargs:
     `compute_orthogonality_metric: bool = False`,
     `orthogonality_groupby_keys: tuple[str, ...] = ("sample",)`,
     `orthogonality_min_cells_per_stratum: int = 16`.
3. **Validate**: `pytest tests/test_disentangle_metrics.py tests/test_multimodal_disentangle.py -q`.

**Quantitative go/no-go benchmark.**

| Metric | Source | Pass | Reject |
|---|---|---|---|
| Helper unit tests | `pytest` | All green | Any failure |
| Overhead per training step (CPU smoke run) | `scripts/smoke_vignettes.py --epochs 5` | ≤ +5% wall time vs disabled | > +15% |
| Numerical agreement vs NumPy reference on synthetic data | unit test | abs error < 1e-4 | ≥ 1e-4 |

If pass → unlock F2, F3, F4. If reject → fix the helper before any downstream work.

---

### F2 — Counterfactual latent editing module (MVP, single-modality)

**Background.** Shared–private + label-based PoE create a structured latent space where
biologically meaningful interventions (condition translation, batch removal, donor
transfer) are well-defined. Today users have no high-level API to perform these edits
or quantify their reliability. The MVP exposes deterministic latent operators, an
encode→edit→decode pipeline, and disentanglement diagnostics — no model retraining
required, no architectural change.

**Architecture rationale (auditable).**

- z_shared is aligned by `_label_based_poe` and supervised by `disentangle_label_shared_weight`,
  so directional edits on z_shared correspond to label/condition shifts.
- z_private retains group identity (preserved by `disentangle_group_private_weight`),
  so keeping it fixed during edits preserves domain effects.
- Per-group decoders preserve domain shift → counterfactuals are intentionally
  group-specific.
- Posterior means (`logtheta_loc`) give deterministic, reproducible edits; posterior
  variance gives an uncertainty band.

**Scope (MVP).**

- New module `src/spVIPESmulti/interventions/` with the following files and public API:

  ```text
  src/spVIPESmulti/interventions/
    __init__.py              # re-exports public API
    latent_operators.py      # arithmetic, interpolation, replacement (pure functions)
    counterfactual.py        # encode_cells, decode_counterfactual, predict_counterfactual,
                             # transfer_condition, edit_latent, CounterfactualResult
    diagnostics.py           # leakage_score, condition_separability,
                             # latent_variance_utilization, integration_report
    utils.py                 # _get_group_decoder, _library_correction, _posterior_sample
  ```

- Public functions (single-modality only):

  ```python
  encode_cells(model, adata, group_idx=None, include_variance=True)
  latent_arithmetic(z, direction, weight=1.0)
  latent_interpolation(z_src, z_tgt, alpha)
  latent_replacement(z, dimension, value)
  decode_counterfactual(model, z_shared, z_private, group_idx, adata,
                        include_uncertainty=True, return_components=False, batch_size=512)
  predict_counterfactual(model, adata, cells=None, group_idx=0,
                         intervention="arithmetic", direction=None,
                         target_cells=None, alpha=1.0, dimension=None, value=None,
                         return_uncertainty=True)
  transfer_condition(model, adata, cells, condition_from, condition_to,
                     group_src, group_dst, latent_type="shared")
  leakage_score(model, adata, group_key, label_key=None)
  condition_separability(model, adata, label_key)
  integration_report(model, adata, group_key, label_key=None)
  ```

- Returns `CounterfactualResult` dataclass with `.X`, `.uncertainty`, `.info`.

**Non-goals (deferred to F5/F7).**

- Per-modality editing (multimodal MVP marked `xfail`).
- Perturbation vector learning via gradient ascent / classifier gradients.
- Donor/condition-conditional counterfactuals (needs F4 covariate registration).

**Resolved design questions** (from the prior counterfactual audit):

- **Q1 — replacement mode.** `predict_counterfactual` accepts `intervention="replacement"`
  with `dimension=`/`value=` kwargs. Single entry point, signature documented.
- **Q2 — disentanglement warning.** Always emit a runtime warning when
  `leakage_score(..., shared) > 0.4`. Threshold is documented; no verbosity flag.
- **Q3 — uncertainty calibration.** Documented in tutorial only; no CI assertion.
- **Q4 — performance benchmarking.** Log timings in tests; do not assert. Hardware-dependent.

**TDD plan.**

1. **Test files (write first):**
   - `tests/test_counterfactual_basics.py` — operator shape/dtype/NaN, encode keys,
     variance positivity, decode shape, end-to-end shape, uncertainty grows with
     edit magnitude (Spearman ρ > 0 sanity), `group_idx` bounds error.
   - `tests/test_counterfactual_integration.py` — library-size preservation,
     reconstruction quality (Pearson > 0.7 on toy data), `transfer_condition` mean
     direction within tolerance, edit composition equivalence,
     `xfail` for multimodal call.
   - `tests/test_counterfactual_diagnostics.py` — `leakage_score` ∈ [0, 1],
     `condition_separability` valid, leakage with `disentangle_preset="full"` ≤
     leakage with `disentangle_preset="off"` on a tiny model,
     `silhouette(z_private)` > silhouette of random labels.
2. **Fixture:** `tests/conftest.py` adds `minimal_model_adata` (2 groups × 2 labels ×
   50 genes × 200 cells, 5 epochs, marked `@pytest.mark.integration`).
3. **Implement** in the order: `latent_operators` → `utils` → `counterfactual`
   (`encode_cells`, `decode_counterfactual`, `predict_counterfactual`,
   `transfer_condition`) → `diagnostics`.
4. **Tutorial notebook:** `docs/notebooks/counterfactual_interventions_tutorial.ipynb`
   covering disease→healthy translation, batch removal via z_private replacement,
   uncertainty visualization, diagnostic report.

**Quantitative go/no-go benchmark.**

Run on the `minimal_model_adata` fixture (CI) and on the Kang IFN dataset
(manual smoke via `docs/notebooks/kang_ifn_commit_old.ipynb`). Compare baseline vs counterfactual outputs.

| Metric | Pass | Reject |
|---|---|---|
| Unit + integration tests | All green | Any failure |
| Cycle consistency: ‖encode(decode(z)) − z‖₂ / ‖z‖₂ | ≤ 0.10 | > 0.20 |
| Reconstruction Pearson on identity edit (weight=0) vs original | ≥ 0.95 | < 0.85 |
| `transfer_condition` recovers mean direction (cosine sim) | ≥ 0.85 | < 0.7 |
| `leakage_score(shared)` on tutorial data | < 0.40 | ≥ 0.60 |
| `silhouette(z_private)` by group | ≥ 0.30 | < 0.10 |
| Encoding 10K cells (informational, no assertion) | logged | — |
| Decoding 10K counterfactuals (informational) | logged | — |
| Test suite delta | ≤ +60 s | > +180 s |

If pass → ship MVP. If reject → first inspect F1 metrics; if shared/private leakage
is itself the cause, F3 + F4 must land before re-running F2's benchmark.

---

### F3 — Optional shared–private orthogonality loss

**Background.** F1 measures conditional dependence; F3 lets us **penalize** it.
Hypothesis: a within-stratum partial-correlation penalty reduces leakage without
materially harming reconstruction or integration.

**Scope.**

- New module/model kwargs (default 0.0, fully backward compatible):
  - `pcorr_weight: float = 0.0`
  - `pcorr_min_cells_per_stratum: int = 16`
  - `pcorr_groupby_keys: tuple[str, ...] = ("sample",)`
- New preset key `orthogonality_weight` in `_disentangle_presets.py`; existing presets
  set it to 0.0. New presets:
  - `minimal_safe_bio` — conservative on/off matrix.
  - `full_bio` — F3 + F4 enabled at moderate defaults.
- Add penalty term in `_compute_disentangle_losses`, included in both `loss` and
  `_loss_multimodal`, scaled like existing multimodal private terms.
- Warmup schedule: penalty off for first 30% of epochs, linear ramp to target by 60%.
- Logged metric name: `orthogonality_loss`.

**TDD plan.**

1. **Test first** (`tests/test_orthogonality_loss.py`):
   - Construction: passes with positive `pcorr_weight`; `ValueError` on negative.
   - Default-off: with `pcorr_weight=0.0`, total loss matches baseline within 1e-6.
   - Finite loss with weight on, multimodal parity.
   - Metric `orthogonality_loss` appears in `extra_metrics` only when enabled.
2. **Implement** WP1 + WP2 + WP5 from prior plan in a single change: kwargs, preset
   keys, helper reuse from F1, warmup ramp.
3. **Validate**: targeted suite then full suite.

**Quantitative go/no-go benchmark.**

Run matrix (3 seeds each) on the Kang IFN dataset:

- baseline (current settings)
- `pcorr_weight ∈ {0.01, 0.05, 0.10, 0.20}`

Decision rules (hard gates):

| Criterion | Pass | Reject |
|---|---|---|
| Within-stratum mean orthogonality (vs baseline) | **≥ 20% reduction** | < 10% reduction |
| Reconstruction loss degradation | ≤ 5% | > 10% |
| iLISI degradation | ≤ 10% | > 20% |
| kBET degradation | ≤ 10% | > 20% |
| cLISI / kNN purity | within ±5% or improved | > 5% drop |
| Cross-seed CV on core metrics | ≤ 0.20 | > 0.30 |

Promotion → adopt the smallest `pcorr_weight` that satisfies all "Pass" rules.
Reject → keep F1 metrics, drop the penalty, re-evaluate after F4 lands.

Artifacts: `audits/F3/metrics.csv`, `audits/F3/summary.md`, `audits/F3/recommendation.json`.

---

### F4 — Condition/donor/batch covariate heads + losses

**Background.** Today the only categorical supervision is `label_key` (cell type) and
optionally `sample_key` (registered but unused as a signal). For population-scale
biology we need explicit nuisance removal (batch, donor) on `z_shared` and explicit
retention of donor identity on `z_private`. This is the canonical second-pass
disentanglement upgrade.

**Scope.**

- `setup_anndata(...)` gains optional `condition_key` and `donor_key`, registered with
  `CategoricalObsField` exactly like `label_key`/`sample_key`. Existing call sites
  unaffected.
- Model `__init__` exposes flags (`use_condition`, `n_conditions`, `use_donor`,
  `n_donors`) to the module. Validation: each requested loss must have its key
  registered, otherwise raise `ValueError` with an actionable message.
- New constructor weights (all default 0.0):
  - `disentangle_batch_shared_weight` (GRL CE on `z_shared`)
  - `disentangle_donor_shared_weight` (GRL CE on `z_shared`)
  - `disentangle_donor_private_weight` (supervised CE on `z_private`)
- New module heads using existing `FCLayers` pattern:
  - `q_batch_shared`, `q_donor_shared` (adversarial via GRL)
  - `q_donor_private` (supervised); reuse multimodal private-loop helper to apply
    across all per-modality private latents.
- Preset extensions: `_disentangle_presets.py` gains the three new keys; existing
  presets set them to 0.0. `minimal_safe_bio` enables donor-private only;
  `full_bio` enables all three at moderate defaults (start 0.5).
- Logged metrics: `disentangle_batch_shared_loss`, `disentangle_donor_shared_loss`,
  `disentangle_donor_private_loss`.

**TDD plan.**

1. **Test first** (`tests/test_covariate_heads.py`):
   - `setup_anndata` registers `condition_key`/`donor_key` correctly.
   - Model construction with new weights enabled but missing key raises `ValueError`
     containing the missing key name.
   - Negative-weight validation parity with existing weights.
   - Finite loss when each head is enabled in isolation; metric appears in
     `extra_metrics` only when its weight is > 0.
   - Default-off equivalence (numerical match to baseline within 1e-6 when all new
     weights are 0).
   - Multimodal parity (loop scaling matches existing multimodal private terms).
2. **Extend** existing `tests/test_multimodal_disentangle.py` and
   `tests/test_regression_fixes.py` for preset integrity and missing-key guards.
3. **Implement** in order: WP1+WP2 (API + presets + validators) → WP3 (heads) →
   WP4 (losses) — single feature branch.
4. **Validate**: `pytest tests/test_multimodal_disentangle.py tests/test_regression_fixes.py
   tests/test_multigroup_multimodal.py tests/test_covariate_heads.py -q`, then full suite.

**Quantitative go/no-go benchmark.**

3-seed matrix on the Kang IFN dataset:

- baseline (no new heads)
- donor-private only (using `replicate` as donor_key)
- donor-shared (GRL) only
- batch-shared (GRL) only (using `label` as batch/condition)
- `full_bio` preset (all three)

| Criterion | Pass | Reject |
|---|---|---|
| Donor classifier accuracy on `z_private` (held-out) | **≥ baseline + 10pp** | < baseline + 2pp |
| Donor classifier accuracy on `z_shared` (held-out) | **≤ baseline − 10pp** | ≥ baseline |
| Batch (sample) classifier accuracy on `z_shared` | **≤ baseline − 10pp** | ≥ baseline |
| Cell-type classifier accuracy on `z_shared` | within ±3pp of baseline | drop > 5pp |
| Reconstruction loss degradation | ≤ 5% | > 10% |
| iLISI / kBET | within ±10% of baseline | > 20% drop |
| Cross-seed CV on core metrics | ≤ 0.20 | > 0.30 |

Promotion → adopt `full_bio` defaults if all "Pass" rules hold; otherwise adopt the
combination of heads that pass individually. Reject → keep heads as opt-in only,
do not change presets.

Artifacts: `audits/F4/metrics.csv`, `audits/F4/summary.md`, `audits/F4/recommendation.json`.

---

### F5 — Donor/condition-aware counterfactual protocols

**Background.** F2's MVP exposes generic latent edits. Once F4 registers `donor_key`
and `condition_key`, we can ship rigorous counterfactual protocols for the actual
biological questions (individual A under X "as if" under Y).

**Scope (extensions to `interventions/counterfactual.py`).**

- **Protocol P1 — unmatched private swap.** For each B-cell, sample a private latent
  from a random A-cell; keep B's shared latent; decode in B's decoder.
- **Protocol P2 — label-matched private swap.** Same as P1 but match on `label_key`.
- **Protocol P3 — label + donor/timepoint matched.** Match on `label_key` plus a
  user-supplied stratum list; report fallback counts when matches are missing.
- **Condition-shift protocol.** For donor *i* under condition X, replace the
  condition-associated direction (mean(X) − mean(Y) on `z_shared` within donor *i*
  when available) and decode. If `condition_key` not registered, raise an actionable
  error (no diagnostic-only fallback in MVP).

**TDD plan.**

1. **Test first** (`tests/test_counterfactual_protocols.py`):
   - Each protocol returns shapes matching input cells × n_genes.
   - P3 fallback counts surfaced in `result.info["fallback_counts"]`.
   - Condition shift without `condition_key` raises `ValueError` listing required keys.
   - Cycle consistency on identity protocol (alpha=0) returns inputs within tolerance.
2. **Implement** new functions; reuse F2 building blocks; no new model methods.

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| Cycle consistency (latent L2 / ‖z‖₂) | ≤ 0.10 | > 0.20 |
| Realism: NB log-likelihood vs target-group cells | within 1 nat per cell of target | > 3 nats |
| Donor classifier agreement on counterfactuals (when donor preserved) | ≥ 0.85 | < 0.7 |
| Cell-type classifier agreement under condition shift | ≥ 0.85 | < 0.7 |
| P2 vs P1 realism delta | P2 strictly better | P2 ≤ P1 |

Promotion → expose protocols in tutorial. Reject → if P2 ≤ P1, the disentanglement
itself is insufficient; loop back to F3 + F4 weight tuning before reshipping.

Artifacts: `audits/F5/counterfactuals.csv`, `audits/F5/summary.md`,
`audits/F5/recommendation.json`.

---

### F6 — Knowledge-informed prototype graph regularizer (Phase 3)

**Background.** Reuse the shared prototypes already maintained for the contrastive
objective. A Laplacian penalty over a label-derived adjacency encourages biologically
related prototypes to live near each other in `z_shared`.

**Scope.**

- New optional weight `graph_regularizer_weight: float = 0.0` (default off).
- Adjacency from `label_key` and optional `condition_key`/`donor_key`.
- Penalty `tr(P^T L P)`. Edge weighting and minimum-support filtering supported.
- Strict fallback: missing required labels → warn (default) or raise (strict mode).
- Logged metric `graph_regularizer_loss`.

**TDD plan.**

1. **Test first** (`tests/test_graph_regularizer.py`): default-off equivalence,
   Laplacian-symmetric adjacency, finite loss, missing-key behavior in both modes.
2. **Implement** in `src/spVIPESmulti/module/spVIPESmultimodule.py` (and a small
   helper in `src/spVIPESmulti/module/utils.py` if needed).

**Quantitative go/no-go benchmark.**

3-seed matrix on the Kang IFN dataset:

- baseline (best F4 preset)
- + `graph_regularizer_weight ∈ {0.05, 0.1, 0.2}`

| Criterion | Pass | Reject |
|---|---|---|
| kNN purity on rare cell types (e.g., smallest cluster by cell-type distribution) | **≥ baseline + 0.05** | within baseline ± 0.02 |
| cLISI on minority types | **improvement ≥ 5%** | drop |
| Reconstruction loss degradation | ≤ 5% | > 10% |
| iLISI / kBET | within ±10% of baseline | > 20% drop |
| Cross-seed CV | ≤ 0.20 | > 0.30 |

Promotion → smallest weight satisfying all gates. Reject → leave as opt-in advanced
feature only.

Artifacts: `audits/F6/metrics.csv`, `audits/F6/summary.md`, `audits/F6/recommendation.json`.

---

### F7 — Counterfactual consistency loss + perturbation vectors (Phase 3)

**Background.** Once F2/F5 expose deterministic edits and F4 supplies covariate heads,
we can close the loop by adding (a) a latent-only counterfactual consistency loss
during training, and (b) gradient-based perturbation vector learning at inference.

**Scope.**

- Latent-only counterfactual consistency penalty (no decoder rollout) gated behind
  `counterfactual_consistency_weight: float = 0.0`.
- Inference-time `learn_perturbation_vector(model, adata, condition_pairs,
  latent_type, method)` supporting `mean_difference`, `classifier_gradient`,
  `gradient_ascent`.
- `predict_perturbation_response(model, adata, perturbation_vector, magnitude)`.

**TDD plan.**

1. **Test first** (`tests/test_perturbation_vectors.py`):
   - `mean_difference` on synthetic two-condition data recovers the planted shift
     within cosine similarity ≥ 0.9.
   - `classifier_gradient` produces a non-zero direction whose application increases
     classifier logit for the target class.
   - Default-off counterfactual consistency loss leaves training loss unchanged.
2. **Implement** in `src/spVIPESmulti/interventions/perturbation.py` and a single
   loss term in `_compute_disentangle_losses`.

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| Recovered direction cosine similarity (synthetic) | ≥ 0.90 | < 0.70 |
| Held-out classifier logit increase under predicted perturbation | ≥ 50% relative | < 10% |
| Training loss with `counterfactual_consistency_weight=0` matches baseline | match within 1e-6 | drift |
| Training-time overhead with weight on | ≤ +10% wall time | > +25% |

Promotion → expose in tutorial + API. Reject → keep as research script only.

Artifacts: `audits/F7/metrics.csv`, `audits/F7/summary.md`, `audits/F7/recommendation.json`.

---

### F8 — Optional SysVI-style VampPrior for shared latent (Phase 3)

**Background.** SysVI uses a VampPrior to improve biological preservation while keeping
integration strong. The prior is more expressive than a standard Gaussian and can reduce
over-regularization of biologically meaningful modes.

**Scope.**

- Add optional prior mode for shared latent only (private latent unchanged):
  - `shared_prior: Literal["standard_normal", "vamp"] = "standard_normal"`
  - `shared_prior_components: int = 5`
  - `shared_prior_trainable: bool = True`
  - `shared_prior_pseudoinput_strategy: Literal["random_cells", "stratified_labels"] = "stratified_labels"`
- Implement behind default-off behavior (`shared_prior="standard_normal"`), preserving
  baseline numerics.
- Use posterior-sample Monte Carlo KL estimation for VampPrior (as in sysVI/scvi prior logic).
- Add logged metric: `kl_shared_prior` plus component diagnostics (`vamp_weight_entropy`).

**Non-goals.**

- No changes to decoder architecture, PoE math, or private latent prior in the MVP.
- No multimodal-specific prior variants in first pass.

**TDD plan.**

1. **Test first** (`tests/test_vampprior_shared.py`):
   - Constructor validation for valid/invalid prior args.
   - Default-off equivalence (`standard_normal`) within 1e-6.
   - Finite KL/loss with `shared_prior="vamp"` and small pseudoinput set.
   - State-dict save/load parity for VampPrior params.
2. **Implement** in module + model constructor plumbing, including pseudoinput init.
3. **Validate** with targeted tests and one Kang smoke run.

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| Tests (`tests/test_vampprior_shared.py`) | All green | Any failure |
| cLISI / kNN purity vs baseline | improved or within ±3% | > 5% drop |
| iLISI / kBET vs baseline | within ±10% | > 20% drop |
| Reconstruction loss degradation | ≤ 5% | > 10% |
| Cross-seed CV | ≤ 0.20 | > 0.30 |

Promotion -> keep smallest component count satisfying all pass gates.
Reject -> keep `shared_prior="standard_normal"` as default and defer.

Artifacts: `audits/F8/metrics.csv`, `audits/F8/summary.md`, `audits/F8/recommendation.json`.

---

### F9 — Optional SysVI-style latent cycle-consistency regularizer (Phase 3)

**Background.** SysVI strengthens integration by decoding a cell with a switched batch
covariate and re-encoding it, then penalizing latent drift between original and cycled
embeddings (on standardized latent coordinates).

**Scope.**

- Add optional cycle loss path (default-off):
  - `latent_cycle_weight: float = 0.0`
  - `latent_cycle_key: str = "sample"` (fallback to batch key)
  - `latent_cycle_on: Literal["shared", "shared_private"] = "shared"`
- Implement random alternative category selection per cell (must differ from original).
- Compute standardized latent MSE between original and cycle pass means (sysVI-style).
- Log `latent_cycle_loss` and `latent_cycle_active_fraction`.

**Non-goals.**

- No cycle on expression-space loss in first pass.
- No adversarial replacement; this is additive and opt-in.

**TDD plan.**

1. **Test first** (`tests/test_latent_cycle_loss.py`):
   - Default-off equivalence within 1e-6.
   - Positive finite cycle loss when enabled.
   - Ensures switched category differs from source category.
   - Multimodal parity path is finite.
2. **Implement** cycle helpers and training loss wiring.
3. **Validate** targeted + full suite.

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| Tests (`tests/test_latent_cycle_loss.py`) | All green | Any failure |
| iLISI (batch mixing) improvement vs baseline | ≥ +10% | < +3% |
| Cell-type classifier on shared latent | within ±3pp | drop > 5pp |
| Reconstruction loss degradation | ≤ 5% | > 10% |
| Training-time overhead | ≤ +12% | > +25% |

Promotion -> tune `latent_cycle_weight` minimally to pass gates.
Reject -> keep feature opt-in and disabled in presets.

Artifacts: `audits/F9/metrics.csv`, `audits/F9/summary.md`, `audits/F9/recommendation.json`.

---

### F10 — CellDISECT-aligned Kang benchmark and metrics pack (Phase 1.5)

**Background.** CellDISECT provides a public Kang counterfactual benchmark and
disentanglement analyses (MI/CAG/fairness style metrics). Mirroring these gives a
strong external yardstick for both disentanglement and counterfactual quality.

**Scope.**

- Add benchmark scripts and metric helpers (evaluation only, no model architecture changes):
  - `scripts/benchmark_kang_celldisect_parity.py`
  - `src/spVIPESmulti/metrics_celldisect.py`
- Include at minimum these model anchors on identical splits:
  - `spVIPESmulti` current baseline
  - external `CellDISECT`
  - optional `biolord` and `scDisInFact` when reproducible in environment
- Evaluate both split families:
  - leave-one-cell-type-out Kang splits
  - hard multi-cell-type held-out splits
- Save per-cell-type outputs:
  - `*_pearson.csv`, `*_delta_pearson.csv`, `*_emd.csv`
  - `*_disentangle.csv` (CAG, maxMIG/concatMIG/minMIG, optional fairness metrics)

**Non-goals.**

- No requirement to beat CellDISECT before F2/F4 completion.
- No dependency pinning changes forced into the core package install.

**TDD plan.**

1. **Test first** (`tests/test_celldisect_metric_parity.py`):
   - Pearson/delta-Pearson implementations match NumPy/SciPy reference.
   - Top-DE selection and Wasserstein aggregation shape checks.
   - CAG and MIG helper outputs are bounded/finite.
2. **Implement** metrics helpers and script CLI.
3. **Validate** by running one small Kang split smoke and checking artifact schema.

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| Metric helper tests | All green | Any failure |
| Reproducibility across 3 seeds (Pearson/delta, EMD) | CV ≤ 0.20 | CV > 0.30 |
| Artifact completeness per split | all expected files produced | missing files |
| Baseline positioning clarity | ranked table + recommendation JSON | ambiguous/no verdict |

Promotion -> keep F10 as mandatory external audit gate for F2/F4/F5 changes.
Reject -> fix audit harness before shipping new disentanglement features.

Artifacts: `audits/F10/metrics.csv`, `audits/F10/summary.md`, `audits/F10/recommendation.json`.

---

## 3. Validation commands (canonical order)

After each feature lands:

```bash
# Targeted (per feature)
pytest tests/test_disentangle_metrics.py -q                # F1
pytest tests/test_counterfactual_basics.py tests/test_counterfactual_integration.py \
       tests/test_counterfactual_diagnostics.py -q         # F2
pytest tests/test_orthogonality_loss.py -q                 # F3
pytest tests/test_covariate_heads.py tests/test_multimodal_disentangle.py \
       tests/test_regression_fixes.py -q                   # F4
pytest tests/test_counterfactual_protocols.py -q           # F5
pytest tests/test_graph_regularizer.py -q                  # F6
pytest tests/test_perturbation_vectors.py -q               # F7
pytest tests/test_vampprior_shared.py -q                   # F8
pytest tests/test_latent_cycle_loss.py -q                  # F9
pytest tests/test_celldisect_metric_parity.py -q           # F10

# Full suite (mandatory before promotion)
pytest tests/ -q --ignore=tests/test_evaluate.py

# Smoke API combinations (after F2, F4, F5)
python scripts/smoke_vignettes.py --epochs 5 --cells_per_group 300
```

Optional grep sanity after edits:

```bash
rg "pcorr_weight|orthogonality_weight|disentangle_batch_shared_weight|disentangle_donor_shared_weight|disentangle_donor_private_weight|graph_regularizer_weight|counterfactual_consistency_weight|shared_prior|latent_cycle_weight" src tests
```

---

## 4. Acceptance criteria (apply to every feature)

A feature is "done" only when **all** are true:

1. Existing API behavior unchanged for users who do not pass new args.
2. New loss/metric terms are no-ops at default weights (numerical equivalence ≤ 1e-6).
3. New metrics appear in `extra_metrics` only when the corresponding weight > 0 or
   metric flag is on.
4. Targeted tests pass, full suite passes (`pytest tests/ -q --ignore=tests/test_evaluate.py`).
5. Quantitative go/no-go benchmark from §2 satisfies all "Pass" rules across ≥ 3 seeds.
6. Artifacts written to `audits/<feature_id>/` and a one-paragraph entry appended to
   `PROGRESS.md`.

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Loss-scale imbalance destabilizes early training | Default low weights, reuse existing warmup scaling, F1 metric makes leakage visible early |
| Covariate biology vs nuisance-removal conflict (e.g., condition encodes biology) | New heads optional; default presets keep them off; document trade-offs |
| Multimodal private-term overweighting | Reuse existing per-modality scaling pattern in `_loss_multimodal` |
| Graph regularizer collapses rare labels | Edge weighting + minimum-support filter |
| VampPrior pseudoinputs collapse or drift | stratified pseudoinput init + entropy monitoring + standard prior fallback |
| Cycle loss over-correction harms biology | strict weight sweep with cell-type retention gate and default-off |
| Small strata produce noisy conditional correlation | `orthogonality_min_cells_per_stratum` filter; report excluded strata count |
| Counterfactuals "look plausible" but fail identity | Donor/condition classifier checks + cycle-consistency tests in F2/F5 benchmarks |
| External benchmark mismatch vs CellDISECT protocol | lock split definitions + metric parity tests + artifact schema checks |
| Benchmark instability across seeds | Reject-on-CV gate (CV > 0.20 / 0.30) in every benchmark table |

---

## 6. Immediate next coding step

Land **F1** in a single change set:

1. Write failing tests in `tests/test_disentangle_metrics.py`.
2. Add `_within_stratum_corr_norm` helper + wire into single-modal and multimodal
   loss paths behind the three new kwargs (default-off).
3. Run targeted suite, then full suite.
4. Write `audits/F1/summary.md` with the overhead measurement on the smoke run.

Once F1 is green and the overhead gate passes, start F2 (counterfactual MVP) in
parallel with the F3+F4 design slice; F2 ships independently of F3/F4 because it
does not change training.

After F2 baseline APIs are in place, activate F10 (CellDISECT-aligned Kang audit
harness) and use `audits/kang_ifnb/` as the primary benchmark lane, with optional
F10 parity mirrors under `audits/F10/`.

---

## 7. Cross-references

- Active queue + deferred backlog: `PLAN.md`.
- Implementation history: `PROGRESS.md`.
- Next-session pointer: `HANDOFF.md`.
- Architecture & commands: `CLAUDE.md`.
- Older roadmap of feature specs: `ImplementationPlan.md` (still authoritative for
  performance backlog items P-PERF-2, P-PERF-4, N5-D, N5-E, P6).
