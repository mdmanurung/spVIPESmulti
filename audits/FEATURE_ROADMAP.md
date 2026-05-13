# spVIPESmulti — Feature Roadmap

Date: 2026-05-10 (compressed 2026-05-13)
Status: Active
Supersedes: `DISENTANGLE_SECOND_PASS_ACTION_PLAN.md`, `COUNTERFACTUAL_DESIGN.md`,
`COUNTERFACTUAL_AUDIT.md`, `audits/SECOND_PASS_AUDIT_PLAN.md` (all merged here).

______________________________________________________________________

## 0. How to read this document

Single source of truth for the next batch of feature work. Every feature ships with:

1. **Background & motivation** — what problem it solves and why now.
1. **Scope & non-goals** — explicit boundaries.
1. **TDD implementation plan** — failing tests first, then implementation, then validation.
1. **Quantitative go/no-go benchmark** — concrete numerical gates.

Features are ordered by **scientific readiness**. F8-F14 are optional extension
tracks; schedule only when prerequisite benchmark gates exist.

| # | Feature | Status | Phase | Depends on |
|---|---|---|---|---|
| F1 | Conditional orthogonality instrumentation | ✅ **closed** | Phase 1 | — |
| F4 | Condition/donor/batch covariate heads + losses | ✅ **done** (preset rejected) | Phase 1.5 | F1 |
| F2 | Safe counterfactual latent editing module (MVP) | ✅ **done** | Phase 2 | F1, F4-lite |
| F3 | Optional shared–private orthogonality loss | ⚠️ **archived/default-off** | Phase 2 | F1, F4 |
| F10 | CellDISECT-aligned Kang benchmark + metrics pack | ✅ F10a/F10b done | Phase 1.5 | F1 |
| F5 | Donor/condition-aware counterfactual protocols | ✅ **done** | Phase 2 | F2, F4 |
| F6 | Graph-informed prototype regularizer | deferred | Phase 3 | F4 |
| F7 | Counterfactual consistency loss + perturbation vectors | deferred | Phase 3 | F2, F4 |
| F8 | Optional SysVI-style VampPrior for shared latent | deferred | Phase 3 | F1, F10 |
| F9 | Optional SysVI-style latent cycle-consistency regularizer | deferred | Phase 3 | F4, F10 |
| F11 | Nonlinear dependence diagnostics (HSIC / MI / partial corr) | implemented; audit iterate | Phase 2 | F1 |
| F12 | Conditional decoder / MMD alignment track | deferred | Phase 3 | F4, F10 |
| F13 | Artifact/QC latent track | deferred | Phase 4 | QC labels, F10 |
| F14 | Causal / coupled-VAE research track | research | Research | F2, F5, F10 |

Cross-cutting hard constraints:

- No rewrites to encoders, decoders, PoE strategy, or latent dimensionality for F1-F11.
- All new losses must be **opt-in** (default weights = 0.0); backward compatible.
- Single-modal and multimodal paths must remain feature-parity.
- Counterfactual outputs are **associative predictions**, not causal claims, unless a
  benchmark uses interventional ground truth and passes the corresponding audit gates.

______________________________________________________________________

## 1. Shared infrastructure

### 1.1 Existing code anchors (do not duplicate)

- Disentanglement preset plumbing: `src/spVIPESmulti/model/_disentangle_presets.py`,
  `src/spVIPESmulti/model/spvipesmulti.py`.
- Disentanglement heads + loss accumulation: `src/spVIPESmulti/module/spVIPESmultimodule.py`
  (`_compute_disentangle_losses`, `loss`, `_loss_multimodal`, `_label_based_poe`).
- Latent extraction & embedding: `src/spVIPESmulti/model/spvipesmulti.py`
  (`get_latent_representation`, `embed`).
- Counterfactual API: `src/spVIPESmulti/interventions/` (F2).
- Test entry points: `tests/test_multimodal_disentangle.py`, `tests/test_regression_fixes.py`,
  `tests/test_multigroup_multimodal.py`.

### 1.2 Reproducibility & evaluation defaults

- Default dataset: Kang IFN (`ctrl` vs `stim`, immune cells from PertPy) via
  `docs/notebooks/kang_ifn_commit_old.ipynb`. Remove megakaryocytes before analysis.
- Fixed seed set: **3 seeds minimum**. Identical train/val split, batch size, early-stopping.
- Report mean ± std across seeds. Reject any variant with CV > 0.2 on core metrics.
- Artifacts: `audits/<feature_id>/` — tidy CSV + Markdown summary + `recommendation.json`.
- General benchmark lane: `audits/kang_ifnb/` (append-only `metrics.csv` + short per-run note).
- Required external anchors per feature (log unavailability as explicit skip rows):
  - original `spVIPES`, `contrastiveVAE` (F4/F6/F8/F9)
  - CellDISECT parity (F2/F5/F10)

### 1.3 Standard metric set

Integration metrics on `z_shared`: `iLISI`, `cLISI`, `kBET`, `kNN purity` per cell type,
`Leiden ARI`, silhouette by group/label.

Latent-quality metrics: reconstruction loss, KL shared/private, active latent dimensions,
orthogonality `corr(z_shared, z_private)` mean/worst-stratum Frobenius norm (F1).

Counterfactual metrics (F2/F5): cycle consistency, realism under target decoder,
identity preservation, OOD rejection (Mahalanobis, library-size ratio, low-likelihood).

CellDISECT-aligned metrics (F10): Pearson(mean), delta-Pearson, top-DE cosine,
Wasserstein (top-20 DE and all HVGs), CAG, MIG-shaped scores.

### 1.4 CellDISECT Kang protocol (reference)

F10a/F10b are implemented. For future benchmark runs, see `audits/F10/` for the
locked split definitions, hyperparameter settings, and parity artifact schema.
External CellDISECT rows are recorded as `status="skipped"` when the package is
unavailable; do not silently omit them.

### 1.5 Architecture inspiration tracks

- **scGen** → F2 uses condition centroid shifts as first-class perturbation mode.
- **scDisInFact / biolord** → F4 provides explicit condition/donor/batch factor control.
- **CellDISECT** → F10 provides external counterfactual/disentanglement audit metrics.
- **trVAE / MMD-VAE** → F12 (deferred): MMD alignment or conditional-decoder behavior.
- **FactorVAE / HSIC** → F11 (deferred): nonlinear dependence metrics.
- **CRADLE-VAE** → F13 (deferred): artifact/QC latent with explicit artifact labels.
- **scCausalVI / CoupledVAE** → F14 (research): causal assumptions; no core API changes.

______________________________________________________________________

## 2. Feature specifications

______________________________________________________________________

### F1 — Conditional orthogonality instrumentation ✅ CLOSED

**Artifacts:** `audits/F1/` (pass: `-1.5164%` overhead gate).
**Validation:** `pytest tests/test_disentangle_metrics.py tests/test_multimodal_disentangle.py -q`
**What it does:** Logs within-stratum `corr(z_shared, z_private)` Frobenius norm in
`extra_metrics` (`orthogonality_within_stratum`, `orthogonality_worst_stratum`) behind
`compute_orthogonality_metric=True`. No loss change.

______________________________________________________________________

### F2 — Safe counterfactual latent editing module (MVP) ✅ DONE

**Artifacts:** `audits/F10/` (smoke), tutorial notebook at `docs/notebooks/counterfactual_interventions_tutorial.ipynb`.
**Validation:** `pytest tests/test_counterfactual_basics.py tests/test_counterfactual_integration.py tests/test_counterfactual_diagnostics.py tests/test_celldisect_metric_parity.py -q`

**Public API** (`src/spVIPESmulti/interventions/`):

- `encode_cells`, `decode_counterfactual`, `predict_counterfactual`, `transfer_condition`
- `leakage_score`, `condition_separability`, `integration_report`
- Returns `CounterfactualResult` with `.X`, `.uncertainty`, `.info` (includes `ood_flags`, `rejected_mask`).
- First-class perturbation: scGen-style centroid shift `mean(z_shared|Y) - mean(z_shared|X)`.
- OOD thresholds data-derived (Mahalanobis 95th pct, library ratio `[0.5, 2.0]`, low-likelihood 5th pct).

**Non-goals (deferred to F5/F7):** Per-modality editing, perturbation vector learning,
donor/condition-conditional protocols beyond centroid shift, conditional decoder generation.

______________________________________________________________________

### F3 — Optional shared–private orthogonality loss ⚠️ ARCHIVED

**Background.** F1 measures conditional dependence; F3 penalizes residual dependence.
Smoke audit (1-seed/2-epoch) returned `reject`; real multi-seed audit also rejected
promotion. Keep implemented for manual experiments, but do not recommend a nonzero default.

**Scope.**

- Model/module kwargs (all default 0.0): `orthogonality_weight` in presets and constructors.
- Penalty term in `_compute_disentangle_losses`; `orthogonality_loss` logged only when > 0.
- Warmup: off for first 30% of epochs, linear ramp to target by 60%.
- F11 may later replace/augment with HSIC/MI-style penalties.

**TDD plan.**

1. `tests/test_orthogonality_loss.py`: positive `orthogonality_weight` constructs; `ValueError` on negative;
   default-off numerical match within 1e-6; finite loss when enabled; metric appears only when > 0.
1. Implement: kwargs → preset keys → helper reuse from F1 → warmup ramp.
1. `pytest tests/test_orthogonality_loss.py tests/test_f3_benchmark.py -q`, then full suite.

**Quantitative go/no-go benchmark.**

Run 3-seed matrix on Kang: baseline + `orthogonality_weight ∈ {0.01, 0.05, 0.10, 0.20}`.

| Criterion | Pass | Reject |
|---|---|---|
| Within-stratum mean orthogonality (vs baseline) | **≥ 20% reduction** | < 10% reduction |
| Reconstruction loss degradation | ≤ 5% | > 10% |
| iLISI degradation | ≤ 10% | > 20% |
| kBET degradation | ≤ 10% | > 20% |
| cLISI / kNN purity | within ±5% or improved | > 5% drop |
| Cross-seed CV on core metrics | ≤ 0.20 | > 0.30 |

Promotion → smallest weight satisfying all pass gates.
Artifacts: `audits/F3/metrics.csv`, `audits/F3/summary.md`, `audits/F3/recommendation.json`
(`reject`; archived/default-off).

______________________________________________________________________

### F4 — Condition/donor/batch covariate heads ✅ DONE (preset promotion rejected)

**Artifacts:** `audits/F4/` (3-seed probe: rejected preset promotion).
**Validation:** `pytest tests/test_covariate_heads.py tests/test_multimodal_disentangle.py tests/test_regression_fixes.py -q`

**What's available:** `condition_key`, `donor_key` registration in `setup_anndata`;
default-off `disentangle_batch_shared_weight`, `disentangle_donor_shared_weight`,
`disentangle_donor_private_weight`; presets `minimal_safe_bio` and `full_bio` (opt-in/manual only).

**Decision:** F4 heads and bio presets are retained for reproducibility and manual experiments.
Current audit evidence does not support recommending them as defaults.
No `batch_key` confirmed in Kang default mapping; `batch_shared` rows skipped.

______________________________________________________________________

### F5 — Donor/condition-aware counterfactual protocols ✅ DONE

**Background.** F2's MVP exposes generic latent edits. F4's `donor_key`/`condition_key`
enable rigorous per-individual counterfactual protocols.

**Artifacts:** `audits/F5/`
**Validation:** `pytest tests/test_counterfactual_protocols.py tests/test_counterfactual_basics.py tests/test_counterfactual_integration.py tests/test_counterfactual_diagnostics.py -q`

**Scope (implemented in `interventions/protocols.py`).**

- P1: unmatched private swap (random donor-i → donor-j private latent, keep shared).
- P2: label-matched private swap.
- P3: label + donor/timepoint matched with fallback-count reporting.
- Condition-shift protocol: `delta = mean(z_shared|Y) - mean(z_shared|X)` within donor i.
  Requires `condition_key` registered; raises `ValueError` otherwise.

**TDD plan.**

1. `tests/test_counterfactual_protocols.py`: shape checks for P1-P3; P3 fallback counts in
   `result.info["fallback_counts"]`; condition shift without key raises; identity protocol tolerance.
1. Implement using F2 building blocks; no new model methods.

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| Cycle consistency (latent L2 / ‖z‖₂) | ≤ 0.10 | > 0.20 |
| Realism: NB log-likelihood vs target-group cells | within 1 nat per cell | > 3 nats |
| Donor classifier agreement (donor preserved) | ≥ 0.85 | < 0.7 |
| Cell-type classifier agreement under condition shift | ≥ 0.85 | < 0.7 |
| P2 vs P1 realism delta | P2 strictly better | P2 ≤ P1 |

Artifacts: `audits/F5/counterfactuals.csv`, `audits/F5/summary.md`, `audits/F5/recommendation.json`.

______________________________________________________________________

### F6 — Knowledge-informed prototype graph regularizer (Phase 3)

**Background.** Laplacian penalty over label-derived adjacency encourages related
prototypes to cluster in `z_shared`.

**Scope.**

- `graph_regularizer_weight: float = 0.0` (default off). Adjacency from `label_key` + optional covariates.
- Penalty: `tr(P^T L P)`. Edge weighting and minimum-support filtering.
- Logged metric: `graph_regularizer_loss`.

**TDD plan.**

1. `tests/test_graph_regularizer.py`: default-off equivalence, Laplacian symmetry, finite loss,
   missing-key behavior (warn or raise).
1. Implement in `spVIPESmultimodule.py` (+ small helper in `module/utils.py`).

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| kNN purity on rare cell types | **≥ baseline + 0.05** | within baseline ± 0.02 |
| cLISI on minority types | **≥ 5% improvement** | drop |
| Reconstruction loss degradation | ≤ 5% | > 10% |
| iLISI / kBET | within ±10% | > 20% drop |
| Cross-seed CV | ≤ 0.20 | > 0.30 |

Artifacts: `audits/F6/`.

______________________________________________________________________

### F7 — Counterfactual consistency loss + perturbation vectors (Phase 3)

**Scope.**

- `counterfactual_consistency_weight: float = 0.0` (latent-only; no decoder rollout).
- Inference: `learn_perturbation_vector(model, adata, condition_pairs, latent_type, method)`,
  supporting `mean_difference`, `classifier_gradient`, `gradient_ascent`.
- `predict_perturbation_response(model, adata, perturbation_vector, magnitude)`.

**TDD plan.**

1. `tests/test_perturbation_vectors.py`: `mean_difference` recovers planted shift (cosine ≥ 0.9);
   `classifier_gradient` increases target logit; default-off matches baseline within 1e-6.
1. Implement in `src/spVIPESmulti/interventions/perturbation.py` + one loss term.

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| Recovered direction cosine (synthetic) | ≥ 0.90 | < 0.70 |
| Held-out classifier logit increase | ≥ 50% relative | < 10% |
| Loss with weight=0 matches baseline | within 1e-6 | drift |
| Training overhead with weight on | ≤ +10% wall time | > +25% |

Artifacts: `audits/F7/`.

______________________________________________________________________

### F8 — Optional SysVI-style VampPrior for shared latent (Phase 3)

**Scope.**

- `shared_prior: Literal["standard_normal", "vamp"] = "standard_normal"` (default off).
- `shared_prior_components: int = 5`, `shared_prior_trainable: bool = True`,
  `shared_prior_pseudoinput_strategy: Literal["random_cells", "stratified_labels"] = "stratified_labels"`.
- MC KL estimation for VampPrior; logged metrics: `kl_shared_prior`, `vamp_weight_entropy`.

**TDD plan.**

1. `tests/test_vampprior_shared.py`: constructor validation; default-off equivalence within 1e-6;
   finite KL/loss with `vamp`; save/load parity.
1. Implement in module + model constructor; pseudoinput init.

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| Tests | All green | Any failure |
| cLISI / kNN purity vs baseline | improved or within ±3% | > 5% drop |
| iLISI / kBET vs baseline | within ±10% | > 20% drop |
| Reconstruction loss degradation | ≤ 5% | > 10% |
| Cross-seed CV | ≤ 0.20 | > 0.30 |

Artifacts: `audits/F8/`.

______________________________________________________________________

### F9 — Optional SysVI-style latent cycle-consistency regularizer (Phase 3)

**Scope.**

- `latent_cycle_weight: float = 0.0` (default off). `latent_cycle_key: str = "sample"`.
- Random alternative category selection per cell (must differ from original).
- Standardized latent MSE between original and cycle pass means.
- Logged metrics: `latent_cycle_loss`, `latent_cycle_active_fraction`.

**TDD plan.**

1. `tests/test_latent_cycle_loss.py`: default-off within 1e-6; positive finite loss when on;
   switched category differs from source; multimodal parity finite.
1. Implement cycle helpers + training loss wiring.

**Quantitative go/no-go benchmark.**

| Criterion | Pass | Reject |
|---|---|---|
| Tests | All green | Any failure |
| iLISI improvement vs baseline | ≥ +10% | < +3% |
| Cell-type classifier on shared latent | within ±3pp | drop > 5pp |
| Reconstruction loss degradation | ≤ 5% | > 10% |
| Training overhead | ≤ +12% | > +25% |

Artifacts: `audits/F9/`.

______________________________________________________________________

### F10 — CellDISECT-aligned Kang benchmark and metrics pack ✅ F10a/F10b done

**Artifacts:** `audits/F10/` (F10b smoke; informational — external CellDISECT not installed).
**Validation:** `pytest tests/test_celldisect_metric_parity.py tests/test_celldisect_parity_runner.py -q`

**What's available:**

- F10a: `spVIPESmulti.interventions.metrics` — Pearson, delta-Pearson, top-DE cosine,
  Wasserstein, CAG, MIG-shaped scores, skipped-baseline artifact rows.
- F10b: `scripts/benchmark_kang_celldisect_parity.py` — optional parity runner; records
  explicit `status="skipped"` rows when external CellDISECT unavailable.

**F10 promotion gate** (for future real parity run with external CellDISECT installed):

| Criterion | Pass | Reject |
|---|---|---|
| Metric helper tests | All green | Any failure |
| Reproducibility across 3 seeds (Pearson/delta, EMD) | CV ≤ 0.20 | CV > 0.30 |
| Artifact completeness per split | all expected files | missing files |

______________________________________________________________________

### F11 — Nonlinear dependence diagnostics (HSIC / MI / partial correlation)

**Background.** F1's correlation norm captures only linear dependence. HSIC and
partial-correlation diagnostics can reveal nonlinear leakage.

**First actionable slice.** `hsic_rbf(z_shared, z_private, bandwidth="median")` and
`partial_corr_residualized(...)` as standalone metric helpers with synthetic tests.
Defer MI/total-correlation estimators until sample-size choices are benchmarked.

**Benchmark gate.** Promote only if metrics are finite, reproducible across 3 seeds
(CV ≤ 0.30), and explain failures not visible in F1 on at least one audit run.

Artifacts: `audits/F11/metrics.csv`, `audits/F11/summary.md`,
`audits/F11/recommendation.json`. First real Kang audit returned `iterate`: all rows
completed and hidden nonlinear signal appeared in 2/3 seeds, but HSIC CV was 0.3116
against the 0.30 promotion gate.

______________________________________________________________________

### F12 — Conditional decoder / MMD alignment track

**First actionable slice.** Implement `mmd_alignment_weight=0.0` metric/loss helper only.
Conditional decoder changes require a separate design doc (alters decoder inputs and
saved-model compatibility).

**Benchmark gate.** Promote only if conditional/MMD variants improve counterfactual
Pearson/delta-Pearson or DE recovery without degrading reconstruction, iLISI/kBET,
or cell-type retention beyond the existing F2/F4 gates.

______________________________________________________________________

### F13 — Artifact/QC latent track

**First actionable slice.** Add artifact/QC probe metrics and audit schema only. Do not
add a new latent block until at least one benchmark dataset with explicit artifact labels
is checked into the audit workflow.

**Benchmark gate.** Promote only on perturbation datasets with known artifact labels and
only if artifact removal improves QC realism without erasing perturbation signal.

______________________________________________________________________

### F14 — Causal / coupled-VAE research track

Research-only until F2/F5/F10 establish reliable counterfactual benchmarks.
No core API or architecture changes allowed until a separate design document defines
assumptions, data requirements, and failure modes.

______________________________________________________________________

## 3. Validation commands (canonical order)

```bash
pytest tests/test_disentangle_metrics.py -q                             # F1
pytest tests/test_covariate_heads.py tests/test_multimodal_disentangle.py \
       tests/test_regression_fixes.py -q                                # F4
pytest tests/test_counterfactual_basics.py tests/test_counterfactual_integration.py \
       tests/test_counterfactual_diagnostics.py -q                      # F2
pytest tests/test_orthogonality_loss.py -q                              # F3
pytest tests/test_counterfactual_protocols.py -q                        # F5
pytest tests/test_graph_regularizer.py -q                               # F6
pytest tests/test_perturbation_vectors.py -q                            # F7
pytest tests/test_vampprior_shared.py -q                                # F8
pytest tests/test_latent_cycle_loss.py -q                               # F9
pytest tests/test_celldisect_metric_parity.py -q                        # F10

# Full suite (mandatory before promotion)
pytest tests/ -q --ignore=tests/test_evaluate.py

# Smoke API combinations (after F2, F4, F5)
python scripts/smoke_vignettes.py --epochs 5 --cells_per_group 300
```

______________________________________________________________________

## 4. Acceptance criteria (apply to every feature)

A feature is "done" only when **all** are true:

1. Existing API behavior unchanged for users who do not pass new args.
1. New loss/metric terms are no-ops at default weights (numerical equivalence ≤ 1e-6).
1. New metrics appear in `extra_metrics` only when the corresponding weight > 0.
1. Targeted tests pass; full suite passes.
1. Quantitative go/no-go benchmark from §2 satisfies all "Pass" rules across ≥ 3 seeds.
1. Artifacts written to `audits/<feature_id>/` and entry appended to `audits/PROGRESS.md`.

______________________________________________________________________

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Loss-scale imbalance destabilizes early training | Default low weights, reuse warmup scaling, F1 makes leakage visible early |
| Covariate biology vs nuisance-removal conflict | New heads optional; default presets keep them off; document trade-offs |
| Graph regularizer collapses rare labels | Edge weighting + minimum-support filter |
| VampPrior pseudoinputs collapse or drift | Stratified init + entropy monitoring + standard prior fallback |
| Cycle loss over-correction harms biology | Strict weight sweep with cell-type retention gate and default-off |
| Small strata produce noisy conditional correlation | `min_cells_per_stratum` filter; report excluded strata count |
| Counterfactuals fail identity preservation | Donor/condition classifier checks + cycle-consistency tests |
| External benchmark mismatch vs CellDISECT | Lock split definitions + metric parity tests + artifact schema checks |
| Nonlinear metrics overfit small batches | Metric-only F11 first; require finite/reproducible before loss use |

______________________________________________________________________

## 6. Current next step

**F3 decision:** Complete. The real 3-seed Kang audit in `audits/F3/` rejected
promotion; keep `orthogonality_weight=0.0` by default.

**F11 status:** Metric helpers and the audit runner are implemented. The first real Kang
audit returned `iterate` because HSIC CV was 0.3116 against the 0.30 promotion gate.

**Next feature:** Decide whether to stabilize/re-audit F11 or move to the next deferred
track. F6-F9 remain Phase 3/deferred.

______________________________________________________________________

## 7. Cross-references

- Active queue + deferred backlog: `audits/PLAN.md`.
- Implementation history: `audits/PROGRESS.md`.
- Next-session pointer: `audits/HANDOFF.md`.
- Architecture & commands: `CLAUDE.md`.
