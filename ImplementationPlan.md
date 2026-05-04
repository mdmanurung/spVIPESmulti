# Feature Roadmap for spVIPESmulti

## Context

A brainstorm of features that could be added — architectural,
interpretability, or quality-of-life. The package (v1.0.0) already has a
mature core: PoE encoder/decoder, NF prior, full disentanglement objective
(including multimodal — P8 just landed), an scIB-style metrics module, and
post-training utility/plotting helpers.

This document is a **prioritized menu of follow-ups**, not a single committed
plan. Items are grouped by theme, each with rough cost (S/M/L) and the
specific file(s) it would touch. Pick which threads to pursue and a focused
implementation plan for those can be drafted from this menu.

This file is complementary to `PLANS.md` — `PLANS.md` tracks deferred
literature-derived items (P5–P7), this file is the broader roadmap including
API and interpretability work.

---

## Theme A — API streamlining & interpretability QoL (low effort, high frequency-of-use)

These remove boilerplate that every notebook currently repeats. They are the
fastest wins.

### A1. Auto-infer `group_indices_list` from `adata` (S)
**Problem.** Every public method that takes `group_indices_list` (`train`,
`get_latent_representation`) currently forces the user to write:
```python
group_indices_list = [list(map(int, g)) for g in adata.uns["groups_obs_indices"]]
```
This appears in every tutorial and is brittle (silent drift from `setup_anndata`).

**Fix.** Make `group_indices_list` optional everywhere; default to reading
`adata.uns["groups_obs_indices"]`. Add a single helper
`utils.get_group_indices_list(adata)` so power-users still have an explicit hook.

**Files.** `model/spvipesmulti.py`, `model/_training_mixin.py`, `utils.py`, all
8 tutorial notebooks (drop the boilerplate).

### A2. One-call latent extraction (S)
**Problem.** Users always run `get_latent_representation(...)` then
`utils.store_latents(...)` together. Two steps, two risks of mismatch.

**Fix.** Add `model.embed(adata=None, store=True, batch_size=512, **kwargs)`:
runs the forward pass and writes `obsm["X_spVIPESmulti_shared"]` /
`obsm["X_spVIPESmulti_private_g{i}"]` in one call. Return the array for
back-compat.

**Files.** `model/spvipesmulti.py`, `utils.py`.

### A3. Collapse `disentangle_*_weight` kwargs into a config object (S)
**Problem.** `spVIPESmulti.__init__` has 15+ parameters, 5 of which are
disentanglement weight overrides. Reading the signature is now hard.

**Fix.** Introduce `DisentanglementConfig` dataclass:
```python
DisentanglementConfig(preset="full", contrastive_weight=0.2, ...)
```
Accept either the dataclass *or* the legacy flat kwargs (deprecate, don't
remove, since v1.0.0 just shipped).

**Files.** `model/spvipesmulti.py`, `module/spvipesmultimodule.py`,
`docs/notebooks/disentangle_ablation.ipynb`.

### A4. Merge `prepare_adatas` + `setup_anndata` into one call (S)
**Problem.** Two-step setup (data prep → class-method registration) is a
common stumbling block, especially in multimodal mode where users mix up
which step writes which `.uns` key.

**Fix.** Have `prepare_adatas` / `prepare_multimodal_adatas` accept the same
kwargs as `setup_anndata` (`groups_key`, `label_key`, `batch_key`,
`modality_likelihoods`) and call it internally. Keep the class method
available for advanced cases.

**Files.** `data/_utils.py`, `model/spvipesmulti.py`.

### A5. Deterministic seeding (S)
**Problem.** No top-level `seed` parameter; users must remember to call
`torch.manual_seed`, `np.random.seed`, `pl.seed_everything` separately.

**Fix.** Add `seed: int | None = None` to `spVIPESmulti.__init__` and
`train()`. When set, route through `lightning.pytorch.seed_everything` and
seed numpy/torch globally.

**Files.** `model/spvipesmulti.py`, `model/_training_mixin.py`.

---

## Theme B — Interpretability features (medium effort, high analyst value)

Right now the only interpretation hook is `get_loadings()` + `get_top_genes()`.
Several things are missing that users expect from an scvi-family model.

### B1. Latent-aware differential expression (M)
**What.** Rank genes per latent dim with **statistical significance**, not
just loading magnitude. Use a simple decoder-perturbation strategy: hold all
other latents fixed at their posterior means, perturb one dim by ±1 SD, and
rank reconstructed gene expression by log-fold change. Bayesian DE in the
spirit of scvi's `differential_expression`.

**Output.** DataFrame with columns `dim, gene, lfc, prob_de, bayes_factor`.

**Files.** new `spVIPESmulti/de.py`, `pl.py` (volcano plot helper),
`docs/notebooks/` (new vignette).

### B2. Posterior sampling & uncertainty quantification (M)
**What.** Expose `model.get_posterior(adata, n_samples=100)` returning a
distribution-like object with `.mean`, `.std`, `.sample()`. Currently
`get_latent_representation` has `mc_samples=5000` for a single mean estimate
but doesn't surface the full distribution.

**Use case.** Per-cell uncertainty on the shared/private split — useful for
flagging cells whose private latent has high variance (likely transitional
states or low-confidence integrations).

**Files.** `model/spvipesmulti.py`, `module/spvipesmultimodule.py`.

### B3. Held-out reconstruction & NLL evaluation (S)
**What.** A `model.evaluate(adata, indices=None)` method that returns
held-out NLL per modality, reconstruction R² per gene/protein, and the
disentanglement classifier accuracy. Already partially present in the
disentanglement validation scripts but not exposed publicly.

**Why.** PLANS.md verification checklist asks for "held-out NLL — should not
degrade" but the public API has no convenient way to compute it.

**Files.** `model/spvipesmulti.py`, `metrics.py` (extend), reuse code from
`scripts/validate_disentanglement.py`.

### B4. Factor annotation helper (S)
**What.** `utils.annotate_factors(model, adata, marker_genes={"T_cell":
[...], "B_cell": [...]})` — for each latent dim, score how well its top
loadings match each marker set (Jaccard or hypergeometric). Returns a
factor → label mapping.

**Why.** Manual factor interpretation is the #1 user pain point in
disentanglement-style VAEs.

**Files.** new function in `utils.py`, plotting helper in `pl.py`.

### B5. Single-call "integration report" with metrics + plots (S)
**What.** `metrics.integration_report` already returns a DataFrame. Add
`pl.integration_report(...)` that produces a single multi-panel figure (UMAP
shared / per-group private / metric bar chart) — what users build by hand
today.

**Files.** `pl.py`.

---

## Theme C — Architectural extensions (large effort, deferred items from PLANS.md)

These already have design notes in `PLANS.md`. Listed here for
completeness; pick at most one per release cycle.

| Item | Source | Cost | Notes |
|---|---|---|---|
| **C1.** Counterfactual cross-group augmentation (P5) | CellDISECT | L | ~30–50% compute overhead; needs per-group private posterior bank |
| **C2.** Multi-covariate PoE (P6) | CellDISECT | L | Major refactor of `groups_lengths` → nested covariate metadata; affects every component |
| **C3.** Reference-group decoder masking (P7) | Multi-ContrastiveVAE | M | Asymmetric comparisons only; risk of degenerate solution |
| **C4.** MuData I/O for multimodal mode | scverse standard | M | Convert `prepare_multimodal_adatas` output ↔ MuData; ecosystem alignment |
| **C5.** Hyperparameter search recipe (Optuna) | — | M | Notebook + helper for tuning `n_dimensions_*`, weight schedules; no hard dependency |

---

## Theme D — Operational / training UX (small, polish-y)

### D1. Checkpoint resumption / fine-tuning API (S)
**What.** Document `save()`/`load()` for resuming or fine-tuning. Currently
inherits from scvi `BaseModelClass` with no spVIPES-specific examples.

**Files.** `docs/notebooks/` (new short vignette), README.

### D2. Training diagnostic auto-checks (S)
**What.** After `train()`, automatically check for: KL collapse (per-dim KL
< 0.01), classifier accuracy plateau (when disentanglement is on), NaN/Inf in
the loss curve. Print a single diagnostic block. Can be silenced.

**Files.** `model/_training_mixin.py`.

### D3. Mixed-precision shortcut (S)
**What.** `model.train(..., precision="bf16-mixed")` already works through
`trainer_kwargs`, but it's undocumented. Add a one-line README example +
docstring.

**Files.** README, `model/_training_mixin.py` docstring.

---

## Recommendation: where to start

Theme A (especially A1, A2, A4) is the highest **value/effort ratio** — it
removes friction every notebook and example currently has, and is fully
backward-compatible.

Theme B (B1 differential expression, B4 factor annotation) is the highest
**value to analysts** trying to interpret a trained model. Both are net-new
capability.

Theme C is heavyweight; defer until A+B settle.

---

## Verification (whichever items get picked)

- Run `pytest -v` (current baseline: 23 passed, 1 skipped).
- Run `python scripts/smoke_vignettes.py` (8/8 pass after P0 fixes).
- For Theme B items, run `scripts/validate_disentanglement.py` to confirm no
  regression in `z_shared`/`z_private` separation metrics.
- For each new public function, add a unit test in `tests/` and a one-line
  example to the corresponding tutorial notebook.

## Critical files for any change in Themes A/B

- `src/spVIPESmulti/model/spvipesmulti.py` — public model class
- `src/spVIPESmulti/model/_training_mixin.py` — `train()` signature
- `src/spVIPESmulti/module/spvipesmultimodule.py` — Lightning module, loss
- `src/spVIPESmulti/utils.py` — post-training helpers
- `src/spVIPESmulti/pl.py` — plotting helpers
- `src/spVIPESmulti/metrics.py` — scIB-style metrics, `integration_report`
- `src/spVIPESmulti/data/_utils.py` — `prepare_adatas` /
  `prepare_multimodal_adatas`
