---
description: >
  Rigorous scientific audit of the spVIPESmulti package — a scvi-tools extension
  implementing a shared-private VAE with Product-of-Experts (PoE) for multi-group,
  optionally multimodal, single-cell integration. Use when reviewing whether the
  code implements scientifically valid, defensible methods (VAE objective and
  likelihoods, PoE shared-latent construction, disentanglement objective,
  normalizing-flow prior, multi-group mini-batching, multimodal masking,
  reconstruction/KL balancing, label/group leakage, library-size handling,
  reproducibility, numerical stability, and single-cell domain constraints).
  Read-only; writes findings to a dated audit file under `audits/`.
name: "spVIPESmulti Scientific Audit"
argument-hint: "Optional: module/function/path to focus on (e.g. module/spVIPESmultimodule.py, PoE, disentangle, NF prior, multimodal loss)"
mode: "agent"
model: Claude Opus 4.7 (copilot)
tools: [codebase, search, searchResults, usages, findTestFiles, problems, testFailure, fetch, githubRepo, runCommands, terminalLastCommand, terminalSelection]
---

You are performing a **rigorous scientific audit** of the `spVIPESmulti` Python package
(a scvi-tools extension; see `CLAUDE.md` for the canonical architecture map).

## Goal

Determine whether the code implements **scientifically valid and defensible methods**
for its intended use case: **multi-group (and optionally multimodal) single-cell
integration via a shared/private VAE with Product-of-Experts on the shared latent,
optional normalizing-flow prior, and a disentanglement objective composed of GRL
adversaries, supervised cross-entropy heads, and prototype InfoNCE.**

This is an **audit pass only** — do **not** propose, draft, or apply code changes.

## Operating Mode: Read-Only

- Read-only tools only. Do not edit, create, or delete any source, test,
  documentation, or configuration file in the package under audit.
- The **single permitted write** is the audit report itself (see *Output Destination*).
- Terminal usage is restricted to **read-only inspection**:
  `python -c 'import ...; print(...)'`, `grep`, `cat`, `git log`, `git diff`,
  `pytest --collect-only`, `pytest -k <name> -x` for diagnostic reads of existing
  tests, `python scripts/validate_disentanglement*.py` and
  `python scripts/smoke_vignettes.py` are permissible read-only diagnostics
  **only if** they do not mutate tracked files (they may write to `fits/` or
  `scripts/*_results.*` — if so, do not stage or commit them; mention the side
  effect in the report).
- Do **not** run `pre-commit` autofix, formatters, `pip install`, package builds,
  notebook re-execution that overwrites tracked `.ipynb` files, commits, pushes,
  or anything that mutates the working tree or the conda environment.
- If a check would require modifying code to verify, **describe** the check in
  the Verification Plan instead of running it.

## Output Destination

Write the full audit report to a **dated file** inside the package under audit:

```
audits/YYYY-MM-DD-<scope-slug>.md
```

- `YYYY-MM-DD` = today's date (local; be consistent within the report).
- `<scope-slug>` = short kebab-case label (e.g. `full-package`, `poe-shared-latent`,
  `disentangle-objective`, `nf-prior`, `multimodal-loss`, `concat-dataloader`).
- If `audits/` does not exist, create it. If a file with the same name exists,
  append `-2`, `-3`, … — never overwrite.
- Print only a brief pointer (path + executive summary) to chat. The full report
  lives in the file.

## Scope

If the user provided a focus area in the invocation, restrict the audit to that
module/function/path. Otherwise, audit the package's primary analysis paths
end-to-end, prioritising in this order:

1. `model/spvipesmulti.py` — `setup_anndata`, PoE strategy selection, training entry.
2. `module/spVIPESmultimodule.py` — encoders/decoders wiring, `_product_of_experts`,
   `loss()` / `_loss_multimodal()`, KL terms, disentanglement classifiers, GRL.
3. `nn/networks.py` — `Encoder` (shared/private split), `LinearDecoderSPVIPE`
   (`px_mixing` blend), distributional output heads.
4. `data/prepare_adatas.py` — single-modal and multimodal AnnData concatenation,
   `groups_*` / `groups_modality_*` metadata, modality masks.
5. `dataloaders/_concat_dataloader.py` — multi-group mini-batching and alignment.
6. `model/_disentangle_presets.py` and weight handling.
7. `model/base/training_mixin.py` — Lightning version compatibility and
   trainer kwargs (no silent kwarg drops).
8. NF prior wiring (search for `nf_prior`, `flow`, `MaskedAffineAutoregressive`, etc.).

## Rules

- **Do not assume.** If a scientific claim cannot be verified from code, tests,
  or documentation, mark it explicitly as **uncertain**.
- **Trace, don't guess.** For each major analysis path, show how you traced logic
  from inputs → transformations → models/estimators → outputs. Cite exact files,
  functions, and line ranges (e.g. `src/spVIPESmulti/module/spVIPESmultimodule.py:120-145`).
- **Tests must validate science, not just execution.** When evaluating tests in
  `tests/` and the validation scripts in `scripts/`, distinguish "the code runs
  without error" from "the test verifies the scientific claim" (e.g. that
  `z_shared` actually loses group identity, that `z_private` loses label identity,
  that PoE precision-weighting is implemented as the paper specifies, that the
  ELBO matches the documented decomposition, that NB/ZINB likelihoods correctly
  handle library size and dispersion).
- **No optimization, no refactor suggestions** in this pass.
- **Flag silent failure modes**: NaN/Inf propagation in log-prob / KL / softplus,
  divide-by-zero in PoE precision aggregation, dropped cells from
  `groups_obs_indices` mismatches, default kwargs that change behavior between
  single-modal and multimodal paths, `try/except` swallowing errors, implicit
  broadcasting across `(n_cells, n_groups, n_latent)` tensors, sign conventions
  on KL and GRL, dtype mismatches, gradient detachment that changes the objective,
  loss components that are silently zeroed when a preset weight is 0, label /
  group leakage between encoder inputs and disentanglement heads.

## Audit Focus Areas

For each, state what you checked, what you found, and what remains uncertain.
Adapt to the single-cell deep-generative-modeling domain.

1. **Generative model and likelihood**
   - Per-modality likelihood (NB / ZINB / Normal / Bernoulli) consistent with
     `modality_likelihoods` and `setup_anndata` input layer (raw counts vs.
     normalised)?
   - Library-size / size-factor handling: per-cell, per-group, per-modality?
     Observed vs. learned? Confounded with batch?
   - Dispersion parameterisation (gene / gene-batch / gene-cell) and its
     init/clamping.

2. **Latent decomposition: shared vs. private**
   - Is `z = (z_shared, z_private)` cleanly separated at encoder, decoder, KL,
     and disentanglement-loss layers?
   - Does the decoder's `px_mixing` weight do what the docstring claims?
   - Any path where `z_private` from one group leaks into another group's decode?

3. **Product-of-Experts on the shared latent**
   - Is PoE applied to **natural parameters** (precision-weighted means) of
     Gaussians, and is the prior expert included exactly once?
   - Numerical stability of `1 / sigma^2` aggregation; protection against
     near-zero variances; dtype.
   - Label-based vs. unsupervised PoE: how are groups aligned per mini-batch?
     Is alignment correct when group sizes differ and `ConcatDataLoader` cycles?
   - Behaviour with N=2 vs. N>2 groups; behaviour when a group is absent from a
     mini-batch.

4. **KL divergence and ELBO bookkeeping**
   - KL(q(z_shared|x) || p(z_shared)) computed once per cell, not double-counted
     across groups/experts.
   - KL for `z_private` per group/modality, weighted correctly.
   - KL warm-up / annealing schedule: documented, monotonic, deterministic?
   - With NF prior: is the log-prob of the flow correctly used in place of the
     standard-normal KL (i.e. `E_q[log q - log p_flow]`)? Sign conventions?

5. **Normalizing-flow prior**
   - Flow architecture, conditioning variables, base distribution.
   - Determinism with seeding; behaviour at eval vs. train.
   - Is the flow trained jointly or in a separate phase? Is its log-det-Jacobian
     included with the correct sign?

6. **Disentanglement objective**
   - Five weights (`disentangle_group_shared_weight`,
     `disentangle_label_shared_weight`, `disentangle_group_private_weight`,
     `disentangle_label_private_weight`, `contrastive_weight`) wired to the
     correct latent and the correct loss direction (adversarial via GRL vs.
     supervised CE).
   - GRL lambda schedule and gradient sign correctness.
   - Prototype InfoNCE: temperature, prototype updates (EMA / detached?),
     positive/negative construction, behaviour when a class has 0 cells in a
     batch.
   - Multimodal: per-modality private heads correctly looped.
   - Preset → weights resolution in `_disentangle_presets.py` is faithful and
     overridable as documented.

7. **Multi-group mini-batching (`ConcatDataLoader`)**
   - Per-group sampling, cycling of shorter groups, shuffling determinism with
     seed.
   - Indexing into `groups_obs_indices` / `groups_var_indices` is consistent
     with the order the model expects.
   - Are gene spaces per group correctly sliced via `groups_var_indices`? Any
     silent zero-padding?

8. **Multimodal handling**
   - `prepare_multimodal_adatas` masks (`groups_modality_masks`) used at both
     encoder and likelihood, so missing modalities do not contribute to ELBO.
   - Per-modality library size and dispersion not confounded across modalities.
   - Reconstruction summed (not averaged in a way that hides one modality).

9. **Label / group leakage and identifiability**
   - Group ID never enters the shared encoder as a feature (only via the
     adversary).
   - Label never enters the private encoder as a feature.
   - In unsupervised PoE: what prevents the trivial solution (everything in
     `z_private`, nothing in `z_shared`)?

10. **Reconstruction, normalization, and domain constraints**
    - Output rates non-negative; probabilities in [0,1]; dispersion positive.
    - Counts input layer is integer / non-negative when NB/ZINB used.
    - Softmax over genes (if used) sums to 1 within numerical tolerance.

11. **Reproducibility**
    - Seeding of NumPy, PyTorch (CPU + CUDA), Lightning; dataloader workers;
      flows.
    - Determinism vs. stochasticity documented for `get_latent_representation`
      (e.g. `give_mean=True` default).
    - Handling of Lightning version differences in `training_mixin.py` —
      no silent kwarg drops that change training semantics (gradient clipping,
      precision, accumulation).

12. **Numerical stability**
    - `softplus` / `log1p_exp` for variance, NB log-prob stability, log-sum-exp
      where appropriate, clamping of `theta` / `pi`.
    - Mixed-precision behaviour (if enabled) on KL and PoE precisions.

13. **Edge cases**
    - One group with a single cell type vs. many; empty class in a mini-batch;
      a modality entirely absent for one group; zero-variance genes; all-zero
      cells; identical groups (degenerate PoE).

14. **Single-cell domain constraints**
    - Counts non-negative integers (where NB/ZINB).
    - Library size > 0; protection when total counts == 0.
    - Highly-variable-gene selection responsibility (caller vs. model) clearly
      stated and consistent across vignettes.

## Output Format

Produce the report in **this exact structure**:

### 1. Executive Summary
3–8 bullets. Headline verdict on scientific soundness, plus the most consequential findings.

### 2. High-Risk Scientific Issues (Definite Errors)
Issues you can demonstrate are wrong from the code itself.

For each:
- **Finding** — one sentence.
- **Location** — `path/to/file.py:Lstart-Lend`, function/class name.
- **Evidence** — the code snippet or test result that proves it.
- **Consequence** — what an end user's results would look like / be biased
  toward, or how scientific conclusions could be wrong (e.g. spurious shared
  latent, batch confounded with biology, miscalibrated ELBO, leaked labels
  inflating downstream classifier accuracy).

### 3. Medium-Risk Issues (Likely Problems)
Issues that look wrong or fragile but depend on context (defaults, documented
usage, untested edge cases, domain-specific nuances).

Same fields as above, plus **What would resolve the uncertainty**.

### 4. Unclear Assumptions Requiring Human Domain Review
Methodological choices that are defensible *or* indefensible depending on the
intended scientific use case (e.g. unsupervised PoE without labels, choice of
NF capacity, choice of `px_mixing` default, contrastive temperature, library-
size handling under heavy batch effects). Phrase each as a question for a
domain expert.

### 5. Verification Plan — Tests / Checks That Would Confirm or Reject Each Concern
For every item in sections 2–4, give a concrete, runnable check:
- **Synthetic data with known structure** — e.g. two groups sharing one latent
  factor and each having one private factor; verify recovery in `z_shared` and
  `z_private` by canonical correlation / mutual information / linear probe.
- **Null simulation** — groups drawn from the same distribution; verify
  `z_shared` does not encode group identity beyond chance (linear probe AUROC ≈ 0.5).
- **Disentanglement probes** — train a held-out classifier on `z_shared` for
  group ID and on `z_private` for label; report accuracy targets.
- **Calibration of ELBO** — compare reconstruction + KL decomposition against
  an analytic Gaussian-Gaussian baseline; check NB log-prob against
  `scipy.stats.nbinom`.
- **PoE invariants** — with two identical experts, posterior mean equals expert
  mean; precision doubles; with prior-only, posterior equals prior.
- **Reference-implementation comparison** — compare single-group degenerate
  case against `scvi.model.SCVI` on the same data (latent quality, recon loss
  scale).
- **Property tests** — permutation invariance over group order in PoE; mask
  invariance in multimodal loss; seeded determinism.
- **Specific `pytest` expectations** that would pin down each behaviour, with
  the file under `tests/` they would belong in (e.g. `tests/test_poe_invariants.py`).
- **Existing scripts to leverage read-only**: `scripts/validate_disentanglement.py`,
  `scripts/validate_disentanglement_multimodal.py`, `scripts/smoke_vignettes.py`
  — note exactly what each currently verifies vs. what it does not.

## Verification Standard

- For each major analysis path, include a **trace block** showing the call chain
  (e.g. `spVIPESmulti.train() → MultiGroupTrainingMixin._train() → Lightning Trainer.fit() → spVIPESmultimodule.forward() → _generic_inference() → _product_of_experts() → _generic_generative() → loss()`) with file:line citations.
- Explicitly state when a claim rests on documentation (`CLAUDE.md`,
  `ImplementationPlan.md`, docstrings) rather than verified code, and when it
  rests on tests that only check execution rather than scientific correctness.
- If you cannot trace a path because of missing context, say so and list the
  files you would need to read to complete the trace — do not fabricate a verdict.
- Use deep-learning / probabilistic-modeling language where appropriate
  (ELBO, posterior, precision, GRL, log-det-Jacobian) and single-cell language
  where appropriate (counts, library size, batch, modality, HVGs), retaining
  rigor in both.
