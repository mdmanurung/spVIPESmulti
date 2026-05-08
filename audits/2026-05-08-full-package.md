# spVIPESmulti — Scientific & Statistical Audit

**Date:** 2026-05-08
**Scope:** full-package (read-only audit; no code changed)
**Auditor mode:** trace-only; trace files cited as `path:Lstart-Lend`.
**Primary surfaces audited:**
- `src/spVIPESmulti/module/spVIPESmultimodule.py` (VAE math, PoE, losses, disentanglement)
- `src/spVIPESmulti/nn/networks.py` (encoder/decoder)
- `src/spVIPESmulti/module/utils.py` (likelihood factory, gradient reversal)
- `src/spVIPESmulti/model/spvipesmulti.py` (`differential_abundance`, `evaluate`,
  `get_latent_representation`, `_aggregate_shared_posterior`)
- `src/spVIPESmulti/metrics.py` (iLISI/cLISI/kBET/silhouette/recon error)
- `src/spVIPESmulti/traversal.py` (per-dim gene response)
- `src/spVIPESmulti/data/prepare_adatas.py` (data assembly)
- `src/spVIPESmulti/model/_disentangle_presets.py`

This audit is **read-only**. No source, test, doc, or config file was edited.
The single write is this report.

---

## 1. Executive Summary

- **Likelihood / data mismatch (CRITICAL).** When `log_variational_generative=True`
  (the default), the negative-binomial reconstruction is computed against
  `log1p(counts)` — fractional, non-integer values that lie outside NB
  support. The integer-roundness guard is explicitly bypassed
  ([`spVIPESmultimodule.py:1273-1283`](src/spVIPESmulti/module/spVIPESmultimodule.py#L1273-L1283)).
  Every reported reconstruction loss in the default training mode is therefore
  evaluating a count distribution at non-count targets; gradients still flow,
  so training "works", but the loss has no probabilistic meaning. Affects
  single-modal **and** multimodal NB modalities.
- **PoE across unrelated cells (CRITICAL semantics).** The unsupervised
  ([`_poe_n`](src/spVIPESmulti/module/spVIPESmultimodule.py#L302-L440)) and
  label-based ([`_label_based_poe`](src/spVIPESmulti/module/spVIPESmultimodule.py#L739-L893))
  PoE combine encoder posteriors of **arbitrarily-ordered cells from
  different groups by row index**. Cells are not paired in any biological
  sense — a cell's PoE posterior is a precision-weighted product with whatever
  other-group cell happens to share its mini-batch row. Result: stochastic,
  order-dependent shared posteriors and downstream embeddings.
- **Gaussian likelihood cannot fit log-normalized data (CRITICAL for
  multimodal CITE-seq tutorial).** The Gaussian branch of `build_likelihood`
  uses the L1-normalized simplex `px_scale ∈ [0,1]` as the mean
  ([`module/utils.py:43-46`](src/spVIPESmulti/module/utils.py#L43-L46)), so the
  protein/log-normalized modality is forced to live on the unit simplex while
  observed values are typically in `[-3, 6]`. NLL is dominated by an
  un-fittable scale offset.
- **`differential_abundance()` is not a hypothesis test.** Returns a signed
  squared-Mahalanobis difference per cell with no null distribution, no
  p-value, no FDR control, and no CI
  ([`spvipesmulti.py:798-846`](src/spVIPESmulti/model/spvipesmulti.py#L798-L846)).
  The accompanying warning is also conceptually inverted: when shared-latent
  alignment **succeeds**, `mu_a ≈ mu_b` collapses the score toward zero. Tests
  cover sign and shape only; no calibration test exists
  ([`tests/test_differential_abundance.py`](tests/test_differential_abundance.py)).
- **`get_latent_representation(..., normalized=False, give_mean=True)` returns a
  single posterior sample, not the mean** ([`spvipesmulti.py:1413-1432`](src/spVIPESmulti/model/spvipesmulti.py#L1413-L1432)).
  All embeddings stored by `embed()` and consumed by `evaluate`, UMAP,
  iLISI/cLISI, kBET, etc. are stochastic single rsamples. Documented behavior
  ("Give mean of distribution or sample from it") is not honoured for the
  default un-normalized path.
- **kBET / iLISI / cLISI implementations are heuristic proxies.**
  Naming matches the published statistics
  ([Büttner 2019](https://doi.org/10.1038/s41592-018-0254-1); [Korsunsky
  2019](https://doi.org/10.1038/s41592-019-0619-0)) but the math is not.
  kBET reports `exp(-mean(chi2))` instead of the rejection rate;
  iLISI/cLISI use unit-weighted Simpson over kNN instead of Gaussian
  perplexity-tuned weights.
- **Silhouette is not per-group despite the function name and column.**
  `per_group_silhouette` returns one global value computed on pooled cells
  and `integration_report` writes that same scalar to every per-group row
  ([`metrics.py:194-228, 295-316`](src/spVIPESmulti/metrics.py#L194-L316)).
- **Several silent or fragile numerical paths**: NF-prior KL is a 1-sample
  Monte Carlo estimate (high variance, unreported);
  `_jeffreys_divergence_loss` aggregates each batch to a single moment-matched
  Gaussian by averaging means and variances — this is **not** symmetric KL
  between the two posteriors and is a biased proxy.
- **Overall verdict.** Engineering is solid (eval-mode toggling, GRL,
  preset wiring, weight overrides). The *probabilistic core* (likelihood,
  PoE semantics, posterior-summary APIs) and the *evaluation layer*
  (DA test, integration metrics) carry correctness defects that bias every
  reported number. Most are fixable without architectural change but require
  acknowledgement before any quantitative claim is published.

---

## 2. High-Risk Scientific / Statistical Issues (Definite Errors)

### 2.1 NB likelihood evaluated at log1p-transformed targets

- **Finding** — Default training computes `−NB.log_prob(log(1+x))` instead of
  `−NB.log_prob(x)`; targets are not in the NB support.
- **Location** — `src/spVIPESmulti/module/spVIPESmultimodule.py:1259-1287`
  (single-modal `loss`); `src/spVIPESmulti/module/spVIPESmultimodule.py:1351-1399`
  (multimodal `_loss_multimodal`). Strict-support guard skipped via
  `transformed_for_nb=True` at L1271-1283 / L1391-1397.
- **Evidence** —
  ```python
  if self.log_variational_generative:
      x_target = torch.log(1 + x_obs)
      ...
  recon_loss = -generative_outputs["private_poe"][str(g)]["px"].log_prob(x_target).sum(-1)
  ```
  `px` is `NegativeBinomialMixture` constructed from raw counts' `library`
  via `px_rate = exp(library) * px_scale`, so the rate is on the count scale
  but the target is on the log1p scale. `_validate_likelihood_observations`
  is called with `transformed_for_nb=True`, which **disables** the
  integer-roundness check that would otherwise catch this.
- **Consequence** — The loss is not the negative log-likelihood of the
  observed counts under the model; ELBO, reconstruction-loss diagnostics,
  validation NLL, and any held-out-NLL comparison reported in
  `evaluate(...).held_out_metrics` are mis-scaled in ways that depend on
  library size (because `log(1+x) ≪ x` for large counts). Model fit may
  still occur because gradients of `-NB.log_prob(log1p(x))` w.r.t. `px_rate`
  are still informative, but selection on this loss (e.g. choosing
  `n_dimensions_shared`, comparing presets) is comparing the wrong
  quantities.

### 2.2 PoE combines unrelated cells across groups by row index

- **Finding** — Across-group product of experts is computed row-wise on
  per-group encoder outputs. There is no biological correspondence between
  row `i` of group A's posterior batch and row `i` of group B's posterior
  batch; these are just whatever cells happened to land at index `i` after
  the `ConcatDataLoader`'s independent shuffles.
- **Location** —
  - `_poe_n` pads each group to `max_batch_size` and stacks along a new
    leading dim before calling `_product_of_experts`:
    `src/spVIPESmulti/module/spVIPESmultimodule.py:392-440`.
  - `_label_based_poe` masks each group to cells of the current label and
    feeds the masked tensors directly into `_poe_n` — pairing is again
    by row order within the label-masked sub-batch:
    `src/spVIPESmulti/module/spVIPESmultimodule.py:739-893`.
  - `_product_of_experts` computes `mus_joint = sum_k mu_k / var_k`
    along the leading dim, which is the across-group axis:
    `src/spVIPESmulti/module/spVIPESmultimodule.py:614-622`.
- **Evidence** —
  ```python
  # _poe_n
  stacked_mus = torch.stack(padded_locs, dim=0)        # (N_groups, max_batch, latent)
  stacked_logvars = torch.stack(padded_logvars, dim=0)
  mus_joint, logvars_joint = self._product_of_experts(stacked_mus, stacked_logvars)
  ```
  The PoE is then sliced back per group:
  `g_mu = mus_joint[:g_size]`. Cell *i* of group g receives a posterior
  that is the precision-weighted product of its own encoder *and* the
  encoders of cells `i` from all other groups (or the unit prior when the
  other group ran out of cells). Those cells are unrelated.
- **Consequence** — The shared posterior for any given cell is
  **non-deterministic across reshuffles**, depends on minibatch
  composition, and is not a valid posterior under any joint generative
  model that pairs cells. Results (UMAPs, DA scores, integration metrics)
  vary across runs / dataloader seeds in ways not attributable to model
  stochasticity but to arbitrary row alignment. The label-based variant
  partially mitigates this by restricting to same-label cells, but pairing
  is still arbitrary within the label sub-batch.

### 2.3 Gaussian likelihood mean is a probability simplex

- **Finding** — For `likelihood_type="gaussian"` the Normal mean is
  `px_scale = L1_normalize(mixing*px_rate_priv + (1-m)*px_rate_shared)`,
  i.e. each row sums to 1 and every coordinate lies in `[0, 1]`. Real
  CITE-seq protein data after CLR / log-normalisation typically spans
  `[-3, 6]`, and even after centring is unbounded, so the model cannot
  represent the data location at all.
- **Location** —
  - `LinearDecoderSPVIPE.forward` produces the simplex
    `px_scale`: `src/spVIPESmulti/nn/networks.py:357-365`.
  - `build_likelihood` for Gaussian:
    `src/spVIPESmulti/module/utils.py:42-46` — `mean = px_scale`.
  - Per-feature scale comes from `log_scale_gaussian`, but `mean` itself is
    bounded.
- **Evidence** —
  ```python
  px_scale = torch.nn.functional.normalize(
      mixing * px_rate_private + (1 - mixing) * px_rate_shared, p=1, dim=-1
  )
  ...
  mean = px_scale
  scale = torch.exp(log_scale).clamp(min=1e-4).expand_as(mean)
  return Normal(loc=mean, scale=scale)
  ```
- **Consequence** — Gaussian-modality reconstruction loss is dominated by a
  rigid offset between observed log-protein values and the simplex output.
  Per-feature `log_scale` will inflate to absorb the residual variance,
  making the likelihood near-flat and gradient signal on `z` weak. The
  multimodal CITE-seq tutorial therefore cannot meaningfully fit the
  protein modality, and the reported reconstruction loss for any Gaussian
  modality is uninformative. The Gaussian path also receives a
  `log_variational_generative` log1p applied **only when
  `likelihood_type=="nb"`** (correct in `_loss_multimodal:1391-1395`), so
  that part is fine.

### 2.4 `differential_abundance` returns a heuristic, not a test

- **Finding** — Returns a continuous score
  `score(c) = ‖(z_c − μ_a)/σ_a‖² − ‖(z_c − μ_b)/σ_b‖²` with no null
  distribution, no p-value, no q-value, no CI, and no calibration check.
  The published methods this most resembles (e.g. Milo, MELD, scIB-DA)
  estimate sample-level abundance with a calibrated null.
- **Location** — `src/spVIPESmulti/model/spvipesmulti.py:798-867` and
  `_aggregate_shared_posterior` at `:651-705`.
- **Evidence** —
  ```python
  d_a = np.sum(np.square((z - mu_a) / scale_a), axis=1)
  d_b = np.sum(np.square((z - mu_b) / scale_b), axis=1)
  score_values[idxs_arr] = (d_a - d_b).astype(np.float32)
  ```
  `μ_a, μ_b` are means of per-(group, sample) posterior means; `σ_a, σ_b`
  are RMS of per-(group, sample) posterior scales.
- **Consequence** — Users have no way to declare a cell "differentially
  abundant" at any controlled error rate. There is no test of whether the
  score distribution in group A differs from that in group B beyond what
  would be expected by chance. The warning at `:792-797` is also inverted
  in implication: when `disentangle_group_shared_weight > 0` and PoE works,
  groups share the latent, `μ_a → μ_b`, and the score collapses; the
  warning fires only when one would expect the score to be most usable.

### 2.5 `get_latent_representation(give_mean=True, normalized=False)` returns a sample

- **Finding** — In the un-normalized branch the function ignores the
  `give_mean` flag and stores `qz.rsample()` instead of the posterior mean.
- **Location** —
  `src/spVIPESmulti/model/spvipesmulti.py:1413-1432` (shared) and
  `:1434-1448` (private).
- **Evidence** —
  ```python
  if not normalized:
      latent_shared[g].append(poe_log_z.cpu())   # rsample, regardless of give_mean
  else:
      ...
      if give_mean:
          samples = qz_poe.sample([mc_samples])
          theta = torch.nn.functional.softmax(samples, dim=-1).mean(dim=0)
  ```
  `poe_log_z = qz.rsample()` is set inside `_supervised_poe`/`_label_based_poe`
  (`spVIPESmultimodule.py:670-686`, `:887-893`).
- **Consequence** — All embeddings written by `embed()` (default
  `normalized=False`) and consumed by `evaluate()`, UMAP helpers, iLISI/cLISI,
  kBET, kNN-purity, Leiden ARI, traversal seed sampling, and DA score
  computation are stochastic single-sample noise around the posterior mean,
  not the mean itself. Re-running `embed()` produces different
  numbers; reported integration metrics will not be reproducible across
  invocations.

### 2.6 `per_group_silhouette` is global; columns mislabeled

- **Finding** — `per_group_silhouette` computes a single silhouette on
  pooled per-group private embeddings and returns one scalar; in
  `integration_report` this same scalar is written to every per-group row.
- **Location** — `src/spVIPESmulti/metrics.py:194-228` and `:295-316`.
- **Evidence** —
  ```python
  sil = per_group_silhouette(all_z, all_g)       # one scalar
  for group_name in z_private_dict:
      rows.append({..., "silhouette": sil})       # same value in every row
  ```
- **Consequence** — Per-group silhouette diagnostics in tutorial outputs
  and `evaluate(include_private=True)` reports a constant; no per-group
  separation can actually be inspected via this column. Misleads users
  comparing group-specific representation quality.

### 2.7 kBET/iLISI/cLISI are not the published statistics

- **Finding** — The functions named `kbet`, `ilisi`, `clisi` do not implement
  the kBET (Büttner 2019) or LISI (Korsunsky 2019) statistics.
  - `kbet` averages the per-cell χ² distance to expected group frequencies
    and returns `exp(-mean)`. The published kBET statistic is the rejection
    rate of a per-neighbourhood χ² test at a fixed α; that requires a
    comparison to χ²_{n_groups−1} percentiles, which is absent here.
  - `ilisi`/`clisi` compute equally-weighted Simpson diversity over the
    raw kNN set. Published LISI uses Gaussian perplexity-tuned weights so
    the statistic is comparable across local densities.
- **Location** — `src/spVIPESmulti/metrics.py:30-141`.
- **Consequence** — Reported "kBET" / "iLISI" / "cLISI" numbers are not
  comparable to literature values, even up to monotone transformation,
  because the weighting and the null-comparison logic differ. Users will
  reasonably assume otherwise from the docstrings ("Inverse Simpson's
  diversity index over k-NN neighbours" cites the same intuition as LISI;
  "Chi-squared proxy for kBET" hints at the gap but is easily overlooked).

---

## 3. Medium-Risk Issues (Likely Problems)

### 3.1 Jeffreys integration loss aggregates each batch to a single Gaussian

- **Finding** — `_jeffreys_divergence_loss` reduces a `(B, D)` batch of
  posteriors to a single normal whose mean is the average of means and
  whose variance is the average of variances, then computes symmetric KL
  between the two scalars. This is **not** moment-matching to the
  marginal mixture (the correct mixture variance is
  `mean(var) + var(mean)`), and it is not symmetric KL between the
  posteriors of the two groups.
- **Location** — `src/spVIPESmulti/module/spVIPESmultimodule.py:601-622`.
- **Evidence** —
  ```python
  var1 = logvar1.exp().mean(dim=0)
  var2 = logvar2.exp().mean(dim=0)
  agg_mu1 = mu1.mean(dim=0)
  agg_mu2 = mu2.mean(dim=0)
  ...
  return (kl(rv1, rv2) + kl(rv2, rv1)).sum()
  ```
- **Consequence** — A loss that should measure distributional alignment
  between two groups instead measures alignment between the means of the
  per-group encoder means, ignoring within-group dispersion entirely.
  Underestimates inter-group divergence whenever within-group variance is
  comparable to between-group variance — which is the regime of interest.
- **What would resolve the uncertainty** — Compare the implemented loss
  to a sample-based MMD or to a true Gaussian mixture symmetric-KL
  upper bound on a synthetic two-group dataset with known overlap.

### 3.2 NF-prior KL is a single-sample Monte Carlo estimator

- **Finding** — `_nf_kl` returns `log q(z|x) − log p_flow(z)` for a single
  rsample `z`. This is unbiased but high-variance, and the variance is
  not tracked or bounded. `nf_target="both"` doubles the noise. No
  stop-gradient on the flow when target is private (or on the encoder when
  fitting the flow) is inspected.
- **Location** — `src/spVIPESmulti/module/spVIPESmultimodule.py:624-651`,
  consumed at `:1287-1297`.
- **Consequence** — When the NF prior is enabled, ELBO gradients carry
  injected noise on the order of `log p_flow` per cell per step, producing
  noisy training and unstable preset comparisons. NF and standard prior
  ELBOs are not directly comparable.
- **What would resolve** — Track Monte Carlo variance over the batch and
  compare against the analytic KL when the flow is forced to standard
  normal initialisation.

### 3.3 `reconstruction_error` Poisson NLL drops `log(x!)` and uses only `px_rate_shared`

- **Finding** — Reports
  `mean(-x*log(rate) + rate)` per cell. Two issues:
  (i) the Stirling term `log(x!)` is dropped (constant w.r.t. parameters
  but the value is no longer a log-likelihood and cannot be compared to
  external Poisson NLLs);
  (ii) `rate` is `px_rate_shared` only, ignoring the mixture component
  used in training, so it is inconsistent with what the model actually
  optimises.
- **Location** — `src/spVIPESmulti/metrics.py:494-512`.
- **What would resolve** — Either label the column as "Poisson rate
  cross-entropy" or use `torch.distributions.Poisson(rate).log_prob(x)` and
  mix `px_rate_private` / `px_rate_shared` according to `px_mixing`.

### 3.4 RMSE in `reconstruction_error` compares simplex `px_scale` to count proportions

- **Finding** — Compares the L1-normalised mixture `px_scale` against
  `x_raw / library`. Both are on the simplex, so the metric is
  scale-equivariant in library size but is dominated by the dropout / zero
  mass; for a typical scRNA-seq cell with 90 % zeros the RMSE is
  saturated by zero-cell-zero-prediction agreement. Comparing two models
  with this RMSE has poor discrimination.
- **Location** — `src/spVIPESmulti/metrics.py:485-490`.

### 3.5 `latent_dimension_stats` "is_vanished" threshold is hard-coded at 0.5

- **Finding** — The 0.5 threshold has no theoretical basis (KL collapse is
  better diagnosed via per-dim KL or per-dim posterior-prior std ratio).
  Encoder logvar is **clamped to [-4, 4]** at
  `src/spVIPESmulti/nn/networks.py:154-158`, so post-softmax `theta` cannot
  collapse to a δ even when the underlying Normal would.
- **Location** — `src/spVIPESmulti/metrics.py:382-431`.

### 3.6 BatchNorm + small final batches at inference

- **Finding** — `get_latent_representation` defaults to `drop_last=False`
  (`spvipesmulti.py:368-370`) and runs BatchNorm in eval mode using running
  stats. Combined with `_split_tensors_by_group` filtering, a group with a
  single residual cell goes through BatchNorm-eval (fine) but through
  encoder.lvar_encoder's BatchNorm with `BatchNorm1d` running stats — fine
  in eval, but the 1-d clamp `clamp(-4, 4)` then dominates. Acceptable but
  worth noting.
- **Location** — `src/spVIPESmulti/nn/networks.py:99-108, 154-158`;
  `src/spVIPESmulti/model/spvipesmulti.py:368-378, 1407-1413`.

### 3.7 `_label_based_poe` "single-group" branch silently uses dummy prior

- **Finding** — When only one group has any cells of label `ℓ`, the PoE
  for that label is computed against an explicit dummy prior with
  `logvar=30`, then assigned to that group. Cells of label `ℓ` in *other*
  groups receive empty tensors, which are later overwritten by zeros in
  the per-cell scatter at `spVIPESmultimodule.py:868-879`. Whether those
  zero-filled rows leak into downstream loss depends on label coverage in
  the actual minibatch, not the dataset — so behaviour is order-dependent.
- **Location** — `src/spVIPESmulti/module/spVIPESmultimodule.py:826-867`.
- **What would resolve** — A test that flips a label minibatch composition
  between two seeds and verifies bit-identical posteriors for cells whose
  membership did not change.

### 3.8 `differential_abundance` warning logic is inverted

- **Finding** — Warns *only* when alignment-inducing terms are off, but the
  DA score is **degenerate when alignment is on** (μ_a → μ_b). Users
  who follow the warning by enabling disentanglement will worsen, not
  improve, the score's discrimination.
- **Location** — `src/spVIPESmulti/model/spvipesmulti.py:786-797`.

### 3.9 Class-weighted CE with `sum`-then-mean rescaling

- **Finding** — Per-class inverse-frequency weights normalised to
  `sum = n_labels` (`spvipesmulti.py:174-185`) interact with
  `F.cross_entropy(..., weight=...)` which already divides by
  `sum(weights * 1[batch])`. The further per-pair `loss_val * (n_groups /
  n_pairs)` rescale at `spVIPESmultimodule.py:1145-1149` then re-normalises
  in modality count. Net effect on the relative magnitude of components 2
  vs 4 vs 5 is non-obvious and not documented; small classes can become
  dominant or under-weighted depending on minibatch composition.
- **Location** — `src/spVIPESmulti/model/spvipesmulti.py:174-185`,
  `src/spVIPESmulti/module/spVIPESmultimodule.py:1110-1166`.

### 3.10 `prepare_adatas` `groups_obs_indices` uses `pd.Series.values` order

- **Finding** — Index assembly on `multigroups_adata.obs["groups"].values
  == k` (`prepare_adatas.py:135-141`) is correct only as long as
  `ad.concat(..., join="outer", label="groups")` preserves the iteration
  order of the dict. This is the case for AnnData ≥ 0.10 with Python's
  insertion-ordered dict, but no test pins this contract; an upstream
  AnnData behaviour change would silently re-order indices and break the
  per-group slicing in `inference`.
- **Location** — `src/spVIPESmulti/data/prepare_adatas.py:130-150`.

---

## 4. Unclear Assumptions Requiring Human Domain Review

1. **Is the multimodal Gaussian likelihood path intended to model
   log-normalised protein/ADT data, or only data already rescaled to
   [0, 1]?** If the former, §2.3 is a defect; if the latter, the docstring
   ("for log-normalized data") is wrong.
2. **Is the unsupervised PoE strategy supposed to provide a meaningful
   posterior per cell, or only a "regulariser that pulls cells together"?**
   If the latter, the per-cell shared latent should not be exposed via
   `get_latent_representation`/`embed`; if the former, §2.2 needs
   resolution (e.g. mean-field aggregation per group rather than row-wise
   PoE).
3. **What null distribution would `differential_abundance` quote** in a
   manuscript? Permutation across groups? Bootstrap over samples? Without
   this choice it is unclear whether the function is a prototype helper or
   intended for inference.
4. **Should `embed()` return posterior means or rsamples by default?**
   Standard scvi-tools convention (`get_latent_representation(give_mean=True)`)
   is means; the current implementation deviates only in the
   un-normalized path (§2.5).
5. **What is the intended interpretation of the `reconstruction_error`
   columns** ("rmse", "poisson_nll") relative to the actual NB-mixture
   training objective? They quantify a *different* model than the one
   being trained.
6. **Is `traversal.traverse_latent` interpretable when the decoder is the
   default low-rank mixer (rank 4)?** The mixer non-linearly couples
   shared and private latents (`mix_up(ReLU(mix_down(...)))`), so a per-
   dimension ±3σ traversal sees only a slice of the surface. Should the
   gene-effect score average over multiple `z_private` draws rather than
   `z_private = 0`?
7. **Is the disentanglement objective expected to produce a calibrated
   adversarial equilibrium with `disentangle_label_private_weight = 0.05`
   (full preset)?** That weight was tuned to avoid silhouette collapse on
   one dataset (Atypical 76 % CRXV); it is unclear whether this transfers
   to other label distributions.

---

## 5. Verification Plan

For each item in §2–§4, a concrete read-only or simulation-based check
that would confirm or reject the concern. None of these should require
modifying production source.

### V2.1 — NB-on-log1p

- Synthetic NB(μ, θ) draws of shape (n_cells, n_genes); fit a one-group
  spVIPESmulti with `log_variational_generative=True` and again with
  `False`. Compare:
  - `−NB.log_prob(x)` on held-out cells in each model.
  - Recovery of the simulated `μ` via `px_rate_shared`.
- Expected (if §2.1 is correct): `log_variational_generative=False`
  recovers `μ` to within MC noise; `True` underestimates μ for high-count
  genes by a factor of `log(1+x)/x`.

### V2.2 — Row-wise PoE across groups

- With a fixed dataset, call `inference()` twice with two different
  `ConcatDataLoader` seeds. For cells whose group, label, and minibatch
  presence are identical across the two runs, assert
  `poe_stats[g]["logtheta_loc"]` differs by more than encoder noise.
  A truly per-cell posterior should be invariant to the row index of
  *other groups'* cells.
- Quantify: median absolute change in `logtheta_loc` per cell across
  two seeds, vs. the analogous change for `private_stats[g]["log_z"]`
  (which has no cross-group coupling).

### V2.3 — Gaussian on log-normalised protein

- Simulate ADT-like targets `y ~ Normal(μ, 0.5)` with `μ ∈ [-3, 3]`.
  Fit single-group multimodal model with one Gaussian modality. Inspect
  `px_scale.min()`, `px_scale.max()` after training and compare to
  `y.min()`, `y.max()`.
- Expected: `px_scale` range stuck inside `[0, 1]`; per-feature
  `log_scale_gaussian` saturates at large positive values.

### V2.4 — DA calibration

- Two-group dataset with **identical** per-sample composition (no true
  abundance shift). Compute `differential_abundance` and inspect the
  distribution of `da_score`. Under the null it should be symmetric around
  zero with a finite, calibrated quantile against the same statistic
  computed on permuted group labels.
- Property test: shuffle group labels among samples 200 times, recompute
  the score per cell, and report the empirical permutation p-value
  distribution. With no real signal, the histogram should be uniform on
  [0, 1].

### V2.5 — `give_mean` ignored when `normalized=False`

- Call `model.get_latent_representation(give_mean=True, normalized=False)`
  twice with the same data. Compare the returned `shared_reordered`
  arrays element-wise. They should be identical (mean is deterministic);
  current implementation will differ at the rsample noise scale
  (≈0.6 in std after `clamp(-4,4)` initialisation).

### V2.6 — Per-group silhouette is constant

- For any model with two private latents, call
  `integration_report(..., z_private_dict=...)` and assert that the
  `silhouette` column has more than one distinct value across the
  per-group rows. Current implementation will fail this assertion.

### V2.7 — kBET / LISI vs reference

- Run `harmonypy.compute_lisi(...)` (Korsunsky's reference impl) and
  `kBET` from `scIB` on the same `(rep, groups)` and compare to the
  values produced here. Property check: monotone correlation should be
  high but absolute values will differ; demand at least monotone agreement
  on a 5-point grid of synthetic mixing levels.

### V3.1 — Jeffreys mean-pool vs per-cell

- Two batches drawn from `Normal(0, 1)` and `Normal(δ, 1)` with the same
  per-cell σ. Compare implemented loss to the analytic symmetric-KL
  between the two posteriors averaged per cell. Expect implemented loss
  to under-report by a factor proportional to `var(mu_per_cell)/var(noise)`.

### V3.2 — NF KL variance

- Hold encoder fixed; estimate `KL(q‖p_flow)` with `n_samples ∈ {1, 8,
  64}` and compute the standard error of the per-cell KL. Report
  variance reduction and decide whether 1 sample suffices.

### V3.3/V3.4 — reconstruction_error semantics

- Inject scale by multiplying observed counts by 10×; assert RMSE is
  invariant (it should be, by simplex normalisation) but Poisson NLL is
  not (it should scale with counts). If Poisson NLL is invariant under
  10× scaling, the bug is a normalisation in the metric implementation.

### V3.7 — Label-PoE order dependence

- Build a test minibatch where label `ℓ` exists only in group 0. Permute
  the cell order in group 0 within the batch and assert bit-identical
  shared posteriors for all cells of label `ℓ`. Failure indicates
  row-order dependence.

### V4.1 — Gaussian likelihood intent

- Domain question only; no execution. Decide whether to scale Gaussian
  output to data range (e.g. via an unconstrained linear head) or
  document the requirement that protein data be pre-rescaled to `[0, 1]`.

### V4.6 — Traversal interpretability under low-rank mixer

- Compare per-gene effect rankings produced by `traverse_latent` with
  `use_low_rank_mixer=True` vs `False` on the same trained model.
  Spearman correlation of top-50 gene rankings per dimension; if low,
  the traversal is mixer-architecture-dependent and the docstring should
  warn.

---

## Verification Standard — Trace Blocks

### Trace A — single-modal training loss path

`model.train()` →
`MultiGroupTrainingMixin.train()` → Lightning `Trainer.fit()` →
`spVIPESmultimodule.forward()` →
`_get_inference_input` (`spVIPESmultimodule.py:481-501`) →
`inference()` (`spVIPESmultimodule.py:524-571`) →
per-group `Encoder.forward` (`networks.py:111-167`) [returns Normal posterior with `logvar.clamp(-4,4)`] →
`_supervised_poe` (`spVIPESmultimodule.py:583-600`) → `_label_based_poe`
(`spVIPESmultimodule.py:739-893`) **or** `_poe_n` (`spVIPESmultimodule.py:302-440`) →
`generative()` (`spVIPESmultimodule.py:894-925`) →
`LinearDecoderSPVIPE.forward` (`networks.py:329-368`) →
`NegativeBinomialMixture(mu1=px_rate_priv, mu2=px_rate_shared, theta1=px_r,
mixture_logits=px_mixing)` →
`loss()` (`spVIPESmultimodule.py:1241-1334`) where target is
`log1p(x_obs)` if `log_variational_generative` (default True) ⇒ §2.1.

### Trace B — `differential_abundance`

`spVIPESmulti.differential_abundance()` (`spvipesmulti.py:706-867`) →
`get_shared_posterior` (`:474-505`) → `get_latent_representation`
(`:336-413`) [§2.5: `loc` is mean, but `shared_reordered` is rsample] →
`_aggregate_shared_posterior` (`:651-705`) [aggregates per (group, sample)
mean of means; per-sample posterior scale is `sqrt(mean(scale²))`] →
`differential_abundance` body computes
`d = sum(((z − μ_g)/σ_g)²)` and returns `d_a − d_b` per cell with no
null distribution (§2.4).

### Trace C — integration_report ↔ evaluate

`spVIPESmulti.evaluate()` (`spvipesmulti.py:1198-1372`) →
`get_latent_representation` (rsample if `normalized=False`, §2.5) →
`spVIPESmulti.metrics.integration_report` (`metrics.py:230-316`) →
`ilisi/clisi/kbet/knn_purity/leiden_ari/per_group_silhouette` —
silhouette is global but written per row (§2.6); kBET/LISI are
non-published variants (§2.7).

### Claims that rest on documentation only (not verified)

- The `disentangle_warmup` flag scales the disentangle aggregate by
  `kl_weight`; whether the schedule used by the trainer mixin is the
  scvi-tools KL annealing schedule was not traced into
  `MultiGroupTrainingMixin` or Lightning callbacks for this audit.
- The `use_jeffreys_integ` term is described as "multigrate's alignment
  strategy"; the comparison to multigrate's actual loss was not made
  against external source.

---

## Closing Note

This audit identifies issues whose impact ranges from **invalidates
quantitative comparisons** (§2.1, §2.2, §2.3, §2.4, §2.5, §2.6, §2.7)
down to **fragile or imprecise diagnostics** (§3). None of the items
require an architectural redesign; most can be addressed by either
(a) fixing the target/parameter pairing in a likelihood call or
(b) renaming a metric and adding the missing null/correction
machinery. No code changes were made as part of this pass, per audit
rules.
