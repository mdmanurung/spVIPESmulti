# Architecture comparison: spVIPESmulti vs. original spVIPES

Date: 2026-05-09
Scope: deep variational autoencoder architecture only (encoder, decoder, prior,
generative likelihood, inference graph, PoE construction, loss). Training-loop /
data-loader differences are excluded except where they change the math.

Sources
- Reference (original): `audits/spVIPES/src/spVIPES/{module/spVIPESmodule.py, nn/networks.py, model/spvipes.py}`
- Current (multi): `src/spVIPESmulti/{module/spVIPESmultimodule.py, nn/networks.py, model/spvipesmulti.py, model/_disentangle_presets.py}`

---

## 1. High-level summary

Both packages share the same backbone idea: each group has its own
shared / private encoder pair and a `LinearDecoderSPVIPE` that fuses
`(z_private, z_shared)` into a `NegativeBinomialMixture` likelihood, with the
shared posteriors of all groups merged via Gaussian Product-of-Experts (PoE)
prior to decoding.

`spVIPESmulti` keeps that backbone but diverges in five substantive ways:

1. **Posterior parameterisation** — `LayerNorm` + `logvar` clamp on
   `mu_encoder/lvar_encoder` (vs. `BatchNorm1d`, no clamp).
2. **Decoder mixture math** — true convex combination of private + shared NB
   rates, plus a *correct* slicing of the concatenated latent vector. The
   original silently dropped the private contribution and crossed the
   `(z_private, z_shared)` slices in `generative()`.
3. **PoE generalised to N ≥ 2 groups** with vectorised label-based fusion
   (original was hard-coded to N = 2 with a per-cell Python loop). The
   transport-plan and paired-data PoE branches are removed.
4. **Prior** — optional normalising-flow prior (`zuko` NSF / MAF) over either
   shared, private, or both latents, with a Monte-Carlo KL surrogate.
   Original uses a fixed `N(0, I)` prior throughout.
5. **Auxiliary objectives** — a CellDISECT-style four-classifier
   disentanglement block (two GRL adversarial + two supervised CE) plus
   prototype InfoNCE on the shared latent, plus an optional pairwise
   Jeffreys (symmetric KL) integration loss across PoE posteriors. Original
   has none of these; its loss is reconstruction + 4 fixed-prior KL terms.

Two further differences are bug-for-correctness rather than design choices:
the original mixes *only* the shared NB rate (private contribution is
multiplied by zero) and decodes from a mis-sliced latent vector; both are
fixed in `spVIPESmulti`.

A multimodal extension (per-(group, modality) encoders/decoders + intra-group
PoE → inter-group PoE) is also introduced; this is a structural extension of
the same VAE rather than a re-architecture.

---

## 2. Encoder

| Aspect | Original `spVIPES.nn.Encoder` | `spVIPESmulti.nn.Encoder` |
|---|---|---|
| Hidden layers | `Linear(n_in+cat, h) → ReLU → Linear(h, h) → ReLU → Dropout` | Same backbone, but activation is selectable (`silu` default, `relu`, `leakyrelu`). |
| Posterior heads | `Linear(h, k) → BatchNorm1d(k)` for both `mu` and `logvar` | `Linear(h, k) → LayerNorm(k)` for both. |
| `logvar` range | Unbounded | Hard-clamped to `[-4, 4]` after the linear+norm. |
| One-hot covariate gate | `n_input + cat_dim * True` (a no-op `True` multiplier) | `n_input + cat_dim` (the `True` was vestigial). |
| Output dict | `{loc, logvar, scale, log_z, theta, qz}` | identical |

### Theoretical implications

**BatchNorm → LayerNorm on the posterior heads.**
`BatchNorm1d` keeps separate running statistics for train and eval mode.
At inference, the mean/variance of the encoded latent depend on the
*minibatch composition* during training and on whichever running mean/var
happened to be accumulated last — so `model.encode(x)` is no longer a pure
function of `x`. In the original code this manifests as `get_latent_representation`
producing different `mu` for the same cell depending on how many cells are
in the encode batch. `LayerNorm` normalises *per sample over the latent
dimension*, so train- and eval-mode forward passes are identical and the
encoder becomes deterministic in `mu`. The cost is that batch-level shift
in the activations is no longer absorbed by the head; in practice this
matters little because the preceding `fc1/fc2` already saw the batch
distribution. The `spVIPESmulti` codebase explicitly notes that this change
breaks checkpoint compatibility (the parameter names of `BatchNorm1d` —
`weight, bias, running_mean, running_var, num_batches_tracked` — differ from
those of `LayerNorm` — `weight, bias`).

**`logvar.clamp(-4, 4)` (i.e. σ ∈ [≈0.135, ≈7.39]).**
Two failure modes are prevented:
- **Posterior collapse / over-confidence.** `logvar < -4` would let the
  encoder push `KL ≈ ½(μ² + σ² − 1 − log σ²)` to be very large in absolute
  value or, in the other direction, make `qz` nearly a δ-function so the
  GRL gradients on `z` become unbounded.
- **Variance explosion under adversarial GRL pressure.** When a gradient
  reversal layer is wrapped around `z` to fool a label classifier, the
  cheapest way for the encoder to confuse the classifier is to inflate σ:
  the noisier `z` is, the lower the classifier's accuracy, the smaller the
  reversed loss. Without a clamp this is a positive-feedback loop that
  diverges. The original spVIPES has no GRL, so it does not need the clamp,
  which is why this is a fix specific to the new disentanglement objectives.

**SiLU as default activation.**
SiLU (`x · σ(x)`, also known as Swish) is smooth, non-monotonic for
`x < 0`, and has a non-zero gradient for moderately negative inputs. In
deep VAEs it tends to give faster early-epoch convergence than ReLU and
slightly better calibrated posteriors (less dead-unit propagation). For a
two-layer encoder this is a small, mostly noise-level improvement, but it
costs nothing.

---

## 3. Decoder

Both packages use `LinearDecoderSPVIPE` with two parallel one-layer factor
regressors (`factor_regressor_private`, `factor_regressor_shared`) that
produce normalised `px_scale_*`, then a small "mixer" subnetwork that
produces logits over genes for a `NegativeBinomialMixture`.

### 3.1 Mixer head

| Aspect | Original | `spVIPESmulti` |
|---|---|---|
| Always present | `sigmoid_decoder` (FC, n_in → 256, batch-norm, ReLU) followed by `mixture` (FC, 256+n_in → n_genes) | Same path by default. |
| Optional alternative | — | `use_low_rank_mixer=True`: replace the two FC blocks with `Linear(n_in, r) → ReLU → Linear(r, n_genes)`, default `r=4`. |

**Implication.** The default mixer is a ~`(n_in × 256) + (256+n_in) × G`
parameter bag — for typical `n_in = 35`, `G = 2000` it is ≈540 K parameters
*per group*. The low-rank variant collapses this to `(n_in + G) × r` ≈ 8 K
parameters per group at `r = 4`. Documentation (the docstring) records that
the rank-4 ablation costs ≈4% reconstruction quality — a classic
parameter-vs-fit trade-off. Mathematically the low-rank mixer constrains
the gene-wise mixing logits to live in a rank-`r` subspace of gene-space;
this is a strong inductive bias that genes share a small number of
"mixing programs" (e.g. an axis from "shared-only" to "private-only"
genes). In a low-rank regime overfitting on small datasets is reduced at
the cost of underfitting fine-grained per-gene mixing.

### 3.2 Mixing combination (the silent bug fix)

Original `forward` (final two lines before return):
```python
mixing = 1 / (1 + torch.exp(-px_mixing))
px_scale = torch.nn.functional.normalize((1 - mixing) * px_rate_shared, p=1, dim=-1)
```

`spVIPESmulti.LinearDecoderSPVIPE.forward`:
```python
mixing = torch.sigmoid(px_mixing)
px_scale = torch.nn.functional.normalize(
    mixing * px_rate_private + (1 - mixing) * px_rate_shared, p=1, dim=-1
)
```

Note that `px_scale` is the *fused* normalised rate. It is **not** what is
actually fed to `NegativeBinomialMixture` (which receives `px_rate_private`
and `px_rate_shared` separately as `mu1` / `mu2` together with
`mixture_logits=px_mixing`), so this expression only matters when the
caller asks for the merged scale (e.g. `get_normalized_expression`). Still:

- The original computes `(1 − mixing) · px_rate_shared` only — i.e. it
  literally drops the private contribution from the merged scale.
- It is then `L1`-normalised, which puts it back on a simplex but with a
  silently *re-weighted* shape because the missing `mixing · px_rate_private`
  term means genes whose private signal is large receive 0 mass.

Downstream this means the original's "fused expression" output is biased
toward shared-only signal; in the new code it is the proper convex mixture
that the `NegativeBinomialMixture` likelihood is supposed to represent.

### 3.3 Latent slicing in `generative()`

Both modules concatenate as `cat((private, poe_shared), dim=-1)` so the
layout is `[private (P=10), shared (S=25)]`.

Original:
```python
stats["log_z"][:, self.n_dimensions_shared : self.n_dimensions_private + self.n_dimensions_shared]  # → "private"
stats["log_z"][:, : self.n_dimensions_shared]                                                       # → "shared"
```
With `S=25, P=10` this passes columns `[25:35]` as `z_private` and `[0:25]`
as `z_shared`. But the actual layout puts private at `[0:10]` and shared
at `[10:35]`. So:
- `z_private` argument receives the **last 10 dims of the shared PoE
  latent** (a slice from the middle of the shared block).
- `z_shared` argument receives the **10 private dims followed by the first
  15 shared dims**.

The decoder's `factor_regressor_private` and `factor_regressor_shared` are
linear, so they will still produce a valid `px_scale`, but the meaning of
the loadings is scrambled: `factor_regressor_private` is conditioned on a
slice of the shared latent, and vice-versa. In particular,
`get_loadings(..., type_latent="private")` in the original returns a
weight matrix whose columns are *not* the gradients of the reconstruction
with respect to `z_private` — they are gradients with respect to the
shared-PoE block.

`spVIPESmulti` slices correctly:
```python
combined_log_z[:, : self.n_dimensions_private]   # private
combined_log_z[:, self.n_dimensions_private :]   # shared
```

### 3.4 `get_loadings` correctness

Both implementations divide out the BatchNorm statistics (`gamma / sqrt(var + eps)`)
to recover an interpretable weight matrix. This is fine in the original
because the decoder's `factor_regressor_*` use BatchNorm at the head. The
*encoder* change to LayerNorm in `spVIPESmulti` does not affect
`get_loadings` because `get_loadings` only inspects decoder weights.

---

## 4. Product of Experts (PoE)

### 4.1 Routing

| Strategy | Original | `spVIPESmulti` |
|---|---|---|
| Label-based (supervised) | Yes (N = 2 only) | Yes, N ≥ 2; only supported strategy. |
| Optimal transport (cluster-based) | Yes | Removed (raises `NotImplementedError`). |
| Paired data | Yes | Removed (raises `NotImplementedError`). |
| Unsupervised fallback | Implicit (encoder PoE only) | Removed at the model level (`label_key` is required). |

### 4.2 Mathematical core

Both implement the standard Gaussian PoE:
$$
\sigma_{\text{joint}}^{-2} = 1 + \sum_g \sigma_g^{-2}, \quad
\mu_{\text{joint}} = \sigma_{\text{joint}}^{2} \sum_g \mu_g \sigma_g^{-2}.
$$
The leading `1` corresponds to the contribution of an `N(0, I)` prior
expert. This is identical in both packages (`_product_of_experts`).

### 4.3 N-group generalisation

The original `_poe2` is hard-coded to two groups, with explicit zero/one
padding when the two minibatches have unequal cell counts (handled by
manually reshaping `inverse_vars` and `mus_vars` for groups 1 and 2). The
label-based version then loops over `common_labels`, calls `_poe2`
per-label, and re-scatters per-cell into the output tensor with a
`label_count` dictionary tracking insertion order — an O(N\_cells) Python
loop with `.item()` calls that triggers GPU↔CPU syncs every iteration.

`spVIPESmulti._poe_n` accepts an arbitrary `dict[group, stats]` and:

- Pads each group's `(loc, logvar)` to `max_batch_size` with neutral
  values `(0, 0)` (i.e. precision 1, contributing only the prior expert),
  stacks along a new axis, and applies the standard PoE in a single
  vectorised reduction.
- Slices the joint result back to each group's true row count.

`spVIPESmulti._label_based_poe` then:

- Iterates over the union of labels across all groups (not just the
  intersection).
- For groups that are missing a given label, contributes
  `logvar = 30` (precision ≈ `e^{-30}`) instead of the original's
  `logvar = 0` (precision = 1). This avoids the **double-prior bug**: the
  original would add an `N(0, I)` "missing-group" expert *on top of* the
  PoE's built-in `+1` prior term, doubly shrinking the posterior toward
  the origin for any label held by only one group.
- Reassembles the per-cell output with **vectorised boolean-mask scatters**
  rather than per-cell `.item()` indexing, eliminating the per-cell GPU
  sync.

#### Theoretical implications

- **Statistical correctness for one-of-a-kind labels.** With the original's
  `logvar=0` padding, a cell whose label exists in only one group gets a
  shared posterior whose precision is `1 + ⅔ + 1 = 2.67` (own + missing's
  flat-but-informative + global), pulling `μ_shared` halfway toward zero
  with no theoretical justification. The new code makes the missing
  group's contribution numerically negligible, so the posterior reduces
  to the encoder's own posterior plus the global prior — exactly the
  behaviour you would expect.
- **Compositionality.** With `_poe_n` accepting an arbitrary group dict,
  the same routine is reused for the multimodal *intra-group* PoE (across
  modalities) and *inter-group* PoE (across groups), guaranteeing
  identical math at both levels.

### 4.4 What was lost

- **Optimal-transport PoE** (`_cluster_based_poe`, `_paired_poe`,
  `_get_batch_transport_plans`) is removed. Theoretically these were the
  most appealing branches because they didn't need ground-truth labels:
  they used a precomputed OT plan to construct soft per-cluster matchings
  between groups before applying PoE. The original's implementation was
  expensive (full transport plan in memory, per-batch slicing) and never
  empirically validated to outperform the label-based path; the
  `spVIPESmulti` audit explicitly notes that the unsupervised PoE
  produced no integration signal beyond what the disentanglement losses
  themselves provided. The trade-off, however, is that `spVIPESmulti`
  *requires* a label key — which limits its applicability to cohorts
  without ground-truth annotations or trustworthy reference mapping.

---

## 5. Prior

| | Original | `spVIPESmulti` |
|---|---|---|
| Prior on `z_private` | `N(0, I)` (fixed) | `N(0, I)` or NSF / MAF flow |
| Prior on `z_shared (post-PoE)` | `N(0, I)` (fixed) | `N(0, I)` or NSF / MAF flow |
| KL to flow | — | Monte-Carlo KL `E_q[log q(z) − log p_flow(z)]` with optional `n_mc_samples > 1` |

### Theoretical implications

The standard normal prior makes the aggregate posterior
`q(z) = E_x[q(z|x)]` mismatched to a Gaussian if the latent has cluster
structure; this is the well-known "prior holes" problem
(Hoffman & Johnson 2016; Tomczak & Welling 2018, *VampPrior*). When
single-cell data has tens of cell types, an expressive prior — here a
neural spline flow (Durkan et al., 2019) or masked autoregressive flow
(Papamakarios et al., 2017) — gives a better fit and reduces *KL
underutilisation* (the encoder being penalised for putting clusters in
distinct regions because such an arrangement has high KL under the
unimodal prior). The cost is:

- **Optimisation difficulty.** Flows are trained jointly with the VAE;
  in early epochs the flow is not yet a good fit, and the MC-KL surrogate
  can have high variance — hence the `n_mc_samples > 1` option, which is a
  standard variance-reduction technique.
- **Loss of an analytic KL term.** The closed-form Gaussian KL is replaced
  by `log q(z|x) − log p_flow(z)`. This is an unbiased estimator but the
  variance increases the gradient noise compared to the analytic case.

The flow is unconditional (`context=0`), so it represents a single
aggregate prior, not a mixture-of-priors per group / per label. That is a
deliberate simplification: a per-label flow prior would essentially
duplicate the role of the disentanglement objective.

---

## 6. Loss

### 6.1 Original `loss()`

```
total = mean(
    recon_g1 + recon_g2
  + kl_w · ( kl_priv_g1 + kl_priv_g2 + kl_poe_g1 + kl_poe_g2 )
)
```
i.e. four KL terms (two private + two PoE), all against `N(0, I)`, plus
two NB-mixture reconstruction terms. Hard-coded to two groups.

### 6.2 `spVIPESmulti.loss()`

```
total = Σ_g w_g · ( recon_g + kl_w · kl_priv_g + kl_w · kl_poe_g )
       + (1 / n_groups) · disentangle_total
       + jeffreys_integ_weight · jeffreys_loss          [optional]
```
where each `kl_*` term is either an analytic Gaussian KL or a flow-based
MC-KL depending on `use_nf_prior` and `nf_target`, and `w_g` is either
`1/n_groups` or a user-specified vector of group weights normalised to
sum to one.

The multimodal variant `_loss_multimodal` further sums the reconstruction
and per-modality private KL across modalities within each group and
divides the per-modality private KL by `n_modalities` so the *shared*
PoE KL is not penalised once per modality.

### 6.3 Bug-fix: counts vs. log-counts as the NB target

The original (and its `log_variational_generative` flag) effectively
applies `log(1 + x)` before passing the data into the NB-mixture
likelihood:
```python
if self.log_variational_generative:
    x = {i: torch.log(1 + xs) for i, xs in x.items()}
...
reconstruction_loss_groups_1_poe = -generative_outputs["private_poe"]["0"]["px"].log_prob(x[0]).sum(-1)
```
But `NegativeBinomialMixture.log_prob` expects non-negative integer counts
on the count scale; passing `log(1+x)` evaluates the NB log-PMF on
non-integer values close to zero (`log_prob` in scvi-tools accepts the
fractional input but the resulting "loss" is no longer a count likelihood
— it's the NB log-density evaluated at a transformed argument, which has
no statistical meaning here). In practice this still trains because the
gradient still drives `μ` toward `x`, but the loss magnitude, the
calibration of `θ` (dispersion), and the per-cell normalisation are all
distorted.

`spVIPESmulti` always passes the raw counts as the NB target (the
inference-time `log1p` is kept for the encoder input only). For Gaussian
modalities the input is assumed to already be log-normalised, which is
the convention everywhere else in scvi-tools.

### 6.4 Auxiliary losses

`spVIPESmulti` adds three families:

1. **Adversarial / supervised classifier block** (`_compute_disentangle_losses`)
   - GRL on `z_shared` against group identity (DANN-style domain adversary).
   - GRL on `z_private` against label identity.
   - Supervised CE on `z_shared` against label.
   - Supervised CE on `z_private` against group identity.
2. **Prototype InfoNCE** (Khosla et al., 2020; SupCon variant) on `z_shared`,
   with EMA-updated per-(group, label) prototypes used as both positives
   (own-group prototype) and negatives (other-group prototypes for other
   labels).
3. **Pairwise Jeffreys (symmetric KL) integration loss** between every pair
   of groups' PoE posteriors — a moment-matching alignment regulariser
   inspired by `multigrate`.

These are mutually orthogonal in design (each can be ablated by setting
its weight to zero) and have well-understood theoretical roles:

- **GRL terms** are the classic Ganin & Lempitsky (2015) domain-adversarial
  loss; minimising the *negative* CE through the GRL at the encoder is
  equivalent (under the optimal classifier assumption) to minimising
  `H(Y | Z)` — i.e. maximising mutual information between `Z` and `Y` —
  which here is what we *don't* want for the targeted axis, hence the
  reversal: the encoder is pushed toward `I(Z; Y) ≈ 0` for that axis.
- **Supervised CE terms** are variational lower bounds on `I(Z; Y)`
  (Barber & Agakov, 2003), so adding them as positive terms to be
  minimised maximises that mutual information.
- **InfoNCE** is a tighter (in low-bias regime) MI lower bound than CE
  (Oord et al., 2018; Poole et al., 2019); using EMA prototypes
  dramatically reduces memory vs. an all-pairs SupCon loss and gives
  semantic stability across batches.
- **Jeffreys** between PoE posteriors is a symmetric distributional
  alignment: `KL(p‖q) + KL(q‖p)` is a finite, symmetric divergence that
  weighs both mode-covering and mode-seeking errors. Because PoE is
  already supervised, this term plays the role of a *fine-grained*
  alignment regulariser — useful when label boundaries are noisy.

The `kl_weight` warmup is applied **only** to the GRL components (not to
the supervised CE) — explicitly noted in code (`W-055`). This is the
correct choice: warm-starting an adversary against a high-variance
posterior diverges; warm-starting a classifier on noisy `z` simply
produces a noisy gradient.

---

## 7. Multimodal extension (no analogue in original)

The original is single-modality (count matrix per group). `spVIPESmulti`
adds a strict generalisation:

- Per `(group, modality)` shared and private encoders.
- Two-level PoE: intra-group across modalities (combining shared
  posteriors of e.g. RNA + ADT for the same cell) and then inter-group
  across groups.
- Per-modality decoders with selectable likelihood (`nb` for counts,
  `gaussian` for already-log-normalised continuous data).
- Per-modality weighting in the loss.

Theoretically the intra-group PoE is the standard MVAE / `totalVI`
construction (Wu et al., MVAE 2018; Gayoso et al., totalVI 2021):
treating each modality's encoder as an "expert" and combining via PoE
gives the optimal Gaussian approximation to the joint posterior assuming
modality-wise conditional independence given `z_shared`. Stacking that
with an inter-group PoE for cross-cohort integration is the natural
extension and is internally consistent because both levels use the same
`_poe_n` routine.

Note that the *private* latent is still per-(group, modality) and is
**not** PoE-fused — only shared latents are integrated. That matches the
shared/private decomposition: by construction, modality-specific
information lives in the modality's own private space.

---

## 8. Side-by-side cheat sheet

| Component | Original | `spVIPESmulti` |
|---|---|---|
| Posterior head norm | `BatchNorm1d` | `LayerNorm` |
| `logvar` clamp | none | `[-4, 4]` |
| Encoder activation | `ReLU` (fixed) | `SiLU` default |
| Decoder mixture combiner | `(1−m)·shared` (private dropped) | `m·private + (1−m)·shared` |
| `generative()` slicing | swapped (private↔shared) | correct |
| NB target | `log(1+x)` (incorrect) | raw `x` |
| Prior | `N(0,I)` only | `N(0,I)` or NSF / MAF flow |
| KL on flow | n/a | MC estimator, ≥1 sample |
| PoE groups supported | 2 (hard-coded) | N ≥ 2 (vectorised) |
| Missing-label expert | `logvar=0` (double-prior bug) | `logvar=30` (≈zero precision) |
| PoE strategies | label / OT / paired | label only |
| Disentanglement | none | 4 classifiers + InfoNCE |
| Cross-group alignment | implicit via PoE | + optional Jeffreys |
| Multimodal | no | yes (intra-group → inter-group PoE) |
| Group balancing | none | `group_loss_weights`, normalised |

---

## 9. Net effect on the model's probabilistic semantics

The original spVIPES, taken literally as written, optimises a quantity
that is *not* the ELBO of the implied generative model:

- The mixture combiner zeroing out the private contribution means the
  effective forward map decoupled `z_private` from `px_scale` (it still
  enters via `mu1 = px_rate_private` in the NB mixture, but with a logits
  vector that was learned against the wrong combination).
- The slicing swap in `generative()` means `factor_regressor_private`
  was actually conditioned on the shared latent and vice versa.
- The `log(1+x)` target on the NB likelihood means the reconstruction
  term is not the NB log-likelihood of the data.

These three issues together imply the original ELBO surrogate does not
correspond to any consistent generative model; training converges
empirically because each individual term still has a sensible gradient
direction, but the resulting latent space is a side effect of an
ill-specified objective.

`spVIPESmulti` restores a consistent ELBO:

$$
\log p(x) \geq \mathbb{E}_{q(z_p, z_s | x)} \big[ \log p(x | z_p, z_s) \big]
- \mathrm{KL}(q(z_p | x) \,\|\, p(z_p))
- \mathrm{KL}(q(z_s | x) \,\|\, p(z_s)),
$$

with `q(z_s | x)` being the PoE of the per-group encoders, `p(z_p)` and
`p(z_s)` either standard normal or a learned flow, and `p(x | z_p, z_s)`
the NB mixture evaluated on raw counts. The auxiliary losses (GRL, CE,
InfoNCE, Jeffreys) augment but do not replace this ELBO.

---

## 10. Recommendations / open questions

1. **Re-introducing an unsupervised PoE** would re-open the original's
   broadest use case. A principled replacement for the removed
   transport-plan path could be a *learned* alignment (e.g. an adversarial
   matcher on `z_shared` plus an SCANVI-style pseudo-label loop), avoiding
   precomputation of an OT plan.
2. **The flow prior's expressivity vs. reliability** is worth a small
   ablation — at low data scale (a few thousand cells per group), the
   flow can overfit and effectively "memorise" the empirical aggregate
   posterior; at that point the KL term goes to zero and the ELBO becomes
   a reconstruction-only objective.
3. **The clamp `logvar ∈ [-4, 4]`** is conservative on the *upper* end
   (σ ≈ 7.4 is large in a 25-D latent). If only the GRL feedback loop is
   the concern, a tighter upper bound (e.g. 2 → σ ≈ 2.7) could be tried.
4. **Per-(group, modality) private** but **per-group shared (post-PoE)**
   is the right factorisation for the multimodal case. If a future
   extension needs *modality-specific shared* (e.g. a "shared-RNA-only"
   axis), that would require an additional private/shared split per
   modality, not just per group.

