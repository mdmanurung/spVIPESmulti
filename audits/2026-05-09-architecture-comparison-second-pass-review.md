# Second-pass review of `2026-05-09-architecture-comparison-spVIPES-vs-spVIPESmulti.md`

Date: 2026-05-09
Method: every claim in the prior report was re-checked against the source files
listed below. Line numbers reference the files in this repo at the time of
writing.

Files audited
- Original: `audits/spVIPES/src/spVIPES/module/spVIPESmodule.py` (ref-mod)
- Original: `audits/spVIPES/src/spVIPES/nn/networks.py` (ref-nn)
- Current: `src/spVIPESmulti/module/spVIPESmultimodule.py` (cur-mod)
- Current: `src/spVIPESmulti/nn/networks.py` (cur-nn)
- Current: `src/spVIPESmulti/model/_disentangle_presets.py` (cur-pre)

Verdict: most architectural claims hold and are now anchored to specific
lines. Three claims were inaccurate and are corrected here. Two claims
were overstated and are softened. New caveats are recorded for the
mixer-combiner and "ELBO inconsistency" claims.

---

## A. Confirmed claims (with line-anchored citations)

### A.1 Encoder posterior heads: `BatchNorm1d` → `LayerNorm`

- ref-nn lines 73–83: `mu_encoder` / `lvar_encoder` use `nn.BatchNorm1d(n_topics)`.
- cur-nn lines 95–105: same heads use `nn.LayerNorm(n_topics)`, with an
  in-source comment explicitly noting the train-vs-eval determinism
  motivation and the loss of checkpoint compatibility.

### A.2 `logvar` clamp `[-4, 4]`

- ref-nn line 124: `logtheta_logvar = self.lvar_encoder(data)` — no clamp.
- cur-nn lines 149–151: `logtheta_logvar = self.lvar_encoder(data).clamp(-4, 4)`,
  with an in-source comment naming the GRL-feedback failure mode.

### A.3 Encoder activation: `ReLU` (fixed) → `SiLU` (default, configurable)

- ref-nn line 67: `self.relu = nn.ReLU()`.
- cur-nn lines 12–16, 67, 82–88: dispatch table `_ACTIVATIONS` with
  `silu` default and `silu / relu / leakyrelu` allowed.

### A.4 Decoder mixer combiner

- ref-nn lines 386–388:
  ```python
  mixing = 1 / (1 + torch.exp(-px_mixing))
  px_scale = torch.nn.functional.normalize((1 - mixing) * px_rate_shared, p=1, dim=-1)
  ```
  Only the shared rate enters the merged `px_scale`.
- cur-nn lines 379–382:
  ```python
  mixing = torch.sigmoid(px_mixing)
  px_scale = torch.nn.functional.normalize(
      mixing * px_rate_private + (1 - mixing) * px_rate_shared, p=1, dim=-1
  )
  ```
  Both rates enter, as a true convex combination.

**Caveat (sharpened from prior report).** This `px_scale` is **not** what
`NegativeBinomialMixture.log_prob` is evaluated on during training — the
likelihood is constructed at ref-mod line 794 / cur-mod line 947 from
`mu1=px_rate_private`, `mu2=px_rate_shared`, `mixture_logits=px_mixing`,
which is the proper mixture and is unaffected by the combiner choice.
So the original's combiner does **not** corrupt the training loss; it
only corrupts user-facing outputs that read `px_scale`
(e.g. `get_normalized_expression`-style returns). The prior report
correctly noted this in the body but should have stated it more
prominently. The combiner is therefore a **bug in the auxiliary output**,
not in the training objective.

### A.5 Slicing in `generative()` (private / shared swap)

- ref-mod lines 745–749 build the layout `[private (P) | poe_shared (S)]`:
  ```python
  groups_1_private_poe_log_z = torch.cat((groups_1_private_log_z, groups_1_poe_log_z), dim=-1)
  ```
- ref-mod lines 757–758 then slice as if the layout were `[shared | private]`:
  ```python
  stats["log_z"][:, self.n_dimensions_shared : self.n_dimensions_private + self.n_dimensions_shared],  # → z_private arg
  stats["log_z"][:, : self.n_dimensions_shared],                                                       # → z_shared arg
  ```
  With `(P, S) = (10, 25)` this passes columns `[25:35]` (the last 10 of
  the shared block) to the decoder's `z_private` argument, and columns
  `[0:25]` (private + first 15 of shared) to `z_shared`.
- cur-mod lines 943–944 are aligned with the `[private | shared]` layout:
  ```python
  combined_log_z[:, : self.n_dimensions_private],
  combined_log_z[:, self.n_dimensions_private :],
  ```

**Caveat (added).** The decoder's `factor_regressor_private` and
`factor_regressor_shared` are linear maps to a simplex; with the
swapped slicing, the model is still a valid density on counts, just
that the parameter named "private" is actually conditioned on (most of)
the shared latent and vice versa. So this is a **mis-labelling /
interpretability bug** with downstream consequences for
`get_loadings`, **not** a violation of the ELBO derivation. The prior
report implied the latter; corrected here.

### A.6 NB target uses `log(1+x)` in original loss

- ref-mod constructor default at line 93: `log_variational_generative: bool = True`.
- ref-mod lines 820–824:
  ```python
  if self.log_variational_generative:
      x = {i: torch.log(1 + xs) for i, xs in x.items()}  # logvariational

  reconstruction_loss_groups_1_poe = -generative_outputs["private_poe"]["0"]["px"].log_prob(x[0]).sum(-1)
  ```
  The NB-mixture log-prob is evaluated on `log(1 + counts)` by default.
- cur-mod lines 1313–1320:
  ```python
  # W-011: always use raw counts as the NB likelihood target.
  # log_variational_generative previously applied log1p here which
  # is incorrect — NegativeBinomialMixture expects non-negative
  # integer counts, not log-transformed values.
  x_target = x_obs
  ...
  recon_loss = -generative_outputs["private_poe"][str(g)]["px"].log_prob(x_target).sum(-1)
  ```

**Caveat (sharpened).** `NegativeBinomialMixture.log_prob` is mathematically
defined for any non-negative real input (uses `lgamma`), so the original
*does* compute a valid log-density. The point is that the density is for
the *transformed* observation `log(1+x)`, which is not the data-generating
distribution we want to model. This is best described as
"optimises a valid ELBO of a model that doesn't match the data" rather
than "doesn't optimise an ELBO at all".

### A.7 Original PoE is hard-coded to N=2

- ref-mod lines 766–769: explicit guard
  ```python
  if (len(private_stats.items()) > 2) or (len(shared_stats.items()) > 2):
      raise ValueError(...)
  ```
- ref-mod line 287 (`_poe2`): `if len(shared_stats.keys()) > 2: raise ValueError(...)`.
- ref-mod loss block lines 823–890 only references `groups_1_*` and `groups_2_*`.
- cur-mod `_poe_n` lines 380–445 accepts an arbitrary group dict and
  vectorises padding-then-stack-then-PoE.

### A.8 Removed branches: `_cluster_based_poe`, `_paired_poe`

- cur-mod lines 470–471: `_cluster_based_poe` raises
  `NotImplementedError("Cluster-based PoE has been removed. Use label-based PoE (label_key=...) instead.")`.
- cur-mod (around the paired branch): `_paired_poe` similarly raises.
  (Searchable from cur-mod via the string "Paired PoE has been removed".)
- ref-mod lines 178–264 contain the original `_cluster_based_poe`;
  ref-mod lines 444–500 contain `_paired_poe`; ref-mod lines 412–428
  contain the routing logic that selected between OT, paired and label
  branches.

### A.9 Per-cell Python loop with `.item()` in original PoE reassembly

- ref-mod lines 779–787 (in `_label_based_poe`):
  ```python
  for i, label in enumerate(groups_1_labels):
      count = label_count.get(label.item(), 0)
      ...
      tensor_index = count % poe_stats[label.item()][0]["logtheta_loc"].size(0)
      groups_1_output["logtheta_loc"][i] = poe_stats[label.item()][0]["logtheta_loc"][tensor_index, :]
  ```
- cur-mod lines 894–908: vectorised boolean-mask scatter:
  ```python
  for label in all_labels:
      mask = labels_g == label
      if not mask.any():
          continue
      poe_g = poe_stats_per_label[label][g]
      for k in stat_keys:
          group_output[k][mask] = poe_g[k]
  ```

### A.10 NF prior block

- cur-mod lines 7 import `zuko.flows`; lines 305–315 construct
  `flow_prior_shared` / `flow_prior_private` based on
  `nf_target ∈ {"shared", "private", "both"}`.
- cur-mod `_nf_kl` lines 754–793: MC estimator with `n_mc_samples` knob.
- ref-mod has no `zuko` import and uses fixed-prior `kl(qz, Normal(0, I))`
  at ref-mod lines 838–870.

### A.11 Disentanglement classifier block + InfoNCE

- cur-mod lines 320–376 instantiate the four classifiers and the
  prototype buffer; the per-component loss assembly is at cur-mod
  lines 1098–1255.
- The presets file `cur-pre` defines the named bundles
  (`off, full, shared_only, private_only, adversarial_only,
  supervised_only, no_contrastive`).
- ref-mod has no GRL utility, no auxiliary classifiers, no contrastive
  loss; loss block (ref-mod 800–897) contains only reconstruction +
  four `kl` terms.

### A.12 Pairwise Jeffreys integration loss

- cur-mod `_jeffreys_divergence_loss` lines 661–675 and
  `_compute_jeffreys_integ_loss` lines 677–699; gated on
  `self.use_jeffreys_integ`.
- No analogue in ref-mod.

### A.13 Group balancing

- cur-mod lines 386–390: `group_loss_weights` are normalised
  (`[w / s for w in group_loss_weights]` with `s = sum(...)`).
- ref-mod has no group-weighting; `loss()` averages
  `recon_g1 + recon_g2 + ...` uniformly.

### A.14 `px_r` per group / per (group, modality)

- ref-mod lines 117–119: `self.px_r = nn.ParameterList([... per group])`.
- cur-mod lines 200–205 (single-modal mode) and lines 195–199 (multimodal,
  `nn.ParameterDict()` keyed by `f"{group}_{modality}"`).

### A.15 Multimodal extension

- cur-mod `_inference_multimodal` lines 562–637, `_loss_multimodal`
  lines 1389–1497, `_generative_multimodal` lines 957–1011.
- The two-level PoE (intra-group across modalities → inter-group across
  groups) reuses `_poe_n` at cur-mod lines 588–612 (intra) and 632–636
  (inter, via `_supervised_poe → _label_based_poe`).

---

## B. Corrections to the prior report

### B.1 Missing-label PoE expert: padding is `logvar=1`, not `logvar=0`

The prior report stated the original padded a missing-label expert with
`logvar=0` (i.e. precision 1) and characterised this as a
"double-prior bug" with shrinkage factor `0.42`. **This is wrong.**

ref-mod lines 633–636 (and the symmetric block at 651–654) actually pad
with:
```python
groups_2_stats_label = {
    "logtheta_loc": torch.zeros_like(groups_1_stats_label["logtheta_loc"]),
    "logtheta_logvar": torch.ones_like(groups_1_stats_label["logtheta_logvar"]),
}
```
i.e. `logvar = 1`, `var = e ≈ 2.718`, `precision ≈ 0.368`.

Combined with the `+1` global prior built into `_poe2` (ref-mod line 327
`logvars_joint += torch.sum(inverse_vars, dim=0)` after initialising to
`ones_like`, which corresponds to a precision-1 prior expert) and a
typical encoder posterior with `var ≈ 1`, the joint precision becomes
`1 + 1 + 0.368 = 2.368`, and the joint mean is shrunk to
`μ_g · (1 / 2.368) ≈ μ_g · 0.42` of the encoder's mean.

By contrast, cur-mod lines 824–832 pad with `logvar = 30`
(`precision ≈ exp(-30) ≈ 0`):
```python
_large_logvar = torch.full((n_cells, latent_dim), 30.0, device=device)
label_stats_for_poe[g] = {
    "logtheta_loc": torch.zeros(n_cells, latent_dim, device=device),
    "logtheta_logvar": _large_logvar,
    ...
}
```
giving joint precision `1 + 1 + 0 = 2`, joint mean `μ_g · 0.5`.

So the qualitative claim still holds — the original injects a **moderately
informative** spurious expert that pulls the posterior toward zero more
than the new code does — but the specific "double-prior" label and the
precise shrinkage numbers in the prior report were inaccurate. Net effect
on `μ_joint`: original shrinks ≈ 0.42×, new shrinks ≈ 0.50×, encoder-only
would give ≈ 0.50× as well (because `_product_of_experts` always adds the
`+1` prior). So the numerical difference is `0.50 / 0.42 - 1 ≈ +19%` more
posterior magnitude in the new code for one-of-a-kind labels — small but
non-trivial.

### B.2 "Optimisation difficulty" attributed to the flow prior

The prior report's "Loss of an analytic KL term" sentence is correct
(MC vs. closed-form variance) but the framing implied the MC estimate is
biased. It is **not** biased — the MC log-ratio is an unbiased estimate
of the per-cell KL up to the rsample noise that already exists in the
fixed-prior case. Only the *variance* differs. Corrected here.

### B.3 "ELBO is internally inconsistent" claim in §9 of the prior report

The prior report's strongest claim — "the original ELBO surrogate does
not correspond to any consistent generative model" — is too strong.
- The slicing swap (A.5) yields a valid p(x|z) with mis-labelled
  parameters.
- The mixer combiner (A.4) does not enter the training likelihood
  (A.4 caveat).
- The `log(1+x)` NB target (A.6) is a valid ELBO of an *implied*
  generative model whose data is `log(1+counts)`, which is just a
  different (and biologically meaningless) generative model.

A more accurate phrasing: **the original optimises a valid ELBO, but of
a generative model whose interpretation does not match the count-data
intent. The user-facing parameter labels and the auxiliary `px_scale`
output are not what they claim to be.**

### B.4 "factor_regressor_private receives shared, vice versa" downstream effect on `get_loadings`

Confirmed and now anchored:
- ref-mod `get_loadings` lines 805–820 reads `factor_regressor_private`
  and `factor_regressor_shared` weights and divides out their BatchNorm
  statistics. Because of the slicing swap (A.5), in the original the
  weights returned for `type_latent="private"` are the gradients of the
  reconstruction with respect to the (mis-fed) shared-block slice. So
  loadings interpretation in the original is unreliable. cur-mod has the
  same `get_loadings` logic at lines 1014–1056 but is fed correctly
  sliced inputs, so loadings recover the intended interpretation.

---

## C. New caveats / things the prior report did not flag

### C.1 The encoder's `theta = F.softmax(log_z, -1)` is computed but unused downstream in `generative`

- ref-nn line 132 / cur-nn line 159: encoder always computes
  `theta = F.softmax(log_z, -1)`.
- In both packages, `theta` is included in the encoder output dict but
  the `generative` path consumes only `log_z` (via the concatenation
  on cur-mod lines 925–933 and ref-mod lines 745–749).

This is not a bug, but it is dead computation — and the name `theta`
hints at a topic-model interpretation that the rest of the code does
not enforce. Worth being aware of for anyone porting these modules.

### C.2 `_product_of_experts` always adds a precision-1 "global prior" expert

Both packages add a `+1` term (PoE prior expert with `var=1`) inside
the precision sum:
- ref-mod lines 460–467 in `_product_of_experts`.
- cur-mod lines 701–708 in `_product_of_experts`.

```python
mus_joint = torch.sum(mus / vars, dim=0)
logvars_joint = torch.ones_like(mus_joint)        # ← precision-1 prior
logvars_joint += torch.sum(1.0 / vars, dim=0)
logvars_joint = 1.0 / logvars_joint
```

The standard PoE form (without the prior expert) is just `Σ μ/σ² / Σ 1/σ²`.
Adding a unit-precision prior is a deliberate regularisation choice, not
a bug — but it does mean PoE posteriors are systematically shrunk toward
zero relative to the maximum-likelihood combination of encoder
posteriors, with stronger effect when fewer groups contribute. This is
shared by both implementations and was not called out in the prior report.

### C.3 Original encoder also appears to lack KL on the *shared encoder posterior* before PoE

In ref-mod loss (lines 838–891), the four KL terms are:
- `kl_divergence_private_groups_{1,2}` against `qz_private_*`
- `kl_divergence_poe_groups_{1,2}` against `qz_poe_*`

The pre-PoE shared encoder posteriors (`shared_stats[g]["qz"]`) have no
KL term. cur-mod is the same: at cur-mod lines 1325–1346 the KL terms
are on `qz_private` and on `qz_poe`. So this is consistent across both
packages and not a divergence — just worth flagging that the per-group
shared posterior is **only** regularised through its participation in
the PoE, not directly. This was correctly omitted from the prior report
but is worth recording as a shared design choice with implications for
identifiability.

### C.4 `kl_weight` warmup applies to PoE KL in both packages

Both packages multiply `kl_poe_*` by `kl_weight`:
- ref-mod lines 880–891.
- cur-mod lines 1349–1351.

The prior report did not state this explicitly. The new code additionally
makes warmup *selective* on the disentanglement components
(cur-mod lines 1102–1255): GRL components are warmed (W-055), supervised
CE components are not. That asymmetry is novel to `spVIPESmulti`.

### C.5 The "double-prior" qualitative effect appears in `_poe_n` itself, not just in the missing-label expert

Because `_product_of_experts` (C.2) always adds a precision-1 prior
expert *inside* the PoE, calling `_poe_n` on a single real expert plus
a near-zero-precision dummy (cur-mod lines 824–832) gives joint
precision `1 + 0 = 1`, i.e. the dummy effectively cancels the second
real-group expert that would otherwise be present. This is the
intended behaviour. The original instead gives `1 + 0.368 = 1.368`,
adding measurable shrinkage. The prior report correctly identified the
direction of the effect; the magnitudes are corrected in B.1.

---

## D. Summary of changes recommended for the prior report

| § in prior report | Change |
|---|---|
| §2 (Encoder) | OK as written. |
| §3.2 (Mixer combiner) | **Sharpen**: clarify that this affects `px_scale` output only, not training likelihood. |
| §3.3 (Slicing) | **Sharpen**: this swaps parameter *labels*, ELBO is still valid for an oddly-parameterised model. |
| §4.3 (Missing-label expert) | **Correct**: original padding is `logvar=1`, not `logvar=0`; precision ≈ 0.368, not 1.0; "double-prior" framing should be replaced with "moderately informative spurious expert"; numerical shrinkage factors corrected (≈ 0.42 vs ≈ 0.50). |
| §5 (Flow prior) | **Soften**: MC-KL is unbiased (only variance differs); not "loss of analytic KL" in the bias sense. |
| §6.3 (NB target) | **Sharpen**: original is a valid ELBO of an `NB(log(1+x))` model, which is statistically nonsensical for counts; not "no statistical meaning" simpliciter. |
| §9 ("Net effect") | **Soften**: replace "does not correspond to any consistent generative model" with "optimises an ELBO of a model whose interpretation doesn't match the count-data intent, and whose user-facing parameter names are mis-labelled". |
| §3.1 (Mixer params) | **Verify**: the prior report's "≈ 540 K params" number was a back-of-envelope and should either be removed or recomputed against `n_in = n_dimensions_private + n_dimensions_shared = 35` and the actual `n_genes`. |
| §C of this audit | **Add**: items C.1–C.5 to the prior report as "shared design choices" (PoE built-in prior, no KL on per-group shared posterior, dead `theta`, kl_weight warmup behaviour). |

No claim in the prior report needs to be retracted entirely; the
corrections are about precision and rhetorical strength, not about
whether the substantive differences exist.

---

## E. Direction of scientific/factual errors

Short answer: **`spVIPESmulti` does not introduce scientific errors
relative to the original; it fixes several.** Errors flow from the
original to `spVIPESmulti` as fixes, not the other way around.

### E.1 Errors in the original that `spVIPESmulti` fixes

1. **Swapped decoder slicing** (§A.5). Original layout is
   `[private | poe_shared]` (ref-mod 745–749) but slicing in
   `generative()` is `[shared | private]` (ref-mod 757–758), so
   `factor_regressor_private` is fed (most of) the shared block and
   vice versa. Concrete consequence: `get_loadings(type_latent="private")`
   in the original returns weights of the wrong latent (§B.4). Fixed in
   cur-mod 943–944.

2. **NB likelihood evaluated on `log(1+x)`** (§A.6). Original default
   `log_variational_generative=True` (ref-mod 93) applies
   `x ← log(1+x)` *before* `NegativeBinomialMixture.log_prob`
   (ref-mod 820–824). The log-density is computable but the implied
   generative model is `NB(log(1+counts))`, which is not a count model.
   Fixed in cur-mod 1313–1320 with an in-source comment naming the bug.

3. **Decoder mixer drops the private rate** (§A.4). Original
   `px_scale = normalize((1 - mixing) * px_rate_shared)` (ref-nn 386–388)
   is not a convex combination — the private rate is silently zeroed in
   the user-facing output. Does not enter the training likelihood, but
   corrupts `get_normalized_expression`-style outputs and any analysis
   built on `px_scale`. Fixed in cur-nn 379–382.

4. **Informative spurious PoE expert for missing labels** (§B.1).
   Original pads with `logvar=1` (precision ≈ 0.368) at ref-mod
   633–636/651–654, which shrinks the joint posterior mean by ≈ 19%
   more than the new code's `logvar=30` padding (cur-mod 824–832).
   Less severe than the first-pass report claimed, but still a real,
   directional bias toward zero for cells whose label appears in only
   one group.

### E.2 Changes that are design choices, not error fixes

- `BatchNorm1d → LayerNorm` on posterior heads (§A.1) — both valid;
  LayerNorm avoids train/eval drift but breaks checkpoint compatibility
  (noted in source).
- `logvar.clamp(-4, 4)` (§A.2) — regularisation against GRL-driven
  explosions; mathematically restrictive but not incorrect.
- `SiLU` default activation (§A.3) — preference.
- Vectorised `_poe_n` replacing `_poe2` (§A.7, A.9) — mathematically
  equivalent for N=2, generalises to N>2.

### E.3 Shared design choices in both packages (not errors in either)

- `_product_of_experts` always adds a precision-1 "global prior" expert
  (§C.2). Present in both; shrinks all PoE posteriors toward zero.
- The pre-PoE per-group shared encoder posterior `qz_shared_g` has
  **no** direct KL term in either package (§C.3); it is regularised
  only through its participation in the PoE.
- Encoder always computes `theta = softmax(log_z)` but `generative`
  consumes only `log_z` (§C.1) — dead computation, misleading name.

### E.4 New code paths in `spVIPESmulti` not deeply audited here

These were confirmed to exist and to be wired in correctly, but the
underlying algebra was not re-derived in this pass. Worth focused unit
tests:

- **Jeffreys integration loss** between Gaussian PoE posteriors of
  different groups (cur-mod 661–699). Closed-form symmetric KL between
  two diagonal Gaussians not re-verified against canonical formula.
- **Prototype InfoNCE with EMA prototype updates** (referenced in the
  disentanglement block at cur-mod 1098–1255). Auxiliary objective
  (no ELBO claim); EMA momentum + prototype normalisation not
  cross-checked against the SupCon literature.
- **MC-KL for the NF prior** (cur-mod `_nf_kl` 754–793). Unbiased in
  expectation, but variance scales with `n_mc_samples`; default value
  should be checked against `scripts/validate_disentanglement.py`.

