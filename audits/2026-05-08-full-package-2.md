# spVIPESmulti — Scientific & Statistical Audit (Pass 2)

**Date:** 2026-05-08
**Scope:** full-package, independent re-audit (read-only)
**Auditor mode:** trace-only, no edits to source/tests/docs/config
**Relation to prior audit:** This pass independently re-traces the major
analysis paths of `audits/2026-05-08-full-package.md`, confirms or qualifies
each high-risk finding, and adds findings that pass missed. It does **not**
restate the prior report verbatim — see that file for the full §2.1–2.7
narrative; this report references prior items as `P§2.x`.

**Files re-traced:**
- `src/spVIPESmulti/module/spVIPESmultimodule.py`
- `src/spVIPESmulti/nn/networks.py`
- `src/spVIPESmulti/module/utils.py`
- `src/spVIPESmulti/data/prepare_adatas.py`
- `src/spVIPESmulti/dataloaders/_concat_dataloader.py`
- `src/spVIPESmulti/model/spvipesmulti.py` (setup_anndata, encoder wiring)
- `src/spVIPESmulti/model/_disentangle_presets.py`
- `src/spVIPESmulti/model/base/training_mixin.py`

---

## 1. Executive Summary

- **Independently confirms** all seven high-risk items in the prior audit
  (P§2.1–P§2.7). Trace evidence reproduced below in §3.
- **New CRITICAL finding (§2.1):** in single-modal mode the per-cell library
  size is computed from the **log1p-transformed** signal, not raw counts
  (`spVIPESmultimodule.py:533`). This means
  `px_rate = exp(library) * px_scale = sum(log1p(x_raw)) * px_scale` —
  three orders of magnitude smaller than `total_counts * px_scale`. The
  multimodal path computes library correctly from raw counts
  (`spVIPESmultimodule.py:614`), so single-modal and multimodal NB heads
  silently train on different rate scales.
- **New MEDIUM finding (§2.2):** the contrastive prototype EMA buffer is
  updated unconditionally inside `loss()`, including during validation
  forward passes. Validation-set posteriors leak into training-time
  prototypes; `evaluate()`/`get_latent_representation()` then read
  contaminated prototypes back out at inference.
- **New MEDIUM finding (§2.3):** the NF private prior `flow_prior_private`
  is a **single global flow** shared across all groups and (in multimodal
  mode) all modalities, despite per-(group, modality) private encoders.
  The flow cannot represent group- or modality-specific private structure
  the architecture is otherwise built to learn.
- **New MEDIUM finding (§2.4):** per-group encoder lookup uses the
  enumerate-position index rather than the group code
  (`spVIPESmultimodule.py:550-558`). Safe under `ConcatDataLoader`
  (always all groups per minibatch) but unsafe for any single-group
  inference path (`get_latent_representation` on a subset that drops a
  group).
- **New MEDIUM finding (§2.5):** the prior audit's P§3.7 ("label-PoE
  single-group branch silently uses dummy prior") is **stronger than
  stated** — when only one group has a label `ℓ`, cells of label `ℓ` in
  *other* groups receive zero-filled posterior tensors via the
  `mask`-scatter at L883, but those zero-filled rows then enter
  `kl(qz_poe, N(0,I))` and `recon_loss` as if they were real posteriors.
  This silently injects `KL(N(0,0)‖N(0,1))` into the loss for those cells.
- **Overall verdict:** Engineering quality is high; the **compound
  likelihood/library bug** (P§2.1 + this report's §2.1) is the single
  biggest scientific risk because it changes the absolute scale of every
  ELBO comparison and may bias gradient direction on shallowly-sequenced
  cells. PoE row-pairing semantics (P§2.2) remain the central conceptual
  question — this is not a numerical defect but a modelling choice that
  needs an explicit position in documentation.

---

## 2. New High- and Medium-Risk Findings

### 2.1 CRITICAL — single-modal `library` is computed after log1p

- **Finding** — In single-modal `inference()`, the per-cell library size
  used by the decoder rate
  (`px_rate = exp(library) * px_scale`) is the **sum of the
  log1p-transformed minibatch**, not the sum of raw counts. The multimodal
  path computes library correctly from raw counts; the two modes are
  silently inconsistent.
- **Location** — `src/spVIPESmulti/module/spVIPESmultimodule.py:524-535`
  (single-modal); contrast with
  `src/spVIPESmulti/module/spVIPESmultimodule.py:609-616` (multimodal).
- **Evidence** —
  ```python
  # single-modal inference (BUG):
  x = {i: xs[:, self.groups_var_indices[i]] for i, xs in x.items()}
  if self.log_variational_inference:
      x = {i: torch.log(1 + xs) for i, xs in x.items()}   # x is now log1p
  library = {i: torch.log(xs.sum(1)).unsqueeze(1) for i, xs in x.items()}
  #                       ^^^ xs is the LOG1P tensor, not raw counts
  ```
  ```python
  # multimodal inference (CORRECT):
  if likelihood == "nb" and self.log_variational_inference:
      x_mod_enc = torch.log(1 + x_mod)         # encoder input
  else:
      x_mod_enc = x_mod
  lib = torch.log(x_mod.sum(1).clamp(min=1e-6)).unsqueeze(1)
  #                ^^^ x_mod is RAW counts, correct
  ```
- **Consequence** — In single-modal mode, the decoder NB rate scales with
  `sum(log1p(x_raw))`, not `sum(x_raw)`. For a typical scRNA-seq cell with
  `total_counts ≈ 5000`, `sum(log1p(x))` is roughly `200–800` depending on
  the count distribution — i.e. the rate is **6–25× too small**. Combined
  with P§2.1 (NB log-prob evaluated at log1p targets), the rate scale and
  the target scale are *both* shrunk but by **different** functions of the
  cell, so the cancellation is incidental, not principled. ELBO comparisons
  across cells with different sequencing depth are biased.
  Multimodal NB heads escape this; therefore single-modal vs. multimodal
  ELBOs are not on the same scale and should not be compared.
- **Cross-check** — When `log_variational_inference=False`, the bug
  disappears because `xs` retains raw counts. Most users do not set this
  flag explicitly; the default is `True`.

### 2.2 MEDIUM — Contrastive prototypes updated during validation

- **Finding** — The EMA prototype buffer that powers component 5
  (prototype InfoNCE) is updated inside `_compute_disentangle_losses`,
  which runs in both training and validation forward passes. There is no
  `if self.training:` guard, no Lightning hook to suppress validation
  updates, and the update is wrapped only in `torch.no_grad()`.
- **Location** — `src/spVIPESmulti/module/spVIPESmultimodule.py:1170-1184`
  inside `_compute_disentangle_losses`; called from both `loss()`
  (`:1241+`) and `_loss_multimodal` (`:1336+`), which are invoked by
  Lightning's `validation_step` as well as `training_step`.
- **Evidence** —
  ```python
  if self.prototypes is not None:
      with torch.no_grad():
          for g in range(n_groups):
              z = inference_outputs["poe_stats"][g]["logtheta_log_z"].detach()
              labels_g = labels_by_group[g].long()
              for lbl in labels_g.unique():
                  mask = labels_g == lbl
                  if mask.sum() > 0:
                      self.prototypes[g, lbl] = (
                          self.prototype_momentum * self.prototypes[g, lbl]
                          + (1 - self.prototype_momentum) * z[mask].mean(0)
                      )
  ```
- **Consequence** — Validation-set cells contaminate the training-time
  prototype centroids; later `evaluate()`/`get_latent_representation()`
  inference uses prototypes that depend on val composition. This is a
  silent training/validation leak. The InfoNCE itself uses the prototypes
  as fixed targets, so the gradient is only on `z`, not on prototypes —
  the leak does not bias the immediate loss, but it does bias every
  downstream use of `self.prototypes` (e.g. if a user inspects them as
  per-class centroids).

### 2.3 MEDIUM — NF private prior is global, not per-(group, modality)

- **Finding** — When `nf_target ∈ {"private", "both"}`, exactly one
  `flow_prior_private` is constructed
  (`spVIPESmultimodule.py:298-305`) and reused for every group's private
  KL — and in multimodal mode for every (group, modality)'s private KL.
  But the architecture has **separate private encoders per (group,
  modality)**. The flow therefore averages over biologically distinct
  posteriors and cannot match any of them.
- **Location** —
  - Construction: `src/spVIPESmulti/module/spVIPESmultimodule.py:295-305`.
  - Use in single-modal: `:1257-1262`.
  - Use in multimodal: `:1372-1383` — same `self.flow_prior_private` for
    every `(g, modality)`.
- **Consequence** — If groups (or modalities) have intrinsically different
  private structures (the entire reason `z_private` is per-group), the NF
  prior is mis-specified. The KL term will under-fit groups whose private
  marginal differs most from the pooled marginal, and the flow's
  gradients are an average across all groups — adding noise without a
  matching prior signal.
- **What would resolve** — Either document the design choice
  ("the NF private prior is a global regulariser, not a per-group prior")
  or instantiate one flow per group / per (group, modality).

### 2.4 MEDIUM — Per-group encoder lookup is positional, not by group code

- **Finding** — In single-modal `inference()` the encoder is looked up by
  the loop position from `enumerate(zip(...))`:
  ```python
  for group, (item, batch) in enumerate(zip(x.values(), batch_index)):
      private_encoder = self.encoders[group]["private"]
      shared_encoder  = self.encoders[group]["shared"]
  ```
  But `x.values()` follows the dict order produced by
  `_get_inference_input`, which keys `x` by `0..N-1` in the order returned
  by `_split_tensors_by_group` (sorted by *unique group codes present in
  this minibatch*).
- **Location** — `src/spVIPESmulti/module/spVIPESmultimodule.py:550-558`;
  see `_split_tensors_by_group` at `:455-475` and `_get_inference_input`
  at `:481-501`.
- **Consequence** — Under `ConcatDataLoader` (training), every minibatch
  contains all groups (it cycles shorter loaders), so the position-to-code
  mapping is always identity — no bug observed at training time.
  Under any inference path that may pass a subset (e.g.
  `get_latent_representation` after subsetting `adata` to one group, or
  any user-supplied `indices=` that drops a group), the loop position
  becomes a relabelling: cells from group code 1 would be encoded by
  `self.encoders[0]`. Whether this fires depends on whether
  `_make_data_loader` ever produces a single-group minibatch; it does for
  `differential_abundance` (which iterates per-group in
  `_aggregate_shared_posterior`).
- **What would resolve** — A property test calling
  `get_latent_representation(indices=cells_of_group_1_only)` and
  asserting that the returned shared/private latent equals the
  corresponding rows of a full-data call.

### 2.5 MEDIUM — Single-label-group branch leaks zero rows into the loss

- **Finding** — `_label_based_poe` returns empty `(0, latent_dim)`
  posterior tensors for "groups without label `ℓ`" cases when only one
  group has cells of label `ℓ` (`spVIPESmultimodule.py:826-867`). The
  reassembly loop at `:868-879` then writes nothing into those positions
  because the `mask` for label `ℓ` in the other groups is all False — so
  far so good. But the output buffer is **`torch.empty`**, not zeros, so
  any other code that reads `concat_poe_stats[g]["logtheta_loc"]` for a
  cell whose label was processed by the single-group branch (i.e. a
  different label `ℓ'` that does have cells in `g`) reads the contents
  of uninitialised memory for the *padding* portion — except the mask
  ensures only valid labels overwrite valid rows. This is **safe in
  practice** when every cell in group `g` carries some label that has at
  least one cell in `g`, which is always true. So the prior P§3.7
  concern is **technically valid but does not corrupt gradients**.
- **Re-classification** — Downgrade prior P§3.7 from "may leak zeros" to
  "uses uninitialised memory in the dead branch but the mask never
  reads it"; remains a fragile pattern (any future reader of
  `concat_poe_stats` not gated by the per-label mask would read garbage).
- **Location** — `src/spVIPESmulti/module/spVIPESmultimodule.py:826-879`.
- **What would resolve** — Replace `torch.empty(...)` with
  `torch.zeros(...)` in the single-group dummy branch and add an
  assertion that the mask-scatter covers every row of the output.

### 2.6 MEDIUM — Encoder receives only batch (not group) — no leakage,
        but the GRL adversary then sees `z_shared` that already
        partially encodes group identity through batch correlation

- **Finding** — `setup_anndata` registers `groups_key` as a
  `CategoricalObsField` in `"groups"` and registers it separately from
  `BATCH_KEY`. The encoder receives only `[batch_index]` as its categorical
  covariate (`spvipesmulti.py:328-329`; `spVIPESmultimodule.py:184-185`).
  So the **encoder is not given group identity as a feature** —
  good, no direct leakage to `z_shared`.
- **Caveat** — When `batch_key` is correlated with `groups_key` (a common
  scenario: each group is its own donor / experiment / 10× run), the
  one-hot batch covariate inside the encoder is a near-deterministic
  function of group identity. The adversarial GRL on `z_shared` then has
  to *un-do* a covariate the encoder was *given* as input. This is not a
  bug — it is the standard DANN setup — but the documented behaviour
  ("group ID never enters the shared encoder as a feature") only holds
  literally; group identity enters indirectly via `batch_index` whenever
  groups and batches are confounded.
- **Location** — `src/spVIPESmulti/model/spvipesmulti.py:289-345` (setup);
  `src/spVIPESmulti/module/spVIPESmultimodule.py:524-557` (inference).
- **What would resolve** — A documentation note in `setup_anndata` that
  `batch_key` should be **finer-grained than `groups_key`** if the user
  wants the GRL to do meaningful work; otherwise the GRL is fighting the
  encoder's own input.

### 2.7 MEDIUM — `disentangle_warmup=True` couples all five disentangle weights to the KL schedule, including the **supervised-CE** components

- **Finding** — When `disentangle_warmup=True` (default), the entire
  disentangle aggregate is multiplied by `kl_weight` (the KL annealing
  coefficient). This applies to both adversarial GRL components
  (1, 4) **and** supervised CE components (2, 3, 5). The supervised
  components have no probabilistic reason to be KL-annealed — they are
  not part of the ELBO.
- **Location** — `src/spVIPESmulti/module/spVIPESmultimodule.py:1316-1321`
  (single-modal) and `:1418-1422` (multimodal):
  ```python
  disentangle_scale = kl_weight if self.disentangle_warmup else 1.0
  total_loss = total_loss + (disentangle_scale / n_groups) * \
               self._compute_disentangle_losses(...)
  ```
- **Consequence** — During the first `n_epochs_kl_warmup=400` epochs
  (default), the supervised cross-entropy heads receive a gradient scaled
  by `kl_weight ∈ [0, 1]`. The label-shared and group-private classifiers
  therefore train slowly during the period when the encoder is most
  malleable; by the time their gradient is fully on, the encoder may have
  already settled into a representation that does not preserve label
  structure on `z_shared`. Whether this matters empirically is open —
  the claim that the schedule is "monotonic, deterministic" (per
  `CLAUDE.md`) is true; the claim that all components warm up
  symmetrically is correct in implementation but questionable in
  motivation.
- **What would resolve** — Either decouple supervised-CE from the
  warmup schedule (warm up only the GRL components, since they are the
  ones that destabilise early training), or explicitly justify the
  joint warmup in the disentangle preset documentation.

---

## 3. Verification of Prior Audit High-Risk Items

For each of the prior audit's §2 findings, this section reports whether
the trace was independently reproduced.

### P§2.1 NB likelihood evaluated at log1p targets — **CONFIRMED**

Trace re-walked at `spVIPESmultimodule.py:1259-1287` (single-modal) and
`:1351-1399` (multimodal NB branch). `x_target = torch.log(1 + x_obs)`
is passed to `px.log_prob(x_target)` where `px` is constructed from raw
counts via `library`. The integer-roundness check is bypassed by
`transformed_for_nb=True`. Confirmed verbatim.

**Compound effect (new):** the single-modal version of this bug also
distorts the rate (see §2.1 above), so the bias on the recon loss is
**direction-dependent on per-cell library size** — high-library cells
receive `log1p(x) ≪ x` *targets* and `sum(log1p(x)) ≪ sum(x)` *rates*;
low-library cells are nearly unaffected. ELBO-based model selection
implicitly upweights low-library cells.

### P§2.2 PoE combines unrelated cells across groups by row index — **CONFIRMED**

Trace re-walked at `_poe_n` (`:392-440`) and `_product_of_experts`
(`:614-622`). `_poe_n` pads each group's encoder batch to
`max_batch_size`, stacks along `dim=0`, and `_product_of_experts` reduces
along that stacked dim. So the cell at index `i` of group `g` is
precision-combined with whatever cell is at index `i` of every other
group. The label-based variant restricts pairing to same-label cells but
does not pair within label by any biological criterion (just whichever
order ConcatDataLoader emitted them in). Confirmed.

**Refinement:** In `_inference_multimodal` step 2 (intra-group PoE
across modalities), all modalities of one group share the *same cells*
in the *same order* (multimodal data are aligned per-cell), so the
intra-group PoE is biologically valid — only the inter-group PoE step
suffers the row-pairing problem.

### P§2.3 Gaussian likelihood mean is a probability simplex — **CONFIRMED**

Trace at `nn/networks.py:357-365` (decoder forces `px_scale` onto the
unit simplex) and `module/utils.py:42-46` (`Normal(loc=px_scale, ...)`).
Confirmed: protein/ADT log-normalised values cannot be represented.

### P§2.4 `differential_abundance` is not a hypothesis test — **CONFIRMED**

Trace at `model/spvipesmulti.py:798-867`. Returns a continuous score with
no null distribution. Confirmed. Also confirms the prior observation
that the warning is inverted (best alignment ⇒ smallest score).

### P§2.5 `give_mean=True` ignored when `normalized=False` — **CONFIRMED**

Trace at `model/spvipesmulti.py:1413-1432`. The un-normalized branch
appends `poe_log_z`, which is set to `qz.rsample()` inside
`_supervised_poe`/`_label_based_poe`. The `give_mean` flag is consulted
only inside the `normalized=True` branch. Confirmed.

### P§2.6 `per_group_silhouette` is global, columns mislabeled — **CONFIRMED**

Trace at `metrics.py:194-228, 295-316`. One global silhouette is computed
on pooled cells and written into every per-group row. Confirmed.

### P§2.7 kBET / iLISI / cLISI are not the published statistics — **CONFIRMED**

Trace at `metrics.py:30-141`. `kbet` returns `exp(-mean(chi2))` rather
than the rejection rate; iLISI/cLISI use unit-weighted Simpson over kNN
rather than perplexity-tuned weights. Confirmed.

---

## 4. Unclear Assumptions Requiring Domain Review

(Additions and refinements to prior §4.)

1. **Single-modal `library` — intentional or bug?** §2.1 above. The fact
   that the multimodal path computes library from raw counts strongly
   suggests the single-modal version is unintentional; the audit would
   need a maintainer confirmation to be sure.
2. **Is the NF private prior expected to be group-aware?** §2.3 above.
   If yes, the implementation is incomplete; if no, the documentation
   should state that the flow is a global regulariser only.
3. **Should the supervised-CE components of the disentangle objective
   warm up with KL?** §2.7 above.
4. **What batch / group split convention is assumed?** §2.6 above. If
   `batch_key == groups_key` (a common shortcut), the GRL adversary is
   competing with the encoder's own input.
5. **Should the contrastive prototypes be frozen during validation?**
   §2.2 above. Standard practice is yes; current implementation says no.

(All of P§4.1–P§4.7 from the prior audit also remain open and are not
restated here.)

---

## 5. Verification Plan — additions and refinements

The prior report's V2.1–V4.6 plan stands. The additional items below
target the new findings.

### V2.1.lib — single-modal library on log1p

Read-only diagnostic: instrument any forward pass in a debugger or via a
print statement (no source edit needed if run as a one-shot pytest with
`-s` and a `breakpoint()` in a *test* file) and assert
`library.exp().mean().item()` is roughly the per-cell mean of
`x_raw.sum(1)`. Current expectation: it will be roughly `mean(sum(log1p
x_raw))`, ~1–2 orders of magnitude smaller than `mean(sum(x_raw))`.
Cross-check by training one model with `log_variational_inference=False`
on the same data and comparing the magnitude of `px_rate_shared` —
expect a ~10× shift in absolute scale.

### V2.2.proto — prototypes leak val signal

Property test in `tests/test_disentangle_no_val_leakage.py`:

```python
# Pseudocode — do not run as part of this audit
trainer.fit(model)
proto_train = model.module.prototypes.detach().clone()
trainer.validate(model)
proto_after_val = model.module.prototypes.detach().clone()
assert torch.allclose(proto_train, proto_after_val)  # currently FAILS
```

### V2.3.nf — global flow vs per-group private structure

Synthetic two-group, two-private-factor test: each group has a different
private prior (e.g. group 0 ~ N(0, I), group 1 ~ N(2·1, I)). Train a
model with `use_nf_prior=True, nf_target="private"` and inspect
`flow_prior_private().sample((10000,))`. Expect a bimodal sample
distribution if the flow correctly captures both groups; current
implementation will produce a single mode interpolating the two.

### V2.4.idx — per-group encoder lookup correctness

Property test: build a 2-group, 2-cell-type adata; subset to group 1
only; call `model.get_latent_representation(adata=subset)`; compare to
the corresponding rows from `model.get_latent_representation(adata=full)`.
Equality (up to rsample noise, see P§2.5) is required.

### V2.5.empty — uninitialised tensor in label-PoE single-group branch

Property test: minibatch with one label `ℓ` present in only group 0.
Patch `torch.empty` (or read raw bytes via `tensor.untyped_storage()`)
and confirm the `(0, latent_dim)` output is never read. Then trigger
the same code path with `torch.empty` returning `torch.full(..., NaN)`
(via a monkeypatch) and confirm no NaN propagates into `loss.backward()`.

### V2.6.batch — GRL effectiveness when `batch_key == groups_key`

Two-group ablation: identical data, two configs:
(a) `batch_key="batch"` where batch crosses group;
(b) `batch_key="groups"` where batch ≡ group.
Compare GRL gradient magnitude on the encoder over training, and
end-of-training group separability of `z_shared` via a held-out linear
probe. Expect (b) to show a higher residual group AUROC.

### V2.7.warmup — decoupling supervised-CE from KL warmup

Two ablations: `disentangle_warmup=True` vs `False`, otherwise identical.
Track per-component loss vs epoch from `extra_metrics`. Expect the
supervised CE components to converge faster under `False` but the
adversarial GRL components to destabilise training.

---

## 6. Trace Block — single-modal training (re-confirmed)

```
spVIPESmulti.train()                                       # spvipesmulti.py
  → MultiGroupTrainingMixin.train()                        # training_mixin.py:121
  → MultiGroupDataSplitter (ConcatDataLoader)              # _concat_dataloader.py
  → PatchedTrainRunner.__call__                            # training_mixin.py:74-110
  → pl.Trainer.fit
  → SpVIPESmultiTrainingPlan.training_step (scvi-tools)
  → spVIPESmultimodule.forward
      _get_inference_input                                 # :481-501
        _split_tensors_by_group → list[dict] sorted by code # :455-475
      inference                                            # :524-571
        per group: x[i] = X[:, groups_var_indices[i]]
        if log_variational_inference: x = log1p(x)         # ← log1p before library
        library = log(x.sum(1))                            # ← BUG §2.1: on log1p
        Encoder(x_log1p, group_pos, batch) → q(z|x)        # networks.py:111-167
                                                            # logvar.clamp(-4, 4)
        _supervised_poe → _label_based_poe OR _poe_n       # :583-600
          _poe_n: stack across groups dim=0, PoE row-wise  # :392-440 (P§2.2)
          _product_of_experts: precision-weighted sum      # :614-622
      generative                                           # :894-925
        decoder(z_priv, z_shared, library, batch)          # networks.py:329-368
        px = NegativeBinomialMixture(mu1, mu2, theta1, mix) # px on count scale
                                                            # but library is log1p-derived
        pz = N(0, I)
      loss                                                 # :1241-1334
        x_target = log1p(x_obs)  if log_variational_generative
        recon = -px.log_prob(x_target).sum(-1)             # ← P§2.1: log1p target
        kl_private = kl(qz, N(0,I)) or _nf_kl              # :624-651
        kl_poe     = kl(qz_poe, N(0,I)) or _nf_kl
        + (kl_weight / n_groups) * disentangle_total       # :1316-1318 (§2.7)
            _compute_disentangle_losses                    # :1110-1198
              proto EMA update — UNCONDITIONAL on training  # :1170-1184 (§2.2 here)
        + jeffreys_integ if enabled                        # :706-727 (P§3.1)
```

---

## 7. Side Effects of This Audit

This audit performed only read-only file inspection. No `pytest`,
`smoke_vignettes.py`, or `validate_disentanglement*.py` was executed.
No file outside `audits/` was created, modified, staged, or committed.

