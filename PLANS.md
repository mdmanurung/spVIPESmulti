# Deferred Implementation Plans

Ideas identified through literature review (Multi-ContrastiveVAE, CellDISECT, sysVI)
that have not yet been implemented, ordered by estimated impact.

---

## P0 — Apply two pending disentanglement bug fixes (DONE)

**Status:** **Done.** Both fixes landed in commit `ab77c5f` on branch
`claude/implement-p0-fixes-olw8x` and are documented under "Fixed" in
`CHANGELOG.md`. Verified in the same session:

- `pytest -v` → 23 passed, 1 skipped (matches the expected baseline).
- `python scripts/smoke_vignettes.py` → 8/8 PASS, including the two
  `*-full` cases that previously hit `IndexError` and the multimodal-disentangle
  case that instantiates two models in one process (which previously hit the
  `register_buffer` `KeyError` from Fix 1).

Fix 1 made `register_buffer("prototypes", ...)` PyTorch 2.x-safe by always
owning the attribute through `nn.Module._buffers` (using `None` in the off
branch) instead of pre-assigning `self.prototypes = None`. Fix 2 lifted the
`.long()` cast on `labels_by_group[g]` out of the `unique()` loop in the
contrastive EMA update so `self.prototypes[g, lbl]` indexes with an integer
tensor.

---

## P5 — Counterfactual cross-group augmentation

**Source:** CellDISECT (Megas et al., 2025)

**What it does:**
For each cell in group A, generate a synthetic "group-B version" by decoding its
`z_shared` together with a `z_private` sampled from group B's posterior bank, then
re-encoding to verify the shared representation is preserved.

**Why deferred:** Requires a second encoder forward pass per step (~30–50% compute
overhead) and a maintained per-group private posterior bank. Subsumes standalone cycle
consistency; best implemented after disentanglement classifiers (P1–P4) are validated.

**Loss terms:**
```
L_cycle  = ||z_shared_A − re_encode(decode(z_shared_A, z_private_B, group=B))||²
L_classify = D_group(re_encode(x̃_AB)) should be group-invariant  [reuse q_group_shared]
```

**Key implementation notes:**
- `self.decoders[g].forward(dispersion, z_private, z_shared, library, *cat_list)`
  already accepts raw `z_private` and `z_shared` tensors directly.
- Private posterior bank: maintain EMA of `z_private` per group (similar to prototype
  buffer already in place for contrastive).
- Gate behind `counterfactual_weight: float = 0.0`.

---

## P6 — Multi-covariate generalization

**Source:** CellDISECT (Megas et al., 2025)

**What it does:**
Extend `groups_key` from a single string to a list of covariate keys (e.g.,
`["batch", "condition", "donor"]`). Each covariate gets its own private encoder and
private latent, with disentanglement classifiers for every covariate pair.

**Why deferred:** Major architectural refactor — affects data preparation, AnnData
registration, encoder/decoder instantiation, PoE strategy selection, and the loss
function. Requires a new `groups_modality_lengths`-style multi-covariate metadata
structure in `adata.uns`.

**Scope:**
- `data/_utils.py`: extend `prepare_adatas` to accept multiple covariate keys.
- `module/spVIPESmodule.py`: replace single `groups_lengths` dict with nested
  `covariates_lengths: dict[str, dict]`.
- `model/spvipes.py`: accept `covariate_keys: list[str]` alongside `groups_key`.
- Disentanglement classifiers: scale to `C(C+1)` auxiliary networks for `C` covariates.

---

## P7 — Reference-group decoder masking

**Source:** Multi-ContrastiveVAE (Wang et al., ICLR 2024) — `salient=0` for controls

**What it does:**
Designate one group as a reference (e.g., unperturbed / wild-type). For that group,
mask the private latent contribution at the decoder (set mixing weight = 1, so
reconstruction uses only `z_shared`). This forces other groups' private latents to
encode *deviation from the reference* rather than arbitrary group-specific variation.

**Why deferred:** Useful only for asymmetric comparisons (treatment vs. control,
perturbed vs. unperturbed). Not appropriate for symmetric comparisons (cross-species,
cross-modality). Risk of degenerate solution if reference group dominates PoE and pulls
all groups' shared latents toward the reference manifold.

**Implementation notes:**
- Add `reference_group: str | None = None` to `setup_anndata`.
- In `LinearDecoderSPVIPE.forward()`, if group matches reference, override
  `px_mixing` to a large positive constant (≈ `+∞`) so `sigmoid(px_mixing) ≈ 1`
  (shared-only reconstruction).
- Do **not** implement as KL reweighting — that risks collapsing the shared space.
- Gate behind `reference_group` parameter; document clearly as asymmetric-only.

---

## P8 — Disentanglement support for multimodal mode

**Source:** Internal — extension of currently-implemented disentanglement objective

**What it does:**
Currently `_loss_multimodal()` returns early before the disentanglement block in `loss()`,
so disentanglement/contrastive does not apply when `is_multimodal=True`. Extend the disentanglement objective to
support multimodal data by either adding per-modality classifiers or by
sharing a single classifier on the joint multimodal shared latent (computed
via PoE across modalities).

**Why deferred:** Single-modality disentanglement should be benchmarked first to confirm
empirical gains before adding multimodal complexity.

**Implementation notes:**
- Option A: per-modality `q_label_*` classifiers, one per (group, modality).
  More flexible but more parameters.
- Option B: single classifier on the post-PoE shared latent. Simpler; assumes
  shared latent is consistent across modalities.
- Group classifier likely shared across modalities (group ID is one per cell).
- Add disentanglement block to `_loss_multimodal()` mirroring the single-modality flow,
  reusing `_compute_disentangle_losses()` by parameterising it on the latent source.

---

## Verification checklist (for all deferred plans)

When implementing any plan above, validate against the existing simulated benchmark:

| Metric | Tool | Expected direction |
|---|---|---|
| Group mixing in `z_shared` | kBET, iLISI | ↑ (more mixing) |
| Label preservation in `z_shared` | cLISI, ARI | ↑ (better separation) |
| Group separability in `z_private` | silhouette per group | ↑ |
| Reconstruction quality | held-out NLL | should not degrade |
| Training stability | loss curve | no divergence over 100 epochs |
