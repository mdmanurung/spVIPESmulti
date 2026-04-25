# Deferred Implementation Plans

Ideas identified through literature review (Multi-ContrastiveVAE, CellDISECT, sysVI)
that have not yet been implemented, ordered by estimated impact.

---

## P0 — Apply two pending disentanglement bug fixes (URGENT)

**Status:** **Not yet pushed.** Validated empirically in the session that
produced this PR (see `scripts/validate_disentanglement.py`,
`scripts/validation_results.{json,md}`, `scripts/smoke_vignettes.py`,
`scripts/smoke_vignettes_results.csv`). Both fixes are required for
`disentangle_preset='full'` (and `shared_only`, `supervised_only`, plus any
custom config with `contrastive_weight > 0`) to run on PyTorch >= 2.x.

The session hit a persistent HTTP 503 on `git push receive-pack` (~30 retries
with exponential backoff, all failed). Files were pushed one at a time via
the GitHub MCP API. The patched `src/spVIPES/module/spVIPESmodule.py`
(1389 lines / ~60 KB) was the only file too large to push reliably as a
single MCP `create_or_update_file` `content` argument, so the fixes are
captured here as exact diffs that any subsequent session can re-apply.

### Fix 1 — `register_buffer("prototypes", ...)` crash on PyTorch 2.x

**File:** `src/spVIPES/module/spVIPESmodule.py`
**Lines:** ~330–334 (end of `__init__`, just before the
`_cluster_based_poe` method definition)

PyTorch 2.x's `register_buffer()` refuses to re-bind a name already present
in `__dict__`. The original code assigns `self.prototypes = None` first,
which works on the first instance but crashes on the second with
`KeyError: "attribute 'prototypes' already exists"`. This breaks any
workflow that builds two `spVIPES` models in the same process (e.g.,
`scripts/validate_disentanglement.py`, the ablation notebook).

**Original:**

```python
        # Optional prototype buffer for contrastive InfoNCE on z_shared
        self.prototypes = None
        if contrastive_weight > 0 and use_labels:
            self.register_buffer("prototypes", torch.zeros(n_groups, n_labels, n_dimensions_shared))
            self.prototype_momentum = 0.99
```

**Replace with:**

```python
        # Optional prototype buffer for contrastive InfoNCE on z_shared.
        # Use register_buffer either way so the attribute is owned by nn.Module's
        # _buffers dict — assigning self.prototypes = None first crashes on
        # PyTorch >= 2.x because register_buffer refuses to re-bind a name
        # already present in __dict__.
        if contrastive_weight > 0 and use_labels:
            self.register_buffer("prototypes", torch.zeros(n_groups, n_labels, n_dimensions_shared))
            self.prototype_momentum = 0.99
        else:
            self.register_buffer("prototypes", None)
```

### Fix 2 — float-tensor indexing in the contrastive prototype EMA update

**File:** `src/spVIPES/module/spVIPESmodule.py`
**Lines:** ~1221–1232 (in `_compute_disentangle_losses`, the
"Component 5 (contrastive)" branch)

`labels_by_group[g]` is a float tensor (categorical codes from scvi-tools'
dataloader). The other label-using branches of `_compute_disentangle_losses`
all cast `.long()` before use (see lines ~1184, 1214, 1242 — all
`labels_by_group[g].long()` arguments to `F.cross_entropy`). The contrastive
branch missed the cast, and `self.prototypes[g, lbl]` with a float `lbl`
raises:

```
IndexError: tensors used as indices must be long, int, byte or bool tensors
```

This blocks `disentangle_preset='full'` (`contrastive_weight=0.5`),
`'shared_only'` (`contrastive_weight=0.5`), and `'supervised_only'`
(`contrastive_weight=0.5`).

**Original:**

```python
        # Component 5 (contrastive): InfoNCE on z_shared via EMA prototypes
        if self.prototypes is not None:
            with torch.no_grad():
                for g in range(n_groups):
                    z = inference_outputs["poe_stats"][g]["logtheta_log_z"].detach()
                    for lbl in labels_by_group[g].unique():
                        mask = labels_by_group[g] == lbl
                        if mask.sum() > 0:
                            self.prototypes[g, lbl] = (
                                self.prototype_momentum * self.prototypes[g, lbl]
                                + (1 - self.prototype_momentum) * z[mask].mean(0)
                            )
```

**Replace with (lift `.long()` cast out of the unique-loop):**

```python
        # Component 5 (contrastive): InfoNCE on z_shared via EMA prototypes
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

### How a fresh session should re-apply these

1. Check out branch `claude/plan-next-steps-07VbS` (or fresh from `main`
   after this PR merges).
2. Apply both edits above in `src/spVIPES/module/spVIPESmodule.py`.
3. Verify locally:
   - `pip install -e ".[dev,test]" igraph leidenalg tabulate`
   - `pytest -v` — expect 23 pass, 1 skipped.
   - `python scripts/smoke_vignettes.py` — expect 8/8 PASS in
     `scripts/smoke_vignettes_results.csv`. **Without the fixes** the two
     `*-full` cases (`single-modality / 2-group / NSF prior on shared / full`
     and `single-modality / 3-group / label PoE / full`) FAIL with the
     `IndexError` from Fix 2. The harness now also catches Fix 1 simply by
     instantiating two models in the same process during the
     `case_multimodal_disentangle_must_raise` test.
   - (Optional, ~5 min on CPU) `python scripts/validate_disentanglement.py`.
     Expect ARI on `z_shared` ≈ 0.272 (off) lifting to 0.31–0.37 on
     disentanglement-enabled presets, mirroring
     `scripts/validation_results.md`.
4. `git push` (or, if the proxy is still flaky, push the single file via
   `mcp__github__create_or_update_file` with the path
   `src/spVIPES/module/spVIPESmodule.py`).

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
