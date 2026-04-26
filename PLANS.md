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

## P1 — Migrate to scvi-tools 1.x

**Status:** Pending.

**Why it is needed:**
`pyproject.toml` pins `scvi-tools>=0.20.0,<0.21`. The package cannot be
installed alongside any other scvi-based tool (e.g. scvi-tools itself, scANVI,
Tangram) that uses scvi ≥ 0.21. Every file in `src/spVIPES/data/` imports from
private scvi sub-modules that were reorganised in 0.21 and again in 1.0; the
pin is the only thing keeping the package from crashing at import time.

---

### What changed in scvi-tools 1.0

| Private API used | Where used in spVIPES | Status in scvi 1.x |
|---|---|---|
| `scvi._types.AnnOrMuData` | `_manager.py` line 18, `_utils.py` line 33, `fields/_base_field.py` line 7 | Module removed — `scvi._types` no longer exists |
| `scvi._types.MinifiedDataType` | `data/_utils.py` line 33 | Same — module removed |
| `scvi.data._utils.get_anndata_attribute()` | `fields/_base_field.py` line 9, `_manager.py` line 28 (via local import) | Still private; no stable public path |
| `scvi.dataloaders._anntorchdataset.AnnTorchDataset` | `data/_manager.py` line 19 | Promoted to `scvi.data.AnnTorchDataset` (public) in 1.x — verify exact path on install |
| `scvi.dataloaders._data_splitting.validate_data_split()` | `data/_multi_datasplitter.py` line 9 | Removed — must be inlined |
| `scvi.model._utils.parse_use_gpu_arg()` | `data/_multi_datasplitter.py` line 10 | Removed — `use_gpu` concept dropped; Lightning 2.x uses `accelerator`/`devices` |
| `scvi.utils.attrdict` | `data/_manager.py` line 20, ×8 call sites | Removed — **not** replaceable with `SimpleNamespace` (see T4 below) |
| `scvi.data._constants` (whole module) | `fields/_base_field.py` line 8 | Still private; the local copy in `data/_constants.py` already has every key needed |

Public APIs used by the model and module layers (`REGISTRY_KEYS`, `BaseModelClass`,
`BaseModuleClass`, `LossOutput`, `auto_move_data`, `FCLayers`, `TrainingPlan`,
`setup_anndata_dsp`, `CategoricalObsField`, `LayerField`, `NegativeBinomialMixture`,
`scvi.settings`) are **stable** across 0.20 → 1.x and require no changes.

`TrainRunner` is public but its `use_gpu` kwarg was removed in 1.x (see T6 below).

---

### Task breakdown (file-by-file)

**T1 — `pyproject.toml`**
- Change pin from `"scvi-tools>=0.20.0,<0.21"` to `"scvi-tools>=1.0,<2"`.
- Remove the explanatory comment above the pin (it will be resolved).
- Verify `requires-python` upper bound: scvi 1.x supports Python 3.10+; update
  to `>=3.10,<3.13` or whatever scvi 1.x's declared range is.

**T2 — `src/spVIPES/data/fields/_base_field.py`**

Three imports need changing (lines 7–9):

1. `from scvi._types import AnnOrMuData` → define locally at the top of the file:
   ```python
   from anndata import AnnData
   from mudata import MuData
   AnnOrMuData = Union[AnnData, MuData]
   ```
2. `from scvi.data import _constants` → `from .. import _constants`
   (`_base_field.py` lives in `data/fields/`; the local `data/_constants.py` already
   defines every constant used: `_DR_ATTR_NAME`, `_DR_ATTR_KEY`, `_DR_MOD_KEY`).
3. `from scvi.data._utils import get_anndata_attribute` → `from .._utils import get_anndata_attribute`
   (the function is already fully implemented locally in `data/_utils.py` lines 40–67).

**T3 — `src/spVIPES/data/_utils.py`**

Line 33: `from scvi._types import AnnOrMuData, MinifiedDataType`

- Define locally at module top (same approach as T2 for `AnnOrMuData`).
- `MinifiedDataType` is only used as a return-type annotation in
  `_get_adata_minify_type()` (line 249) and `_is_minified()` (line 253).
  Replace with `str | None` — the local `_constants.ADATA_MINIFY_TYPE` is already
  a NamedTuple whose only field is `LATENT_POSTERIOR: str`.
- `from scvi import settings` (line 32) — keep as-is; `scvi.settings` is public.

**T4 — `src/spVIPES/data/_manager.py`**

Three imports need changing (lines 18–20):

1. `from scvi._types import AnnOrMuData` → local alias (same as T2).
2. `from scvi.dataloaders._anntorchdataset import AnnTorchDataset` →
   `from scvi.data import AnnTorchDataset` (verify exact path after installing scvi 1.x;
   the type is used only in the return annotation of `create_torch_dataset` and in
   the `dataset = AnnTorchDataset(self, …)` call on line 329).
3. `from scvi.utils import attrdict` — **do not replace with `types.SimpleNamespace`**.
   `_manager.py` uses the returned objects with *both* attribute access
   (`data_loc.attr_name`) and item access (`data_loc[_constants._DR_ATTR_NAME]`)
   in `get_from_registry` (lines 371–374) and `_view_data_registry` (line 441).
   `SimpleNamespace` only supports attribute access and would break `[]` indexing.
   Instead, add a local `attrdict` to `data/_utils.py` (or a new `data/_compat.py`):
   ```python
   class attrdict(dict):
       """Dict subclass that also allows attribute-style access."""
       def __getattr__(self, key):
           try:
               return self[key]
           except KeyError:
               raise AttributeError(key)
       def __setattr__(self, key, value):
           self[key] = value
   ```
   Then `from spVIPES.data._utils import attrdict` (or wherever it lives).
   The eight `attrdict(…)` call sites in `_manager.py` and the type annotations on
   `data_registry`, `summary_stats`, `_get_data_registry_from_registry`,
   `_get_summary_stats_from_registry`, `get_state_registry`,
   `_view_summary_stats`, and `_view_data_registry` all continue to work unchanged.

**T5 — `src/spVIPES/data/_multi_datasplitter.py`**

Lines 9–10: remove both private imports.

- Inline `validate_data_split` (called on line 58). The logic is:
  ```python
  def _validate_data_split(n: int, train_size: float, validation_size: float | None):
      if train_size > 1.0 or train_size <= 0.0:
          raise ValueError("train_size must be between 0 and 1")
      n_train = max(1, round(n * train_size))
      if validation_size is None:
          n_val = n - n_train
      else:
          n_val = max(1, round(n * validation_size))
      if n_train + n_val > n:
          raise ValueError("train_size + validation_size > 1")
      return n_train, n_val
  ```
- Replace `parse_use_gpu_arg(self.use_gpu, return_device=True)` on line 81 with
  inline device resolution that handles all input types that the old API accepted
  (`None/True` → cuda if available, `False` → cpu, `int` → `cuda:<n>`,
  `str` → treat as device string):
  ```python
  import torch
  _gpu = self.use_gpu
  if _gpu is None or _gpu is True:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  elif _gpu is False:
      self.device = torch.device("cpu")
  elif isinstance(_gpu, int):
      self.device = torch.device(f"cuda:{_gpu}")
  else:
      self.device = torch.device(_gpu)  # str like "cuda:0"
  self.pin_memory = settings.dl_pin_memory_gpu_training and self.device.type == "cuda"
  ```
  (`accelerator` was only used for the `pin_memory` guard; `self.device` is stored
  on the splitter but never read externally — `ConcatDataLoader` does not use it.)

**T6 — `src/spVIPES/model/base/training_mixin.py`**

`TrainRunner` in scvi 1.x removed the `use_gpu` kwarg. Line 121:
```python
runner = TrainRunner(
    self,
    training_plan=training_plan,
    data_splitter=data_splitter,
    max_epochs=max_epochs,
    use_gpu=use_gpu,   # ← removed in scvi 1.x
    **trainer_kwargs,
)
```
Replace with:
```python
runner = TrainRunner(
    self,
    training_plan=training_plan,
    data_splitter=data_splitter,
    max_epochs=max_epochs,
    **trainer_kwargs,
)
```
Also remove the `use_gpu` parameter from `MultiGroupTrainingMixin.train()`'s
signature and docstring (lines 24, 46–53), and stop passing `use_gpu=use_gpu`
to `MultiGroupDataSplitter` on line 109. Callers that currently pass `use_gpu`
should use Lightning's `accelerator`/`devices` kwargs via `trainer_kwargs` instead.

**T7 — `src/spVIPES/data/_constants.py`** (verify, no edits expected)

The local `_constants.py` already defines all constants referenced by `_manager.py`
and `_base_field.py` (`_SCVI_UUID_KEY`, `_MANAGER_UUID_KEY`, `_SCVI_VERSION_KEY`,
`_MODEL_NAME_KEY`, `_SETUP_ARGS_KEY`, `_FIELD_REGISTRIES_KEY`,
`_DATA_REGISTRY_KEY`, `_STATE_REGISTRY_KEY`, `_SUMMARY_STATS_KEY`,
`_DR_ATTR_NAME`, `_DR_ATTR_KEY`, `_DR_MOD_KEY`, `_ADATA_MINIFY_TYPE_UNS_KEY`).
Confirm no new constants were added to `scvi.data._constants` in 1.x that the
code relies on.

---

### Validation

After all tasks are done:

1. `pip install -e ".[dev]"` with scvi-tools 1.x installed — zero import errors.
2. `pytest -v` — 23 passed, 1 skipped (same baseline as P0).
3. `python scripts/smoke_vignettes.py` — 8/8 PASS.
4. `python scripts/validate_disentanglement.py` — results in the same ball-park
   as `scripts/validation_results.md`.
5. `python scripts/validate_disentanglement_multimodal.py` — completes without error.

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

## P8 — Disentanglement support for multimodal mode (DONE)

**Status:** **Done.** The disentanglement objective is now wired into the
multimodal loss path. Construction-time `ValueError` for multimodal +
`disentangle_*_weight > 0` removed; `_loss_multimodal` now invokes
`_compute_disentangle_losses` before returning. See "Added — Multimodal
disentanglement support" in `CHANGELOG.md` for details.

**Design choice (Option B, with per-modality private extension):**
Components 1, 2, 5 (shared / contrastive) operate on the post-PoE shared
latent — modality-agnostic by construction. Components 3, 4 (private) loop
over each modality's private latent in `_compute_disentangle_losses` (using
`inference_outputs["per_modality_private"][(g, mod)]`), summing the
per-modality cross-entropy terms with the same classifier weights. This
preserves full multimodal information without adding new parameters.

**Open follow-up:** Empirical benchmark — there is no multimodal
disentanglement validation script analogous to
`scripts/validate_disentanglement.py`. The current verification only
confirms the code runs and produces finite, grad-flowing losses; whether
multimodal disentanglement actually improves group mixing / label
preservation in `z_shared` is untested.

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
