# API Reference

spVIPES v0.3.0 — shared-private Variational Inference with Product of Experts and Supervision.

## Quick reference

| Symbol | Kind | Import path |
|---|---|---|
| `spVIPES` | Model class | `spVIPES.model.spVIPES` |
| `setup_anndata` | Class method | `spVIPES.model.spVIPES.setup_anndata` |
| `train` | Instance method | `model.train(...)` |
| `get_latent_representation` | Instance method | `model.get_latent_representation(...)` |
| `get_loadings` | Instance method | `model.get_loadings()` |
| `prepare_adatas` | Function | `spVIPES.data.prepare_adatas` |
| `prepare_multimodal_adatas` | Function | `spVIPES.data.prepare_multimodal_adatas` |
| `spVIPESmodule` | PyTorch module | `spVIPES.module.spVIPESmodule` |
| `Encoder` | Neural network | `spVIPES.nn.Encoder` |
| `LinearDecoderSPVIPE` | Neural network | `spVIPES.nn.LinearDecoderSPVIPE` |
| `ConcatDataLoader` | DataLoader | `spVIPES.dataloaders.ConcatDataLoader` |

---

## Model

### `spVIPES`

The main user-facing model class. Extends `scvi.model.base.BaseModelClass` and
`MultiGroupTrainingMixin`. Call `setup_anndata` first, then construct the model,
then call `train`.

```python
import spVIPES

adata = spVIPES.data.prepare_adatas({"ctrl": adata_ctrl, "treat": adata_treat})
spVIPES.model.spVIPES.setup_anndata(adata, groups_key="groups", label_key="cell_type")
model = spVIPES.model.spVIPES(adata)
model.train(group_indices_list=group_indices_list, max_epochs=100)
latents = model.get_latent_representation(group_indices_list=group_indices_list, batch_size=512)
```

```{eval-rst}
.. currentmodule:: spVIPES

.. autosummary::
    :toctree: generated
    :template: class.rst

    model.spvipes.spVIPES

.. autoclass:: spVIPES.model.spvipes.spVIPES
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | — | AnnData registered via `setup_anndata`. |
| `n_hidden` | `int` | `128` | Hidden layer width for all encoder / decoder networks. |
| `n_dimensions_shared` | `int` | `25` | Dimensionality of the shared latent space z_shared. |
| `n_dimensions_private` | `int` | `10` | Dimensionality of each group's private latent space z_private. |
| `dropout_rate` | `float` | `0.1` | Dropout probability applied in all encoder / decoder hidden layers. |
| `use_nf_prior` | `bool` | `False` | Replace the standard N(0,I) prior with a normalizing-flow prior (Zuko). See [NF prior reference](#normalizing-flow-prior). |
| `nf_type` | `str` | `"NSF"` | Flow architecture when `use_nf_prior=True`. `"NSF"` (Neural Spline Flow) or `"MAF"` (Masked Autoregressive Flow). |
| `nf_transforms` | `int` | `3` | Number of sequential flow transforms. |
| `nf_target` | `str` | `"shared"` | Which latent(s) get the flow prior. One of `"shared"`, `"private"`, or `"both"`. |
| `disentangle_preset` | `str` | `"off"` | Named preset for the disentanglement objective. See [Disentanglement presets](#disentanglement-presets). |
| `disentangle_group_shared_weight` | `float or None` | `None` | Override the preset's weight for the adversarial group-classifier on z_shared (GRL). |
| `disentangle_label_shared_weight` | `float or None` | `None` | Override the preset's weight for the supervised label-classifier on z_shared. |
| `disentangle_group_private_weight` | `float or None` | `None` | Override the preset's weight for the supervised group-classifier on z_private. |
| `disentangle_label_private_weight` | `float or None` | `None` | Override the preset's weight for the adversarial label-classifier on z_private (GRL). |
| `contrastive_weight` | `float or None` | `None` | Override the preset's weight for the prototype InfoNCE loss on z_shared. |
| `contrastive_temperature` | `float` | `0.1` | Temperature for the InfoNCE softmax. |
| `modality_loss_weights` | `dict[str, float] or None` | `None` | Per-modality scalar multipliers on the reconstruction loss. E.g. `{"rna": 1.0, "protein": 5.0}` to up-weight the protein term. Multimodal mode only. |
| `use_jeffreys_integ` | `bool` | `False` | Add a Jeffreys (symmetric KL) integration loss between every pair of group PoE posteriors on z_shared. |
| `jeffreys_integ_weight` | `float` | `1.0` | Scalar multiplier on the Jeffreys integration loss. |
| `**model_kwargs` | | | Forwarded to `spVIPESmodule`. |

> **Tip:** Individual weight overrides stack on top of a preset.
> `spVIPES(adata, disentangle_preset="full", contrastive_weight=0.0)` enables
> all classifiers but turns off the InfoNCE term.

---

### `setup_anndata`

```python
spVIPES.model.spVIPES.setup_anndata(
    adata,
    groups_key,
    label_key=None,
    batch_key=None,
    layer=None,
    modality_likelihoods=None,
)
```

Registers fields on `adata` via `AnnDataManager` and selects the PoE strategy.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | — | The concatenated AnnData produced by `prepare_adatas` or `prepare_multimodal_adatas`. |
| `groups_key` | `str` | — | Column in `adata.obs` identifying which group each cell belongs to. |
| `label_key` | `str or None` | `None` | Column in `adata.obs` with cell-type labels. Required for label-based PoE and for label-dependent disentanglement components. |
| `batch_key` | `str or None` | `None` | Column in `adata.obs` for technical batch. Adds a one-hot batch covariate to each encoder / decoder. |
| `layer` | `str or None` | `None` | Key in `adata.layers` to use as the count matrix. Defaults to `adata.X`. |
| `modality_likelihoods` | `dict[str, str] or None` | `None` | Overrides / sets the per-modality likelihood. Written into `adata.uns["modality_likelihoods"]`. Supported values: `"nb"`, `"gaussian"`. |

**PoE strategy selection:**

| Condition | Strategy | Group limit |
|---|---|---|
| `label_key` provided | Label-based PoE | N ≥ 2 |
| `transport_plan_key` + `match_clusters=False` | OT-paired PoE | exactly 2 |
| `transport_plan_key` + `match_clusters=True` | OT-cluster PoE | exactly 2 |

For multimodal data or N > 2 groups, always use label-based PoE.

---

### `train`

```python
model.train(
    group_indices_list,
    max_epochs=None,
    batch_size=128,
    train_size=0.9,
    validation_size=None,
    early_stopping=False,
    n_epochs_kl_warmup=400,
    n_steps_kl_warmup=None,
    plan_kwargs=None,
    **trainer_kwargs,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `group_indices_list` | `list[list[int]]` | — | One inner list per group containing integer indices into the registered AnnData. Typically: `[list(map(int, g)) for g in adata.uns["groups_obs_indices"]]`. |
| `max_epochs` | `int or None` | auto | Training epochs. Auto-computed as `min(round(20000/n_cells * 400), 400)` if `None`. |
| `batch_size` | `int` | `128` | Mini-batch size per group. |
| `train_size` | `float` | `0.9` | Fraction of cells used for training. |
| `validation_size` | `float or None` | `None` | Fraction for validation. Defaults to `1 - train_size`. Remaining cells form the test set. |
| `early_stopping` | `bool` | `False` | Enable early stopping via Lightning's `EarlyStopping` callback. |
| `n_epochs_kl_warmup` | `int` | `400` | Number of epochs over which the KL weight is linearly annealed from 0 to 1. |
| `n_steps_kl_warmup` | `int or None` | `None` | Step-based KL warmup. Takes precedence over `n_epochs_kl_warmup` if set. |
| `plan_kwargs` | `dict or None` | `None` | Extra keyword arguments forwarded to `scvi.train.TrainingPlan`. |
| `**trainer_kwargs` | | | Forwarded to `pl.Trainer`. Use `accelerator="gpu", devices=1` to select a GPU (replaces the removed `use_gpu=True`). |

---

### `get_latent_representation`

```python
latents = model.get_latent_representation(
    group_indices_list,
    adata=None,
    indices=None,
    normalized=False,
    give_mean=True,
    mc_samples=5000,
    batch_size=None,
    drop_last=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `group_indices_list` | `list[list[int]]` | — | Same structure as `train`. |
| `adata` | `AnnData or None` | `None` | Optionally pass an unseen AnnData with the same variable layout. |
| `indices` | `Sequence[int] or None` | `None` | Subset of cells. `None` = all cells. |
| `normalized` | `bool` | `False` | Return softmax-normalized embeddings instead of the raw log-space latent. |
| `give_mean` | `bool` | `True` | Return the posterior mean; if `False`, returns a single sample. |
| `mc_samples` | `int` | `5000` | Monte Carlo samples used to approximate the mean for non-closed-form distributions. |
| `batch_size` | `int or None` | `None` | Inference mini-batch size. Must be provided explicitly (no global default). |
| `drop_last` | `bool or None` | `None` | Drop the last incomplete batch. Defaults to `False`. |

**Return value** — `dict` with the following keys:

| Key | Shape | Description |
|---|---|---|
| `"shared"` | `{g: (n_g, n_shared)}` | Shared PoE latent, in dataloader order. |
| `"shared_reordered"` | `{g: (n_g, n_shared)}` | Shared latent reindexed to the original `group_indices_list` order. **Use this for downstream analysis.** |
| `"private"` | `{g: (n_g, n_private)}` | Group-level private latent, dataloader order. |
| `"private_reordered"` | `{g: (n_g, n_private)}` | Private latent, original cell order. |
| `"private_multimodal"` | `{(g, mod): (n_g, n_private)}` | Per-(group, modality) private latent, dataloader order. **Multimodal only.** |
| `"private_multimodal_reordered"` | `{(g, mod): (n_g, n_private)}` | Per-(group, modality) private latent, original cell order. **Multimodal only.** |

---

### `get_loadings`

```python
loadings = model.get_loadings()
```

Returns a `dict` keyed by `(group_idx, "shared")` and `(group_idx, "private")`,
each a `pd.DataFrame` of shape `(n_features, n_latent)` with
batch-normalisation-scaled weights from the linear decoder.

> **Multimodal note:** `get_loadings()` uses integer group-index keys and only
> works for the single-modality decoder code path. In multimodal mode, access
> decoder weights directly via `model.module.decoders[(group_idx, modality)]`.
> See the tutorial notebook (§16) for a ready-to-use `multimodal_loadings()`
> helper function.

---

## Data preparation

### `prepare_adatas`

```python
from spVIPES.data import prepare_adatas

mdata = prepare_adatas(adatas, layers=None)
```

Concatenates multiple single-modality AnnData objects into one AnnData suitable
for spVIPES. At least 2 groups are required.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adatas` | `dict[str, AnnData]` | — | Mapping from group name to AnnData. Dict insertion order determines group indices. |
| `layers` | `list or None` | `None` | Reserved; not yet implemented. |

**Metadata written to `.uns`:**

| Key | Type | Description |
|---|---|---|
| `groups_lengths` | `dict[int, int]` | Number of features (n_vars) per group. |
| `groups_var_indices` | `list[ndarray]` | Column indices in the concatenated AnnData for each group's variables. |
| `groups_obs_indices` | `list[ndarray]` | Row indices for each group's cells. |
| `groups_obs_names` | `list[Index]` | Original `obs_names` for each group. |
| `groups_var_names` | `dict[str, Index]` | Variable names (with group prefix) per group. |
| `groups_mapping` | `dict[int, str]` | Mapping from group index to group name. |
| `groups_obs` | `dict[str, DataFrame]` | `.obs` DataFrame per group. |

**Notes:**

- Variable names are prefixed with the group name (e.g. `"ctrl_GAPDH"`) to
  avoid collisions in the outer-join concatenation.
- `adata.obs` gains a `"groups"` column and an `"indices"` column (within-group
  integer index, required by `setup_anndata`).

```{eval-rst}
.. autosummary::
    :toctree: generated

    data.prepare_adatas.prepare_adatas

.. autofunction:: spVIPES.data.prepare_adatas.prepare_adatas
```

---

### `prepare_multimodal_adatas`

```python
from spVIPES.data import prepare_multimodal_adatas

mdata = prepare_multimodal_adatas(adatas, modality_likelihoods=None)
```

Concatenates a nested `{group: {modality: AnnData}}` dict into one AnnData
for multimodal spVIPES runs. At least 2 groups are required.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adatas` | `dict[str, dict[str, AnnData]]` | — | Outer keys = group names, inner keys = modality names. All groups must share at least one modality. Cells within a group must be identical across modalities. |
| `modality_likelihoods` | `dict[str, str] or None` | `None` | Likelihood per modality. Supported: `"nb"` (NegativeBinomial, default) and `"gaussian"` (for log-normalised data). |

**Metadata written to `.uns`** (all standard `prepare_adatas` keys plus):

| Key | Type | Description |
|---|---|---|
| `is_multimodal` | `bool` | Always `True`. |
| `modality_names` | `list[str]` | Ordered list of modality names. |
| `modality_likelihoods` | `dict[str, str]` | Per-modality likelihood choice. |
| `groups_modality_lengths` | `dict[int, dict[str, int]]` | Feature count per (group, modality) pair. |
| `groups_modality_var_indices` | `dict[int, dict[str, ndarray]]` | Column indices in the concatenated AnnData for each (group, modality) pair. |
| `groups_modality_masks` | `dict[int, dict[str, bool]]` | `False` for missing modalities in asymmetric setups (RNA-only group, etc.). |

**Typical workflow:**

```python
adatas_dict = {
    "spleen": {"rna": adata_rna_sp, "protein": adata_prot_sp},
    "lymph":  {"rna": adata_rna_ln, "protein": adata_prot_ln},
}
mdata = spVIPES.data.prepare_multimodal_adatas(
    adatas_dict,
    modality_likelihoods={"rna": "nb", "protein": "nb"},
)
spVIPES.model.spVIPES.setup_anndata(
    mdata,
    groups_key="tissue",
    label_key="cell_types",
    modality_likelihoods={"rna": "nb", "protein": "nb"},
)
group_indices_list = [list(map(int, g)) for g in mdata.uns["groups_obs_indices"]]
model = spVIPES.model.spVIPES(mdata, use_nf_prior=True, nf_type="NSF")
model.train(group_indices_list=group_indices_list, max_epochs=100, batch_size=512)
```

```{eval-rst}
.. autosummary::
    :toctree: generated

    data.prepare_adatas.prepare_multimodal_adatas

.. autofunction:: spVIPES.data.prepare_adatas.prepare_multimodal_adatas
```

---

## Disentanglement presets

The disentanglement objective is controlled by five scalar weights on auxiliary
losses. A named preset sets all five at once; individual parameters can then
override any weight.

**The five loss components:**

| Weight parameter | Target | Mechanism | Goal |
|---|---|---|---|
| `disentangle_group_shared_weight` | z_shared | Adversarial GRL classifier for group identity | Erase group signal from shared latent |
| `disentangle_label_shared_weight` | z_shared | Supervised classifier for cell-type label | Preserve biological signal in shared latent |
| `disentangle_group_private_weight` | z_private | Supervised classifier for group identity | Preserve group signal in private latent |
| `disentangle_label_private_weight` | z_private | Adversarial GRL classifier for cell-type label | Erase biological signal from private latent |
| `contrastive_weight` | z_shared | Prototype InfoNCE (EMA) across groups | Cross-group semantic alignment |

**Available presets:**

| Preset | group_shared | label_shared | group_private | label_private | contrastive |
|---|---|---|---|---|---|
| `"off"` (default) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `"full"` | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 |
| `"shared_only"` | 1.0 | 1.0 | 0.0 | 0.0 | 0.5 |
| `"private_only"` | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| `"adversarial_only"` | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| `"supervised_only"` | 0.0 | 1.0 | 1.0 | 0.0 | 0.5 |
| `"no_contrastive"` | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

Label-dependent components (`disentangle_label_shared_weight`,
`disentangle_label_private_weight`, `contrastive_weight`) require `label_key`
to be passed to `setup_anndata`. All five components work in both single-modal
and multimodal mode (components 3 & 4 loop over each modality's private latent
in multimodal mode).

**Example — full preset with InfoNCE disabled:**

```python
model = spVIPES.model.spVIPES(
    mdata,
    disentangle_preset="full",
    contrastive_weight=0.0,
)
```

---

## Normalizing-flow prior

When `use_nf_prior=True`, the standard N(0, I) prior on the selected latent is
replaced by a Zuko normalizing-flow distribution. The KL divergence is computed
by Monte Carlo:

```
KL(q(z|x) || p_flow(z)) ≈ log q(z|x) − log p_flow(z)
```

Flow parameters are optimised jointly with the rest of the VAE through the
single ELBO objective — no separate warm-up schedule is required.

| Parameter | Values | Notes |
|---|---|---|
| `nf_type` | `"NSF"`, `"MAF"` | NSF (Neural Spline Flow) is the recommended default; MAF is faster but less expressive. |
| `nf_transforms` | int ≥ 1 | Number of sequential transforms. More transforms → more flexible prior but more parameters. Typical range: 3–8. |
| `nf_target` | `"shared"`, `"private"`, `"both"` | `"both"` fits independent flows on shared and private, doubling flow parameter count. |

---

## PyTorch module

### `spVIPESmodule`

The underlying `scvi.module.base.BaseModuleClass` that owns all PyTorch
parameters. Access via `model.module` after construction.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    module.spVIPESmodule.spVIPESmodule

.. autoclass:: spVIPES.module.spVIPESmodule.spVIPESmodule
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

**Key attributes:**

| Attribute | Description |
|---|---|
| `module.encoders` | `dict[int, Encoder]` — one shared encoder per group (single-modal), or `dict[(int, str), Encoder]` per (group, modality) in multimodal mode. |
| `module.decoders` | `dict[int, LinearDecoderSPVIPE]` (single-modal) or `dict[(int, str), LinearDecoderSPVIPE]` (multimodal). |
| `module.private_encoders` | `dict[int, Encoder]` — one private encoder per group. |
| `module.n_dimensions_shared` | Shared latent dimensionality. |
| `module.n_dimensions_private` | Private latent dimensionality. |
| `module.is_multimodal` | `True` if built from multimodal data. |
| `module.group_modalities` | `dict[int, list[str]]` — modalities present per group index. |

---

## Neural network components

### `Encoder`

Variational encoder mapping an input feature vector to a Gaussian posterior
over a latent space. Outputs shared and private latent statistics.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    nn.networks.Encoder

.. autoclass:: spVIPES.nn.networks.Encoder
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

### `LinearDecoderSPVIPE`

Linear decoder that accepts both shared and private latents and blends their
reconstructions via a learned mixing weight `px_mixing`.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    nn.networks.LinearDecoderSPVIPE

.. autoclass:: spVIPES.nn.networks.LinearDecoderSPVIPE
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

---

## Internals

These classes are part of the public package namespace but are primarily used
internally or by advanced users extending spVIPES.

### `AnnDataManager`

Vendored from scvi-tools 0.x (removed in scvi-tools 1.x). Manages field
registration, data retrieval, and state validation for a registered AnnData.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    data._manager.AnnDataManager

.. autoclass:: spVIPES.data._manager.AnnDataManager
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

### `ConcatDataLoader`

Multi-group DataLoader used during both training and inference. Cycles smaller
groups to match the largest group size within each epoch.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    dataloaders._concat_dataloader.ConcatDataLoader

.. autoclass:: spVIPES.dataloaders._concat_dataloader.ConcatDataLoader
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

### `AnnDataLoader`

Single-group DataLoader wrapping an `AnnDataManager`.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    dataloaders._ann_dataloader.AnnDataLoader

.. autoclass:: spVIPES.dataloaders._ann_dataloader.AnnDataLoader
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```
