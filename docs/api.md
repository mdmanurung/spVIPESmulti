# API Reference

spVIPESmulti v1.0.0 — shared-private Variational Inference with Product of Experts and Supervision.

**Authors:** Mikhael Dito Manurung · Claudio Novella Rausell · D.J.M. Peters · A. Mahfouz

## Quick reference

| Symbol | Kind | Import path |
|---|---|---|
| `spVIPESmulti` | Model class | `spVIPESmulti.model.spVIPESmulti` |
| `setup_anndata` | Class method | `spVIPESmulti.model.spVIPESmulti.setup_anndata` |
| `train` | Instance method | `model.train(...)` |
| `get_latent_representation` | Instance method | `model.get_latent_representation(...)` |
| `get_loadings` | Instance method | `model.get_loadings()` |
| `prepare_adatas` | Function | `spVIPESmulti.data.prepare_adatas` |
| `prepare_multimodal_adatas` | Function | `spVIPESmulti.data.prepare_multimodal_adatas` |
| `store_latents` | Function | `spVIPESmulti.utils.store_latents` |
| `add_latent_dims_to_obs` | Function | `spVIPESmulti.utils.add_latent_dims_to_obs` |
| `compute_shared_umap` | Function | `spVIPESmulti.utils.compute_shared_umap` |
| `compute_private_umaps` | Function | `spVIPESmulti.utils.compute_private_umaps` |
| `get_top_genes` | Function | `spVIPESmulti.utils.get_top_genes` |
| `score_cells_on_factor` | Function | `spVIPESmulti.utils.score_cells_on_factor` |
| `heatmap_loadings` | Function | `spVIPESmulti.pl.heatmap_loadings` |
| `umap_shared` | Function | `spVIPESmulti.pl.umap_shared` |
| `umap_private` | Function | `spVIPESmulti.pl.umap_private` |
| `factor_violin` | Function | `spVIPESmulti.pl.factor_violin` |
| `training_curves` | Function | `spVIPESmulti.pl.training_curves` |
| `loadings_dotplot` | Function | `spVIPESmulti.pl.loadings_dotplot` |
| `spVIPESmultimodule` | PyTorch module | `spVIPESmulti.module.spVIPESmultimodule` |
| `Encoder` | Neural network | `spVIPESmulti.nn.Encoder` |
| `LinearDecoderSPVIPE` | Neural network | `spVIPESmulti.nn.LinearDecoderSPVIPE` |
| `ConcatDataLoader` | DataLoader | `spVIPESmulti.dataloaders.ConcatDataLoader` |

---

## Model

### `spVIPESmulti`

The main user-facing model class. Extends `scvi.model.base.BaseModelClass` and
`MultiGroupTrainingMixin`. Call `setup_anndata` first, then construct the model,
then call `train`.

```python
import spVIPESmulti

adata = spVIPESmulti.data.prepare_adatas({"ctrl": adata_ctrl, "treat": adata_treat})
spVIPESmulti.model.spVIPESmulti.setup_anndata(adata, groups_key="groups", label_key="cell_type")
model = spVIPESmulti.model.spVIPESmulti(adata)
model.train(group_indices_list=group_indices_list, max_epochs=100)
latents = model.get_latent_representation(group_indices_list=group_indices_list, batch_size=512)
```

```{eval-rst}
.. currentmodule:: spVIPESmulti

.. autosummary::
    :toctree: generated
    :template: class.rst

    model.spvipesmulti.spVIPESmulti

.. autoclass:: spVIPESmulti.model.spvipesmulti.spVIPESmulti
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
| `**model_kwargs` | | | Forwarded to `spVIPESmultimodule`. |

> **Tip:** Individual weight overrides stack on top of a preset.
> `spVIPESmulti(adata, disentangle_preset="full", contrastive_weight=0.0)` enables
> all classifiers but turns off the InfoNCE term.

---

### `setup_anndata`

```python
spVIPESmulti.model.spVIPESmulti.setup_anndata(
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
| `label_key` omitted | Unsupervised PoE | N ≥ 2 |

For multimodal data or N > 2 groups, label-based PoE is recommended.

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
from spVIPESmulti.data import prepare_adatas

mdata = prepare_adatas(adatas, layers=None)
```

Concatenates multiple single-modality AnnData objects into one AnnData suitable
for spVIPESmulti. At least 2 groups are required.

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

.. autofunction:: spVIPESmulti.data.prepare_adatas.prepare_adatas
```

---

### `prepare_multimodal_adatas`

```python
from spVIPESmulti.data import prepare_multimodal_adatas

mdata = prepare_multimodal_adatas(adatas, modality_likelihoods=None)
```

Concatenates a nested `{group: {modality: AnnData}}` dict into one AnnData
for multimodal spVIPESmulti runs. At least 2 groups are required.

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
mdata = spVIPESmulti.data.prepare_multimodal_adatas(
    adatas_dict,
    modality_likelihoods={"rna": "nb", "protein": "nb"},
)
spVIPESmulti.model.spVIPESmulti.setup_anndata(
    mdata,
    groups_key="tissue",
    label_key="cell_types",
    modality_likelihoods={"rna": "nb", "protein": "nb"},
)
group_indices_list = [list(map(int, g)) for g in mdata.uns["groups_obs_indices"]]
model = spVIPESmulti.model.spVIPESmulti(mdata, use_nf_prior=True, nf_type="NSF")
model.train(group_indices_list=group_indices_list, max_epochs=100, batch_size=512)
```

```{eval-rst}
.. autosummary::
    :toctree: generated

    data.prepare_adatas.prepare_multimodal_adatas

.. autofunction:: spVIPESmulti.data.prepare_adatas.prepare_multimodal_adatas
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
model = spVIPESmulti.model.spVIPESmulti(
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

### `spVIPESmultimodule`

The underlying `scvi.module.base.BaseModuleClass` that owns all PyTorch
parameters. Access via `model.module` after construction.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    module.spVIPESmultimodule.spVIPESmultimodule

.. autoclass:: spVIPESmulti.module.spVIPESmultimodule.spVIPESmultimodule
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

.. autoclass:: spVIPESmulti.nn.networks.Encoder
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

.. autoclass:: spVIPESmulti.nn.networks.LinearDecoderSPVIPE
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

---

## Utilities

The `spVIPESmulti.utils` module provides ready-to-use post-training helpers that
consolidate the manual boilerplate repeated in every tutorial notebook.

```python
import spVIPESmulti
```

### `store_latents`

```python
spVIPESmulti.utils.store_latents(
    adata,
    latents,
    group_indices_list,
    obsm_prefix="X_spVIPESmulti",
)
```

Stitches per-group latent arrays (returned by `get_latent_representation`)
back into `adata.obsm` using the original cell order. Handles all latent
types returned by the model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | — | The combined AnnData passed to `setup_anndata`. |
| `latents` | `dict` | — | Dict returned by `model.get_latent_representation(...)`. |
| `group_indices_list` | `list[ndarray]` | — | Cell index arrays, one per group (same as used for training). |
| `obsm_prefix` | `str` | `"X_spVIPESmulti"` | Prefix for keys written to `adata.obsm`. |

**Keys written to `adata.obsm`:**

| Key | Description |
|---|---|
| `{prefix}_shared` | Shared PoE latent for all cells. |
| `{prefix}_private_g{i}` | Per-group private latent (single-modal). |
| `{prefix}_private_{gi}_{modality}` | Per-(group, modality) private latent (multimodal). |

```python
latents = model.get_latent_representation(group_indices_list, batch_size=512)
spVIPESmulti.utils.store_latents(adata, latents, group_indices_list)
# adata.obsm["X_spVIPESmulti_shared"] is now populated
```

---

### `add_latent_dims_to_obs`

```python
spVIPESmulti.utils.add_latent_dims_to_obs(
    adata,
    obsm_key,
    prefix=None,
    max_dims=None,
)
```

Copies latent dimensions from `adata.obsm` into `adata.obs` columns so they
can be used directly as `color=` arguments in scanpy plots.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | — | AnnData containing `obsm_key`. |
| `obsm_key` | `str` | — | Key in `adata.obsm` to read from (e.g. `"X_spVIPESmulti_private_g0"`). |
| `prefix` | `str or None` | `None` | Column name prefix. Defaults to `obsm_key` with leading `"X_"` stripped. |
| `max_dims` | `int or None` | `None` | Maximum number of dimensions to copy. `None` copies all. |

New obs columns are named `{prefix}_0`, `{prefix}_1`, …

```python
spVIPESmulti.utils.add_latent_dims_to_obs(adata_g0, "X_spVIPESmulti_private_g0", max_dims=5)
sc.pl.violin(adata_g0, "spVIPESmulti_private_g0_1", groupby="cell_type")
```

---

### `compute_shared_umap`

```python
spVIPESmulti.utils.compute_shared_umap(
    adata,
    obsm_key="X_spVIPESmulti_shared",
    n_neighbors=15,
    min_dist=0.3,
    umap_key="X_umap_spvipesmulti_shared",
)
```

Runs `scanpy.pp.neighbors` + `scanpy.tl.umap` on the shared latent and stores
the result under a named key, without overwriting any existing `X_umap`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | — | AnnData with `obsm_key` populated by `store_latents`. |
| `obsm_key` | `str` | `"X_spVIPESmulti_shared"` | Source key in `adata.obsm`. |
| `n_neighbors` | `int` | `15` | Number of neighbours for the kNN graph. |
| `min_dist` | `float` | `0.3` | UMAP `min_dist` parameter. |
| `umap_key` | `str` | `"X_umap_spvipesmulti_shared"` | Destination key in `adata.obsm`. |

```python
spVIPESmulti.utils.compute_shared_umap(adata)
spVIPESmulti.pl.umap_shared(adata, color="cell_type")
```

---

### `compute_private_umaps`

```python
spVIPESmulti.utils.compute_private_umaps(
    adatas_per_group,
    obsm_key="X_spVIPESmulti_private",
    n_neighbors=15,
    min_dist=0.3,
    umap_key="X_umap_spvipesmulti_private",
)
```

Computes a UMAP embedding of each group's private latent. Accepts a
`{group_name: AnnData}` mapping and updates each AnnData in-place.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adatas_per_group` | `dict[str, AnnData]` | — | Mapping from group name to per-group AnnData. |
| `obsm_key` | `str` | `"X_spVIPESmulti_private"` | Source key in each group's `obsm`. |
| `n_neighbors` | `int` | `15` | Number of neighbours. |
| `min_dist` | `float` | `0.3` | UMAP `min_dist`. |
| `umap_key` | `str` | `"X_umap_spvipesmulti_private"` | Destination key in each group's `obsm`. |

```python
adatas = {"day0": adata_g0, "day3": adata_g1}
spVIPESmulti.utils.compute_private_umaps(adatas)
spVIPESmulti.pl.umap_private(adatas, color="cell_type")
```

---

### `get_top_genes`

```python
spVIPESmulti.utils.get_top_genes(
    loadings_df=None,
    *,
    model=None,
    group_idx=0,
    latent_type="shared",
    n_top=10,
    signed=True,
)
```

Ranks genes by loading magnitude per latent dimension. Can fetch loadings from
a pre-computed DataFrame or directly from a fitted model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loadings_df` | `pd.DataFrame or None` | `None` | Pre-computed loadings (shape `(n_genes, n_dims)`). If `None`, `model` must be provided. |
| `model` | object | `None` | Fitted spVIPESmulti model used to fetch loadings automatically. |
| `group_idx` | `int` | `0` | Group index for `model.get_loadings()`. |
| `latent_type` | `str` | `"shared"` | `"shared"` or `"private"`. |
| `n_top` | `int` | `10` | Number of top genes per dimension. |
| `signed` | `bool` | `True` | If `True`, return top positive and top negative genes separately. If `False`, rank by absolute value. |

**Returns** `pd.DataFrame` with columns:
- `dim` — dimension name (e.g. `"Z_shared_0"`)
- `pos_genes`, `neg_genes` — when `signed=True`
- `top_genes` — when `signed=False`

```python
top = spVIPESmulti.utils.get_top_genes(model=model, n_top=5)
print(top[["dim", "pos_genes"]].to_string(index=False))
```

---

### `score_cells_on_factor`

```python
spVIPESmulti.utils.score_cells_on_factor(
    adata,
    dim_idx,
    obsm_key,
    col_name=None,
)
```

Writes a single latent dimension from `adata.obsm` into `adata.obs`. Useful
when you only want to colour plots by one specific factor.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | — | AnnData object. |
| `dim_idx` | `int` | — | Zero-based index of the latent dimension to extract. |
| `obsm_key` | `str` | — | Key in `adata.obsm` to read from. |
| `col_name` | `str or None` | `None` | Obs column name. Defaults to `"{obsm_key_stripped}_{dim_idx}"`. |

```python
spVIPESmulti.utils.score_cells_on_factor(adata_g0, dim_idx=2, obsm_key="X_spVIPESmulti_private_g0")
sc.pl.violin(adata_g0, "spVIPESmulti_private_g0_2", groupby="cell_type")
```

---

## Plotting

The `spVIPESmulti.pl` module provides standalone plotting functions that accept
pre-computed arrays or AnnData objects. All functions can be used independently
of the training workflow.

```python
import spVIPESmulti
```

### `heatmap_loadings`

```python
spVIPESmulti.pl.heatmap_loadings(
    loadings_df=None,
    *,
    model=None,
    group_idx=0,
    latent_type="shared",
    n_top=5,
    figsize=None,
    ax=None,
)
```

Draws a seaborn heatmap of the top-`n_top` genes (by absolute loading) for
every latent dimension. Requires `seaborn`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loadings_df` | `pd.DataFrame or None` | `None` | Pre-computed loadings. If `None`, `model` must be provided. |
| `model` | object | `None` | Fitted spVIPESmulti model. |
| `group_idx` | `int` | `0` | Group index for `model.get_loadings()`. |
| `latent_type` | `str` | `"shared"` | `"shared"` or `"private"`. |
| `n_top` | `int` | `5` | Number of top genes per dimension. |
| `figsize` | `tuple or None` | auto | Figure size. |
| `ax` | `Axes or None` | `None` | Existing matplotlib axes to draw on. |

Returns the `Axes` object for further customisation.

```python
ax = spVIPESmulti.pl.heatmap_loadings(model=model, n_top=10)
ax.figure.savefig("loadings.pdf")
```

---

### `umap_shared`

```python
spVIPESmulti.pl.umap_shared(adata, color, basis="X_umap_spvipesmulti_shared", **kwargs)
```

Thin wrapper around `scanpy.pl.embedding` that defaults `basis` to the key
written by `compute_shared_umap`. All extra keyword arguments are forwarded.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | — | AnnData with the shared UMAP in `obsm[basis]`. |
| `color` | `str or list[str]` | — | `adata.obs` key(s) or gene name(s) to colour by. |
| `basis` | `str` | `"X_umap_spvipesmulti_shared"` | Key in `adata.obsm` with 2-D coordinates. |
| `**kwargs` | | | Forwarded to `scanpy.pl.embedding`. |

```python
spVIPESmulti.pl.umap_shared(adata, color=["cell_type", "groups"])
```

---

### `umap_private`

```python
spVIPESmulti.pl.umap_private(
    adatas_per_group,
    color,
    basis="X_umap_spvipesmulti_private",
    ncols=3,
    figsize=None,
    **kwargs,
)
```

Creates a grid of per-group private UMAP panels. Returns the
`matplotlib.figure.Figure`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adatas_per_group` | `dict[str, AnnData]` | — | Mapping from group name to AnnData (built by `compute_private_umaps`). |
| `color` | `str` | — | Single `adata.obs` key or gene name to colour all panels. |
| `basis` | `str` | `"X_umap_spvipesmulti_private"` | Key in each group's `obsm`. |
| `ncols` | `int` | `3` | Number of columns in the grid. |
| `figsize` | `tuple or None` | auto | Total figure size (defaults to `(5*ncols, 4*nrows)`). |
| `**kwargs` | | | Forwarded to `scanpy.pl.embedding`. |

```python
fig = spVIPESmulti.pl.umap_private(adatas, color="cell_type")
fig.savefig("private_umaps.pdf")
```

---

### `factor_violin`

```python
spVIPESmulti.pl.factor_violin(
    adata,
    dim_idx,
    groupby,
    obsm_key,
    latent_type="private",
    ax=None,
    **kwargs,
)
```

Violin plot of a single latent factor stratified by a cell metadata column.
If the factor column is not already in `adata.obs`, it is added automatically
via `score_cells_on_factor`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | — | AnnData object. |
| `dim_idx` | `int` | — | Zero-based index of the latent dimension. |
| `groupby` | `str` | — | Column in `adata.obs` to group by. |
| `obsm_key` | `str` | — | Key in `adata.obsm` containing the latent matrix. |
| `latent_type` | `str` | `"private"` | Used only to construct the default obs column name. |
| `ax` | `Axes or None` | `None` | Existing axes to draw on. |
| `**kwargs` | | | Forwarded to `scanpy.pl.violin`. |

```python
spVIPESmulti.pl.factor_violin(
    adata_g0, dim_idx=1, groupby="cell_type", obsm_key="X_spVIPESmulti_private_g0"
)
```

---

### `training_curves`

```python
spVIPESmulti.pl.training_curves(model, metrics=None, figsize=None)
```

Multi-panel plot of training history metrics. One sub-panel per metric.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | object | — | Fitted spVIPESmulti model with a `history` attribute. |
| `metrics` | `list[str] or None` | `None` | Keys from `model.history` to plot. `None` plots all available. |
| `figsize` | `tuple or None` | auto | Total figure size. |

Returns a `matplotlib.figure.Figure`.

```python
fig = spVIPESmulti.pl.training_curves(model)
fig.savefig("training.pdf")
```

---

### `loadings_dotplot`

```python
spVIPESmulti.pl.loadings_dotplot(
    adata,
    dims,
    groupby,
    *,
    loadings_df=None,
    model=None,
    group_idx=0,
    latent_type="shared",
    n_top=5,
    **kwargs,
)
```

Draws a `scanpy.pl.dotplot` of the top genes for selected latent dimensions.
For each requested dimension, the `n_top` genes with the largest absolute
loadings are collected and passed as `var_names`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | — | AnnData whose `var_names` contain the genes. |
| `dims` | `list[int] or list[str]` | — | Dimensions to visualise (integer indices or column name strings). |
| `groupby` | `str` | — | Column in `adata.obs` for the y-axis grouping. |
| `loadings_df` | `pd.DataFrame or None` | `None` | Pre-computed loadings. If `None`, `model` must be provided. |
| `model` | object | `None` | Fitted spVIPESmulti model. |
| `group_idx` | `int` | `0` | Group index for `model.get_loadings()`. |
| `latent_type` | `str` | `"shared"` | `"shared"` or `"private"`. |
| `n_top` | `int` | `5` | Number of top genes per dimension. |
| `**kwargs` | | | Forwarded to `scanpy.pl.dotplot`. |

```python
spVIPESmulti.pl.loadings_dotplot(adata, dims=[0, 2, 4], groupby="cell_type", model=model)
```

---

## Internals

These classes are part of the public package namespace but are primarily used
internally or by advanced users extending spVIPESmulti.

### `AnnDataManager`

Vendored from scvi-tools 0.x (removed in scvi-tools 1.x). Manages field
registration, data retrieval, and state validation for a registered AnnData.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    data._manager.AnnDataManager

.. autoclass:: spVIPESmulti.data._manager.AnnDataManager
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

.. autoclass:: spVIPESmulti.dataloaders._concat_dataloader.ConcatDataLoader
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

.. autoclass:: spVIPESmulti.dataloaders._ann_dataloader.AnnDataLoader
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```
