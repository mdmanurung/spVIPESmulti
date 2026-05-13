# Phase 1 - Package Map

## Runtime Manifest

- Audit date: 2026-05-13
- Environment: `conda run -n spvm env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1`
- Python: 3.11.15
- spVIPESmulti: 1.0.0
- scvi-tools: 1.4.2
- torch: 2.11.0+cu128
- lightning: 2.6.1
- anndata: 0.12.14
- numpy: 2.4.4
- scipy: 1.17.1
- pytest: 9.0.3
- Tooling installed for audit: ruff 0.15.12, mypy 2.1.0, build 1.5.0,
  mdformat 1.0.0, markdownlint 0.48.0, prettier 3.8.3,
  pytest-benchmark 5.2.3, hypothesis 6.152.7, syrupy 5.1.0.

## Package Metadata

- `pyproject.toml`: Hatchling build backend, package `spVIPESmulti`, version
  `1.0.0`, Python `>=3.10`.
- Main dependencies: `anndata>=0.10`, `scvi-tools>=1.0,<2`, `torch`,
  `zuko>=1.0.0`, `scanpy>=1.9.1`, `pandas>=1.5.0`, `numpy>=1.21.0`.
- Optional extras: `dev`, `enrichment`, `doc`, `test`.
- Native source inventory: no `.pyx`, `.pxd`, `.c`, `.cpp`, `.cu`, `.h`, or
  `.hpp` files were found under `src/spVIPESmulti`.

## Public API Surface

- Top-level `src/spVIPESmulti/__init__.py` exports `data`, `interventions`,
  `metrics`, `model`, `module`, `nn`, `pl`, `traversal`, and `utils`.
- Core model export: `src/spVIPESmulti/model/__init__.py` exports
  `spVIPESmulti`.
- Data export: `src/spVIPESmulti/data/__init__.py` exports
  `prepare_adatas`, `prepare_multimodal_adatas`, `AnnDataManager`, and
  `AnnDataManagerValidationCheck`.
- Dataloader export mismatch found: `src/spVIPESmulti/dataloaders/__init__.py`
  imports only `AnnDataLoader` and `ConcatDataLoader`, but `__all__` also lists
  `AnnTorchDataset`, `SupervisedConcatDataLoader`, and `ClassDataLoader`.

## scvi-tools Extension Points

- Model class: `spVIPESmulti.model.spvipesmulti.spVIPESmulti`, subclassing
  `MultiGroupTrainingMixin` and `scvi.model.base.BaseModelClass`.
- Module class: `spVIPESmulti.module.spVIPESmultimodule.spVIPESmultimodule`,
  subclassing `scvi.module.base.BaseModuleClass`.
- Training plan: `SpVIPESmultiTrainingPlan`, subclassing
  `scvi.train.TrainingPlan`.
- Data module: `MultiGroupDataSplitter`, subclassing
  `lightning.pytorch.LightningDataModule`.
- Data manager and loaders: custom vendored `AnnDataManager`, `AnnDataLoader`,
  and `ConcatDataLoader`.

## AnnData Registry Map

`spVIPESmulti.setup_anndata` registers:

- `REGISTRY_KEYS.X_KEY` with `LayerField`.
- `REGISTRY_KEYS.BATCH_KEY` with `CategoricalObsField`.
- `"groups"` with `CategoricalObsField`.
- `"indices"` with `CategoricalObsField`.
- Optional `"labels"`, `"sample"`, `"condition"`, and `"donor"` with
  `CategoricalObsField`.

The core module consumes `X`, `batch`, `groups`, `indices`, and optionally
`labels`. Optional `sample`, `condition`, and `donor` are consumed by posterior
aggregation, counterfactual protocols, or donor disentanglement paths.

## Installed scvi-tools Contracts Checked

- `LossOutput` accepts `loss`, optional reconstruction and KL records,
  optional classification fields, `extra_metrics`, `n_obs_minibatch`, and
  optional summed reconstruction and KL records.
- `BaseModuleClass.forward(self, tensors, ..., compute_loss=True)` returns
  `(inference_outputs, generative_outputs)` or those plus `LossOutput`.
- `TrainingPlan.configure_optimizers(self)`.
- `BaseModelClass._validate_anndata(self, adata=None, copy_if_view=True)`
  validates or transfers a manager for a provided AnnData, but does not itself
  replace every downstream use of `self.adata_manager`.

## Baseline Diagnostics

Logs are under `audits/SCVI_EXTENSION_AUDIT/logs/`.

- `pip install -e ".[dev,test,doc,enrichment]"`: completed.
- `ruff format --check .`: FAIL, 54 files would be reformatted.
- `ruff check .`: FAIL, 755 errors.
- `mdformat --check .`: FAIL, multiple Markdown files not formatted.
- `markdownlint "**/*.md"`: FAIL, line-length and other existing Markdown
  lint errors; markdownlint is now installed in `spvm`.
- `mypy src/spVIPESmulti`: FAIL, 185 errors in 21 files.
- `python -m pytest -p no:cacheprovider --collect-only -q`: PASS,
  285 tests collected before additive audit tests.
- `python -m pytest -p no:cacheprovider -q --maxfail=1 --disable-warnings`:
  PASS, 284 passed, 1 skipped, 213 warnings before additive audit tests.
- `python -m build --sdist --wheel --outdir audits/SCVI_EXTENSION_AUDIT/dist`:
  PASS.
- `python -m sphinx -W -b html docs audits/SCVI_EXTENSION_AUDIT/docs_build_html`:
  FAIL, import error from `zmq`/`libzmq.so.5` requiring
  `GLIBCXX_3.4.29`.

Note: direct `pytest` and `sphinx-build` resolved to stale `~/.local/bin`
entry points under `PYTHONNOUSERSITE=1`. The audit baseline therefore used
`python -m pytest` and `python -m sphinx` to bind execution to the `spvm`
interpreter while preserving the user-site guard.
