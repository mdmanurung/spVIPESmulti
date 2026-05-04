# Plan: Port MrVI label-free differential abundance to spVIPESmulti

## Context

The user wants to bring MrVI's label-free differential abundance (DA) into spVIPESmulti
(branch `claude/integrate-mrvi-methods-ftVpf`). MrVI computes DA in a sample-corrected
latent `u` by aggregating per-cell variational posteriors `q(u | x_n) = N(μ_n, σ_n²)`
into per-sample Gaussian mixtures `q_s`, averaging them into per-group mixtures `q_A`,
and scoring each cell as `r_n = log q_{A₁}(z_n) − log q_{A₂}(z_n)`. The analog in
spVIPESmulti is the post-PoE shared latent `z_shared`. This plan ports the math
faithfully while flagging an architectural caveat that the previous audit missed at
first: spVIPESmulti uses **per-group** shared encoders, so a precondition for
unbiased DA is that an alignment loss is active.

## Audit findings (corrections from prior message)

| Prior claim | Verdict | Correction |
|---|---|---|
| `poe_stats[g]` exposes `logtheta_loc`/`logtheta_scale`/`logtheta_qz` in both single- and multimodal paths | ✓ confirmed | None — verified at `module/spVIPESmultimodule.py:723, 826, 832–836`; multimodal path goes through `_supervised_poe → _label_based_poe`, which still produces these. |
| Surfacing `(μ, σ)` from `_process_batches` is ~30 LOC | ⚠ overstated | Actually ~12–15 LOC. Pattern is identical to existing `latent_shared` collection at `model/spvipesmulti.py:399–401` and the `np.argsort(g_indices)` reorder at `:497–502`. |
| `setup_anndata` registers only groups/label/batch — adding `sample_key` is small | ✓ confirmed | `model/spvipesmulti.py:244–304`. No data-prep helper depends on a sample column; addition is model-side metadata only. |
| `z_shared` is only group-invariant when `disentangle_group_shared > 0` or `use_jeffreys_integ=True` | ⚠ understated | Per-group encoders are **independent** modules (`module/spVIPESmultimodule.py` encoder dict keyed by group). Even cells with identical biology will land in *different regions* of `z_shared` purely from encoder-weight differences unless an alignment loss drives the encoders toward equivalence. This is a stronger constraint than a "soft caveat" — it must be a documented precondition. Jeffreys is a softer regularizer than adversarial GRL but both work. |
| `MixtureSameFamily` approach matches existing PyTorch usage | ✓ confirmed | Codebase uses `torch.distributions.Normal`; mixture API is compatible. |
| `get_latent_representation` returns per-group shared latents with a `_reordered` view in original obs order | ✓ confirmed | `_format_results` at `model/spvipesmulti.py:484–525` returns `output["shared_reordered"][g]`. |

**Other surprise from audit:** `_process_all_cells_with_cycling` at
`model/spvipesmulti.py:443–482` is dead code (never called from
`get_latent_representation`). Don't extend it.

## Approach

Implement the MrVI DA workflow as three new model methods that consume `z_shared`'s
posterior parameters. No changes to module forward, training loop, or losses.

### Files to modify

1. **`src/spVIPESmulti/model/spvipesmulti.py`**
   - Extend `_process_batches` (`:366–442`): collect `outputs["poe_stats"][g]["logtheta_loc"]` and `outputs["poe_stats"][g]["logtheta_scale"]` parallel to the existing `latent_shared[g]` list.
   - Extend `_format_results` (`:484–525`): add `output["shared_loc"]`, `output["shared_scale"]`, plus their `_reordered` variants using the same `np.argsort(g_indices)` pattern at `:497–502`.
   - Add `setup_anndata` parameter `sample_key: str` **required for DA** (`:244`); register `CategoricalObsField("samples", sample_key)` after the existing fields at `:280`. Keep optional at registration time (don't break existing setup calls), but raise a clear `ValueError` from `differential_abundance` if `sample_key` was not registered.
   - Add three new public methods on the model:
     - `get_shared_posterior(group_indices_list, ...)` — thin wrapper that returns the new `(loc, scale)` arrays in original obs order.
     - `get_aggregated_posterior(adata=None, sample=None, group_indices_list=...)` — builds `MixtureSameFamily(Categorical, Independent(Normal, 1))` over the cells of one sample (or all cells if `sample=None`).
     - `differential_abundance(sample_cov_keys, sample_subset=None, group_indices_list=..., compute_log_enrichment=False)` — for each covariate value, averages per-sample mixtures into a per-group mixture; returns an `xarray.Dataset` with dims `(cells, covariate)` matching MrVI's API. Add `xarray` to `pyproject.toml` deps.
   - Add a precondition check at the top of `differential_abundance`: warn (not error) if neither `disentangle_group_shared_weight > 0` nor `use_jeffreys_integ=True` was set on the model — with text explaining encoder-induced placement bias.

2. **`src/spVIPESmulti/__init__.py`** — re-export nothing new (methods live on the model class).

3. **`tests/test_differential_abundance.py`** (new) — unit tests covering:
   - Synthetic 2-group AnnData with a known sample column → DA score sign matches the simulated covariate shift.
   - Per-cell output length equals total cells.
   - Sample-subset filtering reduces the support correctly.
   - Precondition warning fires when both alignment losses are off.

### Reused functions / utilities

- `_process_batches` and `_format_results` (`model/spvipesmulti.py:366`, `:484`) — extend in place; reuse the `original_indices` reorder pattern verbatim.
- `ConcatDataLoader` (`model/spvipesmulti.py:355`) — reuse for posterior extraction.
- `outputs["poe_stats"][g]["logtheta_loc"|"logtheta_scale"|"logtheta_qz"]` (`module/spVIPESmultimodule.py:832–836`) — already produced by both single- and multimodal paths via `_label_based_poe`; nothing to wire up inside the module.
- `torch.distributions.{Normal, Categorical, Independent, MixtureSameFamily}` — `Normal` is already imported at `module/spVIPESmultimodule.py:11`; the others are pure stdlib of `torch.distributions`.

### Out of scope

- `differential_expression`, `get_outlier_cell_sample_pairs`, `get_local_sample_distances` — defer to a follow-up; DE in particular requires counterfactual generation through the per-group decoders (`module/spVIPESmultimodule.py:841+`), which is materially more invasive than DA.
- KDE fallback — withdrawn; it's the σ→0 limit of the same Gaussian mixture, not a separate option.
- Modifying inference/training. All changes are post-hoc on the trained model.

## Verification

End-to-end test on the existing CINEMA-OT vignette (`docs/notebooks/cinemaot_nf_vignette.ipynb`):

1. Train a model with `disentangle_preset="full"` (alignment loss active) on the CINEMA-OT data with a synthetic per-cell `sample` column (e.g., random partition of each group into 4 pseudo-samples).
2. Call `model.differential_abundance(sample_cov_keys=["groups"])`.
3. Plot per-cell DA scores on the shared UMAP (`spVIPESmulti.pl.umap_shared`) — expect a smooth gradient that aligns with regions of known group-specific abundance.
4. Repeat with `disentangle_preset="off"` and confirm the precondition warning fires; informally, expect the DA map to look noisier / dominated by group-encoder placement.
5. `pytest tests/test_differential_abundance.py -v` passes.
6. `python scripts/smoke_vignettes.py` — confirm no regressions in existing vignettes.

## Resolved decisions

- **Unit of comparison**: per-sample aggregation, then per-group average (matches MrVI). `sample_key` registration is required for DA.
- **Return type**: `xarray.Dataset` matching MrVI's API. `xarray` will be added to `pyproject.toml` runtime deps.
