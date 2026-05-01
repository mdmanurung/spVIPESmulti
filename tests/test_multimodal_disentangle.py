"""End-to-end test for multimodal disentanglement (P8).

Unlike `test_multigroup_multimodal.py`, this test imports the full spVIPESmulti
package (which transitively pulls in scvi-tools, torch, jax, etc.). It
exercises the loss function path that wires `_compute_disentangle_losses`
into `_loss_multimodal`.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

import anndata as ad

import spVIPESmulti
from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader


def _make_mod(n_obs, n_vars, seed):
    rng = np.random.default_rng(seed)
    X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
    a = ad.AnnData(X=csr_matrix(X))
    a.obs_names = [f"c{i}_{seed}" for i in range(n_obs)]
    a.var_names = [f"g{i}" for i in range(n_vars)]
    return a


def _make_multimodal_adata(n_groups=3, n_per_group=60, n_rna=50, n_prot=20, n_celltypes=3):
    """3-group, 2-modality (rna+protein) AnnData with a shared `cell_types` label."""
    groups = {}
    rng = np.random.default_rng(0)
    for gi in range(n_groups):
        gname = f"g{gi}"
        rna = _make_mod(n_per_group, n_rna, seed=10 * gi)
        prot = _make_mod(n_per_group, n_prot, seed=10 * gi + 1)
        rna.obs_names = [f"{gname}_c{i}" for i in range(n_per_group)]
        prot.obs_names = rna.obs_names
        cts = pd.Categorical(rng.choice([f"ct{j}" for j in range(n_celltypes)], n_per_group))
        # Both modalities must carry the same obs columns or `merge="same"` drops them.
        rna.obs["cell_types"] = cts
        prot.obs["cell_types"] = cts
        groups[gname] = {"rna": rna, "protein": prot}

    prepared = spVIPESmulti.data.prepare_multimodal_adatas(
        groups, modality_likelihoods={"rna": "nb", "protein": "nb"}
    )
    # `prepare_multimodal_adatas`'s outer concat preserves the obs column when
    # both modalities had it; just confirm it survived.
    assert "cell_types" in prepared.obs.columns
    return prepared


def _build_and_run_one_step(prepared, *, disentangle_preset, batch_size=32):
    """Build the model, pull one batch from the dataloader, run loss."""
    spVIPESmulti.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_types",
        modality_likelihoods={"rna": "nb", "protein": "nb"},
    )
    model = spVIPESmulti.model.spVIPESmulti(
        prepared,
        n_hidden=32,
        n_dimensions_shared=8,
        n_dimensions_private=4,
        disentangle_preset=disentangle_preset,
    )

    gi = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]
    scdl = ConcatDataLoader(
        model.adata_manager, indices_list=gi, shuffle=False, batch_size=batch_size, drop_last=False
    )
    tensors_by_group = next(iter(scdl))
    inference_inputs = model.module._get_inference_input(tensors_by_group)
    inference_outputs = model.module.inference(**inference_inputs)
    generative_inputs = model.module._get_generative_input(tensors_by_group, inference_outputs)
    generative_outputs = model.module.generative(**generative_inputs)
    loss_output = model.module.loss(tensors_by_group, inference_outputs, generative_outputs)
    return model, loss_output


class TestMultimodalDisentangle:
    """P8: disentanglement objective in multimodal mode."""

    def test_construction_does_not_raise(self):
        """Multimodal + disentangle_preset='full' should construct cleanly."""
        prepared = _make_multimodal_adata()
        spVIPESmulti.model.spVIPESmulti.setup_anndata(
            prepared, groups_key="groups", label_key="cell_types",
            modality_likelihoods={"rna": "nb", "protein": "nb"},
        )
        model = spVIPESmulti.model.spVIPESmulti(
            prepared,
            n_hidden=32, n_dimensions_shared=8, n_dimensions_private=4,
            disentangle_preset="full",
        )
        assert model.module.is_multimodal is True
        assert model.module.q_group_shared is not None
        assert model.module.q_label_shared is not None
        assert model.module.q_group_private is not None
        assert model.module.q_label_private is not None
        assert model.module.prototypes is not None

    def test_loss_emits_disentangle_metrics(self):
        """All five disentangle/contrastive metrics should appear in extra_metrics."""
        prepared = _make_multimodal_adata()
        _, loss_output = _build_and_run_one_step(prepared, disentangle_preset="full")
        metrics = loss_output.extra_metrics
        for key in (
            "disentangle_group_shared_loss",
            "disentangle_label_shared_loss",
            "disentangle_group_private_loss",
            "disentangle_label_private_loss",
            "contrastive_loss",
        ):
            assert key in metrics, f"Missing metric: {key}. Got: {list(metrics.keys())}"
            v = float(metrics[key])
            assert np.isfinite(v), f"{key} is not finite: {v}"

    def test_disentangle_off_emits_no_metrics(self):
        """preset='off' should not emit disentangle metrics (helper early-exits)."""
        prepared = _make_multimodal_adata()
        _, loss_output = _build_and_run_one_step(prepared, disentangle_preset="off")
        metrics = loss_output.extra_metrics
        for key in (
            "disentangle_group_shared_loss",
            "disentangle_label_shared_loss",
            "disentangle_group_private_loss",
            "disentangle_label_private_loss",
            "contrastive_loss",
        ):
            assert key not in metrics, f"Unexpected metric leaked: {key}"

    def test_loss_is_finite_and_grad_flows(self):
        """The combined loss should be a finite scalar with grad-fn attached."""
        prepared = _make_multimodal_adata()
        _, loss_output = _build_and_run_one_step(prepared, disentangle_preset="full")
        loss = loss_output.loss
        assert torch.is_tensor(loss)
        assert loss.dim() == 0
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        assert loss.requires_grad, "loss should require grad"
        loss.backward()

    def test_label_required_guard_still_fires(self):
        """Without labels, label-using disentangle weights should still raise."""
        # Build adata WITHOUT label_key; setup_anndata won't register labels.
        groups = {}
        rng = np.random.default_rng(0)
        for gi in range(3):
            gname = f"g{gi}"
            rna = _make_mod(60, 50, seed=10 * gi)
            prot = _make_mod(60, 20, seed=10 * gi + 1)
            rna.obs_names = [f"{gname}_c{i}" for i in range(60)]
            prot.obs_names = rna.obs_names
            groups[gname] = {"rna": rna, "protein": prot}
        prepared = spVIPESmulti.data.prepare_multimodal_adatas(
            groups, modality_likelihoods={"rna": "nb", "protein": "nb"}
        )
        spVIPESmulti.model.spVIPESmulti.setup_anndata(
            prepared, groups_key="groups",  # no label_key
            modality_likelihoods={"rna": "nb", "protein": "nb"},
        )
        with pytest.raises(ValueError, match="use_labels=True"):
            spVIPESmulti.model.spVIPESmulti(
                prepared, n_hidden=32, n_dimensions_shared=8, n_dimensions_private=4,
                disentangle_preset="full",
            )
