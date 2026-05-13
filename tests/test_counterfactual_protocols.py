"""Tests for F5 donor/condition-aware counterfactual protocols."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import spVIPESmulti as sv


def _make_protocol_adata(include_condition=True, include_donor=True, include_label=True):
    rng = np.random.default_rng(31)
    groups = {}
    specs = {
        "a_target": {
            "cell_type": ["T", "B", "T", "B", "T", "B", "T", "B"],
            "condition": ["ctrl", "ctrl", "ctrl", "stim", "stim", "ctrl", "stim", "ctrl"],
            "donor": ["d0", "d1", "d2", "d0", "d1", "d2", "d3", "d4"],
            "timepoint": ["t0", "t0", "t1", "t1", "t0", "t1", "t0", "t1"],
        },
        "b_source": {
            "cell_type": ["T", "B", "T", "B", "T", "B", "T", "B"],
            "condition": ["stim", "stim", "stim", "ctrl", "ctrl", "stim", "ctrl", "stim"],
            "donor": ["d0", "d9", "d9", "d9", "d9", "d8", "d3", "d4"],
            "timepoint": ["t0", "t0", "t1", "t1", "t0", "t1", "t0", "t1"],
        },
    }
    for gi, (name, obs) in enumerate(specs.items()):
        x = rng.poisson(4 + gi, size=(8, 10)).astype(np.float32)
        a = ad.AnnData(X=csr_matrix(x))
        a.obs_names = [f"{name}_c{i}" for i in range(8)]
        a.var_names = [f"g{i}" for i in range(10)]
        if include_label:
            a.obs["cell_type"] = obs["cell_type"]
        if include_condition:
            a.obs["condition"] = obs["condition"]
        if include_donor:
            a.obs["donor"] = obs["donor"]
        a.obs["timepoint"] = obs["timepoint"]
        groups[name] = a

    prepared = sv.data.prepare_adatas(groups)
    sv.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type" if include_label else None,
        condition_key="condition" if include_condition else None,
        donor_key="donor" if include_donor else None,
    )
    return prepared


def _model(prepared):
    return sv.model.spVIPESmulti(
        prepared,
        n_hidden=16,
        n_dimensions_shared=4,
        n_dimensions_private=3,
        dropout_rate=0.0,
        disentangle_preset="off",
    )


def test_private_swap_unmatched_shape_and_metadata():
    import spVIPESmulti.interventions as svi

    prepared = _make_protocol_adata()
    model = _model(prepared)
    cells = prepared.uns["groups_obs_indices"][0][:3]

    result = svi.private_swap_unmatched(
        model,
        prepared,
        cells=cells,
        group_idx=0,
        source_group_idx=1,
        seed=7,
        include_uncertainty=False,
    )

    assert result.X.shape == (3, len(model.module.groups_var_indices[0]))
    assert result.info["protocol"] == "private_swap_unmatched"
    assert result.info["z_shared_source"] == "target"
    assert result.info["target_obs_indices"].tolist() == list(cells)
    assert result.info["source_obs_indices"].shape == (3,)


def test_private_swap_label_matched_uses_matching_source_labels():
    import spVIPESmulti.interventions as svi

    prepared = _make_protocol_adata()
    model = _model(prepared)
    cells = prepared.uns["groups_obs_indices"][0][:4]

    result = svi.private_swap_label_matched(
        model,
        prepared,
        cells=cells,
        group_idx=0,
        source_group_idx=1,
        seed=1,
        include_uncertainty=False,
    )

    labels = prepared.obs["cell_type"].astype(str).to_numpy()
    assert result.info["protocol"] == "private_swap_label_matched"
    assert np.array_equal(labels[result.info["source_obs_indices"]], labels[result.info["target_obs_indices"]])


def test_private_swap_label_matched_raises_without_label_key():
    import spVIPESmulti.interventions as svi

    prepared = _make_protocol_adata(include_label=False)
    model = _model(prepared)

    with pytest.raises(ValueError, match="label_key"):
        svi.private_swap_label_matched(
            model,
            prepared,
            cells=prepared.uns["groups_obs_indices"][0][:2],
            group_idx=0,
            source_group_idx=1,
        )


def test_private_swap_stratified_reports_deterministic_fallback_counts():
    import spVIPESmulti.interventions as svi

    prepared = _make_protocol_adata()
    model = _model(prepared)
    cells = prepared.uns["groups_obs_indices"][0][:3]

    first = svi.private_swap_stratified(
        model,
        prepared,
        cells=cells,
        group_idx=0,
        source_group_idx=1,
        timepoint_key="timepoint",
        seed=4,
        include_uncertainty=False,
    )
    second = svi.private_swap_stratified(
        model,
        prepared,
        cells=cells,
        group_idx=0,
        source_group_idx=1,
        timepoint_key="timepoint",
        seed=4,
        include_uncertainty=False,
    )

    assert first.info["fallback_counts"] == {
        "label_donor_timepoint": 1,
        "label_donor": 0,
        "label": 2,
        "unmatched": 0,
    }
    assert first.info["match_tiers"] == ["label_donor_timepoint", "label", "label"]
    assert np.array_equal(first.info["source_obs_indices"], second.info["source_obs_indices"])


def test_donor_condition_shift_requires_registered_keys():
    import spVIPESmulti.interventions as svi

    no_condition = _make_protocol_adata(include_condition=False)
    with pytest.raises(ValueError, match="condition_key"):
        svi.donor_condition_shift(
            _model(no_condition),
            no_condition,
            cells=no_condition.uns["groups_obs_indices"][0][:1],
            condition_from="ctrl",
            condition_to="stim",
            group_idx=0,
        )

    no_donor = _make_protocol_adata(include_donor=False)
    with pytest.raises(ValueError, match="donor_key"):
        svi.donor_condition_shift(
            _model(no_donor),
            no_donor,
            cells=no_donor.uns["groups_obs_indices"][0][:1],
            condition_from="ctrl",
            condition_to="stim",
            group_idx=0,
        )


def test_donor_condition_shift_uses_within_donor_direction():
    import spVIPESmulti.interventions as svi

    prepared = _make_protocol_adata()
    model = _model(prepared)
    cells = prepared.uns["groups_obs_indices"][0][:1]
    encoded = svi.encode_cells(model, prepared)
    full_shared = np.zeros((prepared.n_obs, model.module.n_dimensions_shared), dtype=np.float32)
    for group, arr in encoded["shared"].items():
        full_shared[encoded["obs_indices"][group]] = arr

    condition = prepared.obs["condition"].astype(str).to_numpy()
    donor = prepared.obs["donor"].astype(str).to_numpy()
    donor_mask = donor == donor[int(cells[0])]
    expected = full_shared[(condition == "stim") & donor_mask].mean(axis=0) - full_shared[
        (condition == "ctrl") & donor_mask
    ].mean(axis=0)

    result = svi.donor_condition_shift(
        model,
        prepared,
        cells=cells,
        condition_from="ctrl",
        condition_to="stim",
        group_idx=0,
        include_uncertainty=False,
    )

    assert result.info["fallback_counts"] == {"donor_specific": 1, "global": 0}
    assert np.allclose(result.info["direction"][0], expected)


def test_existing_transfer_condition_still_available():
    import spVIPESmulti.interventions as svi

    assert hasattr(svi, "encode_cells")
    assert hasattr(svi, "decode_counterfactual")
    assert hasattr(svi, "predict_counterfactual")
    assert hasattr(svi, "transfer_condition")
