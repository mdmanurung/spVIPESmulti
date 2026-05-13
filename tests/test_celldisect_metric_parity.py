"""Tests for F10a CellDISECT-style internal metric helpers."""

import csv

import numpy as np

from spVIPESmulti.interventions import metrics as m


def test_counterfactual_pearson_matches_numpy():
    x_pred = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]])
    x_true = np.array([[1.0, 2.5, 3.5], [2.0, 3.5, 4.5]])
    expected = np.corrcoef(x_pred.mean(axis=0), x_true.mean(axis=0))[0, 1]
    assert np.isclose(m.counterfactual_pearson(x_pred, x_true), expected)


def test_delta_pearson_matches_reference():
    x_ctrl = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    x_true = np.array([[2.0, 1.0, 4.0], [2.0, 1.0, 4.0]])
    x_pred = np.array([[3.0, 1.0, 7.0], [3.0, 1.0, 7.0]])
    expected = np.corrcoef((x_pred - x_ctrl).mean(axis=0), (x_true - x_ctrl).mean(axis=0))[0, 1]
    assert np.isclose(m.delta_pearson(x_ctrl, x_true, x_pred), expected)


def test_top_de_selection_uses_abs_true_delta():
    x_ctrl = np.zeros((2, 5))
    x_true = np.array([[0.0, -5.0, 1.0, 3.0, 2.0], [0.0, -5.0, 1.0, 3.0, 2.0]])
    top = m.select_top_de_genes(x_ctrl, x_true, n_top=3)
    assert top.tolist() == [1, 3, 4]


def test_wasserstein_aggregation_shape_and_finite():
    x_pred = np.array([[0.0, 1.0], [2.0, 3.0]])
    x_true = np.array([[1.0, 1.0], [3.0, 5.0]])
    out = m.wasserstein_gene_marginals(x_pred, x_true, top_idx=[1])
    assert out["per_gene"].shape == (2,)
    assert np.isfinite(out["per_gene"]).all()
    assert np.isfinite(out["mean_all"])
    assert np.isfinite(out["mean_top"])


def test_cag_and_mig_helpers_bounded_finite():
    labels = np.array(["a", "a", "b", "b"])
    z_i = np.array([[0.0, 0.1], [0.1, 0.0], [5.0, 5.1], [5.1, 5.0]])
    z_minus = np.zeros_like(z_i)
    cag = m.classifier_accuracy_gap(z_i, z_minus, labels)
    assert 0.0 <= cag["acc_i"] <= 1.0
    assert 0.0 <= cag["acc_minus_i"] <= 1.0
    assert -1.0 <= cag["cag"] <= 1.0

    mig = m.mig_scores(z_i, labels)
    assert set(mig) == {"maxMIG", "concatMIG", "minMIG"}
    assert all(0.0 <= value <= 1.0 for value in mig.values())


def test_artifact_schema_writer_records_skipped_external_rows(tmp_path):
    path = m.write_artifact_schema(
        tmp_path / "metrics.csv",
        [
            {
                "model": "CellDISECT",
                "split": "split_CD14",
                "metric": "counterfactual_pearson",
                "value": "",
                "status": "skipped",
                "notes": "external install unavailable",
            }
        ],
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_id"] == ""
    assert rows[0]["dataset"] == ""
    assert rows[0]["model"] == "CellDISECT"
    assert rows[0]["status"] == "skipped"
