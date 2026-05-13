"""Unit tests for F3 benchmark recommendation gates."""

import sys

from scripts.benchmark_f3_orthogonality import parse_args, recommend_f3_variant


def _row(weight, seed=0, **overrides):
    row = {
        "notes": "ok",
        "seed": seed,
        "orthogonality_weight": weight,
        "orthogonality_within_stratum": 1.0 if weight == 0 else 0.75,
        "reconstruction_loss_per_cell": 100.0 if weight == 0 else 102.0,
        "iLISI": 1.0,
        "kBET": 0.9,
        "cLISI": 2.0 if weight == 0 else 1.95,
        "knn_purity": 0.8,
        "leiden_ari": None,
        "active_dims_shared": 10,
        "train_wall_time_sec": 10.0 if weight == 0 else 10.5,
    }
    row.update(overrides)
    return row


def _rows_for_weights(weights):
    rows = []
    for seed in (0, 1, 2):
        rows.append(_row(0.0, seed=seed))
        for weight in weights:
            rows.append(_row(weight, seed=seed))
    return rows


def test_parse_args_default_weights_include_roadmap_matrix(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_f3_orthogonality.py"])

    cfg = parse_args()

    assert cfg.weights == [0.01, 0.05, 0.1, 0.2]


def test_recommend_f3_variant_passes_when_all_roadmap_gates_pass():
    recommendation = recommend_f3_variant(_rows_for_weights([0.1]))

    assert recommendation["verdict"] == "pass"
    assert recommendation["recommended_weight"] == 0.1
    assert recommendation["candidates"][0]["passes"] is True
    assert recommendation["candidates"][0]["diagnostics"]["wall_overhead"] is not None


def test_recommend_f3_variant_recommends_smallest_passing_weight():
    recommendation = recommend_f3_variant(_rows_for_weights([0.1, 0.2]))

    assert recommendation["verdict"] == "pass"
    assert recommendation["recommended_weight"] == 0.1


def test_recommend_f3_variant_rejects_when_orthogonality_gate_fails():
    rows = _rows_for_weights([0.1])
    for row in rows:
        if row["orthogonality_weight"] == 0.1:
            row["orthogonality_within_stratum"] = 0.9

    recommendation = recommend_f3_variant(rows)

    assert recommendation["verdict"] == "reject"
    assert recommendation["recommended_weight"] is None
    assert "orthogonality reduction <20%" in recommendation["candidates"][0]["failures"]


def test_recommend_f3_variant_iterates_when_baseline_missing():
    recommendation = recommend_f3_variant([_row(0.1, seed=seed) for seed in (0, 1, 2)])

    assert recommendation["verdict"] == "iterate"
    assert recommendation["recommended_weight"] is None


def test_recommend_f3_variant_iterates_when_clisi_missing():
    rows = _rows_for_weights([0.1])
    rows[-1]["cLISI"] = None

    recommendation = recommend_f3_variant(rows)

    assert recommendation["verdict"] == "iterate"
    assert recommendation["recommended_weight"] is None
    assert "missing required F3 gate metrics" in recommendation["reason"]


def test_recommend_f3_variant_iterates_when_seed_coverage_missing():
    rows = [row for row in _rows_for_weights([0.1]) if not (row["seed"] == 2 and row["orthogonality_weight"] == 0.1)]

    recommendation = recommend_f3_variant(rows)

    assert recommendation["verdict"] == "iterate"
    assert recommendation["recommended_weight"] is None
    assert "seed coverage" in recommendation["reason"]


def test_recommend_f3_variant_rejects_when_cross_seed_cv_fails():
    rows = _rows_for_weights([0.1])
    ortho_by_seed = {0: 0.4, 1: 0.75, 2: 1.0}
    for row in rows:
        if row["orthogonality_weight"] == 0.1:
            row["orthogonality_within_stratum"] = ortho_by_seed[row["seed"]]

    recommendation = recommend_f3_variant(rows)

    assert recommendation["verdict"] == "reject"
    assert recommendation["recommended_weight"] is None
    assert any("cross-seed CV" in failure for failure in recommendation["candidates"][0]["failures"])
