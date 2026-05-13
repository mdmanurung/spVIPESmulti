"""Unit tests for F3 benchmark recommendation gates."""

from scripts.benchmark_f3_orthogonality import recommend_f3_variant


def _row(weight, **overrides):
    row = {
        "notes": "ok",
        "orthogonality_weight": weight,
        "orthogonality_within_stratum": 1.0 if weight == 0 else 0.75,
        "reconstruction_loss_per_cell": 100.0,
        "iLISI": 1.0,
        "kBET": 0.9,
        "knn_purity": 0.8,
        "leiden_ari": 0.7,
        "active_dims_shared": 10,
        "train_wall_time_sec": 10.0 if weight == 0 else 10.5,
    }
    row.update(overrides)
    return row


def test_recommend_f3_variant_passes_when_all_gates_pass():
    recommendation = recommend_f3_variant([
        _row(0.0),
        _row(0.1),
    ])

    assert recommendation["verdict"] == "pass"
    assert recommendation["recommended_weight"] == 0.1
    assert recommendation["candidates"][0]["passes"] is True


def test_recommend_f3_variant_rejects_when_orthogonality_gate_fails():
    recommendation = recommend_f3_variant([
        _row(0.0),
        _row(0.1, orthogonality_within_stratum=0.9),
    ])

    assert recommendation["verdict"] == "reject"
    assert recommendation["recommended_weight"] is None
    assert "orthogonality reduction <20%" in recommendation["candidates"][0]["failures"]


def test_recommend_f3_variant_iterates_when_baseline_missing():
    recommendation = recommend_f3_variant([
        _row(0.1),
    ])

    assert recommendation["verdict"] == "iterate"
    assert recommendation["recommended_weight"] is None
