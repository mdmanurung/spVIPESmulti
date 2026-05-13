"""Tests for the F10b CellDISECT parity runner helpers."""

import importlib.util
import sys
from pathlib import Path

import csv
import json
import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_kang_celldisect_parity.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("benchmark_kang_celldisect_parity", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules[_SCRIPT_SPEC.name] = _SCRIPT_MODULE
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)

Config = _SCRIPT_MODULE.Config
METRIC_NAMES = _SCRIPT_MODULE.METRIC_NAMES
ARTIFACT_FIELDS = _SCRIPT_MODULE.ARTIFACT_FIELDS
external_celldisect_rows = _SCRIPT_MODULE.external_celldisect_rows
metric_rows_for_prediction = _SCRIPT_MODULE.metric_rows_for_prediction
write_artifacts = _SCRIPT_MODULE.write_artifacts
strip_group_prefixes = _SCRIPT_MODULE.strip_group_prefixes


def _cfg(tmp_path):
    return Config(
        run_id="f10b_test",
        kang_h5ad_path="unused.h5ad",
        seeds=[0],
        max_epochs=1,
        batch_size=8,
        max_cells_per_condition=10,
        n_top_genes=5,
        n_shared=2,
        n_private=2,
        n_hidden=8,
        condition_key="label",
        label_key="cell_type",
        donor_key="replicate",
        splits=["cd14_mono"],
        condition_from=None,
        condition_to=None,
        disentangle_preset="full",
        audit_dir=str(tmp_path / "F10"),
    )


def test_metric_rows_for_prediction_are_schema_complete_and_finite(tmp_path):
    cfg = _cfg(tmp_path)
    x_ctrl = np.array([[1.0, 1.0, 1.0], [1.0, 1.5, 1.0]])
    x_true = np.array([[2.0, 1.0, 4.0], [2.2, 1.0, 4.5]])
    x_pred = np.array([[1.8, 1.0, 3.5], [2.1, 1.0, 4.0]])

    rows = metric_rows_for_prediction(cfg, 0, "spVIPESmulti", "cd14_mono", x_ctrl, x_true, x_pred)

    assert {row["metric"] for row in rows} == set(METRIC_NAMES)
    for row in rows:
        assert set(ARTIFACT_FIELDS).issubset(row)
        assert row["model"] == "spVIPESmulti"
        assert row["status"] == "ok"
        assert np.isfinite(float(row["value"]))


def test_missing_external_celldisect_import_is_skipped(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)

    def _raise(_name):
        raise ModuleNotFoundError("not installed")

    monkeypatch.setattr(_SCRIPT_MODULE.importlib, "import_module", _raise)

    rows = external_celldisect_rows(cfg, 0, "cd14_mono")

    assert {row["model"] for row in rows} == {"CellDISECT"}
    assert {row["status"] for row in rows} == {"skipped"}
    assert all("external install unavailable" in row["notes"] for row in rows)


def test_strip_group_prefixes_maps_decoder_names_to_raw_genes():
    assert strip_group_prefixes(["stim_ACTB", "stim_IFI6"], "stim") == ["ACTB", "IFI6"]


def test_write_artifacts_creates_metrics_summary_recommendation_and_kang_note(tmp_path):
    cfg = _cfg(tmp_path)
    rows = metric_rows_for_prediction(
        cfg,
        0,
        "spVIPESmulti",
        "cd14_mono",
        np.array([[1.0, 1.0, 1.0], [1.0, 1.5, 1.0]]),
        np.array([[2.0, 1.0, 4.0], [2.2, 1.0, 4.5]]),
        np.array([[1.8, 1.0, 3.5], [2.1, 1.0, 4.0]]),
    )
    rows.extend(external_celldisect_rows(cfg, 0, "cd14_mono"))

    paths = write_artifacts(rows, cfg, audit_dir=tmp_path / "F10", mirror_dir=tmp_path / "kang_ifnb")

    assert set(paths) == {"metrics", "summary", "recommendation", "kang_note"}
    for path in paths.values():
        assert path.exists()
    with paths["metrics"].open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert set(ARTIFACT_FIELDS).issubset(metric_rows[0])
    assert {row["model"] for row in metric_rows} == {"spVIPESmulti", "CellDISECT"}
    rec = json.loads(paths["recommendation"].read_text(encoding="utf-8"))
    assert rec["verdict"] in {"informational", "pass"}
    assert rec["promotion"].startswith("audit harness only")
