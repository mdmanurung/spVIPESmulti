"""pytest configuration for audit regression tests."""
import os
import pytest

# Hide CUDA devices before any Lightning/PyTorch CUDA detection so that
# integration tests force CPU-only execution without triggering the CUDA
# driver initialisation. This is safe on CPU-only HPC nodes and nodes with
# incompatible CUDA drivers (e.g. driver 12080 vs PyTorch CUDA requirements).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "audit_regression: regression tests pinned to findings in audits/2026-05-08-full-package*.md",
    )


def pytest_addoption(parser):
    # Use a try/except so re-registration in the top-level conftest is harmless.
    try:
        parser.addoption(
            "--runaudit",
            action="store_true",
            default=False,
            help="Run slow audit calibration tests (permutation nulls, reference comparisons).",
        )
    except ValueError:
        pass


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runaudit", default=False):
        return
    skip_slow = pytest.mark.skip(reason="slow audit calibration test; pass --runaudit to enable")
    for item in items:
        if "calibration" in item.nodeid and "audit_regression" in item.keywords:
            item.add_marker(skip_slow)
