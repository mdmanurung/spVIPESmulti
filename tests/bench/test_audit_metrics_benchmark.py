from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

pytestmark = pytest.mark.skip(reason="PERF-001 manual benchmark scaffold; keep deselected from default CI")


def test_perf_001_kbet_hot_loop_benchmark(benchmark) -> None:
    """Benchmark the current kBET Python/pandas loop implementation."""

    from spVIPESmulti.metrics import kbet

    rng = np.random.default_rng(0)
    rep = rng.normal(size=(128, 8)).astype(np.float32)
    groups = np.array(["a", "b", "c", "d"] * 32)

    result = benchmark(kbet, rep, groups, 15)

    assert 0.0 <= result <= 1.0
