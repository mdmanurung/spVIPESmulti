from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

pytestmark = pytest.mark.skipif(
    os.environ.get("SPVIPES_RUN_BENCHMARKS") != "1",
    reason="PERF-001 manual benchmark; set SPVIPES_RUN_BENCHMARKS=1 to run",
)


def test_perf_001_kbet_vectorized_benchmark(benchmark) -> None:
    """Benchmark the current vectorized kBET implementation."""

    from spVIPESmulti.metrics import kbet

    rng = np.random.default_rng(0)
    rep = rng.normal(size=(800, 16)).astype(np.float32)
    groups = np.array(["a", "b", "c", "d"] * 200)

    result = benchmark(kbet, rep, groups, 20)

    assert 0.0 <= result <= 1.0
