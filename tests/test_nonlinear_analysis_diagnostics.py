from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict

from anyfesolver.corotational_performance import note_dense_consistent_rotation
from anyfesolver.nonlinear_analysis_diagnostics import (
    capture_nonlinear_analysis_diagnostics,
)
from anyfesolver.vectorized_hill48 import record_hill48_execution


@dataclass
class _DummyResult:
    info: Dict[str, Any] = field(default_factory=dict)


@capture_nonlinear_analysis_diagnostics
def _concurrent_diagnostic_probe(
    path: str,
    barrier: threading.Barrier,
) -> _DummyResult:
    barrier.wait(timeout=5.0)
    if path == "hill48":
        record_hill48_execution(
            point_count=7,
            curve_name="linear",
            compiled=True,
        )
        return _DummyResult({"kinematics": "von_karman"})
    note_dense_consistent_rotation()
    return _DummyResult(
        {
            "kinematics": "corotational",
            "corotational_tangent": "consistent",
        }
    )


def test_analysis_diagnostics_are_isolated_between_concurrent_threads() -> None:
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        hill_future = executor.submit(_concurrent_diagnostic_probe, "hill48", barrier)
        corotational_future = executor.submit(
            _concurrent_diagnostic_probe,
            "corotational",
            barrier,
        )
        hill = hill_future.result(timeout=10.0)
        corotational = corotational_future.result(timeout=10.0)

    hill_performance = hill.info["nonlinear_performance"]
    assert hill_performance["hill48"]["compiled_call_count"] == 1
    assert hill_performance["hill48"]["compiled_point_count"] == 7
    assert hill_performance["corotational"]["activated"] is False

    corotational_performance = corotational.info["nonlinear_performance"]
    assert corotational_performance["hill48"]["public_call_count"] == 0
    assert corotational_performance["corotational"]["activated"] is False
    assert corotational_performance["corotational"]["exercised"] is True
    assert (
        corotational_performance["corotational"][
            "dense_consistent_rotations"
        ]
        == 1
    )
