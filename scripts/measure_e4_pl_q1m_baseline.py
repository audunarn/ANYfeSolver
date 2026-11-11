"""Measure the qualified-Q4 Q1M gate-1 performance baseline.

This is a burn-in evidence producer, not a production benchmark API.  It
performs one warm-up and eleven timed repetitions, verifies the qualified Q4
shared-cache assembly against a scalar assembly, and makes no speed claim.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable

import numpy as np

from anyfesolver import FEModel, QualifiedE4PLShellElement, assemble_stiffness_matrix
from anyfesolver.elements import create_element


SCHEMA = "anysolver.s4.e4-pl-q1m-performance-observation-v1"
BASELINE_SCHEMA = "anysolver.s4.e4-pl-q1m-performance-baseline-v1"
NO_SPEED_CLAIM = "GATE_1_BASELINE_ONLY_NO_SPEED_CLAIM"
WARMUPS = 1
REPETITIONS = 11
MEASUREMENT_NAMES = (
    "qualified_q4_cached_tangent",
    "qualified_q4_warm_global_assembly",
)

HARD_GATE_NODES = {
    "batch_path_equality": [
        "tests/test_e4_pl_workflow_parity.py::test_global_assembly_uses_candidate_scalar_kernel_and_activity_lifecycle",
        "tests/test_e4_pl_workflow_parity.py::test_structured_mesh_cold_assembly_reuses_translation_equivalent_geometry",
    ],
    "q4_numerical_parity": [
        "tests/test_e4_pl_workflow_parity.py::test_global_assembly_uses_candidate_scalar_kernel_and_activity_lifecycle",
    ],
    "warm_cache_reuse": [
        "tests/test_e4_pl_workflow_parity.py::test_structured_mesh_cold_assembly_reuses_translation_equivalent_geometry",
    ],
}


def timing_summary(samples_ns: list[int]) -> dict[str, Any]:
    """Return canonical integer timing statistics for an odd-sized sample."""

    if len(samples_ns) != REPETITIONS:
        raise ValueError(f"exactly {REPETITIONS} timing samples are required")
    if any(type(value) is not int or value < 0 for value in samples_ns):
        raise ValueError("timing samples must be nonnegative integer nanoseconds")
    ordered = sorted(samples_ns)
    median_ns = int(statistics.median(ordered))
    mad_ns = int(statistics.median(abs(value - median_ns) for value in ordered))
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mad_ns": mad_ns,
        "median_ns": median_ns,
        "p95_ns": int(ordered[p95_index]),
        "samples_ns": list(samples_ns),
    }


def _model() -> FEModel:
    model = FEModel("q1m_performance_baseline")
    model.add_material("steel", 210.0e9, 0.3)
    node_id = 1
    for element_id in range(1, 5):
        offset = 2.0 * (element_id - 1)
        node_ids: list[int] = []
        for x, y, z in (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ):
            model.add_node(node_id, x + offset, y, z)
            node_ids.append(node_id)
            node_id += 1
        element = create_element(
            "shell",
            element_id,
            node_ids,
            "steel",
            thickness=0.02,
        )
        if type(element) is not QualifiedE4PLShellElement:
            raise AssertionError("public Q4 selector did not create the qualified element")
        model.add_element(element_id, element)
    return model


def _scalar_global(model: FEModel) -> np.ndarray:
    size = model.mesh.dof_manager.total_dofs
    result = np.zeros((size, size), dtype=float)
    for element in model.mesh.elements.values():
        material = model.get_material(element.material_name)
        local = element.compute_stiffness_matrix(model.mesh, material)
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        result[np.ix_(dofs, dofs)] += local
    return result


def _measure(call: Callable[[], Any]) -> tuple[list[int], list[Any]]:
    for _ in range(WARMUPS):
        call()
    samples: list[int] = []
    results: list[Any] = []
    for _ in range(REPETITIONS):
        started = time.perf_counter_ns()
        result = call()
        samples.append(time.perf_counter_ns() - started)
        results.append(result)
    return samples, results


def collect_performance_observation() -> dict[str, Any]:
    """Collect strict Q1M hard-gate observations and timing baselines."""

    model = _model()
    cold, cold_info = assemble_stiffness_matrix(model)
    cold_dense = cold.toarray()
    scalar_dense = _scalar_global(model)
    diagnostic = cold_info["diagnostics"]["qualified_e4_pl_stiffness"]

    cached_component_ids = tuple(
        id(element._qualified_components) for element in model.mesh.elements.values()
    )
    tangent_element = next(iter(model.mesh.elements.values()))
    tangent_material = model.get_material(tangent_element.material_name)
    tangent_reference = tangent_element.compute_stiffness_matrix(
        model.mesh, tangent_material
    ).copy()

    tangent_samples, tangent_results = _measure(
        lambda: tangent_element.compute_stiffness_matrix(model.mesh, tangent_material)
    )
    assembly_samples, assembly_results = _measure(
        lambda: assemble_stiffness_matrix(model)
    )

    numerical_parity = bool(np.array_equal(cold_dense, scalar_dense))
    batch_path_equality = all(
        np.array_equal(matrix.toarray(), scalar_dense)
        and info["diagnostics"]["qualified_e4_pl_stiffness"] == diagnostic
        for matrix, info in assembly_results
    )
    warm_cache_reuse = (
        diagnostic
        == {
            "path": "shared_geometry_cache",
            "element_count": 4,
            "unique_geometry_count": 1,
        }
        and cached_component_ids
        == tuple(
            id(element._qualified_components)
            for element in model.mesh.elements.values()
        )
        and all(np.array_equal(matrix, tangent_reference) for matrix in tangent_results)
    )

    observations = {
        "batch_path_equality": batch_path_equality,
        "q4_numerical_parity": numerical_parity,
        "warm_cache_reuse": bool(warm_cache_reuse),
    }
    failed = sorted(name for name, passed in observations.items() if not passed)
    if failed:
        raise AssertionError(f"Q1M performance hard gate failed: {failed}")

    hard_gates = {
        name: {
            "evidence_nodes": HARD_GATE_NODES[name],
            "observed": observations[name],
            "status": "PASS",
        }
        for name in sorted(HARD_GATE_NODES)
    }
    baseline = {
        "measurements": {
            "qualified_q4_cached_tangent": timing_summary(tangent_samples),
            "qualified_q4_warm_global_assembly": timing_summary(assembly_samples),
        },
        "repetitions": REPETITIONS,
        "schema": BASELINE_SCHEMA,
        "speed_claim": NO_SPEED_CLAIM,
        "warmups": WARMUPS,
    }
    return {
        "hard_gates": hard_gates,
        "performance_baseline": baseline,
        "schema": SCHEMA,
    }


__all__ = [
    "BASELINE_SCHEMA",
    "HARD_GATE_NODES",
    "MEASUREMENT_NAMES",
    "NO_SPEED_CLAIM",
    "REPETITIONS",
    "SCHEMA",
    "WARMUPS",
    "collect_performance_observation",
    "timing_summary",
]
