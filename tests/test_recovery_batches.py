from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import RecoveryConfig, ResourceConfig, generate_simple_panel_mesh
from anyfesolver.recovery import (
    _compute_one_element_stress,
    recover_element_stresses_with_report,
)


def _panel():
    return generate_simple_panel_mesh(
        4.0,
        2.0,
        0.01,
        num_divisions_x=4,
        num_divisions_y=3,
    )


def _large_panel():
    return generate_simple_panel_mesh(
        10.0,
        10.0,
        0.01,
        num_divisions_x=10,
        num_divisions_y=10,
    )


def test_chunked_scalar_recovery_preserves_order_and_exact_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _large_panel()
    displacement = np.linspace(
        0.0,
        1.0e-5,
        model.mesh.dof_manager.total_dofs,
    )
    recovery = RecoveryConfig(
        components=["von_mises"],
    )
    monkeypatch.setattr("anyfesolver.recovery.JIT_ENABLED", False)

    serial, serial_report = recover_element_stresses_with_report(
        model,
        displacement,
        recovery,
        resource_config=ResourceConfig(recovery_threads=1),
    )
    threaded, threaded_report = recover_element_stresses_with_report(
        model,
        displacement,
        recovery,
        resource_config=ResourceConfig(recovery_threads=2),
    )

    assert list(threaded) == list(serial)
    for element_id in serial:
        np.testing.assert_array_equal(
            threaded[element_id]["von_mises"],
            serial[element_id]["von_mises"],
        )
    metadata = threaded_report.metadata
    assert threaded_report.backend == "thread_pool"
    assert metadata["recovery_backend"] == "scalar_chunk_thread_pool"
    assert metadata["chunk_count"] < threaded_report.item_count
    assert metadata["plan_reused"] is True
    assert sum(metadata["batch_counts"].values()) == threaded_report.item_count
    assert serial_report.metadata["plan_reused"] is False


def test_selection_is_runtime_only_and_plan_is_bounded() -> None:
    model = _large_panel()
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)

    first, first_report = recover_element_stresses_with_report(
        model,
        displacement,
        RecoveryConfig(),
        resource_config=ResourceConfig(recovery_threads=2),
    )
    cached_plan = model.mesh._recovery_batch_plan
    second, second_report = recover_element_stresses_with_report(
        model,
        displacement,
        RecoveryConfig(),
        resource_config=ResourceConfig(recovery_threads=2),
    )

    assert list(first) == list(model.mesh.elements)
    assert list(second) == list(model.mesh.elements)
    assert model.mesh._recovery_batch_plan is cached_plan
    assert first_report.metadata["plan_reused"] is False
    assert second_report.metadata["plan_reused"] is True


def test_plan_ignores_load_revision_and_invalidates_on_geometry() -> None:
    model = _large_panel()
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)

    recover_element_stresses_with_report(model, displacement)
    first = model.mesh._recovery_batch_plan
    model.bump_revision("load")
    _values, load_report = recover_element_stresses_with_report(model, displacement)
    assert model.mesh._recovery_batch_plan is first
    assert load_report.metadata["plan_reused"] is True

    node = model.mesh.nodes[2]
    model.set_node_coordinates(2, node.x, node.y, node.z + 1.0e-4)
    _values, geometry_report = recover_element_stresses_with_report(
        model, displacement
    )
    assert model.mesh._recovery_batch_plan is not first
    assert geometry_report.metadata["plan_reused"] is False


def test_report_serialization_exposes_fallback_diagnostics() -> None:
    model = _panel()
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)

    _values, report = recover_element_stresses_with_report(
        model,
        displacement,
        resource_config=ResourceConfig(recovery_threads=3),
    )
    payload = report.to_dict()

    assert payload["metadata"]["fallback_element_count"] == len(
        model.mesh.elements
    )
    assert payload["metadata"]["eligible_element_count"] == 0
    assert "below_recovery_plan_threshold" in payload["metadata"]["fallback_reasons"]
    assert payload["metadata"]["plan_retained_bytes"] == 0


def test_compiled_isotropic_s4_matches_scalar_oracle_for_warped_global_output() -> None:
    model = _large_panel()
    for node_id, node in tuple(model.mesh.nodes.items()):
        model.set_node_coordinates(
            node_id,
            node.x,
            node.y,
            2.0e-3 * np.sin(0.7 * node.x + 0.4 * node.y),
        )
    rng = np.random.default_rng(81)
    displacement = rng.normal(
        scale=2.0e-5,
        size=model.mesh.dof_manager.total_dofs,
    )

    compiled, report = recover_element_stresses_with_report(
        model,
        displacement,
        return_global=True,
        resource_config=ResourceConfig(recovery_threads=3),
    )
    scalar = {}
    for element_id in model.mesh.elements:
        item = _compute_one_element_stress(
            model,
            displacement,
            element_id,
            return_global=True,
        )
        assert item is not None
        scalar[item[0]] = item[1]

    assert report.backend == "thread_pool"
    assert report.metadata["recovery_backend"] == "compiled_isotropic_s4"
    assert report.metadata["eligible_element_count"] == len(model.mesh.elements)
    assert list(compiled) == list(scalar)
    for element_id, expected in scalar.items():
        assert compiled[element_id].keys() == expected.keys()
        for field, expected_value in expected.items():
            if isinstance(expected_value, str):
                assert compiled[element_id][field] == expected_value
            else:
                np.testing.assert_allclose(
                    compiled[element_id][field],
                    expected_value,
                    rtol=2.0e-13,
                    atol=1.0e-7,
                )
