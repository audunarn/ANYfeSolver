"""Qualification regressions for material-history and shell patch recovery."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from anysolver import (
    FEModel,
    LegacyQ4DeprecationWarning,
    LegacyShellElement,
    PatchRecoveryConfig,
    QuadraticBeamElement,
    ShellElement,
    generate_simple_panel_mesh,
    recover_shell_patch_stresses,
    recover_stress_result,
)
from anysolver.corotational import rotation_matrix_from_vector


_SURFACE_COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "xz")
_SURFACE_KEYS = tuple(
    f"global_{component}_{surface}"
    for surface in ("top", "bot")
    for component in _SURFACE_COMPONENTS
)


def _constant_shell_state(
    element: ShellElement,
    stress: tuple[float, float, float],
    *,
    num_layers: int = 3,
) -> dict[str, np.ndarray]:
    count = len(element.gauss_points) * num_layers
    return {
        "layer_strain": np.zeros((count, 3), dtype=float),
        "plastic_strain": np.zeros((count, 3), dtype=float),
        "layer_stress": np.tile(np.asarray(stress, dtype=float), (count, 1)),
        "alpha": np.zeros(count, dtype=float),
    }


def _surface_tensor(stresses: dict[str, object], surface: str, index: int = 0) -> np.ndarray:
    def value(component: str) -> float:
        return float(
            np.asarray(
                stresses[f"global_{component}_{surface}"],
                dtype=float,
            ).reshape(-1)[index]
        )

    return np.array(
        [
            [value("xx"), value("xy"), value("xz")],
            [value("xy"), value("yy"), value("yz")],
            [value("xz"), value("yz"), value("zz")],
        ],
        dtype=float,
    )


def _constant_global_surface_stresses(
    element: ShellElement,
    *,
    xx: float = 1.0,
) -> dict[str, np.ndarray]:
    count = len(element.gauss_points)
    return {
        key: np.full(count, xx if key.startswith("global_xx_") else 0.0)
        for key in _SURFACE_KEYS
    }


def test_nonlinear_result_displacements_are_default_and_mismatch_is_rejected() -> None:
    model = generate_simple_panel_mesh(
        1.0,
        1.0,
        0.01,
        num_divisions_x=1,
        num_divisions_y=1,
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        displacement[node.dofs[0]] = 2.0e-5 * node.x
    element = model.mesh.elements[1]
    nonlinear = SimpleNamespace(
        displacements=displacement.copy(),
        element_states={1: _constant_shell_state(element, (125.0, 25.0, 5.0))},
        info={"kinematics": "von_karman"},
        status="completed",
        load_factor=0.75,
    )

    recovered = recover_stress_result(model, nonlinear_result=nonlinear)

    assert recovered.provenance.mode == "material_history"
    assert recovered.provenance.analysis_context == {
        "kinematics": "von_karman",
        "result_type": "SimpleNamespace",
        "status": "completed",
        "load_factor": 0.75,
        "corotational_shell_rotation_stable": False,
    }
    assert recovered.element_stresses[1]["membrane_xx"][0] == pytest.approx(125.0)

    mismatched = displacement.copy()
    mismatched[model.mesh.nodes[1].dofs[0]] += 1.0e-6
    with pytest.raises(
        ValueError,
        match="supplied displacements do not match nonlinear_result.displacements",
    ):
        recover_stress_result(
            model,
            mismatched,
            nonlinear_result=nonlinear,
        )


def test_quadratic_beam_history_preserves_stations_and_weighted_resultants() -> None:
    model = FEModel("quadratic_beam_recovery")
    material = model.add_material(
        "steel",
        elastic_modulus=210.0e9,
        poisson_ratio=0.3,
    )
    for node_id, x in enumerate((0.0, 1.0, 2.0), start=1):
        model.add_node(node_id, x, 0.0, 0.0)
    section = {
        "area": 0.01,
        "Iy": 8.0e-5,
        "Iz": 1.6e-4,
        "J": 2.0e-6,
        "c_y": 0.2,
        "c_z": 0.1,
        "torsion_modulus": 0.004,
        "shear_factor_y": 0.8,
        "shear_factor_z": 0.75,
    }
    element = QuadraticBeamElement(
        1,
        [1, 2, 3],
        "steel",
        section,
    )
    model.add_element(1, element)

    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    end = model.mesh.nodes[3]
    displacement[end.dofs[1]] = 0.004
    displacement[end.dofs[2]] = 0.002
    displacement[end.dofs[3]] = 0.001

    fiber_y = np.array([-0.2, 0.2, -0.2, 0.2])
    fiber_z = np.array([-0.1, -0.1, 0.1, 0.1])
    fiber_weights = np.array([0.001, 0.002, 0.003, 0.004])
    stress_by_station = 1.0e6 * np.array(
        [
            [100.0, 150.0, 250.0, 350.0],
            [-400.0, -250.0, -100.0, 50.0],
            [80.0, -120.0, 200.0, -300.0],
        ]
    )
    state = {
        "fiber_strain": np.zeros(stress_by_station.size),
        "fiber_stress": stress_by_station.reshape(-1),
        "fiber_y": fiber_y,
        "fiber_z": fiber_z,
        "fiber_weights": fiber_weights,
        "fiber_station_count": len(element.GAUSS_POINTS),
        "plastic_strain": np.zeros(stress_by_station.size),
        "alpha": np.zeros(stress_by_station.size),
    }

    recovered = recover_stress_result(
        model,
        displacement,
        element_states={1: state},
    )
    stress = recovered.element_stresses[1]

    expected_force = stress_by_station @ fiber_weights
    expected_moment_y = (stress_by_station * fiber_z[None, :]) @ fiber_weights
    expected_moment_z = (stress_by_station * fiber_y[None, :]) @ fiber_weights
    expected_axial = float(np.mean(expected_force) / np.sum(fiber_weights))
    expected_bending_y = expected_moment_y * section["c_z"] / section["Iy"]
    expected_bending_z = expected_moment_z * section["c_y"] / section["Iz"]
    shear_y = (
        material.shear_modulus
        * section["shear_factor_y"]
        * (displacement[end.dofs[1]] / 2.0)
    )
    shear_z = (
        material.shear_modulus
        * section["shear_factor_z"]
        * (displacement[end.dofs[2]] / 2.0)
    )
    torsion = (
        material.shear_modulus
        * section["J"]
        * (displacement[end.dofs[3]] / 2.0)
        / section["torsion_modulus"]
    )
    expected_vm = np.sqrt(
        stress_by_station**2
        + 3.0 * (shear_y**2 + shear_z**2 + torsion**2)
    )

    assert stress["fiber_station_count"] == 3
    np.testing.assert_allclose(stress["axial_force_by_station"], expected_force)
    np.testing.assert_allclose(
        stress["bending_moment_y_by_station"],
        expected_moment_y,
    )
    np.testing.assert_allclose(
        stress["bending_moment_z_by_station"],
        expected_moment_z,
    )
    assert stress["axial_stress"] == pytest.approx(expected_axial)
    assert stress["bending_stress_y"] == pytest.approx(
        expected_bending_y[np.argmax(np.abs(expected_bending_y))]
    )
    assert stress["bending_stress_z"] == pytest.approx(
        expected_bending_z[np.argmax(np.abs(expected_bending_z))]
    )
    assert stress["shear_stress_y"] == pytest.approx(abs(shear_y))
    assert stress["shear_stress_z"] == pytest.approx(abs(shear_z))
    assert stress["torsional_stress"] == pytest.approx(abs(torsion))
    assert stress["von_mises"] == pytest.approx(float(np.max(expected_vm)))
    assert stress["mixed_reconstruction_von_mises"] == pytest.approx(
        float(np.max(expected_vm))
    )
    assert (
        recovered.provenance.per_element_component_sources[1]["shear_and_torsion"]
        == "elastic_reconstruction_from_same_solution"
    )
    assert (
        recovered.provenance.per_element_component_sources[1][
            "mixed_reconstruction_von_mises"
        ]
        == "committed_beam_fiber_state_plus_elastic_shear_and_torsion"
    )
    assert (
        recovered.provenance.per_element_component_sources[1]["von_mises"]
        == (
            "committed_elastic_beam_fiber_state_plus_"
            "elastic_shear_and_torsion"
        )
    )


def test_corotational_shell_history_rotates_global_tensor_objectively() -> None:
    model = generate_simple_panel_mesh(
        1.0,
        1.0,
        0.01,
        num_divisions_x=1,
        num_divisions_y=1,
    )
    element = model.mesh.elements[1]
    state = _constant_shell_state(element, (120.0, 35.0, 18.0))
    zero = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)

    reference = recover_stress_result(
        model,
        zero,
        element_states={1: state},
        kinematics="corotational",
        return_global=True,
    )

    rotation_vector = np.array([0.42, -0.31, 0.57], dtype=float)
    rotation = rotation_matrix_from_vector(rotation_vector)
    coordinates = np.vstack(
        [model.mesh.nodes[node_id].coords() for node_id in element.node_ids]
    )
    centroid = np.mean(coordinates, axis=0)
    rotated_coordinates = (rotation @ (coordinates - centroid).T).T + centroid
    rigid_displacement = np.zeros_like(zero)
    for local_index, node_id in enumerate(element.node_ids):
        node = model.mesh.nodes[node_id]
        rigid_displacement[node.dofs[:3]] = (
            rotated_coordinates[local_index] - coordinates[local_index]
        )
        rigid_displacement[node.dofs[3:]] = rotation_vector

    rotated = recover_stress_result(
        model,
        nonlinear_result=SimpleNamespace(
            displacements=rigid_displacement,
            element_states={1: state},
            info={"kinematics": "corotational"},
            status="completed",
            load_factor=1.0,
        ),
        return_global=True,
    )

    reference_tensor = _surface_tensor(reference.element_stresses[1], "top")
    rotated_tensor = _surface_tensor(rotated.element_stresses[1], "top")
    np.testing.assert_allclose(
        rotated_tensor,
        rotation @ reference_tensor @ rotation.T,
        # The corotational pull-back subtracts metre-scale coordinates, so
        # its documented rigid-motion floor is above machine epsilon.
        rtol=2.0e-7,
        atol=5.0e-6,
    )
    np.testing.assert_allclose(
        rotated.element_stresses[1]["von_mises"],
        reference.element_stresses[1]["von_mises"],
        rtol=2.0e-7,
        atol=5.0e-6,
    )
    assert (
        rotated.provenance.per_element_component_sources[1]["stress_frame"]
        == "current_corotated_center_frame"
    )


def test_discontinuous_patch_preserves_both_material_side_values() -> None:
    model = generate_simple_panel_mesh(
        2.0,
        1.0,
        0.01,
        num_divisions_x=2,
        num_divisions_y=1,
    )
    model.add_material(
        "second_material",
        elastic_modulus=205.0e9,
        poisson_ratio=0.29,
    )
    element_ids = sorted(model.mesh.elements)
    left = model.mesh.elements[element_ids[0]]
    right = model.mesh.elements[element_ids[1]]
    right.material_name = "second_material"
    shared_nodes = set(left.node_ids) & set(right.node_ids)
    states = {
        int(left.element_id): _constant_shell_state(left, (100.0, 0.0, 0.0)),
        int(right.element_id): _constant_shell_state(right, (-100.0, 0.0, 0.0)),
    }

    patch = recover_stress_result(
        model,
        np.zeros(model.mesh.dof_manager.total_dofs, dtype=float),
        element_states=states,
        patch_config=PatchRecoveryConfig(),
    ).nodal_stresses

    assert patch is not None
    for node_id in shared_nodes:
        assert node_id in patch["discontinuous_node_ids"]
        assert node_id not in patch["nodal"]
        records = patch["nodal_regions"][node_id]
        assert len(records) == 2
        values = sorted(
            record["values"]["global_xx_top"] for record in records
        )
        assert values == pytest.approx([-100.0, 100.0])
        assert all(abs(value) == pytest.approx(100.0) for value in values)
        assert patch["node_diagnostics"][node_id]["reason"] == "material_discontinuity"
    assert patch["max_von_mises"] == pytest.approx(100.0)


def test_patch_rejects_reduced_q8_and_warped_q4_outside_qualified_scope() -> None:
    q8 = FEModel("reduced_q8_patch_guard")
    q8.add_material("steel", 210.0e9, 0.3)
    q8_coordinates = (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.5, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.5, 0.0),
    )
    for node_id, coordinates in enumerate(q8_coordinates, start=1):
        q8.add_node(node_id, *coordinates)
    q8_element = ShellElement(
        1,
        list(range(1, 9)),
        "steel",
        thickness=0.01,
        reduced_integration=True,
    )
    q8.add_element(1, q8_element)
    q8_patch = recover_shell_patch_stresses(
        q8,
        {1: _constant_global_surface_stresses(q8_element)},
    )

    assert q8_patch["skipped_element_reasons"] == {
        1: "unqualified_reduced_q8_topology"
    }
    assert q8_patch["nodal"] == {}
    assert set(q8_patch["discontinuous_node_ids"]) == set(range(1, 9))

    warped = FEModel("warped_q4_patch_guard")
    warped.add_material("steel", 210.0e9, 0.3)
    warped_coordinates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.2),
        (0.0, 1.0, 0.0),
    )
    for node_id, coordinates in enumerate(warped_coordinates, start=1):
        warped.add_node(node_id, *coordinates)
    with pytest.warns(LegacyQ4DeprecationWarning, match="temporary rollback"):
        warped_element = LegacyShellElement(
            1,
            [1, 2, 3, 4],
            "steel",
            thickness=0.01,
        )
    warped.add_element(1, warped_element)
    warped_patch = recover_shell_patch_stresses(
        warped,
        {1: _constant_global_surface_stresses(warped_element)},
        config=PatchRecoveryConfig(planarity_relative_tolerance=1.0e-3),
    )

    assert warped_patch["skipped_element_reasons"][1].startswith("warped_shell_")
    assert warped_patch["nodal"] == {}
    assert set(warped_patch["discontinuous_node_ids"]) == {1, 2, 3, 4}


def test_patch_indicator_is_explicitly_a_stress_discrepancy_not_energy_norm() -> None:
    model = generate_simple_panel_mesh(
        2.0,
        1.0,
        0.01,
        num_divisions_x=2,
        num_divisions_y=2,
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        displacement[node.dofs[0]] = 1.0e-4 * node.x
        displacement[node.dofs[1]] = -2.5e-5 * node.y

    patch = recover_stress_result(
        model,
        displacement,
        patch_config=PatchRecoveryConfig(include_error_indicator=True),
    ).nodal_stresses

    assert patch is not None
    indicator = patch["error_indicator"]
    assert indicator["status"] == "available"
    assert indicator["type"] == "normalized_global_surface_stress_l2"
    assert indicator["is_energy_norm_estimate"] is False
    assert "not a ZZ compliance-energy error estimate" in indicator["interpretation"]
    assert (
        patch["qualification"]["indicator_semantics"]
        == "surface_stress_l2_discrepancy_not_energy_norm"
    )


def test_scientific_provenance_excludes_nondeterministic_timing_by_default() -> None:
    model = generate_simple_panel_mesh(
        1.0,
        1.0,
        0.01,
        num_divisions_x=1,
        num_divisions_y=1,
    )
    recovered = recover_stress_result(
        model,
        np.zeros(model.mesh.dof_manager.total_dofs, dtype=float),
    )

    assert recovered.execution_report is not None
    assert "execution" not in recovered.provenance_dict()
    assert (
        recovered.provenance_dict(include_execution=True)["execution"][
            "elapsed_seconds"
        ]
        >= 0.0
    )
