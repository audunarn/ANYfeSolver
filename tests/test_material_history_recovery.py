"""Unified material-history and guarded shell patch recovery tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver import (
    BeamElement,
    BoundaryCondition,
    DNVC208MaterialCurve,
    FEModel,
    LoadCase,
    PatchRecoveryConfig,
    create_fe_result,
    generate_simple_panel_mesh,
    recover_stress_result,
    solve_static_nonlinear,
)


def _constant_shell_state(element, stress: tuple[float, float, float], num_layers: int = 3):
    n_points = len(element.gauss_points) * num_layers
    layer_stress = np.tile(np.asarray(stress, dtype=float), (n_points, 1))
    return {
        "layer_strain": np.zeros((n_points, 3), dtype=float),
        "plastic_strain": np.zeros((n_points, 3), dtype=float),
        "layer_stress": layer_stress,
        "alpha": np.full(n_points, 0.02, dtype=float),
    }


def _global_membrane_displacement(
    model, eps_x: float, eps_y: float, gamma_xy: float
) -> np.ndarray:
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        displacement[node.dofs[0]] = eps_x * node.x
        displacement[node.dofs[1]] = eps_y * node.y + gamma_xy * node.x
    return displacement


def test_unified_recovery_uses_committed_shell_history_and_owns_snapshot() -> None:
    model = generate_simple_panel_mesh(
        1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    element = model.mesh.elements[1]
    state = _constant_shell_state(element, (250.0, 50.0, 10.0))
    nonlinear = SimpleNamespace(element_states={1: state})

    recovered = recover_stress_result(
        model, displacement, nonlinear_result=nonlinear
    )

    expected_vm = np.sqrt(250.0**2 - 250.0 * 50.0 + 50.0**2 + 3.0 * 10.0**2)
    assert recovered.provenance.mode == "material_history"
    assert recovered.provenance.per_element_source[1] == "committed_shell_layer_state"
    np.testing.assert_allclose(recovered.element_stresses[1]["von_mises"], expected_vm)
    recovered_shell = recovered.element_stresses[1]
    expected_membrane_resultants = np.broadcast_to(
        np.asarray((250.0, 50.0, 10.0)) * float(element.thickness),
        recovered_shell["membrane_resultants"].shape,
    )
    np.testing.assert_allclose(
        recovered_shell["membrane_resultants"],
        expected_membrane_resultants,
    )
    np.testing.assert_allclose(
        recovered_shell["bending_resultants"],
        np.zeros_like(recovered_shell["bending_resultants"]),
        atol=1.0e-15,
    )
    # Zero displacement would produce zero elastic stress, proving the state
    # rather than displacement reconstruction supplied the reported stress and
    # resultants.
    assert np.max(recovered.element_stresses[1]["von_mises"]) > 0.0

    state["layer_stress"][0, 0] = -9999.0
    assert recovered.committed_element_states[1]["layer_stress"][0, 0] == 250.0
    np.testing.assert_allclose(
        recovered.element_stresses[1]["membrane_resultants"],
        expected_membrane_resultants,
    )


def test_shell_history_von_mises_excludes_elastic_transverse_shear() -> None:
    model = generate_simple_panel_mesh(
        1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    displacement[model.mesh.nodes[3].dofs[2]] = 1.0e-3
    element = model.mesh.elements[1]
    state = _constant_shell_state(element, (250.0, 50.0, 10.0))

    recovered = recover_stress_result(
        model,
        displacement,
        element_states={1: state},
    )

    expected_vm = np.sqrt(250.0**2 - 250.0 * 50.0 + 50.0**2 + 3.0 * 10.0**2)
    np.testing.assert_allclose(
        recovered.element_stresses[1]["von_mises"],
        expected_vm,
    )
    assert np.max(
        recovered.element_stresses[1]["mixed_reconstruction_von_mises"]
    ) > np.max(recovered.element_stresses[1]["von_mises"])
    assert (
        recovered.provenance.per_element_component_sources[1]["von_mises"]
        == "committed_shell_layer_state_in_plane"
    )


def test_committed_elastic_shell_state_keeps_transverse_shear_in_von_mises() -> None:
    model = generate_simple_panel_mesh(
        1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    displacement[model.mesh.nodes[3].dofs[2]] = 1.0e-3
    element = model.mesh.elements[1]
    state = _constant_shell_state(element, (250.0, 50.0, 10.0))
    state["alpha"].fill(0.0)

    recovered = recover_stress_result(
        model,
        displacement,
        element_states={1: state},
    )

    np.testing.assert_allclose(
        recovered.element_stresses[1]["von_mises"],
        recovered.element_stresses[1]["mixed_reconstruction_von_mises"],
    )
    assert (
        recovered.provenance.per_element_component_sources[1]["von_mises"]
        == (
            "committed_elastic_shell_layer_state_plus_"
            "elastic_transverse_shear"
        )
    )


def test_unified_recovery_uses_beam_fiber_history_and_labels_mixed_fallback() -> None:
    model = FEModel("beam_history")
    model.add_material("steel", elastic_modulus=210.0e9, poisson_ratio=0.3)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 2.0, 0.0, 0.0)
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_element(2, BeamElement(2, [2, 3], "steel", section))
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    displacement[model.mesh.nodes[2].dofs[1]] = 1.0e-3
    state = {
        "fiber_strain": np.array([0.01, -0.005]),
        "fiber_stress": np.array([320.0e6, -280.0e6]),
        "fiber_y": np.array([-0.07, 0.03]),
        "fiber_z": np.array([0.0, 0.0]),
        "fiber_weights": np.array([0.003, 0.007]),
        "fiber_station_count": 1,
        "plastic_strain": np.array([0.008, -0.003]),
        "alpha": np.array([0.008, 0.003]),
    }

    recovered = recover_stress_result(
        model, displacement, element_states={1: state}
    )

    assert recovered.provenance.mode == "mixed"
    assert recovered.provenance.history_aware_element_ids == (1,)
    assert recovered.provenance.elastic_reconstruction_element_ids == (2,)
    assert recovered.element_stresses[1]["von_mises"] == pytest.approx(320.0e6)
    assert (
        recovered.element_stresses[1]["mixed_reconstruction_von_mises"]
        > recovered.element_stresses[1]["von_mises"]
    )
    assert recovered.element_stresses[1]["axial_stress"] == pytest.approx(-100.0e6)
    assert (
        recovered.provenance.per_element_component_sources[1][
            "axial_fibers_and_section_resultants"
        ]
        == "committed_beam_fiber_state"
    )
    assert (
        recovered.provenance.per_element_component_sources[1]["von_mises"]
        == "committed_beam_fiber_state"
    )
    assert recovered.element_stresses[2]["von_mises"] > 0.0


def test_create_fe_result_exposes_history_provenance_and_committed_states() -> None:
    model = generate_simple_panel_mesh(
        1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    states = {1: _constant_shell_state(model.mesh.elements[1], (100.0, 0.0, 0.0))}

    result = create_fe_result(
        model,
        displacement,
        {"solver_type": "unit"},
        element_states=states,
    )

    assert result.stress_recovery_provenance["mode"] == "material_history"
    assert sorted(result.committed_element_states) == [1]
    assert result.solver_info["stress_recovery"]["history_aware_element_ids"] == [1]


def test_actual_plastic_shell_solve_recovers_return_mapped_committed_state() -> None:
    model = generate_simple_panel_mesh(
        1.0,
        1.0,
        0.01,
        num_divisions_x=1,
        num_divisions_y=1,
    )
    model.materials["steel"].hardening_curve = DNVC208MaterialCurve(
        sigma_prop=5.0e6,
        sigma_yield=5.5e6,
        sigma_yield_2=6.0e6,
        eps_p_y1=0.004,
        eps_p_y2=0.015,
        K=100.0e6,
        n=0.2,
    )
    model.clear_boundary_conditions()
    model.add_boundary_condition(
        BoundaryCondition(
            "clamp",
            [1, 2],
            {
                "ux": 0.0,
                "uy": 0.0,
                "uz": 0.0,
                "rx": 0.0,
                "ry": 0.0,
                "rz": 0.0,
            },
        )
    )
    load = LoadCase("plastic_bending")
    load.add_nodal_load(4, forces=np.array([0.0, 0.0, 300.0]))
    nonlinear = solve_static_nonlinear(
        model,
        load,
        num_steps=4,
        max_iterations=30,
        tolerance=1.0e-6,
    )

    assert nonlinear.status in {"completed", "stopped_at_limit"}
    assert nonlinear.load_factor > 0.0
    assert nonlinear.element_states
    assert max(
        float(np.max(np.asarray(state.get("alpha", 0.0), dtype=float)))
        for state in nonlinear.element_states.values()
    ) > 0.0

    recovered = recover_stress_result(
        model,
        nonlinear_result=nonlinear,
        return_global=True,
    )

    assert recovered.provenance.mode == "material_history"
    assert recovered.provenance.analysis_context["kinematics"] == "von_karman"
    assert all(
        source == "committed_shell_layer_state"
        for source in recovered.provenance.per_element_source.values()
    )
    assert all(
        np.all(np.isfinite(stress["von_mises"]))
        for stress in recovered.element_stresses.values()
    )


def test_elastic_linear_patch_is_consistent_and_has_near_zero_indicator() -> None:
    model = generate_simple_panel_mesh(
        2.0, 1.0, 0.01, num_divisions_x=2, num_divisions_y=2
    )
    material = model.get_material("steel")
    eps = (1.2e-4, -0.4e-4, 0.7e-4)
    displacement = _global_membrane_displacement(model, *eps)
    expected_xx = material.elastic_modulus / (1.0 - material.poisson_ratio**2) * (
        eps[0] + material.poisson_ratio * eps[1]
    )

    recovered = recover_stress_result(
        model,
        displacement,
        patch_config=PatchRecoveryConfig(include_error_indicator=True),
    )
    patch = recovered.nodal_stresses

    assert recovered.provenance.mode == "elastic_only"
    assert patch is not None
    assert patch["qualified_node_ids"]
    assert patch["fallback_node_ids"] == []
    for values in patch["nodal"].values():
        assert values["global_xx_top"] == pytest.approx(expected_xx, rel=1.0e-10)
        assert values["global_xx_bot"] == pytest.approx(expected_xx, rel=1.0e-10)
    assert patch["error_indicator"]["status"] == "available"
    assert patch["error_indicator"]["relative"] < 1.0e-12


def test_patch_uses_committed_integration_point_stresses() -> None:
    model = generate_simple_panel_mesh(
        2.0, 1.0, 0.01, num_divisions_x=2, num_divisions_y=1
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    states = {
        element_id: _constant_shell_state(element, (180.0, 30.0, 0.0))
        for element_id, element in model.mesh.elements.items()
    }

    recovered = recover_stress_result(
        model,
        displacement,
        element_states=states,
        patch_config=PatchRecoveryConfig(),
    )
    patch = recovered.nodal_stresses

    assert recovered.provenance.mode == "material_history"
    assert patch is not None
    assert patch["fallback_node_ids"] == []
    assert all(
        value["global_xx_top"] == pytest.approx(180.0)
        for value in patch["nodal"].values()
    )


def test_patch_material_guard_and_condition_guard_fall_back_explicitly() -> None:
    model = generate_simple_panel_mesh(
        2.0, 1.0, 0.01, num_divisions_x=2, num_divisions_y=1
    )
    model.add_material("other", elastic_modulus=200.0e9, poisson_ratio=0.25)
    model.mesh.elements[2].material_name = "other"
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    # A zero field is enough: this test targets the qualification guards.
    discontinuous = recover_stress_result(
        model, displacement, patch_config=PatchRecoveryConfig()
    ).nodal_stresses
    assert discontinuous is not None
    assert any(
        diagnostics["reason"] == "material_discontinuity"
        for diagnostics in discontinuous["node_diagnostics"].values()
    )

    ill_conditioned = recover_stress_result(
        model,
        displacement,
        patch_config=PatchRecoveryConfig(
            condition_limit=1.01,
            material_continuity_required=False,
        ),
    ).nodal_stresses
    assert ill_conditioned is not None
    assert ill_conditioned["fallback_node_ids"]
    assert all(
        diagnostics["reason"] == "ill_conditioned_patch"
        for diagnostics in ill_conditioned["node_diagnostics"].values()
    )


def test_patch_recovery_is_deterministic() -> None:
    model = generate_simple_panel_mesh(
        2.0, 1.0, 0.01, num_divisions_x=2, num_divisions_y=2
    )
    displacement = _global_membrane_displacement(
        model, 0.8e-4, -0.2e-4, 0.3e-4
    )
    config = PatchRecoveryConfig(include_error_indicator=True)

    first = recover_stress_result(
        model, displacement, patch_config=config
    ).nodal_stresses
    second = recover_stress_result(
        model, displacement, patch_config=config
    ).nodal_stresses

    assert first is not None and second is not None
    assert first["qualified_node_ids"] == second["qualified_node_ids"]
    assert first["fallback_node_ids"] == second["fallback_node_ids"]
    assert first["node_diagnostics"] == second["node_diagnostics"]
    assert first["nodal"] == second["nodal"]
    assert first["error_indicator"] == second["error_indicator"]
