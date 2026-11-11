"""Ordinary dead nodal force control uses the same coupled contact owner."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear
from anyfesolver.boundary import BoundaryCondition, LoadCase
from scripts.general_contact_d1_free_target_press_probe import build_free_target_case
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def _case():
    model, definition, settings, program = build_case(4)
    settings = replace(
        settings, enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=0.1,
        monolithic_linear_solver="block_response_dev",
        augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
    )
    load = LoadCase("contact force")
    for item in program.nodal_forces:
        assert item.dof == "uz"
        load.add_nodal_load(item.node_id,
                            forces=np.array([0.0, 0.0, item.values_n[-1]]))
    return model, definition, settings, program, load


def test_ordinary_force_matches_controlled_path_and_split_restart():
    original_model, definition, settings, program, _ = _case()
    controlled = solve_static_nonlinear(
        original_model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
    )
    assert controlled.status == "completed"

    ordinary_model, definition, settings, _program, load = _case()
    ordinary_model.add_boundary_condition(BoundaryCondition(
        "source_x_support", (1, 6), {"ux": 0.0},
    ))
    ordinary = solve_static_nonlinear(
        ordinary_model, load_case=load, num_steps=2,
        contact=definition, contact_settings=settings, max_iterations=18,
        emit_restart_checkpoint=True, record_increment_snapshots=True,
    )
    assert ordinary.status == "completed", (
        ordinary.failure_reason, ordinary.info.get("contact_failure"),
    )
    assert [snapshot.step_index for snapshot in ordinary.snapshots] == [1, 2]
    np.testing.assert_array_equal(ordinary.snapshots[-1].displacements,
                                  ordinary.displacements)
    np.testing.assert_allclose(ordinary.displacements, controlled.displacements,
                               rtol=0, atol=1e-10)
    for name in ("integration_force", "minimum_gap", "numerical_pressure_pa"):
        np.testing.assert_allclose(
            ordinary.info["general_contact"][name],
            controlled.info["general_contact"][name], rtol=0, atol=1e-9,
        )

    half = solve_static_nonlinear(
        ordinary_model, load_case=load, num_steps=2, max_load_factor=0.5,
        contact=definition, contact_settings=settings, max_iterations=18,
        emit_restart_checkpoint=True, record_increment_snapshots=True,
    )
    assert half.status == "completed" and len(half.steps) == 1
    assert [snapshot.step_index for snapshot in half.snapshots] == [1]
    resumed = solve_static_nonlinear(
        ordinary_model, load_case=load, num_steps=2,
        contact=definition, contact_settings=settings, max_iterations=18,
        restart_checkpoint=half.to_restart_checkpoint(),
        record_increment_snapshots=True,
    )
    assert resumed.status == "completed"
    assert [snapshot.step_index for snapshot in resumed.snapshots] == [2]
    np.testing.assert_allclose(resumed.displacements, ordinary.displacements,
                               rtol=0, atol=1e-10)


def test_ordinary_force_rejects_moments_and_other_load_types():
    model, definition, settings, _program, load = _case()
    load.add_nodal_load(5, moments=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="translation forces only"):
        solve_static_nonlinear(model, load_case=load, num_steps=2,
                               contact=definition, contact_settings=settings)


def test_deformable_s3_target_ordinary_force_matches_controlled_press():
    """Compare two ordinary static interfaces on the same deformable pair."""
    def free_target_case():
        return build_free_target_case(
            4, steps_per_segment=2, four_phase=False,
            enforcement_solver="monolithic_active_set",
        )

    controlled_model, definition, settings, program = free_target_case()
    controlled = solve_static_nonlinear(
        controlled_model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
    )
    assert controlled.status == "completed", controlled.failure_reason

    ordinary_model, definition, settings, program = free_target_case()
    ordinary_model.add_boundary_condition(BoundaryCondition(
        "source_x_support", (1, 6), {"ux": 0.0},
    ))
    load = LoadCase("deformable target press")
    for item in program.nodal_forces:
        assert item.dof == "uz"
        load.add_nodal_load(item.node_id, forces=np.array(
            [0.0, 0.0, item.values_n[-1]], dtype=float,
        ))
    ordinary = solve_static_nonlinear(
        ordinary_model, load_case=load, num_steps=2,
        contact=definition, contact_settings=settings, max_iterations=18,
    )
    assert ordinary.status == "completed", (
        ordinary.failure_reason, ordinary.info.get("contact_failure"),
    )
    np.testing.assert_allclose(
        ordinary.displacements, controlled.displacements,
        rtol=0.0, atol=1.0e-10,
    )
    for name in ("integration_force", "minimum_gap", "numerical_pressure_pa"):
        np.testing.assert_allclose(
            ordinary.info["general_contact"][name],
            controlled.info["general_contact"][name],
            rtol=0.0, atol=1.0e-9,
        )
    assert ordinary.info["general_contact"]["minimum_gap"] >= -5.0e-5
    assert ordinary.info["general_contact"]["integration_force"][2] > 0.0
    target_free_uz = ordinary_model.mesh.dof_manager.get_node_dofs(12)[2]
    assert abs(ordinary.displacements[target_free_uz]) > 1.0e-10
