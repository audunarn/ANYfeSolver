"""Prescribed radial-tool translation through nonlinear static contact."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    NonlinearConvergenceSettings,
    RigidContactBody,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.sphere import _SphereShellManager


def _moving_sphere():
    model = FEModel("q4_moving_sphere")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, (x, y) in enumerate((
        (-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1),
    ), 1):
        model.add_node(node_id, x, y, -0.01)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.02,
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("skin", (1,), side="top"),),
        rigid_bodies=(RigidContactBody(
            "tool", "sphere", (0.0, 0.0, 5.002), radius=5.0,
            path_translation=(0.0, 0.0, -0.003),
        ),),
        interactions=(ContactInteraction(
            "tool_skin", "skin", "tool", 1.0e5, 4.0e-3,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general",
                                     maximum_shell_tilt=0.05)
    return model, definition, settings


def test_moving_sphere_relative_motion_work_and_contract():
    model, definition, settings = _moving_sphere()
    report = validate_general_contact(model, definition, settings)
    assert report.valid and "TRANSLATING_SPHERE" in report.profile
    manager = _SphereShellManager(model, definition, settings)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in range(1, 5):
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 0.001
    manager.set_path_factor(1.0 / 3.0)
    moving_force, _K, moving_result = manager.evaluate(displacement)
    fixed = replace(definition, rigid_bodies=(replace(
        definition.rigid_bodies[0], path_translation=None,
    ),))
    fixed_manager = _SphereShellManager(model, fixed, settings)
    doubled = 2.0 * displacement
    fixed_force, _K, fixed_result = fixed_manager.evaluate(doubled)
    np.testing.assert_allclose(moving_force, fixed_force, atol=1.0e-9)
    assert moving_result.penalty_energy == pytest.approx(fixed_result.penalty_energy)
    assert manager.restart_contract()["body"]["path_translation"] == [0.0, 0.0, -0.003]
    assert "path_translation" not in fixed_manager.restart_contract()["body"]
    start = np.zeros_like(displacement)
    assert manager.swept_increment_admissible(start, displacement, 0.0, 1.0 / 3.0)


def test_moving_sphere_s3_physical_skin_offset():
    model = FEModel("s3_moving_sphere")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, (x, y) in enumerate(((-0.1, -0.1), (0.1, -0.1),
                                      (0.0, 0.2)), 1):
        model.add_node(node_id, x, y, -0.005)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3), "elastic", formulation="e4-pl-s3-v2d",
        thickness=0.02, reference_normal=(0, 0, 1),
        reference_surface_offset=0.005,
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("skin", (1,), side="top"),),
        rigid_bodies=(RigidContactBody(
            "tool", "sphere", (0.0, 0.0, 5.002), radius=5.0,
            path_translation=(0.0, 0.0, -0.006),
        ),),
        interactions=(ContactInteraction(
            "tool_skin", "skin", "tool", 1.0e5, 5.0e-3,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general",
                                     maximum_shell_tilt=0.05)
    assert validate_general_contact(model, definition, settings).valid
    manager = _SphereShellManager(model, definition, settings)
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    manager.set_path_factor(1.0)
    force, _K, result = manager.evaluate(zero, tangent=False)
    assert result.minimum_gap == pytest.approx(-0.004, abs=1.0e-5)
    assert result.active_pressure_dofs > 0
    fixed = replace(definition, rigid_bodies=(replace(
        definition.rigid_bodies[0], path_translation=None,
    ),))
    fixed_manager = _SphereShellManager(model, fixed, settings)
    fixed_displacement = zero.copy()
    for node_id in (1, 2, 3):
        fixed_displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 0.006
    fixed_force, _K, fixed_result = fixed_manager.evaluate(
        fixed_displacement, tangent=False,
    )
    np.testing.assert_allclose(force, fixed_force, atol=1.0e-9)
    assert fixed_result.penalty_energy == pytest.approx(result.penalty_energy)


def test_moving_sphere_swept_through_crossing_and_admission():
    model, definition, settings = _moving_sphere()
    manager = _SphereShellManager(model, definition, settings)
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    crossing = replace(definition, rigid_bodies=(replace(
        definition.rigid_bodies[0], path_translation=(0.0, 0.0, -10.004),
    ),))
    crossing_manager = _SphereShellManager(model, crossing, settings)
    assert not crossing_manager.swept_increment_admissible(zero, zero, 0.0, 1.0)
    assert crossing_manager.swept_trial_cutbacks == 1
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition("normal_anchor", (1,),
                                                    {"uz": 0.0}))
    stopped = solve_static_nonlinear(
        model, LoadCase("fast_tool"), contact=crossing,
        contact_settings=settings, num_steps=1, min_step_fraction=0.125,
    )
    assert stopped.status == "contact_gap_exceeded"
    assert stopped.load_factor == 0.0
    assert stopped.info["general_contact"]["swept_trial_cutbacks"] > 0
    plane = replace(definition, rigid_bodies=(RigidContactBody(
        "tool", "plane", (0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0), path_translation=(0.0, 0.0, 0.001),
    ),))
    assert not validate_general_contact(model, plane, settings).valid


def test_moving_sphere_force_path_work_and_restart():
    model, definition, settings = _moving_sphere()
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "normal_anchor", (1,), {"uz": 0.0},
    ))
    load = LoadCase("tool_drive")
    fixed_steps = NonlinearConvergenceSettings(
        growth_factor=1.0, max_step_factor=1.0,
    )

    def solve(factor, checkpoint=None, contact=definition, steps=4):
        return solve_static_nonlinear(
            model, load, contact=contact, contact_settings=settings,
            max_load_factor=factor, num_steps=steps, max_iterations=18,
            convergence_settings=fixed_steps,
            restart_checkpoint=checkpoint, emit_restart_checkpoint=True,
        )

    full = solve(1.0, steps=8)
    assert full.status == "completed", (
        full.failure_reason, full.info.get("general_contact"),
    )
    contact = full.info["general_contact"]
    assert contact["active_pressure_dofs"] > 0
    assert contact["gap_tolerance_met"]
    assert contact["rigid_contact_work_trapezoid"] < 0.0
    assert contact["prescribed_actuator_work_trapezoid"] > 0.0
    assert contact["rigid_tool_position"][2] == pytest.approx(4.999)
    assert full.restart_checkpoint["version"] == 2
    half = solve(0.5)
    assert half.status == "completed", half.failure_reason
    resumed = solve(1.0, half.to_restart_checkpoint(), steps=8)
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1.0e-6, atol=1.0e-8)
    assert resumed.info["general_contact"]["rigid_contact_work_trapezoid"] == pytest.approx(
        contact["rigid_contact_work_trapezoid"], rel=1.0e-5, abs=1.0e-8,
    )
    changed_motion = replace(definition, rigid_bodies=(replace(
        definition.rigid_bodies[0], path_translation=(0.0, 0.0, -0.004),
    ),))
    with pytest.raises(ValueError, match="restart|incompatible|contract"):
        solve(1.0, half.to_restart_checkpoint(), contact=changed_motion)


def test_moving_cylinder_force_path():
    model, sphere, settings = _moving_sphere()
    cylinder = replace(sphere, rigid_bodies=(replace(
        sphere.rigid_bodies[0], kind="cylinder", normal=(0.0, 1.0, 0.0),
    ),))
    report = validate_general_contact(model, cylinder, settings)
    assert report.valid and "TRANSLATING_INFINITE_CYLINDER" in report.profile
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition("normal_anchor", (1,),
                                                    {"uz": 0.0}))
    result = solve_static_nonlinear(
        model, LoadCase("roller_drive"), contact=cylinder,
        contact_settings=settings, num_steps=4, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["active_pressure_dofs"] > 0
    assert summary["gap_tolerance_met"]
    assert summary["rigid_contact_work_trapezoid"] < 0.0
    np.testing.assert_allclose(summary["shell_force"],
                               -np.asarray(summary["rigid_force"]), atol=1.0e-9)
    assert result.restart_checkpoint["version"] == 2


def test_moving_sphere_displacement_path_uses_control_fraction():
    model, definition, settings = _moving_sphere()
    report = validate_general_contact(
        model, definition, settings, analysis_control="displacement",
    )
    assert report.valid and "DISPLACEMENT_STATIC" in report.profile
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition("normal_anchor", (1,),
                                                    {"uz": 0.0}))
    load = LoadCase("downward_control_load")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2, dof="uz", target_displacement=-0.0002,
        ),
        num_steps=4, max_iterations=18, emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["profile"] == report.profile
    assert summary["gap_tolerance_met"]
    assert summary["rigid_tool_position"][2] == pytest.approx(4.999)
    assert summary["rigid_contact_work_trapezoid"] < 0.0
    assert result.info["contact_path_factor"] == pytest.approx(1.0)
