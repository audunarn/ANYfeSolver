"""Fixed infinite-cylinder contact against physical Q4/S3 shell skins."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    RigidContactBody, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.sphere import _SphereShellManager


def _q4_cylinder():
    model = FEModel("q4_fixed_cylinder")
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
            "roller", "cylinder", (0.0, 0.0, 5.0),
            normal=(0.0, 2.0, 0.0), radius=5.0,
        ),),
        interactions=(ContactInteraction(
            "roller_skin", "skin", "roller", 1.0e5, 4.0e-3,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general",
                                     maximum_shell_tilt=0.05)
    return model, definition, settings


def test_cylinder_axis_contract_and_exact_admission():
    model, definition, settings = _q4_cylinder()
    assert definition.rigid_bodies[0].normal == (0.0, 1.0, 0.0)
    report = validate_general_contact(model, definition, settings)
    assert report.valid
    assert "CYLINDER" in report.profile
    with pytest.raises(ValueError, match="axis and radius"):
        RigidContactBody("invalid", "cylinder", (0, 0, 0), radius=1.0)
    with pytest.raises(ValueError, match="axis must be nonzero"):
        RigidContactBody("invalid", "cylinder", (0, 0, 0),
                         normal=(0, 0, 0), radius=1.0)
    wrong_side = replace(definition, surfaces=(replace(
        definition.surfaces[0], side="bottom",
    ),))
    assert not validate_general_contact(model, wrong_side, settings).valid


def test_cylinder_q4_work_tangent_axis_invariance_and_wrench():
    model, definition, settings = _q4_cylinder()
    manager = _SphereShellManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    axial = np.zeros_like(u)
    for node_id in range(1, 5):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        u[dofs[2]] = 0.003
        axial[dofs[1]] = 1.0
    force, tangent, result = manager.evaluate(u)
    assert result.active_pressure_dofs > 0
    assert result.minimum_gap == pytest.approx(-0.003, abs=1.0e-5)
    assert result.gap_tolerance_met
    np.testing.assert_allclose(np.asarray(result.shell_force)
                               + np.asarray(result.rigid_force), 0.0, atol=1.0e-9)
    np.testing.assert_allclose(np.asarray(result.shell_moment)
                               + np.asarray(result.rigid_moment), 0.0, atol=1.0e-9)
    translated_force, _K, translated = manager.evaluate(u + 0.01 * axial,
                                                         tangent=False)
    np.testing.assert_allclose(translated_force, force, atol=1.0e-9)
    assert translated.penalty_energy == pytest.approx(result.penalty_energy)
    assert float(force @ axial) == pytest.approx(0.0, abs=1.0e-10)
    np.testing.assert_allclose(tangent @ axial, 0.0, atol=1.0e-8)
    direction = np.linspace(-0.4, 0.6, len(u))
    h = 1.0e-7
    plus_force, _K, plus = manager.evaluate(u + h * direction, tangent=False)
    minus_force, _K, minus = manager.evaluate(u - h * direction, tangent=False)
    assert (plus.penalty_energy - minus.penalty_energy) / (2 * h) == pytest.approx(
        float(force @ direction), rel=1.0e-6, abs=1.0e-7,
    )
    np.testing.assert_allclose((plus_force - minus_force) / (2 * h),
                               tangent @ direction, rtol=1.0e-5, atol=1.0e-6)


def test_cylinder_s3_physical_offset_and_axis_origin_invariance():
    model = FEModel("s3_fixed_cylinder")
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
            "roller", "cylinder", (0, 0, 5),
            normal=(0, 1, 0), radius=5,
        ),),
        interactions=(ContactInteraction(
            "skin_roller", "skin", "roller", 1.0e5, 4.0e-3,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general",
                                     maximum_shell_tilt=0.05)
    assert validate_general_contact(model, definition, settings).valid
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in range(1, 4):
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 0.003
    manager = _SphereShellManager(model, definition, settings)
    force, _tangent, result = manager.evaluate(displacement, tangent=False)
    assert result.minimum_gap == pytest.approx(-0.003, abs=1.0e-5)
    assert result.active_pressure_dofs > 0
    shifted = replace(definition, rigid_bodies=(replace(
        definition.rigid_bodies[0], point=(0.0, 7.0, 5.0),
    ),))
    shifted_manager = _SphereShellManager(model, shifted, settings)
    shifted_force, _K, shifted_result = shifted_manager.evaluate(
        displacement, tangent=False,
    )
    np.testing.assert_allclose(shifted_force, force, atol=1.0e-9)
    assert shifted_result.minimum_gap == pytest.approx(result.minimum_gap)


def test_cylinder_swept_axis_crossing_is_cut_back():
    model, definition, settings = _q4_cylinder()
    manager = _SphereShellManager(model, definition, settings)
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = start.copy()
    for node_id in range(1, 5):
        end[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 10.0
    fraction = manager.admissible_step_fraction(start, end)
    assert 0.0 < fraction < 1.0e-3
    assert manager._certify_gap(start, fraction * end,
                                manager.gap_tolerance)[0] >= -manager.gap_tolerance
    with pytest.raises(ValueError, match="behind selected"):
        manager.evaluate(end, tangent=False)


def test_cylinder_static_displacement_path_and_restart():
    model, definition, settings = _q4_cylinder()
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press")
    for node_id in range(1, 5):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, 0.1))

    def solve(factor, checkpoint=None):
        return solve_static_nonlinear(
            model, load, contact=definition, contact_settings=settings,
            control="displacement",
            displacement_control=DisplacementControl(
                weighted_dofs={(node, "uz"): 0.25 for node in range(1, 5)},
                target_displacement=0.003,
            ),
            max_load_factor=factor, num_steps=2, max_iterations=18,
            restart_checkpoint=checkpoint, emit_restart_checkpoint=True,
        )

    full = solve(1.0)
    assert full.status == "completed", full.failure_reason
    assert full.info["general_contact"]["active_pressure_dofs"] > 0
    assert full.info["general_contact"]["gap_tolerance_met"]
    assert "CYLINDER" in full.info["general_contact"]["profile"]
    half = solve(0.5)
    assert half.status == "completed", half.failure_reason
    resumed = solve(1.0, checkpoint=half.to_restart_checkpoint())
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1.0e-6, atol=1.0e-8)
    changed_axis = replace(definition, rigid_bodies=(replace(
        definition.rigid_bodies[0], normal=(1.0, 0.0, 0.0),
    ),))
    with pytest.raises(ValueError, match="restart|incompatible|contract"):
        solve_static_nonlinear(
            model, load, contact=changed_axis, contact_settings=settings,
            control="displacement",
            displacement_control=DisplacementControl(
                weighted_dofs={(node, "uz"): 0.25 for node in range(1, 5)},
                target_displacement=0.003,
            ),
            max_load_factor=1.0, num_steps=2, max_iterations=18,
            restart_checkpoint=half.to_restart_checkpoint(),
        )


def test_cylinder_static_force_control():
    model, definition, settings = _q4_cylinder()
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "normal_anchor", (1,), {"uz": 0.0},
    ))
    load = LoadCase("press_to_cylinder")
    for node_id in (2, 3, 4):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, 0.5))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=4, max_iterations=18, emit_restart_checkpoint=True,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("general_contact"),
    )
    summary = result.info["general_contact"]
    assert summary["active_pressure_dofs"] > 0
    assert summary["gap_tolerance_met"]
    np.testing.assert_allclose(summary["shell_force"],
                               -np.asarray(summary["rigid_force"]), atol=1.0e-9)
    assert result.restart_checkpoint["version"] == 2
