"""Prescribed radial obstacle motion against classical round beam curves."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl,
    LoadCase, RigidContactBody, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_radial import _BeamRadialManager


def _moving_model(*, kind="cylinder", quadratic=False, travel=0.012):
    model = FEModel("moving_radial_round_beam")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    nodes = ((-0.5, 0.0, 0.03), (0.0, 0.0, 0.03), (0.5, 0.0, 0.03))
    selected = nodes if quadratic else (nodes[0], nodes[2])
    for identifier, point in enumerate(selected, 1):
        model.add_node(identifier, *point)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4}
    model.add_element(1, (QuadraticBeamElement if quadratic else BeamElement)(
        1, list(range(1, len(selected) + 1)), "elastic", section,
    ))
    body = RigidContactBody(
        "obstacle", kind, (0.0, 0.0, 0.0), radius=0.01,
        normal=(1.0, 0.0, 0.0) if kind == "cylinder" else None,
        path_translation=(0.0, 0.0, travel),
    )
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=0.01),),
        rigid_bodies=(body,),
        interactions=(ContactInteraction(
            "line_radial", "tube", "obstacle", None, 0.004,
            line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-7,
    )
    return model, definition, settings


def _shift(model, node_ids, dz):
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in node_ids:
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = dz
    return u


def test_moving_cylinder_preserves_relative_geometry_and_tool_work():
    model, definition, settings = _moving_model()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = _BeamRadialManager(model, definition, settings)
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    initial = manager.evaluate(zero, tangent=False)[2]
    assert initial.minimum_gap == pytest.approx(0.01)
    manager.set_path_factor(1.0)
    _force, tangent, final = manager.evaluate(zero)
    assert final.minimum_gap == pytest.approx(-0.002)
    assert final.line_contact_energy == pytest.approx(0.02, rel=1.0e-10)
    assert final.beam_force[2] == pytest.approx(20.0, rel=1.0e-10)
    assert final.rigid_force[2] == pytest.approx(-20.0, rel=1.0e-10)
    np.testing.assert_allclose(np.asarray(final.beam_moment) + final.rigid_moment,
                               np.zeros(3), atol=1.0e-9)
    common = _shift(model, (1, 2), 0.012)
    np.testing.assert_allclose(manager.evaluate(common, tangent=False)[2].minimum_gap,
                               initial.minimum_gap, atol=1.0e-12)
    assert tangent is not None
    total_work = sum(manager.tool_work_increment(
        zero, zero, i / 12.0, (i + 1) / 12.0,
    ) for i in range(12))
    # Independent work for a fixed beam: F_tool=-k(d-0.01)L after first touch.
    assert total_work == pytest.approx(-0.02, rel=1.0e-10)
    assert manager.path_factor == pytest.approx(1.0)
    assert manager.point[2] == pytest.approx(0.012)


def test_moving_sphere_detects_clear_endpoint_through_crossing():
    model, definition, settings = _moving_model(kind="sphere", travel=0.06)
    manager = _BeamRadialManager(model, definition, settings)
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    assert manager.evaluate(zero, tangent=False)[2].minimum_gap == pytest.approx(0.01)
    manager.set_path_factor(1.0)
    assert manager.evaluate(zero, tangent=False)[2].minimum_gap == pytest.approx(0.01)
    assert not manager.swept_increment_admissible(zero, zero, 0.0, 1.0)
    manager.set_path_factor(0.0)
    assert manager.point[2] == pytest.approx(0.0)


def test_moving_sphere_b3_curve_uses_current_tool_pose():
    model, definition, settings = _moving_model(kind="sphere", quadratic=True)
    manager = _BeamRadialManager(model, definition, settings)
    trial = _shift(model, (2,), -0.012)
    initial = manager.evaluate(trial, tangent=False)[2]
    assert initial.minimum_gap == pytest.approx(-0.002)
    manager.set_path_factor(1.0)
    moved = manager.evaluate(trial, tangent=False)[2]
    assert moved.minimum_gap < initial.minimum_gap
    shifted_beam = trial + _shift(model, (1, 2, 3), 0.012)
    aligned = manager.evaluate(shifted_beam, tangent=False)[2]
    assert aligned.minimum_gap == pytest.approx(initial.minimum_gap)
    assert aligned.line_contact_energy == pytest.approx(initial.line_contact_energy)


def test_moving_cylinder_coupled_path_restart_and_binding():
    model, definition, settings = _moving_model()
    model.add_boundary_condition(BoundaryCondition(
        "root", (1,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "tip_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("tool_and_tip")
    load.add_nodal_load(2, forces=(0.0, 0.0, 1.0))
    common = dict(contact=definition, contact_settings=settings,
                  control="displacement",
                  displacement_control=DisplacementControl(
                      node_id=2, dof="uz", target_displacement=0.004,
                  ), num_steps=8, max_iterations=20)
    full = solve_static_nonlinear(model, load, emit_restart_checkpoint=True, **common)
    assert full.status == "completed", (full.failure_reason, full.info.get("general_contact"))
    summary = full.info["general_contact"]
    assert summary["gap_tolerance_met"]
    assert summary["rigid_tool_position"][2] == pytest.approx(0.012)
    assert summary["beam_force"][2] == pytest.approx(-summary["rigid_force"][2])
    assert np.isfinite(summary["rigid_contact_work_trapezoid"])
    half = solve_static_nonlinear(
        model, load, max_load_factor=0.5, emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1.0e-6, atol=1.0e-8)
    changed = replace(definition, rigid_bodies=(replace(
        definition.rigid_bodies[0], path_translation=(0.0, 0.0, 0.011),
    ),))
    with pytest.raises(ValueError, match="restart|incompatible|contract"):
        solve_static_nonlinear(
            model, load, restart_checkpoint=half.to_restart_checkpoint(),
            **(common | {"contact": changed}),
        )


def test_moving_cylinder_force_control_advances_tool_and_records_work():
    model, definition, settings = _moving_model()
    model.add_boundary_condition(BoundaryCondition(
        "root", (1,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "tip_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("small_tip_force_and_tool")
    load.add_nodal_load(2, forces=(0.0, 0.0, 1.0e-5))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=8, max_iterations=20,
    )
    assert result.status == "completed", (result.failure_reason,
                                          result.info.get("general_contact"))
    contact = result.info["general_contact"]
    assert contact["gap_tolerance_met"]
    assert contact["rigid_tool_position"][2] == pytest.approx(0.012)
    assert contact["rigid_contact_work_trapezoid"] < 0.0
    assert contact["prescribed_actuator_work_trapezoid"] == pytest.approx(
        -contact["rigid_contact_work_trapezoid"]
    )


def test_moving_sphere_b3_coupled_force_route():
    model, definition, settings = _moving_model(kind="sphere", quadratic=True)
    model.add_boundary_condition(BoundaryCondition(
        "ends_fixed", (1, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "middle_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("sphere_tool_and_small_middle_force")
    load.add_nodal_load(2, forces=(0.0, 0.0, 1.0e-5))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=8, max_iterations=20,
    )
    assert result.status == "completed", (result.failure_reason,
                                          result.info.get("general_contact"))
    contact = result.info["general_contact"]
    assert contact["gap_tolerance_met"]
    assert contact["active_line_length"] > 0.0
    assert contact["rigid_tool_position"][2] == pytest.approx(0.012)
    assert contact["beam_force"][2] == pytest.approx(-contact["rigid_force"][2])
