"""Declared round B2/B3 line contact against an analytic rigid plane."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    RigidContactBody, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_plane import _BeamPlaneManager


def _model(*, quadratic=False, vertical=False, end_caps="none"):
    model = FEModel("round_beam_plane")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    if vertical:
        points = ((0.0, 0.0, 0.015), (0.0, 0.0, 0.115))
    else:
        points = ((-0.5, 0.0, 0.015), (0.5, 0.0, 0.015))
    if quadratic:
        points = (points[0], tuple(0.5 * (a + b) for a, b in zip(*points)),
                  points[1])
    for node, point in enumerate(points, 1):
        model.add_node(node, *point)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4,
               "J": 1.0e-4}
    element = (QuadraticBeamElement(1, list(range(1, 4)), "elastic", section)
               if quadratic else BeamElement(1, [1, 2], "elastic", section))
    model.add_element(1, element)
    definition = ContactDefinition(
        surfaces=(ContactSurface(
            "round_line", (1,), beam_radius=0.01,
            beam_end_caps=end_caps,
        ),),
        rigid_bodies=(RigidContactBody(
            "plane", "plane", (0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
        ),),
        interactions=(ContactInteraction(
            "line_plane", "round_line", "plane", None, 0.002,
            line_stiffness=1.0e4,
            endpoint_stiffness=1.0e4 if end_caps == "hemisphere" else None,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general")
    return model, definition, settings


def test_b2_line_potential_force_tangent_and_gap_certificate():
    model, definition, settings = _model()
    report = validate_general_contact(model, definition, settings)
    assert report.valid and "ROUND_TUBE" in report.profile
    manager = _BeamPlaneManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2):
        trial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.006
    force, tangent, summary = manager.evaluate(trial)
    assert summary.minimum_gap == pytest.approx(-0.001)
    assert summary.active_line_length == pytest.approx(1.0)
    assert summary.candidate_cells == 1
    assert summary.active_intervals == 1
    assert summary.integration_points == 3
    assert summary.line_contact_energy == pytest.approx(0.005)
    assert summary.beam_force[2] == pytest.approx(10.0)
    assert summary.rigid_force[2] == pytest.approx(-10.0)
    assert summary.to_dict()["line_stiffness_N_per_m2"] == 1.0e4
    direction = np.zeros_like(trial)
    direction[model.mesh.dof_manager.get_node_dofs(2)[2]] = 1.0
    step = 1.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2.0 * step),
                               tangent @ direction, rtol=1.0e-7, atol=1.0e-8)
    crossed = trial.copy()
    crossed[model.mesh.dof_manager.get_node_dofs(2)[2]] = -0.02
    assert manager.admissible_step_fraction(np.zeros_like(trial), crossed) < 1.0
    assert not manager.swept_increment_admissible(
        np.zeros_like(trial), crossed, 0.0, 1.0,
    )


def test_b3_quadratic_interior_contact_and_hemisphere_ownership():
    model, definition, settings = _model(quadratic=True)
    manager = _BeamPlaneManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 3):
        trial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.004
    trial[model.mesh.dof_manager.get_node_dofs(2)[2]] = -0.006
    force, tangent, summary = manager.evaluate(trial)
    assert summary.minimum_gap == pytest.approx(-0.001)
    assert summary.active_line_length == pytest.approx(2.0**-0.5)
    assert summary.active_intervals == 1
    root = 2.0**-0.5
    a, c = 0.002, -0.001
    independent_energy = 0.5 * 1.0e4 * 0.5 * 2.0 * (
        a * a * root**5 / 5.0 + 2.0 * a * c * root**3 / 3.0
        + c * c * root
    )
    independent_force = -1.0e4 * 0.5 * 2.0 * (
        a * root**3 / 3.0 + c * root
    )
    assert summary.line_contact_energy == pytest.approx(independent_energy, rel=1.0e-12)
    assert summary.beam_force[2] == pytest.approx(independent_force, rel=1.0e-12)
    assert summary.beam_force[2] > 0.0
    assert summary.active_endpoints == 0
    direction = np.zeros_like(trial)
    direction[model.mesh.dof_manager.get_node_dofs(2)[2]] = 1.0
    step = 1.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2.0 * step),
                               tangent @ direction, rtol=1.0e-6, atol=1.0e-8)

    vertical, cap_definition, cap_settings = _model(vertical=True,
                                                    end_caps="hemisphere")
    cap_manager = _BeamPlaneManager(vertical, cap_definition, cap_settings)
    cap_trial = np.zeros(vertical.mesh.dof_manager.total_dofs)
    cap_trial[vertical.mesh.dof_manager.get_node_dofs(1)[2]] = -0.006
    _cap_force, cap_tangent, cap_summary = cap_manager.evaluate(cap_trial)
    assert cap_summary.active_endpoints == 1
    assert cap_summary.endpoint_contact_energy == pytest.approx(0.005)
    assert cap_summary.to_dict()["endpoint_stiffness_N_per_m"] == 1.0e4
    cap_direction = np.zeros_like(cap_trial)
    cap_direction[vertical.mesh.dof_manager.get_node_dofs(1)[2]] = 1.0
    cap_step = 1.0e-7
    cap_plus = cap_manager.evaluate(cap_trial + cap_step * cap_direction,
                                    tangent=False)[0]
    cap_minus = cap_manager.evaluate(cap_trial - cap_step * cap_direction,
                                     tangent=False)[0]
    np.testing.assert_allclose((cap_plus - cap_minus) / (2.0 * cap_step),
                               cap_tangent @ cap_direction,
                               rtol=1.0e-7, atol=1.0e-7)


def test_b2_line_contact_runs_through_displacement_solver_and_checkpoint():
    model, definition, settings = _model()
    model.add_boundary_condition(BoundaryCondition(
        "root", (1,), {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "tip_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("tip_drive")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2, dof="uz", target_displacement=-0.006,
        ),
        num_steps=4, max_iterations=18, emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["active_line_length"] > 0.0
    assert summary["gap_tolerance_met"]
    assert summary["minimum_gap"] == pytest.approx(-0.001)
    assert summary["beam_force"][2] == pytest.approx(-summary["rigid_force"][2])
    assert result.restart_checkpoint["version"] == 2
    changed_radius = replace(definition, surfaces=(replace(
        definition.surfaces[0], beam_radius=0.012,
    ),))
    with pytest.raises(ValueError, match="restart|incompatible|contract"):
        solve_static_nonlinear(
            model, load, contact=changed_radius, contact_settings=settings,
            control="displacement",
            displacement_control=DisplacementControl(
                node_id=2, dof="uz", target_displacement=-0.006,
            ),
            num_steps=4, max_iterations=18,
            restart_checkpoint=result.to_restart_checkpoint(),
        )


def test_b3_deformed_curve_runs_through_static_solver():
    model, definition, settings = _model(quadratic=True)
    model.add_boundary_condition(BoundaryCondition(
        "ends", (1, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "middle_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("midpoint_drive")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2, dof="uz", target_displacement=-0.006,
        ),
        num_steps=4, max_iterations=18,
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["minimum_gap"] == pytest.approx(-0.001)
    assert summary["active_line_length"] == pytest.approx((1.0 / 6.0)**0.5)
    assert summary["gap_tolerance_met"]
    assert summary["beam_force"][2] > 0.0


def test_b2_line_contact_runs_through_force_control():
    model, definition, settings = _model()
    model.add_boundary_condition(BoundaryCondition(
        "root", (1,), {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "tip_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("tip_force")
    load.add_nodal_load(2, forces=(0.0, 0.0, -0.2))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=8, max_iterations=20, min_step_fraction=0.05,
    )
    assert result.status == "completed", (
        result.failure_reason, result.load_factor,
        result.info.get("general_contact"),
    )
    summary = result.info["general_contact"]
    assert summary["active_line_length"] > 0.0
    assert summary["gap_tolerance_met"]


def test_beam_plane_rejects_missing_endpoint_law_and_initial_interference():
    model, definition, settings = _model(end_caps="hemisphere")
    missing_geometry = replace(definition, surfaces=(replace(
        definition.surfaces[0], beam_radius=None, beam_end_caps="none",
    ),))
    assert not validate_general_contact(model, missing_geometry, settings).valid
    missing = replace(definition, interactions=(replace(
        definition.interactions[0], endpoint_stiffness=None,
    ),))
    report = validate_general_contact(model, missing, settings)
    assert not report.valid
    assert any("endpoint_stiffness" in issue for issue in report.issues)
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    initial[model.mesh.dof_manager.get_node_dofs(2)[2]] = -0.02
    with pytest.raises(ValueError, match="initial penetration"):
        _BeamPlaneManager(model, definition, settings, initial)
    model.add_material("elastic", 1.0e3, 0.3, yield_stress=10.0)
    nonlinear = validate_general_contact(model, definition, settings)
    assert not nonlinear.valid
    assert any("isotropic elastic" in issue for issue in nonlinear.issues)


def test_hemisphere_endpoint_is_not_assigned_to_a_beam_junction():
    model, definition, settings = _model(vertical=True, end_caps="hemisphere")
    model.add_node(3, 0.0, 0.0, 0.215)
    model.add_element(2, BeamElement(
        2, [2, 3], "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    manager = _BeamPlaneManager(model, definition, settings)
    assert set(manager.endpoints) == {1}


def test_common_rigid_translation_preserves_beam_plane_work_and_wrench():
    model, definition, settings = _model(quadratic=True)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    trial[model.mesh.dof_manager.get_node_dofs(2)[2]] = -0.006
    base_force, _tangent, base = _BeamPlaneManager(
        model, definition, settings,
    ).evaluate(trial)
    moved, moved_definition, moved_settings = _model(quadratic=True)
    shift = np.array([0.11, -0.07, 0.23])
    for node in moved.mesh.nodes.values():
        node.x += shift[0]
        node.y += shift[1]
        node.z += shift[2]
    moved_definition = replace(moved_definition, rigid_bodies=(replace(
        moved_definition.rigid_bodies[0], point=tuple(shift),
    ),))
    moved_force, _tangent, shifted = _BeamPlaneManager(
        moved, moved_definition, moved_settings,
    ).evaluate(trial)
    np.testing.assert_allclose(moved_force, base_force, atol=1.0e-10)
    assert shifted.line_contact_energy == pytest.approx(base.line_contact_energy)
    np.testing.assert_allclose(
        shifted.beam_moment,
        np.asarray(base.beam_moment) + np.cross(shift, base.beam_force),
        atol=1.0e-10,
    )


def test_b3_thin_interval_meets_radius_scaled_gap_budget_and_tighter_one_fails():
    model, definition, settings = _model(quadratic=True)
    model.add_boundary_condition(BoundaryCondition(
        "ends", (1, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "middle_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("thin_interval_drive")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))

    def solve(gap_tolerance):
        interaction = replace(definition.interactions[0],
                              gap_tolerance=gap_tolerance)
        selected = replace(definition, interactions=(interaction,))
        return solve_static_nonlinear(
            model, load, contact=selected, contact_settings=settings,
            control="displacement",
            displacement_control=DisplacementControl(
                node_id=2, dof="uz", target_displacement=-0.005008,
            ),
            num_steps=4, max_iterations=18,
        )

    accepted = solve(1.0e-5)  # 10^-3 of the declared 10 mm radius.
    assert accepted.status == "completed", accepted.failure_reason
    summary = accepted.info["general_contact"]
    assert summary["minimum_gap"] == pytest.approx(-8.0e-6, abs=1.0e-12)
    assert summary["active_line_length"] == pytest.approx(
        (8.0e-6 / 0.005008)**0.5, rel=1.0e-10,
    )
    assert summary["gap_tolerance_met"]
    rejected = solve(5.0e-6)
    assert rejected.status == "contact_gap_exceeded"
    assert len(rejected.steps) < len(accepted.steps)
