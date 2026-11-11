"""Local interior line-to-physical-shell work controls."""

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_shell import InteriorBeamShellLine
from anyfesolver.general_contact.beam_shell import _BeamShellManager
from anyfesolver.general_contact.overlap import PlanarFace


def _triangle():
    return PlanarFace((2, 1), np.array((
        (-1.0, -1.0, 0.0), (2.0, -1.0, 0.0), (-1.0, 2.0, 0.0),
    )))


def _kernel(reference):
    return InteriorBeamShellLine(
        reference, _triangle(), np.array((0.0, 0.0, 1.0)),
        radius=0.1, line_stiffness=1000.0, rim_clearance=0.2,
        integration_relative_tolerance=1.0e-9, maximum_subdivisions=5,
    )


def test_b2_uniform_closure_transfers_analytic_line_force_and_wrench():
    beam = np.array(((0.0, 0.0, 0.2), (0.5, 0.0, 0.2)))
    kernel = _kernel(beam)
    current = beam.copy()
    current[:, 2] = 0.09
    state = kernel.evaluate(current, _triangle().corners)
    assert state.minimum_gap == pytest.approx(-0.01)
    assert state.active_reference_length == pytest.approx(0.5)
    assert state.energy == pytest.approx(0.025)
    np.testing.assert_allclose(np.sum(state.beam_residual, axis=0),
                               (0.0, 0.0, -5.0), atol=1.0e-11)
    np.testing.assert_allclose(np.sum(state.shell_residual, axis=0),
                               (0.0, 0.0, 5.0), atol=1.0e-11)
    total_moment = sum(np.cross(point, force) for point, force in zip(
        np.r_[current, _triangle().corners],
        np.r_[state.beam_residual, state.shell_residual],
    ))
    np.testing.assert_allclose(total_moment, np.zeros(3), atol=1.0e-11)


def test_b2_beam_and_deformable_s3_virtual_work():
    beam = np.array(((0.0, 0.0, 0.2), (0.5, 0.0, 0.2)))
    kernel = _kernel(beam)
    current = beam.copy()
    current[:, 2] = 0.09
    shell = _triangle().corners.copy()
    state = kernel.evaluate(current, shell)
    beam_direction = np.array(((0.2, -0.1, 0.3), (-0.15, 0.1, -0.2)))
    shell_direction = np.array(((0.05, 0.0, 0.1), (0.0, 0.02, -0.15),
                                (-0.02, -0.01, 0.05)))
    step = 1.0e-7
    plus = kernel.evaluate(current + step * beam_direction,
                           shell + step * shell_direction)
    minus = kernel.evaluate(current - step * beam_direction,
                            shell - step * shell_direction)
    numerical = (plus.energy - minus.energy) / (2 * step)
    analytic = (float(np.sum(state.beam_residual * beam_direction))
                + float(np.sum(state.shell_residual * shell_direction)))
    assert numerical == pytest.approx(analytic, rel=1.0e-6)


def test_b3_uses_curved_deformed_interior_and_rejects_rim_approach():
    reference = np.array(((0.0, 0.0, 0.2), (0.25, 0.0, 0.2),
                          (0.5, 0.0, 0.2)))
    kernel = _kernel(reference)
    current = reference.copy()
    current[:, 2] = (0.12, 0.08, 0.12)
    state = kernel.evaluate(current, _triangle().corners)
    assert state.minimum_gap == pytest.approx(-0.02)
    assert state.active_intervals == 1
    assert 0.0 < state.active_reference_length < kernel.reference_length
    np.testing.assert_allclose(
        np.sum(state.beam_residual + 0.0, axis=0)
        + np.sum(state.shell_residual, axis=0),
        np.zeros(3), atol=1.0e-9,
    )
    escaped = current.copy()
    escaped[1, 0] = 1.8
    with pytest.raises(ValueError, match="rim"):
        kernel.evaluate(escaped, _triangle().corners)


def test_bottom_side_and_common_rigid_transform_preserve_line_work():
    triangle = _triangle()
    reference = np.array(((0.0, 0.0, -0.2), (0.5, 0.0, -0.2)))
    current = reference.copy()
    current[:, 2] = -0.09
    lower = InteriorBeamShellLine(
        reference, triangle, np.array((0.0, 0.0, -1.0)),
        radius=0.1, line_stiffness=1000.0, rim_clearance=0.2,
        integration_relative_tolerance=1.0e-9, maximum_subdivisions=5,
    ).evaluate(current, triangle.corners)
    assert lower.minimum_gap == pytest.approx(-0.01)
    assert lower.energy == pytest.approx(0.025)
    np.testing.assert_allclose(np.sum(lower.beam_residual, axis=0),
                               (0.0, 0.0, 5.0), atol=1.0e-11)
    angle = 0.37
    rotation = np.array(((np.cos(angle), 0.0, np.sin(angle)),
                         (0.0, 1.0, 0.0),
                         (-np.sin(angle), 0.0, np.cos(angle))))
    shift = np.array((2.0, -3.0, 1.0))
    moved_triangle = PlanarFace(triangle.feature_id,
                                triangle.corners @ rotation.T + shift)
    transformed = InteriorBeamShellLine(
        reference @ rotation.T + shift, moved_triangle,
        np.array((0.0, 0.0, -1.0)) @ rotation.T,
        radius=0.1, line_stiffness=1000.0, rim_clearance=0.2,
        integration_relative_tolerance=1.0e-9, maximum_subdivisions=5,
    ).evaluate(current @ rotation.T + shift, moved_triangle.corners)
    assert transformed.energy == pytest.approx(lower.energy, rel=1.0e-12)
    np.testing.assert_allclose(transformed.beam_residual,
                               lower.beam_residual @ rotation.T,
                               atol=1.0e-10)


def test_planar_q4_uses_original_bilinear_face_for_line_work():
    face = PlanarFace((2, 1), np.array((
        (-1.0, -1.0, 0.0), (2.0, -1.0, 0.0),
        (2.0, 2.0, 0.0), (-1.0, 2.0, 0.0),
    )))
    beam = np.array(((0.0, 0.0, 0.2), (0.5, 0.0, 0.2)))
    kernel = InteriorBeamShellLine(
        beam, face, np.array((0.0, 0.0, 1.0)),
        radius=0.1, line_stiffness=1000.0, rim_clearance=0.2,
        integration_relative_tolerance=1.0e-9, maximum_subdivisions=5,
    )
    current = beam.copy()
    current[:, 2] = 0.09
    state = kernel.evaluate(current, face.corners)
    assert state.energy == pytest.approx(0.025)
    assert state.minimum_gap == pytest.approx(-0.01)
    # Choose an affine out-of-plane perturbation; a warped Q4 is outside
    # this development profile and is checked separately.
    direction = np.array(((0.1, -0.1, 0.2), (-0.1, 0.2, -0.05),
                          (0.02, -0.03, -0.3), (-0.02, 0.03, -0.05)))
    step = 1.0e-7
    plus = kernel.evaluate(current, face.corners + step * direction).energy
    minus = kernel.evaluate(current, face.corners - step * direction).energy
    assert (plus - minus) / (2 * step) == pytest.approx(
        float(np.sum(state.shell_residual * direction)), rel=1.0e-6,
    )


def _coupled_model():
    model = FEModel("interior_round_beam_shell")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node, point in enumerate((
        (0.0, 0.0, 0.025), (0.5, 0.0, 0.025),
        (-1.0, -1.0, 0.0), (2.0, -1.0, 0.0), (-1.0, 2.0, 0.0),
    ), 1):
        model.add_node(node, *point)
    model.add_element(1, BeamElement(
        1, [1, 2], "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    model.add_element(2, create_shell_element(
        2, (3, 4, 5), "elastic", thickness=0.02,
        formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
        reference_surface_offset=0.0,
    ))
    definition = ContactDefinition(
        surfaces=(
            ContactSurface("tube", (1,), beam_radius=0.01),
            ContactSurface("skin", (2,), side="top"),
        ),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "line_skin", "tube", "skin", None, 0.002,
            line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", beam_shell_rim_clearance=0.02,
        beam_shell_difference_step=1.0e-8,
        line_integration_relative_tolerance=1.0e-8,
    )
    return model, definition, settings


def _coupled_q4_model():
    model = FEModel("interior_round_beam_q4")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node, point in enumerate((
        (0.0, 0.0, 0.025), (0.5, 0.0, 0.025),
        (-1.0, -1.0, 0.0), (2.0, -1.0, 0.0),
        (2.0, 2.0, 0.0), (-1.0, 2.0, 0.0),
    ), 1):
        model.add_node(node, *point)
    model.add_element(1, BeamElement(
        1, [1, 2], "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    model.add_element(2, create_shell_element(
        2, (3, 4, 5, 6), "elastic", thickness=0.02,
        formulation="e4-pl", reference_normal=(0, 0, 1),
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=0.01),
                  ContactSurface("skin", (2,), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "line_skin", "tube", "skin", None, 0.002,
            line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", beam_shell_rim_clearance=0.02,
        beam_shell_difference_step=1.0e-8,
        line_integration_relative_tolerance=1.0e-8,
    )
    return model, definition, settings


def test_coupled_b2_s3_force_work_tangent_and_rim_rejection():
    model, definition, settings = _coupled_model()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = _BeamShellManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2):
        trial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.006
    force, tangent, summary = manager.evaluate(trial)
    assert summary.minimum_gap == pytest.approx(-0.001)
    assert summary.active_line_length == pytest.approx(0.5)
    assert summary.line_contact_energy == pytest.approx(0.0025)
    assert summary.beam_force[2] == pytest.approx(5.0)
    assert summary.shell_force[2] == pytest.approx(-5.0)
    np.testing.assert_allclose(
        np.asarray(summary.beam_moment) + summary.shell_moment,
        np.zeros(3), atol=1.0e-9,
    )
    direction = np.zeros_like(trial)
    direction[model.mesh.dof_manager.get_node_dofs(2)[2]] = 1.0
    step = 1.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2 * step),
                               tangent @ direction, rtol=1.0e-5, atol=1.0e-6)
    # The shell reference plane is fixed in this profile, but the tangent
    # still needs the physical top-skin geometry derivative.
    shell_direction = np.zeros_like(trial)
    shell_direction[model.mesh.dof_manager.get_node_dofs(3)[3]] = 1.0
    plus_force = manager._assemble(
        trial + step * shell_direction, enforce_motion=False,
    )[0]
    minus_force = manager._assemble(
        trial - step * shell_direction, enforce_motion=False,
    )[0]
    np.testing.assert_allclose(
        (plus_force - minus_force) / (2 * step),
        tangent @ shell_direction, rtol=1.0e-5, atol=1.0e-6,
    )
    escaped = trial.copy()
    escaped[model.mesh.dof_manager.get_node_dofs(2)[0]] = 0.49
    with pytest.raises(ValueError, match="rim"):
        manager.evaluate(escaped, tangent=False)
    assert manager.admissible_step_fraction(trial, escaped) < 1.0


def test_interior_slide_preserves_potential_and_swept_side():
    model, definition, settings = _coupled_model()
    manager = _BeamShellManager(model, definition, settings)
    closed = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2):
        closed[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.006
    translated = closed.copy()
    for node in (1, 2):
        translated[model.mesh.dof_manager.get_node_dofs(node)[0]] = 0.1
    first = manager.evaluate(closed, tangent=False)[2]
    second = manager.evaluate(translated, tangent=False)[2]
    assert second.line_contact_energy == pytest.approx(first.line_contact_energy)
    assert second.beam_force[2] == pytest.approx(first.beam_force[2])
    np.testing.assert_allclose(
        np.asarray(second.beam_force) + second.shell_force,
        np.zeros(3), atol=1.0e-9,
    )
    np.testing.assert_allclose(
        np.asarray(second.beam_moment) + second.shell_moment,
        np.zeros(3), atol=1.0e-9,
    )
    assert manager.swept_increment_admissible(closed, translated, 0.0, 1.0)
    penetrated = closed.copy()
    for node in (1, 2):
        penetrated[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    assert not manager.swept_increment_admissible(closed, penetrated, 0.0, 1.0)
    with pytest.raises(ValueError, match="initial penetration"):
        _BeamShellManager(model, definition, settings, initial_displacement=closed)


def test_newton_pose_outside_budget_can_only_move_toward_opening():
    model, definition, settings = _coupled_model()
    manager = _BeamShellManager(model, definition, settings)
    outside = np.zeros(model.mesh.dof_manager.total_dofs)
    opening = outside.copy()
    closing = outside.copy()
    for node in (1, 2):
        dof = model.mesh.dof_manager.get_node_dofs(node)[2]
        outside[dof] = -0.0070000002
        opening[dof] = -0.006
        closing[dof] = -0.008
    assert manager.admissible_step_fraction(outside, opening) == 1.0
    assert manager.admissible_step_fraction(outside, closing) == 0.0
    with pytest.raises(ValueError, match="increment start"):
        manager.swept_increment_admissible(outside, opening, 0.0, 1.0)


def test_interior_beam_shell_static_route():
    model, definition, settings = _coupled_model()
    model.add_boundary_condition(BoundaryCondition(
        "shell_fixed", (3, 4, 5),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "beam_root", (1,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "beam_tip_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press_beam_into_skin")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
    )
    assert result.status == "completed", (
        result.failure_reason, result.load_factor,
        result.info.get("general_contact"),
    )
    contact = result.info["general_contact"]
    assert contact["gap_tolerance_met"]
    assert contact["beam_force"][2] > 0.0
    assert contact["shell_force"][2] == pytest.approx(-contact["beam_force"][2])
    half = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
        max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1.0e-7, atol=1.0e-9)


def test_planar_q4_interior_static_route_restart_and_warp_rejection():
    model, definition, settings = _coupled_q4_model()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert "Q4_INTERIOR" in report.profile
    manager = _BeamShellManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2):
        trial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.006
    force, tangent, summary = manager.evaluate(trial)
    assert summary.minimum_gap == pytest.approx(-0.001)
    assert summary.beam_force[2] == pytest.approx(5.0)
    assert summary.shell_force[2] == pytest.approx(-5.0)
    direction = np.zeros_like(trial)
    direction[model.mesh.dof_manager.get_node_dofs(2)[2]] = 1.0
    step = 1.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose(
        (plus - minus) / (2 * step), tangent @ direction,
        rtol=1.0e-5, atol=1.0e-6,
    )
    common_shell = np.zeros_like(trial)
    for node in (3, 4, 5, 6):
        common_shell[model.mesh.dof_manager.get_node_dofs(node)[2]] = 1.0
    plus = manager.evaluate(trial + step * common_shell, tangent=False)[0]
    minus = manager.evaluate(trial - step * common_shell, tangent=False)[0]
    np.testing.assert_allclose(
        (plus - minus) / (2 * step), tangent @ common_shell,
        rtol=1.0e-5, atol=1.0e-6,
    )
    model.add_boundary_condition(BoundaryCondition(
        "shell_fixed", (3, 4, 5, 6),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "beam_root", (1,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "beam_tip_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press_beam_into_q4")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    full = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
    )
    assert full.status == "completed", full.failure_reason
    half = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
        max_load_factor=1.0, restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1.0e-7, atol=1.0e-9)

    warped_model, warped_definition, warped_settings = _coupled_q4_model()
    warped_model.mesh.nodes[5].z += 1.0e-3
    warped = validate_general_contact(warped_model, warped_definition,
                                      warped_settings)
    assert not warped.valid
    assert any("planar" in issue for issue in warped.issues)


def test_b3_s3_interior_manager_uses_quadratic_curve_and_rejects_nonuniform_shell_motion():
    model = FEModel("quadratic_beam_shell")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node, point in enumerate((
        (0.0, 0.0, 0.025), (0.25, 0.0, 0.025), (0.5, 0.0, 0.025),
        (-1.0, -1.0, 0.0), (2.0, -1.0, 0.0), (-1.0, 2.0, 0.0),
    ), 1):
        model.add_node(node, *point)
    model.add_element(1, QuadraticBeamElement(
        1, [1, 2, 3], "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    model.add_element(2, create_shell_element(
        2, (4, 5, 6), "elastic", thickness=0.02,
        formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
        reference_surface_offset=0.0,
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=0.01),
                  ContactSurface("skin", (2,), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "line_skin", "tube", "skin", None, 0.002,
            line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", beam_shell_rim_clearance=0.02,
        beam_shell_difference_step=1.0e-8,
        line_integration_relative_tolerance=1.0e-8,
    )
    manager = _BeamShellManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    trial[model.mesh.dof_manager.get_node_dofs(1)[2]] = -0.004
    trial[model.mesh.dof_manager.get_node_dofs(2)[2]] = -0.008
    trial[model.mesh.dof_manager.get_node_dofs(3)[2]] = -0.004
    force, tangent, summary = manager.evaluate(trial)
    assert summary.minimum_gap == pytest.approx(-0.003)
    assert summary.active_intervals == 1
    assert 0.0 < summary.active_line_length < 0.5
    np.testing.assert_allclose(
        np.asarray(summary.beam_force) + summary.shell_force,
        np.zeros(3), atol=1.0e-9,
    )
    direction = np.zeros_like(trial)
    direction[model.mesh.dof_manager.get_node_dofs(2)[2]] = 1.0
    step = 1.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2 * step),
                               tangent @ direction, rtol=1.0e-5, atol=1.0e-6)
    warped = trial.copy()
    warped[model.mesh.dof_manager.get_node_dofs(4)[2]] = 1.0e-4
    with pytest.raises(ValueError, match="common shell translation"):
        manager.evaluate(warped, tangent=False)
