"""Physical seam ownership for a B2 round line on a planar shell patch."""

import numpy as np
import pytest
from scipy.integrate import quad

from anyfesolver import (
    BoundaryCondition, FEModel, LoadCase, BeamElement, QuadraticBeamElement,
    create_shell_element,
    solve_static_nonlinear,
)
from anyfesolver.general_contact import (
    ContactDefinition, ContactInteraction, ContactSolverSettings, ContactSurface,
    validate_general_contact,
)
from anyfesolver.general_contact.beam_shell_patch import _BeamShellPatchManager
from anyfesolver.general_contact.beam_shell_patch import _patch_windows
from anyfesolver.general_contact.overlap import PlanarFace


def _patch_model():
    model = FEModel("beam_across_internal_shell_seam")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node, point in enumerate((
        (-0.5, 0.0, 0.025), (0.5, 0.0, 0.025),
        (-1.0, -1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0), (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0), (1.0, 1.0, 0.0),
    ), 1):
        model.add_node(node, *point)
    model.add_element(1, BeamElement(
        1, [1, 2], "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    for eid, nodes in ((2, (3, 4, 5, 6)), (3, (4, 7, 8, 5))):
        model.add_element(eid, create_shell_element(
            eid, nodes, "elastic", thickness=0.02,
            formulation="e4-pl", reference_normal=(0, 0, 1),
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=0.01),
                  ContactSurface("skin", (2, 3), side="top")),
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


def _closed(model):
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2):
        trial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.006
    return trial


def _curved_patch_model(*, crossing_reference=False):
    model = FEModel("curved_b3_across_shell_seam_twice")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    beam_points = (((-0.5, 0.0, 0.025), (0.0, 0.0, 0.025), (0.5, 0.0, 0.025))
                   if crossing_reference else
                   ((-0.01, -0.5, 0.025), (-0.01, 0.0, 0.025),
                    (-0.01, 0.5, 0.025)))
    for node, point in enumerate((
        *beam_points,
        (-1.0, -1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0), (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0), (1.0, 1.0, 0.0),
    ), 1):
        model.add_node(node, *point)
    model.add_element(1, QuadraticBeamElement(
        1, [1, 2, 3], "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    for eid, nodes in ((2, (4, 5, 6, 7)), (3, (5, 8, 9, 6))):
        model.add_element(eid, create_shell_element(
            eid, nodes, "elastic", thickness=0.02,
            formulation="e4-pl", reference_normal=(0, 0, 1),
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=0.01),
                  ContactSurface("skin", (2, 3), side="top")),
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


def test_internal_seam_partitions_constant_gap_line_without_double_counting():
    model, definition, settings = _patch_model()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = _BeamShellPatchManager(model, definition, settings)
    trial = _closed(model)
    force, tangent, result = manager.evaluate(trial)
    # Independent potential for a constant 1 mm closure over a 1 m line:
    # 0.5 * 10,000 N/m² * (0.001 m)² * 1 m = 0.005 J.
    assert result.minimum_gap == pytest.approx(-0.001)
    assert result.line_contact_energy == pytest.approx(0.005, rel=1.0e-11)
    assert result.active_line_length == pytest.approx(1.0)
    assert result.active_intervals == 2
    assert result.beam_force[2] == pytest.approx(10.0)
    assert result.shell_force[2] == pytest.approx(-10.0)
    np.testing.assert_allclose(np.asarray(result.beam_force) + result.shell_force,
                               np.zeros(3), atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(result.beam_moment) + result.shell_moment,
                               np.zeros(3), atol=1.0e-10)
    direction = np.zeros_like(trial)
    direction[model.mesh.dof_manager.get_node_dofs(2)[2]] = 1.0
    step = 1.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2 * step), tangent @ direction,
                               rtol=1.0e-5, atol=1.0e-6)
    common_shell = np.zeros_like(trial)
    for node in (3, 4, 5, 6, 7, 8):
        common_shell[model.mesh.dof_manager.get_node_dofs(node)[2]] = 1.0
    plus = manager.evaluate(trial + step * common_shell, tangent=False)[0]
    minus = manager.evaluate(trial - step * common_shell, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2 * step), tangent @ common_shell,
                               rtol=1.0e-5, atol=1.0e-6)

    moved = trial.copy()
    for node in (1, 2):
        moved[model.mesh.dof_manager.get_node_dofs(node)[0]] += 0.1
    shifted_force, _, shifted = manager.evaluate(moved, tangent=False)
    assert shifted.line_contact_energy == pytest.approx(result.line_contact_energy)
    assert shifted.beam_force[2] == pytest.approx(10.0)
    assert not np.allclose(shifted_force, force)
    assert manager.swept_increment_admissible(trial, moved, 0.0, 1.0)

    escaped = moved.copy()
    escaped[model.mesh.dof_manager.get_node_dofs(2)[0]] += 0.5
    with pytest.raises(ValueError, match="rim"):
        manager.evaluate(escaped, tangent=False)


def test_exact_seam_line_has_one_stable_owner_and_nonconforming_patch_fails():
    model, definition, _ = _patch_model()
    from anyfesolver.general_contact.overlap import reference_shell_face

    faces = tuple(reference_shell_face(model, 2, eid, "top") for eid in (2, 3))
    ids = ((3, 4, 5, 6), (4, 7, 8, 5))
    line = np.array(((0.0, -0.5, 0.025), (0.0, 0.5, 0.025)))
    windows, rim = _patch_windows(faces, ids, line, 0.02)
    assert len(windows) == 1
    assert windows[0] == pytest.approx((0, -1.0, 1.0))
    assert rim == pytest.approx(0.5)
    reversed_windows, _ = _patch_windows(faces[::-1], ids[::-1], line, 0.02)
    assert faces[windows[0][0]].feature_id == faces[::-1][reversed_windows[0][0]].feature_id
    with pytest.raises(ValueError, match="rim|connected|nonmanifold"):
        _patch_windows(faces, (ids[0], (9, 7, 8, 5)), line, 0.02)

    third = PlanarFace((2, 4), np.array((
        (1.0, -1.0, 0.01), (2.0, -1.0, 0.01),
        (2.0, 1.0, 0.01), (1.0, 1.0, 0.01),
    )))
    across_three = np.array(((-0.5, 0.0, 0.025), (1.5, 0.0, 0.025)))
    windows, rim = _patch_windows(
        (*faces, third), (*ids, (7, 9, 10, 8)), across_three, 0.02,
    )
    np.testing.assert_allclose(windows, ((0, -1.0, -0.5),
                                         (1, -0.5, 0.5),
                                         (2, 0.5, 1.0)), atol=1.0e-12)
    assert rim == pytest.approx(0.5)


def test_curved_b3_crosses_internal_seam_twice_with_exact_line_work():
    model, definition, settings = _curved_patch_model()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = _BeamShellPatchManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3):
        trial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.006
    trial[model.mesh.dof_manager.get_node_dofs(2)[0]] = 0.02
    _, tangent, result = manager.evaluate(trial)
    assert result.minimum_gap == pytest.approx(-0.001)
    assert result.line_contact_energy == pytest.approx(0.005, rel=1.0e-11)
    assert result.beam_force[2] == pytest.approx(10.0)
    assert result.shell_force[2] == pytest.approx(-10.0)
    assert result.active_intervals == 3
    np.testing.assert_allclose(np.asarray(result.beam_moment) + result.shell_moment,
                               np.zeros(3), atol=1.0e-10)
    direction = np.zeros_like(trial)
    direction[model.mesh.dof_manager.get_node_dofs(2)[0]] = 1.0
    step = 1.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2 * step), tangent @ direction,
                               rtol=1.0e-5, atol=1.0e-6)
    varying = trial.copy()
    varying[model.mesh.dof_manager.get_node_dofs(2)[2]] -= 0.0002
    varied = manager.evaluate(varying, tangent=False)[2]
    independent, _ = quad(
        lambda xi: 0.5 * 1.0e4 * (-0.001 - 0.0002 * (1.0 - xi * xi)) ** 2 * 0.5,
        -1.0, 1.0, epsabs=1.0e-14,
    )
    assert varied.line_contact_energy == pytest.approx(independent, rel=1.0e-10)
    escaped = trial.copy()
    escaped[model.mesh.dof_manager.get_node_dofs(2)[0]] = 0.8
    with pytest.raises(ValueError, match="rim"):
        manager.evaluate(escaped, tangent=False)


def test_b3_patch_coupled_static_and_restart():
    model, definition, settings = _curved_patch_model(crossing_reference=True)
    model.add_boundary_condition(BoundaryCondition(
        "shell_fixed", (4, 5, 6, 7, 8, 9),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "beam_root", (1,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "beam_guides", (2, 3),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press_b3_across_seam")
    load.add_nodal_load(3, forces=(0.0, 0.0, -0.1))
    full = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
        tolerance=1.0e-8,
    )
    assert full.status == "completed", (full.failure_reason, full.info.get("general_contact"))
    assert full.info["general_contact"]["gap_tolerance_met"]
    half = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
        tolerance=1.0e-8,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
        tolerance=1.0e-8,
        restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1.0e-7, atol=1.0e-9)


def test_patch_static_route_and_restart():
    model, definition, settings = _patch_model()
    model.add_boundary_condition(BoundaryCondition(
        "shell_fixed", (3, 4, 5, 6, 7, 8),
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
    load = LoadCase("press_beam_across_shell_seam")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    full = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
    )
    assert full.status == "completed", (full.failure_reason, full.info.get("general_contact"))
    assert full.info["general_contact"]["gap_tolerance_met"]
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
