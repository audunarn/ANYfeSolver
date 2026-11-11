"""Admission checks for the opt-in planar Q4/S3 interior slice."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import BoundaryCondition, create_shell_element, solve_static_nonlinear
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.control import SolveCancelled
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact import semismooth_path
from anyfesolver.general_contact.semismooth import linearize
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from examples.general_contact_d1_interior_controls_dev import (
    _independent_projected_gap_bound,
)
from scripts.general_contact_d1_interior_press_dev import build_interior_press
from scripts.general_contact_d1_target_displacement_dev import (
    build_case as build_target_displacement_case,
)
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def _free_map(model):
    model.apply_boundary_conditions()
    stiffness, _info = assemble_stiffness_matrix(model)
    zeros = np.zeros(model.mesh.dof_manager.total_dofs)
    return build_constraint_transformation(
        stiffness, zeros, model,
    )[2]


@pytest.mark.parametrize("width", (0.25, 0.0625))
@pytest.mark.parametrize("slope", (0.0, 1.0e-4))
def test_planar_q4_source_vertical_energy_tracks_q1_shear(width, slope):
    model = FEModel("planar_q4_vertical_energy_control")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node, point in enumerate(((0.0, 0.0, 0.0),
                                  (width, 0.0, slope*width),
                                  (width, width, slope*width),
                                  (0.0, width, 0.0)), 1):
        model.add_node(node, *point)
    normal = np.asarray((-slope, 0.0, 1.0))
    normal /= np.linalg.norm(normal)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.05,
        reference_normal=tuple(normal)))
    stiffness, _ = assemble_stiffness_matrix(model)
    vertical = [model.mesh.dof_manager.get_node_dofs(node)[2]
                for node in range(1, 5)]
    actual = stiffness[np.ix_(vertical, vertical)].toarray()
    shear_resultant = (5.0/6.0)*0.05*1.0e5/(2.0*(1.0+0.3))
    exact_q1 = np.asarray((
        (2/3, -1/6, -1/3, -1/6),
        (-1/6, 2/3, -1/6, -1/3),
        (-1/3, -1/6, 2/3, -1/6),
        (-1/6, -1/3, -1/6, 2/3),
    ))
    np.testing.assert_allclose(actual/shear_resultant, exact_q1,
                               rtol=0.0,
                               atol=1.0e-12 if slope == 0.0 else 1.0e-7)


@pytest.mark.parametrize("slope", (0.0, 1.0e-4, 0.1))
def test_planar_q4_physical_normal_energy_matches_local_q1_shear(slope):
    width = 0.25
    model = FEModel("planar_q4_normal_energy_control")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node, point in enumerate(((0., 0., 0.),
                                  (width, 0., slope*width),
                                  (width, width, slope*width),
                                  (0., width, 0.)), 1):
        model.add_node(node, *point)
    normal = np.asarray((-slope, 0., 1.))
    normal /= np.linalg.norm(normal)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.05,
        reference_normal=tuple(normal)))
    stiffness, _ = assemble_stiffness_matrix(model)
    test_space = np.zeros((stiffness.shape[0], 4))
    for column, node in enumerate(range(1, 5)):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        test_space[dofs[0], column] = normal[0]
        test_space[dofs[2], column] = normal[2]
    actual = test_space.T @ stiffness @ test_space
    shear_resultant = (5.0/6.0)*0.05*1.0e5/(2.0*(1.0+0.3))
    m1 = np.asarray(((1/3, 1/6), (1/6, 1/3)))
    k1 = np.asarray(((1., -1.), (-1., 1.)))
    corners = ((0, 0), (1, 0), (1, 1), (0, 1))
    hx = width*np.sqrt(1+slope*slope)
    expected = np.asarray([
        [shear_resultant*(
            width/hx*k1[ix, jx]*m1[iy, jy]
            + hx/width*m1[ix, jx]*k1[iy, jy])
         for jx, jy in corners]
        for ix, iy in corners
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_target_internal_s3_seam_is_covered_and_physical_rim_is_topological():
    model, definition, settings, _program = build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed", target_columns=2,
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.d1_physical_rim_node_ids == frozenset(
        (*range(1, 6), 6, 10, 11, 15, 16, 20, *range(21, 26))
    )
    assert manager.require_d1_coverage(*manager._positions(
        np.zeros(model.mesh.dof_manager.total_dofs),
    )) is None


def test_interior_trial_swept_cutback_respects_independent_gap_budget():
    model, definition, settings, _program = build_interior_press(
        source_support="perimeter", source_rows=4, source_columns=4,
        target_columns=1, target_rows=1,
        reference_slope=1.0e-4,
        load_shape="interior_distributed", slide_body="target",
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = start.copy()
    for row in range(1, 4):
        for column in range(1, 4):
            node = 1+row*5+column
            end[model.mesh.dof_manager.get_node_dofs(node)[2]] = -1.0e-3
    assert not manager.swept_increment_admissible(start, end, 0.0, 1.0)
    fraction = manager.admissible_step_fraction(start, end)
    # At an interior vertex the 40 µm opening reaches the −50 µm budget
    # after 90 µm of this 1 mm affine physical-skin trial.
    assert 0.0 < fraction <= 0.0900001
    assert manager.swept_increment_admissible(
        start, fraction*end, 0.0, fraction)
    independent_lower = _independent_projected_gap_bound(
        model, definition, fraction*end,
        definition.interactions[0].gap_tolerance,
    )
    assert independent_lower >= -definition.interactions[0].gap_tolerance


def test_saved_eight_s3_press_has_free_source_rotation_contact_tangent():
    saved = Path(__file__).resolve().parents[1] / (
        "docs/d1_review_support/d1_coupled_target8s3_bump_overlap_20260929.json"
    )
    pose = json.loads(saved.read_text(encoding="utf-8"))
    assert pose["status"] == "completed" and pose["accepted_steps"] == 2
    model, definition, settings, _program = build_interior_press(
        force_n=0.04, reference_slope=1.0e-4,
        source_support="perimeter", source_columns=8, source_rows=8,
        target_columns=2, target_rows=2,
        target_reference_full_left_clamp=True,
        load_shape="physical_bump", slide_body="target",
    )
    manager = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="overlap_local_dev"),
    )
    manager.set_normal_multipliers(
        pose["history"][-1]["contact"]["normal_multiplier_estimate_pa"])
    displacement = np.asarray(pose["displacements"])
    direction = model.mesh.dof_manager.get_node_dofs(40)[4]  # source ry
    force, tangent, center = manager.evaluate(displacement, tangent=True)
    h = 1.0e-5
    trial = displacement.copy()
    trial[direction] += h
    plus, _, plus_state = manager.evaluate(trial, tangent=False)
    trial[direction] -= 2*h
    minus, _, minus_state = manager.evaluate(trial, tangent=False)
    derived = np.asarray(tangent[:, direction].toarray()).reshape(-1)
    difference = (plus-minus)/(2*h)
    assert center.active_pressure_dofs == plus_state.active_pressure_dofs == (
        minus_state.active_pressure_dofs) == 1
    assert center.integration_triangles == plus_state.integration_triangles == (
        minus_state.integration_triangles)
    assert abs(force[direction]) > 1.0e-8  # resolved physical couple
    assert np.linalg.norm(difference) > 1.0e-4  # nonzero mapped work change
    assert np.linalg.norm(derived-difference)/np.linalg.norm(difference) < 1.0e-5


def test_saved_common_slope_press_has_consistent_small_source_couple():
    saved = Path(__file__).resolve().parents[1] / (
        "docs/d1_review_support/d1_sloped_source_couple_press_20260929.json"
    )
    pose = json.loads(saved.read_text(encoding="utf-8"))
    assert pose["status"] == "completed" and pose["accepted_steps"] == 2
    model, definition, settings, _program = build_interior_press(
        force_n=0.04, reference_slope=0.01,
        source_support="perimeter", source_columns=4, source_rows=4,
        target_columns=1, target_rows=1,
        load_shape="interior_distributed", slide_body="target",
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    manager.set_normal_multipliers(
        pose["history"][-1]["contact"]["normal_multiplier_estimate_pa"])
    displacement = np.asarray(pose["displacements"])
    force, tangent, center = manager.evaluate(displacement, tangent=True)
    free_source_nodes = (7, 8, 9, 12, 13, 14, 17, 18, 19)
    selected = max(free_source_nodes, key=lambda node: abs(
        force[model.mesh.dof_manager.get_node_dofs(node)[4]]))
    dofs = model.mesh.dof_manager.get_node_dofs(selected)
    # Common source/target slope leaves their normal traction nearly
    # collinear with the source director; it is a negative couple control.
    assert abs(force[dofs[4]]) < 1.0e-7
    h = 1.0e-5
    trial = displacement.copy()
    trial[dofs[4]] += h
    plus, _, plus_state = manager.evaluate(trial, tangent=False)
    plus_skin = manager._positions(trial)[0][
        manager.law.source_node_ids.index(selected)]
    trial[dofs[4]] -= 2*h
    minus, _, minus_state = manager.evaluate(trial, tangent=False)
    minus_skin = manager._positions(trial)[0][
        manager.law.source_node_ids.index(selected)]
    assert center.active_pressure_dofs == plus_state.active_pressure_dofs == (
        minus_state.active_pressure_dofs) == 1
    assert center.integration_triangles == plus_state.integration_triangles == (
        minus_state.integration_triangles)
    mapped_couple = float(force[dofs[:3]] @ ((plus_skin-minus_skin)/(2*h)))
    assert mapped_couple == pytest.approx(force[dofs[4]], rel=1e-5, abs=1e-10)
    predicted = np.asarray(tangent[:, dofs[4]].toarray()).reshape(-1)
    observed = (plus-minus)/(2*h)
    assert np.linalg.norm(observed) > 1e-4
    assert np.linalg.norm(predicted-observed)/np.linalg.norm(observed) < 1e-5


def test_partial_target_boundary_cut_fails_before_solve():
    model, definition, settings, _program = build_case(
        4, source_rows=2, target_right_x_m=1.3,
        full_width_target=False,
    )
    settings = replace(
        settings, augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        d1_interior_profile=True,
        d1_interior_coverage_tolerance_m2=1.0e-10,
        d1_interior_boundary_force_tolerance_n=1.0e-10,
    )
    with pytest.raises(ValueError, match="partial target coverage"):
        GeneralS3PatchManager(model, definition, settings)


def test_active_rim_pressure_rejects_but_interior_pressure_can_pass():
    model, definition, settings, _program = build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed",
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    transform = _free_map(model)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    _force, _tangent, certificate = manager.evaluate(displacement,
                                                       tangent=False)
    ids = certificate.source_pressure_node_ids
    support = np.asarray(certificate.reference_support_m2)

    def with_force(node):
        pressure = np.zeros(len(ids))
        index = ids.index(node)
        pressure[index] = 0.01 / support[index]
        return replace(certificate, numerical_pressure_pa=tuple(pressure))

    assert "physical source rim node 1" in manager.d1_interior_admission_issue(
        with_force(1), displacement, transform,
    )
    assert manager.d1_interior_admission_issue(
        with_force(13), displacement, transform,
    ) is None


def test_prescribed_normal_is_checked_through_physical_work_map():
    model, definition, settings, _program = build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed", reference_slope=0.0,
    )
    model.add_boundary_condition(BoundaryCondition(
        "center_normal_fixed", (13,), {"uz": 0.0},
    ))
    manager = GeneralS3PatchManager(model, definition, settings)
    transform = _free_map(model)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    _force, _tangent, certificate = manager.evaluate(displacement,
                                                       tangent=False)
    pressure = np.zeros(len(certificate.source_pressure_node_ids))
    index = certificate.source_pressure_node_ids.index(13)
    pressure[index] = 0.01 / certificate.reference_support_m2[index]
    with_center_pressure = replace(
        certificate, numerical_pressure_pa=tuple(pressure),
    )
    assert "prescribed-normal source node 13" in (
        manager.d1_interior_admission_issue(
            with_center_pressure, displacement, transform,
        )
    )


def test_guarded_active_press_restarts_and_binds_profile_settings():
    def case():
        return build_interior_press(
            source_support="perimeter", source_rows=4,
            load_shape="interior_distributed", force_n=0.04,
        )

    model, definition, settings, program = case()
    full = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert full.status == "completed"
    assert len(full.steps) == 2

    model, definition, settings, program = case()
    half = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed"
    assert len(half.steps) == 1
    assert half.restart_checkpoint is not None

    with pytest.raises(ValueError, match="incompatible"):
        model, definition, settings, program = case()
        solve_static_nonlinear(
            model, contact=definition, contact_settings=replace(
                settings, d1_interior_boundary_force_tolerance_n=2.0e-10,
            ), contact_path_program=program, max_iterations=18,
            restart_checkpoint=half.to_restart_checkpoint(),
        )

    model, definition, settings, program = case()
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed"
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["integration_force"],
        full.info["general_contact"]["integration_force"],
        rtol=0.0, atol=1.0e-10,
    )


def _guarded_left_edge_press_case():
    return build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed", force_n=0.04,
        reference_slope=1.0e-4, slide_body="target",
    )


def test_guarded_left_edge_press_cancel_after_step_keeps_model_reusable():

    class CancelBeforeSecondStep:
        def raise_if_cancelled(self, stage=""):
            if stage == "contact_path.step:2":
                raise SolveCancelled("left-edge development interruption",
                                     stage=stage)

    model, definition, settings, program = _guarded_left_edge_press_case()
    with pytest.raises(SolveCancelled,
                       match="left-edge development interruption"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            cancellation_token=CancelBeforeSecondStep(),
        )
    rerun = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    fresh_model, fresh_definition, fresh_settings, fresh_program = (
        _guarded_left_edge_press_case())
    fresh = solve_static_nonlinear(
        fresh_model, contact=fresh_definition, contact_settings=fresh_settings,
        contact_path_program=fresh_program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert rerun.status == fresh.status == "completed"
    np.testing.assert_allclose(rerun.displacements, fresh.displacements,
                               rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        rerun.info["general_contact"]["integration_force"],
        fresh.info["general_contact"]["integration_force"],
        rtol=0.0, atol=1.0e-10,
    )
    assert rerun.to_restart_checkpoint() == fresh.to_restart_checkpoint()


def test_guarded_left_edge_contact_exception_restores_model(monkeypatch):
    entered_second_step = [False]
    raised = [False]

    class MarkSecondStep:
        def raise_if_cancelled(self, stage=""):
            if stage == "contact_path.step:2":
                entered_second_step[0] = True

    original = semismooth_path.linearize

    def raise_after_contact(*args, **kwargs):
        work = original(*args, **kwargs)
        if entered_second_step[0] and not raised[0]:
            raised[0] = True
            raise RuntimeError("injected left-edge candidate failure")
        return work

    monkeypatch.setattr(semismooth_path, "linearize", raise_after_contact)
    model, definition, settings, program = _guarded_left_edge_press_case()
    with pytest.raises(RuntimeError,
                       match="injected left-edge candidate failure"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            cancellation_token=MarkSecondStep(),
        )
    assert raised[0]
    monkeypatch.setattr(semismooth_path, "linearize", original)

    rerun = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    fresh_model, fresh_definition, fresh_settings, fresh_program = (
        _guarded_left_edge_press_case())
    fresh = solve_static_nonlinear(
        fresh_model, contact=fresh_definition, contact_settings=fresh_settings,
        contact_path_program=fresh_program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert rerun.status == fresh.status == "completed"
    np.testing.assert_allclose(rerun.displacements, fresh.displacements,
                               rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        rerun.info["general_contact"]["integration_force"],
        fresh.info["general_contact"]["integration_force"],
        rtol=0.0, atol=1.0e-10,
    )
    assert rerun.to_restart_checkpoint() == fresh.to_restart_checkpoint()


def test_two_target_displacement_controls_bound_force_reference():
    model, definition, settings, program = build_target_displacement_case(
        normal_motion_m=-1.0e-5, force_reference_n=0.04,
    )
    with pytest.raises(ValueError,
                       match="force scale exceeds the contact-free elastic"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
        )


@pytest.mark.parametrize("slide_body", ["target_diagonal_dev", "target"])
def test_guarded_target_motion_open_recontact_split_restart(slide_body):
    def case():
        return build_interior_press(
            source_support="perimeter", source_rows=4,
            load_shape="interior_distributed", force_n=0.04,
            reference_slope=1.0e-4, four_phase=True,
            slide_body=slide_body,
        )

    model, definition, settings, program = case()
    full = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert full.status == "completed", full.failure_reason
    assert len(full.steps) == 8
    full_history = full.info["force_displacement_history"]
    assert full_history[3]["contact"]["active_pressure_dofs"] > 0
    assert full_history[5]["contact"]["active_pressure_dofs"] == 0
    assert full_history[7]["contact"]["active_pressure_dofs"] > 0

    model, definition, settings, program = case()
    opened = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        max_load_factor=0.75, emit_restart_checkpoint=True,
    )
    assert opened.status == "completed", opened.failure_reason
    assert len(opened.steps) == 6
    assert opened.restart_checkpoint is not None

    model, definition, settings, program = case()
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        restart_checkpoint=opened.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    assert len(resumed.steps) == 8
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=0.0, atol=1.0e-10)
    for resumed_row, full_row in zip(
            resumed.info["force_displacement_history"], full_history):
        np.testing.assert_allclose(
            resumed_row["contact"]["integration_force"],
            full_row["contact"]["integration_force"],
            rtol=0.0, atol=1.0e-10,
        )


def test_corrected_left_edge_slide_contact_directional_tangent():
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "general_contact_d1_left_edge_slide_pose.json").read_text())
    model, definition, settings, program = build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed", force_n=0.04,
        reference_slope=1.0e-4, four_phase=True,
        slide_body="target",
    )
    assert program.total_steps == 8
    nodal_force = np.asarray(fixture["nodal_force_n"])
    assert np.count_nonzero(nodal_force > 0.0) == 1
    manager = GeneralS3PatchManager(model, definition, settings)
    transform = _free_map(model)
    manager.restrict_tangent_to_reduced_variations(transform)
    displacement = np.asarray(fixture["displacements"])
    base = linearize(manager, displacement, nodal_force, tangent=True)

    desired = np.zeros_like(displacement)
    source_dofs = model.mesh.dof_manager.get_node_dofs(13)
    target_lower = model.mesh.dof_manager.get_node_dofs(27)
    target_upper = model.mesh.dof_manager.get_node_dofs(28)
    desired[source_dofs[0]] = 0.25
    desired[source_dofs[2]] = 1.0
    desired[source_dofs[4]] = 0.2
    desired[target_lower[2]] = -0.7
    desired[target_lower[4]] = 0.15
    desired[target_upper[2]] = 0.3
    direction = np.asarray(transform @ (transform.T @ desired)).reshape(-1)
    predicted_force = np.asarray(base.tangent @ direction).reshape(-1)
    predicted_gap = base.gap_gradient @ direction
    assert np.linalg.norm(predicted_force) > 1.0e-8
    assert np.linalg.norm(predicted_gap) > 1.0e-8

    errors = []
    for step_m in (1.0e-6, 5.0e-7, 2.5e-7):
        plus = linearize(manager, displacement+step_m*direction,
                         nodal_force, tangent=False)
        minus = linearize(manager, displacement-step_m*direction,
                          nodal_force, tangent=False)
        observed_force = (plus.force-minus.force)/(2.0*step_m)
        observed_gap = ((plus.state.normalized_gap
                         - minus.state.normalized_gap)/(2.0*step_m))
        force_error = np.linalg.norm(observed_force-predicted_force)/max(
            np.linalg.norm(observed_force), np.linalg.norm(predicted_force),
            1.0e-12,
        )
        gap_error = np.linalg.norm(observed_gap-predicted_gap)/max(
            np.linalg.norm(observed_gap), np.linalg.norm(predicted_gap),
            1.0e-12,
        )
        force_from_gap_gradient = -nodal_force @ (
            (plus.gap_gradient-minus.gap_gradient)/(2.0*step_m))
        gradient_force_error = np.linalg.norm(
            observed_force-force_from_gap_gradient,
        )/max(np.linalg.norm(observed_force), 1.0e-12)
        errors.append((step_m, force_error, gap_error,
                       gradient_force_error,
                       np.linalg.norm(transform.T @ (
                           observed_force-predicted_force))
                       / max(np.linalg.norm(transform.T @ observed_force),
                             1.0e-12)))
    assert all(force_error <= 1.0e-5 and gap_error <= 1.0e-5
               and gradient_force_error <= 1.0e-5
               and reduced_force_error <= 1.0e-5
               for _, force_error, gap_error, gradient_force_error,
               reduced_force_error in errors), {
                   "errors": errors,
                   "largest_force_rows": np.argsort(np.abs(
                       predicted_force-observed_force))[-8:].tolist(),
               }
