"""Multiple conforming Q4 pressure faces against a nonmatching S3 patch."""

import os
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction, ContactPathControl,
    ContactPathProgram, ContactSolverSettings, ContactSurface,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.s3_patch_static import (
    GeneralS3PatchManager, REFINED_Q4_S3_AUGMENTED_PROFILE,
    REFINED_Q4_S3_PROJECTED_AUGMENTED_PROFILE,
)


def _two_q4_two_s3():
    model = FEModel("refined_q4_pressure_patch")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    source = (
        (0.1, 0.1, 0.055), (0.6, 0.1, 0.055), (1.1, 0.1, 0.055),
        (0.1, 1.1, 0.055), (0.6, 1.1, 0.055), (1.1, 1.1, 0.055),
    )
    target = (
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    )
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    for element_id, nodes in ((1, (1, 2, 5, 4)), (2, (2, 3, 6, 5))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            reference_normal=(0.0, 0.0, 1.0),
        ))
    for element_id, nodes in ((3, (7, 8, 10)), (4, (8, 9, 10))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0.0, 0.0, 1.0),
            reference_surface_offset=0.01,
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("q4_source", (1, 2), "bottom"),
                  ContactSurface("s3_target", (3, 4), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "q4_source", "s3_target", 1000.0, 0.02,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2.0e-5,
        maximum_general_s3_energy_evaluations=5000,
        q4_source_area_tolerance=1.0e-9,
        q4_source_weighted_gap_tolerance=1.0e-10,
        maximum_q4_source_integration_triangles=300,
        pressure_tolerance=0.02,
    )
    return model, definition, settings


def test_refined_q4_source_public_work_and_opening():
    model, definition, settings = _two_q4_two_s3()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert report.profile == REFINED_Q4_S3_AUGMENTED_PROFILE
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.law.geometry.differentiation == "pair_local"
    assert manager.law.source_node_ids == (1, 2, 3, 4, 5, 6)
    opening = manager.evaluate(np.zeros(manager.n_dof), tangent=False)[2]
    assert opening.active_pressure_dofs == 0
    u = np.zeros(manager.n_dof)
    for node in range(1, 7):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    force, _tangent, result = manager.evaluate(u, tangent=False)
    assert result.active_pressure_dofs == 6
    assert result.gap_tolerance_met
    assert result.energy_evaluations <= settings.maximum_general_s3_energy_evaluations
    np.testing.assert_allclose(
        np.asarray(result.integration_force) + result.target_force,
        0.0, atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.integration_moment) + result.target_moment,
        0.0, atol=1.0e-6,
    )
    assert np.linalg.norm(force) > 0.0
    assert manager.restart_contract()["profile"] == REFINED_Q4_S3_AUGMENTED_PROFILE
    assert not validate_general_contact(model, replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="penalty"),
    )), settings).valid
    exhausted = validate_general_contact(
        model, definition, replace(
            settings, maximum_general_s3_energy_evaluations=3535,
        ),
    )
    assert not exhausted.valid
    assert any("Hessian exceeds" in issue for issue in exhausted.issues)
    projected_settings = replace(settings, q4_source_gap_certificate="projected_matching")
    projected_report = validate_general_contact(model, definition, projected_settings)
    assert projected_report.valid, projected_report.issues
    assert projected_report.profile == REFINED_Q4_S3_PROJECTED_AUGMENTED_PROFILE
    projected = GeneralS3PatchManager(model, definition, projected_settings)
    assert projected.restart_contract()["profile"] == projected_report.profile
    assert projected.restart_contract()["settings"]["q4_source_gap_certificate"] == "projected_matching"
    projected_result = projected.evaluate(u, tangent=False)[2]
    assert result.minimum_gap <= projected_result.minimum_gap
    assert projected_result.minimum_gap <= projected_result.sampled_overlap_gap + 1e-12


def test_refined_q4_source_reduced_tangent_keeps_shared_node_work():
    model, definition, settings = _two_q4_two_s3()
    manager = GeneralS3PatchManager(model, definition, settings)
    free = [model.mesh.dof_manager.get_node_dofs(node)[axis]
            for node in range(1, 7) for axis in (0, 2)]
    transform = sparse.eye(manager.n_dof, format="csr")[:, free]
    manager.restrict_tangent_to_reduced_variations(transform)
    u = np.zeros(manager.n_dof)
    for node in range(1, 7):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    _force, tangent, result = manager.evaluate(u)
    assert result.active_pressure_dofs == 6
    assert result.energy_evaluations < 1000
    direction = np.zeros(manager.n_dof)
    direction[model.mesh.dof_manager.get_node_dofs(2)[0]] = 0.2
    direction[model.mesh.dof_manager.get_node_dofs(5)[2]] = 0.5
    step = 1.0e-4
    plus = manager.evaluate(u + step*direction, tangent=False)[0]
    minus = manager.evaluate(u - step*direction, tangent=False)[0]
    predicted = transform.T @ (tangent @ direction)
    measured = transform.T @ ((plus-minus)/(2.0*step))
    np.testing.assert_allclose(predicted, measured, rtol=3.0e-3, atol=8.0e-4)


def test_refined_q4_reports_sampled_weak_open_pressure_underresolution():
    model, definition, settings = _two_q4_two_s3()
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], gap_tolerance=5.0e-5,
    ),))
    manager = GeneralS3PatchManager(model, definition, settings)
    displacement = np.zeros(manager.n_dof)
    for node in range(1, 7):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.0149
    displacement[model.mesh.dof_manager.get_node_dofs(1)[2]] -= 0.0002
    force, _tangent, weak_open = manager.evaluate(displacement, tangent=False)
    projected = manager.projected_normal_multipliers(displacement)
    assert weak_open.sampled_overlap_gap == pytest.approx(-1.0e-4, abs=1.0e-8)
    assert not weak_open.gap_tolerance_met
    assert weak_open.active_pressure_dofs == 0
    assert all(value > 0.0 for value in weak_open.normalized_weighted_gap_m)
    assert np.linalg.norm(force) == 0.0
    np.testing.assert_array_equal(projected, manager.normal_multipliers)
    assert manager.has_unresolved_weak_open_gap(weak_open, projected)

    # The conservative whole-face lower bound is insufficient by itself.
    bound_only = replace(weak_open, sampled_overlap_gap=0.0)
    assert not manager.has_unresolved_weak_open_gap(bound_only, projected)
    assert not manager.has_unresolved_weak_open_gap(
        replace(weak_open, reference_support_m2=(0.0,)*6), projected,
    )

    active_pose = np.zeros(manager.n_dof)
    for node in range(1, 7):
        active_pose[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.01501
    _force, _tangent, active = manager.evaluate(active_pose, tangent=False)
    assert active.active_pressure_dofs == 6
    assert not manager.has_unresolved_weak_open_gap(
        active, manager.projected_normal_multipliers(active_pose),
    )


def test_refined_q4_pressure_failure_uses_path_transaction(monkeypatch):
    model, definition, settings = _two_q4_two_s3()
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (7, 8, 9, 10),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4, 5, 6),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.001, 0.001)),
        steps_per_segment=1,
    )
    calls = []

    def forced_failure(_manager, result, _projected):
        calls.append(result.minimum_gap)
        return True

    monkeypatch.setattr(
        GeneralS3PatchManager, "has_unresolved_weak_open_gap", forced_failure,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert calls
    assert result.status == "diverged"
    assert result.failure_reason == "contact_pressure_space_underresolved"
    assert result.info["contact_failure"]["sampled_overlap_gap"] is not None
    trace = result.info["contact_failure"]["augmentation_trace"]
    assert len(trace) == 1
    assert trace[0]["active_pressure_dofs"] == 0
    assert trace[0]["source_contact_force_n"] == [0.0, 0.0, 0.0]
    assert trace[0]["sampled_overlap_gap_m"] is not None
    assert trace[0]["pressure_rank"] is None
    assert isinstance(trace[0]["pressure_gram_rank"], int)
    assert trace[0]["pressure_gram_condition_estimate"] is not None
    assert trace[0]["minimum_weighted_gap_m"] is None
    assert len(trace[0]["normalized_weighted_gap_m"]) == 6
    assert len(trace[0]["normal_multiplier_estimate_pa"]) == 6
    assert len(trace[0]["source_pressure_node_ids"]) == 6
    assert result.steps == []
    assert result.restart_checkpoint is None
    np.testing.assert_array_equal(result.displacements, 0.0)


@pytest.mark.skipif(os.environ.get("ANYSOLVER_LONG_CONTACT") != "1",
                    reason="opt-in coupled multi-Q4 contact path")
def test_refined_q4_source_active_two_control_path():
    model, definition, settings = _two_q4_two_s3()
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], gap_tolerance=0.005,
    ),))
    settings = replace(settings, maximum_augmentations=8)
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (7, 8, 9, 10),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4, 5, 6),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505)),
        steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    contact = result.info["general_contact"]
    assert contact["profile"] == REFINED_Q4_S3_AUGMENTED_PROFILE
    assert contact["active_pressure_dofs"] == 6
    assert contact["gap_tolerance_met"]
    assert contact["pressure_gram_rank"] == 6
    np.testing.assert_allclose(
        np.asarray(contact["integration_force"]) + contact["target_force"],
        0.0, atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(contact["integration_moment"]) + contact["target_moment"],
        0.0, atol=1.0e-6,
    )
    assert result.restart_checkpoint["version"] == 2


@pytest.mark.skipif(os.environ.get("ANYSOLVER_LONG_CONTACT") != "1",
                    reason="opt-in coupled multi-Q4 active split restart")
def test_refined_q4_source_active_split_restart_matches_full_path():
    model, definition, settings = _two_q4_two_s3()
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], gap_tolerance=0.005,
    ),))
    settings = replace(settings, maximum_augmentations=8)
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (7, 8, 9, 10),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4, 5, 6),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505), (0.0, -0.01506)),
        steps_per_segment=1,
    )
    options = dict(contact=definition, contact_settings=settings,
                   contact_path_program=program, max_iterations=12,
                   emit_restart_checkpoint=True)
    full = solve_static_nonlinear(model, **options)
    assert full.status == "completed", full.failure_reason
    prefix = solve_static_nonlinear(model, max_load_factor=0.5, **options)
    assert prefix.status == "completed", prefix.failure_reason
    assert prefix.info["general_contact"]["active_pressure_dofs"] == 6
    assert prefix.info["general_contact"]["gap_tolerance_met"]
    assert prefix.restart_checkpoint["version"] == 2
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=prefix.to_restart_checkpoint(), **options,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=0.0, atol=1.0e-10)
    for field in ("integration_force", "target_force",
                  "integration_moment", "target_moment"):
        np.testing.assert_allclose(
            resumed.info["general_contact"][field],
            full.info["general_contact"][field], rtol=0.0, atol=1.0e-8,
        )
    assert (resumed.to_restart_checkpoint()["path_state"]["normal_multipliers"]
            == full.to_restart_checkpoint()["path_state"]["normal_multipliers"])
    assert resumed.to_restart_checkpoint() == full.to_restart_checkpoint()
