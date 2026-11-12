"""Projected Q1 normal work for one physical Q4 face on each side."""

from dataclasses import replace

import numpy as np
from scipy import sparse

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    DisplacementControl, LoadCase, NonlinearConvergenceSettings,
    solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_q4_q4_static import _case


def test_q4_q4_local_projected_work_opening_and_warped_tangent():
    model, definition, settings = _case()
    manager = GeneralS3PatchManager(model, definition, settings)
    law = MovingS3PatchProjectedNormal(manager.law)
    u = np.zeros(manager.n_dof)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.015
    source, target = manager._positions(u)
    multipliers = np.full(4, 10.0)
    state = law.evaluate(source, target, multipliers)
    np.testing.assert_allclose(state.normalized_gap, 0.0, atol=1e-10)
    np.testing.assert_allclose(state.pressure, 10.0, atol=1e-8)
    force = -law.work(source, target, multipliers, tangent=False).residual
    np.testing.assert_allclose(force.reshape(-1, 3).sum(axis=0), 0.0,
                               atol=1e-7)
    opened = source.copy()
    opened[:, 2] += 0.02
    state_open = law.evaluate(opened, target, multipliers)
    np.testing.assert_array_equal(
        law.projected_multiplier(state_open, multipliers), 0.0,
    )

    source[:, 2] -= 0.001
    target[2, 2] += 0.001
    work = law.work(source, target, multipliers)
    forces = -work.residual.reshape(-1, 3)
    points = np.concatenate((source, target))
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.cross(points, forces).sum(axis=0), 0.0,
                               atol=1e-6)
    direction = np.zeros(law.n_coordinates)
    direction[:12:3] = 0.5
    step = 1e-4
    coordinates = points.reshape(-1)

    def residual(sign):
        values = (coordinates+sign*step*direction).reshape(-1, 3)
        return law.work(values[:4], values[4:], multipliers,
                        tangent=False).residual

    numerical = (residual(1)-residual(-1))/(2*step)
    np.testing.assert_allclose(work.tangent @ direction, numerical,
                               rtol=3e-3, atol=8e-4)


def _augmented_case():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented",
                gap_tolerance=5e-3),
    ))
    settings = replace(settings, pressure_tolerance=2e-2,
                       maximum_augmentations=16)
    return model, definition, settings


def test_q4_q4_public_augmented_work_and_reduced_tangent():
    model, definition, settings = _augmented_case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.profile == report.profile
    free = [model.mesh.dof_manager.get_node_dofs(node)[2]
            for node in (1, 2, 3, 4)]
    transform = sparse.eye(manager.n_dof, format="csr")[:, free]
    manager.restrict_tangent_to_reduced_variations(transform)
    manager.set_normal_multipliers(np.full(4, 10.0))
    u = np.zeros(manager.n_dof)
    u[free] = -0.016
    force, tangent, contact = manager.evaluate(u)
    assert contact.active_pressure_dofs == 4
    assert contact.energy_evaluations < 300
    np.testing.assert_allclose(
        np.asarray(contact.integration_force)+contact.target_force,
        0.0, atol=1e-6,
    )
    direction = np.zeros(manager.n_dof)
    direction[free] = 0.5
    step = 1e-4
    plus = manager.evaluate(u+step*direction, tangent=False)[0]
    minus = manager.evaluate(u-step*direction, tangent=False)[0]
    np.testing.assert_allclose(
        transform.T @ (tangent @ direction),
        transform.T @ ((plus-minus)/(2*step)),
        rtol=3e-3, atol=8e-4,
    )


def test_q4_q4_augmented_four_phase_public_path():
    model, definition, settings = _augmented_case()
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    path = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505),
                   (0.05, -0.01505), (0.05, 0.0),
                   (0.05, -0.01505)),
        steps_per_segment=1,
    )
    common = dict(
        contact=definition, contact_settings=settings,
        contact_path_program=path, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    result = solve_static_nonlinear(model, **common)
    assert result.status == "completed", (
        result.failure_reason, result.info.get("contact_failure"),
    )
    phases = result.info["force_displacement_history"]
    assert [phase["contact"]["active_pressure_dofs"] > 0
            for phase in phases] == [True, True, False, True]
    assert all(phase["contact"]["gap_tolerance_met"] for phase in phases)
    assert phases[2]["contact"]["normal_multiplier_estimate_pa"] == [0.0]*4
    for phase in phases:
        contact = phase["contact"]
        np.testing.assert_allclose(
            np.asarray(contact["integration_force"])+contact["target_force"],
            0.0, atol=1e-6,
        )
    opened = solve_static_nonlinear(model, max_load_factor=0.75, **common)
    assert opened.status == "completed", opened.failure_reason
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=opened.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        result.info["general_contact"]["normal_multiplier_estimate_pa"],
        rtol=1e-6, atol=1e-7,
    )


def _ordinary_case(*, force_control):
    model, definition, settings = _augmented_case()
    if force_control:
        for node in (1, 2, 3, 4):
            model.mesh.get_node(node).z -= 0.015
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("drive")
    if force_control:
        model.add_boundary_condition(BoundaryCondition(
            "source_clamp", (1, 4), {"uz": 0.0},
        ))
        for node in (2, 3):
            load.add_nodal_load(node, forces=(0.0, 0.0, -0.05))
    else:
        load.add_nodal_load(1, forces=(0.0, 0.0, -1.0))
    return model, definition, settings, load


def _assert_ordinary_contact(result):
    assert result.status == "completed", (
        result.failure_reason, result.info.get("first_failed_iteration_reason"),
    )
    contact = result.info["general_contact"]
    assert contact["active_pressure_dofs"] > 0
    assert contact["gap_tolerance_met"]
    assert any(row["augmentation_iterations"] > 1 for row in
               result.info["force_displacement_history"])
    assert len(result.to_restart_checkpoint()["path_state"]["normal_multipliers"]) == 4
    np.testing.assert_allclose(
        np.asarray(contact["integration_force"])+contact["target_force"],
        0.0, atol=1e-6,
    )


def test_q4_q4_augmented_ordinary_force_control():
    model, definition, settings, load = _ordinary_case(force_control=True)
    common = dict(
        contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman",
        max_iterations=8, min_step_fraction=0.1,
        convergence_settings=NonlinearConvergenceSettings(
            growth_factor=1.0, max_step_factor=1.0,
        ),
        emit_restart_checkpoint=True,
    )
    result = solve_static_nonlinear(model, load, num_steps=4, **common)
    _assert_ordinary_contact(result)
    half = solve_static_nonlinear(
        model, load, num_steps=2, max_load_factor=0.5, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, num_steps=2,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    assert [row["load_factor"] for row in
            resumed.info["force_displacement_history"]] == [0.25, 0.5, 0.75, 1.0]
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1e-2, atol=1e-6)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        result.info["general_contact"]["normal_multiplier_estimate_pa"],
        rtol=5e-2, atol=2e-2,
    )


def test_q4_q4_augmented_ordinary_displacement_control():
    model, definition, settings, load = _ordinary_case(force_control=False)
    common = dict(
        contact=definition, contact_settings=settings,
        control="displacement", kinematics="von_karman",
        max_iterations=15, emit_restart_checkpoint=True,
        displacement_control=DisplacementControl(
            node_id=1, dof="uz", target_displacement=-0.01505,
        ),
    )
    result = solve_static_nonlinear(model, load, num_steps=2, **common)
    _assert_ordinary_contact(result)
    half = solve_static_nonlinear(
        model, load, num_steps=1, max_load_factor=0.5, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, num_steps=1,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1e-2, atol=1e-6)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        result.info["general_contact"]["normal_multiplier_estimate_pa"],
        rtol=5e-2, atol=2e-2,
    )


def test_q4_q4_augmented_single_press_tighter_gap_rejects_and_rolls_back():
    model, definition, settings = _augmented_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_stiffness=10000.0,
                gap_tolerance=5e-5),
    ))
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    path = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505)),
        steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=path, max_iterations=40,
    )
    assert result.status == "diverged"
    assert result.failure_reason == "contact_gap_exceeded"
    np.testing.assert_array_equal(result.displacements, 0.0)
    np.testing.assert_array_equal(
        result.info["general_contact"]["normal_multiplier_estimate_pa"],
        0.0,
    )
