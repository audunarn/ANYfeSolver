"""Local Q1 projected normal work on a physical Q4 pressure side."""

from dataclasses import replace

import numpy as np
from scipy import sparse

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    DisplacementControl,
    LoadCase, NonlinearConvergenceSettings, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_q4_s3_patch_static import _case


def _projected_case():
    model, definition, settings = _case()
    manager = GeneralS3PatchManager(model, definition, settings)
    law = MovingS3PatchProjectedNormal(manager.law)
    u = np.zeros(manager.n_dof)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.015
    return model, manager, law, u


def test_q4_source_projected_constant_pressure_has_balanced_normal_work():
    _model, manager, law, u = _projected_case()
    source, target = manager._positions(u)
    multipliers = np.full(4, 10.0)
    state = law.evaluate(source, target, multipliers)
    np.testing.assert_allclose(state.normalized_gap, 0.0, atol=1e-10)
    np.testing.assert_allclose(state.pressure, 10.0, atol=1e-8)
    work = law.work(source, target, multipliers, tangent=False)
    forces = -work.residual.reshape(-1, 3)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-7)
    assert abs(float(forces[:4, :2].sum())) < 1e-7
    np.testing.assert_allclose(law.projected_multiplier(state, multipliers),
                               10.0, atol=1e-8)
    opened = source.copy()
    opened[:, 2] += 0.02
    state_open = law.evaluate(opened, target, multipliers)
    np.testing.assert_array_equal(
        law.projected_multiplier(state_open, multipliers), 0.0,
    )


def test_q4_source_projected_smooth_tangent_matches_directional_work():
    _model, manager, law, u = _projected_case()
    source, target = manager._positions(u)
    source[:, 2] -= 0.001
    multipliers = np.full(4, 10.0)
    work = law.work(source, target, multipliers)
    direction = np.zeros(law.n_coordinates)
    direction[:12:3] = 0.5
    step = 1e-4
    coordinates = np.concatenate((source, target)).reshape(-1)

    def residual(sign):
        values = (coordinates + sign*step*direction).reshape(-1, 3)
        return law.work(values[:4], values[4:], multipliers,
                        tangent=False).residual

    numerical = (residual(1)-residual(-1))/(2*step)
    np.testing.assert_allclose(work.tangent @ direction, numerical,
                               rtol=3e-3, atol=8e-4)


def test_q4_source_augmented_public_force_map_and_pressure():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented"),
    ))
    settings = replace(settings, pressure_tolerance=2e-2)
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.profile == report.profile
    manager.set_normal_multipliers(np.full(4, 10.0))
    u = np.zeros(manager.n_dof)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.015
    force, _tangent, result = manager.evaluate(u, tangent=False)
    assert result.active_pressure_dofs == 4
    np.testing.assert_allclose(result.numerical_pressure_pa, 10.0, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(result.integration_force)+result.target_force,
        0.0, atol=1e-7,
    )
    assert np.linalg.norm(force) > 0.0


def test_q4_source_reduced_tangent_keeps_free_coordinate_derivative():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented"),
    ))
    settings = replace(settings, pressure_tolerance=2e-2)
    manager = GeneralS3PatchManager(model, definition, settings)
    free = [model.mesh.dof_manager.get_node_dofs(node)[axis]
            for node in (1, 2, 3, 4) for axis in (0, 2)]
    transform = sparse.eye(manager.n_dof, format="csr")[:, free]
    manager.restrict_tangent_to_reduced_variations(transform)
    manager.set_normal_multipliers(np.full(4, 10.0))
    u = np.zeros(manager.n_dof)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.016
    _force, tangent, result = manager.evaluate(u)
    assert result.energy_evaluations < 300
    direction = np.zeros(manager.n_dof)
    direction[free[1::2]] = 0.5
    step = 1e-4
    plus = manager.evaluate(u+step*direction, tangent=False)[0]
    minus = manager.evaluate(u-step*direction, tangent=False)[0]
    predicted = transform.T @ (tangent @ direction)
    numerical = transform.T @ ((plus-minus)/(2*step))
    np.testing.assert_allclose(predicted, numerical,
                               rtol=3e-3, atol=8e-4)


def test_q4_source_warped_current_skin_balances_moment_and_tangent():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented"),
    ))
    settings = replace(settings, pressure_tolerance=2e-2)
    manager = GeneralS3PatchManager(model, definition, settings)
    manager.set_normal_multipliers(np.full(4, 10.0))
    u = np.zeros(manager.n_dof)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.016
    u[model.mesh.dof_manager.get_node_dofs(3)[2]] += 0.001
    _force, tangent, result = manager.evaluate(u)
    assert result.active_pressure_dofs == 4
    np.testing.assert_allclose(
        np.asarray(result.integration_force)+result.target_force,
        0.0, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.integration_moment)+result.target_moment,
        0.0, atol=1e-6,
    )
    direction = np.zeros(manager.n_dof)
    for node in (1, 2, 3, 4):
        direction[model.mesh.dof_manager.get_node_dofs(node)[2]] = 0.5
    step = 1e-4
    plus = manager.evaluate(u+step*direction, tangent=False)[0]
    minus = manager.evaluate(u-step*direction, tangent=False)[0]
    np.testing.assert_allclose(tangent @ direction, (plus-minus)/(2*step),
                               rtol=3e-3, atol=8e-4)


def _path_case():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented",
                gap_tolerance=5e-3),
    ))
    settings = replace(settings, pressure_tolerance=2e-2,
                       maximum_augmentations=12)
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.0155)),
        steps_per_segment=1,
    )
    return model, definition, settings, program


def test_q4_source_augmented_overdriven_corner_rolls_back():
    model, definition, settings, program = _path_case()
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
    )
    assert result.status == "diverged"
    assert result.failure_reason == "maximum_augmentations_reached"
    trace = result.info["contact_failure"]["augmentation_trace"]
    assert len(trace) == settings.maximum_augmentations
    assert trace[-1]["pressure_change"] > settings.pressure_tolerance
    assert trace[-1]["pressure_change"] < trace[0]["pressure_change"]
    np.testing.assert_array_equal(result.displacements, 0.0)
    np.testing.assert_array_equal(
        result.info["general_contact"]["normal_multiplier_estimate_pa"], 0.0,
    )


def test_q4_source_augmented_exhausted_update_rolls_back():
    model, definition, settings, program = _path_case()
    settings = replace(settings, maximum_augmentations=1,
                       pressure_tolerance=1e-5)
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
    )
    assert result.status != "completed"
    assert result.failure_reason == "maximum_augmentations_reached"
    np.testing.assert_array_equal(result.displacements, 0.0)
    np.testing.assert_array_equal(
        result.info["general_contact"]["normal_multiplier_estimate_pa"], 0.0,
    )


def test_q4_source_augmented_four_phase_path():
    model, definition, settings, _program = _path_case()
    settings = replace(settings, maximum_augmentations=16)
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505),
                   (0.05, -0.01505), (0.05, 0.0),
                   (0.05, -0.01505)),
        steps_per_segment=1,
    )
    common = dict(
        contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
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
    half = solve_static_nonlinear(model, max_load_factor=0.75, **common)
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        result.info["general_contact"]["normal_multiplier_estimate_pa"],
        rtol=1e-6, atol=1e-7,
    )


def test_q4_source_augmented_loaded_force_control():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented",
                gap_tolerance=5e-3),
    ))
    settings = replace(settings, pressure_tolerance=2e-2,
                       maximum_augmentations=16)
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
    model.add_boundary_condition(BoundaryCondition(
        "source_clamp", (1, 4), {"uz": 0.0},
    ))
    load = LoadCase("press")
    for node in (2, 3):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.05))
    common = dict(
        contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=2,
        max_iterations=8, min_step_fraction=0.1,
        convergence_settings=NonlinearConvergenceSettings(
            growth_factor=1.0, max_step_factor=1.0,
        ),
        emit_restart_checkpoint=True,
    )
    result = solve_static_nonlinear(model, load, num_steps=4,
                                    **{key: value for key, value in common.items()
                                       if key != "num_steps"})
    assert result.status == "completed", (
        result.failure_reason, result.info.get("first_failed_iteration_reason"),
    )
    contact = result.info["general_contact"]
    assert contact["active_pressure_dofs"] > 0
    assert contact["gap_tolerance_met"]
    assert any(row["augmentation_iterations"] > 1 for row in
               result.info["force_displacement_history"])
    assert len(result.to_restart_checkpoint()["path_state"]["normal_multipliers"]) == 4
    half = solve_static_nonlinear(model, load, max_load_factor=0.5, **common)
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    assert [row["load_factor"] for row in
            result.info["force_displacement_history"]] == [0.25, 0.5, 0.75, 1.0]
    assert [row["load_factor"] for row in
            half.info["force_displacement_history"]] == [0.25, 0.5]
    assert [row["load_factor"] for row in
            resumed.info["force_displacement_history"]] == [0.25, 0.5, 0.75, 1.0]
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1e-2, atol=1e-6)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        contact["normal_multiplier_estimate_pa"],
        rtol=5e-2, atol=2e-2,
    )


def test_q4_source_augmented_ordinary_displacement_control():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented",
                gap_tolerance=5e-3),
    ))
    settings = replace(settings, pressure_tolerance=2e-2,
                       maximum_augmentations=16)
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("drive")
    load.add_nodal_load(1, forces=(0.0, 0.0, -1.0))
    common = dict(
        contact=definition, contact_settings=settings,
        control="displacement", kinematics="von_karman",
        max_iterations=15, emit_restart_checkpoint=True,
        displacement_control=DisplacementControl(
            node_id=1, dof="uz", target_displacement=-0.01505,
        ),
    )
    result = solve_static_nonlinear(model, load, num_steps=2, **common)
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
        contact["normal_multiplier_estimate_pa"],
        rtol=5e-2, atol=2e-2,
    )
