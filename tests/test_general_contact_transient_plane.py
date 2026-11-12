"""Independent checks for the fixed-plane Q4 and S3 Newmark development slices."""

import math
import json
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    ContactDefinition,
    ContactInteraction,
    ContactNodalLoadHistory,
    ContactSurface,
    ContactTransientSolverControls,
    LoadCase,
    RigidContactBody,
    TransientConfig,
    TransientContactSettings,
    create_shell_element,
    solve_transient_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.algebraic_dynamics import build_declared_algebraic_basis
from anyfesolver.assembly import assemble_mass_matrix, build_constraint_transformation
from anyfesolver.matrix_assembly import assemble_load_vector, assemble_stiffness_matrix
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.general_contact.core import (
    _PlanarQ4PlaneManager, active_contact_manager, contact_scope,
)
from anyfesolver.general_contact.transient import _minimum_hermite_gap


def _guided_plate():
    model = FEModel("uniform_q4_impact")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, xyz in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
         (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)), 1
    ):
        model.add_node(node_id, *xyz)
    model.add_element(1, create_shell_element(
        1, [1, 2, 3, 4], "elastic", thickness=0.05,
    ))
    model.add_boundary_condition(BoundaryCondition(
        "guide", [1, 2, 3, 4],
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    contact = ContactDefinition(
        surfaces=(ContactSurface("skin", (1,)),),
        rigid_bodies=(RigidContactBody(
            "floor", "plane", (0.0, 0.0, -0.01), (0.0, 0.0, 1.0),
        ),),
        interactions=(ContactInteraction(
            "support", "skin", "floor", 1.0e5, 0.001,
        ),),
    )
    velocity = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (1, 2, 3, 4):
        velocity[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -1.0
    return model, contact, velocity


def _constant_q4_load(force_per_node=-0.005):
    load = LoadCase("constant_q4_downward")
    for node_id in (1, 2, 3, 4):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, force_per_node))
    return load


def test_q4_constant_nodal_load_matches_free_acceleration_and_restart():
    model, contact, velocity = _guided_plate()
    velocity[:] = 0.0
    load = _constant_q4_load()
    full = solve_transient_contact(
        model, contact, TransientConfig(.001, .01,
                                        initial_velocity=velocity),
        base_load_case=load,
    )
    # Four nodes carry 0.02 N downward; the guided plate mass is 0.05 kg.
    expected_acceleration = -0.02/0.05
    for node_id in (1, 2, 3, 4):
        dof = model.mesh.dof_manager.get_node_dofs(node_id)[2]
        assert full.displacements[-1, dof] == pytest.approx(
            0.5*expected_acceleration*.01**2, abs=2e-11,
        )
        assert full.velocities[-1, dof] == pytest.approx(
            expected_acceleration*.01, abs=2e-9,
        )
        assert full.accelerations[0, dof] == pytest.approx(
            expected_acceleration, abs=2e-9,
        )
    assert all(item.active_pressure_dofs == 0 for item in full.contact)
    expected_work = 0.5*0.02*(0.02/0.05)*full.times**2
    np.testing.assert_allclose(full.external_nodal_work_j, expected_work,
                               rtol=2e-9, atol=2e-14)
    np.testing.assert_allclose(full.external_nodal_work_j,
                               0.5*0.05*(0.02/0.05*full.times)**2,
                               rtol=2e-9, atol=2e-14)

    split_model, split_contact, split_velocity = _guided_plate()
    split_velocity[:] = 0.0
    first = solve_transient_contact(
        split_model, split_contact,
        TransientConfig(.001, .005, initial_velocity=split_velocity),
        base_load_case=_constant_q4_load(),
        emit_restart_checkpoint=True,
    )
    resumed_model, resumed_contact, resumed_velocity = _guided_plate()
    resumed_velocity[:] = 0.0
    continuation = solve_transient_contact(
        resumed_model, resumed_contact,
        TransientConfig(.001, .01, initial_velocity=resumed_velocity),
        base_load_case=_constant_q4_load(),
        restart_checkpoint=first.restart_checkpoint,
    )
    np.testing.assert_allclose(continuation.displacements[-1],
                               full.displacements[-1], atol=2e-11)
    np.testing.assert_allclose(continuation.velocities[-1],
                               full.velocities[-1], atol=2e-9)
    np.testing.assert_allclose(continuation.accelerations[0],
                               first.accelerations[-1], atol=2e-9)
    assert continuation.external_nodal_work_j[0] == pytest.approx(
        first.external_nodal_work_j[-1], abs=2e-14,
    )
    np.testing.assert_allclose(
        continuation.external_nodal_work_j,
        full.external_nodal_work_j[len(first.times)-1:],
        rtol=2e-9, atol=2e-14,
    )

    changed_model, changed_contact, changed_velocity = _guided_plate()
    changed_velocity[:] = 0.0
    with pytest.raises(ValueError, match="checkpoint.*(analysis|incompatible)"):
        solve_transient_contact(
            changed_model, changed_contact,
            TransientConfig(.001, .01, initial_velocity=changed_velocity),
            base_load_case=_constant_q4_load(-0.006),
            restart_checkpoint=first.restart_checkpoint,
        )
    renamed_model, renamed_contact, renamed_velocity = _guided_plate()
    renamed_velocity[:] = 0.0
    renamed_load = _constant_q4_load()
    renamed_load.name = "different_load_identity"
    with pytest.raises(ValueError, match="checkpoint.*(analysis|incompatible)"):
        solve_transient_contact(
            renamed_model, renamed_contact,
            TransientConfig(.001, .01, initial_velocity=renamed_velocity),
            base_load_case=renamed_load,
            restart_checkpoint=first.restart_checkpoint,
        )


def test_q4_linear_nodal_load_history_work_and_knot_restart():
    history = ContactNodalLoadHistory(((0.0, 0.0), (.005, .5), (.01, 1.0)))
    model, contact, velocity = _guided_plate()
    velocity[:] = 0.0
    full = solve_transient_contact(
        model, contact, TransientConfig(.003, .01, initial_velocity=velocity),
        base_load_case=_constant_q4_load(), nodal_load_history=history,
        emit_restart_checkpoint=True,
    )
    np.testing.assert_allclose(full.times, (0.0, .003, .005, .008, .01),
                               rtol=0.0, atol=1e-14)
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    assert full.accelerations[0, dof] == pytest.approx(0.0, abs=1e-12)
    assert full.velocities[-1, dof] == pytest.approx(-.002, abs=1e-9)
    assert full.displacements[-1, dof] == pytest.approx(
        -.4*.01**2/6.0, abs=4e-7,
    )
    assert full.external_nodal_work_j[0] == 0.0
    coarse_work_error = abs(full.external_nodal_work_j[-1]-1e-7)
    assert coarse_work_error < 3e-9
    refined_model, refined_contact, refined_velocity = _guided_plate()
    refined_velocity[:] = 0.0
    refined = solve_transient_contact(
        refined_model, refined_contact,
        TransientConfig(.0015, .01, initial_velocity=refined_velocity),
        base_load_case=_constant_q4_load(), nodal_load_history=history,
    )
    assert abs(refined.external_nodal_work_j[-1]-1e-7) < coarse_work_error/3.0
    assert all(state.active_pressure_dofs == 0 for state in full.contact)
    assert full.restart_checkpoint["version"] == 2
    assert full.restart_checkpoint["state"]["external_work_j"] == pytest.approx(
        full.external_nodal_work_j[-1], abs=1e-16,
    )

    first_model, first_contact, first_velocity = _guided_plate()
    first_velocity[:] = 0.0
    first = solve_transient_contact(
        first_model, first_contact,
        TransientConfig(.003, .005, initial_velocity=first_velocity),
        base_load_case=_constant_q4_load(), nodal_load_history=history,
        emit_restart_checkpoint=True,
    )
    assert first.restart_checkpoint["version"] == 2
    resumed_model, resumed_contact, resumed_velocity = _guided_plate()
    resumed_velocity[:] = 0.0
    resumed = solve_transient_contact(
        resumed_model, resumed_contact,
        TransientConfig(.003, .01, initial_velocity=resumed_velocity),
        base_load_case=_constant_q4_load(), nodal_load_history=history,
        restart_checkpoint=first.restart_checkpoint,
    )
    for name in ("times", "displacements", "velocities", "accelerations",
                 "external_nodal_work_j"):
        combined = np.concatenate((getattr(first, name),
                                   getattr(resumed, name)[1:]), axis=0)
        np.testing.assert_allclose(combined, getattr(full, name),
                                   rtol=0.0, atol=1e-12)

    changed_model, changed_contact, changed_velocity = _guided_plate()
    changed_velocity[:] = 0.0
    with pytest.raises(ValueError, match="checkpoint analysis is incompatible"):
        solve_transient_contact(
            changed_model, changed_contact,
            TransientConfig(.003, .01, initial_velocity=changed_velocity),
            base_load_case=_constant_q4_load(),
            nodal_load_history=ContactNodalLoadHistory(
                ((0.0, 0.0), (.005, .4), (.01, 1.0)),
            ),
            restart_checkpoint=first.restart_checkpoint,
        )
    tampered = json.loads(json.dumps(first.restart_checkpoint))
    tampered["state"]["external_work_j"] += 1e-9
    with pytest.raises(ValueError, match="checkpoint integrity mismatch"):
        solve_transient_contact(
            changed_model, changed_contact,
            TransientConfig(.003, .01, initial_velocity=changed_velocity),
            base_load_case=_constant_q4_load(), nodal_load_history=history,
            restart_checkpoint=tampered,
        )


def test_q4_nodal_load_history_rejects_invalid_or_unsupported_input():
    for points in (((.001, 0.0), (.01, 1.0)),
                   ((0.0, 0.0), (.005, 1.0), (.004, 2.0)),
                   ((0.0, 0.0), (.01, float("nan")))):
        with pytest.raises(ValueError, match="nodal load history"):
            ContactNodalLoadHistory(points)
    history = ContactNodalLoadHistory(((0.0, 0.0), (.01, 1.0)))
    model, contact, velocity = _guided_plate()
    with pytest.raises(ValueError, match="nodal LoadCase"):
        solve_transient_contact(
            model, contact, TransientConfig(.001, .01,
                                            initial_velocity=velocity),
            nodal_load_history=history,
        )
    with pytest.raises(ValueError, match="cover t_end"):
        solve_transient_contact(
            model, contact, TransientConfig(.001, .011,
                                            initial_velocity=velocity),
            base_load_case=_constant_q4_load(), nodal_load_history=history,
        )
    near_coincident = ContactNodalLoadHistory(
        ((0.0, 0.0), (1e-12, .5), (.01, 1.0)),
    )
    with pytest.raises(ValueError, match="knots are closer"):
        solve_transient_contact(
            model, contact, TransientConfig(.001, .01,
                                            initial_velocity=velocity),
            base_load_case=_constant_q4_load(),
            nodal_load_history=near_coincident,
        )


def test_q4_constant_nodal_load_and_contact_match_mass_spring_impact():
    model, contact, velocity = _guided_plate()
    result = solve_transient_contact(
        model, contact, TransientConfig(.0001, .011,
                                        initial_velocity=velocity),
        base_load_case=_constant_q4_load(),
    )
    mass, downward_force, contact_stiffness = .05, .02, 1.0e5
    free_acceleration = downward_force/mass
    # Downward displacement first reaches the 0.01 m plane gap at t_hit.
    hit_time = (-1.0 + math.sqrt(1.0+2*free_acceleration*.01)) / free_acceleration
    impact_speed = 1.0 + free_acceleration*hit_time
    duration = .011-hit_time
    omega = math.sqrt(contact_stiffness/mass)
    penetration = (downward_force/contact_stiffness
                   * (1.0-math.cos(omega*duration))
                   + impact_speed/omega*math.sin(omega*duration))
    expected_displacement = -.01-penetration
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    assert result.contact[-1].active_pressure_dofs > 0
    assert result.external_nodal_work_j[-1] == pytest.approx(
        -0.02*result.displacements[-1, dof], abs=2e-12,
    )
    assert result.displacements[-1, dof] == pytest.approx(
        expected_displacement, abs=2.0e-5,
    )


def test_q4_ramped_nodal_load_during_contact_matches_forced_oscillator():
    model, contact, velocity = _guided_plate()
    history = ContactNodalLoadHistory(((0.0, 0.0), (.012, 1.0)))
    result = solve_transient_contact(
        model, contact, TransientConfig(.0001, .011,
                                        initial_velocity=velocity),
        base_load_case=_constant_q4_load(), nodal_load_history=history,
    )
    mass, stiffness, force_rate = .05, 1.0e5, .02/.012
    acceleration_rate = force_rate/mass
    hit_time = .01
    for _ in range(5):
        hit_time -= (
            hit_time+acceleration_rate*hit_time**3/6.0-.01
        )/(1.0+acceleration_rate*hit_time**2/2.0)
    impact_speed = 1.0+acceleration_rate*hit_time**2/2.0
    duration = .011-hit_time
    omega = math.sqrt(stiffness/mass)
    force_hit = force_rate*hit_time
    penetration = (
        (force_hit+force_rate*duration)/stiffness
        - force_hit/stiffness*math.cos(omega*duration)
        + (impact_speed-force_rate/stiffness)/omega
        * math.sin(omega*duration)
    )
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    assert result.contact[-1].active_pressure_dofs > 0
    assert result.displacements[-1, dof] == pytest.approx(
        -.01-penetration, abs=2.0e-5,
    )
    assert result.external_nodal_work_j[-1] > 0.0


def test_transient_development_load_admission_is_nodal_and_mass_consistent():
    model, contact, velocity = _guided_plate()
    load = _constant_q4_load()
    load.add_pressure_load(1, 1.0)
    with pytest.raises(ValueError, match="nodal-only"):
        solve_transient_contact(
            model, contact, TransientConfig(.001, .001,
                                            initial_velocity=velocity),
            base_load_case=load,
        )

    model, contact, velocity = _guided_s3_descriptor_impact()
    load = LoadCase("algebraic_s3_moment")
    load.add_nodal_load(1, moments=(0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="unresolved algebraic force"):
        solve_transient_contact(
            model, contact, TransientConfig(.001, .001,
                                            initial_velocity=velocity),
            base_load_case=load,
            solver_controls=ContactTransientSolverControls(
                allow_s3_descriptor_dev=True,
            ),
        )


def _guided_s3_descriptor_impact(*, tilt=0.0, span=1.0):
    director = np.asarray((tilt, 0.0, 1.0), dtype=float)
    director /= np.linalg.norm(director)
    tangent_x = span*np.asarray((director[2], 0.0, -director[0]))
    tangent_y = np.asarray((0.0, span, 0.0))
    model = FEModel("guided_s3_descriptor_impact")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, corner in enumerate((np.zeros(3), tangent_x, tangent_y), 1):
        model.add_node(node_id, *(corner + .035*director))
    model.add_element(1, create_shell_element(
        1, (1, 2, 3), "elastic", formulation="e4-pl-s3-v2d",
        thickness=.05, reference_normal=tuple(director),
        reference_surface_offset=.01,
    ))
    model.add_boundary_condition(BoundaryCondition(
        "in_plane_guide", (1, 2, 3), {"ux": 0.0, "uy": 0.0},
    ))
    contact = ContactDefinition(
        surfaces=(ContactSurface("physical_bottom", (1,), side="bottom"),),
        rigid_bodies=(RigidContactBody(
            "floor", "plane", (0.0, 0.0, -.01-span*director[0]),
            (0.0, 0.0, 1.0),
        ),),
        interactions=(ContactInteraction(
            "s3_floor", "physical_bottom", "floor", 1.0e5, .001,
        ),),
    )
    velocity = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (1, 2, 3):
        velocity[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -1.0
    return model, contact, velocity


def _uniform_s3_normal_load(force_n=-.02):
    load = LoadCase("uniform_s3_normal_dead_load")
    for node_id in (1, 2, 3):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, force_n/3.0))
    return load


def test_s3_descriptor_constant_load_free_motion_impact_and_restart():
    controls = ContactTransientSolverControls(allow_s3_descriptor_dev=True)
    model, contact, velocity = _guided_s3_descriptor_impact()
    velocity[:] = 0.0
    free = solve_transient_contact(
        model, contact, TransientConfig(.001, .005,
                                        initial_velocity=velocity),
        base_load_case=_uniform_s3_normal_load(), solver_controls=controls,
    )
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    assert free.accelerations[0, dof] == pytest.approx(-.8, abs=1e-10)
    assert free.velocities[-1, dof] == pytest.approx(-.004, abs=1e-10)
    assert free.displacements[-1, dof] == pytest.approx(-1e-5, abs=1e-10)
    assert free.external_nodal_work_j[-1] == pytest.approx(2e-7, abs=1e-12)
    assert all(state.active_pressure_dofs == 0 for state in free.contact)

    impact_model, impact_contact, impact_velocity = _guided_s3_descriptor_impact()
    impact = solve_transient_contact(
        impact_model, impact_contact,
        TransientConfig(.0001, .012, initial_velocity=impact_velocity),
        base_load_case=_uniform_s3_normal_load(), solver_controls=controls,
    )
    hit_time = (-1.0+math.sqrt(1.0+2.0*.8*.01))/.8
    impact_speed = 1.0+.8*hit_time
    duration = .011-hit_time
    omega = math.sqrt(1.0e5/.05)
    penetration = (
        .02/50000.0*(1.0-math.cos(omega*duration))
        + impact_speed/omega*math.sin(omega*duration)
    )
    sample = int(np.argmin(abs(impact.times-.011)))
    assert impact.contact[sample].active_pressure_dofs > 0
    assert impact.displacements[sample, dof] == pytest.approx(
        -.01-penetration, abs=2e-5,
    )
    assert all(state.gap_tolerance_met for state in impact.contact)

    split_model, split_contact, split_velocity = _guided_s3_descriptor_impact()
    first = solve_transient_contact(
        split_model, split_contact,
        TransientConfig(.0001, .011, initial_velocity=split_velocity),
        base_load_case=_uniform_s3_normal_load(), solver_controls=controls,
        emit_restart_checkpoint=True,
    )
    assert first.restart_checkpoint["version"] == 1
    resume_model, resume_contact, resume_velocity = _guided_s3_descriptor_impact()
    resumed = solve_transient_contact(
        resume_model, resume_contact,
        TransientConfig(.0001, .012, initial_velocity=resume_velocity),
        base_load_case=_uniform_s3_normal_load(), solver_controls=controls,
        restart_checkpoint=first.restart_checkpoint,
    )
    for name in ("displacements", "velocities", "accelerations",
                 "external_nodal_work_j"):
        combined = np.concatenate((getattr(first, name),
                                   getattr(resumed, name)[1:]), axis=0)
        np.testing.assert_allclose(combined, getattr(impact, name),
                                   rtol=0.0, atol=1e-10)
    changed_model, changed_contact, changed_velocity = _guided_s3_descriptor_impact()
    with pytest.raises(ValueError, match="checkpoint analysis is incompatible"):
        solve_transient_contact(
            changed_model, changed_contact,
            TransientConfig(.0001, .012, initial_velocity=changed_velocity),
            base_load_case=_uniform_s3_normal_load(-.021),
            solver_controls=controls, restart_checkpoint=first.restart_checkpoint,
        )


def test_s3_descriptor_nonuniform_load_initial_dae_acceleration():
    model, contact, velocity = _guided_s3_descriptor_impact()
    velocity[:] = 0.0
    load = LoadCase("nonuniform_s3_normal_dead_load")
    for node_id, force in zip((1, 2, 3), (-.010, -.007, -.003)):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, force))
    result = solve_transient_contact(
        model, contact, TransientConfig(.0005, .001,
                                        initial_velocity=velocity),
        base_load_case=load,
        solver_controls=ContactTransientSolverControls(
            allow_s3_descriptor_dev=True,
        ),
    )
    stiffness, _ = assemble_stiffness_matrix(model)
    mass, _ = assemble_mass_matrix(model)
    generalized_load, _ = assemble_load_vector(model, load)
    _, _, transform, _, free_dofs, _ = build_constraint_transformation(
        stiffness, np.zeros(mass.shape[0]), model,
    )
    reduced_mass_sparse = (transform.T @ mass @ transform).tocsr()
    reduced_mass = reduced_mass_sparse.toarray()
    reduced_stiffness = (transform.T @ stiffness @ transform).toarray()
    null = build_declared_algebraic_basis(
        model, mass, reduced_mass_sparse, transform, free_dofs,
        dense_size_limit=512,
    ).reduced_basis.toarray()
    reduced_load = np.asarray(transform.T @ generalized_load).reshape(-1)
    equations = np.vstack((reduced_mass, null.T @ reduced_stiffness))
    right = np.r_[reduced_load, np.zeros(null.shape[1])]
    independent_acceleration = np.linalg.lstsq(
        equations, right, rcond=None,
    )[0]
    actual = result.accelerations[0, free_dofs]
    np.testing.assert_allclose(actual, independent_acceleration,
                               rtol=1e-8, atol=1e-9)
    assert np.linalg.norm(reduced_mass @ actual-reduced_load) < 1e-9
    assert np.linalg.norm(null.T @ reduced_stiffness @ actual) < 1e-8
    assert all(state.gap_tolerance_met for state in result.contact)


def test_s3_descriptor_impact_retains_massless_rotations_and_skin_offset():
    model, contact, velocity = _guided_s3_descriptor_impact()
    result = solve_transient_contact(
        model, contact, TransientConfig(.0002, .012,
                                        initial_velocity=velocity),
        solver_controls=ContactTransientSolverControls(
            allow_s3_descriptor_dev=True,
        ),
    )
    assert result.profile.endswith("NEWMARK_DESCRIPTOR_DEV_V0")
    assert result.contact[0].minimum_gap == pytest.approx(.01, abs=1e-12)
    assert result.contact[-1].active_pressure_dofs > 0
    assert result.contact[-1].gap_tolerance_met
    assert max(result.equilibrium_residuals) < 1.0e-7
    omega = math.sqrt(1.0e5/.05)
    assert min(result.swept_minimum_gaps) == pytest.approx(
        -1.0/omega, abs=3.0e-6,
    )
    exact_normal = -.01-math.sin(omega*.002)/omega
    node_normal = model.mesh.dof_manager.get_node_dofs(1)[2]
    coarse_error = abs(result.displacements[-1, node_normal]-exact_normal)
    fine_model, fine_contact, fine_velocity = _guided_s3_descriptor_impact()
    fine = solve_transient_contact(
        fine_model, fine_contact,
        TransientConfig(.0001, .012, initial_velocity=fine_velocity),
        solver_controls=ContactTransientSolverControls(
            allow_s3_descriptor_dev=True,
        ),
    )
    assert abs(fine.displacements[-1, node_normal]-exact_normal) < coarse_error
    mass = .5*.05
    normal_speed = result.velocities[-1, node_normal]
    gap = result.contact[-1].minimum_gap
    energy = .5*mass*normal_speed**2 + .5*(1.0e5*.5)*min(gap, 0.0)**2
    assert abs(energy-.5*mass)/(.5*mass) < .01


def test_tilted_s3_descriptor_impact_couples_offset_torque():
    model, contact, velocity = _guided_s3_descriptor_impact(
        tilt=.002, span=.1,
    )
    result = solve_transient_contact(
        model, contact, TransientConfig(.0002, .012,
                                        initial_velocity=velocity),
        solver_controls=ContactTransientSolverControls(
            allow_s3_descriptor_dev=True,
        ),
    )
    assert result.contact[0].minimum_gap == pytest.approx(.01, abs=1e-10)
    assert result.contact[-1].active_pressure_dofs > 0
    assert result.contact[-1].gap_tolerance_met
    rotation_dofs = [model.mesh.dof_manager.get_node_dofs(node)[4]
                     for node in (1, 2, 3)]
    assert np.max(np.abs(result.displacements[-1, rotation_dofs])) > 1e-10
    assert max(result.equilibrium_residuals) < 1.0e-7
    K0, _ = assemble_stiffness_matrix(model)
    M, _ = assemble_mass_matrix(model)
    _, _, T, _, independent, _ = build_constraint_transformation(
        K0, np.zeros(M.shape[0]), model,
    )
    basis = build_declared_algebraic_basis(
        model, M, (T.T @ M @ T).tocsr(), T, independent,
        dense_size_limit=512,
    ).reduced_basis
    assert basis.shape[1] == 9
    with contact_scope(model, contact,
                       initial_displacement=np.zeros(M.shape[0])) as manager:
        final_position = result.displacements[-1]
        final_velocity = result.velocities[-1]
        final_acceleration = result.accelerations[-1]
        force, tangent, _ = _assemble_nonlinear_system(
            model, final_position, {}, 3, tangent=True,
            kinematics="von_karman",
        )
        reduced_balance = np.asarray(T.T @ (
            M @ final_acceleration + force
        )).reshape(-1)
        algebraic_balance = np.asarray(basis.T @ reduced_balance).reshape(-1)
        assert np.linalg.norm(reduced_balance) < 1.0e-7
        assert np.linalg.norm(algebraic_balance) < 1.0e-7
        # Use a different perturbation from the rate recovery's 1e-5 s.
        independent_step_s = 5.0e-6
        direction = independent_step_s*final_velocity
        _, tangent_plus, _ = _assemble_nonlinear_system(
            model, final_position+direction, {}, 3, tangent=True,
            kinematics="von_karman",
        )
        _, tangent_minus, _ = _assemble_nonlinear_system(
            model, final_position-direction, {}, 3, tangent=True,
            kinematics="von_karman",
        )
        velocity_rate = np.asarray(
            basis.T @ T.T @ (tangent @ final_velocity)
        ).reshape(-1)
        curvature = np.asarray(
            (tangent_plus-tangent_minus) @ final_velocity
        ).reshape(-1)/(2.0*independent_step_s)
        acceleration_rate = np.asarray(
            basis.T @ T.T @ (tangent @ final_acceleration+curvature)
        ).reshape(-1)
        assert np.linalg.norm(velocity_rate) < 1.0e-8
        assert np.linalg.norm(acceleration_rate) < 1.0e-7
        contact_force, _, contact_state = manager.evaluate(
            final_position, tangent=False,
        )
    step_s = result.times[-1]-result.times[-2]
    inertial_velocity_defect = M @ (
        result.velocities[-1]-result.velocities[-2]
        - .5*step_s*(result.accelerations[-2]+result.accelerations[-1])
    )
    inertial_position_defect = M @ (
        result.displacements[-1]-result.displacements[-2]
        - step_s*result.velocities[-2]
        - .25*step_s**2*(result.accelerations[-2]+result.accelerations[-1])
    )
    assert np.linalg.norm(inertial_velocity_defect) < 1.0e-10
    assert np.linalg.norm(inertial_position_defect) < 1.0e-10
    director = model.mesh.elements[1].physical_reference_director
    actual_couple = -sum(contact_force[dof] for dof in rotation_dofs)
    expected_couple = .035*director[0]*contact_state.shell_force[2]
    assert actual_couple == pytest.approx(expected_couple, rel=1e-10, abs=1e-10)


def test_s3_descriptor_impact_requires_opt_in_and_consistent_open_start():
    model, contact, velocity = _guided_s3_descriptor_impact()
    with pytest.raises(ValueError, match="opted-in single S3"):
        solve_transient_contact(
            model, contact, TransientConfig(.0002, .001,
                                            initial_velocity=velocity),
        )
    controls = ContactTransientSolverControls(allow_s3_descriptor_dev=True)
    nonuniform = velocity.copy()
    nonuniform[model.mesh.dof_manager.get_node_dofs(1)[2]] = -.8
    with pytest.raises(ValueError, match="uniform rigid-normal"):
        solve_transient_contact(
            model, contact, TransientConfig(.0002, .001,
                                            initial_velocity=nonuniform),
            solver_controls=controls,
        )
    touching = replace(
        contact, rigid_bodies=(replace(contact.rigid_bodies[0],
                                       point=(0.0, 0.0, 0.0)),),
    )
    with pytest.raises(ValueError, match="strictly open initial gap"):
        solve_transient_contact(
            model, touching, TransientConfig(.0002, .001,
                                             initial_velocity=velocity),
            solver_controls=controls,
        )


def test_s3_descriptor_active_contact_restart_matches_fresh_model():
    controls = ContactTransientSolverControls(allow_s3_descriptor_dev=True)
    model, contact, velocity = _guided_s3_descriptor_impact()
    full = solve_transient_contact(
        model, contact, TransientConfig(.0002, .012,
                                        initial_velocity=velocity),
        solver_controls=controls,
    )
    first_model, first_contact, first_velocity = _guided_s3_descriptor_impact()
    first = solve_transient_contact(
        first_model, first_contact,
        TransientConfig(.0002, .011, initial_velocity=first_velocity),
        solver_controls=controls, emit_restart_checkpoint=True,
    )
    assert first.contact[-1].active_pressure_dofs > 0
    resume_model, resume_contact, resume_velocity = _guided_s3_descriptor_impact()
    resumed = solve_transient_contact(
        resume_model, resume_contact,
        TransientConfig(.0002, .012, initial_velocity=resume_velocity),
        solver_controls=controls, restart_checkpoint=first.restart_checkpoint,
    )
    for name in ("displacements", "velocities", "accelerations"):
        combined = np.vstack((getattr(first, name),
                              getattr(resumed, name)[1:]))
        np.testing.assert_allclose(combined, getattr(full, name),
                                   rtol=1e-10, atol=1e-10)


def test_s3_descriptor_cutback_retries_from_committed_state():
    controls = ContactTransientSolverControls(allow_s3_descriptor_dev=True)
    model, contact, velocity = _guided_s3_descriptor_impact()
    config = TransientConfig(.012, .024, initial_velocity=velocity)
    result = solve_transient_contact(
        model, contact, config, solver_controls=controls,
    )
    assert result.cutbacks > 0
    assert result.times[-1] == pytest.approx(.024)
    assert np.all(np.diff(result.times) > 0.0)
    assert all(state.gap_tolerance_met for state in result.contact)
    assert max(result.equilibrium_residuals) < 1.0e-7
    assert active_contact_manager() is None
    mass, _ = assemble_mass_matrix(model)
    for step in range(1, len(result.times)):
        dt = result.times[step]-result.times[step-1]
        velocity_defect = mass @ (
            result.velocities[step]-result.velocities[step-1]
            - .5*dt*(result.accelerations[step-1]+result.accelerations[step])
        )
        position_defect = mass @ (
            result.displacements[step]-result.displacements[step-1]
            - dt*result.velocities[step-1]
            - .25*dt**2*(result.accelerations[step-1]
                         + result.accelerations[step])
        )
        assert np.linalg.norm(velocity_defect) < 1.0e-10
        assert np.linalg.norm(position_defect) < 1.0e-10

    fresh_model, fresh_contact, fresh_velocity = _guided_s3_descriptor_impact()
    repeat = solve_transient_contact(
        fresh_model, fresh_contact,
        TransientConfig(.012, .024, initial_velocity=fresh_velocity),
        solver_controls=controls,
    )
    assert repeat.cutbacks == result.cutbacks
    np.testing.assert_array_equal(repeat.times, result.times)
    for name in ("displacements", "velocities", "accelerations"):
        np.testing.assert_allclose(getattr(repeat, name), getattr(result, name),
                                   rtol=0.0, atol=1.0e-12)


def test_uniform_impact_matches_independent_mass_spring_solution():
    model, contact, velocity = _guided_plate()
    result = solve_transient_contact(
        model, contact, TransientContactSettings(0.0002, 0.016, velocity),
    )
    # The guided plate has one rigid normal mode: m = density*area*thickness,
    # k = declared pressure stiffness*area.  Its first contact starts at .01 s.
    omega = math.sqrt(1.0e5 / 0.05)
    release_time = 0.01 + math.pi / omega
    exact_final_displacement = -0.01 + (0.016 - release_time)
    assert result.profile.endswith("NEWMARK_PENALTY_DEV_V1")
    assert result.times[-1] == pytest.approx(0.016)
    assert result.cutbacks == 0
    assert result.displacements[-1, 2] == pytest.approx(
        exact_final_displacement, abs=1.0e-5,
    )
    assert result.velocities[-1, 2] == pytest.approx(1.0, abs=0.01)
    assert min(result.swept_minimum_gaps) == pytest.approx(
        -1.0 / omega, abs=2.0e-6,
    )
    assert max(abs(value) for value in result.equilibrium_residuals) < 1.0e-7
    assert all(value.gap_tolerance_met for value in result.contact)
    assert active_contact_manager() is None
    finer_model, finer_contact, finer_velocity = _guided_plate()
    finer = solve_transient_contact(
        finer_model, finer_contact,
        TransientContactSettings(0.0001, 0.016, finer_velocity),
    )
    assert abs(finer.velocities[-1, 2] - 1.0) < abs(result.velocities[-1, 2] - 1.0)
    assert abs(finer.velocities[-1, 2] ** 2 - 1.0) < 0.01
    assert abs(finer.displacements[-1, 2] - exact_final_displacement) < 5.0e-6


def test_hermite_sweep_detects_interior_crossing_with_open_endpoints():
    model, contact, _ = _guided_plate()
    manager = _PlanarQ4PlaneManager(model, contact)
    size = model.mesh.dof_manager.total_dofs
    start = np.zeros(size)
    end = start.copy()
    first_velocity = np.zeros(size)
    last_velocity = np.zeros(size)
    for node_id in (1, 2, 3, 4):
        dof = model.mesh.dof_manager.get_node_dofs(node_id)[2]
        first_velocity[dof] = -10.0
        last_velocity[dof] = 10.0
    assert manager.evaluate(start)[2].minimum_gap == pytest.approx(0.01)
    assert manager.evaluate(end)[2].minimum_gap == pytest.approx(0.01)
    assert _minimum_hermite_gap(
        manager, start, end, first_velocity, last_velocity, 0.01,
    ) == pytest.approx(-0.015)


def test_rejected_step_leaves_model_reusable():
    model, contact, velocity = _guided_plate()
    with pytest.raises(RuntimeError, match="committed state retained"):
        solve_transient_contact(
            model, contact,
            TransientContactSettings(
                0.012, 0.012, velocity,
                maximum_newton_iterations=1, maximum_cutbacks=0,
            ),
        )
    assert active_contact_manager() is None
    result = solve_transient_contact(
        model, contact, TransientContactSettings(0.0002, 0.012, velocity),
    )
    assert result.times[-1] == pytest.approx(0.012)
    assert result.contact[-1].active_pressure_dofs > 0


def test_loaded_rejected_step_keeps_q4_model_and_work_reusable():
    model, contact, velocity = _guided_plate()
    load = _constant_q4_load()
    with pytest.raises(RuntimeError, match="committed state retained"):
        solve_transient_contact(
            model, contact,
            TransientConfig(.012, .012, initial_velocity=velocity),
            base_load_case=load,
            solver_controls=ContactTransientSolverControls(
                maximum_newton_iterations=1, maximum_cutbacks=0,
            ),
        )
    assert active_contact_manager() is None
    retry = solve_transient_contact(
        model, contact,
        TransientConfig(.0002, .012, initial_velocity=velocity),
        base_load_case=load,
    )
    fresh_model, fresh_contact, fresh_velocity = _guided_plate()
    fresh = solve_transient_contact(
        fresh_model, fresh_contact,
        TransientConfig(.0002, .012, initial_velocity=fresh_velocity),
        base_load_case=_constant_q4_load(),
    )
    assert retry.contact[-1].active_pressure_dofs > 0
    for name in ("times", "displacements", "velocities", "accelerations",
                 "external_nodal_work_j"):
        np.testing.assert_allclose(getattr(retry, name), getattr(fresh, name),
                                   rtol=0.0, atol=1.0e-12)


def test_transient_config_initial_state_and_sparse_output_keep_event_checks():
    full_model, full_contact, velocity = _guided_plate()
    initial = np.zeros_like(velocity)
    for node_id in (1, 2, 3, 4):
        initial[full_model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -0.002
    full = solve_transient_contact(
        full_model, full_contact,
        TransientConfig(0.0002, 0.012, initial_displacement=initial,
                        initial_velocity=velocity),
    )
    sampled_model, sampled_contact, sampled_velocity = _guided_plate()
    sampled = solve_transient_contact(
        sampled_model, sampled_contact,
        TransientConfig(0.0002, 0.012, save_every=7,
                        initial_displacement=initial,
                        initial_velocity=sampled_velocity),
        solver_controls=ContactTransientSolverControls(),
    )
    assert len(sampled.newton_iterations) == len(full.newton_iterations)
    assert len(sampled.times) < len(full.times)
    assert sampled.external_nodal_work_j.shape == sampled.times.shape
    np.testing.assert_array_equal(sampled.external_nodal_work_j,
                                  np.zeros_like(sampled.times))
    assert sampled.times[0] == 0.0
    assert sampled.times[-1] == pytest.approx(0.012)
    assert sampled.displacements[0, 2] == pytest.approx(-0.002)
    omega = math.sqrt(1.0e5 / 0.05)
    release_time = 0.008 + math.pi / omega
    assert sampled.displacements[-1, 2] == pytest.approx(
        -0.01 + (0.012 - release_time), abs=1.0e-5,
    )
    np.testing.assert_allclose(sampled.displacements[-1], full.displacements[-1])
    np.testing.assert_allclose(sampled.velocities[-1], full.velocities[-1])
    assert min(sampled.swept_minimum_gaps) == pytest.approx(
        min(full.swept_minimum_gaps), abs=1.0e-12,
    )
    assert max(sampled.equilibrium_residuals) == pytest.approx(
        max(full.equilibrium_residuals), abs=1.0e-12,
    )
    for output_index in range(1, len(sampled.times)):
        previous_step = int(np.argmin(np.abs(
            full.times - sampled.times[output_index - 1]
        )))
        current_step = int(np.argmin(np.abs(
            full.times - sampled.times[output_index]
        )))
        assert full.times[previous_step] == pytest.approx(
            sampled.times[output_index - 1], abs=1.0e-14,
        )
        assert full.times[current_step] == pytest.approx(
            sampled.times[output_index], abs=1.0e-14,
        )
        assert sampled.swept_minimum_gaps[output_index] == pytest.approx(
            min(full.swept_minimum_gaps[previous_step + 1:current_step + 1]),
            abs=1.0e-12,
        )
        assert sampled.equilibrium_residuals[output_index] == pytest.approx(
            max(full.equilibrium_residuals[previous_step + 1:current_step + 1]),
            abs=1.0e-12,
        )


@pytest.mark.parametrize("option", (
    {"beta": 0.3}, {"rayleigh_alpha": 0.01}, {"output_nodes": (1,)},
))
def test_transient_config_rejects_unadmitted_options(option):
    model, contact, velocity = _guided_plate()
    with pytest.raises(ValueError, match="not admitted|only undamped"):
        solve_transient_contact(
            model, contact,
            TransientConfig(0.0002, 0.001, initial_velocity=velocity, **option),
        )


def test_transient_config_rejects_inconsistent_initial_displacement():
    model, contact, velocity = _guided_plate()
    initial = np.zeros_like(velocity)
    initial[model.mesh.dof_manager.get_node_dofs(1)[0]] = 0.001
    with pytest.raises(ValueError, match="initial_displacement does not satisfy"):
        solve_transient_contact(
            model, contact,
            TransientConfig(0.0002, 0.001,
                            initial_displacement=initial,
                            initial_velocity=velocity),
        )


def test_active_contact_restart_replays_newmark_path_from_fresh_model():
    full_model, full_contact, velocity = _guided_plate()
    full = solve_transient_contact(
        full_model, full_contact,
        TransientConfig(0.0002, 0.016, initial_velocity=velocity),
    )
    first_model, first_contact, first_velocity = _guided_plate()
    first = solve_transient_contact(
        first_model, first_contact,
        TransientConfig(0.0002, 0.012, initial_velocity=first_velocity),
        emit_restart_checkpoint=True,
    )
    assert first.contact[-1].active_pressure_dofs > 0
    checkpoint = json.loads(json.dumps(first.restart_checkpoint))
    resumed_model, resumed_contact, resumed_velocity = _guided_plate()
    resumed = solve_transient_contact(
        resumed_model, resumed_contact,
        TransientConfig(0.0002, 0.016, initial_velocity=resumed_velocity),
        restart_checkpoint=checkpoint,
    )
    for name in ("times", "displacements", "velocities", "accelerations"):
        combined = np.concatenate((getattr(first, name),
                                   getattr(resumed, name)[1:]), axis=0)
        np.testing.assert_allclose(combined, getattr(full, name),
                                   rtol=1.0e-11, atol=1.0e-11)
    assert first.newton_iterations + resumed.newton_iterations == full.newton_iterations
    np.testing.assert_allclose(
        np.r_[first.swept_minimum_gaps, resumed.swept_minimum_gaps[1:]],
        full.swept_minimum_gaps, rtol=0.0, atol=1.0e-11,
    )


def test_restart_preserves_configured_nonzero_initial_pose():
    full_model, full_contact, velocity = _guided_plate()
    initial = np.zeros_like(velocity)
    for node_id in (1, 2, 3, 4):
        initial[full_model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -0.002
    full = solve_transient_contact(
        full_model, full_contact,
        TransientConfig(0.0002, 0.012, initial_displacement=initial,
                        initial_velocity=velocity),
    )
    split_model, split_contact, split_velocity = _guided_plate()
    first = solve_transient_contact(
        split_model, split_contact,
        TransientConfig(0.0002, 0.010, initial_displacement=initial,
                        initial_velocity=split_velocity),
        emit_restart_checkpoint=True,
    )
    assert first.contact[-1].active_pressure_dofs > 0
    resume_model, resume_contact, resume_velocity = _guided_plate()
    resumed = solve_transient_contact(
        resume_model, resume_contact,
        TransientConfig(0.0002, 0.012, initial_displacement=initial,
                        initial_velocity=resume_velocity),
        restart_checkpoint=first.restart_checkpoint,
    )
    np.testing.assert_allclose(
        np.vstack((first.displacements, resumed.displacements[1:])),
        full.displacements, rtol=1.0e-11, atol=1.0e-11,
    )


def test_transient_restart_rejects_tampering_and_changed_time_step():
    model, contact, velocity = _guided_plate()
    first = solve_transient_contact(
        model, contact, TransientConfig(0.0002, 0.012,
                                        initial_velocity=velocity),
        emit_restart_checkpoint=True,
    )
    tampered = json.loads(json.dumps(first.restart_checkpoint))
    tampered["state"]["q"][2] += 0.001
    resume_model, resume_contact, resume_velocity = _guided_plate()
    with pytest.raises(ValueError, match="integrity mismatch"):
        solve_transient_contact(
            resume_model, resume_contact,
            TransientConfig(0.0002, 0.016,
                            initial_velocity=resume_velocity),
            restart_checkpoint=tampered,
        )
    with pytest.raises(ValueError, match="analysis is incompatible"):
        solve_transient_contact(
            resume_model, resume_contact,
            TransientConfig(0.0001, 0.016,
                            initial_velocity=resume_velocity),
            restart_checkpoint=first.restart_checkpoint,
        )
    changed_contact = replace(
        resume_contact,
        interactions=(replace(resume_contact.interactions[0],
                              normal_stiffness=2.0e5),),
    )
    with pytest.raises(ValueError, match="analysis is incompatible"):
        solve_transient_contact(
            resume_model, changed_contact,
            TransientConfig(0.0002, 0.016,
                            initial_velocity=resume_velocity),
            restart_checkpoint=first.restart_checkpoint,
        )
    with pytest.raises(ValueError, match="requires save_every=1"):
        solve_transient_contact(
            resume_model, resume_contact,
            TransientConfig(0.0002, 0.016, save_every=2,
                            initial_velocity=resume_velocity),
            restart_checkpoint=first.restart_checkpoint,
        )
    assert active_contact_manager() is None


def test_transient_restart_does_not_export_truncated_terminal_step():
    model, contact, velocity = _guided_plate()
    with pytest.raises(ValueError, match="untruncated final time step"):
        solve_transient_contact(
            model, contact,
            TransientConfig(0.0002, 0.0121, initial_velocity=velocity),
            emit_restart_checkpoint=True,
        )
    assert active_contact_manager() is None
    with pytest.raises(ValueError, match="one accepted time step"):
        solve_transient_contact(
            model, contact,
            TransientConfig(0.0002, 0.0, initial_velocity=velocity),
            emit_restart_checkpoint=True,
        )
    reused = solve_transient_contact(
        model, contact,
        TransientConfig(0.0002, 0.0121, initial_velocity=velocity),
    )
    assert reused.times[-1] == pytest.approx(0.0121)
