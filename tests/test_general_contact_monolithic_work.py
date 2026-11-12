"""Mapped two-body virtual work for the opt-in nodal-force equations."""

import numpy as np

from anyfesolver import solve_static_nonlinear
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.general_contact.core import suspend_contact
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import (
    linearize, raw_certificate, reference_nodal_support,
)
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case
from scripts.general_contact_d1_free_target_press_probe import build_free_target_case


def test_mapped_source_target_force_and_smooth_tangent():
    model, definition, settings, _program = build_case(
        4, load_profile="distributed", total_force_n=0.3,
        target_right_x_m=1.3,
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    z = 0.3*reference_nodal_support(manager.law)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    rng = np.random.default_rng(41)
    direction = rng.normal(size=len(u))
    direction /= np.linalg.norm(direction)

    base = linearize(manager, u, z, tangent=True)
    step = 2e-7
    plus = linearize(manager, u+step*direction, z, tangent=False)
    minus = linearize(manager, u-step*direction, z, tangent=False)
    force_fd = (plus.force-minus.force)/(2*step)
    predicted = base.tangent @ direction
    assert np.linalg.norm(force_fd-predicted)/np.linalg.norm(predicted) < 1e-5
    gap_fd = (plus.state.normalized_gap-minus.state.normalized_gap)/(2*step)
    gap_predicted = base.gap_gradient @ direction
    assert np.linalg.norm(gap_fd-gap_predicted)/np.linalg.norm(gap_predicted) < 1e-5

    local_physical = -base.local_force.reshape(-1, 3)
    source_count = len(manager.law.source_node_ids)
    np.testing.assert_allclose(
        local_physical[:source_count].sum(axis=0)
        + local_physical[source_count:].sum(axis=0),
        0.0, atol=1e-9,
    )
    source, target = manager._positions(u)
    position = np.vstack((source, target))
    np.testing.assert_allclose(
        np.cross(position, local_physical).sum(axis=0),
        0.0, atol=1e-9,
    )
    target_direction = np.zeros_like(u)
    for node_id in manager.law.target_node_ids:
        target_direction[model.mesh.dof_manager.get_node_dofs(node_id)] = (
            rng.normal(size=6)
        )
    target_direction /= np.linalg.norm(target_direction)
    target_plus = linearize(manager, u+step*target_direction, z, tangent=False)
    target_minus = linearize(manager, u-step*target_direction, z, tangent=False)
    target_work = (np.dot(z, target_plus.state.normalized_gap)
                   - np.dot(z, target_minus.state.normalized_gap))/(2*step)
    np.testing.assert_allclose(
        target_work, -np.dot(base.force, target_direction),
        rtol=1e-5, atol=1e-8,
    )
    target_force_fd = (target_plus.force-target_minus.force)/(2*step)
    target_tangent = base.tangent @ target_direction
    assert np.linalg.norm(target_force_fd-target_tangent)/np.linalg.norm(
        target_tangent
    ) < 1e-5
    source_dofs = np.concatenate([
        model.mesh.dof_manager.get_node_dofs(node_id)
        for node_id in manager.law.source_node_ids
    ])
    assert np.linalg.norm(target_tangent[source_dofs]) > 1e-9
    _force, _tangent, projected = manager.evaluate(u, tangent=False)
    reported = raw_certificate(manager, u, base, projected)
    assert reported.energy_evaluations == base.evaluations
    assert reported.pair_geometry_evaluations == base.evaluations
    np.testing.assert_allclose(reported.numerical_pressure_pa,
                               base.state.pressure, rtol=0.0, atol=0.0)


def test_deformed_s3_target_translation_and_rotation_contact_work():
    """Check authoritative target work at an accepted coupled press pose."""
    model, definition, settings, program = build_free_target_case(
        4, steps_per_segment=2, enforcement_solver="monolithic_active_set",
    )
    accepted = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert accepted.status == "completed", accepted.failure_reason
    assert len(accepted.steps) == 2
    u = accepted.displacements
    z = np.asarray(accepted.to_restart_checkpoint()["path_state"]["normal_forces"])
    manager = GeneralS3PatchManager(model, definition, settings)
    base = linearize(manager, u, z, tangent=True)
    assert np.count_nonzero(z > 0.0) > 0
    target_node = manager.law.target_node_ids[2]
    dofs = model.mesh.dof_manager.get_node_dofs(target_node)
    reference_target = manager._positions(np.zeros_like(u))[1]
    current_target = manager._positions(u)[1]
    assert np.linalg.norm(current_target[2]-reference_target[2]) > 1.0e-5
    assert abs(u[dofs[4]]) > 1.0e-5
    step = 1.0e-7
    for axis in (2, 4):  # Normal displacement and physical director rotation.
        direction = np.zeros_like(u)
        direction[dofs[axis]] = 1.0
        plus = linearize(manager, u+step*direction, z, tangent=False)
        minus = linearize(manager, u-step*direction, z, tangent=False)
        observed = (plus.force-minus.force)/(2.0*step)
        predicted = base.tangent @ direction
        assert np.linalg.norm(predicted) > 1.0e-6
        assert np.linalg.norm(observed-predicted)/np.linalg.norm(predicted) < 1.0e-5
        gap_work = np.dot(z, plus.state.normalized_gap
                          - minus.state.normalized_gap)/(2.0*step)
        np.testing.assert_allclose(gap_work, -np.dot(base.force, direction),
                                   rtol=1.0e-5, atol=1.0e-16)
        if axis == 4:
            # Check the physical moment row explicitly; the full-vector norm
            # otherwise lets translational force rows mask an error here.
            assert abs(predicted[dofs[axis]]) > 5.0e-13
            np.testing.assert_allclose(observed[dofs[axis]],
                                       predicted[dofs[axis]],
                                       rtol=1.0e-5, atol=0.0)
            positive_target = manager._positions(u+step*direction)[1][2]
            negative_target = manager._positions(u-step*direction)[1][2]
            assert np.linalg.norm((positive_target-negative_target)
                                  /(2.0*step)) > 0.01
            assert abs(base.force[dofs[axis]]) > 1.0e-12
    source, target = manager._positions(u)
    positions = np.vstack((source, target))
    physical_forces = -base.local_force.reshape(-1, 3)
    np.testing.assert_allclose(physical_forces.sum(axis=0), 0.0, atol=1.0e-9)
    np.testing.assert_allclose(
        np.cross(positions, physical_forces).sum(axis=0),
        0.0, atol=1.0e-9,
    )

    # Recover the two individual clamp reactions from the accepted structural
    # residual. The public result sums their components, so checking only that
    # six-vector would lose the force lever arm between the two clamp nodes.
    with suspend_contact():
        structural_force, _tangent, _states = _assemble_nonlinear_system(
            model, u, accepted.element_states, 5, tangent=False,
            kinematics="von_karman", require_full_coordinates=True,
        )
    imbalance = structural_force + base.force
    clamp_force = np.zeros(3)
    clamp_moment = np.zeros(3)
    clamp_components = np.zeros(6)
    for node in (manager.law.target_node_ids[0],
                 manager.law.target_node_ids[-1]):
        node_dofs = model.mesh.dof_manager.get_node_dofs(node)
        reaction = imbalance[node_dofs]
        reference = model.mesh.nodes[node]
        current = (np.array((reference.x, reference.y, reference.z))
                   + u[node_dofs[:3]])
        clamp_components += reaction
        clamp_force += reaction[:3]
        clamp_moment += np.cross(current, reaction[:3]) + reaction[3:6]
    np.testing.assert_allclose(
        clamp_components,
        accepted.steps[-1].support_reactions["target_left_clamp"],
        rtol=0.0, atol=1.0e-12,
    )
    target_force = np.zeros(3)
    target_moment = np.zeros(3)
    for node in manager.law.target_node_ids:
        node_dofs = model.mesh.dof_manager.get_node_dofs(node)
        reference = model.mesh.nodes[node]
        current = (np.array((reference.x, reference.y, reference.z))
                   + u[node_dofs[:3]])
        physical = -base.force[node_dofs]
        target_force += physical[:3]
        target_moment += np.cross(current, physical[:3]) + physical[3:6]
    force_scale = np.linalg.norm(target_force)
    moment_scale = max(np.linalg.norm(target_moment), 1.3*force_scale)
    assert force_scale > 1.0e-5
    assert np.linalg.norm(clamp_force + target_force)/force_scale < 1.0e-6
    # Tolerance drift under the adopted S3 VK law: VK membrane forces are not
    # rigid-rotation objective, so the target's internal forces carry a net
    # moment (measured 2.733e-10 N m = 1.236e-6 of moment_scale at this pose;
    # 4.2e-7 before VK). Free-node residual moment is 8e-16. Tolerance =
    # max(1e-6, 1.5 x measured) rounded up to 2e-6.
    assert np.linalg.norm(clamp_moment + target_moment)/moment_scale < 2.0e-6
