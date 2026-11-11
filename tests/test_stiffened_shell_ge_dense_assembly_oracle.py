"""Bounded independent dense scatter oracle for the sparse GE point owner."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver._ge_beam3_variational_shell import response as variational_shell_response
from anyfesolver._stiffened_shell_ge_experimental import (
    ExperimentalGEAnalysis, NodalAction,
)
from anyfesolver._stiffened_shell_ge_finite_experimental import (
    FinitePointDriver, _position_pair,
)
from anyfesolver.corotational import corotational_element_response

from test_stiffened_shell_ge_experimental import _panel


def _dense_reference(owner, definition):
    """Assemble local operators by a separate dense gather/scatter path."""
    count = owner._K.shape[0]
    shell_count = owner._shell_count
    residual = np.zeros(count)
    stiffness = np.zeros((count, count))

    for eid, element, _compiled_slots, origin in owner._shell_parts:
        slots = np.array([6*owner._shell_index[node]+axis
                          for node in element.node_ids for axis in range(6)])
        if len(element.node_ids) == 3:
            force, tangent, _ = element.compute_nonlinear_response(
                owner._shell_model.mesh,
                owner._shell_model.get_material("shell"),
                np.zeros(18), origin, 3, True)
        else:
            force, tangent, _ = corotational_element_response(
                owner._shell_model, eid, element, np.zeros(24), True,
                committed_state=origin, num_layers=3,
                tangent_mode="consistent")
        np.add.at(residual, slots, force)
        stiffness[np.ix_(slots, slots)] += tangent

    beam_nodal = 6*len(owner._beam_ids)
    for member, (element, _compiled_slots, origin) in enumerate(
            owner._beam_parts):
        slots = np.array(
            [shell_count+6*owner._beam_index[node]+axis
             for node in element.node_ids for axis in range(6)] +
            list(range(shell_count+beam_nodal+24*member,
                       shell_count+beam_nodal+24*(member+1))))
        positions = np.array([owner._beam_nodes[node][0]
                              for node in element.node_ids])
        frames = np.array([owner._beam_nodes[node][1]
                           for node in element.node_ids])
        response = element.operator.evaluate(
            positions, np.zeros((3, 3)), frames,
            np.tile(np.eye(3), (2, 1, 1)), np.zeros(18),
            origin=origin, check=element.operator.guard)
        tangent = spatial_jacobian(
            response.residual, response.hessian+response.hessian_low)
        np.add.at(residual, slots, response.residual)
        stiffness[np.ix_(slots, slots)] += tangent

    constraints = np.zeros((6*len(definition.points), count))
    for joint_index, tie in enumerate(definition.points):
        shell_node = owner._shell_index[tie.shell_node]
        beam_node = owner._beam_index[tie.beam_node]
        positions = np.array((owner._shell_xyz[shell_node],
                              owner._beam_nodes[tie.beam_node][0]))
        frames = np.array((np.eye(3),
                           owner._beam_nodes[tie.beam_node][1]))
        joint = RigidPoseJoint(positions, frames)
        evaluated = joint.evaluate(positions, frames, np.zeros(6))
        assert np.max(np.abs(evaluated.constraints)) <= 1e-10
        slots = np.r_[np.arange(6*shell_node, 6*shell_node+6),
                      np.arange(shell_count+6*beam_node,
                                shell_count+6*beam_node+6)]
        constraints[6*joint_index:6*joint_index+6, slots] = (
            evaluated.constraint_jacobian)
    return residual, stiffness, constraints


@pytest.mark.parametrize("q4,s3", [
    (((1, 2, 5, 4), (2, 3, 6, 5)), ()),
    (((1, 2, 5, 4),), ((2, 3, 6), (2, 6, 5))),
])
def test_dense_local_assembly_matches_sparse_owner(q4, s3):
    base = _panel()
    definition = replace(base, shell=replace(
        base.shell, q4_elements=q4, s3_elements=s3))
    owner = ExperimentalGEAnalysis(definition)
    residual, stiffness, constraints = _dense_reference(owner, definition)
    assert np.max(np.abs(residual)) < 1e-10
    sparse_stiffness = owner._K.toarray()
    # One mixed Q4/S3 entry nearly cancels under a different accumulation
    # order. Judge assembly against the registered relative matrix criterion,
    # while retaining the raw difference for failure diagnostics.
    stiffness_error = np.linalg.norm(stiffness-sparse_stiffness) / np.linalg.norm(
        sparse_stiffness)
    assert stiffness_error <= 1e-10, stiffness_error
    np.testing.assert_allclose(constraints, owner._C.toarray(),
                               rtol=1e-10, atol=1e-12)

    action = (NodalAction("beam", 105, (0., 0., 100., 0., 0., 0.)),)
    sparse_result = owner.solve_linear(action)
    load = owner._accepted[1]
    free = owner._free
    dense_system = np.block([
        [stiffness[np.ix_(free, free)], constraints[:, free].T],
        [constraints[:, free], np.zeros((len(constraints), len(constraints)))],
    ])
    dense_solution = np.linalg.solve(
        dense_system, np.r_[load[free], np.zeros(len(constraints))])
    sparse_coordinates = np.r_[sparse_result.shell_displacement,
                               sparse_result.beam_coordinates]
    coordinate_error = np.linalg.norm(
        sparse_coordinates[free]-dense_solution[:len(free)]) / np.linalg.norm(
            dense_solution[:len(free)])
    multiplier_error = np.linalg.norm(
        sparse_result.multipliers-dense_solution[len(free):]) / np.linalg.norm(
            dense_solution[len(free):])
    assert coordinate_error <= 1e-10, coordinate_error
    assert multiplier_error <= 1e-10, multiplier_error
    q = np.zeros(len(load))
    q[free] = dense_solution[:len(free)]
    reactions = stiffness@q+constraints.T@dense_solution[len(free):]-load
    reaction_error = (np.linalg.norm(
        sparse_result.reactions[owner._fixed]-reactions[owner._fixed]) /
        np.linalg.norm(reactions[owner._fixed]))
    assert reaction_error <= 1e-10, reaction_error


def _dense_finite(owner, definition, driver, state, factor, load):
    """Gather finite local operators by model IDs, independent of CSR scatter."""
    physical = owner._K.shape[0]
    size = physical+6*len(definition.points)
    residual = np.zeros(size)
    tangent = np.zeros((size, size))
    shell_count = owner._shell_count
    beam_nodal = 6*len(owner._beam_ids)
    beam_displacements = state.q[shell_count:shell_count+beam_nodal].reshape(-1, 6)[:, :3]
    beam_reference = np.array([owner._beam_nodes[node][0]
                               for node in owner._beam_ids])
    beam_positions, beam_low = _position_pair(beam_reference, beam_displacements)

    def add(slots, force, local):
        np.add.at(residual, slots, force)
        tangent[np.ix_(slots, slots)] += local

    for (eid, element, _, _), origin in zip(owner._shell_parts,
                                            driver._shell_origins):
        slots = np.array([6*owner._shell_index[node]+axis
                          for node in element.node_ids for axis in range(6)])
        if len(element.node_ids) == 3:
            force, local, _ = variational_shell_response(
                owner._shell_model, element, state.q[slots], origin, 3)
        else:
            # Mirror the driver's checkpoint-bound Q4 numerical policy so the
            # oracle assembles the identical local operator; the scatter, not
            # the rotation-difference formulation, is under assessment.
            force, local, _ = corotational_element_response(
                owner._shell_model, eid, element, state.q[slots], True,
                committed_state=origin, num_layers=3,
                tangent_mode="consistent", centered_shell_reference=True,
                stable_shell_rotation=driver._stable_q4)
        add(slots, force, local)

    for member, ((element, _, _), origin) in enumerate(zip(
            owner._beam_parts, driver._beam_origins)):
        node_indices = np.array([owner._beam_index[node]
                                 for node in element.node_ids])
        slots = np.array(
            [shell_count+6*node+axis for node in node_indices for axis in range(6)]
            +list(range(shell_count+beam_nodal+24*member,
                        shell_count+beam_nodal+24*(member+1))))
        response = element.operator.evaluate(
            beam_positions[node_indices], beam_low[node_indices],
            state.frames[node_indices], state.cells[member],
            state.resultants[member], origin=origin,
            check=element.operator.guard)
        local = spatial_jacobian(response.residual,
                                 response.hessian+response.hessian_low)
        add(slots, response.residual, local)

    for index, tie in enumerate(definition.points):
        shell_node = owner._shell_index[tie.shell_node]
        beam_node = owner._beam_index[tie.beam_node]
        shell_slots = np.arange(6*shell_node, 6*shell_node+6)
        beam_slots = np.arange(shell_count+6*beam_node,
                               shell_count+6*beam_node+6)
        shell_position, shell_low = _position_pair(
            owner._shell_xyz[shell_node], state.q[shell_slots[:3]])
        reference = np.array((owner._shell_xyz[shell_node],
                              owner._beam_nodes[tie.beam_node][0]))
        reference_frames = np.array((np.eye(3),
                                     owner._beam_nodes[tie.beam_node][1]))
        joint = RigidPoseJoint(reference, reference_frames)
        ports = joint.evaluate_ports(
            np.array((shell_position, beam_positions[beam_node])),
            np.array((np.eye(3), state.frames[beam_node])),
            state.multipliers[6*index:6*index+6],
            rotation_coordinates=np.array((state.q[shell_slots[3:]],
                                           np.zeros(3))),
            charted_ports=(True, False),
            position_low=np.array((shell_low, beam_low[beam_node])))
        slots = np.r_[shell_slots, beam_slots,
                      np.arange(physical+6*index, physical+6*index+6)]
        add(slots, ports.pose.residual,
            ports.pose.spatial_jacobian@ports.coordinate_map)

    residual[:physical] -= factor*load
    return residual, tangent


@pytest.mark.parametrize("q4,s3", [
    (((1, 2, 5, 4), (2, 3, 6, 5)), ()),
    (((1, 2, 5, 4),), ((2, 3, 6), (2, 6, 5))),
])
def test_dense_finite_local_assembly_matches_sparse_trial(q4, s3):
    base = _panel()
    definition = replace(base, shell=replace(
        base.shell, q4_elements=q4, s3_elements=s3))
    owner = ExperimentalGEAnalysis(definition)
    actions = (NodalAction("beam", 105, (0., 0., 1000., 0., 0., 0.)),)
    accepted = owner.solve(actions, (.5, 1.))
    assert accepted.status == "completed"
    driver = owner._finite_driver
    load = driver._load(actions)
    state = driver._state
    factor = 1.05
    sparse_residual, sparse_tangent, *_ = driver._assemble(
        state, factor, driver._shell_origins, driver._beam_origins,
        pattern=load)
    dense_residual, dense_tangent = _dense_finite(
        owner, definition, driver, state, factor, load)
    residual_scale = max(np.linalg.norm(dense_residual), np.linalg.norm(load), 1.)
    assert np.linalg.norm(dense_residual-sparse_residual)/residual_scale <= 1e-10
    tangent_scale = np.linalg.norm(dense_tangent)
    assert np.linalg.norm(dense_tangent-sparse_tangent.toarray())/tangent_scale <= 1e-10

    physical = owner._K.shape[0]
    free = np.r_[owner._free, np.arange(physical, len(dense_residual))]
    dense_system = dense_tangent[np.ix_(free, free)]
    sparse_system = sparse_tangent.toarray()[np.ix_(free, free)]
    dense_step = np.linalg.solve(dense_system, -dense_residual[free])
    sparse_step = np.linalg.solve(sparse_system, -sparse_residual[free])
    for system, right, step in ((dense_system, dense_residual[free], dense_step),
                                (sparse_system, sparse_residual[free], sparse_step)):
        backward = np.linalg.norm(system@step+right) / max(
            np.linalg.norm(system)*np.linalg.norm(step)+np.linalg.norm(right), 1.)
        assert backward <= 1e-10
    # This indefinite, mixed-unit KKT system amplifies last-bit scatter-order
    # differences in the correction. The matrix and equation checks above
    # carry the registered tolerance; this is a development diagnostic.
    assert np.linalg.norm(dense_step-sparse_step)/np.linalg.norm(dense_step) <= 1e-8
