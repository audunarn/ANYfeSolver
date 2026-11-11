"""Bounded mechanics checks for the private BL5 director-bond candidate."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._stiffened_shell_bl5_local_experimental import BL5LocalOperator
from anyfesolver._stiffened_shell_l1_local_experimental import (
    L1LocalOperator, NativeHalfCellPort,
)
from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_l1_native_connection_probe import L1LogConnection


def _fixture(shell_y=0.):
    beam = build(4, z_offset=.0675, assemble=False)
    element, cell, weight, _ = beam._locate(.95)
    reference = np.array(((.9, shell_y, 0.), (1., shell_y, 0.)))
    state = beam.context.initial.mechanical
    native_reference = beam.context.elements[element][1].operator.reference
    port = NativeHalfCellPort.from_reference(
        native_reference, beam.context.nodes[element], element, cell, weight,
        beam.context.nodal_count, beam.context.count)
    operator = BL5LocalOperator.at_reference(
        reference, np.zeros((2, 6)), .035, port, state)
    return beam, reference, state, operator


def _rigid_beam_state(state, omega, pivot, translation=(0., 0., 0.)):
    """Transform a local test pose without invoking the clamped model driver."""
    q = Rotation.from_rotvec(omega).as_matrix()
    return replace(
        state,
        positions=(state.positions-pivot)@q.T+pivot+translation,
        position_low=np.zeros_like(state.position_low),
        nodal_frames=np.einsum('ij,njk->nik', q, state.nodal_frames),
        cell_rotations=np.einsum('ij,abjk->abik', q, state.cell_rotations),
    )


def test_bl5_virgin_rows_equal_aligned_l1_without_yaw():
    """Check the full local work map, not just a selected saved mode.

    This identity applies to the aligned flat reference fixture. It does not
    establish finite-law equivalence or qualify a multiplier space.
    """
    beam, reference, state, bl5 = _fixture()
    element, cell, weight, _ = beam._locate(.95)
    spatial_port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    l1 = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, spatial_port, state,
        beam_offset=(0., 0., -.05), station=.5)
    bl5_rows = bl5.evaluate(
        np.zeros((2, 6)), state, np.zeros(5)).gap_jacobian
    owner_l1 = L1LocalOperator.at_reference(
        reference, np.zeros((2, 6)), .035, bl5.port, state,
        offset=(0., 0., -.05), station=.5)
    owner_l1_rows = owner_l1.evaluate(
        np.zeros((2, 6)), state, np.zeros(6)).gap_jacobian
    l1_rows = np.empty((6, 21))
    slots = bl5.port.active_beam_slots()
    for column in range(21):
        direction = np.zeros(21)
        direction[column] = 1.
        beam_direction = np.zeros(beam.context.count)
        beam_direction[slots] = direction[12:]
        _, l1_rows[:, column] = l1.evaluate(
            np.zeros((2, 6)), state,
            shell_direction=direction[:12].reshape(2, 6),
            beam_direction=beam_direction)

    # BL5 uses a beam-frame translation and two director-basis components.
    # The L1 frame uses the opposite translation and pitch/roll ordering here.
    expected = np.vstack((-l1_rows[:3], -l1_rows[4], l1_rows[3]))
    np.testing.assert_allclose(bl5_rows, expected, rtol=0., atol=2e-12)
    np.testing.assert_allclose(owner_l1_rows, l1_rows, rtol=0., atol=2e-12)
    assert np.linalg.matrix_rank(bl5_rows) == 5
    assert np.linalg.matrix_rank(expected) == 5

    # A supported shell-edge in-plane gradient changes L1 yaw alone. It must
    # remain released by BL5 and by the five-row comparator.
    gradient = np.zeros(21)
    gradient[1], gradient[7] = -1., 1.
    np.testing.assert_allclose(bl5_rows@gradient, 0., atol=2e-12)
    np.testing.assert_allclose(expected@gradient, 0., atol=2e-12)
    assert abs(l1_rows[5]@gradient) > 1.


def test_bl5_reference_work_and_owned_derivatives():
    beam, reference, state, operator = _fixture()
    dual = np.array((120., -85., 150., 5., -7.))
    local = operator.evaluate(np.zeros((2, 6)), state, dual)
    np.testing.assert_allclose(local.gap, 0., atol=2e-15)
    assert local.gap.shape == (5,)
    assert local.gap_jacobian.shape == (5, 21)
    assert local.hessian.shape == (21, 21)
    np.testing.assert_allclose(local.gradient,
                               local.gap_jacobian.T@dual, atol=2e-12)
    np.testing.assert_allclose(local.hessian, local.hessian.T, atol=2e-12)
    assert abs(local.value) <= 2e-12
    reference[:] = 100.
    np.testing.assert_allclose(operator.evaluate(
        np.zeros((2, 6)), state, dual).gap, 0., atol=2e-15)
    with pytest.raises(ValueError):
        local.gradient[:] = 1.
    np.testing.assert_allclose(operator.evaluate(
        np.zeros((2, 6)), state, dual).gradient,
        local.gap_jacobian.T@dual, atol=2e-12)


def test_bl5_common_rigid_motion_and_released_eccentric_spin():
    beam, reference, state, operator = _fixture(shell_y=.03)
    omega = np.array((.16, -.11, .19))
    shift = np.array((.13, -.07, .09))
    q = Rotation.from_rotvec(omega).as_matrix()
    advanced = _rigid_beam_state(state, omega, np.zeros(3), shift)
    total = np.zeros((2, 6))
    for i, position in enumerate(reference):
        total[i, :3] = q@position+shift-position
        total[i, 3:] = omega
    np.testing.assert_allclose(operator.evaluate(
        total, advanced, np.ones(5)).gap, 0., atol=2e-11)

    # The shell remains fixed while the whole beam spins about the actual
    # attachment point. Its axis must move around the eccentric lateral arm.
    pivot = np.array((.95, .03, .0175))
    spin = np.array((0., 0., .21))
    released = _rigid_beam_state(state, spin, pivot)
    np.testing.assert_allclose(operator.evaluate(
        np.zeros((2, 6)), released, np.ones(5)).gap, 0., atol=2e-11)
    assert np.linalg.norm(released.positions-state.positions) > 1e-3
    with pytest.raises(ValueError, match="antipodal"):
        flipped = np.zeros((2, 6))
        flipped[:, 3] = np.pi
        operator.evaluate(flipped, state, np.zeros(5))


def test_bl5_rigid_pose_rows_match_five_channel_offset_chain():
    """Check the proposed external comparator's declared linear work space."""
    beam, reference, state, operator = _fixture()
    local = operator.evaluate(np.zeros((2, 6)), state, np.zeros(5))
    shell_mid = reference.mean(axis=0)
    beam_station = np.array((.95, 0., .0675))
    shell_arm = np.array((0., 0., .035/2))
    beam_arm = np.array((0., 0., -.05))
    np.testing.assert_allclose(shell_mid+shell_arm,
                               beam_station+beam_arm, atol=1e-14)

    def fields(direction):
        shell_u, shell_r, beam_u, beam_r = direction.reshape(4, 3)
        shell = np.zeros((2, 6))
        for row, position in enumerate(reference):
            shell[row, :3] = shell_u+np.cross(shell_r,
                                              position-shell_mid)
            shell[row, 3:] = shell_r
        native = np.zeros(beam.context.count)
        for node, position in enumerate(state.positions):
            native[6*node:6*node+3] = beam_u+np.cross(
                beam_r, position-beam_station)
            native[6*node+3:6*node+6] = beam_r
        for element in range(len(state.cell_rotations)):
            for cell in range(2):
                start = beam.context.nodal_count+24*element+3*cell
                native[start:start+3] = beam_r
        return shell.ravel(), native[local.beam_slots]

    native_rows = []
    chain_rows = []
    for direction in np.eye(12):
        shell, native = fields(direction)
        native_rows.append(local.gap_jacobian@np.r_[shell, native])
        shell_u, shell_r, beam_u, beam_r = direction.reshape(4, 3)
        shell_top = shell_u+np.cross(shell_r, shell_arm)
        beam_bottom = beam_u+np.cross(beam_r, beam_arm)
        chain_rows.append(np.r_[shell_top-beam_bottom,
                                shell_r[:2]-beam_r[:2]])
    native_rows = np.column_stack(native_rows)
    chain_rows = np.column_stack(chain_rows)
    np.testing.assert_allclose(native_rows[:3], chain_rows[:3],
                               rtol=0., atol=2e-12)
    assert np.linalg.matrix_rank(native_rows) == 5
    assert np.linalg.matrix_rank(chain_rows) == 5
    def row_projector(rows):
        return rows.T@np.linalg.solve(rows@rows.T, rows)
    np.testing.assert_allclose(row_projector(native_rows),
                               row_projector(chain_rows),
                               rtol=0., atol=2e-12)
    np.testing.assert_allclose(native_rows[:, (5, 11)]@np.array((1., 1.)),
                               0., atol=2e-12)


def test_bl5_preserves_independent_shell_director_shear():
    beam, reference, state, original = _fixture()
    sheared = np.zeros((2, 6))
    sheared[:, 3] = .12
    operator = BL5LocalOperator.at_reference(
        reference, sheared, .035, original.port, state)
    np.testing.assert_allclose(operator.evaluate(
        sheared, state, np.ones(5)).gap, 0., atol=2e-14)
    assert abs(operator.reference_director[1]) > .1


def test_bl5_rejects_degenerate_material_tangent():
    _beam, reference, state, original = _fixture()
    tangent_director = np.zeros((2, 6))
    tangent_director[:, 4] = np.pi/2
    with pytest.raises(ValueError, match="projected material-line tangent"):
        BL5LocalOperator.at_reference(
            reference, tangent_director, .035, original.port, state)


def test_bl5_physical_force_and_moment_work_balance():
    _beam, reference, state, operator = _fixture(shell_y=.03)
    local = operator.evaluate(
        np.zeros((2, 6)), state, np.array((115., -87., 49., 6., -4.)))
    shell_forces = np.stack((local.gradient[:3], local.gradient[6:9]))
    beam_forces = np.stack((local.gradient[12:15], local.gradient[15:18]))
    np.testing.assert_allclose(
        shell_forces.sum(axis=0)+beam_forces.sum(axis=0),
        0., atol=2e-12)
    shell_moment = sum((np.cross(position, force) for position, force in
                        zip(reference, shell_forces)), np.zeros(3))
    beam_moment = sum((np.cross(state.positions[node], force)
                       for node, force in zip(operator.port.nodes, beam_forces)),
                      np.zeros(3))
    total = (shell_moment+beam_moment+local.gradient[3:6]+
             local.gradient[9:12]+local.gradient[18:21])
    np.testing.assert_allclose(total, 0., atol=2e-11)


def test_bl5_actual_update_directional_first_and_second_variation():
    beam, _reference, state, operator = _fixture(shell_y=.03)
    rng = np.random.default_rng(1307)
    direction = rng.normal(size=21)
    direction[3:6] *= .3
    direction[9:12] *= .3
    direction[18:21] *= .3
    dual = np.array((90., -35., 70., 3., -5.))
    base = operator.evaluate(np.zeros((2, 6)), state, dual)

    def trial(signed_step):
        beam_step = np.zeros(beam.context.count)
        beam_step[base.beam_slots.astype(int)] = signed_step*direction[12:]
        moved = beam.context.advance(state, beam_step)
        shell = signed_step*direction[:12].reshape(2, 6)
        return operator.evaluate(shell, moved, dual)

    for step in (1e-4, 1e-5):
        plus, minus = trial(step), trial(-step)
        numerical_gap = (plus.gap-minus.gap)/(2*step)
        np.testing.assert_allclose(
            numerical_gap, base.gap_jacobian@direction,
            rtol=2e-6, atol=1e-8)
        numerical_curvature = (plus.value-2*base.value+minus.value)/step**2
        predicted_curvature = direction@base.hessian@direction
        np.testing.assert_allclose(
            numerical_curvature, predicted_curvature,
            rtol=3e-5, atol=3e-6)
