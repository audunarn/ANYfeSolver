"""Bounded L1 kinematics only; no formal F03 execution or acceptance."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver.corotational import rotation_matrix_from_vector
from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_l1_native_connection_probe import (
    L1LogConnection, SegmentedL1WeakWork, beam_frame_gap_from_shell,
    constrained_beam_to_shell_multiplier, shell_trace, six_gap, spatial_gap,
)


def _case():
    beam = build(4, z_offset=.0675, assemble=False)
    element, cell, weight, _ = beam._locate(.95)
    port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    reference = np.array(((.9, 0., 0.), (1., 0., 0.)))
    return beam, port, reference


def test_reference_six_rows_and_native_half_cell_identity():
    beam, port, reference = _case()
    state = beam.context.initial.mechanical
    assert port.cell == 1
    assert 0. < port.weight < 1.
    gap, _ = six_gap(reference, np.zeros((2, 6)), .035, port, state)
    np.testing.assert_allclose(gap, 0., atol=2e-16)

    # The line carries an independent transverse shear channel, while the
    # shell-normal angular row is retained as the sixth physical point row.
    total = np.zeros((2, 6))
    total[:, 1] = 1e-4
    shear, _ = six_gap(reference, total, .035, port, state)
    np.testing.assert_allclose(shear[1], 1e-4, atol=1e-15)
    total = np.zeros((2, 6))
    total[1, 1] = 1e-4
    yaw, _ = six_gap(reference, total, .035, port, state)
    assert abs(yaw[5]) > 9e-4


def test_objective_physical_pose_work_pairing():
    beam, port, reference = _case()
    total = np.array(((.001, -.002, .0004, .03, -.02, .01),
                      (.002, .001, -.0003, -.01, .02, -.015)))
    shell, frame, _, _ = shell_trace(reference, total, .035)
    high, low, beam_frame = port.pose_pair(beam.context.initial.mechanical)
    beam_position = high+low+beam_frame@np.array((0., 0., -.05))
    original = spatial_gap(shell, frame, beam_position, beam_frame)
    rigid = rotation_matrix_from_vector(np.array((.2, -.15, .12)))
    shift = np.array((.3, -.7, .2))
    transformed = spatial_gap(rigid@shell+shift, rigid@frame,
                              rigid@beam_position+shift, rigid@beam_frame)
    np.testing.assert_allclose(transformed, original, atol=3e-16)
    dual = np.array((3., -5., 7., 11., -13., 17.))
    np.testing.assert_allclose(dual@transformed, dual@original, atol=1e-14)


def test_actual_e4_total_and_native_advance_directional_work():
    beam, port, reference = _case()
    rng = np.random.default_rng(761)
    base = rng.normal(size=beam.context.count)*2e-4
    base[:6] = 0.
    state = beam.context.advance(beam.context.initial.mechanical, base)
    total = rng.normal(size=(2, 6))*np.array((.001, .001, .001, .04, .04, .04))
    shell_direction = rng.normal(size=(2, 6))*np.array((.001, .001, .001, .03, .03, .03))
    beam_direction = rng.normal(size=beam.context.count)*.001
    beam_direction[:6] = 0.
    dual = np.array((2., -3., 5., 7., -11., 13.))
    gap, derivative = six_gap(reference, total, .035, port, state,
                              shell_direction=shell_direction,
                              beam_direction=beam_direction)
    assert np.linalg.norm(gap) > 1e-6
    step = 1e-5

    def actual_update(sign):
        trial = beam.context.advance(state, sign*step*beam_direction)
        return six_gap(reference, total+sign*step*shell_direction,
                       .035, port, trial)[0]

    finite = (actual_update(1.)-actual_update(-1.))/(2*step)
    np.testing.assert_allclose(derivative, finite, rtol=2e-7, atol=2e-10)
    np.testing.assert_allclose(dual@derivative, dual@finite,
                               rtol=2e-7, atol=2e-9)


def test_all_six_local_channels_have_independent_physical_variations():
    beam, port, reference = _case()
    state = beam.context.initial.mechanical
    zero = np.zeros((2, 6))
    beam_zero = np.zeros(beam.context.count)
    columns = []
    for axis in range(3):
        direction = np.zeros((2, 6))
        direction[:, axis] = 1.
        columns.append(six_gap(reference, zero, .035, port, state,
                               shell_direction=direction,
                               beam_direction=beam_zero)[1])
    for axis in (3, 4):
        direction = np.zeros((2, 6))
        direction[:, axis] = 1.
        columns.append(six_gap(reference, zero, .035, port, state,
                               shell_direction=direction,
                               beam_direction=beam_zero)[1])
    yaw = np.zeros((2, 6))
    yaw[0, 1], yaw[1, 1] = -.05, .05
    columns.append(six_gap(reference, zero, .035, port, state,
                           shell_direction=yaw, beam_direction=beam_zero)[1])
    assert np.linalg.matrix_rank(np.column_stack(columns), tol=1e-9) == 6


def test_degenerate_projected_material_tangent_fails_closed():
    reference = np.array(((0., 0., 0.), (0., 0., 1.)))
    with pytest.raises(ValueError, match="projected E4 material-line tangent"):
        shell_trace(reference, np.zeros((2, 6)), .035)


def test_declared_non_midpoint_shell_station_and_domain():
    reference = np.array(((.75, 0., 0.), (.875, 0., 0.)))
    total = np.zeros((2, 6))
    total[1, :3] = (.002, .003, -.001)
    direction = np.zeros((2, 6))
    direction[0, :3] = (.01, -.02, .03)
    direction[1, 3:5] = (.04, -.03)
    station = .21132486540518713
    point, frame, point_step, frame_step = shell_trace(
        reference, total, .035, direction, station=station)
    h = 1e-6
    plus = shell_trace(reference, total+h*direction, .035, station=station)
    minus = shell_trace(reference, total-h*direction, .035, station=station)
    np.testing.assert_allclose((plus[0]-minus[0])/(2*h), point_step,
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose((plus[1]-minus[1])/(2*h), frame_step,
                               rtol=1e-8, atol=1e-9)
    assert np.isfinite(point).all() and np.isfinite(frame).all()
    for bad in (-.01, 1.01, np.nan):
        with pytest.raises(ValueError, match="station"):
            shell_trace(reference, total, .035, station=bad)


def test_reference_offset_log_is_stress_free_at_nontrivial_reference():
    beam, port, reference = _case()
    base = np.zeros(beam.context.count)
    base[6:9] = np.array((.002, -.001, .0005))
    state = beam.context.advance(beam.context.initial.mechanical, base)
    total = np.array(((.001, .002, .0003, .02, -.01, .03),
                      (.002, -.001, -.0004, -.01, .025, -.015)))
    connection = L1LogConnection.at_reference(reference, total, .035, port, state)
    gap, derivative = connection.evaluate(total, state)
    np.testing.assert_allclose(gap, 0., atol=2e-15)
    assert derivative is None
    assert np.linalg.norm(connection.joint._offset) > 0.
    assert np.linalg.norm(connection.joint._relative-np.eye(3)) > 1e-4


def test_log_connection_owns_immutable_reference_geometry():
    beam, port, reference = _case()
    state = beam.context.initial.mechanical
    connection = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, port, state)
    expected = connection.evaluate(np.zeros((2, 6)), state)[0]
    reference[0, 0] += .1
    np.testing.assert_array_equal(
        connection.evaluate(np.zeros((2, 6)), state)[0], expected)
    with pytest.raises(ValueError):
        connection.reference.setflags(write=True)


def test_log_first_variation_at_nonzero_relative_rotation_actual_updates():
    beam, port, reference = _case()
    state = beam.context.initial.mechanical
    connection = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, port, state)
    rng = np.random.default_rng(832)
    base = rng.normal(size=beam.context.count)*4e-4
    base[:6] = 0.
    trial_state = beam.context.advance(state, base)
    total = np.array(((.001, -.002, .0003, .23, -.12, .05),
                      (.002, .001, -.0004, .18, .09, -.02)))
    shell_direction = rng.normal(size=(2, 6))*np.array((.001, .001, .001, .04, .04, .04))
    beam_direction = rng.normal(size=beam.context.count)*.001
    beam_direction[:6] = 0.
    gap, derivative = connection.evaluate(
        total, trial_state, shell_direction=shell_direction,
        beam_direction=beam_direction)
    assert np.linalg.norm(gap[3:]) > .1
    dual = np.array((2., -3., 5., 7., -11., 13.))
    step = 2e-5

    def actual_update(sign):
        updated = beam.context.advance(trial_state, sign*step*beam_direction)
        return connection.evaluate(total+sign*step*shell_direction, updated)[0]

    finite = (actual_update(1.)-actual_update(-1.))/(2*step)
    np.testing.assert_allclose(derivative, finite, rtol=4e-7, atol=2e-10)
    np.testing.assert_allclose(dual@derivative, dual@finite,
                               rtol=4e-7, atol=2e-9)


def test_reference_offset_log_is_objective_under_common_rigid_motion():
    beam, port, reference = _case()
    state = beam.context.initial.mechanical
    connection = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, port, state)
    total = np.array(((.001, -.002, .0003, .14, -.07, .03),
                      (.002, .001, -.0004, .11, .06, -.02)))
    shell, shell_frame, _, _ = shell_trace(reference, total, .035)
    high, low, beam_frame = port.pose_pair(state)
    beam_position = high+low+beam_frame@np.array((0., 0., -.05))
    positions = np.array((shell, beam_position))
    frames = np.array((shell_frame, beam_frame))
    original = connection.joint.evaluate(positions, frames, np.zeros(6))
    rigid = rotation_matrix_from_vector(np.array((.3, -.2, .1)))
    shift = np.array((.4, -.7, .2))
    transformed = connection.joint.evaluate(positions@rigid.T+shift,
                                            rigid@frames, np.zeros(6))
    np.testing.assert_allclose(transformed.constraints,
                               original.constraints, atol=2e-14)
    np.testing.assert_allclose(connection.evaluate(total, state)[0],
                               original.constraints, atol=2e-14)


def test_log_rejects_relative_rotation_outside_chart():
    beam, port, reference = _case()
    state = beam.context.initial.mechanical
    connection = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, port, state)
    reversed_line = np.zeros((2, 6))
    reversed_line[0, 0] = -.8
    reversed_line[1, 0] = -1.0
    with pytest.raises(ValueError, match="relative rotation must remain below"):
        connection.evaluate(reversed_line, state)


@pytest.mark.parametrize("angle", (np.pi/2, np.pi))
def test_local_angular_chart_rejects_singular_relative_rotations(angle):
    rotated = rotation_matrix_from_vector(np.array((angle, 0., 0.)))
    with pytest.raises(ValueError, match="outside the local probe chart"):
        spatial_gap(np.zeros(3), np.eye(3), np.zeros(3), rotated)


def test_composed_nonzero_multiplier_work_tangent_and_second_plateau():
    beam, port, reference = _case()
    initial = beam.context.initial.mechanical
    connection = L1LogConnection.at_reference(reference, np.zeros((2, 6)),
                                               .035, port, initial)
    rng = np.random.default_rng(517)
    beam_base = rng.normal(size=beam.context.count)*.002
    beam_base[:6] = 0.
    state = beam.context.advance(initial, beam_base)
    total = np.array(((.002, -.001, .0003, .2, -.1, .04),
                      (.003, .002, -.0004, .13, .09, -.03)))
    dual = np.array((120., -85., 150., 5., -7., 11.))
    value, gap, residual, hessian = connection.work_tangent(total, state, dual)
    np.testing.assert_allclose(gap, connection.evaluate(total, state)[0], atol=2e-14)
    np.testing.assert_allclose(value, dual@gap, atol=2e-13)
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-12)

    shell_direction = rng.normal(size=(2, 6))*np.array((.03, .03, .03, .2, .2, .2))
    beam_direction = rng.normal(size=beam.context.count)*.03
    beam_direction[:6] = 0.
    for element in range(len(beam.context.elements)):
        begin = beam.context.nodal_count+24*element
        beam_direction[begin:begin+6] *= 5.
    direction = np.r_[shell_direction.ravel(), beam_direction]
    analytic_first = float(residual@direction)
    analytic_second = float(direction@hessian@direction)
    assert abs(analytic_second) > 1e-5
    first_gap = connection.evaluate(total, state,
                                    shell_direction=shell_direction,
                                    beam_direction=beam_direction)[1]
    np.testing.assert_allclose(analytic_first, dual@first_gap, atol=3e-11)

    def work(step):
        updated = beam.context.advance(state, step*beam_direction)
        return float(dual@connection.evaluate(total+step*shell_direction, updated)[0])

    errors = []
    for step in (.02, .01, .005):
        finite = (work(step)-2*value+work(-step))/(step*step)
        errors.append(abs(finite-analytic_second))
    finite_first = (work(.005)-work(-.005))/.01
    np.testing.assert_allclose(finite_first, analytic_first,
                               rtol=3e-5, atol=2e-7)
    assert errors[1] < .4*errors[0]
    assert errors[2] < .4*errors[1]
    assert errors[2] <= 3e-6*max(1., abs(analytic_second))

    shell_second = rng.normal(size=(2, 6))*np.array((.02, .02, .02, .1, .1, .1))
    beam_second = rng.normal(size=beam.context.count)*.02
    beam_second[:6] = 0.
    mixed = float(direction@hessian@np.r_[shell_second.ravel(), beam_second])

    def work_mixed(first, second):
        updated = beam.context.advance(
            state, first*beam_direction+second*beam_second)
        trial = total+first*shell_direction+second*shell_second
        return float(dual@connection.evaluate(trial, updated)[0])

    mixed_errors = []
    for step in (.02, .01, .005):
        finite = (work_mixed(step, step)-work_mixed(step, -step)-
                  work_mixed(-step, step)+work_mixed(-step, -step))/(4*step*step)
        mixed_errors.append(abs(finite-mixed))
    assert mixed_errors[1] < .4*mixed_errors[0]
    assert mixed_errors[2] < .4*mixed_errors[1]
    assert mixed_errors[2] <= 2e-6*max(1., abs(mixed))

    # Only the two endpoint translations and one retained half-cell rotation
    # can carry B3 point work; the 18 resultants per element have zero columns.
    for element in range(len(beam.context.elements)):
        begin = 12+beam.context.nodal_count+24*element+6
        np.testing.assert_array_equal(residual[begin:begin+18], 0.)
        np.testing.assert_array_equal(hessian[begin:begin+18], 0.)


def test_log_work_objectivity_and_reference_ownership():
    beam, port, reference = _case()
    initial = beam.context.initial.mechanical
    connection = L1LogConnection.at_reference(reference, np.zeros((2, 6)),
                                               .035, port, initial)
    total = np.array(((.001, -.002, .0003, .12, -.08, .04),
                      (.002, .001, -.0004, .09, .05, -.02)))
    dual = np.array((2., -3., 5., 7., -11., 13.))
    original = connection.work_tangent(total, initial, dual)
    rigid_vector = np.array((.2, -.12, .08))
    rigid = rotation_matrix_from_vector(rigid_vector)
    shift = np.array((.3, -.2, .1))
    transformed_total = total.copy()
    for node in range(2):
        transformed_total[node, :3] = (
            rigid@(reference[node]+total[node, :3])+shift-reference[node])
        transformed_total[node, 3:] = Rotation.from_matrix(
            rigid@rotation_matrix_from_vector(total[node, 3:])).as_rotvec()
    step = np.zeros(beam.context.count)
    for node, position in enumerate(initial.positions):
        if node == 0:  # the fixed support is outside the sampled half-cell
            continue
        step[6*node:6*node+3] = rigid@position+shift-position
        step[6*node+3:6*node+6] = rigid_vector
    for element in range(len(beam.context.elements)):
        begin = beam.context.nodal_count+24*element
        step[begin:begin+6] = np.tile(rigid_vector, 2)
    transformed_state = beam.context.advance(initial, step)
    transformed = connection.work_tangent(transformed_total, transformed_state, dual)
    np.testing.assert_allclose(transformed[0], original[0], atol=4e-13)
    np.testing.assert_allclose(transformed[1], original[1], atol=4e-14)
    assert not connection.reference.flags.writeable
    reference[0, 0] = -4.
    np.testing.assert_allclose(connection.work_tangent(total, initial, dual)[0],
                               original[0], atol=1e-14)
    with pytest.raises(TypeError, match="retained native mechanical state"):
        connection.work_tangent(total, object(), dual)


def _eccentric_constrained_pose(beam, reference, rigid_vector):
    initial = beam.context.initial.mechanical
    rigid = rotation_matrix_from_vector(rigid_vector)
    shift = np.array((.3, -.2, .1))
    total = np.zeros((2, 6))
    for node in range(2):
        total[node, :3] = rigid@reference[node]+shift-reference[node]
        total[node, 3:] = rigid_vector
    step = np.zeros(beam.context.count)
    for node, position in enumerate(initial.positions):
        if node == 0:  # supported node lies outside the sampled half-cell
            continue
        step[6*node:6*node+3] = rigid@position+shift-position
        step[6*node+3:6*node+6] = rigid_vector
    for element in range(len(beam.context.elements)):
        begin = beam.context.nodal_count+24*element
        step[begin:begin+6] = np.tile(rigid_vector, 2)
    return total, beam.context.advance(initial, step)


def test_shell_and_beam_frame_gap_relation_at_finite_eccentric_pose():
    beam, port, reference = _case()
    initial = beam.context.initial.mechanical
    connection = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, port, initial,
        beam_offset=(.018, .012, -.04))
    base = np.zeros(beam.context.count)
    cell = beam.context.nodal_count+24*port.element_index+3*port.cell
    base[cell:cell+3] = (.11, -.08, .04)
    base[6*beam.context.nodes[port.element_index][port.cell]:][:3] = (.001, -.002, .0004)
    state = beam.context.advance(initial, base)
    total = np.array(((.003, -.002, .001, .18, -.11, .05),
                      (.001, .004, -.0008, .12, .09, -.06)))
    shell_gap = connection.evaluate(total, state)[0]
    beam_gap = connection.work_tangent(
        total, state, np.zeros(6), translation_frame="beam")[1]
    predicted = beam_frame_gap_from_shell(
        shell_gap[:3], shell_gap[3:], connection.joint._relative,
        connection.joint._offset)
    np.testing.assert_allclose(beam_gap[:3], predicted, atol=3e-14)
    np.testing.assert_allclose(beam_gap[3:], shell_gap[3:], atol=3e-14)
    assert np.linalg.norm(connection.joint._offset) > .02
    assert np.linalg.norm(shell_gap[3:]) > .05


def test_constrained_dual_work_origin_wrench_shear_and_curvature_difference():
    beam, port, reference = _case()
    initial = beam.context.initial.mechanical
    connection = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, port, initial,
        beam_offset=(.018, .012, -.04))
    total, state = _eccentric_constrained_pose(
        beam, reference, np.array((.22, -.14, .09)))
    beam_dual = np.array((120., -80., 50., 5., -3., 2.))
    shell_dual = constrained_beam_to_shell_multiplier(
        beam_dual, connection.joint._relative, connection.joint._offset)
    assert np.linalg.norm(shell_dual[3:]-beam_dual[3:]) > 1.
    shell_work = connection.work_tangent(total, state, shell_dual)
    beam_work = connection.work_tangent(
        total, state, beam_dual, translation_frame="beam")
    np.testing.assert_allclose(shell_work[1], 0., atol=4e-14)
    np.testing.assert_allclose(beam_work[1], 0., atol=4e-14)
    np.testing.assert_allclose(shell_work[2], beam_work[2], atol=5e-11)
    assert np.linalg.norm(shell_work[3]-beam_work[3]) > 1e-3

    # The pose-level residual gives equal and opposite forces and a balanced
    # total moment about the same spatial origin, including the eccentric arm.
    shell, shell_frame, _, _ = shell_trace(reference, total, .035)
    high, low, beam_frame = port.pose_pair(state)
    beam_point = high+low+beam_frame@np.asarray(connection.beam_offset)
    pose = connection.joint.evaluate(np.array((shell, beam_point)),
                                     np.array((shell_frame, beam_frame)), shell_dual)
    f_shell, m_shell, f_beam, m_beam = pose.residual[:12].reshape(4, 3)
    np.testing.assert_allclose(f_shell+f_beam, 0., atol=2e-12)
    np.testing.assert_allclose(m_shell+m_beam+
                               np.cross(shell, f_shell)+np.cross(beam_point, f_beam),
                               0., atol=2e-11)

    # Independently differentiate the saved beam-frame convention at the
    # physical poses; its origin wrench equals the transformed shell work.
    e_beam = -connection.joint._relative.T@connection.joint._offset
    def beam_pose_work(positions, frames):
        translation = frames[1].T@(positions[0]-positions[1])-e_beam
        rotation = Rotation.from_matrix(
            connection.joint._relative.T@frames[0].T@frames[1]).as_rotvec()
        return float(beam_dual@np.r_[translation, rotation])
    positions = np.array((shell, beam_point))
    frames = np.array((shell_frame, beam_frame))
    beam_pose_residual = np.zeros(12)
    step_size = 1e-6
    for column in range(12):
        values = []
        for sign in (1., -1.):
            trial_positions = positions.copy()
            trial_frames = frames.copy()
            port_index, component = divmod(column, 6)
            if component < 3:
                trial_positions[port_index, component] += sign*step_size
            else:
                spin = np.zeros(3)
                spin[component-3] = sign*step_size
                trial_frames[port_index] = rotation_matrix_from_vector(spin)@frames[port_index]
            values.append(beam_pose_work(trial_positions, trial_frames))
        beam_pose_residual[column] = (values[0]-values[1])/(2*step_size)
    np.testing.assert_allclose(beam_pose_residual, pose.residual[:12],
                               rtol=3e-9, atol=4e-8)
    bf_s, bm_s, bf_b, bm_b = beam_pose_residual.reshape(4, 3)
    np.testing.assert_allclose(bf_s+bf_b, 0., atol=4e-8)
    np.testing.assert_allclose(bm_s+bm_b+
                               np.cross(shell, bf_s)+np.cross(beam_point, bf_b),
                               0., atol=7e-8)

    shear = np.zeros((2, 6))
    shear[:, 1] = 1.
    beam_zero = np.zeros(beam.context.count)
    actual_shear = connection.evaluate(
        total, state, shell_direction=shear, beam_direction=beam_zero)[1]
    assert abs(actual_shear[1]) > .8
    shear_direction = np.r_[shear.ravel(), beam_zero]
    np.testing.assert_allclose(shell_work[2]@shear_direction,
                               beam_work[2]@shear_direction, atol=2e-11)

    # Fixed dual coordinates give different off-constraint potentials and
    # curvature, although their first work agrees on the constraint set.
    direction = np.zeros(12+beam.context.count)
    direction[1], direction[7], direction[10] = -.03, .04, .15
    direction[12+cell_index(beam, port)+1] = .03
    curvature = float(direction@(beam_work[3]-shell_work[3])@direction)
    assert abs(curvature) > 1e-5
    h = .01
    def difference(scale):
        trial = total+scale*direction[:12].reshape(2, 6)
        updated = beam.context.advance(state, scale*direction[12:])
        return (connection.work_tangent(trial, updated, beam_dual,
                                         translation_frame="beam")[0]-
                connection.work_tangent(trial, updated, shell_dual)[0])
    finite_curvature = (difference(h)-2*difference(0.)+difference(-h))/(h*h)
    np.testing.assert_allclose(finite_curvature, curvature,
                               rtol=3e-4, atol=2e-6)


def cell_index(beam, port):
    return beam.context.nodal_count+24*port.element_index+3*port.cell


def test_drill_only_shell_chart_has_no_point_stiffness_in_either_frame():
    beam, port, reference = _case()
    state = beam.context.initial.mechanical
    connection = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, port, state,
        beam_offset=(.018, .012, -.04))
    total = np.zeros((2, 6))
    total[:, 5] = .2
    dual = np.array((11., -7., 5., 3., -2., 13.))
    direction = np.zeros(12+beam.context.count)
    direction[5] = direction[11] = 1.
    for frame in ("shell", "beam"):
        work, gap, residual, hessian = connection.work_tangent(
            total, state, dual, translation_frame=frame)
        np.testing.assert_allclose(gap, 0., atol=2e-15)
        np.testing.assert_allclose(work, 0., atol=2e-13)
        np.testing.assert_allclose(residual@direction, 0., atol=2e-12)
        np.testing.assert_allclose(direction@hessian@direction, 0., atol=2e-11)


def _segmented_case():
    beam = build(8, z_offset=.0675, assemble=False)
    cuts = np.array((.75, .875, 1.))  # two actual native half-cell cuts
    reference = np.column_stack((cuts, np.zeros(3), np.zeros(3)))
    initial = beam.context.initial.mechanical

    def port_at(station):
        element, cell, weight, _ = beam._locate(station)
        return NativeHalfCellSpatialPort(beam.context, element, cell, weight)

    line = SegmentedL1WeakWork.at_reference(
        reference, np.zeros((3, 6)), .035, cuts, port_at, initial)
    return beam, line, reference, cuts


def test_two_segment_reference_quadrature_and_half_cell_ownership():
    beam, line, reference, cuts = _segmented_case()
    assert len(line.connections) == 4
    np.testing.assert_allclose(sum(line.weights), cuts[-1]-cuts[0], atol=1e-15)
    for index, connection in enumerate(line.connections):
        segment = index//2
        port = connection.port
        assert (port.element_index, port.cell) == tuple(
            beam._locate(.5*(cuts[segment]+cuts[segment+1]))[:2])
        assert 0. < port.weight < 1.
        shell, frame, _, _ = shell_trace(
            reference[segment:segment+2], np.zeros((2, 6)), .035,
            station=connection.station)
        high, low, beam_frame = port.pose_pair(beam.context.initial.mechanical)
        np.testing.assert_allclose(shell, high+low+beam_frame@np.array((0., 0., -.05)),
                                   atol=2e-15)
        np.testing.assert_allclose(frame, beam_frame, atol=2e-15)
    zero = np.zeros((3, 6))
    value, gaps, residual, hessian = line.work_tangent(
        zero, beam.context.initial.mechanical, np.ones((3, 6)))
    np.testing.assert_allclose(gaps, 0., atol=2e-15)
    np.testing.assert_allclose(value, 0., atol=2e-15)
    assert residual.shape == (18+beam.context.count,)
    assert hessian.shape == (residual.size, residual.size)

    # This Gauss rule integrates the piecewise-affine dual field exactly.
    dual = np.arange(18., dtype=float).reshape(3, 6)
    sampled = sum(line.weights[i]*((1.-c.station)*dual[i//2]+c.station*dual[i//2+1])
                  for i, c in enumerate(line.connections))
    exact = sum(.5*(cuts[i+1]-cuts[i])*(dual[i]+dual[i+1]) for i in range(2))
    np.testing.assert_allclose(sampled, exact, atol=5e-16)


def test_segmented_reference_measure_and_port_correspondence_are_explicit():
    beam, line, reference, cuts = _segmented_case()
    initial = beam.context.initial.mechanical
    total = np.zeros((3, 6))

    def correct_port(station):
        element, cell, weight, _ = beam._locate(station)
        return NativeHalfCellSpatialPort(beam.context, element, cell, weight)

    changed = reference.copy()
    changed[1, 1] = .04
    with pytest.raises(ValueError, match="reference length"):
        SegmentedL1WeakWork.at_reference(
            changed, total, .035, cuts, correct_port, initial)

    def wrong_port(_station):
        return correct_port(.5)

    with pytest.raises(ValueError, match="correspondence mismatch"):
        SegmentedL1WeakWork.at_reference(
            reference, total, .035, cuts, wrong_port, initial)

    offset_line = SegmentedL1WeakWork.at_reference(
        reference, total, .035, cuts, correct_port, initial,
        beam_offset=(0., 0., -.04),
        expected_reference_offset_shell=(0., 0., .01))
    np.testing.assert_allclose(offset_line.work_tangent(
        total, initial, np.ones((3, 6)))[1], 0., atol=2e-15)

    with pytest.raises(ValueError, match="four ordered"):
        SegmentedL1WeakWork(line.connections[:2], line.weights[:2],
                            line.stations[:2])


def test_segmented_work_derivative_objectivity_wrench_and_retained_action():
    beam, line, reference, _ = _segmented_case()
    initial = beam.context.initial.mechanical
    step = np.zeros(beam.context.count)
    for connection in line.connections:
        port = connection.port
        cell = beam.context.nodal_count+24*port.element_index+3*port.cell
        step[cell:cell+3] = (.014, -.009, .011)
    state = beam.context.advance(initial, step)
    total = np.array(((.001, -.002, .0005, .012, -.008, .003),
                      (.002, .001, -.0003, -.009, .013, -.004),
                      (-.001, .0007, .0002, .007, -.006, .002)))
    dual = np.array(((3., -5., 7., 11., -13., 17.),
                     (-2., 4., 6., -8., 10., -12.),
                     (5., 3., -4., 9., -7., 2.)))
    value, gaps, residual, hessian = line.work_tangent(total, state, dual)
    expected = sum(line.weights[i]*gaps[i]@(
        (1.-c.station)*dual[i//2]+c.station*dual[i//2+1])
        for i, c in enumerate(line.connections))
    np.testing.assert_allclose(value, expected, atol=5e-15)

    direction = np.zeros(residual.size)
    direction[:18] = np.linspace(-.03, .04, 18)
    for connection in line.connections:
        port = connection.port
        cell = beam.context.nodal_count+24*port.element_index+3*port.cell
        direction[18+cell:18+cell+3] = (.06, -.04, .03)
    h = 2e-5

    def updated(scale):
        shell = total+scale*direction[:18].reshape(3, 6)
        native = beam.context.advance(state, scale*direction[18:])
        return line.work_tangent(shell, native, dual)[0]

    first = (updated(h)-updated(-h))/(2*h)
    second = (updated(h)-2*value+updated(-h))/(h*h)
    np.testing.assert_allclose(first, residual@direction, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(second, direction@hessian@direction,
                               rtol=2e-4, atol=3e-7)

    rigid = rotation_matrix_from_vector((.17, -.11, .09))
    shift = np.array((.3, -.2, .1))
    for index, connection in enumerate(line.connections):
        segment = index//2
        shell, frame, _, _ = shell_trace(
            reference[segment:segment+2], total[segment:segment+2], .035,
            station=connection.station)
        high, low, beam_frame = connection.port.pose_pair(state)
        beam_point = high+low+beam_frame@np.asarray(connection.beam_offset)
        dual_at = ((1.-connection.station)*dual[segment]+connection.station*
                   dual[segment+1])
        pose = connection.joint.evaluate(
            np.array((shell, beam_point)), np.array((frame, beam_frame)), dual_at)
        moved = connection.joint.evaluate(
            np.array((rigid@shell+shift, rigid@beam_point+shift)),
            np.array((rigid@frame, rigid@beam_frame)), dual_at)
        np.testing.assert_allclose(moved.constraints, pose.constraints, atol=2e-15)
        np.testing.assert_allclose(dual_at@moved.constraints,
                                   dual_at@pose.constraints, atol=2e-14)
        f_s, m_s, f_b, m_b = pose.residual[:12].reshape(4, 3)
        np.testing.assert_allclose(f_s+f_b, 0., atol=2e-12)
        np.testing.assert_allclose(m_s+m_b+np.cross(shell, f_s)+
                                   np.cross(beam_point, f_b), 0., atol=2e-11)

    shear = np.zeros((3, 6))
    shear[:, 1] = .1
    sheared = line.work_tangent(total+shear, state, dual)[1]
    assert np.linalg.norm(sheared[:, 1]-gaps[:, 1]) > .15
    retained = np.zeros(beam.context.count)
    first_port = line.connections[0].port
    cell = beam.context.nodal_count+24*first_port.element_index+3*first_port.cell
    retained[cell+2] = .1
    spun = line.work_tangent(total, beam.context.advance(state, retained), dual)[1]
    assert np.linalg.norm(spun[:2, 5]-gaps[:2, 5]) > .1


def test_segmented_common_shell_drill_has_no_stiffness():
    beam, line, _, _ = _segmented_case()
    total = np.zeros((3, 6))
    total[:, 5] = .2
    dual = np.ones((3, 6))
    value, gaps, residual, hessian = line.work_tangent(
        total, beam.context.initial.mechanical, dual)
    drill = np.zeros(residual.size)
    drill[5] = drill[11] = drill[17] = 1.
    np.testing.assert_allclose(gaps, 0., atol=2e-15)
    np.testing.assert_allclose(value, 0., atol=2e-15)
    np.testing.assert_allclose(residual@drill, 0., atol=2e-12)
    np.testing.assert_allclose(drill@hessian@drill, 0., atol=2e-11)
