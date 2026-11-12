"""Bounded mechanics checks for the unqualified physical-width shell rail."""

import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.transform import Rotation

from anyfesolver.corotational import rotation_matrix_from_vector
from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver.elements import create_shell_element
from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_plate_patch_work import raw_e4_tangent
from scripts.b3_e4_f03_l1_physical_rail import (
    FiniteRailDensityBasis, FlatQ4PhysicalRail, RailL1LogConnection,
    SegmentedRailWeakWork, reference_spatial_wrench_map,
    transverse_shape_integrals,
)
from scripts.b3_e4_f03_paired_yaw_upper_screen import flat_yaw_p0_action


def _rail(thickness=.035):
    y = np.array((-.1, 0., .1))
    reference = np.array([[(x, value, 0.) for value in y]
                          for x in (.9, 1.)])
    return FlatQ4PhysicalRail.from_reference(reference, y, (-.05, .05), thickness)


def test_transverse_integral_is_width_owned_and_not_centerline_sampling():
    rail = _rail()
    np.testing.assert_allclose(rail.weights, (.125, .75, .125), atol=1e-15)
    assert not rail.weights.flags.writeable
    y = np.array((-.1, 0., .1))
    weights = transverse_shape_integrals(y, (-.025, .05))
    gauss, gauss_weights = np.polynomial.legendre.leggauss(3)
    independent = np.zeros(3)
    for lo, hi in ((-.025, 0.), (0., .05)):
        for point, weight in zip(gauss, gauss_weights):
            at = (lo+hi)/2.+point*(hi-lo)/2.
            if at < 0.:
                shape = ((-at)/.1, 1.+at/.1, 0.)
            else:
                shape = (0., 1.-at/.1, at/.1)
            independent += weight*(hi-lo)/2.*np.asarray(shape)/.075
    np.testing.assert_allclose(weights, independent, rtol=0., atol=1e-14)
    original = rail.reference.copy()
    with pytest.raises(ValueError, match="strip"):
        transverse_shape_integrals(y, (-.11, .05))
    np.testing.assert_array_equal(rail.reference, original)


def test_virgin_rail_variation_matches_owned_e4_q4_shape_field():
    """Check first-order compatibility, not a finite interior director claim."""
    rail = _rail()
    element = create_shell_element(
        1, [1, 2, 3, 4], "shell", formulation="e4-pl",
        thickness=rail.thickness, reference_normal=(0., 0., 1.))
    rng = np.random.default_rng(619)
    direction = rng.normal(scale=.2, size=(2, 3, 6))
    station = .29
    zero = np.zeros_like(direction)
    _, _, center_step, frame_step = rail.pose(
        zero, station=station, direction=direction)
    integrated_top = np.zeros(3)
    integrated_director = np.zeros(3)
    integrated_top_gradient = np.zeros(3)
    abscissae, gauss_weights = np.polynomial.legendre.leggauss(3)
    ez = np.array((0., 0., 1.))
    width = rail.bounds[1]-rail.bounds[0]
    for cell, (a, b) in enumerate(zip(rail.y_cuts[:-1], rail.y_cuts[1:])):
        lo, hi = max(a, rail.bounds[0]), min(b, rail.bounds[1])
        if hi <= lo:
            continue
        nodal = direction[[0, 1, 1, 0],
                          [cell, cell, cell+1, cell+1]]
        for point, weight in zip(abscissae, gauss_weights):
            y = .5*(lo+hi)+.5*(hi-lo)*point
            xi, eta = 2*station-1., 2*(y-a)/(b-a)-1.
            shape, dshape_xi, _ = element.compute_shape_functions(xi, eta)
            translation = shape@nodal[:, :3]
            rotation = shape@nodal[:, 3:]
            director = np.cross(rotation, ez)
            top_nodal = nodal[:, :3]+.5*rail.thickness*np.cross(
                nodal[:, 3:], ez)
            factor = .5*(hi-lo)*weight/width
            integrated_top += factor*(translation+.5*rail.thickness*director)
            integrated_director += factor*director
            integrated_top_gradient += factor*2/(rail.reference[1, 0, 0]-
                                                 rail.reference[0, 0, 0]) * (
                dshape_xi@top_nodal)
    np.testing.assert_allclose(center_step, integrated_top,
                               rtol=0., atol=2e-14)
    np.testing.assert_allclose(frame_step[:, 2], integrated_director,
                               rtol=0., atol=2e-14)
    ex = np.array((1., 0., 0.))
    tangent_step = integrated_top_gradient-ez*(
        ez@integrated_top_gradient+integrated_director@ex)
    tangent_step -= ex*(ex@tangent_step)
    side_step = (np.cross(integrated_director, ex)+
                 np.cross(ez, tangent_step))
    np.testing.assert_allclose(frame_step[:, 0], tangent_step,
                               rtol=0., atol=2e-14)
    np.testing.assert_allclose(frame_step[:, 1], side_step,
                               rtol=0., atol=2e-14)


def test_common_finite_rigid_motion_is_objective():
    rail = _rail()
    initial = np.zeros((2, 3, 6))
    c0, r0, _, _ = rail.pose(initial, station=.37)
    axis = np.array((.3, -.4, .5))
    axis /= np.linalg.norm(axis)
    angle = .42
    vector = axis*angle
    rotation = rotation_matrix_from_vector(vector)
    translation = np.array((.04, -.03, .07))
    total = initial.copy()
    total[:, :, :3] = (rail.reference@rotation.T+translation-
                       rail.reference)
    total[:, :, 3:] = vector
    center, frame, _, _ = rail.pose(total, station=.37)
    np.testing.assert_allclose(center, rotation@c0+translation,
                               rtol=0., atol=3e-14)
    np.testing.assert_allclose(frame, rotation@r0, rtol=0., atol=3e-14)

    deformed = np.random.default_rng(81).normal(scale=.004, size=(2, 3, 6))
    old_center, old_frame, _, _ = rail.pose(deformed, station=.37)
    transformed = deformed.copy()
    transformed[:, :, :3] = (
        (rail.reference+deformed[:, :, :3])@rotation.T + translation -
        rail.reference)
    for i in range(2):
        for j in range(3):
            local = rotation_matrix_from_vector(deformed[i, j, 3:])
            transformed[i, j, 3:] = Rotation.from_matrix(
                rotation@local).as_rotvec()
    new_center, new_frame, _, _ = rail.pose(transformed, station=.37)
    np.testing.assert_allclose(new_center, rotation@old_center+translation,
                               rtol=0., atol=3e-14)
    np.testing.assert_allclose(new_frame, rotation@old_frame,
                               rtol=0., atol=3e-14)


def test_finite_pose_matches_independent_width_quadrature():
    rail = _rail()
    random = np.random.default_rng(109)
    total = random.normal(scale=.015, size=(2, 3, 6))
    station = .31
    center, frame, _, _ = rail.pose(total, station=station)
    nodal_director = np.array([
        [rotation_matrix_from_vector(total[i, j, 3:])[:, 2]
         for j in range(3)] for i in range(2)])
    nodal_top = rail.reference+total[:, :, :3]+.5*rail.thickness*nodal_director
    average_top, average_director = np.zeros((2, 3)), np.zeros((2, 3))
    gauss, quadrature = np.polynomial.legendre.leggauss(4)
    for cell, (lo, hi) in enumerate(((-.05, 0.), (0., .05))):
        for point, weight in zip(gauss, quadrature):
            y = (lo+hi)/2.+point*(hi-lo)/2.
            local = (y-rail.y_cuts[cell])/(rail.y_cuts[cell+1]-rail.y_cuts[cell])
            shape = np.array((1.-local, local))
            average_top += (weight*(hi-lo)/2./.1 *
                            np.einsum("j,ijk->ik", shape,
                                      nodal_top[:, cell:cell+2]))
            average_director += (weight*(hi-lo)/2./.1 *
                                 np.einsum("j,ijk->ik", shape,
                                           nodal_director[:, cell:cell+2]))
    expected_center = ((1.-station)*average_top[0]+station*average_top[1])
    normal = ((1.-station)*average_director[0]+
              station*average_director[1])
    normal /= np.linalg.norm(normal)
    raw = average_top[1]-average_top[0]
    tangent = raw-normal*(normal@raw)
    tangent /= np.linalg.norm(tangent)
    expected_frame = np.column_stack((tangent, np.cross(normal, tangent), normal))
    np.testing.assert_allclose(center, expected_center, rtol=0., atol=2e-14)
    np.testing.assert_allclose(frame, expected_frame, rtol=0., atol=2e-14)


def test_actual_chart_derivative_and_shear_drill_freedom():
    rail = _rail()
    random = np.random.default_rng(919)
    total = random.normal(scale=.003, size=(2, 3, 6))
    direction = random.normal(scale=.06, size=(2, 3, 6))
    center, frame, dc, dr = rail.pose(total, station=.29,
                                       direction=direction)
    h = 1e-6
    plus_c, plus_r, _, _ = rail.pose(total+h*direction, station=.29)
    minus_c, minus_r, _, _ = rail.pose(total-h*direction, station=.29)
    np.testing.assert_allclose(dc, (plus_c-minus_c)/(2*h),
                               rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(dr, (plus_r-minus_r)/(2*h),
                               rtol=2e-7, atol=2e-9)
    np.testing.assert_allclose(frame.T@frame, np.eye(3), atol=2e-14)

    zero = np.zeros_like(total)
    drill = zero.copy()
    drill[:, :, 5] = 1.
    _, _, drill_center, drill_frame = rail.pose(zero, direction=drill)
    np.testing.assert_allclose(drill_center, 0., atol=1e-15)
    np.testing.assert_allclose(drill_frame, 0., atol=1e-15)
    shear = zero.copy()
    shear[1, :, 2] = .1
    _, _, _, shear_frame = rail.pose(zero, direction=shear)
    np.testing.assert_allclose(shear_frame, 0., atol=1e-14)
    assert np.linalg.norm(rail.pose(zero, direction=shear)[2]) > 0.


def test_yaw_work_uses_width_average_and_director_rotation_port():
    rail = _rail()
    zero = np.zeros((2, 3, 6))
    lateral = zero.copy()
    lateral[1, :, 1] = (0., 1., 0.)
    yaw = rail.spin(zero, lateral)[2]
    assert yaw == pytest.approx(.75/.1, abs=1e-13)
    roll = zero.copy()
    roll[:, :, 3] = 1.
    assert rail.spin(zero, roll)[0] == pytest.approx(1., abs=1e-13)


def test_native_six_channel_reference_and_actual_update_derivative():
    rail = _rail()
    beam = build(4, z_offset=.0675, assemble=False)
    element, cell, weight, _ = beam._locate(.95)
    port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    state = beam.context.initial.mechanical
    zero = np.zeros((2, 3, 6))
    connection = RailL1LogConnection.at_reference(rail, zero, port, state)
    np.testing.assert_allclose(connection.evaluate(zero, state)[0],
                               0., atol=3e-15)
    wrong_element, wrong_cell, wrong_weight, _ = beam._locate(.925)
    wrong_port = NativeHalfCellSpatialPort(
        beam.context, wrong_element, wrong_cell, wrong_weight)
    with pytest.raises(ValueError, match="reference station/offset"):
        RailL1LogConnection.at_reference(rail, zero, wrong_port, state)
    nonvirgin_shell = zero.copy()
    nonvirgin_shell[1, 1, 1] = 1e-3
    with pytest.raises(ValueError, match="virgin"):
        RailL1LogConnection.at_reference(
            rail, nonvirgin_shell, port, state)
    changed = np.zeros(beam.context.count)
    changed[6] = 1e-4
    with pytest.raises(ValueError, match="virgin"):
        RailL1LogConnection.at_reference(
            rail, zero, port,
            beam.context.advance(state, changed))
    eccentric = RailL1LogConnection.at_reference(
        rail, zero, port, state, beam_offset=(0., 0., -.04),
        expected_reference_offset_shell=(0., 0., .01))
    np.testing.assert_allclose(eccentric.evaluate(zero, state)[0],
                               0., atol=3e-15)

    columns = []
    beam_zero = np.zeros(beam.context.count)
    for axis in range(3):
        direction = zero.copy()
        direction[:, :, axis] = 1.
        columns.append(connection.evaluate(
            zero, state, shell_direction=direction,
            beam_direction=beam_zero)[1])
    for axis in (3, 4):
        direction = zero.copy()
        direction[:, :, axis] = 1.
        columns.append(connection.evaluate(
            zero, state, shell_direction=direction,
            beam_direction=beam_zero)[1])
    lateral = zero.copy()
    lateral[1, 1, 1] = .1
    columns.append(connection.evaluate(
        zero, state, shell_direction=lateral,
        beam_direction=beam_zero)[1])
    assert np.linalg.matrix_rank(np.column_stack(columns), tol=1e-10) == 6

    rng = np.random.default_rng(127)
    total = rng.normal(scale=.002, size=(2, 3, 6))
    shell_direction = rng.normal(scale=.01, size=(2, 3, 6))
    beam_direction = rng.normal(scale=1e-3, size=beam.context.count)
    beam_direction[:6] = 0.
    native_increment = rng.normal(scale=1e-4, size=beam.context.count)
    native_increment[:6] = 0.
    moved = beam.context.advance(state, native_increment)
    gap, derivative = connection.evaluate(
        total, moved, shell_direction=shell_direction,
        beam_direction=beam_direction)
    assert np.linalg.norm(gap) > 0.
    h = 1e-5
    plus = connection.evaluate(
        total+h*shell_direction,
        beam.context.advance(moved, h*beam_direction))[0]
    minus = connection.evaluate(
        total-h*shell_direction,
        beam.context.advance(moved, -h*beam_direction))[0]
    np.testing.assert_allclose(derivative, (plus-minus)/(2*h),
                               rtol=3e-7, atol=3e-9)


def test_roll_and_yaw_ports_have_physical_shell_energy_without_drilling():
    xs = np.array((0., .45, .9, 1.))
    ys = np.array((-.1, 0., .1))
    _, physical = raw_e4_tangent(xs, ys, thickness=.035)
    count = physical.shape[0]
    roll, yaw, drill = (np.zeros(count) for _ in range(3))
    for j in range(len(ys)):
        for i in (2, 3):
            roll[6*(j*len(xs)+i)+3] = 1.
            drill[6*(j*len(xs)+i)+5] = 1.
    yaw[6*(1*len(xs)+3)+1] = 1.
    assert float(roll@physical@roll) > 0.
    assert float(yaw@physical@yaw) > 0.
    assert abs(float(drill@physical@drill)) < 1e-8*float(roll@physical@roll)


def test_normal_moment_virtual_work_is_width_distributed():
    rail = _rail()
    beam = build(4, z_offset=.0675, assemble=False)
    element, cell, weight, _ = beam._locate(.95)
    port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    state = beam.context.initial.mechanical
    zero = np.zeros((2, 3, 6))
    connection = RailL1LogConnection.at_reference(rail, zero, port, state)
    beam_zero = np.zeros(beam.context.count)
    shell_y_forces = np.zeros((2, 3))
    for i in range(2):
        for j in range(3):
            direction = zero.copy()
            direction[i, j, 1] = 1.
            shell_y_forces[i, j] = connection.evaluate(
                zero, state, shell_direction=direction,
                beam_direction=beam_zero)[1][5]
    np.testing.assert_allclose(shell_y_forces.sum(axis=1),
                               (10., -10.), atol=2e-13)
    np.testing.assert_allclose(shell_y_forces.sum(axis=0), 0., atol=2e-13)
    np.testing.assert_allclose(shell_y_forces[0]/10., rail.weights,
                               atol=2e-14)
    x = rail.reference[:, :, 0]
    assert np.sum(x*shell_y_forces) == pytest.approx(-1., abs=2e-14)
    beam_spin = np.zeros(beam.context.count)
    column = beam.context.nodal_count+24*element+3*cell+2
    beam_spin[column] = 1.
    assert connection.evaluate(
        zero, state, shell_direction=zero,
        beam_direction=beam_spin)[1][5] == pytest.approx(1., abs=2e-13)


@pytest.mark.parametrize("thickness", (.035, .07))
def test_rail_transfers_physical_wrench_without_shell_drilling_completion(
        thickness):
    rail = _rail(thickness)
    beam = build(4, z_offset=.5*thickness+.05, assemble=False)
    element, cell, weight, _ = beam._locate(.95)
    port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    state = beam.context.initial.mechanical
    zero = np.zeros((2, 3, 6))
    connection = RailL1LogConnection.at_reference(rail, zero, port, state)
    shell, shell_frame, _, _ = rail.pose(zero)
    high, low, beam_frame = port.pose_pair(state)
    beam_point = high+low+beam_frame@connection.beam_offset
    xs, ys = np.array((0., .45, .9, 1.)), rail.y_cuts
    _, physical = raw_e4_tangent(xs, ys, thickness=rail.thickness)
    free = np.array([6*(j*len(xs)+i)+axis
                     for j in range(len(ys)) for i in range(1, len(xs))
                     for axis in range(5)])
    factor = linalg.cho_factor(physical[np.ix_(free, free)], lower=True)

    for multiplier in (np.array((0., 0., 0., 1., 0., 0.)),
                       np.array((0., 0., 0., 0., 1., 0.)),
                       np.array((0., 0., 0., 0., 0., 1.)),
                       np.array((2., -3., 5., 1., 2., -1.))):
        _, gap, gradient, _ = connection.work_tangent(
            zero, state, multiplier)
        np.testing.assert_allclose(gap, 0., rtol=0., atol=2e-14)
        local_shell = gradient[:zero.size].reshape(2, 3, 6)
        force = np.zeros(physical.shape[0])
        for i in range(2):
            for j in range(3):
                node = j*len(xs)+(i+2)
                force[6*node:6*node+6] += local_shell[i, j]
        np.testing.assert_array_equal(force[5::6], 0.)
        nodal = local_shell.reshape(-1, 6)
        shell_force = nodal[:, :3].sum(axis=0)
        shell_moment = np.sum(
            np.cross(rail.reference.reshape(-1, 3), nodal[:, :3])+
            nodal[:, 3:], axis=0)
        pose = connection.joint.evaluate(
            np.array((shell, beam_point)),
            np.array((shell_frame, beam_frame)), multiplier)
        beam_force, beam_moment = pose.residual[6:9], pose.residual[9:12]
        np.testing.assert_allclose(shell_force, pose.residual[:3],
                                   rtol=0., atol=2e-13)
        np.testing.assert_allclose(
            shell_moment, pose.residual[3:6]+np.cross(shell, shell_force),
            rtol=0., atol=3e-13)
        np.testing.assert_allclose(shell_force+beam_force, 0.,
                                   rtol=0., atol=2e-13)
        np.testing.assert_allclose(
            shell_moment+beam_moment+np.cross(beam_point, beam_force),
            0., rtol=0., atol=3e-13)
        arm = beam_frame@connection.beam_offset
        beam_axis_wrench = np.r_[beam_force,
                                 beam_moment+np.cross(arm, beam_force)]
        np.testing.assert_allclose(
            gradient[zero.size:], port.pullback(state, beam_axis_wrench),
            rtol=0., atol=3e-13)
        displacement = linalg.cho_solve(factor, force[free])
        physical_work = float(force[free]@displacement)
        assert physical_work > 0.
        full_displacement = np.zeros_like(force)
        full_displacement[free] = displacement
        reactions = physical@full_displacement-force
        np.testing.assert_allclose(reactions[free], 0., rtol=0., atol=2e-7)
        reaction_nodes = reactions.reshape(-1, 6)
        points = np.array([(x, y, 0.) for y in ys for x in xs])
        reaction_force = reaction_nodes[:, :3].sum(axis=0)
        reaction_moment = np.sum(
            np.cross(points, reaction_nodes[:, :3])+reaction_nodes[:, 3:],
            axis=0)
        np.testing.assert_allclose(shell_force+reaction_force, 0.,
                                   rtol=0., atol=2e-7)
        np.testing.assert_allclose(shell_moment+reaction_moment, 0.,
                                   rtol=0., atol=2e-7)
        np.testing.assert_allclose(
            displacement@physical[np.ix_(free, free)]@displacement,
            physical_work, rtol=2e-12, atol=1e-16)


def _segmented_rail(*, beam_offset=(0., 0., -.05),
                    expected_offset=(0., 0., 0.),
                    x_cuts=(.5, .75, 1.)):
    beam = build(4, z_offset=.0675, assemble=False)
    y = np.array((-.1, 0., .1))
    reference = np.array([[(x, value, 0.) for value in y]
                          for x in x_cuts])

    def port_at(station):
        element, cell, weight, _ = beam._locate(station)
        return NativeHalfCellSpatialPort(
            beam.context, element, cell, weight)

    line = SegmentedRailWeakWork.at_reference(
        reference, y, (-.05, .05), .035, port_at,
        beam.context.initial.mechanical, beam_offset=beam_offset,
        expected_reference_offset_shell=expected_offset)
    return line, beam


def test_reference_spatial_wrench_map_matches_joint_virtual_work_and_origin():
    shell = Rotation.from_rotvec((.3, -.2, .1)).as_matrix()
    beam = Rotation.from_rotvec((-.1, .4, .2)).as_matrix()
    frames = np.array((shell, beam))
    shell_position = np.array((2., -3., 4.))
    arm = np.array((.12, -.07, .3))
    positions = np.array((shell_position, shell_position+arm))
    joint = RigidPoseJoint(positions, frames)
    covector = np.array((1., 2., 3., 4., 5., 6.))
    force, moment = np.split(reference_spatial_wrench_map(joint)@covector, 2)
    beam_couple = beam@covector[3:]
    np.testing.assert_allclose(force, shell@covector[:3],
                               rtol=0., atol=2e-15)
    np.testing.assert_allclose(moment, beam_couple+np.cross(arm, force),
                               rtol=0., atol=2e-15)
    result = joint.evaluate(positions, frames, covector)
    residual = result.residual[:12]
    np.testing.assert_allclose(residual, np.r_[-force, -moment,
                                               force, beam_couple],
                               rtol=0., atol=2e-14)
    variation = np.array((.03, -.02, .04, .01, .06, -.05,
                          -.04, .02, .05, -.03, .04, .02))
    spatial_translation = (variation[6:9]-variation[:3]-
                           np.cross(variation[9:12], arm))
    spatial_rotation = variation[9:12]-variation[3:6]
    assert covector@result.constraint_jacobian@variation == pytest.approx(
        force@spatial_translation+moment@spatial_rotation,
        rel=0., abs=2e-14)
    shift = np.array((40., -23., 11.))
    shifted = RigidPoseJoint(positions+shift, frames)
    np.testing.assert_allclose(reference_spatial_wrench_map(shifted),
                               reference_spatial_wrench_map(joint),
                               rtol=0., atol=2e-14)
    turn = Rotation.from_rotvec((.25, .1, -.3)).as_matrix()
    rotated = RigidPoseJoint(np.array([turn@position for position in positions]),
                             np.array([turn@frame for frame in frames]))
    physical_turn = np.zeros((6, 6))
    physical_turn[:3, :3] = turn
    physical_turn[3:, 3:] = turn
    np.testing.assert_allclose(reference_spatial_wrench_map(rotated),
                               physical_turn@reference_spatial_wrench_map(joint),
                               rtol=0., atol=2e-14)


def test_flat_reference_physical_gram_uses_actual_offset_and_scales():
    line, _ = _segmented_rail(beam_offset=(0., 0., 0.),
                              expected_offset=(0., 0., .05))
    basis = FiniteRailDensityBasis.on_two_segments(
        (.5, .75, 1.), ("P1", "P0", "P1", "P0", "P1", "P0"))
    density = np.array((.8, -.3, 1.2, .4, -.6, 1.1))
    coefficients = []
    for channel, kind in enumerate(basis.kinds):
        coefficients.extend([density[channel]]*(2 if kind == "P0" else 3))
    coefficients = np.asarray(coefficients)
    np.testing.assert_allclose(basis.density_samples(coefficients),
                               np.tile(density, (4, 1)),
                               rtol=0., atol=2e-16)
    force_scale, length_scale = 5., .2
    gram = basis.flat_reference_physical_quadrature_gram(
        line, force_scale=force_scale,
        characteristic_length=length_scale)
    assert not gram.flags.writeable
    assert np.linalg.eigvalsh(gram).min() > 0.
    force, moment = np.split(
        reference_spatial_wrench_map(line.connections[0].joint)@density, 2)
    length = .5
    expected = length**2*(force@force/force_scale**2+
                          moment@moment/(force_scale*length_scale)**2)
    assert coefficients@gram@coefficients == pytest.approx(expected,
                                                           rel=2e-14)
    beam_axis_couple = line.connections[0].joint.reference_frames[1]@density[3:]
    assert np.linalg.norm(moment-beam_axis_couple) > 1e-3
    with pytest.raises(ValueError, match="positive frozen"):
        basis.flat_reference_physical_quadrature_gram(
            line, force_scale=0., characteristic_length=length_scale)


def test_finite_density_basis_has_exact_piecewise_l2_gram_and_owned_data():
    cuts = np.array((.5, .7, 1.))
    kinds = ("P0", "P1", "P0", "P1", "P0", "P1")
    basis = FiniteRailDensityBasis.on_two_segments(cuts, kinds)
    cuts[1] = .8
    np.testing.assert_array_equal(basis.cuts, (.5, .7, 1.))
    for array in (basis.cuts, basis.stations, basis.weights,
                  basis.sample_matrix, basis.raw_l2_gram):
        assert not array.flags.writeable
    lengths = (.2, .3)
    p0 = np.diag(lengths)
    p1 = np.array(((lengths[0]/3, lengths[0]/6, 0.),
                   (lengths[0]/6, sum(lengths)/3, lengths[1]/6),
                   (0., lengths[1]/6, lengths[1]/3)))
    expected = np.zeros_like(basis.raw_l2_gram)
    offset = 0
    for kind in kinds:
        block = p0 if kind == "P0" else p1
        n = len(block)
        expected[offset:offset+n, offset:offset+n] = block
        offset += n
    np.testing.assert_allclose(basis.raw_l2_gram, expected,
                               rtol=0., atol=2e-16)
    assert np.linalg.eigvalsh(basis.raw_l2_gram).min() > 0.
    coefficients = np.arange(offset, dtype=float)/7.
    samples = basis.density_samples(coefficients)
    coefficients[:] = 0.
    assert np.linalg.norm(samples) > 0.
    with pytest.raises(ValueError, match="P0/P1"):
        FiniteRailDensityBasis.on_two_segments((.5, .7, 1.),
                                               ("P0",)*5+("dual",))


def test_finite_density_coefficients_are_conjugate_to_projected_line_gaps():
    line, beam = _segmented_rail()
    basis = FiniteRailDensityBasis.on_two_segments(
        (.5, .75, 1.), ("P1", "P0", "P1", "P0", "P1", "P0"))
    np.testing.assert_array_equal(basis.stations, line.stations)
    rng = np.random.default_rng(119)
    shell = rng.normal(scale=.008, size=(3, 3, 6))
    coefficients = rng.normal(scale=.5, size=basis.raw_l2_gram.shape[0])
    state = beam.context.initial.mechanical
    work, gaps, gradient, hessian = basis.work_tangent(
        line, shell, state, coefficients)
    rows = basis.constraint_rows(gaps)
    assert work == pytest.approx(coefficients@rows, rel=2e-12, abs=2e-12)
    direct = line.work_tangent(shell, state,
                               basis.density_samples(coefficients))
    for actual, expected in zip((work, gaps, gradient, hessian), direct):
        np.testing.assert_allclose(actual, expected, rtol=0., atol=0.)
    direction = rng.normal(size=len(coefficients))
    h = 1e-5
    forward = basis.work_tangent(line, shell, state,
                                 coefficients+h*direction)[0]
    backward = basis.work_tangent(line, shell, state,
                                  coefficients-h*direction)[0]
    assert (forward-backward)/(2*h) == pytest.approx(
        rows@direction, rel=2e-8, abs=2e-9)
    wrong = FiniteRailDensityBasis.on_two_segments(
        (.5, .74, 1.), ("P0",)*6)
    with pytest.raises(ValueError, match="do not match"):
        wrong.work_tangent(line, shell, state,
                           np.zeros(wrong.raw_l2_gram.shape[0]))


def test_segmented_rail_constant_yaw_density_cancels_interior_shell_work():
    line, beam = _segmented_rail()
    density = np.zeros((4, 6))
    density[:, 5] = 1.
    work, gaps, gradient, hessian = line.work_tangent(
        np.zeros((3, 3, 6)), beam.context.initial.mechanical, density)
    assert work == pytest.approx(0., abs=1e-13)
    np.testing.assert_allclose(gaps, 0., rtol=0., atol=2e-14)
    np.testing.assert_allclose(line.weights.sum(), .5, rtol=0., atol=2e-15)
    shell = gradient[:54].reshape(3, 3, 6)
    expected = np.array((.125, .75, .125))
    np.testing.assert_allclose(shell[:, :, 1],
                               np.array((expected, np.zeros(3), -expected)),
                               rtol=0., atol=2e-13)
    np.testing.assert_array_equal(shell[:, :, 5], 0.)
    assert np.linalg.norm(hessian-hessian.T, ord=np.inf) < 2e-9
    nodal_count = beam.context.nodal_count
    beam_rotation_z = sum(
        gradient[54+nodal_count+24*element+3*cell+2]
        for element in range(len(beam.context.elements)) for cell in (0, 1))
    assert beam_rotation_z == pytest.approx(.5, abs=2e-13)


def test_growing_rail_space_preserves_shared_work_and_exact_density_gram():
    cuts = np.array((.5, .625, .75, .875, 1.))
    line, beam = _segmented_rail(x_cuts=cuts)
    kinds = ("P0", "P1", "P0", "P1", "P0", "P1")
    basis = FiniteRailDensityBasis.on_material_segments(cuts, kinds)
    assert basis.sample_matrix.shape == (48, 27)
    assert len(line.connections) == 8
    lengths = np.diff(cuts)
    p0 = np.diag(lengths)
    p1 = np.diag(np.r_[lengths[0], lengths[:-1]+lengths[1:],
                           lengths[-1]]/3.)
    p1 += np.diag(lengths/6., 1)+np.diag(lengths/6., -1)
    expected = linalg.block_diag(*(p0 if kind == "P0" else p1
                                   for kind in kinds))
    np.testing.assert_allclose(basis.raw_l2_gram, expected,
                               rtol=0., atol=2e-16)

    density = np.zeros((8, 6))
    density[:, 5] = 1.
    zero = np.zeros((5, 3, 6))
    work, gaps, gradient, _ = line.work_tangent(
        zero, beam.context.initial.mechanical, density)
    assert work == pytest.approx(0., abs=1e-13)
    np.testing.assert_allclose(gaps, 0., rtol=0., atol=2e-14)
    shell = gradient[:zero.size].reshape(zero.shape)
    expected_edge = np.array((.125, .75, .125))
    np.testing.assert_allclose(shell[:, :, 1],
                               np.vstack((expected_edge, np.zeros((3, 3)),
                                          -expected_edge)),
                               rtol=0., atol=2e-13)
    np.testing.assert_array_equal(shell[:, :, 5], 0.)

    xs = np.r_[(0., .25), cuts]
    _, physical = raw_e4_tangent(xs, np.array((-.1, 0., .1)),
                                 thickness=.035)
    free = np.array([6*(j*len(xs)+i)+axis
                     for j in range(3) for i in range(1, len(xs))
                     for axis in range(5)])
    factor = linalg.cho_factor(physical[np.ix_(free, free)], lower=True)
    load = np.zeros(len(physical))
    for i in range(len(cuts)):
        for j in range(3):
            node = j*len(xs)+i+2
            load[6*node:6*node+6] += shell[i, j]
    np.testing.assert_array_equal(load[5::6], 0.)
    compliance = float(load[free]@linalg.cho_solve(factor, load[free]))
    assert compliance > 0.

    rng = np.random.default_rng(243)
    moved_shell = rng.normal(scale=.004, size=zero.shape)
    coefficients = rng.normal(scale=.2, size=27)
    work, gaps, _, _ = basis.work_tangent(
        line, moved_shell, beam.context.initial.mechanical, coefficients)
    assert work == pytest.approx(coefficients@basis.constraint_rows(gaps),
                                 rel=2e-12, abs=2e-12)


@pytest.mark.parametrize("macro_factor,segment_counts", (
    (1, (2, 4, 8)), (2, (4, 8))))
def test_physical_core_does_not_uniformly_control_growing_p0_yaw(
        macro_factor, segment_counts):
    """Fixed-ratio P0 coarsening retains a physical-core shear witness."""
    energies = []
    for segments in segment_counts:
        cuts = np.linspace(.5, 1., segments+1)
        h = .5/segments
        line, beam = _segmented_rail(x_cuts=cuts)
        zero = np.zeros((segments+1, 3, 6))
        density = np.zeros((2*segments, 6))
        density[:, 5] = np.repeat(
            (-1.)**(np.arange(segments)//macro_factor), 2)
        _, gaps, gradient, _ = line.work_tangent(
            zero, beam.context.initial.mechanical, density)
        np.testing.assert_allclose(gaps, 0., rtol=0., atol=2e-14)

        xs = np.r_[(0., .25), cuts]
        _, physical = raw_e4_tangent(xs, np.array((-.1, 0., .1)),
                                     thickness=.035)
        motion = np.zeros_like(zero)
        motion[macro_factor::2*macro_factor, :, 1] = (
            h*np.array((1., -1., 1.)))
        q = np.zeros(len(physical))
        load = np.zeros_like(q)
        local = gradient[:zero.size].reshape(zero.shape)
        for i in range(len(cuts)):
            for j in range(3):
                node = j*len(xs)+i+2
                dofs = slice(6*node, 6*node+6)
                q[dofs] = motion[i, j]
                load[dofs] += local[i, j]
        np.testing.assert_array_equal(load[5::6], 0.)
        assert q@load == pytest.approx(.25/macro_factor, abs=2e-13)
        energy = float(q@physical@q)
        assert energy > 0.
        energies.append(energy)
    np.testing.assert_allclose(
        np.array(energies[:-1])/np.array(energies[1:]), 4.,
        rtol=2e-12, atol=0.)


def test_constant_yaw_has_free_endpoint_physical_core_witness():
    """Constant yaw still performs endpoint work on a shrinking-energy mode."""
    energies = []
    for segments in (2, 4, 8):
        cuts = np.linspace(.5, 1., segments+1)
        line, beam = _segmented_rail(x_cuts=cuts)
        zero = np.zeros((segments+1, 3, 6))
        density = np.zeros((2*segments, 6))
        density[:, 5] = 1.
        _, gaps, gradient, _ = line.work_tangent(
            zero, beam.context.initial.mechanical, density)
        np.testing.assert_allclose(gaps, 0., rtol=0., atol=2e-14)

        xs = np.r_[(0., .25), cuts]
        _, physical = raw_e4_tangent(xs, np.array((-.1, 0., .1)),
                                     thickness=.035)
        q = np.zeros(len(physical))
        load = np.zeros_like(q)
        local = gradient[:zero.size].reshape(zero.shape)
        for i in range(len(cuts)):
            for j in range(3):
                node = j*len(xs)+i+2
                dofs = slice(6*node, 6*node+6)
                if i == segments:
                    q[6*node+1] = (1., -1., 1.)[j]
                load[dofs] += local[i, j]
        np.testing.assert_array_equal(load[5::6], 0.)
        assert q@load == pytest.approx(.5, abs=2e-13)
        energy = float(q@physical@q)
        assert energy > 0.
        energies.append(energy)
    np.testing.assert_allclose(
        np.array(energies[:-1])/np.array(energies[1:]), 2.,
        rtol=2e-12, atol=0.)


def test_paired_refinement_screen_work_matches_actual_rail_spin():
    """Check the screen's closed-form flat yaw work against the pose map."""
    y = np.array((-.1, 0., .1))
    bounds = (-.047, .046)
    h = .125
    phi = (-1.)**np.arange(len(y))
    average = transverse_shape_integrals(y, bounds) @ phi
    for name in ("macro_p0", "constant_endpoint"):
        amplitude = np.zeros(5)
        density = np.array((1., 1., -1., -1.))
        if name == "macro_p0":
            amplitude[2] = h
        else:
            amplitude[-1] = 1.
            density[:] = 1.
        actual = 0.
        for i in range(4):
            x0, x1 = .5+i*h, .5+(i+1)*h
            reference = np.array([[(x, value, 0.) for value in y]
                                  for x in (x0, x1)])
            rail = FlatQ4PhysicalRail.from_reference(
                reference, y, bounds, .035)
            direction = np.zeros((2, len(y), 6))
            direction[:, :, 1] = amplitude[i:i+2, None]*phi[None, :]
            actual += density[i]*h*rail.spin(
                np.zeros_like(direction), direction)[2]
        assert actual == pytest.approx(
            average * (density @ np.diff(amplitude)), abs=2e-14)


def test_full_p0_yaw_rows_include_top_surface_rotation():
    """Every analytic row must match the actual flat physical rail spin."""
    xs = np.linspace(0., 1., 9)
    y = np.array((-.1, 0., .1))
    weights = transverse_shape_integrals(y, (-.047, .046))
    action = flat_yaw_p0_action(xs, y, weights, .035)
    rng = np.random.default_rng(1903)
    q = rng.normal(scale=.01, size=(len(y), len(xs), 6))
    actual = []
    h = xs[1]-xs[0]
    for i in range(4, 8):
        reference = np.array([[(x, value, 0.) for value in y]
                              for x in xs[i:i+2]])
        rail = FlatQ4PhysicalRail.from_reference(
            reference, y, (-.047, .046), .035)
        direction = q[:, i:i+2, :].transpose(1, 0, 2)
        actual.append(h*rail.spin(np.zeros_like(direction), direction)[2])
    np.testing.assert_allclose(action@q.ravel(), actual, rtol=0., atol=2e-14)
    rotation_only = q.copy()
    rotation_only[:, :, :3] = 0.
    assert np.linalg.norm(action@rotation_only.ravel()) > 1e-5


def test_segmented_rail_constant_density_has_line_wrench_and_shell_compliance():
    line, beam = _segmented_rail()
    xs = np.array((0., .25, .5, .75, 1.))
    ys = np.array((-.1, 0., .1))
    _, physical = raw_e4_tangent(xs, ys, thickness=.035)
    free = np.array([6*(j*len(xs)+i)+axis
                     for j in range(len(ys)) for i in range(1, len(xs))
                     for axis in range(5)])
    factor = linalg.cho_factor(physical[np.ix_(free, free)], lower=True)
    reference = np.array([[(x, y, 0.) for y in ys]
                          for x in (.5, .75, 1.)])
    mean_contact = np.array((.75, 0., .0175))
    length = .5
    for density in (np.array((0., 0., 0., 1., 0., 0.)),
                    np.array((0., 0., 0., 0., 1., 0.)),
                    np.array((0., 0., 0., 0., 0., 1.)),
                    np.array((2., -3., 5., 1., 2., -1.))):
        _, gaps, gradient, _ = line.work_tangent(
            np.zeros((3, 3, 6)), beam.context.initial.mechanical,
            np.tile(density, (4, 1)))
        np.testing.assert_allclose(gaps, 0., rtol=0., atol=2e-14)
        local_shell = gradient[:54].reshape(3, 3, 6)
        force = np.zeros(physical.shape[0])
        for i in range(3):
            for j in range(3):
                node = j*len(xs)+(i+2)
                force[6*node:6*node+6] += local_shell[i, j]
        np.testing.assert_array_equal(force[5::6], 0.)
        nodal = local_shell.reshape(-1, 6)
        shell_force = nodal[:, :3].sum(axis=0)
        shell_moment = np.sum(
            np.cross(reference.reshape(-1, 3), nodal[:, :3])+
            nodal[:, 3:], axis=0)
        np.testing.assert_allclose(shell_force, -length*density[:3],
                                   rtol=0., atol=3e-13)
        np.testing.assert_allclose(
            shell_moment,
            -length*(density[3:]+np.cross(mean_contact, density[:3])),
            rtol=0., atol=5e-13)
        context = beam.context
        native = gradient[54:]
        nodal = native[:context.nodal_count].reshape(-1, 6)
        native_points = (context.initial.mechanical.positions+
                         context.initial.mechanical.position_low)
        native_force = nodal[:, :3].sum(axis=0)
        native_moment = np.sum(
            np.cross(native_points, nodal[:, :3])+nodal[:, 3:], axis=0)
        for element in range(len(context.elements)):
            base = context.nodal_count+24*element
            native_moment += native[base:base+6].reshape(2, 3).sum(axis=0)
            np.testing.assert_array_equal(native[base+6:base+24], 0.)
        np.testing.assert_allclose(shell_force+native_force, 0.,
                                   rtol=0., atol=3e-13)
        np.testing.assert_allclose(shell_moment+native_moment, 0.,
                                   rtol=0., atol=5e-13)
        displacement = linalg.cho_solve(factor, force[free])
        assert float(force[free]@displacement) > 0.
        full = np.zeros_like(force)
        full[free] = displacement
        reactions = physical@full-force
        np.testing.assert_allclose(reactions[free], 0., rtol=0., atol=5e-7)
        supported = reactions.reshape(-1, 6)
        positions = np.array([(x, y, 0.) for y in ys for x in xs])
        np.testing.assert_allclose(
            shell_force+supported[:, :3].sum(axis=0),
            0., rtol=0., atol=5e-7)
        np.testing.assert_allclose(
            shell_moment+np.sum(np.cross(positions, supported[:, :3])+
                                supported[:, 3:], axis=0),
            0., rtol=0., atol=5e-7)


def test_segmented_rail_shared_node_work_and_curvature_use_one_parent():
    line, beam = _segmented_rail()
    rng = np.random.default_rng(703)
    shell = rng.normal(scale=.035, size=(3, 3, 6))
    native_step = rng.normal(scale=.01, size=beam.context.count)
    native_step[:6] = 0.
    state = beam.context.advance(beam.context.initial.mechanical, native_step)
    density = rng.normal(scale=.6, size=(4, 6))
    work, gaps, gradient, hessian = line.work_tangent(shell, state, density)
    assert np.isfinite(work) and np.isfinite(gaps).all()
    shell_direction = rng.normal(scale=.02, size=shell.shape)
    beam_direction = rng.normal(scale=.002, size=beam.context.count)
    beam_direction[:6] = 0.
    direction = np.r_[shell_direction.ravel(), beam_direction]
    h = 1e-4
    plus = line.work_tangent(
        shell+h*shell_direction,
        beam.context.advance(state, h*beam_direction), density)[0]
    minus = line.work_tangent(
        shell-h*shell_direction,
        beam.context.advance(state, -h*beam_direction), density)[0]
    np.testing.assert_allclose(gradient@direction, (plus-minus)/(2*h),
                               rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(direction@hessian@direction,
                               (plus-2*work+minus)/h**2,
                               rtol=3e-4, atol=2e-5)


@pytest.mark.parametrize("shell_scale,native_scale",
                         ((.001, 2e-4), (.06, .01)))
def test_multiplier_work_gradient_and_curvature_follow_actual_updates(
        shell_scale, native_scale):
    rail = _rail()
    beam = build(4, z_offset=.0675, assemble=False)
    element, cell, weight, _ = beam._locate(.95)
    port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    state = beam.context.initial.mechanical
    zero = np.zeros((2, 3, 6))
    connection = RailL1LogConnection.at_reference(rail, zero, port, state)
    rng = np.random.default_rng(510)
    total = rng.normal(scale=shell_scale, size=zero.shape)
    native_step = rng.normal(scale=native_scale, size=beam.context.count)
    native_step[:6] = 0.
    moved = beam.context.advance(state, native_step)
    shell_direction = rng.normal(scale=.02, size=zero.shape)
    beam_direction = rng.normal(scale=.002, size=beam.context.count)
    beam_direction[:6] = 0.
    multiplier = np.array((1.1, -.8, .7, .4, -.3, .6))

    work, gap, gradient, hessian = connection.work_tangent(
        total, moved, multiplier)
    expected_gap, derivative = connection.evaluate(
        total, moved, shell_direction=shell_direction,
        beam_direction=beam_direction)
    direction = np.r_[shell_direction.ravel(), beam_direction]
    np.testing.assert_allclose(gap, expected_gap, rtol=0., atol=2e-12)
    assert work == pytest.approx(multiplier@gap, rel=2e-12, abs=2e-12)
    np.testing.assert_allclose(gradient@direction, multiplier@derivative,
                               rtol=3e-9, atol=3e-11)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0., atol=2e-10)
    assert not np.any(gradient[total.size+beam.context.nodal_count:
                               total.size+beam.context.nodal_count+
                               24*element])

    for step_size in (2e-4, 1e-4):
        plus = connection.evaluate(
            total+step_size*shell_direction,
            beam.context.advance(moved, step_size*beam_direction))[0]
        minus = connection.evaluate(
            total-step_size*shell_direction,
            beam.context.advance(moved, -step_size*beam_direction))[0]
        curvature_fd = multiplier@(plus-2*gap+minus)/step_size**2
        np.testing.assert_allclose(direction@hessian@direction,
                                   curvature_fd, rtol=2e-4, atol=2e-5)

    step_size = 1e-4
    mixed = np.empty((2, 2))
    for i, shell_sign in enumerate((-1., 1.)):
        for j, beam_sign in enumerate((-1., 1.)):
            candidate = beam.context.advance(
                moved, beam_sign*step_size*beam_direction)
            mixed[i, j] = multiplier@connection.evaluate(
                total+shell_sign*step_size*shell_direction, candidate)[0]
    cross_fd = (mixed[1, 1]-mixed[1, 0]-mixed[0, 1]+mixed[0, 0]) / (
        4*step_size**2)
    cross_exact = (shell_direction.ravel()@hessian[:total.size, total.size:]@
                   beam_direction)
    np.testing.assert_allclose(cross_exact, cross_fd,
                               rtol=2e-4, atol=2e-5)
