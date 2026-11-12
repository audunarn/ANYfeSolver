"""Six-channel native retained half-cell pose and chart derivatives."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver.boundary import BoundaryCondition
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_generalized_state import Context, Program

from scripts._b3_e4_f03_native_fixture import SECTION_DIAGONAL, build
from scripts._b3_e4_f03_native_spatial_port import (NativeHalfCellSpatialPort,
                                                     NativeNodalSpatialPort)


def test_reference_fixture_can_match_square_bar_torsion_without_changing_other_laws():
    original = build(4)
    square_j = 1.40577014955e-5
    matched = build(4, z_offset=.085, torsion_constant=square_j)
    original_section = next(iter(original.model.mesh.elements.values())).section
    matched_section = next(iter(matched.model.mesh.elements.values())).section
    np.testing.assert_array_equal(np.diag(original_section.elastic), SECTION_DIAGONAL)
    np.testing.assert_array_equal(np.diag(matched_section.elastic)[[0, 1, 2, 4, 5]],
                                  SECTION_DIAGONAL[[0, 1, 2, 4, 5]])
    assert matched_section.elastic[3, 3] == pytest.approx(
        SECTION_DIAGONAL[3] * square_j / (2 * 8.333333333333333e-6))
    assert matched.dense_oracle_max_error < 1e-9
    with pytest.raises(ValueError, match="torsion constant"):
        build(4, torsion_constant=0.)


def _curved_context():
    model = FEModel("f03-curved-pre-rotated-station-port")
    coordinates, frames = [], []
    for node, x in enumerate((0., .5, 1.), 1):
        y = .08*x*(1-x)
        model.add_node(node, x, y, 0.)
        coordinates.append((x, y, 0.))
        tangent = np.array((1., .08*(1-2*x), 0.))
        tangent /= np.linalg.norm(tangent)
        transverse = np.array((-tangent[1], tangent[0], 0.))
        frame = np.column_stack((tangent, transverse, (0., 0., 1.)))
        frames.append(Rotation.from_rotvec(.12*x*tangent).as_matrix()@frame)
    reference = CenteredCurvedBeam3ReferenceGeometry(
        np.array(coordinates), np.array(frames))
    section = EllipsoidalGeneralizedSection(
        np.diag(SECTION_DIAGONAL), np.eye(6), 1e20, 1e9)
    native = NativeGeneralizedStaticElement(
        1, (1, 2, 3), reference, section, order=4)
    model.add_element(1, native)
    model.materials[native.material_name] = native.section
    model.add_boundary_condition(BoundaryCondition(
        "root", [1], {dof: 0. for dof in
                      ("ux", "uy", "uz", "rx", "ry", "rz")}))
    context = Context(model, Program(
        (1.,), DistributedPattern(LinePattern(()), ())))
    return context, reference


def test_straight_native_port_matches_existing_planar_trace():
    fixture = build(4)
    state = fixture.context.initial.mechanical
    for station in (0., .25, .32, .75, 1.):
        element, cell, weight, _ = fixture._locate(station)
        port = NativeHalfCellSpatialPort(
            fixture.context, element, cell, weight)
        position, frame = port.pose(state)
        old_position, old_frame = fixture.pose(station)
        np.testing.assert_allclose(position, old_position, rtol=0, atol=1e-14)
        np.testing.assert_allclose(frame, old_frame, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            port.tangent(state)[[0, 1, 5]],
            fixture.planar_trace_jacobian(station), rtol=0, atol=1e-14)


@pytest.mark.parametrize("half_cells", (8, 16, 32))
def test_straight_station_trace_has_analytical_position_and_twist_order(
        half_cells):
    fixture = build(half_cells)
    context = fixture.context
    initial = context.initial.mechanical
    curvature, twist_rate = 1e-4, 1e-5
    delta = np.zeros(context.count)
    x = initial.positions[:, 0]+initial.position_low[:, 0]
    delta[2:context.nodal_count:6] = .5*curvature*x*x
    for half in range(half_cells):
        element, cell = divmod(half, 2)
        delta[context.nodal_count+24*element+3*cell] = (
            twist_rate*(half+.5)/half_cells)
    trial = context.advance(initial, delta)
    points, weights = np.polynomial.legendre.leggauss(5)
    h = 1/half_cells
    position_error_sq = rotation_error_sq = 0.
    for half in range(half_cells):
        element, cell = divmod(half, 2)
        for point, weight in zip(points, weights):
            fraction = (point+1)/2
            station = (half+fraction)*h
            port = NativeHalfCellSpatialPort(context, element, cell, fraction)
            high, low, frame = port.pose_pair(trial)
            z = high[2]+low[2]
            rx = Rotation.from_matrix(frame).as_rotvec()[0]
            position_error_sq += (h*weight/2)*(
                z-.5*curvature*station**2)**2
            rotation_error_sq += (h*weight/2)*(
                rx-twist_rate*station)**2
    np.testing.assert_allclose(np.sqrt(position_error_sq),
                               curvature*h*h/np.sqrt(120), rtol=1e-9)
    np.testing.assert_allclose(np.sqrt(rotation_error_sq),
                               twist_rate*h/np.sqrt(12), rtol=1e-9)


def test_curved_pre_rotated_port_first_second_and_work_variations():
    context, reference = _curved_context()
    original = context.initial.mechanical
    port = NativeHalfCellSpatialPort(context, 0, 0, .37)
    position, frame = port.pose(original)
    np.testing.assert_allclose(position, reference.position(-.63),
                               rtol=0, atol=2e-15)
    np.testing.assert_allclose(frame, reference.frame(-.63),
                               rtol=0, atol=2e-15)
    base = np.zeros(context.count)
    base[6:9] = (.002, -.003, .001)
    base[12:15] = (-.001, .002, .001)
    base[context.nodal_count:context.nodal_count+6] = (
        .04, -.03, .02, -.02, .01, .03)
    state = context.advance(original, base)
    first, second = np.zeros(context.count), np.zeros(context.count)
    first[6:12] = (.2, -.15, .07, .1, -.2, .15)
    first[12:15] = (-.12, .06, .08)
    first[context.nodal_count:context.nodal_count+6] = (
        .23, -.18, .13, .04, .01, -.02)
    second[6:9] = (.07, .04, -.06)
    second[context.nodal_count:context.nodal_count+6] = (
        -.14, .11, .09, .02, -.03, .01)
    h = 1e-5
    plus = port.pose(context.advance(state, h*first))
    minus = port.pose(context.advance(state, -h*first))
    numerical = np.r_[
        (plus[0]-minus[0])/(2*h),
        Rotation.from_matrix(plus[1]@minus[1].T).as_rotvec()/(2*h)]
    np.testing.assert_allclose(port.tangent(state)@first, numerical,
                               rtol=2e-7, atol=2e-9)
    wrench = np.array((18., -25., 31., 7., -9., 13.))
    assert abs(port.pullback(state, wrench)@first-wrench@numerical) < 2e-7
    lever = state.cell_rotations[0, 0]@reference.half_cell_lift(0, .37)
    cell_column = context.nodal_count
    np.testing.assert_allclose(
        port.pullback(state, wrench)[cell_column:cell_column+3],
        np.cross(lever, wrench[:3])+wrench[3:], rtol=0, atol=1e-13)
    pure_rotation = np.zeros(context.count)
    pure_rotation[cell_column:cell_column+3] = (.12, -.07, .04)
    h_rotation = 1e-5
    plus_position = port.pose(context.advance(state, h_rotation*pure_rotation))[0]
    minus_position = port.pose(context.advance(state, -h_rotation*pure_rotation))[0]
    force_work = wrench[:3]@(plus_position-minus_position)/(2*h_rotation)
    moment_work = wrench[3:]@pure_rotation[cell_column:cell_column+3]
    assert abs((force_work+moment_work)-
               port.pullback(state, wrench)@pure_rotation) < 2e-9

    h = 2e-3
    pp = port.pose(context.advance(state, h*(first+second)))
    pm = port.pose(context.advance(state, h*(first-second)))
    mp = port.pose(context.advance(state, h*(-first+second)))
    mm = port.pose(context.advance(state, -h*(first+second)))
    position_second, frame_second = port.second_variation(
        state, first, second)
    np.testing.assert_allclose(
        position_second, (pp[0]-pm[0]-mp[0]+mm[0])/(4*h*h),
        rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(
        frame_second, (pp[1]-pm[1]-mp[1]+mm[1])/(4*h*h),
        rtol=2e-5, atol=2e-8)
    numerical_force_curvature = wrench[:3]@(
        pp[0]-pm[0]-mp[0]+mm[0])/(4*h*h)
    assert abs(numerical_force_curvature-wrench[:3]@position_second) < 1e-6


def test_half_cell_boundary_has_explicit_two_sides():
    context, reference = _curved_context()
    original = context.initial.mechanical
    left = NativeHalfCellSpatialPort(context, 0, 0, 1.)
    right = NativeHalfCellSpatialPort(context, 0, 1, 0.)
    for a, b in zip(left.pose(original), right.pose(original)):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)
    np.testing.assert_allclose(left.pose(original)[1], reference.frame(0.),
                               rtol=0, atol=1e-14)
    assert not np.array_equal(left.tangent(original)[3:],
                              right.tangent(original)[3:])
    different_cells = np.zeros(context.count)
    different_cells[context.nodal_count:context.nodal_count+3] = (.13, -.06, .03)
    different_cells[context.nodal_count+3:context.nodal_count+6] = (
        -.04, .11, -.08)
    deformed = context.advance(original, different_cells)
    left_position, left_frame = left.pose(deformed)
    right_position, right_frame = right.pose(deformed)
    np.testing.assert_allclose(left_position, right_position,
                               rtol=0, atol=1e-14)
    assert np.linalg.norm(left_frame-right_frame) > .1
    assert not np.array_equal(left.tangent(deformed)[3:],
                              right.tangent(deformed)[3:])


def test_pose_pair_retains_subulp_native_translation_and_owns_arrays():
    context, _ = _curved_context()
    delta = np.zeros(context.count)
    delta[6] = 2.**-60
    state = context.advance(context.initial.mechanical, delta)
    port = NativeHalfCellSpatialPort(context, 0, 0, 1.)
    high, low, frame = port.pose_pair(state)
    assert high[0] == .5
    assert low[0] == delta[6]
    high[0], low[0], frame[0, 0] = 99., 99., 99.
    again = port.pose_pair(state)
    assert again[0][0] == .5 and again[1][0] == delta[6]
    assert again[2][0, 0] != 99.


def test_pose_pair_preserves_low_part_at_large_native_coordinate():
    context, _ = _curved_context()
    port = NativeHalfCellSpatialPort(context, 0, 0, 1.)
    large_step = np.zeros(context.count)
    large_step[6] = 1e16
    large = context.advance(context.initial.mechanical, large_step)
    small_step = np.zeros(context.count)
    small_step[6] = .25
    shifted = context.advance(large, small_step)
    high, low, _ = port.pose_pair(shifted)
    assert high[0] == shifted.positions[1, 0]
    assert low[0] == .75
    assert port.pose(shifted)[0][0] == high[0]


def test_interior_affine_station_preserves_large_origin_fraction():
    context, reference = _curved_context()
    original = context.initial.mechanical
    positions = original.positions.copy()
    positions[1, 0], positions[2, 0] = 1e16, 1e16+2.
    state = context.make(replace(original, positions=positions).descriptor())
    port = NativeHalfCellSpatialPort(context, 0, 1, .37)
    assert reference.half_cell_lift(1, .37)[0] == 0.
    high, low, _ = port.pose_pair(state)
    assert high[0] == 1e16
    assert low[0] == .74
    lows = state.position_low.copy()
    lows[2, 0] = .125
    shifted = context.make(replace(state, position_low=lows).descriptor())
    high, low, _ = port.pose_pair(shifted)
    assert high[0] == 1e16
    assert abs(low[0]-(.74+.37*.125)) < 1e-16


def test_port_rejects_malformed_trial_mechanics():
    context, _ = _curved_context()
    port = NativeHalfCellSpatialPort(context, 0, 0, .37)
    original = context.initial.mechanical
    bad_positions = original.positions.copy()
    bad_positions[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        port.pose(replace(original, positions=bad_positions))
    bad_rotation = original.cell_rotations.copy()
    bad_rotation[0, 0, 0, 0] = 2.
    with pytest.raises(ValueError, match="frame"):
        port.tangent(replace(original, cell_rotations=bad_rotation))
    with pytest.raises(ValueError, match="strict mechanical shape"):
        port.pose(replace(original, positions=original.positions[:2]))


@pytest.mark.parametrize("cell,weight", [(2, .5), (0, -1.), (1, 1.01)])
def test_invalid_half_cell_or_weight_rejected(cell, weight):
    context, _ = _curved_context()
    with pytest.raises(ValueError, match="valid native element"):
        NativeHalfCellSpatialPort(context, 0, cell, weight)


def test_loaded_junction_has_one_nodal_pose_but_two_cell_frame_traces():
    fixture=build(16)
    context=fixture.context
    load=np.zeros(context.count)
    load[6*16+3]=100.
    free=np.array([i for i in range(context.count)
                   if i not in set(context.fixed)])
    step=np.zeros(context.count)
    step[free]=np.linalg.solve(fixture.tangent[np.ix_(free,free)].toarray(),
                               load[free])
    assert np.linalg.norm((fixture.tangent@step-load)[free])<1e-10
    trial=context.advance(context.initial.mechanical,step)
    left=NativeHalfCellSpatialPort(context,3,1,1.)
    right=NativeHalfCellSpatialPort(context,4,0,0.)
    node=NativeNodalSpatialPort(context,8)
    positions=[port.pose_pair(trial) for port in (left,node,right)]
    for high,low,_ in positions:
        np.testing.assert_allclose(high+low,(.5,0.,0.),rtol=0,atol=1e-12)
    angles=[Rotation.from_matrix(frame).as_rotvec()[0]
            for _,_,frame in positions]
    assert angles[0]<angles[1]<angles[2]
    assert angles[2]-angles[0]>1e-6
    np.testing.assert_allclose(node.tangent(trial)[:,6*8:6*9],
                               np.eye(6),rtol=0,atol=0)
    assert np.linalg.norm(left.tangent(trial)-right.tangent(trial))>1.


def test_nodal_port_uses_actual_chart_and_owns_pose_outputs():
    context,_=_curved_context()
    port=NativeNodalSpatialPort(context,1)
    initial=context.initial.mechanical
    base=np.zeros(context.count)
    base[6:12]=(1e-3,-2e-3,3e-3,.12,-.07,.05)
    state=context.advance(initial,base)
    direction=np.zeros(context.count)
    direction[6:12]=(.3,-.4,.2,.05,-.03,.07)
    h=1e-6
    plus=context.advance(state,h*direction)
    minus=context.advance(state,-h*direction)
    ph,pl,pr=port.pose_pair(plus)
    mh,ml,mr=port.pose_pair(minus)
    observed=np.r_[((ph+pl)-(mh+ml))/(2*h),
                   Rotation.from_matrix(pr@mr.T).as_rotvec()/(2*h)]
    np.testing.assert_allclose(observed,port.tangent(state)@direction,
                               rtol=1e-8,atol=1e-10)
    other=np.zeros(context.count)
    other[6:12]=(-.2,.1,.4,-.04,.08,.03)
    second_position,second_frame=port.second_variation(state,direction,other)
    np.testing.assert_array_equal(second_position,np.zeros(3))
    hh=2e-4
    def frame_with(a,b):
        return port.pose_pair(context.advance(state,hh*(a*direction+b*other)))[2]
    observed_second=(frame_with(1,1)-frame_with(1,-1)-
                     frame_with(-1,1)+frame_with(-1,-1))/(4*hh*hh)
    np.testing.assert_allclose(observed_second,second_frame,rtol=1e-5,atol=1e-8)
    wrench=np.array((3.,-5.,7.,11.,-13.,17.))
    assert direction@port.pullback(state,wrench)==pytest.approx(
        wrench@(port.tangent(state)@direction),rel=1e-14)
    before=state.nodal_frames[1].copy()
    ph[:]=0.;pl[:]=0.;pr[:]=0.
    np.testing.assert_array_equal(state.nodal_frames[1],before)
    assert np.linalg.norm(port.pose_pair(state)[2])>0.
    with pytest.raises(ValueError,match="valid native physical node"):
        NativeNodalSpatialPort(context,True)


def test_offset_native_reference_preserves_local_operator_parity():
    fixture=build(16,z_offset=.1)
    high,low,frame=NativeNodalSpatialPort(
        fixture.context,8).pose_pair(fixture.context.initial.mechanical)
    np.testing.assert_allclose(high+low,(.5,0.,.1),rtol=0,atol=1e-14)
    np.testing.assert_allclose(frame,np.eye(3),rtol=0,atol=1e-14)
    assert fixture.dense_oracle_max_error<1e-9
    with pytest.raises(ValueError,match="reference offset"):
        build(4,z_offset=float("nan"))
