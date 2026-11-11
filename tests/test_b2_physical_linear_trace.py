"""Independent bounded checks of the private F09-T1 physical station port."""

import numpy as np
import pytest

from anyfesolver._b2_physical_flexibility_linear import (
    LinearOperator, LinearSection, RepresentabilityError,
)
from anyfesolver._b2_physical_linear_definition import PhysicalB2Member


STATIONS = (0., 1/8, 1/3, .5, 7/8, 1.)
OFFSET = np.array([.1, .2, -.15])
GENERAL_S = np.diag([10., 7., 9., 3., 2., 4.])
for _i, _j, _value in ((0, 1, .6), (0, 3, .2), (2, 4, .3), (1, 5, -.25)):
    GENERAL_S[_i, _j] = GENERAL_S[_j, _i] = _value


def _scalar():
    return LinearSection.isotropic(E=200e9, nu=.3, area=.01, Iy=8e-6,
                                   Iz=5e-6, J=1e-5, ky=.7, kz=.8)


def _operator(section=None):
    return LinearOperator(2., _scalar() if section is None else section)


def _member(p0=(1., 2., -3.), p1=(3., 2., -3.), y=(0., 1., 0.)):
    return PhysicalB2Member("B2-A", "geometry/B2-A", (10, 11),
                            (p0, p1), y, "section-A", _scalar())


def _close(actual, expected, scale, relative=1e-10, floor=1e-11):
    expected = np.asarray(expected)
    np.testing.assert_array_less(np.abs(actual-expected),
                                 np.maximum(relative*np.maximum(np.abs(expected),
                                                                1e-12*scale),
                                            floor*scale) + 1e-16*scale)


def _bvp_pose(section, nodal, station, offset=(0., 0., 0.), return_initial=False):
    """Independent initial-resultant boundary-value solve; no B/H or trace imports."""
    length = 2.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    compliance = np.linalg.inv(np.asarray(section.stiffness))

    def integrate(initial, x):
        places = x*(nodes+1)/2
        resultants = np.tile(initial, (len(nodes), 1))
        resultants[:, 4] += initial[2]*places
        resultants[:, 5] -= initial[1]*places
        strains = resultants@compliance.T
        integral = x/2 * (weights@strains)
        weighted = x/2 * ((weights*(x-places))@strains)
        pose = np.empty(6)
        pose[0] = nodal[0]+integral[0]
        pose[1] = nodal[1]+x*nodal[5]+integral[1]+weighted[5]
        pose[2] = nodal[2]-x*nodal[4]+integral[2]-weighted[4]
        pose[3:] = nodal[3:6]+integral[3:]
        return pose

    base = integrate(np.zeros(6), length)
    columns = np.column_stack([integrate(np.eye(6)[i], length)-base
                               for i in range(6)])
    initial = np.linalg.solve(columns, nodal[6:]-base)
    pose = integrate(initial, length*station)
    pose[:3] += np.cross(pose[3:], offset)
    return (pose, initial) if return_initial else pose


@pytest.mark.parametrize("section", [_scalar(), LinearSection.generalized(GENERAL_S)],
                         ids=["scalar", "generalized"])
def test_rigid_modes_endpoints_and_independent_boundary_value(section):
    op = _operator(section)
    for row, columns in enumerate(op._trace_polynomials):
        for column, polynomial in enumerate(columns):
            assert polynomial[0] == int(column == row)
            assert sum(polynomial) == int(column == row+6)
    state = np.array([.009, -.004, .002, .005, -.006, .003,
                      -.004, .008, .001, -.002, .004, -.005])
    trace = op.trace(state, stations=STATIONS)
    assert not op.trace(np.zeros(12), stations=STATIONS).pose.any()
    _close(trace.pose[0], state[:6], .05)
    _close(trace.pose[-1], state[6:], .05)
    if section.kind == "elastic_generalized":
        for i, t in enumerate(STATIONS):
            _close(trace.pose[i], _bvp_pose(section, state, t), .05)
    for origin in (np.zeros(3), np.array([.1, -.2, .3])):
        for direction in np.eye(6):
            a, omega = direction[:3]*.01, direction[3:]*.01
            rigid = np.r_[a+np.cross(omega, origin), omega,
                          a+np.cross(omega, origin+np.array([2., 0., 0.])), omega]
            result = op.trace(rigid, stations=STATIONS, offset_local_m=OFFSET)
            for i, t in enumerate(STATIONS):
                point = origin+np.array([2*t, 0., 0.])+OFFSET
                _close(result.pose[i], np.r_[a+np.cross(omega, point), omega], .05)


def test_scalar_tip_force_moment_torque_and_shear_patch():
    op = _operator()
    tangent = op.evaluate(np.zeros(12)).tangent
    section = op.section
    E, nu, _, Iy, Iz, J, ky, kz = section.scalar_parameters
    G = E/(2*(1+nu))
    x = np.array(STATIONS)*2
    for load_index, moment_index, EI, GA, sign in (
        (7, 5, E*Iz, G*.01*ky, 1), (8, 4, E*Iy, G*.01*kz, -1),
    ):
        for signed_p in (1000., -1000.):
            load = np.zeros(6)
            load[load_index-6] = signed_p
            u = np.r_[np.zeros(6), np.linalg.solve(tangent[6:, 6:], load)]
            trace = op.trace(u, stations=STATIONS)
            expected_displacement = signed_p*x*x*(6-x)/(6*EI) + signed_p*x/GA
            expected_rotation = signed_p*x*(4-x)/(2*EI)*sign
            _close(trace.pose[:, load_index-6], expected_displacement, .01)
            _close(trace.pose[:, moment_index], expected_rotation, .01)
    for index, stiffness in ((9, G*J), (10, E*Iy), (11, E*Iz)):
        for signed_m in (200., -200.):
            load = np.zeros(6)
            load[index-6] = signed_m
            u = np.r_[np.zeros(6), np.linalg.solve(tangent[6:, 6:], load)]
            trace = op.trace(u, stations=STATIONS)
            _close(trace.pose[:, index-6], signed_m*x/stiffness, .01)
            if index == 10:
                _close(trace.pose[:, 2], -signed_m*x*x/(2*E*Iy), .01)
            elif index == 11:
                _close(trace.pose[:, 1], signed_m*x*x/(2*E*Iz), .01)
    for signed_p in (1000., -1000.):
        load = np.zeros(6)
        load[0] = signed_p
        u = np.r_[np.zeros(6), np.linalg.solve(tangent[6:, 6:], load)]
        _close(op.trace(u, stations=STATIONS).pose[:, 0], signed_p*x/(E*.01), .01)
    soft = LinearSection.isotropic(E=E, nu=nu, area=.01, Iy=Iy, Iz=Iz,
                                   J=J, ky=.35, kz=kz)
    soft_op = _operator(soft)
    load = np.zeros(6)
    load[1] = 1000.
    soft_u = np.r_[np.zeros(6), np.linalg.solve(
        soft_op.evaluate(np.zeros(12)).tangent[6:, 6:], load)]
    soft_v = soft_op.trace(soft_u, stations=STATIONS).pose[:, 1]
    _close(soft_v, 1000*x*x*(6-x)/(6*E*Iz)+1000*x/(G*.01*.35), .01)
    stiff_u = np.r_[np.zeros(6), np.linalg.solve(tangent[6:, 6:], load)]
    assert np.all(soft_v[1:] > op.trace(stiff_u, stations=STATIONS).pose[1:, 1])


def test_generalized_offset_jacobian_and_signed_virtual_work():
    section = LinearSection.generalized(GENERAL_S)
    op = _operator(section)
    u = np.linspace(-.02, .03, 12)
    direction = np.linspace(.05, -.03, 12)
    wrench = np.array([1., -.8, .7, -.5, .2, .3])
    trace = op.trace(u, stations=STATIONS, offset_local_m=OFFSET)
    for signed_offset in (OFFSET, -OFFSET):
        current = op.trace(u, stations=STATIONS, offset_local_m=signed_offset)
        for i, t in enumerate(STATIONS):
            _close(current.pose[i], _bvp_pose(section, u, t, signed_offset), .05)
            for column in range(12):
                unit = .05*np.eye(12)[column]
                oracle = _bvp_pose(section, unit, t, signed_offset)
                predicted = .05*current.jacobian[i, :, column]
                assert np.all(np.abs(predicted-oracle)
                              <= 1e-8*np.maximum(np.abs(oracle), 1e-4*.05))
            for eta in (1e-3, 1e-4, 1e-5):
                plus = op.trace(u+eta*direction, stations=(t,),
                                offset_local_m=signed_offset).pose[0]
                minus = op.trace(u-eta*direction, stations=(t,),
                                 offset_local_m=signed_offset).pose[0]
                finite_difference = (plus-minus)/(2*eta)
                predicted = current.jacobian[i]@direction
                limits = 1e-8*np.maximum(np.abs(predicted), 1e-4*.05)
                assert np.all(np.abs(finite_difference-predicted) <= limits)
                if eta == 1e-3:
                    # The registered work tolerance uses the widest central
                    # step; smaller steps independently test the derivative
                    # plateau but amplify cancellation of binary64 poses.
                    lhs = wrench@finite_difference
                    rhs = direction@current.jacobian[i].T@wrench
                    assert abs(lhs-rhs) <= 1e-10*max(abs(rhs), 1e-4*(.05+.05))
    center = op.trace(u, stations=(.5,)).jacobian[0]
    offset = trace.jacobian[3]
    for col in range(12):
        _close(offset[:3, col]-center[:3, col],
               np.cross(center[3:, col], OFFSET), .05)
    copied = trace.pose.copy()
    trace.pose[:] = 9
    trace.jacobian[:] = 8
    trace.reference_points[:] = 7
    _close(op.trace(u, stations=STATIONS, offset_local_m=OFFSET).pose, copied, .05)

    # Independent section equilibrium and strain kinematics at interior ports.
    compliance = np.linalg.inv(GENERAL_S)
    for t in STATIONS[1:-1]:
        center, initial = _bvp_pose(section, u, t, return_initial=True)
        x = 2*t
        resultant = initial.copy()
        resultant[4] += initial[2]*x
        resultant[5] -= initial[1]*x
        strain = compliance@resultant
        h = 1e-5
        spatial = (op.trace(u, stations=((x+h)/2,)).pose[0]
                   - op.trace(u, stations=((x-h)/2,)).pose[0])/(2*h)
        expected = np.array([strain[0], strain[1]+center[5],
                             strain[2]-center[4], *strain[3:]])
        _close(spatial, expected, .05, relative=1e-8, floor=1e-9)


def test_global_rotation_reversal_and_adjoint_port():
    forward = _member()
    reverse = _member(p0=(3., 2., -3.), p1=(1., 2., -3.))
    global_u = np.array([.009, -.004, .002, .005, -.006, .003,
                         -.004, .008, .001, -.002, .004, -.005])
    swap = np.r_[np.arange(6, 12), np.arange(6)]
    a = forward.trace_global(global_u, stations=STATIONS, offset_local_m=OFFSET)
    b = reverse.trace_global(global_u[swap], stations=tuple(1-t for t in STATIONS),
                             offset_local_m=OFFSET*np.array([-1., 1., -1.]))
    _close(a.reference_points_m, b.reference_points_m, 2.)
    _close(a.pose, b.pose, .01)
    _close(a.jacobian, b.jacobian[:, :, swap], .01)
    wrench = np.array([1000., -500., 700., 200., -100., 150.])
    direction = np.linspace(-.01, .01, 12)
    for i in range(len(STATIONS)):
        lhs = wrench@a.jacobian[i]@direction
        rhs = direction@a.jacobian[i].T@wrench
        assert abs(lhs-rhs) <= 1e-10*max(abs(rhs), 1e-4*(1000*.01+200*.01))
        for eta in (1e-3, 1e-4, 1e-5):
            plus = forward.trace_global(global_u+eta*direction, stations=(STATIONS[i],),
                                        offset_local_m=OFFSET).pose[0]
            minus = forward.trace_global(global_u-eta*direction, stations=(STATIONS[i],),
                                         offset_local_m=OFFSET).pose[0]
            finite_difference = (plus-minus)/(2*eta)
            predicted = a.jacobian[i]@direction
            assert np.all(np.abs(finite_difference-predicted)
                          <= 1e-8*np.maximum(np.abs(predicted), 1e-4*.01))
            if eta == 1e-3:
                assert (abs(wrench@finite_difference-rhs)
                        <= 1e-10*max(abs(rhs), 1e-4*(1000*.01+200*.01)))
    for radians in (np.pi/2, .37):
        c, s = np.cos(radians), np.sin(radians)
        angle = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        rotated = _member(p0=angle@np.array(forward.reference_points_m[0]),
                          p1=angle@np.array(forward.reference_points_m[1]),
                          y=angle@np.array(forward.local_y_global))
        global_transform = np.kron(np.eye(4), angle)
        rotated_trace = rotated.trace_global(global_transform@global_u,
                                             stations=STATIONS, offset_local_m=OFFSET)
        six = np.kron(np.eye(2), angle)
        _close(rotated_trace.reference_points_m, a.reference_points_m@angle.T, 2.)
        _close(rotated_trace.pose, a.pose@six.T, .01)
        _close(rotated_trace.jacobian,
               np.einsum("ab,nbc,cd->nad", six, a.jacobian, global_transform.T), .01)
        rotated_u = global_transform@global_u
        rotated_direction = global_transform@direction
        for i, t in enumerate(STATIONS):
            eta = 1e-3
            plus = rotated.trace_global(rotated_u+eta*rotated_direction,
                                        stations=(t,), offset_local_m=OFFSET).pose[0]
            minus = rotated.trace_global(rotated_u-eta*rotated_direction,
                                         stations=(t,), offset_local_m=OFFSET).pose[0]
            rotated_wrench = six@wrench
            work = rotated_wrench@(plus-minus)/(2*eta)
            adjoint = rotated_direction@rotated_trace.jacobian[i].T@rotated_wrench
            assert abs(work-adjoint) <= 1e-10*max(abs(adjoint),
                                                  1e-4*(1000*.01+200*.01))


def test_invalid_station_offset_and_state_rejected():
    op = _operator()
    for stations in ((-.01,), (1.01,), (float("nan"),), 0.5):
        with pytest.raises(ValueError, match="stations"):
            op.trace(np.zeros(12), stations=stations)
    for offset in ((1., 2.), (0., float("inf"), 0.)):
        with pytest.raises(ValueError, match="offset"):
            op.trace(np.zeros(12), offset_local_m=offset)
    with pytest.raises(ValueError, match="twelve"):
        op.trace(np.full(12, np.nan))
    with pytest.raises(RepresentabilityError):
        state = np.zeros(12)
        state[3] = 1e308
        op.trace(state, stations=(0.,), offset_local_m=(0., 1e308, 0.))
