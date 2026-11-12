"""Independent local finite-polar candidate controls, no structural solves."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_mixed_ad import rotation_exponential
from scripts import r8_e01_objective_polar_trace as trace
from scripts.r8_e01_material_frame_work import rectangular_patch, shell_material_frame_map


XY = np.array(((0., -.5), (2., -.5), (2., .5), (0., .5)))
STATION = np.array((.8, -.2))
LAM = np.array((3., -2., 1., .4, -.6, .8))


def candidate(sign=1, shell_offset=(0., 0., 0.)):
    return trace.PolarTrace(XY, STATION, np.r_[STATION, sign*.1],
                            np.eye(3), shell_offset)


def state(case):
    ref = case.reference_state()
    delta = np.zeros(30)
    node = delta[:24].reshape(4, 6)
    node[:, :3] = np.column_stack((.04*XY[:, 0]+.02*XY[:, 1],
                                  -.01*XY[:, 0]+.03*XY[:, 1], .06*XY[:, 0]*XY[:, 1]))
    node[:, 3:] = ((.03, -.05, .01), (.07, -.02, -.03),
                   (.04, -.06, .02), (.08, -.01, .04))
    delta[24:] = (.02, -.01, .03, .09, -.04, .07)
    return trace.updated(ref, delta)


def oracle(case, physical):
    """Separate NumPy symmetric eigensystem and SciPy principal-log oracle."""
    n, dx, dy = rectangular_patch(case.xy, case.station)
    director = n @ physical.directors
    director /= np.linalg.norm(director)
    f = (physical.positions-physical.positions[0]).T @ np.column_stack((dx, dy))
    e = (np.eye(3)-np.outer(director, director)) @ f
    values, vectors = np.linalg.eigh(e.T @ e)
    q = e @ (vectors @ np.diag(values**-.5) @ vectors.T)
    rs = np.column_stack((q, director))
    center = physical.positions[0]+n @ (physical.positions-physical.positions[0])
    port = center+rs @ case.shell_offset
    ct = rs.T @ (physical.beam_position-port)-case.e0
    cr = Rotation.from_matrix(case.reference_beam_rotation.T @ rs.T @ physical.beam_rotation).as_rotvec()
    return np.r_[ct, cr], rs


@pytest.mark.parametrize('sign', (-1, 1))
def test_reference_consistency_and_v1_linearization(sign):
    case = candidate(sign)
    result = trace.evaluate(case, case.reference_state(), LAM)
    np.testing.assert_allclose(result.constraint, 0., rtol=0., atol=1e-14)
    # Native reference eccentricity is applied once. This is compatibility
    # linearization, rather than the derivative of the shell-frame offset alone.
    linear = shell_material_frame_map(XY, STATION, (0., 0., sign*.1))
    np.testing.assert_allclose(result.jacobian[:, :24], -linear, rtol=0., atol=1e-14)
    np.testing.assert_allclose(result.jacobian[:, 24:], np.eye(6), rtol=0., atol=1e-14)
    np.testing.assert_allclose(result.shell_frame, np.eye(3), rtol=0., atol=1e-14)


def test_analytic_square_root_against_independent_eigensystem():
    case = candidate(shell_offset=(.02, -.01, .03))
    physical = state(case)
    result = trace.evaluate(case, physical, LAM)
    value, frame = oracle(case, physical)
    np.testing.assert_allclose(result.constraint, value, rtol=0., atol=2e-14)
    np.testing.assert_allclose(result.shell_frame, frame, rtol=0., atol=2e-14)
    assert np.linalg.det(result.shell_frame) == pytest.approx(1., abs=2e-14)
    np.testing.assert_allclose(result.shell_frame[:, 2], result.director, rtol=0., atol=1e-14)


def test_large_port_gauge_preserves_actual_constraint_value_and_derivatives():
    ordinary = candidate()
    huge = candidate(shell_offset=(2.**54, -2.**54, 2.**54))
    physical = ordinary.reference_state()
    physical = replace(physical, beam_position=physical.beam_position+(1., .5, -.25))
    baseline = trace.evaluate(ordinary, physical, LAM)
    result = trace.evaluate(huge, physical, LAM)
    np.testing.assert_allclose(result.constraint[:3], (1., .5, -.25), rtol=0., atol=1e-14)
    np.testing.assert_array_equal(result.constraint, baseline.constraint)
    np.testing.assert_array_equal(result.jacobian, baseline.jacobian)
    np.testing.assert_array_equal(result.hessian, baseline.hessian)
    assert result.jacobian[0, 24] == pytest.approx(1., abs=1e-14)
    # The same gauge invariance must hold with a genuinely finite recovered frame.
    physical = state(ordinary)
    baseline = trace.evaluate(ordinary, physical, LAM)
    result = trace.evaluate(huge, physical, LAM)
    np.testing.assert_array_equal(result.constraint, baseline.constraint)
    np.testing.assert_array_equal(result.covector, baseline.covector)
    np.testing.assert_array_equal(result.hessian, baseline.hessian)


def test_nonidentity_reference_beam_alignment_and_finite_common_motion():
    alignment = rotation_exponential((.23, -.17, .19))
    case = replace(candidate(), reference_beam_rotation=alignment)
    reference = case.reference_state()
    result = trace.evaluate(case, reference, LAM)
    np.testing.assert_allclose(result.constraint, 0., rtol=0., atol=1e-14)
    linear = shell_material_frame_map(XY, STATION, (0., 0., .1))
    np.testing.assert_allclose(result.jacobian[3:, :24], -alignment.T @ linear[3:],
                               rtol=0., atol=1e-14)
    np.testing.assert_allclose(result.jacobian[3:, 27:], alignment.T, rtol=0., atol=1e-14)
    physical = state(case)
    result = trace.evaluate(case, physical, LAM)
    value, frame = oracle(case, physical)
    np.testing.assert_allclose(result.constraint, value, rtol=0., atol=2e-14)
    np.testing.assert_allclose(result.shell_frame, frame, rtol=0., atol=2e-14)
    r = rotation_exponential((.7, -.4, .3))
    repeated = trace.evaluate(case, trace.superposed(physical, r, (2., -3., 1.)), LAM)
    transform = np.kron(np.eye(10), r)
    np.testing.assert_allclose(repeated.constraint, result.constraint, rtol=0., atol=2e-13)
    np.testing.assert_allclose(repeated.covector, transform @ result.covector, rtol=0., atol=2e-12)
    np.testing.assert_allclose(repeated.hessian, transform @ result.hessian @ transform.T,
                               rtol=0., atol=2e-11)


def test_common_finite_rotation_translation_and_coordinate_covariance():
    case = candidate(); physical = state(case)
    r = rotation_exponential((.7, -.4, .3))
    result = trace.evaluate(case, physical, LAM)
    moved = trace.superposed(physical, r, (13., -21., 8.))
    repeated = trace.evaluate(case, moved, LAM)
    np.testing.assert_allclose(repeated.constraint, result.constraint, rtol=0., atol=2e-13)
    np.testing.assert_allclose(repeated.shell_frame, r @ result.shell_frame, rtol=0., atol=2e-13)
    transform = np.kron(np.eye(10), r)
    np.testing.assert_allclose(repeated.covector, transform @ result.covector, rtol=0., atol=2e-12)
    np.testing.assert_allclose(repeated.hessian, transform @ result.hessian @ transform.T,
                               rtol=0., atol=2e-11)


def test_finite_symmetric_shear_and_independent_director():
    case = candidate(); ref = case.reference_state()
    x = ref.positions.copy(); x[:, 0] += .2*XY[:, 1]; x[:, 1] += .2*XY[:, 0]
    # Symmetric positive in-plane stretch has identity polar rotation; a
    # projected line tangent has yaw atan(.2), hence defines a different trace.
    result = trace.evaluate(case, replace(ref, positions=x), LAM)
    np.testing.assert_allclose(result.shell_frame, np.eye(3), atol=1e-14)
    assert np.arctan2(.2, 1.) > .19
    x = ref.positions.copy(); x[:, 2] = .03*XY[:, 0]
    d = np.tile(rotation_exponential((0., .12, 0.)) @ (0., 0., 1.), (4, 1))
    result = trace.evaluate(case, replace(ref, positions=x, directors=d), LAM)
    normal = np.array((-.03, 0., 1.)); normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(result.director, d[0], atol=1e-14)
    assert np.linalg.norm(result.director-normal) > .14


@pytest.mark.parametrize('sign', (-1, 1))
def test_physical_wrench_balance_and_origin_shift(sign):
    case = candidate(sign, (.02, -.01, .03)); physical = state(case)
    result = trace.evaluate(case, physical, LAM)
    for origin in ((0., 0., 0.), (11., -7., 3.)):
        force, moment = trace.wrench_balance(physical, result, origin)
        np.testing.assert_allclose(force, 0., rtol=0., atol=1e-12)
        np.testing.assert_allclose(moment, 0., rtol=0., atol=1e-11)
    # Virtual work of the physical covector against an arbitrary local motion.
    direction = np.sin(np.arange(30)+.2)
    assert result.covector @ direction == pytest.approx(LAM @ (result.jacobian @ direction), abs=1e-12)


def test_normal_moment_is_translation_couple_not_nodal_yaw_penalty():
    case = candidate(); ref = case.reference_state()
    result = trace.evaluate(case, ref, (0., 0., 0., 0., 0., 1.))
    node = result.covector[:24].reshape(4, 6)
    np.testing.assert_array_equal(node[:, 5], 0.)
    np.testing.assert_allclose(node[:, :3].sum(axis=0), 0., atol=1e-14)
    torque = np.cross(ref.positions, node[:, :3]).sum(axis=0)
    np.testing.assert_allclose(torque, (0., 0., -1.), atol=1e-14)
    assert result.covector[29] == pytest.approx(1., abs=1e-14)


def test_actual_update_derivative_plateau_and_full_multiplier_hessian():
    case = candidate(); physical = state(case)
    result = trace.evaluate(case, physical, LAM)
    v = np.cos(np.arange(30)+.1); v /= np.linalg.norm(v)
    expected_j = result.jacobian @ v
    expected_h = result.hessian @ v
    first_errors, second_errors = [], []
    for h in (2e-4, 1e-4, 5e-5):
        plus = trace.evaluate(case, physical, LAM, increments=h*v)
        minus = trace.evaluate(case, physical, LAM, increments=-h*v)
        # Independent physical update route agrees with anchored-map values.
        actual = trace.evaluate(case, trace.updated(physical, h*v), LAM)
        np.testing.assert_allclose(actual.constraint, plus.constraint, rtol=0., atol=2e-14)
        j = (plus.constraint-minus.constraint)/(2*h)
        hv = (plus.covector-minus.covector)/(2*h)
        first_errors.append(np.linalg.norm(j-expected_j)/np.linalg.norm(expected_j))
        second_errors.append(np.linalg.norm(hv-expected_h)/np.linalg.norm(expected_h))
    assert all(error <= 1e-6 for error in first_errors), first_errors
    assert all(error <= 1e-6 for error in second_errors), second_errors
    skew = np.linalg.norm(result.hessian-result.hessian.T)/np.linalg.norm(result.hessian)
    assert skew <= 1e-9
    np.testing.assert_allclose(result.hessian,
                               np.einsum('i,ijk->jk', LAM, result.constraint_hessians),
                               rtol=0., atol=1e-14)
    zero = trace.evaluate(case, physical, np.zeros(6))
    np.testing.assert_array_equal(zero.hessian, 0.)
    np.testing.assert_array_equal(zero.covector, 0.)


def test_nonzero_chart_gradient_is_not_relabelled_as_physical_wrench():
    case = candidate(); physical = state(case)
    result = trace.evaluate(case, physical, LAM, increments=np.full(30, .01))
    with pytest.raises(trace.TraceDomainError, match='recenter'):
        trace.wrench_balance(physical, result)


@pytest.mark.parametrize('kind', ('collapse', 'flip', 'director-cancellation', 'metric-unresolved', 'log-pole'))
def test_explicit_domain_rejections(kind):
    case = candidate(); ref = case.reference_state()
    x = ref.positions.copy(); d = ref.directors.copy(); rb = ref.beam_rotation.copy()
    if kind == 'collapse': x[:, 0] = 0.
    elif kind == 'flip': x[:, 0] *= -1.
    elif kind == 'metric-unresolved': x[:, 0] *= 1e-9
    elif kind == 'log-pole': rb = rotation_exponential((0., 0., np.pi))
    else:
        case = replace(case, station=(1., 0.))
        d[2:] *= -1.
    bad = trace.PhysicalState(x, d, ref.beam_position, rb)
    with pytest.raises(trace.TraceDomainError):
        trace.evaluate(case, bad, LAM)


def test_input_ownership_nonfinite_and_no_mutation_on_failure():
    x = np.column_stack((XY, np.zeros(4)))
    directors = np.tile((0., 0., 1.), (4, 1))
    physical = trace.PhysicalState(x, directors, np.r_[STATION, .1], np.eye(3))
    x[:] = np.nan; directors[:] = 0.
    assert np.isfinite(physical.positions).all()
    original = physical.positions.copy()
    result = trace.evaluate(candidate(), physical, LAM)
    assert not result.covector.flags.writeable
    assert not physical.positions.flags.writeable
    with pytest.raises(trace.TraceDomainError):
        trace.updated(physical, np.full(30, np.nan))
    with pytest.raises(trace.TraceDomainError):
        replace(physical, beam_rotation=-np.eye(3))
    np.testing.assert_array_equal(physical.positions, original)
