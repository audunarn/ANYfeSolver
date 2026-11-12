"""Bounded native gravity mechanics checks, not qualification evidence."""

import numpy as np
import pytest

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_p5.algebra import rotation, skew
from anyfesolver._stiffened_shell_native_gravity import compile_native_gravity


def reference(curved=False):
    nodes = np.array([[-1., 0., 0.], [0., .25 if curved else 0., 0.], [1., 0., 0.]])
    frames = []
    for xi in (-1., 0., 1.):
        tangent = np.array([1., -.5*xi if curved else 0., 0.])
        tangent /= np.linalg.norm(tangent)
        second = np.array([0., 0., 1.])
        frames.append(np.column_stack((tangent, second, np.cross(tangent, second))))
    return Reference(nodes, frames)


def inertia(mass=2.5, offset=(0., 0., 0.)):
    offset = np.array(offset)
    result = np.zeros((6, 6))
    result[:3, :3] = mass*np.eye(3)
    result[:3, 3:] = -skew(mass*offset)
    result[3:, :3] = result[:3, 3:].T
    result[3:, 3:] = np.diag([.4, .5, .6])+mass*((offset@offset)*np.eye(3)-np.outer(offset, offset))
    return result


def state(ref):
    x = ref.coordinates+np.array([[.01, .02, -.03], [-.02, .01, .03], [.04, -.03, .02]])
    u = np.array([rotation([.15, -.12, .08]), rotation([-.07, .11, .19])])
    return x, np.zeros((3, 3)), u, np.array([1.2, -9.81, 2.3])


@pytest.mark.parametrize('order', [4, 8, 24])
def test_straight_centered_analytic_mass_force_and_zero_unused_slots(order):
    ref = reference()
    compiled = compile_native_gravity(ref, inertia(), order)
    np.testing.assert_allclose(compiled.nodal_masses, [1.25, 2.5, 1.25], rtol=2e-15)
    assert not np.any(compiled.cell_first_moments)
    x, low, u, g = state(ref)
    work = compiled.evaluate(x, low, u, g)
    expected = np.zeros(42)
    for i, mass in enumerate([1.25, 2.5, 1.25]):
        expected[6*i:6*i+3] = mass*g
    np.testing.assert_allclose(work.gradient, expected, rtol=2e-15)
    np.testing.assert_allclose(work.total_force, 5*g, rtol=2e-15)
    np.testing.assert_allclose(work.value, sum(np.dot(expected[6*i:6*i+3], x[i]-ref.coordinates[i])
                                            for i in range(3)), rtol=3e-15)
    assert not np.any(work.hessian)
    assert not np.any(work.spatial_jacobian)
    np.testing.assert_allclose(work.total_moment, sum(np.cross(x[i], expected[6*i:6*i+3])
                                                    for i in range(3)), atol=1e-14)


def test_straight_eccentric_first_moment_actual_left_update_and_chart_hessian():
    ref = reference()
    offset = np.array([.1, .2, -.15])
    compiled = compile_native_gravity(ref, inertia(offset=offset))
    expected_b = ref.frame(0.)@(2.5*offset)
    np.testing.assert_allclose(compiled.cell_first_moments, np.tile(expected_b, (2, 1)), atol=3e-16)
    x, low, u, g = state(ref)
    work = compiled.evaluate(x, low, u, g)
    step = 2e-5
    for cell in (0, 1):
        sl = slice(18+3*cell, 21+3*cell)
        for axis in range(3):
            delta = np.eye(3)[axis]*step
            plus, minus = u.copy(), u.copy()
            plus[cell] = rotation(delta)@u[cell]
            minus[cell] = rotation(-delta)@u[cell]
            wp = compiled.evaluate(x, low, plus, g)
            wm = compiled.evaluate(x, low, minus, g)
            np.testing.assert_allclose((wp.value-wm.value)/(2*step), work.gradient[18+3*cell+axis],
                                       rtol=2e-9, atol=2e-10)
            np.testing.assert_allclose((wp.gradient[sl]-wm.gradient[sl])/(2*step),
                                       work.spatial_jacobian[sl, 18+3*cell+axis], rtol=2e-9, atol=2e-10)
        direction = np.array([.7, -.3, .4])
        plus, minus = u.copy(), u.copy()
        plus[cell] = rotation(step*direction)@u[cell]
        minus[cell] = rotation(-step*direction)@u[cell]
        second = (compiled.evaluate(x, low, plus, g).value-2*work.value
                  +compiled.evaluate(x, low, minus, g).value)/step**2
        np.testing.assert_allclose(second, direction@work.hessian[sl, sl]@direction, atol=7e-6)
    np.testing.assert_allclose(work.hessian, .5*(work.spatial_jacobian+work.spatial_jacobian.T), atol=1e-15)
    assert not np.any(work.gradient[24:])
    assert not np.any(work.hessian[24:])
    for sl in (slice(3, 6), slice(9, 12), slice(15, 18)):
        assert not np.any(work.gradient[sl])


@pytest.mark.parametrize('order', [4, 8])
def test_curved_centered_parity_with_independent_line_work_ad(order):
    from anyfesolver._ge_beam3_fibre_line_work import evaluate

    ref = reference(curved=True)
    x, low, u, g = state(ref)
    actual = compile_native_gravity(ref, inertia(), order).evaluate(x, low, u, g)
    expected = evaluate(ref, x, low, u, 2.5*g, order=order)
    np.testing.assert_allclose(actual.value, expected.value, rtol=4e-14, atol=1e-14)
    np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=4e-14, atol=1e-14)
    np.testing.assert_allclose(actual.hessian, expected.hessian, rtol=4e-14, atol=1e-14)


def test_order24_rigid_acceleration_matches_same_station_current_rest_mass():
    from anyfesolver._ge_beam3_p5_centered.mass import current_rest_mass

    ref = reference(curved=True)
    tensor = inertia(offset=(.1, .2, -.15))
    x, low, u, g = state(ref)
    compiled = compile_native_gravity(ref, tensor)
    mass = current_rest_mass(ref, tensor, 24, x, low, u)
    acceleration = np.zeros(24)
    for node in range(3):
        acceleration[6*node:6*node+3] = g
    np.testing.assert_allclose(compiled.evaluate(x, low, u, g).gradient[:24], mass@acceleration,
                               rtol=5e-14, atol=2e-14)


def test_covariance_and_origin_moment_shift():
    ref = reference(curved=True)
    tensor = inertia(offset=(.1, .2, -.15))
    x, low, u, g = state(ref)
    base = compile_native_gravity(ref, tensor).evaluate(x, low, u, g)
    q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = np.array([4., -2., 1.])
    moved_ref = Reference(ref.coordinates@q.T+shift, np.einsum('ij,njk->nik', q, ref.nodal_triads))
    moved_u = np.array([q@v@q.T for v in u])
    moved = compile_native_gravity(moved_ref, tensor).evaluate(x@q.T+shift, low@q.T, moved_u, q@g)
    transform = np.eye(42)
    for start in (0, 6, 12, 18, 21):
        transform[start:start+3, start:start+3] = q
    np.testing.assert_allclose(moved.value, base.value, atol=2e-14)
    np.testing.assert_allclose(moved.gradient, transform@base.gradient, atol=2e-14)
    np.testing.assert_allclose(moved.hessian, transform@base.hessian@transform.T, atol=2e-14)
    np.testing.assert_allclose(moved.spatial_jacobian, transform@base.spatial_jacobian@transform.T, atol=2e-14)
    np.testing.assert_allclose(moved.total_force, q@base.total_force, atol=2e-14)
    np.testing.assert_allclose(moved.total_moment, q@base.total_moment+np.cross(shift, q@base.total_force), atol=5e-14)


def test_equivalent_length_units_preserve_mass_and_scale_work():
    ref = reference(curved=True)
    tensor = inertia(offset=(.1, .2, -.15))
    x, low, u, g = state(ref)
    compiled = compile_native_gravity(ref, tensor)
    base = compiled.evaluate(x, low, u, g)
    scale = 1000.
    scaled_tensor = tensor.copy()
    scaled_tensor[:3, :3] /= scale
    scaled_tensor[3:, 3:] *= scale
    scaled_ref = Reference(ref.coordinates*scale, ref.nodal_triads)
    scaled = compile_native_gravity(scaled_ref, scaled_tensor)
    actual = scaled.evaluate(x*scale, low*scale, u, g*scale)
    np.testing.assert_allclose(scaled.nodal_masses, compiled.nodal_masses, rtol=3e-15)
    np.testing.assert_allclose(actual.value, base.value*scale**2, rtol=2e-14)
    np.testing.assert_allclose(actual.total_force, base.total_force*scale, rtol=3e-15)
    np.testing.assert_allclose(actual.total_moment, base.total_moment*scale**2, rtol=2e-14)


def test_compensated_displacement_zero_gravity_and_owned_arrays():
    ref = reference()
    shift = 2.**40
    shifted = Reference(ref.coordinates+shift, ref.nodal_triads)
    tensor = inertia()
    compiled = compile_native_gravity(shifted, tensor)
    tensor[:] = 0.
    low = np.full((3, 3), 2.**-20)
    u = np.tile(np.eye(3), (2, 1, 1))
    g = np.array([1., 2., 3.])
    work = compiled.evaluate(shifted.coordinates, low, u, g)
    np.testing.assert_allclose(work.value, 5*2.**-20*sum(g), rtol=2e-15)
    zero = compiled.evaluate(shifted.coordinates, low, u, np.zeros(3))
    assert zero.value == 0.
    assert not np.any(zero.gradient)
    assert not np.any(zero.hessian)
    assert not np.any(zero.spatial_jacobian)
    assert not np.any(zero.total_force)
    assert not np.any(zero.total_moment)
    for array in (compiled.coordinates, compiled.inertia, compiled.nodal_masses,
                  compiled.cell_first_moments, work.gradient, work.hessian,
                  work.spatial_jacobian, work.total_force, work.total_moment):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert not np.shares_memory(compiled.coordinates, shifted.coordinates)
    assert not np.shares_memory(compiled.inertia, tensor)


@pytest.mark.parametrize('kind', ['anisotropic_mass', 'non_skew', 'impossible_second_moment'])
def test_spd_alone_does_not_admit_nonphysical_inertia(kind):
    tensor = inertia()
    if kind == 'anisotropic_mass':
        tensor[0, 0] *= 1.2
    elif kind == 'non_skew':
        tensor[0, 3] = tensor[3, 0] = .01
    else:
        tensor[3:, 3:] = np.diag([10., 1., 1.])
    assert np.linalg.eigvalsh(tensor)[0] > 0.
    with pytest.raises(ValueError, match='physical spatial inertia'):
        compile_native_gravity(reference(), tensor)


def test_invalid_inputs_fail_closed():
    ref = reference()
    for order in (True, 0, 6, 24.):
        with pytest.raises(ValueError):
            compile_native_gravity(ref, inertia(), order)
    for tensor in (np.full((6, 6), np.nan), inertia().astype(complex), np.eye(3)):
        with pytest.raises(ValueError):
            compile_native_gravity(ref, tensor)
    compiled = compile_native_gravity(ref, inertia())
    x, low, u, g = state(ref)
    bad_u = u.copy()
    bad_u[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        compiled.evaluate(x, low, bad_u, g)
    with pytest.raises(ValueError):
        compiled.evaluate(x, np.ones((3, 3)), u, g)
    with pytest.raises(ValueError, match='representable'):
        compiled.evaluate(x, low, u, np.array([np.finfo(float).max, 0., 0.]))
    tiny = compile_native_gravity(ref, inertia(mass=.25))
    with pytest.raises(ValueError, match='representable'):
        tiny.evaluate(x, low, u, np.array([np.nextafter(0., 1.), 0., 0.]))


def test_eccentric_torque_underflow_fails_without_losing_representable_nodal_force():
    ref = reference()
    compiled = compile_native_gravity(ref, inertia(offset=(1e-130, 0., 0.)))
    # Nodal forces are ordinary 1e-200 values, but the eccentric torque is
    # genuinely nonzero at 1e-330 and cannot be returned in binary64.
    with pytest.raises(ValueError, match='component is not representable'):
        compiled.evaluate(ref.coordinates, np.zeros((3, 3)), np.tile(np.eye(3), (2, 1, 1)),
                          np.array([0., 1e-200, 0.]))


def test_underflow_boundary_allows_exact_cancellation_and_recovers_subnormal_sum():
    from anyfesolver._stiffened_shell_native_gravity import _resolve_zero_underflow

    smallest = float(np.nextafter(0., 1.))
    assert _resolve_zero_underflow(0., ((smallest, .25), (-smallest, .25))) == 0.
    assert _resolve_zero_underflow(0., ((smallest, .5), (smallest, .5))) == smallest
    with pytest.raises(ValueError, match='component is not representable'):
        _resolve_zero_underflow(0., ((smallest, .25),))
