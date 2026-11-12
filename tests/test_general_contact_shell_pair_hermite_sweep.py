"""Independent analytic checks; sampled values diagnose bounds, never certify."""
import math

import numpy as np
import pytest

from anyfesolver.general_contact.shell_pair_hermite_sweep import shell_pair_hermite_lower


TRIANGLE = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
QUAD = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])


def arguments(target=TRIANGLE, gap=1.):
    target = target.copy()
    target[:, 2] -= gap
    return dict(a0=TRIANGLE.copy(), a1=TRIANGLE.copy(), b0=target.copy(),
                b1=target.copy(), source_v0=np.zeros((3, 3)),
                source_v1=np.zeros((3, 3)), target_v0=np.zeros(target.shape),
                target_v1=np.zeros(target.shape), dt=1.)


def scalar_path(g0, g1, v0, v1, dt=1., target=TRIANGLE):
    data = arguments(target, g0)
    # Both bodies translate: A_z=.4 t, B_z=A_z-g(t).
    data['a1'][:, 2] = .4
    data['b1'][:, 2] = .4-g1
    for name in ('source_v0', 'source_v1'):
        data[name][:, 2] = .4/dt
    data['target_v0'][:, 2] = .4/dt-v0
    data['target_v1'][:, 2] = .4/dt-v1
    data['dt'] = dt
    return data


def analytic_extrema(g0, g1, v0, v1, dt):
    # Independent power-basis derivative-root oracle.
    c = (g0, dt*v0, 3*(g1-g0)-dt*(2*v0+v1),
         2*(g0-g1)+dt*(v0+v1))
    values = [g0, g1]
    aa, bb, cc = 3*c[3], 2*c[2], c[1]
    if aa == 0:
        roots = [] if bb == 0 else [-cc/bb]
    elif bb*bb-4*aa*cc < 0:
        roots = []
    else:
        discriminant = math.sqrt(bb*bb-4*aa*cc)
        roots = [(-bb-discriminant)/(2*aa), (-bb+discriminant)/(2*aa)]
    for t in roots:
        if 0 < t < 1:
            values.append(sum(x*t**k for k, x in enumerate(c)))
    return min(values), max(values)


@pytest.mark.parametrize('target', [TRIANGLE, QUAD])
@pytest.mark.parametrize('g0,g1,v0,v1,dt', [
    (.1, .2, -2., 1., 1.), (1., 2., -.25, .5, 2.),
    (.25, .25, -2., 2., 1.), (-.5, -.25, 1., -.5, .5),
])
def test_complete_cubic_bound_encloses_independent_analytic_minimum(target, g0, g1, v0, v1, dt):
    result = shell_pair_hermite_lower(**scalar_path(g0, g1, v0, v1, dt, target))
    minimum, _ = analytic_extrema(g0, g1, v0, v1, dt)
    assert result.success, result
    assert result.lower_gap <= minimum+1.e-15
    assert result.panels[0][:2] == (0., 1.)


def test_clear_endpoints_hide_cubic_crossing():
    data = scalar_path(.25, .25, -2., 2.)
    minimum, _ = analytic_extrema(.25, .25, -2., 2., 1.)
    assert minimum == -.25
    result = shell_pair_hermite_lower(**data)
    assert result.success and result.lower_gap < -.25
    guarded = shell_pair_hermite_lower(**data, gap_tolerance_m=0., maximum_depth=4)
    assert not guarded.success and guarded.lower_gap is None
    assert guarded.status == 'gap_unresolved'


def test_tolerance_refines_original_polynomials_and_covers_time():
    # g=1-3.6t+3.6t² has minimum .1 although whole-panel cubic controls <0.
    data = scalar_path(1., 1., -3.6, 3.6)
    raw = shell_pair_hermite_lower(**data)
    guarded = shell_pair_hermite_lower(**data, gap_tolerance_m=0.)
    assert raw.success and raw.lower_gap < 0.
    assert guarded.success and guarded.lower_gap > 0.
    assert guarded.visited_panels > raw.visited_panels
    assert guarded.panels[0][0] == 0. and guarded.panels[-1][1] == 1.
    assert all(a[1] == b[0] for a, b in zip(guarded.panels, guarded.panels[1:]))


@pytest.mark.parametrize('gap', [0., 2.**-1022, -2.**-1022])
def test_zero_and_subnormal_finite_bounds(gap):
    result = shell_pair_hermite_lower(**arguments(gap=gap))
    assert result.success and np.isfinite(result.lower_gap)
    assert result.lower_gap <= gap
    if gap == 0.:
        assert shell_pair_hermite_lower(**arguments(gap=gap), gap_tolerance_m=0.).success


def interpolate(first, last, rate0, rate1, dt, t):
    # Direct Hermite basis, independently of production Bernstein controls.
    return ((2*t**3-3*t**2+1)*first + (-2*t**3+3*t**2)*last
            + dt*(t**3-2*t**2+t)*rate0 + dt*(t**3-t**2)*rate1)


@pytest.mark.parametrize('target', [TRIANGLE, QUAD])
@pytest.mark.parametrize('gap', [2., -.25])
def test_nonuniform_varying_normals_all_corner_gaps_enclosed(target, gap):
    data = arguments(target, gap=gap)
    data['source_v0'][1, 2] = .8
    data['source_v1'][2, 2] = -.5
    data['a1'][1, 2] = .2
    data['a1'][2, 2] = -.3
    data['target_v0'][1, 2] = -.6
    data['target_v1'][-1, 2] = .4
    data['b1'][1, 2] += .1
    data['b1'][-1, 2] -= .15
    result = shell_pair_hermite_lower(**data)
    assert result.success, result
    sampled = []
    for t in np.linspace(0., 1., 129):
        a = interpolate(data['a0'], data['a1'], data['source_v0'], data['source_v1'], 1., t)
        b = interpolate(data['b0'], data['b1'], data['target_v0'], data['target_v1'], 1., t)
        normal = np.cross(a[1]-a[0], a[2]-a[0])
        normal /= np.linalg.norm(normal)
        sampled.append(np.min((a[:, None]-b[None, :]) @ normal))
    assert result.lower_gap <= min(sampled)
    if gap > 0.:
        assert result.lower_gap > 1.


@pytest.mark.parametrize('gap', [1., -1.])
def test_varying_normal_against_independent_analytic_norm_oracle(gap):
    data = arguments(gap=gap)
    # Identical tilted parallel triangles: h(t)=t on vertex 1 gives
    # N=(-t,0,1), so every point-pair gap equals g/sqrt(1+t²).
    for key in ('a1', 'b1'):
        data[key][1, 2] += 1.
    for key in ('source_v0', 'source_v1', 'target_v0', 'target_v1'):
        data[key][1, 2] = 1.
    result = shell_pair_hermite_lower(**data)
    oracle = gap/math.sqrt(2.) if gap > 0. else gap
    assert result.success and result.lower_gap <= oracle
    assert result.lower_gap >= oracle-1.e-15


def test_finite_inputs_with_unrepresentable_lower_gap_fail_explicitly():
    data = arguments()
    data['a0'][:, 2] = data['a1'][:, 2] = -1.e308
    data['b0'][:, 2] = data['b1'][:, 2] = 1.e308
    result = shell_pair_hermite_lower(**data)
    assert not result.success and result.lower_gap is None
    assert result.status == 'unrepresentable_bound'


@pytest.mark.parametrize('body', ['source', 'target'])
def test_hidden_source_or_target_inversion_with_clear_endpoints(body):
    data = arguments()
    prefix = body
    data[prefix+'_v0'][2, 1] = -8.
    data[prefix+'_v1'][2, 1] = 8.
    result = shell_pair_hermite_lower(**data)
    assert not result.success and result.lower_gap is None
    assert result.status == 'orientation_invalid'


def test_original_q4_corner_inversion_is_not_hidden_by_triangle_diagonal():
    data = arguments(QUAD)
    # Corner 2 lies inside the triangle (0,1,3); original Q4 corner 2 flips.
    data['b0'][2, :2] = [.2, .2]
    data['b1'] = data['b0'].copy()
    result = shell_pair_hermite_lower(**data)
    assert not result.success and 'corner 2' in result.detail


def test_hidden_original_q4_corner_inversion_between_clear_endpoints():
    data = arguments(QUAD)
    data['target_v0'][2, :2] = -3.2
    data['target_v1'][2, :2] = 3.2
    # Only the original corner-2 Jacobian reverses at the temporal midpoint;
    # corners 1 and 3 retain positive .2 Jacobians there.
    result = shell_pair_hermite_lower(**data)
    assert not result.success and result.status == 'orientation_invalid'
    assert 'corner 2' in result.detail


@pytest.mark.parametrize('body', ['a', 'b'])
def test_initial_degenerate_face_rejected_explicitly(body):
    data = arguments()
    for endpoint in ('0', '1'):
        data[body+endpoint][2] = data[body+endpoint][1]
    result = shell_pair_hermite_lower(**data)
    assert not result.success and result.status == 'orientation_invalid'


def test_initial_opposite_orientation_rejected():
    data = arguments()
    for key in ('b0', 'b1'):
        data[key] = data[key][[0, 2, 1]]
    result = shell_pair_hermite_lower(**data)
    assert not result.success and 'orientation' in result.detail


def test_orientation_subdivision_and_explicit_resource_limits():
    data = arguments()
    data['source_v0'][2, 1] = -3.6
    data['source_v1'][2, 1] = 3.6
    resolved = shell_pair_hermite_lower(**data)
    assert resolved.success and resolved.visited_panels > 1
    limited = shell_pair_hermite_lower(**data, maximum_panels=1)
    assert not limited.success and limited.status == 'panel_budget_exhausted'
    assert limited.visited_panels == 1
    shallow = shell_pair_hermite_lower(**data, maximum_depth=0)
    assert not shallow.success and shallow.status == 'orientation_unresolved'


def test_cancellation_before_setup_and_during_subdivision():
    assert shell_pair_hermite_lower(**arguments(), cancel=lambda: True).visited_panels == 0
    calls = []
    def cancel():
        calls.append(1)
        return len(calls) == 3
    data = scalar_path(1., 1., -3.6, 3.6)
    result = shell_pair_hermite_lower(**data, gap_tolerance_m=0., cancel=cancel)
    assert not result.success and result.status == 'cancelled'
    assert result.visited_panels == 1 and len(calls) == 3


@pytest.mark.parametrize('change', [
    {'dt': True}, {'dt': 1+0j}, {'dt': 0.}, {'dt': np.nan}, {'dt': [1.]},
    {'maximum_panels': True}, {'maximum_panels': 0}, {'maximum_panels': 1.5},
    {'maximum_depth': False}, {'maximum_depth': -1}, {'maximum_depth': 1+0j},
    {'gap_tolerance_m': True}, {'gap_tolerance_m': -1.}, {'gap_tolerance_m': np.inf},
    {'gap_tolerance_m': 0+0j}, {'cancel': True},
    {'a0': np.ones((3, 3), dtype=complex)}, {'a0': np.ones((3, 3), dtype=bool)},
    {'source_v0': np.full((3, 3), np.nan)}, {'target_v1': np.zeros((4, 3))},
    {'b0': np.ones((5, 3))}, {'b0': [[0., 0.], [1.]]},
])
def test_malformed_inputs_and_controls_rejected(change):
    data = arguments()
    data.update(change)
    with pytest.raises(ValueError):
        shell_pair_hermite_lower(**data)


def test_mixed_boolean_coordinate_rejected():
    data = arguments()
    data['a0'] = [[False, 0., 0.], [1., 0., 0.], [0., 1., 0.]]
    with pytest.raises(ValueError):
        shell_pair_hermite_lower(**data)


def test_common_exact_rigid_rotation_and_translation():
    data = scalar_path(.5, .75, -.25, .125, target=QUAD)
    result = shell_pair_hermite_lower(**data)
    rotation = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
    # Signed permutation is a proper rigid rotation and exact in binary.
    assert np.linalg.det(rotation) == 1.
    for key in ('a0', 'a1', 'b0', 'b1'):
        data[key] = data[key] @ rotation.T + np.array([4., -2., 8.])
    for key in ('source_v0', 'source_v1', 'target_v0', 'target_v1'):
        data[key] = data[key] @ rotation.T
    transformed = shell_pair_hermite_lower(**data)
    assert transformed.success and result.success
    assert transformed.lower_gap == result.lower_gap
    assert transformed.visited_panels == result.visited_panels
