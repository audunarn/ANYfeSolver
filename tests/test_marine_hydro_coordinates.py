"""Prospective fixed algebraic cases; selectors keep <=4base poses/process."""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from anyfesolver.marine_rigid import RigidBodyState
from anyfesolver.marine_hydro_coordinates import (
    EquilibriumHydroFrame, MarineHydroCoordinatesError, evaluate_coupled,
    ForcePullback, InertiaPullback, CoupledHydroEvaluation)

FD = 1e-5
TIME_STEP = 1e-3
ALGEBRA = 1e-10
DERIVATIVE = 1e-8
V = np.array((.3, -.2, .4, .5, -.3, .2))
A = np.array((.1, .2, -.4, -.2, .3, .1))


def rotation(v):
    x, y, z = v
    return expm(np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.))))


def frame(body_id='a'):
    return EquilibriumHydroFrame(body_id, (.1, .2, -.3), rotation((.2, -.1, .3)),
                                 rotation((-.1, .3, .4)), (.4, -.2, .3))


def reference_h(frame, state):
    # Independent SciPy exponential/log implementation, no production SO(3).
    R = rotation(state.rotation_vector) @ np.asarray(state.anchor)
    E, R0 = np.asarray(frame.coefficient_axes), np.asarray(frame.equilibrium_rotation)
    rho = np.asarray(frame.reference_point_body)
    return np.r_[E.T @ (np.asarray(state.position)+R@rho-np.asarray(frame.equilibrium_position)-R0@rho),
                 Rotation.from_matrix(E.T@R@R0.T@E).as_rotvec()]


def moved(state, field, delta):
    if field == 'q':
        q = np.r_[state.position, state.rotation_vector]+delta
        return replace(state, position=q[:3], rotation_vector=q[3:])
    return replace(state, **{field: np.asarray(getattr(state, field))+delta})


def independent_rates(frame, state):
    q = np.r_[state.position, state.rotation_vector]
    v, a = np.asarray(state.velocity), np.asarray(state.acceleration)
    values = {}
    for k in (-2, -1, 0, 1, 2):
        t = k*TIME_STEP
        trial = q+t*v+.5*t*t*a
        values[k] = reference_h(frame, replace(state, position=trial[:3], rotation_vector=trial[3:]))
    velocity = (values[-2]-8*values[-1]+8*values[1]-values[2])/(12*TIME_STEP)
    acceleration = (-values[-2]+16*values[-1]-30*values[0]+16*values[1]-values[2])/(12*TIME_STEP**2)
    return velocity, acceleration


def blocks_close(actual, expected, tolerance=DERIVATIVE):
    assert actual.shape == expected.shape
    # 3x3 output/input blocks use separate SI units and1-unit floors.
    for i in range(actual.shape[0]//3):
        for j in range(actual.shape[1]//3):
            section = (slice(3*i, 3*i+3), slice(3*j, 3*j+3))
            error = np.linalg.norm(actual[section]-expected[section])
            scale = max(np.linalg.norm(expected[section]), 1.)
            assert error <= tolerance*scale, (i, j, error/scale, tolerance)


@pytest.mark.kinematics
@pytest.mark.parametrize('case,theta,anchor_delta,position', (
    ('equilibrium', (0., 0., 0.), (0., 0., 0.), (.1, .2, -.3)),
    ('series-boundary', (.02, 0., 0.), (0., 0., 0.), (.8, -.3, 1.1)),
    ('distinct-anchor', (.25, -.2, .1), (-.1, .15, .2), (.8, -.3, 1.1)),
    ('near-seam-recharted', (0., 0., 0.), (math.pi-.04, 0., 0.), (.8, -.3, 1.1)),
), ids=('equilibrium', 'series-boundary', 'distinct-anchor', 'near-seam-recharted'))
def test_exact_pose_rates_and_coordinate_derivatives(case, theta, anchor_delta, position):
    f = frame()
    state = RigidBodyState(position, theta, V, A, rotation(anchor_delta)@f.equilibrium_rotation)
    result = f.evaluate(state)
    reference = reference_h(f, state)
    for rows in (slice(0, 3), slice(3, 6)):
        assert np.linalg.norm(result.coordinates[rows]-reference[rows]) <= ALGEBRA*max(np.linalg.norm(reference[rows]), 1.)
    columns = []
    for j in range(6):
        e = FD*np.eye(6)[j]
        columns.append((reference_h(f, moved(state, 'q', e))-reference_h(f, moved(state, 'q', -e)))/(2*FD))
    blocks_close(result.jacobian, np.column_stack(columns))
    velocity, acceleration = independent_rates(f, state)
    for actual, expected in ((result.velocity, velocity), (result.acceleration, acceleration)):
        for rows in (slice(0, 3), slice(3, 6)):
            assert np.linalg.norm(actual[rows]-expected[rows]) <= DERIVATIVE*max(np.linalg.norm(expected[rows]), 1.)
    for j in range(3):
        e = FD*np.eye(6)[3+j]
        plus, minus = f.evaluate(moved(state, 'q', e)), f.evaluate(moved(state, 'q', -e))
        blocks_close(result.jacobian_derivatives[j], (plus.jacobian-minus.jacobian)/(2*FD))
        for k in range(3):
            blocks_close(result.jacobian_second_derivatives[j, k],
                         (plus.jacobian_derivatives[k]-minus.jacobian_derivatives[k])/(2*FD))
    force_h = np.array((1., -2., 3., -.4, .7, .2))
    assert abs((result.jacobian.T@force_h)@V-force_h@result.velocity) <= ALGEBRA*max(abs(force_h@result.velocity), 1.)
    assert np.linalg.norm(result.world_rate_map @ state.world_velocity-result.velocity) <= ALGEBRA
    for rows in (slice(0, 3), slice(3, 6)):
        actual = (result.world_rate_map @ state.world_velocity)[rows]
        expected = result.velocity[rows]
        assert np.linalg.norm(actual-expected) <= ALGEBRA*max(np.linalg.norm(expected), 1.)


@pytest.mark.composition
def test_nonzero_chart_with_distinct_near_seam_equilibrium_log():
    test_exact_pose_rates_and_coordinate_derivatives('mixed-near-seam',
        (.1, -.05, .08), (math.pi-.14, 0., 0.), (.8, -.3, 1.1))


def coupled_case():
    fa = frame('a')
    fb = EquilibriumHydroFrame('b', (2., -.4, .2), rotation((-.3, .2, .1)),
                              rotation((.4, -.2, -.3)), (-.3, .5, .1))
    sa = RigidBodyState((.8, -.3, 1.1), (.25, -.2, .1), V, A,
                        rotation((.13, -.06, .11))@fa.equilibrium_rotation)
    sb = RigidBodyState((2.2, -.1, .5), (-.15, .1, .2), (-.1, .2, .3, -.2, .1, .4),
                        (.2, -.1, .3, .1, -.2, .2), rotation((-.1, .07, .15))@fb.equilibrium_rotation)
    L = np.arange(144, dtype=float).reshape(12, 12)/600+np.diag(np.linspace(1., 2., 12))
    return (fa, fb), {'b': sb, 'a': sa}, L@L.T


def coupled_fd(frames, states, force, field):
    columns = []
    for f in frames:
        for j in range(6):
            e = FD*np.eye(6)[j]
            values = []
            for sign in (1., -1.):
                trial = dict(states)
                trial[f.body_id] = moved(states[f.body_id], field, sign*e)
                values.append(force(evaluate_coupled(frames, trial)))
            columns.append((values[0]-values[1])/(2*FD))
    return np.column_stack(columns)


@pytest.mark.restoring
def test_coupled_baseline_force_tangent_world_wrench_and_order():
    frames, states, C = coupled_case()
    result = evaluate_coupled(frames, states)
    preload = np.linspace(1., 3., 12)
    F = preload-C@result.coordinates
    pulled = result.pullback_force(F, C)
    force = lambda r: r.pullback_force(preload-C@r.coordinates, C).external_force
    blocks_close(pulled.stiffness, -coupled_fd(frames, states, force, 'q'))
    assert result.body_ids == ('a', 'b') and result.dof_keys[6] == 'b::surge'
    np.testing.assert_array_equal(evaluate_coupled(frames, dict(reversed(tuple(states.items())))).coordinates, result.coordinates)
    velocity = np.concatenate([states[f.body_id].velocity for f in frames])
    world_velocity = np.concatenate([states[f.body_id].world_velocity for f in frames])
    power = F@result.velocity
    assert abs(pulled.external_force@velocity-power) <= ALGEBRA*max(abs(power), 1.)
    assert abs(pulled.world_wrench@world_velocity-power) <= ALGEBRA*max(abs(power), 1.)
    # Nonzero baseline connection must not collapse to naive congruence.
    assert np.linalg.norm(pulled.stiffness-result.jacobian.T@C@result.jacobian) > 1e-3


@pytest.mark.inertia
def test_full_coupled_inertia_force_bias_and_total_tangents():
    frames, states, matrix = coupled_case()
    result = evaluate_coupled(frames, states)
    pulled = result.pullback_added_inertia(matrix)
    force = lambda r: r.pullback_added_inertia(matrix).internal_force
    for field, tangent in (('q', pulled.stiffness), ('velocity', pulled.damping), ('acceleration', pulled.mass)):
        blocks_close(tangent, coupled_fd(frames, states, force, field))
    acceleration = np.concatenate([states[f.body_id].acceleration for f in frames])
    blocks_close(pulled.mass, pulled.mass.T, ALGEBRA)
    assert np.linalg.norm(pulled.mass[:6, 6:]) > .1
    for rows in (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)):
        scale = max(np.linalg.norm(pulled.internal_force[rows]), 1.)
        assert np.linalg.norm((pulled.mass@acceleration+pulled.geometric_force-pulled.internal_force)[rows]) <= ALGEBRA*scale
    independently_accelerated = np.concatenate([independent_rates(f, states[f.body_id])[1] for f in frames])
    expected = result.jacobian.T @ matrix @ independently_accelerated
    for rows in (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)):
        assert np.linalg.norm(pulled.internal_force[rows]-expected[rows]) <= DERIVATIVE*max(np.linalg.norm(expected[rows]), 1.)


@pytest.mark.asymmetric
def test_arbitrary_source_matrices_are_preserved_without_symmetrizing():
    # Deliberately nonphysical matrices test algebra preservation, not admission.
    frames, states, _ = coupled_case()
    matrix = np.triu(np.arange(144, dtype=float).reshape(12, 12)/600)+np.diag(np.linspace(1., 2., 12))
    result = evaluate_coupled(frames, states)
    preload = np.linspace(1., 3., 12)
    source_force = preload-matrix@result.coordinates
    restoring = result.pullback_force(source_force, matrix)
    force = lambda r: r.pullback_force(preload-matrix@r.coordinates, matrix).external_force
    blocks_close(restoring.stiffness, -coupled_fd(frames, states, force, 'q'))
    inertia = result.pullback_added_inertia(matrix)
    blocks_close(inertia.mass, result.jacobian.T@matrix@result.jacobian, ALGEBRA)
    assert np.linalg.norm(inertia.mass-inertia.mass.T) > .1
    expected = result.jacobian.T@matrix@result.acceleration
    for rows in (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)):
        assert np.linalg.norm(inertia.internal_force[rows]-expected[rows]) <= ALGEBRA*max(np.linalg.norm(expected[rows]), 1.)
    force = lambda r: r.pullback_added_inertia(matrix).internal_force
    for field, tangent in (('q', inertia.stiffness), ('velocity', inertia.damping), ('acceleration', inertia.mass)):
        blocks_close(tangent, coupled_fd(frames, states, force, field))


@pytest.mark.refusal
def test_input_refusals_and_immutable_geometry_outputs():
    f = frame()
    for change in ({'coefficient_axes': np.diag((-1., 1., 1.))},
                   {'reference_point_body': (True, 0., 0.)},
                   {'equilibrium_position': (float('nan'), 0., 0.)}, {'body_id': 'a::roll'}):
        with pytest.raises(MarineHydroCoordinatesError):
            replace(f, **change)
    state = RigidBodyState(f.equilibrium_position, anchor=f.equilibrium_rotation)
    result = evaluate_coupled((f,), {'a': state})
    assert np.linalg.norm(result.coordinates) <= ALGEBRA
    for states in ({}, {'a': state, 'b': state}):
        with pytest.raises(MarineHydroCoordinatesError):
            evaluate_coupled((f,), states)
    with pytest.raises(MarineHydroCoordinatesError):
        evaluate_coupled((f, f), {'a': state})
    for force, matrix in ((np.zeros(5), np.eye(6)), (np.zeros(6), np.eye(5)),
                          (np.zeros(6), np.full((6, 6), float('inf')))):
        with pytest.raises(MarineHydroCoordinatesError):
            result.pullback_force(force, matrix)
    for arr in (result.coordinates, result.jacobian, result.blocks[0].jacobian_second_derivatives):
        with pytest.raises(ValueError):
            arr.setflags(write=True)
    seam = RigidBodyState(f.equilibrium_position, anchor=rotation((math.pi, 0., 0.))@f.equilibrium_rotation)
    with pytest.raises(MarineHydroCoordinatesError, match='principal seam'):
        f.evaluate(seam)
    # Changing equilibrium creates a new record, never relabels old data.
    changed = replace(f, equilibrium_position=(8., 0., 0.))
    assert np.linalg.norm(changed.evaluate(state).coordinates) > 1.
    assert f.equilibrium_position == (.1, .2, -.3)
    with pytest.raises(MarineHydroCoordinatesError):
        f.evaluate(None)
    with pytest.raises(MarineHydroCoordinatesError):
        evaluate_coupled((f,), {'a': None})
    with pytest.raises(MarineHydroCoordinatesError):
        CoupledHydroEvaluation((result.blocks[0], result.blocks[0]))
    input_axes = np.asarray(f.coefficient_axes).copy()
    input_rho = np.asarray(f.reference_point_body).copy()
    isolated = replace(f, coefficient_axes=input_axes, reference_point_body=input_rho)
    input_axes[:] = 0.
    input_rho[:] = 0.
    assert isolated.coefficient_axes == f.coefficient_axes
    assert isolated.reference_point_body == f.reference_point_body
    for frames in (None, 3, ()):
        with pytest.raises(MarineHydroCoordinatesError):
            evaluate_coupled(frames, {})
    for blocks in (None, 3, ()):
        with pytest.raises(MarineHydroCoordinatesError):
            CoupledHydroEvaluation(blocks)
    for force in (1., np.zeros(3), np.zeros(0), 'bad', (True, 0., 0., 0., 0., 0.)):
        with pytest.raises(MarineHydroCoordinatesError):
            ForcePullback(force, np.eye(6), np.zeros(6))
        with pytest.raises(MarineHydroCoordinatesError):
            InertiaPullback(force, np.eye(6), np.eye(6), np.eye(6), np.zeros(6))
    with pytest.raises(MarineHydroCoordinatesError):
        ForcePullback(np.zeros(6), np.eye(3), np.zeros(6))
    with pytest.raises(MarineHydroCoordinatesError):
        InertiaPullback(np.zeros(6), np.eye(6), np.eye(6), np.eye(6), np.zeros(3))
