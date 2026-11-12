"""Additive dry mechanics checks with independent matrix-exponential oracles.

Seven fixed states; no trajectories or random draws. Every derivative is checked
by a force/moment x translation/rotation block. Existing row checks stay intact.
"""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet
from anyfesolver.marine_rigid import RigidBody6D

I = ((3., .2, .1), (.2, 4., .3), (.1, .3, 5.))
Z = ((math.cos(.4), -math.sin(.4), 0.),
     (math.sin(.4), math.cos(.4), 0.), (0., 0., 1.))
X = ((1., 0., 0.), (0., 0., -1.), (0., 1., 0.))
CASES = (
    ('offset-general', 7., (.2, -.1, .3), I, (.25, -.2, .1), Z),
    ('series-boundary-zero-cg', 2.5, (0., 0., 0.),
     ((1., .1, 0.), (.1, 1.4, .05), (0., .05, 1.8)), (.02, 0., 0.), Z),
    ('zero-chart-rotated-anchor', 7., (.2, -.1, .3), I, (0., 0., 0.), X),
    ('near-principal-seam', 7., (.2, -.1, .3), I, (math.pi-.03, 0., 0.), X),
)
# Prospective SI references: 1 N, 1 Nm and 1 m / 1 rad (or their time derivatives).
# Absolute near-zero allowance is 1e-8 of the corresponding derivative-unit floor.
FD_STEP = 1e-5
DERIVATIVE_TOLERANCE = 1e-8
PHYSICS_TOLERANCE = 1e-10
POWER_STEP = 1e-3
VELOCITY = (.3, -.4, .2, .5, -.3, .2)
ACCELERATION = (.4, .2, -.3, -.2, .1, .5)
ZERO = (0.,)*6


def skew(v):
    x, y, z = v
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def vee(v):
    return np.array((v[2, 1], v[0, 2], v[1, 0]))


def independent_physics(body, state):
    """Newton-Euler reference; no production SO(3), force or tangent helpers.

    The (1,3) block of exp([[A,E,F/2],[0,A,E],[0,0,A]]) is half
    D^2 exp(A)[E,E] + half D exp(A)[F]. Thus twice that block is Rddot.
    """
    theta = np.asarray(state.rotation_vector)
    velocity = np.asarray(state.velocity)
    acceleration = np.asarray(state.acceleration)
    A, E, F = skew(theta), skew(velocity[3:]), skew(acceleration[3:])
    augmented = np.zeros((9, 9))
    for section in (slice(0, 3), slice(3, 6), slice(6, 9)):
        augmented[section, section] = A
    augmented[:3, 3:6] = E
    augmented[3:6, 6:] = E
    augmented[:3, 6:] = .5*F
    matrix = expm(augmented)
    anchor = np.asarray(state.anchor)
    R = matrix[:3, :3] @ anchor
    Rdot = matrix[:3, 3:6] @ anchor
    Rddot = 2.*matrix[:3, 6:] @ anchor
    omega_matrix = Rdot @ R.T
    alpha_matrix = Rddot @ R.T - omega_matrix @ omega_matrix
    omega = vee(.5*(omega_matrix-omega_matrix.T))
    alpha = vee(.5*(alpha_matrix-alpha_matrix.T))
    exp_A = expm(A)
    rate_map = np.eye(6)
    for j in range(3):
        differential = expm_frechet(A, skew(np.eye(3)[j]), compute_expm=False)
        spatial = differential @ exp_A.T
        rate_map[3:, 3+j] = vee(.5*(spatial-spatial.T))
    c = R @ np.asarray(body.cg_body)
    inertia = R @ np.asarray(body.inertia_cg_body) @ R.T
    cg_acc = acceleration[:3] + np.cross(alpha, c) + np.cross(omega, np.cross(omega, c))
    force = body.mass*cg_acc
    moment = inertia @ alpha + np.cross(omega, inertia @ omega) + np.cross(c, force)
    covector = rate_map.T @ np.r_[force, moment]
    cg_velocity = velocity[:3] + np.cross(omega, c)
    energy = .5*body.mass*np.dot(cg_velocity, cg_velocity) + .5*omega @ inertia @ omega
    return covector, float(energy)


def check_blocks(actual, expected, kind, case):
    input_units = {'stiffness': ('m', 'rad'), 'damping': ('m/s', 'rad/s'),
                   'mass': ('m/s^2', 'rad/s^2')}[kind]
    for row, output_unit in enumerate(('N', 'Nm')):
        rows = slice(3*row, 3*row+3)
        for col, input_unit in enumerate(input_units):
            cols = slice(3*col, 3*col+3)
            # References all equal one in declared SI units, not mixed-unit norms.
            reference = expected[rows, cols]
            error = np.linalg.norm(actual[rows, cols]-reference)
            scale = max(np.linalg.norm(reference), 1.)
            assert error <= DERIVATIVE_TOLERANCE*scale, (
                case, kind, f'{output_unit}/({input_unit})', error/scale,
                DERIVATIVE_TOLERANCE)


@pytest.mark.parametrize('case,mass,cg,inertia,theta,anchor', CASES, ids=[case[0] for case in CASES])
def test_dry_derivatives_against_independent_dimensioned_blocks(
        case, mass, cg, inertia, theta, anchor, velocity=VELOCITY,
        acceleration=ACCELERATION):
    body = RigidBody6D(body_id=case, mass=mass, cg_body=cg,
                      inertia_cg_body=inertia, reference_origin=(.8, -.2, 1.1))
    state = body.state(position=(.8, -.2, 1.1), rotation_vector=theta, anchor=anchor,
                       velocity=velocity, acceleration=acceleration)
    reference, energy = independent_physics(body, state)
    actual = body.inertial_force(state)
    for rows in (slice(0, 3), slice(3, 6)):
        assert np.linalg.norm(actual[rows]-reference[rows]) <= PHYSICS_TOLERANCE*max(np.linalg.norm(reference[rows]), 1.)
    assert abs(body.kinetic_energy(state)-energy) <= PHYSICS_TOLERANCE*max(energy, 1.)
    q = np.r_[state.position, state.rotation_vector]
    qdot, qddot = np.asarray(state.velocity), np.asarray(state.acceleration)
    # Local five-point derivative of independent kinetic energy, in watts.
    # These nearby algebraic states are not an integrated trajectory.
    energies = {}
    for offset in (-2, -1, 1, 2):
        time = offset*POWER_STEP
        qq = q+time*qdot+.5*time*time*qddot
        trial = replace(state, position=qq[:3], rotation_vector=qq[3:],
                        velocity=qdot+time*qddot)
        energies[offset] = independent_physics(body, trial)[1]
    energy_rate = (energies[-2]-8*energies[-1]+8*energies[1]-energies[2])/(12*POWER_STEP)
    assert abs(actual @ qdot-energy_rate) <= PHYSICS_TOLERANCE*max(abs(energy_rate), 1.)
    tangents = body.dry_tangents(state)
    for field, coordinate in (('stiffness', 'q'), ('damping', 'qdot'), ('mass', 'qddot')):
        columns = []
        for j in range(6):
            values = []
            for sign in (1., -1.):
                change = sign*FD_STEP*np.eye(6)[j]
                qq = q+change if coordinate == 'q' else q
                vv = qdot+change if coordinate == 'qdot' else qdot
                aa = qddot+change if coordinate == 'qddot' else qddot
                trial = replace(state, position=qq[:3], rotation_vector=qq[3:],
                                velocity=vv, acceleration=aa)
                values.append(independent_physics(body, trial)[0])
            columns.append((values[0]-values[1])/(2.*FD_STEP))
        finite_difference = np.column_stack(columns)
        check_blocks(getattr(tangents, field), finite_difference, field, case)


@pytest.mark.parametrize('case,velocity,acceleration', (
    ('zero-velocity', ZERO, ACCELERATION),
    ('zero-acceleration', VELOCITY, ZERO),
    ('zero-rates', ZERO, ZERO),
))
def test_dry_zero_rates_and_accelerations(case, velocity, acceleration):
    _, mass, cg, inertia, theta, anchor = CASES[0]
    test_dry_derivatives_against_independent_dimensioned_blocks(
        case, mass, cg, inertia, theta, anchor, velocity, acceleration)
