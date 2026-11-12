"""Focused W4 link oracles: algebra, power, action-reaction and FD tangents.

Development evidence only; no qualification campaign. Numerical case
counts per invocation: 12 random analytic states + 3 fixed cases = 15
(fixed prospectively in the governing W4 contract).
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver.marine_rigid import RigidBody6D, RigidBodyState
from anyfesolver.marine_link import MechanicalLink6D, MarineLinkError


def close(actual, expected, tol=1e-10):
    assert np.linalg.norm(np.asarray(actual) - expected) <= tol*max(
        np.linalg.norm(expected), 1e-12)


def close_abs(actual, expected, tol=1e-10):
    assert np.linalg.norm(np.asarray(actual) - expected) <= tol


def body_a():
    return RigidBody6D('alpha', 3., (.4, -.2, .1),
                       ((2., .1, 0.), (.1, 3., .2), (0., .2, 4.)))


def body_b():
    return RigidBody6D('beta', 5., (-.3, .5, .2),
                       ((5., .2, .1), (.2, 6., 0.), (.1, 0., 7.)))


def make_link():
    rng = np.random.default_rng(7)
    K = rng.normal(size=(6, 6))
    K = K @ K.T + np.eye(6)
    C = .05*(lambda m: m @ m.T)(rng.normal(size=(6, 6)))
    p0 = (.3, -.2, .1, .05, -.03, .02)
    return MechanicalLink6D(
        'link-1', 'alpha', 'beta', (.8, 0., .3), (-.5, .2, 0.),
        np.eye(3), Rotation.from_rotvec([.2, -.1, .3]).as_matrix(),
        (1.2, -.3, .4), Rotation.from_rotvec([.1, .2, -.15]).as_matrix(),
        K, C, p0)


def random_pair(rng, ba, bb, scale=.3):
    sa = ba.state(position=rng.normal(size=3), rotation_vector=rng.normal(size=3)*scale,
                  velocity=rng.normal(size=6)*.5,
                  anchor=Rotation.from_rotvec(rng.normal(size=3)).as_matrix())
    sb = bb.state(position=rng.normal(size=3), rotation_vector=rng.normal(size=3)*scale,
                  velocity=rng.normal(size=6)*.5,
                  anchor=Rotation.from_rotvec(rng.normal(size=3)).as_matrix())
    return sa, sb


def test_link_algebra_power_and_action_reaction_6_states():
    rng = np.random.default_rng(42)
    link = make_link()
    ba, bb = body_a(), body_b()
    K = np.asarray(link.stiffness)
    C = np.asarray(link.damping)
    p0 = np.asarray(link.preload)
    for _ in range(6):
        sa, sb = random_pair(rng, ba, bb)
        ev = link.evaluate(sa, sb)
        A = sa.rotation @ np.asarray(link.axes_a)
        B = sb.rotation @ np.asarray(link.axes_b)
        t = ((np.asarray(sb.position) + sb.rotation @ np.asarray(link.rho_b))
             - (np.asarray(sa.position) + sa.rotation @ np.asarray(link.rho_a)))
        close(ev.strain[:3], A.T @ t - np.asarray(link.rest_translation))
        expected_phi = Rotation.from_matrix(
            A.T @ B @ np.asarray(link.rest_rotation).T).as_rotvec()
        close(ev.strain[3:], expected_phi)
        qdot = np.r_[np.asarray(sa.velocity), np.asarray(sb.velocity)]
        close(ev.strain_rate, ev.jacobian @ qdot)
        close(ev.local_force, K @ ev.strain + p0 + C @ ev.strain_rate)
        close_abs([ev.potential_energy],
                  [.5*ev.strain @ K @ ev.strain + p0 @ ev.strain])
        close_abs([ev.damping_power], [ev.strain_rate @ C @ ev.strain_rate])
        # Power identity: qdot.Qext = -Udot - dissipation (exact algebra).
        elastic = K @ ev.strain + p0
        close_abs(qdot @ np.r_[ev.chart_force_a, ev.chart_force_b],
                  -(elastic @ ev.strain_rate
                    + ev.strain_rate @ C @ ev.strain_rate))
        # Action-reaction: total force and moment about a COMMON world origin.
        close_abs(ev.world_wrench_a + ev.world_wrench_b, np.zeros(6))
        # World wrenches are the B^-T recovery of the chart covectors,
        # including the frame-dependent translation spring torque on A.
        for state, chart_force, wrench in ((sa, ev.chart_force_a, ev.world_wrench_a),
                                           (sb, ev.chart_force_b, ev.world_wrench_b)):
            about_reference = np.linalg.solve(state.rate_map.T, chart_force)
            origin = (np.r_[about_reference[:3],
                            about_reference[3:] + np.cross(np.asarray(state.position),
                                                           about_reference[:3])])
            close_abs(origin, wrench)


def test_link_tangents_match_finite_differences_6_states():
    rng = np.random.default_rng(1234)
    link = make_link()
    ba, bb = body_a(), body_b()
    h = 4e-6
    from dataclasses import replace

    def pair_force(sa, sb):
        ev = link.evaluate(sa, sb)
        return np.r_[ev.chart_force_a, ev.chart_force_b]

    for _ in range(6):
        sa, sb = random_pair(rng, ba, bb)
        ev = link.evaluate(sa, sb)
        for j in range(12):
            if j < 6:
                def perturbed(delta):
                    return (replace(sa, position=np.asarray(sa.position) + delta[:3],
                                    rotation_vector=np.asarray(sa.rotation_vector) + delta[3:]), sb)
                def perturbed_dot(delta):
                    return (replace(sa, velocity=np.asarray(sa.velocity) + delta), sb)
            else:
                def perturbed(delta):
                    return (sa, replace(sb, position=np.asarray(sb.position) + delta[:3],
                                        rotation_vector=np.asarray(sb.rotation_vector) + delta[3:]))
                def perturbed_dot(delta):
                    return (sa, replace(sb, velocity=np.asarray(sb.velocity) + delta))
            direction = np.zeros(6)
            direction[j % 6] = 1.
            kfd = -(pair_force(*perturbed(h*direction))
                    - pair_force(*perturbed(-h*direction)))/(2*h)
            cfd = -(pair_force(*perturbed_dot(h*direction))
                    - pair_force(*perturbed_dot(-h*direction)))/(2*h)
            force_rows = [0, 1, 2, 6, 7, 8]
            moment_rows = [3, 4, 5, 9, 10, 11]
            for matrix, fd in ((ev.stiffness, kfd), (ev.damping, cfd)):
                for rows in (force_rows, moment_rows):
                    expected = fd[rows]
                    actual = matrix[rows, j]
                    assert np.linalg.norm(expected - actual) <= 1e-8*max(
                        np.linalg.norm(expected), 1.)
        assert np.linalg.norm(ev.mass) == 0.


def test_reference_state_factory_rest_strain_and_preload():
    link = make_link()
    ba, bb = body_a(), body_b()
    sa = ba.state(position=(.1, .2, .3), rotation_vector=(.2, -.1, .1),
                  anchor=Rotation.from_rotvec([.4, .1, -.2]).as_matrix())
    sb = bb.state(position=(1.1, -.2, .4), rotation_vector=(-.1, .2, .3),
                  anchor=Rotation.from_rotvec([-.2, .3, .1]).as_matrix())
    rested = MechanicalLink6D.from_reference_states(
        link_id='rested', body_a='alpha', body_b='beta',
        rho_a=link.rho_a, rho_b=link.rho_b, axes_a=link.axes_a, axes_b=link.axes_b,
        state_a=sa, state_b=sb, stiffness=link.stiffness, damping=link.damping,
        preload=link.preload)
    ev = rested.evaluate(sa, sb)
    close_abs(ev.strain, np.zeros(6), 1e-12)
    close(ev.local_force, np.asarray(link.preload))
    close_abs([ev.potential_energy], [np.asarray(link.preload) @ np.zeros(6)])


def test_positive_axial_preload_pulls_endpoints_together():
    K = np.diag((30., 30., 30., 4., 4., 4.))
    link = MechanicalLink6D('preloaded', 'alpha', 'beta', (0., 0., 0.), (0., 0., 0.),
                             np.eye(3), np.eye(3), (1., 0., 0.), np.eye(3),
                             K, .1*np.eye(6), (2., 0., 0., 0., 0., 0.))
    ba, bb = body_a(), body_b()
    sa = ba.state(position=(0., 0., 0.))
    sb = bb.state(position=(1., 0., 0.))
    ev = link.evaluate(sa, sb)
    close_abs(ev.strain, np.zeros(6), 1e-12)
    close(ev.world_wrench_a[:3], (2., 0., 0.))
    close_abs(ev.world_wrench_a[3:], np.zeros(3))
    close_abs(ev.world_wrench_a + ev.world_wrench_b, np.zeros(6))


def test_link_refusals():
    K = np.eye(6)
    with pytest.raises(MarineLinkError):
        MechanicalLink6D('same', 'alpha', 'alpha', (0., 0., 0.), (0., 0., 0.),
                         np.eye(3), np.eye(3), (0., 0., 0.), np.eye(3),
                         np.array([[1., 2.], [3., 4.], [5., 6.]]), np.eye(6))
    asymmetric = np.array([[2., 1., 0., 0., 0., 0.], [0., 2., 0., 0., 0., 0.],
                           [0., 0., 2., 0., 0., 0.], [0., 0., 0., 2., 0., 0.],
                           [0., 0., 0., 0., 2., 0.], [0., 0., 0., 0., 0., 2.]])
    with pytest.raises(MarineLinkError):
        MechanicalLink6D('asym', 'alpha', 'beta', (0., 0., 0.), (0., 0., 0.),
                         np.eye(3), np.eye(3), (0., 0., 0.), np.eye(3),
                         asymmetric, np.eye(6))
    with pytest.raises(MarineLinkError):
        MechanicalLink6D('negative', 'alpha', 'beta', (0., 0., 0.), (0., 0., 0.),
                         np.eye(3), np.eye(3), (0., 0., 0.), np.eye(3),
                         -np.eye(6), np.eye(6))
    with pytest.raises(MarineLinkError):
        MechanicalLink6D('axes', 'alpha', 'beta', (0., 0., 0.), (0., 0., 0.),
                         np.diag((1., 1., -1.)), np.eye(3), (0., 0., 0.), np.eye(3),
                         np.eye(6), np.eye(6))
    with pytest.raises(MarineLinkError):
        MechanicalLink6D('tol', 'alpha', 'beta', (0., 0., 0.), (0., 0., 0.),
                         np.eye(3), np.eye(3), (0., 0., 0.), np.eye(3),
                         np.eye(6), np.eye(6), psd_tolerance=-1.)
    link = MechanicalLink6D('branch', 'alpha', 'beta', (0., 0., 0.), (0., 0., 0.),
                             np.eye(3), np.eye(3), (0., 0., 0.), np.eye(3),
                             np.eye(6), np.eye(6))
    ba, bb = body_a(), body_b()
    near_seam = ba.state(rotation_vector=(np.pi - 1e-7, 0., 0.))
    far = bb.state(position=(2., 0., 0.), rotation_vector=(0., 0., 0.))
    with pytest.raises(MarineLinkError, match='seam'):
        link.evaluate(near_seam, far)
    with pytest.raises(MarineLinkError):
        link.evaluate(ba.state(), 'not-a-state')
