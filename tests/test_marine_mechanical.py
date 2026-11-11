"""Prospective W4 native development cases, fixed before first execution.

Selectors required: do not run all trajectories in one process. Cases 1/2/3
contain four 100-step trajectories together; case4 is one 100-step trajectory.
No BEM, external qualification, parameter sweep or structural campaign.
"""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver.marine_rigid import RigidBody6D
from anyfesolver.marine_link import MechanicalLink6D
from anyfesolver.marine_contributions import DynamicContribution, ContributionResult, HISTORY_CHART
from anyfesolver.marine_mechanical import (
    MarineMechanicalModel, MarineMechanicalRunSpec, MarineMechanicalSolveError,
    solve_marine_mechanical)


def sphere(bid='a', mass=3., position=(0., 0., 0.)):
    return RigidBody6D(bid, mass, (0., 0., 0.), np.eye(3)*2, position)


@pytest.mark.parametrize('gravity', [None, (.2, -.3, -9.81)])
def test_spherical_free_and_gravity(gravity):
    body = sphere()
    initial = body.state(position=(1., -.2, .4), velocity=(.3, -.2, .1, .7, -.4, .2),
                         anchor=Rotation.from_rotvec((.2, -.1, .3)).as_matrix())
    result = solve_marine_mechanical(MarineMechanicalModel((body,), gravity_world=gravity),
        {'a': initial}, MarineMechanicalRunSpec(.001, 100))
    assert result.status == 'completed' and len(result.accepted) == 101
    g = np.zeros(3) if gravity is None else np.array(gravity)
    for frame in result.accepted:
        t = frame.time
        position = np.array(initial.position) + t*np.array(initial.velocity[:3]) + .5*t*t*g
        orientation = Rotation.from_rotvec(t*np.array(initial.velocity[3:])).as_matrix() @ initial.rotation
        np.testing.assert_allclose(frame.positions['a'], position, atol=1e-10, rtol=0)
        np.testing.assert_allclose(frame.rotations['a'], orientation, atol=1e-10, rtol=0)
        np.testing.assert_allclose(frame.world_accelerations['a'][:3], g, atol=1e-10, rtol=0)


def test_nonspherical_offset_free_body_conservation():
    body = RigidBody6D('a', 7., (.3, -.4, .2),
        ((4., .1, .2), (.1, 5., .3), (.2, .3, 6.)))
    state = body.state(position=(.2, .3, -.1), velocity=(.3, -.2, .1, .8, -.5, .4),
                       anchor=Rotation.from_rotvec((.2, -.1, .3)).as_matrix())
    result = solve_marine_mechanical(MarineMechanicalModel((body,)), {'a': state},
                                    MarineMechanicalRunSpec(.001, 100))
    invariant = []
    for f in result.accepted:
        R, v, w = f.rotations['a'], f.world_velocities['a'][:3], f.world_velocities['a'][3:]
        c = R @ np.array(body.cg_body)
        Ic = R @ np.array(body.inertia_cg_body) @ R.T
        vg = v + np.cross(w, c)
        p = body.mass*vg
        H = np.cross(f.positions['a']+c, p) + Ic @ w
        energy = .5*body.mass*(vg @ vg) + .5*(w @ Ic @ w)
        invariant.append((p, H, energy))
    assert result.status == 'completed'
    for p, H, energy in invariant:
        for value, baseline in zip((p, H, energy), invariant[0]):
            assert np.linalg.norm(value-baseline) <= 1e-6*max(np.linalg.norm(baseline), 1.)


def test_axial_two_body_link_linear_limit():
    a, b = sphere('a', 3.), sphere('b', 5., (1.1, 0., 0.))
    link = MechanicalLink6D('axial', 'a', 'b', (0., 0., 0.), (0., 0., 0.),
        np.eye(3), np.eye(3), (1., 0., 0.), np.eye(3),
        np.diag((20., 0., 0., 0., 0., 0.)), np.zeros((6, 6)))
    result = solve_marine_mechanical(MarineMechanicalModel((a, b), (link,)),
        {'a': a.state(), 'b': b.state()}, MarineMechanicalRunSpec(.001, 100))
    cm, omega = 5.*1.1/8., np.sqrt(20.*(1./3.+1./5.))
    for f in result.accepted:
        distance = 1. + .1*np.cos(omega*f.time)
        expected = (cm-5./8.*distance, cm+3./8.*distance)
        for bid, x in zip(('a', 'b'), expected):
            assert abs(f.positions[bid][0]-x) <= 1e-3*.1
    assert result.status == 'completed'


def test_nonlinear_framed_offset_link_work_and_balance():
    a = RigidBody6D('a', 3., (.2, 0., .1), np.diag((2., 3., 4.)))
    b = RigidBody6D('b', 5., (-.1, .1, 0.), np.diag((4., 5., 6.)), (1.2, .2, -.1))
    refs = (a.state(), b.state(anchor=Rotation.from_rotvec((.05, -.03, .02)).as_matrix()))
    link = MechanicalLink6D.from_reference_states(link_id='offset', body_a='a', body_b='b',
        state_a=refs[0], state_b=refs[1],
        rho_a=(.4, 0., .2), rho_b=(-.3, .1, 0.),
        axes_a=Rotation.from_rotvec((.1, -.2, .05)).as_matrix(),
        axes_b=Rotation.from_rotvec((-.1, .1, .2)).as_matrix(),
        stiffness=np.diag((10., 8., 12., 3., 4., 5.)),
        damping=np.diag((.1, .2, .15, .05, .05, .05)), preload=(.2, -.1, .05, 0., .02, -.01))
    states = {'a': replace(refs[0], anchor=Rotation.from_rotvec((.03, .01, -.02)).as_matrix(),
                          velocity=(.02, -.01, .01, .03, -.02, .01)),
              'b': replace(refs[1], position=np.array(refs[1].position)+(.05, -.02, .03),
                          velocity=(-.01, .01, .02, -.01, .02, .01))}
    result = solve_marine_mechanical(MarineMechanicalModel((a, b), (link,)), states,
                                    MarineMechanicalRunSpec(.001, 100))
    assert result.status == 'completed' and len(result.accepted) == 101
    for f in result.accepted:
        forces = f.force_decomposition['mechanical-links']
        netforce = forces['a'][:3]+forces['b'][:3]
        netmoment = sum((forces[bid][3:]+np.cross(f.positions[bid], forces[bid][:3])
                         for bid in ('a', 'b')), np.zeros(3))
        assert np.linalg.norm(netforce) <= 1e-10
        assert np.linalg.norm(netmoment) <= 1e-10
    for f in (result.accepted[0], result.accepted[50], result.accepted[100]):
        current = {bid: body.state(position=f.positions[bid], anchor=f.rotations[bid],
                                  velocity=f.world_velocities[bid]) for bid, body in (('a', a), ('b', b))}
        h = 1e-5
        potential = []
        for sign in (1., -1.):
            moved = {}
            for bid, s in current.items():
                nu = f.world_velocities[bid]
                moved[bid] = replace(s, position=f.positions[bid]+sign*h*nu[:3],
                    anchor=Rotation.from_rotvec(sign*h*nu[3:]).as_matrix() @ f.rotations[bid])
            potential.append(link.evaluate(moved['a'], moved['b']).potential_energy)
        Udot = (potential[0]-potential[1])/(2*h)
        power = sum(f.world_velocities[bid] @ f.force_decomposition['mechanical-links'][bid]
                    for bid in ('a', 'b'))
        assert abs(power+Udot+f.link_damping_power) <= 1e-8*max(abs(power), abs(Udot), 1.)


class CountingMass(DynamicContribution):
    contribution_id = 'extra'
    ownership_key = 'test-extra-inertia'
    def __init__(self, mass=2., force=6., during_trial=None):
        self.mass, self.force, self.during_trial = mass, force, during_trial
    def evaluate_trial(self, history, states, layout):
        previous = history.get(self.contribution_id, {'steps': 0})['steps']
        if self.during_trial:
            self.during_trial(previous)
        a = states['a'].acceleration[0]
        M = np.zeros((6, 6)); M[0, 0] = self.mass
        F = np.zeros(6); F[0] = self.force-self.mass*a
        return ContributionResult(self.contribution_id, F, np.zeros((6, 6)), np.zeros((6, 6)),
                                  M, {}, {'steps': previous+1})


def test_acceleration_once_history_and_immutable_output():
    body = sphere(mass=4.)
    result = solve_marine_mechanical(MarineMechanicalModel((body,), contributions=(CountingMass(),)),
        {'a': body.state()}, MarineMechanicalRunSpec(.01, 3))
    assert result.histories['extra']['steps'] == 3  # initialization/trials never advance history
    for f in result.accepted:
        assert f.world_accelerations['a'][0] == pytest.approx(1., abs=1e-10)
        assert f.positions['a'][0] == pytest.approx(.5*f.time*f.time, abs=1e-10)
        with pytest.raises(ValueError):
            f.rotations['a'].setflags(write=True)
        with pytest.raises(TypeError):
            f.positions['a'] = np.zeros(3)


def test_mid_trial_and_final_progress_cancellation():
    body = sphere(mass=4.)
    cancelled = {'value': False}
    def during_trial(previous):
        if previous == 1:
            cancelled['value'] = True
    model = MarineMechanicalModel((body,), contributions=(CountingMass(during_trial=during_trial),))
    result = solve_marine_mechanical(model, {'a': body.state()}, MarineMechanicalRunSpec(.01, 3),
                                    cancel=lambda: cancelled['value'])
    assert result.status == 'cancelled' and result.accepted[-1].step == 1
    assert result.histories['extra']['steps'] == 1
    cancelled['value'] = False
    def progress(frame):
        if frame.step == 3:
            cancelled['value'] = True
    model = MarineMechanicalModel((body,), contributions=(CountingMass(),))
    result = solve_marine_mechanical(model, {'a': body.state()}, MarineMechanicalRunSpec(.01, 3),
                                    cancel=lambda: cancelled['value'], progress=progress)
    assert result.status == 'cancelled' and result.accepted[-1].step == 3
    assert result.histories['extra']['steps'] == 3


def test_singular_initial_mass_and_transport_failure_preserve_evidence():
    body = sphere(mass=4.)
    model = MarineMechanicalModel((body,), contributions=(CountingMass(mass=-4.),))
    with pytest.raises(MarineMechanicalSolveError) as caught:
        solve_marine_mechanical(model, {'a': body.state()}, MarineMechanicalRunSpec(.01, 3))
    assert caught.value.reason == 'singular' and not caught.value.partial.accepted
    assert not len(caught.value.histories)
    class ChartMemory(CountingMass):
        history_basis = HISTORY_CHART
        def history_transport(self, history, info):
            assert info.for_body('a') is not None
            if history['steps'] == 2:
                raise RuntimeError('transport failed at second step')
            return {'steps': history['steps'], 'anchor': info.for_body('a').new_anchor}
    model = MarineMechanicalModel((body,), contributions=(ChartMemory(),))
    with pytest.raises(MarineMechanicalSolveError) as caught:
        solve_marine_mechanical(model, {'a': body.state()}, MarineMechanicalRunSpec(.01, 3))
    assert caught.value.partial.accepted[-1].step == 1
    assert caught.value.histories['extra']['steps'] == 1
