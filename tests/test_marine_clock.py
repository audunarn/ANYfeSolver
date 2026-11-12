"""Fixed short clock/context checks; <=3steps, no physical source campaign."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver.marine_rigid import RigidBody6D
from anyfesolver.marine_contributions import (
    ContributionResult, ContributionSet, DynamicContribution,
    MarineContributionsError, StepInformation)
from anyfesolver.marine_mechanical import (
    MarineMechanicalModel, MarineMechanicalRunSpec, solve_marine_mechanical)


def body():
    return RigidBody6D('a', 4., (0., 0., 0.), 2.*np.eye(3), (0., 0., 0.))


def output(cid, force=None, history=None):
    return ContributionResult(cid, np.zeros(6) if force is None else force,
        np.zeros((6, 6)), np.zeros((6, 6)), np.zeros((6, 6)), {}, history)


class TimedForce(DynamicContribution):
    contribution_id = 'timed-force'
    ownership_key = 'test-clock-force'

    def __init__(self):
        self.prepared, self.evaluations = [], []

    def prepare_step_with_context(self, history, information):
        if not isinstance(information, StepInformation):
            raise MarineContributionsError('explicit clock required')
        self.prepared.append(information)

    def evaluate_trial(self, history, states, layout):
        raise MarineContributionsError('explicit clock required')

    def evaluate_trial_with_context(self, history, states, layout, information):
        assert information is self.prepared[-1]
        self.evaluations.append(information)
        force = np.zeros(6)
        force[0] = 4.*(2.+information.time)
        previous = history.get(self.contribution_id, {'steps': 0})['steps']
        return output(self.contribution_id, force, {'steps': previous+1})


class VelocityMemory(DynamicContribution):
    contribution_id = 'velocity-memory'
    ownership_key = 'test-clock-memory'

    def __init__(self, cancellation=None):
        self.cancellation = cancellation

    def evaluate_trial(self, history, states, layout):
        raise MarineContributionsError('explicit clock required')

    def evaluate_trial_with_context(self, history, states, layout, information):
        if information.phase == 'initialization':
            assert not information.previous_states and not len(history)
            # A conspicuous initialization trial must never become history.
            return output(self.contribution_id, history={'times': (-99.,), 'velocities': (), 'steps': 99})
        previous = history.get(self.contribution_id)
        if previous is None:
            assert information.step_index == 1 and information.previous_time == 0.
            times = (information.previous_time,)
            velocities = (information.previous_states['a'].world_velocity,)
            count = 0
        else:
            assert previous['times'][-1] == information.previous_time
            times, velocities, count = previous['times'], previous['velocities'], previous['steps']
        if self.cancellation is not None and information.step_index == 2:
            self.cancellation['cancel'] = True
        return output(self.contribution_id, history={'times': times+(information.time,),
            'velocities': velocities+(states['a'].world_velocity,), 'steps': count+1})


@pytest.mark.clock
def test_time_force_and_memory_seed_use_actual_accepted_initial_state():
    b = body()
    forcing, memory = TimedForce(), VelocityMemory()
    spec = MarineMechanicalRunSpec(.1, 3)
    initial = b.state(rotation_vector=(.2, -.1, .3),
        anchor=Rotation.from_rotvec((.1, .2, -.1)).as_matrix(),
        velocity=(.5, .2, -.1, .2, -.3, .1), acceleration=(99.,)*6)
    result = solve_marine_mechanical(MarineMechanicalModel((b,), contributions=(forcing, memory)),
                                    {'a': initial}, spec)
    assert result.status == 'completed'
    np.testing.assert_allclose(result.accepted[0].rotations['a'], initial.rotation, rtol=0., atol=1e-10)
    np.testing.assert_allclose(result.accepted[0].world_velocities['a'], initial.world_velocity, rtol=0., atol=1e-10)
    for frame in result.accepted:
        t = frame.time
        assert abs(frame.world_accelerations['a'][0]-(2.+t)) <= 1e-10*max(2.+t, 1.)
        assert abs(frame.world_velocities['a'][0]-(.5+2.*t+.5*t*t)) <= 1e-10
        assert frame.residual_norm <= spec.residual_tolerance
    assert [i.phase for i in forcing.prepared] == ['initialization', 'step', 'step', 'step']
    assert [i.step_index for i in forcing.prepared] == [0, 1, 2, 3]
    for information in forcing.prepared:
        assert information.time == information.step_index*spec.dt
        assert all(i is information for i in forcing.evaluations if i.step_index == information.step_index)
        if information.step_index == 0:
            assert not information.previous_states
        else:
            previous = result.accepted[information.step_index-1]
            state = information.previous_states['a']
            np.testing.assert_allclose(state.position, previous.positions['a'], rtol=0., atol=1e-10)
            np.testing.assert_allclose(state.rotation, previous.rotations['a'], rtol=0., atol=1e-10)
            np.testing.assert_allclose(state.world_velocity, previous.world_velocities['a'], rtol=0., atol=1e-10)
            np.testing.assert_allclose(state.world_acceleration, previous.world_accelerations['a'], rtol=0., atol=1e-10)
    stored = result.histories['velocity-memory']
    assert stored['steps'] == result.histories['timed-force']['steps'] == 3
    assert stored['times'] == tuple(frame.time for frame in result.accepted)
    for velocity, frame in zip(stored['velocities'], result.accepted):
        np.testing.assert_allclose(velocity, frame.world_velocities['a'], rtol=0., atol=1e-10)


@pytest.mark.cancel
def test_cancelled_timed_trial_preserves_last_accepted_clock_memory():
    b = body()
    flag = {'cancel': False}
    forcing = TimedForce()
    result = solve_marine_mechanical(MarineMechanicalModel((b,),
        contributions=(forcing, VelocityMemory(flag))), {'a': b.state(velocity=(.5, 0., 0., 0., 0., 0.))},
        MarineMechanicalRunSpec(.1, 3), cancel=lambda: flag['cancel'])
    assert result.status == 'cancelled' and [f.step for f in result.accepted] == [0, 1]
    assert result.histories['velocity-memory']['times'] == (0., .1)
    assert result.histories['timed-force']['steps'] == result.histories['velocity-memory']['steps'] == 1
    assert forcing.evaluations[-1].step_index == 2 and forcing.evaluations[-1].time == .2
    np.testing.assert_array_equal(forcing.evaluations[-1].previous_states['a'].world_velocity,
                                  result.accepted[-1].world_velocities['a'])


@pytest.mark.owner
def test_clock_snapshot_trials_initial_commit_refusal_and_legacy_hooks():
    b = body()
    state = b.state(acceleration=(2., 0., 0., 0., 0., 0.))
    provider = TimedForce()
    owner = ContributionSet(('a',), (provider,))
    with pytest.raises(MarineContributionsError, match='explicit clock'):
        owner.prepare_step()
    step = owner.prepare_step(StepInformation(0, .1, 'initialization'))
    trial = owner.evaluate_trial(step, {'a': state})
    with pytest.raises(MarineContributionsError, match='cannot commit'):
        owner.commit(step, trial)
    assert not len(owner.committed_history)
    owner.abort_step(step)
    source = {'a': state}
    information = StepInformation(1, .1, 'step', source)
    source.clear()
    with pytest.raises(FrozenInstanceError):
        information.time = 8.
    with pytest.raises(TypeError):
        information.previous_states['a'] = b.state()
    step = owner.prepare_step(information)
    first = owner.evaluate_trial(step, {'a': state})
    second = owner.evaluate_trial(step, {'a': b.state(position=(.1, 0., 0.))})
    with pytest.raises(MarineContributionsError, match='stale'):
        owner.commit(step, first)
    assert not len(owner.committed_history)
    owner.commit(step, second)
    assert owner.committed_history['timed-force']['steps'] == 1
    assert provider.evaluations[-1] is provider.evaluations[-2] is information
    class Legacy(DynamicContribution):
        contribution_id, ownership_key = 'legacy', 'test-legacy'
        def prepare_step(self, history):
            self.previous = history
        def evaluate_trial(self, history, states, layout):
            assert history is self.previous
            return output('legacy', history={'value': 1})
    legacy = ContributionSet(('a',), (Legacy(),))
    step = legacy.prepare_step()
    trial = legacy.evaluate_trial(step, {'a': state})
    legacy.commit(step, trial)
    assert legacy.committed_history['legacy']['value'] == 1


@pytest.mark.refusal
def test_context_refuses_invalid_clock_phase_layout_and_previous_states():
    state = body().state()
    valid = dict(step_index=1, dt=.1, phase='step', previous_states={'a': state})
    for changes in ({'step_index': True}, {'step_index': -1}, {'dt': True}, {'dt': 0.},
        {'dt': float('inf')}, {'dt': 10**1000}, {'step_index': 10**1000},
        {'phase': np.array(['step'])}, {'phase': 'initialization'}, {'step_index': 0},
        {'previous_states': {}}, {'previous_states': None}, {'previous_states': {'a': None}}):
        with pytest.raises(MarineContributionsError):
            StepInformation(**(valid|changes))
    with pytest.raises(MarineContributionsError):
        StepInformation(0, .1, 'initialization', {'a': state})
    owner = ContributionSet(('a',))
    with pytest.raises(MarineContributionsError):
        owner.prepare_step({})
    with pytest.raises(MarineContributionsError):
        owner.prepare_step(StepInformation(1, .1, 'step', {'b': state}))
    assert not len(owner.committed_history)
