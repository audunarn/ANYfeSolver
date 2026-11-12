"""Focused W4 contribution-owner oracles: admission, atomic history, FD tangents.

Development evidence only; no qualification campaign. Numerical case
counts per invocation: 4 random analytic states + 1 fixed assembly state
(fixed prospectively in the governing W4 contract); protocol cases are
non-numerical.
"""
import numpy as np
from dataclasses import replace
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver.marine_rigid import RigidBody6D
from anyfesolver.marine_link import MechanicalLink6D
from anyfesolver.marine_contributions import (
    API_VERSION, BodyLayout, ChartMove, CommittedHistory, ContributionResult,
    ContributionSet, DryGravityContribution, DynamicContribution,
    HISTORY_CHART, HISTORY_PHYSICAL, MarineContributionsCancelled,
    MarineContributionsError, MechanicalLinkSetContribution, RechartInfo,
    TrialEvaluation, WorldWrenchContribution)


def close(actual, expected, tol=1e-10):
    assert np.linalg.norm(np.asarray(actual) - expected) <= tol*max(
        np.linalg.norm(expected), 1e-12)


def body():
    return RigidBody6D('alpha', 7., (1., -.4, .8),
                       ((4., .3, .2), (.3, 5., -.1), (.2, -.1, 6.)))


def state_for(b, rng, scale=.35):
    return b.state(position=rng.normal(size=3), rotation_vector=rng.normal(size=3)*scale,
                   velocity=rng.normal(size=6), anchor=Rotation.from_rotvec(
                       rng.normal(size=3)).as_matrix())


class RecordingContribution(DynamicContribution):
    """Protocol test double recording every trial's committed history."""

    def __init__(self, contribution_id='recorder', ownership_key='recorder',
                 basis=HISTORY_PHYSICAL, transport=None, history=None,
                 error=None, nonfinite=False, bad_shape=False, bad_id=False,
                 not_result=False, zero_mass=False):
        self.contribution_id = contribution_id
        self.ownership_key = ownership_key
        self.history_basis = basis
        self.history_transport = transport
        self._history = history if history is not None else {'steps': 0}
        self._error = error
        self._nonfinite = nonfinite
        self._bad_shape = bad_shape
        self._bad_id = bad_id
        self._not_result = not_result
        self._zero_mass = zero_mass
        self.seen_histories = []
        self.trials = 0

    def evaluate_trial(self, committed_history, states, layout):
        self.seen_histories.append(committed_history)
        self.trials += 1
        if self._error is not None:
            raise self._error
        size = layout.size
        force = np.zeros(size)
        if self._nonfinite:
            force[0] = np.inf
        shape = (size - 1, size) if self._bad_shape else (size, size)
        result_id = 'wrong-id' if self._bad_id else self.contribution_id
        if self._not_result:
            return {'not': 'a result'}
        mass = np.zeros((size, size))
        if self._zero_mass:
            mass[0, 0] = -1.
        previous = committed_history.get(self.contribution_id, self._history)
        history = {'steps': previous['steps'] + 1,
                    'payload': [self.trials]}
        return ContributionResult(result_id, force, np.zeros(shape),
                                  np.zeros(shape), mass, {'trials': self.trials},
                                  history)


def test_native_gravity_and_wrench_tangents_match_fd_4_states():
    rng = np.random.default_rng(20261011)
    b = body()
    gravity = (.4, -1.2, -9.81)
    point = (.2, .7, -.5)
    force = (3., -2., 1.5)
    moment = (-.6, .8, .2)
    h = 1e-5
    for kind in ('gravity', 'wrench'):
        contribution = (DryGravityContribution(b, gravity) if kind == 'gravity'
                         else WorldWrenchContribution('wrench-1', b, point, force, moment))
        owner = ContributionSet(('alpha',), (contribution,))
        for _ in range(2):
            state = state_for(b, rng)
            step = owner.prepare_step()
            trial = owner.evaluate_trial(step, {'alpha': state})
            if kind == 'gravity':
                expected = b.gravity_force(state, gravity)
            else:
                expected = (b.point_force(state, point, force)
                           + state.rate_map.T @ np.r_[np.zeros(3), np.asarray(moment)])
            close(trial.external_force, expected)
            for j in range(3):
                direction = np.zeros(3)
                direction[j] = 1.
                from dataclasses import replace
                plus = replace(state, rotation_vector=np.asarray(state.rotation_vector) + h*direction)
                minus = replace(state, rotation_vector=np.asarray(state.rotation_vector) - h*direction)
                if kind == 'gravity':
                    q_plus, q_minus = b.gravity_force(plus, gravity), b.gravity_force(minus, gravity)
                else:
                    q_plus = (b.point_force(plus, point, force)
                              + plus.rate_map.T @ np.r_[np.zeros(3), np.asarray(moment)])
                    q_minus = (b.point_force(minus, point, force)
                               + minus.rate_map.T @ np.r_[np.zeros(3), np.asarray(moment)])
                fd = (q_plus - q_minus)/(2*h)
                # K = -dQext/dq: rotational rows only; translation rows vanish.
                close(fd[3:], -trial.stiffness[3:6, 3+j], 1e-8)
                assert np.linalg.norm(fd[:3]) <= 1e-9
            assert np.linalg.norm(trial.stiffness[:3, :]) == 0.
            assert np.linalg.norm(trial.stiffness[:, :3]) == 0.
            assert np.linalg.norm(trial.damping) == 0.
            assert np.linalg.norm(trial.mass) == 0.
            assert trial.component_forces[contribution.contribution_id].shape == (6,)
            close(trial.component_forces[contribution.contribution_id], expected)
            owner.commit(step, trial)


def test_native_link_set_assembly_matches_direct_evaluation():
    b1, b2 = body(), RigidBody6D('beta', 3., (-.2, .5, .1),
                                  ((2., .1, 0.), (.1, 3., .2), (0., .2, 4.)))
    rng = np.random.default_rng(5)
    K = rng.normal(size=(6, 6))
    K = K @ K.T + np.eye(6)
    link = MechanicalLink6D('link-1', 'alpha', 'beta', (.5, 0., .2), (-.3, .1, 0.),
                             np.eye(3), Rotation.from_rotvec([.2, 0., -.1]).as_matrix(),
                             (1., 0., 0.), np.eye(3), K, .05*np.eye(6))
    owner = ContributionSet(('alpha', 'beta'),
                            (MechanicalLinkSetContribution((link,)),))
    sa = b1.state(position=(.1, .2, -.3), rotation_vector=(.05, -.02, .03),
                  velocity=(.1, -.2, .3, .01, .02, -.01))
    sb = b2.state(position=(1.1, -.2, .4), rotation_vector=(-.03, .04, .01),
                  velocity=(-.05, .1, .2, .0, -.02, .03))
    step = owner.prepare_step()
    trial = owner.evaluate_trial(step, {'alpha': sa, 'beta': sb})
    direct = link.evaluate(sa, sb)
    close(trial.external_force[:6], direct.chart_force_a)
    close(trial.external_force[6:], direct.chart_force_b)
    columns = list(range(12))
    close(trial.stiffness[np.ix_(columns, columns)], direct.stiffness)
    close(trial.damping[np.ix_(columns, columns)], direct.damping)
    assert np.linalg.norm(trial.mass) == 0.
    assert trial.diagnostics['mechanical-links']['link-1:potential_energy'] == (
        direct.potential_energy)
    owner.commit(step, trial)


def test_admission_refusals():
    good = RecordingContribution()
    with pytest.raises(MarineContributionsError):
        ContributionSet((), (good,))
    with pytest.raises(MarineContributionsError):
        ContributionSet(('alpha', 'alpha'), (good,))
    with pytest.raises(MarineContributionsError):
        ContributionSet(('alpha',), ('not-a-contribution',))
    with pytest.raises(MarineContributionsError):
        ContributionSet(('alpha',), (RecordingContribution(), RecordingContribution()))
    with pytest.raises(MarineContributionsError):
        ContributionSet(('alpha',), (RecordingContribution(),
                                     RecordingContribution('other', ownership_key='recorder')))
    with pytest.raises(MarineContributionsError):
        ContributionSet(('alpha',), (RecordingContribution(ownership_key='dry-mass:x'),))
    with pytest.raises(MarineContributionsError):
        ContributionSet(('alpha',), (RecordingContribution(basis='galactic'),))
    with pytest.raises(MarineContributionsError):
        ContributionSet(('alpha',), (RecordingContribution(basis=HISTORY_CHART),))
    with pytest.raises(MarineContributionsError):
        ContributionSet(('alpha',), (), cancel='not-callable')
    with pytest.raises(MarineContributionsError):
        BodyLayout(('alpha',)).index('beta')


def test_same_committed_history_across_rejected_trials_and_tokens():
    b = body()
    contribution = RecordingContribution()
    owner = ContributionSet(('alpha',), (contribution,))
    state = b.state()
    step = owner.prepare_step()
    trial_one = owner.evaluate_trial(step, {'alpha': state})
    assert trial_one.trial_token != step.token
    trial_two = owner.evaluate_trial(step, {'alpha': state})
    assert trial_two.trial_token != trial_one.trial_token
    # Both trials of the step read the SAME committed history object.
    assert contribution.seen_histories[0] is contribution.seen_histories[1]
    assert contribution.seen_histories[0] is owner.committed_history
    # The rejected first trial cannot commit: its token is stale.
    with pytest.raises(MarineContributionsError):
        owner.commit(step, trial_one)
    committed = owner.commit(step, trial_two)
    assert committed['recorder']['steps'] == 1
    assert owner.committed_history['recorder']['steps'] == 1
    # A committed step accepts no further trials or commits.
    with pytest.raises(MarineContributionsError):
        owner.evaluate_trial(step, {'alpha': state})
    with pytest.raises(MarineContributionsError):
        owner.commit(step, trial_two)
    # The next step reads the newly committed history.
    step_two = owner.prepare_step()
    owner.evaluate_trial(step_two, {'alpha': state})
    assert contribution.seen_histories[2]['recorder']['steps'] == 1


def test_commit_refusals_stale_double_and_foreign_steps():
    b = body()
    owner = ContributionSet(('alpha',), (RecordingContribution(),))
    state = b.state()
    step = owner.prepare_step()
    trial = owner.evaluate_trial(step, {'alpha': state})
    foreign_owner = ContributionSet(('alpha',), (RecordingContribution(),))
    foreign = foreign_owner.prepare_step()
    with pytest.raises(MarineContributionsError):
        owner.commit(foreign, trial)
    with pytest.raises(MarineContributionsError):
        owner.commit(step, 'not-a-trial')
    with pytest.raises(MarineContributionsError):
        owner.commit(step, trial, rechart_info='not-rechart')
    committed = owner.commit(step, trial)
    assert committed is owner.committed_history
    # A trial from a foreign step can never commit into another step.
    step_two = owner.prepare_step()
    trial_two = owner.evaluate_trial(step_two, {'alpha': state})
    with pytest.raises(MarineContributionsError):
        owner.commit(step_two, trial)
    with pytest.raises(MarineContributionsError):
        owner.commit(step_two, replace(trial_two))
    with pytest.raises(MarineContributionsError):
        owner.prepare_step()
    # Wrong states mapping is refused before any contribution runs.
    with pytest.raises(MarineContributionsError):
        owner.evaluate_trial(step_two, {'beta': state})
    with pytest.raises(MarineContributionsError):
        owner.evaluate_trial(step_two, {'alpha': state, 'beta': state})


def test_committed_history_immutable_and_deep_copied():
    source = {'payload': [1, 2], 'nested': {'a': 1}}
    history = CommittedHistory(source)
    source['payload'].append(3)
    source['extra'] = True
    assert list(history['payload']) == [1, 2]
    assert 'extra' not in history
    replaced = history.replaced({'payload': [9]})
    assert list(replaced['payload']) == [9]
    assert list(history['payload']) == [1, 2]
    with pytest.raises(TypeError):
        history['payload'] = [4]
    # Trial histories are deep-copied at commit; later mutation is impossible.
    b = body()
    contribution = RecordingContribution()
    owner = ContributionSet(('alpha',), (contribution,))
    step = owner.prepare_step()
    trial = owner.evaluate_trial(step, {'alpha': b.state()})
    original = trial.trial_histories['recorder']
    owner.commit(step, trial)
    with pytest.raises(TypeError):
        original['steps'] = 99
    with pytest.raises(AttributeError):
        original['payload'].append('mutated')
    assert owner.committed_history['recorder']['steps'] == 1
    assert owner.committed_history['recorder']['payload'] == (1,)


def test_mid_trial_cancellation_preserves_history():
    b = body()
    calls = {'cancelled': False}

    def cancel():
        return calls['cancelled']

    owner = ContributionSet(('alpha',), (RecordingContribution(),
                                         RecordingContribution('second', 'second')),
                            cancel=cancel)
    state = b.state()
    step = owner.prepare_step()
    trial = owner.evaluate_trial(step, {'alpha': state})
    owner.commit(step, trial)
    assert owner.committed_history['recorder']['steps'] == 1
    step = owner.prepare_step()
    class CancelDuringTrial(RecordingContribution):
        def evaluate_trial(self, history, states, layout):
            result = super().evaluate_trial(history, states, layout)
            calls['cancelled'] = True
            return result
    cancelling_owner = ContributionSet(('alpha',), (CancelDuringTrial(),), cancel=cancel)
    cancelling_step = cancelling_owner.prepare_step()
    with pytest.raises(MarineContributionsCancelled):
        cancelling_owner.evaluate_trial(cancelling_step, {'alpha': state})
    assert not len(cancelling_owner.committed_history)
    # The cancelled trial changed nothing.
    assert owner.committed_history['recorder']['steps'] == 1
    with pytest.raises(MarineContributionsCancelled):
        owner.evaluate_trial(step, {'alpha': state})


def test_nested_history_arrays_and_trial_arrays_are_immutable():
    array = np.arange(3.)
    history = CommittedHistory({'a': array, 'nested': {'values': [1, 2]}})
    array[:] = 99
    assert tuple(history['a']) == (0., 1., 2.)
    with pytest.raises(ValueError):
        history['a'].setflags(write=True)
    with pytest.raises(TypeError):
        history['nested']['values'] = (3,)
    with pytest.raises(TypeError):
        history._mapping = {}
    owner = ContributionSet(('alpha',), (RecordingContribution(),))
    step = owner.prepare_step()
    trial = owner.evaluate_trial(step, {'alpha': body().state()})
    with pytest.raises(ValueError):
        trial.external_force.setflags(write=True)
    owner.abort_step(step)
    with pytest.raises(MarineContributionsError):
        owner.commit(step, trial)


def test_transport_failure_or_late_cancel_never_partially_commits():
    cancelled = {'value': False}
    def transport(history, info):
        cancelled['value'] = True
        return history
    owner = ContributionSet(('alpha',), (
        RecordingContribution(), RecordingContribution('chart', 'chart',
            basis=HISTORY_CHART, transport=transport)), cancel=lambda: cancelled['value'])
    step = owner.prepare_step()
    trial = owner.evaluate_trial(step, {'alpha': body().state()})
    with pytest.raises(MarineContributionsCancelled):
        owner.commit(step, trial, RechartInfo(()))
    assert not len(owner.committed_history)
    cancelled['value'] = False
    with pytest.raises(MarineContributionsError):
        owner.commit(step, trial, RechartInfo(()))


def test_failed_provider_nonfinite_and_malformed_results():
    b = body()
    state = b.state()
    for kwargs in ({'error': RuntimeError('provider failed')},
                   {'nonfinite': True}, {'bad_shape': True},
                   {'bad_id': True}, {'not_result': True}):
        owner = ContributionSet(('alpha',), (RecordingContribution(**kwargs),))
        step = owner.prepare_step()
        with pytest.raises((RuntimeError, MarineContributionsError)):
            owner.evaluate_trial(step, {'alpha': state})
        assert len(owner.committed_history) == 0


def test_chart_history_transport_on_commit():
    b = body()
    transported = []

    def transport(history, rechart_info):
        transported.append((history, rechart_info))
        return {'steps': history['steps'], 'moved': True}

    physical = RecordingContribution('physical', 'physical')
    charted = RecordingContribution('charted', 'charted', basis=HISTORY_CHART,
                                    transport=transport)
    owner = ContributionSet(('alpha',), (physical, charted))
    state = b.state()
    step = owner.prepare_step()
    trial = owner.evaluate_trial(step, {'alpha': state})
    move = ChartMove('alpha', tuple(state.rotation_vector),
                     tuple(tuple(row) for row in state.anchor),
                     (tuple(np.eye(3)[0]), tuple(np.eye(3)[1]), tuple(np.eye(3)[2])))
    rechart = RechartInfo((move,))
    owner.commit(step, trial, rechart_info=rechart)
    assert owner.committed_history['physical']['steps'] == 1
    assert 'moved' not in owner.committed_history['physical']
    assert owner.committed_history['charted'] == {'steps': 1, 'moved': True}
    assert transported[0][0]['steps'] == 1
    assert transported[0][1] is rechart
    assert transported[0][1].for_body('alpha') is move
    assert transported[0][1].for_body('beta') is None
    assert API_VERSION == 'marine_contributions_v1'
    assert isinstance(trial, TrialEvaluation)
