"""Fixed short public-solver protocol cases, not physical qualification.

Fault cases intentionally inject invalid operators or loads. Run faults/history
selectors separately: four cases each, <=3steps, no random states.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from anyfesolver.marine_rigid import RigidBody6D
from anyfesolver.marine_contributions import DynamicContribution, ContributionResult, HISTORY_CHART
from anyfesolver.marine_mechanical import (
    MarineMechanicalModel, MarineMechanicalRunSpec, MarineMechanicalSolveError,
    solve_marine_mechanical)


def body():
    return RigidBody6D('a', 4., (0., 0., 0.), 2.*np.eye(3), (0., 0., 0.))


class StepFault(DynamicContribution):
    contribution_id = 'memory'
    ownership_key = 'test-only-memory'

    def __init__(self, fault=None):
        self.fault = fault

    def evaluate_trial(self, history, states, layout):
        previous = history.get('memory', {'steps': 0})['steps']
        force = np.zeros(6)
        force[0] = 4.
        mass = np.zeros((6, 6))
        if previous == 1 and self.fault:
            # Inconsistent derivative in iteration case is deliberate injection.
            force[0] = 8.+4.*states['a'].acceleration[0]
            if self.fault == 'singular':
                mass[0, 0] = -4.
        return ContributionResult('memory', force, np.zeros((6, 6)),
                                  np.zeros((6, 6)), mass, {}, {'steps': previous+1})


class ChartVector(DynamicContribution):
    contribution_id = 'memory'
    ownership_key = 'test-only-memory'
    history_basis = HISTORY_CHART
    world_vector = np.array((1., 2., 3.))

    def __init__(self, cancel_flag=None):
        self.cancel_flag = cancel_flag

    def evaluate_trial(self, history, states, layout):
        previous = history.get('memory')
        local = (np.asarray(states['a'].anchor).T @ self.world_vector if previous is None
                 else np.asarray(previous['local']))
        count = 0 if previous is None else previous['steps']
        return ContributionResult('memory', np.zeros(6), np.zeros((6, 6)),
            np.zeros((6, 6)), np.zeros((6, 6)), {}, {'steps': count+1, 'local': local})

    def history_transport(self, history, info):
        move = info.for_body('a')
        local = np.asarray(move.new_anchor).T @ np.asarray(move.old_anchor) @ history['local']
        if self.cancel_flag is not None and history['steps'] == 2:
            self.cancel_flag['cancel'] = True
        return {'steps': history['steps'], 'local': local}


@pytest.mark.faults
@pytest.mark.parametrize('reason', ('singular', 'iteration_limit'))
def test_late_solver_failure_preserves_committed_inventory(reason):
    b = body()
    provider = StepFault('singular' if reason == 'singular' else 'iteration')
    spec = MarineMechanicalRunSpec(.01, 3, max_newton_iterations=1, line_search=False)
    with pytest.raises(MarineMechanicalSolveError) as caught:
        solve_marine_mechanical(MarineMechanicalModel((b,), contributions=(provider,)),
                               {'a': b.state()}, spec)
    error = caught.value
    assert error.reason == reason and error.partial.status == 'failed'
    assert [frame.step for frame in error.partial.accepted] == [0, 1]
    assert error.histories['memory']['steps'] == 1
    assert error.partial.histories['memory']['steps'] == 1
    assert all(frame.residual_norm <= spec.residual_tolerance for frame in error.partial.accepted)


@pytest.mark.faults
@pytest.mark.parametrize('line_search', (False, True))
def test_chart_candidate_rejection_reports_actual_operation(line_search):
    class BranchLoad(StepFault):
        def evaluate_trial(self, history, states, layout):
            force = np.zeros(6)
            if states['a'].position[0] > .005:
                force[5] = 2e8
            return ContributionResult('memory', force, np.zeros((6, 6)),
                np.zeros((6, 6)), np.zeros((6, 6)), {}, {'steps': 1})
    b = body()
    initial = b.state(velocity=(1., 0., 0., 0., 0., 0.))
    with pytest.raises(MarineMechanicalSolveError) as caught:
        solve_marine_mechanical(MarineMechanicalModel((b,), contributions=(BranchLoad(),)),
            {'a': initial}, MarineMechanicalRunSpec(.01, 1, line_search=line_search))
    error = caught.value
    assert error.reason == ('line_search' if line_search else 'branch_violation')
    assert [frame.step for frame in error.partial.accepted] == [0]
    assert len(error.histories) == 0
    rejected = [log for log in error.logs if log.event == 'rejected']
    assert rejected and all('branch-violation' in log.detail for log in rejected)
    assert len([log for log in error.logs if log.step == 1 and log.event == 'trial']) == 1


@pytest.mark.history
def test_cancellation_after_chart_transport_preserves_previous_inventory():
    b = body()
    flag = {'cancel': False}
    provider = ChartVector(flag)
    initial = b.state(velocity=(0., 0., 0., .2, -.3, .1),
                      anchor=Rotation.from_rotvec((.2, .1, -.3)).as_matrix())
    result = solve_marine_mechanical(MarineMechanicalModel((b,), contributions=(provider,)),
        {'a': initial}, MarineMechanicalRunSpec(.01, 3), cancel=lambda: flag['cancel'])
    assert result.status == 'cancelled'
    assert [frame.step for frame in result.accepted] == [0, 1]
    assert result.histories['memory']['steps'] == 1
    expected = result.accepted[-1].rotations['a'].T @ provider.world_vector
    np.testing.assert_allclose(result.histories['memory']['local'], expected, rtol=0., atol=1e-10)


@pytest.mark.history
def test_nontrivial_chart_history_preserves_world_vector_and_residual():
    b = body()
    provider = ChartVector()
    initial = b.state(velocity=(0., 0., 0., .2, -.3, .1),
                      anchor=Rotation.from_rotvec((.2, .1, -.3)).as_matrix())
    spec = MarineMechanicalRunSpec(.01, 3)
    result = solve_marine_mechanical(MarineMechanicalModel((b,), contributions=(provider,)),
                                    {'a': initial}, spec)
    assert result.status == 'completed' and result.histories['memory']['steps'] == 3
    assert [frame.step for frame in result.accepted] == [0, 1, 2, 3]
    assert all(frame.residual_norm <= spec.residual_tolerance for frame in result.accepted)
    final = result.accepted[-1]
    assert np.linalg.norm(final.rotations['a']-result.accepted[0].rotations['a']) > 1e-3
    reconstructed = final.rotations['a'] @ result.histories['memory']['local']
    assert np.linalg.norm(reconstructed-provider.world_vector) <= 1e-10*max(np.linalg.norm(provider.world_vector), 1.)


@pytest.mark.history
def test_progress_exception_retains_the_committed_step():
    b = body()
    def progress(frame):
        if frame.step == 2:
            raise RuntimeError('injected progress failure')
    with pytest.raises(MarineMechanicalSolveError) as caught:
        solve_marine_mechanical(MarineMechanicalModel((b,), contributions=(StepFault(),)),
            {'a': b.state()}, MarineMechanicalRunSpec(.01, 3), progress=progress)
    error = caught.value
    assert error.partial.status == 'failed'
    assert [frame.step for frame in error.partial.accepted] == [0, 1, 2]
    assert error.histories['memory']['steps'] == 2
    assert 'injected progress failure' in str(error)


@pytest.mark.diagnostics
def test_nonfinite_initial_increment_has_nonfinite_reason(monkeypatch):
    b = body()
    calls = []
    def injected(matrix, rhs):
        calls.append((matrix.copy(), rhs.copy()))
        return np.full(6, np.nan)
    monkeypatch.setattr(np.linalg, 'solve', injected)
    with pytest.raises(MarineMechanicalSolveError) as caught:
        solve_marine_mechanical(MarineMechanicalModel((b,), contributions=(StepFault(),)),
            {'a': b.state()}, MarineMechanicalRunSpec(.01, 1))
    error = caught.value
    assert error.reason == 'nonfinite' and not error.partial.accepted
    assert not len(error.histories)
    assert str(error) == 'nonfinite initial acceleration Newton increment'
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.diag((4., 4., 4., 2., 2., 2.)))
    np.testing.assert_array_equal(calls[0][1], (4., 0., 0., 0., 0., 0.))


@pytest.mark.diagnostics
@pytest.mark.parametrize('overflow', ('norm', 'subtraction'))
def test_residual_overflow_is_nonfinite_with_uncommitted_evidence(overflow):
    class OverflowLoad(StepFault):
        def evaluate_trial(self, history, states, layout):
            force = np.zeros(6)
            if overflow == 'norm':
                force[:] = np.finfo(float).max
            else:
                force[0] = (1. if states['a'].acceleration[0] == 0. else -1.)*np.finfo(float).max
            return ContributionResult('memory', force, np.zeros((6, 6)),
                np.zeros((6, 6)), np.zeros((6, 6)), {}, {'steps': 1})
    b = body()
    with pytest.raises(MarineMechanicalSolveError) as caught:
        solve_marine_mechanical(MarineMechanicalModel((b,), contributions=(OverflowLoad(),)),
            {'a': b.state()}, MarineMechanicalRunSpec(.01, 1))
    error = caught.value
    assert error.reason == 'nonfinite' and not error.partial.accepted
    assert not len(error.histories)
    assert ('nonfinite residual norm' if overflow == 'norm' else
            'nonfinite residual or tangent during trial') == str(error)
