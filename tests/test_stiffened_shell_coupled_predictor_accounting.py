"""Coupled predictor work integrity and scratch ownership regressions."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from anyfesolver._stiffened_shell_increment_control import (
    AdaptiveIncrementControl, FiniteIncrementFailure, IncrementAttempt,
    _AdaptiveState, validate_control)
from anyfesolver._stiffened_shell_ge_finite_experimental import (
    FinitePointDriver, _RotationStepTooLarge, BL5TrialDomainError,
    DistributingPoseDomainError)


def _accepted(factor, previous):
    return IncrementAttempt(factor=factor, accepted_from=previous, target=1.,
        status='accepted', iterations=1, evaluations=2, linear_solves=1,
        rejected_trials=0, equilibrium_ratio=0., constraint_max=0.)


def _probe_domain_history(evaluations=301):
    # Every Newton iteration may evaluate a probe whose local update fails
    # before a second linear solve. Preserve those failed evaluations.
    state=_AdaptiveState(AdaptiveIncrementControl(4,64), (1.,),20,12,0,0.)
    failed=IncrementAttempt(factor=1.,accepted_from=0.,target=1.,status='rejected',
        iterations=20,evaluations=evaluations,linear_solves=20,rejected_trials=260,
        equilibrium_ratio=1.,constraint_max=0.,failure='finite mixed Newton iteration limit')
    state=state.rejected(failed)
    assert state.next_factor(0.) == .5
    state=state.accepted(_accepted(.5,0.))
    assert state.next_factor(.5) == 1.
    return state.accepted(_accepted(1.,.5))


def test_probe_domain_work_restores_without_invented_linear_solves():
    state=_probe_domain_history()
    validate_control(state,(0.,.5,1.),_allow_coupled_predictor=True)
    assert state.attempts[0].linear_solves == 20
    assert state.attempts[0].evaluations == 301


@pytest.mark.parametrize('flags', ({},{'_allow_fibre_predictor':True}))
def test_other_profiles_keep_their_original_work_bounds(flags):
    with pytest.raises(ValueError,match='invalid adaptive attempt'):
        validate_control(_probe_domain_history(),(0.,.5,1.),**flags)


def test_coupled_probe_work_is_still_bounded():
    with pytest.raises(ValueError,match='invalid adaptive attempt'):
        validate_control(_probe_domain_history(302),(0.,.5,1.),_allow_coupled_predictor=True)


def _driver():
    driver=FinitePointDriver.__new__(FinitePointDriver)
    driver.owner=SimpleNamespace(_shell_count=0,_beam_nodal=0,
                                _beam_parts=(object(),))
    driver._beam_nodal=0
    return driver


def test_probe_only_changes_owned_material_pp_predictor_block():
    driver=_driver(); parent=sparse.eye(24,format='csr')
    sampled=2.*sparse.eye(24,format='csr'); trial=object(); probe=object()
    calls=[]; counts={'linear_solves':0}
    driver._advance=lambda state,direction,amount: (
        calls.append((state,direction.copy(),amount)) or probe)
    def evaluate(state):
        assert state is probe
        return (None,sampled)
    predicted,diagnostic=driver._coupled_probe_predictor(
        parent,np.ones(24),np.arange(24),counts,trial,np.ones(24),evaluate)
    np.testing.assert_array_equal(predicted[:6],-np.ones(6))
    np.testing.assert_array_equal(predicted[6:],-.5*np.ones(18))
    np.testing.assert_array_equal(parent.toarray(),np.eye(24))
    saved=predicted.copy(); sampled.data[:]=17.
    np.testing.assert_array_equal(predicted,saved)
    assert calls[0][0] is trial and calls[0][2] == 1e-5
    assert diagnostic['source']=='probe' and counts['linear_solves']==1


@pytest.mark.parametrize('failure',(_RotationStepTooLarge,BL5TrialDomainError,
                                     DistributingPoseDomainError))
@pytest.mark.parametrize('at_evaluation',(False,True))
def test_only_typed_probe_candidate_failures_return_no_direction(failure,at_evaluation):
    driver=_driver(); counts={'linear_solves':0}; trial=object()
    def advance(*args):
        if not at_evaluation: raise failure('candidate')
        return object()
    def evaluate(*args):
        raise failure('candidate')
    driver._advance=advance
    direction,status=driver._coupled_probe_predictor(None,None,None,counts,trial,None,evaluate)
    assert direction is None and status['status']=='probe-domain'
    assert counts['linear_solves']==0


@pytest.mark.parametrize('at_evaluation',(False,True))
def test_probe_programming_errors_propagate(at_evaluation):
    driver=_driver(); counts={'linear_solves':0}
    def advance(*args):
        if not at_evaluation: raise RuntimeError('programming')
        return object()
    def evaluate(*args):
        raise RuntimeError('programming')
    driver._advance=advance
    with pytest.raises(RuntimeError,match='programming'):
        driver._coupled_probe_predictor(None,None,None,counts,object(),None,evaluate)
    assert counts['linear_solves']==0
@pytest.mark.parametrize('flags', (
    {'_allow_fibre_predictor': True, '_allow_coupled_predictor': True},
    {'_allow_coupled_predictor': 1}, {'_allow_fibre_predictor': 1},
))
def test_predictor_flags_are_boolean_and_exclusive(flags):
    with pytest.raises(ValueError, match='invalid accepted adaptive controller'):
        validate_control(_probe_domain_history(), (0., .5, 1.), **flags)


def test_coupled_extra_solves_remain_bounded_per_iteration():
    from dataclasses import replace
    state = _probe_domain_history()
    corrupt = replace(state.attempts[0], linear_solves=41)
    with pytest.raises(ValueError, match='invalid adaptive attempt'):
        validate_control(replace(state, attempts=(corrupt,) + state.attempts[1:]),
                         (0., .5, 1.), _allow_coupled_predictor=True)


# --- Production routing of the coupled probe inside the real backtrack loop ---

OLD = 0.06382978723404312
MAX_BACKTRACKS = 12
SIZE = 24
PP_START = 6


def _coupled_schedule_driver(merit, *, advance_error=None, probe_merit=OLD):
    """Real _solve_fixed over a mocked coupled-material point owner.

    One beam part: the 18 pp slots are SIZE-PP_START..SIZE. The parent tangent
    is the identity and the probe tangent doubles the pp block, so the spliced
    predictor direction halves exactly the pp components of the parent step.
    """
    owner = SimpleNamespace(
        _coordinate_support=False, _finite_distributing=False,
        _distributing_cyclic=False, _curved_material=False,
        _distributing_material=False, _bl5_native_material=False,
        _bl5_material=True, _bl5_shell_material=True,
        _free=np.arange(SIZE), _fixed=np.array([], dtype=int),
        _K=np.zeros((SIZE, SIZE)), _generation=0,
        _shell_count=0, _beam_nodal=0, _beam_parts=(object(),))
    driver = object.__new__(FinitePointDriver)
    driver.owner = owner
    driver._beam_nodal = 0
    driver._state = SimpleNamespace(multipliers=())
    driver._load_factor = 1.
    # A truthy adaptive marker keeps the fixed-trial handoff off; the coupled
    # probe route itself is gated only by the owner material flags.
    driver._adaptive_state = SimpleNamespace()
    driver._external = None
    driver._paired_fibre_event = False
    driver._shell_origins = ()
    driver._beam_origins = ()
    driver._accepted = None
    driver._prepare_load = lambda actions, targets: ("pattern", None)
    driver._increment_origin = lambda state: state
    matrix = sparse.identity(SIZE, format="csr")
    probe = sparse.diags(
        np.r_[np.ones(PP_START), 2.*np.ones(SIZE-PP_START)]).tocsr()
    residual = np.ones(SIZE)
    amounts, directions, evaluations, published, seen = [], [], [], [], set()

    def advance(trial, full, amount):
        if advance_error is not None and amount == 1e-5:
            raise advance_error('probe')
        amounts.append(float(amount))
        directions.append(np.array(full, copy=True))
        return SimpleNamespace(amount=float(amount), is_probe=amount == 1e-5)

    def assemble(state, factor, shell_origins, beam_origins, pattern=None):
        evaluations.append(state)
        if state is driver._state:
            return (residual, matrix, OLD, 1e-12, None, None)
        if getattr(state, "is_probe", False):
            return (residual, probe, float(probe_merit), 1e-12, None, None)
        if id(state) in seen:
            # The accepted trial re-checked at the next iteration converges.
            return (np.zeros(SIZE), matrix, 1e-9, 1e-10, None, None)
        seen.add(id(state))
        return (residual, matrix, float(merit(float(state.amount))), 1e-12, None, None)

    def publish(trial, evaluation, factor, **kwargs):
        published.append((trial, float(factor)))
        driver._accepted = SimpleNamespace(published=trial)

    driver._advance = advance
    driver._assemble = assemble
    driver._publish_equilibrium = publish
    return driver, amounts, directions, evaluations, published


def test_probe_direction_keeps_remaining_ladder_and_probe_is_never_accepted():
    def merit(amount):
        return OLD*(1.+amount) if amount == 1. else OLD*(1.-.5*amount)

    driver, amounts, directions, evaluations, published = \
        _coupled_schedule_driver(merit)
    result = driver._solve_fixed(None, (1.25,), max_iterations=2,
                                 max_backtracks=MAX_BACKTRACKS)
    # Full parent trial, diagnostic probe at 1e-5, then the spliced direction
    # carries the whole remaining ladder.
    assert amounts == [1., 1e-5, .5]
    parent = -np.ones(SIZE)
    np.testing.assert_allclose(directions[0], parent)
    np.testing.assert_allclose(directions[1], parent)
    np.testing.assert_allclose(directions[2][:PP_START], parent[:PP_START])
    np.testing.assert_allclose(directions[2][PP_START:], .5*parent[PP_START:])
    # The probe evaluation is never accepted or published independently.
    assert [trial.amount for trial, _ in published] == [.5]
    assert all(not trial.is_probe for trial, _ in published)
    assert result is driver._accepted
    # Origin, rejected full trial, probe, accepted cut, re-check of the trial.
    assert len(evaluations) == 5
    assert driver._material_predictor_diagnostics == [dict(
        factor=1.25, iteration=0, status='finite',
        backend='scipy.sparse.linalg.spsolve', source='probe')]


def test_probe_domain_failure_at_production_hook_preserves_parent_direction():
    def merit(amount):
        return OLD*(1.+amount) if amount >= .5 else OLD*(1.-.5*amount)

    driver, amounts, directions, evaluations, published = \
        _coupled_schedule_driver(merit, advance_error=BL5TrialDomainError)
    driver._solve_fixed(None, (1.25,), max_iterations=2,
                        max_backtracks=MAX_BACKTRACKS)
    # The typed probe failure leaves the parent Newton direction in place; the
    # ordinary geometric ladder continues unchanged and still accepts.
    assert amounts == [1., .5, .25]
    for direction in directions:
        np.testing.assert_allclose(direction, -np.ones(SIZE))
    assert [trial.amount for trial, _ in published] == [.25]
    assert driver._material_predictor_diagnostics == [dict(
        factor=1.25, iteration=0, status='probe-domain',
        detail='BL5TrialDomainError')]


def test_cancellation_is_observed_around_the_probe_without_partial_commit():
    from anyfesolver.control import SolveCancelled

    class Token:
        def __init__(self, stage):
            self.stage = stage
        def raise_if_cancelled(self, stage):
            if stage == self.stage:
                raise SolveCancelled('probe cancel', stage=stage)

    def merit(amount):
        return OLD*(1.+amount) if amount == 1. else OLD*(1.-.5*amount)

    driver, amounts, directions, evaluations, published = \
        _coupled_schedule_driver(merit)
    origin = driver._state
    with pytest.raises(SolveCancelled):
        driver._solve_fixed(None, (1.25,),
                            cancellation_token=Token("stiffened-shell.finite.probe"),
                            max_iterations=2, max_backtracks=MAX_BACKTRACKS)
    # The full trial ran and was merit-rejected; cancellation fires before the
    # probe and nothing is published, accepted or recorded.
    assert amounts == [1.]
    assert published == [] and driver._accepted is None
    assert driver._state is origin and driver.owner._generation == 0
    assert driver._material_predictor_diagnostics == []


def test_coupled_probe_exhaustion_reports_truthful_counters_without_partial_commit():
    def merit(amount):
        return OLD*(1.+amount)

    driver, amounts, directions, evaluations, published = \
        _coupled_schedule_driver(merit)
    origin = driver._state
    with pytest.raises(FiniteIncrementFailure) as raised:
        driver._solve_fixed(None, (1.25,), max_iterations=2,
                            max_backtracks=MAX_BACKTRACKS)
    attempt = raised.value.attempt
    assert attempt.failure == "finite mixed backtrack budget exhausted"
    assert attempt.iterations == 1
    # Parent full trial, one probe, then the spliced direction over the whole
    # remaining ladder: never more than max_backtracks+1 backtrack trials.
    assert amounts == [1., 1e-5] + [.5**backtrack
                                    for backtrack in range(1, MAX_BACKTRACKS+1)]
    probe_direction = np.r_[-np.ones(PP_START), -.5*np.ones(SIZE-PP_START)]
    for direction in directions[2:]:
        np.testing.assert_allclose(direction, probe_direction)
    # Truthful work accounting: the probe evaluation and predictor solve are
    # counted, never hidden.
    assert attempt.evaluations == 1 + (MAX_BACKTRACKS+1) + 1
    assert attempt.rejected_trials == MAX_BACKTRACKS+1
    assert attempt.linear_solves == 2
    # No partial commit: nothing published, no accepted result, no state or
    # generation advance.
    assert published == [] and driver._accepted is None
    assert driver._state is origin and driver.owner._generation == 0
