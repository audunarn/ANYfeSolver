"""Probe-window small-step acceptance at coupled backtrack exhaustion.

The coupled probe already pays for one fully evaluated candidate at its
fixed small distance. When strong local curvature makes every geometric
cut overshoot the merit descent window, that measured sample can be the
only decreasing candidate. These tests pin the production routing of the
window acceptance inside the real ``_solve_fixed`` backtrack loop: strict
measured decrease only, per-iteration staleness protection, cancellation
observation, truthful work accounting and no partial commit. The mocked
owner follows tests/test_stiffened_shell_coupled_predictor_accounting.py;
no fixture loads or defaults are exercised here.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver.control import SolveCancelled
from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
from anyfesolver._stiffened_shell_ge_finite_experimental import (
    FinitePointDriver, BL5TrialDomainError)

OLD = 0.06382978723404312
MAX_BACKTRACKS = 12
SIZE = 24
PP_START = 6
LADDER = [.5**backtrack for backtrack in range(1, MAX_BACKTRACKS+1)]


def _window_driver(*, probe_merit, merit, converge=None, probe_failure=None,
                   shell_material=True):
    """Real _solve_fixed over a mocked coupled-material point owner.

    One beam part: the 18 pp slots are SIZE-PP_START..SIZE. The parent
    tangent is the identity and the probe tangent doubles the pp block,
    exactly as in the coupled predictor accounting tests. ``converge``
    decides which re-evaluated trial satisfies the equilibrium check;
    ``probe_failure`` raises a typed domain error from the second probe
    advance onwards.
    """
    owner = SimpleNamespace(
        _coordinate_support=False, _finite_distributing=False,
        _distributing_cyclic=False, _curved_material=False,
        _distributing_material=False, _bl5_native_material=False,
        _bl5_material=True, _bl5_shell_material=shell_material,
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
    amounts, directions, evaluations, published = [], [], [], []
    probe_calls = []

    def advance(trial, full, amount):
        if amount == 1e-5:
            probe_calls.append(len(probe_calls)+1)
            if probe_failure is not None and len(probe_calls) >= 2:
                raise probe_failure('probe')
        amounts.append(float(amount))
        directions.append(np.array(full, copy=True))
        return SimpleNamespace(amount=float(amount), is_probe=amount == 1e-5)

    def assemble(state, factor, shell_origins, beam_origins, pattern=None):
        if converge is not None and converge(state, evaluations):
            evaluations.append(state)
            return (np.zeros(SIZE), matrix, 1e-9, 1e-10, None, None)
        evaluations.append(state)
        if state is driver._state:
            return (residual, matrix, OLD, 1e-12, None, None)
        if getattr(state, "is_probe", False):
            return (residual, probe, float(probe_merit), 1e-12, None, None)
        return (residual, matrix, float(merit(float(state.amount))), 1e-12, None, None)

    def publish(trial, evaluation, factor, **kwargs):
        published.append((trial, float(factor)))
        driver._accepted = SimpleNamespace(published=trial)

    driver._advance = advance
    driver._assemble = assemble
    driver._publish_equilibrium = publish
    return driver, amounts, directions, evaluations, published


def test_exhausted_ladder_accepts_the_measured_probe_window_decrease():
    def merit(amount):
        return OLD*(1.+amount)

    driver, amounts, directions, evaluations, published = _window_driver(
        probe_merit=OLD*(1.-1e-4), merit=merit,
        converge=lambda state, seen: getattr(state, "is_probe", False) and
        any(state is previous for previous in seen))
    origin = driver._state
    result = driver._solve_fixed(None, (1.25,), max_iterations=2,
                                 max_backtracks=MAX_BACKTRACKS)
    # The full trial, the diagnostic probe and the whole exhausted geometric
    # ladder ran unchanged; no extra cut was invented.
    assert amounts == [1., 1e-5] + LADDER
    probe_state = next(state for state in evaluations
                       if getattr(state, "is_probe", False))
    # The measured window sample becomes the next trial, and only the
    # ordinary loop-top equilibrium check publishes it.
    assert [trial for trial, _ in published] == [probe_state]
    assert published[0][1] == 1.25
    assert result is driver._accepted
    # Truthful accounting: the probe evaluation was already counted; the
    # window acceptance adds no evaluation or solve, only the ordinary
    # re-check of the accepted trial.
    assert len(evaluations) == 1 + 1 + 1 + MAX_BACKTRACKS + 1
    assert driver._material_predictor_diagnostics == [dict(
        factor=1.25, iteration=0, status='finite',
        backend='scipy.sparse.linalg.spsolve', source='probe'), dict(
        factor=1.25, iteration=0, status='probe-window-accept')]
    # No partial commit: the origin state and generation are untouched.
    assert driver._state is origin and driver.owner._generation == 0


def test_probe_window_requires_a_strict_measured_decrease():
    def merit(amount):
        return OLD*(1.+amount)

    driver, amounts, directions, evaluations, published = _window_driver(
        probe_merit=OLD*(1.+1e-4), merit=merit)
    origin = driver._state
    with pytest.raises(FiniteIncrementFailure) as raised:
        driver._solve_fixed(None, (1.25,), max_iterations=2,
                            max_backtracks=MAX_BACKTRACKS)
    attempt = raised.value.attempt
    assert attempt.failure == "finite mixed backtrack budget exhausted"
    assert attempt.iterations == 1
    # Identical counters to the pre-window exhaustion: the probe evaluation
    # and predictor solve stay counted, nothing is hidden or added.
    assert attempt.evaluations == 1 + 1 + 1 + MAX_BACKTRACKS
    assert attempt.rejected_trials == MAX_BACKTRACKS + 1
    assert attempt.linear_solves == 2
    assert [entry['status'] for entry
            in driver._material_predictor_diagnostics] == ['finite']
    assert published == [] and driver._accepted is None
    assert driver._state is origin and driver.owner._generation == 0


def test_stale_window_from_an_earlier_iteration_is_never_accepted():
    # The merit response is a pure function of the trial state: the .5 cut
    # measures .75*OLD when accepted in iteration 0 and re-measures the
    # identical .75*OLD when the loop re-evaluates that same accepted state.
    def merit(amount):
        if amount == .5:
            return OLD*(1.-.5*amount)
        return OLD*(1.+amount)

    driver, amounts, directions, evaluations, published = _window_driver(
        probe_merit=OLD*.5, merit=merit,
        probe_failure=BL5TrialDomainError)
    origin = driver._state
    with pytest.raises(FiniteIncrementFailure) as raised:
        driver._solve_fixed(None, (1.25,), max_iterations=2,
                            max_backtracks=MAX_BACKTRACKS)
    attempt = raised.value.attempt
    assert attempt.failure == "finite mixed backtrack budget exhausted"
    # Iteration 0: full trial, decreasing probe, accepted .5 cut. Iteration 1:
    # full trial, typed probe failure (no amount, no evaluation), exhausted
    # ladder; the re-measured .75*OLD reference refuses the equal .5 cut and
    # every smaller cut increases, so the second iteration exhausts.
    assert amounts == [1., 1e-5, .5, 1.] + LADDER
    # Iteration 0's probe record decreased against both iterations'
    # references (OLD and .75*OLD), so only the per-iteration staleness
    # protection can refuse it at this exhaustion; the failure stays truthful.
    assert driver._material_predictor_diagnostics == [dict(
        factor=1.25, iteration=0, status='finite',
        backend='scipy.sparse.linalg.spsolve', source='probe'), dict(
        factor=1.25, iteration=1, status='probe-domain',
        detail='BL5TrialDomainError')]
    # Truthful accounting across both iterations: the failed probe advance
    # evaluates nothing and adds no solve; both Newton solves and
    # iteration 0's splice solve stay counted, and the exhaustion reports
    # the re-measured current state.
    assert attempt.iterations == 2
    assert attempt.evaluations == 1 + 1 + 1 + 1 + 1 + 1 + MAX_BACKTRACKS
    assert attempt.rejected_trials == 2 + MAX_BACKTRACKS
    assert attempt.linear_solves == 3
    assert attempt.equilibrium_ratio == OLD*(1.-.5*.5)
    assert published == [] and driver._accepted is None
    assert driver._state is origin and driver.owner._generation == 0


def test_window_acceptance_observes_cancellation_without_partial_commit():
    class Token:
        def __init__(self):
            self.stage_calls = {}

        def raise_if_cancelled(self, stage):
            self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1
            if (stage == "stiffened-shell.finite.probe" and
                    self.stage_calls[stage] >= 3):
                raise SolveCancelled('window cancel', stage=stage)

    def merit(amount):
        return OLD*(1.+amount)

    token = Token()
    driver, amounts, directions, evaluations, published = _window_driver(
        probe_merit=OLD*(1.-1e-4), merit=merit)
    origin = driver._state
    with pytest.raises(SolveCancelled):
        driver._solve_fixed(None, (1.25,), cancellation_token=token,
                            max_iterations=2, max_backtracks=MAX_BACKTRACKS)
    # The ladder fully exhausted; the third probe-stage checkpoint sits in
    # the window path and fires before any acceptance is recorded.
    assert amounts == [1., 1e-5] + LADDER
    assert [entry['status'] for entry
            in driver._material_predictor_diagnostics] == ['finite']
    assert published == [] and driver._accepted is None
    assert driver._state is origin and driver.owner._generation == 0


def test_window_stays_off_the_uncoupled_route():
    def merit(amount):
        return OLD*(1.+amount)

    driver, amounts, directions, evaluations, published = _window_driver(
        probe_merit=OLD*(1.-1e-4), merit=merit, shell_material=False)
    with pytest.raises(FiniteIncrementFailure) as raised:
        driver._solve_fixed(None, (1.25,), max_iterations=2,
                            max_backtracks=MAX_BACKTRACKS)
    assert raised.value.attempt.failure == "finite mixed backtrack budget exhausted"
    # Without the coupled shell material there is no probe and no window;
    # the ordinary ladder and its truthful failure are unchanged.
    assert amounts == [1.] + LADDER
    assert driver._material_predictor_diagnostics == []
    assert published == [] and driver._accepted is None
