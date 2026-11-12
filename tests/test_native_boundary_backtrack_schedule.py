"""Call-path regression for the finite backtrack schedule near advisory events.

Drives the real FinitePointDriver._solve_fixed backtrack loop with mocked
assembly, advance and publication, so the trial schedule, budget accounting
and commit behaviour are observed on the actual call path rather than an
extracted helper. The mocked owner selects the native-material boundary gate
(owner._bl5_native_material) that admits the advisory first-event probe.
"""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse

from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure

# Observed equilibrium ratio of the saved failing prefix attempt; the exact
# value only fixes the merit scale, no tolerance depends on it.
OLD = 0.06382978723404312
MAX_BACKTRACKS = 12


def scheduled_driver(event, merit):
    """Real _solve_fixed over a mocked native-material point owner."""
    size = 3
    owner = SimpleNamespace(
        _coordinate_support=False, _finite_distributing=False,
        _distributing_cyclic=False, _curved_material=False,
        _distributing_material=False, _bl5_native_material=True,
        _bl5_material=False, _bl5_shell_material=False,
        _free=np.arange(size), _fixed=np.array([], dtype=int),
        _K=np.zeros((size, size)), _generation=0)
    driver = object.__new__(FinitePointDriver)
    driver.owner = owner
    driver._state = SimpleNamespace(multipliers=())
    driver._load_factor = 1.
    # A truthy pending marker keeps the elastic handoff path off, matching the
    # adaptive native-material configuration of the failing coupled solves.
    driver._adaptive_state = SimpleNamespace()
    driver._external = None
    driver._paired_fibre_event = False
    driver._shell_origins = ()
    driver._beam_origins = ()
    driver._accepted = None
    driver._prepare_load = lambda actions, targets: ("pattern", None)
    driver._increment_origin = lambda state: state
    driver._native_boundary_amount = lambda state, increment: event
    matrix = sparse.identity(size, format="csr")
    residual = np.full(size, 0.1)
    amounts, evaluations, published, seen = [], [], [], set()

    def advance(trial, full, amount):
        amounts.append(float(amount))
        return SimpleNamespace(amount=float(amount))

    def assemble(state, factor, shell_origins, beam_origins, pattern=None):
        evaluations.append(state)
        if state is not driver._state:
            if id(state) in seen:
                # The accepted trial re-checked at the next iteration converges.
                return (np.zeros(size), matrix, 1e-9, 1e-10, None, None)
            seen.add(id(state))
            return (residual, matrix, float(merit(float(state.amount))),
                    1e-12, None, None)
        return (residual, matrix, OLD, 1e-12, None, None)

    def publish(trial, evaluation, factor, **kwargs):
        published.append((trial, float(factor)))
        driver._accepted = SimpleNamespace(published=trial)

    driver._advance = advance
    driver._assemble = assemble
    driver._publish_equilibrium = publish
    return driver, amounts, evaluations, published


def test_tiny_rejected_event_does_not_starve_admissible_geometric_cut():
    def merit(amount):
        if amount <= 1e-6:
            # A near-zero move leaves the comparison merit unchanged.
            return OLD
        return OLD * (1. + amount) if amount == 1. else OLD * (1. - .5*amount)

    driver, amounts, evaluations, published = scheduled_driver(1e-12, merit)
    result = driver._solve_fixed(None, (1.25,), max_iterations=2,
                                 max_backtracks=MAX_BACKTRACKS)
    # Full trial, advisory probe, then the ordinary geometric cut 0.5 which is
    # admissible; the pre-fix schedule never sampled it and exhausted instead.
    assert amounts == [1., 1e-12, .5]
    assert [trial.amount for trial, _ in published] == [.5]
    assert result is driver._accepted
    # One origin, three backtrack trials, one re-check of the accepted trial.
    assert len(evaluations) == 5


def test_exhausted_budget_still_fails_without_partial_commit_or_extra_evaluations():
    def merit(amount):
        # Genuine no-descent: every positive amount increases the merit.
        return OLD if amount <= 1e-6 else OLD * (1. + amount)

    driver, amounts, evaluations, published = scheduled_driver(1e-12, merit)
    origin = driver._state
    with pytest.raises(FiniteIncrementFailure) as raised:
        driver._solve_fixed(None, (1.25,), max_iterations=2,
                            max_backtracks=MAX_BACKTRACKS)
    attempt = raised.value.attempt
    assert attempt.failure == "finite mixed backtrack budget exhausted"
    assert attempt.iterations == 1
    # Same trial budget as the pre-fix event schedule: the full trial, the
    # probe and the ordinary ladder, never more than max_backtracks+1 trials.
    assert amounts == [1., 1e-12] + [.5**(backtrack-1)
                                     for backtrack in range(2, MAX_BACKTRACKS+1)]
    assert attempt.evaluations == 1 + len(amounts) == 14
    assert attempt.rejected_trials == len(amounts) == 13
    assert attempt.linear_solves == 1
    # No partial commit: nothing published, no accepted result, no state or
    # generation advance.
    assert published == []
    assert driver._accepted is None
    assert driver._state is origin and driver.owner._generation == 0


@pytest.mark.parametrize("event", [None, .7])
def test_ordinary_schedule_unchanged_without_starving_event(event):
    def merit(amount):
        return OLD * (1. + amount) if amount >= .5 else OLD * (1. - .5*amount)

    driver, amounts, evaluations, published = scheduled_driver(event, merit)
    driver._solve_fixed(None, (1.25,), max_iterations=2,
                        max_backtracks=MAX_BACKTRACKS)
    # Byte-identical geometric ladder whenever no event narrows the first cut.
    assert amounts == [1., .5, .25]
    assert [trial.amount for trial, _ in published] == [.25]


def test_accepted_boundary_probe_keeps_existing_semantics():
    def merit(amount):
        return OLD * (1. + amount) if amount == 1. else OLD * (1. - .5*amount)

    driver, amounts, evaluations, published = scheduled_driver(.3, merit)
    driver._solve_fixed(None, (1.25,), max_iterations=2,
                        max_backtracks=MAX_BACKTRACKS)
    # The advisory probe still runs first and still accepts when admissible.
    assert amounts == [1., .3]
    assert [trial.amount for trial, _ in published] == [.3]
