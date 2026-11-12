"""Consequences of the private candidate in the production Newton loop.

Synthetic residuals isolate acceptance, work accounting and cancellation.
They do not establish physical chart-energy parity or checkpoint replay.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver.control import SolveCancelled
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite


def _driver(monkeypatch, responses, *, selected=True):
    owner = SimpleNamespace(
        _coordinate_support=False, _finite_distributing=False,
        _distributing_cyclic=False, _distributing_material=False,
        _curved_material=False,
        _bl5_material=False, _bl5_shell_material=False,
        _bl5_native_material=False, _free=np.array([0]),
        _fixed=np.array([], dtype=int), _K=np.zeros((1, 1)),
        _generation=0, _identity='synthetic-model')
    driver = object.__new__(finite.FinitePointDriver)
    driver.owner = owner
    driver._state = SimpleNamespace(index=0, multipliers=())
    driver._load_factor = 0.
    driver._adaptive_state = SimpleNamespace()
    driver._external = None
    driver._programme = None
    driver._paired_fibre_event = False
    driver._shell_origins = driver._beam_origins = ()
    driver._accepted = None
    driver._admissible_polish = selected
    driver._prepare_load = lambda actions, targets: ('pattern', None)
    driver._increment_origin = lambda state: state
    # Admission is a separate integration concern; this synthetic owner only
    # isolates the REAL _solve_fixed acceptance and publication routing.
    monkeypatch.setattr(finite, 'supports_native_chart_polish', lambda o: o is owner)
    seen, published = [], []

    def assemble(state, *args, **kwargs):
        seen.append(state.index)
        relative, gap = responses[min(state.index, len(responses)-1)]
        return (np.ones(1), sparse.eye(1, format='csr'),
                relative, gap, None, None)

    driver._assemble = assemble
    driver._advance = lambda state, step, amount: SimpleNamespace(
        index=state.index+1, multipliers=())

    def publish(state, evaluation, factor, **kwargs):
        row = kwargs['attempt']('accepted', kwargs['iteration'],
                                evaluation[2], evaluation[3])
        published.append((state.index, row, seen[-1]))
        driver._accepted = row

    driver._publish_equilibrium = publish
    return driver, seen, published


def test_default_stops_at_first_equilibrium(monkeypatch):
    driver, seen, published = _driver(monkeypatch, [(8e-9, 1e-10)], selected=False)
    driver._solve_fixed(None, (1.,), max_iterations=2, max_backtracks=0)
    assert seen == [0]
    assert published[0][1].linear_solves == 0


def test_selected_retains_improvement_and_counts_all_work(monkeypatch):
    driver, seen, published = _driver(monkeypatch,
        [(8e-9, 1e-10), (4e-9, 1e-10), (9e-9, 1e-10)])
    driver._solve_fixed(None, (1.,), max_iterations=2, max_backtracks=0)
    index, row, payload_index = published[0]
    assert index == payload_index == 1
    assert row.iterations == row.linear_solves == 2
    assert row.evaluations == len(seen)
    assert row.rejected_trials == 1
    assert seen[-1] == 1  # discarded candidate cannot supply recovery


@pytest.mark.parametrize('response', [(9e-9, 1e-10), (1e-9, 2e-9)])
def test_no_improvement_or_compatibility_violation_keeps_valid_state(monkeypatch, response):
    driver, seen, published = _driver(monkeypatch, [(8e-9, 1e-10), response])
    driver._solve_fixed(None, (1.,), max_iterations=1, max_backtracks=0)
    index, row, payload_index = published[0]
    assert index == payload_index == 0
    assert row.iterations == row.linear_solves == 1
    assert row.rejected_trials == 1
    assert row.evaluations == len(seen)
    assert row.equilibrium_ratio == 8e-9


def test_iteration_ceiling_reserve_publishes_retained_state(monkeypatch):
    driver, seen, published = _driver(monkeypatch,
        [(8e-9, 1e-10), (4e-9, 1e-10)])
    driver._solve_fixed(None, (1.,), max_iterations=1, max_backtracks=0)
    index, row, payload_index = published[0]
    # At the iteration ceiling only the one required retained-state
    # re-evaluation remains inside the replay ceiling: the bounded reserve
    # refuses the optional improvement and publishes the re-verified
    # retained state with truthful counters, never over the ceiling.
    assert index == payload_index == 0
    assert row.iterations == row.linear_solves == 1
    assert row.evaluations == len(seen) == 1 + (0 + 2) * 1
    assert row.rejected_trials == 0
    assert row.equilibrium_ratio == 8e-9


def test_worst_work_publishes_inside_the_replay_ceiling(monkeypatch):
    # Synthetic worst work of the actual caller loop: a full backtrack
    # ladder at every iteration with acceptance only at the last cutback,
    # so the polishing candidate spends the whole per-attempt evaluation
    # budget. The bounded reserve stops the last optional improvement and
    # publishes the re-verified retained state exactly at the controller
    # replay ceiling; the exact counters replay cleanly.
    from anyfesolver._stiffened_shell_increment_control import (
        AdaptiveIncrementControl, _AdaptiveState, validate_control)
    max_iterations, max_backtracks = 2, 1
    driver, seen, published = _driver(monkeypatch, [
        (8e-9, 1e-10),  # iteration 0 trial: first retained state
        (9e-9, 1e-10),  # full step refused: higher merit
        (4e-9, 1e-10),  # half step accepted: new retained state
        (9e-9, 1e-10),  # full step refused: higher merit
        (2e-9, 1e-10),  # half step the reserve refuses to re-verify
    ])
    # Distinct synthetic candidates per cutback, as in the real ladder: the
    # full Newton step and the halved cutback advance to different states.
    driver._advance = lambda state, step, amount: SimpleNamespace(
        index=state.index + (1 if amount == 1. else 2), multipliers=())
    driver._solve_fixed(None, (1.,), max_iterations=max_iterations,
                        max_backtracks=max_backtracks)
    index, row, payload_index = published[0]
    ceiling = 1 + (max_backtracks + 2) * max_iterations
    assert index == payload_index == 2  # re-verified retained state
    assert row.evaluations == len(seen) == ceiling  # honest, never over
    assert row.iterations == row.linear_solves == max_iterations
    assert row.rejected_trials == 2
    assert row.equilibrium_ratio == 4e-9 and row.constraint_max <= 1e-9
    state = _AdaptiveState(AdaptiveIncrementControl(2, 8), (1.,),
                           max_iterations, max_backtracks, 1, 1.,
                           attempts=(row,))
    validate_control(state, (0., 1.))  # controller replay accepts exact counters


def test_cancel_before_publish_preserves_prefix(monkeypatch):
    driver, _, published = _driver(monkeypatch,
        [(8e-9, 1e-10), (9e-9, 1e-10)])
    origin = driver._state

    def safe_point(token, label):
        if label == 'stiffened-shell.finite.commit':
            raise SolveCancelled('test cancellation')

    monkeypatch.setattr(finite, 'cancellation_safe_point', safe_point)
    with pytest.raises(SolveCancelled):
        driver._solve_fixed(None, (1.,), max_iterations=1, max_backtracks=0)
    assert not published and driver._accepted is None
    assert driver._state is origin and driver.owner._generation == 0


def test_private_checkpoint_identity_does_not_change_default(monkeypatch):
    driver, _, _ = _driver(monkeypatch, [(8e-9, 1e-10)], selected=False)
    assert driver._checkpoint_identity() == driver.owner._identity
    driver._admissible_polish = True
    assert driver._checkpoint_identity() == finite.admissible_polish_identity(
        driver.owner._identity)
    del driver._admissible_polish
    assert driver._checkpoint_identity() == driver.owner._identity


@pytest.mark.parametrize('staged,adaptive', [(False, False), (True, True)])
def test_selected_real_owner_checkpoint_round_trip_and_atomic_refusal(request, staged, adaptive):
    from anyfesolver import stiffened_shell as ss
    from test_stiffened_shell_native_plastic_chart import definition, programme

    model = definition()

    def owner(selected):
        result = ss.create_analysis(definition=model)
        request.addfinalizer(result.close)
        finite.select_admissible_polish_candidate(result, selected)
        return result

    selected = owner(True)
    loads = (programme(selected) if staged else
             (ss.NodalAction('beam', 105, (0., 0., 1000., 0., 0., 0.)),))
    options = dict(max_iterations=4, max_backtracks=2)
    if adaptive:
        options['increment_control'] = ss.AdaptiveIncrementControl(2, 8)
    selected.solve(loads, (.001,), **options)
    raw = selected.checkpoint()
    restored = owner(True)
    restored.restore(raw)
    assert restored.checkpoint() == raw
    unselected = owner(False)
    with pytest.raises(ValueError):
        unselected.restore(raw)
    assert unselected._accepted is None and unselected._finite_driver is None

    legacy = owner(False)
    legacy_loads = programme(legacy) if staged else loads
    legacy.solve(legacy_loads, (.001,), **options)
    legacy_raw = legacy.checkpoint()
    receiver = owner(True)
    with pytest.raises(ValueError):
        receiver.restore(legacy_raw)
    assert receiver._accepted is None and receiver._finite_driver is None

    # Actual chart framing, in addition to the plain and controller wrappers.
    if not staged:
        selected.rebase_rotations(selected.recover().generation)
        chart_raw = selected.checkpoint()
        chart_receiver = owner(True)
        chart_receiver.restore(chart_raw)
        assert chart_receiver.checkpoint() == chart_raw
