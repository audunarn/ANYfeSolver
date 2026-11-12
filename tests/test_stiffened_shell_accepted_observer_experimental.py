"""Postcommit finite observers; lifecycle checks, not mechanics qualification."""

from dataclasses import FrozenInstanceError, fields, is_dataclass
import json

import numpy as np
import pytest

from anyfesolver import stiffened_shell
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_ge_checkpoint_binary import MAGIC as BINARY_MAGIC, decode
from anyfesolver._stiffened_shell_increment_control import decode_control
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_finite_experimental import _q4_shell_j2_panel


ACTIONS = (stiffened_shell.NodalAction("beam", 105, (0., 0., 1000., 0., 0., 0.)),)
TARGETS = (.25, .5)


def _options(adaptive):
    return {"increment_control": stiffened_shell.AdaptiveIncrementControl(2, 8)} if adaptive else {}


def _arrays(value, path=()):
    """Enumerate arrays in this bounded typed result, including material records."""
    if isinstance(value, np.ndarray):
        yield path, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _arrays(getattr(value, field.name), path + (field.name,))
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _arrays(item, path + (index,))


def _committed(owner, notified, adaptive, requested_targets=TARGETS):
    published = owner.recover()
    assert notified is not published
    assert notified.generation == published.generation == owner._generation
    assert notified.accepted_factors == published.accepted_factors
    assert notified.load_factor == published.load_factor
    assert notified.status == published.status
    saved = owner.checkpoint()
    assert owner.checkpoint() == saved
    inner = saved
    if adaptive:
        controller, inner = decode_control(saved, owner._identity)
        assert controller == owner._finite_driver._adaptive_state
        assert controller.targets == requested_targets
        assert controller.attempts == notified.increment_attempts
        assert tuple(row.factor for row in controller.attempts if row.status == "accepted") == notified.accepted_factors
        assert controller.cursor == sum(target in notified.accepted_factors for target in requested_targets)
        assert controller.completed == (notified.load_factor == requested_targets[-1])
        assert notified.status == ("completed" if controller.completed else "accepted_prefix")
    else:
        assert notified.increment_attempts == ()
        assert owner._finite_driver._adaptive_state is None
    if inner.startswith(BINARY_MAGIC):
        # Both layered material and stable elastic point fixtures use binary
        # framing; assert the complete array layout for the selected codec.
        def shapes(kind, generation):
            assert kind in ("finite_shell_material", "finite_stable_q4")
            count, members = owner._K.shape[0], len(owner._beam_parts)
            stations = sum(len(part[0].operator.cell.stations) for part in owner._beam_parts)
            arrays = (("q", (count,)), ("frames", (len(owner._beam_ids), 3, 3)),
                    ("cells", (members, 2, 3, 3)), ("resultants", (members, 18)),
                    ("multipliers", (owner._C.shape[0],)), ("pattern", (count,)),
                    ("accepted_factors", (generation,)))
            if kind == "finite_stable_q4":
                return arrays
            return arrays + (("beam_plastic", (stations, 6, 2)),
                    ("beam_accumulated", (stations, 2)),
                    ("shell_plastic", (len(owner._shell_parts), 12, 3)),
                    ("shell_alpha", (len(owner._shell_parts), 12)))
        _, generation, arrays = decode(inner, owner._identity, shapes)
        payload = {"generation": generation, **arrays}
    else:
        payload = json.loads(inner)
    assert payload["generation"] == notified.generation
    assert tuple(payload["accepted_factors"]) == notified.accepted_factors
    np.testing.assert_array_equal(np.asarray(payload["q"])[:owner._shell_count],
                                  published.shell_displacement)
    for name, recovered_name in (("frames", "beam_frames"), ("cells", "cell_frames"),
                                 ("resultants", "cell_resultants"), ("multipliers", "multipliers")):
        np.testing.assert_array_equal(payload[name], getattr(published, recovered_name))
    return published, saved


@pytest.mark.parametrize("adaptive", (False, True), ids=("fixed", "adaptive"))
def test_accepted_observer_once_detached_and_committed_defaults_unchanged(adaptive, request):
    # Existing material fixture also exercises nested layer-response arrays.
    # Small physical forcing keeps this ownership/default check independent of
    # plastic convergence or genuine adaptive substeps (tested separately).
    observer_actions = (stiffened_shell.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    definition = _q4_shell_j2_panel()
    owner = stiffened_shell.create_analysis(definition=definition)
    oracle = stiffened_shell.create_analysis(definition=definition)
    request.addfinalizer(owner.close)
    request.addfinalizer(oracle.close)
    notifications, retained = [], []

    def observe(notified):
        published, saved = _committed(owner, notified, adaptive)
        assert notified.accepted_factors == TARGETS[:len(notifications) + 1]
        assert notified.generation == len(notifications) + 1
        with pytest.raises(FrozenInstanceError):
            notified.generation = -1
        public_arrays = dict(_arrays(published))
        copied_arrays = dict(_arrays(notified))
        assert set(copied_arrays) == set(public_arrays)
        assert any(path[0] == "shell_q4_material" for path in copied_arrays)
        originals = {path: array.copy() for path, array in public_arrays.items()}
        state = owner._finite_driver._state
        for path, array in copied_arrays.items():
            np.testing.assert_array_equal(array, public_arrays[path])
            assert not np.shares_memory(array, public_arrays[path])
            for internal in (state.q, state.frames, state.cells, state.resultants, state.multipliers):
                assert not np.shares_memory(array, internal)
            if array.size:
                if array.flags.writeable:
                    array.flat[0] += 123.0  # Observer changes its own detached copy.
                else:
                    with pytest.raises(ValueError):
                        array.flat[0] += 123.0
        for path, array in public_arrays.items():
            np.testing.assert_array_equal(array, originals[path])
        assert owner.checkpoint() == saved
        retained.append((published, originals))
        notifications.append(notified)

    result = owner.solve(observer_actions, TARGETS, on_accepted=observe, **_options(adaptive))
    assert len(notifications) == len(TARGETS)
    assert result is owner.recover() and result.status == "completed"
    assert [row.load_factor for row in notifications] == list(TARGETS)
    assert len({id(row) for row in notifications}) == len(TARGETS)
    # Later publication cannot change an earlier public snapshot.
    for published, originals in retained:
        for path, array in _arrays(published):
            np.testing.assert_array_equal(array, originals[path])
    for path, first in _arrays(notifications[0]):
        assert not np.shares_memory(first, dict(_arrays(notifications[1]))[path])
    default_result = oracle.solve(observer_actions, TARGETS, **_options(adaptive))
    assert default_result.accepted_factors == result.accepted_factors
    assert oracle.checkpoint() == owner.checkpoint()
    for path, array in _arrays(result):
        np.testing.assert_array_equal(array, dict(_arrays(default_result))[path])


@pytest.mark.parametrize("adaptive", (False, True), ids=("fixed", "adaptive"))
@pytest.mark.parametrize("error_kind", ("programming", "finite-failure", "memory"))
def test_observer_exception_propagates_without_retry_preserves_commit(
        adaptive, error_kind, monkeypatch, request):
    owner = stiffened_shell.create_analysis(definition=_panel())
    request.addfinalizer(owner.close)
    calls, notifications = [], []
    original = FinitePointDriver._solve_fixed

    def counted(driver, actions, targets, **kwargs):
        calls.append(tuple(targets))
        return original(driver, actions, targets, **kwargs)

    monkeypatch.setattr(FinitePointDriver, "_solve_fixed", counted)
    if error_kind == "finite-failure":
        # A deliberate observer exception, not an injected mechanics failure.
        error = stiffened_shell.FiniteIncrementFailure(stiffened_shell.IncrementAttempt(
            .25, 0., .25, "rejected", 1, 2, 1, 1, 1., 0.,
            "finite mixed Newton iteration limit"))
    elif error_kind == "memory":
        error = MemoryError("observer allocation failure")
    else:
        error = RuntimeError("observer programming failure")

    def observe(notified):
        published, saved = _committed(owner, notified, adaptive)
        notifications.append((published, saved))
        raise error

    with pytest.raises(type(error)) as caught:
        owner.solve(ACTIONS, TARGETS, on_accepted=observe, **_options(adaptive))
    assert caught.value is error
    assert len(calls) == len(notifications) == 1
    published, saved = notifications[0]
    assert owner.recover() is published
    assert published.accepted_factors == TARGETS[:1] and published.generation == 1
    assert owner.checkpoint() == saved
    if adaptive:
        assert [row.status for row in published.increment_attempts] == ["accepted"]
        assert owner._finite_driver._adaptive_state.depth == 0
    # Exception propagation releases the writer guard; continuation is permitted.
    targets = TARGETS if adaptive else TARGETS[1:]
    continued = owner.solve(ACTIONS, targets, **_options(adaptive))
    assert continued.accepted_factors == TARGETS and continued.generation == 2
    assert len(notifications) == 1


def test_adaptive_observer_notifies_accepted_substeps_never_rejected_trials(monkeypatch, request):
    owner = stiffened_shell.create_analysis(definition=_panel())
    request.addfinalizer(owner.close)
    original, attempts, notifications = FinitePointDriver._solve_fixed, [], []

    def bounded(driver, actions, targets, **kwargs):
        factor = targets[0]
        attempts.append((factor, driver._load_factor, driver.owner._generation))
        if factor - driver._load_factor > .251:
            # Scheduling-only controlled rejection; accepted steps solve normally.
            raise stiffened_shell.FiniteIncrementFailure(stiffened_shell.IncrementAttempt(
                float(factor), float(driver._load_factor), float(factor), "rejected",
                1, 2, 1, 1, 1., 0., "finite mixed backtrack budget exhausted"))
        return original(driver, actions, targets, **kwargs)

    monkeypatch.setattr(FinitePointDriver, "_solve_fixed", bounded)

    def observe(notified):
        published, saved = _committed(owner, notified, True, requested_targets=(.5,))
        notifications.append((notified, published, saved))

    result = owner.solve(ACTIONS, (.5,), on_accepted=observe,
                         increment_control=stiffened_shell.AdaptiveIncrementControl(2, 8))
    assert attempts == [(.5, 0., 0), (.25, 0., 0), (.5, .25, 1)]
    assert [notified.load_factor for notified, _, _ in notifications] == [.25, .5]
    assert [notified.generation for notified, _, _ in notifications] == [1, 2]
    assert [row.status for row in notifications[0][0].increment_attempts] == ["rejected", "accepted"]
    assert [row.status for row in result.increment_attempts] == ["rejected", "accepted", "accepted"]
    assert result is owner.recover() and result.accepted_factors == TARGETS


@pytest.mark.parametrize("adaptive", (False, True), ids=("fixed", "adaptive"))
def test_observer_cancellation_stops_next_increment_with_committed_prefix(adaptive, request):
    owner = stiffened_shell.create_analysis(definition=_panel())
    request.addfinalizer(owner.close)
    token, notifications = CancellationToken(), []

    def observe(notified):
        notifications.append(_committed(owner, notified, adaptive))
        token.cancel("observer accepted-prefix cancellation")

    with pytest.raises(SolveCancelled, match="observer accepted-prefix cancellation"):
        owner.solve(ACTIONS, TARGETS, cancellation_token=token,
                    on_accepted=observe, **_options(adaptive))
    assert len(notifications) == 1
    published, saved = notifications[0]
    assert owner.recover() is published
    assert published.generation == 1 and published.accepted_factors == TARGETS[:1]
    assert owner.checkpoint() == saved
    if adaptive:
        controller, _ = decode_control(saved, owner._identity)
        assert controller.cursor == 1 and not controller.completed
        assert [row.status for row in controller.attempts] == ["accepted"]
    targets = TARGETS if adaptive else TARGETS[1:]
    assert owner.solve(ACTIONS, targets, **_options(adaptive)).accepted_factors == TARGETS


@pytest.mark.parametrize("adaptive", (False, True), ids=("fixed", "adaptive"))
def test_observer_reentrant_writers_rejected_reads_remain_available(adaptive, request):
    owner = stiffened_shell.create_analysis(definition=_panel())
    request.addfinalizer(owner.close)
    notifications = []

    def observe(notified):
        published, saved = _committed(owner, notified, adaptive)
        writers = (
            lambda: owner.solve(ACTIONS, (.75,)),
            lambda: owner.solve_linear(ACTIONS),
            lambda: owner.rebase_rotations(expected_generation=notified.generation),
            lambda: owner.restore(saved),
            owner.close,
        )
        for write in writers:
            with pytest.raises(RuntimeError, match="reentrant mixed state mutation is not permitted"):
                write()
            assert owner.recover() is published
            assert owner.checkpoint() == saved
        notifications.append(notified.generation)

    result = owner.solve(ACTIONS, TARGETS, on_accepted=observe, **_options(adaptive))
    assert notifications == [1, 2]
    assert result is owner.recover() and result.accepted_factors == TARGETS


@pytest.mark.parametrize("observer", (0, "observer", object()), ids=("number", "text", "object"))
def test_noncallable_observer_rejected_before_driver_or_numerics(observer, monkeypatch, request):
    owner = stiffened_shell.create_analysis(definition=_panel())
    request.addfinalizer(owner.close)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid observer reached finite driver construction")

    monkeypatch.setattr(FinitePointDriver, "__init__", forbidden)
    with pytest.raises(TypeError, match="on_accepted must be callable or None"):
        owner.solve(ACTIONS, TARGETS, on_accepted=observer)
    assert owner._finite_driver is None and owner._generation == 0
    with pytest.raises(RuntimeError, match="no accepted"):
        owner.recover()
