"""Consequential staged-path/lifecycle checks; no formulation qualification."""
from dataclasses import replace
import json
import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_staged_load_experimental import (
    decode_checkpoint, encode_checkpoint,
)
from test_stiffened_shell_ge_experimental import _panel


def _owner(request, definition=None):
    owner = ss.create_analysis(definition=definition or _panel())
    request.addfinalizer(owner.close)
    return owner


def _actions(fy=0., fz=10.):
    return (ss.NodalAction("beam", 105, (0., fy, fz, 0., 0., 0.)),)


def _programme(owner):
    return ss.StagedDeadLoadProgramme(owner.preflight()["identity"],
                                     (_actions(), _actions(5., 10.), _actions(5., 5.)))


def _numeric_checkpoint(owner, raw):
    """Inspect/re-encode the selected codec with dimensions from this model.

    Stable Q4 snapshots are binary; the historical JSON path remains supported.
    Do not infer layout from untrusted snapshot metadata for corruption tests.
    """
    from anyfesolver import _stiffened_shell_ge_checkpoint_binary as binary
    if not raw.startswith(binary.MAGIC):
        return json.loads(raw), lambda value:json.dumps(value,sort_keys=True,
            separators=(',',':')).encode('ascii')
    count, members = owner._K.shape[0], len(owner._beam_parts)
    key = 'pattern' if owner._finite_driver._programme is None else 'applied_dead_load'
    shapes = (('q',(count,)),('frames',(len(owner._beam_ids),3,3)),
        ('cells',(members,2,3,3)),('resultants',(members,18)),
        ('multipliers',(owner._C.shape[0],)),(key,(count,)),
        ('accepted_factors',(owner._generation,)))
    identity = owner._finite_driver._checkpoint_identity()
    kind,generation,arrays = binary.decode(raw,identity,lambda _kind,_gen:shapes)
    assert kind in ('finite','finite_stable_q4')
    value = dict(model=identity,**{name:row.tolist() for name,row in arrays.items()})
    def encode(value):
        return binary.encode(kind,identity,generation,tuple(
            (name,np.asarray(value[name],dtype=float)) for name,_shape in shapes))
    return value,encode


def _compare(a, b):
    assert a.accepted_factors == b.accepted_factors
    assert a.load_position == b.load_position
    assert a.load_programme_identity == b.load_programme_identity
    for field in ("beam_positions", "beam_frames", "cell_frames",
                  "cell_resultants", "multipliers", "support_reactions",
                  "applied_dead_load"):
        np.testing.assert_allclose(getattr(a, field), getattr(b, field), rtol=1e-8, atol=1e-10)
    assert a.free_equilibrium_ratio <= 1e-8 and a.constraint_max <= 1e-9


@pytest.mark.parametrize("adaptive", (False, True))
def test_independent_stages_restart_boundary_interior_and_owned_observation(request, adaptive):
    owner, restarted = _owner(request), _owner(request)
    programme = _programme(owner)
    targets = (.5, 1., 1.5, 2.)
    options = {"increment_control": ss.AdaptiveIncrementControl(2, 16)} if adaptive else {}
    notices = []
    def observe(result):
        published = owner.recover()
        assert published.load_position == result.load_position
        assert published.load_stage_cursor == int(result.load_position)
        compiled = owner._finite_driver._programme
        np.testing.assert_array_equal(published.applied_dead_load,
                                      compiled.load_at(result.load_position))
        assert not np.shares_memory(result.applied_dead_load, published.applied_dead_load)
        result.applied_dead_load.setflags(write=True)
        result.applied_dead_load[:] = 123.
        notices.append(owner.checkpoint())
    final = owner.solve(programme, targets, on_accepted=observe, **options)
    assert len(notices) == len(targets)
    assert final.load_stage_cursor == 2 and final.load_stage_fraction == 0.
    assert final.load_programme_complete is False
    assert owner._finite_driver._pattern is None
    # Interior and exact knot restoration preserve controller/history and force.
    for index in (0, 1, 2):
        restored = _owner(request)
        prefix = restored.restore(notices[index])
        assert prefix.load_position == targets[index]
        resumed = restored.solve(programme, targets if adaptive else targets[index+1:], **options)
        _compare(final, resumed)
    result = restarted.restore(notices[-1])
    _compare(final, result)
    complete = restarted.solve(programme, (3.,))
    assert complete.load_programme_complete and complete.load_stage_cursor == 3
    np.testing.assert_array_equal(complete.applied_dead_load,
                                  restarted._finite_driver._programme.load_at(3.))


def test_single_stage_matches_legacy_and_keeps_legacy_checkpoint_contract(request):
    owner, legacy = _owner(request), _owner(request)
    actions = _actions()
    staged = ss.StagedDeadLoadProgramme(owner._identity, (actions,))
    a, b = owner.solve(staged, (.5, 1.)), legacy.solve(actions, (.5, 1.))
    for name in ("shell_displacement", "beam_frames", "cell_frames", "cell_resultants",
                 "multipliers", "support_reactions"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert b.load_position is None and b.applied_dead_load is None
    payload,_encode = _numeric_checkpoint(legacy,legacy.checkpoint())
    assert payload["model"] == legacy._identity and "pattern" in payload
    assert "applied_dead_load" not in payload
    with pytest.raises(ValueError, match="programme changed"):
        legacy.solve(staged, (1.5,))
    with pytest.raises(ValueError, match="cannot be mixed"):
        owner.solve(actions, (1.5,))


def test_bad_future_actions_and_skipped_boundaries_fail_before_trials(request, monkeypatch):
    owner = _owner(request)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid programme reached numerical evaluation")
    monkeypatch.setattr(FinitePointDriver, "_assemble", forbidden)
    with pytest.raises(ValueError):
        owner.solve(_programme(owner), (1.5,))
    unknown = ss.StagedDeadLoadProgramme(owner._identity,
        (_actions(), (ss.NodalAction("beam", 999, (1., 0., 0., 0., 0., 0.)),)))
    with pytest.raises(ValueError):
        owner.solve(unknown, (.5,))
    assert owner._finite_driver is None and owner._generation == 0


def test_callback_failure_and_cancel_preserve_exact_programme_prefix(request):
    owner = _owner(request)
    programme = _programme(owner)
    control = ss.AdaptiveIncrementControl(2, 16)
    token = CancellationToken()
    def cancel(result):
        token.cancel("after complete accepted stage")
    with pytest.raises(SolveCancelled):
        owner.solve(programme, (1., 2.), increment_control=control,
                    on_accepted=cancel, cancellation_token=token)
    prefix = owner.checkpoint()
    assert owner.recover().load_position == 1.
    changed = ss.StagedDeadLoadProgramme(owner._identity,
        (_actions(), _actions(6., 10.), _actions(5., 5.)))
    with pytest.raises(ValueError, match="programme changed"):
        owner.solve(changed, (1., 2.), increment_control=control)
    assert owner.checkpoint() == prefix
    result = owner.solve(programme, (1., 2.), increment_control=control)
    assert result.load_position == 2.
    # Even a typed numerical exception from the callback is not backtracked.
    error = ss.FiniteIncrementFailure(ss.IncrementAttempt(3., 2., 3., "rejected",
        1, 1, 1, 1, 1., 0., "observer"))
    def fail(result):
        raise error
    with pytest.raises(ss.FiniteIncrementFailure) as caught:
        owner.solve(programme, (3.,), on_accepted=fail, increment_control=control)
    assert caught.value is error
    assert owner.recover().load_programme_complete


def test_fresh_cancellation_and_serialization_failure_do_not_bind_programme(request, monkeypatch):
    owner = _owner(request)
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled):
        owner.solve(_programme(owner), (.5,), cancellation_token=token)
    assert owner._finite_driver is None and owner._generation == 0
    with monkeypatch.context() as scoped:
        def fail(*args, **kwargs):
            raise MemoryError("prepared durable state")
        scoped.setattr(FinitePointDriver, "_prospective_checkpoint", fail)
        with pytest.raises(MemoryError, match="durable"):
            owner.solve(_programme(owner), (.5,))
    assert owner._finite_driver is None and owner._generation == 0
    assert owner.solve(_actions(), (.5,)).load_position is None


def test_stripped_tampered_and_changed_future_checkpoint_are_atomic(request):
    owner = _owner(request)
    programme = _programme(owner)
    owner.solve(programme, (1., 1.5))
    frozen = owner.checkpoint()
    descriptor, position, inner = decode_checkpoint(frozen)
    compiled = descriptor.compile(owner, owner._finite_driver._load)
    assert position == 1.5
    numeric,reencode = _numeric_checkpoint(owner,inner)
    assert "applied_dead_load" in numeric and "pattern" not in numeric
    numeric["applied_dead_load"][owner._shell_count+6*owner._beam_index[105]+1] += 1.
    altered = reencode(numeric)
    changed = ss.StagedDeadLoadProgramme(owner._identity,
        (_actions(), _actions(6., 10.), _actions(5., 5.)))
    bad = (inner, frozen[:-1]+bytes([frozen[-1]^1]),
           encode_checkpoint(compiled, 1., inner),
           encode_checkpoint(compiled, position, altered),
           encode_checkpoint(changed.compile(owner, owner._finite_driver._load), position, inner))
    for checkpoint in bad:
        fresh = _owner(request)
        with pytest.raises(ValueError):
            fresh.restore(checkpoint)
        assert fresh._finite_driver is None and fresh._generation == 0
    assert owner.checkpoint() == frozen


def test_stage_rebase_restores_physical_state_and_continues(request):
    owner, control, restored = _owner(request), _owner(request), _owner(request)
    programme = _programme(owner)
    owner.solve(programme, (.5, 1.))
    control.solve(programme, (.5, 1.))
    rebased = owner.rebase_rotations(owner._generation)
    prefix = restored.restore(owner.checkpoint())
    assert prefix.generation == rebased.generation
    for path in (owner, control, restored):
        path.solve(programme, (1.5, 2.))
    _compare(owner.recover(), restored.recover())
    _compare(owner.recover(), control.recover())
