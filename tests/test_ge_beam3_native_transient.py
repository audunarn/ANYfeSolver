"""Native trajectory development checks; complete module did not finish.

Selected installed checks passed; partial source assertions and failures are retained.

Actual native-owner tests remain distinct from cheap injected controller oracles.
Existing endpoint energy/momentum gates are not regraded by this test candidate.
"""
from dataclasses import replace, asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import numpy as np
import pytest

from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.dynamics import TransientConfig
from anyfesolver.ge_beam3_native_dynamics import solve_transient
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_native_transient import (
    NativeTransientError, _settings, _advance, _record, _result)
from anyfesolver._ge_beam3_native_midpoint import MidpointContext
from examples.native_contact_free_transient_dev import build, run


def save_result(name, result):
    """Optional separate raw records; ordinary pytest needs no environment."""
    path = os.environ.get('NATIVE_TRANSIENT_RESULT_RECORD')
    if path:
        root = Path(path); root.mkdir(exist_ok=True)
        (root/(name+'.json')).write_text(json.dumps(result.to_dict(), indent=2)+'\n', encoding='utf-8')


def assert_fresh(owner):
    context = owner._owner()
    assert context.current is None and context._current is None
    assert context._current_bytes is None and not context._issued and not context._candidates
    assert context._initial_checks is None and context._restore_checks is None
    assert context._work_session is None and context._dynamic._active_session is None


def reseal(row):
    row['checkpoint_sha256'] = sha({key:value for key,value in row.items()
                                    if key != 'checkpoint_sha256'})
    return canonical(row)


def accounting_bytes(value):
    fields = asdict(value)
    fields['rank_mass_checks'] = json.loads(fields['rank_mass_checks'])
    return canonical(fields)


def test_public_native_trajectory_matches_manual_and_split():
    owner, config = build(t_end=.002, save_every=2)
    manual, _ = build(t_end=.002, save_every=2)
    state = manual.initialize(physical_velocity15=config.initial_velocity)
    for _ in range(2):
        candidate = manual.trial(state, dt=.001)
        preview = manual.candidate_endpoint_accounting(candidate)
        prepared = manual.prepare_candidate_checkpoint(candidate)
        state = manual.commit(candidate)
        assert prepared == manual.create_checkpoint()
        assert accounting_bytes(preview) == accounting_bytes(manual.endpoint_accounting(state))
    full = solve_transient(owner, config)
    save_result('public_full', full)
    assert canonical(owner.current) == canonical(state)
    assert full.success and list(full.times) == [0., .002]
    prefix_owner, prefix = run(t_end=.001, save_every=2)
    resumed_owner, resumed = run(t_end=.002, save_every=2, checkpoint=prefix.checkpoint,
        expected_sha256=sha256(prefix.checkpoint).hexdigest())
    save_result('public_resumed', resumed)
    assert full.checkpoint == resumed.checkpoint
    assert canonical(full.to_dict()) == canonical(resumed.to_dict())
    assert canonical(resumed_owner.current) == canonical(owner.current)
    detached = full.to_dict()
    detached['final']['state']['mechanical']['positions'][0][0] = 123.
    assert full.to_dict()['final']['state']['mechanical']['positions'][0][0] != 123.
    assert prefix_owner.current.epoch == 1


def test_free_uniform_translation_independent_reference():
    """Analytic rigid translation with zero elastic strain and no applied load."""
    owner, config = build(t_end=.001)
    velocity = np.array((.01, -.004, .003), dtype=np.float64)
    physical = np.concatenate((np.tile(velocity, 3), np.zeros(6)))
    reference = owner._owner()._reference_mechanical
    positions = reference.positions+reference.position_low
    result = solve_transient(owner, replace(config, initial_velocity=physical))
    save_result('uniform_translation', result)
    row = result.to_dict()['final']
    actual = np.array(row['state']['mechanical']['positions'])+np.array(row['state']['mechanical']['position_low'])
    assert np.max(np.abs(actual-(positions+.001*velocity)))/2. < 1e-6
    assert np.max(np.abs(np.array(row['state']['physical_velocity24'])[
        [0,1,2,6,7,8,12,13,14]]-np.tile(velocity, 3)))/.01 < 1e-6
    # rhoA=2.7 and reference r(x)=(x,.2(1-x^2),0), x in[-1,1].
    # Independent exact arc-length primitive; no production inertia kernel.
    length = np.sqrt(1.16)+np.arcsinh(.4)/.4
    mass = 2.7*length
    kinetic = .5*mass*float(velocity@velocity)
    assert abs(row['accounting']['kinetic_energy']-kinetic)/kinetic < 1e-6


def test_physical15_reference_increment_is_actual_rigid_so3_pose():
    owner, config = build(t_end=0.)
    reference = owner._owner()._reference_mechanical
    angle = .001
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(((c,-s,0.),(s,c,0.),(0.,0.,1.)))
    points = reference.positions+reference.position_low
    translated = points@rotation.T+np.array((.002,-.001,.003))
    increment = np.concatenate(((translated-points).ravel(), [0.,0.,angle,0.,0.,angle]))
    result = solve_transient(owner, replace(config, initial_displacement=increment,
                                          initial_velocity=np.zeros(15)))
    save_result('physical15_pose', result)
    mechanical = result.to_dict()['initial']['state']['mechanical']
    actual = np.array(mechanical['positions'])+np.array(mechanical['position_low'])
    assert np.max(np.abs(actual-translated))/2. < 1e-6
    expected = np.array([rotation@cell for cell in reference.cell_rotations[0]])
    assert np.max(np.abs(np.array(mechanical['cell_rotations'])[0]-expected)) < 1e-11


@pytest.mark.parametrize('clamped,loaded', [(False, True), (True, True)])
def test_public_native_actual_loaded_work(clamped, loaded):
    _, result = run(clamped=clamped, loaded=loaded, t_end=.001)
    save_result('loaded_clamped' if clamped else 'loaded_free', result)
    data = result.to_dict()
    assert data['accepted_steps'] == 1 and data['status'] == 'completed'
    step = data['steps'][0]
    assert step['stage_residual_norm'] <= 1e-11
    assert step['endpoint_algebraic_norm'] <= 1e-11
    assert np.isfinite(step['midpoint_load_work_j'])
    assert np.linalg.norm(data['final']['physical_load']['force']) > 0.
    # The journal is the declared discrete physical work, not an assertion of
    # exact conservation or continuum temporal accuracy.
    assert step['midpoint_load_work_j'] == float(np.dot(
        step['midpoint_external_load42'], step['midpoint_physical_increment42']))
    initial = data['initial']['accounting']; final = data['final']['accounting']
    support = step['midpoint_support']
    force_scale, moment_scale, dt = .03, .05, .001
    delta_p = sum((np.array(final[name])-np.array(initial[name]) for name in
                   ('linear_momentum','linear_momentum_low')), np.zeros(3))
    delta_h = sum((np.array(final[name])-np.array(initial[name]) for name in
                   ('angular_momentum','angular_momentum_low')), np.zeros(3))
    impulse = np.array(step['midpoint_force_impulse_ns'])+np.array(support['physical_force_impulse_ns'])
    angular = np.array(step['midpoint_moment_impulse_nms'])+np.array(support['physical_moment_impulse_nms'])
    assert np.linalg.norm(delta_p-impulse)/(force_scale*dt) <= 1e-6
    assert np.linalg.norm(delta_h-angular)/(moment_scale*dt) <= 1e-6
    if not clamped:
        length = np.sqrt(1.16)+np.arcsinh(.4)/.4
        assert np.linalg.norm(np.array(step['midpoint_force_impulse_ns'])-
                              np.array((0.,-.01*length*dt,0.)))/(force_scale*dt) <= 1e-6
    else:
        assert support['physical_work_j'] == 0. and support['algebraic_work_j'] == 0.
        fixed = data['initial']['state']['mechanical']
        arm = np.array(fixed['positions'])[0]+np.array(fixed['position_low'])[0]
        load = data['initial']['physical_load']
        arm -= np.array(load['momentum_origin_high'])+np.array(load['momentum_origin_low'])
        reaction = np.array(support['physical_reaction42'])
        couple = np.array(support['algebraic_reaction42'])[3:6]
        expected = dt*(np.cross(arm,reaction[:3])+couple)
        assert np.linalg.norm(expected-np.array(support['physical_moment_impulse_nms']))/(moment_scale*dt) < 1e-6


def test_candidate_evidence_is_authenticated_nonpublishing(monkeypatch):
    owner, config = build(t_end=.001)
    initial = owner.initialize(physical_velocity15=config.initial_velocity)
    candidate = owner.trial(initial, dt=.001)
    before = canonical(initial)
    for method in (owner.candidate_endpoint_accounting, owner.prepare_candidate_checkpoint):
        with pytest.raises(ValueError):
            method(replace(candidate, dt=.002))
        token = CancellationToken(); token.cancel('before preview')
        with pytest.raises(SolveCancelled):
            method(candidate, cancellation_token=token)
        assert owner.current is initial and canonical(initial) == before
        original_dt = candidate.dt
        object.__setattr__(candidate, 'dt', .002)
        with pytest.raises(ValueError):
            method(candidate)
        object.__setattr__(candidate, 'dt', original_dt)
    original = MidpointContext._endpoint_accounting
    def broken(self, state):
        raise RuntimeError('injected accounting failure')
    with monkeypatch.context() as scope:
        scope.setattr(MidpointContext, '_endpoint_accounting', broken)
        with pytest.raises(RuntimeError, match='injected'):
            owner.candidate_endpoint_accounting(candidate)
        assert owner.current is initial
    assert MidpointContext._endpoint_accounting is original
    owner.commit(candidate)
    with pytest.raises(ValueError):
        owner.prepare_candidate_checkpoint(candidate)


def test_genesis_preparation_cancellation_and_serialization_roll_back(monkeypatch):
    import anyfesolver._ge_beam3_native_transient as driver
    for cancel in (False, True):
        owner, config = build(t_end=0.)
        token = CancellationToken()
        def fail(*args, **kwargs):
            if cancel:
                token.cancel('during genesis preparation')
                token.raise_if_cancelled()
            raise NativeTransientError('injected serialization failure')
        with monkeypatch.context() as scope:
            scope.setattr(driver, '_record', fail)
            with pytest.raises((NativeTransientError, SolveCancelled)):
                solve_transient(owner, config, cancellation_token=token)
        assert_fresh(owner)


def test_restore_correspondence_and_identity_fail_before_publication():
    _, prefix = run(t_end=0.)
    row = json.loads(prefix.checkpoint)
    for corrupt in ('accounting', 'time'):
        value = json.loads(prefix.checkpoint)
        if corrupt == 'accounting':
            value['controller']['initial']['accounting']['total_energy'] += 1.
            value['controller']['outputs'][0]['accounting']['total_energy'] += 1.
        else:
            value['controller']['time'] = .001
        raw = reseal(value)
        owner, config = build(t_end=.001)
        with pytest.raises(ValueError):
            solve_transient(owner, config, checkpoint=raw, expected_sha256=sha256(raw).hexdigest())
        assert_fresh(owner)
    owner, config = build(dt=.0005, t_end=.001)
    with pytest.raises(ValueError, match='identity'):
        solve_transient(owner, config, checkpoint=prefix.checkpoint,
                        expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert_fresh(owner)
    assert row['controller']['time'] == 0.


@pytest.mark.parametrize('change', [dict(hht_alpha=-.1), dict(beta=.3), dict(rayleigh_alpha=.01),
    dict(include_stress_history=True), dict(output_nodes=(1,)), dict(save_every=True)])
def test_config_refusal_without_native_construction(change):
    config = replace(TransientConfig(dt=.001, t_end=.002), **change)
    with pytest.raises(ValueError):
        _settings(config, 10, 2, None)


def test_controller_cutback_and_precommit_failure_oracle(monkeypatch):
    """No synthetic mechanics proof: injected controller behavior only."""
    import anyfesolver._ge_beam3_native_transient as driver
    owner, config = build(t_end=.001)
    prefix = solve_transient(owner, replace(config, t_end=0.))
    capsule = json.loads(prefix.checkpoint)
    state = owner.current
    controller = capsule['controller']; contract = capsule['contract']
    native = capsule['native_checkpoint'].encode('ascii')
    original_trial = owner._prepare_trial
    class Delegate:
        def __init__(self):
            self.calls = 0
        def _prepare_trial(self, accepted, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError('midpoint stage iteration limit')
            return original_trial(accepted, **kwargs)
        def __getattr__(self, name):
            return getattr(owner, name)
    fake = Delegate()
    result = _advance(fake, state, .001, capsule['identity'], contract, controller,
                      native, None, prefix)
    assert fake.calls == 3 and result.to_dict()['accepted_steps'] == 2
    assert len(result.to_dict()['steps'][0]['cutbacks']) == 1
    assert owner.current.epoch == 2
    # A second owner tests real issued-candidate cleanup on a serialization
    # exception; no commit/cutback or hidden fallback is permitted.
    other, other_config = build(t_end=0.)
    previous = solve_transient(other, other_config)
    data = json.loads(previous.checkpoint)
    with monkeypatch.context() as scope:
        scope.setattr(driver, '_result', lambda *a, **k: (_ for _ in ()).throw(ValueError('codec failure')))
        with pytest.raises(NativeTransientError) as error:
            _advance(other, other.current, .001, data['identity'], data['contract'], data['controller'],
                data['native_checkpoint'].encode('ascii'), None, previous)
    assert error.value.partial is previous and other.current.epoch == 0
    assert not other._owner()._candidates


def test_controller_inflight_cancel_keeps_native_prefix(monkeypatch):
    owner, config = build(t_end=0.)
    previous = solve_transient(owner, config)
    row = json.loads(previous.checkpoint)
    token = CancellationToken()
    original = MidpointContext.prepare_candidate_checkpoint
    def cancel_after_prepare(self, candidate, **kwargs):
        raw = original(self, candidate, **kwargs)
        token.cancel('after candidate evidence before commit')
        return raw
    monkeypatch.setattr(MidpointContext, 'prepare_candidate_checkpoint', cancel_after_prepare)
    with pytest.raises(SolveCancelled) as error:
        _advance(owner, owner.current, .001, row['identity'], row['contract'], row['controller'],
            row['native_checkpoint'].encode('ascii'), token, previous)
    assert error.value.partial is previous and owner.current.epoch == 0
    assert not owner._owner()._candidates


def test_keyboard_interrupt_precommit_discards_candidate(monkeypatch):
    import anyfesolver._ge_beam3_native_transient as driver
    owner, config = build(t_end=0.)
    previous = solve_transient(owner, config)
    row = json.loads(previous.checkpoint)
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt('during serialization')
    monkeypatch.setattr(driver, '_result', interrupt)
    with pytest.raises(KeyboardInterrupt) as error:
        _advance(owner, owner.current, .001, row['identity'], row['contract'], row['controller'],
            row['native_checkpoint'].encode('ascii'), None, previous)
    assert error.value.partial is previous and owner.current.epoch == 0
    assert not owner._owner()._candidates


def test_keyboard_interrupt_after_authentic_commit_publishes_prepared_prefix(monkeypatch):
    owner, config = build(t_end=0.)
    previous = solve_transient(owner, config)
    row = json.loads(previous.checkpoint)
    original = MidpointContext.commit
    signal = KeyboardInterrupt('after authentic commit before controller return')
    def interrupt_after_commit(self, candidate, **kwargs):
        original(self, candidate, **kwargs)
        raise signal
    monkeypatch.setattr(MidpointContext, 'commit', interrupt_after_commit)
    with pytest.raises(KeyboardInterrupt) as error:
        _advance(owner, owner.current, .001, row['identity'], row['contract'], row['controller'],
            row['native_checkpoint'].encode('ascii'), None, previous)
    assert error.value is signal
    partial = error.value.partial
    assert partial is not previous and partial.success
    assert owner.current.epoch == 1 and owner.current.time == .001
    assert error.value.accepted_state is owner.current
    assert not owner._owner()._candidates
    capsule = json.loads(partial.checkpoint)
    assert capsule['controller']['accepted_steps'] == 1
    assert error.value.native_checkpoint == capsule['native_checkpoint'].encode('ascii')
    assert owner.create_checkpoint() == error.value.native_checkpoint
    assert partial.to_dict()['final']['epoch'] == owner.current.epoch
    save_result('postcommit_interrupt_prefix', partial)


@pytest.mark.parametrize('control_signal', [KeyboardInterrupt, SystemExit])
def test_control_signal_before_commit_preserves_old_prefix(monkeypatch, control_signal):
    owner, config = build(t_end=0.)
    previous = solve_transient(owner, config)
    row = json.loads(previous.checkpoint)
    signal = control_signal('before native commit')
    def interrupt_before_commit(self, candidate, **kwargs):
        raise signal
    monkeypatch.setattr(MidpointContext, 'commit', interrupt_before_commit)
    with pytest.raises(control_signal) as error:
        _advance(owner, owner.current, .001, row['identity'], row['contract'], row['controller'],
            row['native_checkpoint'].encode('ascii'), None, previous)
    assert error.value is signal and error.value.partial is previous
    assert error.value.accepted_state is owner.current and owner.current.epoch == 0
    assert error.value.native_checkpoint == row['native_checkpoint'].encode('ascii')
    assert not owner._owner()._candidates


@pytest.mark.parametrize('after_atomic_exit', [True, False])
def test_trial_receipt_control_signal_cleans_only_actual_issuance(monkeypatch, after_atomic_exit):
    owner, config = build(t_end=0.)
    previous = solve_transient(owner, config)
    row = json.loads(previous.checkpoint)
    signal = KeyboardInterrupt('after issued trial' if after_atomic_exit else 'captured then native rollback')
    original = MidpointContext._prepare_trial
    captured = []
    def interrupted_trial(self, accepted, *, prepare, **kwargs):
        def receive(candidate):
            captured.append(candidate)
            prepare(candidate)
            if not after_atomic_exit:
                raise signal
        result = original(self, accepted, prepare=receive, **kwargs)
        assert captured and result is captured[0]
        raise signal
    monkeypatch.setattr(MidpointContext, '_prepare_trial', interrupted_trial)
    with pytest.raises(KeyboardInterrupt) as error:
        _advance(owner, owner.current, .001, row['identity'], row['contract'], row['controller'],
            row['native_checkpoint'].encode('ascii'), None, previous)
    assert captured and error.value is signal
    assert error.value.partial is previous and owner.current.epoch == 0
    assert error.value.accepted_state is owner.current
    assert error.value.native_checkpoint == row['native_checkpoint'].encode('ascii')
    assert not owner._owner()._candidates
    assert owner.create_checkpoint() == error.value.native_checkpoint
