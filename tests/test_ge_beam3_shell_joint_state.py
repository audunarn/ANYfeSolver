"""Genuine coupled transactions and replay, not standalone-state conversion."""
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest

from anyfesolver._ge_beam3_shell_joint_state import CoupledShellBeamAnalysis, decode
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_shell_joint_trial import specimen


def owner(topology='Q4', curved=False, targets=(.5, 1., 0., -.5, 0.), **controls):
    assembly = specimen(topology, curved)
    force = np.zeros(assembly.shell_count+assembly.beam.nodal_count)
    force[8] = .0002
    fixed = tuple(range(6)) if topology == 'S3-V2D' else (*range(6), *range(18, 24))
    return CoupledShellBeamAnalysis(assembly, targets=targets, shell_fixed=fixed,
        nodal_forces=force, **controls)


def save(path, result):
    path.write_bytes(result.checkpoint)


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
@pytest.mark.parametrize('curved', (False, True))
def test_actual_global_load_unload_reverse_recovery_and_fresh_owner_restart(topology, curved, tmp_path):
    made = owner(topology, curved)
    standalone = canonical(made.assembly.beam.initial)
    whole = made.solve()
    assert whole.status == 'completed', whole.failure
    assert whole.state.cursor == 5
    records = decode(whole.checkpoint)['records']
    assert all(row['metric'] <= 1e-11 and row['correction'] <= 1e-11 for row in records)
    assert max(np.linalg.norm(row['constraints']) for row in records) <= 1e-11
    assert np.linalg.norm(records[1]['state']['multipliers']) > 1e-8
    assert np.linalg.norm(whole.state.shell_u) < 1e-10
    before = canonical(whole.state.descriptor())
    recovery = made.recover(whole.state)
    assert canonical(whole.state.descriptor()) == before
    assert canonical(made.assembly.beam.initial) == standalone
    with pytest.raises(ValueError): made.assembly.beam._require_issued(whole.state)
    fresh = owner(topology, curved)
    prefix = fresh.solve(stop_after=2)
    assert prefix.status == 'paused' and prefix.state.cursor == 2
    replay = owner(topology, curved)
    resumed = replay.solve(checkpoint=prefix.checkpoint, expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert resumed.status == 'completed', resumed.failure
    assert resumed.checkpoint == whole.checkpoint
    assert canonical(replay.recover(resumed.state)) == canonical(recovery)
    assert not whole.production_qualified and not whole.state.production_qualified
    save(tmp_path/'whole.json', whole); save(tmp_path/'prefix.json', prefix)
    (tmp_path/'recovery.json').write_bytes(canonical(recovery))


@pytest.mark.parametrize('stage,accepted_count', (('coupled.before_commit', 0), ('coupled.committed', 1)))
def test_cancel_at_commit_boundary_keeps_only_the_real_atomic_prefix(stage, accepted_count, tmp_path):
    made = owner(); token = CancellationToken()
    def observe(row):
        if row['stage'] == stage: token.cancel()
    stopped = made.solve(cancellation_token=token, progress=observe)
    assert stopped.status == 'cancelled'
    assert stopped.state.cursor == accepted_count
    assert len(decode(stopped.checkpoint)['records']) == accepted_count
    assert len(made._issued) == accepted_count+1
    assert not made._lock.locked() and not made.assembly._lock.locked()
    another = owner()
    resumed = another.solve(checkpoint=stopped.checkpoint, expected_sha256=sha256(stopped.checkpoint).hexdigest())
    assert resumed.status == 'completed', resumed.failure
    assert resumed.checkpoint == owner().solve().checkpoint
    save(tmp_path/'cancelled.json', stopped); save(tmp_path/'resumed.json', resumed)


def test_failed_newton_cannot_commit_candidate_or_change_either_origin(tmp_path):
    made = owner(max_iterations=1)
    before = canonical(made.initial.descriptor())
    result = made.solve()
    assert result.status == 'failed' and 'Newton limit' in result.failure
    assert result.state is made.initial and result.state.cursor == 0
    assert canonical(made.initial.descriptor()) == before
    assert len(made._issued) == 1 and len(decode(result.checkpoint)['records']) == 0
    assert not made._lock.locked() and not made.assembly._lock.locked()
    save(tmp_path/'failed-prefix.json', result)


@pytest.fixture(scope='module')
def capsule():
    made = owner(targets=(1.,))
    result = made.solve()
    assert result.status == 'completed', result.failure
    return result.checkpoint


@pytest.mark.parametrize('kind', ('schema', 'definition', 'parameter', 'beam_history', 'shell_history',
    'constraint', 'residual', 'iteration', 'qualification', 'duplicate', 'nonfinite', 'extra', 'external_hash'))
def test_mutation_replay_rejects_without_issuing_a_partial_prefix(capsule, kind):
    made = owner(targets=(1.,)); value = json.loads(capsule)
    if kind == 'schema': value['schema'] = 'standalone-beam'
    elif kind == 'definition': value['definition']['targets'] = [.5]
    elif kind == 'parameter': value['records'][0]['parameter'] = .5
    elif kind == 'beam_history': value['records'][0]['state']['beam_histories'][0]['stations'][0]['accumulated'][0] = 1.
    elif kind == 'shell_history': value['records'][0]['state']['shell_history']['extra'] = True
    elif kind == 'constraint': value['records'][0]['constraints'][0] = .1
    elif kind == 'residual': value['records'][0]['residual'][0] += .1
    elif kind == 'iteration': value['records'][0]['iterations'] = True
    elif kind == 'qualification': value['production_qualified'] = True
    elif kind == 'extra': value['extra'] = 1
    raw = canonical(value)
    if kind == 'duplicate': raw = raw.replace(b'{', b'{"schema":"duplicate",', 1)
    elif kind == 'nonfinite': raw = raw.replace(b'false', b'NaN', 1)
    expected = '0'*64 if kind == 'external_hash' else sha256(raw).hexdigest()
    with pytest.raises(ValueError): made.solve(checkpoint=raw, expected_sha256=expected)
    assert len(made._issued) == 1
    assert not made._lock.locked() and not made.assembly._lock.locked()


@pytest.mark.parametrize('kind', ('foreign', 'forged', 'mutated', 'busy', 'assembly_busy', 'cancel', 'controls', 'model'))
def test_state_ownership_and_unsupported_controls_fail_closed(kind):
    made = owner(); state = made.initial
    if kind == 'foreign': state = owner().initial
    elif kind == 'forged': state = replace(state)
    elif kind == 'mutated': object.__setattr__(state, 'production_qualified', True)
    elif kind == 'busy': made._lock.acquire()
    elif kind == 'assembly_busy': made.assembly._lock.acquire()
    elif kind == 'model': made.assembly.element.thickness *= 2
    try:
        with pytest.raises((ValueError, RuntimeError, SolveCancelled)):
            if kind == 'cancel':
                token = CancellationToken(); token.cancel(); made.solve(cancellation_token=token)
            elif kind == 'controls': made.solve(stop_after=True)
            else: made.recover(state)
    finally:
        if kind == 'busy': made._lock.release()
        if kind == 'assembly_busy': made.assembly._lock.release()
    assert not made._lock.locked() and not made.assembly._lock.locked()


def test_nodal_couples_and_inconsistent_supports_not_silently_projected():
    made = owner(); a = made.assembly
    force = made.forces.copy(); force[3] = 1.
    with pytest.raises(ValueError, match='couples'):
        CoupledShellBeamAnalysis(a, targets=(1.,), shell_fixed=made.shell_fixed, nodal_forces=force)
    with pytest.raises(ValueError, match='support'):
        CoupledShellBeamAnalysis(a, targets=(1.,), shell_fixed=(True,), nodal_forces=made.forces)


@pytest.mark.parametrize('side', ('shell', 'beam'))
def test_rotation_correction_cannot_be_diluted_by_large_displacement(side, monkeypatch):
    from types import SimpleNamespace
    made = owner(); a = made.assembly
    chosen = np.zeros(a.count)
    slot = 9 if side == 'shell' else a.shell_count+9
    chosen[slot] = 1e-8
    monkeypatch.setattr(np.linalg, 'solve', lambda *_: chosen[list(made.free)])
    response = SimpleNamespace(tangent=np.eye(a.count))
    _, _, correction = made._step(response, np.zeros(a.count), made.initial.mechanical,
        np.full(a.shell_count, 1e6), np.zeros(6))
    assert correction >= 1e-8


def test_cancelled_replay_issues_no_states(capsule, monkeypatch):
    made = owner(targets=(1.,)); token = CancellationToken(); original = made._propose
    def cancelled(*args, **kwargs):
        result = original(*args, **kwargs)
        token.cancel()
        return result
    monkeypatch.setattr(made, '_propose', cancelled)
    with pytest.raises(SolveCancelled):
        made.solve(checkpoint=capsule, expected_sha256=sha256(capsule).hexdigest(), cancellation_token=token)
    assert len(made._issued) == 1
    assert not made._lock.locked() and not made.assembly._lock.locked()


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
def test_coupled_plastic_beam_history_advances_once_and_replays(topology, tmp_path):
    from anyfesolver._ge_beam3_retained_generalized_state import Context, Program
    from anyfesolver._ge_beam3_shell_joint_trial import ShellBeamTrialAssembly
    from test_ge_beam3_native_generalized_restart import make as plastic_model, pattern
    targets = (.25, .5, 1., .5, 0., -.5, 0.)
    def build():
        model = plastic_model('curved-plastic')
        beam = Context(model, Program((1.,), pattern(model)))
        xyz = np.array([[0., 0., .2], [1., 0., .2], [1., 1., .2], [0., 1., .2]])
        if topology == 'S3-V2D': xyz = xyz[:3]
        a = ShellBeamTrialAssembly(beam, topology=topology, coordinates=xyz,
            reference_normal=np.array([0., 0., 1.]), thickness=.2, elastic_modulus=1000.,
            poisson_ratio=.3, shell_node=2, beam_node=3)
        forces = np.zeros(a.shell_count+a.beam.nodal_count)
        fixed = tuple(range(6)) if topology == 'S3-V2D' else (*range(6), *range(18, 24))
        return CoupledShellBeamAnalysis(a, targets=targets, shell_fixed=fixed, nodal_forces=forces)
    made = build(); whole = made.solve()
    assert whole.status == 'completed', whole.failure
    assert any(sum(station.accumulated) > 0. for cell in whole.state.beam_histories for station in cell.stations)
    records = decode(whole.checkpoint)['records']
    for previous, current in zip(records, records[1:]):
        assert current['state']['beam_origins'] == previous['state']['beam_histories']
        assert current['state']['shell_origin'] == previous['state']['shell_history']
    raw = decode(whole.checkpoint); raw['records'] = raw['records'][:3]; prefix = canonical(raw)
    another = build()
    resumed = another.solve(checkpoint=prefix, expected_sha256=sha256(prefix).hexdigest())
    assert resumed.status == 'completed', resumed.failure
    assert resumed.checkpoint == whole.checkpoint
    assert canonical(another.recover(resumed.state)) == canonical(made.recover(whole.state))
    assert len(made.assembly.beam._issued) == 1  # Only the genuine standalone genesis.
    assert whole.state.cursor == 7
    save(tmp_path/'plastic-whole.json', whole)
    (tmp_path/'plastic-recovery.json').write_bytes(canonical(another.recover(resumed.state)))
