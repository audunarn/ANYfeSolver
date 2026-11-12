"""Persistent definition owner; unchanged 120s inner owners never renewed."""
from copy import deepcopy
from hashlib import sha256
from time import monotonic
import numpy as np
import pytest

from anyfesolver._ge_beam3_durable_coupled import DurableCoupledBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_shell_joint_state import decode
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_coupled_modes import fixture, arguments


def durable(topology='Q4', curved=False, **changes):
    original = fixture(topology, curved); a = original.assembly
    masses = arguments(original)['section_inertias']
    definitions = tuple(NativeBeamDefinition.capture(e, masses[i]) for i, e in a.beam.elements)
    settings = dict(shell=dict(topology=a.topology, coordinates=a.coordinates, reference_normal=a.normal,
        thickness=a.element.thickness, elastic_modulus=1000., poisson_ratio=.3,
        shell_node=a.shell_node, beam_node=a.beam_node), targets=original.targets,
        shell_fixed=original.shell_fixed, nodal_forces=original.forces, shell_density=2.)
    settings.update(changes)
    made = DurableCoupledBeamAnalysis(definitions, tuple(a.beam.model.boundary_conditions), a.beam.program, **settings)
    return made, original


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
@pytest.mark.parametrize('curved', (False, True))
def test_later_operations_reconstruct_and_replay_without_renewing_old_owner(topology, curved, monkeypatch, tmp_path):
    made, direct = durable(topology, curved)
    expected = direct.solve(stop_after=2)
    assert expected.status == 'paused', expected.failure
    reference_recovery = canonical(direct.recover(expected.state))
    first = made.solve(stop_after=1)
    assert first.status == 'paused', first.failure
    # Simulate idle time, not a relaxed deadline: the original owner expires.
    import anyfesolver._ge_beam3_coupled_beam_subdomain as subdomain
    clock = monotonic()+240.
    monkeypatch.setattr(subdomain, 'monotonic', lambda: monotonic()+240.)
    with pytest.raises(RuntimeError, match='deadline'): direct.guard()
    continued = made.solve(checkpoint=first.checkpoint, expected_sha256=first.checkpoint_sha256, stop_after=2)
    assert continued.status == 'paused', continued.failure
    backend, _ = made._backend(continued.checkpoint, continued.checkpoint_sha256)
    assert backend == expected.checkpoint
    recovered = made.recover(continued.checkpoint, expected_sha256=continued.checkpoint_sha256)
    assert canonical(recovered) == reference_recovery
    assert not made._lock.locked()
    assert direct.assembly.beam.started < clock-120.
    with pytest.raises(RuntimeError, match='deadline'): direct.guard()
    (tmp_path/'durable.json').write_bytes(canonical(dict(checkpoint=continued.checkpoint.decode('ascii'),
        recovery=recovered, old_deadline_unchanged=True, production_qualified=False)))


@pytest.mark.parametrize('operation', ('recover', 'modes', 'buckling'))
@pytest.mark.parametrize('fault', ('hash', 'schema', 'backend', 'identity', 'cancel', 'concurrent', 'mutation'))
def test_rejected_inputs_do_not_construct_inner_owner(operation, fault, monkeypatch):
    made, _ = durable()
    first = made.solve(stop_after=0)
    raw = first.checkpoint; expected = first.checkpoint_sha256
    token = None
    if fault == 'hash': expected = '0'*64
    elif fault in ('schema', 'backend', 'identity'):
        value = decode(raw)
        value[{'schema':'schema', 'backend':'backend_sha256', 'identity':'definition_sha256'}[fault]] = 'changed'
        raw = canonical(value); expected = sha256(raw).hexdigest()
    elif fault == 'cancel': token = CancellationToken(); token.cancel()
    elif fault == 'concurrent': made._lock.acquire()
    elif fault == 'mutation': made._controls['max_iterations'] -= 1
    def forbidden(*args, **kwargs): raise AssertionError('invalid envelope entered mechanics')
    monkeypatch.setattr(made, '_new', forbidden)
    options = dict(expected_sha256=expected, cancellation_token=token)
    if operation == 'modes': options.update(bounds=(-100., 10000.), num_modes=2)
    if operation == 'buckling': options.update(bounds=(0., 1000.), num_modes=2)
    try:
        with pytest.raises((ValueError, RuntimeError, SolveCancelled)):
            getattr(made, operation)(raw, **options)
    finally:
        if made._lock.locked(): made._lock.release()


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
def test_real_spectral_queries_replay_exact_accepted_state(topology, tmp_path):
    made, direct = durable(topology)
    a = made.solve(stop_after=1); b = direct.solve(stop_after=1)
    assert a.status == b.status == 'paused', (a.failure, b.failure)
    from anyfesolver._ge_beam3_coupled_modes import modes, buckling
    expected_modes = modes(direct, b.state, **arguments(direct), bounds=(-100., 10000.), num_modes=2)
    expected_buckling = buckling(direct, b.state, bounds=(0., 1000.), num_modes=2)
    actual_modes = made.modes(a.checkpoint, expected_sha256=a.checkpoint_sha256, bounds=(-100., 10000.), num_modes=2)
    actual_buckling = made.buckling(a.checkpoint, expected_sha256=a.checkpoint_sha256, bounds=(0., 1000.), num_modes=2)
    assert canonical(actual_modes) == canonical(expected_modes)
    assert canonical(actual_buckling) == canonical(expected_buckling)
    (tmp_path/'spectral.json').write_bytes(canonical(dict(modes=actual_modes, buckling=actual_buckling,
        checkpoint=a.checkpoint.decode('ascii'), production_qualified=False)))


def test_cancelled_solve_preserves_real_prefix_and_does_not_advance_input(tmp_path):
    made, _ = durable()
    first = made.solve(stop_after=1); before = first.checkpoint
    token = CancellationToken()
    def observed(row):
        if row['stage'] == 'coupled.before_commit': token.cancel()
    stopped = made.solve(checkpoint=before, expected_sha256=first.checkpoint_sha256,
        stop_after=2, cancellation_token=token, progress=observed)
    assert stopped.status == 'cancelled' and stopped.cursor == first.cursor
    assert stopped.checkpoint == before and not made._lock.locked()
    made.recover(before, expected_sha256=first.checkpoint_sha256)
    (tmp_path/'cancelled.json').write_bytes(canonical(dict(checkpoint=before.decode('ascii'),
        cancelled_status=stopped.status, production_qualified=False)))


def test_near_limit_inner_payload_roundtrips_without_tightening_historical_bound(monkeypatch):
    from anyfesolver._ge_beam3_durable_coupled import MAX_BYTES, OUTER_MAX_BYTES, _outer
    made, _ = durable()
    backend = canonical(dict(padding='"'*(MAX_BYTES//2-32)))
    assert MAX_BYTES-100 < len(backend) <= MAX_BYTES
    def forbidden(*args, **kwargs): raise AssertionError('serialization invoked mechanics')
    monkeypatch.setattr(made, '_new', forbidden)
    wrapped = made._envelope(backend)
    assert MAX_BYTES < len(wrapped) <= OUTER_MAX_BYTES
    assert made._backend(wrapped, sha256(wrapped).hexdigest()) == (backend, sha256(backend).hexdigest())
    with pytest.raises(ValueError): _outer(b' '*(OUTER_MAX_BYTES+1))
    with pytest.raises(ValueError): made._envelope(canonical(dict(padding='a'*MAX_BYTES)))


@pytest.mark.parametrize('raw', (b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":1e999}\n',
    b' {"a":1}\n', b'['*33+b'0'+b']'*33+b'\n', b'"\xff"\n'))
def test_strict_outer_decoder(raw):
    from anyfesolver._ge_beam3_durable_coupled import _outer
    with pytest.raises(ValueError): _outer(raw)


def test_cancel_after_commit_returns_prefix_replayable_by_fresh_facade(tmp_path):
    made, _ = durable(); token = CancellationToken()
    def observed(row):
        if row['stage'] == 'coupled.committed': token.cancel()
    stopped = made.solve(stop_after=2, cancellation_token=token, progress=observed)
    assert stopped.status == 'cancelled' and stopped.cursor == 1
    fresh, _ = durable()
    resumed = fresh.solve(checkpoint=stopped.checkpoint, expected_sha256=stopped.checkpoint_sha256, stop_after=2)
    complete, _ = durable(); expected = complete.solve(stop_after=2)
    assert resumed.status == expected.status == 'paused'
    assert resumed.checkpoint == expected.checkpoint
    (tmp_path/'after-commit.json').write_bytes(canonical(dict(cancelled=stopped.checkpoint.decode('ascii'),
        resumed=resumed.checkpoint.decode('ascii'), production_qualified=False)))


def test_actual_failed_newton_preserves_replayable_genesis(tmp_path):
    made, _ = durable(max_iterations=1)
    reference = made.solve(stop_after=0)
    failed = made.solve(stop_after=2)
    assert failed.status == 'failed' and 'Newton limit' in failed.failure
    assert failed.cursor == 0 and failed.checkpoint == reference.checkpoint
    fresh, _ = durable(max_iterations=1)
    recovery = fresh.recover(failed.checkpoint, expected_sha256=failed.checkpoint_sha256)
    assert recovery['coupled_cursor'] == 0
    (tmp_path/'failed-prefix.json').write_bytes(canonical(dict(checkpoint=failed.checkpoint.decode('ascii'),
        recovery=recovery, failed_status=failed.status, production_qualified=False)))


def test_mutating_caller_configuration_cannot_change_owned_definition():
    original, _ = durable()
    shell = deepcopy(original._shell); controls = deepcopy(original._controls)
    definitions = tuple(NativeBeamDefinition(raw) for raw in original._definitions)
    from anyfesolver.boundary import BoundaryCondition
    boundaries = (BoundaryCondition('extra-support', [3], {'ux':0.}),)
    program = deepcopy(original._program)
    made = DurableCoupledBeamAnalysis(definitions, boundaries, program,
        shell=shell, shell_density=2., **controls)
    before = made.identity
    shell['coordinates'].setflags(write=True); shell['coordinates'][:, 2] += .3
    controls['nodal_forces'].setflags(write=True); controls['nodal_forces'][:] += .01
    boundaries[0].dof_constraints['ux'] = .1
    object.__setattr__(program, 'targets', (2.,))
    made._guard(); assert made.identity == before
    assert made.solve(stop_after=0).status == 'paused'


def test_changed_physical_density_rejects_checkpoint_before_reconstruction(monkeypatch):
    original, _ = durable(); reference = original.solve(stop_after=0)
    changed, _ = durable(shell_density=3.)
    def forbidden(*args, **kwargs): raise AssertionError('foreign density entered mechanics')
    monkeypatch.setattr(changed, '_new', forbidden)
    with pytest.raises(ValueError, match='definition'):
        changed.recover(reference.checkpoint, expected_sha256=reference.checkpoint_sha256)
