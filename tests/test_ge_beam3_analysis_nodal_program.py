"""Owned retained nodal forces: actual state, direct-owner parity and fail-closed routes."""
from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from anyfesolver import _ge_beam3_analysis_nodal_program as route
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_native_analysis import analysis
from test_ge_beam3_native_generalized_modal import make
from test_ge_beam3_native_fibre_static_element import problem as fibre_problem
from test_ge_beam3_retained_nodal_loading import program


def digest(raw):
    return sha256(raw).hexdigest()


@pytest.mark.parametrize('macros,curved', ((1, False), (1, True), (2, True)))
def test_actual_owned_solve_restart_recovery_and_direct_parity(macros, curved, tmp_path):
    model, _, inertias = make(curved, macros, clamped=True)
    made = analysis(model, inertias)
    p = program()
    direct = route.owner.solve(model, p)
    whole = made.solve_nodal_program(p)
    assert direct.status == whole.status == 'completed', whole.backend_result.failure
    assert whole.backend_result.checkpoint == direct.checkpoint
    assert canonical(whole.backend_result.state) == canonical(direct.state)
    assert not whole.production_qualified
    imported = made.import_nodal_program_checkpoint(p, direct.checkpoint, expected_sha256=digest(direct.checkpoint))
    assert imported == whole.checkpoint
    prefix = made.nodal_program_checkpoint_prefix(p, whole.checkpoint, 1, expected_sha256=whole.checkpoint_sha256)
    resumed = made.solve_nodal_program(p, checkpoint=prefix, expected_sha256=digest(prefix))
    assert resumed.status == 'completed' and resumed.checkpoint == whole.checkpoint
    fields = made.recover_nodal_program(p, whole.checkpoint, expected_sha256=whole.checkpoint_sha256)
    context = route.owner.Context(model, p)
    state, records = context.restore(direct.checkpoint, expected_sha256=digest(direct.checkpoint))
    before = canonical(state)
    assert canonical(fields) == canonical(context.recover(state))
    assert canonical(state) == before and context.checkpoint(records) == direct.checkpoint
    if curved and macros == 1:
        packet, modes = route.owner.solve_modes(model, p, direct.checkpoint, inertias,
            expected_sha256=digest(direct.checkpoint), bounds=(0., 1e5))
        owned = made.nodal_program_modes(p, whole.checkpoint, expected_sha256=whole.checkpoint_sha256,
            bounds=(0., 1e5))
        assert canonical(owned.packet) == canonical(packet)
        assert canonical(owned.modes) == canonical(modes)
        assert owned.definition_graph_sha256 == made.identity
        assert owned.checkpoint_sha256 == whole.checkpoint_sha256
        assert np.all(owned.modes.eigenvalues > 0.)
        (tmp_path / 'modes.json').write_bytes(canonical(owned))
    for method in (made.recover, made.recover_spatial_couples):
        with pytest.raises(ValueError):
            method(whole.checkpoint, expected_sha256=whole.checkpoint_sha256)
    (tmp_path / 'checkpoint.json').write_bytes(whole.checkpoint)
    (tmp_path / 'recovery.json').write_bytes(canonical(fields))


@pytest.fixture(scope='module')
def accepted():
    model, _, _ = make(False, clamped=True)
    made = analysis(model)
    result = made.solve_nodal_program(program(), stop_after=1)
    assert result.status == 'paused', result.backend_result.failure
    return made, result


@pytest.mark.parametrize('kind', ('schema', 'graph', 'program', 'backend_hash', 'qualification',
    'extra', 'duplicate', 'nonfinite', 'overflow', 'whitespace', 'external_hash'))
def test_rejects_bad_authority_before_context(accepted, monkeypatch, kind):
    made, accepted = accepted
    value = json.loads(accepted.checkpoint)
    if kind == 'schema': value['schema'] = 'GE_BEAM3_MODEL_OWNED_ANALYSIS_CHECKPOINT_V1'
    elif kind == 'graph': value['definition_graph_sha256'] = '0' * 64
    elif kind == 'program': value['program']['targets'] = [.9]
    elif kind == 'backend_hash': value['backend_sha256'] = '0' * 64
    elif kind == 'qualification': value['production_qualified'] = True
    elif kind == 'extra': value['extra'] = None
    raw = canonical(value)
    if kind == 'duplicate': raw = raw.replace(b'{', b'{"schema":"duplicate",', 1)
    elif kind == 'nonfinite': raw = raw.replace(b'false', b'NaN', 1)
    elif kind == 'overflow': raw = raw.replace(b'false', b'1e9999', 1)
    elif kind == 'whitespace': raw += b' '
    expected = '0' * 64 if kind == 'external_hash' else digest(raw)
    def forbidden(*args, **kwargs):
        raise AssertionError('native context entered')
    monkeypatch.setattr(route.owner, 'Context', forbidden)
    with pytest.raises(ValueError):
        made.recover_nodal_program(program(), raw, expected_sha256=expected)
    assert not made._lock.locked()


def test_rehashed_native_mutation_and_cross_programme_rejected(accepted):
    made, result = accepted
    raw = json.loads(result.backend_result.checkpoint)
    raw['records'][0]['recovery_sha256'] = '0' * 64
    bad = canonical(raw)
    with pytest.raises(ValueError):
        made.import_nodal_program_checkpoint(program(), bad, expected_sha256=digest(bad))
    with pytest.raises(ValueError):
        made.solve_nodal_program(replace(program(), targets=(.5, 1., 1.5)),
            checkpoint=result.checkpoint, expected_sha256=result.checkpoint_sha256)
    for cursor in (True, -1, 2):
        with pytest.raises(ValueError):
            made.nodal_program_checkpoint_prefix(program(), result.checkpoint, cursor,
                expected_sha256=result.checkpoint_sha256)


@pytest.mark.parametrize('kind', ('owner', 'stop', 'observer', 'pair', 'program', 'bounds'))
def test_controls_fail_before_mechanics(accepted, monkeypatch, kind):
    made, result = accepted
    if kind == 'owner': made = analysis(fibre_problem()[0])
    def forbidden(*args, **kwargs):
        raise AssertionError('native context entered')
    monkeypatch.setattr(route.owner, 'Context', forbidden)
    with pytest.raises(ValueError):
        if kind == 'stop': made.solve_nodal_program(program(), stop_after=True)
        elif kind == 'observer': made.solve_nodal_program(program(), progress='bad')
        elif kind == 'pair': made.solve_nodal_program(program(), checkpoint=result.checkpoint)
        elif kind == 'program': made.solve_nodal_program({})
        elif kind == 'bounds': made.nodal_program_modes(program(), result.checkpoint,
            expected_sha256=result.checkpoint_sha256, bounds=(1., 0.))
        else: made.solve_nodal_program(program())


@pytest.mark.parametrize('when', ('before_commit', 'committed'))
def test_cancel_preserves_actual_prefix_and_resumes(when, tmp_path):
    model, _, _ = make(False, clamped=True)
    made = analysis(model)
    token = CancellationToken()
    def progress(row):
        if row['stage'] == 'retained-generalized.' + when: token.cancel()
    stopped = made.solve_nodal_program(program(), cancellation_token=token, progress=progress)
    assert stopped.status == 'cancelled'
    assert stopped.backend_result.completed_targets == (0 if when == 'before_commit' else 1)
    finished = made.solve_nodal_program(program(), checkpoint=stopped.checkpoint,
        expected_sha256=stopped.checkpoint_sha256)
    assert finished.status == 'completed'
    assert finished.checkpoint == made.solve_nodal_program(program()).checkpoint
    assert not made._lock.locked()
    (tmp_path / 'cancelled.json').write_bytes(stopped.checkpoint)
    (tmp_path / 'resumed.json').write_bytes(finished.checkpoint)


def test_early_cancel_failure_and_programme_mutation(accepted, monkeypatch):
    made, _ = accepted
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled):
        made.solve_nodal_program(program(), cancellation_token=token)
    failed = made.solve_nodal_program(program(iterations=1))
    assert failed.status == 'failed' and failed.backend_result.completed_targets == 0
    p = program()
    def changed(row): object.__setattr__(p, 'targets', (.2, .4))
    with pytest.raises(ValueError, match='programme changed'):
        made.solve_nodal_program(p, progress=changed)
    assert not made._lock.locked()


@pytest.mark.parametrize('case', ('curved-plastic', 'connected-plastic'))
def test_actual_plastic_load_unload_reverse_and_resume(case, tmp_path):
    from test_ge_beam3_native_generalized_restart import make as plastic_model, pattern
    model = plastic_model(case)
    made = analysis(model)
    p = route.owner.Program((.25, .5, 1., .5, 0., -.5, 0.), pattern(model), program().nodal_forces)
    whole = made.solve_nodal_program(p)
    assert whole.status == 'completed', whole.backend_result.failure
    state = whole.backend_result.state
    assert any(sum(station.accumulated) > 0. for cell in state.histories for station in cell.stations)
    prefix = made.nodal_program_checkpoint_prefix(p, whole.checkpoint, 3,
        expected_sha256=whole.checkpoint_sha256)
    resumed = made.solve_nodal_program(p, checkpoint=prefix, expected_sha256=digest(prefix))
    assert resumed.status == 'completed' and resumed.checkpoint == whole.checkpoint
    before = canonical(state)
    fields = made.recover_nodal_program(p, whole.checkpoint, expected_sha256=whole.checkpoint_sha256)
    assert canonical(state) == before
    (tmp_path / 'plastic-checkpoint.json').write_bytes(whole.checkpoint)
    (tmp_path / 'plastic-recovery.json').write_bytes(canonical(fields))
