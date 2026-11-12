"""Model ownership of the unchanged native force/couple/history protocols."""
from hashlib import sha256
import json

import numpy as np
import pytest

from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._ge_beam3_native_analysis import COMBINED_WORKFLOW, FORCE_WORKFLOW_SCHEMA
from anyfesolver._ge_beam3_native_force_adaptation import AdaptiveForcePolicy
from anyfesolver._ge_beam3_native_generalized_combined_couples import solve_combined_static, _RUN
from anyfesolver._ge_beam3_native_generalized_combined_restart import LoadPoint, encode_checkpoint, decode_checkpoint
from anyfesolver._ge_beam3_native_generalized_restart import LoadPoint as DistributedPoint
from anyfesolver._ge_beam3_native_generalized_program import _PROGRAM
from anyfesolver._ge_beam3_native_generalized_loading import _ACTIVE
from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_native_analysis import analysis
from test_ge_beam3_native_generalized_restart import make, pattern, scale, CASES, EMPTY
from test_ge_beam3_native_generalized_combined_restart import moment_scale, packet, snapshots


MOMENTS = SpatialNodalMoments(((3, .015, -.01, .008),))


def owned(case=CASES[0]):
    return analysis(make(case))


def unlocked(made):
    assert made._lock.acquire(blocking=False)
    made._lock.release()
    made._guard()
    assert _RUN.get() is None and _PROGRAM.get() is None and _ACTIVE.get() is None


@pytest.fixture(scope='module', params=CASES)
def saved(request):
    case = request.param
    made = owned(case)
    line = pattern(made.model)
    run = made.solve_spatial_couples(MOMENTS, distributed_pattern=line)
    assert run.status == 'completed', run.backend_result.info
    return case, line, run


def test_direct_native_equivalence_and_physical_recovery(saved, tmp_path):
    case, line, run = saved
    source = make(case)
    initial = {i:e.init_model_bound_nonlinear_state(source.mesh, e.section, 1)
               for i, e in source.mesh.elements.items()}
    direct, _ = solve_combined_static(source, MOMENTS, distributed_pattern=line)
    assert direct.status == 'completed'
    assert canonical(packet(direct)) == canonical(packet(run.backend_result))
    chain = (dict(load_point=LoadPoint(DistributedPoint(0., EMPTY, line), None, MOMENTS),
        displacements=np.zeros(source.mesh.dof_manager.total_dofs), states=initial),)+snapshots(direct, line, MOMENTS)
    raw = encode_checkpoint(source, chain)
    made = owned(case)
    actual, digest = made._backend(run.checkpoint, run.checkpoint_sha256, workflow=COMBINED_WORKFLOW)
    assert actual == raw
    assert json.loads(run.checkpoint)['schema'] == FORCE_WORKFLOW_SCHEMA
    before = canonical(decode_checkpoint(made.model, raw, expected_sha256=digest))
    recovered = made.recover_spatial_couples(run.checkpoint, expected_sha256=run.checkpoint_sha256)
    for i, fields in recovered.items():
        assert fields['formulation_id'] == made.model.mesh.elements[i].formulation_id
        assert fields['fibre_stress_status'] == 'RESULTANT_LEVEL_ONLY_NO_FIBRE_STRESSES_SUPPLIED'
        for station in fields['stations']:
            assert 'fibres' not in station
            local = station['resultants']+station['resultants_low']
            physical = station['global_resultants']+station['global_resultants_low']
            variation = np.sin(np.arange(6)+.4)
            frame = station['current_frame']
            transported = np.r_[frame@variation[:3], frame@variation[3:]]
            assert abs(local@variation-physical@transported)/max(1., abs(local@variation)) <= 1e-11
    assert canonical(decode_checkpoint(made.model, raw, expected_sha256=digest)) == before
    (tmp_path/'checkpoint.json').write_bytes(run.checkpoint)
    (tmp_path/'recovery.json').write_bytes(canonical(recovered))
    unlocked(made)


def test_prefix_continuation_unload_and_failed_step(saved, tmp_path):
    case, line, whole = saved
    made = owned(case)
    prefix = made.spatial_couple_checkpoint_prefix(whole.checkpoint, 1, expected_sha256=whole.checkpoint_sha256)
    resumed = made.solve_spatial_couples(moment_scale(MOMENTS, .5), distributed_pattern=scale(line, .5),
        steps=1, checkpoint=prefix, expected_sha256=sha256(prefix).hexdigest())
    assert resumed.status == 'completed'
    assert canonical(packet(resumed.backend_result)) == canonical(packet(whole.backend_result))
    before = canonical(resumed.backend_result.element_states)
    failed = owned(case).solve_spatial_couples(MOMENTS, distributed_pattern=line, steps=1,
        max_iterations=1, checkpoint=resumed.checkpoint, expected_sha256=resumed.checkpoint_sha256)
    assert failed.status != 'completed' and not failed.backend_result.snapshots
    assert canonical(failed.backend_result.element_states) == before
    assert failed.checkpoint == resumed.checkpoint
    unloaded = owned(case).solve_spatial_couples(moment_scale(MOMENTS, -.5), distributed_pattern=scale(line, -.5),
        steps=1, checkpoint=resumed.checkpoint, expected_sha256=resumed.checkpoint_sha256)
    assert unloaded.status == 'completed'
    for eid, state in unloaded.backend_result.element_states.items():
        old = resumed.backend_result.element_states[eid]
        assert state['previous_state_sha256'] == old['state_sha256']
        assert canonical(state['origins']) == canonical(old['response'].history)
    fields = owned(case).recover_spatial_couples(unloaded.checkpoint, expected_sha256=unloaded.checkpoint_sha256)
    assert canonical(resumed.backend_result.element_states) == before
    (tmp_path/'prefix.json').write_bytes(prefix)
    (tmp_path/'resumed.json').write_bytes(resumed.checkpoint)
    (tmp_path/'unloaded.json').write_bytes(unloaded.checkpoint)
    (tmp_path/'unloaded-fields.json').write_bytes(canonical(fields))
    unlocked(made)


def test_no_conservative_or_other_workflow_relabelling(saved):
    case, line, run = saved
    made = owned(case)
    actions = (
        lambda: made.recover(run.checkpoint, expected_sha256=run.checkpoint_sha256),
        lambda: made.checkpoint_prefix(run.checkpoint, 1, expected_sha256=run.checkpoint_sha256),
        lambda: made.current_modes(run.checkpoint, expected_sha256=run.checkpoint_sha256),
        lambda: made.buckling_modes(run.checkpoint, expected_sha256=run.checkpoint_sha256, bounds=(0., 10.)),
        lambda: made.solve_distributed(line, checkpoint=run.checkpoint, expected_sha256=run.checkpoint_sha256),
    )
    for action in actions:
        with pytest.raises(ValueError): action()
    unlocked(made)


@pytest.mark.parametrize('kind', ('schema', 'workflow', 'graph', 'owner', 'backend', 'duplicate', 'nonfinite'))
def test_strict_workflow_envelope(kind):
    made = owned()
    raw = made._envelope(b'{}', workflow=COMBINED_WORKFLOW)
    data = json.loads(raw)
    if kind == 'schema': data['schema'] = 'other'
    elif kind == 'workflow': data['workflow'] = 'other'
    elif kind == 'graph': data['definition_graph_sha256'] = '0'*64
    elif kind == 'owner': data['owner'] = 'PHYSICAL_FIBRE_NODAL'
    elif kind == 'backend': data['backend_sha256'] = '0'*64
    raw = canonical(data)
    if kind == 'duplicate': raw = raw.replace(b'{', b'{"schema":"duplicate",', 1)
    if kind == 'nonfinite': raw = raw.replace(b'false', b'NaN', 1)
    with pytest.raises(ValueError):
        made.recover_spatial_couples(raw, expected_sha256=sha256(raw).hexdigest())
    unlocked(made)


@pytest.mark.parametrize('options', ({'steps':True}, {'line_search':1}, {'step_policy':{}},
    {'step_policy':AdaptiveForcePolicy(cutback_levels=6), 'steps':2}, {'progress_callback':4}))
def test_controls_rejected_before_initialization(options, monkeypatch):
    made = owned()
    def forbidden(*a, **k): raise AssertionError('invalid controls entered mechanics')
    monkeypatch.setattr(made, '_initial', forbidden)
    with pytest.raises(ValueError): made.solve_spatial_couples(MOMENTS, **options)
    unlocked(made)


def test_cancel_before_decode_and_during_unaccepted_trial(monkeypatch):
    made = owned(CASES[1])
    token = CancellationToken()
    token.cancel('do not capture')
    with pytest.raises(SolveCancelled):
        made.solve_spatial_couples(MOMENTS, checkpoint=b'bad', expected_sha256='0'*64, cancellation_token=token)
    token = CancellationToken()
    element = made._elements[0]
    original = element.compute_nonlinear_response
    captured = []
    def stop_trial(*a, **k):
        result = original(*a, **k)
        context, view = k['native_material_context'], k['native_rotation_trial']
        if not captured and not np.array_equal(view.trial_coordinates, view.committed_coordinates):
            captured.append((context.store, canonical(context.store.materialize()), context.store.generation))
            token.cancel('stop combined unaccepted trial')
        return result
    monkeypatch.setattr(element, 'compute_nonlinear_response', stop_trial)
    with pytest.raises(SolveCancelled):
        made.solve_spatial_couples(MOMENTS, distributed_pattern=pattern(made.model), cancellation_token=token)
    assert captured
    store, before, generation = captured[0]
    assert store.generation == generation and canonical(store.materialize()) == before
    assert not store.has_active_trial and not store.native_rotation_store.has_active_trial
    unlocked(made)


def test_final_progress_cancellation_does_not_issue_envelope(monkeypatch):
    made, token = owned(), CancellationToken()
    def forbidden(*a, **k): raise AssertionError('cancelled solve issued a checkpoint')
    monkeypatch.setattr(made, '_envelope', forbidden)
    with pytest.raises(SolveCancelled):
        made.solve_spatial_couples(MOMENTS, steps=1, cancellation_token=token,
            progress_callback=lambda event:token.cancel('stop at final accepted observation'))
    unlocked(made)


def test_adaptive_cutback_preserves_exact_accepted_origin(monkeypatch, tmp_path):
    import anyfesolver.nonlinear_static as solver
    made = owned(CASES[1])
    line = pattern(made.model)
    fixed = made.solve_spatial_couples(MOMENTS, distributed_pattern=line, steps=2)
    assert fixed.status == 'completed'
    made = owned(CASES[1])
    before = canonical(made._initial())
    original = solver.factorize
    attempts = []
    def reject_once(matrix, kind, *a, **k):
        if str(k.get('signature', '')).startswith('nonlinear.static_newton'):
            store = _RUN.get().store
            if not attempts:
                assert canonical(store.materialize()) == before
                attempts.append('rejected')
                raise RuntimeError('prescribed one-shot factorization rejection')
            if len(attempts) == 1:
                assert store.generation == 0 and canonical(store.materialize()) == before
                attempts.append('accepted-origin-restored')
        return original(matrix, kind, *a, **k)
    monkeypatch.setattr(solver, 'factorize', reject_once)
    adaptive = made.solve_spatial_couples(MOMENTS, distributed_pattern=line, steps=1,
        step_policy=AdaptiveForcePolicy(cutback_levels=1, growth=False))
    assert adaptive.status == 'completed'
    assert attempts == ['rejected', 'accepted-origin-restored']
    assert adaptive.checkpoint == fixed.checkpoint
    assert [s.load_factor for s in adaptive.backend_result.snapshots] == [.5, 1.]
    (tmp_path/'adaptive-checkpoint.json').write_bytes(adaptive.checkpoint)
    unlocked(made)
