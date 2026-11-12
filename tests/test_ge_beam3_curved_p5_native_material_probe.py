"""Real native material transaction checks; no qualification/activation claim."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from anyfesolver.nonlinear_element_evaluation import evaluate_nonlinear_element, NativeNonlinearEvaluationError
from anyfesolver.nonlinear_state import StateTransactionError
from docs.reference_cases import ge_beam3_curved_p5_native_material_probe as probe
from docs.reference_cases.ge_beam3_curved_p5_section_probe import SectionHistory
from test_ge_beam3_curved_p5_algebra_probe import reference, assert_scaled_close
from test_ge_beam3_curved_p5_native_chart_probe import sample
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law, mask


def session(two=False):
    model=FEModel('private-p5-native-material')
    refs=[reference(.4)]
    if two:
        second=reference(.4, twist=.1)
        refs.append(CurvedBeam3ReferenceGeometry(second.coordinates+[2., 0., 0.], second.nodal_triads))
    for i, ref in enumerate(refs):
        nodes=tuple(range(2*i+1, 2*i+4))
        for node, point in zip(nodes, ref.coordinates):
            if node not in model.mesh.nodes: model.add_node(node, *point)
        model.add_element(i+1, probe.PrivateP5NativeElement(i+1, nodes, ref, law()))
    return probe.NativeMaterialSession(model)


def snapshot(s):
    return (probe.canonical(s.store.materialize()), s.store.generation,
            s.store.native_rotation_store.committed_full_displacement.copy(),
            s.store.native_rotation_store.committed_rotation_matrices.copy())


def unchanged(s, before):
    after=snapshot(s)
    assert before[:2]==after[:2]
    assert np.array_equal(before[2], after[2]) and np.array_equal(before[3], after[3])


def accept(s, u):
    token=s.begin(u)
    for key in s.elements: s.evaluate(token, key)
    s.commit(token)


def test_actual_scalar_dispatch_and_atomic_material_rotation_commit(monkeypatch):
    s=session();before=snapshot(s);called=[];original=probe.evaluate_nonlinear_element
    def counted(*args, **kwargs):
        called.append(kwargs['state_token']);return original(*args, **kwargs)
    monkeypatch.setattr(probe, 'evaluate_nonlinear_element', counted)
    token=s.begin(sample());force, tangent, candidate=s.evaluate(token, 1)
    unchanged(s, before)
    assert called==[token] and any(mask(candidate['response']))
    assert candidate['origins']==before_histories(s)
    assert_scaled_close(tangent, tangent.T)
    assert force.shape==(18,)
    assert s.commit(token)==1
    assert s.store.generation==s.store.native_rotation_store.generation==1
    assert probe.canonical(s.store[1])==probe.canonical(candidate)
    assert probe.canonical(s.replay()[1])==probe.canonical(candidate)
    assert not s.store.has_active_trial


def before_histories(s): return s.store[1]['histories']


def test_line_search_replacement_uses_committed_origins_not_rejected_plastic_state():
    s=session();initial=snapshot(s)
    first=s.begin(sample()*1.5);_, _, rejected=s.evaluate(first, 1)
    second=s.begin(sample()*.5);_, _, chosen=s.evaluate(second, 1, tangent=False)
    assert rejected['histories']!=chosen['histories']
    assert rejected['origins']==chosen['origins']==before_histories(s)
    unchanged(s, initial)
    with pytest.raises(probe.NativeMaterialError): s.commit(first)
    s.commit(second)
    assert s.store[1]['histories']==chosen['histories']


def test_loading_unloading_reversal_and_replay_do_not_advance_state():
    s=session();previous=before_histories(s)
    for i, factor in enumerate((1., 1.8, .3, -1.2), 1):
        token=s.begin(sample()*factor);_, _, candidate=s.evaluate(token, 1)
        assert candidate['origins']==previous
        assert all(new.accumulated>=old.accumulated for new, old in zip(candidate['histories'], previous))
        s.commit(token);previous=candidate['histories']
        before=snapshot(s)
        for _ in range(2): s.replay()
        unchanged(s, before)
        assert s.store[1]['epoch']==i


def test_late_second_element_failure_keeps_all_committed_states(monkeypatch):
    s=session(True);before=snapshot(s);u=np.sin(np.arange(30))*.02;token=s.begin(u)
    s.evaluate(token, 1)
    def failed(*args, **kwargs): raise RuntimeError('second element local failure')
    monkeypatch.setattr(s.elements[2], '_solve', failed)
    with pytest.raises(RuntimeError, match='second'): s.evaluate(token, 2)
    with pytest.raises(probe.NativeMaterialError, match='every element'): s.commit(token)
    unchanged(s, before);s.discard(token);unchanged(s, before)
    assert not s.store.has_active_trial


def test_late_commit_validation_failure_is_atomic(monkeypatch):
    s=session(True);before=snapshot(s);token=s.begin(np.sin(np.arange(30))*.015)
    for key in s.elements: s.evaluate(token, key)
    def failed(*args, **kwargs): raise RuntimeError('late accepted replay failure')
    monkeypatch.setattr(s.elements[2], 'validate_model_bound_nonlinear_state', failed)
    with pytest.raises(RuntimeError, match='late accepted'): s.commit(token)
    unchanged(s, before);s.discard(token)


def test_shared_node_commit_and_next_trial_use_same_operator():
    s=session(True);u=np.sin(np.arange(30)+.2)*.02;accept(s, u)
    first=s.store[1];second=s.store[2]
    assert np.array_equal(first['committed_nodal_rotation_matrices'][2], second['committed_nodal_rotation_matrices'][0])
    before=snapshot(s);token=s.begin(u*1.1)
    for key in s.elements: s.evaluate(token, key)
    s.discard(token);unchanged(s, before)


@pytest.mark.parametrize('mutation', ['history', 'response', 'model', 'keys', 'epoch', 'positions', 'rotations'])
def test_resealed_state_mutations_rejected(mutation):
    s=session();accept(s, sample());state=deepcopy(s.store[1]);element=s.elements[1]
    if mutation=='history': state['histories']=(SectionHistory(),)+state['histories'][1:]
    elif mutation=='response': state['response']=replace(state['response'], potential=state['response'].potential+.01)
    elif mutation=='model': state['model_sha256']='0'*64
    elif mutation=='keys': state['unexpected']=True
    elif mutation=='epoch': state['epoch']=True
    elif mutation=='positions': state['committed_positions'][1, 0]+=.01
    else: state['committed_nodal_rotation_matrices'][1, 0, 0]+=.1
    state=probe.seal(state)
    before=snapshot(s)
    with pytest.raises(ValueError): element.validate_model_bound_nonlinear_state(s.model.mesh, None, state, 1)
    unchanged(s, before)


def test_mutated_store_trial_rejected_before_commit():
    s=session();before=snapshot(s);token=s.begin(sample());_, _, state=s.evaluate(token, 1)
    state['histories']=(SectionHistory(),)+state['histories'][1:]
    s.store.set_trial_state(token, 1, probe.seal(state))
    with pytest.raises(probe.NativeMaterialError, match='candidate changed'): s.commit(token)
    unchanged(s, before);s.discard(token)


def test_returned_candidate_does_not_alias_session_state():
    s=session();token=s.begin(sample());_, _, state=s.evaluate(token, 1)
    state['committed_total_u'][:]=99.
    s.commit(token)
    assert np.array_equal(s.store[1]['committed_total_u'], sample())


def test_absent_native_context_and_unsupported_routes_fail_closed():
    s=session();element=s.elements[1]
    with pytest.raises(NativeNonlinearEvaluationError):
        evaluate_nonlinear_element(element, s.model.mesh, None, sample(), s.store[1], 1, True,
            committed_states={}, state_token=None, element_id=1)
    with pytest.raises(probe.NativeMaterialError):
        element.compute_nonlinear_response(s.model.mesh, None, sample(), s.store[1])
    for name in ('compute_stiffness_matrix', 'compute_mass_matrix', 'compute_geometric_stiffness_matrix',
                 'compute_internal_forces', 'compute_stresses', 'to_dict'):
        with pytest.raises(probe.NativeMaterialError): getattr(element, name)(s.model.mesh, None)


def test_dirty_model_rejected_and_discard_still_available():
    s=session();before=snapshot(s);token=s.begin(sample());s.evaluate(token, 1)
    s.elements[1].section._yield*=2
    with pytest.raises(probe.NativeMaterialError, match='identity'): s.commit(token)
    s.discard(token);unchanged(s, before)


def test_native_session_force_tangent_directional_check_after_plastic_commit():
    s=session();accept(s, sample());u=sample()*1.3;direction=np.sin(np.arange(18)+.1)/8;h=1e-6
    outputs=[]
    for value in (u, u+h*direction, u-h*direction):
        token=s.begin(value);outputs.append(s.evaluate(token, 1));s.discard(token)
    base, plus, minus=outputs
    assert mask(base[2]['response'])==mask(plus[2]['response'])==mask(minus[2]['response'])
    assert_scaled_close((plus[0]-minus[0])/(2*h), base[1]@direction, 1e-7)
    assert s.store.generation==1


def test_real_reference_assembly_matches_scalar_dispatch_and_can_commit():
    s=session(True);u=np.sin(np.arange(30))*.015;token=s.begin(u)
    expected_force=np.zeros(30);expected_tangent=np.zeros((30, 30))
    for key, element in s.elements.items():
        force, tangent, _=s.evaluate(token, key);ids=np.array(element.get_dof_mapping(s.model.mesh))
        expected_force[ids]+=force;expected_tangent[np.ix_(ids, ids)]+=tangent
    s.discard(token)
    token, force, tangent=s.assemble(u)
    assert_scaled_close(force, expected_force)
    assert_scaled_close(tangent.toarray(), expected_tangent)
    s.commit(token)
    assert s.store.generation==1 and len(s.replay())==2


def test_real_assembly_failure_discards_whole_native_candidate(monkeypatch):
    s=session(True);before=snapshot(s)
    def failed(*args, **kwargs): raise RuntimeError('real assembly late failure')
    monkeypatch.setattr(s.elements[2], '_solve', failed)
    with pytest.raises(RuntimeError, match='real assembly'): s.assemble(np.sin(np.arange(30))*.015)
    unchanged(s, before)
    assert not s.store.has_active_trial and not s.store.native_rotation_store.has_active_trial


def test_detached_native_view_cannot_be_reused_after_discard():
    s=session();element=s.elements[1];token=s.begin(sample())
    view=s.store.native_element_rotation_view(token, 1, element.node_ids, element.native_reference_directors(s.model.mesh))
    s.discard(token)
    with pytest.raises(StateTransactionError):
        element.compute_nonlinear_response(s.model.mesh, None, sample(), s.store[1], native_rotation_trial=view)
    token=s.begin(sample())
    with pytest.raises(probe.NativeMaterialError, match='detached'):
        element.compute_nonlinear_response(s.model.mesh, None, sample(), s.store[1], native_rotation_trial=view)
    s.discard(token)


def test_dof_ownership_change_rejected_before_mechanics(monkeypatch):
    s=session();token=s.begin(sample());before=snapshot(s)
    node=s.model.mesh.nodes[2];node.dofs=list(reversed(node.dofs))
    def forbidden(*args, **kwargs): raise AssertionError('must not evaluate changed layout')
    monkeypatch.setattr(s.elements[1], '_solve', forbidden)
    with pytest.raises(probe.NativeMaterialError, match='ownership'): s.evaluate(token, 1)
    s.discard(token);unchanged(s, before)


def test_failed_initialization_does_not_publish_section_registration(monkeypatch):
    model=FEModel('private-init-failure');ref=reference()
    for i, point in enumerate(ref.coordinates, 1): model.add_node(i, *point)
    element=probe.PrivateP5NativeElement(1, (1, 2, 3), ref, law())
    model.add_element(1, element);before=dict(model.materials)
    def failed(*args, **kwargs): raise RuntimeError('initial local solve failed')
    monkeypatch.setattr(element, '_solve', failed)
    with pytest.raises(RuntimeError, match='initial local'): probe.NativeMaterialSession(model)
    assert model.materials==before and element._native_session is None
