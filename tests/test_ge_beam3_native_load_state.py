"""Actual native assembly with explicit load-trial ownership; not public routing."""

from copy import deepcopy

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.nonlinear_state import NonlinearStateStore, create_model_native_rotation_store, StateTransactionError
from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement, LoadStateStore
from anyfesolver._ge_beam3_p5_loads.core import canonical, seal
from anyfesolver._ge_beam3_p5_loads import codec
from test_ge_beam3_centered_native import problem as centered_problem


def problem(*, shift=0., plastic=False, count=1, line_force=(.03,-.02,.01)):
    original,_ = centered_problem(shift=shift,plastic=plastic,count=count)
    model = FEModel('native-loaded-beam')
    for i,node in original.mesh.nodes.items(): model.add_node(i,*node.coords())
    for i,previous in original.mesh.elements.items():
        element = NativeP5BeamElement(i,previous.node_ids,previous.core.reference,previous.core.section,
                                     line_force=np.array(line_force))
        model.add_element(i,element); model.materials[element.material_name] = element.core.section
    for boundary in original.boundary_conditions: model.add_boundary_condition(deepcopy(boundary))
    return model


def store_for(model, states=None, total=None, cls=LoadStateStore):
    if states is None:
        states = {i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1) for i,e in model.mesh.elements.items()}
    if total is None: total = np.zeros(model.mesh.dof_manager.total_dofs)
    store = cls.from_shell_layouts((),states)
    store.attach_native_rotation_store(create_model_native_rotation_store(model,states,total))
    return store


def assemble(model,store,total,parameter):
    with store.load_parameter_scope(parameter):
        return _assemble_nonlinear_system(model,total,store,1)


def commit(model,store,total):
    coordinates = np.array([node.coords() for node in model.mesh.nodes.values()])+total.reshape(-1,6)[:,:3]
    store.commit(store.active_trial_token(),accepted_full_displacement=total,accepted_full_coordinates=coordinates)


def equilibrium(model,store,total,parameter):
    """Small test-only Newton loop, using the actual assembly/state seam.

    All line work is already in F=V_q-p*W_q after local condensation; do not
    subtract a second equivalent nodal vector. The root is fully fixed.
    """
    total = total.copy(); free = np.arange(6,len(total))
    try:
        for _ in range(12):
            force,matrix,_ = assemble(model,store,total,parameter)
            norm = np.linalg.norm(force[free])
            if norm <= 1e-11:
                commit(model,store,total)
                return total,force
            increment = np.linalg.solve(matrix.toarray()[np.ix_(free,free)],-force[free])
            for cut in range(8):
                candidate = total.copy(); candidate[free] += (.5**cut)*increment
                trial,_,_ = assemble(model,store,candidate,parameter)
                if np.linalg.norm(trial[free]) < norm:
                    total = candidate; break
            else: raise AssertionError('test Newton line search exhausted')
        raise AssertionError('test Newton iteration bound exhausted')
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())


def test_native_assembly_binds_load_to_trial_commit_and_replay():
    model = problem(); store = store_for(model); total = np.zeros(18)
    before = canonical(store.materialize())
    force,_,trial = assemble(model,store,total,.5)
    assert store.committed_load_parameter == 0.
    assert trial[1]['material_state']['load_parameter'] == .5
    assert canonical(store.materialize()) == before and np.linalg.norm(force) > 0.
    commit(model,store,total)
    assert store.committed_load_parameter == .5 and store.generation == 1
    state = store[1]; element = model.mesh.elements[1]
    replay = element.validate_model_bound_nonlinear_state(model.mesh,element.core.section,state,1,
        expected_committed_total_u=total)
    assert canonical(replay) == canonical(state)
    assert not store.has_active_trial and not store.native_rotation_store.has_active_trial


def test_unscoped_and_ordinary_stores_cannot_silently_supply_zero_load():
    model = problem(); total = np.zeros(18)
    for cls in (LoadStateStore,NonlinearStateStore):
        store = store_for(model,cls=cls); before = canonical(store.materialize())
        with pytest.raises((ValueError,StateTransactionError)):
            _assemble_nonlinear_system(model,total,store,1)
        assert canonical(store.materialize()) == before
        assert not store.has_active_trial and not store.native_rotation_store.has_active_trial


def test_loaded_native_elastic_equilibrium_and_reactions():
    model = problem(); store = store_for(model); total = np.zeros(18)
    for value in (.5,1.): total,force = equilibrium(model,store,total,value)
    assert np.linalg.norm(total) > 0. and np.linalg.norm(force[6:]) <= 1e-11
    assert store.committed_load_parameter == 1.
    element = model.mesh.elements[1]
    recovered = element.recover_native_fields(model.mesh,store[1])
    assert recovered['load_parameter'] == 1.
    assert recovered['load_policy'] == 'SPATIAL_DEAD_FORCE_PER_REFERENCE_ARCLENGTH_V1'
    from docs.reference_cases.ge_beam3_centered_line_load_oracle import evaluate
    state = store[1]['material_state']; response = state['response']
    applied = evaluate(element.core.reference.coordinates,total.reshape(3,6)[:,:3],
        response.local_rotations,element.core.line_force)
    assert np.linalg.norm(force[:3]+applied.nodal_measures.sum()*element.core.line_force) <= 1e-11
    current = state['committed_positions']+state['committed_position_low']
    moment = sum((np.cross(current[i]-current[0],applied.nodal_measures[i]*element.core.line_force)
                  for i in range(3)),np.zeros(3))
    moment += sum((np.cross(response.local_rotations[c]@applied.cell_lift_integrals[c],element.core.line_force)
                   for c in (0,1)),np.zeros(3))
    assert np.linalg.norm(force[3:6]+moment) <= 1e-11


@pytest.mark.parametrize('phase',['set','commit'])
def test_resealed_parameter_mismatch_is_rejected_before_any_accepted_swap(phase):
    model = problem(); store = store_for(model); total = np.zeros(18)
    before = canonical(store.materialize()); assemble(model,store,total,.5)
    token = store.active_trial_token(); bad = deepcopy(store.trial_view(token)[1])
    bad['material_state']['load_parameter'] = .75
    bad['material_state'] = seal(bad['material_state']); bad = seal(bad)
    try:
        with pytest.raises(StateTransactionError,match='load parameter disagrees'):
            if phase == 'set': store.set_trial_state(token,1,bad)
            else:
                store._fallback_trial[1] = bad
                commit(model,store,total)
        assert canonical(store.materialize()) == before
        assert store.generation == store.native_rotation_store.generation == 0
    finally: store.discard_trial(token)


@pytest.mark.parametrize('value',[None,True,1,float('nan'),float('inf')])
def test_parameter_scope_rejects_implicit_or_nonfinite_values(value):
    store = LoadStateStore()
    with pytest.raises(ValueError,match='float load parameter'):
        with store.load_parameter_scope(value): pass
    assert not store.has_active_trial


def test_scope_exception_cancels_only_its_unaccepted_trial():
    model = problem(); store = store_for(model); total = np.zeros(18)
    before = canonical(store.materialize())
    with pytest.raises(RuntimeError,match='cancel fixture'):
        with store.load_parameter_scope(.5):
            _assemble_nonlinear_system(model,total,store,1)
            raise RuntimeError('cancel fixture')
    assert canonical(store.materialize()) == before and store.committed_load_parameter == 0.
    assert not store.has_active_trial and not store.native_rotation_store.has_active_trial


def test_parameter_is_bound_to_exact_live_token_and_thread():
    from concurrent.futures import ThreadPoolExecutor
    model = problem(); store = store_for(model); total = np.zeros(18)
    assemble(model,store,total,.5); token = store.active_trial_token()
    try:
        import threading
        assert store._load_trial[2] is threading.current_thread()
        assert store.native_load_parameter(token,1) == .5
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(store.native_load_parameter,token,1)
            with pytest.raises(StateTransactionError,match='another trial or thread'): future.result(timeout=2.)
        with store.load_parameter_scope(.75):
            with pytest.raises(StateTransactionError,match='cannot be rebound'):
                store.native_load_parameter(token,1)
        with pytest.raises(StateTransactionError,match='bound native element'): store.native_load_parameter(token,999)
    finally: store.discard_trial(token)
    with pytest.raises(StateTransactionError): store.native_load_parameter(token,1)


def test_force_direction_is_model_bound_and_cannot_change_during_history():
    model = problem(); element = model.mesh.elements[1]; store = store_for(model)
    before = canonical(store.materialize())
    element.core.line_force.setflags(write=True); element.core.line_force[0] *= 1.1
    with pytest.raises(ValueError,match='model identity'):
        assemble(model,store,np.zeros(18),.5)
    assert canonical(store.materialize()) == before and not store.has_active_trial


def test_loaded_plastic_program_and_typed_split_restart_match():
    parameters = (.5,1.,0.,-.5)
    model = problem(plastic=True); store = store_for(model); total = np.zeros(18)
    previous = store[1]['material_state']['histories']; checkpoint = None
    for parameter in parameters:
        total,_ = equilibrium(model,store,total,parameter)
        state = store[1]['material_state']
        assert state['origins'] == previous
        assert all(a.accumulated >= b.accumulated for a,b in zip(state['histories'],previous))
        previous = state['histories']
        if parameter == 1.:
            assert any(h.accumulated > 0. for h in previous)
            checkpoint = (model.mesh.elements[1].serialize_native_material_state(model.mesh,store[1]),total.copy())
    fresh = problem(plastic=True); element = fresh.mesh.elements[1]
    decoded = element.validate_model_bound_nonlinear_state(fresh.mesh,element.core.section,checkpoint[0],1,
        expected_committed_total_u=checkpoint[1])
    resumed = store_for(fresh,{1:decoded},checkpoint[1]); new_total = checkpoint[1]
    assert resumed.committed_load_parameter == 1.
    for parameter in parameters[2:]: new_total,_ = equilibrium(fresh,resumed,new_total,parameter)
    assert np.array_equal(new_total,total)
    assert canonical(resumed.materialize()) == canonical(store.materialize())
    assert canonical(element.serialize_native_material_state(fresh.mesh,resumed[1])) == canonical(
        model.mesh.elements[1].serialize_native_material_state(model.mesh,store[1]))


@pytest.mark.parametrize('mutation',['missing','integer','nonfinite','old_schema'])
def test_loaded_codec_rejects_missing_or_untyped_parameter(mutation):
    model = problem(); element = model.mesh.elements[1]; state = store_for(model)[1]
    if mutation == 'missing': del state['material_state']['load_parameter']
    if mutation == 'integer': state['material_state']['load_parameter'] = 0
    if mutation == 'nonfinite': state['material_state']['load_parameter'] = float('inf')
    if mutation == 'old_schema':
        envelope = dict(schema='GE_BEAM3_P5_CENTERED_TYPED_STATE_JSON_V3',payload=canonical(state).decode('ascii'))
        with pytest.raises(ValueError): codec.decode(envelope,order=8)
    else:
        with pytest.raises(ValueError): codec.encode(state,order=8)


@pytest.mark.parametrize('plastic',[False,True])
def test_shared_node_loaded_path_is_identical_under_large_exact_translation(plastic):
    made = []
    for shift in (0.,2.**40):
        model = problem(plastic=plastic,count=2,shift=shift); store = store_for(model); total = np.zeros(30)
        for parameter in (.5,1.): total,_ = equilibrium(model,store,total,parameter)
        made.append((total,tuple(canonical(store[i]['material_state']['response']) for i in (1,2))))
        for i in (1,2): assert store[i]['material_state']['load_parameter'] == 1.
        first = store[1]['committed_nodal_rotation_matrices'][2]
        second = store[2]['committed_nodal_rotation_matrices'][0]
        assert np.array_equal(first,second)
    assert np.array_equal(made[0][0],made[1][0]) and made[0][1] == made[1][1]


def test_restart_store_rejects_mixed_element_load_parameters():
    model = problem(count=2); store = store_for(model); total = np.zeros(30)
    try:
        assemble(model,store,total,.5); a = deepcopy(dict(store.trial_view(store.active_trial_token())))
        assemble(model,store,total,.75); b = deepcopy(dict(store.trial_view(store.active_trial_token())))
        with pytest.raises(StateTransactionError,match='parameters disagree'):
            LoadStateStore.from_shell_layouts((),{1:a[1],2:b[2]})
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())


def test_detached_resealed_parameter_change_fails_loaded_origin_replay():
    model = problem(); store = store_for(model); total = np.zeros(18)
    assemble(model,store,total,.5); commit(model,store,total)
    state = store[1]; element = model.mesh.elements[1]
    state['material_state']['load_parameter'] = 0.
    state['material_state'] = seal(state['material_state']); state = seal(state)
    with pytest.raises(ValueError,match='replay mismatch'):
        element.validate_model_bound_nonlinear_state(model.mesh,element.core.section,state,1)


@pytest.mark.parametrize('plastic',[False,True])
def test_native_chart_load_parameter_derivative_matches_actual_assembly(plastic):
    from test_ge_beam3_curved_p5_native_chart_probe import sample
    model = problem(plastic=plastic,shift=2.**40); store = store_for(model); total = sample()
    element = model.mesh.elements[1]; step = 1e-5
    try:
        _,_,states = assemble(model,store,total,.7); state = deepcopy(states[1])
        token = store.active_trial_token(); context = store.native_material_context(token,1)
        view = store.native_element_rotation_view(token,1,element.node_ids,element.native_reference_directors(model.mesh))
        derivative = element.native_load_parameter_derivative(model.mesh,state,
            native_rotation_trial=view,native_material_context=context)
        spatial,_ = element.core._load_parameter_derivative(state['material_state'])
        assert np.linalg.norm(spatial-derivative['residual_parameter_derivative']) > 1e-8
        plus,_,plus_states = assemble(model,store,total,.7+step)
        branch_plus = tuple(s.response.plastic_active for s in plus_states[1]['material_state']['response'].stations)
        minus,_,minus_states = assemble(model,store,total,.7-step)
        branch_minus = tuple(s.response.plastic_active for s in minus_states[1]['material_state']['response'].stations)
        assert branch_plus == branch_minus == tuple(s.response.plastic_active for s in state['material_state']['response'].stations)
        assert np.linalg.norm((plus-minus)/(2*step)-derivative['residual_parameter_derivative']) <= 1e-7
        assert not derivative['residual_parameter_derivative'].flags.writeable
        with pytest.raises(StateTransactionError):
            element.native_load_parameter_derivative(model.mesh,state,native_rotation_trial=view,native_material_context=context)
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())


def test_packaged_load_work_is_exact_extraction_of_checked_probe():
    import ast
    import inspect
    from docs.reference_cases.ge_beam3_centered_line_load_probe import load_work as source
    from anyfesolver._ge_beam3_p5_loads.work import load_work as packaged
    assert ast.dump(ast.parse(inspect.getsource(source))) == ast.dump(ast.parse(inspect.getsource(packaged)))


def test_zero_line_force_is_byte_identical_to_preserved_centered_native():
    from test_ge_beam3_curved_p5_native_chart_probe import sample
    old,_ = centered_problem(shift=2.**40,plastic=True)
    new = problem(shift=2.**40,plastic=True,line_force=(0.,0.,0.)); total = sample()
    old_store = store_for(old,cls=NonlinearStateStore); new_store = store_for(new)
    try:
        a,k_a,states_a = _assemble_nonlinear_system(old,total,old_store,1)
        b,k_b,states_b = assemble(new,new_store,total,2.)
        assert np.array_equal(a,b) and np.array_equal(k_a.toarray(),k_b.toarray())
        assert canonical(states_a[1]['material_state']['response']) == canonical(states_b[1]['material_state']['response'])
    finally:
        for store in (old_store,new_store):
            if store.has_active_trial: store.discard_trial(store.active_trial_token())


@pytest.mark.parametrize('action',['discard','replace'])
def test_foreign_thread_cannot_discard_or_replace_owned_load_trial(action):
    from concurrent.futures import ThreadPoolExecutor
    model = problem(); store = store_for(model); total = np.zeros(18)
    before = canonical(store.materialize()); assemble(model,store,total,.5)
    token = store.active_trial_token()
    def attempt():
        if action == 'discard': store.discard_trial(token)
        else: assemble(model,store,total,.75)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(attempt)
            with pytest.raises(StateTransactionError,match='another thread'): future.result(timeout=3.)
        assert store.active_trial_token() is token
        assert store.native_load_parameter(token,1) == .5
        assert canonical(store.materialize()) == before
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())
