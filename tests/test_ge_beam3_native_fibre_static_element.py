"""Actual state-store/dispatch integration for the retained physical-fibre core."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition,LoadCase
from anyfesolver.nonlinear_state import NonlinearStateStore,create_model_native_rotation_store,StateTransactionError
from anyfesolver.nonlinear_static import _assemble_nonlinear_system,solve_static_nonlinear
from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement,seal
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_fibre_line_program import model as source_model
from test_ge_beam3_schur_line_program import save


def problem(curved=True,plastic=True):
    source=source_model(curved=curved,plastic=plastic); op=source.mesh.elements[1].operator
    made=FEModel('private-native-physical-fibre')
    for i,point in enumerate(op.reference.coordinates,1): made.add_node(i,*point)
    element=NativeFibreStaticElement(1,(1,2,3),op.reference,op.section,order=4)
    made.add_element(1,element); made.materials[element.material_name]=element.section
    made.add_boundary_condition(BoundaryCondition('root',[1],{k:0. for k in ('ux','uy','uz','rx','ry','rz')}))
    return made,element


def store_for(m,e):
    states={1:e.init_model_bound_nonlinear_state(m.mesh,e.section,1)}
    store=NonlinearStateStore.from_shell_layouts((),states)
    store.attach_native_rotation_store(create_model_native_rotation_store(m,states,np.zeros(18)))
    return store


def sample():
    return np.array([[0.,0.,0.,0.,0.,0.],[.014,.002,-.001,.006,-.012,.008],[.035,.004,-.002,.015,-.02,.01]]).ravel()


@pytest.mark.parametrize('curved,plastic',[(False,False),(True,False),(True,True)])
def test_real_store_commit_discard_and_fixed_origins(curved,plastic,tmp_path):
    m,e=problem(curved,plastic); store=store_for(m,e); before=canonical(store.materialize())
    force,matrix,payload=_assemble_nonlinear_system(m,sample(),store,1)
    assert canonical(store.materialize())==before
    token=store.active_trial_token(); candidate=payload[1]
    e._validate(m.mesh,candidate,sample())
    store.commit(token,accepted_full_displacement=sample(),accepted_full_coordinates=e.operator.reference.coordinates+sample().reshape(3,6)[:,:3])
    assert store.generation==1 and canonical(store[1])==canonical(candidate)
    accepted=canonical(store.materialize()); previous=store[1]
    trial=.4*sample(); _,_,later=_assemble_nonlinear_system(m,trial,store,1)
    assert canonical(later[1]['origins'])==canonical(previous['response'].history)
    store.discard_trial(store.active_trial_token())
    assert canonical(store.materialize())==accepted and store.generation==1
    if plastic: assert any(row[2]>0 for station in previous['response'].history.stations for row in station.rows)
    save(tmp_path/'state.json',dict(state=previous,force=force,matrix=matrix.toarray(),discard_preserved=True,production_qualified=False))


@pytest.mark.parametrize('mutation',['potential','forces','origin','position_low','epoch'])
def test_resealed_corruption_rejected_before_commit(mutation,tmp_path):
    m,e=problem(); store=store_for(m,e); before=canonical(store.materialize())
    _,_,payload=_assemble_nonlinear_system(m,sample(),store,1); token=store.active_trial_token()
    broken=deepcopy(payload[1]); r=broken['response']
    if mutation=='potential': broken['response']=replace(r,potential=r.potential+.01)
    elif mutation=='forces': broken['response']=replace(r,resultants=r.resultants+.01)
    elif mutation=='origin': broken['origins']=r.history
    elif mutation=='position_low': broken['position_low']=broken['position_low']+1e-4
    else: broken['epoch']+=1
    broken=seal(broken)
    with pytest.raises(ValueError): store.set_trial_state(token,1,broken)
    store._fallback_trial[1]=broken
    with pytest.raises(ValueError):
        store.commit(token,accepted_full_displacement=sample(),accepted_full_coordinates=e.operator.reference.coordinates+sample().reshape(3,6)[:,:3])
    assert canonical(store.materialize())==before and store.generation==store.native_rotation_store.generation==0
    store.discard_trial(token)
    save(tmp_path/'rejected.json',dict(mutation=mutation,accepted_state=before.decode(),generation=store.generation))


def test_stale_foreign_and_missing_live_context():
    m,e=problem(); store=store_for(m,e)
    token=store.begin_trial(full_displacement=sample(),full_coordinates=e.operator.reference.coordinates+sample().reshape(3,6)[:,:3])
    view=store.native_element_rotation_view(token,1,(1,2,3),e.native_reference_directors(m.mesh)); context=store.native_material_context(token,1)
    state=store[1]
    with pytest.raises(ValueError,match='live'): e.compute_nonlinear_response(m.mesh,e.section,sample(),state)
    other=store_for(m,e); t2=other.begin_trial(full_displacement=np.zeros(18),full_coordinates=e.operator.reference.coordinates)
    foreign=other.native_material_context(t2,1)
    with pytest.raises(ValueError): e.compute_nonlinear_response(m.mesh,e.section,sample(),state,native_rotation_trial=view,native_material_context=foreign)
    other.discard_trial(t2); store.discard_trial(token)
    with pytest.raises(StateTransactionError): e.compute_nonlinear_response(m.mesh,e.section,sample(),state,native_rotation_trial=view,native_material_context=context)


def test_identical_pose_foreign_model_context_rejected_before_mechanics(monkeypatch):
    m,e=problem(); store=store_for(m,e); other_m,other_e=problem(); other=store_for(other_m,other_e)
    coords=e.operator.reference.coordinates+sample().reshape(3,6)[:,:3]
    token=store.begin_trial(full_displacement=sample(),full_coordinates=coords)
    other_token=other.begin_trial(full_displacement=sample(),full_coordinates=coords)
    view=store.native_element_rotation_view(token,1,(1,2,3),e.native_reference_directors(m.mesh))
    foreign=other.native_material_context(other_token,1)
    foreign.require_view(view)  # Pose-only compatibility is intentionally insufficient.
    def forbidden(*args,**kwargs): raise AssertionError('mechanics or state replay entered')
    monkeypatch.setattr(e,'_validate',forbidden)
    with pytest.raises(ValueError,match='foreign native'): e.compute_nonlinear_response(m.mesh,e.section,sample(),store[1],native_rotation_trial=view,native_material_context=foreign)
    store.discard_trial(token); other.discard_trial(other_token)


def test_supplied_state_must_be_current_store_state(monkeypatch):
    m,e=problem(); store=store_for(m,e)
    token=store.begin_trial(full_displacement=sample(),full_coordinates=e.operator.reference.coordinates+sample().reshape(3,6)[:,:3])
    view=store.native_element_rotation_view(token,1,(1,2,3),e.native_reference_directors(m.mesh)); context=store.native_material_context(token,1)
    wrong=deepcopy(store[1]); wrong['epoch']=9; wrong=seal(wrong)
    def forbidden(*args,**kwargs): raise AssertionError('mechanics or replay entered')
    monkeypatch.setattr(e,'_validate',forbidden)
    with pytest.raises(ValueError,match='current store'): e.compute_nonlinear_response(m.mesh,e.section,sample(),wrong,native_rotation_trial=view,native_material_context=context)
    store.discard_trial(token)


@pytest.mark.parametrize('curved,plastic',[(False,False),(True,False),(True,True)])
def test_actual_newton_driver_native_fibre_smoke(curved,plastic,tmp_path):
    m,e=problem(curved,plastic); load=LoadCase('tip'); load.add_nodal_load(3,forces=np.array([.35 if plastic else .035,-.012,.006]))
    result=solve_static_nonlinear(m,load,num_steps=2,max_iterations=12,tolerance=1e-10,num_layers=1,min_step_fraction=1.,record_increment_snapshots=True)
    save(tmp_path/'driver.json',dict(status=result.status,displacements=result.displacements,states=result.element_states))
    assert result.status=='completed',result.info
    e.validate_model_bound_nonlinear_state(m.mesh,e.section,result.element_states[1],1,expected_committed_total_u=result.displacements)
    assert len(result.snapshots)==2
    assert canonical(result.snapshots[1].element_states[1]['origins'])==canonical(result.snapshots[0].element_states[1]['response'].history)
    if plastic: assert any(row[2]>0 for station in result.element_states[1]['response'].history.stations for row in station.rows)
    with pytest.raises(ValueError,match='not qualified'): e.compute_mass_matrix(m.mesh,e.section)


def test_actual_failed_increment_keeps_virgin_history(tmp_path):
    m,e=problem(True,True); load=LoadCase('tip'); load.add_nodal_load(3,forces=np.array([.35,-.012,.006]))
    initial=e.init_model_bound_nonlinear_state(m.mesh,e.section,1)
    result=solve_static_nonlinear(m,load,num_steps=1,max_iterations=1,tolerance=1e-10,num_layers=1,min_step_fraction=1.)
    save(tmp_path/'failed-driver.json',dict(status=result.status,displacements=result.displacements,states=result.element_states))
    assert result.status!='completed'
    assert canonical(result.element_states[1])==canonical(initial)


def test_native_subulp_common_translation_keeps_coordinate_low_part(tmp_path):
    _,source=problem(True,False); reference=source.operator.reference
    coordinates=reference.coordinates+np.array([1e9,0.,0.])
    ref=Reference(coordinates,reference.nodal_triads)
    m=FEModel('native-subulp-coordinate-authority')
    for i,row in enumerate(coordinates,1): m.add_node(i,*row)
    e=NativeFibreStaticElement(1,(1,2,3),ref,source.section,order=4)
    m.add_element(1,e); m.materials[e.material_name]=e.section
    store=store_for(m,e); total=np.zeros((3,6)); total[:,0]=1e-9; total=total.ravel()
    assert np.array_equal(coordinates+total.reshape(3,6)[:,:3],coordinates)
    force,_,payload=_assemble_nonlinear_system(m,total,store,1)
    state=payload[1]
    assert np.array_equal(state['positions'],coordinates)
    assert np.array_equal(state['position_low'][:,0],np.full(3,1e-9))
    assert np.linalg.norm(force)<=1e-11
    token=store.active_trial_token(); store.commit(token,accepted_full_displacement=total,accepted_full_coordinates=coordinates)
    assert canonical(store[1])==canonical(state)
    save(tmp_path/'subulp.json',dict(state=state,force=force,translation_lost_in_high_only=True))
