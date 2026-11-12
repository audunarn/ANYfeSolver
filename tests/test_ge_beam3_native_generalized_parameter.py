"""Continuation column checked against separately re-solved native trials."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver._ge_beam3_native_generalized_parameter import parameter_column
from anyfesolver._ge_beam3_native_generalized_loading import assemble_distributed_trial,DistributedPattern
from anyfesolver._ge_beam3_native_generalized_element import seal
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from anyfesolver.nonlinear_state import NonlinearStateStore,create_model_native_rotation_store
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_native_generalized_restart import make,pattern,scale
from test_ge_beam3_native_fibre_static_element import sample
from test_ge_beam3_native_spatial_couples import left_jacobian
from test_ge_beam3_schur_line_program import save


def setup(case):
    m=make(case);n=m.mesh.dof_manager.total_dofs
    states={i:e.init_model_bound_nonlinear_state(m.mesh,e.section,1) for i,e in m.mesh.elements.items()}
    store=NonlinearStateStore.from_shell_layouts((),states)
    store.attach_native_rotation_store(create_model_native_rotation_store(m,states,np.zeros(n)))
    u=np.zeros(n)
    for node in sorted(m.mesh.nodes):
        if node!=1:u[list(m.mesh.dof_manager.get_node_dofs(node))]=sample().reshape(3,6)[min(node-1,2)]
    return m,store,u


def net(m,store,u,load,p,moment):
    f,_,states,external=assemble_distributed_trial(m,u,store,scale(load,p))
    value=f-external
    for node,*v in moment.rows:
        dofs=list(m.mesh.dof_manager.get_node_dofs(node)[3:])
        # Independent closed Rodrigues Jacobian, not production pullback.
        eid,e=next((i,e) for i,e in m.mesh.elements.items() if node in e.node_ids)
        previous=store[eid]['committed_total_u'].reshape(3,6)[e.node_ids.index(node),3:]
        value[dofs]-=p*left_jacobian(u[dofs]-previous).T@np.array(v)
    return value,states


@pytest.mark.parametrize('case',['straight-elastic','curved-plastic','connected-plastic'])
def test_parameter_directional(case,tmp_path):
    m,store,u=setup(case);load=pattern(m);moment=SpatialNodalMoments(((3,.15,-.1,.08),));p=.7
    before=canonical(store.materialize());_,states=net(m,store,u,load,p,moment)
    result=parameter_column(m,store,states,load,nodal_moments=moment);expected=result['column'];errors=[];observed=[]
    states=deepcopy(dict(states))  # Preserve this trial before issuing perturbations.
    for h in (2e-4,1e-4,5e-5):
        fp,_=net(m,store,u,load,p+h,moment);fm,_=net(m,store,u,load,p-h,moment)
        v=(fp-fm)/(2*h);observed.append(v);errors.append(float(np.linalg.norm(v-expected)/max(1.,np.linalg.norm(expected))))
    assert max(errors)<=1e-7 and canonical(store.materialize())==before
    if case!='straight-elastic':assert any(sum(s.accumulated)>0 for state in states.values() for s in state['response'].history.stations)
    save(tmp_path/'column.json',dict(result=result,steps=(2e-4,1e-4,5e-5),errors=errors,observed=observed,committed_unchanged=True))
    store.discard_trial(store.active_trial_token())


def test_accepted_origin_and_zero_direction(tmp_path):
    m,store,u=setup('curved-plastic');load=pattern(m);moment=SpatialNodalMoments(((3,.15,-.1,.08),))
    _,states=net(m,store,u,load,.7,moment);state=deepcopy(states[1])
    store.commit(store.active_trial_token(),accepted_full_displacement=u,accepted_full_coordinates=state['positions'])
    before=canonical(store.materialize());next_u=.9*u
    _,trial=net(m,store,next_u,load,.4,moment)
    result=parameter_column(m,store,trial,load,nodal_moments=moment)
    zero=parameter_column(m,store,trial,DistributedPattern(LinePattern(()),()))
    np.testing.assert_array_equal(zero['column'],np.zeros(18))
    assert all(not np.any(v) for v in zero['internal_parameter_lifts'].values())
    observed=[];errors=[]
    for h in (2e-4,1e-4,5e-5):
        plus,_=net(m,store,next_u,load,.4+h,moment);minus,_=net(m,store,next_u,load,.4-h,moment)
        value=(plus-minus)/(2*h);observed.append(value)
        errors.append(float(np.linalg.norm(value-result['column'])/max(1.,np.linalg.norm(result['column']))))
    assert max(errors)<=1e-7 and canonical(store.materialize())==before and store.generation==1
    store.discard_trial(store.active_trial_token())
    save(tmp_path/'accepted-origin.json',dict(result=result,zero=zero,errors=errors,observed=observed,generation=1,accepted_unchanged=True))


@pytest.mark.parametrize('kind',['missing','foreign','stale','state','jacobian','pattern','moment','absent-node','live-mutation'])
def test_parameter_authority(kind,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_generalized_parameter as module
    m,store,u=setup('straight-elastic');load=pattern(m);before=canonical(store.materialize())
    _,states=net(m,store,u,load,.7,SpatialNodalMoments(((3,.1,.0,.0),)))
    states=deepcopy(dict(states));moment=None
    if kind=='missing':store.discard_trial(store.active_trial_token())
    elif kind=='foreign':
        other,foreign,v=setup('straight-elastic');net(other,foreign,v,pattern(other),.7,SpatialNodalMoments(((3,.1,.0,.0),)))
        with pytest.raises(ValueError):parameter_column(m,foreign,states,load)
        assert not foreign.has_active_trial
        assert canonical(store.materialize())==before
        store.discard_trial(store.active_trial_token())
        save(tmp_path/'rejection.json',dict(kind=kind,rejected=True,committed_unchanged=True));return
    elif kind=='stale':net(m,store,.9*u,load,.7,SpatialNodalMoments(((3,.1,.0,.0),)))
    elif kind=='state':states[1]['epoch']+=1;states[1]=seal(states[1])
    elif kind=='jacobian':
        r=states[1]['response'];j=r.full_spatial_jacobian.copy();j[0,18]+=.01
        states[1]['response']=replace(r,full_spatial_jacobian=j);states[1]=seal(states[1])
    elif kind=='pattern':object.__setattr__(load,'signature','0'*64)
    elif kind=='moment':moment=((3,.1,0.,0.),)
    elif kind=='absent-node':moment=SpatialNodalMoments(((999,.1,0.,0.),))
    elif kind=='live-mutation':
        original=module.line_work
        def changed(*a,**kw):
            result=original(*a,**kw);object.__setattr__(load,'signature','0'*64);return result
        monkeypatch.setattr(module,'line_work',changed)
    with pytest.raises(ValueError):parameter_column(m,store,states,load,nodal_moments=moment)
    assert not store.has_active_trial and canonical(store.materialize())==before
    save(tmp_path/'rejection.json',dict(kind=kind,rejected=True,committed_unchanged=True))
