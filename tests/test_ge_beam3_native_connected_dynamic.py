"""Prepared connected native development controls; all UNEXECUTED."""
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from examples.native_connected_dynamic_dev import build,velocity
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver.control import CancellationToken


def test_independent_connected_physical_reference_polynomial_identities():
    from examples.native_connected_balance_reference import (
        REFERENCE_POSITIONS,physical_reference,clamp_reference,case_velocity)
    velocity_global=np.zeros(78);v=np.array((.01,.02,.03));omega=2.
    for n in range(5):velocity_global[6*n:6*n+3]=v
    for k in range(4):velocity_global[30+3*k]=omega
    state=dict(positions=REFERENCE_POSITIONS.copy(),position_low=np.zeros((5,3)),
        nodal_frames=np.tile(np.eye(3),(5,1,1)),cell_rotations=np.tile(np.eye(3),(2,2,1,1)),
        resultants=np.zeros((2,18)),velocity=velocity_global,acceleration=np.zeros(78))
    ref=physical_reference(state)
    # Integral length2, density2; four halfcells share nodes but not material.
    assert abs(ref['kinetic_energy']-(.5*4.*(v@v)+.5*.04*omega**2))<1e-14
    assert np.linalg.norm(ref['linear_momentum']-4.*v)<1e-14
    assert np.linalg.norm(ref['angular_momentum']-np.array((.04*omega,0.,0.)))<1e-14
    assert ref['stored_elastic_energy']==ref['mixed_potential']==0.
    # All three separated native blocks are populated: the earlier interleaved
    # diagnostic layout cannot accidentally pass this dot-product control.
    state['resultants'][0,0]=3.;state['resultants'][0,3]=5.;state['resultants'][0,6]=7.
    ref=physical_reference(state);expected=.5*.5*((9.+25.)/1e5+(49./3.)/30.)
    assert abs(ref['stored_elastic_energy']-expected)<1e-15
    assert abs(ref['mixed_potential']+expected)<1e-15
    assert abs(ref['compatibility_work_witness']+2.*expected)<1e-15
    assert abs(ref['independent_p_dot_resultant_residual']+2.*expected)<1e-15
    assert abs(np.dot(state['resultants'].ravel(),ref['independent_resultant_residual'])-ref['independent_p_dot_resultant_residual'])<1e-15
    support=clamp_reference(state,.00025,True)
    assert np.array_equal(support['force_impulse_ns'],np.array((-.00075,0.,0.)))
    assert support['work_j']==0.
    # Independent root hand-integration of this fixed two-element initializer.
    # These are analytic inputs/control values, never observed solver outcomes.
    for clamped,energy,linear,angular in (
        (False,.05230416666666667,(.04,.125,0.),(.061,-.012833333333333333,0.)),
        (True,.0516,(.035,.10,.005),(.061,-.008666666666666666,.020833333333333332))):
        state['resultants'][:]=0.;state['velocity'][:]=0.
        physical=tuple(6*n+a for n in range(5)for a in range(3))+tuple(range(30,42))
        state['velocity'][list(physical)]=case_velocity(clamped)
        ref=physical_reference(state)
        assert abs(ref['kinetic_energy']-energy)<1e-14
        assert np.linalg.norm(ref['linear_momentum']-linear)<1e-14
        assert np.linalg.norm(ref['angular_momentum']-angular)<1e-14


def test_independent_connected_clamp_work_and_case_refusal():
    from examples.native_connected_balance_reference import (
        REFERENCE_POSITIONS,elastic_reference,clamp_reference,require_case_definition,_finite_errors)
    owner=build();before=owner.checkpoint()if owner.current is not None else None
    require_case_definition(owner,False)
    with pytest.raises(ValueError):require_case_definition(owner,True)
    assert owner.current is None and before is None
    def exp(vector):
        angle=float(np.linalg.norm(vector));x,y,z=vector
        cross=np.array(((0.,-z,y),(z,0.,-x),(-y,x,0.)))
        if angle==0.:return np.eye(3)
        return np.eye(3)+np.sin(angle)/angle*cross+(1.-np.cos(angle))/angle**2*(cross@cross)
    state=dict(positions=REFERENCE_POSITIONS.copy(),position_low=np.zeros((5,3)),
        nodal_frames=np.tile(np.eye(3),(5,1,1)),cell_rotations=np.tile(np.eye(3),(2,2,1,1)),
        resultants=np.zeros((2,18)),velocity=np.zeros(78),acceleration=np.zeros(78))
    state['cell_rotations'][0,0]=exp(np.array((.03,-.02,.01)))
    state['resultants'][0,:3]=(.5,-.3,.2);state['resultants'][0,6:9]=(.7,-.2,.4)
    support=clamp_reference(state,1.,True)
    couple=support['moment_impulse_nms']-np.cross(state['positions'][0],support['force_impulse_ns'])
    direction=np.array((.2,-.3,.4))
    for h in(1e-5,3e-6):
        plus={**state,'nodal_frames':state['nodal_frames'].copy()}
        minus={**state,'nodal_frames':state['nodal_frames'].copy()}
        plus['nodal_frames'][0]=exp(h*direction);minus['nodal_frames'][0]=exp(-h*direction)
        fd=(elastic_reference(plus)['mixed_potential']-elastic_reference(minus)['mixed_potential'])/(2*h)
        assert abs(fd-couple@direction)<=1e-5*np.linalg.norm(couple)*np.linalg.norm(direction)
    bad={**state,'cell_rotations':state['cell_rotations'].copy()};bad['cell_rotations'][0,0,0,0]=float('nan')
    with pytest.raises(ValueError):elastic_reference(bad)
    with pytest.raises(ValueError):_finite_errors({'reported_error':float('nan')})


def test_global_shared_map_mass_and_rigid_reference():
    owner=build();state=owner.initialize(physical_velocity=velocity(owner))
    assert owner.size==78 and len(owner.physical)==27
    assert owner.maps[0][12:18]==owner.maps[1][:6]
    r,j,fi,m,*_=owner._evaluate(state)
    assert np.count_nonzero(m[list(owner.algebraic),:])==0
    assert np.count_nonzero(m[:,list(owner.algebraic)])==0
    p=list(owner.p);np.linalg.cholesky(m[np.ix_(p,p)])
    before=state;candidate=owner.trial(state,dt=.00025);state=owner.commit(candidate)
    assert np.linalg.norm(state.positions-owner.reference_positions-np.array((.01,.02,0.))*state.time)<1e-11
    expected=rotation(np.array((2.*state.time,0.,0.)))
    assert np.linalg.norm(state.cell_rotations-np.tile(expected,(owner.e,2,1,1)))<1e-9
    assert np.linalg.norm(state.nodal_frames-np.tile(expected,(owner.n,1,1)))<1e-9
    assert np.linalg.norm(state.acceleration[list(owner.p)])<1e-8
    # Shared-node mass includes BOTH adjoining half cells, not one owner only.
    shared=owner.node_index[30];assert m[6*shared,6*shared]>m[0,0]
    assert before is not owner.current


def test_complete_global_work_and_spatial_tangent():
    owner=build();state=owner.initialize();direction=np.zeros(owner.size)
    direction[6*owner.node_index[30]+1]=1e-5
    direction[6*owner.node_index[30]+5]=1e-5
    direction[6*owner.n+2]=1e-5;direction[6*owner.n+8]=-1e-5
    r,j,_,_,_,_,account=owner._evaluate(state)
    for h in (1e-2,1e-3,1e-4):
        plus=owner._advance(state,h*direction);minus=owner._advance(state,-h*direction)
        rp,_,_,_,_,_,ap=owner._evaluate(plus);rm,_,_,_,_,_,am=owner._evaluate(minus)
        assert abs((ap['elastic_energy']-am['elastic_energy'])/(2*h)-r@direction)<=1e-5*1e-5
        assert np.linalg.norm((rp-rm)/(2*h)-j@direction)<=1e-5*max(np.linalg.norm(j@direction),1e-8)


def test_atomic_candidate_mutation_discard_and_fresh_restart():
    owner=build();state=owner.initialize(physical_velocity=velocity(owner));raw0=owner.checkpoint()
    candidate=owner.trial(state,dt=.00025)
    forged=replace(candidate,predecessor='0'*64)
    with pytest.raises(ValueError):owner.commit(forged)
    owner.discard(candidate);assert owner.current is state and owner.checkpoint()==raw0
    first=owner.commit(owner.trial(state,dt=.00025));raw=owner.checkpoint()
    restored=build();other=restored.restore_checkpoint(raw,expected_sha256=sha256(raw).hexdigest())
    assert canonical(first)==canonical(other) and restored.checkpoint()==raw
    end1=owner.commit(owner.trial(first,dt=.00025))
    end2=restored.commit(restored.trial(other,dt=.00025))
    assert canonical(end1)==canonical(end2) and owner.checkpoint()==restored.checkpoint()


def test_cancellation_and_operator_mutation_preserve_origin():
    owner=build();state=owner.initialize();raw=owner.checkpoint()
    token=CancellationToken();token.cancel()
    with pytest.raises(Exception):owner.trial(state,dt=.00025,cancellation_token=token)
    assert owner.current is state and owner.checkpoint()==raw
    owner.maps=(owner.maps[0],owner.maps[0])
    with pytest.raises(ValueError):owner.trial(state,dt=.00025)


def test_corrupt_or_incompatible_restore_refuses_without_publication():
    owner=build();owner.initialize();raw=owner.checkpoint();fresh=build(clamped=True)
    with pytest.raises(ValueError):fresh.restore_checkpoint(raw,expected_sha256=sha256(raw).hexdigest())
    assert fresh.current is None
    fresh=build();data=json.loads(raw);data['body']['state']['acceleration'][0]=1.
    unsigned={k:v for k,v in data['body'].items()};data['sha256']=sha256(canonical(unsigned)).hexdigest()
    corrupt=canonical(data)
    with pytest.raises(ValueError):fresh.restore_checkpoint(corrupt,expected_sha256=sha256(corrupt).hexdigest())
    assert fresh.current is None


def test_restore_block_admission_refuses_before_publication(monkeypatch):
    owner=build();owner.initialize();raw=owner.checkpoint();fresh=build()
    original=fresh._evaluate
    def singular(state,token=None):
        values=list(original(state,token));values[3]=np.zeros_like(values[3]);return tuple(values)
    monkeypatch.setattr(fresh,'_evaluate',singular)
    with pytest.raises(np.linalg.LinAlgError):fresh.restore_checkpoint(raw,expected_sha256=sha256(raw).hexdigest())
    assert fresh.current is None
    with pytest.raises(ValueError):fresh._norm(np.full(fresh.size,np.nan),fresh.free)


def test_unrepresentable_time_step_is_atomic_refusal():
    owner=build();state=owner.initialize();raw=owner.checkpoint()
    with pytest.raises(ValueError):owner.trial(state,dt=1e-200)
    assert owner.current is state and owner.checkpoint()==raw
    # A trusted rest-state capsule at time1 isolates timestamp representability.
    data=json.loads(raw);data['body']['state']['time']=1.;data['body']['state']['epoch']=1
    data['sha256']=sha256(canonical(data['body'])).hexdigest();rest=canonical(data)
    fresh=build();accepted=fresh.restore_checkpoint(rest,expected_sha256=sha256(rest).hexdigest())
    with pytest.raises(ValueError):fresh.trial(accepted,dt=1e-20)
    assert fresh.current is accepted and fresh.checkpoint()==rest


def test_stressed_spinning_shared_stage_complete_jacobian():
    owner=build();origin=owner.initialize();step=np.zeros(owner.size)
    step[6*owner.node_index[30]+1]=2e-5
    step[6*owner.node_index[30]+5]=2e-4
    step[6*owner.n+2]=1e-3;step[6*owner.n+8]=-7e-4
    step[6*owner.n+6*owner.e:]=np.linspace(-.02,.03,18*owner.e)
    local=owner._advance(origin,step);data=owner._data(local);v=np.zeros(owner.size)
    v[list(owner.physical)]=np.linspace(-.04,.06,len(owner.physical));data['velocity']=v
    local=owner._state(data)  # Off-equilibrium operator probe, NOT admitted initial state.
    unknown=np.zeros(owner.size);unknown[list(owner.p)]=np.linspace(-.2,.3,len(owner.p))
    unknown[list(owner.z)]=np.linspace(-1e-4,2e-4,len(owner.z))
    direction=np.zeros(owner.size);direction[6*owner.node_index[30]+1]=.1
    direction[6*owner.node_index[30]+5]=.01;direction[6*owner.n+2]=.2
    direction[6*owner.n+6*owner.e+4]=.005
    _,residual,jacobian,_=owner._stage(local,unknown,.00025,None)
    expected=jacobian@direction
    assert np.linalg.norm(expected)>0
    for h in (1e-2,1e-3,1e-4):
        _,plus,_,_=owner._stage(local,unknown+h*direction,.00025,None)
        _,minus,_,_=owner._stage(local,unknown-h*direction,.00025,None)
        error=np.linalg.norm(((plus-minus)/(2*h)-expected)/owner.scale)
        scale=np.linalg.norm(expected/owner.scale)
        assert error/scale<=1e-5


def test_second_element_exception_and_exit_signal_roll_back(monkeypatch):
    owner=build();state=owner.initialize();raw=owner.checkpoint();issued=dict(owner._issued)
    second=owner.elements[1].operator;original=type(second).evaluate
    def interrupt(self,*args,**kwargs):
        if self is second:raise KeyboardInterrupt('actual second-element boundary')
        return original(self,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(type(second),'evaluate',interrupt)
        with pytest.raises(KeyboardInterrupt):owner.trial(state,dt=.00025)
    assert owner.current is state and not owner._candidates
    assert owner._issued==issued and owner.checkpoint()==raw
    candidate=owner.trial(state,dt=.00025);pending=dict(owner._candidates);guard=owner._guard
    count=0
    def exit_signal(*args,**kwargs):
        nonlocal count
        count+=1;guard(*args,**kwargs)
        if count==2:raise KeyboardInterrupt('operation exit before publication')
    with monkeypatch.context() as patch:
        patch.setattr(owner,'_guard',exit_signal)
        with pytest.raises(KeyboardInterrupt):owner.commit(candidate)
    assert owner.current is state and owner._candidates==pending and owner._issued==issued
    owner.discard(candidate);assert owner.checkpoint()==raw


@pytest.mark.parametrize('routing',['node_index','n','e','size','p','z','free'])
def test_actual_routing_mutation_refuses_atomically(routing):
    owner=build();state=owner.initialize();issued=dict(owner._issued)
    if routing=='node_index':owner.node_index={**owner.node_index,30:0}
    elif routing in ('n','e','size'):setattr(owner,routing,getattr(owner,routing)+1)
    else:setattr(owner,routing,())
    with pytest.raises(ValueError):owner.trial(state,dt=.00025)
    assert owner.current is state and owner._issued==issued and not owner._candidates


def test_resigned_flattened_cell_shape_refuses_without_normalization():
    owner=build();owner.initialize();data=json.loads(owner.checkpoint())
    data['body']['state']['cell_rotations']=np.asarray(data['body']['state']['cell_rotations']).reshape(-1).tolist()
    data['sha256']=sha256(canonical(data['body'])).hexdigest();raw=canonical(data)
    fresh=build()
    with pytest.raises(ValueError,match='cell frame inventory'):
        fresh.restore_checkpoint(raw,expected_sha256=sha256(raw).hexdigest())
    assert fresh.current is None and not fresh._issued and not fresh._candidates
