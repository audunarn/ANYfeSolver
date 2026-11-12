"""Small shared-node dynamic/state checks, not assembled qualification."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from docs.reference_cases import ge_beam3_curved_p5_dynamic_assembly_probe as assembly
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation
from docs.reference_cases.ge_beam3_curved_p5_continuum_probe import parabolic_references
from docs.reference_cases.ge_beam3_curved_p5_history_path_probe import digest
from docs.reference_cases.ge_beam3_curved_p5_mass_probe import reference_kinetic_factors
from docs.reference_cases.ge_beam3_curved_p5_modal_chain_probe import clamped_full_inertia_chain
from docs.reference_cases.ge_beam3_curved_p5_midpoint_dynamics_probe import MidpointDynamicProbe
from test_ge_beam3_curved_p5_algebra_probe import section
from test_ge_beam3_curved_p5_mass_probe import section_mass


def close(a,b,tolerance=1e-11):
    assert np.linalg.norm(np.asarray(a)-np.asarray(b))<=tolerance*max(1.,np.linalg.norm(b))


def model(count=2,height=.4,*,order=8,fixed_nodes=(0,),rolled=False):
    refs=list(parabolic_references(height,count))
    if rolled:
        refs[-1]=CurvedBeam3ReferenceGeometry(refs[-1].coordinates,refs[-1].nodal_triads@rotation([.3,0.,0.]))
    maps=[(2*i,2*i+1,2*i+2) for i in range(count)]
    return assembly.DynamicAssemblyProbe(refs,maps,[section()]*count,[section_mass()]*count,
                                         fixed_nodes=fixed_nodes,order=order)


def load(made):
    result=np.zeros((made._nodes,3));result[-1]=[.01,-.02,.015]
    return result


def test_one_macro_matches_preserved_midpoint_stage_and_endpoint():
    made=model(1);element=MidpointDynamicProbe(parabolic_references(.4,1)[0],section(),section_mass(),order=8)
    a=made.trial(.02,load(made));b=element.trial(.02,load(made))
    close(a.increment,b.increment);close(a.response.stage.residual,b.response.residual)
    close(a.response.stage.tangent,b.response.tangent)
    close(a.response.state.positions,b.response.state.positions)
    close(a.response.state.rotations@element._reference.nodal_triads,b.response.state.vertex_frames)
    close(a.response.state.cell_rotations[0],b.response.state.cell_rotations)
    close(a.response.elastic_energy,b.response.elastic_energy)
    close(a.response.kinetic_energy,b.response.kinetic_energy)
    made.commit(a);assert digest(made.replay())==digest(a.response)


@pytest.mark.parametrize('count',[2,4])
def test_reference_effective_matrix_matches_full_inertia_chain_pencil(count):
    made=model(count,order=24);h=.02
    response=made._stage(made.committed,np.zeros(made._size),h,np.zeros((made._nodes,3)))
    pencil=clamped_full_inertia_chain(parabolic_references(.4,count),section(),section_mass())
    chart=np.eye(made._size)*.5
    for node in range(made._nodes): chart[6*node+3:6*node+6,6*node+3:6*node+6]=np.eye(3)
    close(response.tangent,pencil.full_stiffness@chart+2*pencil.full_mass/h**2)
    traces=np.array([6*i+j for i in range(made._nodes) for j in (3,4,5)])
    assert np.array_equal(pencil.full_mass[:,traces],np.zeros((made._size,len(traces))))


def test_nonzero_assembled_analytic_jacobian_and_virtual_work():
    made=model();z=.002*np.sin(np.arange(made._size)+.2);z[:6]=0.
    direction=np.cos(np.arange(made._size)+.3);direction[:6]=0.;epsilon=1e-7
    origin=made.committed;response=made._stage(origin,z,.04,load(made))
    plus=made._stage(origin,z+epsilon*direction,.04,load(made))
    minus=made._stage(origin,z-epsilon*direction,.04,load(made))
    close((plus.residual-minus.residual)/(2*epsilon),response.tangent@direction,1e-7)
    local_work=sum(float(direction[slots]@(f+i)) for slots,f,i in
                   zip(made._slots,response.element_forces,response.element_inertia))
    external=sum(float(direction[6*i:6*i+3]@force) for i,force in enumerate(load(made)))
    close(float(direction@response.residual),local_work-external)


@pytest.mark.parametrize('rolled',[False,True])
def test_shared_rotations_balance_assembled_not_elementwise_moments(rolled):
    made=model(rolled=rolled);before=digest(made.committed);trial=made.trial(.02,load(made));r=trial.response
    assert digest(made.committed)==before
    assert trial.residual_norm<=1e-11 and r.trace_residual_norm<=1e-11
    assert r.trace_iterations<=8 and r.trace_evaluations<=64
    # Node 2 is element 0's end and element 1's beginning. Neither incident
    # endpoint moment should have been independently forced to zero.
    for parts in (r.stage.element_forces,r.endpoint_element_forces):
        left=parts[0][15:18];right=parts[1][3:6]
        assert np.linalg.norm(left)>1e-7 and np.linalg.norm(right)>1e-7
        close(left+right,np.zeros(3))
    assert np.linalg.norm(r.stage.elastic_force[made._nodal:])>1e-6
    close(r.stage.elastic_force[made._nodal:]+r.stage.inertia[made._nodal:],np.zeros(12))
    normalized=r.endpoint_force[made._traces]/made._length
    close(normalized,np.zeros(len(made._traces)))
    assert abs(np.linalg.norm(normalized)-r.trace_residual_norm)<1e-25
    assert np.array_equal(r.stage.inertia[made._traces],np.zeros(len(made._traces)))
    made.commit(trial);assert digest(made.replay())==digest(r)
    assert all(e.committed.epoch==0 and e._pending is None for e in made._elements)


def test_curved_loading_reversal_and_fresh_replay_are_whole_graph_transactions():
    made=model();hashes=[]
    for epoch,scale in enumerate((1.,-.5,0.),1):
        before=made.committed;trial=made.trial(.02,scale*load(made))
        assert digest(trial.origin)==digest(before) and digest(made.committed)==digest(before)
        made.commit(trial);assert made.committed.epoch==epoch
        hashes.append(digest(made.replay()))
        assert hashes[-1]==digest(trial.response)
    saved=digest(made.committed)
    object.__setattr__(trial.response.state,'epoch',999)
    copy=made.committed;copy.positions[-1,0]+=1.
    assert digest(made.committed)==saved and digest(made.replay().state)==saved


def test_constant_observer_covariance_for_shared_deformation_rotations():
    first=model();g=rotation([1.4,-1.8,.9]);shift=np.array([2.,-1.,3.])
    refs=[e._reference.rigidly_transformed(g,shift) for e in first._elements]
    second=assembly.DynamicAssemblyProbe(refs,[(0,1,2),(2,3,4)],[section()]*2,[section_mass()]*2,order=8)
    a=first.trial(.02,load(first)).response.state;b=second.trial(.02,load(first)@g.T).response.state
    close(b.positions,a.positions@g.T+shift);close(b.rotations,g@a.rotations@g.T)
    close(b.cell_rotations,g@a.cell_rotations@g.T)
    close(b.nodal_velocity,a.nodal_velocity@g.T);close(b.cell_angular_velocity,a.cell_angular_velocity@g.T)


def test_free_chain_uniform_translation_has_no_artificial_rotational_mass():
    refs=parabolic_references(0.,2);inertia=np.diag([2.,2.,2.,.2,.1,.15])
    made=assembly.DynamicAssemblyProbe(refs,[(0,1,2),(2,3,4)],[section()]*2,[inertia]*2,fixed_nodes=(),order=8)
    mass=np.zeros((made._size,made._size))
    for ref,slots in zip(refs,made._slots):
        factors=reference_kinetic_factors(ref,section(),inertia,order=8)
        mass[np.ix_(slots,slots)]+=factors.full.T@factors.full
    accel=np.array([.02,-.03,.01]);a=np.zeros(made._size)
    for node in range(made._nodes): a[6*node:6*node+3]=accel
    full_load=mass@a;forces=full_load[:made._nodal].reshape(made._nodes,6)[:,:3]
    close(full_load[made._nodal:],np.zeros(12))
    h=.02;trial=made.trial(h,forces);made.commit(trial)
    close(made.committed.positions-made._coordinates,np.tile(.5*h*h*accel,(made._nodes,1)))
    close(made.committed.nodal_velocity,np.tile(h*accel,(made._nodes,1)))
    close(made.committed.cell_rotations,np.tile(np.eye(3),(2,2,1,1)))


def test_late_last_element_commit_failure_cannot_partially_publish(monkeypatch):
    made=model();trial=made.trial(.02,load(made));before=digest(made.committed)
    def fail(*args,**kwargs): raise ValueError('injected last-element failure')
    monkeypatch.setattr(made._elements[-1]._elastic,'_jet',fail)
    with pytest.raises(ValueError,match='last-element'): made.commit(trial)
    assert digest(made.committed)==before
    assert all(e.committed.epoch==0 and e._pending is None for e in made._elements)


@pytest.mark.parametrize('field',['force','endpoint_part','state','metadata','model','slots','scale'])
def test_rehashed_receipt_and_changed_model_mutations_are_rejected(field):
    made=model();trial=made.trial(.02,load(made));before=digest(made.committed)
    if field=='force': object.__setattr__(trial,'forces',load(made)*2)
    if field=='endpoint_part':
        parts=list(trial.response.endpoint_element_forces);parts[-1]=parts[-1]+1.
        object.__setattr__(trial.response,'endpoint_element_forces',tuple(parts))
    if field=='state': object.__setattr__(trial.response,'state',replace(trial.response.state,time=.4))
    if field=='metadata': object.__setattr__(trial,'evaluations',129)
    if field=='model': made._elements[-1]._order=9
    if field=='slots': made._slots=tuple(row[::-1].copy() for row in made._slots)
    if field=='scale': made._length*=2
    made._pending_digest=digest(trial)
    with pytest.raises(assembly.DynamicTransactionError): made.commit(trial)
    assert digest(made.committed)==before


def test_discard_failure_bounds_and_foreign_receipts_preserve_state():
    made=model();before=digest(made.committed)
    for kwargs in ({'max_iterations':0},{'max_evaluations':1}):
        with pytest.raises(assembly.DynamicStepError): made.trial(.02,load(made),**kwargs)
        assert digest(made.committed)==before and made._pending is None
    trial=made.trial(.02,load(made))
    with pytest.raises(assembly.DynamicTransactionError): made.commit(deepcopy(trial))
    made.discard(trial);again=made.trial(.02,load(made))
    assert digest(trial)==digest(again) and digest(made.committed)==before


def test_four_macro_loaded_step_retains_all_eight_cell_rotations():
    made=model(4);trial=made.trial(.02,load(made));made.commit(trial)
    assert made.committed.cell_rotations.shape==(4,2,3,3)
    assert made.committed.cell_angular_velocity.shape==(4,2,3)
    assert trial.residual_norm<=1e-11 and trial.response.trace_residual_norm<=1e-11
    assert digest(made.replay())==digest(trial.response)


def test_element_enumeration_does_not_change_global_physics():
    first=model();refs=[element._reference for element in first._elements]
    second=assembly.DynamicAssemblyProbe(refs[::-1],[(2,3,4),(0,1,2)],[section()]*2,[section_mass()]*2,order=8)
    a=first.trial(.02,load(first)).response;b=second.trial(.02,load(first)).response
    close(b.state.positions,a.state.positions);close(b.state.rotations,a.state.rotations)
    close(b.state.nodal_velocity,a.state.nodal_velocity)
    close(b.state.cell_rotations[::-1],a.state.cell_rotations)
    close(b.state.cell_angular_velocity[::-1],a.state.cell_angular_velocity)
    close(b.elastic_energy,a.elastic_energy);close(b.kinetic_energy,a.kinetic_energy)


def test_invalid_steps_and_changed_fixed_state_are_rejected():
    made=model();before=digest(made.committed)
    for h in (True,0.,-.1,.11,float('nan')):
        with pytest.raises(assembly.DynamicStepError): made.trial(h,load(made))
    for kwargs in ({'max_iterations':True},{'max_iterations':17},{'max_evaluations':129}):
        with pytest.raises(ValueError): made.trial(.02,load(made),**kwargs)
    state=made.committed;positions=state.positions.copy();positions[0,0]+=.01
    with pytest.raises(assembly.DynamicTransactionError):
        made._stage(replace(state,positions=positions),np.zeros(made._size),.02,load(made))
    assert digest(made.committed)==before


def test_graph_validation_and_watchdog(monkeypatch):
    refs=parabolic_references(.4,2)
    for maps in ([(0,1,2),(2,3,True)],[(0,1,2),(0,1,2)],[(0,1,2),(3,4,5)],[(0,1,2),(2,3,6)]):
        with pytest.raises(ValueError): assembly.DynamicAssemblyProbe(refs,maps,[section()]*2,[section_mass()]*2)
    shifted=refs[-1].rigidly_transformed(np.eye(3),[.1,0.,0.])
    with pytest.raises(ValueError,match='shared reference'):
        assembly.DynamicAssemblyProbe([refs[0],shifted],[(0,1,2),(2,3,4)],[section()]*2,[section_mass()]*2)
    made=model();before=digest(made.committed);ticks=iter((0.,601.))
    monkeypatch.setattr(assembly.time,'monotonic',lambda:next(ticks))
    with pytest.raises(assembly.DynamicStepError,match='wall limit'): made.trial(.02,load(made))
    assert digest(made.committed)==before and made._pending is None
