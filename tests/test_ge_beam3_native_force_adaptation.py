"""Actual native adaptive stepping and rollback, not production qualification."""
from hashlib import sha256
import numpy as np
import pytest
from anyfesolver._ge_beam3_native_force_adaptation import AdaptiveForcePolicy,describe,require_capacity,settings
from anyfesolver._ge_beam3_native_generalized_combined_couples import solve_combined_static,_RUN
from anyfesolver._ge_beam3_native_generalized_program import _PROGRAM
from anyfesolver._ge_beam3_native_generalized_loading import _ACTIVE,DistributedPattern
from anyfesolver._ge_beam3_native_generalized_combined_restart import LoadPoint,encode_checkpoint,decode_checkpoint
from anyfesolver._ge_beam3_native_generalized_restart import LoadPoint as DistributedPoint
from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from test_ge_beam3_native_generalized_restart import make,pattern,packet,CASES,EMPTY
from test_ge_beam3_native_generalized_combined_restart import snapshots
from test_ge_beam3_native_generalized_combined_couples import convert,simple
from test_ge_beam3_schur_line_program import save


@pytest.mark.parametrize('case',CASES)
def test_actual_cutback_reproduces_fixed_history(case,monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    fixed_model=make(case);line=pattern(fixed_model);moments=SpatialNodalMoments(((3,.015,-.01,.008),))
    fixed,_=solve_combined_static(fixed_model,moments,distributed_pattern=line)
    assert fixed.status=='completed',fixed.info
    model=make(case);initial={i:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for i,e in model.mesh.elements.items()}
    before=canonical(initial);original=solver.factorize;attempts=[];captured=[]
    def rejected_once(matrix,kind,*args,**kwargs):
        if str(kwargs.get('signature','')).startswith('nonlinear.static_newton'):
            run=_RUN.get();store=run.store
            if not attempts:
                assert canonical(store.materialize())==before
                captured.append(store);attempts.append('rejected_initial_factorization')
                raise RuntimeError('prescribed one-shot factorization rejection')
            if len(attempts)==1:
                assert canonical(store.materialize())==before and store.generation==0
                attempts.append('cutback_reentered_from_exact_accepted_origin')
        return original(matrix,kind,*args,**kwargs)
    monkeypatch.setattr(solver,'factorize',rejected_once)
    policy=AdaptiveForcePolicy(cutback_levels=1,growth=False)
    result,evidence=solve_combined_static(model,moments,distributed_pattern=line,steps=1,step_policy=policy)
    save(tmp_path/'fixed.json',packet(fixed));save(tmp_path/'adaptive.json',dict(result=packet(result),evidence=evidence))
    assert result.status=='completed',result.info
    assert attempts==['rejected_initial_factorization','cutback_reentered_from_exact_accepted_origin']
    assert [float(s.load_factor) for s in result.snapshots]==[.5,1.]
    assert canonical(packet(result))==canonical(packet(fixed))
    adaptation=result.info['convergence_adaptation']
    assert sum(a['action']=='cutback_after_nonconvergence' for a in adaptation)==1
    assert captured[0].generation==2 and not captured[0].has_active_trial
    chain=(dict(load_point=LoadPoint(DistributedPoint(0.,EMPTY,line),None,moments),
                displacements=np.zeros(model.mesh.dof_manager.total_dofs),states=initial),)+snapshots(result,line,moments)
    raw=encode_checkpoint(model,chain);save(tmp_path/'checkpoint.json',raw)
    fresh=make(case);decoded=decode_checkpoint(fresh,raw,expected_sha256=sha256(raw).hexdigest())
    assert encode_checkpoint(fresh,decoded)==raw
    fields={str(i):recover_native_fields(e,fresh.mesh,decoded[-1]['states'][i],
        expected_committed_total_u=decoded[-1]['displacements'][list(e.get_dof_mapping(fresh.mesh))]) for i,e in fresh.mesh.elements.items()}
    save(tmp_path/'recovery.json',fields)
    assert _PROGRAM.get() is None and _RUN.get() is None and _ACTIVE.get() is None
    save(tmp_path/'cutback.json',dict(case=case,attempts=attempts,adaptation=adaptation,
        fixed_state_byte_identical=True,accepted_factors=[.5,1.],checkpoint_roundtrip=True,production_qualified=False))


@pytest.mark.parametrize('change',[
    {'cutback_levels':True},{'cutback_levels':-1},{'cutback_levels':7},{'growth':1},
    {'fast_iterations':False},{'fast_iterations':0},{'slow_iterations':25},{'slow_iterations':4}])
def test_policy_fields_reject(change,tmp_path):
    with pytest.raises(ValueError):AdaptiveForcePolicy(**change).descriptor()
    save(tmp_path/'rejected.json',dict(change=change,rejected=True))


@pytest.mark.parametrize('kind',['wrong-type','non-dyadic-steps','boolean-steps','iteration-type','line-search-type','capacity'])
def test_invalid_controls_before_solver(kind,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_generalized_combined_couples as module
    kwargs=dict(step_policy=AdaptiveForcePolicy(),steps=2)
    if kind=='wrong-type':kwargs['step_policy']={}
    elif kind=='non-dyadic-steps':kwargs['steps']=3
    elif kind=='boolean-steps':kwargs['steps']=True
    elif kind=='iteration-type':kwargs['max_iterations']=True
    elif kind=='line-search-type':kwargs['line_search']=1
    else:kwargs.update(step_policy=AdaptiveForcePolicy(cutback_levels=6),steps=2)
    def forbidden(*a,**k):raise AssertionError('invalid controls entered solver')
    monkeypatch.setattr(module,'solve_distributed_model',forbidden)
    with pytest.raises(ValueError):solve_combined_static(object(),None,**kwargs)
    assert _PROGRAM.get() is None and _RUN.get() is None
    save(tmp_path/'rejected.json',dict(kind=kind,rejected_before_solver=True))


def test_capacity_and_settings(tmp_path):
    policy=AdaptiveForcePolicy(cutback_levels=2,growth=False)
    require_capacity(policy,2,56)
    for count in (57,-1,True):
        with pytest.raises(ValueError):require_capacity(policy,2,count)
    made=settings(policy,2,24,True)
    assert made.min_step_fraction==.25 and made.growth_factor==1. and made.max_step_factor==1.
    assert made.max_line_search_cuts==8 and made.line_search=='always'
    assert describe(None,2,24,True) is None
    save(tmp_path/'settings.json',dict(policy=describe(policy,2,24,True),settings=made.to_dict(),capacity_rejected=True))


def test_actual_growth_nonconvergence_and_resume_capacity(monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    model=convert(simple()[0]);line=DistributedPattern(LinePattern(()),((1,.12,0.,0.),));moments=SpatialNodalMoments(((3,.3,0.,0.),))
    initial={i:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for i,e in model.mesh.elements.items()}
    policy=AdaptiveForcePolicy(cutback_levels=1,growth=True)
    result,evidence=solve_combined_static(model,moments,distributed_pattern=line,steps=4,step_policy=policy)
    assert result.status=='completed',result.info
    factors=[float(s.load_factor) for s in result.snapshots];assert factors==[.25,.75,1.]
    for snapshot,step in zip(result.snapshots,result.steps):
        for i,s in enumerate((0.,.5,1.)):
            angle=snapshot.load_factor*(.3*s+.12*(s-s*s/2))/2
            np.testing.assert_allclose(snapshot.element_states[1]['committed_nodal_rotation_matrices'][i],rotation([angle,0.,0.]),atol=1e-11,rtol=0.)
        np.testing.assert_allclose(step.support_reactions['root'][3:],[-snapshot.load_factor*.42,0.,0.],atol=1e-11,rtol=0.)
    save(tmp_path/'growth.json',dict(result=packet(result),evidence=evidence,factors=factors))
    chain=(dict(load_point=LoadPoint(DistributedPoint(0.,EMPTY,line),None,moments),displacements=np.zeros(18),states=initial),)+snapshots(result,line,moments)
    raw=encode_checkpoint(model,chain);save(tmp_path/'growth-checkpoint.json',raw)
    failed_model=convert(simple()[0]);virgin={i:e.init_model_bound_nonlinear_state(failed_model.mesh,e.section,1) for i,e in failed_model.mesh.elements.items()}
    failed,events=solve_combined_static(failed_model,moments,distributed_pattern=line,steps=1,max_iterations=1,
        step_policy=AdaptiveForcePolicy(cutback_levels=2,growth=False))
    assert failed.status=='diverged' and not failed.snapshots
    assert canonical(failed.element_states)==canonical(virgin)
    adaptations=failed.info['convergence_adaptation']
    assert [a['load_factor'] for a in adaptations]==[1.,.5,.25]
    assert all(a['action']=='cutback_after_nonconvergence' for a in adaptations)
    save(tmp_path/'exhausted.json',dict(result=packet(failed),events=events,virgin_preserved=True))
    def forbidden(*a,**k):raise AssertionError('exhausted restart capacity entered solver')
    monkeypatch.setattr(solver,'solve_static_nonlinear',forbidden)
    with pytest.raises(ValueError,match='capacity'):
        solve_combined_static(convert(simple()[0]),moments,distributed_pattern=line,steps=16,
            step_policy=AdaptiveForcePolicy(cutback_levels=2),initial_checkpoint=raw,expected_sha256=sha256(raw).hexdigest())
    save(tmp_path/'resume-capacity.json',dict(rejected_before_solver=True,accepted_count=3))
    assert _PROGRAM.get() is None and _RUN.get() is None and _ACTIVE.get() is None


def test_live_policy_mutation_discards_before_commit(monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    model=convert(simple()[0]);policy=AdaptiveForcePolicy(cutback_levels=1,growth=False);original=solver.factorize;captured=[]
    def changed(matrix,kind,*a,**k):
        handle=original(matrix,kind,*a,**k)
        if str(k.get('signature','')).startswith('nonlinear.static_newton'):
            store=_RUN.get().store;captured.append((store,canonical(store.materialize())))
            object.__setattr__(policy,'growth',True)
        return handle
    monkeypatch.setattr(solver,'factorize',changed)
    with pytest.raises(ValueError,match='authority'):
        solve_combined_static(model,SpatialNodalMoments(((3,.3,0.,0.),)),steps=1,step_policy=policy)
    assert captured and all(store.generation==0 and canonical(store.materialize())==before and not store.has_active_trial for store,before in captured)
    assert _PROGRAM.get() is None and _RUN.get() is None and _ACTIVE.get() is None
    save(tmp_path/'mutation.json',dict(rejected_before_commit=True,virgin_preserved=True,production_qualified=False))
