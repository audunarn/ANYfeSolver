"""Private adaptive control, authenticated state and unchanged fixed-step mechanics."""
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import _ge_beam3_native_arc as arc
from anyfesolver import _ge_beam3_native_arc_adaptive as adaptive
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_native_arc import bar
from test_ge_beam3_schur_line_program import save


def make():
    model,base=bar()
    program=adaptive.AdaptiveArcProgram(replace(base,steps=(.05,)),3,.0001,.2)
    return model,program


def packet(r):
    return dict(status=r.status,steps=r.completed_steps,attempts=r.completed_attempts,parameter=r.parameter,
        u=r.displacements,reaction=r.physical_imbalance,checkpoint_sha256=sha256(r.checkpoint).hexdigest(),
        failure=r.failure,events=r.events)


@pytest.fixture(scope='module')
def accepted(tmp_path_factory):
    root=tmp_path_factory.mktemp('adaptive-bar');m,p=make();r=adaptive.solve_adaptive_arc(m,p)
    save(root/'result.json',packet(r));save(root/'checkpoint.json',r.checkpoint)
    assert r.status=='completed',r.failure
    return root,p,r


def test_fixed_equivalence_and_restart(accepted):
    root,p,r=accepted;m,_=make()
    chain,rows,attempts,inner=adaptive.decode_checkpoint(m,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    assert tuple(v['step_size'] for v in rows)==(.05,.1,.2)
    assert all(v['disposition']=='ACCEPTED' for v in attempts)
    fresh,_=make();fixed=arc.solve_arc(fresh,replace(p.prototype,steps=(.05,.1,.2)))
    assert fixed.status=='completed' and fixed.checkpoint==inner
    assert np.array_equal(fixed.displacements,r.displacements) and fixed.parameter==r.parameter
    fresh,_=make();prefix=adaptive.solve_adaptive_arc(fresh,p,stop_after=1)
    assert prefix.status=='paused'
    fresh,_=make();again=adaptive.solve_adaptive_arc(fresh,p,checkpoint=prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert again.status=='completed' and again.checkpoint==r.checkpoint
    assert adaptive.wrap_checkpoint(m,p,inner,attempts)==r.checkpoint and len(chain)==4
    save(root/'equivalence.json',dict(fixed_inner_byte_identical=True,resume_byte_identical=True,steps=[v['step_size'] for v in rows]))


@pytest.mark.parametrize('kind',('origin','step','order','iterations','next','terminal','cutbacks','program','inner','hash','duplicate','nonfinite'))
def test_mutation(accepted,kind,tmp_path):
    _,p,r=accepted;v=json.loads(r.checkpoint)
    if kind=='origin':v['attempts'][0]['origin_sha256']='0'*64
    elif kind=='step':v['attempts'][0]['step_size']*=.5
    elif kind=='order':v['attempts'][0]['attempt']=2
    elif kind=='iterations':v['attempts'][0]['iterations']+=1
    elif kind=='next':v['next_step']*=.5
    elif kind=='terminal':v['terminal']=True
    elif kind=='cutbacks':v['cutbacks']=1
    elif kind=='program':v['program']['max_attempts']=127;v['program_sha256']=sha(v['program'])
    elif kind=='inner':v['inner_checkpoint']['accepted_chain']['snapshots'][-1]['displacements'][12]+=.01
    v['checkpoint_sha256']=sha({k:x for k,x in v.items() if k!='checkpoint_sha256'})
    raw=canonical(v)
    if kind=='duplicate':raw=raw.replace(b'"terminal":false',b'"terminal":false,"terminal":false')
    if kind=='nonfinite':raw=raw.replace(b'"next_step":0.2',b'"next_step":NaN')
    m,_=make()
    with pytest.raises(ValueError):adaptive.decode_checkpoint(m,p,raw,expected_sha256='0'*64 if kind=='hash' else sha256(raw).hexdigest())
    save(tmp_path/'rejected.json',dict(kind=kind,rejected=True))


@pytest.mark.parametrize('kind',('count','minimum','maximum','nan','cuts','attempts','prototype','stop','attempt-stop','hash','nested'))
def test_preflight(kind,monkeypatch,tmp_path):
    m,p=make();kwargs={};token=None
    if kind=='count':p=replace(p,accepted_steps=True)
    elif kind=='minimum':p=replace(p,minimum_step=0.)
    elif kind=='maximum':p=replace(p,maximum_step=.3)
    elif kind=='nan':p=replace(p,minimum_step=float('nan'))
    elif kind=='cuts':p=replace(p,max_cutbacks=9)
    elif kind=='attempts':p=replace(p,max_attempts=2)
    elif kind=='prototype':p=replace(p,prototype=replace(p.prototype,steps=(.05,.1)))
    elif kind=='stop':kwargs['stop_after']=True
    elif kind=='attempt-stop':kwargs['stop_after_attempts']=True
    elif kind=='hash':kwargs.update(checkpoint=b'{}\n',expected_sha256='0'*64)
    elif kind=='nested':token=arc._ARC.set(object())
    def tripwire(*a,**kw):raise AssertionError('mechanics entered')
    monkeypatch.setattr(arc,'assemble',tripwire)
    try:
        with pytest.raises(ValueError):adaptive.solve_adaptive_arc(m,p,**kwargs)
    finally:
        if token is not None:arc._ARC.reset(token)
    save(tmp_path/'preflight.json',dict(kind=kind,rejected_before_assembly=True))


def test_rejected_real_trial_and_attempt_resume(monkeypatch,tmp_path):
    m,p=make();p=replace(p,accepted_steps=1)
    original=arc.attempt_step;calls=0
    def once(*a,**kw):
        nonlocal calls
        calls+=1;result=original(*a,**kw)
        if calls==1:raise arc.ArcConvergenceFailure('native arc iteration bound exhausted')
        return result
    monkeypatch.setattr(arc,'attempt_step',once)
    prefix=adaptive.solve_adaptive_arc(m,p,stop_after_attempts=1)
    assert prefix.status=='paused' and prefix.completed_steps==0 and prefix.completed_attempts==1
    fresh,_=make();chain,rows,attempts,inner=adaptive.decode_checkpoint(fresh,p,prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert len(chain)==1 and rows==() and attempts[0]['disposition']=='CUTBACK'
    assert np.array_equal(chain[0]['displacements'],np.zeros(18))
    monkeypatch.setattr(arc,'attempt_step',original)
    again=adaptive.solve_adaptive_arc(fresh,p,checkpoint=prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert again.status=='completed' and again.completed_attempts==2,again.failure
    fresh,_=make();_,rows,attempts,_=adaptive.decode_checkpoint(fresh,p,again.checkpoint,expected_sha256=sha256(again.checkpoint).hexdigest())
    assert rows[0]['step_size']==.025 and attempts[0]['origin_sha256']==attempts[1]['origin_sha256']
    save(tmp_path/'rejection.json',dict(real_uncommitted_trial_discarded=True,zero_origin_preserved=True,result=packet(again)))


@pytest.mark.parametrize('kind',('minimum','cutbacks','attempt-budget'))
def test_convergence_exhaustion(kind,monkeypatch,tmp_path):
    m,p=make();p=replace(p,accepted_steps=1)
    if kind=='minimum':p=replace(p,minimum_step=.05)
    elif kind=='cutbacks':p=replace(p,max_cutbacks=0)
    else:p=replace(p,max_attempts=1)
    def fail(*a,**kw):raise arc.ArcConvergenceFailure('native arc line search exhausted')
    monkeypatch.setattr(arc,'attempt_step',fail)
    r=adaptive.solve_adaptive_arc(m,p);assert r.status=='failed' and r.completed_steps==0 and r.completed_attempts==1
    fresh,_=make();chain,rows,attempts,_=adaptive.decode_checkpoint(fresh,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    assert len(chain)==1 and rows==() and attempts[-1]['disposition']=='EXHAUSTED'
    with pytest.raises(ValueError,match='exhausted'):adaptive.solve_adaptive_arc(fresh,p,checkpoint=r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    save(tmp_path/'exhausted.json',dict(kind=kind,result=packet(r),resume_rejected=True))


@pytest.mark.parametrize('kind',('cancel','authority','factorization','observer-convergence'))
def test_fatal_preserves_prefix(kind,monkeypatch,tmp_path):
    m,p=make();prefix=adaptive.solve_adaptive_arc(m,p,stop_after=1);assert prefix.status=='paused'
    m,p=make();token=CancellationToken();original=arc.factorize;committed=False
    def observer(row):
        nonlocal committed
        if row['stage']=='committed' and row['step']==1:committed=True
        if row['stage']=='before_commit' and row['step']==2:
            if kind=='cancel':token.cancel('prescribed adaptive cancellation')
            if kind=='authority':object.__setattr__(p,'maximum_step',.15)
            if kind=='observer-convergence':raise arc.ArcConvergenceFailure('native arc line search exhausted')
    def factor(*a,**kw):
        if committed and kind=='factorization':raise RuntimeError('prescribed factorization failure')
        return original(*a,**kw)
    monkeypatch.setattr(arc,'factorize',factor)
    r=adaptive.solve_adaptive_arc(m,p,progress=observer,cancellation_token=token)
    assert r.status==('cancelled' if kind=='cancel' else 'failed') and r.completed_steps==1 and r.completed_attempts==1
    assert r.checkpoint==prefix.checkpoint and arc._ARC.get() is None
    save(tmp_path/'preserved.json',dict(kind=kind,result=packet(r),prefix_exact=True))


def test_physical_convergence_cutback(tmp_path):
    from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
    from anyfesolver._ge_beam3_native_line_loading import LinePattern
    from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
    m,p=make()
    base=replace(p.prototype,steps=(.2,),max_iterations=1,
        distributed=DistributedPattern(LinePattern(((1,.4,-.1,.15),)),()),
        nodal_moments=SpatialNodalMoments(((3,.1,.15,-.12),)))
    p=adaptive.AdaptiveArcProgram(base,2,.0001,.2,max_cutbacks=8,max_attempts=16)
    r=adaptive.solve_adaptive_arc(m,p,progress=lambda row:print(row,flush=True))
    save(tmp_path/'result.json',packet(r));save(tmp_path/'checkpoint.json',r.checkpoint)
    assert r.status=='completed',r.failure
    fresh,_=make();chain,rows,attempts,_=adaptive.decode_checkpoint(fresh,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    assert any(a['disposition']=='CUTBACK' for a in attempts) and len(chain)==3
    assert all(v['iterations']<=1 for v in rows)
    save(tmp_path/'cutback.json',dict(physical_convergence_cutback=True,steps=[v['step_size'] for v in rows],attempts=attempts))


def test_curved_plastic_source(tmp_path):
    from test_ge_beam3_native_generalized_restart import make as curved,pattern
    from anyfesolver._ge_beam3_native_translation import TranslationProgram,solve_translation
    from anyfesolver._ge_beam3_native_arc_source import TranslationArcSource
    from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
    from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
    model=curved('curved-plastic');translation=TranslationProgram((.03,),3,'ux',pattern(model),SpatialNodalMoments(((3,.015,-.01,.008),)))
    source=solve_translation(model,translation);save(tmp_path/'source.json',source.checkpoint);assert source.status=='completed',source.failure
    prototype=arc.ArcProgram((.04,),1.,translation.distributed,translation.nodal_moments,
        source=TranslationArcSource(translation,source.checkpoint,sha256(source.checkpoint).hexdigest(),1.))
    p=adaptive.AdaptiveArcProgram(prototype,2,.01,.08)
    model=curved('curved-plastic');r=adaptive.solve_adaptive_arc(model,p)
    save(tmp_path/'result.json',packet(r));save(tmp_path/'checkpoint.json',r.checkpoint);assert r.status=='completed',r.failure
    check=curved('curved-plastic');chain,rows,attempts,inner=adaptive.decode_checkpoint(check,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    assert len(chain)==4 and len(rows)==2
    positive=sum(sum(h.accumulated)>0 for state in chain[-1]['states'].values() for h in state['response'].history.stations)
    assert positive==8
    fields={i:recover_native_fields(e,check.mesh,chain[-1]['states'][i],expected_committed_total_u=chain[-1]['displacements'][list(e.get_dof_mapping(check.mesh))]) for i,e in check.mesh.elements.items()}
    assert adaptive.wrap_checkpoint(check,p,inner,attempts)==r.checkpoint
    save(tmp_path/'recovery.json',dict(fields=fields,positive_plastic_stations=positive,steps=[v['step_size'] for v in rows]))



def test_adaptive_rigid_covariance(tmp_path):
    from anyfesolver.fe_core import FEModel
    from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
    from anyfesolver._ge_beam3_native_line_loading import LinePattern
    from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
    from anyfesolver._ge_beam3_p5.algebra import rotation
    m,p=make();force=np.array([.4,-.1,.15]);moment=np.array([.1,.15,-.12])
    p=replace(p,prototype=replace(p.prototype,distributed=DistributedPattern(LinePattern(((1,*map(float,force)),)),()),
        nodal_moments=SpatialNodalMoments(((3,*map(float,moment)),))))
    r=adaptive.solve_adaptive_arc(m,p);assert r.status=='completed',r.failure
    s=rotation([.4,-.2,.7]);shift=np.array([2.,-3.,1.]);moved=FEModel('adaptive-arc-rotated')
    for i,node in m.mesh.nodes.items():moved.add_node(i,*(s@node.coords()+shift))
    for i,e in m.mesh.elements.items():
        ref=Reference(np.array([s@x+shift for x in e.operator.reference.coordinates]),np.array([s@q for q in e.operator.reference.nodal_triads]))
        element=NativeGeneralizedStaticElement(i,e.node_ids,ref,e.section,order=4);moved.add_element(i,element);moved.materials[element.material_name]=element.section
    for bc in m.boundary_conditions:moved.add_boundary_condition(bc)
    other=replace(p,prototype=replace(p.prototype,distributed=DistributedPattern(LinePattern(((1,*map(float,s@force)),)),()),
        nodal_moments=SpatialNodalMoments(((3,*map(float,s@moment)),))))
    changed=adaptive.solve_adaptive_arc(moved,other);assert changed.status=='completed',changed.failure
    expected=np.array([s@v for v in r.displacements.reshape(-1,3)]).ravel()
    reaction=np.array([s@v for v in r.physical_imbalance.reshape(-1,3)]).ravel()
    errors=dict(displacement=float(np.max(np.abs(changed.displacements-expected))),reaction=float(np.max(np.abs(changed.physical_imbalance-reaction))),parameter=abs(changed.parameter-r.parameter))
    assert max(errors.values())<=1e-11
    a=json.loads(r.checkpoint)['attempts'];b=json.loads(changed.checkpoint)['attempts']
    assert [(v['step_size'],v['disposition']) for v in a]==[(v['step_size'],v['disposition']) for v in b]
    save(tmp_path/'covariance.json',dict(errors=errors,base=packet(r),moved=packet(changed),same_step_schedule=True))


def test_curved_plastic_rejection(monkeypatch,tmp_path):
    from test_ge_beam3_native_generalized_restart import make as curved,pattern
    from anyfesolver._ge_beam3_native_translation import TranslationProgram,solve_translation
    from anyfesolver._ge_beam3_native_arc_source import TranslationArcSource
    from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
    model=curved('curved-plastic');translation=TranslationProgram((.03,),3,'ux',pattern(model),SpatialNodalMoments(((3,.015,-.01,.008),)))
    source=solve_translation(model,translation);save(tmp_path/'source.json',source.checkpoint);assert source.status=='completed',source.failure
    prototype=arc.ArcProgram((.04,),1.,translation.distributed,translation.nodal_moments,
        source=TranslationArcSource(translation,source.checkpoint,sha256(source.checkpoint).hexdigest(),1.))
    p=adaptive.AdaptiveArcProgram(prototype,1,.01,.08)
    original=arc.attempt_step;calls=0;rejected_accumulated=[]
    def once(*args,**kwargs):
        nonlocal calls
        calls+=1;outcome=original(*args,**kwargs)
        if calls==1:
            trial=outcome[2]
            rejected_accumulated.extend(float(v) for state in trial.values() for h in state['response'].history.stations for v in h.accumulated)
            raise arc.ArcConvergenceFailure('native arc iteration bound exhausted')
        return outcome
    monkeypatch.setattr(arc,'attempt_step',once)
    model=curved('curved-plastic');r=adaptive.solve_adaptive_arc(model,p)
    save(tmp_path/'result.json',packet(r));save(tmp_path/'checkpoint.json',r.checkpoint);assert r.status=='completed',r.failure
    monkeypatch.setattr(arc,'attempt_step',original)
    model=curved('curved-plastic');chain,rows,attempts,inner=adaptive.decode_checkpoint(model,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    fixed=arc.solve_arc(curved('curved-plastic'),replace(prototype,steps=(.02,)))
    assert fixed.status=='completed' and fixed.checkpoint==inner,fixed.failure
    accepted_accumulated=[float(v) for state in chain[-1]['states'].values() for h in state['response'].history.stations for v in h.accumulated]
    assert sum(rejected_accumulated)>sum(accepted_accumulated)>0
    assert len(attempts)==2 and attempts[0]['disposition']=='CUTBACK' and attempts[0]['origin_sha256']==attempts[1]['origin_sha256']
    save(tmp_path/'rollback.json',dict(fixed_inner_byte_identical=True,rejected_accumulated=rejected_accumulated,
        accepted_accumulated=accepted_accumulated,plastic_trial_discarded=True,source_prefix_preserved=True))



@pytest.mark.parametrize('step,iterations,cuts,expected',[
    (.1,0,0,.2),(.1,4,0,.1),(.1,7,0,.05),(.1,2,1,.1),(.0001,8,0,.0001),(.2,0,0,.2)])
def test_policy_boundaries(step,iterations,cuts,expected,tmp_path):
    _,p=make()
    assert adaptive.next_after_accept(p,step,iterations,cuts)==expected
    save(tmp_path/'policy.json',dict(step=step,iterations=iterations,cutbacks=cuts,next_step=expected))


def test_total_attempt_budget_after_acceptance(monkeypatch,tmp_path):
    m,p=make();p=replace(p,accepted_steps=2,max_attempts=2)
    original=arc.attempt_step;calls=0
    def once(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==1:raise arc.ArcConvergenceFailure('native arc iteration bound exhausted')
        return original(*a,**kw)
    monkeypatch.setattr(arc,'attempt_step',once)
    r=adaptive.solve_adaptive_arc(m,p)
    assert r.status=='failed' and r.completed_steps==1 and r.completed_attempts==2
    fresh,_=make();chain,rows,attempts,_=adaptive.decode_checkpoint(fresh,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    assert len(chain)==2 and len(rows)==1 and [v['disposition'] for v in attempts]==['CUTBACK','ACCEPTED']
    with pytest.raises(ValueError,match='exhausted'):
        adaptive.solve_adaptive_arc(fresh,p,checkpoint=r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    save(tmp_path/'budget.json',dict(result=packet(r),accepted_history_valid=True,terminal_resume_rejected=True))
