"""Combined distributed/nodal work through actual native assembly and Newton."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_native_combined_couples import solve_combined_static,external_at,_Run,_RUN
from anyfesolver._ge_beam3_native_distributed_loading import DistributedPattern,assemble_distributed_trial,_ACTIVE
from anyfesolver._ge_beam3_native_distributed_program import solve_distributed_model,_PROGRAM
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from test_ge_beam3_native_distributed import problem,convert,pattern,packet
from test_ge_beam3_native_spatial_couples import left_jacobian,make as simple
from test_ge_beam3_native_fibre_static_element import store_for,sample
from test_ge_beam3_schur_line_program import save


@pytest.mark.parametrize('plastic',[False,True])
def test_combined_directional_work(plastic,tmp_path):
    m,e=problem(True,plastic);store=store_for(m,e);before=canonical(store.materialize());u=sample();load=pattern(m)
    moment=SpatialNodalMoments(((3,.2,-.15,.12),));run=_Run(m,moment);token=_RUN.set(run);v=np.sin(np.arange(18)+.3);v[:6]=0.
    try:
        force,k,_,line=assemble_distributed_trial(m,u,store,load)
        nodal,kt=external_at(m,store,u,.7,tangent=True);target=.7*np.array([.2,-.15,.12]);a=left_jacobian(u[15:18])
        np.testing.assert_allclose(nodal[15:18],a.T@target,atol=1e-11,rtol=0.)
        work=float(abs(nodal@v-target@(a@v[15:18])));assert work<=1e-11
        analytical=(k-kt)@v;errors=[]
        for h in (2e-5,1e-5):
            fp,_,_,lp=assemble_distributed_trial(m,u+h*v,store,load,tangent=False)
            ep,_=external_at(m,store,u+h*v,.7,tangent=False)
            fm,_,_,lm=assemble_distributed_trial(m,u-h*v,store,load,tangent=False)
            em,_=external_at(m,store,u-h*v,.7,tangent=False)
            errors.append(float(np.linalg.norm(((fp-lp-ep)-(fm-lm-em))/(2*h)-analytical)/max(1.,np.linalg.norm(analytical))))
        assert max(errors)<=1e-7 and canonical(store.materialize())==before
        save(tmp_path/'directional.json',dict(plastic=plastic,work_error=work,errors=errors,committed_unchanged=True,
            tangent=(k-kt).toarray(),conservative_potential=False))
    finally:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
        _RUN.reset(token)


def test_combined_analytical_torsion(monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    from anyfesolver.linalg import MatrixClass
    model=convert(simple()[0]);calls=[];original=solver.factorize;tip=.3;density=.12
    def observed(matrix,kind,*args,**kwargs):
        if str(kwargs.get('signature','')).startswith('nonlinear.static_newton'):
            calls.append(dict(kind=kind.value,asymmetry=float(np.linalg.norm((matrix-matrix.T).toarray()))))
        return original(matrix,kind,*args,**kwargs)
    monkeypatch.setattr(solver,'factorize',observed)
    result,evidence=solve_combined_static(model,SpatialNodalMoments(((3,tip,0.,0.),)),
        distributed_pattern=DistributedPattern(LinePattern(()),((1,density,0.,0.),)))
    save(tmp_path/'torsion.json',dict(result=packet(result),evidence=evidence,factorizations=calls))
    assert result.status=='completed',result.info
    for snapshot,step in zip(result.snapshots,result.steps):
        state=snapshot.element_states[1]
        for i,s in enumerate((0.,.5,1.)):
            angle=snapshot.load_factor*(tip*s+density*(s-s*s/2))/2
            np.testing.assert_allclose(state['committed_nodal_rotation_matrices'][i],rotation([angle,0.,0.]),atol=1e-11,rtol=0.)
        np.testing.assert_allclose(step.support_reactions['root'][3:],[-snapshot.load_factor*(tip+density),0.,0.],atol=1e-11,rtol=0.)
    assert calls and all(c['kind']==MatrixClass.GENERAL.value for c in calls)
    assert result.info['native_combined_couple_general_matrix'] and max(c['asymmetry'] for c in calls)>1e-3


def test_zero_nodal_limit_and_fixed_reaction(tmp_path):
    a,_=problem(False,False);b,_=problem(False,False)
    old,_=solve_distributed_model(a,pattern(a));new,events=solve_combined_static(b,None,distributed_pattern=pattern(b))
    save(tmp_path/'zero-limit.json',dict(old=packet(old),new=packet(new),events=events))
    assert old.status==new.status=='completed' and canonical(packet(old))==canonical(packet(new))
    root=convert(simple()[0]);moment=np.array([.2,-.1,.3])
    result,events=solve_combined_static(root,SpatialNodalMoments(((1,*map(float,moment)),)),steps=1)
    save(tmp_path/'root-reaction.json',dict(result=packet(result),events=events,reactions=result.steps[0].support_reactions))
    assert result.status=='completed';np.testing.assert_array_equal(result.displacements,np.zeros(18))
    np.testing.assert_allclose(result.steps[0].support_reactions['root'][3:],-moment,atol=1e-11,rtol=0.)


@pytest.mark.parametrize('mutation',['pattern','foreign-store','unissued-pose'])
def test_live_load_authority(mutation,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_combined_couples as module
    m,e=problem();store=store_for(m,e);before=canonical(store.materialize());u=sample()
    run=_Run(m,SpatialNodalMoments(((3,.1,.2,.3),)));token=_RUN.set(run)
    try:
        assemble_distributed_trial(m,u,store,pattern(m));external_at(m,store,u,.5,tangent=True)
        if mutation=='pattern':
            original=module.exp_chart_terms
            def change(v):
                value=original(v);run.proportional=SpatialNodalMoments(((3,.2,.2,.3),));return value
            monkeypatch.setattr(module,'exp_chart_terms',change)
            with pytest.raises(ValueError,match='authority'):external_at(m,store,u,.5,tangent=True)
            assert not store.has_active_trial
        elif mutation=='foreign-store':
            foreign,fe=problem();other=store_for(foreign,fe)
            with pytest.raises(ValueError,match='foreign'):external_at(m,other,u,.5,tangent=True)
        else:
            store.discard_trial(store.active_trial_token())
            with pytest.raises(ValueError,match='unissued'):external_at(m,store,u,.5,tangent=True)
        assert canonical(store.materialize())==before
        save(tmp_path/'authority.json',dict(mutation=mutation,rejected=True,committed_unchanged=True))
    finally:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
        _RUN.reset(token)


@pytest.mark.parametrize('kind',['effective-sum','external-norm','combined-norm','absent-node','wrong-type'])
def test_preflight_and_range(kind,tmp_path):
    m,e=problem(False,False)
    if kind=='effective-sum':
        large=SpatialNodalMoments(((3,1e308,0.,0.),))
        with pytest.raises(ValueError,match='range'):_Run(m,large,large).effective(1.)
    elif kind=='external-norm':
        store=store_for(m,e);run=_Run(m,SpatialNodalMoments(((3,1e200,0.,0.),)));token=_RUN.set(run)
        try:
            with pytest.raises(ValueError,match='range'):external_at(m,store,np.zeros(18),1.,tangent=True)
        finally:_RUN.reset(token)
    elif kind=='combined-norm':
        with pytest.raises(ValueError,match='range'):solve_combined_static(m,SpatialNodalMoments(((3,.1,0.,0.),)),
            distributed_pattern=DistributedPattern(LinePattern(((1,1e200,0.,0.),)),((1,.01,0.,0.),)))
    else:
        moments=SpatialNodalMoments(((999,.1,0.,0.),)) if kind=='absent-node' else ((3,.1,0.,0.),)
        with pytest.raises(ValueError):solve_combined_static(m,moments)
    assert _RUN.get() is None and _PROGRAM.get() is None and _ACTIVE.get() is None
    save(tmp_path/'rejected.json',dict(kind=kind,rejected=True,production_qualified=False))
