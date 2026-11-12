"""Prospective dimensional-gap and continuation discriminator, development only."""
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import numpy as np
import pytest
from anyfesolver.general_contact.native_s3_swept import native_s3_swept_admissible
from test_general_contact_native_multiple_s3_static import _example, rebuild, independent_energy
from anyfesolver.control import CancellationToken
from anyfesolver.general_contact import NativeContactActivationSettings


def candidate(*, activation=False, beams=1, stiffness=1.2e6, targets=(.25,.5,.75,1.), activation_gap=1e-7):
    base=_example.analysis(beams=beams,targets=targets,line_stiffness_n_per_m2=stiffness)
    owner=rebuild(base,pairs=tuple(replace(p,gap_tolerance_m=5e-5) for p in base.pairs),
        activation_settings=NativeContactActivationSettings(activation_gap,64) if activation else None)
    rows=[]
    current=[]; last_step=[]
    assemble=owner._assemble
    def measured(*args):
        started=time.perf_counter(); result=assemble(*args)
        rows.append({'parameter':args[2],'metrics':result[2],
            'minimum_gap_m':result[4].contact.minimum_gap_m,'elapsed_s':time.perf_counter()-started,
            'shell_displacement_m':args[1].tolist()})
        current[:]=[args,result]
        return result
    owner._assemble=measured
    original_step=owner._step
    def captured(residual,tangent):
        step,correction=original_step(residual,tangent)
        last_step[:]=[current[0],residual.copy(),tangent.copy(),step.copy()]
        return step,correction
    owner._step=captured
    run=owner.solve()
    probes=[];linear_error=None
    if run.status!='completed' and last_step:
        args,residual,tangent,step=last_step
        free=np.r_[owner.context.free,owner.context.count+owner.shell_free]
        linear_error=float(np.linalg.norm((tangent@step+residual)[free])/np.linalg.norm(residual[free]))
        for fraction in (1/256,1/512,1/1024,1/4096,1e-5,1e-6,1e-7):
            pose=owner.context.advance(args[0],fraction*step[:owner.context.count])
            u=args[1]+fraction*step[owner.context.count:]
            changed=assemble(pose,u,args[2],args[3])
            error=np.linalg.norm(((changed[0]-residual)/fraction-tangent@step)[free])/max(1.,np.linalg.norm((tangent@step)[free]))
            probes.append({'fraction':fraction,'metrics':changed[2],
                'minimum_gap_m':changed[4].contact.minimum_gap_m,
                'directional_error':float(error),'admitted':owner._contact_admissible(changed[4]),
                'active_intervals':[r.contact.active_intervals for _,r in changed[4].pairs]})
    record=os.environ.get('ANYSOLVER_DEVELOPMENT_RECORD')
    if record:
        Path(record).with_name(f'penalty_{beams}_{int(stiffness)}_{int(activation)}.json').write_text(json.dumps({
            'activation_gap_m':activation_gap,'maximum_geometry_queries':64,'line_stiffness_n_per_m2':stiffness,'gap_tolerance_m':5e-5,'targets':owner.program.targets,
            'profile':run.profile,'activation_queries':getattr(run,'activation_queries',0),
            'checkpoint_sha256':run.checkpoint_sha256,'status':run.status,'failure':run.failure,'completed_targets':run.accepted.completed_targets,
            'contact':run.accepted.contact.contact.to_dict(),'rows':rows,
            'free_linear_solve_error':linear_error,'probes':probes},
            default=lambda a:a.tolist(),indent=2),encoding='utf-8')
    return owner,run


@pytest.mark.parametrize('beams',(1,2))
def test_derived_stronger_line_penalty_and_dimensional_gap(beams):
    owner,run=candidate(activation=True,beams=beams,activation_gap=1e-7 if beams==1 else 1e-6)
    assert run.status=='completed',run.failure
    assert owner._contact_admissible(run.accepted.contact)
    independent_gap=independent_minimum_gap(owner,run.accepted.mechanical,run.accepted.shell_displacement)
    assert independent_gap>=-5e-5
    assert independent_gap==pytest.approx(run.accepted.contact.contact.minimum_gap_m,abs=1e-14)
    accepted=run.accepted; n=owner.context.count
    residual,tangent,metrics,_,work=owner._assemble(accepted.mechanical,accepted.shell_displacement,1.,accepted.native_histories)
    assert max(metrics)<=owner.equilibrium_tolerance
    assert np.all(np.abs(accepted.shell_displacement[owner.shell_free])>1e-10)
    np.testing.assert_allclose(work.contact.beam_force_n+work.contact.shell_force_n,0.,atol=1e-9)
    np.testing.assert_allclose(work.contact.beam_moment_nm+work.contact.shell_moment_nm,0.,atol=1e-9)
    reaction=np.zeros(3)
    for dof in owner.context.fixed:
        _,axis=divmod(dof,6)
        if axis<3:reaction[axis]+=residual[dof]
    for dof in owner.shell_fixed:
        _,axis=divmod(dof,6)
        if axis<3:reaction[axis]+=residual[n+dof]
    np.testing.assert_allclose(reaction,(0.,.01,.01*beams),atol=1e-8)
    assert abs(independent_energy(owner,accepted.mechanical,accepted.shell_displacement)-work.contact.potential_j)<1e-10
    direction=np.zeros(len(residual)); direction[n+owner.shell_free]=.002
    for i in range(beams):
        direction[owner.context.nodal_count+24*i]=.003
        direction[owner.context.slots[i][14]]=-.004
    errors=[]
    for h in (1e-4,1e-5,1e-6):
        plus=owner._assemble(owner.context.advance(accepted.mechanical,h*direction[:n]),accepted.shell_displacement+h*direction[n:],1.,accepted.native_histories)[0]
        minus=owner._assemble(owner.context.advance(accepted.mechanical,-h*direction[:n]),accepted.shell_displacement-h*direction[n:],1.,accepted.native_histories)[0]
        predicted=tangent@direction
        errors.append(float(np.linalg.norm((plus-minus)/(2*h)-predicted)/max(1.,np.linalg.norm(predicted))))
    assert max(errors)<1e-5
    record=os.environ.get('ANYSOLVER_DEVELOPMENT_RECORD')
    if record:
        Path(record).with_name(f'strong_work_{beams}.json').write_text(json.dumps({'independent_gap_m':independent_gap,'tangent_errors':errors,'metrics':metrics,'reaction_n':reaction.tolist(),'queries':run.activation_queries}))


def test_existing_fixed_target_nonpass_preserves_state():
    owner,run=candidate()
    assert run.status=='failed' and run.accepted.completed_targets==0
    assert 'line-search/gap limit' in run.failure
    np.testing.assert_array_equal(run.accepted.shell_displacement,owner.initial.shell_displacement)


def test_geometry_only_matches_full_work_at_retained_and_skin_poses():
    owner=_example.analysis(beams=2)
    for scale in (0.,.1,1.):
        step=np.zeros(owner.context.count)
        for i in range(2):
            step[owner.context.nodal_count+24*i+3:owner.context.nodal_count+24*i+6]=scale*np.array([.012,.02,.015])
            step[6*owner.context.nodes[i][-1]+2]=-.002*scale
        state=owner.context.advance(owner.initial.mechanical,step)
        u=owner.initial.shell_displacement.copy(); u[owner.shell_free]=scale*.0003
        for p,i in zip(owner.pairs,owner.beam_indices):
            nodes=owner.context.nodes[i]
            args=(state.positions[nodes],state.position_low[nodes],state.cell_rotations[i])
            result=p.work.evaluate(*args,u).contact
            gap,rim=p.work.contact.geometry_gap(*args,p.work.shell.current_positions(u))
            assert gap==result.minimum_gap_m
            assert rim==result.minimum_rim_clearance_m


def test_hidden_activation_prefix_and_complete_query_cap():
    base=_example.analysis(beams=2)
    owner=rebuild(base,activation_settings=NativeContactActivationSettings(1e-7,64))
    u=owner.initial.shell_displacement
    setup=np.zeros(owner.context.count)
    for node in owner.context.nodes[1][1:]: setup[6*node+2]=.04899
    setup[owner.context.nodal_count+24+3]=-np.pi/2-.05
    start=owner.context.advance(owner.initial.mechanical,setup)
    step=np.zeros(owner.context.count+len(u)); step[owner.context.nodal_count+24+3]=.1
    end=owner.context.advance(start,step[:owner.context.count])
    mid=owner.context.advance(start,.5*step[:owner.context.count])
    a=owner._contact(start,u); b=owner._contact(end,u); m=owner._contact(mid,u)
    assert all(r.contact.minimum_gap_m>0. for _,r in a.pairs+b.pairs)
    hidden=m.pairs[2][1].contact.minimum_gap_m
    assert -base.pairs[2].gap_tolerance_m < hidden < -owner.activation_settings.activation_gap_m
    assert owner._swept_admissible(start,u,step,end,u)
    with pytest.raises(RuntimeError,match='query budget exhausted'):
        owner._activation_fraction(start,u,step,1.,a,None)
    assert owner.activation_total_queries==64
    # The difficult full path is explicitly unresolved; a separate shorter
    # prefix can be certified without increasing the resource cap.
    bounded=rebuild(base,activation_settings=NativeContactActivationSettings(1e-7,64))
    fraction=bounded._activation_fraction(start,u,step,.125,a,None)
    assert 0.<fraction<=.125
    assert 0<bounded.activation_total_queries<=64
    # Recheck the actual chosen prefix using a finer independent dense path
    # diagnostic, alongside the production conservative certificate above.
    for alpha in np.linspace(0.,fraction,101):
        pose=owner.context.advance(start,alpha*step[:owner.context.count])
        assert independent_minimum_gap(owner,pose,u)>=-1e-7
    short=rebuild(base,activation_settings=NativeContactActivationSettings(1e-7,5))
    with pytest.raises(RuntimeError,match='query budget exhausted'):
        short._activation_fraction(start,u,step,1.,a,None)
    assert short.activation_total_queries==5


def test_projected_normal_spin_certificate():
    owner=_example.analysis(); u=owner.initial.shell_displacement
    start=owner.initial.mechanical; p=owner.pairs[0]; i=owner.beam_indices[0]
    step=np.zeros(owner.context.count+len(u)); step[owner.context.nodal_count+2]=.1
    end=owner.context.advance(start,step[:owner.context.count]); nodes=owner.context.nodes[i]
    def gap(pose):
        return p.work.contact.geometry_gap(pose.positions[nodes],pose.position_low[nodes],pose.cell_rotations[i],p.work.shell.current_positions(u))
    a,b=gap(start),gap(end)
    args=(owner.context,p.work,i,p.gap_tolerance_m,1,owner._recoverable_trial_error,start,u,step,end,u,*[a[0],b[0],a[1],b[1]])
    assert native_s3_swept_admissible(*args,geometry_only=True,projected_gap=True)
    assert not native_s3_swept_admissible(*args,geometry_only=True)
    for alpha in np.linspace(0.,1.,11):
        assert gap(owner.context.advance(start,alpha*step[:owner.context.count]))[0]==pytest.approx(.001,abs=1e-15)


def strong_owner(*, targets=(.25,.5,.75,1.), stiffness=1.2e6, settings=None):
    base=_example.analysis(targets=targets,line_stiffness_n_per_m2=stiffness)
    return rebuild(base,pairs=tuple(replace(p,gap_tolerance_m=5e-5) for p in base.pairs),
        activation_settings=NativeContactActivationSettings(1e-7,64) if settings is None else settings)


def test_activation_transaction_restart_and_identity(monkeypatch):
    owner=strong_owner(); token=CancellationToken()
    stopped=owner.solve(cancellation_token=token,progress=lambda *_:token.cancel('split at first target'))
    assert stopped.status=='cancelled' and stopped.accepted.completed_targets==1
    second=strong_owner(); later=second.pairs[-1].work; original=later.evaluate
    def fail(*_):raise RuntimeError('injected later activation pair failure')
    failed=second.solve(progress=lambda *_:monkeypatch.setattr(later,'evaluate',fail))
    assert failed.status=='failed' and failed.accepted.completed_targets==1
    assert 'later activation pair' in failed.failure
    assert failed.checkpoint==stopped.checkpoint
    np.testing.assert_array_equal(failed.accepted.shell_displacement,stopped.accepted.shell_displacement)
    monkeypatch.setattr(later,'evaluate',original)
    resumed=second.solve(checkpoint=failed.checkpoint,expected_sha256=failed.checkpoint_sha256)
    assert resumed.status=='completed',resumed.failure
    full=strong_owner().solve()
    assert full.status=='completed',full.failure
    assert resumed.checkpoint==full.checkpoint
    incompatible=strong_owner(settings=NativeContactActivationSettings(2e-7,64))
    with pytest.raises(ValueError,match='definition/settings binding'):
        incompatible.solve(checkpoint=failed.checkpoint,expected_sha256=failed.checkpoint_sha256)
    old=rebuild(owner,pairs=owner.pairs)
    with pytest.raises(ValueError,match='definition/settings binding'):
        old.solve(checkpoint=failed.checkpoint,expected_sha256=failed.checkpoint_sha256)
    zero=strong_owner(settings=NativeContactActivationSettings(1e-7,1))
    rejected=zero.solve()
    assert rejected.status=='failed' and rejected.accepted.completed_targets==0
    assert 'query budget exhausted' in rejected.failure
    np.testing.assert_array_equal(rejected.accepted.shell_displacement,zero.initial.shell_displacement)
    owner.activation_settings=NativeContactActivationSettings(2e-7,64)
    with pytest.raises(ValueError):owner._guard()


def test_stronger_setting_and_target_path_sensitivity():
    runs=[]
    for stiffness,targets in ((1.2e6,(.25,.5,.75,1.)),(2.4e6,(.25,.5,.75,1.)),(1.2e6,(.125,.25,.375,.5,.625,.75,.875,1.))):
        owner=strong_owner(stiffness=stiffness,targets=targets); run=owner.solve()
        assert run.status=='completed',run.failure
        assert owner._contact_admissible(run.accepted.contact)
        runs.append(run)
    assert -runs[1].accepted.contact.contact.minimum_gap_m < -runs[0].accepted.contact.contact.minimum_gap_m
    np.testing.assert_allclose(runs[2].accepted.shell_displacement,runs[0].accepted.shell_displacement,rtol=1e-6,atol=1e-10)
    np.testing.assert_allclose(runs[2].accepted.mechanical.positions,runs[0].accepted.mechanical.positions,rtol=1e-6,atol=1e-10)
    record=os.environ.get('ANYSOLVER_DEVELOPMENT_RECORD')
    if record:
        Path(record).with_name('sensitivity.json').write_text(json.dumps([{'gap_m':r.accepted.contact.contact.minimum_gap_m,'beam_force_n':r.accepted.contact.contact.beam_force_n.tolist(),'potential_j':r.accepted.contact.contact.potential_j,'queries':r.activation_queries} for r in runs]))


def independent_minimum_gap(owner,mechanical,u):
    """Direct retained interpolation and scalar quadratic extrema; no gap kernel."""
    minimum=float('inf')
    for pair,i in zip(owner.pairs,owner.beam_indices):
        skin=pair.work.shell
        vertices=skin.reference_positions+(skin.work_map@u).reshape(3,3)
        normal=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]); normal/=np.linalg.norm(normal)
        points=mechanical.positions[owner.context.nodes[i]]+mechanical.position_low[owner.context.nodes[i]]
        for c in (0,1):
            lift=mechanical.cell_rotations[i,c]@pair.work.contact.beam_reference.half_cell_lift(c,.5)
            constant=float(normal@(points[c]-vertices[0])-pair.work.contact.radius)
            linear=float(normal@(points[c+1]-points[c]+4*lift))
            quadratic=-4*float(normal@lift)
            stations=[0.,1.]
            if quadratic:
                turning=-linear/(2*quadratic)
                if 0.<turning<1.:stations.append(turning)
            minimum=min(minimum,*(constant+linear*t+quadratic*t*t for t in stations))
    return minimum


def test_public_tight_example_and_controls():
    for beams,gap in ((1,1e-7),(2,1e-6)):
        owner=_example.tight_gap_analysis(beams=beams)
        assert owner.profile=='NATIVE_B3_GE_MULTIPLE_S3_ACTIVATION_NEWTON_STATIC_DEV_V1'
        assert owner.activation_settings==NativeContactActivationSettings(gap,64)
        assert all(p.gap_tolerance_m==5e-5 and p.work.contact.stiffness==1.2e6 for p in owner.pairs)
    for gap,cap in ((True,64),(float('nan'),64),(0.,64),(1e-7,True),(1e-7,0),(1e-7,513)):
        with pytest.raises(ValueError):NativeContactActivationSettings(gap,cap)
    for gap in (1e-10,5e-5):
        with pytest.raises(ValueError,match='inside every pair gap budget'):
            strong_owner(settings=NativeContactActivationSettings(gap,64))
