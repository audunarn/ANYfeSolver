"""Bounded physical-side material friction: work, coverage, actual path and restart."""
import copy
import hashlib
import json
import os
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from examples.general_contact_s3_friction_physical_path import build_case,solve_case
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.s3_sliding_friction import S3SlidingFrictionManager,S3SlidingFrictionSettings

STAGE=os.environ.get('FRICTION_STAGE','unit')
EVIDENCE=Path(os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE','.'))


def save(value):
    name=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if name:Path(name).write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()))


def manager(case):
    m,d,s,f,_=case
    return S3SlidingFrictionManager(m,GeneralS3PatchManager(m,d,s),f)


def mode(model,nodes,axis):
    value=np.zeros(60)
    for node in nodes:value[model.mesh.dof_manager.get_node_dofs(node)[axis]]=1.
    return value


def wrench(model,u,residual):
    force=np.zeros(3);moment=np.zeros(3)
    for node in range(1,11):
        n=model.mesh.get_node(node);d=model.mesh.dof_manager.get_node_dofs(node)
        position=np.array((n.x,n.y,n.z))+u[d[:3]]
        force+=residual[d[:3]];moment+=np.cross(position,residual[d[:3]])+residual[d[3:]]
    return dict(force=float(np.linalg.norm(force)/.02),moment=float(np.linalg.norm(moment)/.02))


@pytest.mark.skipif(STAGE!='unit',reason='separate bounded stage')
def test_canonical_material_work_and_whole_path():
    from anyfesolver.general_contact.s3_friction_coverage import full_coverage
    results=[];reference=None
    for ps,pt in ((1,1),(1,-1),(-1,1),(-1,-1)):
        case=build_case(ps,pt);m=case[0];owner=manager(case)
        x=mode(m,range(1,7),0);z=mode(m,range(1,7),2);tx=mode(m,(8,9),0)
        pressed=-.015005*z;owner.commit(owner.prepare_commit(pressed));old=owner.checkpoint_state()
        u=pressed+.39*x+1e-8*tx
        force,K,_=owner.evaluate(u);direction=.01*x+.002*z+.003*tx
        h=1e-5
        fd=(owner.evaluate(u+h*direction,tangent=False)[0]-owner.evaluate(u-h*direction,tangent=False)[0])/(2*h)
        error=float(np.linalg.norm(K@direction-fd)/max(np.linalg.norm(fd),1e-14))
        assert error<=1e-5
        assert owner.checkpoint_state()==old
        candidate=owner.prepare_commit(u);assert set(candidate[2]['branch'])=={'slip'}
        assert candidate[2]['dissipation_increment_j']>0
        assert abs(candidate[2]['return_storage_drop_j']-candidate[2]['dissipation_increment_j']-candidate[2]['return_projection_loss_j'])<=1e-12
        balance=wrench(m,u,force);assert max(balance.values())<=1e-6
        # Independent uniform flat law and reaction sign, not production gap.
        normal=owner.normal.evaluate(u,tangent=False)[0]
        friction=force-normal
        assert float(friction@x)==pytest.approx(.16*.3*.005,abs=1e-12)
        assert float(friction@(x+mode(m,(7,8,9,10),0)))==pytest.approx(0.,abs=1e-12)
        assert float(friction@tx)!=0
        identifiers=[(p.adapter.element_id,p.adapter.node_ids,p.barycentric) for p in owner.points]
        if reference is None:reference=(force,candidate[2],identifiers)
        else:
            np.testing.assert_allclose(force,reference[0],rtol=1e-9,atol=1e-12)
            assert identifiers==reference[2]
            np.testing.assert_allclose(candidate[2]['traction_pa'],reference[1]['traction_pa'],atol=1e-12)
        # First-touch sign must agree for every director/winding declaration.
        touchcase=(*case[:3],replace(case[3],activation_history='linear_constant_normal_touch_dev'),case[4])
        touch=manager(touchcase);first=touch.prepare_commit(pressed+1e-7*x)
        alpha=.015/.015005
        np.testing.assert_allclose(first[2]['first_touch_fraction'],alpha,atol=1e-12)
        assert max(np.linalg.norm(t) for t in first[2]['traction_pa'])>0
        results.append(dict(source_polarity=ps,target_polarity=pt,tangent_error=error,wrench=balance,
            material_ids=identifiers,first_touch_fraction=first[2]['first_touch_fraction'],summary=candidate[2]))
    # Independent quadratic halfplane: y-t*x=.12-1.01*t+t^2;
    # both endpoints positive, midpoint negative. No sample-only acceptance.
    def adapter(ids,points,motion):
        return SimpleNamespace(node_ids=ids,reference_positions=np.array(points,float),
            work_map=sparse.csr_matrix(np.array(motion,float).reshape(9,1)))
    a=(0.,0.,0.);b=(3.,0.,0.);c=(3.,4.,0.);d=(0.,4.,0.)
    target={1:adapter((10,11,13),(a,b,d),((0,0,0),(0,3,0),(0,0,0))),
            2:adapter((11,12,13),(b,c,d),((0,3,0),(0,0,0),(0,0,0)))}
    source=adapter((1,2,3),((1.01,.12,.1),(1.015,.12,.1),(1.01,.125,.1)),((-1,0,0),)*3)
    fake=SimpleNamespace(n_dof=1,target_adapters=target,source_adapters=(source,),
        boundary_edges={(10,11),(11,12),(12,13),(10,13)},point_reference=(),point_maps=(),work_columns=np.array([0]))
    assert full_coverage(fake,np.zeros(1),np.zeros(1))
    assert full_coverage(fake,np.ones(1),np.ones(1))
    assert F(.12)-F(1,2)*F(1.01)+F(1,4)<0
    assert not full_coverage(fake,np.zeros(1),np.ones(1))
    # Actual stored dense point can be outside even if face corners are inside.
    fake.point_reference=(np.array((4.,.2,.1)),);fake.point_maps=(np.zeros((3,1)),)
    assert not full_coverage(fake,np.zeros(1),np.zeros(1))
    # Static oblique planes are deliberately outside the new certificate.
    oblique=copy.deepcopy(fake);oblique.point_reference=();oblique.point_maps=()
    for ad in oblique.target_adapters.values():ad.reference_positions[:,2]=.1*ad.reference_positions[:,0]
    assert not full_coverage(oblique,np.zeros(1),np.zeros(1))
    tiny=copy.deepcopy(oblique)
    for ad in (*tiny.source_adapters,*tiny.target_adapters.values()):
        ad.reference_positions*=1e-200;ad.work_map*=1e-200
    assert not full_coverage(tiny,np.zeros(1),np.zeros(1))
    save(dict(verified=True,polarity_cases=results,interior_only_refusal=True,dense_point_refusal=True,oblique_refusal=True))


def physical_gates(case,result,restart=None):
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    m=case[0];owner=manager(case);rows=[]
    if restart:owner.restore(restart['path_state']['friction_state'],np.asarray(restart['displacements']))
    for snap in result.snapshots:
        u=np.asarray(snap.displacements);step=snap.step_index
        # Literal admitted skin map: rotations zero, source bottom .030,
        # target top .015. Exact binary-pose z envelope covers all P1 faces.
        source,target=owner.normal._positions(u)
        independent_source=np.array([[m.mesh.get_node(n).x+u[m.mesh.dof_manager.get_node_dofs(n)[0]],m.mesh.get_node(n).y,.055-.025+u[m.mesh.dof_manager.get_node_dofs(n)[2]]] for n in owner.normal.law.source_node_ids])
        independent_target=np.array([[m.mesh.get_node(n).x+u[m.mesh.dof_manager.get_node_dofs(n)[0]],m.mesh.get_node(n).y,.025-.01] for n in owner.normal.law.target_node_ids])
        np.testing.assert_allclose(source,independent_source,rtol=0,atol=1e-17)
        np.testing.assert_allclose(target,independent_target,rtol=0,atol=1e-17)
        sourcez=[F(float(p[2])) for p in source];targetz=[F(float(p[2])) for p in target]
        gap=min(sourcez)-max(targetz);assert gap>=-F(5e-5)
        assert all(np.all(u[m.mesh.dof_manager.get_node_dofs(n)[3:]]==0.) for n in range(1,11))
        owner.require_friction_path(u,u)
        phase=result.info['force_displacement_history'][step-1]
        residual=owner.evaluate(u,tangent=False)[0]
        balance=wrench(m,u,residual);assert max(balance.values())<=1e-6
        # Independent elastic structural recovery uses the actual public five
        # layers and fresh elastic records, rather than transplanting terminal
        # pose-bound records onto another accepted pose.
        structural,_,states=_assemble_nonlinear_system(m,u,{},5,tangent=False,require_full_coordinates=True)
        _discard_nonlinear_state_candidate(result.element_states)
        total=structural+residual;rx,rz=phase['reaction_generalized']
        errors=[float(total@mode(m,range(1,7),0)-rx),float(total@mode(m,range(1,7),2)-rz)]
        errors.extend(float(total[m.mesh.dof_manager.get_node_dofs(n)[0]]) for n in (8,9))
        equilibrium=sum(abs(v) for v in errors)/.02;assert equilibrium<=1e-6
        target_work=float(residual[36:]@u[36:])
        if step in (3,5):assert abs(target_work)>1e-18
        owner.commit(owner.prepare_commit(u))
        # Normal and friction certificate from accepted commit; immutable
        # history reconstruction is used only to recover same-pose residual.
        rows.append(dict(step=step,gap_fraction=[gap.numerator,gap.denominator],gap_m=float(gap),
            target_right_ux_m=[float(u[m.mesh.dof_manager.get_node_dofs(n)[0]]) for n in (8,9)],
            friction=phase['friction'],contact=phase['contact'],
            coupled_equilibrium_normalized=equilibrium,complete_contact_wrench=balance,
            target_residual_dot_displacement_j=target_work))
        assert phase['contact']['gap_tolerance_met']
        assert phase['friction']['dissipation_increment_j']>=-1e-12
        assert abs(phase['friction']['return_storage_drop_j']-phase['friction']['dissipation_increment_j']-phase['friction']['return_projection_loss_j'])<=1e-12
        assert np.linalg.norm(np.array(phase['contact']['integration_force'])+phase['contact']['target_force'])/.02<=1e-6
        assert np.linalg.norm(np.array(phase['contact']['integration_moment'])+phase['contact']['target_moment'])/.02<=1e-6
        if step==6:
            assert gap>0 and set(phase['friction']['branch'])=={'open'}
            assert phase['friction']['storage_j']==0.
        else:assert phase['contact']['active_pressure_dofs']>0
    return rows


@pytest.mark.skipif(STAGE not in ('full','prefix','active-prefix','resume','active-resume'),reason='separate bounded stage')
def test_public_installed_lifecycle_and_restart():
    case=build_case();stage=STAGE;through={'prefix':6,'active-prefix':3}.get(stage,7)
    reference=None;restart=None
    if stage.endswith('resume'):
        prefix=json.loads((EVIDENCE/('active-prefix.json' if stage=='active-resume' else 'prefix.json')).read_text())
        restart=prefix['checkpoint'];reference=json.loads((EVIDENCE/'full.json').read_text())
    result=solve_case(case,through_step=through,restart=restart)
    report=dict(status=result.status,info=result.info,displacements=result.displacements,
        snapshots=[dict(step=s.step_index,displacements=s.displacements) for s in result.snapshots],
        checkpoint=result.to_restart_checkpoint() if result.restart_checkpoint else None)
    save(report)
    assert result.status=='completed',result.info
    expected=list(range(1,through+1)) if restart is None else list(range((3 if stage=='active-resume' else 6)+1,8))
    assert [s.step_index for s in result.snapshots]==expected
    report['poses']=physical_gates(case,result,restart);report['snapshot_inventory']=expected
    phases=result.info['force_displacement_history']
    if through>=3:
        assert set(phases[1]['friction']['branch'])=={'stick'}
        assert set(phases[2]['friction']['branch'])=={'slip'}
        assert phases[2]['friction']['target_elements']!=phases[0]['friction']['target_elements']
        if restart is None:assert max(abs(x) for x in report['poses'][2]['target_right_ux_m'])>1e-12
    if reference:
        np.testing.assert_allclose(result.displacements,reference['displacements'],rtol=1e-7,atol=1e-9)
        assert result.info['force_displacement_history']==reference['info']['force_displacement_history']
        report['fresh_restart_endpoint_and_history_parity']=True
    report['verified']=True;save(report)


@pytest.mark.skipif(STAGE not in ('legacy-baseline','safety'),reason='separate bounded stage')
def test_default_contract_and_refusal_atomicity(monkeypatch):
    from test_general_contact_s3_sliding_friction import _manager
    old=_manager()[-1]
    digest=hashlib.sha256(json.dumps(old.restart_contract(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if STAGE=='legacy-baseline':save(dict(legacy_contract_sha256=digest,verified=True));return
    baseline=EVIDENCE/'legacy-baseline.json'
    if not baseline.exists():baseline=EVIDENCE.parent/'legacy-baseline.json'
    assert digest==json.loads(baseline.read_text())['legacy_contract_sha256']
    full=json.loads((EVIDENCE/'full.json').read_text());opened=full['poses'][5]
    assert F(*opened['gap_fraction'])>0
    assert opened['contact']['active_pressure_dofs']==0 and opened['contact']['regularization_storage_j']==0.
    assert opened['friction']['storage_j']==0. and max(abs(v) for v in opened['target_right_ux_m'])<=1e-12
    # Records are physical traction, opposite the contact residual convention.
    assert all(v[0]<0 for v in full['poses'][2]['friction']['traction_pa'])
    assert all(v[0]>0 for v in full['poses'][4]['friction']['traction_pa'])
    case=build_case();owner=manager(case);before=owner.checkpoint_state()
    outside=mode(case[0],range(1,7),0)*2
    with pytest.raises(ValueError,match='certified|coverage'):owner.prepare_commit(outside)
    assert owner.checkpoint_state()==before
    owner.commit(owner.prepare_commit(-.015005*mode(case[0],range(1,7),2)))
    before=owner.checkpoint_state()
    monkeypatch.setattr(owner,'_work',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('trial exception')))
    with pytest.raises(RuntimeError,match='trial exception'):owner.prepare_commit(np.asarray(before['displacement']))
    assert owner.checkpoint_state()==before
    from anyfesolver.control import CancellationToken,SolveCancelled
    token=CancellationToken();token.cancel('physical friction precancel')
    from anyfesolver import solve_static_nonlinear
    m,d,s,f,p=build_case();model_before=repr(m)
    with pytest.raises(SolveCancelled):solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,contact_path_program=p,cancellation_token=token)
    assert repr(m)==model_before
    prefix=json.loads((EVIDENCE/'prefix.json').read_text())
    with pytest.raises(ValueError,match='contract|fingerprint|incompatible'):
        solve_case((m,d,s,replace(f,friction_coefficient=.4),p),restart=prefix['checkpoint'])
    tamper=copy.deepcopy(prefix['checkpoint']);tamper['displacements'][0]+=1e-4
    with pytest.raises(ValueError):solve_case(build_case(),restart=tamper)
    # Exact friction subtype envelope must refuse before a material assembly.
    from test_general_contact_s3_friction_static import _case
    import anyfesolver.nonlinear_static as driver
    import anyfesolver.general_contact.static_path as path_driver
    sentinel=lambda *a,**k: (_ for _ in ()).throw(AssertionError('material allocated'))
    monkeypatch.setattr(driver,'_assemble_nonlinear_system',sentinel)
    monkeypatch.setattr(path_driver,'_assemble_nonlinear_system',sentinel)
    with pytest.raises(ValueError,match='S3SlidingFrictionSettings'):
        solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=_case()[3],contact_path_program=p)
    save(dict(verified=True,legacy_contract_sha256=digest,candidate_exception_atomic=True,
        rejected_coverage_atomic=True,precancel_declared_model_unchanged=True,incompatible_friction_refused=True,tamper_refused=True,
        wrong_friction_subtype_before_material=True,saved_full_strict_normal_friction_release=True,
        saved_full_slip_reversal_signs=True))


@pytest.mark.skipif(STAGE!='rollback',reason='separate bounded stage')
def test_public_trial_exception_and_inflight_cancellation(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact.runtime import active_contact_manager
    import anyfesolver.general_contact.s3_friction_static as factory
    import anyfesolver.nonlinear_static as driver
    prefix=json.loads((EVIDENCE/'active-prefix.json').read_text())['checkpoint']
    immutable=copy.deepcopy(prefix);expected=prefix['path_state']['friction_state']
    normalize=lambda x:json.loads(json.dumps(x))
    original_factory=factory.create_s3_friction_manager;reports=[]
    for kind in ('exception','cancellation'):
        case=build_case();m,d,s,f,p=case;before=repr(m);token=CancellationToken();captured=[];calls=[]
        def capture(model,normal,settings):
            owner=original_factory(model,normal,settings);captured.append(owner)
            work=owner._work;restore=owner.restore;armed=[False]
            def arm_after_restore(*a,**k):
                restored=restore(*a,**k);armed[0]=True;return restored
            owner.restore=arm_after_restore
            def inject(*a,**k):
                if armed[0] and not calls:
                    calls.append(True)
                    assert normalize(owner.checkpoint_state())==expected
                    if kind=='exception':raise RuntimeError('injected public trial failure')
                    token.cancel('injected in-flight friction cancellation')
                return work(*a,**k)
            owner._work=inject
            return owner
        context=active_contact_manager()
        with monkeypatch.context() as patch:
            patch.setattr(factory,'create_s3_friction_manager',capture)
            error=RuntimeError if kind=='exception' else SolveCancelled
            with pytest.raises(error):
                driver.solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,
                    contact_path_program=p,restart_checkpoint=prefix,tolerance=1e-12,max_iterations=18,
                    emit_restart_checkpoint=True,record_increment_snapshots=True,cancellation_token=token)
        assert calls and len(captured)==1
        assert normalize(captured[0].checkpoint_state())==expected
        assert np.all(captured[0].normal.normal_multipliers==0.)
        assert active_contact_manager() is context and repr(m)==before and prefix==immutable
        reports.append(dict(kind=kind,trial_entered=True,committed_active_origin_unchanged=True,
            normal_multipliers_unchanged=True,context_restored=True,checkpoint_unchanged=True,
            declared_model_unchanged=True))
    # Public unsafe seed: no material trial or accepted contact/history commit.
    from anyfesolver.general_contact import ContactPathProgram
    m,d,s,f,p=build_case();bad=ContactPathProgram(p.controls,((0.,0.),(2.,-.015005)),1)
    import anyfesolver.general_contact.static_path as path_driver
    rejected_owners=[];before=repr(m);context=active_contact_manager()
    def capture_rejected(*a):
        owner=original_factory(*a);rejected_owners.append(owner);return owner
    with monkeypatch.context() as patch:
        sentinel=lambda *a,**k: (_ for _ in ()).throw(AssertionError('unsafe seed material trial'))
        patch.setattr(factory,'create_s3_friction_manager',capture_rejected)
        patch.setattr(driver,'_assemble_nonlinear_system',sentinel)
        patch.setattr(path_driver,'_assemble_nonlinear_system',sentinel)
        rejected=driver.solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,
            contact_path_program=bad,tolerance=1e-12,max_iterations=18,emit_restart_checkpoint=True)
    assert rejected.info['failure_reason']=='contact_path_predictor_gap_exceeded'
    assert rejected.info['force_displacement_history']==[]
    assert rejected.restart_checkpoint is None and rejected.info['contact_path']['accepted_step']==0
    assert len(rejected_owners)==1 and all(h.target_element is None for h in rejected_owners[0].history)
    assert np.all(rejected_owners[0].committed_displacement==0.)
    assert repr(m)==before and active_contact_manager() is context
    save(dict(verified=True,public_restored_active_trial_rollback=reports,unsafe_seed_before_material=True))


def test_uncancelled_sliding_friction_reports_zero_reaction_predictor(monkeypatch):
    # Outside every stage gate: one real step on the unchanged physical-side
    # build_case() fixture. The call-through spy runs the real helper first,
    # verifies the projected-profile zero allowance, checks its finite
    # nonnegative result and returns zero; the solve completes with the zero
    # reporting diagnostic and the expected step-one contact-activation branch.
    import anyfesolver.general_contact.static_path as path_driver
    real = path_driver._elastic_displacement_force_predictor
    seen = []
    def spy(*args, **kwargs):
        seen.append(kwargs.get("allow_zero", False))
        value = real(*args, **kwargs)
        assert np.isfinite(value) and value >= 0.
        return 0.
    monkeypatch.setattr(path_driver, "_elastic_displacement_force_predictor", spy)
    case = build_case()
    result = solve_case(case, through_step=1)
    assert seen == [True]
    assert result.status == 'completed', result.info
    assert result.info['contact_free_elastic_force_predictor_n'] == 0.
    assert set(result.info['force_displacement_history'][0]['friction']['branch']) == {'activation'}
