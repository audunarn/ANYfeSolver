"""Unchanged fixture; exact mesh-owner control lift and installed workflow."""
import os,json,copy,hashlib
from pathlib import Path
import numpy as np
import pytest
from dataclasses import replace
from examples.general_contact_s3_mesh_control_lift import build_case
import examples.general_contact_s3_mesh_moving_friction as old_example
import test_general_contact_s3_mesh_moving_friction as reuse


def _patch_fixture(monkeypatch):
    monkeypatch.setattr(old_example,'build_case',build_case)


@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE') or not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not os.environ.get('CONTROL_LIFT_BASELINE'),
    reason='Archived staged inputs FRICTION_STAGE, AUTOMATIC_WORKFLOW_EVIDENCE, CONTROL_LIFT_BASELINE absent; this replay requires its original workflow evidence.')
def test_lift_public(monkeypatch):
    _patch_fixture(monkeypatch)
    if os.environ.get('MESH_CAPTURE')=='1':
        import sys
        original_trace=sys.settrace;captured=[]
        output=Path(os.environ['AUTOMATIC_RUNTIME_RESULT']).parent/'diagnostic-extra.json'
        def attach(existing):
            def combined(frame,event,arg):
                existing(frame,event,arg)
                if (event=='return' and frame.f_code.co_name=='evaluate'
                        and frame.f_code.co_filename.endswith('semismooth_path.py') and isinstance(arg,tuple)):
                    captured.append(dict(merit=float(arg[-1]),residual_l2_n=float(np.linalg.norm(arg[4])),
                        control_error_m=np.asarray(arg[5]).tolist(),complementarity_l1_n=float(np.sum(np.abs(arg[6]))),
                        reactions_n=np.asarray(frame.f_locals['at_mu']).tolist(),
                        seed_or_candidate_controls=np.asarray(frame.f_locals['control_rows']@frame.f_locals['at_q']).tolist()))
                    output.write_text(json.dumps(captured,indent=2))
                return combined
            original_trace(combined)
        monkeypatch.setattr(sys,'settrace',attach)
    reuse.test_mesh_public()
    stage=os.environ['FRICTION_STAGE'];evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    data=json.loads((evidence/(stage+'.json')).read_text())
    assert data['info']['contact_path']['path_predictor']=='control_projection_dev'
    if stage in ('A1','A2','B2'):
        old=Path(os.environ['CONTROL_LIFT_BASELINE'])/'outputs'/('A1.json' if stage=='A1' else 'A2.json')
        baseline=json.loads(old.read_text())
        np.testing.assert_allclose(data['displacement'],baseline['displacement'],rtol=0.,atol=1e-10)
        np.testing.assert_allclose(data['checkpoint']['path_state']['normal_forces'],baseline['checkpoint']['path_state']['normal_forces'],rtol=1e-8,atol=1e-12)


@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE') or not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='Archived staged inputs FRICTION_STAGE, AUTOMATIC_WORKFLOW_EVIDENCE absent; this replay requires its original workflow evidence.')
def test_lift_proof(monkeypatch):
    _patch_fixture(monkeypatch)
    reuse.test_mesh_saved_proof()


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='Archived staged inputs AUTOMATIC_WORKFLOW_EVIDENCE absent; this replay requires its original workflow evidence.')
def test_lift_rollback(monkeypatch):
    _patch_fixture(monkeypatch)
    reuse.test_mesh_rollback(monkeypatch)


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not os.environ.get('CONTROL_LIFT_BASELINE'),
    reason='Archived staged inputs AUTOMATIC_WORKFLOW_EVIDENCE, CONTROL_LIFT_BASELINE absent; this replay requires its original workflow evidence.')
def test_lift_safety(monkeypatch):
    _patch_fixture(monkeypatch)
    reuse.test_mesh_refusal(monkeypatch)
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.general_contact.core import _contact_settings_contract
    baseline=Path(os.environ['CONTROL_LIFT_BASELINE'])
    incoming=json.loads((baseline/'outputs/A2.json').read_text())['checkpoint']
    m,d,s,f,p,_=build_case();before=repr(m)
    assert 'path_predictor' not in _contact_settings_contract(replace(s,path_predictor='none'))
    assert _contact_settings_contract(s)['path_predictor']=='control_projection_dev'
    supplied=copy.deepcopy(incoming)
    with pytest.raises(ValueError):
        solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,contact_path_program=p,
            restart_checkpoint=supplied,tolerance=1e-12,max_iterations=18)
    assert supplied==incoming and repr(m)==before
    current=json.loads((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'A2.json').read_text())['checkpoint']
    m,d,s,f,p,_=build_case();before=repr(m);supplied=copy.deepcopy(current)
    with pytest.raises(ValueError):
        solve_static_nonlinear(m,contact=d,contact_settings=replace(s,path_predictor='none'),
            contact_friction=f,contact_path_program=p,restart_checkpoint=supplied,
            tolerance=1e-12,max_iterations=18)
    assert supplied==current and repr(m)==before
    m,d,s,f,p,_=build_case()
    restored=solve_static_nonlinear(m,contact=d,contact_settings=replace(s,path_predictor='none'),
        contact_friction=f,contact_path_program=p,restart_checkpoint=incoming,
        tolerance=1e-12,max_iterations=18,max_load_factor=2/6,record_increment_snapshots=True)
    assert restored.status=='completed' and not restored.snapshots
    np.testing.assert_allclose(restored.displacements,json.loads((baseline/'outputs/A2.json').read_text())['displacement'],rtol=0.,atol=1e-10)
    reuse._save(dict(legacy_default_omission=True,default_none_checkpoint_continuation=True,old_none_restart_refused=True,
        predictor_restart_under_none_refused=True,precancel_tamper_and_wrong_owner_reused=True))


def test_lift_admission(monkeypatch):
    from dataclasses import asdict
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.general_contact.core import validate_general_contact
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    from anyfesolver.general_contact.s3_sliding_friction import S3MovingNormalFrictionSettings
    m,d,s,f,p,_=build_case()
    assert validate_general_contact(m,d,s,analysis_control='contact_path').valid
    assert not validate_general_contact(m,d,s,analysis_control='force').valid
    normal=GeneralS3PatchManager(m,d,s);owner=create_s3_friction_manager(m,normal,f)
    initial=np.zeros(60);seed=initial.copy()
    for node in (1,2,3,4,5):seed[m.mesh.dof_manager.get_node_dofs(node)[0]]=.39
    before=copy.deepcopy(owner.checkpoint_state())
    assert owner.admissible_step_fraction(initial,seed)==1.
    assert owner.checkpoint_state()==before
    import anyfesolver.general_contact.static_path as consumer
    with monkeypatch.context() as patch:
        patch.setattr(consumer,'_assemble_nonlinear_system',lambda *a,**k:(_ for _ in ()).throw(AssertionError('wrong owner reached material')))
        with pytest.raises(ValueError,match='moving-normal friction requires'):
            solve_static_nonlinear(m,contact=d,contact_settings=s,
                contact_friction=S3MovingNormalFrictionSettings(**asdict(f)),contact_path_program=p)
    reuse._save(dict(normal_prerequisite_admitted=True,unused_force_context_refused=True,
        actual_full_translation_chord_certified=True,no_history_commit=True,onepair_subtype_refused_before_material=True))


def test_lift_unsafe_seed(monkeypatch):
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.general_contact.runtime import active_contact_manager
    import anyfesolver.general_contact.static_path as consumer
    import anyfesolver.general_contact.s3_friction_static as factory
    m,d,s,f,p,_=build_case();p=replace(p,waypoints=((0.,0.),(.95,0.),*p.waypoints[2:]))
    before=repr(m);context=active_contact_manager();created=[];origins=[]
    original=factory.create_s3_friction_manager
    def capture(*args):
        owner=original(*args);created.append(owner);origins.append(copy.deepcopy(owner.checkpoint_state()));return owner
    with monkeypatch.context() as patch:
        patch.setattr(factory,'create_s3_friction_manager',capture)
        patch.setattr(consumer,'_assemble_nonlinear_system',lambda *a,**k:(_ for _ in ()).throw(AssertionError('unsafe lift reached material')))
        result=solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,
            contact_path_program=p,tolerance=1e-12,max_iterations=18,max_load_factor=1/6)
    assert result.status!='completed'
    assert not result.snapshots
    assert created and created[0].checkpoint_state()==origins[0]
    assert repr(m)==before and active_contact_manager() is context
    reuse._save(dict(status=result.status,info=result.info,unsafe_seed_refused_before_material=True,
        no_accepted_snapshot=True,committed_history_unchanged=True,context_restored=True))


def test_lift_dyadic_enclosure():
    from fractions import Fraction as F
    from anyfesolver.general_contact.s3_friction_dyadic_interval import _bounds,compress,outward_dyadic_160,METHOD
    from anyfesolver.general_contact import s3_friction_chart_interval as I
    from anyfesolver.general_contact.s3_friction_mesh_coverage import mesh_moving_coverage
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    values=[F(0),F(1,3),F(-1,3),F(2**401+17,3),F(-2**401-17,3),F(1,2**1074),F(-1,2**1074),F(2**311-1,2**313+1)]
    for x in values:
        lower,upper=_bounds(x,160)
        assert lower<=x<=upper
        if x==0:assert lower==upper==0
    a=(F(1,3),F(2,3));b=(F(7,5),F(8,5))
    exact=[I.add(a,b),I.sub(a,b),I.mul(a,b),I.div(a,b),I.square(a),I.sqrt_interval(b),I.normalize((a,b,(F(2),F(3))))]
    def contains(outer,inner):
        if isinstance(inner[0],tuple):return all(contains(x,y) for x,y in zip(outer,inner))
        return outer[0]<=inner[0]<=inner[1]<=outer[1]
    with outward_dyadic_160():
        wide=[I.add(a,b),I.sub(a,b),I.mul(a,b),I.div(a,b),I.square(a),I.sqrt_interval(b),I.normalize((a,b,(F(2),F(3))))]
    assert all(contains(x,y) for x,y in zip(wide,exact))
    assert compress(a)==a
    with pytest.raises(RuntimeError):
        with outward_dyadic_160():raise RuntimeError('scope unwind')
    assert compress(a)==a and I.div(a,b)==exact[3]
    reuse._save(dict(integer_and_composed_enclosures=True,scope_reset_and_default_values=True,geometry_pending=True))
    m,d,s,f,p,_=build_case();owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
    first=np.zeros(60);safe=first.copy();unsafe=first.copy()
    for node in (1,2,3,4,5):
        dof=m.mesh.dof_manager.get_node_dofs(node)[0];safe[dof]=.39;unsafe[dof]=.95
    corpus=[(first,first),(first,safe),(unsafe,unsafe)]
    old=[mesh_moving_coverage(owner,*case) for case in corpus]
    with outward_dyadic_160():new=[mesh_moving_coverage(owner,*case) for case in corpus]
    assert old==new==[True,True,False]
    assert owner.restart_contract()['material_path_interval']==METHOD
    none=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,replace(s,path_predictor='none')),f)
    assert 'material_path_interval' not in none.restart_contract()
    reuse._save(dict(integer_enclosures=True,composed_primary_operation_enclosures=True,
        error_scope_reset=True,default_primary_values_exact_equal=True,branch_corpus=old,
        optin_method=METHOD,none_contract_omits_method=True))


def test_lift_physical_ray_bound(monkeypatch):
    from fractions import Fraction as F
    from anyfesolver.general_contact.s3_friction_mesh_gap import pair_lower, _constant
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    # Remote target corners lie far below the extrapolated tilted source plane;
    # actual covered source gap is -1um at its lowest corner.
    source=np.array([[.2,.2,0.],[.3,.2,-1e-6],[.2,.3,0.]])
    target=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    lo=pair_lower(_constant([source])[0],_constant([target])[0])
    # Exact squared radical inequality: |g| = 1um*sqrt(1+(1e-5)^2).
    assert -1.001e-6 < lo <= -1e-6
    unsafe=source.copy();unsafe[:,2]-=60e-6
    assert pair_lower(_constant([unsafe])[0],_constant([target])[0]) < -50e-6
    opposite=target[[0,2,1]]
    with pytest.raises(ValueError):pair_lower(_constant([source])[0],_constant([opposite])[0])
    # Binary affine nodes: safe endpoints but genuinely unsafe time interior.
    s0=np.array([[.8,.1,.1],[.85,.1,.1],[.8,.15,.1]])
    s1=s0+np.array([-.6,0.,.2])
    t0=target;t1=target.copy();t1[:,2]=t1[:,0]
    affine=lambda a,b:tuple(tuple(((F(float(x)),F(float(x))),(F(float(y))-F(float(x)),)*2) for x,y in zip(p,q)) for p,q in zip(a,b))
    assert pair_lower(_constant([s0])[0],_constant([t0])[0])>=0
    assert pair_lower(_constant([s1])[0],_constant([t1])[0])>=0
    assert pair_lower(affine(s0,s1),affine(t0,t1))<-50e-6
    midpoint=[(F(float(x))+F(float(y)))/2 for x,y in zip(s0[0],s1[0])]
    exact_mid_gap=midpoint[2]-midpoint[0]/2
    assert exact_mid_gap < -F(50,1000000)
    m,d,settings,f,p,_=build_case();normal=GeneralS3PatchManager(m,d,settings)
    owner=create_s3_friction_manager(m,normal,f)
    zero=np.zeros(60)
    assert normal._gap_lower(normal._positions(zero))>=-50e-6
    assert normal._swept_admissible(zero,zero)
    assert owner.restart_contract()['physical_gap_certificate']=='covered_source_ray_interval_v1'
    none=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,replace(settings,path_predictor='none')),f)
    assert 'physical_gap_certificate' not in none.restart_contract()
    # A successful early pair must never hide a later unresolved pair.
    import anyfesolver.general_contact.s3_friction_mesh_gap as gap
    calls=[0]
    def partial(*args):
        calls[0]+=1
        if calls[0]%2==0:raise ValueError('later pair unresolved')
        return 0.
    with monkeypatch.context() as patch:
        patch.setattr(gap,'pair_lower',partial)
        patch.setattr(owner,'require_friction_path',lambda *args:None)
        assert gap.path_lower(owner,zero,zero)==-np.inf
        # Actual global path with source passing through the target plane.
    end=zero.copy()
    for node in (2,3,4,5):end[m.mesh.dof_manager.get_node_dofs(node)[2]]=-100e-6
    assert not normal._swept_admissible(zero,end)
    reuse._save(dict(local_source_ray_lower_m=lo,unsafe_corner_refused=True,
        reversed_alignment_refused=True,normal_scalar_and_whole_path_consumers=True,
        method='covered_source_ray_interval_v1',metric='ideal_physical_source_ray_over_enclosed_skins',
        computed_jet2_rounding_qualification=False,default_none_method_omitted=True,
        interior_binary_ray_witness_m=float(exact_mid_gap)))


def test_lift_controls(monkeypatch):
    _patch_fixture(monkeypatch)
    reuse.test_mesh_certificate_controls(monkeypatch)


def test_lift_selector_integrity(monkeypatch):
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    reports=[]
    for change in ('settings','profile','owner'):
        m,d,s,f,p,_=build_case();normal=GeneralS3PatchManager(m,d,s)
        owner=create_s3_friction_manager(m,normal,f);before=copy.deepcopy(owner.checkpoint_state())
        zero=np.zeros(60);positions=normal._positions(zero)
        if change=='settings':normal.settings=replace(s,path_predictor='none')
        elif change=='profile':normal.profile='changed_profile'
        else:normal._mesh_physical_ray_owner=None
        with pytest.raises(ValueError,match='prepared mesh'):
            owner.require_friction_path(zero,zero)
        with pytest.raises(ValueError,match='prepared mesh'):
            normal._gap_lower(positions)
        assert owner.checkpoint_state()==before
        reports.append(change)
    reuse._save(dict(selector_profile_owner_mutations_refused=reports,committed_history_unchanged=True))
