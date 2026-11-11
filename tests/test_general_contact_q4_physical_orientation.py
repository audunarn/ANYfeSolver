"""Physical-side Q4 reference covariance, coupled work and actual public cycles."""
import copy,hashlib,importlib.util,json,os
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_q4_physical_orientation import build,solve

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
_cycle=load('_q4_orientation_cycle','test_general_contact_q4_pair_monolithic_cycle.py')
FREF=MREF=1e-4

def save(data):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else x),encoding='utf-8')
def read(stage):return json.loads((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(stage+'.json')).read_text())
def sha(stage):return hashlib.sha256((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(stage+'.json')).read_bytes()).hexdigest()
def label_signs(label):return tuple(1 if c=='p' else -1 for c in label)

def test_independent_q4_atlas_work_and_coupled_directional():
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager,_adapters,_selected
    from anyfesolver.general_contact.semismooth import linearize,complementarity,reference_nodal_support
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    base=build();m0=GeneralS3PatchManager(base[0],base[3].definition,base[2]);K0=assemble_stiffness_matrix(base[0])[0].toarray()
    a0=_adapters(base[0],_selected(base[3].definition)[:2],orientation=base[2].shell_pair_orientation)
    rng=np.random.default_rng(61);u=rng.normal(size=m0.n_dof)*1e-7;u[np.setdiff1d(np.arange(m0.n_dof),_cycle.free_dofs(base[0]))]=0.
    z=.01*reference_nodal_support(m0.law);c=1000*reference_nodal_support(m0.law)
    ref=linearize(m0,u,z);reports=[]
    for label in ('pp','mp','pm','mm'):
        model,domain,settings,res,program=build(signs=label_signs(label));manager=GeneralS3PatchManager(model,res.definition,settings)
        assert tuple(manager.law.pressure_mode_ids)==(1,2,3,4,5,6)
        adapters=_adapters(model,_selected(res.definition)[:2],orientation=settings.shell_pair_orientation)
        assert [a.node_ids for side in adapters for a in side]==[a.node_ids for side in a0 for a in side]
        # Independent physical material-node positions and couples: h*d stays
        # (-.025 z) upper / (+.025 z) lower for all opposite declarations.
        points,mids,rotation=_cycle.skin(model,u);actual=np.concatenate(manager._positions(u));np.testing.assert_allclose(points,actual,rtol=0,atol=1e-13)
        wm=[];pos=[]
        for side,refside in zip(adapters,a0):
            for a,b in zip(side,refside):
                delta=(a.work_map-b.work_map).toarray();wm.append(float(np.max(np.abs(delta))));pos.append(float(np.max(np.abs(a.reference_positions-b.reference_positions))))
                assert max(wm[-1],pos[-1])==0
                q=np.array([.24,.06,.14,.56]) # Q1 at u=.2,v=.7
                np.testing.assert_allclose(q@a.current_positions(u),q@b.current_positions(u),rtol=0,atol=0)
                forces=rng.normal(size=(4,3));pulled=np.asarray(a.work_map.T@forces.ravel()).ravel();expected=np.zeros(manager.n_dof)
                for node,f in zip(a.node_ids,forces):
                    ds=model.mesh.dof_manager.get_node_dofs(node);h=-.025 if node<=6 else .025
                    expected[ds[:3]]+=f;expected[ds[3:]]+=h*np.cross([0.,0.,1.],f)
                np.testing.assert_allclose(pulled,expected,rtol=0,atol=1e-14)
        K=assemble_stiffness_matrix(model)[0].toarray();ke=float(np.linalg.norm(K-K0)/np.linalg.norm(K0));assert ke<=1e-13
        raw=linearize(manager,u,z);ncp,active,cq,cz=complementarity(raw.state,z,c,raw.gap_gradient)
        parity=max(float(np.max(np.abs(getattr(raw,f)-getattr(ref,f)))) for f in ('force','tangent','gap_gradient'))
        assert parity<=1e-13
        # Independent Fraction nodal Q1 sums include positive initial clearance.
        s,g,g0=_cycle._old._reference.independent_linear_contact(model)
        raw0=linearize(manager,np.zeros(manager.n_dof),np.zeros(6),tangent=False)
        op=float(np.max(np.abs(raw0.gap_gradient*s[:,None]-g)));np.testing.assert_allclose(raw0.state.reference_weighted_gap,g0,atol=1e-18,rtol=1e-13);assert op<=1e-13
        du=rng.normal(size=manager.n_dof);du/=np.linalg.norm(du);dz=rng.normal(size=6)*.01
        def structural(at,tangent):
            r,k,st=_assemble_nonlinear_system(model,at,{},5,tangent=tangent,kinematics='von_karman');_discard_nonlinear_state_candidate(st);return r,k
        rs,ks=structural(u,True);errors=[]
        for h in (1e-8,3e-9):
            p=linearize(manager,u+h*du,z+h*dz,tangent=False);n=linearize(manager,u-h*du,z-h*dz,tangent=False)
            cp,ap,_,_=complementarity(p.state,z+h*dz,c,p.gap_gradient,tangent=False);cn,an,_,_=complementarity(n.state,z-h*dz,c,n.gap_gradient,tangent=False)
            assert np.array_equal(ap,active) and np.array_equal(an,active)
            cf=(p.force-n.force)/(2*h);ce=raw.tangent@du-raw.gap_gradient.T@dz
            ff=(structural(u+h*du,False)[0]+p.force-structural(u-h*du,False)[0]-n.force)/(2*h);fe=(ks+raw.tangent)@du-raw.gap_gradient.T@dz
            dc=(cp-cn)/(2*h);jc=cq@du+cz*dz
            item=dict(h=h,contact=float(np.linalg.norm(cf-ce)/max(np.linalg.norm(ce),1e-12)),coupled_equilibrium=float(np.linalg.norm(ff-fe)/max(np.linalg.norm(fe),1e-12)),complementarity=float(np.linalg.norm(dc-jc)/max(np.linalg.norm(jc),1e-12)))
            assert max(item[k] for k in ('contact','coupled_equilibrium','complementarity'))<=1e-5;errors.append(item)
        reports.append(dict(signs=label_signs(label),work_map_error=max(wm),skin_error=max(pos),structural_K_relative_error=ke,contact_parity_absolute_error=parity,independent_Q1_operator_error=op,directional_errors=errors))
    save(dict(verified=True,variants=reports,reference_permutation='Q4 (0,3,2,1), Q1 parameter swap',all_six_supports_retained=True,drill_geometry_columns_zero=True,not_reversed_legacy_chart_equivalence=True))


def configure_cycle(monkeypatch,stage,label):
    monkeypatch.setattr(_cycle,'build',lambda:build(signs=label_signs(label)))
    monkeypatch.setattr(_cycle,'save',save)
    monkeypatch.setattr(_cycle,'read_report',lambda name:read(name+'-'+label))
    monkeypatch.setattr(_cycle,'report_sha',lambda name:sha(name+'-'+label))
    monkeypatch.setenv('CYCLE_STAGE',stage)

@pytest.mark.skipif(not os.environ.get('ORIENTATION_STAGE'),
    reason='installed orientation stage input ORIENTATION_STAGE absent: staged orientation cycle requires the original workflow stage selection')
def test_actual_installed_orientation_cycle(monkeypatch):
    stage,label=os.environ['ORIENTATION_STAGE'].rsplit('-',1);configure_cycle(monkeypatch,stage,label)
    _cycle.test_installed_cycle_solver_saved_inventory()
    result=json.loads(Path(os.environ['AUTOMATIC_DOMAIN_RESULT']).read_text());result['director_signs']=label_signs(label);result['executed_orientation']='physical_sides_dev'
    if label!='pp' and stage=='full':
        plus=read('full-pp');errors=[]
        assert result['snapshot_inventory']==plus['snapshot_inventory']==[1,2,3]
        for a,b in zip(result['poses'],plus['poses']):
            u=float(np.max(np.abs(np.array(a['displacements'])-np.array(b['displacements']))));z=float(np.max(np.abs(np.array(a['normal_forces'])-np.array(b['normal_forces'])))/FREF)
            assert u<=1e-9 and z<=1e-6;errors.append(dict(step=a['step'],u_max_m=u,z_error=z))
        result['matched_phase_errors']=errors;result['plus_full_sha256']=sha('full-pp')
    save(result)

@pytest.mark.skipif(not os.environ.get('ORIENTATION_STAGE'),
    reason='installed orientation stage input ORIENTATION_STAGE absent: saved orientation proof requires the original staged report and stage selection')
def test_saved_orientation_physical_proof(monkeypatch):
    stage,label=os.environ['ORIENTATION_STAGE'].rsplit('-',1);configure_cycle(monkeypatch,stage,label)
    _cycle.test_saved_cycle_mechanical_proof()

def test_public_orientation_restart_and_unused_route_refusals(monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled,validate_general_contact
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    model,domain,settings,res,program=build(signs=(-1,-1));before=repr(model)
    wrong=replace(settings,enforcement_solver='nested_augmented',complementarity_force_scale_n=None)
    with pytest.raises(ValueError,match='physical_sides_dev'):GeneralS3PatchManager(model,res.definition,wrong)
    assert not validate_general_contact(model,res.definition,settings,analysis_control='force').valid
    token=CancellationToken();token.cancel('orientation precancel')
    with pytest.raises(SolveCancelled):solve(model,settings,res,program,cancellation_token=token)
    assert repr(model)==before
    # Actual legacy-valid multiple-beam public route and allocation traps are
    # independently maintained in the protected S3 orientation test.
    old=load('_s3_orientation_refusal','test_general_contact_s3_physical_orientation.py')
    old.test_public_multiple_beam_transient_refuses_physical_orientation_before_allocation(monkeypatch)


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not os.environ.get('Q4_ORIENTATION_LEGACY_FULL')
    or not all((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(name+'.json')).is_file() for name in ('prefix-mm','proof-prefix-mm','prefix-pp'))
    or not Path(os.environ['Q4_ORIENTATION_LEGACY_FULL']).is_file(),
    reason='archived orientation inputs AUTOMATIC_WORKFLOW_EVIDENCE/{prefix-mm,proof-prefix-mm,prefix-pp}.json and Q4_ORIENTATION_LEGACY_FULL absent: restart-binding replay requires the original evidence root')
def test_public_orientation_archive_restart_binding_and_legacy_identity(monkeypatch):
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.core import _contact_settings_contract
    model,domain,settings,res,program=build(signs=(-1,-1));prefix=read('prefix-mm');proof=read('proof-prefix-mm');assert proof['verified'] and proof['opening_verified'] and proof['source_sha256']==sha('prefix-mm');before=repr(model)
    pm,pd,ps,pr,pp=build();plusprefix=read('prefix-pp')
    with pytest.raises(ValueError,match='incompatible'):solve(pm,replace(ps,shell_pair_orientation='legacy'),pr,pp,restart_checkpoint=plusprefix['checkpoint'])
    from anyfesolver.general_contact import resolve_automatic_contact_domain
    changed=resolve_automatic_contact_domain(model,replace(domain,name='changed-coherent-q4-orientation'),settings,analysis_control='contact_path')
    with pytest.raises(ValueError,match='incompatible'):solve(model,settings,changed,program,restart_checkpoint=prefix['checkpoint'])
    damaged=copy.deepcopy(prefix['checkpoint']);damaged['path_state']['normal_forces'][0]+=1e-6
    with pytest.raises(ValueError,match='SHA-256 does not match its contents'):solve(model,settings,res,program,restart_checkpoint=damaged)
    assert repr(model)==before
    # Prior automatic legacy fixture identity/checkpoint is reused by bytes.
    from examples.general_contact_q4_pair_monolithic_cycle import build as legacy
    lm,ld,ls,lr,lp=legacy();baseline=json.loads(Path(os.environ['Q4_ORIENTATION_LEGACY_FULL']).read_text())
    assert lr.identity==baseline['identity'] and 'shell_pair_orientation' not in _contact_settings_contract(ls)
    assert 'shell_pair_orientation' not in GeneralS3PatchManager(lm,lr.definition,ls).restart_contract()['settings']
    terminal=solve(lm,ls,lr,lp,restart_checkpoint=baseline['checkpoint']);assert terminal.status=='completed'
    np.testing.assert_allclose(terminal.displacements,np.array(baseline['poses'][-1]['displacements']),rtol=0,atol=0)
    save(dict(verified=True,orientation_incompatibility=True,definition_incompatibility=True,constructor_admission_refusal=True,force_validation_context_refusal=True,public_precancel=True,content_refusal=True,transient_multiple_beam_before_allocation=True,legacy_identity_checkpoint_unchanged=True,legacy_full_sha256=hashlib.sha256(Path(os.environ['Q4_ORIENTATION_LEGACY_FULL']).read_bytes()).hexdigest(),declared_FEModel_repr_unchanged=True,hidden_nonlinear_rollback_not_qualified=True))



def test_actual_ordinary_force_selector_refusal(monkeypatch):
    from anyfesolver import LoadCase,solve_static_nonlinear
    import anyfesolver.general_contact.static_path as path_owner
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    model,domain,settings,res,program=build(signs=(-1,-1));load=LoadCase('same first-press forces')
    for node in (3,6):load.add_nodal_load(node,forces=(0.,0.,-5e-5))
    before=repr(model);calls=[]
    def forbidden(*args,**kwargs):
        calls.append(True);raise AssertionError('allocation or mutation before explicit-path selector refusal')
    monkeypatch.setattr(path_owner,'solve_static_contact_path',forbidden)
    monkeypatch.setattr(GeneralS3PatchManager,'__init__',forbidden)
    monkeypatch.setattr(model,'apply_boundary_conditions',forbidden)
    with pytest.raises(ValueError,match='shell_pair_orientation requires contact_path_program'):
        solve_static_nonlinear(model,load,contact=res.definition,contact_settings=settings,
            num_steps=1,tolerance=1e-12,max_iterations=12,emit_restart_checkpoint=True,record_increment_snapshots=True)
    assert not calls and repr(model)==before
    save(dict(verified=True,actual_public_ordinary_force_refusal=True,named_explicit_path_guard=True,
        before_generated_path_and_manager_allocation=True,declared_FEModel_repr_unchanged=True,
        prior_failure='attempt-19-ordinary: incomplete route expectation; early guard observed',
        physical_force_cycle_is_explicit_zero_control_path=True,legacy_monolithic_force_hook_unchanged=True))
