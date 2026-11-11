"""Bounded coupled Q4/Q4 equations, independent reference and actual force route."""
import copy,hashlib,itertools,json,os
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_q4_pair_monolithic import build,solve
import importlib.util
_spec=importlib.util.spec_from_file_location('_bending_reference',Path(__file__).with_name('test_general_contact_q4_pair_bending.py'))
_reference=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_reference)
skin,free_dofs,exact_pose_gap,independent_linear_contact=(_reference.skin,_reference.free_dofs,_reference.exact_pose_gap,_reference.independent_linear_contact)

FREF=1e-4
MREF=1e-4


def save(data):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else x),encoding='utf-8')


def hard_reference(model,load):
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    K=assemble_stiffness_matrix(model)[0].toarray();free=free_dofs(model);kf=K[np.ix_(free,free)]
    s,g,g0=independent_linear_contact(model);gf=g[:,free];f=load.get_load_vector(model.mesh,model.mesh.dof_manager)
    A=gf@np.linalg.solve(kf,gf.T);q=g0+gf@np.linalg.solve(kf,f[free]);solutions=[]
    for bits in itertools.product((False,True),repeat=6):
        active=np.array(bits);p=np.zeros(6)
        if active.any():p[active]=np.linalg.solve(A[np.ix_(active,active)],-q[active])
        w=A@p+q
        if min(p)>=-1e-12 and min(w)>=-1e-14 and max(np.abs(p*w))<=1e-14:solutions.append(p)
    assert solutions;pressure=solutions[0];u=np.zeros(len(f));u[free]=np.linalg.solve(kf,f[free]+gf.T@pressure)
    error=np.linalg.norm((K@u-g.T@pressure-f)[free])/FREF;assert error<=1e-6
    eig=np.linalg.eigvalsh(A/np.sqrt(s[:,None]*s[None,:]));assert eig[0]>1e-10*eig[-1]
    return dict(support=s,g=g,pressure=pressure,forces=s*pressure,displacements=u,equilibrium_error=error,compliance_eigenvalues=eig,structuralK_reused=True,penalty_diagonal=False)


def test_independent_reference_and_full_coupled_directional_jacobian():
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager,REFINED_Q4_Q4_MONOLITHIC_AUGMENTED_PROFILE
    from anyfesolver.general_contact.semismooth import linearize,complementarity,reference_nodal_support
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    model,domain,settings,res,load=build();reference=hard_reference(model,load);manager=GeneralS3PatchManager(model,res.definition,settings)
    assert manager.profile==REFINED_Q4_Q4_MONOLITHIC_AUGMENTED_PROFILE
    zero=np.zeros(manager.n_dof);raw0=linearize(manager,zero,np.zeros(6),tangent=True)
    assert tuple(manager.law.pressure_mode_ids)==(1,2,3,4,5,6)
    operator_error=np.max(np.abs(raw0.gap_gradient*reference['support'][:,None]-reference['g']));assert operator_error<=1e-6
    rng=np.random.default_rng(61);u=rng.normal(size=manager.n_dof)*1e-7;u[np.setdiff1d(np.arange(manager.n_dof),free_dofs(model))]=0.
    z=.01*reference_nodal_support(manager.law);c=1000*reference_nodal_support(manager.law)
    raw=linearize(manager,u,z);ncp,active,cq,cz=complementarity(raw.state,z,c,raw.gap_gradient)
    du=rng.normal(size=manager.n_dof);du/=np.linalg.norm(du);dz=rng.normal(size=6)*.01
    def structural(at,tangent):
        r,k,states=_assemble_nonlinear_system(model,at,{},5,tangent=tangent,kinematics='von_karman');_discard_nonlinear_state_candidate(states);return r,k
    rs,ks=structural(u,True);errors=[]
    for h in (1e-8,3e-9):
        plus=linearize(manager,u+h*du,z+h*dz,tangent=False);minus=linearize(manager,u-h*du,z-h*dz,tangent=False)
        cp,ap,_,_=complementarity(plus.state,z+h*dz,c,plus.gap_gradient,tangent=False)
        cm,am,_,_=complementarity(minus.state,z-h*dz,c,minus.gap_gradient,tangent=False)
        assert np.array_equal(ap,active) and np.array_equal(am,active)
        cf=(plus.force-minus.force)/(2*h);ce=raw.tangent@du-raw.gap_gradient.T@dz
        full=(structural(u+h*du,False)[0]+plus.force-structural(u-h*du,False)[0]-minus.force)/(2*h)
        expected=(ks+raw.tangent)@du-raw.gap_gradient.T@dz
        dc=(cp-cm)/(2*h);jc=cq@du+cz*dz
        item=dict(h=h,contact=np.linalg.norm(cf-ce)/max(np.linalg.norm(ce),1e-12),coupled_equilibrium=np.linalg.norm(full-expected)/max(np.linalg.norm(expected),1e-12),complementarity=np.linalg.norm(dc-jc)/max(np.linalg.norm(jc),1e-12))
        assert max(item[k] for k in ('contact','coupled_equilibrium','complementarity'))<=1e-5;errors.append(item)
    points,mids,rot=skin(model,u);np.testing.assert_allclose(points,np.concatenate(manager._positions(u)),atol=1e-13,rtol=0)
    expected=np.zeros(manager.n_dof)
    for n,row in enumerate(raw.local_force.reshape(-1,3),1):
        d=model.mesh.dof_manager.get_node_dofs(n);expected[d[:3]]=row;expected[d[3:]]=(-.025 if n<=6 else .025)*np.cross([0.,0.,1.],row)
    assert np.max(np.abs(expected-raw.force))<=1e-13
    # Explicit selected generalized derivative at a kink and its active one-sided ray.
    kink=copy.copy(raw0.state);from dataclasses import replace as dc_replace
    kink=dc_replace(kink,normalized_gap=np.zeros(6));zz=np.zeros(6)
    C,a,qj,zj=complementarity(kink,zz,c,raw0.gap_gradient);assert not a.any() and np.array_equal(zj,np.ones(6))
    ray=-np.ones(6)*1e-8;kink_right=dc_replace(kink,normalized_gap=ray)
    right=complementarity(kink_right,zz,c,raw0.gap_gradient,tangent=False)[0];np.testing.assert_allclose(right,c*ray,atol=1e-20,rtol=0)
    save(dict(verified=True,operator_error=operator_error,hard_contact_reference=reference,full_coupled_errors=errors,positive_support=raw.state.reference_support,pressure_mode_ids=manager.law.pressure_mode_ids,kink_one_sided_checked=True))


@pytest.mark.skipif(not os.environ.get('MONOLITHIC_STAGE'),
    reason='installed monolithic stage input MONOLITHIC_STAGE absent: staged monolithic solve requires the original workflow stage selection')
def test_installed_force_bending_stage():
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.semismooth import linearize,complementarity,reference_nodal_support
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    stage=os.environ['MONOLITHIC_STAGE'];evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']);kwargs={}
    model,domain,settings,res,load=build(regularization=2000. if stage=='alternate-c' else 1000.)
    if stage=='prefix':kwargs['max_load_factor']=.5
    if stage=='resume':
        prefix=json.loads((evidence/'prefix.json').read_text());assert prefix['verified'];kwargs['restart_checkpoint']=prefix['checkpoint']
    result=solve(model,settings,res,load,**kwargs);report=dict(stage=stage,status=result.status,failure_reason=result.failure_reason,identity=res.identity,info=result.info,displacements=result.displacements,checkpoint=result.restart_checkpoint,snapshot_inventory=[s.step_index for s in result.snapshots]);save(report)
    assert result.status=='completed',(result.failure_reason,result.info.get('contact_failure'))
    expected_inventory={'full':[1,2],'alternate-c':[1,2],'prefix':[1],'resume':[2]};assert report['snapshot_inventory']==expected_inventory[stage]
    u=result.displacements;z=np.array(result.restart_checkpoint['path_state']['normal_forces']);manager=GeneralS3PatchManager(model,res.definition,settings);raw=linearize(manager,u,z)
    C,active,_,_=complementarity(raw.state,z,manager.stiffness*reference_nodal_support(manager.law),raw.gap_gradient)
    comp=np.sum(np.abs(C))/FREF;assert comp<=1e-8 and np.min(z)>=0 and active.any()
    structural,_,states=_assemble_nonlinear_system(model,u,result.element_states,5,tangent=False,kinematics='von_karman');_discard_nonlinear_state_candidate(states)
    scale=.5 if stage=='prefix' else 1.;external=load.get_load_vector(model.mesh,model.mesh.dof_manager)*scale
    error=np.linalg.norm((structural+raw.force-external)[free_dofs(model)])/FREF;assert error<=1e-6
    points,mids,rotation=skin(model,u);actual=np.concatenate(manager._positions(u));np.testing.assert_allclose(points,actual,atol=1e-13,rtol=0);gap=exact_pose_gap(actual)
    uz=np.array([u[model.mesh.dof_manager.get_node_dofs(n)[2]] for n in range(1,16)]);theta=np.array([u[model.mesh.dof_manager.get_node_dofs(n)[3:]] for n in range(1,16)])
    motion=dict(upper_uz_range_m=float(np.ptp(uz[:6])),lower_uz_norm_m=float(np.linalg.norm(uz[6:])),rotation_norms_rad=[float(np.linalg.norm(theta[:6])),float(np.linalg.norm(theta[6:]))],rotational_skin_displacement_m=float(np.linalg.norm(rotation)))
    assert motion['upper_uz_range_m']>1e-9 and motion['lower_uz_norm_m']>1e-10 and min(motion['rotation_norms_rad'])>1e-9 and motion['rotational_skin_displacement_m']>1e-11
    physical=-raw.local_force.reshape(-1,3);pf=np.linalg.norm(physical.sum(axis=0))/FREF;pm=np.linalg.norm(np.cross(actual,physical).sum(axis=0))/MREF
    sm=np.cross(mids,physical).sum(axis=0)+sum((-.025 if n<=6 else .025)*np.cross([0.,0.,1.],physical[n-1]) for n in range(1,16));se=np.linalg.norm(sm)/MREF
    assert max(pf,pm,se)<=1e-6
    assert tuple(manager.law.pressure_mode_ids)==(1,2,3,4,5,6) and np.min(raw.state.reference_support)>0
    reference=hard_reference(model,load)
    report.update(normal_forces=z,pressure=raw.state.pressure,support=raw.state.reference_support,normalized_gaps=raw.state.normalized_gap,complementarity_error=comp,equilibrium_error=error,gap=gap,motion=motion,wrenches=dict(physical_force=pf,physical_moment=pm,structural_coordinate_moment=se,second_order_skin_deficit=np.cross(rotation,physical).sum(axis=0)),nonlinear_vs_linear_relative=float(np.linalg.norm(u-reference['displacements']*scale)/np.linalg.norm(reference['displacements']*scale)),external_work_j=float(external@u),both_body_instantaneous_contact_work_j=[float(raw.force[[model.mesh.dof_manager.get_node_dofs(n)[k] for n in ns for k in range(6)]]@u[[model.mesh.dof_manager.get_node_dofs(n)[k] for n in ns for k in range(6)]]) for ns in (range(1,7),range(7,16))])
    if stage in ('resume','alternate-c'):
        fullpath=evidence/'full.json';full=json.loads(fullpath.read_text());assert full['verified']
        report['comparison_full_sha256']=hashlib.sha256(fullpath.read_bytes()).hexdigest();report['displacement_max_difference_m']=float(np.max(np.abs(u-np.array(full['displacements']))));report['normal_force_max_difference_n']=float(np.max(np.abs(z-np.array(full['normal_forces']))))
        assert report['displacement_max_difference_m']<=1e-9 and report['normal_force_max_difference_n']/FREF<=1e-6
    report['verified']=True;save(report)


def test_actual_restart_enforcement_and_route_refusal(monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled,solve_static_nonlinear,solve_static_arc_length,ArcLengthControl,validate_general_contact
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    import anyfesolver.general_contact.static_path as path_owner
    model,domain,settings,res,load=build();before=repr(model)
    token=CancellationToken();token.cancel('monolithic public precancel')
    with pytest.raises(SolveCancelled):solve(model,settings,res,load,cancellation_token=token)
    with pytest.raises(ValueError,match='friction requires an explicit contact path program'):solve(model,settings,res,load,contact_friction=object())
    with pytest.raises(ValueError,match='default structural options'):solve(model,settings,res,load,kinematics='corotational')
    with pytest.raises(ValueError,match='jet2_q4_q4_dev'):
        solve_static_arc_length(model,load,control=ArcLengthControl(initial_load_increment=1e-5,minimum_load_increment=1e-6,maximum_load_increment=1e-5,max_steps=1),contact=res.definition,contact_settings=settings)
    assert repr(model)==before
    # Direct manager admission must also reject a warped reference face.
    warped=build();warped[0].mesh.nodes[11].z+=1e-3
    with pytest.raises(ValueError,match='planar'):GeneralS3PatchManager(warped[0],warped[3].definition,warped[2])
    interaction=replace(res.definition.interactions[0],target=res.definition.interactions[0].surface)
    self_pair=replace(res.definition,interactions=(interaction,));assert not validate_general_contact(model,self_pair,settings).valid
    committed=[]
    def forbidden(*args,**kwargs):committed.append(True);raise AssertionError('physical candidate committed in rejected iteration')
    monkeypatch.setattr(path_owner,'_commit_nonlinear_state_candidate',forbidden)
    rejected=solve_static_nonlinear(model,load,contact=res.definition,contact_settings=settings,num_steps=2,max_iterations=1,tolerance=1e-12,emit_restart_checkpoint=True)
    assert rejected.status!='completed' and not committed and np.array_equal(rejected.displacements,np.zeros_like(rejected.displacements))
    assert not rejected.steps
    if rejected.restart_checkpoint is not None:assert rejected.restart_checkpoint['path_state']['accepted_step']==0
    assert repr(model)==before
    from examples.general_contact_multiple_beam_s3_transient import build as transient_build
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.general_contact.transient import solve_transient_contact,_admit,ContactTransientSolverControls
    import anyfesolver.general_contact.multiple_beam_s3_transient as transient_owner
    tm,td,ts,velocity=transient_build();controls=ContactTransientSolverControls(allow_multiple_deformable_beam_s3_dev=True);assert _admit(tm,td,ts,controls)==transient_owner.PROFILE
    prior=repr(tm);allocation=[]
    def allocate(*args,**kwargs):allocation.append(True);raise AssertionError('transient allocation before selector refusal')
    monkeypatch.setattr(transient_owner,'pair_definitions',allocate);monkeypatch.setattr(transient_owner.MultipleBeamS3TransientManager,'__init__',allocate);monkeypatch.setattr(tm,'apply_boundary_conditions',allocate)
    with pytest.raises(ValueError,match='jet2_q4_q4_dev'):
        solve_transient_contact(tm,td,TransientConfig(dt=1e-5,t_end=1e-5,initial_velocity=velocity),contact_settings=replace(ts,general_s3_differentiation='jet2_q4_q4_dev'),solver_controls=controls)
    assert not allocation and repr(tm)==prior
    save(dict(verified=True,public_precancel=True,friction_refusal=True,finite_rotation_route_refusal=True,warped_reference_refusal=True,selfcontact_refusal=True,arc_refusal=True,legacy_valid_multiple_beam_public_transient_refusal_before_allocation=True,rejected_iteration_no_accepted_increment=True,rejected_iteration_no_accepted_candidate_commit=True,rejected_status=rejected.status,rejected_reason=rejected.failure_reason,declared_FEModel_repr_unchanged=True,transaction_scope='elastic fixture, declared FEModel fields and accepted-candidate commit trap; generic hidden material-trial rollback remains open'))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'prefix.json').is_file(),
    reason='archived monolithic input AUTOMATIC_WORKFLOW_EVIDENCE/prefix.json absent: restart-enforcement replay requires the original staged checkpoint')
def test_actual_archive_checkpoint_restart_enforcement():
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.general_contact.automatic_domain import resolve_automatic_contact_domain
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']);prefix=json.loads((evidence/'prefix.json').read_text());assert prefix['verified']
    model,domain,settings,res,load=build();before=repr(model)
    changed=replace(domain,name='monolithic-changed-provenance');changed_res=resolve_automatic_contact_domain(model,changed,settings,analysis_control='force')
    with pytest.raises(ValueError,match='incompatible'):solve(model,settings,changed_res,load,restart_checkpoint=prefix['checkpoint'])
    assert repr(model)==before
    # The admitted historical penalty route is a different enforcement contract.
    penalty_domain=replace(domain,defaults=replace(domain.defaults,normal_law='projected_penalty'))
    penalty_settings=replace(settings,enforcement_solver='nested_augmented',complementarity_force_scale_n=None)
    penalty_res=resolve_automatic_contact_domain(model,penalty_domain,penalty_settings,analysis_control='force')
    with pytest.raises(ValueError,match='incompatible'):
        solve_static_nonlinear(model,load,contact=penalty_res.definition,contact_settings=penalty_settings,tolerance=1e-12,num_steps=2,restart_checkpoint=prefix['checkpoint'])
    assert repr(model)==before
    damaged=copy.deepcopy(prefix['checkpoint']);damaged['path_state']['normal_forces'][0]+=1e-6
    with pytest.raises(ValueError,match='SHA-256 does not match its contents'):solve(model,settings,res,load,restart_checkpoint=damaged)
    assert repr(model)==before
    save(dict(verified=True,provenance_refusal=True,enforcement_checkpoint_refusal=True,content_hash_refusal=True,prefix_sha256=hashlib.sha256((evidence/'prefix.json').read_bytes()).hexdigest()))
