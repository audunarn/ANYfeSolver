"""Independent seed, actual installed tilted relative path and physical gates."""
import copy,hashlib,json,math,os,importlib.util
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_tilted_relative_sliding import build,solve,rigid_pose,rotation
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize,complementarity,reference_nodal_support
spec=importlib.util.spec_from_file_location('_tilted_finite_proof',Path(__file__).with_name('test_general_contact_finite_rotation_shell_work.py'))
_finite_proof=importlib.util.module_from_spec(spec);spec.loader.exec_module(_finite_proof)
decimal_director,save,read,sha=(_finite_proof.decimal_director,_finite_proof.save,_finite_proof.read,_finite_proof.sha)

FREF=1e-4;MREF=1e-4

@pytest.fixture(scope='module')
def public_record():
    """Only the new bound installed producer, never fabricated mechanics."""
    record_path=os.environ.get('TILTED_PUBLIC_RECORD')
    if not record_path:pytest.skip('archived public record TILTED_PUBLIC_RECORD absent: saved-assessment replay requires the original public record witness')
    return json.loads(Path(record_path).read_text())

@pytest.fixture(scope='module')
def public_fixture():
    return build()

@pytest.mark.parametrize('value',[None,{}, {'schema':'TILTED_RELATIVE_Q4_PUBLIC_OUTPUT_V1','checkpoint':None}, {'schema':'unknown','checkpoint':None}])
def test_public_restart_refusal_before_solve(tmp_path,monkeypatch,value):
    import examples.general_contact_tilted_relative_sliding as entry
    calls=[]
    monkeypatch.setattr(entry,'solve',lambda *a,**k:calls.append(True))
    path=tmp_path/'restart.json';path.write_text(json.dumps(value));out=tmp_path/'refusal.json'
    assert entry.main(['--restart',str(path),'--output',str(out)])==1
    assert not calls and json.loads(out.read_text())['status']=='NONPASS'

def test_public_saved_assessment_actual(public_record,public_fixture):
    from examples.general_contact_tilted_relative_output import assess
    model,domain,settings,res,program=public_fixture
    result=assess(public_record,model,settings,res,program)
    assert result['status']=='PASS' and result['complete_fresh_workflow'] and result['assessed_poses']==list(range(1,17))
    assert result['producer_reported_equilibrium_gate_verified'] is False

@pytest.mark.parametrize('mutation',['duplicate_history','wrong_snapshot','missing_state','state_nan','finite_terminal_state','wrong_pressure','wrong_program','wrong_pose','null_checkpoint'])
def test_public_saved_refusals(public_record,public_fixture,mutation):
    from examples.general_contact_tilted_relative_output import assess
    value=copy.deepcopy(public_record);model,domain,settings,res,program=public_fixture
    if mutation=='duplicate_history':value['history'].append(value['history'][0])
    elif mutation=='wrong_snapshot':value['poses'][0]['snapshot']['displacements'][0]+=.001
    elif mutation=='missing_state':value['poses'][0]['element_state_records'].pop()
    elif mutation=='state_nan':value['poses'][0]['element_state_records'][0]['state']['test_nonfinite']=float('nan')
    elif mutation=='finite_terminal_state':value['poses'][-1]['element_state_records'][0]['state']['alpha'][0]+=.001
    elif mutation=='wrong_pressure':value['poses'][0]['z'][0]=-1.
    elif mutation=='wrong_program':value['fixture']['program']['waypoints'][-1][0]+=.001
    elif mutation=='wrong_pose':value['poses'][0]['u'][0]+=.001
    elif mutation=='null_checkpoint':value['checkpoint']=None
    with pytest.raises((ValueError,AssertionError,KeyError)):assess(value,model,settings,res,program)

@pytest.mark.parametrize('mutation',['wrong_pressure','duplicate_history','missing_state'])
def test_public_rich_prefix_refusal_before_advancement(tmp_path,monkeypatch,public_record,mutation):
    import examples.general_contact_tilted_relative_sliding as entry
    value=copy.deepcopy(public_record)
    if mutation=='wrong_pressure':value['poses'][0]['z'][0]=-1.
    elif mutation=='duplicate_history':value['history'].append(value['history'][0])
    else:value['poses'][0]['element_state_records'].pop()
    path=tmp_path/'prefix.json';path.write_text(json.dumps(value));calls=[]
    monkeypatch.setattr(entry,'solve',lambda *a,**k:calls.append(True))
    assert entry.main(['--through-step','16','--restart',str(path)])==1
    assert not calls

@pytest.mark.skipif(not all(os.environ.get(name) for name in ('TILTED_CURRENT_OPEN_CHECKPOINT','TILTED_PUBLIC_SUFFIX','TILTED_LEGACY_REFUSAL')),
    reason='archived tilted inputs TILTED_CURRENT_OPEN_CHECKPOINT/TILTED_PUBLIC_SUFFIX/TILTED_LEGACY_REFUSAL absent: bare-checkpoint and suffix replay requires the original public records')
def test_public_bare_checkpoint_schema_and_suffix_truth(public_record):
    from examples.general_contact_tilted_relative_output import load_restart,assess
    cp,prefix,source=load_restart(os.environ['TILTED_CURRENT_OPEN_CHECKPOINT'],16)
    assert cp['path_state']['accepted_step']==15 and not prefix and source['bare_checkpoint']
    saved=json.loads(Path(os.environ['TILTED_PUBLIC_SUFFIX']).read_text())
    assert saved['status']=='completed' and saved['accepted_steps']==[16] and [p['step'] for p in saved['poses']]==[16]
    assert saved['restart_source']['bare_checkpoint'] and not saved['terminal_restart']
    np.testing.assert_allclose(saved['checkpoint']['displacements'],public_record['checkpoint']['displacements'],rtol=1e-8,atol=1e-12)
    np.testing.assert_allclose(saved['checkpoint']['path_state']['normal_forces'],public_record['checkpoint']['path_state']['normal_forces'],rtol=1e-7,atol=1e-12)
    model,domain,settings,res,program=build()
    with pytest.raises(ValueError,match='complete saved pose inventory'):assess(saved,model,settings,res,program)
    # Retain the actual historical analysis-contract refusal as evidence; do not
    # rerun its already consumed failing numerical attempt.
    legacy=json.loads(Path(os.environ['TILTED_LEGACY_REFUSAL']).read_text())
    assert legacy['status']=='NONPASS' and 'analysis contract is incompatible' in legacy['failure_reason']

@pytest.mark.skipif(not os.environ.get('TILTED_CURRENT_PREFIX'),
    reason='archived tilted input TILTED_CURRENT_PREFIX absent: rich terminal-restore replay requires the original prefix record')
def test_public_rich_terminal_restore(tmp_path,public_record):
    from examples.general_contact_tilted_relative_sliding import main
    path=tmp_path/'fresh.json';path.write_text(json.dumps(public_record));out=tmp_path/'terminal.json'
    assert main(['--through-step','16','--restart',str(path),'--output',str(out)])==0
    saved=json.loads(out.read_text());assert saved['terminal_restart'] and saved['accepted_steps']==[] and len(saved['poses'])==16
    np.testing.assert_array_equal(saved['checkpoint']['displacements'],public_record['checkpoint']['displacements'])
    # A rich genuine prefix supplies earlier poses; this is a real increment,
    # separately from the zero-new-increment terminal restore above.
    source=Path(os.environ['TILTED_CURRENT_PREFIX']);original=source.read_bytes();rich=tmp_path/'rich-suffix.json'
    assert main(['--through-step','16','--restart',str(source),'--output',str(rich)])==0
    result=json.loads(rich.read_text());assert result['accepted_steps']==[16] and len(result['poses'])==16 and not result['terminal_restart']
    np.testing.assert_allclose(result['checkpoint']['displacements'],public_record['checkpoint']['displacements'],rtol=1e-8,atol=1e-12)
    np.testing.assert_allclose(result['checkpoint']['path_state']['normal_forces'],public_record['checkpoint']['path_state']['normal_forces'],rtol=1e-7,atol=1e-12)
    assert result['poses'][:15]==json.loads(original)['poses'] and source.read_bytes()==original
    from examples.general_contact_tilted_relative_output import assess
    model,domain,settings,res,program=build();assessment=assess(result,model,settings,res,program)
    assert assessment['status']=='PASS' and not assessment['complete_fresh_workflow']

def test_public_serialization_failure_preserves_returned_pose(tmp_path,monkeypatch,public_record):
    from types import SimpleNamespace
    import examples.general_contact_tilted_relative_sliding as entry
    import examples.general_contact_tilted_relative_output as output
    pose=public_record['poses'][0]
    snapshot=SimpleNamespace(step_index=1,displacements=np.asarray(pose['u']),element_states={},to_dict=lambda **k:pose['snapshot'])
    result=SimpleNamespace(info={'force_displacement_history':[pose['record']]},snapshots=[snapshot],restart_checkpoint=None,status='failed',failure_reason='test returned prefix')
    monkeypatch.setattr(entry,'solve',lambda *a,**k:result)
    def refuse(*a,**k):raise ValueError('test serializer failure')
    monkeypatch.setattr(output,'_state_records',refuse)
    path=tmp_path/'partial.json'
    assert entry.main(['--output',str(path)])==1
    saved=json.loads(path.read_text());assert saved['status']=='NONPASS' and saved['accepted_steps']==[1] and saved['serialization_failures']
    np.testing.assert_array_equal(saved['poses'][0]['u'],pose['u'])
    assert saved['checkpoint'] is None and 'element_state_records' not in saved['poses'][0]

def rows_for(model,program):
    rows=np.zeros((len(program.controls),model.mesh.dof_manager.total_dofs));names=dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5)
    for i,c in enumerate(program.controls):
        for n,name,w in c.terms:rows[i,model.mesh.dof_manager.get_node_dofs(n)[names[name]]]+=w
    return rows


def test_independent_partition_seed_and_current_work_tangent(monkeypatch):
    from scipy.sparse import eye
    from anyfesolver.general_contact.finite_rotation_predictor import per_surface_seed_descriptor,per_surface_rigid_seed
    from anyfesolver.general_contact.finite_rotation_state import owner_local_coordinates
    model,domain,settings,res,program=build();manager=GeneralS3PatchManager(model,res.definition,settings)
    desc=per_surface_seed_descriptor(model,res.definition,program);n=manager.n_dof;rows=rows_for(model,program)
    q=rigid_pose(model,math.pi/3.)
    q[model.mesh.dof_manager.get_node_dofs(3)[3]]+=.002
    q[model.mesh.dof_manager.get_node_dofs(8)[2]]-=3e-7
    seed,record=per_surface_rigid_seed(model,program,program.target_at(14),eye(n,format='csr'),np.zeros(n),q,rows,desc)
    before=manager.finite_skin.pose(q);after=manager.finite_skin.pose(seed)
    assert np.max(np.abs(after.positions[:6]-before.positions[:6]-np.array((0.,.02,0.))))<=1e-13
    np.testing.assert_array_equal(after.positions[6:],before.positions[6:])
    np.testing.assert_array_equal(seed[manager.finite_skin.dofs[:,3:]],q[manager.finite_skin.dofs[:,3:]])
    errors=[]
    for element in model.mesh.elements.values():
        dofs=element.get_dof_mapping(model.mesh)
        e=np.max(np.abs(owner_local_coordinates(model,element,seed[dofs])-owner_local_coordinates(model,element,q[dofs])))
        errors.append(float(e));assert e<=1e-13
    assert manager._swept_admissible(q,seed)
    # Reuse the independent full spin-work/q-z directional discriminator at the
    # NEW partially overlapping tilted geometry, with current settings/profile.
    proof=_finite_proof
    def posed(model,angle):
        u=rigid_pose(model,angle)
        for nid in range(1,7):u[model.mesh.dof_manager.get_node_dofs(nid)[1]]+=.02
        return u
    with monkeypatch.context() as context:
        context.setattr(proof,'build',build);context.setattr(proof,'rigid_pose',posed)
        proof.test_director_work_and_coupled_chart_jacobian()
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:
        result=json.loads(Path(result_root).read_text())
        result.update(seed=record,partitions=list(desc),target_physical_skin_unchanged=True,source_only_skin_translation_m=.02,owner_local_coordinate_preservation_errors=errors,actual_seed_path_certified=True)
        save(result)


@pytest.mark.skipif(not os.environ.get('FINITE_STAGE'),
    reason='installed tilted stage input FINITE_STAGE absent: staged tilted solve requires the original workflow stage selection')
def test_installed_relative_path():
    stage=os.environ['FINITE_STAGE'];step=int(stage.split('-')[1]);previous={4:None,8:4,12:8,13:12,14:13,15:14,16:15}[step]
    model,domain,settings,res,program=build();kwargs=dict(max_load_factor=step/16.)
    if previous is not None:kwargs['restart_checkpoint']=read('run-'+str(previous))['checkpoint']
    result=solve(model,settings,res,program,**kwargs);history={r['step_index']:r for r in result.info.get('force_displacement_history',[])};poses=[]
    for snapshot in result.snapshots:
        row=history[snapshot.step_index];contact=row['contact'];z=np.asarray(contact['numerical_pressure_pa'])*np.asarray(contact['reference_support_m2'])
        if result.restart_checkpoint is not None and snapshot.step_index==result.restart_checkpoint['path_state']['accepted_step']:z=np.asarray(result.restart_checkpoint['path_state']['normal_forces'])
        poses.append(dict(step=snapshot.step_index,u=snapshot.displacements,z=z,record=row))
    save(dict(status=result.status,failure_reason=result.failure_reason,identity=res.identity,checkpoint=result.restart_checkpoint,info=result.info,poses=poses,snapshot_inventory=[p['step'] for p in poses],previous_sha256=sha('run-'+str(previous)) if previous else None,program=program.contract()))
    assert result.status=='completed',result.failure_reason
    assert [p['step'] for p in poses]==list(range((previous or 0)+1,step+1))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived tilted inputs AUTOMATIC_WORKFLOW_EVIDENCE run-4..run-16.json absent: saved-pose physical-gate replay requires the original staged evidence root')
def test_saved_relative_poses_independent_physical_gates():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    spec=importlib.util.spec_from_file_location('_tilted_pose_oracle',Path(__file__).parents[1]/'docs/d1_review_support/contact_tilted_relative_sliding_20261008/oracle.py')
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    stages=['run-'+str(i) for i in (4,8,12,13,14,15,16)];chunks=[read(s) for s in stages]
    assert all(c['status']=='completed' for c in chunks)
    assert all(c['previous_sha256']==sha(stages[i-1]) for i,c in enumerate(chunks) if i)
    poses=[p for c in chunks for p in c['poses']];assert [p['step'] for p in poses]==list(range(1,17))
    model,domain,settings,res,program=build();manager=GeneralS3PatchManager(model,res.definition,settings);C=rows_for(model,program)
    fixed=[model.mesh.dof_manager.get_node_dofs(9)[1]];free=np.setdiff1d(np.arange(manager.n_dof),fixed)
    names=dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5);reports=[]
    for pose in poses:
        step=pose['step'];q=np.array(pose['u']);z=np.array(pose['z']);manager.require_finite_deformation(q)
        skin=manager.finite_skin.pose(q);independent=[]
        for i,dofs in enumerate(manager.finite_skin.dofs):
            independent.append(manager.finite_skin.reference[i]+q[dofs[:3]]+manager.finite_skin.offsets[i]*decimal_director(q[dofs[3:]],manager.finite_skin.directors[i]))
        agreement=float(np.max(np.abs(skin.positions-independent)));assert agreement<=1e-13
        angle=min(step,12)*math.pi/36.;axis=rotation(angle)@np.array((0.,0.,1.));ax=rotation(angle)@np.array((1.,0.,0.));ay=np.array((0.,1.,0.))
        gap=oracle.posed_bounds(skin.positions,axis);footprint=oracle.projected_footprint(skin.positions,ax,ay)
        raw=linearize(manager,q,z);c=1000*reference_nodal_support(manager.law);ncp,active,_,_=complementarity(raw.state,z,c,raw.gap_gradient,tangent=False)
        ext=np.zeros(manager.n_dof)
        for force,value in zip(program.nodal_forces,program.force_values_at(step)):
            ext[model.mesh.dof_manager.get_node_dofs(force.node_id)[names[force.dof]]]+=value
        r,k,trial=_assemble_nonlinear_system(model,q,{},5,tangent=False,kinematics='corotational',corotational_tangent='consistent',require_full_coordinates=True)
        _discard_nonlinear_state_candidate(trial)
        applied=ext+C.T@np.array(pose['record']['reaction_generalized'])
        stopping_scale=max(1.,float(np.linalg.norm(applied[free])))
        assert stopping_scale==1.,'bound-derived accuracy requires measured scale factor1'
        eq=float(np.linalg.norm((r+raw.force-applied)[free],1)/FREF)
        complement=float(np.linalg.norm(ncp,1)/FREF);assert eq<=1e-6 and complement<=1e-8 and min(z)>=0
        physical=-raw.local_force.reshape(-1,3);pf=float(np.linalg.norm(physical.sum(axis=0),1)/FREF);pm=float(np.linalg.norm(np.cross(skin.positions,physical).sum(axis=0),1)/MREF)
        mids=manager.finite_skin.reference+q[manager.finite_skin.dofs[:,:3]];nodal=raw.force[manager.finite_skin.dofs]
        sm=float(np.linalg.norm(np.cross(mids,-nodal[:,:3]).sum(axis=0)-nodal[:,3:].sum(axis=0),1)/MREF)
        assert max(pf,pm,sm)<=1e-6
        zero=step<=12 or step==15
        if zero:assert gap['strict_positive'] and np.linalg.norm(z,1)/FREF<=1e-8
        else:
            assert max(z)>0 and np.min(raw.state.reference_support)>0
            qrigid=rigid_pose(model,math.pi/3.)
            if step>=14:
                for n in range(1,7):qrigid[model.mesh.dof_manager.get_node_dofs(n)[1]]+=.02
            bending=q-qrigid
            upper=manager.finite_skin.dofs[:6];lower=manager.finite_skin.dofs[6:]
            assert np.max(np.abs(bending[upper[:,:3]]))>1e-10 and np.max(np.abs(bending[lower[:,:3]]))>1e-10
            assert np.max(np.abs(bending[upper[:,3:]]))>1e-10 and np.max(np.abs(bending[lower[:,3:]]))>1e-10
        work=[float(np.sum(raw.force[manager.finite_skin.dofs[:6,:3]]@axis)*1e-6),float(np.sum(raw.force[manager.finite_skin.dofs[6:,:3]]@axis)*1e-6)]
        assert abs(sum(work))/(FREF*1e-6)<=1e-6
        if not zero:assert min(abs(x) for x in work)>1e-12
        reports.append(dict(step=step,equilibrium_l1_relative=eq,solver_stopping_force_scale=stopping_scale,free_residual_coordinates=len(free),complementarity_l1_relative=complement,physical_force_relative=pf,physical_moment_relative=pm,structural_spatial_moment_relative=sm,independent_gap=gap,independent_footprint=footprint,independent_skin_max_error_m=agreement,local_deformation_maxima=manager.last_finite_deformation_maxima,pressure_ids=list(manager.law.pressure_mode_ids),support=raw.state.reference_support,z=z,both_body_virtual_translation_work_j=work,upper_skin_y=skin.positions[:6,1],upper_actual_uy=q[manager.finite_skin.dofs[:6,1]],target_anchor_coordinates=q[model.mesh.dof_manager.get_node_dofs(7)],reactions=pose['record']['reaction_generalized']))
    press=reports[12];slide=reports[13];opened=reports[14];recontact=reports[15]
    travel=np.array(slide['upper_skin_y'])-press['upper_skin_y'];assert min(travel)>.01
    actuator=np.array(slide['upper_actual_uy'])-press['upper_actual_uy'];assert max(abs(actuator[[0,3]]-.02))<=1e-12
    np.testing.assert_allclose(slide['target_anchor_coordinates'],press['target_anchor_coordinates'],rtol=0,atol=1e-12)
    area_change=abs(slide['independent_footprint']['total_area_m2']-press['independent_footprint']['total_area_m2']);assert area_change>1e-3
    material_seam_change=[abs(b['source_material_v']-a['source_material_v']) for a,b in zip(press['independent_footprint']['seam_material_coordinates'],slide['independent_footprint']['seam_material_coordinates'])]
    assert min(material_seam_change)>.01
    # Same force/pose controls after opening: compare recontact to saved slide,
    # not to pre-slide press. This is cycle endpoint parity, not another solve.
    np.testing.assert_allclose(poses[-1]['u'],poses[13]['u'],rtol=1e-8,atol=1e-12)
    np.testing.assert_allclose(poses[-1]['z'],poses[13]['z'],rtol=1e-7,atol=1e-12)
    save(dict(verified=True,source_sha256={s:sha(s) for s in stages},snapshot_inventory=list(range(1,17)),poses=reports,source_actuator_slide_m=.02,all_six_actual_skin_y_travel_m=travel,independent_projected_area_change_m2=area_change,independent_target_seam_material_parameter_changes=material_seam_change,target_anchor_held=True,strict_opening_verified=True,active_open_fresh_checkpoint_chain=True,cycle_endpoint_parity=True,integrated_work_not_claimed=True))


def test_relative_admission_and_restart_safety(monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled,solve_static_nonlinear,validate_general_contact
    from anyfesolver.general_contact.finite_rotation_predictor import finite_surface_partitions,per_surface_seed_descriptor,per_surface_rigid_seed
    from scipy.sparse import eye
    model,domain,settings,res,program=build();before=repr(model)
    token=CancellationToken();token.cancel('tilted relative precancel')
    with pytest.raises(SolveCancelled):solve(model,settings,res,program,cancellation_token=token)
    with pytest.raises(ValueError,match='finite_frame_seed requires contact_path_program'):
        solve_static_nonlinear(model,contact=res.definition,contact_settings=settings)
    assert repr(model)==before
    # Exact source/runtime selector guards are independent of unsupported physics.
    assert not validate_general_contact(model,res.definition,replace(settings,shell_skin_kinematics='reference_linear'),analysis_control='contact_path').valid
    assert not validate_general_contact(model,res.definition,settings,analysis_control='force').valid
    orphan=build();orphan[0].add_node(99,2.,2.,2.)
    with pytest.raises(ValueError,match='cover every structural element and node'):finite_surface_partitions(orphan[0],orphan[3].definition)
    partial_surfaces=list(res.definition.surfaces);partial_surfaces[0]=replace(partial_surfaces[0],element_ids=partial_surfaces[0].element_ids[:-1]);partial=replace(res.definition,surfaces=tuple(partial_surfaces))
    with pytest.raises(ValueError):finite_surface_partitions(model,partial)
    overlap=replace(res.definition,surfaces=(res.definition.surfaces[0],replace(res.definition.surfaces[1],element_ids=res.definition.surfaces[0].element_ids)))
    with pytest.raises(ValueError,match='overlap'):finite_surface_partitions(model,overlap)
    shared=replace(res.definition,surfaces=(replace(res.definition.surfaces[0],element_ids=(1,)),replace(res.definition.surfaces[1],element_ids=(2,))))
    with pytest.raises(ValueError,match='missing/shared nodes'):finite_surface_partitions(model,shared)
    disconnected=replace(res.definition,surfaces=(replace(res.definition.surfaces[0],element_ids=(1,3)),replace(res.definition.surfaces[1],element_ids=(2,4,5,6))))
    with pytest.raises(ValueError,match='not one connected'):finite_surface_partitions(model,disconnected)
    # Specifically remove target anchor rz, keeping all remaining controls legal.
    index=next(i for i,c in enumerate(program.controls) if c.terms==((7,'rz',1.),))
    absent=replace(program,controls=program.controls[:index]+program.controls[index+1:],waypoints=tuple(w[:index]+w[index+1:] for w in program.waypoints))
    with pytest.raises(ValueError,match='complete six-coordinate'):per_surface_seed_descriptor(model,res.definition,absent)
    descriptors=per_surface_seed_descriptor(model,res.definition,program);rows=rows_for(model,program);q=rigid_pose(model,math.pi/3.);target=program.target_at(14).copy();target[0]+=1e-4
    with pytest.raises(ValueError,match='consistent reference frame'):per_surface_rigid_seed(model,program,target,eye(manager_n:=len(q),format='csr'),np.zeros(manager_n),q,rows,descriptors)
    from examples.general_contact_multiple_beam_s3_transient import build as dynamic
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.general_contact.transient import solve_transient_contact,_admit,ContactTransientSolverControls
    import anyfesolver.general_contact.multiple_beam_s3_transient as owner
    dm,dd,ds,velocity=dynamic();controls=ContactTransientSolverControls(allow_multiple_deformable_beam_s3_dev=True);assert _admit(dm,dd,ds,controls)==owner.PROFILE
    class WrongSettings:finite_frame_seed='per_surface_reference_frames_dev'
    with pytest.raises(ValueError,match='requires exact contact definitions/settings'):_admit(dm,dd,WrongSettings(),controls)
    calls=[]
    def forbidden(*args,**kwargs):calls.append(True);raise AssertionError('allocation before frame selector refusal')
    monkeypatch.setattr(owner,'pair_definitions',forbidden);monkeypatch.setattr(owner.MultipleBeamS3TransientManager,'__init__',forbidden);monkeypatch.setattr(dm,'apply_boundary_conditions',forbidden)
    with pytest.raises(ValueError,match='finite_frame_seed'):
        solve_transient_contact(dm,dd,TransientConfig(dt=1e-5,t_end=1e-5,initial_velocity=velocity),contact_settings=replace(ds,finite_frame_seed='per_surface_reference_frames_dev'),solver_controls=controls)
    assert not calls
    from anyfesolver import LoadCase,ArcLengthControl,solve_static_arc_length
    from examples.general_contact_triangulated_rigid_patch import case
    import anyfesolver.general_contact.arc_contact as arc_owner
    am,ad,aset=case();assert validate_general_contact(am,ad,aset,analysis_control='arc_length').valid
    load=LoadCase('unused-independent-frame');load.add_nodal_load(2,forces=(0.,0.,-.01));arc_before=repr(am)
    arc_calls=[]
    def no_arc(*args,**kwargs):arc_calls.append(True);raise AssertionError('arc allocation before seed refusal')
    with monkeypatch.context() as context:
        context.setattr(arc_owner.TriangulatedRigidManager,'__init__',no_arc);context.setattr(am,'apply_boundary_conditions',no_arc)
        with pytest.raises(ValueError,match='finite_frame_seed'):
            solve_static_arc_length(am,load,control=ArcLengthControl(initial_load_increment=1e-5,minimum_load_increment=1e-6,maximum_load_increment=1e-5,max_steps=1),contact=ad,contact_settings=replace(aset,finite_frame_seed='per_surface_reference_frames_dev'))
    assert not arc_calls and repr(am)==arc_before
    save(dict(verified=True,public_precancel=True,declared_model_repr_unchanged=True,implicit_force_refused=True,
        unused_old_skin_refused=True,orphan_refused=True,partial_partition_refused=True,overlapping_elements_refused=True,
        shared_nodes_refused=True,disconnected_refused=True,missing_anchor_refused=True,inconsistent_frame_refused=True,
        legacy_valid_transient_before_allocation=True,legacy_valid_arc_before_allocation=True,wrong_type_preserved=True,
        transaction_scope='elastic declared model fields/material and accepted states; generic hidden rollback not qualified'))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not all((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(name+'.json')).is_file() for name in ('run-15','proof-chain')),
    reason='archived tilted inputs AUTOMATIC_WORKFLOW_EVIDENCE run-15.json/proof-chain.json absent: checkpoint restart-refusal replay requires the original staged evidence root')
def test_relative_archive_checkpoint_restart_refusals():
    model,domain,settings,res,program=build();open_report=read('run-15');assert read('proof-chain')['strict_opening_verified'];checkpoint=open_report['checkpoint'];before=repr(model)
    with pytest.raises(ValueError,match='incompatible'):solve(model,replace(settings,finite_frame_seed='common_reference_frame_dev'),res,program,restart_checkpoint=checkpoint,max_load_factor=1.)
    changed=replace(program,waypoints=program.waypoints[:-1]+(tuple(v+1e-8 if i==1 else v for i,v in enumerate(program.waypoints[-1])),))
    with pytest.raises(ValueError,match='incompatible'):solve(model,settings,res,changed,restart_checkpoint=checkpoint)
    damaged=copy.deepcopy(checkpoint);damaged['path_state']['normal_forces'][0]+=1e-5
    with pytest.raises(ValueError,match='SHA-256'):solve(model,settings,res,program,restart_checkpoint=damaged)
    assert repr(model)==before
    save(dict(verified=True,open_checkpoint_sha256=sha('run-15'),seed_selector_restart_refusal=True,program_identity_refusal=True,contents_refused=True))


@pytest.mark.skipif(not Path('C:/ANYworkspaceWork/contact_finite_rotation_shell_work_20261008/outputs/run-15.json').is_file(),
    reason='archived legacy checkpoint contact_finite_rotation_shell_work_20261008/outputs/run-15.json absent: legacy default continuation replay requires the original saved run')
def test_legacy_default_finite_checkpoint_continuation():
    from examples.general_contact_finite_rotation_shell_work import build as prior,solve as prior_solve
    from anyfesolver.general_contact.core import _contact_settings_contract
    path=Path('C:/ANYworkspaceWork/contact_finite_rotation_shell_work_20261008/outputs/run-15.json');stored=json.loads(path.read_text())
    model,domain,settings,res,program=prior();assert settings.finite_frame_seed=='common_reference_frame_dev'
    assert 'finite_frame_seed' not in _contact_settings_contract(settings) and res.identity==stored['identity']
    result=prior_solve(model,settings,res,program,restart_checkpoint=stored['checkpoint'])
    assert result.status=='completed' and not result.snapshots
    np.testing.assert_array_equal(result.displacements,stored['poses'][-1]['u'])
    save(dict(verified=True,legacy_default_automatic_identity_unchanged=True,legacy_v6_owner_local_checkpoint_continues=True,zero_new_increments=True,old_output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),request_identity_cross_package_not_claimed=True))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived tilted inputs AUTOMATIC_WORKFLOW_EVIDENCE run-4..run-16.json absent: equilibrium discriminator replay requires the original staged evidence root')
def test_saved_equilibrium_discriminator():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    model,domain,settings,res,program=build();manager=GeneralS3PatchManager(model,res.definition,settings);C=rows_for(model,program)
    fixed=[model.mesh.dof_manager.get_node_dofs(9)[1]];free=np.setdiff1d(np.arange(manager.n_dof),fixed);names=dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5);rows=[]
    for stage in ('run-4','run-8','run-12','run-13','run-14','run-15','run-16'):
        for pose in read(stage)['poses']:
            q=np.asarray(pose['u']);z=np.asarray(pose['z']);raw=linearize(manager,q,z);ext=np.zeros(manager.n_dof)
            for force,value in zip(program.nodal_forces,program.force_values_at(pose['step'])):ext[model.mesh.dof_manager.get_node_dofs(force.node_id)[names[force.dof]]]+=value
            r,k,trial=_assemble_nonlinear_system(model,q,{},5,tangent=False,kinematics='corotational',corotational_tangent='consistent',require_full_coordinates=True);_discard_nonlinear_state_candidate(trial)
            residual=(r+raw.force-ext-C.T@np.asarray(pose['record']['reaction_generalized']))[free]
            norm1=float(np.linalg.norm(residual,1));norm2=float(np.linalg.norm(residual))
            rows.append(dict(step=pose['step'],l1_n=norm1,l2_n=norm2,l1_normalized=norm1/FREF,solver_record_norms={k:v for k,v in pose['record'].items() if 'norm' in k or 'residual' in k},passes_frozen_gate=norm1/FREF<=1e-6))
    save(dict(verified=True,rows=rows,source_sha256={s:sha(s) for s in ('run-4','run-8','run-12','run-13','run-14','run-15','run-16')},question='pose attribution and L2 stopping versus registered L1 equilibrium; no solver rerun'))
