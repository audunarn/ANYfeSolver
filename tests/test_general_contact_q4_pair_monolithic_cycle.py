"""New force cycle with independent saved-pose proof and a fresh open restart."""
import copy,hashlib,importlib.util,json,os
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_q4_pair_monolithic_cycle import build,solve

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
_old=load_module('_monolithic_reference',Path(__file__).with_name('test_general_contact_q4_pair_monolithic.py'))
_oracle=load_module('_cycle_oracle',Path(__file__).parents[1]/'docs/d1_review_support/contact_q4_pair_monolithic_cycle_20261008/oracle.py')
skin,free_dofs,exact_pose_gap,hard_reference=_old.skin,_old.free_dofs,_old.exact_pose_gap,_old.hard_reference
FREF=1e-4
MREF=1e-4
def save(data):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else x),encoding='utf-8')
def read_report(stage):return json.loads((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/f'{stage}.json').read_text())
def report_sha(stage):return hashlib.sha256((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/f'{stage}.json').read_bytes()).hexdigest()


def test_simple_exact_oracle_and_initial_clearance_reference():
    from anyfesolver import LoadCase
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    model,domain,settings,res,program=build();manager=GeneralS3PatchManager(model,res.definition,settings)
    points=np.concatenate(manager._positions(np.zeros(manager.n_dof)));np.testing.assert_allclose(points,skin(model,np.zeros(manager.n_dof))[0],atol=1e-13,rtol=0)
    certificate=_oracle.strict_separation(points);assert certificate['strict_positive'] and certificate['source_normal_gap_lower_m']>0
    delta=F(float(model.mesh.nodes[1].z-.025))-F(float(model.mesh.nodes[7].z+.025));assert delta>0
    support,g,g0=_old._reference.independent_linear_contact(model);np.testing.assert_allclose(g0,support*float(delta),atol=1e-22,rtol=1e-14)
    load=LoadCase('independent press');[load.add_nodal_load(n,forces=(0.,0.,-5e-5)) for n in (3,6)];reference=hard_reference(model,load)
    assert np.max(reference['forces'])>0 and reference['penalty_diagonal'] is False
    source=np.array(((0.,0.,2e-7),(1.,0.,2e-7+1e-5),(1.,1.,2e-7+1e-5),(0.,1.,2e-7)))
    target=np.array(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)))
    tilted=_oracle.strict_separation(np.vstack((source,target)),((1,2,3,4),),((5,6,7,8),));assert tilted['strict_positive'] and tilted['source_normal_gap_lower_m']>0
    zero=_oracle.strict_separation(np.vstack((target,target)),((1,2,3,4),),((5,6,7,8),));assert not zero['strict_positive']
    adverse=source.copy();adverse[0,2]=-2e-7
    negative=_oracle.strict_separation(np.vstack((adverse,target)),((1,2,3,4),),((5,6,7,8),));assert not negative['strict_positive']
    save(dict(verified=True,initial_clearance=certificate,represented_delta_fraction=str(delta),independent_initial_weighted_gap=g0,hard_contact_reference=reference,tilted_control=tilted,zero_control=zero,adverse_control=negative,expensive_cell_proof_not_implemented=True))


@pytest.mark.skipif(not os.environ.get('CYCLE_STAGE'),
    reason='installed cycle stage input CYCLE_STAGE absent: staged cycle solve requires the original workflow stage selection')
def test_installed_cycle_solver_saved_inventory():
    stage=os.environ['CYCLE_STAGE'];model,domain,settings,res,program=build();kwargs={}
    if stage=='prefix':kwargs['max_load_factor']=2/3
    if stage=='resume':
        proof=read_report('proof-prefix');assert proof['verified'] and proof['opening_verified'] and proof['source_sha256']==report_sha('prefix')
        kwargs['restart_checkpoint']=read_report('prefix')['checkpoint']
    result=solve(model,settings,res,program,**kwargs);history={row['step_index']:row for row in result.info.get('force_displacement_history',[])}
    snapshots=[]
    for snapshot,step in zip(result.snapshots,result.steps[-len(result.snapshots):]):
        record=history[snapshot.step_index];contact=record['contact'];z=np.array(contact['numerical_pressure_pa'])*np.array(contact['reference_support_m2'])
        if result.restart_checkpoint is not None and snapshot.step_index==result.restart_checkpoint['path_state']['accepted_step']:z=np.array(result.restart_checkpoint['path_state']['normal_forces'])
        snapshots.append(dict(step=snapshot.step_index,displacements=snapshot.displacements,normal_forces=z,force_values=program.force_values_at(snapshot.step_index),contact=contact,record=record,reactions=step.support_reactions))
    report=dict(stage=stage,status=result.status,failure_reason=result.failure_reason,identity=res.identity,program=program.contract(),checkpoint=result.restart_checkpoint,info=result.info,snapshot_inventory=[s['step'] for s in snapshots],poses=snapshots,solver_completed=result.status=='completed',physical_fixture=dict(upper_midplane_z_m=.04+2e-7,target_midplane_z_m=-.01,initial_clearance_declared_m=2e-7,skins='reference_skin_linearized_director_work_map_v1'),phase_force_provenance='accepted p*S, terminal checkpoint exact z replaces reconstruction')
    save(report);assert result.status=='completed',(result.failure_reason,result.info.get('contact_failure'))
    assert report['snapshot_inventory']=={'full':[1,2,3],'prefix':[1,2],'resume':[3]}[stage]
    if stage=='resume':
        full=read_report('full');proof=read_report('proof-full');assert proof['verified'] and proof['source_sha256']==report_sha('full')
        a,b=snapshots[-1],full['poses'][-1]
        report['full_sha256']=report_sha('full');report['restart_u_max_error_m']=float(np.max(np.abs(np.array(a['displacements'])-np.array(b['displacements']))));report['restart_z_max_error_n']=float(np.max(np.abs(np.array(a['normal_forces'])-np.array(b['normal_forces']))))
        assert report['restart_u_max_error_m']<=1e-9 and report['restart_z_max_error_n']/FREF<=1e-6
    report['inventory_verified']=True;save(report)


@pytest.mark.skipif(not os.environ.get('CYCLE_STAGE'),
    reason='installed cycle stage input CYCLE_STAGE absent: saved mechanical proof requires the original staged report and stage selection')
def test_saved_cycle_mechanical_proof():
    from anyfesolver import LoadCase
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.semismooth import linearize,complementarity,reference_nodal_support
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    stage=os.environ['CYCLE_STAGE'].removeprefix('proof-');source=read_report(stage);assert source['solver_completed'] and source['inventory_verified']
    model,domain,settings,res,program=build();assert source['identity']==res.identity and source['program']==program.contract()
    manager=GeneralS3PatchManager(model,res.definition,settings);reports=[];reference_cache={}
    for pose in source['poses']:
        step=pose['step'];u=np.array(pose['displacements']);z=np.array(pose['normal_forces']);raw=linearize(manager,u,z,tangent=False)
        C,active,_,_=complementarity(raw.state,z,manager.stiffness*reference_nodal_support(manager.law),raw.gap_gradient,tangent=False)
        c=float(np.sum(np.abs(C))/FREF);assert c<=1e-8 and np.min(z)>=0
        # Pure elastic admitted fixture: force depends on accepted pose, not
        # constitutive history. Fresh {} does not qualify nonlinear rollback.
        structural,_,states=_assemble_nonlinear_system(model,u,{},5,tangent=False,kinematics='von_karman');_discard_nonlinear_state_candidate(states)
        external=np.zeros(manager.n_dof)
        for force,value in zip(program.nodal_forces,program.force_values_at(step)):external[model.mesh.dof_manager.get_node_dofs(force.node_id)[2]]=value
        equilibrium=float(np.linalg.norm((structural+raw.force-external)[free_dofs(model)])/FREF);assert equilibrium<=1e-6
        points,mids,rotation=skin(model,u);actual=np.concatenate(manager._positions(u));np.testing.assert_allclose(points,actual,atol=1e-13,rtol=0)
        uz=np.array([u[model.mesh.dof_manager.get_node_dofs(n)[2]] for n in range(1,16)]);theta=np.array([u[model.mesh.dof_manager.get_node_dofs(n)[3:]] for n in range(1,16)])
        motion=dict(upper_uz_range_m=float(np.ptp(uz[:6])),target_uz_norm_m=float(np.linalg.norm(uz[6:])),body_rotation_norms_rad=[float(np.linalg.norm(theta[:6])),float(np.linalg.norm(theta[6:]))],rotational_skin_displacement_m=float(np.linalg.norm(rotation)))
        assert motion['upper_uz_range_m']>1e-9 and motion['body_rotation_norms_rad'][0]>1e-9
        if step==2:
            gap=_oracle.strict_separation(actual);assert gap['strict_positive'] and gap['source_normal_gap_lower_m']>0,('global z enclosure cannot certify strict opening',gap)
            assert not active.any() and np.sum(np.abs(z))/FREF<=1e-8
            opening=True
        else:
            gap=exact_pose_gap(actual);assert active.any() and np.max(z)>0
            assert motion['target_uz_norm_m']>1e-10 and min(motion['body_rotation_norms_rad'])>1e-9 and motion['rotational_skin_displacement_m']>1e-11
            opening=False
        assert tuple(manager.law.pressure_mode_ids)==(1,2,3,4,5,6) and np.min(raw.state.reference_support)>0
        physical=-raw.local_force.reshape(-1,3);pf=float(np.linalg.norm(physical.sum(axis=0))/FREF);pm=float(np.linalg.norm(np.cross(actual,physical).sum(axis=0))/MREF)
        sm=np.cross(mids,physical).sum(axis=0)+sum((-.025 if n<=6 else .025)*np.cross([0.,0.,1.],physical[n-1]) for n in range(1,16));se=float(np.linalg.norm(sm)/MREF);assert max(pf,pm,se)<=1e-6
        key=tuple(pose['force_values'])
        if key not in reference_cache:
            load=LoadCase('independent phase');[load.add_nodal_load(item.node_id,forces=(0.,0.,value)) for item,value in zip(program.nodal_forces,key)];reference_cache[key]=hard_reference(model,load)
        ref=reference_cache[key]
        bodywork=[float(raw.force[[model.mesh.dof_manager.get_node_dofs(n)[k] for n in ns for k in range(6)]]@u[[model.mesh.dof_manager.get_node_dofs(n)[k] for n in ns for k in range(6)]]) for ns in (range(1,7),range(7,16))]
        reports.append(dict(step=step,equilibrium_error=equilibrium,complementarity_error=c,normal_forces=z,support=raw.state.reference_support,pressure=raw.state.pressure,normalized_gap=raw.state.normalized_gap,gap=gap,opening_verified=opening,motion=motion,wrenches=dict(origin=[0.,0.,0.],physical_force=pf,physical_moment=pm,structural_coordinate_moment=se,second_order_skin_deficit=np.cross(rotation,physical).sum(axis=0)),reactions=pose['reactions'],external_endpoint_work_j=float(external@u),both_body_instantaneous_contact_work_j=bodywork,nonlinear_vs_linear_relative=float(np.linalg.norm(u-ref['displacements'])/max(np.linalg.norm(ref['displacements']),1e-30)),no_nonlinear_linear_gate_claim=True))
    assert [p['step'] for p in reports]=={'full':[1,2,3],'prefix':[1,2],'resume':[3]}[stage]
    save(dict(verified=True,stage=stage,source_sha256=report_sha(stage),identity=res.identity,snapshot_inventory=[p['step'] for p in reports],poses=reports,opening_verified=any(p['opening_verified'] for p in reports),elastic_pose_recomputation=True,integrated_work_not_claimed=True))


def test_actual_cycle_definition_content_and_precancel_refusal():
    from anyfesolver import CancellationToken,SolveCancelled
    model,domain,settings,res,program=build();before=repr(model)
    token=CancellationToken();token.cancel('cycle public precancel')
    with pytest.raises(SolveCancelled):solve(model,settings,res,program,cancellation_token=token)
    assert repr(model)==before
    save(dict(verified=True,public_precancel=True,declared_FEModel_repr_unchanged=True))


def test_actual_empty_control_admission_and_dependent_control_refusal():
    from anyfesolver import CancellationToken,SolveCancelled
    from anyfesolver.general_contact.static_path import ContactPathProgram,ContactPathControl
    model,domain,settings,res,program=build();before=repr(model)
    # The legitimate pure-force programme has zero controls: the preflight
    # rank of its empty control matrix is zero, so the actual driver must
    # reach the declared cancellation point instead of a zero-size rank
    # reduction, without mutating the declared model.
    token=CancellationToken();token.cancel('empty-control preflight precancel')
    with pytest.raises(SolveCancelled):solve(model,settings,res,program,cancellation_token=token)
    assert repr(model)==before
    # Nonempty dependent controls (one free DOF declared twice with
    # different weights) and fully constrained controls (clamped nodes)
    # still refuse through the same rank check, before any model mutation.
    dependent=ContactPathProgram(
        controls=(ContactPathControl.node(3,'uz'),ContactPathControl(((3,'uz',2.0),))),
        waypoints=((0.,0.),(1.,2.)),steps_per_segment=1)
    with pytest.raises(ValueError,match='constrained or dependent'):
        solve(model,settings,res,dependent)
    clamped=ContactPathProgram(
        controls=(ContactPathControl.node(1,'ux'),ContactPathControl.node(1,'uy')),
        waypoints=((0.,0.),(1.,1.)),steps_per_segment=1)
    with pytest.raises(ValueError,match='constrained or dependent'):
        solve(model,settings,res,clamped)
    assert repr(model)==before
    save(dict(verified=True,empty_control_rank_zero_admission=True,
        dependent_control_refusal=True,fully_constrained_control_refusal=True,
        refusals_before_model_mutation=True,declared_FEModel_repr_unchanged=True))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not all((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(name+'.json')).is_file() for name in ('prefix','proof-prefix')),
    reason='archived cycle inputs AUTOMATIC_WORKFLOW_EVIDENCE prefix.json/proof-prefix.json absent: restart-binding replay requires the original evidence root')
def test_actual_archive_cycle_checkpoint_restart_refusals():
    from anyfesolver.general_contact import resolve_automatic_contact_domain
    model,domain,settings,res,program=build();prefix=read_report('prefix');proof=read_report('proof-prefix');assert proof['verified'] and proof['opening_verified'] and proof['source_sha256']==report_sha('prefix');before=repr(model)
    changed_forces=list(program.nodal_forces);changed_forces[0]=replace(changed_forces[0],values_n=(0.,-5e-5,+5e-5,-4e-5));changed=replace(program,nodal_forces=tuple(changed_forces))
    with pytest.raises(ValueError,match='incompatible'):solve(model,settings,res,changed,restart_checkpoint=prefix['checkpoint'])
    altered=replace(domain,name='incompatible-cycle-provenance');altered_res=resolve_automatic_contact_domain(model,altered,settings,analysis_control='contact_path')
    with pytest.raises(ValueError,match='incompatible'):solve(model,settings,altered_res,program,restart_checkpoint=prefix['checkpoint'])
    damaged=copy.deepcopy(prefix['checkpoint']);damaged['path_state']['normal_forces'][0]+=1e-6
    with pytest.raises(ValueError,match='SHA-256 does not match its contents'):solve(model,settings,res,program,restart_checkpoint=damaged)
    save(dict(verified=True,cycle_definition_refusal=True,provenance_refusal=True,content_hash_refusal=True,prefix_sha256=report_sha('prefix'),proof_prefix_sha256=report_sha('proof-prefix'),hidden_rollback_not_qualified=True))
