"""Saved evidence for the fixed tilted Q4 fixture, not engineering admission.

Stored residual/NCP diagnostics are producer-reported. Physical vertices and
gap bounds are reconstructed independently; input self-hashes establish only
consistency. Earlier caller-supplied poses are not checkpoint-authenticated.
"""
import hashlib,json,math
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
from anyfesolver.nonlinear_restart import load_nonlinear_checkpoint,_state_records,model_configuration_fingerprint
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from docs.d1_review_support.contact_finite_rotation_shell_work_20261008.oracle import posed_bounds

SCHEMA='TILTED_RELATIVE_Q4_PUBLIC_OUTPUT_V1'
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=json_value,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)
def finite_vector(value,size,name):
    a=np.asarray(value,float);require(a.shape==(size,) and np.isfinite(a).all(),name+' finite full vector');return a
def finite_tree(value):
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)
def terminal_states(snapshot_records,checkpoint_records,model,u,coordinates):
    # static_path snapshots are copied before final owner sealing. Compare ALL
    # mechanical fields; only the two documented final seal fields are separate.
    from anyfesolver.general_contact.finite_rotation_state import owner_local_coordinates,COORDINATES
    require(coordinates==COORDINATES,'exact finite owner state coordinates')
    require([r['element_id'] for r in checkpoint_records]==sorted(model.mesh.elements),'complete terminal checkpoint state inventory')
    seals={'qualified_q4_committed_binding','state_integrity_sha256'}
    for saved,committed in zip(snapshot_records,checkpoint_records,strict=True):
        require(saved['element_id']==committed['element_id'],'terminal state coordinate order')
        state=committed['state'];raw=saved['state']
        require(isinstance(state,dict) and isinstance(raw,dict) and seals<=state.keys() and not seals.intersection(raw),'snapshot-before-seal/final owner seal inventory')
        binding=state['qualified_q4_committed_binding'];digest=state['state_integrity_sha256']
        require(isinstance(binding,dict) and binding.get('num_layers')==5 and isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdefABCDEF' for c in digest),'current owner seal shape/layers')
        require(raw=={k:v for k,v in state.items() if k not in seals},'terminal mechanical state/checkpoint correspondence')
        element=model.mesh.elements[committed['element_id']];dofs=np.asarray(element.get_dof_mapping(model.mesh),int)
        local=owner_local_coordinates(model,element,u[dofs])
        # Existing owner binding/hash guards, without constitutive replay. No
        # checkpoint seal is generated and no current state is advanced.
        element.validate_committed_current_tangent_binding(model.mesh,model.get_material(element.material_name),local,state,5)
def decimal_director(theta,dref):
    # Same independent80-digit exponential used by the retained finite-Q4
    # verification, without production SO3/contact geometry kernels.
    with localcontext() as ctx:
        ctx.prec=80;a=[Decimal(float(v)) for v in theta];term=[Decimal(float(v)) for v in dref];total=term.copy()
        for n in range(1,100):
            x,y,z=term;ax,ay,az=a;term=[(ay*z-az*y)/n,(az*x-ax*z)/n,(ax*y-ay*x)/n];total=[v+w for v,w in zip(total,term)]
        return np.asarray([float(v) for v in total])
def fixture(model,settings,resolution,program):
    require(model.mesh.dof_manager.total_dofs==90 and len(model.mesh.nodes)==15 and program.total_steps==16,'registered15node90coordinate16step fixture')
    require(settings.shell_skin_kinematics=='corotational_nodal_director_dev' and settings.finite_frame_seed=='per_surface_reference_frames_dev','exact existing finite-frame selectors')
    manager=GeneralS3PatchManager(model,resolution.definition,settings);skin=manager.finite_skin
    require(skin is not None and skin.dofs.shape==(15,6),'exact finite physical skin inventory')
    return dict(model_fingerprint=model_configuration_fingerprint(model),identity=resolution.identity,program=program.contract(),
        reference=skin.reference,directors=skin.directors,offsets=skin.offsets,dofs=skin.dofs,pressure_ids=list(manager.law.pressure_mode_ids)),skin
def controls(model,program,u,step):
    names=dict(ux=0,uy=1,uz=2,rx=3,ry=4,rz=5)
    actual=np.array([sum(w*u[model.mesh.dof_manager.get_node_dofs(n)[names[d]]] for n,d,w in c.terms) for c in program.controls])
    target=program.target_at(step);scale=np.maximum(1e-9,np.max(np.abs(program.waypoints),axis=0))
    require(np.max(np.abs(actual-target)/scale)<=1e-6,'actual full program controls')
    require(u[model.mesh.dof_manager.get_node_dofs(9)[1]]==0.,'registered target yaw support')
def load_restart(path,through_step):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    require(isinstance(value,dict),'restart root mapping')
    report=value if value.get('schema')==SCHEMA else None
    if report is None and 'checkpoint' in value:raise ValueError('unknown saved workflow schema')
    cp=report.get('checkpoint') if report else value
    require(cp is not None,'requested restart has null checkpoint')
    cp=load_nonlinear_checkpoint(cp);step=cp['path_state']['accepted_step']
    require(isinstance(step,int) and 0<=step<=through_step<=16,'restart requested step range')
    prefix=[]
    if report:
        require(report['schema']==SCHEMA and report['status']=='completed','completed current saved restart report')
        prefix=report['poses'];require([p['step'] for p in prefix]==list(range(1,step+1)),'complete provided prefix inventory')
        for pose in prefix:finite_vector(pose['u'],90,'prefix pose')
        require(np.array_equal(prefix[-1]['u'],finite_vector(cp['displacements'],90,'checkpoint pose')) if prefix else step==0,'terminal prefix/checkpoint equality')
    return cp,prefix,dict(path=str(Path(path).resolve()),sha256=sha(path),prefix_scope='Provided prefix consistency only; earlier fields are not checkpoint-authenticated',bare_checkpoint=report is None)
def assess(record,model,settings,resolution,program):
    expected,skin=fixture(model,settings,resolution,program)
    require(record['schema']==SCHEMA and record['fixture']==json.loads(json.dumps(expected,default=json_value)) and record['identity']==resolution.identity and record['program']==program.contract(),'exact actual fixture/program/skin identity')
    poses=record['poses'];indices=[p['step'] for p in poses]
    require(type(record['accepted_step_total']) is int and 1<=record['accepted_step_total']<=16,'finite accepted-step range')
    require(indices==list(range(1,record['accepted_step_total']+1)),'complete saved pose inventory')
    history={r['step_index']:r for r in record['history']};require(len(history)==len(record['history'])==len(indices) and [r['step_index'] for r in record['history']]==indices,'unique ordered complete pose/history inventory')
    rows=[]
    for pose in poses:
        step=pose['step'];u=finite_vector(pose['u'],90,'pose');z=finite_vector(pose['z'],len(expected['pressure_ids']),'all pressure forces')
        require(np.min(z)>=0 and pose['record']==history[step],'nonnegative force/full history correspondence');controls(model,program,u,step)
        snapshot=pose['snapshot'];require(snapshot['step_index']==step and snapshot['load_factor']==step/16. and snapshot['num_dofs']==90 and np.array_equal(snapshot['displacements'],u),'full snapshot/pose correspondence')
        states=pose['element_state_records'];ids=[s['element_id'] for s in states]
        require(ids==sorted(model.mesh.elements) and snapshot['element_state_ids']==ids and finite_tree(states),'complete finite saved element-state inventory')
        contact=pose['record']['contact'];require(contact['source_pressure_node_ids']==expected['pressure_ids'],'pressure coordinate identity')
        S=finite_vector(contact['reference_support_m2'],len(z),'support');p=finite_vector(contact['numerical_pressure_pa'],len(z),'pressure')
        require(np.min(S)>0 and np.min(p)>=0 and np.allclose(p*S,z,rtol=1e-12,atol=1e-16),'all positive supports and force/pressure consistency')
        points=np.array([r+u[d[:3]]+h*decimal_director(u[d[3:]],normal) for r,d,h,normal in zip(skin.reference,skin.dofs,skin.offsets,skin.directors,strict=True)])
        angle=min(step,12)*math.pi/36.;axis=np.array((math.sin(angle),0.,math.cos(angle)))
        gap=posed_bounds(points,axis);opening=step<=12 or step==15
        require(gap['lower_m']>=-50e-6,'independent physical gap budget')
        require((gap['strict_positive'] and np.linalg.norm(z,1)/1e-4<=1e-8) if opening else max(z)>0,'actual open/active pressure and independent separation')
        row=history[step];require(np.isfinite(row['residual_norm']) and row['residual_norm']>=0,'finite reported residual')
        require(np.isfinite(row['complementarity_relative_force']) and 0<=row['complementarity_relative_force']<=1e-8,'reported NCP gate')
        rows.append(dict(step=step,independent_gap=gap,producer_reported_residual_norm=row['residual_norm'],producer_reported_NCP_relative=row['complementarity_relative_force'],current_full_residual_reassembled=False))
    cp=record['checkpoint'];require(cp is not None,'completed saved workflow checkpoint required');load_nonlinear_checkpoint(cp)
    require(cp['model_fingerprint']==expected['model_fingerprint'],'current model checkpoint identity')
    require(cp['path_state']['accepted_step']==record['accepted_step_total'] and cp['path_state']['history']==record['history'],'complete terminal checkpoint history')
    require(np.array_equal(poses[-1]['u'],cp['displacements']) and np.array_equal(poses[-1]['z'],cp['path_state']['normal_forces']),'terminal full pose/pressure checkpoint correspondence')
    terminal_states(poses[-1]['element_state_records'],cp['element_states'],model,np.asarray(cp['displacements']),record['element_state_coordinates'])
    require(record['status']=='completed','completed supplied workflow')
    return dict(status='PASS',assessed_poses=indices,complete_fresh_workflow=record['restart_source'] is None and indices==list(range(1,17)),rows=rows,
        scientific_acceptance=False,producer_reported_equilibrium_gate_verified=False,terminal_owner_binding_guards_verified=True,scope='Independent fixed-fixture skin/gap and saved control/history/pressure/material-state consistency; terminal owner binding/hash guards checked at reference deformational coordinates, no constitutive semantic replay; residual is finite reported diagnostic, NCP is producer-reported, no fresh assembled equilibrium or full owner-state validation; self-hashes do not authenticate arbitrary earlier fields')
def report_result(result,model,settings,resolution,program,prefix,restart_source,descriptor=None):
    # Capture each returned full displacement before optional state serialization.
    # Missing/failed state data remains explicit NONPASS; no checkpoint is made.
    if descriptor is None:descriptor,_=fixture(model,settings,resolution,program)
    history=result.info.get('force_displacement_history',[]);by_step={r['step_index']:r for r in history};poses=list(prefix);errors=[]
    for snapshot in result.snapshots:
        pose=dict(step=snapshot.step_index,u=finite_vector(snapshot.displacements,90,'returned snapshot'))
        poses.append(pose)
        try:
            row=by_step[snapshot.step_index];c=row['contact'];z=np.asarray(c['numerical_pressure_pa'])*np.asarray(c['reference_support_m2'])
            if result.restart_checkpoint is not None and snapshot.step_index==result.restart_checkpoint['path_state']['accepted_step']:z=np.asarray(result.restart_checkpoint['path_state']['normal_forces'])
            pose.update(z=z,record=row,snapshot=snapshot.to_dict(include_displacements=True))
            pose['element_state_records']=_state_records(snapshot.element_states,_guard_model=model)
        except Exception as exc:
            message=f'step {snapshot.step_index}: {type(exc).__name__}: {exc}';errors.append(message);pose['serialization_failure']=message
    return dict(schema=SCHEMA,status='NONPASS' if errors else result.status,failure_reason='; '.join(errors) if errors else result.failure_reason,producer_status=result.status,serialization_failures=errors,identity=resolution.identity,fixture=descriptor,program=program.contract(),
        capability=result.info.get('contact_path'),element_state_coordinates=result.info.get('element_state_coordinates'),accepted_steps=[p.step_index for p in result.snapshots],
        accepted_step_total=max((r['step_index'] for r in history),default=0),poses=poses,history=history,checkpoint=result.restart_checkpoint,
        restart_source=restart_source,terminal_restart=bool(restart_source and not result.snapshots),scientific_acceptance=False)
