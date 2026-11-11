"""Current D1 registered 16-Q4/32-S3 weighted-displacement development cycle.

The missing historical control/preflight source is not reconstructed. This
compatible development control uses the registered physical-bump work map,
documented press endpoint and NCP scale. It establishes no acceptance transfer.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib,json,time
from pathlib import Path
import numpy as np
import anyfesolver
from anyfesolver import ContactPathControl,ContactPathProgram,solve_static_nonlinear,validate_general_contact
from scripts.general_contact_d1_interior_press_dev import build_saved_cycle,verify_saved_cycle,validate_restart_poses
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import _json_default

ENDPOINT_M=-9.340222503369683e-5
NCP_SCALE_N=0.013376864330719142
REGISTERED_INPUTS=dict(source_q4_faces=16,target_s3_faces=32,source_columns=4,source_rows=4,target_columns=4,target_rows=4,
    force_n=.018,slide_distance_m=.04,reference_slope=.0001,target_young_multiplier=5.,target_y0_m=-.1,
    source_reference_downshift_m=0.,source_support='perimeter',source_y_profile=None,offline_structural_reference_y16=False,
    load_shape='physical_bump',four_phase=True,steps_per_segment=2,slide_intervals=1,slide_body='target',
    target_reference_full_left_clamp=True,target_left_edge_motion=True,shared_edge_tie=False,q4_gap_certificate='overlap_local_dev')

def build_displacement_cycle(saved):
    for key,value in REGISTERED_INPUTS.items():
        if key not in saved or saved[key]!=value:raise ValueError('unsupported registered fixture input: '+key)
    model,definition,settings,force=build_saved_cycle(saved)
    if (saved['source_q4_faces'],saved['target_s3_faces'],saved['force_n'],saved['slide_distance_m'],saved['load_shape']) != (16,32,.018,.04,'physical_bump'):
        raise ValueError('weighted displacement requires the registered 16-Q4/32-S3 fixture')
    terms=tuple((load.node_id,'uz',-load.values_n[1]/saved['force_n']) for load in force.nodal_forces)
    if not np.isclose(sum(term[2] for term in terms),1.,rtol=0.,atol=1e-11):
        raise ValueError('registered physical-bump work map does not balance')
    program=ContactPathProgram(controls=(*force.controls,ContactPathControl(terms)),
        waypoints=tuple((*row,ENDPOINT_M*factor) for row,factor in zip(force.waypoints,(0.,1.,1.,0.,1.),strict=True)),
        steps_per_segment=2)
    return model,definition,replace(settings,complementarity_force_scale_n=NCP_SCALE_N),program

def load_restart(path):
    checkpoint=json.loads(path.read_text())['restart_checkpoint']
    if checkpoint is None:raise ValueError('restart source contains no committed checkpoint')
    return checkpoint

def validate_saved_displacement(saved):
    """Check complete inventory, finite shapes, controls, norms and endpoint.

    The correspondence flag describes these consistency checks; it does not
    authenticate arbitrary intermediate fields. Exact intermediate equality
    for the actual bound fresh/split runs is checked by RESTART_COMPARISON.
    """
    model,definition,settings,program=build_displacement_cycle(saved)
    if saved['status']!='completed' or saved['accepted_steps']!=8:raise ValueError('displacement assessment requires eight completed poses')
    if saved['external_nodal_load_count']!=0 or saved['endpoint_m']!=ENDPOINT_M or saved['ncp_scale_n']!=NCP_SCALE_N:
        raise ValueError('displacement loads or control scales changed')
    if tuple(tuple(term) for term in saved['control_terms'])!=program.controls[-1].terms:raise ValueError('weighted control terms changed')
    poses=np.asarray(saved['accepted_pose_displacements']);checkpoint=saved['restart_checkpoint'];history=checkpoint['path_state']['history']
    if poses.shape!=(8,model.mesh.dof_manager.total_dofs) or not np.all(np.isfinite(poses)):
        raise ValueError('displacement poses must have finite complete coordinates')
    for terminal in (saved['displacements'],checkpoint['displacements']):
        vector=np.asarray(terminal)
        if vector.shape!=(model.mesh.dof_manager.total_dofs,) or not np.all(np.isfinite(vector)):
            raise ValueError('terminal displacement must have finite complete coordinates')
    steps=checkpoint['path_state']['steps']
    if len(poses)!=8 or len(history)!=8 or len(steps)!=8 or saved['history']!=history or checkpoint['path_state']['accepted_step']!=8:
        raise ValueError('complete displacement pose/history/checkpoint correspondence changed')
    if not np.array_equal(poses[-1],saved['displacements']) or not np.array_equal(poses[-1],checkpoint['displacements']):
        raise ValueError('terminal displacement disagrees with snapshot/checkpoint')
    errors=[]
    for step,(pose,row) in enumerate(zip(poses,history,strict=True),1):
        declared=np.asarray(row['target']);norm=steps[step-1]['displacement_norm']
        if declared.shape!=(3,) or not np.all(np.isfinite(declared)) or not np.isfinite(norm):
            raise ValueError('displacement target/step norm must be finite and complete')
        if abs(float(np.linalg.norm(pose))-steps[step-1]['displacement_norm'])>1e-10:
            raise ValueError('pose differs from committed step norm')
        if row['step_index']!=step or row.get('nodal_forces_n') or row.get('external_nodal_work_trapezoid_j',0.)!=0.:
            raise ValueError('displacement history has wrong step or external load')
        actual=np.asarray([sum(weight*pose[model.mesh.dof_manager.get_node_dofs(node)[('ux','uy','uz').index(dof)]] for node,dof,weight in c.terms) for c in program.controls])
        target=program.target_at(step)
        error=max(float(np.max(np.abs(actual-target))),float(np.max(np.abs(np.asarray(row['target'])-target))))
        if error>1e-10:raise ValueError('displacement pose or intermediate target changed')
        errors.append(error)
    return dict(displacement_specific_checks_met=True,maximum_control_error_m=max(errors),zero_external_loads=True,complete_pose_history_checkpoint_correspondence=True)

def assess_saved(path,output):
    saved=json.loads(path.read_text());checks=validate_saved_displacement(saved)
    verify_saved_cycle(path,output)
    record=json.loads(output.read_text());record.update(checks);output.write_text(json.dumps(record,indent=2)+'\n')

def run(output,fixture,restart=None,max_path_fraction=1.):
    if output.exists():raise FileExistsError(output)
    saved=json.loads(fixture.read_text());model,definition,settings,program=build_displacement_cycle(saved)
    report=validate_general_contact(model,definition,settings)
    if not report.valid:raise ValueError(report.issues)
    checkpoint=load_restart(restart) if restart else None
    prefix=validate_restart_poses(json.loads(restart.read_text()),checkpoint) if restart else []
    started=time.perf_counter()
    result=solve_static_nonlinear(model,contact=definition,contact_settings=settings,contact_path_program=program,
        max_iterations=18,max_load_factor=max_path_fraction,emit_restart_checkpoint=True,
        record_increment_snapshots=True,restart_checkpoint=checkpoint)
    payload={key:saved[key] for key in ('source_q4_faces','target_s3_faces','force_n','reference_slope','source_columns','source_rows','target_columns','target_rows','target_young_multiplier','target_y0_m','source_reference_downshift_m','source_support','source_y_profile','offline_structural_reference_y16','load_shape','four_phase','steps_per_segment','slide_distance_m','slide_intervals','slide_body','target_reference_full_left_clamp','target_left_edge_motion','shared_edge_tie','q4_gap_certificate')}
    new_poses=[np.asarray(s.displacements).tolist() for s in result.snapshots]
    payload.update(candidate='current_d1_registered_weighted_displacement_development',control_provenance='current physical-bump work map; documented endpoint/NCP scale; historical control/preflight source unavailable',
        anyfesolver_module=str(Path(anyfesolver.__file__).resolve()),fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        endpoint_m=ENDPOINT_M,ncp_scale_n=NCP_SCALE_N,control_terms=program.controls[-1].terms,
        external_nodal_load_count=len(program.nodal_forces),status=result.status,failure_reason=result.failure_reason,
        accepted_steps=len(result.steps),elapsed_seconds=time.perf_counter()-started,displacements=result.displacements.tolist(),
        history=result.info.get('force_displacement_history'),accepted_pose_displacements=prefix+new_poses,new_increment_count=len(new_poses),restart_prefix_pose_count=len(prefix),
        restart_checkpoint=result.to_restart_checkpoint() if result.restart_checkpoint is not None else None,restart_source=str(restart) if restart else None,
        restart_source_sha256=hashlib.sha256(restart.read_bytes()).hexdigest() if restart else None,
        prefix_pose_provenance='caller-supplied earlier poses; checkpoint endpoint validated; input-file hash retained')
    output.write_text(json.dumps(payload,indent=2,default=_json_default)+'\n')
    print(json.dumps({key:payload[key] for key in ('status','accepted_steps','failure_reason','elapsed_seconds')}))
    if result.status!='completed':raise RuntimeError(result.failure_reason)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--restart',type=Path);p.add_argument('--max-path-fraction',type=float,default=1.);p.add_argument('--assess-saved',type=Path)
    a=p.parse_args()
    if a.assess_saved:assess_saved(a.assess_saved,a.output)
    else:run(a.output,a.fixture,a.restart,a.max_path_fraction)
