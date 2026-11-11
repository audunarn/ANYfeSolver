"""Complete registered V16 public paths and explicitly scoped endpoint reference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from examples.general_contact_enriched_d1_path import run
from scripts.general_contact_enriched_path_geometry import PhysicalField,union_certificate
from scripts.general_contact_enriched_path_bounds import global_gap_certificate
from scripts.general_contact_enriched_path_reference import assess_endpoint
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.control import SolveCancelled


def publish(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


def equal_tree(a,b,*,tolerance=1e-10):
    if isinstance(a,dict):
        return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k],tolerance=tolerance) for k in a)
    if isinstance(a,list):
        return isinstance(b,list) and len(a)==len(b) and all(equal_tree(x,y,tolerance=tolerance) for x,y in zip(a,b))
    if type(a) in (float,int) and type(b) in (float,int):
        return bool(np.isfinite(a) and np.isfinite(b) and abs(a-b)<=tolerance)
    return type(a)is type(b) and a==b


def independent_publication(directory,*,cancel=lambda:False):
    """Assesses actual candidate pose before atomic publication; no quadrature."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    def assess(owner,candidate):
        payload=json.loads(candidate);row=payload['records'][-1]
        # These original predicates are complete at this accepted candidate;
        # do not execute later physical poses after a known lifecycle failure.
        if row['step_index']==6 and (row['active_pressure_modes']!=0 or any(v!=0. for v in row['pressure'])):
            raise EnrichedProfileError('original exact-zero pressure/open6 lifecycle failed')
        if row['step_index'] in (7,8) and row['active_pressure_modes']<=0:
            raise EnrichedProfileError('prospectively adopted V16 positive-recontact7,8 lifecycle failed')
        field=PhysicalField(owner.model,tuple(p.definition for p in owner.template.pairs),row['coordinates'],cancel=cancel)
        pose=np.asarray(row['coordinates']);sources,targets=field.vertices(pose)
        topology=union_certificate(field,sources,targets)
        gap=global_gap_certificate(field,pose)
        if gap['lower_m'] < -50e-6:raise EnrichedProfileError('independent original global path gap failed')
        publish(directory/f"pose-{row['step_index']:02d}.json",dict(
            status='CANDIDATE_ASSESSMENT_NOT_ACCEPTANCE',step=row['step_index'],candidate_sha256=hashlib.sha256(candidate).hexdigest(),
            source_count=len(sources),target_count=len(targets),topology=topology,gap=gap,
            independent_geometry=True,production_gap_certificate_not_reused=True))
    return assess


def verify_current_path(result):
    rows=result['records']
    if len(rows)!=8 or [r['step_index'] for r in rows]!=list(range(1,9)):
        raise EnrichedProfileError('all8 registered accepted poses required')
    for row in rows:
        if (len(row['coordinates'])!=364 or len(row['nodal_forces'])!=25
            or row['pressure_ids']!=list(range(1,26)) or row['equilibrium']>1e-6 or row['NCP']>1e-8
            or row['control_scaled_error']>1e-6 or row['base_KKT_relative']>1e-6
            or max(row['force_balance'],row['moment_balance'])>1e-6
            or row['physical_gap_certificate']['lower_m'] < -50e-6
            or not all(np.all(np.isfinite(row[name])) for name in
                ('coordinates','nodal_forces','load','reduced','control_reactions_N','support','normalized_gap','pressure'))
            or min(row['nodal_forces'])<0):
            raise EnrichedProfileError('complete original current perpose residual/control/reaction/wrench/finite predicate failed')
    if (rows[5]['active_pressure_modes']!=0 or rows[6]['active_pressure_modes']<=0 or rows[7]['active_pressure_modes']<=0
        or any(value!=0. for value in rows[5]['pressure'])):
        raise EnrichedProfileError('original exact-zero pressure/open6 and prospectively adopted V16 recontact7,8 required')


def verify_committed_proofs(directory,proof_directory,steps):
    directory,proof_directory=Path(directory),Path(proof_directory)
    for step in steps:
        commit=json.loads((directory/f'commit-{step:02d}.json').read_bytes())
        proof=json.loads((proof_directory/f'pose-{step:02d}.json').read_bytes())
        candidate=directory/f'candidate-{step:02d}.json'
        if (commit['status']!='ACCEPTED_PHYSICAL_STEP' or proof['status']!='CANDIDATE_ASSESSMENT_NOT_ACCEPTANCE'
            or commit['step_index']!=step or proof['step']!=step
            or commit['candidate_sha256']!=hashlib.sha256(candidate.read_bytes()).hexdigest()
            or commit['checkpoint_sha256']!=proof['candidate_sha256']):
            raise EnrichedProfileError('independent physical proof/actual accepted commit correspondence failed')


def complete_paths(output,*,cancel=lambda:False,include_extended_reference=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    paths={};owners={}
    for mode in ('force','displacement'):
        _,_,owner,result=run(output/mode,mode=mode,cancel=cancel,
            on_candidate=independent_publication(output/(mode+'-independent'),cancel=cancel))
        owners[mode]=owner;paths[mode]=result
        verify_current_path(result)
        verify_committed_proofs(output/mode,output/(mode+'-independent'),range(1,9))
    _,_,prefix,prefix_result=run(output/'displacement-prefix',mode='displacement',through=2,cancel=cancel,
        on_candidate=independent_publication(output/'prefix-independent',cancel=cancel))
    raw=prefix.checkpoint();sha=hashlib.sha256(raw).hexdigest()
    verify_committed_proofs(output/'displacement-prefix',output/'prefix-independent',range(1,3))
    if prefix_result['records'][-1]['active_pressure_modes']<=0:
        raise EnrichedProfileError('registered restart must be genuine active step2')
    publication=prefix._publication
    try:prefix.advance(cancel=lambda:True)
    except SolveCancelled:pass
    else:raise EnrichedProfileError('actual active-prefix cancellation did not refuse')
    if prefix._publication is not publication or prefix.checkpoint()!=raw or prefix._busy:
        raise EnrichedProfileError('actual active-prefix cancellation was not atomic')
    # Consequential cancellation AFTER a genuine next physical candidate was
    # solved, immediately before owner publication; no accepted seed changes.
    late_cancel=[False];late_candidate={}
    def cancel_before_commit(candidate):
        late_candidate.update(status='CANDIDATE_NOT_COMMITTED',step=json.loads(candidate)['accepted_step'],
            sha256=hashlib.sha256(candidate).hexdigest(),bytes=len(candidate))
        late_cancel[0]=True
    try:prefix.advance(cancel=lambda:late_cancel[0],before_publish=cancel_before_commit)
    except SolveCancelled:pass
    else:raise EnrichedProfileError('post-solve actual prefix cancellation did not refuse')
    if (late_candidate.get('step')!=3 or prefix._publication is not publication
        or prefix.checkpoint()!=raw or prefix._busy):
        raise EnrichedProfileError('post-solve actual prefix cancellation published state')
    publish(output/'ACTUAL_LATE_CANCEL.json',late_candidate)
    node=prefix.model.mesh.nodes[1];original=node.z
    try:
        node.z=original+1e-6
        try:prefix.checkpoint()
        except EnrichedProfileError:pass
        else:raise EnrichedProfileError('actual accepted-prefix physical mutation was not refused')
    finally:node.z=original
    if prefix.checkpoint()!=raw or prefix._publication is not publication:
        raise EnrichedProfileError('mutation refusal changed accepted prefix')
    _,_,restored,split=run(output/'displacement-restart',mode='displacement',prefix=output/'displacement-prefix/checkpoint.json',
        prefix_sha256=sha,cancel=cancel,on_candidate=independent_publication(output/'restart-independent',cancel=cancel))
    verify_committed_proofs(output/'displacement-restart',output/'restart-independent',range(3,9))
    verify_current_path(split)
    continuous=paths['displacement']
    correspondence=equal_tree(continuous['records'],split['records'])
    a=json.loads(owners['displacement'].checkpoint());b=json.loads(restored.checkpoint())
    a['terminal_step_checkpoint']=json.loads(a['terminal_step_checkpoint'])
    b['terminal_step_checkpoint']=json.loads(b['terminal_step_checkpoint'])
    checkpoint_equal=equal_tree(a,b)
    if not correspondence or not checkpoint_equal:
        raise EnrichedProfileError('original every-pose/full-checkpoint 2+6 correspondence failed')
    evidence=dict(status='PASS_REGISTERED_PATH_LIFECYCLE_ONLY',force_poses=8,displacement_poses=8,
        authentic_active_prefix=2,restored_new_poses=6,every_pose_correspondence=correspondence,
        full_checkpoint_correspondence=checkpoint_equal,tolerance=1e-10,prefix_sha256=sha,
        new_accepted_solve_count=24,post_solve_cancelled_candidate_count=1,
        independent_perpose_proofs=24,all_proofs_bound_to_actual_commit=True,
        descriptors={mode:r['descriptor'] for mode,r in paths.items()},scientific_acceptance=False)
    # A fresh real owner, same physical model but incompatible immutable schedule.
    # Admission must reject BEFORE evaluating or publishing a restored state.
    from dataclasses import replace
    from examples.general_contact_enriched_d1_path import build
    from anyfesolver.general_contact.enriched_path import EnrichedPathOwner
    model,definition,fresh,_=build('displacement')
    points=list(fresh.program.waypoints);last=list(points[-1]);last[0]+=.001;points[-1]=tuple(last)
    changed=replace(fresh.program,waypoints=tuple(points))
    incompatible=EnrichedPathOwner(model,definition,changed,quadrature=fresh.template.quadrature,
        force_scale_n=fresh.criteria.force_scale_n,physical_force_scale_n=.018)
    try:incompatible.restore(raw,expected_sha256=sha)
    except EnrichedProfileError:pass
    else:raise EnrichedProfileError('incompatible authentic prefix programme was accepted')
    if incompatible._publication is not None or incompatible._busy:
        raise EnrichedProfileError('incompatible programme refusal published state')
    try:fresh.restore(raw+b' ',expected_sha256=sha)
    except EnrichedProfileError:pass
    else:raise EnrichedProfileError('changed prefix bytes were accepted with old authentication')
    evidence.update(actual_active_prefix_cancel_atomic=True,actual_accepted_prefix_mutation_refused=True,
        actual_late_post_solve_cancel_atomic=True,actual_uncommitted_candidate=late_candidate,
        actual_incompatible_programme_refused=True,changed_prefix_bytes_refused=True)
    publish(output/'PATH_ASSESSMENT.json',evidence)
    evidence.update(status='PASS_SCOPED_V16_PUBLIC_PATH_DELIVERY_ONLY',
        extended_full364_work_reference='PREPARED_REQUIRED_OPEN_UNRUN',
        smooth_branch_tangent='REQUIRED_OPEN_UNRUN',pressure_stability='REQUIRED_OPEN_UNRUN')
    publish(output/'ASSESSMENT.json',evidence)
    if not include_extended_reference:return evidence
    # Separately adopted future gate: one force endpoint plus its new response.
    # No accidental 22-pose multiplication of complete derivative/reference work.
    assessment=assess_endpoint(owners['force'],output=output/'force-endpoint-reference',cancel=cancel)
    publish(output/'force-endpoint-reference/ASSESSMENT.json',assessment)
    if assessment['status']!='PASS' or not all(v is True for v in assessment['gates'].values()):
        raise EnrichedProfileError('complete original endpoint predicate families NONPASS')
    evidence.update(status='PASS_SCOPED_V16_PATH_AND_ENDPOINT_ONLY',endpoint_assessment='force-endpoint-reference/ASSESSMENT.json',
        broader_D1='OPEN: pressure stability, full tangent/work breadth, independent review/install/CI/release')
    publish(output/'ASSESSMENT.json',evidence)
    return evidence
