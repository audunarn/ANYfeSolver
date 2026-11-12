"""Bounded installed guided-source press/20mm slide/open/recontact."""
from dataclasses import asdict,replace
import copy,hashlib,json,os
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.general_contact import ContactInteraction,resolve_automatic_contact_domain as resolve



def _independent_pose_gap(model,u):
    """Exact rational vertical gap of independently constructed binary poses.

    Fixed directors/rotations make source bottom z=.055-.025+uz and target
    top z=(.025-.01)+uz. Construct the stored binary pose with scalar additions,
    then clip Q4 source polygons against P1 target triangles using Fractions.
    Height is affine, so its extrema occur at clipped vertices. This is an
    accepted-pose oracle, not a swept-path certificate.
    """
    points={}
    for node in range(1,11):
        n=model.mesh.get_node(node);d=model.mesh.dof_manager.get_node_dofs(node)
        assert np.all(u[d[3:]]==0.) and u[d[1]]==0.
        if node>6:assert u[d[0]]==0.
        assert n.z==(.055 if node<=6 else 0.)
        offset=-.025 if node<=6 else float(.025-.01)
        points[node]=tuple(F(float(v)) for v in
            (float(n.x+u[d[0]]),float(n.y+u[d[1]]),float(float(n.z+offset)+u[d[2]])))
    return _exact_clip_gap(points)


def _exact_clip_gap(points):
    source_faces=((1,2,5,4),(2,3,6,5));target_faces=((7,8,10),(8,9,10))
    def turn(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    minima=[];polygons=[];areas=[]
    for ids in source_faces:
        source=[points[n] for n in ids]
        assert len({q[2] for q in source})==1
        assert all(turn(source[j],source[(j+1)%4],source[(j+2)%4])>0 for j in range(4))
        for tids in target_faces:
            target=[points[n] for n in tids];area=turn(*target);assert area>0
            poly=[q[:2] for q in source]
            for j in range(3):
                a,b=target[j],target[(j+1)%3];out=[]
                if not poly:break
                previous=poly[-1];dp=turn(a,b,previous)
                for current in poly:
                    dc=turn(a,b,current)
                    if (dc>=0)!=(dp>=0):
                        t=dp/(dp-dc)
                        out.append(tuple(previous[k]+t*(current[k]-previous[k]) for k in range(2)))
                    if dc>=0:out.append(current)
                    previous,dp=current,dc
                poly=out
            if len(poly)<3 or sum(turn((0,0),poly[j],poly[(j+1)%len(poly)]) for j in range(len(poly)))==0:continue
            exact_area=sum(turn((0,0),poly[j],poly[(j+1)%len(poly)]) for j in range(len(poly)))/2
            assert exact_area>0
            areas.append(exact_area)
            gaps=[]
            for q in poly:
                b=turn(target[0],q,target[2])/area;c=turn(target[0],target[1],q)/area
                assert b>=0 and c>=0 and b+c<=1
                height=(1-b-c)*target[0][2]+b*target[1][2]+c*target[2][2]
                gaps.append(source[0][2]-height)
            minima.extend(gaps)
            polygons.append(dict(source_nodes=ids,target_nodes=tids,
                polygon_xy_m=[[float(v) for v in q] for q in poly],minimum_gap_m=float(min(gaps))))
    assert minima
    minimum=min(minima)
    assert minimum>=-F(5e-5)
    return dict(kind='exact_rational_gap_of_independently_constructed_binary_pose',
        minimum_gap_m=float(minimum),lower_m=float(np.nextafter(float(minimum),-np.inf)),
        upper_m=float(np.nextafter(float(minimum),np.inf)),numerator=minimum.numerator,
        denominator=minimum.denominator,polygons=polygons,fixed_directors=True,
        overlap_area_m2=float(sum(areas)),area_numerator=sum(areas).numerator,
        area_denominator=sum(areas).denominator,
        interpretation='stored binary accepted pose; not full map-roundoff or swept coverage qualification')

def _footprint(model,resolution,settings,u):
    """Report actual projected pair polygons, not a new admission certificate."""
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.q4_source_overlap import projected_q4_source_face
    from anyfesolver.general_contact.overlap import PlanarFace,planar_projected_overlap
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    source,target=manager._positions(u)
    pairs=[]
    for si in manager.law.source_face_indices:
        face,normal=projected_q4_source_face(source[si])
        for ti in manager.law.target_face_indices:
            points=target[ti]
            projected=points-((points-source[si][0])@normal)[:,None]*normal
            overlap=planar_projected_overlap(face,PlanarFace((2,1),projected))
            if overlap is not None:
                pairs.append(dict(source_indices=list(map(int,si)),target_indices=list(map(int,ti)),
                    polygon_m=overlap.polygon,area_m2=overlap.area))
    return dict(area_m2=sum(p['area_m2'] for p in pairs),pairs=pairs,
        physical_source_nodes_m=source,physical_target_nodes_m=target,
        nominal_undeformed_rectangle_area_m2=.792 if u[model.mesh.dof_manager.get_node_dofs(1)[0]]>.01 else .81)


@pytest.mark.skipif(os.environ.get('ANY_RUN_AUTOMATIC_SLIDING')!='1',
                   reason='bounded installed refined automatic deformable lifecycle')
def test_installed_automatic_sliding_stage():
    """One guarded stage per invocation, sharing only retained JSON evidence."""
    import anyfesolver,numba
    from threadpoolctl import threadpool_info
    from anyfesolver import ContactDefinition,solve_static_nonlinear
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from examples.general_contact_automatic_domain import build_projected_shell_pair,solve_shell_pair
    stage=os.environ['AUTOMATIC_SLIDING_STAGE'].split('-')[0]
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    output=Path(os.environ['AUTOMATIC_DOMAIN_RESULT'])
    model,domain,settings,resolution,path=build_projected_shell_pair(lifecycle=True,sliding=True,
        path_predictor=os.environ.get('AUTOMATIC_PATH_PREDICTOR','none'))
    report=dict(stage=stage,identity=resolution.identity,selections=[asdict(x) for x in resolution.selections],
        domain=domain.to_dict(),settings=asdict(settings),criteria=dict(
            force_scale_n=.02,moment_scale_nm=.02,virtual_translation_m=1e-6,
            force_moment_normalized_tolerance=1e-6,gap_budget_m=5e-5,
            displacement_parity_rtol=1e-7,displacement_parity_atol_m=1e-9,
            scale_basis='unchanged positive development scales k1000*area1*20um and lever1m',
            lateral_control_atol_m=1e-10,all_source_lateral_motion_min_m=.01,
            footprint_area_change_min_m2=1e-6,opening_clearance="strict_exact_rational_positive",
            target_return_atol_m=1e-10),
        physical_inputs=dict(source_midplane_z_m=.055,source_bottom_offset_m=-.025,
            target_top_offset_m=float(.025-.01),source_director=(0.,0.,1.),target_director=(0.,0.,1.),
            path=path.contract()),
        runtime_before_solve=dict(anyfesolver_path=anyfesolver.__file__,
            automatic_domain_path=__import__('inspect').getfile(resolve),
            jit_disabled=bool(numba.config.DISABLE_JIT),numba_threads=int(numba.get_num_threads()),
            pools=threadpool_info()))
    def save():
        output.write_text(json.dumps(report,indent=2,
            default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
    save()
    assert report['runtime_before_solve']['jit_disabled']
    assert report['runtime_before_solve']['numba_threads']==1
    assert all(pool['num_threads']==1 for pool in report['runtime_before_solve']['pools'])
    controls=dict(emit_restart_checkpoint=True,record_increment_snapshots=True)
    prefix_file='prefix.json'
    full_file='full.json'
    if stage=='prefix':controls['max_load_factor']=3/4
    if stage=='resume':
        prefix=json.loads((evidence/prefix_file).read_text(encoding='utf-8'))
        assert prefix['identity']==resolution.identity and prefix['verified']
        assert prefix['opening_verified']
        controls['restart_checkpoint']=prefix['checkpoint']
    if stage=='refusal':
        prefix=json.loads((evidence/prefix_file).read_text(encoding='utf-8'))
        before=repr(model)
        changed=replace(domain,name='changed-shell-provenance')
        changed_resolution=resolve(model,changed,settings,analysis_control='contact_path' if settings.path_predictor!='none' else 'displacement')
        with pytest.raises(ValueError,match='incompatible'):
            solve_shell_pair(model,changed,settings,changed_resolution,path,
                restart_checkpoint=prefix['checkpoint'])
        assert repr(model)==before
        if settings.path_predictor!='none':
            changed_settings=replace(settings,path_predictor='none')
            changed_resolution=resolve(model,domain,changed_settings,analysis_control='displacement')
            with pytest.raises(ValueError,match='incompatible'):
                solve_shell_pair(model,domain,changed_settings,changed_resolution,path,
                    restart_checkpoint=prefix['checkpoint'])
            assert repr(model)==before
            report['predictor_refusal']=True
        tampered=copy.deepcopy(prefix['checkpoint'])
        tampered['displacements'][0]+=1e-6
        with pytest.raises(ValueError,match='SHA-256 does not match its contents'):
            solve_shell_pair(model,domain,settings,resolution,path,restart_checkpoint=tampered)
        assert repr(model)==before
        report.update(verified=True,provenance_refusal=True,declared_repr_unchanged=True,tamper_refusal=True);save();return
    if stage=='explicit':
        definition=ContactDefinition(resolution.definition.surfaces,(),(
            ContactInteraction('ordinary-explicit','q4_source','s3_target',1000.,5e-5,
                               normal_law='projected_penalty'),))
        result=solve_static_nonlinear(model,contact=definition,contact_settings=settings,
            contact_path_program=path,max_iterations=15,**controls)
    else:result=solve_shell_pair(model,domain,settings,resolution,path,**controls)
    report.update(status=result.status,failure_reason=result.failure_reason,
        contact=result.info.get('general_contact'),phases=result.info.get('force_displacement_history'),
        displacements=result.displacements)
    if result.restart_checkpoint is not None:report['checkpoint']=result.to_restart_checkpoint()
    save()
    assert result.status=='completed',result.failure_reason
    phases=report['phases']
    source_x=[model.mesh.dof_manager.get_node_dofs(n)[0] for n in range(1,7)]
    report['source_ux_m']=result.displacements[source_x].tolist()
    assert abs(result.displacements[source_x[0]]-.02)<=1e-10
    assert np.all(result.displacements[source_x]>.01)
    for index,row in enumerate(phases,1):
        np.testing.assert_allclose(row['target'],path.target_at(index),rtol=0.,atol=1e-10)
    expected_inventory={'full':[1,2,3,4],'explicit':[1,2,3,4],'prefix':[1,2,3],'resume':[4]}[stage]
    assert [x.step_index for x in result.snapshots]==expected_inventory
    report['accepted_snapshots']=[dict(step=x.step_index,source_ux_m=x.displacements[source_x],
        target_uz_m=x.displacements[model.mesh.dof_manager.get_node_dofs(9)[2]],
        footprint=_footprint(model,resolution,settings,x.displacements),
        independent_gap=_independent_pose_gap(model,x.displacements)) for x in result.snapshots]
    save()
    if stage in ('full','explicit','prefix'):
        by_step={x['step']:x for x in report['accepted_snapshots']}
        assert abs(by_step[1]['target_uz_m'])>1e-10
        assert np.all(np.asarray(by_step[2]['source_ux_m'])>.01)
        assert abs(by_step[2]['footprint']['area_m2']-by_step[1]['footprint']['area_m2'])>1e-6
        assert abs(by_step[2]['independent_gap']['overlap_area_m2']-by_step[1]['independent_gap']['overlap_area_m2'])>1e-6
        opened=by_step[3]
        assert F(opened['independent_gap']['numerator'],opened['independent_gap']['denominator'])>0
        assert opened['independent_gap']['lower_m']>0.
        assert abs(opened['target_uz_m'])<=1e-10
        assert phases[2]['contact']['regularization_storage_j']==0.
        report['opening_verified']=True;save()
    if settings.path_predictor!='none':
        assert result.info['contact_path']['path_predictor']==settings.path_predictor
        report['executed_path_predictor']=result.info['contact_path']['path_predictor'];save()
    expected={'full':[True,True,False,True],'explicit':[True,True,False,True],
              'prefix':[True,True,False],'resume':[True,True,False,True]}[stage]
    assert [row['contact']['active_pressure_dofs']>0 for row in phases]==expected
    wrench=[]
    for row,active in zip(phases,expected):
        contact=row['contact']
        errors=dict(force=float(np.linalg.norm(np.asarray(contact['integration_force'])+contact['target_force'])/.02),
                    moment=float(np.linalg.norm(np.asarray(contact['integration_moment'])+contact['target_moment'])/.02))
        wrench.append(errors)
        assert max(errors.values())<=1e-6
        assert contact['wrench_origin_m']==[0.,0.,0.]
        assert contact['gap_tolerance_met'] and contact['minimum_gap']>=-5e-5
        assert contact['contact_work_convention']=='projected_weak_gap_nonpotential_v1'
        assert (contact['regularization_storage_j']>0.)==active
        assert contact['penalty_energy']==0.
    report['phase_wrench_errors']=wrench;save()
    target_dof=model.mesh.dof_manager.get_node_dofs(9)[2]
    report['target_node9_uz_m']=float(result.displacements[target_dof])
    if stage!='prefix':
        assert abs(result.displacements[target_dof])>1e-10
        manager=GeneralS3PatchManager(model,resolution.definition,settings)
        residual,_,contact_result=manager.evaluate(result.displacements,tangent=False)
        from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
        structural,_,states=_assemble_nonlinear_system(model,result.displacements,result.element_states,
            5,tangent=False,kinematics='von_karman')
        _discard_nonlinear_state_candidate(states)
        equilibrium=(structural+residual)[target_dof]
        report['target_equilibrium_normalized_error']=abs(float(equilibrium))/.02
        report['support_reactions']=result.steps[-1].support_reactions
        report['actuator_reaction_n']=float(np.sum((structural+residual)[
            [model.mesh.dof_manager.get_node_dofs(n)[2] for n in range(1,7)]]))
        report['target_support_reaction_n']=float(np.sum((structural+residual)[
            [model.mesh.dof_manager.get_node_dofs(n)[2] for n in (7,8,10)]]))
        save()
        assert abs(equilibrium)/.02<=1e-6
        assert abs(report['actuator_reaction_n'])>1e-6
        assert abs(report['target_support_reaction_n'])>1e-6
        assert abs(report['actuator_reaction_n']+report['target_support_reaction_n'])/.02<=1e-6
        source_probe=np.zeros(manager.n_dof);target_probe=np.zeros(manager.n_dof)
        for node in range(1,7):source_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
        for node in range(7,11):target_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
        ws=float(residual@source_probe);wt=float(residual@target_probe)
        source_work=float(residual@(result.displacements*(source_probe!=0)))
        target_work=float(residual@(result.displacements*(target_probe!=0)))
        report['both_body_work']=dict(source_virtual_work_j=ws,target_virtual_work_j=wt,
            common_translation_normalized_error=abs(ws+wt)/(.02*1e-6),
            source_residual_dot_displacement_j=source_work,target_residual_dot_displacement_j=target_work,
            interpretation='instantaneous residual work; not path-integrated work or regularization storage')
        save()
        assert abs(ws)>1e-12 and abs(wt)>1e-12
        assert abs(ws+wt)/(.02*1e-6)<=1e-6
        assert abs(source_work)>1e-12 and abs(target_work)>1e-12
        np.testing.assert_allclose(ws,-contact_result.integration_force[2]*1e-6,rtol=1e-6,atol=2e-14)
        np.testing.assert_allclose(wt,-contact_result.target_force[2]*1e-6,rtol=1e-6,atol=2e-14)
    if stage in ('explicit','resume'):
        full=json.loads((evidence/full_file).read_text(encoding='utf-8'))
        assert full['identity']==resolution.identity and full['verified']
        np.testing.assert_allclose(result.displacements,full['displacements'],rtol=1e-7,atol=1e-9)
        response_errors={}
        for key,scale in (('integration_force',.02),('target_force',.02),
                          ('integration_moment',.02),('target_moment',.02),
                          ('minimum_gap',5e-5),('regularization_storage_j',2e-7)):
            error=float(np.linalg.norm(np.asarray(report['contact'][key])-full['contact'][key])/scale)
            response_errors[key]=error
            assert error<=1e-6
        report['normalized_endpoint_response_parity']=response_errors
        phase_errors=[]
        for current,reference in zip(phases,full['phases']):
            errors={}
            assert current['contact']['active_pressure_dofs']==reference['contact']['active_pressure_dofs']
            for key,scale in (('integration_force',.02),('target_force',.02),('integration_moment',.02),
                              ('target_moment',.02),('minimum_gap',5e-5),('regularization_storage_j',2e-7)):
                errors[key]=float(np.linalg.norm(np.asarray(current['contact'][key])-reference['contact'][key])/scale)
                assert errors[key]<=1e-6
            phase_errors.append(errors)
        report['normalized_per_phase_response_parity']=phase_errors;save()
    assert report['checkpoint']
    report['verified']=True;save()


@pytest.mark.skipif(os.environ.get('ANY_RUN_AUTOMATIC_SLIDING')!='1',reason='bounded retained opening evidence')
def test_saved_full_has_genuine_opening():
    root=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']).parent
    full_file=root/'outputs/full.json'
    meta=json.loads((root/'attempt-01-full/result.json').read_text())
    assert hashlib.sha256(full_file.read_bytes()).hexdigest()==meta['result_sha256']
    snapshot=root/'attempt-01-full/inputs/tests/test_general_contact_automatic_sliding.py'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==meta['source_sha256']['tests/test_general_contact_automatic_sliding.py']
    full=json.loads(full_file.read_text())
    assert full['verified'] and full['status']=='completed'
    assert [x['step'] for x in full['accepted_snapshots']]==[1,2,3,4]
    areas={}
    for pose in full['accepted_snapshots']:
        area=F(0)
        for pair in pose['independent_gap']['polygons']:
            poly=[[F(float(v)) for v in q] for q in pair['polygon_xy_m']]
            area+=sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1]))/2
        assert area>0
        areas[pose['step']]=float(area)
    assert abs(areas[2]-areas[1])>1e-6
    opened=next(x for x in full['accepted_snapshots'] if x['step']==3)
    exact=F(opened['independent_gap']['numerator'],opened['independent_gap']['denominator'])
    assert exact>0 and opened['independent_gap']['lower_m']>0.
    assert abs(opened['target_uz_m'])<=1e-10
    contact=full['phases'][2]['contact']
    assert contact['active_pressure_dofs']==0 and contact['regularization_storage_j']==0.
    Path(os.environ['AUTOMATIC_DOMAIN_RESULT']).write_text(json.dumps(dict(verified=True,
        full_result_sha256=meta['result_sha256'],strict_clearance_m=float(exact),
        target_return_uz_m=opened['target_uz_m'],zero_storage=True,independent_binary_polygon_areas_m2=areas,
        interpretation='authenticated saved exact accepted-pose oracle; full solve not replayed'),indent=2))
