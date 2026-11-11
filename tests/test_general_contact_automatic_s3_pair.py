"""Bounded automatic S3 pair, coarse-lower integration / upper actuation."""
from dataclasses import asdict,replace
import copy,hashlib,json,os
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.general_contact import ContactInteraction,resolve_automatic_contact_domain as resolve



def _independent_pose_gap(model,u):
    """Exact rational projected S3 polygons; squared radical gap criterion."""
    points={}
    for node in range(1,11):
        n=model.mesh.get_node(node);d=model.mesh.dof_manager.get_node_dofs(node)
        assert np.all(u[d[3:]]==0.) and u[d[1]]==0.
        if node>6:assert u[d[0]]==0.
        offset=-.025 if node<=6 else float(.025-.01)
        points[node]=tuple(F(float(v)) for v in
            (float(n.x+u[d[0]]),float(n.y+u[d[1]]),float(float(n.z+offset)+u[d[2]])))
    def sub(a,b):return tuple(x-y for x,y in zip(a,b))
    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def turn(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    gaps=[];areas=[];polygons=[];positive=True;tilts=[]
    for ids in ((7,10,8),(8,10,9)):
        source=[points[n] for n in ids];N=cross(sub(source[1],source[0]),sub(source[2],source[0]));nn=dot(N,N)
        assert N[2]<0 and nn>0 and turn(*source)<0
        tilts.append(float(N[0]*N[0]+N[1]*N[1]))
        for tids in ((1,5,2),(1,4,5),(2,6,3),(2,5,6)):
            target=[points[n] for n in tids];assert len({q[2] for q in target})==1
            projected=[]
            for q in target:
                factor=dot(sub(q,source[0]),N)/nn
                projected.append(tuple(q[k]-factor*N[k] for k in range(3)))
            assert turn(*projected)<0
            poly=[q[:2] for q in projected]
            for j in range(3):
                a,b=source[j],source[(j+1)%3];out=[]
                if not poly:break
                previous=poly[-1];dp=-turn(a,b,previous)
                for current in poly:
                    dc=-turn(a,b,current)
                    if (dc>=0)!=(dp>=0):
                        t=dp/(dp-dc);out.append(tuple(previous[k]+t*(current[k]-previous[k]) for k in range(2)))
                    if dc>=0:out.append(current)
                    previous,dp=current,dc
                poly=out
            if len(poly)<3:continue
            area=-sum(turn((0,0),poly[j],poly[(j+1)%len(poly)]) for j in range(len(poly)))/2
            if area==0:continue
            assert area>0;areas.append(area);pair_gaps=[];coefficients=[];hats=[]
            for xy in poly:
                z=(dot(N,source[0])-N[0]*xy[0]-N[1]*xy[1])/N[2]
                C=(z-target[0][2])/N[2]
                if C<0:assert C*C*nn<=F(5e-5)**2
                positive=positive and C>0
                coefficients.append(C)
                area_source=turn(*source)
                b=turn(source[0],xy,source[2])/area_source;c=turn(source[0],source[1],xy)/area_source
                hats.append((1-b-c,b,c))
                gap=float(C)*float(nn)**.5;gaps.append(gap);pair_gaps.append(gap)
            support=[F(0)]*3;weighted=[F(0)]*3
            for j in range(1,len(poly)-1):
                js=(0,j,j+1);piece=-turn(poly[0],poly[j],poly[j+1])/2
                for n in range(3):
                    values=[hats[k][n] for k in js];cs=[coefficients[k] for k in js]
                    support[n]+=piece*sum(values)/3
                    weighted[n]+=piece*(sum(values)*sum(cs)+sum(a*b for a,b in zip(values,cs)))/12
            polygons.append(dict(reference_support=[float(v) for v in support],reference_weighted_gap=[float(v)*float(nn)**.5 for v in weighted],source_nodes=ids,target_nodes=tids,polygon_xy_m=[[float(v) for v in q] for q in poly],minimum_gap_m=min(pair_gaps)))
    assert gaps
    minimum=min(gaps)
    return dict(kind='exact_rational_projection_and_squared_radical_gap_gate',minimum_gap_m=minimum,
        approximate_display_only=True,
        strict_positive=positive,tilted_source_normal_squared_xy=max(tilts),polygons=polygons,projected_xy_reference_area_m2=float(sum(areas)),
        area_numerator=sum(areas).numerator,area_denominator=sum(areas).denominator,
        interpretation='binary pose; display radical rounded; exact squared criterion; not swept/map-error qualification')


def _footprint(model,resolution,settings,u):
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.overlap import PlanarFace,planar_projected_overlap
    manager=GeneralS3PatchManager(model,resolution.definition,settings);source,target=manager._positions(u)
    pairs=[]
    for si in manager.law.source_face_indices:
        points=source[si];normal=np.cross(points[1]-points[0],points[2]-points[0]);normal/=np.linalg.norm(normal)
        for ti in manager.law.target_face_indices:
            projected=target[ti]-((target[ti]-points[0])@normal)[:,None]*normal
            overlap=planar_projected_overlap(PlanarFace((1,1),points),PlanarFace((2,1),projected))
            if overlap is not None:pairs.append(dict(polygon_m=overlap.polygon,area_m2=overlap.area))
    return dict(area_m2=sum(p['area_m2'] for p in pairs),pairs=pairs,
        physical_source_nodes_m=source,physical_target_nodes_m=target)


@pytest.mark.skipif(os.environ.get('ANY_RUN_AUTOMATIC_S3_PAIR')!='1',
                   reason='bounded installed refined automatic deformable lifecycle')
def test_installed_automatic_s3_pair_stage():
    """One guarded stage per invocation, sharing only retained JSON evidence."""
    import anyfesolver,numba
    from threadpoolctl import threadpool_info
    from anyfesolver import ContactDefinition,solve_static_nonlinear
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from examples.general_contact_automatic_s3_pair import build_s3_pair,solve_shell_pair
    stage=os.environ['AUTOMATIC_S3_PAIR_STAGE'].split('-')[0]
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    output=Path(os.environ['AUTOMATIC_DOMAIN_RESULT'])
    model,domain,settings,resolution,path=build_s3_pair()
    report=dict(stage=stage,identity=resolution.identity,selections=[asdict(x) for x in resolution.selections],
        domain=domain.to_dict(),settings=asdict(settings),criteria=dict(
            force_scale_n=.02,moment_scale_nm=.02,virtual_translation_m=1e-6,
            force_moment_normalized_tolerance=1e-6,gap_budget_m=5e-5,
            displacement_parity_rtol=1e-7,displacement_parity_atol_m=1e-9,
            scale_basis='unchanged positive development scales k1000*area1*20um and lever1m',
            lateral_control_atol_m=1e-10,all_upper_lateral_motion_min_m=.01,
            footprint_area_change_min_m2=1e-6,opening_clearance="strict_exact_rational_positive",
            target_return_atol_m=1e-10),
        physical_inputs=dict(integration_role='lower_coarser',controlled_physical_body='upper',source_work_labels='physical_upper_not_integration_role',upper_actuator_midplane_z_m=.055,upper_physical_bottom_offset_m=-.025,lower_pressure_midplane_z_m=0.,
            lower_physical_top_offset_m=float(.025-.01),upper_director=(0.,0.,-1.),lower_director=(0.,0.,-1.),
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
            ContactInteraction('ordinary-explicit','lower','upper',1000.,5e-5,
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
    report['upper_actuator_ux_m']=result.displacements[source_x].tolist()
    assert abs(result.displacements[source_x[0]]-.02)<=1e-10
    assert np.all(result.displacements[source_x]>.01)
    for index,row in enumerate(phases,1):
        np.testing.assert_allclose(row['target'],path.target_at(index),rtol=0.,atol=1e-10)
    expected_inventory={'full':[1,2,3,4],'explicit':[1,2,3,4],'prefix':[1,2,3],'resume':[4]}[stage]
    assert [x.step_index for x in result.snapshots]==expected_inventory
    report['accepted_snapshots']=[dict(step=x.step_index,upper_actuator_ux_m=x.displacements[source_x],
        target_uz_m=x.displacements[model.mesh.dof_manager.get_node_dofs(9)[2]],
        footprint=_footprint(model,resolution,settings,x.displacements),
        independent_gap=_independent_pose_gap(model,x.displacements)) for x in result.snapshots]
    save()
    if stage in ('full','explicit','prefix'):
        by_step={x['step']:x for x in report['accepted_snapshots']}
        assert abs(by_step[1]['target_uz_m'])>1e-10
        assert by_step[1]['independent_gap']['tilted_source_normal_squared_xy']>1e-20
        assert np.all(np.asarray(by_step[2]['upper_actuator_ux_m'])>.01)
        assert abs(by_step[2]['footprint']['area_m2']-by_step[1]['footprint']['area_m2'])>1e-6
        assert abs(by_step[2]['independent_gap']['projected_xy_reference_area_m2']-by_step[1]['independent_gap']['projected_xy_reference_area_m2'])>1e-6
        opened=by_step[3]
        assert opened['independent_gap']['strict_positive']
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
        free_unknowns=[model.mesh.dof_manager.get_node_dofs(n)[0] for n in range(2,7)]+[target_dof]
        report['all_free_equilibrium_normalized_error']=float(np.max(np.abs((structural+residual)[free_unknowns]))/.02)
        save()
        assert report['all_free_equilibrium_normalized_error']<=1e-6
        assert abs(report['actuator_reaction_n'])>1e-6
        assert abs(report['target_support_reaction_n'])>1e-6
        assert abs(report['actuator_reaction_n']+report['target_support_reaction_n'])/.02<=1e-6
        source_probe=np.zeros(manager.n_dof);target_probe=np.zeros(manager.n_dof)
        for node in range(1,7):source_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
        for node in range(7,11):target_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
        ws=float(residual@source_probe);wt=float(residual@target_probe)
        source_work=float(residual@(result.displacements*(source_probe!=0)))
        target_work=float(residual@(result.displacements*(target_probe!=0)))
        report['both_body_work']=dict(upper_virtual_work_j=ws,lower_virtual_work_j=wt,
            common_translation_normalized_error=abs(ws+wt)/(.02*1e-6),
            upper_residual_dot_displacement_j=source_work,lower_residual_dot_displacement_j=target_work,
            interpretation='instantaneous residual work; not path-integrated work or regularization storage')
        save()
        assert abs(ws)>1e-12 and abs(wt)>1e-12
        assert abs(ws+wt)/(.02*1e-6)<=1e-6
        assert abs(source_work)>1e-12 and abs(target_work)>1e-12
        np.testing.assert_allclose(ws,-contact_result.target_force[2]*1e-6,rtol=1e-6,atol=2e-14)
        np.testing.assert_allclose(wt,-contact_result.integration_force[2]*1e-6,rtol=1e-6,atol=2e-14)
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


def test_s3_orientation_work_map_and_refusal():
    from examples.general_contact_automatic_s3_pair import build_s3_pair
    from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
    down=build_s3_pair(resolve_domain=False);up=build_s3_pair(director_z=1,resolve_domain=False)
    for eid in range(1,7):
        a=S3SkinWorkAdapter.from_model(down[0],1,eid,'top' if eid<=4 else 'bottom')
        b=S3SkinWorkAdapter.from_model(up[0],1,eid,'bottom' if eid<=4 else 'top')
        permutation=[b.node_ids.index(n) for n in a.node_ids]
        rows=[3*j+k for j in permutation for k in range(3)]
        np.testing.assert_array_equal(a.reference_positions,b.reference_positions[permutation])
        np.testing.assert_array_equal(a.work_map.toarray(),b.work_map.toarray()[rows])
        u=np.arange(a.work_map.shape[1])*1e-7
        np.testing.assert_array_equal(a.current_positions(u),b.current_positions(u)[permutation])
    with pytest.raises(ValueError):resolve(up[0],up[1],up[2],analysis_control='contact_path')


def test_s3_smooth_full_work_tangent_and_unsafe_path():
    from examples.general_contact_automatic_s3_pair import build_s3_pair
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from scipy import sparse
    model,domain,settings,resolution,path=build_s3_pair()
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    u=np.zeros(manager.n_dof);direction=u.copy()
    for n in range(1,7):
        dofs=model.mesh.dof_manager.get_node_dofs(n);u[dofs[0]]=.02;u[dofs[2]]=-.015005
        direction[dofs[0]]=.6;direction[dofs[2]]=.4
    dof=model.mesh.dof_manager.get_node_dofs(9)[2];u[dof]=-1e-6;direction[dof]=-.2
    direction/=np.linalg.norm(direction)
    manager.restrict_tangent_to_reduced_variations(sparse.csr_matrix(direction[:,None]))
    force,tangent,result=manager.evaluate(u)
    oracle=_independent_pose_gap(model,u)
    assert oracle['tilted_source_normal_squared_xy']>1e-13
    from anyfesolver.general_contact.s3_overlap_derivatives import s3_overlap_derivatives
    from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
    reference={};current={}
    for eid in range(1,7):
        adapter=S3SkinWorkAdapter.from_model(model,1,eid,'top' if eid<=4 else 'bottom')
        for n,ref,cur in zip(adapter.node_ids,adapter.reference_positions,adapter.current_positions(u)):
            reference[n]=ref;current[n]=cur
    kernel_errors=[]
    for pair in oracle['polygons']:
        ids=pair['source_nodes'];tids=pair['target_nodes']
        local=s3_overlap_derivatives(np.array([reference[n] for n in ids]),np.array([current[n] for n in ids]),np.array([current[n] for n in tids]))
        support_error=float(np.linalg.norm(local.reference_support-pair['reference_support']))
        gap_error=float(np.linalg.norm(local.reference_weighted_gap-pair['reference_weighted_gap']))
        assert support_error<=1e-12 and gap_error<=1e-12
        kernel_errors.append(dict(support_absolute_m2=support_error,weighted_gap_absolute_m3=gap_error))
    errors=[]
    for h in (1e-7,3e-8):
        plus=manager.evaluate(u+h*direction,tangent=False)[0];minus=manager.evaluate(u-h*direction,tangent=False)[0]
        finite=(plus-minus)/(2*h);exact=tangent@direction
        errors.append(float(np.linalg.norm(finite-exact)/max(np.linalg.norm(exact),1e-12)))
    unsafe=u.copy()
    for n in range(1,7):unsafe[model.mesh.dof_manager.get_node_dofs(n)[2]]=-.02
    assert not manager._swept_admissible(np.zeros_like(u),unsafe)
    output=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if output:Path(output).write_text(json.dumps(dict(tangent_errors=errors,independent_gap=oracle,unsafe_endpoint_path_refused=True,kernel_scalar_integral_errors=kernel_errors),indent=2))
    assert max(errors)<=1e-5,errors


def test_s3_predictor_route_and_precancel_safety():
    from anyfesolver import CancellationToken,SolveCancelled,validate_general_contact
    from examples.general_contact_automatic_s3_pair import build_s3_pair,solve_shell_pair
    model,domain,settings,resolution,path=build_s3_pair()
    assert not validate_general_contact(model,resolution.definition,settings,analysis_control='force').valid
    # Current typed refusal (static_path.py): the invalid friction object is
    # rejected by the S3SlidingFrictionSettings/projected-penalty profile gate.
    with pytest.raises(ValueError,match='physical-side/predicted friction path requires S3SlidingFrictionSettings'):
        solve_shell_pair(model,domain,settings,resolution,path,contact_friction=object())
    before=repr(model);token=CancellationToken();token.cancel('S3 pair public path precancel')
    with pytest.raises(SolveCancelled):solve_shell_pair(model,domain,settings,resolution,path,cancellation_token=token)
    assert repr(model)==before
