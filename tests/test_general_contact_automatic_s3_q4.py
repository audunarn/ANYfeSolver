"""Bounded installed guided-source press/20mm slide/open/recontact."""
from dataclasses import asdict,replace
import copy,hashlib,json,os
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.general_contact import ContactInteraction,resolve_automatic_contact_domain as resolve



from test_general_contact_automatic_q4_pair import _turn,_clip_exact,_bilinear_extrema

def _footprint(model,resolution,settings,u):
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.overlap import PlanarFace,planar_projected_overlap
    manager=GeneralS3PatchManager(model,resolution.definition,settings);source,target=manager._positions(u);pairs=[]
    for si in manager.law.source_face_indices:
        normal=np.cross(source[si][1]-source[si][0],source[si][2]-source[si][0]);normal/=np.linalg.norm(normal)
        projected=target-((target-source[si][0])@normal)[:,None]*normal
        overlap=planar_projected_overlap(PlanarFace((1,1),source[si]),PlanarFace((2,1),projected))
        assert overlap is not None
        pairs.append(dict(source_indices=list(map(int,si)),area_m2=overlap.area,polygon_m=overlap.polygon))
    return dict(area_m2=sum(p['area_m2'] for p in pairs),pairs=pairs,physical_source_nodes_m=source,physical_target_nodes_m=target)

@pytest.mark.skipif(os.environ.get('ANY_RUN_S3_Q4')!='1',
                   reason='bounded installed refined automatic deformable lifecycle')
def test_installed_automatic_s3_q4_stage(monkeypatch):
    """One guarded stage per invocation, sharing only retained JSON evidence."""
    import anyfesolver,numba
    from threadpoolctl import threadpool_info
    from anyfesolver import ContactDefinition,solve_static_nonlinear
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from examples.general_contact_automatic_s3_q4 import build,solve_shell_pair
    stage=os.environ['S3_Q4_STAGE'].split('-')[0]
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    output=Path(os.environ['AUTOMATIC_DOMAIN_RESULT'])
    model,domain,settings,resolution,path=build(lifecycle=True,sliding=True,
        path_predictor=os.environ.get('AUTOMATIC_PATH_PREDICTOR','control_projection_dev'))
    report=dict(stage=stage,identity=resolution.identity,selections=[asdict(x) for x in resolution.selections],
        domain=domain.to_dict(),settings=asdict(settings),criteria=dict(
            force_scale_n=.02,moment_scale_nm=.02,virtual_translation_m=1e-6,
            force_moment_normalized_tolerance=1e-6,gap_budget_m=5e-5,
            displacement_parity_rtol=1e-7,displacement_parity_atol_m=1e-9,
            scale_basis='unchanged positive development scales k1000*area1*20um and lever1m',
            lateral_control_atol_m=1e-10,all_source_lateral_motion_min_m=.01,
            pairwise_partition_area_change_min_m2=1e-6,opening_clearance="strict_exact_rational_positive",
            target_return_atol_m=1e-10),
        physical_inputs=dict(source_reference_z_m=.065,source_supported_reference_offset_m=.01,source_bottom_physical_offset_m=-.035,
            target_midplane_z_m=-.01,target_top_offset_m=.025,source_director=(0.,0.,1.),target_director=(0.,0.,1.),
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
    import anyfesolver.general_contact.s3_q4_swept as swept
    certificate=swept.s3_q4_affine_lower;enclosures=[]
    def recorded_certificate(*args,**kw):
        error=kw.get('endpoint_errors')
        if error is not None:enclosures.append(float(max(np.max(x) for x in error)))
        return certificate(*args,**kw)
    monkeypatch.setattr(swept,'s3_q4_affine_lower',recorded_certificate)
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
            ContactInteraction('ordinary-explicit','s3_source','q4_target',1000.,5e-5,
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
    source_x=[model.mesh.dof_manager.get_node_dofs(n)[0] for n in range(1,5)]
    report['source_ux_m']=result.displacements[source_x].tolist()
    assert abs(result.displacements[source_x[0]]-.02)<=1e-10
    assert np.all(result.displacements[source_x]>.01)
    for index,row in enumerate(phases,1):
        np.testing.assert_allclose(row['target'],path.target_at(index),rtol=0.,atol=1e-10)
    expected_inventory={'full':[1,2,3,4],'explicit':[1,2,3,4],'prefix':[1,2,3],'resume':[4]}[stage]
    assert [x.step_index for x in result.snapshots]==expected_inventory
    report['accepted_snapshots']=[dict(step=x.step_index,source_ux_m=x.displacements[source_x],
        target_uz_m=x.displacements[model.mesh.dof_manager.get_node_dofs(7)[2]],
        footprint=_footprint(model,resolution,settings,x.displacements),
        independent_gap=_independent_pose_gap(model,x.displacements)) for x in result.snapshots]
    save()
    if stage in ('full','explicit','prefix'):
        by_step={x['step']:x for x in report['accepted_snapshots']}
        assert abs(by_step[1]['target_uz_m'])>1e-10
        assert np.all(np.asarray(by_step[2]['source_ux_m'])>.01)
        assert abs(by_step[2]['footprint']['pairs'][0]['area_m2']-by_step[1]['footprint']['pairs'][0]['area_m2'])>1e-6
        assert abs(by_step[2]['independent_gap']['polygons'][0]['area_m2']-by_step[1]['independent_gap']['polygons'][0]['area_m2'])>1e-6
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
    target_dof=model.mesh.dof_manager.get_node_dofs(7)[2]
    report['target_node7_uz_m']=float(result.displacements[target_dof])
    if stage!='prefix':
        assert abs(result.displacements[target_dof])>1e-10
        manager=GeneralS3PatchManager(model,resolution.definition,settings)
        residual,_,contact_result=manager.evaluate(result.displacements,tangent=False)
        from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
        structural,_,states=_assemble_nonlinear_system(model,result.displacements,result.element_states,
            5,tangent=False,kinematics='von_karman')
        _discard_nonlinear_state_candidate(states)
        equilibrium=(structural+residual)[target_dof]
        report['all_free_equilibrium_normalized_error']=float(np.linalg.norm((structural+residual)[[model.mesh.dof_manager.get_node_dofs(n)[0] for n in range(2,5)]+[target_dof]])/.02)
        assert report['all_free_equilibrium_normalized_error']<=1e-6
        report['target_equilibrium_normalized_error']=abs(float(equilibrium))/.02
        report['support_reactions']=result.steps[-1].support_reactions
        report['actuator_reaction_n']=float(np.sum((structural+residual)[
            [model.mesh.dof_manager.get_node_dofs(n)[2] for n in range(1,5)]]))
        report['target_support_reaction_n']=float(np.sum((structural+residual)[
            [model.mesh.dof_manager.get_node_dofs(n)[2] for n in (5,6,8)]]))
        save()
        assert abs(equilibrium)/.02<=1e-6
        assert abs(report['actuator_reaction_n'])>1e-6
        assert abs(report['target_support_reaction_n'])>1e-6
        assert abs(report['actuator_reaction_n']+report['target_support_reaction_n'])/.02<=1e-6
        source_probe=np.zeros(manager.n_dof);target_probe=np.zeros(manager.n_dof)
        for node in range(1,5):source_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
        for node in range(5,9):target_probe[model.mesh.dof_manager.get_node_dofs(node)[2]]=1e-6
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
    report['whole_path_endpoint_enclosure_calls']=len(enclosures)
    report['maximum_endpoint_error_m']=max(enclosures)
    assert len(enclosures)>0 and max(enclosures)>0
    report['verified']=True;save()





def _independent_pose_gap(model,u):
    points={}
    for node in range(1,9):
        n=model.mesh.get_node(node);d=model.mesh.dof_manager.get_node_dofs(node)
        assert np.all(u[d[3:]]==0.) and u[d[1]]==0.
        if node>4:assert u[d[0]]==0.
        assert n.z==(.065 if node<=4 else -.01)
        offset=-.035 if node<=4 else .025
        points[node]=tuple(F(float(v)) for v in (n.x+u[d[0]],n.y+u[d[1]],float(n.z+offset)+u[d[2]]))
    target=[points[n] for n in (5,6,7,8)]
    assert [q[:2] for q in target]==[(0,0),(1,0),(1,1),(0,1)]
    z0,z1,z2,z3=[q[2] for q in target];a,b,c,d=z0,z1-z0,z3-z0,z0-z1+z2-z3
    minima=[];areas=[];polygons=[]
    for ids in ((1,2,3),(1,3,4)):
        source=[points[n] for n in ids];assert len({q[2] for q in source})==1 and _turn(*source)>0
        poly=_clip_exact([q[:2] for q in source],[q[:2] for q in target]);assert len(poly)>=3
        area=sum(p[0]*q[1]-p[1]*q[0] for p,q in zip(poly,poly[1:]+poly[:1]))/2;assert area>0
        lower,upper=_bilinear_extrema(poly,(source[0][2]-a,-b,-c,-d));minima.append(lower);areas.append(area)
        polygons.append(dict(source_nodes=ids,area_m2=float(area),minimum_gap_m=float(lower),polygon_xy_m=[[float(v) for v in q] for q in poly]))
    minimum=min(minima);assert minimum>=-F(5e-5)
    return dict(kind='exact_rational_horizontal_P1_source_original_Q1_target_binary_pose',minimum_gap_m=float(minimum),
        lower_m=float(np.nextafter(float(minimum),-np.inf)),numerator=minimum.numerator,denominator=minimum.denominator,
        strict_positive=minimum>0,overlap_area_m2=float(sum(areas)),polygons=polygons,
        interpretation='stored binary accepted pose; all edge/interior stationary extrema,not general map/swept qualification')


def _triangle_moment(vertices,i,j):
    """Exact simplex monomial moments in independent barycentric coordinates."""
    import math
    terms={(0,0,0):F(1)}
    for axis,count in ((0,i),(1,j)):
        for _ in range(count):
            out={}
            for powers,value in terms.items():
                for k in range(3):
                    new=list(powers);new[k]+=1;new=tuple(new);out[new]=out.get(new,F(0))+value*vertices[k][axis]
            terms=out
    area2=_turn(*vertices)
    return sum(value*area2*F(math.prod(math.factorial(n) for n in powers),math.factorial(sum(powers)+2)) for powers,value in terms.items())

def _exact_source_pair(source,target):
    source=[tuple(F(float(v)) for v in q[:2]) for q in source];target=[tuple(F(float(v)) for v in q[:2]) for q in target]
    poly=_clip_exact(source,target);assert len(poly)>=3
    area2=_turn(*source);support=[];gap=[]
    for k in range(3):
        a,b=source[(k+1)%3],source[(k+2)%3]
        value=lambda q:_turn(a,b,q)/area2
        constant=value((0,0));bx=value((1,0))-constant;by=value((0,1))-constant
        hat={(0,0):constant,(1,0):bx,(0,1):by}
        # Current horizontal source0 against original Q1height polynomial.
        height={(0,0):-F('.003'),(1,0):-F('.0004'),(0,1):-F('.0002'),(1,1):-F('.0003')}
        weighted={}
        for (i,j),v in hat.items():
            for (r,t),w in height.items():weighted[i+r,j+t]=weighted.get((i+r,j+t),F(0))+v*w
        def integral(p):return sum(v*_triangle_moment((poly[0],poly[k],poly[k+1]),i,j) for k in range(1,len(poly)-1) for (i,j),v in p.items())
        support.append(float(integral(hat)));gap.append(float(integral(weighted)))
    return np.array(support),np.array(gap)


def test_independent_scalar_shared_work_and_tangent():
    from anyfesolver.general_contact.s3_q4_overlap_derivatives import s3_q4_overlap_derivatives
    from anyfesolver.general_contact.s3_patch_mortar_energy import MovingS3PatchPrimalMortarPenalty,S3PatchFace
    from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
    source=np.array([[-.2,-.15,0.],[1.8,-.15,0.],[1.8,1.85,0.],[-.2,1.85,0.]])
    target_reference=np.array([[0.,0.,.003],[1.,0.,.003],[1.,1.,.003],[0.,1.,.003]])
    target=target_reference.copy();x,y=target[:,:2].T;target[:,2]=.003+.0004*x+.0002*y+.0003*x*y
    controls=dict(support_tolerance=1e-9,weighted_gap_tolerance=1e-10,support_gradient_tolerance=1e-5,
        support_hessian_tolerance=1e-5,weighted_gap_gradient_tolerance=1e-5,weighted_gap_hessian_tolerance=1e-5,maximum_triangles=128)
    expected_support=np.zeros(4);expected_gap=np.zeros(4);errors=[]
    for ids in ((1,2,3),(1,3,4)):
        indices=np.array(ids)-1;expected=_exact_source_pair(source[indices],target)
        actual=s3_q4_overlap_derivatives(source[indices],source[indices],target,**controls)
        np.testing.assert_allclose(actual.reference_support,expected[0],rtol=0,atol=1e-9)
        np.testing.assert_allclose(actual.reference_weighted_gap,expected[1],rtol=0,atol=1e-10)
        expected_support[indices]+=expected[0];expected_gap[indices]+=expected[1]
        errors.append(dict(support_error_m2=float(np.max(np.abs(actual.reference_support-expected[0]))),gap_error_m3=float(np.max(np.abs(actual.reference_weighted_gap-expected[1]))),evaluated_cells=actual.integration_triangles))
    geometry=MovingS3PatchPrimalMortarPenalty(tuple(S3PatchFace(ids,source[np.array(ids)-1]) for ids in ((1,2,3),(1,3,4))),
        (S3PatchFace((5,6,7,8),target_reference),),1000.,difference_step=2e-5,differentiation='jet2_s3_q4',
        q4_weighted_gap_tolerance=1e-10,maximum_q4_integration_triangles=128,q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,maximum_q4_source_integration_triangles=128,q4_source_derivative_tolerances=(1e-5,)*4,maximum_candidate_pairs=2)
    law=MovingS3PatchProjectedNormal(geometry);work=law.work(source,target,np.zeros(4))
    np.testing.assert_allclose(work.state.reference_support,expected_support,rtol=0,atol=1e-9)
    np.testing.assert_allclose(work.state.reference_weighted_gap,expected_gap,rtol=0,atol=1e-10)
    np.testing.assert_allclose(work.state.pressure,1000*np.maximum(-expected_gap/expected_support,0),rtol=0,atol=1e-6)
    assert np.all(work.state.reference_support>0)
    crossface=float(np.linalg.norm(work.tangent[3:6,9:12]));assert crossface>1e-10
    source[:,2]=.0001*source[:,0]+.0002*source[:,1]+.0003*source[:,0]*source[:,1]
    pose=np.concatenate((source,target));direction=np.random.default_rng(88).normal(size=pose.shape);direction/=np.linalg.norm(direction)
    work=law.work(source,target,np.zeros(4));tangent_errors=[]
    for h in (1e-7,3e-8):
        plus,minus=pose+h*direction,pose-h*direction
        fd=(law.work(plus[:4],plus[4:],np.zeros(4),tangent=False).residual-law.work(minus[:4],minus[4:],np.zeros(4),tangent=False).residual)/(2*h)
        action=work.tangent@direction.reshape(-1);error=float(np.linalg.norm(fd-action)/max(np.linalg.norm(action),1e-12));assert error<=1e-5;tangent_errors.append(error)
    translation=np.tile((0.,0.,1e-6),8);error=abs(float(work.residual@translation))/(.02*1e-6);assert error<=1e-6
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(verified=True,scalar=errors,independent_shared_sums_and_pressure=True,
        source_pressure_node_ids=list(law.source_node_ids),cross_face_tangent_block_norm=crossface,both_body_directional_tangent_errors=tangent_errors,
        common_translation_work_normalized=error),indent=2),encoding='utf-8')

def test_public_transient_refuses_unused_predictor(monkeypatch):
    from examples.general_contact_multiple_beam_s3_transient import build
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.general_contact.transient import solve_transient_contact,_admit,ContactTransientSolverControls
    import anyfesolver.general_contact.multiple_beam_s3_transient as owner
    model,definition,settings,velocity=build()
    controls=ContactTransientSolverControls(allow_multiple_deformable_beam_s3_dev=True)
    before=repr(model)
    assert _admit(model,definition,settings,controls)==owner.PROFILE
    assert repr(model)==before
    class WrongSettings:
        path_predictor='control_projection_dev'
    with pytest.raises(ValueError,match='requires exact contact definitions/settings'):
        _admit(model,definition,WrongSettings(),controls)
    with pytest.raises(ValueError,match='requires exact contact definitions/settings'):
        _admit(model,object(),replace(settings,path_predictor='control_projection_dev'),controls)
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(True)
        raise AssertionError('allocation or boundary mutation reached before selector refusal')
    monkeypatch.setattr(owner,'pair_definitions',forbidden)
    monkeypatch.setattr(owner.MultipleBeamS3TransientManager,'__init__',forbidden)
    monkeypatch.setattr(model,'apply_boundary_conditions',forbidden)
    with pytest.raises(ValueError,match='path_predictor requires'):
        solve_transient_contact(model,definition,TransientConfig(dt=1e-5,t_end=1e-5,initial_velocity=velocity),
            contact_settings=replace(settings,path_predictor='control_projection_dev'),solver_controls=controls)
    assert not calls and repr(model)==before
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(verified=True,
        actual_public_multiple_beam_flag=True,legacy_fixture_admitted=owner.PROFILE,
        named_selector_refusal=True,before_pair_inventory_and_manager_allocation=True,
        before_boundary_mutation=True,repr_fields_unchanged=True,wrong_type_contract_preserved=True),indent=2))




def test_default_identity_physical_map_and_public_safety(monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled,solve_static_nonlinear
    from examples.general_contact_automatic_s3_q4 import build,solve_shell_pair
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    candidate=build();model,domain,settings,res,path=candidate;before=repr(model)
    manager=GeneralS3PatchManager(model,res.definition,settings);positions=manager._positions(np.zeros(manager.n_dof))
    np.testing.assert_allclose(positions[0][:,2],.03,atol=1e-17,rtol=0);np.testing.assert_allclose(positions[1][:,2],.015,atol=1e-17,rtol=0)
    assert len(manager.law.source_node_ids)==4 and len(manager.law.target_node_ids)==4
    assert res.selections[0].source=='s3_source'
    legacy=build(lifecycle=True,sliding=False,path_predictor='none')
    legacy_manager=GeneralS3PatchManager(legacy[0],legacy[3].definition,legacy[2])
    def forbidden_error(*args):raise AssertionError('legacy-none used newly opt-in endpoint map errors')
    monkeypatch.setattr(legacy_manager,'_position_errors',forbidden_error)
    assert legacy_manager._swept_admissible(np.zeros(legacy_manager.n_dof),np.zeros(legacy_manager.n_dof))
    error_function=manager._position_errors;observed=[]
    def record_error(u):
        values=error_function(u);assert all(np.isfinite(v).all() for v in values);observed.append(max(float(np.max(v)) for v in values));return values
    monkeypatch.setattr(manager,'_position_errors',record_error)
    assert manager._swept_admissible(np.zeros(manager.n_dof),np.zeros(manager.n_dof))
    assert len(observed)==2 and min(observed)>0

    with pytest.raises(ValueError,match='path_predictor requires'):
        solve_static_nonlinear(model,contact=res.definition,contact_settings=settings)
    # Current typed refusal (static_path.py): the invalid friction object is
    # rejected by the S3SlidingFrictionSettings/projected-penalty profile gate.
    with pytest.raises(ValueError,match='physical-side/predicted friction path requires S3SlidingFrictionSettings'):
        solve_shell_pair(*candidate,contact_friction=object())
    token=CancellationToken();token.cancel('S3/Q4 public precancel')
    with pytest.raises(SolveCancelled):solve_shell_pair(*candidate,cancellation_token=token)
    assert repr(model)==before
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(verified=True,
        physical_skin_positions=True,coarser_S3_role=True,legacy_none_endpoint_error_trap=True,optin_nonzero_endpoint_errors=True,actual_public_unused_force_refusal=True,friction_refusal=True,precancel=True,repr_fields_unchanged=True),indent=2),encoding='utf-8')


@pytest.mark.skipif(not os.environ.get('S3_Q4_DEFAULT_BASELINE') or not Path(os.environ['S3_Q4_DEFAULT_BASELINE']).is_file(),
    reason='archived baseline input S3_Q4_DEFAULT_BASELINE absent: default-identity and terminal-restart replay requires the original baseline witness')
def test_default_identity_archive_baseline_terminal_replay():
    from examples.general_contact_automatic_s3_pair import build_s3_pair,solve_shell_pair as oldsolve
    old=json.loads(Path(os.environ['S3_Q4_DEFAULT_BASELINE']).read_text())
    model,domain,settings,res,path=build_s3_pair();assert res.identity==old['identity']
    terminal=oldsolve(model,domain,settings,res,path,restart_checkpoint=old['checkpoint'])
    assert terminal.status=='completed';np.testing.assert_array_equal(terminal.displacements,old['displacements'])
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(verified=True,old_default_identity_and_terminal_restart=True),indent=2),encoding='utf-8')
