"""Independent finite-gap work and weak-pressure public continuation evidence."""
import os,json,copy
from pathlib import Path
import numpy as np
import pytest
from dataclasses import replace
from examples.general_contact_s3_mesh_weak_pressure_friction import build_case
import examples.general_contact_s3_mesh_moving_friction as old_example
import test_general_contact_s3_mesh_moving_friction as reuse
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager


def manager():
    case=build_case();m,d,s,f,p,r=case
    owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
    return case,owner


def test_weak_local_work_and_front(monkeypatch):
    from anyfesolver.general_contact.semismooth import linearize
    case,owner=manager();m,d,s,f,p,r=case
    u=r['manufactured_pose'].copy();z=r['supported_force'].copy()
    owner.commit(owner.validate_successor(u,z))
    # Controlled finite gap, actual shared P1 partial supported pressure.
    for node in (2,3,4,5):u[m.mesh.dof_manager.get_node_dofs(node)[2]]+=5e-6
    for node in (1,2,3,4,5):u[m.mesh.dof_manager.get_node_dofs(node)[0]]+=1e-6
    z=np.array([0.,0.,0.,.006])
    force,history,records=owner._force(u,z)
    successor=owner.validate_successor(u,z)
    assert successor[2]['positive_pressure_locally_open_points']
    assert successor[2]['active_face_max_abs_gap_m']<=5e-5
    # Independent target normal derivative from both target triangle edges.
    corners={e:a.current_positions(u) for e,a in owner.target_adapters.items()}
    raw=sum((np.cross(c[1]-c[0],c[2]-c[0]) for c in corners.values()),np.zeros(3));norm=np.linalg.norm(raw);n=raw/norm
    v=np.sin(np.arange(60)+.31)*1e-3
    dnraw=np.zeros(3)
    for e,c in corners.items():
        maps=owner.target_adapters[e].work_map.toarray().reshape(3,3,60);velocity=maps@v
        dnraw+=np.cross(velocity[1]-velocity[0],c[2]-c[0])+np.cross(c[1]-c[0],velocity[2]-velocity[0])
    dn=(np.eye(3)-np.outer(n,n))@dnraw/norm
    independent=0.;direct=0.;normal_term=0.
    for j,(point,area,rec) in enumerate(zip(owner.points,owner.weights,records)):
        if rec['target_element'] is None:continue
        e=rec['target_element'];bary=np.array(rec['target_barycentric']);a=owner.target_adapters[e]
        x=owner.point_reference[j]+owner.point_maps[j]@u[owner.work_columns];y=bary@corners[e];rr=x-y;gap=n@rr
        dx=point.position_map@v;dy=bary@(a.work_map@v).reshape(3,3)
        traction=-f.tangential_stiffness_Pa_per_m*((np.eye(3)-np.outer(n,n))@rr)
        np.testing.assert_allclose(traction,rec['traction_pa'],rtol=1e-7,atol=1e-10)
        dv=(np.eye(3)-np.outer(n,n))@(dx-dy)-n*(rr@dn)-gap*dn
        independent-=area*traction@dv;direct-=area*traction@(dx-dy);normal_term+=area*gap*(traction@dn)
    err=abs(force@v-independent)/max(abs(independent),1e-12)
    assert err<=1e-5
    assert abs(independent-direct-normal_term)<=1e-12
    # Both-body physical force/couple includes the normal-variation nodal work.
    src,tgt=owner.normal._positions(u);ids=owner.normal.law.source_node_ids+owner.normal.law.target_node_ids
    forces=[];moments=[]
    for node,point in zip(ids,np.concatenate((src,tgt))):
        dofs=m.mesh.dof_manager.get_node_dofs(node);ff=force[dofs[:3]];forces.append(ff);moments.append(np.cross(point,ff))
    assert np.linalg.norm(np.sum(forces,axis=0))/.02<=1e-6
    assert np.linalg.norm(np.sum(moments,axis=0))/.02<=1e-6
    omega=np.array([.2,-.3,.1]);rigid=np.zeros(60)
    for node in range(1,11):
        dd=m.mesh.dof_manager.get_node_dofs(node);pt=m.mesh.get_node(node)
        current=np.array([pt.x,pt.y,pt.z])+u[dd[:3]]
        rigid[dd[:3]]=np.cross(omega,current);rigid[dd[3:6]]=omega
    assert abs(force@rigid)<=1e-12
    base=linearize(owner,u,z,tangent=True);w=np.array([.2,-.3,.4,.1])*1e-3;errors=[]
    # Stay inside the material path domain: affine target/source variations.
    vv=np.zeros(60)
    for node in range(1,6):vv[m.mesh.dof_manager.get_node_dofs(node)[0]]=.001
    for node in (2,3,4,5):
        pt=m.mesh.get_node(node);vv[m.mesh.dof_manager.get_node_dofs(node)[2]]=(.2+pt.x*.1+pt.y*.3)*1e-3
    for node in (7,8,9,10):
        pt=m.mesh.get_node(node);dd=m.mesh.dof_manager.get_node_dofs(node);vv[dd[0]]=(.1+pt.x*.2)*1e-3;vv[dd[2]]=(.2+pt.x*.1+pt.y*.3)*1e-3
    for h in (1e-4,5e-5):
        fd=(linearize(owner,u+h*vv,z+h*w,tangent=False).force-linearize(owner,u-h*vv,z-h*w,tangent=False).force)/(2*h)
        exact=base.tangent@vv+base.force_pressure_gradient@w;errors.append(float(np.linalg.norm(fd-exact)/max(np.linalg.norm(fd),1e-12)))
    assert max(errors)<=1e-5
    # At an inactive material front, the existing branch-frozen derivative is
    # the nonpositive-pressure one-sided generalized selection. Do not claim
    # the opposite one-sided derivative is identical across the kink.
    zero=np.array([0.,0.,.006,0.])
    front_records=owner._force(u,zero)[2]
    inactive=[j for j,r0 in enumerate(front_records) if r0['pressure_pa']==0.]
    assert len(inactive)==3 and all(front_records[j]['branch']=='open' for j in inactive)
    # Independent P1 positivity: the first face has exactly zero coefficients;
    # a nonpositive probe has zero Coulomb load/traction for every material point.
    assert all(set(owner.pressure_indices[j])=={0,1,3} for j in inactive)
    front=linearize(owner,u,zero,tangent=True);direction=np.array([-1.,-1.,0.,-1.])*1e-3
    h=1e-5
    actual=(linearize(owner,u,zero+h*direction,tangent=False).force-front.force)/h
    expected=front.force_pressure_gradient@direction
    front_error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(actual),1e-12))
    assert front_error<=1e-5
    negative_records=owner._force(u,zero+h*direction)[2]
    assert all(negative_records[j]['pressure_pa']==0. and np.linalg.norm(negative_records[j]['traction_pa'])==0. for j in inactive)
    released=owner.validate_successor(u,np.zeros(4))
    assert all(h.target_element is None and h.stored_energy_Pa_m==0 for h in released[1])
    assert not released[2]['physical_all_points_strictly_open'] or all(x=='open' for x in released[2]['branch'])
    owner.commit(released);reborn=owner.validate_successor(u,z)
    assert all(b in ('activation','open') for b in reborn[2]['branch']) and reborn[2]['storage_j']==0.
    # Counter-control: fixed actual x/y, rotating normal at finite gap creates
    # projection elastic storage. Explicitly a numerical weak-gap effect.
    gg=5e-6;angle=1e-4;nn=np.array([np.sin(angle),0.,np.cos(angle)])
    rr=np.array([0.,0.,gg]);projection=(np.eye(3)-np.outer(nn,nn))@rr
    numerical_storage=.5*f.tangential_stiffness_Pa_per_m*float(projection@projection)*sum(owner.weights)
    assert numerical_storage>0. and numerical_storage<=1e-12
    reuse._save(dict(fixed_material_points_rotating_normal_storage_j=numerical_storage,physical_material_point_velocity=0.,
        physical_virtual_work_relative=err,direct_material_work_j=direct,numerical_normal_variation_work_j=normal_term,
        full_qz_directional_errors=errors,actual_zero_pressure_face_points=inactive,inactive_front_nonpositive_generalized_error=front_error,partial_pressure=z.tolist(),release_rebirth=True,weak_positive_pressure_local_opening=True,
        scope='independent finite-gap bothbody work; not pointwise complementarity'))


@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE') or not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='Archived staged inputs FRICTION_STAGE, AUTOMATIC_WORKFLOW_EVIDENCE absent; this replay requires its original workflow evidence.')
def test_weak_public(monkeypatch):
    monkeypatch.setattr(old_example,'build_case',build_case)
    monkeypatch.setattr(reuse,'_pose_gate',_weak_pose_gate)
    reuse.test_mesh_public()
    if os.environ['FRICTION_STAGE']=='B6':
        evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']);left=json.loads((evidence/'A6.json').read_text());right=json.loads((evidence/'B6.json').read_text())
        for a,b in zip(left['info']['force_displacement_history'],right['info']['force_displacement_history']):
            for key in ('pressure_inactive_points','positive_pressure_locally_open_points','physical_all_points_strictly_open'):
                assert a['friction'][key]==b['friction'][key]
            for key,tol in (('active_face_max_abs_gap_m',1e-10),('target_normal_delta',1e-10),('numerical_normal_variation_endpoint_work_bound_j',1e-12)):
                assert abs(a['friction'][key]-b['friction'][key])<=tol


def _weak_pose_gate(model,definition,settings,friction,u,z):
    from fractions import Fraction as F
    normal=GeneralS3PatchManager(model,definition,settings);owner=create_s3_friction_manager(model,normal,friction)
    source,target=normal._positions(u);ids=normal.law.source_node_ids+normal.law.target_node_ids
    independent=[]
    for node,h in zip(ids,(-.025,)*len(source)+(.015,)*len(target)):
        pt=model.mesh.get_node(node);dd=model.mesh.dof_manager.get_node_dofs(node)
        independent.append(np.array([pt.x,pt.y,pt.z])+u[dd[:3]]+h*(np.array([0.,0.,1.])+np.cross(u[dd[3:6]],[0.,0.,1.])))
    np.testing.assert_allclose(np.concatenate((source,target)),independent,rtol=0,atol=1e-16)
    exact=lambda a:[[F(float(x)) for x in row] for row in a]
    sub=lambda a,b:[x-y for x,y in zip(a,b)]
    cross=lambda a,b:[a[i]*b[j]-a[j]*b[i] for i,j in ((1,2),(2,0),(0,1))]
    dense=owner.point_reference+owner.point_maps@u[owner.work_columns];points=exact(np.concatenate((source,dense)))
    report=[];normals=[]
    for adapter in owner.target_adapters.values():
        c=exact(adapter.current_positions(u));n=cross(sub(c[1],c[0]),sub(c[2],c[0]));nn=sum(x*x for x in n)
        gaps=[sum(x*y for x,y in zip(sub(pt,c[0]),n)) for pt in points]
        assert nn>0
        if np.all(np.asarray(z)==0):assert min(gaps)>0
        else:
            assert min(z)>=0
            assert all(g*g<=F(float(friction.finite_material_gap_error_m))**2*nn for g in gaps)
        report.append(dict(element=adapter.element_id,exact_numerator=[str(g) for g in gaps],exact_normal_squared=str(nn),normalized_point_gap_approx=[float(g)/np.sqrt(float(nn)) for g in gaps]))
        normals.append(np.array(n,dtype=float))
    return dict(all_facets=report,target_raw_normal=np.sum(normals,axis=0).tolist())


@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE') or not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='Archived staged inputs FRICTION_STAGE, AUTOMATIC_WORKFLOW_EVIDENCE absent; this replay requires its original workflow evidence.')
def test_weak_saved_proof():
    """Recompute gates from saved poses without another coupled solve."""
    from examples.general_contact_s3_mesh_weak_pressure_friction import build_case
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    from anyfesolver.general_contact.semismooth import linearize
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    prefix=os.environ['FRICTION_STAGE']=='prefix-proof'
    through=3 if os.environ['FRICTION_STAGE']=='slide-proof' else 2 if prefix else 6
    saved=json.loads((evidence/('A3.json' if through==3 else 'A2.json' if prefix else 'A6.json')).read_text())
    case=build_case();model,definition,settings,friction,program,_=case
    normal=GeneralS3PatchManager(model,definition,settings)
    owner=create_s3_friction_manager(model,normal,friction)
    snapshots=[s for name in (('A1','A2') if prefix else ('B2','A3') if through==3 else ('B2','A3','A5','A6'))
               for s in json.loads((evidence/(name+'.json')).read_text())['snapshots']]
    history=saved['info']['force_displacement_history']
    assert [s['step_index'] for s in snapshots]==list(range(1,through+1))
    assert [s['step_index'] for s in history]==list(range(1,through+1))
    proof=[]
    for snap,phase in zip(snapshots,history):
        step=snap['step_index'];u=np.array(snap['displacement']);z=np.array(phase['friction']['supported_force_n'])
        geometric=_weak_pose_gate(*case[:4],u,z)
        raw=linearize(owner,u,z,tangent=False)
        _,_,records=owner._force(u,z)
        structural,_,_=_assemble_nonlinear_system(model,u,{},5,tangent=False,require_full_coordinates=True)
        external=np.zeros(60)
        for load,value in zip(program.nodal_forces,program.force_values_at(step)):
            dof=model.mesh.dof_manager.get_node_dofs(load.node_id)[('ux','uy','uz').index(load.dof)]
            external[dof]+=value
        residual=structural+raw.force-external
        reactions=np.array(phase['reaction_generalized'])
        # Literal MPC/path work rows. Dependent source ux/uy rows are summed,
        # while every independent contact normal and target ux is retained.
        modes=[sum(residual[model.mesh.dof_manager.get_node_dofs(n)[a]] for n in (1,2,3,4,5))-reactions[a] for a in (0,1)]
        modes += [residual[model.mesh.dof_manager.get_node_dofs(n)[2]] for n in (2,3,4,5)]
        for axis in (0,2):
            r={n:residual[model.mesh.dof_manager.get_node_dofs(n)[axis]] for n in (7,8,9,10)}
            modes.extend((r[7]-r[9],r[8]+r[9],r[10]+r[9]))
        equilibrium=float(np.linalg.norm(modes,1)/.02)
        source,target=normal._positions(u);points=np.concatenate((source,target))
        force_rows=[];moment_rows=[];lever_errors=[]
        for node,h,point in zip((2,3,4,5,7,8,9,10),(-.025,)*4+(.015,)*4,points):
            dofs=np.array(model.mesh.dof_manager.get_node_dofs(node))
            f=raw.force[dofs[:3]];m=raw.force[dofs[3:6]]
            independent_m=np.cross(np.array([0.,0.,h]),f)
            lever_errors.extend(m-independent_m)
            force_rows.append(f);moment_rows.append(np.cross(point,f))
        force_wrench=float(np.linalg.norm(np.sum(force_rows,axis=0))/.02)
        physical_moment=float(np.linalg.norm(np.sum(moment_rows,axis=0))/.02)
        lever_error=float(np.linalg.norm(lever_errors)/.02)
        # Independent complete nodal virtual work with declared offset couples.
        variation=np.cos(np.arange(60)+.7)*1e-3
        work=0.
        for node,h,f in zip((2,3,4,5,7,8,9,10),(-.025,)*4+(.015,)*4,force_rows):
            dofs=np.array(model.mesh.dof_manager.get_node_dofs(node))
            dx=variation[dofs[:3]]+np.cross(variation[dofs[3:6]],[0.,0.,h])
            work+=f@dx
        work_error=float(abs(raw.force@variation-work)/max(abs(work),1e-12))
        diss=sum(r['dissipation_increment_j'] for r in records)
        projection_loss=sum(r['return_projection_loss_j'] for r in records)
        return_drop=sum(r['return_storage_drop_j'] for r in records)
        item=dict(step=step,equilibrium_relative=equilibrium,physical_force_wrench_relative=force_wrench,
            physical_skin_moment_relative=physical_moment,independent_lever_relative=lever_error,
            physical_virtual_work_relative=work_error,complementarity=phase['complementarity_relative_force'],
            dissipation_increment_j=diss,projection_loss_j=projection_loss,return_storage_drop_j=return_drop,
            branch=phase['friction']['branch'],geometry=geometric,
            target_ux=[u[model.mesh.dof_manager.get_node_dofs(n)[0]] for n in (7,8,9,10)])
        target_normal=np.array(geometric['target_raw_normal']);target_normal/=np.linalg.norm(target_normal)
        source_normal=np.cross(source[1]-source[0],source[2]-source[0]);source_normal/=np.linalg.norm(source_normal)
        item['source_target_normal_cosine']=float(source_normal@target_normal)
        item['finite_gap_extension_work_bound_j']=float(max(abs(g) for f in geometric['all_facets'] for g in f['normalized_point_gap_approx'])*sum(owner.weights)*max(np.linalg.norm(r['traction_pa']) for r in records)*2.)
        proof.append(item);reuse._save(dict(scope='saved_actual_binary_poses_independent_work',phases=proof))
        assert max(equilibrium,force_wrench,physical_moment,lever_error)<=1e-6,item
        assert work_error<=1e-5,item
        assert phase['complementarity_relative_force']<=1e-8
        assert diss>=-1e-12 and return_drop+1e-12>=diss
        # This prior hard-contact displacement-only proxy is reported separately.
        # Weak profile has explicit50um numerical gap, not hard1e-10 closure.
        item['finite_gap_proxy_scope']='endpoint traction*gap envelope, not work creation or path integral'
        item['numerical_normal_variation_endpoint_work_bound_j']=owner.validate_successor(u,z)[2]['numerical_normal_variation_endpoint_work_bound_j']
        owner.commit(owner.validate_successor(u,z))
    if through<6:
        reuse._save(dict(phases=proof,scope='accepted_press_stick_endpoint_work_not_full_cycle',unmet=['slip','reversal','open','recontact','integrated_facet_transfer','integrated_normal_delta']))
        return
    assert all(b=='open' for b in proof[4]['branch'])
    assert any(b=='slip' for b in proof[2]['branch'])
    assert max(abs(x) for x in proof[2]['target_ux'])>1e-10
    normals=[np.array(p['geometry']['target_raw_normal']) for p in proof]
    normalized=[n/np.linalg.norm(n) for n in normals]
    assert np.linalg.norm(normalized[2]-normalized[0])>1e-12
    assert any(a!=b for a,b in zip(history[0]['friction']['target_elements'],history[2]['friction']['target_elements']))
    reuse._save(dict(phases=proof,target_normal_delta=normalized[2]-normalized[0],material_facet_transfer=True,scope='accepted_endpoint_work_bound_not_path_integral'))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='Archived staged inputs AUTOMATIC_WORKFLOW_EVIDENCE absent; this replay requires its original workflow evidence.')
def test_weak_rollback(monkeypatch):
    monkeypatch.setattr(old_example,'build_case',build_case)
    reuse.test_mesh_rollback(monkeypatch)


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='Archived staged inputs AUTOMATIC_WORKFLOW_EVIDENCE absent; this replay requires its original workflow evidence.')
def test_weak_safety(monkeypatch):
    from dataclasses import asdict
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact import S3MeshWeakPressureFrictionSettings
    import anyfesolver.general_contact.static_path as driver
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']);case=build_case();m,d,s,f,p,_=case
    with pytest.raises(TypeError):S3MeshWeakPressureFrictionSettings(**{k:v for k,v in asdict(f).items() if k!='finite_material_gap_error_m'})
    with pytest.raises(ValueError):replace(f,closed_point_gap_tolerance_m=1e-11)
    # Preserved old hard owner still refuses the actual accepted partial state.
    from examples.general_contact_s3_mesh_control_lift import build_case as hard_case
    hard=hard_case();hard_owner=create_s3_friction_manager(hard[0],GeneralS3PatchManager(*hard[:3]),hard[3])
    slide=json.loads((evidence/'A3.json').read_text());u=np.array(slide['displacement']);z=np.array(slide['info']['force_displacement_history'][2]['friction']['supported_force_n'])
    before=copy.deepcopy(hard_owner.checkpoint_state())
    with pytest.raises(ValueError,match='partial pressure'):hard_owner.validate_successor(u,z)
    assert hard_owner.checkpoint_state()==before
    incoming=json.loads((evidence/'A2.json').read_text())['checkpoint'];original=copy.deepcopy(incoming)
    before_model=repr(m);token=CancellationToken();token.cancel('weak profile precancel')
    with pytest.raises(SolveCancelled):solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,contact_path_program=p,cancellation_token=token)
    assert repr(m)==before_model
    for kind in ('gap-setting','tamper','old-owner'):
        m,d,s,f,p,_=build_case();given=copy.deepcopy(original);before=repr(m)
        if kind=='gap-setting':f=replace(f,finite_material_gap_error_m=4e-5)
        elif kind=='old-owner':f=hard[3]
        else:given['path_state']['friction_state']['displacement'][0]+=.001
        supplied_before=copy.deepcopy(given)
        with pytest.raises(ValueError):solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,contact_path_program=p,restart_checkpoint=given,tolerance=1e-12,max_iterations=18)
        assert repr(m)==before and given==supplied_before
    # No normal-only exposure on the new public full owner dependency.
    m,d,s,f,p,_=build_case();before=repr(m)
    with monkeypatch.context() as patch:
        patch.setattr(driver,'_assemble_nonlinear_system',lambda *a,**k:(_ for _ in ()).throw(AssertionError('normal-only reached material')))
        with pytest.raises(ValueError):solve_static_nonlinear(m,contact=d,contact_settings=s,contact_path_program=p)
    assert repr(m)==before
    from test_general_contact_s3_sliding_friction import _manager
    import hashlib
    baseline=json.loads((evidence.parent/'default_contract_baseline.json').read_text())
    legacy=hashlib.sha256(json.dumps(_manager()[-1].restart_contract(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert legacy==baseline['legacy_contract_sha256']
    reuse._save(dict(required_explicit_gap=True,nondefault_inherited_hard_control_refused=True,old_hard_partial_guard_unchanged=True,
        incompatible_weak_gap_old_owner_checkpoint=True,precancel=True,normal_only_before_material=True,legacy_identity=legacy))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='Archived staged inputs AUTOMATIC_WORKFLOW_EVIDENCE absent; this replay requires its original workflow evidence.')
def test_weak_terminal_scope():
    from examples.general_contact_s3_mesh_moving_friction import solve_case
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']);saved=json.loads((evidence/'A6.json').read_text())
    case,owner=manager();u=np.array(saved['displacement']);checkpoint=saved['checkpoint'];z=np.array(checkpoint['path_state']['normal_forces'])
    owner.restore(checkpoint['path_state']['friction_state'],u)
    successor=owner.validate_successor(u,z)
    assert successor[2]['finite_material_gap_error_scope']=='accepted_endpoint_only'
    result=solve_case(case,6,checkpoint)
    assert result.status=='completed' and not result.snapshots
    assert result.info['contact_path']['finite_material_gap_error_scope']=='accepted_endpoint_only'
    np.testing.assert_allclose(result.displacements,u,rtol=0.,atol=1e-10)
    reuse._save(dict(status=result.status,scope=result.info['contact_path']['finite_material_gap_error_scope'],new_increments=0,
        final_checkpoint_state=result.to_restart_checkpoint()['path_state']['friction_state']))
