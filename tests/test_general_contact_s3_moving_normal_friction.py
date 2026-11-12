"""Local material-motion discriminators; no moving-normal admission is implied."""
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from anyfesolver.general_contact.s3_sliding_friction import S3SlidingFrictionManager

import json,os
import pytest
from pathlib import Path


def _save(data):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(data,indent=2,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()),encoding='utf-8')


def test_moving_normal_unit():
    from examples.general_contact_s3_moving_normal_friction import build_case,TILT
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager,S3_MONOLITHIC_PROFILE
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    from anyfesolver.general_contact.semismooth import linearize,reference_nodal_support
    model,definition,settings,friction,program,reference=build_case()
    normal=GeneralS3PatchManager(model,definition,settings)
    assert normal.profile==S3_MONOLITHIC_PROFILE
    manager=create_s3_friction_manager(model,normal,friction)
    u=reference['manufactured_pose'];z=reference['supported_force']
    assert normal.admissible_step_fraction(np.zeros(48),u)==1.
    assert manager.admissible_step_fraction(np.zeros(48),u)==1.
    support=reference_nodal_support(manager.law)
    np.testing.assert_allclose(support,np.full(3,.005/3),rtol=1e-14,atol=1e-17)
    raw=linearize(manager,u,z,tangent=True)
    # Independent manufactured physical point-force comparison at closure.
    expected=reference['manufactured_force']
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    expected=expected-assemble_stiffness_matrix(model)[0]@u
    translations=np.asarray([model.mesh.dof_manager.get_node_dofs(i)[:3] for i in (2,3,4,6,7,8)]).reshape(-1)
    np.testing.assert_allclose(raw.force[translations],expected[translations],rtol=1e-6,atol=1e-12)
    manager.commit(manager.validate_successor(u,z))
    slip=u.copy()
    for node in (2,3,4):
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        slip[dofs[0]]+=1e-4;slip[dofs[2]]-=TILT*1e-4
    base=linearize(manager,slip,z,tangent=True)
    v=np.cos(np.arange(48)+.3)*1e-3;w=np.array([.2,-.3,.4])*1e-3
    errors=[]
    for h in (1e-4,5e-5):
        plus=linearize(manager,slip+h*v,z+h*w,tangent=False)
        minus=linearize(manager,slip-h*v,z-h*w,tangent=False)
        fd=(plus.force-minus.force)/(2*h)
        exact=base.tangent@v+base.force_pressure_gradient@w
        errors.append(float(np.linalg.norm(fd-exact)/max(np.linalg.norm(fd),1e-12)))
    assert max(errors)<=1e-5
    # Source-ray outside control: target orthogonal projection is inside but
    # source-directed projection is outside. Independent determinant is -.04.
    assert -F(1,5)*F(3,5)+F(1,10)*F(4,5)==-F(1,25)
    from anyfesolver.general_contact.s3_friction_moving_coverage import full_moving_coverage
    target=SimpleNamespace(reference_positions=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
                           work_map=csr_matrix((9,1)))
    source=SimpleNamespace(reference_positions=np.array([[.1,.1,.2],[.1,.12,.2],[.084,.1,.212]]),
                           work_map=csr_matrix((9,1)))
    control=SimpleNamespace(n_dof=1,target_adapters={1:target},source_adapters=(source,),
        point_reference=np.array([[.1,.1,.2]]),point_maps=np.zeros((1,3,1)),work_columns=np.array([0]))
    assert not full_moving_coverage(control,np.zeros(1),np.zeros(1))
    _save(dict(profile=manager.profile,support=support,directional_errors=errors,
               coupling_pressure_norm=float(np.linalg.norm(base.force_pressure_gradient)),
               manufactured_scope='qualified_structural_K_and_independent_planar_P1_contact_load'))


def _pose_gate(model,definition,settings,friction,u,z):
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    normal=GeneralS3PatchManager(model,definition,settings)
    manager=create_s3_friction_manager(model,normal,friction)
    source,target=normal._positions(u)
    # Independently reconstruct the fixed linearized physical skin convention.
    independent=[]
    for nodes,h in (((2,3,4),-.025),((6,7,8),.015)):
        for node in nodes:
            p=model.mesh.get_node(node)
            dofs=model.mesh.dof_manager.get_node_dofs(node)
            independent.append(np.array([p.x,p.y,p.z])+u[dofs[:3]]
                +h*(np.array([0.,0.,1.])+np.cross(u[dofs[3:6]],[0.,0.,1.])))
    np.testing.assert_allclose(np.concatenate((source,target)),independent,rtol=0.,atol=1e-16)
    exact=[[F(float(x)) for x in row] for row in np.concatenate((source,target))]
    def sub(a,b):return [x-y for x,y in zip(a,b)]
    def cross(a,b):return [a[i]*b[j]-a[j]*b[i] for i,j in ((1,2),(2,0),(0,1))]
    n=cross(sub(exact[4],exact[3]),sub(exact[5],exact[3]));nn=sum(x*x for x in n)
    dense=manager.point_reference+manager.point_maps@np.asarray(u)[manager.work_columns]
    actual_points=exact[:3]+[[F(float(x)) for x in row] for row in dense]
    c=[sum(x*y for x,y in zip(sub(p,exact[3]),n)) for p in actual_points]
    assert nn>0
    if np.all(np.asarray(z)==0.):assert min(c)>0
    else:
        assert min(z)>0
        assert all(x*x<=F(1,10**10)**2*nn for x in c)
    return dict(exact_numerator=[str(x) for x in c],exact_normal_squared=str(nn),
                normalized_point_gap_approx=[float(x)/np.sqrt(float(nn)) for x in c],
                target_raw_normal=np.array(n,dtype=float).tolist())


@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE'),
    reason='installed friction stage input FRICTION_STAGE absent: staged moving-normal solve requires the original workflow stage selection')
def test_moving_normal_public():
    from examples.general_contact_s3_moving_normal_friction import build_case,solve_case
    stage=os.environ['FRICTION_STAGE'];evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    chain={'A1':(1,None),'A3':(3,'A1'),'A5':(5,'A3'),'A6':(6,'A5'),
           'B2':(2,None),'B4':(4,'B2'),'B6':(6,'B4')}
    case=build_case();through=1 if stage=='smoke' else 3 if stage=='prefix' else 5 if stage=='open-prefix' else 6
    restart=None
    if stage in chain:
        through,previous=chain[stage]
        if previous:restart=json.loads((evidence/(previous+'.json')).read_text())['checkpoint']
    if stage in ('resume','open-resume'):
        restart=json.loads((evidence/('prefix.json' if stage=='resume' else 'open-prefix.json')).read_text())['checkpoint']
    result=solve_case(case,through,restart)
    data=dict(status=result.status,info=result.info,displacement=result.displacements,
              checkpoint=result.to_restart_checkpoint() if result.status=='completed' else None,
              snapshots=[dict(step_index=s.step_index,displacement=s.displacements)
                         for s in result.snapshots])
    _save(data)
    assert result.status=='completed',result.info
    state=data['checkpoint']['path_state']
    z=np.array(state['normal_forces'])
    data['point_gap']=_pose_gate(*case[:4],result.displacements,z)
    start=0 if restart is None else restart['path_state']['accepted_step']
    assert [s['step_index'] for s in data['snapshots']]==list(range(start+1,through+1))
    if stage=='B6':
        full=json.loads((evidence/'A6.json').read_text())
        np.testing.assert_allclose(result.displacements,full['displacement'],rtol=0.,atol=1e-10)
        np.testing.assert_allclose(z,full['checkpoint']['path_state']['normal_forces'],rtol=1e-8,atol=1e-12)
        left=full['info']['force_displacement_history'];right=data['info']['force_displacement_history']
        assert [s['step_index'] for s in left]==[s['step_index'] for s in right]==list(range(1,7))
        for a,b in zip(left,right):
            assert a['friction']['branch']==b['friction']['branch']
            np.testing.assert_allclose(a['friction']['supported_force_n'],b['friction']['supported_force_n'],rtol=1e-8,atol=1e-12)
            # Traction is reference-area density: compare its point force,
            # not an unlabelled Pa norm. Work uses the frozen dimensional gate.
            np.testing.assert_allclose(np.array(a['friction']['traction_pa'])*(.005/3),
                np.array(b['friction']['traction_pa'])*(.005/3),rtol=1e-8,atol=1e-12)
            for field in ('storage_j','dissipation_increment_j','released_storage_j','return_projection_loss_j'):
                assert abs(a['friction'][field]-b['friction'][field])<=1e-12
        lhs=full['checkpoint']['path_state']['friction_state'];rhs=state['friction_state']
        np.testing.assert_allclose(lhs['displacement'],rhs['displacement'],rtol=0.,atol=1e-10)
        for a,b in zip(lhs['history'],rhs['history']):
            assert a['target_element']==b['target_element']
            np.testing.assert_allclose(a['barycentric'],b['barycentric'],rtol=0.,atol=1e-10)
            assert abs(a['cumulative_plastic_slip_m']-b['cumulative_plastic_slip_m'])<=1e-10
            for field in ('cumulative_dissipation_Pa_m','stored_energy_Pa_m'):
                assert abs(a[field]-b[field])*(.005/3)<=1e-12
        data['restart_friction_state_exact_equal']=lhs==rhs
        data['restart_phase_summary_exact_equal']=[a['friction']==b['friction'] for a,b in zip(left,right)]
    if stage in ('resume','open-resume'):
        full=json.loads((evidence/'full.json').read_text())
        np.testing.assert_allclose(result.displacements,full['displacement'],rtol=0.,atol=1e-10)
        np.testing.assert_allclose(z,full['checkpoint']['path_state']['normal_forces'],rtol=1e-8,atol=1e-12)
    _save(data)


def test_finite_gap_projected_anchor_countercontrols():
    """Check the actual existing work, not an alternative friction history."""
    g = F(-5, 1_000_000)
    old_n = (F(0), F(0), F(1))
    rotated_n = (F(3, 5), F(0), F(4, 5))
    r = (F(0), F(0), g)
    dot = sum(a*b for a, b in zip(rotated_n, r))
    projected = tuple(a-b*dot for a, b in zip(r, rotated_n))
    assert projected == (-12*g/25, F(0), 9*g/25)
    assert sum(a*a for a in projected) == (3*g/5)**2
    # Rotating a source ray about the unchanged actual skin point has the same
    # nonzero shear scale. This control is for the unadopted ray proposal only.
    ray = tuple(a-g*b for a, b in zip(r, rotated_n))
    ray_tangent = (ray[0], ray[1], F(0))
    assert ray_tangent == (-3*g/5, F(0), F(0))

    # One local coordinate moves other target vertices to change its normal,
    # while both actual material points x and y remain fixed. No contact/FE
    # admission is exercised by this deliberately minimal algebra fixture.
    manager = S3SlidingFrictionManager.__new__(S3SlidingFrictionManager)
    manager.n_dof = 1
    manager.work_columns = np.array([0])
    manager.point_reference = np.array([[0., 0., float(g)]])
    manager.point_maps = np.zeros((1, 3, 1))
    target_map = np.zeros((3, 3, 1))
    target_map[1, 2, 0] = -1.
    manager.target_maps = {1: target_map}
    manager.target_adapters = {1: SimpleNamespace(node_ids=(1, 2, 3))}
    corners = {1: np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])}
    n = np.array(old_n, dtype=float)
    dn = manager._normal_variation(corners, n, local=True)
    elastic, work, anchor = manager._work(
        None, np.zeros(1), 1, (1., 0., 0.), n, dn,
        local_index=0, corners=corners,
    )
    np.testing.assert_array_equal(elastic, np.zeros(3))
    np.testing.assert_array_equal(anchor, np.zeros(3))
    np.testing.assert_array_equal(dn[:, 0], (1., 0., 0.))
    np.testing.assert_array_equal(work[:, 0], (-float(g), 0., 0.))
    # In contrast, common rigid spin moves source x by g*ex. The existing Dn
    # lever cancels this exactly; deleting it breaks rigid-motion work.
    manager.point_maps[0, 0, 0] = float(g)
    _, rigid_work, _ = manager._work(
        None, np.zeros(1), 1, (1., 0., 0.), n, dn,
        local_index=0, corners=corners,
    )
    np.testing.assert_array_equal(rigid_work, np.zeros((3, 1)))
    # Physical actual-skin force pair t=ex has nonzero moment g*ey. Its
    # power under omega=ey is the same g; no independent couple is included.
    np.testing.assert_array_equal(np.cross(np.array(r, dtype=float), [1., 0., 0.]),
                                  (0., float(g), 0.))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived moving-normal inputs AUTOMATIC_WORKFLOW_EVIDENCE A1/A3/A5/A6.json absent: saved-pose proof requires the original staged evidence root')
def test_moving_normal_saved_proof():
    """Recompute gates from saved poses without another coupled solve."""
    from examples.general_contact_s3_moving_normal_friction import build_case
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    from anyfesolver.general_contact.semismooth import linearize
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    saved=json.loads((evidence/'A6.json').read_text())
    case=build_case();model,definition,settings,friction,program,_=case
    normal=GeneralS3PatchManager(model,definition,settings)
    owner=create_s3_friction_manager(model,normal,friction)
    snapshots=[s for name in ('A1','A3','A5','A6')
               for s in json.loads((evidence/(name+'.json')).read_text())['snapshots']]
    history=saved['info']['force_displacement_history']
    assert [s['step_index'] for s in snapshots]==list(range(1,7))
    assert [s['step_index'] for s in history]==list(range(1,7))
    proof=[]
    for snap,phase in zip(snapshots,history):
        step=snap['step_index'];u=np.array(snap['displacement']);z=np.array(phase['friction']['supported_force_n'])
        geometric=_pose_gate(*case[:4],u,z)
        raw=linearize(owner,u,z,tangent=False)
        _,_,records=owner._force(u,z)
        structural,_,_=_assemble_nonlinear_system(model,u,{},5,tangent=False,require_full_coordinates=True)
        external=np.zeros(48)
        for load,value in zip(program.nodal_forces,program.force_values_at(step)):
            dof=model.mesh.dof_manager.get_node_dofs(load.node_id)[('ux','uy','uz').index(load.dof)]
            external[dof]+=value
        residual=structural+raw.force-external
        reactions=np.array(phase['reaction_generalized'])
        # Literal MPC/path work rows. Dependent source ux/uy rows are summed,
        # while every independent contact normal and target ux is retained.
        modes=[sum(residual[model.mesh.dof_manager.get_node_dofs(n)[a]] for n in (2,3,4))-reactions[a] for a in (0,1)]
        modes += [residual[model.mesh.dof_manager.get_node_dofs(n)[2]] for n in (2,3,4)]
        modes += [residual[model.mesh.dof_manager.get_node_dofs(n)[a]] for n in (6,7,8) for a in (0,2)]
        equilibrium=float(np.linalg.norm(modes,1)/.02)
        source,target=normal._positions(u);points=np.concatenate((source,target))
        force_rows=[];moment_rows=[];lever_errors=[]
        for node,h,point in zip((2,3,4,6,7,8),(-.025,)*3+(.015,)*3,points):
            dofs=np.array(model.mesh.dof_manager.get_node_dofs(node))
            f=raw.force[dofs[:3]];m=raw.force[dofs[3:6]]
            independent_m=np.cross(np.array([0.,0.,h]),f)
            lever_errors.extend(m-independent_m)
            force_rows.append(f);moment_rows.append(np.cross(point,f))
        force_wrench=float(np.linalg.norm(np.sum(force_rows,axis=0))/.02)
        physical_moment=float(np.linalg.norm(np.sum(moment_rows,axis=0))/.02)
        lever_error=float(np.linalg.norm(lever_errors)/.02)
        # Independent complete nodal virtual work with declared offset couples.
        variation=np.cos(np.arange(48)+.7)*1e-3
        work=0.
        for node,h,f in zip((2,3,4,6,7,8),(-.025,)*3+(.015,)*3,force_rows):
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
            target_ux=[u[model.mesh.dof_manager.get_node_dofs(n)[0]] for n in (6,7,8)])
        target_normal=np.array(geometric['target_raw_normal']);target_normal/=np.linalg.norm(target_normal)
        source_normal=np.cross(source[1]-source[0],source[2]-source[0]);source_normal/=np.linalg.norm(source_normal)
        item['source_target_normal_cosine']=float(source_normal@target_normal)
        item['finite_gap_extension_work_bound_j']=float(max(abs(g) for g in geometric['normalized_point_gap_approx'])*sum(owner.weights)*max(np.linalg.norm(r['traction_pa']) for r in records)*2.)
        proof.append(item);_save(dict(scope='saved_actual_binary_poses_independent_work',phases=proof))
        assert max(equilibrium,force_wrench,physical_moment,lever_error)<=1e-6,item
        assert work_error<=1e-5,item
        assert phase['complementarity_relative_force']<=1e-8
        assert diss>=-1e-12 and return_drop+1e-12>=diss
        assert item['finite_gap_extension_work_bound_j']<=1e-12
        owner.commit(owner.validate_successor(u,z))
    assert all(b=='open' for b in proof[4]['branch'])
    assert any(b=='slip' for b in proof[2]['branch'])
    assert max(abs(x) for x in proof[2]['target_ux'])>1e-10
    normals=[np.array(p['geometry']['target_raw_normal']) for p in proof]
    assert max(np.linalg.norm(n[:2])/np.linalg.norm(n) for n in normals)>1e-8


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'A3.json').is_file(),
    reason='archived moving-normal input AUTOMATIC_WORKFLOW_EVIDENCE/A3.json absent: rollback replay requires the original staged checkpoint')
def test_moving_normal_rollback(monkeypatch):
    import copy,pytest
    from examples.general_contact_s3_moving_normal_friction import build_case
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact.runtime import active_contact_manager
    import anyfesolver.general_contact.s3_friction_static as factory
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    checkpoint=json.loads((evidence/'A3.json').read_text())['checkpoint']
    immutable=copy.deepcopy(checkpoint);expected=checkpoint['path_state']['friction_state']
    original=factory.create_s3_friction_manager;reports=[]
    normalize=lambda x:json.loads(json.dumps(x))
    for kind in ('exception','cancellation'):
        model,definition,settings,friction,program,_=build_case()
        before=repr(model);context=active_contact_manager();token=CancellationToken();captured=[];calls=[]
        def capture(*args):
            owner=original(*args);captured.append(owner)
            restore=owner.restore;work=owner._work;armed=[False]
            def arm(*a,**k):
                value=restore(*a,**k);armed[0]=True;return value
            def inject(*a,**k):
                if armed[0] and not calls:
                    calls.append(True);assert normalize(owner.checkpoint_state())==expected
                    if kind=='exception':raise RuntimeError('injected moving-normal trial')
                    token.cancel('injected moving-normal trial cancellation')
                return work(*a,**k)
            owner.restore=arm;owner._work=inject;return owner
        with monkeypatch.context() as patch:
            patch.setattr(factory,'create_s3_friction_manager',capture)
            with pytest.raises(RuntimeError if kind=='exception' else SolveCancelled):
                solve_static_nonlinear(model,contact=definition,contact_settings=settings,
                    contact_friction=friction,contact_path_program=program,restart_checkpoint=checkpoint,
                    max_iterations=18,tolerance=1e-12,cancellation_token=token)
        assert calls and len(captured)==1
        assert normalize(captured[0].checkpoint_state())==expected
        assert repr(model)==before and active_contact_manager() is context and checkpoint==immutable
        reports.append(dict(kind=kind,trial_entered=True,committed_friction_origin_unchanged=True,
            supplied_checkpoint_unchanged=True,declared_model_unchanged=True,context_restored=True))
    _save(dict(reports=reports,scope='elastic_fixture_no_generic_hidden_material_rollback_claim'))


def test_moving_normal_refusal(monkeypatch):
    import copy,pytest
    from examples.general_contact_s3_moving_normal_friction import build_case
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact.s3_sliding_friction import S3SlidingFrictionSettings
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    case=build_case();model,definition,settings,friction,program,_=case;before=repr(model)
    token=CancellationToken();token.cancel('moving friction precancel')
    with pytest.raises(SolveCancelled):
        solve_static_nonlinear(model,contact=definition,contact_settings=settings,
            contact_friction=friction,contact_path_program=program,cancellation_token=token)
    assert repr(model)==before
    normal=GeneralS3PatchManager(model,definition,settings)
    with pytest.raises(ValueError):
        create_s3_friction_manager(model,normal,S3SlidingFrictionSettings(.3,1000.,(2e-8,)*48,145))
    import anyfesolver.general_contact.static_path as path_driver
    with monkeypatch.context() as patch:
        patch.setattr(path_driver,'_assemble_nonlinear_system',
            lambda *a,**k: (_ for _ in ()).throw(AssertionError('wrong subtype reached material')))
        with pytest.raises(ValueError,match='refined S3 projected-penalty'):
            solve_static_nonlinear(model,contact=definition,contact_settings=settings,
                contact_friction=S3SlidingFrictionSettings(.3,1000.,(2e-8,)*48,145),
                contact_path_program=program)
    owner=create_s3_friction_manager(model,normal,friction)
    origin=copy.deepcopy(owner.checkpoint_state())
    with pytest.raises(ValueError,match='partial pressure'):
        owner.validate_successor(case[5]['manufactured_pose'],np.array([.001,0.,.001]))
    assert owner.checkpoint_state()==origin
    _save(dict(precancel=True,wrong_subtype=True,partial_pressure_successor_no_mutation=True,
               scientific_force_scale_n=.02,scientific_moment_scale_nm=.02,solver_force_scale_n=.001))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'A3.json').is_file(),
    reason='archived moving-normal inputs AUTOMATIC_WORKFLOW_EVIDENCE/A3.json and default_contract_baseline.json absent: refusal replay requires the original staged checkpoint')
def test_moving_normal_archive_checkpoint_refusals():
    import copy,pytest,hashlib
    from dataclasses import replace
    from examples.general_contact_s3_moving_normal_friction import build_case
    from anyfesolver import solve_static_nonlinear
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    from test_general_contact_s3_sliding_friction import _manager
    old=_manager()[-1]
    legacy_digest=hashlib.sha256(json.dumps(old.restart_contract(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    baseline=json.loads((evidence.parent/'default_contract_baseline.json').read_text())
    assert legacy_digest==baseline['legacy_contract_sha256']
    checkpoint=json.loads((evidence/'A3.json').read_text())['checkpoint']
    for candidate in ('friction-setting','tamper'):
        m,d,s,f,p,_=build_case();before=repr(m);incoming=copy.deepcopy(checkpoint)
        if candidate=='friction-setting':f=replace(f,pressure_step_n=2e-8)
        else:incoming['path_state']['friction_state']['displacement'][0]+=.001
        with pytest.raises(ValueError):
            solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,
                contact_path_program=p,restart_checkpoint=incoming,tolerance=1e-12,max_iterations=18)
        assert repr(m)==before
    _save(dict(incompatible_friction_restart=True,tampered_checkpoint=True,legacy_contract_sha256=legacy_digest))
