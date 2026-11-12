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


def test_mesh_unit():
    from examples.general_contact_s3_mesh_moving_friction import build_case,TILT
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager,S3_MESH_MONOLITHIC_PROFILE
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    from anyfesolver.general_contact.semismooth import linearize,reference_nodal_support
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    model,definition,settings,friction,program,reference=build_case()
    normal=GeneralS3PatchManager(model,definition,settings)
    assert normal.profile==S3_MESH_MONOLITHIC_PROFILE
    manager=create_s3_friction_manager(model,normal,friction)
    u=reference['manufactured_pose'];z=reference['supported_force']
    manager.require_friction_path(np.zeros(60),u)
    support=reference_nodal_support(manager.law)
    np.testing.assert_allclose(support,[.01/6,.01/3,.01/6,.01/3],rtol=1e-14,atol=1e-17)
    raw=linearize(manager,u,z,tangent=False)
    expected=reference['manufactured_force']-assemble_stiffness_matrix(model)[0]@u
    translations=np.asarray([model.mesh.dof_manager.get_node_dofs(i)[:3] for i in (2,3,4,5,7,8,9,10)]).reshape(-1)
    np.testing.assert_allclose(raw.force[translations],expected[translations],rtol=1e-6,atol=1e-12)
    manager.commit(manager.validate_successor(u,z))
    slip=u.copy()
    for node in range(1,6):
        dofs=model.mesh.dof_manager.get_node_dofs(node);slip[dofs[0]]+=.39
        if node!=1:slip[dofs[2]]-=TILT*.39
    manager.require_friction_path(u,slip)
    base=linearize(manager,slip,z,tangent=True)
    v=np.zeros(60)
    for node in range(1,6):v[model.mesh.dof_manager.get_node_dofs(node)[0]]=.001
    for node in (2,3,4,5):
        p=model.mesh.get_node(node);v[model.mesh.dof_manager.get_node_dofs(node)[2]]=(.2+p.x*.1+p.y*.3)*1e-3
    for node in (7,8,9,10):
        p=model.mesh.get_node(node);d=model.mesh.dof_manager.get_node_dofs(node)
        v[d[0]]=(.1+p.x*.2)*1e-3;v[d[2]]=(.2+p.x*.1+p.y*.3)*1e-3
    w=np.array([.2,-.3,.4,.1])*1e-3;errors=[]
    for h in (1e-4,5e-5):
        plus=linearize(manager,slip+h*v,z+h*w,tangent=False)
        minus=linearize(manager,slip-h*v,z-h*w,tangent=False)
        fd=(plus.force-minus.force)/(2*h);exact=base.tangent@v+base.force_pressure_gradient@w
        errors.append(float(np.linalg.norm(fd-exact)/max(np.linalg.norm(fd),1e-12)))
    assert max(errors)<=1e-5
    # Shared nodes3/5 couple the two source faces; complete q-z directions
    # include both bodies and all four supported pressure coefficients.
    initial=manager._force(u,z)[2];moved=manager._force(slip,z)[2]
    assert any(a['target_element']!=b['target_element'] for a,b in zip(initial,moved))
    gram=normal.law.geometry.evaluate(*normal._positions(slip)).pressure_gram
    assert np.linalg.matrix_rank(gram)==4 and np.min(support)>0
    _save(dict(profile=manager.profile,support=support,directional_errors=errors,
       initial_facets=[r['target_element'] for r in initial],moved_facets=[r['target_element'] for r in moved],
       pressure_gram_rank=4,scope='independent_P1_seed_and_full_shared_qz_direction_at_cross_facet_overlap'))


def _pose_gate(model,definition,settings,friction,u,z):
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    normal=GeneralS3PatchManager(model,definition,settings);manager=create_s3_friction_manager(model,normal,friction)
    source,target=normal._positions(u);ids=normal.law.source_node_ids+normal.law.target_node_ids
    independent=[]
    for node,h in zip(ids,(-.025,)*len(source)+(.015,)*len(target)):
        p=model.mesh.get_node(node);d=model.mesh.dof_manager.get_node_dofs(node)
        independent.append(np.array([p.x,p.y,p.z])+u[d[:3]]+h*(np.array([0.,0.,1.])+np.cross(u[d[3:6]],[0.,0.,1.])))
    np.testing.assert_allclose(np.concatenate((source,target)),independent,rtol=0,atol=1e-16)
    exact=lambda rows:[[F(float(x)) for x in row] for row in rows]
    def sub(a,b):return [x-y for x,y in zip(a,b)]
    def cross(a,b):return [a[i]*b[j]-a[j]*b[i] for i,j in ((1,2),(2,0),(0,1))]
    dense=manager.point_reference+manager.point_maps@np.asarray(u)[manager.work_columns]
    points=exact(np.concatenate((source,dense)));report=[]
    normals=[]
    for adapter in manager.target_adapters.values():
        c=exact(adapter.current_positions(u));n=cross(sub(c[1],c[0]),sub(c[2],c[0]));nn=sum(x*x for x in n)
        gaps=[sum(x*y for x,y in zip(sub(p,c[0]),n)) for p in points]
        assert nn>0
        if np.all(np.asarray(z)==0):assert min(gaps)>0
        else:
            assert min(z)>0
            assert all(x*x<=F(1,10**10)**2*nn for x in gaps)
        report.append(dict(element=adapter.element_id,exact_numerator=[str(x) for x in gaps],exact_normal_squared=str(nn),normalized_point_gap_approx=[float(x)/np.sqrt(float(nn)) for x in gaps]))
        normals.append(np.array(n,dtype=float))
    summed=np.sum(normals,axis=0)
    return dict(all_facets=report,target_raw_normal=summed.tolist())


@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE'),
    reason='installed friction stage input FRICTION_STAGE absent: staged mesh moving-friction solve requires the original workflow stage selection')
def test_mesh_public():
    from examples.general_contact_s3_mesh_moving_friction import build_case,solve_case
    # Instrument only the decisive repair capture, retaining incremental
    # on-disk counters even if the resource guard terminates the process.
    if os.environ.get('MESH_CAPTURE')=='1':
        import time
        import anyfesolver.general_contact.s3_friction_static as factory
        original=factory.create_s3_friction_manager;counter={};events=[]
        path=Path(os.environ['AUTOMATIC_RUNTIME_RESULT']).parent/'diagnostic.json'
        def capture(*args):
            owner=original(*args)
            for name in ('require_friction_path','linearize_contact','_force','_work'):
                method=getattr(owner,name)
                def timed(*a,_name=name,_method=method,**k):
                    started=time.perf_counter()
                    try:return _method(*a,**k)
                    finally:
                        item=counter.setdefault(_name,dict(count=0,seconds=0.));item['count']+=1;item['seconds']+=time.perf_counter()-started
                        if _name=='linearize_contact':events.append(dict(u_norm=float(np.linalg.norm(a[0])),z=np.asarray(a[1]).tolist(),tangent=k.get('tangent',True)))
                        path.write_text(json.dumps(dict(counters=counter,events=events),indent=2))
                setattr(owner,name,timed)
            return owner
        factory.create_s3_friction_manager=capture
        import sys
        def trace(frame,event,arg):
            if event=='line' and frame.f_code.co_filename.endswith('semismooth_path.py'):
                if frame.f_lineno in (327,370,415,450):
                    values={k:frame.f_locals[k] for k in ('iteration','attempt','fraction','merit','new_merit','candidate_error','backtracks') if k in frame.f_locals}
                    events.append(dict(line=frame.f_lineno,values=values))
                    path.write_text(json.dumps(dict(counters=counter,events=events),indent=2,default=str))
            return trace
        sys.settrace(trace)
    stage=os.environ['FRICTION_STAGE'];evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    chain={'A1':(1,None),'A2':(2,'A1'),'A3':(3,'A2'),'A5':(5,'A3'),'A6':(6,'A5'),
           'diagnostic-A3':(3,'A2'),'B2':(2,None),'B4':(4,'A3'),'B6':(6,'B4')}
    case=build_case();through=1 if stage in ('smoke','capture') else 3 if stage=='prefix' else 5 if stage=='open-prefix' else 6
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


@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE') or not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='saved mesh friction inputs FRICTION_STAGE/AUTOMATIC_WORKFLOW_EVIDENCE absent: saved-pose proof requires the original staged evidence root')
def test_mesh_saved_proof():
    """Recompute gates from saved poses without another coupled solve."""
    from examples.general_contact_s3_mesh_moving_friction import build_case
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    from anyfesolver.general_contact.semismooth import linearize
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    prefix=os.environ['FRICTION_STAGE']=='prefix-proof'
    saved=json.loads((evidence/('A2.json' if prefix else 'A6.json')).read_text())
    case=build_case();model,definition,settings,friction,program,_=case
    normal=GeneralS3PatchManager(model,definition,settings)
    owner=create_s3_friction_manager(model,normal,friction)
    snapshots=[s for name in (('A1','A2') if prefix else ('A1','A2','A3','A5','A6'))
               for s in json.loads((evidence/(name+'.json')).read_text())['snapshots']]
    history=saved['info']['force_displacement_history']
    assert [s['step_index'] for s in snapshots]==list(range(1,3 if prefix else 7))
    assert [s['step_index'] for s in history]==list(range(1,3 if prefix else 7))
    proof=[]
    for snap,phase in zip(snapshots,history):
        step=snap['step_index'];u=np.array(snap['displacement']);z=np.array(phase['friction']['supported_force_n'])
        geometric=_pose_gate(*case[:4],u,z)
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
        proof.append(item);_save(dict(scope='saved_actual_binary_poses_independent_work',phases=proof))
        assert max(equilibrium,force_wrench,physical_moment,lever_error)<=1e-6,item
        assert work_error<=1e-5,item
        assert phase['complementarity_relative_force']<=1e-8
        assert diss>=-1e-12 and return_drop+1e-12>=diss
        assert item['finite_gap_extension_work_bound_j']<=1e-12
        owner.commit(owner.validate_successor(u,z))
    if prefix:
        _save(dict(phases=proof,scope='accepted_press_stick_endpoint_work_not_full_cycle',unmet=['slip','reversal','open','recontact','integrated_facet_transfer','integrated_normal_delta']))
        return
    assert all(b=='open' for b in proof[4]['branch'])
    assert any(b=='slip' for b in proof[2]['branch'])
    assert max(abs(x) for x in proof[2]['target_ux'])>1e-10
    normals=[np.array(p['geometry']['target_raw_normal']) for p in proof]
    normalized=[n/np.linalg.norm(n) for n in normals]
    assert np.linalg.norm(normalized[2]-normalized[0])>1e-12
    assert any(a!=b for a,b in zip(history[0]['friction']['target_elements'],history[2]['friction']['target_elements']))
    _save(dict(phases=proof,target_normal_delta=normalized[2]-normalized[0],material_facet_transfer=True,scope='accepted_endpoint_work_bound_not_path_integral'))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'A2.json').is_file(),
    reason='archived mesh friction input AUTOMATIC_WORKFLOW_EVIDENCE/A2.json absent: rollback replay requires the original staged checkpoint')
def test_mesh_rollback(monkeypatch):
    import copy,pytest
    from examples.general_contact_s3_mesh_moving_friction import build_case
    from anyfesolver import solve_static_nonlinear
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact.runtime import active_contact_manager
    import anyfesolver.general_contact.s3_friction_static as factory
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    checkpoint=json.loads((evidence/'A2.json').read_text())['checkpoint']
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


def test_mesh_refusal(monkeypatch):
    import copy,pytest
    from examples.general_contact_s3_mesh_moving_friction import build_case
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
    import anyfesolver.general_contact.static_path as owned_driver
    with monkeypatch.context() as patch:
        patch.setattr(owned_driver,'_assemble_nonlinear_system',lambda *a,**k:(_ for _ in ()).throw(AssertionError('normal-only mesh reached material')))
        with pytest.raises(ValueError,match='requires exact S3MeshMovingNormalFrictionSettings'):
            solve_static_nonlinear(model,contact=definition,contact_settings=settings,contact_path_program=program,tolerance=1e-12,max_iterations=18)
    normal=GeneralS3PatchManager(model,definition,settings)
    with pytest.raises(ValueError):
        create_s3_friction_manager(model,normal,S3SlidingFrictionSettings(.3,1000.,(2e-8,)*60,201))
    import anyfesolver.general_contact.static_path as path_driver
    with monkeypatch.context() as patch:
        patch.setattr(path_driver,'_assemble_nonlinear_system',
            lambda *a,**k: (_ for _ in ()).throw(AssertionError('wrong subtype reached material')))
        with pytest.raises(ValueError,match='refined S3 projected-penalty'):
            solve_static_nonlinear(model,contact=definition,contact_settings=settings,
                contact_friction=S3SlidingFrictionSettings(.3,1000.,(2e-8,)*60,201),
                contact_path_program=program)
    owner=create_s3_friction_manager(model,normal,friction)
    origin=copy.deepcopy(owner.checkpoint_state())
    with pytest.raises(ValueError,match='partial pressure'):
        owner.validate_successor(case[5]['manufactured_pose'],np.array([.001,0.,.001,.001]))
    assert owner.checkpoint_state()==origin
    _save(dict(precancel=True,wrong_subtype=True,partial_pressure_successor_no_mutation=True,
               scientific_force_scale_n=.02,scientific_moment_scale_nm=.02,solver_force_scale_n=.01))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'A2.json').is_file(),
    reason='archived mesh friction inputs AUTOMATIC_WORKFLOW_EVIDENCE/A2.json and default_contract_baseline.json absent: refusal replay requires the original staged checkpoint')
def test_mesh_archive_checkpoint_refusals():
    import copy,pytest,hashlib
    from dataclasses import replace
    from examples.general_contact_s3_mesh_moving_friction import build_case
    from anyfesolver import solve_static_nonlinear
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
    from test_general_contact_s3_sliding_friction import _manager
    old=_manager()[-1]
    legacy_digest=hashlib.sha256(json.dumps(old.restart_contract(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    baseline=json.loads((evidence.parent/'default_contract_baseline.json').read_text())
    assert legacy_digest==baseline['legacy_contract_sha256']
    checkpoint=json.loads((evidence/'A2.json').read_text())['checkpoint']
    for candidate in ('friction-setting','tamper'):
        m,d,s,f,p,_=build_case();before=repr(m);incoming=copy.deepcopy(checkpoint)
        if candidate=='friction-setting':f=replace(f,pressure_step_n=2e-8)
        else:incoming['path_state']['friction_state']['displacement'][0]+=.001
        with pytest.raises(ValueError):
            solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,
                contact_path_program=p,restart_checkpoint=incoming,tolerance=1e-12,max_iterations=18)
        assert repr(m)==before
    _save(dict(incompatible_friction_restart=True,tampered_checkpoint=True,legacy_contract_sha256=legacy_digest))


def test_mesh_certificate_controls(monkeypatch):
    import pytest
    from examples.general_contact_s3_mesh_moving_friction import build_case
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
    import anyfesolver.general_contact.s3_friction_mesh_monolithic as module
    from anyfesolver.general_contact.s3_friction_mesh_coverage import mesh_moving_coverage,_chart_check
    case=build_case();m,d,s,f,p,r=case;owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
    original=module.mesh_moving_coverage;calls=[]
    def counted(*a):calls.append(True);return original(*a)
    with monkeypatch.context() as patch:
        patch.setattr(module,'mesh_moving_coverage',counted)
        owner.require_friction_path(np.zeros(60),r['manufactured_pose'])
        owner.require_friction_path(np.zeros(60),r['manufactured_pose'])
        assert len(calls)==1
        changed=r['manufactured_pose'].copy();changed[m.mesh.dof_manager.get_node_dofs(2)[0]]+=1e-9
        owner.require_friction_path(np.zeros(60),changed)
        assert len(calls)==2
        # Authoritative normal-map mutation must not reuse the cached branch.
        owner.normal.map.data[0]+=1e-9
        with pytest.raises(ValueError,match='prepared mesh normal geometry changed'):
            owner.require_friction_path(np.zeros(60),r['manufactured_pose'])
        assert len(calls)==2
    for kind in ('point_reference','point_maps','target_maps','weights','adapter_work'):
        m,d,s,f,p,r=build_case();changed_owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
        if kind=='weights':changed_owner.weights[0]*=2
        elif kind=='target_maps':
            changed_owner.target_maps[5]=changed_owner.target_maps[5].copy();changed_owner.target_maps[5][0,0,0]+=1e-9
        elif kind=='adapter_work':changed_owner.source_adapters[0].work_map.data[0]+=1e-9
        else:
            value=list(getattr(changed_owner,kind));value[0]=value[0].copy();value[0].flat[0]+=1e-9;setattr(changed_owner,kind,tuple(value))
        with pytest.raises(ValueError,match='prepared mesh friction geometry changed'):
            changed_owner.require_friction_path(np.zeros(60),r['manufactured_pose'])
    m,d,s,f,p,r=build_case();pair_owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
    pair_owner.normal.law.geometry.pair_laws[0].reference_area*=2
    with pytest.raises(ValueError,match='prepared mesh normal geometry changed'):
        pair_owner.require_friction_path(np.zeros(60),r['manufactured_pose'])
    # Endpoint-plane identity does not establish coplanarity of its chord.
    last=np.zeros(60)
    last[m.mesh.dof_manager.get_node_dofs(8)[2]]=1.
    last[m.mesh.dof_manager.get_node_dofs(9)[0]]=.1
    last[m.mesh.dof_manager.get_node_dofs(9)[2]]=1.1
    assert F(11,20)-F(21,20)*F(1,2)==F(1,40)
    assert not mesh_moving_coverage(owner,np.zeros(60),last)
    # Positive consecutive turns of a pentagram do not imply convexity.
    xy=[(np.cos(2*np.pi*k/5),np.sin(2*np.pi*k/5),0.) for k in (0,2,4,1,3)]
    poly=lambda p:tuple(((F(float(x)),F(float(x))), (F(0),F(0))) for x in p)
    rim=[poly(v) for v in xy];origin=poly((0.,0.,0.))
    facets=[(origin,rim[k],rim[(k+1)%5]) for k in range(5)]
    source=[poly((-.01,-.01,0.)),poly((.01,-.01,0.)),poly((0.,.01,0.))]
    assert not _chart_check(source,facets,rim,())
    # Exercise the same guard through the actual public path, before material.
    import anyfesolver.general_contact.s3_friction_static as factory
    import anyfesolver.general_contact.static_path as driver
    from anyfesolver import solve_static_nonlinear
    import copy
    original_factory=factory.create_s3_friction_manager
    m,d,s,f,p,r=build_case();before=repr(m);captured=[];origins=[]
    def corrupted(*args):
        owner=original_factory(*args);captured.append(owner);origins.append(copy.deepcopy(owner.checkpoint_state()));owner.normal.map.data[0]+=1e-9;return owner
    with monkeypatch.context() as patch:
        patch.setattr(factory,'create_s3_friction_manager',corrupted)
        patch.setattr(driver,'_assemble_nonlinear_system',lambda *a,**k:(_ for _ in ()).throw(AssertionError('mutated geometry reached material')))
        with pytest.raises(ValueError,match='prepared mesh normal geometry changed'):
            solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,contact_path_program=p,tolerance=1e-12,max_iterations=18)
    assert repr(m)==before and captured and captured[0].checkpoint_state()==origins[0]
    _save(dict(public_mutated_geometry_refusal_before_material=True,identical_binary_chord_cache=True,changed_chord_rechecked=True,
        nonplanar_affine_chord_refused=True,self_crossing_star_refused=True))
