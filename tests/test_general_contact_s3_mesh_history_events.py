"""Actual equilibrated pressure release; declared finite-resolution convention."""
import copy,json,os
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_s3_mesh_history_events import build_case,solve_case
from anyfesolver.general_contact import S3MeshHistoryPathFrictionSettings
from anyfesolver.general_contact.s3_friction_history_path import expand_program,guarded_invocation,register_entry
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager

def save(value):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()))
def evidence():return Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])

def test_prepare():
    case=build_case();m,d,s,f,p,reference=case
    assert reference['normal_seed_n']>0
    assert max(abs(v) for row in reference['force_endpoints'] for v in row)>=s.complementarity_force_scale_n
    save(dict(reference=reference,settings=asdict(s),friction=asdict(f),program=p.contract(),
        criteria=dict(equilibrium=1e-6,wrench=1e-6,C=1e-8,gap_m=5e-5,work_j=1e-12,tangent=1e-5)))

def test_unit():
    case=build_case();m,d,s,f,p,_=case
    expanded,identity=expand_program(p,2);assert expanded.total_steps==4
    assert identity['original_user_program']==p.contract()
    np.testing.assert_array_equal(expanded.target_at(3),(np.array(p.waypoints[1])+np.array(p.waypoints[2]))/2)
    for bad in (0,-1,True,1.5):
        with pytest.raises(ValueError):replace(f,physical_subincrements_per_user_increment=bad)
    with pytest.raises(TypeError):S3MeshHistoryPathFrictionSettings(**{k:v for k,v in asdict(f).items() if k!='physical_subincrements_per_user_increment'})
    with pytest.raises(MemoryError):expand_program(p,5001)
    normal=GeneralS3PatchManager(m,d,s);owner=create_s3_friction_manager(m,normal,f)
    original=copy.deepcopy(owner.checkpoint_state());z=normal.normal_multipliers.copy()
    with pytest.raises(RuntimeError,match='late publication'):
        def operation():
            register_entry(owner);owner.committed_displacement=np.ones(60);normal.set_normal_multipliers(np.ones_like(z))
            raise RuntimeError('late publication')
        guarded_invocation(operation)
    assert owner.checkpoint_state()==original;np.testing.assert_array_equal(normal.normal_multipliers,z)
    save(dict(expansion=True,required_ratio=True,guard_restores_origin=True,profile=owner.restart_contract()['friction_profile']))

@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE'),
    reason='installed history stage input FRICTION_STAGE absent: staged history-event solve requires the original workflow stage selection')
def test_public():
    stage=os.environ['FRICTION_STAGE'];ratio,step=map(int,stage[1:].split('-'))
    case=build_case(ratio);incoming=None
    if step>1:
        prior=f'R{ratio}-{step-1}' if stage.startswith('R') or step==2 else f'I{ratio}-{step-1}'
        incoming=json.loads((evidence()/(prior+'.json')).read_text())['checkpoint']
    result=solve_case(case,step,incoming)
    saved=dict(status=result.status,info=result.info,displacement=result.displacements,
        snapshots=[dict(step_index=x.step_index,displacement=x.displacements) for x in result.snapshots],
        checkpoint=result.to_restart_checkpoint() if result.status=='completed' else None)
    save(saved)
    assert result.status=='completed',result.info
    checkpoint=saved['checkpoint'];(evidence()/(os.environ.get('EVIDENCE_STAGE',stage)+'-checkpoint.json')).write_text(json.dumps(checkpoint,indent=2))
    assert checkpoint['path_state']['accepted_step']==step
    assert len(checkpoint['path_state']['history'])==step
    assert len(saved['snapshots'])==1
    assert result.info['contact_path']['material_history_path']['physical_subincrements_per_user_increment']==ratio
    if stage.startswith('I'):
        baseline=json.loads((evidence()/f'R{ratio}-{step}.json').read_text())
        np.testing.assert_allclose(result.displacements,baseline['displacement'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(checkpoint['path_state']['normal_forces'],baseline['checkpoint']['path_state']['normal_forces'],rtol=1e-8,atol=1e-12)
        assert checkpoint['path_state']['friction_state']==baseline['checkpoint']['path_state']['friction_state']
        assert [x['friction'] for x in checkpoint['path_state']['history']]==[x['friction'] for x in baseline['checkpoint']['path_state']['history']]

@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived history inputs AUTOMATIC_WORKFLOW_EVIDENCE R{ratio}-{step}.json absent: saved-pose proof requires the original staged evidence root')
def test_proof():
    from test_general_contact_s3_mesh_weak_pressure_friction import _weak_pose_gate
    from anyfesolver.general_contact.semismooth import linearize
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    proof=[]
    for ratio in (2,4):
        case=build_case(ratio);m,d,s,f,p,_=case;expanded,_=expand_program(p,ratio)
        owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
        for step in range(1,2*ratio+1):
            saved=json.loads((evidence()/f'R{ratio}-{step}.json').read_text());cp=saved['checkpoint']
            phase=cp['path_state']['history'][-1];u=np.array(saved['displacement']);z=np.array(cp['path_state']['normal_forces'])
            geometry=_weak_pose_gate(m,d,s,f,u,z)
            raw=linearize(owner,u,z,tangent=False)
            structural,_,_=_assemble_nonlinear_system(m,u,{},5,tangent=False,require_full_coordinates=True)
            external=np.zeros(60)
            for load,value in zip(expanded.nodal_forces,expanded.force_values_at(step)):
                external[m.mesh.dof_manager.get_node_dofs(load.node_id)[('ux','uy','uz').index(load.dof)]]+=value
            residual=structural+raw.force-external;reaction=np.array(phase['reaction_generalized'])
            modes=[sum(residual[m.mesh.dof_manager.get_node_dofs(n)[a]] for n in range(1,6))-reaction[a] for a in (0,1)]
            modes += [residual[m.mesh.dof_manager.get_node_dofs(n)[2]] for n in (2,3,4,5)]
            ru={n:residual[m.mesh.dof_manager.get_node_dofs(n)[0]] for n in (7,8,9,10)}
            modes.extend((ru[7]-ru[9],ru[8]+ru[9],ru[10]+ru[9]))
            modes.append(residual[m.mesh.dof_manager.get_node_dofs(8)[2]]+residual[m.mesh.dof_manager.get_node_dofs(9)[2]]-reaction[2])
            eq=np.linalg.norm(modes,1)/.02
            source,target=owner.normal._positions(u);rows=[];moments=[];lever=[];virtual=0.;variation=np.cos(np.arange(60))*.001
            for node,h,point in zip((2,3,4,5,7,8,9,10),(-.025,)*4+(.015,)*4,np.concatenate((source,target))):
                dd=m.mesh.dof_manager.get_node_dofs(node);F=raw.force[dd[:3]];M=raw.force[dd[3:]]
                rows.append(F);moments.append(np.cross(point,F));lever.extend(M-np.cross([0,0,h],F))
                virtual+=F@(variation[dd[:3]]+np.cross(variation[dd[3:]],[0,0,h]))
            metrics=dict(eq=float(eq),wrench=float(np.linalg.norm(np.sum(rows,axis=0))/.02),moment=float(np.linalg.norm(np.sum(moments,axis=0))/.02),lever=float(np.linalg.norm(lever)/.02),virtual=float(abs(raw.force@variation-virtual)/max(abs(virtual),1e-12)),C=phase['complementarity_relative_force'])
            item=dict(ratio=ratio,step=step,z=z,branch=phase['friction']['branch'],geometry=geometry,metrics=metrics,friction=phase['friction']);proof.append(item);save(dict(phases=proof))
            assert max(metrics[k] for k in ('eq','wrench','moment','lever'))<=1e-6,item
            assert metrics['virtual']<=1e-5 and metrics['C']<=1e-8
            assert abs(phase['friction']['endpoint_work_identity_error_j'])<=1e-12
            if ratio>1 and step==3*ratio//2:assert np.all(z==0),item
            if step==2*ratio:assert max(z)>0,item
            owner.commit(owner.validate_successor(u,z))
    save(dict(phases=proof,scope='actual_equilibrated_finite_resolution_not_event_free',hidden_pressure_release=True))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not all((Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/(name+'.json')).is_file() for name in ('R2-2','I2-3')),
    reason='archived history inputs AUTOMATIC_WORKFLOW_EVIDENCE R2-2.json/I2-3.json absent: atomic rollback replay requires the original staged checkpoints')
def test_atomic(monkeypatch):
    import anyfesolver.general_contact.static_path as path
    import anyfesolver.general_contact.s3_friction_static as factory
    from anyfesolver.control import CancellationToken
    from anyfesolver.general_contact.runtime import _ACTIVE_CONTACT
    from anyfesolver import solve_static_nonlinear
    stage=os.environ['FRICTION_STAGE'];kind=os.environ.get('ATOMIC_KIND','lease')
    incoming=json.loads((evidence()/('I2-3.json' if kind=='latecancel' else 'R2-2.json')).read_text())['checkpoint'];original=copy.deepcopy(incoming)
    case=build_case();m,d,s,f,p,_=case;declared_before=repr((m.boundary_conditions,m.constraint_equations,m.load_cases));before=repr(m);physical_before=[(m.mesh.get_node(n).x,m.mesh.get_node(n).y,m.mesh.get_node(n).z) for n in range(1,11)];program_before=p.contract();token=CancellationToken();captured=[];fault=[]
    oldfactory=factory.create_s3_friction_manager
    def capture(*args):
        owner=oldfactory(*args);captured.append(owner)
        restore=owner.restore
        def restored(*a):
            result=restore(*a);owner.entry_state=copy.deepcopy(owner.checkpoint_state());owner.entry_normal=owner.normal.normal_multipliers.copy()
            if kind in ('friction','latecancel'):
                commit=owner.commit
                def broken(v):
                    commit(v);fault.append('friction_commit_returned')
                    if kind=='latecancel':token.cancel('injected late commit cancellation');return
                    raise RuntimeError('injected post friction commit')
                owner.commit=broken
            return result
        owner.restore=restored;return owner
    monkeypatch.setattr(factory,'create_s3_friction_manager',capture)
    if kind=='structure':
        commit=path._commit_nonlinear_state_candidate
        def broken(*a,**kw):
            commit(*a,**kw);fault.append('structural_commit_returned');raise RuntimeError('injected post structure commit')
        monkeypatch.setattr(path,'_commit_nonlinear_state_candidate',broken)
    if kind in ('lease','cancel'):
        lease=path._run_with_qualified_assembly_runtime_lease
        def outer(*a,**kw):
            result=lease(*a,**kw);fault.append('lease_returned_result')
            if kind=='cancel':
                from anyfesolver.control import cancellation_safe_point
                token.cancel('injected late cancellation');cancellation_safe_point(token)
            raise RuntimeError('injected post lease return')
        monkeypatch.setattr(path,'_run_with_qualified_assembly_runtime_lease',outer)
    if kind=='result':
        def broken(*a,**kw):
            fault.append('result_constructor_reached');raise RuntimeError('injected result publication')
        monkeypatch.setattr(path,'NonlinearStaticResult',broken)
    if kind=='checkpoint':
        def broken(*a,**kw):
            fault.append('checkpoint_creation_reached');raise RuntimeError('injected checkpoint publication')
        monkeypatch.setattr(path,'create_nonlinear_checkpoint',broken)
    active=_ACTIVE_CONTACT.get()
    with pytest.raises(Exception,match='injected'):
        solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=f,contact_path_program=p,
            restart_checkpoint=incoming,max_load_factor=(1. if kind=='latecancel' else .75),tolerance=1e-12,max_iterations=18,
            emit_restart_checkpoint=True,record_increment_snapshots=True,cancellation_token=token)
    assert len(captured)==1 and len(fault)==1
    assert [(m.mesh.get_node(n).x,m.mesh.get_node(n).y,m.mesh.get_node(n).z) for n in range(1,11)]==physical_before
    assert p.contract()==program_before
    assert repr((m.boundary_conditions,m.constraint_equations,m.load_cases))==declared_before
    owner=captured[0];assert owner.checkpoint_state()==owner.entry_state
    np.testing.assert_array_equal(owner.normal.normal_multipliers,owner.entry_normal)
    assert incoming==original and repr(m)==before and _ACTIVE_CONTACT.get() is active
    save(dict(kind=kind,fault_reached=fault,actual_public_invocation_rollback=True,restored_entry=True,
        supplied_checkpoint_unchanged=True,model_repr_unchanged=True,physical_nodes_program_bc_constraints_load_declarations_unchanged=True,context_unchanged=True))

def test_safety():
    case=build_case();m,d,s,f,p,_=case
    with pytest.raises(ValueError):solve_case(case,0)


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'R2-2.json').is_file(),
    reason='archived history input AUTOMATIC_WORKFLOW_EVIDENCE/R2-2.json absent: safety replay requires the original staged checkpoint')
def test_archive_checkpoint_safety():
    case=build_case();m,d,s,f,p,_=case
    incoming=json.loads((evidence()/'R2-2.json').read_text())['checkpoint'];original=copy.deepcopy(incoming)
    changed=(*case[:3],replace(f,physical_subincrements_per_user_increment=4),*case[4:])
    with pytest.raises(ValueError):solve_case(changed,4,incoming)
    assert incoming==original
    from anyfesolver.general_contact import S3MeshWholeChordGapFrictionSettings
    old=(*case[:3],S3MeshWholeChordGapFrictionSettings(**{k:v for k,v in asdict(f).items() if k!='physical_subincrements_per_user_increment'}),*case[4:])
    from anyfesolver import solve_static_nonlinear
    with pytest.raises(ValueError):solve_static_nonlinear(m,contact=d,contact_settings=s,contact_friction=old[3],contact_path_program=p,restart_checkpoint=incoming,tolerance=1e-12,max_iterations=18)
    save(dict(ratio_incompatible=True,old_profile_incompatible=True,checkpoint_unchanged=True))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived history inputs AUTOMATIC_WORKFLOW_EVIDENCE R2-1.json/R2-2.json absent: directional replay requires the original staged evidence root')
def test_directional():
    from anyfesolver.general_contact.semismooth import linearize
    m,d,s,f,p,_=build_case();owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
    prior=json.loads((evidence()/'R2-1.json').read_text())['checkpoint']
    owner.restore(prior['path_state']['friction_state'],np.array(prior['displacements']))
    saved=json.loads((evidence()/'R2-2.json').read_text());u=np.array(saved['displacement']);z=np.array(saved['checkpoint']['path_state']['normal_forces'])
    dq=np.zeros(60)
    for node,w in zip((2,3,4,5),(.3,.4,.2,.1)):dq[m.mesh.dof_manager.get_node_dofs(node)[2]]=w*.01
    for node,w in zip((7,8,9,10),(.1,.3,.4,.2)):dq[m.mesh.dof_manager.get_node_dofs(node)[0]]=w*.01
    dz=np.array([.2,.3,.4,.1])*1e-4
    raw=linearize(owner,u,z);analytic=raw.tangent@dq+raw.force_pressure_gradient@dz;errors=[]
    for h in (2e-7,1e-7):
        plus=linearize(owner,u+h*dq,z+h*dz,tangent=False).force
        minus=linearize(owner,u-h*dq,z-h*dz,tangent=False).force
        fd=(plus-minus)/(2*h);errors.append(float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(fd),1e-12)))
    save(dict(bothbody_qz_directional=errors,criterion=1e-5,committed_history_origin='R2-1'))
    assert max(errors)<=1e-5

@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived history inputs AUTOMATIC_WORKFLOW_EVIDENCE R{ratio}-{2*ratio}.json absent: resolution replay requires the original staged evidence root')
def test_resolution():
    data=[]
    for ratio in (2,4):
        terminal=json.loads((evidence()/f'R{ratio}-{2*ratio}.json').read_text())
        history=terminal['checkpoint']['path_state']['history'];data.append(dict(ratio=ratio,
            inventory=[p['step_index'] for p in history],
            pressure=[p['friction']['supported_force_n'] for p in history],
            branches=[p['friction']['branch'] for p in history],
            friction_summaries=[p['friction'] for p in history],
            external_trapezoid=sum(p.get('external_nodal_work_trapezoid_j',0.) for p in history),
            control_trapezoid=sum(p.get('control_work_trapezoid_j',0.) for p in history),
            material_endpoint_contractions=sum(p['friction']['material_relative_endpoint_contraction_j'] for p in history),
            numerical_endpoint_contractions=sum(p['friction']['numerical_normal_variation_endpoint_contraction_j'] for p in history)))
        assert data[-1]['inventory']==list(range(1,2*ratio+1))
        assert all(v==0 for v in history[3*ratio//2-1]['friction']['supported_force_n'])
        assert max(history[-1]['friction']['supported_force_n'])>0
        assert all(b=='activation' for b in history[-1]['friction']['branch'])
    save(dict(resolutions=data,scope='finite_resolution_comparison_not_continuum_error_bound',
              endpoint_work_scope='contractions_not_integrals',quadrature_work_scope='accepted_subincrement_trapezoids'))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived history inputs AUTOMATIC_WORKFLOW_EVIDENCE R1-1.json/R1-2.json absent: comparator replay requires the original staged evidence root')
def test_comparator():
    from fractions import Fraction as F
    prior=json.loads((evidence()/'R1-1.json').read_text());refused=json.loads((evidence()/'R1-2.json').read_text())
    assert refused['status']=='stopped_at_limit' and refused['checkpoint'] is None
    assert refused['info']['contact_path']['accepted_step']==1
    np.testing.assert_array_equal(refused['displacement'],prior['displacement'])
    assert refused['info']['force_displacement_history']==prior['info']['force_displacement_history']
    m,d,s,f,p,_=build_case(1);owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
    seed=np.array(prior['displacement'])
    for node in range(1,6):seed[m.mesh.dof_manager.get_node_dofs(node)[0]]=.125
    for node in (8,9):seed[m.mesh.dof_manager.get_node_dofs(node)[2]]=2.**-14
    points=owner.normal._positions(seed)[0];unsafe=[]
    for adapter in owner.target_adapters.values():
        t=[[F(float(x)) for x in row] for row in adapter.current_positions(seed)]
        a=[t[1][i]-t[0][i] for i in range(3)];b=[t[2][i]-t[0][i] for i in range(3)]
        n=[a[i]*b[j]-a[j]*b[i] for i,j in ((1,2),(2,0),(0,1))];nn=sum(x*x for x in n)
        for point in points:
            h=sum((F(float(point[i]))-t[0][i])*n[i] for i in range(3))
            if h<0 and h*h>F(float(5e-5))**2*nn:unsafe.append(dict(numerator=str(h),norm_squared=str(nn),gap_approx_m=float(h)/float(nn)**.5))
    assert unsafe
    save(dict(endpoint_only_status=refused['status'],no_second_commit=True,last_valid_prefix_published=True,
        seed_independent_binary_physical_gap_violation=unsafe,
        refusal=refused['info']['contact_path_predictor_failure'],
        terminal_history_energy_comparison='UNMET: endpoint-only lift unsafe under unchanged guard',
        discrete_history_resolution_scope='ratios2/4 establish equilibrated release/rebirth, not continuum truth'))
