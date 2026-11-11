"""Connected finite fixed-tool development checks; no acceptance claim."""
import copy,json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from examples.general_contact_finite_v2d_rigid_patch import build,solve,fixture,json_value,report_result
from anyfesolver.general_contact.finite_rigid_patch import FiniteRigidPatchManager,PROFILE,admitted_definition
from anyfesolver.general_contact.finite_rigid_support import slab_support_and_ownership,tool_normal_gap_lower


def oracle():
    import importlib.util
    spec=importlib.util.spec_from_file_location('independent_finite_rigid',Path(__file__).with_name('finite_v2d_rigid_patch_oracle.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def emit(name,value):
    root=Path(os.environ.get('FINITE_RIGID_DETAILS',os.environ.get('FINITE_RIGID_WORKFLOW','.')))
    root.mkdir(parents=True,exist_ok=True);(root/(name+'.json')).write_text(json.dumps(json_value(value),allow_nan=False,indent=2))


def test_declared_penalty_exact_owner_and_work_maps():
    m,d,s,p,q=build();manager=FiniteRigidPatchManager(m,d,s,q)
    assert admitted_definition(m,d,s) and manager.profile==PROFILE
    assert d.interactions[0].normal_law=='penalty' and not manager.augmented
    assert manager.n_dof==24 and len(manager.law.source_node_ids)==4
    pose=manager.finite_skin.pose(q)
    assert np.max(np.abs((pose.coordinate_map-pose.spin_map).toarray()))>1e-3
    assert np.count_nonzero(pose.coordinate_map.toarray()[12:])==0
    assert manager.point_gap_lower(q)>=-5e-5


@pytest.mark.parametrize('kind',('native','affine','pressure','chart'))
def test_refuse_unsupported_owner_or_settings(kind):
    from dataclasses import replace
    m,d,s,p,q=build()
    if kind=='native':
        from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
        old=m.mesh.elements[1]
        m.mesh.elements[1]=QualifiedE4PLS3ShellElement(1,old.node_ids,'elastic',thickness=.05,reference_normal=(0.,0.,1.))
    elif kind=='affine':s=replace(s,shell_skin_kinematics='reference_linear')
    elif kind=='pressure':s=replace(s,q4_source_pressure_basis='selected_q1_knots_dev',q4_source_pressure_knot_columns=(0,1))
    else:s=replace(s,shell_pair_orientation='physical_sides_dev')
    assert not admitted_definition(m,d,s)


def test_fresh_negative_seed_refused_and_pending_checkpoint_blocked():
    m,d,s,p,q=build();normal=np.array((np.sin(.5),0.,np.cos(.5)))
    negative=q.copy()
    for nid in (1,2,3,4):negative[m.mesh.dof_manager.get_node_dofs(nid)[:3]]-=2e-5*normal
    with pytest.raises(ValueError,match='initial or committed'):FiniteRigidPatchManager(m,d,s,negative)
    pending=dict(checkpoint_sha256='0'*64,displacements=negative.tolist())
    manager=FiniteRigidPatchManager(m,d,s,negative,_pending_checkpoint=pending)
    with pytest.raises(ValueError,match='pending'):manager.evaluate(negative)
    with pytest.raises(ValueError,match='pending'):manager._swept_admissible(negative,negative)
    with pytest.raises(ValueError,match='differs'):manager.admit_validated_committed_state(SimpleNamespace(payload={'checkpoint_sha256':'1'*64},displacements=negative))


def test_tool_normal_conversion_relative_rotation_and_error_refusal():
    # Actual source-normal gap stays bounded under relative tilt. Distant
    # fixed-target vertices are not source-face material points.
    angle=.08;source=np.array(((0.,0.,2e-6),(1.,0.,2e-6),(0.,1.,2e-6)))
    source[:,2]+=np.sin(angle)*source[:,0]
    target=np.array(((-10.,-10.,0.),(10.,-10.,0.),(-10.,10.,0.)))
    zero=np.zeros_like(source);reference=np.array(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)))
    assert tool_normal_gap_lower(source,source,target,reference,target,endpoint_errors=(zero,zero))>=0
    uncertain=np.full_like(source,1.)
    assert tool_normal_gap_lower(source,source,target,reference,target,endpoint_errors=(uncertain,uncertain))< -5e-5


def synthetic_manager(targets):
    points=np.vstack(targets)
    law=SimpleNamespace(source_face_indices=(np.array((0,1,2)),),target_face_indices=tuple(np.arange(3*i,3*i+3) for i in range(len(targets))))
    return SimpleNamespace(law=law,reference=(None,points))


def test_supported_endpoints_do_not_certify_disconnected_interior():
    targets=[np.array(((x,-1.,0.),(x+1.,-1.,0.),(x,.5,0.))) for x in (0.,3.)]
    manager=synthetic_manager(targets);a=np.array(((.1,-.8,-1e-6),(.4,-.8,-1e-6),(.1,-.4,-1e-6)));b=a+np.array((3.,0.,0.));zero=np.zeros_like(a)
    assert slab_support_and_ownership(manager,a,a,zero,zero)
    assert slab_support_and_ownership(manager,b,b,zero,zero)
    assert not slab_support_and_ownership(manager,a,b,zero,zero)


def test_positive_support_does_not_hide_second_component_rim():
    targets=[np.array(((-5.,-1.,0.),(5.,-1.,0.),(5.,1.5,0.))),np.array(((-5.,-1.,0.),(5.,1.5,0.),(-5.,1.5,0.))),np.array(((-.04,1.92,0.),(.04,1.92,0.),(0.,2.08,0.)))]
    manager=synthetic_manager(targets);middle=np.array(((-.8,0.,-1e-6),(.8,0.,-1e-6),(0.,2.,-1e-6)));a=middle-np.array((3.,0.,0.));b=middle+np.array((3.,0.,0.));zero=np.zeros_like(a)
    assert slab_support_and_ownership(manager,a,a,zero,zero)
    assert slab_support_and_ownership(manager,b,b,zero,zero)
    assert not slab_support_and_ownership(manager,a,b,zero,zero)


def test_independent_complete_penalty_work_and_additive_tangent():
    m,d,s,p,q=build();manager=FiniteRigidPatchManager(m,d,s,q);normal=np.array((np.sin(.5),0.,np.cos(.5)))
    for nid in (1,2,3,4):q[m.mesh.dof_manager.get_node_dofs(nid)[:3]]-=2e-5*normal
    production,H,_=manager.evaluate(q);ref=oracle().response(m,d.rigid_bodies[0],q,manager.stiffness)
    error=float(np.linalg.norm(production-ref['force'],1)/1e-4);assert error<=1e-6
    v=np.linspace(-.4,.7,24);v/=np.linalg.norm(v);measurements=[]
    for h in (1e-5,3e-6):
        plus=oracle().response(m,d.rigid_bodies[0],q+h*v,manager.stiffness)
        minus=oracle().response(m,d.rigid_bodies[0],q-h*v,manager.stiffness)
        assert plus['signature']==ref['signature']==minus['signature']
        fd=(plus['force']-minus['force'])/(2*h);actual=H@v
        tangent_error=float(np.linalg.norm(actual-fd)/max(np.linalg.norm(actual),np.linalg.norm(fd),1e-12))
        assert tangent_error<=1e-5;measurements.append(dict(h=h,tangent_error=tangent_error))
    emit('independent_work_tangent',dict(complete24_force_normalized_L1=error,steps=measurements,scope='disposable active candidate; unchanged spatial-spin B/additive A'))


def fresh():
    data=json.loads(Path(os.environ['FINITE_RIGID_FRESH']).read_text());assert data['status']=='completed'
    return data


@pytest.mark.skipif(not os.environ.get('FINITE_RIGID_FRESH')
    or not Path(os.environ['FINITE_RIGID_FRESH']).is_file(),
    reason='archived fresh run under FINITE_RIGID_FRESH absent: saved gap/wrench/penalty/integration assessment requires the saved accepted suffix and history')
def test_saved_independent_gap_equilibrium_wrench_penalty_and_integration():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate,_weighted_external_load_system
    m,d,s,p,seed=build();m.apply_boundary_conditions();data=fresh();assert data['fixture']==json_value(fixture(m,d,s,p,seed))
    poses=data['accepted_suffix'];rows=data['history'];assert len(poses)==len(rows)>0
    assert [v['step_index'] for v in poses]==[v['step_index'] for v in rows]==list(range(1,len(poses)+1))
    factors=[float(v['load_factor']) for v in poses]
    assert all(a<b for a,b in zip(factors,factors[1:])) and abs(factors[-1]-4.)<=1e-12
    assert all(any(abs(value-target)<=1e-12 for value in factors) for target in (1.,2.,3.,4.))
    reports=[];fixed=m.mesh.dof_manager.get_node_dofs(1)[:6];free=np.setdiff1d(np.arange(24),fixed)
    td=[x for nid in (1,2,3,4) for x in m.mesh.dof_manager.get_node_dofs(nid)[:3]];rd=[x for nid in (1,2,3,4) for x in m.mesh.dof_manager.get_node_dofs(nid)[3:6]]
    for snapshot,row in zip(poses,rows):
        q=np.asarray(snapshot['displacements']);assert q.shape==(24,) and np.isfinite(q).all();np.testing.assert_array_equal(q[fixed],seed[fixed])
        manager=FiniteRigidPatchManager(m,d,s,seed);force,_,contact=manager.evaluate(q,tangent=False)
        internal,_,trial=_assemble_nonlinear_system(m,q,{},5,tangent=False,kinematics='corotational',corotational_tangent='consistent',require_full_coordinates=True)
        _discard_nonlinear_state_candidate(trial)
        stage=float(row['load_factor']);assert abs(stage-float(snapshot['load_factor']))<=1e-12
        weights=p.stage_factors(stage)
        external,_=_weighted_external_load_system(m,[(st.load_case,weights[st.name]) for st in p.stages],q,tangent=False)
        residual=np.asarray(internal).reshape(-1)+force-external;projected=np.zeros(24);projected[free]=residual[free]
        eqf=float(np.linalg.norm(projected[td],1)/1e-4);eqm=float(np.linalg.norm(projected[rd],1)/1e-4)
        independent=oracle().response(m,d.rigid_bodies[0],q,manager.stiffness);gap=oracle().gap_enclosure(m,d.rigid_bodies[0],q)
        support=np.asarray(contact.reference_support_m2);pressure=np.asarray(contact.numerical_pressure_pa);g=independent['weighted_gap']/independent['support']
        penalty=float(np.linalg.norm((pressure-manager.stiffness*np.maximum(-g,0))*support,1)/1e-4)
        integration=float(np.linalg.norm(force-independent['force'],1)/1e-4)
        controls=[]
        for order in (4,8):
            quadrature=oracle().response(m,d.rigid_bodies[0],q,manager.stiffness,order=order)
            assert quadrature['signature']==independent['signature']
            control=float(np.linalg.norm(quadrature['force']-independent['force'],1)/1e-4)
            assert control<=1e-7
            controls.append(dict(order=order,complete_force_L1=control))
        mass=float(np.max(np.abs(support-independent['support'])/independent['support']))
        positions=manager.finite_skin.pose(q).positions;physical=-manager.law.work(positions[:4],positions[4:],tangent=False).residual.reshape(-1,3)
        wrenchf=np.linalg.norm(physical.sum(axis=0),1)/1e-4;wrenchm=np.linalg.norm(np.cross(positions,physical).sum(axis=0),1)/1e-4
        assert row['stage_factors']==p.stage_factors(stage)
        reports.append(dict(stage=stage,eq_force=eqf,eq_moment=eqm,penalty_relation_L1=penalty,integration=integration,quadrature_controls=controls,mass=mass,gap=gap,wrench_force=float(wrenchf),wrench_moment=float(wrenchm)))
        emit('saved_assessment_progress',dict(status='INCOMPLETE',measured_poses=reports))
        assert eqf<=1e-6 and eqm<=1e-6 and penalty<=1e-8 and integration<=1e-6 and mass<=1e-11 and gap['lower_m']>=-5e-5
        assert wrenchf<=1e-6 and wrenchm<=1e-6
        if abs(stage-3.)<=1e-12:assert gap['lower_m']>0 and np.max(pressure)==0
        elif any(abs(stage-value)<=1e-12 for value in (1.,2.,4.)):assert np.max(pressure)>0
    bending=[v for nid in (2,3,4) for v in m.mesh.dof_manager.get_node_dofs(nid)[3:5]]
    last=np.asarray(poses[-1]['displacements']);response=float(np.max(np.abs(last[bending]-seed[bending])))
    gamma=512*np.finfo(float).eps/(1-512*np.finfo(float).eps);roundoff=gamma*max(1.,float(np.max(np.abs(last[bending]))))
    assert response>max(1e-10,roundoff),'existing finite free-bending witness must exceed roundoff'
    emit('saved_assessment',dict(status='PASS',poses=reports,source_free_bending_rad=response,roundoff_rad=roundoff,inherited_witness_rad=1e-10,enforcement='penalty; penalty-relation check is not AL complementarity'))


@pytest.mark.parametrize('prefix_name,start_factor',(('active-prefix2.json',2.),('open-prefix.json',3.)))
def test_actual_active_and_open_restart(prefix_name,start_factor):
    from anyfesolver.general_contact.finite_v2d_state import prepare_v2d_states
    from examples.general_contact_finite_v2d_rigid_patch import main
    fresh_path=os.environ.get('FINITE_RIGID_FRESH')
    if not fresh_path or not Path(fresh_path).is_file():
        pytest.skip('archived fresh run under FINITE_RIGID_FRESH absent: active/open restart replay requires the saved fresh displacements and diagnostics')
    root_env=os.environ.get('FINITE_RIGID_WORKFLOW')
    if not root_env or not (Path(root_env)/prefix_name).is_file():
        pytest.skip('archived '+prefix_name+' under FINITE_RIGID_WORKFLOW absent: active/open restart replay requires the saved prefix checkpoint')
    data=fresh();root=Path(root_env);prefix_path=root/prefix_name
    prefix=json.loads(prefix_path.read_text());before=json.dumps(prefix,sort_keys=True)
    assert prefix['status']=='completed' and prefix['checkpoint']['path_state']['load_factor']==start_factor
    assert bool(np.max(prefix['diagnostics']['general_contact']['numerical_pressure_pa'])>0)==(start_factor==2.)
    details=Path(os.environ.get('FINITE_RIGID_DETAILS',str(root)));details.mkdir(parents=True,exist_ok=True)
    suffix_path=details/(prefix_name.removesuffix('.json')+'_public_suffix.json')
    assert main(['--restart',str(prefix_path),'--output',str(suffix_path)])==0
    result=json.loads(suffix_path.read_text());snapshots=result['accepted_suffix']
    assert result['status']=='completed',result['failure_reason']
    assert snapshots and all(v['load_factor']>start_factor for v in snapshots)
    assert abs(snapshots[-1]['load_factor']-4.)<=1e-12
    assert [v['step_index'] for v in snapshots]==list(range(prefix['checkpoint']['path_state']['step_index']+1,prefix['checkpoint']['path_state']['step_index']+1+len(snapshots)))
    np.testing.assert_allclose(result['displacements'],data['displacements'],rtol=1e-8,atol=1e-12)
    def supported(info):
        c=info['general_contact'];assert c['source_pressure_node_ids']==[1,2,3,4]
        return np.asarray(c['numerical_pressure_pa'])*np.asarray(c['reference_support_m2'])
    np.testing.assert_allclose(supported(result['diagnostics']),supported(data['diagnostics']),rtol=1e-7,atol=1e-12)
    m,d,s,p,q=build();cp=result['checkpoint'];states={v['element_id']:v['state'] for v in cp['element_states']}
    prepare_v2d_states(m,np.asarray(cp['displacements']),states,5,{},restart=True)
    terminal_path=details/(prefix_name.removesuffix('.json')+'_public_terminal0.json')
    assert main(['--restart',str(suffix_path),'--output',str(terminal_path)])==0
    restored=json.loads(terminal_path.read_text());assert restored['status']=='completed' and not restored['accepted_suffix']
    np.testing.assert_array_equal(restored['displacements'],result['displacements'])
    bare_path=details/(prefix_name.removesuffix('.json')+'_actual_bare_checkpoint.json')
    bare_path.write_text(json.dumps(cp,allow_nan=False))
    bare_terminal=details/(prefix_name.removesuffix('.json')+'_bare_terminal0.json')
    assert main(['--restart',str(bare_path),'--output',str(bare_terminal)])==0
    bare=json.loads(bare_terminal.read_text());assert bare['status']=='completed' and not bare['accepted_suffix']
    np.testing.assert_array_equal(bare['displacements'],result['displacements'])
    assert json.dumps(json.loads(prefix_path.read_text()),sort_keys=True)==before
    emit(prefix_name.removesuffix('.json')+'_restart_summary',dict(status='PASS',actual_public_rich_advancement=True,
        suffix_indices=[v['step_index'] for v in snapshots],rich_terminal0=True,bare_terminal0=True,
        supported_force_rtol=1e-7,supported_force_atol=1e-12,u_rtol=1e-8,u_atol=1e-12,
        actual_typed_owners=len(states),owner_state_scope='model-bound existing V2D validator; no checkpoint reseal'))


def test_actual_finite_director_curvature_hidden_crossing_refused():
    m,d,s,p,q=build();normal=np.array((np.sin(.5),0.,np.cos(.5)));alpha=.09
    start=q.copy();end=q.copy();shift=20e-6-.035*(1-np.cos(alpha))-2e-7
    for nid in (1,2,3,4):
        ids=m.mesh.dof_manager.get_node_dofs(nid)
        start[ids[:3]]+=shift*normal;end[ids[:3]]+=shift*normal
        start[ids[4]]-=alpha;end[ids[4]]+=alpha
    manager=FiniteRigidPatchManager(m,d,s,start)
    assert manager.point_gap_lower(start)>=-5e-5 and manager.point_gap_lower(end)>=-5e-5
    assert manager.point_gap_lower((start+end)/2)<-5e-5
    assert not manager._swept_admissible(start,end)


def test_actual_relative_finite_source_tool_plane_admitted():
    m,d,s,p,q=build();initial=oracle().skins(m,q)[1];anchor=initial[0];angle=.42
    c,t=np.cos(angle),np.sin(angle);rotation=np.array(((c,0.,t),(0.,1.,0.),(-t,0.,c)))
    reference=oracle().skins(m,np.zeros(24))[0]
    for i,nid in enumerate((1,2,3,4)):
        ids=m.mesh.dof_manager.get_node_dofs(nid);node=m.mesh.nodes[nid];x=np.array((node.x,node.y,node.z))
        physical=anchor+rotation@(reference[i]-reference[0]);director=rotation@np.array((0.,0.,1.))
        q[ids[:3]]=physical-x+.035*director;q[ids[3:6]]=(0.,angle,0.)
    manager=FiniteRigidPatchManager(m,d,s,q)
    assert manager.point_gap_lower(q)>=-5e-5
    assert manager._swept_admissible(q,q)
    emit('relative_finite_geometry',dict(relative_rotation_rad=.08,scope='disposable relative rigid pose; no structural trajectory claim'))


def test_exact_artificial_diagonal_touch_stays_visible_conservative_refusal():
    # A zero-area neighbour touch can be owned by the represented triangle at
    # endpoints. This sufficient path proof intentionally refuses the tie.
    target=[np.array(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))),np.array(((1.,0.,0.),(1.,1.,0.),(0.,1.,0.)))]
    source=np.array(((.1,.1,-1e-6),(.5,.1,-1e-6),(.5,.5,-1e-6)));zero=np.zeros_like(source)
    assert not slab_support_and_ownership(synthetic_manager(target),source,source,zero,zero)


@pytest.mark.skipif(not os.environ.get('FINITE_RIGID_WORKFLOW')
    or not (Path(os.environ['FINITE_RIGID_WORKFLOW'])/'open-prefix.json').is_file(),
    reason='archived open-prefix.json under FINITE_RIGID_WORKFLOW absent: material/contact cancellation rollback requires the saved prefix checkpoint')
def test_actual_cancellation_after_contact_and_material_trial(monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled
    import anyfesolver.nonlinear_static as driver
    from anyfesolver.general_contact.core import _ACTIVE_CONTACT
    root=Path(os.environ['FINITE_RIGID_WORKFLOW']);prefix=json.loads((root/'open-prefix.json').read_text())
    m,d,s,p,q=build();token=CancellationToken();before=json.dumps(prefix,sort_keys=True);context=_ACTIVE_CONTACT.get();observed={}
    assembly=driver._assemble_nonlinear_system_unchecked;evaluate=FiniteRigidPatchManager.evaluate
    def state_bytes(states):return json.dumps(json_value(dict(states)),sort_keys=True,allow_nan=False)
    def material(*args,**kwargs):
        store=args[2];observed['store']=store;observed['before']=state_bytes(store)
        value=assembly(*args,**kwargs);observed['trial']=value[2] is not None;return value
    def contact(manager,*args,**kwargs):
        value=evaluate(manager,*args,**kwargs);observed['contact']=True
        token.cancel('actual ordinary finite rigid material/contact rollback');return value
    with monkeypatch.context() as local:
        local.setattr(driver,'_assemble_nonlinear_system_unchecked',material)
        local.setattr(FiniteRigidPatchManager,'evaluate',contact)
        with pytest.raises(SolveCancelled) as exc:solve(m,d,s,p,q,restart_checkpoint=prefix['checkpoint'],cancellation_token=token)
    assert observed['trial'] and observed['contact']
    assert state_bytes(observed['store'])==observed['before']
    assert json.dumps(prefix,sort_keys=True)==before and _ACTIVE_CONTACT.get() is context
    assert driver._assemble_nonlinear_system_unchecked is assembly and FiniteRigidPatchManager.evaluate is evaluate
    emit('cancellation',dict(actual_material_and_contact_trial=True,committed_store_rollback=True,checkpoint_input_unchanged=True,context_and_hooks_restored=True,stage=exc.value.stage))


@pytest.mark.skipif(not os.environ.get('FINITE_RIGID_FRESH')
    or not Path(os.environ['FINITE_RIGID_FRESH']).is_file(),
    reason='archived fresh run under FINITE_RIGID_FRESH absent: owner-mutation and corrupt rich checkpoint refusals require the saved checkpoint and fixture')
def test_actual_checkpoint_owner_mutation_and_null_input_refused(monkeypatch,tmp_path):
    from anyfesolver.general_contact.finite_v2d_state import prepare_v2d_states
    from examples.general_contact_finite_v2d_rigid_patch import main,SCHEMA
    data=fresh();m,d,s,p,q=build();cp=data['checkpoint'];states={v['element_id']:copy.deepcopy(v['state']) for v in cp['element_states']}
    states[1]['committed_total_u'][0]+=1e-4
    with pytest.raises((ValueError,TypeError)):prepare_v2d_states(m,np.asarray(cp['displacements']),states,5,{},restart=True)
    import anyfesolver.nonlinear_static as driver
    from anyfesolver.general_contact.core import _ACTIVE_CONTACT
    context=_ACTIVE_CONTACT.get();assembly=driver._assemble_nonlinear_system_unchecked
    corrupted=copy.deepcopy(data);corrupted['checkpoint']['element_states'][0]['state']['committed_total_u'][0]+=1e-4
    bad_owner=tmp_path/'corrupt-owner.json';bad_owner.write_text(json.dumps(corrupted,allow_nan=False))
    owner_output=tmp_path/'owner-refusal.json'
    assert main(['--restart',str(bad_owner),'--output',str(owner_output)])==1
    owner_refusal=json.loads(owner_output.read_text())
    assert owner_refusal['status']=='NONPASS' and not owner_refusal.get('returned_accepted_suffix')
    assert 'checkpoint' in owner_refusal['failure_reason'].lower()
    assert _ACTIVE_CONTACT.get() is context and driver._assemble_nonlinear_system_unchecked is assembly
    bad=tmp_path/'missing.json';bad.write_text(json.dumps(dict(schema=SCHEMA,fixture=data['fixture'],checkpoint=None)))
    import examples.general_contact_finite_v2d_rigid_patch as entry
    def forbidden(*args,**kwargs):raise AssertionError('solve must not launch on null checkpoint')
    with monkeypatch.context() as local:
        local.setattr(entry,'solve',forbidden);output=tmp_path/'refusal.json'
        assert main(['--restart',str(bad),'--output',str(output)])==1
    refusal=json.loads(output.read_text());assert refusal['status']=='NONPASS' and 'checkpoint missing' in refusal['failure_reason']
    emit('owner_refusal',dict(incoming_owner_seal_retained=True,finite_committed_coordinate_mutation_refused=True,
        actual_corrupt_rich_checkpoint_refusal=True,zero_accepted_suffix=True,exception_context_restored=True,
        scope='Existing model-bound owner validator plus real public integrity refusal; no forged/resealed checkpoint',null_checkpoint_no_solve=True))


def test_actual_null_checkpoint_refused_without_saved_run(monkeypatch,tmp_path):
    from examples.general_contact_finite_v2d_rigid_patch import main,SCHEMA
    import examples.general_contact_finite_v2d_rigid_patch as entry
    m,d,s,p,q=build()
    bad=tmp_path/'missing.json';bad.write_text(json.dumps(dict(schema=SCHEMA,fixture=json_value(fixture(m,d,s,p,q)),checkpoint=None),allow_nan=False))
    def forbidden(*args,**kwargs):raise AssertionError('solve must not launch on null checkpoint')
    with monkeypatch.context() as local:
        local.setattr(entry,'solve',forbidden);output=tmp_path/'refusal.json'
        assert main(['--restart',str(bad),'--output',str(output)])==1
    refusal=json.loads(output.read_text());assert refusal['status']=='NONPASS' and 'checkpoint missing' in refusal['failure_reason']
    emit('null_input_refusal',dict(null_checkpoint_no_solve=True,fixture_identity_accepted=True,
        scope='current public-entry null-checkpoint refusal; input-free, no archived fresh run required'))


def test_spatial_spin_additive_chart_work_counterexample_one_radian():
    m,d,s,p,q=build();manager=FiniteRigidPatchManager(m,d,s,q)
    coordinate=np.zeros(24)
    for nid in (1,2,3,4):coordinate[m.mesh.dof_manager.get_node_dofs(nid)[5]]=1.
    pose=manager.finite_skin.pose(coordinate);local=np.zeros(24);local[1]=1.
    dof=m.mesh.dof_manager.get_node_dofs(1)[3];h=-.035
    actual=float((pose.coordinate_map.T@local)[dof]);spatial=float((pose.spin_map.T@local)[dof])
    assert abs(actual-(-h*np.sin(1.)))<1e-14 and abs(spatial-(-h))<1e-14
    assert abs(actual-spatial)>1e-3
    emit('chart_work_counterexample',dict(theta_rad=1.,true_additive_work=actual,spatial_spin_work=spatial,scope='map counterexample, no committed structural state or trajectory'))


def test_actual_public_entry_selects_only_admitted_ordinary_finite_profile(monkeypatch):
    import anyfesolver.nonlinear_static as driver
    from anyfesolver.general_contact.core import active_contact_manager
    from dataclasses import replace
    m,d,s,p,q=build();seen=[]
    class ReachedOrdinaryOwner(Exception):pass
    def reached(*args,**kwargs):
        manager=active_contact_manager();assert manager.profile==PROFILE
        seen.append(manager.profile);raise ReachedOrdinaryOwner
    with monkeypatch.context() as local:
        local.setattr(driver,'_solve_static_nonlinear_under_lease',reached)
        with pytest.raises(ReachedOrdinaryOwner):solve(m,d,s,p,q,through_stage=1)
        with pytest.raises(ValueError,match='shell_skin_kinematics requires contact_path_program'):
            solve(m,d,replace(s,shell_pair_orientation='physical_sides_dev'),p,q,through_stage=1)
    assert seen==[PROFILE]
    assert active_contact_manager() is None


def test_public_stage1_driver_sentinel_is_refused_before_solve(monkeypatch,tmp_path):
    import examples.general_contact_finite_v2d_rigid_patch as entry
    def forbidden(*args,**kwargs):raise AssertionError('stage1 must not launch the full programme')
    monkeypatch.setattr(entry,'solve',forbidden)
    output=tmp_path/'sentinel-refusal.json'
    with pytest.raises(SystemExit) as exc:
        entry.main(['--through-stage','1','--output',str(output)])
    assert exc.value.code==2 and not output.exists()
