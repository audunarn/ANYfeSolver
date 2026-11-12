"""Focused mixed finite-owner caller, work, state and refusal checks."""
import copy
import json
import math
import os
from dataclasses import replace,asdict
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import pytest
from importlib.util import spec_from_file_location,module_from_spec
from examples.general_contact_finite_v2d_pair import build, solve, rigid_pose
from anyfesolver.general_contact.finite_v2d_state import (
    PROFILE, STATIC_PROFILE, COORDINATES, owner_contract, admitted_definition,
    v2d_state_contract, prepare_v2d_states, seal_v2d_states,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize, complementarity, reference_nodal_support

FREF=1e-4;MREF=1e-4


def get_oracle():
    # Only output-dependent assessment imports this independent reference.
    spec=spec_from_file_location('finite_mixed_independent_oracle',Path(__file__).with_name('general_contact_finite_v2d_pair_oracle.py'))
    module=module_from_spec(spec);spec.loader.exec_module(module)
    return module


def workflow_fresh(root):
    return json.loads((root/os.environ.get('FINITE_V2D_FRESH_NAME','fresh.json')).read_text())


def require_saved_fixture(saved,model,definition,settings,program):
    from examples.general_contact_finite_v2d_pair import SCHEMA
    from anyfesolver.nonlinear_restart import model_configuration_fingerprint
    canonical=lambda v:json.loads(json.dumps(v))
    assert saved['schema']==SCHEMA and saved['status']=='completed'
    fixture=saved['fixture']
    assert (fixture['nodes'],fixture['elements'],fixture['dofs'])==(8,4,48)
    assert fixture['contact']==canonical(asdict(definition))
    assert fixture['settings']==canonical(asdict(settings)) and fixture['program']==canonical(program.contract())
    checkpoint=saved['checkpoint'];assert checkpoint['model_fingerprint']==model_configuration_fingerprint(model)
    assert np.array_equal(np.asarray(saved['displacements']),np.asarray(checkpoint['displacements']))
    assert np.array_equal(np.asarray(saved['accepted_suffix'][-1]['displacements']),np.asarray(checkpoint['displacements']))


def independent_director(theta,dref):
    """Independent decimal exponential series; no production rotation kernel."""
    with localcontext() as context:
        context.prec=80
        a=[Decimal(float(v)) for v in theta]
        term=[Decimal(float(v)) for v in dref];total=term.copy()
        for n in range(1,100):
            x,y,z=term;ax,ay,az=a
            term=[(ay*z-az*y)/n,(az*x-ax*z)/n,(ax*y-ay*x)/n]
            total=[v+w for v,w in zip(total,term)]
        return np.array([float(v) for v in total])


def finite_contract(model):
    return dict(profile=PROFILE,control='contact_path',element_state_coordinates=COORDINATES,
        finite_state_owners=owner_contract(model),shell_skin_kinematics='corotational_nodal_director_dev',
        structural_kinematics='corotational',corotational_tangent='consistent',
        enforcement_solver='monolithic_active_set',
        finite_rigid_seed='common_reference_frame_left_transport_project_controls_v1')


def emit(name,value):
    if os.environ.get('FINITE_V2D_DIAGNOSTICS'):
        root=Path(os.environ['FINITE_V2D_DIAGNOSTICS']);root.mkdir(parents=True,exist_ok=True)
        (root/(name+'.json')).write_text(json.dumps(value,indent=2,
            default=lambda v:v.tolist() if hasattr(v,'tolist') else v),encoding='utf-8')


def test_exact_mixed_fixture_and_inventory():
    model,definition,settings,program=build()
    assert model.mesh.dof_manager.total_dofs==48
    assert len(model.mesh.nodes)==8 and len(model.mesh.elements)==4
    assert program.total_steps==16 and len(program.controls)==21
    assert admitted_definition(model,definition,settings)
    assert v2d_state_contract(finite_contract(model))
    owners=owner_contract(model)
    assert [v['owner'] for v in owners]==['native_parity_s3_v2d']*4
    # The first fixture's counts are not the production state contract.
    reduced=finite_contract(model);reduced['finite_state_owners']=owners[::2]
    assert v2d_state_contract(reduced)
    manager=GeneralS3PatchManager(model,definition,settings)
    assert manager.profile==STATIC_PROFILE and manager.finite_skin is not None
    assert manager.law.pressure_mode_ids==(1,2,3,4)
    assert [len(v) for v in manager.law.source_face_indices]==[3,3]
    assert [len(v) for v in manager.law.target_face_indices]==[3,3]
    np.testing.assert_allclose(manager.finite_skin.offsets[:4],-.035,rtol=0,atol=1e-17)
    np.testing.assert_allclose(manager.finite_skin.offsets[4:],.015,rtol=0,atol=1e-17)


@pytest.mark.parametrize('change',('coordinate','duplicate','empty','wrong_owner'))
def test_typed_contract_refuses_mutation(change):
    model,*_=build();contract=finite_contract(model)
    if change=='coordinate':contract['element_state_coordinates']='global_all_elements'
    elif change=='duplicate':contract['finite_state_owners'].append(contract['finite_state_owners'][-1])
    elif change=='empty':contract['finite_state_owners']=[]
    else:contract['finite_state_owners'][-1]['owner']='qualified_s3_companion'
    with pytest.raises(ValueError):v2d_state_contract(contract)


@pytest.mark.parametrize('change',('linear_skin','native_companion','orphan','shared_nodes','selected_pressure'))
def test_mixed_admission_refuses_scope_change(change):
    from anyfesolver import create_shell_element
    model,definition,settings,_=build()
    if change=='linear_skin':settings=replace(settings,shell_skin_kinematics='reference_linear')
    elif change=='selected_pressure':settings=replace(settings,q4_source_pressure_basis='selected_q1_knots_dev',q4_source_pressure_knot_columns=(0,1))
    elif change=='orphan':definition=replace(definition,surfaces=(definition.surfaces[0],replace(definition.surfaces[1],element_ids=(3,))))
    elif change=='shared_nodes':model.mesh.elements[3].node_ids=(1,6,8)
    else:
        from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
        model.mesh.elements[3]=QualifiedE4PLS3ShellElement(3,(5,6,8),'elastic',thickness=.05,
                                                  reference_normal=(0.,0.,1.))
        assert type(model.mesh.elements[3]) is QualifiedE4PLS3ShellElement
        assert model.mesh.elements[3].formulation_native_total_lagrangian is True
    assert not admitted_definition(model,definition,settings)


def test_p1_admission_refuses_nondefault_q4_knots_directly():
    model,definition,settings,_=build()
    # Isolate the new predicate from the settings constructor/core validator.
    # Such a forged Q4 control must not be silently ignored by a P1 owner.
    incompatible=copy.deepcopy(settings)
    object.__setattr__(incompatible,'q4_source_pressure_knot_columns',(0,1))
    assert incompatible.q4_source_pressure_basis=='fine_nodal_q1'
    assert not admitted_definition(model,definition,incompatible)


def test_in_cap_spatial_spin_is_not_additive_chart_work():
    model,definition,settings,_=build();manager=GeneralS3PatchManager(model,definition,settings)
    skin=manager.finite_skin;q=np.zeros(48)
    # Counterexample uses an actual admitted V2D target skin node and theta=1.
    index=list(manager.law.source_node_ids+manager.law.target_node_ids).index(8)
    dofs=skin.dofs[index];q[dofs[5]]=1.
    pose=skin.pose(q);f=np.zeros_like(pose.positions);f[index,1]=1.
    chart_delta=np.zeros(48);chart_delta[dofs[3]]=1.
    spatial=pose.spin_map.T@f.ravel();h=skin.offsets[index]
    expected=-h*math.sin(1.)
    actual=float(f.ravel()@(pose.coordinate_map@chart_delta))
    assert abs(actual-expected)<=1e-14
    assert abs(float(spatial@chart_delta)+h)<=1e-14
    assert abs(actual-float(spatial@chart_delta))>1e-4
    independent=np.array([independent_director(q[d[3:]],ref) for d,ref in zip(skin.dofs,skin.directors)])
    np.testing.assert_allclose(pose.directors,independent,rtol=0,atol=1e-14)
    emit('spin_chart',dict(theta_rad=1.,physical_offset_m=h,additive_work=actual,
                          naive_spatial_dot_chart=float(spatial@chart_delta),expected_work=expected))


def test_actual_both_body_contact_and_structural_q_z_tangent():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    model,definition,settings,_=build();manager=GeneralS3PatchManager(model,definition,settings)
    q=rigid_pose(model,.6);normal=np.array((math.sin(.6),0.,math.cos(.6)))
    for nid in range(1,5):q[model.mesh.dof_manager.get_node_dofs(nid)[:3]]-=2e-6*normal
    manager.require_finite_deformation(q)
    z=np.full(4,1e-5);c=1000.*reference_nodal_support(manager.law)
    raw=linearize(manager,q,z)
    assert raw.gradient_only_pairs==0, 'full mixed Hessian branch unresolved'
    _,active,cq,cz=complementarity(raw.state,z,c,raw.gap_gradient)
    rng=np.random.default_rng(704);dq=rng.normal(size=48);dq/=np.linalg.norm(dq)
    dz=rng.normal(size=4)*1e-5
    def structural(at,tangent):
        force,k,state=_assemble_nonlinear_system(model,at,{},5,tangent=tangent,
                               kinematics='corotational',corotational_tangent='consistent')
        _discard_nonlinear_state_candidate(state)
        return force,k
    rs,ks=structural(q,True);rows=[]
    derivative=raw.tangent@dq-raw.force_gradient.T@dz
    expected=np.r_[ks@dq+derivative,cq@dq+cz*dz]
    for h in (1e-5,3e-6):
        plus=linearize(manager,q+h*dq,z+h*dz,tangent=False)
        minus=linearize(manager,q-h*dq,z-h*dz,tangent=False)
        cp,ap,_,_=complementarity(plus.state,z+h*dz,c,plus.gap_gradient,tangent=False)
        cn,an,_,_=complementarity(minus.state,z-h*dz,c,minus.gap_gradient,tangent=False)
        assert np.array_equal(active,ap) and np.array_equal(active,an)
        contact_fd=(plus.force-minus.force)/(2*h)
        full_fd=np.r_[(structural(q+h*dq,False)[0]+plus.force-structural(q-h*dq,False)[0]-minus.force)/(2*h),(cp-cn)/(2*h)]
        contact_error=float(np.linalg.norm(derivative-contact_fd)/max(np.linalg.norm(derivative),1e-12))
        full_error=float(np.linalg.norm(expected-full_fd)/max(np.linalg.norm(expected),1e-12))
        rows.append(dict(h=h,contact_tangent_error=contact_error,full_tangent_error=full_error))
    emit('coupled_tangent',dict(rows=rows,active=active,gradient_only_pairs=raw.gradient_only_pairs))
    assert all(r['contact_tangent_error']<=1e-5 and r['full_tangent_error']<=1e-5 for r in rows)


def test_typed_actual_owner_roundtrip_and_mutation():
    from anyfesolver.corotational import corotational_element_response
    model,*_=build();q=rigid_pose(model,.6);states={}
    for eid,element in model.mesh.elements.items():
        dofs=element.get_dof_mapping(model.mesh)
        _,_,states[eid]=corotational_element_response(model,eid,element,q[dofs],True,
                                                     tangent_mode='consistent')
    info={};sealed=seal_v2d_states(model,q,states,5,info)
    restored=prepare_v2d_states(model,q,sealed,5,{},restart=True)
    assert info['element_state_coordinates']==COORDINATES
    for eid in (1,2,3,4):
        element=model.mesh.elements[eid];material=model.get_material(element.material_name)
        assert restored[eid]['solver_kinematics']=='COROTATIONAL'
        before=element.serialize_nonlinear_state(model.mesh,material,sealed[eid],5)
        assert element.serialize_nonlinear_state(model.mesh,material,restored[eid],5)==before
        dofs=element.get_dof_mapping(model.mesh)
        first=corotational_element_response(model,eid,element,q[dofs],True,sealed[eid],tangent_mode='consistent')
        second=corotational_element_response(model,eid,element,q[dofs],True,restored[eid],tangent_mode='consistent')
        np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])
        assert element.serialize_nonlinear_state(model.mesh,material,sealed[eid],5)==before
    changed=copy.deepcopy(sealed);changed[3]['committed_constitutive_u'][0]+=1e-7
    with pytest.raises(ValueError):prepare_v2d_states(model,q,changed,5,{},restart=True)
    with pytest.raises(ValueError):prepare_v2d_states(model,q,{1:sealed[1]},5,{},restart=True)
    emit('typed_state',dict(owners=owner_contract(model),records=len(sealed),global_coordinate_tag='COROTATIONAL'))


def test_triangle_whole_path_refuses_crossing_and_preserves_state():
    from anyfesolver.general_contact.finite_rotation_sweep import finite_q4_path_admissible
    model,definition,settings,_=build();manager=GeneralS3PatchManager(model,definition,settings)
    start=np.zeros(48);end=start.copy()
    for nid in range(1,5):end[model.mesh.dof_manager.get_node_dofs(nid)[2]]=-1e-4
    before=manager.normal_multipliers.copy()
    assert finite_q4_path_admissible(manager,start,start)
    assert not finite_q4_path_admissible(manager,start,end)
    np.testing.assert_array_equal(manager.normal_multipliers,before)


def test_legacy_q4_profile_keeps_its_actual_manager():
    from examples.general_contact_tilted_relative_sliding import build as old_build
    from anyfesolver.general_contact.s3_patch_static import FINITE_Q4_Q4_MONOLITHIC_PROFILE
    model,_,settings,resolution,_=old_build()
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    assert manager.profile==FINITE_Q4_Q4_MONOLITHIC_PROFILE
    assert all(len(face)==4 for face in manager.law.target_face_indices)


def test_internal_diagonal_tie_retains_full_tangent_refusal():
    from anyfesolver.general_contact.s3_overlap_derivatives import UnsupportedOverlapBranch
    model,definition,settings,_=build()
    for nid in range(1,5):model.mesh.nodes[nid].y-=.03
    manager=GeneralS3PatchManager(model,definition,settings)
    assert manager.law.shared_edge_tie is None
    try:
        raw=linearize(manager,np.zeros(48),np.full(4,1e-5),tangent=True)
    except UnsupportedOverlapBranch:
        return
    assert raw.gradient_only_pairs>0, 'tied full-Hessian branch must remain unresolved'


def independent_physical_gap_lower(model,q):
    return get_oracle().gap_bound(model,q)


@pytest.mark.skipif(not os.environ.get('FINITE_V2D_WORKFLOW'),
    reason='archived workflow input FINITE_V2D_WORKFLOW absent: saved-fixture replay requires the original delivery root')
def test_saved_complete_workflow_independent_skin_gap():
    root=Path(os.environ['FINITE_V2D_WORKFLOW']);saved=workflow_fresh(root)
    model,definition,settings,program=build()
    require_saved_fixture(saved,model,definition,settings,program)
    assert saved['status']=='completed' and len(saved['accepted_suffix'])==16
    assert [s['step_index'] for s in saved['accepted_suffix']]==list(range(1,17))
    lower=[]
    for snapshot in saved['accepted_suffix']:
        q=np.asarray(snapshot['displacements']);assert q.shape==(48,) and np.isfinite(q).all()
        assert snapshot['element_state_ids']==[1,2,3,4]
        lower.append(independent_physical_gap_lower(model,q))
    emit('physical_gap',dict(scope='independent conservative P1/P1 whole-face lower on all returned poses',
                            lower_m=lower,gate_m=-5e-5))
    assert all(v['gate_pass'] for v in lower)


@pytest.mark.skipif(not os.environ.get('FINITE_V2D_WORKFLOW'),
    reason='archived workflow input FINITE_V2D_WORKFLOW absent: saved-fixture replay requires the original delivery root')
def test_saved_actual_scaled_equilibrium_wrench_ncp_and_integration():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    oracle=get_oracle()
    root=Path(os.environ['FINITE_V2D_WORKFLOW']);saved=workflow_fresh(root)
    model,definition,settings,program=build();model.apply_boundary_conditions()
    require_saved_fixture(saved,model,definition,settings,program)
    manager=GeneralS3PatchManager(model,definition,settings)
    C=np.zeros((len(program.controls),model.mesh.dof_manager.total_dofs));names=('ux','uy','uz','rx','ry','rz')
    for row,control in enumerate(program.controls):
        for nid,name,weight in control.terms:C[row,model.mesh.dof_manager.get_node_dofs(nid)[names.index(name)]]+=weight
    fixed=model.mesh.dof_manager.get_node_dofs(6)[1];free=np.setdiff1d(np.arange(model.mesh.dof_manager.total_dofs),[fixed])
    force_dofs=[d for nid in model.mesh.nodes for d in model.mesh.dof_manager.get_node_dofs(nid)[:3]]
    moment_dofs=[d for nid in model.mesh.nodes for d in model.mesh.dof_manager.get_node_dofs(nid)[3:]]
    history=saved['history'];assert len(history)==16
    assert [int(row['step_index']) for row in history]==list(range(1,17))
    reports=[]
    for snapshot,row in zip(saved['accepted_suffix'],history):
        step=snapshot['step_index'];q=np.asarray(snapshot['displacements']);contact=row['contact']
        assert q.shape==(48,) and np.isfinite(q).all() and step==row['step_index']
        assert np.max(np.abs(C@q-program.target_at(step)))<=5e-12 and q[fixed]==0.
        assert contact['source_pressure_node_ids']==[1,2,3,4]
        pressure=np.asarray(contact['numerical_pressure_pa']);saved_support=np.asarray(contact['reference_support_m2'])
        assert pressure.shape==saved_support.shape==(4,) and np.isfinite(np.r_[pressure,saved_support]).all()
        assert np.all(saved_support>0) and np.all(pressure>=0)
        z=pressure*saved_support;raw=linearize(manager,q,z,tangent=False)
        support=np.asarray(raw.state.reference_support)
        assert np.all(support>0) and np.max(np.abs(saved_support-support)/support)<=1e-11
        internal,_,trial=_assemble_nonlinear_system(model,q,{},5,tangent=False,
            kinematics='corotational',corotational_tangent='consistent',require_full_coordinates=True)
        _discard_nonlinear_state_candidate(trial)
        external=np.zeros(48)
        for force,value in zip(program.nodal_forces,program.force_values_at(step)):
            external[model.mesh.dof_manager.get_node_dofs(force.node_id)[names.index(force.dof)]]+=value
        mu=np.asarray(row['reaction_generalized']);assert mu.shape==(21,) and np.isfinite(mu).all()
        residual=np.asarray(internal).reshape(-1)+raw.force-external-C.T@mu
        projected=np.zeros(48);projected[free]=residual[free]
        eq_force=float(np.linalg.norm(projected[force_dofs],1)/FREF)
        eq_moment=float(np.linalg.norm(projected[moment_dofs],1)/MREF)
        c=1000.*reference_nodal_support(manager.law)
        ncp,active,_,_=complementarity(raw.state,z,c,raw.gap_gradient,tangent=False)
        ncp_error=float(np.linalg.norm(ncp,1)/FREF)
        if step<=12 or step==15:
            assert np.linalg.norm(z,1)/FREF<=1e-8,'registered open phase must release pressure'
        else:
            assert np.max(z)>0,'registered press/slide/recontact phase must carry actual pressure'
        pose=manager.finite_skin.pose(q);physical=-np.asarray(raw.local_force).reshape(-1,3)
        fs=physical[:4].sum(axis=0);ft=physical[4:].sum(axis=0)
        ms=np.cross(pose.positions[:4],physical[:4]).sum(axis=0)
        mt=np.cross(pose.positions[4:],physical[4:]).sum(axis=0)
        force_balance=float(np.linalg.norm(fs+ft,1)/FREF);moment_balance=float(np.linalg.norm(ms+mt,1)/MREF)
        nodal=raw.force[manager.finite_skin.dofs]
        mid=manager.finite_skin.reference+q[manager.finite_skin.dofs[:,:3]]
        spin_balance=float(np.linalg.norm(np.cross(mid,-nodal[:,:3]).sum(axis=0)-nodal[:,3:].sum(axis=0),1)/MREF)
        independent,_=oracle.vertices(model,q)
        integration=[]
        for order in (4,8):
            base=oracle.integral(model,independent,z,order);measurements=[]
            assert not base['ties'],'independent integration branch tie unresolved'
            for h in (1e-5,3e-6):
                derivatives=[]
                for direction in range(6):
                    delta={nid:np.zeros(3) for nid in independent}
                    for nid in range(1,5):
                        delta[nid]=(np.eye(3)[direction] if direction<3
                                    else np.cross(np.eye(3)[direction-3],independent[nid]))
                    plus=oracle.integral(model,{nid:p+h*delta[nid] for nid,p in independent.items()},z,order)
                    minus=oracle.integral(model,{nid:p-h*delta[nid] for nid,p in independent.items()},z,order)
                    assert plus['signature']==base['signature']==minus['signature'] and not plus['ties'] and not minus['ties']
                    derivatives.append((plus['work']-minus['work'])/(2*h))
                reference_force=-np.asarray(derivatives[:3]);reference_moment=-np.asarray(derivatives[3:])
                measurements.append(dict(h=h,force=reference_force,moment=reference_moment,
                    force_error=float(np.linalg.norm(reference_force-fs,1)/FREF),
                    moment_error=float(np.linalg.norm(reference_moment-ms,1)/MREF)))
            integration.append(dict(order=order,support_relative_max=float(np.max(np.abs(base['support']-support)/base['support'])),
                measurements=measurements,signature=base['signature'],chart_certificate=base['chart_certificate']))
        uncertainty=[]
        for a,b in zip(integration[0]['measurements'],integration[1]['measurements']):
            uncertainty.append(dict(h=a['h'],force=float(np.linalg.norm(a['force']-b['force'],1)/FREF),
                                    moment=float(np.linalg.norm(a['moment']-b['moment'],1)/MREF)))
        report=dict(step=step,eq_force_l1=eq_force,eq_moment_l1=eq_moment,ncp_l1=ncp_error,
            physical_force_balance_l1=force_balance,physical_moment_balance_l1=moment_balance,
            nodal_spatial_moment_balance_l1=spin_balance,integration=integration,quadrature_control=uncertainty,
            Fref_N=FREF,Mref_Nm=MREF,Lref_m=1.,reaction_scope='stored producer reactions with fresh current structural/contact assembly',
            reference_scope='independent fixed-z finite-skin P1/P1 quadrature; empirical order/step controls')
        reports.append(report);emit('scaled_workflow',dict(complete=False,poses=reports))
        assert np.isfinite(np.r_[internal,raw.force,residual,ncp]).all()
        assert max(eq_force,eq_moment,force_balance,moment_balance,spin_balance)<=1e-6 and ncp_error<=1e-8
        assert all(v['support_relative_max']<=1e-11 for v in integration)
        assert all(max(v['force_error'],v['moment_error'])<=1e-6 for order in integration for v in order['measurements'])
        assert all(max(v['force'],v['moment'])<=1e-7 for v in uncertainty)
    assert len(reports)==16
    emit('scaled_workflow',dict(complete=True,poses=reports))


@pytest.mark.skipif(not os.environ.get('FINITE_V2D_WORKFLOW'),
    reason='archived workflow input FINITE_V2D_WORKFLOW prefix.json absent: cancellation replay requires the original delivery root')
def test_mixed_actual_cancellation_after_material_contact_trial(monkeypatch):
    # This test's output dependency is prospectively bound in its own job;
    # it never invents a prefix from a terminal snapshot.
    root=Path(os.environ['FINITE_V2D_WORKFLOW'])
    prefix=json.loads((root/'prefix.json').read_text());fresh=workflow_fresh(root)
    model,definition,settings,program=build()
    # Cancel only after actual structural/material AND contact trial work.
    # No response values, laws, state coordinates or controls are mocked.
    from anyfesolver import CancellationToken,SolveCancelled
    from examples.general_contact_finite_v2d_pair import json_value
    import anyfesolver.general_contact.semismooth_path as increment
    from anyfesolver.general_contact.runtime import _ACTIVE_CONTACT
    from anyfesolver.nonlinear_restart import model_configuration_fingerprint
    token=CancellationToken();captured={};input_bytes=json.dumps(prefix,sort_keys=True)
    before_model=model_configuration_fingerprint(model);before_context=_ACTIVE_CONTACT.get()
    assemble=increment._assemble_nonlinear_system;contact_call=increment.linearize
    def state_bytes(states):return json.dumps(json_value(dict(states)),sort_keys=True,allow_nan=False)
    def observed_assembly(*args,**kwargs):
        committed=args[2];captured['store']=committed;captured['before']=state_bytes(committed)
        result=assemble(*args,**kwargs);captured['material_trial_returned']=result[2] is not None
        return result
    def cancel_after_contact(manager,*args,**kwargs):
        captured['manager']=manager;captured['normal_before']=manager.normal_multipliers.copy()
        result=contact_call(manager,*args,**kwargs);captured['actual_contact_trial']=True
        token.cancel('mixed material/contact trial rollback discriminator')
        return result
    with monkeypatch.context() as local:
        local.setattr(increment,'_assemble_nonlinear_system',observed_assembly)
        local.setattr(increment,'linearize',cancel_after_contact)
        with pytest.raises(SolveCancelled) as stopped:
            solve(model,definition,settings,program,restart_checkpoint=prefix['checkpoint'],cancellation_token=token)
    assert captured['material_trial_returned'] and captured['actual_contact_trial']
    assert captured['before']==state_bytes(captured['store'])
    np.testing.assert_array_equal(captured['manager'].normal_multipliers,captured['normal_before'])
    assert json.dumps(prefix,sort_keys=True)==input_bytes
    assert model_configuration_fingerprint(model)==before_model and _ACTIVE_CONTACT.get() is before_context
    assert increment._assemble_nonlinear_system is assemble and increment.linearize is contact_call
    emit('cancellation',dict(stage=stopped.value.stage,actual_material_contact_trial_before_cancel=True,committed_state_rollback=True,checkpoint_input_unchanged=True,context_and_hook_restored=True))


@pytest.mark.skipif(not os.environ.get('FINITE_V2D_WORKFLOW'),
    reason='archived workflow input FINITE_V2D_WORKFLOW prefix.json/fresh.json absent: restart-suffix replay requires the original delivery root')
def test_public_actual_restart_suffix_and_terminal_restore():
    root=Path(os.environ['FINITE_V2D_WORKFLOW'])
    prefix=json.loads((root/'prefix.json').read_text());fresh=workflow_fresh(root)
    model,definition,settings,program=build()
    require_saved_fixture(prefix,model,definition,settings,program)
    require_saved_fixture(fresh,model,definition,settings,program)
    assert prefix['checkpoint']['path_state']['accepted_step']==15
    resumed=solve(model,definition,settings,program,restart_checkpoint=prefix['checkpoint'])
    assert resumed.status=='completed',resumed.failure_reason
    assert len(resumed.snapshots)==1 and resumed.snapshots[0].step_index==16
    np.testing.assert_allclose(resumed.displacements,fresh['displacements'],rtol=1e-8,atol=1e-12)
    resumed_z=np.asarray(resumed.restart_checkpoint['path_state']['normal_forces']);fresh_z=np.asarray(fresh['checkpoint']['path_state']['normal_forces'])
    assert resumed_z.shape==fresh_z.shape==(4,)
    np.testing.assert_allclose(resumed_z,fresh_z,rtol=1e-7,atol=1e-12)
    def history_z(rows):
        assert len(rows)==16 and [r['step_index'] for r in rows]==list(range(1,17))
        contact=rows[-1]['contact'];assert contact['source_pressure_node_ids']==list(range(1,5))
        return np.asarray(contact['numerical_pressure_pa'])*np.asarray(contact['reference_support_m2'])
    np.testing.assert_allclose(history_z(resumed.info['force_displacement_history']),resumed_z,rtol=1e-7,atol=1e-12)
    np.testing.assert_allclose(history_z(fresh['history']),fresh_z,rtol=1e-7,atol=1e-12)
    assert resumed.info['element_state_coordinates']==fresh['diagnostics']['element_state_coordinates']==COORDINATES
    assert resumed.info['finite_state_owners']==fresh['diagnostics']['finite_state_owners']==owner_contract(model)
    def validate_owner_records(checkpoint):
        records=checkpoint['element_states'];assert [r['element_id'] for r in records]==[1,2,3,4]
        states={r['element_id']:r['state'] for r in records};q=np.asarray(checkpoint['displacements'])
        # The adapter calls each exact model-bound owner validator. No incoming seal is rewritten.
        prepare_v2d_states(model,q,states,5,{},restart=True)
        for eid,state in states.items():
            assert isinstance(state['state_integrity_sha256'],str) and len(state['state_integrity_sha256'])==64
            assert state['solver_kinematics']=='COROTATIONAL'
            dofs=model.mesh.elements[eid].get_dof_mapping(model.mesh)
            np.testing.assert_array_equal(state['committed_total_u'],q[dofs])
            assert np.asarray(state['committed_constitutive_u']).shape==(18,)
        return states
    resumed_states=validate_owner_records(resumed.restart_checkpoint);fresh_states=validate_owner_records(fresh['checkpoint'])
    assert sorted(resumed.snapshots[0].element_states)==[1,2,3,4]
    for eid in resumed_states:
        for name in ('plastic_strain','alpha'):
            np.testing.assert_array_equal(resumed_states[eid][name],fresh_states[eid][name])
            np.testing.assert_array_equal(resumed.snapshots[0].element_states[eid][name],resumed_states[eid][name])
    from examples.general_contact_finite_v2d_pair import report_result
    emit('actual_suffix',report_result(resumed,model,definition,settings,program))
    # Target uz is guided at its four vertices; the six non-anchor target
    # bending rotations are unknowns. Their actual elastic response must be
    # nonzero, rather than describing this target as freely translating uz.
    target_bending=[d for nid in (6,7,8) for d in model.mesh.dof_manager.get_node_dofs(nid)[3:5]]
    rigid=rigid_pose(model,math.pi/3.)
    response=float(np.max(np.abs(resumed.displacements[target_bending]-rigid[target_bending])))
    gamma=512*np.finfo(float).eps/(1-512*np.finfo(float).eps)
    uncertainty=gamma*max(1.,float(np.max(np.abs(resumed.displacements[target_bending]))))
    # Preserve the existing finite-Q4 free-bending witness threshold rather
    # than accepting an arbitrary nonzero difference at machine precision.
    assert response>max(1e-10,uncertainty)
    restored=solve(model,definition,settings,program,restart_checkpoint=resumed.restart_checkpoint)
    assert restored.status=='completed' and not restored.snapshots
    np.testing.assert_array_equal(restored.displacements,resumed.displacements)
    np.testing.assert_array_equal(restored.restart_checkpoint['path_state']['normal_forces'],resumed_z)
    validate_owner_records(restored.restart_checkpoint)
    emit('actual_terminal_restore',report_result(restored,model,definition,settings,program))
    emit('restart',dict(suffix_steps=[s.step_index for s in resumed.snapshots],terminal_new_steps=0,
                       target_free_bending_rotation_max_rad=response,numerical_roundoff_bound_rad=uncertainty,
                       inherited_free_bending_witness_rad=1e-10,typed_owners=owner_contract(model),
                       supported_force_max_difference_n=float(np.max(np.abs(resumed_z-fresh_z))),
                       owner_model_validation=True,elastic_constitutive_history_equal=True))


@pytest.mark.skipif(not os.environ.get('FINITE_V2D_WORKFLOW'),
    reason='archived workflow input FINITE_V2D_WORKFLOW fresh.json absent: saved-release replay requires the original delivery root')
def test_saved_release_strict_positive_physical_opening():
    root=Path(os.environ['FINITE_V2D_WORKFLOW']);saved=workflow_fresh(root)
    model,definition,settings,program=build();require_saved_fixture(saved,model,definition,settings,program)
    snapshot=saved['accepted_suffix'][14];assert snapshot['step_index']==15
    q=np.asarray(snapshot['displacements']);assert q.shape==(48,) and np.isfinite(q).all()
    witness=get_oracle().opening_bound(model,q)
    emit('strict_release_opening',dict(step=15,**witness))
    assert witness['gate_pass'], 'zero pressure and the 50um safety bound alone do not prove physical opening'


def test_nonaffine_triangle_sweep_refuses_hidden_director_crossing():
    from anyfesolver.general_contact.finite_rotation_sweep import finite_q4_path_admissible
    model,definition,settings,_=build();manager=GeneralS3PatchManager(model,definition,settings)
    alpha=.09;h=-.035;initial_gap=2e-7;end_gap=2e-5
    # Both endpoints are prescribed safe analytically; the bottom skin has its
    # deepest position at the unobserved theta=0 inside the additive chord.
    translation=end_gap-initial_gap-h*(math.cos(alpha)-1.)
    left=np.zeros(48);right=left.copy()
    for nid in range(1,5):
        d=model.mesh.dof_manager.get_node_dofs(nid);left[d[2]]=right[d[2]]=translation
        left[d[3]]=-alpha;right[d[3]]=alpha
    oracle=get_oracle();assert oracle.opening_bound(model,left)['gate_pass'] and oracle.opening_bound(model,right)['gate_pass']
    middle=(left+right)/2;mid=oracle.gap_bound(model,middle)
    assert mid['lower_m'] < -5e-5
    before=manager.normal_multipliers.copy()
    assert not finite_q4_path_admissible(manager,left,right)
    np.testing.assert_array_equal(before,manager.normal_multipliers)
    emit('hidden_director_crossing',dict(alpha_rad=alpha,end_gap_m=end_gap,midpoint_lower_m=mid['lower_m'],
        scope='geometry-only disposable chord; no changes to public controls or loads'))


def test_affected_mixed_profile_admission_unchanged():
    from examples.general_contact_finite_q4_v2d import build as old_build
    from anyfesolver.general_contact.finite_mixed_state import STATIC_PROFILE as old_profile
    model,definition,settings,_=old_build();manager=GeneralS3PatchManager(model,definition,settings)
    assert manager.profile==old_profile


@pytest.mark.skipif(not os.environ.get('FINITE_V2D_LEGACY_EVIDENCE'),
    reason='archived legacy evidence input FINITE_V2D_LEGACY_EVIDENCE absent: installed legacy suffix replay requires the original evidence root')
def test_affected_q4_v2d_actual_installed_suffix():
    from examples.general_contact_finite_q4_v2d import build as old_build,solve as old_solve,report_result
    from anyfesolver.general_contact.finite_mixed_state import COORDINATES as old_coordinates,prepare_mixed_states,owner_contract as old_owners
    root=Path(os.environ['FINITE_V2D_LEGACY_EVIDENCE'])
    prefix=json.loads((root/'prefix.json').read_text());fresh=json.loads((root/'fresh.json').read_text())
    assert prefix['status']==fresh['status']=='completed'
    assert prefix['checkpoint']['path_state']['accepted_step']==15
    model,definition,settings,program=old_build()
    result=old_solve(model,definition,settings,program,restart_checkpoint=prefix['checkpoint'])
    assert result.status=='completed',result.failure_reason
    assert [s.step_index for s in result.snapshots]==[16]
    assert result.info['element_state_coordinates']==old_coordinates
    assert result.info['finite_state_owners']==old_owners(model)
    np.testing.assert_allclose(result.displacements,fresh['displacements'],rtol=1e-8,atol=1e-12)
    z=result.restart_checkpoint['path_state']['normal_forces']
    np.testing.assert_allclose(z,fresh['checkpoint']['path_state']['normal_forces'],rtol=1e-7,atol=1e-12)
    def validate(checkpoint):
        states={v['element_id']:v['state'] for v in checkpoint['element_states']}
        assert sorted(states)==[1,2,3,4]
        return prepare_mixed_states(model,np.asarray(checkpoint['displacements']),states,5,{},restart=True)
    current_states=validate(result.restart_checkpoint);fresh_states=validate(fresh['checkpoint'])
    for eid in current_states:
        for name in ('plastic_strain','alpha'):
            np.testing.assert_array_equal(current_states[eid][name],fresh_states[eid][name])
    emit('legacy_q4_v2d_actual_suffix',report_result(result,model,definition,settings,program))
    terminal=old_solve(model,definition,settings,program,restart_checkpoint=result.restart_checkpoint)
    assert terminal.status=='completed' and not terminal.snapshots
    np.testing.assert_array_equal(terminal.displacements,result.displacements)
    np.testing.assert_array_equal(terminal.restart_checkpoint['path_state']['normal_forces'],z)
    validate(terminal.restart_checkpoint)
    emit('legacy_q4_v2d_terminal_restore',report_result(terminal,model,definition,settings,program))
    emit('legacy_q4_v2d_restart',dict(real_suffix_steps=[16],terminal_new_steps=0,
         element_state_coordinates=old_coordinates,typed_owner_validation=True,
         scope='affected Q4/V2D solve/seal/restart on new wheel; prior full trajectory reused by impact'))


def test_actual_converged_caller_encloses_candidate_skin(monkeypatch):
    import inspect
    model,definition,settings,program=build();observed=[]
    original=GeneralS3PatchManager.point_gap_lower
    gap=GeneralS3PatchManager._gap_lower
    def point(manager,displacement):
        frame=inspect.currentframe().f_back
        observed.append((Path(frame.f_code.co_filename).name,frame.f_code.co_name))
        return original(manager,displacement)
    def enclosed(manager,positions,*,endpoint_errors=None):
        if manager.profile==STATIC_PROFILE:
            assert endpoint_errors is not None
            assert all(np.isfinite(v).all() and np.all(v>=0) for v in endpoint_errors)
        return gap(manager,positions,endpoint_errors=endpoint_errors)
    with monkeypatch.context() as local:
        local.setattr(GeneralS3PatchManager,'point_gap_lower',point)
        local.setattr(GeneralS3PatchManager,'_gap_lower',enclosed)
        result=solve(model,definition,settings,program,max_load_factor=1/16.)
    assert result.status=='completed',result.failure_reason
    assert [s.step_index for s in result.snapshots]==[1]
    assert any(file=='semismooth_path.py' and name!='_proposal_gap_diagnostic' for file,name in observed)
    assert GeneralS3PatchManager.point_gap_lower is original and GeneralS3PatchManager._gap_lower is gap
    emit('converged_point_gap',dict(actual_accepted_steps=[1],caller_frames=observed,
         explicit_finite_skin_endpoint_errors=True,hooks_restored=True))


@pytest.mark.parametrize('lower',(-float('inf'),float('inf'),float('nan'),1.25e-5))
def test_unresolved_gap_diagnostic_retains_strict_json_refusal(lower):
    from types import SimpleNamespace
    from anyfesolver.general_contact.semismooth_path import _proposal_gap_diagnostic
    manager=SimpleNamespace(point_gap_lower=lambda displacement:lower)
    diagnostic=_proposal_gap_diagnostic(manager,np.zeros(48))
    payload=json.loads(json.dumps(dict(status='NONPASS',contact_failure=diagnostic),allow_nan=False))
    assert payload['status']=='NONPASS'
    if np.isfinite(lower):
        assert diagnostic==dict(swept_proposed_gap_lower_m=lower,swept_proposed_geometry_error=None)
    else:
        assert diagnostic['swept_proposed_gap_lower_m'] is None
        assert diagnostic['swept_proposed_geometry_error']=='nonfinite or unresolved physical gap bound'
