"""Fixed automatic finite selection, actual caller and independent saved-field checks."""
import copy,hashlib,json,math,os
from dataclasses import asdict,replace
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import numpy as np
import pytest
import examples.general_contact_finite_automatic_shell as public
from examples.general_contact_finite_automatic_shell import build,solve,rigid_pose,report_result
from anyfesolver.general_contact.automatic_domain import (AutomaticContactResolution,AutomaticContactDomain,
    AutomaticContactDomainError,resolve_automatic_contact_domain,verify_automatic_contact_resolution)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize,complementarity,reference_nodal_support
from anyfesolver.general_contact.finite_v2d_state import COORDINATES,owner_contract,prepare_v2d_states
FREF=MREF=1e-4

def canonical(value):return json.loads(json.dumps(value))
def get_oracle():
    spec=spec_from_file_location('finite_automatic_independent',Path(__file__).with_name('general_contact_finite_automatic_shell_oracle.py'))
    module=module_from_spec(spec);spec.loader.exec_module(module);return module

def emit(name,value):
    if os.environ.get('FINITE_AUTOMATIC_DIAGNOSTICS'):
        root=Path(os.environ['FINITE_AUTOMATIC_DIAGNOSTICS']);root.mkdir(parents=True,exist_ok=True)
        (root/(name+'.json')).write_text(json.dumps(value,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else v,allow_nan=False))

def workflow_fresh(root):return json.loads((root/'fresh.json').read_text())

def require_saved_fixture(saved,model,domain,settings,resolution,program):
    from anyfesolver.nonlinear_restart import model_configuration_fingerprint
    public.require_report_identity(saved,model,domain,settings,resolution,program)
    assert saved['status']=='completed'
    assert saved['checkpoint']['model_fingerprint']==model_configuration_fingerprint(model)
    assert np.array_equal(saved['displacements'],saved['accepted_suffix'][-1]['displacements'])
    assert saved['diagnostics']['automatic_resolution_identity']==resolution.identity
    assert saved['fixture']['source_pressure_node_ids']==[5,6,7,8]

def reseal(payload):
    payload['identity']=hashlib.sha256(json.dumps(payload['contract'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return payload

def test_resolution_exact_roundtrip_and_original_physical_rows():
    model,domain,settings,resolution,program=build()
    restored=AutomaticContactResolution.from_dict(canonical(resolution.to_dict()))
    assert canonical(restored.to_dict())==canonical(resolution.to_dict())
    assert verify_automatic_contact_resolution(model,domain,restored,settings)==resolution.definition
    manager=GeneralS3PatchManager(model,resolution.definition,settings)
    assert manager.law.pressure_mode_ids==(5,6,7,8)
    assert len(model.mesh.nodes)==8 and model.mesh.dof_manager.total_dofs==48 and len(model.mesh.elements)==4
    assert public.fixture(model,domain,settings,resolution,program)['source_elements']==[3,4]
    assert public.fixture(model,domain,settings,resolution,program)['target_elements']==[1,2]
    assert set(f.node_id for f in program.nodal_forces)=={2,4}
    from examples.general_contact_finite_v2d_pair import build as previous
    original,_,oldsettings,oldprogram=previous()
    assert program.contract()==oldprogram.contract() and settings==oldsettings
    for eid,e in model.mesh.elements.items():
        assert e.node_ids==original.mesh.elements[eid].node_ids
        np.testing.assert_array_equal(e.reference_normal,original.mesh.elements[eid].reference_normal)
    emit('selection',dict(resolution=resolution.to_dict(),physical_program_unchanged=True,pressure_node_ids=list(manager.law.pressure_mode_ids)))

@pytest.mark.parametrize('kind',('identity','unknown','redundant_definition','redundant_selection','policy','dropped_pair','reversed_role','face_unknown','bool_node'))
def test_resolution_refuses_tampering_even_resealed(kind):
    *_,resolution,_=build();value=canonical(resolution.to_dict())
    if kind=='identity':value['identity']='0'*64
    elif kind=='unknown':value['unknown']=1
    elif kind=='redundant_definition':value['definition']['interactions'][0]['gap_tolerance']=1e-3
    elif kind=='redundant_selection':value['selections'][0]['reason']='forged'
    elif kind=='policy':value['contract']['policy']='unknown';reseal(value)
    elif kind=='dropped_pair':value['contract']['selections']=[];value['selections']=[];value['definition']['interactions']=[];reseal(value)
    elif kind=='reversed_role':
        row=value['contract']['selections'][0];row['source'],row['target']=row['target'],row['source'];reseal(value)
    elif kind=='face_unknown':value['contract']['reference'][0]['faces'][0]['unknown']=True;reseal(value)
    else:value['contract']['reference'][0]['faces'][0]['node_ids'][0]=True;reseal(value)
    with pytest.raises(AutomaticContactDomainError):AutomaticContactResolution.from_dict(value)

def test_existing_nonpositive_node_labels_refused_by_patch_admission():
    from anyfesolver import FEModel,create_shell_element
    original,domain,settings,_,_=build();model=FEModel('integer-label-copy')
    model.add_material('elastic',1e5,.3,density=1.)
    labels={nid:nid-5 for nid in original.mesh.nodes}
    for nid,n in original.mesh.nodes.items():model.add_node(labels[nid],n.x,n.y,n.z)
    for eid,e in original.mesh.elements.items():
        model.add_element(eid,create_shell_element(eid,tuple(labels[n] for n in e.node_ids),'elastic',thickness=.05,
            reference_surface_offset=.01,reference_normal=(0.,0.,1.),formulation='e4-pl-s3-v2d'))
    with pytest.raises(AutomaticContactDomainError,match='distinct material nodes') as stopped:
        resolve_automatic_contact_domain(model,domain,settings,analysis_control='contact_path')
    assert stopped.value.code=='unsupported_execution'
    # Mesh/structural labels permit this model, but the complete contact route
    # does not. Preserve the patch admission rather than broaden its owner.
    positive,positive_domain,positive_settings,resolution,_=build()
    restored=AutomaticContactResolution.from_dict(canonical(resolution.to_dict()))
    verify_automatic_contact_resolution(positive,positive_domain,restored,positive_settings)
    value=canonical(resolution.to_dict())
    value['contract']['reference'][0]['faces'][0]['node_ids'][0]=0;reseal(value)
    with pytest.raises(AutomaticContactDomainError):AutomaticContactResolution.from_dict(value)

def test_restore_does_not_resolve_or_replace_selection(monkeypatch):
    model,domain,settings,resolution,program=build()
    def forbidden(*a,**k):raise AssertionError('restoration called resolver')
    monkeypatch.setattr(public,'resolve_automatic_contact_domain',forbidden)
    again=build(saved_domain=domain.to_dict(),saved_resolution=canonical(resolution.to_dict()))
    assert again[3].identity==resolution.identity and again[4].contract()==program.contract()

@pytest.mark.parametrize('change',('model','settings','domain'))
def test_model_bound_guard_refuses_changes(change,monkeypatch):
    model,domain,settings,resolution,program=build()
    if change=='model':model.mesh.nodes[2].x+=.001
    elif change=='settings':settings=replace(settings,pressure_tolerance=1e-7)
    else:domain=replace(domain,name='changed')
    def forbidden(*a,**k):raise AssertionError('structural solve reached before identity guard')
    monkeypatch.setattr(public,'paired_solve',forbidden)
    with pytest.raises(AutomaticContactDomainError):solve(model,domain,settings,resolution,program)

@pytest.mark.parametrize('kind',('bare_missing_selection','rich_missing_checkpoint','rich_missing_selection'))
def test_public_restart_refuses_missing_identity_before_solve(kind,tmp_path,monkeypatch):
    value={'schema':'ANYSOLVER_NONLINEAR_CHECKPOINT_V2'}
    if kind.startswith('rich'):
        model,domain,settings,resolution,program=build()
        value=dict(schema=public.SCHEMA,automatic_domain=domain.to_dict(),automatic_resolution=canonical(resolution.to_dict()),checkpoint=None)
        if kind=='rich_missing_selection':value.pop('automatic_resolution')
    path=tmp_path/'input.json';path.write_text(json.dumps(value));output=tmp_path/'output.json'
    def forbidden(*a,**k):raise AssertionError('solve launched')
    monkeypatch.setattr(public,'solve',forbidden)
    assert public.main(['--restart',str(path),'--output',str(output)])==1
    saved=json.loads(output.read_text());assert saved['status']=='NONPASS' and 'AssertionError' not in saved['failure_reason']

def test_explicit_same_role_contact_operator_parity():
    model,domain,settings,resolution,program=build()
    automatic=GeneralS3PatchManager(model,resolution.definition,settings)
    pair=resolution.definition.interactions[0]
    explicit=replace(resolution.definition,interactions=(replace(pair,name='explicit-same-lower-pressure-role'),))
    other=GeneralS3PatchManager(model,explicit,settings)
    q=np.zeros(model.mesh.dof_manager.total_dofs);z=np.full(4,1e-5)
    a=linearize(automatic,q,z,tangent=True);b=linearize(other,q,z,tangent=True)
    assert a.gradient_only_pairs==b.gradient_only_pairs==0
    for name in ('force','gap_gradient','force_gradient','force_pressure_gradient'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    np.testing.assert_array_equal(a.tangent.toarray(),b.tangent.toarray())
    np.testing.assert_array_equal(a.state.reference_support,b.state.reference_support)
    assert automatic.law.pressure_mode_ids==other.law.pressure_mode_ids==(5,6,7,8)

def test_explicit_precedence_keeps_properties_and_fixed_orientation():
    from examples.general_contact_finite_v2d_pair import build as paired
    model,definition,settings,_=paired()
    original=definition.interactions[0]
    domain=replace(public.declared_domain(definition),pair_overrides=(original,))
    resolution=resolve_automatic_contact_domain(model,domain,settings,analysis_control='contact_path')
    assert resolution.selections[0].source==original.surface
    restored=AutomaticContactResolution.from_dict(canonical(resolution.to_dict()))
    verify_automatic_contact_resolution(model,domain,restored,settings)
    actual=restored.definition.interactions[0]
    assert replace(actual,name=original.name)==original
    changed=canonical(resolution.to_dict());changed['contract']['explicit_pairs']=[];reseal(changed)
    with pytest.raises(AutomaticContactDomainError):AutomaticContactResolution.from_dict(changed)



@pytest.mark.skipif(not os.environ.get('FINITE_AUTOMATIC_WORKFLOW'),
    reason='archived workflow input FINITE_AUTOMATIC_WORKFLOW absent: saved-fixture replay requires the original delivery root')
def test_saved_complete_workflow_independent_skin_gap():
    root=Path(os.environ['FINITE_AUTOMATIC_WORKFLOW']);saved=workflow_fresh(root)
    model,domain,settings,resolution,program=build();definition=resolution.definition
    require_saved_fixture(saved,model,domain,settings,resolution,program)
    assert saved['status']=='completed' and len(saved['accepted_suffix'])==16
    assert [s['step_index'] for s in saved['accepted_suffix']]==list(range(1,17))
    lower=[]
    for snapshot in saved['accepted_suffix']:
        q=np.asarray(snapshot['displacements']);assert q.shape==(48,) and np.isfinite(q).all()
        assert snapshot['element_state_ids']==[1,2,3,4]
        lower.append(get_oracle().gap_bound(model,q,definition))
    emit('physical_gap',dict(scope='independent conservative P1/P1 whole-face lower on all returned poses',
                            lower_m=lower,gate_m=-5e-5))
    assert all(v['gate_pass'] for v in lower)

@pytest.mark.skipif(not os.environ.get('FINITE_AUTOMATIC_WORKFLOW'),
    reason='archived workflow input FINITE_AUTOMATIC_WORKFLOW absent: saved-fixture replay requires the original delivery root')
def test_saved_actual_scaled_equilibrium_wrench_ncp_and_integration():
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system,_discard_nonlinear_state_candidate
    oracle=get_oracle()
    root=Path(os.environ['FINITE_AUTOMATIC_WORKFLOW']);saved=workflow_fresh(root)
    model,domain,settings,resolution,program=build();definition=resolution.definition;model.apply_boundary_conditions()
    require_saved_fixture(saved,model,domain,settings,resolution,program)
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
        assert contact['source_pressure_node_ids']==[5,6,7,8]
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
        independent,_=oracle.vertices(model,q,definition)
        integration=[]
        for order in (4,8):
            base=oracle.integral(model,independent,z,order,definition);measurements=[]
            assert not base['ties'],'independent integration branch tie unresolved'
            for h in (1e-5,3e-6):
                derivatives=[]
                for direction in range(6):
                    delta={nid:np.zeros(3) for nid in independent}
                    for nid in range(5,9):
                        delta[nid]=(np.eye(3)[direction] if direction<3
                                    else np.cross(np.eye(3)[direction-3],independent[nid]))
                    plus=oracle.integral(model,{nid:p+h*delta[nid] for nid,p in independent.items()},z,order,definition)
                    minus=oracle.integral(model,{nid:p-h*delta[nid] for nid,p in independent.items()},z,order,definition)
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

@pytest.mark.skipif(not os.environ.get('FINITE_AUTOMATIC_WORKFLOW'),
    reason='archived workflow input FINITE_AUTOMATIC_WORKFLOW prefix.json absent: cancellation replay requires the original delivery root')
def test_automatic_actual_cancellation_after_material_contact_trial(monkeypatch):
    # This test's output dependency is prospectively bound in its own job;
    # it never invents a prefix from a terminal snapshot.
    root=Path(os.environ['FINITE_AUTOMATIC_WORKFLOW'])
    prefix=json.loads((root/'prefix.json').read_text());fresh=workflow_fresh(root)
    model,domain,settings,resolution,program=build(saved_domain=prefix['automatic_domain'],saved_resolution=prefix['automatic_resolution']);definition=resolution.definition
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
            solve(model,domain,settings,resolution,program,restart_checkpoint=prefix['checkpoint'],cancellation_token=token)
    assert captured['material_trial_returned'] and captured['actual_contact_trial']
    assert captured['before']==state_bytes(captured['store'])
    np.testing.assert_array_equal(captured['manager'].normal_multipliers,captured['normal_before'])
    assert json.dumps(prefix,sort_keys=True)==input_bytes
    assert model_configuration_fingerprint(model)==before_model and _ACTIVE_CONTACT.get() is before_context
    assert increment._assemble_nonlinear_system is assemble and increment.linearize is contact_call
    emit('cancellation',dict(stage=stopped.value.stage,actual_material_contact_trial_before_cancel=True,committed_state_rollback=True,checkpoint_input_unchanged=True,context_and_hook_restored=True))

@pytest.mark.skipif(not os.environ.get('FINITE_AUTOMATIC_WORKFLOW') or not os.environ.get('FINITE_AUTOMATIC_DIAGNOSTICS'),
    reason='archived workflow inputs FINITE_AUTOMATIC_WORKFLOW prefix.json/fresh.json and FINITE_AUTOMATIC_DIAGNOSTICS absent: restart-suffix replay requires the original delivery root')
def test_public_actual_restart_suffix_and_terminal_restore(monkeypatch):
    root=Path(os.environ['FINITE_AUTOMATIC_WORKFLOW']);prefix=json.loads((root/'prefix.json').read_text());fresh=workflow_fresh(root)
    model,domain,settings,resolution,program=build(saved_domain=prefix['automatic_domain'],saved_resolution=prefix['automatic_resolution'])
    require_saved_fixture(prefix,model,domain,settings,resolution,program);require_saved_fixture(fresh,model,domain,settings,resolution,program)
    assert prefix['checkpoint']['path_state']['accepted_step']==15
    before=json.dumps(prefix,sort_keys=True,allow_nan=False);identity=resolution.identity
    details=Path(os.environ['FINITE_AUTOMATIC_DIAGNOSTICS']);details.mkdir(parents=True,exist_ok=True)
    def forbidden(*a,**k):raise AssertionError('saved selection was re-resolved')
    monkeypatch.setattr(public,'resolve_automatic_contact_domain',forbidden)
    suffix=details/'actual_suffix.json'
    assert public.main(['--restart',str(root/'prefix.json'),'--through-step','16','--output',str(suffix)])==0
    resumed=json.loads(suffix.read_text())
    public.require_report_identity(resumed,model,domain,settings,resolution,program)
    assert resumed['status']=='completed' and [r['step_index'] for r in resumed['accepted_suffix']]==[16]
    assert resumed['automatic_resolution']['identity']==identity
    np.testing.assert_allclose(resumed['displacements'],fresh['displacements'],rtol=1e-8,atol=1e-12)
    resumed_z=np.asarray(resumed['checkpoint']['path_state']['normal_forces']);fresh_z=np.asarray(fresh['checkpoint']['path_state']['normal_forces'])
    assert resumed_z.shape==fresh_z.shape==(4,)
    np.testing.assert_allclose(resumed_z,fresh_z,rtol=1e-7,atol=1e-12)
    def history_z(rows):
        assert len(rows)==16 and [r['step_index'] for r in rows]==list(range(1,17))
        contact=rows[-1]['contact'];assert contact['source_pressure_node_ids']==[5,6,7,8]
        return np.asarray(contact['numerical_pressure_pa'])*np.asarray(contact['reference_support_m2'])
    np.testing.assert_allclose(history_z(resumed['history']),resumed_z,rtol=1e-7,atol=1e-12)
    np.testing.assert_allclose(history_z(fresh['history']),fresh_z,rtol=1e-7,atol=1e-12)
    assert resumed['diagnostics']['element_state_coordinates']==fresh['diagnostics']['element_state_coordinates']==COORDINATES
    assert canonical(resumed['diagnostics']['finite_state_owners'])==canonical(owner_contract(model))
    def owner_records(checkpoint):
        records=checkpoint['element_states'];assert [r['element_id'] for r in records]==[1,2,3,4]
        states={r['element_id']:r['state'] for r in records};q=np.asarray(checkpoint['displacements'])
        prepare_v2d_states(model,q,states,5,{},restart=True)
        for eid,state in states.items():
            assert len(state['state_integrity_sha256'])==64 and state['solver_kinematics']=='COROTATIONAL'
            np.testing.assert_array_equal(state['committed_total_u'],q[model.mesh.elements[eid].get_dof_mapping(model.mesh)])
            assert np.asarray(state['committed_constitutive_u']).shape==(18,)
        return states
    a=owner_records(resumed['checkpoint']);b=owner_records(fresh['checkpoint'])
    assert sorted(map(int,resumed['accepted_suffix'][0]['element_states']))==[1,2,3,4]
    for eid in a:
        for name in ('plastic_strain','alpha'):
            np.testing.assert_array_equal(a[eid][name],b[eid][name])
            np.testing.assert_array_equal(resumed['accepted_suffix'][0]['element_states'][str(eid)][name],a[eid][name])
    free_bending=[d for nid in (6,7,8) for d in model.mesh.dof_manager.get_node_dofs(nid)[3:5]]
    q=np.asarray(resumed['displacements']);rigid=rigid_pose(model,math.pi/3.)
    response=float(np.max(np.abs(q[free_bending]-rigid[free_bending])))
    gamma=512*np.finfo(float).eps/(1-512*np.finfo(float).eps)
    uncertainty=gamma*max(1.,float(np.max(np.abs(q[free_bending]))));assert response>max(1e-10,uncertainty)
    terminal=details/'actual_terminal_restore.json'
    assert public.main(['--restart',str(suffix),'--through-step','16','--output',str(terminal)])==0
    restored=json.loads(terminal.read_text());assert restored['status']=='completed' and restored['accepted_suffix']==[]
    np.testing.assert_array_equal(restored['displacements'],resumed['displacements'])
    np.testing.assert_array_equal(restored['checkpoint']['path_state']['normal_forces'],resumed_z)
    owner_records(restored['checkpoint']);assert restored['automatic_resolution']['identity']==identity
    bare=details/'bare_checkpoint.json';domain_file=details/'domain.json';selection_file=details/'selection.json'
    bare.write_text(json.dumps(resumed['checkpoint'],allow_nan=False))
    domain_file.write_text(json.dumps(resumed['automatic_domain'],allow_nan=False))
    selection_file.write_text(json.dumps(resumed['automatic_resolution'],allow_nan=False))
    bare_output=details/'actual_bare_terminal_restore.json'
    assert public.main(['--restart',str(bare),'--domain',str(domain_file),'--selection',str(selection_file),
        '--through-step','16','--output',str(bare_output)])==0
    bare_restored=json.loads(bare_output.read_text())
    assert bare_restored['status']=='completed' and bare_restored['accepted_suffix']==[]
    np.testing.assert_array_equal(bare_restored['displacements'],resumed['displacements'])
    np.testing.assert_array_equal(bare_restored['checkpoint']['path_state']['normal_forces'],resumed_z)
    owner_records(bare_restored['checkpoint'])
    assert before==json.dumps(json.loads((root/'prefix.json').read_text()),sort_keys=True,allow_nan=False)
    emit('restart',dict(suffix_steps=[16],terminal_new_steps=0,bare_terminal_new_steps=0,fixed_selection_identity=identity,resolver_not_called=True,
        lower_physical_body_free_bending_rotation_max_rad=response,numerical_roundoff_bound_rad=uncertainty,
        inherited_free_bending_witness_rad=1e-10,typed_owners=owner_contract(model),
        supported_force_max_difference_n=float(np.max(np.abs(resumed_z-fresh_z))),owner_model_validation=True,
        elastic_constitutive_history_equal=True,prefix_input_unchanged=True))


@pytest.mark.skipif(not os.environ.get('FINITE_AUTOMATIC_WORKFLOW'),
    reason='archived workflow input FINITE_AUTOMATIC_WORKFLOW fresh.json absent: saved-release replay requires the original delivery root')
def test_saved_release_strict_positive_physical_opening():
    root=Path(os.environ['FINITE_AUTOMATIC_WORKFLOW']);saved=workflow_fresh(root)
    model,domain,settings,resolution,program=build()
    require_saved_fixture(saved,model,domain,settings,resolution,program)
    snapshot=saved['accepted_suffix'][14];assert snapshot['step_index']==15
    q=np.asarray(snapshot['displacements']);assert q.shape==(48,) and np.isfinite(q).all()
    row=saved['history'][14];assert row['step_index']==15
    contact=row['contact'];assert contact['source_pressure_node_ids']==[5,6,7,8]
    support=np.asarray(contact['reference_support_m2']);pressure=np.asarray(contact['numerical_pressure_pa'])
    assert support.shape==pressure.shape==(4,) and np.all(support>0) and np.all(pressure>=0)
    assert np.isfinite(np.r_[support,pressure]).all() and np.linalg.norm(support*pressure,1)/FREF<=1e-8
    oracle=get_oracle()
    extended=oracle.opening_bound(model,q,resolution.definition)
    emit('extended_target_plane_opening_diagnostic',dict(step=15,claim='sufficient extension only; not the opening gate',**extended))
    witness=oracle.source_plane_opening_bound(model,q,resolution.definition)
    emit('strict_release_opening_source_plane',dict(step=15,**witness))
    assert witness['gate_pass'] and witness['all_pairs']==4 and witness['minimum_distance_lower_m']>0

@pytest.mark.skipif(not os.environ.get('FINITE_AUTOMATIC_PAIRED_PREFIX') or not Path(os.environ['FINITE_AUTOMATIC_PAIRED_PREFIX']).is_file(),
    reason='archived paired prefix input FINITE_AUTOMATIC_PAIRED_PREFIX absent: paired-checkpoint refusal replay requires the original saved prefix')
def test_paired_checkpoint_refuses_different_fixed_automatic_selection(tmp_path):
    # Same physical builder, but a different pressure side and interaction ID.
    # This is a refusal query, never a replay/migration of the paired trajectory.
    original=Path(os.environ['FINITE_AUTOMATIC_PAIRED_PREFIX'])
    paired=json.loads(original.read_text());before=original.read_bytes()
    model,domain,settings,resolution,program=build()
    from anyfesolver.nonlinear_restart import model_configuration_fingerprint
    assert paired['checkpoint']['model_fingerprint']==model_configuration_fingerprint(model)
    bare=tmp_path/'paired-checkpoint.json';domain_file=tmp_path/'domain.json';selection_file=tmp_path/'selection.json'
    bare.write_text(json.dumps(paired['checkpoint'],allow_nan=False))
    domain_file.write_text(json.dumps(domain.to_dict(),allow_nan=False))
    selection_file.write_text(json.dumps(canonical(resolution.to_dict()),allow_nan=False))
    output=tmp_path/'refused.json'
    assert public.main(['--restart',str(bare),'--domain',str(domain_file),'--selection',str(selection_file),
        '--through-step','16','--output',str(output)])==1
    refusal=json.loads(output.read_text());assert refusal['status']=='NONPASS'
    assert 'analysis contract' in refusal['failure_reason']
    assert not refusal.get('accepted_suffix') and not refusal.get('returned_accepted_suffix')
    assert original.read_bytes()==before
    emit('different_selection_refusal',dict(same_physical_model=True,old_pressure_nodes=[1,2,3,4],
        current_pressure_nodes=[5,6,7,8],failure_reason=refusal['failure_reason'],accepted_new_steps=0,
        prior_checkpoint_bytes_unchanged=True,no_migration_or_resealing=True))

