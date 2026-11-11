"""UNEXECUTED source-prepared discriminators for the global native/S3 route."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver._ge_beam3_connected_dynamic import _RetainedMidpointAssembly
from anyfesolver._ge_beam3_native_multi_shell_contact import AggregateWorkBudget,PROFILE,RESOURCE,TRANSITION,BOUNDARY,PATH_FAILURE
from anyfesolver._ge_beam3_native_transient import NativeTransientError
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver.general_contact.native_transient_multi_shell import NativeTransientS3ContactPair,S3ContactBody,solve_native_transient_multi_shell
from examples.general_contact_native_multi_shell_impact_dev import build,config,run,selections,FORCE_REFERENCE_N,MOMENT_REFERENCE_NM


def _initial(owner):
    cfg=config(owner)
    with owner._operation():return owner._initialize_state(cfg.initial_velocity,None,cfg.initial_displacement)


def test_global_shared_node_structural_assembly_and_actual_mass_once():
    owner=build(separation=.2);state=_initial(owner)
    assert owner.size==210 and owner.native_size==168 and owner.shell_size==42
    assert len(owner.physical)==81 and len(owner.algebraic)==129
    assert owner.shell_layout[0][1]==24 and len(owner.shell_bodies[0].model.mesh.elements)==2
    assert owner.shell_topologies[0]['internal_edges']==((4,7),)
    r,j,fi,m,dv,dq,account=owner._evaluate(state)
    bare=_RetainedMidpointAssembly._evaluate(owner,state)
    np.testing.assert_array_equal(m[:168,:168],bare[3][:168,:168])
    np.testing.assert_array_equal(m[168:,168:],owner.shell_mass)
    np.testing.assert_array_equal(m[list(owner.algebraic)],np.zeros((129,210)))
    for body,(o,size,_,_) in zip(owner.shell_bodies,owner.shell_layout):
        expected_k=np.zeros((size,size));expected_m=np.zeros_like(expected_k)
        for element in body.model.mesh.elements.values():
            slots=element.get_dof_mapping(body.model.mesh);material=body.model.materials[element.material_name]
            expected_k[np.ix_(slots,slots)]+=element.compute_stiffness_matrix(body.model.mesh,material)
            expected_m[np.ix_(slots,slots)]+=element.compute_mass_matrix(body.model.mesh,material)
        np.testing.assert_array_equal(owner.shell_stiffness[o:o+size,o:o+size],expected_k)
        np.testing.assert_array_equal(owner.shell_mass[o:o+size,o:o+size],expected_m)
    assert account['contact_regularization_storage']==0.
    np.testing.assert_array_equal(j[168:,168:],owner.shell_stiffness)
    np.testing.assert_allclose(r[168:],owner.shell_stiffness@state.shell_displacement,atol=1e-12)
    assert set(owner.contributors[0].mapping[42:])==set(owner.contributors[1].mapping[42:])
    assert set(owner.contributors[1].mapping[42:])==set(owner.contributors[2].mapping[42:])
    native,shell,pairs=selections(.2)
    subset=build(selections_override=(native,shell,tuple(p for p in pairs if p.native_element_id!=3)))
    subset_state=_initial(subset)
    np.testing.assert_array_equal(subset.shell_stiffness,owner.shell_stiffness)
    np.testing.assert_array_equal(subset.shell_mass,owner.shell_mass)
    assert len(subset.contributors)==3
    assert subset.contributors[0].mapping==owner.contributors[0].mapping
    observed=[];original=subset._unselected_orientation_leg
    def observe(*args):observed.append(args[0].element_id);return original(*args)
    subset._unselected_orientation_leg=observe
    path=subset._swept(subset_state,subset_state,subset_state,None)
    subset._unselected_orientation_leg=original
    assert observed==[22,22] and path['checked_intervals']>=8
    from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
    adapter=S3SkinWorkAdapter.from_model(shell[0].model,7,22,'top')
    skin=adapter.current_positions(subset_state.shell_displacement[:24]);budget=AggregateWorkBudget(1,1,1)
    original(adapter,skin,skin,0,budget,None)
    with pytest.raises(MemoryError,match=RESOURCE):original(adapter,skin,skin,0,budget,None)
    assert budget.used['swept']==1


def test_full210_common_stage_derivative_work_shared_skin_rotation_and_crossblocks():
    owner=build();accepted=_initial(owner);u=np.zeros(owner.size);dt=.02
    stage,r,tangent,account=owner._stage(accepted,u,dt,None)
    assert account['contact_regularization_storage']>0.
    chart=owner._midpoint_chart(np.zeros(owner.size),u,dt)
    direction=np.sin(np.arange(owner.size)+.3);direction[list(owner.p)]*=100.
    global_r=np.zeros(owner.size);global_j=np.zeros((owner.size,owner.size))
    for c,work in zip(owner.contributors,owner._contacts(stage)):
        idx=list(c.mapping);global_r[idx]+=work.residual;global_j[np.ix_(idx,idx)]+=work.tangent.toarray()
        assert np.linalg.norm(work.tangent[:42,42:].toarray())>0
        assert np.linalg.norm(work.tangent[42:,:42].toarray())>0
        _,_,_,z=owner.shell_layout[c.shell_index]
        assert np.linalg.norm(work.tangent[:,[42+i for i in z]].toarray())>0
    for epsilon in (1e-6,1e-7):
        plus=owner._stage(accepted,u+epsilon*direction,dt,None)
        minus=owner._stage(accepted,u-epsilon*direction,dt,None)
        fd=(plus[1]-minus[1])/(2*epsilon)
        assert np.linalg.norm((fd-tangent@direction)/owner.scale)<=1e-5*max(1.,np.linalg.norm(fd/owner.scale))
        def contact_vector(s):
            out=np.zeros(owner.size)
            for c,w in zip(owner.contributors,owner._contacts(s)):out[list(c.mapping)]+=w.residual
            return out
        cfd=(contact_vector(plus[0])-contact_vector(minus[0]))/(2*epsilon)
        assert np.linalg.norm((cfd-global_j@chart@direction)/owner.scale)<=1e-5*max(owner.force_tolerance_n,np.linalg.norm(cfd/owner.scale))
        efd=(plus[3]['contact_regularization_storage']-minus[3]['contact_regularization_storage'])/(2*epsilon)
        work=float(global_r@chart@direction)
        assert abs(efd-work)<=1e-5*max(owner.energy_tolerance_j,abs(work))
    assert not np.any(stage.velocity[[owner.native_size+i for i in owner.shell_z]])


def test_actual_error_map_row_column_bounds_zero_curvature_and_aggregate_gates():
    owner=build();state=_initial(owner)
    for c in owner.contributors:
        args=owner._arguments(c,state);a=np.zeros((27,owner.size))
        for cell in (0,1):
            cs=c.work.contact.cells[cell].evaluate(*args[:3])
            assert not np.any(cs.rotated_midpoint_lift)
            _,curvature=cs.pullback(np.arange(9,dtype=float),np.zeros((9,9)))
            assert not np.any(curvature)
            a[9*cell:9*cell+9,list(c.mapping[:42])]=cs.geometric_jacobian
        a[18:,list(c.mapping[42:])]=c.work.shell.work_map.toarray()
        dimension=np.ones(owner.size)
        dimension[[6*i+j for i in range(owner.n) for j in (3,4,5)]]=1./owner.length
        dimension[6*owner.n:6*owner.n+6*owner.e]=1./owner.length
        dimension[[owner.native_size+i for i in owner.shell_z]]=1./owner.length
        a*=dimension[None,:];bound=c.error_factor/np.sqrt(27.)
        assert np.max(np.sum(np.abs(a),axis=0))<=bound
        assert np.max(np.sum(np.abs(a),axis=1))<=bound
        assert not np.any(a[:,[6*i+j for i in range(owner.n) for j in (3,4,5)]])
    bounds=owner._evaluate(state)[-1]['native_multi_shell_contact']['error_bounds']
    assert all(x<=y for x,y in zip(bounds,(1e-8,1e-7,1e-6)))
    definition=owner._definition()
    assert definition['moment_scale_nm']==owner.length*definition['force_scale_n']


def test_public_coupled_two_step_multiple_body_response_and_both_path_legs():
    owner,result=run();data=result.to_dict()
    assert data['accepted_steps']==2 and data['accepted_time']==.006
    assert data['profile']==owner.trajectory_profile and not data['production_qualified']
    active=False
    for step in data['steps']:
        assert step['stage_residual_norm']<=1e-11 and step['endpoint_algebraic_norm']<=1e-11
        account=step['stage']['accounting'];active|=account['contact_regularization_storage']>0.
        assert len(account['native_multi_shell_contact']['pairs'])==4
        certificate=step['native_multi_shell_contact']
        assert len(certificate['pairs'])==4 and certificate['checked_intervals']>=8
        assert all(len(p['per_leg_checked_intervals'])==2 for p in certificate['pairs'])
        assert certificate['minimum_certified_gap_m']==-.003
        assert np.isfinite(certificate['stage_contact_internal_work_j'])
        for pair in account['native_multi_shell_contact']['pairs']:
            assert pair['minimum_gap_m']>=-.003
            assert np.linalg.norm(np.array(pair['beam_force_n'])+pair['shell_force_n'])/FORCE_REFERENCE_N<=1e-6
            assert np.linalg.norm(np.array(pair['beam_moment_nm'])+pair['shell_moment_nm'])/MOMENT_REFERENCE_NM<=1e-6
    assert active
    for o,size,p,_ in owner.shell_layout:assert np.linalg.norm(owner.current.shell_displacement[o+np.array(p)])>0.
    assert not np.array_equal(owner.current.velocity[:168],np.array(data['initial']['state']['velocity'])[:168])


def test_genuine_split_restart_chain_and_authenticated_translation_work_tamper():
    def forge_initial(owner, checkpoint):
        # Use the prefix producer inside its original construction lifetime.
        # The independent restart and full solves can outlive that owner.
        changed=json.loads(checkpoint);record=changed['body']['controller']['initial']
        record['state']['shell_displacement'][owner.shell_p[0]]+=1e-6
        candidate=owner._state(record['state']);record['state_sha256']=sha(candidate)
        record['accounting']=json.loads(canonical(owner._evaluate(candidate)[-1]))
        changed['checkpoint_sha256']=sha(changed['body'])
        return canonical(changed)

    replay=os.environ.get('NMS8_RESTART_REPLAY')
    if replay is None:
        owner,prefix=run(steps=1)
        forged_initial=forge_initial(owner,prefix.checkpoint)
        resumed_owner,resumed=run(checkpoint=prefix.checkpoint);full_owner,full=run()
        prefix_state=owner.current;resumed_state=resumed_owner.current;full_state=full_owner.current
    else:
        # Only guarded immutable authentic PUBLIC producer artifacts. Ordinary
        # pytest stays fresh. This branch compares saved results, not new solves.
        binding_path=Path(replay);binding=json.loads(binding_path.read_text())
        assert sha256(binding_path.read_bytes()).hexdigest()==os.environ['NMS8_RESTART_BINDING_SHA256']
        for name,digest in binding['files'].items():assert sha256(Path(name).read_bytes()).hexdigest()==digest
        def saved(label):
            payload=json.loads(Path(binding['records'][label]['result']).read_text())
            capsule=Path(binding['records'][label]['checkpoint']).read_bytes()
            return SimpleNamespace(checkpoint=capsule,to_dict=lambda:payload)
        prefix=saved('prefix');resumed=saved('resumed');full=saved('full');owner=build()
        assert binding['candidate_identity']==owner.identity
        forged_initial=forge_initial(owner,prefix.checkpoint)
        prefix_state=owner._state(prefix.to_dict()['final']['state'])
        resumed_state=owner._state(resumed.to_dict()['final']['state']);full_state=owner._state(full.to_dict()['final']['state'])
    assert prefix.to_dict()['accepted_steps']==1 and resumed.to_dict()['accepted_steps']==2
    assert resumed.to_dict()['steps'][1]['predecessor_sha256']==sha(prefix_state)
    assert resumed.checkpoint==full.checkpoint and sha(resumed_state)==sha(full_state)
    changed=json.loads(prefix.checkpoint)
    changed['body']['controller']['steps'][0]['native_multi_shell_contact']['shell_endpoint_projection_increment'][0]+=1e-6
    changed['checkpoint_sha256']=sha(changed['body']);raw=canonical(changed);fresh=build()
    with pytest.raises(NativeTransientError,match='native contact path/work journal mismatch'):
        solve_native_transient_multi_shell(fresh,config(fresh),checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),max_steps=64,max_cutbacks=8,minimum_dt=.003/256)
    assert fresh.current is None and not fresh._candidates
    raw=forged_initial;fresh=build()
    with pytest.raises(NativeTransientError,match='initial physical contract mismatch'):
        solve_native_transient_multi_shell(fresh,config(fresh),checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),max_steps=64,max_cutbacks=8,minimum_dt=.003/256)
    assert fresh.current is None


def test_aggregate_geometry_integration_and_two_leg_path_nonrenewal():
    budget=AggregateWorkBudget(1,1,1)
    for name in ('geometry','integration','swept'):
        budget.debit(name)
        with pytest.raises(MemoryError,match=RESOURCE):budget.debit(name)
        assert budget.used[name]==1
    owner=build(maximum_geometry_evaluations=1)
    with pytest.raises(MemoryError,match=RESOURCE):_initial(owner)
    assert owner.current is None and owner.last_operation_work['geometry']==1
    owner=build(maximum_integration_intervals=1);state=_initial(owner)
    with pytest.raises(MemoryError,match=RESOURCE):
        with owner._operation():owner._stage(state,np.zeros(owner.size),.02,None)
    assert owner.last_operation_work['integration']==1 and owner.current is state
    owner=build(maximum_swept_intervals=7);state=_initial(owner)
    with pytest.raises(ValueError,match=PATH_FAILURE):
        with owner._operation():owner._swept(state,state,state,None)
    assert owner.last_operation_work['swept']==7 and owner.current is state


def test_internal_edge_transition_refusal_and_inverted_skin_restore_admission():
    owner=build(separation=.2);state=_initial(owner);data=owner._data(state)
    # Translate beam1 to the artificial diagonal, without adding an edge force.
    positions=state.positions.copy();positions[:3,1]+=3.;data['positions']=positions
    changed=owner._state(data)
    with pytest.raises(ValueError,match=BOUNDARY) as failure:owner._admit_saved_state(changed)
    assert ((4,7),TRANSITION) in failure.value.boundary_inventory
    assert ((4,5),'PHYSICAL_EXTERIOR_RIM') in failure.value.boundary_inventory
    data=owner._data(state);positions=state.positions.copy();positions[:3,1]-=3.;data['positions']=positions
    with pytest.raises(ValueError,match=BOUNDARY) as failure:owner._admit_saved_state(owner._state(data))
    assert ((4,5),'PHYSICAL_EXTERIOR_RIM') in failure.value.boundary_inventory
    assert owner.shell_topologies[0]['internal_edge_semantics']==TRANSITION
    # Orientation proof is mandatory independently of positive height/gap.
    q=state.shell_displacement.copy();first=owner.shell_bodies[0].model
    q[first.mesh.dof_manager.get_node_dofs(5)[0]]=-24.
    changed=replace(state,shell_displacement=q)
    refused=AggregateWorkBudget(1,1,1)
    with pytest.raises(ValueError,match='physical side/orientation'):
        owner._geometry(owner.contributors[0],changed,refused)
    assert refused.used['geometry']==1
    with pytest.raises(ValueError,match='small-motion'):owner._admit_saved_state(changed)


def test_topology_duplicate_pair_model_alias_and_definition_mutation_refusals(monkeypatch):
    native,shell,pairs=selections()
    with pytest.raises(ValueError,match='duplicate contact'):build(selections_override=(native,shell,pairs+(pairs[0],)))
    with pytest.raises(ValueError,match='alias'):build(selections_override=(native,(shell[0],S3ContactBody(8,shell[0].model)),pairs))
    model=shell[0].model;model.mesh.elements[23]=model.mesh.elements[21]
    with pytest.raises(ValueError):build(selections_override=(native,shell,pairs))
    owner=build();state=_initial(owner);owner.shell_bodies[0].model.materials['elastic'].elastic_modulus*=2
    with pytest.raises(ValueError):owner.trial(state,dt=.003)
    assert owner.current is state and not owner._candidates
    from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
    native,shell,pairs=selections()
    too_many=tuple(replace(native[0],body_id=i) for i in (1,)+tuple(range(100,119)))
    def forbidden(*args):raise AssertionError('native preparation reached before global cap')
    monkeypatch.setattr(NativeBeamDefinition,'instantiate',forbidden)
    with pytest.raises(MemoryError,match='before preparation'):
        build(selections_override=(too_many,shell,pairs))


def test_cancellation_unresolved_path_atomic_rollback_and_common_native_map():
    owner=build();state=_initial(owner);before=canonical(state)
    token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):owner.trial(state,dt=.003,cancellation_token=token)
    assert canonical(owner.current)==before and not owner._candidates
    original=owner._swept
    def unresolved(*args):raise ValueError(PATH_FAILURE)
    owner._swept=unresolved
    with pytest.raises(ValueError,match=PATH_FAILURE):owner.trial(state,dt=.003)
    owner._swept=original
    assert canonical(owner.current)==before and not owner._candidates
    # A second explicit target uses exactly the SAME retained native42 map.
    native,shell,pairs=selections(.2);second=shell[1].model
    for n,p in zip((4,5,6),((-6.,-6.),(6.,-6.),(6.,6.))):
        second.mesh.nodes[n].x,second.mesh.nodes[n].y=p
    pairs=pairs[:3]+(NativeTransientS3ContactPair(1,1,8,21),)
    multiple=build(selections_override=(native,shell,pairs))
    assert multiple.contributors[0].mapping[:42]==multiple.contributors[1].mapping[:42]
    assert multiple.contributors[0].mapping[42:]!=multiple.contributors[1].mapping[42:]


def test_default_none_kernel_compatibility_and_old_exact_owner_identity():
    from examples.general_contact_native_shell_impact_dev import build as old_build,config as old_config
    owner=old_build();cfg=old_config(owner)
    with owner._operation():state=owner._initialize_state(cfg.initial_velocity,None,cfg.initial_displacement)
    stage=owner._stage(state,np.zeros(60),.02,None)[0]
    args=(stage.positions,stage.position_low,stage.cell_rotations[0],stage.shell_displacement)
    a=owner.contact.evaluate(*args);b=owner.contact.evaluate(*args,_work_budget=None)
    assert canonical(a.contact)==canonical(b.contact)
    np.testing.assert_array_equal(a.residual,b.residual);np.testing.assert_array_equal(a.tangent.toarray(),b.tangent.toarray())
    identity=owner.identity;kernel=owner.contact.contact.identity
    ledger=AggregateWorkBudget(1,1000,1);c=owner.contact.evaluate(*args,_work_budget=ledger)
    assert canonical(a.contact)==canonical(c.contact) and ledger.used['geometry']==1
    assert owner.identity==identity and owner.contact.contact.identity==kernel
    assert owner.size==60 and owner.route_profile!=PROFILE


def test_repeated_guard_retains_shell_configuration_methods_and_matrix_mutation_refusals():
    owner=build();state=_initial(owner);before=canonical(state)
    element=owner.shell_bodies[0].model.mesh.elements[21]
    def refused():
        with pytest.raises((ValueError,RuntimeError)):
            owner.trial(state,dt=.003)
        assert canonical(owner.current)==before and not owner._candidates
    calls=[];original=owner._shell_contract
    def counted(*args,**kwargs):
        calls.append(kwargs.get('assemble',True));return original(*args,**kwargs)
    owner._shell_contract=counted
    for _ in range(3):owner._guard()
    assert calls==[]
    old_angle=element.material_angle_deg
    object.__setattr__(element,'material_angle_deg',old_angle+1.)
    try:refused()
    finally:object.__setattr__(element,'material_angle_deg',old_angle)
    old=element.hourglass_stabilization
    object.__setattr__(element,'hourglass_stabilization',1.)
    try:refused()
    finally:object.__setattr__(element,'hourglass_stabilization',old)
    cls=type(element);old=vars(cls)['gauss_weights']
    type.__setattr__(cls,'gauss_weights',property(lambda self:np.array((1.,1.,1.))))
    try:refused()
    finally:type.__setattr__(cls,'gauss_weights',old)
    object.__setattr__(element,'compute_mass_matrix',lambda *args:np.zeros((18,18)))
    try:refused()
    finally:object.__delattr__(element,'compute_mass_matrix')
    cls=type(element);old=cls.compute_stiffness_matrix
    type.__setattr__(cls,'compute_stiffness_matrix',lambda *args:np.zeros((18,18)))
    try:refused()
    finally:type.__setattr__(cls,'compute_stiffness_matrix',old)
    old_flag=vars(cls)['legacy_stiffness_batch_eligible']
    assert 'get_dof_mapping' not in vars(cls)
    type.__delattr__(cls,'legacy_stiffness_batch_eligible')
    type.__setattr__(cls,'get_dof_mapping',lambda *args:())
    try:refused()
    finally:
        type.__delattr__(cls,'get_dof_mapping')
        type.__setattr__(cls,'legacy_stiffness_batch_eligible',old_flag)
    matrix=owner.shell_mass;owner.shell_mass=np.array(matrix,copy=True)
    try:refused()
    finally:owner.shell_mass=matrix
    changed=np.array(matrix,copy=True);changed[0,0]+=1.;changed.flags.writeable=False
    owner.shell_mass=changed
    try:refused()
    finally:owner.shell_mass=matrix
    owner._guard();owner._matrix_witness()
