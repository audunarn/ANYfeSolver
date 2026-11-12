"""Development gates for the bounded native/S3 common-stage candidate; executed evidence is separately bound."""
from dataclasses import replace
from fractions import Fraction as F
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import create_native_transient_shell_analysis, solve_native_transient_shell
from anyfesolver._ge_beam3_connected_dynamic import _RetainedMidpointAssembly
from anyfesolver._ge_beam3_native_shell_contact import PATH_FAILURE, PROFILE, SHELL_P, SHELL_Z
from anyfesolver._ge_beam3_native_transient import NativeTransientError
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver.control import CancellationToken, SolveCancelled
from examples.general_contact_native_shell_impact_dev import build, config, run


def _initial(owner):
    cfg=config(owner)
    with owner._operation():
        return owner._initialize_state(cfg.initial_velocity,None,cfg.initial_displacement)


def test_actual_formulation_mass_open_equations_and_permuted_node_momentum():
    owner=build(separation=.2,shell_insertion=(6,4,5));state=_initial(owner)
    r,j,fi,m,dv,dq,account=owner._evaluate(state)
    br,bj,bfi,bm,bdv,bdq,baccount=_RetainedMidpointAssembly._evaluate(owner,state)
    assert owner.size==60 and len(owner.physical)==24 and len(owner.algebraic)==36
    np.testing.assert_array_equal(m[list(owner.algebraic)],np.zeros((36,60)))
    np.testing.assert_array_equal(m[:,list(owner.algebraic)],np.zeros((60,36)))
    np.testing.assert_array_equal(m[42:,42:],owner.shell_mass)
    np.testing.assert_allclose(r[42:],owner.shell_stiffness@state.shell_displacement,atol=1e-12)
    np.testing.assert_array_equal(j[42:,42:],owner.shell_stiffness)
    np.testing.assert_array_equal(fi[42:],owner.shell_mass@state.acceleration[42:])
    assert account['contact_regularization_storage']==0.
    q=state.shell_displacement;v=state.velocity[42:].copy()
    for node,speed in zip(owner.contact.shell.node_ids,((.02,.03,.04),(-.01,.05,.06),(.07,-.02,.08))):
        v[owner.shell_model.mesh.dof_manager.get_node_dofs(node)[:3]]=speed
    data=owner._data(state);data['velocity']=np.r_[state.velocity[:42],v];state=owner._state(data)
    actual=owner._evaluate(state)[-1];bare=_RetainedMidpointAssembly._evaluate(owner,state)[-1]
    slots=np.array([owner.shell_model.mesh.dof_manager.get_node_dofs(n)[:3] for n in owner.contact.shell.node_ids])
    momentum=(owner.shell_mass@v)[slots]
    positions=np.array([owner.shell_model.mesh.nodes[n].coords() for n in owner.contact.shell.node_ids])+q[slots]
    np.testing.assert_allclose(np.array(actual['linear_momentum'])+actual['linear_momentum_low']-
        np.array(bare['linear_momentum'])-bare['linear_momentum_low'],momentum.sum(axis=0),atol=1e-12)
    np.testing.assert_allclose(np.array(actual['angular_momentum'])+actual['angular_momentum_low']-
        np.array(bare['angular_momentum'])-bare['angular_momentum_low'],np.cross(positions,momentum).sum(axis=0),atol=1e-12)
    assert not np.array_equal(slots,np.array(SHELL_P).reshape(3,3))
    reference=owner._reference_physical_pose(state)
    assert not np.any(reference.shell_displacement[list(SHELL_P)])
    changed=state.shell_displacement.copy();changed[SHELL_P[0]]+=1e-5
    assert not owner._physical_pose_matches(state,replace(state,shell_displacement=changed))


def test_common_stage_full60_derivative_offset_rotation_and_both_crossblocks():
    owner=build();accepted=_initial(owner);u=np.zeros(60);dt=.02
    stage,r,tangent,account=owner._stage(accepted,u,dt,None)
    assert account['contact_regularization_storage']>0.
    work=owner._contact(stage);contact_j=work.tangent.toarray()
    assert np.linalg.norm(contact_j[:42,42:])>0 and np.linalg.norm(contact_j[42:,:42])>0
    assert np.linalg.norm(contact_j[:,[42+i for i in SHELL_Z]])>0
    assert np.array_equal(stage.velocity[[42+i for i in SHELL_Z]],np.zeros(9))
    direction=np.sin(np.arange(60)+.3);direction[list(owner.p)]*=100.
    increment=np.zeros(60);increment[list(owner.p)]=.5*dt*dt*direction[list(owner.p)]
    increment[list(owner.z)]=direction[list(owner.z)]
    chart=owner._midpoint_chart(np.zeros(60),u,dt)
    assert np.array_equal(chart[np.ix_([42+i for i in SHELL_Z],[42+i for i in SHELL_Z])],np.eye(9))
    for epsilon in (1e-6,1e-7):
        plus=owner._stage(accepted,u+epsilon*direction,dt,None)
        minus=owner._stage(accepted,u-epsilon*direction,dt,None)
        fd=(plus[1]-minus[1])/(2*epsilon)
        assert np.linalg.norm((fd-tangent@direction)/owner.scale)<=1e-5*max(1.,np.linalg.norm(fd/owner.scale))
        contact_fd=(owner._contact(plus[0]).residual-owner._contact(minus[0]).residual)/(2*epsilon)
        assert np.linalg.norm((contact_fd-contact_j@chart@direction)/owner.scale)<=1e-5*max(
            owner.contact.contact.force_tolerance,np.linalg.norm(contact_fd/owner.scale))
        energy_fd=(plus[3]['contact_regularization_storage']-minus[3]['contact_regularization_storage'])/(2*epsilon)
        expected=float(work.residual@chart@direction)
        assert abs(energy_fd-expected)<=1e-5*max(owner.contact.contact.energy_tolerance,abs(expected))
    skin_before=owner.contact.shell.current_positions(stage.shell_displacement)
    dq=np.zeros(18);dq[SHELL_Z[0]]=1e-4
    assert np.linalg.norm(owner.contact.shell.current_positions(stage.shell_displacement+dq)-skin_before)>0.


def test_public_two_step_common_stage_impact_and_actual_projected_path():
    owner,result=run();data=result.to_dict()
    assert data['accepted_steps']==2 and data['accepted_time']==.006
    assert data['contact']=='NATIVE_RETAINED_ROUND_LINE_S3_INTERIOR' and not data['production_qualified']
    active=False
    for step in data['steps']:
        assert step['stage_residual_norm']<=1e-11 and step['endpoint_algebraic_norm']<=1e-11
        path=step['native_shell_contact'];active|=step['stage']['accounting']['contact_regularization_storage']>0.
        assert path['checked_intervals']==sum(path['per_leg_checked_intervals'])<=owner.maximum_swept_intervals
        assert len(path['per_leg_checked_intervals'])==2 and min(path['per_leg_checked_intervals'])>0
        assert path['minimum_certified_gap_m']>=-owner.maximum_penetration_m
        assert np.isfinite(path['stage_contact_internal_work_j'])
        expected=np.array(step['endpoint']['state']['shell_displacement'])-step['stage']['state']['shell_displacement']
        np.testing.assert_array_equal(path['shell_endpoint_projection_increment'],expected)
        for record in (step['stage'],step['endpoint']):
            account=record['accounting'];contact=account['native_shell_contact']
            assert account['total_energy']==account['elastic_energy']+account['kinetic_energy']+account['contact_regularization_storage']
            assert account['elastic_energy']==account['native_elastic_energy']+account['shell_elastic_energy']
            assert contact['moment_origin_m']==[0.,0.,0.]
            np.testing.assert_allclose(np.array(contact['beam_force_n'])+contact['shell_force_n'],0.,atol=1e-7)
            np.testing.assert_allclose(np.array(contact['beam_moment_nm'])+contact['shell_moment_nm'],0.,atol=1e-7)
            assert contact['minimum_gap_m']>=-owner.maximum_penetration_m
            assert contact['minimum_rim_clearance_m']>=owner.contact.contact.radius+owner.contact.contact.rim_clearance
            assert 'NOT_MATERIAL_SURFACE_VELOCITY' in account['shell_rate_semantics']
    assert active
    initial=data['initial'];final=data['final']
    assert not np.array_equal(initial['state']['positions'],final['state']['positions'])
    assert not np.array_equal(initial['state']['shell_displacement'],final['state']['shell_displacement'])
    assert any(np.linalg.norm(s['native_shell_contact']['shell_endpoint_projection_increment'])>0 for s in data['steps'])
    np.testing.assert_allclose(final['accounting']['linear_momentum'],initial['accounting']['linear_momentum'],atol=1e-6)
    assert np.isfinite(final['accounting']['angular_momentum']).all()


def test_fresh_one_plus_one_restart_exact_bytes_and_replay_tamper_refusal():
    owner,prefix=run(steps=1);resumed_owner,resumed=run(checkpoint=prefix.checkpoint);full_owner,full=run()
    assert prefix.to_dict()['accepted_steps']==1 and resumed.to_dict()['accepted_steps']==2
    assert resumed.to_dict()['steps'][1]['predecessor_sha256']==sha(owner.current)
    assert resumed.checkpoint==full.checkpoint and sha(resumed_owner.current)==sha(full_owner.current)
    changed=json.loads(prefix.checkpoint)
    changed['body']['controller']['steps'][0]['native_shell_contact']['shell_endpoint_projection_increment'][3]+=.001
    changed['checkpoint_sha256']=sha(changed['body']);raw=canonical(changed);fresh=build()
    with pytest.raises(NativeTransientError,match='native contact path/work journal mismatch'):
        solve_native_transient_shell(fresh,config(fresh),checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),
            max_steps=64,max_cutbacks=8,minimum_dt=.003/256)
    assert fresh.current is None and not fresh._candidates
    # Authentic operators reevaluate a changed initial shell translation; the
    # digest/accounting are repaired, so the declared physical contract must
    # itself refuse the changed state before publication. No extra solve.
    changed=json.loads(prefix.checkpoint);record=changed['body']['controller']['initial']
    record['state']['shell_displacement'][SHELL_P[0]]+=1e-6
    candidate=owner._state(record['state'])
    record['state_sha256']=sha(candidate)
    record['accounting']=json.loads(canonical(owner._evaluate(candidate)[-1]))
    changed['checkpoint_sha256']=sha(changed['body']);raw=canonical(changed);fresh=build()
    with pytest.raises(NativeTransientError,match='initial physical contract mismatch'):
        solve_native_transient_shell(fresh,config(fresh),checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),
            max_steps=64,max_cutbacks=8,minimum_dt=.003/256)
    assert fresh.current is None and not fresh._candidates


def test_touch_constant_plane_tight_budget_and_shared_two_leg_cap():
    owner=build(maximum_penetration_m=.00005,separation=.05,maximum_swept_intervals=1)
    beam=owner.reference_positions.copy();skin=owner.contact.shell.reference_positions.copy()
    assert owner._leg_certificate(beam,beam,skin,skin,F(0),F(0),None)==1
    budget={'visited':0}
    assert owner._leg_certificate(beam,beam,skin,skin,F(0),F(0),None,budget)==1
    with pytest.raises(ValueError,match=PATH_FAILURE) as failure:
        owner._leg_certificate(beam,beam,skin,skin,F(0),F(0),None,budget)
    assert failure.value.path_checked_intervals==1 and failure.value.path_interval_cap==1


def test_clear_leg_endpoints_do_not_admit_intermediate_penetration_or_wrong_side():
    owner=build(maximum_swept_intervals=16)
    beam=np.array(((.5,.5,0.),(.51,.5,0.),(.52,.5,0.)))
    s0=np.array(((0.,0.,-.08),(1.,0.,-.08),(0.,10.,-.08)))
    s1=np.array(((0.,0.,-.33),(2.,0.,.67),(0.,10.,-.33)))
    def gaps(skin):
        normal=np.cross(skin[1]-skin[0],skin[2]-skin[0]);normal/=np.linalg.norm(normal)
        return (beam-skin[0])@normal-owner.contact.contact.radius
    assert np.min(gaps(s0))>0 and np.min(gaps(s1))>0
    assert np.min(gaps(.5*(s0+s1))) < -owner.maximum_penetration_m
    with pytest.raises(ValueError,match=PATH_FAILURE):owner._leg_certificate(beam,beam,s0,s1,F(0),F(0),None)
    wrong=owner.contact.shell.reference_positions.copy();wrong[:,2]=.1
    with pytest.raises(ValueError,match=PATH_FAILURE):owner._leg_certificate(owner.reference_positions,owner.reference_positions,wrong,wrong,F(0),F(0),None)


def test_trusted_native_capsule_inverted_skin_and_small_motion_refusal():
    owner=build(separation=.2,maximum_shell_displacement_m=20.);state=owner.initialize()
    data=owner._data(state);q=state.shell_displacement.copy()
    for node in owner.contact.shell.node_ids:
        x=int(owner.shell_model.mesh.dof_manager.get_node_dofs(node)[0])
        q[x]=-2*owner.shell_model.mesh.nodes[node].coords()[0]
    data['shell_displacement']=q;data['positions']=state.positions+np.array((0.,0.,-.45))
    invalid=owner._state(data)
    r,_,_,mass,*_=owner._evaluate(invalid)
    acceleration=np.zeros(60);p=list(owner.p);acceleration[p]=np.linalg.solve(mass[np.ix_(p,p)],-r[p])
    data['acceleration']=acceleration;invalid=owner._state(data)
    assert owner._norm(owner._evaluate(invalid)[0],owner.z)<=1e-11
    raw=owner._encode_state(invalid);fresh=build(separation=.2,maximum_shell_displacement_m=20.)
    with pytest.raises(ValueError,match='side or orientation'):
        fresh.restore_checkpoint(raw,expected_sha256=sha256(raw).hexdigest())
    assert fresh.current is None and not fresh._candidates
    data=owner._data(state);data['shell_displacement']=np.full(18,1.)
    with pytest.raises(ValueError,match='small-motion'):owner._state(data)


def test_side_definition_mutation_cancellation_and_trial_rollback():
    for options in ({'shell_side':'bottom'},{'shell_side':'midsurface'},{'director_polarity':-1}):
        with pytest.raises(ValueError):build(**options)
    owner=build();state=_initial(owner);before=canonical(state)
    token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):owner.trial(state,dt=.003,cancellation_token=token)
    assert canonical(owner.current)==before and not owner._candidates
    original=owner._swept
    def unresolved(*args,**kwargs):raise ValueError(PATH_FAILURE)
    owner._swept=unresolved
    with pytest.raises(ValueError,match=PATH_FAILURE):owner.trial(state,dt=.003)
    owner._swept=original
    assert canonical(owner.current)==before and not owner._candidates
    owner.shell_model.materials['elastic'].elastic_modulus*=2
    with pytest.raises(ValueError):owner.trial(state,dt=.003)
    assert canonical(owner.current)==before and not owner._candidates
