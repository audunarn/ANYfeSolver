"""C8 common-stage candidate gates; recorded execution outcomes remain separate.

These are prospective gates, not recorded passes or engineering acceptance.
"""
from dataclasses import replace
from hashlib import sha256
import json
import os
import faulthandler
import time
from pathlib import Path
import numpy as np
import pytest
from anyfesolver._ge_beam3_connected_dynamic import _RetainedMidpointAssembly
from anyfesolver._ge_beam3_native_pair_contact import PATH_FAILURE, PROFILE
from anyfesolver._ge_beam3_native_transient import NativeTransientError
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.general_contact import solve_native_transient_pair
from examples.general_contact_native_pair_impact_dev import build, config, run


def test_common_stage_contact_cross_blocks_and_complete_directional_tangent():
    owner = build(); accepted = owner.initialize(physical_velocity=config(owner).initial_velocity)
    u = np.zeros(owner.size); dt = .01
    stage, residual, tangent, account = owner._stage(accepted, u, dt, None)
    assert account['contact_regularization_storage'] > 0.
    r, j, _, mass, _, _, _ = owner._evaluate(stage)
    br, bj, *_ = _RetainedMidpointAssembly._evaluate(owner, stage)
    a, b = list(owner.maps[0]), list(owner.maps[1])
    assert np.linalg.norm((j-bj)[np.ix_(a,b)]) > 0.
    assert np.linalg.norm((j-bj)[np.ix_(b,a)]) > 0.
    assert np.array_equal(mass[list(owner.algebraic)], np.zeros((len(owner.algebraic),owner.size)))
    assert np.array_equal(mass[:,list(owner.algebraic)], np.zeros((owner.size,len(owner.algebraic))))
    # Actual stage path derivatives, including algebraic traces and resultants.
    direction = np.sin(np.arange(owner.size)+.3); direction[list(owner.p)] *= 100.
    for epsilon in (1e-6, 1e-7):
        plus = owner._stage(accepted,u+epsilon*direction,dt,None)
        minus = owner._stage(accepted,u-epsilon*direction,dt,None)
        fd = (plus[1]-minus[1])/(2*epsilon)
        assert np.linalg.norm((fd-tangent@direction)/owner.scale) <= 1e-5*max(1.,np.linalg.norm(fd/owner.scale))
        # Zero physical lift leaves only the physical translation contact work.
        contact_chart_direction = np.zeros(owner.size)
        for node in range(owner.n):
            contact_chart_direction[6*node:6*node+3] = .25*dt*dt*direction[6*node:6*node+3]
        contact_work = (r-br) @ contact_chart_direction
        energy_fd = (plus[3]['contact_regularization_storage']-minus[3]['contact_regularization_storage'])/(2*epsilon)
        assert abs(energy_fd-contact_work) <= 1e-5*max(owner.contact.energy_tolerance,abs(contact_work))
        # Isolate contact from the much larger inertial derivative, so the
        # complete-system derivative check cannot conceal missing cross blocks.
        plus_contact = owner.contact.evaluate(*owner._poses(plus[0])).residual
        minus_contact = owner.contact.evaluate(*owner._poses(minus[0])).residual
        contact_fd = np.zeros(owner.size)
        contact_fd[list(owner.maps[0]+owner.maps[1])] = (plus_contact-minus_contact)/(2*epsilon)
        expected_contact = (j-bj) @ contact_chart_direction
        assert np.linalg.norm((contact_fd-expected_contact)/owner.scale) <= 1e-5*max(
            owner.contact.force_tolerance,np.linalg.norm(contact_fd/owner.scale))
    contact = owner.contact.evaluate(*owner._poses(stage))
    np.testing.assert_allclose(contact.beam_a_force_n+contact.beam_b_force_n,0.,atol=1e-7)
    np.testing.assert_allclose(contact.beam_a_moment_nm+contact.beam_b_moment_nm,0.,atol=1e-7)
    assert np.linalg.norm(r-br) > 0.


def test_public_two_body_impact_genuine_stages_storage_and_path():
    owner = build(); evaluate = owner.contact.evaluate
    timing = None; stack = None; restore = []
    if os.environ.get('C8_JOB_DIR'):
        timing = Path(os.environ['C8_JOB_DIR'])/'call_timing.ndjson'
        stack = (Path(os.environ['C8_JOB_DIR'])/'active_stacks.log').open('w')
        faulthandler.dump_traceback_later(12.,repeat=True,file=stack)
        def stamp(name,phase,args):
            record = dict(name=name,phase=phase,monotonic=time.monotonic())
            for value in args:
                if hasattr(value,'epoch'):record.update(epoch=value.epoch,state_time=value.time)
            if name == 'stage':record['unknown_norm'] = float(np.linalg.norm(args[1])); record['dt'] = args[2]
            with timing.open('a') as stream:stream.write(json.dumps(record)+'\n')
        def timed(name,original):
            def call(*args,**kwargs):
                stamp(name,'start',args)
                try:return original(*args,**kwargs)
                finally:stamp(name,'end',args)
            return call
        for name in ('_stage','_endpoint','_gap','_swept','_evaluate'):
            original = getattr(owner,name);restore.append((owner,name,original));setattr(owner,name,timed(name[1:],original))
        original = _RetainedMidpointAssembly._evaluate
        restore.append((_RetainedMidpointAssembly,'_evaluate',original))
        _RetainedMidpointAssembly._evaluate = timed('structural_evaluate',original)
    def observed(*poses,**kwargs):
        diagnostics = {}; kwargs['_diagnostics'] = diagnostics
        if timing is not None:stamp('contact_evaluate','start',())
        try:
            return evaluate(*poses,**kwargs)
        except Exception as failure:
            if os.environ.get('C8_JOB_DIR'):
                from anyfesolver.general_contact.native_retained_whole_pair import _private_interval_open, _private_factored_gap
                if 'source_curve' in diagnostics:
                    source,target = diagnostics['source_curve'],diagnostics['target_curve']
                    left,right = diagnostics['interval']
                    diagnostics['exact_private_open'] = _private_interval_open(source,target,left,right,owner.contact.radius)
                    diagnostics['exact_gap_brackets'] = tuple(_private_factored_gap(source,target,t,owner.contact.radius) for t in (left,.5*(left+right),right))
                def convert(value):
                    if isinstance(value,np.ndarray):return value.tolist()
                    if isinstance(value,np.generic):return value.item()
                    raise TypeError(type(value).__name__)
                (Path(os.environ['C8_JOB_DIR'])/'contact_failure_diagnostic.json').write_text(json.dumps(dict(
                    failure=str(failure),poses=poses,diagnostics=diagnostics),default=convert,indent=2)+'\n')
            raise
        finally:
            if timing is not None:stamp('contact_evaluate','end',())
    owner.contact.evaluate = observed
    try:
        result = solve_native_transient_pair(owner,config(owner),max_steps=64,max_cutbacks=8,minimum_dt=.003/256)
    finally:
        owner.contact.evaluate = evaluate
        for target,name,original in reversed(restore):setattr(target,name,original)
        if stack is not None:
            faulthandler.cancel_dump_traceback_later(); stack.close()
    data = result.to_dict()
    assert result.success and data['accepted_steps'] >= 2
    assert data['profile'] == owner.trajectory_profile and data['production_qualified'] is False
    assert len(owner.physical) == 30 and owner.size == 84
    assert any(s['stage']['accounting']['contact_regularization_storage'] > 0. for s in data['steps'])
    for step in data['steps']:
        assert step['stage_residual_norm'] <= 1e-11 and step['endpoint_algebraic_norm'] <= 1e-11
        path = step['native_contact']
        assert path['certified_geometry'] == 'EXACT_AFFINE_AND_ROUNDED_ADVANCE_RETAINED_CURVE_FAMILY'
        assert path['arithmetic_range_error_m'] > 0.
        assert path['minimum_certified_gap_m'] >= -owner.maximum_penetration_m
        assert path['checked_intervals'] <= owner.maximum_swept_intervals
        assert np.isfinite(path['stage_contact_internal_work_j'])
        for record in (step['stage'],step['endpoint']):
            account = record['accounting']
            assert account['total_energy'] == account['elastic_energy']+account['kinetic_energy']+account['contact_regularization_storage']
            contact = account['native_contact']
            assert contact['contact_kernel_identity'] == owner.contact.identity
            assert contact['contact_kernel_service'] == 'NATIVE_RETAINED_WHOLE_Q2_PAIR_DEV_V14_POOLED_PHYSICAL_GAP_QUADRATURE'
            assert contact['derivative_error_chart'] == '30_UNIQUE_PHYSICAL_CONTROL_POINT_COORDINATES_IN_METRES'
            assert contact['moment_origin_m'] == [0.,0.,0.]
            assert contact['force_semantics'] == 'PHYSICAL_CONTACT_FORCE_ON_BODY_NEGATIVE_POTENTIAL_GRADIENT'
            assert contact['assembled_residual_semantics'] == 'INTERNAL_POTENTIAL_GRADIENT_ADDED_TO_NATIVE_EQUILIBRIUM'
            np.testing.assert_allclose(np.array(contact['beam_a_force_n'])+contact['beam_b_force_n'],0.,atol=1e-7)
            np.testing.assert_allclose(np.array(contact['beam_a_moment_nm'])+contact['beam_b_moment_nm'],0.,atol=1e-7)
    # Both bodies respond; neither body is a prescribed or lagged target.
    first = np.array(data['initial']['state']['positions'])
    last = np.array(data['final']['state']['positions'])
    assert np.mean(last[:3,1]-first[:3,1]) > 0.
    assert np.mean(last[3:,1]-first[3:,1]) < 0.
    initial = data['initial']['accounting']; final = data['final']['accounting']
    np.testing.assert_allclose(final['linear_momentum'], initial['linear_momentum'], atol=1e-6)
    # H/energy are reported diagnostics, not exact invariants of this SO(3) method.
    assert np.isfinite(final['angular_momentum']).all() and np.isfinite(final['total_energy'])


def test_open_contact_uses_unchanged_structural_inertia_and_no_contact_force():
    owner = build(separation=.2); state = owner.initialize()
    values = owner._evaluate(state); bare = _RetainedMidpointAssembly._evaluate(owner,state)
    for actual, expected in zip(values[:6],bare[:6]):
        np.testing.assert_array_equal(actual,expected)
    assert values[-1]['contact_regularization_storage'] == 0.
    assert values[-1]['elastic_energy'] == bare[-1]['elastic_energy']


def test_restart_full_path_reconstruction_and_tamper_refusal():
    owner, prefix = run(steps=1)
    resumed_owner, resumed = run(checkpoint=prefix.checkpoint)
    full_owner, full = run()
    assert prefix.to_dict()['accepted_steps'] == 1
    assert resumed.to_dict()['accepted_steps'] == 2
    assert resumed.to_dict()['steps'][1]['predecessor_sha256'] == sha(owner.current)
    assert resumed.checkpoint == full.checkpoint
    assert sha(resumed_owner.current) == sha(full_owner.current)
    data = json.loads(prefix.checkpoint)
    data['body']['controller']['steps'][0]['native_contact']['minimum_certified_gap_m'] += .01
    data['checkpoint_sha256'] = sha(data['body']); raw = canonical(data)
    fresh = build()
    with pytest.raises(NativeTransientError):
        solve_native_transient_pair(fresh,config(fresh),checkpoint=raw,
            expected_sha256=sha256(raw).hexdigest(),max_steps=64,minimum_dt=.003/256)
    assert fresh.current is None and not fresh._issued


def test_initial_penetration_and_config_refuse_atomically():
    owner = build(separation=.099)
    with pytest.raises(ValueError):
        solve_native_transient_pair(owner,config(owner))
    assert owner.current is None and not owner._issued and not owner._candidates
    owner = build()
    with pytest.raises(NativeTransientError):
        solve_native_transient_pair(owner,replace(config(owner),rayleigh_alpha=.1))
    assert owner.current is None
    source = build(); state = source.initialize()
    increment = np.zeros(source.size)
    for node in source.elements[0].node_ids:
        increment[6*source.node_index[node]+1] = .01
    # Genuine retained endpoint algebraic/acceleration equilibrium is not a
    # penetration certificate. A trusted external digest cannot waive admission.
    bad_state, _ = source._endpoint(source._advance(state,increment),None)
    raw = source._encode_state(bad_state)
    fresh = build()
    with pytest.raises(ValueError,match='penetration budget'):
        fresh.restore_checkpoint(raw,expected_sha256=sha256(raw).hexdigest())
    assert fresh.current is None and not fresh._issued and not fresh._candidates


def test_through_crossing_path_fails_even_with_clear_endpoints():
    owner = build(); accepted = owner.initialize()
    u = np.zeros(owner.size); dt = .01
    # Swap entire body transverse positions. Both endpoints are open, the
    # actual affine path crosses both round centreline surfaces in between.
    for nodes, sign in ((owner.elements[0].node_ids,1.),(owner.elements[1].node_ids,-1.)):
        for node in nodes:
            u[6*owner.node_index[node]+1] = sign*.1005*2/dt**2
    assert owner._gap(owner._advance(accepted,owner._increment(accepted,u,dt)),0.) >= 0.
    with pytest.raises(ValueError,match=PATH_FAILURE):
        owner._swept(accepted,u,dt,None)
    assert owner.current is accepted and not owner._candidates


def test_path_cutback_replay_and_cancelled_receipt_cleanup(monkeypatch):
    owner = build(); original = owner._swept; calls = 0
    def once(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError(PATH_FAILURE)
        return original(*args,**kwargs)
    monkeypatch.setattr(owner,'_swept',once)
    result = solve_native_transient_pair(owner,config(owner,1,.001),max_steps=64,minimum_dt=.001/256)
    assert result.to_dict()['steps'][0]['cutbacks'] == [{'dt':.001,'reason':PATH_FAILURE}]
    fresh = build()
    replay = solve_native_transient_pair(fresh,config(fresh,1,.001),checkpoint=result.checkpoint,
        expected_sha256=sha256(result.checkpoint).hexdigest(),max_steps=64,minimum_dt=.001/256)
    assert replay.checkpoint == result.checkpoint
    owner = build(); token = CancellationToken(); original = owner._prepare_trial
    def cancel(*args,**kwargs):
        answer = original(*args,**kwargs); token.cancel('after genuine receipt'); return answer
    monkeypatch.setattr(owner,'_prepare_trial',cancel)
    with pytest.raises(SolveCancelled) as caught:
        solve_native_transient_pair(owner,config(owner,1),cancellation_token=token)
    assert caught.value.partial.to_dict()['accepted_steps'] == 0
    assert owner.current.epoch == 0 and not owner._candidates


def test_native_contact_identity_change_and_event_budget_refuse():
    with pytest.raises(ValueError):
        build(maximum_penetration_m=.1)
    with pytest.raises(ValueError):
        build(maximum_swept_intervals=4097)
    from anyfesolver._ge_beam3_native_pair_contact import NativePairContactAnalysis, ENDPOINT_POLICY
    from anyfesolver.general_contact.native_retained_whole_pair import NativeRetainedWholeCurvePair
    from examples.general_contact_native_pair_impact_dev import definitions
    curved = definitions(separation=.2,rise=.01)
    a,_ = curved[0].instantiate(); b,_ = curved[1].instantiate()
    valid = build(); c = valid.contact
    pair = NativeRetainedWholeCurvePair(a.element_id,a.operator.reference,a.node_ids,c.radius_a,
        b.element_id,b.operator.reference,b.node_ids,c.radius_b,
        line_stiffness_n_per_m2=c.stiffness,distance_tolerance_m=c.distance_tolerance,
        feature_tolerance_m=c.feature_tolerance,search_parameter_tolerance=c.search_parameter_tolerance,
        maximum_search_depth=c.maximum_search_depth,maximum_search_cells=c.maximum_search_cells,
        energy_tolerance_j=c.energy_tolerance,force_tolerance_n=c.force_tolerance,
        tangent_tolerance_n_per_m=c.tangent_tolerance,maximum_integration_intervals=c.maximum_integration_intervals,
        certify_curved_ownership=True,pooled_quadrature_budget=True)
    assert any(np.any(a.operator.reference.half_cell_lift(k,.5)) for k in (0,1))
    with pytest.raises(ValueError,match='straight zero-lift'):
        NativePairContactAnalysis(curved,(),pair,endpoint_policy=ENDPOINT_POLICY,
            maximum_penetration_m=.003,maximum_swept_intervals=128,inertia_order=4)
    owner = build(); state = owner.initialize()
    owner.contact.stiffness *= 2
    with pytest.raises(ValueError):
        owner.trial(state,dt=.001)
    assert owner.current is state and not owner._candidates
    owner = build(); state = owner.initialize(); owner.contact.radius += .01
    with pytest.raises(ValueError):
        owner.trial(state,dt=.001)
    assert owner.current is state and not owner._candidates
    owner = build(maximum_swept_intervals=1); state = owner.initialize()
    u = np.zeros(owner.size)
    for node in owner.elements[0].node_ids:
        u[6*owner.node_index[node]+1] = .04*2/.01**2
    with pytest.raises(ValueError,match=PATH_FAILURE):
        owner._swept(state,u,.01,None)
