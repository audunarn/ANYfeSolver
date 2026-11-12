"""Stage-event checks: pressure law, physical gap, state and work are separate."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.transient import ContactTransientSolverControls,solve_transient_contact
from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
from test_general_contact_transient_plane import _guided_plate
from test_general_contact_transient_q4_plastic import (CONTROLS as PLASTIC_CONTROLS,
    run as plastic_run,physical,scalar_reference)


EVENTS=ContactTransientSolverControls(q4_plane_event_subdivision_dev=True,equilibrium_tolerance=1e-9)


def elastic(*,dt=.001,end=.014,checkpoint=None,emit=False,controls=EVENTS):
    model,definition,velocity=_guided_plate()
    result=solve_transient_contact(model,definition,TransientConfig(dt,end,initial_velocity=velocity),
        solver_controls=controls,restart_checkpoint=checkpoint,emit_restart_checkpoint=emit)
    return model,result


def normal_reference(times):
    mass,stiffness,gap=.05,1e5,.01
    w,v,a=0.,-1.,0.
    rows=[(w,v,a)]
    for left,right in zip(times,times[1:]):
        dt=right-left;a0=.25*dt*dt
        predictor=w+dt*v+a0*a
        next_w=((mass*predictor-a0*stiffness*gap)/(mass+a0*stiffness)
            if predictor+gap<0. else predictor)
        next_a=(next_w-predictor)/a0
        v+=.5*dt*(a+next_a);w,a=next_w,next_a
        rows.append((w,v,a))
    return np.array(rows)


def test_elastic_plane_events_match_independent_newmark_and_energy():
    model,result=elastic()
    reference=normal_reference(result.times)
    dof=model.mesh.dof_manager.get_node_dofs(1)[2]
    np.testing.assert_allclose(result.displacements[:,dof],reference[:,0],atol=1e-10)
    np.testing.assert_allclose(result.velocities[:,dof],reference[:,1],atol=1e-7)
    np.testing.assert_allclose(result.accelerations[:,dof],reference[:,2],atol=1e-4)
    kinetic=.5*.05*result.velocities[:,dof]**2
    storage=np.array([row.penalty_energy for row in result.contact])
    np.testing.assert_allclose(kinetic+storage,.025,rtol=1e-6,atol=1e-10)
    assert result.contact[-1].active_pressure_dofs==0
    assert result.localized_contact_event_times
    assert result.event_endpoint_evaluations>0
    assert result.plane_event_stage_evaluations>len(result.newton_iterations)
    assert all(row.gap_tolerance_met for row in result.contact)
    assert min(result.swept_minimum_gaps)>=-.001


def test_plane_event_budget_exhaustion_is_explicit_and_profile_is_opt_in():
    with pytest.raises(RuntimeError,match='budget exhausted'):
        elastic(controls=replace(EVENTS,maximum_plane_event_stage_evaluations=1))
    _,old=elastic(controls=ContactTransientSolverControls())
    assert old.localized_contact_event_times==()
    assert old.plane_event_stage_evaluations==0
    assert old.profile=='PLANAR_Q4_FIXED_PLANE_NEWMARK_PENALTY_DEV_V1'


def test_failed_endpoint_query_counts_and_cutback_preserves_equilibrium(monkeypatch):
    import anyfesolver.general_contact.plane_newmark_events as events
    from anyfesolver.general_contact.event_subdivision import EventNeedsCutback
    original=events.localize_endpoint_event
    attempts=0
    failed=False
    def controlled(callback,*args,**kwargs):
        nonlocal attempts,failed
        def counted(duration):
            nonlocal attempts
            attempts+=1
            return callback(duration)
        if not failed:
            failed=True
            counted(args[0]*.5)
            raise EventNeedsCutback('controlled endpoint-family failure')
        return original(counted,*args,**kwargs)
    monkeypatch.setattr(events,'localize_endpoint_event',controlled)
    model,result=elastic(dt=.0009)
    dof=model.mesh.dof_manager.get_node_dofs(1)[2]
    np.testing.assert_allclose(result.displacements[:,dof],normal_reference(result.times)[:,0],atol=1e-10)
    assert failed and result.cutbacks>0
    assert result.event_endpoint_evaluations==attempts
    assert all(row.gap_tolerance_met for row in result.contact)


@pytest.mark.parametrize('end',[.00037,.00999999998])
def test_arbitrary_end_truncation_cannot_emit_event_checkpoint(end):
    with pytest.raises(ValueError,match='untruncated final time step'):
        elastic(end=end,emit=True)


def test_cancellation_in_event_family_after_yield_preserves_reusable_model(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact.core import active_contact_manager
    import anyfesolver.general_contact.q4_material_checkpoint as codec
    import anyfesolver.general_contact.plane_newmark_events as events
    from test_general_contact_transient_q4_plastic import fixture
    model,_,_,modes,definition=fixture()
    controls=replace(PLASTIC_CONTROLS,q4_plane_event_subdivision_dev=True)
    token=CancellationToken()
    prepare,localize=codec.prepare_q4_material_checkpoint,events.localize_endpoint_event
    yielded=False
    cancelled=False
    def observe(*args,**kwargs):
        nonlocal yielded
        packet=prepare(*args,**kwargs)
        yielded=yielded or bool(np.min(args[2][1]['alpha'])>0.)
        return packet
    def interrupt(callback,*args,**kwargs):
        def at_endpoint(duration):
            nonlocal cancelled
            if yielded:
                cancelled=True
                token.cancel('owned yielded event-family interruption')
            return callback(duration)
        return localize(at_endpoint,*args,**kwargs)
    def solve(end,token=None):
        return solve_transient_contact(model,definition,
            TransientConfig(3e-5,end,initial_velocity=modes@np.array([.2,-1.])),
            solver_controls=controls,cancellation_token=token)
    with monkeypatch.context() as patch:
        patch.setattr(codec,'prepare_q4_material_checkpoint',observe)
        patch.setattr(events,'localize_endpoint_event',interrupt)
        with pytest.raises(SolveCancelled,match='stage evaluation'):
            solve(.0012,token)
    assert yielded and cancelled and active_contact_manager() is None
    reused=solve(.0003)
    _,fresh,_=plastic_run(dt=3e-5,end=.0003,controls=controls)
    np.testing.assert_array_equal(reused.displacements,fresh.displacements)
    assert reused.element_states==fresh.element_states


@pytest.fixture(scope='module')
def plastic_events():
    controls=replace(PLASTIC_CONTROLS,q4_plane_event_subdivision_dev=True)
    return plastic_run(dt=3e-5,controls=controls),controls


def test_plastic_plane_events_preserve_independent_core_and_endpoint_work(plastic_events):
    (_,result,modes),_=plastic_events
    reference=scalar_reference(result.times)
    coordinates=physical(result.displacements,modes)
    np.testing.assert_allclose(coordinates,[row[0] for row in reference],atol=1e-11)
    np.testing.assert_allclose(physical(result.velocities,modes),[row[1] for row in reference],atol=1e-8)
    for history,row in zip(result.material_state_history,reference):
        np.testing.assert_allclose(history[1]['alpha'],row[4],atol=1e-10)
        np.testing.assert_allclose(history[1]['plastic_strain'],np.tile([row[3],-row[3],0.],(12,1)),atol=1e-10)
    velocities=physical(result.velocities,modes)
    alpha=np.array([state[1]['alpha'][0] for state in result.material_state_history])
    p=np.array([state[1]['plastic_strain'][0][0] for state in result.material_state_history])
    G=1e5/2.6;tau=2*G*(coordinates[:,0]-p)
    energy=.5/120*velocities[:,0]**2+.5*.05*velocities[:,1]**2
    energy+=.05*tau*tau/(2*G)+.05*20*alpha
    energy+=np.array([item.penalty_energy for item in result.contact])
    # Yield timing is not localized by this contact-only event profile.
    assert np.max(np.abs(energy/energy[0]-1))<.01
    assert result.localized_contact_event_times
    assert alpha[-1]>0. and abs(tau[-1])<20/np.sqrt(3.)


def test_yielded_plane_event_restart_binds_phase_and_controls(plastic_events):
    (_,whole,modes),controls=plastic_events
    candidates=[i for i,(state,c) in enumerate(zip(whole.material_state_history,whole.contact))
        if state[1]['alpha'][0]>0. and c.active_pressure_dofs>0]
    split=float(whole.times[candidates[0]])
    _,prefix,_=plastic_run(dt=3e-5,end=split,emit=True,controls=controls)
    _,resumed,_=plastic_run(dt=3e-5,checkpoint=prefix.restart_checkpoint,controls=controls)
    assert prefix.restart_checkpoint['version']==9
    assert resumed.material_state_history[0][1]==prefix.element_states[1]
    for name,tol in (('displacements',1e-10),('velocities',1e-7),('accelerations',1e-4)):
        np.testing.assert_allclose(np.concatenate([getattr(prefix,name)[:-1],getattr(resumed,name)]),
            getattr(whole,name),rtol=0.,atol=tol)
    with pytest.raises(ValueError,match='incompatible'):
        plastic_run(dt=3e-5,checkpoint=prefix.restart_checkpoint,
            controls=replace(controls,plane_event_gap_tolerance_m=2e-10))
    bad=copy.deepcopy(prefix.restart_checkpoint)
    bad['state']['plane_event_state']['departing_phase'][0]*=-1
    bad['checkpoint_sha256']=analysis_configuration_fingerprint({k:v for k,v in bad.items() if k!='checkpoint_sha256'})
    with pytest.raises(ValueError,match='phase/support'):
        plastic_run(dt=3e-5,checkpoint=bad,controls=controls)
