"""Independent affine Newmark/plateau oracle for contact-driven plasticity.

Reference arithmetic uses no production contact, material, assembly or solver call.
Mode/geometry constants below are frozen inputs, not fitted reference outputs.
"""
import itertools
import json
import os
from pathlib import Path
import runpy

import numpy as np
import pytest

from anyfesolver.assembly import assemble_mass_matrix
from anyfesolver import ContactTransientSolverControls, TransientConfig, solve_transient_contact
from anyfesolver.general_contact.core import ContactSolverSettings,_PlanarQ4PlaneManager

example=runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_q4_oblique_plastic_impact_dev.py'))
E,NU,H,RHO=1.e9,.3,.05,1.e4
G,YIELD,KN,GAP=E/2.6,.2,4.e11,.0008
NORMAL=np.array([np.sin(2.e-5),0.,np.cos(2.e-5)])
MASS=np.array([RHO*H/6.,RHO*H])
TAU_Y=YIELD/np.sqrt(3.)
D=np.array([[-NORMAL[0]/6.,NORMAL[2]],[NORMAL[0]/6.,NORMAL[2]]])
B=D[:,0]+NORMAL[2]*GAP
FORCE_SCALES=np.array([5.,1.5e7])


def contact(q):
    gaps=B+D@q
    pressure=KN*np.maximum(-gaps,0.)
    residual=KN*D.T@(.5*np.minimum(gaps,0.))
    storage=.25*KN*np.dot(np.minimum(gaps,0.),np.minimum(gaps,0.))
    return residual,pressure,storage


def reference(times,*,enabled=True):
    q=np.zeros(2);v=np.array([0.,-1.]);a=np.zeros(2)
    p=alpha=0.
    rows=[dict(q=q.copy(),v=v.copy(),a=a.copy(),p=p,alpha=alpha,tau=0.)]
    for left,right in zip(times,times[1:]):
        h=right-left; a0=.25*h*h
        predictor=q+h*v+a0*a
        candidates=[]
        for active,branch in itertools.product(itertools.product((False,True),repeat=2),(-1,0,1)):
            if not enabled and any(active):continue
            support=.5*np.asarray(active,dtype=float) if enabled else np.zeros(2)
            stiffness=KN*D.T@(support[:,None]*D)
            constant=KN*D.T@(support*B)
            if branch==0:
                stiffness[0,0]+=4*G*H
                constant[0]-=4*G*H*p
            else:
                constant[0]+=branch*2*H*TAU_Y
            trial_a=np.linalg.solve(np.diag(MASS)+a0*stiffness,-(stiffness@predictor+constant))
            trial=predictor+a0*trial_a
            gaps=B+D@trial
            if enabled and any((gap>1.e-14 if flag else gap<-1.e-14) for flag,gap in zip(active,gaps)):
                continue
            trial_tau=2*G*(trial[0]-p)
            if (branch==0 and abs(trial_tau)>TAU_Y+1.e-10
                    or branch!=0 and branch*trial_tau<TAU_Y-1.e-10):continue
            tau=trial_tau if branch==0 else branch*TAU_Y
            force=np.array([2*H*tau,0.])+(contact(trial)[0] if enabled else 0.)
            residual=MASS*trial_a+force
            if np.max(np.abs(residual)/FORCE_SCALES)>1.e-8:continue
            candidates.append((trial,trial_a,tau))
        assert candidates,'independent affine branch enumeration found no admissible solution'
        trial,trial_a,tau=candidates[0]
        for other,_,_ in candidates[1:]:
            np.testing.assert_allclose(other,trial,rtol=0.,atol=1.e-14)
        next_p=trial[0]-tau/(2*G)
        alpha+=2*abs(next_p-p)/np.sqrt(3.)
        p=next_p
        v=v+.5*h*(a+trial_a);q,a=trial,trial_a
        rows.append(dict(q=q.copy(),v=v.copy(),a=a.copy(),p=p,alpha=alpha,tau=tau))
    return rows


def test_oblique_fixture_mass_and_both_mode_work_match_independent_formulas():
    model,material,element,modes,definition=example['build_model']()
    np.testing.assert_allclose(modes.T@assemble_mass_matrix(model)[0]@modes,
        np.diag(MASS),rtol=2.e-13,atol=1.e-11)
    manager=_PlanarQ4PlaneManager(model,definition,ContactSolverSettings())
    for q in (np.array([0.,0.]),np.array([-1.e-10,-GAP]),np.array([-1.e-8,-GAP-2.e-5])):
        expected_tau=float(np.clip(2*G*q[0],-TAU_Y,TAU_Y))
        structural,_,_=element.compute_nonlinear_response(model.mesh,material,modes@q,None,3,True)
        actual=modes.T@structural
        assert np.max(np.abs(actual-[2*H*expected_tau,0.])/FORCE_SCALES)<1.e-6
        force,_,result=manager.evaluate(modes@q,tangent=True)
        expected,pressure,storage=contact(q)
        assert np.max(np.abs(modes.T@force-expected)/FORCE_SCALES)<1.e-6
        np.testing.assert_allclose(result.penalty_energy,storage,rtol=1.e-11,atol=1.e-12)
        # True corners use ±1/2, independently of mortar's weighted ±1/6.
        physical_gap=NORMAL[2]*(GAP+q[1])-abs(NORMAL[0]*(1+q[0]))*.5
        np.testing.assert_allclose(result.minimum_gap,physical_gap,rtol=0.,atol=1.e-14)


@pytest.fixture(scope='module',params=('coarse','half'))
def impact(request):
    dt=3.e-5 if request.param=='coarse' else 1.5e-5
    model,result,modes=example['solve'](dt=dt)
    q,v,a=(example['modal'](values,modes) for values in
        (result.displacements,result.velocities,result.accelerations))
    rows=reference(result.times)
    record=dict(dt_s=dt,times_s=result.times.tolist(),q=q.tolist(),v=v.tolist(),a=a.tolist(),
        alpha=[state[1]['alpha'][0] for state in result.material_state_history],
        plastic_xx=[state[1]['plastic_strain'][0][0] for state in result.material_state_history],
        reference= [{key:(value.tolist() if isinstance(value,np.ndarray) else value)
            for key,value in row.items()} for row in rows],
        cutback_diagnostics=result.cutback_diagnostics,
        events_s=result.localized_contact_event_times,
        endpoint_requests=result.event_endpoint_evaluations,
        stage_evaluations=result.plane_event_stage_evaluations,
        minimum_physical_path_gap_m=float(np.min(result.swept_minimum_gaps)))
    folder=Path(os.environ.get('RUNNER_TEMP',str(Path(__file__).parent)))
    (folder/('oblique_impact_'+request.param+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    return model,result,modes


def test_oblique_impact_drives_predicted_plastic_history_and_modal_equilibrium(impact):
    _,result,modes=impact
    expected=reference(result.times)
    assert max(row['alpha'] for row in expected)<.005
    assert all(np.max(state[1]['alpha'])<.005 for state in result.material_state_history)
    q=example['modal'](result.displacements,modes)
    v=example['modal'](result.velocities,modes)
    a=example['modal'](result.accelerations,modes)
    np.testing.assert_allclose(q[:,0],[row['q'][0] for row in expected],rtol=1.e-5,atol=2.e-15)
    np.testing.assert_allclose(q[:,1],[row['q'][1] for row in expected],rtol=0.,atol=1.e-11)
    np.testing.assert_allclose(v[:,0],[row['v'][0] for row in expected],rtol=1.e-5,atol=2.e-11)
    np.testing.assert_allclose(v[:,1],[row['v'][1] for row in expected],rtol=0.,atol=1.e-7)
    errors=[]
    for position,acceleration,states,row in zip(q,a,result.material_state_history,expected):
        actual=states[1]
        np.testing.assert_allclose(actual['alpha'],row['alpha'],rtol=1.e-5,atol=2.e-15)
        np.testing.assert_allclose(actual['plastic_strain'],np.tile([row['p'],-row['p'],0.],(12,1)),
            rtol=1.e-5,atol=2.e-15)
        tau=2*G*(position[0]-np.asarray(actual['plastic_strain'])[0,0])
        forces=np.array([2*H*tau,0.])+contact(position)[0]
        errors.append(np.abs(MASS*acceleration+forces)/FORCE_SCALES)
    assert np.max(errors)<1.e-6
    alpha=np.array([state[1]['alpha'][0] for state in result.material_state_history])
    first=np.flatnonzero(alpha>1.e-15)[0]
    assert q[0,0]==v[0,0]==0.
    assert result.contact[first].active_pressure_dofs>0
    assert alpha[-1]>1.e-9 and result.element_states[1]['plastic_strain'][0][0]<0.
    assert result.localized_contact_event_times and all(row.gap_tolerance_met for row in result.contact)
    assert min(result.swept_minimum_gaps)>=-5.e-5
    assert len(result.cutback_diagnostics)==result.cutbacks


def test_independent_no_contact_counterfactual_has_no_membrane_motion_or_yield(impact):
    _,result,modes=impact
    counterfactual=reference(result.times,enabled=False)
    assert all(row['q'][0]==row['v'][0]==row['p']==row['alpha']==0. for row in counterfactual)
    first_active=next(i for i,row in enumerate(result.contact) if row.active_pressure_dofs)
    q=example['modal'](result.displacements,modes)
    np.testing.assert_allclose(q[:first_active,0],0.,rtol=0.,atol=1.e-18)


def test_contact_driven_yielded_restart_keeps_material_origin_and_constraint_work():
    controls=ContactTransientSolverControls(allow_q4_plastic_dev=True,
        save_q4_material_history_dev=True,q4_plane_event_subdivision_dev=True,
        save_constraint_reactions_dev=True,equilibrium_tolerance=1.e-13)
    def run(end,checkpoint=None,emit=False):
        model,_,_,modes,definition=example['build_model']()
        result=solve_transient_contact(model,definition,
            TransientConfig(3.e-5,end,initial_velocity=modes@np.array([0.,-1.])),
            solver_controls=controls,restart_checkpoint=checkpoint,
            emit_restart_checkpoint=emit)
        return result,modes
    # Previously saved nominal step, not an arbitrary END in the event gap band.
    prefix,modes=run(.0008933632224845885,emit=True)
    assert prefix.contact[-1].active_pressure_dofs>0
    assert np.min(prefix.element_states[1]['alpha'])>0.
    resumed,_=run(.0011,checkpoint=prefix.restart_checkpoint)
    assert resumed.material_state_history[0][1]==prefix.element_states[1]
    times=np.r_[prefix.times[:-1],resumed.times]
    positions=np.concatenate((prefix.displacements[:-1],resumed.displacements))
    velocities=np.concatenate((prefix.velocities[:-1],resumed.velocities))
    states=(*prefix.material_state_history[:-1],*resumed.material_state_history)
    expected=reference(times)
    q,v=example['modal'](positions,modes),example['modal'](velocities,modes)
    np.testing.assert_allclose(q,[row['q'] for row in expected],rtol=0.,atol=1.e-11)
    np.testing.assert_allclose(v[:,0],[row['v'][0] for row in expected],rtol=1.e-5,atol=2.e-11)
    np.testing.assert_allclose(v[:,1],[row['v'][1] for row in expected],rtol=0.,atol=1.e-7)
    for state,row in zip(states,expected):
        np.testing.assert_allclose(state[1]['alpha'],row['alpha'],rtol=1.e-5,atol=2.e-15)
        np.testing.assert_allclose(state[1]['plastic_strain'],np.tile([row['p'],-row['p'],0.],(12,1)),
                                   rtol=1.e-5,atol=2.e-15)
    for result in (prefix,resumed):
        reaction_work=result.constraint_reactions@modes
        assert np.max(np.abs(reaction_work)/FORCE_SCALES)<1.e-6
    assert resumed.contact[-1].active_pressure_dofs==0
