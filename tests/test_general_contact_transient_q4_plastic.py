"""Independent affine deviatoric plasticity plus rigid-normal plane impact.

The physical modes are uncoupled. This verifies concurrent state ownership and
restart, not interaction-driven plasticity or general nonproportional loading.
"""
import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSurface, RigidContactBody, create_shell_element)
from anyfesolver.assembly import assemble_mass_matrix
from anyfesolver.dynamics import TransientConfig
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.core import active_contact_manager
from anyfesolver.general_contact.transient import ContactTransientSolverControls, solve_transient_contact
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint

E, NU, H, RHO = 1.e5, .3, .05, 1.
G, YIELD, KN, GAP = E/(2*(1+NU)), 20., 4.e7, .0008
DT, END = 2.5e-5, .003
CONTROLS = ContactTransientSolverControls(allow_q4_plastic_dev=True,
    save_q4_material_history_dev=True, equilibrium_tolerance=1.e-9)


def fixture():
    model = FEModel('affine-plastic-plane-impact')
    curve = DNVC208MaterialCurve(sigma_prop=YIELD,sigma_yield=YIELD,
        sigma_yield_2=YIELD,eps_p_y1=.005,eps_p_y2=.01,K=20.,n=.2)
    material = model.add_material('plastic',E,NU,density=RHO,yield_stress=YIELD,
        hardening_curve=curve)
    for node, (x,y) in enumerate(((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)),1):
        model.add_node(node,x,y,0.)
    element = create_shell_element(1,[1,2,3,4],'plastic',thickness=H)
    model.add_element(1,element)
    model.add_boundary_condition(BoundaryCondition('zero rotations',[1,2,3,4],
        dict(rx=0.,ry=0.,rz=0.)))
    n = model.mesh.dof_manager.total_dofs
    modes = np.zeros((n,2))
    master_e = model.mesh.dof_manager.get_node_dofs(2)[0]
    master_w = model.mesh.dof_manager.get_node_dofs(1)[2]
    for node in (1,2,3,4):
        x,y = model.mesh.nodes[node].x,model.mesh.nodes[node].y
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        modes[dofs[0],0],modes[dofs[1],0],modes[dofs[2],1]=x,-y,1.
        for dof,master,factor in ((dofs[0],master_e,2*x),
                (dofs[1],master_e,-2*y),(dofs[2],master_w,1.)):
            if dof != master:
                model.add_constraint_equation(terms=((dof,1.),(master,-factor)))
    definition = ContactDefinition(surfaces=(ContactSurface('skin',(1,)),),
        rigid_bodies=(RigidContactBody('floor','plane',(0.,0.,-GAP),(0.,0.,1.)),),
        interactions=(ContactInteraction('support','skin','floor',KN,5.e-5),))
    return model,material,element,modes,definition


def run(*, end=END,dt=DT,checkpoint=None,emit=False,controls=CONTROLS,token=None):
    model,material,element,modes,definition=fixture()
    result=solve_transient_contact(model,definition,
        TransientConfig(dt,end,initial_velocity=modes@np.array([.2,-1.])),
        solver_controls=controls,restart_checkpoint=checkpoint,
        emit_restart_checkpoint=emit,cancellation_token=token)
    return model,result,modes


def scalar_reference(times):
    """Closed two-coordinate plateau law; no production material/contact calls."""
    mass=np.array([RHO*H/6.,RHO*H])
    q=np.zeros(2); v=np.array([.2,-1.]); acc=np.zeros(2)
    p=alpha=0.
    rows=[(q.copy(),v.copy(),acc.copy(),p,alpha,0.)]
    tau_y=YIELD/math.sqrt(3.)
    for left,right in zip(times,times[1:]):
        dt=right-left; predictor=q+dt*v+.25*dt*dt*acc
        vp=v+.5*dt*acc; a0=.25*dt*dt
        # Each branch is affine: solve it directly, then check its inequalities.
        # This avoids a fictitious Newton stall at floating-point cancellation.
        elastic=(mass[0]*predictor[0]+a0*4*G*H*p)/(mass[0]+a0*4*G*H)
        tau_trial=2*G*(elastic-p)
        tau=float(np.clip(tau_trial,-tau_y,tau_y))
        trial=np.array([elastic,predictor[1]])
        p_next=p
        if abs(tau_trial)>tau_y:
            trial[0]=predictor[0]-a0*2*H*tau/mass[0]
            p_next=trial[0]-tau/(2*G)
        if trial[1]+GAP<0.:
            trial[1]=(mass[1]*predictor[1]-a0*KN*GAP)/(mass[1]+a0*KN)
        force=np.array([2*H*tau,KN*min(trial[1]+GAP,0.)])
        a=(trial-predictor)/a0
        assert np.linalg.norm(mass*a+force)<=1.e-9*max(1.,np.linalg.norm(force))
        alpha+=2*abs(p_next-p)/math.sqrt(3.); p=p_next
        q=trial; acc=a; v=vp+.5*dt*acc
        rows.append((q.copy(),v.copy(),acc.copy(),p,alpha,tau))
    return rows


def physical(values,modes):
    return np.asarray(values)@modes@np.linalg.inv(modes.T@modes)


def test_physical_mode_mass_and_qualified_q4_affine_work():
    model,material,element,modes,_=fixture()
    np.testing.assert_allclose(modes.T@assemble_mass_matrix(model)[0]@modes,
        np.diag([RHO*H/6.,RHO*H]),rtol=2e-13,atol=1e-14)
    for strain,tau in ((1.e-5,2*G*1.e-5),(.0003,YIELD/math.sqrt(3.))):
        pose=modes@np.array([strain,.0004])
        force,_,state=element.compute_nonlinear_response(model.mesh,material,pose,None,3,True)
        np.testing.assert_allclose(modes.T@force,[2*H*tau,0.],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(state['layer_strain'],np.tile([strain,-strain,0.],(12,1)),atol=1e-14)


@pytest.fixture(scope='module')
def complete_impact():
    return run()


def test_q4_plastic_impact_matches_independent_state_and_unloading(complete_impact):
    _,result,modes=complete_impact
    reference=scalar_reference(result.times)
    np.testing.assert_allclose(physical(result.displacements,modes),[r[0] for r in reference],rtol=1e-8,atol=1e-11)
    np.testing.assert_allclose(physical(result.velocities,modes),[r[1] for r in reference],rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(physical(result.accelerations,modes),[r[2] for r in reference],rtol=1e-7,atol=1e-5)
    assert len(result.material_state_history)==len(result.times)
    for history,row,coordinates in zip(result.material_state_history,reference,physical(result.displacements,modes)):
        state=history[1]
        np.testing.assert_allclose(state['plastic_strain'],np.tile([row[3],-row[3],0.],(12,1)),atol=1e-10)
        np.testing.assert_allclose(state['alpha'],row[4],atol=1e-10)
        # Compact accepted states carry plastic core, not recovered stresses.
        # Infer the deviatoric elastic stress from the independently mapped pose.
        actual_tau=2*G*(coordinates[0]-np.asarray(state['plastic_strain'])[:,0])
        np.testing.assert_allclose(actual_tau,row[5],atol=1e-7)
    alpha=np.array([r[4] for r in reference])
    assert np.max(alpha)<.005 and alpha[-1]>0.
    first_yield=np.flatnonzero(alpha>1e-12)[0]
    assert result.contact[first_yield].active_pressure_dofs>0
    assert reference[-1][5]<YIELD/math.sqrt(3.)*.9
    assert result.contact[-1].active_pressure_dofs==0
    assert result.cutbacks==0
    assert all(item.gap_tolerance_met for item in result.contact)


def test_yielded_active_restart_preserves_return_map_origin_and_settings(complete_impact):
    _,whole,modes=complete_impact
    _,prefix,_=run(end=.0009,emit=True)
    assert prefix.contact[-1].active_pressure_dofs>0
    assert np.min(prefix.element_states[1]['alpha'])>0.
    _,resumed,_=run(checkpoint=prefix.restart_checkpoint)
    packet=prefix.restart_checkpoint['state']['material_checkpoint']
    assert resumed.material_state_history[0][1]==packet['element_states'][0]['state']
    assert resumed.material_state_history[0][1]['qualified_q4_algorithmic_origin']==prefix.element_states[1]['qualified_q4_algorithmic_origin']
    for name,tol in (('displacements',1e-10),('velocities',1e-8),('accelerations',1e-5)):
        joined=np.concatenate([getattr(prefix,name)[:-1],getattr(resumed,name)])
        np.testing.assert_allclose(joined,getattr(whole,name),rtol=0.,atol=tol)
    for name in ('alpha','plastic_strain'):
        np.testing.assert_allclose(resumed.element_states[1][name],whole.element_states[1][name],atol=1e-9)
    with pytest.raises(ValueError,match='incompatible'):
        run(checkpoint=prefix.restart_checkpoint,controls=replace(CONTROLS,save_q4_material_history_dev=False))
    bad=copy.deepcopy(prefix.restart_checkpoint)
    bad['state']['material_checkpoint']['element_states'][0]['state']['alpha'][0]+=.1
    bad['checkpoint_sha256']=analysis_configuration_fingerprint({k:v for k,v in bad.items() if k!='checkpoint_sha256'})
    with pytest.raises(ValueError,match='integrity|fingerprint|hash'):
        run(checkpoint=bad)
    saved=copy.deepcopy(prefix.restart_checkpoint)
    prefix.element_states[1]['alpha'][0]+=.2
    prefix.material_state_history[-1][1]['plastic_strain'][0][0]+=.2
    assert prefix.restart_checkpoint==saved


@pytest.mark.parametrize('cancel',[False,True])
def test_failure_after_yield_leaves_model_and_contact_context_reusable(monkeypatch,cancel):
    from anyfesolver.control import CancellationToken,SolveCancelled
    model,_,_,modes,definition=fixture()
    token=CancellationToken()
    import anyfesolver.general_contact.q4_material_checkpoint as codec
    prepare=codec.prepare_q4_material_checkpoint
    observed=[]
    def fail_after_yield(model,full,states,**kwargs):
        packet=prepare(model,full,states,**kwargs)
        if np.min(states[1]['alpha'])>0.:
            observed.append(copy.deepcopy(packet))
            if cancel:
                token.cancel()
            else:
                raise ArithmeticError('owned yielded candidate failure')
        return packet
    def solve():
        return solve_transient_contact(model,definition,
            TransientConfig(DT,.001,initial_velocity=modes@np.array([.2,-1.])),
            solver_controls=CONTROLS,cancellation_token=token if cancel else None)
    with monkeypatch.context() as patch:
        patch.setattr(codec,'prepare_q4_material_checkpoint',fail_after_yield)
        with pytest.raises(SolveCancelled if cancel else ArithmeticError):
            solve()
    assert observed and active_contact_manager() is None
    reused=solve_transient_contact(model,definition,
        TransientConfig(DT,.001,initial_velocity=modes@np.array([.2,-1.])),solver_controls=CONTROLS)
    _,fresh,_=run(end=.001)
    np.testing.assert_array_equal(reused.displacements,fresh.displacements)
    assert reused.element_states==fresh.element_states


def test_cutback_from_yielded_parent_preserves_material_history(monkeypatch):
    import anyfesolver.general_contact.transient as driver
    import anyfesolver.general_contact.q4_material_checkpoint as codec
    prepare=codec.prepare_q4_material_checkpoint
    bounds=driver._minimum_hermite_gap
    yielded,rejected=False,False
    def observe(model,full,states,**kwargs):
        nonlocal yielded
        packet=prepare(model,full,states,**kwargs)
        yielded=yielded or bool(np.min(states[1]['alpha'])>0.)
        return packet
    def reject_one(*args,**kwargs):
        nonlocal rejected
        gap=bounds(*args,**kwargs)
        if yielded and not rejected:
            rejected=True
            return -1.e-4  # Controlled failed path gate, not altered mechanics.
        return gap
    with monkeypatch.context() as patch:
        patch.setattr(codec,'prepare_q4_material_checkpoint',observe)
        patch.setattr(driver,'_minimum_hermite_gap',reject_one)
        _,result,modes=run(end=.0013)
    assert rejected and result.cutbacks==1
    reference=scalar_reference(result.times)
    np.testing.assert_allclose(physical(result.displacements,modes),[r[0] for r in reference],atol=1e-11)
    for state,row in zip(result.material_state_history,reference):
        np.testing.assert_allclose(state[1]['alpha'],row[4],atol=1e-10)
        np.testing.assert_allclose(state[1]['plastic_strain'],np.tile([row[3],-row[3],0.],(12,1)),atol=1e-10)


@pytest.mark.parametrize('name,value',[('tool_work_absolute_tolerance_j',2e-10),
    ('tool_work_relative_tolerance',2e-8),('maximum_tool_work_subdivisions',65)])
def test_existing_moving_tool_restart_still_binds_work_controls(name,value):
    from types import SimpleNamespace
    from anyfesolver.general_contact.rigid_translation import ContactRigidTranslationHistory
    from anyfesolver.general_contact.transient import _restart_analysis_contract
    tool=ContactRigidTranslationHistory('plane',((0.,0.,0.,0.),(.1,0.,0.,.01)))
    manager=SimpleNamespace(restart_contract=lambda:{'fixture':'existing-plane'})
    transient=TransientConfig(.001,.1,initial_velocity=np.zeros(1))
    controls=ContactTransientSolverControls(allow_classical_beam_dev=True)
    def contract(selection):
        return _restart_analysis_contract(transient,selection,manager,'existing-beam',
            None,np.zeros(1),None,(tool,))
    original=contract(controls)
    assert name in original['controls']
    assert analysis_configuration_fingerprint(original)!=analysis_configuration_fingerprint(contract(replace(controls,**{name:value})))


def test_plastic_profile_requires_explicit_supported_material_and_controls():
    from anyfesolver.general_contact.transient import _admit
    from anyfesolver.general_contact.core import ContactSolverSettings
    from anyfesolver.material_curves import LinearHardeningCurve
    model,material,_,_,definition=fixture()
    with pytest.raises(ValueError,match='only isotropic elastic'):
        _admit(model,definition,ContactSolverSettings(),ContactTransientSolverControls())
    material.hardening_curve=LinearHardeningCurve(sigma_yield=20.,hardening_modulus_value=0.)
    with pytest.raises(ValueError,match='isotropic DNV'):
        _admit(model,definition,ContactSolverSettings(),CONTROLS)
    with pytest.raises(ValueError,match='requires the Q4 plastic'):
        ContactTransientSolverControls(save_q4_material_history_dev=True)
    with pytest.raises(ValueError,match='separate fixed-plane'):
        ContactTransientSolverControls(allow_q4_plastic_dev=True,allow_classical_beam_dev=True)
