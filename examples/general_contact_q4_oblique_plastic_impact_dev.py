"""Manufactured oblique-plane impact excites initially quiescent plastic membrane.

Explicit development profile, not an engineering material or impact qualification.
Small membrane work requires modal checks; total-energy agreement is insufficient.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from anyfesolver import (BoundaryCondition,ContactDefinition,ContactInteraction,
    ContactSurface,DNVC208MaterialCurve,FEModel,RigidContactBody,create_shell_element)
from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.transient import ContactTransientSolverControls,solve_transient_contact

E,NU,H,RHO=1.e9,.3,.05,1.e4
YIELD,KN,GAP,ANGLE=.2,4.e11,.0008,2.e-5


def build_model():
    model=FEModel('manufactured-oblique-plastic-q4-impact')
    curve=DNVC208MaterialCurve(sigma_prop=YIELD,sigma_yield=YIELD,
        sigma_yield_2=YIELD,eps_p_y1=.005,eps_p_y2=.01,K=YIELD,n=.2)
    material=model.add_material('plastic',E,NU,density=RHO,
        yield_stress=YIELD,hardening_curve=curve)
    for node,(x,y) in enumerate(((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)),1):
        model.add_node(node,x,y,0.)
    element=create_shell_element(1,[1,2,3,4],'plastic',thickness=H)
    model.add_element(1,element)
    model.add_boundary_condition(BoundaryCondition('zero rotations',[1,2,3,4],
        dict(rx=0.,ry=0.,rz=0.)))
    modes=np.zeros((model.mesh.dof_manager.total_dofs,2))
    master_e=model.mesh.dof_manager.get_node_dofs(2)[0]
    master_w=model.mesh.dof_manager.get_node_dofs(1)[2]
    for node in (1,2,3,4):
        x,y=model.mesh.nodes[node].x,model.mesh.nodes[node].y
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        modes[dofs[0],0],modes[dofs[1],0],modes[dofs[2],1]=x,-y,1.
        for dof,master,factor in ((dofs[0],master_e,2*x),
                (dofs[1],master_e,-2*y),(dofs[2],master_w,1.)):
            if dof!=master:
                model.add_constraint_equation(terms=((dof,1.),(master,-factor)))
    definition=ContactDefinition(surfaces=(ContactSurface('skin',(1,)),),
        rigid_bodies=(RigidContactBody('floor','plane',(0.,0.,-GAP),
            (float(np.sin(ANGLE)),0.,float(np.cos(ANGLE)))),),
        interactions=(ContactInteraction('support','skin','floor',KN,5.e-5),))
    return model,material,element,modes,definition


def solve(*,dt=3.e-5,end=.003,checkpoint=None,emit=False):
    model,_,_,modes,definition=build_model()
    result=solve_transient_contact(model,definition,
        TransientConfig(dt,end,initial_velocity=modes@np.array([0.,-1.])),
        solver_controls=ContactTransientSolverControls(allow_q4_plastic_dev=True,
            save_q4_material_history_dev=True,q4_plane_event_subdivision_dev=True,
            equilibrium_tolerance=1.e-13),restart_checkpoint=checkpoint,
        emit_restart_checkpoint=emit)
    return model,result,modes


def modal(values,modes):
    return np.asarray(values)@modes@np.linalg.inv(modes.T@modes)


def run(*,dt=3.e-5,end=.003):
    _,result,modes=solve(dt=dt,end=end)
    q,v=modal(result.displacements,modes),modal(result.velocities,modes)
    p=np.array([state[1]['plastic_strain'][0][0] for state in result.material_state_history])
    alpha=np.array([state[1]['alpha'][0] for state in result.material_state_history])
    if np.max(alpha)>=.005:
        raise ValueError('manufactured plateau diagnostic requires alpha below .005')
    tau=2*(E/(2*(1+NU)))*(q[:,0]-p)
    kinetic=.5*(RHO*H/6)*v[:,0]**2+.5*(RHO*H)*v[:,1]**2
    elastic=H*tau*tau/(2*(E/(2*(1+NU))))
    plastic=H*YIELD*alpha
    energy=kinetic+elastic+plastic+np.array([row.penalty_energy for row in result.contact])
    first=np.flatnonzero(alpha>1.e-15)
    imports={}
    for name,module in tuple(sys.modules.items()):
        value=getattr(module,'__file__',None)
        if (name=='anyfesolver' or name.startswith('anyfesolver.')) and value:
            path=Path(value).resolve()
            imports[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(status='manufactured development; not engineering acceptance',
        profile=result.profile,dt_s=dt,end_s=float(result.times[-1]),
        accepted_steps=len(result.newton_iterations),cutbacks=result.cutbacks,
        cutback_diagnostics=result.cutback_diagnostics,
        localized_events_s=list(result.localized_contact_event_times),
        endpoint_requests=result.event_endpoint_evaluations,
        stage_evaluations=result.plane_event_stage_evaluations,
        initial_membrane_coordinate=float(q[0,0]),initial_membrane_velocity=float(v[0,0]),
        final_membrane_coordinate=float(q[-1,0]),final_membrane_velocity=float(v[-1,0]),
        maximum_alpha=float(np.max(alpha)),final_plastic_xx=float(p[-1]),
        first_yield_time_s=float(result.times[first[0]]) if len(first) else None,
        first_yield_contact_active=bool(result.contact[first[0]].active_pressure_dofs) if len(first) else False,
        minimum_physical_path_gap_m=float(np.min(result.swept_minimum_gaps)),
        all_endpoint_gap_checks_pass=all(row.gap_tolerance_met for row in result.contact),
        total_energy_diagnostic=dict(initial_j=float(energy[0]),
            maximum_relative_defect=float(np.max(np.abs(energy/energy[0]-1.))),
            limitation='normal impact dominates; does not verify tiny membrane/plastic work'),
        imported_sources=imports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt',type=float,default=3.e-5)
    parser.add_argument('--end',type=float,default=.003)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=json.dumps(run(dt=args.dt,end=args.end),indent=2,allow_nan=False)
    if args.output:args.output.write_text(report+'\n',encoding='utf-8')
    print(report)
