"""Guided elastic shell shear between opposing physical top/bottom skins.

Development example, not engineering acceptance. Constant manufactured MPCs
retain uniform translation and affine normal slope, with in-plane/rotation
coordinates constrained. The shell genuinely strains in shear and membrane.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from anyfesolver import (BoundaryCondition,ContactDefinition,ContactInteraction,
    ContactSurface,ContactTransientSolverControls,FEModel,RigidContactBody,
    TransientConfig,create_shell_element,solve_transient_contact)

H,RHO,E,NU=.05,1.,1.e5,.3
GAP,KN,GAP_TOLERANCE=1.e-5,1.e9,5.e-6
MASS=np.diag([RHO*H,RHO*H/12.])
SHEAR_STIFFNESS=(5./6.)*E/(2*(1+NU))*H
MEMBRANE_QUARTIC=E*H/(8*(1-NU*NU))
INITIAL_VELOCITY=np.array([.002,.05])
CONTROLS=ContactTransientSolverControls(allow_multiple_shell_planes_dev=True,
    q4_plane_event_subdivision_dev=True,batch_plane_event_fronts_dev=True,
    save_constraint_reactions_dev=True,
    equilibrium_tolerance=1.e-11)


def build_model(*,cells=4):
    if type(cells) is not int or cells<1:
        raise ValueError('positive integral x subdivisions required')
    model=FEModel('deforming-q4-opposing-planes')
    material=model.add_material('elastic',E,NU,density=RHO)
    for j,y in enumerate((-.5,.5)):
        for i,x in enumerate(np.linspace(-.5,.5,cells+1)):
            model.add_node(j*(cells+1)+i+1,float(x),y,0.)
    for i in range(cells):
        nodes=[i+1,i+2,cells+i+3,cells+i+2]
        model.add_element(i+1,create_shell_element(i+1,nodes,'elastic',thickness=H))
    model.add_boundary_condition(BoundaryCondition('guided shear',list(model.mesh.nodes),
        dict(ux=0.,uy=0.,rx=0.,ry=0.,rz=0.)))
    modes=np.zeros((6*len(model.mesh.nodes),2))
    left=model.mesh.dof_manager.get_node_dofs(1)[2]
    right=model.mesh.dof_manager.get_node_dofs(cells+1)[2]
    for node in model.mesh.nodes.values():
        dof=model.mesh.dof_manager.get_node_dofs(node.id)[2]
        x=float(node.x)
        modes[dof]=[1.,x]
        if dof not in (left,right):
            model.add_constraint_equation(terms=((dof,1.),(left,-(.5-x)),(right,-(.5+x))))
    ids=tuple(model.mesh.elements)
    definition=ContactDefinition(
        surfaces=(ContactSurface('lower_skin',ids,side='bottom'),
                  ContactSurface('upper_skin',ids,side='top')),
        rigid_bodies=(RigidContactBody('lower_plane','plane',(0.,0.,-H/2-GAP),(0.,0.,1.)),
                      RigidContactBody('upper_plane','plane',(0.,0.,H/2+GAP),(0.,0.,-1.))),
        interactions=(ContactInteraction('lower','lower_skin','lower_plane',KN,GAP_TOLERANCE),
                      ContactInteraction('upper','upper_skin','upper_plane',KN,GAP_TOLERANCE)))
    return model,definition,modes


def modal(values,modes):
    return np.asarray(values)@modes@np.linalg.inv(modes.T@modes)


def solve(*,dt=1.e-5,end=.0018,checkpoint=None,emit=False):
    model,definition,modes=build_model()
    result=solve_transient_contact(model,definition,
        TransientConfig(dt,end,initial_velocity=modes@INITIAL_VELOCITY),
        solver_controls=CONTROLS,restart_checkpoint=checkpoint,emit_restart_checkpoint=emit)
    return model,result,modes


def run(*,dt=1.e-5,end=.0018):
    _,result,modes=solve(dt=dt,end=end)
    q,v=modal(result.displacements,modes),modal(result.velocities,modes)
    internal=.5*SHEAR_STIFFNESS*q[:,1]**2+MEMBRANE_QUARTIC*q[:,1]**4
    energy=.5*np.einsum('ij,jk,ik->i',v,MASS,v)+internal+np.array([r.penalty_energy for r in result.contact])
    return dict(status='development; not engineering acceptance',profile=result.profile,
        times_s=result.times.tolist(),modal_positions_m_and_slope=q.tolist(),
        modal_velocities_m_s_and_slope_s=v.tolist(),
        maximum_energy_relative_defect=float(np.max(np.abs(energy/energy[0]-1.))),
        maximum_shear_energy_j=float(np.max(.5*SHEAR_STIFFNESS*q[:,1]**2)),
        maximum_membrane_energy_j=float(np.max(MEMBRANE_QUARTIC*q[:,1]**4)),
        maximum_allowed_mode_reaction=float(np.max(np.abs(result.constraint_reactions@modes))),
        cutbacks=result.cutbacks,cutback_diagnostics=result.cutback_diagnostics,
        events_s=result.localized_contact_event_times,
        endpoint_requests=result.event_endpoint_evaluations,stage_evaluations=result.plane_event_stage_evaluations,
        pair_swept_minimum_gaps_m=result.pair_swept_minimum_gaps,
        pair_results=[r.to_dict() for r in result.contact])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt',type=float,default=1.e-5)
    parser.add_argument('--end',type=float,default=.0018)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=json.dumps(run(dt=args.dt,end=args.end),indent=2,allow_nan=False)
    if args.output:args.output.write_text(report+'\n',encoding='utf-8')
    print(report)
