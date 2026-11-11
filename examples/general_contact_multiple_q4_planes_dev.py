"""Three guided physical Q4 patches coupled by MPCs, impacting two fixed planes.

Explicit development profile; no engineering acceptance. The third patch has
normal displacement (q1+q2)/2, so its actual inertia and contact couple both modes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSurface, ContactTransientSolverControls, FEModel, RigidContactBody,
    TransientConfig, create_shell_element, solve_transient_contact)

H, RHO, E, NU = .05, 1., 1.e5, .3
PATCH_MODES=np.array([[1.,0.],[0.,1.],[.5,.5]])
MASS=RHO*H*PATCH_MODES.T@PATCH_MODES
GAPS=np.array([.01,.008,.01])
STIFFNESS=np.array([1.e5,2.e5,1.e5])
CONTROLS=ContactTransientSolverControls(allow_multiple_shell_planes_dev=True,
    q4_plane_event_subdivision_dev=True,save_constraint_reactions_dev=True,
    equilibrium_tolerance=1.e-10)


def build_model():
    model=FEModel('three-coupled-q4-two-planes')
    model.add_material('elastic',E,NU,density=RHO)
    modes=np.zeros((72,2))
    for patch in range(3):
        nodes=[]
        for corner,(x,y) in enumerate(((0.,0.),(1.,0.),(1.,1.),(0.,1.))):
            node=patch*4+corner+1
            model.add_node(node,x+2*patch,y,0.)
            nodes.append(node)
        model.add_element(patch+1,create_shell_element(patch+1,nodes,'elastic',thickness=H))
    model.add_boundary_condition(BoundaryCondition('in-plane and rotations',list(range(1,13)),
        dict(ux=0.,uy=0.,rx=0.,ry=0.,rz=0.)))
    masters=[model.mesh.dof_manager.get_node_dofs(node)[2] for node in (1,5)]
    for patch,coefficients in enumerate(PATCH_MODES):
        for node in range(4*patch+1,4*patch+5):
            dof=model.mesh.dof_manager.get_node_dofs(node)[2]
            modes[dof]=coefficients
            if dof not in masters:
                terms=((dof,1.),)+tuple((master,-float(value))
                    for master,value in zip(masters,coefficients) if value)
                model.add_constraint_equation(terms=terms)
    definition=ContactDefinition(
        surfaces=(ContactSurface('outer',(1,3),side='bottom'),
                  ContactSurface('inner',(2,),side='bottom')),
        rigid_bodies=(RigidContactBody('floor_a','plane',(0.,0.,-.035),(0.,0.,1.)),
                      RigidContactBody('floor_b','plane',(0.,0.,-.033),(0.,0.,1.))),
        interactions=(ContactInteraction('first','outer','floor_a',1.e5,.0015),
                      ContactInteraction('second','inner','floor_b',2.e5,.0009)))
    return model,definition,modes


def solve(*,dt=.0005,end=.04,checkpoint=None,emit=False):
    model,definition,modes=build_model()
    result=solve_transient_contact(model,definition,
        TransientConfig(dt,end,initial_velocity=modes@np.array([-1.,-.8])),
        solver_controls=CONTROLS,restart_checkpoint=checkpoint,emit_restart_checkpoint=emit)
    return model,result,modes


def modal(values,modes):
    return np.asarray(values)@modes@np.linalg.inv(modes.T@modes)


def run(*,dt=.0005,end=.04):
    _,result,modes=solve(dt=dt,end=end)
    q,v=modal(result.displacements,modes),modal(result.velocities,modes)
    kinetic=.5*np.einsum('ij,jk,ik->i',v,MASS,v)
    energy=kinetic+np.array([row.penalty_energy for row in result.contact])
    return dict(status='development; not engineering acceptance',profile=result.profile,
        times_s=result.times.tolist(),modal_positions_m=q.tolist(),modal_velocities_m_s=v.tolist(),
        maximum_energy_relative_defect=float(np.max(np.abs(energy/energy[0]-1.))),
        cutbacks=result.cutbacks,cutback_diagnostics=result.cutback_diagnostics,
        events_s=result.localized_contact_event_times,
        endpoint_requests=result.event_endpoint_evaluations,stage_evaluations=result.plane_event_stage_evaluations,
        pair_swept_minimum_gaps_m=result.pair_swept_minimum_gaps,
        pair_results=[row.to_dict() for row in result.contact])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt',type=float,default=.0005)
    parser.add_argument('--end',type=float,default=.04)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=json.dumps(run(dt=args.dt,end=args.end),indent=2,allow_nan=False)
    if args.output:args.output.write_text(report+'\n',encoding='utf-8')
    print(report)
