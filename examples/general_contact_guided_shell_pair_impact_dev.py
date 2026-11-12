"""Finite-mass Q4/S3 contact recoil using one coupled Newmark stage.

Uniform guided translations give an independent two-body oscillator reference.
The target uses two larger S3 faces; source is a physical Q4 bottom skin.
Explicit development profile, not general shell-impact engineering acceptance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from anyfesolver import (BoundaryCondition,ContactDefinition,ContactInteraction,
    ContactSurface,ContactTransientSolverControls,FEModel,TransientConfig,
    create_shell_element,solve_transient_contact)

H=.02
GAP=.0002
KN=3.e8
MASSES=np.array([20.,14.4])
INITIAL_VELOCITY=np.array([-.1,0.])
CONTROLS=ContactTransientSolverControls(allow_guided_shell_pair_dev=True,
    q4_plane_event_subdivision_dev=True,batch_plane_event_fronts_dev=True,
    save_constraint_reactions_dev=True,equilibrium_tolerance=1.e-11)


def build_model(*,target_kind='s3'):
    model=FEModel('finite-shell-pair-recoil')
    model.add_material('source',1.e7,.3,density=1000.)
    model.add_material('target',1.e7,.3,density=500.)
    source_positions=((0.,0.,H+GAP),(1.,0.,H+GAP),(1.,1.,H+GAP),(0.,1.,H+GAP))
    target_positions=((-0.1,-0.1,0.),(1.1,-0.1,0.),(1.1,1.1,0.),(-0.1,1.1,0.))
    for node,point in enumerate((*source_positions,*target_positions),1):model.add_node(node,*point)
    model.add_element(1,create_shell_element(1,[1,2,3,4],'source',thickness=H))
    if target_kind=='s3':
        for eid,nodes in ((2,[5,6,7]),(3,[5,7,8])):
            model.add_element(eid,create_shell_element(eid,nodes,'target',thickness=H,
                formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.)))
        target_elements=(2,3)
    elif target_kind=='q4':
        model.add_element(2,create_shell_element(2,[5,6,7,8],'target',thickness=H))
        target_elements=(2,)
    else:raise ValueError('target_kind must be q4 or s3')
    model.add_boundary_condition(BoundaryCondition('guides',list(range(1,9)),dict(ux=0.,uy=0.,rx=0.,ry=0.,rz=0.)))
    modes=np.zeros((48,2))
    for column,nodes in enumerate((range(1,5),range(5,9))):
        master=model.mesh.dof_manager.get_node_dofs(nodes.start)[2]
        for node in nodes:
            dof=model.mesh.dof_manager.get_node_dofs(node)[2];modes[dof,column]=1.
            if dof!=master:model.add_constraint_equation(terms=((dof,1.),(master,-1.)))
    definition=ContactDefinition(surfaces=(ContactSurface('source_skin',(1,),side='bottom'),
        ContactSurface('target_skin',target_elements,side='top')),
        rigid_bodies=(),
        interactions=(ContactInteraction('shell_pair','source_skin','target_skin',KN,2.e-5),))
    return model,definition,modes


def modal(values,modes):return np.asarray(values)@modes@np.linalg.inv(modes.T@modes)


def solve(*,dt=5.e-5,end=.005,target_kind='s3',checkpoint=None,emit=False,controls=CONTROLS,cancellation_token=None):
    model,definition,modes=build_model(target_kind=target_kind)
    result=solve_transient_contact(model,definition,
        TransientConfig(dt,end,initial_velocity=modes@INITIAL_VELOCITY),
        solver_controls=controls,restart_checkpoint=checkpoint,emit_restart_checkpoint=emit,
        cancellation_token=cancellation_token)
    return model,result,modes


def run(*,dt=5.e-5,end=.005,target_kind='s3'):
    _,result,modes=solve(dt=dt,end=end,target_kind=target_kind)
    q,v=modal(result.displacements,modes),modal(result.velocities,modes)
    momentum=v@MASSES
    energy=.5*(v*v)@MASSES+np.array([r.penalty_energy for r in result.contact])
    return dict(status='development; not engineering acceptance',profile=result.profile,
        target_kind=target_kind,times_s=result.times.tolist(),q_m=q.tolist(),v_m_s=v.tolist(),
        maximum_momentum_error_kg_m_s=float(np.max(np.abs(momentum-momentum[0]))),
        maximum_energy_relative_defect=float(np.max(np.abs(energy/energy[0]-1.))),
        maximum_allowed_mode_reaction_n=float(np.max(np.abs(result.constraint_reactions@modes))),
        contact=[r.to_dict() for r in result.contact],swept_minimum_gaps_m=result.swept_minimum_gaps.tolist(),
        cutbacks=result.cutbacks,cutback_diagnostics=result.cutback_diagnostics,
        events_s=result.localized_contact_event_times,
        endpoint_requests=result.event_endpoint_evaluations,stage_evaluations=result.plane_event_stage_evaluations)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt',type=float,default=5.e-5)
    parser.add_argument('--end',type=float,default=.005)
    parser.add_argument('--target-kind',choices=('q4','s3'),default='s3')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=json.dumps(run(dt=args.dt,end=args.end,target_kind=args.target_kind),indent=2,allow_nan=False)
    if args.output:args.output.write_text(report+'\n',encoding='utf-8')
    print(report)
