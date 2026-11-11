"""Development example: explicit round B2/B3 impact against a fixed plane."""
import argparse

import numpy as np

from anyfesolver import (
    BoundaryCondition,ContactDefinition,ContactInteraction,ContactSolverSettings,
    ContactSurface,RigidContactBody,
)
from anyfesolver.dynamics import TransientConfig
from anyfesolver.elements import BeamElement,QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.transient import ContactRigidTranslationHistory,ContactTransientSolverControls,solve_transient_contact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quadratic',action='store_true',help='Use a straight-reference B3 beam')
    parser.add_argument('--deformable',action='store_true',help='Excite nonuniform bending and retain free ry inertia')
    parser.add_argument('--multiple-planes',action='store_true',help='Impact orthogonal fixed planes with separate line laws')
    parser.add_argument('--tool-motion',action='store_true',help='Prescribe constant normal plane translation and report input work')
    args = parser.parse_args()
    if args.deformable and not args.quadratic:
        parser.error('--deformable example requires --quadratic')
    if args.deformable and args.multiple_planes:
        parser.error('--deformable and --multiple-planes are separate examples')
    model = FEModel('round_beam_plane_impact_dev')
    model.add_material('elastic',1e3,.3,density=1.)
    positions = (-.5,0.,.5) if args.quadratic else (-.5,.5)
    for node,x in enumerate(positions,1):
        model.add_node(node,x,.015 if args.multiple_planes else 0.,.015)
    nodes = list(model.mesh.nodes)
    section = dict(area=1e-3,Iy=1e-4,Iz=1e-4,J=1e-4)
    model.add_element(1,(QuadraticBeamElement if args.quadratic else BeamElement)(
        1,nodes,'elastic',section))
    fixed = ('ux','uy','rx','rz') if args.deformable else ('ux','uy','rx','ry','rz')
    if args.multiple_planes:
        fixed = ('ux','rx','ry','rz')
    model.add_boundary_condition(BoundaryCondition('guide',nodes,{name:0. for name in fixed}))
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    speeds = (-.1,-.35,-.1) if args.deformable else (-.5,)*len(nodes)
    for node,speed in zip(nodes,speeds):
        initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = speed
        if args.multiple_planes:
            initial[model.mesh.dof_manager.get_node_dofs(node)[1]] = -.5
            initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.25
    definition = ContactDefinition(
        surfaces=(ContactSurface('round_line',(1,),beam_radius=.01,beam_end_caps='none'),),
        rigid_bodies=(RigidContactBody('plane','plane',(0.,0.,0.),normal=(0.,0.,1.)),),
        interactions=(ContactInteraction('line_plane','round_line','plane',None,.002,
                                         line_stiffness=1e4),),
    )
    if args.multiple_planes:
        definition = ContactDefinition(surfaces=definition.surfaces,
            rigid_bodies=definition.rigid_bodies+(RigidContactBody('y_plane','plane',(0.,0.,0.),normal=(0.,1.,0.)),),
            interactions=(ContactInteraction('line_plane','round_line','plane',None,.002,line_stiffness=7e3),
                          ContactInteraction('y_pair','round_line','y_plane',None,.002,line_stiffness=3e3)))
    end_time = .023 if args.deformable or args.multiple_planes else .013
    motions = tuple(ContactRigidTranslationHistory(body.name,
        ((0.,0.,0.,0.), (end_time,*(float(.1*end_time*x) for x in body.normal))))
        for body in definition.rigid_bodies) if args.tool_motion else ()
    result = solve_transient_contact(model,definition,
        TransientConfig(.0001,end_time,initial_velocity=initial),
        contact_settings=ContactSolverSettings(pair_motion='general'),
        solver_controls=ContactTransientSolverControls(allow_classical_beam_dev=True,
            allow_multiple_beam_planes_dev=args.multiple_planes), rigid_translation_histories=motions)
    tool_summary = (dict(prescribed_tool_work_j=result.prescribed_tool_work_j[-1],
                        prescribed_tool_work_error_j=result.prescribed_tool_work_error_j[-1])
                    if args.tool_motion else {})
    if args.multiple_planes:
        print(dict(profile=result.profile,accepted_steps=len(result.newton_iterations),
            cutbacks=result.cutbacks,minimum_swept_gap_m=float(np.min(result.swept_minimum_gaps)),
            maximum_residual_n=float(np.max(result.equilibrium_residuals)),
            maximum_pair_forces_n={name:float(max(np.linalg.norm(dict(c.pairs)[name].beam_force) for c in result.contact))
                                  for name,_ in result.contact[-1].pairs},
            final_pair_active={name:c.active_line_cells for name,c in result.contact[-1].pairs}, **tool_summary))
        return
    print(dict(profile=result.profile,accepted_steps=len(result.newton_iterations),
        cutbacks=result.cutbacks,minimum_swept_gap_m=float(np.min(result.swept_minimum_gaps)),
        maximum_residual_n=float(np.max(result.equilibrium_residuals)),
        maximum_line_force_n=max(c.beam_force[2] for c in result.contact),
        partial_line_contact=any(0. < c.active_line_length < .99 for c in result.contact),
        final_line_active=result.contact[-1].active_line_cells, **tool_summary))


if __name__ == '__main__':
    main()
