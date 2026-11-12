"""Development-only partial S3 press/slide/reverse/open/recontact workflow.

Run against an installed ANYfeSolver package. The target has free in-plane degrees
of freedom; source normal deformation is solved, not tied to the controlled node.
The default five-increment run can take several minutes. This is not C5 acceptance.
Use --max-load-factor 0.2 for the first press and --checkpoint-output to save it;
use --restart with that file to continue the same complete path.
"""
import argparse
import json
from pathlib import Path

from anyfesolver import BoundaryCondition, create_shell_element, solve_static_nonlinear
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact import (
    ContactDefinition, ContactInteraction, ContactSurface, ContactSolverSettings,
    ContactPathControl, ContactPathProgram,
)
from anyfesolver.general_contact.s3_sliding_friction import S3SlidingFrictionSettings
from anyfesolver.general_contact.s3_friction_replay import S3ReplayFrictionSettings


def run_example(*, max_load_factor=1.0, restart_checkpoint=None):
    model=FEModel('Partial S3 material-replay friction development')
    model.add_material('elastic',1e5,.3,density=1)
    positions=((.2,.2,.065),(.6,.2,.065),(.2,.6,.065),
               (0,0,0),(1,0,0),(0,1,0),(1,1,0))
    for node,position in enumerate(positions,1):model.add_node(node,*position)
    for element,nodes in ((1,(1,2,3)),(2,(4,5,6)),(3,(5,7,6))):
        model.add_element(element,create_shell_element(element,nodes,'elastic',
            thickness=.05,formulation='e4-pl-s3-v2d',reference_normal=(0,0,1),
            reference_surface_offset=.01))
    model.add_boundary_condition(BoundaryCondition('target_plane',(4,5,6,7),
        {d:0 for d in ('uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('target_origin',(4,),{'ux':0,'uy':0}))
    model.add_boundary_condition(BoundaryCondition('target_orientation',(5,),{'uy':0}))
    model.add_boundary_condition(BoundaryCondition('source_guide',(1,2,3),
        {d:0 for d in ('uy','rx','ry','rz')}))
    contact=ContactDefinition((ContactSurface('source',(1,),'bottom'),
        ContactSurface('target',(2,3),'top')),(),
        (ContactInteraction('normal','source','target',1000.,.02,
                            normal_law='projected_penalty'),))
    normal=ContactSolverSettings(pair_motion='general',normal_discretization='primal_mortar',
        general_s3_difference_step=2e-5,general_s3_differentiation='jet2_s3_dev')
    material=S3SlidingFrictionSettings(.3,1000.,(2e-7,)*model.mesh.dof_manager.total_dofs,
        maximum_candidate_evaluations=145,activation_history='linear_constant_normal_touch_dev')
    friction=S3ReplayFrictionSettings(material,1,1e-5,1e-5,1e-6,event_subdivisions=4,
        prepared_pressure_paths=True,prepared_partial_pressure_paths=True)
    path=ContactPathProgram((ContactPathControl.node(1,'ux'),ContactPathControl.node(1,'uz')),
        ((0,0),(0,-.02),(.55,-.02),(.54,-.02),(.54,0),(.54,-.02)),1)
    result=solve_static_nonlinear(model,contact=contact,contact_settings=normal,
        contact_friction=friction,contact_path_program=path,max_iterations=18,tolerance=1e-8,
        max_load_factor=max_load_factor,restart_checkpoint=restart_checkpoint,
        emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.info)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-load-factor',type=float,default=1.)
    parser.add_argument('--restart',type=Path)
    parser.add_argument('--checkpoint-output',type=Path)
    args=parser.parse_args()
    restart=json.loads(args.restart.read_text(encoding='utf-8')) if args.restart else None
    result=run_example(max_load_factor=args.max_load_factor,restart_checkpoint=restart)
    checkpoint=result.to_restart_checkpoint()
    if args.checkpoint_output:
        args.checkpoint_output.write_text(json.dumps(checkpoint,indent=2),encoding='utf-8')
    rows=result.info['force_displacement_history']
    print(json.dumps(dict(status=result.status,scope='development; not C5 acceptance',
        accepted_steps=checkpoint['path_state']['accepted_step'],
        profile=rows[-1]['friction']['profile'],
        reactions_n=[row['reaction_generalized'] for row in rows],
        cumulative_dissipation_j=rows[-1]['friction']['cumulative_dissipation_j'],
        storage_j=rows[-1]['friction']['storage_j']),indent=2))


if __name__=='__main__':main()
