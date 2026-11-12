"""Connected nonmatching S3 hard-normal friction with a deformable affine target.

Source in-plane translations are tied; source normal coordinates are independent.
Target ux/uz follow declared parallelogram MPCs. The first load is manufactured using disclosed structural K and independent
planar P1 skin forces. Subsequent tangential/open/recontact steps solve the same
coupled equations. This is development code verification, not qualification.
"""
import argparse,json
from pathlib import Path
import numpy as np
from anyfesolver import FEModel,BoundaryCondition,create_shell_element,solve_static_nonlinear
from anyfesolver.general_contact import (ContactDefinition,ContactSurface,ContactInteraction,
    ContactSolverSettings,ContactPathControl,ContactPathProgram,ContactPathNodalForce,
    S3MeshMovingNormalFrictionSettings)

TILT=1e-6
ZTOTAL=.01


def build_case():
    model=FEModel('s3-mesh-moving-normal-hard-friction')
    model.add_material('elastic',1e5,.3,density=1.)
    points=((.15,.19,.0400002),(.25,.29,.0400002),(.35,.29,.0400002),(.35,.39,.0400002),(.25,.39,.0400002),
            (-.1,-.1,0.),(0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.))
    for node,p in enumerate(points,1):model.add_node(node,*p)
    faces=((1,2,5),(2,3,5),(3,4,5),(6,7,10),(7,8,10),(8,9,10))
    for eid,nodes in enumerate(faces,1):
        model.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.05,
            formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=.01 if eid>3 else 0.))
    model.add_boundary_condition(BoundaryCondition('source-real-normal-clamp',(1,),{'uz':0.}))
    model.add_boundary_condition(BoundaryCondition('target-real-clamp',(6,),{d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('fixed-angular-profile',tuple(range(1,11)),{d:0. for d in ('rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('target-inplane-guide',(7,8,9,10),{'uy':0.}))
    for axis in (0,1):
        master=model.mesh.dof_manager.get_node_dofs(2)[axis]
        for node in (1,3,4,5):
            dep=model.mesh.dof_manager.get_node_dofs(node)[axis]
            model.add_constraint_equation(terms=((dep,1.),(master,-1.)),dependent_dof=dep,source_id=f'source-rigid-translation-{axis}-{node}')
    for axis in (0,2):
        dofs={n:model.mesh.dof_manager.get_node_dofs(n)[axis] for n in (7,8,9,10)}
        model.add_constraint_equation(terms=((dofs[9],1.),(dofs[8],-1.),(dofs[10],-1.),(dofs[7],1.)),
            dependent_dof=dofs[9],source_id=f'target-affine-parallelogram-{axis}')
    definition=ContactDefinition((ContactSurface('source',(2,3),'bottom'),ContactSurface('target',(5,6),'top')),(),
        (ContactInteraction('hard-material','source','target',1000.,5e-5,normal_law='augmented'),))
    settings=ContactSolverSettings(pair_motion='general',normal_discretization='primal_mortar',
        general_s3_differentiation='jet2_s3_dev',general_s3_difference_step=2e-5,
        q4_source_gap_certificate='whole_face',augmentation_support_resolution='exact',
        shell_pair_orientation='physical_sides_dev',enforcement_solver='monolithic_active_set',
        complementarity_force_scale_n=.01,pressure_tolerance=1e-8)
    friction=S3MeshMovingNormalFrictionSettings(.3,1000.,(2e-8,)*60,201,
                                           pressure_step_n=1e-8,closed_point_gap_tolerance_m=1e-10)
    # Manufactured code-verification load, independent of production contact.
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    K=assemble_stiffness_matrix(model)[0]
    assert K.shape==(60,60)
    pose=np.zeros(60);opened=np.zeros(60)
    for node in (2,3,4,5):
        dof=model.mesh.dof_manager.get_node_dofs(node)[2]
        pose[dof]=-2e-7-TILT*points[node-1][0];opened[dof]=2e-6
    for node in (7,8,9,10):pose[model.mesh.dof_manager.get_node_dofs(node)[2]]=-TILT*points[node-1][0]
    normal=np.array([TILT,0.,1.]);normal/=np.linalg.norm(normal)
    force=K@pose;openforce=K@opened
    source_nodes=(2,3,4,5);target_nodes=(7,8,9,10)
    support=np.array([.01/6,.01/3,.01/6,.01/3])
    source_pressure=ZTOTAL/.01
    # Independent uniform reference-area pressure on two source P1 faces;
    # the initial rectangle is wholly within target facet (7,8,10).
    target_bary=np.array([.36,.30,0.,.34])
    for node,area in zip(source_nodes,support):force[np.asarray(model.mesh.dof_manager.get_node_dofs(node)[:3])]-=source_pressure*area*normal
    for node,weight in zip(target_nodes,target_bary):force[np.asarray(model.mesh.dof_manager.get_node_dofs(node)[:3])]+=ZTOTAL*weight*normal
    waypoints=((0.,0.),(0.,0.),(1e-8,0.),(.39,0.),(.3899995,0.),(.389998,0.),(.389998,0.))
    loads=[]
    for node in (*source_nodes,*target_nodes):
        for axis,name in enumerate(('ux','uy','uz')):
            j=model.mesh.dof_manager.get_node_dofs(node)[axis]
            loads.append(ContactPathNodalForce(node,name,(0.,float(force[j]),float(force[j]),
                float(force[j]),float(force[j]),float(openforce[j]),float(force[j]))))
    program=ContactPathProgram((ContactPathControl.node(2,'ux'),ContactPathControl.node(2,'uy')),
                               waypoints,1,tuple(loads))
    return model,definition,settings,friction,program,dict(manufactured_pose=pose,
        open_reference_pose=opened,manufactured_force=force,open_force=openforce,
        supported_force=source_pressure*support,target_barycentric=target_bary,
        structural_operator_reused=True,contact_load_independent=True)


def solve_case(case,through_step=6,restart=None):
    model,definition,settings,friction,program,_=case
    return solve_static_nonlinear(model,contact=definition,contact_settings=settings,
        contact_friction=friction,contact_path_program=program,max_iterations=18,tolerance=1e-12,
        max_load_factor=through_step/6,emit_restart_checkpoint=True,
        record_increment_snapshots=True,restart_checkpoint=restart)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through-step',type=int,default=6,choices=range(1,7))
    parser.add_argument('--restart',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();checkpoint=json.loads(args.restart.read_text()) if args.restart else None
    result=solve_case(build_case(),args.through_step,checkpoint)
    args.output.write_text(json.dumps(dict(status=result.status,info=result.info,
        checkpoint=result.to_restart_checkpoint() if result.status=='completed' else None),
        default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item(),indent=2))
    if result.status!='completed':raise RuntimeError(result.info)


if __name__=='__main__':main()
