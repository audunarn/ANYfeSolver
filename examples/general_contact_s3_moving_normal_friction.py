"""Full-face hard-normal S3 friction with a supported, moving target plane.

The first load is manufactured using disclosed structural K and independent
planar P1 skin forces. Subsequent tangential/open/recontact steps solve the same
coupled equations. This is development code verification, not qualification.
"""
import argparse,json
from pathlib import Path
import numpy as np
from anyfesolver import FEModel,BoundaryCondition,create_shell_element,solve_static_nonlinear
from anyfesolver.general_contact import (ContactDefinition,ContactSurface,ContactInteraction,
    ContactSolverSettings,ContactPathControl,ContactPathProgram,ContactPathNodalForce,
    S3MovingNormalFrictionSettings)

TILT=1e-6
ZTOTAL=.001


def build_case():
    model=FEModel('s3-moving-normal-hard-friction')
    model.add_material('elastic',1e5,.3,density=1.)
    points=((.55,.55,.0400002),(.65,.55,.0400002),(.65,.65,.0400002),(.55,.65,.0400002),
            (0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.))
    for node,p in enumerate(points,1):model.add_node(node,*p)
    for eid,nodes in enumerate(((1,2,4),(2,3,4),(5,6,8),(6,7,8)),1):
        model.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.05,
            formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),
            reference_surface_offset=.01 if eid>2 else 0.))
    model.add_boundary_condition(BoundaryCondition('real-open-state-clamps',(1,5),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('fixed-angular-profile',(2,3,4,6,7,8),
        {d:0. for d in ('rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('target-inplane-guide',(6,7,8),{'uy':0.}))
    for axis in (0,1):
        master=model.mesh.dof_manager.get_node_dofs(2)[axis]
        for node in (3,4):
            dep=model.mesh.dof_manager.get_node_dofs(node)[axis]
            model.add_constraint_equation(terms=((dep,1.),(master,-1.)),dependent_dof=dep,
                                          source_id=f'source-contact-translation-{axis}-{node}')
    definition=ContactDefinition((ContactSurface('source',(2,),'bottom'),
                                  ContactSurface('target',(4,),'top')),(),
        (ContactInteraction('hard-material','source','target',1000.,5e-5,normal_law='augmented'),))
    settings=ContactSolverSettings(pair_motion='general',normal_discretization='primal_mortar',
        general_s3_differentiation='jet2_s3_dev',general_s3_difference_step=2e-5,
        q4_source_gap_certificate='whole_face',augmentation_support_resolution='exact',
        shell_pair_orientation='physical_sides_dev',enforcement_solver='monolithic_active_set',
        complementarity_force_scale_n=.001,pressure_tolerance=1e-8)
    friction=S3MovingNormalFrictionSettings(.3,1000.,(2e-8,)*48,145,
                                           pressure_step_n=1e-8,closed_point_gap_tolerance_m=1e-10)
    # Manufactured code-verification load, independent of production contact.
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    K=assemble_stiffness_matrix(model)[0]
    assert K.shape==(48,48)
    pose=np.zeros(48);opened=np.zeros(48)
    for node in (2,3,4):
        dof=model.mesh.dof_manager.get_node_dofs(node)[2]
        pose[dof]=-2e-7-TILT*points[node-1][0];opened[dof]=2e-6
    for node in (6,7,8):pose[model.mesh.dof_manager.get_node_dofs(node)[2]]=-TILT*points[node-1][0]
    normal=np.array([TILT,0.,1.]);normal/=np.linalg.norm(normal)
    force=K@pose;openforce=K@opened
    source_nodes=(2,3,4);target_nodes=(6,7,8)
    target_xy=np.array([points[i-1][:2] for i in target_nodes])
    # Uniform pressure: exact P1 material-node source force z_i*n. The
    # target linear shape integral equals its value at the source centroid.
    centroid=np.mean(np.array([points[i-1][:2] for i in source_nodes]),axis=0)
    target_bary=np.linalg.solve(np.vstack((target_xy.T,np.ones(3))),np.r_[centroid,1.])
    for node in source_nodes:force[np.asarray(model.mesh.dof_manager.get_node_dofs(node)[:3])]-=ZTOTAL/3*normal
    for node,weight in zip(target_nodes,target_bary):
        force[np.asarray(model.mesh.dof_manager.get_node_dofs(node)[:3])]+=ZTOTAL*weight*normal
    waypoints=((0.,0.),(0.,0.),(1e-8,0.),(1e-4,0.),(9.95e-5,0.),(9.8e-5,0.),(9.8e-5,0.))
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
        supported_force=np.full(3,ZTOTAL/3),target_barycentric=target_bary,
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
