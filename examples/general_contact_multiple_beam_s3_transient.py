"""Two distinct elastic tubes impacting one deformable S3 skin.

Explicit development example, verified by focused source controls; installed
verification is recorded separately. Three uniform physical modes use actual
beam/shell mass, without contact-supplied inertia or hidden constraints.
"""
import numpy as np
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface,
                       ContactInteraction, ContactSolverSettings, create_shell_element)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.transient import (solve_transient_contact,
                                                ContactTransientSolverControls)


def build():
    model=FEModel('multiple_tubes_shared_skin')
    model.add_material('beam',1.e3,.3,density=1.)
    model.add_material('skin',1.e5,.3,density=1.)
    positions={1:(0.,-.1,.020005),2:(.5,-.1,.020005),
               6:(0.,.1,.020005),7:(.5,.1,.020005),8:(.25,.1,.020005),
               3:(-1.,-1.,0.),4:(2.,-1.,0.),5:(-1.,2.,0.)}
    for node,point in positions.items():model.add_node(node,*point)
    section=dict(area=1.e-3,Iy=1.e-4,Iz=1.e-4,J=1.e-4)
    model.add_element(1,BeamElement(1,[1,2],'beam',section.copy()))
    model.add_element(3,QuadraticBeamElement(3,[6,8,7],'beam',section.copy()))
    model.add_element(2,create_shell_element(2,(3,4,5),'skin',thickness=.02,
        formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=0.))
    model.add_boundary_condition(BoundaryCondition('physical_normal_guides',list(positions),
        dict.fromkeys(('ux','uy','rx','ry','rz'),0.)))
    z=lambda node:model.mesh.dof_manager.get_node_dofs(node)[2]
    for master,slaves in ((1,(2,)),(6,(8,7)),(3,(4,5))):
        for node in slaves:
            model.add_constraint_equation(terms=((z(node),1.),(z(master),-1.)),
                source_id=f'physical_uniform_{master}_{node}',dependent_dof=z(node))
    definition=ContactDefinition((ContactSurface('tube_a',(1,),beam_radius=.01),
        ContactSurface('tube_b',(3,),beam_radius=.01),ContactSurface('skin',(2,),side='top')),(),
        (ContactInteraction('line_a','tube_a','skin',None,1.e-5,line_stiffness=1.e5),
         ContactInteraction('line_b','tube_b','skin',None,1.e-5,line_stiffness=1.e5)))
    settings=ContactSolverSettings(pair_motion='general',beam_shell_rim_clearance=.02,
        line_integration_relative_tolerance=1.e-8,beam_shell_geometry='jet2_deformable_s3_dev',
        beam_shell_maximum_swept_intervals=128)
    velocity=np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1,2,6,8,7):velocity[z(node)]=-.04
    for node in (3,4,5):velocity[z(node)]=.01
    return model,definition,settings,velocity


def run(*,end=.0005,dt=1.e-5,restart=None,cancellation_token=None,emit_checkpoint=True):
    model,definition,settings,velocity=build()
    result=solve_transient_contact(model,definition,
        TransientConfig(dt=dt,t_end=end,initial_velocity=velocity),contact_settings=settings,
        solver_controls=ContactTransientSolverControls(allow_multiple_deformable_beam_s3_dev=True,
            save_constraint_reactions_dev=True),emit_restart_checkpoint=emit_checkpoint,
        restart_checkpoint=restart,cancellation_token=cancellation_token)
    return model,result


if __name__=='__main__':
    _,result=run()
    print(dict(profile=result.profile,states=len(result.times),
        pair_names=[name for name,_ in result.contact[-1].pairs],
        active=sum(row.active_line_length>0 for row in result.contact),
        final_gap=result.contact[-1].minimum_gap))
