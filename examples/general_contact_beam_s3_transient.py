"""Elastic moving beam/deformable-skin impact, explicit development profile."""
import numpy as np
from anyfesolver import BoundaryCondition, ContactDefinition, ContactSurface, ContactInteraction, ContactSolverSettings, create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.transient import solve_transient_contact, ContactTransientSolverControls


def build(*,quadratic=False,unequal=False,line_stiffness=1.e5,flexural=False):
    if flexural and not quadratic:raise ValueError('flexural fixture requires the actual B3 interpolation')
    model=FEModel('beam_skin_impact')
    model.add_material('beam',1.e3,.3,density=1.)
    model.add_material('skin',1.e5,.3,density=1.)
    for node,p in enumerate(((0.,0.,.02005),(.5,0.,.02005),(-1.,-1.,0.),(2.,-1.,0.),(-1.,2.,0.)),1):
        model.add_node(node,*p)
    cls=BeamElement;ids=[1,2]
    if quadratic:
        model.add_node(6,.25,0.,.02005);cls=QuadraticBeamElement;ids=[1,6,2]
    model.add_element(1,cls(1,ids,'beam',dict(area=1.e-3,Iy=1.e-4,Iz=1.e-4,J=1.e-4)))
    model.add_element(2,create_shell_element(2,(3,4,5),'skin',thickness=.02,formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=0.))
    if flexural:
        model.add_boundary_condition(BoundaryCondition('beam_bending_guides',ids,dict.fromkeys(('ux','uy','rx','rz'),0.)))
        model.add_boundary_condition(BoundaryCondition('skin_director_guides',[3,4,5],dict.fromkeys(('ux','uy','rx','ry','rz'),0.)))
    else:
        model.add_boundary_condition(BoundaryCondition('normal_guides',list(model.mesh.nodes),dict.fromkeys(('ux','uy','rx','ry','rz'),0.)))
    dof=lambda node:model.mesh.dof_manager.get_node_dofs(node)[2]
    for node in (() if flexural else ids[1:]):
        model.add_constraint_equation(terms=((dof(node),1.),(dof(1),-1.)),source_id=f'beam_uniform_{node}',dependent_dof=dof(node))
    if not unequal:
        for node in (4,5):model.add_constraint_equation(terms=((dof(node),1.),(dof(3),-1.)),source_id=f'skin_uniform_{node}',dependent_dof=dof(node))
    definition=ContactDefinition((ContactSurface('tube',(1,),beam_radius=.01),ContactSurface('skin',(2,),side='top')),(),(ContactInteraction('line','tube','skin',None,1.e-5,line_stiffness=line_stiffness),))
    settings=ContactSolverSettings(pair_motion='general',beam_shell_rim_clearance=.02,line_integration_relative_tolerance=1.e-8,beam_shell_geometry='jet2_deformable_s3_dev',beam_shell_maximum_swept_intervals=128)
    velocity=np.zeros(model.mesh.dof_manager.total_dofs)
    for node in ids:velocity[dof(node)]=-.04
    if flexural:
        velocity[dof(6)]=-.044
        for node,rate in ((1,.032),(6,0.),(2,-.032)):
            velocity[model.mesh.dof_manager.get_node_dofs(node)[4]]=rate
    for node,value in zip((3,4,5),(.01,.009,.011) if unequal else (.01,.01,.01)):velocity[dof(node)]=value
    return model,definition,settings,velocity


def run(*,end=.0018,dt=1.e-5,quadratic=False,unequal=False,restart=None,line_stiffness=1.e5,flexural=False):
    model,definition,settings,velocity=build(quadratic=quadratic,unequal=unequal,line_stiffness=line_stiffness,flexural=flexural)
    result=solve_transient_contact(model,definition,TransientConfig(dt=dt,t_end=end,initial_velocity=velocity),contact_settings=settings,
        solver_controls=ContactTransientSolverControls(allow_deformable_beam_s3_dev=True,save_constraint_reactions_dev=True),
        emit_restart_checkpoint=True,restart_checkpoint=restart)
    return model,result


if __name__=='__main__':
    _,result=run()
    print(dict(states=len(result.times),active=sum(x.active_line_length>0 for x in result.contact),minimum_gap=min(x.minimum_gap for x in result.contact),final_gap=result.contact[-1].minimum_gap))
