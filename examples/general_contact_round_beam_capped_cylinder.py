"""Explicit exterior round B3/capped-cylinder static development example."""
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, RigidCappedCylinder, LoadCase,
    solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import QuadraticBeamElement


def run():
    model=FEModel('round beam on finite cylinder')
    model.add_material('elastic',1e3,.3,density=1.)
    for node,x in enumerate((-.5,0.,.5),1): model.add_node(node,x,0.,.03)
    model.add_element(1,QuadraticBeamElement(1,[1,2,3],'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    model.add_boundary_condition(BoundaryCondition('ends',(1,3),
        {key:0. for key in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('middle guide',(2,),
        {key:0. for key in ('ux','uy','rx','ry','rz')}))
    load=LoadCase('middle load'); load.add_nodal_load(2,forces=(0.,0.,-1.))
    contact=ContactDefinition(
        surfaces=(ContactSurface('tube',(1,),beam_radius=.01),),
        rigid_bodies=(RigidCappedCylinder('tool',(0.,0.,0.),(1.,0.,0.),.01,.1),),
        interactions=(ContactInteraction('line','tube','tool',normal_stiffness=None,
            line_stiffness=1e4,gap_tolerance=.004),))
    settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-7)
    result=solve_static_nonlinear(model,load,contact=contact,contact_settings=settings,
        num_steps=4,max_iterations=25,tolerance=1e-8,emit_restart_checkpoint=True)
    if result.status!='completed': raise RuntimeError(result.failure_reason)
    return result


if __name__=='__main__':
    result=run(); contact=result.info['general_contact']
    print({'status':result.status,'profile':contact['profile'],
        'beam_force_N':contact['beam_force'],'minimum_gap_lower_m':contact['minimum_gap'],
        'active_line_length_m':contact['active_line_length'],'admitted_for_application':False})
