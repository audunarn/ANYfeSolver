"""Elastic B3 bar with declared actual hemisphere support on a fixed infinite cylinder."""
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface,
    ContactInteraction, ContactSolverSettings, RigidContactBody, LoadCase,
    solve_static_nonlinear)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel


def run():
    model=FEModel('hemisphere_cylinder_B3_bar')
    model.add_material('elastic',1000.,.3,density=1.)
    for node,x in enumerate((.112,.212,.312),1):model.add_node(node,x,0.,0.)
    model.add_element(1,QuadraticBeamElement(1,(1,2,3),'elastic',
        {'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    model.add_boundary_condition(BoundaryCondition('fixed',(3,),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('guides',(1,2),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    definition=ContactDefinition((ContactSurface('tube',(1,),beam_radius=.01,beam_end_caps='hemisphere'),),
        (RigidContactBody('cylinder','cylinder',(0.,0.,0.),normal=(0.,0.,1.),radius=.1),),
        (ContactInteraction('pair','tube','cylinder',None,1e-5,line_stiffness=1e4,endpoint_stiffness=1e5),))
    settings=ContactSolverSettings(pair_motion='general',beam_end_cap_work='hemisphere_support_dev',
        line_integration_relative_tolerance=1e-7,maximum_line_subdivisions=14,maximum_swept_subdivisions=16)
    load=LoadCase('press');load.add_nodal_load(1,forces=(-.2,0.,0.))
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=20,tolerance=1e-10,emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    return result


if __name__=='__main__':
    result=run();info=result.info['general_contact']
    print({'status':result.status,'profile':info['profile'],'minimum_gap_m':info['minimum_gap'],
        'active_endpoints':info['active_endpoints'],'rigid_force_N':info['rigid_force'],
        'admitted_for_application':False})
