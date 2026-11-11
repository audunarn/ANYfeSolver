"""B3 physical ends against a fixed plane and sphere; developmental static workflow."""
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface,
    ContactInteraction, ContactSolverSettings, RigidContactBody, LoadCase,
    solve_static_nonlinear)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel


def run():
    model=FEModel('plane_sphere_B3_physical_ends')
    model.add_material('elastic',1000.,.3,density=1.)
    for node,x in enumerate((.112,.212,.312),1):model.add_node(node,x,0.,0.)
    model.add_element(1,QuadraticBeamElement(1,(1,2,3),'elastic',
        {'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    model.add_boundary_condition(BoundaryCondition('midpoint',(2,),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('guides',(1,3),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    definition=ContactDefinition((ContactSurface('tube',(1,),beam_radius=.01,beam_end_caps='hemisphere'),),
        (RigidContactBody('left','plane',(.1,0.,0.),(1.,0.,0.)),
         RigidContactBody('right','sphere',(.424,0.,0.),radius=.1)),
        tuple(ContactInteraction(name,'tube',target,None,1e-5,
            line_stiffness=1e4,endpoint_stiffness=1e5)
            for name,target in (('left pair','left'),('right pair','right'))))
    settings=ContactSolverSettings(pair_motion='general',beam_end_cap_work='hemisphere_support_dev',
        line_integration_relative_tolerance=1e-7,maximum_line_subdivisions=14,maximum_swept_subdivisions=16)
    load=LoadCase('opposed outward loads');load.add_nodal_load(1,forces=(-.2,0.,0.))
    load.add_nodal_load(3,forces=(.2,0.,0.))
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=20,tolerance=1e-10,emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    return result


if __name__=='__main__':
    result=run();info=result.info['general_contact']
    print({'status':result.status,'profile':info['profile'],
        'rigid_bodies':info['rigid_bodies'],'minimum_gap_m':info['minimum_gap'],
        'active_endpoints':info['active_endpoints'],'admitted_for_application':False})
