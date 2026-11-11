"""Development-only guided B3 pressed outward against two physical-end obstacles."""
from anyfesolver import (BoundaryCondition,ContactDefinition,ContactInteraction,ContactSolverSettings,
    ContactSurface,RigidContactBody,RigidCappedCylinder,LoadCase,solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import QuadraticBeamElement


def run(right_obstacle='sphere'):
    if right_obstacle not in ('sphere','cylinder','capped_cylinder'):
        raise ValueError('choose sphere, cylinder or capped_cylinder')
    model=FEModel('B3 two physical ends')
    model.add_material('elastic',1e3,.3,density=1.)
    for node,x in enumerate((.112,.212,.312),1):model.add_node(node,x,0.,0.)
    model.add_element(1,QuadraticBeamElement(1,(1,2,3),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    model.add_boundary_condition(BoundaryCondition('midpoint',(2,),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('end guides',(1,3),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    right=(RigidCappedCylinder('right',(.424,0.,0.),(1.,0.,0.),.01,.1) if right_obstacle=='capped_cylinder'
        else RigidContactBody('right',right_obstacle,(.424,0.,0.),radius=.1,
            normal=(0.,0.,1.) if right_obstacle=='cylinder' else None))
    definition=ContactDefinition((ContactSurface('tube',(1,),beam_radius=.01,beam_end_caps='hemisphere'),),
        (RigidContactBody('left','plane',(.1,0.,0.),(1.,0.,0.)),right),
        tuple(ContactInteraction(name,'tube',body,None,1e-5,line_stiffness=1e4,endpoint_stiffness=1e5)
            for name,body in (('left pair','left'),('right pair','right'))))
    settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-7)
    load=LoadCase('outward press');load.add_nodal_load(1,forces=(-.2,0.,0.))
    load.add_nodal_load(3,forces=(.2,0.,0.))
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=20,tolerance=1e-10,emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    return result


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--right-obstacle',choices=('sphere','cylinder','capped_cylinder'),default='sphere')
    result=run(parser.parse_args().right_obstacle);info=result.info['general_contact']
    print(json.dumps({'status':result.status,'profile':info['profile'],
        'active_endpoints':info['active_endpoints'],'rigid_bodies':info['rigid_bodies'],
        'minimum_gap_m':info['minimum_gap'],'admitted_for_application':False}))
