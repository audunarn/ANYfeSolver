"""Development-only physical-end contact between two guided elastic B2 bars."""
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement


def run():
    model=FEModel('two physical B2 ends')
    model.add_material('elastic',1e3,.3,density=1.)
    for node,x in enumerate((-.212,-.012,.012,.212),1):model.add_node(node,x,0.,0.)
    section={'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}
    for element,nodes in ((1,(1,2)),(2,(3,4))):
        model.add_element(element,BeamElement(element,nodes,'elastic',section))
    model.add_boundary_condition(BoundaryCondition('roots',(1,4),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('axial guides',(2,3),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    definition=ContactDefinition(tuple(ContactSurface(name,(element,),beam_radius=.01,
        beam_end_caps='hemisphere') for name,element in (('a',1),('b',2))),(),
        (ContactInteraction('pair','a','b',None,1e-5,line_stiffness=1e4,endpoint_stiffness=1e5),))
    settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-7)
    load=LoadCase('symmetric axial closure')
    load.add_nodal_load(2,forces=(.2,0.,0.));load.add_nodal_load(3,forces=(-.2,0.,0.))
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=25,tolerance=1e-10,emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    return result


if __name__=='__main__':
    import json
    result=run();info=result.info['general_contact']
    print(json.dumps({'status':result.status,'profile':info['profile'],
        'active_endpoints':info['active_endpoints'],'minimum_gap_m':info['minimum_gap'],
        'line_storage_J':info['line_contact_energy'],'endpoint_storage_J':info['endpoint_contact_energy'],
        'admitted_for_application':False}))
