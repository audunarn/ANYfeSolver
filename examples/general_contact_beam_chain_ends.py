"""Development-only physical-end contact between nonmatching elastic B2 chains."""
import numpy as np
from anyfesolver import (BoundaryCondition,ContactDefinition,ContactInteraction,ContactSolverSettings,
    ContactSurface,LoadCase,solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement


def run(cells=4,target_cells=3):
    if any(type(n) is not int or not 2<=n<=16 for n in (cells,target_cells)):
        raise ValueError('choose between two and sixteen cells per chain')
    model=FEModel('nonmatching physical B2 chain ends')
    model.add_material('elastic',1e3,.3,density=1.)
    for node,x in enumerate(np.r_[np.linspace(-.212,-.012,cells+1),
                                 np.linspace(.012,.212,target_cells+1)],1):model.add_node(node,x,0.,0.)
    groups=(tuple(range(1,cells+1)),tuple(range(cells+1,cells+target_cells+1)))
    nodes=(tuple(range(1,cells+2)),tuple(range(cells+2,cells+target_cells+3)))
    section={'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}
    for group,lane in zip(groups,nodes):
        for element,a,b in zip(group,lane[:-1],lane[1:]):
            model.add_element(element,BeamElement(element,(a,b),'elastic',section))
    model.add_boundary_condition(BoundaryCondition('roots',(nodes[0][0],nodes[1][-1]),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('axial guides',(*nodes[0][1:],*nodes[1][:-1]),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    contact=ContactDefinition(tuple(ContactSurface(name,group,beam_radius=.01,beam_end_caps='hemisphere')
        for name,group in zip(('a','b'),groups)),(),
        (ContactInteraction('pair','a','b',None,1e-5,line_stiffness=1e4,endpoint_stiffness=1e5),))
    settings=ContactSolverSettings(pair_motion='general',beam_chain_tangent='consistent_dev',
        line_integration_relative_tolerance=1e-7)
    load=LoadCase('symmetric closure');load.add_nodal_load(nodes[0][-1],forces=(.2,0.,0.))
    load.add_nodal_load(nodes[1][0],forces=(-.2,0.,0.))
    result=solve_static_nonlinear(model,load,contact=contact,contact_settings=settings,num_steps=4,
        max_iterations=25,tolerance=1e-10,emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    return result


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=int,default=4);parser.add_argument('--target-cells',type=int,default=3)
    args=parser.parse_args();result=run(args.cells,args.target_cells);info=result.info['general_contact']
    print(json.dumps({'status':result.status,'profile':info['profile'],'active_endpoints':info['active_endpoints'],
        'minimum_gap_m':info['minimum_gap'],'line_storage_J':info['line_contact_energy'],
        'endpoint_storage_J':info['endpoint_contact_energy'],'admitted_for_application':False}))
