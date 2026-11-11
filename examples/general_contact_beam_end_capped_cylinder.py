"""Opt-in axial B2 with separate line and hemispherical endpoint penalties."""
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, RigidCappedCylinder, RigidContactBody, LoadCase,
    solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement


def run(*,plane=False,radial=None):
    if radial not in (None,'sphere','cylinder') or (plane and radial is not None):
        raise ValueError('select one plane, radial sphere/cylinder, or capped-cylinder obstacle')
    model=FEModel('declared round physical beam end')
    model.add_material('elastic',1e3,.3,density=1.)
    model.add_node(1,.112,0.,0.);model.add_node(2,.312,0.,0.)
    model.add_element(1,BeamElement(1,(1,2),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    model.add_boundary_condition(BoundaryCondition('fixed',(2,),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('axial guide',(1,),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    load=LoadCase('press');load.add_nodal_load(1,forces=(-.2,0.,0.))
    obstacle=(RigidContactBody('tool',radial,(0.,0.,0.),
        normal=(0.,0.,1.) if radial=='cylinder' else None,radius=.1) if radial else
        RigidContactBody('tool','plane',(.1,0.,0.),(1.,0.,0.)) if plane else
        RigidCappedCylinder('tool',(0.,0.,0.),(1.,0.,0.),.01,.1))
    definition=ContactDefinition(
        (ContactSurface('tube',(1,),beam_radius=.01,beam_end_caps='hemisphere'),),
        (obstacle,),
        (ContactInteraction('line_and_end','tube','tool',None,1e-5,
            line_stiffness=1e4,endpoint_stiffness=1e5),))
    settings=ContactSolverSettings(pair_motion='general',
        line_integration_relative_tolerance=1e-7)
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=20,tolerance=1e-10,emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    geometry=parser.add_mutually_exclusive_group()
    geometry.add_argument('--plane',action='store_true',help='Use a fixed infinite plane')
    geometry.add_argument('--radial',choices=('sphere','cylinder'),help='Use a fixed radial obstacle')
    args=parser.parse_args()
    result=run(plane=args.plane,radial=args.radial);info=result.info['general_contact']
    print({'status':result.status,'profile':info['profile'],
        'physical_end_fields':info.get('endpoint_fields'),'beam_force_N':info['beam_force'],
        'gap_lower_m':info['minimum_gap'],'admitted_for_application':False})
