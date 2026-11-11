"""Development example: round B2 pressing a deformable, fixed-director S3 skin.

Run with PYTHONPATH=src. Not an independently accepted/installed capability.
All dimensions SI; the explicit finite line penalty is not hard contact.
"""
import json
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface, ContactInteraction,
                       ContactSolverSettings, LoadCase, create_shell_element, solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement

def fixture():
    model=FEModel('deformable_line_skin_dev')
    model.add_material('beam',1.e3,.3,density=1.)
    model.add_material('skin',1.e5,.3,density=1.)
    for node,p in enumerate(((0.,0.,.02005),(.5,0.,.02005),(-1.,-1.,0.),(2.,-1.,0.),(-1.,2.,0.)),1):
        model.add_node(node,*p)
    model.add_element(1,BeamElement(1,[1,2],'beam',{'area':1.e-3,'Iy':1.e-4,'Iz':1.e-4,'J':1.e-4}))
    model.add_element(2,create_shell_element(2,(3,4,5),'skin',thickness=.02,
                      formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=0.))
    definition=ContactDefinition((ContactSurface('tube',(1,),beam_radius=.01),
                         ContactSurface('skin',(2,),side='top')),(),
                         (ContactInteraction('line','tube','skin',None,1.e-5,line_stiffness=1.e5),))
    settings=ContactSolverSettings(pair_motion='general',beam_shell_rim_clearance=.02,
                     line_integration_relative_tolerance=1.e-8,
                     beam_shell_geometry='jet2_deformable_s3_dev',beam_shell_maximum_swept_intervals=128)
    return model,definition,settings


def constraints(model):
    names=('ux','uy','uz','rx','ry','rz')
    model.add_boundary_condition(BoundaryCondition('fixed',[1,3,4],dict.fromkeys(names,0.)))
    model.add_boundary_condition(BoundaryCondition('guides',[2,5],dict.fromkeys(('ux','uy','rx','ry','rz'),0.)))


def main():
    model, contact, settings=fixture();constraints(model)
    load=LoadCase('press');load.add_nodal_load(2,forces=(0.,0.,-.001))
    result=solve_static_nonlinear(model,load,contact=contact,contact_settings=settings,
                                 num_steps=4,max_iterations=20,tolerance=1.e-10,
                                 kinematics='von_karman',emit_restart_checkpoint=True)
    if result.status!='completed':
        raise RuntimeError(result.failure_reason)
    print(json.dumps({'status':result.status,'contact':result.info['general_contact']},indent=2))
    return result

if __name__=='__main__':
    main()
