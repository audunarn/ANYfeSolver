"""Small-motion B2 line/caps pressing a deformable interior S3 physical skin.

Development profile: fixed shell directors, frictionless elastic materials,
explicit radius-covering face clearance and distinct line/endpoint stiffness.
"""
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface,
    ContactInteraction, ContactSolverSettings, LoadCase, create_shell_element,
    solve_static_nonlinear)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def build():
    model=FEModel('deforming_S3_hemisphere_support')
    model.add_material('beam',1.e3,.3,density=1.)
    model.add_material('skin',1.e5,.3,density=1.)
    points=((0.,0.,.02005),(.5,0.,.02005),(-1.,-1.,0.),(2.,-1.,0.),(-1.,2.,0.))
    for node,p in enumerate(points,1):model.add_node(node,*p)
    model.add_element(1,BeamElement(1,(1,2),'beam',
        {'area':.001,'Iy':1.e-4,'Iz':1.e-4,'J':1.e-4}))
    model.add_element(2,create_shell_element(2,(3,4,5),'skin',thickness=.02,
        formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=0.))
    model.add_boundary_condition(BoundaryCondition('fixed',(1,3,4),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('guides',(2,5),
        {d:0. for d in ('ux','uy','rx','ry','rz')}))
    definition=ContactDefinition((
        ContactSurface('tube',(1,),beam_radius=.01,beam_end_caps='hemisphere'),
        ContactSurface('skin',(2,),side='top')),(),
        (ContactInteraction('line plus ends','tube','skin',None,1.e-5,
            line_stiffness=1.e5,endpoint_stiffness=1.e3),))
    settings=ContactSolverSettings(pair_motion='general',beam_shell_rim_clearance=.02,
        line_integration_relative_tolerance=1.e-8,beam_shell_geometry='jet2_deformable_s3_dev',
        beam_shell_maximum_swept_intervals=128,beam_end_cap_work='hemisphere_support_dev')
    load=LoadCase('press');load.add_nodal_load(2,forces=(0.,0.,-.001))
    return model,definition,settings,load


def run():
    model,definition,settings,load=build()
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=25,tolerance=1.e-10,emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    return result


if __name__=='__main__':
    result=run();info=result.info['general_contact']
    print({'status':result.status,'profile':info['profile'],
        'minimum_gap_m':info['minimum_gap'],'active_endpoints':info['active_endpoints'],
        'beam_force_N':info['beam_force'],'shell_force_N':info['shell_force'],
        'admitted_for_application':False})
