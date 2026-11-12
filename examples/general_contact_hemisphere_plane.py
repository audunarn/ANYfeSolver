"""Opt-in static hemisphere/rim seam; a development example, not qualification."""
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface,
    ContactInteraction, ContactSolverSettings, RigidContactBody, LoadCase,
    solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement


def run():
    model=FEModel('hemisphere_plane_closed_seam')
    model.add_material('elastic',1000.,.3,density=1.)
    model.add_node(1,-.5,0.,.0105);model.add_node(2,.5,0.,.0125)
    model.add_element(1,BeamElement(1,(1,2),'elastic',
        {'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    for node,drop in ((1,-.003),(2,-.007)):
        values={d:0. for d in ('ux','uy','uz','rx','ry','rz')};values['uz']=drop
        if node==2:values.pop('rx')
        model.add_boundary_condition(BoundaryCondition(f'pose{node}',(node,),values))
    definition=ContactDefinition(surfaces=(ContactSurface('beam',(1,),
        beam_radius=.01,beam_end_caps='hemisphere'),),
        rigid_bodies=(RigidContactBody('plane','plane',point=(0.,0.,0.),normal=(0.,0.,1.)),),
        interactions=(ContactInteraction('pair','beam','plane',normal_stiffness=None,
            line_stiffness=1e4,endpoint_stiffness=1e4,gap_tolerance=.006),))
    settings=ContactSolverSettings(pair_motion='general',beam_end_cap_work='hemisphere_support_dev')
    result=solve_static_nonlinear(model,LoadCase('proportional prescribed pose'),
        contact=definition,contact_settings=settings,num_steps=8,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    return result


if __name__=='__main__':
    result=run();info=result.info['general_contact']
    print({'status':result.status,'profile':info['profile'],
        'endpoint_fields':info['endpoint_fields'],'beam_force_N':info['beam_force'],
        'beam_moment_Nm_about_global_origin':info['beam_moment'],
        'gap_lower_m':info['minimum_gap'],'admitted_for_application':False})
