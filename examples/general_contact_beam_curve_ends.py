"""Explicit quadratic-beam physical ends: two elastic guided bars (development).

Each genuine cap/cap contributes one point spring; the distributed line penalty
uses its own N/m² setting. This is a code example, not engineering qualification.
"""
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, solve_static_nonlinear)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel


def run():
    model=FEModel('quadratic_physical_end_example')
    model.add_material('elastic',1000.,.3,density=1.)
    for i,x in enumerate((-.2,-.1,0.,.104,.204,.304),1):model.add_node(i,x,0.,0.)
    section={'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}
    model.add_element(1,QuadraticBeamElement(1,(1,2,3),'elastic',section))
    model.add_element(2,QuadraticBeamElement(2,(4,5,6),'elastic',section))
    model.add_boundary_condition(BoundaryCondition('outer',(1,6),{'ux':0.}))
    model.add_boundary_condition(BoundaryCondition('guide',(1,2,3,4,5,6),
        {name:0. for name in ('uy','uz','rx','ry','rz')}))
    contact=ContactDefinition(surfaces=(
        ContactSurface('a',(1,),beam_radius=.05,beam_end_caps='hemisphere'),
        ContactSurface('b',(2,),beam_radius=.05,beam_end_caps='hemisphere')),
        rigid_bodies=(),interactions=(ContactInteraction('pair','a','b',normal_stiffness=None,
            line_stiffness=1000.,endpoint_stiffness=50.,gap_tolerance=.008),))
    settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-6,
        maximum_line_subdivisions=16,maximum_swept_subdivisions=18,
        beam_pair_gap_accuracy=1e-5,maximum_beam_pair_gap_cells=10000,maximum_beam_pair_gap_depth=25)
    load=LoadCase('two_bar_load');load.add_nodal_load(3,forces=(.2,0.,0.))
    load.add_nodal_load(4,forces=(-.2,0.,0.))
    return solve_static_nonlinear(model,load,contact=contact,contact_settings=settings,
        num_steps=4,max_iterations=25,tolerance=1e-10,emit_restart_checkpoint=True)


if __name__=='__main__':
    result=run()
    print(result.status,result.info.get('general_contact'))
