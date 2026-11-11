"""Connected two-S3 physical patch against a nonmatching finite rigid tool."""
from anyfesolver import (BoundaryCondition,ContactDefinition,ContactSurface,ContactInteraction,
    LoadCase,create_shell_element,solve_static_nonlinear)
from examples.general_contact_triangulated_rigid import case as single_case


def case():
    model,definition,settings=single_case()
    model.add_node(4,.8,.85,.00501)
    # Replace the first triangle and add a second sharing the complete 2--3 edge.
    model.mesh.elements.clear()
    for eid,nodes in ((1,(1,2,3)),(2,(2,4,3))):
        model.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.01,
            formulation='e4-pl-s3-v2d',reference_normal=(0,0,1)))
    model.boundary_conditions.clear()
    model.add_boundary_condition(BoundaryCondition('guide',(1,2,3,4),
        {key:0. for key in ('ux','uy','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('root',(1,),{'uz':0.}))
    definition=ContactDefinition((ContactSurface('skin',(1,2),'bottom'),),definition.rigid_bodies,
        (ContactInteraction('patch_tool','skin','tool',1e6,5e-5),))
    return model,definition,settings


def analysis():
    model,definition,settings=case();load=LoadCase('patch_press')
    for node in (2,3,4): load.add_nodal_load(node,forces=(0.,0.,-.05))
    return solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        kinematics='von_karman',num_steps=2,max_iterations=15,emit_restart_checkpoint=True)


if __name__=='__main__':
    result=analysis();print(result.status,result.info.get('general_contact'))
