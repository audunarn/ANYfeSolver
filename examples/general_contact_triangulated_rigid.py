"""Finite planar tool, formulation-owned S3 skin, ordinary static driver."""
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSurface, ContactSolverSettings, create_shell_element, LoadCase,
    solve_static_nonlinear)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact import TriangulatedRigidContactBody


def case(diagonal=0):
    model=FEModel('finite_rigid_s3')
    model.add_material('elastic',1e5,.3,density=1.)
    for i,p in enumerate(((.2,.15,.00501),(.85,.2,.00501),(.3,.8,.00501)),1):
        model.add_node(i,*p)
    model.add_element(1,create_shell_element(1,(1,2,3),'elastic',thickness=.01,
        formulation='e4-pl-s3-v2d',reference_normal=(0,0,1)))
    model.add_boundary_condition(BoundaryCondition('guide',(1,2,3),
        {key:0. for key in ('ux','uy','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('root',(1,),{'uz':0.}))
    faces=((1,(0,1,2)),(2,(0,2,3))) if diagonal==0 else ((1,(0,1,3)),(2,(1,2,3)))
    tool=TriangulatedRigidContactBody('tool',((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)),faces)
    definition=ContactDefinition((ContactSurface('skin',(1,),'bottom'),),(tool,),
        (ContactInteraction('skin_tool','skin','tool',1e6,5e-5),))
    settings=ContactSolverSettings(pair_motion='general',normal_discretization='primal_mortar',
        general_s3_difference_step=1e-6)
    return model,definition,settings


def analysis():
    model,definition,settings=case()
    load=LoadCase('press')
    load.add_nodal_load(2,forces=(0.,0.,-.05))
    load.add_nodal_load(3,forces=(0.,0.,-.05))
    return solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        kinematics='von_karman',num_steps=2,max_iterations=15,emit_restart_checkpoint=True)


if __name__=='__main__':
    result=analysis()
    print(result.status,result.info.get('general_contact'))
