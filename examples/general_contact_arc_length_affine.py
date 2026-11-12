"""Affine support plus optional equilibrium preload, using the public arc route."""
from examples.general_contact_triangulated_rigid_patch import case
from anyfesolver import BoundaryCondition, LoadCase, ArcLengthControl, solve_static_arc_length


def build(preload=False,preload_force=.001):
    model,contact,settings=case()
    model.boundary_conditions[-1] = BoundaryCondition('root',(1,),{'uz':-2e-6})
    load=LoadCase('affine_arc_press')
    for node in (2,3,4):load.add_nodal_load(node,forces=(0.,0.,-.05))
    constant=None
    if preload:
        constant=LoadCase('physical_preload')
        for node in (2,3,4):constant.add_nodal_load(node,forces=(0.,0.,-preload_force))
    return model,contact,settings,load,constant


def analysis(steps=3,restart_checkpoint=None,preload=True,preload_force=.001,**extra):
    model,contact,settings,load,constant=build(preload,preload_force)
    control=ArcLengthControl(initial_load_increment=2.5e-5,minimum_load_increment=5e-7,
        maximum_load_increment=2.5e-5,load_scaling=.01,growth_factor=1.,max_steps=steps,preload_steps=2)
    return solve_static_arc_length(model,load,constant_load_case=constant,control=control,
        contact=contact,contact_settings=settings,max_iterations=15,
        emit_restart_checkpoint=True,record_increment_snapshots=True,
        restart_checkpoint=restart_checkpoint,**extra)


if __name__=='__main__':
    result=analysis()
    print(result.status,len(result.steps),result.info.get('general_contact'))
