"""Contact-aware spherical continuation of a connected physical S3 patch."""
from examples.general_contact_triangulated_rigid_patch import case
from anyfesolver import LoadCase, solve_static_arc_length, ArcLengthControl


def analysis(steps=4, restart_checkpoint=None):
    model, contact, settings = case()
    load = LoadCase('arc_patch_press')
    for node in (2,3,4): load.add_nodal_load(node,forces=(0.,0.,-.05))
    control = ArcLengthControl(initial_load_increment=2.5e-5,minimum_load_increment=5e-7,
        maximum_load_increment=2.5e-5,load_scaling=.01,growth_factor=1.,max_steps=steps)
    return solve_static_arc_length(model,load,control=control,contact=contact,
        contact_settings=settings,max_iterations=15,emit_restart_checkpoint=True,
        record_increment_snapshots=True,restart_checkpoint=restart_checkpoint)


if __name__ == '__main__':
    result=analysis()
    print(result.status,result.info.get('general_contact'),len(result.steps))
