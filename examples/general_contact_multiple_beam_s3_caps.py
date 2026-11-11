"""Explicit B2 contact against two deforming physical S3 interiors.

Installed-tested development example; not an engineering qualification.
"""
from anyfesolver import (FEModel, BoundaryCondition, LoadCase, ContactDefinition,
    ContactSurface, ContactInteraction, ContactSolverSettings, solve_static_nonlinear,
    create_shell_element)
from anyfesolver.elements import BeamElement


def build():
    model = FEModel('two_deforming_s3_caps_dev')
    model.add_material('beam', 1.e3, .3, density=1.)
    model.add_material('skin', 1.e5, .3, density=1.)
    positions = ((0., 0., .02005), (.5, 0., .02005),
        (-1., -1., 0.), (2., -1., 0.), (-1., 2., 0.),
        (-1., -.02005, 1.02005), (2., -.02005, 1.02005), (-1., -.02005, -1.97995))
    for node, p in enumerate(positions, 1):
        model.add_node(node, *p)
    model.add_element(1, BeamElement(1, (1, 2), 'beam',
        {'area': .001, 'Iy': 1.e-4, 'Iz': 1.e-4, 'J': 1.e-4}))
    for element, nodes, normal in ((2, (3, 4, 5), (0., 0., 1.)),
                                   (3, (6, 7, 8), (0., 1., 0.))):
        model.add_element(element, create_shell_element(element, nodes, 'skin', thickness=.02,
            formulation='e4-pl-s3-v2d', reference_normal=normal, reference_surface_offset=0.))
    model.add_boundary_condition(BoundaryCondition('fixed', (1, 3, 4, 6, 7),
        dict.fromkeys(('ux', 'uy', 'uz', 'rx', 'ry', 'rz'), 0.)))
    for name, node, constrained in (('tip', 2, ('ux', 'rx', 'ry', 'rz')),
        ('z_skin', 5, ('ux', 'uy', 'rx', 'ry', 'rz')),
        ('y_skin', 8, ('ux', 'uz', 'rx', 'ry', 'rz'))):
        model.add_boundary_condition(BoundaryCondition(name, (node,), dict.fromkeys(constrained, 0.)))
    definition = ContactDefinition((ContactSurface('tube', (1,), beam_radius=.01, beam_end_caps='hemisphere'),
        ContactSurface('z_skin', (2,), side='top'), ContactSurface('y_skin', (3,), side='top')), (),
        tuple(ContactInteraction(name, 'tube', target, None, 1.e-5,
            line_stiffness=1.e5, endpoint_stiffness=1.e3)
            for name, target in (('z_pair', 'z_skin'), ('y_pair', 'y_skin'))))
    settings = ContactSolverSettings(pair_motion='general', beam_shell_rim_clearance=.02,
        line_integration_relative_tolerance=1.e-8, beam_shell_geometry='jet2_deformable_s3_dev',
        beam_shell_maximum_swept_intervals=128, beam_end_cap_work='hemisphere_support_dev')
    load = LoadCase('two_skin_press')
    load.add_nodal_load(2, forces=(0., -.001, -.001))
    return model, definition, settings, load


def run():
    model, definition, settings, load = build()
    result = solve_static_nonlinear(model, load, contact=definition, contact_settings=settings,
        num_steps=4, max_iterations=25, tolerance=1.e-10, emit_restart_checkpoint=True)
    if result.status != 'completed':
        raise RuntimeError(result.failure_reason)
    return result


if __name__ == '__main__':
    result = run()
    print(result.status, result.info['general_contact'])
