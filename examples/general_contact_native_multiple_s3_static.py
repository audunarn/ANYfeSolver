"""Development example: retained native beams and two deformable S3 skins.

This explicit elastic interior profile is not engineering acceptance.
"""
import json
import numpy as np
from anyfesolver import create_shell_element
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces, Program
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact import NativeS3ContactPair, create_native_contact_analysis
from anyfesolver.general_contact.native_retained_line_s3 import NativeRetainedRoundLineS3Interior
from anyfesolver.general_contact.native_retained_line_s3_work import NativeRetainedLineS3Work
from anyfesolver.general_contact.overlap import reference_shell_face
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter


def analysis(*, beams=1, iterations=24, targets=(.5, 1.), line_stiffness_n_per_m2=10000.,
             gap_tolerance_m=.001, activation_settings=None):
    definitions, roots, forces = [], [], []
    axes = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')
    section = EllipsoidalGeneralizedSection(
        np.diag((1000., 400., 350., .8, 1., 1.2)), np.eye(6), 1e6, 1.)
    for i in range(beams):
        nodes = tuple(range(1+3*i, 4+3*i))
        points = np.array(((0., .3*i, 0.), (1., .2+.3*i, 0.), (2., .3*i, 0.)))
        frames = []
        for slope in (.4, 0., -.4):
            t = np.array((1., slope, 0.)); t /= np.linalg.norm(t)
            n = np.array((0., 0., 1.))
            frames.append(np.column_stack((t, n, np.cross(t, n))))
        reference = CenteredCurvedBeam3ReferenceGeometry(points, np.array(frames))
        element = NativeGeneralizedStaticElement(11+i, nodes, reference, section, order=4)
        definitions.append(NativeBeamDefinition.capture(element, np.diag((2., 2., 2., .07, .09, .11))))
        roots.append(BoundaryCondition('root'+str(i), [nodes[0]], {a: 0. for a in axes}))
        forces.append((nodes[2], 0., -.01 if i == 0 else 0., -.01))
    program = Program(targets, DistributedPattern(LinePattern(()), ()),
                      NodalDeadForces(tuple(forces)), max_iterations=iterations)
    shell = FEModel('native_multiple_s3')
    shell.add_material('elastic', 1e5, .3, density=1.)
    faces = ((21, (4, 5, 6), ((-5., -5., -.076), (5., -5., -.076), (0., 10., -.076)), (0, 0, 1), 'uz'),
             (22, (7, 8, 9), ((-5., -.076, 5.), (5., -.076, 5.), (0., -.076, -10.)), (0, 1, 0), 'uy'))
    for eid, nodes, points, normal, free in faces:
        for nid, point in zip(nodes, points): shell.add_node(nid, *point)
        shell.add_element(eid, create_shell_element(eid, nodes, 'elastic', thickness=.05,
            formulation='e4-pl-s3-v2d', reference_normal=normal, reference_surface_offset=0.))
        shell.add_boundary_condition(FixedSupport('edge'+str(eid), [nodes[0], nodes[2]]))
        shell.add_boundary_condition(BoundaryCondition('vertex'+str(eid), [nodes[1]],
            {a: 0. for a in axes if a != free}))
    pairs = []
    for i, definition in enumerate(definitions):
        element, _ = definition.instantiate()
        for eid, nodes, _, _, _ in faces[:2 if i == 0 else 1]:
            body = eid-14
            skin = S3SkinWorkAdapter.from_model(shell, body, eid, 'top')
            kernel = NativeRetainedRoundLineS3Interior(element.element_id,
                element.operator.reference, tuple(element.node_ids), eid,
                reference_shell_face(shell, body, eid, 'top'), nodes,
                radius_m=.05, line_stiffness_n_per_m2=line_stiffness_n_per_m2, rim_clearance_m=.1,
                geometry_tolerance_m=1e-10, energy_tolerance_j=1e-10,
                force_tolerance_n=1e-9, tangent_tolerance_n_per_m=1e-7,
                maximum_integration_intervals=100)
            pairs.append(NativeS3ContactPair(f'beam{i}-skin{eid}', NativeRetainedLineS3Work(kernel, skin), gap_tolerance_m, 128))
    return create_native_contact_analysis(tuple(definitions), tuple(roots), program, shell,
                                         tuple(pairs), equilibrium_tolerance=1e-8,
                                         activation_settings=activation_settings)


def tight_gap_analysis(*, beams=1):
    """Explicit stronger line law and activation controls; development only."""
    from anyfesolver.general_contact import NativeContactActivationSettings
    return analysis(beams=beams, targets=(.25,.5,.75,1.),
        line_stiffness_n_per_m2=1.2e6, gap_tolerance_m=5e-5,
        activation_settings=NativeContactActivationSettings(1e-7 if beams==1 else 1e-6,64))


def main():
    owner = analysis(beams=2)
    run = owner.solve()
    if run.status != 'completed': raise RuntimeError(run.failure)
    print(json.dumps({'profile': run.profile, 'status': run.status,
        'completed_targets': run.accepted.completed_targets,
        'shell_displacement_norm_m': float(np.linalg.norm(run.accepted.shell_displacement)),
        'minimum_gap_m': run.accepted.contact.contact.minimum_gap_m,
        'checkpoint_sha256': run.checkpoint_sha256}, sort_keys=True))


if __name__ == '__main__': main()
