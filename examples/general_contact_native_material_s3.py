"""Interior native round material patch against a deformable S3 physical skin.

Frictionless static development: one or two native beams and physical S3 facets.
Strictly interior straight half-cell patches and small-rotation elastic S3 skins.
Caps, friction and dynamics remain unadmitted.
"""
import json
import sys
from math import pi

import numpy as np

from anyfesolver import create_shell_element
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces, Program
from anyfesolver.general_contact.native_material_s3 import NativeMaterialS3Patch, NativeRetainedMaterialS3Work
from anyfesolver.general_contact.native_retained_s3_static import NativeRetainedRoundS3Analysis
from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter


def analysis(*, beams=1, targets=None, facets=1):
    if type(beams) is not int or beams not in (1, 2):
        raise ValueError('example supports one or two explicit native beams')
    if type(facets) is not int or facets not in (1,2):
        raise ValueError('example supports one or two physical S3 facets')
    if targets is None:
        targets=(.5,1.,0.,.5,1.) if beams==1 and facets==1 else (.25,.5,.75,1.,0.,.25,.5,.75,1.)
    frame = np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
    section = EllipsoidalGeneralizedSection(np.diag((1000., 400., 350., .8, 1., 1.2)),
                                           np.eye(6), 1e6, 1.)
    elements, definitions, roots, forces = [], [], [], []
    for i in range(beams):
        points = np.array(((0., .3*i, 0.), (1., .3*i, 0.), (2., .3*i, 0.)))
        reference = CenteredCurvedBeam3ReferenceGeometry(points, np.tile(frame, (3, 1, 1)))
        nodes = tuple(range(1+3*i, 4+3*i))
        element = NativeGeneralizedStaticElement(1+i, nodes, reference, section, order=4)
        elements.append(element)
        definitions.append(NativeBeamDefinition.capture(element, np.diag((2., 2., 2., .07, .09, .11))))
        roots.append(BoundaryCondition('clamped' if i==0 else 'clamped'+str(i), [nodes[0]],
                         {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
        forces.append((nodes[2], 0., -.003 if facets==2 and i==0 else 0., -.005 if i==0 else -.003))
    program = Program(targets, DistributedPattern(LinePattern(()), ()),
                      NodalDeadForces(tuple(forces)), max_iterations=24)
    shell = FEModel('native material patch and deformable S3')
    shell.add_material('elastic', 1e8, .3, density=1.)
    for node, (x, y) in zip((4, 5, 6), ((-5., -5.), (5., -5.), (0., 10.))):
        shell.add_node(node, x, y, -.0751)
    shell.add_element(21, create_shell_element(21, (4, 5, 6), 'elastic', thickness=.05,
        formulation='e4-pl-s3-v2d', reference_normal=(0, 0, 1), reference_surface_offset=0.))
    shell.add_boundary_condition(FixedSupport('edge', [4, 6]))
    shell.add_boundary_condition(BoundaryCondition('moving vertex', [5],
                                 {name: 0. for name in ('ux', 'uy', 'rx', 'ry', 'rz')}))
    if facets==2:
        for node,(x,z) in zip((7,8,9),((-5.,5.),(5.,5.),(0.,-10.))):
            shell.add_node(node,x,-.0751,z)
        shell.add_element(22,create_shell_element(22,(7,8,9),'elastic',thickness=.05,
            formulation='e4-pl-s3-v2d',reference_normal=(0,1,0),reference_surface_offset=0.))
        shell.add_boundary_condition(FixedSupport('vertical edge',[7,9]))
        shell.add_boundary_condition(BoundaryCondition('vertical moving vertex',[8],
            {name:0. for name in ('ux','uz','rx','ry','rz')}))
    # Make every adapter after assembling the full authoritative shell columns.
    skin = S3SkinWorkAdapter.from_model(shell,7,21,'top')
    works = []
    for element in elements:
        rule = NativeRoundSurfaceRule(element.element_id, element.operator.reference, .05, xi_range=(.1, .9),
            angle_range=(pi-.3, pi+.3), axial_order=1, angular_order=1)
        patch = NativeMaterialS3Patch(rule, tuple(element.node_ids), normal_stiffness_Pa_per_m=1e6,
            integration_settings=StraightNormalIntegrationSettings(1e-12, 1e-10, 1e-9, 64, 2.),
            geometry_tolerance_m=1e-10, rim_clearance_m=1e-6)
        works.append(NativeRetainedMaterialS3Work(patch, skin))
    if facets==2:
        element=elements[0]
        vertical_skin=S3SkinWorkAdapter.from_model(shell,8,22,'top')
        vertical_rule=NativeRoundSurfaceRule(element.element_id,element.operator.reference,.05,
            xi_range=(.1,.9),angle_range=(pi/2-.3,pi/2+.3),axial_order=1,angular_order=1)
        vertical_patch=NativeMaterialS3Patch(vertical_rule,tuple(element.node_ids),normal_stiffness_Pa_per_m=1e6,
            integration_settings=StraightNormalIntegrationSettings(1e-12,1e-10,1e-9,64,2.),
            geometry_tolerance_m=1e-10,rim_clearance_m=1e-6)
        works.append(NativeRetainedMaterialS3Work(vertical_patch,vertical_skin))
    if beams==2 or facets==2:
        from anyfesolver.general_contact import NativeS3ContactPair, create_native_contact_analysis
        pairs=tuple(NativeS3ContactPair('beam'+str(work.contact.beam_element_id-1)+
            ('-skin'+str(work.shell.element_id) if facets==2 else ''),work,5e-5,128) for work in works)
        return create_native_contact_analysis(tuple(definitions),tuple(roots),program,shell,pairs,
            equilibrium_tolerance=1e-10)
    return NativeRetainedRoundS3Analysis(tuple(definitions),tuple(roots),program,shell,
        works[0], gap_tolerance_m=5e-5, equilibrium_tolerance=1e-10, maximum_swept_intervals=128)


def main(*, beams=1,facets=1):
    run = analysis(beams=beams,facets=facets).solve()
    if run.status != 'completed':
        raise RuntimeError(run.failure)
    contact = run.accepted.contact.contact
    print(json.dumps(dict(profile=run.profile, status=run.status,
        qualification='implemented/tested development; not engineering accepted',
        completed_targets=run.accepted.completed_targets,
        normal_stiffness_Pa_per_m=1e6, measure='fixed reference material area',
        minimum_gap_lower_bound_m=contact.minimum_gap_m,
        minimum_gap_estimate_m=contact.minimum_gap_estimate_m,
        beam_force_n=contact.beam_force_n.tolist(), shell_force_n=contact.shell_force_n.tolist(),
        beam_moment_nm=contact.beam_moment_nm.tolist(), shell_moment_nm=contact.shell_moment_nm.tolist(),
        wrench_origin_m=(0., 0., 0.), shell_displacement_norm_m=float(np.linalg.norm(run.accepted.shell_displacement)),
        normal_storage_j=contact.potential_j, checkpoint_sha256=run.checkpoint_sha256), indent=2))


if __name__ == '__main__':
    main(beams=2 if '--two-beams' in sys.argv else 1,facets=2 if '--two-facets' in sys.argv else 1)
