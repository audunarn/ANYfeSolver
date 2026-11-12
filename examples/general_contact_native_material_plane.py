"""Retained native round material patch: frictionless static development example.

The selected interior material patch is the contact domain. It does not admit
caps, unselected surface collision, friction, dynamics or generic FEModel use.
"""
import json
from math import pi

import numpy as np

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces,Program
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.general_contact.native_material_plane import NativeRoundMaterialPlane
from anyfesolver.general_contact.native_retained_static import NativeRetainedRoundPlaneAnalysis
from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule


def analysis():
    points=np.array(((0.,0.,0.),(1.,0.,0.),(2.,0.,0.)))
    frame=np.array(((1.,0.,0.),(0.,0.,-1.),(0.,1.,0.)))
    reference=CenteredCurvedBeam3ReferenceGeometry(points,np.tile(frame,(3,1,1)))
    section=EllipsoidalGeneralizedSection(np.diag((1000.,400.,350.,.8,1.,1.2)),np.eye(6),1e6,1.)
    element=NativeGeneralizedStaticElement(1,(1,2,3),reference,section,order=4)
    definition=NativeBeamDefinition.capture(element,np.diag((2.,2.,2.,.07,.09,.11)))
    root=BoundaryCondition('clamped',[1],{name:0. for name in ('ux','uy','uz','rx','ry','rz')})
    program=Program((.5,1.),DistributedPattern(LinePattern(()),()),
        NodalDeadForces(((3,0.,0.,-.005),)),max_iterations=24)
    rule=NativeRoundSurfaceRule(1,reference,.05,xi_range=(.1,.9),
        angle_range=(pi-.3,pi+.3),axial_order=1,angular_order=1)
    contact=NativeRoundMaterialPlane(rule,plane_origin_m=(0.,0.,-.0501),plane_normal=(0.,0.,1.),
        normal_stiffness_Pa_per_m=1e6,
        integration_settings=StraightNormalIntegrationSettings(1e-12,1e-10,1e-9,64,2.))
    return NativeRetainedRoundPlaneAnalysis((definition,),(root,),program,contact,
        gap_tolerance_m=5e-5,equilibrium_tolerance=1e-10,maximum_swept_intervals=128)


def main():
    run=analysis().solve()
    if run.status!='completed':raise RuntimeError(run.failure)
    contact=run.accepted.contact
    print(json.dumps(dict(profile=run.profile,status=run.status,
        qualification='implemented/tested development; not engineering accepted',
        normal_stiffness_Pa_per_m=1e6,measure='fixed reference material area',
        completed_targets=run.accepted.completed_targets,
        minimum_gap_lower_bound_m=contact.minimum_gap_m,
        minimum_gap_estimate_m=contact.minimum_gap_estimate_m,
        contact_force_n=contact.beam_force_n.tolist(),contact_moment_nm=contact.beam_moment_nm.tolist(),
        wrench_origin_m=(0.,0.,0.),normal_storage_j=contact.potential_j,
        checkpoint_sha256=run.checkpoint_sha256),indent=2))


if __name__=='__main__':main()
