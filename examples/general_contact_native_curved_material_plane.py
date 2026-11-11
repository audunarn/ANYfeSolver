"""Curved retained native material patch: bounded static development example.

One elastic B3-GE beam and one fixed plane; selected interior skin only.
No caps, friction, dynamics, unselected collisions or engineering admission.
Run with an installed ANYfeSolver package on PYTHONPATH.
"""
import json
from math import hypot,pi
import numpy as np

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces,Program
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.general_contact.native_curved_plane_work import NativeCurvedMaterialPlaneWork
from anyfesolver.general_contact.native_retained_static import NativeRetainedRoundPlaneAnalysis
from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule


def analysis(targets=(1.,0.,1.),*,normal_stiffness_Pa_per_m=1e7,max_iterations=24):
    points=[];frames=[]
    for x in np.linspace(0.,2.,3):
        points.append((x,.2*(1-(x-1)**2),0.))
        tangent=np.array((1.,.4*(1-x),0.));tangent/=hypot(*tangent)
        second=np.array((0.,0.,1.))
        frames.append(np.column_stack((tangent,second,np.cross(tangent,second))))
    reference=CenteredCurvedBeam3ReferenceGeometry(np.array(points),np.array(frames))
    section=EllipsoidalGeneralizedSection(np.diag((1000.,400.,350.,.8,1.,1.2)),np.eye(6),1e6,1.)
    element=NativeGeneralizedStaticElement(1,(1,2,3),reference,section,order=4)
    definition=NativeBeamDefinition.capture(element,np.diag((2.,2.,2.,.07,.09,.11)))
    root=BoundaryCondition('clamped',[1],{name:0. for name in ('ux','uy','uz','rx','ry','rz')})
    program=Program(targets,DistributedPattern(LinePattern(()),()),
        NodalDeadForces(((3,0.,0.,-.005),)),max_iterations=max_iterations)
    rule=NativeRoundSurfaceRule(1,reference,.03,xi_range=(.1,.9),
        angle_range=(pi-.3,pi+.3),axial_order=1,angular_order=1)
    contact=NativeCurvedMaterialPlaneWork(rule,plane_origin_m=(0.,0.,-.0301),
        plane_normal=(0.,0.,1.),normal_stiffness_Pa_per_m=normal_stiffness_Pa_per_m,
        integration_settings=StraightNormalIntegrationSettings(1e-10,1e-8,1e-6,64,2.))
    return NativeRetainedRoundPlaneAnalysis((definition,),(root,),program,contact,
        gap_tolerance_m=3e-5,equilibrium_tolerance=1e-10,maximum_swept_intervals=128)


def main():
    run=analysis().solve()
    if run.status!='completed':raise RuntimeError(run.failure)
    rows=json.loads(run.checkpoint)['records']
    print(json.dumps(dict(status=run.status,profile=run.profile,
        qualification='implemented/tested development; not independently accepted or released',
        normal_stiffness_Pa_per_m=1e7,measure='actual fixed reference material area',
        wrench_origin_m=(0.,0.,0.),gap_kind='continuous whole-material-patch lower bound',
        targets=[dict(target=row['target'],contact=row['contact']) for row in rows],
        checkpoint_sha256=run.checkpoint_sha256),indent=2))


if __name__=='__main__':main()
