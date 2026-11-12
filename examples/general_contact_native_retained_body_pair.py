"""Development example: two connected two-element physical-fibre bodies."""
import numpy as np
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_fibre_section import Fibre, FlowCurve, PhysicalFibreSection
from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces, Program
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.general_contact.native_retained_body_static import NativeRetainedBodyPairAnalysis


def analysis(*,elements_per_body=2,separation=.100001,force=-.001,targets=(.05,),max_coordinates=256,reference_rise_m=0.):
    curve=FlowCurve((0.,),(1e6,),1.)
    fibres=tuple(Fibre(str(i),y,z,.00025,1e6,curve) for i,(y,z) in enumerate(
        ((.03,0.),(-.03,0.),(0.,.03),(0.,-.03))))
    section=PhysicalFibreSection(fibres,np.diag(np.sqrt((0.,400.,350.,.8,.55,.75))))
    inertia=np.diag((2.,2.,2.,.07,.09,.11));definitions=[];bodies=[]
    for side,y in enumerate((0.,separation)):
        chain=[];offset=side*(2*elements_per_body+1)
        for e in range(elements_per_body):
            x=np.linspace(2*e/elements_per_body,2*(e+1)/elements_per_body,3)
            z=y+reference_rise_m*(1-(x-1)**2)
            frames=[]
            for slope in 2*reference_rise_m*(1-x):
                tangent=np.array((1.,slope,0.))/np.hypot(1.,slope)
                frames.append(np.column_stack((tangent,(-tangent[1],tangent[0],0.),(0.,0.,1.))))
            ref=Reference(np.column_stack((x,z,np.zeros(3))),np.array(frames))
            eid=11+side*elements_per_body+e;nodes=tuple(offset+2*e+j+1 for j in range(3))
            element=NativeFibreStaticElement(eid,nodes,ref,section,order=4)
            definitions.append(NativeBeamDefinition.capture(element,inertia));chain.append(eid)
        bodies.append(tuple(chain))
    fixed={name:0. for name in ('ux','uy','uz','rx','ry','rz')}
    boundaries=tuple(BoundaryCondition(str(side),[side*(2*elements_per_body+1)+1],fixed) for side in (0,1))
    program=Program(targets,DistributedPattern(LinePattern(()),()),
        NodalDeadForces(((2*(2*elements_per_body+1),0.,force,0.),)),max_iterations=24)
    return NativeRetainedBodyPairAnalysis(tuple(definitions),boundaries,program,tuple(bodies),max_coordinates=max_coordinates)
