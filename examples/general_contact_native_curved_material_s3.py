"""Curved retained B3-GE / deformable S3 interior: static development example."""
import json
import argparse
from hashlib import sha256
from pathlib import Path
from math import hypot,pi
import numpy as np
from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition,FixedSupport
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_nodal_loading import NodalDeadForces,Program
from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings
from anyfesolver.general_contact.native_material_s3 import NativeMaterialS3Patch
from anyfesolver.general_contact.native_curved_s3_work import NativeCurvedMaterialS3Work
from anyfesolver.general_contact.s3_skin_work import S3SkinWorkAdapter
from anyfesolver.general_contact import NativeS3ContactPair,create_native_contact_analysis


def analysis(targets=(1.,0.,1.),*,normal_stiffness_Pa_per_m=1e7,two_facets=False,two_beams=False,physical_fibres=False,plastic_fibres=False,
             integration_budget_policy='full_reference',root_tangent_enclosure_policy='spectral'):
    if plastic_fibres and not physical_fibres:
        raise ValueError('plastic fibre fixture requires the physical-fibre section family')
    points=[];frames=[]
    for x in np.linspace(0.,2.,3):
        points.append((x,.2*(1-(x-1)**2),0.))
        tangent=np.array((1.,.4*(1-x),0.));tangent/=hypot(*tangent)
        second=np.array((0.,0.,1.));frames.append(np.column_stack((tangent,second,np.cross(tangent,second))))
    reference=CenteredCurvedBeam3ReferenceGeometry(np.array(points),np.array(frames))
    if physical_fibres:
        from anyfesolver._ge_beam3_fibre_section import Fibre,FlowCurve,PhysicalFibreSection
        from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
        curve=FlowCurve.linear(2.,1e5) if plastic_fibres else FlowCurve((0.,),(1e6,),1.)
        fibres=tuple(Fibre(str(i),y,z,.00025,1e6,curve) for i,(y,z) in enumerate(
            ((.03,0.),(-.03,0.),(0.,.03),(0.,-.03))))
        section=PhysicalFibreSection(fibres,np.diag(np.sqrt((0.,400.,350.,.8,.55,.75))))
        element=NativeFibreStaticElement(1,(1,2,3),reference,section,order=4)
    else:
        section=EllipsoidalGeneralizedSection(np.diag((1000.,400.,350.,.8,1.,1.2)),np.eye(6),1e6,1.)
        element=NativeGeneralizedStaticElement(1,(1,2,3),reference,section,order=4)
    definition=NativeBeamDefinition.capture(element,np.diag((2.,2.,2.,.07,.09,.11)))
    root=BoundaryCondition('clamped',[1],{name:0. for name in ('ux','uy','uz','rx','ry','rz')})
    beams=[(1,(1,2,3),reference)];definitions=[definition];roots=[root]
    forces=[(3,0.,0.,-.005)]
    if two_beams:
        shifted=np.array(points);shifted[:,1]+=.25
        second_reference=CenteredCurvedBeam3ReferenceGeometry(shifted,np.array(frames))
        second_element=type(element)(2,(8,9,10),second_reference,section,order=4)
        definitions.append(NativeBeamDefinition.capture(second_element,np.diag((2.,2.,2.,.07,.09,.11))))
        beams.append((2,(8,9,10),second_reference))
        roots.append(BoundaryCondition('second clamped',[8],{name:0. for name in ('ux','uy','uz','rx','ry','rz')}))
        forces.append((10,0.,0.,-.003))
    program=Program(targets,DistributedPattern(LinePattern(()),()),NodalDeadForces(tuple(forces)),max_iterations=24)
    shell=FEModel('curved material and deformable S3');shell.add_material('elastic',1e8,.3,density=1.)
    if two_facets:
        coordinates={4:(1.,-5.),5:(1.,5.),6:(-5.,0.),7:(7.,0.)}
        facets=((21,(6,4,5),(-.9,-.1)),(22,(4,7,5),(.1,.9)))
    else:
        coordinates={4:(-5.,-5.),5:(5.,-5.),6:(0.,10.)}
        facets=((21,(4,5,6),(.1,.9)),)
    for node,xy in coordinates.items():shell.add_node(node,*xy,-.0551)
    for eid,nodes,_ in facets:
        shell.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.05,
            formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=0.))
    shell.add_boundary_condition(FixedSupport('edge',[n for n in coordinates if n!=5]))
    shell.add_boundary_condition(BoundaryCondition('moving vertex',[5],
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    pairs=[]
    for beam_id,beam_nodes,beam_reference in beams:
        for eid,_,interval in facets:
            skin=S3SkinWorkAdapter.from_model(shell,7,eid,'top')
            rule=NativeRoundSurfaceRule(beam_id,beam_reference,.03,xi_range=interval,angle_range=(pi-.3,pi+.3),axial_order=1,angular_order=1)
            patch=NativeMaterialS3Patch(rule,beam_nodes,normal_stiffness_Pa_per_m=normal_stiffness_Pa_per_m,
                integration_settings=StraightNormalIntegrationSettings(1e-10,1e-8,1e-6,64,2.),
                geometry_tolerance_m=1e-10,rim_clearance_m=1e-6)
            name=f'curved material / top {eid}' if two_facets else 'curved material / top'
            if two_beams:name=f'beam {beam_id} / {name}'
            pairs.append(NativeS3ContactPair(name,NativeCurvedMaterialS3Work(patch,skin,
                integration_budget_policy=integration_budget_policy,
                root_tangent_enclosure_policy=root_tangent_enclosure_policy),3e-5,128))
    return create_native_contact_analysis(tuple(definitions),tuple(roots),program,shell,tuple(pairs),equilibrium_tolerance=1e-10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets',type=float,nargs='+',default=(1.,0.,1.),help='explicit load targets; default press/open/recontact')
    parser.add_argument('--normal-stiffness',type=float,default=1e7,help='explicit surface normal regularization in Pa/m')
    parser.add_argument('--integration-budget-policy',choices=('full_reference','certified_active'),default='full_reference')
    parser.add_argument('--root-tangent-enclosure-policy',choices=('spectral','componentwise','anchored'),default='spectral')
    parser.add_argument('--through-target',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--restart',type=Path)
    parser.add_argument('--checkpoint-out',type=Path)
    parser.add_argument('--two-facets',action='store_true',help='two explicit disjoint source patches against connected S3 facets')
    parser.add_argument('--two-beams',action='store_true',help='two separately clamped curved beams sharing the deformable target')
    parser.add_argument('--physical-fibres',action='store_true',help='declared axial/biaxial physical-fibre section and elastic background')
    parser.add_argument('--plastic-fibres',action='store_true',help='physical fibres with explicit 2 Pa yield stress and 1e5 Pa linear hardening')
    parser.add_argument('--recover-material',type=Path,help='recover the accepted --restart checkpoint to material-results JSON without solving another target')
    parser.add_argument('--material-stations',action='store_true',help='include optional station and fibre fields in material recovery')
    args=parser.parse_args()
    settings=dict(targets=tuple(args.targets),normal_stiffness_Pa_per_m=args.normal_stiffness,
        two_facets=args.two_facets,two_beams=args.two_beams,
        physical_fibres=args.physical_fibres or args.plastic_fibres,plastic_fibres=args.plastic_fibres,
        integration_budget_policy=args.integration_budget_policy,
        root_tangent_enclosure_policy=args.root_tangent_enclosure_policy)
    raw=None if args.restart is None else args.restart.read_bytes()
    if args.material_stations and args.recover_material is None:
        parser.error('--material-stations requires --recover-material')
    if args.recover_material is not None:
        if raw is None or not (args.physical_fibres or args.plastic_fibres):
            parser.error('--recover-material requires --restart and a physical-fibre fixture')
        owner=analysis(**{**settings,'physical_fibres':True})
        report=owner.recover_material(raw,sha256(raw).hexdigest(),include_stations=args.material_stations)
        args.recover_material.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(status='recovered',profile=owner.profile,completed_targets=report['completed_targets'],
            checkpoint_sha256=report['checkpoint_sha256'],material_results=str(args.recover_material)),indent=2))
        return
    if args.through_target>1 and raw is None:
        parser.error('--restart is required for subsequent bounded stages')
    if args.through_target>len(args.targets):
        parser.error('--through-target exceeds the explicit target count')
    if raw is not None and json.loads(raw)['completed_targets']!=args.through_target-1:
        parser.error('restart must precede the requested target by exactly one stage')
    run=analysis(**settings).solve(checkpoint=raw,expected_sha256=None if raw is None else sha256(raw).hexdigest(),
        stop_after=args.through_target)
    if run.status not in ('paused','completed'):raise RuntimeError(run.failure)
    if args.checkpoint_out is not None:args.checkpoint_out.write_bytes(run.checkpoint)
    print(json.dumps(dict(status=run.status,profile=run.profile,completed_targets=run.accepted.completed_targets,
        qualification='development only; not independently accepted or released',
        targets=[dict(target=row['target'],contact=row['contact']) for row in json.loads(run.checkpoint)['records']],
        checkpoint_sha256=run.checkpoint_sha256),indent=2))


if __name__=='__main__':main()
