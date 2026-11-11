"""UNEXECUTED four-beam/two-shell-body common-midpoint development fixture."""
import argparse
from hashlib import sha256
import json
import numpy as np
from anyfesolver import create_shell_element, TransientConfig
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver.general_contact.native_transient_multi_shell import SELECTOR
from anyfesolver.general_contact import (NativeContactBody,S3ContactBody,
    NativeTransientS3ContactPair,create_native_transient_multi_shell_analysis,solve_native_transient_multi_shell)


FORCE_REFERENCE_N=5.
MOMENT_REFERENCE_NM=5.


def selections(separation=.0505):
    beams=[]
    # First body diagonal separates faces. Each whole beam is strictly inside
    # its selected triangle; two independent beams load the same first face.
    for i,(x,y) in enumerate(((0.,-3.),(1.,-4.),(-3.,2.),(15.,0.)),1):
        points=np.array(((x,y,0.),(x+.5,y,0.),(x+1.,y,0.)))
        reference=CenteredCurvedBeam3ReferenceGeometry(points,np.repeat(np.eye(3)[None],3,axis=0))
        section=EllipsoidalGeneralizedSection(np.diag((1e4,3e3,3e3,30.,20.,10.)),np.eye(6),1e6,1.)
        element=NativeGeneralizedStaticElement(i,tuple(range(3*i-2,3*i+1)),reference,section,order=4,
            arithmetic_policy='GE_BEAM3_CENTERED_STABLE_SO3_AND_PRECISE_WORK_DEV_V1')
        beams.append(NativeContactBody(i,(NativeBeamDefinition.capture(element,np.diag((2.,2.,2.,.02,.01,.01))),)))
    first=FEModel('shared_two_face_S3_body');first.add_material('elastic',1e5,.3,density=1.)
    for node,x,y in ((6,-6.,6.),(4,-6.,-6.),(7,6.,6.),(5,6.,-6.)):
        first.add_node(node,x,y,-separation-.025)
    for eid,nodes in ((21,(4,5,7)),(22,(4,7,6))):
        first.add_element(eid,create_shell_element(eid,nodes,'elastic',thickness=.05,formulation='e4-pl-s3-v2d',
            reference_normal=(0,0,1),reference_surface_offset=0.,director_polarity=1))
    second=FEModel('second_S3_body');second.add_material('elastic',1e5,.3,density=1.)
    for node,x,y in ((5,22.,-4.),(6,16.,6.),(4,10.,-4.)):
        second.add_node(node,x,y,-separation-.025)
    second.add_element(21,create_shell_element(21,(4,5,6),'elastic',thickness=.05,formulation='e4-pl-s3-v2d',
        reference_normal=(0,0,1),reference_surface_offset=0.,director_polarity=1))
    bodies=(S3ContactBody(7,first),S3ContactBody(8,second))
    pairs=(NativeTransientS3ContactPair(1,1,7,21),NativeTransientS3ContactPair(2,2,7,21),
           NativeTransientS3ContactPair(3,3,7,22),NativeTransientS3ContactPair(4,4,8,21))
    return tuple(beams),bodies,pairs


def build(*,separation=.0505,maximum_geometry_evaluations=4096,maximum_integration_intervals=1000,
          maximum_swept_intervals=128,selections_override=None):
    native,shell,pairs=selections(separation) if selections_override is None else selections_override
    return create_native_transient_multi_shell_analysis(SELECTOR,native,shell,pairs,radius_m=.05,
        line_stiffness_n_per_m2=100.,patch_boundary_clearance_m=.1,geometry_tolerance_m=1e-10,
        energy_tolerance_j=1e-8,force_tolerance_n=1e-7,tangent_tolerance_n_per_m=1e-6,
        maximum_geometry_evaluations=maximum_geometry_evaluations,
        maximum_integration_intervals=maximum_integration_intervals,maximum_penetration_m=.003,
        maximum_swept_intervals=maximum_swept_intervals,maximum_shell_displacement_m=.01,
        maximum_shell_rotation_rad=.05,inertia_order=24)


def config(owner,steps=2,dt=.003):
    displacement=np.zeros(len(owner.p));velocity=np.zeros_like(displacement);index={d:i for i,d in enumerate(owner.p)}
    for node in range(owner.n):velocity[index[6*node+2]]=-.1
    for o,_,p,_ in owner.shell_layout:
        for local in p:
            if local%6==2:velocity[index[owner.native_size+o+local]]=.02
    return TransientConfig(t_end=steps*dt,dt=dt,save_every=1,initial_displacement=displacement,initial_velocity=velocity)


def run(steps=2,dt=.003,checkpoint=None):
    owner=build();result=solve_native_transient_multi_shell(owner,config(owner,steps,dt),checkpoint=checkpoint,
        expected_sha256=None if checkpoint is None else sha256(checkpoint).hexdigest(),
        max_steps=64,max_cutbacks=8,minimum_dt=dt/256)
    return owner,result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--dt',type=float,default=.003)
    parser.add_argument('--checkpoint-file');parser.add_argument('--result-out');parser.add_argument('--checkpoint-out')
    args=parser.parse_args()
    from pathlib import Path
    checkpoint=None if args.checkpoint_file is None else Path(args.checkpoint_file).read_bytes()
    _,result=run(args.steps,args.dt,checkpoint=checkpoint)
    if args.result_out:Path(args.result_out).write_bytes(result._payload)
    if args.checkpoint_out:Path(args.checkpoint_out).write_bytes(result.checkpoint)
    print(json.dumps(result.to_dict(),sort_keys=True))


if __name__=='__main__':main()
