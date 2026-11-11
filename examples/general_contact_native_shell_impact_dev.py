"""Development workflow: common-stage free native beam and elastic S3.

Explicit small-motion shell profile; two bodies and actual offset skin. No
broader engineering acceptance is implied; scoped installed execution evidence is recorded separately.
"""
import argparse
from hashlib import sha256
import json
import numpy as np
from anyfesolver import (create_shell_element, TransientConfig,
    create_native_transient_shell_analysis, solve_native_transient_shell)
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver.general_contact.native_transient_shell import SELECTOR


def build(*,maximum_penetration_m=.003,maximum_swept_intervals=128,separation=.0505,
          shell_insertion=(4,5,6),shell_side='top',director_polarity=1,maximum_shell_displacement_m=.01):
    points=np.array(((0.,0.,0.),(1.,0.,0.),(2.,0.,0.)))
    frames=np.repeat(np.eye(3)[None],3,axis=0)
    reference=CenteredCurvedBeam3ReferenceGeometry(points,frames)
    section=EllipsoidalGeneralizedSection(np.diag((1e4,3e3,3e3,30.,20.,10.)),np.eye(6),1e6,1.)
    element=NativeGeneralizedStaticElement(1,(1,2,3),reference,section,order=4,
        arithmetic_policy='GE_BEAM3_CENTERED_STABLE_SO3_AND_PRECISE_WORK_DEV_V1')
    definition=NativeBeamDefinition.capture(element,np.diag((2.,2.,2.,.02,.01,.01)))
    shell=FEModel('native_s3_midpoint_dev');shell.add_material('elastic',1e5,.3,density=1.)
    coordinates={4:(-4.,-4.),5:(4.,-4.),6:(0.,6.)}
    for node in shell_insertion:
        x,y=coordinates[node];shell.add_node(node,x,y,-separation-.025)
    shell.add_element(21,create_shell_element(21,(4,5,6),'elastic',thickness=.05,
        formulation='e4-pl-s3-v2d',reference_normal=(0,0,1),reference_surface_offset=0.,director_polarity=director_polarity))
    return create_native_transient_shell_analysis(SELECTOR,(definition,),(),shell,
        shell_body_id=7,shell_element_id=21,shell_side=shell_side,radius_m=.05,
        line_stiffness_n_per_m2=100.,rim_clearance_m=.1,geometry_tolerance_m=1e-10,
        energy_tolerance_j=1e-8,force_tolerance_n=1e-7,tangent_tolerance_n_per_m=1e-6,
        maximum_integration_intervals=1000,maximum_penetration_m=maximum_penetration_m,
        maximum_swept_intervals=maximum_swept_intervals,maximum_shell_displacement_m=maximum_shell_displacement_m,
        maximum_shell_rotation_rad=.05,inertia_order=24)


def config(owner,steps=2,dt=.003):
    displacement=np.zeros(len(owner.physical));velocity=np.zeros_like(displacement)
    index={d:i for i,d in enumerate(owner.physical)}
    for node in range(3):velocity[index[6*node+2]]=-.1
    for node,value in zip(owner.contact.shell.node_ids,(.0002,-.0001,.0003)):
        z=int(owner.shell_model.mesh.dof_manager.get_node_dofs(node)[2])
        displacement[index[42+z]]=value
        velocity[index[42+z]]=.02
    return TransientConfig(t_end=steps*dt,dt=dt,save_every=1,
        initial_displacement=displacement,initial_velocity=velocity)


def run(steps=2,dt=.003,checkpoint=None):
    owner=build();result=solve_native_transient_shell(owner,config(owner,steps,dt),checkpoint=checkpoint,
        expected_sha256=None if checkpoint is None else sha256(checkpoint).hexdigest(),
        max_steps=64,max_cutbacks=8,minimum_dt=dt/256)
    return owner,result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--dt',type=float,default=.003);args=parser.parse_args()
    _,result=run(args.steps,args.dt);print(json.dumps(result.to_dict(),sort_keys=True))


if __name__=='__main__':main()
