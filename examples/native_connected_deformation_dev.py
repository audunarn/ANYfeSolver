"""Public connected clamped axial vibration candidate; source UNEXECUTED."""
import argparse
import json
import numpy as np
from anyfesolver.dynamics import TransientConfig
from anyfesolver.ge_beam3_native import define_beam,SELECTOR as BEAM,EllipsoidalGeneralizedSection
from anyfesolver.ge_beam3_native_connected_dynamics import create_analysis,solve_transient,SELECTOR
from anyfesolver.boundary import BoundaryCondition
from examples import native_connected_deformation_reference as reference


def build():
    section=EllipsoidalGeneralizedSection(np.diag((reference.EA,3.e4,3.e4,30.,20.,10.)),np.eye(6),1.e6,1.)
    inertia=np.diag((reference.RHO_A,reference.RHO_A,reference.RHO_A,.02,.01,.01))
    coordinates=reference.ORIGIN+np.arange(5)[:,None]*reference.ELL*reference.DIRECTION
    definitions=tuple(define_beam(BEAM,number,nodes,coordinates[list(indices)],
        np.tile(reference.FRAME,(3,1,1)),section,inertia,quadrature=4)
        for number,nodes,indices in ((1,(10,20,30),(0,1,2)),(2,(30,40,50),(2,3,4))))
    root=BoundaryCondition('root',[10],dict.fromkeys(('ux','uy','uz','rx','ry','rz'),0.))
    return create_analysis(SELECTOR,definitions,(root,))


def config(owner,steps=4,dt=reference.DT):
    q,v=reference.initial();u=np.zeros(len(owner.physical));speed=np.zeros_like(u)
    u[:3*owner.n]=(q[:,None]*reference.DIRECTION).ravel()
    speed[:3*owner.n]=(v[:,None]*reference.DIRECTION).ravel()
    return TransientConfig(dt=dt,t_end=steps*dt,initial_displacement=u,initial_velocity=speed,save_every=1)


def run(steps=4,dt=reference.DT,checkpoint=None,expected_sha256=None):
    owner=build()
    result=solve_transient(owner,config(owner,steps,dt),checkpoint=checkpoint,expected_sha256=expected_sha256)
    return owner,result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=4)
    args=parser.parse_args();owner,result=run(args.steps)
    data=result.to_dict()
    print(json.dumps(dict(status=data['status'],steps=data['accepted_steps'],
        time=data['accepted_time'],elastic_energy=data['final']['accounting']['elastic_energy'],
        shared_nodes=owner.n,reference='FOUR_AFFINE_P1_HALF_CELLS; UNQUALIFIED'),allow_nan=False))


if __name__=='__main__':main()
