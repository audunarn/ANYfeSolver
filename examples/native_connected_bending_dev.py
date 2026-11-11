"""Public connected transverse bending development fixture; source UNEXECUTED."""
import argparse
import json
import numpy as np
from anyfesolver.dynamics import TransientConfig
from anyfesolver.ge_beam3_native import define_beam,SELECTOR as BEAM,EllipsoidalGeneralizedSection
from anyfesolver.ge_beam3_native_connected_dynamics import create_analysis,solve_transient,SELECTOR
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_centered_so3_strain import POLICY as STABLE_SO3_POLICY
from examples import native_connected_bending_reference as reference


def build():
    section=EllipsoidalGeneralizedSection(np.diag((reference.EA,reference.GA,3.e4,30.,20.,reference.EI)),np.eye(6),1.e6,1.)
    inertia=np.diag((reference.RHO_A,reference.RHO_A,reference.RHO_A,reference.JX,reference.JY,reference.JZ))
    coordinates=reference.ORIGIN+np.arange(5)[:,None]*reference.ELL*np.array((1.,0.,0.))
    definitions=tuple(define_beam(BEAM,number,nodes,coordinates[list(indices)],
        np.tile(np.eye(3),(3,1,1)),section,inertia,quadrature=4,arithmetic_policy=STABLE_SO3_POLICY)
        for number,nodes,indices in ((1,(10,20,30),(0,1,2)),(2,(30,40,50),(2,3,4))))
    root=BoundaryCondition('root',[10],dict.fromkeys(('ux','uy','uz','rx','ry','rz'),0.))
    return create_analysis(SELECTOR,definitions,(root,))


def prepare():
    """Independent prospective lowest-mode input; no production eigen recovery."""
    return reference.initial_mode()


def config(owner,mode,steps=4):
    if type(steps) is not int or steps not in (0,2,4):raise ValueError('prepared zero/two/four-step fixture only')
    q=mode['q'];u=np.zeros(len(owner.physical))
    u[:3*owner.n].reshape(owner.n,3)[:,0]=q[:5]
    u[:3*owner.n].reshape(owner.n,3)[:,1]=q[5:10]
    u[3*owner.n:].reshape(2*owner.e,3)[:,2]=q[10:]
    return TransientConfig(dt=mode['dt'],t_end=steps*mode['dt'],initial_displacement=u,
        initial_velocity=np.zeros_like(u),save_every=1)


def run(steps=4,checkpoint=None,expected_sha256=None):
    mode=prepare();owner=build()
    result=solve_transient(owner,config(owner,mode,steps),checkpoint=checkpoint,expected_sha256=expected_sha256)
    return owner,result,mode


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,choices=(0,2,4),default=4)
    args=parser.parse_args();owner,result,mode=run(args.steps);data=result.to_dict()
    print(json.dumps(dict(status=data['status'],steps=data['accepted_steps'],time=data['accepted_time'],
        phase=mode['omega']*data['accepted_time'],amplitude_m=reference.AMP,
        energy_j=data['final']['accounting']['total_energy'],shared_nodes=owner.n,
        reference='RETAINED_AFFINE_HALF_CELL_PLANAR_BENDING; UNQUALIFIED'),allow_nan=False))


if __name__=='__main__':main()
