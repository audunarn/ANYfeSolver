"""Actual torque-free elastic nonprincipal native DEVELOPMENT trajectory."""
import argparse
import json
import math
import numpy as np
from anyfesolver.dynamics import TransientConfig
from anyfesolver.ge_beam3_native import define_beam,SELECTOR as BEAM,EllipsoidalGeneralizedSection
from anyfesolver.ge_beam3_native_dynamics import create_analysis,SELECTOR,solve_transient
from anyfesolver._ge_beam3_centered_so3_strain import POLICY
from examples import native_nonprincipal_reference as ref


def build(*,steps=ref.STEPS,dt=ref.DT):
    if type(steps) is not int or not 0<=steps<=4*ref.STEPS:raise ValueError('example step inventory')
    if not math.isfinite(dt) or dt not in (ref.DT,ref.DT/2,ref.DT/4):raise ValueError('example frozen timestep inventory')
    if steps*dt>ref.STEPS*ref.DT:raise ValueError('example frozen horizon')
    section=EllipsoidalGeneralizedSection(np.diag((ref.EA,ref.GA,ref.GA,ref.EI,ref.EI,ref.EI)),np.eye(6),1.e6,1.)
    inertia=np.diag((ref.RHO,ref.RHO,ref.RHO,*ref.JDENSITY))
    definition=define_beam(BEAM,1,(1,2,3),ref.POINTS,np.tile(np.eye(3),(3,1,1)),section,inertia,
                           quadrature=4,arithmetic_policy=POLICY)
    owner=create_analysis(SELECTOR,(definition,),(),stage_coordinates='acceleration')
    velocity=np.r_[np.cross(np.mean(ref.OMEGA,axis=0),ref.POINTS).ravel(),ref.OMEGA.ravel()]
    config=TransientConfig(dt=dt,t_end=steps*dt,initial_velocity=velocity)
    return owner,config


def run(*,steps=ref.STEPS,dt=ref.DT,checkpoint=None,expected_sha256=None):
    owner,config=build(steps=steps,dt=dt)
    return owner,solve_transient(owner,config,checkpoint=checkpoint,expected_sha256=expected_sha256)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,default=ref.STEPS)
    p.add_argument('--dt',type=float,default=ref.DT);args=p.parse_args()
    _,result=run(steps=args.steps,dt=args.dt);data=result.to_dict()
    final=data.get('final');accounting={} if final is None else final['accounting']
    print(json.dumps(dict(status=data['status'],accepted_steps=data['accepted_steps'],
        time_seconds=data['accepted_time'],total_energy_j=accounting.get('total_energy'),
        scope='FREE_ELASTIC_NONPRINCIPAL_NOT_RIGID_EULER_TOP',production_qualified=False),allow_nan=False))
    return 0 if result.success else 1


if __name__=='__main__':raise SystemExit(main())
