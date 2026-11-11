"""Finite principal-spin plus force/torque native DEVELOPMENT example.

Source candidate, UNEXECUTED until separate loading-scope authority is granted.
The oracle is independent declared geometry/integrals; this is not C7 acceptance.
"""
import argparse
import json
from dataclasses import asdict, replace
import numpy as np
from anyfesolver.dynamics import TransientConfig
from anyfesolver.ge_beam3_native import (define_beam, SELECTOR as DEFINITION_SELECTOR,
    EllipsoidalGeneralizedSection, LinePattern, DistributedPattern)
from anyfesolver.ge_beam3_native_dynamics import create_analysis, SELECTOR, solve_transient
from examples.native_manufactured_loading_reference import LoadingInputs, endpoint, interval, scales


def build(inputs=LoadingInputs(), *, end_time=None):
    points = np.array(((-inputs.length/2,0.,0.),(0.,0.,0.),(inputs.length/2,0.,0.)),dtype=np.float64)
    section = EllipsoidalGeneralizedSection(np.diag(inputs.stiffness_diagonal),np.eye(6),1.e6,1.)
    inertia = np.diag((inputs.mass_per_length,)*3+(inputs.polar_inertia_per_length,
        inputs.transverse_inertia_y,inputs.transverse_inertia_z))
    definition = define_beam(DEFINITION_SELECTOR,1,(1,2,3),points,np.tile(np.eye(3),(3,1,1)),
        section,inertia,quadrature=4)
    pattern = DistributedPattern(LinePattern(((1,*inputs.force_per_length),)),
        ((1,*inputs.couple_per_length),))
    owner = create_analysis(SELECTOR,(definition,),(),load_pattern=pattern)
    displacement = np.concatenate((np.zeros(9),(inputs.initial_angle,0.,0.)*2))
    velocity = np.concatenate((inputs.initial_translation_velocity*3,(inputs.initial_spin,0.,0.)*2))
    config = TransientConfig(dt=inputs.dt,t_end=inputs.steps*inputs.dt if end_time is None else end_time,
        initial_displacement=displacement,initial_velocity=velocity,save_every=1)
    return owner,config


def run(inputs=LoadingInputs(), *, end_time=None, checkpoint=None, expected_sha256=None):
    owner, config = build(inputs,end_time=end_time)
    result = solve_transient(owner,config,checkpoint=checkpoint,expected_sha256=expected_sha256)
    return owner,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--dt',type=float,default=.0025)
    args=parser.parse_args()
    inputs=replace(LoadingInputs(),steps=args.steps,dt=args.dt)
    _,result=run(inputs)
    row=result.to_dict();reference=endpoint(inputs,float(row['accepted_time']))
    print(json.dumps(dict(status=row['status'],accepted_steps=row['accepted_steps'],
        inputs=asdict(inputs),scales=scales(inputs),reference=reference,
        expected_load_work=interval(inputs,0.,float(row['accepted_time']))['work'],
        actual_final=row['final'],production_qualified=False)))


if __name__=='__main__':
    main()
