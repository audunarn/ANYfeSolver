"""Two free complete native beams in one midpoint contact solve; UNVERIFIED.

Explicit round-centreline symmetric line penalty, no cap-area endpoint force.
Running this example requires separate execution authority in the C8 campaign.
"""
import argparse
from hashlib import sha256
import json
import numpy as np
from anyfesolver.dynamics import TransientConfig
from anyfesolver.ge_beam3_native import define_beam, SELECTOR as BEAM, EllipsoidalGeneralizedSection
from anyfesolver.general_contact.native_transient_pair import (
    SELECTOR, ENDPOINT_POLICY, create_native_transient_pair_analysis,
    solve_native_transient_pair,
)


def definitions(separation=.1005, rise=0.):
    section = EllipsoidalGeneralizedSection(
        np.diag((1e4,3e3,3e3,30.,20.,10.)), np.eye(6), 1e6, 1.)
    inertia = np.diag((2.,2.,2.,.02,.01,.01))
    frames = []
    for slope in (2*rise,0.,-2*rise):
        tangent = np.array((1.,slope,0.))/np.hypot(1.,slope)
        normal = np.array((-tangent[1],tangent[0],0.))
        frames.append(np.column_stack((tangent,normal,(0.,0.,1.))))
    return tuple(define_beam(BEAM, k, nodes,
        np.array(((0., y, 0.), (1., y+rise, 0.), (2., y, 0.)), dtype=np.float64),
        np.array(frames), section, inertia, quadrature=4,
        arithmetic_policy='GE_BEAM3_CENTERED_STABLE_SO3_AND_PRECISE_WORK_DEV_V1')
        for k,nodes,y in ((1,(10,20,30),0.), (2,(40,50,60),separation)))


def build(*, separation=.1005, maximum_penetration_m=.003,
          maximum_swept_intervals=128):
    return create_native_transient_pair_analysis(SELECTOR, definitions(separation), (),
        radius_a_m=.05, radius_b_m=.05, endpoint_policy=ENDPOINT_POLICY,
        line_stiffness_n_per_m2=100., distance_tolerance_m=1e-10,
        feature_tolerance_m=1e-9, search_parameter_tolerance=1e-3,
        maximum_search_depth=20, maximum_search_cells=10000,
        energy_tolerance_j=1e-8, force_tolerance_n=1e-7,
        tangent_tolerance_n_per_m=1e-6, maximum_integration_intervals=1000,
        maximum_penetration_m=maximum_penetration_m,
        maximum_swept_intervals=maximum_swept_intervals, inertia_order=4)


def config(owner, steps=2, dt=.003):
    velocity = np.zeros(len(owner.physical), dtype=np.float64)
    for node in owner.elements[0].node_ids:
        velocity[3*owner.node_index[node]+1] = .1
    for node in owner.elements[1].node_ids:
        velocity[3*owner.node_index[node]+1] = -.1
    return TransientConfig(dt=dt, t_end=steps*dt, initial_velocity=velocity,
                           save_every=1)


def run(steps=2, dt=.003, *, checkpoint=None):
    owner = build()
    result = solve_native_transient_pair(owner, config(owner, steps, dt),
        checkpoint=checkpoint,
        expected_sha256=sha256(checkpoint).hexdigest() if checkpoint is not None else None,
        max_steps=64, max_cutbacks=8, minimum_dt=dt/256)
    return owner, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--dt', type=float, default=.003)
    args = parser.parse_args()
    owner, result = run(args.steps, args.dt)
    data = result.to_dict()
    print(json.dumps(dict(profile=data['profile'], status=data['status'],
        accepted_time=data['accepted_time'], accepted_steps=data['accepted_steps'],
        final_accounting=data['final']['accounting'],
        checkpoint_sha256=sha256(result.checkpoint).hexdigest(),
        checkpoint_bytes=len(result.checkpoint), engineering_accepted=False)))


if __name__ == '__main__':
    main()
