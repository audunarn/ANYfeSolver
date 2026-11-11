"""Public native midpoint trajectory DEVELOPMENT example.

Scoped source/private-installed verification is recorded in the guide; no C7
acceptance is supplied. The
existing one-element native definition fixture is reused, not reinterpreted.
"""
import argparse
import json
import numpy as np

from anyfesolver.boundary import BoundaryCondition
from anyfesolver.dynamics import TransientConfig
from anyfesolver.ge_beam3_native import LinePattern, DistributedPattern
from anyfesolver.ge_beam3_native_dynamics import create_analysis, SELECTOR, solve_transient
from examples.native_contact_free_finite_velocity_dev import build_definition, initial_velocity


def build(*, clamped=False, loaded=False, dt=.001, t_end=.002, save_every=1):
    """Same authoritative fixture, optionally homogeneous clamp/constant load."""
    if type(clamped) is not bool or type(loaded) is not bool:
        raise ValueError('exact boolean example options required')
    definition, points = build_definition()
    boundaries = (() if not clamped else (BoundaryCondition('native_first_clamp', [1],
        dict(ux=0., uy=0., uz=0., rx=0., ry=0., rz=0.)),))
    load = DistributedPattern(LinePattern(((1, 0., -.01, 0.),)), ()) if loaded else None
    owner = create_analysis(SELECTOR, (definition,), boundaries, load_pattern=load)
    velocity = initial_velocity(points)
    if clamped:
        velocity[:3] = 0.
    config = TransientConfig(dt=dt, t_end=t_end, save_every=save_every,
                             initial_velocity=np.array(velocity, dtype=np.float64))
    return owner, config


def run(*, clamped=False, loaded=False, dt=.001, t_end=.002, save_every=1,
        checkpoint=None, expected_sha256=None, cancellation_token=None):
    owner, config = build(clamped=clamped, loaded=loaded, dt=dt, t_end=t_end,
                          save_every=save_every)
    result = solve_transient(owner, config, checkpoint=checkpoint,
        expected_sha256=expected_sha256, cancellation_token=cancellation_token)
    return owner, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clamped', action='store_true')
    parser.add_argument('--loaded', action='store_true')
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--t-end', type=float, default=.002)
    args = parser.parse_args()
    _, result = run(clamped=args.clamped, loaded=args.loaded, dt=args.dt, t_end=args.t_end)
    row = result.to_dict()
    print(json.dumps(dict(status=row['status'], profile=row['profile'],
        accepted_steps=row['accepted_steps'], time_seconds=row['accepted_time'],
        total_energy_j=row['final']['accounting']['total_energy'],
        production_qualified=False)))


if __name__ == '__main__':
    main()
