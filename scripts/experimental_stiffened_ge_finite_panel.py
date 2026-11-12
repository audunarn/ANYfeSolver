"""Experimental multi-stiffener finite panel using the full Q4 work pullback.

Matching eccentric point bonds only; this is not line/footprint qualification,
a nonlinear capacity model, or the registered practical-scale acceptance run.
The historical rotated-force policy is retained for compatibility, but is not
used here because its finite physical moment balance is not established.
Run with ``python -m scripts.experimental_stiffened_ge_finite_panel``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time

import numpy as np
from anyfesolver import stiffened_shell as ss
from scripts.experimental_stiffened_ge_multi_member_panel import make_definition, actions_for


def physical_balance(definition, result, actions):
    """Current-configuration balance for this fully clamped-root fixture."""
    support = result.support_reactions[:6*len(definition.shell.node_ids)].reshape(-1, 6)
    positions = np.asarray(definition.shell.coordinates_m) + result.shell_displacement.reshape(-1, 6)[:, :3]
    force = support[:, :3].sum(axis=0)
    moment = (support[:, 3:]+np.cross(positions, support[:, :3])).sum(axis=0)
    # This fixture uses canonical ascending model IDs for the public beam rows.
    beam_ids = sorted({node for beam in definition.beams
                       for node in json.loads(beam.raw)["node_ids"]})
    beam_positions = dict(zip(beam_ids, result.beam_positions))
    scale = 0.
    for action in actions:
        applied = result.load_factor*np.asarray(action.wrench)
        force += applied[:3]
        moment += applied[3:]+np.cross(beam_positions[action.node], applied[:3])
        scale += np.linalg.norm(applied[:3])
    return float(np.linalg.norm(force)/scale), float(np.linalg.norm(moment)/(2.*scale))


def run(cells_per_stiffener=4, stiffeners=3):
    start = time.perf_counter()
    definition = replace(make_definition(cells_per_stiffener, stiffeners),
                         q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    actions = actions_for(definition)
    owner = ss.create_analysis(definition=definition)
    resumed = None
    try:
        first = owner.solve(actions, (.01, .02), max_iterations=8, max_backtracks=4)
        assert first.status == "completed" and not first.production_qualified
        saved = owner.checkpoint()
        resumed = ss.create_analysis(definition=definition)
        resumed.restore(saved)
        assert resumed.checkpoint() == saved
        direct = owner.solve(actions, (.03,), max_iterations=8, max_backtracks=4)
        restarted = resumed.solve(actions, (.03,), max_iterations=8, max_backtracks=4)
        for field in ("shell_displacement", "beam_positions", "beam_frames",
                      "cell_frames", "cell_resultants", "multipliers", "support_reactions"):
            np.testing.assert_array_equal(getattr(direct, field), getattr(restarted, field))
        assert owner.checkpoint() == resumed.checkpoint()
        force, moment = physical_balance(definition, direct, actions)
        assert direct.free_equilibrium_ratio <= 1e-8 and direct.constraint_max <= 1e-9
        assert force <= 1e-8 and moment <= 1e-8, (force, moment)
        return dict(status=direct.status, production_qualified=False,
            q4_policy=definition.q4_pullback_policy,
            shell_elements=len(definition.shell.q4_elements), beam_elements=len(definition.beams),
            accepted_factors=direct.accepted_factors,
            equilibrium=direct.free_equilibrium_ratio, compatibility=direct.constraint_max,
            force_balance=force, moment_balance=moment, exact_restart=True,
            checkpoint_bytes=len(saved), elapsed_seconds=time.perf_counter()-start)
    finally:
        owner.close()
        if resumed is not None:
            resumed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-per-stiffener", type=int, default=4)
    parser.add_argument("--stiffeners", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.cells_per_stiffener, args.stiffeners), indent=2))
