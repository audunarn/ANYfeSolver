"""Experimental gravity with a selected-node rigid footprint and chart restart.

The two-shell/two-beam mass definition is shared with the gravity example. The
beam attaches only through the right shell patch. This intentionally rigid,
six-channel selected-node connection is not an area weld or distributing pad.
Run through an installed ANYfeSolver; this example is not G4 qualification.
"""
from contextlib import ExitStack
from dataclasses import replace
import json

import numpy as np
from anyfesolver import stiffened_shell as ss
from stiffened_shell_ge_gravity import definition as point_definition


def definition():
    return replace(point_definition(), points=(),
        footprints=(ss.NodalRigidFootprint("tip-pad", (2, 3, 6, 5), 105,
                                           ((2, 3, 6, 5),)),),
        footprint_chart_policy=ss.GE_NODAL_FOOTPRINT_INCREMENT_ORIGIN_CHART_V2)


def workflow():
    with ExitStack() as resources:
        original = ss.create_analysis(definition=definition())
        restarted = ss.create_analysis(definition=definition())
        resources.callback(original.close)
        resources.callback(restarted.close)
        gravity = ss.GEBodyLoad(original.preflight()["identity"], (0., 0., -9.81))
        prefix = original.solve(gravity, (.25,), max_iterations=20, max_backtracks=12)
        assert prefix.status == "completed" and prefix.accepted_factors == (.25,)
        event = original.rebase_rotations(expected_generation=prefix.generation)
        saved = original.checkpoint()
        restarted.restore(saved)
        assert restarted.checkpoint() == saved
        final = original.solve(gravity, (.5, 1.), max_iterations=20, max_backtracks=12)
        continued = restarted.solve(gravity, (.5, 1.), max_iterations=20, max_backtracks=12)
        assert final.status == continued.status == "completed"
        assert final.accepted_factors == continued.accepted_factors == (.25, .5, 1.)
        assert original.checkpoint() == restarted.checkpoint()
        assert final.free_equilibrium_ratio <= 1e-8 and final.constraint_max <= 1e-9
        selection = ss.NodalRigidFootprintSelection(("tip-pad",))
        group = original.recover(selection=selection)[0]
        np.testing.assert_allclose(np.asarray(group.shell_action)+np.asarray(group.beam_action),
                                   0., rtol=0., atol=1e-8)
        assert np.linalg.norm(final.multipliers) > 0.
        return dict(status=final.status, experimental=True, production_qualified=False,
            mass_kg=final.body_load.total_mass,
            applied_force_N=final.body_load.applied_force,
            applied_moment_Nm=final.body_load.applied_moment_about_origin,
            accepted_factors=final.accepted_factors,
            equilibrium=final.free_equilibrium_ratio, compatibility=final.constraint_max,
            reset_generation=event.generation, checkpoint_bytes=len(saved),
            attachment="selected-node rigid tip pad; footprint-only declaration")


if __name__ == "__main__":
    print(json.dumps(workflow(), indent=2))
