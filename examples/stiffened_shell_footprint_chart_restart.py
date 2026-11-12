"""Opt-in elastic footprint chart reset and four-path restart comparison.

Run with an installed ANYfeSolver. This selected-node rigid connection locks all
six pose channels; it is not a distributing footprint, area weld or qualified
physical connection. The existing unrebased example and default remain intact.
"""
from contextlib import ExitStack
from dataclasses import replace
import json

import numpy as np
from anyfesolver import stiffened_shell as ss
from stiffened_shell_nodal_rigid_footprint import definition


def main():
    model = replace(definition(),
        footprint_chart_policy=ss.GE_NODAL_FOOTPRINT_INCREMENT_ORIGIN_CHART_V2)
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    with ExitStack() as resources:
        branches = []
        for _ in range(4):
            owner = ss.create_analysis(definition=model)
            resources.callback(owner.close)
            branches.append(owner)
        original, restored, reset, reset_restored = branches
        prefix = original.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
        saved = original.checkpoint()
        restored.restore(saved); reset.restore(saved)
        event = reset.rebase_rotations(expected_generation=prefix.generation)
        reset_restored.restore(reset.checkpoint())
        results = [owner.solve(actions, (1.,), max_iterations=20, max_backtracks=12)
                   for owner in branches]
        reference = results[0]
        normalized = []
        for result in results:
            assert result.status == "completed" and result.accepted_factors == (.25, .5, 1.)
            assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
            np.testing.assert_allclose(result.beam_positions/2., reference.beam_positions/2., rtol=0., atol=1e-9)
            np.testing.assert_allclose(result.shell_displacement.reshape(-1, 6)[:, :3]/2.,
                reference.shell_displacement.reshape(-1, 6)[:, :3]/2., rtol=0., atol=1e-9)
            for name in ("shell_frames", "beam_frames", "cell_frames"):
                np.testing.assert_allclose(getattr(result, name), getattr(reference, name), rtol=0., atol=1e-9)
            difference = (result.cell_resultants-reference.cell_resultants)/np.array((10.,)*6+(20.,)*12)
            normalized.append(float(np.max(np.abs(difference))))
            assert normalized[-1] <= 1e-9
            for name in ("support_reactions", "multipliers"):
                scale = np.array((10.,)*3+(20.,)*3)
                np.testing.assert_allclose(getattr(result, name).reshape(-1, 6)/scale,
                    getattr(reference, name).reshape(-1, 6)/scale, rtol=0., atol=1e-9)
        groups = [owner.recover(selection=ss.NodalRigidFootprintSelection(("tip-pad",)))[0]
                  for owner in branches]
        for group in groups:
            np.testing.assert_allclose(np.asarray(group.shell_action)+np.asarray(group.beam_action),
                                       0., rtol=0., atol=1e-8)
        print(json.dumps(dict(status="EXPERIMENTAL_NOT_QUALIFIED",
            model_identity=original.preflight()["identity"],
            chart_policy=original.preflight()["footprint_chart_admission_policy"],
            accepted_factors=reference.accepted_factors,
            iterations=[result.iterations for result in results],
            generations=[result.generation for result in results],
            reset_generation=event.generation,
            max_scaled_retained_force_moment_differences=normalized,
            complete_scope="elastic Q4 selected-node footprint; full rotational supports",
            checkpoint_bytes=len(saved)), sort_keys=True))


if __name__ == "__main__":
    main()
