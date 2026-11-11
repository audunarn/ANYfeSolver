"""Headless experimental selected-node footprint; no weld qualification.

Run against an installed ANYfeSolver with optional --finite. The selected right
Q4 cell's four vertices share a rigid pose at beam105. The left cell transfers
load to the clamped shell root. All native B3 retained coordinates remain.
"""
import argparse
import json
import numpy as np
from anyfesolver import b3_ge, stiffened_shell as ss


def definition():
    e = 210e9; nu = .3; g = e/(2.*(1.+nu))
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*.01, g*.008, g*.008, g*1.6666666666666667e-5,
                 e*8.333333333333333e-6, e*8.333333333333333e-6)),
        np.eye(6), 1e20, 1e9)
    coordinates = np.array([(x, -.5, .1) for x in (0., .5, 1., 1.5, 2.)])
    beams = tuple(b3_ge.define_beam(
        "b3-ge", i+1, (101+2*i, 102+2*i, 103+2*i),
        coordinates[2*i:2*i+3], np.tile(np.eye(3), (3, 1, 1)),
        section, np.eye(6)) for i in range(2))
    shell = ss.ShellPanel((1, 2, 3, 4, 5, 6),
        ((0., -.5, 0.), (1., -.5, 0.), (2., -.5, 0.),
         (0., .5, 0.), (1., .5, 0.), (2., .5, 0.)),
        ((1, 2, 5, 4), (2, 3, 6, 5)), .07, e, nu, 7850.)
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        shell, beams, (ss.PointConnection(1, 101),),
        tuple(ss.FixedCoordinate("shell", node, axis)
              for node in (1, 4) for axis in range(6)),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        footprints=(ss.NodalRigidFootprint("tip-pad", (2, 3, 6, 5), 105,
                                         ((2, 3, 6, 5),)),))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finite", action="store_true")
    args = parser.parse_args()
    model = definition(); owner = ss.create_analysis(definition=model)
    resumed = ss.create_analysis(definition=model)
    try:
        # Finite point ownership admits translation dead loads. Linear reference
        # work additionally exercises a directly applied physical moment.
        actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0. if args.finite else 1., 0.)),)
        result = (owner.solve(actions, (.25, .5, 1.), max_iterations=20, max_backtracks=12)
                  if args.finite else owner.solve_linear(actions))
        group, = owner.recover(selection=ss.NodalRigidFootprintSelection(("tip-pad",)))
        saved = owner.checkpoint(); restored = resumed.restore(saved)
        recovered, = resumed.recover(selection=ss.NodalRigidFootprintSelection(("tip-pad",)))
        scale = np.array((10.,)*3+(20.,)*3)
        np.testing.assert_allclose(np.array(recovered.shell_action)/scale,
                                   np.array(group.shell_action)/scale, rtol=0., atol=1e-9)
        assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
        print(json.dumps(dict(status="EXPERIMENTAL_NOT_QUALIFIED",
            analysis="finite" if args.finite else "reference_linear",
            model_identity=owner.preflight()["identity"], constraints=owner.preflight()["constraints"],
            footprint=group.footprint_id, law=group.law, shell_nodes=group.shell_node_ids,
            complete_patch=model.footprints[0].patch_cells,
            normal_spin="locked by the six-channel rigid pose law",
            common_origin_m=group.common_origin_m,
            shell_exerted_wrench=group.shell_action, beam_exerted_wrench=group.beam_action,
            per_node_records=len(group.point_wrenches), generation=restored.generation,
            equilibrium_ratio=result.free_equilibrium_ratio, constraint_max=result.constraint_max,
            checkpoint_bytes=len(saved)), sort_keys=True))
    finally:
        owner.close(); resumed.close()


if __name__ == "__main__":
    main()
