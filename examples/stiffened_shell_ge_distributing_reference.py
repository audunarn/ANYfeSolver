"""Experimental reference-linear distributing connection and checkpoint recovery.

Three flat Cartesian Q4 cells carry a native B3 member through a rectangular
two-cell area. Mean directors transmit roll/pitch; in-plane force couples
transmit yaw without shell drilling actions. This is a reduced distributing
idealization, not a pointwise weld or a qualified physical connection.
Finite steps, plasticity, spectra and non-nodal load wrappers are not admitted.
"""
import json

import numpy as np
from anyfesolver import b3_ge, stiffened_shell as ss


def definition():
    e, nu = 210e9, .3
    g = e/(2*(1+nu))
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*.01, g*.008, g*.008, g*1.6666666666666667e-5,
                 e*8.333333333333333e-6, e*8.333333333333333e-6)),
        np.eye(6), 1e20, 1e9)
    beam = b3_ge.define_beam("b3-ge", 1, (101, 102, 103),
        np.array(((1., 0., .1), (2., 0., .1), (3., 0., .1))),
        np.tile(np.eye(3), (3, 1, 1)), section, np.eye(6))
    shell = ss.ShellPanel(tuple(range(1, 9)),
        tuple((x, y, 0.) for y in (-.5, .5) for x in (0., 1., 2., 3.)),
        ((1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7)), .07, e, nu, 7850.)
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        shell, (beam,), (),
        tuple(ss.FixedCoordinate("shell", node, axis)
              for node in (1, 5) for axis in range(6)),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        distributing_footprints=(ss.GEReferenceDistributingFootprint(
            "area-pad", (2, 3), 102, (1., 3., -.5, .5)),))


def main():
    model = definition()
    owner = ss.create_analysis(definition=model)
    resumed = ss.create_analysis(definition=model)
    try:
        actions = (ss.NodalAction("shell", 2, (0., 0., -100., 0., 0., 0.)),
                   ss.NodalAction("shell", 4, (0., 0., 100., 0., 0., 0.)),
                   ss.NodalAction("beam", 103, (0., 20., 0., 0., 0., 3.)))
        result = owner.solve_linear(actions)
        selection = ss.GEReferenceDistributingFootprintSelection(("area-pad",))
        group, = owner.recover(selection=selection)
        checkpoint = owner.checkpoint()
        restored = resumed.restore(checkpoint)
        recovered, = resumed.recover(selection=selection)
        np.testing.assert_allclose(recovered.shell_action, group.shell_action, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(restored.beam_coordinates, result.beam_coordinates, rtol=1e-10, atol=1e-12)
        second = owner.solve_linear(actions)
        assert second.generation == result.generation+1
        assert owner.preflight()["reference_linear_factorizations"] == 1
        assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
        print(json.dumps(dict(status="EXPERIMENTAL_NOT_QUALIFIED", analysis="reference_linear",
            model_identity=owner.preflight()["identity"], law=group.law,
            footprint=group.footprint_id, shell_nodes=group.shell_node_ids,
            centroid_m=group.centroid_m.tolist(), area_m2=group.area_m2,
            shell_action_about_centroid=group.shell_action.tolist(),
            beam_action_about_node=group.beam_action.tolist(), gap=group.gap.tolist(),
            retained_dofs=owner.preflight()["retained_dofs"],
            equilibrium_ratio=result.free_equilibrium_ratio, constraint_max=result.constraint_max,
            checkpoint_bytes=len(checkpoint), factorization_count=1), sort_keys=True))
    finally:
        owner.close()
        resumed.close()


if __name__ == "__main__":
    main()
