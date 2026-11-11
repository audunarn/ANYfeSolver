"""Small experimental two-Q4/two-B3 point-connected panel.

Run with an installed development build of ANYfeSolver. This exercises a
reference-linear integration slice; it is not a qualified stiffened-shell
example or a nonlinear response prediction.
"""

import json
import argparse

import numpy as np

import anyfesolver
from anyfesolver import b3_ge
from anyfesolver.stiffened_shell import (
    B3_GE_E4_PL_STIFFENED_SHELL_V1, FixedCoordinate, LinearDefinition,
    NodalAction, PointConnection, ShellPanel,
)


def make_definition(beam_family="generalized"):
    if beam_family not in ("generalized", "physical_fibre", "mixed"):
        raise ValueError("explicit generalized, physical_fibre or mixed beam family required")
    young = 210e9
    shear = young/(2*(1+.3))
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((young*.01, shear*.008, shear*.008,
                 shear*1.6666666666666667e-5,
                 young*8.333333333333333e-6,
                 young*8.333333333333333e-6)),
        np.eye(6), 1e20, 1e9)
    arm = np.sqrt(8.333333333333333e-6/.01)
    physical_section = None
    if beam_family != "generalized":
        physical_section = b3_ge.PhysicalFibreSection(
            tuple(b3_ge.Fibre(str(i), y*arm, z*arm, .0025, young,
                             b3_ge.FlowCurve((0.,), (1e20,), 1e9))
                  for i, (y, z) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1)))),
            np.diag(np.sqrt((0., shear*.008, shear*.008,
                             shear*1.6666666666666667e-5, 0., 0.))))
    beam_nodes = tuple((x, -.5, .1) for x in (0., .5, 1., 1.5, 2.))
    beams = tuple(b3_ge.define_beam(
        "b3-ge", index+1,
        (101+2*index, 102+2*index, 103+2*index),
        np.asarray(beam_nodes[2*index:2*index+3]),
        np.tile(np.eye(3), (3, 1, 1)),
        section if beam_family == "generalized" or (beam_family == "mixed" and index == 0)
        else physical_section, np.eye(6))
        for index in range(2))
    shell = ShellPanel(
        (1, 2, 3, 4, 5, 6),
        ((0., -.5, 0.), (1., -.5, 0.), (2., -.5, 0.),
         (0., .5, 0.), (1., .5, 0.), (2., .5, 0.)),
        ((1, 2, 5, 4), (2, 3, 6, 5)),
        .07, young, .3, 7850.)
    return LinearDefinition(
        B3_GE_E4_PL_STIFFENED_SHELL_V1, shell, beams,
        (PointConnection(1, 101), PointConnection(2, 103),
         PointConnection(3, 105)),
        tuple(FixedCoordinate("shell", node, component)
              for node in (1, 4) for component in range(6)))


def run(beam_family="generalized"):
    definition = make_definition(beam_family)
    analysis = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    first = analysis.solve_linear((NodalAction(
        "beam", 105, (0., 0., 100., 0., 0., 0.)),))
    saved = analysis.checkpoint()
    continued = b3_ge.create_stiffened_shell_analysis(
        formulation="b3-ge", definition=definition)
    continued.restore(saved)
    second = continued.solve_linear((NodalAction(
        "beam", 105, (0., 0., 200., 0., 0., 0.)),))
    result = {
        "capabilities": analysis.capabilities,
        "preflight": analysis.preflight(),
        "first_tip_z_m": float(first.beam_coordinates[6*4+2]),
        "continued_tip_z_m": float(second.beam_coordinates[6*4+2]),
        "support_z_n": float(np.sum(first.reactions[:6*6].reshape(-1, 6)[:, 2])),
        "first_equilibrium_ratio": first.free_equilibrium_ratio,
        "first_constraint_max": first.constraint_max,
        "checkpoint_bytes": len(saved),
    }
    analysis.close()
    continued.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-family", choices=("generalized", "physical_fibre", "mixed"),
                        default="generalized")
    print(json.dumps(run(parser.parse_args().beam_family), sort_keys=True, allow_nan=False))
