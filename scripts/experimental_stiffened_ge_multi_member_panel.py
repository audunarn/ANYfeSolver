"""Connected multi-stiffener reference-linear development screen.

Point bonds are explicitly experimental. This checks a usable sparse route,
not the registered nonmatching G5 workload or a mesh-convergence claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
from anyfesolver import b3_ge, stiffened_shell as ss


def make_definition(cells_per_stiffener=8, stiffeners=4, *,
                    shell_transverse_cells=None, tie_stride=1):
    if (type(cells_per_stiffener) is not int or cells_per_stiffener < 1 or
            type(stiffeners) is not int or stiffeners < 2):
        raise ValueError("positive longitudinal cells and at least two stiffeners required")
    nx = cells_per_stiffener
    ny = stiffeners-1 if shell_transverse_cells is None else shell_transverse_cells
    if (type(ny) is not int or ny < stiffeners-1 or
            type(tie_stride) is not int or tie_stride < 1 or nx % tie_stride):
        raise ValueError("resolved transverse shell rows and dividing point-tie stride required")
    beam_rows = tuple(round(j*ny/(stiffeners-1)) for j in range(stiffeners))
    node = lambda i, j: 1+j*(nx+1)+i
    coordinates = tuple((2.*i/nx, -0.5+j/ny, 0.)
                        for j in range(ny+1) for i in range(nx+1))
    shell = ss.ShellPanel(tuple(range(1, len(coordinates)+1)), coordinates,
        tuple((node(i,j), node(i+1,j), node(i+1,j+1), node(i,j+1))
              for j in range(ny) for i in range(nx)), .01, 210e9, .3, 7850.)
    young, shear = 210e9, 210e9/2.6
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((young*1e-4, shear*8e-5, shear*8e-5,
                 shear*2e-8, young*1e-8, young*1e-8)),
        np.eye(6), 1e20, 1e9)
    beams, points = [], []
    # Distinct parallel member nodes; consecutive elements share end nodes.
    # Shell IDs and beam IDs remain separate model identities.
    for j, shell_row in enumerate(beam_rows):
        base = 100_000+j*(2*nx+1)
        for i in range(nx):
            beams.append(b3_ge.define_beam("b3-ge", 1+j*nx+i,
                (base+2*i, base+2*i+1, base+2*i+2),
                np.array([(2.*(i+k/2)/nx, -0.5+shell_row/ny, .03)
                          for k in range(3)]),
                np.tile(np.eye(3), (3,1,1)), section, np.eye(6)))
        points.extend(ss.PointConnection(node(i,shell_row), base+2*i)
                      for i in range(0,nx+1,tie_stride))
    fixed = tuple(ss.FixedCoordinate("shell", node(0,j), component)
                  for j in range(ny+1) for component in range(6))
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        shell, tuple(beams), tuple(points), fixed)


def actions_for(definition, force=100.):
    # One declared spatial dead force at each physical stiffener end.
    shell_positions = dict(zip(definition.shell.node_ids, definition.shell.coordinates_m))
    end_x = max(position[0] for position in shell_positions.values())
    end_nodes = tuple(point.beam_node for point in definition.points
                      if shell_positions[point.shell_node][0] == end_x)
    # Mixed forces also excite the offset x force lever arm; Fz alone would not.
    return tuple(ss.NodalAction("beam", node, (.1*force,.03*force,force,0.,0.,0.))
                 for node in end_nodes)


def check_balance(definition, result, actions):
    support = result.reactions[:6*len(definition.shell.node_ids)].reshape(-1,6)
    force = support[:,:3].sum(axis=0)
    moment = (support[:,3:]+np.cross(np.asarray(definition.shell.coordinates_m),
                                    support[:,:3])).sum(axis=0)
    shell_positions = dict(zip(definition.shell.node_ids, definition.shell.coordinates_m))
    # This fixture's declared eccentric reference axis is 30 mm above the shell.
    positions = {point.beam_node: np.asarray(shell_positions[point.shell_node])+[0.,0.,.03]
                 for point in definition.points}
    for action in actions:
        wrench = np.asarray(action.wrench)
        force += wrench[:3]
        moment += wrench[3:]+np.cross(positions[action.node], wrench[:3])
    scale = sum(np.linalg.norm(action.wrench[:3]) for action in actions)
    return float(np.linalg.norm(force)/scale), float(np.linalg.norm(moment)/(2.*scale))


def run(cells_per_stiffener=8, stiffeners=4, *,
        shell_transverse_cells=None, tie_stride=1):
    started = time.perf_counter()
    definition = make_definition(cells_per_stiffener, stiffeners,
        shell_transverse_cells=shell_transverse_cells, tie_stride=tie_stride)
    definition_seconds = time.perf_counter()-started
    actions = actions_for(definition)
    started = time.perf_counter()
    owner = ss.create_analysis(definition=definition)
    compile_seconds = time.perf_counter()-started
    try:
        started = time.perf_counter()
        first = owner.solve_linear(actions)
        solve_seconds = time.perf_counter()-started
        force_error, moment_error = check_balance(definition, first, actions)
        checkpoint = owner.checkpoint()
        second = owner.solve_linear(actions_for(definition, 200.))
        assert owner.preflight()["reference_linear_factorizations"] == 1
        np.testing.assert_allclose(second.shell_displacement,
            2.*first.shell_displacement, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(second.beam_coordinates,
            2.*first.beam_coordinates, rtol=1e-10, atol=1e-14)
        assert force_error <= 1e-8 and moment_error <= 1e-8
        started = time.perf_counter()
        resumed = ss.create_analysis(definition=definition)
        restore_compile_seconds = time.perf_counter()-started
        try:
            started = time.perf_counter()
            resumed.restore(checkpoint)
            restored = resumed.recover()
            restore_seconds = time.perf_counter()-started
            for name in ("shell_displacement", "beam_coordinates", "multipliers", "reactions"):
                np.testing.assert_array_equal(getattr(first,name), getattr(restored,name))
            assert resumed.checkpoint() == checkpoint
            continued = resumed.solve_linear(actions_for(definition, 200.))
            for name in ("shell_displacement", "beam_coordinates", "multipliers", "reactions"):
                np.testing.assert_array_equal(getattr(second,name), getattr(continued,name))
        finally:
            resumed.close()
        return dict(status=first.status, production_qualified=False,
            shell_elements=len(definition.shell.q4_elements),
            beam_elements=len(definition.beams), stiffeners=stiffeners,
            point_tie_stride=tie_stride,
            preflight=owner.preflight(), checkpoint_bytes=len(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
            force_balance_ratio=force_error, moment_balance_ratio=moment_error,
            equilibrium_ratio=first.free_equilibrium_ratio,
            constraint_max=first.constraint_max,
            seconds=dict(definition=definition_seconds, compile=compile_seconds,
                         first_solve=solve_seconds, restore_compile=restore_compile_seconds,
                         restore=restore_seconds))
    finally:
        owner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--stiffeners", type=int, default=4)
    parser.add_argument("--shell-transverse-cells", type=int)
    parser.add_argument("--tie-stride", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.cells,args.stiffeners,
        shell_transverse_cells=args.shell_transverse_cells, tie_stride=args.tie_stride), indent=2))
