"""Bounded native E4/B3 counterpart to the flat-web CalculiX development deck.

This compares declared physical geometry, loads and work only. The welded S4
web and weak L1 pose law are different connection/section idealizations; a
single mesh cannot qualify either or score formal F03 stability.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import scipy
from threadpoolctl import threadpool_info

import anyfesolver
from anyfesolver import b3_ge, stiffened_shell
from anyfesolver._stiffened_shell_l1_assembly_experimental import SPACE_A, SPACE_B
from scripts.b3_e4_f03_ccx_matched_t_reference import (
    E_PA, NU, PLATE_T_M, WEB_H_M, WEB_T_M, _section,
)


def build(space: str):
    """Construct the four-edge clamped flat-web line on a 1 by 0.2 m plate."""
    if space not in (SPACE_A, SPACE_B):
        raise ValueError("one complete L1 development space required")
    nx = ny = 4
    g = E_PA / (2 * (1 + NU))
    web = _section("flat")
    area = web["area_m2"]
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((E_PA * area,
                 g * web["nominal_shear_area_m2"],
                 g * web["nominal_shear_area_m2"],
                 g * web["J_rect_m4"],
                 E_PA * web["Iy_m4"], E_PA * web["Iz_m4"])),
        np.eye(6), 1e20, 1e9)

    xs = np.linspace(0., 1., nx + 1)
    ys = np.linspace(-.1, .1, ny + 1)
    plate_nodes = tuple((float(x), float(y), 0.) for y in ys for x in xs)
    width = nx + 1
    q4 = tuple((1 + j*width + i, 2 + j*width + i,
                2 + (j+1)*width + i, 1 + (j+1)*width + i)
               for j in range(ny) for i in range(nx))
    shell = stiffened_shell.ShellPanel(
        tuple(range(1, len(plate_nodes)+1)), plate_nodes, q4,
        PLATE_T_M, E_PA, NU, 7850.)

    axis_z = PLATE_T_M/2 + WEB_H_M/2
    beam_points = np.column_stack((xs, np.zeros(nx+1),
                                   np.full(nx+1, axis_z)))
    beams = tuple(b3_ge.define_beam(
        "b3-ge", element+1,
        (101+2*element, 102+2*element, 103+2*element),
        beam_points[2*element:2*element+3],
        np.tile(np.eye(3), (3, 1, 1)), section, np.eye(6))
        for element in range(nx//2))
    line_nodes = tuple(1 + (ny//2)*width + i for i in range(nx+1))
    line = stiffened_shell.ExperimentalL1Line(
        line_nodes, tuple(divmod(half, 2) for half in range(nx)), space,
        beam_offset=(0., 0., -WEB_H_M/2))
    fixed = tuple(stiffened_shell.FixedCoordinate("shell", 1+j*width, axis)
                  for j in range(ny+1) for axis in range(6)) + tuple(
                  stiffened_shell.FixedCoordinate("beam", 101, axis)
                  for axis in range(6))
    model = stiffened_shell.LinearDefinition(
        stiffened_shell.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        shell, beams, (), fixed, (line,))
    return model, line_nodes[-1], 101+nx, web


def evaluate() -> dict:
    results = {}
    for space in (SPACE_A, SPACE_B):
        model, plate_tip, beam_tip, web = build(space)
        if space == SPACE_A:
            # P1 g_z fixes root uz. The same P1 moments of g_y-a*g_twist
            # fix uy, since the shell-root translation and twist are fixed.
            # Keep ux (pitch uses P0) and all independent native nodal rotations.
            # Removing these two duplicate direct supports preserves the clamp.
            model = replace(model, fixed=tuple(f for f in model.fixed
                if not (f.domain == "beam" and f.node == 101 and f.component in (1, 2))))
        owner = stiffened_shell.create_analysis(definition=model)
        try:
            preflight = owner.preflight()
            # A +Fy at the plate's physical top surface has -z*Fy moment
            # about the midplane x-axis. The CalculiX reference nodes sit on
            # that surface using SHELL SECTION OFFSET=0.5.
            actions = (
                ("plate_fy", stiffened_shell.NodalAction(
                    "shell", plate_tip,
                    (0., 100., 0., -100.*PLATE_T_M/2, 0., 0.))),
                ("web_mx", stiffened_shell.NodalAction(
                    "beam", beam_tip, (0., 0., 0., 10., 0., 0.))),
            )
            cases = {}
            cross_work = {}
            for name, action in actions:
                result = owner.solve_linear((action,))
                if (result.status != "completed" or
                        result.free_equilibrium_ratio > 1e-8 or
                        result.constraint_max > 1e-9):
                    raise RuntimeError("bounded native flat-web equilibrium failed")
                if name == "plate_fy":
                    shell_slot = model.shell.node_ids.index(plate_tip)
                    dof = result.shell_displacement[6*shell_slot:6*shell_slot+6]
                    work = 100.*dof[1] - 100.*PLATE_T_M/2*dof[3]
                    beam_slot = owner._beam_index[beam_tip]
                    cross_work["web_mx_under_plate_fy_j"] = float(
                        10.*result.beam_coordinates[6*beam_slot+3])
                else:
                    beam_slot = owner._beam_index[beam_tip]
                    work = 10.*result.beam_coordinates[6*beam_slot+3]
                    shell_slot = model.shell.node_ids.index(plate_tip)
                    dof = result.shell_displacement[6*shell_slot:6*shell_slot+6]
                    cross_work["plate_fy_under_web_mx_j"] = float(
                        100.*dof[1] - 100.*PLATE_T_M/2*dof[3])
                shell_q = result.shell_displacement
                beam_q = result.beam_coordinates
                split = owner._shell_count
                shell_energy = .5*float(shell_q@(
                    owner._K[:split, :split]@shell_q))
                beam_energy = .5*float(beam_q@(
                    owner._K[split:, split:]@beam_q))
                if abs(shell_energy+beam_energy-.5*work) > 1e-8*max(
                        abs(work), 1e-12):
                    raise RuntimeError("native domain energy/work balance failed")
                cases[name] = {
                    "load_displacement_work_j": float(work),
                    "shell_internal_energy_j": shell_energy,
                    "beam_internal_energy_j": beam_energy,
                    "free_equilibrium_ratio": result.free_equilibrium_ratio,
                    "constraint_max": result.constraint_max,
                    "beam_root_uy_m": float(result.beam_coordinates[6*owner._beam_index[101]+1]),
                    "beam_root_uz_m": float(result.beam_coordinates[6*owner._beam_index[101]+2]),
                }
            results[space] = {
                "cases": cases,
                "cross_work_j": cross_work,
                "reciprocity_relative": abs(
                    cross_work["web_mx_under_plate_fy_j"]-
                    cross_work["plate_fy_under_web_mx_j"])/max(
                    abs(cross_work["web_mx_under_plate_fy_j"]),
                    abs(cross_work["plate_fy_under_web_mx_j"]), 1e-30),
                "matrix_rows": int(owner._A.shape[0]),
                "matrix_nnz": int(owner._A.nnz),
                "factorizations": owner.preflight()["reference_linear_factorizations"],
                "linear_refinements": owner.preflight()["reference_linear_refinements"],
                "line_rows": int(owner._C.shape[0]),
                "web_section": web,
            }
        finally:
            owner.close()
    return {
        "classification": "matched_geometry_development_not_same_law_or_F03_assessment",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": sys.version.split()[0],
                    "numpy": np.__version__, "scipy": scipy.__version__,
                    "anyfesolver_import": str(Path(anyfesolver.__file__).resolve()),
                    "effective_threadpools": [
                        {"internal_api": pool.get("internal_api"),
                         "num_threads": pool.get("num_threads")}
                        for pool in threadpool_info()]},
        "geometry_m": {"plate": [1., .2, PLATE_T_M],
                       "web": [1., WEB_T_M, WEB_H_M],
                       "beam_axis_z": PLATE_T_M/2 + WEB_H_M/2,
                       "material_line": [0., 1.]},
        "root": "all x=0 shell coordinates and native beam root pose fixed",
        "plate_load": "100 N Fy on physical top, represented by Fy and -1.75 Nm Mx at E4 midplane",
        "web_load": "10 Nm Mx at B3 axis; CalculiX uses a resultant-equivalent two-force couple",
        "spaces": results,
        "limitations": [
            "S4 welded web has finite cross-section deformation absent from B3.",
            "Nominal B3 shear areas are an engineering approximation.",
            "The S4 reference is not an exact weak L1 line-law oracle.",
            "One level cannot establish convergence, stability or accuracy.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
