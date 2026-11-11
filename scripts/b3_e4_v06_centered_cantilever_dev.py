"""Bounded engineering-load screen of a centered eccentric GE/Q4 rib.

Compares ordinary tip force/moment responses with a declared narrow-strip
transformed-section reference. This screen does not qualify V06 or E01.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from anyfesolver import b3_ge
from anyfesolver._stiffened_shell_ge_experimental import (
    PROFILE, FixedCoordinate, LinearDefinition, NodalAction, PointConnection,
    ShellPanel,
)
from anyfesolver.stiffened_shell import create_analysis


E = 210e9
NU = 0.3
G = E/(2.*(1.+NU))
LENGTH = 2.0
WIDTH = 0.2
THICKNESS = 0.07
ECCENTRICITY = 0.1
AREA = 0.01
IY = 8.333333333333333e-6
IX = 1.6666666666666667e-5
SHEAR_AREA = 0.008
LOADS = ((1000., 0.), (0., 100.), (1000., 100.))


def definition(intervals: int, sign: int):
    if intervals not in (2, 4, 8) or sign not in (-1, 1):
        raise ValueError("registered centered-rib grids/signs required")
    transverse = intervals
    def shell_id(row, column):
        return 1+row*(intervals+1)+column
    ids = tuple(shell_id(row, col) for row in range(transverse+1)
                for col in range(intervals+1))
    xyz = tuple((LENGTH*col/intervals,
                 WIDTH*(row/transverse-.5), 0.)
                for row in range(transverse+1)
                for col in range(intervals+1))
    cells = tuple((shell_id(row, col), shell_id(row, col+1),
                   shell_id(row+1, col+1), shell_id(row+1, col))
                  for row in range(transverse) for col in range(intervals))
    shell = ShellPanel(ids, xyz, cells, THICKNESS, E, NU, 7850.)
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((E*AREA, G*SHEAR_AREA, G*SHEAR_AREA,
                 G*IX, E*IY, E*IY)),
        np.eye(6), 1e20, 1e9)
    beam_xyz = np.array([(LENGTH*i/(2*intervals), 0.,
                          sign*ECCENTRICITY)
                         for i in range(2*intervals+1)])
    beams = tuple(b3_ge.define_beam(
        "b3-ge", col+1, (101+2*col, 102+2*col, 103+2*col),
        beam_xyz[2*col:2*col+3], np.tile(np.eye(3), (3, 1, 1)),
        section, np.eye(6)) for col in range(intervals))
    center = transverse//2
    points = tuple(PointConnection(shell_id(center, col), 101+2*col)
                   for col in range(intervals+1))
    fixed = tuple(FixedCoordinate("shell", shell_id(row, 0), axis)
                  for row in range(transverse+1) for axis in range(6))
    return LinearDefinition(PROFILE, shell, beams, points, fixed), 101+2*intervals


def _reference(sign: int, force: float, moment: float):
    # Free transverse contraction/anticlastic beam-strip idealization.
    a = E*WIDTH*THICKNESS+E*AREA
    c = E*AREA*sign*ECCENTRICITY
    d = (E*WIDTH*THICKNESS**3/12.+E*IY+
         E*AREA*ECCENTRICITY**2)
    strain, curvature = np.linalg.solve(
        np.array(((a, c), (c, d))),
        np.array((force, sign*ECCENTRICITY*force+moment)))
    tip_x = LENGTH*(strain+sign*ECCENTRICITY*curvature)
    tip_ry = LENGTH*curvature
    return {"strain": float(strain), "curvature_per_m": float(curvature),
            "tip_x_m": float(tip_x), "tip_ry_rad": float(tip_ry),
            "half_applied_work_j": 0.5*(force*tip_x+moment*tip_ry),
            "section_A": a, "section_C": c, "section_D": d}


def _cross(a, b):
    return np.cross(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def evaluate(intervals: int, sign: int):
    model, tip = definition(intervals, sign)
    owner = create_analysis(definition=model)
    cases = []
    for force, moment in LOADS:
        result = owner.solve_linear((NodalAction(
            "beam", tip, (force, 0., 0., 0., moment, 0.)),))
        if result.status != "completed":
            raise RuntimeError(f"centered V06 solve: {result.status}")
        base = 6*owner._beam_index[tip]
        tip_x = float(result.beam_coordinates[base])
        tip_ry = float(result.beam_coordinates[base+4])
        theory = _reference(sign, force, moment)
        support_force = np.zeros(3)
        support_moment = np.zeros(3)
        for i, position in enumerate(owner._shell_xyz):
            reaction = result.reactions[6*i:6*i+6]
            support_force += reaction[:3]
            support_moment += reaction[3:]+_cross(position, reaction[:3])
        applied_force = np.array((force, 0., 0.))
        applied_moment = (np.array((0., moment, 0.))+
                          _cross(owner._beam_nodes[tip][0], applied_force))
        shell_rows = result.shell_displacement.reshape(-1, 6)
        mid = intervals//2
        local_strain = []
        local_curvature = []
        for row in range(intervals+1):
            left = row*(intervals+1)+mid-1
            right = left+1
            dx = LENGTH/intervals
            local_strain.append((shell_rows[right, 0]-
                                 shell_rows[left, 0])/dx)
            local_curvature.append((shell_rows[right, 4]-
                                    shell_rows[left, 4])/dx)
        cases.append({
            "beam_tip_force_x_n": force,
            "beam_tip_moment_y_nm": moment,
            "reference": theory,
            "tip_x_m": tip_x,
            "tip_ry_rad": tip_ry,
            "tip_x_difference_m": tip_x-theory["tip_x_m"],
            "tip_ry_difference_rad": tip_ry-theory["tip_ry_rad"],
            "relative_tip_x_where_nonzero": (
                abs(tip_x-theory["tip_x_m"])/abs(theory["tip_x_m"])
                if abs(theory["tip_x_m"]) > 1e-12 else None),
            "relative_tip_ry_where_nonzero": (
                abs(tip_ry-theory["tip_ry_rad"])/abs(theory["tip_ry_rad"])
                if abs(theory["tip_ry_rad"]) > 1e-12 else None),
            "half_applied_work_j": 0.5*(force*tip_x+moment*tip_ry),
            "center_section_shell_axial_strain_by_row":
                list(map(float, local_strain)),
            "center_section_shell_curvature_by_row":
                list(map(float, local_curvature)),
            "support_force_error_n": float(np.max(np.abs(
                support_force+applied_force))),
            "support_moment_error_nm": float(np.max(np.abs(
                support_moment+applied_moment))),
            "free_equilibrium_ratio": result.free_equilibrium_ratio,
            "constraint_max": result.constraint_max,
        })
    return {"shell_intervals": intervals,
            "transverse_intervals": intervals,
            "offset_sign": sign,
            "physical_dofs": owner._shell_count+6*len(owner._beam_ids),
            "retained_dofs": owner._K.shape[0]-owner._shell_count-
                             6*len(owner._beam_ids),
            "matrix_nnz": int(owner._A.nnz),
            "linear_factorizations": owner._linear_factorizations,
            "cases": cases}


def main(output: Path):
    if output.exists():
        raise FileExistsError("preserve previous centered-rib evidence")
    cases = [evaluate(n, sign) for n in (2, 4, 8) for sign in (-1, 1)]
    record = {
        "classification": "V06 centered eccentric rib; development only",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "geometry": {"length_m": LENGTH, "width_m": WIDTH,
                     "shell_thickness_m": THICKNESS,
                     "beam_area_m2": AREA, "beam_iy_m4": IY,
                     "offset_m": ECCENTRICITY},
        "cases": cases,
        "limit": ("Ordinary beam-tip loads; free-edge narrow-strip theory is an "
                  "engineering comparator, not an exact discrete oracle."),
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = main(args.output)
    summary = [{"n": item["shell_intervals"], "sign": item["offset_sign"],
                "factorizations": item["linear_factorizations"],
                "tip_x_error": [case["relative_tip_x_where_nonzero"]
                                for case in item["cases"]],
                "tip_ry_error": [case["relative_tip_ry_where_nonzero"]
                                 for case in item["cases"]]}
               for item in result["cases"]]
    print(json.dumps(summary))
