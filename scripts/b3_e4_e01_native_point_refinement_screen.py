"""Compare bounded native GE point panels with saved independent OpenSees cases.

This is a source-development screen. It neither freezes the final E01 case
contract nor qualifies a physical quantity on installed artifacts.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import anyfesolver
from anyfesolver import b3_ge
from anyfesolver._stiffened_shell_ge_experimental import (
    PROFILE, FixedCoordinate, LinearDefinition, NodalAction, PointConnection,
    ShellPanel,
)
from anyfesolver.stiffened_shell import create_analysis


E = 210e9
NU = .3
G = E / (2 * (1 + NU))


def definition(intervals: int, sign: int, transverse_intervals: int = 1):
    def shell_id(row, column):
        return 1 + row * (intervals + 1) + column

    tip_node = 101 + 2 * intervals
    shell_ids = tuple(range(1, (intervals + 1) *
                            (transverse_intervals + 1) + 1))
    shell_xyz = tuple(
        (2. * i / intervals, -.5 + row / transverse_intervals, 0.)
        for row in range(transverse_intervals + 1)
        for i in range(intervals + 1))
    cells = tuple((shell_id(row, i), shell_id(row, i + 1),
                   shell_id(row + 1, i + 1), shell_id(row + 1, i))
                  for row in range(transverse_intervals)
                  for i in range(intervals))
    shell = ShellPanel(shell_ids, shell_xyz, cells, .07, E, NU, 7850.)
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((E * .01, G * .008, G * .008,
                 G * 1.6666666666666667e-5,
                 E * 8.333333333333333e-6,
                 E * 8.333333333333333e-6)),
        np.eye(6), 1e20, 1e9)
    beam_xyz = np.asarray([(i / intervals, -.5, sign * .1)
                           for i in range(2 * intervals + 1)])
    beams = tuple(b3_ge.define_beam(
        "b3-ge", i + 1, (101 + 2 * i, 102 + 2 * i, 103 + 2 * i),
        beam_xyz[2 * i:2 * i + 3], np.tile(np.eye(3), (3, 1, 1)),
        section, np.eye(6)) for i in range(intervals))
    points = tuple(PointConnection(shell_id(0, i), 101 + 2 * i)
                   for i in range(intervals + 1))
    fixed = tuple(FixedCoordinate("shell", node, axis)
                  for row in range(transverse_intervals + 1)
                  for node in (shell_id(row, 0),) for axis in range(6))
    return LinearDefinition(PROFILE, shell, beams, points, fixed), tip_node


def solve(intervals: int, sign: int, transverse_intervals: int = 1):
    model, tip_node = definition(intervals, sign, transverse_intervals)
    owner = create_analysis(definition=model)
    result = owner.solve_linear((NodalAction(
        "beam", tip_node, (1000., 0., 0., 0., 0., 0.)),))
    if result.status != "completed":
        raise RuntimeError(f"native GE returned {result.status}")
    reactions = result.reactions[:owner._shell_count].reshape(-1, 6)
    support = {str(1 + row * (intervals + 1)):
               reactions[row * (intervals + 1)].tolist()
               for row in range(transverse_intervals + 1)}
    shell_response = {
        str(1 + row * (intervals + 1) + intervals):
        result.shell_displacement[
            6 * (row * (intervals + 1) + intervals):
            6 * (row * (intervals + 1) + intervals + 1)].tolist()
        for row in (0, transverse_intervals)
    }
    return {
        "offset_sign": sign, "shell_intervals": intervals,
        "transverse_intervals": transverse_intervals,
        "point_stations": intervals + 1,
        "tip_translation_m": result.beam_coordinates[6 * 2 * intervals:
                                                     6 * 2 * intervals + 3].tolist(),
        "tip_rotation_rad": result.beam_coordinates[6 * 2 * intervals + 3:
                                                    6 * 2 * intervals + 6].tolist(),
        "support_reactions": support,
        "selected_shell_displacements": shell_response,
        "free_equilibrium_ratio": result.free_equilibrium_ratio,
        "constraint_max": result.constraint_max,
    }


def compare(native: dict, external: dict):
    quantities = {}
    def difference(a, b):
        if len(a) != len(b):
            raise ValueError("response dimension mismatch")
        return {
            "native": a, "external": b,
            "absolute_difference": [abs(x - y) for x, y in zip(a, b)],
            "relative_to_external_where_nonzero": [
                abs(x - y) / abs(y) if abs(y) > 1e-12 else None
                for x, y in zip(a, b)],
        }

    for label in ("tip_translation_m", "tip_rotation_rad"):
        a, b = native[label], external[label]
        quantities[label] = difference(a, b)
    for node, a in native["selected_shell_displacements"].items():
        b = external["node_displacements"][node]
        quantities[f"shell_node_{node}"] = difference(a, b)

    def support_resultant(support, intervals, transverse_intervals):
        force = np.zeros(3)
        moment = np.zeros(3)
        for row in range(transverse_intervals + 1):
            node = str(1 + row * (intervals + 1))
            wrench = np.asarray(support[node], dtype=float)
            if wrench.shape != (6,) or not np.isfinite(wrench).all():
                raise ValueError("finite six-component support wrench required")
            xyz = np.asarray((0., -.5 + row / transverse_intervals, 0.))
            force += wrench[:3]
            moment += wrench[3:] + np.cross(xyz, wrench[:3])
        return np.r_[force, moment].tolist()

    intervals = native["shell_intervals"]
    transverse = native["transverse_intervals"]
    quantities["support_resultant_about_origin"] = difference(
        support_resultant(native["support_reactions"], intervals, transverse),
        support_resultant(external["support_reactions"], intervals, transverse))
    return quantities


def main(external_dir: Path, output: Path, two_d_refinement: bool = False):
    if output.exists():
        raise FileExistsError("preserve previous comparison")
    record = {
        "classification": "E01 source-development comparison; not accepted",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "solver_import": str(Path(anyfesolver.__file__).resolve()),
        "two_d_refinement": two_d_refinement,
        "external_files": {}, "cases": [], "failure": None,
    }
    try:
        for intervals in (2, 4, 8):
            transverse = intervals // 2 if two_d_refinement else 1
            filename = (f"results-2d-n{intervals}.json" if two_d_refinement
                        else f"results-n{intervals}-work.json")
            path = external_dir / filename
            raw = path.read_bytes()
            ext = json.loads(raw)
            if ext["failure"] is not None or len(ext["cases"]) != 2:
                raise ValueError(f"incomplete external cases: {path}")
            record["external_files"][path.name] = sha256(raw).hexdigest()
            for sign in (-1, 1):
                counterpart = next(case for case in ext["cases"]
                                   if case["offset_sign"] == sign)
                if (counterpart["shell_intervals"] != intervals
                        or counterpart.get("transverse_intervals", 1)
                        != transverse):
                    raise ValueError("external resolution mismatch")
                native = solve(intervals, sign, transverse)
                record["cases"].append({
                    "native": native,
                    "external_offset_sign": sign,
                    "external_shell_intervals": intervals,
                    "external_transverse_intervals": transverse,
                    "comparison": compare(native, counterpart),
                })
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
    output.write_text(json.dumps(record, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--two-d-refinement", action="store_true")
    args = parser.parse_args()
    result = main(args.external_dir, args.output, args.two_d_refinement)
    print(json.dumps({"failure": result["failure"],
                      "completed_cases": len(result["cases"])}))
    if result["failure"]:
        raise SystemExit(1)
