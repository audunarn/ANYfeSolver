"""Bounded material-line row screen over declared beam:shell bond ratios.

This tests one provisional resolution rule only. It does not assemble a KKT
system or establish physical transfer, locking, or uniform inf-sup stability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import anyfesolver
import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_weak_line_candidate import LineMultiplierSpace
from scripts.b3_e4_f03_material_line_loaded import _scales
from scripts.b3_e4_f03_material_line_position import MaterialLinePosition
from scripts.b3_e4_f03_material_line_space_screen import (
    _rows, _scaled_row_diagnostics,
)
from scripts.b3_e4_f03_plate_patch_work import local_grid
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


# Existing C1 fixture bond cuts give 2, 4, 8, or 16 native half-cells.
# Every declared ratio uses the same 0.9–1.0 m physical material line.
CASES = (
    (2, 26, "1:8"),
    (4, 26, "1:4"),
    (4, 18, "1:2"),
    (8, 18, "1:1"),
    (8, 14, "2:1"),
    (8, 12, "4:1"),
    (16, 12, "8:1"),
)
KINDS = ("piecewise_constant", "continuous_linear", "dual_linear")


def evaluate(*, selective_yaw: bool = False) -> dict:
    expected = (Path(__file__).resolve().parents[1]/"src"/"anyfesolver").resolve()
    if Path(anyfesolver.__file__).resolve().parent != expected:
        raise RuntimeError("ratio screen imported another ANYfeSolver checkout")
    started = monotonic()
    deadline = started+120.
    cases = {}
    cases_to_check = (CASES + ((16, 14, "4:1-L2"),)
                      if selective_yaw else CASES)
    for shell_cells, beam_halves, label in cases_to_check:
        if monotonic() > deadline:
            raise TimeoutError("bounded C1 ratio row-screen deadline")
        xs, ys = local_grid(shell_cells, transverse_cells=4)
        beam = build(beam_halves, z_offset=.0675,
                     torsion_constant=square_torsion_constant(),
                     reference_profile="c1_tangent_2um", assemble=False)
        bond_halves = int(np.sum(np.diff(beam.abscissae)[
            beam.abscissae[:-1] >= .9-1e-12] > 0.))
        ratio_label = label.split("-")[0]
        if (shell_cells*1 != int(ratio_label.split(":")[0])*bond_halves /
                int(ratio_label.split(":")[1])):
            raise ArithmeticError("declared beam:shell bond ratio changed")
        intervals = min(shell_cells, 2*bond_halves)
        if shell_cells % intervals:
            raise ArithmeticError("provisional multiplier cuts leave shell hosts")
        line = MaterialLinePosition(xs, ys, beam)
        ns = 6*len(xs)*len(ys)
        fixed_shell = {6*(j*len(xs)+i)+axis
                       for j in range(len(ys)) for i, x in enumerate(xs)
                       if abs(x+.5) < 1e-12 for axis in range(6)}
        fixed_beam = {ns+int(i) for i in beam.context.fixed}
        free = np.setdiff1d(np.arange(ns+beam.context.count),
                            np.array(sorted(fixed_shell|fixed_beam)))
        yaw_intervals = shell_cells//2 if selective_yaw else None
        row_results = {}
        for kind in (KINDS[1:] if selective_yaw else KINDS):
            if selective_yaw:
                p0_rows, p0_masses = _rows(line, LineMultiplierSpace(
                    "piecewise_constant", tuple(np.linspace(0., 1., intervals+1))))
                yaw_rows, yaw_masses = _rows(line, LineMultiplierSpace(
                    kind, tuple(np.linspace(0., 1., yaw_intervals+1))))
                rows = np.vstack([p0_rows[axis::6] for axis in range(5)] +
                                 [yaw_rows[5::6]])
                masses = None
            else:
                space = LineMultiplierSpace(
                    kind, tuple(np.linspace(0., 1., intervals+1)))
                rows, masses = _rows(line, space)
            scales, _ = _scales(ns, beam, free, row_count=len(rows))
            diagnostic = _scaled_row_diagnostics(rows, free, scales[:len(free)])
            if diagnostic["scaled_row_rank"] > len(rows):
                raise ArithmeticError("reported line rank exceeds row count")
            row_results[kind] = {
                "rows": len(rows),
                "rank": diagnostic["scaled_row_rank"],
                "smallest_unit_row_singular": min(
                    diagnostic["scaled_row_singular_values"]),
                "rank_tolerance": diagnostic["rank_tolerance"],
                "active_primal_columns": diagnostic["active_primal_columns"],
                **({"p0_basis_integrals_m": p0_masses.tolist(),
                    "yaw_basis_integrals_m": yaw_masses.tolist()}
                   if selective_yaw else {"basis_integrals_m": masses.tolist()}),
            }
        cases[label] = {
            "shell_bond_cells": shell_cells,
            "beam_half_cells_total": beam_halves,
            "beam_bond_half_cells": bond_halves,
            "multiplier_intervals": intervals,
            "yaw_multiplier_intervals": yaw_intervals,
            "station_count": len(line.stations),
            "spaces": row_results,
        }
    return {
        "classification": "bounded material-line ratio row screen, not mixed stability or F03 acceptance",
        "selective_yaw": selective_yaw,
        "question": ("Does P0 for five channels and every-second-shell-cell P1/dual yaw avoid dependent supported rows at the registered ratios?"
                     if selective_yaw else
                     "Does m=min(shell bond cells,2*beam bond half-cells) avoid dependent projected rows at registered ratios?"),
        "ratio_convention": "shell bond cell count : native beam half-cell count on the same 0.9-1.0 m line",
        "geometry": "Q4/y4, 35 mm shell, 100 mm square beam, 67.5 mm eccentricity, same C1 taper and root supports",
        "status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "cases": cases,
        "elapsed_s": monotonic()-started,
        "limits": ["existing <=60-row and <=1200-active-column unit-row diagnostic",
                   "no tangent, KKT solve, physical load, or mesh-uniform stability claim",
                   "one provisional policy and one shell/beam grid per ratio",
                   "rank alone cannot select P0, P1, dual, or a channel-selective space"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True).strip(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "anyfesolver_import_path": str(
            Path(anyfesolver.__file__).resolve()),
        "effective_threadpools": threadpool_info(),
        "env": {k: os.environ.get(k) for k in
                ("NUMBA_DISABLE_JIT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
    }


def evaluate_yaw_resolution(*, four_to_one: bool = False) -> dict:
    """Fixed-primal yaw row/work screen, including the 4:1 missing mode."""
    expected = (Path(__file__).resolve().parents[1]/"src"/"anyfesolver").resolve()
    if Path(anyfesolver.__file__).resolve().parent != expected:
        raise RuntimeError("yaw screen imported another ANYfeSolver checkout")
    started = monotonic()
    shell_cells = 16 if four_to_one else 8
    beam_half_cells = 14 if four_to_one else 18
    xs, ys = local_grid(shell_cells, transverse_cells=4)
    beam = build(beam_half_cells, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um", assemble=False)
    bond_halves = int(np.sum(np.diff(beam.abscissae)[
        beam.abscissae[:-1] >= .9-1e-12] > 0.))
    if bond_halves != (4 if four_to_one else 8):
        raise ArithmeticError("fixed beam bond inventory changed")
    line = MaterialLinePosition(xs, ys, beam)
    ns = 6*len(xs)*len(ys)
    fixed_shell = {6*(j*len(xs)+i)+axis
                   for j in range(len(ys)) for i, x in enumerate(xs)
                   if abs(x+.5) < 1e-12 for axis in range(6)}
    fixed_beam = {ns+int(i) for i in beam.context.fixed}
    free = np.setdiff1d(np.arange(ns+beam.context.count),
                        np.array(sorted(fixed_shell|fixed_beam)))
    p0_rows, p0_masses = _rows(line, LineMultiplierSpace(
        "piecewise_constant", tuple(np.linspace(0., 1., 9))))
    if len(p0_rows) != 48 or abs(np.sum(p0_masses)-.1) > 1e-12:
        raise ArithmeticError("fixed five-channel P0 work changed")
    reference_intervals = 16 if four_to_one else 8
    reference_p0, reference_masses = _rows(line, LineMultiplierSpace(
        "piecewise_constant", tuple(np.linspace(0., 1., reference_intervals+1))))
    if abs(np.sum(reference_masses)-.1) > 1e-12:
        raise ArithmeticError("full-resolution P0 work changed")
    # Only primal coordinate scales are needed here. The historical loaded
    # helper admits at most 60 multiplier rows, while this read-only
    # reference-space comparison has 96 rows and assembles no KKT.
    p0_scales, _ = _scales(ns, beam, free, row_count=len(p0_rows))
    primal_scale = p0_scales[:len(free)]

    def unit_rows(rows):
        physical = rows[:, free]*primal_scale[None, :]
        norms = np.linalg.norm(physical, axis=1)
        if np.min(norms) <= 0.:
            raise ArithmeticError("zero physical row in yaw-space comparison")
        return physical/norms[:, None]

    def basis(rows):
        _, singular, right = np.linalg.svd(rows, full_matrices=False)
        rank = int(np.sum(singular > 1e-10*singular[0]))
        return right[:rank], rank

    def defect(rows, row_basis):
        return float(np.linalg.norm(rows-(rows@row_basis.T)@row_basis)/
                     np.linalg.norm(rows))

    p0_unit = unit_rows(reference_p0)
    p0_basis, p0_rank = basis(p0_unit)
    if not four_to_one and p0_rank != 48:
        raise ArithmeticError("fixed P0 reference row space changed")
    results = {}
    tested_intervals = (8, 12, 16) if four_to_one else (4, 5, 6, 7, 8)
    for intervals in tested_intervals:
        if monotonic()-started > 120.:
            raise TimeoutError("bounded yaw-resolution row deadline")
        kinds = {}
        row_spaces = {}
        for kind in (KINDS if four_to_one else KINDS[1:]):
            yaw, measures = _rows(line, LineMultiplierSpace(
                kind, tuple(np.linspace(0., 1., intervals+1))))
            if np.min(measures) <= 0. or abs(np.sum(measures)-.1) > 1e-12:
                raise ArithmeticError("yaw basis measure changed")
            rows = np.vstack([p0_rows[axis::6] for axis in range(5)] +
                             [yaw[5::6]])
            scales, _ = _scales(ns, beam, free, row_count=len(rows))
            if not np.array_equal(primal_scale, scales[:len(free)]):
                raise ArithmeticError("physical coordinate scales changed")
            diagnostic = _scaled_row_diagnostics(
                rows, free, scales[:len(free)])
            candidate_unit = unit_rows(rows)
            row_spaces[kind] = candidate_unit
            candidate_basis, candidate_rank = basis(candidate_unit)
            if candidate_rank != diagnostic["scaled_row_rank"]:
                raise ArithmeticError("independent rank checks disagree")
            c = rows[:, free]
            q = np.sin(np.arange(len(free))*.137)
            dual = np.cos(np.arange(len(rows))*.173)
            left = float(dual@(c@q))
            right = float((c.T@dual)@q)
            work_defect = abs(left-right)/max(1., abs(left), abs(right))
            if work_defect > 1e-12:
                raise ArithmeticError("selective yaw transpose work changed")
            kinds[kind] = {
                "rows": len(rows),
                "rank": diagnostic["scaled_row_rank"],
                "minimum_unit_row_singular": diagnostic[
                    "scaled_row_singular_values"][-1],
                "active_primal_columns": diagnostic["active_primal_columns"],
                "transpose_work_relative_defect": work_defect,
                "candidate_into_p0_row_span_relative_defect": defect(
                    candidate_unit, p0_basis),
                "p0_into_candidate_row_span_relative_defect": defect(
                    p0_unit, candidate_basis),
            }
        same_m_p0_span = {}
        if four_to_one:
            p0_yaw_basis, _ = basis(row_spaces["piecewise_constant"])
            for kind in KINDS[1:]:
                candidate_basis, _ = basis(row_spaces[kind])
                same_m_p0_span[kind] = {
                    "candidate_into_p0_relative_defect": defect(
                        row_spaces[kind], p0_yaw_basis),
                    "p0_into_candidate_relative_defect": defect(
                        row_spaces["piecewise_constant"], candidate_basis),
                }
        results[f"yaw_m{intervals}"] = {"yaw_intervals": intervals,
                                        "spaces": kinds,
                                        "same_m_p0_yaw_span": same_m_p0_span}
    return {
        "classification": "fixed-primal selective yaw resolution row/work diagnostic",
        "status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "question": ("Which exact yaw actions are missing from x16/B3-14 P0 m8, and can the m16 physical trace space be represented?"
                     if four_to_one else
                     "Can more longitudinal yaw test intervals remain independent before the known m8 alias?"),
        "decision": "rank/work admission only; physical loaded response and mixed stability remain separate",
        "geometry": (f"C1 Q4 x{shell_cells}/y4, B3-{beam_half_cells}, "
                     f"{'4:1' if four_to_one else '1:1'} on 0.9-1.0 m line; P0 m8 on five channels"),
        "reference_p0_intervals": reference_intervals,
        "reference_p0_rows": len(reference_p0),
        "reference_p0_rank": p0_rank,
        "base_p0_intervals": 8,
        "cases": results,
        "elapsed_s": monotonic()-started,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_git": subprocess.check_output(("git", "rev-parse", "HEAD"),
                                              text=True).strip(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "anyfesolver_import_path": str(
            Path(anyfesolver.__file__).resolve()),
        "effective_threadpools": threadpool_info(),
        "env": {k: os.environ.get(k) for k in
                ("NUMBA_DISABLE_JIT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "limits": ["same supported fixed primal and 120-second row deadline",
                   "no KKT solve, physical load, or mesh-uniform stability claim"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selective-yaw", action="store_true")
    parser.add_argument("--yaw-resolution-screen", action="store_true")
    parser.add_argument("--x16-yaw-capacity", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sum((args.selective_yaw, args.yaw_resolution_screen,
            args.x16_yaw_capacity)) > 1:
        raise ValueError("select one bounded row screen")
    with threadpool_limits(limits=1):
        result = (evaluate_yaw_resolution(four_to_one=args.x16_yaw_capacity)
                  if args.yaw_resolution_screen or args.x16_yaw_capacity
                  else evaluate(selective_yaw=args.selective_yaw))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({ratio: {name: [item["rank"], item["rows"]]
                             for name, item in case["spaces"].items()}
                      for ratio, case in result["cases"].items()}))


if __name__ == "__main__":
    main()
