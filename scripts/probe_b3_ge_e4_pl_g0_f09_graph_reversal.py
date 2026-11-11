"""Bounded F09 generalized-section graph reversal discriminator.

Preparatory only: no public B2 dispatch, admission, or qualification claim.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "anyfesolver"
if __name__ != "__main__":
    import anyfesolver
if "anyfesolver" not in sys.modules:
    package = types.ModuleType("anyfesolver")
    package.__path__ = [str(SRC)]
    sys.modules["anyfesolver"] = package
for name in ("_b2_physical_flexibility_linear", "_b2_physical_linear_definition"):
    qualified = f"anyfesolver.{name}"
    if qualified not in sys.modules:
        spec = importlib.util.spec_from_file_location(qualified, SRC / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)

from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import PhysicalB2Member


POINTS = ((0., 0., 0.), (2., 0., 0.), (2., 1.5, 0.))
LOADS = ((.4, -.9, .7, -.2, .3, .1),
         (-.25, .6, -.8, .4, -.15, .35))
COMPONENT_TOL = 2e-12  # Diagnostic floor: 1 m or 1 rad.
WORK_TOL = 2e-12
BALANCE_TOL = 2e-12  # Relative to separate applied N and Nm scales.


def section_matrix():
    matrix = np.diag([22., 13., 17., 9., 7., 11.])
    for i, j, value in ((0, 4, 1.5), (1, 5, -.7), (2, 3, .6)):
        matrix[i, j] = matrix[j, i] = value
    return matrix


def members(reverse_second=False, transform_section=True):
    section = section_matrix()
    first = PhysicalB2Member("m0", "source0", (0, 1), POINTS[:2],
                             (0., 1., 0.), "generalized", LinearSection.generalized(section))
    if reverse_second:
        # The physical y axis remains fixed: x and z change sign. A section
        # tensor needs an explicit reorientation; the definition does not infer it.
        sign = np.diag([-1., 1., -1., -1., 1., -1.])
        revised = sign @ section @ sign if transform_section else section
        second = PhysicalB2Member("m1", "source1", (2, 1),
                                  (POINTS[2], POINTS[1]), (-1., 0., 0.),
                                  "generalized/reoriented" if transform_section else "generalized/unchanged",
                                  LinearSection.generalized(revised))
    else:
        second = PhysicalB2Member("m1", "source1", (1, 2),
                                  (POINTS[1], POINTS[2]), (-1., 0., 0.),
                                  "generalized", LinearSection.generalized(section))
    return first, second


def assemble(members_):
    matrix = sparse.lil_matrix((18, 18), dtype=float)
    for beam in members_:
        transform = beam.local_dof_map()
        local = beam.local_operator().evaluate(np.zeros(12)).tangent
        element = transform.T @ local @ transform
        indices = [6 * node + i for node in beam.node_ids for i in range(6)]
        for i, row in enumerate(indices):
            for j, col in enumerate(indices):
                matrix[row, col] += element[i, j]
    return matrix.tocsc()


def exact_solve(matrix, rhs):
    """Rational elimination for the independent section-compliance integral."""
    n = len(rhs)
    rows = [[Fraction(float(matrix[i][j])) for j in range(n)]
            + [Fraction(float(rhs[i]))] for i in range(n)]
    for k in range(n):
        pivot = next(i for i in range(k, n) if rows[i][k])
        rows[k], rows[pivot] = rows[pivot], rows[k]
        divisor = rows[k][k]
        rows[k] = [v / divisor for v in rows[k]]
        for i in range(n):
            if i != k:
                factor = rows[i][k]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[k])]
    return [row[-1] for row in rows]


def physical_work(members_, tip_wrench):
    """Exact cut-wrench Castigliano energy, independent of element matrices."""
    f = tuple(map(Fraction, tip_wrench[:3]))
    m = tuple(map(Fraction, tip_wrench[3:]))
    tip = tuple(map(Fraction, POINTS[2]))
    cross = lambda a, b: (a[1]*b[2]-a[2]*b[1],
                          a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    total = Fraction(0)
    for beam in members_:
        start, end = (tuple(map(Fraction, p)) for p in beam.reference_points_m)
        # These exact reference frames come from the declared geometry, not
        # the product frame function or its local basic map.
        frame = (((1, 0, 0), (0, 1, 0), (0, 0, 1)) if beam.member_id == "m0"
                 else (((0, -1, 0), (-1, 0, 0), (0, 0, -1)) if beam.node_ids == (2, 1)
                       else ((0, 1, 0), (-1, 0, 0), (0, 0, 1))))
        densities = []
        for t in (Fraction(0), Fraction(1, 2), Fraction(1)):
            cut = tuple(a + t * (b-a) for a, b in zip(start, end))
            arm = tuple(a-b for a, b in zip(tip, cut))
            torque = tuple(a+b for a, b in zip(m, cross(arm, f)))
            wrench = [sum(axis[i]*v[i] for i in range(3))
                      for v in (f, torque) for axis in frame]
            strain = exact_solve(beam.section.stiffness, wrench)
            densities.append(sum(a*b for a, b in zip(wrench, strain)))
        length = Fraction(float(beam.length_m))
        total += length * (densities[0] + 4*densities[1] + densities[2]) / 12
    return float(total)


def evaluate():
    forward, reverse = members(), members(True)
    original = assemble(forward)
    changed = assemble(reverse)
    wrong = assemble(members(True, False))
    free = np.arange(6, 18)
    norm = max(1., float(np.max(np.abs(original.data))))
    matrix_error = float(np.max(np.abs((changed-original).toarray()))) / norm
    wrong_matrix_difference = float(np.max(np.abs((wrong-original).toarray()))) / norm
    cases = []
    for tip in LOADS:
        load = np.zeros(18)
        load[12:] = tip
        answers = []
        for matrix in (original, changed):
            displacement = np.zeros(18)
            displacement[free] = splu(matrix[free, :][:, free]).solve(load[free])
            reaction = matrix @ displacement - load
            answers.append((displacement, reaction))
        a, b = answers
        tip_error = float(np.max(np.abs(a[0][12:] - b[0][12:])
                                / np.maximum(1., np.abs(a[0][12:]))))
        reference_a = physical_work(forward, tip)
        reference_b = physical_work(reverse, tip)
        wrong_reference = physical_work(members(True, False), tip)
        energies = [.5 * float(item[0] @ load) for item in answers]
        work_error = max(abs(energy-reference_a) / max(1., abs(reference_a))
                         for energy in energies)
        reference_error = abs(reference_a-reference_b) / max(1., abs(reference_a))
        unchanged_section_work_difference = abs(reference_a-wrong_reference) / max(1., abs(reference_a))
        force_scale = max(abs(v) for v in tip[:3])
        moment_scale = max(abs(v) for v in tip[3:])
        force_balance = max(float(np.max(np.abs(item[1][6:18].reshape(2, 6)[:, :3])))/force_scale
                            for item in answers)
        moment_balance = max(float(np.max(np.abs(item[1][6:18].reshape(2, 6)[:, 3:])))/moment_scale
                             for item in answers)
        cases.append(dict(tip_wrench=tip, tip_error=tip_error, work_error=work_error,
                          reference_error=reference_error,
                          unchanged_section_work_difference=unchanged_section_work_difference,
                          free_force_residual_ratio=force_balance,
                          free_moment_residual_ratio=moment_balance,
                          physical_work=reference_a))
    passed = (matrix_error < COMPONENT_TOL and wrong_matrix_difference > 1e-4
              and all(c["tip_error"] < COMPONENT_TOL and c["work_error"] < WORK_TOL
                      and c["reference_error"] < WORK_TOL
                      and c["unchanged_section_work_difference"] > 1e-4
                      and c["free_force_residual_ratio"] < BALANCE_TOL
                      and c["free_moment_residual_ratio"] < BALANCE_TOL for c in cases))
    return dict(passed=passed, matrix_error=matrix_error,
                unchanged_section_matrix_difference=wrong_matrix_difference,
                cases=cases, diagnostic_limits=dict(component=COMPONENT_TOL,
                                                    work=WORK_TOL, balance=BALANCE_TOL))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "cases": len(result["cases"]),
                      "output": str(args.output)}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
