"""Unregistered B2 mixed-load conditioning diagnostic, not F09 qualification.

Compare a binary64 sparse solve with (1) an exact solve of its *rounded*
assembled matrix and (2) an independent rational section-wrench force method.
The extreme section is deliberately outside any established admission range.
"""

from __future__ import annotations

from fractions import Fraction as Q
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import PhysicalB2Member


AREA = 2.0 ** -80
POINTS = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.5, 0.0))
AXES = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
FRAMES = (((1, 0, 0), (0, 1, 0), (0, 0, 1)),
          ((0, 1, 0), (-1, 0, 0), (0, 0, 1)))


def _exact_solve(matrix, rhs):
    """Bounded rational Gauss-Jordan solve independent of the B2 operator."""
    n = len(rhs)
    # Preserve exact intermediates from the force method. The assembled
    # binary64 graph inputs are still converted by their exact float value.
    rational = lambda value: value if type(value) is Q else Q(float(value))
    a = [[rational(matrix[i][j]) for j in range(n)] + [rational(rhs[i])]
         for i in range(n)]
    for column in range(n):
        pivot_row = next((i for i in range(column, n) if a[i][column]), None)
        if pivot_row is None:
            raise ArithmeticError("exact rounded graph is singular")
        a[column], a[pivot_row] = a[pivot_row], a[column]
        pivot = a[column][column]
        a[column] = [item / pivot for item in a[column]]
        for row in range(n):
            if row != column:
                multiplier = a[row][column]
                if multiplier:
                    a[row] = [left - multiplier * right
                              for left, right in zip(a[row], a[column])]
    return tuple(row[-1] for row in a)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _force_method_energy(section, terminal):
    """Independent exact cut-wrench energy for the prescribed elbow.

    Wrenches are affine on each straight member, so Simpson integration of
    their quadratic complementary energy is exact. FRAMES are prescribed
    directly; they are not obtained from the product member transform.
    """
    tip = tuple(Q(v) for v in POINTS[-1])
    force, couple = terminal[:3], terminal[3:]
    total = Q(0)
    for member, frame in enumerate(FRAMES):
        start = tuple(Q(v) for v in POINTS[member])
        end = tuple(Q(v) for v in POINTS[member + 1])
        length = Q(2) if member == 0 else Q(3, 2)
        integrand = []
        for t in (Q(0), Q(1, 2), Q(1)):
            point = tuple(a + t * (b - a) for a, b in zip(start, end))
            lever = tuple(a - b for a, b in zip(tip, point))
            moment_arm = _cross(lever, force)
            moment = tuple(couple[i] + moment_arm[i] for i in range(3))
            wrench = tuple(sum(Q(axis[j]) * vector[j] for j in range(3))
                           for vector in (force, moment) for axis in frame)
            strain = _exact_solve(section.stiffness, wrench)
            integrand.append(sum(a * b for a, b in zip(wrench, strain)))
        total += length * (integrand[0] + 4 * integrand[1] + integrand[2]) / 12
    return total


def _members(section):
    return tuple(PhysicalB2Member(
        f"m{i}", f"source-{i}", (i, i + 1),
        (POINTS[i], POINTS[i + 1]), AXES[i], "extreme-scalar", section,
    ) for i in range(2))


def _float_graph(members, load):
    rows, columns, values = [], [], []
    for member in members:
        transform = member.local_dof_map()
        local = member.local_operator().evaluate(np.zeros(12)).tangent
        tangent = transform.T @ local @ transform
        dofs = tuple(range(6 * member.node_ids[0], 6 * member.node_ids[0] + 6)) + \
               tuple(range(6 * member.node_ids[1], 6 * member.node_ids[1] + 6))
        for i, row in enumerate(dofs):
            for j, column in enumerate(dofs):
                rows.append(row)
                columns.append(column)
                values.append(float(tangent[i, j]))
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(18, 18)).tocsc()
    free = np.arange(6, 18)
    reduced = matrix[free, :][:, free].tocsc()
    result = splu(reduced).solve(load[free])
    u = np.zeros(18)
    u[free] = result
    return u, matrix, reduced


def screen(area=AREA):
    section = LinearSection.isotropic(
        E=1.0, nu=0.0, area=area, Iy=1.0, Iz=1.0, J=1.0, ky=1.0, kz=1.0,
    )
    terminal = (area / 2, area / 3, area / 5, .25, -.125, .375)
    load = np.zeros(18)
    load[12:] = terminal
    displacement, matrix, reduced = _float_graph(_members(section), load)
    exact_rounded = _exact_solve(reduced.toarray(), load[6:])
    applied = tuple(Q(float(value)) for value in terminal)
    reference_tip = []
    for axis in range(6):
        plus, minus = list(applied), list(applied)
        plus[axis] += 1
        minus[axis] -= 1
        reference_tip.append((_force_method_energy(section, plus)
                              - _force_method_energy(section, minus)) / 2)
    reference = np.array([float(value) for value in reference_tip])
    rounded_tip = np.array([float(value) for value in exact_rounded[6:]])
    sparse_tip = displacement[12:]
    translation_scale = np.maximum(1.0, np.abs(reference[:3]))
    rotation_scale = np.maximum(1.0, np.abs(reference[3:]))
    residual = matrix @ displacement - load
    free_force_residual = residual[6:][[0, 1, 2, 6, 7, 8]]
    free_moment_residual = residual[6:][[3, 4, 5, 9, 10, 11]]
    force_scale = max(abs(value) for value in terminal[:3])
    moment_scale = max(abs(value) for value in terminal[3:])
    return {
        "scope": "UNREGISTERED_ADVERSE_DIAGNOSTIC_NOT_F09_ACCEPTANCE",
        "area_m2": area,
        "tip_load": terminal,
        "physical_force_method_tip": reference.tolist(),
        "exact_rounded_matrix_tip": rounded_tip.tolist(),
        "binary64_sparse_tip": sparse_tip.tolist(),
        "exact_rounded_vs_physical_translation_scaled_max": float(np.max(np.abs(rounded_tip[:3] - reference[:3]) / translation_scale)),
        "exact_rounded_vs_physical_rotation_scaled_max": float(np.max(np.abs(rounded_tip[3:] - reference[3:]) / rotation_scale)),
        "sparse_vs_physical_translation_scaled_max": float(np.max(np.abs(sparse_tip[:3] - reference[:3]) / translation_scale)),
        "sparse_vs_physical_rotation_scaled_max": float(np.max(np.abs(sparse_tip[3:] - reference[3:]) / rotation_scale)),
        "sparse_free_residual_norm": float(np.linalg.norm(residual[6:])),
        "sparse_free_force_residual_to_tip_load": float(np.max(np.abs(free_force_residual)) / force_scale),
        "sparse_free_moment_residual_to_tip_load": float(np.max(np.abs(free_moment_residual)) / moment_scale),
        "estimated_reduced_condition_2": float(np.linalg.cond(reduced.toarray())),
        "matrix_order": int(reduced.shape[0]),
        "matrix_nnz": int(reduced.nnz),
    }


if __name__ == "__main__":
    print(json.dumps({"moderate": screen(1.0e-4), "extreme": screen()},
                     sort_keys=True, allow_nan=False))
