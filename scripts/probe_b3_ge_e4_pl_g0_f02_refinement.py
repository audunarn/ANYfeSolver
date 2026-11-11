"""G0 F02-R: physical in-plane footprint thickness/refinement screen."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEMesh, Material


def specimen(n: int, topology: str, thickness: float) -> dict:
    coordinates = np.linspace(-1., 1., n + 1)
    mesh = FEMesh()
    for j, y in enumerate(coordinates):
        for i, x in enumerate(coordinates):
            mesh.add_node(1 + j * (n + 1) + i, float(x), float(y), 0.)
    material = Material("steel", 210e9, .3, density=7850.)
    row, col, data = [], [], []
    cells = 0
    for j in range(n):
        for i in range(n):
            a = 1 + j * (n + 1) + i
            b, d = a + 1, a + n + 1
            c = d + 1
            patches = ((a, b, c, d),) if topology == "Q4" else (
                (a, b, c), (a, c, d))
            for nodes in patches:
                cells += 1
                shell = create_shell_element(
                    cells, list(nodes), "steel",
                    formulation="e4-pl" if topology == "Q4" else "e4-pl-s3-v2d",
                    thickness=thickness, reference_normal=(0., 0., 1.))
                physical = np.asarray(
                    shell.compute_stiffness_components(mesh, material)["physical"])
                local = [6 * k + axis for k in range(len(nodes))
                         for axis in (0, 1)]
                block = physical[np.ix_(local, local)]
                global_dof = [2 * (node - 1) + axis for node in nodes
                              for axis in (0, 1)]
                for a_local, a_dof in enumerate(global_dof):
                    for b_local, b_dof in enumerate(global_dof):
                        row.append(a_dof)
                        col.append(b_dof)
                        data.append(block[a_local, b_local])
    size = 2 * (n + 1) ** 2
    k = sparse.coo_matrix(
        (data, (row, col)), shape=(size, size)).tocsr()
    u = np.zeros(size)
    outer, footprint = set(), set()
    for j, y in enumerate(coordinates):
        for i, x in enumerate(coordinates):
            node = j * (n + 1) + i
            if i in (0, n) or j in (0, n):
                outer.update((2 * node, 2 * node + 1))
            if abs(x) <= .25 + 1e-12 and abs(y) <= .25 + 1e-12:
                footprint.update((2 * node, 2 * node + 1))
                u[2 * node:2 * node + 2] = (-y, x)
    fixed = np.array(sorted(outer | footprint))
    free = np.array(sorted(set(range(size)) - set(fixed)))
    u[free] = spsolve(k[free, :][:, free].tocsc(),
                      -k[free, :][:, fixed] @ u[fixed])
    r = k @ u
    moment = sum(
        -y * r[2 * (j * (n + 1) + i)]
        + x * r[2 * (j * (n + 1) + i) + 1]
        for j, y in enumerate(coordinates)
        for i, x in enumerate(coordinates)
        if 2 * (j * (n + 1) + i) in footprint)
    energy2 = float(u @ r)
    constrained_norm = float(np.linalg.norm(r[fixed]))
    free_norm = float(np.linalg.norm(r[free]))
    return {
        "n": n, "topology": topology, "thickness": thickness,
        "shells": cells, "footprint_nodes": len(footprint) // 2,
        "matrix_dimension": size, "nnz": k.nnz,
        "physical_moment": float(moment), "twice_energy": energy2,
        "work_relative": abs(moment - energy2) / max(abs(moment), abs(energy2)),
        "free_relative": free_norm / constrained_norm,
        "absolute_free_residual": free_norm,
        "absolute_constrained_reaction": constrained_norm,
    }


def main() -> int:
    rows = [specimen(n, topology, t)
            for topology in ("Q4", "S3-V2D")
            for n in (8, 16, 32) for t in (.035, .07)]
    comparisons = []
    for topology in ("Q4", "S3-V2D"):
        series = []
        for n in (8, 16, 32):
            thin, thick = (next(row for row in rows
                                if row["topology"] == topology
                                and row["n"] == n and row["thickness"] == t)
                           for t in (.035, .07))
            series.append(thick["physical_moment"])
            comparisons.append({
                "topology": topology, "n": n,
                "thick_over_thin": thick["physical_moment"] / thin["physical_moment"],
                "thickness_ratio_error": abs(thick["physical_moment"] /
                                             thin["physical_moment"] - 2),
            })
        for i in (0, 1):
            comparisons.append({
                "topology": topology, "refinement": [8, 16] if i == 0 else [16, 32],
                "successive_relative_change": abs(series[i + 1] - series[i]) /
                                              abs(series[i + 1]),
            })
    passed = (
        len(rows) == 12
        and all(np.isfinite(row["physical_moment"])
                and row["physical_moment"] > 0
                and np.isfinite(row["twice_energy"])
                and np.isfinite(row["free_relative"])
                and np.isfinite(row["absolute_constrained_reaction"])
                and row["absolute_constrained_reaction"] > 0
                and row["work_relative"] <= 1e-8
                and row["free_relative"] <= 1e-8 for row in rows)
        and all(item["thickness_ratio_error"] <= 1e-10
                for item in comparisons if "thickness_ratio_error" in item)
    )
    print(json.dumps({
        "case": "G0_F02_REFINEMENT_V1",
        "status": "PREPARATORY_MEMBRANE_REFINEMENT_PASS" if passed else "FAILED",
        "rows": rows, "comparisons": comparisons,
        "scope": "in-plane physical channel only; no director/shear or F02 acceptance",
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
