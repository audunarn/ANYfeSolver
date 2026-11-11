"""G0 F02 scalar physical torque path through a finite membrane footprint."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEMesh, Material


def specimen(n: int, topology: str) -> dict:
    coords = np.linspace(-1., 1., n + 1)
    mesh = FEMesh()
    for j, y in enumerate(coords):
        for i, x in enumerate(coords):
            mesh.add_node(1 + j * (n + 1) + i, float(x), float(y), 0.)
    law = Material("steel", 210e9, .3, density=7850.)
    rows, cols, values = [], [], []
    cells = 0
    for j in range(n):
        for i in range(n):
            a = 1 + j * (n + 1) + i
            b = a + 1
            d = a + n + 1
            c = d + 1
            patches = ((a, b, c, d),) if topology == "Q4" else ((a, b, c), (a, c, d))
            for nodes in patches:
                cells += 1
                shell = create_shell_element(cells, list(nodes), "steel",
                    formulation="e4-pl" if topology == "Q4" else "e4-pl-s3-v2d",
                    thickness=.07, reference_normal=(0., 0., 1.))
                physical = np.asarray(shell.compute_stiffness_components(mesh, law)["physical"])
                selection = [6 * k + axis for k in range(len(nodes)) for axis in (0, 1)]
                sub = physical[np.ix_(selection, selection)]
                dofs = [2 * (node - 1) + axis for node in nodes for axis in (0, 1)]
                for r, dof_r in enumerate(dofs):
                    for col, dof_c in enumerate(dofs):
                        rows.append(dof_r)
                        cols.append(dof_c)
                        values.append(sub[r, col])
    count = 2 * (n + 1)**2
    stiffness = sparse.coo_matrix((values, (rows, cols)), shape=(count, count)).tocsr()
    displacement = np.zeros(count)
    outer, footprint = set(), set()
    for j, y in enumerate(coords):
        for i, x in enumerate(coords):
            node = j * (n + 1) + i
            if i in (0, n) or j in (0, n):
                outer.update((2 * node, 2 * node + 1))
            if abs(x) <= .25 + 1e-12 and abs(y) <= .25 + 1e-12:
                footprint.update((2 * node, 2 * node + 1))
                displacement[2 * node:2 * node + 2] = (-y, x)
    fixed = np.array(sorted(outer | footprint))
    free = np.array(sorted(set(range(count)) - set(fixed)))
    displacement[free] = spsolve(stiffness[free, :][:, free].tocsc(),
                                -stiffness[free, :][:, fixed] @ displacement[fixed])
    reaction = stiffness @ displacement
    moment = 0.
    for j, y in enumerate(coords):
        for i, x in enumerate(coords):
            node = j * (n + 1) + i
            if 2 * node in footprint:
                moment += -y * reaction[2 * node] + x * reaction[2 * node + 1]
    work = float(displacement @ reaction)
    relative = abs(moment - work) / max(1., abs(moment), abs(work))
    return {"n": n, "topology": topology, "shells": cells,
            "footprint_nodes": len(footprint) // 2, "moment": float(moment),
            "twice_energy": work, "work_error": relative,
            "free_equilibrium": float(np.linalg.norm(reaction[free]) /
                                      max(1., np.linalg.norm(reaction[fixed])))}


def main() -> int:
    cases = [specimen(n, topology) for topology in ("Q4", "S3-V2D") for n in (8, 16)]
    passed = all(row["moment"] > 0. and np.isfinite(row["moment"])
                 and row["work_error"] <= 1e-8 and row["free_equilibrium"] <= 1e-8
                 for row in cases)
    print(json.dumps({"case": "G0_F02_MEMBRANE_FOOTPRINT_V1",
                      "status": "diagnostic_pass" if passed else "failed", "results": cases}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
