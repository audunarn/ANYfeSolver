"""Bounded, independent two-sided BL5 body-port response diagnostic.

This is an incremental *body* comparison, not the V5 loaded solve, an F03
assessment, or a qualification of a beam/shell connection. Each continuum
half uses its original C3D20 element matrices with duplicated cut nodes.
Only the six weighted rigid-section coordinates at the remote face are fixed;
all face warping remains free. The native halves retain their actual 24 local
internal coordinates and independent cut-side half-cell rotations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

# The supervisor invokes this source with -I. Bind the adjacent source checkout
# explicitly so the development fixtures cannot resolve a different scripts
# namespace from an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._stiffened_shell_ge_experimental import _beam_reference
from scripts import r8_bl5_matched_native_reference as native
from scripts import r8_bl5_released_director_reference as solid
from scripts import r8_bl5_weak_continuum_reference as weak


IDENTITY = "R8_BL5_TWO_SIDED_BODY_PORT_RESPONSE_DEV_V1"
CENTER = np.array((.5, 0., .04))
ACTION = np.array((0., 5.59377556437, 0., .123123784342, 0., -.402617092486))


def _solve(k, c, f):
    """Solve the small indefinite constrained system, without a dense inverse."""
    k = sparse.csc_matrix(k)
    c = sparse.csc_matrix(c)
    # Native B3 retains an indefinite mixed internal block. A positive
    # diagonal is neither expected nor required; scale by absolute row work
    # without modifying the operator or assuming positive definiteness.
    row_scale = np.asarray(abs(k).sum(axis=1)).ravel()
    if np.any(row_scale <= 0.) or not np.isfinite(row_scale).all():
        raise ValueError("finite nonzero mixed-operator rows required")
    dq = 1. / np.sqrt(row_scale)
    cd = c @ sparse.diags(dq)
    row_norm = np.sqrt(np.asarray(cd.power(2).sum(axis=1)).ravel())
    if np.any(row_norm <= 0.) or not np.isfinite(row_norm).all():
        raise ValueError("independent finite remote section rows required")
    dl = 1. / row_norm
    d = sparse.diags(dq)
    l = sparse.diags(dl)
    system = sparse.bmat(((d @ k @ d, d @ c.T @ l),
                          (l @ c @ d, None)), format="csc")
    rhs = np.r_[dq * f, np.zeros(c.shape[0])]
    factor = splu(system)
    z = factor.solve(rhs)
    for _ in range(3):
        z += factor.solve(rhs - system @ z)
    q = dq * z[:len(f)]
    multiplier = dl * z[len(f):]
    equation = k @ q + c.T @ multiplier - f
    scale = max(np.linalg.norm(f), np.linalg.norm(k @ q),
                np.linalg.norm(c.T @ multiplier), 1e-14)
    error = float(np.linalg.norm(equation) / scale)
    gap = float(np.linalg.norm(c @ q))
    if error > 1e-9 or gap > 1e-9 or not np.isfinite(q).all():
        raise ValueError(f"port equation/remote-pose failure: {error}, {gap}")
    return q, dict(equation_relative=error, remote_gap=gap,
                   kkt_dimension=int(system.shape[0]), k_nnz=int(k.nnz))


def _solid_half(model, local_elements, side):
    element_ids = tuple(eid for eid in model["beam_elements"]
                        if (max(model["physical_nodes"][n][0] for n in model["elements"][eid])
                            <= .5 if side == "left" else
                            min(model["physical_nodes"][n][0] for n in model["elements"][eid]) >= .5))
    if len(element_ids) != 2:
        raise ValueError("expected two original C3D20 elements per half")
    node_ids = tuple(sorted({n for eid in element_ids for n in model["elements"][eid]}))
    index = {n: 3 * i for i, n in enumerate(node_ids)}
    rows, cols, data = [], [], []
    for eid in element_ids:
        _, ke = local_elements[eid]
        slots = np.array([index[n] + axis for n in model["elements"][eid]
                          for axis in range(3)])
        rows.extend(np.repeat(slots, 60))
        cols.extend(np.tile(slots, 60))
        data.extend(ke.ravel())
    k = sparse.coo_matrix((data, (rows, cols)),
                          shape=(3 * len(node_ids),) * 2).tocsr()
    k.sum_duplicates()
    return element_ids, node_ids, index, k


def _face_map(model, element_ids, index, x):
    """L2 projected six section coordinates and their dual load on a Q8 face."""
    face_nodes = tuple(n for n in index
                       if abs(model["physical_nodes"][n][0] - x) < 1e-14)
    fi = {n: 3 * i for i, n in enumerate(face_nodes)}
    m = np.zeros((3 * len(face_nodes),) * 2)
    points, weights = np.polynomial.legendre.leggauss(3)
    for eid in element_ids:
        ids, low, high, _, _ = solid._brick(model["elements"][eid],
                                            model["physical_nodes"])
        if abs((low[0] if abs(x - low[0]) < 1e-14 else high[0]) - x) > 1e-14:
            continue
        if abs(x - low[0]) > 1e-14 and abs(x - high[0]) > 1e-14:
            continue
        xi = -1. if abs(x - low[0]) < 1e-14 else 1.
        jacobian = (high[1] - low[1]) * (high[2] - low[2]) / 4.
        for iy, y in enumerate(points):
            for iz, z in enumerate(points):
                shape, _ = solid._shape((xi, y, z))
                trace = np.zeros(3 * len(face_nodes))
                for n, value in zip(ids, shape):
                    if n in fi:
                        trace[fi[n]] += value
                    elif abs(value) > 1e-13:
                        raise ValueError("non-face C3D20 interpolation weight")
                scalar = np.outer(trace[::3], trace[::3])
                for axis in range(3):
                    m[axis::3, axis::3] += scalar * jacobian * weights[iy] * weights[iz]
    if not np.isfinite(m).all() or np.linalg.norm(m) == 0.:
        raise ValueError("positive face L2 mass required")
    r = np.zeros((3 * len(face_nodes), 6))
    for n, j in fi.items():
        arm = np.asarray(model["physical_nodes"][n]) - CENTER
        r[j:j+3, :3] = np.eye(3)
        for axis in range(3):
            r[j:j+3, 3+axis] = np.cross(np.eye(3)[axis], arm)
    gram = r.T @ m @ r
    if np.linalg.eigvalsh(gram)[0] <= 0.:
        raise ValueError("six independent physical rigid-section coordinates required")
    projector = np.linalg.solve(gram, r.T @ m)
    load = m @ r @ np.linalg.solve(gram, np.eye(6))
    full_projection = np.zeros((6, 3 * len(index)))
    full_load = np.zeros((3 * len(index), 6))
    for n, j in fi.items():
        slot = index[n]
        full_projection[:, slot:slot+3] = projector[:, j:j+3]
        full_load[slot:slot+3] = load[j:j+3]
    if np.max(abs(r.T @ load - np.eye(6))) > 1e-12:
        raise ValueError("face wrench and section coordinate are not dual")
    return full_projection, full_load, dict(nodes=len(face_nodes),
        gram_condition=float(np.linalg.cond(gram)), area=float(np.sum(m[::3, ::3])))


def _continuum(model, elements, side, p):
    ids, _, index, k = _solid_half(model, elements, side)
    cut, remote = (.5, 0.) if side == "left" else (.5, 1.)
    a, fshape, cut_meta = _face_map(model, ids, index, cut)
    c, _, remote_meta = _face_map(model, ids, index, remote)
    f = fshape @ p
    q, checks = _solve(k, c, f)
    response = a @ q
    work = float(f @ q)
    energy = float(q @ (k @ q))
    mismatch = abs(work-energy)/max(abs(work), abs(energy), 1e-20)
    if mismatch > 1e-8 or abs(work-float(p @ response)) > max(abs(work), 1e-20)*1e-12:
        raise ValueError("continuum work/energy duality failed")
    return dict(side=side, action=p.tolist(), pose=response.tolist(),
                work_J=work, energy_J=energy, work_energy_relative=mismatch,
                cut=cut_meta, remote=remote_meta, **checks)


def _native(side, p):
    definitions = native.definition().beams
    ids, index, nodes, pieces, count, _ = _beam_reference(definitions)
    element, slots, origin = pieces[0 if side == "left" else 1]
    positions = np.array([nodes[node][0] for node in element.node_ids])
    frames = np.array([nodes[node][1] for node in element.node_ids])
    result = element.operator.evaluate(positions, np.zeros((3, 3)), frames,
        np.tile(np.eye(3), (2, 1, 1)), np.zeros(18),
        origin=origin, check=element.operator.guard)
    k = spatial_jacobian(result.residual, result.hessian + result.hessian_low)
    c = np.zeros((6, 42))
    remote = 0 if side == "left" else 12
    c[:, remote:remote+6] = np.eye(6)
    port = np.zeros((6, 42))
    cut = 12 if side == "left" else 0
    port[:3, cut:cut+3] = np.eye(3)
    retained = 21 if side == "left" else 18
    port[3:, retained:retained+3] = np.eye(3)
    f = port.T @ p
    q, checks = _solve(k, c, f)
    response = port @ q
    work = float(f @ q)
    energy = float(q @ (k @ q))
    mismatch = abs(work-energy)/max(abs(work), abs(energy), 1e-20)
    if mismatch > 1e-8 or abs(work-float(p @ response)) > max(abs(work), 1e-20)*1e-12:
        raise ValueError("native work/energy duality failed")
    return dict(side=side, action=p.tolist(), pose=response.tolist(),
                work_J=work, energy_J=energy, work_energy_relative=mismatch,
                native_retained_spin_local_slots=list(range(retained,retained+3)),
                **checks)


def main(output):
    model = weak.specimen()
    _, elements = weak.assemble_body(model)
    checks = {}
    for label, p in (("v7_resultant", ACTION),
                     ("pure_mz", np.array((0., 0., 0., 0., 0., 1.)))):
        checks[label] = {}
        for name, evaluator in (("continuum", lambda side, action: _continuum(model, elements, side, action)),
                                ("native", _native)):
            left = evaluator("left", p)
            right = evaluator("right", -p)
            total = left["work_J"] + right["work_J"]
            checks[label][name] = dict(left=left, right=right,
                relative_compliance_J=total,
                relative_pose=(np.asarray(left["pose"])-np.asarray(right["pose"])).tolist())
    result = dict(identity=IDENTITY, status="DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFIED",
        cut_center_m=CENTER.tolist(), applied_wrench=ACTION.tolist(),
        method="incremental separate halves; remote six-pose restraint; free solid warping; native retained cut spin",
        caveat="new remote supports and no original V5 load; body-port response only, not V5 work allocation or BL5/F03 acceptance",
        checks=checks)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
