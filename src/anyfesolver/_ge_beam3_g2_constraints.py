"""Private G2 physical constraints; analytic chart derivatives, no mechanics."""
from dataclasses import dataclass
from fractions import Fraction
import json
import numpy as np
from ._ge_beam3_g1_elastic import owned, canonical, sha
from ._ge_beam3_p5.arrays import _frames
from ._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, transpose, so3_exp, so3_log

SCHEMA = "GE_BEAM3_G2_CONSTRAINTS_V1"


def number(value):
    return float(owned(value, ()))


def axes(value, *, empty=False):
    b = owned(value)
    if b.ndim != 2 or b.shape[0] != 3 or b.shape[1] not in ((0, 1, 2, 3) if empty else (1, 2, 3)):
        raise ValueError("explicit selected orientation axes required")
    if np.linalg.norm(b.T @ b-np.eye(b.shape[1])) > 1e-11:
        raise ValueError("orthonormal selected axes required")
    return b.tolist()


def compile_affine(size, rows):
    """Exact rational dependency-first expansion; no rank-repair heuristics."""
    if type(size) is not int or not 1 <= size <= 36 or type(rows) not in (tuple, list) or len(rows) > size:
        raise ValueError("bounded affine inventory required")
    by = {}; order = []; active = set(); resolved = {}
    def dof(d):
        if type(d) is not int or not 0 <= d < size or d % 6 >= 3:
            raise ValueError("affine constraints require translation DOFs")
        return d
    for row in rows:
        if type(row) is not dict or set(row) != {"dependent", "masters", "offset", "rate"}:
            raise ValueError("affine row schema")
        d = dof(row["dependent"])
        if d in by: raise ValueError("duplicate dependent DOF")
        masters = {}; number(row["offset"]); number(row["rate"])
        if type(row["masters"]) not in (tuple, list): raise ValueError("master list required")
        for pair in row["masters"]:
            if len(pair) != 2: raise ValueError("master pair required")
            m, a = pair; dof(m)
            if m in masters: raise ValueError("duplicate master DOF")
            masters[m] = Fraction(number(a))
        by[d] = (masters, Fraction(number(row["offset"])), Fraction(number(row["rate"])))
    free = tuple(d for d in range(size) if d not in by)
    def expand(d):
        if d in resolved: return resolved[d]
        if d in active: raise ValueError("cyclic affine constraints")
        if d not in by: return ({d: Fraction(1)}, Fraction(0), Fraction(0))
        active.add(d); weights = {}; masters, b, c = by[d]
        for m, a in sorted(masters.items()):
            w, mb, mc = expand(m)
            for k, v in w.items(): weights[k] = weights.get(k, Fraction(0))+a*v
            b += a*mb; c += a*mc
        active.remove(d); order.append(d); resolved[d] = (weights, b, c)
        return resolved[d]
    for d in sorted(by): expand(d)
    rows_exact = [expand(d) for d in range(size)]
    T = owned([[float(w.get(k, 0)) for k in free] for w, _, _ in rows_exact], (size, len(free)))
    offset = owned([float(b) for _, b, _ in rows_exact])
    rate = owned([float(c) for _, _, c in rows_exact])
    return T, offset, rate, tuple(order)


@dataclass(frozen=True, eq=False, init=False)
class ConstraintSet:
    _raw: bytes

    def __init__(self, node_ids, positions, frames, *, affine=(), orientations=(), poses=()):
        if (type(node_ids) is not tuple or not 3 <= len(node_ids) <= 6 or
                any(type(n) is not int or n <= 0 for n in node_ids) or len(set(node_ids)) != len(node_ids)):
            raise ValueError("explicit unique native node IDs required")
        x = owned(positions, (len(node_ids), 3))
        R = _frames(frames, len(node_ids), "physical node frames")
        if any(type(rows) not in (tuple, list) or len(rows) > 6*len(node_ids) for rows in (affine, orientations, poses)):
            raise ValueError("bounded constraint inventory required")
        def node(n):
            if type(n) is not int or n not in node_ids: raise ValueError("unknown constraint node")
        arows = []
        compile_affine(6*len(node_ids), affine)
        for r in affine:
            arows.append(dict(dependent=r["dependent"], masters=[[m, number(a)] for m, a in r["masters"]],
                              offset=number(r["offset"]), rate=number(r["rate"])))
        orows = []
        for r in orientations:
            if type(r) is not dict or set(r) != {"node", "axes", "target"}: raise ValueError("orientation schema")
            node(r["node"])
            orows.append(dict(node=r["node"], axes=axes(r["axes"]),
                              target=_frames([r["target"]], 1, "target frame")[0].tolist()))
        prows = []
        for r in poses:
            if type(r) is not dict or set(r) != {"master", "slave", "offset", "frame", "axes"}:
                raise ValueError("relative pose schema")
            node(r["master"]); node(r["slave"])
            if r["master"] == r["slave"]: raise ValueError("distinct relative pose nodes required")
            prows.append(dict(master=r["master"], slave=r["slave"], offset=owned(r["offset"], (3,)).tolist(),
                              frame=_frames([r["frame"]], 1, "relative frame")[0].tolist(),
                              axes=axes(r["axes"], empty=True)))
        if not arows and not orows and not prows: raise ValueError("explicit constraints required")
        if len(arows)+sum(len(r["axes"][0]) for r in orows)+sum(3+len(r["axes"][0]) for r in prows) > 6*len(node_ids):
            raise ValueError("more constraints than coordinates")
        body = dict(schema=SCHEMA, node_ids=node_ids, positions=x, frames=R,
                    affine=arows, orientations=orows, poses=prows)
        object.__setattr__(self, "_raw", canonical(body))

    def descriptor(self): return json.loads(self._raw)

    @property
    def identity(self): return sha(self.descriptor())

    @classmethod
    def from_descriptor(cls, body):
        if type(body) is not dict or set(body) != {"schema", "node_ids", "positions", "frames", "affine", "orientations", "poses"} or body["schema"] != SCHEMA:
            raise ValueError("constraint descriptor schema")
        made = cls(tuple(body["node_ids"]), body["positions"], body["frames"], affine=body["affine"],
                   orientations=body["orientations"], poses=body["poses"])
        if made._raw != canonical(body): raise ValueError("noncanonical constraint definition")
        return made

    def target_frames(self, targets=None):
        rows = self.descriptor()["orientations"]
        if not rows:
            result = owned(np.zeros((0, 3, 3)) if targets is None or (type(targets) is list and targets == []) else targets, (0, 3, 3))
        else:
            result = owned(_frames([r["target"] for r in rows] if targets is None else targets, len(rows), "target program"))
        return result

    def evaluate(self, total, committed, rotations, *, control=0., targets=None):
        body = self.descriptor(); ids = body["node_ids"]; n = len(ids); size = 6*n
        total, committed = owned(total, (size,)), owned(committed, (size,))
        q = _frames(rotations, n, "committed shared rotations"); load = number(control)
        target = self.target_frames(targets); variables = [Jet2.variable(v, i, size) for i, v in enumerate(total)]
        D = {}; X = {}; rows = []; rates = []
        for i, node in enumerate(ids):
            delta = [variables[6*i+3+j]-committed[6*i+3+j] for j in range(3)]
            D[node] = matmul(so3_exp(delta), constant_matrix(q[i] @ np.asarray(body["frames"][i]), size))
            X[node] = [variables[6*i+j]+body["positions"][i][j] for j in range(3)]
        for r in body["affine"]:
            value = variables[r["dependent"]]-r["offset"]-r["rate"]*load
            for m, a in r["masters"]: value -= a*variables[m]
            rows.append(value); rates.append(-r["rate"])
        for r, frame in zip(body["orientations"], target):
            vector = so3_log(matmul(constant_matrix(frame.T, size), D[r["node"]]))
            for axis in np.asarray(r["axes"]).T:
                rows.append(sum((v*a for v, a in zip(vector, axis)), Jet2.constant(0., size))); rates.append(0.)
        for r in body["poses"]:
            master, slave = r["master"], r["slave"]
            for j in range(3):
                value = X[slave][j]-X[master][j]-sum((D[master][j][k]*r["offset"][k] for k in range(3)), Jet2.constant(0., size))
                rows.append(value); rates.append(0.)
            if np.asarray(r["axes"]).shape[1]:
                vector = so3_log(matmul(constant_matrix(np.asarray(r["frame"]).T, size), matmul(transpose(D[master]), D[slave])))
                for axis in np.asarray(r["axes"]).T:
                    rows.append(sum((v*a for v, a in zip(vector, axis)), Jet2.constant(0., size))); rates.append(0.)
        g = owned([v.value for v in rows]); J = owned([v.gradient for v in rows]); H = owned([v.hessian for v in rows])
        singular = np.linalg.svd(J, compute_uv=False)
        floor = 64*np.finfo(float).eps*max(J.shape)*max(1., singular[0])
        if len(rows) > size or singular[-1] <= floor: raise ValueError("rank-deficient constraint Jacobian")
        return g, J, H, owned(rates)
