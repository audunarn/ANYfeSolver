"""Private G3a native graph owner; no mixed-family admission or public routing.

G2's constrained Newton/transaction equations are retained in a separate owner.
G1/G2 solve dispatch is deliberately not an entry point to this owner.
"""
from hashlib import sha256
import json
import threading
import os
from contextlib import contextmanager
from time import monotonic
import numpy as np
from .fe_core import FEModel
from .nonlinear_state import NonlinearStateStore, create_model_native_rotation_store
from ._ge_beam3_g1_analysis import ElasticAnalysis, runtime_digest, describe
from ._ge_beam3_g1_elastic import owned, canonical, sha, solve, ElasticSection
from ._ge_beam3_g1_element import ElasticElement
from ._ge_beam3_g3_constraints import GraphConstraintSet, number
from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_p5.chart import exp_chart_terms

SCHEMA = "GE_BEAM3_G3_GRAPH_ELASTIC_RESTART_V1"
MAX_BYTES = 8*1024*1024
_ENTRY = object()
POLICY = "NATIVE_EXACT_ELASTIC_ONLY_NO_CROSS_FAMILY_ADAPTERS"


def progress(stage):
    if os.environ.get("G3_PROGRESS") == "1":
        print("G3 CHECKPOINT "+stage, flush=True)


def topology(elements, constraints):
    """Validate bounded incidence before model allocation or state evaluation."""
    if (type(elements) is not tuple or not 1 <= len(elements) <= 8 or
            any(type(e) is not ElasticElement for e in elements) or
            type(constraints) is not GraphConstraintSet):
        raise ValueError("G3 requires 1..8 exact native elastic elements and graph constraints")
    if len({e.element_id for e in elements}) != len(elements):
        raise ValueError("unique element IDs required")
    body = constraints.descriptor()
    ids = body["node_ids"]; positions = np.asarray(body["positions"])
    if len(ids) > 32 or 6*len(ids)+24*len(elements) > 384:
        raise ValueError("G3 graph allocation bound")
    lookup = dict(zip(ids, positions)); incidence = {n: [] for n in ids}
    adjacency = {n: set() for n in ids}; midpoints = set(); endpoints = set()
    for e in sorted(elements, key=lambda e: e.element_id):
        points = e.operator.reference.coordinates
        if not np.array_equal(points[1], (points[0]+points[2])/2):
            raise ValueError("G3 straight exact-midpoint fixture required")
        for n, x in zip(e.node_ids, points):
            if n not in lookup or not np.array_equal(x, lookup[n]):
                raise ValueError("G3 unknown node or conflicting shared coordinates")
            incidence[n].append(e.element_id)
        a, m, b = e.node_ids
        if m in midpoints: raise ValueError("shared midpoint ownership")
        midpoints.add(m); endpoints.update((a, b))
        for left, right in ((a, m), (m, b)):
            adjacency[left].add(right); adjacency[right].add(left)
    if midpoints & endpoints or any(not owners for owners in incidence.values()):
        raise ValueError("G3 midpoint conflict or orphan node")
    components = []; remaining = set(ids)
    while remaining:
        active = [min(remaining)]; found = set()
        while active:
            n = active.pop()
            if n in found: continue
            found.add(n); active.extend(sorted(adjacency[n]-found, reverse=True))
        remaining -= found; components.append(sorted(found))
    return dict(policy=POLICY, adapter_allowlist=[], components=components,
                cycle_rank=2*len(elements)-len(ids)+len(components),
                incidence=[[n, incidence[n]] for n in ids],
                element_ids=sorted(e.element_id for e in elements), node_ids=ids,
                external_dofs=6*len(ids), internal_coordinates=24*len(elements))


def rigid_basis(constraints, components):
    """Analytical independent rigid spaces for every incidence component."""
    body = constraints.descriptor(); ids = body["node_ids"]
    R = np.zeros((6*len(ids), 6*len(components)))
    for c, component in enumerate(components):
        origin = np.asarray(body["positions"][ids.index(component[0])])
        for n in component:
            i = ids.index(n); x = np.asarray(body["positions"][i])-origin
            R[6*i:6*i+3, 6*c:6*c+3] = np.eye(3)
            R[6*i:6*i+3, 6*c+3:6*c+6] = np.column_stack([np.cross(a, x) for a in np.eye(3)])
            R[6*i+3:6*i+6, 6*c+3:6*c+6] = np.eye(3)
    return owned(R)


def restart_graph_preflight(definition, graph):
    """Reconstruct serialized graph authority without element or state work."""
    rows = definition["elements"]
    if type(rows) is not list or not 1 <= len(rows) <= 8:
        raise ValueError("G3 restart element bound")
    ids = []; connectivity = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError("G3 restart element schema")
        eid, nodes = row.get("element_id"), row.get("node_ids")
        if (type(eid) is not int or eid <= 0 or type(nodes) is not list or len(nodes) != 3 or
                any(type(n) is not int or n <= 0 for n in nodes) or len(set(nodes)) != 3):
            raise ValueError("G3 restart connectivity schema")
        ids.append(eid); connectivity.append((eid, nodes))
    if len(set(ids)) != len(ids): raise ValueError("G3 duplicate restart elements")
    nodes = sorted({n for _, row in connectivity for n in row})
    if len(nodes) > 32 or 6*len(nodes)+24*len(rows) > 384:
        raise ValueError("G3 restart graph bound")
    incidence = {n: [] for n in nodes}; adjacency = {n: set() for n in nodes}
    for eid, row in sorted(connectivity):
        for n in row: incidence[n].append(eid)
        for a, b in zip(row, row[1:]):
            adjacency[a].add(b); adjacency[b].add(a)
    remaining = set(nodes); components = []
    while remaining:
        active = [min(remaining)]; found = set()
        while active:
            n = active.pop()
            if n in found: continue
            found.add(n); active.extend(sorted(adjacency[n]-found, reverse=True))
        remaining -= found; components.append(sorted(found))
    expected = dict(policy=POLICY, adapter_allowlist=[], components=components,
                    cycle_rank=2*len(rows)-len(nodes)+len(components),
                    incidence=[[n, incidence[n]] for n in nodes], element_ids=sorted(ids),
                    node_ids=nodes, external_dofs=6*len(nodes), internal_coordinates=24*len(rows))
    # Canonical comparison rejects bool/int substitution and extra/missing keys.
    if canonical(graph) != canonical(expected):
        raise ValueError("G3 restart graph preflight mismatch")



class NativeGraphAnalysis:
    """One shared pose store, element-owned internals, bounded native graph."""

    # Reuse assembly/state validation, not a nested owner or a base solve.
    free = ElasticAnalysis.free
    _graph_identity = ElasticAnalysis._graph_identity
    _state_digest = ElasticAnalysis._state_digest
    publish = ElasticAnalysis.publish

    def __init__(self, elements, constraints):
        if type(self) is not NativeGraphAnalysis:
            raise ValueError("exact G3 owner required")
        inventory = topology(elements, constraints)
        if any(e._mesh is not None or e._owner_guard is not None for e in elements):
            raise ValueError("G3 requires fresh exclusively owned elements")
        progress("capture")
        self.elements = tuple(sorted(elements, key=lambda e: e.element_id))
        self.constraints = constraints; self._constraint_identity = constraints.identity
        self._inventory = canonical(inventory)
        body = constraints.descriptor(); self.size = 6*len(body["node_ids"])
        _, J, _, _ = constraints.evaluate(np.zeros(self.size), np.zeros(self.size),
                                          np.tile(np.eye(3), (self.size//6, 1, 1)))
        R = rigid_basis(constraints, inventory["components"])
        constrained_rigid = J @ R
        singular = np.linalg.svd(constrained_rigid, compute_uv=False)
        floor = 64*np.finfo(float).eps*max(constrained_rigid.shape)*max(1., singular[0])
        if len(singular) < R.shape[1] or singular[-1] <= floor:
            raise ValueError("G3 unsupported component rigid space")
        self.model = FEModel("G3 native graph elastic")
        for n, x in zip(body["node_ids"], body["positions"]): self.model.add_node(n, *x)
        for e in self.elements:
            self.model.add_element(e.element_id, e); self.model.materials[e.material_name] = e.section
        self.fixed = (); self._free = np.frombuffer(np.arange(self.size, dtype=np.int64).tobytes(), dtype=np.int64)
        self._definition = canonical(dict(elements=[describe(e) for e in self.elements], fixed=[]))
        self._runtime = runtime_digest(); self._lock = threading.Lock(); self._active_loads = None
        from .matrix_assembly import _get_cached_sparsity_pattern
        _get_cached_sparsity_pattern(self.model.mesh, "tangent_stiffness")
        self._sparsity_identity = sha(self.model.mesh._sparsity_cache)
        progress("local solve")
        initial = {e.element_id: e.init_model_bound_nonlinear_state(self.model.mesh, e.section, 1) for e in self.elements}
        self.store = NonlinearStateStore.from_shell_layouts((), initial)
        self.store.attach_native_rotation_store(create_model_native_rotation_store(self.model, initial, np.zeros(self.size)))
        self._owned_store = self.store
        self._owned_rotations = self.store.native_rotation_store
        self._accepted = (); self._journal_digest = sha(self._accepted)
        self._initial_digest = self._state_digest(); self._graph = self._graph_identity()
        for e in self.elements: e._owner_guard = self._guard
        self._guard()

    @property
    def inventory(self):
        return json.loads(self._inventory)

    def _guard(self):
        if (type(self) is not NativeGraphAnalysis or type(self.constraints) is not GraphConstraintSet or
                self.constraints.identity != self._constraint_identity or
                canonical(topology(self.elements, self.constraints)) != self._inventory):
            raise ValueError("G3 frozen graph/constraint authority changed")
        mesh = self.model.mesh
        if self.store is not self._owned_store or self.store.native_rotation_store is not self._owned_rotations:
            raise ValueError("G3 shared state owner changed")
        if (list(mesh.nodes) != self.inventory["node_ids"] or
                list(mesh.elements) != self.inventory["element_ids"] or
                set(self.model.materials) != {"default", *(e.material_name for e in self.elements)} or
                sha(getattr(mesh, "_sparsity_cache", None)) != self._sparsity_identity):
            raise ValueError("G3 assembly ordering/cache authority changed")
        for i, e in enumerate(self.elements):
            if e._load is not None:
                if (self._active_loads is None or e._load[0] is not self.store or
                        not np.array_equal(e._load[1], self._active_loads[0][i]) or
                        not np.array_equal(e._load[2], self._active_loads[1][i])):
                    raise ValueError("G3 unbound load channel")
        ElasticAnalysis._guard(self)

    @contextmanager
    def _exclusive(self, entry=None):
        # In particular, unbound G2.solve(self, ...) supplies no G3 capability.
        if entry is not _ENTRY or type(self) is not NativeGraphAnalysis:
            raise ValueError("G3 exact-owner entry required; G1/G2 dispatch forbidden")
        if not self._lock.acquire(blocking=False): raise RuntimeError("G3 owner already in use")
        try:
            self._guard()
            yield
        finally:
            self._lock.release()

    def _evaluate(self, total, nodal, lines, couples):
        self._guard()
        self._active_loads = (owned(lines), owned(couples))
        progress("assembly")
        try:
            return ElasticAnalysis._evaluate(self, total, nodal, *self._active_loads)
        finally:
            self._active_loads = None

    def recover(self):
        with self._exclusive(_ENTRY):
            if self.store.has_active_trial: raise RuntimeError("committed recovery required")
            return {e.element_id: e.operator.recover(self.store[e.element_id]["response"]["rotations"],
                                                    self.store[e.element_id]["response"]["resultants"])
                    for e in self.elements}

    def reference_rhs(self, right_hand_sides):
        """Ephemeral multi-RHS solve; no persistent numerical factor cache.

        Restricted to the virgin reference operator. No state or RHS cache is
        retained, and a second owner cannot acquire another owner's factor.
        """
        with self._exclusive(_ENTRY):
            if self.store.has_active_trial or self.store.generation:
                raise ValueError("G3 reference RHS requires virgin committed state")
            rhs = owned(right_hand_sides)
            if rhs.ndim != 2 or rhs.shape[0] != self.size or not 1 <= rhs.shape[1] <= 8:
                raise ValueError("bounded reference RHS columns required")
            try:
                zero = np.zeros(self.size); shape = (len(self.elements), 3)
                g, J, _, _ = self._constraint_trial(zero, 0., None)
                if np.linalg.norm(g) > 1e-11: raise ValueError("reference constraints must be homogeneous")
                _, K, _ = self._evaluate(zero, zero, np.zeros(shape), np.zeros(shape))
                A = np.block([[K, J.T], [J, np.zeros((len(J), len(J)))]])
                result = solve(A, np.vstack([rhs, np.zeros((len(J), rhs.shape[1]))]))
                self._guard()
                return owned(result)
            finally:
                if self.store.has_active_trial: self.store.discard_trial(self.store.active_trial_token())

    def _constraint_trial(self, total, control, targets):
        r = self.store.native_rotation_store
        return self.constraints.evaluate(total, r.committed_full_displacement, r.committed_rotation_matrices,
                                         control=control, targets=targets)

    def _system(self, total, multipliers, nodal, lines, couples, control, targets):
        """Actual constrained Newton operator, also checked by directional tests."""
        g, J, H, rate = self._constraint_trial(total, control, targets)
        r, K, _ = self._evaluate(total, nodal, lines, couples)
        residual = np.r_[r+J.T @ multipliers, g]
        tangent = np.block([[K+np.einsum("i,ijk->jk", multipliers, H), J.T],
                            [J, np.zeros((len(g), len(g)))]])
        return residual, tangent, r, J, rate

    def solve(self, nodal, *, control=0., targets=None, lines=None, couples=None, cancel=lambda: False):
        with self._exclusive(_ENTRY):
            if self.store.has_active_trial: raise RuntimeError("pending external trial must be discarded")
            if len(self._accepted) >= 128: raise ValueError("G3 history bound")
            load = number(control); frames = self.constraints.target_frames(targets)
            nodal = owned(nodal, (self.size,))
            shape = (len(self.elements), 3)
            lines = owned(np.zeros(shape) if lines is None else lines, shape)
            couples = owned(np.zeros(shape) if couples is None else couples, shape)
            total = self.store.native_rotation_store.committed_full_displacement.copy()
            mu = np.zeros(len(self._constraint_trial(total, load, frames)[0]))
            started = monotonic()
            def check():
                if cancel(): raise InterruptedError("G3 cancelled; accepted prefix retained")
                self._guard()
                if monotonic()-started > 600: raise TimeoutError("G3 global deadline")
            def system(u, multipliers):
                check()
                return self._system(u, multipliers, nodal, lines, couples, load, frames)
            try:
                for iteration in range(25):
                    residual, tangent, r, J, rate = system(total, mu)
                    error = np.linalg.norm(residual)
                    if error <= 1e-11:
                        check(); token = self.store.active_trial_token()
                        pending = self.store.materialize(trial_token=token)
                        entry = dict(nodal=nodal.tolist(), lines=lines.tolist(), couples=couples.tolist(),
                                     control=load, targets=frames.tolist(), u=total.tolist(), multipliers=mu.tolist(),
                                     state_hash=sha(pending))
                        history = self._accepted+(entry,); journal = sha(history)
                        reactions = owned(-J.T @ mu); physical = reactions.copy()
                        delta = total-self.store.native_rotation_store.committed_full_displacement
                        for i in range(len(total)//6):
                            A, _ = exp_chart_terms(delta[6*i+3:6*i+6])
                            physical[6*i+3:6*i+6] = solve(A.T, reactions[6*i+3:6*i+6])
                        result = dict(u=owned(total), multipliers=owned(mu), reactions=reactions,
                                      spatial_reactions=owned(physical), control_work_rate=number(mu @ rate),
                                      residual_norm=number(error), iterations=iteration, history_length=len(history))
                        coordinates = np.array([n.coords() for n in self.model.mesh.nodes.values()])+total.reshape(-1, 6)[:, :3]
                        # Logging is fallible: finish it before atomic publication.
                        progress("acceptance prepare")
                        self._guard()
                        self.store.commit(token, accepted_full_displacement=total, accepted_full_coordinates=coordinates)
                        self._accepted = history; self._journal_digest = journal
                        return result
                    if iteration == 24: break
                    increment = solve(tangent, -residual)
                    for cut in range(9):
                        alpha = .5**cut; candidate = total+alpha*increment[:self.size]
                        cmu = mu+alpha*increment[self.size:]
                        if np.linalg.norm(system(candidate, cmu)[0]) < error:
                            total, mu = candidate, cmu; break
                    else: raise ValueError("G3 constrained line search failed")
                raise ValueError("G3 constrained iteration limit")
            finally:
                if self.store.has_active_trial: self.store.discard_trial(self.store.active_trial_token())

    def checkpoint(self):
        with self._exclusive(_ENTRY):
            if self.store.has_active_trial or runtime_digest() != self._runtime:
                raise ValueError("G3 checkpoint requires frozen committed state")
            if self._accepted and sha(self.store.materialize()) != self._accepted[-1]["state_hash"]:
                raise ValueError("G3 journal head mismatch")
            body = dict(schema=SCHEMA, runtime=self._runtime, definition=json.loads(self._definition),
                        constraints=self.constraints.descriptor(), graph=self.inventory, initial=self._initial_digest,
                        history=self._accepted, final=self._state_digest(), state=self.store.materialize())
            data = canonical(body)
            if len(data) > MAX_BYTES: raise ValueError("G3 checkpoint byte bound")
            progress("output")
            return data

    @classmethod
    def resume(cls, data, expected_sha256):
        if cls is not NativeGraphAnalysis: raise ValueError("exact G3 restart owner required")
        if type(data) is not bytes or len(data) > MAX_BYTES or sha256(data).hexdigest() != expected_sha256:
            raise ValueError("externally authenticated bounded G3 checkpoint required")
        def pairs(rows):
            result = {}
            for k, v in rows:
                if k in result: raise ValueError("duplicate checkpoint key")
                result[k] = v
            return result
        def bad(v): raise ValueError("nonfinite checkpoint")
        body = json.loads(data, object_pairs_hook=pairs, parse_constant=bad)
        if canonical(body) != data or set(body) != {"schema", "runtime", "definition", "constraints", "graph", "initial", "history", "final", "state"}:
            raise ValueError("G3 canonical checkpoint schema")
        if body["schema"] != SCHEMA or body["runtime"] != runtime_digest(): raise ValueError("G3 checkpoint family/runtime mismatch")
        definition = body["definition"]
        if type(definition) is not dict or set(definition) != {"elements", "fixed"} or definition["fixed"] != [] or type(body["history"]) is not list or len(body["history"]) > 128:
            raise ValueError("G3 definition/history schema")
        restart_graph_preflight(definition, body["graph"])
        constraints = GraphConstraintSet.from_descriptor(body["constraints"])
        elements = []
        from ._ge_beam3_g1_elastic import POLICY
        for d in definition["elements"]:
            if set(d) != {"element_id", "node_ids", "coordinates", "nodal_triads", "regularity_relative_tolerance", "rotation_tolerance", "frame_tolerance", "section", "order"}:
                raise ValueError("G3 element schema")
            s = d["section"]
            if set(s) != {"policy", "name", "stiffness"} or s["policy"] != POLICY: raise ValueError("G3 section identity")
            ref = Reference(d["coordinates"], d["nodal_triads"], regularity_relative_tolerance=d["regularity_relative_tolerance"],
                            rotation_tolerance=d["rotation_tolerance"], frame_tolerance=d["frame_tolerance"])
            elements.append(ElasticElement(d["element_id"], tuple(d["node_ids"]), ref, ElasticSection(s["stiffness"], s["name"]), order=d["order"]))
        if canonical(topology(tuple(elements), constraints)) != canonical(body["graph"]):
            raise ValueError("G3 graph policy/identity mismatch")
        made = cls(tuple(elements), constraints)
        progress("restart")
        if made._initial_digest != body["initial"]: raise ValueError("G3 initial identity mismatch")
        for entry in body["history"]:
            if set(entry) != {"nodal", "lines", "couples", "control", "targets", "u", "multipliers", "state_hash"}:
                raise ValueError("G3 accepted-history schema")
            made.solve(entry["nodal"], control=entry["control"], targets=entry["targets"], lines=entry["lines"], couples=entry["couples"])
            if canonical(made._accepted[-1]) != canonical(entry): raise ValueError("G3 accepted history replay mismatch")
        if made.checkpoint() != data: raise ValueError("G3 committed replay mismatch")
        return made
