"""Private G2 constrained elastic owner. No public routing or G1 source edits."""
from hashlib import sha256
import json
import threading
from time import monotonic
import numpy as np
from .fe_core import FEModel
from .nonlinear_state import NonlinearStateStore, create_model_native_rotation_store
from ._ge_beam3_g1_analysis import ElasticAnalysis, runtime_digest, describe, MAX_BYTES
from ._ge_beam3_g1_elastic import owned, canonical, sha, solve, ElasticSection
from ._ge_beam3_g1_element import ElasticElement
from ._ge_beam3_g2_constraints import ConstraintSet, number
from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_p5.chart import exp_chart_terms

SCHEMA = "GE_BEAM3_G2_ELASTIC_RESTART_V1"


class ConstrainedAnalysis(ElasticAnalysis):
    """Shares G1 assembly/validation; all supports are explicit constraints."""
    def __init__(self, elements, constraints):
        if (type(elements) is not tuple or not 1 <= len(elements) <= 2 or
                any(type(e) is not ElasticElement for e in elements) or type(constraints) is not ConstraintSet):
            raise ValueError("G2 requires one/two exact elastic elements and captured constraints")
        if len({e.element_id for e in elements}) != len(elements): raise ValueError("unique elements required")
        self.elements = tuple(sorted(elements, key=lambda e: e.element_id)); self.model = FEModel("G2 constrained elastic")
        nodes = {}
        for e in self.elements:
            for n, point in zip(e.node_ids, e.operator.reference.coordinates):
                if n in nodes and not np.array_equal(nodes[n], point): raise ValueError("shared reference mismatch")
                nodes[n] = point
        definition = constraints.descriptor()
        if definition["node_ids"] != sorted(nodes) or not np.array_equal(definition["positions"], [nodes[n] for n in sorted(nodes)]):
            raise ValueError("constraint graph/reference mismatch")
        self.constraints = constraints; self._constraint_identity = constraints.identity
        self.size = 6*len(nodes)
        constraints.evaluate(np.zeros(self.size), np.zeros(self.size), np.tile(np.eye(3), (len(nodes), 1, 1)))
        for n in sorted(nodes): self.model.add_node(n, *nodes[n])
        for e in self.elements:
            self.model.add_element(e.element_id, e); self.model.materials[e.material_name] = e.section
        # No hidden fixed DOFs. The all-DOF immutable selection authenticates the
        # inherited assembler; every support is present in the multiplier system.
        self.fixed = (); self._free = np.frombuffer(np.arange(self.size, dtype=np.int64).tobytes(), dtype=np.int64)
        self._definition = canonical(dict(elements=[describe(e) for e in self.elements], fixed=[]))
        self._runtime = runtime_digest(); self._lock = threading.Lock()
        initial = {e.element_id: e.init_model_bound_nonlinear_state(self.model.mesh, e.section, 1) for e in self.elements}
        self.store = NonlinearStateStore.from_shell_layouts((), initial)
        self.store.attach_native_rotation_store(create_model_native_rotation_store(self.model, initial, np.zeros(self.size)))
        self._accepted = (); self._journal_digest = sha(self._accepted)
        self._initial_digest = self._state_digest(); self._graph = self._graph_identity()
        for e in self.elements: e._owner_guard = self._guard

    def _guard(self):
        if type(self) is not ConstrainedAnalysis or type(self.constraints) is not ConstraintSet or self.constraints.identity != self._constraint_identity:
            raise ValueError("G2 frozen constraint authority changed")
        super()._guard()

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
        with self._exclusive():
            if self.store.has_active_trial: raise RuntimeError("pending external trial must be discarded")
            if len(self._accepted) >= 128: raise ValueError("G2 history bound")
            load = number(control); frames = self.constraints.target_frames(targets)
            nodal = owned(nodal, (self.size,))
            shape = (len(self.elements), 3)
            lines = owned(np.zeros(shape) if lines is None else lines, shape)
            couples = owned(np.zeros(shape) if couples is None else couples, shape)
            total = self.store.native_rotation_store.committed_full_displacement.copy()
            mu = np.zeros(len(self._constraint_trial(total, load, frames)[0]))
            started = monotonic()
            def check():
                if cancel(): raise InterruptedError("G2 cancelled; accepted prefix retained")
                self._guard()
                if monotonic()-started > 600: raise TimeoutError("G2 global deadline")
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
                    else: raise ValueError("G2 constrained line search failed")
                raise ValueError("G2 constrained iteration limit")
            finally:
                if self.store.has_active_trial: self.store.discard_trial(self.store.active_trial_token())

    def checkpoint(self):
        with self._exclusive():
            if self.store.has_active_trial or runtime_digest() != self._runtime:
                raise ValueError("G2 checkpoint requires frozen committed state")
            if self._accepted and sha(self.store.materialize()) != self._accepted[-1]["state_hash"]:
                raise ValueError("G2 journal head mismatch")
            body = dict(schema=SCHEMA, runtime=self._runtime, definition=json.loads(self._definition),
                        constraints=self.constraints.descriptor(), initial=self._initial_digest,
                        history=self._accepted, final=self._state_digest(), state=self.store.materialize())
            data = canonical(body)
            if len(data) > MAX_BYTES: raise ValueError("G2 checkpoint byte bound")
            return data

    @classmethod
    def resume(cls, data, expected_sha256):
        if type(data) is not bytes or len(data) > MAX_BYTES or sha256(data).hexdigest() != expected_sha256:
            raise ValueError("externally authenticated bounded G2 checkpoint required")
        def pairs(rows):
            result = {}
            for k, v in rows:
                if k in result: raise ValueError("duplicate checkpoint key")
                result[k] = v
            return result
        def bad(v): raise ValueError("nonfinite checkpoint")
        body = json.loads(data, object_pairs_hook=pairs, parse_constant=bad)
        if canonical(body) != data or set(body) != {"schema", "runtime", "definition", "constraints", "initial", "history", "final", "state"}:
            raise ValueError("G2 canonical checkpoint schema")
        if body["schema"] != SCHEMA or body["runtime"] != runtime_digest(): raise ValueError("G2 checkpoint family/runtime mismatch")
        definition = body["definition"]
        if set(definition) != {"elements", "fixed"} or definition["fixed"] != [] or type(body["history"]) is not list or len(body["history"]) > 128:
            raise ValueError("G2 definition/history schema")
        elements = []
        from ._ge_beam3_g1_elastic import POLICY
        for d in definition["elements"]:
            if set(d) != {"element_id", "node_ids", "coordinates", "nodal_triads", "regularity_relative_tolerance", "rotation_tolerance", "frame_tolerance", "section", "order"}:
                raise ValueError("G2 element schema")
            s = d["section"]
            if set(s) != {"policy", "name", "stiffness"} or s["policy"] != POLICY: raise ValueError("G2 section identity")
            ref = Reference(d["coordinates"], d["nodal_triads"], regularity_relative_tolerance=d["regularity_relative_tolerance"],
                            rotation_tolerance=d["rotation_tolerance"], frame_tolerance=d["frame_tolerance"])
            elements.append(ElasticElement(d["element_id"], tuple(d["node_ids"]), ref, ElasticSection(s["stiffness"], s["name"]), order=d["order"]))
        made = cls(tuple(elements), ConstraintSet.from_descriptor(body["constraints"]))
        if made._initial_digest != body["initial"]: raise ValueError("G2 initial identity mismatch")
        for entry in body["history"]:
            if set(entry) != {"nodal", "lines", "couples", "control", "targets", "u", "multipliers", "state_hash"}:
                raise ValueError("G2 accepted-history schema")
            made.solve(entry["nodal"], control=entry["control"], targets=entry["targets"], lines=entry["lines"], couples=entry["couples"])
            if canonical(made._accepted[-1]) != canonical(entry): raise ValueError("G2 accepted history replay mismatch")
        if made.checkpoint() != data: raise ValueError("G2 committed replay mismatch")
        return made
