"""Private bounded G1 owner using the real assembler and native state store."""
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
from time import monotonic
import uuid
import numpy as np
from .fe_core import FEModel
from .nonlinear_static import _assemble_nonlinear_system
from .nonlinear_state import NonlinearStateStore, create_model_native_rotation_store
from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_p5.chart import exp_chart_terms
from ._ge_beam3_g1_elastic import ElasticSection, owned, canonical, sha, solve
from ._ge_beam3_g1_element import ElasticElement

SCHEMA = "GE_BEAM3_G1_ELASTIC_RESTART_V1"
MAX_BYTES = 2*1024*1024


def runtime_digest():
    root = Path(__file__).parent
    # Include all runtime Python files, not only the new adapter. LF accounts
    # for Git checkout conversion; wheel content retains the same text binding.
    return sha([(p.relative_to(root).as_posix(), sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest())
                for p in sorted(root.rglob("*.py"))])


def describe(element):
    r = element.operator.reference
    return dict(element_id=element.element_id, node_ids=list(element.node_ids),
                coordinates=r.coordinates, nodal_triads=r.nodal_triads,
                regularity_relative_tolerance=r._regularity_relative_tolerance,
                rotation_tolerance=r._rotation_tolerance, frame_tolerance=r._frame_tolerance,
                section=element.section.descriptor(), order=element.operator.order)


class ElasticAnalysis:
    def __init__(self, elements, fixed_dofs):
        if type(elements) is not tuple or not 1 <= len(elements) <= 2 or any(type(e) is not ElasticElement for e in elements):
            raise ValueError("G1 requires one or two exact elastic elements")
        if len({e.element_id for e in elements}) != len(elements):
            raise ValueError("unique elements required")
        self.model = FEModel("G1 elastic integration")
        self.elements = tuple(sorted(elements, key=lambda e: e.element_id))
        nodes = {}
        for e in self.elements:
            for n, point in zip(e.node_ids, e.operator.reference.coordinates):
                if n in nodes and not np.array_equal(nodes[n], point):
                    raise ValueError("shared node reference mismatch")
                nodes[n] = point
        for n in sorted(nodes):
            self.model.add_node(n, *nodes[n])
        for e in self.elements:
            self.model.add_element(e.element_id, e)
            self.model.materials[e.material_name] = e.section
        self.size = self.model.mesh.dof_manager.total_dofs
        if type(fixed_dofs) is not tuple or any(type(d) is not int or not 0 <= d < self.size for d in fixed_dofs):
            raise ValueError("explicit zero support DOFs required")
        if len(set(fixed_dofs)) != len(fixed_dofs) or not fixed_dofs:
            raise ValueError("unique supported DOFs required")
        self.fixed = tuple(sorted(fixed_dofs))
        for n in self.model.mesh.nodes:
            if len(set(self.model.mesh.dof_manager.get_node_dofs(n)[3:]) & set(self.fixed)) not in (0, 3):
                raise ValueError("G1 requires complete rotational support triples")
        # Bytes-backed indexing has no writable owner or writable alias. Public
        # access is read-only; the guard also authenticates private replacement.
        self._free = np.frombuffer(np.array([d for d in range(self.size)
                                            if d not in self.fixed], dtype=np.int64).tobytes(), dtype=np.int64)
        self._lock = threading.Lock()
        self._definition = canonical(dict(elements=[describe(e) for e in self.elements], fixed=list(self.fixed)))
        self._runtime = runtime_digest()
        initial = {e.element_id: e.init_model_bound_nonlinear_state(self.model.mesh, e.section, 1) for e in self.elements}
        self.store = NonlinearStateStore.from_shell_layouts((), initial)
        self.store.attach_native_rotation_store(create_model_native_rotation_store(self.model, initial, np.zeros(self.size)))
        self._accepted = ()
        self._journal_digest = sha(self._accepted)
        self._initial_digest = self._state_digest()
        self._graph = self._graph_identity()
        for e in self.elements:
            e._owner_guard = self._guard

    @property
    def free(self):
        return self._free

    def _graph_identity(self):
        return sha(dict(definition=json.loads(self._definition),
                        nodes=[(n, node.coords(), list(self.model.mesh.dof_manager.get_node_dofs(n)))
                               for n, node in sorted(self.model.mesh.nodes.items())],
                        elements=[(i, e.to_dict()) for i, e in sorted(self.model.mesh.elements.items())],
                        fixed=self.fixed, free=self.free, size=self.size))

    def _guard(self):
        expected = tuple(d for d in range(self.model.mesh.dof_manager.total_dofs)
                         if d not in json.loads(self._definition)["fixed"])
        if (self.size != self.model.mesh.dof_manager.total_dofs or
                type(self._free) is not np.ndarray or self._free.dtype != np.dtype(np.int64) or
                self._free.shape != (len(expected),) or self._free.flags.writeable or
                not isinstance(self._free.base, bytes) or tuple(self._free) != expected):
            raise ValueError("G1 frozen free-DOF authority changed")
        if sha(self._accepted) != self._journal_digest or len(self._accepted) != self.store.generation:
            raise ValueError("accepted journal/owner generation changed")
        if self._graph_identity() != self._graph or canonical(dict(elements=[describe(e) for e in self.elements], fixed=list(self.fixed))) != self._definition:
            raise ValueError("G1 frozen graph changed")
        if (self.model.constraint_equations or self.model.boundary_conditions or self.model.load_cases or
                self.model.mesh.dof_manager._constrained_dofs or self.model.mesh.point_masses or
                self.model.mesh.element_activity is not None):
            raise ValueError("G1 unsupported graph feature")
        for e in self.elements:
            if e._owner_guard != self._guard:
                raise ValueError("G1 owner preparation guard changed")
            e._check(self.model.mesh)
            if self.model.materials.get(e.material_name) is not e.section:
                raise ValueError("G1 section binding changed")

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("G1 owner already in use")
        try:
            self._guard()
            yield
        finally:
            self._lock.release()

    def _state_digest(self):
        r = self.store.native_rotation_store
        return sha(dict(states=self.store.materialize(), u=r.committed_full_displacement,
                        Q=r.committed_rotation_matrices, epoch=self.store.generation))

    def _evaluate(self, total, nodal, lines, couples):
        self._guard()
        total = owned(total, (self.size,))
        if np.any(total[list(self.fixed)]):
            raise ValueError("nonzero fixed displacement not admitted in G1")
        for e, line, couple in zip(self.elements, lines, couples):
            e._load = (self.store, line, couple)
        try:
            force, K, trial = _assemble_nonlinear_system(self.model, total, self.store, 1)
        finally:
            for e in self.elements:
                e._load = None
        matrix = K.toarray()
        delta = (total-self.store.native_rotation_store.committed_full_displacement).reshape(-1, 6)
        force = force-nodal
        for n, row in enumerate(delta):
            start = 6*n+3; A, dA = exp_chart_terms(row[3:])
            moment = nodal[start:start+3]
            force[start:start+3] += moment-A.T @ moment
            for k in range(3):
                matrix[start:start+3, start+k] -= dA[:, :, k].T @ moment
        return force, matrix, trial

    def solve(self, nodal, *, lines=None, couples=None, cancel=lambda: False):
        # This entry point owns only G1's fixed-support equations and journal.
        # A subclass must not bypass its own constraints via explicit base dispatch.
        if type(self) is not ElasticAnalysis:
            raise ValueError("G1 solve requires an exact G1 owner")
        with self._exclusive():
            if self.store.has_active_trial:
                raise RuntimeError("pending external trial must be discarded")
            if len(self._accepted) >= 128:
                raise ValueError("G1 accepted-history limit")
            nodal = owned(nodal, (self.size,))
            lines = owned(np.zeros((len(self.elements), 3)) if lines is None else lines, (len(self.elements), 3))
            couples = owned(np.zeros((len(self.elements), 3)) if couples is None else couples, (len(self.elements), 3))
            total = self.store.native_rotation_store.committed_full_displacement.copy()
            started = monotonic()
            def check():
                if cancel():
                    raise InterruptedError("G1 cancelled; accepted prefix retained")
                # A callback is external code even when it returns False.
                self._guard()
                if monotonic()-started > 600:
                    raise TimeoutError("G1 global solve deadline")
            try:
                for iteration in range(25):
                    check()
                    force, K, _ = self._evaluate(total, nodal, lines, couples)
                    error = np.linalg.norm(force[self.free])
                    if error <= 1e-11:
                        check()
                        token = self.store.active_trial_token()
                        coordinates = np.array([n.coords() for n in self.model.mesh.nodes.values()])+total.reshape(-1, 6)[:, :3]
                        # Preallocate the complete next journal before store publication.
                        pending = self.store.materialize(trial_token=token)
                        Q = self.store.native_element_rotation_view(token, self.elements[0].element_id,
                                                                   self.elements[0].node_ids,
                                                                   self.elements[0].native_reference_directors(self.model.mesh))
                        del Q  # Require a live issued view before publication.
                        next_entry = dict(nodal=nodal.tolist(), lines=lines.tolist(), couples=couples.tolist(),
                                          state_hash=sha(pending), u=total.tolist())
                        next_history = self._accepted+(next_entry,)
                        next_digest = sha(next_history)
                        # Revalidate after preparation and before publication.
                        self._guard()
                        self.store.commit(token, accepted_full_displacement=total, accepted_full_coordinates=coordinates)
                        self._accepted = next_history
                        self._journal_digest = next_digest
                        return dict(u=owned(total), reactions=owned(force), iterations=iteration,
                                    state_sha256=self._state_digest(), history_length=len(self._accepted))
                    if iteration == 24:
                        break
                    step = solve(K[np.ix_(self.free, self.free)], -force[self.free], assume_a="gen")
                    for cut in range(9):
                        check()
                        candidate = total.copy(); candidate[self.free] += .5**cut*step
                        changed, _, _ = self._evaluate(candidate, nodal, lines, couples)
                        if np.linalg.norm(changed[self.free]) < error:
                            total = candidate
                            break
                    else:
                        raise ValueError("G1 global line search failed")
                raise ValueError("G1 global iteration limit")
            finally:
                if self.store.has_active_trial:
                    self.store.discard_trial(self.store.active_trial_token())

    def recover(self):
        with self._exclusive():
            if self.store.has_active_trial:
                raise RuntimeError("recovery requires committed state")
            return {e.element_id: e.operator.recover(self.store[e.element_id]["response"]["rotations"],
                                                    self.store[e.element_id]["response"]["resultants"])
                    for e in self.elements}

    def checkpoint(self):
        with self._exclusive():
            if self.store.has_active_trial or runtime_digest() != self._runtime:
                raise ValueError("checkpoint requires frozen runtime and committed state")
            if self._accepted and sha(self.store.materialize()) != self._accepted[-1]["state_hash"]:
                raise ValueError("committed state is not the accepted journal head")
            data = canonical(dict(schema=SCHEMA, runtime=self._runtime, definition=json.loads(self._definition),
                                  initial=self._initial_digest, history=self._accepted,
                                  final=self._state_digest(), state=self.store.materialize()))
            if len(data) > MAX_BYTES:
                raise ValueError("G1 checkpoint byte limit")
            return data

    @classmethod
    def resume(cls, data, expected_sha256):
        if type(data) is not bytes or len(data) > MAX_BYTES or sha256(data).hexdigest() != expected_sha256:
            raise ValueError("externally authenticated bounded checkpoint required")
        def pairs(rows):
            made = {}
            for k, v in rows:
                if k in made:
                    raise ValueError("duplicate checkpoint key")
                made[k] = v
            return made
        def bad(value):
            raise ValueError("nonfinite checkpoint")
        body = json.loads(data, object_pairs_hook=pairs, parse_constant=bad)
        if canonical(body) != data or set(body) != {"schema", "runtime", "definition", "initial", "history", "final", "state"}:
            raise ValueError("canonical exact checkpoint schema required")
        if body["schema"] != SCHEMA or body["runtime"] != runtime_digest():
            raise ValueError("checkpoint family/runtime mismatch")
        definition = body["definition"]
        if type(definition) is not dict or set(definition) != {"elements", "fixed"} or type(body["history"]) is not list:
            raise ValueError("checkpoint definition/history schema")
        if len(body["history"]) > 128:
            raise ValueError("bounded accepted history required")
        elements = []
        for d in definition["elements"]:
            if set(d) != {"element_id", "node_ids", "coordinates", "nodal_triads", "regularity_relative_tolerance",
                          "rotation_tolerance", "frame_tolerance", "section", "order"}:
                raise ValueError("elastic definition schema")
            section = d["section"]
            from ._ge_beam3_g1_elastic import POLICY
            if set(section) != {"policy", "name", "stiffness"} or section["policy"] != POLICY:
                raise ValueError("elastic section family mismatch")
            ref = Reference(d["coordinates"], d["nodal_triads"],
                            regularity_relative_tolerance=d["regularity_relative_tolerance"],
                            rotation_tolerance=d["rotation_tolerance"], frame_tolerance=d["frame_tolerance"])
            elements.append(ElasticElement(d["element_id"], tuple(d["node_ids"]), ref,
                                           ElasticSection(section["stiffness"], section["name"]), order=d["order"]))
        made = cls(tuple(elements), tuple(definition["fixed"]))
        if made._initial_digest != body["initial"]:
            raise ValueError("virgin checkpoint identity mismatch")
        for entry in body["history"]:
            if type(entry) is not dict or set(entry) != {"nodal", "lines", "couples", "state_hash", "u"}:
                raise ValueError("accepted history schema")
            made.solve(entry["nodal"], lines=entry["lines"], couples=entry["couples"])
            if canonical(made._accepted[-1]) != canonical(entry):
                raise ValueError("accepted history replay mismatch")
        if made.checkpoint() != data:
            raise ValueError("committed state replay mismatch")
        return made

    def publish(self, output):
        data = self.checkpoint(); target = Path(output)
        if target.exists() or target.is_symlink() or not target.parent.is_dir():
            raise ValueError("exclusive output in existing directory required")
        stage = target.parent / ("."+target.name+"."+uuid.uuid4().hex+".pending")
        try:
            with stage.open("xb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            if stage.read_bytes() != data:
                raise ValueError("staged checkpoint mismatch")
            os.link(stage, target)  # Atomic no-overwrite publication on the same volume.
        finally:
            if stage.exists():
                stage.unlink()
        return sha256(data).hexdigest()
