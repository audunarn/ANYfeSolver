"""Private native elastic Element adapter using the shared material protocol."""
from copy import deepcopy
from fractions import Fraction
import numpy as np
from .elements import Element
from ._native_material_protocol import PROTOCOL, NativeMaterialValidator, NativeMaterialContext
from ._ge_beam3_generalized_static_boundary import chart_pullback
from ._ge_beam3_g1_elastic import owned, sha, canonical
from ._ge_beam3_g1_operator import ElasticOperator, local_solve

FORMULATION = "CANDIDATE_GE_BEAM3_G1_ELASTIC_STATIC_V1"
SCHEMA = "GE_BEAM3_G1_ELASTIC_STATE_V1"


class ElasticElement(Element):
    num_nodes = 3
    dofs_per_node = 6
    formulation_id = FORMULATION
    formulation_native_total_lagrangian = True
    native_material_state_protocol = PROTOCOL
    production_qualified = False

    def __init__(self, element_id, node_ids, reference, section, *, order=4):
        if type(element_id) is not int or element_id <= 0 or type(node_ids) is not tuple or len(node_ids) != 3:
            raise ValueError("explicit element and three node identities required")
        if len(set(node_ids)) != 3 or any(type(n) is not int or n <= 0 for n in node_ids):
            raise ValueError("positive distinct node IDs required")
        super().__init__(element_id, node_ids, "g1-elastic-"+str(element_id))
        self.operator = ElasticOperator(reference, section, order=order)
        self.section = section
        self.identity = sha(self.to_dict())
        self._mesh = None; self._validator = None; self._issued = None
        self._load = None
        self._owner_guard = None

    def to_dict(self):
        return dict(formulation_id=FORMULATION, element_id=self.element_id, node_ids=self.node_ids,
                    operator=self.operator.identity, section=self.section.identity,
                    production_qualified=False)

    def _check(self, mesh):
        if type(self) is not ElasticElement or mesh.elements.get(self.element_id) is not self:
            raise ValueError("exact elastic element ownership required")
        if self._mesh is None:
            self._mesh = mesh
        if self._mesh is not mesh or sha(self.to_dict()) != self.identity:
            raise ValueError("elastic model or definition changed")
        if (self.formulation_id != FORMULATION or self.production_qualified is not False
                or self.num_nodes != 3 or self.dofs_per_node != 6
                or self.section is not self.operator.section):
            raise ValueError("elastic formulation/section authority changed")
        self.operator.guard()
        if not np.array_equal(self.get_node_coordinates(mesh), self.operator.reference.coordinates):
            raise ValueError("reference coordinates changed")

    def native_reference_directors(self, mesh):
        self._check(mesh)
        return self.operator.reference.nodal_triads[:, :, 2].copy()

    def get_node_coordinates(self, mesh):
        return np.array([mesh.nodes[n].coords() for n in self.node_ids])

    def compute_stiffness_matrix(self, mesh, material):
        return self.init_model_bound_nonlinear_state(mesh, material, 1)["response"]["tangent"].copy()

    def _coordinates(self, total):
        total = owned(total, (18,)); X = self.operator.reference.coordinates
        x = X+total.reshape(3, 6)[:, :3]; low = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                low[i, j] = float(Fraction(float(X[i, j]))+Fraction(float(total[6*i+j]))-Fraction(float(x[i, j])))
        return x, low

    def _state(self, epoch, total, rotations, response, previous, line, couple):
        value = dict(schema=SCHEMA, element_identity=self.identity, epoch=epoch,
                     committed_total_u=owned(total, (18,)),
                     committed_nodal_rotation_matrices=owned(rotations, (3, 3, 3)),
                     response=response, history=(), previous_state_sha256=previous,
                     line=owned(line, (3,)), couple=owned(couple, (3,)))
        return {**value, "state_sha256": sha(value)}

    def init_model_bound_nonlinear_state(self, mesh, material, num_layers):
        self._arguments(mesh, material, num_layers)
        total = np.zeros(18); q = np.tile(np.eye(3), (3, 1, 1))
        x, low = self._coordinates(total)
        response = local_solve(self.operator, x, low, self.operator.reference.nodal_triads,
                               np.tile(np.eye(3), (2, 1, 1)), np.zeros(18), line=np.zeros(3), couple=np.zeros(3))
        return self._state(0, total, q, response, None, np.zeros(3), np.zeros(3))

    def _arguments(self, mesh, material, num_layers):
        self._check(mesh)
        if material is not self.section or type(num_layers) is not int or num_layers != 1:
            raise ValueError("elastic section/layer identity mismatch")

    def _validate(self, state, displacement):
        keys = {"schema", "element_identity", "epoch", "committed_total_u",
                "committed_nodal_rotation_matrices", "response", "history",
                "previous_state_sha256", "line", "couple", "state_sha256"}
        if type(state) is not dict or set(state) != keys:
            raise ValueError("elastic state schema")
        if state["schema"] != SCHEMA or state["element_identity"] != self.identity or state["history"] != ():
            raise ValueError("elastic state definition/history mismatch")
        if type(state["epoch"]) is not int or state["epoch"] < 0:
            raise ValueError("elastic state epoch")
        if sha({k: v for k, v in state.items() if k != "state_sha256"}) != state["state_sha256"]:
            raise ValueError("elastic state hash mismatch")
        if not np.array_equal(state["committed_total_u"], displacement):
            raise ValueError("elastic accepted displacement mismatch")
        x, low = self._coordinates(displacement); r = state["response"]
        value = self.operator.evaluate(x, low, state["committed_nodal_rotation_matrices"] @ self.operator.reference.nodal_triads,
                                       r["rotations"], r["resultants"], line=state["line"], couple=state["couple"])
        if canonical(value) != canonical(r["full"]) or r["history"] != () or r["internal_error"] > 1e-11:
            raise ValueError("elastic internal state/replay mismatch")

    def create_native_material_validator(self, mesh):
        self._check(mesh)
        self._issued = None
        def validate(state, *, previous_state, native_view, displacement, phase):
            self._validate(state, displacement)
            if phase == "committed":
                if previous_state is not None:
                    raise ValueError("unexpected previous committed state")
                matrices = native_view.committed_rotation_matrices
            elif phase == "trial":
                issued = self._issued
                if issued is None or issued[1] != state["state_sha256"]:
                    raise ValueError("elastic candidate was not issued")
                context = issued[0]; context.require_view(native_view)
                if context.store._native_element_bindings[self.element_id].material_validator is not validator:
                    raise ValueError("foreign elastic validator")
                if (state["previous_state_sha256"] != previous_state["state_sha256"]
                        or state["epoch"] != previous_state["epoch"]+1):
                    raise ValueError("elastic accepted-origin linkage")
                matrices = native_view.trial_rotation_matrices
            else:
                raise ValueError("unknown validation phase")
            if not np.array_equal(state["committed_nodal_rotation_matrices"], matrices):
                raise ValueError("elastic shared pose mismatch")
            # The store calls this again during commit preparation. Validate
            # owner authority after all material/pose validation, before any
            # shared-state pointer is published.
            if self._owner_guard is not None:
                self._owner_guard()
        validator = NativeMaterialValidator(validate)
        self._validator = validator
        return validator

    def validate_model_bound_nonlinear_state(self, mesh, material, state, num_layers, *, expected_committed_total_u=None):
        self._arguments(mesh, material, num_layers)
        self._validate(state, state["committed_total_u"] if expected_committed_total_u is None else expected_committed_total_u)
        return deepcopy(state)

    def compute_nonlinear_response(self, mesh, material, displacement, state=None, num_layers=1, tangent=True,
                                   *, native_rotation_trial=None, native_material_context=None):
        self._arguments(mesh, material, num_layers)
        context, view = native_material_context, native_rotation_trial
        if type(context) is not NativeMaterialContext or context.element_id != self.element_id:
            raise ValueError("issued elastic material context required")
        context.require_view(view)
        binding = context.store._native_element_bindings[self.element_id]
        if binding.material_validator is not self._validator or canonical(context.store[self.element_id]) != canonical(state):
            raise ValueError("foreign store or elastic origin")
        if self._load is None or self._load[0] is not context.store:
            raise ValueError("owned elastic load scope required")
        line, couple = self._load[1:]
        self._validate(state, state["committed_total_u"])
        total = owned(displacement, (18,)); x, low = self._coordinates(total)
        if not np.array_equal(x, view.trial_coordinates):
            raise ValueError("elastic trial coordinates mismatch")
        previous = state["response"]
        response = local_solve(self.operator, x, low, view.trial_rotation_matrices @ self.operator.reference.nodal_triads,
                               previous["rotations"], previous["resultants"], line=line, couple=couple,
                               check=lambda: context.require_view(view))
        force, matrix, _ = chart_pullback(response["residual"], response["tangent"], view.rotation_coordinate_increment)
        candidate = self._state(state["epoch"]+1, total, view.trial_rotation_matrices,
                                response, state["state_sha256"], line, couple)
        self._issued = (context, candidate["state_sha256"])
        return force.copy(), matrix.copy() if tangent else None, candidate

    def _unsupported(self, *args, **kwargs):
        raise ValueError("G1 elastic static-only private workflow")
    compute_mass_matrix = _unsupported
    compute_geometric_stiffness_matrix = _unsupported
    compute_stresses = _unsupported
    compute_internal_forces = _unsupported
