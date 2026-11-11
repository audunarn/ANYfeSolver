"""Private M_B2/M_B3 reference-linear slices; no public graph admission.

One native element plus one scalar legacy beam. This reference problem has
no nonlinear pose/history owner and does not implement restart or shell
families. The native full stationary operator is evaluated only at reference.
"""
from dataclasses import asdict
from fractions import Fraction
import json
import os
import threading
import numpy as np
from .elements import BeamElement, QuadraticBeamElement
from .fe_core import FEModel, Material
from ._ge_beam3_g1_element import ElasticElement
from ._ge_beam3_g1_elastic import canonical, owned, sha, solve
from ._ge_beam3_g1_operator import schur

POLICY = "G3B_M_B2_REFERENCE_LINEAR_TRANSLATIONS_ONLY_DEVELOPMENT_V1"
B3_POLICY = "G3B_M_B3_REFERENCE_LINEAR_TRANSLATIONS_ONLY_DEVELOPMENT_V1"
_DEFAULT_POLICY = object()
SECTION_KEYS = {"area", "Iy", "Iz", "J", "shear_factor_y", "shear_factor_z", "orientation"}


def _family(problem):
    if type(problem) is B2TranslationReferenceProblem: return POLICY, BeamElement, 2
    if type(problem) is B3TranslationReferenceProblem: return B3_POLICY, QuadraticBeamElement, 3
    raise ValueError("exact closed G3b beam reference problem required")


def progress(stage):
    if os.environ.get("G3_PROGRESS") == "1":
        print("G3B CHECKPOINT "+stage, flush=True)


def invariant(a, b):
    error = np.linalg.norm(a-b)/max(1., np.linalg.norm(a), np.linalg.norm(b))
    if not np.isfinite(error) or error > 1e-11:
        raise ValueError("G3b reference identity failed")


class _BeamTranslationReferenceProblem:
    """Captured linear operators, disjoint IDs and a force-only translation tie.

    solve() is stateless: it never commits native rotations or material state.
    Returned small-rotation coordinates are not finite-rotation history.
    """
    def __init__(self, native, legacy, material, *, nodes, fixed_nodes, tie,
                 policy=_DEFAULT_POLICY, adapter_allowlist=(), rotation_targets=None):
        expected_policy, legacy_type, count = _family(self)
        if policy is _DEFAULT_POLICY: policy = expected_policy
        if (policy != expected_policy or
                type(adapter_allowlist) is not tuple or adapter_allowlist or rotation_targets is not None):
            raise ValueError("G3b translation-only reference policy required")
        if type(native) is not ElasticElement or type(legacy) is not legacy_type or type(material) is not Material:
            raise ValueError("G3b requires exact native and registered scalar legacy beam/material types")
        if native._mesh is not None or native._owner_guard is not None:
            raise ValueError("fresh unowned native element required")
        if (native.element_id == legacy.element_id or type(legacy.element_id) is not int or legacy.element_id <= 0 or
                len(legacy.node_ids) != count or any(type(n) is not int or n <= 0 for n in legacy.node_ids) or
                len(set(legacy.node_ids)) != count or set(native.node_ids) & set(legacy.node_ids)):
            raise ValueError("disjoint positive element/node identities required; no shared rotations")
        if type(nodes) is not dict or len(nodes) != 3+count or set(nodes) != set(native.node_ids) | set(legacy.node_ids):
            raise ValueError("exact registered mixed-beam node inventory required")
        if any(type(n) is not int or n <= 0 for n in nodes): raise ValueError("positive exact node IDs required")
        ids = tuple(sorted(nodes)); positions = owned([nodes[n] for n in ids], (3+count, 3))
        if count == 3:
            bx = positions[[ids.index(n) for n in legacy.node_ids]]
            if not np.array_equal(bx[1], (bx[0]+bx[2])/2):
                raise ValueError("M_B3 requires a straight exact midpoint")
            if not np.array_equal(owned(legacy.eccentricity, (3,)), np.zeros(3)):
                raise ValueError("M_B3 eccentricity is not admitted")
        if not np.array_equal(positions[[ids.index(n) for n in native.node_ids]], native.operator.reference.coordinates):
            raise ValueError("native reference coordinate mismatch")
        x = native.operator.reference.coordinates
        if not np.array_equal(x[1], (x[0]+x[2])/2): raise ValueError("straight exact midpoint required")
        if (type(fixed_nodes) is not tuple or len(fixed_nodes) != 2 or
                any(type(n) is not int for n in fixed_nodes) or
                set(fixed_nodes) != {native.node_ids[0], legacy.node_ids[-1]}):
            raise ValueError("independently fixed native and legacy roots required")
        expected = dict(components=[0, 1, 2], masters=[[legacy.node_ids[0], "1"]],
                        offset=[0, 0, 0], slave=native.node_ids[-1])
        if type(tie) is not dict or canonical(tie) != canonical(expected):
            raise ValueError("exact translation-only unit tie required")
        if not np.array_equal(positions[ids.index(native.node_ids[-1])], positions[ids.index(legacy.node_ids[0])]):
            raise ValueError("trace positions must coincide; no hidden offset arm")
        if set(legacy.cross_section) != SECTION_KEYS or legacy.generalized_section is not None or legacy._fiber_plasticity is not None:
            raise ValueError("explicit scalar elastic beam section only")
        if material.hardening_curve is not None: raise ValueError("reference-elastic material only")
        scalars = owned([legacy.cross_section[k] for k in sorted(SECTION_KEYS-{"orientation"})])
        if np.any(scalars <= 0): raise ValueError("positive scalar section required")
        owned(legacy.cross_section["orientation"], (3,))
        self.native, self.legacy, self.material = native, legacy, material
        self.ids, self.positions, self.size = ids, positions, 6*(3+count)
        self.fixed_nodes = tuple(sorted(fixed_nodes)); self._tie = canonical(expected)
        self._lock = threading.Lock()
        self.model = FEModel("G3b mixed-beam reference development")
        for n, point in zip(ids, positions): self.model.add_node(n, *point)
        for element in sorted((native, legacy), key=lambda e: e.element_id):
            self.model.add_element(element.element_id, element)
        self._dofs = tuple(tuple(6*ids.index(n)+j for n in e.node_ids for j in range(6)) for e in (native, legacy))
        # Exact rational elimination: independent support rows, then the tie.
        fixed = tuple(6*ids.index(n)+j for n in self.fixed_nodes for j in range(6))
        slave = tuple(6*ids.index(native.node_ids[-1])+j for j in range(3))
        master = tuple(6*ids.index(legacy.node_ids[0])+j for j in range(3))
        free = tuple(i for i in range(self.size) if i not in fixed+slave)
        rows = [[Fraction(0) for _ in free] for _ in range(self.size)]
        for j, i in enumerate(free): rows[i][j] = Fraction(1)
        for a, b in zip(slave, master): rows[a] = list(rows[b])
        T = np.array(rows, dtype=float); J = np.zeros((15, self.size))
        for i, dof in enumerate(fixed): J[i, dof] = 1
        for i, (a, b) in enumerate(zip(slave, master)): J[12+i, a], J[12+i, b] = 1, -1
        assert np.array_equal(J @ T, np.zeros((15, len(free))))
        self.T, self.J = owned(T), owned(J)
        self._definition = self._describe()
        progress("capture")
        native.operator.guard()
        full = native.operator.evaluate(x, np.zeros((3, 3)), native.operator.reference.nodal_triads,
                                        np.tile(np.eye(3), (2, 1, 1)), np.zeros(18))
        invariant(full["residual"], np.zeros(42))
        _, nk, lift, _ = schur(full["residual"], full["jacobian"])
        self.native_full, self.native_lift = owned(full["jacobian"]), owned(lift)
        progress("local solve")
        # Always use the family's actual elastic K, never its zero-force stub.
        bk = legacy.compute_stiffness_matrix(self.model.mesh, material)
        K = np.zeros((self.size, self.size))
        for dofs, block in zip(self._dofs, (nk, bk)): K[np.ix_(dofs, dofs)] += block
        invariant(K, K.T)
        self.K = owned(K); self.reduced = owned(T.T @ K @ T)
        np.linalg.cholesky(self.reduced)
        self._operators = sha(self._operator_data())
        self._guard()
        progress("assembly")

    def _describe(self):
        return canonical(dict(policy=_family(self)[0], native=self.native.to_dict(), legacy=self.legacy.to_dict(),
            eccentricity=self.legacy.eccentricity if _family(self)[2] == 3 else None,
            cross_section=self.legacy.cross_section, material=asdict(self.material),
            scalar_capture=[self.legacy._A, self.legacy._Iy, self.legacy._Iz, self.legacy._J,
                            self.legacy._ky, self.legacy._kz], orientation=self.legacy._orientation,
            nodes=[(n, self.model.mesh.nodes[n].coords(), list(self.model.mesh.dof_manager.get_node_dofs(n))) for n in self.ids],
            declared_positions=self.positions, size=self.size, fixed=self.fixed_nodes,
            tie=json.loads(self._tie), dofs=self._dofs, T=self.T, J=self.J))

    def _operator_data(self):
        return dict(K=self.K, reduced=self.reduced, full=self.native_full, lift=self.native_lift)

    def _guard(self):
        _, legacy_type, count = _family(self)
        if (self.model.constraint_equations or self.model.boundary_conditions or self.model.load_cases or
                self.model.mesh.dof_manager._constrained_dofs or self.model.mesh.point_masses or
                self.model.mesh.element_activity is not None or self.size != 6*(3+count) or
                self.model.mesh.dof_manager.total_dofs != self.size):
            raise ValueError("G3b unsupported reference graph feature")
        if (type(self.native) is not ElasticElement or type(self.legacy) is not legacy_type or
                type(self.material) is not Material or self.native._mesh is not None or
                self.native._owner_guard is not None or self.legacy.generalized_section is not None or
                self.legacy._fiber_plasticity is not None or self.material.hardening_curve is not None):
            raise ValueError("G3b reference-only ownership changed")
        if (tuple(self.model.mesh.nodes) != self.ids or
                self.model.mesh.elements.get(self.native.element_id) is not self.native or
                self.model.mesh.elements.get(self.legacy.element_id) is not self.legacy or
                len(self.model.mesh.elements) != 2 or self._describe() != self._definition or
                sha(self._operator_data()) != self._operators):
            raise ValueError("G3b frozen reference graph/operator changed")
        self.native.operator.guard()

    def solve(self, nodal, *, mode="REFERENCE_LINEAR", rotation_targets=None, cancel=lambda: False):
        policy, _, _ = _family(self)
        if mode != "REFERENCE_LINEAR" or rotation_targets is not None:
            raise ValueError("finite mixed programs and rotational coupling remain unsupported")
        if not self._lock.acquire(blocking=False): raise RuntimeError("G3b reference problem already in use")
        try:
            self._guard(); f = owned(nodal, (self.size,))
            if cancel(): raise InterruptedError("G3b reference solve cancelled")
            self._guard()
            u = self.T @ solve(self.reduced, self.T.T @ f, assume_a="pos")
            balance = self.K @ u-f
            mu = solve(self.J @ self.J.T, -self.J @ balance, assume_a="pos")
            invariant(balance+self.J.T @ mu, np.zeros(self.size)); invariant(self.J @ u, np.zeros(15))
            internal = self.native_lift @ u[list(self._dofs[0])]
            invariant(self.native_full[18:, :18] @ u[list(self._dofs[0])]+self.native_full[18:, 18:] @ internal, np.zeros(24))
            result = dict(policy=policy, qualification=False, node_ids=self.ids, u=owned(u),
                multipliers=owned(mu), support_reactions=owned(-self.J[:12].T @ mu[:12]),
                tie_forces=owned(-self.J[12:].T @ mu[12:]), internal=owned(internal),
                residual=owned(balance+self.J.T @ mu), energy=float(u @ self.K @ u/2),
                native_stations=self.native.operator.cell.recover(internal[6:]),
                legacy_recovery=self.legacy.compute_stresses(self.model.mesh, u[list(self._dofs[1])], self.material),
                native_station_policy="REFERENCE_LINEAR_STATIONARY_RECOVERY_NOT_FINITE_FRAMES")
            progress("recovery")
            if cancel(): raise InterruptedError("G3b reference solve cancelled before return")
            self._guard(); progress("output")
            return result
        finally:
            self._lock.release()

    def checkpoint(self):
        raise ValueError("G3b beam smoke has no restart/state authority; full G3b owner is a later step")


class B2TranslationReferenceProblem(_BeamTranslationReferenceProblem):
    """Exact legacy B2 scalar-section development fixture."""


class B3TranslationReferenceProblem(_BeamTranslationReferenceProblem):
    """Exact legacy straight-midpoint B3 scalar-section development fixture."""
