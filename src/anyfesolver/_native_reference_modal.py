"""Private reference-modal integration for retained inertial element coordinates.

Not a public selector or a qualification disposition. Element mechanics supply
energy factors; this module assembles the native DOF map and common constraints.
Only homogeneous nodal supports are admitted. No mass floor, spectral filtering,
or static elimination of an inertial coordinate is permitted.
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse

from .assembly import build_constraint_transformation
from .control import cancellation_safe_point


POLICY = "PRIVATE_RETAINED_INTERNAL_INERTIA_REFERENCE_MODAL_V1"


def _owned(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError("finite modal arrays required")
    return np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)


@dataclass(frozen=True)
class ReferenceModalBlock:
    element_id: int
    nodal_dofs: tuple
    algebraic_slots: tuple
    elastic_factor: np.ndarray
    kinetic_factor: np.ndarray
    identity: str

    def __post_init__(self):
        if type(self.element_id) is not int or self.element_id < 0:
            raise ValueError("exact element ID required")
        for slots in (self.nodal_dofs, self.algebraic_slots):
            if (type(slots) is not tuple or any(type(i) is not int or i < 0 for i in slots)
                    or len(set(slots)) != len(slots)):
                raise ValueError("distinct exact integer slots required")
        if (not self.nodal_dofs or any(i >= len(self.nodal_dofs) for i in self.algebraic_slots)
                or type(self.identity) is not str or len(self.identity) != 64
                or any(c not in "0123456789abcdef" for c in self.identity)):
            raise ValueError("valid nodal/algebraic layout and identity required")
        elastic, kinetic = _owned(self.elastic_factor), _owned(self.kinetic_factor)
        if (elastic.ndim != 2 or kinetic.ndim != 2 or not elastic.shape[0]
                or not kinetic.shape[0] or elastic.shape[1] != kinetic.shape[1]
                or not len(self.nodal_dofs) < elastic.shape[1] <= 256):
            raise ValueError("matching factors with retained internal coordinates required")
        if np.any(kinetic[:, self.algebraic_slots] != 0.):
            raise ValueError("declared algebraic coordinates must have exactly zero inertia")
        object.__setattr__(self, "elastic_factor", elastic)
        object.__setattr__(self, "kinetic_factor", kinetic)


@dataclass(frozen=True)
class ReferenceModalResult:
    eigenvalues: np.ndarray
    nodal_modes: np.ndarray
    internal_modes: tuple
    full_modes: np.ndarray
    full_stiffness: np.ndarray
    full_mass: np.ndarray
    dynamic_map: np.ndarray
    free_dofs: tuple
    algebraic_dofs: tuple
    block_identities: tuple
    normalized_residual: float
    policy: str = POLICY
    production_qualified: bool = False


def solve_reference_modal(model, blocks, *, check_inputs, num_modes=6,
                          cancellation_token=None):
    """Small native-model reference pencil; guards bracket all solve stages.

    ``check_inputs(stage, blocks)`` is mandatory and owned by the candidate adapter.
    It must reject changed model, mechanics, inertia and boundary identities.
    Factors are snapshotted; this function does not invoke element mechanics.
    The 256-coordinate development bound is not an engineering acceptance gate.
    """
    if not callable(check_inputs):
        raise ValueError("bound input guard required")

    def checkpoint(stage):
        cancellation_safe_point(cancellation_token, "native_reference_modal." + stage)
        check_inputs(stage, blocks)
        cancellation_safe_point(cancellation_token, "native_reference_modal." + stage)

    checkpoint("start")
    if type(num_modes) is not int or not 1 <= num_modes <= 256:
        raise ValueError("one through 256 requested modes required")
    if model.constraint_equations or model.mesh.element_activity is not None or model.mesh.point_masses:
        raise ValueError("private reference modal path excludes MPC, activity and point masses")
    if type(blocks) is not tuple or not blocks or any(type(b) is not ReferenceModalBlock for b in blocks):
        raise ValueError("exact immutable block tuple required")
    if tuple(b.element_id for b in blocks) != tuple(sorted(model.mesh.elements)):
        raise ValueError("complete ordered element coverage required")
    # Revalidate even a deliberately corrupted dataclass before use.
    blocks = tuple(ReferenceModalBlock(b.element_id, b.nodal_dofs, b.algebraic_slots,
                   b.elastic_factor, b.kinetic_factor, b.identity) for b in blocks)
    nodal = model.mesh.dof_manager.total_dofs
    total = nodal + sum(b.elastic_factor.shape[1] - len(b.nodal_dofs) for b in blocks)
    if not 1 <= nodal < total <= 256:
        raise ValueError("bounded native reference model required")
    elastic_rows, kinetic_rows, internal_layout = [], [], []
    declarations = {}
    offset = nodal
    for block in blocks:
        checkpoint("block")
        actual = tuple(int(i) for i in model.mesh.elements[block.element_id].get_dof_mapping(model.mesh))
        if actual != block.nodal_dofs or any(i >= nodal for i in actual):
            raise ValueError("element DOF ownership mismatch")
        for slot, dof in enumerate(actual):
            declared = slot in block.algebraic_slots
            if dof in declarations and declarations[dof] != declared:
                raise ValueError("inconsistent shared algebraic declaration")
            declarations[dof] = declared
        count = block.elastic_factor.shape[1] - len(actual)
        internal = tuple(range(offset, offset + count))
        internal_layout.append((block.element_id, internal))
        columns = actual + internal
        for source, target in ((block.elastic_factor, elastic_rows), (block.kinetic_factor, kinetic_rows)):
            rows = np.zeros((source.shape[0], total))
            rows[:, columns] = source
            target.append(rows)
        offset += count
    if set(declarations) != set(range(nodal)):
        raise ValueError("unowned nodal coordinates")
    elastic, kinetic = np.vstack(elastic_rows), np.vstack(kinetic_rows)
    stiffness, mass = elastic.T @ elastic, kinetic.T @ kinetic
    checkpoint("constraints")
    _, _, transform, origin, independent, info = build_constraint_transformation(
        sparse.csr_matrix(stiffness[:nodal, :nodal]), np.zeros(nodal), model)
    if info["num_mpc_constraints"] or np.any(origin != 0.):
        raise ValueError("only homogeneous nodal supports are admitted")
    expected = np.eye(nodal)[:, independent]
    if not np.array_equal(transform.toarray(), expected):
        raise ValueError("only nodal selector constraints are admitted")
    free = tuple(int(i) for i in independent) + tuple(range(nodal, total))
    algebraic = tuple(i for i in free if declarations.get(i, False))
    physical = tuple(i for i in free if i not in algebraic)
    if np.any(kinetic[:, algebraic] != 0.):
        raise ValueError("assembled algebraic inertia must be exactly absent")
    mapping = np.zeros((total, len(physical)))
    mapping[list(physical)] = np.eye(len(physical))
    if algebraic:
        # Global least-squares trace equilibrium, not element-local trace or
        # cell-spin condensation. Retain energy factors to avoid Schur subtraction.
        trace_factor = elastic[:, algebraic]
        orthogonal, triangular = np.linalg.qr(trace_factor, mode="reduced")
        np.linalg.cholesky(triangular.T @ triangular)
        mapping[list(algebraic)] = -np.linalg.solve(triangular, orthogonal.T @ elastic[:, physical])
    checkpoint("algebraic_equilibrium")
    reduced_elastic, reduced_kinetic = elastic @ mapping, kinetic @ mapping
    k, m = reduced_elastic.T @ reduced_elastic, reduced_kinetic.T @ reduced_kinetic
    if num_modes > len(physical):
        raise ValueError("requested modes exceed retained dynamic dimension")
    np.linalg.cholesky(m)  # Undeclared mass kernels fail, never receive fake inertia.
    values, vectors = linalg.eigh(k, m, subset_by_index=(0, num_modes - 1))
    modes = mapping @ vectors
    for column in range(num_modes):
        pivot = int(np.argmax(np.abs(modes[:, column])))
        if modes[pivot, column] < 0.: modes[:, column] *= -1.
    residual = (stiffness @ modes - (mass @ modes) * values)[list(free)]
    scale = max(1., np.linalg.norm(stiffness) * np.linalg.norm(modes),
                np.linalg.norm(mass) * np.linalg.norm(modes * values))
    error = float(np.linalg.norm(residual) / scale)
    if not np.isfinite(error) or error > 1e-11:
        raise ValueError("full-pencil eigenpair residual failed")
    if np.linalg.norm(modes.T @ mass @ modes - np.eye(num_modes)) > 1e-11:
        raise ValueError("physical modal mass normalization failed")
    checkpoint("output")
    return ReferenceModalResult(_owned(values), _owned(modes[:nodal]),
        tuple((eid, _owned(modes[list(indices)])) for eid, indices in internal_layout),
        _owned(modes), _owned(stiffness), _owned(mass), _owned(mapping), free, algebraic,
        tuple((b.element_id, b.identity) for b in blocks), error)
