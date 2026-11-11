"""
Assembly and Solver Module

This module provides functions for:
- assembling global stiffness and load vectors,
- applying constraints,
- solving linear systems,
- post-processing reactions, stresses and displacements.

Constraint handling
-------------------
The linear solver uses an explicit transformation/reduction of the global
system. Fixed boundary conditions and beam-shell MPC/eccentricity constraints
are eliminated before solving:

    u = T q + u0
    T.T K T q = T.T (F - K u0)

For unsupported free-free models, the solver can additionally suppress the six
rigid body modes by solving an augmented nullspace system:

    [K_red  Q] [q]      [F_red]
    [Q.T    0] [lambda] [0    ]

where Q contains orthonormal reduced rigid-body modes. This gives a unique gauge
for the displacement field without introducing artificial support stiffness. If
the load vector has a rigid-body component, the solve still returns the gauged
solution and reports the imbalance through solver diagnostics.
"""

from __future__ import annotations

import json
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab, gmres, minres

from .cases import make_result_case
from .constraint_audit import constraint_residual_summary, require_valid_constraints
from .control import CancellationToken, ProgressCallback, cancellation_safe_point, emit_progress
from .element_capabilities import require_model_element_capabilities
from .linalg import FactorizationCache, MatrixClass, factorize, factorize_cached
from .matrix_assembly import (
    assemble_load_matrix,
    assemble_mass_matrix as _canonical_assemble_mass_matrix,
    assemble_system as _canonical_assemble_system,
)
from .nonlinear_state import require_no_native_total_lagrangian_elements
from .recovery import ResourceConfig
from .recovery_batches import (
    _qualified_recovery_observation_guard,
    _run_with_qualified_recovery_runtime_lease,
)
from .threading_policy import resource_threaded, thread_policy_diagnostics

if TYPE_CHECKING:
    from .analysis_session import AnalysisSession
    from .boundary import LoadCase
    from .fe_core import FEModel, FEMesh


_SYMMETRIC_DIAGONAL_EQUILIBRATION_ID = (
    "SYMMETRIC_DIAGONAL_CONGRUENCE_Q1C_V1"
)
_SYMMETRIC_DIAGONAL_FLOOR_RATIO = 1.0e-30
_LINEAR_STATIC_BACKWARD_ERROR_LIMIT = 1.0e-10


def assemble_system(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    include_mass: bool = False,
) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[str, Any]]:
    """Compatibility wrapper around :mod:`anyfesolver.matrix_assembly`."""
    return _canonical_assemble_system(model, load_case, include_mass)


def assemble_mass_matrix(model: "FEModel") -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Compatibility wrapper around :func:`matrix_assembly.assemble_mass_matrix`."""
    return _canonical_assemble_mass_matrix(model)


def _constraint_value_map(model: "FEModel") -> Dict[int, float]:
    """Return fixed prescribed displacement values keyed by global DOF."""
    dof_manager = model.mesh.dof_manager
    values: Dict[int, float] = {}
    for bc in model.boundary_conditions:
        for dof, value in bc.get_constrained_dofs(dof_manager):
            if dof in values and not np.isclose(values[dof], value):
                raise ValueError(f"Conflicting prescribed displacement for DOF {dof}: {values[dof]} vs {value}")
            values[dof] = float(value)
    return values


def _collect_mpc_constraints(model: "FEModel") -> List[Dict[str, Any]]:
    """Collect model equations and element slave/master constraints."""
    constraints: List[Dict[str, Any]] = []
    for index, equation in enumerate(getattr(model, "constraint_equations", ())):
        raw_terms = getattr(equation, "terms", getattr(equation, "coefficients", ()))
        term_items = raw_terms.items() if hasattr(raw_terms, "items") else raw_terms
        terms: Dict[int, float] = {}
        for dof, coefficient in term_items:
            terms[int(dof)] = terms.get(int(dof), 0.0) + float(coefficient)
        slave = int(getattr(equation, "dependent_dof"))
        pivot = float(terms.pop(slave))
        constraints.append(
            {
                "slave": slave,
                "masters": {
                    int(dof): -float(coefficient) / pivot
                    for dof, coefficient in terms.items()
                    if abs(float(coefficient)) > 0.0
                },
                "value": float(getattr(equation, "rhs", getattr(equation, "value", 0.0))) / pivot,
                "label": str(getattr(equation, "source_id", "") or f"equation_{index}"),
                "kind": "equation",
            }
        )
    for element in model.mesh.elements.values():
        getter = getattr(element, "get_mpc_constraints", None)
        if getter is None:
            continue
        element_constraints = getter(model.mesh)
        if element_constraints:
            constraints.extend(element_constraints)
    return constraints


def build_constraint_transformation(
    K: sparse.csr_matrix,
    F: np.ndarray,
    model: "FEModel",
) -> Tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build the reduced system using fixed DOFs and linear MPC slave relations.

    The returned system is K_red q = F_red and the full displacement vector is
    recovered from u = T q + u0.
    """
    constraint_audit = require_valid_constraints(model)
    total_dofs = int(K.shape[0])
    fixed_values: Dict[int, float] = {}
    mpc_constraints: List[Dict[str, Any]] = []
    for equation in constraint_audit.equations:
        coefficients = {int(dof): float(value) for dof, value in equation.coefficients}
        dependent = int(equation.dependent_dof)
        pivot = coefficients.pop(dependent)
        prescribed = float(equation.value) / pivot
        if equation.kind == "support":
            previous = fixed_values.get(dependent)
            if previous is not None and not np.isclose(previous, prescribed):
                # Normally unreachable because require_valid_constraints has
                # already rejected the conflict; retain a defensive guard.
                raise ValueError(
                    f"Conflicting prescribed displacement for DOF {dependent}: "
                    f"{previous} vs {prescribed}"
                )
            fixed_values[dependent] = prescribed
            continue
        mpc_constraints.append(
            {
                "slave": dependent,
                "masters": {
                    dof: -coefficient / pivot
                    for dof, coefficient in coefficients.items()
                },
                "value": prescribed,
                "label": equation.origin,
                "kind": equation.kind,
            }
        )

    slave_constraints: Dict[int, Dict[str, Any]] = {}
    for constraint in mpc_constraints:
        slave = int(constraint["slave"])
        if slave in fixed_values:
            raise ValueError(f"DOF {slave} is both fixed and used as an MPC slave")
        if slave in slave_constraints:
            raise ValueError(f"DOF {slave} has multiple MPC slave definitions")
        masters = {int(k): float(v) for k, v in constraint.get("masters", {}).items() if abs(float(v)) > 0.0}
        if slave in masters:
            raise ValueError(f"MPC constraint for DOF {slave} references itself as a master")
        slave_constraints[slave] = {
            "masters": masters,
            "value": float(constraint.get("value", 0.0)),
            "label": constraint.get("label", "mpc"),
        }

    fixed_dofs = set(int(dof) for dof in fixed_values)
    slave_dofs = set(slave_constraints)
    dependent_dofs = fixed_dofs | slave_dofs
    independent_dofs = np.array([dof for dof in range(total_dofs) if dof not in dependent_dofs], dtype=int)
    independent_index = {int(dof): i for i, dof in enumerate(independent_dofs)}

    # Topological sort of slave_constraints to resolve cascading dependencies
    visited = {}  # 0 = unvisited, 1 = visiting, 2 = visited
    topo_order = []

    def visit(node: int):
        state = visited.get(node, 0)
        if state == 1:
            raise ValueError(f"Circular MPC dependency detected containing DOF {node}")
        if state == 2:
            return

        visited[node] = 1
        if node in slave_constraints:
            for master in slave_constraints[node]["masters"]:
                if master in slave_constraints:
                    visit(master)
        visited[node] = 2
        topo_order.append(node)

    for slave in slave_constraints:
        if visited.get(slave, 0) == 0:
            visit(slave)

    # Process and resolve master-slave dependencies recursively in topological order
    for slave in topo_order:
        constraint = slave_constraints[slave]
        resolved_masters: Dict[int, float] = {}
        resolved_value = constraint["value"]

        for master, coefficient in constraint["masters"].items():
            if master in slave_constraints:
                sub_masters = slave_constraints[master]["resolved_masters"]
                sub_value = slave_constraints[master]["resolved_value"]
                resolved_value += coefficient * sub_value
                for sub_m, sub_coeff in sub_masters.items():
                    resolved_masters[sub_m] = resolved_masters.get(sub_m, 0.0) + coefficient * sub_coeff
            else:
                resolved_masters[master] = resolved_masters.get(master, 0.0) + coefficient

        constraint["resolved_masters"] = resolved_masters
        constraint["resolved_value"] = resolved_value

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    u0 = np.zeros(total_dofs, dtype=float)

    for dof, value in fixed_values.items():
        u0[int(dof)] = float(value)

    for col, dof in enumerate(independent_dofs):
        rows.append(int(dof))
        cols.append(col)
        data.append(1.0)

    for slave, constraint in slave_constraints.items():
        value = constraint["resolved_value"]
        for master, coefficient in constraint["resolved_masters"].items():
            if master in fixed_values:
                value += coefficient * fixed_values[master]
            else:
                try:
                    col = independent_index[master]
                except KeyError as exc:
                    raise ValueError(f"MPC master DOF {master} is outside the active system") from exc
                rows.append(slave)
                cols.append(col)
                data.append(coefficient)
        u0[slave] = value

    T = sparse.csr_matrix((data, (rows, cols)), shape=(total_dofs, len(independent_dofs)))
    residual_offset = F - K @ u0
    K_red = (T.T @ K @ T).tocsr()
    F_red = np.asarray(T.T @ residual_offset, dtype=float).reshape(-1)

    info = {
        "method": "transformation",
        "num_total_dofs": total_dofs,
        "num_independent_dofs": int(len(independent_dofs)),
        "num_fixed_dofs": int(len(fixed_dofs)),
        "num_mpc_slave_dofs": int(len(slave_dofs)),
        "num_mpc_constraints": int(len(mpc_constraints)),
        "num_generalized_constraint_equations": int(
            sum(str(constraint.get("kind", "mpc")) == "equation" for constraint in mpc_constraints)
        ),
        "fixed_dofs": sorted(fixed_dofs),
        "slave_dofs": sorted(slave_dofs),
        "audit": constraint_audit.to_dict(),
    }
    return K_red, F_red, T, u0, independent_dofs, info


def reconstruct_full_solution(T: sparse.csr_matrix, q: np.ndarray, u0: np.ndarray) -> np.ndarray:
    """Reconstruct full displacement vector from reduced unknowns."""
    return np.asarray(T @ q + u0, dtype=float).reshape(-1)


def _node_components(model: "FEModel") -> List[List[int]]:
    """Connected components from elements and MPC node relationships."""
    mesh = model.mesh
    adjacency: Dict[int, set] = {int(node_id): set() for node_id in mesh.nodes}

    def connect(node_ids: List[int]) -> None:
        ids = [int(node_id) for node_id in node_ids if int(node_id) in adjacency]
        for node_id in ids:
            adjacency[node_id].update(other for other in ids if other != node_id)

    for element in mesh.elements.values():
        connect(list(getattr(element, "node_ids", [])))
    for constraint in _collect_mpc_constraints(model):
        slave_node, _, _ = mesh.dof_manager.get_dof_info(int(constraint["slave"]))
        nodes = [slave_node]
        for master in constraint.get("masters", {}):
            master_node, _, _ = mesh.dof_manager.get_dof_info(int(master))
            nodes.append(master_node)
        connect([node_id for node_id in nodes if node_id >= 0])

    components: List[List[int]] = []
    visited = set()
    for node_id in sorted(adjacency):
        if node_id in visited:
            continue
        stack = [node_id]
        component = []
        visited.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current]):
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _rigid_body_modes_full(model: "FEModel", total_dofs: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Build six full-system rigid body modes per connected component."""
    components = _node_components(model)
    # Supported components contribute no candidates. Retain the original full
    # width/order for any unsupported component, but avoid a quadratic zero
    # grid when every component is supported.
    modes = None
    component_info: List[Dict[str, Any]] = []
    constrained_dofs = set(getattr(model.mesh.dof_manager, "_constrained_dofs", set()))
    for component_index, node_ids in enumerate(components):
        if not node_ids:
            continue
        component_dofs: List[int] = []
        for node_id in node_ids:
            node = model.mesh.get_node(node_id)
            if node is not None:
                component_dofs.extend(int(dof) for dof in node.dofs)
        component_supported = any(dof in constrained_dofs for dof in component_dofs)
        coords = np.asarray([model.mesh.get_node(node_id).coords() for node_id in node_ids], dtype=float)
        origin = np.mean(coords, axis=0)
        base = 6 * component_index
        if component_supported:
            component_info.append(
                {
                    "component_index": component_index,
                    "node_ids": [int(node_id) for node_id in node_ids],
                    "centroid": origin.tolist(),
                    "candidate_modes": 0,
                    "supported": True,
                }
            )
            continue
        if modes is None:
            modes = np.zeros((total_dofs, 6 * len(components)), dtype=float)
        for node_id in node_ids:
            node = model.mesh.get_node(node_id)
            x, y, z = node.coords() - origin
            ux, uy, uz, rx, ry, rz = node.dofs[:6]

            modes[ux, base + 0] = 1.0
            modes[uy, base + 1] = 1.0
            modes[uz, base + 2] = 1.0

            modes[uy, base + 3] = -z
            modes[uz, base + 3] = y
            modes[rx, base + 3] = 1.0

            modes[ux, base + 4] = z
            modes[uz, base + 4] = -x
            modes[ry, base + 4] = 1.0

            modes[ux, base + 5] = -y
            modes[uy, base + 5] = x
            modes[rz, base + 5] = 1.0
        component_info.append(
            {
                "component_index": component_index,
                "node_ids": [int(node_id) for node_id in node_ids],
                "centroid": origin.tolist(),
                "candidate_modes": 6,
                "supported": False,
            }
        )
    if modes is None:
        modes = np.zeros((total_dofs, 0), dtype=float)
    return modes, component_info


def _orthonormalize_columns(matrix: np.ndarray, tolerance: float = 1.0e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Return independent orthonormal columns and the kept column indices."""
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float), np.zeros(0, dtype=int)

    kept: List[int] = []
    basis: List[np.ndarray] = []
    scale = max(float(np.linalg.norm(matrix)), 1.0)
    for col in range(matrix.shape[1]):
        vector = np.asarray(matrix[:, col], dtype=float).copy()
        for q in basis:
            vector -= q * float(q @ vector)
        for q in basis:
            vector -= q * float(q @ vector)
        norm = float(np.linalg.norm(vector))
        if norm > tolerance * scale:
            basis.append(vector / norm)
            kept.append(col)

    if not basis:
        return np.zeros((matrix.shape[0], 0), dtype=float), np.zeros(0, dtype=int)
    return np.column_stack(basis), np.asarray(kept, dtype=int)


def build_reduced_rigid_body_modes(
    model: "FEModel",
    independent_dofs: np.ndarray,
    total_dofs: int,
    transformation: Optional[sparse.spmatrix] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build constraint-compatible rigid modes in reduced coordinate space.

    When ``transformation`` is supplied, rigid translations/rotations are
    intersected with the homogeneous affine constraint space.  The small Gram
    eigenproblem operates on at most six columns per connected component and
    correctly retains combinations such as translation along a rotated roller
    while rejecting the restrained normal direction.
    """
    full_modes, component_info = _rigid_body_modes_full(model, total_dofs)
    if len(independent_dofs) == 0:
        return np.zeros((0, 0), dtype=float), {"rank": 0, "kept_mode_indices": [], "components": component_info}
    reduced_seed = full_modes[np.asarray(independent_dofs, dtype=int), :]
    compatibility_rank = int(full_modes.shape[1])
    if transformation is None and full_modes.shape[1] == 0:
        # Legacy untransformed diagnostics reported the nominal six slots per
        # component even when supported components left every slot zero.
        compatibility_rank = 6 * len(component_info)
    compatibility_error = 0.0
    if transformation is None or full_modes.shape[1] == 0:
        reduced_modes = reduced_seed
    else:
        active_columns = np.linalg.norm(full_modes, axis=0) > 1.0e-14
        full_active = full_modes[:, active_columns]
        seed_active = reduced_seed[:, active_columns]
        if full_active.shape[1] == 0:
            reduced_modes = np.zeros((len(independent_dofs), 0), dtype=float)
            compatibility_rank = 0
            mismatch = np.zeros((total_dofs, 0), dtype=float)
        else:
            reconstructed = np.asarray(transformation @ seed_active, dtype=float)
            mismatch = full_active - reconstructed
        gram = np.asarray(mismatch.T @ mismatch, dtype=float)
        gram = 0.5 * (gram + gram.T)
        if gram.shape[0] > 0:
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            reference = max(float(np.linalg.norm(full_active)) ** 2, 1.0)
            threshold = (1.0e-10 ** 2) * reference
            compatible = eigenvalues <= threshold
            compatibility_rank = int(np.count_nonzero(compatible))
            if compatibility_rank:
                coefficient_basis = eigenvectors[:, compatible]
                reduced_modes = seed_active @ coefficient_basis
                compatibility_error = float(
                    np.linalg.norm(mismatch @ coefficient_basis)
                    / max(float(np.linalg.norm(full_active @ coefficient_basis)), 1.0)
                )
            else:
                reduced_modes = np.zeros((len(independent_dofs), 0), dtype=float)
    q_modes, kept = _orthonormalize_columns(reduced_modes)
    return q_modes, {
        "rank": int(q_modes.shape[1]),
        "kept_mode_indices": [int(i) for i in kept],
        "component_count": len(component_info),
        "components": component_info,
        "rank_method": "modified_gram_schmidt_with_rank_tolerance",
        "constraint_compatibility_rank": compatibility_rank,
        "constraint_compatibility_error": compatibility_error,
        "constraint_compatibility_method": (
            "affine_transformation_intersection"
            if transformation is not None
            else "independent_dof_projection"
        ),
        "description": "Six rigid-body candidates per connected component: Tx, Ty, Tz, Rx, Ry, Rz.",
    }


def apply_boundary_conditions(K: sparse.csr_matrix, F: np.ndarray, dof_manager: "DOFManager") -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Legacy penalty boundary-condition application."""
    penalty = 1.0e15
    constrained_dofs = getattr(dof_manager, "_constrained_dofs", set())
    if not constrained_dofs:
        return K, F
    K_modified = K.tolil()
    F_modified = F.copy()
    for dof in constrained_dofs:
        K_modified[dof, dof] = penalty
        F_modified[dof] = 0.0
    return K_modified.tocsr(), F_modified


def _equilibrate_supported_stiffness(
    matrix: sparse.csr_matrix,
) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[str, Any]]:
    """Return a symmetric diagonal congruence of a supported stiffness.

    Shell systems legitimately mix membrane/shear terms of order ``t`` with
    bending terms of order ``t**3``.  Factoring the raw reduced matrix makes
    that physical unit disparity an avoidable source of binary64 error.  The
    Q1C/Q1D qualification chain used the same diagonal congruence before its
    independently checked solve.  Applying it here changes coordinates only::

        K_eq = D K D,  f_eq = D f,  q = D q_eq.

    No stiffness coefficient, constraint, load, or formulation is changed.
    The floor is relative to the largest supported diagonal and exists only to
    keep a singular/near-mechanism diagnostic finite; the subsequent
    factorization and backward-error check still fail closed.
    """

    made = sparse.csr_matrix(matrix, dtype=float)
    if made.shape[0] != made.shape[1]:
        raise ValueError("reduced stiffness matrix must be square")
    if np.any(~np.isfinite(made.data)):
        raise ValueError("reduced stiffness matrix contains NaN/Inf")
    diagonal = np.abs(np.asarray(made.diagonal(), dtype=float))
    if diagonal.size == 0:
        return made.copy(), np.ones(0, dtype=float), {
            "id": _SYMMETRIC_DIAGONAL_EQUILIBRATION_ID,
            "applied": False,
            "diagonal_floor": 0.0,
            "diagonal_ratio": 1.0,
            "scaled_diagonal_ratio": 1.0,
            "disposition": "EMPTY_SYSTEM",
        }
    if np.any(~np.isfinite(diagonal)):
        raise ValueError("reduced stiffness diagonal contains NaN/Inf")
    largest = float(np.max(diagonal))
    if largest <= 0.0:
        # Preserve the singular matrix for the backend's normal failure path.
        return made.copy(), np.ones(made.shape[0], dtype=float), {
            "id": _SYMMETRIC_DIAGONAL_EQUILIBRATION_ID,
            "applied": False,
            "diagonal_floor": 0.0,
            "diagonal_ratio": None,
            "scaled_diagonal_ratio": None,
            "disposition": "NO_POSITIVE_DIAGONAL",
        }
    floor = max(
        largest * _SYMMETRIC_DIAGONAL_FLOOR_RATIO,
        np.finfo(float).tiny,
    )
    protected = np.maximum(diagonal, floor)
    scaling = 1.0 / np.sqrt(protected)
    if np.any(~np.isfinite(scaling)) or np.any(scaling <= 0.0):
        raise ValueError("reduced stiffness equilibration scale is non-finite")
    diagonal_operator = sparse.diags(scaling, format="csr")
    equilibrated = sparse.csr_matrix(diagonal_operator @ made @ diagonal_operator)
    equilibrated.sort_indices()
    scaled_diagonal = np.abs(np.asarray(equilibrated.diagonal(), dtype=float))
    positive = diagonal[diagonal > 0.0]
    scaled_positive = scaled_diagonal[scaled_diagonal > 0.0]
    diagonal_ratio = (
        float(largest / float(np.min(positive))) if positive.size else None
    )
    if diagonal_ratio is not None and not np.isfinite(diagonal_ratio):
        diagonal_ratio = None
    scaled_diagonal_ratio = (
        float(np.max(scaled_positive) / np.min(scaled_positive))
        if scaled_positive.size
        else None
    )
    if scaled_diagonal_ratio is not None and not np.isfinite(scaled_diagonal_ratio):
        scaled_diagonal_ratio = None
    return equilibrated, scaling, {
        "id": _SYMMETRIC_DIAGONAL_EQUILIBRATION_ID,
        "applied": True,
        "diagonal_floor": floor,
        "diagonal_ratio": diagonal_ratio,
        "scaled_diagonal_ratio": scaled_diagonal_ratio,
        "disposition": "EQUILIBRATED",
    }


def _backward_error(
    matrix: sparse.csr_matrix,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Return the normwise relative backward error of one or many solves."""

    made_matrix = sparse.csr_matrix(matrix, dtype=float)
    made_solution = np.asarray(solution, dtype=float)
    made_rhs = np.asarray(rhs, dtype=float)
    if (
        np.any(~np.isfinite(made_matrix.data))
        or np.any(~np.isfinite(made_solution))
        or np.any(~np.isfinite(made_rhs))
    ):
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = np.asarray(made_matrix @ made_solution - made_rhs, dtype=float)
        row_sums = np.asarray(np.abs(made_matrix).sum(axis=1), dtype=float).reshape(-1)
    if np.any(~np.isfinite(residual)) or np.any(~np.isfinite(row_sums)):
        return float("inf")
    matrix_norm = float(np.max(row_sums)) if row_sums.size else 0.0
    if not np.isfinite(matrix_norm):
        return float("inf")
    if made_solution.ndim == 1:
        solution_norm = float(np.linalg.norm(made_solution, ord=np.inf))
        rhs_norm = float(np.linalg.norm(made_rhs, ord=np.inf))
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        denominator = matrix_norm * solution_norm + rhs_norm
        if not all(
            np.isfinite(value)
            for value in (solution_norm, rhs_norm, residual_norm, denominator)
        ):
            return float("inf")
        if denominator <= 0.0:
            return 0.0 if residual_norm == 0.0 else float("inf")
        return residual_norm / denominator
    solution_norm = np.max(np.abs(made_solution), axis=0)
    rhs_norm = np.max(np.abs(made_rhs), axis=0)
    residual_norm = np.max(np.abs(residual), axis=0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = matrix_norm * solution_norm + rhs_norm
        values = np.where(
            denominator > 0.0,
            residual_norm / denominator,
            np.where(residual_norm == 0.0, 0.0, np.inf),
        )
    if (
        np.any(~np.isfinite(solution_norm))
        or np.any(~np.isfinite(rhs_norm))
        or np.any(~np.isfinite(residual_norm))
        or np.any(~np.isfinite(denominator))
        or np.any(~np.isfinite(values))
    ):
        return float("inf")
    return float(np.max(values)) if values.size else 0.0


def _solve_reduced_system(
    K_red: sparse.csr_matrix,
    F_red: np.ndarray,
    solver_type: str,
    *,
    factorization_cache: Optional[FactorizationCache] = None,
    factorization_signature: Optional[str] = None,
    prepared_operator: Any = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if K_red.shape[0] == 0:
        return np.zeros(0), {"status": "converged", "note": "no independent DOFs"}

    if solver_type == "direct":
        try:
            if prepared_operator is None:
                equilibrated, scaling, equilibration_info = _equilibrate_supported_stiffness(K_red)
            else:
                equilibrated, scaling = prepared_operator.matrix, prepared_operator.scaling
                equilibration_info = dict(prepared_operator.info["equilibration"])
            scaled_rhs = scaling * np.asarray(F_red, dtype=float)
            cache_signature = (
                None
                if factorization_signature is None
                else (
                    f"{factorization_signature}:"
                    f"{_SYMMETRIC_DIAGONAL_EQUILIBRATION_ID}"
                )
            )
            if prepared_operator is not None:
                handle = prepared_operator.handle
            elif factorization_cache is None:
                handle = factorize(
                    equilibrated,
                    MatrixClass.SYMMETRIC_INDEFINITE,
                    signature=cache_signature,
                )
            else:
                handle = factorize_cached(
                    equilibrated,
                    MatrixClass.SYMMETRIC_INDEFINITE,
                    cache=factorization_cache,
                    signature=cache_signature,
                )
            if handle.status != "ok":
                return np.zeros(K_red.shape[0]), {
                    "status": "failed",
                    "error": handle.failure_reason,
                    "backend": handle.diagnostics(),
                    "equilibration": equilibration_info,
                }
            q = scaling * handle.solve(scaled_rhs)
            backward_error = _backward_error(K_red, q, F_red)
            if (
                not np.isfinite(backward_error)
                or backward_error > _LINEAR_STATIC_BACKWARD_ERROR_LIMIT
            ):
                backward_error_disposition = (
                    "NONFINITE_OR_OVERFLOW"
                    if not np.isfinite(backward_error)
                    else "LIMIT_EXCEEDED"
                )
                return q, {
                    "status": "failed",
                    "error": (
                        "linear static solve exceeded the certified relative "
                        "backward-error limit"
                    ),
                    "backend": handle.diagnostics(),
                    "equilibration": equilibration_info,
                    "relative_backward_error": (
                        backward_error if np.isfinite(backward_error) else None
                    ),
                    "relative_backward_error_disposition": (
                        backward_error_disposition
                    ),
                    "relative_backward_error_limit": (
                        _LINEAR_STATIC_BACKWARD_ERROR_LIMIT
                    ),
                }
            return q, {
                "status": "converged",
                "backend": handle.diagnostics(),
                "equilibration": equilibration_info,
                "relative_backward_error": backward_error,
                "relative_backward_error_disposition": "CERTIFIED",
                "relative_backward_error_limit": (
                    _LINEAR_STATIC_BACKWARD_ERROR_LIMIT
                ),
            }
        except Exception as exc:
            return np.zeros(K_red.shape[0]), {"status": "failed", "error": str(exc)}

    if solver_type == "gmres":
        q, info = gmres(K_red, F_red, rtol=1.0e-8, atol=1.0e-12, maxiter=1000)
        return np.asarray(q, dtype=float), {"status": "converged" if info == 0 else "not_converged", "iterations": info}

    if solver_type == "minres":
        q, info = minres(K_red, F_red, rtol=1.0e-8, maxiter=1000)
        return np.asarray(q, dtype=float), {"status": "converged" if info == 0 else "not_converged", "iterations": info}

    if solver_type == "bicgstab":
        q, info = bicgstab(K_red, F_red, rtol=1.0e-8, atol=1.0e-12, maxiter=1000)
        return np.asarray(q, dtype=float), {"status": "converged" if info == 0 else "not_converged", "iterations": info}

    raise ValueError(f"Unknown solver type: {solver_type}")


def _solve_nullspace_augmented_system(
    K_red: sparse.csr_matrix,
    F_red: np.ndarray,
    Q: np.ndarray,
    load_imbalance_tolerance: float = 1.0e-7,
    allow_unbalanced_loads: bool = False,
    prepared_operator: Any = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve free-free reduced system with rigid body modes constrained by Q.T q = 0."""
    n = int(K_red.shape[0])
    r = int(Q.shape[1])
    if n == 0:
        return np.zeros(0), {"status": "converged", "note": "no independent DOFs", "nullspace_rank": r}
    if r == 0:
        q, info = _solve_reduced_system(K_red, F_red, "direct", prepared_operator=prepared_operator)
        info["nullspace_rank"] = 0
        return q, info

    Q_sparse = sparse.csr_matrix(Q)
    zero = sparse.csr_matrix((r, r), dtype=float)
    augmented = (sparse.bmat([[K_red, Q_sparse], [Q_sparse.T, zero]], format="csr")
                 if prepared_operator is None else prepared_operator.matrix)
    rhs = np.concatenate([F_red, np.zeros(r, dtype=float)])

    load_components = np.asarray(Q.T @ F_red, dtype=float).reshape(-1)
    load_imbalance_norm = float(np.linalg.norm(load_components))
    load_norm = float(np.linalg.norm(F_red))
    relative_imbalance = load_imbalance_norm / max(load_norm, 1.0)

    warnings: List[str] = []
    if relative_imbalance > load_imbalance_tolerance:
        message = (
            "The external load vector has a non-zero rigid-body component. "
            "For a physical free-free static solution, use self-equilibrated loads."
        )
        if not allow_unbalanced_loads:
            return np.zeros(n), {
                "status": "incompatible_free_free_load",
                "error": message,
                "nullspace_rank": r,
                "rigid_body_load_components": load_components.tolist(),
                "rigid_body_load_imbalance_norm": load_imbalance_norm,
                "relative_rigid_body_load_imbalance": relative_imbalance,
            }
        warnings.append(
            message
            + " The nullspace solve returned a gauged displacement field and balancing generalized reactions."
        )

    try:
        handle = (factorize(augmented, MatrixClass.SYMMETRIC_INDEFINITE)
                  if prepared_operator is None else prepared_operator.handle)
        if handle.status != "ok":
            return np.zeros(n), {
                "status": "failed",
                "error": handle.failure_reason,
                "nullspace_rank": r,
                "rigid_body_load_components": load_components.tolist(),
                "relative_rigid_body_load_imbalance": relative_imbalance,
                "backend": handle.diagnostics(),
            }
        solution = handle.solve(rhs)
        solution = np.asarray(solution, dtype=float).reshape(-1)
        if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
            return np.zeros(n), {
                "status": "singular",
                "error": "NaN/Inf solution in nullspace augmented solve",
                "nullspace_rank": r,
                "rigid_body_load_components": load_components.tolist(),
                "relative_rigid_body_load_imbalance": relative_imbalance,
                "backend": handle.diagnostics(),
            }
    except Exception as exc:
        return np.zeros(n), {
            "status": "failed",
            "error": str(exc),
            "nullspace_rank": r,
            "rigid_body_load_components": load_components.tolist(),
            "relative_rigid_body_load_imbalance": relative_imbalance,
        }

    q = solution[:n]
    multipliers = solution[n:]
    residual = np.asarray(K_red @ q + Q @ multipliers - F_red, dtype=float).reshape(-1)
    gauge = np.asarray(Q.T @ q, dtype=float).reshape(-1)

    return q, {
        "status": "converged",
        "method": "rigid_body_nullspace_augmented",
        "nullspace_rank": r,
        "rigid_body_load_components": load_components.tolist(),
        "rigid_body_lagrange_multipliers": multipliers.tolist(),
        "rigid_body_load_imbalance_norm": load_imbalance_norm,
        "relative_rigid_body_load_imbalance": relative_imbalance,
        "augmented_residual_norm": float(np.linalg.norm(residual)),
        "gauge_residual_norm": float(np.linalg.norm(gauge)),
        "warnings": warnings,
        "backend": handle.diagnostics(),
    }


@resource_threaded
def solve_linear(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    solver_type: str = "direct",
    precond: bool = True,
    constraint_mode: str = "auto",
    allow_unbalanced_free_free: bool = False,
    resource_config: Optional[ResourceConfig] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    session: Optional["AnalysisSession"] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve a linear FE problem using fixed-DOF/MPC transformation.

    constraint_mode:
        - "auto": use the ordinary reduced solve when fixed DOFs exist, otherwise
          use rigid-body nullspace augmentation.
        - "transformation": always use the ordinary reduced solve.
        - "nullspace": always use nullspace augmentation after MPC/fixed reduction.
    """
    operation_started = time.perf_counter()
    cancellation_safe_point(cancellation_token, "linear_static.start")
    mesh = model.mesh
    dof_manager = mesh.dof_manager

    model.apply_boundary_conditions()
    validation_finished = time.perf_counter()
    assembly_started = validation_finished
    if session is None:
        K, F, assembly_info = assemble_system(model, load_case)
        K_red, F_red, T, u0, independent_dofs, constraint_info = (
            build_constraint_transformation(K, F, model)
        )
        constraint_plan = None
    else:
        K, F, assembly_info, constraint_plan, F_red = session.linear_system(
            load_case,
            model,
        )
        K_red = constraint_plan.K_red
        T = constraint_plan.T
        u0 = constraint_plan.u0
        independent_dofs = constraint_plan.independent_dofs
        constraint_info = dict(constraint_plan.info)
    cancellation_safe_point(cancellation_token, "linear_static.after_assembly")
    assembly_finished = time.perf_counter()

    total_dofs = int(K.shape[0])
    if session is None:
        Q, nullspace_info = build_reduced_rigid_body_modes(
            model,
            independent_dofs,
            total_dofs,
            transformation=T,
        )
    else:
        Q, nullspace_info = session.rigid_body_modes(constraint_plan, model)
    mode = (constraint_mode or "auto").strip().lower()
    if mode not in {"auto", "transformation", "nullspace"}:
        raise ValueError(f"Unknown constraint_mode '{constraint_mode}'. Use auto, transformation or nullspace.")
    use_nullspace = mode == "nullspace" or (mode == "auto" and int(constraint_info["num_fixed_dofs"]) == 0 and Q.shape[1] > 0)

    solver_info: Dict[str, Any] = {
        "assembly": assembly_info,
        "solver_type": solver_type,
        "constraint_method": "transformation_fixed_plus_mpc_nullspace" if use_nullspace else "transformation_fixed_plus_mpc",
        "constraint_mode": mode,
        "num_free_dofs": int(len(independent_dofs)),
        "num_constrained_dofs": int(constraint_info["num_fixed_dofs"]),
        "num_mpc_slave_dofs": int(constraint_info["num_mpc_slave_dofs"]),
        "solve_time": 0.0,
        "constraint_info": constraint_info,
        "nullspace_info": nullspace_info,
        "convergence_info": {},
        "thread_policy": thread_policy_diagnostics(resource_config),
        "phase_timings_seconds": {
            "validation": float(validation_finished - operation_started),
            "assembly_and_reduction": float(assembly_finished - assembly_started),
        },
    }
    if session is not None:
        solver_info["analysis_session"] = session.diagnostics()

    solve_started_perf = time.perf_counter()
    solver_info["phase_timings_seconds"]["rigid_mode_preparation"] = float(
        solve_started_perf - assembly_finished
    )
    start_time = time.time()
    if use_nullspace:
        prepared_operator = (None if session is None else
            session.static_operator_plan(constraint_mode="nullspace", model=model))
        q, convergence_info = _solve_nullspace_augmented_system(
            K_red,
            F_red,
            Q,
            allow_unbalanced_loads=allow_unbalanced_free_free,
            prepared_operator=prepared_operator,
        )
    else:
        q, convergence_info = _solve_reduced_system(
            K_red,
            F_red,
            solver_type,
            prepared_operator=(session.static_operator_plan(constraint_mode="transformation", model=model)
                if session is not None and solver_type == "direct" else None),
            factorization_cache=(
                session.factorization_cache
                if session is not None and solver_type == "direct"
                else None
            ),
            factorization_signature=(
                session.factorization_signature("linear_static", constraint_plan)
                if session is not None and solver_type == "direct"
                else None
            ),
        )
    cancellation_safe_point(cancellation_token, "linear_static.after_factorization")
    solver_info["solve_time"] = time.time() - start_time
    solver_info["phase_timings_seconds"]["factorization_and_solve"] = float(
        solver_info["solve_time"]
    )
    solve_finished_perf = time.perf_counter()
    solver_info["convergence_info"] = convergence_info
    if convergence_info.get("status") != "converged":
        displacement = u0.copy()
    else:
        displacement = reconstruct_full_solution(T, q, u0)
    solver_info["constraint_postcheck"] = constraint_residual_summary(model, displacement)
    solver_info["phase_timings_seconds"]["reconstruction_and_postcheck"] = float(
        time.perf_counter() - solve_finished_perf
    )
    solver_info["phase_timings_seconds"]["total"] = float(
        time.perf_counter() - operation_started
    )
    result_case = make_result_case(
        name=f"linear_static:{getattr(load_case, 'name', 'none')}",
        analysis_type="linear_static",
        load_cases=() if load_case is None else (load_case,),
        assembly_info=assembly_info,
        solver_info=solver_info,
        recovery={"displacements": True, "stresses": "on_demand", "reactions": "on_demand"},
        settings={"constraint_mode": mode, "solver_type": solver_type},
        warnings=convergence_info.get("warnings", ()),
    )
    solver_info["result_case"] = result_case.to_dict()
    # Private handoff for application facades that need the exact applied-load
    # resultant.  It is attached after ResultCase serialization and can be
    # consumed without assembling the load a second time.
    solver_info["_assembled_load_vector"] = np.asarray(F, dtype=float)
    emit_progress(
        progress_callback,
        "linear_static_complete",
        "linear_static.complete",
        completed=1,
        total=1,
        status=convergence_info.get("status", "unknown"),
        num_free_dofs=int(len(independent_dofs)),
    )
    return displacement, solver_info


@resource_threaded
def solve_linear_many(
    model: "FEModel",
    load_cases: List[Optional["LoadCase"]],
    constraint_mode: str = "auto",
    factorization_cache: Optional[FactorizationCache] = None,
    resource_config: Optional[ResourceConfig] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    session: Optional["AnalysisSession"] = None,
    allow_unbalanced_free_free: bool = False,
    *, inertia_relief: bool = False, reference_point: Any = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve several unchanged-stiffness static load cases with one factorization.

    The returned displacement matrix has one column per load case. Free-free
    nullspace and optional inertia relief use the same prepared block engine
    as ``iter_linear_batches``; supported-static defaults retain their route.
    """
    if type(inertia_relief) is not bool:
        raise ValueError("inertia_relief must be a boolean")
    if reference_point is not None and not inertia_relief:
        raise ValueError("reference_point requires inertia_relief=True")
    # Check before any downstream model dispatch: adapters must never silently
    # discard the physical inertia-relief request.
    if inertia_relief and getattr(type(model), "_solid_model_protocol", False):
        from .inertia_relief import InertiaReliefError
        raise InertiaReliefError("Inertia relief requires a beam/shell FEModel", code="unsupported_model")
    cancellation_safe_point(cancellation_token, "linear_static_many.start")
    mode = (constraint_mode or "auto").strip().lower()
    batch_route = inertia_relief or mode == "nullspace"
    if mode == "auto" and not batch_route:
        model.apply_boundary_conditions()
        # The batch engine makes the final surviving-mode decision. Preparing
        # a second stiffness/load system here would duplicate its fast path.
        batch_route = not require_valid_constraints(model).fixed_dofs
    if batch_route:
        from .static_batch import StaticCase, iter_linear_batches
        from .analysis_session import AnalysisSession
        if session is not None and factorization_cache is not None and factorization_cache is not session.factorization_cache:
            raise ValueError("nullspace solve requires factorization_cache to match the supplied session cache")
        owned_session = session is None
        active = AnalysisSession(model, factorization_cache=factorization_cache) if owned_session else session
        primary_failure = None
        try:
            blocks = list(iter_linear_batches(model,
                (StaticCase(str(index), case) for index, case in enumerate(load_cases)),
                constraint_mode=mode, allow_unbalanced_free_free=allow_unbalanced_free_free,
                inertia_relief=inertia_relief, reference_point=reference_point,
                session=active, resource_config=resource_config,
                cancellation_token=cancellation_token, progress_callback=progress_callback))
            cache_info = active.factorization_cache.diagnostics()
        except BaseException as error:
            primary_failure = error
            raise
        finally:
            if owned_session:
                try:
                    active.close()
                except BaseException as cleanup_error:
                    if primary_failure is None:
                        raise
                    BaseException.add_note(primary_failure, "analysis session cleanup failed: " + repr(cleanup_error))
        full = np.column_stack([block.displacements for block in blocks]) if blocks else np.zeros((model.mesh.dof_manager.total_dofs, 0))
        effective_mode = blocks[0].diagnostics["constraint_mode"] if blocks else mode
        info = {"status": "converged", "num_result_cases": len(load_cases),
                "constraint_mode": effective_mode, "requested_constraint_mode": mode,
                "analysis_type": "inertia_relief" if inertia_relief else "linear_static_many",
                "case_results": tuple(result for block in blocks for result in block.case_results),
                "static_batches": tuple(block.diagnostics for block in blocks)}
        info["result_case"] = make_result_case(name=info["analysis_type"],
            analysis_type=info["analysis_type"],
            load_cases=tuple(case for case in load_cases if case is not None), solver_info=info,
            settings={"constraint_mode": effective_mode, "num_load_cases": len(load_cases),
                      "inertia_relief": inertia_relief}).to_dict()
        info["factorization_cache"] = cache_info
        if session is not None:
            info["analysis_session"] = session.diagnostics()
            info["factorization_cache"] = session.factorization_cache.diagnostics()
        return full, info
    model.apply_boundary_conditions()
    if session is None:
        K, _, assembly_info = assemble_system(model, None)
        zero_load = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
        K_red, _, T, u0, independent_dofs, constraint_info = (
            build_constraint_transformation(K, zero_load, model)
        )
        constraint_plan = None
    else:
        K, _, assembly_info, constraint_plan, _ = session.linear_system(None, model)
        K_red = constraint_plan.K_red
        T = constraint_plan.T
        u0 = constraint_plan.u0
        independent_dofs = constraint_plan.independent_dofs
        constraint_info = dict(constraint_plan.info)
    F_matrix, load_matrix_info = assemble_load_matrix(model, load_cases)
    F_red = np.asarray(T.T @ (F_matrix - (K @ u0)[:, None]), dtype=float)

    total_dofs = int(K.shape[0])
    if session is None:
        Q, nullspace_info = build_reduced_rigid_body_modes(
            model,
            independent_dofs,
            total_dofs,
            transformation=T,
        )
    else:
        Q, nullspace_info = session.rigid_body_modes(constraint_plan, model)
    mode = (constraint_mode or "auto").strip().lower()
    if mode not in {"auto", "transformation"}:
        raise ValueError("solve_linear_many supports constraint_mode 'auto' or 'transformation'")
    if mode == "auto" and int(constraint_info["num_fixed_dofs"]) == 0 and Q.shape[1] > 0:
        raise ValueError("solve_linear_many requires supports; use individual solve_linear for free-free nullspace solves")

    start = time.time()
    local_cache = (
        factorization_cache
        or (session.factorization_cache if session is not None else None)
        or FactorizationCache(name="linear_static_many", max_entries=1)
    )
    revisions = getattr(model, "revision_signature", lambda: getattr(model.mesh, "revision_signature", lambda: {})())()
    if session is None:
        stiffness_signature = json.dumps(
            {
                "analysis": "linear_static_many",
                "matrix": "K_reduced",
                "shape": tuple(int(v) for v in K_red.shape),
                "nnz": int(K_red.nnz),
                "model_revision": revisions,
                "constraint_method": constraint_info.get("method"),
                "num_independent_dofs": int(len(independent_dofs)),
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    else:
        stiffness_signature = session.factorization_signature(
            "linear_static_many",
            constraint_plan,
        )
    prepared_operator = (None if session is None or (factorization_cache is not None
        and factorization_cache is not session.factorization_cache) else
        session.static_operator_plan(constraint_mode="transformation", model=model))
    if prepared_operator is None:
        equilibrated, scaling, equilibration_info = _equilibrate_supported_stiffness(K_red)
    else:
        equilibrated, scaling = prepared_operator.matrix, prepared_operator.scaling
        equilibration_info = dict(prepared_operator.info["equilibration"])
    equilibrated_signature = (
        f"{stiffness_signature}:{_SYMMETRIC_DIAGONAL_EQUILIBRATION_ID}"
    )
    handle = prepared_operator.handle if prepared_operator is not None else factorize_cached(
        equilibrated,
        MatrixClass.SYMMETRIC_INDEFINITE,
        cache=local_cache,
        signature=equilibrated_signature,
    )
    cancellation_safe_point(cancellation_token, "linear_static_many.after_factorization")
    if handle.status != "ok":
        info = {
            "status": "failed",
            "error": handle.failure_reason,
            "assembly": assembly_info,
            "load_matrix": load_matrix_info,
            "constraint_info": constraint_info,
            "nullspace_info": nullspace_info,
            "backend": handle.diagnostics(),
            "equilibration": equilibration_info,
            "factorization_cache": local_cache.diagnostics(),
            "solve_time": time.time() - start,
            "thread_policy": thread_policy_diagnostics(resource_config),
        }
        if session is not None:
            info["analysis_session"] = session.diagnostics()
        displacement = np.tile(u0.reshape(-1, 1), (1, len(load_cases)))
        info["constraint_postcheck"] = constraint_residual_summary(model, displacement)
        info["result_case"] = make_result_case(
            name="linear_static_many",
            analysis_type="linear_static_many",
            load_cases=tuple(load_case for load_case in load_cases if load_case is not None),
            assembly_info={**assembly_info, "load_matrix": load_matrix_info},
            solver_info=info,
            recovery={"displacements": True, "stresses": "on_demand", "reactions": "on_demand"},
            settings={"constraint_mode": mode, "num_load_cases": len(load_cases)},
        ).to_dict()
        return displacement, info
    scaled_rhs = scaling[:, None] * F_red
    q_matrix = scaling[:, None] * handle.solve_many(scaled_rhs)
    backward_error = _backward_error(K_red, q_matrix, F_red)
    cancellation_safe_point(cancellation_token, "linear_static_many.after_solve")
    if (
        not np.isfinite(backward_error)
        or backward_error > _LINEAR_STATIC_BACKWARD_ERROR_LIMIT
    ):
        backward_error_disposition = (
            "NONFINITE_OR_OVERFLOW"
            if not np.isfinite(backward_error)
            else "LIMIT_EXCEEDED"
        )
        info = {
            "status": "failed",
            "error": (
                "linear static many solve exceeded the certified relative "
                "backward-error limit"
            ),
            "assembly": assembly_info,
            "load_matrix": load_matrix_info,
            "constraint_info": constraint_info,
            "nullspace_info": nullspace_info,
            "backend": handle.diagnostics(),
            "equilibration": equilibration_info,
            "factorization_cache": local_cache.diagnostics(),
            "relative_backward_error": (
                backward_error if np.isfinite(backward_error) else None
            ),
            "relative_backward_error_disposition": backward_error_disposition,
            "relative_backward_error_limit": _LINEAR_STATIC_BACKWARD_ERROR_LIMIT,
            "solve_time": time.time() - start,
            "thread_policy": thread_policy_diagnostics(resource_config),
        }
        if session is not None:
            info["analysis_session"] = session.diagnostics()
        displacement = np.tile(u0.reshape(-1, 1), (1, len(load_cases)))
        info["constraint_postcheck"] = constraint_residual_summary(
            model, displacement
        )
        info["result_case"] = make_result_case(
            name="linear_static_many",
            analysis_type="linear_static_many",
            load_cases=tuple(
                load_case for load_case in load_cases if load_case is not None
            ),
            assembly_info={**assembly_info, "load_matrix": load_matrix_info},
            solver_info=info,
            recovery={
                "displacements": True,
                "stresses": "on_demand",
                "reactions": "on_demand",
            },
            settings={
                "constraint_mode": mode,
                "num_load_cases": len(load_cases),
            },
        ).to_dict()
        return displacement, info
    full = np.asarray(T @ q_matrix + u0[:, None], dtype=float)
    info = {
        "status": "converged",
        "assembly": assembly_info,
        "load_matrix": load_matrix_info,
        "constraint_info": constraint_info,
        "nullspace_info": nullspace_info,
        "backend": handle.diagnostics(),
        "equilibration": equilibration_info,
        "factorization_cache": local_cache.diagnostics(),
        "relative_backward_error": backward_error,
        "relative_backward_error_disposition": "CERTIFIED",
        "relative_backward_error_limit": _LINEAR_STATIC_BACKWARD_ERROR_LIMIT,
        "solve_time": time.time() - start,
        "num_result_cases": len(load_cases),
        "thread_policy": thread_policy_diagnostics(resource_config),
    }
    if session is not None:
        info["analysis_session"] = session.diagnostics()
    info["constraint_postcheck"] = constraint_residual_summary(model, full)
    info["result_case"] = make_result_case(
        name="linear_static_many",
        analysis_type="linear_static_many",
        load_cases=tuple(load_case for load_case in load_cases if load_case is not None),
        assembly_info={**assembly_info, "load_matrix": load_matrix_info},
        solver_info=info,
        recovery={"displacements": True, "stresses": "on_demand", "reactions": "on_demand"},
        settings={"constraint_mode": mode, "num_load_cases": len(load_cases)},
    ).to_dict()
    emit_progress(
        progress_callback,
        "linear_static_many_complete",
        "linear_static_many.complete",
        completed=len(load_cases),
        total=len(load_cases),
        num_result_cases=len(load_cases),
        status="converged",
    )
    return full, info


def solve_nonlinear(
    model: "FEModel",
    load_case: "LoadCase",
    max_iterations: int = 20,
    tolerance: float = 1.0e-6,
    method: str = "newton_raphson",
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Deprecated prototype nonlinear solve.

    Use :func:`anyfesolver.solve_static_nonlinear` for the qualified incremental
    geometric/material nonlinear API.
    """
    cancellation_safe_point(cancellation_token, "deprecated_nonlinear.start")
    warnings.warn(
        "solve_nonlinear() is a deprecated prototype; use solve_static_nonlinear()",
        DeprecationWarning,
        stacklevel=2,
    )
    require_no_native_total_lagrangian_elements(
        model,
        context="deprecated solve_nonlinear",
    )
    mesh = model.mesh
    dof_manager = mesh.dof_manager
    model.apply_boundary_conditions()

    K, F_ext, assembly_info = assemble_system(model, load_case)
    K_red, F_red, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K, F_ext, model)
    q = np.zeros(len(independent_dofs), dtype=float)
    u = reconstruct_full_solution(T, q, u0)

    solver_info: Dict[str, Any] = {
        "assembly": assembly_info,
        "method": method,
        "constraint_method": "transformation_fixed_plus_mpc",
        "constraint_info": constraint_info,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "iterations": 0,
        "converged": False,
        "residual_history": [],
        "displacement_norm_history": [],
        "iteration_times": [],
    }

    for iteration in range(max_iterations):
        cancellation_safe_point(
            cancellation_token,
            f"deprecated_nonlinear.iteration:{iteration + 1}",
        )
        iter_start = time.time()
        F_int = compute_internal_forces(model, u)
        residual_full = F_ext - F_int
        residual_red = np.asarray(T.T @ residual_full, dtype=float).reshape(-1)
        residual_norm = float(np.linalg.norm(residual_red))
        solver_info["residual_history"].append(residual_norm)
        emit_progress(
            progress_callback,
            "nonlinear_iteration",
            "deprecated_nonlinear.iteration",
            completed=iteration + 1,
            total=max_iterations,
            iteration=iteration + 1,
            residual_norm=residual_norm,
        )

        if residual_norm < tolerance:
            solver_info["converged"] = True
            solver_info["iterations"] = iteration + 1
            break

        if method == "newton_raphson":
            K, _, _ = assemble_system(model, load_case)
            K_red, _, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K, np.zeros_like(F_ext), model)

        dq, convergence = _solve_reduced_system(K_red, residual_red, "direct")
        if convergence.get("status") != "converged":
            solver_info["convergence_info"] = convergence
            break
        q += dq
        u = reconstruct_full_solution(T, q, u0)
        solver_info["displacement_norm_history"].append(float(np.linalg.norm(dq)))
        solver_info["iteration_times"].append(time.time() - iter_start)

    return u, solver_info


def _compute_internal_forces_under_lease(
    model: "FEModel",
    displacements: np.ndarray,
    *,
    qualified_runtime_guard: Any,
) -> np.ndarray:
    """Compute internal forces for all elements."""
    qualified_runtime_guard(stage="internal-force preflight")
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    F_int = np.zeros(total_dofs, dtype=float)

    for elem_id, element in mesh.elements.items():
        material = model.get_material(element.material_name)
        qualified_runtime_guard(
            stage=f"internal-force material observation for element {elem_id}"
        )
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue
        u_elem = displacements[dof_mapping]
        F_elem = np.asarray(element.compute_internal_forces(mesh, u_elem, material), dtype=float)
        qualified_runtime_guard(
            stage=f"internal-force element observation for element {elem_id}"
        )
        np.add.at(F_int, dof_mapping, F_elem)
    return F_int


def compute_internal_forces(model: "FEModel", displacements: np.ndarray) -> np.ndarray:
    """Compute all element forces under one operation-wide authority lease."""

    return _run_with_qualified_recovery_runtime_lease(
        model,
        context="internal-force computation",
        operation=lambda guard: _compute_internal_forces_under_lease(
            model,
            displacements,
            qualified_runtime_guard=guard,
        ),
    )


def _add_dof_force(force_map: Dict[int, np.ndarray], model: "FEModel", dof: int, value: float) -> None:
    """Accumulate one global-DOF force into a node-indexed six-component map."""
    node_id, local_index, _name = model.mesh.dof_manager.get_dof_info(int(dof))
    if node_id < 0 or local_index < 0:
        return
    force_map.setdefault(int(node_id), np.zeros(6, dtype=float))[int(local_index)] += float(value)


def _compact_force_map(force_map: Dict[int, np.ndarray], tolerance: float = 0.0) -> Dict[int, np.ndarray]:
    """Drop near-zero entries from a node force map."""
    compact: Dict[int, np.ndarray] = {}
    for node_id, values in force_map.items():
        vector = np.asarray(values, dtype=float).reshape(6)
        if np.any(np.abs(vector) > float(tolerance)):
            compact[int(node_id)] = vector
    return compact


def _compute_constraint_force_diagnostics_under_lease(
    model: "FEModel",
    displacements: np.ndarray,
    load_case: Optional["LoadCase"] = None,
    *,
    force_tolerance: float = 0.0,
    qualified_runtime_guard: Any,
    prepared_system: Optional[Tuple[Any, ...]] = None,
) -> Dict[str, Any]:
    """Return separated support, MPC and nullspace force diagnostics.

    The raw residual convention is ``K u - F`` on the unreduced global system.
    ``support_reactions`` contains only fixed-DOF residuals. ``mpc_slave_forces``
    contains residuals on slave DOFs, while ``mpc_master_equivalent_forces``
    pushes those slave residuals through the linear MPC coefficients to show
    the equivalent force/moment components seen by master DOFs.
    """
    mesh = model.mesh
    dof_manager = mesh.dof_manager
    model.apply_boundary_conditions()

    if prepared_system is None:
        K, _, _ = assemble_system(model)
    else:
        K, F_ext, prepared_constraint, prepared_Q, prepared_mode_info = prepared_system
    qualified_runtime_guard(stage="constraint-force stiffness observation")
    if prepared_system is not None:
        F_ext = np.asarray(F_ext, dtype=float).reshape(-1)
        qualified_runtime_guard(stage="constraint-force prepared-load observation")
    elif load_case is None:
        F_ext = np.zeros(dof_manager.total_dofs, dtype=float)
    else:
        F_ext = load_case.get_load_vector(
            mesh,
            dof_manager,
            model.get_material,
            element_activity=getattr(mesh, "element_activity", None),
        )
        qualified_runtime_guard(stage="constraint-force load-provider observation")
        F_ext = np.asarray(F_ext, dtype=float)
        qualified_runtime_guard(stage="constraint-force load-array observation")

    u = np.asarray(displacements, dtype=float).reshape(-1)
    qualified_runtime_guard(stage="constraint-force displacement observation")
    if u.shape[0] != int(K.shape[0]):
        raise ValueError(f"Displacement vector length {u.shape[0]} does not match system size {K.shape[0]}")

    residual = np.asarray(K @ u - F_ext, dtype=float).reshape(-1)
    fixed_dofs = sorted(int(dof) for dof in getattr(dof_manager, "_constrained_dofs", set()))
    mpc_constraints = _collect_mpc_constraints(model)

    support_reactions: Dict[int, np.ndarray] = {}
    mpc_slave_forces: Dict[int, np.ndarray] = {}
    mpc_master_equivalent_forces: Dict[int, np.ndarray] = {}
    mpc_constraint_forces: List[Dict[str, Any]] = []

    for dof in fixed_dofs:
        _add_dof_force(support_reactions, model, dof, residual[dof])

    for index, constraint in enumerate(mpc_constraints):
        slave = int(constraint["slave"])
        slave_force = float(residual[slave])
        slave_node, slave_local, slave_component = dof_manager.get_dof_info(slave)
        _add_dof_force(mpc_slave_forces, model, slave, slave_force)

        master_entries: List[Dict[str, Any]] = []
        for master, coefficient in constraint.get("masters", {}).items():
            master_dof = int(master)
            value = float(coefficient) * slave_force
            _add_dof_force(mpc_master_equivalent_forces, model, master_dof, value)
            master_node, master_local, master_component = dof_manager.get_dof_info(master_dof)
            master_entries.append(
                {
                    "dof": master_dof,
                    "node_id": int(master_node),
                    "local_index": int(master_local),
                    "component": str(master_component),
                    "coefficient": float(coefficient),
                    "equivalent_force": value,
                }
            )

        mpc_constraint_forces.append(
            {
                "index": int(index),
                "label": str(constraint.get("label", f"mpc_{index}")),
                "slave_dof": slave,
                "slave_node_id": int(slave_node),
                "slave_local_index": int(slave_local),
                "slave_component": str(slave_component),
                "slave_force": slave_force,
                "master_equivalent_forces": master_entries,
                "master_equivalent_norm": float(np.linalg.norm([entry["equivalent_force"] for entry in master_entries])),
            }
        )

    if prepared_system is None:
        K_red, _, T, _, independent_dofs, constraint_info = build_constraint_transformation(K, F_ext, model)
        Q, nullspace_info = build_reduced_rigid_body_modes(
            model, independent_dofs, int(K.shape[0]), transformation=T,
        )
    else:
        T = prepared_constraint.T
        constraint_info = dict(prepared_constraint.info)
        Q, nullspace_info = prepared_Q, prepared_mode_info
    reduced_residual = np.asarray(T.T @ residual, dtype=float).reshape(-1)
    if Q.shape[1] > 0:
        nullspace_generalized_forces = np.asarray(Q.T @ reduced_residual, dtype=float).reshape(-1)
    else:
        nullspace_generalized_forces = np.zeros(0, dtype=float)

    return {
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "reduced_residual_norm": float(np.linalg.norm(reduced_residual)),
        "fixed_dofs": fixed_dofs,
        "mpc_slave_dofs": sorted(int(constraint["slave"]) for constraint in mpc_constraints),
        "support_reactions": _compact_force_map(support_reactions, force_tolerance),
        "mpc_slave_forces": _compact_force_map(mpc_slave_forces, force_tolerance),
        "mpc_master_equivalent_forces": _compact_force_map(mpc_master_equivalent_forces, force_tolerance),
        "mpc_constraint_forces": mpc_constraint_forces,
        "support_reaction_norm": float(np.linalg.norm(np.concatenate(list(support_reactions.values()))) if support_reactions else 0.0),
        "mpc_slave_force_norm": float(np.linalg.norm(np.concatenate(list(mpc_slave_forces.values()))) if mpc_slave_forces else 0.0),
        "mpc_master_equivalent_force_norm": float(
            np.linalg.norm(np.concatenate(list(mpc_master_equivalent_forces.values()))) if mpc_master_equivalent_forces else 0.0
        ),
        "nullspace_generalized_forces": nullspace_generalized_forces,
        "nullspace_generalized_force_norm": float(np.linalg.norm(nullspace_generalized_forces)),
        "constraint_info": constraint_info,
        "nullspace_info": nullspace_info,
    }


def compute_constraint_force_diagnostics(
    model: "FEModel",
    displacements: np.ndarray,
    load_case: Optional["LoadCase"] = None,
    *,
    force_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Compute constraint forces under one operation-wide authority lease."""

    return _run_with_qualified_recovery_runtime_lease(
        model,
        context="constraint-force diagnostics",
        operation=lambda guard: _compute_constraint_force_diagnostics_under_lease(
            model,
            displacements,
            load_case,
            force_tolerance=force_tolerance,
            qualified_runtime_guard=guard,
        ),
    )


def _compute_reactions_under_lease(
    model: "FEModel",
    displacements: np.ndarray,
    load_case: "LoadCase",
    *,
    qualified_runtime_guard: Any,
) -> Dict[int, np.ndarray]:
    """Compute legacy combined reactions at fixed and MPC slave DOFs."""
    diagnostics = _compute_constraint_force_diagnostics_under_lease(
        model,
        displacements,
        load_case,
        qualified_runtime_guard=qualified_runtime_guard,
    )
    reactions: Dict[int, np.ndarray] = {}
    for bucket in ("support_reactions", "mpc_slave_forces"):
        for node_id, values in diagnostics[bucket].items():
            reactions.setdefault(int(node_id), np.zeros(6, dtype=float))
            reactions[int(node_id)] += np.asarray(values, dtype=float).reshape(6)
    return _compact_force_map(reactions)


def compute_reactions(
    model: "FEModel",
    displacements: np.ndarray,
    load_case: "LoadCase",
) -> Dict[int, np.ndarray]:
    """Compute reactions under one operation-wide authority lease."""

    return _run_with_qualified_recovery_runtime_lease(
        model,
        context="reaction computation",
        operation=lambda guard: _compute_reactions_under_lease(
            model,
            displacements,
            load_case,
            qualified_runtime_guard=guard,
        ),
    )


def _compute_stresses_under_lease(
    model: "FEModel",
    displacements: np.ndarray,
    return_global: bool = False,
    element_ids: Optional[Sequence[int]] = None,
    *,
    qualified_runtime_guard: Any,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute stresses for all or selected elements."""
    qualified_runtime_guard(stage="stress-recovery preflight")
    qualified_runtime_guard(stage="constraint-force preflight")
    observation_guard = _qualified_recovery_observation_guard(
        qualified_runtime_guard
    )
    observation_namespace = getattr(observation_guard, "__dict__", {})
    runtime_namespace = getattr(qualified_runtime_guard, "__dict__", {})
    exact_beam_candidate = (
        dict.get(
            observation_namespace,
            "_qualified_exact_builtin_beam_candidate",
        )
        if type(observation_namespace) is dict
        else None
    )
    trusted_segment_candidate = (
        dict.get(
            observation_namespace,
            "_qualified_recovery_trusted_segment_candidate",
        )
        if type(observation_namespace) is dict
        else None
    )
    exact_beam_batch_open = False
    captured_beam_dof_mapping = (
        dict.get(runtime_namespace, "_qualified_captured_beam_dof_mapping")
        if type(runtime_namespace) is dict
        else None
    )
    recover_captured_beam = (
        dict.get(runtime_namespace, "_qualified_recover_captured_beam")
        if type(runtime_namespace) is dict
        else None
    )
    mesh = model.mesh
    stresses: Dict[int, Dict[str, np.ndarray]] = {}
    selected = None if element_ids is None else {int(element_id) for element_id in element_ids}
    selected_ids = (
        tuple(int(element_id) for element_id in mesh.elements)
        if selected is None
        else tuple(sorted(selected))
    )
    if return_global:
        require_model_element_capabilities(
            model,
            "global_recovery",
            context="compute_stresses",
            element_ids=selected_ids,
        )
    displacements = np.asarray(displacements, dtype=float)
    for elem_id, element in mesh.elements.items():
        if selected is not None and int(elem_id) not in selected:
            continue
        exact_beam = bool(
            callable(exact_beam_candidate)
            and exact_beam_candidate(element)
        )
        trusted_segment_element = bool(
            callable(trusted_segment_candidate)
            and trusted_segment_candidate(element)
        )
        if exact_beam and not exact_beam_batch_open:
            qualified_runtime_guard(
                stage="exact built-in beam recovery batch preflight"
            )
            exact_beam_batch_open = True
        elif not trusted_segment_element and exact_beam_batch_open:
            qualified_runtime_guard(
                stage="exact built-in beam recovery batch output"
            )
            exact_beam_batch_open = False
        material = model.get_material(element.material_name)
        observation_guard(
            stage=f"stress material observation for element {elem_id}",
            element=element,
            material=material,
        )
        if exact_beam and callable(captured_beam_dof_mapping):
            raw_dof_mapping = captured_beam_dof_mapping(element, material)
        else:
            raw_dof_mapping = element.get_dof_mapping(mesh)
        dof_mapping = np.asarray(raw_dof_mapping, dtype=np.intp)
        if (
            dof_mapping.size == 0
            or int(dof_mapping.min()) < 0
            or int(dof_mapping.max()) >= displacements.size
        ):
            if bool(getattr(element, "recovery_errors_fail_closed", False)):
                raise ValueError(
                    "fail-closed element recovery requires a complete in-range "
                    "DOF mapping"
                )
            continue
        try:
            if exact_beam and callable(recover_captured_beam):
                stresses[elem_id] = recover_captured_beam(
                    element,
                    mesh,
                    displacements[dof_mapping],
                    material,
                    return_global,
                )
            else:
                stresses[elem_id] = element.compute_stresses(
                    mesh,
                    displacements[dof_mapping],
                    material,
                    return_global=return_global,
                )
            observation_guard(
                stage=f"stress element observation for element {elem_id}",
                element=element,
                material=material,
            )
        except (IndexError, ValueError):
            if bool(getattr(element, "recovery_errors_fail_closed", False)):
                raise
            continue
    if exact_beam_batch_open:
        qualified_runtime_guard(
            stage="exact built-in beam recovery batch output"
        )
    return stresses


def compute_stresses(
    model: "FEModel",
    displacements: np.ndarray,
    return_global: bool = False,
    element_ids: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute stresses under one operation-wide authority lease."""

    return _run_with_qualified_recovery_runtime_lease(
        model,
        context="stress computation",
        operation=lambda guard: _compute_stresses_under_lease(
            model,
            displacements,
            return_global,
            element_ids,
            qualified_runtime_guard=guard,
        ),
    )


def extract_node_displacements(displacements: np.ndarray, mesh: "FEMesh") -> Dict[int, np.ndarray]:
    """Extract displacements for each node from the global vector."""
    displacements = np.asarray(displacements, dtype=float)
    return {
        node_id: displacements[np.asarray(node.dofs, dtype=np.intp)]
        for node_id, node in mesh.nodes.items()
    }


def extract_element_displacements(displacements: np.ndarray, mesh: "FEMesh") -> Dict[int, np.ndarray]:
    """Extract element displacement vectors."""
    displacements = np.asarray(displacements, dtype=float)
    elem_displacements: Dict[int, np.ndarray] = {}
    for elem_id, element in mesh.elements.items():
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size:
            elem_displacements[elem_id] = displacements[dof_mapping]
    return elem_displacements
