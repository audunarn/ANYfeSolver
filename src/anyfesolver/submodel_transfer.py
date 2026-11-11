"""Sequential displacement-driven shell submodel transfer.

Implements the ANYfeSolver part of the submodelling contract
(``docs/SUBMODEL_TRANSFER_CONTRACT.md``):

* :func:`interpolate_nodal_field` -- evaluate a source solution at target points
  with the source element's own interpolation functions and the rigid transport
  rule ``u(x_f) = u(x_p) + theta(x_p) x (x_f - x_p)``; alternative hits on a
  shared boundary are evaluated too and their spread is recorded;
* :func:`interpolation_allowance` -- the per-component allowance of one value;
* :func:`build_driven_trace` / :func:`apply_driven_trace` -- the constraint-reduced
  driven boundary: every driven row is reduced to the model's independent
  coordinates, redundant rows are reported and dropped, inconsistent rows and
  numerically ambiguous rank decisions refuse, and constraint slaves are never
  prescribed directly;
* :func:`update_driven_trace_values` -- change only the driven values, atomically,
  so a session refreshes the constraint offset without refactorizing;
* :func:`interface_forces` -- interface forces and retained support reactions from
  one multiplier solve over the physical constraint rows, with equilibrium
  enforced against the force allowance;
* :func:`estimate_static_error_allowance` / :func:`force_allowance` -- the
  first-order allowance on the solver's own equilibrated operator.

This is a nodal-field transfer, not a formulation-native physical recovery.
Source-result admission (a certified linear static solve) belongs to the
calling workflow; this module validates shapes, finiteness, element types and
control parameters and fails closed.  Shell admission is an explicit allow-list
of exact classes, for sources and targets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest

__all__ = [
    "DOF_NAMES",
    "TRACE_POLICIES",
    "JUNCTION_RANK3",
    "JUNCTION_RANK2",
    "SubmodelTransferError",
    "TransferPoint",
    "TransferValue",
    "TraceRow",
    "DrivenTrace",
    "InterfaceForces",
    "StaticErrorAllowance",
    "admitted_transfer_element_types",
    "element_interpolation_weights",
    "interpolate_nodal_field",
    "interpolation_allowance",
    "rotation_rows",
    "build_driven_trace",
    "apply_driven_trace",
    "update_driven_trace_values",
    "interface_forces",
    "interface_resultant",
    "estimate_static_error_allowance",
    "force_allowance",
]

DOF_NAMES: Tuple[str, ...] = ("ux", "uy", "uz", "rx", "ry", "rz")
TRACE_POLICIES: Tuple[str, ...] = ("drive_all", "release_normal_rotation")
_EPS = float(np.finfo(float).eps)
#: Junction rule (contract section 4).
JUNCTION_RANK3 = 1.0e-3
JUNCTION_RANK2 = 1.0e-9
#: An equilibrated reduced row whose surviving size lies above its roundoff bound
#: but below this value is a numerically ambiguous rank decision and refuses.
_RANK_INDEPENDENT = 1.0e-8
#: Same limit the linear static path certifies (assembly._LINEAR_STATIC_BACKWARD_ERROR_LIMIT).
_CERTIFIED_BACKWARD_ERROR = 1.0e-10


class SubmodelTransferError(ValueError):
    """A typed, fail-closed refusal of the submodel transfer."""

    def __init__(self, code: str, message: str, *, details: Optional[Mapping[str, Any]] = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = dict(details or {})


def _finite(value: Any, code: str, what: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise SubmodelTransferError(code, f"{what} is not numeric") from error
    if not np.all(np.isfinite(array)):
        raise SubmodelTransferError(code, f"{what} is not finite")
    return array


def _control(value: Any, what: str, *, positive: bool = False) -> float:
    number = float(_finite(value, "CONTROL_INVALID", what).reshape(()))
    if number < 0.0 or (positive and number == 0.0):
        raise SubmodelTransferError("CONTROL_INVALID", f"{what} must be {'positive' if positive else 'non-negative'}")
    return number


def _six(value: Any, code: str, what: str, *, non_negative: bool = False) -> np.ndarray:
    array = _finite(value, code, what).reshape(-1)
    if array.shape != (6,):
        raise SubmodelTransferError(code, f"{what} needs six components")
    if non_negative and np.any(array < 0.0):
        raise SubmodelTransferError(code, f"{what} must be non-negative")
    return array


def admitted_transfer_element_types() -> Tuple[type, ...]:
    """Exact shell classes admitted as transfer sources and targets (v1)."""

    from .e4_pl_element import QualifiedE4PLShellElement
    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    return (QualifiedE4PLShellElement, NativeParityE4PLS3V2DShellElement)


def _require_admitted(element: Any, element_id: int) -> None:
    admitted = admitted_transfer_element_types()
    if type(element) not in admitted:
        raise SubmodelTransferError(
            "TRANSFER_ELEMENT_NOT_ADMITTED",
            f"element {element_id} is {type(element).__name__}; admitted shells: "
            + ", ".join(cls.__name__ for cls in admitted),
            details={"element_id": int(element_id), "type": type(element).__name__},
        )


def _require_target_shells(model: Any) -> None:
    """Every shell of a target model must be admitted; beams and constraint-only
    elements (couplings) are retained as they are."""

    from .elements import ShellElement

    for element_id, element in model.mesh.elements.items():
        if isinstance(element, ShellElement):
            _require_admitted(element, int(element_id))


def _shape(element: Any, natural: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

    xi, eta = (float(v) for v in _finite(natural, "TRANSFER_NATURAL", "natural coordinates").reshape(2))
    if type(element) is NativeParityE4PLS3V2DShellElement:
        n, dxi, deta = element.compute_native_surface_shape_functions(xi, eta)
    else:
        n, dxi, deta = element.compute_shape_functions(xi, eta)
    return (np.asarray(n, float).reshape(-1), np.asarray(dxi, float).reshape(-1), np.asarray(deta, float).reshape(-1))


def element_interpolation_weights(element: Any, natural: Sequence[float]) -> np.ndarray:
    """Geometric interpolation weights of an admitted element at ``natural``.

    The E4-PL Q4 uses its bilinear shell functions; the S3 V2D its native surface
    functions (it forbids the legacy ``ShellElement`` dispatch).
    """

    return _shape(element, natural)[0]


def _node_six(model: Any, displacements: np.ndarray, node_id: int) -> np.ndarray:
    dofs = model.mesh.dof_manager.get_node_dofs(int(node_id))
    if len(dofs) != 6:
        raise SubmodelTransferError("TRANSFER_NODE_DOFS", f"node {node_id} has no six DOFs")
    return np.asarray(displacements[dofs], dtype=float)


def _vector(model: Any, displacements: Any, code: str) -> np.ndarray:
    u = _finite(displacements, code, "displacement vector").reshape(-1)
    total = int(model.mesh.dof_manager.total_dofs)
    if u.shape != (total,):
        raise SubmodelTransferError(code, f"displacement vector has {u.size} entries, model has {total} DOFs")
    return u


# ----------------------------------------------------------------------
# interpolation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TransferPoint:
    """One target node and its correspondence in a source element.

    ``natural`` uses the source element's convention: ``(xi, eta)`` in
    ``[-1, 1]^2`` for four-node shells, ``(r, s)`` with ``N = (1 - r - s, r, s)``
    for three-node shells.  ``mesher_weights`` (optional) must agree with the
    element's own weights.  ``inverse_residual`` is the correspondence owner's
    in-plane inverse residual.  ``alternatives`` lists the other conforming hits
    as ``(element_id, natural, inverse_residual)``; a two-element form uses the
    selected residual as the caller's common bound.  Each alternative's complete
    allowance enters the selected value's allowance.
    """

    target_node: int
    element_id: int
    natural: Tuple[float, float]
    target_point: Tuple[float, float, float]
    mesher_weights: Optional[Tuple[float, ...]] = None
    inverse_residual: float = 0.0
    alternatives: Tuple[Tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class TransferValue:
    target_node: int
    values: np.ndarray  # (ux, uy, uz, rx, ry, rz), global frame
    source_point: np.ndarray
    offset: np.ndarray  # target point minus source point
    distance: float
    weights: np.ndarray
    source_nodes: Tuple[int, ...]
    element_id: int
    natural: Tuple[float, float]
    inverse_residual: float
    alternative_spread: np.ndarray  # per component, max |value - alternative value|
    alternatives: Tuple["TransferValue", ...] = ()  # other conforming hits, each with its own geometry


def _evaluate(model: Any, u: np.ndarray, element_id: int, natural: Sequence[float], target: np.ndarray, transport: bool):
    element = model.mesh.elements.get(int(element_id))
    if element is None:
        raise SubmodelTransferError("TRANSFER_ELEMENT_MISSING", f"source element {element_id} is not in the model")
    _require_admitted(element, int(element_id))
    weights = element_interpolation_weights(element, natural)
    node_ids = tuple(int(n) for n in element.node_ids)
    if weights.shape != (len(node_ids),) or not np.all(np.isfinite(weights)):
        raise SubmodelTransferError("TRANSFER_WEIGHTS", f"element {element_id} returned invalid weights")
    coordinates = np.array([model.mesh.get_node(n).coords() for n in node_ids], dtype=float)
    source_point = weights @ coordinates
    values = weights @ np.array([_node_six(model, u, n) for n in node_ids])
    offset = target - source_point
    if transport:
        values = values.copy()
        values[:3] = values[:3] + np.cross(values[3:], offset)
    return weights, node_ids, source_point, offset, values


def interpolate_nodal_field(
    model: Any,
    displacements: Any,
    points: Iterable[TransferPoint],
    *,
    transport: bool = True,
    weight_tolerance: float = 1.0e-12,
) -> Dict[int, TransferValue]:
    """Interpolate a source nodal six-vector field at target points."""

    tolerance = _control(weight_tolerance, "weight_tolerance")
    u = _vector(model, displacements, "TRANSFER_SOURCE")
    out: Dict[int, TransferValue] = {}
    for point in points:
        target = _finite(point.target_point, "TRANSFER_TARGET", f"target point of node {point.target_node}").reshape(3)
        residual = _control(point.inverse_residual, "inverse residual")
        weights, node_ids, source_point, offset, values = _evaluate(model, u, point.element_id, point.natural, target, transport)
        if point.mesher_weights is not None:
            given = _finite(point.mesher_weights, "TRANSFER_CONVENTION_MISMATCH", "correspondence weights").reshape(-1)
            if given.shape != weights.shape or float(np.max(np.abs(given - weights))) > tolerance:
                raise SubmodelTransferError(
                    "TRANSFER_CONVENTION_MISMATCH",
                    f"correspondence weights for node {point.target_node} disagree with element {point.element_id}",
                    details={"element_weights": weights.tolist(), "mesher_weights": given.tolist()},
                )
        spread = np.zeros(6)
        alternatives = []
        for record in point.alternatives:
            if len(record) == 3:
                alt_id, alt_natural, alt_raw_residual = record
                alt_residual = _control(alt_raw_residual, "alternative inverse residual")
            elif len(record) == 2:
                # No own residual given: the selected residual is the caller's common bound.
                alt_id, alt_natural = record
                alt_residual = residual
            else:
                raise SubmodelTransferError("TRANSFER_ALTERNATIVE", "alternatives are (element, natural[, residual])")
            alt_weights, alt_nodes, alt_point, alt_offset, alt_values = _evaluate(
                model, u, alt_id, alt_natural, target, transport)
            spread = np.maximum(spread, np.abs(alt_values - values))
            alternatives.append(TransferValue(
                int(point.target_node), alt_values, alt_point, alt_offset, float(np.linalg.norm(alt_offset)),
                alt_weights, alt_nodes, int(alt_id), tuple(float(v) for v in alt_natural), alt_residual, np.zeros(6),
            ))
        if not (np.all(np.isfinite(values)) and np.all(np.isfinite(spread))):
            raise SubmodelTransferError("TRANSFER_NONFINITE", f"interpolated value of node {point.target_node} is not finite")
        out[int(point.target_node)] = TransferValue(
            int(point.target_node), values, source_point, offset, float(np.linalg.norm(offset)), weights, node_ids,
            int(point.element_id), tuple(float(v) for v in point.natural), residual, spread, tuple(alternatives),
        )
    return out


def _gradient_norms(model: Any, u: np.ndarray, value: TransferValue) -> np.ndarray:
    """Per-component physical gradient magnitude, max over the hit and the element corners."""

    element = model.mesh.elements[value.element_id]
    coordinates = np.array([model.mesh.get_node(n).coords() for n in value.source_nodes], dtype=float)
    nodal = np.array([_node_six(model, u, n) for n in value.source_nodes])
    if len(value.source_nodes) == 4:
        samples = [value.natural, (-1, -1), (1, -1), (1, 1), (-1, 1)]
    else:
        samples = [value.natural, (0, 0), (1, 0), (0, 1)]
    best = np.zeros(6)
    for natural in samples:
        _n, dxi, deta = _shape(element, natural)
        jacobian = np.column_stack([coordinates.T @ dxi, coordinates.T @ deta])  # (3, 2)
        pseudo = np.linalg.pinv(jacobian)  # (2, 3)
        derivative = np.vstack([dxi @ nodal, deta @ nodal])  # (2, 6)
        gradients = pseudo.T @ derivative  # (3, 6)
        best = np.maximum(best, np.linalg.norm(gradients, axis=0))
    return best


def interpolation_allowance(
    value: TransferValue,
    model: Any,
    displacements: Any,
    source_displacement_allowance: Optional[Any] = None,
) -> np.ndarray:
    """Per-component allowance of one interpolated value (contract section 6).

    Terms: evaluation roundoff ``(n_e + 1) eps max|u_i|``; propagated source
    allowance ``sum |N_i| du_i``; the in-plane correspondence residual times the
    physical gradient (shape-function derivatives through the element Jacobian,
    maximum over the hit and the corners); the spread of the alternative
    conforming hits; and, through the transport rule, the rotation allowance
    times ``|offset|`` plus its roundoff.
    """

    u = _vector(model, displacements, "TRANSFER_SOURCE")
    du = None
    if source_displacement_allowance is not None:
        du = _finite(source_displacement_allowance, "ALLOWANCE_INVALID", "source allowance").reshape(-1)
        if du.shape != u.shape or np.any(du < 0.0):
            raise SubmodelTransferError("ALLOWANCE_INVALID", "source allowance must match the DOFs and be non-negative")

    def single(candidate: "TransferValue") -> np.ndarray:
        """Complete allowance of one candidate through its own geometry and offset."""

        nodal_values = np.array([_node_six(model, u, n) for n in candidate.source_nodes])
        part = (len(candidate.source_nodes) + 1) * _EPS * np.max(np.abs(nodal_values), axis=0)
        if du is not None:
            part = part + np.abs(candidate.weights) @ np.array([_node_six(model, du, n) for n in candidate.source_nodes])
        coordinates = np.array([model.mesh.get_node(n).coords() for n in candidate.source_nodes], dtype=float)
        # Position uncertainty of the correspondence point (inverse residual plus
        # coordinate roundoff) moves F(p) = u(p) + theta(p) x (x_f - p) by
        # grad(u) dp + (grad(theta) dp) x offset - theta x dp.
        position = candidate.inverse_residual + 4.0 * _EPS * float(np.max(np.abs(coordinates)))
        gradient = _gradient_norms(model, u, candidate)
        distance = float(np.linalg.norm(candidate.offset))
        part = part + position * gradient
        part[:3] = part[:3] + position * float(np.linalg.norm(gradient[3:])) * distance
        part[:3] = part[:3] + position * float(np.linalg.norm(candidate.values[3:]))
        if distance > 0.0:
            rotation = float(np.linalg.norm(part[3:]))
            part[:3] = part[:3] + rotation * distance + 4.0 * _EPS * float(np.linalg.norm(candidate.values[3:])) * distance
        return part

    allowance = single(value)
    # Alternative conforming hits: each bounded by its difference from the selected
    # value plus its own complete allowance.
    extra = np.zeros(6)
    for alternative in value.alternatives:
        extra = np.maximum(extra, np.abs(alternative.values - value.values) + single(alternative))
    allowance = allowance + extra
    if not np.all(np.isfinite(allowance)):
        raise SubmodelTransferError("ALLOWANCE_NONFINITE", "the interpolation allowance is not finite")
    return allowance


# ----------------------------------------------------------------------
# driven trace
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TraceRow:
    node: int
    label: str
    b: Tuple[Tuple[int, float], ...]  # u-space row: (global DOF, coefficient)
    direction: Optional[Tuple[float, float, float]] = None  # rotation direction of P1 rows


@dataclass
class DrivenTrace:
    """A constraint-reduced driven boundary bound to one target model."""

    policy: str
    rows: Tuple[TraceRow, ...]
    values: np.ndarray  # one driven value per row
    uncertainty: np.ndarray  # per-row allowance of the driven value
    transformation: sparse.csr_matrix  # T of the model without the trace
    offset: np.ndarray  # c of the model without the trace
    independent_dofs: np.ndarray
    reduced_rows: Tuple[Dict[int, float], ...]  # a_i = b_i T (q-space), unscaled
    row_scale: np.ndarray  # equilibration: max |a_i|
    kept: Tuple[int, ...]
    redundant: Tuple[int, ...]
    combinations: Tuple[Dict[int, float], ...]  # per kept row, over scaled original rows
    echelon: Tuple[Dict[int, float], ...]  # per kept row, eliminated scaled q-space row
    pivots: Tuple[int, ...]
    redundant_combinations: Dict[int, Dict[int, float]]
    consistency: Dict[int, Tuple[float, float]]  # redundant row -> (remaining, tolerance), scaled
    model_token: int
    retained_fingerprint: str
    applied: Dict[str, Any] = field(default_factory=dict)


def _unit_rows(node: int, dofs: Sequence[int], names: Sequence[str]) -> List[TraceRow]:
    return [TraceRow(int(node), name, ((int(dof), 1.0),)) for dof, name in zip(dofs, names)]


def rotation_rows(
    node: int,
    rotation_dofs: Sequence[int],
    normals: Sequence[Sequence[float]],
    *,
    rank3: float = JUNCTION_RANK3,
    rank2: float = JUNCTION_RANK2,
) -> List[TraceRow]:
    """Rotation rows of one node under P1 (contract section 4)."""

    rank3 = _control(rank3, "rank3", positive=True)
    rank2 = _control(rank2, "rank2", positive=True)
    if not rank2 < rank3 <= 1.0:
        raise SubmodelTransferError("CONTROL_INVALID", "junction thresholds need 0 < rank2 < rank3 <= 1")
    tangents = []
    for raw in normals:
        n = _finite(raw, "TRACE_NORMAL_INVALID", f"incident normal of node {node}").reshape(3)
        length = float(np.linalg.norm(n))
        if length <= 0.0:
            raise SubmodelTransferError("TRACE_NORMAL_INVALID", f"node {node} has a zero incident normal")
        n = n / length
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, helper)
        t1 /= np.linalg.norm(t1)
        tangents.extend((t1, np.cross(n, t1)))
    if not tangents:
        raise SubmodelTransferError("TRACE_NORMAL_MISSING", f"node {node} has no incident normal for P1")
    _u, sigma, vt = np.linalg.svd(np.array(tangents), full_matrices=True)
    if sigma.size < 3:
        sigma = np.concatenate([sigma, np.zeros(3 - sigma.size)])
    ratio = float(sigma[2] / sigma[0])
    if ratio >= rank3:
        return _unit_rows(node, rotation_dofs, DOF_NAMES[3:])
    if ratio <= rank2:
        rows = []
        for k, direction in enumerate(vt[:2]):
            terms = tuple((int(dof), float(c)) for dof, c in zip(rotation_dofs, direction) if c != 0.0)
            rows.append(TraceRow(int(node), f"t{k}", terms, tuple(float(c) for c in direction)))
        return rows
    raise SubmodelTransferError(
        "TRACE_JUNCTION_ILL_CONDITIONED",
        f"node {node}: incident normals give sigma3/sigma1 = {ratio:.3e}, between {rank2:g} and {rank3:g}",
        details={"node": int(node), "ratio": ratio},
    )


def _constraint_map(model: Any) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    from .assembly import build_constraint_transformation

    total = int(model.mesh.dof_manager.total_dofs)
    _k, _f, t, u0, independent, _info = build_constraint_transformation(
        sparse.identity(total, format="csr"), np.zeros(total), model
    )
    t = sparse.csr_matrix(t)
    t.sort_indices()
    return t, np.asarray(u0, dtype=float), np.asarray(independent, dtype=int)


def _physical_rows(model: Any, exclude_supports=(), exclude_equations=()):
    """Physical retained constraint rows: (kind, origin, {dof: coef}, value)."""

    rows = []
    skip_bc = {id(bc) for bc in exclude_supports}
    skip_eq = set(exclude_equations)
    dm = model.mesh.dof_manager
    for bc in model.boundary_conditions:
        if id(bc) in skip_bc:
            continue
        for dof, value in bc.get_constrained_dofs(dm):
            rows.append(("support", str(bc.name), {int(dof): 1.0}, float(value)))
    for position, equation in enumerate(getattr(model, "constraint_equations", ())):
        if position in skip_eq:
            continue
        terms: Dict[int, float] = {}
        for dof, coefficient in equation.coefficients:
            terms[int(dof)] = terms.get(int(dof), 0.0) + float(coefficient)
        rows.append(("equation", str(equation.origin), terms, float(equation.value)))
    for element_id, element in sorted(model.mesh.elements.items()):
        getter = getattr(element, "get_mpc_constraints", None)
        if getter is None:
            continue
        for record in getter(model.mesh) or ():
            terms = {int(record["slave"]): 1.0}
            for master, coefficient in record.get("masters", {}).items():
                terms[int(master)] = terms.get(int(master), 0.0) - float(coefficient)
            rows.append(("mpc", f"{element_id}:{record.get('label', '')}", terms, float(record.get("value", 0.0))))
    return rows


def _rows_fingerprint(model: Any, rows) -> str:
    digest = hashlib.sha256()
    digest.update(repr(int(model.mesh.dof_manager.total_dofs)).encode())
    signature = model.mesh.revision_signature()
    digest.update(repr([(k, signature.get(k)) for k in ("topology", "geometry", "activity")]).encode())
    for kind, origin, terms, value in rows:
        digest.update(repr((kind, origin, sorted(terms.items()), value)).encode())
    return digest.hexdigest()


def _row_value(row: TraceRow, six: np.ndarray) -> float:
    if row.direction is not None:
        return float(np.dot(np.asarray(row.direction), six[3:]))
    return float(six[DOF_NAMES.index(row.label)])


def _row_allowance(row: TraceRow, allowance: Optional[Mapping[int, np.ndarray]]) -> float:
    if allowance is None or row.node not in allowance:
        return 0.0
    node_allow = allowance[row.node]
    if row.direction is not None:
        return float(np.abs(np.asarray(row.direction)) @ node_allow[3:])
    return float(node_allow[DOF_NAMES.index(row.label)])


def _reduced(rows: Sequence[TraceRow], g: np.ndarray, t: sparse.csr_matrix, c: np.ndarray):
    reduced: List[Dict[int, float]] = []
    h = np.empty(len(rows))
    offset_size = np.empty(len(rows))
    for index, row in enumerate(rows):
        acc: Dict[int, float] = {}
        offset = 0.0
        size = 0.0
        for dof, coefficient in row.b:
            start, end = t.indptr[dof], t.indptr[dof + 1]
            for col, value in zip(t.indices[start:end], t.data[start:end]):
                acc[int(col)] = acc.get(int(col), 0.0) + coefficient * float(value)
            offset += coefficient * float(c[dof])
            size += abs(coefficient) * abs(float(c[dof]))
        reduced.append({col: value for col, value in acc.items() if value != 0.0})
        h[index] = float(g[index]) - offset
        offset_size[index] = size
    return reduced, h, offset_size


def _validated_values(rows_nodes: Iterable[int], values: Mapping[int, Any]) -> Dict[int, np.ndarray]:
    out = {}
    for node in rows_nodes:
        if node not in values:
            raise SubmodelTransferError("TRACE_VALUES", f"no driven values for node {node}")
        out[int(node)] = _six(values[node], "TRACE_VALUES", f"driven values of node {node}")
    return out


def _validated_allowance(allowance: Optional[Mapping[int, Any]]) -> Optional[Dict[int, np.ndarray]]:
    if allowance is None:
        return None
    return {int(k): _six(v, "ALLOWANCE_INVALID", f"allowance of node {k}", non_negative=True) for k, v in allowance.items()}


def _uncertainty(rows, g, offset_size, allowance) -> np.ndarray:
    out = np.array([_row_allowance(row, allowance) + 4.0 * _EPS * (abs(g[i]) + offset_size[i]) for i, row in enumerate(rows)])
    if not np.all(np.isfinite(out)):
        raise SubmodelTransferError("ALLOWANCE_NONFINITE", "trace uncertainty is not finite")
    return out


def _combination_tolerance(combo: Mapping[int, float], h_scaled: np.ndarray, unc_scaled: np.ndarray) -> float:
    weight = sum(abs(c) for c in combo.values())
    tol = (sum(abs(c) * unc_scaled[i] for i, c in combo.items())
           + 16.0 * _EPS * max(weight, 1.0) * sum(abs(c * h_scaled[i]) for i, c in combo.items()))
    if not np.isfinite(tol):
        raise SubmodelTransferError("ALLOWANCE_NONFINITE", "a redundant-row tolerance is not finite")
    return float(tol)


def build_driven_trace(
    model: Any,
    values: Mapping[int, Sequence[float]],
    *,
    policy: str,
    normals: Optional[Mapping[int, Sequence[Sequence[float]]]] = None,
    allowance: Optional[Mapping[int, Sequence[float]]] = None,
) -> DrivenTrace:
    """Build the constraint-reduced driven trace for ``model`` (not yet applied).

    ``values`` maps each driven node to its six transferred values; ``policy`` is
    ``"drive_all"`` (P2) or ``"release_normal_rotation"`` (P1, needs ``normals``
    per node).  There is no default policy.  ``model`` must carry its retained
    supports and internal constraints but not the trace.  ``allowance`` (per node,
    six non-negative components) is the transfer allowance of the driven values.

    Rows are equilibrated (divided by their largest reduced coefficient) and
    eliminated with per-entry tracking of the accumulated operand magnitude; an
    entry is zero only below its own roundoff bound.  A surviving row between that
    bound and ``1e-8`` refuses as numerically ambiguous.
    """

    if policy not in TRACE_POLICIES:
        raise SubmodelTransferError("TRACE_POLICY", f"unknown policy {policy!r}; expected one of {TRACE_POLICIES}")
    if not values:
        raise SubmodelTransferError("TRACE_EMPTY", "no driven nodes")
    _require_target_shells(model)
    nodes = sorted(int(n) for n in values)
    checked = _validated_values(nodes, values)
    allowance_map = _validated_allowance(allowance)
    dm = model.mesh.dof_manager
    rows: List[TraceRow] = []
    row_values: List[float] = []
    for node in nodes:
        dofs = dm.get_node_dofs(node)
        if len(dofs) != 6:
            raise SubmodelTransferError("TRACE_NODE_DOFS", f"driven node {node} is not in the target model")
        node_rows = _unit_rows(node, dofs[:3], DOF_NAMES[:3])
        if policy == "drive_all":
            node_rows += _unit_rows(node, dofs[3:], DOF_NAMES[3:])
        else:
            incident = None if normals is None else normals.get(node)
            if not incident:
                raise SubmodelTransferError("TRACE_NORMAL_MISSING", f"P1 needs incident normals for node {node}")
            node_rows += rotation_rows(node, dofs[3:], incident)
        for row in node_rows:
            rows.append(row)
            row_values.append(_row_value(row, checked[node]))
    g = np.asarray(row_values, dtype=float)
    t, c, independent = _constraint_map(model)
    reduced, h, offset_size = _reduced(rows, g, t, c)
    uncertainty = _uncertainty(rows, g, offset_size, allowance_map)
    scale = np.array([max((abs(v) for v in row.values()), default=0.0) for row in reduced])
    safe = np.where(scale > 0.0, scale, 1.0)
    h_scaled = h / safe
    unc_scaled = uncertainty / safe

    pivot_rows: Dict[int, Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]] = {}
    kept: List[int] = []
    echelon: List[Dict[int, float]] = []
    combinations: List[Dict[int, float]] = []
    pivots: List[int] = []
    redundant: List[int] = []
    consistency: Dict[int, Tuple[float, float]] = {}
    redundant_combinations: Dict[int, Dict[int, float]] = {}
    for index, original in enumerate(reduced):
        if scale[index] == 0.0:
            work: Dict[int, float] = {}
            magnitude: Dict[int, float] = {}
        else:
            work = {k: v / scale[index] for k, v in original.items()}
            magnitude = {k: abs(v) for k, v in work.items()}
        combo = {index: 1.0}
        while True:
            hits = [col for col in work if col in pivot_rows and work[col] != 0.0]
            if not hits:
                break
            for col in hits:
                factor = work.get(col, 0.0)
                if factor == 0.0:
                    continue
                prow, pmag, pcombo = pivot_rows[col]
                for key, value in prow.items():
                    if key == col:
                        continue
                    work[key] = work.get(key, 0.0) - factor * value
                    magnitude[key] = magnitude.get(key, 0.0) + abs(factor) * (pmag.get(key, 0.0) + abs(value))
                for key, value in pcombo.items():
                    combo[key] = combo.get(key, 0.0) - factor * value
                work.pop(col, None)
                magnitude.pop(col, None)
        survivors = {k: v for k, v in work.items() if abs(v) > 16.0 * _EPS * magnitude.get(k, 0.0)}
        size = max((abs(v) for v in survivors.values()), default=0.0)
        remaining = float(sum(coef * h_scaled[i] for i, coef in combo.items()))
        if size == 0.0:
            tol = _combination_tolerance(combo, h_scaled, unc_scaled)
            consistency[index] = (remaining, tol)
            if not np.isfinite(remaining) or abs(remaining) > tol:
                row = rows[index]
                raise SubmodelTransferError(
                    "TRACE_INCONSISTENT",
                    f"driven row {row.label} of node {row.node} contradicts retained constraints or other driven rows "
                    f"(remaining {remaining:.3e}, tolerance {tol:.3e}, equilibrated)",
                    details={"node": row.node, "label": row.label, "remaining": remaining, "tolerance": tol},
                )
            redundant.append(index)
            redundant_combinations[index] = dict(combo)
            continue
        if size < _RANK_INDEPENDENT:
            row = rows[index]
            raise SubmodelTransferError(
                "TRACE_RANK_AMBIGUOUS",
                f"driven row {row.label} of node {row.node} is nearly dependent on earlier rows "
                f"(equilibrated surviving size {size:.3e})",
            )
        pivot = max(survivors, key=lambda key: (abs(survivors[key]), -key))
        norm = survivors[pivot]
        pivot_rows[pivot] = (
            {k: v / norm for k, v in survivors.items()},
            {k: magnitude.get(k, 0.0) / abs(norm) for k in survivors},
            {k: v / norm for k, v in combo.items()},
        )
        kept.append(index)
        echelon.append(dict(survivors))
        combinations.append(dict(combo))
        pivots.append(pivot)
    return DrivenTrace(
        policy=policy, rows=tuple(rows), values=g, uncertainty=uncertainty, transformation=t, offset=c,
        independent_dofs=independent, reduced_rows=tuple(reduced), row_scale=scale, kept=tuple(kept),
        redundant=tuple(redundant), combinations=tuple(combinations), echelon=tuple(echelon), pivots=tuple(pivots),
        redundant_combinations=redundant_combinations, consistency=consistency, model_token=id(model),
        retained_fingerprint=_rows_fingerprint(model, _physical_rows(model)),
    )


def _scaled_h(trace: DrivenTrace, g: np.ndarray, allowance_map=None):
    _rows, h, offset_size = _reduced(trace.rows, g, trace.transformation, trace.offset)
    safe = np.where(trace.row_scale > 0.0, trace.row_scale, 1.0)
    uncertainty = _uncertainty(trace.rows, g, offset_size, allowance_map)
    return h / safe, uncertainty / safe, uncertainty


def _emitted(trace: DrivenTrace, h_scaled: np.ndarray):
    prescribed: Dict[int, float] = {}
    equations: List[Tuple[Tuple[Tuple[int, float], ...], float, int]] = []
    for row, combo, pivot in zip(trace.echelon, trace.combinations, trace.pivots):
        rhs = float(sum(coef * h_scaled[i] for i, coef in combo.items()))
        terms = tuple((int(trace.independent_dofs[col]), float(value)) for col, value in sorted(row.items()))
        dependent = int(trace.independent_dofs[pivot])
        if len(terms) == 1:
            prescribed[dependent] = rhs / terms[0][1]
        else:
            equations.append((terms, rhs, dependent))
    for value in list(prescribed.values()) + [rhs for _t, rhs, _d in equations]:
        if not np.isfinite(value):
            raise SubmodelTransferError("TRACE_NONFINITE", "emitted trace value is not finite")
    return prescribed, equations


def _check_redundant(trace: DrivenTrace, h_scaled: np.ndarray, unc_scaled: np.ndarray) -> None:
    for index in trace.redundant:
        combo = trace.redundant_combinations[index]
        remaining = float(sum(coef * h_scaled[i] for i, coef in combo.items()))
        tol = _combination_tolerance(combo, h_scaled, unc_scaled)
        if not np.isfinite(remaining) or abs(remaining) > tol:
            row = trace.rows[index]
            raise SubmodelTransferError(
                "TRACE_INCONSISTENT",
                f"row {row.label} of node {row.node} contradicts the trace (remaining {remaining:.3e}, tolerance {tol:.3e})",
            )


def _require_bound(model: Any, trace: DrivenTrace, *, applied: bool) -> None:
    """The trace must belong to ``model``, whose retained constraints and trace
    constraints must be exactly as recorded."""

    if id(model) != trace.model_token:
        raise SubmodelTransferError("TRACE_MODEL_MISMATCH", "this trace was built for a different model")
    if not applied:
        if trace.applied:
            raise SubmodelTransferError("TRACE_ALREADY_APPLIED", "this trace is already applied")
        rows = _physical_rows(model)
    else:
        if not trace.applied:
            raise SubmodelTransferError("TRACE_NOT_APPLIED", "the trace has not been applied to this model")
        supports = trace.applied["supports"]
        present = {id(bc) for bc in model.boundary_conditions}
        if any(id(bc) not in present for bc in supports.values()):
            raise SubmodelTransferError("TRACE_MODEL_CHANGED", "trace supports are no longer in the model")
        for node, bc in supports.items():
            if bc.node_ids != [node] or set(bc.dof_constraints) != set(trace.applied["support_keys"][node]):
                raise SubmodelTransferError("TRACE_MODEL_CHANGED", f"trace support of node {node} was modified")
        positions = []
        for (position, origin), (terms, dependent) in zip(trace.applied["equations"], trace.applied["equation_structure"]):
            if position >= len(model.constraint_equations):
                raise SubmodelTransferError("TRACE_MODEL_CHANGED", "trace equations are no longer where they were applied")
            equation = model.constraint_equations[position]
            current = tuple((int(d), float(c)) for d, c in equation.coefficients)
            if equation.origin != origin or int(equation.dependent_dof) != dependent or sorted(current) != sorted(terms):
                raise SubmodelTransferError("TRACE_MODEL_CHANGED", f"trace equation {origin} was modified")
            positions.append(position)
        rows = _physical_rows(model, exclude_supports=supports.values(), exclude_equations=positions)
    if _rows_fingerprint(model, rows) != trace.retained_fingerprint:
        raise SubmodelTransferError("TRACE_MODEL_CHANGED", "the model's retained constraints or topology changed since the trace was built")


def apply_driven_trace(model: Any, trace: DrivenTrace, *, name: str = "submodel_trace") -> DrivenTrace:
    """Add the trace to ``model`` as supports and constraint equations."""

    from .boundary import BoundaryCondition

    from .constraint_audit import COEFFICIENT_ATOL

    _require_bound(model, trace, applied=False)
    h_scaled, _unc_fresh, _u = _scaled_h(trace, trace.values)
    unc_scaled = trace.uncertainty / np.where(trace.row_scale > 0.0, trace.row_scale, 1.0)
    prescribed, equations = _emitted(trace, h_scaled)
    for terms, _rhs, _dependent in equations:
        if any(abs(c) <= COEFFICIENT_ATOL for _d, c in terms):
            raise SubmodelTransferError(
                "TRACE_AUDIT_PRUNING",
                f"an emitted trace coefficient is at or below the audit threshold {COEFFICIENT_ATOL:g} and would be dropped",
            )
    dm = model.mesh.dof_manager
    by_node: Dict[int, Dict[str, float]] = {}
    for dof, value in prescribed.items():
        node, local, _name = dm.get_dof_info(dof)
        by_node.setdefault(int(node), {})[DOF_NAMES[int(local)]] = float(value)
    # Transaction: every unsuccessful exit restores the model's constraint lists and
    # the DOF manager's constrained set, then re-raises the original error.
    saved_supports = list(model.boundary_conditions)
    saved_equations = list(model.constraint_equations)
    saved_constrained = set(getattr(dm, "_constrained_dofs", set()))
    supports = {}
    placed, structure = [], []
    try:
        for node, constraints in sorted(by_node.items()):
            bc = BoundaryCondition(name=f"{name}:{node}", node_ids=[node], dof_constraints=dict(constraints))
            model.add_boundary_condition(bc)
            supports[node] = bc
        for k, (terms, rhs, dependent) in enumerate(equations):
            origin = f"{name}:eq{k}"
            model.add_constraint_equation(terms=terms, rhs=rhs, source_id=origin, dependent_dof=dependent)
            added = model.constraint_equations[-1]
            placed.append((len(model.constraint_equations) - 1, added.origin))
            structure.append((tuple((int(d), float(c)) for d, c in added.coefficients), int(added.dependent_dof)))
        _verify_applied(model, trace, h_scaled, unc_scaled)
    except BaseException:
        model.boundary_conditions[:] = saved_supports
        model.constraint_equations[:] = saved_equations
        if hasattr(dm, "_constrained_dofs"):
            dm._constrained_dofs = saved_constrained
        model.bump_revision("boundary")
        raise
    trace.applied = {
        "supports": supports, "support_keys": {n: tuple(bc.dof_constraints) for n, bc in supports.items()},
        "equations": placed, "equation_structure": structure, "name": name,
    }
    return trace


def _verify_applied(model: Any, trace: DrivenTrace, h_scaled: np.ndarray, unc_scaled: np.ndarray) -> None:
    """Every trace row must be enforced by the constraint map the solver actually builds."""

    t2, c2, _independent = _constraint_map(model)
    safe = np.where(trace.row_scale > 0.0, trace.row_scale, 1.0)
    for i, row in enumerate(trace.rows):
        coefficient: Dict[int, float] = {}
        magnitude: Dict[int, float] = {}
        value = 0.0
        value_size = 0.0
        for dof, coef in row.b:
            start, end = t2.indptr[dof], t2.indptr[dof + 1]
            for col, entry in zip(t2.indices[start:end], t2.data[start:end]):
                coefficient[int(col)] = coefficient.get(int(col), 0.0) + coef * float(entry)
                magnitude[int(col)] = magnitude.get(int(col), 0.0) + abs(coef * float(entry))
            value += coef * float(c2[dof])
            value_size += abs(coef * float(c2[dof]))
        # A row is enforced only when every free coordinate cancels to roundoff of
        # its own operands; any surviving dependence refuses.
        if any(abs(v) > 64.0 * _EPS * magnitude[col] for col, v in coefficient.items()):
            raise SubmodelTransferError(
                "TRACE_AUDIT_CHANGED", f"row {row.label} of node {row.node} is not fully enforced by the applied constraints"
            )
        if i in trace.redundant_combinations:
            tol = _combination_tolerance(trace.redundant_combinations[i], h_scaled, unc_scaled) * safe[i]
        else:
            tol = trace.uncertainty[i]
        tol += 64.0 * _EPS * (value_size + abs(trace.values[i]))
        if abs(value - trace.values[i]) > tol:
            raise SubmodelTransferError(
                "TRACE_AUDIT_CHANGED",
                f"row {row.label} of node {row.node} is enforced at {value:.6e} instead of {trace.values[i]:.6e}",
            )


def update_driven_trace_values(
    model: Any,
    trace: DrivenTrace,
    values: Mapping[int, Sequence[float]],
    *,
    allowance: Optional[Mapping[int, Sequence[float]]] = None,
) -> DrivenTrace:
    """Replace only the driven values of an applied trace, atomically.

    Everything is validated and computed before the model is touched; a refused
    update leaves supports, equations and the trace unchanged.  The constraint
    structure is unchanged, so a session refreshes the prescribed offset without
    refactorizing.
    """

    from .constraint_audit import ConstraintEquation

    _require_bound(model, trace, applied=True)
    nodes = sorted({row.node for row in trace.rows})
    checked = _validated_values(nodes, values)
    allowance_map = _validated_allowance(allowance)
    g = np.array([_row_value(row, checked[row.node]) for row in trace.rows])
    h_scaled, unc_scaled, uncertainty = _scaled_h(trace, g, allowance_map)
    _check_redundant(trace, h_scaled, unc_scaled)
    prescribed, equations = _emitted(trace, h_scaled)
    dm = model.mesh.dof_manager
    support_updates = []
    for dof, value in prescribed.items():
        node, local, _name = dm.get_dof_info(dof)
        support_updates.append((trace.applied["supports"][int(node)], DOF_NAMES[int(local)], float(value)))
    equation_updates = []
    for (position, origin), (terms, rhs, dependent) in zip(trace.applied["equations"], equations):
        equation_updates.append((position, ConstraintEquation(terms=terms, rhs=rhs, source_id=origin, dependent_dof=dependent)))
    # Commit: direct public-object edits without a revision bump (values only).
    for bc, name, value in support_updates:
        bc.dof_constraints[name] = value
    for position, equation in equation_updates:
        model.constraint_equations[position] = equation
    trace.values = g
    trace.uncertainty = uncertainty
    return trace


# ----------------------------------------------------------------------
# interface forces
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class InterfaceForces:
    """Accepted work-conjugate interface forces of an applied trace (global frame).

    ``node_forces``: force and moment the removed structure exerts on the target
    model at each driven node.  ``retained_support_reactions``: external reactions
    of the target model's own retained restraints (supports, and model equations
    or couplings that act on rigid motion), for a free-body balance.  Constraints
    that every rigid motion satisfies are internal; their self-equilibrated
    forces are not reported.  Physical and numerical
    (stabilisation) parts are not partitioned (``partitioned`` is False).
    """

    node_forces: Dict[int, np.ndarray]
    retained_support_reactions: Dict[int, np.ndarray]
    multipliers: np.ndarray
    equilibrium_residual: float
    equilibrium_ratio: float
    trace_residual_ratio: float
    partitioned: bool = False


def _assembled(model: Any, load_case: Any):
    from .analysis_session import AnalysisSession
    from .matrix_assembly import assemble_load_vector

    with AnalysisSession(model) as session:
        stiffness = sparse.csr_matrix(session.stiffness_plan().matrix)
    load, _info = assemble_load_vector(model, load_case)
    return stiffness, _finite(load, "FORCE_NONFINITE", "load vector").reshape(-1)


def _check_reconstruction(model: Any, u: np.ndarray, session: Any = None) -> np.ndarray:
    """Component-wise check that ``u`` satisfies the model's current constraints.

    Returns the accepted per-DOF discrepancy (roundoff level), which callers add
    to their allowance.
    """

    from .analysis_session import AnalysisSession

    owned = session is None
    active = AnalysisSession(model) if owned else session
    try:
        active._require_model(model)
        plan = active.constraint_plan()
        t = sparse.csr_matrix(plan.T)
        independent = np.asarray(plan.independent_dofs, dtype=int)
        u0 = np.asarray(plan.u0, dtype=float)
    finally:
        if owned:
            active.close()
    q = u[independent]
    rebuilt = t @ q + u0
    discrepancy = np.abs(rebuilt - u)
    nnz = np.diff(t.indptr).astype(float)
    allowed = 8.0 * _EPS * (np.abs(u0) + (abs(t) @ np.abs(q)) + np.abs(u)) * np.maximum(nnz, 1.0)
    if np.any(discrepancy > allowed):
        worst = int(np.argmax(discrepancy - allowed))
        raise SubmodelTransferError(
            "ALLOWANCE_RECONSTRUCTION",
            f"displacements violate the model's constraints at DOF {worst} by {discrepancy[worst]:.3e}",
        )
    return discrepancy


def interface_forces(
    model: Any,
    displacements: Any,
    load_case: Any,
    trace: DrivenTrace,
    *,
    displacement_allowance: Optional[Any] = None,
) -> InterfaceForces:
    """Recover and accept the interface forces and retained reactions (contract section 5).

    All multipliers are solved together from ``K u - f = C^T mu`` over the
    physical constraint rows ``C``: retained supports, model equations, element
    couplings and the kept trace rows.  Support multipliers are the retained
    reactions; trace multipliers give the interface forces; equation and coupling
    multipliers are internal.  The result is refused unless ``u`` satisfies the
    current constraints, every trace row holds within its allowance (redundant
    rows within their combination tolerance), the constraint rows have full rank,
    and the residual not carried by any constraint lies within the force
    allowance.  Without ``displacement_allowance``, ``u`` must be a certified solve.
    """

    _require_bound(model, trace, applied=True)
    u = _vector(model, displacements, "FORCE_INPUT")
    discrepancy = _check_reconstruction(model, u)
    if displacement_allowance is None:
        estimate = estimate_static_error_allowance(model, u, load_case)
        if estimate.inconclusive or estimate.eta_scaled > _CERTIFIED_BACKWARD_ERROR:
            raise SubmodelTransferError(
                "FORCE_UNCERTIFIED",
                f"displacements are not a certified solve (scaled backward error {estimate.eta_scaled:.3e}, "
                f"kappa*eta {estimate.kappa_scaled * estimate.eta_scaled:.3e})",
            )
        displacement_allowance = estimate.displacement
    du = _finite(displacement_allowance, "ALLOWANCE_INVALID", "displacement allowance").reshape(-1)
    if du.shape != u.shape or np.any(du < 0.0):
        raise SubmodelTransferError("ALLOWANCE_INVALID", "displacement allowance must match the DOFs and be non-negative")
    du = du + discrepancy
    stiffness, load = _assembled(model, load_case)
    residual = stiffness @ u - load
    floor = force_allowance(model, u, load_case, du, assembled=(stiffness, load))

    # Trace rows: kept rows are enforced exactly; redundant rows within their combination tolerance.
    h_scaled, _unc_fresh, _unc = _scaled_h(trace, trace.values)
    safe = np.where(trace.row_scale > 0.0, trace.row_scale, 1.0)
    unc_scaled = trace.uncertainty / safe  # the same uncertainty the build and updates used
    trace_ratio = 0.0
    for i, row in enumerate(trace.rows):
        value = sum(coef * u[dof] for dof, coef in row.b)
        solve_part = sum(abs(coef) * du[dof] for dof, coef in row.b) + 8.0 * _EPS * sum(abs(coef * u[dof]) for dof, coef in row.b)
        if i in trace.redundant_combinations:
            base = _combination_tolerance(trace.redundant_combinations[i], h_scaled, unc_scaled) * safe[i]
        else:
            base = trace.uncertainty[i]
        trace_ratio = max(trace_ratio, abs(value - trace.values[i]) / (base + solve_part + 1.0e-300))
    if not np.isfinite(trace_ratio) or trace_ratio > 1.0:
        raise SubmodelTransferError("FORCE_TRACE_VIOLATED", f"displacements violate the trace (ratio {trace_ratio:.3e})")

    retained = _physical_rows(
        model, exclude_supports=trace.applied["supports"].values(),
        exclude_equations=[p for p, _o in trace.applied["equations"]],
    )
    # Retained supports are unit rows: their reactions follow as remainders.  The
    # other multipliers (model equations, element couplings, kept trace rows) are
    # solved from the DOFs no retained support fixes, per connected component of
    # rows sharing such DOFs.
    supported = set()
    other_rows: List[Dict[int, float]] = []
    external_rows: List[bool] = []
    rigid = _rigid_modes(model)
    for kind, _origin, terms, _value in retained:
        if kind == "support":
            supported.add(next(iter(terms)))
        else:
            other_rows.append(terms)
            external_rows.append(_is_external(terms, rigid))
    trace_start = len(other_rows)
    for i in trace.kept:
        other_rows.append({int(d): float(c) for d, c in trace.rows[i].b})
    for k, row in enumerate(other_rows):
        if not any(dof not in supported for dof in row):
            raise SubmodelTransferError(
                "FORCE_RANK", "a constraint acts only on supported DOFs; its force cannot be separated from the reactions"
            )
    parent = list(range(len(other_rows)))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    owner: Dict[int, int] = {}
    for k, row in enumerate(other_rows):
        for dof in row:
            if dof in supported:
                continue
            if dof in owner:
                parent[find(k)] = find(owner[dof])
            else:
                owner[dof] = k
    components: Dict[int, List[int]] = {}
    for k in range(len(other_rows)):
        components.setdefault(find(k), []).append(k)
    mu = np.zeros(len(other_rows))
    for members in components.values():
        dofs = sorted({dof for k in members for dof in other_rows[k] if dof not in supported})
        local = {dof: j for j, dof in enumerate(dofs)}
        block = np.zeros((len(dofs), len(members)))
        for column, k in enumerate(members):
            for dof, coef in other_rows[k].items():
                if dof in local:
                    block[local[dof], column] += coef
        norms = np.linalg.norm(block, axis=0)
        if np.any(norms == 0.0):
            raise SubmodelTransferError("FORCE_RANK", "a constraint has no unsupported DOF")
        solution, _res, rank, _sv = np.linalg.lstsq(block / norms, residual[dofs], rcond=None)
        if rank != len(members):
            raise SubmodelTransferError(
                "FORCE_RANK", f"{len(members)} coupled constraints have rank {rank} (dependent constraints)"
            )
        mu[members] = solution / norms
    carried = np.zeros_like(residual)
    for k, row in enumerate(other_rows):
        for dof, coef in row.items():
            carried[dof] += coef * mu[k]
    unsupported = np.ones(residual.shape, dtype=bool)
    unsupported[list(supported)] = False
    leftover = np.where(unsupported, np.abs(residual - carried), 0.0)
    ratio = float(np.max(leftover / (floor + 1.0e-300), initial=0.0))
    if not np.isfinite(ratio) or ratio > 1.0:
        raise SubmodelTransferError(
            "FORCE_EQUILIBRIUM",
            f"residual not carried by the constraints exceeds the force allowance (ratio {ratio:.3e})",
        )
    dm = model.mesh.dof_manager
    forces: Dict[int, np.ndarray] = {row.node: np.zeros(6) for row in trace.rows}
    for k, i in enumerate(trace.kept):
        for dof, coefficient in trace.rows[i].b:
            node, local, _ = dm.get_dof_info(dof)
            forces[int(node)][int(local)] += coefficient * float(mu[trace_start + k])
    reactions: Dict[int, np.ndarray] = {}
    for dof in sorted(supported):
        node, local, _ = dm.get_dof_info(int(dof))
        reactions.setdefault(int(node), np.zeros(6))[int(local)] += float(residual[dof] - carried[dof])
    # Retained equations and couplings that act on rigid motion are external restraints.
    omitted = np.zeros(6)
    omitted_bound = np.zeros(6)
    for k, external in enumerate(external_rows):
        for dof, coef in other_rows[k].items():
            node, local, _ = dm.get_dof_info(int(dof))
            force = coef * float(mu[k])
            if external:
                reactions.setdefault(int(node), np.zeros(6))[int(local)] += force
            else:
                mode = rigid.get(int(dof), np.zeros(6))
                omitted += force * mode
                omitted_bound += np.abs(force * mode)
    # The forces of internal constraints must have a zero resultant (to roundoff).
    if np.any(np.abs(omitted) > 64.0 * _EPS * max(len(other_rows), 1) * omitted_bound + 1.0e-300):
        raise SubmodelTransferError(
            "FORCE_INTERNAL_RESULTANT", "forces of constraints classified internal do not cancel"
        )
    return InterfaceForces(forces, reactions, np.asarray(mu[trace_start:], dtype=float),
                           float(np.max(leftover, initial=0.0)), ratio, float(trace_ratio))


def _rigid_modes(model: Any) -> Dict[int, np.ndarray]:
    """Rigid-body modes per DOF: rows (6,) = translations x, y, z and rotations about
    x, y, z through the model centroid."""

    dm = model.mesh.dof_manager
    nodes = model.mesh.nodes
    centroid = np.mean([node.coords() for node in nodes.values()], axis=0) if nodes else np.zeros(3)
    modes: Dict[int, np.ndarray] = {}
    for node_id, node in nodes.items():
        x = np.asarray(node.coords(), dtype=float) - centroid
        dofs = dm.get_node_dofs(node_id)
        for local, dof in enumerate(dofs):
            row = np.zeros(6)
            if local < 3:
                row[local] = 1.0
                for axis in range(3):
                    a = np.zeros(3)
                    a[axis] = 1.0
                    row[3 + axis] = np.cross(a, x)[local]
            else:
                row[local] = 1.0
            modes[int(dof)] = row
    return modes


def _is_external(terms: Mapping[int, float], rigid: Mapping[int, np.ndarray]) -> bool:
    """A constraint row is internal only when every rigid mode satisfies it to the
    roundoff of that mode's own products; then its reaction is self-equilibrated.
    Anything else, including undecidable cases, is treated as an external restraint
    (conservative: reporting a self-equilibrated force as external changes no
    resultant)."""

    action = np.zeros(6)
    bound = np.zeros(6)
    for dof, coef in terms.items():
        mode = rigid.get(int(dof), np.zeros(6))
        action += coef * mode
        bound += np.abs(coef * mode)
    return bool(np.any(np.abs(action) > 16.0 * _EPS * len(terms) * bound))


def interface_resultant(model: Any, forces: Mapping[int, Sequence[float]], about: Sequence[float]) -> np.ndarray:
    """Force and moment resultant of nodal six-vectors about a point (global frame)."""

    point = _finite(about, "FORCE_INPUT", "reference point").reshape(3)
    total = np.zeros(6)
    for node, six in forces.items():
        f = _six(six, "FORCE_INPUT", f"force of node {node}")
        arm = np.asarray(model.mesh.get_node(int(node)).coords(), dtype=float) - point
        total[:3] += f[:3]
        total[3:] += f[3:] + np.cross(arm, f[:3])
    return total


# ----------------------------------------------------------------------
# error allowance
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StaticErrorAllowance:
    """First-order estimated allowance of a linear static solve (not a bound)."""

    eta_scaled: float
    kappa_scaled: float
    displacement: np.ndarray  # per global DOF, metres or radians
    inconclusive: bool
    threshold: float


def estimate_static_error_allowance(
    model: Any,
    displacements: Any,
    load_case: Any,
    *,
    inconclusive_threshold: float = 1.0e-6,
    session: Any = None,
) -> StaticErrorAllowance:
    """Allowance on the solver's own equilibrated reduced operator (contract section 6).

    The backward error and the condition estimate are both computed for the
    equilibrated matrix the solver factorizes; the inverse norm is estimated
    through that factorization.  ``u`` must satisfy the model's constraints
    component-wise; the accepted roundoff discrepancy is added to the allowance.
    """

    from .analysis_session import AnalysisSession
    from .matrix_assembly import assemble_load_vector

    threshold = _control(inconclusive_threshold, "inconclusive_threshold", positive=True)
    u = _vector(model, displacements, "ALLOWANCE_INPUT")
    load, _info = assemble_load_vector(model, load_case)
    load = _finite(load, "ALLOWANCE_NONFINITE", "load vector").reshape(-1)
    owned = session is None
    active = AnalysisSession(model) if owned else session
    try:
        active._require_model(model)
        discrepancy = _check_reconstruction(model, u, active)
        plan = active.static_operator_plan()
        if plan.augmented:
            raise SubmodelTransferError("ALLOWANCE_UNSUPPORTED", "nullspace-augmented (free-free) operators are not supported")
        constraint = plan.constraint
        t = sparse.csr_matrix(constraint.T)
        independent = np.asarray(constraint.independent_dofs, dtype=int)
        if independent.size == 0:
            return StaticErrorAllowance(0.0, 1.0, discrepancy, False, threshold)
        stiffness = sparse.csr_matrix(active.stiffness_plan().matrix)
        q = u[independent]
        scaled = sparse.csr_matrix(plan.matrix)
        norm = float(abs(scaled).sum(axis=1).max())
        if float(abs(scaled - scaled.T).max()) > 1.0e-12 * norm:
            raise SubmodelTransferError("ALLOWANCE_UNSUPPORTED", "the equilibrated operator is not symmetric")
        scale = np.asarray(plan.scaling, dtype=float)
        reduced_load = constraint.reduce_load(stiffness, load)
        y = q / scale
        residual = scaled @ y - scale * reduced_load
        denominator = norm * float(np.max(np.abs(y))) + float(np.max(np.abs(scale * reduced_load)))
        eta = float(np.max(np.abs(residual)) / denominator) if denominator > 0.0 else 0.0
        handle = plan.handle
        if handle is None or handle.status != "ok":
            raise SubmodelTransferError("ALLOWANCE_FACTORIZATION", "the solver's factorization is unavailable")
        n = scaled.shape[0]
        inverse = LinearOperator((n, n), matvec=handle.solve, rmatvec=handle.solve, dtype=float)
        kappa = norm * float(onenormest(inverse))
    finally:
        if owned:
            active.close()
    dq = scale * kappa * eta * float(np.max(np.abs(y)))
    du = np.asarray(abs(t) @ dq, dtype=float).reshape(-1) + discrepancy
    if not (np.isfinite(eta) and np.isfinite(kappa) and np.all(np.isfinite(du))):
        raise SubmodelTransferError("ALLOWANCE_NONFINITE", "the error allowance is not finite")
    return StaticErrorAllowance(eta, kappa, du, bool(kappa * eta > threshold), threshold)


def force_allowance(
    model: Any,
    displacements: Any,
    load_case: Any,
    displacement_allowance: Any,
    *,
    assembled: Optional[Tuple[sparse.csr_matrix, np.ndarray]] = None,
) -> np.ndarray:
    """Per-DOF allowance of ``K u - f``: propagated error plus a matvec roundoff floor."""

    u = _vector(model, displacements, "FORCE_INPUT")
    du = _finite(displacement_allowance, "ALLOWANCE_INVALID", "displacement allowance").reshape(-1)
    if du.shape != u.shape or np.any(du < 0.0):
        raise SubmodelTransferError("ALLOWANCE_INVALID", "displacement allowance must match the DOFs and be non-negative")
    stiffness, load = _assembled(model, load_case) if assembled is None else assembled
    absolute = abs(stiffness)
    nnz = np.diff(stiffness.indptr).astype(float)
    gamma = nnz * _EPS / (1.0 - nnz * _EPS)
    out = np.asarray(absolute @ du + gamma * (absolute @ np.abs(u)) + _EPS * np.abs(load), dtype=float).reshape(-1)
    if not np.all(np.isfinite(out)):
        raise SubmodelTransferError("ALLOWANCE_NONFINITE", "the force allowance is not finite")
    return out
