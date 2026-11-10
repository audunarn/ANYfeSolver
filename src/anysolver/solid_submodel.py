"""One-way connected linear shell -> solid submodel mechanics.

A converged linear shell solution drives the cut boundary of a prepared solid
region (ANYsolid ``PreparedSolid``).  The solid is solved once, with every
coefficient on the cut groups prescribed from the shell field, and the result is
reported with its interface resultants and certification diagnostics.  Nothing is
fed back to the shell (no two-way coupling), and no regional mesher is imported:
the host supplies the correspondence through ``locate``.

Mechanics (see ``docs/SOLID_SUBMODEL.md`` for the limits):

* Source admission is the exact-class allow-list of ``submodel_transfer`` (the
  qualified Q4 and the S3 V2D triangle), planar elements only.
* At a target world point ``x`` with reference point ``p = sum_i N_i X_i`` the
  physical 3-component displacement is ``u(x) = u_ref + theta_bar x (x - p)``,
  where ``u_ref = sum_i N_i u_i`` and ``theta_bar = sum_i N_i theta_i`` are the
  interpolated translation and rotation.  The whole displacement is transported
  from ``p`` to ``x`` once; nodal rotations are never transported against
  per-node lever arms (that would displace the midsurface under varying rotations).
* For the V2D triangle the component of ``u_ref`` along the reference normal is
  replaced by the full interior MIN3 transverse displacement ``w`` taken from the
  element's own dead-pressure operator (``compute_native_pressure_density``), then
  the result is transported once as above.  Native node permutation and the
  physical director conventions stay with the element.
* Every DOF of every cut group (vertices and P2 edge midpoints) is prescribed,
  shared DOFs once.  The constrained linear static problem is solved on the free
  block with a symmetric Jacobi scaling, one refinement step, a backward-error and
  a condition certificate, and an equilibrium check.

Failures raise :class:`SolidSubmodelError` (or :class:`SolidSubmodelCancelled`)
with a stable ``code``; nothing is returned for a refused or cancelled solve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu

from . import submodel_transfer as _transfer

__all__ = [
    "SolidSubmodelError",
    "SolidSubmodelCancelled",
    "ShellDisplacement",
    "GroupResultant",
    "SolidSubmodelResult",
    "ATTRIBUTION_RULE",
    "evaluate_shell_displacement",
    "solve_shell_driven_solid",
]

_EPS = float(np.finfo(float).eps)
_TINY = float(np.finfo(float).tiny)
#: Natural-coordinate overshoot tolerated from the host's inverse mapping.
_NATURAL_TOL = 1.0e-8
#: Planarity / degeneracy tolerances of a source element (relative to its size).
_PLANAR_TOL = 1.0e-8
_AREA_TOL = 1.0e-9
#: Tangential target-to-located-point residual: fraction of the element length plus roundoff
#: (``_TANGENTIAL_EPS_FACTOR`` machine epsilons of the coordinate magnitude).
_TANGENTIAL_RTOL = 1.0e-7
_TANGENTIAL_EPS_FACTOR = 64.0
#: Alternative hits must agree to this fraction of the source field scale.
_HIT_AGREE_RTOL = 1.0e-8
#: Source constraint rows must hold to this fraction of their operands.
_SOURCE_CONSTRAINT_RTOL = 1.0e-9
#: V2D density must reproduce the element's own shape functions on its translations.
_DENSITY_TOL = 1.0e-10
#: Same backward-error limit as the linear static path / submodel_transfer.
_BACKWARD_ERROR_LIMIT = 1.0e-10
_CONDITION_LIMIT = 1.0e12
_ERROR_BOUND_LIMIT = 1.0e-6
_RIGID_RANK_RATIO = 1.0e-8
_BALANCE_LIMIT = 1.0e-6
_SYMMETRY_RTOL = 1.0e-10
ATTRIBUTION_RULE = "equal_split_among_groups_sharing_the_dof"


class SolidSubmodelError(ValueError):
    """A typed, fail-closed refusal of the shell->solid submodel."""

    def __init__(self, code: str, message: str, *, details: Optional[Mapping[str, Any]] = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = dict(details or {})


class SolidSubmodelCancelled(SolidSubmodelError):
    """The caller cancelled the operation; no result exists."""

    def __init__(self, where: str = ""):
        super().__init__("CANCELLED", f"cancelled{' at ' + where if where else ''}", details={"where": where})


def _error(code: str, message: str, **details: Any) -> SolidSubmodelError:
    return SolidSubmodelError(code, message, details=details)


# ----------------------------------------------------------------------
# cancellation
# ----------------------------------------------------------------------
def _cancel_probe(cancel: Any) -> Optional[Callable[[], bool]]:
    if cancel is None:
        return None
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        return lambda: bool(is_set())
    if callable(cancel):
        return lambda: bool(cancel())
    raise _error("CANCEL_INVALID", "cancel must be None, a callable returning a flag, or an object with is_set()")


def _check_cancel(probe: Optional[Callable[[], bool]], where: str) -> None:
    if probe is not None and probe():
        raise SolidSubmodelCancelled(where)


# ----------------------------------------------------------------------
# shell source
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ShellDisplacement:
    """Physical displacement of the shell at one target point."""

    value: np.ndarray  # (3,) global frame, from the first hit
    hit_spread: float  # largest |difference| between agreeing hits
    hits: int
    transverse_correction: float  # V2D MIN3 normal correction (0 for the Q4)
    normal_offset: float  # (target - reference point) . reference normal of the first hit
    director_offset: float  # same, along the element's physical director
    element_ids: Tuple[int, ...]


@dataclass
class _ElementData:
    element: Any
    kind: str  # "q4" or "v2d"
    node_ids: Tuple[int, ...]
    X: np.ndarray  # (n, 3) nodal reference coordinates
    U: np.ndarray  # (n, 6) nodal translations + rotations
    normal: np.ndarray  # (3,) unit reference normal
    director: np.ndarray  # (3,) unit physical director
    length: float = 0.0  # largest node-to-node distance (admitted element length)
    coordinate_scale: float = 0.0  # largest |nodal coordinate| (roundoff scale)


class _Source:
    """Validated shell model + displacement vector, evaluated at physical points."""

    def __init__(self, model: Any, source_u: Any):
        mesh = getattr(model, "mesh", None)
        if (
            mesh is None
            or not isinstance(getattr(mesh, "elements", None), Mapping)
            or not isinstance(getattr(mesh, "nodes", None), Mapping)
            or getattr(mesh, "dof_manager", None) is None
            or not callable(getattr(mesh, "get_node", None))
        ):
            raise _error("SOURCE_INVALID", "fe_model does not expose mesh.elements / mesh.nodes / mesh.dof_manager")
        try:
            total = int(mesh.dof_manager.total_dofs)
        except (AttributeError, TypeError, ValueError) as error:
            raise _error("SOURCE_INVALID", "fe_model has no usable DOF manager") from error
        try:
            u = np.asarray(source_u, dtype=float)
        except (TypeError, ValueError) as error:
            raise _error("SOURCE_INVALID", "source displacement vector is not numeric") from error
        if u.shape != (total,):
            raise _error("SOURCE_INVALID", f"source displacement has shape {u.shape}, model has {total} DOFs")
        if not np.all(np.isfinite(u)):
            raise _error("SOURCE_INVALID", "source displacement vector is not finite")
        try:
            _transfer._require_target_shells(model)
        except _transfer.SubmodelTransferError as error:
            raise _error("SOURCE_ELEMENT_NOT_ADMITTED", str(error), **error.details) from error
        self.model = model
        self.mesh = mesh
        self.u = u.copy()
        self.q4_type, self.v2d_type = _transfer.admitted_transfer_element_types()
        self._cache: Dict[int, _ElementData] = {}
        coords = np.array([np.asarray(node.coords(), dtype=float) for node in mesh.nodes.values()]).reshape(-1, 3)
        self.extent = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))) if coords.size else 0.0
        translations, rotations = [], []
        for node_id in mesh.nodes:
            dofs = mesh.dof_manager.get_node_dofs(int(node_id))
            if len(dofs) == 6:
                six = u[list(dofs)]
                translations.append(np.max(np.abs(six[:3])))
                rotations.append(np.max(np.abs(six[3:])))
        self.scale = max(
            max(translations, default=0.0), max(rotations, default=0.0) * self.extent, 0.0
        )
        self._check_constraints()

    # -- constraint consistency ------------------------------------------------
    def _check_constraints(self) -> None:
        try:
            rows = _transfer._physical_rows(self.model)
        except Exception as error:  # constraint enumeration is owned by the model
            raise _error("SOURCE_CONSTRAINT_CHECK_FAILED", f"cannot enumerate model constraints: {error}") from error
        total = self.u.size
        worst = 0.0
        for kind, origin, terms, value in rows:
            if any(not (0 <= dof < total) for dof in terms):
                raise _error("SOURCE_CONSTRAINT_INCONSISTENT", f"{kind} {origin!r} references an unknown DOF")
            action = sum(coef * self.u[dof] for dof, coef in terms.items())
            operand = sum(abs(coef * self.u[dof]) for dof, coef in terms.items()) + abs(value)
            allowed = _SOURCE_CONSTRAINT_RTOL * (operand + self.scale) + 16.0 * _EPS * operand
            violation = abs(action - value)
            worst = max(worst, violation / (allowed + _TINY))
            if not np.isfinite(violation) or violation > allowed:
                raise _error(
                    "SOURCE_CONSTRAINT_INCONSISTENT",
                    f"source displacement violates {kind} {origin!r} by {violation:.3e} (allowed {allowed:.3e})",
                    kind=kind, origin=origin, violation=float(violation), allowed=float(allowed),
                )
        self.constraint_rows = len(rows)

    # -- element admission -----------------------------------------------------
    def element_data(self, element_id: int) -> _ElementData:
        cached = self._cache.get(element_id)
        if cached is not None:
            return cached
        element = self.mesh.elements.get(element_id)
        if element is None:
            raise _error("SOURCE_ELEMENT_MISSING", f"source element {element_id} is not in the model",
                         element_id=element_id)
        if type(element) is self.q4_type:
            kind = "q4"
        elif type(element) is self.v2d_type:
            kind = "v2d"
        else:
            raise _error("SOURCE_ELEMENT_NOT_ADMITTED",
                         f"element {element_id} is {type(element).__name__}; admitted shells: "
                         f"{self.q4_type.__name__}, {self.v2d_type.__name__}", element_id=element_id)
        node_ids = tuple(int(n) for n in element.node_ids)
        expected = 4 if kind == "q4" else 3
        if len(node_ids) != expected or len(set(node_ids)) != expected:
            raise _error("SOURCE_ELEMENT_INVALID", f"element {element_id} has malformed connectivity",
                         element_id=element_id)
        try:
            X = np.array([np.asarray(self.mesh.get_node(n).coords(), dtype=float) for n in node_ids])
        except (AttributeError, TypeError, ValueError) as error:
            raise _error("SOURCE_ELEMENT_INVALID", f"element {element_id} references a missing node") from error
        if X.shape != (expected, 3) or not np.all(np.isfinite(X)):
            raise _error("SOURCE_ELEMENT_INVALID", f"element {element_id} has invalid node coordinates",
                         element_id=element_id)
        U = np.empty((expected, 6))
        for row, n in enumerate(node_ids):
            dofs = list(self.mesh.dof_manager.get_node_dofs(n))
            if len(dofs) != 6:
                raise _error("SOURCE_ELEMENT_INVALID", f"node {n} of element {element_id} has no six DOFs",
                             element_id=element_id)
            U[row] = self.u[dofs]
        if kind == "q4":
            normal = _check_planar_quad(X, element_id)
            director = normal.copy()
        else:
            normal, director = self._check_triangle(element, element_id, X)
        length = float(max(np.linalg.norm(X[i] - X[j]) for i in range(expected) for j in range(i + 1, expected)))
        data = _ElementData(element, kind, node_ids, X, U, normal, director, length,
                            float(np.max(np.abs(X))))
        self._cache[element_id] = data
        return data

    def _check_triangle(self, element: Any, element_id: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = max(float(np.max(np.linalg.norm(X - X.mean(axis=0), axis=1))), _TINY)
        cross = np.cross(X[1] - X[0], X[2] - X[0])
        if float(np.linalg.norm(cross)) <= _AREA_TOL * scale * scale:
            raise _error("SOURCE_ELEMENT_INVALID", f"V2D element {element_id} is degenerate", element_id=element_id)
        try:
            normal = np.array(element.reference_normal, dtype=float).reshape(3)
            polarity = float(element.director_polarity)
        except (AttributeError, TypeError, ValueError) as error:
            raise _error("SOURCE_ELEMENT_INVALID", f"V2D element {element_id} has no reference normal / director",
                         element_id=element_id) from error
        if not np.all(np.isfinite(normal)) or abs(float(np.linalg.norm(normal)) - 1.0) > 1.0e-8:
            raise _error("SOURCE_ELEMENT_INVALID", f"V2D element {element_id} reference normal is not a unit vector",
                         element_id=element_id)
        if abs(abs(float(normal @ cross)) - float(np.linalg.norm(cross))) > 1.0e-8 * float(np.linalg.norm(cross)):
            raise _error("SOURCE_ELEMENT_INVALID", f"V2D element {element_id} is not planar with its reference normal",
                         element_id=element_id)
        if polarity not in (-1.0, 1.0):
            raise _error("SOURCE_ELEMENT_INVALID", f"V2D element {element_id} director polarity is not +-1",
                         element_id=element_id)
        return normal, polarity * normal

    # -- evaluation ------------------------------------------------------------
    def _weights(self, data: _ElementData, natural: np.ndarray, element_id: int) -> np.ndarray:
        r, s = float(natural[0]), float(natural[1])
        if data.kind == "q4":
            inside = abs(r) <= 1.0 + _NATURAL_TOL and abs(s) <= 1.0 + _NATURAL_TOL
        else:
            inside = r >= -_NATURAL_TOL and s >= -_NATURAL_TOL and r + s <= 1.0 + _NATURAL_TOL
        if not inside:
            raise _error("LOCATE_NATURAL_OUTSIDE",
                         f"natural coordinates {(r, s)} lie outside element {element_id}",
                         element_id=element_id, natural=(r, s))
        try:
            weights = np.asarray(_transfer.element_interpolation_weights(data.element, (r, s)), dtype=float).reshape(-1)
        except _transfer.SubmodelTransferError as error:
            raise _error("SOURCE_ELEMENT_INVALID", str(error), element_id=element_id) from error
        if (
            weights.shape != (len(data.node_ids),)
            or not np.all(np.isfinite(weights))
            or abs(float(weights.sum()) - 1.0) > 1.0e-12
        ):
            raise _error("SOURCE_ELEMENT_INVALID", f"element {element_id} returned malformed interpolation weights",
                         element_id=element_id)
        return weights

    def hit_value(self, element_id: int, natural: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Displacement at ``target`` through one hit: (value, correction, normal_offset, director_offset)."""

        data = self.element_data(element_id)
        weights = self._weights(data, natural, element_id)
        # Displacement of the reference point p = sum N_i X_i: interpolated translation.  The
        # rotation is interpolated too and transported ONCE, from p to the target, below.
        reference_point = weights @ data.X
        offset = target - reference_point
        # The target must lie on the element's reference normal line through p: only the signed
        # normal offset (any thickness) is admissible; the in-plane part is a locator error.  The
        # allowance is scaled by the admitted element length and by the coordinate magnitude
        # (roundoff), never by the distance to unrelated points.
        tangential = float(np.linalg.norm(offset - float(data.normal @ offset) * data.normal))
        coordinate_scale = max(data.coordinate_scale, float(np.max(np.abs(target))))
        allowed = _TANGENTIAL_RTOL * data.length + _TANGENTIAL_EPS_FACTOR * _EPS * coordinate_scale
        if not np.isfinite(tangential) or tangential > allowed:
            raise _error(
                "LOCATE_TANGENTIAL_MISMATCH",
                f"target {target.tolist()} is {tangential:.3e} from the normal line through the located point "
                f"of element {element_id} (allowed {allowed:.3e})",
                element_id=element_id, target=target.tolist(), natural=(float(natural[0]), float(natural[1])),
                reference_point=reference_point.tolist(), tangential_residual=tangential, allowed=float(allowed),
            )
        translation = weights @ data.U[:, :3]
        rotation = weights @ data.U[:, 3:]
        correction = 0.0
        if data.kind == "v2d":
            try:
                density, _area = data.element.compute_native_pressure_density(
                    self.mesh, np.array([[float(natural[0]), float(natural[1])]])
                )
            except Exception as error:  # element-owned geometry / scope guards
                raise _error("SOURCE_ELEMENT_INVALID",
                             f"V2D element {element_id} rejected its transverse operator: {error}",
                             element_id=element_id) from error
            density = np.asarray(density, dtype=float).reshape(-1)
            if density.shape != (18,) or not np.all(np.isfinite(density)):
                raise _error("SOURCE_ELEMENT_INVALID", f"V2D element {element_id} transverse operator is malformed",
                             element_id=element_id)
            translational = np.array([density[6 * k:6 * k + 3] for k in range(3)])
            if float(np.max(np.abs(translational - weights[:, None] * data.normal[None, :]))) > _DENSITY_TOL:
                raise _error("SOURCE_ELEMENT_INVALID",
                             f"V2D element {element_id} transverse operator disagrees with its node order / normal",
                             element_id=element_id)
            w_min3 = float(density @ data.U.reshape(-1))
            # Exact native normal displacement at the reference point replaces the normal
            # component of the interpolated translation (in-plane components untouched).
            correction = w_min3 - float(data.normal @ translation)
            translation = translation + correction * data.normal
        value = translation + np.cross(rotation, offset)
        if not (np.all(np.isfinite(value)) and np.isfinite(correction)):
            raise _error("SOURCE_NONFINITE", f"displacement through element {element_id} is not finite",
                         element_id=element_id)
        return value, correction, float(data.normal @ offset), float(data.director @ offset)

    def displacement(self, target: np.ndarray, hits: Sequence[Tuple[int, np.ndarray]]) -> ShellDisplacement:
        first: Optional[Tuple[np.ndarray, float, float, float]] = None
        spread = 0.0
        tolerance = _HIT_AGREE_RTOL * self.scale
        for element_id, natural in hits:
            result = self.hit_value(element_id, natural, target)
            if first is None:
                first = result
                continue
            difference = float(np.max(np.abs(result[0] - first[0])))
            spread = max(spread, difference)
            if difference > tolerance:
                raise _error(
                    "LOCATE_HITS_DISAGREE",
                    f"hits at {target.tolist()} disagree by {difference:.3e} (allowed {tolerance:.3e})",
                    target=target.tolist(), difference=difference, allowed=tolerance,
                )
        assert first is not None
        return ShellDisplacement(first[0].copy(), spread, len(hits), first[1], first[2], first[3],
                                 tuple(int(e) for e, _n in hits))


def _check_planar_quad(X: np.ndarray, element_id: int) -> np.ndarray:
    center = X.mean(axis=0)
    scale = max(float(np.max(np.linalg.norm(X - center, axis=1))), _TINY)
    normal = np.cross(X[2] - X[0], X[3] - X[1])
    area2 = float(np.linalg.norm(normal))
    if area2 <= _AREA_TOL * scale * scale:
        raise _error("SOURCE_ELEMENT_INVALID", f"Q4 element {element_id} is degenerate", element_id=element_id)
    normal = normal / area2
    if float(np.max(np.abs((X - center) @ normal))) > _PLANAR_TOL * scale:
        raise _error("SOURCE_ELEMENT_INVALID",
                     f"Q4 element {element_id} is warped; only planar facets are admitted", element_id=element_id)
    for i in range(4):
        turn = float(np.cross(X[(i + 1) % 4] - X[i], X[(i + 2) % 4] - X[(i + 1) % 4]) @ normal)
        if turn <= _AREA_TOL * scale * scale:
            raise _error("SOURCE_ELEMENT_INVALID",
                         f"Q4 element {element_id} is not strictly convex (corner {i})", element_id=element_id)
    return normal


def _hits_from_locate(locate: Callable[[np.ndarray], Iterable[Any]], target: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    try:
        raw = locate(target.copy())
    except SolidSubmodelError:
        raise
    except Exception as error:
        raise _error("LOCATE_FAILED", f"locate raised {type(error).__name__}: {error}",
                     target=target.tolist()) from error
    try:
        pairs = list(raw)
    except TypeError as error:
        raise _error("LOCATE_INVALID", "locate must return an iterable of (element_id, natural) pairs") from error
    if not pairs:
        raise _error("LOCATE_MISS", f"no source element found for target {target.tolist()}", target=target.tolist())
    hits: List[Tuple[int, np.ndarray]] = []
    for pair in pairs:
        try:
            element_id, natural = pair
        except (TypeError, ValueError) as error:
            raise _error("LOCATE_INVALID", "each locate hit must be an (element_id, natural) pair") from error
        if isinstance(element_id, (bool, np.bool_)) or not isinstance(element_id, (int, np.integer)):
            raise _error("LOCATE_INVALID", f"element id {element_id!r} is not an integer")
        try:
            coordinates = np.asarray(natural, dtype=float).reshape(-1)
        except (TypeError, ValueError) as error:
            raise _error("LOCATE_INVALID", "natural coordinates are not numeric") from error
        if coordinates.shape != (2,) or not np.all(np.isfinite(coordinates)):
            raise _error("LOCATE_INVALID", "natural coordinates must be two finite numbers")
        hits.append((int(element_id), coordinates))
    return hits


def evaluate_shell_displacement(
    fe_model: Any,
    source_u: Any,
    hits: Iterable[Tuple[int, Sequence[float]]],
    target: Sequence[float],
) -> ShellDisplacement:
    """Physical displacement of the shell at ``target`` through ``hits``.

    ``hits`` are ``(source_element_id, natural_coordinates)`` pairs; several hits
    must agree.  The model and source vector are validated exactly as for
    :func:`solve_shell_driven_solid`.
    """

    source = _Source(fe_model, source_u)
    try:
        x = np.asarray(target, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise _error("TARGET_INVALID", "target is not numeric") from error
    if x.shape != (3,) or not np.all(np.isfinite(x)):
        raise _error("TARGET_INVALID", "target must be a finite 3-vector")
    return source.displacement(x, _hits_from_locate(lambda _x: hits, x))


# ----------------------------------------------------------------------
# prepared solid validation
# ----------------------------------------------------------------------
@dataclass
class _Prepared:
    ndof: int
    coordinates: np.ndarray  # (ndof, 3)
    components: np.ndarray  # (ndof,)
    order: int


def _validate_prepared(prepared: Any) -> _Prepared:
    for name in ("assemble_linear", "boundary_dofs", "traction", "recover", "rigid_modes", "trace"):
        if not callable(getattr(prepared, name, None)):
            raise _error("PREPARED_INVALID", f"prepared solid lacks the {name}() interface")
    order = getattr(prepared, "order", None)
    if isinstance(order, (bool, np.bool_)) or order not in (1, 2):
        raise _error("PREPARED_INVALID", "prepared solid order must be 1 or 2")
    ndof = getattr(prepared, "ndof", None)
    if isinstance(ndof, (bool, np.bool_)) or not isinstance(ndof, (int, np.integer)) or ndof <= 0:
        raise _error("PREPARED_INVALID", "prepared solid ndof must be a positive integer")
    ndof = int(ndof)
    try:
        coordinates = np.asarray(prepared.coefficient_coordinates, dtype=float)
        components = np.asarray(prepared.coefficient_components)
    except (AttributeError, TypeError, ValueError) as error:
        raise _error("PREPARED_INVALID", "coefficient coordinates/components are unavailable") from error
    if coordinates.shape != (ndof, 3) or not np.all(np.isfinite(coordinates)):
        raise _error("PREPARED_INVALID", f"coefficient_coordinates must be finite ({ndof}, 3)")
    if components.shape != (ndof,) or components.dtype.kind not in "iu" or not np.all(np.isin(components, (0, 1, 2))):
        raise _error("PREPARED_INVALID", f"coefficient_components must be integers 0..2 of shape ({ndof},)")
    components = components.astype(np.int64)
    keyed = np.column_stack([coordinates, components])
    if len(np.unique(keyed, axis=0)) != ndof:
        raise _error("PREPARED_INVALID", "coefficient (coordinate, component) pairs are not unique")
    mesh = getattr(prepared, "mesh", None)
    try:
        node_ids = np.asarray(mesh.node_ids)
        node_coordinates = np.asarray(mesh.coordinates, dtype=float)
        groups = mesh.boundary_groups
        node_groups = mesh.node_groups
    except (AttributeError, TypeError, ValueError) as error:
        raise _error("PREPARED_INVALID", "prepared solid mesh lacks node_ids/coordinates/groups") from error
    if (
        node_ids.ndim != 1
        or node_coordinates.shape != (node_ids.size, 3)
        or not np.all(np.isfinite(node_coordinates))
        or not isinstance(groups, Mapping)
        or not isinstance(node_groups, Mapping)
    ):
        raise _error("PREPARED_INVALID", "prepared solid mesh arrays are malformed")
    return _Prepared(ndof, coordinates, components, int(order))


def _group_dofs(prepared: Any, info: _Prepared, name: str) -> np.ndarray:
    try:
        raw = prepared.boundary_dofs(name)
    except (ValueError, KeyError, TypeError) as error:
        raise _error("CUT_GROUP_INVALID", f"cut group {name!r} is not usable: {error}", group=name) from error
    dofs = np.asarray(raw)
    if dofs.ndim != 1 or dofs.dtype.kind not in "iu" or dofs.size == 0:
        raise _error("CUT_GROUP_INVALID", f"cut group {name!r} yields no DOF index array", group=name)
    dofs = dofs.astype(np.int64)
    if dofs.min() < 0 or dofs.max() >= info.ndof or len(np.unique(dofs)) != dofs.size:
        raise _error("CUT_GROUP_INVALID", f"cut group {name!r} has out-of-range or repeated DOFs", group=name)
    locations: Dict[bytes, int] = {}
    for d in dofs:
        key = info.coordinates[d].tobytes()
        locations[key] = locations.get(key, 0) + 1
    if any(count != 3 for count in locations.values()):
        raise _error("CUT_GROUP_INVALID",
                     f"cut group {name!r} does not carry all three components at every location", group=name)
    return np.sort(dofs)


def _resultant(coordinates: np.ndarray, components: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Work-conjugate force and moment about the origin of nodal values on (x, component) DOFs."""

    vectors = np.zeros((components.size, 3))
    vectors[np.arange(components.size), components] = values
    return vectors.sum(axis=0), np.cross(coordinates, vectors).sum(axis=0)


# ----------------------------------------------------------------------
# result
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class GroupResultant:
    """Resultant of the interface forces attributed to one cut group."""

    name: str
    force: np.ndarray  # (3,) global, force the shell boundary exerts on the solid
    moment: np.ndarray  # (3,) global, about the origin
    dof_count: int
    shared_dof_count: int


@dataclass(frozen=True)
class SolidSubmodelResult:
    """Result of a certified one-way shell->solid solve.

    ``interface_force`` / ``interface_moment`` are the force and moment (global
    frame, moment about the origin) that the driven cut boundary exerts on the
    solid: the constraint reactions ``r = K q - f`` on the prescribed DOFs.  With
    free skins they balance the skin load resultants (``skin_force``,
    ``skin_moment``): ``interface + skin = 0``.
    """

    coefficients: np.ndarray
    nodal_displacements: np.ndarray  # (N, 3), aligned to prepared.mesh.node_ids
    recovery: Any
    interface_force: np.ndarray
    interface_moment: np.ndarray
    group_resultants: Dict[str, GroupResultant]
    shared_attribution: Dict[int, Tuple[str, ...]]
    attribution_rule: str
    prescribed_dofs: np.ndarray
    prescribed_values: np.ndarray
    reactions: np.ndarray  # aligned with prescribed_dofs
    skin_force: np.ndarray
    skin_moment: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _nodal_displacements(prepared: Any, info: _Prepared, q: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(prepared.mesh.coordinates, dtype=float)
    lookup: Dict[bytes, Dict[int, int]] = {}
    for d in range(info.ndof):
        lookup.setdefault(info.coordinates[d].tobytes(), {})[int(info.components[d])] = d
    nodal = np.empty((coordinates.shape[0], 3))
    for row, x in enumerate(coordinates):
        entry = lookup.get(np.ascontiguousarray(x).tobytes())
        if entry is None or set(entry) != {0, 1, 2}:
            break
        nodal[row] = [q[entry[0]], q[entry[1]], q[entry[2]]]
    else:
        return nodal
    # Vertices are not coefficient locations bit-for-bit: interpolate through the trace operator.
    try:
        values = sparse.csr_matrix(prepared.trace(coordinates)) @ q
    except ValueError as error:
        raise _error("PREPARED_INVALID", f"nodal displacements cannot be evaluated: {error}") from error
    return np.asarray(values, dtype=float).reshape(-1, 3)


def _validated_cut_groups(cut_groups: Any) -> List[str]:
    if isinstance(cut_groups, (str, bytes)) or not isinstance(cut_groups, (list, tuple)):
        raise _error("CUT_GROUP_INVALID", "cut_groups must be a list or tuple of group names")
    names = list(cut_groups)
    if not names:
        raise _error("CUT_GROUP_INVALID", "at least one cut group is required")
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise _error("CUT_GROUP_INVALID", "cut group names must be distinct non-empty strings")
    return names


def _validated_skin(prepared: Any, info: _Prepared, skin_tractions: Any, cut_names: Sequence[str]) -> List[Tuple[str, np.ndarray]]:
    if skin_tractions is None:
        return []
    if not isinstance(skin_tractions, Mapping):
        raise _error("SKIN_INVALID", "skin_tractions must be a mapping group -> global traction vector")
    groups = prepared.mesh.boundary_groups
    out: List[Tuple[str, np.ndarray]] = []
    for name, vector in skin_tractions.items():
        if not isinstance(name, str) or name not in groups:
            raise _error("SKIN_INVALID", f"skin group {name!r} is not an external boundary group", group=name)
        if name in cut_names:
            raise _error("SKIN_INVALID", f"skin group {name!r} is also a cut group", group=name)
        try:
            traction = np.asarray(vector, dtype=float).reshape(-1)
        except (TypeError, ValueError) as error:
            raise _error("SKIN_INVALID", f"traction of {name!r} is not numeric", group=name) from error
        if traction.shape != (3,) or not np.all(np.isfinite(traction)):
            raise _error("SKIN_INVALID", f"traction of {name!r} must be a finite 3-vector", group=name)
        for cut in cut_names:
            if cut in groups and np.intersect1d(np.asarray(groups[name]), np.asarray(groups[cut])).size:
                raise _error("SKIN_INVALID", f"skin group {name!r} shares facets with cut group {cut!r}", group=name)
        out.append((name, traction))
    return out


# ----------------------------------------------------------------------
# solve
# ----------------------------------------------------------------------
def solve_shell_driven_solid(
    fe_model: Any,
    source_u: Any,
    prepared: Any,
    *,
    cut_groups: Sequence[str],
    locate: Callable[[np.ndarray], Iterable[Tuple[int, Sequence[float]]]],
    skin_tractions: Optional[Mapping[str, Sequence[float]]] = None,
    cancel: Any = None,
) -> SolidSubmodelResult:
    """Drive a prepared solid from a linear shell solution and solve it once.

    Parameters
    ----------
    fe_model, source_u:
        The converged linear shell model and its global displacement vector.  The
        vector must satisfy every support, equation and coupling row of the model.
    prepared:
        ANYsolid ``PreparedSolid`` (P1 or P2).
    cut_groups:
        Boundary (or node) group names whose coefficients are all prescribed.
    locate:
        ``locate(x)`` takes a world coordinate ``ndarray (3,)`` and returns an
        iterable of ``(source_element_id, natural_coordinates)`` pairs on the shell
        reference midsurface (the Q4 uses ``(xi, eta)`` in ``[-1, 1]^2``, the V2D
        ``(r, s)`` with ``N = (1-r-s, r, s)``).  Several hits must agree.
    skin_tractions:
        ``{external boundary group: global traction vector}``.
    cancel:
        ``None``, a callable returning a flag, or an object with ``is_set()``.
    """

    probe = _cancel_probe(cancel)
    _check_cancel(probe, "start")
    if not callable(locate):
        raise _error("LOCATE_INVALID", "locate must be callable")
    info = _validate_prepared(prepared)
    names = _validated_cut_groups(cut_groups)
    skin = _validated_skin(prepared, info, skin_tractions, names)
    source = _Source(fe_model, source_u)

    # -- cut DOFs --------------------------------------------------------------
    group_dofs: Dict[str, np.ndarray] = {}
    owners: Dict[int, List[str]] = {}
    for name in names:
        _check_cancel(probe, f"cut group {name}")
        dofs = _group_dofs(prepared, info, name)
        group_dofs[name] = dofs
        for d in dofs:
            owners.setdefault(int(d), []).append(name)
    prescribed = np.array(sorted(owners), dtype=np.int64)
    listed = sum(len(v) for v in group_dofs.values())
    duplicates = listed - prescribed.size

    # -- driven values ---------------------------------------------------------
    cache: Dict[bytes, ShellDisplacement] = {}
    values = np.empty(prescribed.size)
    spread_max = 0.0
    correction_max = 0.0
    offset_max = 0.0
    hits_max = 0
    element_counts: Dict[str, int] = {}
    for position, d in enumerate(prescribed):
        _check_cancel(probe, "evaluating the shell field")
        x = info.coordinates[d]
        key = x.tobytes()
        shell = cache.get(key)
        if shell is None:
            shell = source.displacement(x.copy(), _hits_from_locate(locate, x))
            cache[key] = shell
            spread_max = max(spread_max, shell.hit_spread)
            correction_max = max(correction_max, abs(shell.transverse_correction))
            offset_max = max(offset_max, abs(shell.normal_offset))
            hits_max = max(hits_max, shell.hits)
            for eid in shell.element_ids:
                kind = source.element_data(eid).kind
                element_counts[kind] = element_counts.get(kind, 0) + 1
        values[position] = shell.value[info.components[d]]
    if not np.all(np.isfinite(values)):
        raise _error("SOURCE_NONFINITE", "driven values are not finite")

    # -- operator and loads ----------------------------------------------------
    _check_cancel(probe, "assembling the solid operator")
    try:
        K = sparse.csr_matrix(prepared.assemble_linear(), dtype=float)
    except (ValueError, TypeError) as error:
        raise _error("PREPARED_OPERATOR_INVALID", f"assemble_linear failed: {error}") from error
    if K.shape != (info.ndof, info.ndof) or not np.all(np.isfinite(K.data)):
        raise _error("PREPARED_OPERATOR_INVALID", "the linear operator is not a finite square matrix of ndof")
    k_norm = float(abs(K).max()) if K.nnz else 0.0
    if k_norm == 0.0 or float(abs(K - K.T).max()) > _SYMMETRY_RTOL * k_norm:
        raise _error("PREPARED_OPERATOR_INVALID", "the linear operator is zero or not symmetric")
    f = np.zeros(info.ndof)
    for name, traction in skin:
        _check_cancel(probe, f"skin traction {name}")
        try:
            load = np.asarray(prepared.traction(name, traction), dtype=float)
        except ValueError as error:
            raise _error("SKIN_INVALID", f"traction of {name!r} failed: {error}", group=name) from error
        if load.shape != (info.ndof,) or not np.all(np.isfinite(load)):
            raise _error("SKIN_INVALID", f"traction load of {name!r} is malformed", group=name)
        f += load

    # -- support check: the prescribed set must remove every rigid mode --------
    _check_cancel(probe, "rigid-mode check")
    try:
        modes = np.asarray(prepared.rigid_modes(), dtype=float)
    except (ValueError, TypeError) as error:
        raise _error("PREPARED_INVALID", f"rigid_modes failed: {error}") from error
    if modes.ndim != 2 or modes.shape[0] != info.ndof or not np.all(np.isfinite(modes)):
        raise _error("PREPARED_INVALID", "rigid_modes must be a finite (ndof, 6m) array")
    singular_values = np.linalg.svd(modes[prescribed], compute_uv=False)
    if singular_values.size < modes.shape[1] or singular_values[-1] <= _RIGID_RANK_RATIO * max(singular_values[0], _TINY):
        raise _error(
            "UNDERCONSTRAINED",
            "the prescribed cut DOFs leave a rigid-body mode of the solid free "
            f"(smallest/largest rigid-mode singular value {singular_values[-1] / max(singular_values[0], _TINY):.3e})",
            singular_values=singular_values.tolist(),
        )

    # -- constrained static solve ---------------------------------------------
    _check_cancel(probe, "factorization")
    free = np.setdiff1d(np.arange(info.ndof), prescribed)
    q = np.zeros(info.ndof)
    q[prescribed] = values
    eta = 0.0
    kappa = 1.0
    refinements = 0
    if free.size:
        K_f = K[free][:, free].tocsc()
        rhs = f[free] - K[free][:, prescribed] @ values
        diagonal = K_f.diagonal()
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            raise _error("SOLVE_SINGULAR", "the free block has a non-positive diagonal (singular or mechanism)")
        d = 1.0 / np.sqrt(diagonal)
        D = sparse.diags(d)
        scaled = sparse.csc_matrix(D @ K_f @ D)
        scaled_rhs = d * rhs
        try:
            lu = splu(scaled)
        except RuntimeError as error:
            raise _error("SOLVE_SINGULAR", f"the free block is singular: {error}") from error
        y = lu.solve(scaled_rhs)
        _check_cancel(probe, "refinement")
        y = y + lu.solve(scaled_rhs - scaled @ y)
        refinements = 1
        if not np.all(np.isfinite(y)):
            raise _error("SOLVE_NONFINITE", "the constrained solve produced non-finite coefficients")
        residual = scaled @ y - scaled_rhs
        denominator = float(abs(scaled).sum(axis=1).max()) * float(np.max(np.abs(y), initial=0.0)) \
            + float(np.max(np.abs(scaled_rhs), initial=0.0))
        eta = float(np.max(np.abs(residual), initial=0.0) / denominator) if denominator > 0.0 else 0.0
        _check_cancel(probe, "condition estimate")
        m = scaled.shape[0]
        if m > 1:
            inverse = LinearOperator((m, m), matvec=lu.solve, rmatvec=lambda v: lu.solve(v, trans="T"), dtype=float)
            kappa = float(abs(scaled).sum(axis=0).max()) * float(onenormest(inverse))
        if not (np.isfinite(eta) and np.isfinite(kappa)):
            raise _error("SOLVE_NONFINITE", "the solve certificate is not finite")
        if kappa > _CONDITION_LIMIT:
            raise _error("POOR_CONDITIONING", f"estimated condition number {kappa:.3e} exceeds {_CONDITION_LIMIT:.0e}",
                         kappa=kappa)
        if eta > _BACKWARD_ERROR_LIMIT:
            raise _error("SOLVE_UNCERTIFIED", f"scaled backward error {eta:.3e} exceeds {_BACKWARD_ERROR_LIMIT:.0e}",
                         eta=eta)
        if kappa * eta > _ERROR_BOUND_LIMIT:
            raise _error("SOLVE_UNCERTIFIED", f"kappa*eta = {kappa * eta:.3e} exceeds {_ERROR_BOUND_LIMIT:.0e}",
                         eta=eta, kappa=kappa)
        q[free] = d * y
    if not np.all(np.isfinite(q)):
        raise _error("SOLVE_NONFINITE", "coefficients are not finite")

    # -- reactions and resultants ---------------------------------------------
    _check_cancel(probe, "reactions")
    r = K @ q - f
    reactions = r[prescribed]
    coords = info.coordinates
    comps = info.components
    free_scale = float(np.max(np.abs(abs(K)[free] @ np.abs(q) + np.abs(f[free])), initial=0.0)) if free.size else 0.0
    free_residual = float(np.max(np.abs(r[free]), initial=0.0))
    free_ratio = free_residual / (free_scale + _TINY)
    interface_force, interface_moment = _resultant(coords[prescribed], comps[prescribed], reactions)
    loaded = np.flatnonzero(f)
    skin_force, skin_moment = _resultant(coords[loaded], comps[loaded], f[loaded]) if loaded.size else (np.zeros(3), np.zeros(3))
    # Scale by the magnitude of the internal forces that cancel (|K||q|) and the loads, so a
    # rigid or unloaded solve compares roundoff with the operator scale, not with itself.
    magnitude = np.asarray(abs(K) @ np.abs(q)).reshape(-1) + np.abs(f)
    force_scale = float(np.sum(magnitude))
    moment_scale = float(np.sum(np.linalg.norm(coords, axis=1) * magnitude))
    force_balance = float(np.linalg.norm(interface_force + skin_force)) / (force_scale + _TINY)
    moment_balance = float(np.linalg.norm(interface_moment + skin_moment)) / (moment_scale + _TINY)
    if not (np.isfinite(force_balance) and np.isfinite(moment_balance)) or max(force_balance, moment_balance, free_ratio) > _BALANCE_LIMIT:
        raise _error(
            "EQUILIBRIUM_VIOLATED",
            f"force/moment/free-residual balance {force_balance:.3e}/{moment_balance:.3e}/{free_ratio:.3e} "
            f"exceeds {_BALANCE_LIMIT:.0e}",
            force_balance=force_balance, moment_balance=moment_balance, free_ratio=free_ratio,
        )

    group_results: Dict[str, GroupResultant] = {}
    shared = {d_: tuple(g) for d_, g in owners.items() if len(g) > 1}
    for name in names:
        dofs = group_dofs[name]
        position = np.searchsorted(prescribed, dofs)
        weight = np.array([1.0 / len(owners[int(d_)]) for d_ in dofs])
        force, moment = _resultant(coords[dofs], comps[dofs], reactions[position] * weight)
        group_results[name] = GroupResultant(
            name, force, moment, int(dofs.size), int(sum(1 for d_ in dofs if int(d_) in shared))
        )
    closure_force = np.sum([g.force for g in group_results.values()], axis=0) - interface_force
    closure_moment = np.sum([g.moment for g in group_results.values()], axis=0) - interface_moment

    _check_cancel(probe, "recovery")
    try:
        recovery = prepared.recover(q)
    except ValueError as error:
        raise _error("RECOVERY_FAILED", f"prepared.recover rejected the coefficients: {error}") from error
    nodal = _nodal_displacements(prepared, info, q)
    _check_cancel(probe, "finish")

    diagnostics = {
        "certification": {
            "certified": True,
            "scaled_backward_error": eta,
            "condition_estimate": kappa,
            "error_bound_estimate": kappa * eta,
            "refinement_steps": refinements,
            "backward_error_limit": _BACKWARD_ERROR_LIMIT,
            "condition_limit": _CONDITION_LIMIT,
            "error_bound_limit": _ERROR_BOUND_LIMIT,
            "rigid_mode_singular_values": singular_values.tolist(),
        },
        "residual": {
            "free_residual_inf": free_residual,
            "free_residual_ratio": free_ratio,
            "force_balance_ratio": force_balance,
            "moment_balance_ratio": moment_balance,
            "balance_limit": _BALANCE_LIMIT,
            "attribution_closure_force": closure_force,
            "attribution_closure_moment": closure_moment,
        },
        "constraints": {
            "cut_groups": tuple(names),
            "prescribed_dofs": int(prescribed.size),
            "free_dofs": int(free.size),
            "listed_group_dofs": int(listed),
            "duplicate_dof_count": int(duplicates),
            "shared_dof_count": len(shared),
        },
        "source": {
            "locations": len(cache),
            "hits_max": hits_max,
            "hit_spread_max": spread_max,
            "v2d_transverse_correction_max": correction_max,
            "normal_offset_max": offset_max,
            "element_kinds": dict(element_counts),
            "constraint_rows_checked": source.constraint_rows,
            "scale": source.scale,
        },
        "prepared": {"order": info.order, "ndof": info.ndof},
        "skin": {name: traction.tolist() for name, traction in skin},
    }
    return SolidSubmodelResult(
        coefficients=q,
        nodal_displacements=nodal,
        recovery=recovery,
        interface_force=interface_force,
        interface_moment=interface_moment,
        group_resultants=group_results,
        shared_attribution=shared,
        attribution_rule=ATTRIBUTION_RULE,
        prescribed_dofs=prescribed,
        prescribed_values=values,
        reactions=reactions,
        skin_force=skin_force,
        skin_moment=skin_moment,
        diagnostics=diagnostics,
    )
