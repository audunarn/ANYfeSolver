"""Projected normal work for a moving, primal-pressure S3 or Q4 shell patch.

Multipliers are held during structural equilibrium. The residual acts through
reference support times the normalized-gap variation, not the derivative of
moving overlap support; this avoids frictionless boundary-motion work.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import re
import sys

import numpy as np

from .moving import _has_shared_edge_transition
from .overlap import PlanarFace, planar_projected_overlap
from .pressure_basis import PressureBasis
from .q4_source_overlap import projected_q4_source_face
from .q4_s3_overlap_derivatives import (
    Q4S3OverlapDerivatives, UnsupportedQ4S3Branch, q4_s3_overlap_derivatives,
    q4_s3_overlap_first_derivatives,
)
from .q4_s3_tiny_wedge import verify_tiny_wedge_derivatives
from .s3_overlap_derivatives import UnsupportedOverlapBranch, s3_overlap_derivatives
from .s3_patch_mortar_energy import (
    MovingS3PatchMortarState, MovingS3PatchMortarWork,
    MovingS3PatchPrimalMortarPenalty,
)


@dataclass(frozen=True)
class _PreparedProjectedPatch:
    owner: object
    state: MovingS3PatchMortarState
    inputs: tuple[tuple[tuple[int, ...], bytes], ...]


@dataclass(frozen=True)
class NodalForceLinearization:
    """Unprojected mortar work and normalized-gap derivatives at one pose."""

    state: MovingS3PatchMortarState
    residual: np.ndarray
    tangent: np.ndarray | None
    gap_gradient: np.ndarray
    energy_evaluations: int
    gradient_only_pairs: int = 0


def _prepared_inputs(*arrays):
    return tuple((item.shape, item.tobytes()) for item in (
        np.ascontiguousarray(value, dtype=float) for value in arrays
    ))


def _geometry_preparation_size(value, seen=None):
    """Conservative retained Python metadata, key and array payload bytes."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        # Charge data even for a view; owned arrays may be double charged.
        return size+value.nbytes
    if isinstance(value, dict):
        return size+sum(_geometry_preparation_size(k, seen)
                        + _geometry_preparation_size(v, seen)
                        for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return size+sum(_geometry_preparation_size(x, seen) for x in value)
    if is_dataclass(value) and not isinstance(value, type):
        return size+_geometry_preparation_size(vars(value), seen)
    return size


def _geometry_preparation_value(value):
    """Exact bounded snapshots of owned geometry configuration, not history."""
    if value is None or type(value) in (bool, int, str, bytes):
        return value
    if isinstance(value, (float, np.floating)):
        return ("float", np.float64(value).tobytes())
    if isinstance(value, np.integer):
        return ("int", int(value))
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object-valued geometry is not cacheable")
        return ("array", value.dtype.str, value.shape,
                np.ascontiguousarray(value).tobytes())
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_geometry_preparation_value(x) for x in value))
    if isinstance(value, dict):
        return ("dict", tuple((k, _geometry_preparation_value(v))
                              for k, v in sorted(value.items())))
    if is_dataclass(value):
        return (type(value), tuple((f.name, _geometry_preparation_value(
            getattr(value, f.name))) for f in fields(value)))
    if (type(value).__module__.startswith("anyfesolver.general_contact.")
            and hasattr(value, "__dict__")):
        return (type(value), id(value), _geometry_preparation_value(vars(value)))
    raise TypeError("unsupported geometry configuration is not cacheable")


def _touching_projected_boundaries(first: PlanarFace, second: PlanarFace,
                                   *, relative_tolerance=1.0e-11) -> bool:
    """Conservatively identify a zero-area projection branch transition."""
    tangent = first.corners[1] - first.corners[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.cross(first.corners[1] - first.corners[0],
                      first.corners[-1] - first.corners[0])
    normal /= np.linalg.norm(normal)
    binormal = np.cross(normal, tangent)
    origin = first.corners[0]
    polygons = tuple(np.column_stack(((face.corners-origin) @ tangent,
                                      (face.corners-origin) @ binormal))
                     for face in (first, second))
    scale = max(float(np.max(np.linalg.norm(p-p[0], axis=1))) for p in polygons)
    tolerance = relative_tolerance*scale
    # A touching point must lie within tolerance of both projected boxes.
    # Work in the same computed frame as the segment predicate; retain that
    # predicate whenever its arithmetic is ambiguous or degenerate.
    if ("raise" not in np.geterr().values()
            and np.isfinite(tolerance) and tolerance >= 0.0
            and all(np.all(np.isfinite(p)) for p in polygons)):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            edges = tuple(np.roll(p, -1, axis=0)-p for p in polygons)
            lengths_squared = tuple(np.einsum("ij,ij->i", e, e) for e in edges)
            coordinate_scale = max(1.0, *(float(np.max(np.abs(p))) for p in polygons))
            padding = tolerance + 256.0*np.finfo(float).eps*coordinate_scale
            lower = tuple(np.nextafter(np.min(p, axis=0)-padding, -np.inf) for p in polygons)
            upper = tuple(np.nextafter(np.max(p, axis=0)+padding, np.inf) for p in polygons)
        if (all(np.all(np.isfinite(length)) and np.all(length >= np.finfo(float).tiny)
                for length in lengths_squared)
                and coordinate_scale <= np.sqrt(np.finfo(float).max)/8.0
                and all(np.all(np.isfinite(bound)) for bound in (*lower, *upper))
                and (np.any(upper[0] < lower[1]) or np.any(upper[1] < lower[0]))):
            return False
    for vertices, edges in ((polygons[0], polygons[1]),
                            (polygons[1], polygons[0])):
        for point in vertices:
            for start, end in zip(edges, np.roll(edges, -1, axis=0)):
                edge = end-start
                fraction = np.clip(np.dot(point-start, edge)/np.dot(edge, edge),
                                   0.0, 1.0)
                if np.linalg.norm(point-(start+fraction*edge)) <= tolerance:
                    return True
    return False


def _touching_shared_target_edge(first: PlanarFace,
                                 target_positions: np.ndarray,
                                 target_node_ids: np.ndarray,
                                 shared_node_ids: set[int],
                                 *, relative_tolerance=1.0e-11) -> bool:
    """Require the touching target edge, excluding exterior/endpoint ties."""
    edge = first.corners[1]-first.corners[0]
    tangent = edge/np.linalg.norm(edge)
    normal = np.cross(edge, first.corners[-1]-first.corners[0])
    normal /= np.linalg.norm(normal)
    binormal = np.cross(normal, tangent)
    origin = first.corners[0]
    source_xy = np.column_stack(((first.corners-origin) @ tangent,
                                 (first.corners-origin) @ binormal))
    target_xy = np.column_stack(((target_positions-origin) @ tangent,
                                 (target_positions-origin) @ binormal))
    scale = max(float(np.max(np.linalg.norm(source_xy-source_xy[0], axis=1))),
                float(np.max(np.linalg.norm(target_xy-target_xy[0], axis=1))))
    tolerance = relative_tolerance*scale

    def near_segment(point, start, end):
        span = end-start
        squared = float(np.dot(span, span))
        if squared <= 0.0:
            return False
        fraction = np.clip(np.dot(point-start, span)/squared,
                           0.0, 1.0)
        return np.linalg.norm(point-(start+fraction*span)) <= tolerance

    shared_touch = False
    exterior_touch = False
    for point in source_xy:
        for edge_index in range(3):
            following = (edge_index+1) % 3
            if near_segment(point, target_xy[edge_index], target_xy[following]):
                if (int(target_node_ids[edge_index]) in shared_node_ids
                        and int(target_node_ids[following]) in shared_node_ids):
                    shared_touch = True
                else:
                    exterior_touch = True
    # A target vertex on a source edge is an endpoint tie; it does not
    # uniquely identify the shared material edge's one-sided branch.
    for point in target_xy:
        for start, end in zip(source_xy, np.roll(source_xy, -1, axis=0)):
            if near_segment(point, start, end):
                exterior_touch = True
    return shared_touch and not exterior_touch


def _q4_s3_pair_value_rtol(local_area: float, derivative_area: float,
                           area_resolution: float) -> float:
    """Bound a float/Jet2 value comparison by resolved polygon disagreement."""
    if min(local_area, derivative_area) <= 0.0:
        raise UnsupportedQ4S3Branch("Q4/S3 paired overlap area is not positive")
    difference = abs(local_area-derivative_area)
    # Q1 support and weighted-gap values vary across a moving wedge, so
    # their float/Jet2 discrepancy may exceed the relative area discrepancy.
    # Keep the ordinary allowance tied to observed geometry disagreement.
    minimum_area = min(local_area, derivative_area)
    relative_area = difference/minimum_area
    relative = 4.0*relative_area
    subresolved = area_resolution >= 1.0e-3*minimum_area
    maximum_area_disagreement = 1.0e-6 if subresolved else 2.5e-7
    if (not np.isfinite(relative) or not np.isfinite(area_resolution)
            or difference > area_resolution
            or relative_area > maximum_area_disagreement):
        raise UnsupportedQ4S3Branch(
            "Q4/S3 paired overlap area agreement is unresolved "
            f"(area_difference_m2={difference:.12g}, "
            f"area_resolution_m2={area_resolution:.12g}, "
            f"relative_value_allowance={relative:.12g})"
        )
    # At a physical edge, the area alone can agree more closely than its Q1
    # support moments when propagated polygon-coordinate area resolution is
    # material relative to the tiny overlap. Allow a bounded 1e-6 comparison
    # only in that subresolved geometry regime. The wedge is still integrated
    # and its derivative and mechanical gates remain in force.
    if subresolved:
        return 1.0e-6
    return 1.0e-9+relative


def _q4_s3_pair_values_agree(local_values, derivative_values, *,
                             value_rtol: float, roundoff_atol: float,
                             local_area: float, derivative_area: float,
                             area_resolution: float,
                             normwise: bool = False) -> bool:
    """Compare Q1 moments; use a vector scale for subresolution or cancellation."""
    local_values = np.asarray(local_values)
    derivative_values = np.asarray(derivative_values)
    if (normwise
            or area_resolution >= 1.0e-3*min(local_area, derivative_area)):
        scale = max(float(np.max(np.abs(local_values))),
                    float(np.max(np.abs(derivative_values))))
        return bool(np.max(np.abs(local_values-derivative_values))
                    <= value_rtol*scale+roundoff_atol)
    return bool(np.allclose(local_values, derivative_values,
                            rtol=value_rtol, atol=roundoff_atol))


def _q4_s3_gap_operand_roundoff(source, target, support_scale: float) -> float:
    """Bound weighted-gap cancellation from interpolated position operands.

    The ordinary summation allowance scales with the already-cancelled gap.
    Nearly coincident physical skins also incur subtraction error at the
    scale of their absolute Cartesian operands. Keep that allowance
    below a billionth of the smallest source edge times Q1 support; very
    distant model origins must be recentered rather than accepted loosely.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if (source.shape != (4, 3) or target.shape not in {(3, 3), (4, 3)}
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(target))
            or not np.isfinite(support_scale) or support_scale <= 0.0):
        raise UnsupportedQ4S3Branch("Q4/S3 gap operands are invalid")
    edges = np.roll(source, -1, axis=0)-source
    lengths = np.linalg.norm(edges, axis=1)
    normal_length = np.linalg.norm(np.cross(
        source[1]-source[0], source[3]-source[0],
    ))
    if (not np.all(np.isfinite(lengths))
            or not np.isfinite(normal_length)
            or np.any(lengths <= 0.0) or normal_length <= 0.0):
        raise UnsupportedQ4S3Branch("Q4/S3 gap operand frame is degenerate")
    operands = np.concatenate((source, target))
    # Sum absolute Cartesian operands so one corner normal cannot understate
    # cancellation when the bilinear Q4's physical normal varies. This
    # covers interpolation/subtraction/projection arithmetic; the existing
    # integral-summation allowance and derivative gates remain separate.
    coordinate_operand_scale = float(np.max(np.sum(
        np.abs(operands), axis=1,
    )))
    distance_roundoff = (64.0*np.finfo(float).eps
                         * coordinate_operand_scale)
    if (not np.isfinite(coordinate_operand_scale)
            or not np.isfinite(distance_roundoff)):
        raise UnsupportedQ4S3Branch("Q4/S3 gap operand bound is not finite")
    if distance_roundoff > 1.0e-9*float(np.min(lengths)):
        raise UnsupportedQ4S3Branch(
            "Q4/S3 gap coordinate cancellation exceeds local resolution"
        )
    allowance = distance_roundoff*support_scale
    if not np.isfinite(allowance):
        raise UnsupportedQ4S3Branch("Q4/S3 gap operand allowance is not finite")
    return allowance


def _s3_gap_operand_roundoff(source, target, support) -> np.ndarray:
    """Per-row cancellation allowance for the two analytic S3 value paths.

    Both paths subtract interpolated Cartesian positions before projecting a
    small gap. Their absolute operand error is amplified by the source frame
    and the target barycentric frame in the source plane. A gamma(64) budget
    covers their short arithmetic chains; conditioning is explicit rather
    than inferred from the already cancelled weighted gap. Poorly resolved
    frames/origins reject, including when this allowance exceeds 1e-9 of the
    smallest source edge. This is a value-roundoff allowance, not an
    integration, derivative or physical-gap tolerance.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    support = np.asarray(support, float)
    if (source.shape != (3, 3) or target.shape != (3, 3)
            or support.shape != (3,) or np.any(support <= 0)
            or not all(np.all(np.isfinite(a)) for a in (source, target, support))):
        raise UnsupportedOverlapBranch("S3 gap operands or supports are invalid")
    source_edges = (source[1:]-source[0]).T
    lengths = np.linalg.norm(np.roll(source, -1, axis=0)-source, axis=1)
    singular = np.linalg.svd(source_edges, compute_uv=False)
    raw_normal = np.cross(source_edges[:, 0], source_edges[:, 1])
    normal_length = np.linalg.norm(raw_normal)
    if (np.any(lengths <= 0) or singular[-1] <= 0 or normal_length <= 0
            or not np.all(np.isfinite(singular))):
        raise UnsupportedOverlapBranch("S3 gap operand source frame is degenerate")
    normal = raw_normal/normal_length
    tangent = source_edges[:, 0]/lengths[0]
    frame = np.stack((tangent, np.cross(normal, tangent)))
    projected_target = frame @ (target[1:]-target[0]).T
    target_singular = np.linalg.svd(projected_target, compute_uv=False)
    if target_singular[-1] <= 0 or not np.all(np.isfinite(target_singular)):
        raise UnsupportedOverlapBranch("S3 gap operand target projection is degenerate")
    source_condition = singular[0]/singular[-1]
    target_condition = target_singular[0]/target_singular[-1]
    operand_scale = float(np.max(np.sum(np.abs(np.concatenate((source, target))), axis=1)))
    gamma = 64*np.finfo(float).eps/(1-64*np.finfo(float).eps)
    distance_roundoff = gamma*operand_scale*(1+source_condition)*(1+target_condition)
    if (not np.isfinite(distance_roundoff)
            or distance_roundoff > 1e-9*float(np.min(lengths))):
        raise UnsupportedOverlapBranch("S3 gap coordinate/frame cancellation exceeds local resolution")
    allowance = support*distance_roundoff
    if not np.all(np.isfinite(allowance)):
        raise UnsupportedOverlapBranch("S3 gap operand allowance is not finite")
    return allowance


class MovingS3PatchProjectedNormal:
    def __init__(self, geometry: MovingS3PatchPrimalMortarPenalty, *,
                 support_resolution: str = "exact",
                 pressure_basis: PressureBasis | None = None):
        if any(len(face.node_ids) not in {3, 4} for face in geometry.source_faces):
            raise ValueError("projected normal pressure space requires S3 or Q4 source faces")
        if support_resolution not in {"exact", "roundoff"}:
            raise ValueError("projected normal support resolution is invalid")
        self.geometry = geometry
        if (pressure_basis is not None
                and pressure_basis.fine_source_node_ids != geometry.source_node_ids):
            raise ValueError("macro pressure basis must match physical source node order")
        self.pressure_basis = pressure_basis
        self.support_resolution = support_resolution
        self.n_coordinates = geometry.n_coordinates
        self.source_node_ids = geometry.source_node_ids
        self.pressure_mode_ids = (geometry.source_node_ids if pressure_basis is None
                                  else pressure_basis.mode_ids)
        self.target_node_ids = geometry.target_node_ids
        self.source_face_indices = geometry.source_face_indices
        self.target_face_indices = geometry.target_face_indices
        self.reference_source = geometry.reference_source
        self.reference_target = geometry.reference_target
        self.stiffness = geometry.stiffness
        self.tangent_active_coordinates: np.ndarray | None = None
        self.shared_edge_tie: tuple[float, np.ndarray, float] | None = None
        # Verification may disable this implementation specialization. The
        # force law, supported gap rows and active nonzero-weight Hessians
        # are unchanged; seam/tiny-wedge recovery retains its full derivative.
        # Verified prototype, but the saved-candidate profile was slower
        # than batched Jet2. Keep it available only for explicit experiments.
        self.zero_weight_first_order = False
        # Private disposable geometry only. Never serialized or owner state.
        self._geometry_preparation_enabled = False
        self._geometry_preparation = None
        self._geometry_preparation_working = None
        self._geometry_preparation_limit_bytes = 32*1024*1024

    def discard_geometry_preparation(self):
        self._geometry_preparation = None
        self._geometry_preparation_working = None

    def _geometry_preparation_key(self, source, target):
        if (not self._geometry_preparation_enabled or self.zero_weight_first_order
                or self.pressure_basis is not None
                or self.geometry.differentiation != "jet2_q4_s3"):
            return None
        try:
            return (_prepared_inputs(source, target),
                    _geometry_preparation_value(self.geometry),
                    _geometry_preparation_value((
                        self.reference_source, self.reference_target,
                        self.source_node_ids, self.target_node_ids,
                        self.source_face_indices, self.target_face_indices,
                        self.pressure_mode_ids, self.support_resolution,
                        self.tangent_active_coordinates, self.shared_edge_tie,
                        self.zero_weight_first_order)),
                    q4_s3_overlap_derivatives,
                    self._geometry_preparation_limit_bytes)
        except (TypeError, RecursionError):
            # Unknown mutable configuration executes the original path.
            return None

    def _prepared_full_pair_derivative(self, pair_index, evaluator,
                                       reference, source, target, **options):
        working = self._geometry_preparation_working
        if working is None or evaluator is not q4_s3_overlap_derivatives:
            return evaluator(reference, source, target, **options)
        key = (pair_index, evaluator, _prepared_inputs(reference, source, target),
               _geometry_preparation_value(options))
        if key in working[1]:
            return working[1][key]
        derivative = evaluator(reference, source, target, **options)
        if type(derivative) is not Q4S3OverlapDerivatives:
            return derivative
        arrays = {f.name: getattr(derivative, f.name) for f in fields(derivative)
                  if isinstance(getattr(derivative, f.name), np.ndarray)}
        # Include request bytes, dataclass/array metadata and dictionary growth.
        # Over-counting shared references is safe; never under-charge payload.
        size = (_geometry_preparation_size(key)
                + _geometry_preparation_size(derivative)
                + 256)  # Conservative per-entry dictionary allocation allowance.
        if working[2]+size > self._geometry_preparation_limit_bytes:
            self.discard_geometry_preparation()
            return derivative
        copies = {name: a.copy() for name, a in arrays.items()}
        for a in copies.values():
            a.setflags(write=False)
        owned = replace(derivative, **copies)
        working[1][key] = owned
        working[2] += size
        return owned

    def set_shared_edge_tie(self, step_m, direction, relative_tolerance):
        """Opt in to a bounded local limit at the D1 two-face S3 diagonal."""
        self.shared_edge_tie = (float(step_m), np.asarray(direction, dtype=float),
                                float(relative_tolerance))

    def set_tangent_active_coordinates(self, coordinates):
        """Limit Hessian work to physical axes reached by reduced unknowns."""
        indices = np.asarray(coordinates, dtype=np.intp).reshape(-1)
        if (np.any(indices < 0) or np.any(indices >= self.n_coordinates)
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("projected patch tangent coordinate selection is invalid")
        self.tangent_active_coordinates = np.sort(indices)

    def _multipliers(self, multipliers):
        values = np.asarray(multipliers, dtype=float)
        if (values.shape != (len(self.pressure_mode_ids),)
                or not np.all(np.isfinite(values)) or np.any(values < 0.0)):
            raise ValueError("patch normal multipliers must be finite nonnegative pressures")
        return values

    def _pressure_state(self, fine_state):
        basis = self.pressure_basis
        if basis is None:
            return fine_state
        support, weighted_gap, gram = basis.transform_integrals(
            fine_state.reference_support,
            fine_state.reference_weighted_gap,
            fine_state.pressure_gram,
        )
        normalized = np.zeros_like(support)
        supported = support > 0.0
        normalized[supported] = weighted_gap[supported]/support[supported]
        pressure = np.zeros_like(support)
        pressure[supported] = self.stiffness*np.maximum(
            -normalized[supported], 0.0,
        )
        energy = 0.5*self.stiffness*float(
            support @ np.minimum(normalized, 0.0)**2,
        )
        return replace(
            fine_state, energy=energy, reference_support=support,
            reference_weighted_gap=weighted_gap, normalized_gap=normalized,
            pressure=pressure, pressure_gram=gram,
        )

    def resolved_support(self, state):
        """Return resolved support for algorithmic multiplier transfer only."""

        support = state.reference_support
        floor = (0.0 if self.support_resolution == "exact" else
                 np.finfo(float).eps*float(np.sum(support)))
        supported = support > floor
        return supported, floor

    def _project(self, state, multipliers):
        state = self._pressure_state(state)
        supported = state.reference_support > 0.0
        pressure = np.zeros_like(state.pressure)
        pressure[supported] = np.maximum(
            0.0, multipliers[supported]
            - self.stiffness*state.normalized_gap[supported],
        )
        # Algorithmic merit is useful for diagnostics, but its derivative is
        # not the virtual-work residual when overlap support changes.
        active_gap = np.zeros_like(state.normalized_gap)
        active_gap[supported] = np.minimum(
            state.normalized_gap[supported]
            - multipliers[supported]/self.stiffness, 0.0,
        )
        merit = 0.5*self.stiffness*float(
            state.reference_support @ (active_gap**2)
        )
        return replace(state, pressure=pressure, energy=merit)

    def evaluate(self, current_source, current_target, multipliers):
        multipliers = self._multipliers(multipliers)
        return self._project(
            self.geometry.evaluate(current_source, current_target), multipliers,
        )

    def prepare_work_state(self, current_source, current_target, multipliers):
        """Bind one disposable projected state to its law, pose and iterate."""
        validated = self._multipliers(multipliers)
        state = self.evaluate(current_source, current_target, validated)
        return _PreparedProjectedPatch(
            self, state,
            _prepared_inputs(current_source, current_target, validated),
        )

    def nodal_force_linearization(self, current_source, current_target,
                                  nodal_forces, *, tangent=True):
        """Reuse only successful full raw derivatives at a byte-exact pose.

        Evaluation counts remain logical request/resource accounting. Pressure,
        normalized work, force and tangent contractions always run anew.
        """
        previous = self._geometry_preparation
        self.discard_geometry_preparation()
        try:
            key = self._geometry_preparation_key(current_source, current_target)
        except BaseException:
            self.discard_geometry_preparation()
            raise
        if key is None:
            self._geometry_preparation_working = None
        elif previous is not None and previous[0] == key:
            # Reuse owned immutable records; no old/pending array duplication.
            self._geometry_preparation_working = previous
        else:
            working = [key, {}, 0]
            working[2] = _geometry_preparation_size(working)
            if working[2] <= self._geometry_preparation_limit_bytes:
                self._geometry_preparation_working = working
            else:
                self._geometry_preparation_working = None
        previous = None  # Drop old-pose arrays before computing another pose.
        try:
            result = self._nodal_force_linearization_uncached(
                current_source, current_target, nodal_forces, tangent=tangent)
            working = self._geometry_preparation_working
            if working is not None and result.gradient_only_pairs == 0:
                self._geometry_preparation = working
            return result
        except BaseException:
            self.discard_geometry_preparation()
            raise
        finally:
            self._geometry_preparation_working = None

    def _nodal_force_linearization_uncached(self, current_source, current_target,
                                           nodal_forces, *, tangent=True):
        """Evaluate the fine Q1 work with independent nodal forces in N.

        All supported gap rows are differentiated, even at zero pressure, so
        an inactive mode can enter the semismooth active set. No multiplier or
        physical material state is changed by this disposable evaluation.
        """
        if self.pressure_basis is not None or self.geometry.differentiation not in {"jet2_s3", "jet2_q4_s3", "jet2_q4_q4"}:
            raise ValueError("nodal-force linearization requires fine Q1 Q4/S3 or Q4/Q4 Jet2 geometry")
        z = np.asarray(nodal_forces, dtype=float)
        if z.shape != (len(self.pressure_mode_ids),) or not np.all(np.isfinite(z)):
            raise ValueError("nodal contact forces must be finite")
        if self.shared_edge_tie is None:
            return self._nodal_force_unresolved(current_source, current_target,
                                                z, tangent=tangent)
        try:
            return self._nodal_force_unresolved(current_source, current_target,
                                                z, tangent=tangent)
        except (UnsupportedQ4S3Branch, ValueError) as exc:
            message = str(exc)
            pair_match = re.search(r"\bpair=(\d+)\b", message)
            degenerate_shared_pair = (
                ("degenerate integration triangle" in message
                 or "paired overlap area agreement is unresolved" in message
                 or "differentiated S3 overlap disagrees with mortar geometry" in message)
                and pair_match is not None
            )
            failed_pair = (int(pair_match.group(1))
                           if pair_match is not None else None)
            if (not ("vertex lies on a clipping boundary" in message
                     or "shared-edge derivative is nonunique" in message
                     or degenerate_shared_pair)
                    or not self._only_artificial_shared_edge_ties(
                        current_source, current_target, failed_pair=failed_pair)):
                raise
            self.discard_geometry_preparation()
            return self._shared_edge_nodal_force_limit(
                current_source, current_target, z,
                tangent=tangent, recoverable_pair=failed_pair,
            )

    def _nodal_force_unresolved(self, current_source, current_target, z, *,
                                tangent, tiny_wedge_pair=None,
                                allow_gradient_only=True):
        """Differentiate fixed nodal-force work on one smooth overlap branch."""
        base = self.geometry.evaluate(current_source, current_target)
        support = base.reference_support
        if np.any(support < 0.0):
            raise ValueError("contact pressure support must be nonnegative")
        # A disappearing overlap can leave a disposable Newton iterate with
        # force on a now-unsupported node. It has no physical work there; its
        # inactive complementarity row drives that algorithmic force to zero.
        pressure = np.zeros_like(support)
        np.divide(z, support, out=pressure, where=support > 0.0)
        base = replace(base, pressure=pressure, energy=0.0)
        return self._jet2_s3_work(
            current_source, current_target, base, tangent=tangent,
            tiny_wedge_pair=tiny_wedge_pair, nodal_force_mode=True,
            allow_gradient_only=allow_gradient_only,
            zero_force_modes=(z == 0.0),
        )

    def _shared_edge_nodal_force_limit(self, current_source, current_target, z,
                                      *, tangent, recoverable_pair):
        """Guard a fixed-force generalized derivative at the artificial S3 seam."""
        step, direction, tolerance = self.shared_edge_tie
        lengths = [np.linalg.norm(face[i]-face[(i+1) % len(face)])
                   for faces, positions in (
                       (self.source_face_indices, self.reference_source),
                       (self.target_face_indices, self.reference_target))
                   for ids in faces for face in (positions[ids],)
                   for i in range(len(face))]
        if step > 1.0e-4*min(lengths):
            raise UnsupportedQ4S3Branch("raw shared-edge tie step exceeds local geometry scale")
        exact = self.geometry.evaluate(current_source, current_target)
        support = exact.reference_support
        supported = support > 0.0
        pressure = np.zeros_like(support)
        np.divide(z, support, out=pressure, where=supported)
        exact = replace(exact, pressure=pressure, energy=0.0)
        observations = []
        evaluations = 4*self.geometry._candidate_evaluation_count(
            self.geometry._candidate_pair_indices(current_source, current_target))
        self.geometry._check_energy_evaluation_budget(evaluations)
        for offset in (step, step/2.0):
            sides = []
            for sign in (-1.0, 1.0):
                shifted = np.asarray(current_source, dtype=float).copy()
                shifted += sign*offset*direction
                try:
                    candidate = self._nodal_force_unresolved(
                        shifted, current_target, z, tangent=tangent,
                        tiny_wedge_pair=recoverable_pair,
                        allow_gradient_only=False,
                    )
                except (UnsupportedQ4S3Branch, ValueError) as exc:
                    raise UnsupportedQ4S3Branch(
                        f"raw shared-edge tie smooth neighbor unresolved "
                        f"(offset_m={sign*offset:.12g}): {exc}"
                    ) from exc
                evaluations += candidate.energy_evaluations
                if evaluations > self.geometry.maximum_energy_evaluations:
                    raise MemoryError("raw Q4/S3 shared-edge tie evaluation budget exhausted")
                if not np.array_equal(candidate.state.reference_support > 0.0, supported):
                    raise UnsupportedQ4S3Branch("raw shared-edge tie changes pressure support")
                for name in ("reference_support", "reference_weighted_gap",
                             "normalized_gap"):
                    observed = getattr(candidate.state, name)
                    reference = getattr(exact, name)
                    scale = max(float(np.max(np.abs(reference))), 1.0e-20)
                    if np.any(np.abs(observed-reference)
                              > tolerance*np.maximum(np.abs(reference),
                                                     1.0e-8*scale)):
                        raise UnsupportedQ4S3Branch(
                            f"raw shared-edge tie {name} disagrees with exact state"
                        )
                sides.append(candidate)
            observations.append(sides)

        def relative_difference(a, b):
            return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),
                                                  np.linalg.norm(b), 1.0e-12))

        def limit(field):
            values = [[getattr(candidate, field) for candidate in sides]
                      for sides in observations]
            means = [0.5*(side[0]+side[1]) for side in values]
            extrapolated = (4.0*means[1]-means[0])/3.0
            errors = [relative_difference(side[0], side[1]) for side in values]
            errors.extend((relative_difference(means[0], means[1]),
                           relative_difference(means[1], extrapolated)))
            if (not np.all(np.isfinite(extrapolated)) or max(errors) > tolerance):
                raise UnsupportedQ4S3Branch(
                    f"raw shared-edge tie {field} limit unresolved "
                    f"(relative_error={max(errors):.12g})"
                )
            return extrapolated

        gap_gradient = limit("gap_gradient")
        residual = limit("residual")
        tangent_matrix = limit("tangent") if tangent else None
        if relative_difference(residual, -(z @ gap_gradient)) > tolerance:
            raise UnsupportedQ4S3Branch("raw shared-edge tie force/gap work disagrees")
        for row in np.flatnonzero(supported):
            first, second = observations
            coarse = 0.5*(first[0].gap_gradient[row]
                          + first[1].gap_gradient[row])
            fine = 0.5*(second[0].gap_gradient[row]
                        + second[1].gap_gradient[row])
            errors = (
                relative_difference(first[0].gap_gradient[row],
                                    first[1].gap_gradient[row]),
                relative_difference(second[0].gap_gradient[row],
                                    second[1].gap_gradient[row]),
                relative_difference(coarse, fine),
                relative_difference(fine, gap_gradient[row]),
            )
            if max(errors) > tolerance:
                raise UnsupportedQ4S3Branch(
                    "raw shared-edge tie supported gap row is nonunique"
                )
        action = np.concatenate((np.tile(direction, len(current_source)),
                                 np.zeros(3*len(current_target))))
        # Geometry-value subtraction at nanometre offsets is less accurate
        # than the directly differentiated row limit. Keep this independent
        # directional check within the existing 1e-5 smooth-branch gate.
        directional_tolerance = max(tolerance, 1.0e-5)
        if (relative_difference((observations[0][1].state.normalized_gap
                                - observations[0][0].state.normalized_gap)/(2.0*step),
                                gap_gradient @ action) > directional_tolerance):
            raise UnsupportedQ4S3Branch("raw shared-edge tie gap direction disagrees")
        if tangent:
            active = (np.arange(self.n_coordinates, dtype=np.intp)
                      if self.tangent_active_coordinates is None else
                      self.tangent_active_coordinates)
            for sides in observations:
                for column in active:
                    if (relative_difference(sides[0].tangent[:, column],
                                            sides[1].tangent[:, column])
                            > 2.0*tolerance):
                        raise UnsupportedQ4S3Branch(
                            "raw shared-edge tie tangent column is nonunique"
                        )
            for column in active:
                coarse = 0.5*(observations[0][0].tangent[:, column]
                              + observations[0][1].tangent[:, column])
                fine = 0.5*(observations[1][0].tangent[:, column]
                            + observations[1][1].tangent[:, column])
                if (max(relative_difference(coarse, fine),
                        relative_difference(fine, tangent_matrix[:, column]))
                        > tolerance):
                    raise UnsupportedQ4S3Branch(
                        "raw shared-edge tie tangent column changes with step"
                    )
            positions = np.vstack((current_source, current_target))
            characteristic_length = max(float(np.max(np.linalg.norm(
                positions-positions[0], axis=1))), 1.0e-20)
            characteristic_force = max(float(np.sum(np.linalg.norm(
                residual.reshape(-1, 3), axis=1))), 1.0e-20)
            observed = ((observations[0][1].residual
                         - observations[0][0].residual)/(2.0*step))[active]
            predicted = (tangent_matrix @ action)[active]
            # A rigid tangential shift can have a true derivative of zero.
            # At nanometre offsets, subtracting O(N) residuals leaves a
            # roundoff quotient. Bound its allowance by the subtraction's
            # actual force scale, with a physical cap. This is a near-null
            # development guard; resolved actions still use the relative
            # directional check and the independent tangent gate.
            residual_sides = observations[0]
            residual_roundoff = (
                32.0*np.finfo(float).eps
                * (np.linalg.norm(residual_sides[0].residual[active])
                   + np.linalg.norm(residual_sides[1].residual[active]))
                / (2.0*step)
            )
            direction_budget = max(
                tolerance*max(float(np.linalg.norm(observed)),
                              float(np.linalg.norm(predicted))),
                min(residual_roundoff,
                    1.0e-5*characteristic_force/characteristic_length),
            )
            if ((np.all(np.isin(np.flatnonzero(action), active))
                 and np.linalg.norm(observed-predicted) > direction_budget)
                    or relative_difference(tangent_matrix, tangent_matrix.T)
                    > tolerance):
                raise UnsupportedQ4S3Branch(
                    "raw shared-edge tie tangent direction or symmetry disagrees"
                )
        positions = np.vstack((current_source, current_target))
        forces = residual.reshape(-1, 3)
        force_scale = max(float(np.sum(np.linalg.norm(forces, axis=1))), 1.0e-20)
        length_scale = max(float(np.max(np.linalg.norm(positions-positions[0],
                                                       axis=1))), 1.0e-20)
        if (np.linalg.norm(forces.sum(axis=0)) > tolerance*force_scale
                or np.linalg.norm(np.cross(positions, forces).sum(axis=0))
                > tolerance*length_scale*force_scale):
            raise UnsupportedQ4S3Branch("raw shared-edge tie wrench is unbalanced")
        return NodalForceLinearization(exact, residual, tangent_matrix,
                                       gap_gradient, evaluations)

    def projected_multiplier(self, state, multipliers):
        multipliers = self._multipliers(multipliers)
        result = np.zeros_like(multipliers)
        supported, _floor = self.resolved_support(state)
        result[supported] = np.maximum(
            0.0, multipliers[supported]
            - self.stiffness*state.normalized_gap[supported],
        )
        return result

    def work(self, current_source, current_target, multipliers, *, tangent=True,
             _prepared_state=None):
        if (self.pressure_basis is not None
                and self.geometry.differentiation != "jet2_q4_s3"):
            raise ValueError("macro pressure work needs the Q4/S3 Jet2 derivative")
        if self.shared_edge_tie is None:
            return self._work_unresolved(current_source, current_target,
                                         multipliers, tangent=tangent,
                                         prepared_state=_prepared_state)
        try:
            return self._work_unresolved(current_source, current_target,
                                         multipliers, tangent=tangent,
                                         prepared_state=_prepared_state)
        except (UnsupportedQ4S3Branch, ValueError) as exc:
            if self.pressure_basis is not None:
                raise
            message = str(exc)
            pair_match = re.search(r"\bpair=(\d+)\b", message)
            degenerate_shared_pair = (
                ("degenerate integration triangle" in message
                 or "paired overlap area agreement is unresolved" in message
                 or "differentiated S3 overlap disagrees with mortar geometry" in message)
                and pair_match is not None
            )
            if (self.geometry.differentiation != "jet2_q4_s3"
                    or not ("vertex lies on a clipping boundary" in message
                            or "shared-edge derivative is nonunique" in message
                            or degenerate_shared_pair)
                    or not self._only_artificial_shared_edge_ties(
                        current_source, current_target,
                        failed_pair=int(pair_match.group(1))
                        if pair_match is not None else None)):
                if degenerate_shared_pair:
                    raise UnsupportedQ4S3Branch(message) from exc
                raise
            return self._shared_edge_limit(current_source, current_target,
                                           multipliers, tangent=tangent,
                                           recoverable_pair=(
                                               int(pair_match.group(1))
                                               if pair_match is not None else None))

    def _reference_smooth_target_edges(self):
        """Find manifold S3 edges with a smooth, convex local reference patch.

        Only a shared material edge can own an artificial clipping tie. A
        physical rim or nonconvex/creased two-face neighborhood has no
        eligible owner. Nonmanifold or duplicate topology disables the route.
        """
        faces = self.target_face_indices
        if len(faces) < 2 or any(len(face) != 3 for face in faces):
            return None
        reference = np.asarray(self.reference_target, dtype=float)
        if not np.all(np.isfinite(reference)):
            return None
        incidence = {}
        distinct_faces = set()
        for face_index, face in enumerate(faces):
            ids = tuple(map(int, face))
            if (len(set(ids)) != 3 or min(ids) < 0
                    or max(ids) >= len(reference)
                    or frozenset(ids) in distinct_faces):
                return None
            distinct_faces.add(frozenset(ids))
            for index in range(3):
                edge = tuple(sorted((ids[index], ids[(index+1) % 3])))
                incidence.setdefault(edge, []).append(face_index)
        if any(len(owners) > 2 for owners in incidence.values()):
            return None
        # Coincident positions with different identities leave material-edge
        # ownership ambiguous even when the incidence map looks manifold.
        used = sorted({int(node) for face in faces for node in face})
        all_edges = [np.linalg.norm(reference[a]-reference[b])
                     for a, b in incidence]
        coordinate_tolerance = 1.0e-10*max(all_edges)
        if any(np.linalg.norm(reference[a]-reference[b]) <= coordinate_tolerance
               for index, a in enumerate(used) for b in used[index+1:]):
            return None
        smooth = {index: [] for index in range(len(faces))}
        for shared, owners in incidence.items():
            if len(owners) != 2:
                continue
            first_index, second_index = owners
            first_ids = tuple(map(int, faces[first_index]))
            second_ids = tuple(map(int, faces[second_index]))
            first_tip = next(node for node in first_ids if node not in shared)
            second_tip = next(node for node in second_ids if node not in shared)
            first_reference = reference[list(first_ids)]
            second_reference = reference[list(second_ids)]
            first_normal = np.cross(first_reference[1]-first_reference[0],
                                    first_reference[2]-first_reference[0])
            second_normal = np.cross(second_reference[1]-second_reference[0],
                                     second_reference[2]-second_reference[0])
            normal_scale = np.linalg.norm(first_normal)*np.linalg.norm(second_normal)
            if (normal_scale <= 0.0 or np.dot(first_normal, second_normal)
                    < (1.0-1.0e-10)*normal_scale):
                continue
            quad = reference[[first_tip, shared[0], second_tip, shared[1]]]
            edge_scale = max(np.linalg.norm(quad[(index+1) % 4]-quad[index])
                             for index in range(4))
            if (edge_scale <= 0.0 or abs(float(
                    (second_reference[0]-first_reference[0]) @ first_normal))
                    > 1.0e-10*edge_scale*np.linalg.norm(first_normal)):
                continue
            turns = np.asarray([
                float(first_normal @ np.cross(
                    quad[(index+1) % 4]-quad[index],
                    quad[(index+2) % 4]-quad[(index+1) % 4],
                )) for index in range(4)
            ])
            turn_floor = 1.0e-10*edge_scale*edge_scale*np.linalg.norm(first_normal)
            if not (np.all(turns > turn_floor) or np.all(turns < -turn_floor)):
                continue
            smooth[first_index].append((second_index, shared))
            smooth[second_index].append((first_index, shared))
        return smooth

    def near_d1_diagonal_tiny_wedge(self, current_source, current_target,
                                   pair_index):
        """Locate one tiny smooth overlap beside an internal S3 edge."""
        smooth = self._reference_smooth_target_edges()
        if smooth is None or self.shared_edge_tie is None:
            return False
        face_count = len(self.target_face_indices)
        source_face, target_face = divmod(pair_index, face_count)
        if (source_face >= len(self.source_face_indices)
                or target_face >= face_count
                or len(self.source_face_indices[source_face]) != 4):
            return False
        source = np.asarray(current_source, dtype=float)
        target = np.asarray(current_target, dtype=float)
        positions = target[self.target_face_indices[target_face]]
        normal = np.cross(positions[1]-positions[0],
                           positions[2]-positions[0])
        if np.linalg.norm(normal) <= 0.0:
            return False
        unit_normal = normal/np.linalg.norm(normal)
        vertices = source[self.source_face_indices[source_face]]
        projected = vertices-((vertices-positions[0]) @ unit_normal)[:, None]*unit_normal
        for other_face, shared in smooth[target_face]:
            other = target[self.target_face_indices[other_face]]
            other_normal = np.cross(other[1]-other[0], other[2]-other[0])
            scale = np.linalg.norm(normal)*np.linalg.norm(other_normal)
            if (scale <= 0.0 or np.dot(normal, other_normal)
                    < (1.0-1.0e-10)*scale):
                continue
            edge = target[shared[1]]-target[shared[0]]
            edge_length = np.linalg.norm(edge)
            if (edge_length <= 0.0
                    or abs(float((other[0]-positions[0]) @ normal))
                    > 1.0e-10*edge_length*np.linalg.norm(normal)):
                continue
            along = ((projected-target[shared[0]]) @ edge)/(edge_length*edge_length)
            distance = np.linalg.norm(
                np.cross(projected-target[shared[0]], edge), axis=1,
            )/edge_length
            if np.any((along >= -1.0e-10) & (along <= 1.0+1.0e-10)
                      & (distance <= self.shared_edge_tie[0])):
                return True
        return False

    def _only_artificial_shared_edge_ties(self, current_source, current_target,
                                          *, failed_pair=None):
        """Admit only projected touches on smooth shared S3 material edges."""
        smooth = self._reference_smooth_target_edges()
        if smooth is None:
            return False
        source = np.asarray(current_source, dtype=float)
        target = np.asarray(current_target, dtype=float)
        direction = self.shared_edge_tie[1]
        found = False
        failed_pair_seen = False
        positive_target_faces = set()
        for source_face, source_ids in enumerate(self.source_face_indices):
            positions = source[source_ids]
            first, normal = projected_q4_source_face(positions)
            for target_face, target_ids in enumerate(self.target_face_indices):
                positions = target[target_ids]
                projected = positions-((positions-first.corners[0]) @ normal)[:, None]*normal
                second = PlanarFace((2, 1), projected)
                if planar_projected_overlap(first, second) is not None:
                    positive_target_faces.add(target_face)
                if not _touching_projected_boundaries(
                        first, second, relative_tolerance=1.0e-10):
                    continue
                # The one-sided direction must resolve each actual tie. A
                # sloped face elsewhere in the patch has no clipping event
                # to own and is checked by the smooth neighbor work instead.
                if abs(float(normal @ direction)) > 1.0e-4:
                    return False
                admitted = False
                for other_face, shared in smooth[target_face]:
                    other_ids = self.target_face_indices[other_face]
                    other = target[other_ids]
                    first_normal = np.cross(positions[1]-positions[0],
                                            positions[2]-positions[0])
                    other_normal = np.cross(other[1]-other[0], other[2]-other[0])
                    scale = np.linalg.norm(first_normal)*np.linalg.norm(other_normal)
                    if (scale <= 0.0 or np.dot(first_normal, other_normal)
                            < (1.0-1.0e-10)*scale):
                        continue
                    if abs(float(direction @ first_normal)) > 1.0e-10*np.linalg.norm(first_normal):
                        continue
                    face_size = max(np.linalg.norm(positions[1]-positions[0]),
                                    np.linalg.norm(other[1]-other[0]))
                    plane_error = abs(float((other[0]-positions[0]) @ first_normal))
                    if plane_error > 1.0e-10*face_size*np.linalg.norm(first_normal):
                        continue
                    shared_positions = target[list(shared)]
                    edge = shared_positions[1]-shared_positions[0]
                    crossing = np.cross(normal, edge)
                    if abs(float(direction @ crossing)) < 0.05*np.linalg.norm(crossing):
                        continue
                    if _touching_shared_target_edge(first, positions, target_ids,
                                                    shared,
                                                    relative_tolerance=1.0e-10):
                        admitted = True
                        break
                if not admitted:
                    return False
                found = True
                if failed_pair == source_face*len(self.target_face_indices)+target_face:
                    failed_pair_seen = True
        # The verified limit compares complete patch work. Restrict it to an
        # effectively two-face active target so unrelated active-face work
        # cannot hide a local edge derivative discrepancy in that norm.
        return (found and len(positive_target_faces) <= 2
                and (failed_pair is None or failed_pair_seen))

    def _shared_edge_limit(self, current_source, current_target, multipliers, *,
                           tangent, recoverable_pair):
        step, direction, tolerance = self.shared_edge_tie
        lengths = [np.linalg.norm(face[i]-face[(i+1) % len(face)])
                   for faces, positions in (
                       (self.source_face_indices, self.reference_source),
                       (self.target_face_indices, self.reference_target))
                   for ids in faces for face in (positions[ids],)
                   for i in range(len(face))]
        if step > 1.0e-4*min(lengths):
            raise UnsupportedQ4S3Branch("shared-edge tie step exceeds local geometry scale")
        exact = self.evaluate(current_source, current_target, multipliers)
        exact_supported, _ = self.resolved_support(exact)
        exact_active = exact.pressure > 0.0
        observations = []
        # Failed first work: base patch plus local float/Jet2 pair passes.
        # The exact state is evaluated again below; charge all four passes.
        evaluations = 4*self.geometry._candidate_evaluation_count(
            self.geometry._candidate_pair_indices(current_source, current_target))
        self.geometry._check_energy_evaluation_budget(evaluations)
        for offset in (step, step/2.0):
            sides = []
            for sign in (-1.0, 1.0):
                shifted = np.asarray(current_source, dtype=float).copy()
                shifted += sign*offset*direction
                try:
                    candidate = self._work_unresolved(
                        shifted, current_target, multipliers, tangent=tangent,
                        tiny_wedge_pair=recoverable_pair,
                    )
                except (UnsupportedQ4S3Branch, ValueError) as exc:
                    raise UnsupportedQ4S3Branch(
                        f"shared-edge tie smooth neighbor unresolved "
                        f"(offset_m={sign*offset:.12g}): {exc}"
                    ) from exc
                evaluations += candidate.energy_evaluations
                if evaluations > self.geometry.maximum_energy_evaluations:
                    raise MemoryError("Q4/S3 shared-edge tie evaluation budget exhausted")
                supported, _ = self.resolved_support(candidate.state)
                if (not np.array_equal(supported, exact_supported)
                        or not np.array_equal(candidate.state.pressure > 0.0,
                                              exact_active)):
                    raise UnsupportedQ4S3Branch("shared-edge tie changes pressure support or active set")
                for name in ("reference_support", "reference_weighted_gap",
                             "normalized_gap", "pressure"):
                    observed = getattr(candidate.state, name)
                    reference = getattr(exact, name)
                    scale = max(float(np.max(np.abs(reference))), 1.0e-20)
                    if np.any(np.abs(observed-reference)
                              > tolerance*np.maximum(np.abs(reference),
                                                     1.0e-8*scale)):
                        raise UnsupportedQ4S3Branch(
                            f"shared-edge tie {name} disagrees with exact state"
                        )
                sides.append(candidate)
            observations.append(sides)

        def relative_difference(a, b):
            return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),
                                                  np.linalg.norm(b), 1.0e-12))

        def limit(field):
            values = [[getattr(candidate, field) for candidate in sides]
                      for sides in observations]
            means = [0.5*(side[0]+side[1]) for side in values]
            extrapolated = (4.0*means[1]-means[0])/3.0
            errors = [relative_difference(side[0], side[1]) for side in values]
            errors.append(relative_difference(means[0], means[1]))
            errors.append(relative_difference(means[1], extrapolated))
            if (not np.all(np.isfinite(extrapolated))
                    or max(errors) > tolerance):
                raise UnsupportedQ4S3Branch(
                    f"shared-edge tie {field} limit unresolved (relative_error={max(errors):.12g})"
                )
            return extrapolated, max(errors)

        residual, residual_error = limit("residual")
        tangent_matrix, tangent_error = limit("tangent") if tangent else (None, 0.0)
        if tangent:
            active = (np.arange(self.n_coordinates, dtype=np.intp)
                      if self.tangent_active_coordinates is None else
                      self.tangent_active_coordinates)
            for offset_index, sides in enumerate(observations):
                if len(active) == 0:
                    break
                first = sides[0].tangent[:, active]
                second = sides[1].tangent[:, active]
                column_scale = max(np.max(np.linalg.norm(first, axis=0)),
                                   np.max(np.linalg.norm(second, axis=0)),
                                   1.0e-12)
                relative_columns = np.linalg.norm(first-second, axis=0) / np.maximum(
                    np.maximum(np.linalg.norm(first, axis=0),
                               np.linalg.norm(second, axis=0)),
                    1.0e-6*column_scale,
                )
                if np.max(relative_columns) > (2.0 if offset_index == 0 else 1.0)*tolerance:
                    raise UnsupportedQ4S3Branch(
                        "shared-edge tie reduced tangent columns disagree"
                    )
            trial_direction = np.zeros_like(np.asarray(current_source, dtype=float))
            trial_direction[:] = direction
            action = np.concatenate((trial_direction.reshape(-1),
                                     np.zeros(3*len(self.target_node_ids))))
            if np.all(np.isin(np.flatnonzero(action), active)):
                # Use the already verified outer ±step residual pair for this
                # directional quotient. At nanometre offsets the inner pair's
                # subtraction can amplify roundoff even as the tangent limit
                # itself remains stable; the tolerance is unchanged.
                centered = (observations[0][1].residual
                            - observations[0][0].residual)/(2.0*step)
                differentiated = tangent_matrix @ action
                if relative_difference(centered[active],
                                       differentiated[active]) > tolerance:
                    raise UnsupportedQ4S3Branch(
                        "shared-edge tie residual/tangent directional limit disagrees"
                    )
        positions = np.vstack((current_source, current_target))
        forces = residual.reshape(-1, 3)
        force_scale = max(float(np.sum(np.linalg.norm(forces, axis=1))), 1.0e-20)
        length_scale = max(float(np.max(np.linalg.norm(positions-positions[0],
                                                       axis=1))), 1.0e-20)
        if (np.linalg.norm(forces.sum(axis=0)) > tolerance*force_scale
                or np.linalg.norm(np.cross(positions, forces).sum(axis=0))
                > tolerance*length_scale*force_scale):
            raise UnsupportedQ4S3Branch("shared-edge tie exact-pose wrench is unbalanced")
        return MovingS3PatchMortarWork(exact, residual, tangent_matrix,
                                       evaluations,
                                       max(residual_error, tangent_error))

    def _work_unresolved(self, current_source, current_target, multipliers, *,
                         tangent=True, tiny_wedge_pair=None,
                         prepared_state=None):
        multipliers = self._multipliers(multipliers)
        coordinates = np.concatenate((np.asarray(current_source, dtype=float),
                                      np.asarray(current_target, dtype=float))).reshape(-1)
        if coordinates.size != self.n_coordinates:
            raise ValueError("patch work has invalid physical coordinate count")
        source_count = 3*len(self.source_node_ids)

        def state_at(values):
            return self.geometry.evaluate(
                values[:source_count].reshape(-1, 3),
                values[source_count:].reshape(-1, 3),
            )

        if prepared_state is None:
            base = self._project(state_at(coordinates), multipliers)
        elif (isinstance(prepared_state, _PreparedProjectedPatch)
              and prepared_state.owner is self
              and prepared_state.inputs == _prepared_inputs(
                  current_source, current_target, multipliers)):
            base = prepared_state.state
        else:
            raise ValueError("prepared projected patch state does not match this candidate")
        count = self.n_coordinates
        if not base.active_pressure_dofs:
            return MovingS3PatchMortarWork(
                base, np.zeros(count), np.zeros((count, count)) if tangent else None,
                len(self.geometry.pair_laws)
                if self.geometry.differentiation in {"pair_local", "jet2_s3", "jet2_q4_s3", "jet2_s3_q4", "jet2_q4_q4"} else 1,
            )
        for source_index in self.source_face_indices:
            source_positions = np.asarray(current_source)[source_index]
            if len(source_positions) == 4:
                first, normal = projected_q4_source_face(source_positions)
            else:
                first = PlanarFace((1, 1), source_positions)
                normal = np.cross(first.corners[1]-first.corners[0],
                                  first.corners[2]-first.corners[0])
                normal /= np.linalg.norm(normal)
            for target_position, target_index in enumerate(self.target_face_indices):
                target = np.asarray(current_target)[target_index]
                projected = target-((target-first.corners[0])@normal)[:, None]*normal
                second = PlanarFace((2, 1), projected)
                overlap = planar_projected_overlap(first, second)
                if (overlap is not None
                        and _has_shared_edge_transition(first, second, overlap)):
                    raise ValueError("S3 patch shared-edge derivative is nonunique")
                if (overlap is None
                        and self.geometry.differentiation in {"jet2_s3", "jet2_q4_s3", "jet2_s3_q4", "jet2_q4_q4"}
                        and _touching_projected_boundaries(first, second)):
                    if self.geometry.differentiation in {"jet2_s3", "jet2_s3_q4", "jet2_q4_q4"}:
                        raise UnsupportedOverlapBranch(
                            "S3 patch zero-area touching overlap has no smooth derivative"
                        )
                    # Use the inactive one-sided branch only for an artificial
                    # diagonal with an overlapping material-edge neighbor.
                    # Exterior rims and unmatched seams remain unresolved.
                    smooth_edges = self._reference_smooth_target_edges()
                    if smooth_edges is None:
                        raise UnsupportedQ4S3Branch(
                            "Q4/S3 zero-area touch has ambiguous target edge ownership"
                        )
                    adjacent_positive = False
                    for other_position, shared in smooth_edges[target_position]:
                        other_index = self.target_face_indices[other_position]
                        if not _touching_shared_target_edge(
                                first, target, target_index, shared):
                            continue
                        other = np.asarray(current_target)[other_index]
                        first_target_normal = np.cross(target[1]-target[0],
                                                       target[2]-target[0])
                        other_target_normal = np.cross(other[1]-other[0],
                                                       other[2]-other[0])
                        normal_product = (np.linalg.norm(first_target_normal)
                                          * np.linalg.norm(other_target_normal))
                        if (normal_product <= 0.0 or
                                np.dot(first_target_normal, other_target_normal)
                                < (1.0-1.0e-10)*normal_product):
                            continue
                        other_projected = other-((other-first.corners[0]) @ normal)[:, None]*normal
                        other_face = PlanarFace((2, 1), other_projected)
                        if planar_projected_overlap(first, other_face) is not None:
                            adjacent_positive = True
                            break
                    if not adjacent_positive:
                        raise UnsupportedQ4S3Branch(
                            "Q4/S3 zero-area touch has no positive adjacent target face"
                        )
        if self.geometry.differentiation == "pair_local":
            return self._pair_local_work(current_source, current_target,
                                         base, tangent=tangent)
        if self.geometry.differentiation in {"jet2_s3", "jet2_q4_s3", "jet2_s3_q4", "jet2_q4_q4"}:
            return self._jet2_s3_work(current_source, current_target,
                                      base, tangent=tangent,
                                      tiny_wedge_pair=tiny_wedge_pair)
        step = self.geometry.difference_step
        evaluations = 1
        cache = {}

        def geometry_at(steps):
            nonlocal evaluations
            key = tuple(sorted(steps))
            if key not in cache:
                evaluations += 1
                if evaluations > self.geometry.maximum_energy_evaluations:
                    raise MemoryError("S3 patch projected-work differentiation budget exhausted")
                trial = coordinates.copy()
                for coordinate, sign in steps:
                    trial[coordinate] += sign*step
                cache[key] = state_at(trial)
            return cache[key]

        positive = [geometry_at(((j, 1),)) for j in range(count)]
        negative = [geometry_at(((j, -1),)) for j in range(count)]
        jw = np.column_stack([
            (plus.reference_weighted_gap-minus.reference_weighted_gap)/(2*step)
            for plus, minus in zip(positive, negative)
        ])
        js = np.column_stack([
            (plus.reference_support-minus.reference_support)/(2*step)
            for plus, minus in zip(positive, negative)
        ])
        gap_variation = jw-base.normalized_gap[:, None]*js
        residual = -(base.pressure @ gap_variation)
        if not tangent:
            return MovingS3PatchMortarWork(base, residual, None, evaluations)
        active_coordinates = (
            np.arange(count, dtype=np.intp)
            if self.tangent_active_coordinates is None else
            self.tangent_active_coordinates
        )
        hessian_w = np.empty((len(self.source_node_ids),
                              len(active_coordinates), len(active_coordinates)))
        hessian_s = np.empty_like(hessian_w)
        for local_j, j in enumerate(active_coordinates):
            hessian_w[:, local_j, local_j] = (
                positive[j].reference_weighted_gap
                - 2*base.reference_weighted_gap
                + negative[j].reference_weighted_gap
            )/step**2
            hessian_s[:, local_j, local_j] = (
                positive[j].reference_support
                - 2*base.reference_support
                + negative[j].reference_support
            )/step**2
            for local_k in range(local_j+1, len(active_coordinates)):
                k = int(active_coordinates[local_k])
                mixed = (
                    geometry_at(((j, 1), (k, 1))).reference_weighted_gap
                    - geometry_at(((j, 1), (k, -1))).reference_weighted_gap
                    - geometry_at(((j, -1), (k, 1))).reference_weighted_gap
                    + geometry_at(((j, -1), (k, -1))).reference_weighted_gap
                )/(4*step**2)
                hessian_w[:, local_j, local_k] = mixed
                hessian_w[:, local_k, local_j] = mixed
                mixed_support = (
                    geometry_at(((j, 1), (k, 1))).reference_support
                    - geometry_at(((j, 1), (k, -1))).reference_support
                    - geometry_at(((j, -1), (k, 1))).reference_support
                    + geometry_at(((j, -1), (k, -1))).reference_support
                )/(4*step**2)
                hessian_s[:, local_j, local_k] = mixed_support
                hessian_s[:, local_k, local_j] = mixed_support
        dp = np.zeros((len(self.source_node_ids), len(active_coordinates)))
        dg = np.zeros_like(dp)
        active = base.pressure > 0.0
        support = base.reference_support[active, None]
        dg[active] = gap_variation[np.ix_(active, active_coordinates)]/support
        dp[active] = -self.stiffness*dg[active]
        support_variation = (
            hessian_w
            - np.einsum("ij,ik->ijk", js[:, active_coordinates], dg)
            - base.normalized_gap[:, None, None]*hessian_s
        )
        reduced_tangent = -(
            gap_variation[:, active_coordinates].T @ dp
            + np.einsum("i,ijk->jk", base.pressure, support_variation)
        )
        tangent_matrix = np.zeros((count, count))
        tangent_matrix[np.ix_(active_coordinates, active_coordinates)] = (
            reduced_tangent
        )
        return MovingS3PatchMortarWork(base, residual, tangent_matrix, evaluations)

    def _jet2_s3_work(self, current_source, current_target, base, *, tangent,
                      tiny_wedge_pair=None, nodal_force_mode=False,
                      allow_gradient_only=True, zero_force_modes=None):
        """Assemble differentiated pair geometry into shared pressure nodes."""
        if (not nodal_force_mode or tiny_wedge_pair is not None):
            self.discard_geometry_preparation()
        count = self.n_coordinates
        source_count = len(self.source_node_ids)
        active_coordinates = (
            np.arange(count, dtype=np.intp)
            if self.tangent_active_coordinates is None else
            self.tangent_active_coordinates
        )
        jw = np.zeros((source_count, count))
        js = np.zeros_like(jw)
        assembled_support = np.zeros(source_count)
        assembled_gap = np.zeros(source_count)
        support_magnitude = np.zeros(source_count)
        gap_magnitude = np.zeros(source_count)
        support_roundoff_budget = np.zeros(source_count)
        gap_roundoff_budget = np.zeros(source_count)
        gap_pair_value_budget = np.zeros(source_count)
        absolute_support_difference = np.zeros(source_count)
        absolute_gap_difference = np.zeros(source_count)
        tiny_wedge_verified = False
        tiny_wedge_scenarios = None
        gradient_only_pairs = 0
        # Keep all residual rows, including prescribed-coordinate reactions.
        # Only columns need restriction to reachable solution variations.
        weighted_hessian = np.zeros((count, len(active_coordinates))) if tangent else None
        support_hessian = np.zeros_like(weighted_hessian) if tangent else None
        zero_area_overlap_pairs = []
        source = np.asarray(current_source, dtype=float)
        target = np.asarray(current_target, dtype=float)
        candidate_indices = self.geometry._candidate_pair_indices(source, target)
        self.geometry._preflight_jet2_candidates(candidate_indices)
        candidate_count = self.geometry._candidate_evaluation_count(candidate_indices)
        basis = self.pressure_basis
        pair_states = base.pair_states
        if pair_states is not None and len(pair_states) != len(self.geometry.pair_laws):
            raise ValueError("prepared contact pair count does not match geometry")
        if pair_states is not None and any(
            (local is not None) != (candidate_indices is None or i in candidate_indices)
            for i, local in enumerate(pair_states)
        ):
            raise ValueError("prepared contact candidates do not match current geometry")
        evaluations = candidate_count  # Same-pose base float integration.
        self.geometry._check_energy_evaluation_budget(evaluations)
        if basis is not None:
            evaluations += candidate_count
            self.geometry._check_energy_evaluation_budget(evaluations)
        fine_base = (self.geometry.evaluate(source, target)
                     if basis is not None else base)
        if basis is None:
            fine_pressure = base.pressure
            fine_support_pressure = base.pressure*base.normalized_gap
        else:
            B = basis.as_array()
            fine_pressure = B @ base.pressure
            fine_support_pressure = B @ (base.pressure*base.normalized_gap)
        fine_work_gap = np.zeros(source_count)
        if nodal_force_mode:
            fine_work_gap[:] = base.normalized_gap
        else:
            np.divide(fine_support_pressure, fine_pressure, out=fine_work_gap,
                      where=fine_pressure > 0.0)
        for pair_index, law in enumerate(self.geometry.pair_laws):
            if candidate_indices is not None and pair_index not in candidate_indices:
                continue
            source_face = pair_index // len(self.geometry.target_faces)
            target_face = pair_index % len(self.geometry.target_faces)
            source_nodes = self.source_face_indices[source_face]
            target_nodes = self.target_face_indices[target_face]
            local_source = source[source_nodes]
            local_target = target[target_nodes]
            if pair_states is None:
                evaluations += 1
                self.geometry._check_energy_evaluation_budget(evaluations)
                local = law.evaluate(local_source, local_target)
            else:
                local = pair_states[pair_index]
            if local is None:
                continue
            if local.reference_overlap_area == 0.0:
                if local.overlap_vertices > 0:
                    zero_area_overlap_pairs.append(pair_index)
                continue
            if not nodal_force_mode and not np.any(fine_pressure[source_nodes]):
                # This pair contributes to shared support and weighted gap,
                # including at nodes active on other pairs. Its own smooth-
                # branch work derivatives vanish with all local pressures.
                assembled_support[source_nodes] += local.reference_support
                assembled_gap[source_nodes] += local.reference_weighted_gap
                support_magnitude[source_nodes] += np.abs(local.reference_support)
                gap_magnitude[source_nodes] += np.abs(local.reference_weighted_gap)
                continue
            evaluations += 1
            self.geometry._check_energy_evaluation_budget(evaluations)
            gradient_only = False
            if len(source_nodes) == 4:
                try:
                    tolerances = self.geometry.q4_source_derivative_tolerances or (None,)*4
                    derivative_options = {
                        "support_gradient_tolerance": tolerances[0],
                        "weighted_gap_gradient_tolerance": tolerances[2],
                        "support_hessian_tolerance": tolerances[1],
                        "weighted_gap_hessian_tolerance": tolerances[3],
                    }
                    evaluator = q4_s3_overlap_derivatives
                    if len(target_nodes) == 4:
                        from .q4_q4_overlap_derivatives import q4_q4_overlap_derivatives
                        evaluator = q4_q4_overlap_derivatives
                    if (len(target_nodes) == 3 and self.zero_weight_first_order and allow_gradient_only
                            and nodal_force_mode
                            and tiny_wedge_pair is None
                            and zero_force_modes is not None
                            and np.all(zero_force_modes[source_nodes])
                            and not np.any(fine_pressure[source_nodes])
                            and not np.any(fine_support_pressure[source_nodes])
                            and np.isfinite(fine_work_gap[source_nodes]).all()):
                        # All coefficients of this pair's local support/gap
                        # Hessians vanish EXACTLY. Its gradients remain in
                        # every pressure row and global normalization term.
                        gradient_only = True
                        gradient_only_pairs += 1
                        evaluator = q4_s3_overlap_first_derivatives
                        derivative_options.pop("support_hessian_tolerance")
                        derivative_options.pop("weighted_gap_hessian_tolerance")
                    derivative = self._prepared_full_pair_derivative(
                        pair_index, evaluator,
                        self.geometry.source_faces[source_face].reference_positions,
                        local_source, local_target,
                        area_tolerance=self.geometry.q4_source_area_tolerance,
                        weighted_gap_tolerance=self.geometry.q4_source_weighted_gap_tolerance,
                        maximum_triangles=self.geometry.maximum_q4_source_integration_triangles,
                        **({"allow_warped_source": True} if len(target_nodes) == 3 else {}),
                        **derivative_options,
                    )
                except UnsupportedQ4S3Branch as exc:
                    raise UnsupportedQ4S3Branch(
                        f"pair={pair_index}, overlap_area_m2={local.overlap_area:.12g}, "
                        f"reference_overlap_area_m2={local.reference_overlap_area:.12g}, "
                        f"integration_triangles={local.integration_triangles}: {exc}"
                    ) from exc
            elif len(target_nodes) == 4:
                from .s3_q4_overlap_derivatives import s3_q4_overlap_derivatives
                controls = self.geometry.q4_source_derivative_tolerances
                derivative = s3_q4_overlap_derivatives(
                    self.geometry.source_faces[source_face].reference_positions,
                    local_source, local_target,
                    support_tolerance=self.geometry.q4_source_area_tolerance,
                    weighted_gap_tolerance=self.geometry.q4_source_weighted_gap_tolerance,
                    support_gradient_tolerance=controls[0],
                    support_hessian_tolerance=controls[1],
                    weighted_gap_gradient_tolerance=controls[2],
                    weighted_gap_hessian_tolerance=controls[3],
                    maximum_triangles=self.geometry.maximum_q4_source_integration_triangles,
                )
            else:
                derivative = s3_overlap_derivatives(
                    self.geometry.source_faces[source_face].reference_positions,
                    local_source, local_target,
                )
            support_scale = max(float(np.max(np.abs(local.reference_support))),
                                float(np.max(np.abs(derivative.reference_support))))
            weighted_gap_scale = max(
                float(np.max(np.abs(local.reference_weighted_gap))),
                float(np.max(np.abs(derivative.reference_weighted_gap))),
            )
            # Both integrations sum work over the adaptive overlap triangles.
            # Bound cancellation near zero by that operation count; the
            # separate quadrature tolerances still govern approximation error.
            roundoff_factor = (512.0*np.finfo(float).eps
                               * max(1, local.integration_triangles))
            support_roundoff = roundoff_factor*support_scale
            weighted_gap_roundoff = roundoff_factor*weighted_gap_scale
            s3_operand_allowance = np.zeros(len(source_nodes))
            value_rtol = 1.0e-9
            if len(source_nodes) == 4:
                # The float and Jet2 clippers use different arithmetic paths.
                # For a tiny wedge, compare their area discrepancy with the
                # propagated polygon-coordinate resolution first. The bounded
                # value allowance also covers Q1 variation over a subresolved
                # overlap; derivative and integration gates remain separate.
                try:
                    value_rtol = _q4_s3_pair_value_rtol(
                        local.overlap_area, derivative.overlap_area,
                        derivative.overlap_area_resolution,
                    )
                except UnsupportedQ4S3Branch as exc:
                    raise UnsupportedQ4S3Branch(
                        f"pair={pair_index}, overlap_area_m2={local.overlap_area:.12g}: {exc}"
                    ) from exc
            support_difference = np.abs(
                derivative.reference_support-local.reference_support)
            gap_difference = np.abs(
                derivative.reference_weighted_gap-local.reference_weighted_gap)
            absolute_support_difference[source_nodes] += support_difference
            absolute_gap_difference[source_nodes] += gap_difference
            if len(source_nodes) == 4:
                support_agrees = _q4_s3_pair_values_agree(
                    local.reference_support, derivative.reference_support,
                    value_rtol=value_rtol, roundoff_atol=support_roundoff,
                    local_area=local.overlap_area,
                    derivative_area=derivative.overlap_area,
                    area_resolution=derivative.overlap_area_resolution,
                )
                gap_agrees = _q4_s3_pair_values_agree(
                    local.reference_weighted_gap,
                    derivative.reference_weighted_gap,
                    value_rtol=value_rtol,
                    roundoff_atol=weighted_gap_roundoff,
                    local_area=local.overlap_area,
                    derivative_area=derivative.overlap_area,
                    area_resolution=derivative.overlap_area_resolution,
                    normwise=True,
                )
                if not gap_agrees:
                    weighted_gap_roundoff = max(
                        weighted_gap_roundoff,
                        _q4_s3_gap_operand_roundoff(
                            local_source, local_target, support_scale,
                        ),
                    )
                    gap_agrees = _q4_s3_pair_values_agree(
                        local.reference_weighted_gap,
                        derivative.reference_weighted_gap,
                        value_rtol=value_rtol,
                        roundoff_atol=weighted_gap_roundoff,
                        local_area=local.overlap_area,
                        derivative_area=derivative.overlap_area,
                        area_resolution=derivative.overlap_area_resolution,
                        normwise=True,
                    )
            else:
                support_agrees = np.allclose(
                    derivative.reference_support, local.reference_support,
                    rtol=value_rtol, atol=support_roundoff,
                )
                gap_agrees = np.allclose(
                    derivative.reference_weighted_gap,
                    local.reference_weighted_gap,
                    rtol=value_rtol, atol=weighted_gap_roundoff,
                )
                if not gap_agrees:
                    support_scale_rows = np.maximum(
                        local.reference_support, derivative.reference_support)
                    if len(target_nodes) == 4:
                        from .s3_q4_gap_resolution import s3_q4_gap_operand_allowance
                        s3_operand_allowance = s3_q4_gap_operand_allowance(
                            local_source, local_target, support_scale_rows,
                            scalar_projection_residual_m=local.projection_residual_m,
                            jet_projection_residual_m=derivative.projection_residual_m,
                            natural_coordinate_bound=max(local.natural_coordinate_bound,
                                                         derivative.natural_coordinate_bound),
                            absolute_support_scale=np.maximum(
                                local.absolute_reference_support,
                                derivative.absolute_reference_support),
                        )
                    else:
                        s3_operand_allowance = _s3_gap_operand_roundoff(
                            local_source, local_target, support_scale_rows,
                        )
                    gap_agrees = np.allclose(
                        derivative.reference_weighted_gap, local.reference_weighted_gap,
                        rtol=value_rtol,
                        atol=weighted_gap_roundoff+s3_operand_allowance,
                    )
            if not support_agrees or not gap_agrees:
                if (pair_index == tiny_wedge_pair
                        and len(source_nodes) == 4
                        and self.shared_edge_tie is not None):
                    if basis is not None:
                        raise UnsupportedQ4S3Branch(
                            "macro pressure tiny seam wedge needs separate qualification"
                        )
                    if tiny_wedge_verified:
                        raise UnsupportedQ4S3Branch(
                            "multiple tiny seam wedges need separate derivative qualification"
                        )
                    tie_step = self.shared_edge_tie[0]
                    if (local.overlap_area > 4.0*tie_step*tie_step
                            or np.any(fine_base.reference_support[source_nodes]
                                      - local.reference_support
                                      < 1.0e6*local.reference_support)):
                        raise UnsupportedQ4S3Branch(
                            "tiny seam wedge exceeds area or independent support budget"
                        )
                    coordinate_map = np.asarray(
                        [3*int(node)+axis for node in source_nodes for axis in range(3)]
                        + [3*(source_count+int(node))+axis
                           for node in target_nodes for axis in range(3)],
                        dtype=np.intp,
                    )
                    local_active = np.flatnonzero(np.isin(
                        coordinate_map, active_coordinates))
                    reference_face = self.geometry.source_faces[source_face].reference_positions
                    edge_lengths = [np.linalg.norm(reference_face[index]
                                                   - reference_face[(index+1) % 4])
                                    for index in range(4)]
                    force_scale = (float(np.sum(np.abs(
                        base.pressure*base.reference_support)))
                        if nodal_force_mode else
                        float(base.pressure @ base.reference_support))
                    if nodal_force_mode and force_scale == 0.0:
                        # All fixed-z work derivatives are exactly zero. The
                        # sampler still needs a positive normalization to
                        # return its geometry discrepancies; the unweighted
                        # supported-row check below is the admission gate.
                        force_scale = 1.0
                    if force_scale <= 0.0:
                        raise UnsupportedQ4S3Branch(
                            "tiny seam wedge has no resolved absolute nodal force"
                        )
                    self.geometry._check_energy_evaluation_budget(evaluations+1)
                    self.discard_geometry_preparation()
                    used, scenarios = verify_tiny_wedge_derivatives(
                        law, local_source, local_target, derivative,
                        fine_pressure[source_nodes],
                        fine_work_gap[source_nodes], local_active,
                        difference_step_m=tie_step/20.0,
                        force_scale_n=force_scale,
                        length_scale_m=min(edge_lengths),
                        maximum_evaluations=(
                            self.geometry.maximum_energy_evaluations-evaluations),
                    )
                    evaluations += used
                    self.geometry._check_energy_evaluation_budget(evaluations)
                    tiny_wedge_verified = True
                    tiny_wedge_scenarios = (
                        source_nodes.copy(), coordinate_map,
                        local_active, scenarios, force_scale, min(edge_lengths),
                    )
                else:
                    support_error = float(np.max(np.abs(
                        derivative.reference_support-local.reference_support,
                    )))
                    gap_error = float(np.max(np.abs(
                        derivative.reference_weighted_gap-local.reference_weighted_gap,
                    )))
                    raise ValueError(
                        "differentiated S3 overlap disagrees with mortar geometry "
                        f"(pair={pair_index}, support_error_m2={support_error:.12g}, "
                        f"weighted_gap_error_m3={gap_error:.12g}, "
                        f"support_scale_m2={support_scale:.12g}, "
                        f"weighted_gap_scale_m3={weighted_gap_scale:.12g}, "
                        f"support_roundoff_m2={support_roundoff:.12g}, "
                        f"weighted_gap_roundoff_m3={weighted_gap_roundoff:.12g}, "
                        f"geometry_value_rtol={value_rtol:.12g}, "
                        f"integration_triangles={local.integration_triangles}, "
                        f"overlap_area_local_m2={local.overlap_area:.12g}, "
                        f"overlap_area_m2={derivative.overlap_area:.12g}, "
                        +
                        (f"overlap_area_resolution_m2={derivative.overlap_area_resolution:.12g}, "
                         if len(source_nodes) == 4 else
                         "overlap_area_resolution_m2=not_supplied_by_analytic_S3, ")
                        +
                        f"support_local_m2={local.reference_support.tolist()}, "
                        f"support_derivative_m2={derivative.reference_support.tolist()}, "
                        f"weighted_gap_local_m3={local.reference_weighted_gap.tolist()}, "
                        f"weighted_gap_derivative_m3={derivative.reference_weighted_gap.tolist()})"
                    )
            assembled_support[source_nodes] += derivative.reference_support
            assembled_gap[source_nodes] += derivative.reference_weighted_gap
            support_magnitude[source_nodes] += np.abs(derivative.reference_support)
            gap_magnitude[source_nodes] += np.abs(derivative.reference_weighted_gap)
            support_roundoff_budget[source_nodes] += support_roundoff
            gap_roundoff_budget[source_nodes] += weighted_gap_roundoff+s3_operand_allowance
            if len(source_nodes) == 4:
                # The pairwise Q4 weighted-gap certificate is normwise.
                # Each node in the assembled sum inherits at most this
                # pair's admitted vector discrepancy by triangle inequality.
                pair_gap_scale = max(
                    float(np.max(np.abs(local.reference_weighted_gap))),
                    float(np.max(np.abs(derivative.reference_weighted_gap))),
                )
                gap_pair_value_budget[source_nodes] += value_rtol*pair_gap_scale
            if not nodal_force_mode and not np.any(fine_pressure[source_nodes]):
                continue
            coordinate_map = np.asarray(
                [3*int(node)+axis for node in source_nodes for axis in range(3)]
                + [3*(source_count+int(node))+axis
                   for node in target_nodes for axis in range(3)],
                dtype=np.intp,
            )
            jw[np.ix_(source_nodes, coordinate_map)] += derivative.weighted_gap_gradient
            js[np.ix_(source_nodes, coordinate_map)] += derivative.support_gradient
            if tangent and not gradient_only:
                local_active = np.flatnonzero(np.isin(coordinate_map, active_coordinates))
                if len(local_active):
                    global_active = coordinate_map[local_active]
                    indices = np.searchsorted(active_coordinates, global_active)
                    local_pressure = fine_pressure[source_nodes]
                    weighted = np.einsum(
                        "i,ijk->jk", local_pressure,
                        derivative.weighted_gap_hessian[:, :, local_active],
                    )
                    supported = np.einsum(
                        "i,ijk->jk",
                        fine_support_pressure[source_nodes],
                        derivative.support_hessian[:, :, local_active],
                    )
                    weighted_hessian[np.ix_(coordinate_map, indices)] += weighted
                    support_hessian[np.ix_(coordinate_map, indices)] += supported
        # Pair-wise values have passed the bounded geometry comparison.
        # Summing signed contributions can make the assembled total tiny;
        # compare against the sum of contribution magnitudes so roundoff and
        # allowed pair errors do not turn into a false zero-scale mismatch.
        summation_tolerance = 1.0e-9 + 64.0*np.finfo(float).eps
        if tiny_wedge_verified:
            reference_length = min(
                np.linalg.norm(face.reference_positions[index]
                               - face.reference_positions[(index+1) % 4])
                for face in self.geometry.source_faces
                for index in range(4)
            )
            support_budget = 1.0e-10*fine_base.reference_support
            gap_budget = (1.0e-10*fine_base.reference_support
                          * np.maximum(np.abs(fine_base.normalized_gap),
                                       1.0e-8*reference_length))
            if (np.any(absolute_support_difference > support_budget)
                    or np.any(absolute_gap_difference > gap_budget)):
                raise UnsupportedQ4S3Branch(
                    "tiny seam wedge exceeds non-cancelling nodal value budget"
                )
            (tiny_source, tiny_coordinate_map, tiny_active, scenarios,
             force_scale, length_scale) = tiny_wedge_scenarios
            selected = active_coordinates
            D = (jw[:, selected]
                 - fine_work_gap[:, None]*js[:, selected])
            for gradient_error, hessian_error in scenarios:
                delta_support = np.zeros((source_count, len(selected)))
                delta_weighted = np.zeros_like(delta_support)
                selected_global = tiny_coordinate_map[tiny_active]
                selected_columns = np.searchsorted(selected, selected_global)
                delta_support[np.ix_(tiny_source, selected_columns)] = (
                    gradient_error[0][:, tiny_active])
                delta_weighted[np.ix_(tiny_source, selected_columns)] = (
                    gradient_error[1][:, tiny_active])
                E = delta_weighted-fine_work_gap[:, None]*delta_support
                if nodal_force_mode:
                    supported_rows = fine_base.reference_support > 0.0
                    normalized_gap_error = np.linalg.norm(
                        E[supported_rows], axis=1,
                    )/fine_base.reference_support[supported_rows]
                    if (not np.all(np.isfinite(normalized_gap_error))
                            or np.any(normalized_gap_error > 1.0e-7)):
                        raise UnsupportedQ4S3Branch(
                            "tiny seam wedge raw supported gap derivative is unresolved"
                        )
                tangent_error_bound = 0.0
                for node in np.flatnonzero(
                        fine_pressure != 0.0 if nodal_force_mode
                        else fine_pressure > 0.0):
                    support = fine_base.reference_support[node]
                    if support <= 0.0:
                        raise UnsupportedQ4S3Branch(
                            "tiny seam wedge active pressure lacks support"
                        )
                    p = abs(fine_pressure[node])
                    d_norm = float(np.linalg.norm(D[node]))
                    e_norm = float(np.linalg.norm(E[node]))
                    ds_norm = float(np.linalg.norm(delta_support[node]))
                    js_norm = float(np.linalg.norm(js[node, selected]))
                    tangent_error_bound += (
                        self.stiffness/support
                        * (2.0*d_norm*e_norm+e_norm*e_norm)
                        + p/support*(ds_norm*d_norm+js_norm*e_norm
                                     + ds_norm*e_norm)
                    )
                for local_node, node in enumerate(tiny_source):
                    p = abs(fine_pressure[node])
                    g = fine_work_gap[node]
                    tangent_error_bound += p*(
                        np.linalg.norm(hessian_error[1][local_node])
                        + abs(g)*np.linalg.norm(hessian_error[0][local_node])
                    )
                budget = 1.0e-5*force_scale/length_scale
                if (not np.isfinite(tangent_error_bound)
                        or tangent_error_bound > budget):
                    raise UnsupportedQ4S3Branch(
                        "tiny seam wedge projected tangent error is unresolved "
                        f"(bound_n_per_m={tangent_error_bound:.12g}, "
                        f"budget_n_per_m={budget:.12g})"
                    )
        if (not all(np.all(np.isfinite(values)) for values in (
                    assembled_support, fine_base.reference_support,
                    assembled_gap, fine_base.reference_weighted_gap,
                    support_magnitude, gap_magnitude,
                    support_roundoff_budget, gap_roundoff_budget,
                    gap_pair_value_budget))
                or np.any(np.abs(assembled_support-fine_base.reference_support)
                   > summation_tolerance*support_magnitude
                     + support_roundoff_budget)
                or np.any(np.abs(assembled_gap-fine_base.reference_weighted_gap)
                          > summation_tolerance*gap_magnitude
                            + gap_roundoff_budget
                            + gap_pair_value_budget)):
            raise ValueError(
                "differentiated S3 patch disagrees with shared pressure support: "
                f"support_error={np.max(np.abs(assembled_support-fine_base.reference_support)):.9g}, "
                f"weighted_gap_error={np.max(np.abs(assembled_gap-fine_base.reference_weighted_gap)):.9g}, "
                f"zero_area_overlap_pairs={zero_area_overlap_pairs}"
            )
        if basis is not None:
            js, jw = basis.transform_derivatives(js, jw)
        gap_variation = jw - base.normalized_gap[:, None]*js
        residual = -(base.pressure @ gap_variation)
        if nodal_force_mode:
            dg = np.zeros_like(gap_variation)
            supported = base.reference_support > 0.0
            dg[supported] = (gap_variation[supported]
                             / base.reference_support[supported, None])
            if not tangent:
                return NodalForceLinearization(base, residual, None, dg,
                                               evaluations, gradient_only_pairs)
            dp_geometry = np.zeros((len(base.pressure), len(active_coordinates)))
            dp_geometry[supported] = (
                -base.pressure[supported, None]
                * js[np.ix_(supported, active_coordinates)]
                / base.reference_support[supported, None]
            )
            cross = np.einsum("i,ij,ik->jk", base.pressure,
                              js, dg[:, active_coordinates])
            tangent_matrix = np.zeros((count, count))
            tangent_matrix[:, active_coordinates] = -(
                gap_variation.T @ dp_geometry
                + weighted_hessian - cross - support_hessian
            )
            if len(active_coordinates) < count:
                # The exact full Hessian is symmetric, and its active block
                # equals the unrestricted assembly. Complete the work mapping
                # of the uncomputed prescribed-coordinate columns from the
                # already differentiated inactive rows instead of leaving a
                # zero transpose block; reduced variations never reach them.
                inactive_coordinates = np.setdiff1d(
                    np.arange(count, dtype=np.intp), active_coordinates)
                tangent_matrix[np.ix_(active_coordinates, inactive_coordinates)] = (
                    tangent_matrix[np.ix_(inactive_coordinates,
                                          active_coordinates)].T
                )
            return NodalForceLinearization(base, residual, tangent_matrix, dg,
                                           evaluations, gradient_only_pairs)
        if not tangent:
            return MovingS3PatchMortarWork(base, residual, None, evaluations)
        active = base.pressure > 0.0
        dg = np.zeros((len(base.pressure), len(active_coordinates)))
        dg[active] = (gap_variation[np.ix_(active, active_coordinates)]
                      / base.reference_support[active, None])
        dp = -self.stiffness*dg
        cross = np.einsum("i,ij,ik->jk", base.pressure, js, dg)
        tangent_matrix = np.zeros((count, count))
        tangent_matrix[:, active_coordinates] = -(
            gap_variation.T @ dp
            + weighted_hessian - cross - support_hessian
        )
        return MovingS3PatchMortarWork(base, residual, tangent_matrix, evaluations)

    def _pair_local_work(self, current_source, current_target, base, *, tangent):
        """Assemble integral derivatives per face pair into shared pressure nodes."""
        count = self.n_coordinates
        active_coordinates = (
            np.arange(count, dtype=np.intp)
            if self.tangent_active_coordinates is None else
            self.tangent_active_coordinates
        )
        source_count = len(self.source_node_ids)
        pressure_count = source_count
        jw = np.zeros((pressure_count, count))
        js = np.zeros_like(jw)
        weighted_hessian = np.zeros((count, count)) if tangent else None
        support_hessian = np.zeros((count, count)) if tangent else None
        step = self.geometry.difference_step
        evaluations = len(self.geometry.pair_laws)  # Base patch integration.
        source = np.asarray(current_source, dtype=float)
        target = np.asarray(current_target, dtype=float)

        for pair_index, law in enumerate(self.geometry.pair_laws):
            source_face = pair_index // len(self.geometry.target_faces)
            target_face = pair_index % len(self.geometry.target_faces)
            source_nodes = self.source_face_indices[source_face]
            target_nodes = self.target_face_indices[target_face]
            if not np.any(base.pressure[source_nodes]):
                continue
            local_positions = np.concatenate((source[source_nodes],
                                              target[target_nodes])).reshape(-1)
            source_local_count = 3*len(source_nodes)
            coordinate_map = np.asarray(
                [3*int(node)+axis for node in source_nodes for axis in range(3)]
                + [3*(source_count+int(node))+axis
                   for node in target_nodes for axis in range(3)],
                dtype=np.intp,
            )
            local_count = len(coordinate_map)
            local_active = np.flatnonzero(np.isin(coordinate_map,
                                                  active_coordinates))
            cache = {}

            def state_at(steps):
                nonlocal evaluations
                key = tuple(sorted(steps))
                if key in cache:
                    return cache[key]
                evaluations += 1
                if evaluations > self.geometry.maximum_energy_evaluations:
                    raise MemoryError("S3 patch pair-local differentiation budget exhausted")
                trial = local_positions.copy()
                for coordinate, sign in steps:
                    trial[coordinate] += sign*step
                cache[key] = law.evaluate(
                    trial[:source_local_count].reshape(-1, 3),
                    trial[source_local_count:].reshape(-1, 3),
                )
                return cache[key]

            middle = state_at(())
            positive = [state_at(((j, 1),)) for j in range(local_count)]
            negative = [state_at(((j, -1),)) for j in range(local_count)]
            local_jw = np.column_stack([
                (plus.reference_weighted_gap-minus.reference_weighted_gap)/(2*step)
                for plus, minus in zip(positive, negative)
            ])
            jw[np.ix_(source_nodes, coordinate_map)] += local_jw
            local_js = np.column_stack([
                (plus.reference_support-minus.reference_support)/(2*step)
                for plus, minus in zip(positive, negative)
            ])
            js[np.ix_(source_nodes, coordinate_map)] += local_js
            if not tangent:
                continue
            if not len(local_active):
                continue
            active_map = coordinate_map[local_active]
            local_pressure = base.pressure[source_nodes]
            local_support_weight = (
                local_pressure*base.normalized_gap[source_nodes]
            )
            local_hessian = np.empty((len(local_active), len(local_active)))
            local_support_hessian = np.empty_like(local_hessian)
            for local_j, j in enumerate(local_active):
                local_hessian[local_j, local_j] = float(local_pressure @ (
                    positive[j].reference_weighted_gap
                    - 2*middle.reference_weighted_gap
                    + negative[j].reference_weighted_gap
                ))/step**2
                local_support_hessian[local_j, local_j] = float(
                    local_support_weight @ (
                        positive[j].reference_support
                        - 2*middle.reference_support
                        + negative[j].reference_support
                    )
                )/step**2
                for local_k in range(local_j+1, len(local_active)):
                    k = int(local_active[local_k])
                    mixed = (
                        state_at(((j, 1), (k, 1))).reference_weighted_gap
                        - state_at(((j, 1), (k, -1))).reference_weighted_gap
                        - state_at(((j, -1), (k, 1))).reference_weighted_gap
                        + state_at(((j, -1), (k, -1))).reference_weighted_gap
                    )/(4*step**2)
                    local_hessian[local_j, local_k] = local_hessian[local_k, local_j] = float(
                        local_pressure @ mixed
                    )
                    mixed_support = (
                        state_at(((j, 1), (k, 1))).reference_support
                        - state_at(((j, 1), (k, -1))).reference_support
                        - state_at(((j, -1), (k, 1))).reference_support
                        + state_at(((j, -1), (k, -1))).reference_support
                    )/(4*step**2)
                    weighted_support = float(local_support_weight @ mixed_support)
                    local_support_hessian[local_j, local_k] = weighted_support
                    local_support_hessian[local_k, local_j] = weighted_support
            weighted_hessian[np.ix_(active_map, active_map)] += local_hessian
            support_hessian[np.ix_(active_map, active_map)] += local_support_hessian

        gap_variation = jw-base.normalized_gap[:, None]*js
        residual = -(base.pressure @ gap_variation)
        if not tangent:
            return MovingS3PatchMortarWork(base, residual, None, evaluations)
        dp = np.zeros((pressure_count, len(active_coordinates)))
        dg = np.zeros_like(dp)
        active = base.pressure > 0.0
        support = base.reference_support[active, None]
        dg[active] = gap_variation[np.ix_(active, active_coordinates)]/support
        dp[active] = -self.stiffness*dg[active]
        cross = np.einsum("i,ij,ik->jk", base.pressure,
                          js[:, active_coordinates], dg)
        tangent_matrix = np.zeros((count, count))
        tangent_matrix[np.ix_(active_coordinates, active_coordinates)] = -(
            gap_variation[:, active_coordinates].T @ dp
            + weighted_hessian[np.ix_(active_coordinates, active_coordinates)]
            - cross
            - support_hessian[np.ix_(active_coordinates, active_coordinates)]
        )
        return MovingS3PatchMortarWork(base, residual, tangent_matrix, evaluations)
