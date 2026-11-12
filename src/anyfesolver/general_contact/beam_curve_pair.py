"""Local B3/B3 centreline distance geometry for distributed round-beam contact.

This work map is not a public pair-contact admission. It exposes all competing
closest points so an eventual line integrator can partition branch changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from anyfesolver._ge_beam3_deterministic_reduction import deterministic_dot, deterministic_norm
from .polynomial_roots import near_real_roots


@dataclass(frozen=True)
class CurveProjection:
    distance: float
    minimizers: tuple[float, ...]
    stationary_candidates: tuple[float, ...]

    @property
    def unique(self) -> bool:
        return len(self.minimizers) == 1


@dataclass(frozen=True)
class CurveDistanceJet:
    gap: float
    source_parameter: float
    target_parameter: float
    gradient: np.ndarray
    hessian: np.ndarray
    target_branch: str


@dataclass(frozen=True)
class B3PairPotentialState:
    energy: float
    residual: np.ndarray
    tangent: np.ndarray
    integration_points: int
    maximum_relative_integration_error: float
    integrated_intervals: int
    projection_evaluations: int


@dataclass(frozen=True)
class CurvePairGapCertificate:
    minimum_sampled_gap: float
    conservative_lower_gap: float
    cells_checked: int
    accuracy: float


def _positions(value: np.ndarray, label: str) -> np.ndarray:
    positions = np.asarray(value, dtype=float)
    if positions.shape != (3, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(f"{label} needs three finite B3 centreline positions")
    if float(np.linalg.norm(positions[2] - positions[0])) <= 0.0:
        raise ValueError(f"{label} B3 endpoints must be distinct")
    first = -3.0 * positions[0] + 4.0 * positions[1] - positions[2]
    second = 2.0 * positions[0] - 4.0 * positions[1] + 2.0 * positions[2]
    curvature_norm = deterministic_dot(second, second)
    station = (float(np.clip(-deterministic_dot(first, second) / (2.0 * curvature_norm),
                             0.0, 1.0)) if curvature_norm > 0.0 else 0.0)
    if (float(deterministic_norm(first + 2.0 * station * second))
            <= 1.0e-12 * float(deterministic_norm(positions[2] - positions[0]))):
        raise ValueError(f"{label} B3 centreline has a stationary cusp")
    return positions


def _shape(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array((2.0*t*t - 3.0*t + 1.0, 4.0*t - 4.0*t*t,
                  2.0*t*t - t)),
        np.array((4.0*t - 3.0, 4.0 - 8.0*t, 4.0*t - 1.0)),
        np.array((4.0, -8.0, 4.0)),
    )


def _curve(positions: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape, first, second = _shape(t)
    # Specify the archived three-term fused order independently of BLAS dispatch.
    from .native_replay_arithmetic import gap_matrix_product
    values = gap_matrix_product(np.array((shape, first, second)), positions)
    return values[0], values[1], values[2]


def closest_on_b3(point: np.ndarray, target_positions: np.ndarray) -> CurveProjection:
    """Find all global closest stations on a quadratic B3 centreline."""

    point = np.asarray(point, dtype=float)
    target = _positions(target_positions, "target")
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("B3 projection point must be a finite three-vector")
    a = point - target[0]
    b = -3.0 * target[0] + 4.0 * target[1] - target[2]
    c = 2.0 * target[0] - 4.0 * target[1] + 2.0 * target[2]
    coefficients = np.array((
        deterministic_dot(a, b),
        2.0 * deterministic_dot(a, c) - deterministic_dot(b, b),
        -3.0 * deterministic_dot(b, c),
        -2.0 * deterministic_dot(c, c),
    ))
    scale = float(np.max(np.abs(coefficients)))
    roots: list[float] = []
    if scale > 0.0:
        polynomial = np.polynomial.Polynomial(coefficients / scale).trim(
            tol=64.0 * np.finfo(float).eps,
        )
        for root in near_real_roots(polynomial.coef):
            if 0.0 < root < 1.0:
                roots.append(root)
    stations = [0.0]
    for root in sorted(roots):
        if root - stations[-1] > 1.0e-10:
            stations.append(root)
    stations.append(1.0)
    distances = [float(deterministic_norm(point - _curve(target, t)[0]))
                 for t in stations]
    smallest = min(distances)
    length_scale = max(deterministic_norm(b), deterministic_norm(c),
                       deterministic_norm(a), smallest, np.finfo(float).tiny)
    threshold = 256.0 * np.finfo(float).eps * length_scale
    candidates = [(float(t), distance) for t, distance in zip(stations, distances)
                  if distance - smallest <= threshold]
    # A root arbitrarily close to a physical endpoint can have the same
    # rounded distance. Treat geometrically indistinguishable stations as one
    # branch, retaining the candidate with the lowest computed distance.
    station_tolerance = (16.0 * np.sqrt(np.finfo(float).eps)
                         * float(deterministic_norm(target[2] - target[0])))
    clusters: list[list[tuple[float, float]]] = []
    for candidate in candidates:
        if (clusters and float(deterministic_norm(
            _curve(target, candidate[0])[0]
            - _curve(target, clusters[-1][-1][0])[0]
        )) <= station_tolerance):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    minimizers = tuple(min(cluster, key=lambda item: item[1])[0]
                       for cluster in clusters)
    return CurveProjection(smallest, minimizers, tuple(stations))


def b3_distance_jet(source_positions: np.ndarray,
                    target_positions: np.ndarray, source_parameter: float,
                    radius: float, *, source_is_a: bool = True) -> CurveDistanceJet:
    """Differentiate the unique smooth closest-point branch in 18 nodal translations."""

    source = _positions(source_positions, "source")
    target = _positions(target_positions, "target")
    s = float(source_parameter)
    if not np.isfinite(s) or not 0.0 <= s <= 1.0:
        raise ValueError("B3 source parameter must lie in [0,1]")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("combined round radius must be finite and positive")
    if type(source_is_a) is not bool:
        raise ValueError("source_is_a must be a boolean")
    source_position = _curve(source, s)[0]
    projection = closest_on_b3(source_position, target)
    if not projection.unique:
        raise ValueError("B3 projection has competing equal-distance minima")
    t = projection.minimizers[0]
    return _b3_distance_jet_at_station(source, target, s, radius, t,
                                       source_is_a=source_is_a)


def _b3_distance_jet_at_station(source, target, s, radius, t, *, source_is_a=True):
    """Internal envelope derivatives at an independently certified station.

    Callers supplying a station must prove constrained KKT and unique strict
    projection convexity. Historical b3_distance_jet selects/validates its
    station exactly as before and shares these identical derivative formulas.
    """
    source_shape = _shape(s)[0]
    source_position = _curve(source, s)[0]
    target_shape, target_shape_t, _target_shape_tt = _shape(t)
    target_position, target_tangent, target_second = _curve(target, t)
    separation = source_position - target_position
    distance = float(deterministic_norm(separation))
    if distance <= 0.0:
        raise ValueError("coincident B3 centrelines cannot define a contact normal")
    normal = separation / distance
    projector = np.eye(3) - np.outer(normal, normal)
    coefficients = (np.r_[source_shape, -target_shape] if source_is_a
                    else np.r_[-target_shape, source_shape])
    coefficients_t = (np.r_[np.zeros(3), -target_shape_t] if source_is_a
                      else np.r_[-target_shape_t, np.zeros(3)])
    jacobian = np.hstack([value * np.eye(3) for value in coefficients])
    jacobian_t = np.hstack([value * np.eye(3) for value in coefficients_t])
    gradient = jacobian.T @ normal
    hessian = jacobian.T @ projector @ jacobian / distance
    branch = "endpoint" if t == 0.0 or t == 1.0 else "interior"
    if branch == "interior":
        distance_t = -target_tangent
        mixed = (jacobian_t.T @ normal
                 + jacobian.T @ projector @ distance_t / distance)
        curvature = (deterministic_dot(distance_t @ projector, distance_t) / distance
                     - deterministic_dot(normal, target_second))
        if curvature <= 0.0:
            raise ValueError("B3 closest-point branch lacks positive curvature")
        hessian -= np.outer(mixed, mixed) / curvature
    return CurveDistanceJet(
        distance - radius, s, t, gradient, hessian, branch,
    )


_GAUSS_8 = np.polynomial.legendre.leggauss(8)
_GAUSS_16 = np.polynomial.legendre.leggauss(16)


def _bezier_controls(positions: np.ndarray, left: float,
                     right: float) -> np.ndarray:
    first = _curve(positions, left)[0]
    last = _curve(positions, right)[0]
    middle = _curve(positions, 0.5 * (left + right))[0]
    return np.array((first, 2.0 * middle - 0.5 * (first + last), last))


def _aabb_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_min, first_max = np.min(first, axis=0), np.max(first, axis=0)
    second_min, second_max = np.min(second, axis=0), np.max(second, axis=0)
    separation = np.maximum(np.maximum(first_min - second_max,
                                       second_min - first_max), 0.0)
    return float(np.linalg.norm(separation))


def _endpoint_projection_cuts(source: np.ndarray,
                              target: np.ndarray) -> tuple[float, ...]:
    """Source stations where a target-endpoint projection changes branch."""

    source_a = source[0]
    source_b = -3.0 * source[0] + 4.0 * source[1] - source[2]
    source_c = 2.0 * source[0] - 4.0 * source[1] + 2.0 * source[2]
    cuts = []
    for t in (0.0, 1.0):
        target_point, target_tangent, _ = _curve(target, t)
        coefficients = np.array((
            deterministic_dot(source_a - target_point, target_tangent),
            deterministic_dot(source_b, target_tangent),
            deterministic_dot(source_c, target_tangent),
        ))
        scale = float(np.max(np.abs(coefficients)))
        if scale <= 0.0:
            continue
        polynomial = np.polynomial.Polynomial(coefficients / scale).trim(
            tol=64.0 * np.finfo(float).eps,
        )
        for root in near_real_roots(polynomial.coef):
            if 0.0 < root < 1.0:
                cuts.append(root)
    return tuple(sorted(set(cuts)))


def _coef_trim(coefficients) -> tuple[float, ...]:
    # numpy.polynomial.polyutils.trimseq, exact: trailing zeros removed,
    # an all-zero series keeps its first element.
    values = tuple(coefficients)
    while len(values) > 1 and values[-1] == 0.0:
        values = values[:-1]
    return values


def _coef_add(first, second) -> tuple[float, ...]:
    # numpy.polynomial.polyutils._add, exact: trimmed elementwise addition
    # into the longer series, then trimmed again.
    first = _coef_trim(first)
    second = _coef_trim(second)
    if len(first) > len(second):
        return _coef_trim((*(
            x + y for x, y in zip(first, second)), *first[len(second):]))
    return _coef_trim((*(
        x + y for x, y in zip(first, second)), *second[len(first):]))


def _coef_sub(first, second) -> tuple[float, ...]:
    # numpy.polynomial.polyutils._sub, exact: when the second series
    # is not shorter, numpy negates it wholesale before adding the
    # first on the shared prefix, so its trailing coefficients carry
    # the negated sign (a mid-series zero becomes -0.0 and survives
    # the final trim); a longer first series keeps its trailing
    # coefficients unchanged.
    first = _coef_trim(first)
    second = _coef_trim(second)
    if len(first) > len(second):
        return _coef_trim((*(
            x - y for x, y in zip(first, second)), *first[len(second):]))
    return _coef_trim((*(
        x - y for x, y in zip(first, second)),
        *(-value for value in second[len(first):])))


def _coef_mul(first, second) -> tuple[float, ...]:
    # numpy.polynomial.polynomial.polymul, exact: trimmed convolution.
    return _coef_trim(np.convolve(
        np.asarray(_coef_trim(first), dtype=float),
        np.asarray(_coef_trim(second), dtype=float),
    ).tolist())


def _coef_pow2(coefficients) -> tuple[float, ...]:
    # numpy.polynomial.polynomial.polypow(c, 2), exact: the squared series
    # keeps untrimmed convolution length.
    coefficients = _coef_trim(coefficients)
    array = np.asarray(coefficients, dtype=float)
    return tuple(np.convolve(array, array).tolist())


def _coef_scalar_div(coefficients, value: float) -> tuple[float, ...]:
    # numpy.polynomial.polynomial.polydiv(c, [value]) quotient, exact.
    coefficients = _coef_trim(coefficients)
    if value == 0:
        raise ZeroDivisionError
    return tuple(x / value for x in coefficients)


def _coef_der(coefficients) -> tuple[float, ...]:
    # numpy.polynomial.polynomial.polyder(c, 1), exact.
    values = tuple(coefficients)
    count = len(values)
    if 1 >= count:
        return (values[0] * 0,) if count else ()
    return tuple(index * values[index] for index in range(1, count))


def _coef_val(station: float, coefficients):
    # Polynomial.__call__ for a scalar station on the default [0, 1]
    # domain/window, exact: mapdomain then polyval Horner evaluation.
    station = 0.0 + 1.0 * station
    value = coefficients[-1] + station * 0
    for index in range(2, len(coefficients) + 1):
        value = coefficients[-index] + value * station
    return value


def _real_polynomial_roots_unit(coefficients: np.ndarray | tuple[float, ...],
                                *, reject_ambiguous_minimum: bool = False
                                ) -> tuple[float, ...]:
    """Isolate every real root on [0, 1] between derivative extrema."""

    values = tuple(float(value) for value in coefficients)
    while len(values) > 1 and values[-1] == 0.0:
        values = values[:-1]
    if len(values) <= 1:
        return ()
    scale = max(abs(value) for value in values)
    normalized = tuple(value / scale for value in values)
    critical = _real_polynomial_roots_unit(_coef_der(normalized))
    stations = (0.0, *critical, 1.0)
    values = tuple(float(_coef_val(station, normalized))
                   for station in stations)
    evaluation_guard = 128.0 * np.finfo(float).eps * float(
        np.sum(np.abs(np.asarray(normalized, dtype=float))),
    )
    if reject_ambiguous_minimum:
        for index in range(1, len(stations) - 1):
            if (abs(values[index]) <= evaluation_guard
                    and values[index - 1] > evaluation_guard
                    and values[index + 1] > evaluation_guard):
                raise MemoryError(
                    "straight B3 target near-tangent contact roots "
                    "cannot be resolved at current precision"
                )
    roots = [station for station, value in zip(stations, values)
             if abs(value) <= evaluation_guard]
    for left, right, value_left, value_right in zip(
        stations[:-1], stations[1:], values[:-1], values[1:],
    ):
        if value_left * value_right < 0.0:
            roots.append(float(brentq(
                lambda station: _coef_val(station, normalized),
                left, right, xtol=1e-14, rtol=1e-14,
            )))
    return tuple(sorted(set(min(1.0, max(0.0, root))
                            for root in roots)))


def _source_interval_target_gap_lower(
    source: np.ndarray, left: float, right: float,
    target: np.ndarray, radius: float,
) -> float:
    """Bound a Q2 source interval using exact curve-to-chord station extrema.

    The target-to-source-chord minimum is piecewise quartic in target station.
    A quadratic source differs from its interval chord by at most its midpoint
    deviation. The floating guard keeps near-touching intervals unresolved.
    """
    start = _curve(source, left)[0]
    end = _curve(source, right)[0]
    middle = _curve(source, 0.5 * (left + right))[0]
    chord_deviation = float(deterministic_norm(middle - 0.5 * (start + end)))
    vector = end - start
    squared_length = deterministic_dot(vector, vector)
    if squared_length <= 0.0:
        raise MemoryError("B3 pair source interval chord is degenerate")

    # Local exact-coefficient arithmetic: each helper replicates the
    # numpy.polynomial.Polynomial operation previously used here (same
    # trimming, convolution, derivative and Horner evaluation order)
    # without the repeated general-purpose series validation.
    target_curve = tuple((
        float(target[0, axis]),
        -3.0*float(target[0, axis]) + 4.0*float(target[1, axis])
        - float(target[2, axis]),
        2.0*float(target[0, axis]) - 4.0*float(target[1, axis])
        + 2.0*float(target[2, axis]),
    ) for axis in range(3))
    offset = tuple(_coef_sub(component, (float(start[axis]),))
                   for axis, component in enumerate(target_curve))
    along = (0.0,)
    for axis, component in enumerate(offset):
        along = _coef_add(along, _coef_mul(
            component, (float(vector[axis]),)))
    projection = _coef_scalar_div(along, squared_length)
    start_distance = (0.0,)
    for component in offset:
        start_distance = _coef_add(start_distance,
                                   _coef_mul(component, component))
    end_distance = (0.0,)
    for axis, component in enumerate(target_curve):
        end_distance = _coef_add(end_distance, _coef_pow2(
            _coef_sub(component, (float(end[axis]),))))
    interior_distance = _coef_sub(
        start_distance,
        _coef_scalar_div(_coef_mul(along, along), squared_length))

    cuts = (0.0, *sorted(set((
        *_real_polynomial_roots_unit(projection),
        *_real_polynomial_roots_unit(_coef_sub(projection, (1.0,))),
    ))), 1.0)
    minimum_squared = np.inf
    coefficient_scale = 0.0
    for first, last in zip(cuts[:-1], cuts[1:]):
        if last - first <= 1.0e-14:
            continue
        projected = float(_coef_val(0.5*(first + last), projection))
        branch = (start_distance if projected <= 0.0 else
                  end_distance if projected >= 1.0 else interior_distance)
        coefficient_scale = max(
            coefficient_scale, float(np.sum(np.abs(branch))),
        )
        stations = (first, *_real_polynomial_roots_unit(_coef_der(branch)), last)
        minimum_squared = min(
            minimum_squared,
            *(float(_coef_val(station, branch)) for station in stations
              if first <= station <= last),
        )
    if not np.isfinite(minimum_squared):
        raise MemoryError("B3 pair interval distance could not be bounded")
    squared_guard = 1024.0 * np.finfo(float).eps * max(
        coefficient_scale, radius*radius, np.finfo(float).tiny,
    )
    position_guard = 1024.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(source))), float(np.max(np.abs(target))),
        radius,
    )
    return (float(np.sqrt(max(0.0, minimum_squared - squared_guard)))
            - radius - chord_deviation - position_guard)


def _straight_target_contact_roots(source: np.ndarray,
                                    target: np.ndarray,
                                    radius: float) -> tuple[float, ...]:
    """All line-contact boundaries against one exactly straight B3 target."""

    axis = target[2] - target[0]
    length_squared = float(axis @ axis)
    if length_squared <= 0.0:
        raise ValueError("straight B3 target has zero length")
    endpoint_roundoff = 128.0 * np.finfo(float).eps
    feature_cuts = (station for station in
                    _endpoint_projection_cuts(source, target)
                    if endpoint_roundoff < station < 1.0 - endpoint_roundoff)
    cuts = (0.0, *sorted(set((*np.linspace(0.0, 1.0, 33)[1:-1],
                               *feature_cuts))), 1.0)
    roots = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        controls = _bezier_controls(source, left, right)
        projected = ((controls - target[0]) @ axis) / length_squared
        projection_roundoff = (128.0 * np.finfo(float).eps
                               * max(1.0,
                                     float(np.max(np.abs(controls)))
                                     / np.sqrt(length_squared),
                                     float(np.max(np.abs(target)))
                                     / np.sqrt(length_squared)))
        if np.max(projected) <= projection_roundoff:
            origin = target[0]
            interior = False
        elif np.min(projected) >= 1.0 - projection_roundoff:
            origin = target[2]
            interior = False
        elif (np.min(projected) >= -projection_roundoff
              and np.max(projected) <= 1.0 + projection_roundoff):
            origin = target[0]
            interior = True
        else:
            raise MemoryError(
                "straight B3 target projection feature cannot be certified "
                f"on [{left:.9g}, {right:.9g}]: "
                f"projection range [{np.min(projected):.9g}, "
                f"{np.max(projected):.9g}]"
            )
        coefficients = np.array((
            controls[0] - origin,
            2.0 * (controls[1] - controls[0]),
            controls[0] - 2.0*controls[1] + controls[2],
        ))
        if interior:
            coefficients -= np.outer((coefficients @ axis) / length_squared,
                                     axis)
        distance_squared = np.polynomial.Polynomial([
            sum(float(coefficients[i] @ coefficients[j])
                for i in range(3) for j in range(3) if i + j == degree)
            for degree in range(5)
        ])
        distance_squared -= radius * radius
        for local in _real_polynomial_roots_unit(
            distance_squared.coef, reject_ambiguous_minimum=True,
        ):
            station = float(left + (right - left) * local)
            if 0.0 < station < 1.0:
                roots.append(station)
    return tuple(sorted(set(roots)))


def certify_b3_pair_gap(positions_a: np.ndarray, positions_b: np.ndarray,
                        radius: float, *, accuracy: float,
                        maximum_cells: int,
                        maximum_depth: int) -> CurvePairGapCertificate:
    """Conservatively bound minimum distance of two quadratic centrelines."""

    a = _positions(positions_a, "beam A")
    b = _positions(positions_b, "beam B")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("combined B3 pair radius must be finite and positive")
    if not np.isfinite(accuracy) or accuracy <= 0.0:
        raise ValueError("B3 pair gap accuracy must be finite and positive")
    if (type(maximum_cells) is not int or maximum_cells < 1
            or type(maximum_depth) is not int or maximum_depth < 0):
        raise ValueError("B3 pair gap budgets are invalid")
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))),
                float(radius), 1.0)
    roundoff = 256.0 * np.finfo(float).eps * scale
    best_sample = float("inf")
    best_lower = float("inf")
    cells_checked = 0
    cells = [(0.0, 1.0, 0.0, 1.0, 0)]
    while cells:
        first_left, first_right, second_left, second_right, depth = cells.pop()
        cells_checked += 1
        if cells_checked > maximum_cells:
            raise MemoryError("B3 pair gap cell budget exhausted")
        controls_a = _bezier_controls(a, first_left, first_right)
        controls_b = _bezier_controls(b, second_left, second_right)
        lower = _aabb_distance(controls_a, controls_b) - radius - roundoff
        first_samples = (first_left, 0.5 * (first_left + first_right), first_right)
        second_samples = (second_left, 0.5 * (second_left + second_right),
                          second_right)
        sampled = min(float(np.linalg.norm(_curve(a, s)[0] - _curve(b, t)[0]))
                      - radius for s in first_samples for t in second_samples)
        best_sample = min(best_sample, sampled)
        if lower >= best_sample - accuracy:
            best_lower = min(best_lower, lower)
            continue
        if depth >= maximum_depth:
            raise MemoryError("B3 pair gap depth budget exhausted")
        first_size = float(np.linalg.norm(np.ptp(controls_a, axis=0)))
        second_size = float(np.linalg.norm(np.ptp(controls_b, axis=0)))
        if first_size >= second_size:
            middle = 0.5 * (first_left + first_right)
            cells.extend(((first_left, middle, second_left, second_right, depth + 1),
                          (middle, first_right, second_left, second_right, depth + 1)))
        else:
            middle = 0.5 * (second_left + second_right)
            cells.extend(((first_left, first_right, second_left, middle, depth + 1),
                          (first_left, first_right, middle, second_right, depth + 1)))
    return CurvePairGapCertificate(best_sample, best_lower, cells_checked,
                                   float(accuracy))


class B3PairPotential:
    """Half-weighted two-direction distributed line potential, locally only."""

    def __init__(self, reference_a: np.ndarray, reference_b: np.ndarray,
                 radius: float, line_stiffness: float, *,
                 integration_relative_tolerance: float,
                 maximum_subdivisions: int,
                 reference_arc_measure: str = "straight"):
        self.reference_a = _positions(reference_a, "beam A reference").copy()
        self.reference_b = _positions(reference_b, "beam B reference").copy()
        if reference_arc_measure not in ("straight", "quadratic"):
            raise ValueError("B3 pair reference arc measure must be explicit")
        for label, reference in (("beam A", self.reference_a),
                                 ("beam B", self.reference_b)):
            length = float(np.linalg.norm(reference[2] - reference[0]))
            if (reference_arc_measure == "straight"
                    and float(np.linalg.norm(reference[1]
                        - 0.5 * (reference[0] + reference[2]))) > 1.0e-10 * length):
                raise ValueError(f"{label} classical B3 reference midpoint must be straight")
        for label, value in (("combined radius", radius),
                             ("line stiffness", line_stiffness),
                             ("integration tolerance", integration_relative_tolerance)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"B3 pair {label} must be finite and positive")
        if type(maximum_subdivisions) is not int or maximum_subdivisions < 1:
            raise ValueError("B3 pair maximum_subdivisions must be a positive integer")
        self.radius = float(radius)
        self.stiffness = float(line_stiffness)
        self.tolerance = float(integration_relative_tolerance)
        self.maximum_subdivisions = maximum_subdivisions
        self.reference_arc_measure = reference_arc_measure
        self.length_a = self._reference_length(self.reference_a)
        self.length_b = self._reference_length(self.reference_b)

    def _reference_speed(self, reference: np.ndarray, station: float) -> float:
        if self.reference_arc_measure == "straight":
            return float(np.linalg.norm(reference[2] - reference[0]))
        return float(np.linalg.norm(_curve(reference, station)[1]))

    def _reference_length(self, reference: np.ndarray) -> float:
        if self.reference_arc_measure == "straight":
            return float(np.linalg.norm(reference[2] - reference[0]))
        points, weights = np.polynomial.legendre.leggauss(32)
        return 0.5*sum(
            float(weight)*self._reference_speed(reference, 0.5*(float(point)+1.0))
            for point, weight in zip(points, weights)
        )

    def _direction(self, source: np.ndarray, target: np.ndarray,
                   length: float, source_is_a: bool):
        reference = self.reference_a if source_is_a else self.reference_b
        target_controls = _bezier_controls(target, 0.0, 1.0)
        points = 0
        scan_points = 0
        maximum_error = 0.0
        integrated_intervals = 0

        def gap_at(station: float) -> float:
            nonlocal scan_points
            scan_points += 1
            return (closest_on_b3(_curve(source, station)[0], target).distance
                    - self.radius)

        roots: list[float] = []

        def scan(left: float, right: float, gap_left: float,
                 gap_right: float, depth: int):
            middle = 0.5 * (left + right)
            gap_middle = gap_at(middle)
            samples = (gap_left, gap_middle, gap_right)
            if any(samples[i] * samples[i + 1] < 0.0 for i in (0, 1)):
                for first_s, last_s, first_g, last_g in (
                    (left, middle, gap_left, gap_middle),
                    (middle, right, gap_middle, gap_right),
                ):
                    if first_g * last_g < 0.0:
                        roots.append(float(brentq(
                            gap_at, first_s, last_s,
                            xtol=1.0e-14, rtol=1.0e-14,
                        )))
                return
            if gap_middle == 0.0:
                roots.append(middle)
                return
            controls = _bezier_controls(source, left, right)
            speed_reach = 0.5 * max(
                float(np.linalg.norm(controls[1] - controls[0])),
                float(np.linalg.norm(controls[2] - controls[1])),
            )
            if min(samples) > 0.0:
                if (_aabb_distance(controls, target_controls) > self.radius
                        or min(samples) > speed_reach
                        or _source_interval_target_gap_lower(
                            source, left, right, target, self.radius,
                        ) > 0.0):
                    return
            elif max(samples) < -speed_reach:
                return
            if depth >= self.maximum_subdivisions:
                raise MemoryError("B3 pair contact interval scan budget exhausted")
            scan(left, middle, gap_left, gap_middle, depth + 1)
            scan(middle, right, gap_middle, gap_right, depth + 1)

        if np.array_equal(target[1], 0.5 * (target[0] + target[2])):
            roots.extend(_straight_target_contact_roots(
                source, target, self.radius,
            ))
        else:
            coarse = np.linspace(0.0, 1.0, 33)
            coarse_gaps = [gap_at(float(station)) for station in coarse]
            for index in range(32):
                scan(float(coarse[index]), float(coarse[index + 1]),
                     coarse_gaps[index], coarse_gaps[index + 1], 0)
        cuts = (0.0, *sorted(set((*roots,
                                  *_endpoint_projection_cuts(source, target)))),
                1.0)
        active_intervals = tuple((left, right)
                                 for left, right in zip(cuts[:-1], cuts[1:])
                                 if right - left > 1.0e-14
                                 and gap_at(0.5 * (left + right)) < 0.0)

        def quadrature(left: float, right: float, rule):
            nonlocal points
            abscissae, weights = rule
            energy = 0.0
            residual = np.zeros(18)
            tangent = np.zeros((18, 18))
            active = False
            for abscissa, weight in zip(abscissae, weights):
                s = 0.5 * ((right - left) * float(abscissa) + right + left)
                station = b3_distance_jet(
                    source, target, s, self.radius,
                    source_is_a=source_is_a,
                )
                points += 1
                if station.gap >= 0.0:
                    continue
                active = True
                measure = (0.5 * (right - left) * float(weight)
                           * self._reference_speed(reference, s))
                factor = self.stiffness * measure
                energy += 0.5 * factor * station.gap * station.gap
                residual += factor * station.gap * station.gradient
                tangent += factor * (
                    np.outer(station.gradient, station.gradient)
                    + station.gap * station.hessian
                )
            return energy, residual, tangent, active

        def integrate(left: float, right: float, depth: int):
            nonlocal maximum_error, integrated_intervals
            lower_bound = _aabb_distance(
                _bezier_controls(source, left, right), target_controls,
            ) - self.radius
            if lower_bound > 0.0:
                return 0.0, np.zeros(18), np.zeros((18, 18))
            low = quadrature(left, right, _GAUSS_8)
            high = quadrature(left, right, _GAUSS_16)
            if not high[3] and not low[3]:
                if depth >= self.maximum_subdivisions:
                    raise MemoryError("B3 pair inactive interval cannot be certified")
            else:
                scales = (
                    max(abs(high[0]), self.stiffness * length * self.radius**2 * 1e-12),
                    max(float(np.linalg.norm(high[1])),
                        self.stiffness * length * self.radius * 1e-12),
                    max(float(np.linalg.norm(high[2])),
                        self.stiffness * length * 1e-12),
                )
                error = max(
                    abs(high[0] - low[0]) / scales[0],
                    float(np.linalg.norm(high[1] - low[1])) / scales[1],
                    float(np.linalg.norm(high[2] - low[2])) / scales[2],
                )
                if error <= self.tolerance:
                    maximum_error = max(maximum_error, error)
                    integrated_intervals += 1
                    return high[:3]
                if depth >= self.maximum_subdivisions:
                    raise MemoryError(
                        "B3 pair line integration subdivision budget exhausted"
                    )
            middle = 0.5 * (left + right)
            first = integrate(left, middle, depth + 1)
            second = integrate(middle, right, depth + 1)
            return tuple(first[i] + second[i] for i in range(3))

        energy, residual, tangent = 0.0, np.zeros(18), np.zeros((18, 18))
        for left, right in active_intervals:
            part = integrate(left, right, 0)
            energy += part[0]
            residual += part[1]
            tangent += part[2]
        return (energy, residual, tangent, points, maximum_error,
                integrated_intervals, scan_points)

    def evaluate(self, positions_a: np.ndarray,
                 positions_b: np.ndarray) -> B3PairPotentialState:
        a = _positions(positions_a, "beam A")
        b = _positions(positions_b, "beam B")
        first = self._direction(a, b, self.length_a, True)
        second = self._direction(b, a, self.length_b, False)
        return B3PairPotentialState(
            0.5 * (first[0] + second[0]),
            0.5 * (first[1] + second[1]),
            0.5 * (first[2] + second[2]),
            first[3] + second[3],
            max(first[4], second[4]),
            first[5] + second[5],
            first[6] + second[6],
        )
