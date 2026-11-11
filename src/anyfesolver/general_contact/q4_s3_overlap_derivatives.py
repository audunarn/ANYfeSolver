"""Smooth-branch derivatives for a physical Q4 source and S3 target.

The Q4 remains one bilinear Q1 face.  At an affine pose, the seven-point rule
integrates value and first/second variations exactly.  Warped poses use the
same adaptive quadrature semantics as the existing Q4 source kernel, with
derivatives of the accepted smooth quadrature branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, sqrt

from .moving import _TRIANGLE_RULE


class UnsupportedQ4S3Branch(ValueError):
    """The requested pose is outside an admitted smooth Q4/S3 branch."""


@dataclass(frozen=True)
class Q4S3OverlapDerivatives:
    """P1/Q1 node integrals and 21-coordinate derivatives.

    ``derivative_quadrature_differences`` sums leafwise coarse/refined
    differences in support gradient, support Hessian, gap gradient and gap
    Hessian order.  It is a resolution indicator, not a certified error bound.
    """

    reference_support: np.ndarray
    reference_weighted_gap: np.ndarray
    support_gradient: np.ndarray
    support_hessian: np.ndarray
    weighted_gap_gradient: np.ndarray
    weighted_gap_hessian: np.ndarray
    overlap_area: float
    overlap_area_resolution: float
    reference_overlap_area: float
    branch_signature: tuple[int, ...]
    integration_triangles: int
    derivative_quadrature_differences: np.ndarray


@dataclass(frozen=True)
class Q4S3OverlapFirstDerivatives:
    """P1/Q1 node integrals and 21-coordinate first derivatives only.

    ``gradient_quadrature_differences`` sums accepted leafwise coarse/refined
    support and weighted-gap gradient differences.  These are resolution
    indicators, not certified errors.  ``integration_tolerances`` records the
    area, weighted-gap, support-gradient and weighted-gap-gradient tolerances
    governing the adaptive branch; gradient entries are ``None`` for affine
    sources, where ``affine_relative_gradient_tolerance`` governs the final
    accepted smooth-branch resolution check.
    """

    reference_support: np.ndarray
    reference_weighted_gap: np.ndarray
    support_gradient: np.ndarray
    weighted_gap_gradient: np.ndarray
    overlap_area: float
    overlap_area_resolution: float
    reference_overlap_area: float
    branch_signature: tuple[int, ...]
    integration_triangles: int
    gradient_quadrature_differences: np.ndarray
    integration_tolerances: tuple[float, float, float | None, float | None]
    affine_relative_gradient_tolerance: float | None


_SIZE = 21

# Experimental process-local opt-in. This helper does not request compilation
# or import the JIT wrapper on the default path.
_COMPILED_BJET_MUL_ENABLED = False
_COMPILED_BJET_MUL_INITIALIZED = False


def _compiled_bjet_mul(av, ag, ah, bv, bg, bh):
    """Multiply second-order jet arrays with length-one batch broadcasting."""
    count = max(len(av), len(bv))
    size = ag.shape[1]
    if (len(av) not in (1, count) or len(bv) not in (1, count)):
        raise ValueError("jet batches cannot be broadcast")
    values = np.empty(count)
    gradients = np.empty((count, size))
    hessians = np.empty((count, size, size))
    for point in range(count):
        left = 0 if len(av) == 1 else point
        right = 0 if len(bv) == 1 else point
        values[point] = av[left]*bv[right]
        for row in range(size):
            gradients[point, row] = (
                ag[left, row]*bv[right] + bg[right, row]*av[left]
            )
            for column in range(size):
                hessians[point, row, column] = (
                    ah[left, row, column]*bv[right]
                    + bh[right, row, column]*av[left]
                    + ag[left, row]*bg[right, column]
                    + ag[left, column]*bg[right, row]
                )
    return values, gradients, hessians


def _enable_compiled_bjet_mul():
    """Opt in only when the optional JIT backend is available."""
    global _COMPILED_BJET_MUL_ENABLED, _COMPILED_BJET_MUL_INITIALIZED
    global _compiled_bjet_mul
    if not _COMPILED_BJET_MUL_INITIALIZED:
        from anyfesolver.jit_compiler import JIT_ENABLED, njit

        if JIT_ENABLED:
            _compiled_bjet_mul = njit(cache=True)(_compiled_bjet_mul)
        _COMPILED_BJET_MUL_INITIALIZED = True
        _COMPILED_BJET_MUL_ENABLED = JIT_ENABLED
    return _COMPILED_BJET_MUL_ENABLED


if os.environ.get("ANYSOLVER_Q4_S3_COMPILED_BJET_MUL") == "1":
    _enable_compiled_bjet_mul()


class _BJet:
    __slots__ = ("v", "g", "h")
    def __init__(self, v, g, h):
        self.v, self.g, self.h = v, g, h
    @classmethod
    def of(cls, j):
        return cls(np.array([j.value]), j.gradient[None, :], j.hessian[None, :, :])
    @classmethod
    def const(cls, values, size):
        values = np.asarray(values, dtype=float)
        n = values.shape[0]
        return cls(values, np.zeros((n, size)), np.zeros((n, size, size)))
    def __add__(self, o):
        if isinstance(o, _BJet):
            return _BJet(self.v + o.v, self.g + o.g, self.h + o.h)
        return _BJet(self.v + o, self.g, self.h)
    __radd__ = __add__
    def __neg__(self):
        return _BJet(-self.v, -self.g, -self.h)
    def __sub__(self, o):
        if isinstance(o, _BJet):
            return _BJet(self.v - o.v, self.g - o.g, self.h - o.h)
        return _BJet(self.v - o, self.g, self.h)
    def __rsub__(self, o):
        return _BJet(o - self.v, -self.g, -self.h)
    def scale(self, a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 0:
            return _BJet(self.v*a, self.g*a, self.h*a)
        return _BJet(self.v*a, self.g*a[:, None], self.h*a[:, None, None])
    def __mul__(self, o):
        if not isinstance(o, _BJet):
            return self.scale(o)
        a, b = self, o
        # The existing optional compiled experiment is scoped to its original
        # 21-coordinate Q4/S3 and S3/Q4 algebra. A process-wide opt-in must not
        # silently select another backend for the 24-coordinate Q4/Q4 route.
        if (_COMPILED_BJET_MUL_ENABLED
                and a.g.shape[1] == _SIZE and b.g.shape[1] == _SIZE):
            return _BJet(*_compiled_bjet_mul(a.v, a.g, a.h,
                                             b.v, b.g, b.h))
        og = a.g[:, :, None]*b.g[:, None, :]
        return _BJet(a.v*b.v, a.g*b.v[:, None] + b.g*a.v[:, None],
                    a.h*b.v[:, None, None] + b.h*a.v[:, None, None] + og + np.swapaxes(og, 1, 2))
    __rmul__ = __mul__
    def unary(self, value, first, second):
        return _BJet(value, first[:, None]*self.g,
                    first[:, None, None]*self.h + second[:, None, None]*(self.g[:, :, None]*self.g[:, None, :]))
    def reciprocal(self):
        if np.any(self.v == 0.0):
            raise ZeroDivisionError("jet reciprocal is singular")
        inv = 1.0/self.v
        return self.unary(inv, -(inv**2), 2.0*inv**3)
    def __truediv__(self, o):
        if isinstance(o, _BJet):
            return self*o.reciprocal()
        return self.scale(1.0/o)
    def sqrt(self):
        if np.any(self.v <= 0.0):
            raise ValueError("jet square root requires a positive value")
        root = np.sqrt(self.v)
        return self.unary(root, 0.5/root, -0.25/root**3)
    def weighted_sum(self, w):
        """Sum over points with weights -> scalar Jet2."""
        return Jet2(float(w @ self.v), w @ self.g, np.tensordot(w, self.h, axes=1))
    def select(self, mask, other):
        return _BJet(np.where(mask, self.v, other.v), np.where(mask[:, None], self.g, other.g),
                    np.where(mask[:, None, None], self.h, other.h))

class _FloatOnly:
    """Value-only accumulator for quantities whose derivatives are never used."""
    __slots__ = ("value",)
    def __init__(self, value): self.value = float(value)
    def __add__(self, o): return _FloatOnly(self.value + o.value)






class _Jet1:
    """Local dense first-order scalar; no second-order storage or arithmetic."""

    __slots__ = ("value", "gradient")

    def __init__(self, value: float, gradient: np.ndarray):
        self.value = float(value)
        self.gradient = gradient

    @classmethod
    def constant(cls, value: float, size: int) -> _Jet1:
        return cls(value, np.zeros(size))

    @classmethod
    def variable(cls, value: float, index: int, size: int) -> _Jet1:
        gradient = np.zeros(size)
        gradient[index] = 1.0
        return cls(value, gradient)

    def _coerce(self, other) -> _Jet1:
        return other if isinstance(other, _Jet1) else _Jet1.constant(float(other), self.gradient.size)

    def __add__(self, other) -> _Jet1:
        made = self._coerce(other)
        return _Jet1(self.value + made.value, self.gradient + made.gradient)

    __radd__ = __add__

    def __neg__(self) -> _Jet1:
        return _Jet1(-self.value, -self.gradient)

    def __sub__(self, other) -> _Jet1:
        return self + (-self._coerce(other))

    def __rsub__(self, other) -> _Jet1:
        return self._coerce(other) - self

    def __mul__(self, other) -> _Jet1:
        made = self._coerce(other)
        return _Jet1(self.value * made.value,
                     self.gradient * made.value + made.gradient * self.value)

    __rmul__ = __mul__

    def reciprocal(self) -> _Jet1:
        if self.value == 0.0:
            raise ZeroDivisionError("jet reciprocal is singular")
        inverse = 1.0 / self.value
        return _Jet1(inverse, -(inverse**2) * self.gradient)

    def __truediv__(self, other) -> _Jet1:
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other) -> _Jet1:
        return self._coerce(other) * self.reciprocal()


def _sqrt_first(argument: _Jet1) -> _Jet1:
    if argument.value <= 0.0:
        raise ValueError("jet square root requires a positive value")
    root = float(np.sqrt(argument.value))
    return _Jet1(root, 0.5 / root * argument.gradient)


def _zero_first() -> _Jet1:
    return _Jet1.constant(0.0, _SIZE)


def _scale_first(value: _Jet1, factor: float) -> _Jet1:
    scalar = float(factor)
    return _Jet1(value.value * scalar, value.gradient * scalar)


def _sum_first(items) -> _Jet1:
    return sum(items, start=_zero_first())


def _dot_first(a, b) -> _Jet1:
    return _sum_first(x*y for x, y in zip(a, b))


def _normalize_first(a):
    length = _sqrt_first(_dot_first(a, a))
    return [x/length for x in a]


def _signed_area_first(poly):
    origin = poly[0]
    return _scale_first(_sum_first(_cross2(_sub(poly[i], origin),
                                           _sub(poly[i+1], origin))
                                   for i in range(1, len(poly)-1)), 0.5)


def _q1_first(r, s):
    one = _Jet1.constant(1.0, _SIZE)
    shape = [_scale_first((one-r)*(one-s), 0.25),
             _scale_first((one+r)*(one-s), 0.25),
             _scale_first((one+r)*(one+s), 0.25),
             _scale_first((one-r)*(one+s), 0.25)]
    dr = [_scale_first(-(one-s), 0.25), _scale_first(one-s, 0.25),
          _scale_first(one+s, 0.25), _scale_first(-(one+s), 0.25)]
    ds = [_scale_first(-(one-r), 0.25), _scale_first(-(one+r), 0.25),
          _scale_first(one+r, 0.25), _scale_first(one-r, 0.25)]
    return shape, dr, ds


def _map_xy_first(shape, corners):
    return [_sum_first(shape[i]*corners[i][k] for i in range(4))
            for k in range(2)]


def _natural_first(point, corners):
    a = np.array([corners[1][k].value-corners[0][k].value for k in range(2)])
    b = np.array([corners[3][k].value-corners[0][k].value for k in range(2)])
    rhs = np.array([point[k].value-corners[0][k].value for k in range(2)])
    u, v = np.linalg.solve(np.column_stack((a, b)), rhs)
    r = _Jet1.constant(2*u-1, _SIZE)
    s = _Jet1.constant(2*v-1, _SIZE)
    for iteration in range(3):
        shape, dr, ds = _q1_first(r, s)
        mapped = _map_xy_first(shape, corners)
        first = _map_xy_first(dr, corners)
        second = _map_xy_first(ds, corners)
        determinant = _cross2(first, second)
        residual = _sub(mapped, point)
        delta_r = _cross2(residual, second)/determinant
        delta_s = _cross2(first, residual)/determinant
        r, s = r-delta_r, s-delta_s
        if iteration >= 1:
            shape, dr, ds = _q1_first(r, s)
            residual = _sub(_map_xy_first(shape, corners), point)
            if iteration == 1 and all(
                abs(x.value) <= 1e-12 and
                np.max(np.abs(x.gradient)) <= 1e-12 for x in residual
            ):
                return shape, dr, ds
    if any(abs(x.value) > 1e-11 or np.max(np.abs(x.gradient)) > 1e-9
           for x in residual):
        raise UnsupportedQ4S3Branch("Q1 projected inversion did not converge")
    return shape, dr, ds


def _zero() -> Jet2:
    return Jet2.constant(0.0, _SIZE)


def _scale(value: Jet2, factor: float) -> Jet2:
    """Scale an existing Jet2 without constructing a zero-derivative jet."""
    scalar = float(factor)
    return Jet2(value.value*scalar,
                value.gradient*scalar,
                value.hessian*scalar)


def _sum(items) -> Jet2:
    return sum(items, start=_zero())


def _sub(a, b):
    return [x-y for x, y in zip(a, b)]


def _dot(a, b):
    return _sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]]


def _cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _normalize(a):
    length = sqrt(_dot(a, a))
    return [x/length for x in a]


def _signed_area(poly):
    origin = poly[0]
    return _scale(_sum(_cross2(_sub(poly[i], origin),
                               _sub(poly[i+1], origin))
                       for i in range(1, len(poly)-1)), 0.5)


def _cross_resolution(a, b, first_error, second_error):
    """Conservative roundoff scale for a 2D determinant of projected points."""
    ax, ay = (abs(x.value) for x in a)
    bx, by = (abs(x.value) for x in b)
    eps = np.finfo(float).eps
    return (64*eps*(ax*by+ay*bx)
            + 2*first_error*(bx+by)
            + 2*second_error*(ax+ay)
            + 8*first_error*second_error)


def _area_resolution(poly, errors):
    origin = poly[0]
    return sum(
        _cross_resolution(_sub(poly[i], origin),
                          _sub(poly[i+1], origin),
                          errors[i]+errors[0], errors[i+1]+errors[0])/2
        for i in range(1, len(poly)-1)
    )


def _isolated_work_vertex(polygon, subject, target, coordinate_error, edge_index):
    """Resolve only an isolated original-vertex C1 event for a work query.

    The existing error envelope identifies candidates, not their sign. Exact
    predicates on the represented projected coordinates select the branch.
    At an exact vertex tie the two branch integrals differ by O(delta**2),
    so values and first derivatives agree, but Hessians need not agree.
    Generated vertices, target corners and edge ties remain unsupported.
    """
    from fractions import Fraction

    def distances(point, error, edge):
        start = target[edge]
        line = _sub(target[(edge + 1) % len(target)], start)
        relative = _sub(point, start)
        d = _cross2(line, relative)
        bound = _cross_resolution(line, relative, 2 * coordinate_error,
                                  error + coordinate_error)
        return d, bound

    values = [distances(point, error, edge_index) for point, error in polygon]
    near = [i for i, (d, error) in enumerate(values) if abs(d.value) <= error]
    if not near:
        return {}, None
    if len(near) != 1:
        raise UnsupportedQ4S3Branch("vertex lies on a clipping boundary")
    i = near[0]
    point, error = polygon[i]
    corner = next((k for k, item in enumerate(subject) if item is point), None)
    if corner is None:
        raise UnsupportedQ4S3Branch("generated vertex lies on a clipping boundary")
    for edge in range(len(target)):
        if edge != edge_index:
            d, bound = distances(point, error, edge)
            if abs(d.value) <= bound:
                raise UnsupportedQ4S3Branch("vertex lies on multiple clipping boundaries")
    # Use the endpoint coordinates themselves, not an already rounded cross
    # product or the width of the ambiguity band, for this represented sign.
    f = lambda value: Fraction.from_float(float(value.value))
    a, b = target[edge_index], target[(edge_index + 1) % len(target)]
    exact = ((f(b[0]) - f(a[0])) * (f(point[1]) - f(a[1]))
             - (f(b[1]) - f(a[1])) * (f(point[0]) - f(a[0])))
    sign = (exact > 0) - (exact < 0)
    if sign:
        inside = sign > 0
    else:
        inside = (values[(i - 1) % len(values)][0].value > 0
                  and values[(i + 1) % len(values)][0].value > 0)
    event = dict(kind="isolated_source_vertex_c1", source_corner=corner,
                 clipping_edge=edge_index, represented_sign=sign,
                 distance=float(values[i][0].value),
                 distance_error=float(values[i][1]), hessian_unique=False)
    return {id(point): bool(inside)}, event


def _clip(subject, target, coordinate_error, *, edge_count=3,
          _work_events=None):
    polygon = [(point, coordinate_error) for point in subject]
    signature = []
    for edge_index in range(edge_count):
        if not polygon:
            raise UnsupportedQ4S3Branch("zero projected overlap")
        work_classes = {}
        if _work_events is not None:
            work_classes, event = _isolated_work_vertex(
                polygon, subject, target, coordinate_error, edge_index)
            if event is not None:
                _work_events.append(event)
                # A negative marker cannot be confused with the strict
                # polygon-length / binary-inside signature entries.
                signature.extend((-71, edge_index, event["source_corner"],
                                  event["represented_sign"]))
        start = target[edge_index]
        edge = _sub(target[(edge_index+1) % edge_count], start)
        old, old_coordinate_error = polygon[-1]
        old_relative = _sub(old, start)
        old_distance = _cross2(edge, old_relative)
        old_error = _cross_resolution(
            edge, old_relative, 2*coordinate_error,
            old_coordinate_error+coordinate_error,
        )
        out = []
        signature.append(len(polygon))
        for point, point_coordinate_error in polygon:
            relative = _sub(point, start)
            distance = _cross2(edge, relative)
            error = _cross_resolution(
                edge, relative, 2*coordinate_error,
                point_coordinate_error+coordinate_error,
            )
            if (abs(old_distance.value) <= old_error
                    or abs(distance.value) <= error):
                if _work_events is None:
                    raise UnsupportedQ4S3Branch("vertex lies on a clipping boundary")
            old_inside = work_classes.get(id(old), old_distance.value > 0)
            inside = work_classes.get(id(point), distance.value > 0)
            signature.append(int(inside))
            if old_inside != inside:
                denominator = old_distance-distance
                denominator_error = (old_error+error
                                     + 64*np.finfo(float).eps*(
                                         abs(old_distance.value)+abs(distance.value)))
                if abs(denominator.value) <= denominator_error:
                    raise UnsupportedQ4S3Branch("degenerate clipping intersection")
                alpha = old_distance/denominator
                alpha_error = (
                    (old_error+abs(alpha.value)*denominator_error)
                    /(abs(denominator.value)-denominator_error)
                    + 16*np.finfo(float).eps
                )
                segment = _sub(point, old)
                intersection = [a+alpha*b for a, b in zip(old, segment)]
                segment_scale = max(abs(item.value) for item in segment)
                endpoint_scale = max(abs(item.value) for item in (*old, *point))
                intersection_error = (
                    old_coordinate_error
                    + abs(alpha.value)*(old_coordinate_error+point_coordinate_error)
                    + alpha_error*segment_scale
                    + 8*np.finfo(float).eps*endpoint_scale
                )
                out.append((intersection, intersection_error))
            if inside:
                out.append((point, point_coordinate_error))
            old, old_coordinate_error = point, point_coordinate_error
            old_distance, old_error = distance, error
        polygon = out
    signature.append(len(polygon))
    return ([point for point, _ in polygon],
            [error for _, error in polygon], tuple(signature))


def _q1(r, s):
    one = Jet2.constant(1.0, _SIZE)
    shape = [_scale((one-r)*(one-s), 0.25),
             _scale((one+r)*(one-s), 0.25),
             _scale((one+r)*(one+s), 0.25),
             _scale((one-r)*(one+s), 0.25)]
    dr = [_scale(-(one-s), 0.25), _scale(one-s, 0.25),
          _scale(one+s, 0.25), _scale(-(one+s), 0.25)]
    ds = [_scale(-(one-r), 0.25), _scale(-(one+r), 0.25),
          _scale(one+r, 0.25), _scale(one-r, 0.25)]
    return shape, dr, ds


def _map_xy(shape, corners):
    return [_sum(shape[i]*corners[i][k] for i in range(4)) for k in range(2)]


def _natural(point, corners):
    # At an affine base pose, this value solve is exact.  Jet Newton iterations
    # then solve the first and second variations of the bilinear inverse.
    a = np.array([corners[1][k].value-corners[0][k].value for k in range(2)])
    b = np.array([corners[3][k].value-corners[0][k].value for k in range(2)])
    rhs = np.array([point[k].value-corners[0][k].value for k in range(2)])
    u, v = np.linalg.solve(np.column_stack((a, b)), rhs)
    r = Jet2.constant(2*u-1, _SIZE)
    s = Jet2.constant(2*v-1, _SIZE)
    for iteration in range(3):
        shape, dr, ds = _q1(r, s)
        mapped = _map_xy(shape, corners)
        first = _map_xy(dr, corners)
        second = _map_xy(ds, corners)
        determinant = _cross2(first, second)
        residual = _sub(mapped, point)
        delta_r = _cross2(residual, second)/determinant
        delta_s = _cross2(first, residual)/determinant
        r, s = r-delta_r, s-delta_s
        # For an affine base pose the starting value is exact.  Two Newton
        # updates then resolve the first and second Jet2 variations.  Skip
        # the third update only once the full residual is at roundoff scale.
        if iteration >= 1:
            shape, dr, ds = _q1(r, s)
            residual = _sub(_map_xy(shape, corners), point)
            if iteration == 1 and all(
                abs(x.value) <= 1e-12 and
                np.max(np.abs(x.gradient)) <= 1e-12 and
                np.max(np.abs(x.hessian)) <= 1e-12 for x in residual
            ):
                return shape, dr, ds
    if any(abs(x.value) > 1e-11 or np.max(np.abs(x.gradient)) > 1e-9
           or np.max(np.abs(x.hessian)) > 1e-7 for x in residual):
        raise UnsupportedQ4S3Branch("Q1 projected inversion did not converge")
    return shape, dr, ds


def _target_shape(point, corners):
    a = _sub(corners[1], corners[0])
    b = _sub(corners[2], corners[0])
    p = _sub(point, corners[0])
    denominator = _cross2(a, b)
    u = _cross2(p, b)/denominator
    v = _cross2(a, p)/denominator
    return [1-u-v, u, v]


def _parts(jets):
    return (np.array([x.value for x in jets]),
            np.stack([x.gradient for x in jets]),
            np.stack([x.hessian for x in jets]))


def _validate(source_reference, source_current, target_current,
              relative_branch_tolerance, enforce_affine):
    reference = np.asarray(source_reference, dtype=float)
    source = np.asarray(source_current, dtype=float)
    target = np.asarray(target_current, dtype=float)
    if (reference.shape != (4, 3) or source.shape != (4, 3)
            or target.shape != (3, 3)
            or not all(np.all(np.isfinite(x)) for x in (reference, source, target))):
        raise ValueError("Q4/S3 overlap needs finite (4,3), (4,3), (3,3) positions")
    if not np.isfinite(relative_branch_tolerance) or relative_branch_tolerance <= 0:
        raise ValueError("relative_branch_tolerance must be finite and positive")
    for name, quad in (("reference", reference), ("current", source)):
        span = max(np.linalg.norm(quad[i]-quad[0]) for i in (1, 2, 3))
        if span <= 0:
            raise UnsupportedQ4S3Branch(f"{name} Q4 is degenerate")
        cross = np.cross(quad[1]-quad[0], quad[3]-quad[0])
        if np.linalg.norm(cross) <= 1e-12*span*span:
            raise UnsupportedQ4S3Branch(f"{name} Q4 is degenerate")
        if ((name == "reference" or enforce_affine) and
                np.linalg.norm(quad[0]-quad[1]+quad[2]-quad[3])
                > relative_branch_tolerance*span):
            raise UnsupportedQ4S3Branch(f"{name} Q4 is warped or non-affine")
    target_span = max(np.linalg.norm(target[i]-target[0]) for i in (1, 2))
    if (target_span <= 0 or np.linalg.norm(np.cross(target[1]-target[0],
                                                    target[2]-target[0]))
            <= 1e-12*target_span*target_span):
        raise UnsupportedQ4S3Branch("S3 target is degenerate")
    return reference, source, target


def _evaluate(reference_source, current_source, current_target, *,
              area_tolerance, weighted_gap_tolerance, maximum_triangles,
              relative_branch_tolerance, enforce_affine,
              derivative_tolerances=None, first_order=False):
    if first_order:
        _zero = _zero_first
        _scale = _scale_first
        _sum = _sum_first
        _dot = _dot_first
        _normalize = _normalize_first
        _signed_area = _signed_area_first
        _natural = _natural_first
        _parts = lambda jets: (np.array([x.value for x in jets]),
                               np.stack([x.gradient for x in jets]))
        jet_type = _Jet1
    else:
        _zero = globals()["_zero"]
        _scale = globals()["_scale"]
        _sum = globals()["_sum"]
        _dot = globals()["_dot"]
        _normalize = globals()["_normalize"]
        _signed_area = globals()["_signed_area"]
        _natural = globals()["_natural"]
        _parts = globals()["_parts"]
        jet_type = Jet2
    reference, source, target = _validate(
        reference_source, current_source, current_target,
        relative_branch_tolerance, enforce_affine,
    )
    if (not np.isfinite(area_tolerance) or area_tolerance <= 0
            or not np.isfinite(weighted_gap_tolerance)
            or weighted_gap_tolerance <= 0
            or type(maximum_triangles) is not int or maximum_triangles < 1):
        raise ValueError("integration tolerances and triangle budget must be positive")
    if not enforce_affine and derivative_tolerances is None:
        raise ValueError("warped Q4 needs explicit derivative tolerances")
    values = np.concatenate((source, target)).reshape(-1)
    points = [[jet_type.variable(values[3*i+k], 3*i+k, _SIZE) for k in range(3)]
              for i in range(7)]
    q, t = points[:4], points[4:]
    e1 = _sub(q[1], q[0])
    e3 = _sub(q[3], q[0])
    normal = _normalize(_cross(e1, e3))
    for i in range(4):
        first = _sub(q[(i+1) % 4], q[i])
        second = _sub(q[(i-1) % 4], q[i])
        if _dot(_cross(first, second), normal).value <= 0:
            raise UnsupportedQ4S3Branch("Q4 is folded or has inverted bilinear geometry")
    ref_cross = np.cross(reference[1]-reference[0],
                         reference[3]-reference[0])
    if np.dot(ref_cross, np.array([x.value for x in normal])) <= 0:
        raise UnsupportedQ4S3Branch("Q4 polarity differs from reference")
    target_cross = _cross(_sub(t[1], t[0]), _sub(t[2], t[0]))
    if _dot(target_cross, normal).value <= 0:
        raise UnsupportedQ4S3Branch("S3 target faces away")
    tangent = _normalize(e1)
    binormal = _cross(normal, tangent)

    def projected(point):
        delta = _sub(point, q[0])
        return [_dot(delta, tangent), _dot(delta, binormal)]

    quad_xy = [projected(x) for x in q]
    triangle_xy = [projected(x) for x in t]
    scale = max(np.hypot(x[0].value, x[1].value)
                for x in (*quad_xy, *triangle_xy))
    face_threshold = relative_branch_tolerance*scale*scale
    raw_scale = max(np.max(np.abs(source)), np.max(np.abs(target)), scale)
    coordinate_error = 64*np.finfo(float).eps*raw_scale
    if _signed_area(quad_xy).value <= face_threshold:
        raise UnsupportedQ4S3Branch("Q4 projection is degenerate or inverted")
    for i in range(4):
        edge1 = _sub(quad_xy[(i+1) % 4], quad_xy[i])
        edge2 = _sub(quad_xy[(i+2) % 4], quad_xy[(i+1) % 4])
        if _cross2(edge1, edge2).value <= face_threshold:
            raise UnsupportedQ4S3Branch("Q4 projection is folded or degenerate")
    if _signed_area(triangle_xy).value <= face_threshold:
        raise UnsupportedQ4S3Branch("S3 projection is degenerate or inverted")
    polygon, polygon_errors, clipping_signature = _clip(
        quad_xy, triangle_xy, coordinate_error,
    )
    if len(polygon) < 3:
        raise UnsupportedQ4S3Branch("zero projected overlap")
    overlap_area = _signed_area(polygon)
    overlap_area_resolution = _area_resolution(polygon, polygon_errors)
    if overlap_area.value <= overlap_area_resolution:
        raise UnsupportedQ4S3Branch("zero projected overlap")

    ref_area_jacobian = np.linalg.norm(ref_cross)/4
    batch_context = None if first_order else _BatchContext(
        q, t, normal, quad_xy, triangle_xy,
    )
    def rule(vertices, vertex_errors):
        area = _scale(_cross2(_sub(vertices[1], vertices[0]),
                              _sub(vertices[2], vertices[0])), 0.5)
        if area.value <= _area_resolution(vertices, vertex_errors):
            raise UnsupportedQ4S3Branch("degenerate integration triangle")
        if not first_order:
            return _rule_batched(
                [vertices], [area], batch_context, face_threshold,
                ref_area_jacobian,
            )[0]
        total = [_zero() for _ in range(24)]
        for bary, relative_weight in _TRIANGLE_RULE:
            xy = [_sum(_scale(vertices[j][k], bary[j]) for j in range(3))
                  for k in range(2)]
            shape, dr, ds = _natural(xy, quad_xy)
            jr = [_sum(dr[i]*q[i][k] for i in range(4)) for k in range(3)]
            js = [_sum(ds[i]*q[i][k] for i in range(4)) for k in range(3)]
            cross = _cross(jr, js)
            projected_jacobian = _dot(cross, normal)
            if projected_jacobian.value <= face_threshold/4:
                raise UnsupportedQ4S3Branch("Q4 projected Jacobian is near singular")
            physical_normal = _normalize(cross)
            source_point = [_sum(shape[i]*q[i][k] for i in range(4))
                            for k in range(3)]
            target_shape = _target_shape(xy, triangle_xy)
            target_point = [_sum(target_shape[i]*t[i][k] for i in range(3))
                            for k in range(3)]
            gap = _dot(physical_normal, _sub(source_point, target_point))
            weight = _scale(_scale(area, relative_weight),
                            ref_area_jacobian)/projected_jacobian
            integrand = ([weight*x for x in shape]
                         + [weight*x*gap for x in shape]
                         + [weight*x*y for x in shape for y in shape])
            total = [a+b for a, b in zip(total, integrand)]
        return total

    def rule_many(vertex_sets, error_sets):
        areas = []
        for vertices, vertex_errors in zip(vertex_sets, error_sets):
            area = _scale(_cross2(_sub(vertices[1], vertices[0]),
                                  _sub(vertices[2], vertices[0])), 0.5)
            if area.value <= _area_resolution(vertices, vertex_errors):
                raise UnsupportedQ4S3Branch("degenerate integration triangle")
            areas.append(area)
        return _rule_batched(
            vertex_sets, areas, batch_context, face_threshold,
            ref_area_jacobian,
        )

    total = ([_zero() for _ in range(24)] if first_order else
             [_zero() for _ in range(8)] + [_FloatOnly(0.0) for _ in range(16)])
    integration_triangles = 0
    visited_triangles = 0
    quadrature_signature: list[int] = []
    derivative_differences = np.zeros(2 if first_order else 4)
    roots = len(polygon)-2
    def integrate(vertices, vertex_errors, area_budget, gap_budget,
                  derivative_budgets):
        nonlocal integration_triangles, visited_triangles
        visited_triangles += 1
        if visited_triangles > maximum_triangles:
            raise UnsupportedQ4S3Branch("adaptive integration exceeds triangle budget")
        a, b, c = vertices
        if first_order:
            coarse = rule(vertices, vertex_errors)
        ab = [_scale(x+y, 0.5) for x, y in zip(a, b)]
        bc = [_scale(x+y, 0.5) for x, y in zip(b, c)]
        ca = [_scale(x+y, 0.5) for x, y in zip(c, a)]
        children = ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
        eps = np.finfo(float).eps
        def midpoint_error(first, second, first_error, second_error):
            return ((first_error+second_error)/2
                    + 4*eps*max(abs(x.value) for x in (*first, *second)))
        ea, eb, ec = vertex_errors
        eab = midpoint_error(a, b, ea, eb)
        ebc = midpoint_error(b, c, eb, ec)
        eca = midpoint_error(c, a, ec, ea)
        children_errors = ((ea, eab, eca), (eab, eb, ebc),
                           (eca, ebc, ec), (eab, ebc, eca))
        if first_order:
            refined = [_zero() for _ in range(24)]
            for child, child_errors in zip(children, children_errors):
                refined = [x+y for x, y in zip(refined, rule(child, child_errors))]
        else:
            coarse, *evaluated_children = rule_many(
                [vertices, *children], [vertex_errors, *children_errors],
            )
            refined = [_zero() for _ in range(8)] + [
                _FloatOnly(0.0) for _ in range(16)
            ]
            for evaluated_child in evaluated_children:
                refined = [x+y for x, y in zip(refined, evaluated_child)]
        area_error = max(abs(refined[i].value-coarse[i].value)
                         for i in (*range(4), *range(8, 24)))
        gap_error = max(abs(refined[i].value-coarse[i].value)
                        for i in range(4, 8))
        support_gradient_error = max(
            np.max(np.abs(refined[i].gradient-coarse[i].gradient))
            for i in range(4))
        gap_gradient_error = max(
            np.max(np.abs(refined[i].gradient-coarse[i].gradient))
            for i in range(4, 8))
        if first_order:
            derivative_errors = (support_gradient_error, gap_gradient_error)
        else:
            derivative_errors = (
                support_gradient_error,
                max(np.max(np.abs(refined[i].hessian-coarse[i].hessian))
                    for i in range(4)),
                gap_gradient_error,
                max(np.max(np.abs(refined[i].hessian-coarse[i].hessian))
                    for i in range(4, 8)),
            )
        if (abs(area_error-area_budget) <= 1e-8*area_budget
                or abs(gap_error-gap_budget) <= 1e-8*gap_budget):
            raise UnsupportedQ4S3Branch("adaptive integration decision is ambiguous")
        if (derivative_budgets is not None and any(
            abs(error-budget) <= 1e-8*budget
            for error, budget in zip(derivative_errors, derivative_budgets)
        )):
            raise UnsupportedQ4S3Branch("derivative integration decision is ambiguous")
        derivative_resolved = (derivative_budgets is None or all(
            error <= budget for error, budget in
            zip(derivative_errors, derivative_budgets)
        ))
        if (area_error <= area_budget and gap_error <= gap_budget
                and derivative_resolved):
            quadrature_signature.append(0)
            if enforce_affine:
                derivative_error = max(derivative_errors)
                if first_order:
                    derivative_scale = max(
                        1.0, *(np.max(np.abs(x.gradient))
                               for x in refined[:8]))
                else:
                    derivative_scale = max(
                        1.0, *(max(np.max(np.abs(x.gradient)),
                                    np.max(np.abs(x.hessian)))
                               for x in refined[:8]))
                if derivative_error > 1e-10*derivative_scale:
                    raise UnsupportedQ4S3Branch(
                        "quadrature does not resolve the support/gap derivatives"
                    )
            integration_triangles += 4
            if integration_triangles > maximum_triangles:
                raise UnsupportedQ4S3Branch("adaptive integration exceeds triangle budget")
            derivative_differences[:] += derivative_errors
            return refined
        quadrature_signature.append(1)
        if integration_triangles+4 > maximum_triangles:
            raise UnsupportedQ4S3Branch("adaptive integration exceeds triangle budget")
        integrated = ([_zero() for _ in range(24)] if first_order else
                      [_zero() for _ in range(8)] +
                      [_FloatOnly(0.0) for _ in range(16)])
        for child, child_errors in zip(children, children_errors):
            integrated = [x+y for x, y in zip(
                integrated, integrate(
                    child, child_errors, area_budget/4, gap_budget/4,
                    None if derivative_budgets is None else
                    tuple(budget/4 for budget in derivative_budgets),
                )
            )]
        return integrated

    for index in range(1, len(polygon)-1):
        vertices = (polygon[0], polygon[index], polygon[index+1])
        vertex_errors = (polygon_errors[0], polygon_errors[index],
                         polygon_errors[index+1])
        total = [x+y for x, y in zip(total, integrate(
            vertices, vertex_errors, area_tolerance/roots,
            weighted_gap_tolerance/roots,
            None if derivative_tolerances is None else
            tuple(tolerance/roots for tolerance in derivative_tolerances),
        ))]
    if first_order:
        support, support_gradient = _parts(total[:4])
        gap, gap_gradient = _parts(total[4:8])
        arrays = (support, support_gradient, gap, gap_gradient)
    else:
        support, support_gradient, support_hessian = _parts(total[:4])
        gap, gap_gradient, gap_hessian = _parts(total[4:8])
        arrays = (support, support_gradient, support_hessian,
                  gap, gap_gradient, gap_hessian)
    if np.min(support) <= 0 or not all(np.all(np.isfinite(x)) for x in arrays):
        raise UnsupportedQ4S3Branch("nonpositive support or nonfinite derivative")
    if first_order:
        return Q4S3OverlapFirstDerivatives(
            support, gap, support_gradient, gap_gradient,
            overlap_area.value, overlap_area_resolution,
            float(np.sum(support)),
            clipping_signature+tuple(quadrature_signature),
            integration_triangles, derivative_differences,
            (area_tolerance, weighted_gap_tolerance,
             *derivative_tolerances) if derivative_tolerances is not None else
            (area_tolerance, weighted_gap_tolerance, None, None),
            1e-10 if enforce_affine else None,
        )
    return Q4S3OverlapDerivatives(
        support, gap, support_gradient, support_hessian,
        gap_gradient, gap_hessian, overlap_area.value,
        overlap_area_resolution,
        float(np.sum(support)), clipping_signature+tuple(quadrature_signature),
        integration_triangles, derivative_differences,
    )


def q4_s3_overlap_derivatives(
    reference_source: np.ndarray, current_source: np.ndarray,
    current_target: np.ndarray, *, area_tolerance: float = 1e-10,
    weighted_gap_tolerance: float = 1e-10, maximum_triangles: int = 32,
    relative_branch_tolerance: float = 1e-10,
    allow_warped_source: bool = False,
    support_gradient_tolerance: float | None = None,
    support_hessian_tolerance: float | None = None,
    weighted_gap_gradient_tolerance: float | None = None,
    weighted_gap_hessian_tolerance: float | None = None,
) -> Q4S3OverlapDerivatives:
    """Differentiate source Q1 reference support/gap in 4+3 xyz node order.

    The reference Q4 must be affine.  The derivative is of the smooth physical
    Q1 and accepted quadrature branch.  Warped current poses require explicit
    opt-in and four positive derivative-resolution tolerances; their integral
    and derivative accuracy depends on those controls and the triangle budget.
    ``relative_branch_tolerance`` qualifies whole-face geometry; local
    determinant resolution governs tiny overlap branches.  Clipping and
    quadrature decision ties are unsupported.  Polygon fans create no
    physical faces.
    """
    if type(allow_warped_source) is not bool:
        raise ValueError("allow_warped_source must be boolean")
    derivative_tolerances = None
    if allow_warped_source:
        derivative_tolerances = (
            support_gradient_tolerance, support_hessian_tolerance,
            weighted_gap_gradient_tolerance, weighted_gap_hessian_tolerance,
        )
        if any(item is None or not np.isfinite(item) or item <= 0
               for item in derivative_tolerances):
            raise ValueError("warped Q4 needs four positive derivative tolerances")
    return _evaluate(
        reference_source, current_source, current_target,
        area_tolerance=area_tolerance,
        weighted_gap_tolerance=weighted_gap_tolerance,
        maximum_triangles=maximum_triangles,
        relative_branch_tolerance=relative_branch_tolerance,
        enforce_affine=not allow_warped_source,
        derivative_tolerances=derivative_tolerances,
    )


def q4_s3_overlap_first_derivatives(
    reference_source: np.ndarray, current_source: np.ndarray,
    current_target: np.ndarray, *, area_tolerance: float = 1e-10,
    weighted_gap_tolerance: float = 1e-10, maximum_triangles: int = 32,
    relative_branch_tolerance: float = 1e-10,
    allow_warped_source: bool = False,
    support_gradient_tolerance: float | None = None,
    weighted_gap_gradient_tolerance: float | None = None,
) -> Q4S3OverlapFirstDerivatives:
    """Evaluate the smooth Q4/S3 overlap value and gradient without Hessians.

    For warped sources, both gradient-resolution tolerances are required.
    The adaptive quadrature branch may differ from the second-order evaluator
    because Hessian resolution does not govern this first-order result.
    Clipping, projection, geometry rejection and triangle-budget semantics are
    shared with the second-order evaluator.
    """
    if type(allow_warped_source) is not bool:
        raise ValueError("allow_warped_source must be boolean")
    derivative_tolerances = None
    if allow_warped_source:
        derivative_tolerances = (support_gradient_tolerance,
                                 weighted_gap_gradient_tolerance)
        if any(item is None or not np.isfinite(item) or item <= 0
               for item in derivative_tolerances):
            raise ValueError("warped Q4 needs two positive gradient tolerances")
    return _evaluate(
        reference_source, current_source, current_target,
        area_tolerance=area_tolerance,
        weighted_gap_tolerance=weighted_gap_tolerance,
        maximum_triangles=maximum_triangles,
        relative_branch_tolerance=relative_branch_tolerance,
        enforce_affine=not allow_warped_source,
        derivative_tolerances=derivative_tolerances,
        first_order=True,
    )


class _BatchContext:
    def __init__(self, q, t, normal, quad_xy, triangle_xy):
        self.q = [[_BJet.of(x) for x in p] for p in q]
        self.t = [[_BJet.of(x) for x in p] for p in t]
        self.normal = [_BJet.of(x) for x in normal]
        self.quad_xy = [[_BJet.of(x) for x in p] for p in quad_xy]
        self.quad_xy_jets = quad_xy
        self.triangle_xy = [[_BJet.of(x) for x in p] for p in triangle_xy]


def _bq1(r, s):
    shape = [((1-r)*(1-s)).scale(0.25), ((1+r)*(1-s)).scale(0.25),
             ((1+r)*(1+s)).scale(0.25), ((1-r)*(1+s)).scale(0.25)]
    dr = [(-(1-s)).scale(0.25), (1-s).scale(0.25), (1+s).scale(0.25), (-(1+s)).scale(0.25)]
    ds = [(-(1-r)).scale(0.25), (-(1+r)).scale(0.25), (1+r).scale(0.25), (1-r).scale(0.25)]
    return shape, dr, ds


def _bsum(items):
    items = list(items)
    total = items[0]
    for x in items[1:]:
        total = total + x
    return total


def _bmap(shape, corners):
    return [_bsum(shape[i]*corners[i][k] for i in range(4)) for k in range(2)]


def _bcross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _bnatural(point, ctx):
    corners_j = ctx.quad_xy_jets
    corners = ctx.quad_xy
    a = np.array([corners_j[1][k].value-corners_j[0][k].value for k in range(2)])
    b = np.array([corners_j[3][k].value-corners_j[0][k].value for k in range(2)])
    rhs = np.stack([point[k].v-corners_j[0][k].value for k in range(2)])
    uv = np.linalg.solve(np.column_stack((a, b)), rhs)
    size = point[0].g.shape[1]
    r = _BJet.const(2*uv[0]-1, size)
    s = _BJet.const(2*uv[1]-1, size)
    done = np.zeros(len(r.v), bool)
    saved = None
    for iteration in range(3):
        shape, dr, ds = _bq1(r, s)
        mapped = _bmap(shape, corners)
        first = _bmap(dr, corners)
        second = _bmap(ds, corners)
        determinant = _bcross2(first, second)
        residual = [m-p for m, p in zip(mapped, point)]
        delta_r = _bcross2(residual, second)/determinant
        delta_s = _bcross2(first, residual)/determinant
        r, s = r-delta_r, s-delta_s
        if iteration >= 1:
            shape, dr, ds = _bq1(r, s)
            residual = [m-p for m, p in zip(_bmap(shape, corners), point)]
            if iteration == 1:
                tiny = np.ones(len(r.v), bool)
                for x in residual:
                    tiny &= ((np.abs(x.v) <= 1e-12)
                             & (np.max(np.abs(x.g), axis=1) <= 1e-12)
                             & (np.max(np.abs(x.h), axis=(1, 2)) <= 1e-12))
                if np.all(tiny):
                    return shape, dr, ds
                done = tiny
                saved = (shape, dr, ds, residual)
    if saved is not None and np.any(done):
        s0, d0, e0, res0 = saved
        shape = [x.select(done, y) for x, y in zip(s0, shape)]
        dr = [x.select(done, y) for x, y in zip(d0, dr)]
        ds = [x.select(done, y) for x, y in zip(e0, ds)]
        residual = [x.select(done, y) for x, y in zip(res0, residual)]
    for x in residual:
        if (np.any(np.abs(x.v) > 1e-11) or np.any(np.max(np.abs(x.g), axis=1) > 1e-9)
                or np.any(np.max(np.abs(x.h), axis=(1, 2)) > 1e-7)):
            raise UnsupportedQ4S3Branch("Q1 projected inversion did not converge")
    return shape, dr, ds


def _cat(jets):
    return _BJet(np.concatenate([j.v for j in jets]), np.concatenate([j.g for j in jets]),
                np.concatenate([j.h for j in jets]))


def _rule_batched(vertex_sets, areas, ctx, face_threshold, ref_area_jacobian):
    m = len(vertex_sets)
    bary = np.asarray([point for point, _ in _TRIANGLE_RULE], dtype=float)
    relative_weights = np.asarray([weight for _, weight in _TRIANGLE_RULE], dtype=float)
    npt = len(relative_weights)
    xy = []
    for k in range(2):
        parts = []
        for vertices in vertex_sets:
            V = [_BJet.of(p[k]) for p in vertices]
            parts.append(_bsum(V[j].scale(bary[:, j]) for j in range(3)))
        xy.append(_cat(parts))
    shape, dr, ds = _bnatural(xy, ctx)
    q, t, normal = ctx.q, ctx.t, ctx.normal
    jr = [_bsum(dr[i]*q[i][k] for i in range(4)) for k in range(3)]
    js = [_bsum(ds[i]*q[i][k] for i in range(4)) for k in range(3)]
    cross = [jr[1]*js[2]-jr[2]*js[1], jr[2]*js[0]-jr[0]*js[2], jr[0]*js[1]-jr[1]*js[0]]
    projected_jacobian = _bsum(x*y for x, y in zip(cross, normal))
    if np.any(projected_jacobian.v <= face_threshold/4):
        raise UnsupportedQ4S3Branch("Q4 projected Jacobian is near singular")
    length = _bsum(x*x for x in cross).sqrt()
    physical_normal = [x/length for x in cross]
    source_point = [_bsum(shape[i]*q[i][k] for i in range(4)) for k in range(3)]
    ta = [x-y for x, y in zip(ctx.triangle_xy[1], ctx.triangle_xy[0])]
    tb = [x-y for x, y in zip(ctx.triangle_xy[2], ctx.triangle_xy[0])]
    tp = [x-y for x, y in zip(xy, ctx.triangle_xy[0])]
    den = _bcross2(ta, tb)
    u = _bcross2(tp, tb)/den
    v = _bcross2(ta, tp)/den
    target_shape = [1-u-v, u, v]
    target_point = [_bsum(target_shape[i]*t[i][k] for i in range(3)) for k in range(3)]
    gap = _bsum(x*(y-z) for x, y, z in zip(physical_normal, source_point, target_point))
    area_points = _cat([_BJet(np.repeat(a.value, npt), np.repeat(a.gradient[None], npt, 0),
                             np.repeat(a.hessian[None], npt, 0)) for a in areas])
    weight = area_points.scale(ref_area_jacobian) / projected_jacobian
    weight = weight.scale(np.tile(relative_weights, m))
    wgap = weight*gap
    ws = [weight*x for x in shape]
    wg = [wgap*x for x in shape]
    out = []
    for block in range(m):
        sel = np.zeros(m*npt); sel[block*npt:(block+1)*npt] = 1.0
        sl = slice(block*npt, (block+1)*npt)
        out.append([x.weighted_sum(sel) for x in ws] + [x.weighted_sum(sel) for x in wg]
                   + [_FloatOnly(np.sum(weight.v[sl]*x.v[sl]*y.v[sl])) for x in shape for y in shape])
    return out
