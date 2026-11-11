"""Exact B2 pair geometry for the developmental physical-end contact profile.

Coordinates may be floats or Fractions; Fractions are never rounded on input.
The path certificate checks endpoint feature branches only. The caller retains
the separate physical-curve regularity and distributed-line branch checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from .beam_temporal_bounds import _outward, _split_curve
from .capped_cylinder import _elevate, _product, _sqrt_bound


def _rational(value):
    return value if isinstance(value, Fraction) else Fraction(value)


def _segment(values):
    result = tuple(tuple(_rational(x) for x in row) for row in values)
    if len(result) != 2 or any(len(row) != 3 for row in result):
        raise ValueError('B2 geometry requires two three-dimensional points')
    return result


def _radius(value):
    value = _rational(value)
    if value < 0:
        raise ValueError('combined radius must be nonnegative')
    return value


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Fraction(0))


def _clamp(value):
    return min(max(value, Fraction(0)), Fraction(1))


def minimum_gap(a, b, radius):
    """Outward lower float bound on the exact rectangle minimum distance-r.

    All four edges, and the interior stationary point when its determinant is
    positive, are considered with rational arithmetic. Parallel and collapsed
    segments retain the complete edge candidates.
    """
    a, b, radius = _segment(a), _segment(b), _radius(radius)
    u, v, w = _sub(a[1], a[0]), _sub(b[1], b[0]), _sub(a[0], b[0])
    aa, bb, ab = _dot(u, u), _dot(v, v), _dot(u, v)
    uw, vw = _dot(u, w), _dot(v, w)

    def squared(s, t):
        return sum((x+s*y-t*z)**2 for x, y, z in zip(w, u, v))

    candidates = [squared(Fraction(s), _clamp((vw+s*ab)/bb) if bb else Fraction(0))
                  for s in (0, 1)]
    candidates += [squared(_clamp((t*ab-uw)/aa) if aa else Fraction(0), Fraction(t))
                   for t in (0, 1)]
    determinant = aa*bb-ab*ab
    if determinant > 0:
        s, t = (ab*vw-bb*uw)/determinant, (aa*vw-ab*uw)/determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(squared(s, t))
    return _outward(_sqrt_bound(min(candidates))-radius, -math.inf)


@dataclass(frozen=True)
class EndpointFeature:
    """Exact closest feature plus a cancellation-resistant numerical gap.

    ``projection`` is the clamped rational target parameter; ``ownership`` is
    (source point-target projection) dot the source outward, unnormalized
    tangent. Negative ownership means the source cap owns the feature.
    ``target_endpoint`` is 0/1 on a clamp and None in the strict interior.
    ``projection_tie`` records raw h==0 or h==bb, including open features.
    ``gap_sign`` is the exact sign of distance-radius; ``gap`` uses rationalized
    subtraction. ``squared_distance`` and ``separation`` remain exact.
    """
    projection: Fraction
    ownership: Fraction
    squared_distance: Fraction
    gap: float
    gap_sign: int
    interior: bool
    target_endpoint: int | None
    projection_tie: bool
    separation: tuple[Fraction, ...]


def endpoint_feature(source, target, endpoint, radius):
    """Return EndpointFeature for source physical-end index 0 or 1.

    A collapsed target raises ValueError. Topological physical-end membership
    and active tie rejection are caller decisions, using the exact fields.
    """
    source, target, radius = _segment(source), _segment(target), _radius(radius)
    if endpoint not in (0, 1):
        raise ValueError('B2 endpoint index must be 0 or 1')
    v, w = _sub(target[1], target[0]), _sub(source[endpoint], target[0])
    outward = _sub(source[endpoint], source[1-endpoint])
    bb, h = _dot(v, v), _dot(w, v)
    if bb <= 0:
        raise ValueError('endpoint target must have positive squared length')
    projection = _clamp(h/bb)
    separation = tuple(x-projection*y for x, y in zip(w, v))
    squared = _dot(separation, separation)
    delta = squared-radius*radius
    sign = (delta > 0)-(delta < 0)
    denominator = _sqrt_bound(squared)+radius
    gap = float(delta/denominator) if denominator else 0.
    if gap == 0. and sign:
        gap = math.nextafter(0., math.inf if sign > 0 else -math.inf)
    interior = 0 < h < bb
    return EndpointFeature(projection, _dot(separation, outward), squared, gap,
                           sign, interior, None if interior else int(h >= bb),
                           h == 0 or h == bb, separation)


def _add(first, second, scale=1):
    degree = max(len(first), len(second))-1
    return tuple(a+scale*b for a, b in zip(_elevate(first, degree), _elevate(second, degree)))


def _dot_poly(first, second):
    result = (Fraction(0),)
    for a, b in zip(first, second):
        result = _add(result, _product(a, b))
    return result


def _affine(first, last):
    return tuple((a, b) for a, b in zip(first, last))


def _vector_sub(first, second):
    return tuple(_add(a, b, -1) for a, b in zip(first, second))


def admissible_endpoint_path(a0, a1, b0, b1, ends_a, ends_b, radius, max_depth):
    """Conservative exact Bernstein certificate for all declared B2 ends.

    Every interval must be open or have a strict, fixed projection branch and
    strict ownership sign. An owned active clamp requires a genuine target end.
    Opening alone may bypass feature ties. No sampled value certifies success;
    samples only reject definite inadmissibility. Budget exhaustion returns False.
    Positive target squared length must also be proved. Curve regularity remains
    a separate caller gate (e.g. regular_curve_path).
    """
    a0, a1, b0, b1 = map(_segment, (a0, a1, b0, b1))
    radius = _radius(radius)
    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError('max_depth must be a nonnegative integer')
    ends_a, ends_b = tuple(ends_a), tuple(ends_b)
    if any(i not in (0, 1) for i in (*ends_a, *ends_b)):
        raise ValueError('physical B2 ends must be indices 0 or 1')
    first = tuple(_affine(x, y) for x, y in zip(a0, a1))
    second = tuple(_affine(x, y) for x, y in zip(b0, b1))

    def direction(source, target, endpoint, target_ends):
        w = _vector_sub(source[endpoint], target[0])
        v = _vector_sub(target[1], target[0])
        outward = _vector_sub(source[endpoint], source[1-endpoint])
        h, bb, ww = _dot_poly(w, v), _dot_poly(v, v), _dot_poly(w, w)
        wa, va = _dot_poly(w, outward), _dot_poly(v, outward)
        hb = _add(h, bb, -1)
        end_one = _add(_add(ww, h, -2), bb)
        line = _add(_product(ww, bb), _product(h, h), -1)
        interior_owner = _add(_product(wa, bb), _product(h, va), -1)
        line_open = _add(line, tuple(radius*radius*x for x in bb), -1)
        zero_open = _add(ww, (radius*radius,), -1)
        one_open = _add(end_one, (radius*radius,), -1)
        one_owner = _add(wa, va, -1)
        # Coordinate hull of p-q for every q on the entire moving segment.
        far_w = _vector_sub(w, v)
        polynomials = (bb, h, hb, line_open, zero_open, one_open,
                       wa, one_owner, interior_owner, *w, *far_w)

        def certify(values, depth):
            bb, h, hb, line_open, zero_open, one_open, wa, one_owner, owner = values[:9]
            if bb[0] <= 0 or bb[-1] <= 0:
                return False
            positive_length = min(bb) > 0
            if positive_length:
                near = [Fraction(0) if min((*x, *y)) <= 0 <= max((*x, *y))
                        else min(abs(z) for z in (*x, *y))
                        for x, y in zip(values[9:12], values[12:15])]
                if sum(x*x for x in near) >= radius*radius or min(line_open) >= 0:
                    return True
                branch = 0 if max(h) < 0 else 1 if min(hb) > 0 else 2 if min(h) > 0 and max(hb) < 0 else None
                if branch is not None:
                    gap_poly = (zero_open, one_open, line_open)[branch]
                    own_poly = (wa, one_owner, owner)[branch]
                    if min(gap_poly) >= 0 or min(own_poly) > 0:
                        return True
                    if max(own_poly) < 0 and (branch == 2 or branch in target_ends):
                        return True
            # Exact endpoint/midpoint values may disprove admissibility, never
            # replace a whole-interval polynomial enclosure.
            split = tuple(_split_curve(poly) for poly in values)
            for index in (0, -1):
                if not point_ok(tuple(poly[index] for poly in values)):
                    return False
            if not point_ok(tuple(pair[0][-1] for pair in split)):
                return False
            if depth >= max_depth:
                return False
            return (certify(tuple(pair[0] for pair in split), depth+1)
                    and certify(tuple(pair[1] for pair in split), depth+1))

        def point_ok(values):
            bb, h, hb, line_open, zero_open, one_open, wa, one_owner, owner = values[:9]
            if bb <= 0:
                return False
            branch = 0 if h <= 0 else 1 if hb >= 0 else 2
            if (zero_open, one_open, line_open)[branch] >= 0:
                return True
            if h == 0 or hb == 0:
                return False
            ownership = (wa, one_owner, owner)[branch]
            return ownership != 0 and (ownership > 0 or branch == 2 or branch in target_ends)

        return certify(polynomials, 0)

    return (all(direction(first, second, i, ends_b) for i in ends_a)
            and all(direction(second, first, i, ends_a) for i in ends_b))


def admissible_line_projection_path(a0, a1, b0, b1, radius, max_depth):
    """Exclude active positive-measure clamp plateaus along both directions.

    A plateau requires source/target tangent orthogonality and one exact clamp
    level. Endpoint features alone cannot detect it when source ends are open.
    Exact Bernstein ranges exclude these simultaneous zeros or prove opening;
    finite unresolved paths fail, including roots that do not change sign.
    """
    a0,a1,b0,b1=map(_segment,(a0,a1,b0,b1));radius=_radius(radius)
    first=tuple(_affine(x,y) for x,y in zip(a0,a1))
    second=tuple(_affine(x,y) for x,y in zip(b0,b1))
    def direction(source,target):
        u=_vector_sub(source[1],source[0]);v=_vector_sub(target[1],target[0])
        w=_vector_sub(source[0],target[0]);bb=_dot_poly(v,v)
        h=_dot_poly(w,v);hb=_add(h,bb,-1);slope=_dot_poly(u,v)
        differences=tuple(_vector_sub(p,q) for p in source for q in target)
        polynomials=(slope,h,hb,bb,*(lane for vector in differences for lane in vector))
        def excludes_zero(poly):return min(poly)>0 or max(poly)<0
        def certify(values,depth):
            slope,h,hb,bb=values[:4]
            if min(bb)>0:
                if excludes_zero(slope) or (excludes_zero(h) and excludes_zero(hb)):
                    return True
                near=[]
                for axis in range(3):
                    controls=tuple(c for vector in range(4) for c in values[4+3*vector+axis])
                    near.append(Fraction(0) if min(controls)<=0<=max(controls)
                                else min(abs(c) for c in controls))
                if sum(x*x for x in near)>=radius*radius:return True
            if depth>=max_depth:return False
            split=tuple(_split_curve(poly) for poly in values)
            return (certify(tuple(pair[0] for pair in split),depth+1) and
                    certify(tuple(pair[1] for pair in split),depth+1))
        return certify(polynomials,0)
    return direction(first,second) and direction(second,first)
