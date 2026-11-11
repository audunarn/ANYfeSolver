"""Whole-face gap enclosure for represented cubic Hermite shell skin paths.

This is a geometry prerequisite, not a projection/pressure topology certificate
or a complete collision-time algorithm. Supplied binary floats are interpreted
as exact rationals; upstream work-map arithmetic is outside this contract.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import comb, gcd, inf, nextafter
from numbers import Integral

import numpy as np

from .plane_newmark_events import _sqrt_interval


@dataclass(frozen=True)
class ShellPairHermiteSweep:
    success: bool
    status: str
    lower_gap: float | None
    # Completed temporal panels: normalized start, end, conservative gap.
    panels: tuple[tuple[float, float, float], ...]
    visited_panels: int
    detail: str = ""


def _coordinates(value, name, count):
    try:
        raw = np.asarray(value)
        # Casting complex or bool data would silently invent coordinates.
        if raw.dtype.kind not in "fiu" or raw.shape != (count, 3):
            raise ValueError
        if any(isinstance(x, (bool, np.bool_, complex, np.complexfloating))
               for x in np.asarray(value, dtype=object).flat):
            raise ValueError
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} requires {count} by 3 finite real coordinates") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} requires {count} by 3 finite real coordinates")
    return tuple(tuple(Fraction(float(x)) for x in row) for row in array)


def _limit(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _product(a, b):
    """Exact Bernstein product: C(n,i) C(m,j) / C(n+m,i+j)."""
    n, m = len(a)-1, len(b)-1
    result = [Fraction(0)]*(n+m+1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] += x*y*Fraction(comb(n, i)*comb(m, j), comb(n+m, i+j))
    return tuple(result)


def _dot(a, b):
    products = tuple(_product(x, y) for x, y in zip(a, b))
    return tuple(sum(values) for values in zip(*products))


def _cross(a, b):
    return tuple(_sub(_product(a[i], b[j]), _product(a[j], b[i]))
                 for i, j in ((1, 2), (2, 0), (0, 1)))


def _normal(nodes, corner):
    n = len(nodes)
    edges = [tuple(_sub(x, y) for x, y in zip(nodes[index], nodes[corner]))
             for index in ((corner+1) % n, (corner-1) % n)]
    return _cross(*edges)


def _nodes(first, last, rate0, rate1, h):
    # Cubic Bernstein controls of the exact endpoint Hermite polynomial.
    return tuple(tuple((x, x+h*v/3, y-h*w/3, y)
                       for x, y, v, w in zip(a, b, c, d))
                 for a, b, c, d in zip(first, last, rate0, rate1))


def _split(coefficients):
    """Exact midpoint de Casteljau subdivision of a Bernstein polynomial.

    Fraction coefficients are aligned at a common denominator so the averaging
    pyramid stays in exact integer arithmetic with power-of-two denominators;
    only the child outputs are canonicalized back to Fractions. Non-Fraction
    coefficients keep the original generic arithmetic, including empty,
    singleton and mixed-type behavior. The endpoint prefetch keeps the
    original unsupported-input refusal before any iteration.
    """
    first, last = coefficients[0], coefficients[-1]
    den = 1
    for c in coefficients:
        if type(c) is not Fraction:
            left, right = [first], [last]
            row = coefficients
            while len(row) > 1:
                row = tuple((x+y)/2 for x, y in zip(row, row[1:]))
                left.append(row[0])
                right.append(row[-1])
            return tuple(left), tuple(reversed(right))
        cd = c.denominator
        if cd != 1:
            den = den // gcd(den, cd) * cd
    row = [c.numerator * (den // c.denominator) for c in coefficients]
    left = [first]
    right = [last]
    d = den
    while len(row) > 1:
        row = [x+y for x, y in zip(row, row[1:])]
        d *= 2
        left.append(Fraction(row[0], d))
        right.append(Fraction(row[-1], d))
    return tuple(left), tuple(reversed(right))


def _lower_float(value):
    """Round downward only when nearest float lies above the exact rational."""
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("bound is not finite") from exc
    if not np.isfinite(result):
        raise ValueError("bound is not finite")
    if Fraction(result) > value:
        result = nextafter(result, -inf)
    if not np.isfinite(result):
        raise ValueError("bound is not finite")
    return result


def shell_pair_hermite_lower(a0, a1, b0, b1, source_v0, source_v1,
                             target_v0, target_v1, dt, *, maximum_panels=128,
                             maximum_depth=12, gap_tolerance_m=None, cancel=None,
                             reference_source=None, reference_target=None,
                             exclude_open_endpoint_approach_tolerance_m=None):
    """Enclose min ((A_i(t)-B_j(t)) dot N_A(t))/|N_A(t)| on 0<=t<=1.

    A is the deforming S3 integration face; B is an S3 or the original Q4.
    Nodal paths use endpoint positions and physical-time velocities, with
    normalized time and positive duration ``dt``. Exact rational Bernstein
    multiplication gives source/target cross normals of degree 6, each gap
    numerator of degree 9, and mutual-normal dots of degree 12. Bernstein
    convex-hull bounds and exact midpoint de Casteljau subdivision cover every
    temporal panel, including extrema between the endpoints.

    The optional open-endpoint approach tolerance watches each original corner
    pair whose two endpoint gaps are certified above that tolerance. Hidden
    negative approach or unresolved bounds cause non-success even when other
    corners are already touching. It is not an exclusion certificate for every
    interpolated material point or weak-pressure mode.

    Every panel must prove positive source/reference-source dot, each original
    target corner/reference-corner dot, and source/target-corner dot. Explicit
    reference skins retain analysis-original polarity across successive steps;
    omission uses this path's starting skins for compatibility. These imply
    nonzero normals, original polarity and consistent source/target orientation.
    The Q4 Jacobian is affine in its two local coordinates, so positive corner
    projections imply positive projection everywhere on the original face.
    S3 linear and Q4 bilinear positions are convex combinations of their
    corners: checking all corner separations encloses every physical point pair.

    Let L be the least degree-9 numerator control and P the least source normal
    projection onto its initial cross R. Cauchy gives |N_A| >= P/|R| > 0.
    Convexity gives |N_A| <= max_k |N_A,k|. A negative L is divided by the
    former lower norm bound; a nonnegative L by the latter upper bound. Exact
    rational square-root intervals and outward float rounding preserve the
    lower enclosure. There is no sampled-path acceptance or tolerance tuning.

    If ``gap_tolerance_m`` is supplied, each panel must additionally establish
    gap >= -tolerance. Otherwise subdivision only resolves geometry orientation;
    a successful negative gap bound is not proof of penetration. Refinement
    always splits the original exact polynomials, never rounded new endpoints.
    Non-success has ``lower_gap=None`` and
    retains completed panels for diagnostics only. ``maximum_panels`` bounds
    visited panels, ``maximum_depth`` bounds bisections, and ``cancel`` is a
    no-argument callback checked before setup and on every visited panel.
    """
    maximum_panels = _limit(maximum_panels, "maximum_panels", 1)
    maximum_depth = _limit(maximum_depth, "maximum_depth", 0)
    if cancel is not None and not callable(cancel):
        raise ValueError("cancel must be callable or None")
    raw_dt = np.asarray(dt)
    if raw_dt.shape != () or raw_dt.dtype.kind not in "fiu":
        raise ValueError("dt must be finite, real and positive")
    try:
        duration = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be finite, real and positive") from exc
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("dt must be finite, real and positive")
    tolerance = None
    if gap_tolerance_m is not None:
        raw_tolerance = np.asarray(gap_tolerance_m)
        if raw_tolerance.shape != () or raw_tolerance.dtype.kind not in "fiu":
            raise ValueError("gap_tolerance_m must be finite, real and nonnegative")
        try:
            tolerance_float = float(gap_tolerance_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("gap_tolerance_m must be finite, real and nonnegative") from exc
        if not np.isfinite(tolerance_float) or tolerance_float < 0:
            raise ValueError("gap_tolerance_m must be finite, real and nonnegative")
        tolerance = Fraction(tolerance_float)
    approach_tolerance = None
    if exclude_open_endpoint_approach_tolerance_m is not None:
        raw = np.asarray(exclude_open_endpoint_approach_tolerance_m)
        if raw.shape != () or raw.dtype.kind not in 'fiu':
            raise ValueError('open-endpoint approach tolerance must be finite, real and nonnegative')
        value = float(raw)
        if not np.isfinite(value) or value < 0.:
            raise ValueError('open-endpoint approach tolerance must be finite, real and nonnegative')
        approach_tolerance = Fraction(value)
    target_shape = np.asarray(b0).shape
    if target_shape not in ((3, 3), (4, 3)):
        raise ValueError("b0 requires original S3 or Q4 corners")
    target_count = target_shape[0]
    supplied = ((a0, "a0", 3), (a1, "a1", 3), (b0, "b0", target_count),
                (b1, "b1", target_count), (source_v0, "source_v0", 3),
                (source_v1, "source_v1", 3), (target_v0, "target_v0", target_count),
                (target_v1, "target_v1", target_count))
    a0, a1, b0, b1, av0, av1, bv0, bv1 = tuple(
        _coordinates(value, name, count) for value, name, count in supplied)
    original_a = a0 if reference_source is None else _coordinates(reference_source, 'reference_source', 3)
    original_b = b0 if reference_target is None else _coordinates(reference_target, 'reference_target', target_count)
    panels = []
    visited = 0

    def result(status, detail=""):
        success = status == "resolved"
        return ShellPairHermiteSweep(success, status,
            min(p[2] for p in panels) if success else None,
            tuple(panels), visited, detail)

    if cancel is not None and cancel():
        return result("cancelled")
    a = _nodes(a0, a1, av0, av1, Fraction(duration))
    b = _nodes(b0, b1, bv0, bv1, Fraction(duration))
    normal = _normal(a, 0)
    reference = _normal(tuple(tuple((value,) for value in point) for point in original_a), 0)
    reference_squared = sum(axis[0]**2 for axis in reference)
    if reference_squared == 0:
        return result("orientation_invalid", "initial source normal is zero")
    reference_upper = _sqrt_interval(reference_squared)[1]
    constraints = [_dot(normal, reference)]
    labels = ["source original polarity"]
    corners = range(target_count) if target_count == 4 else (0,)
    for corner in corners:
        target_normal = _normal(b, corner)
        target_reference = _normal(tuple(tuple((value,) for value in point) for point in original_b), corner)
        constraints.extend((_dot(target_normal, target_reference),
                            _dot(normal, target_normal)))
        labels.extend((f"target corner {corner} original polarity",
                       f"source/target corner {corner} orientation"))
    numerators = tuple(_dot(tuple(_sub(x, y) for x, y in zip(ai, bj)), normal)
                       for ai in a for bj in b)
    watched = ()
    if approach_tolerance is not None:
        squares = tuple(sum(axis[index]**2 for axis in normal) for index in (0,-1))
        if min(squares) > 0:
            upper = tuple(_sqrt_interval(value)[1] for value in squares)
            watched = tuple(i for i,p in enumerate(numerators)
                if p[0] > approach_tolerance*upper[0]
                and p[-1] > approach_tolerance*upper[1])
    # One flat tuple enables exact, consistent subdivision of every polynomial.
    all_polynomials = tuple(constraints)+tuple(normal)+numerators
    pending = [(Fraction(0), Fraction(1), 0, all_polynomials)]
    while pending:
        if cancel is not None and cancel():
            return result("cancelled")
        if visited >= maximum_panels:
            return result("panel_budget_exhausted", "unvisited temporal panels remain")
        start, end, depth, polynomials = pending.pop()
        visited += 1
        checks = polynomials[:len(constraints)]
        for label, coefficients in zip(labels, checks):
            if coefficients[0] <= 0 or coefficients[-1] <= 0:
                return result("orientation_invalid", label)
        unresolved = next((label for label, coefficients in zip(labels, checks)
                           if min(coefficients) <= 0), None)
        if unresolved is not None:
            if depth >= maximum_depth:
                return result("orientation_unresolved", unresolved)
            split = tuple(_split(p) for p in polynomials)
            middle = (start+end)/2
            pending.append((middle, end, depth+1, tuple(p[1] for p in split)))
            pending.append((start, middle, depth+1, tuple(p[0] for p in split)))
            continue
        source_normal = polynomials[len(constraints):len(constraints)+3]
        panel_numerators = polynomials[len(constraints)+3:]
        unresolved_approach = False
        for index in watched:
            p = panel_numerators[index]
            for endpoint in (0,-1):
                norm_upper = _sqrt_interval(sum(axis[endpoint]**2 for axis in source_normal))[1]
                if p[endpoint] < -approach_tolerance*norm_upper:
                    return result('hidden_approach', f'open-endpoint corner pair {index} approaches inside the path')
            if min(p) < 0 and min(p)*reference_upper/min(checks[0]) < -approach_tolerance:
                unresolved_approach = True
        if unresolved_approach:
            if depth >= maximum_depth:
                return result('hidden_approach_unresolved', 'open-endpoint corner path remains unresolved')
            split = tuple(_split(p) for p in polynomials)
            middle = (start+end)/2
            pending.append((middle,end,depth+1,tuple(p[1] for p in split)))
            pending.append((start,middle,depth+1,tuple(p[0] for p in split)))
            continue
        numerator = min(min(p) for p in panel_numerators)
        if numerator < 0:
            lower = numerator*reference_upper/min(checks[0])
        else:
            norm_squared_upper = max(sum(x*x for x in control)
                                     for control in zip(*source_normal))
            lower = numerator/_sqrt_interval(norm_squared_upper)[1]
        try:
            finite_lower = _lower_float(lower)
        except ValueError:
            return result("unrepresentable_bound", "gap lower bound is not finite")
        if tolerance is not None and Fraction(finite_lower) < -tolerance:
            if depth >= maximum_depth:
                return result("gap_unresolved", "panel gap bound does not establish tolerance")
            split = tuple(_split(p) for p in polynomials)
            middle = (start+end)/2
            pending.append((middle, end, depth+1, tuple(p[1] for p in split)))
            pending.append((start, middle, depth+1, tuple(p[0] for p in split)))
            continue
        panels.append((float(start), float(end), finite_lower))
    return result("resolved")
