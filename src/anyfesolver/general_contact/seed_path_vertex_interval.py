"""Outward-rounded vertex-interval whole-path gap certificate (opt-in dev).

A directed-interval sufficient condition for the Q4/S3 seed-path admission
contract, selected only by the explicit ``q4_source_overlap_dual_dev``
development profile. The certified quantity is the actual contract gap

    g(t, u, v) = r * (P . Ns) / |Ns|,
    r = N / (P . T),  N = (s - c0) . T,

with ``s`` the actual bilinear Q4 skin point, ``Ns = dX/du x dX/dv`` the
actual bilinear skin normal, ``P`` the original parent Q4 corner-0 normal,
``T`` the target S3 normal and ``c0`` its corner-0, over the complete
represented pair path ``(t, u, v)`` in [0, 1]^3 including the four
endpoint error boxes. A plane intersection along ``Ns`` is a distinct
quantity for warped geometry and is never equated with this gap.

Every arithmetic operation is an outward-rounded directed float interval
(``math.nextafter``); every enclosure is a whole-domain superset: the
skin point over one box is the per-component hull of its eight exact
trilinear vertex values (corner enclosures blended at each vertex's own
path parameter, retaining the partition-of-unity correlation of the
bilinear weights), and dX/du, dX/dv are the hulls of their four bilinear
vertex values. Target barycentric (foot) enclosures prune only boxes
provably outside the represented overlap; inside and uncertain boxes
must certify the budget. The structure mirrors the independently
reviewed stdlib reference (scripts/general_contact_seed_gap_reference.py)
without sharing its code.

Unresolved charts, exhausted budgets and proven violations all refuse;
this certificate never admits a pair it could not enclose.
"""

from __future__ import annotations

import math

import numpy as np

_ROOT_BOX = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))


def _down(value):
    return math.nextafter(value, -math.inf)


def _up(value):
    return math.nextafter(value, math.inf)


def _add(a, b):
    return (_down(a[0] + b[0]), _up(a[1] + b[1]))


def _sub(a, b):
    return (_down(a[0] - b[1]), _up(a[1] - b[0]))


def _mul(a, b):
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (_down(min(products)), _up(max(products)))


def _divide_positive(a, b):
    """Directed interval division; the caller must guarantee b[0] > 0."""
    if not b[0] > 0.0:
        raise ValueError("directed division needs a strictly positive denominator")
    quotients = (a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1])
    return (_down(min(quotients)), _up(max(quotients)))


def _sqrt_positive(a):
    """Directed interval square root; the caller must guarantee a[0] > 0."""
    if not a[0] > 0.0:
        raise ValueError("directed square root needs a strictly positive interval")
    return (_down(math.sqrt(a[0])), _up(math.sqrt(a[1])))


def _dot(a, b):
    value = _mul((a[0], a[1]), (b[0], b[1]))
    for index in (1, 2):
        value = _add(value, _mul((a[index * 2], a[index * 2 + 1]),
                                 (b[index * 2], b[index * 2 + 1])))
    return value


def _cross(a, b):
    return (_sub(_mul((a[2], a[3]), (b[4], b[5])),
                 _mul((a[4], a[5]), (b[2], b[3]))),
            _sub(_mul((a[4], a[5]), (b[0], b[1])),
                 _mul((a[0], a[1]), (b[2], b[3]))),
            _sub(_mul((a[0], a[1]), (b[4], b[5])),
                 _mul((a[2], a[3]), (b[0], b[1]))))


def _flat(vector):
    """Flat (lo0, hi0, lo1, hi1, lo2, hi2) form of a 3-component interval."""
    return tuple(component for interval in vector for component in interval)


class _PairPath:
    """Validated float corner positions, error boxes and budget."""

    def __init__(self, source_start, source_end, target_start, target_end,
                 endpoint_errors, gap_budget):
        a0, a1, b0, b1 = (np.asarray(x, dtype=float) for x in
                          (source_start, source_end, target_start, target_end))
        if (a0.shape != (4, 3) or a1.shape != (4, 3)
                or b0.shape != (3, 3) or b1.shape != (3, 3)
                or not all(np.all(np.isfinite(x)) for x in (a0, a1, b0, b1))):
            raise ValueError("vertex-interval certificate needs finite Q4/S3 endpoints")
        if not (np.isfinite(gap_budget) and gap_budget >= 0.0):
            raise ValueError("vertex-interval certificate needs a nonnegative budget")
        errors = (0.0, 0.0, 0.0, 0.0) if endpoint_errors is None else endpoint_errors
        if len(errors) != 4:
            raise ValueError("vertex-interval certificate needs four error boxes")
        boxes = []
        for corners, error in zip((a0, a1, b0, b1), errors):
            box = np.broadcast_to(np.asarray(error, dtype=float), corners.shape)
            if not np.all(np.isfinite(box)):
                raise ValueError("vertex-interval certificate needs finite error boxes")
            boxes.append(box.tolist())
        self.a0, self.a1 = a0.tolist(), a1.tolist()
        self.b0, self.b1 = b0.tolist(), b1.tolist()
        self.e0, self.e1, self.e2, self.e3 = boxes
        self.budget = float(gap_budget)


def _corner_at(start, end, error_start, error_end, t):
    """Directed corner-component enclosure at one exact dyadic path value."""
    one_minus_t = 1.0 - t
    center = _add(_mul((one_minus_t, one_minus_t), (start, start)),
                  _mul((t, t), (end, end)))
    error = _add(_mul((one_minus_t, one_minus_t), (error_start, error_start)),
                 _mul((t, t), (error_end, error_end)))
    return (_sub(center, error)[0], _add(center, error)[1])


def _corner_boxes(start, end, error_start, error_end, t0, t1):
    """Directed enclosure of one corner over the whole t interval [t0, t1]."""
    first = _corner_at(start, end, error_start, error_end, t0)
    last = _corner_at(start, end, error_start, error_end, t1)
    return (min(first[0], last[0]), max(first[1], last[1]))


def _vertex_point(path, t0, t1, u0, u1, v0, v1):
    """Per-component hull of the eight trilinear vertex enclosures."""
    low = [None, None, None]
    high = [None, None, None]
    for t in (t0, t1):
        corner = [[_corner_at(path.a0[k][d], path.a1[k][d],
                              path.e0[k][d], path.e1[k][d], t)
                   for d in range(3)] for k in range(4)]
        for u in (u0, u1):
            for v in (v0, v1):
                weights = ((1.0 - u) * (1.0 - v), u * (1.0 - v),
                           u * v, (1.0 - u) * v)
                for d in range(3):
                    acc = _mul((weights[0], weights[0]), corner[0][d])
                    for k in (1, 2, 3):
                        acc = _add(acc, _mul((weights[k], weights[k]),
                                             corner[k][d]))
                    if low[d] is None or acc[0] < low[d]:
                        low[d] = acc[0]
                    if high[d] is None or acc[1] > high[d]:
                        high[d] = acc[1]
    return [(low[d], high[d]) for d in range(3)]


def _vertex_derivative(path, t0, t1, p0, p1, first):
    """Per-component hull of the four bilinear vertex enclosures."""
    edges = ((1, 0), (2, 3)) if first else ((3, 0), (2, 1))
    low = [None, None, None]
    high = [None, None, None]
    for t in (t0, t1):
        corner = [[_corner_at(path.a0[k][d], path.a1[k][d],
                              path.e0[k][d], path.e1[k][d], t)
                   for d in range(3)] for k in range(4)]
        for p in (p0, p1):
            for d in range(3):
                acc = _add(_mul((1.0 - p, 1.0 - p),
                                _sub(corner[edges[0][0]][d],
                                     corner[edges[0][1]][d])),
                           _mul((p, p),
                                _sub(corner[edges[1][0]][d],
                                     corner[edges[1][1]][d])))
                if low[d] is None or acc[0] < low[d]:
                    low[d] = acc[0]
                if high[d] is None or acc[1] > high[d]:
                    high[d] = acc[1]
    return [(low[d], high[d]) for d in range(3)]


def _evaluate_box(path, box):
    """Directed enclosure of the contract gap and foot over one dyadic box."""
    (t0, t1), (u0, u1), (v0, v1) = box
    source = [[_corner_boxes(path.a0[k][d], path.a1[k][d],
                             path.e0[k][d], path.e1[k][d], t0, t1)
               for d in range(3)] for k in range(4)]
    target = [[_corner_boxes(path.b0[k][d], path.b1[k][d],
                             path.e2[k][d], path.e3[k][d], t0, t1)
               for d in range(3)] for k in range(3)]
    point = _vertex_point(path, t0, t1, u0, u1, v0, v1)
    edge1 = [_sub(target[1][d], target[0][d]) for d in range(3)]
    edge2 = [_sub(target[2][d], target[0][d]) for d in range(3)]
    target_normal = _cross(_flat(edge1), _flat(edge2))
    parent = _cross(_flat([_sub(source[1][d], source[0][d]) for d in range(3)]),
                    _flat([_sub(source[3][d], source[0][d]) for d in range(3)]))
    height = _dot(_flat([_sub(point[d], target[0][d]) for d in range(3)]),
                  target_normal)
    projected = _dot(parent, target_normal)
    du = _vertex_derivative(path, t0, t1, v0, v1, True)
    dv = _vertex_derivative(path, t0, t1, u0, u1, False)
    actual = _cross(_flat(du), _flat(dv))
    parent_actual = _dot(parent, actual)
    target_squared = _dot(target_normal, target_normal)
    actual_squared = _dot(actual, actual)
    if not (projected[0] > 0.0 and parent_actual[0] > 0.0
            and actual_squared[0] > 0.0 and target_squared[0] > 0.0):
        return None
    actual_norm = _sqrt_positive(actual_squared)
    radius = _divide_positive(height, projected)
    gap = _divide_positive(_mul(radius, parent_actual), actual_norm)
    foot = [_sub(point[d], _mul(radius, (parent[d * 2], parent[d * 2 + 1])))
            for d in range(3)]
    foot_offset = [_sub(foot[d], target[0][d]) for d in range(3)]
    lambda1 = _divide_positive(
        _dot(_flat([_cross(_flat(foot_offset), _flat(edge2))[d]
                    for d in range(3)]), target_normal), target_squared)
    lambda2 = _divide_positive(
        _dot(_flat([_cross(_flat(edge1), _flat(foot_offset))[d]
                    for d in range(3)]), target_normal), target_squared)
    lambda_sum = _add(lambda1, lambda2)
    if lambda1[0] >= 0.0 and lambda2[0] >= 0.0 and lambda_sum[1] <= 1.0:
        feasibility = "inside"
    elif lambda1[1] < 0.0 or lambda2[1] < 0.0 or lambda_sum[0] > 1.0:
        feasibility = "outside"
    else:
        feasibility = "uncertain"
    return {"gap": gap, "feasibility": feasibility}


def _children(box):
    """The eight dyadic sub-boxes of one box (dyadic midpoints)."""
    (t0, t1), (u0, u1), (v0, v1) = box
    tm = (t0 + t1) / 2
    um = (u0 + u1) / 2
    vm = (v0 + v1) / 2
    return [(tb, ub, vb)
            for tb in ((t0, tm), (tm, t1))
            for ub in ((u0, um), (um, u1))
            for vb in ((v0, vm), (vm, v1))]


def seed_path_vertex_interval_certificate(source_start, source_end,
                                          target_start, target_end, *,
                                          gap_budget, endpoint_errors=None,
                                          max_depth=7, max_boxes=1024,
                                          diagnostics=None):
    """Whole-path vertex-interval sufficient gap certificate for one pair.

    Returns ``(lower, visited, certified)``. ``certified`` is True only when
    every represented-overlap region of the complete ``(t, u, v)`` path was
    enclosed with a resolved chart and a directed gap lower bound at or
    above ``-gap_budget``; ``lower`` is then that valid global lower bound
    of the contract gap over the represented overlap (``+inf`` when the foot
    is provably outside the target everywhere). Any proven violating point,
    unresolved chart or exhausted budget returns ``certified=False`` with
    ``lower=-inf``: the certificate refuses and the caller keeps its
    existing routes. ``diagnostics`` optionally receives the box counts.
    """
    if type(max_depth) is not int or not 0 <= max_depth <= 7:
        raise ValueError("vertex-interval certificate depth must be in [0, 7]")
    if type(max_boxes) is not int or not 1 <= max_boxes <= 32768:
        raise ValueError("vertex-interval certificate needs a box budget in [1, 32768]")
    if diagnostics is not None and type(diagnostics) is not dict:
        raise ValueError("vertex-interval diagnostics need a disposable dict")
    path = _PairPath(source_start, source_end, target_start, target_end,
                     endpoint_errors, gap_budget)
    budget = path.budget
    stack = [(_ROOT_BOX, 0)]
    visited = 0
    safe_lower = None
    safe_boxes = 0
    remaining = 0
    unresolved = 0
    violation = False
    while stack:
        box, depth = stack.pop()
        visited += 1
        exhausted = depth >= max_depth or visited >= max_boxes
        enclosure = _evaluate_box(path, box)
        if enclosure is None:
            if exhausted:
                unresolved += 1
            else:
                stack.extend((child, depth + 1) for child in _children(box))
            continue
        if enclosure["feasibility"] == "outside":
            continue
        if (enclosure["gap"][1] < -budget
                and enclosure["feasibility"] == "inside"):
            violation = True
            break
        if enclosure["gap"][0] >= -budget:
            value = enclosure["gap"][0]
            safe_lower = value if safe_lower is None else min(safe_lower, value)
            safe_boxes += 1
            continue
        if exhausted:
            remaining += 1
        else:
            stack.extend((child, depth + 1) for child in _children(box))
    certified = not (violation or remaining or unresolved)
    lower = (safe_lower if certified and safe_lower is not None
             else float("inf") if certified else -float("inf"))
    if diagnostics is not None:
        diagnostics.update(visited_boxes=visited, safe_boxes=safe_boxes,
                           remaining_boxes=remaining,
                           unresolved_chart_boxes=unresolved,
                           violation_proven=violation, certified=certified,
                           gap_lower_m=None if lower in (float("inf"),
                                                         -float("inf")) else lower)
    return lower, visited, certified
