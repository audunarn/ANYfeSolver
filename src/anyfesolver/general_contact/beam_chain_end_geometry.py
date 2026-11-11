"""Exact nearest-chain endpoint geometry and conservative affine path bounds.

The caller owns material topology, physical curve regularity and reciprocal
cap/cap admission. This module resolves the minimum over target segments, and
certifies strict nearest feature and source ownership branches without sampling
as evidence of whole-path admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .beam_pair_end_geometry import (
    EndpointFeature, _add, _affine, _dot_poly, _radius, _segment,
    _vector_sub, endpoint_feature,
)
from .beam_temporal_bounds import _split_curve
from .capped_cylinder import _product


@dataclass(frozen=True)
class NearestEndpointFeature:
    """Closest target index, feature, material clamp node and distance ties."""

    owner: int
    feature: EndpointFeature
    target_node: int | None
    tied_owners: tuple[int, ...]


def _targets(targets, target_nodes):
    targets = tuple(_segment(target) for target in targets)
    nodes = tuple(tuple(row) for row in target_nodes)
    if not targets or len(targets) != len(nodes):
        raise ValueError('target segments and material-node pairs must match')
    if any(len(row) != 2 or any(not isinstance(n, int) for n in row)
           for row in nodes):
        raise ValueError('each target requires two integer material-node IDs')
    return targets, nodes


def nearest_endpoint_feature(source, targets, target_nodes, endpoint, radius):
    """Resolve the exact minimum, rejecting ambiguous active work maps.

    Strict clamps at the same material node and exact geometric point are one
    feature. Its deterministic owner is the smallest target index. Other exact
    ties, raw projection ties and zero source ownership reject when gap < 0.
    Open features (including gap == 0) may be chosen deterministically.
    """
    source, radius = _segment(source), _radius(radius)
    targets, nodes = _targets(targets, target_nodes)
    features = tuple(endpoint_feature(source, target, endpoint, radius)
                     for target in targets)
    distance = min(feature.squared_distance for feature in features)
    ties = tuple(i for i, feature in enumerate(features)
                 if feature.squared_distance == distance)
    owner = ties[0]
    winner = features[owner]
    node = (None if winner.target_endpoint is None
            else nodes[owner][winner.target_endpoint])
    if winner.gap_sign < 0:
        if any(features[i].projection_tie or features[i].ownership == 0
               for i in ties):
            raise ValueError('active nearest endpoint projection/ownership tie')
        if len(ties) > 1:
            point = (None if winner.target_endpoint is None
                     else targets[owner][winner.target_endpoint])
            if node is None or any(
                features[i].target_endpoint is None
                or nodes[i][features[i].target_endpoint] != node
                or targets[i][features[i].target_endpoint] != point
                for i in ties
            ):
                raise ValueError('active nearest endpoint has distinct feature ties')
    return NearestEndpointFeature(owner, winner, node, ties)


def _polynomials(source, target, endpoint):
    w = _vector_sub(source[endpoint], target[0])
    v = _vector_sub(target[1], target[0])
    outward = _vector_sub(source[endpoint], source[1-endpoint])
    far_w = _vector_sub(w, v)
    bb = _dot_poly(v, v)
    h = _dot_poly(w, v)
    hb = _add(h, bb, -1)
    zero = _dot_poly(w, w)
    one = _dot_poly(far_w, far_w)
    line = _add(_product(zero, bb), _product(h, h), -1)
    wa = _dot_poly(w, outward)
    va = _dot_poly(v, outward)
    interior_owner = _add(_product(wa, bb), _product(h, va), -1)
    return (bb, h, hb, zero, one, line, wa, _add(wa, va, -1),
            interior_owner, w, far_w)


def _branch(values):
    _, h, hb = values[:3]
    if max(h) < 0:
        return 0
    if min(hb) > 0:
        return 1
    if min(h) > 0 and max(hb) < 0:
        return 2
    return None


def _distance_bounds(values):
    """Lower/upper bounds on distance squared; upper needs a strict branch."""
    bb, _, _, zero, one, line = values[:6]
    w, far_w = values[9:11]
    # The coordinate hull contains p(t)-q(t,s) for all t,s in [0,1].
    near = []
    for first, last in zip(w, far_w):
        controls = (*first, *last)
        lo, hi = min(controls), max(controls)
        near.append(Fraction(0) if lo <= 0 <= hi else min(abs(lo), abs(hi)))
    lower = sum((x*x for x in near), Fraction(0))
    branch = _branch(values)
    if min(bb) > 0:
        # Infinite-line distance is a lower bound even on unresolved clamps.
        lower = max(lower, max(Fraction(0), min(line))/max(bb))
        if branch is not None:
            poly = (zero, one, line)[branch]
            if branch == 2:
                lower = max(lower, max(Fraction(0), min(poly))/max(bb))
                upper = max(Fraction(0), max(poly))/min(bb)
            else:
                lower = max(lower, max(Fraction(0), min(poly)))
                upper = max(Fraction(0), max(poly))
            return lower, upper
    return lower, None


def _split_coordinates(segment):
    splits = tuple(tuple(_split_curve(lane) for lane in point)
                   for point in segment)
    return tuple(tuple(tuple(pair[side] for pair in point) for point in splits)
                 for side in (0, 1))


def _point_coordinates(segment, position):
    if position in (0, 1):
        return tuple(tuple(lane[position] for lane in point) for point in segment)
    return tuple(tuple((lane[0]+lane[-1])/2 for lane in point) for point in segment)


def admissible_chain_endpoint_path(
    source0, source1, targets0, targets1, target_nodes, endpoint, radius,
    max_depth, *, allowed_target_nodes=None,
):
    """Certify opening or strict, fixed nearest feature/ownership per interval.

    Original coordinates are rational affine Bernstein polynomials. At each
    half split we split those coordinates and recompute correlated bounds.
    Admission requires either all target distance lower bounds >= radius**2,
    or a strict winner/group whose upper bound is below every other target's
    lower bound. A group contains only strict clamps to the same material node
    with identical coordinate polynomials. Ownership must have one strict sign.

    ``allowed_target_nodes`` restricts active clamps to caller-approved physical
    ends and selected internal junctions; None leaves topology to the caller.
    Samples choose candidates and reject definite failures, never certify an
    interval. Finite unresolved paths, including closed feature switches, fail.
    """
    source0, source1 = _segment(source0), _segment(source1)
    targets0, nodes = _targets(targets0, target_nodes)
    targets1, _ = _targets(targets1, nodes)
    radius = _radius(radius)
    if endpoint not in (0, 1):
        raise ValueError('B2 endpoint index must be 0 or 1')
    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError('max_depth must be a nonnegative integer')
    allowed = None if allowed_target_nodes is None else frozenset(allowed_target_nodes)
    source = tuple(_affine(first, last) for first, last in zip(source0, source1))
    targets = tuple(tuple(_affine(first, last) for first, last in zip(a, b))
                    for a, b in zip(targets0, targets1))
    radius_squared = radius*radius

    def sample(first, others, position):
        try:
            result = nearest_endpoint_feature(
                _point_coordinates(first, position),
                tuple(_point_coordinates(other, position) for other in others),
                nodes, endpoint, radius,
            )
        except ValueError:
            return None
        if (result.feature.gap_sign < 0 and result.target_node is not None
                and allowed is not None and result.target_node not in allowed):
            return None
        return result

    def certify(first, others, depth):
        values = tuple(_polynomials(first, other, endpoint) for other in others)
        # A collapsed endpoint pose is never a valid target segment.
        if any(value[0][0] <= 0 or value[0][-1] <= 0 for value in values):
            return False
        bounds = tuple(_distance_bounds(value) for value in values)
        if all(lower >= radius_squared for lower, _ in bounds):
            return True
        candidate = sample(first, others, Fraction(1, 2))
        if candidate is None or sample(first, others, 0) is None or sample(first, others, 1) is None:
            return False
        group = candidate.tied_owners
        winner = candidate.owner
        branch = _branch(values[winner])
        upper = bounds[winner][1]
        fixed = branch is not None and upper is not None
        if fixed and branch in (0, 1):
            node = nodes[winner][branch]
            fixed = allowed is None or node in allowed
            point = others[winner][branch]
            for i in group:
                other_branch = _branch(values[i])
                fixed = fixed and other_branch in (0, 1)
                if not fixed:
                    break
                fixed = (nodes[i][other_branch] == node
                         and others[i][other_branch] == point)
        elif fixed:
            fixed = len(group) == 1
        if fixed:
            for i in group:
                own = values[i][6+_branch(values[i])]
                if not (min(own) > 0 or max(own) < 0):
                    fixed = False
                    break
        if fixed and all(upper < bounds[i][0] for i in range(len(others)) if i not in group):
            return True
        if depth >= max_depth:
            return False
        split_source = _split_coordinates(first)
        split_targets = tuple(_split_coordinates(other) for other in others)
        return (certify(split_source[0], tuple(pair[0] for pair in split_targets), depth+1)
                and certify(split_source[1], tuple(pair[1] for pair in split_targets), depth+1))

    return certify(source, targets, 0)
