"""Experimental overlap-local conservative Q4/S3 gap bounds.

Instantaneous and swept bounds use original bilinear Q4 geometry. The swept
form assumes a linear path in physical nodal positions and is selected only
by the explicit development certificate profile.
"""

from __future__ import annotations

import numpy as np

from .q4_source_overlap import (
    projected_q4_source_face,
)


def _outward(value, direction):
    return np.nextafter(value, direction)


def _interval(value):
    value = np.asarray(value, dtype=float)
    return value, value


def _add(a, b):
    return (_outward(a[0]+b[0], -np.inf),
            _outward(a[1]+b[1], np.inf))


def _neg(a):
    return -a[1], -a[0]


def _sub(a, b):
    return _add(a, _neg(b))


def _mul(a, b):
    products = np.stack((a[0]*b[0], a[0]*b[1],
                         a[1]*b[0], a[1]*b[1]))
    return (_outward(np.min(products, axis=0), -np.inf),
            _outward(np.max(products, axis=0), np.inf))


def _dot(a, b):
    products = _mul(a, b)
    value = (products[0][..., 0], products[1][..., 0])
    for index in (1, 2):
        value = _add(value, (products[0][..., index],
                             products[1][..., index]))
    return value


def _cross(a, b):
    values = []
    for j, k in ((1, 2), (2, 0), (0, 1)):
        values.append(_sub(
            _mul((a[0][..., j], a[1][..., j]),
                 (b[0][..., k], b[1][..., k])),
            _mul((a[0][..., k], a[1][..., k]),
                 (b[0][..., j], b[1][..., j]))))
    return tuple(np.stack([value[side] for value in values], axis=-1)
                 for side in (0, 1))


def _normal_controls_interval(start, end, corner=0,
                              start_error=0.0, end_error=0.0):
    # For affine edges e(t), f(t), the middle quadratic Bernstein
    # coefficient of e x f is (e0 x f1 + e1 x f0)/2.
    next_corner, previous_corner = ((corner+1) % len(start),
                                    (corner-1) % len(start))
    first_positions = _enclose_positions(start, start_error)
    last_positions = _enclose_positions(end, end_error)
    def at(positions, index):
        return positions[0][index], positions[1][index]
    e0 = _sub(at(first_positions, next_corner), at(first_positions, corner))
    f0 = _sub(at(first_positions, previous_corner), at(first_positions, corner))
    e1 = _sub(at(last_positions, next_corner), at(last_positions, corner))
    f1 = _sub(at(last_positions, previous_corner), at(last_positions, corner))
    first, last = _cross(e0, f0), _cross(e1, f1)
    middle = _mul(_add(_cross(e0, f1), _cross(e1, f0)), _interval(0.5))
    return tuple(np.stack((first[side], middle[side], last[side]))
                 for side in (0, 1))


def _norm_upper(a):
    magnitude = np.maximum(np.abs(a[0]), np.abs(a[1]))
    squared = _mul(_interval(magnitude), _interval(magnitude))
    total = _add(_add((squared[0][..., 0], squared[1][..., 0]),
                      (squared[0][..., 1], squared[1][..., 1])),
                 (squared[0][..., 2], squared[1][..., 2]))
    return _outward(np.sqrt(total[1]), np.inf)


def _positive_divide_upper(numerator, denominator):
    return _outward(numerator/denominator, np.inf)


def _enclose_positions(positions, error):
    error = np.broadcast_to(np.asarray(error, dtype=float), positions.shape)
    if not np.all(np.isfinite(error)) or np.any(error < 0.0):
        raise ValueError("swept endpoint enclosure must be finite and nonnegative")
    if not np.any(error):
        return _interval(positions)
    return (_outward(positions-error, -np.inf),
            _outward(positions+error, np.inf))


def _quadratic_dot_controls(first,second):
    """Directed quartic Bernstein dot controls, preserving joint time."""
    def dot(i,j):return _dot(tuple(x[i] for x in first),tuple(x[j] for x in second))
    middle=_add(_add(dot(0,2),_mul(dot(1,1),_interval(4.))),dot(2,0))
    controls=(dot(0,0),_mul(_add(dot(0,1),dot(1,0)),_interval(.5)),
        (_outward(middle[0]/6.,-np.inf),_outward(middle[1]/6.,np.inf)),
        _mul(_add(dot(1,2),dot(2,1)),_interval(.5)),dot(2,2))
    return controls


def _quadratic_dot_positive(first,second):
    """Strict directed quartic Bernstein positivity, including endpoints."""
    return all(np.all(np.isfinite(x[0])) and np.min(x[0])>0.
               for x in _quadratic_dot_controls(first,second))


def q4_s3_swept_chart_admissible(source_start,source_end,target_start,target_end,*,endpoint_errors=None):
    """Certify every moving Q1 corner/target normal against corner0.

    This is a prerequisite for both gap sufficient conditions: the actual
    projection law requires these strict mutual normal chart predicates.
    Own-reference polarity by itself does not imply them.
    """
    a0,a1,b0,b1=(np.asarray(x,dtype=float) for x in (source_start,source_end,target_start,target_end))
    if (a0.shape!=(4,3) or a1.shape!=(4,3) or b0.shape!=(3,3) or b1.shape!=(3,3)
            or not all(np.all(np.isfinite(x)) for x in (a0,a1,b0,b1))):
        raise ValueError('swept chart needs finite Q4/S3 endpoints')
    errors=(0.,)*4 if endpoint_errors is None else endpoint_errors
    if len(errors)!=4:raise ValueError('swept gap needs four endpoint enclosures')
    normals=[_normal_controls_interval(a0,a1,k,errors[0],errors[1]) for k in range(4)]
    target=_normal_controls_interval(b0,b1,0,errors[2],errors[3])
    return all(_quadratic_dot_positive(normals[0],other) for other in (*normals,target))


def q4_s3_whole_pair_swept_lower(source_start, source_end,
                                 target_start, target_end, *, endpoint_errors=None):
    """Enclose physical Q1/P1 signed gaps on an affine physical nodal path.

    Source/target positions are convex combinations of their nodal positions;
    the original bilinear source's covariant normal is a convex combination
    of its four corner normals. Each nodal difference is affine in time and
    each corner normal is quadratic. Their dot product has cubic Bernstein
    controls. Keeping their time coupling avoids independent moving-plane
    extrema while conservatively retaining every source/target/corner triple.

    A positive fixed-axis projection of every normal control bounds its norm
    away from zero. For negative numerators this norm lower bound gives a
    signed gap lower bound. No overlap clipping or target-plane approximation
    is used. Endpoint enclosures propagate through all directed arithmetic;
    unresolved source/target polarity returns -inf for the existing fallback.
    """
    return _q4_s3_swept_chart_and_lower(
        source_start, source_end, target_start, target_end,
        endpoint_errors=endpoint_errors)[1]


def _q4_s3_swept_chart_and_lower(source_start, source_end,
                               target_start, target_end, *, endpoint_errors=None,
                               source_controls=None, target_controls=None):
    """Internal joint check with interval-local normal controls.

    The manager may supply controls computed by ``_normal_controls_interval``
    for these exact endpoints/errors within its current certify invocation.
    No controls survive a subdivision or a separate candidate evaluation.
    Endpoint admission and all directed bound operations still run here.
    """
    a0,a1,b0,b1=(np.asarray(item,dtype=float) for item in (
        source_start,source_end,target_start,target_end))
    if (a0.shape!=(4,3) or a1.shape!=(4,3) or b0.shape!=(3,3) or b1.shape!=(3,3)
            or not all(np.all(np.isfinite(item)) for item in (a0,a1,b0,b1))):
        raise ValueError('whole-pair sweep needs finite Q4/S3 endpoints')
    errors=(0.,)*4 if endpoint_errors is None else endpoint_errors
    if len(errors)!=4:raise ValueError('swept gap needs four endpoint enclosures')
    bounds=tuple(_enclose_positions(item,error) for item,error in zip((a0,a1,b0,b1),errors))
    if source_controls is None:
        source_controls=tuple(_normal_controls_interval(a0,a1,k,errors[0],errors[1])
                              for k in range(4))
    if target_controls is None:
        target_controls=_normal_controls_interval(b0,b1,0,errors[2],errors[3])
    chart_ok=all(_quadratic_dot_positive(source_controls[0],other)
                 for other in (*source_controls,target_controls))
    if not chart_ok:
        return False,-float('inf')
    try:
        _,reference=projected_q4_source_face(a0)
        projected_q4_source_face(a1)
    except ValueError:
        return True,-float('inf')
    target_axis=np.cross(b0[1]-b0[0],b0[2]-b0[0])
    target_length=float(np.linalg.norm(target_axis))
    if not np.isfinite(target_length) or target_length<=0.:return True,-float('inf')
    target_reference=target_axis/target_length
    target_polarity=_dot(target_controls,_interval(target_reference))[0]
    if not np.all(np.isfinite(target_polarity)) or np.min(target_polarity)<=0.:
        return True,-float('inf')
    normals=tuple(np.stack([source_controls[k][side] for k in range(4)]) for side in (0,1))
    heights=_dot(normals,_interval(reference))[0]
    denominator=float(np.min(heights))
    reference_norm=float(_norm_upper(_interval(reference)))
    if (not all(np.all(np.isfinite(value)) for value in normals)
            or not np.isfinite(denominator) or denominator<=0.
            or not np.isfinite(reference_norm)):
        return True,-float('inf')
    d0=_sub(tuple(value[:,None,:] for value in bounds[0]),
            tuple(value[None,:,:] for value in bounds[2]))
    d1=_sub(tuple(value[:,None,:] for value in bounds[1]),
            tuple(value[None,:,:] for value in bounds[3]))
    def dot(d,k):
        return _dot(tuple(value[:,:,None,:] for value in d),
                    tuple(value[None,None,:,k,:] for value in normals))
    def third(value):
        # Enclose the exact rational weight 1/3, rather than treating its
        # nearest floating representation as an exact Bernstein coefficient.
        return (_outward(value[0]/3.,-np.inf),_outward(value[1]/3.,np.inf))
    controls=(dot(d0,0),
        third(_add(_mul(dot(d0,1),_interval(2.)),dot(d1,0))),
        third(_add(dot(d0,2),_mul(dot(d1,1),_interval(2.)))),dot(d1,2))
    numerator=min(float(np.min(value[0])) for value in controls)
    if not np.isfinite(numerator):return True,-float('inf')
    scaled=_mul(_interval(min(0.,numerator)),_interval(reference_norm))[0]
    return True,float(_outward(scaled/denominator,-np.inf))

def _point(corners: np.ndarray, u: float, v: float) -> np.ndarray:
    return ((1-u)*(1-v)*corners[0] + u*(1-v)*corners[1]
            + u*v*corners[2] + (1-u)*v*corners[3])


def _interval_box_corners(corners,params):
    # Dyadic shape weights are exact; all point arithmetic remains enclosed.
    u,v=np.asarray(params,dtype=float).T
    weights=((1-u)*(1-v),u*(1-v),u*v,(1-u)*v)
    result=_mul(tuple(x[0] for x in corners),_interval(weights[0][:,None]))
    for index in (1,2,3):
        result=_add(result,_mul(tuple(x[index] for x in corners),
                                _interval(weights[index][:,None])))
    return result


def _linear_cross_controls(a0,a1,b0,b1):
    first,last=_cross(a0,b0),_cross(a1,b1)
    middle=_mul(_add(_cross(a0,b1),_cross(a1,b0)),_interval(.5))
    return tuple(np.stack((first[side],middle[side],last[side]))
                 for side in (0,1))


def _linear_quadratic_dot_controls(a0,a1,b):
    def dot(a,k):return _dot(a,tuple(x[k] for x in b))
    def third(v):return (_outward(v[0]/3.,-np.inf),
                         _outward(v[1]/3.,np.inf))
    return (dot(a0,0),third(_add(_mul(dot(a0,1),_interval(2.)),dot(a1,0))),
            third(_add(dot(a0,2),_mul(dot(a1,1),_interval(2.)))),dot(a1,2))


def _elevate_cubic_time_interval(controls):
    """Align cubic time controls with quartic barycentric controls."""
    if len(controls) != 4:
        raise ValueError('overlap dual needs four cubic time controls')
    return (controls[0],
            _add(_mul(controls[0], _interval(.25)),
                 _mul(controls[1], _interval(.75))),
            _add(_mul(controls[1], _interval(.5)),
                 _mul(controls[2], _interval(.5))),
            _add(_mul(controls[2], _interval(.75)),
                 _mul(controls[3], _interval(.25))),
            controls[3])


def _overlap_dual_coefficient_lower(numerator, barycentric, multipliers):
    """Validate one constant nonnegative dual vector on all 5 x 4 controls.

    Numerator units are m**3 and raw barycentric numerator units m**4;
    multipliers have units 1/m. They are certificate coefficients only.
    Invalid/overflowing candidates provide no improvement, never admission.
    """
    weights = np.asarray(multipliers, dtype=float)
    if (weights.shape != (3,) or not np.all(np.isfinite(weights))
            or np.any(weights < 0.) or len(numerator) != 5
            or len(barycentric) != 3 or any(len(c) != 5 for c in barycentric)):
        return -float('inf')
    values = []
    for k in range(5):
        value = numerator[k]
        if (any(np.shape(x) != (4,) for x in value)
                or not all(np.all(np.isfinite(x)) for x in value)
                or np.any(value[0] > value[1])):
            return -float('inf')
        for weight, controls in zip(weights, barycentric):
            coefficient = controls[k]
            if (any(np.shape(x) != (4,) for x in coefficient)
                    or not all(np.all(np.isfinite(x)) for x in coefficient)
                    or np.any(coefficient[0] > coefficient[1])):
                return -float('inf')
            # One weight is fixed across EVERY aligned time/spatial control.
            # Interval subtraction consumes the upper barycentric endpoint.
            if weight != 0.:
                value = _sub(value, _mul(_interval(weight), coefficient))
        if not all(np.all(np.isfinite(x)) for x in value):
            return -float('inf')
        values.append(float(np.min(value[0])))
    return min(values)


def _overlap_dual_axis_lower(n, c, candidates):
    """Validate fixed axis weights in rows, never choose per coefficient."""
    n, c, weights = (np.asarray(x, dtype=float) for x in (n, c, candidates))
    if (n.shape != (20,) or c.shape != (20,) or weights.ndim != 1
            or not np.all(np.isfinite(n)) or not np.all(np.isfinite(c))):
        return -float('inf')
    weights = weights[np.isfinite(weights) & (weights >= 0.)][:, None]
    if not len(weights):
        return -float('inf')
    with np.errstate(over='ignore', invalid='ignore'):
        transformed = _sub(_interval(n[None, :]),
                           _mul(_interval(weights), _interval(c[None, :])))
    valid = np.all(np.isfinite(transformed[0]) & np.isfinite(transformed[1]), axis=1)
    return (float(np.max(np.min(transformed[0][valid], axis=1)))
            if np.any(valid) else -float('inf'))


def _overlap_dual_numerator_lower(numerator, barycentric):
    """Bounded axis-candidate search followed by outward proof validation.

    At most 3 * choose(20, 2) candidates. Rounded line intersections merely
    select exact binary64 constants; no optimizer/optimality is trusted.
    Every candidate is independently checked over all 20 joint controls.
    """
    aligned = _elevate_cubic_time_interval(numerator)
    lower = _overlap_dual_coefficient_lower(aligned, barycentric, (0., 0., 0.))
    if not np.isfinite(lower):
        return lower, 0
    n = np.concatenate([value[0] for value in aligned])
    checked = 0
    for axis, controls in enumerate(barycentric):
        c = np.concatenate([value[1] for value in controls])
        candidates = set()
        for j in range(20):
            for k in range(j):
                denominator = float(c[j]) - float(c[k])
                if denominator == 0.:
                    continue
                weight = (float(n[j]) - float(n[k])) / denominator
                if np.isfinite(weight) and weight > 0.:
                    candidates.add(weight)
        if candidates:
            # Each row has ONE fixed axis multiplier on all 20 controls.
            # Batch outward validation without a coefficient-wise choice.
            candidate = _overlap_dual_axis_lower(n, c, sorted(candidates))
            checked += len(candidates)
            if np.isfinite(candidate):
                lower = max(lower, candidate)
    return lower, checked


def _overlap_dual_vertex_active_sets():
    """Fixed 4-subsets of the 23 certificate-LP constraint rows."""
    sets = []
    for first in range(23):
        for second in range(first+1, 23):
            for third in range(second+1, 23):
                for fourth in range(third+1, 23):
                    sets.append((first, second, third, fourth))
    return np.asarray(sets, dtype=int)


_OVERLAP_DUAL_VERTEX_ACTIVE_SETS = _overlap_dual_vertex_active_sets()


def _overlap_dual_joint_candidates(n, c):
    """Candidate vertex weights of the constant three-component dual LP.

    Variables are (t, lU, lV, lW); the rows t + lambda . c_j <= n_j over
    the 20 joint controls together with lambda >= 0 are searched as a
    bounded candidate enumeration: every 4-subset of the 23 rows is
    treated as a candidate active set, grouped by its nonnegativity
    rows; those weight components are pinned to exact zero and only the
    remaining square system is solved, so a zero component is never
    lost to solve noise. Finite-precision determinant filtering and
    solving do not certify that all vertices or the exact optimal LP
    value were recovered, and unbounded or degenerate LPs are not
    specially handled. Solutions only propose exact binary64 constants;
    each proposal is validated independently with directed arithmetic
    afterwards, so any adopted bound is sound regardless of enumeration
    completeness.
    """
    n = np.asarray(n, dtype=float)
    c = np.asarray(c, dtype=float)
    if (n.shape != (20,) or c.shape != (3, 20)
            or not np.all(np.isfinite(n)) or not np.all(np.isfinite(c))):
        return []
    sets = _OVERLAP_DUAL_VERTEX_ACTIVE_SETS
    gap_rows = sets < 20
    zero_mask = np.zeros((len(sets), 3), dtype=bool)
    for axis in range(3):
        zero_mask[:, axis] = np.any(sets == 20+axis, axis=1)
    unique = set()
    for zeroed in range(8):
        pattern = np.array([(zeroed >> axis) & 1 for axis in range(3)],
                           dtype=bool)
        selected = np.flatnonzero(np.all(zero_mask == pattern, axis=1))
        free = [axis for axis in range(3) if not pattern[axis]]
        size = 1+len(free)
        if not len(selected):
            continue
        gap = sets[selected][gap_rows[selected]].reshape(len(selected), size)
        matrices = np.ones((len(selected), size, size))
        for position, axis in enumerate(free):
            matrices[:, :, 1+position] = c[axis][gap]
        # Explicit trailing singleton RHS dimension: NumPy 2 treats any
        # non-1D RHS as a matrix stack, so a (batch, size) RHS against
        # stacked (batch, size, size) matrices is ambiguous.
        targets = n[gap][:, :, None]
        with np.errstate(over='ignore', invalid='ignore'):
            determinants = np.linalg.det(matrices)
            usable = np.isfinite(determinants) & (determinants != 0.)
            solutions = np.full((len(selected), size), np.nan)
            if np.any(usable):
                chosen = np.flatnonzero(usable)
                try:
                    solutions[chosen] = np.linalg.solve(matrices[chosen],
                                                        targets[chosen])[..., 0]
                except np.linalg.LinAlgError:
                    for index in chosen:
                        try:
                            solutions[index] = np.linalg.solve(
                                matrices[index],
                                targets[index])[:, 0]
                        except np.linalg.LinAlgError:
                            continue
        weights = np.zeros((len(selected), 3))
        for position, axis in enumerate(free):
            weights[:, axis] = solutions[:, 1+position]
        keep = (np.all(np.isfinite(solutions), axis=1)
                & np.all(weights >= 0., axis=1))
        for row in weights[keep]:
            unique.add((float(row[0]), float(row[1]), float(row[2])))
    return sorted(unique)


def _overlap_dual_joint_weight_lower(n, c, weights):
    """Directed batched validation of constant three-component weights.

    Mirrors ``_overlap_dual_coefficient_lower``'s subtraction order on
    every nonzero weight: one weight triple on all 20 joint controls,
    upper barycentric endpoints consumed, outward rounding after every
    operation. Exactly-zero weights are skipped on that scalar gold
    path but subtracted here as enclosing zero intervals, so batched
    zero-weight rows are slightly more conservative, never less.
    Overflowing or otherwise nonfinite rows validate to
    -inf, never admission.
    """
    n = np.asarray(n, dtype=float)
    c = np.asarray(c, dtype=float)
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    if (n.shape != (20,) or c.shape != (3, 20) or weights.shape[1] != 3
            or not np.all(np.isfinite(n)) or not np.all(np.isfinite(c))
            or not np.all(np.isfinite(weights)) or np.any(weights < 0.)):
        return np.full(len(weights), -np.inf)
    with np.errstate(over='ignore', invalid='ignore'):
        lower = np.broadcast_to(n, (len(weights), 20)).copy()
        for axis in range(3):
            product = _outward(weights[:, axis, None]*c[axis][None, :],
                               np.inf)
            lower = _outward(lower-product, -np.inf)
        finite = np.all(np.isfinite(lower), axis=1)
    return np.where(finite, np.min(lower, axis=1), -np.inf)


def _overlap_dual_joint_lower(numerator, barycentric):
    """Bounded candidate search for a constant three-component dual.

    Candidate weight triples come only from the bounded vertex
    enumeration, which does not certify LP optimality or vertex
    completeness; the batched winner is revalidated through the
    gold-path ``_overlap_dual_coefficient_lower`` and the smaller value
    adopted, so no proposal is trusted twice. The directed validation
    of the chosen nonnegative vector is a sound lower bound regardless
    of enumeration completeness; invalid or overflowing candidates
    never improve the returned bound.
    """
    if (len(numerator) != 4 or len(barycentric) != 3
            or any(len(controls) != 5 for controls in barycentric)):
        return -float('inf'), 0
    aligned = _elevate_cubic_time_interval(numerator)
    for value in (*aligned, *(x for controls in barycentric
                              for x in controls)):
        if (any(np.shape(x) != (4,) for x in value)
                or not all(np.all(np.isfinite(x)) for x in value)
                or np.any(value[0] > value[1])):
            return -float('inf'), 0
    n = np.concatenate([value[0] for value in aligned])
    c = np.stack([np.concatenate([value[1] for value in controls])
                  for controls in barycentric])
    candidates = _overlap_dual_joint_candidates(n, c)
    if not candidates:
        return -float('inf'), 0
    weights = np.asarray(candidates, dtype=float)
    values = _overlap_dual_joint_weight_lower(n, c, weights)
    best = int(np.argmax(values))
    if not np.isfinite(values[best]):
        return -float('inf'), len(candidates)
    winner = tuple(float(x) for x in weights[best])
    gold = _overlap_dual_coefficient_lower(aligned, barycentric, winner)
    value = min(float(values[best]), gold)
    return (float(value) if np.isfinite(value) else -float('inf')), len(candidates)


def matching_projection_swept_lower(source_start,source_end,target_start,
                                    target_end,depth=3,*,endpoint_errors=None,
                                    gap_budget=None,adaptive_depth=None,
                                    node_budget=None,diagnostics=None,
                                    refinement_gap_budget=None,
                                    overlap_dual=False):
    """Conservative actual Q1-normal gap over P1 matching projection.

    ``P`` is the original parent Q4 corner0 normal throughout spatial
    subdivision. For represented overlap, s-y=r P and
    g=r |P| (n.P/|P|), with 0<n.P/|P|<=1. Bound the negative projection
    distance using cubic target-plane numerator / positive quartic T.P.
    Directed target-barycentric quartics prune only proven disjoint boxes.
    No sampled shape, first-order triangle or target-normal gap is used.

    All endpoint errors and dyadic point arithmetic are enclosed. Existing
    manager time subdivision supplies shorter affine intervals. ``gap_budget``
    permits early termination only after this complete box is certified.
    Returns (lower, visited); unresolved charts return -inf, not acceptance.
    Optional adaptive refinement retains a complete parent bound whenever
    its finite node budget cannot cover all children. Only uncertified
    potentially overlapping boxes are refined; no samples enter admission.
    ``overlap_dual`` is an explicit development-only opt-in. At terminal
    boxes, one fixed nonnegative vector can tighten the numerator using raw
    barycentric quartics; boxes still refusing the budget after that
    single-axis search additionally try one constant three-component
    vertex-enumerated weight vector. Default arithmetic and subdivision
    are unchanged.
    """
    if type(overlap_dual) is not bool:
        raise ValueError('matching-projection overlap dual needs a bool')
    if type(depth) is not int or not 0<=depth<=7:
        raise ValueError('matching-projection depth must be in [0, 7]')
    if gap_budget is not None and (not np.isfinite(gap_budget) or gap_budget<0.):
        raise ValueError('matching-projection gap budget must be nonnegative')
    if refinement_gap_budget is not None and (not np.isfinite(refinement_gap_budget) or refinement_gap_budget<0.):
        raise ValueError('matching-projection refinement budget must be nonnegative')
    threshold = gap_budget if refinement_gap_budget is None else refinement_gap_budget
    adaptive = adaptive_depth is not None
    if adaptive:
        if (type(adaptive_depth) is not int or not depth<=adaptive_depth<=7
                or type(node_budget) is not int or not 1<=node_budget<=32768
                or threshold is None
                or (gap_budget is not None and gap_budget!=threshold)):
            raise ValueError('adaptive matching projection needs depth<=7, finite node budget and gap budget')
        if gap_budget is None and node_budget<(4**(depth+1)-1)//3:
            raise ValueError('full diagnostic refinement must cover the initial complete partition')
    elif node_budget is not None or refinement_gap_budget is not None:
        raise ValueError('matching-projection node budget needs adaptive depth')
    if diagnostics is not None:
        if type(diagnostics) is not dict:
            raise ValueError('matching-projection diagnostics need a disposable dict')
        diagnostics.update(initial_depth=depth,maximum_depth=adaptive_depth,
                           node_budget=node_budget,visited_nodes=0,
                           node_budget_exhausted=False,depth_exhausted=False,
                           unresolved_chart=False)
        if overlap_dual:
            diagnostics.update(dual_candidates_checked=0,dual_improved_boxes=0,
                               dual_joint_candidates_checked=0,dual_joint_improved_boxes=0)
    a0,a1,b0,b1=tuple(np.asarray(x,dtype=float) for x in
                     (source_start,source_end,target_start,target_end))
    if (a0.shape!=(4,3) or a1.shape!=(4,3) or b0.shape!=(3,3) or b1.shape!=(3,3)
            or not all(np.all(np.isfinite(x)) for x in (a0,a1,b0,b1))):
        raise ValueError('matching projection needs finite Q4/S3 endpoints')
    errors=(0.,)*4 if endpoint_errors is None else endpoint_errors
    if len(errors)!=4:raise ValueError('matching projection needs four error boxes')
    bounds=tuple(_enclose_positions(x,e) for x,e in zip((a0,a1,b0,b1),errors))
    # Preserve original endpoint chart refusal; interval positivity below is
    # additionally required, so an uncertain input cannot gain admission.
    projected_q4_source_face(a0);projected_q4_source_face(a1)
    p=_normal_controls_interval(a0,a1,0,errors[0],errors[1])
    target=_normal_controls_interval(b0,b1,0,errors[2],errors[3])
    normals=tuple(_normal_controls_interval(a0,a1,k,errors[0],errors[1])
                  for k in range(4))
    if not all(_quadratic_dot_positive(p,x) for x in (*normals,target)):
        if diagnostics is not None:diagnostics.update(unresolved_chart=True,visited_nodes=1)
        return -float('inf'),1
    denominator=_quadratic_dot_controls(target,p)
    if not all(np.all(np.isfinite(x)) for value in (*denominator,p)
               for x in value):
        if diagnostics is not None:diagnostics.update(unresolved_chart=True,visited_nodes=1)
        return -float('inf'),1
    d_lower=min(float(np.min(x[0])) for x in denominator)
    p_upper=float(np.max(_norm_upper(p)))
    if not np.isfinite(d_lower) or d_lower<=0. or not np.isfinite(p_upper):
        if diagnostics is not None:diagnostics.update(unresolved_chart=True,visited_nodes=1)
        return -float('inf'),1
    def vertex(side,index):return tuple(x[index] for x in bounds[side])
    e0,e1=_sub(vertex(2,1),vertex(2,0)),_sub(vertex(3,1),vertex(3,0))
    f0,f1=_sub(vertex(2,2),vertex(2,0)),_sub(vertex(3,2),vertex(3,0))
    visited=0
    initial_leaves=[]
    collecting_initial=adaptive
    budget_exhausted=False
    depth_exhausted=False
    def recurse(u0,u1,v0,v1,remaining):
        nonlocal visited,budget_exhausted,depth_exhausted
        visited+=1
        def terminal(value,eligible=False):
            if collecting_initial:
                initial_leaves.append(dict(box=(u0,u1,v0,v1),lower=value,eligible=eligible))
            return value
        params=((u0,v0),(u1,v0),(u1,v1),(u0,v1))
        x0=_sub(_interval_box_corners(bounds[0],params),vertex(2,0))
        x1=_sub(_interval_box_corners(bounds[1],params),vertex(3,0))
        numerator=_linear_quadratic_dot_controls(x0,x1,target)
        if not all(np.all(np.isfinite(x)) for value in numerator for x in value):
            return terminal(-float('inf'))
        n_lower=min(float(np.min(x[0])) for x in numerator)
        if not np.isfinite(n_lower):return terminal(-float('inf'))
        scaled=_mul(_interval(min(0.,n_lower)),_interval(p_upper))[0]
        lower=float(_outward(scaled/d_lower,-np.inf))
        # Certifying the entire box also certifies any overlap subset. Avoid
        # constructing overlap coefficients when no localization is needed.
        if (n_lower>=0. or (gap_budget is not None and lower>=-gap_budget)
                or (adaptive and not collecting_initial and lower>=-threshold)):
            return terminal(lower)
        v_controls=_quadratic_dot_controls(_linear_cross_controls(x0,x1,f0,f1),p)
        w_controls=_quadratic_dot_controls(_linear_cross_controls(e0,e1,x0,x1),p)
        if not all(np.all(np.isfinite(x)) for value in (*v_controls,*w_controls)
                   for x in value):
            return terminal(-float('inf'))
        # Each time control is bilinear on the parameter rectangle. Its
        # four corner extrema enclose all spatial points, including rims.
        bary_upper=[max(float(np.max(x[1])) for x in controls)
                    for controls in (v_controls,w_controls)]
        u_controls=tuple(_sub(_sub(denominator[k],v_controls[k]),w_controls[k]) for k in range(5))
        if not all(np.all(np.isfinite(x)) for value in u_controls for x in value):
            return terminal(-float('inf'))
        bary_upper.append(max(float(np.max(x[1])) for x in u_controls))
        if not all(np.isfinite(x) for x in bary_upper):return terminal(-float('inf'))
        if min(bary_upper)<0.:return terminal(float('inf'))
        if remaining==0:
            if overlap_dual:
                dual,checked=_overlap_dual_numerator_lower(
                    numerator,(u_controls,v_controls,w_controls))
                if diagnostics is not None:
                    diagnostics['dual_candidates_checked']=diagnostics.get('dual_candidates_checked',0)+checked
                if np.isfinite(dual) and dual>n_lower:
                    scaled=_mul(_interval(min(0.,dual)),_interval(p_upper))[0]
                    improved=float(_outward(scaled/d_lower,-np.inf))
                    if np.isfinite(improved) and improved>lower:
                        lower=improved
                        if diagnostics is not None:
                            diagnostics['dual_improved_boxes']=diagnostics.get('dual_improved_boxes',0)+1
                if threshold is not None and lower>=-threshold:
                    return terminal(lower)
                if threshold is not None and lower<-threshold:
                    # The axis dual left this terminal box refusing. One
                    # constant three-component weight vector, proposed by
                    # the bounded vertex enumeration and revalidated with
                    # the same directed arithmetic, can only tighten this
                    # certificate. Boxes meeting the budget never pay for
                    # it, and budgetless development calls are unchanged.
                    joint,joint_checked=_overlap_dual_joint_lower(
                        numerator,(u_controls,v_controls,w_controls))
                    if diagnostics is not None:
                        diagnostics['dual_joint_candidates_checked']=diagnostics.get('dual_joint_candidates_checked',0)+joint_checked
                    if np.isfinite(joint) and joint>n_lower:
                        scaled=_mul(_interval(min(0.,joint)),_interval(p_upper))[0]
                        improved=float(_outward(scaled/d_lower,-np.inf))
                        if np.isfinite(improved) and improved>lower:
                            lower=improved
                            if diagnostics is not None:
                                diagnostics['dual_joint_improved_boxes']=diagnostics.get('dual_joint_improved_boxes',0)+1
                    if lower>=-threshold:
                        return terminal(lower)
            if adaptive and not collecting_initial:depth_exhausted=True
            return terminal(lower,eligible=True)
        um,vm=.5*(u0+u1),.5*(v0+v1)
        boxes=((u0,um,v0,vm),(um,u1,v0,vm),(um,u1,vm,v1),(u0,um,vm,v1))
        if not adaptive:
            return min(recurse(*box,remaining-1) for box in boxes)
        children=[]
        for box in boxes:
            if visited>=node_budget:
                # The entire parent, including every unvisited child,
                # remains covered. Partial children cannot replace it.
                budget_exhausted=True
                return terminal(lower)
            children.append(recurse(*box,remaining-1))
        return max(lower,min(children))
    lower=recurse(0.,1.,0.,1.,depth)
    if adaptive:
        collecting_initial=False
        # Complete the initial depth partition before extending any one leaf.
        # Refinement prioritizes the poorest sufficient bounds, with a fixed
        # stable tie order. All other represented regions remain in the union.
        for leaf in sorted(initial_leaves,key=lambda item:item['lower']):
            if not leaf['eligible'] or leaf['lower']>=-threshold:continue
            if adaptive_depth==depth:
                depth_exhausted=True
                continue
            if visited>=node_budget:
                budget_exhausted=True
                break
            leaf['lower']=max(leaf['lower'],recurse(*leaf['box'],adaptive_depth-depth))
        lower=max(lower,min((item['lower'] for item in initial_leaves),default=lower))
    if diagnostics is not None:
        diagnostics.update(visited_nodes=visited,node_budget_exhausted=budget_exhausted,
                           depth_exhausted=depth_exhausted,
                           unresolved_leaf_count=sum(item['lower']<-threshold for item in initial_leaves)
                           if adaptive else None)
    return lower,visited


def _divide_positive(numerator, denominator):
    """Directed interval division by a strictly positive denominator.

    All four endpoint combinations are consumed: a negative numerator
    over a positive denominator attains its minimum at the denominator's
    lower endpoint (N=-1, D in [1,2] gives [-1,-0.5]), so the endpoint
    pairing valid for nonnegative numerators inverts the result
    otherwise. The caller must supply denominator[0] > 0.
    """
    quotients = np.stack((numerator[0]/denominator[0],
                          numerator[0]/denominator[1],
                          numerator[1]/denominator[0],
                          numerator[1]/denominator[1]))
    return (_outward(np.min(quotients, axis=0), -np.inf),
            _outward(np.max(quotients, axis=0), np.inf))


_OVERLAP_WITNESS_SAMPLE_PARAMS = (
    (0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
    (0.5, 1.0), (0.0, 1.0), (0.0, 0.5), (0.5, 0.5),
)
_OVERLAP_WITNESS_SAMPLE_TIMES = (0.0, 0.5, 1.0)


def overlap_violating_point_witness(source_start,source_end,target_start,
                                    target_end,*,gap_tolerance,
                                    endpoint_errors=None,sample_params=None,
                                    sample_times=None,diagnostics=None):
    """Rigorous cheap refusal witness for one Q4/S3 pair path.

    A witness is reported only when BOTH directed enclosures hold at one
    exact parametric source point and time: the physical source-normal
    gap g = r*(P.Ns)/|Ns| — with r = N/(P.T) the target-foot radius
    (N = (s-c0).T the target-plane height, P the parent Q4 corner-0
    normal, Ns = dX/du x dX/dv the actual parametric Q4 normal at the
    sample point, so the foot y = s - rP gives g = (s-y).Ns/|Ns|) — is
    entirely below -gap_tolerance, and the foot has enclosed barycentric
    coordinates inside the target triangle. Together these prove an
    actual enclosed physical violating point inside the represented
    overlap, so the true minimum gap of the pair path is below
    -gap_tolerance and every valid lower bound of the same pair path —
    including any the expensive adaptive/dual certificate search would
    compute — refuses the budget anyway; running that search is wasted
    work. The target-plane height N/|T| alone is NOT this physical gap:
    with a tilted actual normal Ns the physical gap can be less negative
    than the target-plane height, so the height is never used as the
    witness quantity, only through the certified foot radius.

    This function never admits a path and never returns a domain lower
    bound; the enclosed point value is an upper bound of the local
    physical gap and is reported only as witness evidence. A not-found
    result is no conclusion at all: it retains the complete conservative
    path. Samples only locate a candidate witness; every reported
    quantity is a directed enclosure, and arbitrary supplied sample
    weights and times are enclosed rather than assumed exact (every
    enclosure is deliberately widened by one ULP outward, including the
    dyadic defaults, so no interval is claimed exact-width). Malformed
    inputs raise;
    an unenclosable or grazing chart at one sample skips that sample.
    Returns (found, samples_checked).
    """
    if (not np.isfinite(gap_tolerance) or gap_tolerance < 0.):
        raise ValueError('overlap witness needs a finite nonnegative gap tolerance')
    a0,a1,b0,b1=tuple(np.asarray(x,dtype=float) for x in
                      (source_start,source_end,target_start,target_end))
    errors=(0.,)*4 if endpoint_errors is None else endpoint_errors
    if len(errors)!=4:raise ValueError('overlap witness needs four error boxes')
    if (a0.shape!=(4,3) or a1.shape!=(4,3) or b0.shape!=(3,3) or b1.shape!=(3,3)
            or not all(np.all(np.isfinite(x)) for x in (a0,a1,b0,b1))):
        raise ValueError('overlap witness needs finite Q4/S3 endpoints')
    params=(_OVERLAP_WITNESS_SAMPLE_PARAMS if sample_params is None
            else tuple(sample_params))
    times=(_OVERLAP_WITNESS_SAMPLE_TIMES if sample_times is None
           else tuple(sample_times))
    for u,v in params:
        if (not np.isfinite(u) or not np.isfinite(v)
                or not 0.<=u<=1. or not 0.<=v<=1.):
            raise ValueError('overlap witness sample parameters lie in [0,1]^2')
    for t in times:
        if not np.isfinite(t) or not 0.<=t<=1.:
            raise ValueError('overlap witness sample times lie in [0,1]')
    if diagnostics is not None:
        if type(diagnostics) is not dict:
            raise ValueError('overlap witness diagnostics need a disposable dict')
        diagnostics.update(samples_checked=0,found=False,u=None,v=None,
                           t=None,physical_gap_upper_m=None,
                           foot_barycentric_lower=None,
                           foot_barycentric_upper=None)
    bounds=tuple(_enclose_positions(x,e) for x,e in zip((a0,a1,b0,b1),errors))
    checked=0
    for u,v in params:
        # Enclose the bilinear weights together with their rounded
        # factors; every enclosure is widened one ULP outward, dyadic
        # samples included, so arbitrary supplied samples stay
        # enclosed and none is claimed exact-width.
        mu=(_outward(1.-u,-np.inf),_outward(1.-u,np.inf))
        mv=(_outward(1.-v,-np.inf),_outward(1.-v,np.inf))
        uu,vv=_interval(u),_interval(v)
        weights=(_mul(mu,mv),_mul(uu,mv),_mul(uu,vv),_mul(mu,vv))
        for t in times:
            mt=(_outward(1.-t,-np.inf),_outward(1.-t,np.inf))
            tt=_interval(t)
            def lerp(first,second,k):
                return _add(_mul(mt,(first[0][k],first[1][k])),
                            _mul(tt,(second[0][k],second[1][k])))
            source=tuple(lerp(bounds[0],bounds[1],k) for k in range(4))
            target=tuple(lerp(bounds[2],bounds[3],k) for k in range(3))
            point=_mul(weights[0],source[0])
            for weight,corner in zip(weights[1:],source[1:]):
                point=_add(point,_mul(weight,corner))
            checked+=1
            edge1=_sub(target[1],target[0])
            edge2=_sub(target[2],target[0])
            normal=_cross(edge1,edge2)
            squared=_dot(normal,normal)
            if not all(np.all(np.isfinite(x)) for x in squared):
                continue
            if squared[0]<=0.:continue
            target_norm=(_outward(np.sqrt(squared[0]),-np.inf),
                         _outward(np.sqrt(squared[1]),np.inf))
            if not all(np.isfinite(x) and x>0. for x in target_norm):
                continue
            height=_dot(_sub(point,target[0]),normal)
            parent=_cross(_sub(source[1],source[0]),_sub(source[3],source[0]))
            projected=_dot(parent,normal)
            if not all(np.all(np.isfinite(x)) for x in
                       (height[0],height[1],projected[0],projected[1])):
                continue
            if projected[0]<=0.:continue
            radius=_divide_positive(height,projected)
            if not all(np.isfinite(x) for x in radius):continue
            foot=_sub(point,_mul(radius,parent))
            lambda1=_dot(_cross(_sub(foot,target[0]),edge2),normal)
            lambda2=_dot(_cross(edge1,_sub(foot,target[0])),normal)
            if not all(np.all(np.isfinite(x)) for x in
                       (lambda1[0],lambda1[1],lambda2[0],lambda2[1])):
                continue
            first=_divide_positive(lambda1,squared)
            second=_divide_positive(lambda2,squared)
            if not all(np.isfinite(x) for x in (*first,*second)):
                continue
            if (first[0]<0. or second[0]<0.
                    or _add(first,second)[1]>1.):
                continue
            # Actual parametric Q4 normal at the exact sample point:
            # dX/du = (1-v)(A1-A0) + v(A2-A3),
            # dX/dv = (1-u)(A3-A0) + u(A2-A1).
            du=_add(_mul(mv,_sub(source[1],source[0])),
                    _mul(vv,_sub(source[2],source[3])))
            dv=_add(_mul(mu,_sub(source[3],source[0])),
                    _mul(uu,_sub(source[2],source[1])))
            actual=_cross(du,dv)
            actual_squared=_dot(actual,actual)
            if not all(np.all(np.isfinite(x)) for x in actual_squared):
                continue
            if actual_squared[0]<=0.:continue
            actual_norm=(_outward(np.sqrt(actual_squared[0]),-np.inf),
                         _outward(np.sqrt(actual_squared[1]),np.inf))
            if not all(np.isfinite(x) and x>0. for x in actual_norm):
                continue
            parent_actual=_dot(parent,actual)
            if not all(np.all(np.isfinite(x)) for x in parent_actual):
                continue
            if parent_actual[0]<=0.:continue
            gap=_divide_positive(_mul(radius,parent_actual),actual_norm)
            if not all(np.isfinite(x) for x in gap):continue
            if gap[1]>=-gap_tolerance:continue
            if diagnostics is not None:
                diagnostics.update(found=True,u=float(u),v=float(v),t=float(t),
                                   physical_gap_upper_m=float(gap[1]),
                                   foot_barycentric_lower=
                                   [float(first[0]),float(second[0])],
                                   foot_barycentric_upper=
                                   [float(first[1]),float(second[1])],
                                   samples_checked=checked)
            return True,checked
    if diagnostics is not None:
        diagnostics.update(samples_checked=checked)
    return False,checked


def _validate_overlap_local_point_chart(source, target):
    """Preserve the point fallback's chart refusal before decision pruning.

    This performs no gap integration or admission. The caller also validates
    every physical face before conservative candidate-pair pruning.
    """
    _, normal = projected_q4_source_face(source)
    axis = np.cross(target[1]-target[0], target[2]-target[0])
    if axis @ normal < 0.0:
        axis = -axis
    if float(axis @ normal) <= 0.25*np.linalg.norm(axis):
        raise ValueError("target plane is nearly grazing")


def overlap_local_lower(source: np.ndarray, target: np.ndarray,
                        depth: int) -> tuple[float, int]:
    if depth < 0 or depth > 7:
        raise ValueError("experimental subdivision depth must be in [0, 7]")
    _, normal = projected_q4_source_face(source)
    tangent = source[1]-source[0]
    tangent -= float(tangent @ normal)*normal
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    frame = np.stack((tangent, bitangent), axis=1)
    axis = np.cross(target[1]-target[0], target[2]-target[0])
    if axis @ normal < 0.0:
        axis = -axis
    denominator = float(axis @ normal)
    if denominator <= 0.25*np.linalg.norm(axis):
        raise ValueError("target plane is nearly grazing")
    target_plane = target - ((target-source[0]) @ normal)[:, None]*normal
    target_2d = target_plane @ frame
    target_low = np.min(target_2d, axis=0)
    target_high = np.max(target_2d, axis=0)
    scale = max(1.0, float(np.max(np.abs(source))),
                float(np.max(np.abs(target))))
    margin = 1024.0*np.finfo(float).eps*scale
    visited = 0

    def recurse(u0: float, u1: float, v0: float, v1: float,
                remaining: int) -> float:
        nonlocal visited
        visited += 1
        corners = np.asarray((
            _point(source, u0, v0), _point(source, u1, v0),
            _point(source, u1, v1), _point(source, u0, v1),
        ))
        projected = corners - ((corners-source[0]) @ normal)[:, None]*normal
        box = projected @ frame
        if np.any(np.max(box, axis=0) + margin < target_low) or np.any(
                target_high + margin < np.min(box, axis=0)):
            return float("inf")
        if remaining == 0:
            # Target-plane height is affine in world position; source Q1
            # position is bilinear. Its minimum on a parametric rectangle is
            # therefore at one of these four corners, even if the projected
            # overlap occupies only a subset of the rectangle.
            numerator = float(np.min(corners @ axis)-np.max(target @ axis))
            return min(0.0, numerator/denominator) - margin
        um, vm = 0.5*(u0+u1), 0.5*(v0+v1)
        return min(recurse(*bounds, remaining-1) for bounds in (
            (u0, um, v0, vm), (um, u1, v0, vm),
            (um, u1, vm, v1), (u0, um, vm, v1),
        ))

    return recurse(0.0, 1.0, 0.0, 1.0, depth), visited
def overlap_local_swept_lower(
        source_start: np.ndarray, source_end: np.ndarray,
        target_start: np.ndarray, target_end: np.ndarray,
        depth: int, *, endpoint_errors=None) -> tuple[float, int]:
    """Experimental bound along a *linear physical nodal* Q4/S3 path.

    A fixed frame encloses moving projected boxes. The expansion bounds the
    lateral drift of a matched pair under the parent Q4 projection normal.
    Every retained source subrectangle is bounded over space and time; no
    quadrature sample or instantaneous-only clipping decides path safety.
    """
    if depth < 0 or depth > 7:
        raise ValueError("experimental subdivision depth must be in [0, 7]")
    a0, a1, b0, b1 = (np.asarray(item, dtype=float) for item in (
        source_start, source_end, target_start, target_end,
    ))
    errors = ((0.0,)*4 if endpoint_errors is None else endpoint_errors)
    if len(errors) != 4:
        raise ValueError("swept gap needs four endpoint enclosures")
    input_bounds = tuple(_enclose_positions(item, error)
                         for item, error in zip((a0, a1, b0, b1), errors))
    if (a0.shape != (4, 3) or a1.shape != (4, 3)
            or b0.shape != (3, 3) or b1.shape != (3, 3)
            or not all(np.all(np.isfinite(item)) for item in (a0, a1, b0, b1))):
        raise ValueError("experimental sweep needs Q4/S3 endpoints")
    _, reference = projected_q4_source_face(a0)
    projected_q4_source_face(a1)
    tangent = a0[1]-a0[0]
    tangent -= float(tangent @ reference)*reference
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(reference, tangent)
    frame = np.stack((tangent, bitangent), axis=1)
    scale = max(1.0, *(float(np.max(np.abs(item)))
                       for item in (a0, a1, b0, b1)))
    # Treat stored endpoint/frame floats as exact inputs to this bound.
    # Directed outward arithmetic encloses normal controls and denominators;
    # a positive rounded normal height alone cannot certify a grazing path.
    parent = _normal_controls_interval(a0, a1,
                                       start_error=errors[0], end_error=errors[1])
    if not all(np.all(np.isfinite(value)) for value in parent):
        return -float("inf"), 1
    parent_height_min = float(np.min(_dot(parent, _interval(reference))[0]))
    if not np.isfinite(parent_height_min) or parent_height_min <= 0.0:
        return -float("inf"), 1
    lateral = [_dot(parent, _interval(frame[:, axis_index]))
               for axis_index in (0, 1)]
    drift_slope = np.asarray([
        _positive_divide_upper(
            np.max(np.maximum(np.abs(value[0]), np.abs(value[1]))),
            parent_height_min) for value in lateral])
    if not np.all(np.isfinite(drift_slope)):
        return -float("inf"), 1
    axis = np.cross(b0[1]-b0[0], b0[2]-b0[0])
    axis_length = np.linalg.norm(axis)
    if not np.isfinite(axis_length) or axis_length <= 0.0:
        return -float("inf"), 1
    axis /= axis_length
    if axis @ reference < 0.0:
        axis = -axis

    axis_norm_upper = float(_norm_upper(_interval(axis)))

    def cone_cosine(values):
        maximum = float(np.max(_norm_upper(values)))
        denominator = _outward(maximum*axis_norm_upper, np.inf)
        numerator = float(np.min(_dot(values, _interval(axis))[0]))
        return (float(_outward(numerator/denominator, -np.inf))
                if np.isfinite(numerator) and np.isfinite(denominator)
                and denominator > 0.0 else -float("inf"))

    parent_cosine = cone_cosine(parent)
    physical_cosines = [cone_cosine(_normal_controls_interval(
        a0, a1, corner, errors[0], errors[1]))
                        for corner in range(4)]
    if parent_cosine <= 0.75 or min(physical_cosines) <= 0.75:
        return -float("inf"), 1
    target_endpoints = tuple(np.vstack((input_bounds[2][side],
                                       input_bounds[3][side]))
                             for side in (0, 1))
    target_box = [_dot(target_endpoints, _interval(frame[:, index]))
                  for index in (0, 1)]
    target_low = np.asarray([np.min(value[0]) for value in target_box])
    target_high = np.asarray([np.max(value[1]) for value in target_box])
    target_heights = _dot(target_endpoints, _interval(reference))
    if not all(np.all(np.isfinite(value))
               for bounds in (*target_box, target_heights) for value in bounds):
        return -float("inf"), 1
    # Dyadic subdivision coordinates through depth 7 have exact shape
    # coefficients. Seven rounded multiply/add operations evaluate _point;
    # gamma_8 bounds its componentwise error for the convex Q1 weights.
    unit = np.finfo(float).eps/2.0
    point_error = _outward((8*unit/(1-8*unit))*scale + np.maximum(
        np.max(np.broadcast_to(np.asarray(errors[0]), a0.shape), axis=0),
        np.max(np.broadcast_to(np.asarray(errors[1]), a1.shape), axis=0)), np.inf)
    visited = 0

    def recurse(u0: float, u1: float, v0: float, v1: float,
                remaining: int) -> float:
        nonlocal visited
        visited += 1
        a_start = np.asarray((
            _point(a0, u0, v0), _point(a0, u1, v0),
            _point(a0, u1, v1), _point(a0, u0, v1),
        ))
        a_end = np.asarray((
            _point(a1, u0, v0), _point(a1, u1, v0),
            _point(a1, u1, v1), _point(a1, u0, v1),
        ))
        endpoints = np.vstack((a_start, a_end))
        positions = (_outward(endpoints-point_error, -np.inf),
                     _outward(endpoints+point_error, np.inf))
        heights = _dot(positions, _interval(reference))
        difference_max = max(
            float(_outward(np.max(heights[1])-np.min(target_heights[0]), np.inf)),
            float(_outward(np.max(target_heights[1])-np.min(heights[0]), np.inf)),
        )
        drift = _outward(difference_max*drift_slope, np.inf)
        box = [_dot(positions, _interval(frame[:, index])) for index in (0, 1)]
        box_low = np.asarray([np.min(value[0]) for value in box])
        box_high = np.asarray([np.max(value[1]) for value in box])
        if np.any(_outward(box_high+drift, np.inf) < target_low) or np.any(
                _outward(target_high+drift, np.inf) < box_low):
            return float("inf")
        if remaining == 0:
            differences = _sub(
                (positions[0].reshape(2, 4, 3)[:, :, None],
                 positions[1].reshape(2, 4, 3)[:, :, None]),
                tuple(np.stack((input_bounds[2][side],
                                input_bounds[3][side]))[:, None, :]
                      for side in (0, 1)))
            numerator = float(np.min(_dot(differences, _interval(axis))[0]))
            denominator = _outward(parent_cosine*axis_norm_upper, -np.inf)
            if (not np.isfinite(numerator) or not np.isfinite(denominator)
                    or denominator <= 0.0):
                return -float("inf")
            return float(_outward(min(0.0, numerator)/denominator, -np.inf))
        um, vm = 0.5*(u0+u1), 0.5*(v0+v1)
        return min(recurse(*bounds, remaining-1) for bounds in (
            (u0, um, v0, vm), (um, u1, v0, vm),
            (um, u1, vm, v1), (u0, um, vm, v1),
        ))

    return recurse(0.0, 1.0, 0.0, 1.0, depth), visited
