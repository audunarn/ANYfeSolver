"""Whole-path Q4 gap/chart enclosure for affine solver coordinates.

Physical corners are nonaffine. An analytic exponential curvature remainder
and enclosed map evaluation inflate the chord boxes at every time, including
between subdivision points. This is a sufficient certificate, not sampling.
"""
from __future__ import annotations
import numpy as np
from .finite_rotation_skin import (
    _up, curvature_radius, director_norm_upper, uniform_skin_evaluation_error,
)
from .q4_overlap_local_bound import (
    _interval, _enclose_positions, _normal_controls_interval,
    _sub, _dot, _mul, _add, _outward,
)
from .q4_q4_overlap_derivatives import q4_q4_swept_chart_admissible


def finite_whole_pair_lower(a0, a1, b0, b1, reference_source,
                            reference_target, *, endpoint_errors):
    """Directed cubic numerator with an interval Euclidean normal lower.

    This finite route alone improves the coordinate-axis denominator. Each
    raw-normal component lies in the convex Bernstein hull. Its distance to
    zero is a lower absolute component bound, and the directed sum of squares
    bounds the squared norm. A downward root is verified by an upper-rounded
    square. Every chart predicate remains a prerequisite. Legacy code is kept.
    """
    errors=endpoint_errors
    if len(a0)==len(a1)==len(b0)==len(b1)==3:
        # P1 normals are constant on each face but quadratic along the chord.
        # Outward Bernstein hulls include the finite skin endpoint/curvature
        # boxes. Positive pairwise control dots certify all intermediate
        # normals, relative projection and original reference polarity.
        an=_normal_controls_interval(a0,a1,0,errors[0],errors[1])
        bn=_normal_controls_interval(b0,b1,0,errors[2],errors[3])
        ar=_normal_controls_interval(reference_source,reference_source,0)
        br=_normal_controls_interval(reference_target,reference_target,0)
        for first,second in ((an,bn),(an,ar),(bn,br)):
            dots=_dot(tuple(v[:,None,:] for v in first),tuple(v[None,:,:] for v in second))
            if not np.isfinite(dots[0]).all() or np.min(dots[0])<=0.:
                return -float('inf')
        normals=tuple(v[None,:,:] for v in an)
    else:
        if not q4_q4_swept_chart_admissible(a0,a1,b0,b1,reference_source,
                                           reference_target,endpoint_errors=errors):
            return -float('inf')
        normals=tuple(np.stack([_normal_controls_interval(a0,a1,k,errors[0],errors[1])[s]
                               for k in range(4)]) for s in (0,1))
    bounds=tuple(_enclose_positions(x,e) for x,e in zip((a0,a1,b0,b1),errors))
    lo=np.min(normals[0],axis=(0,1));hi=np.max(normals[1],axis=(0,1))
    distance=np.maximum(0.,np.maximum(lo,-hi))
    squares=np.maximum(0.,np.nextafter(distance*distance,-np.inf))
    norm2=float(np.nextafter(np.nextafter(squares[0]+squares[1],-np.inf)+squares[2],-np.inf))
    if not np.isfinite(norm2) or norm2<=0.:return -float('inf')
    denominator=float(np.nextafter(np.sqrt(norm2),-np.inf))
    while np.nextafter(denominator*denominator,np.inf)>norm2:
        denominator=float(np.nextafter(denominator,-np.inf))
    if denominator<=0.:return -float('inf')
    d0=_sub(tuple(v[:,None,:] for v in bounds[0]),tuple(v[None,:,:] for v in bounds[2]))
    d1=_sub(tuple(v[:,None,:] for v in bounds[1]),tuple(v[None,:,:] for v in bounds[3]))
    def dot(d,k):return _dot(tuple(v[:,:,None,:] for v in d),tuple(v[None,None,:,k,:] for v in normals))
    def third(v):return _outward(v[0]/3.,-np.inf),_outward(v[1]/3.,np.inf)
    controls=(dot(d0,0),third(_add(_mul(dot(d0,1),_interval(2.)),dot(d1,0))),
              third(_add(dot(d0,2),_mul(dot(d1,1),_interval(2.)))),dot(d1,2))
    numerator=min(float(np.min(v[0])) for v in controls)
    if not np.isfinite(numerator):return -float('inf')
    return float(_outward(min(0.,numerator)/denominator,-np.inf))


def _affine_point(start, end, parameter):
    """Enclose the exact affine point of the persisted binary original path."""
    if parameter == 0.:
        return start.copy(), np.zeros_like(start)
    if parameter == 1.:
        return end.copy(), np.zeros_like(end)
    difference = end-start
    value = start+parameter*difference
    # Three operations; subtraction absolute majorant uses both original
    # endpoints, so cancellation does not invalidate the coordinate enclosure.
    eps = np.finfo(float).eps
    gamma = np.nextafter(3*eps/(1.-3*eps), np.inf)
    magnitude = _up(np.abs(start)+_up(parameter*_up(np.abs(end)+np.abs(start))))
    error = _up(gamma*magnitude+3*np.nextafter(0., 1.))
    return value, error


def _pose(skin, start, end, parameter):
    q, qerror = _affine_point(start, end, parameter)
    pose = skin.pose(q)
    errors = qerror[skin.dofs]
    # Exp is Lipschitz in a skew perturbation with norm <= ||delta theta||;
    # orthogonal exponential integral formula has no exp(||theta||) factor.
    theta_error = _up(_up(errors[:, 3]+errors[:, 4])+errors[:, 5])
    rotation_error = _up(_up(np.abs(skin.offsets)*director_norm_upper(skin.directors))
                         *theta_error)
    position_error = _up(_up(pose.position_errors+errors[:, :3])
                         +rotation_error[:, None])
    return q, qerror, pose, position_error


def finite_q4_path_admissible(manager, start, end):
    """Certify every original face pair and the original nonaffine q path."""
    start = np.asarray(start, float).reshape(-1)
    end = np.asarray(end, float).reshape(-1)
    skin = manager.finite_skin
    if (start.shape != (skin.n_dof,) or end.shape != start.shape
            or not np.isfinite(start).all() or not np.isfinite(end).all()):
        raise ValueError('finite skin path has invalid coordinates')
    geometry = manager.law.geometry
    split = len(manager.law.source_node_ids)

    def lower(a, b, ea, eb):
        from .q4_overlap_local_bound import q4_s3_whole_pair_swept_lower
        values=[]
        for i,si in enumerate(manager.law.source_face_indices):
            for j,ti in enumerate(manager.law.target_face_indices):
                operands=(a[:split][si],b[:split][si],a[split:][ti],b[split:][ti])
                errors=(ea[:split][si],eb[:split][si],ea[split:][ti],eb[split:][ti])
                if len(si)==len(ti)==3:
                    values.append(finite_whole_pair_lower(*operands,
                        geometry.source_faces[i].reference_positions,
                        geometry.target_faces[j].reference_positions,endpoint_errors=errors))
                elif len(ti)==3:
                    values.append(q4_s3_whole_pair_swept_lower(*operands,endpoint_errors=errors))
                else:
                    values.append(finite_whole_pair_lower(*operands,
                        geometry.source_faces[i].reference_positions,
                        geometry.target_faces[j].reference_positions,endpoint_errors=errors))
        return min(values)

    left = _pose(skin, start, end, 0.)
    right = _pose(skin, start, end, 1.)
    if lower(left[2].positions, left[2].positions, left[3], left[3]) < -manager.gap_tolerance:
        raise ValueError('committed finite Q4 skin chart/gap unsafe or uncertified')
    if lower(right[2].positions, right[2].positions, right[3], right[3]) < -manager.gap_tolerance:
        return False

    def certify(a, b, pa, pb, depth):
        rotations_a = pa[0][skin.dofs[:, 3:]]
        rotations_b = pb[0][skin.dofs[:, 3:]]
        curve = curvature_radius(
            rotations_a, rotations_b, skin.offsets, skin.directors,
            coordinate_errors=(pa[1][skin.dofs[:, 3:]], pb[1][skin.dofs[:, 3:]]))
        # Endpoint boxes alone cover the ideal exponential field. Include a
        # uniform evaluation bound too, so an interior numerical-map value
        # cannot escape simply because its rounding differs from the endpoints.
        evaluation = uniform_skin_evaluation_error(
            skin, pa[0], pb[0], pa[1], pb[1])
        curve = _up(curve[:, None]+evaluation)
        ea = _up(pa[3]+curve); eb = _up(pb[3]+curve)
        if lower(pa[2].positions, pb[2].positions, ea, eb) >= -manager.gap_tolerance:
            return True
        if depth >= manager.settings.maximum_swept_subdivisions:
            return False
        middle = (a+b)*.5  # exactly represented dyadic times at this bound
        pm = _pose(skin, start, end, middle)
        return (certify(a, middle, pa, pm, depth+1)
                and certify(middle, b, pm, pb, depth+1))

    return certify(0., 1., left, right, 0)
