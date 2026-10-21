"""Development-only whole-face certificate caller and first-touch height.

No new certificate is defined here. Every non-disjoint sphere-S3 / panel-Q4 pair
retains the existing parent certificate and two fixed dyadic partition levels of
its P1/Q1 parameter domains. Every child uses the same
``q4_s3_whole_pair_swept_lower`` with start equal to end (an endpoint, not a swept path).
This is proof subdivision only; the physical/contact face inventory is unchanged.
The original bilinear Q4
is given in reversed chart order so that its normal points away from the sphere
and a positive value means separation; the faces and interpolation are the
original ones. ``-inf`` (unavailable chart/normal predicates) is never hidden.

Pruning proof: a triangle (convex hull of its corners) and an original bilinear
Q4 (convex combination of its corners) lie in the axis boxes of their corners,
widened by the pose position-error enclosures. If the boxes are strictly
separated along some coordinate axis, the faces are disjoint and cannot
penetrate; the pair is then reported as pruned with bound 0.
"""
from __future__ import annotations

import numpy as np

from .q4_overlap_local_bound import q4_s3_whole_pair_swept_lower

PENETRATION_LIMIT_M = 50e-6
_Q4_REVERSED = np.array((0, 3, 2, 1))


class CornerContactRefusal(ValueError):
    """Typed refusal; never converted into a plausible number."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def panel_surface_height(positions, face_slices, x, y, *, tolerance=1e-12):
    """z of the ACTUAL bilinear panel skin above (x, y); None outside; refuse ambiguity."""
    heights = []
    for start, end in face_slices:
        c = positions[start:end]
        if (x < c[:, 0].min() - tolerance or x > c[:, 0].max() + tolerance
                or y < c[:, 1].min() - tolerance or y > c[:, 1].max() + tolerance):
            continue
        xi = eta = 0.
        for _ in range(40):
            n = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                 (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
            dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            residual = n @ c[:, :2] - (x, y)
            jac = np.column_stack((dxi @ c[:, :2], deta @ c[:, :2]))
            step = np.linalg.solve(jac, -residual)
            xi, eta = xi + step[0], eta + step[1]
            if float(np.max(np.abs(step))) < 1e-15:
                break
        n = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        if (abs(xi) <= 1 + 1e-9 and abs(eta) <= 1 + 1e-9
                and float(np.linalg.norm(n @ c[:, :2] - (x, y))) < 1e-12):
            heights.append(float(n @ c[:, 2]))
    if not heights:
        return None
    if max(heights) - min(heights) > 1e-10:
        raise CornerContactRefusal(
            "ambiguous_surface_height", "shared panel faces disagree on height")
    return heights[0]


def first_touch_lift(sphere_positions, panel_positions, panel_slices):
    """Rigid vertical lift at which the lowest sphere corner first meets the panel."""
    lifts = []
    for p in sphere_positions:
        z = panel_surface_height(panel_positions, panel_slices, float(p[0]), float(p[1]))
        if z is None:
            raise CornerContactRefusal(
                "outside_panel", "a sphere corner lies outside the panel footprint")
        lifts.append(z - float(p[2]))
    lifts = np.asarray(lifts)
    order = np.argsort(lifts)[::-1]
    return float(lifts[order[0]]), [int(i) for i in order[:6]], [float(lifts[i]) for i in order[:6]]


def directional_exclusion(s_lo, s_hi, p_lo, p_hi):
    """Pairs a sphere face box cannot penetrate in the oriented (panel-up) gap sense.

    ``above``: the sphere face is wholly ABOVE the panel face (strict, with the
    error enclosures already in the boxes): positive separation along +z, and
    with the panel normals pointing up this is a positive oriented gap.
    ``horizontal``: the boxes are strictly separated in x or y: the sphere face
    does not overlap this panel face in the panel's up-projection; its signed gap
    is then governed by the panel faces it does overlap (every sphere face must
    overlap at least one, checked by the caller). A face wholly BELOW the panel
    is never excluded here: disjoint surface patches do not imply a nonnegative
    oriented gap.
    """
    above = s_lo[2] > p_hi[:, 2]
    horizontal = np.any((s_lo[:2] > p_hi[:, :2]) | (s_hi[:2] < p_lo[:, :2]), axis=1)
    return above, horizontal


# Complete, orientation-preserving parameter partitions; no adaptive depth.
_S3_POINTS = np.array(((1.,0.,0.),(0.,1.,0.),(0.,0.,1.),
                       (.5,.5,0.),(0.,.5,.5),(.5,0.,.5)))
_S3_CELLS = ((0,3,5),(3,1,4),(5,4,2),(3,4,5))
_Q4_PARAMS = np.array(((0.,0.),(.5,0.),(1.,0.),(0.,.5),(.5,.5),(1.,.5),
                        (0.,1.),(.5,1.),(1.,1.)))
_Q4_WEIGHTS = np.array([((1-u)*(1-v),u*(1-v),u*v,(1-u)*v) for u,v in _Q4_PARAMS])
_Q4_CELLS = ((0,1,4,3),(1,2,5,4),(4,5,8,7),(3,4,7,6))


def _exact_affine_points(positions, errors, weights):
    """Enclose exact dyadic skin interpolation including evaluation rounding.

    Fractions represent original float positions, radii and weights exactly.
    A rounded output position is paired with radius sum(w*input_error) plus
    its exact rounding discrepancy, rounded outward. No pose error is dropped.
    """
    from fractions import Fraction as F
    p,e,w = map(lambda a:np.asarray(a,dtype=float),(positions,errors,weights))
    if (p.ndim != 2 or p.shape[1] != 3 or e.shape != p.shape
            or w.ndim != 2 or w.shape[1] != len(p)
            or not all(np.all(np.isfinite(a)) for a in (p,e,w))
            or np.any(e < 0.) or np.any(w < 0.)):
        raise ValueError("fixed subdivision needs finite coordinates/nonnegative error radii")
    out, radius = np.empty((len(w),3)),np.empty((len(w),3))
    for j,row in enumerate(w):
        ww = [F(float(x)) for x in row]
        if sum(ww) != 1:
            raise ValueError("subdivision weights must sum exactly to one")
        for axis in range(3):
            exact = sum(weight*F(float(p[k,axis])) for k,weight in enumerate(ww))
            value = float(exact)
            err = sum(weight*F(float(e[k,axis])) for k,weight in enumerate(ww))
            err += abs(F(value)-exact)
            out[j,axis] = value
            radius[j,axis] = np.nextafter(float(err),np.inf) if err else 0.
    if not np.all(np.isfinite(out)) or not np.all(np.isfinite(radius)):
        raise ValueError("fixed subdivision point enclosure overflow")
    return out,radius


def _cell_box(positions,errors):
    return (np.nextafter(positions-errors,-np.inf).min(axis=0),
            np.nextafter(positions+errors,np.inf).max(axis=0))


def _fixed_pair_lower(sphere,sphere_error,panel,panel_error):
    """Two complete 4x4 partition levels, retaining every available ancestor bound.

    Counts include actually visited cells at both levels (pruned subtrees are not
    visited). Failure paths give one local cell index per visited level for each
    parameter domain. Unavailable intermediate cells are refused without descent.
    """
    checked,pruned,unavailable = 0,0,[]

    def partition(sphere,sphere_error,panel,panel_error,s_path=(),q_path=()):
        nonlocal checked,pruned
        sp,se = _exact_affine_points(sphere,sphere_error,_S3_POINTS)
        pp,pe = _exact_affine_points(panel,panel_error,_Q4_WEIGHTS)
        lower = 0.
        for si,indices in enumerate(_S3_CELLS):
            t,te = sp[list(indices)],se[list(indices)]
            lo,hi = _cell_box(t,te)
            for qi,qindices in enumerate(_Q4_CELLS):
                q,qe = pp[list(qindices)],pe[list(qindices)]
                s_child,q_child = s_path+(si,),q_path+(qi,)
                qlo,qhi = _cell_box(q,qe)
                above,horizontal = directional_exclusion(lo,hi,qlo[None,:],qhi[None,:])
                if above[0] or horizontal[0]:
                    pruned += 1
                    continue
                # Reverse only for the bound, retaining the original Q1 chart
                # when restricting the same surface for the next fixed level.
                qr,qer = q[_Q4_REVERSED],qe[_Q4_REVERSED]
                value = float(q4_s3_whole_pair_swept_lower(
                    qr,qr,t,t,endpoint_errors=(qer,qer,te,te)))
                checked += 1
                if not np.isfinite(value):
                    unavailable.append((s_child,q_child))
                    value = -np.inf
                elif len(s_child) < 2:
                    child = partition(t,te,q,qe,s_child,q_child)
                    # A finite ancestor must never conceal an unavailable leaf.
                    value = max(value,child) if np.isfinite(child) else -np.inf
                lower = min(lower,value)
        return lower

    lower = partition(sphere,sphere_error,panel,panel_error)
    return lower,checked,pruned,unavailable


def whole_face_certificates(sphere_map, sphere_pose, panel_map, panel_pose):
    """Certificates for every sphere-face/panel-face pair (no assumed cap)."""
    sp, pp = sphere_pose.positions, panel_pose.positions
    se, pe = sphere_pose.position_errors, panel_pose.position_errors
    s_lo, s_hi, p_lo, p_hi = [], [], [], []
    for a, b in sphere_map.face_slices:
        s_lo.append((sp[a:b] - se[a:b]).min(axis=0))
        s_hi.append((sp[a:b] + se[a:b]).max(axis=0))
    for a, b in panel_map.face_slices:
        p_lo.append((pp[a:b] - pe[a:b]).min(axis=0))
        p_hi.append((pp[a:b] + pe[a:b]).max(axis=0))
    s_lo, s_hi, p_lo, p_hi = map(np.asarray, (s_lo, s_hi, p_lo, p_hi))
    values, pruned, unavailable = [], 0, []
    coarse_values, child_checked, child_pruned, unavailable_cells = [],0,0,[]
    for i, (sa, sb) in enumerate(sphere_map.face_slices):
        above, horizontal = directional_exclusion(s_lo[i], s_hi[i], p_lo, p_hi)
        if horizontal.all():
            raise CornerContactRefusal(
                "outside_panel", f"sphere face {i} overlaps no panel face horizontally")
        disjoint = above | horizontal
        for j, (pa, pb) in enumerate(panel_map.face_slices):
            if disjoint[j]:
                pruned += 1
                continue
            q = pp[pa:pb][_Q4_REVERSED]
            qe = pe[pa:pb][_Q4_REVERSED]
            coarse = float(q4_s3_whole_pair_swept_lower(
                q, q, sp[sa:sb], sp[sa:sb], endpoint_errors=(qe, qe, se[sa:sb], se[sa:sb])))
            coarse_values.append(coarse)
            if not np.isfinite(coarse):
                # Preserve parent chart/refusal, never use children to bypass it.
                unavailable.append((i,j))
                value = -np.inf
            else:
                child,checked,skipped,failed = _fixed_pair_lower(
                    sp[sa:sb],se[sa:sb],pp[pa:pb],pe[pa:pb])
                child_checked += checked
                child_pruned += skipped
                unavailable_cells.extend((i,j,si,qi) for si,qi in failed)
                if failed:
                    unavailable.append((i,j))
                    value = -np.inf
                else:
                    # Both are valid lower bounds of the same complete geometry.
                    value = max(coarse,child)
            values.append((value, i, j))
    values.sort()
    return dict(
        pairs_total=len(sphere_map.face_slices) * len(panel_map.face_slices),
        pairs_pruned_disjoint=pruned, pairs_certified=len(values),
        unavailable_pairs=unavailable,
        minimum_bound_m=values[0][0] if values else 0.,
        worst=[(v, int(i), int(j)) for v, i, j in values[:5]],
        coarse_minimum_bound_m=min(coarse_values) if coarse_values else 0.,
        fixed_partition=dict(s3_cells=16,q4_cells=16,levels=2,
                             pairs_checked=child_checked,pairs_pruned=child_pruned,
                             unavailable_cells=unavailable_cells),
    )
