"""Sufficient whole-chord coverage for one moving planar S3 target.

Binary fixed-map operands and stored dense material points are enclosed by
rational interval polynomials. No sampled-pose or affine-normal assumption.
"""
from fractions import Fraction as F
import math
import numpy as np
from .s3_friction_coverage import (add, cross, magnitude, mapped_path,
    material_point_paths, positive, vsub)


def verified_norm(raw):
    """Verify a sqrt bracket against the exact binary-input square sum."""
    square = sum(F(float(x))**2 for x in raw)
    if square <= 0:
        raise ValueError("moving S3 normal is degenerate")
    estimate = math.sqrt(float(square))
    lo = hi = estimate
    for _ in range(8):
        if F(lo)**2 <= square <= F(hi)**2:
            if not 0 < lo <= hi < np.finfo(float).max**.25:
                break
            return estimate
        if F(lo)**2 > square: lo = math.nextafter(lo, 0.)
        if F(hi)**2 < square: hi = math.nextafter(hi, math.inf)
    raise ValueError("moving S3 normal norm is not representably enclosed")


def _widen(p, error):
    return ((p[0][0]-error, p[0][1]+error), *p[1:])


def _raw_normal(points):
    u = F(1, 2**53); floor = F(1, 2**1074)
    edges=[]
    for p in points[1:]:
        edge = vsub(p, points[0])
        edges.append(tuple(_widen(v, u*(magnitude(a)+magnitude(b))/(1-u)+floor)
                           for v,a,b in zip(edge,p,points[0])))
    raw=cross(*edges); gamma=3*u/(1-3*u)
    return tuple(_widen(p, gamma*(magnitude(edges[0][j])*magnitude(edges[1][k])
                               +magnitude(edges[0][k])*magnitude(edges[1][j]))+3*floor)
                 for p,(j,k) in zip(raw,((1,2),(2,0),(0,1))))


def _dot(a,b):
    from .s3_friction_coverage import product
    terms=[product(x,y) for x,y in zip(a,b)]
    return add(add(terms[0],terms[1]),terms[2])


def _component_lower(vector):
    def lower(p):
        degree=len(p)-1
        return min(sum(p[i][0]*F(math.comb(k,i),math.comb(degree,i))
                       for i in range(k+1)) for k in range(degree+1))
    return max(max(lower(p),lower(tuple((-hi,-lo) for lo,hi in p))) for p in vector)


def _margin_positive(poly,margin):
    return positive(((poly[0][0]-margin,poly[0][1]-margin),*poly[1:]))


def full_moving_coverage(owner, first, last):
    first=np.asarray(first,float); last=np.asarray(last,float)
    if (first.shape != (owner.n_dof,) or last.shape != first.shape
            or len(owner.target_adapters) != 1 or len(owner.source_adapters) != 1):
        return False
    cap=np.finfo(float).max**.125
    arrays=[first,last,owner.point_reference,owner.point_maps]
    for a in (*owner.source_adapters,*owner.target_adapters.values()):
        arrays.extend((a.reference_positions,a.work_map.data))
    if any(not np.isfinite(a).all() or np.max(np.abs(a),initial=0.)>cap for a in arrays):
        return False
    target=next(iter(owner.target_adapters.values()))
    points=mapped_path(target,first,last)[0]
    raw=_raw_normal(points)
    reference=np.cross(target.reference_positions[1]-target.reference_positions[0],
                       target.reference_positions[2]-target.reference_positions[0])
    # Polarity and a positive lower bound on a raw component protect the
    # normal square/sum normalization against overflow/underflow.
    from .s3_friction_coverage import dot_axis
    if not positive(dot_axis(raw, tuple(F(float(x)) for x in reference))): return False
    minimum=F(float(np.finfo(float).tiny**.25))
    def safe_normal(raw):
        def above_minimum(p):
            return positive(((p[0][0]-minimum,p[0][1]-minimum),*p[1:]))
        return (any(above_minimum(p) or
                    above_minimum(tuple((-hi,-lo) for lo,hi in p)) for p in raw)
                and max(map(magnitude,raw)) <= F(float(np.finfo(float).max**.25)))
    if not safe_normal(raw): return False
    # After multiplication by the positive verified norm, each rounded
    # normalized component differs from raw by <=u/(1-u)*|raw| plus the
    # division subnormal floor times an upper bound on that norm.
    u=F(1,2**53); floor=F(1,2**1074); norm_upper=2*sum(map(magnitude,raw))
    direction=tuple(_widen(p,u*magnitude(p)/(1-u)+floor*norm_upper) for p in raw)
    source=mapped_path(owner.source_adapters[0],first,last)[0]
    sn=_raw_normal(source)
    if not safe_normal(sn): return False
    if not positive(_dot(sn,direction)): return False
    sn_upper=2*sum(map(magnitude,sn))
    source_direction=tuple(_widen(p,u*magnitude(p)/(1-u)+floor*sn_upper) for p in sn)
    material=[*source,*material_point_paths(owner,first,last)]
    from .s3_friction_chart_interval import full_branch
    if not full_branch(source,points,material[len(source):]): return False
    for i in range(3):
        origin=points[i]; edge=vsub(points[(i+1)%3],origin)
        for p in material:
            if not positive(_dot(cross(edge,vsub(p,origin)),direction)): return False
            # Normal mortar projects along the source normal; friction uses
            # the orthogonal target projection. Both actual domains must fit.
            if not positive(_dot(cross(edge,vsub(p,origin)),source_direction)): return False
    return True
