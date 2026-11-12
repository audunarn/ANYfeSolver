"""Directed sufficient whole-pair gap bound for the Q4/Q4 development path.

Both original bilinear surfaces and their affine physical nodal path are kept.
This does not certify projection inversion, coverage, or finite rotations.
"""
from __future__ import annotations
import numpy as np
from .q4_overlap_local_bound import (
    _interval, _enclose_positions, _normal_controls_interval, _sub, _dot,
    _mul, _add, _outward,
)
from .q4_q4_overlap_derivatives import q4_q4_swept_chart_admissible


def q4_q4_whole_pair_swept_lower(source_start, source_end, target_start,
        target_end, reference_source, reference_target, *, endpoint_errors=None):
    """Lower bound every signed source-normal material-point gap for 0<=t<=1.

    D_ij(t) is affine and raw corner normal N_k(t) quadratic. Their product
    has cubic Bernstein controls. Q1 positions and original Q4 raw normals
    are convex combinations, so retaining all 4x4x4 triples is sufficient.
    A strictly positive signed coordinate component bounds |N| from below
    without computing a rounded unit axis or square root. For cmin<0,
    cmin/L is a lower bound; for cmin>=0 zero is sufficient. All operations
    enclose exact binary endpoint values plus supplied absolute map errors.
    Unresolved chart or denominator returns -inf, never the old float cone.
    """
    arrays=tuple(np.asarray(x,dtype=float) for x in
                 (source_start,source_end,target_start,target_end))
    if any(x.shape!=(4,3) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Q4/Q4 sweep needs four finite (4,3) endpoints')
    errors=(0.,)*4 if endpoint_errors is None else endpoint_errors
    if len(errors)!=4:raise ValueError('Q4/Q4 sweep needs four endpoint errors')
    a0,a1,b0,b1=arrays
    bounds=tuple(_enclose_positions(x,e) for x,e in zip(arrays,errors))
    if not q4_q4_swept_chart_admissible(*arrays,reference_source,reference_target,
                                      endpoint_errors=errors):
        return -float('inf')
    normals=tuple(np.stack([_normal_controls_interval(a0,a1,k,errors[0],errors[1])[s]
                           for k in range(4)]) for s in (0,1))
    # The rounded normal only selects an exact coordinate axis. No magnitude
    # or positivity claim relies on it; the directed controls decide those.
    selector=np.cross(a0[1]-a0[0],a0[3]-a0[0])
    axis=int(np.argmax(np.abs(selector)));sign=1. if selector[axis]>=0. else -1.
    component=_mul((normals[0][...,axis],normals[1][...,axis]),_interval(sign))
    denominator=float(np.min(component[0]))
    if not np.isfinite(denominator) or denominator<=0.:return -float('inf')
    d0=_sub(tuple(v[:,None,:] for v in bounds[0]),tuple(v[None,:,:] for v in bounds[2]))
    d1=_sub(tuple(v[:,None,:] for v in bounds[1]),tuple(v[None,:,:] for v in bounds[3]))
    def dot(d,k):
        return _dot(tuple(v[:,:,None,:] for v in d),
                    tuple(v[None,None,:,k,:] for v in normals))
    def third(v):return _outward(v[0]/3.,-np.inf),_outward(v[1]/3.,np.inf)
    controls=(dot(d0,0),third(_add(_mul(dot(d0,1),_interval(2.)),dot(d1,0))),
              third(_add(dot(d0,2),_mul(dot(d1,1),_interval(2.)))),dot(d1,2))
    numerator=min(float(np.min(v[0])) for v in controls)
    if not np.isfinite(numerator):return -float('inf')
    return float(_outward(min(0.,numerator)/denominator,-np.inf))
