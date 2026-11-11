"""Matched S3 source-normal gap over conservative physical Hermite overlap.

Every actual target-barycentric numerator R_e(lambda,t) is bounded above by
U_e(lambda)=sum lambda_j max Bernstein controls(R_e,j). Exact rational clipping
against U_e>=0 contains the entire panel's overlap. H is affine in lambda;
vertex Bernstein bounds and the positive |A|/(A dot B) interval enclose gap.
This is matched source-normal separation, not closest Euclidean distance.
"""
from dataclasses import dataclass,replace
from fractions import Fraction
from math import inf
import numpy as np
from .s3_pressure_hermite_bounds import (
    _ROOT,_prepare_overlap,_PreparationCache,_factor,_blend,_multiply,_upper_float)
from .shell_pair_hermite_sweep import _limit,_split,_lower_float


@dataclass(frozen=True)
class S3OverlapHermiteGapBounds:
    success: bool
    status: str
    lower_gap_m: float|None
    visited_time_panels: int
    visited_spatial_cells: int
    detail: str=''


def _scalar(value,name,positive=False):
    raw=np.asarray(value)
    if raw.shape!=() or raw.dtype.kind not in 'fiu' or not np.isfinite(raw):
        raise ValueError(name+' must be finite and real')
    result=Fraction(float(value))
    if result<0 or (positive and result==0):
        raise ValueError(name+' has an invalid sign')
    return result


def _clip(vertices,coefficients):
    """Closed rational halfplane; point/segment polygons remain represented."""
    if not vertices:return ()
    result=[]
    previous=vertices[-1]
    pv=sum(a*b for a,b in zip(previous,coefficients))
    for current in vertices:
        cv=sum(a*b for a,b in zip(current,coefficients))
        if (pv>=0)!=(cv>=0):
            alpha=pv/(pv-cv)
            point=tuple(a+alpha*(b-a) for a,b in zip(previous,current))
            if not result or result[-1]!=point:result.append(point)
        if cv>=0 and (not result or result[-1]!=current):result.append(current)
        previous,pv=current,cv
    if len(result)>1 and result[-1]==result[0]:result.pop()
    return tuple(result)


def _outer_overlap(face):
    vertices=_ROOT
    R=face.polynomials[7:]
    for edge in range(3):
        vertices=_clip(vertices,tuple(max(p) for p in R[3*edge:3*edge+3]))
    return vertices


def _gap_interval(face,vertices,factor):
    if not vertices:return None
    intervals=[]
    for vertex in vertices:
        H=_blend(face.polynomials[4:7],vertex)
        intervals.append(_multiply((min(H),max(H)),factor))
    return min(v[0] for v in intervals),min(v[1] for v in intervals)


def _endpoints(face):
    return tuple(replace(face,polynomials=tuple((p[end],) for p in face.polynomials))
                 for end in (0,-1))


def s3_overlap_hermite_gap_bounds(reference_source,reference_target,source_faces,
        a0,a1,b0,b1,av0,av1,bv0,bv1,dt,*,gap_tolerance_m,
        maximum_time_panels=128,maximum_spatial_cells=1024,maximum_depth=14,
        exclude_open_endpoint_approach_tolerance_m=None,cancel=None,_preparation_cache=None):
    """Certify gap budget over every represented path overlap, failing closed.

Original polarity is checked before pruning, even for remote invalid faces.
Only an empty polygon is excluded; lower-dimensional closures remain checked.
Optional hidden-approach control covers original source material points with
gaps above its tolerance at both whole-path endpoints, even when endpoint
overlap is absent. Conservative openness halfplanes contain that set. This
guard is separate from the reference-P1 weak pressure-phase certificate.

No-overlap success has lower_gap_m=None/status='no_overlap'; failures also have
no usable gap. Counters bound all endpoint and panel overlap queries. No point
history, multiplier unknown, force, or reference measure is changed.
"""
    duration=_scalar(dt,'dt',True)
    tolerance=_scalar(gap_tolerance_m,'gap_tolerance_m')
    approach=(None if exclude_open_endpoint_approach_tolerance_m is None else
              _scalar(exclude_open_endpoint_approach_tolerance_m,'approach tolerance'))
    maximum_time_panels=_limit(maximum_time_panels,'maximum_time_panels',1)
    maximum_spatial_cells=_limit(maximum_spatial_cells,'maximum_spatial_cells',1)
    maximum_depth=_limit(maximum_depth,'maximum_depth',0)
    raw=np.asarray(reference_source)
    if raw.ndim!=2 or raw.shape[1]!=3 or raw.shape[0]<3:
        raise ValueError('reference_source requires original S3 coordinates')
    visited_time=visited_space=0
    completed=inf
    def result(success,status,detail=''):
        lower=(_lower_float(completed) if success and completed!=inf else None)
        return S3OverlapHermiteGapBounds(success,status,lower,visited_time,visited_space,detail)
    if cancel is not None and cancel():return result(False,'cancelled')
    if _preparation_cache is not None and type(_preparation_cache) is not _PreparationCache:
        raise ValueError('invalid S3 preparation cache')
    prepare=_prepare_overlap if _preparation_cache is None else _preparation_cache.get
    faces=prepare(reference_source,reference_target,source_faces,
        a0,a1,b0,b1,av0,av1,bv0,bv1,duration,len(raw))
    # Keep whole-path openness halfplanes when temporal panels are split.
    pending_faces=[]
    for face in faces:
        openness=[]
        for endpoint in _endpoints(face):
            factor,status=_factor(endpoint)
            if status is not None:return result(False,status)
            if approach is not None:
                # For g>tau>=0, H>0 and g<=H*factor_upper. Therefore actual
                # open source points lie in this conservative closed halfplane.
                openness.append(tuple(p[0]*factor[1]-approach
                                      for p in endpoint.polynomials[4:7]))
        pending_faces.append((face,tuple(openness)))
    pending=[(tuple(pending_faces),0)]
    while pending:
        if cancel is not None and cancel():return result(False,'cancelled')
        if visited_time>=maximum_time_panels:
            return result(False,'time_budget_exhausted','unvisited physical overlap panels remain')
        current,depth=pending.pop();visited_time+=1
        factors=[]
        for face,_ in current:
            factor,status=_factor(face)
            if status=='orientation_invalid':return result(False,status)
            if status is not None:break
            factors.append(factor)
        panel_lower=inf;reason='orientation_unresolved'
        admissible=len(factors)==len(current)
        if admissible:
            for (face,openness),factor in zip(current,factors):
                if cancel is not None and cancel():return result(False,'cancelled')
                # Constant endpoints have exact projected clipping. A certified
                # real endpoint penetration cannot be repaired by subdividing.
                for endpoint in _endpoints(face):
                    if visited_space>=maximum_spatial_cells:return result(False,'spatial_budget_exhausted')
                    visited_space+=1
                    endpoint_factor,status=_factor(endpoint)
                    if status is not None:return result(False,status)
                    value=_gap_interval(endpoint,_outer_overlap(endpoint),endpoint_factor)
                    if value is not None and value[1]<-tolerance:
                        return result(False,'gap_budget_violation','matched endpoint penetration exceeds budget')
                if visited_space>=maximum_spatial_cells:return result(False,'spatial_budget_exhausted')
                visited_space+=1
                vertices=_outer_overlap(face)
                value=_gap_interval(face,vertices,factor)
                if value is not None:
                    panel_lower=min(panel_lower,value[0])
                    if value[0]<-tolerance:
                        admissible=False;reason='gap_unresolved'
                if approach is not None:
                    for halfplane in openness:vertices=_clip(vertices,halfplane)
                    hidden=_gap_interval(face,vertices,factor)
                    if hidden is not None and hidden[0]<-approach:
                        admissible=False;reason='hidden_approach_unresolved'
            if admissible:
                completed=min(completed,panel_lower)
                continue
        if depth>=maximum_depth:
            return result(False,reason,'complete matched path not certified within subdivision depth')
        left=[];right=[]
        for face,openness in current:
            pairs=tuple(_split(p) for p in face.polynomials)
            left.append((replace(face,polynomials=tuple(p[0] for p in pairs)),openness))
            right.append((replace(face,polynomials=tuple(p[1] for p in pairs)),openness))
        pending.append((tuple(right),depth+1));pending.append((tuple(left),depth+1))
    return result(True,'no_overlap' if completed==inf else 'resolved')
