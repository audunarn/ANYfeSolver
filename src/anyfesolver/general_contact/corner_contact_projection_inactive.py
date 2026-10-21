"""Exact inactive-branch certificate; never chooses a closest point.

For a bilinear patch p=c+a*xi+b*eta+m*xi*eta, the oriented scalar
(x-p).(p_xi cross p_eta) is bilinear in xi,eta. Its extrema on the
parameter square occur at the four corners. Interval rational arithmetic
at those corners encloses every admitted input-error box as well.
Distance is 1-Lipschitz in query/patch position; the convex combination of
vertex errors is bounded by their largest L1 norm. Consequently exact nominal
minimum bounds minus those errors certify strictly positive physical gap.
"""
from fractions import Fraction as F
import numpy as np


def _interval(value, error=0.):
    v,e=F(float(value)),F(float(error))
    return v-e,v+e


def _add(a,b): return a[0]+b[0],a[1]+b[1]
def _neg(a): return -a[1],-a[0]
def _sub(a,b): return _add(a,_neg(b))
def _mul(a,b):
    values=[x*y for x in a for y in b]
    return min(values),max(values)
def _scale(a,v): return _mul(a,(F(v),F(v)))
def _sum(items):
    result=(F(0),F(0))
    for item in items: result=_add(result,item)
    return result

def _cross(a,b):
    return [_sub(_mul(a[i],b[j]),_mul(a[j],b[i])) for i,j in ((1,2),(2,0),(0,1))]

def _side_lower(x,xe,c,ce):
    xx=[_interval(v,e) for v,e in zip(x,xe)]
    cc=[[_interval(v,e) for v,e in zip(row,err)] for row,err in zip(c,ce)]
    lower=None
    for local,(xi,eta) in enumerate(((-1,-1),(1,-1),(1,1),(-1,1))):
        dxi=[-(1-eta),1-eta,1+eta,-(1+eta)]
        deta=[-(1-xi),-(1+xi),1+xi,1-xi]
        a=[_scale(_sum(_scale(cc[k][i],dxi[k]) for k in range(4)),F(1,4)) for i in range(3)]
        b=[_scale(_sum(_scale(cc[k][i],deta[k]) for k in range(4)),F(1,4)) for i in range(3)]
        normal=_cross(a,b)
        side=_sum(_mul(_sub(xx[i],cc[local][i]),normal[i]) for i in range(3))
        lower=side[0] if lower is None else min(lower,side[0])
    return lower


def separated_zero_force_proof(x, query_error, positions, position_errors, slices,
                               faces, *, radius, zero_distance):
    """Return auditable g>zero_distance proof, or None; does not admit lambda.

    The caller MUST separately require an explicitly supplied current lambda=0.
    Only a successful certificate permits eliminating that inactive NCP branch.
    All near faces are checked, a conservative superset of possible minima.
    """
    from .corner_contact_geometry import _face_candidates, _face_minimum_bounds
    x=np.asarray(x,dtype=float); xe=np.asarray(query_error,dtype=float)
    positions=np.asarray(positions,dtype=float); errors=np.asarray(position_errors,dtype=float)
    if (positions.ndim!=2 or positions.shape[1:]!=(3,)
            or not np.all(np.isfinite(positions))
            or not np.isfinite(radius) or not np.isfinite(zero_distance)
            or not 0.<zero_distance<radius
            or x.shape!=(3,) or xe.shape!=(3,) or errors.shape!=positions.shape
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(xe))
            or not np.all(np.isfinite(errors)) or np.any(xe<0.) or np.any(errors<0.)):
        return None
    faces=tuple(int(j) for j in faces)
    if not faces or any(j<0 or j>=len(slices) for j in faces): return None
    qerr=sum(F(float(e)) for e in xe)
    vertex_errors=[sum(F(float(e)) for e in row) for row in errors]
    all_error=qerr+max(vertex_errors,default=F(0))
    records=[]; upper_candidates=[]
    radius_exact=F(float(radius)); zero_exact=F(float(zero_distance))
    # The float broadphase is only an enumeration. Recompute every omitted
    # face's exact nominal L-infinity box distance; rounded lo-radius/hi+radius
    # comparisons are not authority for claiming a radius separation.
    omitted=[]
    for face,(a,b) in enumerate(slices):
        if face in faces: continue
        corner=positions[a:b]
        if len(corner)!=4: return None
        lower=F(0)
        for axis in range(3):
            values=[F(float(v)) for v in corner[:,axis]]
            xx=F(float(x[axis]))
            lower=max(lower,min(values)-xx,xx-max(values))
        uncertainty=qerr+max(vertex_errors[a:b],default=F(0))
        physical_lower=max(F(0),lower-uncertainty)
        omitted.append(dict(face=face,nominal_box_Linf_lower_m=str(lower),
            position_error_L1_m=str(uncertainty),physical_distance_lower_m=str(physical_lower)))
    comparison_lower=min([radius_exact]+[F(item['physical_distance_lower_m']) for item in omitted])
    for face in faces:
        a,b=slices[face]; corners=positions[a:b]; cerr=errors[a:b]
        if len(corners)!=4: return None
        rows=[(d,face,xi,eta) for d,xi,eta in _face_candidates(x,corners)]
        bound=_face_minimum_bounds(x,corners,rows)
        if bound is None: return None
        upper,lower,_=bound
        face_error=qerr+max(vertex_errors[a:b],default=F(0))
        # sqrt(2*lower)-E > frozen zero-distance threshold, squared exactly.
        if 2*lower <= (zero_exact+face_error)**2: return None
        side_lower=_side_lower(x,xe,corners,cerr)
        if side_lower is None or side_lower<=0: return None
        records.append(dict(face=face,nominal_half_squared_lower=str(lower),
            nominal_half_squared_upper=str(upper),position_error_L1_m=str(face_error),
            oriented_side_lower=str(side_lower)))
        # A represented feasible point, inflated by its error boxes, must be
        # strictly closer than EVERY exact omitted-face lower and the radius.
        allowance=comparison_lower-face_error
        if allowance>0 and 2*upper<allowance**2: upper_candidates.append(face)
    if not upper_candidates: return None
    return dict(kind='exact_rational_separated_zero_force_branch',
        gap_strictly_greater_than_m=float(zero_distance),gap_positive=True,
        ncp_phi=0.0,lambda_N=0.0,contact_work=0.0,
        face_proofs=records,represented_upper_faces=upper_candidates,
        broadphase_radius_m=float(radius),all_position_error_L1_m=str(all_error),
        omitted_face_proofs=omitted,comparison_distance_lower_m=str(comparison_lower),
        no_closest_owner_or_gradient=True)
