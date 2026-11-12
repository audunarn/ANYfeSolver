"""Smooth original bilinear Q4/Q4 mortar integrals in 24 physical coordinates.

Clipping triangles are integration cells only. Both projected bilinear maps,
source varying normal and reference/current projected measure differentiate.
Adaptive differences are measured resolution indicators, not error certificates.
"""
from dataclasses import dataclass
import numpy as np
from anyfesolver._ge_beam3_mixed_ad import Jet2,sqrt
from .moving import _TRIANGLE_RULE
from .overlap import PlanarFace
from .q4_s3_overlap_derivatives import UnsupportedQ4S3Branch,_clip,_area_resolution,_parts


@dataclass(frozen=True)
class Q4Q4OverlapDerivatives:
    reference_support: np.ndarray
    reference_weighted_gap: np.ndarray
    support_gradient: np.ndarray
    support_hessian: np.ndarray
    weighted_gap_gradient: np.ndarray
    weighted_gap_hessian: np.ndarray
    overlap_area: float
    overlap_area_resolution: float
    reference_overlap_area: float
    branch_signature: tuple
    integration_triangles: int
    derivative_quadrature_differences: np.ndarray
    measured_errors: tuple


def _zero():return Jet2.constant(0.,24)
def _sum(items):return sum(items,start=_zero())
def _sub(a,b):return [x-y for x,y in zip(a,b)]
def _dot(a,b):return _sum(x*y for x,y in zip(a,b))
def _cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def _cross2(a,b):return a[0]*b[1]-a[1]*b[0]
def _normalize(a):
    size=sqrt(_dot(a,a));return [x/size for x in a]
def _q1(r,s):
    return ([(1-r)*(1-s)*.25,(1+r)*(1-s)*.25,(1+r)*(1+s)*.25,(1-r)*(1+s)*.25],
            [-(1-s)*.25,(1-s)*.25,(1+s)*.25,-(1+s)*.25],
            [-(1-r)*.25,-(1+r)*.25,(1+r)*.25,(1-r)*.25])
def _map(shape,corners):return [_sum(shape[i]*corners[i][k] for i in range(4)) for k in range(len(corners[0]))]
def _natural(point,corners):
    # The final Jet residual closes the implicit bilinear equations through
    # second order; a fixed iteration count alone never establishes admission.
    a=np.array([corners[1][k].value-corners[0][k].value for k in range(2)])
    b=np.array([corners[3][k].value-corners[0][k].value for k in range(2)])
    rhs=np.array([point[k].value-corners[0][k].value for k in range(2)])
    u,v=np.linalg.solve(np.column_stack((a,b)),rhs)
    r,s=Jet2.constant(2*u-1,24),Jet2.constant(2*v-1,24)
    for _ in range(4):
        shape,dr,ds=_q1(r,s);first,second=_map(dr,corners),_map(ds,corners)
        determinant=_cross2(first,second)
        if determinant.value<=0:raise UnsupportedQ4S3Branch('Q4 pair inverse Jacobian is unresolved')
        residual=_sub(_map(shape,corners),point)
        r,s=r-_cross2(residual,second)/determinant,s-_cross2(first,residual)/determinant
        shape,dr,ds=_q1(r,s);residual=_sub(_map(shape,corners),point)
        if all(abs(x.value)<=1e-12 and np.max(np.abs(x.gradient))<=1e-12 and np.max(np.abs(x.hessian))<=1e-12 for x in residual):
            break
    if any(abs(x.value)>1e-11 or np.max(np.abs(x.gradient))>1e-9 or np.max(np.abs(x.hessian))>1e-7 for x in residual):
        raise UnsupportedQ4S3Branch('Q4 pair bilinear implicit derivatives did not converge')
    if max(abs(r.value),abs(s.value))>1.+1e-9:
        raise UnsupportedQ4S3Branch('Q4 pair projected inverse leaves material domain')
    return shape,dr,ds



def _batched_rule(vertices, source_xy, target_xy, source, target, reference,
                  normal, threshold):
    """Existing Jet2 batch algebra over the seven unchanged triangle samples.

    Per-point inverse convergence is frozen independently; accumulation order
    remains point then material node. No process cache or geometric shortcut.
    """
    from .q4_s3_overlap_derivatives import _BJet
    size=24;count=len(_TRIANGLE_RULE)
    def zero():return _BJet.const(np.zeros(count),size)
    def summation(items):return sum(items,start=zero())
    def mapping(shape,corners):return [summation(shape[i]*corners[i][k] for i in range(4)) for k in range(len(corners[0]))]
    def dot(a,b):return summation(x*y for x,y in zip(a,b))
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    def cross2(a,b):return a[0]*b[1]-a[1]*b[0]
    def sub(a,b):return [x-y for x,y in zip(a,b)]
    def batch(corners):return [[_BJet.of(jet) for jet in point] for point in corners]
    def natural(point,corners):
        a=np.array([corners[1][k].v[0]-corners[0][k].v[0] for k in range(2)])
        b=np.array([corners[3][k].v[0]-corners[0][k].v[0] for k in range(2)])
        matrix=np.column_stack((a,b))
        # Same two-by-two solve per sample as the scalar baseline.
        uv=np.array([np.linalg.solve(matrix,np.array([point[k].v[j]-corners[0][k].v[0] for k in range(2)])) for j in range(count)])
        r,s=(_BJet.const(2*uv[:,k]-1,size) for k in range(2))
        active=np.ones(count,dtype=bool)
        for _ in range(4):
            shape,dr,ds=_q1(r,s);first,second=mapping(dr,corners),mapping(ds,corners)
            determinant=cross2(first,second)
            if np.any(determinant.v<=0):raise UnsupportedQ4S3Branch('Q4 pair inverse Jacobian is unresolved')
            residual=sub(mapping(shape,corners),point)
            nr,ns=r-cross2(residual,second)/determinant,s-cross2(first,residual)/determinant
            r,s=nr.select(active,r),ns.select(active,s)
            shape,dr,ds=_q1(r,s);residual=sub(mapping(shape,corners),point)
            converged=np.ones(count,dtype=bool)
            for x in residual:
                converged &= (np.abs(x.v)<=1e-12)&(np.max(np.abs(x.g),axis=1)<=1e-12)&(np.max(np.abs(x.h),axis=(1,2))<=1e-12)
            active &= ~converged
            if not np.any(active):break
        if any(np.any(np.abs(x.v)>1e-11) or np.any(np.abs(x.g)>1e-9) or np.any(np.abs(x.h)>1e-7) for x in residual):
            raise UnsupportedQ4S3Branch('Q4 pair bilinear implicit derivatives did not converge')
        if max(np.max(np.abs(r.v)),np.max(np.abs(s.v)))>1.+1e-9:
            raise UnsupportedQ4S3Branch('Q4 pair projected inverse leaves material domain')
        return shape,dr,ds
    area=_cross2(_sub(vertices[1],vertices[0]),_sub(vertices[2],vertices[0]))*.5
    if area.value<=0:raise UnsupportedQ4S3Branch('Q4 pair integration cell is degenerate')
    bv=batch(vertices)
    bary=np.array([item[0] for item in _TRIANGLE_RULE]);weights=np.array([item[1] for item in _TRIANGLE_RULE])
    point=[summation(bv[j][k]*bary[:,j] for j in range(3)) for k in range(2)]
    q,t,ref=batch(source),batch(target),batch(reference)
    shape,dr,ds=natural(point,batch(source_xy));target_shape,_,_=natural(point,batch(target_xy))
    jr,js=mapping(dr,q),mapping(ds,q);raw=cross(jr,js)
    jacobian=dot(raw,[_BJet.of(x) for x in normal])
    if np.any(jacobian.v<=threshold*.25):raise UnsupportedQ4S3Branch('Q4 pair source projected Jacobian is unresolved')
    length=dot(raw,raw).sqrt();physical_normal=[x/length for x in raw]
    gap=dot(physical_normal,sub(mapping(shape,q),mapping(target_shape,t)))
    ref_raw=cross(mapping(dr,ref),mapping(ds,ref))
    reference_jacobian=dot(ref_raw,ref_raw).sqrt()
    measure=_BJet.of(area)*weights*reference_jacobian/jacobian
    support,weighted=[_zero() for _ in range(4)],[_zero() for _ in range(4)]
    contributions=[measure*x for x in shape];gaps=[x*gap for x in contributions]
    for j in range(count):
        for i in range(4):
            x,y=contributions[i],gaps[i]
            support[i]+=Jet2(x.v[j],x.g[j].copy(),x.h[j].copy())
            weighted[i]+=Jet2(y.v[j],y.g[j].copy(),y.h[j].copy())
    return support,weighted


def q4_q4_overlap_derivatives(reference_source,current_source,current_target,*,
        area_tolerance,weighted_gap_tolerance,maximum_triangles,
        support_gradient_tolerance,weighted_gap_gradient_tolerance,
        support_hessian_tolerance,weighted_gap_hessian_tolerance,
        relative_branch_tolerance=1e-10):
    controls=np.array((area_tolerance,weighted_gap_tolerance,support_gradient_tolerance,
        weighted_gap_gradient_tolerance,support_hessian_tolerance,weighted_gap_hessian_tolerance),dtype=float)
    if (not np.isfinite(controls).all() or min(controls)<=0 or type(maximum_triangles) is not int or maximum_triangles<1
            or not np.isfinite(relative_branch_tolerance) or relative_branch_tolerance<=0):
        raise ValueError('Q4 pair requires six positive dimensional controls and triangle budget')
    reference,source,target=(np.asarray(x,dtype=float) for x in (reference_source,current_source,current_target))
    if any(x.shape!=(4,3) or not np.isfinite(x).all() for x in (reference,source,target)):
        raise ValueError('Q4 pair requires original four reference/source/target corners')
    span=max(np.linalg.norm(reference[i]-reference[0]) for i in (1,2,3))
    refnormal=np.cross(reference[1]-reference[0],reference[3]-reference[0])
    if (span<=0 or np.linalg.norm(refnormal)<=1e-12*span*span
            or abs(float((reference[2]-reference[0])@refnormal))/np.linalg.norm(refnormal)>relative_branch_tolerance*span):
        raise UnsupportedQ4S3Branch('Q4 pair first profile requires planar source reference')
    PlanarFace((1,1),reference)  # retain the original planar-reference contract
    values=np.concatenate((source,target)).reshape(-1)
    points=[[Jet2.variable(values[3*i+k],3*i+k,24) for k in range(3)] for i in range(8)]
    q,t=points[:4],points[4:];edge=_sub(q[1],q[0]);raw=_cross(edge,_sub(q[3],q[0]))
    if np.dot(refnormal,[x.value for x in raw])<=0:raise UnsupportedQ4S3Branch('Q4 pair source reference polarity reversed')
    normal,tangent=_normalize(raw),_normalize(edge);binormal=_cross(normal,tangent)
    def projected(x):
        delta=_sub(x,q[0]);return [_dot(delta,tangent),_dot(delta,binormal)]
    source_xy,target_xy=[projected(x) for x in q],[projected(x) for x in t]
    scale=max(np.hypot(x[0].value,x[1].value) for x in (*source_xy,*target_xy))
    threshold=relative_branch_tolerance*scale*scale
    for corners in (source_xy,target_xy):
        for i in range(4):
            if _cross2(_sub(corners[(i+1)%4],corners[i]),_sub(corners[(i+2)%4],corners[(i+1)%4])).value<=threshold:
                raise UnsupportedQ4S3Branch('Q4 pair projected chart is folded or unresolved')
    raw_scale=max(scale,np.max(np.abs(source)),np.max(np.abs(target)))
    polygon,errors,signature=_clip(source_xy,target_xy,64*np.finfo(float).eps*raw_scale,edge_count=4)
    if len(polygon)<3:raise UnsupportedQ4S3Branch('Q4 pair has no resolved overlap')
    area=_sum(_cross2(polygon[j],polygon[(j+1)%len(polygon)]) for j in range(len(polygon)))*.5
    resolution=_area_resolution(polygon,errors)
    if area.value<=resolution:raise UnsupportedQ4S3Branch('Q4 pair overlap is unresolved')
    ref=[[Jet2.constant(float(value),24) for value in point] for point in reference]
    def rule(vertices):
        return _batched_rule(vertices, source_xy, target_xy, q, t, ref, normal, threshold)
    def children(vertices):
        a,b,c=vertices;ab,bc,ca=([.5*(x+y) for x,y in zip(p,q)] for p,q in ((a,b),(b,c),(c,a)))
        return ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))
    support,weighted=[_zero() for _ in range(4)],[_zero() for _ in range(4)]
    pending=[(tuple(polygon[j] for j in (0,i,i+1)),1./(len(polygon)-2)) for i in range(1,len(polygon)-1)]
    measured=np.zeros(6);evaluations=0
    while pending:
        vertices,fraction=pending.pop()
        if evaluations+5>maximum_triangles:raise RuntimeError('Q4 pair derivative triangle budget exhausted')
        coarse=rule(vertices);fine=[[_zero() for _ in range(4)],[_zero() for _ in range(4)]]
        cells=children(vertices)
        for cell in cells:
            candidate=rule(cell)
            for field in range(2):
                for i in range(4):fine[field][i]+=candidate[field][i]
        evaluations+=5;lo,hi=[_parts(x) for x in coarse],[_parts(x) for x in fine]
        error=np.array([np.max(np.abs(hi[field][order]-lo[field][order])) for order in range(3) for field in range(2)])
        if not np.isfinite(error).all():raise UnsupportedQ4S3Branch('nonfinite Q4 pair quadrature difference')
        if np.any(error>fraction*controls):
            pending.extend((cell,fraction*.25) for cell in cells);continue
        measured+=error
        for i in range(4):support[i]+=fine[0][i];weighted[i]+=fine[1][i]
    sv,sg,sh=_parts(support);wv,wg,wh=_parts(weighted)
    if min(sv)<=0 or not all(np.isfinite(x).all() for x in (sv,sg,sh,wv,wg,wh)):
        raise UnsupportedQ4S3Branch('Q4 pair has unsupported pressure modes or nonfinite derivatives')
    return Q4Q4OverlapDerivatives(sv,wv,sg,sh,wg,wh,area.value,resolution,float(sum(sv)),signature,evaluations,
        np.array((measured[2],measured[4],measured[3],measured[5])),tuple(measured))


def q4_q4_swept_chart_admissible(source_start,source_end,target_start,target_end,
        reference_source,reference_target,*,endpoint_errors=None):
    """Whole affine nodal path: all8 moving Q1 corner chart predicates.

    Spatial Q1 raw normals are convex combinations of corner normals. Their
    mutual dot with sourcecorner0 is quartic in time; strict Bernstein control
    positivity is sufficient. Endpoint errors enclose the original fixed map.
    """
    from .q4_overlap_local_bound import _normal_controls_interval,_quadratic_dot_positive,_interval,_dot
    arrays=tuple(np.asarray(x,dtype=float) for x in (source_start,source_end,target_start,target_end))
    if any(x.shape!=(4,3) or not np.isfinite(x).all() for x in arrays):return False
    errors=(0.,)*4 if endpoint_errors is None else endpoint_errors
    a0,a1,b0,b1=arrays
    source=[_normal_controls_interval(a0,a1,i,errors[0],errors[1]) for i in range(4)]
    target=[_normal_controls_interval(b0,b1,i,errors[2],errors[3]) for i in range(4)]
    for controls,reference in ((source,reference_source),(target,reference_target)):
        reference=np.asarray(reference,dtype=float)
        if reference.shape != (4,3) or not np.isfinite(reference).all():return False
        # The binary reference corners define the axis. Enclose their exact
        # subtraction/cross rather than treating the rounded cross as exact.
        reference_controls=_normal_controls_interval(reference,reference,0)
        axis=tuple(value[0] for value in reference_controls)
        for normal in controls:
            low=_dot(normal,axis)[0]
            if not np.isfinite(low).all() or np.min(low)<=0:return False
    return all(_quadratic_dot_positive(source[0],normal) for normal in (*source,*target))
