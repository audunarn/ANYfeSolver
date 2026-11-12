"""Directed rational slab proof of positive P1 support and source ownership.

Witness points are fixed interior barycentric coordinates, not time samples.
Their enclosed projection stays strictly inside one fixed target triangle on
the whole slab. An open neighbourhood then supports all three P1 rows.
"""
from fractions import Fraction as F


def add(a,b):return a[0]+b[0],a[1]+b[1]
def neg(a):return -a[1],-a[0]
def sub(a,b):return add(a,neg(b))
def mul(a,b):
    p=[x*y for x in a for y in b];return min(p),max(p)
def dot(a,b):
    result=(F(0),F(0))
    for x,y in zip(a,b):result=add(result,mul(x,y))
    return result
def vsub(a,b):return [sub(x,y) for x,y in zip(a,b)]
def cross(a,b):return [sub(mul(a[1],b[2]),mul(a[2],b[1])),sub(mul(a[2],b[0]),mul(a[0],b[2])),sub(mul(a[0],b[1]),mul(a[1],b[0]))]
def exact(p):return [(F(float(x)),F(float(x))) for x in p]


def _represented_pair(source,ns,target,bary, *, support_required=False):
    nt=cross(vsub(target[1],target[0]),vsub(target[2],target[0]))
    denominator=dot(nt,ns)
    if denominator[0]<=0:return False
    # Strict opening excludes contact with this entire extended tool plane.
    heights=[dot(nt,vsub(x,target[0])) for x in source]
    if not support_required and min(v[0] for v in heights)>0:return True
    edges=[cross(nt,vsub(target[(k+1)%3],target[k])) for k in range(3)]
    corners=[]
    for inward,p in zip(edges,target):
        corners.append([sub(mul(dot(inward,vsub(x,p)),denominator),
                            mul(dot(inward,ns),h)) for x,h in zip(source,heights)])
    # Strict projected separation excludes all physical contact, without any
    # inference from a floating no-overlap result at the two endpoints.
    if not support_required and any(max(v[1] for v in row)<0 for row in corners):return True
    ns_lower=max(min(abs(v[0]),abs(v[1])) if v[0]*v[1]>0 else F(0) for v in ns)
    ns_upper=sum(max(abs(v[0]),abs(v[1])) for v in ns)
    # Orthogonal projection cannot increase a 3D distance. This encloses
    # the actual source-frame span used by both value and Jet2 clipping.
    span_squared=max(sum(max(abs(v[0]),abs(v[1]))**2 for v in vsub(p,source[0]))
                     for p in source+target)
    magnitude=max(abs(x) for row in corners for interval in row for x in interval)
    for weights in bary:
        margin=min(sum(w*interval[0] for w,interval in zip(weights,row)) for row in corners)
        if margin<=0 or magnitude<=0:continue
        radius=min(min(weights)/2,margin/(4*magnitude))
        # The enclosed witness triangle has area r^2*|ns|/2. Require
        # 100x the actual Jet2 positive-area threshold 1e-10*span^2;
        # this also exceeds value overlap's 1e-24*span^2 cutoff. This
        # certifies represented positive support, not smooth derivatives
        # across clipping ties (ordinary derivative admission still applies).
        if radius*radius*ns_lower/2>F(1,10**8)*span_squared:return True
    return False


def slab_support_and_ownership(manager,a,b,ea,eb):
    boxes=[]
    for p,q,e,f in zip(a,b,ea,eb):
        boxes.append([(min(F(float(x))-F(float(ex)),F(float(y))-F(float(ey))),
                       max(F(float(x))+F(float(ex)),F(float(y))+F(float(ey))))
                      for x,y,ex,ey in zip(p,q,e,f)])
    source_faces=manager.law.source_face_indices
    targets=[list(map(exact,manager.reference[1][ids])) for ids in manager.law.target_face_indices]
    normals=[cross(vsub(boxes[ids[1]],boxes[ids[0]]),vsub(boxes[ids[2]],boxes[ids[0]])) for ids in source_faces]
    # Every projected source interior is owned once. Shared endpoint identities
    # are exact because the finite skin map stores each physical node once.
    for i,ids in enumerate(source_faces):
        for j in range(i+1,len(source_faces)):
            other=source_faces[j]
            separated=False
            for k in range(3):
                u,v=int(ids[k]),int(ids[(k+1)%3])
                inward=cross(normals[i],vsub(boxes[v],boxes[u]))
                remaining=[int(n) for n in other if int(n) not in {u,v}]
                if remaining and all(dot(inward,vsub(boxes[n],boxes[u]))[1]<0 for n in remaining):
                    separated=True;break
            if not separated:return False
    bary=((F(1,3),)*3,(F(1,5),F(1,5),F(3,5)),
          (F(1,5),F(3,5),F(1,5)),(F(3,5),F(1,5),F(1,5)))
    for ids,ns in zip(source_faces,normals):
        witnessed=any(_represented_pair([boxes[int(n)] for n in ids],ns,target,bary,
                                       support_required=True) for target in targets)
        if not witnessed:return False
        if not all(_represented_pair([boxes[int(n)] for n in ids],ns,target,bary) for target in targets):
            return False
    return True


def _boxes(a,b,ea,eb):
    return [[(min(F(float(x))-F(float(ex)),F(float(y))-F(float(ey))),
              max(F(float(x))+F(float(ex)),F(float(y))+F(float(ey))))
             for x,y,ex,ey in zip(p,q,e,f)] for p,q,e,f in zip(a,b,ea,eb)]


def tool_normal_gap_lower(a,b,target,reference_source,reference_target, *, endpoint_errors):
    """Convert the fixed plane height to the law's source-normal ray gap.

    For x on a planar P1 source, g=nt.(x-t0)*|ns|/(nt.ns).
    Corner interval heights bound the entire triangle by affine convexity.
    Boxes must include endpoint rounding, Exp evaluation and path curvature.
    The old source-normal all-corner certificate is a safe fallback only.
    """
    import math
    ea,eb=endpoint_errors
    boxes=_boxes(a,b,ea,eb)
    t=list(map(exact,target))
    ns=cross(vsub(boxes[1],boxes[0]),vsub(boxes[2],boxes[0]))
    nt=cross(vsub(t[1],t[0]),vsub(t[2],t[0]))
    rs=list(map(exact,reference_source));rt=list(map(exact,reference_target))
    rn=cross(vsub(rs[1],rs[0]),vsub(rs[2],rs[0]))
    tn=cross(vsub(rt[1],rt[0]),vsub(rt[2],rt[0]))
    denominator=dot(nt,ns)[0]
    if denominator<=0 or dot(rn,ns)[0]<=0 or dot(tn,nt)[0]<=0:
        from .finite_rotation_sweep import finite_whole_pair_lower
        import numpy as np
        zero=np.zeros_like(target)
        return finite_whole_pair_lower(a,b,target,target,reference_source,reference_target,
            endpoint_errors=(ea,eb,zero,zero))
    # L1 is an exact upper enclosure of the Euclidean source-normal norm.
    upper=sum(max(abs(v[0]),abs(v[1])) for v in ns)
    height=min(dot(nt,vsub(x,t[0]))[0] for x in boxes)
    lower=min(F(0),height)*upper/denominator
    result=float(lower)
    if not math.isfinite(result):return -math.inf
    if F(result)>lower:result=math.nextafter(result,-math.inf)
    return result
