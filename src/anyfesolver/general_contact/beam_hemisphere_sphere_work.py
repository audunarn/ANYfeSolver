"""Actual outward B2/B3 hemisphere support against a spherical obstacle.

Output work coordinates are all beam translations followed by sphere centre.
The caller owns penalty assembly, curve regularity and exterior/swept admission.
No spin/material-surface history is represented.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral

import numpy as np
from anyfesolver._ge_beam3_mixed_ad import Jet2, dot, sqrt
from .beam_hemisphere_support import (_readonly, _vector, _cross, _exact_dot,
                                      _replace_value, HemispherePlaneGeometryError)
from .capped_cylinder import _sqrt_bound


@dataclass(frozen=True)
class HemisphereSphereState:
    gap: float
    gradient: np.ndarray
    hessian: np.ndarray
    skin_point: np.ndarray | None
    rigid_point: np.ndarray | None
    normal: np.ndarray | None
    branch: str


def hemisphere_sphere_work(positions, endpoint, outward_sign, radius,
                           sphere_center, sphere_radius):
    """Constrained cap distance, with complete node/centre derivatives.

    Let q=c-O, d be the actual outward endpoint unit tangent, and a=q.d.
    For a<=0 the sphere minimum gives |q|-R-Ro. For a>0 the minimum lies
    on the cap rim, giving sqrt(a²+(|q-a*d|-R)²)-Ro. These meet C1; the
    sphere Hessian is selected at the seam. A parallel rim is nonunique,
    but strictly open in the exterior domain and has zero penalty work.
    """
    x=np.asarray(positions,dtype=float)
    if x.shape not in ((2,3),(3,3)) or not np.all(np.isfinite(x)):
        raise HemispherePlaneGeometryError('sphere support requires finite B2/B3 positions')
    if (isinstance(endpoint,(bool,np.bool_)) or not isinstance(endpoint,Integral)
            or endpoint not in (0,len(x)-1)):
        raise HemispherePlaneGeometryError('invalid hemisphere endpoint')
    if isinstance(outward_sign,(bool,np.bool_)) or outward_sign not in (-1,1):
        raise HemispherePlaneGeometryError('invalid outward tangent sign')
    R,Ro=float(radius),float(sphere_radius)
    if not all(math.isfinite(r) and r>0 for r in (R,Ro)):
        raise HemispherePlaneGeometryError('hemisphere and sphere radii must be finite positive')
    O=_vector(sphere_center,'sphere_center')
    f=lambda value:Fraction.from_float(float(value))
    xf=[[f(a) for a in row] for row in x];of=[f(a) for a in O]
    coeff=((-1,1) if len(x)==2 else ((-3,4,-1) if endpoint==0 else (1,-4,3)))
    coeff=tuple(outward_sign*a for a in coeff)
    q=[a-b for a,b in zip(xf[endpoint],of)]
    v=[sum((Fraction(a)*row[j] for a,row in zip(coeff,xf)),Fraction(0)) for j in range(3)]
    Q,V,b=_exact_dot(q,q),_exact_dot(v,v),_exact_dot(q,v)
    rf,rof=f(R),f(Ro)
    if V==0 or Q<=max(rf,rof)**2:
        raise HemispherePlaneGeometryError('sphere support requires regular tangent and exterior endpoint')
    P=(Q*V-b*b)/V
    branch='sphere' if b<=0 else ('parallel_rim' if P==0 else 'rim')
    size=3*(len(x)+1)
    if branch=='parallel_rim':
        gap=float(_sqrt_bound(Q+rf*rf)-rof)
        if not math.isfinite(gap) or gap<=0:
            raise HemispherePlaneGeometryError('nonunique parallel-rim support must be open')
        return HemisphereSphereState(gap,_readonly(np.zeros(size)),
            _readonly(np.zeros((size,size))),None,None,None,branch)
    qs,vs=max(map(abs,q)),max(map(abs,v))
    q0=[a/qs for a in q];v0=[a/vs for a in v]
    inc=[[Jet2.variable(0.,3*i+j,size) for j in range(3)] for i in range(len(x)+1)]
    qj=[float(q0[j])+(inc[endpoint][j]-inc[-1][j])/float(qs) for j in range(3)]
    vj=[float(v0[j])+sum((a*inc[i][j]/float(vs) for i,a in enumerate(coeff)),
        Jet2.constant(0.,size)) for j in range(3)]
    if branch=='sphere':
        distance=sqrt(_replace_value(dot(qj,qj),_exact_dot(q0,q0)))*float(qs)
        gj=distance-R-Ro
        gap=float((Q-(rf+rof)**2)/(_sqrt_bound(Q)+rf+rof))
        direction=np.array([float(a/_sqrt_bound(Q)) for a in q])
    else:
        cross0=_cross(q0,v0)
        crossj=[_replace_value(a,bv) for a,bv in zip(_cross(qj,vj),cross0)]
        vn=sqrt(_replace_value(dot(vj,vj),_exact_dot(v0,v0)))
        rho=float(qs)*sqrt(_replace_value(dot(crossj,crossj),_exact_dot(cross0,cross0)))/vn
        axial=float(qs)*_replace_value(dot(qj,vj),_exact_dot(q0,v0))/vn
        sqrtP=_sqrt_bound(P)
        rho_minus_R=(P-rf*rf)/(sqrtP+rf)
        radial=_replace_value(rho-R,rho_minus_R)
        L2=b*b/V+rho_minus_R*rho_minus_R
        distance=sqrt(_replace_value(axial*axial+radial*radial,L2))
        gj=distance-Ro
        # Two conjugates preserve the exact activation sign without a gap tolerance.
        A=Q+rf*rf-rof*rof
        numerator=A*A-4*rf*rf*P
        denominator=(A+2*rf*sqrtP)*(_sqrt_bound(L2)+rof)
        gap=float(numerator/denominator)
        direction0=_cross(v,_cross(q,v))
        denominator0=_sqrt_bound(V)*_sqrt_bound(Q*V-b*b)
        direction=np.array([float(a/denominator0) for a in direction0])
    exact_sign=(Q-(rf+rof)**2) if branch=='sphere' else numerator
    if (not math.isfinite(gap) or (exact_sign!=0 and gap==0.)
            or not np.all(np.isfinite(gj.gradient)) or not np.all(np.isfinite(gj.hessian))):
        raise HemispherePlaneGeometryError('sphere support gap/work is not representable')
    if abs(gj.value-gap)>128*np.finfo(float).eps*max(R,Ro,float(_sqrt_bound(Q))):
        raise HemispherePlaneGeometryError('sphere support AD and exact gap disagree')
    relative=np.array([float(a) for a in q])-R*direction
    normal=direction if branch=='sphere' else relative/float(distance.value)
    skin=x[endpoint]-R*direction;rigid=O+Ro*normal
    if not all(np.all(np.isfinite(a)) for a in (skin,rigid,normal)):
        raise HemispherePlaneGeometryError('sphere support points are not finite')
    return HemisphereSphereState(gap,_readonly(gj.gradient),_readonly(gj.hessian),
        _readonly(skin),_readonly(rigid),_readonly(normal),branch)
