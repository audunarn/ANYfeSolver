"""Actual B2/B3 hemisphere minimum against a fixed infinite cylinder.

The rim minimum moves with every authoritative endpoint-tangent coordinate.
A quartic proposes all stationary points; rational Sturm counting checks that
none were lost. Jet2 envelope differentiation retains the implicit station term.
"""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
import math
import numpy as np
from anyfesolver._ge_beam3_mixed_ad import Jet2, dot, sqrt, sin, cos
from .beam_hemisphere_support import _readonly, _vector, _cross, HemispherePlaneGeometryError


@dataclass(frozen=True)
class HemisphereCylinderState:
    gap: float
    gradient: np.ndarray
    hessian: np.ndarray
    skin_point: np.ndarray | None
    rigid_point: np.ndarray | None
    normal: np.ndarray | None
    branch: str


def _trim(p):
    while len(p)>1 and p[-1]==0:p.pop()
    return p


def _sturm(coefficients):
    p=_trim([(x if isinstance(x,Fraction) else Fraction.from_float(float(x))) for x in coefficients])
    if len(p)==1:return [p]
    sequence=[p,[i*p[i] for i in range(1,len(p))]]
    while len(sequence[-1])>1:
        r=list(sequence[-2]); d=sequence[-1]
        while len(r)>=len(d) and any(r):
            n=len(r)-len(d); ratio=r[-1]/d[-1]
            for i,v in enumerate(d):r[n+i]-=ratio*v
            _trim(r)
        if not any(r):break
        sequence.append(_trim([-v for v in r]))
    return sequence


def _variation(sequence,at):
    signs=[]
    for p in sequence:
        if at is None:value=p[-1]
        elif at is False:value=p[-1]*(-1 if (len(p)-1)%2 else 1)
        else:
            value=Fraction(0)
            for v in reversed(p):value=value*at+v
        if value:signs.append(1 if value>0 else -1)
    return sum(a!=b for a,b in zip(signs,signs[1:]))


def _root_brackets(coefficients,proposals):
    """Each proposed root owns an exact one-root rational interval."""
    sequence=_sturm(coefficients); polynomial=sequence[0]
    count=_variation(sequence,False)-_variation(sequence,None)
    if len(proposals)!=count:raise HemispherePlaneGeometryError('cylinder rim stationary roots unresolved')
    bound=1+max((abs(v/polynomial[-1]) for v in polynomial[:-1]),default=Fraction(0))
    roots=sorted(Fraction.from_float(t) for t in proposals)
    cuts=[-bound]+[(a+b)/2 for a,b in zip(roots,roots[1:])]+[bound]
    intervals=[]
    for root,left,right in zip(roots,cuts,cuts[1:]):
        if not left<root<right or _variation(sequence,left)-_variation(sequence,right)!=1:
            raise HemispherePlaneGeometryError('cylinder rim root separation unresolved')
        intervals.append((float(root),left,right))
    return intervals


def _rim_station(q,e,f,R):
    # Preserve the stationary polynomial of these literal projected vectors.
    # Rounded coefficients must not erase a close competing real root.
    def exact_dot(a,b):
        return sum(Fraction.from_float(float(x))*Fraction.from_float(float(y)) for x,y in zip(a,b))
    A=exact_dot(q,e); B=exact_dot(q,f)
    D=exact_dot(e,e); C=exact_dot(f,f); E=exact_dot(e,f)
    r=Fraction.from_float(R)
    coefficients=[B+r*E,-2*A+2*r*(C-D),-6*r*E,-2*A-2*r*(C-D),-B+r*E]
    pi_station=coefficients[-1]==0
    while len(coefficients)>1 and coefficients[-1]==0:coefficients.pop()
    scale_coeff=max(map(abs,coefficients))
    if scale_coeff==0:raise HemispherePlaneGeometryError('cylinder rim stationary polynomial unresolved')
    approximate=np.array([float(v/scale_coeff) for v in coefficients])
    if any(v and a==0 for v,a in zip(coefficients,approximate)):
        raise HemispherePlaneGeometryError('cylinder rim stationary coefficients unrepresentable')
    roots=np.polynomial.polynomial.polyroots(approximate)
    real=[]
    for value in roots:
        if abs(value.imag)<=64*np.finfo(float).eps*max(1.,abs(value.real)):
            t=float(value.real)
            if not any(abs(t-a)<=256*np.finfo(float).eps*max(1.,abs(t),abs(a)) for a in real):real.append(t)
    intervals=_root_brackets(coefficients,real)
    angles=[(2*math.atan(t),left,right) for t,left,right in intervals]
    # pi is a station only when the quartic leading coefficient vanishes.
    if pi_station:angles.append((math.pi,None,None))
    stations=[]
    scale=max(float(q@q),R*R)
    for theta,left,right in angles:
        for _ in range(8):
            w=q+R*(e*math.cos(theta)+f*math.sin(theta))
            v=R*(-e*math.sin(theta)+f*math.cos(theta))
            acceleration=-R*(e*math.cos(theta)+f*math.sin(theta))
            first=2*float(w@v); second=2*float(v@v+w@acceleration)
            if abs(first)<=64*np.finfo(float).eps*scale:break
            if abs(second)<=64*np.finfo(float).eps*scale:break
            change=first/second
            if not math.isfinite(change) or abs(change)>.5:break
            theta-=change
        w=q+R*(e*math.cos(theta)+f*math.sin(theta))
        value=float(w@w)
        derivative=2*float(w@(R*(-e*math.sin(theta)+f*math.cos(theta))))
        if abs(derivative)>256*np.finfo(float).eps*scale:
            raise HemispherePlaneGeometryError('cylinder rim stationary representative unresolved')
        if left is not None:
            refined=Fraction.from_float(math.tan(theta/2))
            if not left<refined<right:
                raise HemispherePlaneGeometryError('cylinder rim refinement lost root correspondence')
        stations.append((value,theta,w))
    stations.sort(key=lambda s:s[0])
    best=stations[0]
    tied=any(abs(s[0]-best[0])<=256*np.finfo(float).eps*scale for s in stations[1:])
    return best,tied


def hemisphere_cylinder_work(positions, endpoint, outward_sign, radius,
                             cylinder_point, cylinder_axis, cylinder_radius):
    x=np.asarray(positions,dtype=float)
    if x.shape not in ((2,3),(3,3)) or not np.all(np.isfinite(x)):
        raise HemispherePlaneGeometryError('cylinder support requires finite B2/B3 positions')
    if (isinstance(endpoint,(bool,np.bool_)) or not isinstance(endpoint,Integral)
            or endpoint not in (0,len(x)-1)):
        raise HemispherePlaneGeometryError('invalid hemisphere endpoint')
    if isinstance(outward_sign,(bool,np.bool_)) or outward_sign not in (-1,1):
        raise HemispherePlaneGeometryError('invalid outward tangent sign')
    R,Ro=float(radius),float(cylinder_radius)
    if not all(math.isfinite(v) and v>0 for v in (R,Ro)):
        raise HemispherePlaneGeometryError('hemisphere/cylinder radii must be finite positive')
    O=_vector(cylinder_point,'cylinder_point'); axis=_vector(cylinder_axis,'cylinder_axis')
    magnitude=math.hypot(*axis)
    if not math.isfinite(magnitude) or magnitude==0:
        raise HemispherePlaneGeometryError('cylinder axis must be finite nonzero')
    literal_axis=tuple(Fraction.from_float(float(v)) for v in axis)
    aa=sum(v*v for v in literal_axis)
    exactP=tuple(tuple(Fraction(int(i==j))-literal_axis[i]*literal_axis[j]/aa for j in range(3)) for i in range(3))
    P=np.array([[float(v) for v in row] for row in exactP])
    coeff=((-1,1) if len(x)==2 else ((-3,4,-1) if endpoint==0 else (1,-4,3)))
    coeff=tuple(outward_sign*v for v in coeff)
    xf=[[Fraction.from_float(float(v)) for v in row] for row in x]
    # The translating owner supplies exact affine centres. _vector above
    # validates representability but must not round away their local offset.
    of=[v if isinstance(v,Fraction) else Fraction.from_float(float(v)) for v in cylinder_point]
    qf=[v-c for v,c in zip(xf[endpoint],of)]
    tf=[sum(c*(row[j]-xf[endpoint][j]) for c,row in zip(coeff,xf)) for j in range(3)]
    radialf=[sum(v*a for v,a in zip(qf,row)) for row in exactP]
    rho2=sum(v*v for v in radialf)
    if not any(tf) or rho2<=max(Fraction.from_float(R),Fraction.from_float(Ro))**2:
        raise HemispherePlaneGeometryError('cylinder support requires regular tangent and exterior endpoint')
    tangent=np.array([float(v) for v in tf]); tangent_scale=max(abs(tangent))
    tangent=tangent/tangent_scale
    tangent_norm=math.hypot(*tangent)
    radial=np.array([float(v) for v in radialf]); rho=math.hypot(*radial)
    if not math.isfinite(tangent_norm) or not math.isfinite(rho):
        raise HemispherePlaneGeometryError('cylinder support geometry is not representable')
    d=tangent/tangent_norm
    size=3*(len(x)+1); N=size+1
    inc=[[Jet2.variable(0.,3*i+j,N) for j in range(3)] for i in range(len(x)+1)]
    qj=[radial[j]+sum((P[j,k]*(inc[endpoint][k]-inc[-1][k]) for k in range(3)),Jet2.constant(0.,N)) for j in range(3)]
    tj=[tangent[j]+sum((c*inc[i][j]/tangent_scale for i,c in enumerate(coeff)),Jet2.constant(0.,N)) for j in range(3)]
    norm=sqrt(dot(tj,tj)); dj=[v/norm for v in tj]
    def project(v):return [sum((P[i,j]*v[j] for j in range(3)),Jet2.constant(0.,N)) for i in range(3)]
    if sum(a*b for a,b in zip(radialf,tf))<=0:
        distance=sqrt(dot(qj,qj))
        g=distance-R-Ro
        gradient=g.gradient[:size]; H=g.hessian[:size,:size]
        from .capped_cylinder import _sqrt_bound
        total=Fraction.from_float(R)+Fraction.from_float(Ro)
        stable=float((rho2-total*total)/(_sqrt_bound(rho2)+total))
        g.value=stable
        normal=radial/rho
        skin=x[endpoint]-R*normal; branch='sphere'
    else:
        seed=np.eye(3)[int(np.argmin(np.abs(d)))]
        ej=_cross(dj,seed); en=sqrt(dot(ej,ej)); ej=[v/en for v in ej]
        fj=_cross(dj,ej)
        e=np.array([v.value for v in ej]); f=np.array([v.value for v in fj])
        pe,pf=project(ej),project(fj)
        (value,theta,w),tied=_rim_station(radial,np.array([v.value for v in pe]),np.array([v.value for v in pf]),R)
        distance=math.sqrt(max(0.,value)); gap=distance-Ro
        if tied:
            if gap<=0:raise HemispherePlaneGeometryError('active cylinder rim minimum is nonunique')
            return HemisphereCylinderState(gap,_readonly(np.zeros(size)),_readonly(np.zeros((size,size))),None,None,None,'parallel_rim')
        t=Jet2.variable(theta,size,N)
        circle=[qj[j]+R*(pe[j]*cos(t)+pf[j]*sin(t)) for j in range(3)]
        squared=dot(circle,circle)
        if squared.value<=0:raise HemispherePlaneGeometryError('cylinder rim normal is unresolved')
        g=sqrt(squared)-Ro
        curvature=g.hessian[-1,-1]
        tolerance=512*np.finfo(float).eps*max(rho,R,Ro)
        if abs(g.gradient[-1])>tolerance or curvature<=tolerance:
            raise HemispherePlaneGeometryError('cylinder rim stationary work unresolved')
        gradient=g.gradient[:size]
        H=g.hessian[:size,:size]-np.outer(g.hessian[:size,-1],g.hessian[-1,:size])/curvature
        if abs(g.value)<=512*np.finfo(float).eps*max(rho,R,Ro):
            raise HemispherePlaneGeometryError('cylinder rim activation sign unresolved')
        normal=w/math.hypot(*w)
        skin=x[endpoint]+R*(e*math.cos(theta)+f*math.sin(theta)); branch='rim'
    gap=g.value
    rigid=skin-gap*normal
    if not (math.isfinite(gap) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(H))):
        raise HemispherePlaneGeometryError('cylinder support work is not representable')
    return HemisphereCylinderState(gap,_readonly(gradient),_readonly(H),_readonly(skin),_readonly(rigid),_readonly(normal),branch)
