"""Analytic value/Jacobian/Hessian of the experimental two-director trace.

Second-order forward differentiation uses the closed SO(3) expressions; no
perturbed geometry, constitutive state or numerical differentiation is used.
Rotation-domain and reference-geometry admission belong to the caller.
"""
import numpy as np
from math import factorial
from fractions import Fraction

_SINC = tuple((-1.)**n/factorial(2*n+1) for n in range(19))
_COSC = tuple((-1.)**n/factorial(2*n+2) for n in range(19))
_SINCC = tuple((-1.)**n/factorial(2*n+3) for n in range(19))


class DirectorTraceDomainError(ValueError):
    """The explicit trace left the existing principal/relative SO(3) chart."""


class _Jet:
    def __init__(self, value, gradient=None, hessian=None):
        self.v = float(value)
        self.d = np.zeros(6) if gradient is None else gradient
        self.h = np.zeros((6, 6)) if hessian is None else hessian

    def __add__(self, other):
        b = other if isinstance(other, _Jet) else _Jet(other)
        return _Jet(self.v+b.v, self.d+b.d, self.h+b.h)

    __radd__ = __add__

    def __neg__(self):
        return _Jet(-self.v, -self.d, -self.h)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        b = other if isinstance(other, _Jet) else _Jet(other)
        return _Jet(self.v*b.v, self.d*b.v+self.v*b.d,
                    self.h*b.v+self.v*b.h+np.outer(self.d,b.d)+np.outer(b.d,self.d))

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, _Jet):
            return self*_unary(other, 1./other.v, -1./other.v**2, 2./other.v**3)
        return self*(1./other)

    def __rtruediv__(self, other):
        return _unary(self, 1./self.v, -1./self.v**2, 2./self.v**3)*other


def _unary(x, value, first, second):
    return _Jet(value, first*x.d, first*x.h+second*np.outer(x.d,x.d))


def _sqrt(x):
    v=np.sqrt(x.v)
    return _unary(x,v,.5/v,-.25/v**3)


def _sin(x):
    return _unary(x,np.sin(x.v),np.cos(x.v),-np.sin(x.v))


def _cos(x):
    return _unary(x,np.cos(x.v),-np.sin(x.v),-np.cos(x.v))


def _acos(x):
    s=np.sqrt(1.-x.v*x.v)
    return _unary(x,np.arccos(x.v),-1./s,-x.v/s**3)


def _atan2(y,x):
    r=x.v*x.v+y.v*y.v
    fx=-y.v/r; fy=x.v/r
    fxx=2*x.v*y.v/r**2; fyy=-fxx; fxy=(y.v*y.v-x.v*x.v)/r**2
    return _Jet(np.arctan2(y.v,x.v),fx*x.d+fy*y.d,
        fx*x.h+fy*y.h+fxx*np.outer(x.d,x.d)+fyy*np.outer(y.d,y.d)
        +fxy*(np.outer(x.d,y.d)+np.outer(y.d,x.d)))


def _skew(x):
    return np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]],dtype=object)


def _scale(array, scalar):
    return np.array([item*scalar for item in array.flat],dtype=object).reshape(array.shape)


def _coefficients(theta):
    z=sum(t*t for t in theta)
    # Entire functions evaluated without cancellation or a branch switch.
    # On the admitted |theta|<pi domain the first omitted sinc term is
    # pi**38/39! < 1e-27; its first two angular derivatives are also below
    # double roundoff. Horner evaluation preserves the removable zero limit.
    def polynomial(coefficients):
        value=_Jet(coefficients[-1])
        for coefficient in reversed(coefficients[:-1]): value=value*z+coefficient
        return value
    return polynomial(_SINC),polynomial(_COSC),polynomial(_SINCC)


def _exp(theta):
    return np.eye(3,dtype=object)+_exp_minus_identity(theta)


def _exp_minus_identity(theta):
    a,b,_=_coefficients(theta); W=_skew(theta)
    return _scale(W,a)+_scale(W@W,b)


def _left(theta):
    _,b,c=_coefficients(theta); W=_skew(theta)
    return np.eye(3,dtype=object)+_scale(W,b)+_scale(W@W,c)


def _left_scaled_difference(phi, xi):
    """Jl(xi*phi)-Jl(phi), without subtracting their identity parts."""
    z=sum(t*t for t in phi); W=_skew(phi)
    def difference(coefficients, offset):
        result=_Jet(0.)
        for n in range(len(coefficients)-1,-1,-1):
            # xi**power-1=(xi-1)*sum(xi**k), including near endpoints.
            power=2*n+offset
            factor=(xi-1.)*sum(xi**k for k in range(power))
            result=result*z+coefficients[n]*factor
        return result
    return _scale(W,difference(_COSC,1))+_scale(W@W,difference(_SINCC,2))


def _reference_chord_closure(X, slave, eccentricity, xi):
    """Exact-reference arithmetic: retain closure, including sub-ulp gaps."""
    high=[]; low=[]; closure=[]
    for j in range(3):
        chord=Fraction(float(X[1,j]))-Fraction(float(X[0,j]))
        hi=float(chord)
        high.append(hi); low.append(float(chord-Fraction(hi)))
        closure.append(float(Fraction(float(X[0,j]))+Fraction(float(xi))*chord
                             +Fraction(float(eccentricity[j]))-Fraction(float(slave[j]))))
    return np.array(high),np.array(low),np.array(closure)


def _log(R):
    c=(R[0,0]+R[1,1]+R[2,2]-1.)/2.
    z=1.-c
    if z.v < 1e-3:
        a=.5+z/6.+z*z/15.+z*z*z/35.+4.*z*z*z*z/315.+4.*z*z*z*z*z/693.
        return _scale(np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]],dtype=object),a)
    # Dominant quaternion-vector extraction stays regular near pi: scalar
    # part comes from an antisymmetric entry, never sqrt(1+trace(R)).
    i=max(range(3),key=lambda j:R[j,j].v); j=(i+1)%3; k=(i+2)%3
    v=[None,None,None]
    v[i]=.5*_sqrt(1.+R[i,i]-R[j,j]-R[k,k])
    v[j]=(R[j,i]+R[i,j])/(4.*v[i]); v[k]=(R[k,i]+R[i,k])/(4.*v[i])
    w=(R[k,j]-R[j,k])/(4.*v[i])
    if w.v < 0.:
        w=-w; v=[-component for component in v]
    norm=_sqrt(sum(component*component for component in v))
    return _scale(np.array(v,dtype=object),2.*_atan2(norm,w)/norm)


def _inverse(A):
    a,b,c=A[:,0],A[:,1],A[:,2]
    def cross(x,y):
        return np.array([x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]],dtype=object)
    bc=cross(b,c); ca=cross(c,a); ab=cross(a,b)
    det=sum(a[i]*bc[i] for i in range(3))
    return _scale(np.array([bc,ca,ab],dtype=object),1./det)


def director_trace_state(reference_masters, reference_slave, eccentricity, xi, master_state):
    """Return y=(slave u,theta), dy/d(master u,theta), and all six Hessians.

    Master-state order is (u0,theta0,u1,theta1). The 12x12 Hessians include
    translation/rotation cross terms. Inputs are observed, never modified.
    """
    X=np.asarray(reference_masters,dtype=float); x=np.asarray(master_state,dtype=float).reshape(12)
    chart_limit=np.pi-1e-6
    if (not np.all(np.isfinite(x)) or np.linalg.norm(x[3:6])>=chart_limit
            or np.linalg.norm(x[9:12])>=chart_limit):
        raise DirectorTraceDomainError("director trace outside principal rotation-vector chart")
    theta=[]
    for i,value in enumerate(np.r_[x[3:6],x[9:12]]):
        d=np.zeros(6); d[i]=1.; theta.append(_Jet(value,d))
    E0=_exp_minus_identity(theta[:3]); R0=np.eye(3,dtype=object)+E0
    R1=_exp(theta[3:]); phi=_log(R0.T@R1)
    if np.linalg.norm([component.v for component in phi])>=chart_limit:
        raise DirectorTraceDomainError("ambiguous shortest-geodesic director trace")
    RminusI=E0+R0@_exp_minus_identity(phi*xi)
    R=np.eye(3,dtype=object)+RminusI
    QminusXiI=(R0@_left_scaled_difference(phi,xi)@_inverse(_left(phi))@R0.T)*xi
    Q=np.eye(3,dtype=object)*xi+QminusXiI
    chord_hi,chord_lo,closure=_reference_chord_closure(X,reference_slave,eccentricity,xi)
    position=(x[:3]+Q@(x[6:9]-x[:3])+QminusXiI@chord_hi
              +QminusXiI@chord_lo+RminusI@np.asarray(eccentricity,dtype=float)+closure)
    outputs=list(position)+list(_log(R))
    value=np.array([o.v for o in outputs]); J=np.zeros((6,12)); H=np.zeros((6,12,12))
    if np.linalg.norm(value[3:])>=chart_limit:
        raise DirectorTraceDomainError("director trace reached principal rotation-vector chart boundary")
    ridx=np.array([3,4,5,9,10,11]); qv=np.array([[o.v for o in row] for row in Q])
    J[:3,:3]=np.eye(3)-qv; J[:3,6:9]=qv
    for i,o in enumerate(outputs):
        J[i,ridx]=o.d; H[i][np.ix_(ridx,ridx)]=o.h
    for i in range(3):
        for j in range(3):
            H[i,j,ridx]=-Q[i,j].d; H[i,ridx,j]=-Q[i,j].d
            H[i,6+j,ridx]=Q[i,j].d; H[i,ridx,6+j]=Q[i,j].d
    return value,J,H
