"""Conservative same-value-graph empty clipping, full37 kernel fallback.

The uniform derivative bounds are admission evidence only, never returned
physical derivatives. Every uncertain/exceptional bound uses the old kernel.
"""
import math
import numpy as np
from .enriched_overlap import (SIZE, readonly, EnrichedGeometryError, EnrichedMoments, QuadraturePolicy)
_POLICY_VALIDATE = QuadraturePolicy.validate
_POLICY_VALIDATE_CODE = QuadraturePolicy.validate.__code__


class _Fallback(Exception): pass


def _up(value):
    # Bounds far below binary64 overflow, including later outer products.
    if not math.isfinite(value) or value > 1e100: raise _Fallback()
    return float(np.nextafter(value, math.inf)) if value else 0.


class Jet2:
    __slots__ = ('value','g','h')
    def __init__(self,value,g=0.,h=0.):
        self.value=float(value)
        if not math.isfinite(self.value): raise _Fallback()
        self.g,self.h=_up(g),_up(h)
    @classmethod
    def constant(cls,value,size): return cls(value)
    @classmethod
    def variable(cls,value,index,size): return cls(value,1.,0.)
    def _coerce(self,other): return other if isinstance(other,Jet2) else Jet2(float(other))
    def __add__(self,other):
        other=self._coerce(other)
        return Jet2(self.value+other.value,_up(self.g+other.g),_up(self.h+other.h))
    __radd__=__add__
    def __neg__(self): return Jet2(-self.value,self.g,self.h)
    def __sub__(self,other): return self+(-self._coerce(other))
    def __rsub__(self,other): return self._coerce(other)-self
    def __mul__(self,other):
        other=self._coerce(other)
        g=_up(_up(self.g*abs(other.value))+_up(other.g*abs(self.value)))
        h=_up(_up(_up(_up(self.h*abs(other.value))+_up(other.h*abs(self.value)))
                      +_up(self.g*other.g))+_up(other.g*self.g))
        return Jet2(self.value*other.value,g,h)
    __rmul__=__mul__
    def reciprocal(self):
        if self.value==0.: raise ZeroDivisionError('jet reciprocal is singular')
        inverse=1./self.value
        first,second=-(inverse**2),2.*inverse**3
        return unary(self,inverse,first,second)
    def __truediv__(self,other): return self*self._coerce(other).reciprocal()
    def __rtruediv__(self,other): return self._coerce(other)*self.reciprocal()


def unary(a,value,first,second):
    return Jet2(value,_up(abs(first)*a.g),
        _up(_up(abs(first)*a.h)+_up(abs(second)*_up(a.g*a.g))))


def sqrt(a):
    if a.value<=0.: raise ValueError('jet square root requires a positive value')
    root=float(np.sqrt(a.value))
    return unary(a,root,.5/root,-.25/root**3)


def const(x): return Jet2.constant(x,SIZE)

def dot(a,b): return sum((x*y for x,y in zip(a,b)),const(0))
def sub(a,b): return [x-y for x,y in zip(a,b)]
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def cross2(a,b): return a[0]*b[1]-a[1]*b[0]
def unit(a):
    length=sqrt(dot(a,a))
    return [x/length for x in a]


def _empty_candidate(reference_source, current_source, current_target, beta, direction,
            policy, *, current_midsurface, internal_edges, target_node_ids, cancel=lambda: False):
    """Reference-area moments M_i, b_i and int(N_i N_j), with complete jets."""
    policy.validate()
    ref = readonly(reference_source, (4, 3))
    source = readonly(current_source, (4, 3))
    target = readonly(current_target, (3, 3))
    beta = readonly(beta, (4,))
    midsurface = readonly(current_midsurface, (4, 3))
    direction = readonly(direction, (3,))
    if not np.isclose(direction @ direction, 1., rtol=0, atol=64*np.finfo(float).eps):
        raise EnrichedGeometryError("unit physical reference director required")
    variables = [Jet2.variable(x, i, SIZE) for i, x in enumerate(
        np.concatenate((source.ravel(), target.ravel(), beta, midsurface.ravel())))]
    q = [variables[3*i:3*i+3] for i in range(4)]
    t = [variables[12+3*i:15+3*i] for i in range(3)]
    bb = variables[21:25]
    mid = [variables[25+3*i:28+3*i] for i in range(4)]
    body_normal = unit(cross(sub(mid[1], mid[0]), sub(mid[3], mid[0])))
    normal = unit(cross(sub(q[1], q[0]), sub(q[3], q[0])))
    tangent = unit(sub(q[1], q[0]))
    second = cross(normal, tangent)
    reference_normal = np.cross(ref[1]-ref[0], ref[3]-ref[0])
    reference_normal /= np.linalg.norm(reference_normal)
    if not np.allclose(direction, reference_normal, rtol=0, atol=128*np.finfo(float).eps):
        raise EnrichedGeometryError("reference director does not match source orientation")
    target_normal = cross(sub(t[1], t[0]), sub(t[2], t[0]))
    if dot(target_normal, normal).value >= 0:
        raise EnrichedGeometryError("target S3 must face the source skin")

    def project(p):
        return [dot(sub(p, q[0]), tangent), dot(sub(p, q[0]), second)]

    quad = [project(p) for p in q]
    triangle = [project(p) for p in t]
    scale = max(float(np.max(np.abs(source-source[0]))),
                float(np.max(np.abs(target-source[0]))), np.finfo(float).tiny)
    resolution = 256*np.finfo(float).eps*scale*scale
    area2 = cross2(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]))
    if abs(area2.value) <= resolution:
        raise EnrichedGeometryError("degenerate projected S3")
    target_order = (0,2,1) if area2.value < 0 else (0,1,2)
    if area2.value < 0:
        triangle = [triangle[0], triangle[2], triangle[1]]
        t = [t[0], t[2], t[1]]
        area2 = -area2
    for i in range(4):
        if cross2(sub(quad[(i+1) % 4], quad[i]),
                  sub(quad[(i+2) % 4], quad[(i+1) % 4])).value <= resolution:
            raise EnrichedGeometryError("nonconvex projected Q4")
    polygon = quad
    signature = []
    # triangle/t were reversed together above; retain actual node identities.
    ids = [target_node_ids[i] for i in target_order]
    internal = frozenset(frozenset(edge) for edge in internal_edges)
    admitted_internal_tie = False
    for edge in range(3):
        if not polygon:
            break
        start, end = triangle[edge], triangle[(edge+1) % 3]
        line = sub(end, start)
        old = polygon[-1]
        old_d = cross2(line, sub(old, start))
        out = []
        for point in polygon:
            d = cross2(line, sub(point, start))
            # Only a pre-certified INTERNAL vertex tie is permitted here.
            # External ties and shared-edge collinearity remain unsupported.
            if min(abs(old_d.value), abs(d.value)) <= resolution:
                if frozenset((ids[edge],ids[(edge+1)%3])) not in internal:
                    raise EnrichedGeometryError("external vertex on clipping boundary")
                admitted_internal_tie = True
            old_in, inside = old_d.value > 0, d.value > 0
            signature.append(int(inside))
            if old_in != inside:
                alpha = old_d/(old_d-d)
                out.append([a+alpha*b for a, b in zip(old, sub(point, old))])
            if inside:
                out.append(point)
            old, old_d = point, d
        polygon = out
    if len(polygon) < 3:
        zeros = np.zeros(4)
        return EnrichedMoments(readonly(zeros), readonly(zeros), readonly(np.zeros((4, 4))),
                               readonly(np.zeros((4, SIZE))), readonly(np.zeros((4, SIZE))),
                               readonly(np.zeros((4, SIZE, SIZE))),
                               readonly(np.zeros((4, SIZE, SIZE))), tuple(signature), 0, (0., 0., 0.))

    return None


def empty_preflight(*args, **kwargs):
    # A raising or warning-underflow global mode can expose derivative-array
    # arithmetic itself; do not replace that observable original execution.
    if (any(type(x) is not np.ndarray or x.dtype != np.float64 for x in args[:5])
            or type(kwargs.get('current_midsurface')) is not np.ndarray
            or kwargs['current_midsurface'].dtype != np.float64
            or type(kwargs.get('internal_edges')) is not tuple
            or type(kwargs.get('target_node_ids')) is not tuple): return None
    policy=args[5]
    if (type(policy) is not QuadraturePolicy or QuadraturePolicy.validate is not _POLICY_VALIDATE
            or QuadraturePolicy.validate.__code__ is not _POLICY_VALIDATE_CODE
            or any(type(x) is not float for x in (policy.value_absolute,policy.gradient_absolute,policy.hessian_absolute))
            or type(policy.maximum_triangles) is not int
            or any(type(e) is not tuple or any(type(i) is not int for i in e) for e in kwargs['internal_edges'])
            or any(type(i) is not int for i in kwargs['target_node_ids'])): return None
    if any(a.shape!=shape for a,shape in zip(args[:5],((4,3),(4,3),(3,3),(4,),(3,)))):
        return None
    if kwargs['current_midsurface'].shape!=(4,3):return None
    modes=np.geterr()
    if modes!={'divide':'warn','over':'warn','under':'ignore','invalid':'warn'}: return None
    # Pure scalar eligibility BEFORE warning-capable reference normalization.
    # This is fallback eligibility, never a new admitted geometric domain.
    values=[float(x) for a in (*args[:5],kwargs['current_midsurface']) for x in a.flat]
    if any(not math.isfinite(x) or abs(x)>1e20 for x in values): return None
    ref=args[0]
    if ref.shape!=(4,3): return None
    a=[float(ref[1,k])-float(ref[0,k]) for k in range(3)]
    b=[float(ref[3,k])-float(ref[0,k]) for k in range(3)]
    normal=(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    if not 1e-20<max(abs(x)for x in normal)<1e40:return None
    try:
        return _empty_candidate(*args, **kwargs)
    except (_Fallback, ArithmeticError, ValueError):
        return None  # Original evaluation owns every refusal/exception.
