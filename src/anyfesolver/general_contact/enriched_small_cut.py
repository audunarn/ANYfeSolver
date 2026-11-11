"""Certified represented-geometry small cuts; nonlinear-profile internal only.

Exact rational predicates bind the INPUT halfplanes and intersection topology.
The integration chart keeps all37 geometric derivatives; no area is discarded.
This is a finite-precision domain admission, not a physical-gap certificate.
"""
from fractions import Fraction
from dataclasses import dataclass
import math
import numpy as np
from .enriched_overlap import Jet2,SIZE,const,EnrichedGeometryError,SolveCancelled
from .moment_batch_ad import PointJet


def refuse(message):
    raise EnrichedGeometryError('small-cut certification: '+message)


def rational(x):
    x=float(x)
    if not math.isfinite(x):refuse('nonfinite represented geometry')
    return Fraction.from_float(x)


def _sub(a,b):return (a[0]-b[0],a[1]-b[1])
def _cross(a,b):return a[0]*b[1]-a[1]*b[0]
def _area(v):return _cross(_sub(v[1],v[0]),_sub(v[2],v[0]))/2


def _interval_sub(a,b):return (a[0]-b[1],a[1]-b[0])
def _interval_mul(a,b):
    products=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1])
    return (min(products),max(products))


def _orientation_interval(v):
    a=[_interval_sub(v[1][k],v[0][k])for k in range(2)]
    b=[_interval_sub(v[2][k],v[0][k])for k in range(2)]
    out=_interval_sub(_interval_mul(a[0],b[1]),_interval_mul(a[1],b[0]))
    return (out[0]/2,out[1]/2)


def _exact_clip(source,target):
    """Same Sutherland-Hodgman expressions/order, exact binary64-input rationals."""
    polygon=list(source);signature=[]
    if len(source)!=4 or len(target)!=3 or _area(target)<=0:refuse('input polygon shape/orientation')
    for i in range(4):
        if _cross(_sub(source[(i+1)%4],source[i]),_sub(source[(i+2)%4],source[(i+1)%4]))<=0:
            refuse('nonconvex represented source')
    for edge in range(3):
        if not polygon:break
        start,end=target[edge],target[(edge+1)%3];line=_sub(end,start)
        old=polygon[-1];old_d=_cross(line,_sub(old,start));out=[]
        for point in polygon:
            d=_cross(line,_sub(point,start))
            if old_d==0 or d==0:refuse('uncertain/exact boundary tie')
            old_in,inside=old_d>0,d>0;signature.append(int(inside))
            if old_in!=inside:
                alpha=old_d/(old_d-d)
                out.append(tuple(old[k]+alpha*(point[k]-old[k])for k in range(2)))
            if inside:out.append(point)
            old,old_d=point,d
        polygon=out
    return polygon,tuple(signature)


def _point_values(points):
    return [tuple(rational(x.value)for x in point)for point in points]


class _Exact37:
    """Exact second-order Taylor coefficients, ONLY the small clipping/chart."""
    __slots__=('value','gradient','hessian')
    def __init__(self,value,gradient,hessian):
        self.value=value;self.gradient=gradient;self.hessian=hessian
    @classmethod
    def from_jet(cls,jet):
        if jet.gradient.shape!=(SIZE,)or jet.hessian.shape!=(SIZE,SIZE):refuse('exact37 input shape')
        return cls(rational(jet.value),tuple(rational(x)for x in jet.gradient),
                   tuple(tuple(rational(x)for x in row)for row in jet.hessian))
    def __add__(self,other):
        return _Exact37(self.value+other.value,tuple(a+b for a,b in zip(self.gradient,other.gradient)),
                        tuple(tuple(a+b for a,b in zip(ar,br))for ar,br in zip(self.hessian,other.hessian)))
    def __sub__(self,other):return self+other.scale(Fraction(-1))
    def scale(self,number):
        return _Exact37(self.value*number,tuple(a*number for a in self.gradient),
                        tuple(tuple(a*number for a in row)for row in self.hessian))
    def __mul__(self,other):
        a,b=self,other
        return _Exact37(a.value*b.value,
            tuple(a.gradient[i]*b.value+a.value*b.gradient[i]for i in range(SIZE)),
            tuple(tuple(a.hessian[i][j]*b.value+a.value*b.hessian[i][j]+a.gradient[i]*b.gradient[j]+b.gradient[i]*a.gradient[j]
                        for j in range(SIZE))for i in range(SIZE)))
    def __truediv__(self,other):
        if other.value==0:refuse('exact clipping denominator')
        inverse=1/other.value
        reciprocal=_Exact37(inverse,tuple(-g*inverse**2 for g in other.gradient),
            tuple(tuple(2*other.gradient[i]*other.gradient[j]*inverse**3-other.hessian[i][j]*inverse**2
                        for j in range(SIZE))for i in range(SIZE)))
        return self*reciprocal


def _exact_clip37(source,target,cancel):
    polygon=[tuple(_Exact37.from_jet(x)for x in point)for point in source]
    triangle=[tuple(_Exact37.from_jet(x)for x in point)for point in target];signature=[]
    for edge in range(3):
        if cancel():raise SolveCancelled('enriched integration cancelled')
        if not polygon:break
        start,end=triangle[edge],triangle[(edge+1)%3];line=_sub(end,start)
        old=polygon[-1];old_d=_cross(line,_sub(old,start));out=[]
        for point in polygon:
            if cancel():raise SolveCancelled('enriched integration cancelled')
            d=_cross(line,_sub(point,start))
            if d.value==0 or old_d.value==0:refuse('exact37 boundary tie')
            inside=d.value>0;signature.append(int(inside))
            if (old_d.value>0)!=inside:
                alpha=old_d/(old_d-d)
                out.append(tuple(old[k]+alpha*(point[k]-old[k])for k in range(2)))
            if inside:out.append(point)
            old,old_d=point,d
        polygon=out
    return polygon,tuple(signature)


class _ConversionBounds:
    """Compact group envelopes valid for EVERY correctly-rounded coefficient.

    Exact rational comparisons verify the half-ULP guarantee, including the
    subnormal half-ULP2**-1075 which must not be rounded to binary64 zero in
    the report. No whole-integrand/output-H error bound is inferred.
    """
    def __init__(self):
        self.bounds=[None,None,None];self.counts=[0,0,0];self.inexact=[0,0,0];self.events=0
    def convert(self,jet,slot):
        values=(jet.value,*jet.gradient,*(v for row in jet.hessian for v in row));made=[]
        for index,exact in enumerate(values):
            try:value=float(exact)
            except OverflowError:refuse('exact37 conversion overflow')
            if not math.isfinite(value):refuse('nonfinite exact37 conversion')
            error=exact-Fraction.from_float(value)
            group=0 if index==0 else(1 if index<=SIZE else 2)
            self.counts[group]+=1
            if error:
                half_ulp=Fraction.from_float(math.ulp(value))/2
                if abs(error)>half_ulp:refuse('coefficient conversion not correctly rounded')
                exponent=math.frexp(math.ulp(value))[1]-2
                self.inexact[group]+=1
                self.bounds[group]=exponent if self.bounds[group]is None else max(self.bounds[group],exponent)
            made.append(value)
        self.events+=1
        return Jet2(made[0],np.array(made[1:1+SIZE]),np.array(made[1+SIZE:]).reshape(SIZE,SIZE))
    def report(self):
        return dict(groups=['value','gradient37','hessian37x37'],coefficient_counts=self.counts,
            inexact_counts=self.inexact,max_absolute_error_power2=self.bounds,
            interpretation='For EVERY conversion: abs(exact coefficient - returned binary64)<=2**groupExponent; null=>all conversions exact. No output/integrand error bound.',
            conversion_events=self.events,correct_rounding='exact rational half-ULP check including subnormal exponent-1075')


def _rounded_clip(source,target):
    """Authenticate the EXISTING Jet2 primal operation order, not a new clip law."""
    polygon=list(source);signature=[]
    for edge in range(3):
        if not polygon:break
        start,end=target[edge],target[(edge+1)%3];line=_sub(end,start)
        old=polygon[-1];old_d=_cross(line,_sub(old,start));out=[]
        for point in polygon:
            d=_cross(line,_sub(point,start));inside=d>0
            signature.append(int(inside))
            if (old_d>0)!=inside:
                # Jet2 division is multiplication by the reciprocal.
                if old_d-d==0:refuse('rounded clipping denominator')
                alpha=old_d*(1./(old_d-d))
                out.append(tuple(old[k]+alpha*(point[k]-old[k])for k in range(2)))
            if inside:out.append(point)
            old,old_d=point,d
        polygon=out
    return polygon,tuple(signature)


@dataclass(frozen=True)
class SmallCutCertificate:
    root_index:int
    exact_area:Fraction
    rounding_area_lower:Fraction
    rounding_area_upper:Fraction
    signature:tuple

    def report(self):
        return dict(method='INPUT_HALFPLANE_EXACT_RATIONAL_SMALL_CUT_V1',root_index=self.root_index,
                    exact_area_hex=float(self.exact_area).hex(),
                    orientation_lower_hex=float(self.rounding_area_lower).hex(),
                    orientation_upper_hex=float(self.rounding_area_upper).hex(),
                    clipping_signature=list(self.signature),physical_acceptance=False)


def certify_positive_cut(vertices,source,target,polygon,signature,root_index):
    exact,actual_signature=_exact_clip(_point_values(source),_point_values(target))
    rounded,rounded_signature=_rounded_clip([tuple(float(x.value)for x in p)for p in source],
                                           [tuple(float(x.value)for x in p)for p in target])
    if (rounded_signature!=tuple(signature)
            or rounded!=[tuple(float(x.value)for x in p)for p in polygon]):
        refuse('production clipping primal/order binding disagrees')
    if actual_signature!=tuple(signature)or len(exact)!=len(polygon)or len(exact)<3:
        refuse('input halfplane/intersection topology disagrees')
    if type(root_index)is not int or not 0<=root_index<len(exact)-2:refuse('root index')
    expected=(exact[0],exact[root_index+1],exact[root_index+2])
    actual=_point_values(vertices)
    if len(vertices)!=3 or any(vertices[k]is not polygon[i]for k,i in enumerate((0,root_index+1,root_index+2))):
        refuse('fan-root identity')
    # Each component enclosure includes BOTH exact intersection and represented
    # production endpoint. Its strictly-positive interval determinant certifies
    # orientation despite their complete observed rounding displacement.
    boxes=[[(min(a,b),max(a,b))for a,b in zip(e,v)]for e,v in zip(expected,actual)]
    lower,upper=_orientation_interval(boxes);area=_area(expected)
    if area<=0 or lower<=0:refuse('orientation uncertain under clipping rounding enclosure')
    # All actual clipped vertices must have finite37 jets. No derivative is zeroed.
    for point in polygon:
        if len(point)!=2:refuse('point dimension')
        for jet in point:
            if (jet.gradient.shape!=(SIZE,)or jet.hessian.shape!=(SIZE,SIZE)
                    or not np.all(np.isfinite(jet.gradient))or not np.all(np.isfinite(jet.hessian))):
                refuse('nonfinite/wrong37 geometry chain')
    return SmallCutCertificate(root_index,area,lower,upper,actual_signature)



def _inside_box(point_box,triangle):
    for i in range(3):
        a,b=triangle[i],triangle[(i+1)%3];edge=_sub(b,a)
        offset=[(point_box[k][0]-a[k],point_box[k][1]-a[k])for k in range(2)]
        cross=_interval_sub(_interval_mul((edge[0],edge[0]),offset[1]),
                            _interval_mul((edge[1],edge[1]),offset[0]))
        if cross[0]<=0:return False
    return True


def chart_integral(vertices,integrand,triangle_rule,tolerance,maximum_triangles,cancel=lambda:False,*,exact_vertices=None):
    """Exact-dyadic local subdivision, existing25-point/full24-jet adaptive law."""
    if exact_vertices is None:exact_vertices=[tuple(_Exact37.from_jet(x)for x in p)for p in vertices]
    root_values=[tuple(x.value for x in point)for point in exact_vertices]
    area_exact=_cross(_sub(exact_vertices[1],exact_vertices[0]),_sub(exact_vertices[2],exact_vertices[0])).scale(Fraction(1,2))
    if area_exact.value<=0:refuse('exact37 nonpositive chart area')
    conversions=_ConversionBounds()
    e1=[exact_vertices[1][k]-exact_vertices[0][k]for k in range(2)]
    e2=[exact_vertices[2][k]-exact_vertices[0][k]for k in range(2)]
    used=0;difference=np.zeros(3)
    def rule(local):
        nonlocal used
        if cancel():raise SolveCancelled('enriched integration cancelled')
        used+=1
        if used>maximum_triangles:raise EnrichedGeometryError('enriched quadrature budget exhausted')
        determinant=_cross(_sub(local[1],local[0]),_sub(local[2],local[0]))
        if determinant<=0:refuse('local simplex orientation')
        ratio=float(determinant)
        if not math.isfinite(ratio)or ratio<=0 or Fraction.from_float(ratio)!=determinant:
            refuse('dyadic area factor underflow/rounding')
        global_triangle=[tuple(root_values[0][k]+uv[0]*(root_values[1][k]-root_values[0][k])+uv[1]*(root_values[2][k]-root_values[0][k])for k in range(2))for uv in local]
        points=[];seen=set()
        for point_index,(bary,relative)in enumerate(triangle_rule):
            if cancel():raise SolveCancelled('enriched integration cancelled')
            # Freeze reference quadrature/subdivision parameters ONLY.
            uv=tuple(local[0][k]+rational(bary[1])*(local[1][k]-local[0][k])+rational(bary[2])*(local[2][k]-local[0][k])for k in range(2))
            expected=tuple(root_values[0][k]+uv[0]*(root_values[1][k]-root_values[0][k])+uv[1]*(root_values[2][k]-root_values[0][k])for k in range(2))
            exact_point=[exact_vertices[0][k]+e1[k].scale(uv[0])+e2[k].scale(uv[1])for k in range(2)]
            point=[conversions.convert(x,1+2*point_index+k)for k,x in enumerate(exact_point)]
            actual=tuple(rational(x.value)for x in point)
            box=[(min(x,y),max(x,y))for x,y in zip(expected,actual)]
            if actual in seen or not _inside_box(box,global_triangle):
                refuse('global Gauss reconstruction collapsed/uncertain in local cut')
            seen.add(actual);points.append(point)
        if len(points)!=25:refuse('exact25point rule required')
        vals=integrand(points)
        if len(vals)!=24:refuse('full24 moment jets required')
        relative=np.array([weight for bary,weight in triangle_rule])
        area=conversions.convert(area_exact.scale(determinant),0)
        if area.value<=0:refuse('positive local area is not representable')
        weighted_area=PointJet.broadcast(area,25)*relative
        return [(weighted_area*value).sum_to_jet()for value in vals]
    def split(v):
        a,b,c=v;ab=tuple((x+y)/2 for x,y in zip(a,b));bc=tuple((x+y)/2 for x,y in zip(b,c));ca=tuple((x+y)/2 for x,y in zip(c,a))
        return ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))
    def integrate(local,tol,coarse=None):
        coarse=rule(local)if coarse is None else coarse
        children=split(local);parts=[rule(v)for v in children]
        fine=[sum((part[i]for part in parts),const(0))for i in range(24)]
        error=np.array((max(abs(a.value-b.value)for a,b in zip(fine,coarse)),
                        max(np.max(np.abs(a.gradient-b.gradient))for a,b in zip(fine,coarse)),
                        max(np.max(np.abs(a.hessian-b.hessian))for a,b in zip(fine,coarse))))
        if not np.all(np.isfinite(error)):refuse('nonfinite adaptive24 error')
        if np.all(error<=tol):difference[:]+=error;return fine
        refined=[integrate(v,tol/4,part)for v,part in zip(children,parts)]
        return [sum((part[i]for part in refined),const(0))for i in range(24)]
    root=((Fraction(0),Fraction(0)),(Fraction(1),Fraction(0)),(Fraction(0),Fraction(1)))
    out=integrate(root,np.asarray(tolerance))
    return out,used,difference,conversions.report()


def integrate_positive_cut(vertices,source,target,polygon,signature,root_index,
                           integrand,triangle_rule,tolerance,maximum_triangles,cancel=lambda:False):
    if cancel():raise SolveCancelled('enriched integration cancelled')
    certificate=certify_positive_cut(vertices,source,target,polygon,signature,root_index)
    exact,exact_signature=_exact_clip37(source,target,cancel)
    if exact_signature!=tuple(signature)or len(exact)!=len(polygon):refuse('full37 topology binding')
    exact_root=(exact[0],exact[root_index+1],exact[root_index+2])
    if _area([tuple(x.value for x in point)for point in exact_root])!=certificate.exact_area:
        refuse('exact37 clipping value chain disagrees with input certificate')
    out,used,difference,bounds=chart_integral(vertices,integrand,triangle_rule,tolerance,maximum_triangles,cancel,exact_vertices=exact_root)
    report=certificate.report();report['full37_clipping_chain']='EXACT_RATIONAL_INPUT_TAYLOR_INTERSECTIONS'
    report['conversion_enclosures']=bounds
    return out,used,difference,report
