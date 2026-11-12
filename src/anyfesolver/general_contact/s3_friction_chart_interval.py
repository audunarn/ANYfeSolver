"""Fail-closed primary-operation enclosure of a full S3 clipping branch.

Input polynomials already include authoritative map/input roundoff. The box
test encloses normalization, projection and determinants in their actual
order; it admits only the all-source-vertices-strictly-inside branch.
"""
from fractions import Fraction as F
from math import comb,isqrt
from .s3_friction_dyadic_interval import compress

U=F(1,2**53)
FLOOR=F(1,2**1074)


def box(p):
    d=len(p)-1
    controls=[(sum(p[i][0]*F(comb(k,i),comb(d,i)) for i in range(k+1)),
               sum(p[i][1]*F(comb(k,i),comb(d,i)) for i in range(k+1))) for k in range(d+1)]
    return compress((min(v[0] for v in controls),max(v[1] for v in controls)))


def rounded(a):
    error=U/(1-U)*max(map(abs,a))+FLOOR
    return compress((a[0]-error,a[1]+error))


def add(a,b):return rounded((a[0]+b[0],a[1]+b[1]))
def sub(a,b):return rounded((a[0]-b[1],a[1]-b[0]))
def mul(a,b):
    values=[x*y for x in a for y in b]
    return rounded((min(values),max(values)))
def div(a,b):
    if b[0]<=0<=b[1]:raise ValueError('interval denominator contains zero')
    # Jet2 division evaluates reciprocal first, then a multiplication. This
    # wider route also encloses the target frame's direct NumPy quotient.
    reciprocal=rounded((min(F(1)/x for x in b),max(F(1)/x for x in b)))
    return mul(a,reciprocal)
def square(a):
    values=[x*x for x in a]
    return rounded((F(0) if a[0]<=0<=a[1] else min(values),max(values)))


def sqrt_interval(a):
    if a[0]<=0:raise ValueError('interval normal is unresolved')
    # Exact integer square root supplies a dyadic enclosing bracket. No libm
    # result is treated as an exact mathematical endpoint. One IEEE square-
    # root rounding enclosure is then applied to the primary value operation.
    def bracket(x):
        exponent=max(128,128+(x.denominator.bit_length()-x.numerator.bit_length())//2)
        if exponent>1024:raise ValueError('interval square root exceeds bounded precision')
        unit=1<<exponent
        q=isqrt((x.numerator << (2*exponent))//x.denominator)
        return F(q,unit),F(q+1,unit)
    return rounded((bracket(a[0])[0],bracket(a[1])[1]))


def vector_sub(a,b):return tuple(sub(x,y) for x,y in zip(a,b))
def cross(a,b):return tuple(sub(mul(a[i],b[j]),mul(a[j],b[i])) for i,j in ((1,2),(2,0),(0,1)))
def dot(a,b):
    value=(F(0),F(0))
    for x,y in zip(a,b):value=add(value,mul(x,y))
    return value
def norm(a):
    value=(F(0),F(0))
    for x in a:value=add(value,square(x))
    return sqrt_interval(value)
def normalize(a):
    length=norm(a)
    return tuple(div(x,length) for x in a)
def cross2(a,b):return sub(mul(a[0],b[1]),mul(a[1],b[0]))


def full_branch(source_paths,target_paths,material_paths=()):
    source=[tuple(box(p) for p in point) for point in source_paths]
    target=[tuple(box(p) for p in point) for point in target_paths]
    material=[tuple(box(p) for p in point) for point in material_paths]
    try:
        for chart,others in ((source,target),(target,source+material)):
            edge=vector_sub(chart[1],chart[0])
            normal=normalize(cross(edge,vector_sub(chart[2],chart[0])))
            tangent=normalize(edge);binormal=cross(normal,tangent)
            def projected(p):
                delta=vector_sub(p,chart[0])
                return dot(delta,tangent),dot(delta,binormal)
            xy=[((F(0),F(0)),(F(0),F(0))),*[projected(p) for p in chart[1:]]]
            other_xy=[projected(p) for p in others]
            # L1 upper bound avoids transferring hypot/libm rounding. The
            # factor two is a conservative representable upper envelope.
            scale=2*max(sum(max(map(abs,c)) for c in p) for p in (*xy,*other_xy))
            threshold=mul(mul((F(1,10**10),F(1,10**10)),(scale,scale)),(scale,scale))[1]
            if mul(cross2(xy[1],xy[2]),(F(1,2),F(1,2)))[0]<=threshold:return False
            polygon=other_xy[:3] if chart is source else xy
            points=xy if chart is source else other_xy
            signed=(F(0),F(0))
            for i,p in enumerate(polygon):signed=add(signed,cross2(p,polygon[(i+1)%3]))
            if mul(signed,(F(1,2),F(1,2)))[0]<=threshold:return False
            for i,a in enumerate(polygon):
                b=polygon[(i+1)%3]
                edge2=vector_sub(b,a)
                for p in points:
                    if cross2(edge2,vector_sub(p,a))[0]<=threshold:return False
        return True
    except (ValueError,OverflowError):
        return False
