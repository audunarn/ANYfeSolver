"""Reference-P1 joint integrals on guaranteed/possible Hermite overlap.

The caller owns orientation admission, temporal refinement and finite budgets.
Inner polynomials preserve temporal correlation across source vertices; signed
outer-minus-inner bands enclose the uncertain domain without a spatial raster.
"""
from fractions import Fraction as Q
from .s3_pressure_hermite_bounds import _fraction,_multiply,_plus
from .s3_overlap_hermite_gap import _clip


def _integral(vertices,coefficients):
    """Exact integral Q_i*f divided by original reference area; no child modes."""
    values=[Q(0),Q(0),Q(0)]
    for k in range(1,len(vertices)-1):
        triangle=(vertices[0],vertices[k],vertices[k+1])
        fraction=_fraction(triangle)
        fv=tuple(sum(a*b for a,b in zip(v,coefficients)) for v in triangle)
        for i in range(3):
            shapes=tuple(v[i] for v in triangle)
            values[i]+=fraction*(sum(shapes)*sum(fv)+sum(a*b for a,b in zip(shapes,fv)))/12
    return tuple(values)


def _signed_integral(vertices,h,factor,shift,*,lower,positive):
    # Split where the factor endpoint changes with H's sign. Identically zero
    # H selects one branch, avoiding double integration of the nonzero shift.
    if min(h)>=0:branches=((vertices,factor[0 if lower else 1]),)
    elif max(h)<=0:branches=((vertices,factor[1 if lower else 0]),)
    else:
        branches=((_clip(vertices,h),factor[0 if lower else 1]),
                  (_clip(vertices,tuple(-v for v in h)),factor[1 if lower else 0]))
    total=[Q(0),Q(0),Q(0)]
    for polygon,scale in branches:
        coefficients=tuple(scale*v+shift for v in h)
        region=_clip(polygon,coefficients if positive else tuple(-v for v in coefficients))
        for i,value in enumerate(_integral(region,coefficients)):total[i]+=value
    return tuple(total)


def _polygon_cell(face,vertices,tolerance,factor):
    inner=outer=vertices
    R=face.polynomials[7:]
    for edge in range(3):
        original=R[3*edge:3*edge+3]
        inner=_clip(inner,tuple(min(p) for p in original))
        outer=_clip(outer,tuple(max(p) for p in original))
    si=_integral(inner,(Q(1),)*3);so=_integral(outer,(Q(1),)*3)
    H=face.polynomials[4:7]
    # Integrate on fixed inner overlap before bounding time: separate source
    # vertex minima would lose weighted-gap cancellation.
    mass=tuple(_integral(inner,tuple(Q(int(i==j)) for i in range(3))) for j in range(3))
    weighted=tuple(tuple(sum(mass[j][i]*H[j][k] for j in range(3))
                         for i in range(3)) for k in range(len(H[0])))
    hlow=tuple(min(p) for p in H);hhigh=tuple(max(p) for p in H)
    supports=tuple((face.area[0]*lo,face.area[1]*hi) for lo,hi in zip(si,so))
    joints=[]
    for shift in (tolerance,-tolerance):
        if inner!=outer:
            negative_outer=_signed_integral(outer,hlow,factor,shift,lower=True,positive=False)
            negative_inner=_signed_integral(inner,hlow,factor,shift,lower=True,positive=False)
            positive_outer=_signed_integral(outer,hhigh,factor,shift,lower=False,positive=True)
            positive_inner=_signed_integral(inner,hhigh,factor,shift,lower=False,positive=True)
        joint=[]
        for i in range(3):
            coefficients=tuple(row[i] for row in weighted)
            inside=_plus(_multiply((min(coefficients),max(coefficients)),factor),
                         (shift*si[i],)*2)
            band=((negative_outer[i]-negative_inner[i],positive_outer[i]-positive_inner[i])
                  if inner!=outer else (Q(0),Q(0)))
            joint.append(_multiply(face.area,_plus(inside,band)))
        joints.append(tuple(joint))
    return supports,joints[0],joints[1],False
