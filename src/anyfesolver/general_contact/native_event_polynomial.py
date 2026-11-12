"""Precise time numerator for an admitted straight-round/affine-triangle path.

The plane numerator is a cubic polynomial plus two quadratic polynomials
times sin(w*t) and cos(w*t). Its zero set is the physical gap zero set;
normalisation by the positive triangle area does not change crossing order.
This is a scalar geometry certificate, not a material or tangent approximation.
"""
from decimal import Decimal, localcontext
from math import comb, factorial, pi

import numpy as np

from .native_normal_angles import PRECISION, _sincos
from .native_surface_crossing import GapSlabEnclosure


def _v(values):
    return tuple(Decimal(float(x)) for x in values)


def _add(a,b):
    return tuple(x+y for x,y in zip(a,b))


def _scale(a,s):
    return tuple(x*s for x in a)


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def _poly(co,t,order=0):
    return sum((x*Decimal(factorial(i)//factorial(i-order))*(t**(i-order) if i>order else 1)
                for i,x in enumerate(co) if i>=order),Decimal(0))


class MaterialTimeNumerator:
    def __init__(self,p,q,r,w,*,construction_scale=Decimal(1)):
        if not w.is_finite() or not Decimal(0)<=w<=Decimal(2*pi):
            raise ValueError("precise material numerator requires angular norm in [0,2*pi]")
        self.p,self.q,self.r,self.w=tuple(p),tuple(q),tuple(r),w
        self.scale=sum(abs(x) for co in (p,q,r) for x in co)
        # Include uncancelled construction operands, not only the small final
        # coefficients. Large common coordinates can otherwise mask roundoff.
        self.scale=max(self.scale,construction_scale)

    def value(self,t,order=0):
        with localcontext() as context:
            context.prec=PRECISION
            t=Decimal(float(t))
            sine,cosine=_sincos(self.w*t)
            cycles=(cosine,-sine,-cosine,sine)
            result=_poly(self.p,t,order)
            for j in range(order+1):
                factor=Decimal(comb(order,j))*(self.w**j if j else 1)
                result+=factor*(_poly(self.q,t,order-j)*cycles[j%4]
                    +_poly(self.r,t,order-j)*cycles[(j+3)%4])
            return result

    def enclosure(self,left,right):
        if not 0<=left<right<=1:
            raise ValueError('ordered material numerator time slab required')
        with localcontext() as context:
            context.prec=PRECISION
            middle=(left+right)/2
            h=max(Decimal(float(middle))-Decimal(float(left)),
                  Decimal(float(right))-Decimal(float(middle)))
            # |t|<=1: coefficient absolute sums bound each polynomial derivative.
            def bound(co,order):
                return sum((abs(x)*Decimal(factorial(i)//factorial(i-order))
                            for i,x in enumerate(co) if i>=order),Decimal(0))
            curvature=bound(self.p,2)+sum(Decimal(comb(2,j))*(self.w**j if j else 1)*
                (bound(self.q,2-j)+bound(self.r,2-j)) for j in range(3))
            f=self.value(middle);d=self.value(middle,1)
            # Bounded Decimal operations and the existing 80-term sine series.
            # Ten guard digits remain beyond this arithmetic allowance.
            error=Decimal('1e-40')*max(Decimal(1),self.scale)*(1+self.w)**4
            width=abs(d)*h+curvature*h*h/2+error
            dwidth=curvature*h+error
            return GapSlabEnclosure(float(np.nextafter(float(f-width),-np.inf)),
                float(np.nextafter(float(f+width),np.inf)),
                float(np.nextafter(float(d-dwidth),-np.inf)),
                float(np.nextafter(float(d+dwidth),np.inf)))

    def sign(self,t):
        with localcontext() as context:
            context.prec=PRECISION
            error=Decimal('1e-40')*max(Decimal(1),self.scale)*(1+self.w)**4
            value=self.value(t)
            return 1 if value>error else -1 if value < -error else 0


def certified_material_gap(pair,start,translations,angular,shell_increment,*,fraction=1.):
    """Exact signed material gap from the existing precise coefficients.

    The unnormalised precise numerator is divided by the positive current
    oriented face-area norm at the queried fraction, so the returned magnitude
    is the physical normal gap in metres with the certified sign. A certified
    sign of zero (including exact touches) refuses; the precise kernel keeps
    its own angular admission refusal. This is a scalar certificate only.
    """
    numerator=material_time_numerator(pair,start,translations,angular,shell_increment)
    if not numerator.sign(fraction):
        raise ValueError('certified material gap requires a resolved precise sign')
    from .native_surface_path import material_path_pose
    pose=material_path_pose(start,translations,angular,shell_increment,float(fraction))
    corners=pair.target.adapter.current_positions(pose.shell_displacement)
    first=np.asarray(corners[1])-np.asarray(corners[0])
    second=np.asarray(corners[2])-np.asarray(corners[0])
    area=float(np.linalg.norm(np.cross(first,second)))
    if not area>0.:
        raise ValueError('certified material gap requires a positive oriented face-area norm')
    return float(numerator.value(fraction))/area


def material_time_numerator(pair,start,translations,angular,shell_increment):
    """Construct coefficients from authoritative binary inputs, without fits."""
    with localcontext() as context:
        context.prec=PRECISION
        source=pair.source;cell=source.cell
        corners=pair.target.adapter.current_positions(start.shell_displacement)
        velocity=np.asarray(pair.target.adapter.work_map@shell_increment).reshape(3,3)
        operand_scale=max(Decimal(1),*(abs(Decimal(float(x))) for data in
            (start.positions,start.position_low,translations,angular,corners,velocity,
             start.cell_rotations,source.reference.nodal_triads) for x in np.asarray(data).ravel()),
             abs(Decimal(float(source.radius))))
        construction_scale=(1+operand_scale)**8
        corners=tuple(_v(row) for row in corners);velocity=tuple(_v(row) for row in velocity)
        e1=_add(corners[1],_scale(corners[0],-1));e2=_add(corners[2],_scale(corners[0],-1))
        v1=_add(velocity[1],_scale(velocity[0],-1));v2=_add(velocity[2],_scale(velocity[0],-1))
        c=(_cross(e1,e2),_add(_cross(v1,e2),_cross(e1,v2)),_cross(v1,v2))
        weight=Decimal(float(source.t))
        positions=tuple(_add(_v(row),_v(low)) for row,low in zip(start.positions,start.position_low))
        # Match the admitted material point, including its literal weights.
        # The point API permits a small sum-to-one tolerance; replacing it by
        # corner0 would erase that offset under a common normal translation.
        bary=_v(pair.target.barycentric)
        target_point=tuple(sum(bary[j]*corners[j][i] for j in range(3)) for i in range(3))
        target_rate=tuple(sum(bary[j]*velocity[j][i] for j in range(3)) for i in range(3))
        base=_add(_add(_scale(positions[cell],1-weight),_scale(positions[cell+1],weight)),_scale(target_point,-1))
        rate=_add(_add(_scale(_v(translations[cell]),1-weight),_scale(_v(translations[cell+1]),weight)),_scale(target_rate,-1))
        frame=tuple(_v(row) for row in start.cell_rotations[cell])
        reference=source.reference.nodal_triads[0]
        sine,cosine=_sincos(Decimal(float(source.angle)))
        local=tuple(Decimal(float(source.radius))*(cosine*Decimal(float(reference[i,1]))+
            sine*Decimal(float(reference[i,2]))) for i in range(3))
        arm=tuple(_dot(row,local) for row in frame)
        omega=_v(angular[cell]);w=_dot(omega,omega).sqrt()
        if w:
            axis=_scale(omega,1/w);parallel=_scale(axis,_dot(axis,arm))
            radial=_add(arm,_scale(parallel,-1));tangential=_cross(axis,arm)
        else:
            parallel=arm;radial=tangential=(Decimal(0),)*3
        base=_add(base,parallel)
        p=[Decimal(0)]*4
        for i,co in enumerate(c):
            p[i]+=_dot(co,base);p[i+1]+=_dot(co,rate)
        return MaterialTimeNumerator(p,tuple(_dot(co,radial) for co in c),
            tuple(_dot(co,tangential) for co in c),w,construction_scale=construction_scale)
