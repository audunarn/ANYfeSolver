"""Shared exact radial bounds and physical-end work for static round beams.

The fixed linear map is differentiated literally. Bernstein arithmetic reuses
the existing contact utilities; no new differentiation engine is introduced.
"""
from fractions import Fraction
import heapq
import math

import numpy as np
from .capped_cylinder import _controls,_restrict,_product,_sqrt_bound,_fraction as _binary_fraction


def _fraction(value):
    """Retain exact derived radii/centres; convert only floating inputs."""
    return value if isinstance(value,Fraction) else _binary_fraction(value)


def squared_controls(controls):
    products=[_product(lane,lane) for lane in controls]
    return tuple(sum(values) for values in zip(*products))


def radial_inactive(controls,radius):
    q=squared_controls(controls);R=_fraction(radius)
    near=[Fraction(0) if min(lane)<=0<=max(lane) else min(abs(v) for v in lane) for lane in controls]
    if sum(v*v for v in near)>=R*R:return True
    # A temporal affine point path has a convex quadratic squared distance.
    # Its exact vertex proves opening even at a repeated grazing root.
    if len(q)==3:
        a,b,c=q[0]-2*q[1]+q[2],2*(q[1]-q[0]),q[0]
        minimum=min(q[0],q[-1])
        if a>0 and 0<-b<2*a:minimum=min(minimum,c-b*b/(4*a))
        return minimum>=R*R
    return min(q)>=R*R


def radial_cuts(controls,radius,proposals):
    """Open-side activation cuts; proposals alone never certify a boundary."""
    R=_fraction(radius)
    affine=(len(controls[0])==2 or all(lane[1]==(lane[0]+lane[-1])/2 for lane in controls))
    if affine:
        from .beam_chain_consistent import _roots
        lanes=tuple((lane[0],lane[-1]) for lane in controls)
        q=squared_controls(lanes)
        coefficients=(q[0]-2*q[1]+q[2],2*(q[1]-q[0]),q[0]-R*R)
        roots=_roots(coefficients,Fraction(0),Fraction(1))
        proposals=tuple(float(root) for root in roots)
    else:lanes=controls
    # Exact power coefficients are used only to resolve proposed scalar fronts.
    power=[Fraction(0)]*(3 if affine else 5)
    for lane in lanes:
        terms=(lane[0],lane[1]-lane[0]) if affine else (lane[0],2*(lane[1]-lane[0]),lane[0]-2*lane[1]+lane[2])
        for i,a in enumerate(terms):
            for j,b in enumerate(terms):power[i+j]+=a*b
    power[0]-=R*R
    if not affine and isinstance(radius,Fraction):
        # Preserve a small but nonzero activation constant. The historical
        # floating proposal builder trims coefficients relative to the largest
        # one, which can turn a resolvable endpoint interval into a zero root.
        # These are still proposals: exact brackets/refinement below and the
        # integration panel certificates remain authoritative.
        scale=max(abs(v) for v in power)
        if scale:
            polynomial=np.polynomial.Polynomial([float(v/scale) for v in power]).trim()
            proposals=tuple(sorted(set(float(root.real) for root in polynomial.roots()
                if abs(root.imag)<=1e-10 and 0.<root.real<1.)))
    def value(s,derivative=False):
        coefficients=tuple(i*v for i,v in enumerate(power))[1:] if derivative else power
        result=Fraction(0)
        for coefficient in reversed(coefficients):result=result*_fraction(s)+coefficient
        return result
    if not affine and isinstance(radius,Fraction):
        q=squared_controls(controls)
        differences=tuple(b-a for a,b in zip(q,q[1:]))
        monotone=(all(v>=0 for v in differences) or all(v<=0 for v in differences)) and any(differences)
        # A signed derivative Bernstein hull certifies at most one front.
        # Newton from the nearer endpoint supplies its tiny root without the
        # ill-conditioned companion matrix of an almost-affine quartic.
        if monotone and value(0)*value(1)<0:
            origin=0 if abs(value(0))<abs(value(1)) else 1
            slope=value(origin,True)
            if slope:
                candidate=float(Fraction(origin)-value(origin)/slope)
                if 0<candidate<1:proposals=(candidate,)
    cuts=(0.,*proposals,1.);resolved=[]
    for index,proposal in enumerate(proposals,1):
        left,right=(cuts[index-1]+proposal)/2,(proposal+cuts[index+1])/2
        gl,gr=value(left),value(right)
        if gl*gr>=0:resolved.append(proposal);continue
        current=proposal
        for _ in range(8):
            f=value(current);slope=value(current,True)
            if f==0:break
            if slope==0:raise ArithmeticError('radial activation root derivative is unresolved')
            candidate=float(_fraction(current)-f/slope)
            if not left<candidate<right:raise ArithmeticError('radial activation root escaped its verified bracket')
            if candidate==current:
                if f<0:candidate=math.nextafter(candidate,math.inf if slope>0 else -math.inf)
                if value(candidate)>=0:current=candidate;break
            current=candidate
        else:raise ArithmeticError('radial activation root refinement budget exhausted')
        if not 0<current<1 or value(current)<0:
            raise ArithmeticError('radial activation cut lacks an exact open-side certificate')
        resolved.append(current)
    if len(set(resolved))!=len(resolved):raise ArithmeticError('radial activation cuts collapse at work precision')
    return (0.,*sorted(resolved),1.)


def point_sign(point,radius):
    q=sum(lane[0]**2 for lane in point)-_fraction(radius)**2
    return (q>0)-(q<0)


def radial_point_work(point,linear_map,radius):
    q=sum(lane[0]**2 for lane in point)
    if q==0:raise ValueError('radial point contact has an undefined normal')
    length=(_sqrt_bound(q)+_sqrt_bound(q,True))/2
    R=_fraction(radius)
    gap=float((q-R*R)/(length+R))  # No subtraction cancellation at activation.
    unit=np.array([float(lane[0]/length) for lane in point])
    gradient=linear_map.T@unit
    hessian=linear_map.T@(np.eye(3)-np.outer(unit,unit))@linear_map/float(length)
    return gap,gradient,hessian


def end_controls(positions,center,linear_map,local,outward):
    lanes=_controls(positions,center,linear_map);degree=len(positions)-1
    index=0 if local==0 else -1
    point=tuple((lane[index],) for lane in lanes)
    derivative=tuple((int(outward)*degree*(lane[1]-lane[0] if local==0 else
        lane[-1]-lane[-2]),) for lane in lanes)
    return point,derivative


def orientation_controls(point,derivative):
    return tuple(sum(values) for values in zip(*[_product(p,v) for p,v in zip(point,derivative)]))


def ownership_path_admissible(first,last,radius,maximum_depth):
    point=tuple((a[0],b[0]) for a,b in zip(first[0],last[0]))
    derivative=tuple((a[0],b[0]) for a,b in zip(first[1],last[1]))
    def certify(left,right,depth):
        p=_restrict(point,left,right);v=_restrict(derivative,left,right)
        if radial_inactive(p,radius):return True
        q=orientation_controls(p,v)
        if min(q)>0 or max(q)<0:return True
        middle=(left+right)/2
        pm=_restrict(point,middle,middle);vm=_restrict(derivative,middle,middle)
        if point_sign(pm,radius)<0 and orientation_controls(pm,vm)[0]==0:return False
        if depth>=maximum_depth:return False
        return certify(left,middle,depth+1) and certify(middle,right,depth+1)
    return certify(Fraction(0),Fraction(1),0)


def curve_gap_lower(controls,radius,accuracy,maximum_depth,admit_above=None):
    """Conservative entire-curve lower bound with finite best-first refinement."""
    # Exact radius sums are legitimate enclosure inputs. Casting them back
    # through float can round down and invalidate the lower-bound guarantee.
    R=radius if isinstance(radius,Fraction) else _fraction(radius)
    budget=_fraction(accuracy)
    if budget<=0:raise ValueError('radial gap enclosure accuracy must be positive')
    def lower(lanes):
        return _sqrt_bound(max(Fraction(0),min(squared_controls(lanes))))-R
    def upper_point(lanes,s):
        p=_restrict(lanes,s,s)
        return _sqrt_bound(sum(lane[0]**2 for lane in p),True)-R
    upper=min(upper_point(controls,s) for s in (0.,.5,1.))
    heap=[(lower(controls),0,0,controls)];serial=0
    while True:
        low,depth,_,lanes=heap[0]
        if upper-low<=budget or (admit_above is not None and low>=_fraction(admit_above)):
            value=float(low)
            if _fraction(value)>low:value=math.nextafter(value,-math.inf)
            return value
        heapq.heappop(heap)
        if depth>=maximum_depth:raise MemoryError('radial entire-curve gap enclosure budget exhausted')
        for child in (_restrict(lanes,0.,.5),_restrict(lanes,.5,1.)):
            upper=min(upper,upper_point(child,.5))
            serial+=1;heapq.heappush(heap,(lower(child),depth+1,serial,child))


def relative_speed_upper(first,last):
    # Every Bernstein velocity is a convex combination of these controls.
    differences=tuple(tuple(b-a for a,b in zip(x,y)) for x,y in zip(first,last))
    squared=sum(max(abs(value) for value in lane)**2 for lane in differences)
    return _sqrt_bound(squared,True)


def regular_curve_path(first,last,maximum_depth):
    """Certify nonzero physical curve speed over the affine static path.

    The derivative is bilinear in material coordinate and path parameter.
    Its exact tensor Bernstein controls enclose every intermediate derivative;
    subdivision cannot replace an unresolved or singular curve with samples.
    """
    a=_controls(first,np.zeros(3),np.eye(3))
    b=_controls(last,np.zeros(3),np.eye(3))
    degree=len(first)-1
    a=tuple(tuple(degree*(y-x) for x,y in zip(lane,lane[1:])) for lane in a)
    b=tuple(tuple(degree*(y-x) for x,y in zip(lane,lane[1:])) for lane in b)
    if degree==1:
        a=tuple((lane[0],lane[0]) for lane in a)
        b=tuple((lane[0],lane[0]) for lane in b)
    threshold=_fraction(1e-15)**2
    def certify(left,right,depth):
        lower=sum((min((*l,*r)) if min((*l,*r))>0 else
            -max((*l,*r)) if max((*l,*r))<0 else Fraction(0))**2
            for l,r in zip(left,right))
        if lower>threshold:return True
        middle=tuple(tuple((x+y)/2 for x,y in zip(l,r)) for l,r in zip(left,right))
        midpoint=_restrict(middle,Fraction(1,2),Fraction(1,2))
        if sum(lane[0]**2 for lane in midpoint)<=threshold:return False
        if depth>=maximum_depth:return False
        time_width=max(abs(y-x) for l,r in zip(left,right) for x,y in zip(l,r))
        space_width=max((max(lane)-min(lane) for lane in (*left,*right)),default=Fraction(0))
        if time_width>=space_width:
            return certify(left,middle,depth+1) and certify(middle,right,depth+1)
        return (certify(_restrict(left,0.,.5),_restrict(right,0.,.5),depth+1)
            and certify(_restrict(left,.5,1.),_restrict(right,.5,1.),depth+1))
    return certify(a,b,0)
