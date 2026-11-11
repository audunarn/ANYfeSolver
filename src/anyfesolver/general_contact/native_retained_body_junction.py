"""Private complete connected-junction crossing inventory; no C1 snapping."""
from fractions import Fraction
import numpy as np
from scipy.optimize import brentq
from .beam_curve_pair import _curve,_bezier_controls,closest_on_b3
from .native_curved_pair_partition import (CurvedOwnershipPartition,_exact_curve,_dot,_cross,
    _sqrt_up,_sqrt_down,_up,_down,_stable_difference,_normal_box,_product_range)
from .native_retained_whole_pair import _exact_controls


def connected_junction_partition(search,left,right,*,remaining_cells):
    """Prove one transversal remote crossing by complete endpoint/cone inventory.

    Strictly decreasing endpoint stationarity phiR then phiL has one connected
    overlap; this route additionally verifies interior projections at its root. Positive target tangent cone/source crossing proves
    D_s>0 at EVERY distinct equal-distance root, hence at most one; endpoint
    signs prove existence. Root isolation is quantitative, not sampled ownership.
    Vertex cone plateaus and unsupported inventories retain the mature engine.
    """
    source=search.source;targets=search.targets;visited=0
    j,tl=_exact_curve(targets[0],1.);jr,tr=_exact_curve(targets[1],0.)
    if j!=jr:return None
    axis=tuple(b-a for a,b in zip(_exact_curve(targets[0],0.)[0],_exact_curve(targets[1],1.)[0]))
    axis_square=_dot(axis,axis)
    tangents=tuple(_exact_curve(target,t)[1] for target in targets for t in (0.,1.))
    axial=tuple(_dot(axis,t) for t in tangents)
    if axis_square<=0 or min(axial)<=0:return None
    kappa=_sqrt_up(max(_dot(_cross(axis,t),_cross(axis,t))/a**2 for t,a in zip(tangents,axial)))
    st=tuple(_exact_curve(source,t)[1] for t in (left,right));sa=min(_dot(axis,t) for t in st)
    if sa<=0:return None
    sperp=_sqrt_up(max(_dot(_cross(axis,t),_cross(axis,t))/axis_square for t in st))
    beta=_down((_sqrt_down(sa*sa/axis_square)-_up(kappa*sperp))/_sqrt_up(1.+_up(kappa*kappa)))
    if beta<=0:return None
    hull=_exact_controls(source,left,right);convexities=[]
    for target in targets:
        r,v=_exact_curve(target,0.);_,v1=_exact_curve(target,1.)
        a=tuple(y-x for x,y in zip(v,v1));aa=_dot(a,a);va=_dot(v,a)
        u=max(Fraction(0),min(Fraction(1),-va/aa)) if aa else Fraction(0)
        m=min(_dot(v,v)+_dot(tuple(x-y for x,y in zip(r,p)),a)+3*va*u+Fraction(3,2)*aa*u*u for p in hull)
        convexities.append(_down(2*m))
    if min(convexities)<=0:return None
    def phi(t,tangent):return _dot(tuple(x-y for x,y in zip(j,_exact_curve(source,t)[0])),tangent)
    # An omitted adjacent branch has the real shared vertex as its strict
    # global minimum. The winning cell contains that exact material point,
    # hence dominates it over the complete source hull. A positive-measure
    # zero KKT on the WINNING cell is rejected (no classical tangent).
    left_phis=[_dot(tuple(x-y for x,y in zip(j,p)),tl) for p in hull]
    right_phis=[_dot(tuple(x-y for x,y in zip(j,p)),tr) for p in hull]
    if min(right_phis)>0 and any(value!=0 for value in left_phis):
        return CurvedOwnershipPartition(((left,right,0),),(),1,())
    if max(left_phis)<0 and any(value!=0 for value in right_phis):
        return CurvedOwnershipPartition(((left,right,1),),(),1,())
    brackets=[]
    for tangent in (tl,tr):
        if not phi(left,tangent)>0>phi(right,tangent):return None
        lo,hi=left,right
        for _ in range(64):
            visited+=1
            if visited>remaining_cells:raise MemoryError('native body junction inventory cell budget exhausted')
            mid=.5*(lo+hi)
            if mid==lo or mid==hi:break
            if phi(mid,tangent)>0:lo=mid
            else:hi=mid
        else:return None
        brackets.append((lo,hi))
    if not brackets[1][1]<brackets[0][0]:return None
    first,last=brackets[1][1],brackets[0][0]
    def projections(s):
        result=tuple(closest_on_b3(_curve(source,s)[0],target) for target in targets)
        if any(not p.unique for p in result):raise ValueError('native body competing target projections')
        return result
    def difference(s):
        nonlocal visited
        visited+=1
        if visited>remaining_cells:raise MemoryError('native body junction root cell budget exhausted')
        return _stable_difference(source,targets,s,projections(s),convexities)[0]
    f,fe=_stable_difference(source,targets,first,projections(first),convexities)
    l,le=_stable_difference(source,targets,last,projections(last),convexities)
    if not f+fe<0<l-le:return None
    root=float(brentq(difference,first,last,xtol=1e-14,rtol=1e-14,maxiter=64))
    width=4e-14
    if not left<root-width<root+width<right:return None
    ps=projections(root)
    if any(not 0<p.minimizers[0]<1 for p in ps):return None
    point=_curve(source,root)[0];hull=_bezier_controls(source,root-width,root+width)
    reach=max(float(np.linalg.norm(p-point)) for p in hull)
    guard=2048*np.finfo(float).eps*max(1.,float(np.max(np.abs(source)))**2)
    normals=tuple(_normal_box(hull,point,target,p,reach,m,search.roundoff,guard) for target,p,m in zip(targets,ps,convexities))
    if any(n is None for n in normals):return None
    low=normals[0][0]-normals[1][1];high=normals[0][1]-normals[1][0]
    st=np.array((_curve(source,root-width)[1],_curve(source,root+width)[1]))
    products=[_product_range((low[i],high[i]),(min(st[:,i])-search.roundoff,max(st[:,i])+search.roundoff)) for i in range(3)]
    slope_low=sum(p[0] for p in products)-guard
    if slope_low<=guard:return None
    value,error=_stable_difference(source,targets,root,ps,convexities)
    uncertainty=(abs(value)+error)/slope_low+2e-14
    if not uncertainty<width:return None
    return CurvedOwnershipPartition(((left,root,0),(root,right,1)),
        ((root,0,1,uncertainty,slope_low,min(convexities)),),visited,())
