"""Local B2 chain Hessians and moving nearest-owner interface contributions.

This differentiates the existing half-weighted nearest-distance line potential.
Unequal-work persistent or nontransverse active ties are not differentiable.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from fractions import Fraction
from decimal import Decimal,localcontext

from .beam_pair import _projection_work, _GAUSS_8, _GAUSS_16


def _rational_geometry(source,targets):
    return tuple(tuple(Fraction.from_float(float(x)) for x in point) for point in source),tuple(
        tuple(tuple(Fraction.from_float(float(x)) for x in point) for point in target) for target in targets)


def _dot(first,second): return sum((a*b for a,b in zip(first,second)),Fraction(0))


def _point_law(source,target,s,radius,source_is_a=True):
    source,(target,)=_rational_geometry(source,(target,))
    s=Fraction(s)
    point=tuple((1-s)*a+s*b for a,b in zip(*source))
    vector=tuple(b-a for a,b in zip(*target))
    offset=tuple(a-b for a,b in zip(point,target[0]))
    raw=_dot(offset,vector)/_dot(vector,vector)
    t=max(Fraction(0),min(Fraction(1),raw))
    separation=np.array([float(a-t*b) for a,b in zip(offset,vector)])
    return _projection_work((float(1-s),float(s)),(float(1-t),float(t)),separation,
        np.array([float(v) for v in vector]),0<raw<1,radius,source_is_a)


def _coefficients(source,target,s):
    va=tuple(b-a for a,b in zip(*source)); vb=tuple(b-a for a,b in zip(*target))
    offset=tuple(a-b for a,b in zip(source[0],target[0]))
    length2=_dot(vb,vb)
    if length2==0: raise ValueError('beam-chain target cell has zero length')
    t0,t1=_dot(offset,vb)/length2,_dot(va,vb)/length2
    raw=t0+t1*s
    if raw<=0: first,second=offset,va
    elif raw>=1: first,second=tuple(a-b for a,b in zip(offset,vb)),va
    else:
        first=tuple(a-t0*b for a,b in zip(offset,vb))
        second=tuple(a-t1*b for a,b in zip(va,vb))
    return (_dot(second,second),2*_dot(first,second),_dot(first,first)),t0,t1


def _value(coefficients,s):
    a,b,c=coefficients
    return (a*s+b)*s+c


def _roots(coefficients,lo,hi):
    """Exact sign and rational root enclosures around an 80-digit square root.

    q and c/q are formed rationally: adding a tiny radical to a large b must
    not collapse distinct roots before domain filtering. Unresolved enclosure
    overlap with a nonroot domain end is an explicit precision-limit failure.
    """
    coefficients=tuple(Fraction(v) for v in coefficients)
    leading=next((v for v in coefficients if v),None)
    if leading is None:return ()
    # Canonical rational coefficients make root midpoint identity invariant
    # under scaling, including genuine irrational three-feature ties.
    coefficients=tuple(v/leading for v in coefficients)
    a,b,c=coefficients
    if a==0:
        bounds=[] if b==0 else [(-c/b,-c/b)]
    else:
        discriminant=b*b-4*a*c
        if discriminant<0: return ()
        if discriminant==0: bounds=[(-b/(2*a),-b/(2*a))]
        else:
            with localcontext() as ctx:
                ctx.prec=80
                decimal=lambda v: Decimal(v.numerator)/Decimal(v.denominator)
                radical=decimal(discriminant).sqrt()
                exact=Fraction(radical)
                if exact*exact==discriminant:
                    lower=upper=exact
                else:
                    down,up=radical.next_minus(),radical.next_plus()
                    for _ in range(8):
                        lower,upper=Fraction(down),Fraction(up)
                        if lower*lower<=discriminant<=upper*upper: break
                        if lower*lower>discriminant: down=down.next_minus()
                        if upper*upper<discriminant: up=up.next_plus()
                    else: raise ValueError('beam-chain square-root enclosure budget exhausted')
            q0,q1=(-(b+(lower if b>=0 else -lower))/2,
                   -(b+(upper if b>=0 else -upper))/2)
            qlo,qhi=min(q0,q1),max(q0,q1)
            if qlo<=0<=qhi:
                raise ValueError('beam-chain root denominator is unresolved')
            bounds=[tuple(sorted((qlo/a,qhi/a))),tuple(sorted((c/qlo,c/qhi)))]
            bounds.sort()
            if bounds[0][1]>=bounds[1][0]:
                raise ValueError('beam-chain distinct root enclosures overlap at partition precision')
    result=[]
    for low,high in bounds:
        if high<=lo or low>=hi: continue
        if low<=lo<=high:
            if _value(coefficients,lo)==0: continue
            raise ValueError('beam-chain root enclosure overlaps a nonroot domain end')
        if low<=hi<=high:
            if _value(coefficients,hi)==0: continue
            raise ValueError('beam-chain root enclosure overlaps a nonroot domain end')
        result.append((low+high)/2)
    return tuple(sorted(result))


def _windows(source,targets,target_ids,radius):
    """Exact B2 branch polynomials avoid spurious roots at material seams."""
    source,targets=_rational_geometry(source,targets)
    projection={Fraction(0),Fraction(1)}
    for target in targets:
        _,t0,t1=_coefficients(source,target,Fraction(1,2))
        if t1:
            projection.update(root for root in (-t0/t1,(1-t0)/t1) if 0<root<1)
    projection=sorted(projection)
    cuts=set(projection)
    for lo,hi in zip(projection[:-1],projection[1:]):
        coefficients=[_coefficients(source,t,(lo+hi)/2)[0] for t in targets]
        for i,first in enumerate(coefficients):
            for second in coefficients[i+1:]:
                cuts.update(_roots(tuple(a-b for a,b in zip(first,second)),lo,hi))
    intervals=[];R=Fraction.from_float(float(radius))
    ordered=sorted(cuts)
    for lo,hi in zip(ordered[:-1],ordered[1:]):
        s=(lo+hi)/2
        coefficients=[_coefficients(source,t,s)[0] for t in targets]
        owner=min(range(len(targets)),key=lambda i:(_value(coefficients[i],s),target_ids[i]))
        a,b,c=coefficients[owner]; activation=(a,b,c-R*R)
        boundaries=[lo,*_roots(activation,lo,hi),hi]
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            mid=(left+right)/2
            if _value(activation,mid)<=0:
                intervals.append((owner,left,right))
    return tuple(intervals)


def _pair_dofs(source_cell,target_cell):
    return np.r_[source_cell.dofs.reshape(-1),target_cell.dofs.reshape(-1)]


def _mapped_gradients(source_cell,target_cells,local_gradients):
    lanes=[_pair_dofs(source_cell,cell) for cell in target_cells]
    union=np.unique(np.concatenate(lanes))
    gradients=[]
    for dofs,local in zip(lanes,local_gradients):
        result=np.zeros(len(union))
        np.add.at(result,np.searchsorted(union,dofs),local)
        gradients.append(result)
    return union,gradients


def _equal_work(first,second):
    return np.linalg.norm(first-second)<=512*np.finfo(float).eps*max(1.,np.linalg.norm(first),np.linalg.norm(second))


def _nearest_indices(source,targets,s,*,partition=False):
    """Exact squared-distance ownership, with registered irrational root cuts.

    A physical length tolerance can falsely tie nearby but distinct projections
    on a gently bent chain. Rational distances distinguish these features. At a
    partition cut, an irrational root is recognized by its verified enclosure
    midpoint identity rather than rounded distance equality.
    """
    source,targets=_rational_geometry(source,targets);s=Fraction(s)
    coefficients=[_coefficients(source,target,s)[0] for target in targets]
    values=[_value(poly,s) for poly in coefficients]
    winner=min(range(len(values)),key=values.__getitem__)
    cuts={Fraction(0),Fraction(1)}
    if partition:
        for target in targets:
            _,raw,slope=_coefficients(source,target,s)
            if slope:cuts.update(v for v in (-raw/slope,(1-raw)/slope) if 0<v<1)
        # Use the generating projection branch interval, including an exact
        # left projection cut. A nonroot enclosure overlap still fails closed.
        lo=max(v for v in cuts if v<=s) if s<1 else max(v for v in cuts if v<s)
        hi=min(v for v in cuts if v>s) if s<1 else Fraction(1)
    tied=[]
    for index,value in enumerate(values):
        if value==values[winner]:tied.append(index)
        elif partition:
            difference=tuple(a-b for a,b in zip(coefficients[index],coefficients[winner]))
            if s in _roots(difference,lo,hi):tied.append(index)
    return tuple(tied)


def _check_owner(source_cell,source,target_cells,targets,owner,s,radius):
    """Resolve harmless shared-material ties; reject ambiguous active work."""
    gap,gradient,_=_point_law(source,targets[owner],s,radius,True)
    tied=_nearest_indices(source,targets,s)
    if owner not in tied:
        raise ValueError('beam-chain owner partition is inconsistent with closest distance')
    if gap>=0.: return
    _check_projection_plateau(source,targets[owner],s)
    for index,target in enumerate(targets):
        if index==owner: continue
        other,other_gradient,_=_point_law(source,target,s,radius,True)
        if index in tied:
            _check_projection_plateau(source,target,s)
            _,(first,second)=_mapped_gradients(source_cell,
                (target_cells[owner],target_cells[index]),(gradient,other_gradient))
            if not _equal_work(first,second):
                raise ValueError('beam-chain active nearest-owner tie has unequal physical work')


def _check_projection_plateau(source,target,s):
    source,(target,)=_rational_geometry(source,(target,))
    _,raw0,slope=_coefficients(source,target,Fraction(s))
    raw=raw0+Fraction(s)*slope
    if slope==0 and raw in (0,1):
        raise ValueError('beam-chain active projection-clamp plateau is unresolved')


def _check_partition_knot(cell,source,target_cells,targets,windows,s,radius):
    evaluations=[_point_law(source,target,s,radius,True) for target in targets]
    gap=min(row[0] for row in evaluations)
    if gap>=0.: return
    tied=_nearest_indices(source,targets,s,partition=True)
    if len(tied)<2: return
    _,mapped=_mapped_gradients(cell,tuple(target_cells[i] for i in tied),
                               tuple(evaluations[i][1] for i in tied))
    representatives=[]
    for gradient in mapped:
        if not any(_equal_work(gradient,prior) for prior in representatives):
            representatives.append(gradient)
    if len(representatives)==1: return
    if len(representatives)>2:
        raise ValueError('beam-chain ambiguous active triple ownership')
    if s in (0.,1.):
        raise ValueError('beam-chain active domain-end ownership tie is unresolved')
    left=[owner for owner,lo,hi in windows if lo<s<=hi]
    right=[owner for owner,lo,hi in windows if lo<=s<hi]
    if len(left)!=1 or len(right)!=1:
        raise ValueError('beam-chain active tie has unresolved neighboring intervals')
    if left[0] not in tied or right[0] not in tied:
        raise ValueError('beam-chain knot owners are inconsistent with nearest distance')
    if _equal_work(mapped[tied.index(left[0])],mapped[tied.index(right[0])]):
        raise ValueError('beam-chain active repeated owner root is nontransverse')


def moving_owner_block(source_cell,source,target_cells,targets,left_owner,right_owner,s,radius,stiffness):
    """Half-direction moving-boundary Hessian on a transverse active switch.

    D=d_left-d_right increases through zero. With v=D_q, boundary motion
    b_q=-v/D_s adds -0.5*L*k*g/D_s * v v^T. Both target maps are required.
    """
    left_gap,left_gradient,_=_point_law(source,targets[left_owner],s,radius,True)
    right_gap,right_gradient,_=_point_law(source,targets[right_owner],s,radius,True)
    selected=(target_cells[left_owner],target_cells[right_owner])
    dofs,(first,second)=_mapped_gradients(source_cell,selected,(left_gradient,right_gradient))
    difference=first-second
    gap=.5*(left_gap+right_gap)
    zero=np.zeros((len(dofs),len(dofs)))
    if gap>=0.: return dofs,zero,False
    tolerance=512*np.finfo(float).eps*max(radius,source_cell.length,*(cell.length for cell in selected))
    if abs(left_gap-right_gap)>tolerance:
        raise ValueError('beam-chain moving ownership boundary is unresolved')
    # A third distinct physical work map at this distance is not one simple
    # two-owner interface. Deterministic cell ordering does not repair it.
    tied=_nearest_indices(source,targets,s,partition=True)
    if left_owner not in tied or right_owner not in tied:
        raise ValueError('beam-chain switch does not have nearest owners')
    for index,target in enumerate(targets):
        if index in (left_owner,right_owner): continue
        other,gradient,_=_point_law(source,target,s,radius,True)
        if index in tied:
            union,gradients=_mapped_gradients(source_cell,
                (*selected,target_cells[index]),(left_gradient,right_gradient,gradient))
            if not (_equal_work(gradients[0],gradients[2]) or _equal_work(gradients[1],gradients[2])):
                raise ValueError('beam-chain ambiguous active triple ownership')
    if _equal_work(first,second): return dofs,zero,False
    source_vector=source[1]-source[0]
    first_s=float((left_gradient[:3]+left_gradient[3:6])@source_vector)
    second_s=float((right_gradient[:3]+right_gradient[3:6])@source_vector)
    slope=first_s-second_s
    floor=512*np.finfo(float).eps*max(abs(first_s),abs(second_s),np.linalg.norm(source_vector))
    if slope<=floor:
        raise ValueError('beam-chain active owner switch is nontransverse or unresolved')
    return dofs,(-.5*source_cell.length*stiffness*gap/slope)*np.outer(difference,difference),True


def _integrate(manager,source,target,length,left,right):
    def quadrature(lo,hi,rule):
        energy=0.; force=np.zeros(12); block=np.zeros((12,12))
        half=(hi-lo)/2
        measure_scale=float(half)*length
        if measure_scale==0.:
            raise ValueError('beam-chain active panel measure is below floating work resolution')
        for node,weight in zip(*rule):
            s=(lo+hi)/2+half*Fraction.from_float(float(node))
            if not lo<s<hi:
                raise ValueError('beam-chain integration panel has no resolved interior coordinate')
            gap,gradient,hessian=_point_law(source,target,s,manager.radius,True)
            gap=min(gap,0.)
            if gap==0.: continue
            measure=measure_scale*float(weight)
            energy+=.5*manager.stiffness*gap*gap*measure
            force+=manager.stiffness*gap*gradient*measure
            block+=manager.stiffness*(np.outer(gradient,gradient)+gap*hessian)*measure
        return energy,force,block
    def refine(lo,hi,depth):
        low=quadrature(lo,hi,_GAUSS_8); high=quadrature(lo,hi,_GAUSS_16)
        scales=(max(abs(high[0]),1e-14*manager.stiffness*manager.radius**2*length),
            max(np.linalg.norm(high[1]),1e-14*manager.stiffness*manager.radius*length),
            max(np.linalg.norm(high[2]),1e-14*manager.stiffness*length))
        error=max(np.linalg.norm(np.asarray(a)-np.asarray(b))/scale
                  for a,b,scale in zip(high,low,scales))
        if error<=manager.settings.line_integration_relative_tolerance:
            return (*high,24,float(error))
        if depth>=manager.settings.maximum_line_subdivisions:
            raise ValueError('consistent beam-chain integration subdivision budget exhausted')
        mid=(lo+hi)/2
        first=refine(lo,mid,depth+1); second=refine(mid,hi,depth+1)
        return (*(a+b for a,b in zip(first[:4],second[:4])),max(first[4],second[4]))
    return refine(left,right,0)


def assemble_consistent_chain(manager,u,tangent):
    n=manager.model.mesh.dof_manager.total_dofs
    if u.size!=n or not np.all(np.isfinite(u)):
        raise ValueError('beam-chain pair displacement is invalid')
    positions=manager._positions(u)
    force=np.zeros(n); energy=0.; count=points=interfaces=0; error=0.
    rows=[];cols=[];values=[]
    def scatter(dofs,block):
        if not tangent: return
        r,c=np.meshgrid(dofs,dofs,indexing='ij')
        rows.extend(r.reshape(-1));cols.extend(c.reshape(-1));values.extend(block.reshape(-1))
    for direction in (0,1):
        sources=manager.segments[direction]; targets=manager.segments[1-direction]
        poses=positions[direction]; target_poses=positions[1-direction]
        target_ids=tuple(cell.element_id for cell in targets)
        for cell,source in zip(sources,poses):
            windows=_windows(source,target_poses,target_ids,manager.radius)
            count+=len(windows)
            for s in sorted({0.,1.,*(lo for _,lo,_ in windows),*(hi for _,_,hi in windows)}):
                _check_partition_knot(cell,source,targets,target_poses,windows,s,manager.radius)
            for owner,left,right in windows:
                _check_owner(cell,source,targets,target_poses,owner,(left+right)/2,manager.radius)
                e,f,k,p,err=_integrate(manager,source,target_poses[owner],cell.length,left,right)
                dofs=_pair_dofs(cell,targets[owner])
                np.add.at(force,dofs,.5*f); scatter(dofs,.5*k)
                energy+=.5*e; points+=p;error=max(error,err)
            for first,second in zip(windows[:-1],windows[1:]):
                if first[0]==second[0] or first[2]!=second[1]: continue
                dofs,block,nonzero=moving_owner_block(cell,source,targets,target_poses,
                    first[0],second[0],first[2],manager.radius,manager.stiffness)
                scatter(dofs,block);interfaces+=int(nonzero)
    matrix=sparse.coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr() if tangent else None
    return force,matrix,energy,count,points,error,interfaces
