"""Successor numerical sign backend; original mechanics and audits unchanged."""
from math import isfinite
from time import monotonic
import numpy as np
from scipy import linalg
from ._native_factor_chain_modes import (
    mm, reassemble, shifted, inertia as floating_inertia, FactorChainModes,
    _reduce, _orthonormalize, _owned, reassemble_chain_exact_binary64,
)
from ._native_exact_shift_inertia import exact_inertia
from .control import cancellation_safe_point


def inertia(h, mass, shift, checkpoint=lambda: None, *, dimension_limit=64):
    checkpoint()
    try:
        return floating_inertia(h, mass, shift)
    except ValueError as error:
        if str(error) != 'unresolved shifted sign':
            raise
        _, negative, zeros = exact_inertia(h, mass, shift, checkpoint, dimension_limit=dimension_limit)
        if zeros:
            raise ValueError('unresolved shifted sign') from error
        return negative


def brackets(h,mass,bounds,count,width,checkpoint,*,dimension_limit=64):
    def sign(shift): return inertia(h,mass,shift,checkpoint,dimension_limit=dimension_limit)
    if sign(bounds[0])!=0 or sign(bounds[1])<count:
        raise ValueError('bounds do not enclose lowest requested roots')
    results=[]
    for index in range(count):
        lo,hi=bounds
        for _ in range(96):
            checkpoint()
            if hi-lo<=width: break
            mid=lo+(hi-lo)/2
            try: negative=sign(mid)
            except ValueError as exc:
                if str(exc)!='unresolved shifted sign': raise
                left,right=max(lo,mid-width/4),min(hi,mid+width/4)
                a,b=sign(left),sign(right)
                if a<=index<b:
                    lo,hi=left,right; break
                raise ValueError('shifted signs unresolved at requested root width') from exc
            if negative<=index: lo=mid
            else: hi=mid
        else: raise ValueError('signed search iteration limit')
        if hi-lo>width: raise ValueError('signed search width not achieved')
        results.append((lo,hi))
    return np.array(results)


def solve_factor_chain_modes(left_factor,right_factor,geometric,kinetic,free,algebraic,*,bounds,num_modes=6,root_width=1e-10,cancellation_token=None,exact_dimension_limit=64):
    start=monotonic()
    def checkpoint(stage='reassembly'):
        cancellation_safe_point(cancellation_token,'compensated_spectrum.'+stage)
        if monotonic()-start>600.: raise ValueError('compensated spectrum deadline')
    checkpoint('capture')
    if type(exact_dimension_limit) is not int or exact_dimension_limit not in (64,96,160):
        raise ValueError('explicit admitted exact-inertia dimension limit required')
    left,right,g,b=(_owned(x) for x in (left_factor,right_factor,geometric,kinetic))
    if (left.ndim!=2 or right.ndim!=2 or not 1<=left.shape[0]<=8192
            or not 1<=right.shape[0]<=512 or not 1<=right.shape[1]<=256
            or left.shape[1]!=right.shape[0]):
        raise ValueError('bounded matching owned factor chain')
    # Rounded expansion is only a preconditioner for a complete dynamic map.
    # The signed pencil and returned-vector audit use the original chain.
    f=mm(left,right,checkpoint)
    if (b.ndim!=2 or f.ndim!=2 or not 1<=b.shape[0]<=8192 or not 1<=b.shape[1]<=256
            or f.shape[1]!=b.shape[1]): raise ValueError('bounded matching factors')
    if (type(bounds) is not tuple or len(bounds)!=2 or any(type(v) not in (float,int) or not isfinite(v) for v in bounds)
            or bounds[0]>=bounds[1] or type(num_modes) is not int or num_modes<1
            or type(root_width) is not float or not isfinite(root_width) or root_width<=0.):
        raise ValueError('explicit finite root controls required')
    dense_mass=b.T@b
    _,_,mapping=_reduce(f,g,dense_mass,free,algebraic,checkpoint,b)
    h,mass=reassemble(left,right,g,b,mapping,checkpoint)
    if num_modes>len(h): raise ValueError('mode count exceeds dynamic coordinates')
    linalg.cholesky(mass)
    if len(h)>exact_dimension_limit: raise ValueError('dynamic pencil exceeds exact-inertia dimension limit')
    intervals=brackets(h,mass,bounds,num_modes,root_width,checkpoint,dimension_limit=exact_dimension_limit)
    coordinates=[]; values=[]; index=0
    while index<num_modes:
        end=index+1
        while end<num_modes and intervals[end,0]-intervals[end-1,1]<=8*root_width: end+=1
        if (inertia(h,mass,intervals[index,0],checkpoint,dimension_limit=exact_dimension_limit)!=index
                or inertia(h,mass,intervals[end-1,1],checkpoint,dimension_limit=exact_dimension_limit)!=end):
            raise ValueError('requested modes truncate a cluster')
        shifted_matrix,scale=shifted(h,mass,float(intervals[index,0]-2*root_width))
        roots,vectors=linalg.eigh(shifted_matrix); raw=vectors/scale[:,None]
        scores=np.abs(roots)/np.einsum('ij,ij->j',raw,mass@raw)
        basis=_orthonormalize(raw[:,np.argsort(scores,kind='stable')[:end-index]])
        lu=linalg.lu_factor(shifted_matrix)
        for _ in range(2):
            checkpoint('inverse')
            basis=_orthonormalize(linalg.lu_solve(lu,mm(mass,basis,checkpoint)/scale[:,None])/scale[:,None])
        # Complete-map pencil, not the earlier approximate material singular values.
        small_h=mm(basis.T,mm(h,basis,checkpoint),checkpoint); small_m=mm(basis.T,mm(mass,basis,checkpoint),checkpoint)
        rv,ru=linalg.eigh(small_h,small_m); actual=mm(basis,ru,checkpoint)
        if np.any(rv<intervals[index:end,0]-root_width) or np.any(rv>intervals[index:end,1]+root_width):
            raise ValueError('reassembled Ritz roots leave numerical brackets')
        coordinates.append(actual); values.extend(rv); index=end
    c=np.column_stack(coordinates); values=np.array(values)
    hc=mm(h,c,checkpoint); mc=mm(mass,c,checkpoint)
    residual=hc-mc*values
    # Preserve the separate material/geometric ACTION scale. Scaling by the
    # already cancelled tangent action asks for absolute force accuracy when
    # large material and prestress forces balance. Never use the norm of an
    # unrelated stiff direction as the denominator.
    strain=mm(left,mm(right,mapping,checkpoint),checkpoint); material_action=mm(strain.T,mm(strain,c,checkpoint),checkpoint)
    geometric_action=mm(mapping.T,mm(g,mm(mapping,c,checkpoint),checkpoint),checkpoint)
    scale=np.maximum(1.,np.maximum(np.linalg.norm(material_action,axis=0),
        np.maximum(np.linalg.norm(geometric_action,axis=0),np.linalg.norm(mc*values,axis=0))))
    if not np.isfinite(scale).all(): raise ValueError('nonfinite reassembled action scale')
    error=float(np.max(np.linalg.norm(residual,axis=0)/scale))
    if not isfinite(error) or error>1e-11: raise ValueError('reassembled action residual')
    modes=mm(mapping,c,checkpoint); speed=mm(b,modes,checkpoint)
    if np.linalg.norm(mm(speed.T,speed,checkpoint)-np.eye(num_modes))>1e-11: raise ValueError('physical modal mass normalization')
    # Audit returned physical vectors against the ORIGINAL signed bilinear
    # form. A normwise backward error divided by unrelated huge stiffness
    # cannot establish accuracy of weak eigenvalues. Each projected entry
    # uses its own mode-energy scale; no maximum stiffness enters this test.
    original_h,original_m=reassemble_chain_exact_binary64(left,right,g,b,modes,checkpoint)
    ritz_difference=original_h-original_m*values[None,:]
    ritz_scale=np.maximum(1.,np.sqrt(np.abs(values))[:,None]*np.sqrt(np.abs(values))[None,:])
    original_error=float(np.max(np.abs(ritz_difference)/ritz_scale))
    if not isfinite(original_error) or original_error>1e-11:
        raise ValueError('original signed bilinear Ritz identity failed')
    for j in range(num_modes):
        if modes[int(np.argmax(np.abs(modes[:,j]))),j]<0.: modes[:,j]*=-1.
    residual=(f.T@(f@modes)+g@modes-(dense_mass@modes)*values)[list(free)]
    scale=max(1.,(np.linalg.norm(f)**2+np.linalg.norm(g))*np.linalg.norm(modes),
        np.linalg.norm(dense_mass)*np.linalg.norm(modes*values))
    if not isfinite(scale): raise ValueError('nonfinite full backward scale')
    backward=float(np.linalg.norm(residual)/scale)
    if not isfinite(backward) or backward>1e-11: raise ValueError('full backward residual')
    checkpoint('output')
    return FactorChainModes(_owned(values),_owned(modes),_owned(intervals),error,backward,original_error,
        policy='PRIVATE_COMPLETE_MAP_EXACT_SHIFT_FALLBACK_MODES_V1')
