"""Private scale-aware successor of the preserved exact factor-chain kernel.

The preserved factor reduction is a preconditioner, not eigenvalue authority.
Reassemble the ORIGINAL signed bilinear form and kinetic Gram matrix in its
complete dynamic map, then bracket the generalized pencil. No truncated Ritz
subspace, clipping, mass floor, mechanics, state transaction or qualification.
"""
from dataclasses import dataclass
from math import fsum, isfinite
import math
from fractions import Fraction
_fma = getattr(math, 'fma', None)
from time import monotonic
import numpy as np
from scipy import linalg
from anyfesolver._native_signed_kinetic_factor_modes import _reduce, _orthonormalize
from anyfesolver._native_reference_modal import _owned
from anyfesolver.control import cancellation_safe_point
from anyfesolver._dyadic_factor_chain import reassemble_chain_exact_binary64


def product_error(x, y, product):
    # Python <3.13 portability without changing package requirements. Exact
    # rational reconstruction rounds the same residual as fused multiply-add.
    # This is an arithmetic backend, never a mechanical/legacy fallback.
    if _fma is not None:
        return _fma(x, y, -product)
    return float(Fraction(x)*Fraction(y)-Fraction(product))


def dot_terms(a,b):
    terms=[]
    for x,y in zip(a,b):
        x,y=float(x),float(y); product=x*y
        if not isfinite(product): raise ValueError('nonfinite compensated product')
        if product==0. and x!=0. and y!=0.: raise ValueError('underflowed compensated product')
        terms.extend((product,product_error(x,y,product)))
    return terms


def mm(a,b,checkpoint=lambda:None):
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]: raise ValueError('matching matrix dimensions')
    rows=[]
    for row in a:
        checkpoint(); rows.append([fsum(dot_terms(row,col)) for col in b.T])
    return np.array(rows)


def reassemble(left,right,g,b,mapping,checkpoint=lambda:None):
    return reassemble_chain_exact_binary64(left,right,g,b,mapping,checkpoint)


def shifted(h,mass,shift):
    n=len(h)
    a=np.array([[fsum([float(h[i,j]), *dot_terms([-shift],[mass[i,j]])])
        for j in range(n)] for i in range(n)])
    size=np.abs(np.diag(a)); size=np.where(size==0.,np.max(np.abs(a),axis=1),size)
    scale=np.sqrt(size)
    if np.any(scale==0.): raise ValueError('unresolved shifted sign')
    return a/scale[:,None]/scale[None,:],scale


def inertia(h,mass,shift):
    a,_=shifted(h,mass,shift); values=linalg.eigvalsh(a)
    margin=64*np.finfo(float).eps*len(h)*max(1.,np.max(np.abs(values)))
    if np.min(np.abs(values))<=margin: raise ValueError('unresolved shifted sign')
    return int(np.count_nonzero(values<0.))


def brackets(h,mass,bounds,count,absolute,relative,checkpoint):
    if inertia(h,mass,bounds[0])!=0 or inertia(h,mass,bounds[1])<count:
        raise ValueError('bounds do not enclose lowest requested roots')
    results=[];widths=[]
    for index in range(count):
        lo,hi=bounds
        for _ in range(192):
            checkpoint()
            mid=lo+(hi-lo)/2
            width=absolute+relative*abs(mid)
            if hi-lo<=width:break
            try:negative=inertia(h,mass,mid)
            except ValueError as exc:
                if str(exc)!='unresolved shifted sign':raise
                left,right=max(lo,mid-width/4),min(hi,mid+width/4)
                a,b=inertia(h,mass,left),inertia(h,mass,right)
                if a>b:raise ValueError('nonmonotone relative shifted inertia')
                if a<=index<b:
                    lo,hi=left,right;break
                if index<a:
                    hi=left;continue
                if b<=index:
                    lo=right;continue
                raise ValueError('relative shifted signs unresolved') from exc
            if negative<=index:lo=mid
            else:hi=mid
        else:raise ValueError('relative signed search iteration limit')
        width=absolute+relative*abs(lo+(hi-lo)/2)
        if hi-lo>width:raise ValueError('relative signed search width not achieved')
        results.append((lo,hi));widths.append(width)
    return np.array(results),np.array(widths)


@dataclass(frozen=True)
class RelativeFactorChainModes:
    eigenvalues: np.ndarray
    full_modes: np.ndarray
    numerical_brackets: np.ndarray
    spectral_residual: float
    full_backward_residual: float
    original_ritz_residual: float
    policy: str='PRIVATE_COMPLETE_MAP_RELATIVE_FACTOR_CHAIN_SIGNED_MODES_V1'
    bracket_absolute_width: float=1e-10
    bracket_relative_width: float=1e-12
    bilinear_arithmetic: str='EXACT_DYADIC_FACTOR_CHAIN_CONGRUENCE_ROUNDED_ONCE'
    negative_eigenvalues_retained: bool=True
    certified_intervals: bool=False
    production_qualified: bool=False


def solve_relative_factor_chain_modes(left_factor,right_factor,geometric,kinetic,free,algebraic,*,bounds,num_modes=6,root_width=1e-10,relative_width=1e-12,cancellation_token=None):
    start=monotonic()
    def checkpoint(stage='reassembly'):
        cancellation_safe_point(cancellation_token,'compensated_spectrum.'+stage)
        if monotonic()-start>600.: raise ValueError('compensated spectrum deadline')
    checkpoint('capture')
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
            or type(root_width) is not float or not isfinite(root_width) or root_width<=0.
            or type(relative_width) is not float or not isfinite(relative_width) or not 0<relative_width<=1e-10):
        raise ValueError('explicit finite root controls required')
    dense_mass=b.T@b
    _,_,mapping=_reduce(f,g,dense_mass,free,algebraic,checkpoint,b)
    h,mass=reassemble(left,right,g,b,mapping,checkpoint)
    if num_modes>len(h): raise ValueError('mode count exceeds dynamic coordinates')
    linalg.cholesky(mass)
    intervals,widths=brackets(h,mass,bounds,num_modes,root_width,relative_width,checkpoint)
    coordinates=[]; values=[]; index=0
    while index<num_modes:
        end=index+1
        while end<num_modes and intervals[end,0]-intervals[end-1,1]<=8*max(widths[end],widths[end-1]): end+=1
        if inertia(h,mass,intervals[index,0])!=index or inertia(h,mass,intervals[end-1,1])!=end:
            raise ValueError('requested modes truncate a cluster')
        cluster_width=float(np.max(widths[index:end]))
        shifted_matrix,scale=shifted(h,mass,float(intervals[index,0]-2*cluster_width))
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
        if np.any(rv<intervals[index:end,0]-widths[index:end]) or np.any(rv>intervals[index:end,1]+widths[index:end]):
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
    return RelativeFactorChainModes(_owned(values),_owned(modes),_owned(intervals),error,backward,original_error,
        bracket_absolute_width=root_width,bracket_relative_width=relative_width)
