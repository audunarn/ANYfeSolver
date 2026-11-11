"""Bounded frozen-state unilateral direction; never a physical response/seed.

All supported force modes remain present. Coercivity tests use symmetric parts
ONLY for admission; every solve and final equation certificate uses actual input
matrices. No regularization, pivot flooring, pressure clipping or mode removal.
"""
from __future__ import annotations
import numpy as np
from anyfesolver.e4_pl_enriched import EnrichedProfileError,readonly
from .enriched_static import check_cancel


class ColdPredictorError(EnrichedProfileError):pass
class ColdPredictorDeclined(ColdPredictorError):pass


def _maximum(a):
    return float(np.max(np.abs(a)))if a.size else 0.


def _admit_coercive(matrix,label,epsilon):
    if not np.all(np.isfinite(matrix)):
        raise ColdPredictorDeclined(label+' nonfinite scaled matrix')
    size=_maximum(matrix)
    if _maximum(matrix-matrix.T)>epsilon*size:
        raise ColdPredictorDeclined(label+' nonconjugate frozen tangent')
    try:
        # Admission ONLY; never substitute this symmetric part into a solve.
        factor=np.linalg.cholesky((matrix+matrix.T)/2.)
    except np.linalg.LinAlgError as exc:
        raise ColdPredictorDeclined(label+' coercivity not established')from exc
    if not np.all(np.isfinite(factor)):
        raise ColdPredictorDeclined(label+' nonfinite coercivity factor')
    return float(np.min(np.diag(factor)))


def _active_solve_limit(m):
    return 8*m*m+8


def cold_unilateral_direction(hessian,gap_gradient,gap,residual,nodal_forces,
                             *,counts,cancel=lambda:False):
    """Return (d,p,certificate) for the root-adopted frozen unilateral model.

    H*d-G.T*(p-z)=-r; g+G*d>=0; p>=0; p*(g+G*d)=0.
    g is normalized physical gap in metres; p,z are nodal FORCES, not pressure.
    Caller supplies supported rows only and reinstates unsupported p=0.
    """
    check_cancel(cancel)
    h=readonly(hessian);gmat=readonly(gap_gradient)
    if h.ndim!=2 or h.shape[0]!=h.shape[1]or h.shape[0]==0:
        raise ColdPredictorError('nonempty square frozen Hessian required')
    n=h.shape[0]
    if gmat.ndim!=2 or gmat.shape[1]!=n or gmat.shape[0]==0:
        raise ColdPredictorError('all supported gap rows required')
    m=gmat.shape[0]
    gap=readonly(gap,(m,));r=readonly(residual,(n,));z=readonly(nodal_forces,(m,))
    if type(counts)is not dict or 'linear_solves'not in counts or type(counts['linear_solves'])is not int:
        raise ColdPredictorError('owned solve counter required')
    if np.any(z<0):
        raise ColdPredictorError('nonnegative current nodal forces required')
    epsilon=256.*np.finfo(float).eps*max(n,m,1)
    diagonal=np.diag(h)
    if np.any(diagonal<=0):
        raise ColdPredictorDeclined('positive frozen energy diagonal required')
    scale=1./np.sqrt(diagonal)
    hs=(scale[:,None]*h)*scale[None,:]
    h_coercivity=_admit_coercive(hs,'primal',epsilon)
    rhs=np.column_stack((-r-gmat.T @ z,gmat.T))
    scaled_rhs=scale[:,None]*rhs
    check_cancel(cancel)
    try:solved=np.linalg.solve(hs,scaled_rhs)
    except np.linalg.LinAlgError as exc:
        raise ColdPredictorDeclined('frozen primal solve failed')from exc
    counts['linear_solves']+=1
    solved=scale[:,None]*solved
    if not np.all(np.isfinite(solved)):
        raise ColdPredictorDeclined('nonfinite frozen primal solution')
    d0,influence=solved[:,0],solved[:,1:]
    dual=gmat @ influence;q=gap+gmat @ d0
    if not np.all(np.isfinite(dual))or not np.all(np.isfinite(q))or np.any(np.diag(dual)<=0):
        raise ColdPredictorDeclined('supported dual rank/coercivity not established')
    pressure_scale=1./np.sqrt(np.diag(dual))
    a=(pressure_scale[:,None]*dual)*pressure_scale[None,:]
    b=pressure_scale*q
    dual_coercivity=_admit_coercive(a,'dual',epsilon)
    if not np.all(np.isfinite(b)):
        raise ColdPredictorDeclined('nonfinite frozen dual load')
    # Lawson-Hanson-style nonnegative active-set path. Deterministic lowest
    # argmin and exact bound hits; bounded/refuse on cycle, never drop a mode.
    y=np.zeros(m);active=np.zeros(m,dtype=bool);seen={tuple(active)}
    pivot_limit=_active_solve_limit(m);steps=0
    tolerance=epsilon*_maximum(b)
    while True:
        check_cancel(cancel)
        w=a @ y+b
        if not np.all(np.isfinite(w)):
            raise ColdPredictorDeclined('nonfinite frozen dual residual')
        available=np.flatnonzero(~active)
        if not available.size or np.min(w[available])>=-tolerance:break
        chosen=int(available[np.argmin(w[available])]);active[chosen]=True
        while True:
            check_cancel(cancel);steps+=1
            if steps>pivot_limit:raise ColdPredictorDeclined('bounded frozen active-set exhausted')
            ix=np.flatnonzero(active);proposal=np.zeros(m)
            try:proposal[ix]=np.linalg.solve(a[np.ix_(ix,ix)],-b[ix])
            except np.linalg.LinAlgError as exc:
                raise ColdPredictorDeclined('frozen active solve failed')from exc
            counts['linear_solves']+=1
            if not np.all(np.isfinite(proposal)):
                raise ColdPredictorDeclined('nonfinite active force solution')
            if np.all(proposal[ix]>0):
                y=proposal;break
            bad=ix[proposal[ix]<=0]
            denominator=y[bad]-proposal[bad]
            zero=bad[denominator==0]
            if zero.size:
                active[zero]=False;y[zero]=0.
                if not np.any(active):break
                continue
            ratios=y[bad]/denominator
            fraction=float(np.min(ratios))
            if not 0.<=fraction<=1.:
                raise ColdPredictorDeclined('invalid frozen nonnegative bound step')
            y=y+fraction*(proposal-y)
            hit=bad[ratios==fraction]
            active[hit]=False;y[hit]=0.
            if np.any(y<0):
                raise ColdPredictorDeclined('negative frozen bound-step force')
            if not np.any(active):break
        key=tuple(active)
        if key in seen:raise ColdPredictorDeclined('frozen active-set cycle refused')
        seen.add(key)
    p=pressure_scale*y;d=d0+influence @ p
    check_cancel(cancel)
    actual_gap=gap+gmat @ d
    station=h @ d-gmat.T @ (p-z)+r
    if not all(np.all(np.isfinite(x))for x in(d,p,actual_gap,station)):
        raise ColdPredictorDeclined('nonfinite unilateral model certificate')
    # Dimensionless numerical certificates only, never physical acceptance.
    scaled_station=scale*station
    station_scale=max(_maximum(scale*(h @ d)),_maximum(scale*(gmat.T @ (p-z))),
                      _maximum(scale*r),np.finfo(float).tiny)
    actual_w=pressure_scale*actual_gap
    gap_scale=max(_maximum(b),_maximum(a @ y),np.finfo(float).tiny)
    if not all(np.isfinite(x)for x in(station_scale,gap_scale))or not np.all(np.isfinite(actual_w)):
        raise ColdPredictorDeclined('nonfinite scaled model certificate')
    station_error=_maximum(scaled_station)/station_scale
    gap_error=max(0.,-float(np.min(actual_w)))/gap_scale
    complementarity=_maximum(y*actual_w)/max(_maximum(y)*gap_scale,np.finfo(float).tiny)
    if (not all(np.isfinite(x)for x in(station_error,gap_error,complementarity))
            or np.any(p<0)or station_error>epsilon or gap_error>epsilon or complementarity>epsilon):
        raise ColdPredictorDeclined('frozen unilateral model certificate failed')
    certificate=dict(method='COLD_FROZEN_UNILATERAL_ACTIVE_SET_V1',supported_modes=m,
        primal_coordinates=n,active_modes=[int(i)for i in np.flatnonzero(p>0)],
        active_solves=steps,pivot_limit=pivot_limit,numerical_epsilon=epsilon,
        stationarity_relative=station_error,gap_relative=gap_error,
        complementarity_relative=complementarity,
        primal_coercivity_cholesky_min=h_coercivity,dual_coercivity_cholesky_min=dual_coercivity,
        actual_physical_acceptance=False)
    return readonly(d),readonly(p),certificate
