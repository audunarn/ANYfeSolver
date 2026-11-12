"""Exceptional exact sparse active-load certificate, never a forward error claim."""
from fractions import Fraction
import math
import numpy as np
from ._b2_physical_linear_batch_dev import B2NumericalSolveFailure

MAX_TERM_VISITS = 200_000
MAX_CLOSURE_INDICES = 50_000
MAX_RATIONAL_BITS = 4096
_TINY = Fraction(float(np.finfo(float).tiny))
_MAX = Fraction(float(np.finfo(float).max))
_TOL = Fraction(1e-9)


class _Budget:
    def __init__(self):self.used = 0
    def term(self, value):
        self.used += 1
        if self.used > MAX_TERM_VISITS or max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_RATIONAL_BITS:
            raise B2NumericalSolveFailure('affine certificate sparse closure work limit')
        if abs(value)>_MAX:
            raise B2NumericalSolveFailure('affine certificate intermediate not representable')
        return value


def _upper(value):
    if value < 0 or value > _MAX:
        raise B2NumericalSolveFailure('affine certificate bound not representable')
    result = float(value)
    if Fraction(result) < value:result = np.nextafter(result, math.inf)
    if not math.isfinite(result):raise B2NumericalSolveFailure('affine certificate bound not representable')
    return result


def certify_affine(K, C, T, answer, applied, reconstructed, cache):
    """Return certification of net exact g=T.T*f only, with reconstruction bound.

    Cache only structural closure indices; coefficients are read each call.
    No rounded reduced KKT/RHS, dense global operator or force floor is used.
    """
    n, nq = T.shape
    if (K.shape!=(n,n) or C.shape[1]!=n or answer.shape != (nq+C.shape[0],) or applied.shape != (n,) or reconstructed.shape != (n,) or
        not all(np.isfinite(x).all() for x in (K.data,C.data,T.data,answer,applied,reconstructed))):
        raise B2NumericalSolveFailure('finite affine certificate inputs required')
    budget = _Budget();tr = T.tocsr();tc = T.tocsc();kr = K.tocsr();cc = C.tocsc()
    exact = lambda value: Fraction(float(value))
    g = {}
    for i in np.flatnonzero(applied):
        for p in range(tr.indptr[i], tr.indptr[i+1]):
            j = int(tr.indices[p]);g[j] = budget.term(g.get(j,Fraction())+budget.term(exact(tr.data[p])*exact(applied[i])))
    g = {j:v for j,v in g.items() if v}
    if not g:return False
    if any(abs(v)<_TINY or abs(v)>_MAX for v in g.values()):
        raise B2NumericalSolveFailure('affine net projected load not representable as normal binary64')
    active = tuple(sorted(g))
    # Closure construction visits only active T columns and their K/C rows.
    physical = set();needed = set()
    for j in active:
        for i in tc.indices[tc.indptr[j]:tc.indptr[j+1]]:
            physical.add(int(i))
            if len(physical)+len(active)>MAX_CLOSURE_INDICES:raise B2NumericalSolveFailure('affine certificate sparse closure work limit')
    for i in physical:
        for j in kr.indices[kr.indptr[i]:kr.indptr[i+1]]:
            needed.add(int(j))
            if len(physical)+len(needed)+len(active)>MAX_CLOSURE_INDICES:raise B2NumericalSolveFailure('affine certificate sparse closure work limit')
    needed.update(physical)
    count = len(physical)+len(needed)+len(active)
    visits = sum(tc.indptr[j+1]-tc.indptr[j] for j in active)
    visits += sum(kr.indptr[i+1]-kr.indptr[i]+cc.indptr[i+1]-cc.indptr[i] for i in physical)
    visits += sum(tr.indptr[i+1]-tr.indptr[i] for i in needed)
    if count>MAX_CLOSURE_INDICES or visits+budget.used>MAX_TERM_VISITS:
        raise B2NumericalSolveFailure('affine certificate sparse closure work limit')
    # Retain the most recent closure, not an unbounded load-pattern cache.
    closure = (active,tuple(sorted(physical)),tuple(sorted(needed)))
    if cache.get('closure')!=closure:cache['closure']=closure
    q=answer[:nq];lam=answer[nq:];u={};difference={}
    for i in closure[2]:
        value=Fraction()
        for p in range(tr.indptr[i],tr.indptr[i+1]):
            value=budget.term(value+budget.term(exact(tr.data[p])*exact(q[tr.indices[p]])))
        if abs(value)>_MAX:raise B2NumericalSolveFailure('exact affine reconstruction not representable')
        u[i]=value;difference[i]=abs(exact(reconstructed[i])-value)
    residual={};error={}
    for i in closure[1]:
        value=-exact(applied[i]);bound=Fraction()
        for p in range(kr.indptr[i],kr.indptr[i+1]):
            j=int(kr.indices[p]);a=exact(kr.data[p])
            value=budget.term(value+budget.term(a*u[j]));bound=budget.term(bound+budget.term(abs(a)*difference[j]))
        for p in range(cc.indptr[i],cc.indptr[i+1]):
            value=budget.term(value+budget.term(exact(cc.data[p])*exact(lam[cc.indices[p]])))
        residual[i]=value;error[i]=bound
    for j in active:
        value=Fraction();bound=Fraction()
        for p in range(tc.indptr[j],tc.indptr[j+1]):
            i=int(tc.indices[p]);a=exact(tc.data[p])
            value=budget.term(value+budget.term(a*residual[i]));bound=budget.term(bound+budget.term(abs(a)*error[i]))
        cap_exact=_TOL*abs(g[j]);cap=float(cap_exact)
        if Fraction(cap)>cap_exact:cap=np.nextafter(cap,0.)
        if cap<=0 or not math.isfinite(cap):raise B2NumericalSolveFailure('affine relative load cap underflow or overflow')
        if _upper(abs(value)+bound)>cap:return False
    return True
