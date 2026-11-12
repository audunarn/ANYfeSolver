"""Bounded full-system iterative refinement of the preserved Schur factor.

Back-substitution can leave a tiny spurious resultant in an exactly unloaded
equilibrium row. Do not excuse it with an absolute floor: solve its full-system
defect with the same factor and require the original componentwise bound.
"""
from dataclasses import dataclass
from time import monotonic
import numpy as np
from scipy import linalg
from ._ge_beam3_fibre_schur_solver import ResultantSchurSolver as Unrefined, _numeric
from ._native_reference_modal import _owned
from ._ge_beam3_p5_seeded.core import sha

POLICY='GE_BEAM3_SCHUR_WITH_CHECKED_DENSE_MIXED_FALLBACK_DEVELOPMENT_V4'


@dataclass(frozen=True)
class LinearStep:
    increment: np.ndarray
    backward_error: float
    operator_sha256: str
    refinements: int
    full_mixed_fallback: bool
    production_qualified: bool=False


class RefinedResultantSchurSolver:
    __slots__=('base','identity','_fallback','_fallback_solves','_sealed')

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False): raise AttributeError('immutable refined native Schur capture')
        object.__setattr__(self,name,value)

    def __init__(self,layout,matrix):
        self.base=Unrefined(layout,matrix)
        self.identity=sha(dict(policy=POLICY,parent=self.base.identity,maximum_refinements=3,
            fallback='FULL_MIXED_LAPACK_PARTIAL_PIVOT_LU_WITH_UNCHANGED_COMPONENTWISE_CHECK'))
        self._fallback=None; self._fallback_solves=0
        self._sealed=True

    def _apply(self,columns):
        base=self.base; a=base.matrix; g=base.geometry
        reduced_rhs=columns[list(g)].copy(); locals_rhs=[]
        for block,(chol,_) in zip(base.blocks,base._local):
            c_inv_b=linalg.cho_solve((chol,True),columns[list(block)],check_finite=True)
            reduced_rhs+=a[np.ix_(g,block)]@c_inv_b; locals_rhs.append(c_inv_b)
        xg=base._factor.solve_many(reduced_rhs)
        x=np.zeros_like(columns); x[list(g)]=xg
        for block,(_,lift),c_inv_b in zip(base.blocks,base._local,locals_rhs):
            x[list(block)]=lift@xg-c_inv_b
        return x

    def solve(self,rhs):
        started=monotonic(); self.base.guard(); b=_numeric(rhs); a=self.base.matrix
        if b.ndim not in (1,2) or b.shape[0]!=len(a) or (b.ndim==2 and not 1<=b.shape[1]<=16):
            raise ValueError('one complete RHS or at most sixteen RHS columns')
        columns=b if b.ndim==2 else b[:,None]
        x=self._apply(columns) if self._fallback is None else self._full_apply(columns)
        free=list(self.base.layout.free)
        for refinements in range(4):
            self.base.guard()
            residual=columns-a@x
            scale=(abs(a)@abs(x)+abs(columns))[free]
            error=float(np.max(abs(residual[free])/np.maximum(np.finfo(float).tiny,scale)))
            if not np.isfinite(x).all() or not np.isfinite(error): raise ValueError('nonfinite full-system refinement')
            if error<=1e-11: break
            if refinements==3:
                # A homogeneous equilibrium row can retain relative error one
                # while its absolute defect tends to zero. Preserve the matrix
                # and check; use full mixed partial pivoting, never tolerance clipping.
                x=self._full_apply(columns)
                residual=columns-a@x
                scale=(abs(a)@abs(x)+abs(columns))[free]
                error=float(np.max(abs(residual[free])/np.maximum(np.finfo(float).tiny,scale)))
                if not np.isfinite(x).all() or not np.isfinite(error) or error>1e-11:
                    raise ValueError('full mixed fallback fails native Newton residual')
                break
            x+=self._apply(residual)
            if monotonic()-started>60.: raise RuntimeError('native full-system refinement deadline')
        if np.any(x[self.base.layout.fixed]): raise ValueError('fixed native increments changed')
        if monotonic()-started>60.: raise RuntimeError('native refined solve deadline')
        self.base.guard()
        return LinearStep(_owned(x if b.ndim==2 else x[:,0]),error,self.identity,refinements,self._fallback is not None)

    def _full_apply(self,columns):
        base=self.base; free=base.layout.free
        if self._fallback is None:
            lu,piv=linalg.lu_factor(base.matrix[np.ix_(free,free)],check_finite=True)
            if not np.isfinite(lu).all() or np.any(np.diag(lu)==0.): raise ValueError('full mixed fallback factorization failed')
            piv=np.frombuffer(piv.tobytes(),dtype=piv.dtype)
            object.__setattr__(self,'_fallback',(_owned(lu),piv))
        x=np.zeros_like(columns); x[free]=linalg.lu_solve(self._fallback,columns[free],check_finite=True)
        object.__setattr__(self,'_fallback_solves',self._fallback_solves+columns.shape[1])
        return x

    def diagnostics(self):
        data=self.base.diagnostics()
        return {**data,'policy':POLICY,'operator_sha256':self.identity,
                'parent_operator_sha256':self.base.identity,'maximum_refinements':3,
                'full_mixed_fallback':self._fallback is not None,
                'fallback_backend':None if self._fallback is None else 'lapack_dgetrf_dgetrs',
                'factorizations':data['factorizations']+(0 if self._fallback is None else 1),
                'solves':data['solves']+self._fallback_solves}
