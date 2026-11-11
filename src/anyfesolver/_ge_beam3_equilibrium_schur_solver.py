"""Private equilibrium-first recovery after the unchanged geometric Schur solve.

Rather than subtracting two compliance actions to recover force increments,
solve all physical equilibrium rows, completed by selected compatibility rows.
No homogeneous force is clipped to zero. The full supplied matrix is checked.
"""
from dataclasses import dataclass
from time import monotonic
import numpy as np
from scipy import linalg
from ._ge_beam3_fibre_schur_solver import ResultantSchurSolver as GeometricFactor, _numeric
from ._native_reference_modal import _owned
from ._ge_beam3_p5_seeded.core import sha

POLICY='GE_BEAM3_EQUILIBRIUM_FIRST_FORCE_RECOVERY_DEVELOPMENT_V1'


@dataclass(frozen=True)
class LinearStep:
    increment: np.ndarray
    backward_error: float
    operator_sha256: str
    refinements: int
    full_mixed_fallback: bool=False
    production_qualified: bool=False


class EquilibriumSchurSolver:
    __slots__=('base','forces','recovery_rows','recovery_matrix','_recovery_lu','identity','_sealed')

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False): raise AttributeError('immutable equilibrium recovery capture')
        object.__setattr__(self,name,value)

    def __init__(self,layout,matrix):
        started=monotonic(); self.base=GeometricFactor(layout,matrix)
        g=self.base.geometry; self.forces=tuple(j for block in self.base.blocks for j in block)
        p=self.forces; a=self.base.matrix; missing=len(p)-len(g)
        if missing<0: raise ValueError('more free geometric equations than constitutive forces')
        rows=tuple(g)
        if missing:
            # Select a basis completion, NOT a scientific rank classification.
            # If B has full row rank, Q[:,g:] spans its nullspace. Rows of -C
            # restricted there supply the remaining independent equations.
            q,r=linalg.qr(a[np.ix_(g,p)].T,mode='full',check_finite=True)
            if np.any(np.diag(r[:len(g),:])==0.): raise ValueError('dependent physical equilibrium rows')
            null=q[:,len(g):]
            projected=a[np.ix_(p,p)]@null
            _,_,permutation=linalg.qr(projected.T,mode='economic',pivoting=True,check_finite=True)
            rows=(*rows,*(p[int(i)] for i in permutation[:missing]))
        self.recovery_rows=rows; self.recovery_matrix=_owned(a[np.ix_(rows,p)])
        lu,piv=linalg.lu_factor(self.recovery_matrix,check_finite=True)
        if not np.isfinite(lu).all() or np.any(np.diag(lu)==0.): raise ValueError('singular force recovery equations')
        self._recovery_lu=(_owned(lu),np.frombuffer(piv.tobytes(),dtype=piv.dtype))
        self.identity=sha(dict(policy=POLICY,parent=self.base.identity,force_slots=p,
            recovery_rows=rows,recovery_matrix=self.recovery_matrix,maximum_refinements=3))
        if monotonic()-started>60.: raise RuntimeError('equilibrium recovery construction deadline')
        self._sealed=True; self.base.guard()

    def _apply(self,columns):
        base=self.base; a=base.matrix; g=base.geometry
        reduced=columns[list(g)].copy()
        for block,(chol,_) in zip(base.blocks,base._local):
            reduced+=a[np.ix_(g,block)]@linalg.cho_solve((chol,True),columns[list(block)],check_finite=True)
        x=np.zeros_like(columns); x[list(g)]=base._factor.solve_many(reduced)
        rows=self.recovery_rows
        rhs=columns[list(rows)]-a[np.ix_(rows,g)]@x[list(g)]
        x[list(self.forces)]=linalg.lu_solve(self._recovery_lu,rhs,check_finite=True)
        return x

    def solve(self,rhs):
        self.base.guard(); started=monotonic(); b=_numeric(rhs); a=self.base.matrix
        if b.ndim not in (1,2) or b.shape[0]!=len(a) or (b.ndim==2 and not 1<=b.shape[1]<=16):
            raise ValueError('complete RHS and at most sixteen columns')
        columns=b if b.ndim==2 else b[:,None]; x=self._apply(columns); free=list(self.base.layout.free)
        for refinements in range(4):
            residual=columns-a@x
            scale=(abs(a)@abs(x)+abs(columns))[free]
            error=float(np.max(abs(residual[free])/np.maximum(np.finfo(float).tiny,scale)))
            if not np.isfinite(x).all() or not np.isfinite(error): raise ValueError('nonfinite equilibrium recovery')
            if error<=1e-11: break
            if refinements==3: raise ValueError('equilibrium recovery full residual failed')
            self.base.guard(); x+=self._apply(residual)
            if monotonic()-started>60.: raise RuntimeError('equilibrium recovery solve deadline')
        if np.any(x[self.base.layout.fixed]): raise ValueError('fixed increments changed')
        if monotonic()-started>60.: raise RuntimeError('equilibrium recovery solve deadline')
        self.base.guard()
        return LinearStep(_owned(x if b.ndim==2 else x[:,0]),error,self.identity,refinements)

    def diagnostics(self):
        data=self.base.diagnostics()
        return {**data,'policy':POLICY,'operator_sha256':self.identity,
            'parent_operator_sha256':self.base.identity,'recovery_rows':self.recovery_rows,
            'force_recovery_coordinates':len(self.forces),'compatibility_completion_rows':len(self.forces)-len(self.base.geometry),
            'geometric_factorizations':1,'force_recovery_factorizations':1,
            'factorizations':2,'full_mixed_fallback':False}
