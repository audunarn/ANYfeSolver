"""Private algebraic-force elimination for a captured native Newton matrix.

Eliminate only the 18 constitutive resultants per macrocell. Keep all free
nodal coordinates and six physical cell rotations. This is a linear solve,
not mass condensation, a new beam potential, or a production authorization.
The caller supplies the complete spatial residual Jacobian, including loads.
"""
from dataclasses import dataclass
from time import monotonic
import numpy as np
from scipy import linalg, sparse

from .assembly import build_constraint_transformation
from .linalg import MatrixClass, SparseSolverBackend, factorize
from ._ge_beam3_retained_fibre_state import Layout
from ._native_reference_modal import _owned
from ._ge_beam3_p5_seeded.core import sha

POLICY='GE_BEAM3_LOCAL_RESULTANT_SCHUR_NEWTON_DEVELOPMENT_V1'


def _numeric(value):
    a=np.asarray(value)
    if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError('finite real linear system; no booleans')
    return _owned(a)


@dataclass(frozen=True)
class LinearStep:
    increment: np.ndarray
    backward_error: float
    operator_sha256: str
    production_qualified: bool=False


class ResultantSchurSolver:
    """Reusable factor for multiple right-hand sides of ONE frozen Jacobian."""
    __slots__=('layout','matrix','geometry','blocks','reduced','identity','_capture',
               '_local','_factor','_sealed')

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False): raise AttributeError('immutable native Schur capture')
        object.__setattr__(self,name,value)

    def __init__(self,layout,matrix):
        started=monotonic()
        if type(layout) is not Layout: raise ValueError('exact native fibre layout required')
        layout.guard(); self.layout=layout
        n=layout.count
        if n!=layout.nodal_count+24*len(layout.elements) or not 1<=n<=512:
            raise ValueError('bounded complete retained coordinate layout')
        _,_,_,offset,nodal_free,info=build_constraint_transformation(
            sparse.eye(layout.nodal_count,format='csr'),np.zeros(layout.nodal_count),layout.model)
        expected_free=tuple(map(int,nodal_free))+tuple(range(layout.nodal_count,n))
        expected_fixed=tuple(i for i in range(layout.nodal_count) if i not in nodal_free)
        if info['slave_dofs'] or np.any(offset) or tuple(layout.free)!=expected_free or tuple(layout.fixed)!=expected_fixed:
            raise ValueError('supported homogeneous native increment map')
        self.blocks=tuple(tuple(range(layout.nodal_count+24*i+6,layout.nodal_count+24*i+24))
                          for i in range(len(layout.elements)))
        eliminated=tuple(j for block in self.blocks for j in block)
        if tuple(layout.compatibility)!=eliminated: raise ValueError('exact local constitutive-force slots')
        self.geometry=tuple(j for j in expected_free if j not in eliminated)
        if not self.geometry: raise ValueError('physical increment coordinates required')
        self.matrix=_numeric(matrix)
        if self.matrix.shape!=(n,n): raise ValueError('complete native Jacobian shape')
        self._capture=self._layout_identity()
        a=self.matrix; g=self.geometry; reduced=a[np.ix_(g,g)].copy(); local=[]
        for block in self.blocks:
            if monotonic()-started>60.: raise RuntimeError('local Schur construction deadline')
            others=tuple(j for j in eliminated if j not in block)
            if others and (np.any(a[np.ix_(block,others)]) or np.any(a[np.ix_(others,block)])):
                raise ValueError('constitutive forces are not element-local')
            c=-a[np.ix_(block,block)]; pg=a[np.ix_(block,g)]; gp=a[np.ix_(g,block)]
            # Never silently symmetrize, diagonalize or regularize a material block.
            if not np.array_equal(c,c.T) or not np.array_equal(pg,gp.T):
                raise ValueError('exact symmetric constitutive block/coupling required')
            chol=linalg.cholesky(c,lower=True,check_finite=True)
            lift=linalg.cho_solve((chol,True),pg,check_finite=True)
            reduced+=gp@lift
            local.append((_owned(chol),_owned(lift)))
        self._local=tuple(local); self.reduced=_owned(reduced)
        self.identity=sha(dict(policy=POLICY,layout=self._capture,matrix=self.matrix,
                               geometry=g,blocks=self.blocks,reduced=self.reduced))
        self._factor=factorize(sparse.csc_matrix(self.reduced),MatrixClass.GENERAL,
            backend=SparseSolverBackend(),signature=self.identity,options={'solver_threads':1})
        if self._factor.status!='ok': raise ValueError('native reduced matrix factorization failed: '+str(self._factor.failure_reason))
        if monotonic()-started>60.: raise RuntimeError('native Schur factorization deadline')
        self._sealed=True; self.guard()

    def _layout_identity(self):
        return sha(dict(identity=self.layout.identity,count=self.layout.count,
            free=self.layout.free,fixed=self.layout.fixed,compatibility=self.layout.compatibility,
            nodal_count=self.layout.nodal_count,slots=self.layout.slots))

    def guard(self):
        self.layout.guard()
        if self._layout_identity()!=self._capture: raise ValueError('native linearization layout changed')

    def solve(self,rhs):
        self.guard(); started=monotonic(); b=_numeric(rhs); n=len(self.matrix)
        if b.ndim not in (1,2) or b.shape[0]!=n or (b.ndim==2 and not 1<=b.shape[1]<=16):
            raise ValueError('one complete RHS or at most sixteen RHS columns')
        matrix_rhs=b.ndim==2
        columns=b if matrix_rhs else b[:,None]
        g=self.geometry; reduced_rhs=columns[list(g)].copy(); locals_rhs=[]
        for block,(chol,_) in zip(self.blocks,self._local):
            c_inv_b=linalg.cho_solve((chol,True),columns[list(block)],check_finite=True)
            reduced_rhs+=self.matrix[np.ix_(g,block)]@c_inv_b
            locals_rhs.append(c_inv_b)
        # Reuse the same ordinary ANYfeSolver factorization for Newton and merit RHS.
        xg=self._factor.solve_many(reduced_rhs)
        x=np.zeros_like(columns); x[list(g)]=xg
        for block,(_,lift),c_inv_b in zip(self.blocks,self._local,locals_rhs):
            x[list(block)]=lift@xg-c_inv_b
        free=list(self.layout.free)
        residual=(self.matrix@x-columns)[free]
        scale=(abs(self.matrix)@abs(x)+abs(columns))[free]
        error=float(np.max(abs(residual)/np.maximum(np.finfo(float).tiny,scale)))
        if not np.isfinite(x).all() or not np.isfinite(error) or error>1e-11:
            raise ValueError('recovered full native Newton residual failed')
        if np.any(x[self.layout.fixed]): raise ValueError('fixed native increments changed')
        if monotonic()-started>60.: raise RuntimeError('native Schur solve deadline')
        self.guard()
        return LinearStep(_owned(x if matrix_rhs else x[:,0]),error,self.identity)

    def diagnostics(self):
        self.guard()
        return dict(policy=POLICY,operator_sha256=self.identity,
            full_coordinates=len(self.matrix),supported_free_coordinates=len(self.layout.free),
            reduced_coordinates=len(self.geometry),eliminated_force_coordinates=sum(map(len,self.blocks)),
            retained_physical_cell_rotations=6*len(self.blocks),backend=self._factor.backend_name,
            matrix_class=self._factor.matrix_class.value,factorizations=self._factor.factorization_count,
            solves=self._factor.solve_count,production_qualified=False)
