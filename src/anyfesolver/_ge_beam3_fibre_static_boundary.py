"""Private 18-external/24-internal STATIC boundary of the retained potential.

No state is committed. Physical cell rotations are stationary for this static
response only; this object supplies no mass, modal or dynamic reduction.
"""
from dataclasses import dataclass
from decimal import localcontext
from time import monotonic
import numpy as np
from scipy import linalg
from ._ge_beam3_retained_fibre import RetainedFibreOperator
from ._ge_beam3_fibre_cell import CellHistory
from ._ge_beam3_fibre_line_work import evaluate as line_work
from ._ge_beam3_p5.algebra import rotation
from ._ge_beam3_p5.arrays import _array,_frames
from ._ge_beam3_p5_seeded.core import sha
from ._native_reference_modal import _owned
from ._ge_beam3_effective_deadline import effective_deadline_seconds

POLICY='GE_BEAM3_RETAINED_FIBRE_STATIC_BOUNDARY_DEVELOPMENT_V1'


@dataclass(frozen=True)
class StaticTrial:
    rotations: np.ndarray
    resultants: np.ndarray
    residual: np.ndarray
    hessian: np.ndarray
    full_residual: np.ndarray
    full_hessian: np.ndarray
    history: CellHistory
    potential: float
    internal_error: float
    iterations: int
    input_sha256: str
    production_qualified: bool=False
    dynamic_reduction_authorized: bool=False


def _skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def solve_static(operator,positions,position_low,nodal_frames,*,origin,
                 initial_rotations,initial_resultants,spatial_line_force,
                 check=None,max_iterations=24):
    """Fixed-origin local stationarity, including any internal dead-load work."""
    if type(operator) is not RetainedFibreOperator or type(origin) is not CellHistory:
        raise ValueError('exact retained operator and explicit accepted cell origin required')
    if type(max_iterations) is not int or not 0<=max_iterations<=24:
        raise ValueError('bounded static internal iterations')
    if check is not None and not callable(check): raise ValueError('callable static boundary check')
    started=monotonic(); operator.guard()
    # Paired history normalization has the same 80-digit contract as the
    # physical section and existing retained driver, never caller precision.
    with localcontext() as context:
        context.prec=80
        operator.cell._origins(origin)
    x=_array(positions,(3,3),'static trial positions'); low=_array(position_low,(3,3),'static trial low positions')
    q=_frames(nodal_frames,3,'static trial nodal frames')
    u=_frames(initial_rotations,2,'static internal seed rotations').copy()
    p=_array(initial_resultants,(18,),'static internal seed forces').copy()
    load=_array(spatial_line_force,(3,),'explicit static spatial line force')
    identity=sha(dict(policy=POLICY,operator=operator.identity,positions=x,position_low=low,
        nodal_frames=q,origin=origin,line_force=load,initial_rotations=u,initial_resultants=p))
    length=float(np.linalg.norm(operator.reference.coordinates[-1]-operator.reference.coordinates[0]))
    if not np.isfinite(length) or length<=0.: raise ValueError('positive static characteristic length')
    scale=np.r_[np.full(12,length),np.ones(12)]

    def safe():
        if check is not None: check()
        operator.guard()
        if monotonic()-started>effective_deadline_seconds(60.): raise RuntimeError('static internal boundary deadline')

    def evaluate(rotations,forces):
        safe()
        e=operator.evaluate(x,low,q,rotations,forces,origin=origin,check=safe)
        r=e.residual.copy(); h=e.hessian+e.hessian_low; potential=e.potential
        if np.any(load):
            w=line_work(operator.reference,x,low,rotations,load,order=operator.order)
            r-=w.gradient; h=h-w.hessian; potential-=w.value
        return e,r,h,float(potential),float(np.linalg.norm(r[18:]/scale))

    for iteration in range(max_iterations+1):
        e,r,h,potential,error=evaluate(u,p)
        if error<=1e-11:
            # Exact static Schur identity; never infer a dynamic mass policy.
            lift=linalg.solve(h[18:,18:],h[18:,:18],assume_a='gen',check_finite=True)
            condensed=h[:18,:18]-h[:18,18:]@lift
            if not np.isfinite(condensed).all(): raise ValueError('nonfinite static Schur response')
            safe()
            return StaticTrial(_owned(u),_owned(p),_owned(r[:18]),_owned(condensed),
                _owned(r),_owned(h),e.history,potential,error,iteration,identity)
        if iteration==max_iterations: raise ValueError('static internal equilibrium iteration limit')
        jac=h[18:,18:].copy()
        for start in (0,3): jac[start:start+3,start:start+3]-=.5*_skew(r[18+start:21+start])
        factor=linalg.lu_factor(jac,check_finite=True)
        step=linalg.lu_solve(factor,-r[18:],check_finite=True); norm=float(np.linalg.norm(step))
        if not np.isfinite(step).all() or not np.isfinite(norm) or norm==0.: raise ValueError('invalid static internal step')
        angle=max(np.linalg.norm(step[:3]),np.linalg.norm(step[3:6]))
        fraction=min(1.,.45*np.pi/max(float(angle),np.finfo(float).tiny))
        for cut in range(9):
            safe(); delta=step*(fraction*.5**cut)
            trial_u=np.array([rotation(delta[3*c:3*c+3])@u[c] for c in (0,1)])
            trial_p=p+delta[6:]
            _,changed,_,_,metric=evaluate(trial_u,trial_p)
            merit=float(np.linalg.norm(linalg.lu_solve(factor,changed[18:],check_finite=True)))
            if not np.isfinite(merit): raise ValueError('nonfinite static internal merit')
            if metric<=1e-11 or merit<norm:
                u,p=trial_u,trial_p; break
        else: raise ValueError('static internal line search limit')
    raise AssertionError('unreachable static boundary state')
