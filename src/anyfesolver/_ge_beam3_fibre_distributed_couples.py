"""Private static reduction with spatial couples per reference arclength.

The physical field is U_cell R0(s). Its spatial virtual rotation is the cell
rotation variation, so distributed couples act on the internal cell rotations.
They have no general SO(3) potential. No public element/state route is changed.
"""
from dataclasses import dataclass
from decimal import localcontext
from math import fsum
from time import monotonic
import numpy as np
from scipy import linalg
from ._ge_beam3_retained_fibre import RetainedFibreOperator
from ._ge_beam3_fibre_cell import CellHistory
from ._ge_beam3_fibre_line_work import evaluate as line_work
from ._ge_beam3_p5.algebra import rotation, skew
from ._ge_beam3_p5.arrays import _array, _frames
from ._ge_beam3_p5.chart import exp_chart_terms
from ._ge_beam3_p5_seeded.core import sha
from ._native_reference_modal import _owned
from ._ge_beam3_effective_deadline import effective_deadline_seconds

POLICY = 'GE_BEAM3_REFERENCE_SPATIAL_DISTRIBUTED_COUPLE_STATIC_V1'


@dataclass(frozen=True)
class DistributedStaticTrial:
    rotations: np.ndarray
    resultants: np.ndarray
    residual: np.ndarray
    spatial_jacobian: np.ndarray
    internal_lift: np.ndarray
    full_residual: np.ndarray
    conservative_residual: np.ndarray
    conservative_hessian: np.ndarray
    full_spatial_jacobian: np.ndarray
    applied_couple: np.ndarray
    history: CellHistory
    conservative_part_value: float
    internal_error: float
    iterations: int
    input_sha256: str
    conservative_potential: bool = False
    conservative_spectral_authority: bool = False
    production_qualified: bool = False
    dynamic_reduction_authorized: bool = False


def cell_couple_load(operator, spatial_density):
    if type(operator) is not RetainedFibreOperator:
        raise ValueError('exact physical retained operator required')
    operator.guard()
    density = _array(spatial_density, (3,), 'spatial reference couple density')
    result = np.zeros(42)
    for cell in (0, 1):
        measure = fsum(row[3] for row in operator.stations if row[0] == cell)
        if not np.isfinite(measure) or measure <= 0.:
            raise ValueError('positive distributed couple reference measure')
        with np.errstate(over='ignore', invalid='ignore'):
            result[18+3*cell:21+3*cell] = measure*density
    with np.errstate(over='ignore', invalid='ignore'):
        norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or not np.isfinite(norm):
        raise ValueError('distributed couple load exceeds finite norm range')
    operator.guard()
    return _owned(result)


def spatial_jacobian(conservative_residual, conservative_hessian):
    """Derivative of Rc-G, where G is fixed in spatial residual components.

    Rc is the conservative material-minus-dead-line-work residual, NOT Rc-G.
    Hc is its Exp-chart Hessian. D G=0 for this reference spatial load policy.
    """
    residual = _array(conservative_residual, (42,), 'conservative residual')
    jacobian = _array(conservative_hessian, (42, 42), 'conservative Hessian')
    for start in (3, 9, 15, 18, 21):
        jacobian[start:start+3, start:start+3] -= .5*skew(residual[start:start+3])
    return _owned(jacobian)


def chart_pullback(force, jacobian, increments):
    """Pull back a general spatial condensed Jacobian, without symmetrizing."""
    force = _array(force, (18,), 'condensed spatial force')
    jacobian = _array(jacobian, (18, 18), 'condensed spatial Jacobian')
    increments = _array(increments, (3, 3), 'nodal rotation chart increments')
    chart = np.eye(18)
    extra = np.zeros((18, 18))
    for node, vector in enumerate(increments):
        start = 6*node+3
        a, da = exp_chart_terms(vector)
        chart[start:start+3, start:start+3] = a
        for k in range(3):
            extra[start:start+3, start+k] = da[:, :, k].T@force[start:start+3]
    result = chart.T@force
    tangent = chart.T@jacobian@chart+extra
    if not np.isfinite(result).all() or not np.isfinite(tangent).all():
        raise ValueError('nonfinite distributed couple chart response')
    return _owned(result), _owned(tangent), _owned(chart)


def solve_distributed_static(operator, positions, position_low, nodal_frames, *, origin,
                             initial_rotations, initial_resultants, spatial_line_force,
                             spatial_couple_density, check=None, max_iterations=24):
    if type(operator) is not RetainedFibreOperator or type(origin) is not CellHistory:
        raise ValueError('exact retained operator and explicit accepted cell origin required')
    if type(max_iterations) is not int or not 0 <= max_iterations <= 24:
        raise ValueError('bounded distributed static internal iterations')
    if check is not None and not callable(check):
        raise ValueError('callable distributed static boundary check')
    started = monotonic()
    operator.guard()
    with localcontext() as context:
        context.prec = 80
        operator.cell._origins(origin)
    x = _array(positions, (3, 3), 'distributed static positions')
    low = _array(position_low, (3, 3), 'distributed static low positions')
    q = _frames(nodal_frames, 3, 'distributed static nodal frames')
    u = _frames(initial_rotations, 2, 'distributed static seed rotations').copy()
    p = _array(initial_resultants, (18,), 'distributed static seed resultants').copy()
    load = _array(spatial_line_force, (3,), 'explicit spatial reference line force')
    density = _array(spatial_couple_density, (3,), 'explicit spatial reference couple density')
    applied = cell_couple_load(operator, density)
    identity = sha(dict(policy=POLICY, operator=operator.identity, positions=x, position_low=low,
                        nodal_frames=q, origin=origin, line_force=load, couple_density=density,
                        applied_couple=applied, initial_rotations=u, initial_resultants=p))
    length = float(np.linalg.norm(operator.reference.coordinates[-1]-operator.reference.coordinates[0]))
    if not np.isfinite(length) or length <= 0.:
        raise ValueError('positive distributed static characteristic length')
    scale = np.r_[np.full(12, length), np.ones(12)]

    def safe():
        if check is not None:
            check()
        operator.guard()
        if monotonic()-started > effective_deadline_seconds(60.):
            raise RuntimeError('distributed static internal boundary deadline')

    def evaluate(rotations, forces):
        safe()
        e = operator.evaluate(x, low, q, rotations, forces, origin=origin, check=safe)
        rc = e.residual.copy()
        h = e.hessian+e.hessian_low
        value = e.potential
        if np.any(load):
            w = line_work(operator.reference, x, low, rotations, load, order=operator.order)
            rc -= w.gradient
            h = h-w.hessian
            value -= w.value
        net = rc-applied
        error = float(np.linalg.norm(net[18:]/scale))
        if not np.isfinite(error):
            raise ValueError('nonfinite distributed static residual norm')
        return e, rc, h, net, float(value), error

    for iteration in range(max_iterations+1):
        e, rc, h, net, value, error = evaluate(u, p)
        jac = spatial_jacobian(rc, h)
        if error <= 1e-11:
            lift = -linalg.solve(jac[18:, 18:], jac[18:, :18], assume_a='gen', check_finite=True)
            condensed = jac[:18, :18]+jac[:18, 18:]@lift
            if not np.isfinite(condensed).all():
                raise ValueError('nonfinite distributed static Schur response')
            safe()
            return DistributedStaticTrial(_owned(u), _owned(p), _owned(net[:18]), _owned(condensed),
                _owned(lift), _owned(net), _owned(rc), _owned(h), _owned(jac), applied,
                e.history, value, error, iteration, identity)
        if iteration == max_iterations:
            raise ValueError('distributed static internal equilibrium iteration limit')
        factor = linalg.lu_factor(jac[18:, 18:], check_finite=True)
        step = linalg.lu_solve(factor, -net[18:], check_finite=True)
        norm = float(np.linalg.norm(step))
        if not np.isfinite(step).all() or not np.isfinite(norm) or norm == 0.:
            raise ValueError('invalid distributed static internal step')
        angle = max(np.linalg.norm(step[:3]), np.linalg.norm(step[3:6]))
        fraction = min(1., .45*np.pi/max(float(angle), np.finfo(float).tiny))
        for cut in range(9):
            safe()
            delta = step*(fraction*.5**cut)
            trial_u = np.array([rotation(delta[3*c:3*c+3])@u[c] for c in (0, 1)])
            trial_p = p+delta[6:]
            _, _, _, changed, _, metric = evaluate(trial_u, trial_p)
            merit = float(np.linalg.norm(linalg.lu_solve(factor, changed[18:], check_finite=True)))
            if not np.isfinite(merit):
                raise ValueError('nonfinite distributed static internal merit')
            if metric <= 1e-11 or merit < norm:
                u, p = trial_u, trial_p
                break
        else:
            raise ValueError('distributed static internal line search limit')
    raise AssertionError('unreachable distributed static state')
