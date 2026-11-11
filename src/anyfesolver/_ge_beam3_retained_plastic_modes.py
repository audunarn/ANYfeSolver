"""Private explicit-policy spectra at verified paired-plastic states.

Frozen plastic coordinates define an elastic perturbation assumption, not
continued yielding dynamics. The accepted-increment tangent is separately
labeled an operator diagnostic. Neither policy implies qualification.
"""
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from time import monotonic
from math import fsum
from hashlib import sha256
import json
import numpy as np
from scipy.linalg import block_diag
from anyfesolver._ge_beam3_retained_plastic_state import Context
from anyfesolver._ge_beam3_station_resultant_cell import cholesky, solve, ident, sub, mul, tr, mv, pair
from anyfesolver._ge_beam3_lifted_kinetic_factor import current_lifted_kinetic_factor
from anyfesolver._native_factor_chain_modes import solve_factor_chain_modes
from anyfesolver._native_reference_modal import _owned
from anyfesolver._ge_beam3_p5_seeded.core import sha
from anyfesolver.control import cancellation_safe_point

FROZEN = 'FROZEN_PLASTIC_COORDINATES_ELASTIC_PERTURBATION'
ALGORITHMIC = 'ACCEPTED_INCREMENT_ALGORITHMIC_OPERATOR_DIAGNOSTIC'


@dataclass(frozen=True)
class Packet:
    left: np.ndarray
    right: np.ndarray
    geometric: np.ndarray
    kinetic: np.ndarray
    free: tuple
    algebraic: tuple
    identity: str
    material_policy: str
    production_qualified: bool = False


def compliance_factor(probe, response, history, resultants, *, policy, check):
    """80-digit constitutive root; split root pairs remain unexpanded with J."""
    cell = probe.cell
    with localcontext() as ctx:
        ctx.prec = cell.digits
        c = [list(row) for row in cell.c0]
        if policy == ALGORITHMIC:
            if response['derivative_kind'] != 'CLASSICAL_SMOOTH_BRANCH': raise ValueError('yield-boundary spectral derivative is not unique')
            active = [i for i, x in enumerate(response['increments'][0]) if x != 0.]
            if active:
                ga = sub(cell.g, active, range(18))
                term = mul(tr(ga), solve(sub(cell.q, active, active), ga, check))
                c = [[x+y for x, y in zip(row, other)] for row, other in zip(c, term)]
            if [pair(row) for row in c] != [(row[0], row[1]) for row in response['hessian']]:
                raise ValueError('accepted-increment compliance differs from native tangent')
        elif policy != FROZEN: raise ValueError('explicit paired-plastic spectral material policy required')
        root = cholesky(c); factor = solve(root, ident(18), check)
        parts = [pair(row) for row in factor]
        left = np.column_stack(([row[0] for row in parts], [row[1] for row in parts]))
        # For frozen coordinates the accepted plastic state must remain a
        # stationary base; freezing cannot replace it with a virgin state.
        p = [D.from_float(float(x)) for x in resultants]
        plastic = [D.from_float(float(row[0]))+D.from_float(float(row[1])) for row in history]
        gradient = [x+y for x, y in zip(mv(cell.c0, p), mv(tr(cell.g), plastic))]
        gh, gl = pair(gradient)
        given_hi, given_lo = response['gradient']
        error = max(abs(fsum((a, b, -c, -d))) for a, b, c, d in zip(gh, gl, given_hi, given_lo))
        if error > 1e-11: raise ValueError('frozen-history perturbation base is not the accepted state')
        # Source congruence C^-1 = factor.T factor is checked before conversion.
        check(); inverse_error = max(abs(x-D(i == j)) for i, row in enumerate(mul(mul(tr(factor), factor), c)) for j, x in enumerate(row))
        if inverse_error > D('1e-35'): raise ValueError('constitutive inverse factor identity')
        return left


def _prepare(model, program, checkpoint, section_inertias, *, policy, cancellation_token=None):
    if policy not in (FROZEN, ALGORITHMIC): raise ValueError('explicit spectral policy required')
    cancellation_safe_point(cancellation_token, 'paired-plastic-modes.capture')
    started = monotonic(); context = Context(model, program)
    state, _ = context.restore(checkpoint); before = bytes(checkpoint)
    def check():
        cancellation_safe_point(cancellation_token, 'paired-plastic-modes.construct')
        if monotonic()-started > 120: raise ValueError('paired-plastic spectral construction deadline')
        context.guard()
    layout = context.layout; n = layout.nodal_count+6*len(context.probes)
    if n > 80: raise ValueError('small spectral correctness probe only')
    if type(section_inertias) is not dict or any(type(i) is not int for i in section_inertias) or set(section_inertias) != {i for i, _ in layout.elements}: raise ValueError('one explicit section inertia per element')
    inertias = {i:_owned(section_inertias[i]) for i in section_inertias}
    g = np.zeros((n, n)); lefts = []; rights = []; kinetics = []
    for index, ((eid, element), probe) in enumerate(zip(layout.elements, context.probes)):
        check(); mechanical = state.mechanical; nodes = layout.nodes[index]
        response = probe.evaluate(mechanical.positions[nodes], mechanical.position_low[nodes], mechanical.nodal_frames[nodes],
            mechanical.cell_rotations[index], mechanical.resultants[index], origins=state.origins[index].tolist())
        left = compliance_factor(probe, json.loads(response.material), state.histories[index], mechanical.resultants[index], policy=policy, check=check)
        slots = list(element.get_dof_mapping(model.mesh))+list(range(layout.nodal_count+6*index, layout.nodal_count+6*index+6))
        j = response.hessian[24:, :24]; right = np.zeros((36, n)); right[:, slots] = np.vstack((j, j))
        g[np.ix_(slots, slots)] += response.hessian[:24, :24]
        local = current_lifted_kinetic_factor(probe.reference, inertias[eid], mechanical.cell_rotations[index], order=element.core.order)
        kinetic = np.zeros((len(local), n)); kinetic[:, slots] = local
        lefts.append(left); rights.append(right); kinetics.append(kinetic)
    free = tuple(int(i) for i in layout.nodal_free)+tuple(range(layout.nodal_count, n))
    algebraic = tuple(i for i in free if i < layout.nodal_count and i % 6 >= 3)
    b = np.vstack(kinetics)
    if np.any(b[:, algebraic]): raise ValueError('nodal rotation traces must have zero inertia')
    left, right = block_diag(*lefts), np.vstack(rights)
    identity = sha(dict(checkpoint_sha256=sha256(before).hexdigest(), policy=policy,
        left=left, right=right, geometric=g, kinetic=b, free=free, algebraic=algebraic, inertias=inertias))
    check()
    return Packet(_owned(left), _owned(right), _owned(g), _owned(b), free, algebraic, identity, policy), check


@dataclass(frozen=True)
class Analysis:
    material_policy: str
    interpretation: str
    eigenvalues: np.ndarray
    full_modes: np.ndarray
    numerical_brackets: np.ndarray
    spectral_residual: float
    original_ritz_residual: float
    full_backward_residual: float
    operator_identity: str
    checkpoint_sha256: str
    mass_policy: str = 'CURRENT_LIFTED_KINETIC_FIELD_LINEARIZED_AT_REST'
    state_advanced: bool = False
    negative_eigenvalues_retained: bool = True
    certified_intervals: bool = False
    buckling_factor_authorized: bool = False
    production_qualified: bool = False


def solve_modes(model, program, checkpoint, section_inertias, *, material_policy,
                bounds, num_modes=6, root_width=1e-10, cancellation_token=None):
    """Squared-frequency/operator spectrum under an explicitly named policy."""
    packet, guard = _prepare(model, program, checkpoint, section_inertias,
        policy=material_policy, cancellation_token=cancellation_token)
    guard()
    modes = solve_factor_chain_modes(packet.left, packet.right, packet.geometric, packet.kinetic,
        packet.free, packet.algebraic, bounds=bounds, num_modes=num_modes, root_width=root_width,
        cancellation_token=cancellation_token)
    guard()
    interpretation = ('ELASTIC_PERTURBATIONS_WITH_PLASTIC_COORDINATES_FIXED' if material_policy == FROZEN
                      else 'LAST_INCREMENT_OPERATOR_ONLY_NOT_PHYSICAL_VIBRATION_AUTHORITY')
    return packet, Analysis(material_policy, interpretation, modes.eigenvalues, modes.full_modes,
        modes.numerical_brackets, modes.spectral_residual, modes.original_ritz_residual,
        modes.full_backward_residual, packet.identity, sha256(checkpoint).hexdigest())
