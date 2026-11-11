"""Private current-state physical-fibre spectra with explicit material policy.

This is an at-rest perturbation, not finite-velocity plastic dynamics. A last-
increment algorithmic tangent is an operator diagnostic, not vibration authority.
No state is advanced and no buckling load factor is inferred from a frequency.
"""
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from hashlib import sha256
from math import fsum
from time import monotonic
import json
import numpy as np
from scipy.linalg import block_diag

from ._ge_beam3_retained_fibre_state import Context
from ._ge_beam3_station_resultant_cell import zero, ident, mul, mv, tr, solve, cholesky, pair, dec
from ._ge_beam3_lifted_kinetic_factor import current_lifted_kinetic_factor
from ._native_factor_chain_modes import solve_factor_chain_modes
from ._native_reference_modal import _owned
from ._ge_beam3_p5_seeded.core import canonical, sha
from .control import cancellation_safe_point


FROZEN = 'FROZEN_PLASTIC_COORDINATES_ELASTIC_PERTURBATION'
ALGORITHMIC = 'ACCEPTED_INCREMENT_ALGORITHMIC_OPERATOR_DIAGNOSTIC'


def compliance_factor(probe, response, origin, history, resultants, *, policy, check):
    """Reconstruct station tangents before the cell Legendre Schur operation.

Keep a split inverse-compliance root separate from the geometric derivative J.
Forming a rounded dense condensed tangent first loses high-contrast information.
"""
    if policy not in (FROZEN, ALGORITHMIC): raise ValueError('explicit fibre spectral policy required')
    cell = probe.cell; section = cell.section
    with localcontext() as context:
        context.prec = 80
        committed = cell._origins(history); cell._origins(origin)
        if (response['cell_identity'] != cell.identity or canonical(response['history']) != canonical(history)
                or canonical(response['origin']) != canonical(origin)):
            raise ValueError('spectral cell/history authority mismatch')
        if policy == ALGORITHMIC and response['derivative_kind'] != 'CLASSICAL_SMOOTH_BRANCH':
            raise ValueError('yield-boundary spectral derivative is not unique')
        a = [list(row) for row in cell._elastic] if policy == FROZEN else zero(cell.size, cell.size)
        load = [list(row) for row in cell.load_map]
        if len(response['stations']) != len(cell.stations): raise ValueError('complete spectral station inventory required')
        for j, ((_, weight, mapping, indices), fields) in enumerate(zip(cell.stations, response['stations'])):
            check()
            if policy == ALGORITHMIC:
                # Independent strain-entry section evaluation; do not trust a
                # claimed condensed compliance or a station modulus summary.
                # Rebuild in Decimal from individual fibre laws. Converting
                # the already summed high/low section tangent back to Decimal
                # can lose small coupled terms before a high-contrast inverse.
                strain = [dec(h)+dec(l) for h, l in zip(*fields['strain'])]
                if len(strain) != 6: raise ValueError('complete station strain required')
                modulus = [list(row) for row in section._background]
                for fibre, (z0, p0) in zip(section.fibres, section._origins(origin.stations[j])):
                    check(); b = section._map(fibre)
                    _, et, branch = fibre.curve._return(dec(fibre.young), sum((v*e for v, e in zip(b, strain)), D(0)), z0, p0)
                    if branch in ('YIELD_BOUNDARY', 'HARDENING_KNOT'):
                        raise ValueError('yield-boundary station derivative is not unique')
                    for row in range(6):
                        for column in range(6): modulus[row][column] += dec(fibre.area)*et*b[row]*b[column]
                block = mul(mul(tr(mapping), modulus), mapping)
                for i, row in enumerate(indices):
                    for k, column in enumerate(indices): a[row][column] += weight*block[i][k]
        cholesky(a)
        c = mul(tr(load), solve(a, load, check))
        if policy == ALGORITHMIC:
            supplied = [[dec(h)+dec(l) for h, l in zip(row[0], row[1])] for row in response['hessian']]
            if len(supplied) != 18 or any(len(row) != 18 for row in supplied): raise ValueError('complete cell compliance required')
            error = max(abs(x-y) for row, other in zip(c, supplied) for x, y in zip(row, other))
            scale = max(D(1), *(abs(x) for row in c for x in row))
            if error > D('1e-26')*scale: raise ValueError('reconstructed station compliance differs from cell tangent')
        # Holding *accepted* plastic strains fixed produces an affine elastic
        # problem. A virgin tangent is correct only together with this offset.
        p = [dec(float(v)) for v in resultants]
        if len(p) != 18: raise ValueError('eighteen retained resultants required')
        predictor = mv(load, p)
        for (_, weight, mapping, indices), station in zip(cell.stations, committed):
            stress_offset = [D(0)]*6
            for fibre, (z, _) in zip(section.fibres, station):
                for i, b in enumerate(section._map(fibre)):
                    stress_offset[i] += dec(fibre.area)*dec(fibre.young)*z*b
            for i, value in zip(indices, mv(tr(mapping), stress_offset)): predictor[i] += weight*value
        x = [row[0] for row in solve([list(row) for row in cell._elastic], [[v] for v in predictor], check)]
        gh, gl = pair(mv(tr(load), x)); supplied_hi, supplied_lo = response['gradient']
        if len(supplied_hi) != 18 or len(supplied_lo) != 18: raise ValueError('complete cell gradient required')
        error = max(abs(fsum((h, l, -s, -t))) for h, l, s, t in zip(gh, gl, supplied_hi, supplied_lo))
        if error > 1e-11: raise ValueError('frozen fibre history is not the accepted stationary base')
        root = cholesky(c); factor = solve(root, ident(18), check)
        inverse_error = max(abs(x-D(i == j)) for i, row in enumerate(mul(mul(tr(factor), factor), c)) for j, x in enumerate(row))
        if inverse_error > D('1e-35'): raise ValueError('fibre compliance inverse-factor identity')
        parts = [pair(row) for row in factor]; check()
        return _owned(np.column_stack(([row[0] for row in parts], [row[1] for row in parts])))


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


def prepare(model, program, checkpoint, section_inertias, *, material_policy, cancellation_token=None):
    if material_policy not in (FROZEN, ALGORITHMIC): raise ValueError('explicit fibre spectral policy required')
    started = monotonic()
    def local_check():
        cancellation_safe_point(cancellation_token, 'retained-fibre-modes.construct')
        if monotonic()-started > 120.: raise TimeoutError('fibre spectral construction deadline')
    local_check()
    if type(checkpoint) is not bytes: raise ValueError('immutable canonical fibre checkpoint required')
    context = Context(model, program, check=local_check); state, _ = context.restore(checkpoint)
    def guard():
        local_check(); context.guard()
    layout = context.layout; n = layout.nodal_count+6*len(context.probes)
    if n > 80: raise ValueError('small fibre spectral correctness model only')
    if (type(section_inertias) is not dict or any(type(i) is not int for i in section_inertias)
            or set(section_inertias) != {i for i, _ in layout.elements}):
        raise ValueError('one explicit section inertia per element')
    inertias = {i: _owned(value) for i, value in section_inertias.items()}
    g = np.zeros((n, n)); lefts = []; rights = []; kinetics = []
    mechanical = state.mechanical
    for index, ((eid, element), probe) in enumerate(zip(layout.elements, context.probes)):
        guard(); nodes = layout.nodes[index]
        response = probe.evaluate(mechanical.positions[nodes], mechanical.position_low[nodes],
            mechanical.nodal_frames[nodes], mechanical.cell_rotations[index], mechanical.resultants[index],
            origin=state.origins[index], check=local_check)
        left = compliance_factor(probe, json.loads(response.material), state.origins[index], state.histories[index],
            mechanical.resultants[index], policy=material_policy, check=local_check)
        slots = list(element.get_dof_mapping(model.mesh))+list(range(layout.nodal_count+6*index, layout.nodal_count+6*index+6))
        j = response.hessian[24:, :24]; right = np.zeros((36, n)); right[:, slots] = np.vstack((j, j))
        g[np.ix_(slots, slots)] += response.hessian[:24, :24]
        local = current_lifted_kinetic_factor(probe.reference, inertias[eid], mechanical.cell_rotations[index], order=probe.order)
        kinetic = np.zeros((len(local), n)); kinetic[:, slots] = local
        lefts.append(left); rights.append(right); kinetics.append(kinetic)
    nodal_free = tuple(i for i in range(layout.nodal_count) if i not in layout.fixed)
    free = nodal_free+tuple(range(layout.nodal_count, n))
    algebraic = tuple(i for i in nodal_free if i % 6 >= 3)
    left, right, kinetic = block_diag(*lefts), np.vstack(rights), np.vstack(kinetics)
    if np.any(kinetic[:, algebraic]): raise ValueError('nodal rotation traces must have zero inertia')
    identity = sha(dict(checkpoint_sha256=sha256(checkpoint).hexdigest(), material_policy=material_policy,
        operators=[probe.identity for probe in context.probes], left=left, right=right, geometric=g,
        kinetic=kinetic, free=free, algebraic=algebraic, inertias=inertias))
    guard()
    return Packet(_owned(left), _owned(right), _owned(g), _owned(kinetic), free, algebraic, identity, material_policy), guard


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
    packet, guard = prepare(model, program, checkpoint, section_inertias,
        material_policy=material_policy, cancellation_token=cancellation_token)
    modes = solve_factor_chain_modes(packet.left, packet.right, packet.geometric, packet.kinetic,
        packet.free, packet.algebraic, bounds=bounds, num_modes=num_modes, root_width=root_width,
        cancellation_token=cancellation_token)
    guard()
    interpretation = ('ELASTIC_PERTURBATIONS_WITH_PLASTIC_COORDINATES_FIXED' if material_policy == FROZEN
        else 'LAST_INCREMENT_OPERATOR_ONLY_NOT_PHYSICAL_VIBRATION_AUTHORITY')
    return packet, Analysis(material_policy, interpretation, modes.eigenvalues, modes.full_modes,
        modes.numerical_brackets, modes.spectral_residual, modes.original_ritz_residual,
        modes.full_backward_residual, packet.identity, sha256(checkpoint).hexdigest())
