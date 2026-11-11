"""Actual bounded source signed-pencil interval assessor; DEVELOPMENT ONLY.

No numerical imports occur at module import. Exact global operators are sparse
scalar dictionaries; dense blocks are local source42/35/21, physical24 or <=18.
The reference is independent symmetric ideal source mathematics; actual stored
compiler arithmetic is never averaged by this assessor to conceal skew.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math

from scripts.r8_f03_finite_local_dev import (ExactBudget, Unresolved, identity,
    isotropic_shell, multiply, solve, spd_inverse_bound, transpose, zero)
from scripts.r8_f03_finite_inventory_dev import literal_inventory
from scripts.r8_f03_finite_sparse_dev import Caps, directed, normal, sparse_lu_homotopy

CAPS = {'exact_operations': 3_000_000, 'exact_bits': 4096, 'global_dimension': 84,
        'global_entries': 10_000, 'factor_fill': 10_000, 'interval_operations': 1_000_000,
        'interval_bits': 4096, 'rectangle_cells': 2, 'native_members': 1}


@dataclass(frozen=True)
class Inputs:
    cells: int = 2
    a: F = F(1, 4)
    b: F = F(1, 10)
    ell: F = F(2, 5)
    e: F = F(210_000_000_000)
    nu: F = F(3, 10)
    thickness: F = F(7, 200)
    arm_z: F = F(-1, 20)
    width: F = F(1, 5)

    def validate(self):
        values = (self.a, self.b, self.ell, self.e, self.nu, self.thickness, self.arm_z, self.width)
        if (type(self.cells) is not int or not 1 <= self.cells <= 2 or
                any(type(x) is not F for x in values) or min(self.a, self.b, self.ell, self.e, self.thickness) <= 0 or
                self.width < 0 or max(self.a/self.b, self.b/self.a) > 4):
            raise Unresolved('literal rational rectangle/source-input contract')
        if self.arm_z != self.thickness/2-F(27, 400):
            raise Unresolved('original BL5 reference arm/top correspondence')
        if self != Inputs():
            raise Unresolved('only the exact declared two-cell/one-member default inventory is supported')


def original_native_section():
    """Preserved coefficient literal arithmetic, not retuned geometric stiffness."""
    e, nu = 210e9, .3
    g = e/(2*(1+nu))
    cn = (F(e*.01), F(g*.008), F(g*.008))
    cm = (F(g*1.6666666666666667e-5), F(e*8.333333333333333e-6), F(e*8.333333333333333e-6))
    return cn, cm


def diagonal(values):
    return [[F(v) if i == j else F() for j, v in enumerate(values)] for i in range(len(values))]


def add_exact(entries, i, j, value, budget):
    if value:
        entries[i, j] = budget.check(entries.get((i, j), F())+value)
        if len(entries) > CAPS['global_entries']:
            raise Unresolved('exact global sparse entry cap')


def add_stored(entries, i, j, value):
    value = normal(float(value))
    if value:
        entries[i, j] = normal(entries.get((i, j), 0.)+value)
        if len(entries) > CAPS['global_entries']:
            raise Unresolved('stored global sparse entry cap')


def native_exact_b_d(inputs, budget):
    """Original source ordering eta0,eta1,logs0L/R,logs1L/R; all24 x."""
    length, ell = 2*inputs.a*inputs.cells, inputs.ell
    half = length/2
    cn, cm = original_native_section()
    b, d = zero(18, 24), zero(18, 18)
    for cell in range(2):
        left, right, phi = 6*cell, 6*(cell+1), 18+3*cell
        for axis in range(3):
            b[3*cell+axis][left+axis] = -F(1)
            b[3*cell+axis][right+axis] = F(1)
            b[6+6*cell+axis][phi+axis] = 1/ell
            b[6+6*cell+axis][left+3+axis] = -1/ell
            b[9+6*cell+axis][right+3+axis] = 1/ell
            b[9+6*cell+axis][phi+axis] = -1/ell
            d[3*cell+axis][3*cell+axis] = half/cn[axis]
            for i in range(2):
                for j in range(2):
                    d[6+6*cell+3*i+axis][6+6*cell+3*j+axis] = half/F(3 if i == j else 6)/cm[axis]
        b[3*cell+1][phi+2] = -half/ell
        b[3*cell+2][phi+1] = half/ell
    # Source kernel certificate at this actual member: fixed six root physical
    # coordinates leave an exact18-column independent local B minor.
    minor = [[r[j] for j in range(6, 24)] for r in b]
    inv = solve(minor, identity(18), budget)
    if multiply(minor, inv, budget) != identity(18):
        raise Unresolved('actual native exact18-column work minor')
    return b, d, {'D_positive': spd_inverse_bound(d, budget), 'B_minor_columns': tuple(range(6, 24)),
                  'B_minor_identity': True, 'retained_physical_coordinates': 24, 'signed_block': '-D'}


def exact_reference(inputs=Inputs()):
    inputs.validate()
    budget = ExactBudget(operation_cap=CAPS['exact_operations'], bit_cap=CAPS['exact_bits'])
    c, drill = isotropic_shell(inputs.e, inputs.nu, inputs.thickness)
    cn, cm = original_native_section()
    inventory = literal_inventory(inputs.a, inputs.b, inputs.ell, inputs.cells, c, drill,
        diagonal(cn), diagonal(cm), arm_z=inputs.arm_z, width=inputs.width,
        shell_top=inputs.thickness/2, budget=budget)
    nx_full = 12*(inputs.cells+1)+24
    shell_count = nx_full-24
    dim_full = nx_full+18+5
    if dim_full > CAPS['global_dimension']:
        raise Unresolved('literal global sparse dimension cap')
    full = {}
    work = inventory['shell_work']
    local_k = multiply(multiply(transpose(work['rows_z']), work['H_N_per_m'], budget), work['rows_z'], budget)
    for cell in range(inputs.cells):
        nodes = (2*cell, 2*(cell+1), 2*(cell+1)+1, 2*cell+1)
        slots = [6*node+axis for node in nodes for axis in range(6)]
        for i, row in enumerate(local_k):
            for j, value in enumerate(row): add_exact(full, slots[i], slots[j], value, budget)
    b, d, native_proof = native_exact_b_d(inputs, budget)
    for i, row in enumerate(b):
        for j, value in enumerate(row):
            add_exact(full, nx_full+i, shell_count+j, value, budget)
            add_exact(full, shell_count+j, nx_full+i, value, budget)
    for i, row in enumerate(d):
        for j, value in enumerate(row): add_exact(full, nx_full+i, nx_full+j, -value, budget)
    # Frozen force-equivalent density lambda=T mu; barC=T.T C, not the
    # numerical owner residual scale and not an invented six-channel wrench.
    scales = (F(1), F(1), F(1), inputs.ell, inputs.ell)
    for channel, row in enumerate(inventory['P0_five_channel_actions_z']):
        for j, value in enumerate(row):
            add_exact(full, nx_full+18+channel, j, scales[channel]*value, budget)
            add_exact(full, j, nx_full+18+channel, scales[channel]*value, budget)
    fixed = set(range(12)) | set(range(shell_count, shell_count+5))
    physical_free = tuple(i for i in range(nx_full) if i not in fixed)
    free = physical_free+tuple(range(nx_full, dim_full))
    index = {old: new for new, old in enumerate(free)}
    ideal = {(index[i], index[j]): v for (i, j), v in full.items() if i in index and j in index and v}
    g = {(index[i], index[j]): v for i, j, v in inventory['Gz_sparse_exact'] if i in index and j in index and v}
    if any(ideal.get((j, i), F()) != value for (i, j), value in ideal.items()):
        raise Unresolved('independent ideal source pencil not exactly symmetric')
    nx = len(physical_free)
    # The local H proof + reconstruction + exact applicable five-row minor
    # implies constrained positive ideal S0. Source D>0 then fixes inertia,
    # independently of LU factorization or any floating candidate eigenvalue.
    proof = {'ideal_symmetric': True, 'native': native_proof,
             'literal_scaled_P0_row_rank': 5, 'P0_minor': inventory['P0_row_rank_certificate'],
             'supported_kernel': 'shell root removes all six; native root0..4 leaves yaw; actual P0 ty closes yaw',
             'known_ideal_inertia': (nx, 18+5, 0),
             'inertia_basis': 'positive local work/reconstruction and exact C minor; D positive; signed Schur congruence',
             'scientific_floor_comparison': None, 'growing_inventory_claim': False}
    return {'inputs': inputs, 'inventory': inventory, 'ideal': ideal, 'G': g, 'full_free': free,
            'physical_free': physical_free, 'nx_full': nx_full, 'shell_count': shell_count,
            'nx': nx, 'dim': len(free), 'D': d, 'B': b, 'proof': proof,
            'kappa': inventory['candidate_ideal_kappa_N_per_m'], 'exact_operations': budget.operations}


def source_stored(reference):
    """Unchanged actual reference-local compiler routes; no owner solve/factor.

    Preserve every selected source step, including any already-existing local
    shell pullback arithmetic. This adapter adds no averaging. The independent
    ideal dictionary encloses final compiler storage and its observed skew.
    """
    import numpy as np
    from scipy import sparse
    from anyfesolver import b3_ge
    from anyfesolver import _stiffened_shell_ge_experimental as compiler
    from anyfesolver._stiffened_shell_l1_assembly_experimental import ExperimentalBL5Line, compile_lines
    p = reference['inputs']; ell = float(p.ell)
    length = float(2*p.a*p.cells)
    coordinates = tuple((float(2*p.a*column), float(2*p.b*side), 0.)
                        for column in range(p.cells+1) for side in range(2))
    ids = tuple(range(1, len(coordinates)+1))
    cells = tuple((2*i+1, 2*i+3, 2*i+4, 2*i+2) for i in range(p.cells))
    panel = compiler.ShellPanel(ids, coordinates, cells, float(p.thickness), float(p.e), float(p.nu), 7850.)
    cn, cm = original_native_section()
    section = b3_ge.EllipsoidalGeneralizedSection(np.diag([float(x) for x in (*cn, *cm)]), np.eye(6), 1e20, 1e9)
    beam_definition = b3_ge.define_beam('b3-ge', 1, (101, 102, 103),
        np.array(((0., 0., .0675), (length/2, 0., .0675), (length, 0., .0675))),
        np.tile(np.eye(3), (3, 1, 1)), section, np.eye(6))
    model, shell_ids, xyz, shell_index, shell_parts = compiler._shell_reference(panel)
    beam_ids, beam_index, beam_nodes, beam_parts, beam_count, _ = compiler._beam_reference((beam_definition,))
    shell_count = reference['shell_count']
    _, shell_scale, beam_scale = compiler._reference_scales(xyz, beam_nodes, beam_parts, shell_count, beam_count)
    shell_r, shell_k, shell_error = compiler._assemble_shell(model, shell_parts, shell_count, shell_scale,
        compiler.Q4_HISTORICAL_POLICY)
    beam_r, beam_k, beam_error = compiler._assemble_beam(beam_parts, beam_nodes, beam_index, beam_count, beam_scale)
    # Compile the actual whole-member five-channel P0 row inventory. This
    # invokes the same local42 spatial beam response and line quadrature as
    # the real reference owner, without constructing/factorizing that owner.
    line = ExperimentalBL5Line(tuple(2*i+1 for i in range(p.cells+1)), ((0, 0), (0, 1)), multiplier_span=2)
    c, samples = compile_lines((line,), shell_ids=shell_ids, shell_xyz=xyz,
        shell_cells=panel.q4_elements, thickness=panel.thickness_m,
        beam_ids=beam_ids, beam_nodes=beam_nodes, beam_parts=beam_parts,
        shell_count=shell_count, beam_count=beam_count)
    k = sparse.block_diag((shell_k, beam_k), format='coo')
    if k.shape != (reference['nx_full']+18,)*2 or c.shape != (5, reference['nx_full']+18):
        raise Unresolved('actual source42/physical24/LOCAL18 compiler role mismatch')
    zscale = [1.]*k.shape[0]
    for node in range(shell_count//6):
        for axis in range(3, 6): zscale[6*node+axis] = 1./ell
    for node in range(3):
        for axis in range(3, 6): zscale[shell_count+6*node+axis] = 1./ell
    for i in range(shell_count+18, reference['nx_full']): zscale[i] = 1./ell
    full = {}
    for i, j, value in zip(k.row, k.col, k.data):
        add_stored(full, int(i), int(j), normal(normal(float(value)*zscale[i])*zscale[j]))
    c = c.tocoo()
    for channel, j, value in zip(c.row, c.col, c.data):
        value = normal(float(value)*(ell if channel >= 3 else 1.))
        value = normal(value*zscale[j])
        row = reference['nx_full']+18+int(channel)
        add_stored(full, row, int(j), value); add_stored(full, int(j), row, value)
    index = {old: new for new, old in enumerate(reference['full_free'])}
    entries = {(index[i], index[j]): v for (i, j), v in full.items() if i in index and j in index and v}
    return {'entries': entries, 'adapter': 'UNCHANGED_ACTUAL_REFERENCE_LOCAL_COMPILERS',
            'routes': ('_shell_reference/_assemble_shell historical elastic-reference pullback',
                       '_beam_reference/_assemble_beam full42 evaluate+spatial_jacobian',
                       'compile_lines actual five-channel BL5 P0/span2'),
            'source_q4_pullback_policy': compiler.Q4_HISTORICAL_POLICY,
            'source_shell_reference_residual_max': float(np.max(abs(shell_r), initial=0.)),
            'source_beam_reference_residual_max': float(np.max(abs(beam_r), initial=0.)),
            'source_local_scaled_reference_errors': (float(shell_error), float(beam_error)),
            'compiled_line_samples': len(samples), 'source_native_local_shape': (42, 42),
            'native_physical_coordinates': 24, 'native_resultants': 18,
            'owner_factorizations': 0, 'owner_solves': 0}


def congruence(reference):
    """Invertible computational J separate from scientific G; no norm fitting."""
    k0 = float(reference['inventory']['k0_N_per_m'])
    xscale = normal(1./math.sqrt(k0))
    diagonal_j = [xscale]*reference['nx']
    diagonal_j += [normal(1./math.sqrt(float(reference['D'][i][i]))) for i in range(18)]
    # Geometry-only row maxima in z coordinates, fixed before factorization.
    for channel, row in enumerate(reference['inventory']['P0_five_channel_actions_z']):
        scalar = reference['inputs'].ell if channel >= 3 else F(1)
        size = max(abs(v*scalar) for v in row)
        diagonal_j.append(normal(1./(float(size)*xscale)))
    if len(diagonal_j) != reference['dim'] or any(v <= 0 for v in diagonal_j):
        raise Unresolved('positive invertible congruence')
    return diagonal_j


def sparse_float(entries, dimension):
    from scipy import sparse
    if dimension > CAPS['global_dimension'] or len(entries) > CAPS['global_entries']:
        raise Unresolved('bounded sparse construction')
    items = sorted((k, v) for k, v in entries.items() if v)
    return sparse.coo_matrix(([normal(float(v)) for _, v in items],
        ([k[0] for k, _ in items], [k[1] for k, _ in items])), shape=(dimension, dimension), dtype='float64').tocsc()


def endpoint_intervals(reference, stored, j, t, caps):
    """Exact scalar ideal-vs-stored enclosure of every rounded operation.

    Exact reference assembly was independent. Comparing actual binary64 stored
    source entries to it includes local solve/quadrature/material error, sparse
    scatter, computational scaling, shift, and skew simultaneously. No fitted
    epsilon margin or symmetry repair substitutes for these exact differences.
    """
    ideal, g = reference['ideal'], reference['G']
    central, radius, exact = {}, {}, {}
    shift = t*reference['kappa']
    for (i, k) in set(ideal) | set(stored['entries']) | set(g):
        physical = ideal.get((i, k), F())-shift*g.get((i, k), F())
        exact[i, k] = physical*F(j[i])*F(j[k])
        # Execute and retain the rounded stored endpoint, never its transpose.
        value = normal(stored['entries'].get((i, k), 0.)-float(shift)*float(g.get((i, k), F())))
        value = normal(normal(value*j[i])*j[k])
        if value: central[i, k] = value
        difference = abs(exact[i, k]-F(value))
        upper = directed(difference, True, caps)
        if upper: radius[i, k] = upper
        if F(upper) < difference:
            raise Unresolved('endpoint outward enclosure failure')
    return {'center': sparse_float(central, reference['dim']), 'radius': sparse_float(radius, reference['dim']),
            'exact': exact, 'center_entries': central, 'radius_entries': radius}


def assess(inputs=Inputs()):
    from anyfesolver.linalg import SparseSolverBackend, MatrixClass
    reference = exact_reference(inputs)
    stored = source_stored(reference)
    j = congruence(reference)
    caps = Caps(dimension=CAPS['global_dimension'], fill=CAPS['factor_fill'],
                operations=CAPS['interval_operations'], bits=CAPS['interval_bits'])
    zero_endpoint = endpoint_intervals(reference, stored, j, F(), caps)
    half_endpoint = endpoint_intervals(reference, stored, j, F(1, 2), caps)
    midpoint = endpoint_intervals(reference, stored, j, F(1, 4), caps)
    handle = SparseSolverBackend().factorize(midpoint['center'], MatrixClass.GENERAL, options={'ordering': 'COLAMD'})
    if handle.status != 'ok' or handle._solver is None:
        raise Unresolved('existing SuperLU physical-pencil factorization failed: '+str(handle.failure_reason))
    provenance = {'assembly': 'independent exact source Q/F/E/H, native B/D and actual P0 work vs executed local source storage',
        'scatter': 'exact sparse ideal scalar accumulation versus actual binary64 source scatter entries',
        'scaling': 'exact F(binary64 J_i)*ideal*F(binary64 J_j) vs actual two binary64 multiplications',
        'shift': 'exact t*kappa*Gz ideal subtraction versus stored rounded shift, t in [0,.5]',
        'physical_reference': 'independent exactly symmetric source ideal; raw stored skew included entrywise without averaging'}
    result = sparse_lu_homotopy(zero_endpoint['center'], half_endpoint['center'], zero_endpoint['radius'], half_endpoint['radius'],
        handle._solver, uncertainty_provenance=provenance, caps=caps)
    # Same supplied LU and same actual source storage, collapsed to its chosen
    # midpoint. This isolates source uncertainty from endpoint shift width;
    # it is a point diagnostic, never the requested uniform half-margin.
    point_caps = Caps(dimension=CAPS['global_dimension'], fill=CAPS['factor_fill'],
        operations=CAPS['interval_operations'], bits=CAPS['interval_bits'])
    point_result = sparse_lu_homotopy(midpoint['center'], midpoint['center'], midpoint['radius'], midpoint['radius'],
        handle._solver, uncertainty_provenance=provenance, caps=point_caps)
    status = 'CERTIFIED_LITERAL_IDEAL_HALF_MARGIN' if result['status'] == 'CERTIFIED_NONSINGULAR_DECLARED_INTERVAL' else 'UNRESOLVED'
    skew = max((abs(v-stored['entries'].get((k, i), 0.)) for (i, k), v in stored['entries'].items()), default=0.)
    return {'status': status, 'qualification': False, 'reference': reference, 'stored': stored,
            'congruence': j, 'endpoint0': zero_endpoint, 'endpoint_half': half_endpoint,
            'certificate': result, 'midpoint_source_error_certificate': point_result,
            'midpoint_endpoint': midpoint, 'known_ideal_inertia': reference['proof']['known_ideal_inertia'],
            'source_stored_max_abs_skew': skew,
            '_factor': handle._solver, 'factor_diagnostics': handle.diagnostics(),
            'literal_ideal_kappa_N_per_m': str(reference['kappa']),
            'literal_preserved_ideal_margin_N_per_m': str(reference['kappa']/2) if status.startswith('CERTIFIED') else None,
            'scientific_floor_comparison': None, 'physical_acceptance': None,
            'scope': 'literal unprestressed elastic actual reference compiler only; original completion retained',
            'stored_nonsymmetric_coercivity_claim': False}
