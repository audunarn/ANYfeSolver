"""Literal half-point target box preservation; DEVELOPMENT, not F03 acceptance.

No numerical imports at module load; unchanged actual compiler66 inventory,
independent repaired kappa/G, and existing sparse backend only. No averaging.
"""
from fractions import Fraction as F

from scripts import r8_f03_physical_interval_dev as physical
from scripts.r8_f03_finite_local_dev import ExactBudget, Unresolved
from scripts.r8_f03_finite_sparse_dev import Caps, directed, sparse_lu_homotopy

CAPS = dict(physical.CAPS)
REQUIRED_TARGETS = {'ideal_half', 'raw_stored_exact_shift', 'rounded_half', 'rounded_unshifted_exact_shift'}


def exact_targets(reference, stored, j, zero, half, budget):
    """Independently bind all four target meanings; don't reuse endpoint radii."""
    ideal, g = reference['ideal'], reference['G']
    targets = {name: {} for name in REQUIRED_TARGETS}
    keys = set(ideal) | set(g) | set(stored['entries']) | set(zero['center_entries']) | set(half['center_entries'])
    for i, k in keys:
        jj = budget.check(F(j[i])*F(j[k]))
        shift = budget.check(reference['kappa']*g.get((i, k), F())*jj/2)
        values = {
            'ideal_half': budget.check(ideal.get((i, k), F())*jj-shift),
            'raw_stored_exact_shift': budget.check(F(stored['entries'].get((i, k), 0.))*jj-shift),
            'rounded_half': F(half['center_entries'].get((i, k), 0.)),
            'rounded_unshifted_exact_shift': budget.check(F(zero['center_entries'].get((i, k), 0.))-shift)}
        for name, value in values.items():
            if value: targets[name][i, k] = value
    return {'dimension': reference['dim'], 'entries': targets,
        'provenance': {
            'ideal_half': 'exact independent repaired source Kideal minus exact kappa Gz/2; exact F(J) congruence',
            'raw_stored_exact_shift': 'exact F(actual normalized physical stored K) minus exact kappa Gz/2; exact F(J) congruence',
            'rounded_half': 'actual binary64 half subtraction and two J multiplications, unaveraged',
            'rounded_unshifted_exact_shift': 'exact F(actual rounded unshifted J-congruence) minus exact J(kappa Gz/2)J'}}


def validate_entries(entries, n):
    if type(entries) is not dict or len(entries) > CAPS['global_entries']:
        raise Unresolved('target sparse storage cap')
    for key, value in entries.items():
        if (type(key) is not tuple or len(key) != 2 or
                any(type(i) is not int or not 0 <= i < n for i in key) or type(value) is not F):
            raise Unresolved('target requires bounded exact rational scalar coordinates')
        if max(value.numerator.bit_length(),value.denominator.bit_length()) > CAPS['exact_bits']:
            raise Unresolved('target exact scalar bit cap')


def validate_box_storage(box):
    n = box.get('dimension')
    if type(n) is not int or not 0 < n <= CAPS['global_dimension']:
        raise Unresolved('box dimension cap')
    for field in ('center_entries','radius_entries'):
        values = box.get(field)
        if type(values) is not dict or len(values) > CAPS['global_entries']:
            raise Unresolved('missing bounded box storage')
        for key,value in values.items():
            if type(value) is not float: raise Unresolved('box storage requires actual binary64 scalar')
            physical.normal(value)
            validate_entries({key:F(value)},n)
            if field=='radius_entries' and value < 0: raise Unresolved('negative box radius')


def target_box(targets, center, budget, caps):
    n = targets.get('dimension')
    if type(n) is not int or not 0 < n <= CAPS['global_dimension']:
        raise Unresolved('target dimension cap')
    if (set(targets.get('entries', {})) != REQUIRED_TARGETS or
            set(targets.get('provenance', {})) != REQUIRED_TARGETS or
            any(type(s) is not str or not s for s in targets['provenance'].values())):
        raise Unresolved('missing complete explicit target/provenance binding')
    for entries in targets['entries'].values(): validate_entries(entries, n)
    if len(center) > CAPS['global_entries']:
        raise Unresolved('box center entry cap')
    exact_center = {key: F(value) for key, value in center.items()}
    validate_entries(exact_center, n)
    keys = set(exact_center)
    for entries in targets['entries'].values(): keys.update(entries)
    if len(keys) > CAPS['global_entries']: raise Unresolved('box union fill cap')
    radii = {}; maximum = F()
    for key in keys:
        difference = max(abs(entries.get(key, F())-exact_center.get(key, F()))
            for entries in targets['entries'].values())
        maximum = max(maximum, difference)
        radius = directed(budget.check(difference), True, caps)
        if radius: radii[key] = radius
    box = {'dimension': n, 'center_entries': dict(center), 'radius_entries': radii,
           'maximum_exact_difference': maximum, 'target_provenance': dict(targets['provenance'])}
    validate_box_storage(box)
    for entries in targets['entries'].values(): assert_enclosed(entries, box)
    return box


def assert_enclosed(entries, box):
    validate_box_storage(box)
    validate_entries(entries, box['dimension'])
    for key in set(entries) | set(box['center_entries']) | set(box['radius_entries']):
        radius = F(box['radius_entries'].get(key, 0.))
        if radius < 0 or abs(entries.get(key, F())-F(box['center_entries'].get(key, 0.))) > radius:
            raise Unresolved('target lies outside separately declared half-point box')


def signed_family_certificate(reference, j, box, budget):
    """Exact zero roles and uniform local18 Gershgorin D-box positivity."""
    nx, n = reference['nx'], reference['dim']
    if (nx, n) != (43, 66): raise Unresolved('literal supported signed role dimensions')
    validate_box_storage(box)
    if box['dimension'] != n or len(j) != n or any(type(v) is not float or physical.normal(v) <= 0 for v in j):
        raise Unresolved('literal box/reference/congruence role dimensions')
    # Both complementary/multiplier cross blocks and multiplier square block
    # must be exactly zero throughout the box, not approximately negligible.
    for i in range(nx+18, n):
        for k in range(nx, n):
            for key in ((i, k), (k, i)):
                if box['center_entries'].get(key, 0.) != 0 or box['radius_entries'].get(key, 0.) != 0:
                    raise Unresolved('p-lambda/lambda-lambda exact zero role violation')
    margins = []
    for i in range(18):
        row = []
        for k in range(18):
            key = (nx+i, nx+k)
            divisor = budget.check(F(j[nx+i])*F(j[nx+k]))
            if divisor <= 0: raise Unresolved('invertible positive congruence required')
            center = budget.check(-F(box['center_entries'].get(key, 0.))/divisor)
            radius = budget.check(F(box['radius_entries'].get(key, 0.))/divisor)
            row.append((center, radius))
        margin = budget.check(row[i][0]-row[i][1]-sum((abs(c)+r for k, (c,r) in enumerate(row) if k != i), F()))
        if margin <= 0: raise Unresolved('local18 signed D-box positivity not established')
        margins.append(margin)
    return {'roles': {'physical': nx, 'resultants': 18, 'multipliers': 5},
        'exact_p_lambda_and_lambda_lambda_zeros': True, 'D_uniform_positive': True,
        'D_method': 'exact physical-coordinate18 Gershgorin row lower bounds for symmetric admissible targets',
        'D_row_lower_bounds': margins, 'D_min_lower_bound': min(margins),
        'admissible_family': 'all exactly symmetric signed targets inside this explicit box; exact zero roles; unchanged supports/G/kappa',
        'known_ideal_half_inertia': reference['proof']['known_ideal_inertia'],
        'ideal_basis': reference['proof'], 'C_rank_basis': 'nonsingularity with exact multiplier zeros: C.T lambda=0 would give a pencil null vector',
        'inertia_basis': 'symmetric convex segment in certified nonsingular box from independent ideal half; D-positive signed Schur congruence'}


def admit_symmetric_target(entries, box, family):
    assert_enclosed(entries, box)
    if any(entries.get((k, i), F()) != value for (i, k), value in entries.items()):
        raise Unresolved('raw nonsymmetric target has only a nonsingularity claim')
    nx = family['roles']['physical']; n = box['dimension']
    if any(entries.get((i, k), F()) or entries.get((k, i), F())
            for i in range(nx+18, n) for k in range(nx, n)):
        raise Unresolved('symmetric target signed zero roles')
    if not family['D_uniform_positive']: raise Unresolved('missing local D positivity')
    return True


def assess():
    from anyfesolver.linalg import SparseSolverBackend, MatrixClass
    reference = physical.exact_reference()
    stored = physical.source_stored(reference)
    j = physical.congruence(reference)
    budget = ExactBudget(operation_cap=CAPS['exact_operations'], bit_cap=CAPS['exact_bits'])
    caps = Caps(dimension=CAPS['global_dimension'], fill=CAPS['factor_fill'],
        operations=CAPS['interval_operations'], bits=CAPS['interval_bits'])
    zero = physical.endpoint_intervals(reference, stored, j, F(), caps)
    half = physical.endpoint_intervals(reference, stored, j, F(1, 2), caps)
    targets = exact_targets(reference, stored, j, zero, half, budget)
    box = target_box(targets, half['center_entries'], budget, caps)
    family = signed_family_certificate(reference, j, box, budget)
    admit_symmetric_target(targets['entries']['ideal_half'], box, family)
    family['symmetric_intersection_nonempty'] = True
    family['nonempty_witness'] = 'separately enclosed exact independent ideal_half'
    center = physical.sparse_float(box['center_entries'], reference['dim'])
    radius = physical.sparse_float(box['radius_entries'], reference['dim'])
    handle = SparseSolverBackend().factorize(center, MatrixClass.GENERAL, options={'ordering': 'COLAMD'})
    if handle.status != 'ok' or handle._solver is None: raise Unresolved('existing half-point SuperLU failed')
    provenance = {'assembly': 'four independently bound exact target definitions; actual unchanged source/compiler arithmetic',
        'scatter': 'exact final sparse target-vs-stored discrepancy', 'scaling': 'explicit exact versus rounded J targets separately enclosed',
        'shift': 'stored-unshifted minus exact kappa Gz/2 explicitly enclosed, not inherited endpoint radius',
        'physical_reference': 'exact repaired ideal half point; raw stored skew retained; no averaging'}
    certificate = sparse_lu_homotopy(center, center, radius, radius, handle._solver,
        uncertainty_provenance=provenance, caps=caps)
    success = certificate['status'] == 'CERTIFIED_NONSINGULAR_DECLARED_INTERVAL'
    exact_operations = reference['exact_operations']+budget.operations
    if exact_operations > CAPS['exact_operations']: raise Unresolved('aggregate exact operation cap')
    return {'status': 'CERTIFIED_LITERAL_HALFPOINT_BOX' if success else 'UNRESOLVED',
        'reference': reference, 'stored': stored, 'congruence': j, 'targets': targets, 'box': box,
        'family': family, 'certificate': certificate, 'factor_diagnostics': handle.diagnostics(), '_factor': handle._solver,
        'target_nonsingularity': {name: success for name in REQUIRED_TARGETS},
        'symmetric_admissible_inertia': reference['proof']['known_ideal_inertia'] if success else None,
        'symmetric_admissible_C_rank': 5 if success else None,
        'literal_numerical_reserve_N_per_m': str(reference['kappa']/2) if success else None,
        'raw_stored_coercivity': None, 'raw_stored_inertia': None,
        'scientific_floor_comparison': None, 'physical_acceptance': None, 'qualification': False,
        'exact_operations': exact_operations,
        'exact_operations_components': {'source_prerequisites': reference['exact_operations'], 'target_box_D': budget.operations}}
