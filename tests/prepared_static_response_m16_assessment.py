"""M16 certified response route assessment harness.

Applies the criteria frozen in docs/agent_plans/PREPARED_STATIC_IMPLEMENTATION_TASK.md
("M16 certified response route") before any M16 measurement. The reference is
canonical staged recovery at the route's own certified displacement u_c. No
criterion constant may be changed after observing results.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / 'tests', ROOT / 'scripts'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from anyfesolver import PatternCase, RecoveryConfig, prepare_static_response_evaluator
from anyfesolver.prepared_static import PreparedStaticEvaluator
from anyfesolver.prepared_static_envelopes import _numeric_fields
from anyfesolver.prepared_static_recovery import (_recover_owned_local, _derived_measure,
    owned_recovery_scope, constraint_force_diagnostics, _array_value)
from anyfesolver.prepared_static_response import _library_schema, _library_vector

# Frozen M16 criteria constants.
UNIT_ROUNDOFF = 2.0 ** -53
TAU_LINEAR = 2.0 ** 10
TAU_DERIVED = 2.0 ** 14
RELATIVE = 1e-12
DERIVED = frozenset(('von_mises', 'equivalent_stress', 'hill_utilization',
                     'principal_stresses_top', 'principal_stresses_bottom'))
DISPLACEMENT_TOLERANCES = {False: (1e-12, 1e-14), True: (1e-11, 1e-13)}


def _root(name):
    return name[0] if isinstance(name, tuple) else name


def tolerance(name, reference, scale):
    tau = TAU_DERIVED if _root(name) in DERIVED else TAU_LINEAR
    return RELATIVE * np.abs(reference) + tau * UNIT_ROUNDOFF * scale


def _probe_scale_operator(recipe, size, linear):
    zero = _recover_owned_local(recipe, np.zeros(size))
    base = _library_vector(zero, linear)
    columns = []
    for k in range(size):
        unit = np.zeros(size)
        unit[k] = 1.0
        columns.append(_library_vector(_recover_owned_local(recipe, unit), linear) - base)
    matrix = np.column_stack(columns) if columns else np.zeros((base.size, 0))
    return np.abs(matrix), np.abs(base)


def _compare(record, reference, scale, summary, location):
    for name, expected in reference.items():
        if name not in record:
            continue
        actual = np.asarray(record[name], dtype=float)
        expected = np.asarray(expected, dtype=float)
        if actual.shape != expected.shape:
            summary['failures'].append(dict(location, component=str(name), reason='shape'))
            continue
        error = np.abs(actual - expected)
        limit = tolerance(name, expected, scale)
        kind = 'derived' if _root(name) in DERIVED else 'linear'
        summary[kind + '_entries'] += int(error.size)
        ratio = float(np.max(error / limit, initial=0.0))
        summary[kind + '_worst_ratio'] = max(summary[kind + '_worst_ratio'], ratio)
        bad = np.flatnonzero((error > limit).reshape(-1))
        for index in bad[:5]:
            summary['failures'].append(dict(location, component=str(name), index=int(index),
                error=float(error.reshape(-1)[index]), limit=float(limit.reshape(-1)[index]), criterion=kind))
        summary[kind + '_failures'] += int(bad.size)


def _canonical_reference(record, requested):
    """Canonical values for the route's emitted names, principal derived from canonical fields."""
    made = {}
    for name in requested:
        if name in record:
            if _array_value(record[name]) is not None:
                made[name] = record[name]
        elif name in DERIVED:
            made[name] = _derived_measure(name, record)
    return made


def _combined_load_case(patterns, coefficients, name, baseline=None):
    """M20: one native original-time LoadCase equal in exact arithmetic to F0 + sum c_j F_j."""
    from anyfesolver import LoadCase
    terms = ([] if baseline is None else [(baseline, 1.0)]) + [(patterns[key], float(value))
                                                                for key, value in coefficients.items()]
    made = LoadCase(name)
    pressure, gravity = {}, None
    for load, scale in terms:
        if load.element_loads or load.added_node_masses or load.follower_pressure:
            raise ValueError('combined assessment loads support nodal, scalar/tuple pressure and gravity only')
        for node, vector in load.nodal_loads.items():
            made.add_nodal_load(node, scale * np.asarray(vector, dtype=float))
        for element, value in load.pressure_loads.items():
            value = np.asarray(value, dtype=float) * scale
            pressure[element] = value if element not in pressure else pressure[element] + value
        if load.gravity is not None:
            gravity = scale * np.asarray(load.gravity, dtype=float) + (0.0 if gravity is None else gravity)
    for element, value in pressure.items():
        made.add_pressure_load(element, float(value) if value.ndim == 0 else tuple(value))
    if gravity is not None:
        made.gravity = np.array(gravity, dtype=float)
    return made


def _reaction_tolerances(maps, g):
    """C4': accumulate g like the reactions (|coefficient| for master-equivalent forces)."""
    combined, master = {}, {}
    def add(target, dof, value):
        node, local, _component = maps.dof_info[int(dof)]
        target.setdefault(int(node), np.zeros(6))[int(local)] += value
    for dof in maps.fixed_dofs:
        add(combined, dof, g[int(dof)])
    for constraint in maps.mpc_constraints:
        slave = int(constraint['slave'])
        add(combined, slave, g[slave])
        for dof, coefficient in constraint['masters'].items():
            add(master, dof, abs(float(coefficient)) * g[slave])
    return combined, master


NODE_QUANTITIES = ('displacements', 'support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces')


def _bucket_tolerances(maps, g):
    """M23: g accumulated per reaction bucket like the reactions (C4', C8b)."""
    made = {bucket: {} for bucket in NODE_QUANTITIES[1:]}
    def add(bucket, dof, value):
        node, local, _component = maps.dof_info[int(dof)]
        if node < 0 or local < 0:
            return
        made[bucket].setdefault(int(node), np.zeros(6))[int(local)] += value
    for dof in maps.fixed_dofs:
        add('support_reactions', dof, g[int(dof)])
    for constraint in maps.mpc_constraints:
        slave = int(constraint['slave'])
        add('mpc_slave_forces', slave, g[slave])
        for dof, coefficient in constraint['masters'].items():
            add('mpc_master_equivalent_forces', dof, abs(float(coefficient)) * g[slave])
    return made


def _assess_node_envelope(result, node_cases, node_limits, summary):
    """C8a (displacements, bitwise) and C8b (reactions, scaled when the case load was compiled)."""
    count = len(node_cases)
    zeros = np.zeros(6)
    for entry in result.extrema:
        if entry.quantity not in NODE_QUANTITIES:
            continue
        node = int(entry.entity_id)
        values = np.stack([np.asarray(case.get(entry.quantity, {}).get(node, zeros), dtype=float).reshape(-1)
                           for case in node_cases])
        if entry.quantity == 'displacements':
            limits, exact = np.zeros_like(values), np.ones(count, dtype=bool)
        else:
            limits = np.stack([zeros if limit is None else np.asarray(limit[entry.quantity].get(node, zeros), dtype=float)
                               for limit in node_limits])
            exact = np.array([limit is None for limit in node_limits])
        for mode, governing in (('minimum', entry.minimum), ('maximum', entry.maximum), ('absolute', entry.absolute)):
            keyed = np.abs(values) if mode == 'absolute' else values
            winners = np.argmin(keyed, axis=0) if mode == 'minimum' else np.argmax(keyed, axis=0)
            route_values = np.asarray(governing.values, dtype=float).reshape(-1)
            route_index = np.asarray(governing.input_indices).reshape(-1)
            for k in range(values.shape[1]):
                summary['node_envelope_entries'] += 1
                c_gov, r_gov = int(winners[k]), int(route_index[k])
                canonical_value = values[c_gov, k]
                bitwise = bool(exact[c_gov] and exact[r_gov])
                limit = 0.0 if bitwise else float(RELATIVE * abs(canonical_value)
                    + TAU_LINEAR * UNIT_ROUNDOFF * max(limits[c_gov, k], limits[r_gov, k]))
                error = abs(route_values[k] - canonical_value)
                if (bitwise and route_values[k] != canonical_value) or error > limit:
                    summary['node_envelope_value_failures'] += 1
                    summary['failures'].append(dict(quantity=entry.quantity, node=node, mode=mode, index=k,
                        route=float(route_values[k]), canonical=float(canonical_value), limit=limit, criterion='C8_value'))
                if r_gov != c_gov:
                    summary['node_identity_differences'] += 1
                    if abs(keyed[r_gov, k] - keyed[c_gov, k]) > limit:
                        summary['node_identity_non_near_ties'] += 1
                        summary['failures'].append(dict(quantity=entry.quantity, node=node, mode=mode, index=k,
                            route_case=r_gov, canonical_case=c_gov, criterion='C8_identity'))
                for field, associated in governing.associated_fields.items():
                    got = np.asarray(associated, dtype=float).reshape(values.shape[1], -1)[k]
                    want = values[r_gov]
                    if exact[r_gov]:
                        ok = np.array_equal(got, want)
                    else:
                        ok = bool(np.all(np.abs(got - want) <= RELATIVE * np.abs(want)
                                         + TAU_LINEAR * UNIT_ROUNDOFF * limits[r_gov]))
                    summary['associated_entries'] += 1
                    if not ok:
                        summary['associated_failures'] += 1
                        summary['failures'].append(dict(quantity=entry.quantity, node=node, mode=mode, index=k,
                            field=str(field), criterion='C8_associated'))


def assess(label, model, patterns, coefficient_sets, *, recovery, envelope=None, baseline=None,
           free=False, batch_size=4, stress_execution='certified_library', native_loads=False, route_options=None):
    summary = dict(label=label, cases=len(coefficient_sets), linear_entries=0, derived_entries=0,
                   linear_failures=0, derived_failures=0, linear_worst_ratio=0.0, derived_worst_ratio=0.0,
                   failures=[], displacement_failures=0, displacement_worst=0.0,
                   reaction_mismatches=0, envelope_value_failures=0, envelope_identity_differences=0,
                   envelope_identity_non_near_ties=0, envelope_entries=0, reconstructed=0,
                   detailed_records_checked=0, library_records_checked=0,
                   old_uniform_screen_failures_vs_direct=0, scaled_criteria_failures_vs_direct=0,
                   native_loads=native_loads, compiled_load_cases=0, c7_worst_ratio=0.0, c7_failures=0,
                   c4prime_worst_ratio=0.0, c4prime_failures=0, master_entries_checked=0,
                   node_envelope_entries=0, node_envelope_value_failures=0, node_identity_differences=0,
                   node_identity_non_near_ties=0, associated_entries=0, associated_failures=0)
    cases = [PatternCase(f'{label}-{i}', c, canonical_load_case=(
                 _combined_load_case(patterns, c, f'{label}-{i}', baseline) if native_loads else None))
             for i, c in enumerate(coefficient_sets)]
    with prepare_static_response_evaluator(model, patterns, baseline_load_case=baseline,
            recovery_config=recovery, envelope_recovery_config=envelope,
            stress_execution=stress_execution, **(route_options or {})) as route:
        blocks = list(route.iter_batches(cases, batch_size=batch_size))
        snapshot = route._snapshot
        n = snapshot.reduced_stiffness.shape[0]
        full_columns = np.asarray(snapshot.transformation @ route._solution_columns[:n])
        full_columns[:, 0] += snapshot.prescribed_offset
        union = route._library_elements(snapshot)
        schemas, operators = {}, {}
        with owned_recovery_scope(tuple(recipe for _mapping, recipe in union.values())):
            for eid, (mapping, recipe) in union.items():
                schemas[eid] = _library_schema(eid, _recover_owned_local(recipe, np.zeros(mapping.size)))
                operators[eid] = _probe_scale_operator(recipe, int(mapping.size), schemas[eid][1])
        u_cases, scales, canonical_fields = [], [], []
        packets = []
        for block in blocks:
            for j, case_id in enumerate(block.case_ids):
                info = block.case_results[j]['solver_info']
                summary['reconstructed'] += int(bool(info.get('source_response', {}).get('reconstructed')))
                packets.append(block.case_results[j])
                u_cases.append(np.asarray(block.displacements[:, j], dtype=float))
        # Native and route loads per case, evaluated before the recovery scope.
        from anyfesolver.prepared_static_loads import evaluate_load_vector
        from anyfesolver.prepared_static_vectorized import c7_ratio
        abs_stiffness = abs(snapshot.stiffness)
        loads_by_case = []
        with owned_recovery_scope(tuple(snapshot.load_maps.pressure_recipes.values())):
            for case in cases:
                _, _, packet = route._ingest_case(case)
                if packet.canonical is None:
                    loads_by_case.append((None, None, None))
                    continue
                native, _info = evaluate_load_vector(snapshot.load_maps, packet.canonical)
                made = None if getattr(route, '_compiled_pressure', None) is None else \
                    route._compiled_pressure.evaluate(snapshot.load_maps, packet.canonical)
                if made is None:
                    loads_by_case.append((native, native, None))
                    continue
                route_load, _info, values, other = made
                term = route._compiled_pressure.term_scale(values, other)
                ratio = c7_ratio(route_load, native, term)
                summary['compiled_load_cases'] += 1
                summary['c7_worst_ratio'] = max(summary['c7_worst_ratio'], ratio)
                summary['c7_failures'] += int(ratio > 1.0)
                loads_by_case.append((native, route_load, term))
        position = 0
        node_cases, node_limits = [], []
        with owned_recovery_scope(tuple(recipe for _mapping, recipe in union.values())):
            for case, result in zip(cases, packets):
                _, _, packet = route._ingest_case(case)
                c = np.zeros(full_columns.shape[1])
                c[0] = 1.0
                c[packet.indices] = packet.coefficients
                u_c = u_cases[position]
                native_load, route_load, term = loads_by_case[position]
                load = route._combine(route._force_columns, packet) if native_load is None else native_load
                # C1: canonical direct solve of the actual (native) load.
                rhs = np.asarray(snapshot.transformation.T @ (load - snapshot.stiffness @ snapshot.prescribed_offset))
                if route._operator_augmented:
                    rhs = np.concatenate((rhs, np.zeros(snapshot.rigid_modes.shape[1])))
                direct = PreparedStaticEvaluator._solve_block(route, np.ascontiguousarray(rhs[:, None]), ())[:, 0]
                u_direct = snapshot.transformation @ direct[:n] + snapshot.prescribed_offset
                rtol, atol = DISPLACEMENT_TOLERANCES[free]
                error = np.abs(u_c - u_direct)
                summary['displacement_failures'] += int(np.count_nonzero(error > atol + rtol * np.abs(u_direct)))
                summary['displacement_worst'] = max(summary['displacement_worst'], float(error.max(initial=0.0)))
                # C4: reactions are actual Ku-F at u_c.
                plan = snapshot.envelope_plan
                need_forces = recovery.include_reactions or (plan is not None and plan.include_reactions)
                forces = None
                if need_forces:
                    forces = constraint_force_diagnostics(snapshot.reaction_maps, snapshot.stiffness,
                        snapshot.transformation, snapshot.rigid_modes, u_c, load,
                        constraint_info=snapshot.constraint_info, nullspace_info=snapshot.nullspace_info)
                if plan is not None:
                    node_case = {}
                    if plan.include_displacements:
                        node_case['displacements'] = {int(node): u_c[np.asarray(snapshot.load_maps.node_dofs[node])]
                                                      for node in plan.node_ids}
                    if plan.include_reactions:
                        for bucket in NODE_QUANTITIES[1:]:
                            node_case[bucket] = forces[bucket]
                    node_cases.append(node_case)
                    node_limits.append(None if term is None else _bucket_tolerances(snapshot.reaction_maps,
                        np.asarray(abs_stiffness @ np.abs(u_c), dtype=float) + term))
                if recovery.include_reactions:
                    combined = {}
                    for bucket in ('support_reactions', 'mpc_slave_forces'):
                        for node, force in forces[bucket].items():
                            combined.setdefault(int(node), np.zeros(6))
                            combined[int(node)] += force
                    if term is None:
                        for node, value in result['reactions'].items():
                            if not np.array_equal(np.asarray(value), combined.get(int(node), np.zeros(6))):
                                summary['reaction_mismatches'] += 1
                    else:
                        # C4': compiled load, scaled criterion against canonical diagnostics with the native load.
                        g = np.asarray(abs_stiffness @ np.abs(u_c), dtype=float) + term
                        reaction_tol, master_tol = _reaction_tolerances(snapshot.reaction_maps, g)
                        pairs = [(result['reactions'], combined, reaction_tol)]
                        route_master = result.get('constraint_forces', {}).get('mpc_master_equivalent_forces')
                        if route_master is not None:
                            pairs.append((route_master, forces['mpc_master_equivalent_forces'], master_tol))
                        for got_map, wanted_map, tol_map in pairs:
                            for node, value in got_map.items():
                                wanted = np.asarray(wanted_map.get(int(node), np.zeros(6)), dtype=float)
                                error = np.abs(np.asarray(value, dtype=float) - wanted)
                                limit = RELATIVE * np.abs(wanted) + TAU_LINEAR * UNIT_ROUNDOFF * tol_map.get(int(node), np.zeros(6))
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    ratio = float(np.max(np.where(error == 0.0, 0.0, error / limit), initial=0.0))
                                summary['c4prime_worst_ratio'] = max(summary['c4prime_worst_ratio'], ratio)
                                summary['c4prime_failures'] += int(ratio > 1.0)
                                if got_map is route_master:
                                    summary['master_entries_checked'] += 1
                magnitude = np.abs(full_columns) @ np.abs(c)
                case_scales, case_fields = {}, {}
                for eid, (mapping, recipe) in union.items():
                    abs_h, abs_r = operators[eid]
                    scale = float(np.max(abs_h @ magnitude[mapping] + abs_r, initial=0.0))
                    case_scales[eid] = scale
                    canonical = _recover_owned_local(recipe, u_c[mapping])
                    location = dict(case=case.case_id, element_id=int(eid))
                    # C2/C3 over every library record the route folds or emits.
                    if stress_execution == 'certified_library':
                        library_record = route._library_record(eid, packet)
                        _compare(library_record, _canonical_reference(canonical, library_record), scale, summary, location)
                        summary['library_records_checked'] += 1
                    emitted = result['element_stresses'].get(eid)
                    if emitted is not None:
                        _compare(emitted, _canonical_reference(canonical, emitted), scale,
                                 summary, dict(location, output='detailed'))
                        summary['detailed_records_checked'] += 1
                        # Diagnostic only: comparison against the direct-solve stream.
                        direct_record = _recover_owned_local(recipe, u_direct[mapping])
                        reference = _canonical_reference(direct_record, emitted)
                        for name, expected in reference.items():
                            actual = np.asarray(emitted[name], dtype=float)
                            expected = np.asarray(expected, dtype=float)
                            if actual.shape != expected.shape:
                                continue
                            summary['old_uniform_screen_failures_vs_direct'] += int(np.count_nonzero(
                                np.abs(actual - expected) > 1e-12 + 1e-12 * np.abs(expected)))
                            summary['scaled_criteria_failures_vs_direct'] += int(np.count_nonzero(
                                np.abs(actual - expected) > tolerance(name, expected, scale)))
                    case_fields[eid] = canonical
                scales.append(case_scales)
                canonical_fields.append(case_fields)
                position += 1
        envelope_result = route.envelope_result()
        if envelope_result is not None:
            _assess_envelope(route, snapshot, envelope_result, canonical_fields, scales, summary)
            _assess_node_envelope(envelope_result, node_cases, node_limits, summary)
        diagnostics = route.diagnostics()['source_response']
        summary['route_diagnostics'] = {key: diagnostics.get(key) for key in (
            'reconstructed_cases', 'library_stress_cases', 'certification_fallback_cases',
            'source_agreement_fallback_cases', 'direct_cases', 'reconstructed_element_records',
            'element_library_elements', 'element_library_numeric_bytes',
            'element_library_worst_verification_residual', 'governing_identity_basis',
            'vectorized_stress_cases', 'vectorized_elements', 'vectorized_groups',
            'vectorized_worst_verification_residual', 'load_execution', 'compiled_load_cases', 'native_load_cases',
            'compiled_pressure_note', 'compiled_pressure_worst_c7_ratio', 'max_source_discrepancy_ratio',
            'source_agreement_fallback_cases', 'reaction_metadata_copies')}
        summary['envelope_diagnostics'] = None if envelope_result is None else {
            key: envelope_result.diagnostics.get(key) for key in ('recovery', 'governing_identity_basis', 'library_stress_cases')}
    summary['failure_count'] = (summary['c7_failures'] + summary['c4prime_failures']
        + summary['node_envelope_value_failures'] + summary['node_identity_non_near_ties']
        + summary['associated_failures']
        + summary['linear_failures'] + summary['derived_failures']
        + summary['displacement_failures'] + summary['reaction_mismatches']
        + summary['envelope_value_failures'] + summary['envelope_identity_non_near_ties']
        + (summary['cases'] - summary['reconstructed']))
    summary['failures'] = summary['failures'][:40]
    return summary


def _assess_envelope(route, snapshot, result, canonical_fields, scales, summary):
    plan = {e.element_id: e for e in snapshot.envelope_plan.elements}
    stacked = {}
    for eid, element in plan.items():
        per_case = [_numeric_fields(dict(fields[eid]), element.components) for fields in canonical_fields]
        stacked[eid] = {name: np.stack([values[name] for values in per_case]) for name in per_case[0]}
    for entry in result.extrema:
        if entry.quantity != 'stress':
            continue
        eid = int(entry.entity_id)
        values = stacked[eid][entry.component].reshape(len(canonical_fields), -1)
        for mode, governing in (('minimum', entry.minimum), ('maximum', entry.maximum), ('absolute', entry.absolute)):
            keyed = np.abs(values) if mode == 'absolute' else values
            winners = np.argmin(keyed, axis=0) if mode == 'minimum' else np.argmax(keyed, axis=0)
            route_values = np.asarray(governing.values, dtype=float).reshape(-1)
            route_index = np.asarray(governing.input_indices).reshape(-1)
            for k in range(values.shape[1]):
                summary['envelope_entries'] += 1
                c_gov = int(winners[k])
                r_gov = int(route_index[k])
                canonical_value = values[c_gov, k]
                scale = max(scales[c_gov][eid], scales[r_gov][eid])
                limit = float(tolerance(entry.component, canonical_value, scale))
                if abs(route_values[k] - canonical_value) > limit:
                    summary['envelope_value_failures'] += 1
                    summary['failures'].append(dict(element_id=eid, component=str(entry.component), mode=mode,
                        index=k, route=float(route_values[k]), canonical=float(canonical_value), limit=limit,
                        criterion='envelope_value'))
                for field, associated in governing.associated_fields.items():
                    got = np.asarray(associated, dtype=float).reshape(values.shape[1], -1)[k]
                    want = np.asarray(stacked[eid][field][r_gov], dtype=float).reshape(-1)
                    allowed = tolerance(field, want, scales[r_gov][eid])
                    summary['associated_entries'] += 1
                    if got.shape != want.shape or np.any(np.abs(got - want) > allowed):
                        summary['associated_failures'] += 1
                        summary['failures'].append(dict(element_id=eid, component=str(entry.component), mode=mode,
                            index=k, field=str(field), criterion='C5_associated'))
                if r_gov != c_gov:
                    summary['envelope_identity_differences'] += 1
                    keyed_route = keyed[r_gov, k]
                    keyed_canonical = keyed[c_gov, k]
                    if abs(keyed_route - keyed_canonical) > limit:
                        summary['envelope_identity_non_near_ties'] += 1
                        summary['failures'].append(dict(element_id=eid, component=str(entry.component), mode=mode,
                            index=k, route_case=r_gov, canonical_case=c_gov, criterion='envelope_identity'))


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def _shell_patterns():
    from test_prepared_static_evaluator import _shell_case
    return {'pressure': _shell_case(0, pressure=2.).load_case, 'nodal': _shell_case(1).load_case,
            'gravity': _shell_case(2, gravity=True).load_case}


def _random_sets(seed, count, names, low, high, *, empty=False, single=None):
    rng = np.random.default_rng(seed)
    sets = []
    if empty:
        sets.append({})
    if single is not None:
        sets.append({single: float(rng.uniform(low, high))})
    while len(sets) < count:
        sets.append({name: float(rng.uniform(low, high)) for name in names})
    return sets


SHELL_COMPONENTS = ['membrane_xx', 'membrane_yy', 'membrane_xy', 'bending_xx', 'bending_yy', 'bending_xy',
                    'shear_xz', 'shear_yz', 'von_mises', 'principal_stresses_top', 'principal_stresses_bottom']


def holdout_a1(stress_execution='certified_library', native_loads=False, route_options=None):
    from test_prepared_static_evaluator import _shell_model
    names = ('pressure', 'nodal', 'gravity')
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True, components=SHELL_COMPONENTS)
    envelope = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True,
                              components=SHELL_COMPONENTS)
    return assess('A1-L' if native_loads else 'A1', _shell_model(), _shell_patterns(),
                  _random_sets(16001, 12, names, -5.0, 5.0, empty=True, single='pressure'),
                  recovery=recovery, envelope=envelope, stress_execution=stress_execution, native_loads=native_loads,
                  route_options=route_options)


def holdout_a2(stress_execution='certified_library', native_loads=False):
    from test_prepared_static_evaluator import _shell_model
    model = _shell_model()
    rotation = _rotation((1.0, 2.0, 3.0), 0.37)
    for node in model.mesh.nodes.values():
        node.x, node.y, node.z = rotation @ np.array([node.x, node.y, node.z])
    names = ('pressure', 'nodal', 'gravity')
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True, components=SHELL_COMPONENTS)
    return assess('A2-L' if native_loads else 'A2', model, _shell_patterns(), _random_sets(16002, 6, names, -5.0, 5.0),
                  recovery=recovery, stress_execution=stress_execution, native_loads=native_loads)


def holdout_a3(stress_execution='certified_library', native_loads=False, route_options=None):
    from benchmark_static_batches import mixed_free_model, free_cases
    model, groups, balancing = mixed_free_model()
    patterns = {name: load for name, _time, load in free_cases(model, groups, balancing, [17, 233, 401])}
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    envelope = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)
    return assess('A3-L' if native_loads else 'A3', model, patterns, _random_sets(16003, 8, tuple(patterns), -2.0, 2.0),
                  recovery=recovery, envelope=envelope, free=True, stress_execution=stress_execution,
                  native_loads=native_loads, route_options=route_options)


def holdout_a4(stress_execution='certified_library', route_options=None):
    from anyfesolver import BoundaryCondition, LoadCase
    from test_prepared_static_evaluator import _bar, _case
    model = _bar(False)
    model.add_boundary_condition(BoundaryCondition('left', [1], {'ux': 0.001, 'uy': 0., 'uz': 0., 'rx': 0., 'ry': 0., 'rz': 0.}))
    model.add_boundary_condition(BoundaryCondition('right', [2], {'uy': 0., 'uz': 0., 'rx': 0., 'ry': 0., 'rz': 0.}))
    constrained = LoadCase('q')
    constrained.add_nodal_load(2, [0.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    patterns = {'p': _case(1, 4).load_case, 'q': constrained}
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    envelope = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)
    return assess('A4', model, patterns, _random_sets(16004, 4, ('p', 'q'), -3.0, 3.0),
                  recovery=recovery, envelope=envelope, baseline=_case(2, 2).load_case,
                  stress_execution=stress_execution, route_options=route_options)


def _mpc_shell_model():
    """M23 A5: supported S3 with two MPC equations on node 4."""
    from test_prepared_static_evaluator import _shell_model
    model = _shell_model()
    def dof(node, local):
        dofs = getattr(model.mesh.nodes[node], 'dofs', None)
        return int(dofs[local]) if dofs is not None and len(dofs) > local else 6 * (node - 1) + local
    model.add_constraint_equation(terms=((dof(4, 2), 1.), (dof(2, 2), -.8)), rhs=0., source_id='m23 uz', dependent_dof=dof(4, 2))
    model.add_constraint_equation(terms=((dof(4, 0), 1.), (dof(2, 0), -.5), (dof(2, 1), -.25)), rhs=0.,
                                  source_id='m23 ux', dependent_dof=dof(4, 0))
    return model


def holdout_a5(stress_execution='certified_vectorized', native_loads=True, route_options=None):
    names = ('pressure', 'nodal', 'gravity')
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True, components=SHELL_COMPONENTS)
    envelope = RecoveryConfig(include_displacements=True, include_stresses=True, include_reactions=True,
                              components=SHELL_COMPONENTS)
    return assess('A5-L' if native_loads else 'A5', _mpc_shell_model(), _shell_patterns(),
                  _random_sets(16005, 8, names, -5.0, 5.0), recovery=recovery, envelope=envelope,
                  stress_execution=stress_execution, native_loads=native_loads, route_options=route_options)


def development_fixtures(stress_execution='certified_library'):
    """Disclosed development fixtures (M8 accuracy and M9 cases); not blind validation."""
    from test_prepared_static_evaluator import _shell_model, _shell_case
    from benchmark_static_batches import mixed_free_model, free_cases
    recovery = RecoveryConfig(include_stresses=True, include_reactions=False, components=SHELL_COMPONENTS)
    results = [assess('D-S3', _shell_model(), {'p': _shell_case(0, pressure=2.).load_case},
                      [{'p': c} for c in (1., .1, .3, 3.7, -1.25)], recovery=recovery,
                      stress_execution=stress_execution)]
    model, groups, balancing = mixed_free_model()
    patterns = {name: load for name, _time, load in free_cases(model, groups, balancing, [0, 48, 576])}
    results.append(assess('D-mixed', model, patterns,
        [{'case-048': .3}, {'case-000': .25, 'case-048': .75}, {'case-048': 1., 'case-576': -.7}],
        recovery=RecoveryConfig(element_ids=[1, 4, 5, 6], include_stresses=True, include_reactions=False), free=True,
        stress_execution=stress_execution))
    return results
