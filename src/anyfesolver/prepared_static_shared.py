"""Shared physical recovery and bounded block winner reduction.

One canonical element record feeds independently filtered selected histories and
numeric envelope tiles. Nothing in this module observes callers or publishes a
transaction; the evaluator owns the final callback/authority publication barrier.
"""
import copy
import types
import numpy as np
from .boundary import _callable_authority, _callable_authority_matches
from .static_batch import _class_state
from .prepared_static_loads import PreparedStaticError, PreparedStaticCapabilityError
from .prepared_static_recovery import (_recover_owned_local, owned_recovery_scope,
    constraint_force_diagnostics, _derived_measure)
from .prepared_static_envelopes import EnvelopeAccumulator, _numeric_fields
from . import prepared_static_envelopes as _envelope_module
from .recovery import _filter_components

CASE_TILE_SIZE = 8
_BUCKETS = ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces')


def _selected_record(compiled, canonical):
    record = {}
    for name, _baseline, _matrix, _shape in compiled.array_components:
        record[name] = copy.deepcopy(canonical[name])
    for name, _value in compiled.constant_components:
        record[name] = copy.deepcopy(canonical[name])
    for name, _kind in compiled.derived_components:
        value = canonical[name] if name in canonical else _derived_measure(name, canonical)
        record[name] = copy.deepcopy(value)
    return record


def _selected_reactions(snapshot, recovery, forces, item):
    combined = {}
    for bucket in ('support_reactions', 'mpc_slave_forces'):
        for node, force in forces[bucket].items():
            combined.setdefault(int(node), np.zeros(6))
            combined[int(node)] += force
    selected = set(snapshot.selected_node_ids)
    item['reactions'] = {node: value for node, value in combined.items()
                         if recovery.node_ids is None or node in selected}
    item['reaction_node_ids'] = list(snapshot.selected_node_ids)
    # Keep complete residual/MPC diagnostic truth. Only selected force buckets
    # are filtered, after the envelope has its independent full-node source.
    made = dict(forces)
    if recovery.node_ids is not None:
        for bucket in _BUCKETS:
            made[bucket] = {node: value for node, value in forces[bucket].items() if node in selected}
    item['constraint_forces'] = made


def _recovery_metadata(compilation):
    if compilation is None:
        return {}, {}
    elements = compilation.elements
    return ({'method': 'prepared_canonical_shared_physical_records',
             'compiled_elements': len(elements), 'probe_count': compilation.probe_count,
             'worst_verification_relative_residual': max((e.verification_relative_residual for e in elements.values()), default=0.),
             'setup_seconds': compilation.setup_seconds},
            {'prepared_affine_recovery': True, 'compiled_elements': len(elements),
             'oracle_probes_at_preparation': compilation.probe_count,
             'canonical_local_refined_elements': list(elements),
             'prepared_q4_staged_elements': [eid for eid, e in elements.items() if e.local_recipe.staged_kernel is not None]})


def _fold_labeled(reduced, quantity, entity, fields, labels):
    try:
        reduced.fold_tile(quantity, entity, fields, labels)
    except Exception as error:
        raise PreparedStaticError('shared envelope tile reduction failed', diagnostics={
            'stage': 'envelope reduction', 'quantity': quantity, 'entity_id': entity,
            'element_id': entity if quantity == 'stress' else None,
            'case_id': labels[0][0], 'input_index': labels[0][2],
            'tile_input_indices': [label[2] for label in labels], 'error': str(error)}) from error


def recover_shared_block(snapshot, full, loads, recovery, labels, certified, envelope, record_source=None):
    # record_source(eid, column) may supply an owned certified-response record
    # for a case column; it returns None to request canonical recovery.
    detailed = {} if not recovery.include_stresses or snapshot.recovery_compilation is None else snapshot.recovery_compilation.elements
    enveloped = {} if envelope is None else {e.element_id: e for e in envelope.plan.elements}
    union = {eid: (entry.dof_mapping, entry.recipe) for eid, entry in enveloped.items()}
    union.update({eid: (entry.dof_mapping, entry.local_recipe) for eid, entry in detailed.items()})
    ordered = sorted(union, key=lambda eid: (type(union[eid][1].execution_owner[0]).__module__,
                                            type(union[eid][1].execution_owner[0]).__qualname__, eid))
    results = [{'status': 'converged', 'solver_info': info, 'node_displacements': {},
                'element_stresses': {}, 'reactions': {}} for info in certified]
    provenance, execution = _recovery_metadata(snapshot.recovery_compilation if detailed else None)
    if detailed:
        for info in certified:
            info['stress_recovery'] = copy.deepcopy(provenance)
            info['recovery'] = copy.deepcopy(execution)
    transaction = None if envelope is None else envelope.transaction()
    statistics = {'physical_element_evaluations': 0, 'reaction_diagnostic_evaluations': 0,
                  'overlapping_element_cases': len(set(detailed) & set(enveloped)) * len(labels),
                  'case_tile_limit': CASE_TILE_SIZE, 'max_case_tile': min(CASE_TILE_SIZE, len(labels)),
                  'envelope_merge_entities': 0}
    with owned_recovery_scope(tuple(union[eid][1] for eid in ordered)):
        for eid in ordered:
            mapping, recipe = union[eid]
            reduced = None if eid not in enveloped else transaction.empty_subset(('stress',), eid)
            for start in range(0, len(labels), CASE_TILE_SIZE):
                stop = min(start + CASE_TILE_SIZE, len(labels))
                tile_fields = {}
                # Keep canonical GEMV per physical sample; the tile vectorizes
                # only winner selection, never mechanical stage arithmetic.
                for column in range(start, stop):
                    case_id, _time, index = labels[column]
                    try:
                        canonical = None if record_source is None else record_source(eid, column)
                        if canonical is None:
                            canonical = _recover_owned_local(recipe, full[mapping, column])
                            statistics['physical_element_evaluations'] += 1
                        else:
                            statistics['library_element_records'] = statistics.get('library_element_records', 0) + 1
                        selected = {}
                        if eid in detailed:
                            selected = _selected_record(detailed[eid], canonical)
                            results[column]['element_stresses'][eid] = _filter_components(selected, recovery.selected_components())
                        if reduced is not None:
                            components = enveloped[eid].components
                            envelope_record = dict(canonical)
                            if components is not None:
                                for name in components:
                                    if name in selected and name not in envelope_record:
                                        envelope_record[name] = selected[name]
                            fields = _numeric_fields(envelope_record, components)
                            for name, value in fields.items():
                                if name not in tile_fields:
                                    tile_fields[name] = np.empty((stop - start,) + value.shape)
                                tile_fields[name][column - start] = value
                    except Exception as error:
                        from .control import SolveCancelled
                        if isinstance(error, SolveCancelled):
                            raise
                        raise PreparedStaticError('shared canonical element recovery failed', diagnostics={
                            'stage': 'case recovery', 'case_id': case_id, 'input_index': index,
                            'element_id': eid, 'error': str(error)}) from error
                if reduced is not None:
                    _fold_labeled(reduced, 'stress', eid, tile_fields, labels[start:stop])
            if reduced is not None:
                transaction.merge_reduced(reduced)
                statistics['envelope_merge_entities'] += 1
        # Canonical force diagnostics are evaluated once per case. A bounded
        # case tile keeps full truth until both independent selections consume it.
        node_reduced = None if transaction is None else transaction.empty_subset(('displacements',) + _BUCKETS)
        for start in range(0, len(labels), CASE_TILE_SIZE):
            stop = min(start + CASE_TILE_SIZE, len(labels))
            force_tile = []
            need_forces = recovery.include_reactions or (envelope is not None and envelope.plan.include_reactions)
            for column in range(start, stop):
                case_id, _time, index = labels[column]
                try:
                    if recovery.include_displacements:
                        for node in snapshot.selected_node_ids:
                            results[column]['node_displacements'][node] = full[snapshot.load_maps.node_dofs[node], column].copy()
                    if need_forces:
                        forces = constraint_force_diagnostics(snapshot.reaction_maps, snapshot.stiffness,
                            snapshot.transformation, snapshot.rigid_modes, full[:, column], loads[:, column],
                            constraint_info=snapshot.constraint_info, nullspace_info=snapshot.nullspace_info)
                        statistics['reaction_diagnostic_evaluations'] += 1
                        force_tile.append(forces)
                        if recovery.include_reactions:
                            _selected_reactions(snapshot, recovery, forces, results[column])
                except Exception as error:
                    from .control import SolveCancelled
                    if isinstance(error, SolveCancelled):
                        raise
                    raise PreparedStaticError('shared canonical nodal recovery failed', diagnostics={
                        'stage': 'case recovery', 'case_id': case_id, 'input_index': index,
                        'error': str(error)}) from error
            if node_reduced is not None:
                for node in envelope.plan.node_ids:
                    if envelope.plan.include_displacements:
                        values = full[snapshot.load_maps.node_dofs[node], start:stop].T.copy()
                        _fold_labeled(node_reduced, 'displacements', node, {'displacements': values}, labels[start:stop])
                    if envelope.plan.include_reactions:
                        for bucket in _BUCKETS:
                            values = np.asarray([forces[bucket].get(node, np.zeros(6)) for forces in force_tile])
                            _fold_labeled(node_reduced, bucket, node, {'force': values}, labels[start:stop])
        if node_reduced is not None:
            transaction.merge_reduced(node_reduced)
            statistics['envelope_merge_entities'] += len(node_reduced.entity_keys)
        if transaction is not None:
            transaction.cases_processed += len(labels)
            statistics['transaction_copied_modes'] = transaction.copied_mode_count
    return results, transaction, statistics


def shared_workspace_bytes(snapshot, tile_width=CASE_TILE_SIZE, include_reactions=False):
    plan = snapshot.envelope_plan
    # Existing plan.tile_bytes covers gathers and field buffers for one case.
    # Force diagnostic maps include dictionaries, residuals and MPC records.
    reaction = (snapshot.total_dofs * 160 + snapshot.reaction_maps.retained_bytes * 2) if (include_reactions or (plan is not None and plan.include_reactions)) else 0
    return int((0 if plan is None else plan.tile_bytes) * tile_width + reaction * tile_width)


def require_shared_recovery_authority():
    if _class_state(EnvelopeAccumulator) != _ENVELOPE_CLASS_FACTS:
        raise PreparedStaticCapabilityError('prepared envelope reducer authority changed')
    for namespace, name, cls, facts in _ENVELOPE_CLASS_BINDINGS:
        if namespace.get(name) is not cls or _class_state(cls) != facts:
            raise PreparedStaticCapabilityError('prepared envelope result authority changed')
    for namespace, name, function, authority, bindings in _SHARED_FACTS + _ENVELOPE_HELPER_FACTS:
        if (namespace.get(name) is not function or not _callable_authority_matches(function, authority)
                or any(function.__globals__.get(key) is not value for key, value in bindings)):
            raise PreparedStaticCapabilityError('shared recovery executable authority changed')

_ENVELOPE_CLASS_FACTS = _class_state(EnvelopeAccumulator)
_ENVELOPE_CLASS_BINDINGS = tuple((vars(_envelope_module), name, value, _class_state(value))
    for name, value in vars(_envelope_module).items()
    if type(value) is type and value.__module__ == _envelope_module.__name__)
_ENVELOPE_HELPER_FACTS = tuple((vars(_envelope_module), name, value, _callable_authority(value),
    tuple((key, value.__globals__[key]) for key in value.__code__.co_names if key in value.__globals__))
    for name, value in vars(_envelope_module).items() if type(value) is types.FunctionType)
# Methods use module helpers and result constructors that class facts alone do
# not capture. Pin their global bindings before any owned accumulator is passed.
_ENVELOPE_HELPER_FACTS += tuple((vars(cls), name, value, _callable_authority(value),
    tuple((key, value.__globals__[key]) for key in value.__code__.co_names if key in value.__globals__))
    for _namespace, _class_name, cls, _facts in _ENVELOPE_CLASS_BINDINGS
    for name, value in vars(cls).items() if type(value) is types.FunctionType)
_SHARED_FACTS = tuple((globals(), name, value, _callable_authority(value),
    tuple((key, value.__globals__[key]) for key in value.__code__.co_names if key in value.__globals__))
    for name, value in tuple(globals().items()) if type(value) is types.FunctionType)
