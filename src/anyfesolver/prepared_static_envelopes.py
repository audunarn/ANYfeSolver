"""Bounded, transactional canonical envelopes for prepared static evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import sys
import numpy as np
from .analysis_session import _immutable_array
from .prepared_static_loads import PreparedStaticCapabilityError, PreparedStaticError
from .prepared_static_recovery import (_capture_owned_local_recovery, _recover_owned_local,
    _capture_span_factory, _capture_span_proofs_hold, _require_capture_span_authority,
    _capture_plain_arguments, _capture_material_is_native, _capture_modules_unhooked,
    _capture_trust_state, _capture_same_state, _capture_owner_matches_identity, _capture_frozen_equal, _authenticate_trusted_capture,
    _recovery_admission, owned_recovery_scope, constraint_force_diagnostics)
from .prepared_static_measures import PRINCIPAL_COMPONENTS, shell_principal_stresses
from .boundary import _callable_authority, _callable_authority_matches
from .static_batch import _class_state
from .prepared_static_recovery import OwnedLocalRecovery

@dataclass(frozen=True)
class PreparedGoverningValues:
    values: np.ndarray
    case_ids: np.ndarray
    times: np.ndarray
    input_indices: np.ndarray
    associated_fields: dict

@dataclass(frozen=True)
class PreparedEnvelopeExtrema:
    quantity: str
    entity_id: int
    component: str
    original_shape: tuple
    minimum: PreparedGoverningValues
    maximum: PreparedGoverningValues
    absolute: PreparedGoverningValues

    @property
    def absolute_maximum(self):
        return self.absolute

@dataclass(frozen=True)
class PreparedEnvelopeResult:
    complete: bool
    cases_processed: int
    extrema: tuple
    snapshot_identity: str
    diagnostics: dict

@dataclass(frozen=True)
class CanonicalEnvelopeElement:
    element_id: int
    dof_mapping: np.ndarray
    recipe: object
    components: tuple
    schema: tuple

@dataclass(frozen=True)
class CanonicalEnvelopePlan:
    elements: tuple
    node_ids: tuple
    include_displacements: bool
    include_reactions: bool
    retained_bytes: int
    tile_bytes: int


def _numeric_fields(record, components=None):
    if components is not None:
        for component in components:
            if component in PRINCIPAL_COMPONENTS and component not in record:
                record[component] = shell_principal_stresses(record, component.removeprefix('principal_stresses_'))
            if component not in record:
                raise PreparedStaticCapabilityError('requested envelope component unavailable: ' + component)
    result = {}
    def collect(value, path):
        if isinstance(value, dict):
            for name, member in value.items():
                collect(member, path + (name,))
            return
        array = np.asarray(value)
        if array.dtype.kind == 'c':
            raise PreparedStaticCapabilityError('complex envelope component is not physical')
        if array.dtype.kind in 'iuf':
            if not np.all(np.isfinite(array)):
                raise PreparedStaticError('nonfinite canonical envelope component: ' + repr(path))
            result[path[0] if len(path) == 1 else path] = np.asarray(array, dtype=float)
    for name in (record if components is None else components):
        before = len(result)
        collect(record[name], (name,))
        if components is not None and len(result) == before:
            raise PreparedStaticCapabilityError('envelope components must be numeric: ' + name)
    return result


def compile_envelope_plan(model, config):
    if config is None:
        return None
    elements = []
    pending = []
    retained = 0
    components = config.selected_components()
    if components is not None:
        components = tuple(components)
        if any(type(name) is not str for name in components):
            raise PreparedStaticCapabilityError('envelope component names must be plain strings')
    _require_schema_authority()
    if config.include_stresses:
        _authenticate_trusted_capture()
        for eid in config.selected_element_ids(model):
            element = model.mesh.elements[eid]
            admitted, formulation, reason = _recovery_admission(model, element, config)
            if not admitted:
                raise PreparedStaticCapabilityError('envelope recovery refused: ' + reason,
                    diagnostics={'element_id': eid, 'formulation': formulation})
            if components and set(components).intersection(PRINCIPAL_COMPONENTS) and 'beam' in formulation:
                raise PreparedStaticCapabilityError('beam principal stress requires a physical station tensor')
            mapping = _immutable_array(np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp))
            recipe = _capture_owned_local_recovery(model, element)
            pending.append((int(eid), mapping, recipe))
            retained += mapping.nbytes + recipe.retained_bytes
    # Live model/config observations and owned native capture stay outside the
    # numerical span. Only private exact recipes and plain selection metadata
    # enter; no callback, iterator or yield crosses this lease.
    _require_schema_authority()
    if any(type(recipe) is not OwnedLocalRecovery for _eid, _mapping, recipe in pending):
        raise PreparedStaticCapabilityError('envelope schema recipe authority changed')
    try:
        with owned_recovery_scope(tuple(recipe for _eid, _mapping, recipe in pending)):
            for eid, mapping, recipe in pending:
                record = _numeric_fields(_recover_owned_local(recipe, np.zeros(mapping.size)), components)
                if not record:
                    raise PreparedStaticCapabilityError('envelope selection has no numeric recovery components')
                schema = tuple((name, tuple(value.shape)) for name, value in record.items())
                elements.append(CanonicalEnvelopeElement(eid, mapping, recipe, components, schema))
    finally:
        primary = sys.exc_info()[1]
        try:
            _require_schema_authority()
        except BaseException as authority_error:
            if primary is None:
                raise
            primary.add_note('envelope schema output authority also failed: ' + str(authority_error))
    # Stable grouping permits one bounded native element/case tile at a time.
    elements.sort(key=lambda item: (type(item.recipe.execution_owner[0]).__module__, type(item.recipe.execution_owner[0]).__qualname__, item.element_id))
    return CanonicalEnvelopePlan(tuple(elements), tuple(config.selected_node_ids(model)),
        bool(config.include_displacements), bool(config.include_reactions), int(retained),
        max((sum((int(np.prod(shape)) if shape else 1) * 32 for _name, shape in e.schema) + e.dof_mapping.nbytes * 4 for e in elements), default=0))


def _require_schema_authority():
    for namespace, name, function, authority, bindings in _SCHEMA_FUNCTION_FACTS:
        if (namespace.get(name) is not function or not _callable_authority_matches(function, authority)
                or any(function.__globals__.get(key) is not value for key, value in bindings)):
            raise PreparedStaticCapabilityError('envelope schema helper authority changed')
    for name, cls, facts in _SCHEMA_CLASS_FACTS:
        if globals().get(name) is not cls or _class_state(cls) != facts:
            raise PreparedStaticCapabilityError('envelope schema record authority changed')


_SCHEMA_FUNCTION_FACTS = tuple((globals(), name, function, _callable_authority(function),
    tuple((key, function.__globals__[key]) for key in function.__code__.co_names if key in function.__globals__))
    for name in ('_capture_owned_local_recovery', '_recover_owned_local',
                 '_numeric_fields', 'shell_principal_stresses', 'owned_recovery_scope',
                 '_capture_span_factory', '_capture_span_proofs_hold', '_require_capture_span_authority',
                 '_capture_plain_arguments', '_capture_material_is_native', '_capture_modules_unhooked',
                 '_capture_trust_state', '_capture_same_state', '_capture_owner_matches_identity',
                 '_capture_frozen_equal')
    for function in (globals()[name],))
from . import prepared_static_measures as _schema_measures
_SCHEMA_FUNCTION_FACTS += tuple((vars(_schema_measures), name, function, _callable_authority(function),
    tuple((key, function.__globals__[key]) for key in function.__code__.co_names if key in function.__globals__))
    for name in ('principal_stresses', 'shell_surface_tensor')
    for function in (getattr(_schema_measures, name),))
_SCHEMA_CLASS_FACTS = tuple((name, cls, _class_state(cls))
    for name in ('OwnedLocalRecovery', 'CanonicalEnvelopeElement', 'CanonicalEnvelopePlan')
    for cls in (globals()[name],))


def _storage(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (sum(_storage(item, seen) for item in value.flat) if value.dtype.hasobject else value.nbytes)
    if isinstance(value, dict):
        return size + sum(_storage(k, seen) + _storage(v, seen) for k, v in dict.items(value))
    if isinstance(value, (tuple, list)):
        return size + sum(_storage(v, seen) for v in value)
    if hasattr(value, '__dict__'):
        return size + _storage(vars(value), seen)
    return size


class EnvelopeAccumulator:
    def __init__(self, plan):
        self.plan = plan
        self.complete = False
        self.cases_processed = 0
        self.staged_elements = tuple(e.element_id for e in plan.elements if e.recipe.staged_kernel is not None)
        self.fallback_reasons = {e.element_id: e.recipe.staged_fallback_reason for e in plan.elements if e.recipe.staged_kernel is None}
        self.entries = {}
        self.entity_keys = {}
        self._copied_modes = set()
        self.copied_mode_count = 0
        self._nbytes_cache = None
        for element in plan.elements:
            self._allocate('stress', element.element_id, element.schema)
        for node in plan.node_ids:
            if plan.include_displacements:
                self._allocate('displacements', node, (('displacements', (6,)),))
            if plan.include_reactions:
                
                for bucket in ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces'):
                    self._allocate(bucket, node, (('force', (6,)),))

    def _allocate(self, quantity, entity, schema):
        self._nbytes_cache = None
        for name, shape in schema:
            size = int(np.prod(shape)) if shape else 1
            modes = []
            for _ in range(3):
                modes.append({'values': np.full(size, np.nan), 'case_ids': np.full(size, None, dtype=object),
                    'times': np.full(size, None, dtype=object), 'input_indices': np.full(size, -1, dtype=np.int64),
                    'associated_fields': {field: np.full((size,) + fieldshape, np.nan) for field, fieldshape in schema}})
            key = (quantity, int(entity), name)
            self.entries[key] = (shape, modes)
            entity_key = (quantity, int(entity))
            self.entity_keys[entity_key] = self.entity_keys.get(entity_key, ()) + (key,)
            self._copied_modes.update((key, mode) for mode in range(3))

    def nbytes(self):
        if self._nbytes_cache is None:
            self._nbytes_cache = _storage((self.entries, self.entity_keys))
        return self._nbytes_cache

    def transaction(self, *, empty=False):
        made = object.__new__(type(self))
        made.plan = self.plan
        made.complete = False
        made.cases_processed = self.cases_processed
        made.staged_elements = self.staged_elements
        made.fallback_reasons = self.fallback_reasons
        made.entries = {} if empty else dict(self.entries)
        # Schema allocation must not mutate a committed owner's index/cache.
        made.entity_keys = {} if empty else dict(self.entity_keys)
        made._copied_modes = set()
        made.copied_mode_count = 0
        # A copied dictionary may have a different allocation from its source;
        # do not inherit the committed state's exact storage measurement.
        made._nbytes_cache = None
        return made

    def _writable_mode(self, key, mode):
        # Even an already-owned mode can change label/time object storage.
        # Invalidate before handing any mutable arrays to the folding code.
        self._nbytes_cache = None
        if (key, mode) not in self._copied_modes:
            shape, modes = self.entries[key]
            made_modes = list(modes)
            old = modes[mode]
            made_modes[mode] = {name: value.copy() for name, value in old.items() if name != 'associated_fields'}
            made_modes[mode]['associated_fields'] = {name: value.copy() for name, value in old['associated_fields'].items()}
            self.entries[key] = (shape, made_modes)
            self._copied_modes.add((key, mode))
            self.copied_mode_count += 1
        return self.entries[key][1][mode]

    def empty_subset(self, quantities, entity=None):
        made = self.transaction(empty=True)
        schemas = {}
        groups = ((quantity, entity) for quantity in quantities) if entity is not None else (
            key for key in self.entity_keys if key[0] in quantities)
        for quantity, item in groups:
            for key in self.entity_keys.get((quantity, item), ()):
                schemas.setdefault((quantity, item), []).append((key[2], self.entries[key][0]))
        for (quantity, item), schema in schemas.items():
            made._allocate(quantity, item, tuple(schema))
        return made

    def fold_tile(self, quantity, entity, fields, labels):
        count = len(labels)
        if not count:
            return
        for name, values in fields.items():
            key = (quantity, int(entity), name)
            shape, modes = self.entries[key]
            actual = np.asarray(values, dtype=float)
            if actual.shape != (count,) + shape or not np.all(np.isfinite(actual)):
                raise PreparedStaticError('canonical envelope tile schema or finiteness changed')
            flat = actual.reshape(count, -1)
            columns = np.arange(flat.shape[1])
            for mode in range(3):
                winners = (np.argmin(flat, axis=0) if mode == 0 else
                           np.argmax(flat if mode == 1 else np.abs(flat), axis=0))
                values = flat[winners, columns]
                indices = np.asarray([labels[int(winner)][2] for winner in winners], dtype=np.int64)
                arrays = self.entries[key][1][mode]
                old = arrays['values']
                better = values < old if mode == 0 else values > old if mode == 1 else np.abs(values) > np.abs(old)
                tie = values == old if mode < 2 else np.abs(values) == np.abs(old)
                update = (arrays['input_indices'] < 0) | better | (tie & (indices < arrays['input_indices']))
                if not np.any(update):
                    continue
                arrays = self._writable_mode(key, mode)
                arrays['values'][update] = values[update]
                arrays['input_indices'][update] = indices[update]
                arrays['case_ids'][update] = [labels[int(winner)][0] for winner in winners[update]]
                arrays['times'][update] = [labels[int(winner)][1] for winner in winners[update]]
                for field, associated in arrays['associated_fields'].items():
                    associated[update] = fields[field][winners[update]]

    def merge_reduced(self, reduced):
        # Merge each component's entire-block winners once. Old modes remain
        # shared unless a winner actually changes; no committed buffer is edited.
        for key, (_shape, candidates) in reduced.entries.items():
            for mode, candidate in enumerate(candidates):
                arrays = self.entries[key][1][mode]
                new, old = candidate['values'], arrays['values']
                better = new < old if mode == 0 else new > old if mode == 1 else np.abs(new) > np.abs(old)
                tie = new == old if mode < 2 else np.abs(new) == np.abs(old)
                update = ((arrays['input_indices'] < 0) | better | (tie & (candidate['input_indices'] < arrays['input_indices']))) & (candidate['input_indices'] >= 0)
                if not np.any(update):
                    continue
                arrays = self._writable_mode(key, mode)
                for name in ('values', 'input_indices', 'case_ids', 'times'):
                    arrays[name][update] = candidate[name][update]
                for name in arrays['associated_fields']:
                    arrays['associated_fields'][name][update] = candidate['associated_fields'][name][update]

    def fold(self, quantity, entity, fields, case_id, time, index):
        self.fold_tile(quantity, entity, {name: np.asarray(value)[None, ...] for name, value in fields.items()},
                       ((case_id, time, index),))

    def recover_block(self, snapshot, full, loads, labels):
        # Bounded tile: one element and one physical case; no model-wide case dict.
        with owned_recovery_scope(tuple(e.recipe for e in self.plan.elements)):
            for element in self.plan.elements:
                for j, (case_id, time, index) in enumerate(labels):
                    try:
                        fields = _numeric_fields(_recover_owned_local(element.recipe, full[element.dof_mapping, j]), element.components)
                        self.fold('stress', element.element_id, fields, case_id, time, index)
                    except Exception as error:
                        from .control import SolveCancelled
                        if isinstance(error, SolveCancelled):
                            raise
                        raise PreparedStaticError('canonical envelope element recovery failed', diagnostics={
                            'stage': 'envelope recovery', 'case_id': case_id, 'input_index': index,
                            'element_id': element.element_id, 'error': str(error)}) from error
            for j, (case_id, time, index) in enumerate(labels):
                try:
                    if self.plan.include_displacements:
                        for node in self.plan.node_ids:
                            self.fold('displacements', node, {'displacements': full[snapshot.load_maps.node_dofs[node], j]}, case_id, time, index)
                    if self.plan.include_reactions:
                        forces = constraint_force_diagnostics(snapshot.reaction_maps, snapshot.stiffness,
                            snapshot.transformation, snapshot.rigid_modes, full[:, j], loads[:, j],
                            constraint_info=snapshot.constraint_info, nullspace_info=snapshot.nullspace_info)
                        for node in self.plan.node_ids:
                            for bucket in ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces'):
                                value = np.asarray(forces[bucket].get(node, np.zeros(6)), dtype=float)
                                self.fold(bucket, node, {'force': value}, case_id, time, index)
                except Exception as error:
                    from .control import SolveCancelled
                    if isinstance(error, SolveCancelled):
                        raise
                    raise PreparedStaticError('canonical envelope nodal recovery failed', diagnostics={
                        'stage': 'envelope recovery', 'case_id': case_id, 'input_index': index,
                        'error': str(error)}) from error
        self.cases_processed += len(labels)

    def result(self, identity):
        entries = []
        for (quantity, entity, name), (shape, modes) in self.entries.items():
            def detached(arrays):
                return PreparedGoverningValues(arrays['values'].reshape(shape).copy(),
                    arrays['case_ids'].reshape(shape).copy(), arrays['times'].reshape(shape).copy(),
                    arrays['input_indices'].reshape(shape).copy(),
                    {key: value.reshape(shape + value.shape[1:]).copy() for key, value in arrays['associated_fields'].items()})
            entries.append(PreparedEnvelopeExtrema(quantity, entity, name, shape, *(detached(mode) for mode in modes)))
        return PreparedEnvelopeResult(self.complete, self.cases_processed, tuple(entries), identity,
            {'recovery': 'canonical_owned_local_every_sample', 'tie_policy': 'exact_earliest_input_index',
             'retained_accumulator_bytes': self.nbytes(), 'tile_elements': 1, 'tile_cases': 8,
             'transaction_copied_modes': self.copied_mode_count,
             'execution_route': 'prepared_q4_stages_with_native_fallback',
             'prepared_q4_staged_elements': list(self.staged_elements),
             'native_fallback_reasons': dict(self.fallback_reasons),
             'vectorized_native_kernel': False})
