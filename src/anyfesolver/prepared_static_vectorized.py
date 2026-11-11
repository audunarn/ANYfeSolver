"""M17 whole-model vectorized certified recovery and array envelopes (opt-in).

Used only by the response evaluator with stress_execution="certified_vectorized".
Each admitted element's affine recovery map H is probed once from its owned
canonical recipe and verified. A block evaluates every element and case with one
batched product per schema group, recomputes derived measures vectorially and
folds whole-model envelopes into arrays with the existing earliest-input exact
tie rule and coherent associated fields. Reactions keep canonical Ku-F code.
Outputs are qualified by assessment against canonical recovery at the same
certified state (M16 criteria C1-C6); this is not a bitwise route.
"""
from __future__ import annotations

import copy
import pickle
import sys
from dataclasses import replace

import numpy as np
from anyfesolver_results.formulas import (shell_surface_von_mises as _shared_shell_von_mises,
                                       beam_von_mises as _shared_beam_von_mises)
from scipy import sparse

from .prepared_static_loads import PreparedStaticError, PreparedStaticCapabilityError
from .analysis_session import _immutable_array
from .prepared_static_loads import evaluate_load_vector, _require_family
from .prepared_static_loads import (_capture_originals, _originals_unchanged, _default_error_state,
                                    _standard_float_environment)
from .prepared_static_recovery import (_recover_owned_local, _recover_owned_local_columns,
    owned_recovery_scope, _array_value, planned_constraint_force_diagnostics,
    _plain_constant, _constant_equal, _AFFINITY_RELATIVE_TOLERANCE, constraint_force_diagnostics,
    constraint_force_maps)
from .prepared_static_measures import PRINCIPAL_COMPONENTS, SHELL_LINEAR_COMPONENTS, principal_stresses
from .prepared_static_envelopes import (PreparedGoverningValues, PreparedEnvelopeExtrema,
    PreparedEnvelopeResult, _numeric_fields)
from .prepared_static_shared import _selected_record, _selected_reactions, _recovery_metadata
from .recovery import _filter_components

DERIVED_ORDER = ('von_mises', 'equivalent_stress', 'hill_utilization') + tuple(PRINCIPAL_COMPONENTS)
BEAM_FIELDS = ('axial_stress', 'bending_stress_y', 'bending_stress_z',
               'shear_stress_y', 'shear_stress_z', 'torsional_stress')
BUCKETS = ('support_reactions', 'mpc_slave_forces', 'mpc_master_equivalent_forces')
SAFE_POINT_STRIDE = 128


# --------------------------------------------------------------------------- derived

def _shell_von_mises(fields):
    return _shared_shell_von_mises({name:np.asarray(fields[name],dtype=float)
                                  for name in SHELL_LINEAR_COMPONENTS})


def _beam_von_mises(fields):
    return _shared_beam_von_mises(fields)


def _surface_tensor(fields, surface):
    mx, my, mxy, bx, by, bxy, sxz, syz = np.broadcast_arrays(*(np.asarray(fields[name], dtype=float)
                                                               for name in SHELL_LINEAR_COMPONENTS))
    sign = 1.0 if surface == 'top' else -1.0
    tensor = np.zeros(mx.shape + (3, 3))
    tensor[..., 0, 0] = mx + sign * bx
    tensor[..., 1, 1] = my + sign * by
    tensor[..., 0, 1] = tensor[..., 1, 0] = mxy + sign * bxy
    tensor[..., 0, 2] = tensor[..., 2, 0] = sxz
    tensor[..., 1, 2] = tensor[..., 2, 1] = syz
    return tensor


def derived_fields(names, fields):
    """Vectorized derived measures in dependency order; KeyError if not reviewable."""
    made = {}
    for name in DERIVED_ORDER:
        if name not in names:
            continue
        if name == 'von_mises':
            if all(key in fields for key in SHELL_LINEAR_COMPONENTS):
                made[name] = _shell_von_mises(fields)
            elif all(key in fields for key in BEAM_FIELDS):
                made[name] = _beam_von_mises(fields)
            else:
                raise KeyError(name)
        elif name == 'equivalent_stress':
            made[name] = np.array(made['von_mises'] if 'von_mises' in made else derived_fields(('von_mises',), fields)['von_mises'])
        elif name == 'hill_utilization':
            vm = made['von_mises'] if 'von_mises' in made else derived_fields(('von_mises',), fields)['von_mises']
            made[name] = np.zeros_like(vm)
        else:
            if not all(key in fields for key in SHELL_LINEAR_COMPONENTS):
                raise KeyError(name)
            made[name] = principal_stresses(_surface_tensor(fields, name.removeprefix('principal_stresses_')))
    return made


# --------------------------------------------------------------------------- schema and probes

def _refusal(element_id, message, **extra):
    return PreparedStaticCapabilityError('certified vectorized recovery: ' + message,
        diagnostics={'stage': 'vectorized recovery preparation', 'element_id': int(element_id), **extra})


def _schema(element_id, zero):
    linear, constants, derived = [], [], []
    offset = 0
    for name in zero:
        value = zero[name]
        content = _array_value(value)
        if name in DERIVED_ORDER:
            if content is None:
                raise _refusal(element_id, 'derived measure is not a float field', component=str(name))
            derived.append(name)
            continue
        if content is not None:
            shape = tuple(int(n) for n in content.shape)
            size = int(np.prod(shape)) if shape else 1
            linear.append((name, shape, type(value) is np.ndarray, offset, size))
            offset += size
            continue
        try:
            constants.append((name, _plain_constant(value)))
        except ValueError as error:
            raise _refusal(element_id, 'recovery component cannot be represented', component=str(name)) from error
    return tuple(zero), tuple(linear), tuple(constants), tuple(derived), offset


def _vector(record, linear, count):
    made = np.empty(count)
    for name, _shape, _is_array, offset, size in linear:
        made[offset:offset + size] = np.asarray(record[name], dtype=float).reshape(-1)
    return made


def _split(values, linear, leading=()):
    """values (..., n_out) -> {name: (..., *shape)}"""
    made = {}
    for name, shape, _is_array, offset, size in linear:
        made[name] = values[..., offset:offset + size].reshape(leading + shape)
    return made


def _relative(residual, terms):
    scale = max([float(np.linalg.norm(term)) for term in terms] + [1.0])
    return float(np.linalg.norm(residual)) / scale


def _probe_element(element_id, recipe, size, wanted_derived, zero=None):
    zero = _recover_owned_local(recipe, np.zeros(size)) if zero is None else zero
    order, linear, constants, derived, count = _schema(element_id, zero)
    shell = all(key in {n for n, *_ in linear} for key in SHELL_LINEAR_COMPONENTS)
    derived = tuple(name for name in DERIVED_ORDER
                    if name in derived or (shell and name in wanted_derived and name in PRINCIPAL_COMPONENTS))
    base = _vector(zero, linear, count)

    def observe(local):
        record = _recover_owned_local(recipe, local)
        if tuple(record) != order:
            raise _refusal(element_id, 'recovery component set changed under probes')
        for name, value in constants:
            if not _constant_equal(value, record[name]):
                raise _refusal(element_id, 'recovery constant changed under probes', component=str(name))
        for name, shape, _is_array, _offset, _size in linear:
            content = _array_value(record[name])
            if content is None or tuple(content.shape) != shape:
                raise _refusal(element_id, 'recovery field changed type or shape', component=str(name))
        return record

    matrix = _batched_unit_probes(recipe, size, linear, count, base, constants)
    batched = matrix is not None
    if matrix is None:
        matrix = np.empty((count, size))
        for k in range(size):
            unit = np.zeros(size)
            unit[k] = 1.0
            matrix[:, k] = _vector(observe(unit), linear, count) - base
    rng = np.random.default_rng(int(element_id) + 17)
    random = rng.uniform(-1.0, 1.0, size)
    rigid = np.zeros(size)
    rigid[0::6] = 0.125
    worst = 0.0
    for local in (random, 1e-3 * random + rigid):
        record = observe(local)
        observed = _vector(record, linear, count)
        predicted = base + matrix @ local
        # The prediction carries rounding relative to |H||v| + |r0| (rigid motion cancels
        # exactly in canonical recovery but only to rounding in the summed columns), so
        # affinity is judged against that internal magnitude as in the M16 criteria scale.
        magnitude = np.abs(matrix) @ np.abs(local) + np.abs(base)
        for name, _shape, _is_array, offset, width in linear:
            part = slice(offset, offset + width)
            relative = _relative(observed[part] - predicted[part], (observed[part], predicted[part], magnitude[part]))
            if not np.isfinite(relative) or relative > _AFFINITY_RELATIVE_TOLERANCE:
                raise _refusal(element_id, 'recovery field is not affine in displacement',
                               component=str(name), relative_residual=relative)
            worst = max(worst, relative)
        if derived:
            try:
                made = derived_fields(derived, _split(predicted, linear))
            except KeyError as error:
                raise _refusal(element_id, 'derived measure has no reviewed vectorized recomputation',
                               component=str(error)) from error
            for name in derived:
                if name in record:
                    native = np.asarray(record[name], dtype=float)
                else:
                    native = derived_fields((name,), _split(observed, linear))[name]
                value = np.asarray(made[name], dtype=float)
                if value.shape != native.shape:
                    raise _refusal(element_id, 'derived measure changed shape', component=str(name))
                relative = _relative(value - native, (value, native))
                if not np.isfinite(relative) or relative > _AFFINITY_RELATIVE_TOLERANCE:
                    raise _refusal(element_id, 'derived measure does not reproduce canonical recovery',
                                   component=str(name), relative_residual=relative)
                worst = max(worst, relative)
    return dict(zero=zero, order=order, linear=linear, constants=constants, derived=derived,
                count=count, base=base, matrix=matrix, worst=worst, batched=batched)


def _batched_unit_probes(recipe, size, linear, count, base, constants):
    """M21: all unit-probe columns of a staged Q4 recipe in one pass, else None.

    Linear fields the column pass does not produce must be stage constants
    (zero columns); anything else keeps the per-vector unit probes.
    """
    if getattr(recipe, 'staged_kernel', None) is None:
        return None
    fields = _recover_owned_local_columns(recipe, np.eye(size))
    if fields is None:
        return None
    stage_constants = {name for name, _value in recipe.staged_kernel.constants}
    matrix = np.zeros((count, size))
    for name, shape, _is_array, offset, width in linear:
        if name in fields:
            values = np.asarray(fields[name], dtype=float)
            if values.shape != tuple(shape) + (size,):
                return None
            matrix[offset:offset + width] = values.reshape(width, size) - base[offset:offset + width, None]
        elif name not in stage_constants:
            return None
    return matrix


class VectorRecovery:
    """Owned immutable grouped affine maps; no model, mesh or recipe is retained after build."""

    def __init__(self, groups, locations, element_info, probes, worst, envelope_sources):
        self.groups = groups                  # list of dicts with arrays
        self.locations = locations            # eid -> (group index, row)
        self.element_info = element_info      # eid -> (order, linear, constants, derived)
        self.probes = probes
        self.worst = worst
        self.envelope_sources = envelope_sources

    def nbytes(self):
        total = 0
        for group in self.groups:
            total += sum(group[key].nbytes for key in ('matrix', 'base', 'dofs', 'ids'))
        return int(total + 256 * len(self.locations))

    def case_workspace_bytes(self):
        return int(sum(8 * 3 * group['matrix'].shape[0] * (group['matrix'].shape[1] + group['matrix'].shape[2])
                       for group in self.groups))


def build_vector_recovery(union, envelope_plan, detailed_components, safe_point, memory_check=None):
    """union: {eid: (dof_mapping, recipe)} in a stable order.

    M22: an element's maps are allocated only after memory_check has accepted an
    estimate that includes them. M26: one pass over the chunks, pipelined: each
    owned scope probes the chunk whose storage was checked last and reads the zero
    records and schemas of the next chunk; memory_check runs between scopes on the
    running estimate (the last call sees the full estimate). This halves the safe
    points and owned-scope entries of the former two passes.
    """
    wanted = set()
    if envelope_plan is not None:
        for element in envelope_plan.elements:
            wanted.update(name for name in (element.components or ()) if name in PRINCIPAL_COMPONENTS)
    wanted.update(name for name in (detailed_components or ()) if name in PRINCIPAL_COMPONENTS)
    infos, probes, worst = {}, 0, 0.0
    items = list(union.items())
    zeros, estimate = {}, 0
    checked = []
    for start in range(0, len(items) + SAFE_POINT_STRIDE, SAFE_POINT_STRIDE):
        reading = items[start:start + SAFE_POINT_STRIDE]
        if not reading and not checked:
            break
        safe_point('prepared_response.vector_schema' if start == 0 else 'prepared_response.vector_recovery')
        with owned_recovery_scope(tuple(recipe for _eid, (_mapping, recipe) in checked + reading)):
            for eid, (mapping, recipe) in checked:
                info = _probe_element(eid, recipe, int(mapping.size), wanted, zeros.pop(eid))
                info['dofs'] = np.asarray(mapping, dtype=np.intp)
                infos[int(eid)] = info
                probes += (1 if info['batched'] else int(mapping.size)) + 3
                worst = max(worst, info['worst'])
            for eid, (mapping, recipe) in reading:
                zero = _recover_owned_local(recipe, np.zeros(int(mapping.size)))
                count = _schema(eid, zero)[4]
                zeros[eid] = zero
                # matrix + base + dofs, held per element, stacked and frozen (about 3x)
                estimate += 3 * 8 * (count * int(mapping.size) + count + int(mapping.size)) + 512
        if reading and memory_check is not None:
            memory_check(estimate)
        checked = reading
    if not items and memory_check is not None:
        memory_check(estimate)
    keyed = {}
    for eid, info in infos.items():
        key = (int(info['dofs'].size), tuple((n, s, a, o, z) for n, s, a, o, z in info['linear']), info['derived'])
        keyed.setdefault(key, []).append(eid)
    groups, locations = [], {}
    for (size, linear, derived), eids in keyed.items():
        index = len(groups)
        groups.append(dict(
            linear=linear, derived=derived,
            ids=_immutable_array(np.asarray(eids, dtype=np.int64)),
            matrix=_immutable_array(np.stack([infos[eid]['matrix'] for eid in eids])),
            base=_immutable_array(np.stack([infos[eid]['base'] for eid in eids])),
            dofs=_immutable_array(np.stack([infos[eid]['dofs'] for eid in eids]))))
        for row, eid in enumerate(eids):
            locations[eid] = (index, row)
    element_info = {eid: (info['order'], info['linear'], info['constants'], info['derived'])
                    for eid, info in infos.items()}
    envelope_sources = None
    if envelope_plan is not None:
        envelope_sources = _envelope_sources(envelope_plan, infos, locations)
    return VectorRecovery(groups, locations, element_info, probes, worst, envelope_sources)


def _envelope_sources(plan, infos, locations):
    """Per schema group: entity ids and, per field, its group/row or constant source."""
    grouped = {}
    for element in plan.elements:
        grouped.setdefault(element.schema, []).append(element)
    sources = []
    for schema, elements in grouped.items():
        fields = {}
        for name, shape in schema:
            parts = {}
            constant_positions, constant_values = [], []
            for position, element in enumerate(elements):
                info = infos[element.element_id]
                group, row = locations[element.element_id]
                linear_names = {n for n, *_ in info['linear']}
                if type(name) is str and (name in linear_names or name in info['derived']):
                    rows, positions = parts.setdefault(group, ([], []))
                    rows.append(row)
                    positions.append(position)
                    continue
                record = copy.deepcopy(info['zero'])
                value = _numeric_fields(record, element.components).get(name)
                if value is None:
                    raise _refusal(element.element_id, 'envelope field has no vectorized source', component=str(name))
                constant_positions.append(position)
                constant_values.append(np.asarray(value, dtype=float).reshape(shape))
            fields[name] = dict(
                shape=tuple(shape),
                parts=tuple((group, np.asarray(rows, dtype=np.intp), np.asarray(positions, dtype=np.intp))
                            for group, (rows, positions) in parts.items()),
                constant_positions=np.asarray(constant_positions, dtype=np.intp),
                constant_values=(np.stack(constant_values) if constant_values else None))
        sources.append(dict(schema=tuple(schema), entity_ids=np.asarray([e.element_id for e in elements], dtype=np.int64),
                            fields=fields))
    return tuple(sources)


# --------------------------------------------------------------------------- evaluation

GROUP_ELEMENT_CHUNK = 512
# M31d: the NumPy functions both orders call (and the guard uses), as loaded.
_GROUP_ORIGINALS = _capture_originals(
    (np, ('matmul', 'ascontiguousarray', 'empty', 'geterr', 'errstate', 'all', 'isfinite')))


def _column_values(matrix, base, dofs, full, columns):
    """The pre-M30 evaluation: one identically shaped batched product per case
    column (results must not depend on block size or neighbouring cases)."""
    values = np.empty((columns,) + base.shape)                                  # (C, E, n_out)
    for column in range(columns):
        local = np.ascontiguousarray(full[dofs, column])[:, :, None]           # (E, n_loc, 1)
        values[column] = np.matmul(matrix, local)[:, :, 0] + base
    return values


def _chunked_values(matrix, base, dofs, full, columns):
    """M30: the same per-element product per column over element chunks, so each
    chunk of the map is reused for every column of the block while it is in cache."""
    values = np.empty((columns,) + base.shape)
    for start in range(0, matrix.shape[0], GROUP_ELEMENT_CHUNK):
        stop = start + GROUP_ELEMENT_CHUNK
        chunk_matrix, chunk_base, chunk_dofs = matrix[start:stop], base[start:stop], dofs[start:stop]
        for column in range(columns):
            local = np.ascontiguousarray(full[chunk_dofs, column])[:, :, None]   # (e, n_loc, 1)
            values[column, start:stop] = np.matmul(chunk_matrix, local)[:, :, 0] + chunk_base
    return values


def evaluate_groups(vector, full):
    """full (dofs, C) -> per group {name: (C, E, *shape)} for linear and derived fields."""
    columns = full.shape[1]
    outputs = []
    for group in vector.groups:
        matrix, base, dofs = group['matrix'], group['base'], group['dofs']
        # M31b (review): the chunked order only with the NumPy functions as loaded
        # (M31d), under NumPy's default error state and (M31c) round to nearest
        # without flush-to-zero, with conditions silenced. There a condition the
        # state reports (overflow, invalid; nothing here divides; underflow is not
        # reported) always leaves an infinity or a NaN, and then the pre-M30 column
        # order is evaluated again, so warnings and exceptions are exactly those of
        # the pre-M30 code; otherwise the column order is taken directly.
        # (np.isfinite raises and warns for nothing, so one check serves both.)
        finite = False
        if (_originals_unchanged(_GROUP_ORIGINALS) and _default_error_state()
                and _standard_float_environment()):
            with np.errstate(all='ignore'):
                values = _chunked_values(matrix, base, dofs, full, columns)
            finite = bool(np.all(np.isfinite(values)))
        if not finite:
            values = _column_values(matrix, base, dofs, full, columns)
            finite = bool(np.all(np.isfinite(values)))
        if not finite:
            raise PreparedStaticError('nonfinite vectorized recovery values')
        elements = values.shape[1]
        fields = _split(values, group['linear'], (columns, elements))
        fields.update(derived_fields(group['derived'], fields))
        outputs.append(fields)
    return outputs


def element_record(vector, outputs, eid, column):
    group, row = vector.locations[eid]
    order, linear, constants, derived = vector.element_info[eid]
    fields = outputs[group]
    working = {}
    for name, _shape, is_array, _offset, _size in linear:
        value = fields[name][column, row]
        working[name] = np.array(value) if is_array else float(value)
    for name in derived:
        working[name] = np.array(fields[name][column, row])
    for name, value in constants:
        working[name] = copy.deepcopy(value)
    made = {name: working[name] for name in order if name in working}
    for name in derived:
        made.setdefault(name, working[name])
    return made


def envelope_fields(vector, outputs, source, columns):
    made = {}
    count = source['entity_ids'].size
    for name, spec in source['fields'].items():
        parts = spec['parts']
        if (spec['constant_values'] is None and len(parts) == 1 and parts[0][2].size == count
                and np.array_equal(parts[0][2], np.arange(count))
                and np.array_equal(parts[0][1], np.arange(outputs[parts[0][0]][name].shape[1]))):
            # M22: the group's array already holds every element in envelope order.
            made[name] = outputs[parts[0][0]][name]
            continue
        array = np.empty((columns, count) + spec['shape'])
        for group, rows, positions in spec['parts']:
            array[:, positions] = outputs[group][name][:, rows]
        if spec['constant_values'] is not None:
            array[:, spec['constant_positions']] = spec['constant_values'][None]
        made[name] = array
    return made


# --------------------------------------------------------------------------- array envelope

class VectorEnvelope:
    """Array accumulator with the EnvelopeAccumulator duck interface used by the evaluator."""

    def __init__(self, plan):
        self.plan = plan
        self.complete = False
        self.cases_processed = 0
        self.copied_mode_count = 0
        self.entries = {}
        self._copied = set()
        self._labels = {}
        self._nbytes = None
        grouped = {}
        for element in plan.elements:
            grouped.setdefault(element.schema, []).append(element.element_id)
        self.stress_groups = []
        for schema, ids in grouped.items():
            key = ('stress', len(self.stress_groups))
            self.stress_groups.append((key, tuple(schema), np.asarray(ids, dtype=np.int64)))
            for name, shape in schema:
                self._allocate(key + (name,), len(ids), tuple(shape), tuple(schema))
        nodes = len(plan.node_ids)
        if plan.include_displacements:
            self._allocate(('displacements', 0, 'displacements'), nodes, (6,), (('displacements', (6,)),))
        if plan.include_reactions:
            for bucket in BUCKETS:
                self._allocate((bucket, 0, 'force'), nodes, (6,), (('force', (6,)),))

    def _allocate(self, key, count, shape, schema):
        size = int(np.prod(shape)) if shape else 1
        modes = []
        for _ in range(3):
            modes.append({'values': np.full((count, size), np.nan),
                          'case_ids': np.full((count, size), None, dtype=object),
                          'times': np.full((count, size), None, dtype=object),
                          'input_indices': np.full((count, size), -1, dtype=np.int64),
                          'associated_fields': {field: np.full((count, size) + tuple(fieldshape), np.nan)
                                                for field, fieldshape in schema}})
        self.entries[key] = (shape, schema, modes)

    def nbytes(self):
        if self._nbytes is None:
            total = sys.getsizeof(self.entries)
            for _shape, _schema, modes in self.entries.values():
                for mode in modes:
                    total += sum(mode[name].nbytes for name in ('values', 'case_ids', 'times', 'input_indices'))
                    total += sum(value.nbytes for value in mode['associated_fields'].values())
            total += sum(sys.getsizeof(label) for label in self._labels.values())
            self._nbytes = int(total)
        return self._nbytes

    def transaction(self):
        made = object.__new__(type(self))
        made.plan = self.plan
        made.complete = False
        made.cases_processed = self.cases_processed
        made.copied_mode_count = 0
        made.entries = dict(self.entries)
        made._copied = set()
        made._labels = dict(self._labels)
        made._nbytes = None
        made.stress_groups = self.stress_groups
        return made

    def _writable(self, key, mode):
        if (key, mode) not in self._copied:
            shape, schema, modes = self.entries[key]
            modes = list(modes)
            old = modes[mode]
            made = {name: value.copy() for name, value in old.items() if name != 'associated_fields'}
            made['associated_fields'] = {name: value.copy() for name, value in old['associated_fields'].items()}
            modes[mode] = made
            self.entries[key] = (shape, schema, modes)
            self._copied.add((key, mode))
            self.copied_mode_count += 1
        return self.entries[key][2][mode]

    def fold(self, key, fields, labels):
        """fields: {name: (C, N, *shape)} for every schema field of this entry group."""
        shape, schema, modes = self.entries[key]
        name = key[2]
        values = np.asarray(fields[name], dtype=float)
        columns, count = values.shape[:2]
        if not np.all(np.isfinite(values)):
            raise PreparedStaticError('vectorized envelope values are not finite')
        flat = values.reshape(columns, count, -1)
        indices = np.asarray([label[2] for label in labels], dtype=np.int64)
        case_ids = np.empty(columns, dtype=object)
        times = np.empty(columns, dtype=object)
        for j, (case_id, time, _index) in enumerate(labels):
            case_ids[j] = case_id
            times[j] = time
            self._labels.setdefault(id(case_id), case_id)
            if time is not None:
                self._labels.setdefault(id(time), time)
        entity = np.arange(count)[:, None]
        for mode in range(3):
            keyed = np.abs(flat) if mode == 2 else flat
            winners = np.argmin(keyed, axis=0) if mode == 0 else np.argmax(keyed, axis=0)   # (N, size)
            new = np.take_along_axis(flat, winners[None], 0)[0]
            new_index = indices[winners]
            arrays = modes[mode]
            old, old_index = arrays['values'], arrays['input_indices']
            better = new < old if mode == 0 else new > old if mode == 1 else np.abs(new) > np.abs(old)
            tie = new == old if mode < 2 else np.abs(new) == np.abs(old)
            update = (old_index < 0) | better | (tie & (new_index < old_index))
            if not np.any(update):
                continue
            arrays = self._writable(key, mode)
            modes = self.entries[key][2]
            arrays['values'][update] = new[update]
            arrays['input_indices'][update] = new_index[update]
            arrays['case_ids'][update] = case_ids[winners][update]
            arrays['times'][update] = times[winners][update]
            for field, _fieldshape in schema:
                source = np.asarray(fields[field], dtype=float)
                gathered = source[winners, entity]          # (N, size, *fieldshape)
                arrays['associated_fields'][field][update] = gathered[update]
        self._nbytes = None

    def result(self, identity):
        entries = []
        for key, (shape, schema, modes) in self.entries.items():
            quantity = key[0]
            if quantity == 'stress':
                ids = next(group[2] for group in self.stress_groups if group[0] == key[:2])
            else:
                ids = np.asarray(self.plan.node_ids, dtype=np.int64)
            for row, entity in enumerate(ids):
                def detached(arrays):
                    return PreparedGoverningValues(arrays['values'][row].reshape(shape).copy(),
                        arrays['case_ids'][row].reshape(shape).copy(), arrays['times'][row].reshape(shape).copy(),
                        arrays['input_indices'][row].reshape(shape).copy(),
                        {field: value[row].reshape(shape + value.shape[2:]).copy()
                         for field, value in arrays['associated_fields'].items()})
                entries.append(((quantity, int(entity), key[2]), PreparedEnvelopeExtrema(
                    quantity, int(entity), key[2], shape, *(detached(mode) for mode in modes))))
        # Match the per-entity accumulator order: elements in plan order, then nodes.
        order = {}
        for element in self.plan.elements:
            for name, _shape in element.schema:
                order[('stress', int(element.element_id), name)] = len(order)
        for node in self.plan.node_ids:
            if self.plan.include_displacements:
                order[('displacements', int(node), 'displacements')] = len(order)
            if self.plan.include_reactions:
                for bucket in BUCKETS:
                    order[(bucket, int(node), 'force')] = len(order)
        entries.sort(key=lambda item: order[item[0]])
        return PreparedEnvelopeResult(self.complete, self.cases_processed, tuple(entry for _key, entry in entries), identity,
            {'recovery': 'certified_vectorized_compiled_affine', 'tie_policy': 'exact_earliest_input_index',
             'governing_identity_basis': 'certified_response_values',
             'retained_accumulator_bytes': self.nbytes(), 'transaction_copied_modes': self.copied_mode_count,
             'execution_route': 'vectorized_schema_groups', 'vectorized_native_kernel': False})


def _selected_reactions_lean(snapshot, recovery, forces, item):
    """M24 opt-in: selected reactions and force maps only (same values as the full path)."""
    combined = {}
    for bucket in ('support_reactions', 'mpc_slave_forces'):
        for node, force in forces[bucket].items():
            combined.setdefault(int(node), np.zeros(6))
            combined[int(node)] += force
    selected = set(snapshot.selected_node_ids)
    keep = (lambda node: True) if recovery.node_ids is None else (lambda node: node in selected)
    item['reactions'] = {node: value for node, value in combined.items() if keep(node)}
    item['reaction_node_ids'] = list(snapshot.selected_node_ids)
    made = {'diagnostics': 'lean_selected_reaction_maps'}
    for bucket in BUCKETS:
        made[bucket] = {node: value for node, value in forces[bucket].items() if keep(node)}
    item['constraint_forces'] = made


def recover_vector_block(snapshot, full, loads, recovery, labels, certified, envelope, vector, compilation,
                         metadata_copies=None, reaction_plan=None, lean_selected=False):
    columns = full.shape[1]
    outputs = evaluate_groups(vector, full)
    results = [{'status': 'converged', 'solver_info': info, 'node_displacements': {},
                'element_stresses': {}, 'reactions': {}} for info in certified]
    detailed = {} if not recovery.include_stresses or compilation is None else compilation.elements
    if detailed:
        provenance, execution = _recovery_metadata(compilation)
        execution = dict(execution, execution_route='certified_vectorized_compiled_affine')
        for info in certified:
            info['stress_recovery'] = copy.deepcopy(provenance)
            info['recovery'] = copy.deepcopy(execution)
        selected = recovery.selected_components()
        for eid, compiled in detailed.items():
            for column in range(columns):
                record = element_record(vector, outputs, eid, column)
                results[column]['element_stresses'][eid] = _filter_components(_selected_record(compiled, record), selected)
    transaction = None if envelope is None else envelope.transaction()
    statistics = {'physical_element_evaluations': 0, 'vectorized_groups': len(vector.groups),
                  'vectorized_element_cases': len(vector.locations) * columns,
                  'reaction_diagnostic_evaluations': 0, 'max_case_tile': columns}
    if transaction is not None:
        for (key, schema, ids), source in zip(transaction.stress_groups, vector.envelope_sources):
            fields = envelope_fields(vector, outputs, source, columns)
            for name, _shape in schema:
                transaction.fold(key + (name,), fields, labels)
    need_forces = recovery.include_reactions or (envelope is not None and envelope.plan.include_reactions)
    forces_per_case = []
    for column in range(columns):
        if recovery.include_displacements:
            for node in snapshot.selected_node_ids:
                results[column]['node_displacements'][node] = full[snapshot.load_maps.node_dofs[node], column].copy()
        if need_forces and reaction_plan is not None:
            # M22: planned diagnostics, identical to constraint_force_diagnostics.
            forces = planned_constraint_force_diagnostics(reaction_plan, snapshot.stiffness,
                snapshot.transformation, snapshot.rigid_modes, full[:, column], loads[:, column],
                constraint_info=snapshot.constraint_info, nullspace_info=snapshot.nullspace_info,
                metadata_copies=metadata_copies, maps_only=(not recovery.include_reactions) or lean_selected)
        elif need_forces and recovery.include_reactions:
            forces = constraint_force_diagnostics(snapshot.reaction_maps, snapshot.stiffness,
                snapshot.transformation, snapshot.rigid_modes, full[:, column], loads[:, column],
                constraint_info=snapshot.constraint_info, nullspace_info=snapshot.nullspace_info,
                metadata_copies=metadata_copies)
        elif need_forces:
            # Envelope-only reactions: the same force maps without diagnostics records.
            forces = constraint_force_maps(snapshot.reaction_maps, snapshot.stiffness,
                                           full[:, column], loads[:, column])
        if need_forces:
            statistics['reaction_diagnostic_evaluations'] += 1
            forces_per_case.append(forces)
            if recovery.include_reactions and lean_selected:
                _selected_reactions_lean(snapshot, recovery, forces, results[column])
            elif recovery.include_reactions:
                _selected_reactions(snapshot, recovery, forces, results[column])
    if transaction is not None:
        nodes = envelope.plan.node_ids
        if envelope.plan.include_displacements:
            dofs = np.stack([np.asarray(snapshot.load_maps.node_dofs[node]) for node in nodes]) if nodes else np.zeros((0, 6), dtype=np.intp)
            values = np.moveaxis(full[dofs], 2, 0) if nodes else np.zeros((columns, 0, 6))
            transaction.fold(('displacements', 0, 'displacements'), {'displacements': values}, labels)
        if envelope.plan.include_reactions:
            position = {int(node): i for i, node in enumerate(nodes)}
            for bucket in BUCKETS:
                values = np.zeros((columns, len(nodes), 6))
                for column, forces in enumerate(forces_per_case):
                    for node, force in forces[bucket].items():
                        index = position.get(int(node))
                        if index is not None:
                            values[column, index] = force
                transaction.fold((bucket, 0, 'force'), {'force': values}, labels)
        transaction.cases_processed += len(labels)
        statistics['transaction_copied_modes'] = transaction.copied_mode_count
    return results, transaction, statistics


# --------------------------------------------------------------------------- M20 helpers

def _structurally_identical(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is np.ndarray:
        return (left.dtype == right.dtype and left.shape == right.shape
                and left.flags.writeable == right.flags.writeable and np.array_equal(left, right))
    if type(left) is dict:
        return list(left) == list(right) and all(_structurally_identical(left[k], right[k]) for k in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_structurally_identical(a, b) for a, b in zip(left, right))
    return left == right or (left != left and right != right)


def make_metadata_copier(constraint_info, nullspace_info):
    """Fresh independent copies of plain snapshot metadata, equal to deepcopy output.

    The snapshot admits only plain containers, scalars and numeric arrays
    (prepared_static_snapshot._plain_copy), so a pickle made once from the owned
    snapshot reproduces deepcopy's structure. The equivalence is checked once;
    any difference keeps the deepcopy path (returns None).
    """
    reference = (copy.deepcopy(dict(constraint_info)), copy.deepcopy(dict(nullspace_info)))
    try:
        blob = pickle.dumps((dict(constraint_info), dict(nullspace_info)), protocol=4)
        trial = pickle.loads(blob)
    except Exception:
        return None
    if not _structurally_identical(tuple(trial), reference):
        return None
    return lambda: tuple(pickle.loads(blob))


class CompiledPressure:
    """Dead-pressure loads from the verified per-element bases as one sparse product.

    Columns hold activity_scale * basis scattered to global DOFs. Only cases whose
    every pressure entry has a compiled basis of its kind are evaluated here;
    anything else (follower, missing basis, refused element) returns None and the
    caller uses the native kernel. Rounding differs from the native quadrature
    order; agreement is verified (criterion C7) before use.
    """

    def __init__(self, maps):
        rows, cols, values, index, column = [], [], [], {}, 0
        for element_id, pressure_map in maps.pressure_maps.items():
            mapping = np.asarray(pressure_map.dof_mapping, dtype=np.intp)
            scale = float(pressure_map.activity_scale)
            if pressure_map.scalar_basis is not None:
                basis = np.asarray(pressure_map.scalar_basis, dtype=float).reshape(-1)
                index[(int(element_id), 'scalar')] = (column, 1)
                rows.append(mapping)
                cols.append(np.full(mapping.size, column))
                values.append(scale * basis)
                column += 1
            if pressure_map.tuple_basis is not None:
                basis = np.asarray(pressure_map.tuple_basis, dtype=float)
                index[(int(element_id), 'tuple')] = (column, basis.shape[1])
                for k in range(basis.shape[1]):
                    rows.append(mapping)
                    cols.append(np.full(mapping.size, column + k))
                    values.append(scale * basis[:, k])
                column += basis.shape[1]
        total = int(maps.total_dofs)
        if column:
            matrix = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                                       shape=(total, column))
        else:
            matrix = sparse.csr_matrix((total, 0))
        self.index = index
        self.columns = column
        self.matrix = matrix
        self.abs_matrix = abs(matrix)
        for made in (self.matrix, self.abs_matrix):
            for array in (made.data, made.indices, made.indptr):
                array.setflags(write=False)
        # Pressure kinds exercised by the preparation C7 check; others stay native.
        self.verified_kinds = frozenset()
        self.refused = frozenset(int(eid) for eid, reasons in dict(maps.refused_elements).items()
                                 if any(kind == 'pressure' for kind, _reason in reasons))

    def nbytes(self):
        return int(sum(a.nbytes for m in (self.matrix, self.abs_matrix) for a in (m.data, m.indices, m.indptr)))

    def needs_native_pressure(self, owned):
        """True when evaluating this packet would run the native pressure kernel."""
        if not owned.pressure:
            return False
        if owned.follower:
            return True
        for element_id, (kind, value) in owned.pressure:
            entry = self.index.get((int(element_id), kind))
            if int(element_id) in self.refused or entry is None or kind not in self.verified_kinds:
                return True
            if kind != 'scalar' and np.asarray(value).size != entry[1]:
                return True
        return False

    def pressure_values(self, owned, *, kinds=None):
        if owned.follower or not owned.pressure:
            return None
        kinds = self.verified_kinds if kinds is None else kinds
        values = np.zeros(self.columns)
        for element_id, (kind, value) in owned.pressure:
            element_id = int(element_id)
            if element_id in self.refused:
                return None
            entry = self.index.get((element_id, kind))
            if entry is None or kind not in kinds:
                return None
            column, width = entry
            if kind == 'scalar':
                values[column] += float(value)
            else:
                array = np.asarray(value, dtype=float).reshape(-1)
                if array.size != width:
                    return None
                values[column:column + width] += array
        return values

    def evaluate(self, maps, owned, *, kinds=None):
        """(load, info, pressure values, non-pressure load) or None for native evaluation."""
        values = self.pressure_values(owned, kinds=kinds)
        if values is None:
            return None
        _require_family(maps, 'pressure', owned.name or '?')
        other, info = evaluate_load_vector(maps, replace(owned, pressure=()))
        load = other + np.asarray(self.matrix @ values, dtype=float)
        if not np.all(np.isfinite(load)):
            return None
        info = dict(info, load_norm=float(np.linalg.norm(load)), load_execution='compiled_pressure_bases')
        return load, info, values, other

    def term_scale(self, values, other):
        return np.asarray(self.abs_matrix @ np.abs(values), dtype=float) + np.abs(other)


C7_RELATIVE = 1e-12
C7_TAU = 2.0 ** 10


def c7_ratio(compiled, native, scale):
    """Largest |compiled - native| / (1e-12 |native| + 2^10 u scale); <= 1 passes C7."""
    limit = C7_RELATIVE * np.abs(native) + C7_TAU * 2.0 ** -53 * scale
    error = np.abs(compiled - native)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(error == 0.0, 0.0, error / limit)
    return float(np.max(ratio, initial=0.0))
