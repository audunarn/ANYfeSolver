"""Source-exact numeric publication for certified response capsule exports.

Only publication changes. Every matrix-vector result is checked against the
original complete per-element source operation before it is consumed. Derived
measures retain their source scalar/array types and expressions.
"""
from dataclasses import dataclass
import time
import numpy as np

from .prepared_static_recovery import _derived_measure, constraint_force_diagnostics, owned_recovery_scope
from .prepared_static_shared import _selected_reactions, _recovery_metadata, _selected_record
from .prepared_static_loads import PreparedStaticError
from .control import SolveCancelled

ELEMENT_TILE = 64


def float_bits_equal(left, right):
    """Float64 equality includes signed zero; arithmetic is never changed."""
    left, right = np.asarray(left), np.asarray(right)
    return (left.dtype == right.dtype == np.float64 and left.shape == right.shape
            and np.array_equal(left.view(np.uint64), right.view(np.uint64)))


@dataclass(frozen=True, slots=True)
class NumericField:
    element_ids: tuple
    shapes: tuple
    offsets: np.ndarray
    values: np.ndarray
    array_flags: tuple = ()


@dataclass(frozen=True, slots=True)
class NumericCase:
    metadata: dict
    fields: dict
    node_ids: tuple
    displacements: np.ndarray | None

    def get(self, name, default=None):
        return self.metadata.get(name, default)

    def __contains__(self, name):
        return name in self.metadata

    def __getitem__(self, name):
        return self.metadata[name]


def prepare_numeric_plan(snapshot, library, recovery, native_fallbacks=None):
    """Source-bound layouts once; no field-shape dictionaries per case."""
    detailed = snapshot.recovery_compilation.elements if recovery.include_stresses and snapshot.recovery_compilation is not None else {}
    selected = recovery.selected_components()
    groups = {}
    field_entries = {}
    native_fallbacks = {} if native_fallbacks is None else native_fallbacks
    for eid in sorted(detailed):
        native = eid in native_fallbacks
        entry = native_fallbacks[eid] if native else library[eid]
        compiled = detailed[eid]
        # Detailed source library records also include numeric constant rows
        # (e.g. beam3 station coordinates), even when affine compilation stores
        # those separately from varying array_components.
        names = tuple(dict.fromkeys(tuple(name for name, *_ in entry[1]) + entry[3]
                                    + tuple(name for name, *_ in compiled.derived_components)))
        if selected is not None:
            names = tuple(name for name in names if name in selected)
        # _selected_record derives requested fields absent from the canonical
        # library record (notably principal top/bottom ranks).
        derived = entry[3] + tuple(name for name, *_ in compiled.derived_components if name not in entry[3])
        key = (entry[1], derived, names)
        if native or key not in groups:
            prototype = source_fields(entry[4] if native else entry[4][:, 0], entry[1], derived)
            shapes = {name:(tuple(np.asarray(prototype[name]).shape),isinstance(prototype[name],np.ndarray)) for name in names}
            if not native:groups[key] = [[], shapes]
        else:shapes=groups[key][1]
        if not native:groups[key][0].append(eid)
        for name, (shape, flag) in shapes.items():
            field_entries.setdefault(name, []).append((eid, shape, flag))
    layouts = {}
    for name, entries in field_entries.items():
        entries.sort(key=lambda item:item[0])
        ids = tuple(eid for eid, _, _ in entries)
        shapes = tuple(shape for _, shape, _ in entries)
        flags = tuple(flag for _, _, flag in entries)
        offsets = np.concatenate(([0], np.cumsum([int(np.prod(shape)) if shape else 1 for shape in shapes],dtype=np.int64)))
        offsets.flags.writeable = False
        layouts[name] = ids, shapes, offsets, flags
    made = []
    for (linear, derived, names), (ids, shapes) in groups.items():
        slots = {}
        for name in names:
            field_ids, _, offsets, _ = layouts[name]
            slots[name] = offsets[np.searchsorted(field_ids, ids)]
            slots[name].flags.writeable = False
        made.append((tuple(ids), linear, derived, names, shapes, slots))
    output_bytes = sum(int(offsets[-1])*8 for _,_,offsets,_ in layouts.values())
    # Conservative metadata estimate; source column arrays are shared, not copied.
    retained_bytes = sum(offsets.nbytes+128*len(ids) for ids,_,offsets,_ in layouts.values()) + sum(array.nbytes for group in made for array in group[-1].values())
    return tuple(made), layouts, output_bytes, retained_bytes


def source_vectors(entries, packet, combine):
    """Original complete per-element GEMV, without a redundant grouped trial.

    The original GEMV remains the admission oracle for every element/input,
    not just preparation probes. Always consume those actual values; zero
    trial/fallback counters mean no grouped trial, not a proven equivalence.
    """
    matrices = [entry[4] for entry in entries]
    start = time.perf_counter()
    expected = np.stack([combine(matrix, packet) for matrix in matrices])
    oracle_seconds = time.perf_counter()-start
    return expected, 0, oracle_seconds, 0.


def source_fields(values, linear, derived):
    """Source measures per element, preserving Python scalar exponentiation."""
    fields, offset = {}, 0
    for name, shape, is_array in linear:
        size = int(np.prod(shape)) if shape else 1
        part = values[offset:offset + size]
        offset += size
        fields[name] = part.reshape(shape) if is_array else float(part[0])
    for name in derived:
        fields[name] = _derived_measure(name, fields)
    return fields


def recover_numeric_cases(snapshot, full, loads, recovery, labels, certified,
                          library, packets, combine, plan, safe_point,
                          native_fallbacks=None,native_recover=None):
    """No per-case element dictionaries; bounded temporary element views.

    Complete source vectors and derived measures are consumed immediately into
    component rows. Nodal/force diagnostics use the unchanged source services.
    """
    detailed = snapshot.recovery_compilation.elements if recovery.include_stresses and snapshot.recovery_compilation is not None else {}
    groups, layouts, _, _ = plan
    native_fallbacks = {} if native_fallbacks is None else native_fallbacks
    native_ids = tuple(eid for eid in sorted(native_fallbacks) if eid in detailed)
    provenance, execution = _recovery_metadata(snapshot.recovery_compilation if detailed else None)
    results = []
    statistics = {'physical_element_evaluations':0, 'reaction_diagnostic_evaluations':0,
                  'reconstructed_element_records':0, 'numeric_element_cases':0,
                  'numeric_kernel_fallback_tiles':0,'numeric_grouped_trial_tiles':0,
                  'numeric_vector_execution':'original_element_gemv',
                  'source_oracle_seconds':0.,'grouped_trial_seconds':0.,
                  'derived_source_seconds':0.,'field_publication_seconds':0.,'nodal_source_seconds':0.}
    statistics.update(native_affinity_element_cases=0,native_affinity_seconds=0.)
    # One case's rows at a time; no dictionary of records across elements.
    for column, packet in enumerate(packets):
        metadata = {'status':'converged', 'solver_info':certified[column]}
        if detailed:
            import copy
            certified[column]['stress_recovery'] = copy.deepcopy(provenance)
            certified[column]['recovery'] = copy.deepcopy(execution)
        rows = {name:np.empty(int(offsets[-1]),dtype=np.float64) for name,(_,_,offsets,_) in layouts.items()}
        for ids, linear, derived, names, shapes, slots in groups:
            for start in range(0, len(ids), ELEMENT_TILE):
                safe_point()
                tile = ids[start:start + ELEMENT_TILE]
                entries = [library[eid] for eid in tile]
                values, fallback, oracle_seconds, grouped_seconds = source_vectors(entries, packet, combine)
                statistics['source_oracle_seconds'] += oracle_seconds
                statistics['grouped_trial_seconds'] += grouped_seconds
                statistics['numeric_kernel_fallback_tiles'] += fallback
                fields, offset = {}, 0
                for name, shape, _is_array in linear:
                    size = int(np.prod(shape)) if shape else 1
                    fields[name] = values[:,offset:offset+size]
                    offset += size
                # Exact source measures, retaining scalar types and original
                # per-element LAPACK shapes. Temporary mappings never escape.
                measure_start = time.perf_counter()
                for position, eid in enumerate(tile):
                    try:
                        original = source_fields(values[position], linear, derived)
                        for name in derived:
                            value = np.asarray(original[name],dtype=np.float64).reshape(-1)
                            if name not in fields:
                                fields[name] = np.empty((len(tile),len(value)),dtype=np.float64)
                            fields[name][position] = value
                    except Exception as error:
                        raise PreparedStaticError('shared canonical element recovery failed',diagnostics={'stage':'case recovery','case_id':labels[column][0],'input_index':labels[column][2],'element_id':eid,'error':str(error)}) from error
                statistics['derived_source_seconds'] += time.perf_counter()-measure_start
                publish_start = time.perf_counter()
                for name in names:
                    value = fields[name]
                    if not np.all(np.isfinite(value)):
                        raise PreparedStaticError('numeric source recovery is nonfinite', diagnostics={'case_id':labels[column][0],'input_index':labels[column][2],'component':name})
                    indices = slots[name][start:start+len(tile),None] + np.arange(value.shape[1])
                    rows[name][indices] = value
                statistics['numeric_element_cases'] += len(tile)
                statistics['field_publication_seconds'] += time.perf_counter()-publish_start
        for start in range(0,len(native_ids),ELEMENT_TILE):
            safe_point(native=True)
            tile=native_ids[start:start+ELEMENT_TILE]
            native_started=time.perf_counter()
            with owned_recovery_scope(tuple(native_fallbacks[eid][7] for eid in tile)):
                for eid in tile:
                    fallback=native_fallbacks[eid]
                    try:
                        canonical=native_recover(fallback,full[fallback[6],column])
                        selected=_selected_record(detailed[eid],canonical)
                    except (PreparedStaticError,SolveCancelled):
                        raise
                    except Exception as error:
                        raise PreparedStaticError('native affinity source recovery failed',diagnostics={'stage':'case recovery','case_id':labels[column][0],
                            'input_index':labels[column][2],'element_id':eid,'error':str(error)}) from error
                    for name,(ids,shapes,offsets,_flags) in layouts.items():
                        position=int(np.searchsorted(ids,eid))
                        if position>=len(ids) or ids[position]!=eid:continue
                        try:value=np.asarray(selected[name],dtype=np.float64)
                        except KeyError as error:
                            raise PreparedStaticError('native affinity selected field is missing',diagnostics={'stage':'case recovery',
                                'case_id':labels[column][0],'input_index':labels[column][2],'element_id':eid,'component':name}) from error
                        if tuple(value.shape)!=shapes[position] or not np.all(np.isfinite(value)):
                            raise PreparedStaticError('native affinity recovery field is invalid',diagnostics={'case_id':labels[column][0],'input_index':labels[column][2],'element_id':eid,'component':name})
                        rows[name][offsets[position]:offsets[position+1]]=value.reshape(-1)
                    statistics['native_affinity_element_cases']+=1
                    statistics['physical_element_evaluations']+=1
            safe_point(native=True)
            statistics['native_affinity_seconds']+=time.perf_counter()-native_started
        fields = {}
        for name, values in rows.items():
            ids, shapes, offsets, flags = layouts[name]
            values.flags.writeable = False
            fields[name] = NumericField(ids, shapes, offsets, values, flags)
        node_start = time.perf_counter()
        node_ids = tuple(snapshot.selected_node_ids)
        displacement = None
        if recovery.include_displacements:
            rows = np.concatenate([snapshot.load_maps.node_dofs[node] for node in node_ids]) if node_ids else np.empty(0,dtype=np.intp)
            displacement = full[rows,column].copy()
            displacement.flags.writeable = False
        if recovery.include_reactions:
            forces = constraint_force_diagnostics(snapshot.reaction_maps, snapshot.stiffness,
                snapshot.transformation, snapshot.rigid_modes, full[:,column],loads[:,column],
                constraint_info=snapshot.constraint_info,nullspace_info=snapshot.nullspace_info)
            _selected_reactions(snapshot,recovery,forces,metadata)
            statistics['reaction_diagnostic_evaluations'] += 1
        statistics['nodal_source_seconds'] += time.perf_counter()-node_start
        results.append(NumericCase(metadata, fields, node_ids, displacement))
    return results, None, statistics
