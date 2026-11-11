"""Solver-owned export of ANYpost recovery capsules (contract v1).

``export_recovery_capsule`` consumes one open prepared response evaluator and
a batch of pattern cases, evaluates them in bounded certified increments and
emits the plain-data capsule mapping defined by the ANYpost implementation
contract (see ``docs/RESULT_CAPSULE.md``).  No certification is fabricated
here: a case is reconstructable only when the evaluator itself reconstructed
it on the admitted compiled basis, and every other certified case is
exported as explicit stored arrays with its actual route label and
provenance.

``export_recovery_capsule_to_sink`` drives the same bounded export through
an owner-neutral sink (see ``SINK_METHODS``): case result dictionaries are
consumed and discarded per batch, and the sink receives coefficients, case
records and per-component stored rows incrementally.  Emitted numeric rows
are released as soon as the sink has consumed them; only compact
signature/index metadata is retained, so streaming memory stays bounded by
one batch instead of the case count.  The in-memory
``export_recovery_capsule`` keeps its signature and delegates to an internal
sink; its memory is output-sized (stored arrays plus compact case records),
never all detailed case results.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from types import MappingProxyType
from sys import getsizeof as _observer_sizeof

from .prepared_static_loads import PreparedStaticError
from .prepared_static_response import (PatternCase, PreparedStaticResponseEvaluator,
    _require_response_authority, _callable_authority, _callable_authority_matches)
from .control import cancellation_safe_point
from .prepared_static_measures import PRINCIPAL_COMPONENTS
from .prepared_static_numeric import NumericCase, NumericField, float_bits_equal
from time import perf_counter_ns as _capsule_guard_clock_ns
from anyfesolver_results import (reconstruct_response, derive_field, ReconstructionError,
    DerivationError, validate_storage_prediction)
import anyfesolver_results as _storage_runtime

_STORAGE_SCHEMA_INPUTS = tuple(sorted(validate_storage_prediction.__globals__['_INPUTS'].items()))
_STORAGE_NUMPY_PINS = tuple((name,vars(np)[name],_callable_authority(vars(np)[name]))
                          for name in ('array_equal','nditer'))

def _require_storage_schema_authority():
    _require(_storage_runtime.validate_storage_prediction is validate_storage_prediction,
             'storage validator entrypoint changed after source capture')
    _require(type(_storage_runtime.STORAGE_PREDICTION_API) is int and _storage_runtime.STORAGE_PREDICTION_API == 1,
             'storage validator capability changed after source capture')
    _require(tuple(sorted(validate_storage_prediction.__globals__['_INPUTS'].items())) == _STORAGE_SCHEMA_INPUTS,
             'storage validator grammar changed after source capture')
    for name,function,facts in _STORAGE_NUMPY_PINS:
        _require(vars(np).get(name) is function and _callable_authority_matches(function,facts),
                 'storage validation NumPy binding changed after source capture')

_CAPSULE_RESPONSE_AUTHORITY = (_require_response_authority,
                              _callable_authority(_require_response_authority))

def _require_native_response_authority(evaluator):
    function, facts = _CAPSULE_RESPONSE_AUTHORITY
    _require(_callable_authority_matches(function, facts),
             'source response authority checker changed')
    function(evaluator)

SCHEMA = 'anysolver.recovery_capsule'
VERSION = 2
CONTRACT_ID = f'{SCHEMA}/{VERSION}'
RUNTIME_ID = 'anysolver-results/2'
DEFAULT_BATCH_SIZE = 32
PRINCIPAL_RANKS = (1, 2, 3)

#: Derived record fields that are never exported: the fake zero Hill
#: utilisation exists only as a placeholder and must not become a field.
EXCLUDED_DERIVED = ('hill_utilization',)

_SHELL_INPUTS = ('membrane_xx', 'membrane_yy', 'membrane_xy',
                'bending_xx', 'bending_yy', 'bending_xy', 'shear_xz', 'shear_yz')
_BEAM_INPUTS = ('axial_stress', 'bending_stress_y', 'bending_stress_z',
                'shear_stress_y', 'shear_stress_z', 'torsional_stress')

#: Nodal displacement components with their documented SI units
#: (translations m, rotations rad), matching the canonical
#: ``describe_result_quantities`` displacement descriptor.
DISPLACEMENT_COMPONENTS = (('UX', 'm'), ('UY', 'm'), ('UZ', 'm'),
                           ('RX', 'rad'), ('RY', 'rad'), ('RZ', 'rad'))
#: Nodal reaction components (force N, moment N*m), matching the canonical
#: reaction descriptor ``mixed:N,N*m``.
REACTION_COMPONENTS = (('FX', 'N'), ('FY', 'N'), ('FZ', 'N'),
                       ('MX', 'N*m'), ('MY', 'N*m'), ('MZ', 'N*m'))

#: Structural sink protocol (owner-neutral callbacks, no ANYpost import):
#: an object with these four callable attributes receives one bounded export.
SINK_METHODS = ('capsule_begin', 'capsule_case', 'capsule_stored', 'capsule_end')
FAILURE_SCHEMA = 'anysolver.recovery_capsule_failure'

_STRESS_QUANTITY = 'stress'
_STRESS_UNIT = 'Pa'
_STRESS_FIELDS = frozenset({
    'membrane_xx', 'membrane_yy', 'membrane_xy', 'bending_xx', 'bending_yy',
    'bending_xy', 'shear_xz', 'shear_yz', 'axial_stress', 'bending_stress_y',
    'bending_stress_z', 'shear_stress_y', 'shear_stress_z', 'torsional_stress',
    'von_mises', 'equivalent_stress',
})
#: Explicit SI unit conventions for non-stress element record fields.
#: Resultant force/moment quantities carry their documented units; strains
#: are dimensionless and are never classified as stress.
_FIELD_UNITS = {
    'membrane_strain': '1',
    'transverse_shear_strain': '1',
    'curvature': '1/m',
    'bubble_rotations': 'rad',
    'membrane_resultants': 'N/m',
    'bending_resultants': 'N*m/m',
    'transverse_shear_resultants': 'N/m',
    'reference_surface_offset': 'm',
    'section_origin_offset_from_reference': 'm',
    'physical_top_offset_from_reference': 'm',
    'physical_bottom_offset_from_reference': 'm',
}
_UNCLASSIFIED_UNIT = 'unclassified'


class RecoveryCapsuleError(PreparedStaticError):
    """Refusal of a recovery-capsule export request (no capsule is emitted)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryCapsuleError(message)


def _immutable(*args, **kwargs):
    raise TypeError('exported recovery capsule snapshots are read-only')


class _ReadOnlyDict(dict):
    """JSON-compatible snapshot; ordinary mapping mutation is refused."""
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        # Explicit cloning produces independent construction data. It must
        # not retain array aliases to the emitted immutable snapshot.
        from copy import deepcopy
        result = {}
        memo[id(self)] = result
        result.update((deepcopy(key, memo), deepcopy(value, memo))
                      for key, value in self.items())
        return result


class _ReadOnlyList(list):
    """JSON-compatible sequence retaining the public list equality contract."""
    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = sort = reverse = __iadd__ = __imul__ = _immutable

    def __deepcopy__(self, memo):
        from copy import deepcopy
        result = []
        memo[id(self)] = result
        result.extend(deepcopy(value, memo) for value in self)
        return result


def _readonly_data(value):
    """Freeze owned export data without copying large numeric response bases.

    Arrays are views backed by a read-only owned export array, so callers
    cannot re-enable writing on the emitted view. This is a normal Python
    snapshot contract, not a security boundary against deliberate access to
    private backing objects or builtin mutation bypasses.
    """
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
        view = value.view()
        view.setflags(write=False)
        return view
    if isinstance(value, Mapping):
        return _ReadOnlyDict({key: _readonly_data(item) for key, item in value.items()})
    if isinstance(value, list):
        return _ReadOnlyList(_readonly_data(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_readonly_data(item) for item in value)
    return value


def _require_observer_environment(_namespace=globals(),
        _bindings=tuple((name,name in globals(),globals().get(name)) for name in
            ('all','any','bool','bytes','dict','float','frozenset','getattr','hasattr','id','int','isinstance',
             'len','list','max','min','range','set','sorted','str','sum','tuple','type','zip')),
        _refuse=_require):
    # New observer helpers must not acquire an absent builtin shadow after
    # external observation; old producer/global policies are unchanged.
    for name,present,value in _bindings:
        _refuse((name in _namespace)==present and (not present or _namespace[name] is value),
                'source observer executable namespace changed')


def _observer_freeze(value):
    """Independent immutable snapshots; no writeable ndarray owner is exposed."""
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key:_observer_freeze(item) for key,item in value.items()})
    if isinstance(value, (list,tuple)):
        return tuple(_observer_freeze(item) for item in value)
    return value


def _observer_array_facts(value):
    """Own every view and the terminal bytes, not merely the immediate base."""
    facts=[]
    while type(value) is np.ndarray:
        base=value.base
        _require(not value.flags.owndata and value.flags.c_contiguous and not value.flags.writeable,
                 'source observer array layout changed')
        facts.append((value,base,value.dtype,value.shape,value.strides,
                      value.__array_interface__['data'][0]))
        value=base
    _require(type(value) is bytes,'source observer array backing changed')
    # The terminal bytes is a separate strong reference: __setstate__ on an
    # intermediate ndarray must not release the original buffer underneath
    # another view while the external callback is still running.
    return tuple(facts),value


def _require_observer_array_facts(facts):
    chain,terminal=facts
    for value,base,dtype,shape,strides,pointer in chain:
        _require(value.base is base and value.dtype==dtype and value.shape==shape
                 and value.strides==strides and value.flags.c_contiguous
                 and not value.flags.owndata and not value.flags.writeable
                 and value.__array_interface__['data'][0]==pointer,
                 'source observer array layout changed: backing chain')
    _require(bool(chain) and chain[-1][1] is terminal and type(terminal) is bytes,
             'source observer array backing changed')


def _observer_numeric_bytes(value, seen=None):
    """Conservative active numeric ownership, without walking model objects."""
    if seen is None:seen=set()
    if id(value) in seen:return 0
    seen.add(id(value))
    if isinstance(value,np.ndarray):return _observer_sizeof(value)+value.nbytes
    if isinstance(value,NumericCase):
        return _observer_sizeof(value)+_observer_numeric_bytes(value.metadata,seen)+_observer_numeric_bytes(value.displacements,seen)+_observer_numeric_bytes(value.node_ids,seen)+_observer_numeric_bytes(value.fields,seen)
    if isinstance(value,Mapping):return _observer_sizeof(value)+sum(_observer_numeric_bytes(key,seen)+_observer_numeric_bytes(item,seen) for key,item in value.items())
    if isinstance(value,(list,tuple)):return _observer_sizeof(value)+sum(_observer_numeric_bytes(item,seen) for item in value)
    if isinstance(value,NumericField):
        return _observer_sizeof(value)+sum(_observer_numeric_bytes(item,seen) for item in (value.element_ids,value.shapes,value.offsets,value.values,value.array_flags))
    return _observer_sizeof(value)


def _diagnostic_json(value):
    """Finite diagnostic JSON, explicitly tagging nonfinite diagnostic values.

    This never changes engineering data. A NaN/overflow mentioned in a
    failure must not itself prevent that failure's diagnostics being saved.
    """
    plain = _plain_json(value)

    def finite(item):
        if type(item) is float and not np.isfinite(item):
            label = 'NaN' if np.isnan(item) else ('+Infinity' if item > 0 else '-Infinity')
            return {'numeric_kind': 'nonfinite_float', 'value': label}
        if isinstance(item, dict):
            return {key: finite(child) for key, child in item.items()}
        if isinstance(item, list):
            return [finite(child) for child in item]
        return item

    return finite(plain)


def _report_export_failure(sink, error, cases, model_identity, mesh_identity):
    callback = getattr(sink, 'capsule_failed', None)
    if not callable(callback):
        return
    from .control import SolveCancelled
    record = {
        'schema': FAILURE_SCHEMA, 'version': 1, 'runtime': RUNTIME_ID,
        'model_identity': model_identity, 'mesh_identity': mesh_identity,
        'status': 'cancelled' if isinstance(error, (SolveCancelled, KeyboardInterrupt)) else 'failed',
        'error_type': type(error).__qualname__, 'error_module': type(error).__module__,
        'message': str(error),
        'input_case_ids': [case.case_id for case in cases]
        if isinstance(cases, (list, tuple)) and all(type(case) is PatternCase for case in cases)
        else [],
    }
    try:
        record['diagnostics'] = _diagnostic_json(getattr(error, 'diagnostics', {}))
    except Exception as diagnostic_error:
        # Preserve the primary error while explicitly identifying diagnostic
        # content that cannot be serialized. Never silently stringify it.
        record['diagnostics_unavailable'] = {
            'error_type': type(diagnostic_error).__qualname__,
            'message': str(diagnostic_error),
        }
    try:
        callback(_readonly_data(record))
    except BaseException as callback_error:
        if hasattr(error, 'add_note'):
            error.add_note('recovery export failure diagnostics could not be retained: '
                           + type(callback_error).__qualname__ + ': ' + str(callback_error))


def _plain_json(value: Any) -> Any:
    """Lossless JSON-safe copy of solver diagnostics and provenance values.

    Arrays are preserved as nested JSON arrays with their shape; nothing is
    truncated or stringified.  A value that cannot be represented in plain
    JSON is a typed refusal, never a silent loss.
    """
    if value is None or type(value) in (bool, str, int, float):
        return value
    if isinstance(value, np.generic):
        return _plain_json(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf':
            raise RecoveryCapsuleError(
                f'provenance arrays must be numeric, not dtype {value.dtype!r}')
        return [_plain_json(item) for item in value.tolist()]
    if type(value) is dict:
        return {str(key): _plain_json(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    raise RecoveryCapsuleError(
        f'provenance value of type {type(value).__name__!r} has no lossless JSON form')


def _field_vector(record: Mapping[str, Any], name: str) -> Optional[np.ndarray]:
    value = record.get(name)
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype.kind != 'f' or not np.all(np.isfinite(value)):
            return None
        return value.reshape(-1)
    if isinstance(value, np.generic) and value.dtype.kind in 'iuf':
        return np.asarray(float(value), dtype=float).reshape(1)
    if type(value) in (int, float):
        return np.asarray(float(value), dtype=float).reshape(1)
    return None


def _constant_json(value: Any) -> Any:
    """One element-record constant in explicit stored JSON form."""
    if isinstance(value, tuple):
        return [_constant_json(item) for item in value]
    if value is None or type(value) in (bool, str, int, float):
        return value
    if isinstance(value, np.generic):
        return _constant_json(value.item())
    return _plain_json(value)


def _is_constant_value(value: Any) -> bool:
    if value is None or type(value) in (bool, str):
        return True
    if isinstance(value, np.ndarray):
        # Float arrays are spatial fields; only non-numeric arrays are
        # record constants.
        return value.dtype.kind not in 'iuf'
    if isinstance(value, tuple):
        return True
    if isinstance(value, np.generic):
        return value.dtype.kind not in 'iuf'
    return False


def _quantity_of(name: str) -> Tuple[str, str, Dict[str, Any]]:
    """Canonical quantity id, SI unit and metadata for one element field."""
    if name in _STRESS_FIELDS or name.startswith('principal_stress_'):
        return _STRESS_QUANTITY, _STRESS_UNIT, {}
    if name in _FIELD_UNITS:
        return name, _FIELD_UNITS[name], {}
    return name, _UNCLASSIFIED_UNIT, {'unit_classification': 'solver_field_without_declared_unit'}


#: Coordinate-free scalar element fields (distances from a reference surface).
_INVARIANT_FIELD_NAMES = frozenset({
    'von_mises', 'equivalent_stress', 'reference_surface_offset',
    'section_origin_offset_from_reference', 'physical_top_offset_from_reference',
    'physical_bottom_offset_from_reference',
})


def _physical_basis(name: str) -> str:
    """Canonical physical coordinate basis of one component name.

    The semantics follow the solver's own quantity descriptors: tensor and
    vector components resolved in element axes are ``element_local``, nodal
    quantities are ``global``, principal eigenvalue components are
    ``principal`` and coordinate-free scalars are ``invariant``.  Fields the
    solver does not classify carry no basis claim.
    """
    if name.startswith('principal_stress_'):
        return 'principal'
    if name in _INVARIANT_FIELD_NAMES:
        return 'invariant'
    if name in _STRESS_FIELDS or name in _FIELD_UNITS:
        return 'element_local'
    return 'unclassified'


#: Convention of the exported numeric element frames: the source's own local
#: axes as columns in global coordinates (``frame @ tensor_local @
#: frame.T == tensor_global``; ``frame.T`` maps global components to local).
_FRAME_CONVENTION = 'local_axes_as_columns_in_global_coordinates'
_FRAME_TRANSFORM = ('frame @ tensor_local @ frame.T == tensor_global; '
                     'frame.T maps global components to local')


def _finite_json_matrix(matrix: Any, what: str) -> List[List[float]]:
    values = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RecoveryCapsuleError(f'{what} must be finite')
    return [[float(value) for value in row] for row in values]


def _owner_local_frame(owned_element: Any, owned_mesh: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """(frame, None) or (None, reason): the source-exact local frame.

    The frame is read from the snapshot's private captured owner only,
    never from the mutable live model, using the same source functions the
    element's own recovery uses.  Element types without a verified
    source-exact extraction carry an explicit unavailable reason; nothing
    is inferred from sample counts.
    """
    cls = type(owned_element)
    try:
        coordinates = np.asarray(owned_element.get_node_coordinates(owned_mesh),
                                  dtype=np.float64)
        from .e4_pl_s3_element import QualifiedE4PLS3ShellElement, triangle_frame
        if cls is QualifiedE4PLS3ShellElement:
            frame, _local, _quality = triangle_frame(coordinates,
                                                     owned_element.reference_normal)
            return np.asarray(frame, dtype=np.float64), None
        from .elements import BeamElement, QuadraticBeamElement, _beam_rotation_matrix
        if isinstance(owned_element, QuadraticBeamElement):
            # Source-exact chord axis: QuadraticBeamElement
            # ._beam_frame_and_transform orients on the end-node chord
            # coords[2] - coords[0], not on the middle node.
            axis = coordinates[2] - coordinates[0]
            length = float(np.linalg.norm(axis))
            if not np.isfinite(length) or length <= 0.0:
                return None, 'owner geometry has a near-zero beam length'
            rotation = _beam_rotation_matrix(axis / length, owned_element._orientation)
            return np.asarray(rotation, dtype=np.float64), None
        if isinstance(owned_element, BeamElement):
            axis = coordinates[1] - coordinates[0]
            length = float(np.linalg.norm(axis))
            if not np.isfinite(length) or length <= 0.0:
                return None, 'owner geometry has a near-zero beam length'
            rotation = _beam_rotation_matrix(axis / length, owned_element._orientation)
            return np.asarray(rotation, dtype=np.float64), None
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as error:
        return None, f'source frame extraction failed: {type(error).__name__}'
    return None, ('no source-exact local frame extraction is registered '
                  'for this element type')


def _owner_sampling_rule(owned_element: Any) -> Tuple[Optional[tuple], Optional[str]]:
    """(rule key, None) or (None, reason): the native sampling of one element.

    The rule key is ``(kind, coordinate_system, coordinates, weights,
    coordinates_unavailable_reason)`` with the source's own native station
    table (coordinates and weights as plain JSON lists in source order).
    Where the source exposes no native station coordinates the reason is
    explicit; locations are never inferred from sample counts.
    """
    cls = type(owned_element)
    try:
        from .e4_pl_s3_element import QualifiedE4PLS3ShellElement
        if cls is QualifiedE4PLS3ShellElement:
            points = np.asarray(owned_element.gauss_points, dtype=np.float64)
            weights = np.asarray(owned_element.gauss_weights, dtype=np.float64)
            if (points.ndim != 2 or points.shape[1] != 2
                    or weights.shape != (points.shape[0],)):
                return None, 'owner quadrature tables have an unexpected shape'
            if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
                return None, 'owner quadrature tables are not finite'
            return (('triangle_gauss_stations', 'natural_triangle_r_s',
                     tuple((float(r), float(s)) for r, s in points),
                     tuple(float(weight) for weight in weights), None), None)
        from .elements import BeamElement, QuadraticBeamElement
        if isinstance(owned_element, BeamElement):
            # Generalized-section recovery is station-resolved and its native
            # stations differ by branch: the quadratic branch samples its own
            # GAUSS_POINTS table in source order, the two-node force-based
            # branch samples the fixed interpolation stations (-1, 0, 1).
            if getattr(owned_element, 'generalized_section', None) is not None:
                if isinstance(owned_element, QuadraticBeamElement):
                    stations = np.asarray(owned_element.GAUSS_POINTS,
                                          dtype=np.float64)
                else:
                    stations = np.asarray((-1.0, 0.0, 1.0), dtype=np.float64)
                if stations.ndim != 1 or not np.all(np.isfinite(stations)):
                    return None, 'owner beam station table has an unexpected shape'
                return (('beam_natural_xi_stations', 'natural_beam_xi',
                         tuple(float(station) for station in stations), None, None), None)
            # Ordinary beam recovery is one scalar section envelope per
            # element; the three Gauss points are stiffness quadrature, not
            # recovery stations, so no native station coordinates exist.
            return (('element_section_scalar', 'none', None, None,
                     'beam recovery is one scalar section envelope per element; '
                     'the source exposes no native station coordinates'), None)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as error:
        return None, f'source sampling extraction failed: {type(error).__name__}'
    return None, ('no source-exact native sampling extraction is registered '
                  'for this element type')


def _model_recovery_metadata(compilation: Any) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """Shared once-per-model recovery metadata (plain JSON) and bindings.

    Numeric local frames and native sampling rules are captured once per
    model from the snapshot's private recipe owners (never the mutable live
    model).  Sampling rules are deduplicated by content into a shared rule
    table; element bindings reference the shared tables compactly instead
    of duplicating full rules or frames per case or per component.  No
    factorisations, stiffness, executables or owner objects are exported:
    the mapping is finite plain JSON only.  Missing source-exact data is
    recorded with an explicit unavailable reason.
    """
    if compilation is None:
        return ({'capture': 'prepared_static_snapshot_private_recipe_owners',
                 'unavailable_reason': 'the snapshot has no compiled element recovery'},
                {})
    frames: Dict[str, List[List[float]]] = {}
    rule_ids: Dict[tuple, str] = {}
    rule_tables: Dict[str, Dict[str, Any]] = {}
    bindings: Dict[int, Dict[str, Any]] = {}
    for element_id in sorted(compilation.elements):
        recipe = compilation.elements[element_id].local_recipe
        binding: Dict[str, Any] = {}
        owner = getattr(recipe, 'execution_owner', None) if recipe is not None else None
        if not owner:
            binding.update(element_type='unknown', frame=False,
                           frame_unavailable_reason='no captured private recipe owner',
                           sampling_rule=None,
                           sampling_unavailable_reason='no captured private recipe owner',
                           sample_count=None)
        else:
            owned_element, owned_mesh, _owned_material = owner
            binding['element_type'] = type(owned_element).__name__
            frame, frame_reason = _owner_local_frame(owned_element, owned_mesh)
            if frame is not None:
                frames[str(element_id)] = _finite_json_matrix(
                    frame, f'local frame of element {element_id}')
                binding['frame'] = True
            else:
                binding['frame'] = False
                binding['frame_unavailable_reason'] = frame_reason
            rule_key, rule_reason = _owner_sampling_rule(owned_element)
            if rule_key is not None:
                rule_id = rule_ids.get(rule_key)
                if rule_id is None:
                    rule_id = f'rule-{len(rule_ids)}'
                    rule_ids[rule_key] = rule_id
                    kind, coordinate_system, coordinates, weights, reason = rule_key
                    table: Dict[str, Any] = {
                        'kind': kind,
                        'coordinate_system': coordinate_system,
                        'order': 'source_native_station_order',
                    }
                    if coordinates is not None:
                        table['coordinates'] = [list(point) if isinstance(point, tuple)
                                               else point for point in coordinates]
                        table['station_count'] = len(coordinates)
                    else:
                        table['coordinates'] = None
                        table['coordinates_unavailable_reason'] = reason
                    if weights is not None:
                        table['weights'] = list(weights)
                    rule_tables[rule_id] = table
                binding['sampling_rule'] = rule_id
                binding['sample_count'] = rule_tables[rule_id].get('station_count', 1)
            else:
                binding['sampling_rule'] = None
                binding['sampling_unavailable_reason'] = rule_reason
                binding['sample_count'] = None
        bindings[int(element_id)] = binding
    metadata = {
        'capture': 'prepared_static_snapshot_private_recipe_owners',
        'frames': {'convention': _FRAME_CONVENTION,
                   'transform': _FRAME_TRANSFORM,
                   'elements': frames},
        'sampling_rules': rule_tables,
        'element_bindings': {str(element_id): bindings[element_id]
                             for element_id in sorted(bindings)},
    }
    return _plain_json(metadata), bindings


class _Layout:
    """Spatial layout of one component: entity ids and per-entity sizes.

    ``add`` is the only mutation and is the only layout revision: it
    invalidates the cached immutable signature, so every stored row captured
    under one revision shares the same signature object instead of retaining
    a fresh entity-id/size tuple per row.  Per-entity sizes are indexed so
    discovery does not rescan the whole layout per element.
    """

    __slots__ = ('element_ids', 'sizes', '_sizes_by_id', '_signature')

    def __init__(self) -> None:
        self.element_ids: List[int] = []
        self.sizes: List[int] = []
        self._sizes_by_id: Dict[int, int] = {}
        self._signature: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None

    def add(self, element_id: int, size: int) -> None:
        element_id = int(element_id)
        size = int(size)
        self.element_ids.append(element_id)
        self.sizes.append(size)
        self._sizes_by_id.setdefault(element_id, size)
        self._signature = None

    def association(self) -> np.ndarray:
        return np.repeat(np.asarray(self.element_ids, dtype=np.int64),
                         np.asarray(self.sizes, dtype=np.int64))

    def count(self) -> int:
        return int(sum(self.sizes))

    def size_of(self, element_id: int) -> Optional[int]:
        return self._sizes_by_id.get(int(element_id))

    def signature(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        signature = self._signature
        if signature is None:
            signature = (tuple(self.element_ids), tuple(self.sizes))
            self._signature = signature
        return signature


def _variant_for(name: str, linear_names: frozenset) -> Optional[str]:
    """Operation variant of one derived name for one element's linear fields."""
    if name == 'von_mises':
        if all(key in linear_names for key in _SHELL_INPUTS):
            return 'shell'
        if all(key in linear_names for key in _BEAM_INPUTS):
            return 'beam'
        return None
    if name == 'equivalent_stress':
        # Mirrors the element's von Mises variant so both quantities split
        # the same way in mixed element sets.
        if all(key in linear_names for key in _SHELL_INPUTS):
            return 'shell'
        if all(key in linear_names for key in _BEAM_INPUTS):
            return 'beam'
        return None
    if name in PRINCIPAL_COMPONENTS:
        return 'shell' if all(key in linear_names for key in _SHELL_INPUTS) else None
    return None


def _library_basis(evaluator: PreparedStaticResponseEvaluator) -> Optional[dict]:
    """Baseline plus pattern response columns from the certified element library.

    The library columns are exactly the evaluator's admitted combination
    basis: column 0 is the baseline record and every pattern column is the
    zero-subtracted incremental record, so capsule reconstruction with
    ``anyfesolver_results.reconstruct_response`` reproduces the evaluator's
    admitted ``_combine`` arithmetic.
    """
    library = evaluator._element_library
    native = evaluator._native_affinity_fallbacks
    if not library and not native:
        return None
    unsafe_fields = {name for entry in native.values() for name,_shape,_flag in entry[1]}
    unsafe_derived = set()
    for entry in native.values():
        names=frozenset(name for name,_shape,_flag in entry[1])
        unsafe_derived.update((name,_variant_for(name,names)) for name in entry[3])
    layouts: Dict[str, _Layout] = {}
    columns: Dict[str, List[np.ndarray]] = {}
    shapes: Dict[str, Dict[int, List[int]]] = {}
    derived_elements: Dict[str, Dict[str, List[int]]] = {}
    element_linear_sizes: Dict[int, Dict[str, int]] = {}
    constants: Dict[int, Dict[str, Any]] = {}
    for element_id in sorted(set(library)|set(native)):
        entry=library[element_id] if element_id in library else native[element_id]
        order, linear, element_constants, derived, element_columns = entry[:5]
        linear_names = frozenset(name for name, _shape, _is_array in linear)
        offset = 0
        sizes: Dict[str, int] = {}
        for name, shape, _is_array in linear:
            size = int(np.prod(shape)) if shape else 1
            segment = (np.asarray(element_columns[offset:offset + size, :], dtype=np.float64)
                       if element_id in library and name not in unsafe_fields else None)
            offset += size
            if segment is not None:
                layouts.setdefault(name, _Layout()).add(element_id, size)
                columns.setdefault(name, []).append(segment)
                shapes.setdefault(name, {})[element_id] = [int(n) for n in shape]
            sizes[name] = size
        element_linear_sizes[element_id] = sizes
        constants[element_id] = {name: _constant_json(value)
                                 for name, value in element_constants}
        for name in derived:
            variant = _variant_for(name, linear_names)
            if variant is not None:
                derived_elements.setdefault(name, {}).setdefault(variant, []).append(element_id)
    responses = {name: np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float64)
                 for name, parts in columns.items()}
    return {'responses': responses, 'layouts': layouts, 'shapes': shapes,
            'derived_elements': derived_elements, 'element_sizes': element_linear_sizes,
            'constants': constants,'qualified_elements':tuple(sorted(library)),
            'native_stored_derived':unsafe_derived}


def _derived_layout(name: str, variant: str, element_ids: Sequence[int],
                    element_sizes: Dict[int, Dict[str, int]]) -> Optional[Tuple[str, _Layout, tuple]]:
    """Layout and specification inputs for one derived variant.

    Returns ``(operation, layout, input_names)``, or ``None`` when the
    variant's inputs do not align sample-by-sample on every element.
    """
    if name == 'von_mises':
        operation = 'shell_von_mises' if variant == 'shell' else 'beam_von_mises'
        inputs = _SHELL_INPUTS if variant == 'shell' else _BEAM_INPUTS
    elif name in PRINCIPAL_COMPONENTS:
        operation, inputs = 'shell_principal_stress', _SHELL_INPUTS
    else:
        return None
    layout = _Layout()
    for element_id in element_ids:
        size = None
        for input_name in inputs:
            element_size = element_sizes.get(element_id, {}).get(input_name)
            if element_size is None:
                return None
            if size is None:
                size = element_size
            elif element_size != size:
                return None
        layout.add(element_id, size)
    return operation, layout, inputs


def _case_certified(case_result: Mapping[str, Any]) -> bool:
    info = case_result.get('solver_info') or {}
    return (case_result.get('status') == 'converged'
            and info.get('status') == 'converged'
            and info.get('relative_backward_error_disposition') == 'CERTIFIED')


def _case_portable(case_result: Mapping[str, Any]) -> bool:
    info = case_result.get('solver_info') or {}
    source = info.get('source_response') or {}
    return bool(source.get('reconstructed'))


def _case_route(case_result: Mapping[str, Any], stress_execution: str) -> str:
    info = case_result.get('solver_info') or {}
    source = info.get('source_response') or {}
    label = source.get('stress_execution') or stress_execution
    state = source.get('state')
    if state:
        return f'prepared_static_response/{label}/{state}'
    return f'prepared_static_response/{label}'


def _stored_row(case_result: Mapping[str, Any], name: str, layout: _Layout) -> Optional[np.ndarray]:
    stresses = case_result.get('element_stresses') or {}
    parts = []
    for element_id, size in zip(layout.element_ids, layout.sizes):
        record = stresses.get(element_id)
        if record is None:
            return None
        vector = _field_vector(record, name)
        if vector is None or vector.size != size:
            return None
        parts.append(vector)
    return np.concatenate(parts) if parts else np.zeros(0)


def _principal_row(case_result: Mapping[str, Any], name: str, rank: int,
                   layout: _Layout) -> Optional[np.ndarray]:
    """Exact stored scalar extraction of one principal rank per sample."""
    stresses = case_result.get('element_stresses') or {}
    parts = []
    for element_id, size in zip(layout.element_ids, layout.sizes):
        record = stresses.get(element_id)
        if record is None:
            return None
        value = record.get(name)
        if not isinstance(value, np.ndarray) or value.dtype.kind != 'f':
            return None
        if value.ndim != 2 or value.shape[0] != size or value.shape[1] != len(PRINCIPAL_RANKS):
            return None
        if not np.all(np.isfinite(value)):
            return None
        parts.append(np.asarray(value[:, rank - 1], dtype=np.float64))
    return np.concatenate(parts) if parts else np.zeros(0)


class _MemorySink:
    """Internal sink that assembles the in-memory capsule mapping."""

    def __init__(self) -> None:
        self.capsule: Dict[str, Any] = {}
        self.cases: List[Dict[str, Any]] = []
        self.coefficients: List[np.ndarray] = []
        self.rows: Dict[str, List[Tuple[int, np.ndarray]]] = {}

    def capsule_begin(self, header: Mapping[str, Any]) -> None:
        self.capsule = dict(header)

    def capsule_case(self, record: Mapping[str, Any], coefficients: np.ndarray) -> None:
        self.cases.append(dict(record))
        self.coefficients.append(np.asarray(coefficients, dtype=np.float64))

    def capsule_stored(self, key: str, case_indices: np.ndarray, values: np.ndarray) -> None:
        entries = self.rows.setdefault(key, [])
        indices = np.asarray(case_indices, dtype=np.int64)
        rows = np.asarray(values, dtype=np.float64)
        for position in range(indices.shape[0]):
            entries.append((int(indices[position]), np.asarray(rows[position], dtype=np.float64)))

    def capsule_end(self, final: Mapping[str, Any]) -> None:
        self.capsule['cases'] = tuple(self.cases)
        pattern_count = len(self.capsule['patterns'])
        self.capsule['coefficients'] = (
            np.ascontiguousarray(np.stack(self.coefficients), dtype=np.float64)
            if self.coefficients else np.zeros((0, pattern_count), dtype=np.float64))
        self.capsule['descriptors'] = tuple(final['descriptors'])
        self.capsule['associations'] = dict(final['associations'])
        self.capsule['responses'] = dict(final['responses'])
        self.capsule['derived'] = dict(final['derived'])
        self.capsule['element_constants'] = dict(final['element_constants'])
        if 'storage_prediction' not in final:
            self.capsule.pop('storage_prediction', None)
        superseded = final['stored_superseded_cases']
        stored: Dict[str, Dict[str, Any]] = {}
        for key, entries in self.rows.items():
            dropped = set(superseded.get(key, ()))
            kept = [(index, row) for index, row in entries if index not in dropped]
            if kept:
                stored[key] = {
                    'case_indices': np.asarray([index for index, _row in kept], dtype=np.int64),
                    'values': np.ascontiguousarray(np.stack([row for _index, row in kept]),
                                                   dtype=np.float64),
                }
        self.capsule['stored'] = stored


class _CapsuleExport:
    """One bounded capsule export driven through a sink."""

    def __init__(self, evaluator, cases, *, model_identity, mesh_identity,
                 geometry=None, solver_identity=None, batch_size=None, recovery_execution='auto', replay_version=2,
                 cancellation_token=None, source_field_observer=None):
        _require(type(evaluator) is PreparedStaticResponseEvaluator,
                 'recovery capsule export requires a prepared static response evaluator')
        _require(not evaluator.closed, 'recovery capsule export requires an open evaluator')
        _require(type(model_identity) is str and model_identity, 'model_identity must be a nonempty string')
        _require(type(mesh_identity) is str and mesh_identity, 'mesh_identity must be a nonempty string')
        _require(isinstance(cases, (list, tuple)), 'cases must be a list or tuple of PatternCase')
        for case in cases:
            _require(type(case) is PatternCase, 'capsule cases must be PatternCase instances')
        case_ids = [case.case_id for case in cases]
        _require(len(set(case_ids)) == len(case_ids), 'capsule case ids must be unique')
        self.evaluator = evaluator
        self.cases = list(cases)
        self.cancellation_token = cancellation_token
        _require(source_field_observer is None or callable(source_field_observer),
                 'source_field_observer must be callable or None')
        self.source_field_observer = source_field_observer
        self._observer_batch_bytes = 0
        self._observer_metadata = {} if source_field_observer is not None else None
        self._observer_metadata_bytes = 0
        self.model_identity = model_identity
        self.mesh_identity = mesh_identity
        self.geometry = geometry
        self.solver_identity = solver_identity
        size = DEFAULT_BATCH_SIZE if batch_size is None else int(batch_size)
        _require(size >= 1, 'batch_size must be positive')
        self.batch_size = size
        _require(type(recovery_execution) is str and recovery_execution in ('auto','records'), 'recovery_execution must be auto or records')
        self.recovery_execution = recovery_execution
        _require(type(replay_version) is int and replay_version in (1,2),'replay_version must be 1 or 2')
        self.replay_version = replay_version
        self.runtime_id = f'anysolver-results/{replay_version}'
        self.contract_id = f'{SCHEMA}/{replay_version}'
        self.combination = 'ordered_float64/1' if replay_version == 2 else 'legacy_blas/1'
        self._numeric_indices = {}
        self._dependency_indices = {}
        self._runtime_guard_checks = 0
        self._native_affinity_opt_in = evaluator._numerical_affinity_fallback
        self._native_response_guard_checks = 0
        self._native_response_guard_ns = 0

        diagnostics = evaluator.diagnostics()['source_response']
        self.pattern_names = tuple(diagnostics['pattern_names'])
        self.stress_execution = diagnostics['stress_execution']
        self.pattern_count = len(self.pattern_names)
        self.pattern_position = {name: index for index, name in enumerate(self.pattern_names)}

        self.basis = _library_basis(evaluator) if self.stress_execution == 'certified_library' else None

        # Component state accumulated across batches (output-sized only).
        self.layouts: Dict[str, _Layout] = dict(self.basis['layouts']) if self.basis else {}
        self.extractors: Dict[str, Callable[[Mapping[str, Any]], Optional[np.ndarray]]] = {}
        self.responses: Dict[str, np.ndarray] = dict(self.basis['responses']) if self.basis else {}
        self.derived: Dict[str, Any] = {}
        self._stored_derived_fields = {}
        self._native_field_names = {name for entry in evaluator._native_affinity_fallbacks.values()
            for name in tuple(item[0] for item in entry[1])+entry[3]}
        self.descriptor_info: Dict[str, Dict[str, Any]] = {}
        self.element_constants: Dict[int, Dict[str, Any]] = (
            {int(eid): dict(values) for eid, values in self.basis['constants'].items()}
            if self.basis else {})
        # Stored-row bookkeeping is split so the stream stays bounded: the
        # numeric rows pending emission are released as soon as the sink has
        # consumed them; only compact signature/index metadata survives for
        # the final descriptors and validity accounting.
        self.stored_rows: Dict[str, List[Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], int]]] = {}
        self.pending_rows: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        self.displacement_portable: List[Optional[bool]] = []
        # Shared once-per-model recovery metadata (frames and deduplicated
        # native sampling rules) and the per-element native record shapes
        # observed while scanning stored fields.
        self.recovery_metadata: Dict[str, Any] = {}
        self.recovery_bindings: Dict[int, Dict[str, Any]] = {}
        self.native_shapes: Dict[str, Dict[int, List[int]]] = {}
        # Record field names already covered by the portable basis, its
        # registered derived components or its constants; anything else in a
        # case record gets an explicit stored fallback.
        self.covered_fields: frozenset = (
            frozenset(self.basis['layouts']) | frozenset(self.basis['derived_elements'])
            | {name for values in self.element_constants.values() for name in values}
            if self.basis else frozenset(EXCLUDED_DERIVED))

        self._build_derived_components()
        self._prepare_displacements()
        self._storage_prediction_retained = 0
        self.storage_prediction = self._build_storage_prediction()

    def _build_storage_prediction(self):
        """Compression inputs only; never restore a rejected engineering basis.

        The role is omitted for selected/load-dependent layouts not proven by
        this source-owned static table, and for fully native libraries.
        """
        library=self.evaluator._element_library
        native=self.evaluator._native_affinity_fallbacks
        config=self.evaluator._recovery
        if (self.replay_version!=2 or not native or not library or self.basis is None
                or getattr(config,'element_ids',None) is not None
                or getattr(config,'components',None) is not None):
            return None
        fields={name for entry in native.values() for name,_shape,_flag in entry[1]}
        counts={name:sum(sizes.get(name,0) for sizes in self.basis['element_sizes'].values()) for name in fields}
        counts={name:count for name,count in counts.items() if count}
        expected=sum(count*8*(self.pattern_count+2) for count in counts.values())
        # Admit raw arrays, read-only ownership wrappers and bounded Python
        # sorting/layout scratch before any new retained numeric allocation.
        self._check_replay_memory(2*expected+96*(len(library)+len(native))+65536)
        element_ids=sorted(self.basis['element_sizes'])
        responses={};associations={}
        for name,count in sorted(counts.items()):
            matrix=np.zeros((count,self.pattern_count+1),dtype=np.float64)
            ids=np.empty(count,dtype=np.int64);position=0
            for ordinal,eid in enumerate(element_ids):
                if ordinal%64==0:self._export_cancellation_point('recovery_capsule.storage_prediction')
                size=self.basis['element_sizes'][eid].get(name,0)
                if not size:continue
                ids[position:position+size]=eid
                if eid in library:
                    offset=0;found=False
                    for field,shape,_flag in library[eid][1]:
                        width=int(np.prod(shape)) if shape else 1
                        if field==name:
                            _require(width==size,'storage predictor/source field shape changed')
                            matrix[position:position+size]=library[eid][4][offset:offset+size]
                            found=True;break
                        offset+=width
                    _require(found,'qualified storage predictor field is missing')
                # Actual native-refused rows stay literal codec-zero inputs.
                position+=size
            _require(position==count,'storage predictor/source row count changed')
            responses[name]=matrix;associations[name]=ids
        derived={}
        for name in ('von_mises','equivalent_stress'):
            variants=self.basis['derived_elements'].get(name,{})
            for variant in sorted(variants):
                key=name if len(variants)==1 else f'{name}:{variant}'
                if key not in self._stored_derived_fields:continue
                if name=='von_mises':
                    inputs=_SHELL_INPUTS if variant=='shell' else _BEAM_INPUTS if variant=='beam' else ()
                    if not inputs or any(field not in responses for field in inputs):continue
                    first=inputs[0]
                    if any(not np.array_equal(associations[first],associations[field]) for field in inputs):continue
                    derived[key]={'operation':'shell_von_mises' if variant=='shell' else 'beam_von_mises','inputs':{field:field for field in inputs}}
                    associations[key]=associations[first]
                else:
                    source=f'von_mises:{variant}' if f'von_mises:{variant}' in derived else 'von_mises'
                    if source not in derived:continue
                    derived[key]={'operation':'equivalent_stress','inputs':{'von_mises':source}}
                    associations[key]=associations[source]
        if not responses:return None
        role={'version':1,'runtime':'anysolver-results/2','combination':'ordered_float64/1',
              'responses':responses,'derived':derived,'associations':associations,
              'targets':sorted(set(responses)|set(derived)),
              'provenance':{'purpose':'lossless_byte_prediction_only','qualified_element_count':len(library),
                            'native_element_ids':sorted(native),'native_prediction':'literal_zero','principal':'plain_actual_source_words'}}
        validate_storage_prediction(role,patterns=self.pattern_names)
        seen=set();retained=0
        for value in tuple(responses.values())+tuple(associations.values()):
            if id(value) not in seen:retained+=value.nbytes;seen.add(id(value))
        self._storage_prediction_retained=retained
        return _readonly_data(role)

    # -- preparation -----------------------------------------------------

    def _prepare_displacements(self) -> None:
        """Baseline+pattern displacement columns for the selected nodes.

        The columns are the evaluator's own retained solved columns pushed
        through the same transformation and prescribed offset the evaluator
        applies, so per-case parity against the evaluator's actual
        accumulation order can be established exactly.
        """
        self.displacement_keys: List[str] = []
        self.displacement_columns: Optional[np.ndarray] = None
        self.displacement_nodes: Tuple[int, ...] = ()
        self.displacement_rows_matrix = None
        self.displacement_prescribed = None
        snapshot = self.evaluator._snapshot
        selected = tuple(snapshot.selected_node_ids)
        solution = self.evaluator._solution_columns
        if not selected or solution is None:
            return
        node_dofs = snapshot.load_maps.node_dofs
        try:
            dofs = np.concatenate([np.asarray(node_dofs[node], dtype=np.intp)
                                   for node in selected])
        except KeyError:
            return
        if dofs.size != 6 * len(selected) or np.any(dofs < 0) or np.any(dofs >= snapshot.total_dofs):
            return
        reduced = snapshot.reduced_stiffness.shape[0]
        rows = snapshot.transformation[dofs].tocsr()
        columns = np.asarray(rows @ np.asarray(solution[:reduced], dtype=np.float64),
                            dtype=np.float64)
        columns[:, 0] += np.asarray(snapshot.prescribed_offset[dofs], dtype=np.float64)
        if not np.all(np.isfinite(columns)):
            return
        self.displacement_keys = [f'displacement:{component}'
                                  for component, _unit in DISPLACEMENT_COMPONENTS]
        self.displacement_columns = columns
        self.displacement_nodes = selected
        self.displacement_rows_matrix = rows
        self.displacement_prescribed = np.asarray(snapshot.prescribed_offset[dofs],
                                                  dtype=np.float64)
        for key in self.displacement_keys:
            layout = _Layout()
            for node in selected:
                layout.add(node, 1)
            self.layouts[key] = layout

    def _build_derived_components(self) -> None:
        """Derived specs and layouts from the portable basis, if any."""
        if not self.basis:
            return
        von_mises_layouts: Dict[str, _Layout] = {}
        # von Mises (and the principal stress ranks) first; equivalent stress
        # follows because its input key is the exported von Mises key.
        ordered_names = [n for n in sorted(self.basis['derived_elements']) if n != 'equivalent_stress']
        if 'equivalent_stress' in self.basis['derived_elements']:
            ordered_names.append('equivalent_stress')
        for name in ordered_names:
            variants = self.basis['derived_elements'][name]
            for variant, element_ids in sorted(variants.items()):
                if name == 'equivalent_stress':
                    source_layout = von_mises_layouts.get(variant)
                    if source_layout is None:
                        continue
                    operation, inputs = 'equivalent_stress', ('von_mises',)
                    layout = _Layout()
                    for element_id in element_ids:
                        size = source_layout.size_of(element_id)
                        if size is None:
                            layout = None
                            break
                        layout.add(element_id, size)
                    if layout is None:
                        continue
                else:
                    made = _derived_layout(name, variant, element_ids, self.basis['element_sizes'])
                    if made is None:
                        continue
                    operation, layout, inputs = made
                if name in PRINCIPAL_COMPONENTS:
                    surface = name.removeprefix('principal_stresses_')
                    exported = [(f'principal_stress_{rank}:{surface}',
                                 {'operation': operation, 'surface': surface, 'rank': rank}, rank)
                                for rank in PRINCIPAL_RANKS]
                else:
                    exported = [(name if len(variants) == 1 else f'{name}:{variant}',
                                 {'operation': operation}, None)]
                for key, spec, rank in exported:
                    native_stored = (name,variant) in self.basis.get('native_stored_derived',())
                    if name == 'equivalent_stress':
                        native_stored = native_stored or ('von_mises',variant) in self.basis.get('native_stored_derived',())
                        vm_key=(f'von_mises:{variant}' if f'von_mises:{variant}' in self.layouts else 'von_mises')
                        native_stored = native_stored or vm_key in self._stored_derived_fields
                    elif any(input_name not in self.basis['layouts'] for input_name in inputs):
                        native_stored = True
                    spec_inputs = {}
                    for input_name in inputs:
                        if input_name in self.basis['layouts']:
                            spec_inputs[input_name] = input_name
                        elif name == 'equivalent_stress':
                            spec_inputs[input_name] = vm_key
                        else:
                            spec_inputs[input_name] = key
                    spec['inputs'] = spec_inputs
                    if native_stored:
                        self._stored_derived_fields[key]=name
                        if rank is None:
                            self.extractors[key]=(lambda case_result,_name=name,_layout=layout: _stored_row(case_result,_name,_layout))
                    else:
                        self.derived[key] = spec
                    self.layouts[key] = layout
                    quantity_id, unit, metadata = _quantity_of(key.rsplit(':', 1)[0])
                    if name in PRINCIPAL_COMPONENTS:
                        metadata = dict(metadata, surface=surface, rank=rank)
                    elif name == 'von_mises' and variant == 'shell':
                        metadata = dict(metadata, surfaces=('top', 'bottom'), combination='max')
                    component = key if name in PRINCIPAL_COMPONENTS or native_stored else key.rsplit(':', 1)[0]
                    self.descriptor_info[key] = {
                        'quantity_id': quantity_id,
                        'component': component,
                        'location': 'element',
                        'unit': unit,
                        'basis': _physical_basis(key.rsplit(':', 1)[0]),
                        'reconstruction': 'explicit_per_case_values' if native_stored else 'derived_from_reconstructable_inputs',
                        'sampling': 'element_sample',
                        'metadata': metadata,
                    }
                    if rank is not None:
                        self.extractors[key] = (
                            lambda case_result, _name=name, _rank=rank, _layout=layout:
                            _principal_row(case_result, _name, _rank, _layout))
                    if name == 'von_mises':
                        von_mises_layouts[variant] = layout

    # -- per-case processing ---------------------------------------------

    def _coefficient_row(self, case: PatternCase) -> Tuple[np.ndarray, List[int]]:
        values = case.coefficients
        if isinstance(values, Mapping):
            pairs = values.items()
        else:
            _require(len(values) == self.pattern_count, 'dense coefficient length must equal pattern count')
            pairs = zip(self.pattern_names, values)
        row = np.zeros(self.pattern_count, dtype=np.float64)
        active: List[int] = []
        for name, value in pairs:
            position = self.pattern_position.get(name)
            _require(position is not None, f'unknown source response pattern: {name!r}')
            row[position] = float(value)
            if value:
                active.append(position + 1)
        return row, active

    def _actual_displacements(self, case_result: Mapping[str, Any],
                              portable: bool, active: Sequence[int],
                              coefficients: np.ndarray) -> Optional[np.ndarray]:
        """The evaluator's actual nodal displacement vector for the layout nodes."""
        if isinstance(case_result, NumericCase) and case_result.displacements is not None:
            if case_result.node_ids == self.displacement_nodes:
                return case_result.displacements
            positions = {node:i for i,node in enumerate(case_result.node_ids)}
            if any(node not in positions for node in self.displacement_nodes):
                return None
            indices = np.concatenate([np.arange(6*positions[node],6*positions[node]+6) for node in self.displacement_nodes])
            return case_result.displacements[indices]
        recorded = case_result.get('node_displacements') or {}
        if recorded:
            parts = []
            for node in self.displacement_nodes:
                value = recorded.get(node)
                if value is None:
                    return None
                vector = np.asarray(value, dtype=np.float64).reshape(-1)
                if vector.size != 6:
                    return None
                parts.append(vector)
            return np.concatenate(parts)
        if not portable or self.replay_version == 2:
            return None
        # The evaluator's reconstructed accumulation order, replicated
        # exactly: combine the retained solved columns, then transform.  The
        # selected columns are multiplied by their own entries of the full
        # coefficient row (positions are 1-based, in the original active
        # order); an empty active set is the baseline alone.
        solution = self.evaluator._solution_columns
        if solution is None:
            return None
        active_array = np.asarray(active, dtype=np.intp)
        weights = coefficients[active_array - 1] if active_array.size else None
        combined = (solution[:, 0] + (solution[:, active_array] @ weights
                                     if active_array.size else 0.))
        reduced = self.evaluator._snapshot.reduced_stiffness.shape[0]
        actual = np.asarray(self.displacement_rows_matrix @ combined[:reduced], dtype=np.float64)
        return actual + self.displacement_prescribed

    def _grow_layout(self, key: str, layout: _Layout, entity_id: int, size: int) -> None:
        """Add one entity to a component layout, failing closed if stale.

        A layout change after stored rows were captured would supersede rows
        the sink may already have consumed; the export is refused atomically
        instead of silently dropping previously exported rows.
        """
        _require(not self.stored_rows.get(key),
                 f'component {key!r} cannot change layout after stored rows '
                 'were captured; refusing to supersede exported rows')
        layout.add(entity_id, size)

    def _register_principal_field(self, element_id: int, name: str, value: Any) -> None:
        """Register per-rank scalar keys for one native principal-stress field.

        The native ``principal_stresses_(top,bottom)`` record fields hold one
        descending-eigenvalue triple per sample; they are never flattened
        into a generic vector field.  Each rank becomes its own scalar
        component (``principal_stress_{rank}:{surface}``) with the surface
        preserved, one scalar per sample, and explicit fallback rows
        extracted through :func:`_principal_row`.
        """
        if (not isinstance(value, np.ndarray) or value.dtype.kind != 'f'
                or value.ndim != 2 or value.shape[1] != len(PRINCIPAL_RANKS)
                or not np.all(np.isfinite(value))):
            return
        surface = name.removeprefix('principal_stresses_')
        for rank in PRINCIPAL_RANKS:
            key = f'principal_stress_{rank}:{surface}'
            layout = self.layouts.setdefault(key, _Layout())
            if layout.size_of(element_id) is None:
                self._grow_layout(key, layout, element_id, int(value.shape[0]))
                self.native_shapes.setdefault(name, {})[element_id] = (
                    [int(size) for size in value.shape])
            if key not in self.extractors:
                self.extractors[key] = (
                    lambda case_result, _name=name, _rank=rank, _layout=layout:
                    _principal_row(case_result, _name, _rank, _layout))
                self.descriptor_info[key] = {
                    'quantity_id': _STRESS_QUANTITY,
                    'component': key,
                    'location': 'element',
                    'unit': _STRESS_UNIT,
                    'basis': 'principal',
                    'reconstruction': 'explicit_per_case_values',
                    'sampling': 'element_sample',
                    'metadata': {'surface': surface, 'rank': rank},
                }

    def _scan_records(self, case_result: Mapping[str, Any]) -> None:
        """Grow stored layouts and element constants from one case's records."""
        if isinstance(case_result, NumericCase):
            for name, field in case_result.fields.items():
                if name in EXCLUDED_DERIVED or name in self.covered_fields:
                    continue
                if name in self.native_shapes:
                    continue
                for position,eid in enumerate(field.element_ids):
                    shape = field.shapes[position]
                    value = field.values[field.offsets[position]:field.offsets[position+1]].reshape(shape)
                    if name in PRINCIPAL_COMPONENTS:
                        self._register_principal_field(eid,name,value)
                    else:
                        layout = self.layouts.setdefault(name,_Layout())
                        if layout.size_of(eid) is None:
                            self._grow_layout(name,layout,eid,value.size)
                            self.native_shapes.setdefault(name,{})[eid] = list(shape) if field.array_flags[position] else [1]
            return
        stresses = case_result.get('element_stresses') or {}
        for element_id in sorted(int(eid) for eid in stresses):
            record = stresses[element_id]
            _require(isinstance(record, Mapping), 'element stress records must be mappings')
            constants = self.element_constants.setdefault(int(element_id), {})
            for field_name, value in record.items():
                if field_name in EXCLUDED_DERIVED or field_name in self.covered_fields:
                    continue
                if field_name in PRINCIPAL_COMPONENTS:
                    self._register_principal_field(int(element_id), field_name, value)
                    continue
                layout = self.layouts.get(field_name)
                if layout is None:
                    if _is_constant_value(value):
                        constants.setdefault(field_name, _constant_json(value))
                        continue
                    vector = _field_vector(record, field_name)
                    if vector is None:
                        continue
                    layout = self.layouts.setdefault(field_name, _Layout())
                if layout.size_of(int(element_id)) is None:
                    vector = _field_vector(record, field_name)
                    if vector is not None:
                        self._grow_layout(field_name, layout, int(element_id), vector.size)
                        self.native_shapes.setdefault(field_name, {})[int(element_id)] = (
                            [int(size) for size in value.shape]
                            if isinstance(value, np.ndarray) and value.dtype.kind == 'f'
                            else [1])

    def _process_case(self, index: int, case_result: Mapping[str, Any]) -> None:
        case = self.cases[index]
        coefficients, active = self._coefficient_row(case)
        certified = _case_certified(case_result)
        portable = bool(certified and self.basis is not None and self.basis['qualified_elements'] and _case_portable(case_result))
        info = case_result.get('solver_info') or {}
        record: Dict[str, Any] = {
            'case_id': case.case_id,
            # The exact original request position: the identity the source
            # envelope tie policy (exact earliest input_index) is defined on.
            'input_index': int(index),
            'status': str(case_result.get('status')),
            'certified': bool(certified),
            'portable': portable,
            'route': _case_route(case_result, self.stress_execution),
            'time': None if case.time is None else float(case.time),
            'active_indices': [int(i) for i in active],
            'provenance': _plain_json({
                'relative_backward_error': info.get('relative_backward_error'),
                'relative_backward_error_limit': info.get('relative_backward_error_limit'),
                'relative_backward_error_disposition': info.get('relative_backward_error_disposition'),
                'source_response': info.get('source_response'),
                'load': info.get('load'),
            }),
        }
        if self.replay_version == 2:
            record['unavailable_components'] = []
            record['correction_reasons'] = {}
        reaction_node_ids = case_result.get('reaction_node_ids')
        if reaction_node_ids:
            record['reaction_node_ids'] = [int(node) for node in reaction_node_ids]
        # Failed records must retain their source explanation, rather than
        # only the five successful-certification provenance fields above.
        # These are diagnostic metadata, never an admission of result fields.
        for name in ('diagnostics', 'error', 'message', 'failure_reason'):
            if name in case_result:
                record[name] = _plain_json(case_result[name])
        if not certified:
            record['solver_info'] = _plain_json(info)
            self.displacement_portable.append(None)
            record['displacement_portable'] = None
            return record, coefficients

        # Element fields and registered derived components: stored rows for
        # non-portable certified cases (the portable basis reconstructs the
        # rest); explicit stored fallback for every other record field.
        self._scan_records(case_result)
        if self.source_field_observer is not None:
            self._observe_source_case(index, case_result, record)
        if not portable:
            for key in sorted(self.layouts):
                if key.startswith(('displacement:', 'reaction:')):
                    continue
                if key not in self.responses and key not in self.derived:
                    continue
                row = self._element_row(key, case_result)
                if row is not None:
                    self._add_stored_row(key, index, row)
        for key in sorted(self.layouts):
            if key.startswith(('displacement:', 'reaction:')):
                continue
            if key in self.responses or key in self.derived:
                continue
            row = self._element_row(key, case_result)
            if row is not None:
                self._add_stored_row(key, index, row)
                if self.replay_version == 2 and key.startswith('principal_stress_'):
                    record['correction_reasons'][key]='spectral_replay_not_certified_portable'
            elif self.replay_version == 2:
                record['unavailable_components'].append(key)

        # Portable basis fields are admitted per case only when the
        # runtime's own field-level accumulation reproduces the evaluator's
        # certified record bitwise.  The certified library accumulates at the
        # element-vector level (``_combine`` over the whole local column
        # block), so an active multi-pattern combination can round
        # differently than the per-field reconstruction; that is a real
        # per-case divergence, not a tolerance question.  Diverging
        # components keep an explicit stored row with the certified values
        # (stored outputs override basis reconstruction for those exact
        # case/component pairs); no arithmetic or tolerance is relaxed.
        if portable:
            active_array = np.asarray(active, dtype=np.intp)
            weights = coefficients[active_array - 1] if active_array.size else None
            for key in sorted(self.responses):
                actual = self._element_row(key, case_result)
                if actual is None:
                    if self.replay_version == 2:
                        record['unavailable_components'].append(key)
                    continue
                responses = self.responses[key]
                reconstructed = None
                if self.replay_version == 2:
                    self._check_replay_memory(responses.shape[0]*(active_array.size+4)*8)
                    try:
                        reconstructed = reconstruct_response(responses,coefficients[active_array-1],
                            active_indices=active_array,combination=self.combination)
                    except ReconstructionError:
                        record['correction_reasons'][key] = 'ordered_replay_refused'
                else:
                    reconstructed = (responses[:,0]+(responses[:,active_array]@weights if active_array.size else 0.))
                if reconstructed is None or not float_bits_equal(reconstructed, actual):
                    self._add_stored_row(key, index, actual)
                    if self.replay_version == 2:
                        record['correction_reasons'].setdefault(key,'source_bits_differ')

        if self.replay_version == 2:
            self._admit_derived(index,case_result,record,portable)
            if not portable:
                for key in self.responses:
                    if self._element_row(key,case_result) is None:
                        record['unavailable_components'].append(key)

        # Displacements: per-case parity against the evaluator's actual
        # accumulation order; explicit stored rows when parity fails or the
        # case is not portable.  The selected columns are multiplied by
        # their own entries of the full coefficient row (1-based positions,
        # original active order); an empty active set is the baseline alone.
        displacement_flag: Optional[bool] = None
        if self.displacement_columns is not None:
            actual = self._actual_displacements(case_result, portable, active, coefficients)
            if actual is not None:
                active_array = np.asarray(active, dtype=np.intp)
                weights = coefficients[active_array - 1] if active_array.size else None
                reconstructed = None
                if self.replay_version == 2:
                    self._check_replay_memory(self.displacement_columns.shape[0]*(active_array.size+4)*8)
                    try:
                        reconstructed = reconstruct_response(self.displacement_columns,coefficients[active_array-1],
                            active_indices=active_array,combination=self.combination)
                    except ReconstructionError:
                        pass
                else:
                    reconstructed = (self.displacement_columns[:,0]+(self.displacement_columns[:,active_array]@weights if active_array.size else 0.))
                displacement_flag = bool(reconstructed is not None and float_bits_equal(reconstructed,actual)
                                         and (portable or self.replay_version == 1))
                if not displacement_flag:
                    for position, _component in enumerate(DISPLACEMENT_COMPONENTS):
                        self._add_stored_row(self.displacement_keys[position], index,
                                             np.asarray(actual[position::6], dtype=np.float64))
                        if self.replay_version == 2:
                            record['correction_reasons'][self.displacement_keys[position]] = ('source_nonportable' if not portable
                                else 'source_bits_differ' if reconstructed is not None else 'ordered_replay_refused')
            elif self.replay_version == 2:
                record['unavailable_components'].extend(self.displacement_keys)
        self.displacement_portable.append(displacement_flag)
        record['displacement_portable'] = displacement_flag

        # Reactions: actual queryable components (bitwise Ku-F actuals).
        reactions = case_result.get('reactions') or {}
        if reactions:
            for position, (component, _unit) in enumerate(REACTION_COMPONENTS):
                key = f'reaction:{component}'
                layout = self.layouts.setdefault(key, _Layout())
                for node in sorted(int(n) for n in reactions):
                    if layout.size_of(node) is None:
                        self._grow_layout(key, layout, node, 1)
                parts = []
                for node in layout.element_ids:
                    value = reactions.get(node)
                    if value is None:
                        parts = None
                        break
                    vector = np.asarray(value, dtype=np.float64).reshape(-1)
                    if vector.size != 6 or not np.all(np.isfinite(vector)):
                        parts = None
                        break
                    parts.append(np.asarray(vector[position], dtype=np.float64).reshape(1))
                if parts:
                    self._add_stored_row(key, index, np.concatenate(parts))
        if self.replay_version == 2:
            record['unavailable_components'] = sorted(set(record['unavailable_components']))
        return record, coefficients

    def _observer_shapes(self,key,case_result,layout):
        if key.startswith(('reaction:','displacement:')):
            return tuple(() for _ in layout.element_ids)
        field_name=self._stored_derived_fields.get(key,key.rsplit(':',1)[0] if key in self.derived else key)
        principal=key.startswith('principal_stress_')
        if principal:field_name='principal_stresses_'+key.split(':',1)[1]
        if isinstance(case_result,NumericCase):
            field=case_result.fields.get(field_name)
            shapes={} if field is None else dict(zip(field.element_ids,field.shapes))
        else:
            shapes={eid:tuple(np.asarray(record[field_name]).shape) for eid,record in
                (case_result.get('element_stresses') or {}).items() if field_name in record}
        return tuple(tuple(shapes.get(eid,(size,)))[:-1] if principal else tuple(shapes.get(eid,(size,)))
                     for eid,size in zip(layout.element_ids,layout.sizes))

    def _observer_layout(self,key,case_result,layout):
        # Current layout bindings are observed even when its owned signature
        # and the NumericField's immutable stock tuples can be reused.
        signature=layout.signature()
        _require(tuple(layout.element_ids)==signature[0] and tuple(layout.sizes)==signature[1],
                 'source observer layout no longer matches its owned signature')
        field_name=self._stored_derived_fields.get(key,key.rsplit(':',1)[0] if key in self.derived else key)
        if key.startswith('principal_stress_'):field_name='principal_stresses_'+key.split(':',1)[1]
        field=case_result.fields.get(field_name) if isinstance(case_result,NumericCase) else None
        shape_binding=None if field is None else field.shapes
        ids_binding=None if field is None else field.element_ids
        numeric_node=isinstance(case_result,NumericCase) and key.startswith('displacement:')
        if numeric_node:
            shape_binding=();ids_binding=case_result.node_ids
            if type(ids_binding) is tuple and all(type(i) is int for i in ids_binding) and ids_binding==signature[0]:
                ids_binding=signature[0]
        previous=self._observer_metadata.get(key)
        if (previous is not None and previous[0] is signature
                and (field is not None or numeric_node) and previous[1] is shape_binding and previous[2] is ids_binding):
            count=layout.count()
            for name in ('associations','native_ordinals'):
                value=previous[3][name]
                _require(value.dtype==np.int64 and value.shape==(count,) and value.strides==(8,)
                         and value.flags.c_contiguous and not value.flags.writeable,
                         'source observer cached array layout changed')
                _require_observer_array_facts(previous[5][name])
            return previous[3]
        # Never retain histories: release the replaced entry before checking
        # simultaneous new snapshot/conversion admission.
        if previous is not None:
            self._observer_metadata_bytes-=previous[4];del self._observer_metadata[key]
            previous=None
        shapes=self._observer_shapes(key,case_result,layout)
        size=layout.count()*32+len(layout.element_ids)*128+4096
        self._check_replay_memory(self._observer_batch_bytes+self._observer_metadata_bytes+size*2+65536)
        associations=layout.association()
        ordinals=np.concatenate([np.arange(n,dtype=np.int64) for n in layout.sizes]) if layout.sizes else np.empty(0,dtype=np.int64)
        metadata=_observer_freeze({'associations':associations,'native_ordinals':ordinals,
            'layout':'source_scalar_rows/C','element_ids':signature[0],'element_sizes':signature[1],
            'native_shapes':shapes})
        # Immutable tuple grammar is classified only after full stock checks.
        # Public record shapes are always observed afresh and rebuilt.
        stock=(type(shape_binding) is tuple and type(ids_binding) is tuple
            and all(type(i) is int for i in ids_binding)
            and all(type(shape) is tuple and all(type(n) is int for n in shape) for shape in shape_binding))
        if stock:
            # Keep the helper binding in this method's co_names so the
            # transitive runtime guard pins its module-global identity.
            buffers={}
            for name in ('associations','native_ordinals'):
                buffers[name]=_observer_array_facts(metadata[name])
            self._observer_metadata[key]=(signature,shape_binding,ids_binding,metadata,size,buffers)
            self._observer_metadata_bytes+=size
        return metadata

    def _observe_source_case(self, index, case_result, record):
        """Actual owned source fields, before any replay or correction admission."""
        _require_observer_environment()
        keys=set(self.layouts)
        reactions=case_result.get('reactions') or {}
        for component,_unit in REACTION_COMPONENTS:
            if reactions:keys.add('reaction:'+component)
        # Bound full-case source snapshots, original selector copies and metadata.
        rows=sum(layout.count() for layout in self.layouts.values())+6*len(reactions)
        workspace=self._observer_batch_bytes+self._observer_metadata_bytes+rows*512+65536
        self._check_replay_memory(workspace)
        case_meta={name:record[name] for name in ('case_id','input_index','status','certified','route','time','provenance')}
        fields={};unavailable=[]
        actual_displacements=None
        if self.displacement_columns is not None:
            actual_displacements=self._actual_displacements(case_result,False,(),np.empty(0,dtype=np.float64))
        for key in sorted(keys):
            self._export_cancellation_point('recovery_capsule.source_observer.snapshot')
            if key.startswith('displacement:'):
                component=key.split(':',1)[1];column=[c for c,_ in DISPLACEMENT_COMPONENTS].index(component)
                row=None if actual_displacements is None else actual_displacements[column::6]
                layout=self.layouts[key];quantity='displacement';unit='rad' if column>=3 else 'm';location='node';basis='global'
            elif key.startswith('reaction:'):
                if not reactions:
                    unavailable.append(key);continue
                component=key.split(':',1)[1];column=[c for c,_ in REACTION_COMPONENTS].index(component)
                layout=_Layout()
                for node in sorted(int(n) for n in reactions):layout.add(node,1)
                row=np.asarray([reactions[node][column] for node in layout.element_ids],dtype=np.float64)
                quantity='reaction';unit=REACTION_COMPONENTS[column][1];location='node';basis='global'
            else:
                layout=self.layouts[key];row=self._element_row(key,case_result)
                quantity,unit,_metadata=_quantity_of(key);component=key;location='element';basis=_physical_basis(key)
                info=self.descriptor_info.get(key,{})
                quantity=info.get('quantity_id',quantity);unit=info.get('unit',unit);basis=info.get('basis',basis)
            if row is None:
                unavailable.append(key);continue
            row=np.asarray(row,dtype=np.float64).reshape(-1)
            _require(len(row)==layout.count() and bool(np.all(np.isfinite(row))),
                     'source observer field is incomplete or nonfinite')
            metadata=self._observer_layout(key,case_result,layout)
            fields[key]=MappingProxyType({'key':key,'quantity_id':quantity,'component':component,
                'location':location,'unit':unit,'basis':basis,'values':_observer_freeze(row),**metadata})
        payload=MappingProxyType({'schema':'anysolver.source_fields','version':1,
            'model_identity':self.model_identity,'mesh_identity':self.mesh_identity,
            'case':_observer_freeze(case_meta),'fields':MappingProxyType(fields),'unavailable_components':tuple(unavailable),
            'excluded_components':EXCLUDED_DERIVED})
        array_layouts=tuple(_observer_array_facts(field[name])
            for field in fields.values() for name in ('values','associations','native_ordinals'))
        for facts in array_layouts:
            value,base,dtype,shape,strides,pointer=facts[0][0]
            _require(strides==(8,) and value.flags.c_contiguous and not value.flags.writeable,
                     'source observer array layout changed')
            _require_observer_array_facts(facts)
        # Retain checker bindings before external code; failures cannot redirect
        # the post-observation authority checks through mutable globals.
        capsule_guard=_require_capsule_runtime
        source_guard=_require_native_response_authority
        environment_guard=_require_observer_environment
        capsule_guard();source_guard(self.evaluator);self.evaluator._require_open()
        self._export_cancellation_point('recovery_capsule.source_observer.before')
        failure=None
        try:self.source_field_observer(payload)
        except BaseException as error:failure=error
        try:
            capsule_guard();source_guard(self.evaluator);self.evaluator._require_open();environment_guard()
            self._export_cancellation_point('recovery_capsule.source_observer.after')
            for facts in array_layouts:
                _require_observer_array_facts(facts)
        except BaseException as secondary:
            if failure is not None:
                if failure.__cause__ is None:raise failure from secondary
                try:BaseException.add_note(failure,'secondary source observer guard failure: '+type(secondary).__name__+': '+str(secondary))
                except BaseException:pass
                raise failure
            raise
        if failure is not None:
            raise failure

    def _check_replay_memory(self, workspace):
        """Account for retained output bases/selectors and one replay workspace."""
        retained = sum(array.nbytes for array in self.responses.values())
        retained += getattr(self, '_storage_prediction_retained', 0)
        if self.displacement_columns is not None:
            retained += self.displacement_columns.nbytes
        retained += sum(value[2].nbytes for value in self._dependency_indices.values() if value[2] is not None)
        retained += sum(value[2].nbytes for value in self._numeric_indices.values() if value[2] is not None)
        self.evaluator._check_library_memory(self.evaluator._library_bytes+retained+2*self._pending_numeric_bytes()+workspace)

    def _aligned_input(self, source_key, target_key, row):
        source, target = self.layouts[source_key], self.layouts[target_key]
        left,right = source.signature(),target.signature()
        key = (source_key,target_key,id(left),id(right))
        if key not in self._dependency_indices:
            if left == right:
                self._dependency_indices[key]=(left,right,None)
                return row
            self._check_replay_memory(target.count()*8+source.count()*24)
            offsets,position = {},0
            for eid,size in zip(*left):
                offsets[eid]=(position,size); position += size
            pieces=[]
            for eid,size in zip(*right):
                if eid not in offsets or offsets[eid][1] != size:
                    return None
                start=offsets[eid][0]
                pieces.append(np.arange(start,start+size,dtype=np.intp))
            indices=np.concatenate(pieces) if pieces else np.empty(0,dtype=np.intp)
            indices.flags.writeable=False
            self._dependency_indices[key]=(left,right,indices)
        indices=self._dependency_indices[key][2]
        return row if indices is None else row[indices]

    def _admit_derived(self,index,case_result,record,portable):
        visited=set()
        def admit(key):
            if key in visited:
                return
            visited.add(key)
            spec=self.derived[key]
            for dependency in spec['inputs'].values():
                if dependency in self.derived:
                    admit(dependency)
            actual=self._element_row(key,case_result)
            if actual is None:
                record['unavailable_components'].append(key)
                return
            spectral=spec['operation']=='shell_principal_stress'
            if not portable:
                record['correction_reasons'][key]='source_nonportable'
                return  # original nonportable path already emitted the row
            reason='spectral_replay_not_certified_portable' if spectral else None
            if not spectral:
                self._check_replay_memory(self.layouts[key].count()*(len(spec['inputs'])+12)*8)
                inputs={}
                for name,dependency in spec['inputs'].items():
                    value=self._element_row(dependency,case_result)
                    if value is None:
                        reason='missing_source_dependency';break
                    value=self._aligned_input(dependency,key,value)
                    if value is None:
                        reason='unsupported_dependency_layout';break
                    inputs[name]=value
                if reason is None:
                    try:
                        with np.errstate(over='ignore',invalid='ignore'):
                            candidate=derive_field(spec,inputs)
                        if not float_bits_equal(candidate,actual):
                            reason='derived_source_bits_differ'
                    except DerivationError:
                        reason='derived_replay_refused'
            if reason is not None:
                self._add_stored_row(key,index,actual)
                record['correction_reasons'][key]=reason
        for key in self.derived:
            admit(key)

    def _published_derived(self):
        return {key:spec for key,spec in self.derived.items()
                if self.replay_version == 1 or spec['operation'] != 'shell_principal_stress'}

    def _element_row(self, key: str, case_result: Mapping[str, Any]) -> Optional[np.ndarray]:
        if isinstance(case_result, NumericCase):
            rank = None
            if key.startswith('principal_stress_'):
                name, surface = key.split(':',1)
                rank = int(name.removeprefix('principal_stress_'))
                field_name = 'principal_stresses_'+surface
            else:
                field_name = self._stored_derived_fields.get(key,key.rsplit(':',1)[0] if key in self.derived else key)
            field = case_result.fields.get(field_name)
            if field is None:
                return None
            layout = self.layouts[key]
            cache_key = (key,id(field.offsets),id(layout.signature()))
            if cache_key not in self._numeric_indices:
                if rank is None and field.element_ids == tuple(layout.element_ids) and np.array_equal(np.diff(field.offsets),layout.sizes):
                    indices = None
                else:
                    positions = {eid:i for i,eid in enumerate(field.element_ids)}
                    chunks = []
                    for eid,size in zip(layout.element_ids,layout.sizes):
                        if eid not in positions:
                            return None
                        position = positions[eid]
                        begin,end = field.offsets[position:position+2]
                        if rank is not None:
                            if field.shapes[position] != (size,3):
                                return None
                            chunks.append(begin+np.arange(size)*3+rank-1)
                        else:
                            if end-begin != size:
                                return None
                            chunks.append(np.arange(begin,end))
                    indices = np.concatenate(chunks) if chunks else np.empty(0,dtype=np.intp)
                    indices.flags.writeable = False
                self._numeric_indices[cache_key] = (field.offsets,layout.signature(),indices)
            indices = self._numeric_indices[cache_key][2]
            return field.values if indices is None else field.values[indices]
        extractor = self.extractors.get(key)
        if extractor is not None:
            return extractor(case_result)
        field_name = key.rsplit(':', 1)[0] if key in self.derived else key
        return _stored_row(case_result, field_name, self.layouts[key])

    def _add_stored_row(self, key: str, index: int, row: np.ndarray) -> None:
        layout = self.layouts[key]
        self.stored_rows.setdefault(key, []).append((layout.signature(), index))
        self.pending_rows.setdefault(key, []).append((index, row))

    # -- batch and finalization ------------------------------------------

    def _process_batch(self, sink, batch) -> None:
        if self.source_field_observer is not None:
            self._observer_batch_bytes=_observer_numeric_bytes(batch.case_results)+_observer_numeric_bytes(batch.displacements)
        for position, case_result in zip(batch.input_indices, batch.case_results):
            self._guard_runtime()
            record, coefficients = self._process_case(int(position), case_result)
            self._guard_runtime()
            sink.capsule_case(_readonly_data(record), _readonly_data(coefficients))
            self._guard_runtime()

    def _guard_runtime(self):
        _require_capsule_runtime()
        if self.storage_prediction is not None:_require_storage_schema_authority()
        self._runtime_guard_checks += 1
        if self._native_affinity_opt_in:
            started=_capsule_guard_clock_ns()
            _require_native_response_authority(self.evaluator)
            self.evaluator._require_open()
            self._native_response_guard_checks+=1
            self._native_response_guard_ns+=_capsule_guard_clock_ns()-started

    def _export_cancellation_point(self, stage):
        if self.cancellation_token is None:
            return
        # Hold both proofs across the caller token observation: the callback
        # cannot replace the next checker or an evaluator method unnoticed.
        capsule_guard = _require_capsule_runtime
        response_guard = _require_native_response_authority
        capsule_guard()
        response_guard(self.evaluator)
        cancellation_safe_point(self.cancellation_token, stage)
        capsule_guard()
        response_guard(self.evaluator)
        self.evaluator._require_open()

    def _pending_numeric_bytes(self) -> int:
        """Numeric bytes of stored rows not yet emitted to the sink."""
        return int(sum(row.nbytes
                       for pending in self.pending_rows.values()
                       for _index, row in pending))

    def _stored_events(self) -> Dict[str, Dict[str, Any]]:
        """Per-batch stored-row increments not yet emitted to the sink.

        The emitted numeric rows are released for garbage collection as soon
        as the sink has consumed them; only the compact signature/index
        metadata is retained for the final descriptors and validity
        accounting, so pending memory is bounded by one batch, not by the
        case count.
        """
        events: Dict[str, Dict[str, Any]] = {}
        for key in sorted(self.pending_rows):
            pending = self.pending_rows[key]
            if not pending:
                continue
            events[key] = {
                'case_indices': np.asarray([index for index, _row in pending],
                                           dtype=np.int64),
                'values': np.ascontiguousarray(
                    np.stack([row for _index, row in pending]), dtype=np.float64),
            }
            pending.clear()
        return events

    def _descriptor_recovery_refs(self, key: str) -> Dict[str, Any]:
        """Compact references into the shared once-per-model metadata tables."""
        rules: set = set()
        frames = False
        for element_id in self.layouts[key].element_ids:
            binding = self.recovery_bindings.get(int(element_id))
            if binding is None:
                continue
            if binding.get('sampling_rule'):
                rules.add(binding['sampling_rule'])
            frames = frames or bool(binding.get('frame'))
        refs: Dict[str, Any] = {}
        if rules:
            refs['sampling_rules'] = sorted(rules)
        if frames:
            refs['local_frames'] = 'model_recovery_metadata.frames'
        return refs

    def _stored_element_shapes(self, key: str) -> Optional[Dict[str, List[int]]]:
        """Per-element source shapes of one stored component, when observed."""
        shapes = self.native_shapes.get(key)
        if shapes is None and key.startswith('principal_stress_'):
            surface = key.split(':', 1)[1]
            native = self.native_shapes.get(f'principal_stresses_{surface}')
            if native:
                # The per-rank component exports one scalar per native row.
                shapes = {element_id: [native_shape[0]] for element_id, native_shape
                          in native.items() if len(native_shape) == 2}
        if not shapes:
            return None
        return {str(element_id): list(shape)
                for element_id, shape in sorted(shapes.items())}

    def _descriptors(self) -> Tuple[Dict[str, Any], ...]:
        displacement_units = dict(DISPLACEMENT_COMPONENTS)
        reaction_units = dict(REACTION_COMPONENTS)
        descriptors = []
        for key in sorted(self.layouts):
            stored_indices = [index for _signature, index in self.stored_rows.get(key, ())]
            if key in self.derived:
                info = dict(self.descriptor_info[key])
            elif key in self.descriptor_info:
                info = dict(self.descriptor_info[key])
            elif key.startswith('displacement:'):
                component = key.split(':', 1)[1]
                info = {'quantity_id': 'displacement', 'component': component,
                        'location': 'node', 'unit': displacement_units[component],
                        'basis': 'global', 'sampling': 'node',
                        'reconstruction': 'compiled_affine_pattern_basis',
                        'metadata': {'dof_association': 'node_dofs',
                                     'parity': 'exact_bitwise_per_case'}}
            elif key.startswith('reaction:'):
                component = key.split(':', 1)[1]
                info = {'quantity_id': 'reaction', 'component': component,
                        'location': 'node', 'unit': reaction_units[component],
                        'basis': 'global', 'sampling': 'node',
                        'reconstruction': 'explicit_per_case_values',
                        'metadata': {'source': 'actual_full_Ku_minus_F',
                                     'support_ids_in_case_provenance': True}}
            else:
                quantity_id, unit, metadata = _quantity_of(key)
                if key in self.responses:
                    reconstruction = 'compiled_affine_pattern_basis'
                elif stored_indices:
                    reconstruction = 'explicit_per_case_values'
                else:
                    reconstruction = 'none'
                info = {'quantity_id': quantity_id, 'component': key,
                        'location': 'element', 'unit': unit,
                        'basis': _physical_basis(key), 'sampling': 'element_sample',
                        'reconstruction': reconstruction,
                        'metadata': metadata}
            if info['location'] == 'element':
                info['metadata'] = {**info['metadata'], **self._descriptor_recovery_refs(key)}
                native_ids=[eid for eid in sorted(self.evaluator._native_affinity_fallbacks) if self.layouts[key].size_of(eid) is not None]
                if native_ids and key not in self.responses and key not in self.derived:
                    info['metadata']={**info['metadata'],'native_affinity_fallback_elements':native_ids,
                        'portable_replay_refusal':'native_numerical_affinity_fallback; complete source rows stored'}
            if (key in self.responses or key in self.derived
                    or (key.startswith('displacement:')
                        and self.displacement_columns is not None)):
                availability = 'reconstructable'
            elif stored_indices:
                availability = 'stored'
            else:
                availability = 'unavailable'
            if self.replay_version == 2 and key.startswith('principal_stress_'):
                availability = 'stored' if stored_indices else 'unavailable'
                info['reconstruction'] = 'explicit_per_case_values'
                info['metadata'] = {**info['metadata'],
                    'portable_replay_refusal':'spectral_replay_not_certified_portable'}
            descriptor = {
                'key': key,
                'quantity_id': info['quantity_id'],
                'component': info['component'],
                'location': info['location'],
                'unit': info['unit'],
                'basis': info['basis'],
                'reconstruction': info['reconstruction'],
                'sampling': info['sampling'],
                'availability': availability,
                'recovery_route': f'prepared_static_response/{self.stress_execution}',
            }
            if availability != 'unavailable':
                # The source envelope reducers (static and prepared) retain
                # the exact earliest original input_index on equal extrema;
                # every available component declares that tie policy so a
                # consumer can reproduce the solver's governing-case identity.
                descriptor['tie_policy'] = 'exact_earliest_input_index'
            if info['metadata']:
                descriptor['metadata'] = _plain_json(info['metadata'])
            if self.basis and key in self.basis['shapes']:
                descriptor['element_shapes'] = {str(eid): shape
                                               for eid, shape in sorted(self.basis['shapes'][key].items())}
            else:
                stored_shapes = self._stored_element_shapes(key)
                if stored_shapes:
                    descriptor['element_shapes'] = stored_shapes
            if stored_indices:
                descriptor['stored_case_count'] = len(stored_indices)
            if key.startswith('displacement:'):
                descriptor['portable_case_count'] = sum(
                    1 for flag in self.displacement_portable if flag)
            descriptors.append(descriptor)
        return tuple(descriptors)

    def _merged_responses(self) -> Dict[str, np.ndarray]:
        """Portable response columns including the displacement components."""
        responses = dict(self.responses)
        if self.displacement_columns is not None:
            for position, key in enumerate(self.displacement_keys):
                responses[key] = np.ascontiguousarray(
                    self.displacement_columns[position::6, :], dtype=np.float64)
        return responses

    def _finalize(self) -> Dict[str, Any]:
        associations = {key: self.layouts[key].association() for key in sorted(self.layouts)}
        superseded: Dict[str, List[int]] = {}
        for key, entries in self.stored_rows.items():
            final_signature = self.layouts[key].signature()
            dropped = sorted({index for signature, index in entries
                              if signature != final_signature})
            if dropped:
                superseded[key] = dropped
        _require(not any(superseded.values()),
                 'stored rows were superseded by a component layout change; '
                 'refusing to silently drop previously exported rows')
        responses = self._merged_responses()
        result = {
            'descriptors': self._descriptors(),
            'associations': associations,
            'responses': {name: responses[name] for name in sorted(responses)},
            'derived': dict(sorted(self._published_derived().items())),
            'element_constants': {str(eid): dict(values)
                                  for eid, values in sorted(self.element_constants.items())},
            'stored_superseded_cases': superseded,
            'case_count': len(self.cases),
        }
        if self.replay_version == 2:
            result['replay_diagnostics']={'runtime_guard_checks':self._runtime_guard_checks}
            if self._native_affinity_opt_in:
                result['replay_diagnostics'].update(native_affinity_response_guard_checks=self._native_response_guard_checks,
                    native_affinity_response_guard_seconds=self._native_response_guard_ns/1.e9)
        if self.storage_prediction is not None and any(
                self.stored_rows.get(key) for key in self.storage_prediction['targets']):
            validate_storage_prediction(self.storage_prediction,patterns=self.pattern_names,
                                        descriptors=result['descriptors'],associations=associations)
            result['storage_prediction']=self.storage_prediction
            result.setdefault('replay_diagnostics',{})['storage_prediction_retained_bytes']=self._storage_prediction_retained
        return result

    def run(self, sink) -> None:
        geometry_arrays = {}
        if self.geometry is not None:
            _require(isinstance(self.geometry, Mapping), 'geometry must be a mapping of array names to arrays')
            for name, value in self.geometry.items():
                _require(type(name) is str, 'geometry names must be strings')
                array = np.asarray(value)
                _require(array.dtype.kind in 'biuf', f'geometry array {name!r} must be numeric')
                # Geometry is supplied by the caller rather than an owned
                # solver response array; capture it once before publication.
                geometry_arrays[name] = np.array(array, copy=True, order='C')

        identity = dict(self.solver_identity) if self.solver_identity is not None else {}
        from . import __version__
        identity.setdefault('solver', 'ANYfeSolver')
        identity.setdefault('version', __version__)
        identity.setdefault('stress_execution', self.stress_execution)
        identity.setdefault('export_contract', self.contract_id)
        identity.setdefault('runtime', self.runtime_id)

        snapshot = self.evaluator._snapshot
        # Shared once-per-model recovery metadata, captured from the
        # snapshot's private recipe owners before any case is processed.
        self.recovery_metadata, self.recovery_bindings = _model_recovery_metadata(
            snapshot.recovery_compilation)
        model_provenance = {
            'export_contract': self.contract_id,
            'runtime': self.runtime_id,
            'snapshot_content_digest': str(snapshot.content_digest),
            'source_revision': {str(name): int(value)
                                for name, value in dict(snapshot.source_revision).items()},
            'captured': 'prepared_static_snapshot',
            'model_identity_binding': 'caller_stable_id_bound_to_snapshot_digest',
        }
        if self.evaluator._native_affinity_fallbacks:
            model_provenance['native_affinity_fallback']={
                'policy':'explicit_opt_in_finite_numerical_probe_refusals',
                'elements':sorted(self.evaluator._native_affinity_fallbacks),
                'refusals':{str(eid):[dict(reason) for reason in entry[5]] for eid,entry in sorted(self.evaluator._native_affinity_fallbacks.items())}}

        responses = self._merged_responses()
        header = {
            'schema': SCHEMA,
            'version': self.replay_version,
            'runtime': self.runtime_id,
            'contract': self.contract_id,
            'model_identity': self.model_identity,
            'mesh_identity': self.mesh_identity,
            'model_provenance': model_provenance,
            'model_recovery_metadata': self.recovery_metadata,
            'solver_identity': identity,
            'geometry': geometry_arrays,
            'patterns': self.pattern_names,
            'responses': {name: responses[name] for name in sorted(responses)},
            'associations': {key: self.layouts[key].association()
                             for key in sorted(self.layouts)
                             if key in self.responses or key.startswith('displacement:')},
            'derived': dict(sorted(self._published_derived().items())),
        }
        if self.replay_version == 2:
            header['combination']=self.combination
            header['replay_assumptions']={'precision':'binary64','rounding':'requires_round_to_nearest',
                'subnormal_handling':'requires_preservation; not a cross-platform certificate',
                'spectral':'explicit_source_values'}
        if self.storage_prediction is not None:
            header['storage_prediction']=self.storage_prediction
        self._export_cancellation_point('recovery_capsule.before_publication')
        sink.capsule_begin(_readonly_data(header))
        self._guard_runtime()

        completed = 0
        if self.cases:
            iterator = (self.evaluator._iter_capsule_batches if self.recovery_execution == 'auto'
                        else self.evaluator.iter_batches)
            for batch in iterator(list(self.cases), batch_size=self.batch_size,
                                  cancellation_token=self.cancellation_token):
                self._process_batch(sink, batch)
                completed += len(batch.input_indices)
                events = self._stored_events()
                for key in sorted(events):
                    event = events[key]
                    sink.capsule_stored(key, _readonly_data(event['case_indices']),
                                       _readonly_data(event['values']))
                    self._guard_runtime()
                # The batch and its detailed case result dictionaries go out
                # of scope here; only compact records and per-component rows
                # are retained.
        _require(completed == len(self.cases), 'the evaluator did not complete every case')
        final = self._finalize()
        _require(bool(final['associations']),
                 'the evaluator produced no recoverable fields for a capsule')
        self._export_cancellation_point('recovery_capsule.before_final_publication')
        if self._native_affinity_opt_in:self._guard_runtime()
        sink.capsule_end(_readonly_data(final))
        self._guard_runtime()


def _require_sink(sink: Any) -> None:
    for name in SINK_METHODS:
        _require(callable(getattr(sink, name, None)),
                 f'recovery capsule sink requires a callable {name} method')


def export_recovery_capsule_to_sink(evaluator, cases, *, sink, model_identity,
                                    mesh_identity, geometry=None, solver_identity=None,
                                    batch_size=None, recovery_execution='auto', replay_version=2,
                                    cancellation_token=None, source_field_observer=None):
    """Stream one recovery capsule export through an owner-neutral sink.

    The sink is a structural protocol (``SINK_METHODS``): ``capsule_begin``
    receives the static metadata (identities, patterns, geometry and, on the
    portable route, the response bases), ``capsule_case`` receives one compact
    case record plus its coefficient row, ``capsule_stored`` receives
    per-batch stored-row increments per component, and ``capsule_end``
    receives the final descriptors, associations, derived specifications,
    element constants and any superseded stored cases.  Detailed case result
    dictionaries are consumed and discarded per bounded batch; the exporter
    never retains them.

    New exports default to replay_version=2 (ordered binary64 replay with exact
    source corrections). Explicit replay_version=1 preserves the legacy native
    BLAS replay contract and its documented limitations.
    """
    _require_sink(sink)
    try:
        _require_capsule_runtime()
        export = _CapsuleExport(evaluator, cases, model_identity=model_identity,
                                mesh_identity=mesh_identity, geometry=geometry,
                                solver_identity=solver_identity, batch_size=batch_size,
                                recovery_execution=recovery_execution,replay_version=replay_version,
                                cancellation_token=cancellation_token, source_field_observer=source_field_observer)
        export.run(sink)
    except BaseException as error:
        _report_export_failure(sink, error, cases, model_identity, mesh_identity)
        raise


def export_recovery_capsule(evaluator, cases, *, model_identity, mesh_identity,
                            geometry=None, solver_identity=None, batch_size=None, recovery_execution='auto', replay_version=2,
                            cancellation_token=None):
    """Export one recovery capsule for a completed certified pattern-case batch.

    The evaluator must be an open :class:`PreparedStaticResponseEvaluator`;
    ``cases`` is a finite sequence of :class:`PatternCase` instances with
    unique ids.  The batch is consumed through the evaluator's certified batch
    iterator in bounded increments (``batch_size`` defaults to 32) and case
    result dictionaries are discarded per increment; the returned mapping is
    output-sized (stored arrays plus compact case records), not a retention
    of all detailed case results.  Components are reconstructable only where
    the evaluator reconstructed them on its admitted compiled basis; every
    other certified case is exported with explicit stored arrays and its
    actual route label.  Failed cases remain discoverable and never yield
    success fields.

    replay_version=2 is the default; it preserves actual source values through
    exact primitive, displacement and derived corrections. Principal ranks are
    explicit source values until a portable spectral contract is proven.
    """
    sink = _MemorySink()
    export_recovery_capsule_to_sink(evaluator, cases, sink=sink,
                                    model_identity=model_identity, mesh_identity=mesh_identity,
                                    geometry=geometry, solver_identity=solver_identity,
                                    batch_size=batch_size,recovery_execution=recovery_execution,replay_version=replay_version,
                                    cancellation_token=cancellation_token)
    from anyfesolver_results import validate_capsule
    validate_capsule(sink.capsule)
    return _readonly_data(sink.capsule)


from .capsule_runtime_guard import make_capsule_runtime_guard
_require_capsule_runtime = make_capsule_runtime_guard(globals(),
    (reconstruct_response,derive_field,validate_storage_prediction),(_CapsuleExport,_Layout,_MemorySink))
