"""Capsule validation and response reconstruction (NumPy only)."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .errors import CapsuleValidationError, ReconstructionError

SCHEMA = 'anysolver.recovery_capsule'
VERSION = 2
RUNTIME_ID = 'anysolver-results/2'
SUPPORTED_RUNTIME_IDS = ('anysolver-results/1','anysolver-results/2')
SUPPORTED_COMBINATIONS = ('legacy_blas/1','ordered_float64/1')
AVAILABILITIES = ('stored', 'reconstructable', 'unavailable')
# The completed-status set ANYpost admits for certified cases; a certified
# failure status is never part of it.
COMPLETED_STATUSES = ('completed', 'complete', 'success', 'succeeded', 'ok', 'converged')
GEOMETRY_NAMES = ('positions', 'triangles', 'lines', 'triangle_to_element',
                  'line_to_element', 'node_ids', 'element_ids')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CapsuleValidationError(message)


def _is_plain_json(value: Any) -> bool:
    if value is None or type(value) in (bool, str):
        return True
    if type(value) is int:
        return True
    if type(value) is float:
        # JSON has no NaN or infinity; a non-finite float is not plain data.
        return math.isfinite(value)
    # Solver snapshots can use JSON-compatible read-only dict/list
    # subclasses. Their contents still obey the same finite plain-data
    # admission; no arbitrary objects or executable archive values enter.
    if isinstance(value, list):
        return all(_is_plain_json(item) for item in value)
    if isinstance(value, dict):
        return all(type(key) is str and _is_plain_json(item) for key, item in value.items())
    return False


def validate_capsule(capsule: Mapping[str, Any]) -> None:
    """Lightweight shape/identity/status checks for a recovery capsule.

    This is a structural admission check, not a certification: it refuses
    malformed capsules so consumers never guess at layout, and it admits
    failed cases (they stay discoverable) while reconstruction itself only
    ever admits explicitly completed certified cases.
    """
    if not isinstance(capsule, Mapping):
        raise CapsuleValidationError('capsule must be a mapping')
    _require(capsule.get('schema') == SCHEMA, 'capsule schema is not ' + SCHEMA)
    version = capsule.get('version')
    _require(type(version) is int and version in (1,2), 'unsupported capsule version')
    _require(capsule.get('runtime') == f'anysolver-results/{version}', 'capsule runtime identity mismatch')
    expected = 'legacy_blas/1' if version == 1 else 'ordered_float64/1'
    _require(capsule.get('combination',expected if version == 1 else None) == expected,
             'capsule combination identity mismatch')
    for name in ('model_identity', 'mesh_identity'):
        identity = capsule.get(name)
        _require(type(identity) is str and identity, f'{name} must be a nonempty string')
    _require(isinstance(capsule.get('solver_identity'), Mapping) and _is_plain_json(dict(capsule['solver_identity'])),
             'solver_identity must be a JSON mapping')
    provenance = capsule.get('model_provenance')
    if provenance is not None:
        _require(isinstance(provenance, Mapping) and _is_plain_json(dict(provenance)),
                 'model_provenance must be a JSON mapping when present')
        digest = provenance.get('snapshot_content_digest')
        _require(type(digest) is str and digest,
                 'model_provenance needs a nonempty snapshot_content_digest')
    recovery_metadata = capsule.get('model_recovery_metadata')
    if recovery_metadata is not None:
        # Shared once-per-model frames/sampling tables: plain JSON only.
        _require(isinstance(recovery_metadata, Mapping)
                 and _is_plain_json(dict(recovery_metadata)),
                 'model_recovery_metadata must be a JSON mapping when present')
    constants = capsule.get('element_constants')
    if constants is not None:
        _require(isinstance(constants, Mapping), 'element_constants must be a mapping when present')
        for element_id, values in constants.items():
            _require(type(element_id) is str, 'element_constants keys must be strings')
            _require(isinstance(values, Mapping) and _is_plain_json(dict(values)),
                     f'element_constants[{element_id!r}] must be a JSON mapping')
    geometry = capsule.get('geometry')
    _require(isinstance(geometry, Mapping), 'geometry must be a mapping')
    for name, value in geometry.items():
        _require(type(name) is str, 'geometry names must be strings')
        _require(isinstance(value, np.ndarray) and value.dtype.kind in 'biuf',
                 f'geometry array {name!r} must be a numeric array')
    if version == 2:
        positions=geometry.get('positions')
        if positions is not None:
            _require(positions.ndim == 2 and positions.shape[1] == 3
                     and np.all(np.isfinite(positions)),'positions must be finite (node_count,3)')
        for name in ('element_ids','node_ids'):
            if name in geometry:
                value=geometry[name]
                _require(value.dtype.kind in 'iu' and value.ndim == 1,
                         f'{name} must be a one-dimensional integer identity table')
                _require(len(np.unique(value)) == len(value),f'{name} must be unique')
                if name == 'node_ids':
                    _require(positions is not None and len(value) == len(positions),'node_ids must match positions')
        for topology,width in (('triangles',3),('lines',2)):
            if topology in geometry:
                indices=geometry[topology]
                _require(positions is not None and indices.dtype.kind in 'iu'
                         and indices.ndim == 2 and indices.shape[1] == width,
                         f'{topology} must be dense node indices with width {width}')
                _require(np.all(indices >= 0) and np.all(indices < len(positions)),
                         f'{topology} node indices are out of bounds')
        for name, topology in (('triangle_to_element','triangles'),('line_to_element','lines')):
            if name in geometry:
                value=geometry[name]
                _require(value.dtype.kind in 'iu' and value.ndim == 1,
                         f'{name} must be a one-dimensional integer mapping')
                _require(topology in geometry and len(value) == len(geometry[topology]),
                         f'{name} must match {topology} rows')
                _require('element_ids' in geometry,'dense element mappings require element_ids')
                _require(np.all(value >= 0) and np.all(value < len(geometry['element_ids'])),
                         f'{name} must index element_ids')
    patterns = capsule.get('patterns')
    _require(type(patterns) is tuple, 'patterns must be a tuple')
    _require(all(type(name) is str for name in patterns), 'pattern names must be strings')
    _require(len(set(patterns)) == len(patterns), 'pattern names must be unique')
    pattern_count = len(patterns)
    coefficients = capsule.get('coefficients')
    _require(isinstance(coefficients, np.ndarray) and coefficients.dtype == np.float64
             and coefficients.ndim == 2 and coefficients.shape[1] == pattern_count,
             'coefficients must be float64 (case_count, pattern_count)')
    _require(np.all(np.isfinite(coefficients)), 'coefficients must be finite')
    case_count = coefficients.shape[0]
    cases = capsule.get('cases')
    _require(type(cases) is tuple and len(cases) == case_count,
             'cases must be a tuple matching the coefficient rows')
    seen_cases = set()
    for case in cases:
        _require(isinstance(case, Mapping) and _is_plain_json(dict(case)),
                 'each case must be a JSON mapping')
        case_id = case.get('case_id')
        _require(type(case_id) is str and case_id, 'each case needs a nonempty case_id')
        _require(case_id not in seen_cases, 'case_id values must be unique')
        seen_cases.add(case_id)
        _require(type(case.get('status')) is str, f'case {case_id!r} needs a status string')
        _require(type(case.get('certified')) is bool, f'case {case_id!r} needs a certified flag')
        _require(type(case.get('route')) is str and case['route'],
                 f'case {case_id!r} needs a route string')
        if case['certified']:
            _require(case['status'] in COMPLETED_STATUSES,
                     f'case {case_id!r} is certified but its status is not a completed status')
        active_indices = case.get('active_indices')
        if active_indices is not None:
            # One-based original-order pattern positions, unique and in range,
            # validated exactly like a reconstruction request.
            try:
                _integer_active_indices(active_indices, pattern_count + 1)
            except ReconstructionError as error:
                raise CapsuleValidationError(str(error)) from error
        portable = case.get('portable')
        if portable is not None:
            _require(type(portable) is bool, f'case {case_id!r} needs a boolean portable flag')
            if portable:
                _require(case['certified'],
                         f'case {case_id!r} is portable but not certified')
        if 'displacement_portable' in case:
            _require(case['displacement_portable'] is None or type(case['displacement_portable']) is bool,
                     'displacement_portable must be boolean or null')
        if 'unavailable_components' in case:
            value=case['unavailable_components']
            _require(isinstance(value,list) and all(type(key) is str for key in value)
                     and len(value)==len(set(value)), 'unavailable_components must contain unique strings')
        if 'correction_reasons' in case:
            reasons=case['correction_reasons']
            _require(isinstance(reasons,Mapping) and all(type(key) is str and type(reason) is str
                     for key,reason in reasons.items()),'correction_reasons must map strings to strings')
    descriptors = capsule.get('descriptors')
    _require(type(descriptors) is tuple, 'descriptors must be a tuple')
    keys = set()
    pairs = set()
    for descriptor in descriptors:
        _require(isinstance(descriptor, Mapping) and _is_plain_json(dict(descriptor)),
                 'each descriptor must be a JSON mapping')
        key = descriptor.get('key')
        _require(type(key) is str and key, 'each descriptor needs a nonempty key')
        _require(key not in keys, 'descriptor keys must be unique')
        keys.add(key)
        for name in ('quantity_id', 'component', 'location', 'unit', 'basis',
                     'sampling'):
            _require(type(descriptor.get(name)) is str,
                     f'descriptor {key!r} needs a string {name}')
        # `reconstruction` is an optional v1 extension: absent is admitted,
        # but a present value must be a string (None is not a string).
        if 'reconstruction' in descriptor:
            _require(type(descriptor['reconstruction']) is str,
                     f'descriptor {key!r} needs a string reconstruction')
        pair = (descriptor['quantity_id'], descriptor['component'])
        _require(pair not in pairs,
                 'descriptor (quantity_id, component) pairs must be unique')
        pairs.add(pair)
        _require(descriptor.get('availability') in AVAILABILITIES,
                 f'descriptor {key!r} has an unknown availability')
    if version == 2:
        for case in cases:
            _require(type(case.get('portable')) is bool,'runtime2 cases require a portable flag')
            _require(set(case.get('unavailable_components',())) <= keys,
                     'unavailable_components reference unknown descriptors')
            _require(set(case.get('correction_reasons',{})) <= keys,
                     'correction_reasons reference unknown descriptors')
    associations = capsule.get('associations')
    _require(isinstance(associations, Mapping), 'associations must be a mapping')
    for key, value in associations.items():
        _require(type(key) is str, 'association keys must be strings')
        _require(isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in 'iu',
                 f'association {key!r} must be a 1-D integer array')
    responses = capsule.get('responses')
    _require(isinstance(responses, Mapping), 'responses must be a mapping')
    spatial = {}
    for key, value in responses.items():
        _require(type(key) is str, 'response keys must be strings')
        _require(isinstance(value, np.ndarray) and value.dtype == np.float64 and value.ndim == 2
                 and value.shape[1] == pattern_count + 1,
                 f'response {key!r} must be float64 (spatial_count, pattern_count+1)')
        _require(np.all(np.isfinite(value)), f'response {key!r} must be finite')
        _require(key in associations, f'response {key!r} has no association array')
        _require(key in keys, f'response {key!r} has no descriptor')
        association = associations[key]
        _require(association.shape[0] == value.shape[0],
                 f'response {key!r} disagrees with its association length')
        spatial[key] = value.shape[0]
    stored = capsule.get('stored')
    _require(isinstance(stored, Mapping), 'stored must be a mapping')
    for key, value in stored.items():
        _require(type(key) is str, 'stored keys must be strings')
        _require(isinstance(value, Mapping), f'stored {key!r} must be a mapping')
        indices = value.get('case_indices')
        values = value.get('values')
        _require(isinstance(indices, np.ndarray) and indices.ndim == 1 and indices.dtype.kind in 'iu',
                 f'stored {key!r} needs integer case_indices')
        _require(isinstance(values, np.ndarray) and values.dtype == np.float64 and values.ndim == 2,
                 f'stored {key!r} needs float64 values (stored_case_count, spatial_count)')
        _require(np.all(np.isfinite(values)), f'stored {key!r} values must be finite')
        _require(indices.shape[0] == values.shape[0],
                 f'stored {key!r} case_indices and values disagree')
        _require(np.all((indices >= 0) & (indices < case_count)),
                 f'stored {key!r} case_indices are out of range')
        _require(len(set(indices.tolist())) == indices.shape[0],
                 f'stored {key!r} case_indices must be unique')
        _require(key in associations, f'stored {key!r} has no association array')
        _require(key in keys, f'stored {key!r} has no descriptor')
        _require(associations[key].shape[0] == values.shape[1],
                 f'stored {key!r} disagrees with its association length')
        if key in spatial:
            _require(spatial[key] == values.shape[1],
                     f'stored {key!r} disagrees with the response spatial count')
    derived = capsule.get('derived')
    _require(isinstance(derived, Mapping), 'derived must be a mapping')
    for key, specification in derived.items():
        _require(type(key) is str, 'derived keys must be strings')
        _require(isinstance(specification, Mapping) and _is_plain_json(dict(specification)),
                 f'derived {key!r} must be a JSON mapping')
        _require(type(specification.get('operation')) is str,
                 f'derived {key!r} needs an operation')
        inputs = specification.get('inputs')
        _require(isinstance(inputs, Mapping), f'derived {key!r} needs an inputs mapping')
        _require(key in keys, f'derived {key!r} has no descriptor')
        for input_name, target in inputs.items():
            _require(type(input_name) is str and type(target) is str,
                     f'derived {key!r} inputs must map names to component keys')
            _require(target in keys,
                     f'derived {key!r} input {input_name!r} references '
                     f'unknown component {target!r}')
    _require_acyclical_derived(derived)
    if 'storage_prediction' in capsule:
        _require(version == 2, 'storage prediction requires capsule2')
        from .storage_validation import validate_storage_prediction
        validate_storage_prediction(capsule['storage_prediction'], patterns=patterns,
                                    descriptors=descriptors, associations=associations)
        for key in capsule['storage_prediction']['targets']:
            _require(key not in responses and key not in derived,
                     'storage-only target cannot enter engineering replay')


def _require_acyclical_derived(derived: Mapping[str, Any]) -> None:
    """Refuse derived specifications whose dependencies form a cycle."""
    targets = {key: {target for target in specification.get('inputs', {}).values()
                     if target in derived}
               for key, specification in derived.items()
               if isinstance(specification, Mapping)}
    state: dict = {}

    def visit(key: str) -> None:
        if state.get(key) == 1:
            raise CapsuleValidationError(f'derived components form a cycle at {key!r}')
        if state.get(key) == 2:
            return
        state[key] = 1
        for target in targets.get(key, ()):
            visit(target)
        state[key] = 2

    for key in targets:
        visit(key)


def _as_float_array(values: Any, message: str) -> np.ndarray:
    """Float64 conversion whose failure is the documented typed refusal."""
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ReconstructionError(message) from error


def _integer_active_indices(active_indices: Any, column_count: int) -> np.ndarray:
    """Strict 1-D integer pattern-column positions, unique and in range.

    Fractional, boolean and duplicate indices are refused instead of being
    silently coerced; the original sparse order of valid indices is preserved.
    Range and uniqueness are validated on the original values before any
    fixed-width conversion, so oversized plain integers (e.g. ``2**100``)
    surface as the documented typed refusal instead of a raw OverflowError.
    """
    if isinstance(active_indices, (list, tuple)):
        if any(type(index) is not int for index in active_indices):
            raise ReconstructionError('active_indices must be integer indices')
        if any(not 1 <= index < column_count for index in active_indices):
            raise ReconstructionError('active_indices must reference pattern columns')
        if len(set(active_indices)) != len(active_indices):
            raise ReconstructionError('active_indices must be unique')
        return np.asarray(active_indices, dtype=np.intp)
    try:
        active = np.asarray(active_indices)
    except (TypeError, ValueError, OverflowError) as error:
        raise ReconstructionError('active_indices must be an integer array') from error
    if active.ndim != 1:
        raise ReconstructionError('active_indices must be one dimensional')
    if active.size and active.dtype.kind not in 'iu':
        raise ReconstructionError('active_indices must be integer indices')
    if active.size:
        if np.any(active < 1) or np.any(active >= column_count):
            raise ReconstructionError('active_indices must reference pattern columns')
        if np.unique(active).shape[0] != active.shape[0]:
            raise ReconstructionError('active_indices must be unique')
        try:
            active = active.astype(np.intp, copy=False)
        except OverflowError as error:
            raise ReconstructionError('active_indices must be integer indices') from error
    return active


def _validated_combination(combination):
    if type(combination) is not str or combination not in SUPPORTED_COMBINATIONS:
        raise ReconstructionError('unsupported response combination')
    return combination


def _combine_response(baseline: np.ndarray, selected: np.ndarray,
                      weights: np.ndarray, combination='legacy_blas/1') -> np.ndarray:
    """Replay the named legacy BLAS or ordered binary64 accumulation.

    Both the ordinary reconstruction and the prepared replay delegate to
    this helper, so there is a single numerical implementation.  An empty
    active order reduces to the baseline alone.
    """
    combination = _validated_combination(combination)
    if selected.shape[1] == 0:
        result = np.array(baseline, dtype=np.float64, copy=True)
    elif combination == 'ordered_float64/1':
        accumulated = np.zeros(baseline.shape,dtype=np.float64)
        term = np.empty(baseline.shape,dtype=np.float64)
        with np.errstate(over='ignore',invalid='ignore'):
            for column in range(selected.shape[1]):
                np.multiply(selected[:,column],weights[column],out=term)
                np.add(accumulated,term,out=accumulated)
            result = np.add(baseline,accumulated)
        if not np.all(np.isfinite(result)):
            raise ReconstructionError('ordered response replay is nonfinite')
    else:
        result = baseline + selected @ weights
    if combination == 'ordered_float64/1':
        result=np.frombuffer(result.tobytes(),dtype=np.float64).reshape(result.shape)
    result.setflags(write=False)
    return result


def _validated_response_columns(responses: Any, active_indices: Any) -> tuple:
    """Admitted float64 response matrix and its validated active columns."""
    values = _as_float_array(responses, 'responses must be 2-D with a baseline column')
    if values.ndim != 2 or values.shape[1] < 1:
        raise ReconstructionError('responses must be 2-D with a baseline column')
    if active_indices is None:
        active = np.arange(1, values.shape[1], dtype=np.intp)
    else:
        active = _integer_active_indices(active_indices, values.shape[1])
    if not np.all(np.isfinite(values)):
        raise ReconstructionError('response columns must be finite')
    return values, active


def _validated_weights(coefficients: Any, active_count: int) -> np.ndarray:
    """Admitted 1-D finite weights matching the active pattern count."""
    weights = _as_float_array(coefficients, 'coefficients must match the active pattern count')
    if weights.ndim != 1 or weights.shape[0] != active_count:
        raise ReconstructionError('coefficients must match the active pattern count')
    if not np.all(np.isfinite(weights)):
        raise ReconstructionError('coefficients must be finite')
    return weights


def reconstruct_response(responses: Any, coefficients: Any, *, active_indices=None,
                         combination='legacy_blas/1') -> np.ndarray:
    """Reconstruct one scalar spatial field from capsule response columns.

    ``responses`` is a ``(spatial_count, pattern_count+1)`` float64 array whose
    column 0 is the baseline (added exactly once).  ``active_indices`` are
    positions in the response columns (column 0 excluded); ``None`` means all
    pattern columns. The default ``legacy_blas/1`` retains the original
    ``responses[:, 0] + responses[:, active] @ coefficients`` expression.
    ``ordered_float64/1`` materializes each multiply/add in active-column order
    and adds the baseline last. Capsule2 consumers must explicitly select it;
    the source exporter separately certifies exact bits or stores corrections.
    """
    values, active = _validated_response_columns(responses, active_indices)
    weights = _validated_weights(coefficients, active.shape[0])
    return _combine_response(values[:, 0], values[:, active], weights,combination)


_CREATE = object()


class PreparedResponse:
    """Immutable owned baseline and selected response columns for replay.

    Instances are created only by :func:`prepare_response`; the construction
    token keeps archives from inventing a prepared object.  ``baseline`` is
    the copied baseline column, ``selected`` the copied original-order active
    columns (exactly the array the ordinary reconstruction expression
    materializes), both read-only float64 without any alias to caller-mutable
    memory.  ``nbytes`` is the exact total retained numeric bytes
    (baseline + selected) as a Python int for archive cache accounting.
    """

    __slots__ = ('_baseline', '_selected', '_nbytes', '_combination')

    def __setattr__(self,name,value):
        if hasattr(self,name):
            raise AttributeError('prepared responses are immutable')
        object.__setattr__(self,name,value)

    def __init__(self, baseline: np.ndarray, selected: np.ndarray, _create: Any = None,
                 combination='legacy_blas/1') -> None:
        if _create is not _CREATE:
            raise ReconstructionError('prepared responses are created by prepare_response')
        self._baseline = baseline
        self._selected = selected
        self._nbytes = int(baseline.nbytes + selected.nbytes)
        self._combination = _validated_combination(combination)

    @property
    def combination(self) -> str:
        return self._combination

    @property
    def baseline(self) -> np.ndarray:
        return self._baseline

    @property
    def selected(self) -> np.ndarray:
        return self._selected

    @property
    def nbytes(self) -> int:
        return self._nbytes


def prepare_response(responses: Any, *, active_indices=None,
                     combination='legacy_blas/1') -> PreparedResponse:
    """Prepare one response matrix for repeated reconstruction.

    Validates the same response matrix and original one-based active column
    indices as :func:`reconstruct_response` and owns immutable float64 copies
    of the baseline column and the original-order selected matrix
    (``responses[:, active]``, preserving the exact memory layout the
    ordinary reconstruction uses, including sparse and reversed active
    orders).  An empty active order retains the baseline only.  The caller
    may freely mutate or release ``responses`` afterwards; replay is
    unaffected.
    """
    values, active = _validated_response_columns(responses, active_indices)
    baseline = np.frombuffer(values[:,0].tobytes(),dtype=np.float64)
    chosen=values[:,active]
    order='F' if chosen.flags.f_contiguous else 'C'
    selected = np.frombuffer(chosen.tobytes(order=order),dtype=np.float64).reshape(chosen.shape,order=order)
    baseline.setflags(write=False)
    selected.setflags(write=False)
    return PreparedResponse(baseline, selected, _CREATE,combination)


def reconstruct_prepared_response(prepared: Any, packed_coefficients: Any, *, combination=None) -> np.ndarray:
    """Reconstruct one field from a :func:`prepare_response` result.

    Executes precisely the ordinary accumulation expression
    (``baseline + selected @ packed_coefficients``) through the same shared
    helper, returning a read-only float64 array.  Only objects created by
    ``prepare_response`` are admitted; per-request weights are validated for
    shape and finiteness exactly like the ordinary reconstruction.
    """
    if type(prepared) is not PreparedResponse:
        raise ReconstructionError('prepared must come from prepare_response')
    algorithm = _validated_combination(prepared._combination)
    if combination is not None and _validated_combination(combination) != algorithm:
        raise ReconstructionError('prepared response combination mismatch')
    weights = _validated_weights(packed_coefficients, prepared._selected.shape[1])
    return _combine_response(prepared._baseline, prepared._selected, weights,algorithm)
