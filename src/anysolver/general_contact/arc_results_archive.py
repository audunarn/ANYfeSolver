"""Bounded inert arc-length contact result exports; no solver restoration."""
from __future__ import annotations
from dataclasses import dataclass, fields
import hashlib
from types import MappingProxyType
import numpy as np
from . import transient_results_archive as codec
from ._results_archive_common import _vector, _state, _diagnostics, _encode_committed_results
from . import static_results_archive as static

SCHEMA = 'ANYSOLVER_ARC_CONTACT_RESULTS_ARCHIVE_V1'
MAX_BYTES = codec.MAX_BYTES
CONVENTIONS = MappingProxyType({**dict(codec.CONVENTIONS),
    'analysis': 'arc_length_contact',
    'steps': 'accepted_increments_with_producer_indices',
    'snapshots': 'committed_generalized_displacement_and_material_data',
    'load_factor': 'producer_arc_proportional_parameter',
    'arc_radius': 'producer_spherical_metric_radius',
    'peaks': 'producer_peak_load_factor_index_and_accepted_step_flags'})
ArcContactArchiveError = codec.TransientContactArchiveError


def _types():
    from anysolver.arc_length import ArcLengthResult, ArcLengthStep
    from anysolver.nonlinear_static import NonlinearIncrementSnapshot
    _, records = codec._types()
    records = dict(records)
    for cls in (ArcLengthStep, NonlinearIncrementSnapshot):
        records[cls.__module__ + '.' + cls.__qualname__] = cls
    return ArcLengthResult, records


def _roles(cls, data):
    if cls.__module__ != 'anysolver.arc_length':
        return static._roles(cls, data)
    for key,value in data.items():
        if key in ('step_index','iterations','retries'):
            if (type(value) is not int and not isinstance(value,np.integer)) or isinstance(value,(bool,np.bool_)) or value < 0:
                codec._fail('invalid arc record counter')
        elif key == 'is_peak':
            if type(value) not in (bool,np.bool_): codec._fail('invalid arc peak flag')
        elif key == 'support_reactions':
            if type(value) is not dict: codec._fail('invalid support mapping')
            for name,vector in value.items():
                if type(name) is not str or type(vector) not in (tuple,list) or len(vector) != 6:
                    codec._fail('invalid generalized support dimensions')
                for component in vector: codec._number(component)
        else:
            codec._number(value)


def _validate(data):
    producer, _ = _types()
    if type(data) is not dict or set(data) != {f.name for f in fields(producer)}:
        codec._fail('arc producer field inventory mismatch')
    if type(data['status']) is not str or not data['status']:
        codec._fail('invalid arc result status')
    _vector(data['displacements'])
    codec._number(data['load_factor'])
    codec._number(data['peak_load_factor'])
    peak = data['peak_step_index']
    if peak is not None and ((type(peak) is not int and not isinstance(peak,np.integer)) or
            isinstance(peak,(bool,np.bool_)) or peak < 0):
        codec._fail('invalid peak step identity')
    _state(data['element_states'])
    if type(data['info']) is not dict or not any(
            key in data['info'] for key in ('general_contact', 'general_contact_history')):
        codec._fail('ordinary arc contact diagnostics required; no-contact result is unsupported')
    _diagnostics(data['info'])
    if data['restart_checkpoint'] is not None and type(data['restart_checkpoint']) is not dict:
        codec._fail('arc checkpoint must be mapping or None')
    for name, clsname, container in (('steps', 'anysolver.arc_length.ArcLengthStep', list),
                                   ('snapshots', 'anysolver.nonlinear_static.NonlinearIncrementSnapshot', tuple)):
        rows = data[name]
        if type(rows) is not container:
            codec._fail('invalid arc record sequence')
        previous = -1
        for row in rows:
            if type(row) is not dict or row.get('record_type') != clsname:
                codec._fail('invalid arc record role')
            values = row['fields']
            index = values['step_index']
            if type(index) is not int or index < 0 or index <= previous:
                codec._fail('invalid accepted record ordering')
            previous = index
            if name == 'snapshots':
                _vector(values['displacements'], data['displacements'].size)
                _state(values['element_states'])
            else:
                expected = {key:(int(value) if key in ('step_index','iterations','retries') else
                    bool(value) if key=='is_peak' else
                    {str(k):[float(x) for x in v] for k,v in value.items()} if key=='support_reactions' else
                    float(value)) for key,value in values.items()}
                summary = row['summary']
                if type(summary) is not dict or set(summary) != set(expected):
                    codec._fail('arc step summary inventory mismatch')
                for key in expected:
                    if codec._canonical(codec._encode(summary[key],codec._Budget(MAX_BYTES),{})) != codec._canonical(codec._encode(expected[key],codec._Budget(MAX_BYTES),{})):
                        codec._fail('arc step summary differs from exact fields')
                reactions = values['support_reactions']
                if type(reactions) is not dict:
                    codec._fail('invalid support reaction mapping')
                for key, vector in reactions.items():
                    if type(key) is not str or type(vector) not in (list, tuple) or len(vector) != 6:
                        codec._fail('invalid generalized support reaction dimensions')
                    for value in vector: codec._number(value)


def _read(raw, cap):
    return codec._read(raw, cap, schema=SCHEMA, conventions=CONVENTIONS,
                       records=_types()[1], validator=_validate, record_roles=_roles)


@dataclass(frozen=True, slots=True)
class ContactArcArchive:
    """Immutable canonical bytes; each decode returns detached inert data."""
    _raw: bytes
    _max_bytes: int = MAX_BYTES

    def __post_init__(self):
        _read(self._raw, codec._limit(self._max_bytes))

    @property
    def sha256(self):
        return hashlib.sha256(self._raw).hexdigest()

    def to_bytes(self):
        return self._raw

    def to_dict(self):
        return _read(self._raw, self._max_bytes)


def serialize_arc_contact_results(results, *, max_bytes=MAX_BYTES):
    """Export all ten producer fields using the existing exact numeric codec."""
    cap = codec._limit(max_bytes)
    producer, records = _types()
    if type(results) is not producer:
        codec._fail('exact ArcLengthResult required')
    try: data = {f.name: getattr(results, f.name) for f in fields(results)}
    except AttributeError: codec._fail('incomplete ArcLengthResult producer')
    from anysolver.arc_length import ArcLengthStep
    if type(data['steps']) is not list: codec._fail('arc steps must be list')
    for step in data['steps']:
        if type(step) is not ArcLengthStep: codec._fail('exact accepted step required')
        try: step_data = {f.name:getattr(step,f.name) for f in fields(step)}
        except AttributeError: codec._fail('incomplete accepted arc step')
        _roles(type(step), step_data)
    raw = _encode_committed_results(data, cap, records, schema=SCHEMA,
                                    conventions=CONVENTIONS,
                                    snapshot_error='arc snapshots must be tuple')
    _read(raw, cap)
    return raw


def read_arc_contact_results(raw, *, max_bytes=MAX_BYTES):
    return ContactArcArchive(raw, codec._limit(max_bytes))
