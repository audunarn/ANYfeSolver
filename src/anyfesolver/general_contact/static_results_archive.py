"""Bounded inert ordinary-static contact result exports; no solver restoration."""
from __future__ import annotations
from dataclasses import dataclass, fields
import hashlib
from types import MappingProxyType
import numpy as np
from . import transient_results_archive as codec
from .._wire_types import wire_class_name
from ._results_archive_common import _vector, _state, _diagnostics, _encode_committed_results

SCHEMA = 'ANYSOLVER_STATIC_CONTACT_RESULTS_ARCHIVE_V1'
MAX_BYTES = codec.MAX_BYTES
CONVENTIONS = MappingProxyType({**dict(codec.CONVENTIONS),
    'analysis': 'ordinary_nonlinear_static_contact',
    'steps': 'accepted_increments_with_producer_indices',
    'snapshots': 'committed_generalized_displacement_and_material_data',
    'load_factor': 'producer_proportional_or_program_parameter'})
StaticContactArchiveError = codec.TransientContactArchiveError


def _types():
    from anyfesolver.nonlinear_static import (NonlinearStaticResult, NonlinearStaticStep,
                                          NonlinearIncrementSnapshot)
    _, records = codec._types()
    records = dict(records)
    for cls in (NonlinearStaticStep, NonlinearIncrementSnapshot):
        records[wire_class_name(cls)] = cls
    return NonlinearStaticResult, records


def _roles(cls, data):
    if cls.__module__ != 'anyfesolver.nonlinear_static':
        return codec._record_roles(cls, data)
    for key, value in data.items():
        if key in ('step_index', 'iterations', 'deleted_element_count'):
            if type(value) is not int or value < 0: codec._fail('invalid static record counter')
        elif key == 'support_reactions':
            if type(value) is not dict: codec._fail('invalid support mapping')
            for name, vector in value.items():
                if type(name) is not str or type(vector) not in (tuple,list) or len(vector) != 6:
                    codec._fail('invalid support dimensions')
                for component in vector: codec._number(component)
        elif key in ('displacements', 'element_states'):
            continue  # Shape checks belong to the static validator.
        elif key == 'active_stage':
            if value is not None and (type(value) is not str or not value): codec._fail('invalid active stage')
        elif key == 'control_value' and value is None:
            continue
        else:
            codec._number(value)


def _validate(data):
    producer, _ = _types()
    if type(data) is not dict or set(data) != {f.name for f in fields(producer)}:
        codec._fail('static producer field inventory mismatch')
    if type(data['status']) is not str or not data['status']:
        codec._fail('invalid static result status')
    _vector(data['displacements'])
    codec._number(data['load_factor'])
    _state(data['element_states'])
    if type(data['info']) is not dict or not any(
            key in data['info'] for key in ('general_contact', 'general_contact_history')):
        codec._fail('ordinary static contact diagnostics required; no-contact result is unsupported')
    _diagnostics(data['info'])
    if data['restart_checkpoint'] is not None and type(data['restart_checkpoint']) is not dict:
        codec._fail('static checkpoint must be mapping or None')
    for name, clsname, container in (('steps', 'NonlinearStaticStep', list),
                                   ('snapshots', 'NonlinearIncrementSnapshot', tuple)):
        rows = data[name]
        if type(rows) is not container:
            codec._fail('invalid static record sequence')
        previous = -1
        for row in rows:
            if type(row) is not dict or row.get('record_type') != 'anysolver.nonlinear_static.' + clsname:
                codec._fail('invalid static record role')
            values = row['fields']
            index = values['step_index']
            if type(index) is not int or index < 0 or index <= previous:
                codec._fail('invalid accepted record ordering')
            previous = index
            if name == 'snapshots':
                _vector(values['displacements'], data['displacements'].size)
                _state(values['element_states'])
            else:
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
class ContactStaticArchive:
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


def serialize_static_contact_results(results, *, max_bytes=MAX_BYTES):
    """Export all eight producer fields using the existing exact numeric codec."""
    cap = codec._limit(max_bytes)
    producer, records = _types()
    if type(results) is not producer:
        codec._fail('exact NonlinearStaticResult required')
    data = {f.name: getattr(results, f.name) for f in fields(results)}
    from anyfesolver.nonlinear_static import NonlinearStaticStep
    if type(data['steps']) is not list: codec._fail('static steps must be list')
    for step in data['steps']:
        if type(step) is not NonlinearStaticStep: codec._fail('exact accepted step required')
        _roles(type(step), {f.name:getattr(step,f.name) for f in fields(step)})
    raw = _encode_committed_results(data, cap, records, schema=SCHEMA,
                                    conventions=CONVENTIONS,
                                    snapshot_error='static snapshots must be tuple')
    _read(raw, cap)
    return raw


def read_static_contact_results(raw, *, max_bytes=MAX_BYTES):
    return ContactStaticArchive(raw, codec._limit(max_bytes))
