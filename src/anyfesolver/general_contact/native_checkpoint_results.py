"""Bounded inert reader for an existing native physical-fibre pair capsule.

Digest integrity is data integrity, never mechanical replay or admission.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .transient_results_archive import _pairs

SCHEMA = 'NATIVE_PHYSICAL_FIBRE_WHOLE_CURVE_PAIR_CHECKPOINT_DEV_V1'
PROFILE = 'NATIVE_B3_GE_PHYSICAL_FIBRE_WHOLE_CURVE_PAIR_STATIC_DEV_V1'
MAX_BYTES = 2*1024*1024
MAX_DEPTH = 32
MAX_NODES = 200_000
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_COUNTS = ('search_cells', 'integration_intervals', 'integration_evaluations',
           'integration_cells_checked', 'bounded_omission_cells',
           'unresolved_search_intervals')
_SCALARS = ('potential_j', 'conservative_lower_gap_m', 'minimum_sampled_gap_m',
            'energy_error_j', 'force_error_n', 'tangent_error_n_per_m')
_VECTORS = ('beam_a_force_n', 'beam_b_force_n', 'beam_a_moment_nm', 'beam_b_moment_nm')


class NativeContactCheckpointResultError(ValueError):
    """Unsupported, malformed or resource-exceeding inert native capsule."""


def _fail(message):
    raise NativeContactCheckpointResultError(message)


def _canonical(value):
    # Exact existing native producer convention; no new wire format.
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False)+'\n').encode('ascii')


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        _fail('exact native data keys required')


def _digest(value):
    if type(value) is not str or not _HEX.fullmatch(value):
        _fail('native identity/digest requires lowercase SHA256')


def _number(value, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail('finite native numeric component required')
    if nonnegative and value < 0:
        _fail('negative native energy/error/count')


def _array(value, shape):
    if not shape:
        _number(value)
    elif type(value) is not list or len(value) != shape[0]:
        _fail('native array shape mismatch')
    else:
        for item in value:
            _array(item, shape[1:])


def _validate(doc):
    _keys(doc, ('schema', 'profile', 'identity', 'completed_targets', 'records',
                'checkpoint_sha256'))
    if doc['schema'] != SCHEMA or doc['profile'] != PROFILE:
        _fail('unsupported native checkpoint family/profile')
    _digest(doc['identity'])
    _digest(doc['checkpoint_sha256'])
    count = doc['completed_targets']
    if type(count) is not int or not 0 <= count <= MAX_NODES:
        _fail('native accepted count invalid')
    if type(doc['records']) is not list or len(doc['records']) != count:
        _fail('native accepted record count mismatch')
    unsigned = {key: value for key, value in doc.items() if key != 'checkpoint_sha256'}
    if _sha(unsigned) != doc['checkpoint_sha256']:
        _fail('native unsigned digest mismatch')
    previous = _sha(dict(schema=SCHEMA, identity=doc['identity']))
    for index, row in enumerate(doc['records'], 1):
        _keys(row, ('target', 'previous_sha256', 'mechanical', 'histories', 'contact'))
        if type(row['target']) is not int or row['target'] != index:
            _fail('native target index invalid')
        _digest(row['previous_sha256'])
        if row['previous_sha256'] != previous:
            _fail('native accepted-record chain mismatch')
        mechanical = row['mechanical']
        _keys(mechanical, ('positions', 'position_low', 'nodal_frames',
                           'cell_rotations', 'resultants'))
        for key, shape in (('positions', (6, 3)), ('position_low', (6, 3)),
                           ('nodal_frames', (6, 3, 3)),
                           ('cell_rotations', (2, 2, 3, 3)), ('resultants', (2, 18))):
            _array(mechanical[key], shape)
        histories = row['histories']
        if type(histories) is not list or len(histories) != 2:
            _fail('native two-body history shape')
        for history in histories:
            _keys(history, ('cell_identity', 'stations'))
            _digest(history['cell_identity'])
            stations = history['stations']
            if type(stations) is not list or not 1 <= len(stations) <= 48:
                _fail('native station extent invalid')
            width = None
            for station in stations:
                _keys(station, ('section_identity', 'rows'))
                _digest(station['section_identity'])
                rows = station['rows']
                if type(rows) is not list or not 1 <= len(rows) <= 256:
                    _fail('native fibre row extent invalid')
                if width is not None and len(rows) != width:
                    _fail('native station fibre extent mismatch')
                width = len(rows)
                _array(rows, (width, 4))
        contact = row['contact']
        _keys(contact, _COUNTS+_SCALARS+_VECTORS)
        for key in _COUNTS:
            if type(contact[key]) is not int or contact[key] < 0:
                _fail('native contact count requires nonnegative integer')
        for key in _SCALARS:
            _number(contact[key], key not in ('conservative_lower_gap_m', 'minimum_sampled_gap_m'))
        for key in _VECTORS:
            _array(contact[key], (3,))
        previous = _sha(row)


def _read(raw, max_bytes):
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
        _fail('native reader max_bytes must be within the hard byte cap')
    if type(raw) is not bytes or not 0 < len(raw) <= max_bytes:
        _fail('native bytes input exceeds limit or has wrong type')
    depth = nodes = 0
    quoted = escaped = False
    for byte in raw:
        if byte > 127:
            _fail('native capsule must be ASCII')
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            nodes += 1
        elif byte in (93, 125):
            depth -= 1
        elif byte == 44:
            nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_NODES:
            _fail('native JSON depth/node limit before parsing')
    def floating(text):
        value = float(text)
        if not math.isfinite(value):
            _fail('nonfinite native number')
        return value
    try:
        doc = json.loads(raw, object_pairs_hook=_pairs,
            parse_constant=lambda value: _fail('nonstandard native numeric constant'),
            parse_float=floating,
            parse_int=lambda text: int(text) if len(text) <= 21 else _fail('oversized native integer'))
        _validate(doc)
        if _canonical(doc) != raw:
            _fail('noncanonical native ASCII JSON')
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        if type(exc) is NativeContactCheckpointResultError:
            raise
        raise NativeContactCheckpointResultError('invalid native checkpoint data') from exc
    return doc


@dataclass(frozen=True, slots=True, init=False)
class NativeContactCheckpointResults:
    """Immutable original bytes; every data view is independently detached."""
    _raw: bytes

    def __init__(self, raw, *, max_bytes=MAX_BYTES):
        _read(raw, max_bytes)
        object.__setattr__(self, '_raw', raw)

    def to_bytes(self):
        return self._raw

    def to_dict(self):
        return json.loads(self._raw)

    @property
    def identity(self):
        return self.to_dict()['identity']

    @property
    def completed_targets(self):
        return self.to_dict()['completed_targets']


def read_native_contact_checkpoint_results(raw, *, max_bytes=MAX_BYTES, expected_identity=None):
    """Read DATA only: no native construction, callbacks, replay or restart."""
    if expected_identity is not None:
        _digest(expected_identity)
    result = NativeContactCheckpointResults(raw, max_bytes=max_bytes)
    if expected_identity is not None and result.identity != expected_identity:
        _fail('native expected identity mismatch')
    return result
