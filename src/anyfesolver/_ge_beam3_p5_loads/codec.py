"""Load-parameter native successor to the frozen centered P5 package.

Private development candidate: no public selector or qualification authority.
Unchanged physical operators are imported from the preserved P5 package.
"""

import json

from anyfesolver._ge_beam3_p5_loads.core import canonical, NativeMaterialError
from anyfesolver._ge_beam3_p5 import typed_schema as schema_source


SCHEMA='GE_BEAM3_P5_LOAD_TYPED_STATE_JSON_V4'
MAX_BYTES=2*1024*1024


def _keys(value, keys):
    if type(value) is not dict or set(value)!=set(keys):
        raise NativeMaterialError('exact typed native checkpoint keys required')


def _load(raw):
    if type(raw) is not str:
        raise NativeMaterialError('native checkpoint payload must be canonical ASCII text')
    try: data=raw.encode('ascii')
    except UnicodeEncodeError as exc: raise NativeMaterialError('native checkpoint must be ASCII') from exc
    if not 0<len(data)<=MAX_BYTES: raise NativeMaterialError('native checkpoint byte limit exceeded')
    quoted=escaped=False;depth=0
    for byte in data:
        if quoted:
            if escaped: escaped=False
            elif byte==92: escaped=True
            elif byte==34: quoted=False
        elif byte==34: quoted=True
        elif byte in (91, 123):
            depth+=1
            if depth>32: raise NativeMaterialError('native checkpoint nesting limit exceeded')
        elif byte in (93, 125): depth-=1
    value=json.loads(raw, object_pairs_hook=schema_source._pairs, parse_constant=schema_source._constant)
    if canonical(value)!=data: raise NativeMaterialError('noncanonical native checkpoint')
    return value


def encode(state, *, order):
    raw=canonical(state).decode('ascii')
    envelope=dict(schema=SCHEMA, payload=raw)
    decode(envelope, order=order)
    return envelope


def decode(envelope, *, order):
    _keys(envelope, ('schema', 'payload'))
    if envelope['schema']!=SCHEMA or type(order) is not int or order not in (4, 8, 24):
        raise NativeMaterialError('native checkpoint schema or quadrature mismatch')
    raw=_load(envelope['payload'])
    outer_keys=('driver_schema', 'driver_sha256', 'committed_total_u', 'committed_nodal_rotation_matrices',
                'material_state', 'state_sha256')
    _keys(raw, outer_keys)
    inner=raw['material_state']
    _keys(inner, ('schema', 'model_sha256', 'epoch', 'committed_total_u', 'committed_positions',
                  'coordinate_policy', 'committed_position_low', 'load_parameter',
                  'committed_nodal_rotation_matrices', 'origins', 'histories', 'response', 'state_sha256'))
    old_state, old_trial=schema_source._schemas(order)
    histories=old_state[1]['histories'];response=old_trial[1]['response']
    decode_value=schema_source._decode
    array=lambda *shape: ('array', shape)
    made=dict(inner)
    for key in ('schema', 'model_sha256', 'state_sha256', 'coordinate_policy'):
        if type(inner[key]) is not str: raise NativeMaterialError('exact native identity string required')
    made['epoch']=decode_value(inner['epoch'], ('integer', 2**31-1))
    made['load_parameter']=decode_value(inner['load_parameter'], float)
    for key, shape in (('committed_total_u', (18,)), ('committed_positions', (3, 3)),
                       ('committed_position_low', (3, 3)),
                       ('committed_nodal_rotation_matrices', (3, 3, 3))):
        made[key]=decode_value(inner[key], array(*shape))
    for key in ('origins', 'histories'): made[key]=decode_value(inner[key], histories)
    made['response']=decode_value(inner['response'], response)
    result=dict(raw);result['material_state']=made
    for key in ('driver_schema', 'driver_sha256', 'state_sha256'):
        if type(raw[key]) is not str: raise NativeMaterialError('exact driver identity string required')
    result['committed_total_u']=decode_value(raw['committed_total_u'], array(18))
    result['committed_nodal_rotation_matrices']=decode_value(raw['committed_nodal_rotation_matrices'], array(3, 3, 3))
    if canonical(result)!=envelope['payload'].encode('ascii'):
        raise NativeMaterialError('typed native state did not round-trip exactly')
    return result
