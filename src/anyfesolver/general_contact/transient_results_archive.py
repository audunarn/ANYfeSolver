"""Bounded, inert snapshots of transient-contact results; never a restart owner."""
from __future__ import annotations

import base64
from dataclasses import dataclass, fields
import hashlib
import json
import math
import struct
from types import MappingProxyType
import numpy as np
from .._wire_types import wire_class_name

SCHEMA = 'ANYSOLVER_TRANSIENT_CONTACT_RESULTS_ARCHIVE_V1'
MAX_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 64
MAX_NODES = 200_000
MAX_ARRAY_ELEMENTS = 2_000_000
CONVENTIONS = MappingProxyType({
    'numeric': 'exact_binary_bits_v1',
    'time': 's', 'gap': 'm', 'energy_and_work': 'J',
    'physical_contact_force': 'N', 'physical_contact_moment': 'N*m',
    'surface_pressure': 'Pa', 'surface_stiffness': 'N/m^3',
    'line_pressure': 'N/m', 'line_stiffness': 'N/m^2',
    'endpoint_force': 'N', 'endpoint_stiffness': 'N/m',
    'structural_kinematics': 'producer_generalized_chart_translations_m_rotations_rad',
    'constraint_reactions': 'raw_generalized_work_conjugate_not_physical_moments',
    'record_identity': 'inert_producer_type_and_exact_fields',
    'archive_authority': 'data_only_confers_no_admission_or_qualification',
})
_GAPS = frozenset({'minimum_gap', 'sampled_overlap_gap', 'minimum_physical_gap',
    'swept_minimum_gaps', 'pair_swept_minimum_gaps', 'normalized_weighted_gap_m'})


class TransientContactArchiveError(ValueError):
    """A result snapshot violates the closed archive contract."""


def _fail(message):
    raise TransientContactArchiveError(message)


def _limit(value):
    if type(value) is not int or not 1 <= value <= MAX_BYTES:
        _fail('max_bytes must be a positive integer at most16MiB')
    return value


def _types():
    # Fixed imports of producer classes, never names supplied by an archive.
    from . import transient as t
    from .deformable_beam_s3_static import DeformableBeamS3Results
    from .beam_s3_rotational_transient import RotationalBeamS3Results
    records = (RotationalBeamS3Results, DeformableBeamS3Results, t.ContactResults, t._PairResults, t._PatchResults, t._BeamPlaneResults,
               t.MultipleBeamRigidResults, t.MultipleShellPlaneResults, t._RadialResults,
               t.FreeRigidBodyState)
    return t.TransientContactResults, {wire_class_name(c): c for c in records}


class _Budget:
    def __init__(self, cap):
        self.cap, self.bytes, self.nodes, self.elements = cap, 4096, 0, 0

    def visit(self, depth, byte_count=128, elements=0):
        self.nodes += 1
        self.bytes += byte_count
        self.elements += elements
        if depth > MAX_DEPTH or self.nodes > MAX_NODES or self.elements > MAX_ARRAY_ELEMENTS:
            _fail('archive depth/node/array-element bound exceeded')
        if self.bytes > self.cap:
            _fail('archive byte limit exceeded before allocation')


def _dtype(text):
    # Closed primitive numerical dtypes; no objects, records or executable data.
    if type(text) is not str or text not in {
            '|b1','|i1','|u1','<i2','>i2','<u2','>u2','<i4','>i4','<u4','>u4',
            '<i8','>i8','<u8','>u8','<f4','>f4','<f8','>f8'}:
        _fail('unsupported numeric dtype')
    return np.dtype(text)


def _finite(value, gap):
    if np.any(np.isnan(value)) or np.any(np.isneginf(value)):
        _fail('NaN or negative infinity is not archive data')
    if not gap and np.any(np.isposinf(value)):
        _fail('positive infinity is allowed only for gap data')


def _encode(value, budget, records, depth=0, gap=False):
    budget.visit(depth)
    if value is None or type(value) in (bool, int, str):
        if type(value) is int and not -(2**63) <= value < 2**64:
            _fail('integer outside bounded archive range')
        if type(value) is str:
            # ensure_ascii uses two six-byte escapes for astral codepoints.
            budget.visit(depth, 12*len(value)+2)
        return value
    if type(value) is float:
        _finite(value, gap)
        return {'tag':'positive_gap_infinity'} if math.isinf(value) else {
            'tag':'binary64','bits':struct.pack('>d', value).hex()}
    if isinstance(value, np.generic):
        dtype = _dtype(value.dtype.str)
        _finite(value, gap)
        return {'tag':'numpy_scalar','dtype':dtype.str,'bits':value.tobytes().hex(),
                'nonfinite':'positive_gap_infinity' if np.isposinf(value) else None}
    if type(value) is np.ndarray:
        dtype = _dtype(value.dtype.str)
        if value.ndim > 8:
            _fail('array dimension bound exceeded')
        budget.visit(depth, 4*((value.nbytes+2)//3)+128, value.size)
        _finite(value, gap)
        return {'tag':'array','dtype':dtype.str,'shape':list(value.shape),
                'data':base64.b64encode(value.tobytes(order='C')).decode('ascii'),
                'nonfinite':'positive_gap_infinity' if np.any(np.isposinf(value)) else None}
    if type(value) in (tuple, list):
        budget.visit(depth, 2*len(value))
        return {'tag':'tuple' if type(value) is tuple else 'list',
                'items':[_encode(x,budget,records,depth+1,gap) for x in value]}
    if type(value) in (dict, MappingProxyType):
        budget.visit(depth, 2*len(value))
        items=[]
        for key,item in value.items():
            if type(key) not in (str,int):
                _fail('mapping keys must be strings or integers')
            items.append([_encode(key,budget,records,depth+1),
                _encode(item,budget,records,depth+1,gap or key in _GAPS)])
        return {'tag':'mapping','items':items}
    name=wire_class_name(type(value))
    if name in records and type(value) is records[name]:
        data={f.name:getattr(value,f.name) for f in fields(value)}
        encoded_fields=_encode(data,budget,records,depth+1,gap)
        # A known producer's summary retains computed physical origins/conventions.
        summary=value.to_dict() if name.rsplit('.',1)[-1]!='FreeRigidBodyState' else None
        return {'tag':'record','record_type':name,
                'fields':encoded_fields,
                'summary':_encode(summary,budget,records,depth+1,gap)}
    _fail('unsupported archive value type: '+name)


def _keys(value, names):
    if type(value) is not dict or set(value)!=set(names):
        _fail('unknown or missing archive node fields')


def _number(value, *, gap=False):
    if type(value) not in (int,float,np.float32,np.float64) and not isinstance(value,np.integer):
        _fail('record scalar requires numeric value')
    if type(value) in (bool,np.bool_) or (not math.isfinite(value) and not (gap and value==float('inf'))):
        _fail('record scalar requires finite numeric value or explicit positive gap infinity')


def _record_roles(cls,data):
    for descriptor in fields(cls):
        name=descriptor.name;value=data[name];annotation=str(descriptor.type)
        if value is None:
            if 'None' in annotation:continue
            _fail('unexpected None in producer record')
        if annotation in ('float',"<class 'float'>",'float | None'):
            _number(value,gap=name in _GAPS)
        elif annotation in ('int',"<class 'int'>",'int | None'):
            if (type(value) is not int and not isinstance(value,np.integer)) or isinstance(value,np.bool_) or value<0:
                _fail('record counter/rank requires nonnegative integer')
        elif annotation in ('bool',"<class 'bool'>"):
            if type(value) not in (bool,np.bool_):_fail('record boolean requires boolean value')
        elif annotation in ('str',"<class 'str'>",'str | None'):
            if type(value) is not str or not value:_fail('record text requires nonempty string')
        elif name!='rotation' and 'tuple[float' in annotation:
            if type(value) not in (tuple,list,np.ndarray):_fail('record numerical sequence required')
            for x in value:_number(x,gap=name in _GAPS)
        elif 'tuple[int' in annotation:
            if type(value) not in (tuple,list) or any(type(x) is not int and not isinstance(x,np.integer) or
                    type(x) in (bool,np.bool_) for x in value):_fail('record integer sequence required')
        elif name=='rotation':
            for row in value:
                for x in row:_number(x)
        elif name=='pairs':
            if type(value) is not tuple:_fail('record pairs must be tuple')
            for pair in value:
                if type(pair) is not tuple or len(pair)!=2 or type(pair[0]) is not str or not pair[0]:
                    _fail('invalid nested pair identity')
                if type(pair[1]) is not dict or pair[1].get('record_type') in (
                        None,'anysolver.general_contact.free_rigid.FreeRigidBodyState'):
                    _fail('invalid nested contact record')
        elif name=='body_names':
            if type(value) is not tuple or any(type(x) is not str or not x for x in value):
                _fail('invalid rigid body name sequence')
            if len(value) not in (0,len(data['pairs'])):_fail('rigid body/pair count mismatch')
        else:_fail('unrecognized producer record role')


def _decode(value, budget, records, depth=0, gap=False, record_roles=None):
    budget.visit(depth)
    if value is None or type(value) in (bool,int,str):
        if type(value) is int and not -(2**63) <= value < 2**64:
            _fail('integer outside bounded archive range')
        return value
    if type(value) is not dict or type(value.get('tag')) is not str:
        _fail('unsupported untagged archive value')
    tag=value['tag']
    if tag=='positive_gap_infinity':
        _keys(value,('tag',))
        if not gap:_fail('infinity outside gap data')
        return float('inf')
    if tag=='binary64':
        _keys(value,('tag','bits'))
        bits=value['bits']
        if type(bits) is not str or len(bits)!=16 or any(c not in '0123456789abcdef' for c in bits):
            _fail('invalid binary64 bits')
        result=struct.unpack('>d',bytes.fromhex(bits))[0]
        if not math.isfinite(result):_fail('nonfinite binary64 must use explicit gap infinity')
        return result
    if tag in ('array','numpy_scalar'):
        _keys(value,('tag','dtype','shape','data','nonfinite') if tag=='array' else ('tag','dtype','bits','nonfinite'))
        dtype=_dtype(value['dtype'])
        shape=value['shape'] if tag=='array' else []
        if type(shape) is not list or len(shape)>8 or any(type(x) is not int or x<0 or
                x>MAX_ARRAY_ELEMENTS for x in shape):
            _fail('invalid array dimensions')
        count=math.prod(shape)
        size=count*dtype.itemsize
        budget.visit(depth, size, count)
        text=value['data'] if tag=='array' else value['bits']
        if type(text) is not str or len(text)!=(4*((size+2)//3) if tag=='array' else 2*size):
            _fail('array byte dimensions mismatch')
        try:
            raw=base64.b64decode(text,validate=True) if tag=='array' else bytes.fromhex(text)
        except (ValueError,TypeError) as exc:
            raise TransientContactArchiveError('invalid numeric bytes') from exc
        if len(raw)!=size:_fail('decoded array size mismatch')
        result=np.frombuffer(raw,dtype=dtype).reshape(shape)
        _finite(result,gap)
        expected='positive_gap_infinity' if np.any(np.isposinf(result)) else None
        if value['nonfinite']!=expected:_fail('numeric nonfinite marker mismatch')
        return result.copy() if tag=='array' else result[()]
    if tag in ('tuple','list'):
        _keys(value,('tag','items'))
        if type(value['items']) is not list:_fail('invalid sequence')
        result=[_decode(x,budget,records,depth+1,gap,record_roles) for x in value['items']]
        return tuple(result) if tag=='tuple' else result
    if tag=='mapping':
        _keys(value,('tag','items'))
        if type(value['items']) is not list:_fail('invalid mapping')
        result={}
        for row in value['items']:
            if type(row) is not list or len(row)!=2:_fail('invalid mapping entry')
            key=_decode(row[0],budget,records,depth+1,False,record_roles)
            if type(key) not in (str,int) or key in result:_fail('invalid or duplicate mapping key')
            result[key]=_decode(row[1],budget,records,depth+1,gap or key in _GAPS,record_roles)
        return result
    if tag=='record':
        _keys(value,('tag','record_type','fields','summary'))
        name=value['record_type']
        if type(name) is not str or name not in records:_fail('unknown producer record type')
        data=_decode(value['fields'],budget,records,depth+1,gap,record_roles)
        if type(data) is not dict or set(data)!={f.name for f in fields(records[name])}:
            _fail('producer record field inventory mismatch')
        (record_roles or _record_roles)(records[name],data)
        for field,item in data.items():
            if field in ('shell_force','rigid_force','beam_force','integration_force','target_force',
                    'shell_moment','shell_nodal_work_moment','shell_affine_moment_discrepancy','rigid_moment','beam_moment','integration_moment','target_moment',
                    'centre_of_mass_m','velocity_m_s','acceleration_m_s2','angular_velocity_rad_s',
                    'contact_force_n','linear_momentum_kg_m_s','angular_momentum_about_origin_kg_m2_s'):
                if type(item) not in (tuple,list) or len(item)!=3:
                    _fail('invalid physical vector dimensions')
                if any(type(x) not in (float,int,np.float32,np.float64,np.int32,np.int64) or
                       not math.isfinite(x) for x in item):_fail('physical vectors require finite numeric components')
            if field=='rotation':
                if type(item) not in (tuple,list) or len(item)!=3 or any(
                        type(row) not in (tuple,list) or len(row)!=3 for row in item):
                    _fail('invalid rigid rotation dimensions')
                if any(type(x) not in (float,int,np.float32,np.float64,np.int32,np.int64) or
                       not math.isfinite(x) for row in item for x in row):
                    _fail('rotation requires finite numeric components')
        if 'source_pressure_node_ids' in data:
            n=len(data['source_pressure_node_ids'])
            for field in ('normalized_weighted_gap_m','numerical_pressure_pa','reference_support_m2',
                          'normal_multiplier_estimate_pa'):
                if field in data and len(data[field]) not in (0,n):_fail('invalid pressure/support dimensions')
        return {'record_type':name,'fields':data,
                'summary':_decode(value['summary'],budget,records,depth+1,gap,record_roles)}
    _fail('unknown archive tag')


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')


def _validate(data):
    producer,_=_types()
    if type(data) is not dict or set(data)!={f.name for f in fields(producer)}:
        _fail('transient producer field inventory mismatch')
    if type(data['profile']) is not str or not data['profile']:_fail('invalid producer profile')
    times=data['times']
    if type(times) is not np.ndarray or times.ndim!=1 or times.size<1 or not np.all(np.isfinite(times)):
        _fail('invalid saved times')
    n=times.size
    if times[0]<0 or np.any(times[1:]<=times[:-1]):_fail('saved times must be strictly increasing')
    shape=data['displacements'].shape if type(data['displacements']) is np.ndarray else ()
    if len(shape)!=2 or shape[0]!=n or shape[1]<1:_fail('invalid kinematic dimensions')
    for name in ('displacements','velocities','accelerations'):
        array=data[name]
        if type(array) is not np.ndarray or array.shape!=shape or not np.all(np.isfinite(array)):
            _fail('inconsistent kinematic arrays')
    for name in ('swept_minimum_gaps','equilibrium_residuals','external_nodal_work_j'):
        if type(data[name]) is not np.ndarray or data[name].shape!=(n,):_fail('inconsistent saved scalar arrays')
    _,records=_types()
    contact_names=set(records)-{'anysolver.general_contact.free_rigid.FreeRigidBodyState'}
    if type(data['contact']) is not tuple or len(data['contact'])!=n or any(
            type(x) is not dict or x.get('record_type') not in contact_names for x in data['contact']):
        _fail('inconsistent contact history')
    if type(data['newton_iterations']) is not tuple or any(
            type(x) is not int or x<0 for x in data['newton_iterations']):_fail('invalid iteration history')
    for name in ('cutbacks','event_endpoint_evaluations','plane_event_stage_evaluations'):
        if type(data[name]) is not int or data[name]<0:_fail('invalid event counter')
    reactions=data['constraint_reactions']
    if type(reactions) is not np.ndarray or reactions.shape not in ((0,0),shape):
        _fail('invalid generalized reaction dimensions')
    for name in ('prescribed_tool_work_j','prescribed_tool_work_error_j','free_rigid_bodies',
                 'material_state_history','pair_swept_minimum_gaps'):
        if type(data[name]) is not tuple or len(data[name]) not in (0,n):_fail('invalid optional saved history')
        if any(type(row) is not dict for row in data[name]):_fail('optional history rows must be mappings')
        if name in ('prescribed_tool_work_j','prescribed_tool_work_error_j','free_rigid_bodies','pair_swept_minimum_gaps'):
            if data[name] and any(set(row)!=set(data[name][0]) for row in data[name]):
                _fail('optional history identities changed')
            if any(type(key) is not str for row in data[name] for key in row):_fail('invalid optional history identity')
        if name in ('prescribed_tool_work_j','prescribed_tool_work_error_j'):
            if any(type(x) not in (float,np.float64) or not math.isfinite(x) or
                   (name=='prescribed_tool_work_error_j' and x<0) for row in data[name] for x in row.values()):
                _fail('invalid tool work/error values')
        if name=='free_rigid_bodies' and any(type(x) is not dict or
                x.get('record_type')!='anysolver.general_contact.free_rigid.FreeRigidBodyState'
                for row in data[name] for x in row.values()):_fail('invalid free-body history')
    for name in ('represented_path_transition_times','localized_contact_event_times'):
        if type(data[name]) is not tuple or any(type(x) is not float or not math.isfinite(x) or x<0
                                               for x in data[name]):_fail('invalid event time history')
    if type(data['cutback_diagnostics']) is not tuple or any(type(x) is not dict for x in data['cutback_diagnostics']):
        _fail('invalid cutback diagnostics')
    if type(data['element_states']) is not dict:_fail('invalid material state mapping')
    if data['restart_checkpoint'] is not None and type(data['restart_checkpoint']) is not dict:
        _fail('restart payload must be a plain mapping or None')


def _pairs(rows):
    result={}
    for key,value in rows:
        if key in result:_fail('duplicate JSON object key')
        result[key]=value
    return result


def _read(raw,cap, *, schema=SCHEMA, conventions=CONVENTIONS, records=None, validator=None, record_roles=None):
    if type(raw) is not bytes:_fail('archive input must be bytes')
    if len(raw)>cap:_fail('archive input byte limit exceeded before parsing')
    # Bound parser nesting and possible container allocation before json.loads.
    depth=0;quoted=False;escaped=False;tokens=0
    for byte in raw:
        if quoted:
            if escaped:escaped=False
            elif byte==92:escaped=True
            elif byte==34:quoted=False
        elif byte==34:quoted=True
        elif byte in (123,91):
            depth+=1;tokens+=1
            if depth>MAX_DEPTH or tokens>MAX_NODES:_fail('JSON structure bound exceeded before parsing')
        elif byte in (125,93):depth-=1
        elif byte==44:
            tokens+=1
            if tokens>MAX_NODES:_fail('JSON node bound exceeded before parsing')
    try:
        doc=json.loads(raw,object_pairs_hook=_pairs,
            parse_constant=lambda x:_fail('nonstandard JSON numeric constant'),
            parse_float=lambda x:_fail('untagged floating point'),
            parse_int=lambda x:int(x) if len(x)<=21 else _fail('oversized JSON integer'))
    except (ValueError,UnicodeError,RecursionError) as exc:
        raise TransientContactArchiveError('invalid strict archive JSON') from exc
    _keys(doc,('schema','version','conventions','payload','sha256'))
    if doc['schema']!=schema or type(doc['version']) is not int or doc['version']!=1:
        _fail('unknown archive schema/version')
    if doc['conventions']!=conventions:_fail('stale archive conventions')
    expected=dict(doc);digest=expected.pop('sha256')
    if type(digest) is not str or digest!=hashlib.sha256(_canonical(expected)).hexdigest():
        _fail('archive integrity mismatch')
    if _canonical(doc)!=raw:_fail('archive JSON must use canonical encoding')
    if records is None:_,records=_types()
    data=_decode(doc['payload'],_Budget(cap),records,record_roles=record_roles)
    (validator or _validate)(data)
    return data


@dataclass(frozen=True,slots=True)
class ContactTransientArchive:
    """Immutable byte snapshot. Every decoded view owns its mutable data."""
    _raw: bytes
    _max_bytes: int = MAX_BYTES

    def __post_init__(self):
        _read(self._raw,_limit(self._max_bytes))

    @property
    def sha256(self):
        return hashlib.sha256(self._raw).hexdigest()

    def to_bytes(self):
        return self._raw

    def to_dict(self):
        """Return independent arrays/mappings and inert producer records."""
        return _read(self._raw,self._max_bytes)


def _prepare_material_data(data, cap):
    # Explicit solver state protocol only. Each element is materialized once;
    # known state buffers are preflighted before the owned-copy operation.
    from anyfesolver.nonlinear_state import NonlinearStateStore
    material_total=0;material_nodes=0
    closed_nodes=0
    def material_mapping(mapping):
        if type(mapping) is not NonlinearStateStore:return mapping
        if mapping.has_active_trial:_fail('active material trial cannot be archived')
        # A store with arbitrary fallback/sidecar objects cannot safely call
        # its deepcopy protocol. Validate copied data without invoking it.
        def closed(value,depth=0):
            nonlocal closed_nodes
            closed_nodes+=1
            if closed_nodes>MAX_NODES:_fail('material value node limit')
            if depth>MAX_DEPTH:_fail('material value depth limit')
            if value is None or type(value) in (str,int,bool,float):return
            if type(value) is np.ndarray or isinstance(value,np.generic):
                _dtype(value.dtype.str);return
            if type(value) in (dict,MappingProxyType):
                for key,item in value.items():
                    if type(key) not in (str,int):_fail('unsupported material key')
                    closed(item,depth+1)
                return
            if type(value) in (tuple,list):
                for item in value:closed(item,depth+1)
                return
            _fail('unsupported material object before owned-copy operation')
        closed(mapping._fallback_committed)
        closed(mapping._fallback_sidecars)
        for batch in mapping._batches.values():
            for attr in ('_sidecars','_fallback_committed'):
                if hasattr(batch,attr):closed(getattr(batch,attr))
        path=set()
        def buffers(value,depth=0):
            nonlocal material_total,material_nodes
            material_nodes+=1
            if material_nodes>MAX_NODES:_fail('material buffer node limit')
            if id(value) in path:_fail('cyclic material buffer graph')
            if depth>MAX_DEPTH:_fail('material buffer depth limit')
            path.add(id(value))
            try:
                material_total+=128
                if type(value) is np.ndarray:material_total+=2*value.nbytes
                elif type(value) in (dict,MappingProxyType):
                    for v in value.values():buffers(v,depth+1)
                elif type(value) in (list,tuple):
                    for v in value:buffers(v,depth+1)
                elif type(value).__module__=='anyfesolver.nonlinear_state' and hasattr(value,'__dict__'):
                    buffers(vars(value),depth+1)
                if material_total>cap//2:_fail('material buffer byte limit before materialization')
            finally:path.remove(id(value))
        buffers(mapping)
        return mapping.materialize_owned()
    data['element_states']=material_mapping(data['element_states'])
    if type(data.get('material_state_history')) is tuple:
        data['material_state_history']=tuple(material_mapping(x) for x in data['material_state_history'])
    return data


def serialize_transient_contact_results(results, *, max_bytes=MAX_BYTES):
    """Snapshot every exact producer field into bounded strict JSON bytes."""
    cap=_limit(max_bytes)
    producer,records=_types()
    if type(results) is not producer:_fail('exact TransientContactResults required')
    data={f.name:getattr(results,f.name) for f in fields(results)}
    data=_prepare_material_data(data,cap)
    payload=_encode(data,_Budget(cap),records)
    document=dict(schema=SCHEMA,version=1,conventions=dict(CONVENTIONS),payload=payload)
    document['sha256']=hashlib.sha256(_canonical(document)).hexdigest()
    raw=_canonical(document)
    if len(raw)>cap:_fail('archive encoded byte limit exceeded')
    _read(raw,cap)
    return raw


def read_transient_contact_results(raw, *, max_bytes=MAX_BYTES):
    """Validate archive bytes; do not reconstruct solver/material objects."""
    return ContactTransientArchive(raw,_limit(max_bytes))
