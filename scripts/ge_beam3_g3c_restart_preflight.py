"""Inert private restart preflight. Passing is NOT proof of a numerical state.

The reviewed coordinator must validate its complete source/environment lease
before calling this module. No anyfesolver package or numerical library is loaded.
Returned bytes remain untrusted commitments until genuine owner replay succeeds.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import stat
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'docs/reference_cases/ge_beam3_g3c_history_restart_contract_v1.json'
CONTRACT_SHA = '7afe9d79096338c122280c20f83b6e7c5681ea0a382e32cb15e1e9349b9372e6'
REVIEW = 'docs/reference_cases/ge_beam3_g3c_history_restart_design_review_v1.json'
REVIEW_SHA = '057fbd280d5dc4c00438387860b2e38d25abcccd4d520970606221b2402589fe'
DEFINITION = 'src/anyfesolver/_ge_beam3_g3c_definition.py'
DEFINITION_SHA = '4b46b870fec010e3378df83c374d763f858d8f25820c14f9704dc6f0a656f0c2'
MAP_SHA = '30848dea0530f5a49b44cea184f4ff38d73f7e3449131086d9f2c9f312a51ca8'
HEX = re.compile('[0-9a-f]{64}')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False)+'\n').encode('ascii')


def digest(value):
    return sha256(canonical(value)).hexdigest()


def strict(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 8*1024**2:
        raise ValueError('bounded canonical bytes required')
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def bad(_): raise ValueError('nonfinite JSON')
    try:
        value = json.loads(raw.decode('ascii'), object_pairs_hook=pairs, parse_constant=bad)
        if canonical(value) != raw: raise ValueError('noncanonical JSON')
        return value
    except (RecursionError, UnicodeError, OverflowError) as exc:
        raise ValueError('malformed bounded JSON') from exc


def hash_value(value):
    if type(value) is not str or HEX.fullmatch(value) is None:
        raise ValueError('lowercase SHA-256 required')


def bound(path, expected):
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('reparse authority input')
    if not path.is_file(): raise ValueError('regular authority file required')
    raw = path.read_bytes().replace(b'\r\n', b'\n')
    if sha256(raw).hexdigest() != expected: raise ValueError('changed authority: '+str(path))
    return raw


def authorities():
    contract = strict(bound(ROOT/CONTRACT, CONTRACT_SHA))
    review = strict(bound(ROOT/REVIEW, REVIEW_SHA))
    if review['decision'] != 'ACCEPTED_G3C_HISTORY_RESTART_DESIGN_ONLY' or review['findings']:
        raise ValueError('unaccepted restart design')
    # This fixed hash-bound file is itself standard-library-only. Loading it
    # directly avoids anyfesolver.__init__ and all numerical/package imports.
    raw = bound(ROOT/DEFINITION, DEFINITION_SHA)
    module = ModuleType('_g3c_inert_definition')
    module.__file__ = str(ROOT/DEFINITION)
    # Execute exactly the verified fixed source bytes, not a second file read
    # or a potentially stale .pyc selected by an import loader.
    exec(compile(raw, str(ROOT/DEFINITION), 'exec'), module.__dict__)
    return contract, module


def rotation(value):
    array(value, (3, 3))
    if any(abs(v) > 1.00000000001 for row in value for v in row):
        raise ValueError('proper rotation component bound')
    error = math.sqrt(sum((sum(value[k][i]*value[k][j] for k in range(3))-(1. if i == j else 0.))**2
                          for i in range(3) for j in range(3)))
    determinant = sum(value[0][i]*(value[1][(i+1)%3]*value[2][(i+2)%3]
                      - value[1][(i+2)%3]*value[2][(i+1)%3]) for i in range(3))
    if not math.isfinite(error) or error > 1e-11 or abs(determinant-1.) > 1e-11:
        raise ValueError('proper rotation required')


def array(value, dimensions, zero=False):
    if dimensions:
        if type(value) is not list or len(value) != dimensions[0]: raise ValueError('array shape')
        for item in value: array(item, dimensions[1:], zero)
    elif type(value) is not float or not math.isfinite(value) or (zero and value != 0.):
        raise ValueError('finite binary64 array leaf required')


def layout(value, name, layouts, dimensions):
    fields = layouts[name]
    if type(value) is not dict or set(value) != set(fields): raise ValueError('exact '+name+' schema')
    for key, rule in fields.items():
        item = value[key]
        if rule in layouts:
            layout(item, rule, layouts, dimensions)
        elif rule.startswith('literal:'):
            text = rule[8:]
            literal = {'false': False, 'true': True}.get(text, text)
            if type(item) is not type(literal) or item != literal: raise ValueError('literal '+key)
        elif rule == 'sha256': hash_value(item)
        elif rule == 'sha256|null':
            if item is not None: hash_value(item)
        elif rule == 'empty_list':
            if type(item) is not list or item: raise ValueError('empty history required')
        elif rule == 'positive_int':
            if type(item) is not int or item <= 0: raise ValueError('positive integer required')
        elif rule.startswith('int['):
            low, high = map(int, rule[4:-1].split('..'))
            if type(item) is not int or not low <= item <= high: raise ValueError('bounded integer')
        elif rule == 'f64' or rule == 'f64[0..1e-11]':
            array(item, ())
            if rule != 'f64' and not 0. <= item <= 1e-11: raise ValueError('native internal residual')
        elif rule.startswith(('f64[', 'zero_f64[')):
            sizes = rule[rule.index('[')+1:-1].split(',')
            shape = tuple(int(s) if s.isdigit() else dimensions[s] for s in sizes)
            array(item, shape, rule.startswith('zero_'))
        elif rule.startswith('proper_rotation['):
            s = rule[16:-1]; count = int(s) if s.isdigit() else dimensions[s]
            if type(item) is not list or len(item) != count: raise ValueError('rotation count')
            for matrix in item: rotation(matrix)
        elif rule.startswith('entry['):
            if type(item) is not list or len(item) > 128: raise ValueError('history bound')
            for entry in item: layout(entry, 'entry', layouts, dimensions)
        elif rule.startswith(('native_row[', 'adapter_row[')):
            kind, count = rule[:-1].split('[')
            if type(item) is not list or len(item) != dimensions[count]: raise ValueError('element count')
            for row in item:
                d = dict(dimensions)
                if kind == 'adapter_row':
                    if type(row) is not dict or type(row.get('element_id')) is not int or row['element_id'] not in dimensions['adapter_nodes']:
                        raise ValueError('unregistered adapter')
                    d['6*family_nodes'] = 6*len(dimensions['adapter_nodes'][row['element_id']])
                layout(row, kind, layouts, d)
        elif rule == 'load_command|prepare_command':
            if type(item) is not dict: raise ValueError('command object')
            kind = {'LOAD_STAGE': 'load_command', 'PREPARE_COMMON_MOTION': 'prepare_command'}.get(item.get('kind'))
            if kind is None: raise ValueError('unknown command')
            layout(item, kind, layouts, dimensions)
        elif rule.startswith('enum:'):
            if type(item) is not str or item not in rule[5:].split(','): raise ValueError('enum')
        elif rule in ('registered_graph', 'registered_variant', 'registered_local_policy'):
            if type(item) is not str: raise ValueError('registered string required')
        else: raise ValueError('unimplemented layout rule: '+rule)


@dataclass(frozen=True)
class PreflightCommitments:
    """A syntactically checked packet, NOT an authenticated accepted owner."""
    raw: bytes
    expected_sha256: str
    fixture_id: str
    variant: str
    common_motion: str
    epoch: int


def preflight(raw, expected_sha256, *, expected_runtime_sha256):
    hash_value(expected_sha256); hash_value(expected_runtime_sha256)
    if type(raw) is not bytes or not 0 < len(raw) <= 8*1024**2 or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('external checkpoint digest mismatch')
    packet = strict(raw)
    contract, source = authorities()
    layouts = dict(contract['inherited_layouts']); layouts['checkpoint'] = contract['checkpoint_layout']
    if type(packet) is not dict or set(packet) != set(layouts['checkpoint']): raise ValueError('checkpoint schema')
    definition = packet['definition']
    layout(definition, 'definition', layouts, {})
    expected, expanded, _ = source.expand(**{k:definition[k] for k in ('fixture_id', 'variant', 'common_motion')})
    expected['policy_sha256'] = MAP_SHA
    if canonical(definition) != canonical(expected): raise ValueError('definition authority mismatch')
    if packet['runtime_sha256'] != expected_runtime_sha256: raise ValueError('runtime authority mismatch')
    graph = expanded['graph']; ids = [r[0] for r in graph['nodes']]
    native = [e for e in graph['elements'] if e['family'] == 'NATIVE']
    adapters = [e for e in graph['elements'] if e['family'] != 'NATIVE']
    dims = {'n':len(ids), '6*n':6*len(ids), 'constraint_count':6*(len(graph['fixed_nodes'])+len(graph['joints'])),
            'native_count':len(native), 'adapter_count':len(adapters),
            'adapter_nodes':{e['id']:e['nodes'] for e in adapters}}
    layout(packet, 'checkpoint', layouts, dims)
    state = packet['final_state']; history = packet['history']; epoch = len(history)
    if state['epoch'] != epoch or packet['definition_sha256'] != digest(definition) or state['definition_sha256'] != digest(definition):
        raise ValueError('definition/epoch link')
    if packet['final_sha256'] != digest(state): raise ValueError('final state hash')
    previous = None; origin = packet['initial_sha256']; scale = None
    seen_states = {origin}
    for index, entry in enumerate(history):
        if entry['epoch'] != index+1 or entry['previous_entry_sha256'] != previous or entry['origin_sha256'] != origin:
            raise ValueError('journal linkage')
        if entry['accepted_state_sha256'] in seen_states:
            raise ValueError('cyclic state commitments across epochs')
        seen_states.add(entry['accepted_state_sha256'])
        source.command(entry['command'], history[:index], definition['common_motion'])
        if entry['command']['kind'] == 'LOAD_STAGE':
            current = entry['command']['force_scale']
            if scale is not None and scale != current: raise ValueError('force scale history switch')
            scale = current
        previous = digest(entry); origin = entry['accepted_state_sha256']
    if origin != packet['final_sha256'] or state['previous_sha256'] != (history[-1]['origin_sha256'] if history else None):
        raise ValueError('final journal link')
    if [r['element_id'] for r in state['native_rows']] != [e['id'] for e in native] or [r['element_id'] for r in state['adapter_rows']] != [e['id'] for e in adapters]:
        raise ValueError('element order/identity')
    for row, element in zip(state['native_rows'], native):
        payload = row['payload']; indices = [ids.index(n) for n in element['nodes']]
        if payload['epoch'] != epoch or (payload['previous_state_sha256'] is None) != (epoch == 0): raise ValueError('native epoch/link')
        if digest({k:v for k,v in payload.items() if k != 'state_sha256'}) != payload['state_sha256']: raise ValueError('native hash')
        if canonical(payload['committed_total_u']) != canonical([state['total_u'][6*i+j] for i in indices for j in range(6)]) or canonical(payload['committed_nodal_rotation_matrices']) != canonical([state['rotations'][i] for i in indices]):
            raise ValueError('native graph pose mismatch')
    for row, element in zip(state['adapter_rows'], adapters):
        indices = [ids.index(n) for n in element['nodes']]
        if row['policy'] != expanded['local_policies'][element['family']] or row['epoch'] != epoch:
            raise ValueError('adapter policy/epoch')
        if row['origin_sha256'] != state['previous_sha256'] or (row['previous_sha256'] is None) != (epoch == 0): raise ValueError('adapter origin/link')
        pose = dict(node_ids=element['nodes'], total_translations=[state['total_u'][6*i:6*i+3] for i in indices], rotations=[state['rotations'][i] for i in indices])
        if row['pose_sha256'] != digest(pose): raise ValueError('adapter pose hash')
    return PreflightCommitments(raw, expected_sha256, definition['fixture_id'], definition['variant'], definition['common_motion'], epoch)
