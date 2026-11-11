"""Closed research fixture expansion; standard library only, no model imports."""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = 'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json'
FIXTURE_SHA = 'd5fecc80c3203ac7d191b4031d64bf35c56d2900066cb01fc1dfa4276d24c006'
CONTRACT = 'docs/reference_cases/ge_beam3_g3c_mixed_owner_contract_v1.json'
CONTRACT_SHA = '471d05d2db6c604c39e30320e279e3a8ccd47186cb549969a596a00c62d967e5'
REVIEW = 'docs/reference_cases/ge_beam3_g3c_mixed_owner_design_review_v1.json'
REVIEW_SHA = 'c7efca597db0739b83eeeda98f55dd714cea1c4a05e35dbfc6a7f269cc8c91e3'
U = ((0,-1,0),(1,0,0),(0,0,1))
SHIFT = (2,-3,1)

def canonical(v):
    return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')

def digest(v):
    return sha256(canonical(v)).hexdigest()

def strict(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 8388608:
        raise ValueError('bounded canonical bytes required')
    def pairs(rows):
        out = {}
        for k,v in rows:
            if k in out: raise ValueError('duplicate key')
            out[k] = v
        return out
    def bad(_): raise ValueError('nonfinite JSON')
    try:
        v=json.loads(raw.decode('ascii'),object_pairs_hook=pairs,parse_constant=bad)
        if canonical(v)!=raw: raise ValueError('noncanonical bytes')
        return v
    except (UnicodeError,RecursionError,OverflowError) as exc:
        raise ValueError('invalid bounded JSON') from exc

def bound(path, expected):
    raw=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
    if sha256(raw).hexdigest()!=expected: raise ValueError('frozen authority changed: '+path)
    return strict(raw)

def mv(a,x): return [sum(a[i][j]*x[j] for j in range(3)) for i in range(3)]
def mm(a,b): return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def expand(fixture_id, variant='BASE', common_motion='NONE'):
    if any(type(v) is not str for v in (fixture_id,variant,common_motion)):
        raise ValueError('exact fixture selectors required')
    c=bound(CONTRACT,CONTRACT_SHA); review=bound(REVIEW,REVIEW_SHA)
    if review['findings'] or review['decision']!='ACCEPTED_G3C_MIXED_OWNER_DESIGN_IMPLEMENTATION_ONLY':
        raise ValueError('unaccepted design')
    f=bound(FIXTURE,FIXTURE_SHA)
    by={g['id']:g for g in f['graphs']}
    if fixture_id not in by or variant not in f['variants'] or common_motion not in ('NONE','CM0','CM1','CM2','CM3'):
        raise ValueError('unregistered graph/variant/motion')
    g=deepcopy(by[fixture_id]); definitions=deepcopy(f['definitions'])
    anchors=[dict(element_id=e['id'],anchor_node=e['nodes'][0 if e['family']=='B2' else 1])
             for e in g['elements'] if e['family'] in ('B2','B3')]
    if variant=='SHUFFLED_INSERTION':
        for key in ('nodes','elements','joints'): g[key].reverse()
    if variant=='RENUMBERED':
        node=lambda n:10000+7*n
        for row in g['nodes']: row[0]=node(row[0])
        for e in g['elements']: e.update(id=20000+5*e['id'],nodes=[node(n) for n in e['nodes']])
        for j in g['joints']: j.update(id=30000+3*j['id'],master=node(j['master']),slave=node(j['slave']))
        for a in anchors: a.update(element_id=20000+5*a['element_id'],anchor_node=node(a['anchor_node']))
        g['fixed_nodes']=[node(n) for n in g['fixed_nodes']]; g['load_node']=node(g['load_node'])
    if variant=='CONNECTIVITY_REVERSED':
        permutations={'NATIVE':[2,1,0],'B2':[1,0],'B3':[2,1,0],'Q4':[0,3,2,1],'S3':[0,2,1]}
        for e in g['elements']: e['nodes']=[e['nodes'][i] for i in permutations[e['family']]]
    if variant=='PROPER_GLOBAL_TRANSFORM':
        for row in g['nodes']: row[1]=[a+b for a,b in zip(mv(U,row[1]),SHIFT)]
        for e in g['elements']: e['orientation']=mv(U,e['orientation'])
        for j in g['joints']:
            for key in ('master_frame','slave_frame'): j[key]=mm(U,j[key])
        definitions['legacy']['orientation']=mv(U,definitions['legacy']['orientation'])
        definitions['shell']['reference_normal']=mv(U,definitions['shell']['reference_normal'])
    g['nodes'].sort(key=lambda r:r[0]); g['elements'].sort(key=lambda r:r['id']); g['joints'].sort(key=lambda r:r['id'])
    g['fixed_nodes'].sort(); anchors.sort(key=lambda r:r['element_id'])
    ids=[row[0] for row in g['nodes']]
    if (len(ids)>32 or len(set(ids))!=len(ids) or len(g['elements'])>8 or
        6*len(ids)+24*sum(e['family']=='NATIVE' for e in g['elements'])>384 or
        6*(len(g['fixed_nodes'])+len(g['joints']))>96): raise ValueError('graph bound')
    for e in g['elements']:
        if any(n not in ids for n in e['nodes']) or len(set(e['nodes']))!=len(e['nodes']):
            raise ValueError('invalid incidence')
    expanded=dict(graph=g,definitions=definitions,local_policies=c['policies'],anchors=anchors)
    definition=dict(fixture_id=fixture_id,variant=variant,common_motion=common_motion,
        fixture_sha256=FIXTURE_SHA,policy_sha256=CONTRACT_SHA,expanded_sha256=digest(expanded))
    return definition,expanded,f['programs']

def command(value, history, common_motion):
    # Copy/strictly classify before any callback or mechanics.
    value=strict(canonical(value))
    if type(value) is not dict: raise ValueError('command object required')
    prep=0 if common_motion=='NONE' else 4
    index=len(history)
    if index>=128: raise ValueError('accepted history bound')
    if index<prep:
        expected=dict(kind='PREPARE_COMMON_MOTION',step=index+1)
        if canonical(value)!=canonical(expected): raise ValueError('preparation sequence')
    else:
        stage=(index-prep)%5
        if set(value)!={'kind','load_factor','force_scale','root_stage'} or value['kind']!='LOAD_STAGE':
            raise ValueError('load command schema')
        if type(value['root_stage']) is not int or value['root_stage']!=stage:
            raise ValueError('root stage sequence')
        for key in ('load_factor','force_scale'):
            if type(value[key]) is not float or not math.isfinite(value[key]): raise ValueError('binary64 command scalar')
        if value['load_factor']!=[0.,.5,1.,.25,0.][stage] or value['force_scale'] not in (.01,1.,10.):
            raise ValueError('registered load values')
        if stage and history[-1]['command']['force_scale']!=value['force_scale']:
            raise ValueError('changed force scale inside program')
    return value
