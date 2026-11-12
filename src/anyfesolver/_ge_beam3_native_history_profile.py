"""Explicit private native-chain capacity; no global/context-local limit changes."""
import json
from ._ge_beam3_p5_seeded.codec import _load as legacy_load
from ._ge_beam3_p5_seeded.core import canonical
from ._ge_beam3_p5 import typed_schema

HISTORY8M='GE_BEAM3_NATIVE_GENERALIZED_HISTORY_8M_V2'


def require(profile):
    if profile is not None and (type(profile) is not str or profile!=HISTORY8M):
        raise ValueError('explicit registered native history profile required')


def limit(profile):
    require(profile);return 2*1024**2 if profile is None else 8*1024**2


def binding(profile):
    require(profile);return {} if profile is None else {'history_profile':profile}


def schema(base,profile):
    require(profile);return base if profile is None else base.removesuffix('_V1')+'_HISTORY8M_V2'


def load(raw,profile):
    bound=limit(profile)
    if type(raw) is not bytes or not 0<len(raw)<=bound:raise ValueError('native history payload byte bound')
    text=raw.decode('ascii')
    if profile is None:return legacy_load(text)
    quoted=escaped=False;depth=0
    for byte in raw:
        if quoted:
            if escaped:escaped=False
            elif byte==92:escaped=True
            elif byte==34:quoted=False
        elif byte==34:quoted=True
        elif byte in (91,123):
            depth+=1
            if depth>32:raise ValueError('native history nesting bound')
        elif byte in (93,125):depth-=1
    value=json.loads(text,object_pairs_hook=typed_schema._pairs,parse_constant=typed_schema._constant)
    if canonical(value)!=raw:raise ValueError('canonical native history required')
    return value


def envelope(value,keys,base,profile):
    expected=set(keys)|set(binding(profile))
    if type(value) is not dict or set(value)!=expected or value['schema']!=schema(base,profile):
        raise ValueError('native history schema/profile/key mismatch')
    if profile is not None and value['history_profile']!=profile:
        raise ValueError('native history caller/profile mismatch')
