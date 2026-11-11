"""Experimental absolute native distributed-load stages, with an implicit zero.

Programme positions are not load multipliers. Geometry is compiled once; each
stage interpolates spatial force/couple densities per reference arclength.
"""
from dataclasses import dataclass, field, replace
from fractions import Fraction
from hashlib import sha256
from math import floor, isfinite

import numpy as np

from ._stiffened_shell_ge_distributed_load import GEBeamLineAction, GEBeamDistributedLoad
from ._stiffened_shell_ge_checkpoint_binary import MAX_CHECKPOINT_BYTES
from ._stiffened_shell_staged_load_experimental import (
    _canonical, _identity, _json, _progress, StagedLoadResolutionError,
)
from ._stiffened_shell_native_gravity import _owned

POLICY = 'STIFFENED_SHELL_GE_DISTRIBUTED_PROGRAMME_V1_DEV'
MAGIC = b'ANYSGE_DISTR_STAGE1\0'
MAX_METADATA_BYTES = 1024*1024
MAX_VALUES = 1000000


@dataclass(frozen=True)
class GEBeamDistributedProgramme:
    model_identity: str
    endpoints: tuple[tuple[GEBeamLineAction, ...], ...]
    identity: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'model_identity', _identity(self.model_identity))
        if type(self.endpoints) is not tuple or not 1 <= len(self.endpoints) <= 64:
            raise ValueError('bounded nonempty distributed stage tuple required')
        if any(type(row) is not tuple for row in self.endpoints):
            raise ValueError('tuple of actions required at each distributed stage')
        if not 0 < sum(map(len, self.endpoints)) <= 16384:
            raise ValueError('bounded nonzero distributed action inventory required')
        rows = []
        for row in self.endpoints:
            if any(type(a) is not GEBeamLineAction for a in row):
                raise ValueError('typed native line actions required')
            if len({a.element_id for a in row}) != len(row):
                raise ValueError('duplicate distributed element within stage')
            rows.append(tuple(sorted(row, key=lambda a: a.element_id)))
        object.__setattr__(self, 'endpoints', tuple(rows))
        object.__setattr__(self, 'identity', sha256(_canonical(self._core())).hexdigest())
        if len(self.serialize()) > MAX_METADATA_BYTES:
            raise ValueError('distributed programme metadata bound exceeded')

    def _core(self):
        return dict(policy=POLICY, model_identity=self.model_identity,
                    endpoints=[[a.payload() for a in row] for row in self.endpoints])

    def to_payload(self): return dict(self._core(), identity=self.identity)
    def serialize(self): return _canonical(self.to_payload())

    @classmethod
    def from_payload(cls, value):
        if (type(value) is not dict or set(value) != {'policy','model_identity','endpoints','identity'}
                or value['policy'] != POLICY or type(value['endpoints']) is not list
                or not 1 <= len(value['endpoints']) <= 64):
            raise ValueError('strict distributed programme descriptor required')
        if any(type(row) is not list for row in value['endpoints']):
            raise ValueError('distributed stage lists required')
        if not 0 < sum(map(len,value['endpoints'])) <= 16384:
            raise ValueError('distributed programme action bound exceeded')
        rows=[]
        for row in value['endpoints']:
            actions=[]
            for a in row:
                if type(a) is not dict or set(a) != {'element_id','force_N_per_reference_m','couple_Nm_per_reference_m'}:
                    raise ValueError('strict distributed stage action required')
                actions.append(GEBeamLineAction(**a))
            rows.append(tuple(actions))
        result=cls(value['model_identity'],tuple(rows))
        if result.identity != value['identity'] or result.serialize() != _canonical(value):
            raise ValueError('distributed programme identity/canonical mismatch')
        return result

    @classmethod
    def deserialize(cls, raw):
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_METADATA_BYTES:
            raise ValueError('bounded distributed programme bytes required')
        result=cls.from_payload(_json(raw))
        if result.serialize()!=raw: raise ValueError('canonical programme bytes required')
        return result

    def compile(self, owner):
        actions={}
        for row in self.endpoints:
            for a in row: actions.setdefault(a.element_id,a)
        if (len(self.endpoints)+1)*len(actions)*6 > MAX_VALUES:
            raise ValueError('compiled distributed programme storage bound exceeded')
        template=GEBeamDistributedLoad(self.model_identity,tuple(actions.values())).compile(owner)
        indices={a.element_id:i for i,a in enumerate(template.request.actions)}
        values=np.zeros((len(self.endpoints)+1,len(indices),6))
        for stage,row in enumerate(self.endpoints,1):
            for a in row:
                values[stage,indices[a.element_id]]=a.force_N_per_reference_m+a.couple_Nm_per_reference_m
        return CompiledDistributedProgramme(self,template,_owned(values))


@dataclass(frozen=True)
class CompiledDistributedProgramme:
    request: GEBeamDistributedProgramme
    template: object
    densities: np.ndarray

    @property
    def augmented_model_identity(self):
        return sha256(_canonical(dict(policy=POLICY,model=self.request.model_identity,
                                      programme=self.request.identity))).hexdigest()

    @property
    def result_field(self): return 'distributed_load'

    def validate_targets(self, accepted_position, targets):
        accepted,_,_=_progress(len(self.request.endpoints),accepted_position)
        if type(targets) is not tuple or not targets: raise ValueError('nonempty stage targets required')
        values=tuple(_progress(len(self.request.endpoints),p)[0] for p in targets)
        if any(b<=a for a,b in zip((accepted,)+values,values)):
            raise ValueError('distributed programme positions must advance strictly')
        if not set(range(floor(accepted)+1,floor(values[-1])+1)).issubset(values):
            raise ValueError('distributed programme must include every integer boundary')
        return values

    def densities_at(self, position):
        _,cursor,fraction=_progress(len(self.request.endpoints),position)
        if fraction==0.: return self.densities[cursor]
        a,b=self.densities[cursor:cursor+2]
        result=np.zeros_like(a); wb=Fraction(fraction); wa=1-wb
        for index in zip(*np.nonzero((a!=0.) | (b!=0.))):
            exact=wa*Fraction(float(a[index]))+wb*Fraction(float(b[index]))
            try: value=float(exact)
            except OverflowError as exc:
                raise StagedLoadResolutionError('distributed interpolation overflow') from exc
            if (not isfinite(value) or (exact!=0 and value==0.) or
                    (0.<abs(value)<np.finfo(float).tiny and Fraction(value)!=exact)):
                raise StagedLoadResolutionError('distributed interpolation unresolved')
            result[index]=value
        return _owned(result)

    def evaluate_at(self,state,position):
        force,jac,recovery=self.template.evaluate(state,densities=self.densities_at(position))
        return force,jac,replace(recovery,identity=self.request.identity)

    def programme_result(self,result,position):
        _,cursor,fraction=_progress(len(self.request.endpoints),position)
        return replace(result,load_programme_identity=self.request.identity,
            load_position=float(position),load_stage_cursor=cursor,load_stage_fraction=fraction,
            load_programme_complete=cursor==len(self.request.endpoints))

    def checkpoint(self,inner):
        if type(inner) is not bytes or not inner: raise ValueError('inner checkpoint required')
        header=_canonical(dict(policy=POLICY,request=self.request.to_payload(),inner_sha256=sha256(inner).hexdigest()))
        if len(header)>MAX_METADATA_BYTES or len(MAGIC)+4+len(header)+len(inner)+32>MAX_CHECKPOINT_BYTES:
            raise ValueError('distributed programme checkpoint bound exceeded')
        content=MAGIC+len(header).to_bytes(4,'little')+header+inner
        return content+sha256(content).digest()


def decode_checkpoint(raw):
    if type(raw) is not bytes or not raw.startswith(MAGIC) or not len(MAGIC)+36<len(raw)<=MAX_CHECKPOINT_BYTES:
        raise ValueError('bounded distributed programme checkpoint required')
    start=len(MAGIC)+4; size=int.from_bytes(raw[len(MAGIC):start],'little')
    if not 0<size<=MAX_METADATA_BYTES or start+size>=len(raw)-32 or sha256(memoryview(raw)[:-32]).digest()!=raw[-32:]:
        raise ValueError('distributed programme checkpoint bound/digest mismatch')
    header_raw=raw[start:start+size]; header=_json(header_raw)
    if type(header) is not dict or set(header)!={'policy','request','inner_sha256'} or header['policy']!=POLICY or _canonical(header)!=header_raw:
        raise ValueError('strict distributed programme checkpoint metadata required')
    request=GEBeamDistributedProgramme.from_payload(header['request']); inner=raw[start+size:-32]
    if sha256(inner).hexdigest()!=header['inner_sha256']: raise ValueError('distributed programme inner digest mismatch')
    return request,inner
