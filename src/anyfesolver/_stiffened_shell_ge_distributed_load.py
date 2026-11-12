"""Explicit native force/couple densities, per reference arclength.

Fixed spatial force work is conservative; fixed spatial couples have no scalar
potential. Full native cell rotations and actual element quadrature are retained.
New load identity binds objective Q4 / stable elastic S3 finite arithmetic.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from ._stiffened_shell_staged_load_experimental import _canonical, _identity, _number, _json
from ._stiffened_shell_ge_checkpoint_binary import MAX_CHECKPOINT_BYTES
from ._stiffened_shell_native_gravity import _owned, _sum, _product, _finite_array
from ._ge_beam3_p5.compensated_coordinates import split_sum

POLICY='STIFFENED_SHELL_GE_NATIVE_DISTRIBUTED_LOAD_V1_DEV'
MAGIC=b'ANYSGE_LINE1\0'
MAX_METADATA_BYTES=1024*1024
MAX_ACTIONS=16384


def admits_owner(owner):
    """Attachment scope shared by public capabilities and load compilation."""
    if owner._definition.distributing_footprints or owner._definition.finite_distributing_footprints:
        return False
    if owner._definition.footprints:
        return False
    if owner._line_samples:
        from ._stiffened_shell_q4_algorithmic_pullback import POLICY
        return bool(owner._finite_bl5 and owner._definition.q4_pullback_policy == POLICY)
    return bool(owner._definition.points)


def _vector(value):
    if np.shape(value)!=(3,): raise ValueError('three finite load density components required')
    return tuple(_number(v) for v in value)


@dataclass(frozen=True)
class GEBeamLineAction:
    element_id: int
    force_N_per_reference_m: tuple=(0.,0.,0.)
    couple_Nm_per_reference_m: tuple=(0.,0.,0.)

    def __post_init__(self):
        if type(self.element_id) is not int or self.element_id<=0:
            raise ValueError('positive native element identifier required')
        for name in ('force_N_per_reference_m','couple_Nm_per_reference_m'):
            object.__setattr__(self,name,_vector(getattr(self,name)))
        if not any(self.force_N_per_reference_m+self.couple_Nm_per_reference_m):
            raise ValueError('each native distributed action must be nonzero')

    def payload(self):
        return dict(element_id=self.element_id,force_N_per_reference_m=list(self.force_N_per_reference_m),
                    couple_Nm_per_reference_m=list(self.couple_Nm_per_reference_m))


@dataclass(frozen=True)
class GEBeamDistributedLoad:
    model_identity: str
    actions: tuple[GEBeamLineAction,...]
    identity: str=field(init=False)

    def __post_init__(self):
        object.__setattr__(self,'model_identity',_identity(self.model_identity))
        if type(self.actions) is not tuple or not 0<len(self.actions)<=MAX_ACTIONS or any(type(a) is not GEBeamLineAction for a in self.actions):
            raise ValueError('bounded tuple of native beam line actions required')
        ids=[a.element_id for a in self.actions]
        if len(set(ids))!=len(ids):raise ValueError('duplicate native line action element')
        object.__setattr__(self,'actions',tuple(sorted(self.actions,key=lambda a:a.element_id)))
        object.__setattr__(self,'identity',sha256(_canonical(self._core())).hexdigest())
        if len(self.serialize())>MAX_METADATA_BYTES:raise ValueError('bounded distributed descriptor required')

    def _core(self):
        return dict(policy=POLICY,model_identity=self.model_identity,actions=[a.payload() for a in self.actions])

    def to_payload(self):return dict(self._core(),identity=self.identity)
    def serialize(self):return _canonical(self.to_payload())

    @classmethod
    def from_payload(cls,value):
        if type(value) is not dict or set(value)!={'policy','model_identity','actions','identity'} or value['policy']!=POLICY or type(value['actions']) is not list:
            raise ValueError('strict distributed load descriptor required')
        if not 0<len(value['actions'])<=MAX_ACTIONS:raise ValueError('bounded action inventory required')
        rows=[]
        for row in value['actions']:
            if type(row) is not dict or set(row)!={'element_id','force_N_per_reference_m','couple_Nm_per_reference_m'}:
                raise ValueError('strict native distributed action required')
            rows.append(GEBeamLineAction(**row))
        result=cls(value['model_identity'],tuple(rows))
        if result.identity!=value['identity'] or result.serialize()!=_canonical(value):
            raise ValueError('distributed descriptor identity/canonical mismatch')
        return result

    @classmethod
    def deserialize(cls,raw):
        if type(raw) is not bytes or not 0<len(raw)<=MAX_METADATA_BYTES:
            raise ValueError('bounded distributed load bytes required')
        result=cls.from_payload(_json(raw))
        if result.serialize()!=raw:raise ValueError('canonical distributed load bytes required')
        return result

    def compile(self,owner):
        from ._stiffened_shell_native_distributed_load import compile_native_distributed
        owner._require_open()
        if self.model_identity!=owner._identity:raise ValueError('distributed load belongs to a different model')
        if not admits_owner(owner):
            raise ValueError('distributed load requires point or objective elastic Q4 BL5 attachments')
        parts={element.element_id:(i,element,slots) for i,(element,slots,_) in enumerate(owner._beam_parts)}
        compiled=[]
        for action in self.actions:
            if action.element_id not in parts:raise ValueError('distributed action references unknown native element')
            i,element,slots=parts[action.element_id]
            kernel=compile_native_distributed(element.operator,action.force_N_per_reference_m,action.couple_Nm_per_reference_m)
            compiled.append((kernel,tuple(int(v)+owner._shell_count for v in slots),i,action.element_id))
        identity=sha256(_canonical(dict(model=self.model_identity,distributed_load=self.identity))).hexdigest()
        context=GEDistributedContext(self,identity,owner._K.shape[0],len(owner._beam_parts),tuple(compiled))
        reference=SimpleNamespace(q=np.zeros(context.count),cells=np.tile(np.eye(3),(context.beam_count,2,1,1)))
        force,_,recovery=context.evaluate(reference)
        object.__setattr__(context,'reference_gradient',force)
        object.__setattr__(context,'reference_recovery',recovery)
        return context


@dataclass(frozen=True)
class GEDistributedRecovery:
    identity: str
    applied_force: tuple
    applied_moment_about_origin: tuple
    force_potential: float
    potential: float | None
    element_ids: tuple

    def scaled(self,factor):
        factor=_number(factor)
        return GEDistributedRecovery(self.identity,tuple(_product(self.applied_force,factor)),
            tuple(_product(self.applied_moment_about_origin,factor)),float(_product(self.force_potential,factor)),
            None if self.potential is None else float(_product(self.potential,factor)),self.element_ids)


@dataclass(frozen=True)
class GEDistributedContext:
    request: GEBeamDistributedLoad
    augmented_model_identity: str
    count: int
    beam_count: int
    beams: tuple
    reference_gradient: np.ndarray=field(init=False,repr=False)
    reference_recovery: GEDistributedRecovery=field(init=False)

    @property
    def result_field(self):return 'distributed_load'

    def checkpoint(self,inner):return encode_checkpoint(self,inner)

    def evaluate(self,state,*,densities=None):
        # Rows follow the compiled element order, not caller/model numbering.
        # This is a private evaluation seam for programmes with their own load
        # identity. The existing public descriptor and its recovery stay fixed.
        if densities is not None:
            densities=_finite_array(densities,(len(self.beams),6),'distributed stage densities')
        q=np.asarray(state.q);cells=np.asarray(state.cells)
        if np.iscomplexobj(q) or q.shape!=(self.count,) or not np.isfinite(q).all() or cells.shape!=(self.beam_count,2,3,3):
            raise ValueError('finite matching distributed load state required')
        terms=[[] for _ in range(self.count)];rows=[];columns=[];values=[];forces=[];moments=[];potentials=[];conservative=True
        for row,(kernel,slots,i,element_id) in enumerate(self.beams):
            displacement=q[list(slots[:18])].reshape(3,6)[:,:3]
            high=np.empty((3,3));low=np.empty((3,3))
            for at in np.ndindex(high.shape):
                high[at],low[at]=split_sum((float(kernel.coordinates[at]),float(displacement[at])))
            result=(kernel.evaluate(high,low,cells[i]) if densities is None else
                    kernel.evaluate_density(high,low,cells[i],densities[row,:3],densities[row,3:]))
            for slot,v in zip(slots,result.external_force,strict=True):terms[slot].append(float(v))
            rr,cc=np.nonzero(result.spatial_jacobian)
            rows.extend(slots[int(r)] for r in rr);columns.extend(slots[int(c)] for c in cc)
            values.extend(result.spatial_jacobian[rr,cc])
            forces.append(result.total_force);moments.append(result.total_moment)
            potentials.append(result.force_potential);conservative=conservative and result.potential is not None
        force=_owned([_sum(t) for t in terms]);jac=sparse.coo_matrix((values,(rows,columns)),shape=(self.count,self.count)).tocsr()
        if not np.isfinite(jac.data).all():raise ValueError('finite distributed Jacobian required')
        potential=_sum(potentials)
        recovery=GEDistributedRecovery(self.request.identity,tuple(_sum(f[i] for f in forces) for i in range(3)),
            tuple(_sum(m[i] for m in moments) for i in range(3)),potential,potential if conservative else None,
            tuple(a.element_id for a in self.request.actions))
        return force,jac,recovery


def encode_checkpoint(context,inner):
    if type(context) is not GEDistributedContext or type(inner) is not bytes or not inner:
        raise ValueError('compiled distributed context and checkpoint required')
    header=_canonical(dict(policy=POLICY,request=context.request.to_payload(),inner_sha256=sha256(inner).hexdigest()))
    if len(header)>MAX_METADATA_BYTES or len(MAGIC)+4+len(header)+len(inner)+32>MAX_CHECKPOINT_BYTES:
        raise ValueError('distributed checkpoint byte bound exceeded')
    content=MAGIC+len(header).to_bytes(4,'little')+header+inner
    return content+sha256(content).digest()


def decode_checkpoint(raw):
    if type(raw) is not bytes or not raw.startswith(MAGIC) or not len(MAGIC)+4+32<len(raw)<=MAX_CHECKPOINT_BYTES:
        raise ValueError('bounded distributed checkpoint framing required')
    start=len(MAGIC)+4;size=int.from_bytes(raw[len(MAGIC):start],'little')
    if not 0<size<=MAX_METADATA_BYTES or start+size>=len(raw)-32 or sha256(memoryview(raw)[:-32]).digest()!=raw[-32:]:
        raise ValueError('distributed checkpoint bound/digest mismatch')
    header_raw=raw[start:start+size];header=_json(header_raw)
    if type(header) is not dict or set(header)!={'policy','request','inner_sha256'} or header['policy']!=POLICY or _canonical(header)!=header_raw:
        raise ValueError('strict distributed checkpoint metadata required')
    request=GEBeamDistributedLoad.from_payload(header['request']);inner=raw[start+size:-32]
    if sha256(inner).hexdigest()!=header['inner_sha256']:raise ValueError('distributed inner checkpoint mismatch')
    return request,inner
