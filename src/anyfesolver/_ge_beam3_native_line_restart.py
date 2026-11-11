"""Typed, load-path-bound accepted chains for private native line statics.

An external checkpoint SHA-256 is mandatory. A self hash cannot authenticate
a maliciously replaced complete valid history. No pickle or history inference.
"""
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic
import numpy as np
from scipy import sparse
from .assembly import build_constraint_transformation
from ._native_rotation_state import rotation_exponential
from ._ge_beam3_native_line_static_element import NativeLineFibreStaticElement, SCHEMA as STATE_SCHEMA
from ._ge_beam3_native_line_loading import LinePattern, nodal_force_vector
from ._ge_beam3_native_line_program import model_identity
from ._ge_beam3_native_fibre_restart import _keys, _float, _array, _history
from ._ge_beam3_fibre_static_boundary import StaticTrial
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES

SCHEMA = 'GE_BEAM3_SUPPORTED_NATIVE_LINE_STATIC_RESTART_V1'


@dataclass(frozen=True)
class LoadPoint:
    parameter: float
    constant: LinePattern
    proportional: LinePattern

    def require(self, model):
        _float(self.parameter)
        if not 0. <= self.parameter <= 1. or type(self.constant) is not LinePattern or type(self.proportional) is not LinePattern:
            raise ValueError('exact bounded native line load point')
        self.constant.require(model.mesh); self.proportional.require(model.mesh)

    def effective(self, model):
        self.require(model); rows=[]
        for eid in sorted(model.mesh.elements):
            force=self.constant.force(eid)+self.parameter*self.proportional.force(eid)
            if np.any(force): rows.append((eid,*map(float,force)))
        return LinePattern(tuple(rows))


def _pattern(value, model):
    _keys(value,('rows','signature'))
    if type(value['rows']) is not list or any(type(row) is not list for row in value['rows']):
        raise ValueError('native line restart pattern rows')
    made=LinePattern(tuple(tuple(row) for row in value['rows']))
    made.require(model.mesh)
    if canonical(made)!=canonical(value): raise ValueError('native line restart pattern signature/round trip')
    return made


def _point(value, model):
    _keys(value,('parameter','constant','proportional'))
    made=LoadPoint(_float(value['parameter']),_pattern(value['constant'],model),_pattern(value['proportional'],model))
    made.require(model)
    if canonical(made)!=canonical(value): raise ValueError('native line restart load point round trip')
    return made


def _state(value, element, model):
    _keys(value,('schema','element_identity','epoch','previous_state_sha256','committed_total_u','committed_nodal_rotation_matrices',
        'positions','position_low','origins','seed_rotations','seed_resultants','response','state_sha256','load_pattern'))
    if value['schema']!=STATE_SCHEMA: raise ValueError('native line restart state schema')
    made=dict(value)
    for name,shape in (('committed_total_u',(18,)),('committed_nodal_rotation_matrices',(3,3,3)),('positions',(3,3)),
                       ('position_low',(3,3)),('seed_rotations',(2,3,3)),('seed_resultants',(18,))):
        made[name]=_array(value[name],shape)
    made['load_pattern']=_pattern(value['load_pattern'],model)
    made['origins']=_history(value['origins'],element); r=value['response']
    _keys(r,('rotations','resultants','residual','hessian','full_residual','full_hessian','history','potential',
        'internal_error','iterations','input_sha256','production_qualified','dynamic_reduction_authorized'))
    if r['production_qualified'] is not False or r['dynamic_reduction_authorized'] is not False:
        raise ValueError('line restart cannot grant qualification')
    if type(r['iterations']) is not int or not 0<=r['iterations']<=24 or type(r['input_sha256']) is not str:
        raise ValueError('native line restart response metadata')
    made['response']=StaticTrial(_array(r['rotations'],(2,3,3)),_array(r['resultants'],(18,)),_array(r['residual'],(18,)),
        _array(r['hessian'],(18,18)),_array(r['full_residual'],(42,)),_array(r['full_hessian'],(42,42)),
        _history(r['history'],element),_float(r['potential']),_float(r['internal_error']),r['iterations'],r['input_sha256'])
    if canonical(made)!=canonical(value): raise ValueError('native line restart typed state round trip')
    return made


def capture_solver_state(model,element_id,state,*,exact_guard):
    element=model.mesh.elements.get(element_id)
    if type(element) is not NativeLineFibreStaticElement: raise TypeError('exact native line restart element')
    try: raw=canonical(state)
    finally: exact_guard(model,context='native line restart input observation')
    made=_state(_load(raw.decode('ascii')),element,model)
    element._validate(model.mesh,made)
    exact_guard(model,context='native line restart input validation')
    return made


def _model(model):
    identity=model_identity(model); elements=tuple(sorted(model.mesh.elements.items()))
    nodes=tuple(sorted(model.mesh.nodes)); n=model.mesh.dof_manager.total_dofs
    for index,node in enumerate(nodes):
        if tuple(model.mesh.dof_manager.get_node_dofs(node))!=tuple(range(6*index,6*index+6)):
            raise ValueError('native line restart node/DOF ordering')
    _,_,t,offset,free,info=build_constraint_transformation(sparse.eye(n,format='csr'),np.zeros(n),model)
    if info['slave_dofs'] or np.any(offset) or t.nnz!=len(free) or np.any(t.data!=1.) or len(free)==n:
        raise ValueError('native line restart homogeneous supported selection required')
    fixed=tuple(i for i in range(n) if i not in free)
    return elements,n,tuple(map(int,free)),fixed,identity


def supported_solver_coordinates(model,transform,offset,displacements):
    _,n,free,fixed,_=_model(model); expected=sparse.eye(n,format='csr')[:,list(free)]
    if transform.shape!=expected.shape or (transform!=expected).nnz or np.any(offset):
        raise ValueError('native line restart exact supported map')
    u=np.asarray(displacements)
    if u.shape!=(n,) or u.dtype!=np.float64 or not np.isfinite(u).all() or np.any(u[list(fixed)]):
        raise ValueError('native line restart supported displacement')
    q=u[list(free)].copy()
    if not np.array_equal(transform@q,u): raise ValueError('native line restart coordinate reconstruction')
    return q,0.


def _wire(snapshot,elements):
    _keys(snapshot,('load_point','displacements','states'))
    if type(snapshot['load_point']) is not LoadPoint or type(snapshot['states']) is not dict or set(snapshot['states'])!={i for i,_ in elements}:
        raise ValueError('complete typed native line restart snapshot')
    return dict(load_point=snapshot['load_point'],displacements=snapshot['displacements'],
        states=[dict(element_id=i,state=snapshot['states'][i]) for i,_ in elements])


def validate_chain(model,snapshots):
    started=monotonic(); elements,n,free,fixed,identity=_model(model)
    if type(snapshots) is not tuple or not 1<=len(snapshots)<=65: raise ValueError('bounded complete native line chain')
    previous=None
    for index,snapshot in enumerate(snapshots):
        _wire(snapshot,elements); point=snapshot['load_point']; pattern=point.effective(model)
        u=np.asarray(snapshot['displacements'])
        if u.shape!=(n,) or u.dtype!=np.float64 or not np.isfinite(u).all() or np.any(u[list(fixed)]):
            raise ValueError('native line restart displacement/support mismatch')
        if index==0 and (point.parameter!=0. or pattern.rows or np.any(u)):
            raise ValueError('native line restart stress-free genesis required')
        net=np.zeros(n); shared={}
        for eid,e in elements:
            if monotonic()-started>60.: raise RuntimeError('native line restart validation deadline')
            state=snapshot['states'][eid]; mapping=list(e.get_dof_mapping(model.mesh))
            e._validate(model.mesh,state,u[mapping])
            if state['epoch']!=index or state['load_pattern']!=pattern:
                raise ValueError('native line restart epoch/effective-load mismatch')
            net[mapping]+=state['response'].residual
            for local,node in enumerate(e.node_ids):
                rotation=state['committed_nodal_rotation_matrices'][local]
                if node in shared and not np.array_equal(shared[node],rotation): raise ValueError('native line shared rotation disagreement')
                shared[node]=rotation
            if previous is not None:
                old=previous['states'][eid]
                if state['previous_state_sha256']!=old['state_sha256'] or canonical(state['origins'])!=canonical(old['response'].history):
                    raise ValueError('native line restart predecessor/history mismatch')
                if not np.array_equal(state['seed_rotations'],old['response'].rotations) or not np.array_equal(state['seed_resultants'],old['response'].resultants):
                    raise ValueError('native line restart internal seed mismatch')
                increments=(state['committed_total_u']-old['committed_total_u']).reshape(3,6)[:,3:]
                expected=np.array([rotation_exponential(d)@r for d,r in zip(increments,old['committed_nodal_rotation_matrices'])])
                if not np.array_equal(expected,state['committed_nodal_rotation_matrices']): raise ValueError('native line multiplicative rotation mismatch')
        external=nodal_force_vector(model,pattern)
        error=float(np.linalg.norm(net[list(free)])/max(1.,np.linalg.norm(external)))
        if not np.isfinite(error) or error>1e-11: raise ValueError('native line accepted snapshot not equilibrated')
        previous=snapshot
    if _model(model)[-1]!=identity: raise ValueError('native line restart model changed')
    return identity


def encode_checkpoint(model,snapshots):
    identity=validate_chain(model,snapshots); elements=_model(model)[0]
    body=dict(schema=SCHEMA,model_sha256=identity,snapshots=[_wire(s,elements) for s in snapshots],production_qualified=False)
    raw=canonical({**body,'checkpoint_sha256':sha(body)})
    if len(raw)>MAX_BYTES: raise ValueError('native line restart byte limit')
    return raw


def decode_checkpoint(model,raw,*,expected_sha256):
    if type(raw) is not bytes or type(expected_sha256) is not str or sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('native line restart external SHA-256 mismatch')
    value=_load(raw.decode('ascii')); _keys(value,('schema','model_sha256','snapshots','production_qualified','checkpoint_sha256'))
    elements,n,_,_,identity=_model(model); body={k:v for k,v in value.items() if k!='checkpoint_sha256'}
    if value['schema']!=SCHEMA or value['production_qualified'] is not False or value['model_sha256']!=identity or value['checkpoint_sha256']!=sha(body):
        raise ValueError('native line restart model/schema/hash mismatch')
    if type(value['snapshots']) is not list or not 1<=len(value['snapshots'])<=65: raise ValueError('native line snapshot count')
    snapshots=[]
    for record in value['snapshots']:
        _keys(record,('load_point','displacements','states')); rows=record['states']
        if type(rows) is not list or len(rows)!=len(elements): raise ValueError('native line restart complete states')
        states={}
        for row,(eid,e) in zip(rows,elements):
            _keys(row,('element_id','state'))
            if type(row['element_id']) is not int or row['element_id']!=eid: raise ValueError('native line ordered element identity')
            states[eid]=_state(row['state'],e,model)
        snapshots.append(dict(load_point=_point(record['load_point'],model),displacements=_array(record['displacements'],(n,)),states=states))
    snapshots=tuple(snapshots); validate_chain(model,snapshots)
    if encode_checkpoint(model,snapshots)!=raw: raise ValueError('native line restart canonical typed round trip')
    return snapshots
