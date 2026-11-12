"""Private typed supported-model restart chain for native fibre statics.

No pickle, mechanical fallback, history fabrication or public routing. The
caller must preserve/verify the external SHA-256 for provenance; self-hashes
detect inconsistency, not malicious replacement of an entire valid history.
"""
from hashlib import sha256
from math import isfinite
from time import monotonic
import numpy as np
from scipy import sparse
from .assembly import build_constraint_transformation
from ._native_rotation_state import rotation_exponential
from ._ge_beam3_native_fibre_static_element import NativeFibreStaticElement,SCHEMA as STATE_SCHEMA
from ._ge_beam3_fibre_static_boundary import StaticTrial
from ._ge_beam3_fibre_cell import CellHistory
from ._ge_beam3_fibre_section import FibreHistory
from ._ge_beam3_p5_seeded.core import canonical,sha
from ._ge_beam3_p5_seeded.codec import _load,MAX_BYTES
from ._native_reference_modal import _owned

SCHEMA='GE_BEAM3_SUPPORTED_NATIVE_FIBRE_STATIC_RESTART_V1'


def capture_solver_state(model,element_id,state,*,exact_guard):
    """Detach only this exact private native type at the solver input boundary.

    This captures one supplied state, not an authenticated restart chain. The
    complete checkpoint must be validated by decode_checkpoint before resuming.
    Recheck the caller's operation guard even when serialization raises.
    """
    element=model.mesh.elements.get(element_id)
    if type(element) is not NativeFibreStaticElement:
        raise TypeError('exact private native fibre restart element required')
    try:
        raw=canonical(state)
    finally:
        exact_guard(model,context='native fibre solver restart input observation')
    made=_state(_load(raw.decode('ascii')),element)
    element._validate(model.mesh,made)
    exact_guard(model,context='native fibre solver restart input validation')
    return made


def supported_solver_coordinates(model,transform,offset,displacements):
    """Exact coordinate extraction for this candidate's selection-only supports."""
    _,_,n,free,fixed,_=_model(model)
    expected=sparse.eye(n,format='csr')[:,list(free)]
    if transform.shape!=expected.shape or (transform!=expected).nnz or np.any(offset):
        raise ValueError('native restart requires exact supported selection map')
    u=np.asarray(displacements)
    if u.shape!=(n,) or u.dtype!=np.float64 or not np.isfinite(u).all() or np.any(u[list(fixed)]):
        raise ValueError('native restart exact supported displacement required')
    q=u[list(free)].copy()
    if not np.array_equal(transform@q,u): raise ValueError('native restart exact coordinate reconstruction')
    return q,0.


def _keys(value,keys):
    if type(value) is not dict or set(value)!=set(keys): raise ValueError('exact native restart keys')


def _float(value):
    if type(value) is not float or not isfinite(value): raise ValueError('finite binary64 restart value')
    return value


def _array(value,shape):
    def visit(v,dimensions):
        if not dimensions: return _float(v)
        if type(v) is not list or len(v)!=dimensions[0]: raise ValueError('exact native restart array shape')
        return [visit(x,dimensions[1:]) for x in v]
    return _owned(visit(value,shape))


def _history(value,element):
    _keys(value,('cell_identity','stations'))
    if type(value['cell_identity']) is not str or type(value['stations']) is not list or len(value['stations'])!=2*element.operator.order:
        raise ValueError('native restart station history extent')
    stations=[]
    for station in value['stations']:
        _keys(station,('section_identity','rows'))
        if type(station['section_identity']) is not str: raise ValueError('native restart section identity')
        rows=_array(station['rows'],(len(element.section.fibres),4))
        stations.append(FibreHistory(station['section_identity'],tuple(tuple(map(float,row)) for row in rows)))
    return CellHistory(value['cell_identity'],tuple(stations))


def _state(value,element):
    _keys(value,('schema','element_identity','epoch','previous_state_sha256','committed_total_u','committed_nodal_rotation_matrices',
        'positions','position_low','origins','seed_rotations','seed_resultants','response','state_sha256'))
    if value['schema']!=STATE_SCHEMA: raise ValueError('native restart state schema')
    made=dict(value)
    for name,shape in (('committed_total_u',(18,)),('committed_nodal_rotation_matrices',(3,3,3)),('positions',(3,3)),
                       ('position_low',(3,3)),('seed_rotations',(2,3,3)),('seed_resultants',(18,))):
        made[name]=_array(value[name],shape)
    made['origins']=_history(value['origins'],element); r=value['response']
    _keys(r,('rotations','resultants','residual','hessian','full_residual','full_hessian','history','potential',
             'internal_error','iterations','input_sha256','production_qualified','dynamic_reduction_authorized'))
    if r['production_qualified'] is not False or r['dynamic_reduction_authorized'] is not False: raise ValueError('restart cannot grant qualification')
    if type(r['iterations']) is not int or not 0<=r['iterations']<=24 or type(r['input_sha256']) is not str: raise ValueError('native restart response metadata')
    made['response']=StaticTrial(_array(r['rotations'],(2,3,3)),_array(r['resultants'],(18,)),_array(r['residual'],(18,)),
        _array(r['hessian'],(18,18)),_array(r['full_residual'],(42,)),_array(r['full_hessian'],(42,42)),
        _history(r['history'],element),_float(r['potential']),_float(r['internal_error']),r['iterations'],r['input_sha256'])
    if canonical(made)!=canonical(value): raise ValueError('native restart typed round trip differs')
    return made


def _model(model):
    elements=tuple(sorted(model.mesh.elements.items())); nodes=tuple(sorted(model.mesh.nodes))
    n=model.mesh.dof_manager.total_dofs
    if not 1<=len(elements)<=16 or not 1<=n<=512 or n!=6*len(nodes): raise ValueError('bounded complete native restart model')
    if model.constraint_equations or model.mesh.point_masses or model.mesh.element_activity is not None: raise ValueError('restart MPC/activity/dynamics not admitted')
    for eid,e in elements:
        if type(eid) is not int or type(e) is not NativeFibreStaticElement or eid!=e.element_id or model.materials.get(e.material_name) is not e.section:
            raise ValueError('exact complete native restart elements/materials')
        e._check(model.mesh)
    if set(nodes)!={node for _,e in elements for node in e.node_ids}: raise ValueError('unconnected native restart node')
    for index,node in enumerate(nodes):
        if tuple(model.mesh.dof_manager.get_node_dofs(node))!=tuple(range(6*index,6*index+6)): raise ValueError('native restart node/DOF ordering')
    _,_,t,offset,free,info=build_constraint_transformation(sparse.eye(n,format='csr'),np.zeros(n),model)
    if info['slave_dofs'] or np.any(offset) or t.nnz!=len(free) or np.any(t.data!=1.) or len(free)==n: raise ValueError('homogeneous supported restart only')
    fixed=tuple(i for i in range(n) if i not in free)
    for index in range(len(nodes)):
        if len(set(range(6*index+3,6*index+6))&set(fixed)) not in (0,3): raise ValueError('partial rotation support not qualified')
    descriptor=dict(nodes=[(i,model.mesh.nodes[i].coords(),list(model.mesh.dof_manager.get_node_dofs(i))) for i in nodes],
        elements=[(i,e.to_dict()) for i,e in elements],boundaries=[vars(b) for b in model.boundary_conditions],fixed=fixed)
    return elements,nodes,n,tuple(map(int,free)),fixed,sha(descriptor)


def _forces(rows,nodes,n):
    if type(rows) is not tuple or not 1<=len(rows)<=len(nodes): raise ValueError('explicit bounded nodal force pattern')
    force=np.zeros(n); ids=[]
    for row in rows:
        if type(row) is not tuple or len(row)!=4 or type(row[0]) is not int or row[0] not in nodes: raise ValueError('restart nodal force row')
        ids.append(row[0]); index=nodes.index(row[0]); force[6*index:6*index+3]=[_float(v) for v in row[1:]]
    if ids!=sorted(set(ids)) or not np.any(force): raise ValueError('unique ordered nonzero force pattern')
    return force


def _wire(snapshot,elements):
    _keys(snapshot,('load_factor','displacements','states'))
    if type(snapshot['states']) is not dict or set(snapshot['states'])!={i for i,_ in elements}: raise ValueError('complete restart element states')
    return dict(load_factor=snapshot['load_factor'],displacements=snapshot['displacements'],
        states=[dict(element_id=i,state=snapshot['states'][i]) for i,_ in elements])


def validate_chain(model,nodal_forces,snapshots):
    started=monotonic(); elements,nodes,n,free,fixed,identity=_model(model); force=_forces(nodal_forces,nodes,n)
    if type(snapshots) is not tuple or not 1<=len(snapshots)<=65: raise ValueError('bounded complete restart chain')
    previous=None
    for index,snapshot in enumerate(snapshots):
        _wire(snapshot,elements); factor=_float(snapshot['load_factor']); u=np.asarray(snapshot['displacements'])
        if u.shape!=(n,) or u.dtype.kind!='f' or not np.isfinite(u).all() or np.any(u[list(fixed)]): raise ValueError('restart displacement/support mismatch')
        if index==0 and (factor!=0. or np.any(u)): raise ValueError('restart requires stress-free genesis')
        residual=-factor*force.copy(); shared={}
        for eid,e in elements:
            if monotonic()-started>60.: raise RuntimeError('native restart validation deadline')
            state=snapshot['states'][eid]; mapping=list(e.get_dof_mapping(model.mesh)); e._validate(model.mesh,state,u[mapping])
            if state['epoch']!=index: raise ValueError('restart skipped or duplicated state epoch')
            residual[mapping]+=state['response'].residual
            for local,node in enumerate(e.node_ids):
                rotation=state['committed_nodal_rotation_matrices'][local]
                if node in shared and not np.array_equal(shared[node],rotation): raise ValueError('restart shared-node rotation disagreement')
                shared[node]=rotation
            if previous is not None:
                old=previous['states'][eid]
                if state['previous_state_sha256']!=old['state_sha256'] or canonical(state['origins'])!=canonical(old['response'].history): raise ValueError('restart predecessor/history chain mismatch')
                if not np.array_equal(state['seed_rotations'],old['response'].rotations) or not np.array_equal(state['seed_resultants'],old['response'].resultants): raise ValueError('restart internal seed chain mismatch')
                increments=(state['committed_total_u']-old['committed_total_u']).reshape(3,6)[:,3:]
                expected=np.array([rotation_exponential(d)@r for d,r in zip(increments,old['committed_nodal_rotation_matrices'])])
                if not np.array_equal(expected,state['committed_nodal_rotation_matrices']): raise ValueError('restart multiplicative rotation chain mismatch')
        error=float(np.linalg.norm(residual[list(free)])/max(1.,np.linalg.norm(factor*force)))
        if not isfinite(error) or error>1e-11: raise ValueError('restart accepted snapshot not equilibrated')
        previous=snapshot
    if _model(model)[-1]!=identity: raise ValueError('restart model changed during validation')
    return identity


def encode_checkpoint(model,nodal_forces,snapshots):
    identity=validate_chain(model,nodal_forces,snapshots); elements=_model(model)[0]
    body=dict(schema=SCHEMA,model_sha256=identity,nodal_forces=nodal_forces,
        snapshots=[_wire(s,elements) for s in snapshots],production_qualified=False)
    raw=canonical({**body,'checkpoint_sha256':sha(body)})
    if len(raw)>MAX_BYTES: raise ValueError('native restart byte limit')
    return raw


def decode_checkpoint(model,raw,*,expected_sha256):
    if type(raw) is not bytes or type(expected_sha256) is not str or sha256(raw).hexdigest()!=expected_sha256: raise ValueError('native restart external SHA-256 mismatch')
    value=_load(raw.decode('ascii')); _keys(value,('schema','model_sha256','nodal_forces','snapshots','production_qualified','checkpoint_sha256'))
    body={k:v for k,v in value.items() if k!='checkpoint_sha256'}
    elements,nodes,n,_,_,identity=_model(model)
    if value['schema']!=SCHEMA or value['production_qualified'] is not False or value['model_sha256']!=identity or value['checkpoint_sha256']!=sha(body): raise ValueError('native restart model/schema/hash mismatch')
    if type(value['nodal_forces']) is not list or any(type(r) is not list for r in value['nodal_forces']): raise ValueError('native restart force list')
    forces=tuple(tuple(r) for r in value['nodal_forces']); _forces(forces,nodes,n)
    if type(value['snapshots']) is not list or not 1<=len(value['snapshots'])<=65: raise ValueError('native restart snapshot count')
    snapshots=[]
    for record in value['snapshots']:
        _keys(record,('load_factor','displacements','states'))
        rows=record['states']
        if type(rows) is not list or len(rows)!=len(elements): raise ValueError('native restart complete states')
        states={}
        for row,(eid,e) in zip(rows,elements):
            _keys(row,('element_id','state'))
            if type(row['element_id']) is not int or row['element_id']!=eid: raise ValueError('native restart ordered element identity')
            states[eid]=_state(row['state'],e)
        snapshots.append(dict(load_factor=_float(record['load_factor']),displacements=_array(record['displacements'],(n,)),states=states))
    snapshots=tuple(snapshots); validate_chain(model,forces,snapshots)
    if encode_checkpoint(model,forces,snapshots)!=raw: raise ValueError('native restart canonical typed round trip')
    return forces,snapshots
