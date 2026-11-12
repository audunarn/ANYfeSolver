"""Private generalized retained-coordinate state and authenticated replay.

No research imports, legacy beam mechanics, public routing or native model
state mutation. One immutable accepted chain owns geometry and section history.
"""
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic
import numpy as np
from ._ge_beam3_native_generalized_restart import _model, _history
from ._ge_beam3_native_generalized_program import retained_model_identity
from .constraint_audit import require_valid_constraints
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_fibre_line_work import evaluate as line_work
from ._ge_beam3_generalized_static_boundary import cell_couple_load, spatial_jacobian
from ._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair
from ._ge_beam3_p5.algebra import rotation
from ._ge_beam3_p5.arrays import _frames
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES
from ._native_reference_modal import _owned
from ._ge_beam3_operator_validation_scope import operator_call
from ._ge_beam3_effective_deadline import effective_deadline_seconds

SCHEMA='GE_BEAM3_RETAINED_GENERALIZED_DISTRIBUTED_ACCEPTED_CHAIN_V1'
POLICY='CANDIDATE_GE_BEAM3_RETAINED_GENERALIZED_DISTRIBUTED_FORCE_V1'

def _retained_size(nodal, elements):
    from ._ge_beam3_refinement_capacity import retained_limits
    maximum,full,_=retained_limits()
    if (type(nodal) is not int or type(elements) is not int or not 1<=elements<=maximum
            or not 6<=nodal<=512 or nodal%6 or nodal+24*elements>full):
        raise ValueError('bounded retained generalized system')
    return nodal+24*elements

@dataclass(frozen=True)
class Program:
    targets: tuple
    pattern: DistributedPattern
    max_iterations: int=24
    max_backtracks: int=8

    def __post_init__(self):
        if (type(self.targets) is not tuple or not 1<=len(self.targets)<=16
                or any(type(v) is not float or not np.isfinite(v) or abs(v)>16 for v in self.targets)
                or type(self.pattern) is not DistributedPattern
                or type(self.max_iterations) is not int or not 1<=self.max_iterations<=24
                or type(self.max_backtracks) is not int or not 0<=self.max_backtracks<=8):
            raise ValueError('explicit bounded retained generalized program')

@dataclass(frozen=True)
class Mechanical:
    positions: np.ndarray
    position_low: np.ndarray
    nodal_frames: np.ndarray
    cell_rotations: np.ndarray
    resultants: np.ndarray

    def descriptor(self):return {k:getattr(self,k) for k in self.__dataclass_fields__}

@dataclass(frozen=True)
class State:
    mechanical: Mechanical
    origins: tuple
    histories: tuple
    completed_targets: int
    model_sha256: str

class Context:
    program_type=Program
    schema=SCHEMA
    policy=POLICY

    def __init__(self,model,program):
        if type(program) is not self.program_type:raise ValueError('exact generalized retained program')
        program.__post_init__();program.pattern.require(model.mesh)
        self.started=monotonic();self.model=model;self.program=program
        self.elements,self.nodal_count,nodal_free,self.fixed,self.model_identity=_model(model,retained=True)
        self.support_identity=canonical(require_valid_constraints(model))
        self.node_ids=tuple(sorted(model.mesh.nodes));self.index={node:i for i,node in enumerate(self.node_ids)}
        self.count=_retained_size(self.nodal_count,len(self.elements))
        self.free=np.array((*nodal_free,*range(self.nodal_count,self.count)),dtype=int)
        self.nodes=[];self.slots=[];frames=[None]*len(self.node_ids);self.scale=np.ones(self.count)
        self.length=max(float(np.linalg.norm(e.operator.reference.coordinates[-1]-e.operator.reference.coordinates[0])) for _,e in self.elements)
        self.equilibrium=list(nodal_free);self.compatibility=[]
        for i,(eid,e) in enumerate(self.elements):
            nodes=[self.index[n] for n in e.node_ids];first=self.nodal_count+24*i
            self.nodes.append(nodes);self.slots.append(list(e.get_dof_mapping(model.mesh))+list(range(first,first+24)))
            self.equilibrium.extend(range(first,first+6));self.compatibility.extend(range(first+6,first+24))
            self.scale[first:first+12]=self.length
            for j,node in enumerate(nodes):
                frame=e.operator.reference.nodal_triads[j]
                if frames[node] is not None and not np.array_equal(frames[node],frame):raise ValueError('shared material-frame authority differs')
                frames[node]=frame
        for i in range(len(self.node_ids)):self.scale[6*i+3:6*i+6]=self.length
        self.reference_positions=_owned([model.mesh.nodes[i].coords() for i in self.node_ids])
        self.reference_frames=_owned(frames)
        self.program_bytes=canonical(program)
        self.identity=sha(dict(schema=self.schema,formulation=self.policy,model=self.model_identity,program=program,
            operators=[e.operator.identity for _,e in self.elements],node_ids=self.node_ids))
        history=tuple(e.operator.cell.virgin() for _,e in self.elements)
        mechanical=self.make(dict(positions=self.reference_positions,position_low=np.zeros_like(self.reference_positions),
            nodal_frames=self.reference_frames,cell_rotations=np.tile(np.eye(3),(len(self.elements),2,1,1)),
            resultants=np.zeros((len(self.elements),18))))
        self._issued={};self._issued_records=set()
        self.initial,self.genesis=self.record(mechanical,history,0,0,self.identity)
        self.guard()

    def guard(self):
        if monotonic()-self.started>effective_deadline_seconds(120):raise RuntimeError('retained generalized context deadline')
        # The complete supported transform is validated once in _model above.
        # Recheck all model/DOF authority and the full audited constraint rows,
        # including element MPC providers, without rebuilding sparse I/T/K/F.
        # This is not an object-identity cache or a reduced-frequency guard.
        total=self.model.mesh.dof_manager.total_dofs
        if (type(total) is not type(self.nodal_count) or total!=self.nodal_count
                or retained_model_identity(self.model)!=self.model_identity
                or canonical(require_valid_constraints(self.model))!=self.support_identity
                or canonical(self.program)!=self.program_bytes):
            raise ValueError('retained generalized frozen model/program changed')
        self.program.pattern.require(self.model.mesh)

    def make(self,data,*,decoded=False):
        if type(data) is not dict or set(data)!=set(Mechanical.__dataclass_fields__):raise ValueError('exact mechanical state fields')
        n=len(self.node_ids);ne=len(self.elements);shapes=((n,3),(n,3),(n,3,3),(ne,2,3,3),(ne,18));out=[]
        for (key,shape) in zip(Mechanical.__dataclass_fields__,shapes):
            raw=np.asarray(data[key],dtype=object)
            if raw.shape!=shape or (decoded and any(type(v) is not float for v in raw.flat)):
                raise ValueError('strict mechanical shape/binary64 values')
            a=_owned(data[key])
            if not np.isfinite(a).all():raise ValueError('finite mechanical state')
            out.append(a)
        state=Mechanical(*out)
        for h,l in zip(state.positions.flat,state.position_low.flat):validate_pair(float(h),float(l))
        _frames(state.nodal_frames,n,'retained nodal frames')
        _frames(state.cell_rotations.reshape(-1,3,3),2*ne,'retained cell frames')
        for dof in self.fixed:
            node,axis=divmod(dof,6)
            if axis<3:
                if state.positions[node,axis]!=self.reference_positions[node,axis] or state.position_low[node,axis]!=0.:
                    raise ValueError('fixed retained position changed')
            elif not np.array_equal(state.nodal_frames[node],self.reference_frames[node]):raise ValueError('fixed retained frame changed')
        return state

    def histories(self,values,*,decoded=False):
        if type(values) not in (tuple,list) or len(values)!=len(self.elements):raise ValueError('complete generalized histories')
        return tuple(_history(v if decoded else _load(canonical(v).decode()),e) for v,(_,e) in zip(values,self.elements))

    def nodal_external(self,parameter):
        return np.zeros(self.nodal_count)

    def assemble(self,state,parameter,origins):
        self.guard();state=self.make(state.descriptor());origins=self.histories(origins)
        r=np.zeros(self.count);j=np.zeros((self.count,self.count));responses=[];works=[]
        for i,(eid,e) in enumerate(self.elements):
            nodes=self.nodes[i];op=e.operator
            a=operator_call(op,'evaluate',self.guard,self.started,
                state.positions[nodes],state.position_low[nodes],state.nodal_frames[nodes],
                state.cell_rotations[i],state.resultants[i],origin=origins[i])
            work=line_work(op.reference,state.positions[nodes],state.position_low[nodes],state.cell_rotations[i],
                parameter*self.program.pattern.force(eid),order=op.order)
            rc=a.residual-work.gradient
            local=rc-cell_couple_load(op,parameter*self.program.pattern.density(eid))
            local_j=spatial_jacobian(rc,a.hessian+a.hessian_low-work.hessian)
            r[self.slots[i]]+=local;j[np.ix_(self.slots[i],self.slots[i])]+=local_j
            responses.append(a);works.append(work.value)
        scaled=r/self.scale;metrics=(float(np.linalg.norm(scaled[self.equilibrium])),float(np.linalg.norm(scaled[self.compatibility])))
        self.guard();return r,j,metrics,tuple(responses),tuple(works)

    def step(self,state,r,j):
        from ._native_retained_step import retained_newton_step
        return retained_newton_step(self,state,r,j)

    def advance(self,state,step):
        s={k:v.copy() for k,v in state.descriptor().items()}
        angular=[]
        for node in range(len(self.node_ids)):
            for axis in range(3):
                s['positions'][node,axis],s['position_low'][node,axis]=split_sum((float(state.positions[node,axis]),
                    float(state.position_low[node,axis]),float(step[6*node+axis])))
            v=step[6*node+3:6*node+6];angular.append(v);s['nodal_frames'][node]=rotation(v)@state.nodal_frames[node]
        for i in range(len(self.elements)):
            first=self.nodal_count+24*i
            for cell in (0,1):
                v=step[first+3*cell:first+3*cell+3];angular.append(v)
                s['cell_rotations'][i,cell]=rotation(v)@state.cell_rotations[i,cell]
            s['resultants'][i]+=step[first+6:first+24]
        if max(np.linalg.norm(v) for v in angular)>=.9*np.pi:raise ValueError('retained generalized step requires cutback')
        return self.make(s)

    def _require_issued(self,state):
        self.guard()
        bound=self._issued.get(id(state))
        if (type(state) is not State or bound is None or bound[0] is not state
                or state.model_sha256!=self.identity):raise ValueError('state was not issued by this context')
        if canonical(state)!=bound[1]:raise ValueError('issued state changed')
        return bound[2]

    def recover(self,state):
        self._require_issued(state)
        result=self._recover_validated(state)
        self._require_issued(state)
        return result

    def _recover_validated(self,state):
        if type(state) is not State or state.model_sha256!=self.identity:raise ValueError('recovery model binding')
        self.guard();rows=[]
        for i,(eid,e) in enumerate(self.elements):
            recovered=operator_call(e.operator,'recover',self.guard,self.started,
                state.mechanical.cell_rotations[i],state.mechanical.resultants[i],origin=state.origins[i])
            if canonical([r['history'] for r in recovered])!=canonical(state.histories[i].stations):raise ValueError('recovery history mismatch')
            rows.append(dict(element_id=eid,stations=recovered))
        self.guard();return tuple(rows)

    def record(self,mechanical,origins,cursor,iterations,previous):
        if type(cursor) is not int or not 0<=cursor<=len(self.program.targets):raise ValueError('target cursor')
        if type(iterations) is not int or not 0<=iterations<=self.program.max_iterations:raise ValueError('iteration count')
        mechanical=self.make(mechanical.descriptor());origins=self.histories(origins)
        parameter=0. if cursor==0 else self.program.targets[cursor-1]
        r,j,metrics,responses,work=self.assemble(mechanical,parameter,origins)
        _,correction=self.step(mechanical,r,j)
        if max(*metrics,correction)>1e-11:raise ValueError('only equilibrated compatible converged states may commit')
        histories=self.histories(tuple(a.history for a in responses))
        state=State(mechanical,origins,histories,cursor,self.identity)
        body=dict(target=cursor,parameter=parameter,iterations=iterations,previous_sha256=previous,mechanical=mechanical.descriptor(),
            origins=origins,histories=histories,residual=r,metrics=metrics,correction=correction,work=work,
            material_sha256=sha([a.material.decode() for a in responses]),recovery_sha256=sha(self._recover_validated(state)))
        self.guard();raw=canonical({**body,'record_sha256':sha(body)})
        self._issued[id(state)]=(state,canonical(state),raw)
        self._issued_records.add(raw)
        return state,raw

    def _require_chain(self,records):
        if type(records) is not tuple or len(records)>len(self.program.targets):raise ValueError('bounded issued record chain required')
        previous=_load(self.genesis.decode())
        for index,raw in enumerate(records,1):
            if type(raw) is not bytes or raw not in self._issued_records:raise ValueError('record was not issued by this context')
            row=_load(raw.decode())
            if (row['target']!=index or row['parameter']!=self.program.targets[index-1]
                    or row['previous_sha256']!=previous['record_sha256']
                    or canonical(row['origins'])!=canonical(previous['histories'])):
                raise ValueError('issued record chain is not contiguous')
            previous=row

    def stage(self,mechanical,accepted,records,iterations):
        issued=self._require_issued(accepted);self._require_chain(records)
        if issued!=(records[-1] if records else self.genesis):raise ValueError('issued state differs from predecessor record')
        if type(accepted) is not State or accepted.model_sha256!=self.identity or accepted.completed_targets!=len(records):
            raise ValueError('accepted state and record cursor differ')
        previous=_load((records[-1] if records else self.genesis).decode())['record_sha256']
        return self.record(mechanical,accepted.histories,len(records)+1,iterations,previous)

    def checkpoint(self,records):
        self._require_chain(records)
        body=dict(schema=self.schema,formulation=self.policy,model_sha256=self.identity,program=self.program,
            initial=_load(self.genesis.decode()),records=[_load(r.decode()) for r in records],completed_targets=len(records))
        raw=canonical({**body,'checkpoint_sha256':sha(body)})
        if len(raw)>MAX_BYTES:raise ValueError('checkpoint size limit')
        self.guard();return raw

    def restore(self,raw,*,expected_sha256):
        if type(raw) is not bytes or type(expected_sha256) is not str or sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('external checkpoint authority mismatch')
        value=_load(raw.decode());keys={'schema','formulation','model_sha256','program','initial','records','completed_targets','checkpoint_sha256'}
        if type(value) is not dict or set(value)!=keys:raise ValueError('exact generalized retained checkpoint schema')
        body={k:v for k,v in value.items() if k!='checkpoint_sha256'}
        if (value['schema']!=self.schema or value['formulation']!=self.policy or value['model_sha256']!=self.identity
                or canonical(value['program'])!=self.program_bytes or value['checkpoint_sha256']!=sha(body)
                or canonical(value['initial'])!=self.genesis):raise ValueError('checkpoint model/program/genesis binding')
        cursor=value['completed_targets']
        if type(cursor) is not int or not 0<=cursor<=len(self.program.targets) or type(value['records']) is not list or len(value['records'])!=cursor:
            raise ValueError('checkpoint target extent')
        accepted=self.initial;records=()
        for row in value['records']:
            mechanical=self.make(row['mechanical'],decoded=True)
            proposed,record=self.stage(mechanical,accepted,records,row['iterations'])
            if record!=canonical(row):raise ValueError('accepted geometry/history/work/recovery replay mismatch')
            accepted,records=proposed,(*records,record)
        if self.checkpoint(records)!=raw:raise ValueError('noncanonical accepted chain replay')
        self.guard();return accepted,records
