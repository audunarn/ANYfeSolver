"""Private elastic continuation with an explicit, mechanically replayed seed.

No element equations change. A seed is an equilibrium initial condition, not
proof of a physical history from rest. Plastic origins/responses fail closed.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
import numpy as np

from . import _ge_beam3_retained_translation_control as base
from ._ge_beam3_retained_nodal_loading import Context as Physical, Program as ForceProgram
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES
from ._native_reference_modal import _owned
from .control import cancellation_safe_point, SolveCancelled

SCHEMA='GE_BEAM3_ELASTIC_SEED_CONTINUATION_CHAIN_V1'
POLICY='CANDIDATE_GE_BEAM3_ELASTIC_SEED_CONTINUATION_V1'
SEED_SCHEMA='GE_BEAM3_ELASTIC_EQUILIBRIUM_SEED_INPUT_V1'
Program=base.Program

def _digest(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def _packet(raw,expected):
    if type(raw) is not bytes or len(raw)>MAX_BYTES or not _digest(expected) or sha256(raw).hexdigest()!=expected:
        raise ValueError('external elastic seed hash/byte authority')
    value=_load(raw.decode())
    keys={'schema','model_sha256','operators','control_node','direction','nodal_forces',
          'mechanical','parameter','displacement','source_sha256'}
    if type(value) is not dict or set(value)!=keys or canonical(value)!=raw or value['schema']!=SEED_SCHEMA:
        raise ValueError('exact canonical elastic seed schema')
    if not _digest(value['source_sha256']) or not _digest(value['model_sha256']):raise ValueError('seed provenance hash')
    if any(type(value[k]) is not float or not isfinite(value[k]) for k in ('parameter','displacement')):
        raise ValueError('finite binary64 seed parameter/control')
    return value

@dataclass(frozen=True)
class State:
    mechanical: object
    origins: tuple
    histories: tuple
    completed_targets: int
    parameter: float
    model_sha256: str

class Context(base.Context):
    def __init__(self,model,program,seed,*,expected_seed_sha256):
        data=_packet(seed,expected_seed_sha256)
        if type(program) is not Program:raise ValueError('exact elastic continuation programme')
        program.__post_init__();self.program=program;self.program_bytes=canonical(program)
        self.physical=Physical(model,ForceProgram((0.,),DistributedPattern(LinePattern(()),()),
            program.nodal_forces,program.max_iterations,program.max_backtracks))
        p=self.physical
        if (data['model_sha256']!=p.model_identity or data['operators']!=[e.operator.identity for _,e in p.elements]
                or type(data['control_node']) is not int or data['control_node']!=program.control_node
                or canonical(data['direction'])!=canonical(program.direction)
                or canonical(data['nodal_forces'])!=canonical(program.nodal_forces)):
            raise ValueError('elastic seed model/operator/control/load binding')
        if program.control_node not in p.index:raise ValueError('seed control node absent')
        self.node=p.index[program.control_node]
        slots=list(model.mesh.dof_manager.get_node_dofs(program.control_node)[:3])
        if slots!=list(range(6*self.node,6*self.node+3)) or any(s in p.fixed for s in slots):
            raise ValueError('free seed physical translation ordering required')
        row=np.zeros(p.count);row[slots]=program.direction
        column=np.zeros(p.count);column[:p.nodal_count]=-p.nodal_external(1.)
        if not np.any(column[p.free]):raise ValueError('nonzero seed free force pattern required')
        self.row,self.column=_owned(row),_owned(column)
        self.map_bytes=canonical(dict(node=self.node,row=self.row,column=self.column))
        self.seed_bytes=seed;self.seed_sha256=expected_seed_sha256;self.seed_target=data['displacement']
        self._virgin_bytes=canonical(p.initial.histories)
        self.identity=sha(dict(schema=SCHEMA,policy=POLICY,physical=p.identity,program=program,
            seed_sha256=expected_seed_sha256,maps=self.map_bytes.decode()))
        self._capture=canonical(dict(identity=self.identity,seed=self.seed_sha256,target=self.seed_target,
            virgin=self._virgin_bytes.decode()))
        self._issued={};self._issued_records=set()
        mechanical=p.make(data['mechanical'],decoded=True)
        self.initial,self.genesis=self._record(mechanical,p.initial.histories,0,data['parameter'],0,self.identity)

    def guard(self):
        super().guard()
        if (sha256(self.seed_bytes).hexdigest()!=self.seed_sha256
                or canonical(self.physical.initial.histories)!=self._virgin_bytes
                or canonical(dict(identity=self.identity,seed=self.seed_sha256,target=self.seed_target,
                    virgin=self._virgin_bytes.decode()))!=self._capture):
            raise ValueError('elastic seed capture changed')

    def assemble(self,mechanical,parameter,origins,target):
        self.guard()
        if canonical(origins)!=self._virgin_bytes:raise ValueError('elastic continuation rejects material history')
        answer=super().assemble(mechanical,parameter,origins,target)
        if canonical(tuple(r.history for r in answer[3]))!=self._virgin_bytes:
            raise ValueError('elastic continuation cannot commit plastic trial history')
        return answer

    def _require_issued(self,state,*,expected_snapshot=None):
        self.guard();bound=self._issued.get(id(state))
        if expected_snapshot is not None and type(expected_snapshot) is not bytes:raise ValueError('exact expected snapshot')
        if type(state) is not State or bound is None or bound[0] is not state or state.model_sha256!=self.identity:
            raise ValueError('elastic continuation state not issued by this context')
        encoded=canonical(state)
        if encoded!=bound[1] or (expected_snapshot is not None and encoded!=expected_snapshot):
            raise ValueError('issued elastic continuation state changed')
        return bound[2]

    def _record(self,mechanical,origins,cursor,parameter,iterations,previous):
        if type(cursor) is not int or not 0<=cursor<=len(self.program.targets):raise ValueError('elastic cursor')
        if type(iterations) is not int or not 0<=iterations<=self.program.max_iterations:raise ValueError('elastic iterations')
        target=self.seed_target if cursor==0 else self.program.targets[cursor-1]
        p=self.physical;mechanical=p.make(mechanical.descriptor());origins=p.histories(origins)
        r,j,metrics,responses,work=self.assemble(mechanical,parameter,origins,target)
        _,_,correction,_=self.step(mechanical,parameter,r,j,target)
        if max(*metrics,correction)>1e-11:raise ValueError('only fully converged elastic states may commit')
        histories=p.histories(tuple(response.history for response in responses))
        state=State(mechanical,origins,histories,cursor,parameter,self.identity)
        body=dict(target=cursor,displacement_target=target,parameter=parameter,iterations=iterations,
            previous_sha256=previous,mechanical=mechanical.descriptor(),origins=origins,histories=histories,
            residual=r,metrics=metrics,correction=correction,work=work,control_value=self.value(mechanical),
            material_sha256=sha([a.material.decode() for a in responses]),recovery_sha256=sha(self._recover_validated(state)))
        self.guard();raw=canonical({**body,'record_sha256':sha(body)})
        self._issued[id(state)]=(state,canonical(state),raw);self._issued_records.add(raw)
        return state,raw

    def checkpoint(self,records):
        self._require_chain(records)
        body=dict(schema=SCHEMA,formulation=POLICY,model_sha256=self.identity,program=self.program,
            seed_sha256=self.seed_sha256,initial=_load(self.genesis.decode()),
            records=[_load(r.decode()) for r in records],completed_targets=len(records),
            physical_loading_path_from_rest=False,elastic_only=True)
        raw=canonical({**body,'checkpoint_sha256':sha(body)})
        if len(raw)>MAX_BYTES:raise ValueError('elastic checkpoint byte bound')
        self.guard();return raw

    def restore(self,raw,*,expected_sha256):
        self.guard()
        if type(raw) is not bytes or len(raw)>MAX_BYTES or not _digest(expected_sha256) or sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('external elastic checkpoint authority')
        value=_load(raw.decode())
        expected=_load(self.checkpoint(()).decode())
        if type(value) is not dict or set(value)!=set(expected):raise ValueError('exact elastic checkpoint schema')
        body={k:v for k,v in value.items() if k!='checkpoint_sha256'}
        if canonical(value)!=raw or value['checkpoint_sha256']!=sha(body):raise ValueError('canonical elastic checkpoint hash')
        for key in set(expected)-{'records','completed_targets','checkpoint_sha256'}:
            if canonical(value[key])!=canonical(expected[key]):raise ValueError('elastic checkpoint seed/program binding')
        cursor=value['completed_targets']
        if type(cursor) is not int or not 0<=cursor<=len(self.program.targets) or type(value['records']) is not list or len(value['records'])!=cursor:
            raise ValueError('elastic checkpoint cursor/extent')
        accepted,records=self.initial,();keys=set(_load(self.genesis.decode()))
        for i,row in enumerate(value['records'],1):
            if type(row) is not dict or set(row)!=keys or type(row['target']) is not int or row['target']!=i:
                raise ValueError('elastic record schema/order')
            mechanical=self.physical.make(row['mechanical'],decoded=True)
            proposed,regenerated=self.stage(mechanical,row['parameter'],accepted,records,row['iterations'])
            if regenerated!=canonical(row):raise ValueError('elastic mechanical/history/work/recovery replay mismatch')
            accepted,records=proposed,(*records,regenerated)
        if self.checkpoint(records)!=raw:raise ValueError('elastic replay bytes differ')
        self.guard();return accepted,records

@dataclass(frozen=True)
class Result:
    status: str
    completed_targets: int
    state: State
    checkpoint: bytes
    failure: str | None
    production_qualified: bool=False
    physical_loading_path_from_rest: bool=False

def solve(model,program,seed,*,expected_seed_sha256,checkpoint=None,expected_sha256=None,
          stop_after=None,cancellation_token=None,progress=None):
    cancellation_safe_point(cancellation_token,'elastic-continuation.start')
    if progress is not None and not callable(progress):raise ValueError('callable elastic observer')
    context=Context(model,program,seed,expected_seed_sha256=expected_seed_sha256);p=context.physical
    end=len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0<=end<=len(program.targets):raise ValueError('bounded elastic stop target')
    if checkpoint is None:
        if expected_sha256 is not None:raise ValueError('hash requires checkpoint')
        accepted,records=context.initial,();capsule=context.checkpoint(records)
    else:
        accepted,records=context.restore(checkpoint,expected_sha256=expected_sha256);capsule=checkpoint
    if accepted.completed_targets>end:raise ValueError('elastic continuation cannot rewind')
    status='completed' if end==len(program.targets) else 'paused';failure=None
    def safe(stage,target,iteration=0):
        cancellation_safe_point(cancellation_token,stage);context.guard()
        if progress is not None:progress(dict(stage=stage,target=target,iteration=iteration))
        context.guard();cancellation_safe_point(cancellation_token,stage)
    try:
        safe('elastic-continuation.initialized',accepted.completed_targets)
        for index in range(accepted.completed_targets,end):
            target=program.targets[index];origins=accepted.histories;parameter=accepted.parameter
            trial=context.project(accepted.mechanical,target)
            for iteration in range(program.max_iterations+1):
                safe('elastic-continuation.before_assembly',index+1,iteration)
                r,j,metrics,_,_=context.assemble(trial,parameter,origins,target)
                safe('elastic-continuation.before_factorization',index+1,iteration)
                step,delta,correction,matrix=context.step(trial,parameter,r,j,target)
                if max(*metrics,correction)<=1e-11:
                    proposed,record=context.stage(trial,parameter,accepted,records,iteration)
                    next_records=(*records,record);staged=context.checkpoint(next_records)
                    safe('elastic-continuation.before_commit',index+1,iteration)
                    accepted,records,capsule=proposed,next_records,staged
                    safe('elastic-continuation.committed',index+1,iteration);break
                if iteration==program.max_iterations:raise RuntimeError('elastic continuation Newton limit')
                angular=[step[6*i+3:6*i+6] for i in range(len(p.node_ids))]
                angular.extend(step[p.nodal_count+24*i+3*c:p.nodal_count+24*i+3*c+3] for i in range(len(p.elements)) for c in (0,1))
                fraction=min(1.,.45*np.pi/max(max(float(np.linalg.norm(v)) for v in angular),np.finfo(float).tiny))
                for cut in range(program.max_backtracks+1):
                    safe('elastic-continuation.before_trial',index+1,iteration)
                    amount=fraction*.5**cut;candidate=context.project(p.advance(trial,step*amount),target)
                    next_parameter=float(parameter+delta*amount)
                    changed,_,_,_,_=context.assemble(candidate,next_parameter,origins,target)
                    natural=np.linalg.solve(matrix,-np.r_[changed[p.free],context.value(candidate)-target])
                    if not np.isfinite(natural).all():raise ValueError('nonfinite elastic natural merit')
                    trial_step=np.zeros(p.count);trial_step[p.free]=natural[:-1]
                    merit=context.correction_norm(trial,parameter,trial_step,float(natural[-1]))
                    if merit<correction:trial,parameter=candidate,next_parameter;break
                else:raise RuntimeError('elastic continuation line search limit')
    except SolveCancelled as error:status,failure='cancelled',str(error)
    except Exception as error:status,failure='failed',type(error).__name__+': '+str(error)
    return Result(status,accepted.completed_targets,accepted,capsule,failure)
