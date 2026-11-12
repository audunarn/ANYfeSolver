"""Private bounded adaptive driver over the unchanged native arc-step equations."""
from copy import deepcopy
from dataclasses import dataclass,replace
from hashlib import sha256
from time import monotonic
import numpy as np
from . import _ge_beam3_native_arc as arc
from . import _ge_beam3_native_history_profile as capacity
from ._ge_beam3_native_arc_restart import encode_checkpoint as encode_arc,decode_checkpoint as decode_arc
from ._ge_beam3_native_fibre_restart import _keys
from ._ge_beam3_p5_seeded.core import canonical,sha
from ._native_reference_modal import _owned
from .control import SolveCancelled,cancellation_safe_point

POLICY='GE_BEAM3_NATIVE_OBJECTIVE_ADAPTIVE_ARC_V1'
SCHEMA='GE_BEAM3_NATIVE_OBJECTIVE_ADAPTIVE_ARC_RESTART_V1'
REASONS={'native arc line search exhausted':'LINE_SEARCH','native arc iteration bound exhausted':'ITERATION_BOUND'}


@dataclass(frozen=True)
class AdaptiveArcProgram:
    prototype: object
    accepted_steps: int
    minimum_step: float
    maximum_step: float
    max_cutbacks: int=8
    max_attempts: int=128

    def require(self,model):
        if type(self) is not AdaptiveArcProgram or type(self.prototype) is not arc.ArcProgram:
            raise ValueError('exact adaptive arc programme with one initial step required')
        self.prototype.require(model)
        if len(self.prototype.steps)!=1:raise ValueError('one adaptive initial step required')
        if type(self.accepted_steps) is not int or not 1<=self.accepted_steps<=64:
            raise ValueError('bounded adaptive accepted-step count')
        arc.scalar(self.minimum_step);arc.scalar(self.maximum_step)
        if not 0.<self.minimum_step<=self.prototype.steps[0]<=self.maximum_step<=.25:
            raise ValueError('ordered positive adaptive step bounds')
        if type(self.max_cutbacks) is not int or not 0<=self.max_cutbacks<=8:
            raise ValueError('bounded adaptive cutback count')
        if type(self.max_attempts) is not int or not self.accepted_steps<=self.max_attempts<=128:
            raise ValueError('bounded adaptive attempt count')

    def descriptor(self):
        return dict(policy=POLICY,prototype=self.prototype.descriptor(),accepted_steps=self.accepted_steps,
            minimum_step=self.minimum_step,maximum_step=self.maximum_step,max_cutbacks=self.max_cutbacks,
            max_attempts=self.max_attempts,growth='DOUBLE_AT_MOST_THREE_ITERATIONS_WITHOUT_CUTBACK',
            shrink='HALVE_AT_LEAST_SEVEN_ITERATIONS',rejection='TYPED_CONVERGENCE_ONLY_HALF_STEP',
            rejection_replay='AUTHENTICATED_SOLVER_DIAGNOSTIC_NOT_REEXECUTED',
            production_qualified=False)


def next_after_accept(program,step,iterations,cutbacks):
    if iterations<=3 and cutbacks==0:return min(program.maximum_step,step*2.)
    if iterations>=7:return max(program.minimum_step,step*.5)
    return step


def replay_policy(program,inner,attempts):
    """Check the deterministic policy trace and its links to accepted wire states."""
    if type(attempts) not in (tuple,list) or len(attempts)>program.max_attempts:
        raise ValueError('bounded adaptive trace')
    if type(inner) is not dict or type(inner.get('records')) is not list or type(inner.get('accepted_chain')) is not dict:
        raise ValueError('adaptive inner history schema')
    rows=inner['records'];snapshots=inner['accepted_chain'].get('snapshots')
    if type(snapshots) is not list or not 1<=len(snapshots)<=65:raise ValueError('adaptive snapshot schema')
    origin=len(snapshots)-len(rows)-1
    if origin<0 or len(snapshots)>65 or origin+1+program.accepted_steps>65:
        raise ValueError('bounded complete adaptive source/history')
    step=program.prototype.steps[0];cursor=cuts=0;terminal=False
    accepted=[]
    for i,a in enumerate(attempts,1):
        _keys(a,('attempt','step_index','step_size','origin_sha256','disposition','reason','iterations'))
        if terminal or cursor>=program.accepted_steps or type(a['attempt']) is not int or a['attempt']!=i:
            raise ValueError('adaptive attempt order/terminal mismatch')
        if type(a['step_index']) is not int or a['step_index']!=cursor+1 or arc.scalar(a['step_size'])!=step:
            raise ValueError('adaptive step schedule mismatch')
        if a['origin_sha256']!=sha(snapshots[origin+cursor]):
            raise ValueError('adaptive accepted-origin mismatch')
        if a['disposition']=='ACCEPTED':
            if a['reason'] is not None or type(a['iterations']) is not int or cursor>=len(rows):
                raise ValueError('adaptive acceptance schema')
            row=rows[cursor]
            if a['iterations']!=row['iterations'] or step!=row['step_size']:
                raise ValueError('adaptive acceptance/inner record mismatch')
            accepted.append(step);cursor+=1
            step=next_after_accept(program,step,a['iterations'],cuts);cuts=0
        elif a['disposition'] in ('CUTBACK','EXHAUSTED'):
            if a['reason'] not in REASONS.values() or a['iterations'] is not None:
                raise ValueError('adaptive typed rejection schema')
            may_cut=cuts<program.max_cutbacks and step*.5>=program.minimum_step and i<program.max_attempts
            if (a['disposition']=='CUTBACK')!=may_cut:raise ValueError('adaptive cutback bound mismatch')
            if may_cut:step*=.5;cuts+=1
            else:terminal=True
        else:raise ValueError('adaptive disposition enum')
    if cursor!=len(rows):raise ValueError('adaptive complete acceptance trace required')
    if len(attempts)==program.max_attempts and cursor<program.accepted_steps:terminal=True
    return tuple(accepted),step,cuts,terminal


def wrap_checkpoint(model,program,inner_raw,attempts):
    program.require(model);profile=program.prototype.history_profile
    inner=capacity.load(inner_raw,profile);steps,next_step,cuts,terminal=replay_policy(program,inner,attempts)
    fixed=replace(program.prototype,steps=steps or program.prototype.steps)
    if canonical(inner['program'])!=canonical(fixed.descriptor()) or inner['program_sha256']!=sha(fixed.descriptor()):
        raise ValueError('adaptive inner programme mismatch')
    body=dict(schema=capacity.schema(SCHEMA,profile),model_sha256=arc.model_identity(model),
        program=program.descriptor(),program_sha256=sha(program.descriptor()),attempts=attempts,
        next_step=next_step,cutbacks=cuts,terminal=terminal,inner_checkpoint=inner,production_qualified=False,
        **capacity.binding(profile))
    raw=canonical({**body,'checkpoint_sha256':sha(body)})
    if len(raw)>capacity.limit(profile):raise ValueError('adaptive checkpoint byte limit')
    return raw


def decode_checkpoint(model,program,raw,*,expected_sha256):
    if type(raw) is not bytes or type(expected_sha256) is not str or sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('adaptive external checkpoint SHA-256 mismatch')
    program.require(model);profile=program.prototype.history_profile;v=capacity.load(raw,profile)
    capacity.envelope(v,('schema','model_sha256','program','program_sha256','attempts','next_step','cutbacks',
        'terminal','inner_checkpoint','production_qualified','checkpoint_sha256'),SCHEMA,profile)
    if v['checkpoint_sha256']!=sha({k:x for k,x in v.items() if k!='checkpoint_sha256'}):
        raise ValueError('adaptive checkpoint self hash mismatch')
    if v['production_qualified'] is not False or v['model_sha256']!=arc.model_identity(model):
        raise ValueError('adaptive model/qualification mismatch')
    if v['program_sha256']!=sha(program.descriptor()) or canonical(v['program'])!=canonical(program.descriptor()):
        raise ValueError('adaptive programme mismatch')
    steps,next_step,cuts,terminal=replay_policy(program,v['inner_checkpoint'],v['attempts'])
    if type(v['terminal']) is not bool or type(v['cutbacks']) is not int or v['terminal']!=terminal or v['cutbacks']!=cuts or arc.scalar(v['next_step'])!=next_step:
        raise ValueError('adaptive cursor/policy mismatch')
    fixed=replace(program.prototype,steps=steps or program.prototype.steps)
    inner=canonical(v['inner_checkpoint'])
    chain,records=decode_arc(model,fixed,inner,expected_sha256=sha256(inner).hexdigest())
    attempts=tuple(v['attempts'])
    if wrap_checkpoint(model,program,inner,attempts)!=raw:raise ValueError('adaptive canonical roundtrip mismatch')
    return chain,records,attempts,inner


@dataclass(frozen=True)
class AdaptiveArcResult:
    status: str
    completed_steps: int
    completed_attempts: int
    parameter: float
    displacements: np.ndarray
    physical_imbalance: np.ndarray
    checkpoint: bytes
    failure: object
    events: tuple
    production_qualified: bool=False


def _solve(model,program,*,checkpoint=None,expected_sha256=None,stop_after=None,stop_after_attempts=None,progress=None,cancellation_token=None):
    started=monotonic();program.require(model);prototype=program.prototype
    elements,n,free,identity,maps,metric,previous=arc.capture(model,prototype)
    end=program.accepted_steps if stop_after is None else stop_after
    attempts_end=program.max_attempts if stop_after_attempts is None else stop_after_attempts
    if type(end) is not int or not 0<=end<=program.accepted_steps or type(attempts_end) is not int or not 0<=attempts_end<=program.max_attempts:
        raise ValueError('bounded adaptive stop cursor')
    if progress is not None and not callable(progress):raise ValueError('callable adaptive observer required')
    fingerprint=sha(program.descriptor());events=[]
    def safe(stage):
        cancellation_safe_point(cancellation_token,stage);program.require(model)
        if sha(program.descriptor())!=fingerprint or arc.model_identity(model)!=identity:raise ValueError('adaptive programme/model changed')
        if monotonic()-started>600.:raise RuntimeError('adaptive programme deadline')
    def observe(stage,index,iteration=0,**kwargs):
        safe(stage);event=dict(stage=stage,step=index,iteration=iteration,attempt=active_attempt,**kwargs);events.append(event)
        if progress is not None:
            try:progress(deepcopy(event))
            except SolveCancelled:raise
            except Exception as error:raise RuntimeError('adaptive observer failed: '+str(error)) from error
        safe(stage)
    if checkpoint is not None:
        chain,records,attempts,inner=decode_checkpoint(model,program,checkpoint,expected_sha256=expected_sha256)
        capsule=checkpoint
    else:
        if expected_sha256 is not None:raise ValueError('adaptive hash without checkpoint')
        if prototype.source is None:
            total=np.zeros(n);states={i:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for i,e in elements}
            chain=(dict(load_point=arc.load_point(prototype,0.,genesis=True),displacements=total.copy(),states=states),)
        else:
            from ._ge_beam3_native_arc_source import load_source
            chain,_,_=load_source(model,prototype.source)
        records=attempts=();inner=encode_arc(model,prototype,chain,records)
        capsule=wrap_checkpoint(model,program,inner,attempts)
    _,step,cuts,terminal=replay_policy(program,capacity.load(inner,prototype.history_profile),attempts)
    if terminal:raise ValueError('exhausted adaptive checkpoint cannot be resumed')
    cursor=len(records)
    if cursor>end or len(attempts)>attempts_end:raise ValueError('adaptive cannot rewind accepted/attempt cursor')
    total=chain[-1]['displacements'].copy();states=chain[-1]['states']
    if records:parameter=records[-1]['parameter'];previous=records[-1]['direction'].copy()
    elif prototype.source is None:parameter=0.
    else:
        from ._ge_beam3_native_arc_source import load_source,secant_orientation
        prefix,parameter,old_parameter=load_source(model,prototype.source)
        previous=secant_orientation(prefix,parameter,old_parameter,metric,prototype.source.forward_sign)
    reaction=arc.physical(model,states,arc.effective(prototype,parameter)[1])
    store=arc.make_store(model,states,total);status='paused';failure=None;active_attempt=len(attempts)
    def evaluate(u,p,index,it):
        observe('before_assembly',index,it);value=arc.assemble(model,store,u,prototype,p);safe('after_assembly');return value
    try:
        observe('initialized',cursor)
        while cursor<end and len(attempts)<attempts_end:
            active_attempt=len(attempts)+1
            origin=sha(capacity.load(inner,prototype.history_profile)['accepted_chain']['snapshots'][-1])
            committed=sha(store.materialize())
            base=dict(attempt=len(attempts)+1,step_index=cursor+1,step_size=step,origin_sha256=origin)
            try:
                outcome=arc.attempt_step(model,prototype,store,total,parameter,previous,cursor+1,step,maps,metric,free,evaluate,observe)
            except arc.ArcConvergenceFailure as error:
                if type(error) is not arc.ArcConvergenceFailure or str(error) not in REASONS:raise
                if store.has_active_trial:store.discard_trial(store.active_trial_token())
                if sha(store.materialize())!=committed:raise ValueError('adaptive rejection changed accepted state')
                safe('rejected')
                may_cut=cuts<program.max_cutbacks and step*.5>=program.minimum_step and len(attempts)+1<program.max_attempts
                row=dict(**base,disposition='CUTBACK' if may_cut else 'EXHAUSTED',reason=REASONS[str(error)],iterations=None)
                staged_attempts=attempts+(row,);staged=wrap_checkpoint(model,program,inner,staged_attempts)
                observe('before_rejection_record',cursor+1)
                attempts,capsule=staged_attempts,staged
                if not may_cut:
                    status,failure='failed','adaptive convergence budget exhausted';break
                step*=.5;cuts+=1;observe('cutback',cursor+1,next_step=step);continue
            candidate,p,trial,imbalance,tangent,iteration,residual=outcome
            next_records=records+(dict(index=cursor+1,step_size=step,parameter=p,iterations=iteration,direction=tangent.copy(),arc_residual=residual),)
            next_chain=chain+(dict(load_point=arc.load_point(prototype,p),displacements=candidate.copy(),states=deepcopy(dict(trial))),)
            fixed=replace(prototype,steps=tuple(r['step_size'] for r in next_records))
            staged_inner=encode_arc(model,fixed,next_chain,next_records)
            next_attempts=attempts+(dict(**base,disposition='ACCEPTED',reason=None,iterations=iteration),)
            staged=wrap_checkpoint(model,program,staged_inner,next_attempts)
            observe('before_commit',cursor+1,iteration)
            coordinates=arc.native_trial_full_coordinates(store,model,candidate)
            store.commit(store.active_trial_token(),accepted_full_displacement=candidate,accepted_full_coordinates=coordinates)
            inner,capsule,chain,records,attempts=staged_inner,staged,next_chain,next_records,next_attempts
            total,reaction,parameter,cursor=candidate.copy(),imbalance.copy(),p,cursor+1
            previous=tangent.copy();step=next_after_accept(program,step,iteration,cuts);cuts=0
            observe('committed',cursor,iteration,next_step=step)
        if cursor==program.accepted_steps:status='completed'
        elif len(attempts)==program.max_attempts:status,failure='failed','adaptive total attempt budget exhausted'
    except SolveCancelled as error:status,failure='cancelled',str(error)
    except Exception as error:status,failure='failed',type(error).__name__+': '+str(error)
    finally:
        if store.has_active_trial:store.discard_trial(store.active_trial_token())
    return AdaptiveArcResult(status,cursor,len(attempts),parameter,_owned(total),_owned(reaction),capsule,failure,tuple(events))


def solve_adaptive_arc(model,program,**kwargs):
    if any(v is not None for v in (arc._ARC.get(),arc._PATH.get(),arc._PROGRAM.get(),arc._RUN.get(),arc._ACTIVE.get())):
        raise ValueError('nested adaptive arc programme forbidden')
    token=arc._ARC.set(object())
    try:return _solve(model,program,**kwargs)
    finally:arc._ARC.reset(token)
