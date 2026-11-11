"""Connected adapter to shared native midpoint controller; source unverified."""
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
from types import SimpleNamespace
import math
import numpy as np
from .control import cancellation_safe_point
from ._ge_beam3_p5_seeded.core import canonical,sha
from ._ge_beam3_p5_seeded.codec import _load
from ._ge_beam3_connected_dynamic import ConnectedDynamicAnalysis,ConnectedState,MAX_BYTES
from ._ge_beam3_native_transient import NativeTransientError,NativeTransientResult,_settings
from ._native_midpoint_controller import advance

PROFILE='NATIVE_CONNECTED_CONTACT_FREE_MIDPOINT_TRAJECTORY_DEV_V1'
SCHEMA='NATIVE_CONNECTED_CONTACT_FREE_MIDPOINT_TRAJECTORY_CHECKPOINT_DEV_V1'


def _bytes(value):
    raw=canonical(value)
    if len(raw)>MAX_BYTES:raise NativeTransientError('connected trajectory byte bound exceeded')
    return raw


def _vector(value,size,label):
    if value is None:return np.zeros(size)
    if type(value) is not np.ndarray or value.dtype!=np.float64 or value.shape!=(size,) or not np.isfinite(value).all():
        raise NativeTransientError('finite binary64 physical3N+6E '+label+' required')
    return value.copy()


def _record(owner,state,account):
    return _load(_bytes(dict(time=state.time,epoch=state.epoch,state=state,state_sha256=sha(state),
        accounting=account,load_admission='NO_APPLIED_LOADS_IN_CONNECTED_PROFILE')).decode('ascii'))


def _result(identity,contract,controller,native,status):
    profile=contract['settings']['profile']
    contact_profile=profile!=PROFILE
    schema=contract['settings'].get('trajectory_schema',SCHEMA)
    body=dict(schema=schema,identity=identity,contract=contract,native_checkpoint=native.decode('ascii'),
        native_sha256=sha256(native).hexdigest(),controller=controller,production_qualified=False)
    checkpoint=_bytes(dict(body=body,checkpoint_sha256=sha(body)))
    data=dict(profile=profile,identity=identity,status=status,production_qualified=False,contact=contract['settings'].get('contact_result_label','NATIVE_WHOLE_ROUND_LINE_PENALTY') if contact_profile else 'UNSUPPORTED',
        loads='UNSUPPORTED',integrator='EXISTING_NATIVE_SO3_MIDPOINT',outputs=controller['outputs'],
        steps=controller['steps'],initial=controller['initial'],accepted_time=controller['time'],
        accepted_steps=controller['accepted_steps'],final=controller['steps'][-1]['endpoint'] if controller['steps'] else controller['initial'],
        output_cadence=contract['settings']['save_every'],
        work_semantics='ACTUAL_GLOBAL_MIDPOINT_SUPPORT_WORK; HOMOGENEOUS_CLAMP_ZERO',
        energy_semantics='EXISTING_NATIVE_KINETIC_PLUS_RETAINED_MIXED_POTENTIAL; NO_ENERGY_GATE')
    if contact_profile:
        data['energy_semantics']='KINETIC_PLUS_STRUCTURAL_ELASTIC_PLUS_CONTACT_REGULARIZATION_STORAGE; NO_EXACT_ENERGY_CONSERVATION_CLAIM'
    return NativeTransientResult(_bytes(data),checkpoint)


def _support(owner,candidate,increment):
    physical=tuple(i for i in owner.fixed if i in owner.physical)
    algebraic=tuple(i for i in owner.fixed if i in owner.algebraic)
    if algebraic not in ((),(3,4,5)):raise NativeTransientError('only existing full first-node clamp supported')
    force=np.zeros(3);moment=[Fraction(0) for _ in range(3)]
    reaction=np.zeros(owner.size);reaction[list(owner.fixed)]=candidate.residual[list(owner.fixed)]
    for node in range(owner.n):
        f=reaction[6*node:6*node+3];force+=f
        arm=[Fraction(float(h))+Fraction(float(l)) for h,l in zip(candidate.stage.positions[node],candidate.stage.position_low[node])]
        ff=[Fraction(float(x)) for x in f]
        cross=(arm[1]*ff[2]-arm[2]*ff[1],arm[2]*ff[0]-arm[0]*ff[2],arm[0]*ff[1]-arm[1]*ff[0])
        for a in range(3):moment[a]+=cross[a]+Fraction(float(reaction[6*node+3+a]))
    return dict(physical_fixed_indices=physical,algebraic_fixed_indices=algebraic,
        reaction_global=reaction,force_impulse_ns=force*candidate.dt,
        moment_impulse_nms=np.array([float(x) for x in moment])*candidate.dt,
        work_j=float(reaction@increment),moment_origin_high=(0.,0.,0.),moment_origin_low=(0.,0.,0.),
        semantics='ACTUAL_STAGE_TRANSLATION_REACTION_PLUS_CONSTRAINED_FRAME_COUPLE; NO_FRAME_INERTIA')


def _prepare_step(owner,state,t_end,identity,contract,controller,candidate,dt,failures,planned,truncated,settings,token):
    with owner._operation(token):
        owner._candidate(candidate)
        _,_,_,_,_,_,stage_account=owner._evaluate(candidate.stage,token)
        er,_,ei,_,_,_,endpoint_account=owner._evaluate(candidate.endpoint,token)
        increment=np.zeros(owner.size);p=list(owner.p)
        increment[p]=candidate.dt*state.velocity[p]+.5*candidate.dt**2*candidate.unknown[p]
        increment[list(owner.z)]=candidate.unknown[list(owner.z)]
        support=_support(owner,candidate,increment)
        endpoint=_record(owner,candidate.endpoint,endpoint_account)
        stage=_record(owner,candidate.stage,stage_account)
        native=owner._encode_state(candidate.endpoint)
        step=dict(index=controller['accepted_steps']+1,dt=candidate.dt,start_time=state.time,end_time=candidate.endpoint.time,
            truncated=bool(truncated and not failures),cutbacks=failures,iterations=candidate.iterations,backtracks=candidate.backtracks,
            predecessor_sha256=sha(state),endpoint=endpoint,stage=stage,
            stage_residual_norm=owner._norm(candidate.residual,owner.free),endpoint_algebraic_norm=owner._norm(er,owner.z),
            endpoint_reaction_global=er+ei,midpoint_unknown_global=candidate.unknown,
            midpoint_physical_increment_global=increment,midpoint_support=support)
        step.update(owner._step_extra(state,candidate.unknown,candidate.dt,candidate.stage,candidate.endpoint,token))
        step=_load(_bytes(step).decode('ascii'));count=controller['accepted_steps']+1
        next_controller=dict(time=candidate.endpoint.time,accepted_steps=count,next_dt=dt if failures else planned,
            initial=controller['initial'],steps=controller['steps']+[step],outputs=controller['outputs']+
            ([endpoint] if count%settings['save_every']==0 else []))
        result=_result(identity,contract,next_controller,native,'completed' if candidate.endpoint.time==t_end else 'running')
        owner._candidate(candidate);owner._guard(token,'connected.trajectory.prepared')
        return next_controller,native,result


def _keys(value,keys):
    if type(value) is not dict or set(value)!=set(keys):raise NativeTransientError('exact connected trajectory schema required')


def _replay_planned_dt(step,planned,settings):
    """Replay the producer's adaptive policy, without changing native mechanics."""
    dt=step['dt'];failures=step['cutbacks']
    if type(dt) is not float or not math.isfinite(dt) or not 0.<dt<=planned:
        raise NativeTransientError('connected stored step exceeds planned dt')
    if type(step['truncated']) is not bool or type(failures) is not list or len(failures)>settings['max_cutbacks']:
        raise NativeTransientError('bounded exact cutback metadata required')
    retryable=settings.get('retryable',('connected midpoint iteration limit','connected midpoint backtracking failed'))
    for failure in failures:
        _keys(failure,('dt','reason'))
        value=failure['dt']
        if type(value) is not float or not math.isfinite(value) or value<=0. or type(failure['reason']) is not str or failure['reason'] not in retryable:
            raise NativeTransientError('only connected Newton nonconvergence permits cutback')
    if failures:
        if step['truncated'] or failures[0]['dt']>planned or any(b['dt']!=a['dt']*.5 for a,b in zip(failures,failures[1:])):
            raise NativeTransientError('contiguous connected halving inventory required')
        if failures[-1]['dt']*.5!=dt or dt<settings['minimum_dt']:
            raise NativeTransientError('connected cutback step/floor mismatch')
        return dt
    if step['truncated']:
        # A request-horizon truncation retains planned dt. After horizon extension
        # this may be an interior historical row; request t_end is not an identity.
        if dt>=planned:
            raise NativeTransientError('connected final horizon truncation mismatch')
    elif dt!=planned:
        raise NativeTransientError('connected untruncated planned step mismatch')
    return planned


def _validate_record(owner,record,epoch,time,token):
    cancellation_safe_point(token,'connected.trajectory.restore.record')
    _keys(record,('time','epoch','state','state_sha256','accounting','load_admission'))
    if record['epoch']!=epoch or type(record['epoch']) is not int or record['time']!=time or type(record['time']) is not float:
        raise NativeTransientError('connected record endpoint/time inventory mismatch')
    state=owner._state(record['state'])
    if state.epoch!=epoch or state.time!=time or record['state_sha256']!=sha(state) or canonical(owner._data(state))!=canonical(record['state']):
        raise NativeTransientError('exact connected saved-state correspondence required')
    if record['load_admission']!='NO_APPLIED_LOADS_IN_CONNECTED_PROFILE':raise NativeTransientError('connected load profile changed')
    r,j,fi,m,_,_,account=owner._evaluate(state,token)
    owner._block_admission(j,m)
    if canonical(account)!=canonical(record['accounting']):raise NativeTransientError('connected saved accounting mismatch')
    return state


def _restore(owner,raw,expected,identity,contract,token):
    if type(raw) is not bytes or len(raw)>MAX_BYTES or sha256(raw).hexdigest()!=expected:raise NativeTransientError('bounded trusted connected trajectory required')
    data=_load(raw.decode('ascii'))
    if canonical(data)!=raw:raise NativeTransientError('canonical connected trajectory required')
    _keys(data,('body','checkpoint_sha256'));body=data['body']
    _keys(body,('schema','identity','contract','native_checkpoint','native_sha256','controller','production_qualified'))
    if body['schema']!=contract['settings'].get('trajectory_schema',SCHEMA) or body['identity']!=identity or canonical(body['contract'])!=canonical(contract) or body['production_qualified'] is not False or data['checkpoint_sha256']!=sha(body):
        raise NativeTransientError('connected trajectory identity/integrity mismatch')
    native=body['native_checkpoint'].encode('ascii')
    if len(native)>MAX_BYTES or sha256(native).hexdigest()!=body['native_sha256']:raise NativeTransientError('connected embedded capsule mismatch')
    c=body['controller'];_keys(c,('time','accepted_steps','next_dt','initial','steps','outputs'))
    if type(c['accepted_steps']) is not int or not 0<=c['accepted_steps']<=contract['settings']['max_steps'] or type(c['steps']) is not list or len(c['steps'])!=c['accepted_steps']:
        raise NativeTransientError('connected complete accepted-step inventory required')
    if type(c['time']) is not float or not math.isfinite(c['time']) or type(c['next_dt']) is not float or not contract['settings']['minimum_dt']<=c['next_dt']<=contract['settings']['dt']:
        raise NativeTransientError('connected controller finite time/next_dt required')
    planned=contract['settings']['dt']
    previous=_validate_record(owner,c['initial'],0,0.,token)
    owner._validate_initial_contact(previous)
    # Physical initial fields are bound to the declared reference contract.
    initial_step=np.zeros(owner.size);initial_step[list(owner.physical)]=contract['initial_displacement_physical']
    expected_initial=owner._advance(owner._reference_physical_pose(previous),initial_step)
    if not owner._physical_pose_matches(previous,expected_initial):raise NativeTransientError('connected initial physical contract mismatch')
    if canonical(previous.velocity[list(owner.physical)])!=canonical(contract['initial_velocity_physical']):raise NativeTransientError('connected initial velocity contract mismatch')
    r,_,fi,_,_,_,_=owner._evaluate(previous,token)
    if owner._norm(r,owner.z)>1e-11 or owner._norm(r+fi,owner.p)>1e-11:raise NativeTransientError('connected initial equilibrium mismatch')
    expected_outputs=[c['initial']]
    for index,step in enumerate(c['steps'],1):
        cancellation_safe_point(token,'connected.trajectory.restore.step')
        _keys(step,('index','dt','start_time','end_time','truncated','cutbacks','iterations','backtracks','predecessor_sha256','endpoint','stage','stage_residual_norm','endpoint_algebraic_norm','endpoint_reaction_global','midpoint_unknown_global','midpoint_physical_increment_global','midpoint_support')+owner.extra_step_keys)
        if type(step['dt']) is not float or not math.isfinite(step['dt']) or step['dt']<=0 or step['index']!=index or type(step['index']) is not int or any(type(step[name]) is not float or not math.isfinite(step[name]) for name in ('start_time','end_time')) or step['start_time']!=previous.time or step['end_time']!=previous.time+step['dt'] or step['end_time']<=previous.time or step['predecessor_sha256']!=sha(previous):
            raise NativeTransientError('connected accepted path chain mismatch')
        planned=_replay_planned_dt(step,planned,contract['settings'])
        endpoint=_validate_record(owner,step['endpoint'],index,step['end_time'],token)
        saved_stage=_validate_record(owner,step['stage'],index-1,float(step['start_time']+.5*step['dt']),token)
        for name in ('endpoint_reaction_global','midpoint_unknown_global','midpoint_physical_increment_global'):
            value=np.asarray(step[name])
            if value.shape!=(owner.size,) or not np.isfinite(value).all():raise NativeTransientError('finite complete connected journal vectors required')
        unknown=np.asarray(step['midpoint_unknown_global'],dtype=np.float64)
        if np.any(unknown[list(owner.fixed)]!=0.):
            raise NativeTransientError('fixed connected midpoint unknown must be exactly zero')
        actual_stage,stage_residual,stage_jacobian,_=owner._stage(previous,unknown,step['dt'],token)
        if canonical(actual_stage)!=canonical(saved_stage):raise NativeTransientError('connected represented stage mismatch')
        increment=np.zeros(owner.size);p=list(owner.p)
        increment[p]=step['dt']*previous.velocity[p]+.5*step['dt']**2*unknown[p]
        increment[list(owner.z)]=unknown[list(owner.z)]
        if canonical(increment)!=canonical(step['midpoint_physical_increment_global']):raise NativeTransientError('connected physical increment mismatch')
        physical_increment=increment.copy();physical_increment[list(owner.z)]=0.
        expected_pose=owner._advance(previous,physical_increment)
        if not owner._physical_pose_matches(endpoint,expected_pose):raise NativeTransientError('connected endpoint physical stage mismatch')
        extra=owner._step_extra(previous,unknown,step['dt'],actual_stage,endpoint,token)
        if any(canonical(step[key])!=canonical(value) for key,value in extra.items()):
            raise NativeTransientError('native contact path/work journal mismatch')
        expected_velocity=previous.velocity.copy();expected_velocity[p]+=step['dt']*unknown[p]
        if canonical(endpoint.velocity)!=canonical(expected_velocity):raise NativeTransientError('connected endpoint physical rate mismatch')
        er,_,ei,_,_,_,_=owner._evaluate(endpoint,token)
        if owner._norm(er,owner.z)>1e-11 or owner._norm(er+ei,owner.p)>1e-11:raise NativeTransientError('connected saved endpoint equilibrium mismatch')
        if canonical(er+ei)!=canonical(step['endpoint_reaction_global']):raise NativeTransientError('connected endpoint reaction mismatch')
        candidate=SimpleNamespace(stage=actual_stage,residual=stage_residual,dt=step['dt'])
        if canonical(_support(owner,candidate,increment))!=canonical(step['midpoint_support']):raise NativeTransientError('connected support journal mismatch')
        if owner._norm(stage_residual,owner.free)!=step['stage_residual_norm'] or owner._norm(er,owner.z)!=step['endpoint_algebraic_norm']:raise NativeTransientError('connected replay residual evidence mismatch')
        for name in ('iterations','backtracks'):
            if type(step[name]) is not int or step[name]<0:raise NativeTransientError('exact finite iteration metadata required')
        for name in ('stage_residual_norm','endpoint_algebraic_norm'):
            if type(step[name]) is not float or not math.isfinite(step[name]) or not 0<=step[name]<=1e-11:raise NativeTransientError('accepted complete residual gates required')
        if index%contract['settings']['save_every']==0:expected_outputs.append(step['endpoint'])
        previous=endpoint
    if previous.time!=c['time'] or c['next_dt']!=planned or canonical(c['outputs'])!=canonical(expected_outputs):raise NativeTransientError('connected endpoint/cadence mismatch')
    # Cross-bind the endpoint BEFORE owner restore can publish a new origin.
    native_data=_load(native.decode('ascii'))
    if canonical(native_data['body']['state'])!=canonical(owner._data(previous)):raise NativeTransientError('native/controller endpoint mismatch')
    return c,native


def solve_connected_transient(owner,config, *, checkpoint=None,expected_sha256=None,max_steps=10000,max_cutbacks=8,minimum_dt=None,cancellation_token=None):
    if type(owner) is not ConnectedDynamicAnalysis:raise NativeTransientError('exact connected development owner required')
    return _solve_retained_transient(owner,config,checkpoint=checkpoint,expected_sha256=expected_sha256,
        max_steps=max_steps,max_cutbacks=max_cutbacks,minimum_dt=minimum_dt,cancellation_token=cancellation_token)


def _solve_retained_transient(owner,config, *, checkpoint=None,expected_sha256=None,max_steps=10000,max_cutbacks=8,minimum_dt=None,cancellation_token=None):
    settings=_settings(config,max_steps,max_cutbacks,minimum_dt);settings['profile']=PROFILE
    if type(owner) is not ConnectedDynamicAnalysis:
        settings.update(profile=owner.trajectory_profile,trajectory_schema=owner.trajectory_schema,retryable=owner.retryable)
        if hasattr(owner,'contact_result_label'):settings['contact_result_label']=owner.contact_result_label
    t_end=float(config.t_end);count=len(owner.physical)
    displacement=_vector(config.initial_displacement,count,'initial_displacement');velocity=_vector(config.initial_velocity,count,'initial_velocity')
    contract=dict(analysis_sha256=owner.identity,settings=settings,initial_displacement_physical=displacement,initial_velocity_physical=velocity)
    identity=sha(contract)
    if owner.current is not None:raise NativeTransientError('fresh connected trajectory owner required')
    cancellation_safe_point(cancellation_token,'native_transient.before_initialize')
    if checkpoint is None:
        if expected_sha256 is not None:raise NativeTransientError('expected_sha256 requires checkpoint')
        def initial(state,account,native):
            record=_record(owner,state,account);c=dict(time=state.time,accepted_steps=0,next_dt=settings['dt'],initial=record,steps=[],outputs=[record])
            return c,native,_result(identity,contract,c,native,'completed' if t_end==0. else 'initialized')
        state,(controller,native,published)=owner._prepare_initial(displacement,velocity,cancellation_token=cancellation_token,prepare=initial)
    else:
        def replay():
            controller,native=_restore(owner,checkpoint,expected_sha256,identity,contract,cancellation_token)
            if controller['time']>t_end:raise NativeTransientError('t_end precedes connected prefix')
            return controller,native
        def restored(state,account,raw,controller):
            last=controller['steps'][-1]['endpoint'] if controller['steps'] else controller['initial']
            if canonical(_record(owner,state,account))!=canonical(last):raise NativeTransientError('restored connected accounting/endpoint mismatch')
            return controller,raw,_result(identity,contract,controller,raw,'completed' if state.time==t_end else 'initialized')
        state,(controller,native,published)=owner._prepare_replayed_restore(cancellation_token=cancellation_token,replay=replay,prepare=restored)
    adapter=SimpleNamespace(error=NativeTransientError,retryable=frozenset(owner.retryable),result=_result,prepare_step=_prepare_step)
    return advance(owner,state,t_end,identity,contract,controller,native,cancellation_token,published,adapter)
