"""Contact-free native midpoint trajectory DEVELOPMENT controller.

This module controls time/output only. Native states, SO(3) stages, physical
inertia, constitutive histories and state authentication remain in the existing
NativeDynamicAnalysis owner. No C7 acceptance or execution authority is granted.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
from fractions import Fraction
import math
import numpy as np

from .control import cancellation_safe_point, SolveCancelled
from .dynamics import TransientConfig
from ._ge_beam3_native_dynamic_analysis import NativeDynamicAnalysis
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES
from ._ge_beam3_centered_finite_inertia import PHYSICAL_DOFS


PROFILE = 'NATIVE_CONTACT_FREE_MIDPOINT_TRAJECTORY_DEV_V1'
SCHEMA = 'NATIVE_CONTACT_FREE_MIDPOINT_TRAJECTORY_CHECKPOINT_DEV_V1'
MAX_STEPS = 10000
MAX_CUTBACKS = 32
_RETRYABLE = frozenset(('midpoint stage iteration limit',
                       'midpoint stage residual backtracking failed'))


class NativeTransientError(ValueError):
    """Controller non-success; ``partial`` is the last published prefix."""
    def __init__(self, message, *, partial=None, native_checkpoint=None, accepted_state=None):
        super().__init__(message)
        self.partial = partial
        self.native_checkpoint = native_checkpoint
        self.accepted_state = accepted_state


@dataclass(frozen=True)
class NativeTransientResult:
    """Immutable canonical byte snapshot; decoded views are detached data."""
    _payload: bytes
    checkpoint: bytes

    def to_dict(self):
        return _load(self._payload.decode('ascii'))

    @property
    def times(self):
        return np.array([row['time'] for row in self.to_dict()['outputs']], dtype=np.float64)

    @property
    def success(self):
        return self.to_dict()['status'] == 'completed'


def _positive(value, label):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise NativeTransientError('finite positive '+label+' required')
    return float(value)


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise NativeTransientError('bounded exact integer '+label+' required')
    return value


def _vector(value, label):
    if value is None:
        return np.zeros(15, dtype=np.float64)
    if type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != (15,) or not np.isfinite(value).all():
        raise NativeTransientError('finite binary64 physical15 '+label+' required')
    return np.array(value, copy=True)


def _settings(config, max_steps, max_cutbacks, minimum_dt):
    if type(config) is not TransientConfig:
        raise NativeTransientError('exact existing TransientConfig required')
    dt = _positive(config.dt, 'dt')
    if type(config.t_end) not in (int, float) or not math.isfinite(config.t_end) or config.t_end < 0:
        raise NativeTransientError('finite nonnegative t_end required')
    # Default Newmark numbers are unused compatibility placeholders. This
    # route is the existing native midpoint scheme, never Newmark or HHT.
    if (config.beta != .25 or config.gamma != .5 or config.hht_alpha != 0.
            or config.rayleigh_alpha != 0. or config.rayleigh_beta != 0.
            or config.output_nodes is not None or config.output_elements is not None
            or config.include_stress_history is not False or config.recovery is not None
            or config.resource_config is not None):
        raise NativeTransientError('native midpoint does not admit Newmark/HHT, damping, subsets, stress/recovery or resource overrides')
    cadence = _integer(config.save_every, 1, MAX_STEPS, 'save_every')
    count = _integer(max_steps, 1, MAX_STEPS, 'max_steps')
    cuts = _integer(max_cutbacks, 0, MAX_CUTBACKS, 'max_cutbacks')
    floor = math.ldexp(dt, -cuts) if minimum_dt is None else _positive(minimum_dt, 'minimum_dt')
    if not 0. < floor <= dt:
        raise NativeTransientError('minimum_dt must lie in (0,dt]')
    return dict(dt=dt, save_every=cadence, max_steps=count, max_cutbacks=cuts,
                minimum_dt=floor, profile=PROFILE,
                output_policy='GENESIS_AND_GLOBAL_ACCEPTED_INDEX_MOD_CADENCE',
                final_time_policy='EXACT_REQUESTED_END_NO_UNSOLVED_REMAINDER',
                cutback_policy='HALVE_ONLY_NATIVE_NEWTON_NONCONVERGENCE')


def _record(state, accounting, load):
    fields = asdict(accounting)
    fields['rank_mass_checks'] = _load(fields['rank_mass_checks'].decode('ascii'))
    load_fields = asdict(load)
    del load_fields['external_spatial_jacobian42']
    return _load(_check_size(dict(time=state.time, epoch=state.epoch, state=state,
                state_sha256=sha(state), accounting=fields, physical_load=load_fields)).decode('ascii'))


def _check_size(value):
    raw = canonical(value)
    if len(raw) > MAX_BYTES:
        raise NativeTransientError('native trajectory byte bound exceeded')
    return raw


def _envelope(identity, contract, native, controller):
    body = dict(schema=SCHEMA, identity=identity, contract=contract,
                native_checkpoint=native.decode('ascii'),
                native_sha256=sha256(native).hexdigest(), controller=controller,
                production_qualified=False)
    return _check_size(dict(body, checkpoint_sha256=sha(body)))


def _result(identity, contract, controller, native, status):
    capsule = _envelope(identity, contract, native, controller)
    data = dict(profile=PROFILE, identity=identity, status=status,
                production_qualified=False, contact='UNSUPPORTED',
                integrator='EXISTING_NATIVE_SO3_MIDPOINT',
                outputs=controller['outputs'], steps=controller['steps'],
                initial=controller['initial'], accepted_time=controller['time'],
                accepted_steps=controller['accepted_steps'],
                final=(controller['steps'][-1]['endpoint'] if controller['steps'] else controller['initial']),
                output_cadence=contract['settings']['save_every'],
                work_semantics='MIDPOINT_PHYSICAL_LOAD_DOT_PHYSICAL_INCREMENT; NOT_EXACT_TIME_INTEGRAL',
                energy_semantics='NATIVE_ENDPOINT_KINETIC_PLUS_MIXED_POTENTIAL; NO_ENERGY_ACCEPTANCE_GATE')
    return NativeTransientResult(_check_size(data), capsule)


def solve_native_transient(analysis, config, *, checkpoint=None,
                           expected_sha256=None, max_steps=MAX_STEPS,
                           max_cutbacks=8, minimum_dt=None,
                           cancellation_token=None):
    """Advance one fresh native owner using its physical midpoint equations.

    ``initial_displacement`` is a physical15 *reference increment*: nine nodal
    translations followed by two spatial cell exponential-rotation vectors.
    ``initial_velocity`` has the owner's physical15 order. Nodal frame/history
    inputs are never accepted. Restart requires a fresh owner and trusted full
    outer-byte digest; t_end may extend the same time-step/output contract.
    No progress callback, energy rejection, hidden inertia or solver fallback.
    """
    if type(analysis) is not NativeDynamicAnalysis:
        raise NativeTransientError('exact native dynamic development analysis required')
    settings = _settings(config, max_steps, max_cutbacks, minimum_dt)
    t_end = float(config.t_end)
    displacement = _vector(config.initial_displacement, 'initial_displacement')
    velocity = _vector(config.initial_velocity, 'initial_velocity')
    contract = dict(analysis_sha256=analysis.identity, settings=settings,
                    initial_displacement15=displacement,
                    initial_velocity15=velocity)
    identity = sha(contract)
    cancellation_safe_point(cancellation_token, 'native_transient.before_initialize')
    if analysis.current is not None:
        raise NativeTransientError('fresh never-initialized native owner required')
    if checkpoint is None:
        if expected_sha256 is not None:
            raise NativeTransientError('expected_sha256 requires checkpoint')
        from ._ge_beam3_p5.algebra import rotation
        from ._ge_beam3_p5.compensated_coordinates import split_sum
        reference = analysis._owner()._reference_mechanical
        high = np.empty((3, 3)); low = np.empty((3, 3))
        for node in range(3):
            for axis in range(3):
                high[node, axis], low[node, axis] = split_sum((
                    float(reference.positions[node, axis]),
                    float(reference.position_low[node, axis]),
                    float(displacement[3*node+axis])))
        cells = np.array([rotation(displacement[9+3*c:12+3*c])@reference.cell_rotations[0,c]
                          for c in range(2)])
        def prepare_initial(state, accounting, load, native):
            initial = _record(state, accounting, load)
            controller = dict(time=state.time, accepted_steps=0, next_dt=settings['dt'],
                              initial=initial, outputs=[initial], steps=[])
            result = _result(identity, contract, controller, native,
                             'completed' if t_end == 0. else 'initialized')
            return controller, native, result
        state, prepared = analysis._prepare_initial(high, cells, velocity, position_low=low,
            cancellation_token=cancellation_token, prepare=prepare_initial)
        controller, native, published = prepared
    else:
        controller, native = _restore(checkpoint, expected_sha256, identity, contract)
        if controller['time'] > t_end:
            raise NativeTransientError('t_end precedes committed native checkpoint')
        def prepare_restored(state, accounting, load, raw):
            _validate_restored_endpoint(state, controller, accounting, load)
            return _result(identity, contract, controller, raw,
                           'completed' if state.time == t_end else 'initialized')
        state, published = analysis._prepare_restore(native,
            expected_sha256=sha256(native).hexdigest(),
            cancellation_token=cancellation_token, prepare=prepare_restored)
    # Detailed stepping is kept below so both actual-owner tests and cheap
    # controller oracles exercise one controller, without a second integrator.
    return _advance(analysis, state, t_end, identity, contract, controller,
                    native, cancellation_token, published)


def _advance(analysis, state, t_end, identity, contract, controller, native, token, published):
    from types import SimpleNamespace
    from ._native_midpoint_controller import advance
    adapter = SimpleNamespace(error=NativeTransientError, retryable=_RETRYABLE,
                              result=_result, prepare_step=_prepare_step)
    return advance(analysis, state, t_end, identity, contract, controller, native,
                   token, published, adapter)


def _prepare_step(analysis, state, t_end, identity, contract, controller, candidate,
                  dt, failures, planned, truncated, settings, token):
    audit = analysis.candidate_stage_audit(candidate, cancellation_token=token)
    account = analysis.candidate_endpoint_accounting(candidate, cancellation_token=token)
    endpoint_load = analysis.physical_load_accounting(candidate.endpoint.mechanical)
    stage_load = analysis.physical_load_accounting(candidate.stage.mechanical)
    support = _support_journal(analysis, candidate, stage_load, audit.increment42)
    next_native = analysis.prepare_candidate_checkpoint(candidate, cancellation_token=token)
    endpoint = _record(candidate.endpoint, account, endpoint_load)
    work = float(np.dot(stage_load.external_load42, audit.increment42))
    if not math.isfinite(work):
        raise NativeTransientError('nonfinite midpoint load work journal')
    step = dict(index=controller['accepted_steps']+1, dt=candidate.dt,
        start_time=state.time, end_time=candidate.endpoint.time,
        truncated=bool(truncated and not failures), cutbacks=failures,
        iterations=candidate.iterations, backtracks=candidate.backtracks,
        stage_residual_norm=candidate.stage_residual_norm,
        endpoint_algebraic_norm=candidate.endpoint_algebraic_norm,
        endpoint_reaction42=candidate.endpoint_reaction42,
        predecessor_sha256=sha(state), endpoint=endpoint,
        stage_material_sha256=candidate.stage.material_sha256,
        midpoint_physical_increment42=audit.increment42,
        midpoint_external_load42=stage_load.external_load42,
        midpoint_load_work_j=work,
        midpoint_force_impulse_ns=stage_load.force*candidate.dt,
        midpoint_moment_impulse_nms=stage_load.moment_about_origin*candidate.dt,
        midpoint_support=support)
    step = _load(_check_size(step).decode('ascii'))
    next_count = controller['accepted_steps']+1
    next_controller = dict(time=candidate.endpoint.time, accepted_steps=next_count,
        next_dt=(dt if failures else planned), initial=controller['initial'],
        steps=controller['steps']+[step], outputs=controller['outputs']+
            ([endpoint] if next_count % settings['save_every'] == 0 else []))
    prepared = _result(identity, contract, next_controller, next_native,
        'completed' if candidate.endpoint.time == t_end else 'running')
    return next_controller, next_native, prepared


def _support_journal(analysis, candidate, load, increment42):
    """Actual stage support wrench; algebraic frame couple is not inertia.

    The residual sign is the existing owner reaction convention. The genuine
    constrained first-node frame couple contributes support moment, despite
    zero frame inertia. Free algebraics/resultants never enter that wrench.
    """
    fixed = tuple(int(i) for i in analysis._owner().fixed)
    physical = tuple(i for i in fixed if i in PHYSICAL_DOFS)
    algebraic = tuple(i for i in fixed if i not in PHYSICAL_DOFS)
    if algebraic not in ((), (3, 4, 5)):
        raise NativeTransientError('only the admitted fixed first-node frame couple is supported')
    reaction = np.zeros(42)
    reaction[list(physical)] = candidate.stage.residual[list(physical)]
    force = np.sum([reaction[6*n:6*n+3] for n in range(3)], axis=0)
    moment = [Fraction(0) for _ in range(3)]
    mechanical = candidate.stage.mechanical
    origin = [Fraction(float(h))+Fraction(float(l)) for h,l in zip(
        load.momentum_origin_high, load.momentum_origin_low)]
    for node in range(3):
        lever = [Fraction(float(h))+Fraction(float(l))-o for h,l,o in zip(
            mechanical.positions[node], mechanical.position_low[node], origin)]
        f = [Fraction(float(x)) for x in reaction[6*node:6*node+3]]
        cross = (lever[1]*f[2]-lever[2]*f[1], lever[2]*f[0]-lever[0]*f[2],
                 lever[0]*f[1]-lever[1]*f[0])
        for axis in range(3):
            moment[axis] += cross[axis]
    for cell in range(2):
        for axis in range(3):
            moment[axis] += Fraction(float(reaction[18+3*cell+axis]))
    algebraic_reaction = np.zeros(42)
    algebraic_reaction[list(algebraic)] = candidate.stage.residual[list(algebraic)]
    for axis in range(3):
        moment[axis] += Fraction(float(algebraic_reaction[3+axis]))
    return dict(physical_fixed_indices=physical, algebraic_fixed_indices=algebraic,
        physical_reaction42=reaction, algebraic_reaction42=algebraic_reaction,
        physical_force_impulse_ns=force*candidate.dt,
        physical_moment_impulse_nms=np.array([float(v) for v in moment])*candidate.dt,
        physical_work_j=float(np.dot(reaction, increment42)),
        algebraic_work_j=float(np.dot(algebraic_reaction, increment42)),
        clamped_total_angular_momentum_closure='STAGE_PHYSICAL_FORCE_PLUS_FIXED_FRAME_REACTION_COUPLE')


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise NativeTransientError('exact native trajectory keys required')


def _restore(raw, expected_sha256, identity, contract):
    if (type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES
            or type(expected_sha256) is not str or len(expected_sha256) != 64
            or sha256(raw).hexdigest() != expected_sha256):
        raise NativeTransientError('bounded checkpoint/trusted external SHA256 required')
    value = _load(raw.decode('ascii'))
    _keys(value, ('schema', 'identity', 'contract', 'native_checkpoint',
                  'native_sha256', 'controller', 'production_qualified', 'checkpoint_sha256'))
    body = {key:item for key,item in value.items() if key != 'checkpoint_sha256'}
    if (value['schema'] != SCHEMA or value['production_qualified'] is not False
            or value['identity'] != identity or canonical(value['contract']) != canonical(contract)
            or value['checkpoint_sha256'] != sha(body)):
        raise NativeTransientError('native trajectory schema/identity/integrity mismatch')
    if type(value['native_checkpoint']) is not str:
        raise NativeTransientError('native checkpoint ASCII payload required')
    native = value['native_checkpoint'].encode('ascii')
    if value['native_sha256'] != sha256(native).hexdigest():
        raise NativeTransientError('native checkpoint digest mismatch')
    controller = value['controller']
    _keys(controller, ('time', 'accepted_steps', 'next_dt', 'initial', 'outputs', 'steps'))
    count = _integer(controller['accepted_steps'], 0, contract['settings']['max_steps'], 'accepted_steps')
    if type(controller['steps']) is not list or len(controller['steps']) != count:
        raise NativeTransientError('complete controller step inventory required')
    planned = contract['settings']['dt']
    record = controller['initial']
    _validate_record(record, 0, 0.)
    output = [record]
    for index, step in enumerate(controller['steps'], 1):
        _keys(step, ('index', 'dt', 'start_time', 'end_time', 'truncated', 'cutbacks',
            'iterations', 'backtracks', 'stage_residual_norm', 'endpoint_algebraic_norm',
            'endpoint_reaction42', 'predecessor_sha256', 'endpoint', 'stage_material_sha256',
            'midpoint_physical_increment42', 'midpoint_external_load42', 'midpoint_load_work_j',
            'midpoint_force_impulse_ns', 'midpoint_moment_impulse_nms', 'midpoint_support'))
        if type(step['index']) is not int or step['index'] != index or type(step['truncated']) is not bool:
            raise NativeTransientError('exact sequential native step index/flag required')
        dt = _positive(step['dt'], 'stored dt')
        if (dt > planned or step['start_time'] != record['time']
                or step['end_time'] != record['time']+dt
                or step['predecessor_sha256'] != record['state_sha256']):
            raise NativeTransientError('native trajectory time/lineage mismatch')
        failures = step['cutbacks']
        if type(failures) is not list or len(failures) > contract['settings']['max_cutbacks']:
            raise NativeTransientError('bounded cutback inventory required')
        for failure in failures:
            _keys(failure, ('dt', 'reason'))
            _positive(failure['dt'], 'failed dt')
            if failure['reason'] not in _RETRYABLE:
                raise NativeTransientError('only native Newton nonconvergence permits cutback')
        if failures and (failures[0]['dt'] > planned or any(
                current['dt'] != previous['dt']*.5
                for previous, current in zip(failures, failures[1:]))):
            raise NativeTransientError('contiguous halving cutback inventory required')
        if failures:
            if failures[-1]['dt']*.5 != dt or dt < contract['settings']['minimum_dt']:
                raise NativeTransientError('cutback step/floor mismatch')
            planned = dt
        elif not step['truncated'] and dt != planned:
            raise NativeTransientError('untruncated native step mismatch')
        for name in ('iterations', 'backtracks'):
            _integer(step[name], 0, 2**31-1, name)
        for name in ('stage_residual_norm', 'endpoint_algebraic_norm'):
            if type(step[name]) is not float or not math.isfinite(step[name]) or step[name] < 0:
                raise NativeTransientError('finite nonnegative native residual required')
        for name, shape in (('endpoint_reaction42',(42,)), ('midpoint_physical_increment42',(42,)),
                ('midpoint_external_load42',(42,)), ('midpoint_force_impulse_ns',(3,)),
                ('midpoint_moment_impulse_nms',(3,))):
            _finite_array(step[name], shape)
        stored_work = float(np.dot(np.array(step['midpoint_external_load42']),
                                   np.array(step['midpoint_physical_increment42'])))
        if type(step['midpoint_load_work_j']) is not float or step['midpoint_load_work_j'] != stored_work:
            raise NativeTransientError('midpoint work inventory mismatch')
        support = step['midpoint_support']
        _keys(support, ('physical_fixed_indices','algebraic_fixed_indices','physical_reaction42',
            'algebraic_reaction42','physical_force_impulse_ns','physical_moment_impulse_nms',
            'physical_work_j','algebraic_work_j','clamped_total_angular_momentum_closure'))
        for name, shape in (('physical_reaction42',(42,)), ('algebraic_reaction42',(42,)),
                ('physical_force_impulse_ns',(3,)), ('physical_moment_impulse_nms',(3,))):
            _finite_array(support[name], shape)
        for name in ('physical_fixed_indices', 'algebraic_fixed_indices'):
            indices = support[name]
            if (type(indices) is not list or any(type(i) is not int or not 0 <= i < 42 for i in indices)
                    or len(indices) != len(set(indices))):
                raise NativeTransientError('exact bounded support index inventory required')
        if (any(i not in PHYSICAL_DOFS for i in support['physical_fixed_indices'])
                or any(i in PHYSICAL_DOFS for i in support['algebraic_fixed_indices'])
                or support['algebraic_fixed_indices'] not in ([], [3, 4, 5])
                or support['clamped_total_angular_momentum_closure'] != 'STAGE_PHYSICAL_FORCE_PLUS_FIXED_FRAME_REACTION_COUPLE'):
            raise NativeTransientError('native support physical/algebraic channel mismatch')
        for prefix in ('physical', 'algebraic'):
            value = support[prefix+'_work_j']
            if type(value) is not float or value != float(np.dot(
                    support[prefix+'_reaction42'], step['midpoint_physical_increment42'])):
                raise NativeTransientError('native support work inventory mismatch')
        previous = record
        record = step['endpoint']
        _validate_record(record, index, step['end_time'])
        if record['state']['predecessor_sha256'] != previous['state_sha256']:
            raise NativeTransientError('native state predecessor mismatch')
        if index % contract['settings']['save_every'] == 0:
            output.append(record)
    if (controller['time'] != record['time'] or controller['next_dt'] != planned
            or canonical(controller['outputs']) != canonical(output)):
        raise NativeTransientError('native output cadence/controller endpoint mismatch')
    native_data = _load(native.decode('ascii'))
    if canonical(native_data['state']) != canonical(record['state']):
        raise NativeTransientError('native and controller endpoint disagree')
    return controller, native


def _finite_array(value, shape):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'fi' or not np.isfinite(array).all():
        raise NativeTransientError('finite shaped native trajectory array required')


def _validate_record(record, epoch, time):
    _keys(record, ('time', 'epoch', 'state', 'state_sha256', 'accounting', 'physical_load'))
    state = record['state']
    _keys(state, ('mechanical', 'origins', 'histories', 'physical_velocity24',
        'physical_acceleration24', 'time', 'epoch', 'predecessor_sha256', 'context_sha256'))
    if (type(record['time']) is not float or not math.isfinite(record['time'])
            or type(record['epoch']) is not int or record['epoch'] != epoch
            or record['time'] != time or state['time'] != time or type(state['epoch']) is not int
            or state['epoch'] != epoch or sha(state) != record['state_sha256']):
        raise NativeTransientError('native endpoint state inventory mismatch')
    _keys(state['mechanical'], ('positions', 'position_low', 'nodal_frames', 'cell_rotations', 'resultants'))
    for name, shape in (('positions',(3,3)), ('position_low',(3,3)), ('nodal_frames',(3,3,3)),
                        ('cell_rotations',(1,2,3,3)), ('resultants',(1,18))):
        _finite_array(state['mechanical'][name], shape)
    for name in ('physical_velocity24', 'physical_acceleration24'):
        _finite_array(state[name], (24,))
    account = record['accounting']
    _keys(account, ('state_sha256','time','epoch','linear_momentum','linear_momentum_low',
        'angular_momentum','angular_momentum_low','kinetic_energy','elastic_energy','total_energy',
        'elastic_consistency_witness','resultant_compatibility18','rank_mass_checks'))
    if (account['state_sha256'] != record['state_sha256'] or account['time'] != time
            or type(account['epoch']) is not int or account['epoch'] != epoch):
        raise NativeTransientError('native endpoint accounting identity mismatch')
    for name in ('kinetic_energy','elastic_energy','total_energy','elastic_consistency_witness'):
        if type(account[name]) is not float or not math.isfinite(account[name]):
            raise NativeTransientError('finite native endpoint accounting required')
    for name in ('linear_momentum','linear_momentum_low','angular_momentum','angular_momentum_low'):
        _finite_array(account[name], (3,))
    _finite_array(account['resultant_compatibility18'], (18,))
    load = record['physical_load']
    _keys(load, ('context_sha256','load_sha256','external_load42',
        'line_force_work','applied_couple42','force','moment_about_origin',
        'momentum_origin_high','momentum_origin_low','has_nonpotential_couple'))
    for name, shape in (('external_load42',(42,)),
            ('applied_couple42',(42,)), ('force',(3,)), ('moment_about_origin',(3,)),
            ('momentum_origin_high',(3,)), ('momentum_origin_low',(3,))):
        _finite_array(load[name], shape)
    if (type(load['line_force_work']) is not float or not math.isfinite(load['line_force_work'])
            or type(load['has_nonpotential_couple']) is not bool
            or load['context_sha256'] != state['context_sha256']):
        raise NativeTransientError('native physical load roles/context mismatch')


def _validate_restored_endpoint(state, controller, accounting, load):
    record = controller['steps'][-1]['endpoint'] if controller['steps'] else controller['initial']
    regenerated = _record(state, accounting, load)
    if canonical(regenerated) != canonical(record):
        raise NativeTransientError('restored actual native accounting differs from stored endpoint')
