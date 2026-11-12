"""Bounded transverse event sequences; initially open endpoint development.

The explicit development selection binds its own restart identity. Closed-ending
replay is a standalone development helper with original-origin derivatives. An ending-open
path has zero endpoint force/tangent but retains physical dissipation and slip.
"""
from dataclasses import dataclass,replace
from math import fsum

import numpy as np
from scipy.optimize import brentq

from anyfesolver._ge_beam3_p5_seeded.core import sha
from .friction_point import FrictionPointState,trial_coulomb_point
from .native_retained_station import _readonly
from .native_surface_path import material_path_pose,certify_material_ownership
from .native_surface_support import point_gap_slab_enclosure


@dataclass(frozen=True)
class MaterialSequenceEvent:
    fraction: float
    opening: bool
    bracket: tuple
    gap_residual: float


@dataclass(frozen=True)
class MaterialEventSequence:
    events: tuple
    intervals: int
    path_sha256: str
    endpoint_gap: float
    numerator_endpoint_signs: tuple=(0,0)


def _sequence_endpoint_padding(pair,start,endpoint):
    scale=max(1.,float(np.max(np.abs(start.positions))),
        float(np.max(np.abs(endpoint.positions))),
        float(np.max(np.abs(pair.target.adapter.reference_positions))))
    return 512*np.finfo(float).eps*scale


def _resolve_endpoint_state(gap,sign,padding):
    """Certified endpoint contact state: +1 open, -1 closed.

    A binary64 gap beyond the conservative coordinate-scale padding resolves
    on its own. Inside that padding the literal binary64 evaluation cannot
    resolve the sign; the exact coefficient schedule built from the same
    literal inputs (including the literal barycentric weights) is then
    authoritative. Exact touches and contradictions still refuse.
    """
    if gap==0.:raise ValueError('event sequence endpoint exact touch unresolved')
    binary=1 if gap>0. else -1
    if abs(gap)>padding:
        if sign and sign!=binary:
            raise ValueError('event sequence endpoint sign contradiction')
        return binary
    if not sign:raise ValueError('event sequence endpoint contact state unresolved')
    return sign


def locate_material_event_sequence(pair,start,translations,angular,shell_increment,*,maximum_intervals=256):
    """Cover the complete actual path before admitting any event schedule."""
    if type(maximum_intervals) is not int or maximum_intervals<1:
        raise ValueError('positive shared event interval cap required')
    if sha(pair._definition())!=pair.definition_sha256:
        raise ValueError('native/S3 material-point definition changed')
    ownership=certify_material_ownership(pair,start,translations,angular,shell_increment,
        maximum_intervals=maximum_intervals)
    if not ownership.certified:raise ValueError('event sequence ownership unresolved: '+ownership.reason)
    remaining=maximum_intervals-ownership.intervals
    if remaining<1:raise MemoryError('native material event sequence interval cap exhausted')
    from .native_surface_quadrature import certify_station_orientation_path
    orientation=certify_station_orientation_path(pair.source,start,translations,angular,
        shell_increment,maximum_intervals=remaining)
    # Retain the exact profile validation. Root isolation then uses the same
    # path's unnormalised numerator, preserving cancellation at tiny pulses.
    enclose=point_gap_slab_enclosure(pair,start,translations,angular,shell_increment)
    from .native_event_polynomial import material_time_numerator
    # The existing precise sine kernel has a declared [0,2*pi] argument
    # range. Larger paths retain their original conservative time enclosure;
    # choose this before isolation, never after a failed candidate.
    numerator=None
    if np.linalg.norm(np.asarray(angular)[pair.source.cell])<2*np.pi-32*np.finfo(float).eps:
        numerator=material_time_numerator(pair,start,translations,angular,shell_increment)
        enclose=numerator.enclosure
    cache={}
    def gap(fraction):
        if fraction not in cache:
            pose=material_path_pose(start,translations,angular,shell_increment,float(fraction))
            source=pair.source.evaluate(pose.positions,pose.position_low,pose.cell_rotations).surface_position
            target=pair.target.physical_pose(pose.shell_displacement)
            cache[fraction]=float((source-np.asarray(target.position))@np.asarray(target.material_frame)[:,2])
        return cache[fraction]
    count=ownership.intervals+orientation;pending=[(0.,1.)];events={}
    while pending:
        if count>=maximum_intervals:raise MemoryError('native material event sequence interval cap exhausted')
        left,right=pending.pop();count+=1
        bounds=enclose(left,right)
        if bounds.gap_lower>0. or bounds.gap_upper<0.:continue
        direction=1 if bounds.derivative_lower>0. else -1 if bounds.derivative_upper<0. else 0
        if direction:
            evaluate=(lambda t:float(numerator.value(t))) if numerator is not None else gap
            gl,gr=evaluate(left),evaluate(right)
            if gl*gr>0.:continue
            root=left if gl==0. else right if gr==0. else brentq(evaluate,left,right,
                xtol=4*np.finfo(float).eps,rtol=4*np.finfo(float).eps,maxiter=64)
            residual=gap(root)
            scale=max(1.,float(np.max(np.abs(start.positions))),
                float(np.max(np.abs(pair.target.adapter.reference_positions))))
            if abs(residual)>512*np.finfo(float).eps*scale:
                raise ValueError('event sequence root residual unresolved')
            event=MaterialSequenceEvent(float(root),direction>0,(left,right),residual)
            if root in events and events[root].opening!=event.opening:
                raise ValueError('event sequence root ownership inconsistent')
            events[root]=event
            continue
        middle=(left+right)/2
        if not left<middle<right:raise ValueError('event sequence temporal resolution exhausted')
        pending.extend(((middle,right),(left,middle)))
    ordered=tuple(events[t] for t in sorted(events))
    return MaterialEventSequence(ordered,count,ownership.path_sha256,gap(1.),
        (numerator.sign(0.),numerator.sign(1.)) if numerator is not None else (0,0))


@dataclass(frozen=True)
class NativeS3OpenSequenceResult:
    residual: np.ndarray
    tangent: np.ndarray
    trial: object
    normal_gap_m: float
    event_fractions: tuple
    path_sha256: str
    ownership_intervals: int
    history_evaluations: int
    normal_pressure_Pa: float=0.
    normal_storage_j: float=0.
    profile: str='NATIVE_S3_OPEN_EVENT_SEQUENCE_POINT_DEV_V1'

    def spatial_tangent(self):return self.tangent
    def normal_spatial_tangent(self):return self.tangent
    @property
    def normal_residual(self):return self.residual


def evaluate_open_event_sequence(pair,start,history,translations,angular,shell_increment,*,maximum_intervals=256,sequence=None):
    """Disposable replay of complete closed windows ending strictly open."""
    from .native_surface_activation import _release_history
    if type(history) is not FrictionPointState:
        raise ValueError('open event sequence requires physical friction history')
    if sequence is None:
        sequence=locate_material_event_sequence(pair,start,translations,angular,shell_increment,
            maximum_intervals=maximum_intervals)
    endpoint=material_path_pose(start,translations,angular,shell_increment,1.)
    source=pair.source.evaluate(start.positions,start.position_low,start.cell_rotations).surface_position
    target=pair.target.physical_pose(start.shell_displacement)
    initial_gap=float((source-np.asarray(target.position))@np.asarray(target.material_frame)[:,2])
    padding=_sequence_endpoint_padding(pair,start,endpoint)
    initial_state=_resolve_endpoint_state(initial_gap,sequence.numerator_endpoint_signs[0],padding)
    final_state=_resolve_endpoint_state(sequence.endpoint_gap,sequence.numerator_endpoint_signs[1],padding)
    if final_state<0:
        # The binary64 endpoint evaluation can flip the sign of a gap far
        # below the coordinate-scale padding. The certified schedule ends
        # closed; replay through the closed endpoint law with the same
        # located sequence, without a second certification pass.
        return _closed_sequence_from_schedule(pair,start,history,translations,angular,
            shell_increment,sequence)
    initially_closed=initial_state<0
    if history.partner_key!=(pair.partner_key if initially_closed else None):
        raise ValueError('open event sequence history disagrees with initial contact')
    events=sequence.events
    if ((bool(events) and not events[-1].opening) or
        (initially_closed and not events) or
        any(event.opening!=(initially_closed if i%2==0 else not initially_closed)
            for i,event in enumerate(events))):
        raise ValueError('open event sequence requires alternating roots ending in opening')
    windows=[];a=0. if initially_closed else None
    for event in events:
        if not 0.<event.fraction<1.:raise ValueError('open event sequence endpoint root unresolved')
        if event.opening:
            if a is None:raise ValueError('open event sequence window ownership unresolved')
            windows.append((a,event.fraction));a=None
        else:a=event.fraction
    state=history;dissipations=[];released=[];plastic=np.zeros(3);evaluations=0
    for a,b in windows:
        if not 0.<=a<b<1.:raise ValueError('open event sequence endpoint root unresolved')
        first=material_path_pose(start,translations,angular,shell_increment,a)
        last=material_path_pose(start,translations,angular,shell_increment,b)
        if a>0.:
            frame=pair.target.physical_pose(first.shell_displacement).material_frame
            state=trial_coulomb_point(state,material_frame=frame,relative_increment=(0.,0.,0.),
                normal_pressure=0.,friction_coefficient=pair.friction_coefficient,
                tangential_stiffness=pair.tangential_stiffness_Pa_per_m,partner_key=None).candidate_state
            evaluations+=1
        value=_release_history(pair,first,last,state,np.asarray(translations)*(b-a),
            np.asarray(angular)*(b-a),np.asarray(shell_increment)*(b-a),1.,0.,sequence)
        state=value.trial.candidate_state;evaluations+=value.history_evaluations
        dissipations.append(value.trial.dissipated_increment_density)
        released.append(value.trial.released_storage_density)
        frame=np.asarray(pair.target.physical_pose(last.shell_displacement).material_frame)
        plastic+=frame.T@np.asarray(value.trial.plastic_slip_increment)
    final_frame=pair.target.physical_pose(endpoint.shell_displacement).material_frame
    trial=trial_coulomb_point(state,material_frame=final_frame,relative_increment=(0.,0.,0.),
        normal_pressure=0.,friction_coefficient=pair.friction_coefficient,
        tangential_stiffness=pair.tangential_stiffness_Pa_per_m,partner_key=None)
    trial=replace(trial,dissipated_increment_density=fsum(dissipations),
        released_storage_density=fsum(released),plastic_slip_increment=tuple(np.asarray(final_frame)@plastic))
    size=42+pair.target.adapter.work_map.shape[1]
    return NativeS3OpenSequenceResult(_readonly(np.zeros(size)),_readonly(np.zeros((size,size))),
        trial,sequence.endpoint_gap,tuple(e.fraction for e in events),sequence.path_sha256,
        sequence.intervals,evaluations+1,
        profile=('NATIVE_S3_RELEASED_ENDPOINT_EVENT_SEQUENCE_POINT_DEV_V2' if initially_closed
                 else 'NATIVE_S3_OPEN_EVENT_SEQUENCE_POINT_DEV_V1'))


def evaluate_closed_event_sequence(pair,start,history,translations,angular,shell_increment,*,maximum_intervals=256):
    """Development replay ending in contact after a certified closing event.

    Completed windows use the declared discrete release history. The final
    window uses the first-touch endpoint law and its original-origin derivative.
    This is not a derivative of cumulative work and does not admit persistence
    without a closing event. No physical state is committed here.
    """
    if type(history) is not FrictionPointState:
        raise ValueError('closed event sequence requires physical friction history')
    sequence=locate_material_event_sequence(pair,start,translations,angular,shell_increment,
        maximum_intervals=maximum_intervals)
    return _closed_sequence_from_schedule(pair,start,history,translations,angular,shell_increment,sequence)


def evaluate_closed_or_persistent_sequence(pair,start,history,translations,angular,shell_increment,*,maximum_intervals=256):
    """Explicit sequence policy: certify once, then replay or retain persistence."""
    sequence=locate_material_event_sequence(pair,start,translations,angular,shell_increment,
        maximum_intervals=maximum_intervals)
    endpoint=material_path_pose(start,translations,angular,shell_increment,1.)
    padding=_sequence_endpoint_padding(pair,start,endpoint)
    final_state=_resolve_endpoint_state(sequence.endpoint_gap,sequence.numerator_endpoint_signs[1],padding)
    if final_state>0:
        contribution=evaluate_open_event_sequence(pair,start,history,translations,angular,
            shell_increment,sequence=sequence)
        return contribution,sequence
    if sequence.events:
        return _closed_sequence_from_schedule(pair,start,history,translations,angular,shell_increment,sequence),sequence
    if type(history) is not FrictionPointState or history.partner_key!=pair.partner_key:
        raise ValueError('closed sequence persistence requires matching active physical history')
    return pair.evaluate(start,endpoint,history),sequence


def _closed_sequence_from_schedule(pair,start,history,translations,angular,shell_increment,sequence):
    from .native_surface_activation import _release_history,_closing_event_endpoint,NativeS3ClosedSequenceResult
    if type(history) is not FrictionPointState:
        raise ValueError('closed event sequence requires physical friction history')
    endpoint=material_path_pose(start,translations,angular,shell_increment,1.)
    source=pair.source.evaluate(start.positions,start.position_low,start.cell_rotations).surface_position
    target=pair.target.physical_pose(start.shell_displacement)
    initial_gap=float((source-np.asarray(target.position))@np.asarray(target.material_frame)[:,2])
    padding=_sequence_endpoint_padding(pair,start,endpoint)
    # A precise numerator certifies strict signs below the conservative
    # binary64 coordinate padding, where the literal endpoint evaluation
    # cannot resolve the sign at all. Exact touch and contradictory endpoint
    # geometry still refuse.
    initial_state=_resolve_endpoint_state(initial_gap,sequence.numerator_endpoint_signs[0],padding)
    final_state=_resolve_endpoint_state(sequence.endpoint_gap,sequence.numerator_endpoint_signs[1],padding)
    if final_state>0:
        # Certified strictly open final gap: binary64 endpoint rounding
        # misrouted this path; replay through the open schedule law.
        return evaluate_open_event_sequence(pair,start,history,translations,angular,
            shell_increment,sequence=sequence)
    initially_closed=initial_state<0
    if history.partner_key!=(pair.partner_key if initially_closed else None):
        raise ValueError('closed event sequence history disagrees with initial contact')
    events=sequence.events
    if (not events or events[-1].opening or
        any(e.opening!=(initially_closed if i%2==0 else not initially_closed)
            for i,e in enumerate(events)) or
        any(not 0.<e.fraction<1. for e in events)):
        raise ValueError('closed event sequence requires alternating interior events ending in closure')
    state=history;dissipations=[];released=[];plastic=np.zeros(3);evaluations=0
    a=0. if initially_closed else None
    for event in events[:-1]:
        if not event.opening:
            a=event.fraction
            continue
        b=event.fraction
        if a is None or not a<b:
            raise ValueError('closed event sequence window ownership unresolved')
        first=material_path_pose(start,translations,angular,shell_increment,a)
        last=material_path_pose(start,translations,angular,shell_increment,b)
        if a>0.:
            frame=pair.target.physical_pose(first.shell_displacement).material_frame
            state=trial_coulomb_point(state,material_frame=frame,relative_increment=(0.,0.,0.),
                normal_pressure=0.,friction_coefficient=pair.friction_coefficient,
                tangential_stiffness=pair.tangential_stiffness_Pa_per_m,partner_key=None).candidate_state
            evaluations+=1
        value=_release_history(pair,first,last,state,np.asarray(translations)*(b-a),
            np.asarray(angular)*(b-a),np.asarray(shell_increment)*(b-a),1.,0.,sequence)
        state=value.trial.candidate_state;evaluations+=value.history_evaluations
        dissipations.append(value.trial.dissipated_increment_density)
        released.append(value.trial.released_storage_density)
        frame=np.asarray(pair.target.physical_pose(last.shell_displacement).material_frame)
        plastic+=frame.T@np.asarray(value.trial.plastic_slip_increment)
        a=None
    certified_gap=None
    if sequence.numerator_endpoint_signs[1] and abs(sequence.endpoint_gap)<=padding:
        # The precise schedule certified this sub-padding endpoint that the
        # literal binary64 evaluation cannot resolve; return its exact
        # magnitude so the endpoint law corrects only the Jet2 primal.
        from .native_event_polynomial import certified_material_gap
        certified_gap=certified_material_gap(pair,start,translations,angular,shell_increment)
    value=_closing_event_endpoint(pair,start,endpoint,state,events[-1].fraction,
        sequence.endpoint_gap,padding,sequence,certified_gap=certified_gap)
    frame=np.asarray(pair.target.physical_pose(endpoint.shell_displacement).material_frame)
    trial=replace(value.trial,
        dissipated_increment_density=fsum((*dissipations,value.trial.dissipated_increment_density)),
        released_storage_density=fsum((*released,value.trial.released_storage_density)),
        plastic_slip_increment=tuple(frame@plastic+np.asarray(value.trial.plastic_slip_increment)))
    return NativeS3ClosedSequenceResult(**{**vars(value),'trial':trial,
        'profile':'NATIVE_S3_CLOSED_EVENT_SEQUENCE_POINT_DEV_V1'},
        event_fractions=tuple(e.fraction for e in events),history_evaluations=evaluations+1)
