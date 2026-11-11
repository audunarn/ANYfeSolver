"""Single transverse first touch on the declared native/S3 material path.

Local development: the instantaneous force map holds the event fixed, while
algorithmic derivatives include its sensitivity. No event is physically committed.
"""
from dataclasses import dataclass, replace
from math import fsum
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from anyfesolver._ge_beam3_p5_seeded.core import sha

from anyfesolver._ge_beam3_mixed_ad import Jet2, so3_exp, so3_log, matmul, matvec, constant_matrix, dot, rotation_exponential
from .native_retained_station import _readonly
from .native_surface_path import material_path_pose, certify_material_ownership, affine_gap_polynomial
from .native_surface_s3_friction import _unit, _cross
from .friction_point import FrictionPointState, trial_coulomb_point, objective_relative_tangent_increment


@dataclass(frozen=True)
class NativeS3ActivationResult:
    residual: np.ndarray
    tangent: np.ndarray
    trial: object
    normal_gap_m: float
    normal_pressure_Pa: float
    normal_storage_j: float
    event_fraction: float
    event_gradient_local: np.ndarray
    path_sha256: str
    ownership_intervals: int
    normal_residual: np.ndarray
    normal_tangent: np.ndarray
    profile: str = 'NATIVE_S3_TRANSVERSE_FIRST_TOUCH_POINT_DEV_V1'

    def spatial_tangent(self):
        # Mixed virtual/algorithmic rotations already give spatial covectors.
        return self.tangent

    def normal_spatial_tangent(self):
        return self.normal_tangent


@dataclass(frozen=True)
class NativeS3ClosedSequenceResult(NativeS3ActivationResult):
    event_fractions: tuple = ()
    history_evaluations: int = 0
    profile: str = 'NATIVE_S3_CLOSED_EVENT_SEQUENCE_POINT_DEV_V1'


def _frame(corners):
    a=[corners[1][i]-corners[0][i] for i in range(3)]
    b=[corners[2][i]-corners[0][i] for i in range(3)]
    first=_unit(a); normal=_unit(_cross(a,b)); second=_cross(normal,first)
    return [[first[i],second[i],normal[i]] for i in range(3)]


def _point(corners, weights):
    return [sum(weights[a]*corners[a][i] for a in range(3)) for i in range(3)]


def _freeze_jets(values):
    for value in values:
        value.gradient.setflags(write=False);value.hessian.setflags(write=False)
    return tuple(values)


@lru_cache(maxsize=32)
def _event_rotation_chart(rotation_entries,size):
    """Pure endpoint rotation derivatives; no geometry/history is cached."""
    q=[Jet2.constant(0.,0) if size==0 else Jet2.variable(0.,3+i,size) for i in range(3)]
    rotation=matmul(so3_exp(q),constant_matrix(np.asarray(rotation_entries).reshape(3,3),size))
    return _freeze_jets(so3_log(rotation))


@lru_cache(maxsize=1)
def _endpoint_virtual_rotation():
    q=[Jet2.variable(0.,3+i,30) for i in range(3)]
    v=[Jet2.variable(0.,18+i,30) for i in range(3)]
    return tuple(_freeze_jets(row) for row in matmul(so3_exp(v),so3_exp(q)))


@lru_cache(maxsize=32)
def _endpoint_frame_jets(corner_bytes):
    """Immutable endpoint geometry jets, shared across material coordinates."""
    corners=np.frombuffer(corner_bytes,dtype=np.float64).reshape(3,3)
    y=[[Jet2.constant(corners[a,i],30)+Jet2.variable(0.,6+3*a+i,30)
        +Jet2.variable(0.,21+3*a+i,30) for i in range(3)] for a in range(3)]
    return tuple(_freeze_jets(row) for row in _frame(y))


def evaluate_first_touch(pair, start, history, translations, angular, shell_increment,
                         *, maximum_intervals=256):
    return _evaluate_transverse_event(pair,start,history,translations,angular,shell_increment,
                                      maximum_intervals=maximum_intervals,opening=False)


def evaluate_release(pair,start,history,translations,angular,shell_increment,*,maximum_intervals=256):
    return _evaluate_transverse_event(pair,start,history,translations,angular,shell_increment,
                                      maximum_intervals=maximum_intervals,opening=True)


def _evaluate_transverse_event(pair, start, history, translations, angular, shell_increment,
                              *, maximum_intervals, opening):
    """Actual-gap first-touch traction and complete spatial endpoint tangent.

    A global second-derivative bound proves strict gap decrease, hence unique
    first-root ownership. Unresolved/nonmonotone paths fail rather than dropping
    crossings. This initial branch requires an open physical history.
    """
    if sha(pair._definition()) != pair.definition_sha256:
        raise ValueError('native/S3 material-point definition changed')
    if opening and (type(history) is not FrictionPointState or history.partner_key!=pair.partner_key):
        raise ValueError('release candidate requires matching active physical history')
    if not opening and (type(history) is not FrictionPointState or history.partner_key is not None):
        raise ValueError('first-touch candidate requires released physical history')
    end=material_path_pose(start,translations,angular,shell_increment,1.)
    ownership=certify_material_ownership(pair,start,translations,angular,shell_increment,
                                         maximum_intervals=maximum_intervals)
    if not ownership.certified:
        raise ValueError('first-touch physical ownership unresolved: '+ownership.reason)
    source,target=pair.source,pair.target
    old=source.surface_variations(start.positions,start.position_low,start.cell_rotations)
    new=source.surface_variations(end.positions,end.position_low,end.cell_rotations)
    y0=target.adapter.current_positions(start.shell_displacement)
    y1=target.adapter.current_positions(end.shell_displacement)
    base0=old.station.surface_position-old.surface_arm
    base1=new.station.surface_position-new.surface_arm
    omega=np.asarray(angular)[source.cell]
    a,b=y0[1]-y0[0],y0[2]-y0[0]
    da=(y1[1]-y1[0])-a; db=(y1[2]-y1[0])-b
    c0=np.cross(a,b); c2=np.cross(a+da,b+db)
    c1=.5*(np.cross(a,b+db)+np.cross(a+da,b))
    normal0=c0/np.linalg.norm(c0)
    lower=min(float(normal0@c) for c in (c0,c1,c2))
    if lower<=0:raise ValueError('first-touch target area bound unresolved')
    cp0=np.cross(da,b)+np.cross(a,db)
    cp1=np.cross(da,b+db)+np.cross(a+da,db)
    m1=max(np.linalg.norm(cp0),np.linalg.norm(cp1)); m2=2*np.linalg.norm(np.cross(da,db))
    n1=m1/lower; n2=m2/lower+3*m1*m1/(lower*lower)
    velocity=base1-base0-(y1[0]-y0[0])
    arm=np.linalg.norm(old.surface_arm); angle=np.linalg.norm(omega)
    radius=max(np.linalg.norm(base0-y0[0]),np.linalg.norm(base1-y1[0]))+arm
    speed=np.linalg.norm(velocity)+angle*arm
    second_bound=n2*radius+2*n1*speed+angle*angle*arm
    normal_derivative=(cp0-normal0*(normal0@cp0))/np.linalg.norm(c0)
    derivative0=float(normal_derivative@(old.station.surface_position-y0[0])
                      +normal0@(velocity+np.cross(omega,old.surface_arm)))
    padding=512*np.finfo(float).eps*max(1.,radius,speed,second_bound)
    if ((not opening and derivative0+second_bound>=-padding) or
            (opening and derivative0-second_bound<=padding)):
        raise ValueError('first-touch transverse monotone path unresolved')

    def gap(alpha):
        pose=material_path_pose(start,translations,angular,shell_increment,alpha)
        x=source.evaluate(pose.positions,pose.position_low,pose.cell_rotations).surface_position
        t=target.physical_pose(pose.shell_displacement)
        return float((x-np.asarray(t.position))@np.asarray(t.material_frame)[:,2])

    g0,g1=gap(0.),gap(1.)
    prepared_translations=np.asarray(translations,dtype=float)
    weights=(1.-source.t,source.t)
    base_terms=tuple(tuple((weights[j]*float(start.positions[source.cell+j,i]),
                           weights[j]*float(start.position_low[source.cell+j,i])) for j in range(2))
                     for i in range(3))
    base_velocity=np.array([fsum(weights[j]*float(prepared_translations[source.cell+j,i]) for j in range(2))
                            for i in range(3)])
    def root_gap(alpha):
        # Prepared scalar geometry for root iterations only. Endpoint and final
        # root checks still use the authoritative adapters above/below. Native
        # base motion is affine; the material arm follows the declared Exp path.
        if alpha==0.:return g0
        if alpha==1.:return g1
        base=np.array([fsum((*base_terms[i][0],*base_terms[i][1],alpha*base_velocity[i])) for i in range(3)])
        x=base+rotation_exponential(alpha*omega)@old.surface_arm
        y=y0+alpha*(y1-y0)
        normal=np.cross(y[1]-y[0],y[2]-y[0]);normal/=np.linalg.norm(normal)
        point=np.asarray(target.barycentric)@y
        return float((x-point)@normal)
    exact_release_endpoint=False
    if opening and g1==0.:
        polynomial=affine_gap_polynomial(pair,start,np.asarray(translations),angular,shell_increment)
        exact_release_endpoint=polynomial is not None and sum(polynomial)==0
    if ((not opening and (not g0>padding or not g1<-padding)) or
            (opening and (not g0<-padding or not (g1>padding or exact_release_endpoint)))):
        raise ValueError('first-touch needs resolved open and closed endpoints')
    alpha=brentq(root_gap,0.,1.,xtol=4*np.finfo(float).eps,rtol=4*np.finfo(float).eps,maxiter=64)
    if abs(gap(alpha))>padding:
        raise ValueError('first-touch root residual unresolved')
    if opening:
        return _release_history(pair,start,end,history,translations,angular,shell_increment,
                                alpha,g1,ownership)

    return _closing_event_endpoint(pair,start,end,history,alpha,g1,padding,ownership)


def _certified_endpoint_primal(g,padding,certified_gap):
    """Validate a private certified endpoint gap; correct only the Jet2 primal.

    The certified exact gap is admitted only when finite, strictly closed, and
    in agreement with the endpoint law's own binary64 evaluation within the
    existing work-map mismatch padding. Inside the coordinate padding that
    evaluation cannot resolve the contact sign; the certified magnitude then
    replaces only the primal, preserving the declared chart's gradient/Hessian.
    Exact touches, out-of-profile open gaps, nonfinite values, and
    above-padding contradictions refuse.
    """
    if not np.isfinite(certified_gap):
        raise ValueError('certified endpoint gap must be finite')
    if certified_gap==0.:
        raise ValueError('certified endpoint gap exact touch unresolved')
    if certified_gap>0.:
        raise ValueError('certified endpoint gap contradicts closed endpoint profile')
    if abs(g.value-certified_gap)>padding:
        raise ValueError('certified endpoint gap contradicts endpoint work geometry')
    if abs(g.value)>padding:
        # Agreement with a strictly closed certified gap beyond the padding
        # means the binary64 evaluation resolved a clearly closed endpoint.
        return g
    return Jet2(certified_gap,g.gradient,g.hessian)


def _closing_event_endpoint(pair,start,end,history,alpha,g1,padding,ownership,*,certified_gap=None):
    """Closing-root force map differentiated from the original physical path.

    Caller certifies the root and event schedule. Earlier released windows may
    contribute cumulative history, but must leave zero elastic traction.
    The optional certified_gap is the exact signed endpoint gap from the
    precise coefficient schedule (native_event_polynomial), admitted only for
    endpoints whose binary64 evaluation lies inside the coordinate padding;
    it corrects only the primal above, leaving the declared derivative chart
    and both-body work maps untouched.
    """
    if type(history) is not FrictionPointState or history.partner_key is not None:
        raise ValueError('closing event requires released physical history')
    source,target=pair.source,pair.target
    old=source.surface_variations(start.positions,start.position_low,start.cell_rotations)
    new=source.surface_variations(end.positions,end.position_low,end.cell_rotations)
    y0=target.adapter.current_positions(start.shell_displacement)
    y1=target.adapter.current_positions(end.shell_displacement)
    base0=old.station.surface_position-old.surface_arm
    base1=new.station.surface_position-new.surface_arm

    def event_geometry(q, fraction, size):
        # q is an endpoint spatial perturbation. The algorithmic path is the
        # principal declared spatial chart from the fixed physical origin.
        relative_rotation=end.cell_rotations[source.cell]@start.cell_rotations[source.cell].T
        logarithm=_event_rotation_chart(tuple(relative_rotation.ravel()),size)
        rotated=matvec(so3_exp([fraction*v for v in logarithm]),
                       [Jet2.constant(v,size) for v in old.surface_arm])
        x=[base0[i]+fraction*(base1[i]+q[i]-base0[i])+rotated[i] for i in range(3)]
        y=[[y0[a,i]+fraction*(y1[a,i]+q[6+3*a+i]-y0[a,i]) for i in range(3)] for a in range(3)]
        frame=_frame(y); point=_point(y,target.barycentric)
        relative=[x[i]-point[i] for i in range(3)]
        return frame,relative

    # Implicit first-root gradient. No alpha Hessian is needed for the mixed
    # virtual/algorithmic blocks below (alpha has no virtual dependence).
    q=[Jet2.variable(0.,i,16) for i in range(15)]
    f,r=event_geometry(q,Jet2.variable(alpha,15,16),16)
    G=dot([row[2] for row in f],r)
    if abs(G.value)>padding:raise ValueError('first-touch differentiated root mismatch')
    if G.gradient[15]>=-padding:raise ValueError('first-touch derivative lost transversality')
    alpha_gradient=-G.gradient[:15]/G.gradient[15]
    state_only=getattr(pair,'_history_state_only',False)
    size=0 if state_only else 30
    gradient=np.zeros(size)
    if not state_only:gradient[:15]=alpha_gradient
    event_alpha=Jet2(alpha,gradient,np.zeros((size,size)))
    q=[Jet2.constant(0.,size) if state_only else Jet2.variable(0.,i,size) for i in range(15)]
    v=[Jet2.constant(0.,size) if state_only else Jet2.variable(0.,15+i,size) for i in range(15)]
    event_frame,event_relative=event_geometry(q,event_alpha,size)
    event_components=[sum(event_frame[j][i]*event_relative[j] for j in range(3)) for i in range(3)]
    rotated=matvec(constant_matrix(np.eye(3),0) if state_only else _endpoint_virtual_rotation(),
                   [Jet2.constant(x,size) for x in new.surface_arm])
    x=[base1[i]+q[i]+v[i]+rotated[i] for i in range(3)]
    y=[[y1[a,i]+q[6+3*a+i]+v[6+3*a+i] for i in range(3)] for a in range(3)]
    frame=(_frame(y) if state_only else _endpoint_frame_jets(np.asarray(y1,dtype=np.float64).tobytes()))
    normal=[row[2] for row in frame];point=_point(y,target.barycentric)
    relative=[x[i]-point[i] for i in range(3)]
    transported=matvec(frame,event_components)
    slip=[relative[i]-transported[i] for i in range(3)]
    normal_slip=dot(normal,slip)
    elastic=[slip[i]-normal[i]*normal_slip for i in range(3)]
    g=dot(normal,relative)
    if abs(g.value-g1)>padding:raise ValueError('first-touch endpoint work geometry mismatch')
    if certified_gap is not None:
        g=_certified_endpoint_primal(g,padding,certified_gap)
    pressure=-pair.normal_stiffness_Pa_per_m*g.value
    trial=trial_coulomb_point(history,material_frame=np.array([[j.value for j in row] for row in frame]),
        relative_increment=np.array([j.value for j in elastic]),normal_pressure=pressure,
        friction_coefficient=pair.friction_coefficient,tangential_stiffness=pair.tangential_stiffness_Pa_per_m,
        partner_key=pair.partner_key)
    energy=.5*pair.normal_stiffness_Pa_per_m*g*g
    if state_only:
        count=42+target.adapter.work_map.shape[1]
        zero=_readonly(np.zeros(count));matrix=_readonly(np.zeros((count,count)))
        return NativeS3ActivationResult(zero,matrix,trial,g.value,pressure,
            pair.reference_weight_m2*energy.value,float(alpha),_readonly(alpha_gradient),
            ownership.path_sha256,ownership.intervals,zero,matrix)
    residual=energy.gradient[15:].copy(); tangent=energy.hessian[15:,:15].copy()
    W=np.array([j.gradient[15:] for j in elastic]); D=np.array([j.gradient[:15] for j in elastic])
    H=np.array([j.hessian[15:,:15] for j in elastic]); traction=np.asarray(trial.traction)
    if trial.branch=='stick':
        dt=-pair.tangential_stiffness_Pa_per_m*D
    else:
        e=np.array([j.value for j in elastic]);norm=np.linalg.norm(e);direction=e/norm
        dp=-pair.normal_stiffness_Pa_per_m*g.gradient[:15]
        dt=(-pair.friction_coefficient*pressure/norm*(np.eye(3)-np.outer(direction,direction))@D
            -pair.friction_coefficient*np.outer(direction,dp))
    residual-=W.T@traction
    tangent-=np.einsum('i,ijk->jk',traction,H)+W.T@dt
    count=42+target.adapter.work_map.shape[1]; mapping=np.zeros((15,count))
    for node,weight in ((source.cell,1-source.t),(source.cell+1,source.t)):
        mapping[:3,6*node:6*node+3]=weight*np.eye(3)
    mapping[3:6,18+3*source.cell:21+3*source.cell]=np.eye(3)
    mapping[6:,42:]=target.adapter.work_map.toarray()
    residual=pair.reference_weight_m2*(mapping.T@residual)
    tangent=pair.reference_weight_m2*(mapping.T@tangent@mapping)
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(tangent)):
        raise ValueError('nonfinite first-touch material contribution')
    return NativeS3ActivationResult(_readonly(residual),_readonly(tangent),trial,g.value,pressure,
        pair.reference_weight_m2*energy.value,float(alpha),_readonly(alpha_gradient),ownership.path_sha256,
        ownership.intervals,_readonly(pair.reference_weight_m2*(mapping.T@energy.gradient[15:])),
        _readonly(pair.reference_weight_m2*(mapping.T@energy.hessian[15:,:15]@mapping)))


@dataclass(frozen=True)
class NativeS3ReleaseResult:
    residual: np.ndarray
    tangent: np.ndarray
    trial: object
    normal_gap_m: float
    normal_pressure_Pa: float
    normal_storage_j: float
    event_fraction: float
    path_sha256: str
    ownership_intervals: int
    history_evaluations: int
    profile: str='NATIVE_S3_TRANSVERSE_RELEASE_POINT_DEV_V1'

    def spatial_tangent(self):
        return self.tangent

    @property
    def normal_residual(self):
        return self.residual

    def normal_spatial_tangent(self):
        return self.tangent


def _release_history(pair,start,end,history,translations,angular,shell_increment,alpha,gap,ownership):
    """Disposable path return maps before opening; endpoint force/tangent zero.

    Fixed substeps are a bound definition, not an accuracy certificate. Preserve
    dissipation and separately report the final vanishing storage release.
    """
    previous=start; state=history; released=0.; plastic_material=np.zeros(3); dissipations=[]
    old_source=pair.source.evaluate(previous.positions,previous.position_low,previous.cell_rotations)
    old_target=pair.target.physical_pose(previous.shell_displacement)
    for fraction in np.linspace(0.,alpha,pair.event_history_substeps+1)[1:]:
        current=material_path_pose(start,translations,angular,shell_increment,float(fraction))
        source=pair.source.evaluate(current.positions,current.position_low,current.cell_rotations)
        target=pair.target.physical_pose(current.shell_displacement)
        increment=objective_relative_tangent_increment(source_committed=old_source.surface_position,
            target_committed=old_target.position,source_candidate=source.surface_position,
            target_candidate=target.position,target_frame_committed=old_target.material_frame,
            target_frame_candidate=target.material_frame)
        normal=np.asarray(target.material_frame)[:,2]
        pressure=max(0.,-pair.normal_stiffness_Pa_per_m*float((source.surface_position-np.asarray(target.position))@normal))
        # Root residual is already bounded. The event is the zero-pressure stage.
        if fraction==alpha:pressure=0.
        trial=trial_coulomb_point(state,material_frame=target.material_frame,relative_increment=increment,
            normal_pressure=pressure,friction_coefficient=pair.friction_coefficient,
            tangential_stiffness=pair.tangential_stiffness_Pa_per_m,partner_key=pair.partner_key)
        plastic_material+=np.asarray(target.material_frame).T@np.asarray(trial.plastic_slip_increment)
        dissipations.append(trial.dissipated_increment_density)
        state=trial.candidate_state; released+=trial.released_storage_density; previous=current
        old_source=source; old_target=target
    target=pair.target.physical_pose(end.shell_displacement)
    trial=trial_coulomb_point(state,material_frame=target.material_frame,relative_increment=(0.,0.,0.),
        normal_pressure=0.,friction_coefficient=pair.friction_coefficient,
        tangential_stiffness=pair.tangential_stiffness_Pa_per_m,partner_key=None)
    trial=replace(trial,dissipated_increment_density=fsum(dissipations),
                  released_storage_density=released,
                  plastic_slip_increment=tuple(np.asarray(target.material_frame)@plastic_material))
    count=42+pair.target.adapter.work_map.shape[1]
    return NativeS3ReleaseResult(_readonly(np.zeros(count)),_readonly(np.zeros((count,count))),trial,
        gap,0.,0.,alpha,ownership.path_sha256,ownership.intervals,pair.event_history_substeps+1)
