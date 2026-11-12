"""Local retained round-surface/S3 material-pair work; no analysis admission.

Actual surface-point gap, explicit reference-area weight and persistent material
labels. No centreline normal surrogate, partner transfer or surface quadrature is
invented here. Native second derivatives use its spatial multiplicative chart.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .native_surface_activation import NativeS3ActivationResult

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, so3_exp, matvec, dot, sqrt
from anyfesolver._ge_beam3_p5.arrays import _array, _frames
from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_retained_station import NativeRetainedRoundStation, _readonly
from .s3_friction_point import S3SkinMaterialPoint
from .friction_point import (
    FrictionPointState, FrictionPointTrial, objective_relative_tangent_increment,
    trial_coulomb_point,
)

PROFILE='NATIVE_RETAINED_ROUND_S3_MATERIAL_POINT_FRICTION_DEV_V1'


@dataclass(frozen=True)
class NativeS3MaterialConfiguration:
    positions: np.ndarray
    position_low: np.ndarray
    cell_rotations: np.ndarray
    shell_displacement: np.ndarray

    def __post_init__(self):
        for name in ('positions','position_low'):
            object.__setattr__(self,name,_readonly(_array(getattr(self,name),(3,3),name)))
        object.__setattr__(self,'cell_rotations',_readonly(_frames(self.cell_rotations,2,'cell rotations')))
        u=np.asarray(self.shell_displacement,dtype=float)
        if u.ndim!=1 or not np.all(np.isfinite(u)):raise ValueError('finite shell displacement vector required')
        object.__setattr__(self,'shell_displacement',_readonly(u))


@dataclass(frozen=True)
class NativeS3MaterialPointResult:
    residual: np.ndarray
    chart_tangent: np.ndarray
    trial: FrictionPointTrial
    normal_gap_m: float
    normal_pressure_Pa: float
    normal_storage_j: float
    normal_residual: np.ndarray
    normal_chart_tangent: np.ndarray
    profile: str=PROFILE

    def spatial_tangent(self):
        result=self.chart_tangent.copy()
        result[:42,:42]=spatial_jacobian(self.residual[:42],result[:42,:42])
        return _readonly(result)

    def normal_spatial_tangent(self):
        result=self.normal_chart_tangent.copy()
        result[:42,:42]=spatial_jacobian(self.normal_residual[:42],result[:42,:42])
        return _readonly(result)


@dataclass(frozen=True)
class NativeS3MaterialPathResult:
    """Disposable certified contribution; structural equilibrium still required."""
    configuration: NativeS3MaterialConfiguration
    contribution: NativeS3MaterialPointResult | NativeS3ActivationResult
    path_sha256: str
    certified_intervals: int


def _cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def _unit(v):
    length=sqrt(dot(v,v))
    return [a/length for a in v]


def _rotated_arm_jet(variations):
    """Exact spatial Exp(delta) arm jet at delta=0, in the local 15 chart."""
    arm = variations.surface_arm
    generators = np.array(((0., 0., 0.), (0., 0., -1.), (0., 1., 0.),
                           (0., 0., 1.), (0., 0., 0.), (-1., 0., 0.),
                           (0., -1., 0.), (1., 0., 0.), (0., 0., 0.))).reshape(3, 3, 3)
    gradient = np.einsum('iab,b->ai', generators, arm)
    result = []
    for axis in range(3):
        g = np.zeros(15); h = np.zeros((15, 15))
        g[3:6] = gradient[axis]
        h[3:6, 3:6] = variations.surface_rotation_hessian[axis]
        result.append(Jet2(arm[axis], g, h))
    return result


@lru_cache(maxsize=32)
def _skin_frame_jet(corner_bytes):
    """Bounded immutable geometry-only memo; never stores material response.

    The exact binary64 physical corners determine the frame and its vertex
    derivatives in this fixed local chart. Candidate geometry/polarity checks
    precede this helper on every evaluation, including memo hits.
    """
    corners = np.frombuffer(corner_bytes, dtype=np.float64).reshape(3, 3)
    y = [[Jet2.variable(corners[a, i], 6+3*a+i, 15)
          for i in range(3)] for a in range(3)]
    first = [y[1][i]-y[0][i] for i in range(3)]
    second = [y[2][i]-y[0][i] for i in range(3)]
    t1 = _unit(first); n = _unit(_cross(first, second)); t2 = _cross(n, t1)
    frame = tuple(tuple(row) for row in zip(t1, t2, n))
    for row in frame:
        for entry in row:
            entry.gradient.setflags(write=False); entry.hessian.setflags(write=False)
    return frame


def _local_contact_jets(new, corners, barycentric, old_target, old, history, partner_key):
    """Batch the existing local Jet2 sum/product rules; no new work law."""
    rotated = _rotated_arm_jet(new)
    separation = np.array([j.value + new.station.surface_position[i]-new.surface_arm[i]
                           for i,j in enumerate(rotated)]) - np.asarray(barycentric) @ corners
    B = np.array([j.gradient for j in rotated]); H = np.array([j.hessian for j in rotated])
    B[:, :3] += np.eye(3)
    for a,w in enumerate(barycentric): B[:,6+3*a:9+3*a] -= w*np.eye(3)
    frame = _skin_frame_jet(np.asarray(corners,dtype=np.float64).tobytes())
    F = np.array([[j.value for j in row] for row in frame])
    FB = np.array([[j.gradient for j in row] for row in frame])
    FH = np.array([[j.hessian for j in row] for row in frame])
    n, nB, nH = F[:,2], FB[:,2], FH[:,2]
    components = np.asarray(old_target.material_frame).T @ (old.surface_position-np.asarray(old_target.position))
    d = separation - F @ components
    dB = B - np.einsum('aci,c->ai',FB,components)
    dH = H - np.einsum('acij,c->aij',FH,components)
    def scalar_product(v, vB, vH):
        value = float(n @ v)
        gradient = n @ vB + v @ nB
        cross = nB.T @ vB
        hessian = np.einsum('a,aij->ij',n,vH)+np.einsum('a,aij->ij',v,nH)+cross+cross.T
        return Jet2(value,gradient,hessian)
    q = scalar_product(d,dB,dH)
    retained = np.asarray(history.elastic_slip) if history.partner_key==partner_key else np.zeros(3)
    old_elastic = np.asarray(history.material_frame).T @ retained
    e = d-n*q.value+F@old_elastic
    eB = dB-n[:,None]*q.gradient-nB*q.value+np.einsum('aci,c->ai',FB,old_elastic)
    eH = (dH-n[:,None,None]*q.hessian-nH*q.value
          -nB[:,:,None]*q.gradient[None,None,:]-nB[:,None,:]*q.gradient[None,:,None]
          +np.einsum('acij,c->aij',FH,old_elastic))
    return scalar_product(separation,B,H), [Jet2(e[i],eB[i],eH[i]) for i in range(3)]


class NativeS3MaterialPointFriction:
    def __init__(self,source,target,*,reference_weight_m2,normal_stiffness_Pa_per_m,
                 friction_coefficient,tangential_stiffness_Pa_per_m,maximum_coupled_dofs=60,
                 activation_policy='reject', event_history_substeps=32):
        if type(source) is not NativeRetainedRoundStation or type(target) is not S3SkinMaterialPoint:
            raise TypeError('exact native round station and S3 material point required')
        if target.adapter.side!='top':raise ValueError('initial native/S3 material profile requires target top skin')
        self.source,self.target=source,target
        if activation_policy not in ('reject','transverse_first_touch','transverse_events','transverse_event_sequence'):
            raise ValueError('explicit supported native material activation policy required')
        self.activation_policy=activation_policy
        if type(event_history_substeps) is not int or not 2<=event_history_substeps<=1024:
            raise ValueError('event history substeps must be an integer from 2 to 1024')
        self.event_history_substeps=event_history_substeps
        if type(maximum_coupled_dofs) is not int or maximum_coupled_dofs<1:
            raise ValueError('positive coupled-coordinate resource cap required')
        self.maximum_coupled_dofs=maximum_coupled_dofs
        if 42+target.adapter.work_map.shape[1]>maximum_coupled_dofs:
            raise MemoryError('native/S3 material-point coordinate cap exceeded')
        for name,value,positive in (
            ('reference_weight_m2',reference_weight_m2,True),
            ('normal_stiffness_Pa_per_m',normal_stiffness_Pa_per_m,True),
            ('friction_coefficient',friction_coefficient,False),
            ('tangential_stiffness_Pa_per_m',tangential_stiffness_Pa_per_m,True)):
            value=float(value)
            if not np.isfinite(value) or value<0 or (positive and value==0):
                raise ValueError(f'{name} has invalid dimensions/value')
            setattr(self,name,value)
        self.partner_key=(f'{source.identity}|s3:{target.adapter.body_id}:'
            f'{target.adapter.element_id}:{target.adapter.side}:{target.barycentric!r}')
        self.definition_sha256=sha(self._definition())

    def _definition(self):
        source=self.source;target=self.target;mapping=target.adapter.work_map
        result=dict(profile=PROFILE,source=dict(identity=source.identity,
            reference=source.reference.canonical_data(),element=source.element_id,
            xi=source.xi,radius=source.radius,angle=source.angle,cell=source.cell,t=source.t),
            target=dict(body=target.adapter.body_id,element=target.adapter.element_id,
                side=target.adapter.side,barycentric=target.barycentric,
                reference=target.adapter.reference_positions.tolist(),shape=mapping.shape,
                indptr=mapping.indptr.tolist(),indices=mapping.indices.tolist(),data=mapping.data.tolist()),
            weight=self.reference_weight_m2,normal=self.normal_stiffness_Pa_per_m,
            mu=self.friction_coefficient,tangent=self.tangential_stiffness_Pa_per_m,
            maximum_coupled_dofs=self.maximum_coupled_dofs)
        if self.activation_policy!='reject':result['activation_policy']=self.activation_policy
        if self.activation_policy in ('transverse_events','transverse_event_sequence'):
            result['event_history_substeps']=self.event_history_substeps
        if self.activation_policy=='transverse_event_sequence':
            result['sequence_profile']='NATIVE_S3_OPEN_CLOSED_EVENT_SEQUENCE_POINT_DEV_V2'
        return result

    def evaluate_path(self, committed, history, translations, angular, shell_increment,
                      *, maximum_intervals=256):
        """Evaluate from one physical origin only after path admission.

        Newton candidates never become history origins here. Normal transitions
        require an equilibrated event subdivision; an endpoint law alone cannot
        bypass that boundary. Returned tangent is the endpoint law's complete
        spatial tangent, valid on its admitted smooth persistence branch.
        """
        from .native_surface_path import material_path_pose, certify_material_path, affine_gap_polynomial

        candidate = material_path_pose(committed, translations, angular, shell_increment, 1.)
        point = self.source.evaluate(committed.positions, committed.position_low,
                                     committed.cell_rotations).surface_position
        target = self.target.physical_pose(committed.shell_displacement)
        gap = float((point-np.asarray(target.position)) @ np.asarray(target.material_frame)[:, 2])
        if self.activation_policy in ('transverse_first_touch','transverse_events','transverse_event_sequence'):
            next_point=self.source.evaluate(candidate.positions,candidate.position_low,candidate.cell_rotations).surface_position
            next_target=self.target.physical_pose(candidate.shell_displacement)
            next_gap=float((next_point-np.asarray(next_target.position))@np.asarray(next_target.material_frame)[:,2])
            if self.activation_policy=='transverse_event_sequence' and gap!=0. and next_gap>0.:
                from .native_surface_event_sequence import evaluate_open_event_sequence
                contribution=evaluate_open_event_sequence(self,committed,history,translations,angular,
                    shell_increment,maximum_intervals=maximum_intervals)
                return NativeS3MaterialPathResult(candidate,contribution,contribution.path_sha256,
                    contribution.ownership_intervals)
            if self.activation_policy=='transverse_event_sequence' and gap!=0. and next_gap<0.:
                from .native_surface_event_sequence import evaluate_closed_or_persistent_sequence
                contribution,sequence=evaluate_closed_or_persistent_sequence(self,committed,history,
                    translations,angular,shell_increment,maximum_intervals=maximum_intervals)
                return NativeS3MaterialPathResult(candidate,contribution,sequence.path_sha256,sequence.intervals)
            if gap>0. and next_gap<0.:
                from .native_surface_activation import evaluate_first_touch
                contribution=evaluate_first_touch(self,committed,history,translations,angular,shell_increment,
                                                   maximum_intervals=maximum_intervals)
                return NativeS3MaterialPathResult(candidate,contribution,contribution.path_sha256,
                                                   contribution.ownership_intervals)
            if self.activation_policy in ('transverse_events','transverse_event_sequence') and gap<0. and next_gap>=0.:
                from .native_surface_activation import evaluate_release
                contribution=evaluate_release(self,committed,history,translations,angular,shell_increment,
                                                maximum_intervals=maximum_intervals)
                return NativeS3MaterialPathResult(candidate,contribution,contribution.path_sha256,
                                                   contribution.ownership_intervals)
        polynomial = affine_gap_polynomial(self, committed, np.asarray(translations), angular, shell_increment)
        if polynomial is not None:
            gap = polynomial[0] if polynomial[0] != 0 else polynomial[1]
        elif gap == 0.:
            next_point = self.source.evaluate(candidate.positions, candidate.position_low,
                                             candidate.cell_rotations).surface_position
            next_target = self.target.physical_pose(candidate.shell_displacement)
            gap = float((next_point-np.asarray(next_target.position))
                        @ np.asarray(next_target.material_frame)[:, 2])
        certificate = certify_material_path(self, committed, translations, angular,
            shell_increment, closed=gap <= 0., maximum_intervals=maximum_intervals)
        if not certificate.certified:
            raise ValueError('native/S3 material history path requires subdivision: '+certificate.reason)
        contribution = self.evaluate(committed, candidate, history)
        return NativeS3MaterialPathResult(candidate, contribution, certificate.path_sha256,
                                         certificate.intervals)

    def evaluate(self,committed,candidate,history):
        if sha(self._definition())!=self.definition_sha256:
            raise ValueError('native/S3 material-point definition changed')
        if type(committed) is not NativeS3MaterialConfiguration or type(candidate) is not NativeS3MaterialConfiguration:
            raise TypeError('native/S3 material configurations required')
        if type(history) is not FrictionPointState:raise TypeError('committed friction history required')
        if history.partner_key not in (None,self.partner_key):
            raise ValueError('native/S3 material-point partner history transfer is unsupported')
        old=self.source.evaluate(committed.positions,committed.position_low,committed.cell_rotations)
        new=self.source.surface_variations(candidate.positions,candidate.position_low,candidate.cell_rotations)
        old_target=self.target.physical_pose(committed.shell_displacement)
        target=self.target.physical_pose(candidate.shell_displacement)
        corners=self.target.adapter.current_positions(candidate.shell_displacement)
        frame=np.asarray(target.material_frame);normal=frame[:,2]
        relative=new.station.surface_position-np.asarray(target.position)
        gap=float(relative@normal)
        # Physical closest projection must remain in this owned target facet.
        edges=np.column_stack((corners[1]-corners[0],corners[2]-corners[0]))
        uv=np.linalg.lstsq(edges,new.station.surface_position-corners[0]-gap*normal,rcond=None)[0]
        if min(uv[0],uv[1],1-uv.sum())<=1e-12:
            raise ValueError('native/S3 material point projection requires interior target ownership')
        increment=objective_relative_tangent_increment(source_committed=old.surface_position,
            target_committed=old_target.position,source_candidate=new.station.surface_position,
            target_candidate=target.position,target_frame_committed=old_target.material_frame,
            target_frame_candidate=target.material_frame)
        old_gap=float((old.surface_position-np.asarray(old_target.position))@np.asarray(old_target.material_frame)[:,2])
        if gap<0 and old_gap>0 and np.linalg.norm(increment)>1e-14:
            raise ValueError('native/S3 material activation needs first-touch path subdivision')
        pressure=max(0.,-self.normal_stiffness_Pa_per_m*gap)
        trial=trial_coulomb_point(history,material_frame=frame,relative_increment=increment,
            normal_pressure=pressure,friction_coefficient=self.friction_coefficient,
            tangential_stiffness=self.tangential_stiffness_Pa_per_m,partner_key=self.partner_key)
        if getattr(self,'_history_state_only',False):
            count=42+self.target.adapter.work_map.shape[1]
            zero=_readonly(np.zeros(count));matrix=_readonly(np.zeros((count,count)))
            return NativeS3MaterialPointResult(zero,matrix,trial,gap,pressure,
                self.reference_weight_m2*.5*self.normal_stiffness_Pa_per_m*min(gap,0.)**2,
                zero,matrix)
        # Fifteen local coordinates: base translation, cell spatial rotation,
        # and three physical S3 vertices. Pull back once to authoritative DOFs.
        g,elastic=_local_contact_jets(new,corners,self.target.barycentric,
            old_target,old,history,self.partner_key)
        E=(.5*self.normal_stiffness_Pa_per_m*g*g if gap<0 else Jet2.constant(0.,15))
        residual=E.gradient.copy();tangent=E.hessian.copy()
        if pressure>0:
            e=np.array([v.value for v in elastic]);B=np.array([v.gradient for v in elastic])
            H=np.array([v.hessian for v in elastic]);traction=np.asarray(trial.traction)
            if trial.branch=='stick':
                dt=-self.tangential_stiffness_Pa_per_m*B
            else:
                magnitude=np.linalg.norm(e);direction=e/magnitude
                dp=-self.normal_stiffness_Pa_per_m*g.gradient
                dt=(-self.friction_coefficient*pressure/magnitude*(np.eye(3)-np.outer(direction,direction))@B
                    -self.friction_coefficient*np.outer(direction,dp))
            residual-=B.T@traction
            tangent-=np.einsum('i,ijk->jk',traction,H)+B.T@dt
        count=42+self.target.adapter.work_map.shape[1]
        mapping=np.zeros((15,count))
        for node,weight in ((self.source.cell,1-self.source.t),(self.source.cell+1,self.source.t)):
            mapping[:3,6*node:6*node+3]=weight*np.eye(3)
        mapping[3:6,18+3*self.source.cell:21+3*self.source.cell]=np.eye(3)
        mapping[6:,42:]=self.target.adapter.work_map.toarray()
        residual=self.reference_weight_m2*(mapping.T@residual)
        tangent=self.reference_weight_m2*(mapping.T@tangent@mapping)
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(tangent)):
            raise ValueError('nonfinite native/S3 material-point response')
        return NativeS3MaterialPointResult(_readonly(residual),_readonly(tangent),trial,gap,
            pressure,self.reference_weight_m2*E.value,
            _readonly(self.reference_weight_m2*(mapping.T@E.gradient)),
            _readonly(self.reference_weight_m2*(mapping.T@E.hessian@mapping)))
