"""Affine physical skin descriptor experiment; not a finite-rotation DAE lift."""
from contextlib import contextmanager
from dataclasses import dataclass, fields
import numpy as np
from .beam_s3_transient import BeamS3TransientManager
from .runtime import _ACTIVE_CONTACT
from .deformable_beam_s3_static import DeformableBeamS3Results

PROFILE = 'ELASTIC_B2_B3_AFFINE_ROTATIONAL_S3_ENDPOINT_HERMITE_DEV_V1'
RESTART_SCHEMA = 'ANYSOLVER_BEAM_AFFINE_ROTATIONAL_S3_DEV_CHECKPOINT_V18'


@dataclass(frozen=True)
class RotationalBeamS3Results(DeformableBeamS3Results):
    shell_nodal_work_moment: tuple[float, float, float]
    shell_affine_moment_discrepancy: tuple[float, float, float]
    moment_origin: str
    algebraic_acceleration_semantics: str
    represented_path: str
    touch_roundoff_m: float

    def to_dict(self):
        result=super().to_dict()
        result.update(shell_nodal_work_moment=list(self.shell_nodal_work_moment),
            shell_geometric_skin_moment=list(self.shell_moment),
            shell_affine_moment_discrepancy=list(self.shell_affine_moment_discrepancy),
            moment_origin=self.moment_origin,
            algebraic_acceleration_semantics=self.algebraic_acceleration_semantics,
            represented_path=self.represented_path,touch_roundoff_m=self.touch_roundoff_m)
        return result


class RotationalBeamS3Manager(BeamS3TransientManager):
    rotational_descriptor = True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._report_binding=(self.shell.director.tobytes(),self.shell.physical_offset,self.shell.side)

    def guard(self):
        if self.rotational_descriptor is not True:
            raise ValueError('rotational descriptor routing mutated')
        super().guard()
        if (self.shell.director.tobytes(),self.shell.physical_offset,self.shell.side)!=self._report_binding:
            raise ValueError('rotational descriptor report geometry mutated')

    def _positions(self, displacement):
        u=np.asarray(displacement,float).reshape(-1)
        if u.shape!=(self.work_map.shape[1],) or not np.isfinite(u).all():
            raise ValueError('rotational beam-S3 displacement invalid')
        return self.beam_reference+u[self.beam_dofs[:,:3]],self.shell.current_positions(u)

    def restart_contract(self):
        contract=super().restart_contract()
        contract.update(profile=PROFILE,revision='AFFINE_SKIN_TOTAL_VELOCITY_NULL_SECANT_V1')
        return contract

    def evaluate(self,*args,**kwargs):
        r,K,summary=super().evaluate(*args,**kwargs)
        u=np.asarray(args[0] if args else kwargs['displacement'],float)
        force=-r[self.shell_dofs[:,:3]]
        _,skin=self._positions(u)
        node=skin-self.shell.physical_offset*self.shell.director-self.shell.physical_offset*np.cross(u[self.shell_dofs[:,3:]],self.shell.director)
        work=(np.cross(node,force)-r[self.shell_dofs[:,3:]]).sum(axis=0)
        geometric=np.asarray(summary.shell_moment)
        data={f.name:getattr(summary,f.name) for f in fields(DeformableBeamS3Results)}
        data['profile']=PROFILE
        b,skin=self._positions(u)
        scale=max(float(np.max(abs(b))),float(np.max(abs(skin))),self.kernel.radius)
        result=RotationalBeamS3Results(**data,shell_nodal_work_moment=tuple(work),
            shell_affine_moment_discrepancy=tuple(geometric-work),moment_origin='GLOBAL_ORIGIN_0_M',
            algebraic_acceleration_semantics='NULL_ENDPOINT_VELOCITY_SECANT_NOT_CONTINUUM_DAE',
            represented_path='ALL_COORDINATE_ENDPOINT_HERMITE_AFFINE_SKIN',
            touch_roundoff_m=512*np.finfo(float).eps*scale)
        return r,K,result

    def wrench_report(self, displacement):
        """Separate geometric skin wrench from the affine nodal work wrench."""
        u=np.asarray(displacement,float)
        b,s=self._positions(u)
        state=self.kernel.evaluate(b,s)
        force=-state.gradient.reshape(-1,3)[len(b):]
        node=s-self.shell.physical_offset*self.shell.director
        rotation=u[self.shell_dofs[:,3:]]
        node=node-self.shell.physical_offset*np.cross(rotation,self.shell.director)
        couple=self.shell.physical_offset*np.cross(self.shell.director,force)
        geometric=np.cross(s,force).sum(axis=0)
        work=(np.cross(node,force)+couple).sum(axis=0)
        return dict(shell_force=force.sum(axis=0),geometric_skin_moment=geometric,
                    nodal_work_moment=work,affine_moment_discrepancy=geometric-work)


@contextmanager
def rotational_scope(model,definition,settings,initial_displacement=None):
    manager=RotationalBeamS3Manager(model,definition,settings,initial_displacement)
    token=_ACTIVE_CONTACT.set(manager)
    try:
        yield manager
    finally:
        _ACTIVE_CONTACT.reset(token)


def recover_rates(T,M,basis,tangent,velocity,acceleration,previous_velocity,dt,summary):
    """Total-tangent velocity; physical acceleration plus numerical null secant.

    No second force derivative or finite continuum algebraic acceleration is
    asserted. Minimum/full-line endpoint touch within dimensional roundoff
    refuses. Other isolated topology touches may have continuous H. No C2
    acceleration is claimed; unresolved outgoing full-line branches refuse.
    """
    if abs(summary.minimum_gap)<=summary.touch_roundoff_m:
        raise ValueError('rotational descriptor endpoint touch needs one-sided branch resolution')
    Z=basis.reduced_basis.toarray();gram=Z.T@Z
    K=(T.T@tangent@T).toarray();A=Z.T@K@Z
    def physical(value):
        return value-Z@np.linalg.solve(gram,Z.T@value)
    pv=physical(velocity)
    v=pv+Z@np.linalg.solve(A,-Z.T@K@pv)
    a=physical(acceleration)+Z@np.linalg.solve(gram,Z.T@(v-previous_velocity)/dt)
    if not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError('rotational descriptor nonfinite rate')
    if np.linalg.norm(Z.T@K@v)>1.e-9*max(1.,np.linalg.norm(K@v)):
        raise ValueError('rotational descriptor unresolved tangent velocity')
    if np.linalg.norm(M@(a-acceleration))>1.e-10*max(1.,np.linalg.norm(M@acceleration)):
        raise ValueError('rotational descriptor changed physical inertia')
    return v,a
