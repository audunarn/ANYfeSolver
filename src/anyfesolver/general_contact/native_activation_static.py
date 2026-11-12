"""Explicit activation-limited Newton globalization for retained native/S3 work.

The physical line law and increment transaction are unchanged. A small declared
activation budget limits Newton trials; it is not a changed penetration law.
"""
from dataclasses import dataclass, replace
from math import isfinite
import numpy as np
from anyfesolver.control import cancellation_safe_point
from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_multi_s3_static import NativeRetainedMultiS3Analysis, NativeMultiS3Run
from .native_s3_swept import native_s3_swept_admissible

PROFILE = 'NATIVE_B3_GE_MULTIPLE_S3_ACTIVATION_NEWTON_STATIC_DEV_V1'
SCHEMA = 'NATIVE_B3_GE_MULTIPLE_S3_ACTIVATION_CHECKPOINT_DEV_V1'


@dataclass(frozen=True)
class NativeContactActivationSettings:
    activation_gap_m: float
    maximum_geometry_queries: int

    def __post_init__(self):
        if (type(self.activation_gap_m) not in (int,float)
                or not isfinite(self.activation_gap_m) or self.activation_gap_m <= 0.):
            raise ValueError('native activation requires a positive dimensional gap budget')
        if type(self.maximum_geometry_queries) is not int or not 1 <= self.maximum_geometry_queries <= 512:
            raise ValueError('native activation requires a bounded geometry query cap')


@dataclass(frozen=True)
class NativeActivationS3Run(NativeMultiS3Run):
    profile: str = PROFILE
    # Cumulative activation-controller endpoint and prefix queries on this
    # owner; excludes regular physical sweeps and is not physical history.
    activation_queries: int = 0


class NativeActivationMultiS3Analysis(NativeRetainedMultiS3Analysis):
    profile = PROFILE
    schema = SCHEMA
    run_type = NativeActivationS3Run

    def __init__(self, *args, activation_settings, **kwargs):
        if type(activation_settings) is not NativeContactActivationSettings:
            raise TypeError('exact native contact activation settings required')
        activation_settings.__post_init__()
        self.activation_settings = activation_settings
        self.activation_total_queries = 0
        super().__init__(*args, **kwargs)
        budget = activation_settings.activation_gap_m
        if not max(p.work.contact.geometry_tolerance for p in self.pairs) < budget < min(p.gap_tolerance_m for p in self.pairs):
            raise ValueError('activation budget must exceed geometry tolerance and lie inside every pair gap budget')

    def _identity(self):
        return sha({'profile': PROFILE, 'revision': 'counted_open_endpoint_prefix_projected_normal_v2',
                    'work': super()._identity(), 'activation': self.activation_settings})

    def _activation_fraction(self, start, u_start, step, fraction, contact, cancellation_token):
        self._guard()
        settings = self.activation_settings
        opened = tuple(j for j, (_, r) in enumerate(contact.pairs) if r.contact.minimum_gap_m > 0.)
        if not opened: return fraction
        queries = 0
        def query(j, pose, u):
            nonlocal queries
            cancellation_safe_point(cancellation_token, 'native activation geometry')
            if queries >= settings.maximum_geometry_queries:
                raise RuntimeError('native activation geometry query budget exhausted')
            queries += 1; self.activation_total_queries += 1
            p=self.pairs[j]; i=self.beam_indices[j]; nodes=self.context.nodes[i]
            return p.work.contact.geometry_gap(pose.positions[nodes],pose.position_low[nodes],
                pose.cell_rotations[i],p.work.shell.current_positions(u))
        def sample(alpha):
            increment = alpha*step
            pose = self.context.advance(start, increment[:self.context.count])
            u = u_start+increment[self.context.count:]
            return pose,u,{j:query(j,pose,u) for j in opened}
        gap=settings.activation_gap_m
        while True:
            # Find a represented endpoint under the same finite query budget.
            try:
                end,u_end,values=sample(fraction)
            except (ValueError,MemoryError) as exc:
                if not self._recoverable_trial_error(exc): raise
                fraction *= .5
                if fraction <= np.finfo(float).eps:
                    raise RuntimeError('native activation geometry fraction unresolved') from exc
                continue
            minimum=min(values[j][0] for j in opened)
            if minimum < -.5*gap:
                left=0.; right=fraction
                while True:
                    middle=.5*(left+right)
                    if middle in (left,right): raise RuntimeError('native activation bracket unresolved')
                    end,u_end,values=sample(middle)
                    minimum=min(values[j][0] for j in opened)
                    if -.75*gap <= minimum <= -.25*gap:
                        fraction=middle
                        break
                    if minimum < -.5*gap: right=middle
                    else: left=middle
            # Open endpoints do not exclude hidden activation. Certify every
            # previously open pair, counting intermediate queries in this cap.
            certified=True
            for j in opened:
                p,r=contact.pairs[j]; i=self.beam_indices[j]
                a=r.contact; end_gap,end_rim=values[j]
                # Reserve queries for a shorter prefix if this certificate is
                # too conservative. A reduced interval budget cannot admit a
                # path; it only requests another cutback within the same cap.
                intervals=min(p.maximum_swept_intervals,
                    max(1,(settings.maximum_geometry_queries-queries)//2))
                if not native_s3_swept_admissible(self.context,p.work,i,gap,intervals,
                        self._recoverable_trial_error,start,u_start,fraction*step,end,u_end,
                        a.minimum_gap_m,end_gap,a.minimum_rim_clearance_m,end_rim,
                        geometry_only=True,projected_gap=True,
                        geometry_query=lambda pose,u,j=j:query(j,pose,u)):
                    certified=False
                    break
            if certified: return fraction
            fraction *= .5
            if fraction <= np.finfo(float).eps:
                raise RuntimeError('native activation prefix fraction unresolved')

    def _swept_admissible(self,start,u_start,increment,end,u_end,*_):
        self._guard()
        for p,i in zip(self.pairs,self.beam_indices):
            nodes=self.context.nodes[i]
            first=p.work.contact.geometry_gap(start.positions[nodes],start.position_low[nodes],
                start.cell_rotations[i],p.work.shell.current_positions(u_start))
            last=p.work.contact.geometry_gap(end.positions[nodes],end.position_low[nodes],
                end.cell_rotations[i],p.work.shell.current_positions(u_end))
            if not native_s3_swept_admissible(self.context,p.work,i,p.gap_tolerance_m,
                    p.maximum_swept_intervals,self._recoverable_trial_error,start,u_start,
                    increment,end,u_end,first[0],last[0],first[1],last[1],
                    geometry_only=True,projected_gap=True):
                return False
        return True

    def solve(self, **kwargs):
        result=super().solve(**kwargs)
        return replace(result,activation_queries=self.activation_total_queries)
