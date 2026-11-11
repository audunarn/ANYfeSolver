"""Fixed native round material patch on one owned S3 target facet.

Shares point laws, structural assembly and transaction replay. No target-boundary
clipping or history transfer is implied by a fixed material quadrature rule.
"""
from dataclasses import dataclass,asdict

import numpy as np

from anyfesolver._ge_beam3_p5_seeded.core import sha
from .friction_point import initial_friction_point
from .native_retained_station import _readonly
from .native_surface_quadrature import NativeRoundSurfaceRule
from .native_surface_s3_friction import NativeS3MaterialPointFriction, NativeS3MaterialPathResult


@dataclass(frozen=True)
class NativeSurfacePatchHistory:
    definition_sha256: str
    points: tuple


@dataclass(frozen=True)
class NativeSurfacePatchTrial:
    candidate_state: NativeSurfacePatchHistory


@dataclass(frozen=True)
class NativeSurfacePatchResult:
    residual: np.ndarray
    tangent: np.ndarray
    trial: NativeSurfacePatchTrial
    normal_gap_m: float
    normal_storage_j: float
    points: tuple
    continuous_gap: object=None
    normal_integration_diagnostics: object=None
    friction_integral: object=None

    def spatial_tangent(self):
        return self.tangent


class NativeS3MaterialSurfacePatch:
    """One fixed reference-area rule with independent point histories."""
    def __init__(self, rule, targets, *, normal_stiffness_Pa_per_m,
                 friction_coefficient, tangential_stiffness_Pa_per_m,
                 activation_policy='reject',event_history_substeps=32,gap_check='sampled',
                 normal_integration='point',normal_integration_settings=None,
                 friction_integration='point',friction_integration_settings=None,maximum_replay_segments=256):
        if type(rule) is not NativeRoundSurfaceRule:
            raise TypeError('exact native reference surface rule required')
        if gap_check not in ('sampled','straight_patch'):
            raise ValueError('explicit supported patch gap check required')
        self.gap_check=gap_check
        if normal_integration not in ('point','straight_segmented'):
            raise ValueError('explicit supported native normal integration required')
        if normal_integration=='straight_segmented':
            from .native_surface_normal import StraightNormalIntegrationSettings
            if type(normal_integration_settings) is not StraightNormalIntegrationSettings or gap_check!='straight_patch':
                raise ValueError('segmented normal integration requires explicit settings and continuous straight gap check')
        elif normal_integration_settings is not None:
            raise ValueError('normal integration settings supplied for point integration')
        self.normal_integration=normal_integration
        self.normal_integration_settings=normal_integration_settings
        if friction_integration not in ('point','replay'):
            raise ValueError('explicit supported friction integration required')
        if friction_integration=='replay':
            from .native_surface_friction_integral import NativeSurfaceFrictionIntegrationSettings
            if (normal_integration!='straight_segmented' or
                    type(friction_integration_settings) is not NativeSurfaceFrictionIntegrationSettings):
                raise ValueError('replayed friction requires continuous normal integration and explicit settings')
            if type(maximum_replay_segments) is not int or maximum_replay_segments<1:
                raise ValueError('positive replay history segment cap required')
        elif friction_integration_settings is not None or maximum_replay_segments!=256:
            raise ValueError('unused replay settings on point integration')
        self.friction_integration=friction_integration
        self.friction_integration_settings=friction_integration_settings
        self.maximum_replay_segments=maximum_replay_segments
        targets=tuple(targets)
        if len(targets)!=len(rule.samples):
            raise ValueError('one explicit material target required per surface sample')
        pairs=tuple(NativeS3MaterialPointFriction(sample.station,target,
            reference_weight_m2=sample.reference_weight_m2,
            normal_stiffness_Pa_per_m=normal_stiffness_Pa_per_m,
            friction_coefficient=friction_coefficient,
            tangential_stiffness_Pa_per_m=tangential_stiffness_Pa_per_m,
            activation_policy=activation_policy,event_history_substeps=event_history_substeps)
            for sample,target in zip(rule.samples,targets))
        # This owner has one S3 domain. Require one exact prepared adapter;
        # material target coordinates can vary at every quadrature point.
        adapter=pairs[0].target.adapter
        if any(pair.target.adapter is not adapter for pair in pairs):
            raise ValueError('native material patch requires one prepared target adapter')
        self.rule,self.pairs=rule,pairs
        self.source,self.target=pairs[0].source,pairs[0].target
        self.definition_sha256=sha(self._definition())

    def _definition(self):
        result=dict(profile='NATIVE_S3_FIXED_MATERIAL_PATCH_DEV_V1',
                    rule=self.rule.identity,rule_definition=self.rule.definition,
                    points=[pair._definition() for pair in self.pairs])
        if self.gap_check!='sampled':result['gap_check']=self.gap_check
        if self.normal_integration!='point':
            result['normal_integration']=self.normal_integration
            result['normal_integration_settings']=asdict(self.normal_integration_settings)
        if self.friction_integration!='point':
            result['profile']='NATIVE_S3_REPLAY_MATERIAL_PATCH_DEV_V1'
            result['friction_integration']=self.friction_integration
            result['friction_integration_settings']=asdict(self.friction_integration_settings)
            result['maximum_replay_segments']=self.maximum_replay_segments
        return result

    def initial_history(self,shell_displacement,*,configuration=None,maximum_intervals=256):
        if self.friction_integration=='replay':
            from .native_surface_history import NativeSurfaceReplayState,NativeSurfaceHistoryField
            field=NativeSurfaceHistoryField(self,configuration,maximum_segments=self.maximum_replay_segments,
                                           maximum_intervals=maximum_intervals)
            if not np.array_equal(configuration.shell_displacement,shell_displacement):
                raise ValueError('initial replay shell configuration mismatch')
            return NativeSurfaceReplayState(self.definition_sha256,field.initial,(),maximum_intervals)
        return NativeSurfacePatchHistory(self.definition_sha256,tuple(
            initial_friction_point(pair.target.physical_pose(shell_displacement).material_frame)
            for pair in self.pairs))

    def evaluate_path(self,committed,history,translations,angular,shell_increment,*,maximum_intervals=256):
        if sha(self._definition())!=self.definition_sha256:
            raise ValueError('native surface patch definition changed')
        if self.friction_integration=='replay':
            return self._evaluate_replay(committed,history,translations,angular,shell_increment,maximum_intervals)
        if (type(history) is not NativeSurfacePatchHistory or
                history.definition_sha256!=self.definition_sha256 or len(history.points)!=len(self.pairs)):
            raise ValueError('native surface patch history binding')
        # Guard the material rule and all current surface orientations, then
        # evaluate every point from the same committed physical origin.
        self.rule.evaluate(committed.positions,committed.position_low,committed.cell_rotations)
        orientation_intervals=self.rule.certify_orientation_path(committed,translations,angular,
            shell_increment,maximum_intervals=maximum_intervals)
        values=tuple(pair.evaluate_path(committed,state,translations,angular,shell_increment,
                        maximum_intervals=maximum_intervals)
                     for pair,state in zip(self.pairs,history.points))
        current=values[0].configuration
        self.rule.evaluate(current.positions,current.position_low,current.cell_rotations)
        points=tuple(value.contribution for value in values)
        continuous=None; gap=min(p.normal_gap_m for p in points)
        if self.gap_check=='straight_patch':
            from .native_surface_extrema import straight_patch_s3_gap
            continuous=straight_patch_s3_gap(self.rule,self.target,current)
            gap=min(gap,continuous.lower_bound)
        residual=sum((p.residual for p in points),np.zeros_like(points[0].residual))
        tangent=sum((p.spatial_tangent() for p in points),np.zeros_like(points[0].spatial_tangent()))
        storage=sum(p.normal_storage_j for p in points); diagnostics=None
        if self.normal_integration=='straight_segmented':
            from .native_surface_normal import integrate_straight_normal
            normal=integrate_straight_normal(self.rule,self.target,current,
                self.pairs[0].normal_stiffness_Pa_per_m,self.normal_integration_settings)
            residual+=normal.residual-sum((p.normal_residual for p in points),np.zeros_like(residual))
            tangent+=normal.spatial_tangent()-sum((p.normal_spatial_tangent() for p in points),np.zeros_like(tangent))
            storage=normal.normal_storage_j;diagnostics=normal.diagnostics
        trial=NativeSurfacePatchTrial(NativeSurfacePatchHistory(self.definition_sha256,
                                     tuple(p.trial.candidate_state for p in points)))
        result=NativeSurfacePatchResult(_readonly(residual),_readonly(tangent),trial,
            gap,storage,points,continuous,diagnostics)
        return NativeS3MaterialPathResult(current,result,
            sha(dict(definition=self.definition_sha256,paths=[v.path_sha256 for v in values])),
            orientation_intervals+sum(v.certified_intervals for v in values))

    def _evaluate_replay(self,committed,history,translations,angular,shell_increment,maximum_intervals):
        from .native_surface_history import NativeSurfaceReplayState,NativeSurfaceMotionSegment,NativeSurfaceHistoryField
        from .native_surface_friction_integral import integrate_material_friction
        from .native_surface_normal import integrate_straight_normal
        from .native_surface_extrema import straight_patch_s3_gap
        if (type(history) is not NativeSurfaceReplayState or history.definition_sha256!=self.definition_sha256
                or maximum_intervals!=history.maximum_intervals):
            raise ValueError('native replay history/interval binding')
        if sha(asdict(history.current()))!=sha(asdict(committed)):
            raise ValueError('native replay physical origin mismatch')
        segment=NativeSurfaceMotionSegment(translations,angular,shell_increment)
        nonzero=any(np.any(v) for v in (segment.translations,segment.angular,segment.shell_increment))
        physical_segments=(*history.segments,segment) if nonzero else history.segments
        if len(physical_segments)>self.maximum_replay_segments:
            raise MemoryError('native replay physical history segment cap exhausted')
        orientation=self.rule.certify_orientation_path(committed,translations,angular,shell_increment,
                                                       maximum_intervals=maximum_intervals)
        # Even a zero candidate requires a new endpoint law from committed
        # history. Reusing the previous physical increment's tangent is wrong.
        field=NativeSurfaceHistoryField(self,history.initial,(*history.segments,segment),
            maximum_segments=self.maximum_replay_segments+1,maximum_intervals=maximum_intervals)
        # Only the immutable physical-origin prefix is reusable. The final
        # candidate segment and its endpoint law are always evaluated afresh,
        # including zero candidates and independent commit/restore validation.
        from collections import OrderedDict
        prefix_identity=sha(dict(definition=self.definition_sha256,initial=asdict(history.initial),
            segments=[asdict(v) for v in history.segments],maximum_intervals=maximum_intervals))
        if getattr(self,'_query_prefix_identity',None)!=prefix_identity:
            self._query_prefix_identity=prefix_identity
            self._query_prefix_cache=OrderedDict()
        reserve=self.friction_integration_settings.maximum_cache_bytes//8
        field._query_prefix_cache=self._query_prefix_cache
        field._query_prefix_capacity=reserve//4096
        field._query_prefix_reserved_bytes=field._query_prefix_capacity*4096
        friction=integrate_material_friction(field,self.friction_integration_settings)
        current=field.current
        normal=integrate_straight_normal(self.rule,self.target,current,
            self.pairs[0].normal_stiffness_Pa_per_m,self.normal_integration_settings)
        continuous=straight_patch_s3_gap(self.rule,self.target,current)
        candidate=NativeSurfaceReplayState(self.definition_sha256,history.initial,physical_segments,maximum_intervals)
        result=NativeSurfacePatchResult(_readonly(normal.residual+friction.residual),
            _readonly(normal.tangent+friction.tangent),NativeSurfacePatchTrial(candidate),
            continuous.lower_bound,normal.normal_storage_j,(),continuous,normal.diagnostics,friction)
        return NativeS3MaterialPathResult(current,result,
            sha(dict(definition=self.definition_sha256,origin=history.descriptor(),segment=asdict(segment))),
            orientation+friction.diagnostics['certified_intervals'])
