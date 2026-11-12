"""Material-coordinate replay of a bound native/S3 motion history.

Development infrastructure for changing friction integration points. No history
interpolation, quadrature adaptation, or mechanical acceptance is performed here.
The actual declared increments are retained, including spatial rotation paths.
"""
from dataclasses import dataclass, asdict

import numpy as np

from anyfesolver._ge_beam3_p5_seeded.core import sha
from .friction_point import initial_friction_point
from .native_retained_station import NativeRetainedRoundStation, _readonly
from .native_surface_path import material_path_pose
from .native_surface_quadrature import certify_station_orientation_path, surface_geometry
from .native_surface_s3_friction import (NativeS3MaterialConfiguration,
    NativeS3MaterialPathResult, NativeS3MaterialPointFriction)
from .s3_friction_point import S3SkinMaterialPoint


@dataclass(frozen=True)
class NativeSurfaceMotionSegment:
    translations: np.ndarray
    angular: np.ndarray
    shell_increment: np.ndarray

    def __post_init__(self):
        for name,shape in (('translations',(3,3)),('angular',(2,3)),('shell_increment',(18,))):
            value=np.asarray(getattr(self,name),dtype=float)
            if value.shape!=shape or not np.all(np.isfinite(value)):
                raise ValueError('finite native surface segment increments required')
            object.__setattr__(self,name,_readonly(value))


@dataclass
class NativeSurfaceReplayBudget:
    """Total query-initialization and path count, shared across all queries."""
    maximum_point_paths: int
    point_paths: int=0
    certified_intervals: int=0

    def __post_init__(self):
        if type(self.maximum_point_paths) is not int or self.maximum_point_paths<1:
            raise ValueError('positive replay point-path cap required')
        if self.point_paths!=0 or self.certified_intervals!=0:
            raise ValueError('new replay budget must start at zero')

    def consume(self):
        if self.point_paths>=self.maximum_point_paths:
            raise MemoryError('native surface total replay point-path budget exhausted')
        self.point_paths+=1


@dataclass(frozen=True)
class NativeSurfaceHistoryQuery:
    pair: object
    configuration: NativeS3MaterialConfiguration
    history: object
    last_contribution: object


@dataclass(frozen=True)
class NativeSurfaceReplayState:
    """Immutable physical motion history; transaction alone grants acceptance."""
    definition_sha256: str
    initial: NativeS3MaterialConfiguration
    segments: tuple
    maximum_intervals: int

    def descriptor(self):
        # Checkpoints already store each authoritative structural increment.
        # A chain digest avoids embedding every full prefix into every record;
        # restore reconstructs and checks the physical path by ordinary replay.
        digest=sha(asdict(self.initial))
        for segment in self.segments:digest=sha(dict(previous=digest,segment=asdict(segment)))
        return dict(profile='NATIVE_SURFACE_REPLAY_STATE_DEV_V1',definition=self.definition_sha256,
            path_sha256=digest,segments=len(self.segments),maximum_intervals=self.maximum_intervals)

    def current(self):
        pose=self.initial
        for segment in self.segments:
            pose=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,1.)
        return pose


class NativeSurfaceHistoryField:
    """Reference-projected material anchors and replayable physical path.

    The initial configuration and every actual increment bind this snapshot.
    Snapshots reconstructed from a transaction carry its accepted origin hash.
    Direct construction represents a declared path only, never equilibrium.
    """
    def __init__(self,patch,initial,segments=(),*,maximum_segments=256,maximum_intervals=256,
                 accepted_origin_sha256=None):
        from .native_surface_patch import NativeS3MaterialSurfacePatch
        if type(patch) is not NativeS3MaterialSurfacePatch or type(initial) is not NativeS3MaterialConfiguration:
            raise TypeError('exact patch and complete initial configuration required')
        if patch.gap_check!='straight_patch':
            raise ValueError('material replay initially requires continuous straight patch admission')
        if type(maximum_segments) is not int or maximum_segments<1:
            raise ValueError('positive replay segment cap required')
        if type(maximum_intervals) is not int or maximum_intervals<1:
            raise ValueError('positive replay path interval cap required')
        segments=tuple(segments)
        if len(segments)>maximum_segments:
            raise MemoryError('native surface history segment cap exceeded')
        if any(type(s) is not NativeSurfaceMotionSegment for s in segments):
            raise TypeError('declared native surface motion segments required')
        if accepted_origin_sha256 is not None and (
                not isinstance(accepted_origin_sha256,str) or len(accepted_origin_sha256)!=64):
            raise ValueError('accepted origin hash required')
        self.patch,self.initial,self.segments=patch,initial,segments
        self.maximum_segments=maximum_segments
        self.maximum_intervals=maximum_intervals
        self.accepted_origin_sha256=accepted_origin_sha256
        from .native_surface_extrema import straight_patch_s3_gap
        if straight_patch_s3_gap(patch.rule,patch.target,initial).lower_bound<0:
            raise ValueError('initial continuous history field penetration or unresolved gap')
        self.current=initial
        for segment in segments:
            self.current=material_path_pose(self.current,segment.translations,segment.angular,segment.shell_increment,1.)
        # Explicit anchor formula must reproduce supplied material partners.
        # It is fixed in initial Cartesian geometry, not in candidate geometry.
        for point in patch.pairs:
            anchor=self._target_at(point.source)
            if not np.allclose(anchor.barycentric,point.target.barycentric,rtol=0.,atol=32*np.finfo(float).eps):
                raise ValueError('existing material anchors differ from initial projection profile')
        self.identity=sha(self._definition())

    def _definition(self):
        return dict(profile='NATIVE_SURFACE_INITIAL_PROJECTION_REPLAY_DEV_V1',
            patch=self.patch._definition(),initial=asdict(self.initial),
            segments=[asdict(s) for s in self.segments],maximum_segments=self.maximum_segments,
            maximum_intervals=self.maximum_intervals,
            accepted_origin_sha256=self.accepted_origin_sha256)

    def _target_at(self,source):
        corners=self.patch.target.adapter.current_positions(self.initial.shell_displacement)
        edges=np.column_stack((corners[1]-corners[0],corners[2]-corners[0]))
        point=source.evaluate(self.initial.positions,self.initial.position_low,self.initial.cell_rotations).surface_position
        uv=np.linalg.lstsq(edges,point-corners[0],rcond=None)[0]
        return S3SkinMaterialPoint(self.patch.target.adapter,tuple((1-uv.sum(),*uv)))

    def pair_at(self,xi,angle):
        if sha(self._definition())!=self.identity:
            raise ValueError('native material history field definition changed')
        lo,hi=self.patch.rule.definition['xi_range']; alo,ahi=self.patch.rule.definition['angle_range']
        if not lo<float(xi)<hi or not alo<float(angle)<ahi:
            raise ValueError('history query must lie inside declared material patch')
        base=self.patch.pairs[0]
        source=NativeRetainedRoundStation(base.source.element_id,base.source.reference,float(xi),
            base.source.radius,float(angle))
        target=next((p.target for p in self.patch.pairs if p.source.identity==source.identity),None)
        if target is None:target=self._target_at(source)
        return NativeS3MaterialPointFriction(source,target,reference_weight_m2=1.,
            normal_stiffness_Pa_per_m=base.normal_stiffness_Pa_per_m,
            friction_coefficient=base.friction_coefficient,
            tangential_stiffness_Pa_per_m=base.tangential_stiffness_Pa_per_m,
            maximum_coupled_dofs=base.maximum_coupled_dofs,
            activation_policy=base.activation_policy,event_history_substeps=base.event_history_substeps)

    def _bind_whole_history_target_ownership(self):
        """Bind only the actual motions covered by this evaluation's domain proof."""
        from .native_surface_path import _ownership_motion_identity
        pose=self.initial;paths=[]
        for segment in self.segments:
            paths.append(_ownership_motion_identity(pose,segment.translations,segment.angular,
                segment.shell_increment,self.maximum_intervals))
            pose=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,1.)
        self._whole_history_target_paths=frozenset(paths)
        self._whole_history_target_proof=self.identity


    def _replay_increment(self,pair,pose,history,translations,angular,shell_increment,
            budget,maximum_intervals):
        """Bounded event-by-event replay of one declared increment.

        A refused multi-event increment is replayed as two dyadic substeps, each
        routed through the existing single-event branches. The declared physical
        path is unchanged; the shared point-path budget bounds the subdivision.
        """
        budget.consume()
        from .native_surface_path import _ownership_motion_identity
        lo,hi=self.patch.rule.definition['xi_range'];alo,ahi=self.patch.rule.definition['angle_range']
        domain=(pair.source.reference is self.patch.source.reference
            and pair.source.element_id==self.patch.source.element_id
            and pair.target.adapter is self.patch.target.adapter
            and lo<pair.source.xi<hi and alo<pair.source.angle<ahi)
        motion=_ownership_motion_identity(pose,translations,angular,shell_increment,maximum_intervals)
        if (getattr(self,'_whole_history_target_proof',None)==self.identity and domain
                and motion in getattr(self,'_whole_history_target_paths',())):
            from .native_surface_path import _ownership_path_identity
            pair._prepared_history_ownership=(_ownership_path_identity(pair,pose,translations,angular,
                shell_increment,maximum_intervals),self.identity)
        else:
            pair._prepared_history_ownership=None
        orientation=certify_station_orientation_path(pair.source,pose,translations,angular,
            shell_increment,maximum_intervals=maximum_intervals)
        try:
            value=pair.evaluate_path(pose,history,translations,angular,shell_increment,
                maximum_intervals=maximum_intervals)
        except ValueError as error:
            message=str(error)
            subdividable=('opening/closure or facet exit','first-touch physical ownership unresolved',
                'first-touch target area bound unresolved','first-touch transverse monotone path unresolved',
                'first-touch needs resolved open and closed endpoints','first-touch root residual unresolved')
            ordinary_resolution_request=(message ==
                'native/S3 material history path requires subdivision: path resolution exhausted')
            if (not (any(fragment in message for fragment in subdividable) or ordinary_resolution_request)
                    or pair.activation_policy not in ('transverse_events','transverse_event_sequence')):
                raise
            value=self._sequence_replay(pair,pose,history,translations,angular,shell_increment,
                maximum_intervals)
        budget.certified_intervals+=orientation+value.certified_intervals
        last=value.contribution
        return value.configuration,last,last.trial.candidate_state

    def _sequence_replay(self,pair,pose,history,translations,angular,shell_increment,
            maximum_intervals):
        """Certified event-by-event replay of one refused multi-event increment.

        The tight compensated slab certificates locate the complete alternating
        root schedule, including whole-patch facet ownership; each closed window
        is replayed through the existing release/first-touch endpoint laws. No
        history is dropped and no pulse is assumed negligible: an unresolved
        schedule refuses. This answers the subdivision demand of the ordinary
        point certificate without changing the declared physical path.
        """
        from .native_surface_event_sequence import (evaluate_open_event_sequence,
            evaluate_closed_event_sequence,evaluate_closed_or_persistent_sequence)
        end=material_path_pose(pose,translations,angular,shell_increment,1.)
        def gap_at(current):
            source=pair.source.evaluate(current.positions,current.position_low,
                current.cell_rotations).surface_position
            target=pair.target.physical_pose(current.shell_displacement)
            return float((source-np.asarray(target.position))@np.asarray(target.material_frame)[:,2])
        gap,next_gap=gap_at(pose),gap_at(end)
        if gap>0. and next_gap>0.:
            contribution=evaluate_open_event_sequence(pair,pose,history,translations,angular,
                shell_increment,maximum_intervals=maximum_intervals)
            return NativeS3MaterialPathResult(end,contribution,contribution.path_sha256,
                contribution.ownership_intervals)
        if gap>0.:
            contribution=evaluate_closed_event_sequence(pair,pose,history,translations,angular,
                shell_increment,maximum_intervals=maximum_intervals)
            return NativeS3MaterialPathResult(end,contribution,contribution.path_sha256,
                contribution.ownership_intervals)
        if gap<0. and next_gap>=0.:
            contribution=evaluate_open_event_sequence(pair,pose,history,translations,angular,
                shell_increment,maximum_intervals=maximum_intervals)
            return NativeS3MaterialPathResult(end,contribution,contribution.path_sha256,
                contribution.ownership_intervals)
        if gap<0.:
            contribution,sequence=evaluate_closed_or_persistent_sequence(pair,pose,history,
                translations,angular,shell_increment,maximum_intervals=maximum_intervals)
            return NativeS3MaterialPathResult(end,contribution,sequence.path_sha256,
                sequence.intervals)
        raise ValueError('native/S3 material history path requires subdivision: '
            'unresolved endpoint contact state')

    def query(self,xi,angle,*,budget,maximum_intervals=None):
        if type(budget) is not NativeSurfaceReplayBudget:
            raise TypeError('explicit shared material replay budget required')
        if maximum_intervals is None:maximum_intervals=self.maximum_intervals
        if (type(maximum_intervals) is not int or maximum_intervals<1
                or maximum_intervals>self.maximum_intervals):
            raise ValueError('replay path cap must not exceed bound history cap')
        budget.consume()  # Also bounds query preparation with an empty history.
        pair=self.pair_at(xi,angle); pose=self.initial
        cache=getattr(self,'_query_prefix_cache',None)
        capacity=getattr(self,'_query_prefix_capacity',0)
        key=(float(xi),float(angle),maximum_intervals)
        saved=cache.get(key) if cache is not None and capacity else None
        prefix=self.segments[:-1] if self.segments else ()
        if saved is None:
            surface_geometry(pair.source,pose.positions,pose.position_low,pose.cell_rotations)
            history=initial_friction_point(pair.target.physical_pose(pose.shell_displacement).material_frame)
            paths0=budget.point_paths; intervals0=budget.certified_intervals
            # Prefix contributions are discarded: retain their exact law,
            # root/transversality/ownership checks and physical state, without
            # building unused endpoint force Hessians. The final segment is
            # always a complete fresh contribution, even after exceptions.
            pair._history_state_only=True
            try:
                for segment in prefix:
                    pose,_,history=self._replay_increment(pair,pose,history,segment.translations,
                        segment.angular,segment.shell_increment,budget,maximum_intervals)
            finally:
                pair._history_state_only=False
            if cache is not None and capacity:
                # A conservative 4 KiB charge covers each small immutable pose,
                # state, key and bookkeeping record; no pair/tangent is cached.
                cache[key]=(pose,history,budget.point_paths-paths0,
                            budget.certified_intervals-intervals0)
                cache.move_to_end(key)
                while len(cache)>capacity:cache.popitem(last=False)
        else:
            pose,history,paths,intervals=saved
            cache.move_to_end(key)
            # Resource and certificate diagnostics retain the represented path
            # count even when its immutable prefix is reused.
            for _ in range(paths):budget.consume()
            budget.certified_intervals+=intervals
        last=None
        if self.segments:
            segment=self.segments[-1]
            pose,last,history=self._replay_increment(pair,pose,history,segment.translations,
                segment.angular,segment.shell_increment,budget,maximum_intervals)
        return NativeSurfaceHistoryQuery(pair,pose,history,last)

    @classmethod
    def from_transaction(cls,transaction):
        """Read the already accepted path; never mutate/recommit the owner."""
        from .native_surface_transaction import NativeS3MaterialTransaction
        if type(transaction) is not NativeS3MaterialTransaction:
            raise TypeError('exact native material transaction required')
        transaction._guard()
        owner=transaction.contributor.owner;origin=owner.initial;nodes=owner.context.nodes[0]
        initial=NativeS3MaterialConfiguration(origin.mechanical.positions[nodes],origin.mechanical.position_low[nodes],
            origin.mechanical.cell_rotations[0],origin.shell_displacement)
        segments=[]
        for record in transaction._records:
            local=np.asarray(record['native_increment'])[np.asarray(owner.context.slots[0],dtype=int)]
            segments.append(NativeSurfaceMotionSegment(np.array([local[6*i:6*i+3] for i in range(3)]),
                local[18:24].reshape(2,3),np.asarray(record['shell_increment'])))
        result=cls(transaction.contributor.pair,initial,segments,maximum_segments=transaction.maximum_records,
            maximum_intervals=owner.maximum_swept_intervals,
            accepted_origin_sha256=transaction.origin_sha256)
        current=transaction.accepted
        expected=NativeS3MaterialConfiguration(current.mechanical.positions[nodes],current.mechanical.position_low[nodes],
            current.mechanical.cell_rotations[0],current.shell_displacement)
        if sha(asdict(result.current))!=sha(asdict(expected)):
            raise ValueError('replay path does not bind accepted material configuration')
        return result
