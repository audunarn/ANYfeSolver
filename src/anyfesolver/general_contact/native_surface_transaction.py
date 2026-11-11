"""Contact-enabled transaction/replay using the retained native state machinery.

No equilibrium driver is introduced here. Only converged, path-certified
contributions can advance the bound material/geometry origin atomically.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_p5_seeded.codec import MAX_BYTES, _load
from anyfesolver.control import cancellation_safe_point
from .friction_point import initial_friction_point
from .native_retained_station import _readonly
from .native_surface_contributor import NativeS3MaterialContributor
from .native_surface_path import affine_gap_polynomial
from .native_surface_s3_friction import NativeS3MaterialConfiguration
from .native_surface_patch import NativeS3MaterialSurfacePatch

SCHEMA = 'NATIVE_S3_MATERIAL_TRANSACTION_DEV_V2'


@dataclass(frozen=True)
class NativeS3MaterialAccepted:
    mechanical: object
    native_histories: tuple
    shell_displacement: np.ndarray
    friction_history: object
    parameter: float
    sequence: int


@dataclass(frozen=True)
class NativeS3MaterialProposal:
    origin_sha256: str
    native_increment: np.ndarray
    shell_increment: np.ndarray
    parameter: float
    assembly: object


def _payload(state):
    from .native_surface_history import NativeSurfaceReplayState
    history=(state.friction_history.descriptor() if type(state.friction_history) is NativeSurfaceReplayState
             else asdict(state.friction_history))
    return dict(mechanical=state.mechanical.descriptor(), native_histories=state.native_histories,
        shell_displacement=state.shell_displacement, friction_history=history,
        parameter=state.parameter, sequence=state.sequence)


class NativeS3MaterialTransaction:
    def __init__(self, contributor, *, maximum_records=256):
        if type(contributor) is not NativeS3MaterialContributor:
            raise TypeError('exact native material contributor required')
        if type(maximum_records) is not int or not 1 <= maximum_records <= 4096:
            raise ValueError('bounded positive material record count required')
        self.contributor = contributor
        self.maximum_records = maximum_records
        self.identity = sha(dict(schema=SCHEMA, contributor=contributor.identity,
                                 maximum_records=maximum_records))
        owner = contributor.owner; pair = contributor.pair; origin = owner.initial
        nodes = owner.context.nodes[0]
        pose = NativeS3MaterialConfiguration(origin.mechanical.positions[nodes], origin.mechanical.position_low[nodes],
                                             origin.mechanical.cell_rotations[0], origin.shell_displacement)
        history = (pair.initial_history(origin.shell_displacement,configuration=pose,
            maximum_intervals=owner.maximum_swept_intervals) if type(pair) is NativeS3MaterialSurfacePatch else
                   initial_friction_point(pair.target.physical_pose(origin.shell_displacement).material_frame))
        initial = contributor.evaluate(origin, history, np.zeros(owner.context.count),
                                       np.zeros(len(origin.shell_displacement)), 0.)
        pairs=pair.pairs if type(pair) is NativeS3MaterialSurfacePatch else (pair,)
        for point in pairs:
            gap, _ = affine_gap_polynomial(point, pose, np.zeros((3,3)), np.zeros((2,3)),
                                           np.zeros(len(origin.shell_displacement)))
            if gap < 0:
                raise ValueError('unresolved initial native material-surface penetration')
        continuous=getattr(initial.material.contribution,'continuous_gap',None)
        if continuous is not None and continuous.lower_bound<0:
            # New continuous profile requires a resolved nonnegative initial
            # gap. Roundoff uncertainty is not an interference allowance.
            raise ValueError('unresolved initial continuous native material-surface gap')
        self._require_equilibrium(initial)
        self._accepted = self._make_accepted(initial, 0., 0)
        self._origin_sha = sha(dict(identity=self.identity, state=_payload(self._accepted)))
        self._initial_sha = self._origin_sha
        self._records = ()

    @property
    def accepted(self):
        self._guard()
        context = self.contributor.owner.context
        state = self._accepted
        return NativeS3MaterialAccepted(context.make(state.mechanical.descriptor()),
            context.histories(state.native_histories), _readonly(state.shell_displacement),
            state.friction_history, state.parameter, state.sequence)

    @property
    def origin_sha256(self):
        self._guard()
        return self._origin_sha

    def _guard(self):
        self.contributor._guard()
        if (self.identity != sha(dict(schema=SCHEMA, contributor=self.contributor.identity,
                                     maximum_records=self.maximum_records))
                or self._origin_sha != sha(dict(identity=self.identity, state=_payload(self._accepted)))):
            raise ValueError('native material accepted origin/binding changed')

    def _require_equilibrium(self, assembly):
        owner = self.contributor.owner
        _, correction = owner._step(assembly.residual, assembly.spatial_tangent)
        if max(*assembly.metrics, correction) > owner.equilibrium_tolerance:
            raise ValueError('native material commit requires coupled equilibrium')
        if assembly.material.contribution.normal_gap_m < -owner.gap_tolerance:
            raise ValueError('native material commit gap tolerance')

    def _make_accepted(self, assembly, parameter, sequence):
        context = self.contributor.owner.context
        return NativeS3MaterialAccepted(context.make(assembly.mechanical.descriptor()),
            context.histories(tuple(response.history for response in assembly.native_responses)),
            _readonly(assembly.shell_displacement), assembly.material.contribution.trial.candidate_state,
            parameter, sequence)

    def propose(self, native_increment, shell_increment, parameter, *, cancellation_token=None):
        self._guard()
        cancellation_safe_point(cancellation_token, 'native material candidate')
        parameter = float(parameter)
        assembly = self.contributor.evaluate(self._accepted, self._accepted.friction_history,
                                             native_increment, shell_increment, parameter)
        cancellation_safe_point(cancellation_token, 'native material candidate complete')
        return NativeS3MaterialProposal(self._origin_sha, _readonly(native_increment),
                                        _readonly(shell_increment), parameter, assembly)

    def _capsule(self, records):
        if len(records) > self.maximum_records:
            raise MemoryError('native material checkpoint record cap')
        body = dict(schema=SCHEMA, identity=self.identity, initial_origin_sha256=self._initial_sha, records=records)
        raw = canonical(dict(**body, checkpoint_sha256=sha(body)))
        if len(raw) > MAX_BYTES:
            raise MemoryError('native material checkpoint byte cap')
        return raw

    def checkpoint(self):
        self._guard()
        return self._capsule(self._records)

    def commit(self, proposal, *, cancellation_token=None):
        self._guard()
        if type(proposal) is not NativeS3MaterialProposal or proposal.origin_sha256 != self._origin_sha:
            raise ValueError('native material proposal has stale/foreign origin')
        # Reevaluate from the issued physical origin. Candidate arrays/metrics
        # supplied by a caller never establish acceptance or material state.
        checked = self.propose(proposal.native_increment, proposal.shell_increment,
                               proposal.parameter, cancellation_token=cancellation_token)
        self._require_equilibrium(checked.assembly)
        staged = self._make_accepted(checked.assembly, checked.parameter, self._accepted.sequence+1)
        digest = sha(dict(identity=self.identity, state=_payload(staged)))
        record = dict(origin_sha256=self._origin_sha, native_increment=checked.native_increment,
            shell_increment=checked.shell_increment, parameter=checked.parameter,
            path_sha256=checked.assembly.material.path_sha256, state=_payload(staged), accepted_sha256=digest)
        # Shared canonical native encoding captures the material history records.
        record = _load(canonical(record).decode('ascii'))
        records = (*self._records, record)
        self._capsule(records)
        cancellation_safe_point(cancellation_token, 'native material atomic commit')
        self._accepted, self._origin_sha, self._records = staged, digest, records
        return self._accepted

    @classmethod
    def restore(cls, contributor, checkpoint, expected_sha256, *, maximum_records=256, cancellation_token=None):
        if (type(checkpoint) is not bytes or len(checkpoint) > MAX_BYTES
                or type(expected_sha256) is not str or sha256(checkpoint).hexdigest() != expected_sha256):
            raise ValueError('native material checkpoint external hash/extent')
        data = _load(checkpoint.decode('ascii'))
        if type(data) is not dict or set(data) != {'schema','identity','initial_origin_sha256','records','checkpoint_sha256'}:
            raise ValueError('native material checkpoint schema')
        body = {key:value for key,value in data.items() if key != 'checkpoint_sha256'}
        cancellation_safe_point(cancellation_token, 'native material restore')
        transaction = cls(contributor, maximum_records=maximum_records)
        if (data['schema'] != SCHEMA or data['identity'] != transaction.identity
                or data['initial_origin_sha256'] != transaction._initial_sha
                or data['checkpoint_sha256'] != sha(body)
                or type(data['records']) is not list or len(data['records']) > maximum_records):
            raise ValueError('native material checkpoint definition/settings binding')
        for record in data['records']:
            if type(record) is not dict or set(record) != {
                    'origin_sha256','native_increment','shell_increment','parameter','path_sha256','state','accepted_sha256'}:
                raise ValueError('native material checkpoint record schema')
            if record['origin_sha256'] != transaction.origin_sha256:
                raise ValueError('native material checkpoint origin chain')
            proposal = transaction.propose(np.asarray(record['native_increment']), np.asarray(record['shell_increment']),
                                           record['parameter'], cancellation_token=cancellation_token)
            transaction.commit(proposal, cancellation_token=cancellation_token)
            if canonical(transaction._records[-1]) != canonical(record):
                raise ValueError('native material checkpoint physical replay mismatch')
        if transaction.checkpoint() != checkpoint:
            raise ValueError('noncanonical native material checkpoint')
        return transaction
