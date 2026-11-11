"""Thin development facade for the existing contact-free native midpoint owner."""
import numpy as np

from ._ge_beam3_native_analysis import NativeBeamAnalysis
from ._ge_beam3_native_midpoint import MidpointContext
from ._ge_beam3_centered_finite_inertia import PHYSICAL_DOFS
from ._ge_beam3_retained_generalized_state import Mechanical
from ._ge_beam3_p5_seeded.core import sha
from ._native_reference_modal import _owned


def _array(value, shape, label):
    value = np.asarray(value)
    if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError('finite binary64 '+label+' required')
    return _owned(value)


class NativeDynamicAnalysis:
    """One owner lifetime; issued states/candidates are forwarded unchanged.

    Physical velocity15 order is node1 xyz, node2 xyz, node3 xyz, cell1
    spatial omega xyz, cell2 spatial omega xyz. Nodal frames are algebraic,
    with zero *inertial placeholders*, not constrained frame derivatives.
    Only a single centered elastic ellipsoid element with free supports or a
    homogeneous full first-node clamp is admitted by the existing owner.
    Restart admits only a complete committed snapshot into a matching fresh owner.
    """
    __slots__ = ('_native', '_context', '_binding', '_identity')

    def __init__(self, definitions, boundaries, *, profile_sha256, **settings):
        from .ge_beam3_native_dynamics import PROFILE_SHA256
        if type(profile_sha256) is not str or profile_sha256 != PROFILE_SHA256:
            raise ValueError('exact current native dynamics development profile required')
        native = NativeBeamAnalysis(definitions, boundaries)
        # Constructor admission/rank ownership precedes any physical genesis.
        context = MidpointContext(native, structural_owner='dynamic_elastic', **settings)
        self._native, self._context = native, context
        self._binding = (native, context, native.identity, context._identity)
        self._identity = sha(dict(profile_sha256=profile_sha256,
                                  native=native.identity, midpoint=context._identity))

    def _owner(self):
        native, context, native_id, context_id = self._binding
        if (self._native is not native or self._context is not context
                or native.identity != native_id or context._identity != context_id):
            raise ValueError('native dynamic owner binding changed')
        return context

    @property
    def identity(self):
        context = self._owner()
        # Provenance must reject actual model/settings drift, not just ID drift.
        # No delegated operation is nested inside this read-only owner scope.
        with context._atomic_operation():
            context._guard()
            return self._identity

    @property
    def current(self):
        return self._owner().current

    @property
    def initial_checks(self):
        return self._owner().initial_checks

    @property
    def restore_checks(self):
        return self._owner().restore_checks

    def initialize(self, positions=None, cell_rotations=None, physical_velocity15=None,
                   *, position_low=None, time=0., cancellation_token=None):
        """Initialize from physical coordinates, then solve actual algebraics.

        Positions/high and low have shape (3,3); cell frames shape (2,3,3).
        Frames must be proper SO(3), validated by the existing mechanical owner.
        Omitted positions/frames are the owned reference configuration.
        Histories/resultants and algebraic frame guesses are not public inputs.
        """
        context, mechanical, velocity24 = self._initial_inputs(
            positions, cell_rotations, physical_velocity15, position_low)
        # No outer operation lock: the composed owner authenticates atomically.
        return context.initialize(mechanical, velocity24, time=time,
                                  cancellation_token=cancellation_token)

    def _initial_inputs(self, positions, cell_rotations, physical_velocity15, position_low):
        context = self._owner()
        reference = context._reference_mechanical
        positions = reference.positions if positions is None else _array(positions, (3,3), 'positions')
        low = np.zeros((3,3)) if position_low is None else _array(position_low, (3,3), 'position_low')
        cells = reference.cell_rotations if cell_rotations is None else _array(cell_rotations, (2,3,3), 'cell_rotations')[None]
        velocity = np.zeros(15) if physical_velocity15 is None else _array(physical_velocity15, (15,), 'physical_velocity15')
        velocity24 = np.zeros(24)
        velocity24[list(PHYSICAL_DOFS)] = velocity
        mechanical = Mechanical(_owned(positions), _owned(low),
            _owned(reference.nodal_frames), _owned(cells), _owned(np.zeros((1,18))))
        return context, mechanical, velocity24

    def _prepare_initial(self, positions, cell_rotations, physical_velocity15, *,
                         position_low, cancellation_token, prepare):
        context, mechanical, velocity24 = self._initial_inputs(
            positions, cell_rotations, physical_velocity15, position_low)
        return context._prepare_initialize(mechanical, velocity24, time=0.,
            cancellation_token=cancellation_token, prepare=prepare)

    def _prepare_restore(self, checkpoint, *, expected_sha256,
                         cancellation_token, prepare):
        return self._owner()._prepare_restore(checkpoint, expected_sha256=expected_sha256,
            cancellation_token=cancellation_token, prepare=prepare)

    def trial(self, accepted, *, dt, cancellation_token=None):
        return self._owner().trial(accepted, dt=dt, cancellation_token=cancellation_token)

    def _prepare_trial(self, accepted, *, dt, cancellation_token=None, prepare):
        return self._owner()._prepare_trial(accepted, dt=dt,
            cancellation_token=cancellation_token, prepare=prepare)

    def _discard_prepared_trial(self, candidate):
        return self._owner()._discard_prepared_trial(candidate)

    def candidate_stage_audit(self, candidate, *, cancellation_token=None):
        return self._owner().candidate_stage_audit(candidate, cancellation_token=cancellation_token)

    def commit(self, candidate, *, cancellation_token=None):
        return self._owner().commit(candidate, cancellation_token=cancellation_token)

    def discard(self, candidate):
        return self._owner().discard(candidate)

    def endpoint_accounting(self, state):
        return self._owner().endpoint_accounting(state)

    def candidate_endpoint_accounting(self, candidate, *, cancellation_token=None):
        return self._owner().candidate_endpoint_accounting(candidate,
            cancellation_token=cancellation_token)

    def prepare_candidate_checkpoint(self, candidate, *, cancellation_token=None):
        return self._owner().prepare_candidate_checkpoint(candidate,
            cancellation_token=cancellation_token)

    def physical_load_accounting(self, mechanical):
        return self._owner().physical_load_accounting(mechanical)

    def restore_checkpoint(self, checkpoint, *, expected_sha256, cancellation_token=None):
        return self._owner().restore_checkpoint(checkpoint, expected_sha256=expected_sha256,
                                                cancellation_token=cancellation_token)

    def create_checkpoint(self, *, cancellation_token=None):
        return self._owner().create_checkpoint(cancellation_token=cancellation_token)
