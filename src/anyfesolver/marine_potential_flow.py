"""Native coupled potential-flow DEVELOPMENT contribution.

Only explicit development fixtures in the equilibrium h-gradient basis are
admitted. This is not a real-source/BEM qualification adapter. The base module
does not import Qt, Capytaine, xarray or ANYhydro. Radiation memory is accepted
velocity history in fixed equilibrium coordinates, independent of body rechart.
"""
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from anyfesolver.marine_contributions import (ContributionResult, DynamicContribution,
    HISTORY_PHYSICAL, MarineContributionsError, StepInformation)
from anyfesolver.marine_hydro_coordinates import (EquilibriumHydroFrame,
    MarineHydroCoordinatesError, evaluate_coupled)

API_VERSION = 'marine_potential_flow_development_v1'
DOFS = ('surge', 'sway', 'heave', 'roll', 'pitch', 'yaw')


class MarinePotentialFlowError(MarineContributionsError):
    """Malformed declaration, unknown physics/basis or invalid trial refused."""


def _numeric(value, shape, name):
    try:
        a = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MarinePotentialFlowError(f'{name}: invalid array') from error
    if a.shape != shape or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise MarinePotentialFlowError(f'{name}: finite real shape {shape} required')
    # Immutable backing store; flags cannot be reset to make it writable.
    return np.frombuffer(np.asarray(a, dtype=float).tobytes(), dtype=float).reshape(shape)


def _scalar(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise MarinePotentialFlowError(f'{name}: finite real scalar required')
    try:
        x = float(value)
    except (OverflowError, ValueError) as error:
        raise MarinePotentialFlowError(f'{name}: finite real scalar required') from error
    if not math.isfinite(x) or (x <= 0. if positive else x < 0.):
        raise MarinePotentialFlowError(f'{name}: invalid nonnegative/positive value')
    return x


def _frames(values):
    if not isinstance(values, (tuple, list)) or not values:
        raise MarinePotentialFlowError('explicit nonempty equilibrium frames required')
    frames = tuple(values)
    if any(not isinstance(f, EquilibriumHydroFrame) for f in frames):
        raise MarinePotentialFlowError('frames must be EquilibriumHydroFrame values')
    if len(set(f.body_id for f in frames)) != len(frames):
        raise MarinePotentialFlowError('duplicate equilibrium body identity')
    return frames


def equilibrium_geometry_hash(frames):
    """Stable exact binding to declared frames, not a backend geometry proof."""
    frames = _frames(frames)
    record = [{name: getattr(f, name) for name in f.__dataclass_fields__} for f in frames]
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, eq=False)
class PotentialFlowDevelopmentData:
    """Immutable explicit full coupled arrays; no symmetrization or regularization.

    baseline is reference buoyancy/preload in the h-gradient basis, excluding
    gravity. restoring is incremental buoyancy -dFh/dh. Exact kinematics never
    qualifies a large-motion potential-flow source. motion_bounds has one
    (translation_radius_metres, rotation_radius_radians) row per body. Real BEM
    source consumption needs a separate source-specific admission contract.
    """
    frames: tuple
    equilibrium_hash: str
    dt: float
    kernel: np.ndarray
    added_mass_infinite: np.ndarray
    restoring: np.ndarray
    baseline: np.ndarray
    excitation_real: np.ndarray
    excitation_imag: np.ndarray
    omega: float
    motion_bounds: np.ndarray
    source_identity: str
    source_kind: str = 'development_fixture'
    force_basis: str = 'equilibrium_h_gradient'
    hydrostatics_role: str = 'incremental_buoyancy'
    gravity_included: bool = False
    prehistory: str = 'zero_before_t0'
    excitation_series: object | None = None

    def __post_init__(self):
        frames = _frames(self.frames)
        object.__setattr__(self, 'frames', frames)
        if (not isinstance(self.equilibrium_hash, str)
                or self.equilibrium_hash != equilibrium_geometry_hash(frames)):
            raise MarinePotentialFlowError('equilibrium geometry binding changed or missing')
        declarations = {'source_kind':'development_fixture',
                        'force_basis':'equilibrium_h_gradient',
                        'hydrostatics_role':'incremental_buoyancy',
                        'prehistory':'zero_before_t0'}
        for name, expected in declarations.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value != expected:
                raise MarinePotentialFlowError(f'{name}: unsupported/unknown declaration')
        if self.gravity_included is not False:
            raise MarinePotentialFlowError('gravity must be excluded from fluid coefficients')
        if not isinstance(self.source_identity, str) or not self.source_identity.strip():
            raise MarinePotentialFlowError('explicit source identity required')
        object.__setattr__(self, 'dt', _scalar(self.dt, 'dt', positive=True))
        object.__setattr__(self, 'omega', _scalar(self.omega, 'omega'))
        n = 6*len(frames)
        try:
            shape = np.shape(self.kernel)
        except (ValueError, TypeError) as error:
            raise MarinePotentialFlowError('kernel: invalid array') from error
        if len(shape) != 3 or shape[0] < 2:
            raise MarinePotentialFlowError('kernel must contain at least two lag samples')
        object.__setattr__(self, 'kernel', _numeric(self.kernel, (shape[0], n, n), 'kernel'))
        for name in ('added_mass_infinite', 'restoring'):
            object.__setattr__(self, name, _numeric(getattr(self, name), (n, n), name))
        for name in ('baseline', 'excitation_real', 'excitation_imag'):
            object.__setattr__(self, name, _numeric(getattr(self, name), (n,), name))
        bounds = _numeric(self.motion_bounds, (len(frames), 2), 'motion_bounds')
        if np.any(bounds <= 0.) or np.any(bounds[:, 1] >= math.pi-1e-6):
            raise MarinePotentialFlowError('motion radii must be positive, rotation below log guard')
        object.__setattr__(self, 'motion_bounds', bounds)
        if self.excitation_series is not None:
            try:
                from anyhydro import ExcitationSeries, ExcitationSeriesError
            except ImportError as error:
                raise MarinePotentialFlowError('ANYhydro marine-excitation extra required') from error
            try:
                series = (self.excitation_series if type(self.excitation_series) is ExcitationSeries
                          else ExcitationSeries.from_dict(self.excitation_series))
            except (ExcitationSeriesError, TypeError, ValueError) as error:
                raise MarinePotentialFlowError('named public ANYhydro excitation series required') from error
            if series.source_qualification_status != 'development_fixture':
                raise MarinePotentialFlowError('only explicit development fixture excitation is admitted')
            if series.dof_keys != self.dof_keys:
                raise MarinePotentialFlowError('excitation series full DOF order must match equilibrium frames')
            if any(not np.array_equal(f.coefficient_axes, np.eye(3)) for f in frames):
                raise MarinePotentialFlowError('excitation series requires GLOBAL coefficient axes')
            if self.omega != 0. or np.any(self.excitation_real) or np.any(self.excitation_imag):
                raise MarinePotentialFlowError('series cannot duplicate legacy harmonic forcing')
            object.__setattr__(self,'excitation_series',series)

    @property
    def body_ids(self):
        return tuple(f.body_id for f in self.frames)

    @property
    def dof_keys(self):
        return tuple(f'{b}::{d}' for b in self.body_ids for d in DOFS)

    def with_excitation_series(self, series, *, development_only):
        """Use materialized public excitation; no real-source admission conferred.

        Legacy harmonic forcing is explicitly replaced, never added. The caller
        must bind physical geometry/environment separately before source use.
        """
        if development_only is not True:
            raise MarinePotentialFlowError('development_only=True required for series conversion')
        if series is None:
            raise MarinePotentialFlowError('a named excitation series is required; None cannot clear forcing')
        n = len(self.dof_keys)
        return replace(self, excitation_real=np.zeros(n), excitation_imag=np.zeros(n),
                       omega=0., excitation_series=series)

    @classmethod
    def from_radiation_kernel(cls, artifact, *, frames, equilibrium_hash, restoring,
                              baseline, excitation_real, excitation_imag, omega,
                              motion_bounds, development_only):
        """Snapshot a public ANYhydro kernel, explicitly for development only.

        No dataset hydrostatics are reinterpreted. Its hash is retained as source
        identity; its unqualified status is not changed or treated as acceptance.
        Caller declares the fixture geometry/force basis, not a real-source proof.
        """
        from anyhydro.radiation import RadiationKernel
        if development_only is not True or not isinstance(artifact, RadiationKernel):
            raise MarinePotentialFlowError('public RadiationKernel and development_only=True required')
        frames = _frames(frames)
        ids = tuple(f.body_id for f in frames)
        keys = tuple(f'{b}::{d}' for b in ids for d in DOFS)
        if artifact.body_ids != ids or artifact.dof_keys != keys:
            raise MarinePotentialFlowError('kernel body/DOF order must match the equilibrium frames exactly')
        return cls(frames, equilibrium_hash, artifact.settings.dt, artifact.kernel,
                   artifact.added_mass_infinite, restoring, baseline, excitation_real,
                   excitation_imag, omega, motion_bounds,
                   'ANYhydro RadiationKernel:'+artifact.content_hash())


@dataclass(frozen=True, eq=False)
class PotentialFlowDevelopmentContribution(DynamicContribution):
    """Whole-layout provider; one coupled fluid owner, immutable physical memory."""
    data: PotentialFlowDevelopmentData
    contribution_id: str = field(default='potential-flow-development', init=False)
    ownership_key: str = field(default='potential-flow-whole-layout', init=False)
    history_basis: str = field(default=HISTORY_PHYSICAL, init=False)
    history_transport = None

    def __post_init__(self):
        if not isinstance(self.data, PotentialFlowDevelopmentData):
            raise MarinePotentialFlowError('PotentialFlowDevelopmentData required')

    def evaluate_trial(self, committed_history, states, layout):
        raise MarinePotentialFlowError('potential flow requires explicit StepInformation')

    def _clock(self, information):
        if not isinstance(information, StepInformation) or information.dt != self.data.dt:
            raise MarinePotentialFlowError('explicit clock with exact kernel dt required')

    def _evaluation(self, states):
        try:
            evaluation = evaluate_coupled(self.data.frames, states)
        except MarineHydroCoordinatesError as error:
            raise MarinePotentialFlowError(str(error)) from error
        for block, bounds in zip(evaluation.blocks, self.data.motion_bounds):
            if (math.hypot(*block.coordinates[:3]) > bounds[0]
                    or math.hypot(*block.coordinates[3:]) > bounds[1]):
                raise MarinePotentialFlowError(f'{block.body_id}: outside declared development motion bounds')
        return evaluation

    def _memory(self, committed, information):
        self._clock(information)
        memory = committed.get(self.contribution_id)
        if information.phase == 'initialization':
            if memory is not None:
                raise MarinePotentialFlowError('initialization cannot reuse committed radiation memory')
            return ()
        previous = self._evaluation(information.previous_states).velocity
        if memory is None:
            if information.step_index != 1:
                raise MarinePotentialFlowError('radiation history is missing for a later step')
            return (previous,)
        if (not isinstance(memory, Mapping) or set(memory) != {'dt','step_index','velocities'}
                or type(memory['step_index']) is not int
                or memory['step_index'] != information.step_index-1
                or _scalar(memory['dt'], 'radiation memory dt', positive=True) != self.data.dt):
            raise MarinePotentialFlowError('radiation memory does not match the accepted clock')
        expected = min(information.step_index, self.data.kernel.shape[0]-1)
        velocities = _numeric(memory['velocities'], (expected, len(previous)), 'radiation memory')
        if not np.allclose(velocities[-1], previous, rtol=1e-12, atol=1e-12):
            raise MarinePotentialFlowError('radiation memory differs from the previous accepted velocity')
        return tuple(velocities)

    def prepare_step_with_context(self, committed_history, information):
        # Validation only. No mutable preparation state or history advancement.
        self._memory(committed_history, information)

    def validate_states(self, states):
        """Public instantaneous frame/bounds admission, without force or time advance."""
        return self._evaluation(states)

    def evaluate_static(self, states, layout):
        """Reference buoyancy/preload plus incremental restoring, without time.

        Gravity is excluded by the data contract. No excitation, radiation,
        added inertia, clock or committed/trial history is consumed here.
        Fixed equilibrium geometry and declared motion bounds remain binding.
        """
        from .marine_contributions import BodyLayout
        from .marine_rigid import RigidBodyState
        if not isinstance(layout,BodyLayout) or layout.body_ids != self.data.body_ids:
            raise MarinePotentialFlowError('static potential flow requires exact whole layout')
        if (not isinstance(states,Mapping) or set(states)!=set(layout.body_ids)
                or any(not isinstance(s,RigidBodyState) for s in states.values())):
            raise MarinePotentialFlowError('static potential flow requires exact immutable states')
        for state in states.values():
            if np.any(state.velocity) or np.any(state.acceleration):
                raise MarinePotentialFlowError('static potential flow requires stationary states')
        mapped = self._evaluation(states)
        restoring = -self.data.restoring@mapped.coordinates
        force = mapped.pullback_force(self.data.baseline+restoring,self.data.restoring)
        zeros = np.zeros((layout.size,layout.size))
        return ContributionResult(self.contribution_id,force.external_force,force.stiffness,
            zeros,zeros.copy(),{'status':'development_fixture_static_unqualified',
                'equilibrium_hash':self.data.equilibrium_hash,'source_identity':self.data.source_identity,
                'body_ids':self.data.body_ids,'components':{
                    'reference_baseline':mapped.jacobian.T@self.data.baseline,
                    'incremental_restoring':mapped.jacobian.T@restoring},
                'excluded':['gravity','wave_excitation','radiation','added_inertia']})

    def validate_accepted_history(self, memory, states, step_index):
        """Check the last accepted physical velocity/clock; never mutate history."""
        from anyfesolver.marine_contributions import CommittedHistory
        if type(step_index) is not int or step_index < 1:
            raise MarinePotentialFlowError('accepted history requires a positive completed step index')
        self._memory(CommittedHistory({self.contribution_id:memory}),
                     StepInformation(step_index+1,self.data.dt,'step',states))

    def evaluate_trial_with_context(self, committed_history, states, layout, information):
        self._clock(information)
        if layout.body_ids != self.data.body_ids:
            raise MarinePotentialFlowError('coupled potential flow requires the exact whole body layout')
        velocities = self._memory(committed_history, information)
        mapped = self._evaluation(states)
        n = layout.size
        radiation = np.zeros(n)
        endpoint = np.zeros((n, n))
        if information.phase == 'step':
            max_lag = min(information.step_index, self.data.kernel.shape[0]-1)
            endpoint = .5*self.data.dt*self.data.kernel[0]
            radiation = endpoint @ mapped.velocity
            for lag in range(1, max_lag+1):
                weight = .5 if lag == max_lag else 1.
                radiation += self.data.dt*weight*self.data.kernel[lag]@velocities[-lag]
        if self.data.excitation_series is None:
            phase = self.data.omega*information.time
            if not math.isfinite(phase):
                raise MarinePotentialFlowError('excitation phase overflow')
            excitation = self.data.excitation_real*math.cos(phase)+self.data.excitation_imag*math.sin(phase)
        else:
            from anyhydro import ExcitationSeriesError
            try:
                excitation = self.data.excitation_series.evaluate([information.time])[0]
            except ExcitationSeriesError as error:
                raise MarinePotentialFlowError('invalid prescribed excitation trial') from error
        restoring_force = -self.data.restoring@mapped.coordinates
        hydrostatic = self.data.baseline+restoring_force
        source_force = hydrostatic+excitation-radiation
        force = mapped.pullback_force(source_force, self.data.restoring)
        inertia = mapped.pullback_added_inertia(self.data.added_mass_infinite)
        D = mapped.jacobian
        velocity_q = np.zeros((n, n))
        for i, block in enumerate(mapped.blocks):
            base = 6*i
            v = np.asarray(states[block.body_id].velocity)
            for j in range(3):
                velocity_q[base:base+6, base+3+j] = block.jacobian_derivatives[j]@v
        stiffness = force.stiffness+inertia.stiffness+D.T@endpoint@velocity_q
        damping = inertia.damping+D.T@endpoint@D
        components = {'reference_baseline':D.T@self.data.baseline,
                      'incremental_restoring':D.T@restoring_force, 'excitation':D.T@excitation,
                      'radiation':-D.T@radiation, 'added_inertia':-inertia.internal_force}
        external = force.external_force-inertia.internal_force
        if not all(np.isfinite(a).all() for a in (external, stiffness, damping, inertia.mass)):
            raise MarinePotentialFlowError('nonfinite potential-flow force/tangent refused')
        history = None
        if information.phase == 'step':
            retained = (velocities+(mapped.velocity,))[-(self.data.kernel.shape[0]-1):]
            history = {'dt':self.data.dt, 'step_index':information.step_index,
                       'velocities':tuple(tuple(row) for row in retained)}
        return ContributionResult(self.contribution_id, external, stiffness, damping, inertia.mass,
            {'status':'development_fixture_unqualified', 'source_identity':self.data.source_identity,
             'equilibrium_hash':self.data.equilibrium_hash, 'body_ids':self.data.body_ids,
             **({} if self.data.excitation_series is None else {'excitation_series_hash':self.data.excitation_series.content_hash,
                'wave_realization_hash':self.data.excitation_series.wave_realization_hash,
                'excitation_source_hash':self.data.excitation_series.hydro_content_hash}),
             'time':information.time, 'components':components,
             'geometric_inertia_external':-inertia.geometric_force,
             'world_wrench':force.world_wrench-np.concatenate([
                 block.world_rate_map.T@(self.data.added_mass_infinite@mapped.acceleration)[6*i:6*i+6]
                 for i, block in enumerate(mapped.blocks)])}, history)


def data_to_dict(data):
    """Encode only the trusted named data type; never erase subclass physics."""
    if type(data) is not PotentialFlowDevelopmentData or any(
            type(frame) is not EquilibriumHydroFrame for frame in data.frames):
        raise MarinePotentialFlowError('only exact named potential-flow data/frames can be serialized')
    result = {'api_version':API_VERSION}
    for name in data.__dataclass_fields__:
        value = getattr(data,name)
        if name == 'excitation_series':
            if value is None:
                continue
            value = value.to_dict()
        if name == 'frames':
            value = [{field:np.asarray(getattr(frame,field)).tolist()
                      if field != 'body_id' else frame.body_id
                      for field in frame.__dataclass_fields__} for frame in value]
        elif isinstance(value,np.ndarray):
            value = value.tolist()
        result[name] = value
    return result


def data_from_dict(data):
    """Strict data-only reconstruction; module/class names are never executed."""
    required = (set(PotentialFlowDevelopmentData.__dataclass_fields__)-{'excitation_series'})|{'api_version'}
    if (not isinstance(data,dict) or data.get('api_version') != API_VERSION
            or not required <= set(data) or not set(data) <= required|{'excitation_series'}):
        raise MarinePotentialFlowError('potential-flow fields/version must exactly match the named schema')
    values = {key:value for key,value in data.items() if key != 'api_version'}
    fields = set(EquilibriumHydroFrame.__dataclass_fields__)
    if (not isinstance(values['frames'],(tuple,list)) or not values['frames']
            or any(not isinstance(f,dict) or set(f) != fields for f in values['frames'])):
        raise MarinePotentialFlowError('equilibrium frame fields must exactly match the named schema')
    try:
        values['frames'] = tuple(EquilibriumHydroFrame(**frame) for frame in values['frames'])
        return PotentialFlowDevelopmentData(**values)
    except (TypeError,ValueError,OverflowError) as error:
        raise MarinePotentialFlowError(f'invalid potential-flow data: {error}') from error
