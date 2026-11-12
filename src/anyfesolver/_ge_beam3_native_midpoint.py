"""Private single-element elastic native midpoint development owner.

No public finite-speed admission, static codec mutation or physical condensation.
Algebraic nodal-frame velocity slots are inertial placeholders only: stage and
endpoint nodal frames are independent solved coordinates and generally move.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import json
from threading import Lock, get_ident
import numpy as np

from ._ge_beam3_native_analysis import NativeBeamAnalysis
from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from ._ge_beam3_retained_generalized_state import Context, Program, Mechanical
from ._ge_beam3_dynamic_elastic_state import DynamicElasticContext, _admit_runtime_limit
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_centered_finite_inertia import (
    evaluate_centered_finite_inertia, PHYSICAL_DOFS, ALGEBRAIC_DOFS)
from ._ge_beam3_p5.algebra import rotation
from ._ge_beam3_p5.chart import exp_chart_terms
from ._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._native_reference_modal import _owned
from .control import cancellation_safe_point

IMPLEMENTATION_ID = 'DEVELOPMENT_C7_SINGLE_CENTERED_NATIVE_MIDPOINT_V1'


@dataclass(frozen=True)
class MidpointState:
    mechanical: Mechanical
    origins: tuple
    histories: tuple
    physical_velocity24: np.ndarray
    physical_acceleration24: np.ndarray
    time: float
    epoch: int
    predecessor_sha256: str
    context_sha256: str


@dataclass(frozen=True)
class StageEvaluation:
    mechanical: Mechanical
    residual: np.ndarray
    jacobian: np.ndarray
    configuration_map: np.ndarray
    physical_velocity24: np.ndarray
    physical_acceleration24: np.ndarray
    structural_residual: np.ndarray
    structural_jacobian: np.ndarray
    inertia: object
    material_sha256: str


@dataclass(frozen=True)
class MidpointCandidate:
    endpoint: MidpointState
    stage: StageEvaluation
    unknown42: np.ndarray
    dt: float
    predecessor_sha256: str
    iterations: int
    backtracks: int
    stage_residual_norm: float
    endpoint_algebraic_norm: float
    endpoint_reaction42: np.ndarray


@dataclass(frozen=True)
class CandidateStageAudit:
    """Immutable evidence snapshot; never a state/session acceptance credential."""
    context_sha256: str
    candidate_sha256: str
    stage_sha256: str
    predecessor_sha256: str
    predecessor_epoch: int
    endpoint_epoch: int
    dt: float
    stage: StageEvaluation
    unknown42: np.ndarray
    residual_scales42: np.ndarray
    stage_residual_norm: float
    stage_residual_tolerance: float
    session_provenance: bytes
    stage_coordinates: str
    increment42: np.ndarray


@dataclass(frozen=True)
class EndpointAccounting:
    state_sha256: str
    time: float
    epoch: int
    linear_momentum: np.ndarray
    linear_momentum_low: np.ndarray
    angular_momentum: np.ndarray
    angular_momentum_low: np.ndarray
    kinetic_energy: float
    elastic_energy: float
    total_energy: float
    elastic_consistency_witness: float
    resultant_compatibility18: np.ndarray
    rank_mass_checks: bytes


@dataclass(frozen=True)
class PhysicalLoadAccounting:
    context_sha256: str
    load_sha256: str
    external_load42: np.ndarray
    external_spatial_jacobian42: np.ndarray
    line_force_work: float
    applied_couple42: np.ndarray
    force: np.ndarray
    moment_about_origin: np.ndarray
    momentum_origin_high: np.ndarray
    momentum_origin_low: np.ndarray
    has_nonpotential_couple: bool


class MidpointContext:
    """Immutable issued states, ephemeral trials, and atomic accepted replacement.

    Every external operation runs under the exact analysis operation lock. One
    homogeneous six-component clamp remains the default legacy route. The
    explicit dynamic elastic route also admits zero supports, with the same
    spatial stage equations and a disposable shared structural work session.
    Legacy/default owners retain120s; explicit dynamic owners capture one bounded
    duration at construction. No operation resets that clock.
    """
    def __init__(self, analysis, *, inertia_order=24, max_iterations=12,
                 max_backtracks=8, structural_owner='legacy_clamp',
                 momentum_origin_high=(0., 0., 0.), momentum_origin_low=(0., 0., 0.),
                 load_pattern=None, stage_residual_tolerance=1e-11,
                 stage_coordinates='increment', runtime_limit_seconds=120.):
        if type(analysis) is not NativeBeamAnalysis:
            raise ValueError('exact standalone native analysis required')
        if (type(inertia_order) is not int or inertia_order not in (4, 8, 24)
                or type(max_iterations) is not int or not 1 <= max_iterations <= 12
                or type(max_backtracks) is not int or not 0 <= max_backtracks <= 8):
            raise ValueError('bounded exact midpoint settings')
        if type(structural_owner) is not str or structural_owner not in ('legacy_clamp', 'dynamic_elastic'):
            raise ValueError('explicit private midpoint structural owner required')
        if type(stage_residual_tolerance) not in (int, float):
            raise ValueError('positive finite midpoint stage residual tolerance required')
        try:
            self.stage_residual_tolerance = float(stage_residual_tolerance)
        except (OverflowError, ValueError):
            raise ValueError('positive finite midpoint stage residual tolerance required') from None
        if not np.isfinite(self.stage_residual_tolerance) or self.stage_residual_tolerance <= 0.:
            raise ValueError('positive finite midpoint stage residual tolerance required')
        self._captured_stage_residual_tolerance = self.stage_residual_tolerance
        if type(stage_coordinates) is not str or stage_coordinates not in ('increment', 'acceleration'):
            raise ValueError('explicit midpoint increment or acceleration coordinates required')
        if stage_coordinates == 'acceleration' and structural_owner != 'dynamic_elastic':
            raise ValueError('acceleration coordinates require the dynamic elastic owner')
        self.stage_coordinates = stage_coordinates
        self._captured_stage_coordinates = stage_coordinates
        self.runtime_limit_seconds = _admit_runtime_limit(runtime_limit_seconds)
        self._captured_runtime_limit = self.runtime_limit_seconds
        if structural_owner != 'dynamic_elastic' and self.runtime_limit_seconds != 120.:
            raise ValueError('extended runtime requires the dynamic elastic owner')
        self.structural_owner = structural_owner
        self._selected_owner = structural_owner
        self.momentum_origin_high = _owned(momentum_origin_high)
        self.momentum_origin_low = _owned(momentum_origin_low)
        if self.momentum_origin_high.shape != (3,) or self.momentum_origin_low.shape != (3,):
            raise ValueError('complete momentum origin pairs required')
        for high, low in zip(self.momentum_origin_high, self.momentum_origin_low):
            validate_pair(float(high), float(low))
        if structural_owner == 'legacy_clamp' and (np.any(self.momentum_origin_high) or np.any(self.momentum_origin_low)):
            raise ValueError('legacy midpoint retains its zero momentum origin')
        self._dynamic = (DynamicElasticContext(analysis, max_iterations=max_iterations,
            max_backtracks=max_backtracks, runtime_limit_seconds=self.runtime_limit_seconds)
            if structural_owner == 'dynamic_elastic' else None)
        self._work_session = None
        with analysis._operation():
            elements = analysis._elements
            if (len(elements) != 1 or type(elements[0]) is not NativeGeneralizedStaticElement
                    or type(elements[0].section) is not EllipsoidalGeneralizedSection
                    or len(analysis.model.mesh.nodes) != 3
                    or analysis.model.mesh.dof_manager.total_dofs != 18):
                raise ValueError('one centered generalized element/full 42 order required')
            if structural_owner == 'legacy_clamp' and analysis._admit_boundaries() != tuple(range(6)):
                raise ValueError('explicit homogeneous full clamp on first ordered node required')
            self.analysis = analysis
            self.element = elements[0]
            if load_pattern is not None and (structural_owner != 'dynamic_elastic'
                    or type(load_pattern) is not DistributedPattern):
                raise ValueError('exact constant distributed loading requires dynamic elastic owner')
            # Cloning regenerates signatures; it must never heal a stale or
            # forged supplied exact pattern into newly authorized loading.
            if load_pattern is not None:
                load_pattern.require(analysis.model.mesh)
            self.load_pattern = (None if load_pattern is None else
                DistributedPattern(LinePattern(load_pattern.line.rows), load_pattern.couples))
            if self.load_pattern is not None:
                load_pattern.require(analysis.model.mesh)
                if canonical(load_pattern) != canonical(self.load_pattern):
                    raise ValueError('supplied constant physical load changed during ownership')
                self.load_pattern.require(analysis.model.mesh)
            self._captured_load = canonical(self.load_pattern)
            self.inertia_order = inertia_order
            self.max_iterations = max_iterations
            self.max_backtracks = max_backtracks
            self._static = None
            if structural_owner == 'legacy_clamp':
                self._static = Context(analysis.model, Program((0.,),
                    DistributedPattern(LinePattern(()), ())))
                if (self._static.count != 42 or self._static.slots != [list(range(42))]
                        or self._static.nodal_count != 18):
                    raise ValueError('exact one-element canonical mixed ordering required')
                free = self._static.free
                self.fixed = self._static.fixed
                self._reference_mechanical = self._static.initial.mechanical
                self._reference_origins = self._static.initial.histories
                self.scale = self._static.scale
                self.length = self._static.length
            else:
                free = self._dynamic.free
                self.fixed = self._dynamic.fixed
                self._reference_mechanical = self._dynamic._reference.mechanical
                self._reference_origins = self._dynamic.origins
                self.scale, self.length = self._gate_inputs()
            self.free = np.frombuffer(np.asarray(free, dtype=int).tobytes(), dtype=int)
            self.physical_free = tuple(i for i in PHYSICAL_DOFS if i in self.free)
            self.algebraic_free = tuple(i for i in ALGEBRAIC_DOFS if i in self.free)
            expected = (15, 27) if not self.fixed else (12, 24)
            if (len(self.physical_free), len(self.algebraic_free)) != expected:
                raise ValueError('explicit physical/algebraic supported partition required')
            self._identity = sha(self._identity_data())
            self._current = None
            self._visible_current = None
            self._current_bytes = None
            self._issued = {}
            self._candidates = {}
            self._initial_checks = None
            self._transaction_lock = Lock()
            self._restore_checks = None
            self._restart_contract = None
            if structural_owner == 'dynamic_elastic':
                from ._ge_beam3_native_dynamic_restart import contract
                self._restart_contract = canonical(contract(self))
            self._captured_restart_contract = self._restart_contract

    @property
    def initial_checks(self):
        return None if self._initial_checks is None else json.loads(self._initial_checks)

    @property
    def current(self):
        return self._visible_current

    @property
    def restore_checks(self):
        return None if self._restore_checks is None else json.loads(self._restore_checks)

    def _identity_data(self):
        # The default route's payload is unchanged from its frozen V3 owner.
        data = dict(implementation=IMPLEMENTATION_ID,
            analysis=self.analysis.identity, inertia_order=self.inertia_order,
            iterations=self.max_iterations, backtracks=self.max_backtracks,
            physical=self.physical_free, algebraic=self.algebraic_free, free=self.free)
        if self.structural_owner == 'legacy_clamp':
            data['static_context'] = self._static.identity
        else:
            data.update(structural_owner='dynamic_elastic', dynamic_context=self._dynamic._identity,
                momentum_origin_high=self.momentum_origin_high, momentum_origin_low=self.momentum_origin_low)
            if self.load_pattern is not None:
                data['constant_load_pattern'] = self.load_pattern
        # Preserve exact default/legacy identity bytes; any explicit nondefault
        # numerical acceptance setting binds states, candidates and operations.
        if self.stage_residual_tolerance != 1e-11:
            data['stage_residual_tolerance'] = self.stage_residual_tolerance
        if self.stage_coordinates != 'increment':
            data['stage_coordinates'] = self.stage_coordinates
        return data

    def _gate_inputs(self):
        """Read actual gate factors from the guarded selected structural owner.

        The public aliases are informational and explicitly checked by _guard;
        they cannot replace these numerical acceptance inputs. The legacy
        identity payload and its owner's existing scaling remain unchanged.
        """
        if self.structural_owner == 'legacy_clamp':
            return self._static.scale, self._static.length
        positions = self._dynamic.reference_positions
        length = float(np.linalg.norm(positions[-1]-positions[0]))
        scales = np.ones(42)
        for node in range(3):
            scales[6*node+3:6*node+6] = length
        scales[18:30] = length
        return _owned(scales), length

    @contextmanager
    def _work_operation(self):
        if self.structural_owner == 'legacy_clamp':
            with self.analysis._operation():
                yield
        else:
            with self._dynamic.session() as session:
                self._work_session = session
                try:
                    yield
                finally:
                    self._work_session = None

    @contextmanager
    def _atomic_operation(self):
        # The analysis exit guard runs after the operation body. Keep tentative
        # publication private until that guard succeeds; restore all issuance
        # bookkeeping if it fails. A separate nonblocking lock prevents another
        # midpoint operation entering between analysis-lock release and rollback.
        if not self._transaction_lock.acquire(blocking=False):
            raise ValueError('midpoint context is already in use')
        snapshot = None
        try:
            try:
                with self._work_operation():
                    snapshot = (self._current, self._visible_current, self._current_bytes,
                        self._issued.copy(), self._candidates.copy(), self._initial_checks,
                        self._restore_checks)
                    self._guard_stage_coordinates()
                    self._guard_runtime_limit()
                    if self._selected_owner == 'dynamic_elastic':
                        from ._ge_beam3_native_dynamic_restart import require_contract
                        require_contract(self)
                    yield
                # A setting mutation during native/session exit cannot publish
                # a state under the former numerical acceptance contract.
                self._guard_stage_residual_tolerance()
                self._guard_stage_coordinates()
                self._guard_runtime_limit()
                if self._selected_owner == 'dynamic_elastic':
                    from ._ge_beam3_native_dynamic_restart import require_contract
                    require_contract(self)
                if self.runtime_limit_seconds != 120.:
                    # Exit and artifact validation may consume the final budget.
                    # Check the deadline after that work, before visible publication.
                    self._guard_runtime_limit()
                    self._dynamic._guard_deadline()
                self._visible_current = self._current
            except BaseException:
                if snapshot is not None:
                    (self._current, self._visible_current, self._current_bytes,
                     self._issued, self._candidates, self._initial_checks,
                     self._restore_checks) = snapshot
                raise
        finally:
            self._transaction_lock.release()

    def _guard_stage_residual_tolerance(self):
        if (type(self.stage_residual_tolerance) is not float
                or not np.isfinite(self.stage_residual_tolerance) or self.stage_residual_tolerance <= 0.
                or self.stage_residual_tolerance != self._captured_stage_residual_tolerance):
            raise ValueError('midpoint stage residual tolerance changed')

    def _guard_stage_coordinates(self):
        if (type(self.stage_coordinates) is not str
                or self.stage_coordinates != self._captured_stage_coordinates):
            raise ValueError('midpoint stage coordinates changed')

    def _guard_runtime_limit(self):
        if (type(self.runtime_limit_seconds) is not float
                or self.runtime_limit_seconds != self._captured_runtime_limit):
            raise ValueError('midpoint captured runtime changed')
        if self._selected_owner == 'dynamic_elastic':
            self._dynamic._guard_runtime_limit()
            if self._dynamic.runtime_limit_seconds != self.runtime_limit_seconds:
                raise ValueError('midpoint structural runtime mismatch')

    def _guard(self, token=None, stage='midpoint.guard'):
        cancellation_safe_point(token, stage)
        self.analysis._guard()
        self._guard_stage_residual_tolerance()
        self._guard_stage_coordinates()
        self._guard_runtime_limit()
        if self.structural_owner != self._selected_owner:
            raise ValueError('midpoint structural route changed')
        if self.structural_owner == 'legacy_clamp':
            self._static.guard()
        else:
            self._dynamic._guard()
        if self.load_pattern is not None:
            if self.structural_owner != 'dynamic_elastic' or type(self.load_pattern) is not DistributedPattern:
                raise ValueError('midpoint constant physical load changed')
            self.load_pattern.require(self.analysis.model.mesh)
        if canonical(self.load_pattern) != self._captured_load:
            raise ValueError('midpoint constant physical load changed')
        scale, length = self._gate_inputs()
        if canonical(dict(scale=self.scale, length=self.length)) != canonical(dict(scale=scale, length=length)):
            raise ValueError('midpoint acceptance scale/length changed')
        expected = sha(self._identity_data())
        if self._identity != expected or self.analysis._elements != (self.element,):
            raise ValueError('midpoint context definition/settings changed')

    def _require_current(self, state):
        record = self._issued.get(id(state))
        if (type(state) is not MidpointState or state is not self.current
                or record is None or record[0] is not state
                or canonical(state) != record[1] or record[1] != self._current_bytes
                or state.context_sha256 != self._identity):
            raise ValueError('current issued immutable midpoint predecessor required')
        return sha(state)

    def _structural(self, mechanical, origins):
        if self.structural_owner == 'dynamic_elastic':
            work = self._loaded_work(mechanical, origins)
            return work.residual, work.spatial_jacobian, work.material_sha256
        r, j, _, responses, _ = self._static.assemble(mechanical, 0., origins)
        for response, origin in zip(responses, origins):
            material = json.loads(response.material)
            if canonical(response.history) != canonical(origin):
                raise ValueError('midpoint first slice requires unchanged elastic history')
            if any(row['branch'] != 'ELASTIC'
                   or any(value != 0. for pair in row['dissipation'] for value in pair)
                   for row in material['stations']):
                raise ValueError('midpoint first slice requires strictly elastic stations')
        return r, j, sha([response.material.decode() for response in responses])

    def _loaded_work(self, mechanical, origins):
        work = self._work_session.work(mechanical, origins=origins,
            pattern=self.load_pattern, parameter=0. if self.load_pattern is None else 1.)
        a = list(ALGEBRAIC_DOFS)
        if (np.any(work.external_load42[a]) or np.any(work.external_spatial_jacobian42[a, :])
                or np.any(work.external_spatial_jacobian42[:, a])):
            raise ValueError('physical loading must have exactly zero algebraic work/tangent')
        return work

    def _inertia(self, mechanical, velocity, acceleration):
        kwargs = {}
        if self.structural_owner == 'dynamic_elastic':
            kwargs = dict(momentum_origin_high=self.momentum_origin_high,
                momentum_origin_low=self.momentum_origin_low)
        result = evaluate_centered_finite_inertia(self.element.operator.reference,
            self.analysis._inertias[self.element.element_id],
            mechanical.positions, mechanical.position_low,
            mechanical.cell_rotations[0], velocity, acceleration,
            order=self.inertia_order, **kwargs)
        a = list(ALGEBRAIC_DOFS)
        for matrix in (result.mixed_mass, result.mixed_velocity_derivative,
                       result.mixed_configuration_derivative):
            if np.any(matrix[a, :]) or np.any(matrix[:, a]):
                raise ValueError('algebraic inertia derivative must be exactly zero')
        if np.any(result.mixed_inertia[a]) or np.any(result.mixed_generalized_momentum[a]):
            raise ValueError('algebraic inertia must be exactly zero')
        return result

    def _block_checks(self, j, inertia):
        z = list(self.algebraic_free)
        p = list(self.physical_free)
        cz = j[np.ix_(z, z)]
        singular = np.linalg.svd(cz, compute_uv=False)
        tolerance = 64*np.finfo(float).eps*len(z)*singular[0]
        if not np.isfinite(singular).all() or singular[-1] <= tolerance:
            raise ValueError('endpoint algebraic C_z singular: no time step admitted')
        mass = inertia.mixed_mass[np.ix_(p, p)]
        if np.linalg.norm(mass-mass.T) > 64*np.finfo(float).eps*np.linalg.norm(mass):
            raise ValueError('physical mass symmetry contract failed')
        np.linalg.cholesky((mass+mass.T)/2)
        eigenvalues = np.linalg.eigvalsh((mass+mass.T)/2)
        return dict(algebraic_rank=len(z), algebraic_sigma_min=float(singular[-1]),
            algebraic_sigma_max=float(singular[0]),
            algebraic_condition=float(singular[0]/singular[-1]),
            physical_mass_min_eigenvalue=float(eigenvalues[0]),
            physical_mass_max_eigenvalue=float(eigenvalues[-1]),
            physical_mass_condition=float(eigenvalues[-1]/eigenvalues[0]))

    def _norm(self, residual, indices):
        scale, _ = self._gate_inputs()
        return float(np.linalg.norm((residual/scale)[list(indices)]))

    def _correction(self, mechanical, step):
        _, length = self._gate_inputs()
        nodal = step[:18].reshape(3, 6)
        return max(float(np.linalg.norm(nodal[:, :3]))/length,
            float(np.linalg.norm(nodal[:, 3:])), float(np.linalg.norm(step[18:24])),
            float(np.linalg.norm(step[24:]))/max(1., float(np.linalg.norm(mechanical.resultants))))

    def _solve_algebraic(self, mechanical, origins, token):
        if self.structural_owner == 'dynamic_elastic':
            state = self._work_session.trial_state(mechanical, origins=origins)
            result = self._work_session.consistent(state, pattern=self.load_pattern,
                parameter=0. if self.load_pattern is None else 1., cancellation_token=token)
            return (result.state.mechanical, result.work.residual,
                result.work.spatial_jacobian, result.residual_norm, result.work.material_sha256)
        z = list(self.algebraic_free)
        for iteration in range(self.max_iterations+1):
            self._guard(token, 'midpoint.algebraic')
            r, j, material = self._structural(mechanical, origins)
            inertia = self._inertia(mechanical, np.zeros(24), np.zeros(24))
            self._block_checks(j, inertia)
            step = np.zeros(42)
            step[z] = np.linalg.solve(j[np.ix_(z, z)], -r[z])
            norm = self._norm(r, z)
            if norm <= 1e-11 and self._correction(mechanical, step) <= 1e-11:
                return mechanical, r, j, norm, material
            if iteration == self.max_iterations:
                raise ValueError('endpoint algebraic iteration limit')
            for cut in range(self.max_backtracks+1):
                self._guard(token, 'midpoint.algebraic.trial')
                proposed = self._static.advance(mechanical, step*(2.**(-cut)))
                rt, _, _ = self._structural(proposed, origins)
                if self._norm(rt, z) < norm:
                    mechanical = proposed
                    break
            else:
                raise ValueError('endpoint algebraic residual backtracking failed')
        raise AssertionError('bounded algebraic loop')

    def _endpoint_acceleration(self, mechanical, origins, velocity):
        r, j, _ = self._structural(mechanical, origins)
        inertia = self._inertia(mechanical, velocity, np.zeros(24))
        checks = self._block_checks(j, inertia)
        p = list(self.physical_free)
        acceleration = np.zeros(24)
        acceleration[p] = np.linalg.solve(inertia.mixed_mass[np.ix_(p, p)],
            -(r+inertia.mixed_inertia)[p])
        dynamic = self._inertia(mechanical, velocity, acceleration)
        reaction = r+dynamic.mixed_inertia
        if self._norm(reaction, p) > 1e-11:
            raise ValueError('endpoint physical acceleration residual failed')
        return _owned(acceleration), _owned(reaction), checks

    def initialize(self, mechanical=None, physical_velocity24=None, *, time=0.,
                   cancellation_token=None):
        with self._atomic_operation():
            return self._initialize_state(mechanical, physical_velocity24, time=time,
                                          cancellation_token=cancellation_token)

    def _initialize_state(self, mechanical, physical_velocity24, *, time, cancellation_token):
        self._guard(cancellation_token, 'midpoint.initialize')
        if self.current is not None or type(time) is not float or not np.isfinite(time):
            raise ValueError('one finite typed midpoint initialization required')
        mechanical = self._reference_mechanical if mechanical is None else mechanical
        if type(mechanical) is not Mechanical:
            raise ValueError('exact retained mechanical input required')
        mechanical = self._make(mechanical.descriptor())
        velocity = np.zeros(24) if physical_velocity24 is None else np.asarray(physical_velocity24)
        if velocity.shape != (24,) or velocity.dtype != np.float64 or not np.isfinite(velocity).all():
            raise ValueError('finite binary64 physical velocity24 required')
        inactive = tuple(i for i in range(24) if i not in self.physical_free)
        if np.any(velocity[list(inactive)]):
            raise ValueError('fixed velocity/algebraic inertial placeholders must be zero')
        velocity = _owned(velocity)
        origins = self._reference_origins
        mechanical, _, _, _, _ = self._solve_algebraic(mechanical, origins, cancellation_token)
        acceleration, _, checks = self._endpoint_acceleration(mechanical, origins, velocity)
        state = MidpointState(mechanical, origins, origins, velocity, acceleration,
            time, 0, self._identity, self._identity)
        self._guard(cancellation_token, 'midpoint.initialize.commit')
        self._initial_checks = canonical(checks)
        self._issued[id(state)] = (state, canonical(state))
        self._current = state
        self._current_bytes = canonical(state)
        return state

    def _prepare_initialize(self, mechanical, physical_velocity24, *, time,
                            cancellation_token, prepare):
        """Trusted internal trajectory preparation before genesis publication."""
        from ._ge_beam3_native_dynamic_restart import _encode_body, require_contract
        with self._atomic_operation():
            state = self._initialize_state(mechanical, physical_velocity24, time=time,
                                           cancellation_token=cancellation_token)
            accounting = self._endpoint_accounting(state)
            load = self._physical_load_accounting(state.mechanical)
            native = _encode_body(self, state, header=require_contract(self))
            prepared = prepare(state, accounting, load, native)
            self._guard(cancellation_token, 'midpoint.trajectory_origin.ready')
            self._require_prepared_origin(state)
            return state, prepared

    def _require_prepared_origin(self, state):
        """Authenticate provisional origin while visible current stays fresh."""
        record = self._issued.get(id(state))
        if (type(state) is not MidpointState or state is not self._current
                or record is None or record[0] is not state
                or canonical(state) != record[1] or record[1] != self._current_bytes
                or state.context_sha256 != self._identity):
            raise ValueError('issued immutable prepared native origin required')

    def _stage(self, accepted, u, h, token=None):
        self._guard(token, 'midpoint.stage')
        self._require_current(accepted)
        u = np.asarray(u)
        if u.shape != (42,) or u.dtype != np.float64 or not np.isfinite(u).all() or np.any(u[list(self.fixed)]):
            raise ValueError('supported finite binary64 midpoint unknown42 required')
        p = list(self.physical_free)
        # The opt-in chart uses midpoint acceleration in physical slots only.
        # Algebraic frame/resultant slots retain their original increments.
        acceleration_chart = self.stage_coordinates == 'acceleration'
        physical_increment = np.array(u[:24], copy=True)
        if acceleration_chart:
            physical_increment[p] = h*accepted.physical_velocity24[p]+(.5*h*h)*u[p]
        s = {k: v.copy() for k, v in accepted.mechanical.descriptor().items()}
        a = np.eye(42)
        for node in range(3):
            first = 6*node
            for axis in range(3):
                terms = (float(s['positions'][node, axis]), float(s['position_low'][node, axis]))
                if acceleration_chart:
                    terms += (.5*h*float(accepted.physical_velocity24[first+axis]),
                              .25*h*h*float(u[first+axis]))
                else:
                    terms += (.5*float(u[first+axis]),)
                s['positions'][node, axis], s['position_low'][node, axis] = split_sum(terms)
            a[first:first+3, first:first+3] *= .5
            eta = u[first+3:first+6]
            s['nodal_frames'][node] = rotation(eta)@accepted.mechanical.nodal_frames[node]
            a[first+3:first+6, first+3:first+6] = exp_chart_terms(eta)[0]
        for cell in (0, 1):
            first = 18+3*cell
            half = .5*physical_increment[first:first+3]
            s['cell_rotations'][0, cell] = rotation(half)@accepted.mechanical.cell_rotations[0, cell]
            a[first:first+3, first:first+3] = .5*exp_chart_terms(half)[0]
        if acceleration_chart:
            a[:, p] *= .5*h*h
        s['resultants'][0] += u[24:]
        mechanical = self._make(s)
        velocity = np.zeros(24)
        acceleration = np.zeros(24)
        if acceleration_chart:
            velocity[p] = accepted.physical_velocity24[p]+(.5*h)*u[p]
            acceleration[p] = u[p]
        else:
            velocity[p] = u[p]/h
            acceleration[p] = 2*(velocity[p]-accepted.physical_velocity24[p])/h
        r, j, material = self._structural(mechanical, accepted.histories)
        inertia = self._inertia(mechanical, velocity, acceleration)
        vmap = np.zeros((42, 42))
        bmap = np.zeros((42, 42))
        vmap[p, p] = .5*h if acceleration_chart else 1/h
        bmap[p, p] = 1. if acceleration_chart else 2/h**2
        jacobian = (j+inertia.mixed_configuration_derivative)@a
        jacobian += inertia.mixed_velocity_derivative@vmap+inertia.mixed_mass@bmap
        residual = r+inertia.mixed_inertia
        if not np.isfinite(jacobian).all() or not np.isfinite(residual).all():
            raise ValueError('nonfinite midpoint stage')
        return StageEvaluation(mechanical, _owned(residual), _owned(jacobian), _owned(a),
            _owned(velocity), _owned(acceleration), _owned(r), _owned(j), inertia, material)

    def evaluate_stage(self, accepted, unknown42, *, dt, cancellation_token=None):
        """Evaluate the selected chart; physical slots are a_m or increments.

        Algebraic slots remain frame/resultant increments in both charts.
        The returned Jacobian differentiates the actual selected unknown42.
        """
        with self._atomic_operation():
            self._time_step(dt)
            return self._stage(accepted, unknown42, dt, cancellation_token)

    @staticmethod
    def _time_step(h):
        if (type(h) is not float or not np.isfinite(h) or h <= 0.
                or h*h == 0. or not np.isfinite(2/(h*h))):
            raise ValueError('positive finite typed midpoint dt required')

    def trial(self, accepted, *, dt, cancellation_token=None):
        with self._atomic_operation():
            return self._trial_state(accepted, dt, cancellation_token)

    def _trial_state(self, accepted, dt, cancellation_token):
        self._time_step(dt)
        self._guard(cancellation_token, 'midpoint.trial')
        predecessor = self._require_current(accepted)
        free = list(map(int, self.free))
        p = list(self.physical_free)
        u = np.zeros(42)
        acceleration_chart = self.stage_coordinates == 'acceleration'
        u[p] = (accepted.physical_acceleration24[p] if acceleration_chart
                else dt*accepted.physical_velocity24[p])
        backtracks = 0
        stage = self._stage(accepted, u, dt, cancellation_token)
        for iteration in range(self.max_iterations+1):
            # A successful line-search evaluation already belongs to the
            # exact next unknown and the same committed origin. Keep its
            # immutable work instead of assembling it a second time.
            # Validate ownership/cancellation at every Newton iteration,
            # including iterations which reuse that disposable stage.
            self._guard(cancellation_token, 'midpoint.stage')
            self._require_current(accepted)
            step = np.zeros(42)
            step[free] = np.linalg.solve(stage.jacobian[np.ix_(free, free)], -stage.residual[free])
            norm = self._norm(stage.residual, free)
            correction_step = np.array(step, copy=True)
            if acceleration_chart:
                correction_step[p] *= .5*dt*dt
            if norm <= self.stage_residual_tolerance and self._correction(stage.mechanical, correction_step) <= 1e-11:
                break
            if iteration == self.max_iterations:
                raise ValueError('midpoint stage iteration limit')
            for cut in range(self.max_backtracks+1):
                self._guard(cancellation_token, 'midpoint.stage.trial')
                proposed = u+2.**(-cut)*step
                next_stage = self._stage(accepted, proposed, dt, cancellation_token)
                backtracks += int(cut > 0)
                if self._norm(next_stage.residual, free) < norm:
                    u = proposed
                    stage = next_stage
                    break
            else:
                raise ValueError('midpoint stage residual backtracking failed')
        endpoint_step = np.zeros(42)
        endpoint_step[p] = (dt*accepted.physical_velocity24[p]+(.5*dt*dt)*u[p]
                            if acceleration_chart else u[p])
        if self.structural_owner == 'legacy_clamp':
            endpoint_mechanical = self._static.advance(accepted.mechanical, endpoint_step)
        else:
            high = np.array(accepted.mechanical.positions, copy=True)
            low = np.array(accepted.mechanical.position_low, copy=True)
            for node in range(3):
                for axis in range(3):
                    terms = (float(high[node, axis]), float(low[node, axis]))
                    if acceleration_chart:
                        terms += (dt*float(accepted.physical_velocity24[6*node+axis]),
                                  .5*dt*dt*float(u[6*node+axis]))
                    else:
                        terms += (float(u[6*node+axis]),)
                    high[node, axis], low[node, axis] = split_sum(terms)
            cells = np.array([rotation(endpoint_step[18+3*c:21+3*c])@accepted.mechanical.cell_rotations[0, c]
                for c in (0, 1)]).reshape(1, 2, 3, 3)
            endpoint_mechanical = self._make(dict(positions=high, position_low=low,
                nodal_frames=accepted.mechanical.nodal_frames, cell_rotations=cells,
                resultants=accepted.mechanical.resultants))
        # Stage algebraics are a starting guess, not accepted endpoint data.
        endpoint_mechanical = self._make(dict(
            positions=endpoint_mechanical.positions, position_low=endpoint_mechanical.position_low,
            nodal_frames=stage.mechanical.nodal_frames,
            cell_rotations=endpoint_mechanical.cell_rotations,
            resultants=stage.mechanical.resultants))
        endpoint_mechanical, _, _, alg_norm, _ = self._solve_algebraic(
            endpoint_mechanical, accepted.histories, cancellation_token)
        velocity = np.zeros(24)
        velocity[p] = (accepted.physical_velocity24[p]+dt*u[p] if acceleration_chart
                       else 2*stage.physical_velocity24[p]-accepted.physical_velocity24[p])
        acceleration, reaction, _ = self._endpoint_acceleration(
            endpoint_mechanical, accepted.histories, velocity)
        if not np.isfinite(accepted.time+dt) or accepted.time+dt <= accepted.time:
            raise ValueError('midpoint time increment is not representable')
        endpoint = MidpointState(endpoint_mechanical, accepted.histories, accepted.histories,
            _owned(velocity), acceleration, accepted.time+dt, accepted.epoch+1,
            predecessor, self._identity)
        candidate = MidpointCandidate(endpoint, stage, _owned(u), dt, predecessor,
            iteration, backtracks, norm, alg_norm, reaction)
        self._guard(cancellation_token, 'midpoint.trial.ready')
        self._require_current(accepted)
        self._candidates[id(candidate)] = (candidate, canonical(candidate))
        return candidate

    def _prepare_trial(self, accepted, *, dt, cancellation_token=None, prepare):
        """Trusted internal receipt; not a user callback or candidate commit."""
        with self._atomic_operation():
            candidate = self._trial_state(accepted, dt, cancellation_token)
            self._require_candidate(candidate)
            prepare(candidate)
            self._guard(cancellation_token, 'midpoint.trial.receipt_ready')
            self._require_candidate(candidate)
            return candidate

    def _discard_prepared_trial(self, candidate):
        """Clean this exact receipt, or no-op if native rollback removed it."""
        with self._atomic_operation():
            self._guard()
            if type(candidate) is not MidpointCandidate:
                raise ValueError('typed midpoint prepared trial receipt required')
            record = self._candidates.get(id(candidate))
            if record is None:
                return False
            self._require_candidate(candidate)
            del self._candidates[id(candidate)]
            return True

    def _require_candidate(self, candidate):
        record = self._candidates.get(id(candidate))
        if (type(candidate) is not MidpointCandidate or record is None
                or record[0] is not candidate or canonical(candidate) != record[1]
                or candidate.predecessor_sha256 != self._require_current(self.current)
                or candidate.endpoint.epoch != self.current.epoch+1):
            raise ValueError('issued current-predecessor immutable midpoint candidate required')
        return sha(candidate)

    def candidate_stage_audit(self, candidate, *, cancellation_token=None):
        """Snapshot issued candidate work before commit without reevaluation.

        Authentication and session exit complete before returning evidence.
        A stale/forged candidate never authenticates; the returned immutable
        snapshot has no operation authority and may remain as historical data.
        """
        with self._atomic_operation():
            self._guard(cancellation_token, 'midpoint.candidate_stage_audit')
            signature = self._require_candidate(candidate)
            if self._work_session is not None:
                self._work_session._check()
            owner = self._static.identity if self._static is not None else self._dynamic._identity
            provenance = canonical(dict(context_sha256=self._identity,
                structural_owner=self.structural_owner, work_owner_sha256=owner,
                audit_authentication_session=dict(thread=get_ident(),
                    session_object_id=(None if self._work_session is None else id(self._work_session)),
                    active_scope_authenticated=True, purpose='authenticate already issued candidate; not its original trial session'),
                evidence_only=True, stage_coordinates=self.stage_coordinates,
                predecessor_sha256=candidate.predecessor_sha256,
                predecessor_epoch=self.current.epoch, candidate_sha256=signature))
            scale, _ = self._gate_inputs()
            increment = np.array(candidate.unknown42, copy=True)
            if self.stage_coordinates == 'acceleration':
                p = list(self.physical_free)
                increment[p] = (candidate.dt*self.current.physical_velocity24[p]
                                +(.5*candidate.dt*candidate.dt)*candidate.unknown42[p])
            return CandidateStageAudit(self._identity, signature, sha(candidate.stage),
                candidate.predecessor_sha256, self.current.epoch, candidate.endpoint.epoch,
                candidate.dt, candidate.stage, _owned(candidate.unknown42), _owned(scale),
                candidate.stage_residual_norm, self.stage_residual_tolerance, provenance,
                self.stage_coordinates, _owned(increment))

    def commit(self, candidate, *, cancellation_token=None):
        with self._atomic_operation():
            self._guard(cancellation_token, 'midpoint.before_commit')
            self._require_candidate(candidate)
            state = candidate.endpoint
            self._issued[id(state)] = (state, canonical(state))
            self._current = state
            self._current_bytes = canonical(state)
            self._candidates.clear()
            return state

    def discard(self, candidate):
        with self._atomic_operation():
            self._guard()
            record = self._candidates.get(id(candidate))
            if record is None or record[0] is not candidate or canonical(candidate) != record[1]:
                raise ValueError('issued immutable midpoint candidate required')
            del self._candidates[id(candidate)]

    def create_checkpoint(self, *, cancellation_token=None):
        """Export only the authenticated committed state, with no pending trial."""
        from ._ge_beam3_native_dynamic_restart import encode
        with self._atomic_operation():
            self._guard(cancellation_token, 'midpoint.checkpoint.capture')
            raw = encode(self, self.current)
            self._guard(cancellation_token, 'midpoint.checkpoint.ready')
            return raw

    def restore_checkpoint(self, raw, *, expected_sha256, cancellation_token=None):
        """Fresh-only semantic validation and official atomic local reissuance.

        Stored q/v/a/material/time/lineage remain bit-exact. Newton correction is
        a witness, never an update. No ordinary initializer or static codec.
        """
        from ._ge_beam3_native_dynamic_restart import decode
        with self._atomic_operation():
            return self._restore_state(raw, expected_sha256, cancellation_token)

    def _restore_state(self, raw, expected_sha256, cancellation_token):
        from ._ge_beam3_native_dynamic_restart import decode
        self._guard(cancellation_token, 'midpoint.restore.capture')
        if (self._current is not None or self._visible_current is not None
                or self._current_bytes is not None or self._issued or self._candidates
                or self._initial_checks is not None or self._restore_checks is not None):
            raise ValueError('fresh never-initialized midpoint owner required for restore')
        if self.structural_owner != 'dynamic_elastic':
            raise ValueError('dynamic elastic midpoint restart only')
        dynamic = self._dynamic
        if len(dynamic._issued) != 1 or id(dynamic._reference) not in dynamic._issued:
            raise ValueError('fresh elastic work owner required for restore')
        dynamic._require(dynamic._reference)
        state, initial = decode(self, raw, expected_sha256)
        work = self._loaded_work(state.mechanical, state.histories)
        inertia = self._inertia(state.mechanical, state.physical_velocity24,
                                state.physical_acceleration24)
        rank_mass = self._block_checks(work.spatial_jacobian, inertia)
        z = list(self.algebraic_free)
        step = np.zeros(42)
        step[z] = np.linalg.solve(work.spatial_jacobian[np.ix_(z,z)], -work.residual[z])
        # Exact DynamicElasticContext._consistent392–402 normalization.
        correction = max(float(np.linalg.norm(step[[i for i in z if i < 24]])),
            float(np.linalg.norm(step[24:]))/max(1., float(np.linalg.norm(state.mechanical.resultants))))
        algebraic = self._norm(work.residual, z)
        physical = self._norm(work.residual+inertia.mixed_inertia, self.physical_free)
        if (not np.isfinite((correction, algebraic, physical)).all() or algebraic > 1e-11
                or correction > 1e-11 or physical > 1e-11):
            raise ValueError('stored dynamic endpoint residual/correction balance failed')
        if canonical(work.state.mechanical) != canonical(state.mechanical):
            raise ValueError('dynamic restore validation changed stored coordinates')
        checks = canonical(dict(state_sha256=sha(state), epoch=state.epoch, time=state.time,
            algebraic_residual_norm=algebraic, algebraic_correction_norm=correction,
            physical_balance_norm=physical, rank_mass=rank_mass,
            elastic_consistency_witness=abs(float(state.mechanical.resultants[0]@work.residual[24:42]))))
        self._guard(cancellation_token, 'midpoint.restore.ready')
        self._issued[id(state)] = (state, canonical(state))
        self._current = state
        self._current_bytes = canonical(state)
        self._initial_checks = initial
        self._restore_checks = checks
        return state

    def _prepare_restore(self, raw, *, expected_sha256, cancellation_token, prepare):
        """Trusted internal trajectory validation before restore publication."""
        with self._atomic_operation():
            state = self._restore_state(raw, expected_sha256, cancellation_token)
            accounting = self._endpoint_accounting(state)
            load = self._physical_load_accounting(state.mechanical)
            prepared = prepare(state, accounting, load, raw)
            self._guard(cancellation_token, 'midpoint.trajectory_restore.ready')
            self._require_prepared_origin(state)
            return state, prepared

    def _make(self, data):
        if self.structural_owner == 'legacy_clamp':
            return self._static.make(data)
        return self._work_session.make(data)

    def endpoint_accounting(self, state):
        """Actual endpoint invariants, not instantaneous derivative surrogates."""
        with self._atomic_operation():
            self._guard()
            self._require_current(state)
            return self._endpoint_accounting(state)

    def _endpoint_accounting(self, state):
        """Same endpoint formula; caller owns authentication and atomic scope."""
        if self.structural_owner != 'dynamic_elastic':
            raise ValueError('endpoint accounting requires explicit dynamic elastic owner')
        work = self._loaded_work(state.mechanical, state.histories)
        inertia = self._inertia(state.mechanical, state.physical_velocity24,
            state.physical_acceleration24)
        compatibility = work.residual[24:42]
        witness = abs(float(state.mechanical.resultants[0]@compatibility))
        total = float(inertia.kinetic_energy+work.mixed_potential)
        checks = self._block_checks(work.spatial_jacobian, inertia)
        return EndpointAccounting(sha(state), state.time, state.epoch,
            inertia.linear_momentum, inertia.linear_momentum_low,
            inertia.angular_momentum, inertia.angular_momentum_low,
            inertia.kinetic_energy, work.mixed_potential, total, witness,
            _owned(compatibility), canonical(checks))

    def candidate_endpoint_accounting(self, candidate, *, cancellation_token=None):
        """Authenticated disposable endpoint evidence; never commits a state."""
        with self._atomic_operation():
            self._guard(cancellation_token, 'midpoint.candidate_endpoint_accounting')
            self._require_candidate(candidate)
            result = self._endpoint_accounting(candidate.endpoint)
            self._guard(cancellation_token, 'midpoint.candidate_endpoint_accounting.ready')
            self._require_candidate(candidate)
            return result

    def prepare_candidate_checkpoint(self, candidate, *, cancellation_token=None):
        """Prepare bytes for an issued candidate; no acceptance or publication.

        Only committing this exact candidate makes the capsule a committed
        snapshot. Serialization and ordinary export use the same body encoder.
        """
        from ._ge_beam3_native_dynamic_restart import _encode_body, require_contract
        with self._atomic_operation():
            self._guard(cancellation_token, 'midpoint.candidate_checkpoint.capture')
            self._require_candidate(candidate)
            raw = _encode_body(self, candidate.endpoint, header=require_contract(self))
            self._guard(cancellation_token, 'midpoint.candidate_checkpoint.ready')
            self._require_candidate(candidate)
            return raw

    def physical_load_accounting(self, mechanical):
        """Pure current physical load view; not endpoint/time state acceptance.

        Nodal translations and physical cell spins carry the load; independent
        algebraic nodal frames/resultants do not. Fixed-origin moment recovery
        retains full compensated position pairs. Couples are nonpotential.
        """
        from fractions import Fraction
        with self._atomic_operation():
            self._guard()
            return self._physical_load_accounting(mechanical)

    def _physical_load_accounting(self, mechanical):
        """Existing load formula; caller owns guarded native operation scope."""
        from fractions import Fraction
        if self.structural_owner != 'dynamic_elastic' or type(mechanical) is not Mechanical:
            raise ValueError('explicit dynamic elastic physical mechanical load view required')
        work = self._loaded_work(mechanical, self._reference_origins)
        mechanical = work.state.mechanical
        loads = work.external_load42
        force = sum((loads[6*n:6*n+3] for n in range(3)), np.zeros(3))
        moment = [Fraction(0)]*3
        origin = [Fraction(float(h))+Fraction(float(l)) for h, l in zip(
            self.momentum_origin_high, self.momentum_origin_low)]
        for node in range(3):
            lever = [Fraction(float(h))+Fraction(float(l))-o for h, l, o in zip(
                mechanical.positions[node], mechanical.position_low[node], origin)]
            f = [Fraction(float(v)) for v in loads[6*node:6*node+3]]
            cross = (lever[1]*f[2]-lever[2]*f[1], lever[2]*f[0]-lever[0]*f[2],
                lever[0]*f[1]-lever[1]*f[0])
            for axis in range(3):
                moment[axis] += cross[axis]
        for cell in (0, 1):
            for axis in range(3):
                moment[axis] += Fraction(float(loads[18+3*cell+axis]))
        return PhysicalLoadAccounting(self._identity, work.load_sha256,
            loads, work.external_spatial_jacobian42, work.line_work, work.applied_couple42,
            _owned(force), _owned([float(v) for v in moment]), self.momentum_origin_high,
            self.momentum_origin_low, not work.conservative_potential)
