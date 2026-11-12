"""Private single centered generalized elastic structural/material work owner.

Admits zero supports or the reviewed full clamp without any static equilibrium
genesis inverse. No mass, velocity, time method, static-guard override or contact.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from threading import Lock, get_ident
from time import monotonic
import numpy as np

from ._ge_beam3_native_analysis import NativeBeamAnalysis
from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from ._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from ._ge_beam3_retained_generalized_state import Mechanical
from ._ge_beam3_native_generalized_restart import _history
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_fibre_line_work import evaluate as line_work
from ._ge_beam3_generalized_static_boundary import cell_couple_load, spatial_jacobian
from ._ge_beam3_operator_validation_scope import operator_call
from . import _ge_beam3_operator_validation_scope as _operator_validation
from ._ge_beam3_p5.arrays import _array, _frames
from ._ge_beam3_p5.algebra import rotation
from ._ge_beam3_p5.compensated_coordinates import validate_pair
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._native_reference_modal import _owned
from .control import cancellation_safe_point

IMPLEMENTATION_ID = 'DEVELOPMENT_C7_SINGLE_CENTERED_DYNAMIC_ELASTIC_STATE_V1'
PHYSICAL_DOFS = (0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 22, 23)
ALGEBRAIC_DOFS = (3, 4, 5, 9, 10, 11, 15, 16, 17, *range(24, 42))


def _admit_runtime_limit(value):
    if type(value) not in (int, float):
        raise ValueError('positive finite native dynamic runtime limit at most900 seconds required')
    try:
        duration = float(value)
    except (OverflowError, ValueError):
        raise ValueError('positive finite native dynamic runtime limit at most900 seconds required') from None
    if not np.isfinite(duration) or not 0. < duration <= 900.:
        raise ValueError('positive finite native dynamic runtime limit at most900 seconds required')
    return duration


@dataclass(frozen=True)
class DynamicElasticState:
    mechanical: Mechanical
    origins: tuple
    histories: tuple
    context_sha256: str


@dataclass(frozen=True)
class ElasticWorkEvaluation:
    state: DynamicElasticState
    residual: np.ndarray
    spatial_jacobian: np.ndarray
    conservative_value: float
    mixed_potential: float
    line_work: float
    applied_couple42: np.ndarray
    material_sha256: str
    load_sha256: str
    external_load42: np.ndarray
    external_spatial_jacobian42: np.ndarray
    conservative_potential: bool
    production_qualified: bool = False


@dataclass(frozen=True)
class AlgebraicConsistency:
    state: DynamicElasticState
    work: ElasticWorkEvaluation
    iterations: int
    backtracks: int
    residual_norm: float
    correction_norm: float
    algebraic_rank: int
    algebraic_sigma_min: float
    algebraic_sigma_max: float
    algebraic_condition: float


class AlgebraicRankError(ValueError):
    def __init__(self, singular, vector42, rank):
        super().__init__('fixed-physical algebraic block is singular; no temporal slice admitted')
        self.singular_values = _owned(singular)
        self.null_vector42 = _owned(vector42)
        self.rank = int(rank)


class ElasticWorkSession:
    """Authenticated same-thread disposable work scope; not state acceptance."""
    def __init__(self, context):
        self._context = context
        self._thread = get_ident()
        self._active = True
        self._issued = {}

    def _check(self):
        context = self._context
        if (not self._active or get_ident() != self._thread
                or context._active_session is not self):
            raise ValueError('active same-thread owner-bound elastic work session required')
        context._guard()

    def make(self, mechanical):
        self._check()
        return self._context._make(mechanical)

    def trial_state(self, mechanical, *, origins=None):
        self._check()
        context = self._context
        state = context._issue(mechanical, context.origins if origins is None else origins)
        context._evaluate(state, None, 0.)
        return state

    def evaluate(self, state, *, pattern=None, parameter=0., cancellation_token=None):
        self._check()
        return self._context._evaluate(state, pattern, parameter, cancellation_token)

    def work(self, mechanical, *, origins=None, pattern=None, parameter=0., cancellation_token=None):
        """Issue and validate one transient evaluation without duplicate work."""
        self._check()
        context = self._context
        state = context._issue(mechanical, context.origins if origins is None else origins)
        return context._evaluate(state, pattern, parameter, cancellation_token)

    def consistent(self, state, *, pattern=None, parameter=0., cancellation_token=None):
        self._check()
        return self._context._consistent(state, pattern, parameter, cancellation_token)


class DynamicElasticContext:
    """Exact-analysis-bound immutable elastic work and algebraic consistency.

    State issuance is not time/history acceptance. The material origin is the
    captured virgin elastic state; all evaluations must preserve it exactly.
    """
    def __init__(self, analysis, *, max_iterations=12, max_backtracks=8,
                 runtime_limit_seconds=120.):
        if type(analysis) is not NativeBeamAnalysis:
            raise ValueError('exact standalone native analysis required')
        if (type(max_iterations) is not int or not 1 <= max_iterations <= 12
                or type(max_backtracks) is not int or not 0 <= max_backtracks <= 8):
            raise ValueError('bounded exact elastic consistency settings')
        duration = _admit_runtime_limit(runtime_limit_seconds)
        with analysis._operation():
            elements = analysis._elements
            nodes = tuple(sorted(analysis.model.mesh.nodes))
            if (len(elements) != 1 or type(elements[0]) is not NativeGeneralizedStaticElement
                    or type(elements[0].operator) is not RetainedGeneralizedOperator
                    or type(elements[0].section) is not EllipsoidalGeneralizedSection
                    or type(elements[0].operator.reference) is not CenteredCurvedBeam3ReferenceGeometry
                    or len(nodes) != 3 or analysis.model.mesh.dof_manager.total_dofs != 18
                    or elements[0].node_ids != nodes
                    or tuple(elements[0].get_dof_mapping(analysis.model.mesh)) != tuple(range(18))):
                raise ValueError('exact one centered generalized element/canonical full42 order required')
            fixed = analysis._admit_boundaries()
            if fixed not in ((), tuple(range(6))):
                raise ValueError('zero supports or homogeneous first-node full clamp required')
            self.analysis = analysis
            self.element = elements[0]
            self.fixed = fixed
            self.free = tuple(i for i in range(42) if i not in fixed)
            self.physical_free = tuple(i for i in PHYSICAL_DOFS if i in self.free)
            self.algebraic_free = tuple(i for i in ALGEBRAIC_DOFS if i in self.free)
            self.reference_positions = _owned(self.element.operator.reference.coordinates)
            self.reference_frames = _owned(self.element.operator.reference.nodal_triads)
            self.origins = (self.element.operator.cell.virgin(),)
            self.max_iterations, self.max_backtracks = max_iterations, max_backtracks
            self.started = monotonic()
            self._captured_started = self.started
            self.runtime_limit_seconds = duration
            self._captured_runtime_limit = duration
            self._identity = sha(self._definition())
            self._issued = {}
            self._lock = Lock()
            self._active_session = None
            mechanical = self._make(dict(positions=self.reference_positions,
                position_low=np.zeros((3, 3)), nodal_frames=self.reference_frames,
                cell_rotations=np.tile(np.eye(3), (1, 2, 1, 1)), resultants=np.zeros((1, 18))))
            state = DynamicElasticState(mechanical, self.origins, self.origins, self._identity)
            self._reference = state
            self._issued[id(state)] = (state, canonical(state))

    def _definition(self):
        data = dict(implementation=IMPLEMENTATION_ID, analysis=self.analysis.identity,
            element=self.element.identity, operator=self.element.operator.identity,
            fixed=self.fixed, free=self.free, physical=self.physical_free,
            algebraic=self.algebraic_free, reference_positions=self.reference_positions,
            reference_frames=self.reference_frames, origins=self.origins,
            iterations=self.max_iterations, backtracks=self.max_backtracks)
        if self.runtime_limit_seconds != 120.:
            data['runtime_limit_seconds'] = self.runtime_limit_seconds
        return data

    def _guard_runtime_limit(self):
        if (type(self.runtime_limit_seconds) is not float
                or self.runtime_limit_seconds != self._captured_runtime_limit
                or type(self.started) is not float or self.started != self._captured_started):
            raise ValueError('dynamic elastic captured runtime/clock changed')
        if self.runtime_limit_seconds != 120. and _operator_validation._ACTIVE.get() is not None:
            raise ValueError('extended dynamic runtime excludes optional operator validation scope')

    def _guard_deadline(self):
        self._guard_runtime_limit()
        if monotonic()-self._captured_started > self._captured_runtime_limit:
            raise RuntimeError('dynamic elastic context deadline')

    def _guard(self, token=None, stage='dynamic-elastic.guard'):
        cancellation_safe_point(token, stage)
        self._guard_deadline()
        self.analysis._guard()
        if (sha(self._definition()) != self._identity
                or self.analysis._elements != (self.element,)
                or self.analysis._admit_boundaries() != self.fixed):
            raise ValueError('dynamic elastic frozen definition/supports changed')

    @contextmanager
    def _operation(self):
        if not self._lock.acquire(blocking=False):
            raise ValueError('dynamic elastic context is already in use')
        old = None
        try:
            try:
                with self.analysis._operation():
                    old = self._issued.copy()
                    self._guard()
                    yield
                    self._guard()
            except BaseException:
                if old is not None:
                    self._issued = old
                raise
        finally:
            self._lock.release()

    @contextmanager
    def session(self):
        if not self._lock.acquire(blocking=False):
            raise ValueError('dynamic elastic context is already in use')
        token = None
        old = None
        try:
            try:
                with self.analysis._operation():
                    old = self._issued.copy()
                    self._guard()
                    token = ElasticWorkSession(self)
                    self._active_session = token
                    try:
                        yield token
                        token._check()
                    finally:
                        # No standalone issuance is evicted. Scoped work states
                        # cease to authenticate before the native exit guard.
                        token._active = False
                        token._issued.clear()
                        self._active_session = None
                    self._guard()
            except BaseException:
                if old is not None:
                    self._issued = old
                raise
        finally:
            if token is not None:
                token._active = False
                token._issued.clear()
            self._active_session = None
            self._lock.release()

    def _make(self, data):
        if type(data) is Mechanical:
            data = data.descriptor()
        if type(data) is not dict or set(data) != set(Mechanical.__dataclass_fields__):
            raise ValueError('complete exact retained mechanical fields required')
        high = _array(data['positions'], (3, 3), 'physical positions')
        low = _array(data['position_low'], (3, 3), 'physical low positions')
        for h, l in zip(high.flat, low.flat):
            validate_pair(float(h), float(l))
        nodal = _frames(data['nodal_frames'], 3, 'algebraic nodal frames')
        cells = np.asarray(data['cell_rotations'])
        if cells.shape != (1, 2, 3, 3):
            raise ValueError('exact one-element physical cell frames')
        cells = _frames(cells.reshape(2, 3, 3), 2, 'physical cell frames').reshape(1, 2, 3, 3)
        resultants = _array(data['resultants'], (1, 18), 'algebraic resultants')
        for dof in self.fixed:
            node, axis = divmod(dof, 6)
            if axis < 3:
                if high[node, axis] != self.reference_positions[node, axis] or low[node, axis] != 0.:
                    raise ValueError('prescribed physical translation changed')
            elif not np.array_equal(nodal[node], self.reference_frames[node]):
                raise ValueError('prescribed algebraic frame changed')
        return Mechanical(*map(_owned, (high, low, nodal, cells, resultants)))

    def _origins(self, origins):
        if type(origins) is not tuple or len(origins) != 1:
            raise ValueError('complete one-element material origins required')
        validated = (_history(json.loads(canonical(origins[0])), self.element),)
        if canonical(validated) != canonical(self.origins):
            raise ValueError('first dynamic elastic slice requires captured virgin origins')
        return validated

    def _require(self, state):
        record = self._issued.get(id(state))
        if record is None and self._active_session is not None:
            self._active_session._check()
            record = self._active_session._issued.get(id(state))
        if (type(state) is not DynamicElasticState or record is None or record[0] is not state
                or state.context_sha256 != self._identity or canonical(state) != record[1]):
            raise ValueError('issued immutable analysis-bound elastic state required')
        return state

    def _issue(self, mechanical, origins):
        registry = self._issued
        if self._active_session is not None:
            self._active_session._check()
            registry = self._active_session._issued
        if len(registry) >= 128:
            raise ValueError('bounded elastic trial-state capacity')
        state = DynamicElasticState(self._make(mechanical), self._origins(origins),
            self._origins(origins), self._identity)
        registry[id(state)] = (state, canonical(state))
        return state

    def reference_state(self):
        with self._operation():
            return self._require(self._reference)

    def trial_state(self, mechanical, *, origins=None):
        with self._operation():
            state = self._issue(mechanical, self.origins if origins is None else origins)
            self._evaluate(state, None, 0.)
            return state

    def _evaluate(self, state, pattern, parameter, token=None):
        self._guard(token, 'dynamic-elastic.evaluate')
        self._require(state)
        if pattern is None:
            pattern = DistributedPattern(LinePattern(()), ())
        if type(pattern) is not DistributedPattern or type(parameter) is not float or not np.isfinite(parameter):
            raise ValueError('typed finite native distributed load required')
        pattern.require(self.analysis.model.mesh)
        captured_load = canonical(dict(pattern=pattern, parameter=parameter))
        m = self._make(state.mechanical)
        origins = self._origins(state.origins)
        op = self.element.operator
        arguments = (m.positions, m.position_low, m.nodal_frames, m.cell_rotations[0], m.resultants[0])
        if self.runtime_limit_seconds == 120.:
            response = operator_call(op, 'evaluate', self._guard, self.started,
                *arguments, origin=origins[0])
        else:
            # The optional local-validation scope owns a separate historical120s
            # contract. Extended owners reject that combination in the guard and
            # retain the ordinary full-model callback at every operator check.
            response = op.evaluate(*arguments, check=self._guard, origin=origins[0])
        material = json.loads(response.material)
        if (canonical(response.history) != canonical(origins[0])
                or any(row['branch'] != 'ELASTIC'
                    or any(v != 0. for pair in row['dissipation'] for v in pair)
                    for row in material['stations'])):
            raise ValueError('strict elastic unchanged material histories required')
        work = line_work(op.reference, m.positions, m.position_low, m.cell_rotations[0],
            parameter*pattern.force(self.element.element_id), order=op.order)
        conservative = response.residual-work.gradient
        couple = cell_couple_load(op, parameter*pattern.density(self.element.element_id))
        residual = conservative-couple
        tangent = spatial_jacobian(conservative, response.hessian+response.hessian_low-work.hessian)
        self._guard(token, 'dynamic-elastic.evaluated')
        if canonical(dict(pattern=pattern, parameter=parameter)) != captured_load:
            raise ValueError('elastic trial load changed')
        return ElasticWorkEvaluation(state, _owned(residual), tangent,
            float(response.potential-work.value), float(response.potential), float(work.value),
            _owned(couple), sha(material), sha(json.loads(captured_load)),
            _owned(work.gradient+couple), spatial_jacobian(work.gradient, work.hessian),
            not bool(np.any(couple)))

    def evaluate(self, state, *, pattern=None, parameter=0., cancellation_token=None):
        with self._operation():
            return self._evaluate(state, pattern, parameter, cancellation_token)

    def _algebraic_rank(self, tangent):
        z = list(self.algebraic_free)
        _, singular, vh = np.linalg.svd(tangent[np.ix_(z, z)])
        tolerance = 64*np.finfo(float).eps*len(z)*singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        if rank != len(z):
            null = np.zeros(42)
            null[z] = vh[-1]
            raise AlgebraicRankError(singular, null, rank)
        return len(z), float(singular[-1]), float(singular[0]), float(singular[0]/singular[-1])

    def algebraic_rank(self, state, *, cancellation_token=None):
        with self._operation():
            work = self._evaluate(state, None, 0., cancellation_token)
            rank, minimum, maximum, condition = self._algebraic_rank(work.spatial_jacobian)
            return dict(rank=rank, sigma_min=minimum, sigma_max=maximum, condition=condition)

    def _algebraic_move(self, state, step):
        # Physical positions and cell rotations are carried through bit-exactly.
        m = state.mechanical
        frames = np.array(m.nodal_frames, copy=True)
        for node in range(3):
            frames[node] = rotation(step[6*node+3:6*node+6])@m.nodal_frames[node]
        resultants = m.resultants+step[24:].reshape(1, 18)
        return self._issue(dict(positions=m.positions, position_low=m.position_low,
            nodal_frames=frames, cell_rotations=m.cell_rotations, resultants=resultants), state.origins)

    def consistent(self, state, *, pattern=None, parameter=0., cancellation_token=None):
        with self._operation():
            return self._consistent(state, pattern, parameter, cancellation_token)

    def _consistent(self, state, pattern, parameter, cancellation_token):
        self._require(state)
        z = list(self.algebraic_free)
        source_positions = state.mechanical.positions
        source_low = state.mechanical.position_low
        source_cells = state.mechanical.cell_rotations
        backtracks = 0
        length = float(np.linalg.norm(self.reference_positions[-1]-self.reference_positions[0]))
        for iteration in range(self.max_iterations+1):
            work = self._evaluate(state, pattern, parameter, cancellation_token)
            rank, minimum, maximum, condition = self._algebraic_rank(work.spatial_jacobian)
            step = np.zeros(42)
            step[z] = np.linalg.solve(work.spatial_jacobian[np.ix_(z, z)], -work.residual[z])
            # Same retained row convention: nodal moment residual / length;
            # axial/first-cell compatibility rows / length, remaining rows1.
            scales = np.ones(42)
            for node in range(3):
                scales[6*node+3:6*node+6] = length
            scales[18:30] = length
            norm = float(np.linalg.norm((work.residual/scales)[z]))
            correction = max(float(np.linalg.norm(step[[i for i in z if i < 24]])),
                float(np.linalg.norm(step[24:]))/max(1., float(np.linalg.norm(state.mechanical.resultants))))
            if norm <= 1e-11 and correction <= 1e-11:
                if (not np.array_equal(state.mechanical.positions, source_positions)
                        or not np.array_equal(state.mechanical.position_low, source_low)
                        or not np.array_equal(state.mechanical.cell_rotations, source_cells)):
                    raise ValueError('algebraic consistency changed physical coordinates')
                return AlgebraicConsistency(state, work, iteration, backtracks, norm, correction,
                    rank, minimum, maximum, condition)
            if iteration == self.max_iterations:
                raise ValueError('dynamic elastic algebraic iteration limit')
            for cut in range(self.max_backtracks+1):
                self._guard(cancellation_token, 'dynamic-elastic.algebraic.trial')
                proposed = self._algebraic_move(state, step*(2.**(-cut)))
                trial = self._evaluate(proposed, pattern, parameter, cancellation_token)
                trial_norm = float(np.linalg.norm((trial.residual/scales)[z]))
                if trial_norm < norm:
                    state = proposed
                    backtracks += cut
                    break
            else:
                raise ValueError('dynamic elastic algebraic backtracking failed')
        raise AssertionError('bounded algebraic loop')
