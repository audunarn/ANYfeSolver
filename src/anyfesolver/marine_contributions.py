"""Generic dynamic-contribution trial/history owner (marine_contributions_v1).

A single owner admits dynamic contributions, hands every trial the SAME
committed history, and commits ALL contributions atomically after
convergence, finite checks and cancellation polling. Rejected trials,
failed or cancelled steps, stale trial tokens and duplicate or non-owner
commits can never change committed history. No physical interpretation
or source admission is conferred on an arbitrary developer contribution;
gravity is explicitly selected and never added to unknown material.
"""
import abc
import math
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
import numpy as np

API_VERSION = "marine_contributions_v1"

HISTORY_PHYSICAL = "physical"
HISTORY_CHART = "chart"
_VALID_BASES = (HISTORY_PHYSICAL, HISTORY_CHART)
RESERVED_DRY_MASS_PREFIX = "dry-mass:"


class MarineContributionsError(ValueError):
    """Malformed contribution inputs or refused admission/commit; no fallback."""


class MarineContributionsCancelled(MarineContributionsError):
    """Cancellation observed while polling during trial operations."""


@dataclass(frozen=True)
class BodyLayout:
    """Immutable ordered body layout of the assembled chart DOF vector."""
    body_ids: tuple

    def __post_init__(self):
        ids = tuple(self.body_ids)
        if (not ids or any(not isinstance(i, str) or not i.strip() for i in ids)
                or len(set(ids)) != len(ids)):
            raise MarineContributionsError("body ids must be unique nonempty strings")
        object.__setattr__(self, "body_ids", ids)

    @property
    def size(self):
        return 6*len(self.body_ids)

    def index(self, body_id):
        try:
            return 6*self.body_ids.index(body_id)
        except ValueError as error:
            raise MarineContributionsError(f"unknown body id {body_id!r}") from error


def freeze_value(value):
    """Snapshot finite value data recursively; no arbitrary mutable objects.

    Numeric arrays use an immutable bytes backing store, so callers cannot
    turn their write flag back on. Histories are values, never provider code.
    """
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise MarineContributionsError("history/diagnostic keys must be strings")
        return types.MappingProxyType({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc" or not np.isfinite(value).all():
            raise MarineContributionsError("history arrays must contain finite numeric values")
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, (tuple, list)):
        return tuple(freeze_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, complex, np.number)):
        if not np.isfinite(value):
            raise MarineContributionsError("history/diagnostic numbers must be finite")
        return value.item() if isinstance(value, np.number) else value
    raise MarineContributionsError(f"unsupported mutable history value {type(value).__name__}")


def _frozen_numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise MarineContributionsError(f"{name} must be a finite real array of shape {shape}")
    return freeze_value(np.asarray(array, dtype=float))


class CommittedHistory:
    """Immutable copied mapping; never an alias to provider data."""

    __slots__ = ("_mapping",)

    def __init__(self, mapping=None):
        if mapping is not None and not isinstance(mapping, Mapping):
            raise MarineContributionsError("committed history must be a mapping")
        object.__setattr__(self, "_mapping", freeze_value({} if mapping is None else mapping))

    def __setattr__(self, name, value):
        raise TypeError("committed history is immutable")

    def __getitem__(self, key):
        return self._mapping[key]

    def get(self, key, default=None):
        return self._mapping.get(key, default)

    def __contains__(self, key):
        return key in self._mapping

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def keys(self):
        return self._mapping.keys()

    def items(self):
        return self._mapping.items()

    def replaced(self, items):
        merged = dict(self._mapping)
        merged.update(dict(items))
        return CommittedHistory(merged)


@dataclass(frozen=True)
class ChartMove:
    """One body's accepted rechart: exact new anchor and old chart data."""
    body_id: str
    old_rotation_vector: tuple
    old_anchor: tuple
    new_anchor: tuple

    def __post_init__(self):
        if not isinstance(self.body_id, str) or not self.body_id:
            raise MarineContributionsError("chart move must identify its body")
        vector = _frozen_numeric(self.old_rotation_vector, (3,), "old rotation vector")
        object.__setattr__(self, "old_rotation_vector", tuple(vector))
        for name in ("old_anchor", "new_anchor"):
            matrix = _frozen_numeric(getattr(self, name), (3, 3), name)
            if (np.linalg.norm(matrix.T @ matrix - np.eye(3)) > 1e-12
                    or abs(np.linalg.det(matrix) - 1.) > 1e-12):
                raise MarineContributionsError("chart anchors must be proper SO(3) matrices")
            object.__setattr__(self, name, tuple(tuple(row) for row in matrix))


@dataclass(frozen=True)
class RechartInfo:
    """Immutable rechart description passed to declared transport callbacks."""
    moves: tuple

    def __post_init__(self):
        moves = tuple(self.moves)
        if any(not isinstance(move, ChartMove) for move in moves):
            raise MarineContributionsError("rechart moves must be ChartMove values")
        if len({move.body_id for move in moves}) != len(moves):
            raise MarineContributionsError("rechart body ids must be unique")
        object.__setattr__(self, "moves", moves)

    def for_body(self, body_id):
        for move in self.moves:
            if move.body_id == body_id:
                return move
        return None


@dataclass(frozen=True)
class ContributionResult:
    """One contribution's trial output; arrays are fresh copies."""
    contribution_id: str
    external_force: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    mass: np.ndarray
    diagnostics: dict
    trial_history: object = None


@dataclass(frozen=True)
class StepInformation:
    """Immutable fixed-grid clock and previous accepted-state snapshot.

    Initialization is index0/time0 with no previous accepted state. A normal
    step receives the actual accepted start state (including frame0 on step1).
    Radiation can seed t0 velocity from that state without committing an
    initialization trial or inferring time from provider call counts.
    """
    step_index: int
    dt: float
    phase: str
    previous_states: Mapping = field(default_factory=dict)
    time: float = field(init=False)
    previous_time: float = field(init=False)

    def __post_init__(self):
        from .marine_rigid import RigidBodyState
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise MarineContributionsError('step_index must be a nonnegative integer')
        if isinstance(self.dt, (bool, np.bool_)) or not isinstance(self.dt, (int, float)):
            raise MarineContributionsError('step dt must be positive and finite')
        try:
            delta = float(self.dt)
        except (ValueError, TypeError, OverflowError) as error:
            raise MarineContributionsError('step dt must be positive and finite') from error
        if not math.isfinite(delta) or delta <= 0.:
            raise MarineContributionsError('step dt must be positive and finite')
        if (not isinstance(self.phase, str) or self.phase not in ('initialization', 'step')
                or (self.phase == 'initialization') != (self.step_index == 0)):
            raise MarineContributionsError('phase must match initialization index0 or a positive step index')
        if not isinstance(self.previous_states, Mapping):
            raise MarineContributionsError('previous_states must be an immutable-state mapping')
        previous = dict(self.previous_states)
        if (any(not isinstance(key, str) or not key.strip() for key in previous)
                or any(not isinstance(value, RigidBodyState) for value in previous.values())
                or (self.phase == 'initialization' and previous)
                or (self.phase == 'step' and not previous)):
            raise MarineContributionsError('previous_states must contain accepted states only for a normal step')
        try:
            time = self.step_index*delta
            previous_time = max(self.step_index-1, 0)*delta
        except OverflowError as error:
            raise MarineContributionsError('step clock exceeds finite range') from error
        if not math.isfinite(time) or not math.isfinite(previous_time):
            raise MarineContributionsError('step clock exceeds finite range')
        object.__setattr__(self, 'dt', delta)
        object.__setattr__(self, 'time', time)
        object.__setattr__(self, 'previous_time', previous_time)
        object.__setattr__(self, 'previous_states', types.MappingProxyType(previous))


class DynamicContribution(abc.ABC):
    """Developer contribution protocol.

    evaluate_trial returns the EXTERNAL chart covector and the total
    tangents K=-dQext/dq, C=-dQext/dqdot, Ma=-dQext/dqdd, component
    diagnostics and a fresh trial-local history record. prepare_step reads
    the immutable accepted history; every evaluate_trial of the same step
    reads the SAME committed history. Physical/world or fixed-reference
    histories (HISTORY_PHYSICAL) survive body rechart unchanged;
    chart-bound histories (HISTORY_CHART) require an explicit declared
    transport callback; unknown basis or a missing callback refuses
    admission, and unknown histories are never reset to zero. No physical
    interpretation or source admission is conferred by admission alone.
    """

    contribution_id: str = ""
    ownership_key: str = ""
    history_basis: str = HISTORY_PHYSICAL
    history_transport = None

    def prepare_step(self, committed_history):
        return None

    def prepare_step_with_context(self, committed_history, information):
        """Additive timed hook; legacy providers keep their existing behavior."""
        return self.prepare_step(committed_history)

    @abc.abstractmethod
    def evaluate_trial(self, committed_history, states, layout):
        """Return a ContributionResult for this trial."""

    def evaluate_trial_with_context(self, committed_history, states, layout, information):
        """Same immutable information for every trial; None for legacy callers.

        A time-dependent provider must require explicit StepInformation rather
        than guess a clock when called by an untimed owner.
        """
        return self.evaluate_trial(committed_history, states, layout)


@dataclass(frozen=True)
class TrialEvaluation:
    """Assembled trial output plus fresh trial-local histories."""
    trial_token: int
    external_force: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    mass: np.ndarray
    diagnostics: dict
    trial_histories: dict
    component_forces: dict


@dataclass(frozen=True)
class _StepContext:
    """Immutable step identity and its committed input snapshot."""
    token: int
    committed: CommittedHistory
    information: StepInformation = None


class ContributionSet:
    """The single history owner: admission, trial assembly, atomic commit."""

    def __init__(self, body_ids, contributions=(), cancel=None):
        self._layout = BodyLayout(tuple(body_ids))
        if not self._layout.body_ids:
            raise MarineContributionsError("at least one body is required")
        if len(set(self._layout.body_ids)) != len(self._layout.body_ids):
            raise MarineContributionsError("body ids must be unique and ordered")
        if cancel is not None and not callable(cancel):
            raise MarineContributionsError("cancel must be callable or None")
        contributions = tuple(contributions)
        from .marine_morison import validate_member_ownership
        validate_member_ownership(contributions,self._layout.body_ids)
        seen_ids, seen_keys = set(), set()
        for contribution in contributions:
            if not isinstance(contribution, DynamicContribution):
                raise MarineContributionsError("contributions must be DynamicContribution instances")
            cid = contribution.contribution_id
            key = contribution.ownership_key
            if not isinstance(cid, str) or not cid.strip():
                raise MarineContributionsError("contribution_id must be a nonempty string")
            if not isinstance(key, str) or not key.strip():
                raise MarineContributionsError("ownership_key must be a nonempty string")
            if cid in seen_ids:
                raise MarineContributionsError(f"duplicate contribution id {cid!r}")
            if cid == "dry":
                raise MarineContributionsError("contribution id 'dry' is reserved for dry inertia output")
            if key in seen_keys:
                raise MarineContributionsError(f"duplicate physical ownership key {key!r}")
            if key.startswith(RESERVED_DRY_MASS_PREFIX):
                raise MarineContributionsError(
                    f"ownership key {key!r} is reserved for the solver's dry mass")
            if contribution.history_basis not in _VALID_BASES:
                raise MarineContributionsError(
                    f"contribution {cid!r} declares unknown history basis "
                    f"{contribution.history_basis!r}; admission refused")
            if (contribution.history_basis == HISTORY_CHART
                    and not callable(contribution.history_transport)):
                raise MarineContributionsError(
                    f"chart-bound contribution {cid!r} requires an explicit "
                    "declared transport callback; admission refused")
            seen_ids.add(cid)
            seen_keys.add(key)
        self._contributions = tuple(contributions)
        self._admissions = tuple((c.contribution_id, c.ownership_key, c.history_basis,
                                  c.history_transport) for c in contributions)
        self._committed = CommittedHistory()
        self._cancel = cancel
        self._active_step = None
        self._active_trial = None
        self._token_counter = 0

    @property
    def body_ids(self):
        return self._layout.body_ids

    @property
    def layout(self):
        return self._layout

    @property
    def committed_history(self):
        return self._committed

    def _poll_cancel(self):
        if self._cancel is not None and self._cancel():
            raise MarineContributionsCancelled(
                "cancellation observed during trial operations")

    def _require_admissions(self):
        for c, admitted in zip(self._contributions, self._admissions):
            if (c.contribution_id, c.ownership_key, c.history_basis, c.history_transport) != admitted:
                raise MarineContributionsError("contribution identity/history contract changed after admission")

    def prepare_step(self, information=None):
        """Read the immutable accepted history; open one new step context."""
        if self._active_step is not None:
            raise MarineContributionsError("finish or abort the active step before preparing another")
        if information is not None:
            if not isinstance(information, StepInformation):
                raise MarineContributionsError('information must be StepInformation or None')
            if information.phase == 'step' and set(information.previous_states) != set(self._layout.body_ids):
                raise MarineContributionsError('previous accepted states must match the body layout')
        self._poll_cancel()
        self._require_admissions()
        self._token_counter += 1
        step = _StepContext(self._token_counter, self._committed, information)
        for contribution in self._contributions:
            contribution.prepare_step_with_context(self._committed, information)
            self._poll_cancel()
            self._require_admissions()
        self._active_step = step
        self._active_trial = None
        return step

    def abort_step(self, step):
        """Discard this step's trials without advancing committed histories."""
        if self._active_step is not step:
            raise MarineContributionsError("only the active step can be aborted")
        self._active_step = None
        self._active_trial = None

    def evaluate_trial(self, step, states):
        """Evaluate all contributions against the step's SAME committed history."""
        if self._active_step is not step:
            raise MarineContributionsError("trial evaluation requires the active step")
        if not isinstance(states, Mapping) or set(states) != set(self._layout.body_ids):
            raise MarineContributionsError("states must match the ordered body ids")
        from .marine_rigid import RigidBodyState
        if any(not isinstance(state, RigidBodyState) for state in states.values()):
            raise MarineContributionsError("states must be immutable RigidBodyState values")
        self._require_admissions()
        size = self._layout.size
        force = np.zeros(size)
        stiffness = np.zeros((size, size))
        damping = np.zeros((size, size))
        mass = np.zeros((size, size))
        diagnostics = {}
        histories = {}
        component_forces = {}
        # Rejected/failed trial attempts invalidate the previous candidate too.
        self._active_trial = None
        states = types.MappingProxyType(dict(states))
        for contribution in self._contributions:
            self._poll_cancel()
            result = contribution.evaluate_trial_with_context(
                step.committed, states, self._layout, step.information)
            self._poll_cancel()
            self._require_admissions()
            if not isinstance(result, ContributionResult):
                raise MarineContributionsError(
                    f"contribution {contribution.contribution_id!r} must return a ContributionResult")
            if result.contribution_id != contribution.contribution_id:
                raise MarineContributionsError("contribution result id mismatch")
            component = _frozen_numeric(result.external_force, (size,), "external force")
            force += component
            stiffness += _frozen_numeric(result.stiffness, (size, size), "stiffness")
            damping += _frozen_numeric(result.damping, (size, size), "damping")
            mass += _frozen_numeric(result.mass, (size, size), "mass")
            if not isinstance(result.diagnostics, Mapping):
                raise MarineContributionsError("diagnostics must be a mapping")
            diagnostics[contribution.contribution_id] = freeze_value(result.diagnostics)
            histories[contribution.contribution_id] = freeze_value(result.trial_history)
            component_forces[contribution.contribution_id] = component
        self._token_counter += 1
        trial = TrialEvaluation(self._token_counter,
                                _frozen_numeric(force, (size,), "assembled force"),
                                _frozen_numeric(stiffness, (size, size), "assembled stiffness"),
                                _frozen_numeric(damping, (size, size), "assembled damping"),
                                _frozen_numeric(mass, (size, size), "assembled mass"),
                                freeze_value(diagnostics), freeze_value(histories),
                                freeze_value(component_forces))
        self._active_trial = trial
        return trial

    def commit(self, step, trial, rechart_info=None):
        """Transport trial histories, then atomically commit ALL contributions.

        Stale trial tokens, duplicate commits, committed steps and foreign
        steps are refused; committed history never changes on refusal.
        """
        if not isinstance(rechart_info, RechartInfo) and rechart_info is not None:
            raise MarineContributionsError("rechart_info must be a RechartInfo or None")
        if self._active_step is not step:
            raise MarineContributionsError("only the active step can be committed")
        if step.information is not None and step.information.phase == 'initialization':
            raise MarineContributionsError('initialization trials cannot commit physical history; abort the step')
        if trial is not self._active_trial or not isinstance(trial, TrialEvaluation):
            raise MarineContributionsError("stale trial token refused: only the step's current trial commits")
        # A cancelled/failed commit attempt can never be retried using its
        # previous candidate, even if a caller later clears cancellation.
        self._active_trial = None
        self._poll_cancel()
        self._require_admissions()
        transported = {}
        for contribution in self._contributions:
            history = trial.trial_histories.get(contribution.contribution_id)
            if contribution.history_basis == HISTORY_CHART:
                if rechart_info is None:
                    raise MarineContributionsError("chart history requires an explicit rechart description")
                history = contribution.history_transport(history, rechart_info)
            self._poll_cancel()
            self._require_admissions()
            transported[contribution.contribution_id] = freeze_value(history)
        candidate = CommittedHistory(transported)
        self._poll_cancel()
        self._committed = candidate
        self._active_step = None
        self._active_trial = None
        return self._committed


def _finite_vector(value, count, name):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MarineContributionsError(f"{name}: {error}") from error
    if raw.shape != (count,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise MarineContributionsError(f"{name} must be a finite shape ({count},) array")
    return np.array(raw, dtype=float, copy=True)


class DryGravityContribution(DynamicContribution):
    """Trusted native explicitly selected dry gravity with exact derivatives.

    The world gravity acceleration is a declared constant applied at the
    body's CG material point; the chart tangent includes the exact
    follower derivative. Gravity is never added to unknown material.
    """

    def __init__(self, body, gravity_world=(0., 0., -9.81)):
        from anyfesolver.marine_rigid import RigidBody6D
        if not isinstance(body, RigidBody6D):
            raise MarineContributionsError("body must be a RigidBody6D")
        self._body = body
        self._gravity = _finite_vector(gravity_world, 3, "gravity_world")
        self.contribution_id = f"dry-gravity:{body.body_id}"
        self.ownership_key = f"dry-gravity:{body.body_id}"
        self.history_basis = HISTORY_PHYSICAL
        self.history_transport = None

    def evaluate_trial(self, committed_history, states, layout):
        from anyfesolver.marine_rigid import left_jacobian, left_jacobian_rate, skew
        state = states[self._body.body_id]
        force_world = self._body.mass*self._gravity
        covector = self._body.gravity_force(state, self._gravity)
        base = layout.index(self._body.body_id)
        size = layout.size
        force = np.zeros(size)
        force[base:base+6] = covector
        theta = np.asarray(state.rotation_vector)
        Jl = left_jacobian(theta)
        x = state.rotation @ np.asarray(self._body.cg_body)
        stiffness = np.zeros((size, size))
        lever = np.cross(x, force_world)
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.
            DJl = left_jacobian_rate(theta, e)
            column = DJl.T @ lever + Jl.T @ np.cross(np.cross(Jl @ e, x), force_world)
            stiffness[base+3:base+6, base+3+j] = -column
        return ContributionResult(self.contribution_id, force, stiffness,
                                  np.zeros((size, size)), np.zeros((size, size)),
                                  {"gravity_work_rate": float(
                                      np.asarray(state.velocity) @ covector)})


class WorldWrenchContribution(DynamicContribution):
    """Trusted native constant world wrench at a material point.

    A constant world force at the body-fixed point plus a constant world
    couple; the chart tangent includes the exact follower derivative of
    the point force and the chart map derivative of the couple.
    """

    def __init__(self, contribution_id, body, point_body,
                 force_world=(0., 0., 0.), moment_world=(0., 0., 0.)):
        from anyfesolver.marine_rigid import RigidBody6D
        if not isinstance(contribution_id, str) or not contribution_id.strip():
            raise MarineContributionsError("contribution_id must be a nonempty string")
        if not isinstance(body, RigidBody6D):
            raise MarineContributionsError("body must be a RigidBody6D")
        self.contribution_id = contribution_id
        self.ownership_key = f"world-wrench:{contribution_id}"
        self._body = body
        self._point = _finite_vector(point_body, 3, "point_body")
        self._force = _finite_vector(force_world, 3, "force_world")
        self._moment = _finite_vector(moment_world, 3, "moment_world")
        self.history_basis = HISTORY_PHYSICAL
        self.history_transport = None

    def evaluate_trial(self, committed_history, states, layout):
        from anyfesolver.marine_rigid import left_jacobian, left_jacobian_rate
        state = states[self._body.body_id]
        covector = (self._body.point_force(state, self._point, self._force)
                    + state.rate_map.T @ np.r_[np.zeros(3), self._moment])
        base = layout.index(self._body.body_id)
        size = layout.size
        force = np.zeros(size)
        force[base:base+6] = covector
        theta = np.asarray(state.rotation_vector)
        Jl = left_jacobian(theta)
        x = state.rotation @ self._point
        lever = np.cross(x, self._force) + self._moment
        stiffness = np.zeros((size, size))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.
            DJl = left_jacobian_rate(theta, e)
            column = (DJl.T @ lever
                      + Jl.T @ np.cross(np.cross(Jl @ e, x), self._force))
            stiffness[base+3:base+6, base+3+j] = -column
        return ContributionResult(self.contribution_id, force, stiffness,
                                  np.zeros((size, size)), np.zeros((size, size)),
                                  {"applied_force_norm": float(np.linalg.norm(self._force)),
                                   "applied_moment_norm": float(np.linalg.norm(self._moment))})


class MechanicalLinkSetContribution(DynamicContribution):
    """Trusted native wrapper assembling a set of MechanicalLink6D links."""

    def __init__(self, links):
        from anyfesolver.marine_link import MechanicalLink6D
        links = tuple(links)
        if not links:
            raise MarineContributionsError("at least one link is required")
        for link in links:
            if not isinstance(link, MechanicalLink6D):
                raise MarineContributionsError("links must be MechanicalLink6D instances")
        if len({link.link_id for link in links}) != len(links):
            raise MarineContributionsError("native link ids must be unique")
        self._links = links
        self.contribution_id = "mechanical-links"
        self.ownership_key = "mechanical-links"
        self.history_basis = HISTORY_PHYSICAL
        self.history_transport = None

    def evaluate_trial(self, committed_history, states, layout):
        size = layout.size
        force = np.zeros(size)
        stiffness = np.zeros((size, size))
        damping = np.zeros((size, size))
        mass = np.zeros((size, size))
        diagnostics = {}
        for link in self._links:
            evaluation = link.evaluate(states[link.body_a], states[link.body_b])
            ia = layout.index(link.body_a)
            ib = layout.index(link.body_b)
            force[ia:ia+6] += evaluation.chart_force_a
            force[ib:ib+6] += evaluation.chart_force_b
            columns = list(range(ia, ia+6)) + list(range(ib, ib+6))
            block = np.ix_(columns, columns)
            stiffness[block] += evaluation.stiffness
            damping[block] += evaluation.damping
            mass[block] += evaluation.mass
            diagnostics[f"{link.link_id}:potential_energy"] = evaluation.potential_energy
            diagnostics[f"{link.link_id}:damping_power"] = evaluation.damping_power
        return ContributionResult(self.contribution_id, force, stiffness,
                                  damping, mass, diagnostics)
