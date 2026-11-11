"""Fixed-step nonlinear Newmark mechanics solver (marine_mechanical_v1).

Monolithic fixed-step integration of rigid-body chart DOFs and optional
three-translation mooring nodes:
beta = 1/4, gamma = 1/2 from the public marine_linear constants, accepted
chart anchors fixed throughout each Newton iteration, E delta_a = r with
E = Ma + gamma dt C + beta dt^2 K, the full residual reevaluated at every
iterate, an optional residual line search that reduces the increment
only, typed failures at singular/nonfinite/branch/iteration-limit events,
consistent initial acceleration including acceleration-dependent
contributions once, and an accepted-step rechart with exact rate
transport and history transport before atomic commit. This route is not
HHT and claims no structural FE registration. No BEM, hydro, contact,
Morison or radiation scientific admission is implied. Named marine contributions
remain explicit development candidates until their scientific gates pass.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning
import warnings
from anyfesolver.marine_contributions import (CommittedHistory, ContributionSet,
    DryGravityContribution, DynamicContribution, MarineContributionsCancelled,
    MarineContributionsError, MechanicalLinkSetContribution, RechartInfo,
    ChartMove, StepInformation)
from anyfesolver.marine_contributions import freeze_value
from anyfesolver.marine_linear import NEWMARK_BETA, NEWMARK_GAMMA
from anyfesolver.marine_link import MechanicalLink6D, MarineLinkError
from anyfesolver.marine_rigid import (MarineRigidInputError, RigidBody6D,
                                   RigidBodyState)

API_VERSION = "marine_mechanical_v1"
MOORING_API_VERSION = "marine_mechanical_mooring_v1"
_LINE_SEARCH_SCALES = (1., .5, .25, .125, .0625)


class MarineMechanicalError(ValueError):
    """Base typed error; never a qualified fallback."""


class MarineMechanicalInputError(MarineMechanicalError):
    """Invalid inputs; fails before any execution."""


class MarineMechanicalSolveError(MarineMechanicalError):
    """Runtime failure carrying failed partial evidence, logs and histories."""

    def __init__(self, message, *, reason, partial, logs, histories):
        super().__init__(message)
        self.reason = reason
        self.partial = partial
        self.logs = tuple(logs)
        self.histories = histories


@dataclass(frozen=True)
class MarineMechanicalRunSpec:
    """Immutable fixed-step run specification; no adaptive stepping."""
    dt: float
    steps: int
    residual_tolerance: float = 1e-10
    max_newton_iterations: int = 20
    line_search: bool = True

    def __post_init__(self):
        if (isinstance(self.dt, bool) or not isinstance(self.dt, (int, float))
                or not math.isfinite(float(self.dt)) or float(self.dt) <= 0.):
            raise MarineMechanicalInputError("dt must be a positive finite real number")
        if (isinstance(self.steps, bool) or not isinstance(self.steps, int)
                or self.steps < 1):
            raise MarineMechanicalInputError("steps must be a positive integer")
        if (isinstance(self.residual_tolerance, bool)
                or not isinstance(self.residual_tolerance, (int, float))
                or not math.isfinite(float(self.residual_tolerance))
                or float(self.residual_tolerance) <= 0.):
            raise MarineMechanicalInputError("residual_tolerance must be positive and finite")
        if (isinstance(self.max_newton_iterations, bool)
                or not isinstance(self.max_newton_iterations, int)
                or self.max_newton_iterations < 1):
            raise MarineMechanicalInputError("max_newton_iterations must be a positive integer")
        if not isinstance(self.line_search, bool):
            raise MarineMechanicalInputError("line_search must be a boolean")
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "residual_tolerance", float(self.residual_tolerance))


@dataclass(frozen=True)
class MarineMechanicalModel:
    """Immutable native model: ordered bodies, links, selected gravity, extras."""
    bodies: tuple
    links: tuple = ()
    gravity_world: tuple = None
    contributions: tuple = ()
    mooring: object = None

    def __post_init__(self):
        bodies = tuple(self.bodies)
        if not bodies:
            raise MarineMechanicalInputError("at least one body is required")
        ids = []
        for body in bodies:
            if not isinstance(body, RigidBody6D):
                raise MarineMechanicalInputError("bodies must be RigidBody6D instances")
            ids.append(body.body_id)
        if len(set(ids)) != len(ids):
            raise MarineMechanicalInputError("body ids must be unique and ordered")
        links = tuple(self.links)
        link_ids = []
        for link in links:
            if not isinstance(link, MechanicalLink6D):
                raise MarineMechanicalInputError("links must be MechanicalLink6D instances")
            if link.body_a not in ids or link.body_b not in ids:
                raise MarineMechanicalInputError("link references an unknown body")
            link_ids.append(link.link_id)
        if len(set(link_ids)) != len(link_ids):
            raise MarineMechanicalInputError("link ids must be unique")
        contributions = tuple(self.contributions)
        from .marine_morison import validate_member_ownership
        for contribution in contributions:
            if not isinstance(contribution, DynamicContribution):
                raise MarineMechanicalInputError(
                    "contributions must be DynamicContribution instances")
        try:
            validate_member_ownership(contributions,ids,bodies)
        except MarineContributionsError as error:
            raise MarineMechanicalInputError(str(error)) from error
        if self.gravity_world is not None:
            gravity = np.asarray(self.gravity_world)
            if (gravity.shape != (3,) or gravity.dtype.kind not in 'iuf'
                    or not np.isfinite(gravity).all()):
                raise MarineMechanicalInputError(
                    "gravity_world must be None or a finite 3-vector; gravity is explicit")
            object.__setattr__(self, "gravity_world", tuple(gravity))
        object.__setattr__(self, "bodies", bodies)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "contributions", contributions)
        if self.mooring is not None:
            from .marine_mooring import MooringLineAssembler
            if type(self.mooring) is not MooringLineAssembler or self.mooring.bodies != bodies:
                raise MarineMechanicalInputError('mooring assembler must bind the exact ordered model bodies')
            if self.gravity_world is not None and tuple(self.mooring.gravity_world)!=tuple(self.gravity_world):
                raise MarineMechanicalInputError('body and line gravity must agree')
            if any(isinstance(c.contribution_id,str) and c.contribution_id.startswith('mooring:') for c in contributions):
                raise MarineMechanicalInputError('mooring: contribution identities are reserved for line force diagnostics')
            from .marine_morison import MorisonMemberSetContribution
            line_ids={line.line_id for line in self.mooring.lines if line.hydrodynamics is not None}
            if any(line_ids.intersection(member.member_id for member in c.members)
                   for c in contributions if isinstance(c,MorisonMemberSetContribution)):
                raise MarineMechanicalInputError('duplicate physical line/Morison member hydrodynamic ownership')

    @property
    def body_ids(self):
        return tuple(body.body_id for body in self.bodies)


@dataclass(frozen=True)
class LogRecord:
    """One immutable log event; incomplete trials are retained."""
    step: int
    event: str
    residual_norm: float
    detail: str = ""


@dataclass(frozen=True)
class AcceptedStepRecord:
    """Accepted finite state evidence; rotation matrices are authoritative."""
    step: int
    time: float
    positions: dict
    rotations: dict
    world_velocities: dict
    world_accelerations: dict
    kinetic_energy: float
    link_potential_energy: float
    link_damping_power: float
    force_decomposition: dict
    residual_norm: float
    newton_iterations: int
    force_coordinate_system: str = "world_wrench_at_body_origin"
    nodes: object = None
    mooring_diagnostics: dict = None

    def __post_init__(self):
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise MarineMechanicalError("accepted step must be a nonnegative integer")
        for name in ("time", "kinetic_energy", "link_potential_energy",
                     "link_damping_power", "residual_norm"):
            if not math.isfinite(getattr(self, name)):
                raise MarineMechanicalError(f"accepted {name} must be finite")
        for name in ("positions", "rotations", "world_velocities",
                     "world_accelerations", "force_decomposition"):
            object.__setattr__(self, name, freeze_value(getattr(self, name)))
        ids = set(self.positions)
        for name, shape in (("positions", (3,)), ("rotations", (3, 3)),
                            ("world_velocities", (6,)), ("world_accelerations", (6,))):
            samples = getattr(self, name)
            if set(samples) != ids or any(np.asarray(v).shape != shape for v in samples.values()):
                raise MarineMechanicalError(f"accepted {name} has inconsistent body keys/shapes")
        for matrix in self.rotations.values():
            matrix = np.asarray(matrix)
            if (np.linalg.norm(matrix.T @ matrix - np.eye(3)) > 1e-12
                    or abs(np.linalg.det(matrix) - 1.) > 1e-12):
                raise MarineMechanicalError("accepted rotations must be proper SO(3) matrices")
        for values in self.force_decomposition.values():
            if set(values) != ids or any(np.asarray(v).shape != (6,) for v in values.values()):
                raise MarineMechanicalError("force decomposition body keys/shapes mismatch")
        if (self.nodes is None)!=(self.mooring_diagnostics is None):
            raise MarineMechanicalError('node evidence and mooring diagnostics must appear together')
        if self.nodes is not None:
            from .marine_mooring import MooringNodeState
            if type(self.nodes) is not MooringNodeState:
                raise MarineMechanicalError('accepted nodes must be immutable MooringNodeState')
            object.__setattr__(self,'mooring_diagnostics',freeze_value(self.mooring_diagnostics))


@dataclass(frozen=True)
class MarineMechanicalResult:
    """Completed or cancelled evidence; failed jobs raise instead."""
    status: str
    body_ids: tuple
    dt: float
    steps_requested: int
    accepted: tuple
    histories: CommittedHistory
    logs: tuple
    node_count: int = None

    def __post_init__(self):
        MarineMechanicalRunSpec(self.dt, self.steps_requested)
        ids = tuple(self.body_ids)
        frames, logs = tuple(self.accepted), tuple(self.logs)
        if (not ids or len(set(ids)) != len(ids)
                or any(not isinstance(bid, str) or not bid for bid in ids)):
            raise MarineMechanicalError("result body ids must be unique nonempty strings")
        if not isinstance(self.histories, CommittedHistory):
            raise MarineMechanicalError("result histories must be immutable committed values")
        if any(not isinstance(f, AcceptedStepRecord) for f in frames):
            raise MarineMechanicalError("accepted frames must be AcceptedStepRecord values")
        if any(not isinstance(log, LogRecord) for log in logs):
            raise MarineMechanicalError("logs must be LogRecord values")
        object.__setattr__(self, 'body_ids', ids)
        object.__setattr__(self, 'accepted', frames)
        object.__setattr__(self, 'logs', logs)
        if self.status not in ('completed', 'cancelled', 'failed'):
            raise MarineMechanicalError("unknown mechanical result status")
        if len(self.accepted) > self.steps_requested+1:
            raise MarineMechanicalError("accepted inventory exceeds the requested steps")
        if self.status == 'completed' and len(self.accepted) != self.steps_requested+1:
            raise MarineMechanicalError("completed requires every requested step")
        for index, frame in enumerate(self.accepted):
            if (frame.step != index or frame.time != index*self.dt
                    or set(frame.positions) != set(self.body_ids)):
                raise MarineMechanicalError("accepted inventory does not match the ordered model/timeline")
        if self.node_count is not None and (type(self.node_count) is not int or self.node_count<0):
            raise MarineMechanicalError('result node_count must be a nonnegative integer or None')
        for frame in frames:
            if ((frame.nodes is None)!=(self.node_count is None)
                    or (frame.nodes is not None and frame.nodes.positions.shape!=(self.node_count,3))):
                raise MarineMechanicalError('accepted node inventory does not match the model')


def _require_callable_or_none(value, name):
    if value is not None and not callable(value):
        raise MarineMechanicalInputError(f"{name} must be callable or None")
    return value


def _normalize_state(state):
    """Initial normalization: anchor at the exact current R, theta = 0."""
    if not isinstance(state, RigidBodyState):
        raise MarineMechanicalInputError("initial states must be RigidBodyState values")
    world_velocity = state.world_velocity
    world_acceleration = state.world_acceleration
    return RigidBodyState(position=np.asarray(state.position),
                          rotation_vector=(0., 0., 0.),
                          velocity=np.r_[world_velocity[:3], world_velocity[3:]],
                          acceleration=np.r_[world_acceleration[:3],
                                             world_acceleration[3:]],
                          anchor=state.rotation)


def _linear_solve(operator, rhs):
    if not sparse.issparse(operator):
        return np.linalg.solve(operator,rhs)
    with warnings.catch_warnings():
        warnings.simplefilter('error',MatrixRankWarning)
        try:
            return spsolve(operator.tocsc(),rhs)
        except MatrixRankWarning as error:
            raise np.linalg.LinAlgError('singular sparse operator') from error


def _matrix_finite(value):
    return np.isfinite(value.data if sparse.issparse(value) else value).all()


class _Solve:
    """Internal orchestration for one solve call."""

    def __init__(self, model, initial_states, run_spec, cancel, progress, initial_nodes=None):
        self.model = model
        self.run_spec = run_spec
        self.cancel = cancel
        self.progress = progress
        self.body_ids = model.body_ids
        self.bodies = {body.body_id: body for body in model.bodies}
        self.layout_index = {bid: 6*i for i, bid in enumerate(self.body_ids)}
        self.body_size = 6*len(self.body_ids)
        self.size = self.body_size if model.mooring is None else model.mooring.size
        self.nodes = initial_nodes
        contributions = []
        if model.links:
            contributions.append(MechanicalLinkSetContribution(model.links))
        if model.gravity_world is not None:
            contributions.extend(DryGravityContribution(body, model.gravity_world)
                                 for body in model.bodies)
        contributions.extend(model.contributions)
        self.owner = ContributionSet(self.body_ids, contributions, cancel)
        self.logs = []
        self.accepted = []
        self.states = {bid: _normalize_state(initial_states[bid]) for bid in self.body_ids}
        self.dt = run_spec.dt
        self.beta = NEWMARK_BETA
        self.gamma = NEWMARK_GAMMA

    def log(self, step, event, residual_norm=float("nan"), detail=""):
        self.logs.append(LogRecord(step, event, float(residual_norm), detail))

    def poll(self):
        if self.cancel is not None and self.cancel():
            raise MarineContributionsCancelled("cancellation requested")

    def build_states(self, q, v, a):
        states = {}
        for bid in self.body_ids:
            base = self.layout_index[bid]
            states[bid] = RigidBodyState(
                position=q[base:base+3], rotation_vector=q[base+3:base+6],
                velocity=v[base:base+6], acceleration=a[base:base+6],
                anchor=self.states[bid].anchor)
        return states

    def build_nodes(self,q,v,a):
        if self.model.mooring is None:return None
        from .marine_mooring import MooringNodeState
        return MooringNodeState(q[self.body_size:].reshape(-1,3),
                               v[self.body_size:].reshape(-1,3),a[self.body_size:].reshape(-1,3))

    def evaluate(self, step_context, states, nodes=None):
        trial = self.owner.evaluate_trial(step_context, states)
        force = np.zeros(self.size);force[:self.body_size]=trial.external_force
        stiffness = trial.stiffness.copy()
        damping = trial.damping.copy()
        mass = trial.mass.copy()
        dry = np.zeros(self.size)
        for bid in self.body_ids:
            base = self.layout_index[bid]
            state = states[bid]
            body = self.bodies[bid]
            dry[base:base+6] = body.inertial_force(state)
            tangents = body.dry_tangents(state)
            stiffness[base:base+6, base:base+6] += tangents.stiffness
            damping[base:base+6, base:base+6] += tangents.damping
            mass[base:base+6, base:base+6] += tangents.mass
        mooring=None
        if self.model.mooring is not None:
            self.poll()
            mooring=self.model.mooring.evaluate(states,nodes,time=step_context.information.time)
            self.poll()
            shape=(self.size-self.body_size,)*2
            stiffness=sparse.block_diag((stiffness,sparse.csr_matrix(shape)),format='csr')+mooring.stiffness
            damping=sparse.block_diag((damping,sparse.csr_matrix(shape)),format='csr')+mooring.damping
            mass=sparse.block_diag((mass,sparse.csr_matrix(shape)),format='csr')+mooring.mass
            force+=mooring.force
        with np.errstate(over='ignore', invalid='ignore'):
            residual = force - dry
        if not (np.isfinite(force).all() and np.isfinite(dry).all()
                and np.isfinite(residual).all()
                and _matrix_finite(stiffness) and _matrix_finite(damping)
                and _matrix_finite(mass)):
            raise MarineMechanicalSolveError(
                "nonfinite residual or tangent during trial", reason="nonfinite",
                partial=self.partial("failed"), logs=self.logs,
                histories=self.owner.committed_history)
        return residual, stiffness, damping, mass, trial, dry, mooring

    def residual_norm(self, r):
        # Dimensionless L2 residual with fixed scales of 1 N for each force
        # and 1 Nm for each moment. Their numerical SI values are both one,
        # so explicit division would be identity; no per-group relative
        # normalization is applied. hypot avoids overflow.
        norm = math.hypot(*(float(value) for value in r))
        if not math.isfinite(norm):
            raise MarineMechanicalSolveError(
                "nonfinite residual norm", reason="nonfinite",
                partial=self.partial("failed"), logs=self.logs,
                histories=self.owner.committed_history)
        return norm

    def chart_vectors(self):
        q = np.zeros(self.size)
        v = np.zeros(self.size)
        a = np.zeros(self.size)
        for bid in self.body_ids:
            base = self.layout_index[bid]
            state = self.states[bid]
            q[base:base+3] = np.asarray(state.position)
            q[base+3:base+6] = np.asarray(state.rotation_vector)
            v[base:base+6] = np.asarray(state.velocity)
            a[base:base+6] = np.asarray(state.acceleration)
        if self.nodes is not None:
            q[self.body_size:]=self.nodes.positions.ravel()
            v[self.body_size:]=self.nodes.velocities.ravel()
            a[self.body_size:]=self.nodes.accelerations.ravel()
        return q, v, a

    def partial(self, status):
        return MarineMechanicalResult(status, self.body_ids, self.dt,
                                       self.run_spec.steps, tuple(self.accepted),
                                       self.owner.committed_history, tuple(self.logs),
                                       None if self.model.mooring is None else self.model.mooring.node_count)

    def record(self, step, states, trial, dry, residual_norm, iterations, nodes=None, mooring=None):
        positions, rotations = {}, {}
        world_velocities, world_accelerations = {}, {}
        kinetic = 0.
        for bid in self.body_ids:
            state = states[bid]
            body = self.bodies[bid]
            positions[bid] = np.asarray(state.position).copy()
            rotations[bid] = np.array(state.rotation, dtype=float, copy=True)
            world_velocities[bid] = state.world_velocity.copy()
            world_accelerations[bid] = state.world_acceleration.copy()
            kinetic += body.kinetic_energy(state)
        decomposition = {"dry": {bid: np.linalg.solve(states[bid].rate_map.T,
                                  dry[self.layout_index[bid]:self.layout_index[bid]+6])
                                for bid in self.body_ids}}
        for cid, component in trial.component_forces.items():
            decomposition[cid] = {bid: np.linalg.solve(states[bid].rate_map.T,
                                  component[self.layout_index[bid]:self.layout_index[bid]+6])
                                   for bid in self.body_ids}
        link_potential = 0.
        link_damping = 0.
        for key, value in trial.diagnostics.get("mechanical-links", {}).items():
            number = float(value)
            if not math.isfinite(number):
                raise MarineMechanicalSolveError(
                    "nonfinite link diagnostic", reason="nonfinite",
                    partial=self.partial("failed"), logs=self.logs,
                    histories=self.owner.committed_history)
            if key.endswith(":potential_energy"):
                link_potential += number
            elif key.endswith(":damping_power"):
                link_damping += number
        mooring_diagnostics=None
        if mooring is not None:
            kinetic+=mooring.kinetic_energy
            for name,component in mooring.force_components.items():
                decomposition['mooring:'+name]={bid:np.linalg.solve(states[bid].rate_map.T,
                    component[self.layout_index[bid]:self.layout_index[bid]+6]) for bid in self.body_ids}
            mooring_diagnostics=dict(tensions=mooring.tensions,lengths=mooring.lengths,
                elastic_tensions=mooring.elastic_tensions,elastic_energy=mooring.elastic_energy,
                kinetic_energy=mooring.kinetic_energy,damping_dissipation=mooring.damping_dissipation,
                endpoint_forces=mooring.endpoint_forces,anchor_reactions=mooring.anchor_reactions,
                body_derivative_steps=mooring.body_derivative_steps,
                node_force_components={name:value[self.body_size:].reshape(-1,3) for name,value in mooring.force_components.items()})
            if self.model.mooring.environment is not None:
                mooring_diagnostics.update(relative_drag_power=mooring.relative_drag_power,
                    added_mass_kinetic_energy=mooring.added_mass_kinetic_energy)
            if mooring.contact_diagnostics is not None:
                mooring_diagnostics['contact']=mooring.contact_diagnostics
        record = AcceptedStepRecord(
            step, step*self.dt, positions, rotations, world_velocities,
            world_accelerations, float(kinetic), link_potential, link_damping,
            decomposition, float(residual_norm), int(iterations),nodes=nodes,mooring_diagnostics=mooring_diagnostics)
        return record

    def initial_acceleration(self):
        """Consistent initial acceleration with q/qdot fixed; Ma assembled once."""
        q, v, _ = self.chart_vectors()
        a = np.zeros(self.size)
        step_context = self.owner.prepare_step(
            StepInformation(0, self.dt, 'initialization'))
        current = None
        iterations = 0
        while True:
            states = self.build_states(q, v, a)
            nodes = self.build_nodes(q,v,a)
            r, _, _, mass, trial, dry, mooring = self.evaluate(step_context, states,nodes)
            norm = self.residual_norm(r)
            self.log(0, "trial", norm)
            current = (r, mass, trial, dry, norm)
            if norm <= self.run_spec.residual_tolerance:
                break
            iterations += 1
            if iterations > self.run_spec.max_newton_iterations:
                raise MarineMechanicalSolveError(
                    "initial acceleration Newton iteration limit reached",
                    reason="iteration_limit", partial=self.partial("failed"),
                    logs=self.logs, histories=self.owner.committed_history)
            try:
                delta = _linear_solve(mass, r)
            except np.linalg.LinAlgError as error:
                raise MarineMechanicalSolveError(
                    f"singular initial mass operator: {error}", reason="singular",
                    partial=self.partial("failed"), logs=self.logs,
                    histories=self.owner.committed_history) from error
            if not np.isfinite(delta).all():
                raise MarineMechanicalSolveError(
                    "nonfinite initial acceleration Newton increment",
                    reason="nonfinite", partial=self.partial("failed"),
                    logs=self.logs, histories=self.owner.committed_history)
            a = a + delta
        candidate_states = {}
        for bid in self.body_ids:
            base = self.layout_index[bid]
            state = self.states[bid]
            candidate_states[bid] = RigidBodyState(
                position=np.asarray(state.position), rotation_vector=(0., 0., 0.),
                velocity=np.asarray(state.velocity),
                acceleration=a[base:base+6], anchor=state.anchor)
        record = self.record(0, candidate_states, current[2], current[3], current[4], iterations,nodes,mooring)
        self.poll()
        # Initial acceleration is algebraic initialization, not a history
        # time step. Its trial histories must not advance physical memory.
        self.owner.abort_step(step_context)
        self.states = candidate_states
        self.nodes = nodes
        self.accepted.append(record)

    def run_steps(self):
        beta, gamma, dt = self.beta, self.gamma, self.dt
        for step in range(1, self.run_spec.steps+1):
            self.poll()
            step_context = self.owner.prepare_step(
                StepInformation(step, self.dt, 'step', self.states))
            q_n, v_n, a_n = self.chart_vectors()
            q_pred = q_n + dt*v_n + dt*dt*(0.5 - beta)*a_n
            v_pred = v_n + dt*(1. - gamma)*a_n
            a = a_n.copy()
            states = self.build_states(q_pred + beta*dt*dt*a,
                                       v_pred + gamma*dt*a, a)
            nodes=self.build_nodes(q_pred+beta*dt*dt*a,v_pred+gamma*dt*a,a)
            r, stiffness, damping, mass, trial, dry, mooring = self.evaluate(
                step_context, states,nodes)
            norm = self.residual_norm(r)
            self.log(step, "trial", norm)
            iterations = 0
            while norm > self.run_spec.residual_tolerance:
                iterations += 1
                if iterations > self.run_spec.max_newton_iterations:
                    raise MarineMechanicalSolveError(
                        f"step {step} Newton iteration limit reached",
                        reason="iteration_limit", partial=self.partial("failed"),
                        logs=self.logs, histories=self.owner.committed_history)
                effective = (mass + gamma*dt*damping + beta*dt*dt*stiffness)
                try:
                    delta = _linear_solve(effective, r)
                except np.linalg.LinAlgError as error:
                    raise MarineMechanicalSolveError(
                        f"step {step} effective operator is singular: {error}",
                        reason="singular", partial=self.partial("failed"),
                        logs=self.logs, histories=self.owner.committed_history) from error
                if not np.isfinite(delta).all():
                    raise MarineMechanicalSolveError(
                        f"step {step} nonfinite Newton increment",
                        reason="nonfinite", partial=self.partial("failed"),
                        logs=self.logs, histories=self.owner.committed_history)
                scales = _LINE_SEARCH_SCALES if self.run_spec.line_search else (1.,)
                chosen = None
                rejected_mooring=False
                for scale in scales:
                    candidate = a + scale*delta
                    try:
                        candidate_states = self.build_states(
                            q_pred + beta*dt*dt*candidate,
                            v_pred + gamma*dt*candidate, candidate)
                    except MarineRigidInputError:
                        self.log(step, "rejected", float("nan"),
                                 "branch-violation candidate")
                        continue
                    candidate_nodes=self.build_nodes(q_pred+beta*dt*dt*candidate,
                                                     v_pred+gamma*dt*candidate,candidate)
                    from .marine_mooring import MooringLineError
                    try:
                        outcome = self.evaluate(step_context, candidate_states,candidate_nodes)
                    except MooringLineError as error:
                        self.log(step,'rejected',float('nan'),f'mooring domain: {error}')
                        rejected_mooring=True
                        continue
                    candidate_norm = self.residual_norm(outcome[0])
                    self.log(step, "trial", candidate_norm,
                             "line-search" if scale < 1. else "")
                    if not self.run_spec.line_search or candidate_norm <= norm:
                        chosen = (candidate, candidate_states, outcome, candidate_norm,candidate_nodes)
                        break
                    self.log(step, "rejected", candidate_norm,
                             f"line-search scale {scale}")
                if chosen is None:
                    reason = "line_search" if self.run_spec.line_search else ('mooring_domain' if rejected_mooring else "branch_violation")
                    message = ("found no residual-reducing trial candidate"
                               if self.run_spec.line_search else
                               ("full Newton candidate violates the mooring domain" if rejected_mooring else
                                "full Newton candidate violates the chart domain"))
                    raise MarineMechanicalSolveError(
                        f"step {step} {message}", reason=reason,
                        partial=self.partial("failed"),
                        logs=self.logs, histories=self.owner.committed_history)
                a, states = chosen[0], chosen[1]
                nodes=chosen[4]
                (r, stiffness, damping, mass, trial, dry, mooring), norm = chosen[2], chosen[3]
            # Converged: finite checks, late cancellation poll, transport, commit.
            self.poll()
            moves = []
            for bid in self.body_ids:
                state = states[bid]
                moves.append(ChartMove(bid, tuple(state.rotation_vector),
                                       tuple(np.asarray(state.anchor)),
                                       tuple(np.array(state.rotation, dtype=float))))
            # Validate all output and recharted states before the single
            # history commit. Failed transport/late cancellation retains
            # exactly the previous accepted inventory and histories.
            record = self.record(step, states, trial, dry, norm, iterations,nodes,mooring)
            recharted = {}
            for bid in self.body_ids:
                state = states[bid]
                world_velocity = state.world_velocity
                world_acceleration = state.world_acceleration
                recharted[bid] = RigidBodyState(
                    position=np.asarray(state.position),
                    rotation_vector=(0., 0., 0.),
                    velocity=np.r_[world_velocity[:3], world_velocity[3:]],
                    acceleration=np.r_[world_acceleration[:3],
                                      world_acceleration[3:]],
                    anchor=np.array(state.rotation, dtype=float))
            self.owner.commit(step_context, trial, RechartInfo(tuple(moves)))
            self.states = recharted
            self.nodes = nodes
            self.accepted.append(record)
            self.log(step, "accepted", norm)
            if self.progress is not None:
                self.progress(record)
            self.poll()  # includes cancellation raised during final progress
        return self.partial("completed")


def solve_marine_mechanical(model, initial_states, run_spec, cancel=None,
                            progress=None, *, initial_nodes=None):
    """Integrate body charts and optional world nodes in one Newmark solve.

    Returns a completed result only after every requested step converges
    and commits; cancellation returns cancelled partial evidence with
    only accepted finite states; runtime failures raise
    MarineMechanicalSolveError carrying failed partial evidence, logs and
    histories. Invalid inputs fail before execution.
    """
    if not isinstance(model, MarineMechanicalModel):
        raise MarineMechanicalInputError("model must be a MarineMechanicalModel")
    if not isinstance(run_spec, MarineMechanicalRunSpec):
        raise MarineMechanicalInputError("run_spec must be a MarineMechanicalRunSpec")
    _require_callable_or_none(cancel, "cancel")
    _require_callable_or_none(progress, "progress")
    if not hasattr(initial_states, "keys"):
        raise MarineMechanicalInputError("initial_states must map body ids to states")
    missing = set(model.body_ids) - set(initial_states.keys())
    extra = set(initial_states.keys()) - set(model.body_ids)
    if missing or extra:
        raise MarineMechanicalInputError(
            "initial_states must contain exactly the model's body ids")
    for bid in model.body_ids:
        if not isinstance(initial_states[bid], RigidBodyState):
            raise MarineMechanicalInputError(
                f"initial state for {bid!r} must be a RigidBodyState")
    from .marine_mooring import MooringNodeState, MooringLineError
    if model.mooring is None:
        if initial_nodes is not None:raise MarineMechanicalInputError('nodes require a mooring model')
    elif type(initial_nodes) is not MooringNodeState or initial_nodes.positions.shape!=(model.mooring.node_count,3):
        raise MarineMechanicalInputError('initial_nodes must match the exact mooring node inventory')
    try:
        solve = _Solve(model, initial_states, run_spec, cancel, progress,initial_nodes)
    except (MarineContributionsError, MarineRigidInputError) as error:
        raise MarineMechanicalInputError(str(error)) from error
    try:
        solve.poll()
        solve.initial_acceleration()
        return solve.run_steps()
    except MarineContributionsCancelled:
        solve.log(-1, "cancelled")
        return solve.partial("cancelled")
    except MarineMechanicalSolveError:
        raise
    except MooringLineError as error:
        raise MarineMechanicalSolveError(f'mooring domain during trial: {error}',reason='mooring_domain',
            partial=solve.partial('failed'),logs=solve.logs,histories=solve.owner.committed_history) from error
    except (MarineRigidInputError, MarineLinkError) as error:
        raise MarineMechanicalSolveError(
            f"branch violation during trial: {error}", reason="branch_violation",
            partial=solve.partial("failed"), logs=solve.logs,
            histories=solve.owner.committed_history) from error
    except MarineContributionsError as error:
        raise MarineMechanicalSolveError(
            f"contribution failure during trial: {error}", reason="provider_failure",
            partial=solve.partial("failed"), logs=solve.logs,
            histories=solve.owner.committed_history) from error
    except Exception as error:
        raise MarineMechanicalSolveError(
            f"provider failure during trial: {error}", reason="provider_failure",
            partial=solve.partial("failed"), logs=solve.logs,
            histories=solve.owner.committed_history) from error


def model_to_dict(model):
    """Encode trusted named native types only; unknown contributions refuse."""
    if type(model) is not MarineMechanicalModel:
        raise MarineMechanicalInputError("model must be a MarineMechanicalModel")
    if (any(type(body) is not RigidBody6D for body in model.bodies)
            or any(type(link) is not MechanicalLink6D for link in model.links)):
        raise MarineMechanicalInputError("native serialization cannot erase subclass physics")
    from anyfesolver.marine_contributions import WorldWrenchContribution
    from anyfesolver.marine_potential_flow import PotentialFlowDevelopmentContribution, data_to_dict
    from anyfesolver.marine_morison import MorisonMemberSetContribution, data_to_dict as morison_to_dict
    data = {"api_version": API_VERSION,
            "bodies": [{"body_id": b.body_id, "mass": b.mass,
                        "cg_body": list(b.cg_body),
                        "inertia_cg_body": [list(row) for row in b.inertia_cg_body],
                        "reference_origin": list(b.reference_origin),
                        "reference_orientation": [list(row) for row in b.reference_orientation]}
                       for b in model.bodies],
            "links": [{"link_id": l.link_id, "body_a": l.body_a, "body_b": l.body_b,
                       "rho_a": list(l.rho_a), "rho_b": list(l.rho_b),
                       "axes_a": [list(row) for row in l.axes_a],
                       "axes_b": [list(row) for row in l.axes_b],
                       "rest_translation": list(l.rest_translation),
                       "rest_rotation": [list(row) for row in l.rest_rotation],
                       "stiffness": [list(row) for row in l.stiffness],
                       "damping": [list(row) for row in l.damping],
                       "preload": list(l.preload),
                       "psd_tolerance": l.psd_tolerance} for l in model.links],
            "gravity_world": (None if model.gravity_world is None
                              else list(model.gravity_world)),
            "contributions": []}
    for contribution in model.contributions:
        if type(contribution) is WorldWrenchContribution:
            data["contributions"].append({
                "type": "world_wrench", "contribution_id": contribution.contribution_id,
                "body_id": contribution._body.body_id, "point_body": list(contribution._point),
                "force_world": list(contribution._force),
                "moment_world": list(contribution._moment)})
        elif type(contribution) is PotentialFlowDevelopmentContribution:
            data['contributions'].append({'type':'potential_flow_development',
                                          'data':data_to_dict(contribution.data)})
        elif type(contribution) is MorisonMemberSetContribution:
            data['contributions'].append({'type':'morison_members_development',
                                          'data':morison_to_dict(contribution)})
        else:
            raise MarineMechanicalInputError(
                "only trusted named native contribution types can be serialized")
    if model.mooring is not None:
        from .marine_mooring import data_to_dict as mooring_to_dict
        data['api_version']=MOORING_API_VERSION
        data['mooring']=mooring_to_dict(model.mooring)
    return data


def model_from_dict(data):
    """Reconstruct native types only; arbitrary provider code is never loaded."""
    if not isinstance(data, dict) or data.get("api_version") not in (API_VERSION,MOORING_API_VERSION):
        raise MarineMechanicalInputError("model data must carry this api_version")
    fields={"api_version", "bodies", "links", "gravity_world", "contributions"}
    if data['api_version']==MOORING_API_VERSION:fields.add('mooring')
    if set(data) != fields:
        raise MarineMechanicalInputError("model fields must exactly match the native schema")
    if any(not isinstance(data[name], (tuple, list)) for name in ("bodies", "links", "contributions")):
        raise MarineMechanicalInputError("native model collections must be sequences")
    try:
        body_fields = {"body_id", "mass", "cg_body", "inertia_cg_body", "reference_origin", "reference_orientation"}
        link_fields = {"link_id", "body_a", "body_b", "rho_a", "rho_b", "axes_a", "axes_b",
                       "rest_translation", "rest_rotation", "stiffness", "damping", "preload", "psd_tolerance"}
        if any(not isinstance(item, dict) or set(item) != body_fields for item in data["bodies"]):
            raise MarineMechanicalInputError("body fields must exactly match the native schema")
        if any(not isinstance(item, dict) or set(item) != link_fields for item in data["links"]):
            raise MarineMechanicalInputError("link fields must exactly match the native schema")
        bodies = tuple(RigidBody6D(item["body_id"], item["mass"], item["cg_body"],
                                  item["inertia_cg_body"], item["reference_origin"],
                                  item["reference_orientation"])
                       for item in data["bodies"])
        links = tuple(MechanicalLink6D(item["link_id"], item["body_a"], item["body_b"],
                                       item["rho_a"], item["rho_b"], item["axes_a"],
                                       item["axes_b"], item["rest_translation"],
                                       item["rest_rotation"], item["stiffness"],
                                       item["damping"], item["preload"],
                                       item["psd_tolerance"])
                     for item in data["links"])
    except (KeyError, TypeError, ValueError) as error:
        raise MarineMechanicalInputError(f"invalid native model data: {error}") from error
    from anyfesolver.marine_contributions import WorldWrenchContribution
    contributions = []
    by_id = {body.body_id: body for body in bodies}
    for item in data.get("contributions", ()):
        if isinstance(item,dict) and item.get('type')=='morison_members_development':
            from anyfesolver.marine_morison import data_from_dict as morison_from_dict
            if set(item)!={'type','data'}:
                raise MarineMechanicalInputError('Morison contribution fields must match named schema')
            try:
                contributions.append(morison_from_dict(item['data'],bodies))
            except (KeyError,TypeError,ValueError) as error:
                raise MarineMechanicalInputError(f'invalid native Morison: {error}') from error
            continue
        if isinstance(item,dict) and item.get('type') == 'potential_flow_development':
            from anyfesolver.marine_potential_flow import PotentialFlowDevelopmentContribution, data_from_dict
            if set(item) != {'type','data'}:
                raise MarineMechanicalInputError('potential-flow contribution fields must match the named schema')
            try:
                fluid = data_from_dict(item['data'])
                if fluid.body_ids != tuple(by_id):
                    raise MarineMechanicalInputError('potential-flow body order must match the whole model')
                if any(type(c) is PotentialFlowDevelopmentContribution for c in contributions):
                    raise MarineMechanicalInputError('duplicate potential-flow owner refused')
                contributions.append(PotentialFlowDevelopmentContribution(fluid))
            except (KeyError,TypeError,ValueError) as error:
                raise MarineMechanicalInputError(f'invalid native potential flow: {error}') from error
            continue
        if not isinstance(item, dict) or item.get("type") != "world_wrench":
            raise MarineMechanicalInputError("unknown contribution type refused")
        if set(item) != {"type", "contribution_id", "body_id", "point_body", "force_world", "moment_world"}:
            raise MarineMechanicalInputError("wrench fields must exactly match the native schema")
        try:
            contributions.append(WorldWrenchContribution(
                item["contribution_id"], by_id[item["body_id"]], item["point_body"],
                item["force_world"], item["moment_world"]))
        except (KeyError, TypeError, ValueError) as error:
            raise MarineMechanicalInputError(f"invalid native wrench: {error}") from error
    mooring=None
    if 'mooring' in data:
        from .marine_mooring import data_from_dict as mooring_from_dict
        try:mooring=mooring_from_dict(data['mooring'],bodies)
        except (TypeError,KeyError,ValueError) as error:
            raise MarineMechanicalInputError(f'invalid native mooring: {error}') from error
    return MarineMechanicalModel(bodies, links, data.get("gravity_world"),
                                  tuple(contributions),mooring)


def model_to_json(model):
    import json
    return json.dumps(model_to_dict(model))


def model_from_json(text):
    import json
    try:
        data = json.loads(text)
    except ValueError as error:
        raise MarineMechanicalInputError(f"model json is malformed: {error}") from error
    return model_from_dict(data)


def spec_to_dict(run_spec):
    if type(run_spec) is not MarineMechanicalRunSpec:
        raise MarineMechanicalInputError("run_spec must be a MarineMechanicalRunSpec")
    return {"api_version": API_VERSION, "dt": run_spec.dt, "steps": run_spec.steps,
            "residual_tolerance": run_spec.residual_tolerance,
            "max_newton_iterations": run_spec.max_newton_iterations,
            "line_search": run_spec.line_search}


def spec_from_dict(data):
    if not isinstance(data, dict) or data.get("api_version") != API_VERSION:
        raise MarineMechanicalInputError("spec data must carry this api_version")
    if set(data) != {"api_version", "dt", "steps", "residual_tolerance", "max_newton_iterations", "line_search"}:
        raise MarineMechanicalInputError("spec fields must exactly match the native schema")
    try:
        return MarineMechanicalRunSpec(data["dt"], data["steps"],
                                       data["residual_tolerance"],
                                       data["max_newton_iterations"],
                                       data["line_search"])
    except KeyError as error:
        raise MarineMechanicalInputError(f"spec data is missing {error}") from error


def result_to_dict(result):
    """Export authoritative native evidence; no Euler conversion or code loader.

    Unmeasured log residuals are explicitly null. Nonfinite authoritative
    values and unsupported complex histories are refused, never replaced.
    Cancelled/failed evidence retains its non-success status.
    """
    if type(result) is not MarineMechanicalResult:
        raise MarineMechanicalInputError("result must be a MarineMechanicalResult")
    if any(type(frame) is not AcceptedStepRecord for frame in result.accepted):
        raise MarineMechanicalInputError("native serialization cannot erase subclass result fields")
    def plain(value):
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list, np.ndarray)):
            return [plain(item) for item in value]
        if isinstance(value, np.generic):
            return plain(value.item())
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise MarineMechanicalInputError("unsupported/nonfinite evidence value; explicit serializer required")
    frames = []
    for frame in result.accepted:
        frames.append({name: plain(getattr(frame, name)) for name in (
            'step', 'time', 'positions', 'rotations', 'world_velocities', 'world_accelerations',
            'kinetic_energy', 'link_potential_energy', 'link_damping_power',
            'force_decomposition', 'residual_norm', 'newton_iterations', 'force_coordinate_system')})
        if frame.nodes is not None:
            frames[-1]['nodes']={name:plain(getattr(frame.nodes,name)) for name in ('positions','velocities','accelerations')}
            frames[-1]['mooring_diagnostics']=plain(frame.mooring_diagnostics)
    data={'api_version': API_VERSION if result.node_count is None else MOORING_API_VERSION, 'status': result.status, 'body_ids': list(result.body_ids),
            'dt': result.dt, 'steps_requested': result.steps_requested, 'accepted': frames,
            'histories': plain(dict(result.histories.items())),
            'force_signs': {'dry': 'internal', 'contributions': 'external'},
            'logs': [{'step': log.step, 'event': log.event, 'detail': log.detail,
                      'residual_norm': log.residual_norm if math.isfinite(log.residual_norm) else None}
                     for log in result.logs]}
    if result.node_count is not None:data['node_count']=result.node_count
    return data
