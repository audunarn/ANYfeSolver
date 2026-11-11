"""Development owner for a coupled, prescribed shell-contact trajectory.

The path keeps several prescribed coordinates in one nonlinear transaction.
Their conjugate forces are unknowns, so an opened patch has no unsupported
rigid translation. This owner admits restricted planar and bounded general
S3 patch static profiles; its checkpoint binds the complete path definition.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.control import CancellationToken, SolveCancelled, cancellation_safe_point
from anyfesolver.linalg import MatrixClass, factorize
from anyfesolver.matrix_assembly import (
    _run_with_qualified_assembly_runtime_lease,
    assemble_stiffness_matrix,
)
from anyfesolver.nonlinear_restart import (
    create_nonlinear_checkpoint,
    validate_nonlinear_checkpoint,
)
from anyfesolver.nonlinear_state import NonlinearStateStore, _run_with_nonlinear_state_cleanup


def _control_projection_seed(rows, start, target):
    """Minimum reduced-coordinate correction for independent scaled controls.

    Row scaling does not change the affine constraint set. The correction lies
    in its row space, so every direction in the control nullspace is retained.
    This is a pose predictor only; it supplies no equilibrium or force estimate.
    """
    rows = np.asarray(rows, dtype=float)
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    if (rows.ndim != 2 or not 1 <= rows.shape[0] <= 64
            or rows.shape[1] != start.size or start.ndim != 1
            or target.shape != (rows.shape[0],)
            or not all(np.all(np.isfinite(x)) for x in (rows, start, target))):
        raise ValueError("control projection predictor needs finite bounded control rows")
    # Avoid overflow/underflow in row norms before normalising.
    largest = np.max(np.abs(rows), axis=1)
    if np.any(largest == 0.):
        raise ValueError("control projection predictor has a zero control row")
    scaled = rows / largest[:, None]
    lengths = np.linalg.norm(scaled, axis=1)
    normalized = scaled / lengths[:, None]
    gram = normalized @ normalized.T
    eigenvalues = np.linalg.eigvalsh(gram)
    guard = 64 * np.finfo(float).eps * len(target)
    if eigenvalues[0] <= guard * eigenvalues[-1]:
        raise ValueError("control projection predictor controls are ill-conditioned")
    correction = ((target / largest) - scaled @ start) / lengths
    seed = start + normalized.T @ np.linalg.solve(gram, correction)
    residual = normalized @ seed - (target / largest) / lengths
    rounding_scale = np.maximum(1., np.abs(normalized) @ np.abs(seed)
                                + np.abs((target / largest) / lengths))
    if (not np.all(np.isfinite(seed)) or not np.all(np.isfinite(residual))
            or np.any(np.abs(residual) > guard * rounding_scale)):
        raise ValueError("control projection predictor could not satisfy controls")
    return seed

from anyfesolver.nonlinear_static import (
    NonlinearStaticResult,
    NonlinearStaticStep,
    _activate_nonlinear_state_storage,
    _assemble_nonlinear_system,
    _commit_nonlinear_state_candidate,
    _discard_nonlinear_state_candidate,
    _finalize_nonlinear_element_states,
    _increment_snapshot,
    _materialize_final_nonlinear_states,
    _max_plastic_strain,
    _prepare_initial_states,
    _prepare_qualified_q4_states_for_nonlinear_solve,
    _seal_final_qualified_q4_states,
    _static_step_from_restart,
    _support_reaction_resultants,
)

from .core import (
    ContactDefinition,
    ContactSolverSettings,
    contact_scope,
    suspend_contact,
    validate_general_contact,
    _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION,
    _INITIAL_CONTACT_SEED_REVISION,
    _INITIAL_CONTACT_SEED_GLOBALIZATION_REVISION,
)
from .augmentation import ProjectedAugmentationSecant
from .linearized_pivot import (
    _projected_clip_signature,
    refined_patch_pivot,
    refined_patch_supported_schur,
)
from .supported_schur import (
    propose_supported_schur_backtrack,
    propose_supported_schur_secant_backtrack,
)
from .equilibrated_pressure_pivot import equilibrated_pressure_pivot
from .semismooth_path import solve_increment as solve_semismooth_increment
from .semismooth import raw_certificate


PROFILE = "PLANAR_Q4_S3_MULTI_CONTROL_CONTACT_PATH_DEV_V0"
AUGMENTED_PROFILE = "PLANAR_Q4_S3_MULTI_CONTROL_AUGMENTED_PATH_DEV_V0"
GENERAL_S3_PATCH_PROFILE = "S3_PATCH_GENERAL_NODAL_MULTI_CONTROL_PENALTY_PATH_DEV_V0"
GENERAL_S3_PATCH_AUGMENTED_PROFILE = "S3_PATCH_GENERAL_AUGMENTED_PATH_DEV_V1"
GENERAL_S3_Q4_PATCH_AUGMENTED_PROFILE = "S3_Q4_PATCH_GENERAL_AUGMENTED_PATH_DEV_V1"
GENERAL_Q4_S3_PATCH_AUGMENTED_PROFILE = "Q4_S3_PATCH_GENERAL_AUGMENTED_PATH_DEV_V1"
GENERAL_Q4_S3_PATCH_PROJECTED_AUGMENTED_PROFILE = "Q4_S3_PATCH_GENERAL_AUGMENTED_PATH_DEV_V2"
GENERAL_Q4_Q4_PATCH_AUGMENTED_PROFILE = "Q4_Q4_PATCH_GENERAL_AUGMENTED_PATH_DEV_V1"
GENERAL_Q4_Q4_PATCH_PROJECTED_AUGMENTED_PROFILE = "Q4_Q4_PATCH_GENERAL_AUGMENTED_PATH_DEV_V2"
GENERAL_S3_PATCH_REFINED_AUGMENTED_PROFILE = "S3_PATCH_REFINED_AUGMENTED_PATH_DEV_V1"
GENERAL_Q4_S3_PATCH_REFINED_AUGMENTED_PROFILE = "Q4_S3_PATCH_REFINED_AUGMENTED_PATH_DEV_V1"
GENERAL_Q4_S3_PATCH_REFINED_PROJECTED_AUGMENTED_PROFILE = "Q4_S3_PATCH_REFINED_PROJECTED_AUGMENTED_PATH_DEV_V0"
GENERAL_Q4_S3_PATCH_REFINED_OVERLAP_LOCAL_AUGMENTED_PROFILE = "Q4_S3_PATCH_REFINED_OVERLAP_LOCAL_AUGMENTED_PATH_DEV_V0"
GENERAL_Q4_S3_PATCH_REFINED_ROUNDOFF_AUGMENTED_PROFILE = "Q4_S3_PATCH_REFINED_ROUNDOFF_AUGMENTED_PATH_DEV_V2"
GENERAL_Q4_S3_PATCH_REFINED_GRAPH_AUGMENTED_PROFILE = "Q4_S3_PATCH_REFINED_FIXED_PROJECTION_GRAPH_AUGMENTED_PATH_DEV_V0"
GENERAL_S3_Q4_PATCH_PROFILE = "S3_Q4_PATCH_GENERAL_NODAL_MULTI_CONTROL_PENALTY_PATH_DEV_V0"
GENERAL_Q4_S3_PATCH_PROFILE = "Q4_S3_PATCH_GENERAL_NODAL_MULTI_CONTROL_PENALTY_PATH_DEV_V0"
GENERAL_Q4_Q4_PROFILE = "Q4_Q4_GENERAL_NODAL_MULTI_CONTROL_PENALTY_PATH_DEV_V0"
GENERAL_Q4_S3_PATCH_REFINED_PROJECTED_PENALTY_PROFILE = "Q4_S3_PATCH_REFINED_PROJECTED_FINITE_PENALTY_PATH_DEV_V0"
GENERAL_S3_PATCH_PROJECTED_PENALTY_PROFILE = "S3_PATCH_PROJECTED_FINITE_PENALTY_STATIC_PATH_DEV_V0"
_DOF = {"ux": 0, "uy": 1, "uz": 2}
_FINITE_DOF = {**_DOF, "rx": 3, "ry": 4, "rz": 5}


def _reported_lower_gap(certificate) -> float:
    """Read the physical gap certificate from either admitted path manager."""
    if hasattr(certificate, "minimum_gap"):
        return float(certificate.minimum_gap)
    return float(certificate.conservative_lower_gap)


@dataclass(frozen=True)
class ContactPathControl:
    terms: tuple[tuple[int, str, float], ...]
    coordinate_profile: str = "translation_v1"

    def __post_init__(self) -> None:
        if self.coordinate_profile not in {"translation_v1", "finite_shell_spin_chart_dev"}:
            raise ValueError("unsupported contact path control coordinate_profile")
        allowed = (_DOF if self.coordinate_profile == "translation_v1" else _FINITE_DOF)
        terms = tuple(self.terms)
        if not terms or any(
            type(term) is not tuple or len(term) != 3
            or type(term[0]) is not int or term[0] <= 0
            or term[1] not in allowed
            or not np.isfinite(term[2])
            for term in terms
        ):
            raise ValueError("contact path terms need positive nodes, translation DOFs and finite weights")
        if len({(node, dof) for node, dof, _weight in terms}) != len(terms):
            raise ValueError("contact path control terms must be unique")
        if all(weight == 0.0 for _node, _dof, weight in terms):
            raise ValueError("contact path control cannot be zero")
        object.__setattr__(self, "terms", tuple(
            (node, dof, float(weight)) for node, dof, weight in terms
        ))

    @classmethod
    def node(cls, node_id: int, dof: str, *, coordinate_profile="translation_v1") -> ContactPathControl:
        return cls(((node_id, dof, 1.0),), coordinate_profile=coordinate_profile)


@dataclass(frozen=True)
class ContactPathNodalForce:
    """One nodal translation-force history in N at the path waypoints."""

    node_id: int
    dof: str
    values_n: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.node_id) is not int or self.node_id <= 0 or self.dof not in _DOF:
            raise ValueError("path force needs a positive node and translation DOF")
        values = tuple(float(value) for value in self.values_n)
        if not values or not np.all(np.isfinite(values)):
            raise ValueError("path force values must be finite and nonempty")
        object.__setattr__(self, "values_n", values)


@dataclass(frozen=True)
class ContactPathProgram:
    controls: tuple[ContactPathControl, ...]
    waypoints: tuple[tuple[float, ...], ...]
    steps_per_segment: int = 2
    nodal_forces: tuple[ContactPathNodalForce, ...] = ()
    control_profile: str = "translation_v1"

    def __post_init__(self) -> None:
        if self.control_profile not in {"translation_v1", "finite_shell_triplets_dev"}:
            raise ValueError("unsupported contact path control_profile")
        controls = tuple(self.controls)
        limit = 3 if self.control_profile == "translation_v1" else 24
        if len(controls) > limit or any(
            type(item) is not ContactPathControl for item in controls
        ) or len({item.terms for item in controls}) != len(controls):
            raise ValueError("contact path control count/profile is unsupported")
        if self.control_profile == "translation_v1" and any(
                item.coordinate_profile != "translation_v1" for item in controls):
            raise ValueError("legacy contact path permits translation controls only")
        points = tuple(tuple(float(value) for value in row)
                       for row in self.waypoints)
        if len(points) < 2 or any(len(row) != len(controls) for row in points):
            raise ValueError("contact path waypoints must match the control count")
        if any(not np.all(np.isfinite(row)) for row in points):
            raise ValueError("contact path waypoints must be finite")
        if any(value != 0.0 for value in points[0]):
            raise ValueError("contact path must start at the reference pose")
        forces = tuple(self.nodal_forces)
        if any(type(item) is not ContactPathNodalForce for item in forces):
            raise TypeError("nodal_forces must contain ContactPathNodalForce")
        if len({(item.node_id, item.dof) for item in forces}) != len(forces):
            raise ValueError("contact path nodal forces must have unique DOFs")
        if any(len(item.values_n) != len(points) or item.values_n[0] != 0.0
               for item in forces):
            raise ValueError("path force values must match waypoints and start at zero")
        if any(points[index] == points[index - 1]
               and all(item.values_n[index] == item.values_n[index - 1]
                       for item in forces)
               for index in range(1, len(points))):
            raise ValueError("contact path cannot contain a zero-change segment")
        if type(self.steps_per_segment) is not int or self.steps_per_segment < 1:
            raise ValueError("steps_per_segment must be a positive integer")
        if (len(points) - 1) * self.steps_per_segment > 10_000:
            raise MemoryError("contact path exceeds 10000 accepted-step slots")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "waypoints", points)
        object.__setattr__(self, "nodal_forces", forces)

    @property
    def total_steps(self) -> int:
        return (len(self.waypoints) - 1) * self.steps_per_segment

    def target_at(self, accepted_step: int) -> np.ndarray:
        step = int(accepted_step)
        if not 0 <= step <= self.total_steps:
            raise ValueError("contact path step lies outside the program")
        if step == self.total_steps:
            return np.asarray(self.waypoints[-1], dtype=float)
        segment, local = divmod(step, self.steps_per_segment)
        fraction = local / self.steps_per_segment
        return ((1.0 - fraction) * np.asarray(self.waypoints[segment])
                + fraction * np.asarray(self.waypoints[segment + 1]))

    def force_values_at(self, accepted_step: int) -> np.ndarray:
        step = int(accepted_step)
        if not 0 <= step <= self.total_steps:
            raise ValueError("contact path step lies outside the program")
        if not self.nodal_forces:
            return np.zeros(0)
        if step == self.total_steps:
            return np.asarray([item.values_n[-1] for item in self.nodal_forces])
        segment, local = divmod(step, self.steps_per_segment)
        fraction = local / self.steps_per_segment
        return np.asarray([
            (1.0-fraction)*item.values_n[segment]
            + fraction*item.values_n[segment+1]
            for item in self.nodal_forces
        ])

    def contract(self) -> dict[str, object]:
        result = {
            "controls": [{"terms": [list(term) for term in item.terms],
                          **({"coordinate_profile": item.coordinate_profile}
                             if item.coordinate_profile != "translation_v1" else {})}
                         for item in self.controls],
            "waypoints": [list(row) for row in self.waypoints],
            "steps_per_segment": self.steps_per_segment,
        }
        if self.control_profile != "translation_v1":
            result["control_profile"] = self.control_profile
        if self.nodal_forces:
            result["nodal_forces"] = [
                {"node_id": item.node_id, "dof": item.dof,
                 "values_n": list(item.values_n)}
                for item in self.nodal_forces
            ]
        return result


def _program_from_dead_nodal_load_case(load_case, num_steps: int) -> ContactPathProgram:
    """Own a narrow ordinary force load as a zero-control contact path."""
    from anyfesolver.boundary import LoadCase

    if type(load_case) is not LoadCase or type(num_steps) is not int or num_steps < 1:
        raise ValueError("monolithic ordinary force contact needs a LoadCase and positive num_steps")
    if any(type(value) is not dict for value in (
            load_case.nodal_loads, load_case.element_loads,
            load_case.pressure_loads, load_case.added_node_masses)):
        raise ValueError("monolithic ordinary force contact needs owned load mappings")
    if (load_case.element_loads or load_case.pressure_loads
            or load_case.gravity is not None or load_case.added_node_masses
            or load_case.follower_pressure or not load_case.nodal_loads):
        raise ValueError("monolithic ordinary force contact admits dead nodal loads only")
    forces = []
    if any(type(node_id) is not int or node_id <= 0
           for node_id in load_case.nodal_loads):
        raise ValueError("monolithic nodal load needs exact positive node IDs")
    for node_id, value in sorted(load_case.nodal_loads.items()):
        if type(value) is not np.ndarray or value.dtype.kind not in {"f", "i", "u"}:
            raise ValueError("monolithic nodal load needs primitive numeric arrays")
        vector = np.asarray(value, dtype=float)
        if (vector.shape != (6,) or not np.all(np.isfinite(vector))
                or np.any(vector[3:] != 0.0)):
            raise ValueError("monolithic ordinary force contact admits finite translation forces only")
        for component, dof in enumerate(("ux", "uy", "uz")):
            if vector[component] != 0.0:
                forces.append(ContactPathNodalForce(
                    node_id, dof, (0.0, float(vector[component])),
                ))
    if not forces:
        raise ValueError("monolithic ordinary force contact needs a nonzero nodal force")
    return ContactPathProgram(controls=(), waypoints=((), ()),
                              steps_per_segment=num_steps,
                              nodal_forces=tuple(forces))


def _program_from_simple_displacement_control(control, num_steps: int) -> ContactPathProgram:
    """Own one ordinary nodal translation displacement without applied loads."""
    from anyfesolver.nonlinear_static import DisplacementControl

    if (type(control) is not DisplacementControl
            or type(num_steps) is not int or num_steps < 1
            or type(control.node_id) is not int or control.node_id <= 0
            or type(control.dof) is not str or control.dof not in _DOF
            or control.weighted_dofs is not None
            or type(control.target_displacement) not in {int, float}
            or not np.isfinite(control.target_displacement)
            or control.target_displacement == 0.0):
        raise ValueError(
            "monolithic ordinary displacement contact needs one finite "
            "nonzero nodal translation target"
        )
    return ContactPathProgram(
        controls=(ContactPathControl.node(control.node_id, control.dof),),
        waypoints=((0.0,), (float(control.target_displacement),)),
        steps_per_segment=num_steps,
    )


def _elastic_displacement_force_predictor(stiffness, transform, row,
                                          target, *, allow_zero=False) -> float:
    """Bound a declared force scale by the contact-free control reactions.

    For several controls, use the maximum absolute reaction sum over path
    waypoints. This retains the one-control value and one physical force unit.
    A fully controlled rigid tool can have exactly zero contact-free reaction;
    the explicit S3 projected-penalty and S3/Q4 projected-penalty profiles
    may report that zero without using it as a force normalization. Singular
    uncontrolled systems still reject.
    """
    reduced = sparse.csr_matrix(transform.T @ stiffness @ transform)
    rows = np.asarray(row, dtype=float)
    if rows.ndim == 1:
        rows = rows[None, :]
    targets = np.asarray(target, dtype=float)
    if targets.ndim == 0:
        targets = targets.reshape(1, 1)
    elif targets.ndim == 1:
        targets = targets[None, :]
    controls, coordinates = rows.shape
    if (reduced.shape != (coordinates, coordinates)
            or not 1 <= controls <= 3
            or targets.ndim != 2 or targets.shape[1] != controls
            or not np.all(np.isfinite(targets))
            or not np.any(targets != 0.0)):
        raise ValueError("displacement force predictor has incompatible dimensions")
    saddle = sparse.bmat((
        (reduced, sparse.csr_matrix(-rows.T)),
        (sparse.csr_matrix(rows), sparse.csr_matrix((controls, controls))),
    ), format="csr")
    try:
        solve = factorize(
            saddle, MatrixClass.GENERAL,
            signature="contact_path:elastic_displacement_force_predictor",
        ).solve
    except Exception as exc:
        raise ValueError("contact-free displacement force predictor is unavailable") from exc
    maximum_reaction = 0.0
    for waypoint in targets:
        rhs = np.zeros(coordinates+controls)
        rhs[-controls:] = waypoint
        try:
            response = np.asarray(solve(rhs)).reshape(-1)
        except Exception as exc:
            raise ValueError("contact-free displacement force predictor is unavailable") from exc
        residual = np.asarray(saddle @ response - rhs).reshape(-1)
        scale = abs(saddle) @ abs(response) + abs(rhs)
        reactions = response[-controls:]
        if (not np.all(np.isfinite(response))
                or np.any(abs(residual) > 1e-12 + 1e-9*scale)
                or not np.all(np.isfinite(reactions))):
            raise ValueError("contact-free displacement force predictor is unresolved")
        maximum_reaction = max(maximum_reaction,
                               float(np.sum(np.abs(reactions))))
    if maximum_reaction <= 0.0 and not allow_zero:
        raise ValueError("contact-free displacement force predictor is unresolved")
    return maximum_reaction


def _validate_continuation_history_record(record, program, settings):
    """Validate one declared continuation trace before any Newton work.

    Enabled continuation checkpoints carry an optional ``continuation``
    block on accepted macro records. Every present block must declare the
    bound numerical revision, the enabled halvings budget, a nonempty
    cutback reason, and a contiguous finite dyadic tau chronology from 0
    to the exact macro endpoint whose spans are exactly producer
    reachable: interior spans are dyadic unit fractions 2^-k with
    1 <= k <= maximum_halvings (the producer only ever halves or doubles
    its span from 0.5, so no other interior span can occur), the final
    span is a positive multiple of 2^-maximum_halvings no larger than
    the producer's 0.5 cap, and the accepted span sequence must be
    reachable from the producer's span state machine: a span can grow by
    at most the single doubling the producer performs on acceptance, the
    recorded halvings_used must satisfy the producer identity
    1 + trial_count - accepted_substep_count, and the failures implied
    by the spans (including producer-permitted repeated failures on the
    final truncated interval) must fit that recorded count. Per-substep
    diagnostics must be finite nonnegative counts/norms, the
    external-nodal-work key must be present exactly when the program
    declares nodal forces, the accounting fields must sit inside the
    producer's recorded trial budget, the recorded stagnation controls
    must equal the live settings, and the record's target must match the
    program's own waypoint at its step. Disabled runs never produce
    these blocks and never run this check.
    """
    if type(record) is not dict:
        raise ValueError("contact path checkpoint history is incompatible")
    continuation = record.get("continuation")
    if continuation is None:
        return
    if (type(continuation) is not dict
            or continuation.get("mode") != "declared_physical_dyadic_substeps"
            or continuation.get("revision") != _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION
            or type(continuation.get("substeps")) is not list):
        raise ValueError("contact path checkpoint continuation block is incompatible")
    maximum_halvings = settings.path_continuation_maximum_halvings
    recorded_halvings = continuation.get("maximum_halvings")
    if (type(maximum_halvings) is not int
            or not 0 < maximum_halvings <= 12
            or type(recorded_halvings) is not int
            or not 0 < recorded_halvings <= 12
            or recorded_halvings != maximum_halvings):
        raise ValueError(
            "contact path checkpoint continuation halvings budget is incompatible")
    cutback_reason = continuation.get("cutback_reason")
    if type(cutback_reason) is not str or not cutback_reason:
        raise ValueError(
            "contact path checkpoint continuation cutback reason is invalid")
    step_index = record.get("step_index")
    if (type(step_index) is not int
            or not 0 < step_index <= program.total_steps):
        raise ValueError("contact path checkpoint continuation step is invalid")
    target = np.asarray(record.get("target"), dtype=float)
    if (target.shape != (len(program.controls),)
            or not np.all(np.isfinite(target))
            or not np.allclose(target, program.target_at(step_index),
                               rtol=0.0, atol=1.0e-12)):
        raise ValueError("contact path checkpoint continuation endpoint is inconsistent")
    halvings_used = continuation.get("halvings_used")
    trial_count = continuation.get("trial_count")
    accepted_count = continuation.get("accepted_substep_count")
    patience = continuation.get("stagnation_patience")
    fraction = continuation.get("stagnation_fraction")
    merit_rate = continuation.get("stagnation_merit_rate")
    if (type(halvings_used) is not int
            or not 1 <= halvings_used <= recorded_halvings
            or type(trial_count) is not int
            or not 1 <= trial_count <= (
                (1 << (recorded_halvings + 1))
                + 2*recorded_halvings + 2)
            or type(accepted_count) is not int
            or not 1 <= accepted_count == len(continuation["substeps"])
            or accepted_count > trial_count
            or halvings_used != 1 + trial_count - accepted_count
            or type(patience) is not int or patience < 1
            or patience != settings.path_stagnation_patience
            or type(fraction) is not float or not np.isfinite(fraction)
            or not 0.0 < fraction <= 1.0
            or fraction != settings.path_stagnation_fraction
            or type(merit_rate) is not float or not np.isfinite(merit_rate)
            or not 0.0 < merit_rate < 1.0
            or merit_rate != settings.path_stagnation_merit_rate):
        raise ValueError(
            "contact path checkpoint continuation accounting is invalid")

    def _finite_nonnegative(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(value) and value >= 0.0)

    tau = 0.0
    grid = float(1 << recorded_halvings)
    expects_work = bool(program.nodal_forces)
    # Producer span state: the nominal span starts at 0.5, halves on
    # every failed trial and doubles (capped at 0.5) on every accepted
    # substep, while the halvings counter only grows. depth is the
    # nominal span exponent (nominal = 0.5 * 2^-depth); failures counts
    # the producer failures implied by the accepted spans so far.
    depth = 0
    failures = 0
    for substep in continuation["substeps"]:
        if type(substep) is not dict:
            raise ValueError("contact path checkpoint continuation substep is invalid")
        if (type(substep.get("iterations")) is not int
                or substep["iterations"] < 0
                or type(substep.get("candidate_evaluations")) is not int
                or substep["candidate_evaluations"] < 0
                or type(substep.get("backtracks")) is not int
                or substep["backtracks"] < 0
                or not _finite_nonnegative(substep.get("residual_norm"))
                or not _finite_nonnegative(substep.get("constraint_error"))
                or not _finite_nonnegative(
                    substep.get("complementarity_relative_force"))):
            raise ValueError("contact path checkpoint continuation substep is invalid")
        work = substep.get("external_nodal_work_trapezoid_j")
        if expects_work:
            # The producer emits the signed finite trapezoid work on
            # every accepted substep exactly when the program declares
            # nodal forces; presence must agree with the program.
            if (isinstance(work, bool)
                    or not isinstance(work, (int, float))
                    or not np.isfinite(work)):
                raise ValueError(
                    "contact path checkpoint continuation substep is invalid")
        elif work is not None:
            raise ValueError("contact path checkpoint continuation substep is invalid")
        start = substep.get("tau_start")
        end = substep.get("tau_end")
        if (type(start) is not float or type(end) is not float
                or not np.isfinite(start) or not np.isfinite(end)
                or not 0.0 <= start <= end <= 1.0
                or start != tau):
            raise ValueError("contact path checkpoint continuation chronology is invalid")
        span = end - start
        scaled = span * grid
        if not (scaled > 0.0 and scaled.is_integer()):
            raise ValueError("contact path checkpoint continuation chronology is invalid")
        if end < 1.0:
            # Producer interior spans are exact dyadic unit fractions
            # 2^-k with 1 <= k <= recorded_halvings: on the 2^-M grid
            # that is a power of two no larger than the 0.5 cap.
            units = int(scaled)
            if (units & (units - 1)) or units > (1 << (recorded_halvings - 1)):
                raise ValueError(
                    "contact path checkpoint continuation chronology is invalid")
            # An accepted interior span is exactly the live nominal
            # span, so it can be reached from the previous accepted span
            # only through the producer's halvings: never by more than
            # the one doubling the producer performs on acceptance.
            trial_depth = recorded_halvings - units.bit_length()
            if trial_depth < depth:
                raise ValueError(
                    "contact path checkpoint continuation chronology is invalid")
            failures += trial_depth - depth
            depth = trial_depth - 1 if trial_depth > 1 else 0
        elif span > 0.5:
            # The final span is 1 - tau for the trial that reached the
            # macro endpoint: never larger than the producer's 0.5 cap.
            raise ValueError("contact path checkpoint continuation chronology is invalid")
        else:
            # The final truncated interval is accepted with span 1 - tau
            # while the nominal span stays at least that large; repeated
            # failures on the same interval are producer permitted, so
            # the remaining recorded halvings must fit the nominal span
            # at the final acceptance.
            remaining = (halvings_used - 1) - failures
            final_depth = depth + remaining
            if (remaining < 0
                    or final_depth > recorded_halvings - 1
                    or (1 << (recorded_halvings - 1 - final_depth))
                    < int(scaled)):
                raise ValueError(
                    "contact path checkpoint continuation chronology is invalid")
        tau = end
    if not continuation["substeps"] or tau != 1.0:
        raise ValueError("contact path checkpoint continuation does not reach the macro endpoint")
def solve_static_contact_path(
        model, definition: ContactDefinition, settings: ContactSolverSettings,
        program: ContactPathProgram, *, max_path_fraction: float = 1.0,
        max_iterations: int = 18, tolerance: float = 1.0e-7,
        num_layers: int = 5, restart_checkpoint=None,
        emit_restart_checkpoint: bool = False,
        record_increment_snapshots: bool = False,
        cancellation_token: CancellationToken | None = None,
        friction=None,
        ) -> NonlinearStaticResult:
    """Solve one multi-control path under an admitted static contact profile.

    Opt-in snapshots contain accepted increments produced by this invocation;
    a resumed call does not recreate the checkpoint's earlier poses.
    """

    if type(definition) is not ContactDefinition or type(settings) is not ContactSolverSettings:
        raise TypeError("contact path needs exact contact definitions and settings")
    if type(program) is not ContactPathProgram:
        raise TypeError("contact path needs ContactPathProgram")
    finite = settings.shell_skin_kinematics != "reference_linear"
    if ((program.control_profile != "translation_v1") != finite):
        raise ValueError("finite_shell_triplets_dev requires corotational_nodal_director_dev and conversely")
    structural_kinematics = "corotational" if finite else "von_karman"
    assembly_options = ({"corotational_tangent": "consistent"} if finite else {})
    if settings.initial_contact_seed != "none" and friction is not None:
        # Fail closed: the initial-contact seed is an initial-only
        # frictionless cold-start initializer. Refuse explicitly before
        # any route preparation instead of silently dropping the
        # requested selection at the first macro attempt.
        raise ValueError("initial_contact_seed requires frictionless contact")
    if settings.general_s3_differentiation == "jet2_q4_q4_dev" and friction is not None:
        raise ValueError("jet2_q4_q4_dev currently requires frictionless contact")
    from .s3_sliding_friction import S3MovingNormalFrictionSettings, S3MeshMovingNormalFrictionSettings, S3MeshWeakPressureFrictionSettings, S3MeshWholeChordGapFrictionSettings, S3MeshHistoryPathFrictionSettings
    from .s3_friction_replay import S3RimReplayFrictionSettings
    rim_friction = type(friction) is S3RimReplayFrictionSettings
    if rim_friction and settings.shell_pair_orientation != 'legacy':
        raise ValueError('rim replay requires legacy parallel physical-side declarations')
    history_friction = type(friction) is S3MeshHistoryPathFrictionSettings
    history_policy = None
    if history_friction:
        from .s3_friction_history_path import expand_program
        program, history_policy = expand_program(program, friction.physical_subincrements_per_user_increment)
    whole_chord_friction = type(friction) in {S3MeshWholeChordGapFrictionSettings,S3MeshHistoryPathFrictionSettings}
    weak_pressure_friction = type(friction) in {S3MeshWeakPressureFrictionSettings,S3MeshWholeChordGapFrictionSettings,S3MeshHistoryPathFrictionSettings}
    mesh_moving_friction = type(friction) in {S3MeshMovingNormalFrictionSettings,S3MeshWeakPressureFrictionSettings,S3MeshWholeChordGapFrictionSettings,S3MeshHistoryPathFrictionSettings}
    moving_friction = type(friction) in {S3MovingNormalFrictionSettings,S3MeshMovingNormalFrictionSettings,S3MeshWeakPressureFrictionSettings,S3MeshWholeChordGapFrictionSettings,S3MeshHistoryPathFrictionSettings}
    if moving_friction and (settings.enforcement_solver != "monolithic_active_set"
            or settings.general_s3_differentiation != "jet2_s3_dev"
            or settings.shell_pair_orientation != "physical_sides_dev"
            or (settings.path_predictor != "none" and not mesh_moving_friction)
            or len(definition.surfaces) != 2 or len(definition.interactions) != 1
            or (not mesh_moving_friction and any(len(s.element_ids) != 1 for s in definition.surfaces))
            or definition.interactions[0].normal_law != "augmented"):
        raise ValueError("moving-normal friction requires the explicit full-face S3 monolithic physical-side path")
    if friction is not None and not moving_friction and (settings.shell_pair_orientation != "legacy"
                                 or settings.path_predictor != "none"):
        from .s3_sliding_friction import S3SlidingFrictionSettings
        from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
        if (type(friction) not in {S3SlidingFrictionSettings,S3RimReplayFrictionSettings}
                or settings.general_s3_differentiation != "jet2_s3_dev"
                or settings.normal_discretization != "primal_mortar"
                or settings.enforcement_solver != "nested_augmented"
                or definition.rigid_bodies or len(definition.surfaces) != 2
                or len(definition.interactions) != 1
                or definition.interactions[0].normal_law != "projected_penalty"
                or not all(type(model.mesh.elements.get(eid)) is NativeParityE4PLS3V2DShellElement
                           for surface in definition.surfaces for eid in surface.element_ids)):
            raise ValueError("physical-side/predicted friction path requires S3SlidingFrictionSettings and the refined S3 projected-penalty owner")
    report = validate_general_contact(model, definition, settings,
                                      analysis_control="contact_path")
    if not report.valid:
        raise ValueError("unsupported contact path: " + "; ".join(report.issues))
    from .s3_patch_static import MIXED_PROFILE as mixed_patch_profile
    from .s3_patch_static import S3_MONOLITHIC_PROFILE as s3_monolithic_profile
    from .s3_patch_static import S3_MESH_MONOLITHIC_PROFILE as s3_mesh_monolithic_profile
    from .s3_patch_static import PROFILE as s3_patch_profile
    from .s3_patch_static import SOURCE_Q4_PROFILE as source_q4_patch_profile
    from .s3_patch_static import Q4_PAIR_PROFILE as q4_pair_profile
    from .s3_patch_static import REFINED_Q4_Q4_PROJECTED_PENALTY_PROFILE as q4_q4_projected_profile
    from .s3_patch_static import AUGMENTED_PROFILE as general_augmented_profile
    from .s3_patch_static import AUGMENTED_MIXED_PROFILE as mixed_augmented_profile
    from .s3_patch_static import SOURCE_Q4_AUGMENTED_PROFILE as source_q4_augmented_profile
    from .s3_patch_static import SOURCE_Q4_PROJECTED_AUGMENTED_PROFILE as source_q4_projected_augmented_profile
    from .s3_patch_static import Q4_PAIR_AUGMENTED_PROFILE as q4_pair_augmented_profile
    from .s3_patch_static import Q4_PAIR_PROJECTED_AUGMENTED_PROFILE as q4_pair_projected_augmented_profile
    from .s3_patch_static import REFINED_AUGMENTED_PROFILE as refined_augmented_profile
    from .s3_patch_static import REFINED_Q4_S3_AUGMENTED_PROFILE as refined_q4_s3_augmented_profile
    from .s3_patch_static import REFINED_Q4_S3_PROJECTED_AUGMENTED_PROFILE as refined_q4_s3_projected_profile
    from .s3_patch_static import REFINED_Q4_S3_OVERLAP_LOCAL_AUGMENTED_PROFILE as refined_q4_s3_overlap_local_profile
    from .s3_patch_static import REFINED_Q4_S3_ROUNDOFF_AUGMENTED_PROFILE as refined_q4_s3_roundoff_profile
    from .s3_patch_static import REFINED_Q4_S3_GRAPH_AUGMENTED_PROFILE as refined_q4_s3_graph_profile
    from .s3_patch_static import REFINED_Q4_S3_PROJECTED_PENALTY_PROFILE as refined_q4_s3_projected_penalty_profile
    from .s3_patch_static import REFINED_Q4_Q4_MONOLITHIC_AUGMENTED_PROFILE as q4_q4_monolithic_profile
    from .s3_patch_static import FINITE_Q4_Q4_MONOLITHIC_PROFILE as finite_q4_profile
    from .finite_mixed_state import STATIC_PROFILE as finite_mixed_profile,PROFILE as finite_mixed_path_profile
    from .finite_v2d_state import STATIC_PROFILE as finite_v2d_profile,PROFILE as finite_v2d_path_profile
    from .s3_patch_static import S3_Q4_PROJECTED_PENALTY_PROFILE as s3_q4_projected_penalty_profile
    from .s3_patch_static import S3_PROJECTED_PENALTY_PROFILE as s3_projected_penalty_profile
    from .multiple_projected_pairs import PROFILE as multiple_projected_profile, PATH_PROFILE as multiple_path_profile
    from .s3_patch_static import SupportResetForceBudgetExceeded
    from .q4_s3_overlap_derivatives import UnsupportedQ4S3Branch

    from .triangulated_rigid_static import PROFILE as rigid_mesh_profile, PATCH_PROFILE as rigid_patch_profile
    general_s3_patch = report.profile in {
        rigid_mesh_profile, rigid_patch_profile,
        s3_patch_profile, mixed_patch_profile, source_q4_patch_profile,
        q4_q4_projected_profile, q4_q4_monolithic_profile, finite_q4_profile, finite_mixed_profile, finite_v2d_profile, s3_monolithic_profile, s3_mesh_monolithic_profile, q4_pair_profile, general_augmented_profile, mixed_augmented_profile,
        refined_augmented_profile, refined_q4_s3_augmented_profile,
        refined_q4_s3_projected_profile, refined_q4_s3_projected_penalty_profile,
        s3_projected_penalty_profile, s3_q4_projected_penalty_profile, multiple_projected_profile,
        refined_q4_s3_overlap_local_profile,
        refined_q4_s3_roundoff_profile, refined_q4_s3_graph_profile,
        source_q4_augmented_profile,
        source_q4_projected_augmented_profile,
        q4_pair_augmented_profile, q4_pair_projected_augmented_profile,
    }
    if report.profile == s3_mesh_monolithic_profile and not mesh_moving_friction:
        raise ValueError("S3 mesh monolithic public path requires exact S3MeshMovingNormalFrictionSettings")
    if not general_s3_patch and "RESTRICTED_SLIDING" not in report.profile:
        raise ValueError("unsupported contact path profile: " + report.profile)
    if ((definition.rigid_bodies and report.profile not in (rigid_mesh_profile,rigid_patch_profile)) or
            settings.pair_motion != ("general" if general_s3_patch
                                    else "restricted_sliding")):
        raise ValueError("contact path currently requires a deformable shell pair")
    augmented = definition.interactions[0].normal_law == "augmented"
    semismooth = settings.enforcement_solver == "monolithic_active_set"
    displacement_only = bool(program.controls) and not program.nodal_forces
    if len(program.controls) < 2 and not semismooth:
        raise ValueError("zero/one-control contact path requires monolithic_active_set")
    if len(program.controls) == 1 and program.nodal_forces:
        raise ValueError("one-control mixed force/displacement path is not admitted")
    if displacement_only and len(program.controls) == 1:
        terms = program.controls[0].terms
        if (len(terms) != 1 or terms[0][2] != 1.0
                or len(program.waypoints) != 2
                or program.waypoints[1][0] == 0.0):
            raise ValueError(
                "one-control contact path requires one unit-weight nodal "
                "translation and one nonzero target waypoint"
            )
    if semismooth and report.profile not in {q4_q4_monolithic_profile, finite_q4_profile, finite_mixed_profile, finite_v2d_profile, s3_monolithic_profile, s3_mesh_monolithic_profile} and (
            not augmented
            or report.profile not in {
                refined_q4_s3_augmented_profile,
                refined_q4_s3_projected_profile,
                refined_q4_s3_overlap_local_profile,
            }
            or settings.general_s3_differentiation != "jet2_q4_s3_dev"
            or settings.q4_source_pressure_basis != "fine_nodal_q1"
            or settings.augmentation_support_resolution != "exact"):
        raise ValueError("monolithic_active_set requires fine Q1 refined Q4/S3 Jet2 augmented contact")
    if semismooth:
        declared_force = max(
            (sum(abs(item.values_n[index]) for item in program.nodal_forces)
             for index in range(len(program.waypoints))),
            default=0.0,
        )
        if (not displacement_only and
                (declared_force <= 0.0
                 or settings.complementarity_force_scale_n
                 > declared_force*(1.0+1.0e-12))):
            raise ValueError(
                "monolithic_active_set force scale must be bounded by the "
                "declared maximum absolute path load; displacement-only "
                "paths need a separate physical force reference"
            )
    if settings.augmentation_acceleration != "none" and not augmented:
        raise ValueError("augmentation acceleration requires augmented contact")
    if (settings.augmentation_acceleration == "active_secant_pivot"
            and report.profile not in {
                refined_q4_s3_augmented_profile, refined_q4_s3_roundoff_profile,
            }):
        raise ValueError("active-set pivot requires the refined Q4/S3 augmented patch profile")
    if (settings.augmentation_acceleration == "supported_schur_dev"
            and report.profile not in {
                refined_q4_s3_augmented_profile, refined_q4_s3_projected_profile,
                refined_q4_s3_overlap_local_profile,
            }):
        raise ValueError("supported Schur route requires the refined Q4/S3 augmented patch profile")
    profile = ("S3_MESH_EQUILIBRATED_MATERIAL_SUBINCREMENTS_PATH_DEV_V0"
               if history_friction else
               "S3_MESH_WEAK_P1_WHOLE_CHORD_GAP_FRICTION_MONOLITHIC_PATH_DEV_V0"
               if whole_chord_friction else
               "S3_MESH_WEAK_P1_PRESSURE_MATERIAL_FRICTION_MONOLITHIC_PATH_DEV_V0"
               if weak_pressure_friction else
               "S3_FULL_MESH_HARD_MATERIAL_FRICTION_MONOLITHIC_PATH_DEV_V0"
               if mesh_moving_friction else
               "S3_FULL_FACE_HARD_MATERIAL_FRICTION_MONOLITHIC_PATH_DEV_V0"
               if moving_friction else
               "Q4_Q4_COROTATIONAL_NODAL_DIRECTOR_MONOLITHIC_PATH_DEV_V0"
               if report.profile == finite_q4_profile else
               finite_v2d_path_profile
               if report.profile == finite_v2d_profile else
               finite_mixed_path_profile
               if report.profile == finite_mixed_profile else
               "Q4_Q4_PATCH_REFINED_MONOLITHIC_PATH_STATIC_DEV_V0"
               if report.profile == q4_q4_monolithic_profile else
               "Q4_Q4_PATCH_REFINED_PROJECTED_FINITE_PENALTY_PATH_DEV_V0"
               if report.profile == q4_q4_projected_profile else
               multiple_path_profile if report.profile == multiple_projected_profile else
               "S3_Q4_PROJECTED_FINITE_PENALTY_PATH_STATIC_DEV_V0"
               if report.profile == s3_q4_projected_penalty_profile else
               GENERAL_S3_PATCH_PROJECTED_PENALTY_PROFILE
               if report.profile == s3_projected_penalty_profile else
               GENERAL_Q4_S3_PATCH_REFINED_PROJECTED_PENALTY_PROFILE
               if report.profile == refined_q4_s3_projected_penalty_profile else
               GENERAL_Q4_Q4_PROFILE
               if report.profile == q4_pair_profile else
               GENERAL_Q4_S3_PATCH_PROFILE
               if report.profile == source_q4_patch_profile else
               GENERAL_S3_Q4_PATCH_PROFILE
               if report.profile == mixed_patch_profile else
               GENERAL_S3_Q4_PATCH_AUGMENTED_PROFILE
               if report.profile == mixed_augmented_profile else
               GENERAL_Q4_S3_PATCH_AUGMENTED_PROFILE
               if report.profile == source_q4_augmented_profile else
               GENERAL_Q4_S3_PATCH_PROJECTED_AUGMENTED_PROFILE
               if report.profile == source_q4_projected_augmented_profile else
               GENERAL_Q4_Q4_PATCH_AUGMENTED_PROFILE
               if report.profile == q4_pair_augmented_profile else
               GENERAL_Q4_Q4_PATCH_PROJECTED_AUGMENTED_PROFILE
               if report.profile == q4_pair_projected_augmented_profile else
               GENERAL_S3_PATCH_REFINED_AUGMENTED_PROFILE
               if report.profile == refined_augmented_profile else
               GENERAL_Q4_S3_PATCH_REFINED_AUGMENTED_PROFILE
               if report.profile == refined_q4_s3_augmented_profile else
               GENERAL_Q4_S3_PATCH_REFINED_PROJECTED_AUGMENTED_PROFILE
               if report.profile == refined_q4_s3_projected_profile else
               GENERAL_Q4_S3_PATCH_REFINED_OVERLAP_LOCAL_AUGMENTED_PROFILE
               if report.profile == refined_q4_s3_overlap_local_profile else
               GENERAL_Q4_S3_PATCH_REFINED_ROUNDOFF_AUGMENTED_PROFILE
               if report.profile == refined_q4_s3_roundoff_profile else
               GENERAL_Q4_S3_PATCH_REFINED_GRAPH_AUGMENTED_PROFILE
               if report.profile == refined_q4_s3_graph_profile else
               GENERAL_S3_PATCH_AUGMENTED_PROFILE
               if report.profile == general_augmented_profile else
               GENERAL_S3_PATCH_PROFILE if general_s3_patch else
               AUGMENTED_PROFILE if augmented else PROFILE)
    from anymaterial.isotropic import IsotropicMaterial

    for element in model.mesh.elements.values():
        material = model.get_material(element.material_name)
        if type(material) is not IsotropicMaterial or material.hardening_curve is not None:
            raise ValueError("contact path currently admits elastic isotropic materials only")
    fraction = float(max_path_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("max_path_fraction must lie in (0, 1]")
    requested_step = round(fraction * program.total_steps)
    if (requested_step < 1 or
            abs(requested_step / program.total_steps - fraction) > 1.0e-12):
        raise ValueError("max_path_fraction must end at a registered path step")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("num_layers must be a positive integer")
    if type(record_increment_snapshots) is not bool:
        raise TypeError("record_increment_snapshots must be bool")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    def operation(_lease):
        def run():
            # Entry safe point: a pre-cancelled token must be observed
            # before contact scope, boundary conditions, assembly and the
            # displacement force predictor, so cancellation never depends
            # on predictor admission.
            cancellation_safe_point(cancellation_token, "contact_path.entry")
            with contact_scope(model, definition, settings,
                               allow_augmented=augmented,
                               analysis_control="contact_path") as manager:
                if friction is not None:
                    from .s3_friction_static import create_s3_friction_manager
                    from .runtime import _ACTIVE_CONTACT
                    manager = create_s3_friction_manager(model, manager, friction)
                    # The enclosing scope restores its original context token.
                    _ACTIVE_CONTACT.set(manager)
                # Restricted sliding patch managers carry no macro pressure
                # law; pressure-support equilibrium tightening is gated on
                # this optional capability.
                manager_law = getattr(manager, "law", None)
                contract = {
                    "schema": "ANYSOLVER_STATIC_RESTART_CONTRACT_V2",
                    "control": "contact_path",
                    "profile": profile,
                    "general_contact": manager.restart_contract(),
                    "program": program.contract(),
                    "num_layers": num_layers,
                    "max_iterations": max_iterations,
                    "tolerance": float(tolerance),
                }
                if history_friction:
                    contract["material_history_path"] = history_policy
                if settings.augmentation_acceleration != "none":
                    contract["augmentation_acceleration"] = settings.augmentation_acceleration
                if settings.shell_pair_orientation != "legacy":
                    contract["shell_pair_orientation"] = settings.shell_pair_orientation
                if settings.path_predictor != "none":
                    contract["path_predictor"] = settings.path_predictor
                if finite:
                    from .finite_rotation_state import COORDINATES
                    if report.profile==finite_mixed_profile:
                        from .finite_mixed_state import COORDINATES,owner_contract
                        contract['finite_state_owners']=owner_contract(model)
                    elif report.profile==finite_v2d_profile:
                        from .finite_v2d_state import COORDINATES,owner_contract
                        contract['finite_state_owners']=owner_contract(model)
                    contract["element_state_coordinates"] = COORDINATES
                    contract["shell_skin_kinematics"] = settings.shell_skin_kinematics
                    contract["structural_kinematics"] = "corotational"
                    contract["corotational_tangent"] = "consistent"
                    contract["finite_constraint_domain"] = "zero_offset_selector_rotational_triplets_v1"
                    contract["finite_trial_sweep"] = "every_actual_binary_candidate_chord_v1"
                    contract["finite_rigid_seed"] = "common_reference_frame_left_transport_project_controls_v1"
                    if settings.finite_frame_seed != 'common_reference_frame_dev':
                        from .finite_rotation_predictor import per_surface_seed_descriptor
                        finite_frames=per_surface_seed_descriptor(model,definition,program)
                        contract['finite_frame_seed']=settings.finite_frame_seed
                        contract['finite_rigid_seed']='per_surface_left_transport_project_controls_v1'
                        contract['finite_surface_frames']=list(finite_frames)
                if semismooth:
                    contract["enforcement_solver"] = "monolithic_active_set"
                    contract["control_residual_scaling"] = "per_control_waypoint_max_v1"
                if settings.path_continuation_maximum_halvings:
                    # Enabled continuation binds its numerical revision,
                    # settings and owned state transaction into the analysis
                    # identity so an incompatible checkpoint fails validation
                    # before any Newton work; the disabled contract keeps its
                    # exact prior shape.
                    contract["contact_path_continuation"] = {
                        "revision": _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION,
                        "state_transaction": "owned_elastic_mapping_macro_transaction_v1",
                        "maximum_halvings": settings.path_continuation_maximum_halvings,
                        "stagnation_patience": settings.path_stagnation_patience,
                        "stagnation_fraction": settings.path_stagnation_fraction,
                        "stagnation_merit_rate": settings.path_stagnation_merit_rate,
                    }
                if settings.initial_contact_seed != "none":
                    # The enabled seed binds its revision and selection into
                    # the analysis identity so an incompatible checkpoint
                    # fails validation before any Newton work; the disabled
                    # contract keeps its exact prior shape.
                    contract["contact_initial_seed"] = {
                        "revision": (
                            _INITIAL_CONTACT_SEED_GLOBALIZATION_REVISION
                            if settings.initial_contact_seed ==
                            "linearized_kkt_merit_globalization_dev"
                            else _INITIAL_CONTACT_SEED_REVISION),
                        "selection": settings.initial_contact_seed,
                    }
                if displacement_only:
                    contract["force_reference_method"] = (
                        "contact_free_elastic_displacement_v1"
                    )
                validated = None
                if restart_checkpoint is not None:
                    validated = validate_nonlinear_checkpoint(
                        restart_checkpoint, analysis_kind="static", model=model,
                        analysis_contract=contract, num_layers=num_layers,
                    )
                model.apply_boundary_conditions()
                stiffness, _stiffness_info = assemble_stiffness_matrix(model)
                zeros = np.zeros(model.mesh.dof_manager.total_dofs)
                _, _, T, u0, _independent, _constraint_info = (
                    build_constraint_transformation(stiffness, zeros, model)
                )
                if finite:
                    selector = T.tocsr()
                    selector.eliminate_zeros()
                    if (np.any(np.diff(selector.indptr)>1)
                            or np.any(selector.data != 1.) or np.any(u0 != 0.)
                            or np.any(np.bincount(selector.indices,minlength=selector.shape[1])>1)):
                        raise ValueError("finite shell path requires zero-offset selector constraints without MPC rows")
                if hasattr(manager, "restrict_tangent_to_reduced_variations"):
                    manager.restrict_tangent_to_reduced_variations(T)
                controls = np.zeros((len(program.controls), len(zeros)))
                for index, control in enumerate(program.controls):
                    for node_id, dof_name, weight in control.terms:
                        if model.mesh.get_node(node_id) is None:
                            raise ValueError(f"missing contact path node {node_id}")
                        dof = model.mesh.dof_manager.get_node_dofs(node_id)[
                            (_FINITE_DOF if finite else _DOF)[dof_name]
                        ]
                        controls[index, dof] += weight
                rows = np.asarray(controls @ T, dtype=float)
                # Empty controls and controls projected into an empty reduced
                # space have rank zero. Nonempty dependent controls still fail.
                rank = np.linalg.matrix_rank(rows) if rows.size else 0
                if rank != len(program.controls):
                    raise ValueError("contact path controls are constrained or dependent")
                if finite:
                    from scipy.linalg import null_space
                    variation = T @ null_space(rows)
                    for node in model.mesh.nodes:
                        dofs = model.mesh.dof_manager.get_node_dofs(node)[3:6]
                        block = np.asarray(variation[dofs])
                        rank = np.linalg.matrix_rank(block, tol=256*np.finfo(float).eps
                                                     *max(1., float(np.linalg.norm(block))))
                        if rank not in {0, 3}:
                            raise ValueError("finite shell rotations must be all three constrained or all three free per node")
                if np.any(np.abs(controls @ u0) > 1.0e-12):
                    raise ValueError("contact path requires zero reference control offsets")
                displacement_force_predictor = None
                if displacement_only:
                    displacement_force_predictor = (
                        _elastic_displacement_force_predictor(
                            stiffness, T, rows, program.waypoints[1:],
                            allow_zero=(not semismooth
                                        and manager.profile in {
                                            s3_projected_penalty_profile,
                                            s3_q4_projected_penalty_profile,
                                        }),
                        )
                    )
                    if (semismooth and settings.complementarity_force_scale_n
                            > displacement_force_predictor*(1.0+1.0e-12)):
                        raise ValueError(
                            "monolithic displacement force scale exceeds "
                            "the contact-free elastic reaction predictor "
                            f"(declared={settings.complementarity_force_scale_n:.9g} N, "
                            f"predictor={displacement_force_predictor:.9g} N)"
                        )
                reactions = rows.T
                force_dofs = np.empty(len(program.nodal_forces), dtype=np.intp)
                for index, force in enumerate(program.nodal_forces):
                    if model.mesh.get_node(force.node_id) is None:
                        raise ValueError(f"missing contact path force node {force.node_id}")
                    dof = model.mesh.dof_manager.get_node_dofs(force.node_id)[
                        _DOF[force.dof]
                    ]
                    force_dofs[index] = dof
                info = {"control": "contact_path", "profile": profile,
                        "general_contact_profile": manager.profile}
                if displacement_force_predictor is not None:
                    info["contact_free_elastic_force_predictor_n"] = (
                        displacement_force_predictor
                    )
                accepted_step = 0
                multipliers = np.zeros(len(program.controls))
                normal_forces = np.zeros(len(manager.law.pressure_mode_ids)) if semismooth else None
                previous_steps: list[NonlinearStaticStep] = []
                snapshots = []
                history: list[dict[str, object]] = []
                total_iterations = 0
                if restart_checkpoint is None:
                    q = np.zeros(T.shape[1])
                    committed_states, _ = _prepare_initial_states(
                        model, None, None, num_layers,
                    )
                else:
                    assert validated is not None
                    state = validated.path_state
                    expected_keys = {"mode", "accepted_step", "multipliers",
                                     "reduced_coordinates", "steps", "history",
                                     "total_iterations", "terminal_status"}
                    if augmented:
                        expected_keys.add("normal_multipliers")
                    if semismooth:
                        expected_keys.add("normal_forces")
                    if friction is not None:
                        expected_keys.add("friction_state")
                    if set(state) != expected_keys or state["mode"] != "contact_path":
                        raise ValueError("contact path checkpoint state is incompatible")
                    if state["terminal_status"] != "completed":
                        raise ValueError("contact path checkpoint is not continuable")
                    accepted_step = int(state["accepted_step"])
                    if not 0 <= accepted_step <= program.total_steps:
                        raise ValueError("contact path checkpoint step is invalid")
                    q = np.asarray(state["reduced_coordinates"], dtype=float)
                    multipliers = np.asarray(state["multipliers"], dtype=float)
                    if (q.shape != (T.shape[1],) or
                            multipliers.shape != (len(program.controls),) or
                            not np.all(np.isfinite(q)) or
                            not np.all(np.isfinite(multipliers))):
                        raise ValueError("contact path checkpoint coordinates are invalid")
                    if not np.array_equal(
                        np.asarray(T @ q + u0).reshape(-1),
                        validated.displacements,
                    ):
                        raise ValueError("contact path checkpoint pose is incompatible")
                    if not np.allclose(rows @ q, program.target_at(accepted_step),
                                       rtol=0.0, atol=1.0e-10):
                        raise ValueError("contact path checkpoint target is incompatible")
                    previous_steps = [_static_step_from_restart(item)
                                      for item in state["steps"]]
                    history = copy.deepcopy(state["history"])
                    total_iterations = int(state["total_iterations"])
                    if (len(previous_steps) != accepted_step or
                             len(history) != accepted_step or total_iterations < 0):
                        raise ValueError("contact path checkpoint history is incomplete")
                    if settings.path_continuation_maximum_halvings:
                        for item in history:
                            _validate_continuation_history_record(
                                item, program, settings)
                    if augmented:
                        manager.set_normal_multipliers(
                            np.asarray(state["normal_multipliers"], dtype=float)
                        )
                    if semismooth:
                        normal_forces = np.asarray(state["normal_forces"], dtype=float)
                        if (normal_forces.shape != (len(manager.law.pressure_mode_ids),)
                                or not np.all(np.isfinite(normal_forces))
                                or np.any(normal_forces < 0.0)):
                            raise ValueError("contact path checkpoint nodal forces are invalid")
                    committed_states = validated.element_states
                    if friction is not None:
                        manager.restore(state["friction_state"], validated.displacements)
                if history_friction:
                    from .s3_friction_history_path import register_entry, summarize_work_history
                    register_entry(manager)
                if rim_friction:
                    from .s3_friction_rim_transaction import register_entry
                    register_entry(manager)
                if accepted_step > requested_step:
                    raise ValueError("contact path checkpoint is beyond requested target")
                initial_u = np.asarray(T @ q + u0).reshape(-1)
                if friction is not None and getattr(manager,"certified_material_path",False):
                    manager.require_friction_path(initial_u,initial_u)
                if finite:
                    manager.require_finite_deformation(initial_u)
                    if not manager._swept_admissible(initial_u, initial_u):
                        raise ValueError("finite shell initial/restart pose is unsafe or uncertified")
                if semismooth and restart_checkpoint is not None:
                    source, target_points = manager._positions(initial_u)
                    support_area = manager.law.geometry.evaluate(
                        source, target_points,
                    ).reference_support
                    if np.any((support_area <= 0.0) & (normal_forces != 0.0)):
                        raise ValueError("contact path checkpoint nodal forces are unsupported")
                    expected_pressure = np.zeros_like(support_area)
                    np.divide(normal_forces, support_area, out=expected_pressure,
                              where=support_area > 0.0)
                    if not np.allclose(manager.normal_multipliers, expected_pressure,
                                       rtol=1.0e-10, atol=1.0e-12):
                        raise ValueError("contact path checkpoint pressure and nodal forces are incompatible")
                if finite:
                    from .finite_rotation_state import prepare_finite_states
                    if report.profile==finite_mixed_profile:
                        from .finite_mixed_state import prepare_mixed_states as prepare_finite_states
                    elif report.profile==finite_v2d_profile:
                        from .finite_v2d_state import prepare_v2d_states as prepare_finite_states
                    committed_states = prepare_finite_states(
                        model, initial_u, committed_states, num_layers, info,
                        restart=restart_checkpoint is not None)
                else:
                    committed_states = _prepare_qualified_q4_states_for_nonlinear_solve(
                        model, initial_u, committed_states, num_layers, info,
                        supplied_element_ids=tuple(sorted(int(k) for k in committed_states)),
                        ordinary_restart=restart_checkpoint is not None,
                    )
                continuation_active = (semismooth
                                       and settings.path_continuation_maximum_halvings > 0)
                if continuation_active:
                    # Explicit eligibility diagnosis before any state
                    # activation: the continuation transaction owns one
                    # plain elastic mapping. Actual foreign, store-backed or
                    # formulation-native state owners are rejected here
                    # instead of being silently ignored; the ordinary
                    # activation's elastic store creation is not a refusal
                    # reason by itself.
                    if friction is not None:
                        raise ValueError(
                            "path continuation requires the frictionless elastic route")
                    if (type(committed_states) is not dict
                            or isinstance(committed_states, NonlinearStateStore)):
                        raise ValueError(
                            "path continuation requires its owned plain mapping state transaction")
                    if any(bool(getattr(element,
                                         "formulation_native_total_lagrangian", False))
                           for element in model.mesh.elements.values()):
                        raise ValueError(
                            "path continuation rejects formulation-native state owners")
                if history_friction or rim_friction:
                    committed_states = copy.deepcopy(dict(committed_states))
                    info["nonlinear_state_storage"] = {"mode": "validated_owned_mapping_invocation_publication_v1"}
                elif continuation_active:
                    committed_states = copy.deepcopy(dict(committed_states))
                    info["nonlinear_state_storage"] = {"mode": "owned_elastic_mapping_macro_transaction_v1"}
                else:
                    committed_states = _activate_nonlinear_state_storage(
                        model, committed_states, num_layers, info,
                        kinematics=structural_kinematics, committed_displacements=initial_u,
                    )
                status = "completed"
                failure_reason = None
                scale = max(1.0e-9, max((abs(value) for row in program.waypoints
                                         for value in row), default=0.0))
                control_scales = np.maximum(
                    1.0e-9, np.max(np.abs(np.asarray(program.waypoints)), axis=0),
                ) if semismooth else None
                if semismooth:
                    info["contact_path_control_scales_m"] = control_scales.tolist()
                    if finite:
                        info["contact_path_control_scales_m_rad"] = info.pop("contact_path_control_scales_m")

                    def attempt_semismooth(step_index, attempt_target,
                                           attempt_force_values, origin_q,
                                           origin_multipliers,
                                           origin_normal_forces, origin_states,
                                           stagnation_controls,
                                           initial_contact_seed=None):
                        """One complete coupled semismooth admission attempt.

                        Solves the increment from the supplied physical origin
                        and runs every existing report/raw-certificate/D1 gate
                        before the coordinated commit, so the default macro
                        attempt and every continuation substep share one
                        acceptance path. The attempt never mutates the
                        caller's origin: on any refusal it returns the
                        declared reason without committing (the caller owns
                        rollback), and on success it returns the complete
                        accepted physical state.
                        """
                        attempt_external = np.zeros_like(zeros)
                        attempt_external[force_dofs] = attempt_force_values
                        attempt_reduced_external = np.asarray(
                            T.T @ attempt_external).reshape(-1)
                        step_result = solve_semismooth_increment(
                            model=model, manager=manager, settings=settings,
                            q=origin_q, reactions=origin_multipliers,
                            nodal_forces=origin_normal_forces,
                            committed_states=origin_states,
                            transform=T, offset=u0, control_rows=rows,
                            control_reactions=reactions, target=attempt_target,
                            reduced_external=attempt_reduced_external,
                            num_layers=num_layers, max_iterations=max_iterations,
                            tolerance=tolerance, control_scales=control_scales,
                            path_force_magnitude=float(np.sum(np.abs(attempt_force_values))),
                            cancellation_token=cancellation_token, step=step_index,
                            initial_contact_seed=initial_contact_seed,
                            **({"stagnation_patience": stagnation_controls["patience"],
                                "stagnation_fraction": stagnation_controls["fraction"],
                                "stagnation_merit_rate": stagnation_controls["merit_rate"]}
                               if stagnation_controls is not None else {}),
                        )
                        refused = {"completed": False,
                                   "solver_refusal": True,
                                   "reason": step_result.reason,
                                   "diagnostics": step_result.diagnostics,
                                   "iterations": step_result.iterations,
                                   "candidate_evaluations":
                                   step_result.candidate_evaluations,
                                   "backtracks": step_result.backtracks,
                                   "initial_seed": step_result.initial_seed}
                        if not step_result.completed:
                            return refused
                        attempt_q = step_result.q
                        attempt_multipliers = step_result.reactions
                        attempt_normal_forces = step_result.nodal_forces
                        accepted_u = np.asarray(T @ attempt_q + u0).reshape(-1)
                        raw_contact = step_result.contact
                        support_area = np.asarray(raw_contact.state.reference_support)
                        pressure = np.zeros_like(support_area)
                        np.divide(attempt_normal_forces, support_area, out=pressure,
                                  where=support_area > 0.0)
                        manager.set_normal_multipliers(pressure)
                        reported_force, _reported_tangent, certificate = (
                            manager.evaluate(accepted_u, tangent=False)
                        )
                        force_scale = max(1.0, float(np.linalg.norm(
                            attempt_reduced_external + reactions @ attempt_multipliers)))
                        if (np.linalg.norm(reported_force-raw_contact.force)
                                > tolerance*force_scale
                                or not certificate.gap_tolerance_met
                                or manager.has_unresolved_weak_open_gap(
                                    certificate, manager.normal_multipliers)):
                            refused.update(
                                reason="contact_semismooth_report_or_gap_mismatch",
                                solver_refusal=False,
                                diagnostics=None)
                            return refused
                        projected_pair_evaluations = certificate.pair_geometry_evaluations
                        certificate = raw_certificate(
                            manager, accepted_u, raw_contact, certificate,
                        )
                        admission = getattr(manager,
                                            "d1_interior_admission_issue", None)
                        issue = (admission(certificate, accepted_u, T, control_rows=rows)
                                 if admission is not None else None)
                        if issue is not None:
                            refused.update(
                                reason="contact_d1_interior_profile_rejected",
                                solver_refusal=False,
                                diagnostics={"step_index": step_index,
                                              "issue": issue})
                            return refused
                        friction_successor = (manager.validate_successor(accepted_u,attempt_normal_forces)
                                              if moving_friction else None)
                        if history_friction:
                            # Owned scalar candidate mappings have no persistent trial token.
                            # Recover before the physical tuple assignment; the invocation
                            # guard also restores entry on post-commit/finalization errors.
                            with suspend_contact():
                                recovered, _K, _states = _assemble_nonlinear_system(
                                    model, accepted_u, step_result.candidate_states, num_layers,
                                    tangent=False, kinematics=structural_kinematics,
                                    **assembly_options, require_full_coordinates=True,
                                )
                        accepted_states = _commit_nonlinear_state_candidate(
                            origin_states, step_result.candidate_states,
                            model=model, accepted_displacements=accepted_u,
                        )
                        if friction_successor is not None:
                            manager.commit(friction_successor)
                        if not history_friction:
                            with suspend_contact():
                                recovered, _K, _states = _assemble_nonlinear_system(
                                    model, accepted_u, accepted_states, num_layers,
                                    tangent=False, kinematics=structural_kinematics,
                                    **assembly_options,
                                    require_full_coordinates=True,
                                )
                        _discard_nonlinear_state_candidate(accepted_states)
                        support = _support_reaction_resultants(
                            model, recovered + raw_contact.force
                            - controls.T @ attempt_multipliers - attempt_external,
                        )
                        return {"completed": True,
                                "q": attempt_q,
                                "multipliers": attempt_multipliers,
                                "normal_forces": attempt_normal_forces,
                                "committed_states": accepted_states,
                                "accepted_u": accepted_u,
                                "certificate": certificate,
                                "projected_pair_evaluations":
                                projected_pair_evaluations,
                                "support": support,
                                "friction_successor": friction_successor,
                                "iterations": step_result.iterations,
                                "candidate_evaluations":
                                step_result.candidate_evaluations,
                                "backtracks": step_result.backtracks,
                                "residual_norm": step_result.residual_norm,
                                "constraint_norm": step_result.constraint_norm,
                                "complementarity_norm":
                                step_result.complementarity_norm,
                                "initial_seed": step_result.initial_seed}
                for step in range(accepted_step + 1, requested_step + 1):
                    cancellation_safe_point(cancellation_token,
                                            f"contact_path.step:{step}")
                    start_q = q.copy()
                    start_multipliers = multipliers.copy()
                    start_normal_multipliers = manager.normal_multipliers.copy() if augmented else None
                    start_normal_forces = normal_forces.copy() if semismooth else None
                    target = program.target_at(step)
                    if finite:
                        from .finite_rotation_predictor import finite_rigid_seed
                        try:
                            if settings.finite_frame_seed=='common_reference_frame_dev':
                                seed,seed_info=finite_rigid_seed(model,program,target,T,u0,start_q,rows)
                            else:
                                from .finite_rotation_predictor import per_surface_rigid_seed
                                seed,seed_info=per_surface_rigid_seed(
                                    model,program,target,T,u0,start_q,rows,finite_frames)
                            start_u=np.asarray(T@start_q+u0).reshape(-1)
                            seed_u=np.asarray(T@seed+u0).reshape(-1)
                            cancellation_safe_point(cancellation_token,f"contact_path.finite_seed:{step}")
                            if not manager._swept_admissible(start_u,seed_u):
                                raise ValueError('finite rigid seed whole path is unsafe or uncertified')
                            manager.require_finite_deformation(seed_u)
                        except ValueError as exc:
                            failure_reason='finite_rigid_seed_refused'
                            info['finite_rigid_seed_failure']=dict(step=step,reason=str(exc))
                            status='stopped_at_limit' if previous_steps else 'diverged'
                            break
                        q=seed
                        info.setdefault('finite_rigid_seed_history',[]).append(dict(step=step,**seed_info))
                    if settings.path_predictor == "control_projection_dev":
                        seed = _control_projection_seed(rows, start_q, target)
                        start_u = np.asarray(T @ start_q + u0).reshape(-1)
                        seed_u = np.asarray(T @ seed + u0).reshape(-1)
                        # Admit the complete seed route before evaluating any
                        # structural/material/contact candidate at the seed.
                        seed_fraction = manager.admissible_step_fraction(start_u, seed_u)
                        cancellation_safe_point(cancellation_token,
                                                f"contact_path.predictor:{step}")
                        if seed_fraction != 1.0:
                            failure_reason = "contact_path_predictor_gap_exceeded"
                            info["contact_path_predictor_failure"] = {
                                "step": step, "admissible_fraction": float(seed_fraction),
                            }
                            status = "stopped_at_limit" if previous_steps else "diverged"
                            break
                        q = seed
                    force_values = program.force_values_at(step)
                    external = np.zeros_like(zeros)
                    external[force_dofs] = force_values
                    reduced_external = np.asarray(T.T @ external).reshape(-1)
                    if semismooth:
                        stagnation_controls = (
                            {"patience": settings.path_stagnation_patience,
                             "fraction": settings.path_stagnation_fraction,
                             "merit_rate": settings.path_stagnation_merit_rate}
                            if continuation_active else None)
                        macro_origin_states = (
                            copy.deepcopy(dict(committed_states))
                            if continuation_active else None)
                        # The seed is initial-only: exactly the first macro
                        # attempt of a fresh (non-restart) solve on the
                        # frictionless reference-linear D1 overlap-local
                        # route, whose origin is the stress-free reference
                        # pose with zero control reactions and zero normal
                        # force. Continuation substeps and restarts never
                        # request it, so the behavior is deterministic.
                        initial_seed_selection = (
                            settings.initial_contact_seed
                            if (restart_checkpoint is None and step == 1
                                and friction is None and not finite
                                and report.profile
                                == refined_q4_s3_overlap_local_profile)
                            else None)
                        try:
                            attempt = attempt_semismooth(
                                step, target, force_values, q, multipliers,
                                normal_forces, committed_states,
                                stagnation_controls,
                                initial_contact_seed=initial_seed_selection)
                        except BaseException:
                            if continuation_active:
                                # Cancellation or exception anywhere in the
                                # full macro attempt - including pressure,
                                # report, commit and recovery - restores the
                                # complete macro origin; no partial macro
                                # publication survives. The default path
                                # keeps its existing exception behavior.
                                _discard_nonlinear_state_candidate(
                                    committed_states)
                                q = start_q
                                multipliers = start_multipliers
                                normal_forces = start_normal_forces
                                manager.set_normal_multipliers(
                                    start_normal_multipliers)
                                committed_states = macro_origin_states
                            raise
                        if attempt["initial_seed"] is not None:
                            info["contact_initial_seed"] = attempt["initial_seed"]
                        total_iterations += attempt["iterations"]
                        summed_iterations = attempt["iterations"]
                        summed_candidate_evaluations = attempt["candidate_evaluations"]
                        summed_backtracks = attempt["backtracks"]
                        continuation_record = None
                        if not attempt["completed"]:
                            _discard_nonlinear_state_candidate(committed_states)
                            if not continuation_active:
                                failure_reason = attempt["reason"]
                                if (attempt["solver_refusal"]
                                        or attempt["diagnostics"] is not None):
                                    info["contact_failure"] = attempt["diagnostics"]
                                q = start_q
                                multipliers = start_multipliers
                                normal_forces = start_normal_forces
                                manager.set_normal_multipliers(start_normal_multipliers)
                                status = "stopped_at_limit" if previous_steps else "diverged"
                                break
                            # Declared cutback: restore the complete macro
                            # origin, then retry the same declared macro
                            # segment as bounded physical dyadic substeps
                            # along the program's own interpolation of
                            # targets and nodal loads. A Newton line-search
                            # alpha is never a physical load fraction.
                            q = start_q
                            multipliers = start_multipliers
                            normal_forces = start_normal_forces
                            manager.set_normal_multipliers(start_normal_multipliers)
                            committed_states = macro_origin_states
                            continuation_record = {
                                "mode": "declared_physical_dyadic_substeps",
                                "revision": _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION,
                                "cutback_reason": attempt["reason"],
                                "maximum_halvings":
                                settings.path_continuation_maximum_halvings,
                                "substeps": [],
                            }
                            previous_target = program.target_at(step-1)
                            previous_force_values = program.force_values_at(step-1)
                            boundary = {
                                "q": start_q,
                                "multipliers": start_multipliers,
                                "normal_forces": start_normal_forces,
                                "normal_multipliers": start_normal_multipliers,
                                "force_values": previous_force_values,
                                "u": np.asarray(T @ start_q + u0).reshape(-1),
                            }
                            tau = 0.0
                            span = 0.5
                            halvings = 1
                            trials = 0
                            trial_limit = (
                                (1 << (settings.path_continuation_maximum_halvings + 1))
                                + 2*settings.path_continuation_maximum_halvings + 2)
                            substep_work = 0.0
                            exhausted = None
                            try:
                                while tau < 1.0 and exhausted is None:
                                    trials += 1
                                    if trials > trial_limit:
                                        exhausted = (
                                            "contact_path_continuation_trial_limit")
                                        break
                                    tau_trial = (1.0 if tau + span >= 1.0
                                                 else tau + span)
                                    sub_target = ((1.0 - tau_trial)*previous_target
                                                  + tau_trial*target)
                                    sub_force_values = (
                                        (1.0 - tau_trial)*previous_force_values
                                        + tau_trial*force_values)
                                    cancellation_safe_point(
                                        cancellation_token,
                                        f"contact_path.step:{step}.continuation:{trials}")
                                    sub_attempt = attempt_semismooth(
                                        step, sub_target, sub_force_values,
                                        boundary["q"], boundary["multipliers"],
                                        boundary["normal_forces"],
                                        committed_states, stagnation_controls)
                                    total_iterations += sub_attempt["iterations"]
                                    summed_iterations += sub_attempt["iterations"]
                                    summed_candidate_evaluations += (
                                        sub_attempt["candidate_evaluations"])
                                    summed_backtracks += sub_attempt["backtracks"]
                                    if sub_attempt["completed"]:
                                        accepted_sub_u = sub_attempt["accepted_u"]
                                        sub_work = None
                                        if program.nodal_forces:
                                            sub_work = float(
                                                0.5*(boundary["force_values"]
                                                     + sub_force_values)
                                                @ (accepted_sub_u
                                                   - boundary["u"])[force_dofs])
                                            substep_work += sub_work
                                        continuation_record["substeps"].append({
                                            "tau_start": float(tau),
                                            "tau_end": float(tau_trial),
                                            "iterations": sub_attempt["iterations"],
                                            "residual_norm":
                                            sub_attempt["residual_norm"],
                                            "constraint_error":
                                            sub_attempt["constraint_norm"],
                                            "complementarity_relative_force":
                                            sub_attempt["complementarity_norm"],
                                            "candidate_evaluations":
                                            sub_attempt["candidate_evaluations"],
                                            "backtracks": sub_attempt["backtracks"],
                                            **({"external_nodal_work_trapezoid_j":
                                                sub_work}
                                               if sub_work is not None else {}),
                                        })
                                        tau = tau_trial
                                        span = min(2.0*span, 0.5)
                                        attempt = sub_attempt
                                        committed_states = (
                                            sub_attempt["committed_states"])
                                        boundary = {
                                            "q": sub_attempt["q"],
                                            "multipliers":
                                            sub_attempt["multipliers"],
                                            "normal_forces":
                                            sub_attempt["normal_forces"],
                                            "normal_multipliers":
                                            manager.normal_multipliers.copy(),
                                            "force_values": sub_force_values,
                                            "u": accepted_sub_u,
                                        }
                                    else:
                                        _discard_nonlinear_state_candidate(
                                            committed_states)
                                        if (halvings
                                                >= settings.path_continuation_maximum_halvings):
                                            exhausted = (
                                                "contact_path_continuation_halvings_exhausted")
                                            continue
                                        halvings += 1
                                        span *= 0.5
                                        q = boundary["q"]
                                        multipliers = boundary["multipliers"]
                                        normal_forces = boundary["normal_forces"]
                                        manager.set_normal_multipliers(
                                            boundary["normal_multipliers"])
                            except BaseException:
                                # Failure, exception or cancellation at any
                                # substep restores the complete macro origin;
                                # no partial macro publication survives.
                                _discard_nonlinear_state_candidate(
                                    committed_states)
                                q = start_q
                                multipliers = start_multipliers
                                normal_forces = start_normal_forces
                                manager.set_normal_multipliers(
                                    start_normal_multipliers)
                                committed_states = macro_origin_states
                                raise
                            if exhausted is not None:
                                failure_reason = exhausted
                                info["contact_failure"] = {
                                    "step_index": step,
                                    "cutback_reason":
                                    continuation_record["cutback_reason"],
                                    "halvings": halvings,
                                    "trials": trials,
                                    "accepted_substeps":
                                    len(continuation_record["substeps"]),
                                }
                                q = start_q
                                multipliers = start_multipliers
                                normal_forces = start_normal_forces
                                manager.set_normal_multipliers(
                                    start_normal_multipliers)
                                committed_states = macro_origin_states
                                status = ("stopped_at_limit" if previous_steps
                                          else "diverged")
                                break
                            continuation_record.update({
                                "halvings_used": halvings,
                                "trial_count": trials,
                                "accepted_substep_count":
                                len(continuation_record["substeps"]),
                                "stagnation_patience":
                                settings.path_stagnation_patience,
                                "stagnation_fraction":
                                settings.path_stagnation_fraction,
                                "stagnation_merit_rate":
                                settings.path_stagnation_merit_rate,
                            })
                        q = attempt["q"]
                        multipliers = attempt["multipliers"]
                        normal_forces = attempt["normal_forces"]
                        committed_states = attempt["committed_states"]
                        accepted_u = attempt["accepted_u"]
                        certificate = attempt["certificate"]
                        projected_pair_evaluations = attempt["projected_pair_evaluations"]
                        support = attempt["support"]
                        friction_successor = attempt["friction_successor"]
                        accepted_step = step
                        record = {
                            "step_index": step,
                            **({"user_increment_position": step / friction.physical_subincrements_per_user_increment,
                                "completed_user_increments": step // friction.physical_subincrements_per_user_increment}
                               if history_friction else {}),
                            **({"friction":friction_successor[2]} if friction_successor is not None else {}),
                            "target": target.tolist(),
                            "reaction_generalized": multipliers.tolist(),
                            "residual_norm": attempt["residual_norm"],
                            "constraint_error": attempt["constraint_norm"],
                            "augmentation_iterations": 0,
                            "semismooth_iterations": summed_iterations,
                            "semismooth_linear_solver": settings.monolithic_linear_solver,
                            "semismooth_maximum_pivots": settings.monolithic_maximum_pivots,
                            "complementarity_relative_force":
                            attempt["complementarity_norm"],
                            "candidate_evaluations": summed_candidate_evaluations,
                            "projected_diagnostic_pair_evaluations":
                            projected_pair_evaluations,
                            "backtracks": summed_backtracks,
                            "contact": certificate.to_dict(),
                        }
                        if continuation_record is not None:
                            record["continuation"] = continuation_record
                        if attempt["initial_seed"] is not None:
                            record["initial_seed"] = attempt["initial_seed"]
                        if program.nodal_forces:
                            previous_force_values = program.force_values_at(step-1)
                            start_u = np.asarray(T @ start_q + u0).reshape(-1)
                            record["nodal_forces_n"] = force_values.tolist()
                            if continuation_record is not None:
                                # Summed per-substep trapezoid over the
                                # actually accepted physical substep path.
                                record["external_nodal_work_trapezoid_j"] = float(
                                    substep_work)
                                record["external_nodal_work_scope"] = (
                                    "continuation_substep_trapezoid"
                                    "_not_exact_path_integral")
                            else:
                                record["external_nodal_work_trapezoid_j"] = float(
                                    0.5*(previous_force_values + force_values)
                                    @ (accepted_u-start_u)[force_dofs]
                                )
                        if history_friction:
                            record['control_work_trapezoid_j'] = float(
                                .5*(start_multipliers+multipliers) @ (program.target_at(step)-program.target_at(step-1)))
                            record['load_work_scope'] = 'accepted_subincrement_trapezoid_not_exact_path_integral'
                        history.append(record)
                        previous_steps.append(NonlinearStaticStep(
                            step_index=step,
                            load_factor=step / program.total_steps,
                            iterations=summed_iterations,
                            residual_norm=attempt["residual_norm"],
                            displacement_norm=float(np.linalg.norm(accepted_u)),
                            max_equivalent_plastic_strain=_max_plastic_strain(
                                committed_states),
                            active_stage=f"segment_{(step - 1) // program.steps_per_segment + 1}",
                            support_reactions=support,
                        ))
                        if record_increment_snapshots:
                            snapshots.append(_increment_snapshot(
                                step, step / program.total_steps,
                                accepted_u, committed_states,
                                control_value=(float(target[0])
                                               if target.size else None),
                            ))
                        continue
                    candidate_states = committed_states
                    normal_multiplier_change = 0.0
                    accelerator = (ProjectedAugmentationSecant()
                                   if settings.augmentation_acceleration in {
                                       "active_secant", "active_secant_pivot",
                                   }
                                   else None)
                    pivot_attempts = 0
                    pivot_applied = 0
                    pivot_rejected = 0
                    schur_relinearizations = 0
                    schur_epoch_attempts = 0
                    schur_epoch_relinearizations = 0
                    schur_active_signature = None
                    pressure_map_attempts = 0
                    pivot_backup = None
                    schur_backtrack_source = None
                    schur_backtrack_signature = None
                    support_resets: list[dict[str, object]] = []
                    augmentation_trace: list[dict[str, object]] = []
                    certificate = None
                    residual_norm = float("inf")
                    constraint_norm = float("inf")
                    iteration = 0
                    augmentation_limit = (settings.maximum_augmentations
                                          if augmented else 1)
                    for augmentation in range(1, augmentation_limit + 1):
                        cancellation_safe_point(
                            cancellation_token,
                            f"contact_path.step:{step}.augmentation:{augmentation}",
                        )
                        equilibrated = False
                        for iteration in range(1, max_iterations + 1):
                            cancellation_safe_point(
                                cancellation_token,
                                f"contact_path.step:{step}.augmentation:{augmentation}.iteration:{iteration}",
                            )
                            total_iterations += 1
                            u = np.asarray(T @ q + u0).reshape(-1)
                            try:
                                if friction is not None and getattr(manager,"certified_material_path",False):
                                    manager.require_friction_path(manager.committed_displacement,u)
                                internal, tangent, trial = _assemble_nonlinear_system(
                                    model, u, committed_states, num_layers,
                                    kinematics=structural_kinematics,
                                    **assembly_options,
                                )
                            except UnsupportedQ4S3Branch as exc:
                                candidate_evaluation_error = str(exc)
                                failure_reason = (
                                    "contact_candidate_evaluation_failed"
                                    if pivot_backup is not None else
                                    "contact_derivative_branch_unresolved"
                                )
                                if pivot_backup is None:
                                    augmentation_trace.append({
                                        "augmentation": augmentation,
                                        "derivative_branch_failure": str(exc),
                                    })
                                break
                            except ValueError as exc:
                                if (pivot_backup is None or
                                        settings.augmentation_acceleration
                                        != "supported_schur_dev"):
                                    raise
                                failure_reason = "contact_candidate_evaluation_failed"
                                candidate_evaluation_error = str(exc)
                                break
                            residual = (reduced_external + reactions @ multipliers
                                        - np.asarray(T.T @ internal).reshape(-1))
                            constraint = target - rows @ q
                            residual_norm = float(np.linalg.norm(residual))
                            constraint_norm = float(np.max(np.abs(constraint)))
                            force_scale = max(1.0, float(np.linalg.norm(
                                reduced_external + reactions @ multipliers)))
                            equilibrium_force_limit = tolerance * force_scale
                            if (residual_norm <= equilibrium_force_limit and
                                    constraint_norm <= tolerance * scale):
                                proposed_certificate = manager.evaluate(
                                    u, tangent=False,
                                )[2]
                                if (augmented and getattr(
                                        manager_law, "pressure_basis", None) is not None):
                                    supported_area = np.asarray(
                                        proposed_certificate.reference_support_m2,
                                        dtype=float,
                                    )
                                    supported_area = supported_area[supported_area > 0.0]
                                    if len(supported_area):
                                        equilibrium_force_limit = min(
                                            equilibrium_force_limit,
                                            0.1*settings.pressure_tolerance
                                            *float(np.min(supported_area)),
                                        )
                                if residual_norm <= equilibrium_force_limit:
                                    certificate = proposed_certificate
                                    candidate_states = trial
                                    equilibrated = True
                                    break
                            reduced_tangent = (T.T @ tangent @ T).tocsr()
                            matrix = sparse.bmat([
                                [reduced_tangent, sparse.csr_matrix(-reactions)],
                                [sparse.csr_matrix(rows),
                                 sparse.csr_matrix((len(rows), len(rows)))],
                            ], format="csr")
                            rhs = np.r_[residual, constraint]
                            try:
                                delta = np.asarray(factorize(
                                    matrix, MatrixClass.GENERAL,
                                    signature=f"contact_path:{step}:{augmentation}:{iteration}",
                                ).solve(rhs)).reshape(-1)
                            except Exception:
                                failure_reason = "singular_contact_path_tangent"
                                break
                            if not np.all(np.isfinite(delta)):
                                failure_reason = "nonfinite_contact_path_increment"
                                break
                            proposed = np.asarray(T @ (q + delta[:len(q)]) + u0).reshape(-1)
                            fraction = manager.admissible_step_fraction(u, proposed)
                            if fraction <= 1.0e-12:
                                failure_reason = "contact_gap_exceeded"
                                break
                            q += fraction * delta[:len(q)]
                            multipliers += fraction * delta[len(q):]
                        if failure_reason is not None:
                            if pivot_backup is not None:
                                rejected_candidate = {}
                                if (failure_reason == "contact_candidate_evaluation_failed"
                                        and settings.q4_s3_shared_edge_tie_step_m is not None):
                                    # Retain the bounded failed development pose
                                    # before restoring the accepted algorithmic
                                    # origin. No candidate state is committed.
                                    candidate_source, candidate_target = manager._positions(u)
                                    rejected_candidate = {
                                        "candidate_source_positions_m": candidate_source.tolist(),
                                        "candidate_target_positions_m": candidate_target.tolist(),
                                        "candidate_normal_multipliers_pa":
                                        manager.normal_multipliers.tolist(),
                                    }
                                q, multipliers, fallback_pressure, _prior_change, _retained = pivot_backup
                                manager.set_normal_multipliers(fallback_pressure)
                                pivot_backup = None
                                schur_backtrack_source = None
                                schur_backtrack_signature = None
                                pivot_rejected += 1
                                augmentation_trace.append({
                                    ("schur_rejected" if settings.augmentation_acceleration
                                     == "supported_schur_dev" else "pivot_rejected"): failure_reason,
                                    "augmentation": augmentation,
                                    **({"candidate_evaluation_error":
                                        candidate_evaluation_error}
                                       if failure_reason == "contact_candidate_evaluation_failed"
                                       else {}),
                                    **rejected_candidate,
                                })
                                failure_reason = None
                                accelerator = (ProjectedAugmentationSecant()
                                               if settings.augmentation_acceleration == "active_secant_pivot"
                                               else None)
                                continue
                            break
                        if not equilibrated:
                            if pivot_backup is not None:
                                q, multipliers, fallback_pressure, _prior_change, _retained = pivot_backup
                                manager.set_normal_multipliers(fallback_pressure)
                                pivot_backup = None
                                schur_backtrack_source = None
                                schur_backtrack_signature = None
                                pivot_rejected += 1
                                augmentation_trace.append({
                                    ("schur_rejected" if settings.augmentation_acceleration
                                     == "supported_schur_dev" else "pivot_rejected"):
                                    "maximum_iterations_reached",
                                    "augmentation": augmentation,
                                })
                                accelerator = (ProjectedAugmentationSecant()
                                               if settings.augmentation_acceleration == "active_secant_pivot"
                                               else None)
                                continue
                            failure_reason = "maximum_iterations_reached"
                            break
                        if augmented:
                            if settings.augmentation_support_resolution == "roundoff":
                                try:
                                    projected = manager.projected_normal_multipliers(u)
                                except SupportResetForceBudgetExceeded as exc:
                                    failure_reason = "contact_multiplier_reset_force_budget_exceeded"
                                    augmentation_trace.append({
                                        "augmentation": augmentation,
                                        "support_reset_failure": str(exc),
                                    })
                                    break
                            else:
                                projected = manager.projected_normal_multipliers(u)
                            normal_multiplier_change = float(np.max(np.abs(
                                projected - manager.normal_multipliers
                            )))
                            verified_schur = False
                            if pivot_backup is not None:
                                (fallback_q, fallback_controls, fallback_pressure,
                                 prior_change, retained) = pivot_backup
                                pivot_backup = None
                                actual_active = tuple(np.flatnonzero(
                                    np.asarray(certificate.numerical_pressure_pa) > 0.0,
                                ))
                                active_changed = (
                                    actual_active != retained if isinstance(retained, tuple)
                                    else certificate.active_pressure_dofs > retained
                                )
                                signature_changed = False
                                signature_error = None
                                if schur_backtrack_signature is not None:
                                    try:
                                        source_pose, target_pose = manager._positions(u)
                                        signature_changed = (
                                            (manager.law.geometry.evaluate(
                                                source_pose, target_pose,
                                            ).overlap_signature,
                                             _projected_clip_signature(
                                                 manager.law.geometry,
                                                 source_pose, target_pose))
                                            != schur_backtrack_signature
                                        )
                                    except Exception as exc:
                                        signature_changed = True
                                        signature_error = (
                                            f"{type(exc).__name__}: {exc}")[:240]
                                if (not certificate.gap_tolerance_met
                                        or active_changed
                                        or signature_changed
                                        or normal_multiplier_change >= 0.75*prior_change):
                                    if (settings.augmentation_acceleration == "supported_schur_dev"
                                            and settings.augmentation_schur_maximum_relinearizations
                                            > schur_epoch_relinearizations
                                            and certificate.gap_tolerance_met
                                            and not active_changed
                                            and not signature_changed
                                            and normal_multiplier_change < prior_change
                                            and augmentation <= augmentation_limit-2):
                                        schur_relinearizations += 1
                                        schur_epoch_relinearizations += 1
                                        correction_trace = {
                                            "augmentation": augmentation,
                                            "schur_relinearization": schur_relinearizations,
                                            "schur_intermediate_pressure_change":
                                            normal_multiplier_change,
                                            "schur_prior_pressure_change": prior_change,
                                        }
                                        try:
                                            corrected = refined_patch_supported_schur(
                                                manager, u, tangent, T, rows,
                                                difference_step=settings.augmentation_schur_difference_step_m,
                                                maximum_pressure=settings.augmentation_schur_maximum_pressure_pa,
                                                maximum_pressure_step=settings.augmentation_schur_maximum_step_pa,
                                                maximum_modes=settings.augmentation_schur_maximum_modes,
                                                diagnostics=correction_trace,
                                                check_cancel=lambda: cancellation_safe_point(
                                                    cancellation_token,
                                                    f"contact_path.step:{step}.augmentation:{augmentation}.schur_relinearization",
                                                ),
                                            )
                                        except SolveCancelled:
                                            q = fallback_q
                                            multipliers = fallback_controls
                                            manager.set_normal_multipliers(
                                                fallback_pressure)
                                            raise
                                        except (ValueError, MemoryError, np.linalg.LinAlgError) as exc:
                                            correction_trace["schur_relinearization_ineligible"] = str(exc)
                                            corrected = None
                                        except Exception as exc:
                                            q = fallback_q
                                            multipliers = fallback_controls
                                            manager.set_normal_multipliers(
                                                fallback_pressure)
                                            correction_trace["schur_relinearization_failure"] = str(exc)
                                            augmentation_trace.append(correction_trace)
                                            failure_reason = "contact_supported_schur_relinearization_failed"
                                            break
                                        correction_trace["schur_relinearization_proposed"] = (
                                            corrected is not None)
                                        augmentation_trace.append(correction_trace)
                                        if corrected is not None:
                                            schur_backtrack_source = None
                                            schur_backtrack_signature = None
                                            pivot_backup = (
                                                fallback_q, fallback_controls,
                                                fallback_pressure, prior_change,
                                                corrected.active_indices,
                                            )
                                            manager.set_normal_multipliers(
                                                corrected.multipliers)
                                            continue
                                    q = fallback_q
                                    multipliers = fallback_controls
                                    manager.set_normal_multipliers(fallback_pressure)
                                    pivot_rejected += 1
                                    rejection_trace = {
                                        ("schur_rejected" if settings.augmentation_acceleration
                                         == "supported_schur_dev" else "pivot_rejected"):
                                        "nonlinear_verification",
                                        "augmentation": augmentation,
                                        "pressure_change": normal_multiplier_change,
                                        "minimum_gap": _reported_lower_gap(certificate),
                                        **({"schur_prior_pressure_change": prior_change,
                                            "schur_actual_to_prior_ratio": (
                                                normal_multiplier_change/prior_change
                                                if prior_change > 0.0 else None),
                                            "schur_predicted_active_indices": list(retained),
                                            "schur_actual_active_indices": [
                                                int(index) for index in actual_active],
                                            "schur_active_changed": active_changed,
                                            "schur_overlap_signature_changed": signature_changed,
                                            **({"schur_overlap_signature_error": signature_error}
                                               if signature_error is not None else {}),
                                            "schur_gap_tolerance_met": bool(
                                                certificate.gap_tolerance_met)}
                                           if settings.augmentation_acceleration
                                           == "supported_schur_dev" else {}),
                                    }
                                    augmentation_trace.append(rejection_trace)
                                    if (schur_backtrack_source is not None
                                            and certificate.gap_tolerance_met
                                            and not active_changed
                                            and not signature_changed
                                            and schur_backtrack_signature[1] is not None
                                            and schur_epoch_attempts
                                            < settings.augmentation_schur_maximum_attempts
                                            and augmentation <= augmentation_limit-2):
                                        cancellation_safe_point(
                                            cancellation_token,
                                            f"contact_path.step:{step}.augmentation:{augmentation}.schur_backtrack",
                                        )
                                        schur_epoch_attempts += 1
                                        pivot_attempts += 1
                                        try:
                                            backtrack = propose_supported_schur_secant_backtrack(
                                                schur_backtrack_source,
                                                getattr(certificate,
                                                        "normalized_weighted_gap_m",
                                                        None),
                                                diagnostics=rejection_trace,
                                            )
                                            if backtrack is not None:
                                                rejection_trace["schur_backtrack_model"] = (
                                                    "full_trial_gap_secant")
                                            else:
                                                backtrack = propose_supported_schur_backtrack(
                                                    schur_backtrack_source,
                                                    diagnostics=rejection_trace,
                                                )
                                                if backtrack is not None:
                                                    rejection_trace["schur_backtrack_model"] = (
                                                        "local_linear")
                                        except (ValueError, MemoryError,
                                                np.linalg.LinAlgError) as exc:
                                            rejection_trace["schur_backtrack_ineligible"] = str(exc)
                                            backtrack = None
                                        schur_backtrack_source = None
                                        rejection_trace["schur_backtrack_proposed"] = (
                                            backtrack is not None)
                                        if backtrack is not None:
                                            pivot_backup = (
                                                fallback_q.copy(),
                                                fallback_controls.copy(),
                                                fallback_pressure.copy(),
                                                prior_change,
                                                backtrack.active_indices,
                                            )
                                            rejection_trace["schur_backtrack_fraction"] = (
                                                backtrack.step_fraction)
                                            rejection_trace["schur_backtrack_predicted_residual_pa"] = (
                                                backtrack.predicted_residual_after)
                                            rejection_trace["schur_backtrack_predicted_active_indices"] = (
                                                list(backtrack.active_indices))
                                            manager.set_normal_multipliers(
                                                backtrack.multipliers)
                                            continue
                                    schur_backtrack_source = None
                                    schur_backtrack_signature = None
                                    accelerator = (ProjectedAugmentationSecant()
                                                   if settings.augmentation_acceleration == "active_secant_pivot"
                                                   else None)
                                    continue
                                pivot_applied += 1
                                schur_backtrack_source = None
                                schur_backtrack_signature = None
                                verified_schur = (settings.augmentation_acceleration
                                                  == "supported_schur_dev")
                                accelerator = (ProjectedAugmentationSecant()
                                               if settings.augmentation_acceleration == "active_secant_pivot"
                                               else None)
                            augmentation_trace.append({
                                **({"schur_verified": True} if verified_schur else {}),
                                "pressure_change": normal_multiplier_change,
                                "minimum_gap": _reported_lower_gap(certificate),
                                "residual_norm": residual_norm,
                                **({"macro_equilibrium_force_limit_n": equilibrium_force_limit}
                                   if getattr(manager_law, "pressure_basis", None) is not None else {}),
                                "active_pressure_dofs": certificate.active_pressure_dofs,
                                "source_contact_force_n": (
                                    None if not hasattr(certificate, "integration_force")
                                    else list(certificate.integration_force)
                                ),
                                "sampled_overlap_gap_m": getattr(
                                    certificate, "sampled_overlap_gap", None,
                                ),
                                "pressure_rank": getattr(
                                    certificate, "pressure_rank", None,
                                ),
                                "pressure_gram_rank": getattr(
                                    certificate, "pressure_gram_rank", None,
                                ),
                                "pressure_gradient_condition": getattr(
                                    certificate, "pressure_gradient_condition", None,
                                ),
                                "pressure_gram_condition_estimate": getattr(
                                    certificate, "pressure_gram_condition_estimate", None,
                                ),
                                "pressure_gram_supported_modes": getattr(
                                    certificate, "pressure_gram_supported_modes", None,
                                ),
                                "pressure_gram_scaled_rank": getattr(
                                    certificate, "pressure_gram_scaled_rank", None,
                                ),
                                "pressure_gram_scaled_condition_estimate": getattr(
                                    certificate, "pressure_gram_scaled_condition_estimate", None,
                                ),
                                "minimum_weighted_gap_m": getattr(
                                    certificate, "minimum_weighted_gap_m", None,
                                ),
                                "normalized_weighted_gap_m": list(
                                    getattr(certificate, "normalized_weighted_gap_m", ())
                                ),
                                "reference_support_m2": list(
                                    getattr(certificate, "reference_support_m2", ())
                                ),
                                "normal_multiplier_estimate_pa": list(
                                    getattr(certificate, "normal_multiplier_estimate_pa", ())
                                ),
                                "source_pressure_node_ids": list(
                                    getattr(certificate, "source_pressure_node_ids", ())
                                ),
                            })
                            if settings.augmentation_acceleration == "supported_schur_dev":
                                active_signature = tuple(np.flatnonzero(
                                    np.asarray(certificate.numerical_pressure_pa) > 0.0,
                                ))
                                if active_signature != schur_active_signature:
                                    if schur_active_signature is not None:
                                        augmentation_trace[-1]["schur_active_epoch_reset"] = {
                                            "prior_active_indices": [
                                                int(index) for index in schur_active_signature
                                            ],
                                            "current_active_indices": [
                                                int(index) for index in active_signature
                                            ],
                                            "prior_epoch_attempts": schur_epoch_attempts,
                                            "prior_epoch_relinearizations": (
                                                schur_epoch_relinearizations
                                            ),
                                        }
                                    schur_active_signature = active_signature
                                    schur_epoch_attempts = 0
                                    schur_epoch_relinearizations = 0
                            if (settings.augmentation_support_resolution == "roundoff"
                                    and manager.last_support_reset is not None):
                                reset = {"augmentation": augmentation,
                                         **manager.last_support_reset}
                                augmentation_trace[-1]["support_reset"] = reset
                                support_resets.append(reset)
                            if (hasattr(manager, "has_unresolved_weak_open_gap")
                                    and manager.has_unresolved_weak_open_gap(
                                        certificate, projected,
                                    )):
                                failure_reason = "contact_pressure_space_underresolved"
                                break
                            if (normal_multiplier_change <= settings.pressure_tolerance
                                    and certificate.gap_tolerance_met):
                                if (certificate.active_pressure_dofs == 0
                                        and not np.any(projected)):
                                    # A fully open state has no contact work at
                                    # either multiplier estimate. Store the
                                    # exact projected zero for restart and
                                    # report the certificate for that state.
                                    manager.set_normal_multipliers(projected)
                                    _force, _tangent, certificate = manager.evaluate(
                                        u, tangent=False,
                                    )
                                break
                            proposal = (None if accelerator is None else
                                        accelerator.propose(manager.normal_multipliers,
                                                            projected))
                            augmentation_trace[-1]["acceleration_applied"] = proposal is not None
                            if (proposal is None
                                    and settings.augmentation_acceleration == "supported_schur_dev"
                                    and schur_epoch_attempts < settings.augmentation_schur_maximum_attempts
                                    and augmentation >= 2
                                    and augmentation <= augmentation_limit - 2
                                    and certificate.active_pressure_dofs > 0):
                                pivot_attempts += 1
                                schur_epoch_attempts += 1
                                try:
                                    schur = refined_patch_supported_schur(
                                        manager, u, tangent, T, rows,
                                        difference_step=settings.augmentation_schur_difference_step_m,
                                        maximum_pressure=settings.augmentation_schur_maximum_pressure_pa,
                                        maximum_pressure_step=settings.augmentation_schur_maximum_step_pa,
                                        maximum_modes=settings.augmentation_schur_maximum_modes,
                                        retain_backtrack=settings.augmentation_schur_nonlinear_backtrack_dev,
                                        diagnostics=augmentation_trace[-1],
                                        check_cancel=lambda: cancellation_safe_point(
                                            cancellation_token,
                                            f"contact_path.step:{step}.augmentation:{augmentation}.schur",
                                        ),
                                    )
                                except SolveCancelled:
                                    raise
                                except (ValueError, MemoryError, np.linalg.LinAlgError) as exc:
                                    augmentation_trace[-1]["schur_ineligible"] = str(exc)
                                    if (settings.q4_s3_shared_edge_tie_step_m is not None
                                            and "pressure response derivative disagrees"
                                            in str(exc)):
                                        source_pose, target_pose = manager._positions(u)
                                        augmentation_trace[-1]["schur_ineligible_pose"] = {
                                            "source_positions_m": source_pose.tolist(),
                                            "target_positions_m": target_pose.tolist(),
                                            "normal_multipliers_pa":
                                            manager.normal_multipliers.tolist(),
                                        }
                                    schur = None
                                except Exception as exc:
                                    failure_reason = "contact_supported_schur_linearization_failed"
                                    augmentation_trace[-1]["schur_failure"] = str(exc)
                                    break
                                augmentation_trace[-1]["schur_proposed"] = schur is not None
                                if schur is not None:
                                    schur_backtrack_source = (
                                        schur if settings.augmentation_schur_nonlinear_backtrack_dev
                                        else None
                                    )
                                    if schur_backtrack_source is not None:
                                        try:
                                            source_pose, target_pose = manager._positions(u)
                                            schur_backtrack_signature = (
                                                manager.law.geometry.evaluate(
                                                    source_pose, target_pose,
                                                ).overlap_signature,
                                                _projected_clip_signature(
                                                    manager.law.geometry,
                                                    source_pose, target_pose,
                                                ),
                                            )
                                        except Exception as exc:
                                            schur_backtrack_source = None
                                            schur_backtrack_signature = None
                                            augmentation_trace[-1][
                                                "schur_backtrack_signature_unavailable"
                                            ] = f"{type(exc).__name__}: {exc}"[:240]
                                    pivot_backup = (q.copy(), multipliers.copy(),
                                                    projected.copy(),
                                                    normal_multiplier_change,
                                                    schur.active_indices)
                                    augmentation_trace[-1]["schur_active_indices"] = list(
                                        schur.active_indices)
                                    augmentation_trace[-1]["schur_scaled_condition"] = (
                                        schur.scaled_condition)
                                    augmentation_trace[-1]["schur_predicted_residual_pa"] = (
                                        schur.predicted_residual_after)
                                    augmentation_trace[-1]["schur_predicted_residual_before_pa"] = (
                                        schur.predicted_residual_before)
                                    augmentation_trace[-1]["schur_step_fraction"] = (
                                        schur.step_fraction)
                                    manager.set_normal_multipliers(schur.multipliers)
                                    continue
                            if (proposal is None
                                    and settings.augmentation_acceleration == "active_secant_pivot"
                                    and pivot_attempts == 0
                                    and augmentation >= 10
                                    and augmentation <= augmentation_limit - 5
                                    and 2 <= certificate.active_pressure_dofs <= 4):
                                pivot_attempts = 1
                                try:
                                    pivot = refined_patch_pivot(
                                        manager, u, tangent, T, rows,
                                        difference_step=settings.augmentation_pivot_difference_step_m,
                                        maximum_pressure=settings.augmentation_pivot_max_pressure_pa,
                                        pressure_tolerance=settings.pressure_tolerance,
                                        check_cancel=lambda: cancellation_safe_point(
                                            cancellation_token,
                                            f"contact_path.step:{step}.augmentation:{augmentation}.pivot",
                                        ),
                                    )
                                except SolveCancelled:
                                    raise
                                except ValueError as exc:
                                    failure_reason = "contact_active_set_pivot_resource_or_geometry_failure"
                                    augmentation_trace[-1]["pivot_failure"] = str(exc)
                                    break
                                except Exception as exc:
                                    failure_reason = "contact_active_set_pivot_linearization_failed"
                                    augmentation_trace[-1]["pivot_failure"] = str(exc)
                                    break
                                augmentation_trace[-1]["pivot_proposed"] = pivot is not None
                                if pivot is not None:
                                    pivot_backup = (q.copy(), multipliers.copy(),
                                                    projected.copy(),
                                                    normal_multiplier_change,
                                                    len(pivot.retained_local_indices))
                                    augmentation_trace[-1]["pivot_retained_local_indices"] = list(
                                        pivot.retained_local_indices)
                                    augmentation_trace[-1]["pivot_schur_condition"] = pivot.schur_condition
                                    manager.set_normal_multipliers(pivot.multipliers)
                                    continue
                            if (proposal is None
                                    and settings.augmentation_equilibrated_pressure_step_pa is not None
                                    and pressure_map_attempts == 0
                                    and augmentation >= 10
                                    and augmentation <= augmentation_limit - 2
                                    and np.count_nonzero(projected > 0.0) == 2):
                                pressure_map_attempts = 1
                                try:
                                    mapped = equilibrated_pressure_pivot(
                                        manager, model, u, q, multipliers,
                                        committed_states, num_layers, T, u0,
                                        rows, reduced_external, target,
                                        pressure_step=(
                                            settings.augmentation_equilibrated_pressure_step_pa
                                        ),
                                        maximum_pressure=settings.augmentation_pivot_max_pressure_pa,
                                        max_iterations=max_iterations,
                                        tolerance=tolerance,
                                        check_cancel=lambda: cancellation_safe_point(
                                            cancellation_token,
                                            f"contact_path.step:{step}.augmentation:{augmentation}.pressure_map",
                                        ),
                                    )
                                except SolveCancelled:
                                    raise
                                except (ValueError, np.linalg.LinAlgError) as exc:
                                    augmentation_trace[-1]["pressure_map_failure"] = str(exc)
                                    mapped = None
                                augmentation_trace[-1]["pressure_map_proposed"] = mapped is not None
                                if mapped is not None:
                                    pivot_backup = (q.copy(), multipliers.copy(),
                                                    projected.copy(),
                                                    normal_multiplier_change,
                                                    len(mapped.retained_local_indices))
                                    augmentation_trace[-1]["pressure_map_condition"] = mapped.schur_condition
                                    manager.set_normal_multipliers(mapped.multipliers)
                                    continue
                            manager.set_normal_multipliers(
                                projected if proposal is None else proposal
                            )
                        elif certificate.gap_tolerance_met:
                            break
                        else:
                            failure_reason = "contact_gap_exceeded"
                            break
                    else:
                        failure_reason = "maximum_augmentations_reached"
                    if failure_reason is not None:
                        info["contact_failure"] = {
                            "step_index": step,
                            "augmentation_iterations": augmentation,
                            "newton_iterations_last_augmentation": iteration,
                            "residual_norm": residual_norm,
                            "constraint_norm": constraint_norm,
                            "normal_multiplier_change": normal_multiplier_change,
                            "augmentation_trace": augmentation_trace,
                            "augmentation_acceleration_attempts": (
                                0 if accelerator is None else accelerator.attempts
                            ),
                            "augmentation_acceleration_applied": (
                                0 if accelerator is None else accelerator.applied
                            ),
                            "pivot_attempts": pivot_attempts,
                            "pivot_applied": pivot_applied,
                            "pivot_rejected": pivot_rejected,
                            **({"schur_attempts": pivot_attempts,
                                "schur_relinearizations": schur_relinearizations,
                                "schur_applied": pivot_applied,
                                "schur_rejected": pivot_rejected}
                               if settings.augmentation_acceleration == "supported_schur_dev"
                               else {}),
                            "pressure_map_attempts": pressure_map_attempts,
                            "minimum_gap": (
                                None if certificate is None else
                                _reported_lower_gap(certificate)
                            ),
                            "sampled_overlap_gap": (
                                None if certificate is None else
                                getattr(certificate, "sampled_overlap_gap", None)
                            ),
                            "gap_tolerance_met": (
                                None if certificate is None else
                                certificate.gap_tolerance_met
                            ),
                            "active_pressure_dofs": (
                                None if certificate is None else
                                certificate.active_pressure_dofs
                            ),
                        }
                        _discard_nonlinear_state_candidate(committed_states)
                        q = start_q
                        multipliers = start_multipliers
                        if augmented:
                            manager.set_normal_multipliers(start_normal_multipliers)
                        status = "stopped_at_limit" if previous_steps else "diverged"
                        break
                    accepted_u = np.asarray(T @ q + u0).reshape(-1)
                    admission = getattr(manager,
                                        "d1_interior_admission_issue", None)
                    issue = (admission(certificate, accepted_u, T, control_rows=rows)
                             if admission is not None else None)
                    if issue is not None:
                        failure_reason = "contact_d1_interior_profile_rejected"
                        info["contact_failure"] = {
                            "step_index": step, "issue": issue,
                        }
                        _discard_nonlinear_state_candidate(committed_states)
                        q = start_q
                        multipliers = start_multipliers
                        if augmented:
                            manager.set_normal_multipliers(start_normal_multipliers)
                        status = "stopped_at_limit" if previous_steps else "diverged"
                        break
                    friction_candidate = (manager.prepare_commit(accepted_u)
                                          if friction is not None else None)
                    committed_states = _commit_nonlinear_state_candidate(
                        committed_states, candidate_states,
                        model=model, accepted_displacements=accepted_u,
                    )
                    if friction is not None:
                        manager.commit(friction_candidate)
                    recovered, _K, _states = _assemble_nonlinear_system(
                        model, accepted_u, committed_states, num_layers,
                        tangent=False, kinematics=structural_kinematics,
                        **assembly_options,
                        require_full_coordinates=True,
                    )
                    _discard_nonlinear_state_candidate(committed_states)
                    support = _support_reaction_resultants(
                        model, recovered - controls.T @ multipliers - external,
                    )
                    accepted_step = step
                    record = {
                        "step_index": step,
                        "target": target.tolist(),
                        "reaction_generalized": multipliers.tolist(),
                        "residual_norm": residual_norm,
                        "constraint_error": constraint_norm,
                        "augmentation_iterations": augmentation,
                        "normal_multiplier_change": normal_multiplier_change,
                        "contact": certificate.to_dict(),
                    }
                    if friction is not None:
                        record["friction"] = manager.last_committed_summary
                    if settings.augmentation_support_resolution == "roundoff":
                        record["support_resets"] = support_resets
                    if accelerator is not None:
                        record["augmentation_acceleration"] = settings.augmentation_acceleration
                        record["augmentation_acceleration_attempts"] = accelerator.attempts
                        record["augmentation_acceleration_applied"] = accelerator.applied
                    if settings.augmentation_acceleration == "active_secant_pivot":
                        record["pivot_attempts"] = pivot_attempts
                        record["pivot_applied"] = pivot_applied
                        record["pivot_rejected"] = pivot_rejected
                        if settings.augmentation_equilibrated_pressure_step_pa is not None:
                            record["pressure_map_attempts"] = pressure_map_attempts
                    if settings.augmentation_acceleration == "supported_schur_dev":
                        record["augmentation_acceleration"] = "supported_schur_dev"
                        record["schur_attempts"] = pivot_attempts
                        record["schur_relinearizations"] = schur_relinearizations
                        record["schur_applied"] = pivot_applied
                        record["schur_rejected"] = pivot_rejected
                        record["augmentation_trace"] = augmentation_trace
                    if program.nodal_forces:
                        previous_force_values = program.force_values_at(step-1)
                        start_u = np.asarray(T @ start_q + u0).reshape(-1)
                        record["nodal_forces_n"] = force_values.tolist()
                        record["external_nodal_work_trapezoid_j"] = float(
                            0.5*(previous_force_values + force_values)
                            @ (accepted_u-start_u)[force_dofs]
                        )
                    history.append(record)
                    previous_steps.append(NonlinearStaticStep(
                        step_index=step,
                        load_factor=step / program.total_steps,
                        iterations=iteration,
                        residual_norm=residual_norm,
                        displacement_norm=float(np.linalg.norm(accepted_u)),
                        max_equivalent_plastic_strain=_max_plastic_strain(
                            committed_states
                        ),
                        active_stage=f"segment_{(step - 1) // program.steps_per_segment + 1}",
                        support_reactions=support,
                    ))
                    if record_increment_snapshots:
                        snapshots.append(_increment_snapshot(
                            step, step / program.total_steps,
                            accepted_u, committed_states,
                            control_value=(float(target[0])
                                           if target.size else None),
                        ))
                final_u = np.asarray(T @ q + u0).reshape(-1)
                if finite:
                    manager.require_finite_deformation(final_u)
                finished_states = _materialize_final_nonlinear_states(
                    committed_states, info,
                )
                if previous_steps:
                    finished_states = _finalize_nonlinear_element_states(
                        model, final_u, finished_states, num_layers,
                        kinematics=structural_kinematics,
                    )
                    if finite:
                        from .finite_rotation_state import seal_finite_states
                        if report.profile==finite_mixed_profile:
                            from .finite_mixed_state import seal_mixed_states as seal_finite_states
                        elif report.profile==finite_v2d_profile:
                            from .finite_v2d_state import seal_v2d_states as seal_finite_states
                        finished_states = seal_finite_states(
                            model, final_u, finished_states, num_layers, info)
                    else:
                        finished_states = _seal_final_qualified_q4_states(
                            model, final_u, finished_states, num_layers, info,
                            kinematics=structural_kinematics,
                        )
                info.update({
                    "failure_reason": failure_reason,
                    "status_category": status,
                    "stop_reason": failure_reason,
                    "total_newton_iterations": total_iterations,
                    "force_displacement_history": history,
                    "contact_path": {
                        **({"shell_skin_kinematics": settings.shell_skin_kinematics,
                            "structural_kinematics": "corotational",
                            "corotational_tangent": "consistent",
                            "finite_deformation_maxima": list(manager.last_finite_deformation_maxima)}
                           if finite else {}),
                        "accepted_step": accepted_step,
                        "total_steps": program.total_steps,
                        "targets": program.target_at(accepted_step).tolist(),
                        "reaction_generalized": multipliers.tolist(),
                        "profile": profile,
                        **({'finite_frame_seed':settings.finite_frame_seed,
                            'finite_surface_frames':list(finite_frames)}
                           if finite and settings.finite_frame_seed!='common_reference_frame_dev' else {}),
                        **({"shell_pair_orientation": settings.shell_pair_orientation}
                           if settings.shell_pair_orientation != "legacy" else {}),
                        **({"material_history_path": history_policy,
                            "accepted_physical_subincrements": accepted_step,
                            "user_increment_position": accepted_step / friction.physical_subincrements_per_user_increment,
                            "completed_user_increments": accepted_step // friction.physical_subincrements_per_user_increment,
                            "work_reporting": summarize_work_history(history)}
                           if history_friction else {}),
                        **({"finite_material_gap_error_scope": "whole_original_dof_chord" if whole_chord_friction else "accepted_endpoint_only"}
                           if weak_pressure_friction else {}),
                        **({"path_predictor": settings.path_predictor}
                           if settings.path_predictor != "none" else {}),
                        **({"enforcement_solver": settings.enforcement_solver}
                           if semismooth else {}),
                        **({"monolithic_linear_solver": settings.monolithic_linear_solver,
                            "monolithic_maximum_pivots": settings.monolithic_maximum_pivots}
                           if semismooth else {}),
                        **({"augmentation_acceleration": settings.augmentation_acceleration}
                           if settings.augmentation_acceleration != "none" else {}),
                        **({"nodal_forces_n": program.force_values_at(
                            accepted_step,
                        ).tolist()} if program.nodal_forces else {}),
                    },
                    "general_contact": (history[-1]["contact"] if semismooth and history
                                        else manager.evaluate(final_u, tangent=False)[2].to_dict()),
                })
                checkpoint = None
                if emit_restart_checkpoint and status == "completed":
                    checkpoint = create_nonlinear_checkpoint(
                        analysis_kind="static", model=model,
                        analysis_contract=contract, displacements=final_u,
                        element_states=finished_states,
                        path_state={
                            "mode": "contact_path",
                            "accepted_step": accepted_step,
                            "multipliers": multipliers.tolist(),
                            "reduced_coordinates": q.tolist(),
                            "steps": [item.to_dict() for item in previous_steps],
                            "history": history,
                            "total_iterations": total_iterations,
                            "terminal_status": status,
                            **({"normal_multipliers": manager.normal_multipliers.tolist()}
                               if augmented else {}),
                            **({"normal_forces": normal_forces.tolist()}
                               if semismooth else {}),
                            **({"friction_state": manager.checkpoint_state()}
                               if friction is not None else {}),
                        },
                    )
                return NonlinearStaticResult(
                    previous_steps, status, final_u,
                    accepted_step / program.total_steps,
                    finished_states, info, snapshots=tuple(snapshots),
                    restart_checkpoint=checkpoint,
                )

        return _run_with_nonlinear_state_cleanup(run)

    def leased_operation():
        return _run_with_qualified_assembly_runtime_lease(
            model, context="solve_static_contact_path", operation=operation,
        )
    if history_friction or rim_friction:
        if rim_friction:
            from .s3_friction_rim_transaction import guarded_invocation
        else:
            from .s3_friction_history_path import guarded_invocation
        def published_operation():
            result = leased_operation()
            cancellation_safe_point(cancellation_token, "contact_path.history_publication")
            return result
        return guarded_invocation(published_operation)
    return leased_operation()
