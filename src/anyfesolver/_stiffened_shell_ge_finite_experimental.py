"""Bounded finite-step trial driver for the experimental GE point panel.

All shell, beam and joint evaluations use one accepted predecessor per load
increment. Q4 shell J2 and generalized B3 yielding are experimental.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from ._stiffened_shell_distributing_pose import DistributingPoseDomainError
from copy import copy, deepcopy
from decimal import localcontext
import json
from math import fsum
import warnings
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.spatial.transform import Rotation

from ._ge_beam3_generalized_static_boundary import spatial_jacobian
from ._ge_beam3_pose_joint import _exp_terms
from . import _stiffened_shell_coordinate_support as coordinate_support
from ._stiffened_shell_point_recovery import PointTransferWrench, point_record
from ._stiffened_shell_nodal_footprint import require_rotation_triples
from ._ge_beam3_p5.algebra import skew
from ._ge_beam3_retained_generalized_state import Mechanical
from ._stiffened_shell_bl5_local_experimental import BL5TrialDomainError
from ._stiffened_shell_l1_local_experimental import _owned_slots
from ._ge_beam3_generalized_cell import GeneralizedCellHistory
from ._ge_beam3_generalized_ellipsoid_section import GeneralizedHistory
from ._ge_beam3_fibre_section import PhysicalFibreSection
from ._stiffened_shell_beam_history import (
    arrays as native_history_arrays, histories as native_histories,
    shapes as native_history_shapes, observables as native_observables,
    compensated_kind as native_compensated_kind,
)
from ._ge_beam3_p5.algebra import rotation
from ._ge_beam3_p5.compensated_coordinates import split_sum
from ._ge_beam3_variational_shell import response as variational_shell_response
from ._stiffened_shell_s3_algorithmic_pullback import (
    response as s3_algorithmic_response, local_coordinates as s3_local_coordinates,
)
from .control import cancellation_safe_point
from .corotational import corotational_element_response
from ._stiffened_shell_ge_experimental import (
    FINITE_CHECKPOINT, FINITE_MATERIAL_CHECKPOINT,
    FINITE_SHELL_MATERIAL_CHECKPOINT, FINITE_S3_MATERIAL_CHECKPOINT,
    NodalAction, _owned,
    _linear_equilibrium, _scaled_constraint_gap,
)
from ._stiffened_shell_ge_checkpoint_binary import encode as _encode_binary_checkpoint
from ._stiffened_shell_ge_checkpoint_binary import SCHEMAS as _BINARY_SCHEMAS
from ._stiffened_shell_ge_checkpoint_binary import CURVED_POINT_MATERIAL_KINDS, FINITE_DISTRIBUTING_MATERIAL_KINDS
from ._stiffened_shell_ge_checkpoint_binary import FINITE_DISTRIBUTING_NATIVE_KINDS
from ._stiffened_shell_ge_checkpoint_binary import encode_s3 as _encode_s3_checkpoint
from ._stiffened_shell_ge_checkpoint_binary import encode_chart as _encode_chart_checkpoint
from ._stiffened_shell_ge_checkpoint_binary import NATIVE_SHELL_SCHEMA
from ._stiffened_shell_ge_checkpoint_binary import STABLE_NATIVE_SCHEMA
from ._stiffened_shell_increment_control import (
    AdaptiveIncrementControl, IncrementAttempt, FiniteIncrementFailure,
    AdaptiveIncrementExhausted, _AdaptiveState, encode_control,
)
from ._stiffened_shell_rotation_charts import owned_anchors, physical_frames, total_coordinates
from ._stiffened_shell_q4_algorithmic_pullback import (
    POLICY as Q4_ALGORITHMIC_PULLBACK_POLICY, response as q4_algorithmic_response,
)


MAX_JSON_CHECKPOINT_BYTES = 2_000_000
MAX_JSON_CHECKPOINT_VALUES = 50_000
NATIVE_MATERIAL_CHECKPOINT = _BINARY_SCHEMAS["finite_native_material"]


def _trial_work_ratio(residual, applied, coordinate_scale):
    """State-dependent-load trial merit; never an equilibrium acceptance test.

    Coordinate scales and the parent's load define one work norm for the
    entire backtracking comparison. Small physical components remain in the
    norm and in the separately selected physical acceptance check.
    """
    residual, applied, coordinate_scale = (np.asarray(v, dtype=float)
        for v in (residual, applied, coordinate_scale))
    if (residual.ndim != 1 or residual.shape != applied.shape or
            residual.shape != coordinate_scale.shape or
            not np.isfinite(coordinate_scale).all() or np.any(coordinate_scale <= 0.)):
        raise ValueError("matching pressure trial rows and positive coordinate scales required")
    r = residual*coordinate_scale
    f = applied*coordinate_scale
    if not np.isfinite(r).all() or not np.isfinite(f).all():
        raise ValueError("finite pressure trial work required")
    scale = max(float(np.max(np.abs(r), initial=0.)),
                float(np.max(np.abs(f), initial=0.)))
    if scale == 0.:
        return 0.
    denominator = float(np.linalg.norm(f/scale))
    return float(np.linalg.norm(r/scale))/denominator if denominator else float("inf")


def _pressure_vector_equilibrium(residual, applied, free, shell_count, coordinate_scale):
    """Pressure-only work and active nodal force/moment vector assessment.

    The local vectors contain only free shell coordinates. Zero-applied groups
    are covered by the global work check, not divided by an artificial floor.
    Retained and beam rows participate in that global check only.
    """
    residual, applied, coordinate_scale = (np.asarray(value, dtype=float)
        for value in (residual, applied, coordinate_scale))
    free = np.asarray(free, dtype=int)
    if (residual.ndim != 1 or residual.shape != applied.shape or
            residual.shape != coordinate_scale.shape or shell_count < 0 or
            shell_count % 6 or shell_count > residual.size or free.ndim != 1 or
            np.any(free < 0) or np.any(free >= residual.size) or
            len(np.unique(free)) != len(free) or
            not np.isfinite(residual).all() or not np.isfinite(applied).all()):
        raise ValueError("valid pressure equilibrium rows and free coordinates required")
    global_ratio = _trial_work_ratio(residual[free], applied[free], coordinate_scale[free])
    is_free = np.zeros(residual.size, dtype=bool)
    is_free[free] = True
    local_ratio = 0.
    for start in range(0, shell_count, 3):
        slots = np.arange(start, start+3)
        slots = slots[is_free[slots]]
        if not slots.size or not np.any(applied[slots] != 0.):
            continue
        r, f = residual[slots], applied[slots]
        scale = max(float(np.max(np.abs(r))), float(np.max(np.abs(f))))
        local_ratio = max(local_ratio, float(np.linalg.norm(r/scale) /
                                             np.linalg.norm(f/scale)))
    return max(global_ratio, local_ratio)


def _require_elastic_footprint_histories(histories, context, *, scope="nodal rigid footprints"):
    for history in histories:
        for station in history.stations:
            if (station.accumulated != (0., 0.) or
                    any(value != 0. for pair in station.plastic for value in pair)):
                raise ValueError("experimental "+scope+" admit elastic beam response only: "+context)


def native_shell_checkpoint_shapes(owner, generation):
    """Layout from the compiled model; no dimensions are trusted from a file."""
    count = owner._K.shape[0]
    members = len(owner._beam_parts)
    q4_count = len(owner._definition.shell.q4_elements)
    return (("q", (count,)), ("frames", (len(owner._beam_ids), 3, 3)),
            ("cells", (members, 2, 3, 3)), ("resultants", (members, 18)),
            ("multipliers", (owner._C.shape[0],)),
            ("applied_dead_load" if getattr(owner, "_staged_context", None) is not None else "pattern", (count,)),
            ("accepted_factors", (generation,))) + (
        native_history_shapes(owner._beam_parts, "origin") +
        native_history_shapes(owner._beam_parts, "accepted")) + (
            ("shell_plastic", (q4_count, 12, 3)), ("shell_alpha", (q4_count, 12)),
            ("shell_origin_plastic", (q4_count, 12, 3)),
            ("shell_origin_alpha", (q4_count, 12)),
            ("shell_origin_q", (owner._shell_count,)),
            ("shell_origin_anchors", (len(owner._shell_ids), 3, 3)))


def _position_pair(reference, displacement):
    """Keep the rounded remainder when a physical displacement joins a reference."""
    if reference.shape != displacement.shape:
        raise ValueError("matching finite position coordinates required")
    high = np.empty_like(reference)
    low = np.empty_like(reference)
    for index in np.ndindex(reference.shape):
        high[index], low[index] = split_sum(
            (float(reference[index]), float(displacement[index])))
    return high, low


def _material_arrays(histories):
    stations = tuple(station for history in histories
                     for station in history.stations)
    return (np.asarray([station.plastic for station in stations], dtype=float),
            np.asarray([station.accumulated for station in stations], dtype=float))


def _material_histories(owner, plastic, accumulated):
    histories = []
    index = 0
    for element, _slots, _virgin in owner._beam_parts:
        cell = element.operator.cell
        stations = []
        for _ in cell.stations:
            stations.append(GeneralizedHistory(
                element.section.identity,
                tuple(tuple(map(float, pair)) for pair in plastic[index]),
                tuple(map(float, accumulated[index]))))
            index += 1
        history = GeneralizedCellHistory(cell.identity, tuple(stations))
        with localcontext() as context:
            context.prec = 80
            cell._origins(history)
        histories.append(history)
    return tuple(histories)


def _material_observables(histories):
    return native_observables(histories)


@dataclass(frozen=True)
class BeamFibreStationRecovery:
    """Native accepted station response from its last material predecessor.

    Fibre strains/stresses and plastic increments retain native high/low pairs.
    This record does not label material potential as stored energy/dissipation.
    """
    element_id: int
    cell: int
    station: int
    xi: float
    measure: float
    strain: np.ndarray
    strain_low: np.ndarray
    resultants: np.ndarray
    resultants_low: np.ndarray
    reference_frame: np.ndarray
    current_frame: np.ndarray
    fibres: tuple

@dataclass(frozen=True)
class BeamFibreEnergyRecovery:
    """Paired beam material energy (J) and dissipation of the last increment.

    Stored energy includes elastic background/fibres and recoverable hardening.
    These fields are not global total energy or cumulative dissipation.
    """
    element_id: int
    stored_energy_j: tuple[float, float]
    previous_hardening_energy_j: tuple[float, float]
    incremental_dissipation_j: tuple[float, float]
    incremental_material_potential_j: tuple[float, float]
    energy_recovery_station_updates: int


def _fibre_recovery(owner, state, predecessors, accepted_histories):
    records = []; energies = []
    for i, ((element, _, _), predecessor, accepted) in enumerate(zip(
            owner._beam_parts, predecessors, accepted_histories)):
        if type(element.section) is not PhysicalFibreSection:
            continue
        response = element.operator.recover_material(state.cells[i], state.resultants[i],
                                        origin=predecessor,
                                        check=element.operator.guard,
                                        **({'resultants_low': state.resultants_low[i], 'rotations_low': state.cells_low[i]}
                                           if state.resultants_low is not None else {}))
        rows = response.stations
        if response.history != accepted or tuple(row["history"] for row in rows) != accepted.stations:
            raise ValueError("native accepted fibre recovery history mismatch")
        energies.append(BeamFibreEnergyRecovery(element.element_id,
            response.stored_energy, response.previous_hardening_energy,
            response.incremental_dissipation, response.material_potential,
            response.energy_recovery_station_updates))
        for row in rows:
            records.append(BeamFibreStationRecovery(
                element.element_id, row["cell"], row["station"], row["xi"],
                row["measure"], *(_owned(row[name]) for name in (
                    "strain", "strain_low", "resultants", "resultants_low",
                    "reference_frame", "current_frame")), row["fibres"]))
    return tuple(records), tuple(energies)


def _shell_material_arrays(histories):
    if not histories:
        return _owned(np.empty((0, 12, 3))), _owned(np.empty((0, 12)))
    plastic = np.asarray([state["plastic_strain"] for state in histories],
                         dtype=float)
    alpha = np.asarray([state["alpha"] for state in histories], dtype=float)
    if plastic.shape != (len(histories), 12, 3) or alpha.shape != (len(histories), 12):
        raise ValueError("complete Q4 shell J2 station state required")
    return _owned(plastic), _owned(alpha)


def _shell_material_states(plastic, alpha):
    return tuple({"plastic_strain": _owned(plastic[i], (12, 3)),
                  "alpha": _owned(alpha[i], (12,))}
                 for i in range(len(plastic)))


@dataclass(frozen=True)
class ShellQ4MaterialRecovery:
    """Owned local Q4 plane-stress response, not a full shell stress tensor.

    Rows follow four native Q4 Gauss stations, each with three Lobatto layers
    from bottom to top. Strain columns are (xx, yy, engineering xy); stress
    columns are (xx, yy, tensor xy) in the element frame, in pascals.
    """

    element_id: int
    layer_strain: np.ndarray
    layer_stress: np.ndarray
    plastic_strain: np.ndarray
    alpha: np.ndarray


def _shell_q4_recovery(parts, states):
    if len(parts) != len(states):
        raise ValueError("complete Q4 shell material recovery required")
    return tuple(ShellQ4MaterialRecovery(
        int(part[0]),
        _owned(state["layer_strain"], (12, 3)),
        _owned(state["layer_stress"], (12, 3)),
        _owned(state["plastic_strain"], (12, 3)),
        _owned(state["alpha"], (12,)),
    ) for part, state in zip(parts, states))


@dataclass(frozen=True)
class ShellS3MaterialRecovery:
    """Owned local S3 plane-stress layers and generalized resultants."""

    element_id: int
    layer_strain: np.ndarray
    layer_stress: np.ndarray
    plastic_strain: np.ndarray
    alpha: np.ndarray
    station_generalized_resultant: np.ndarray


def _shell_s3_recovery(parts, states):
    if len(parts) != len(states):
        raise ValueError("complete S3 shell material recovery required")
    return tuple(ShellS3MaterialRecovery(
        int(part[0]),
        _owned(state["layer_strain"], (9, 3)),
        _owned(state["layer_stress"], (9, 3)),
        _owned(state["plastic_strain"], (9, 3)),
        _owned(state["alpha"], (9,)),
        _owned(state["station_generalized_resultant"], (3, 8)),
    ) for part, state in zip(parts, states))


def _require_s3_material_state(state, local_q, *, accepted):
    """Bind native material history to the objective local constitutive chart."""
    expected = np.asarray(local_q, dtype=float)
    constitutive = np.asarray(state["committed_constitutive_u"], dtype=float)
    kinematics = "LINEAR" if accepted else "ELEMENT_LOCAL"
    if (expected.shape != (18,) or constitutive.shape != (18,) or
            not np.array_equal(constitutive, expected) or
            state["solver_kinematics"] != kinematics):
        raise ValueError("S3 material state disagrees with mixed owner coordinates")


def _accepted_shell_material_origins(owner, trial, candidates):
    """Prepare all Q4/S3 histories before the one mixed-state publish point."""
    q4_count = len(owner._definition.shell.q4_elements)
    if len(candidates) != len(owner._shell_parts):
        raise ValueError("complete mixed shell material candidates required")
    q4_plastic, q4_alpha = _shell_material_arrays(candidates[:q4_count])
    origins = list(_shell_material_states(q4_plastic, q4_alpha))
    material = owner._shell_model.get_material("shell")
    for (eid, element, slots, _), candidate in zip(
            owner._shell_parts[q4_count:], candidates[q4_count:]):
        if len(slots) != 18:
            raise ValueError("native S3 material candidate required")
        anchors = None if trial.shell_anchors is None else trial.shell_anchors[slots[::6]//6]
        local_q = s3_local_coordinates(owner._shell_model, element, trial.q[slots],
                                      rotation_anchors=anchors)
        _require_s3_material_state(candidate, local_q, accepted=False)
        origins.append(element.seal_solver_integrated_nonlinear_state(
            owner._shell_model.mesh, material, candidate, 3,
            local_q, kinematics="LINEAR"))
    return tuple(origins)


def _restore_native_s3_histories(owner, q, rotation_anchors, value):
    """Restore coordinate-bound accepted/predecessor S3 states without publishing."""
    q4_count = len(owner._definition.shell.q4_elements)
    origin_q = _owned(value["shell_origin_q"], (owner._shell_count,))
    origin_anchors = owned_anchors(value["shell_origin_anchors"], len(owner._shell_ids))
    fixed = np.asarray(owner._fixed)
    if np.any(origin_q[fixed[fixed < owner._shell_count]] != 0.):
        raise ValueError("invalid shell predecessor support coordinates")
    fixed_set=set(owner._fixed)
    for i,anchor in enumerate(origin_anchors):
        if any(6*i+j in fixed_set for j in (3,4,5)) and np.linalg.norm(anchor-np.eye(3))>1e-12:
            raise ValueError("shell predecessor anchor violates rotational support")
    blobs = value["s3_states"]
    s3_count = len(owner._definition.shell.s3_elements)
    if type(blobs) is not tuple or len(blobs) != 2*s3_count:
        raise ValueError("complete accepted and predecessor S3 states required")
    states = []
    material = owner._shell_model.get_material("shell")
    for coordinates, anchors, family in ((q, rotation_anchors, blobs[:s3_count]),
                                         (origin_q, origin_anchors, blobs[s3_count:])):
        made = []
        for (_eid, element, slots, _), blob in zip(owner._shell_parts[q4_count:], family):
            local_anchors = None if anchors is None else anchors[slots[::6]//6]
            local_q = s3_local_coordinates(owner._shell_model, element, coordinates[slots],
                                          rotation_anchors=local_anchors)
            restored = element.deserialize_nonlinear_state(owner._shell_model.mesh,
                material, blob, 3, expected_committed_total_u=local_q)
            _require_s3_material_state(restored, local_q, accepted=True)
            made.append(restored)
        states.append(tuple(made))
    return states[0], states[1], origin_q, origin_anchors


def _require_native_s3_replay(owner, trial, accepted, candidates):
    """Require exact accepted S3 serialization after same-predecessor replay.

    A binary64 replay of an accepted update can round plastic-strain entries
    at the last ulp without changing the material path. When the serialized
    bytes differ, the replay is still accepted only if re-sealing the repeated
    state with the saved plastic array reproduces the saved bytes exactly
    (every other field and the integrity digest chain are then identical) and
    the plastic arrays themselves stay within the 64-eps array-scale roundoff
    bound. The saved accepted state stays authoritative either way.
    """
    if len(accepted) != len(owner._shell_parts):
        raise ValueError("complete accepted shell material states required")
    q4_count = len(owner._definition.shell.q4_elements)
    repeated = _accepted_shell_material_origins(owner, trial, candidates)
    material = owner._shell_model.get_material("shell")
    from .e4_pl_s3_v2d_state import seal_v2d_state
    for (_eid, element, _slots, _), saved_state, repeated_state in zip(
            owner._shell_parts[q4_count:], accepted[q4_count:], repeated[q4_count:]):
        saved_bytes = element.serialize_nonlinear_state(
            owner._shell_model.mesh, material, saved_state, 3)
        repeated_bytes = element.serialize_nonlinear_state(
            owner._shell_model.mesh, material, repeated_state, 3)
        if saved_bytes == repeated_bytes:
            continue
        saved_plastic = np.asarray(saved_state["plastic_strain"], dtype=float)
        repeated_plastic = np.asarray(repeated_state["plastic_strain"], dtype=float)
        resaved = dict(repeated_state)
        resaved["plastic_strain"] = saved_plastic.copy()
        resaved_bytes = element.serialize_nonlinear_state(
            owner._shell_model.mesh, material, seal_v2d_state(resaved), 3)
        scale = max(float(np.max(np.abs(saved_plastic), initial=0.)),
                    float(np.max(np.abs(repeated_plastic), initial=0.)),
                    np.finfo(float).tiny)
        if (resaved_bytes != saved_bytes or
                np.max(np.abs(saved_plastic - repeated_plastic), initial=0.) >
                64.*np.finfo(float).eps*scale):
            raise ValueError("native S3 checkpoint response disagrees with accepted update")


@dataclass(frozen=True)
class FinitePointResult:
    """Experimental finite result with dimensionless work/row-scaled errors."""
    status: str
    load_factor: float
    shell_displacement: np.ndarray
    beam_positions: np.ndarray
    beam_frames: np.ndarray
    cell_frames: np.ndarray
    cell_resultants: np.ndarray
    multipliers: np.ndarray
    support_reactions: np.ndarray
    free_equilibrium_ratio: float
    constraint_max: float
    iterations: int
    generation: int
    accepted_factors: tuple[float, ...]
    beam_station_accumulated: tuple[tuple[float, ...], ...] = ()
    shell_layer_alpha: tuple[tuple[float, ...], ...] = ()
    shell_q4_material: tuple[ShellQ4MaterialRecovery, ...] = ()
    shell_s3_material: tuple[ShellS3MaterialRecovery, ...] = ()
    production_qualified: bool = False
    shell_frames: np.ndarray | None = None
    shell_rotation_anchors: np.ndarray | None = None
    shell_chart_coordinates: np.ndarray | None = None
    beam_fibre_material: tuple[BeamFibreStationRecovery, ...] = ()
    beam_fibre_energy: tuple[BeamFibreEnergyRecovery, ...] = ()
    beam_generalized_material: tuple = ()
    increment_attempts: tuple[IncrementAttempt, ...] = ()
    beam_shell_line_work: tuple = ()
    point_wrenches: tuple[PointTransferWrench, ...] = ()
    load_programme_identity: str | None = None
    load_position: float | None = None
    load_stage_cursor: int | None = None
    load_stage_fraction: float | None = None
    load_programme_complete: bool | None = None
    applied_dead_load: np.ndarray | None = None
    load_equilibrium_reference_endpoint: int | None = None
    load_equilibrium_reference_policy: str | None = None
    load_equilibrium_reference_used: bool = False
    body_load: object | None = None
    distributed_load: object | None = None
    pressure_load: object | None = None
    distributing_footprint_wrenches: tuple = ()


@dataclass(frozen=True)
class FiniteLineStationWork:
    """Accepted physical station action, before any display smoothing.

    The integrated gradient contains two shell endpoint wrenches, two beam
    endpoint forces and the native retained-cell couple. Weak-row multipliers
    are not display moments. Station gap is distinct from integrated row gap.
    """
    physical_slots: np.ndarray
    constraint_rows: np.ndarray
    reference_measure_m: float
    gap: np.ndarray
    multiplier_density: np.ndarray
    integrated_spatial_gradient: np.ndarray


@dataclass(frozen=True)
class _State:
    q: np.ndarray
    frames: np.ndarray
    cells: np.ndarray
    resultants: np.ndarray
    multipliers: np.ndarray
    shell_anchors: np.ndarray | None = None
    q_low: np.ndarray | None = None
    frames_low: np.ndarray | None = None
    cells_low: np.ndarray | None = None
    resultants_low: np.ndarray | None = None


def _with_shell_chart(result, state, shell_count):
    if state.shell_anchors is None:
        return result
    local = state.q[:shell_count]
    frames = physical_frames(local, state.shell_anchors)
    display = local.copy().reshape(-1, 6)
    display[:, 3:] = Rotation.from_matrix(frames).as_rotvec()
    return replace(result, shell_displacement=_owned(display.ravel()),
                   shell_frames=_owned(frames),
                   shell_rotation_anchors=state.shell_anchors,
                   shell_chart_coordinates=_owned(local))


class _RotationStepTooLarge(ValueError):
    """Candidate update lies outside the admitted local rotation chart."""


def _nodal_translation_pattern(owner, actions):
    """Compile the existing nodal force contract without creating trial state."""
    load = np.zeros(owner._K.shape[0])
    for action in actions:
        if type(action) is not NodalAction or action.domain not in ("shell", "beam"):
            raise ValueError("declared shell or beam nodal action required")
        index = owner._shell_index if action.domain == "shell" else owner._beam_index
        if action.node not in index:
            raise ValueError("action references unknown physical node")
        wrench = _owned(action.wrench, (6,))
        if np.any(wrench[3:] != 0.):
            raise ValueError("finite experimental path admits translation forces only")
        offset = 0 if action.domain == "shell" else owner._shell_count
        load[offset+6*index[action.node]:offset+6*index[action.node]+6] += wrench
    return _owned(load)


def supports_paired_fibre_adaptive(owner):
    """The adopted elastic flat-Q4 physical-fibre point controller domain."""
    from ._ge_beam3_retained_fibre import RetainedFibreOperator
    from ._stiffened_shell_f06_increment_arithmetic import POLICY
    from ._stiffened_shell_affine_endpoint import is_native_point_material
    d = owner._definition
    return bool(owner._beam_parts and is_native_point_material(owner) and d.points
        and d.shell.q4_elements and not d.shell.s3_elements and d.shell.j2_curve is None
        and d.shell_reference_policy == 'flat' and not d.lines and not d.footprints
        and not owner._finite_distributing and not owner._curved_reference
        and not owner._coordinate_support and getattr(owner, '_external_context', None) is None
        and all(type(e.operator) is RetainedFibreOperator and e.operator.arithmetic_policy == POLICY
                for e, _, _ in owner._beam_parts))


def supports_mixed_material_adaptive(owner):
    """Experimental nodal-dead mixed-native/Q4-J2 POINT controller domain.

    This does not extend the homogeneous fibre predictor or its count policy.
    """
    from ._ge_beam3_retained_fibre import RetainedFibreOperator
    from ._ge_beam3_retained_generalized import RetainedGeneralizedOperator
    from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
    from ._stiffened_shell_f06_increment_arithmetic import POLICY
    from ._stiffened_shell_affine_endpoint import is_native_point_material
    d = getattr(owner, "_definition", None)
    if d is None:
        return False
    # Domain predicate only: foreign or incomplete owners are simply
    # outside the mixed-material adaptive domain (fail-closed False);
    # errors during actual evaluation still propagate unchanged.
    parts = getattr(owner, "_beam_parts", None)
    return bool(parts and is_native_point_material(owner) and d.points
        and d.shell.q4_elements and not d.shell.s3_elements and d.shell.j2_curve is not None
        and d.shell_reference_policy == 'flat' and not d.lines and not d.footprints
        and not owner._finite_distributing and not owner._curved_reference
        and not owner._coordinate_support and getattr(owner, '_external_context', None) is None
        and getattr(owner, '_staged_context', None) is None
        and all(getattr(e.operator, 'arithmetic_policy', None) == POLICY and
            ((type(e.operator) is RetainedFibreOperator and type(e.section) is PhysicalFibreSection) or
             (type(e.operator) is RetainedGeneralizedOperator and type(e.section) is EllipsoidalGeneralizedSection))
            for e, _, _ in parts)
        and any(type(e.section) is PhysicalFibreSection for e, _, _ in parts)
        and any(type(e.section) is EllipsoidalGeneralizedSection for e, _, _ in parts))


def supports_generalized_stable_s3(owner):
    """Narrow native generalized POINT domain for the elastic stable S3 policy.

    This reuses the existing cancellation-resistant elastic S3
    deformation for the explicit experimental generalized point owner only.
    It is a private numerical-policy predicate: no public selector, element
    law, or checkpoint format changes with it.
    """
    from ._ge_beam3_retained_generalized import RetainedGeneralizedOperator
    from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
    from ._stiffened_shell_affine_endpoint import is_native_point_material
    d = getattr(owner, "_definition", None)
    if d is None:
        return False
    # Domain predicate only: foreign or incomplete owners stay outside the
    # stable S3 policy (fail-closed False); evaluation errors propagate.
    parts = getattr(owner, "_beam_parts", None)
    return bool(parts and is_native_point_material(owner) and d.points
        and d.shell.s3_elements and not d.shell.q4_elements
        and d.shell.j2_curve is None
        and d.shell_reference_policy == 'flat' and not d.lines and not d.footprints
        and not owner._finite_distributing and not owner._curved_reference
        and not owner._coordinate_support and not owner._has_physical_fibre
        and getattr(owner, '_external_context', None) is None
        and getattr(owner, '_staged_context', None) is None
        and all(type(e.operator) is RetainedGeneralizedOperator and
            type(e.section) is EllipsoidalGeneralizedSection
            for e, _, _ in parts))


def compensated_adaptive_load_admitted(owner, actions):
    """Keep the two adopted load contexts distinct at every public dispatch."""
    from ._stiffened_shell_staged_load_experimental import StagedDeadLoadProgramme
    return ((supports_paired_fibre_adaptive(owner) and type(actions) is StagedDeadLoadProgramme)
        or (supports_mixed_material_adaptive(owner) and type(actions) is tuple
            and all(type(a) is NodalAction and a.domain in ('shell', 'beam')
                    and len(a.wrench) == 6 and not any(a.wrench[3:]) for a in actions)))


ADMISSIBLE_POLISH_CANDIDATE_IDENTITY_SUFFIX = ":native-chart-admissible-polish-candidate-v1"


def admissible_polish_identity(identity):
    """Distinct checkpoint identity of the selected polishing candidate.

    The candidate is not an accepted default: its checkpoints carry this
    explicit policy tag so legacy bytes and selected analyses can never
    silently mix. The identity is unchanged for every unselected analysis.
    """
    if type(identity) is not str or not identity:
        raise ValueError("valid model identity required")
    return identity + ADMISSIBLE_POLISH_CANDIDATE_IDENTITY_SUFFIX


def supports_native_chart_polish(owner):
    """Private domain of the admissible-state polishing candidate.

    This is the native physical-fibre POINT chart route of the chart-energy
    parity fixture: flat algorithmic-pullback elastic Q4 shell, physical-fibre
    beam sections, point connections, and no line/footprint/distributing/
    curved/coordinate-support/external contexts. It is a private numerical
    policy predicate: no public selector, element law, or checkpoint format
    changes with it, and foreign or incomplete owners stay outside (False).
    """
    from ._stiffened_shell_affine_endpoint import is_native_point_material
    d = getattr(owner, "_definition", None)
    if d is None:
        return False
    return bool(is_native_point_material(owner)
        and getattr(owner, "_has_physical_fibre", False)
        and d.shell.q4_elements and not d.shell.s3_elements
        and d.shell.j2_curve is None
        and not d.lines and not d.footprints
        and not owner._finite_distributing and not owner._curved_reference
        and not owner._coordinate_support
        and getattr(owner, "_external_context", None) is None)


def select_admissible_polish_candidate(analysis, enabled):
    """Explicit private opt-in for the admissible-state polishing candidate.

    Default False; no existing driver, checkpoint or public route opts in.
    Once an increment first meets the UNCHANGED physical limits (equilibrium
    work ratio <= 1e-8, constraint gap <= 1e-9), extra Newton work inside the
    same iteration/backtrack budgets may retain only strictly lower-merit
    states that still meet both limits; the last retained state is published.
    The selected policy binds a distinct checkpoint identity: its bytes
    restore only with the policy selected, and legacy bytes restore only
    without it. Selection refuses outside the native physical-fibre point
    chart route and after any accepted state exists.
    """
    if type(enabled) is not bool:
        raise ValueError("strict boolean admissible polish selection required")
    if (getattr(analysis, "_finite_driver", None) is not None or
            getattr(analysis, "_accepted", None) is not None):
        raise ValueError("admissible polish selection requires a fresh analysis")
    if enabled and not supports_native_chart_polish(analysis):
        raise ValueError("admissible polish candidate requires the native physical-fibre point chart route")
    analysis._admissible_polish_candidate = enabled
    # Explicit provenance: distinguishes a strict private False from an owner
    # whose selector was never set (auto-eligible). No public API reads it.
    analysis._admissible_polish_selection = enabled


def _explicit_admissible_polish(owner):
    """Explicit private polishing selection: True/False, or None when absent.

    Only `select_admissible_polish_candidate` sets the provenance marker, so
    None means a fresh owner that never made an explicit choice; legacy
    owners restored before the policy existed also decode as None.
    """
    value = getattr(owner, "_admissible_polish_selection", None)
    return value if type(value) is bool else None


def _probe_checkpoint_model(raw):
    """Bounded model probe of the outermost recognised checkpoint envelope.

    Reads only the size-bounded JSON header of a known envelope framing
    (adaptive controller, shell chart, or binary finite checkpoint) and
    returns its declared model identity. Every digest, canonicality, schema
    and shape check stays with the normal validated decoders that run after
    this probe; unknown or malformed framings return None and never raise.
    A forged header can only select which identity the subsequent full
    validation expects, never bypass it.
    """
    from ._stiffened_shell_increment_control import (
        MAGIC as CONTROL_MAGIC, MAX_METADATA_BYTES,
    )
    from ._stiffened_shell_ge_checkpoint_binary import (
        MAGIC as BINARY_MAGIC, CHART_MAGIC, MAX_HEADER_BYTES,
    )
    if type(raw) is not bytes:
        return None
    for magic, bound in ((CONTROL_MAGIC, MAX_METADATA_BYTES),
                         (CHART_MAGIC, MAX_HEADER_BYTES),
                         (BINARY_MAGIC, MAX_HEADER_BYTES)):
        if not raw.startswith(magic):
            continue
        start = len(magic) + 4
        if len(raw) < start + 33:
            return None
        size = int.from_bytes(raw[len(magic):start], "little")
        if not 1 <= size <= bound or start + size > len(raw) - 32:
            return None
        try:
            header = json.loads(raw[start:start+size].decode("ascii"))
        except (UnicodeError, ValueError, RecursionError):
            return None
        model = header.get("model") if type(header) is dict else None
        return model if type(model) is str else None
    return None


class FinitePointDriver:
    """Finite updates with globally retained native B3 coordinates."""

    def __init__(self, owner):
        self.owner = owner
        # Private polishing candidate: bound only by the explicit selector on
        # the analysis; every unselected owner keeps its exact default route.
        self._admissible_polish = (getattr(owner, "_admissible_polish_candidate",
                                           False) is True)
        if self._admissible_polish and not supports_native_chart_polish(owner):
            raise ValueError("admissible polish candidate requires the native physical-fibre point chart route")
        from ._stiffened_shell_f06_increment_arithmetic import POLICY as compensated_policy
        selected=[getattr(element.operator,'arithmetic_policy',None)==compensated_policy for element,_,_ in owner._beam_parts]
        self._compensated=any(selected)
        if self._compensated and (not all(selected) or owner._definition.lines or owner._definition.footprints
                or owner._finite_distributing or owner._curved_reference
                or getattr(owner,'_external_context',None) is not None):
            raise ValueError('durable compensated native state currently requires native point owner')
        self._compensated_coupled = self._compensated and owner._definition.shell.j2_curve is not None
        if self._compensated:
            from ._stiffened_shell_affine_endpoint import is_native_point_material
            if (not is_native_point_material(owner) or (self._compensated_coupled and
                    (owner._definition.shell.s3_elements or not owner._definition.shell.q4_elements))):
                raise ValueError('compensated native material requires admitted elastic or Q4 algorithmic J2 point policy')
            native_compensated_kind(owner._beam_parts, coupled=self._compensated_coupled)
        from ._ge_beam3_retained_fibre import RetainedFibreOperator
        self._paired_fibre_event = bool(self._compensated and not self._compensated_coupled
            and owner._definition.points and owner._definition.shell.q4_elements
            and not owner._definition.shell.s3_elements
            and owner._definition.shell_reference_policy == 'flat'
            and all(type(element.operator) is RetainedFibreOperator for element,_,_ in owner._beam_parts))
        self._stable_q4 = bool(owner._definition.points and owner._definition.shell.q4_elements and
            not owner._definition.shell.s3_elements and
            owner._definition.shell_reference_policy == "flat" and
            owner._definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_POLICY and
            owner._definition.shell.j2_curve is None and
            not owner._has_physical_fibre and not owner._definition.lines and
            not owner._definition.footprints)
        # Numerical policy is checkpoint-bound. Legacy restored histories keep
        # their original arithmetic rather than changing endpoint validation.
        # The plain elastic S3 generalized point owner reuses the same
        # cancellation-resistant deformation; its old checkpoints still
        # restore with the original arithmetic (see restore).
        self._stable_s3 = bool((owner._coordinate_support or owner._has_physical_fibre or owner._footprint_stable_s3
                                 or owner._curved_s3 or supports_generalized_stable_s3(owner)) and
            owner._definition.shell.s3_elements and
            owner._definition.shell.j2_curve is None)
        beam_ids = owner._beam_ids
        self._beam_nodal = 6*len(beam_ids)
        self._reference_frames = np.array(
            [owner._beam_nodes[node][1] for node in beam_ids], dtype=float)
        self._reference_positions = np.array(
            [owner._beam_nodes[node][0] for node in beam_ids], dtype=float)
        self._state = _State(
            np.zeros(owner._K.shape[0]), self._reference_frames.copy(),
            np.tile(np.eye(3), (len(owner._beam_parts), 2, 1, 1)),
            np.zeros((len(owner._beam_parts), 18)),
            np.zeros(owner._C.shape[0]))
        if self._compensated:self._state=self._zero_compensation(self._state)
        self._shell_origins = tuple(part[3] for part in owner._shell_parts)
        # Immutable preparation is owner-local, not accepted/checkpoint state.
        # Unsupported or material Q4 inputs keep the complete evaluation route.
        self._q4_corrections = {}
        if owner._definition.q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_POLICY:
            for eid, element, slots, _ in owner._shell_parts:
                if len(slots) == 24:
                    material = owner._shell_model.get_material(element.material_name)
                    self._q4_corrections[eid] = element._prepare_nonlinear_linear_correction(
                        owner._shell_model.mesh, material)
        if owner._has_physical_fibre and owner._definition.shell.j2_curve is not None:
            material = owner._shell_model.get_material("shell")
            self._shell_origins = tuple(
                element.seal_solver_integrated_nonlinear_state(
                    owner._shell_model.mesh, material, origin, 3,
                    np.zeros(18), kinematics="LINEAR") if len(slots) == 18 else origin
                for (_eid, element, slots, _), origin in zip(owner._shell_parts, self._shell_origins))
        self._shell_predecessors = self._shell_origins
        self._shell_predecessor_q = _owned(self._state.q[:owner._shell_count])
        self._shell_predecessor_anchors = owned_anchors(
            np.tile(np.eye(3), (len(owner._shell_ids), 1, 1)), len(owner._shell_ids))
        self._beam_origins = tuple(part[2] for part in owner._beam_parts)
        self._beam_predecessors = self._beam_origins
        self._curved_material_predecessors = bool(owner._curved_material)
        self._pattern = None
        self._external = getattr(owner, "_external_context", None)
        if owner._coordinate_support and (self._external is not None or getattr(owner,'_staged_context',None) is not None):
            raise ValueError('coordinate shell supports exclude external and staged contexts')
        if self._external is not None and not getattr(self._external, "finite_supported", True):
            raise ValueError(self._external.finite_error)
        self._external_stable_s3 = bool(self._external is not None and
            owner._definition.shell.s3_elements and owner._definition.shell.j2_curve is None)
        self._stable_s3 = self._stable_s3 or self._external_stable_s3
        if (self._external is not None and owner._definition.shell.q4_elements and
                owner._definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_POLICY):
            raise ValueError("finite configuration-dependent loading requires the objective Q4 algorithmic pullback")
        self._programme = getattr(owner, "_staged_context", None)
        self._applied_load = None
        self._load_factor = 0.
        self._accepted = None
        self._accepted_factors = ()
        self._adaptive_state = None
        self._adaptive_checkpoint = None
        self._line_trial_recovery = None
        self._point_trial_recovery = None
        self._distributing_trial_recovery = None
        self._generalized_trial_recovery = None
        self._native_boundary_trial = None
        self._q4_boundary_trial = None

    def _load(self, actions):
        return _nodal_translation_pattern(self.owner, actions)

    def _checkpoint_identity(self):
        if self._external is not None:
            identity = self._external.augmented_model_identity
        else:
            identity = (self.owner._identity if self._programme is None else
                        self._programme.augmented_model_identity)
        # The selected polishing candidate binds a distinct checkpoint
        # identity at every encoded layer; unselected bytes are unchanged.
        # Restored legacy drivers built before this policy existed simply
        # keep their untagged identity (missing selector means False).
        return (admissible_polish_identity(identity)
                if getattr(self, "_admissible_polish", False) else identity)

    def _equilibrium_ratio(self, residual, applied):
        owner = self.owner
        if getattr(self._external, "pressure_vector_equilibrium", False):
            return _pressure_vector_equilibrium(
                residual[:len(applied)], applied, owner._free, owner._shell_count,
                owner._coordinate_scale)
        return _linear_equilibrium(
            residual[owner._free], applied[owner._free],
            owner._coordinate_scale[owner._free])

    def _frozen_trial_ratio(self, residual, parent_applied, default_ratio):
        """Compare a candidate against the parent's unchanged pressure action."""
        owner = self.owner
        if getattr(self._external, "pressure_vector_equilibrium", False):
            return _pressure_vector_equilibrium(
                residual[:len(parent_applied)], parent_applied, owner._free,
                owner._shell_count, owner._coordinate_scale)
        return default_ratio(residual[owner._free], parent_applied[owner._free],
                             owner._coordinate_scale[owner._free])

    def _prepare_load(self, actions, targets):
        if self.owner._finite_distributing:
            from ._stiffened_shell_ge_experimental import NodalAction
            if type(actions) is not tuple or any(type(a) is not NodalAction for a in actions):
                raise ValueError("finite distributing footprints admit nodal translation forces only")
        from ._stiffened_shell_staged_load_experimental import StagedDeadLoadProgramme
        from ._stiffened_shell_ge_body_load import GEBodyLoad
        from ._stiffened_shell_ge_distributed_load import GEBeamDistributedLoad
        from ._stiffened_shell_ge_distributed_programme import GEBeamDistributedProgramme
        from ._stiffened_shell_ge_pressure_load import GEShellPressureLoad
        if type(actions) in (GEBodyLoad, GEBeamDistributedLoad, GEShellPressureLoad, GEBeamDistributedProgramme):
            if (self.owner._definition.shell.q4_elements and
                    self.owner._definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_POLICY):
                raise ValueError("finite configuration-dependent loading requires the objective Q4 algorithmic pullback")
            if (self._programme is not None or
                    (self._external is not None and actions.identity != self._external.request.identity) or
                    (self._accepted is not None and self._external is None)):
                raise ValueError("finite configuration-dependent load changed after first increment")
            if self._external is None:
                compiled = actions.compile(self.owner)
                if type(actions) is GEBeamDistributedProgramme:
                    compiled.validate_targets(self._load_factor, targets)
                if not getattr(compiled, "finite_supported", True):
                    raise ValueError(compiled.finite_error)
                self._external = compiled
                self._external_stable_s3 = bool(self.owner._definition.shell.s3_elements and
                    self.owner._definition.shell.j2_curve is None)
                self._stable_s3 = self._stable_s3 or self._external_stable_s3
            elif type(actions) is GEBeamDistributedProgramme:
                self._external.validate_targets(self._load_factor, targets)
            return _owned(np.zeros(self.owner._K.shape[0])), None
        if self._external is not None:
            raise ValueError("finite configuration-dependent and nodal/programme loads cannot be mixed")
        if type(actions) is StagedDeadLoadProgramme:
            if self.owner._line_samples and not (self.owner._finite_bl5 and
                    self.owner._definition.q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_POLICY):
                raise ValueError("staged translation dead loads require a point or admitted elastic BL5 owner")
            if self._accepted is not None and (self._programme is None or
                    actions.identity != self._programme.identity):
                raise ValueError("finite load programme changed after first increment")
            programme = (self._programme if self._programme is not None else
                         actions.compile(self.owner, self._load))
            programme.validate_targets(self._load_factor, targets)
            return None, programme
        if self._programme is not None:
            raise ValueError("finite programme and legacy pattern modes cannot be mixed")
        pattern = self._load(actions)
        if self.owner._distributing_cyclic and not np.array_equal(pattern, self.owner._cyclic_reference_pattern):
            raise ValueError("cyclic force pattern must match its model-bound reference actions")
        if self._pattern is not None and not np.array_equal(self._pattern, pattern):
            raise ValueError("finite load pattern changed after first increment")
        return pattern, None

    @staticmethod
    def _programme_result(result, programme, position, applied):
        if programme is None:
            return result
        cursor, fraction = programme.progress_at(position)
        return replace(result, load_programme_identity=programme.identity,
            load_position=float(position), load_stage_cursor=cursor,
            load_stage_fraction=fraction,
            load_programme_complete=cursor == len(programme.programme.endpoints),
            applied_dead_load=_owned(applied),
            load_equilibrium_reference_endpoint=programme.programme.zero_load_reference_endpoint,
            load_equilibrium_reference_policy=(programme.programme.policy if
                programme.programme.zero_load_reference_endpoint is not None else None),
            load_equilibrium_reference_used=(programme.programme.zero_load_reference_endpoint is not None
                and not np.any(applied != 0.)))

    def _assemble(self, state, factor, shell_origins, beam_origins, *, pattern=None,
                  applied_load=None, load_programme=None):
        owner = self.owner
        from ._stiffened_shell_affine_endpoint import is_native_point_material
        if owner._coordinate_support and (self._external is not None or self._programme is not None or applied_load is not None or load_programme is not None):
            raise ValueError('coordinate shell supports exclude alternate load contexts')
        native_point_material=is_native_point_material(owner)
        programme = self._programme if load_programme is None else load_programme
        load_pattern = self._pattern if pattern is None else pattern
        if applied_load is None and programme is not None:
            applied_load = programme.load_at(factor)
        if applied_load is None and load_pattern is None:
            raise RuntimeError("finite load pattern has not been supplied")
        applied_load = factor*load_pattern if applied_load is None else applied_load
        if getattr(owner,"_curved_reference",False):
            if self._external is not None:raise ValueError("curved reference excludes external load contexts")
            owner._require_curved_load(load_pattern)
            owner._require_curved_load(applied_load)
        count = owner._K.shape[0]
        size = count+len(state.multipliers)
        force_terms = [[] for _ in range(size)]
        rows, columns, data = [], [], []
        shell_candidates = []
        beam_candidates = []
        generalized_material = []
        native_boundary_material = []
        q4_boundary_material = []
        if owner._bl5_native_material or self._paired_fibre_event:
            self._native_boundary_trial = None
            self._q4_boundary_trial = None
        beam_displacement = state.q[
            owner._shell_count:owner._shell_count+self._beam_nodal
        ].reshape(-1, 6)[:, :3]
        beam_positions, beam_low = self._beam_position_pair(state)

        def scatter(slots, force, tangent):
            slots = np.asarray(slots, dtype=int)
            force = np.asarray(force, dtype=float)
            tangent = np.asarray(tangent, dtype=float)
            if (force.shape != (len(slots),) or
                    tangent.shape != (len(slots), len(slots)) or
                    not np.isfinite(force).all() or not np.isfinite(tangent).all()):
                raise RuntimeError("finite local mixed trial operator required")
            for slot, value in zip(slots, force):
                force_terms[int(slot)].append(float(value))
            rows.extend(np.repeat(slots, len(slots)))
            columns.extend(np.tile(slots, len(slots)))
            data.extend(tangent.ravel())

        def scatter_sparse(block):
            if not np.isfinite(block.residual).all() or not np.isfinite(block.tangent.data).all():
                raise RuntimeError("finite sparse attachment trial required")
            for slot, value in zip(block.slots, block.residual):
                force_terms[int(slot)].append(float(value))
            entries = block.tangent.tocoo()
            rows.extend(block.slots[entries.row])
            columns.extend(block.slots[entries.col])
            data.extend(entries.data)

        for (eid, element, slots, _), origin in zip(owner._shell_parts, shell_origins):
            anchors = None if state.shell_anchors is None else state.shell_anchors[slots[::6]//6]
            if len(slots) == 18 and owner._definition.shell.j2_curve is None:
                force, tangent, candidate = variational_shell_response(
                    owner._shell_model, element, state.q[slots], origin, 3,
                    rotation_anchors=anchors, stable_translation=self._stable_s3)
            elif len(slots) == 18:
                s3_response = s3_algorithmic_response(
                    owner._shell_model, element, state.q[slots], origin, 3,
                    rotation_anchors=anchors,
                    **({"capture_material_trial": True} if owner._bl5_native_material else {}))
                force, tangent, candidate = s3_response[:3]
                if owner._bl5_native_material:
                    q4_boundary_material.append((tuple(map(int, slots)), s3_response[3]))
            elif owner._definition.q4_pullback_policy == Q4_ALGORITHMIC_PULLBACK_POLICY:
                q4_response = q4_algorithmic_response(
                    owner._shell_model, element, state.q[slots], origin, 3,
                    rotation_anchors=anchors,
                    prepared_correction=self._q4_corrections.get(eid),
                    stable_rotation=owner._distributing_material,
                    **({"capture_material_trial": True} if owner._bl5_native_material else {}))
                force, tangent, candidate = q4_response[:3]
                if owner._bl5_native_material and q4_response[3] is not None:
                    q4_boundary_material.append((tuple(map(int, slots)), q4_response[3]))
            else:
                total, chart_map = (state.q[slots], None) if anchors is None else total_coordinates(state.q[slots], anchors)
                force, tangent, candidate = corotational_element_response(
                    owner._shell_model, eid, element, total, True,
                    committed_state=origin, num_layers=3,
                    tangent_mode="consistent", centered_shell_reference=True,
                    stable_shell_rotation=self._stable_q4)
                if chart_map is not None:
                    tangent = tangent@chart_map
            scatter(slots, force, tangent)
            shell_candidates.append(candidate)

        for i, ((element, slots, _), origin) in enumerate(
                zip(owner._beam_parts, beam_origins)):
            from ._ge_beam3_retained_fibre import RetainedFibreOperator
            capture_boundary = (owner._bl5_native_material or self._paired_fibre_event) and type(element.operator) is RetainedFibreOperator
            nodes = np.array([owner._beam_index[node] for node in element.node_ids])
            response = element.operator.evaluate(
                beam_positions[nodes], beam_low[nodes], state.frames[nodes],
                state.cells[i], state.resultants[i], origin=origin,
                check=element.operator.guard,
                **({'capture_trial': True} if capture_boundary else {}),
                **(dict(vertices_low=state.frames_low[nodes],rotations_low=state.cells_low[i],
                        resultants_low=state.resultants_low[i]) if self._compensated else {}))
            if owner._coordinate_support:
                from ._stiffened_shell_bl5_native_elastic import require_elastic
                require_elastic(element,response,origin)
            if capture_boundary:
                native_boundary_material.append((i, response.trial_material))
            tangent = spatial_jacobian(
                response.residual, response.hessian+response.hessian_low)
            scatter(slots+owner._shell_count, response.residual, tangent)
            if owner._bl5_native_fibre and not owner._bl5_native_material:
                from ._stiffened_shell_bl5_native_elastic import require_elastic
                require_elastic(element, response, origin)
            if getattr(owner, "_finite_bl5", False) and not owner._bl5_material and not owner._bl5_native_fibre and any(
                    station.accumulated != (0., 0.) for station in response.history.stations):
                raise ValueError("experimental finite BL5 admits elastic beam response only")
            if getattr(owner, "_finite_elastic_footprint", False) or (owner._finite_distributing and not owner._distributing_material):
                _require_elastic_footprint_histories((response.history,), "candidate")
            if getattr(owner, "_curved_reference", False) and not getattr(owner,"_curved_material",False):
                _require_elastic_footprint_histories((response.history,), "candidate",scope="curved Q4 points")
            beam_candidates.append(response.history)
            if owner._bl5_material or owner._bl5_native_material or owner._curved_material or owner._distributing_material or native_point_material:
                from ._ge_beam3_retained_generalized import RetainedGeneralizedOperator
                if type(element.operator) is RetainedGeneralizedOperator:
                    generalized_material.append((i,response.material))

        point_records = []
        for i, (joint, slots) in enumerate(zip(owner._connections,
                                               owner._joint_slots)):
            shell_node = slots[0]//6
            beam_node = (slots[6]-owner._shell_count)//6
            shell_position, shell_low = _position_pair(
                owner._shell_xyz[shell_node], state.q[slots[:3]])
            positions = np.array((
                shell_position, beam_positions[beam_node]))
            position_low = np.array((shell_low, beam_low[beam_node]))
            anchors = np.array((np.eye(3) if state.shell_anchors is None else
                                state.shell_anchors[shell_node], state.frames[beam_node]))
            coordinates = np.zeros((2, 3))
            coordinates[0] = state.q[slots[3:6]]
            trial = joint.evaluate_ports(
                positions, anchors, state.multipliers[6*i:6*i+6],
                rotation_coordinates=coordinates,
                charted_ports=(True, False), position_low=position_low,
                **({'anchors_low': np.array((np.zeros((3,3)),state.frames_low[beam_node]))}
                   if self._compensated else {}))
            point_records.append(point_record(
                owner, i, positions, position_low, trial.pose.residual[:12],
                trial.pose.constraints, "current"))
            extended = np.r_[slots, np.arange(count+6*i, count+6*i+6)]
            # Element and load rows are spatial wrenches. Pull only the
            # increment columns through the shell's additive rotation chart;
            # coordinate-pulled joint rows cannot be summed with these rows.
            scatter(extended, trial.pose.residual,
                    trial.pose.spatial_jacobian@trial.coordinate_map)

        distributing_records = []
        if owner._finite_distributing:
            from ._stiffened_shell_ge_distributing_finite import wrench_record
            for adapter, start in owner._finite_distributing_bindings:
                block = adapter.evaluate(owner, state, state.multipliers[start:start+6],
                                         multiplier_row_start=start)
                scatter_sparse(block)
                distributing_records.append(wrench_record(block, adapter.reference, owner._generation+1))

        line_records = []
        if getattr(owner, "_finite_bl5", False):
            shell_frames = physical_frames(state.q[:owner._shell_count], state.shell_anchors)
            mechanical = Mechanical(beam_positions, beam_low, state.frames,
                                    state.cells, state.resultants)
            point_rows = 6*len(owner._connections)
            for sample, (line_rows, basis) in zip(owner._line_samples, owner._line_terms):
                operator, slots, weight, _edge, _alpha = sample
                rows_at = line_rows+point_rows
                shell_nodes = slots[:12:6]//6
                density = basis@state.multipliers[rows_at]
                local = operator.evaluate_spatial(
                    state.q[slots[:12]].reshape(2, 6), shell_frames[shell_nodes],
                    mechanical, density)
                jacobian = local.hessian.copy()
                column_map = np.eye(21)
                for start in (3, 9, 18):
                    jacobian[start:start+3, start:start+3] -= .5*skew(local.gradient[start:start+3])
                for start, node in zip((3, 9), shell_nodes):
                    left = _exp_terms(state.q[6*node+3:6*node+6])[1]
                    if np.linalg.svd(left, compute_uv=False)[-1] <= 1e-11:
                        raise ValueError("singular BL5 shell chart; owner must rebase")
                    column_map[start:start+3, start:start+3] = left
                mapped = local.gap_jacobian@column_map
                tangent = np.block([[jacobian@column_map, local.gap_jacobian.T@basis],
                                    [basis.T@mapped, np.zeros((len(rows_at),len(rows_at)))]])
                force = np.r_[local.gradient, basis.T@local.gap]
                scatter(np.r_[slots, count+rows_at], weight*force, weight*tangent)
                line_records.append(FiniteLineStationWork(
                    _owned_slots(slots,(21,)), _owned_slots(rows_at,(len(rows_at),)), float(weight), _owned(local.gap),
                    _owned(density), _owned(weight*local.gradient)))

        if self._external is not None:
            body_force, body_jacobian, _ = self._external_at(state, factor)
            applied_load = applied_load+body_force
            external = body_jacobian.tocoo()
            rows.extend(external.row)
            columns.extend(external.col)
            data.extend(-external.data)
        for row, value in enumerate(applied_load):
            force_terms[row].append(-float(value))
        residual = np.fromiter((fsum(terms) for terms in force_terms),
                               dtype=float, count=size)
        matrix = sparse.coo_matrix(
            (data, (rows, columns)), shape=(size, size)).tocsc()
        if not np.isfinite(residual).all() or not np.isfinite(matrix.data).all():
            raise RuntimeError("finite global mixed trial required")
        # At zero load the selected cyclic policy uses its owned unit-factor
        # pattern for normalization only; external force stays exactly zero.
        norm_load = applied_load
        if owner._distributing_cyclic and factor == 0. and not np.any(applied_load != 0.):
            norm_load = owner._cyclic_reference_pattern
        elif programme is not None:
            norm_load = programme.equilibrium_load(applied_load)
        support_gap=0.
        if owner._coordinate_support:
            residual,matrix,norm_load,support_gap,physical_residual,physical_applied=coordinate_support.pullback(
                state.q,state.shell_anchors,owner._shell_count,owner._fixed,residual,matrix,norm_load)
        relative = self._equilibrium_ratio(residual, norm_load)
        if owner._coordinate_support:
            relative=max(relative,self._equilibrium_ratio(physical_residual,physical_applied))
        constraint = _scaled_constraint_gap(
            residual[count:], owner._constraint_scale)
        constraint=max(constraint,support_gap)
        self._line_trial_recovery = (state, tuple(line_records))
        self._point_trial_recovery = (state, tuple(point_records))
        self._distributing_trial_recovery = (state, tuple(distributing_records))
        self._generalized_trial_recovery = (state,beam_origins,tuple(generalized_material))
        if owner._bl5_native_material or self._paired_fibre_event:
            self._native_boundary_trial = (state, beam_origins, owner._generation,
                                           tuple(native_boundary_material))
            self._q4_boundary_trial = (state, shell_origins, owner._generation, tuple(q4_boundary_material))
        return residual, matrix, relative, constraint, tuple(shell_candidates), tuple(beam_candidates)

    def _fibre_material_predictor(self, parent, rejected, residual, free, counts):
        """Approximate parent direction; only the rejected trial's material pp block.

        Retained p coefficients use the immutable reference load map in both
        states. Spatial/chart rotation maps do not act on these coordinates.
        Borrow no trial history and copy its compliance before another trial.
        This is not the consistent Jacobian at the parent or an acceptance test.
        """
        if parent.shape != rejected.shape or parent.shape[0] != len(residual):
            raise ValueError('matching owned predictor operators required')
        predictor = parent.tolil(copy=True)
        for i in range(len(self.owner._beam_parts)):
            start = self.owner._shell_count+self._beam_nodal+24*i+6
            slots = np.arange(start, start+18)
            predictor[np.ix_(slots, slots)] = rejected[slots, :][:, slots].toarray().copy()
        predictor = predictor.tocsr()
        counts['linear_solves'] += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', spla.MatrixRankWarning)
                step = spla.spsolve(predictor[free, :][:, free], -residual[free])
        except spla.MatrixRankWarning as error:
            return None, {'status': 'singular', 'backend': type(error).__name__, 'detail': str(error)}
        if not np.isfinite(step).all():
            return None, {'status': 'nonfinite', 'backend': 'scipy.sparse.linalg.spsolve'}
        full = np.zeros(parent.shape[0]); full[free] = step
        return full, {'status': 'finite', 'backend': 'scipy.sparse.linalg.spsolve'}

    def _coupled_probe_predictor(self, parent, residual, free, counts,
                                 trial, increment, evaluate):
        """Loading-branch pp correction from one just-past-kink probe.

        Near yield, a rejected parent direction may use a no-flow branch.
        A directional probe supplies alternative beam material compliance
        blocks while every other row family stays at the parent state.
        This is a predictor, not a consistent-Jacobian claim or a certified
        yield boundary. The probe is never published; acceptance requires
        evaluation of an advanced candidate with the physical checks.
        """
        try:
            probe = self._advance(trial, increment, 1e-5)
            evaluation = evaluate(probe)
        except (_RotationStepTooLarge, BL5TrialDomainError,
                DistributingPoseDomainError) as error:
            return None, {'status': 'probe-domain', 'detail': type(error).__name__}
        # Retain the already-counted probe evaluation. Strong local curvature
        # can make this fixed small-distance sample the only measured
        # decreasing candidate: the geometric ladder cannot reach its scale.
        # This is only a record; the exhaustion path in _solve_fixed re-checks
        # it against the identical acceptance comparison before using it.
        self._coupled_probe_window = (trial, probe, evaluation)
        predicted, diagnostic = self._fibre_material_predictor(
            parent, evaluation[1], residual, free, counts)
        return predicted, dict(diagnostic, source='probe')

    def _native_boundary_amount(self, state, increment):
        """Copy the proposal before any candidate invalidates parent recovery."""
        if not (self.owner._bl5_native_material or getattr(self, "_paired_fibre_event", False)): return None
        cached = self._native_boundary_trial
        if (cached is None or cached[0] is not state or cached[1] is not self._beam_origins or
                cached[2] != self.owner._generation):
            raise ValueError('stale native material boundary trial')
        amounts = []
        if getattr(self, "_paired_fibre_event", False):
            retained = self.owner._shell_count+self._beam_nodal
            for i, material in cached[3]:
                operator = self.owner._beam_parts[i][0].operator
                dp = increment[retained+24*i+6:retained+24*i+24].tolist()
                amount = operator.cell.paired_boundary_amount(material, dp,
                    origin=self._beam_origins[i], resultants=state.resultants[i],
                    resultants_low=state.resultants_low[i], check=operator.guard)
                if amount is not None: amounts.append(amount)
            return min(amounts) if amounts else None
        shell_cached = self._q4_boundary_trial
        if (shell_cached is None or shell_cached[0] is not state or
                shell_cached[1] is not self._shell_origins or shell_cached[2] != self.owner._generation):
            raise ValueError("stale Q4 material boundary trial")
        from ._stiffened_shell_q4_material_event import boundary_amount as q4_boundary_amount
        from ._stiffened_shell_s3_material_event import boundary_amount as s3_boundary_amount
        for slots, material in shell_cached[3]:
            if material.displacement != tuple(state.q[list(slots)]):
                raise ValueError("changed Q4 material boundary coordinates")
            amount = (s3_boundary_amount if len(slots)==18 else q4_boundary_amount)(
                material, increment[list(slots)], origin=material.origin)
            if amount is not None: amounts.append(amount)
        retained = self.owner._shell_count+self._beam_nodal
        for i, material in cached[3]:
            operator = self.owner._beam_parts[i][0].operator
            if material.resultants != tuple(state.resultants[i]):
                raise ValueError('changed native material boundary resultants')
            dp = increment[retained+24*i+6:retained+24*i+24].tolist()
            amount = operator.cell.boundary_amount(material, dp,
                origin=self._beam_origins[i], check=operator.guard)
            if amount is not None: amounts.append(amount)
        return min(amounts) if amounts else None

    def _generalized_records(self,state,origins,accepted,*,include_native=False):
        from ._stiffened_shell_affine_endpoint import is_native_point_material
        include_native = include_native or (self._compensated and is_native_point_material(self.owner))
        if not (self.owner._bl5_material or self.owner._bl5_native_material or self.owner._curved_material or self.owner._distributing_material or
                include_native):return ()
        cached=self._generalized_trial_recovery
        if cached is None or cached[0] is not state or cached[1] is not origins:
            raise ValueError("stale generalized material recovery")
        from ._stiffened_shell_generalized_recovery import build
        return tuple(build(self.owner._beam_parts[i][0],state.cells[i]+state.cells_low[i] if self._compensated else state.cells[i],raw,accepted[i])
                     for i,raw in cached[2])

    def _point_records(self, state):
        if self._point_trial_recovery is None or self._point_trial_recovery[0] is not state:
            raise ValueError("stale finite point recovery")
        return self._point_trial_recovery[1]

    def _distributing_records(self, state, generation):
        if self._distributing_trial_recovery is None or self._distributing_trial_recovery[0] is not state:
            raise ValueError("stale finite distributing recovery")
        return tuple(replace(record, generation=generation) for record in self._distributing_trial_recovery[1])

    def _increment_origin(self, state):
        """Detach one footprint Newton chart without publishing or moving history.

        Equivalent accepted charts start the same numerical problem. The anchor
        remains fixed for the whole increment; derivatives and _advance continue
        to use their existing additive-coordinate contract. Empty-footprint
        owners preserve their historical chart and numerical route.
        """
        owner = self.owner
        if not owner._footprint_increment_chart:
            return state
        require_rotation_triples(owner._fixed, len(owner._shell_ids))
        local = state.q[:owner._shell_count]
        if not np.any(local.reshape(-1, 6)[:, 3:] != 0.):
            return state
        anchors = owned_anchors(physical_frames(local, state.shell_anchors),
                                len(owner._shell_ids))
        q = state.q.copy()
        q[:owner._shell_count].reshape(-1, 6)[:, 3:] = 0.
        return replace(state, q=_owned(q), shell_anchors=anchors)

    @staticmethod
    def _zero_compensation(state):
        return replace(state,q_low=_owned(np.zeros_like(state.q)),frames_low=_owned(np.zeros_like(state.frames)),
            cells_low=_owned(np.zeros_like(state.cells)),resultants_low=_owned(np.zeros_like(state.resultants)))

    def _beam_position_pair(self,state):
        dis=state.q[self.owner._shell_count:self.owner._shell_count+self._beam_nodal].reshape(-1,6)[:,:3]
        if not self._compensated:return _position_pair(self._reference_positions,dis)
        from ._stiffened_shell_f06_increment_arithmetic import Pair
        lo=state.q_low[self.owner._shell_count:self.owner._shell_count+self._beam_nodal].reshape(-1,6)[:,:3]
        p=Pair(self._reference_positions)+Pair(dis,lo)
        return p.high.copy(),p.low.copy()

    def _advance(self, state, increment, amount):
        owner = self.owner
        delta = amount*increment
        q = state.q.copy()
        frames = state.frames.copy()
        cells = state.cells.copy()
        resultants = state.resultants.copy()
        q[:owner._shell_count] += delta[:owner._shell_count]
        if owner._coordinate_support:
            try:
                coordinate_support.advance(state.q,q,state.shell_anchors,owner._shell_count,owner._fixed,delta)
            except coordinate_support.CoordinateSupportChartDomainError as exc:
                raise _RotationStepTooLarge(str(exc)) from exc
        for node in range(len(owner._beam_ids)):
            slot = owner._shell_count+6*node
            q[slot:slot+3] += delta[slot:slot+3]
            angle = delta[slot+3:slot+6]
            if np.linalg.norm(angle) >= .9*np.pi:
                raise _RotationStepTooLarge("finite beam rotation increment requires cutback")
            frames[node] = rotation(angle)@frames[node]
        retained = owner._shell_count+self._beam_nodal
        for i in range(len(owner._beam_parts)):
            base = retained+24*i
            for cell in range(2):
                angle = delta[base+3*cell:base+3*cell+3]
                if np.linalg.norm(angle) >= .9*np.pi:
                    raise _RotationStepTooLarge("finite cell rotation increment requires cutback")
                cells[i, cell] = rotation(angle)@cells[i, cell]
            resultants[i] += delta[base+6:base+24]
        multipliers = state.multipliers+delta[len(q):]
        if self._compensated:
            from ._stiffened_shell_f06_increment_arithmetic import add_state,updated_rotation
            ql=state.q_low.copy();fl=state.frames_low.copy();cl=state.cells_low.copy();pl=state.resultants_low.copy()
            for node in range(len(owner._beam_ids)):
                slot=owner._shell_count+6*node
                q[slot:slot+3],ql[slot:slot+3]=add_state(state.q[slot:slot+3],ql[slot:slot+3],delta[slot:slot+3])
                frames[node],fl[node]=updated_rotation(state.frames[node],fl[node],delta[slot+3:slot+6])
            for i in range(len(owner._beam_parts)):
                base=retained+24*i
                for cell in range(2):
                    cells[i,cell],cl[i,cell]=updated_rotation(state.cells[i,cell],cl[i,cell],delta[base+3*cell:base+3*cell+3])
                resultants[i],pl[i]=add_state(state.resultants[i],pl[i],delta[base+6:base+24])
            return _State(q,frames,cells,resultants,multipliers,state.shell_anchors,
                _owned(ql),_owned(fl),_owned(cl),_owned(pl))
        return _State(q, frames, cells, resultants, multipliers, state.shell_anchors)

    def solve(self, actions, targets, *, cancellation_token=None,
              max_iterations=12, max_backtracks=8, increment_control=None,
              on_accepted=None):
        if self.owner._coordinate_support and (increment_control is not None or self._external is not None or self._programme is not None):
            raise ValueError('coordinate shell supports require fixed-force native point solve')
        if self.owner._distributing_cyclic and increment_control is not None:
            raise ValueError("cyclic distributing material histories admit fixed increments only")
        if self.owner._footprint_increment_chart:
            require_rotation_triples(self.owner._fixed, len(self.owner._shell_ids))
        pending = self._adaptive_state is not None and not self._adaptive_state.completed
        if increment_control is None:
            if pending:
                raise ValueError("pending adaptive sequence requires matching targets and controls")
            return self._solve_fixed(actions, targets, cancellation_token=cancellation_token,
                                     max_iterations=max_iterations, max_backtracks=max_backtracks,
                                     on_accepted=on_accepted)
        if type(increment_control) is not AdaptiveIncrementControl:
            raise ValueError("typed adaptive increment control required")
        if self._compensated:
            from ._stiffened_shell_staged_load_experimental import StagedDeadLoadProgramme
            if not compensated_adaptive_load_admitted(self.owner, actions):
                raise ValueError('compensated adaptive control requires an admitted staged-fibre or nodal-mixed point owner')
        if (type(targets) is not tuple or not targets or
                any(not isinstance(v, (int, float, np.integer, np.floating)) or
                    isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or v <= 0.
                    for v in targets) or
                type(max_iterations) is not int or not 1 <= max_iterations <= 24 or
                type(max_backtracks) is not int or not 0 <= max_backtracks <= 12):
            raise ValueError("finite adaptive targets and Newton controls required")
        targets = tuple(float(v) for v in targets)
        if pending:
            control = self._adaptive_state
            if (targets != control.targets or increment_control != control.control or
                    max_iterations != control.max_iterations or max_backtracks != control.max_backtracks):
                raise ValueError("pending adaptive sequence requires matching targets and controls")
        else:
            if any(a == b for a, b in zip((self._load_factor,)+targets, targets)):
                raise ValueError("adaptive load factors must change")
            control = _AdaptiveState(increment_control, targets, max_iterations,
                                     max_backtracks, 0, float(self._load_factor))
            # Bound metadata with the exact identity the published controller
            # envelope will carry (the selected polishing candidate tags it).
            encode_control(self._checkpoint_identity(), control, b"resource-check")
        remaining = targets[control.cursor:]
        if self.owner._finite_distributing and any(b <= a for a, b in zip(
                (self._load_factor,)+remaining, remaining)):
            raise ValueError("finite distributing footprint force factors must increase monotonically")
        pattern, programme = self._prepare_load(actions, remaining)
        while not control.completed:
            if len(control.attempts) >= control.control.max_attempts:
                raise AdaptiveIncrementExhausted("adaptive increment attempt limit", control, self._accepted_factors)
            factor = control.next_factor(self._load_factor)
            if not np.isfinite(factor) or factor <= 0. or factor == self._load_factor:
                raise AdaptiveIncrementExhausted("adaptive increment is not representable", control, self._accepted_factors)

            def accepted(attempt):
                next_control = control.accepted(replace(attempt, target=control.targets[control.cursor]))
                encode_control(self._checkpoint_identity(), next_control, b"resource-check")
                return next_control

            try:
                self._solve_fixed(actions, (factor,), cancellation_token=cancellation_token,
                    max_iterations=max_iterations, max_backtracks=max_backtracks,
                    _prepared_pattern=pattern, _prepared_programme=programme,
                    _accepted_hook=accepted)
            except FiniteIncrementFailure as error:
                attempt = replace(error.attempt, target=control.targets[control.cursor])
                rejected = replace(control, attempts=control.attempts+(attempt,))
                if control.depth >= control.control.max_cutbacks:
                    raise AdaptiveIncrementExhausted("adaptive increment cutback limit", rejected,
                                                    self._accepted_factors) from error
                control = control.rejected(attempt)
            else:
                control = self._adaptive_state
                # Outside the candidate-failure handler: even a typed numerical
                # exception raised by an observer must not trigger cutback.
                if on_accepted is not None:
                    on_accepted(deepcopy(self._accepted))
        return self._accepted

    def _bind_staged_force_polish(self, programme):
        """Bind the admissible polishing policy for a fresh staged force solve.

        The only auto-selection seam of the private candidate: the actual
        force driver solve, after `_prepare_load` validated and compiled the
        staged programme. Binding requires a fresh uncompensated owner of
        the admitted native point chart domain whose private selector was
        never explicitly set. Single-force solves (no programme) and the
        arc/displacement routes never reach a staged programme here, so
        they keep their default first-admissible route; already accepted or
        restored legacy states never auto switch.
        """
        if (programme is None or self._admissible_polish or
                self._accepted is not None or self._compensated or
                _explicit_admissible_polish(self.owner) is not None or
                not supports_native_chart_polish(self.owner)):
            return
        self._admissible_polish = True

    def _solve_fixed(self, actions, targets, *, cancellation_token=None,
              _prepared_pattern=None, _prepared_programme=None, _accepted_hook=None,
              max_iterations=12, max_backtracks=8, on_accepted=None):
        owner = self.owner
        if owner._coordinate_support:
            from ._stiffened_shell_ge_experimental import NodalAction
            if (type(actions) is not tuple or any(type(a) is not NodalAction or any(a.wrench[3:]) for a in actions) or
                    self._external is not None or self._programme is not None or _prepared_pattern is not None or
                    _prepared_programme is not None or _accepted_hook is not None):
                raise ValueError('coordinate shell supports require direct fixed nodal translation forces')
        if (type(targets) is not tuple or not targets or
                any(not isinstance(v, (int, float, np.integer, np.floating))
                    or isinstance(v, (bool, np.bool_)) or
                    not np.isfinite(v) or (not owner._distributing_cyclic and v <= 0.)
                    for v in targets) or
                any(a == b for a, b in zip((self._load_factor,)+targets,
                                            targets))):
            raise ValueError("positive finite load factors must change at each increment")
        if owner._finite_distributing and not owner._distributing_cyclic and any(b <= a for a,b in zip((self._load_factor,)+targets,targets)):
            raise ValueError("finite distributing footprint force factors must increase monotonically")
        max_trial_halvings=16 if (owner._curved_material or owner._distributing_material) else 12
        if (type(max_iterations) is not int or not 1 <= max_iterations <= 24 or
                type(max_backtracks) is not int or not 0 <= max_backtracks <= max_trial_halvings):
            raise ValueError("bounded finite Newton controls required")
        if _prepared_programme is None and _prepared_pattern is None:
            pattern, programme = self._prepare_load(actions, targets)
        else:
            pattern, programme = _prepared_pattern, _prepared_programme
        # Validated staged compilation precedes any auto policy binding.
        self._bind_staged_force_polish(programme)
        free = np.r_[owner._free,
                     np.arange(owner._K.shape[0],
                               owner._K.shape[0]+len(self._state.multipliers))]
        from ._stiffened_shell_same_parent_trial import FixedTrialHandoff
        allow_material_predictor = (_accepted_hook is not None and
                                    supports_paired_fibre_adaptive(owner))
        allow_coupled_probe = (not allow_material_predictor and
                               owner._bl5_material and owner._bl5_shell_material)
        self._material_predictor_diagnostics = []
        allow_handoff = (not owner._finite_distributing and programme is None and _accepted_hook is None and
                         self._adaptive_state is None)
        # The private polishing candidate stays inside its admitted native
        # point chart solve domain; a late external/unsupported context is an
        # explicit refusal, never a silent policy change. Legacy drivers
        # restored before this policy existed keep their default route
        # (missing selector means False).
        polish = getattr(self, "_admissible_polish", False) is True
        if polish and (self._external is not None or
                not supports_native_chart_polish(owner)):
            raise ValueError("admissible polish candidate requires its native point chart solve domain")
        # Bounded optional-polish reserve: the exact per-attempt evaluation
        # ceiling the controller replay enforces. Once a state is retained,
        # the last evaluation unit inside this ceiling is reserved for the
        # one required retained-state re-verification at publish, so optional
        # improvement can stop there and no published attempt can exceed the
        # replay ceiling. Iteration and backtrack budgets are unchanged.
        polish_evaluation_ceiling = 1 + (max_backtracks + 2) * max_iterations
        for factor in targets:
            handoff = None
            applied_load = None if programme is None else programme.load_at(factor)
            cancellation_safe_point(cancellation_token, "stiffened-shell.finite.increment")
            trial = self._increment_origin(self._state)
            counts = {"evaluations": 0, "linear_solves": 0, "rejected_trials": 0}
            accepted_from = float(self._load_factor)
            def evaluated(state):
                counts["evaluations"] += 1
                return self._assemble(state, factor, self._shell_origins,
                                      self._beam_origins, pattern=pattern,
                                      **({} if programme is None else
                                         {"applied_load": applied_load,
                                          "load_programme": programme}))
            def attempt(status, iteration, relative, gap, failure=None):
                return IncrementAttempt(float(factor), accepted_from, float(factor),
                    status, iteration, **counts, equilibrium_ratio=float(relative),
                    constraint_max=float(gap), failure=failure)
            # Last retained admissible state of the polishing candidate:
            # (state, evaluation). polish_fresh records whether the latest
            # assembly still belongs to that state.
            polish_best = None
            polish_fresh = False

            def polish_reserve_reached():
                # True once only the one required retained-state re-evaluation
                # remains inside the replay ceiling: optional improvement
                # must stop so the published attempt stays admissible to the
                # controller replay with its exact honest counters.
                return (polish and polish_best is not None and
                        counts["evaluations"] + 1 >= polish_evaluation_ceiling)

            def publish_polished(total_work):
                state, evaluation = polish_best
                if not polish_fresh:
                    # A discarded candidate was assembled after the retained
                    # state; deterministically re-assemble it so its recovery
                    # payloads are current, and re-verify both limits before
                    # the atomic commit. Never publish a stale or
                    # inadmissible payload.
                    evaluation = evaluated(state)
                    if not (evaluation[2] <= 1e-8 and evaluation[3] <= 1e-9):
                        raise RuntimeError("admissible polish candidate lost its accepted limits")
                cancellation_safe_point(cancellation_token,
                                        "stiffened-shell.finite.commit")
                # Truthful work: the accepted attempt reports ALL Newton work
                # of this increment, including the rejected polishing trials
                # after the retained state was first found. The retained
                # state's equilibrium metrics stay its own; its stored
                # iteration can never hide the extra solves.
                self._publish_equilibrium(state, evaluation, factor,
                    pattern=pattern, programme=programme,
                    applied_load=applied_load, cancellation_token=cancellation_token,
                    iteration=total_work, attempt=attempt,
                    accepted_hook=_accepted_hook, on_accepted=on_accepted)

            for iteration in range(max_iterations+1):
                cancellation_safe_point(cancellation_token, "stiffened-shell.finite.trial")
                if polish_reserve_reached():
                    # Only the required retained-state re-evaluation remains
                    # inside the replay ceiling: stop optional improvement and
                    # publish the retained state with the truthful work.
                    publish_polished(iteration)
                    break
                # Observe cancellation before validating a single-use payload.
                ready = None if handoff is None else handoff.take(self, trial, factor, pattern)
                handoff = None
                residual, matrix, relative, gap, shell_next, beam_next = (
                    evaluated(trial) if ready is None else ready)
                if relative <= 1e-8 and gap <= 1e-9:
                    if not polish:
                        cancellation_safe_point(cancellation_token,
                                                "stiffened-shell.finite.commit")
                        self._publish_equilibrium(trial,
                            (residual, matrix, relative, gap, shell_next, beam_next),
                            factor, pattern=pattern, programme=programme,
                            applied_load=applied_load, cancellation_token=cancellation_token,
                            iteration=iteration, attempt=attempt,
                            accepted_hook=_accepted_hook, on_accepted=on_accepted)
                        break
                    # Private polishing candidate: the first admissible state is
                    # retained, not published. Later admissible states replace
                    # it only on a strict decrease of the same comparison
                    # merit; anything else finishes at the last valid state.
                    if (polish_best is not None and
                            max(relative, gap) >= max(polish_best[1][2], polish_best[1][3])):
                        polish_fresh = False
                        publish_polished(iteration)
                        break
                    polish_best = (trial,
                        (residual, matrix, relative, gap, shell_next, beam_next))
                    polish_fresh = True
                    if iteration == max_iterations:
                        publish_polished(iteration)
                        break
                if iteration == max_iterations:
                    if polish and polish_best is not None:
                        # The retained admissible state is a complete accepted
                        # increment; the extra polishing work simply failed to
                        # improve it. Publish it with the truthful total work.
                        polish_fresh = False
                        publish_polished(iteration)
                        break
                    raise FiniteIncrementFailure(attempt("rejected", iteration,
                        relative, gap, "finite mixed Newton iteration limit"))
                with warnings.catch_warnings():
                    warnings.simplefilter("error", spla.MatrixRankWarning)
                    counts["linear_solves"] += 1
                    step = spla.spsolve(matrix[free, :][:, free],
                                       -residual[free])
                if not np.isfinite(step).all():
                    raise RuntimeError("finite mixed Newton correction required")
                full = np.zeros(matrix.shape[0])
                full[free] = step
                original_full = full  # retained parent Newton direction
                old_merit = max(relative, gap)
                # A follower load can activate an initially zero component. Comparing
                # candidates in different active-row norms can reject a Newton step
                # despite a large physical residual reduction. Freeze only the
                # backtracking comparison; the next iteration still requires the
                # unchanged current-state equilibrium and compatibility checks.
                merit_applied = None
                trial_ratio = _linear_equilibrium
                vector_merit = getattr(self._external, "pressure_vector_equilibrium", False)
                if getattr(self._external, "freeze_trial_load_norm", False):
                    # _prepare_load excludes combined nodal/programme loads;
                    # this is the complete applied load for this profile.
                    merit_applied = self._external_at(trial, factor)[0]
                    global_ratio = _trial_work_ratio(residual[owner._free],
                        merit_applied[owner._free], owner._coordinate_scale[owner._free])
                    # Once global equilibrium and compatibility meet their
                    # acceptance limits, polish the remaining component checks.
                    # Polishing permits global-ratio increases; the next
                    # current-state acceptance check still requires all limits.
                    if not vector_merit and (global_ratio > 1e-8 or gap > 1e-9):
                        trial_ratio = _trial_work_ratio
                    old_merit = max(self._frozen_trial_ratio(
                        residual, merit_applied, trial_ratio), gap)
                # The paired first-event heuristic remains a fixed-route experiment.
                # Adaptive point increments use the established geometric cuts.
                boundary_amount = self._native_boundary_amount(trial, full) if (owner._bl5_native_material or
                    (self._paired_fibre_event and _accepted_hook is None)) else None
                first_cutback = .5 if boundary_amount is None else min(.5, boundary_amount)
                # A near-zero advisory event must not starve the ordinary geometric
                # ladder: keep the full trial and the boundary probe, and once that
                # probe rejects, resume the standard cuts instead of halving the
                # event forever. The rejected probe barely moved the candidate, so
                # it cannot speak against the ordinary cuts.
                geometric_cuts = boundary_amount is None or first_cutback >= .5
                # One probe sample per iteration: a record from an earlier
                # iteration or a typed probe failure must never be accepted
                # at this iteration's exhaustion.
                self._coupled_probe_window = None
                for backtrack in range(max_backtracks+1):
                    cancellation_safe_point(
                        cancellation_token, "stiffened-shell.finite.backtrack")
                    if polish_reserve_reached():
                        # Reserve the last evaluation unit inside the replay
                        # ceiling for the retained-state re-verification; the
                        # next iteration top publishes the retained state.
                        break
                    try:
                        if backtrack == 0:
                            amount = 1.
                        elif backtrack == 1 or geometric_cuts:
                            amount = first_cutback*.5**(backtrack-1)
                        else:
                            amount = .5**(backtrack-1)
                        candidate = self._advance(trial, full, amount)
                    except _RotationStepTooLarge:
                        if backtrack == 1 and allow_material_predictor:
                            full = original_full
                        counts["rejected_trials"] += 1
                        continue
                    try:
                        before = (FixedTrialHandoff.begin(self, candidate, factor, pattern)
                                  if allow_handoff else None)
                        changed = evaluated(candidate)
                        if polish:
                            polish_fresh = False
                    except (BL5TrialDomainError, DistributingPoseDomainError):
                        if backtrack == 1 and allow_material_predictor:
                            full = original_full
                        counts["rejected_trials"] += 1
                        continue
                    if merit_applied is None:
                        candidate_relative = changed[2]
                    else:
                        candidate_relative = self._frozen_trial_ratio(
                            changed[0], merit_applied, trial_ratio)
                    # Current-state equilibrium is sufficient to advance to
                    # the next mandatory acceptance check. A frozen comparison
                    # norm must not reject an already equilibrated candidate.
                    # Before the first admissible state the polishing
                    # candidate keeps the ordinary Newton ladder unchanged.
                    # Once a state is retained the ladder is stricter: a
                    # candidate advances only when it still meets BOTH
                    # physical limits and strictly decreases the retained
                    # comparison merit; a lower merit that violates either
                    # limit is refused.
                    if ((not polish or polish_best is None) and
                            ((changed[2] <= 1e-8 and changed[3] <= 1e-9) or
                             max(candidate_relative, changed[3]) < old_merit) or
                            polish and polish_best is not None and
                            (changed[2] <= 1e-8 and changed[3] <= 1e-9 and
                             max(candidate_relative, changed[3]) <
                             max(polish_best[1][2], polish_best[1][3]))):
                        trial = candidate
                        if allow_handoff:
                            handoff = FixedTrialHandoff.capture(self, trial, factor, pattern, changed, before=before)
                        break
                    if backtrack == 0 and allow_material_predictor:
                        cancellation_safe_point(cancellation_token, "stiffened-shell.finite.predictor")
                        predicted, diagnostic = self._fibre_material_predictor(
                            matrix, changed[1], residual, free, counts)
                        self._material_predictor_diagnostics.append(dict(
                            factor=float(factor), iteration=iteration, **diagnostic))
                        cancellation_safe_point(cancellation_token, "stiffened-shell.finite.predictor")
                        if predicted is not None:
                            full = predicted
                    if (backtrack == 0 and allow_coupled_probe and
                            not polish_reserve_reached()):
                        # The full Newton step was merit-rejected while the
                        # accepted state may sit at a simultaneous yield kink.
                        # Measured evidence: the minus-side physical rows
                        # agree with the assembled tangent, but the retained
                        # rows show a one-sided finite-difference mismatch
                        # (~1), so no whole-tangent minus-side derivative
                        # claim is made. One directional probe just past the
                        # kink along the rejected direction supplies a
                        # loading-branch sample of the beam pp blocks; it is
                        # not a certified kink boundary. Splicing them into
                        # the parent replaces the direction for the whole
                        # remaining ladder. The probe evaluation itself never
                        # publishes: it supplies the pp blocks and one
                        # measured small-step sample whose acceptance still
                        # requires the ordinary merit and equilibrium checks.
                        cancellation_safe_point(cancellation_token,
                                                "stiffened-shell.finite.probe")
                        predicted, diagnostic = self._coupled_probe_predictor(
                            matrix, residual, free, counts, trial, full, evaluated)
                        self._material_predictor_diagnostics.append(dict(
                            factor=float(factor), iteration=iteration, **diagnostic))
                        cancellation_safe_point(cancellation_token,
                                                "stiffened-shell.finite.probe")
                        if predicted is not None:
                            full = predicted
                    # The copied pp direction gets only the existing half trial.
                    # Rejection restores the parent direction for remaining fractions.
                    if backtrack == 1 and allow_material_predictor:
                        full = original_full
                    counts["rejected_trials"] += 1
                else:
                    if polish and polish_best is not None:
                        # The exhausted ladder found no further admissible
                        # decrease. The retained state is already a complete
                        # accepted increment: publish it with the truthful
                        # total work instead of failing the increment.
                        polish_fresh = False
                        publish_polished(iteration+1)
                        break
                    # The exhausted ladder found no decreasing cut, yet the
                    # coupled probe already paid for one fully evaluated
                    # candidate at its fixed small distance. Measured at the
                    # captured coupled P0 parent, the merit descends there
                    # (secants near -merit) with strong positive curvature,
                    # so every geometric fraction overshoots the descent
                    # window while the probe sample sits inside it. Accept
                    # that measured strict decrease under the identical
                    # comparison; anything else keeps the truthful failure.
                    window = (self._coupled_probe_window
                              if allow_coupled_probe else None)
                    window_state = None
                    if window is not None and window[0] is trial:
                        cancellation_safe_point(
                            cancellation_token, "stiffened-shell.finite.probe")
                        evaluation = window[2]
                        probe_relative = (evaluation[2] if merit_applied is None else
                            self._frozen_trial_ratio(
                                evaluation[0], merit_applied, trial_ratio))
                        if ((evaluation[2] <= 1e-8 and evaluation[3] <= 1e-9) or
                                max(probe_relative, evaluation[3]) < old_merit):
                            window_state = window[1]
                            self._material_predictor_diagnostics.append(dict(
                                factor=float(factor), iteration=iteration,
                                status='probe-window-accept'))
                    if window_state is None:
                        raise FiniteIncrementFailure(attempt("rejected", iteration+1,
                            relative, gap, "finite mixed backtrack budget exhausted"))
                    trial = window_state
        return self._accepted

    def _publish_equilibrium(self, trial, evaluation, factor, *, pattern,
                             programme, applied_load, cancellation_token,
                             iteration, attempt, accepted_hook=None,
                             on_accepted=None):
        """Prepare all recovery/checkpoint data before one atomic publication.

        The force controller supplies load/history semantics. Keeping preparation
        here lets additional controllers share the physical/material transaction.
        Call only from the owner's state-writer context after acceptance checks.
        """
        owner = self.owner
        residual, _, relative, gap, shell_next, beam_next = evaluation
        support = np.zeros(owner._K.shape[0])
        support[owner._fixed] = residual[owner._fixed]
        if owner._coordinate_support:
            support=coordinate_support.reactions(trial.q,trial.shell_anchors,owner._shell_count,owner._fixed,residual,owner._K.shape[0])
        next_factors = self._accepted_factors+(float(factor),)
        if owner._definition.shell.j2_curve is not None:
            next_shell_origins = _accepted_shell_material_origins(
                owner, trial, shell_next)
        else:
            next_shell_origins = shell_next
        q4_count = len(owner._definition.shell.q4_elements)
        next_result = FinitePointResult(
            "completed", float(factor),
            _owned(trial.q[:owner._shell_count]),
            _owned(self._beam_positions(trial)),
            _owned(trial.frames), _owned(trial.cells),
            _owned(trial.resultants), _owned(trial.multipliers),
            _owned(support), relative, gap, iteration,
            owner._generation+1, next_factors,
            _material_observables(beam_next),
            (tuple(tuple(float(v) for v in state["alpha"])
                   for state in shell_next)
             if owner._definition.shell.j2_curve is not None else ()),
            (_shell_q4_recovery(owner._shell_parts[:q4_count],
                                shell_next[:q4_count])
             if owner._definition.shell.j2_curve is not None else ()),
            (_shell_s3_recovery(owner._shell_parts[q4_count:],
                                shell_next[q4_count:])
             if owner._definition.shell.j2_curve is not None else ()))
        cancellation_safe_point(cancellation_token,
                                "stiffened-shell.finite.publish")
        next_result = _with_shell_chart(next_result, trial, owner._shell_count)
        next_result = replace(next_result, point_wrenches=self._point_records(trial),
            beam_generalized_material=self._generalized_records(trial,self._beam_origins,beam_next))
        if owner._finite_distributing:
            next_result = replace(next_result, distributing_footprint_wrenches=
                self._distributing_records(trial, owner._generation+1))
        next_result = self._programme_result(next_result, programme, factor, applied_load)
        next_result = self._external_result(next_result, trial, factor)
        if getattr(owner, "_finite_bl5", False):
            if self._line_trial_recovery[0] is not trial:
                raise ValueError("stale finite BL5 station recovery")
            next_result = replace(next_result,
                beam_shell_line_work=self._line_trial_recovery[1])
        if owner._has_physical_fibre:
            fibre_records, fibre_energies = _fibre_recovery(
                owner, trial, self._beam_origins, beam_next)
            next_result = replace(next_result, beam_fibre_material=fibre_records,
                                  beam_fibre_energy=fibre_energies)
            cancellation_safe_point(cancellation_token,
                                    "stiffened-shell.finite.material-publish")
        previous_shell_q = _owned(self._state.q[:owner._shell_count])
        previous_shell_anchors = (self._shell_predecessor_anchors
            if self._state.shell_anchors is None else self._state.shell_anchors)
        next_control = (None if accepted_hook is None else
            accepted_hook(attempt("accepted", iteration, relative, gap)))
        if next_control is not None:
            next_result = replace(next_result,
                status="completed" if next_control.completed else "accepted_prefix",
                increment_attempts=next_control.attempts)
        next_checkpoint = (None if next_control is None and programme is None and self._external is None and not owner._distributing_material else
            self._prospective_checkpoint(next_result, _state=trial,
                _shell_predecessors=self._shell_origins,
                _shell_predecessor_q=previous_shell_q,
                _shell_predecessor_anchors=previous_shell_anchors,
                _shell_origins=next_shell_origins,
                _beam_predecessors=self._beam_origins, _beam_origins=beam_next,
                _load_factor=float(factor), _accepted_factors=next_factors,
                _pattern=pattern, _programme=programme,
                _applied_load=applied_load, _adaptive_state=next_control,
                _curved_material_predecessors=bool(owner._curved_material)))
        if next_control is not None or programme is not None or self._external is not None or owner._distributing_material:
            cancellation_safe_point(cancellation_token,
                                    "stiffened-shell.finite.controller-publish")
        with owner._commit_lock:
            self._shell_predecessors = self._shell_origins
            self._shell_predecessor_q = previous_shell_q
            self._shell_predecessor_anchors = previous_shell_anchors
            self._state = trial
            self._shell_origins = next_shell_origins
            self._beam_predecessors = self._beam_origins
            self._beam_origins = beam_next
            self._curved_material_predecessors = bool(owner._curved_material)
            self._load_factor = float(factor)
            self._accepted_factors = next_factors
            self._pattern = pattern
            self._programme = programme
            self._applied_load = applied_load
            owner._generation += 1
            self._accepted = next_result
            self._adaptive_state = next_control
            self._adaptive_checkpoint = next_checkpoint
            if owner._finite_driver is None:
                owner._finite_driver = self
        if on_accepted is not None:
            on_accepted(deepcopy(next_result))
        return next_result

    def _beam_positions(self, state):
        if self._compensated:
            high,low=self._beam_position_pair(state)
            return high+low
        positions = self._reference_positions.copy()
        for node in range(len(self.owner._beam_ids)):
            slot = self.owner._shell_count+6*node
            positions[node] += state.q[slot:slot+3]
        return positions

    def recover(self):
        if self._accepted is None:
            raise RuntimeError("no accepted finite mixed increment")
        return self._accepted

    def _external_result(self, result, state, factor):
        if self._external is None:
            return result
        result = replace(result, **{self._external.result_field: self._external_at(state, factor)[2]})
        if hasattr(self._external, 'programme_result'):
            result = self._external.programme_result(result, factor)
        return result

    def _external_at(self, state, factor):
        if hasattr(self._external, 'evaluate_at'):
            return self._external.evaluate_at(state, factor)
        force, jacobian, recovery = self._external.evaluate(state)
        return factor*force, factor*jacobian, recovery.scaled(factor)

    def _prospective_checkpoint(self, result, **changes):
        """Serialize the prepared accepted snapshot before publishing it.

        This read-only view shares immutable compiled data, never owns a
        second analysis or commits a component. No scratch view is retained.
        """
        prepared = copy(self)
        prepared.__dict__.update(changes)
        prepared.owner = SimpleNamespace(**{**self.owner.__dict__,
                                           "_generation": result.generation})
        prepared._accepted = result
        prepared._adaptive_checkpoint = None
        return prepared.checkpoint()

    def rebase_rotations(self, expected_generation, *, cancellation_token=None,
                         preserve_supported_charts=False):
        """Atomic same-load chart event; constitutive histories are untouched."""
        owner = self.owner
        from ._stiffened_shell_affine_endpoint import is_native_point_material
        if type(preserve_supported_charts) is not bool:
            raise ValueError("preserve_supported_charts must be a strict boolean")
        if (preserve_supported_charts and owner._definition.shell.q4_elements and
                owner._definition.q4_pullback_policy != Q4_ALGORITHMIC_PULLBACK_POLICY):
            raise ValueError("selective chart rebase requires explicit algorithmic Q4 pullback policy")
        if type(expected_generation) is not int:
            raise ValueError("integer accepted predecessor generation required")
        with owner._commit_lock:
            if self._accepted is None or expected_generation != owner._generation:
                raise ValueError("stale or missing accepted rebase predecessor")
            previous = self._state
        fixed = set(owner._fixed)
        eligible=[]
        for node in range(len(owner._shell_ids)):
            restrained = [6*node+j in fixed for j in (3, 4, 5)]
            eligible.append(not any(restrained))
            if not owner._coordinate_support and not preserve_supported_charts and any(restrained) and not all(restrained):
                raise ValueError("shell anchor reset requires full rotational triples; partial supports need transformed equations")
        cancellation_safe_point(cancellation_token, "stiffened-shell.rebase.prepare")
        local = previous.q[:owner._shell_count]
        reset=np.asarray(eligible) if preserve_supported_charts else np.ones(len(eligible),dtype=bool)
        if not np.any(local.reshape(-1, 6)[reset, 3:] != 0.):
            raise ValueError("no nonzero shell chart coordinates to rebase")
        frames = physical_frames(local, previous.shell_anchors)
        anchor_values=(np.tile(np.eye(3),(len(eligible),1,1)) if previous.shell_anchors is None else
                       previous.shell_anchors.copy())
        anchor_values[reset]=frames[reset]
        anchors = owned_anchors(anchor_values, len(owner._shell_ids))
        q = previous.q.copy()
        q[:owner._shell_count].reshape(-1, 6)[reset, 3:] = 0.
        trial = replace(previous, q=_owned(q), shell_anchors=anchors)
        if not np.allclose(physical_frames(q[:owner._shell_count], anchors), frames,
                           rtol=0., atol=1e-12):
            raise ValueError("rebase changed physical shell orientations")
        # Rebase is the same accepted constitutive generation, not an unloading
        # update. The exact native beam history guard must replay the accepted
        # update from its stored predecessor: re-evaluating from the accepted
        # history re-runs the return mapping on a zero increment and rounds
        # fibre entries at the last ulp instead of reproducing them.
        native_beam_rebase = (owner._has_physical_fibre or
            (self._compensated and is_native_point_material(owner)))
        rebase_beam_origins = (self._beam_predecessors if native_beam_rebase
            else self._beam_origins)
        residual, _, relative, gap, shell_repeated, beam_repeated = self._assemble(
            trial, self._load_factor,
            self._shell_predecessors if self._compensated_coupled else self._shell_origins,
            rebase_beam_origins)
        if owner._has_physical_fibre or (self._compensated and is_native_point_material(owner)):
            if beam_repeated != self._beam_origins:
                raise ValueError("shell chart rebase changed exact native beam material history")
            saved_arrays = native_history_arrays(
                owner._beam_parts, self._beam_origins, "saved")
            repeated_arrays = native_history_arrays(
                owner._beam_parts, beam_repeated, "repeated")
            for (_saved_name, saved), (_repeated_name, repeated) in zip(
                    saved_arrays, repeated_arrays, strict=True):
                if not np.array_equal(saved, repeated):
                    raise ValueError("shell chart rebase changed native beam material history")
        if owner._bl5_material or getattr(owner,'_curved_material',False):
            for saved, repeated in zip(_material_arrays(self._beam_origins),
                                       _material_arrays(beam_repeated)):
                scale = max(float(np.max(np.abs(saved), initial=0.)),
                            float(np.max(np.abs(repeated), initial=0.)), np.finfo(float).tiny)
                if np.max(np.abs(saved-repeated), initial=0.) > 64.*np.finfo(float).eps*scale:
                    raise ValueError("BL5 chart rebase changed generalized material history")
        if self._compensated_coupled or owner._bl5_shell_material or (getattr(owner,'_curved_material',False) and
                                       owner._definition.shell.j2_curve is not None):
            for saved, repeated in zip(_shell_material_arrays(self._shell_origins),
                                       _shell_material_arrays(shell_repeated)):
                scale = max(float(np.max(np.abs(saved), initial=0.)),
                            float(np.max(np.abs(repeated), initial=0.)), np.finfo(float).tiny)
                if np.max(np.abs(saved-repeated), initial=0.) > 64.*np.finfo(float).eps*scale:
                    raise ValueError("BL5 chart rebase changed shell material history")
        if not np.isfinite(relative) or not np.isfinite(gap) or relative > 1e-8 or gap > 1e-9:
            raise ValueError("mechanically inconsistent shell anchor reset")
        support = np.zeros(owner._K.shape[0]); support[owner._fixed] = residual[owner._fixed]
        if owner._coordinate_support:
            support=coordinate_support.reactions(trial.q,trial.shell_anchors,owner._shell_count,owner._fixed,residual,owner._K.shape[0])
        result = _with_shell_chart(replace(self._accepted,
            support_reactions=_owned(support), free_equilibrium_ratio=relative,
            constraint_max=gap, generation=expected_generation+1,
            point_wrenches=self._point_records(trial)), trial, owner._shell_count)
        result = self._external_result(result, trial, self._load_factor)
        next_checkpoint = (None if self._adaptive_state is None and self._programme is None and self._external is None else
                           self._prospective_checkpoint(result, _state=trial))
        cancellation_safe_point(cancellation_token, "stiffened-shell.rebase.commit")
        with owner._commit_lock:
            if owner._generation != expected_generation or self._state is not previous:
                raise ValueError("stale accepted rebase predecessor")
            self._state = trial
            self._accepted = result
            self._adaptive_checkpoint = next_checkpoint
            owner._generation += 1
        return result

    def checkpoint(self, *, _encode_anchors=True, _encode_control=True,
                   _encode_programme=True, _encode_external=True, _encode_compensated=True):
        if self._accepted is None:
            raise RuntimeError("no accepted finite mixed increment")
        if self._compensated and self._programme is not None and _encode_programme:
            from ._stiffened_shell_staged_load_experimental import encode_checkpoint
            return encode_checkpoint(self._programme, self._load_factor,
                self.checkpoint(_encode_programme=False))
        if self._compensated and _encode_compensated and _encode_anchors and _encode_control and _encode_external:
            if (self._external is not None or (self._adaptive_state is not None and
                    not ((supports_paired_fibre_adaptive(self.owner) and self._programme is not None) or
                         (supports_mixed_material_adaptive(self.owner) and self._programme is None)))):
                raise ValueError('durable compensated controller checkpoint is not implemented')
            # A full nodal controller cache already contains its correction envelope.
            if self._adaptive_checkpoint is not None and _encode_programme:
                return self._adaptive_checkpoint
            from ._stiffened_shell_ge_checkpoint_binary import encode_compensated
            state=self._state
            return encode_compensated(self.checkpoint(_encode_compensated=False,
                _encode_programme=_encode_programme),self._checkpoint_identity(),
                {name:getattr(state,name) for name in ('q_low','frames_low','cells_low','resultants_low')})
        if self.owner._finite_distributing and (self._state.shell_anchors is not None or
                self._programme is not None or self._external is not None):
            raise ValueError("finite distributing checkpoint requires nodal force state without chart/programme/external wrappers")
        if (self._adaptive_checkpoint is not None and _encode_anchors and
                _encode_control and _encode_programme and _encode_external and
                (not self._compensated or _encode_compensated)):
            return self._adaptive_checkpoint
        if self._external is not None and _encode_external:
            return self._external.checkpoint(self.checkpoint(
                _encode_anchors=_encode_anchors, _encode_control=_encode_control,
                _encode_programme=_encode_programme, _encode_external=False))
        if self._programme is not None and _encode_programme:
            from ._stiffened_shell_staged_load_experimental import encode_checkpoint
            return encode_checkpoint(self._programme, self._load_factor,
                self.checkpoint(_encode_anchors=_encode_anchors,
                    _encode_control=_encode_control, _encode_programme=False))
        if self._adaptive_state is not None and _encode_control:
            return encode_control(self._checkpoint_identity(), self._adaptive_state,
                                  self.checkpoint(_encode_control=False, _encode_programme=False, _encode_external=False))
        state = self._state
        anchored = state.shell_anchors is not None
        if anchored and _encode_anchors:
            return _encode_chart_checkpoint(
                self._checkpoint_identity(), self.owner._generation, state.shell_anchors,
                self._accepted.shell_frames, self.checkpoint(
                    _encode_anchors=False, _encode_control=False, _encode_programme=False, _encode_external=False))
        generation = len(self._accepted_factors) if anchored else self.owner._generation
        def kind(name):
            if self._stable_q4:
                if name not in ("finite", "finite_material"):
                    raise RuntimeError("stable Q4 checkpoint requires point mechanics")
                name += "_stable_q4"
            return name.replace("finite", "finite_chart", 1) if anchored else name
        arrays = (("q", state.q), ("frames", state.frames),
                  ("cells", state.cells), ("resultants", state.resultants),
                  ("multipliers", state.multipliers),
                  ("pattern" if self._programme is None else "applied_dead_load",
                   self._pattern if self._programme is None else self._applied_load),
                  ("accepted_factors", np.asarray(self._accepted_factors)))
        from ._stiffened_shell_affine_endpoint import is_native_point_material
        if (self._compensated and
                is_native_point_material(self.owner)):
            material = ()
            if self._compensated_coupled:
                plastic, alpha = _shell_material_arrays(self._shell_origins)
                origin_plastic, origin_alpha = _shell_material_arrays(self._shell_predecessors)
                material = (("shell_plastic", plastic), ("shell_alpha", alpha),
                    ("shell_origin_plastic", origin_plastic), ("shell_origin_alpha", origin_alpha))
            return _encode_binary_checkpoint(kind(native_compensated_kind(self.owner._beam_parts,
                    coupled=self._compensated_coupled)),
                self._checkpoint_identity(),generation,arrays+
                native_history_arrays(self.owner._beam_parts,self._beam_predecessors,'origin')+
                native_history_arrays(self.owner._beam_parts,self._beam_origins,'accepted')+material)
        if self.owner._finite_distributing and not self.owner._distributing_material:
            return _encode_binary_checkpoint("finite_distributing", self._checkpoint_identity(), generation, arrays)
        if self.owner._distributing_native:
            owner = self.owner
            material = (native_history_arrays(owner._beam_parts, self._beam_predecessors, "origin") +
                        native_history_arrays(owner._beam_parts, self._beam_origins, "accepted"))
            if owner._distributing_shell_material:
                plastic, alpha = _shell_material_arrays(self._shell_origins)
                origin_plastic, origin_alpha = _shell_material_arrays(self._shell_predecessors)
                material += (("shell_plastic", plastic), ("shell_alpha", alpha),
                             ("shell_origin_plastic", origin_plastic),
                             ("shell_origin_alpha", origin_alpha))
            return _encode_binary_checkpoint(owner._distributing_checkpoint_kind,
                self._checkpoint_identity(), generation, arrays + material)
        if self.owner._bl5_material or self.owner._distributing_material or (self.owner._curved_material and self._curved_material_predecessors):
            plastic, accumulated = _material_arrays(self._beam_origins)
            origin_plastic, origin_accumulated = _material_arrays(self._beam_predecessors)
            material = (("beam_plastic", plastic), ("beam_accumulated", accumulated),
                        ("beam_origin_plastic", origin_plastic),
                        ("beam_origin_accumulated", origin_accumulated))
            coupled = self.owner._bl5_shell_material or self.owner._distributing_shell_material or (self.owner._curved_material and
                self.owner._definition.shell.j2_curve is not None)
            if coupled:
                shell_plastic, shell_alpha = _shell_material_arrays(self._shell_origins)
                parent_plastic, parent_alpha = _shell_material_arrays(self._shell_predecessors)
                material += (("shell_plastic", shell_plastic), ("shell_alpha", shell_alpha),
                             ("shell_origin_plastic", parent_plastic),
                             ("shell_origin_alpha", parent_alpha))
            checkpoint_kind = (("finite_curved_points_coupled" if coupled else
                "finite_curved_points_material") if self.owner._curved_material else
                "finite_bl5_coupled" if coupled else "finite_bl5_material")
            if self.owner._distributing_material:
                checkpoint_kind = self.owner._distributing_checkpoint_kind
            return _encode_binary_checkpoint(kind(checkpoint_kind),
                self._checkpoint_identity(), generation, arrays + material)
        if getattr(self.owner, "_finite_bl5", False) and not self.owner._bl5_native_fibre:
            return _encode_binary_checkpoint(kind("finite_bl5"),
                self._checkpoint_identity(), generation, arrays)
        generalized_stable = (self._stable_s3 and
            supports_generalized_stable_s3(self.owner))
        if (self._stable_s3 and not generalized_stable and not (
                self.owner._has_physical_fibre or self.owner._footprint_stable_s3
                or self._external_stable_s3 or self.owner._coordinate_support
                or self.owner._curved_s3)):
            # A late staged/external context change must not silently publish
            # stable-live state through the legacy arithmetic branch.
            raise ValueError("stable S3 generalized state requires its unchanged native point domain")
        if (self.owner._has_physical_fibre or self.owner._footprint_stable_s3 or self._external_stable_s3 or
                generalized_stable):
            if self.owner._definition.shell.j2_curve is not None:
                owner = self.owner
                q4_count = len(owner._definition.shell.q4_elements)
                plastic, alpha = _shell_material_arrays(self._shell_origins[:q4_count])
                origin_plastic, origin_alpha = _shell_material_arrays(
                    self._shell_predecessors[:q4_count])
                material = owner._shell_model.get_material("shell")
                states = tuple(element.serialize_nonlinear_state(
                    owner._shell_model.mesh, material, origin, 3)
                    for origins in (self._shell_origins, self._shell_predecessors)
                    for (_eid, element, _slots, _), origin in zip(
                        owner._shell_parts[q4_count:], origins[q4_count:]))
                return _encode_s3_checkpoint(self._checkpoint_identity(), generation,
                    arrays + native_history_arrays(owner._beam_parts, self._beam_predecessors, "origin") +
                    native_history_arrays(owner._beam_parts, self._beam_origins, "accepted") + (
                        ("shell_plastic", plastic), ("shell_alpha", alpha),
                        ("shell_origin_plastic", origin_plastic),
                        ("shell_origin_alpha", origin_alpha),
                        ("shell_origin_q", self._shell_predecessor_q),
                        ("shell_origin_anchors", self._shell_predecessor_anchors)),
                    states, anchored=anchored, native_material=True)
            return _encode_binary_checkpoint(kind("finite_native_material_stable"
                if self._stable_s3 else "finite_native_material"),
                self._checkpoint_identity(), generation, arrays +
                native_history_arrays(self.owner._beam_parts,
                                      self._beam_predecessors, "origin") +
                native_history_arrays(self.owner._beam_parts,
                                      self._beam_origins, "accepted"))
        nonmonotone = any(b <= a for a, b in zip(
            self._accepted_factors, self._accepted_factors[1:]))
        if (self.owner._definition.shell.j2_curve is not None and
                self.owner._definition.shell.s3_elements):
            q4_count = len(self.owner._definition.shell.q4_elements)
            beam_plastic, beam_accumulated = _material_arrays(self._beam_origins)
            shell_plastic, shell_alpha = _shell_material_arrays(
                self._shell_origins[:q4_count])
            material = self.owner._shell_model.get_material("shell")
            s3_states = tuple(element.serialize_nonlinear_state(
                self.owner._shell_model.mesh, material, origin, 3)
                for (_eid, element, _slots, _), origin in zip(
                    self.owner._shell_parts[q4_count:],
                    self._shell_origins[q4_count:]))
            return _encode_s3_checkpoint(
                self._checkpoint_identity(), generation,
                arrays + (("beam_plastic", beam_plastic),
                          ("beam_accumulated", beam_accumulated),
                          ("shell_plastic", shell_plastic),
                          ("shell_alpha", shell_alpha)), s3_states, anchored=anchored)
        if self.owner._definition.shell.j2_curve is not None:
            beam_plastic, beam_accumulated = _material_arrays(self._beam_origins)
            shell_plastic, shell_alpha = _shell_material_arrays(self._shell_origins)
            return _encode_binary_checkpoint(
                kind("finite_shell_material"), self._checkpoint_identity(),
                generation,
                arrays + (("beam_plastic", beam_plastic),
                          ("beam_accumulated", beam_accumulated),
                          ("shell_plastic", shell_plastic),
                          ("shell_alpha", shell_alpha)))
        if (nonmonotone or any(station.accumulated != (0., 0.)
               for history in self._beam_origins
               for station in history.stations)):
            plastic, accumulated = _material_arrays(self._beam_origins)
            return _encode_binary_checkpoint(
                kind("finite_material"), self._checkpoint_identity(),
                generation,
                arrays + (("beam_plastic", plastic),
                          ("beam_accumulated", accumulated)))
        if self._stable_q4 or anchored or sum(value.size for _name, value in arrays) > MAX_JSON_CHECKPOINT_VALUES:
            return _encode_binary_checkpoint(
                kind("finite"), self._checkpoint_identity(), generation, arrays)
        payload = {
            "schema": "B3_GE_E4_PL_STIFFENED_SHELL_EXPERIMENTAL_FINITE_V1",
            "model": self._checkpoint_identity(),
            "generation": self.owner._generation,
            "accepted_factors": self._accepted_factors,
            "q": state.q.tolist(),
            "frames": state.frames.tolist(),
            "cells": state.cells.tolist(),
            "resultants": state.resultants.tolist(),
            "multipliers": state.multipliers.tolist(),
            ("pattern" if self._programme is None else "applied_dead_load"):
                (self._pattern if self._programme is None else self._applied_load).tolist(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("ascii")
        if len(raw) > MAX_JSON_CHECKPOINT_BYTES:
            return _encode_binary_checkpoint(
                "finite", self._checkpoint_identity(), self.owner._generation, arrays)
        return raw

    def restore(self, value, *, rotation_anchors=None, pose_witness=None,
                accepted_generation=None, adaptive_state=None, stable_q4=False, corrections=None):
        """Restore an accepted endpoint after mechanical consistency checks.

        V3 checks material admissibility and re-evaluation from the saved
        accepted material state at this endpoint. It does not prove that bytes
        came from the recorded loading path; that needs trusted provenance or
        an independent replay of the accepted history.
        """
        owner = self.owner
        if owner._coordinate_support and (adaptive_state is not None or self._external is not None or self._programme is not None):
            raise ValueError('coordinate shell supports exclude controller checkpoint contexts')
        if self._accepted is not None:
            raise ValueError("cannot replace a live finite driver numerical policy")
        if bool(owner._finite_distributing) != (value.get("schema") in tuple(
                _BINARY_SCHEMAS[k] for k in ("finite_distributing",) + FINITE_DISTRIBUTING_MATERIAL_KINDS)):
            raise ValueError("matching finite distributing checkpoint law required")
        if owner._finite_distributing and value.get("schema") != _BINARY_SCHEMAS[owner._distributing_checkpoint_kind]:
            raise ValueError("matching finite distributing material checkpoint policy required")
        if owner._distributing_cyclic and adaptive_state is not None:
            raise ValueError("cyclic distributing material checkpoints exclude adaptive state")
        if owner._finite_distributing and (rotation_anchors is not None or
                self._programme is not None or self._external is not None):
            raise ValueError("finite distributing checkpoint excludes chart/programme/external state")
        if type(stable_q4) is not bool or (stable_q4 and not self._stable_q4):
            raise ValueError("stable Q4 checkpoint requires its admitted point model")
        # Old checkpoints keep their original arithmetic during validation and
        # continuation. This driver is private until restoration fully succeeds.
        self._stable_q4 = stable_q4
        load_key = "pattern" if self._programme is None else "applied_dead_load"
        base_keys = {"schema", "model", "generation", "accepted_factors",
                     "q", "frames", "cells", "resultants", "multipliers",
                     load_key}
        if type(value) is not dict:
            raise ValueError("incompatible experimental finite checkpoint")
        if ((value.get("schema") in (_BINARY_SCHEMAS["finite_bl5"],
                _BINARY_SCHEMAS["finite_bl5_material"], _BINARY_SCHEMAS["finite_bl5_coupled"])) != bool(owner._finite_bl5 and not owner._bl5_native_fibre) or
                owner._bl5_material != (value.get("schema") in (_BINARY_SCHEMAS["finite_bl5_material"], _BINARY_SCHEMAS["finite_bl5_coupled"])) or
                owner._bl5_shell_material != (value.get("schema") == _BINARY_SCHEMAS["finite_bl5_coupled"])):
            raise ValueError("finite BL5 checkpoint requires its matching line policy")
        curved_material = value.get("schema") in tuple(_BINARY_SCHEMAS[k] for k in CURVED_POINT_MATERIAL_KINDS)
        curved_shell = value.get("schema") == _BINARY_SCHEMAS["finite_curved_points_coupled"]
        distributing_native = value.get("schema") in tuple(_BINARY_SCHEMAS[k] for k in FINITE_DISTRIBUTING_NATIVE_KINDS)
        distributing_material = (not distributing_native and value.get("schema") in
            tuple(_BINARY_SCHEMAS[k] for k in FINITE_DISTRIBUTING_MATERIAL_KINDS))
        distributing_shell = value.get("schema") in (
            _BINARY_SCHEMAS["finite_distributing_coupled"],
            _BINARY_SCHEMAS["finite_distributing_cyclic_coupled"],
            _BINARY_SCHEMAS["finite_distributing_cyclic_native_coupled"])
        if curved_material and (not owner._curved_material or
                curved_shell != (owner._definition.shell.j2_curve is not None)):
            raise ValueError("curved material predecessor checkpoint requires its matching policy")
        native_shell = value.get("schema") == NATIVE_SHELL_SCHEMA
        # The curved-S3 arithmetic is selected by its distinct model identity;
        # ordinary finite checkpoints must not switch it off on restore.
        stable_s3 = (value.get("schema") == STABLE_NATIVE_SCHEMA or
            (self._compensated and self._stable_s3 and value.get('schema') in tuple(_BINARY_SCHEMAS[k] for k in
                ('finite_compensated_generalized_material', 'finite_compensated_fibre_material', 'finite_compensated_mixed_material', 'finite_compensated_generalized_coupled', 'finite_compensated_fibre_coupled', 'finite_compensated_mixed_coupled'))) or owner._curved_s3 or
            (owner._coordinate_support and bool(owner._definition.shell.s3_elements)))
        if (owner._footprint_stable_s3 or self._external_stable_s3) and not stable_s3:
            raise ValueError("nodal footprint S3 checkpoint requires its stable numerical policy")
        if stable_s3 and not self._stable_s3:
            raise ValueError("stable S3 checkpoint requires its admitted elastic policy")
        # This driver is still private to restore until all validation passes.
        self._stable_s3 = stable_s3
        compensated_material=value.get('schema') in tuple(_BINARY_SCHEMAS[k] for k in
            ('finite_compensated_generalized_material', 'finite_compensated_fibre_material', 'finite_compensated_mixed_material', 'finite_compensated_generalized_coupled', 'finite_compensated_fibre_coupled', 'finite_compensated_mixed_coupled'))
        if self._compensated and not compensated_material:
            raise ValueError('compensated native state requires its exact material schema')
        if compensated_material:
            from ._stiffened_shell_affine_endpoint import is_native_point_material
            if (not self._compensated or not is_native_point_material(owner) or value.get('schema') !=
                    _BINARY_SCHEMAS[native_compensated_kind(owner._beam_parts, coupled=self._compensated_coupled)]):
                raise ValueError('exact compensated native point material domain required')
        native_material = (value.get("schema") in (NATIVE_MATERIAL_CHECKPOINT, STABLE_NATIVE_SCHEMA) or
            native_shell or distributing_native or compensated_material)
        s3_material = (value.get("schema") == FINITE_S3_MATERIAL_CHECKPOINT or
                       (native_shell and bool(owner._definition.shell.s3_elements)))
        shell_material = self._compensated_coupled or value.get("schema") in (
            FINITE_SHELL_MATERIAL_CHECKPOINT, FINITE_S3_MATERIAL_CHECKPOINT, NATIVE_SHELL_SCHEMA, _BINARY_SCHEMAS["finite_bl5_coupled"], _BINARY_SCHEMAS["finite_curved_points_coupled"], _BINARY_SCHEMAS["finite_distributing_coupled"], _BINARY_SCHEMAS["finite_distributing_cyclic_coupled"], _BINARY_SCHEMAS["finite_distributing_cyclic_native_coupled"])
        material = value.get("schema") in (
            _BINARY_SCHEMAS["finite_bl5_material"], _BINARY_SCHEMAS["finite_bl5_coupled"],
            FINITE_MATERIAL_CHECKPOINT, FINITE_SHELL_MATERIAL_CHECKPOINT,
            FINITE_S3_MATERIAL_CHECKPOINT) or curved_material or distributing_material
        expected_keys = (base_keys | {"beam_plastic", "beam_accumulated"}
                         if material else base_keys)
        if owner._bl5_material or curved_material or distributing_material:
            expected_keys |= {"beam_origin_plastic", "beam_origin_accumulated"}
        if self._compensated_coupled or owner._bl5_shell_material or curved_shell or distributing_shell:
            expected_keys |= {"shell_origin_plastic", "shell_origin_alpha"}
        if native_material:
            expected_keys |= {name for prefix in ("origin", "accepted")
                for name, _ in native_history_shapes(owner._beam_parts, prefix)}
        if shell_material:
            expected_keys |= {"shell_plastic", "shell_alpha"}
        if s3_material:
            expected_keys |= {"s3_states"}
        if native_shell:
            expected_keys |= {"shell_origin_plastic", "shell_origin_alpha",
                              "shell_origin_q", "shell_origin_anchors", "s3_states"}
        # The selected polishing candidate restores only its own tagged
        # checkpoints, and legacy bytes restore only without the selection;
        # the two directions refuse explicitly instead of silently mixing.
        polish_selected = getattr(self, "_admissible_polish", False) is True
        expected_model = (admissible_polish_identity(owner._identity)
                          if polish_selected else owner._identity)
        if value["model"] != expected_model:
            if polish_selected:
                raise ValueError(
                    "admissible polish candidate requires its tagged finite checkpoint")
            if value["model"] == admissible_polish_identity(owner._identity):
                raise ValueError(
                    "admissible polish checkpoint requires its selected candidate")
            raise ValueError("incompatible experimental finite checkpoint")
        if (set(value) != expected_keys or
                value["schema"] not in
                (_BINARY_SCHEMAS["finite_compensated_generalized_material"], _BINARY_SCHEMAS["finite_compensated_fibre_material"], _BINARY_SCHEMAS["finite_compensated_mixed_material"], _BINARY_SCHEMAS["finite_compensated_generalized_coupled"], _BINARY_SCHEMAS["finite_compensated_fibre_coupled"], _BINARY_SCHEMAS["finite_compensated_mixed_coupled"], *tuple(_BINARY_SCHEMAS[k] for k in CURVED_POINT_MATERIAL_KINDS + FINITE_DISTRIBUTING_MATERIAL_KINDS), _BINARY_SCHEMAS["finite_distributing"], FINITE_CHECKPOINT, _BINARY_SCHEMAS["finite_bl5"], _BINARY_SCHEMAS["finite_bl5_material"], _BINARY_SCHEMAS["finite_bl5_coupled"], NATIVE_MATERIAL_CHECKPOINT, STABLE_NATIVE_SCHEMA, NATIVE_SHELL_SCHEMA, FINITE_MATERIAL_CHECKPOINT,
                 FINITE_SHELL_MATERIAL_CHECKPOINT,
                 FINITE_S3_MATERIAL_CHECKPOINT) or
                type(value["generation"]) is not int or
                value["generation"] < 1 or
                shell_material != (owner._definition.shell.j2_curve is not None) or
                s3_material != bool(owner._definition.shell.j2_curve is not None
                                    and owner._definition.shell.s3_elements)):
            raise ValueError("incompatible experimental finite checkpoint")
        factors = value["accepted_factors"]
        if owner._finite_distributing and not owner._distributing_cyclic and (type(factors) is not list or
                any(type(v) is not float or not np.isfinite(v) for v in factors) or
                any(b <= a for a, b in zip((0.,) + tuple(factors), factors))):
            raise ValueError("finite distributing checkpoint requires monotone positive force history")
        if (type(factors) is not list or len(factors) != value["generation"] or
                not factors or any(type(v) is not float or not np.isfinite(v)
                                   for v in factors) or
                (not material and not native_material and not getattr(owner,"_finite_bl5",False) and (factors[0] <= 0. or
                    any(b <= a for a, b in zip(factors, factors[1:])))) or
                ((material or native_material or getattr(owner,"_finite_bl5",False)) and not owner._distributing_cyclic and (factors[0] <= 0. or
                    any(b <= 0. or b == a for a, b in
                        zip(factors, factors[1:]))))):
            raise ValueError("invalid accepted finite load history")
        if owner._distributing_cyclic and any(a == b for a, b in zip((0.,) + tuple(factors), factors)):
            raise ValueError("cyclic checkpoint force factors must change at each increment")
        q = _owned(value["q"], (owner._K.shape[0],))
        if rotation_anchors is not None:
            rotation_anchors = owned_anchors(rotation_anchors, len(owner._shell_ids))
            witness = owned_anchors(pose_witness, len(owner._shell_ids))
            if (type(accepted_generation) is not int or
                    accepted_generation < value["generation"] or
                    (accepted_generation == value["generation"] and not owner._footprint_increment_chart)):
                raise ValueError("shell chart generation is incompatible with its load-step count and chart policy")
            if not np.allclose(physical_frames(q[:owner._shell_count], rotation_anchors),
                               witness, rtol=0., atol=1e-11):
                raise ValueError("shell chart anchor disagrees with physical pose witness")
            fixed = set(owner._fixed)
            for i, anchor in enumerate(rotation_anchors):
                if (not owner._coordinate_support and any(6*i+j in fixed for j in (3, 4, 5)) and
                        np.linalg.norm(anchor-np.eye(3)) > 1e-12):
                    raise ValueError("anchored partial or fixed shell support is unsupported")
        elif pose_witness is not None or accepted_generation is not None:
            raise ValueError("complete owned shell chart snapshot required")
        frames = _owned(value["frames"], self._reference_frames.shape)
        cells = _owned(value["cells"], self._state.cells.shape)
        resultants = _owned(value["resultants"], self._state.resultants.shape)
        multipliers = _owned(value["multipliers"], self._state.multipliers.shape)
        pattern = _owned(value[load_key], (owner._K.shape[0],))
        if owner._distributing_cyclic and not np.array_equal(pattern, owner._cyclic_reference_pattern):
            raise ValueError("cyclic checkpoint force pattern disagrees with model-bound reference actions")
        if self._external is not None and hasattr(self._external, 'evaluate_at'):
            self._external.validate_targets(0., tuple(factors))
            if np.any(pattern):
                raise ValueError('distributed programme checkpoint cannot contain a nodal pattern')
        if self._programme is not None:
            self._programme.validate_targets(0., tuple(factors))
            if not np.array_equal(pattern, self._programme.load_at(factors[-1])):
                raise ValueError("staged checkpoint applied force disagrees with programme")
        if native_material:
            # The plain elastic S3 generalized point owner admits exactly its
            # own stable native schema; every other native checkpoint still
            # requires the fibre/footprint/external model as before.
            generalized_stable = (stable_s3 and value.get("schema") == STABLE_NATIVE_SCHEMA
                and supports_generalized_stable_s3(owner))
            if (not compensated_material and not owner._has_physical_fibre
                    and not owner._footprint_stable_s3 and not self._external_stable_s3
                    and not generalized_stable):
                raise ValueError("native material checkpoint requires fibre model")
            beam_origins = native_histories(owner._beam_parts, value, "origin")
            beam_accepted = native_histories(owner._beam_parts, value, "accepted")
        elif material:
            stations = sum(len(part[0].operator.cell.stations)
                           for part in owner._beam_parts)
            plastic = _owned(value["beam_plastic"], (stations, 6, 2))
            accumulated = _owned(value["beam_accumulated"], (stations, 2))
            beam_origins = _material_histories(owner, plastic, accumulated)
            if owner._bl5_material or curved_material or distributing_material:
                beam_accepted = beam_origins
                origin_plastic = _owned(value["beam_origin_plastic"], (stations, 6, 2))
                origin_accumulated = _owned(value["beam_origin_accumulated"], (stations, 2))
                beam_origins = _material_histories(owner, origin_plastic, origin_accumulated)
            if not owner._bl5_material and not curved_material and not distributing_material and not shell_material and not np.any(accumulated) and all(
                    b > a for a, b in zip(factors, factors[1:])):
                raise ValueError("material checkpoint requires yielded or reversed history")
        else:
            beam_origins = self._beam_origins
        if owner._finite_elastic_footprint or (owner._finite_distributing and not owner._distributing_material):
            _require_elastic_footprint_histories(beam_origins, "checkpoint origin")
            if native_material:
                _require_elastic_footprint_histories(beam_accepted, "checkpoint accepted")
        if shell_material:
            q4_count = len(owner._definition.shell.q4_elements)
            shell_count = q4_count if s3_material else len(owner._shell_parts)
            shell_plastic = _owned(value["shell_plastic"], (shell_count, 12, 3))
            shell_alpha = _owned(value["shell_alpha"], (shell_count, 12))
            if np.any(shell_alpha < 0.):
                raise ValueError("nonphysical shell J2 checkpoint state")
            shell_origins = _shell_material_states(shell_plastic, shell_alpha)
            if s3_material and not native_shell:
                blobs = value["s3_states"]
                if (type(blobs) is not tuple or
                        len(blobs) != len(owner._definition.shell.s3_elements)):
                    raise ValueError("complete native S3 checkpoint states required")
                shell_material_def = owner._shell_model.get_material("shell")
                s3_origins = []
                for (_eid, element, slots, _), blob in zip(
                        owner._shell_parts[q4_count:], blobs):
                    anchors = None if rotation_anchors is None else rotation_anchors[slots[::6]//6]
                    local_q = s3_local_coordinates(owner._shell_model, element, q[slots],
                                                  rotation_anchors=anchors)
                    restored = element.deserialize_nonlinear_state(
                        owner._shell_model.mesh, shell_material_def, blob, 3,
                        expected_committed_total_u=local_q)
                    _require_s3_material_state(restored, local_q, accepted=True)
                    s3_origins.append(restored)
                shell_origins += tuple(s3_origins)
        else:
            shell_origins = self._shell_origins
        shell_accepted = shell_origins
        if self._compensated_coupled or owner._bl5_shell_material or curved_shell or distributing_shell:
            origin_plastic = _owned(value["shell_origin_plastic"], (q4_count, 12, 3))
            origin_alpha = _owned(value["shell_origin_alpha"], (q4_count, 12))
            if np.any(origin_alpha < 0.):
                raise ValueError("nonphysical shell J2 predecessor")
            shell_origins = _shell_material_states(origin_plastic, origin_alpha)
        if native_shell:
            origin_plastic = _owned(value["shell_origin_plastic"], (q4_count, 12, 3))
            origin_alpha = _owned(value["shell_origin_alpha"], (q4_count, 12))
            if np.any(origin_alpha < 0.):
                raise ValueError("nonphysical shell J2 predecessor")
            accepted_s3, predecessor_s3, origin_q, origin_anchors = _restore_native_s3_histories(
                owner, q, rotation_anchors, value)
            shell_accepted += accepted_s3
            shell_origins = _shell_material_states(origin_plastic, origin_alpha) + predecessor_s3
        beam_nodal = q[owner._shell_count:
                       owner._shell_count+self._beam_nodal].reshape(-1, 6)
        fixed_layout=owner._fixed
        if owner._coordinate_support:
            fixed_layout=np.array([s for s in owner._fixed if s>=owner._shell_count or s%6<3],dtype=int)
            if coordinate_support.support_gap(q,rotation_anchors,owner._shell_count,owner._fixed)>1e-9:
                raise ValueError('checkpoint violates original shell Log support equations')
        if (np.any(q[fixed_layout] != 0.) or
                np.any(beam_nodal[:, 3:] != 0.) or
                np.any(q[owner._shell_count+self._beam_nodal:] != 0.) or
                np.any(pattern[owner._shell_count+self._beam_nodal:] != 0.) or
                np.any(pattern[:owner._shell_count+self._beam_nodal]
                       .reshape(-1, 6)[:, 3:] != 0.)):
            raise ValueError("invalid finite checkpoint coordinate or load layout")
        for family in (frames.reshape(-1, 3, 3), cells.reshape(-1, 3, 3)):
            if any(np.linalg.norm(frame.T@frame-np.eye(3)) > 1e-9 or
                   abs(np.linalg.det(frame)-1.) > 1e-9 for frame in family):
                raise ValueError("nonphysical finite checkpoint rotation")
        trial = _State(q, frames, cells, resultants, multipliers, rotation_anchors)
        if self._compensated:
            if corrections is None: raise ValueError('durable compensated checkpoint corrections required')
            from ._stiffened_shell_f06_increment_arithmetic import validate_pairs,validate_frames
            if set(corrections)!= {'q_low','frames_low','cells_low','resultants_low'}:
                raise ValueError('complete compensated state required')
            owned={}
            for name,high in (('q_low',q),('frames_low',frames),('cells_low',cells),('resultants_low',resultants)):
                low=np.array(corrections[name],dtype=float,copy=True)
                if low.shape!=high.shape: raise ValueError('compensated state shape mismatch')
                (validate_frames if name in ('frames_low','cells_low') else validate_pairs)(high,low,name)
                owned[name]=low
            beam_start=owner._shell_count
            translated=set(beam_start+6*n+i for n in range(len(owner._beam_ids)) for i in range(3))
            if any(v!=0. for i,v in enumerate(owned['q_low']) if i not in translated):
                raise ValueError('only native translation coordinate corrections are supported')
            trial=replace(trial,**owned)
        elif corrections is not None: raise ValueError('compensated checkpoint requires selected arithmetic policy')
        residual, _, relative, gap, shell_next, beam_next = self._assemble(
            trial, factors[-1], shell_origins, beam_origins,
            pattern=pattern if self._programme is None else None,
            applied_load=None if self._programme is None else pattern)
        if (not np.isfinite(relative) or relative > 1e-8 or gap > 1e-9 or
                (not material and not native_material and any(station.accumulated != (0., 0.)
                    for history in beam_next for station in history.stations))):
            raise ValueError("mechanically inconsistent finite checkpoint")
        if native_material and beam_next != beam_accepted:
            raise ValueError("native material checkpoint update disagrees with accepted history")
        if native_shell and s3_material:
            _require_native_s3_replay(owner, trial, shell_accepted, shell_next)
        if material:
            next_plastic, next_accumulated = _material_arrays(beam_next)
            for saved, repeated in ((plastic, next_plastic),
                                    (accumulated, next_accumulated)):
                scale = max(float(np.max(np.abs(saved), initial=0.)),
                            float(np.max(np.abs(repeated), initial=0.)),
                            np.finfo(float).tiny)
                # Paired Decimal updates can alter only low components when
                # an equilibrium state is re-evaluated from its own origin.
                if np.max(np.abs(saved-repeated), initial=0.) > (
                        64.*np.finfo(float).eps*scale):
                    raise ValueError("mechanically inconsistent material checkpoint")
        if shell_material:
            next_plastic, next_alpha = _shell_material_arrays(
                shell_next[:q4_count])
            for saved, repeated in ((shell_plastic, next_plastic),
                                    (shell_alpha, next_alpha)):
                scale = max(float(np.max(np.abs(saved), initial=0.)),
                            float(np.max(np.abs(repeated), initial=0.)),
                            np.finfo(float).tiny)
                if np.max(np.abs(saved-repeated), initial=0.) > (
                        64.*np.finfo(float).eps*scale):
                    raise ValueError("mechanically inconsistent shell J2 checkpoint")
            if s3_material:
                for saved_state, repeated_state in zip(
                        shell_accepted[q4_count:], shell_next[q4_count:]):
                    for name in ("plastic_strain", "alpha"):
                        saved = np.asarray(saved_state[name], dtype=float)
                        repeated = np.asarray(repeated_state[name], dtype=float)
                        scale = max(float(np.max(np.abs(saved), initial=0.)),
                                    float(np.max(np.abs(repeated), initial=0.)),
                                    np.finfo(float).tiny)
                        if np.max(np.abs(saved-repeated), initial=0.) > (
                                64.*np.finfo(float).eps*scale):
                            raise ValueError(
                                "mechanically inconsistent S3 J2 checkpoint")
        support = np.zeros(owner._K.shape[0])
        support[owner._fixed] = residual[owner._fixed]
        if owner._coordinate_support:
            support=coordinate_support.reactions(trial.q,trial.shell_anchors,owner._shell_count,owner._fixed,residual,owner._K.shape[0])
        result = FinitePointResult(
            "completed", factors[-1], _owned(q[:owner._shell_count]),
            _owned(self._beam_positions(trial)), frames, cells, resultants,
            multipliers, _owned(support), relative, gap, 0,
            value["generation"] if accepted_generation is None else accepted_generation, tuple(factors),
            _material_observables(beam_accepted if native_material or owner._bl5_material or curved_material or distributing_material else beam_origins),
            (tuple(tuple(float(v) for v in state["alpha"])
                   for state in shell_accepted) if shell_material else ()),
            (_shell_q4_recovery(owner._shell_parts[:q4_count],
                                shell_next[:q4_count])
             if shell_material else ()),
            (_shell_s3_recovery(owner._shell_parts[q4_count:],
                                shell_next[q4_count:])
             if s3_material else ()))
        result = _with_shell_chart(result, trial, owner._shell_count)
        result = replace(result, point_wrenches=self._point_records(trial),
            beam_generalized_material=(self._generalized_records(trial,beam_origins,
                beam_accepted if native_material or owner._bl5_material or curved_material or distributing_material else beam_next)
                if not owner._curved_material or curved_material else ()))
        if owner._finite_distributing:
            result = replace(result, distributing_footprint_wrenches=
                self._distributing_records(trial, value["generation"]))
        if getattr(owner, "_finite_bl5", False):
            result = replace(result, beam_shell_line_work=self._line_trial_recovery[1])
        if native_material:
            fibre_records, fibre_energies = _fibre_recovery(
                owner, trial, beam_origins, beam_accepted)
            result = replace(result, beam_fibre_material=fibre_records,
                             beam_fibre_energy=fibre_energies)
        if adaptive_state is not None:
            result = replace(result,
                status="completed" if adaptive_state.completed else "accepted_prefix",
                increment_attempts=adaptive_state.attempts)
        result = self._programme_result(result, self._programme, factors[-1], pattern)
        result = self._external_result(result, trial, factors[-1])
        if self._external is not None and np.any(pattern):
            raise ValueError("body checkpoint contains an undeclared nodal load")
        self._adaptive_state = adaptive_state
        self._state = trial
        self._pattern = pattern if self._programme is None else None
        self._applied_load = None if self._programme is None else pattern
        self._shell_predecessors = shell_origins
        if native_shell:
            self._shell_predecessor_q = origin_q
            self._shell_predecessor_anchors = origin_anchors
        self._shell_origins = shell_accepted if shell_material else shell_next
        self._beam_predecessors = beam_origins
        self._beam_origins = beam_accepted if native_material or owner._bl5_material or curved_material or distributing_material else beam_origins if material else beam_next
        self._curved_material_predecessors = curved_material
        self._load_factor = factors[-1]
        self._accepted_factors = tuple(factors)
        self._accepted = result
        owner._generation = result.generation
        return result
