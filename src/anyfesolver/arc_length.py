"""Bounded Crisfield-style arc-length continuation for nonlinear static analysis.

This module follows one proportional reference load through the first limit
point.  It reuses the production nonlinear element response, constraint
transformation and committed material-state machinery from
:mod:`anyfesolver.nonlinear_static`.

Scope is deliberately limited to the ANYfeSolver capacity workflow:

* one proportional load pattern plus an optional constant preload,
* constrained models (no nonlinear free-free nullspace solve),
* geometric and material nonlinearity already supported by the elements,
* continuation only far enough beyond the peak to confirm the descending
  branch.

The equilibrium equations are

    R(q, lambda) = F_constant + lambda F_reference - F_internal(q) = 0

with the spherical constraint

    dq.T W dq + alpha**2 dlambda**2 = ds**2.

Newton corrections use block elimination.  The tangent is factorized once per
iteration and solved for two right-hand sides instead of assembling a generally
nonsymmetric bordered matrix.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .assembly import build_constraint_transformation
from .cases import make_result_case
from .constraint_audit import constraint_residual_summary
from .control import CancellationToken, ProgressCallback, cancellation_safe_point, emit_progress
from .current_state_tangent import (
    require_exact_qualified_component_lifecycle_api as _EXACT_QUALIFIED_LIFECYCLE_GUARD,
)
from .element_capabilities import (
    require_model_element_capabilities,
    require_model_nonlinear_workflow_capabilities,
)
from .linalg import MatrixClass, factorize
from .matrix_assembly import (
    _run_with_qualified_assembly_runtime_lease,
    assemble_load_vector,
    assemble_stiffness_matrix,
)
from .nonlinear_analysis_diagnostics import capture_nonlinear_analysis_diagnostics
from .nonlinear_state import NonlinearStateStore, _run_with_nonlinear_state_cleanup
from .nonlinear_restart import (
    canonical_checkpoint_json_bytes,
    create_nonlinear_checkpoint,
    load_case_descriptor,
    load_nonlinear_checkpoint,
    validate_nonlinear_checkpoint,
)
from .nonlinear_static import (
    NonlinearIncrementSnapshot,
    _assemble_nonlinear_system,
    _activate_nonlinear_state_storage,
    _commit_nonlinear_state_candidate,
    _copy_initial_states,
    _discard_nonlinear_state_candidate,
    _has_follower_pressure,
    _max_plastic_strain,
    _materialize_final_nonlinear_states,
    _mark_failed_qualified_q4_states,
    _nonlinear_state_summary,
    _owned_imperfection_input,
    _owned_initial_element_states,
    _prepare_qualified_q4_states_for_nonlinear_solve,
    _seal_final_qualified_q4_states,
    _support_reaction_dof_plan,
    _support_reaction_resultants_from_forces,
    _increment_snapshot,
    _weighted_external_load_system,
    _solve_static_nonlinear_under_lease,
)
from .recovery import ResourceConfig, _owned_resource_config_snapshot
from .threading_policy import resource_threaded

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


_SMALL = 1.0e-14


def _arc_predictor_reuse_exclusion(
    *, model, load_cases, contact, follower_active, kinematics, cancellation_token,
    initial_state_supplied=False,
):
    """Keep retry retention out of mutable contributor/callback lifecycles."""
    if contact is not None:
        return "contact_contributor"
    if follower_active:
        return "follower_load"
    if kinematics != "von_karman":
        return "kinematics"
    if cancellation_token is not None:
        # Tokens may override raise_if_cancelled and mutate unversioned input.
        return "cancellation_callback"
    from .boundary import LoadCase
    from .e4_pl_element import QualifiedE4PLShellElement
    from .fe_core import Material

    if any(case is not None and type(case) is not LoadCase for case in load_cases):
        return "custom_load_case"
    if initial_state_supplied:
        return "initial_material_history"
    # This exact class is covered by the full lifecycle guard. Generic
    # element hooks may intentionally depend on observations beyond q/state.
    if not model.mesh.elements or any(
        type(element) is not QualifiedE4PLShellElement
        for element in model.mesh.elements.values()
    ):
        return "unqualified_element_lifecycle"
    # Do not invoke custom constitutive accessors during acceleration admission.
    # Existing runtime guards still validate the exact model/material lifecycle.
    if not isinstance(model.materials, dict):
        return "custom_material_mapping"
    for element in model.mesh.elements.values():
        material = dict.get(model.materials, element.material_name)
        if type(material) is not Material:
            return "noncanonical_elastic_material"
        fields = object.__getattribute__(material, "__dict__")
        yield_stress = dict.get(fields, "yield_stress")
        if (dict.get(fields, "hardening_curve") is not None
                or (type(yield_stress) is not int and type(yield_stress) is not float)
                or yield_stress != 0.0):
            return "plastic_material"
    return None


@dataclass(frozen=True)
class ArcLengthControl:
    """Controls for bounded spherical arc-length continuation.

    ``initial_load_increment`` is used only to construct the first arc radius
    from the initial tangent direction.  Thereafter the radius is adapted in
    path space, not forced to produce a particular load increment.
    """

    initial_load_increment: float = 0.05
    minimum_load_increment: float = 5.0e-4
    maximum_load_increment: float = 0.20
    load_scaling: Optional[float] = None
    rotation_length_scale: Optional[float] = None
    target_iterations: int = 5
    growth_factor: float = 1.25
    cutback_factor: float = 0.5
    max_steps: int = 100
    max_retries_per_step: int = 8
    stop_after_peak_steps: int = 4
    peak_drop_tolerance: float = 1.0e-3
    maximum_absolute_load_factor: Optional[float] = None
    preload_steps: int = 10
    # Post-buckling continuation controls.  When ``post_peak_load_fraction``
    # is set the trace continues past the limit point and stops automatically
    # once the load factor has fallen to that fraction of the recorded peak
    # (set ``stop_after_peak_steps`` high to allow the descending branch).
    # ``max_translation`` is an absolute displacement guard in metres on the
    # largest nodal translation, protecting against runaway post-peak paths.
    post_peak_load_fraction: Optional[float] = None
    max_translation: Optional[float] = None

    def __post_init__(self) -> None:
        if self.initial_load_increment <= 0.0:
            raise ValueError("initial_load_increment must be positive")
        if self.minimum_load_increment <= 0.0:
            raise ValueError("minimum_load_increment must be positive")
        if self.maximum_load_increment < self.initial_load_increment:
            raise ValueError("maximum_load_increment must be at least initial_load_increment")
        if self.minimum_load_increment > self.initial_load_increment:
            raise ValueError("minimum_load_increment must not exceed initial_load_increment")
        if self.load_scaling is not None and self.load_scaling <= 0.0:
            raise ValueError("load_scaling must be positive when supplied")
        if self.rotation_length_scale is not None and self.rotation_length_scale <= 0.0:
            raise ValueError("rotation_length_scale must be positive when supplied")
        if self.target_iterations <= 0:
            raise ValueError("target_iterations must be positive")
        if self.growth_factor < 1.0:
            raise ValueError("growth_factor must be at least 1.0")
        if not (0.0 < self.cutback_factor < 1.0):
            raise ValueError("cutback_factor must be between 0 and 1")
        if self.max_steps <= 0 or self.max_retries_per_step <= 0:
            raise ValueError("max_steps and max_retries_per_step must be positive")
        if self.stop_after_peak_steps <= 0:
            raise ValueError("stop_after_peak_steps must be positive")
        if self.peak_drop_tolerance < 0.0:
            raise ValueError("peak_drop_tolerance must be non-negative")
        if self.maximum_absolute_load_factor is not None and self.maximum_absolute_load_factor <= 0.0:
            raise ValueError("maximum_absolute_load_factor must be positive when supplied")
        if self.preload_steps <= 0:
            raise ValueError("preload_steps must be positive")
        if self.post_peak_load_fraction is not None and not (0.0 < self.post_peak_load_fraction < 1.0):
            raise ValueError("post_peak_load_fraction must be in (0, 1) when supplied")
        if self.max_translation is not None and self.max_translation <= 0.0:
            raise ValueError("max_translation must be positive when supplied")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_load_increment": float(self.initial_load_increment),
            "minimum_load_increment": float(self.minimum_load_increment),
            "maximum_load_increment": float(self.maximum_load_increment),
            "load_scaling": self.load_scaling,
            "rotation_length_scale": self.rotation_length_scale,
            "target_iterations": int(self.target_iterations),
            "growth_factor": float(self.growth_factor),
            "cutback_factor": float(self.cutback_factor),
            "max_steps": int(self.max_steps),
            "max_retries_per_step": int(self.max_retries_per_step),
            "stop_after_peak_steps": int(self.stop_after_peak_steps),
            "peak_drop_tolerance": float(self.peak_drop_tolerance),
            "maximum_absolute_load_factor": self.maximum_absolute_load_factor,
            "preload_steps": int(self.preload_steps),
            "post_peak_load_fraction": self.post_peak_load_fraction,
            "max_translation": self.max_translation,
        }


def _owned_arc_length_control(
    model: "FEModel",
    control: Optional[ArcLengthControl],
    *,
    _exact_guard: Any,
) -> ArcLengthControl:
    """Detach one caller control before any model or load observation."""

    source = ArcLengthControl() if control is None else control

    def observed(name: str) -> Any:
        value = getattr(source, name)
        _exact_guard(model, context=f"arc-length control {name} observation")
        return value

    def converted(name: str, converter: Any) -> Any:
        value = converter(observed(name))
        _exact_guard(model, context=f"arc-length control {name} conversion")
        return value

    def optional_float(name: str) -> Optional[float]:
        value = observed(name)
        if value is None:
            return None
        made = float(value)
        _exact_guard(model, context=f"arc-length control {name} conversion")
        return made

    owned = ArcLengthControl(
        initial_load_increment=converted("initial_load_increment", float),
        minimum_load_increment=converted("minimum_load_increment", float),
        maximum_load_increment=converted("maximum_load_increment", float),
        load_scaling=optional_float("load_scaling"),
        rotation_length_scale=optional_float("rotation_length_scale"),
        target_iterations=converted("target_iterations", int),
        growth_factor=converted("growth_factor", float),
        cutback_factor=converted("cutback_factor", float),
        max_steps=converted("max_steps", int),
        max_retries_per_step=converted("max_retries_per_step", int),
        stop_after_peak_steps=converted("stop_after_peak_steps", int),
        peak_drop_tolerance=converted("peak_drop_tolerance", float),
        maximum_absolute_load_factor=optional_float(
            "maximum_absolute_load_factor"
        ),
        preload_steps=converted("preload_steps", int),
        post_peak_load_fraction=optional_float("post_peak_load_fraction"),
        max_translation=optional_float("max_translation"),
    )
    _exact_guard(model, context="arc-length owned control construction")
    return owned


@dataclass
class ArcLengthStep:
    """One converged point on the equilibrium path."""

    step_index: int
    load_factor: float
    iterations: int
    retries: int
    arc_radius: float
    residual_norm: float
    arc_residual: float
    displacement_norm: float
    load_increment: float
    path_increment_norm: float
    max_equivalent_plastic_strain: float
    is_peak: bool = False
    support_reactions: Dict[str, tuple[float, ...]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "load_factor": float(self.load_factor),
            "iterations": int(self.iterations),
            "retries": int(self.retries),
            "arc_radius": float(self.arc_radius),
            "residual_norm": float(self.residual_norm),
            "arc_residual": float(self.arc_residual),
            "displacement_norm": float(self.displacement_norm),
            "load_increment": float(self.load_increment),
            "path_increment_norm": float(self.path_increment_norm),
            "max_equivalent_plastic_strain": float(self.max_equivalent_plastic_strain),
            "is_peak": bool(self.is_peak),
            "support_reactions": {
                str(name): [float(value) for value in values]
                for name, values in self.support_reactions.items()
            },
        }


@dataclass
class ArcLengthResult:
    """Result from bounded arc-length continuation."""

    steps: List[ArcLengthStep]
    status: str
    displacements: np.ndarray
    load_factor: float
    peak_load_factor: float
    peak_step_index: Optional[int]
    element_states: Dict[int, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    snapshots: tuple[NonlinearIncrementSnapshot, ...] = field(default_factory=tuple)
    restart_checkpoint: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def converged(self) -> bool:
        return self.status in {
            "peak_confirmed",
            "maximum_steps_reached",
            "load_factor_limit_reached",
            "post_buckling_traced",
            "displacement_limit_reached",
        }

    @property
    def capacity_estimate(self) -> float:
        return float(self.peak_load_factor)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "converged": self.converged,
            "load_factor": float(self.load_factor),
            "peak_load_factor": float(self.peak_load_factor),
            "peak_step_index": self.peak_step_index,
            "capacity_estimate": self.capacity_estimate,
            "info": self.info,
            "steps": [step.to_dict() for step in self.steps],
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
        }
        if self.restart_checkpoint is not None:
            payload["restart_checkpoint"] = {
                "schema": self.restart_checkpoint["schema"],
                "version": self.restart_checkpoint["version"],
                "analysis_kind": self.restart_checkpoint["analysis_kind"],
                "checkpoint_sha256": self.restart_checkpoint["checkpoint_sha256"],
            }
        return payload

    def to_contact_archive_bytes(self, *, max_bytes=16 * 1024 * 1024) -> bytes:
        """Export detached contact data; this grants no restart or admission."""
        from .general_contact.arc_results_archive import serialize_arc_contact_results
        return serialize_arc_contact_results(self, max_bytes=max_bytes)

    def to_restart_checkpoint(self) -> Dict[str, Any]:
        if self.restart_checkpoint is None:
            raise ValueError("this result does not contain an arc-length restart checkpoint")
        return copy.deepcopy(self.restart_checkpoint)

    def restart_checkpoint_bytes(self) -> bytes:
        return canonical_checkpoint_json_bytes(self.to_restart_checkpoint())

    @property
    def quantity_metadata(self) -> tuple[Any, ...]:
        from .quantities import describe_result_quantities

        return describe_result_quantities(self)


def _arc_restart_analysis_contract(
    *,
    load_case: Optional["LoadCase"],
    constant_load_case: Optional["LoadCase"],
    settings: ArcLengthControl,
    max_iterations: int,
    tolerance: float,
    arc_tolerance: float,
    num_layers: int,
    resource_config: Optional[ResourceConfig],
    kinematics: str,
    resolved_corotational_tangent: str,
) -> Dict[str, Any]:
    settings_payload = settings.to_dict()
    # max_steps is the number of *additional* accepted path points requested
    # by one invocation.  Every other control remains path-defining.
    settings_payload.pop("max_steps")
    return {
        "schema": "ANYSOLVER_ARC_LENGTH_RESTART_CONTRACT_V1",
        "load_case": load_case_descriptor(load_case),
        "constant_load_case": load_case_descriptor(constant_load_case),
        "control": settings_payload,
        "max_iterations": int(max_iterations),
        "tolerance": float(tolerance),
        "arc_tolerance": float(arc_tolerance),
        "num_layers": int(num_layers),
        "resource_config": (
            None if resource_config is None else resource_config.to_dict()
        ),
        "kinematics": str(kinematics),
        "corotational_tangent": str(resolved_corotational_tangent),
    }


def _arc_step_from_restart(value: Any) -> ArcLengthStep:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint arc-length step must be a mapping")
    expected = {
        "step_index",
        "load_factor",
        "iterations",
        "retries",
        "arc_radius",
        "residual_norm",
        "arc_residual",
        "displacement_norm",
        "load_increment",
        "path_increment_norm",
        "max_equivalent_plastic_strain",
        "is_peak",
        "support_reactions",
    }
    if set(value) != expected or not isinstance(value["support_reactions"], Mapping):
        raise ValueError("checkpoint arc-length step schema is incompatible")
    return ArcLengthStep(
        step_index=int(value["step_index"]),
        load_factor=float(value["load_factor"]),
        iterations=int(value["iterations"]),
        retries=int(value["retries"]),
        arc_radius=float(value["arc_radius"]),
        residual_norm=float(value["residual_norm"]),
        arc_residual=float(value["arc_residual"]),
        displacement_norm=float(value["displacement_norm"]),
        load_increment=float(value["load_increment"]),
        path_increment_norm=float(value["path_increment_norm"]),
        max_equivalent_plastic_strain=float(value["max_equivalent_plastic_strain"]),
        is_peak=bool(value["is_peak"]),
        support_reactions={
            str(name): tuple(float(item) for item in components)
            for name, components in value["support_reactions"].items()
        },
    )


def _restore_arc_path_state(
    value: Mapping[str, Any],
    *,
    n_red: Optional[int] = None,
    total_dofs: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("arc-length checkpoint path_state must be a mapping")
    expected = {
        "mode",
        "load_factor",
        "step_index",
        "steps",
        "reduced_coordinates",
        "exact_base_offset",
        "radius",
        "minimum_radius",
        "maximum_radius",
        "initial_radius",
        "previous_dq",
        "previous_dlambda",
        "peak_load_factor",
        "peak_step_index",
        "descending_steps",
        "max_translation",
        "load_scaling",
        "rotation_length_scale",
        "adaptation_history",
        "support_reaction_history",
        "total_iterations",
        "total_retries",
        "corrector_solve_many_count",
        "corrector_tangent_projection_count",
        "reaction_force_reuse_count",
        "reaction_force_reassembly_count",
        "preload_info",
        "terminal_status",
        "failure_reason",
    }
    legacy_expected = expected - {"exact_base_offset"}
    if frozenset(value) not in {frozenset(expected), frozenset(legacy_expected)} or value.get(
        "mode"
    ) != "arc_length":
        raise ValueError("arc-length checkpoint path schema is incompatible")
    steps_raw = value["steps"]
    if not isinstance(steps_raw, list):
        raise ValueError("arc-length checkpoint steps must be a list")
    steps = [_arc_step_from_restart(item) for item in steps_raw]
    step_index = int(value["step_index"])
    if len(steps) != step_index or [step.step_index for step in steps] != list(
        range(1, step_index + 1)
    ):
        raise ValueError("arc-length checkpoint step ordering is inconsistent")
    lam = float(value["load_factor"])
    if steps and steps[-1].load_factor != lam:
        raise ValueError("arc-length checkpoint load factor differs from its last step")
    previous_dq_raw = value["previous_dq"]
    if previous_dq_raw is None:
        previous_dq = None
    else:
        previous_dq = np.asarray(previous_dq_raw, dtype=float)
        if (
            previous_dq.ndim != 1
            or not np.all(np.isfinite(previous_dq))
            or (n_red is not None and previous_dq.shape != (int(n_red),))
        ):
            raise ValueError("arc-length checkpoint predictor direction is incompatible")
    previous_dlambda = (
        None
        if value["previous_dlambda"] is None
        else float(value["previous_dlambda"])
    )
    if (previous_dq is None) != (previous_dlambda is None):
        raise ValueError("arc-length checkpoint predictor state is incomplete")
    reduced_coordinates = np.asarray(value["reduced_coordinates"], dtype=float)
    if (
        reduced_coordinates.ndim != 1
        or not np.all(np.isfinite(reduced_coordinates))
        or (n_red is not None and reduced_coordinates.shape != (int(n_red),))
    ):
        raise ValueError("arc-length checkpoint reduced coordinates are incompatible")
    exact_base_offset_raw = value.get("exact_base_offset")
    if exact_base_offset_raw is None:
        exact_base_offset = None
    else:
        exact_base_offset = np.asarray(exact_base_offset_raw, dtype=np.float64)
        if (
            exact_base_offset.ndim != 1
            or not np.all(np.isfinite(exact_base_offset))
            or (
                total_dofs is not None
                and exact_base_offset.shape != (int(total_dofs),)
            )
        ):
            raise ValueError(
                "arc-length checkpoint exact base offset is incompatible"
            )
    radius = float(value["radius"])
    min_radius = float(value["minimum_radius"])
    max_radius = float(value["maximum_radius"])
    initial_radius = float(value["initial_radius"])
    if min(radius, min_radius, max_radius, initial_radius) <= 0.0:
        raise ValueError("arc-length checkpoint radii must be positive")
    if min_radius > max_radius:
        raise ValueError("arc-length checkpoint radius bounds are inconsistent")
    peak_step_index = value["peak_step_index"]
    peak_step_index = None if peak_step_index is None else int(peak_step_index)
    if peak_step_index is not None and not (1 <= peak_step_index <= step_index):
        raise ValueError("arc-length checkpoint peak step index is invalid")
    list_names = ("adaptation_history", "support_reaction_history")
    if any(not isinstance(value[name], list) for name in list_names):
        raise ValueError("arc-length checkpoint diagnostic histories must be lists")
    if len(value["support_reaction_history"]) != step_index:
        raise ValueError("arc-length checkpoint reaction history is incomplete")
    if len(value["adaptation_history"]) < step_index:
        raise ValueError("arc-length checkpoint adaptation history is incomplete")
    count_names = (
        "total_iterations",
        "total_retries",
        "corrector_solve_many_count",
        "corrector_tangent_projection_count",
        "reaction_force_reuse_count",
        "reaction_force_reassembly_count",
    )
    counts = {name: int(value[name]) for name in count_names}
    if any(item < 0 for item in counts.values()):
        raise ValueError("arc-length checkpoint counters must be non-negative")
    return {
        **copy.deepcopy(dict(value)),
        **counts,
        "load_factor": lam,
        "step_index": step_index,
        "steps": steps,
        "reduced_coordinates": reduced_coordinates.copy(),
        "exact_base_offset": (
            None if exact_base_offset is None else exact_base_offset.copy()
        ),
        "radius": radius,
        "minimum_radius": min_radius,
        "maximum_radius": max_radius,
        "initial_radius": initial_radius,
        "previous_dq": None if previous_dq is None else previous_dq.copy(),
        "previous_dlambda": previous_dlambda,
        "peak_step_index": peak_step_index,
        "terminal_status": str(value["terminal_status"]),
        "failure_reason": (
            None if value["failure_reason"] is None else str(value["failure_reason"])
        ),
    }


def _arc_restart_path_payload(
    *,
    load_factor: float,
    steps: Sequence[ArcLengthStep],
    reduced_coordinates: np.ndarray,
    exact_base_offset: np.ndarray,
    radius: float,
    minimum_radius: float,
    maximum_radius: float,
    initial_radius: float,
    previous_dq: Optional[np.ndarray],
    previous_dlambda: Optional[float],
    peak_load_factor: float,
    peak_step_index: Optional[int],
    descending_steps: int,
    max_translation: float,
    load_scaling: float,
    rotation_length_scale: float,
    adaptation_history: Sequence[Mapping[str, Any]],
    support_reaction_history: Sequence[Mapping[str, Any]],
    total_iterations: int,
    total_retries: int,
    corrector_solve_many_count: int,
    corrector_tangent_projection_count: int,
    reaction_force_reuse_count: int,
    reaction_force_reassembly_count: int,
    preload_info: Any,
    terminal_status: str,
    failure_reason: Optional[str],
) -> Dict[str, Any]:
    return {
        "mode": "arc_length",
        "load_factor": float(load_factor),
        "step_index": len(steps),
        "steps": [step.to_dict() for step in steps],
        "reduced_coordinates": np.asarray(
            reduced_coordinates, dtype=float
        ).reshape(-1).tolist(),
        "exact_base_offset": np.asarray(
            exact_base_offset, dtype=np.float64
        ).reshape(-1).tolist(),
        "radius": float(radius),
        "minimum_radius": float(minimum_radius),
        "maximum_radius": float(maximum_radius),
        "initial_radius": float(initial_radius),
        "previous_dq": None if previous_dq is None else previous_dq.tolist(),
        "previous_dlambda": previous_dlambda,
        "peak_load_factor": float(peak_load_factor),
        "peak_step_index": peak_step_index,
        "descending_steps": int(descending_steps),
        "max_translation": float(max_translation),
        "load_scaling": float(load_scaling),
        "rotation_length_scale": float(rotation_length_scale),
        "adaptation_history": copy.deepcopy(list(adaptation_history)),
        "support_reaction_history": copy.deepcopy(list(support_reaction_history)),
        "total_iterations": int(total_iterations),
        "total_retries": int(total_retries),
        "corrector_solve_many_count": int(corrector_solve_many_count),
        "corrector_tangent_projection_count": int(corrector_tangent_projection_count),
        "reaction_force_reuse_count": int(reaction_force_reuse_count),
        "reaction_force_reassembly_count": int(reaction_force_reassembly_count),
        "preload_info": copy.deepcopy(preload_info),
        "terminal_status": str(terminal_status),
        "failure_reason": None if failure_reason is None else str(failure_reason),
    }


def _arc_preload_summary(preload: Any) -> Dict[str, Any]:
    """Return the deterministic, restart-relevant constant-preload record.

    ``NonlinearStaticResult.info`` deliberately contains runtime diagnostics
    (including element-id keyed timing maps).  Those diagnostics are useful
    to the caller but are neither canonical JSON nor part of the arc path.
    The committed preload displacement and material state are already bound
    independently by the nonlinear checkpoint.  Keep only the stable result
    facts needed to identify and audit how that base equilibrium was reached.
    """

    steps = [step.to_dict() for step in preload.steps]
    return {
        "schema": "ANYSOLVER_ARC_LENGTH_PRELOAD_SUMMARY_V1",
        "status": str(preload.status),
        "converged": bool(preload.converged),
        "load_factor": float(preload.load_factor),
        "peak_load_factor": float(preload.peak_load_factor),
        "last_converged_load_factor": float(
            preload.last_converged_load_factor
        ),
        "failure_reason": (
            None
            if preload.failure_reason is None
            else str(preload.failure_reason)
        ),
        "stop_reason": (
            None if preload.stop_reason is None else str(preload.stop_reason)
        ),
        "step_count": len(steps),
        "total_iterations": sum(int(step.iterations) for step in preload.steps),
        "steps": steps,
        "element_state_ids": sorted(
            int(element_id) for element_id in preload.element_states
        ),
    }


def _characteristic_length(model: "FEModel") -> float:
    coords = np.asarray(model.mesh.get_node_coordinates(), dtype=float)
    if coords.size == 0:
        return 1.0
    spans = np.ptp(coords, axis=0)
    value = float(np.max(spans))
    return value if value > _SMALL else 1.0


def _full_metric_weights(model: "FEModel", rotation_length_scale: float) -> np.ndarray:
    """Diagonal translation-equivalent weights in full DOF space."""

    total_dofs = model.mesh.dof_manager.total_dofs
    weights = np.ones(total_dofs, dtype=float)
    rotation_weight = float(rotation_length_scale) ** 2
    for dof in range(total_dofs):
        _node_id, local_index, _name = model.mesh.dof_manager.get_dof_info(dof)
        if local_index >= 3:
            weights[dof] = rotation_weight
    return weights


def _reduced_metric(model: "FEModel", T: sparse.spmatrix, rotation_length_scale: float) -> sparse.csr_matrix:
    """Project a translation-equivalent full-DOF metric to reduced coordinates."""

    weights = _full_metric_weights(model, rotation_length_scale)
    W_full = sparse.diags(weights, format="csr")
    return (T.T @ W_full @ T).tocsr()


def _metric_dot(W: sparse.spmatrix, left: np.ndarray, right: np.ndarray) -> float:
    return float(np.asarray(left, dtype=float) @ np.asarray(W @ right, dtype=float))


def _metric_norm(W: sparse.spmatrix, vector: np.ndarray) -> float:
    return float(np.sqrt(max(_metric_dot(W, vector, vector), 0.0)))


def _factorized_solve(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    signature: str,
    *,
    matrix_class: MatrixClass = MatrixClass.SYMMETRIC_INDEFINITE,
) -> np.ndarray:
    handle = factorize(matrix, matrix_class, signature=signature)
    solution = np.asarray(handle.solve(np.asarray(rhs, dtype=float)), dtype=float).reshape(-1)
    if np.any(~np.isfinite(solution)):
        raise np.linalg.LinAlgError("non-finite tangent solution")
    return solution


def _recover_reduced_coordinates(T: sparse.spmatrix, u0: np.ndarray, displacements: np.ndarray) -> np.ndarray:
    rhs = np.asarray(displacements, dtype=float).reshape(-1) - np.asarray(u0, dtype=float).reshape(-1)
    result = sparse_linalg.lsqr(T, rhs, atol=1.0e-12, btol=1.0e-12)
    q = np.asarray(result[0], dtype=float).reshape(-1)
    mismatch = np.asarray(T @ q + u0 - displacements, dtype=float).reshape(-1)
    scale = max(float(np.linalg.norm(displacements)), 1.0)
    if float(np.linalg.norm(mismatch)) > 1.0e-8 * scale:
        raise RuntimeError("could not recover reduced coordinates from preloaded displacement state")
    return q


def _copy_model_with_imperfection(model: "FEModel", imperfection: Optional[Any]) -> "FEModel":
    if imperfection is None:
        return model
    from .imperfections import apply_imperfection

    return apply_imperfection(model, imperfection, copy_model=True)


def _max_nodal_translation(model: "FEModel", displacements: np.ndarray) -> float:
    """Largest nodal translation magnitude in the displacement vector."""
    peak = 0.0
    size = int(displacements.size)
    for node in model.mesh.nodes.values():
        dofs = np.asarray(node.dofs[:3], dtype=np.intp)
        if dofs.size == 0 or int(dofs.max()) >= size:
            continue
        value = float(np.linalg.norm(displacements[dofs]))
        if value > peak:
            peak = value
    return peak


def _finalize_arc_element_states(
    model: "FEModel",
    displacements: np.ndarray,
    element_states: Mapping[int, Any],
    num_layers: int,
    info: Dict[str, Any],
    *,
    kinematics: str,
) -> Dict[int, Any]:
    """Recover installed result caches, then close every qualified Q4 seal."""

    materialized = _materialize_final_nonlinear_states(element_states, info)
    # Resolve this extension dynamically. Batch B installs its elastic layer-
    # state recovery here and that recovered payload must be inside the seal.
    from . import nonlinear_static as _nonlinear_static

    finalized = _nonlinear_static._finalize_nonlinear_element_states(
        model,
        displacements,
        materialized,
        num_layers,
        kinematics=kinematics,
    )
    return _seal_final_qualified_q4_states(
        model,
        displacements,
        finalized,
        num_layers,
        info,
        kinematics=kinematics,
    )


def _finalize_failed_arc_element_states(
    model: "FEModel",
    displacements: np.ndarray,
    element_states: Mapping[int, Any],
    num_layers: int,
    info: Dict[str, Any],
    *,
    kinematics: str,
    failure_reason: str,
) -> Dict[int, Any]:
    """Materialize a failed arc boundary without granting ACTIVE authority."""

    materialized = _materialize_final_nonlinear_states(element_states, info)
    # A failed boundary has no accepted candidate from which displacement-
    # derived recovery may be authored.  Preserve the exact last committed
    # materialized state and attach only the nonauthoritative disposition.
    return _mark_failed_qualified_q4_states(
        model,
        displacements,
        materialized,
        num_layers,
        info,
        failure_reason=str(failure_reason),
        kinematics=kinematics,
    )


@resource_threaded
@capture_nonlinear_analysis_diagnostics
def _solve_static_arc_length_under_lease(
    model: "FEModel",
    load_case: Optional["LoadCase"],
    *,
    constant_load_case: Optional["LoadCase"] = None,
    control: Optional[ArcLengthControl] = None,
    max_iterations: int = 25,
    tolerance: float = 1.0e-6,
    arc_tolerance: float = 1.0e-6,
    num_layers: int = 5,
    imperfection: Optional[Any] = None,
    initial_element_states: Optional[Mapping[int, Any]] = None,
    kinematics: str = "von_karman",
    corotational_tangent: str = "auto",
    progress_callback: Optional[ProgressCallback] = None,
    resource_config: Optional[ResourceConfig] = None,
    cancellation_token: Optional[CancellationToken] = None,
    record_increment_snapshots: bool = False,
    restart_checkpoint: Optional[Any] = None,
    emit_restart_checkpoint: bool = False,
    _qualified_runtime_guard: Any = None,
    _arc_contact: Any = None,
) -> ArcLengthResult:
    """Trace the first nonlinear limit point with spherical arc-length control.

    The optional ``constant_load_case`` is first brought to equilibrium using
    the existing adaptive force-control solver.  Arc-length continuation then
    scales only ``load_case``.  Material states are committed only after a full
    equilibrium-plus-constraint convergence.

    A non-zero prescribed support/MPC field is proportional to the same path
    factor.  It can be the sole continuation driver, so displacement-driven
    models do not need a fictitious nodal load.

    Current-area follower pressure contributes both its displacement-dependent
    force and exact load tangent.  Corotational ``"auto"`` tangent selection
    follows :func:`anyfesolver.solve_static_nonlinear`: it selects the consistent
    chain-rule tangent for follower pressure and the rotated approximation
    otherwise.
    """
    raw_exact_guard = _EXACT_QUALIFIED_LIFECYCLE_GUARD
    lease_model = model

    def exact_guard(
        observed_model: "FEModel",
        *,
        context: str,
    ) -> Dict[str, Any]:
        result = raw_exact_guard(observed_model, context=context)
        # The optional imperfection path uses a solver-owned model copy.  Keep
        # the original non-renewable generation lease while validating the
        # active copy through the exact lifecycle guard.
        _qualified_runtime_guard(lease_model, context=context)
        return result

    exact_guard(model, context="arc-length solve preflight")
    cancellation_safe_point(cancellation_token, "arc_length.start")
    exact_guard(model, context="arc-length start cancellation")

    def owned_scalar(value: Any, converter: Any, name: str) -> Any:
        made = converter(value)
        exact_guard(model, context=f"arc-length {name} conversion")
        return made

    max_iterations = owned_scalar(max_iterations, int, "max_iterations")
    tolerance = owned_scalar(tolerance, float, "tolerance")
    arc_tolerance = owned_scalar(arc_tolerance, float, "arc_tolerance")
    num_layers = owned_scalar(num_layers, int, "num_layers")
    kinematics = owned_scalar(kinematics, str, "kinematics").strip().lower()
    exact_guard(model, context="arc-length kinematics normalization")
    corotational_tangent = owned_scalar(
        corotational_tangent,
        str,
        "corotational_tangent",
    )
    record_increment_snapshots = owned_scalar(
        record_increment_snapshots,
        bool,
        "record_increment_snapshots",
    )
    emit_restart_checkpoint = owned_scalar(
        emit_restart_checkpoint,
        bool,
        "emit_restart_checkpoint",
    )
    settings = _owned_arc_length_control(
        model,
        control,
        _exact_guard=exact_guard,
    )
    imperfection = _owned_imperfection_input(
        model,
        imperfection,
        _exact_guard=exact_guard,
    )
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0.0 or arc_tolerance <= 0.0:
        raise ValueError("tolerances must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    if kinematics not in {"von_karman", "corotational"}:
        raise ValueError("kinematics must be 'von_karman' or 'corotational'")
    parsed_restart_checkpoint: Optional[Dict[str, Any]] = None
    if restart_checkpoint is not None:
        emit_restart_checkpoint = True
        if initial_element_states is not None:
            raise ValueError(
                "restart_checkpoint cannot be combined with initial_element_states"
            )
        parsed_restart_checkpoint = load_nonlinear_checkpoint(
            restart_checkpoint,
            _exact_guard=exact_guard,
            _guard_model=model,
        )
        require_model_element_capabilities(
            model,
            "arc_length_restart_history",
            context="solve_static_arc_length",
        )
    elif emit_restart_checkpoint:
        require_model_element_capabilities(
            model,
            "arc_length_restart_history",
            context="solve_static_arc_length",
        )
    initial_element_states = _owned_initial_element_states(
        model,
        initial_element_states,
        _exact_guard=exact_guard,
    )
    if imperfection is not None:
        require_model_element_capabilities(
            model,
            "initial_fields",
            context="solve_static_arc_length",
        )
    if initial_element_states is not None:
        require_model_element_capabilities(
            model,
            "arc_length_restart_history",
            context="solve_static_arc_length",
            element_ids=initial_element_states,
        )
    require_model_nonlinear_workflow_capabilities(
        model,
        context="solve_static_arc_length",
    )
    follower_active = _has_follower_pressure(
        load_case,
        model=model,
        _exact_guard=exact_guard,
    ) or _has_follower_pressure(
        constant_load_case,
        model=model,
        _exact_guard=exact_guard,
    )
    from .corotational import (
        resolve_corotational_tangent_mode,
        validate_corotational_scope,
    )

    resolved_corotational_tangent = resolve_corotational_tangent_mode(
        kinematics,
        corotational_tangent,
        follower_pressure=follower_active,
    )
    if kinematics == "corotational":
        validate_corotational_scope(model)
        if (
            follower_active
            and resolved_corotational_tangent != "consistent"
        ):
            raise NotImplementedError(
                "Follower pressure with corotational kinematics requires "
                "corotational_tangent='consistent'."
            )
    general_tangent = _arc_contact is not None or follower_active or (
        kinematics == "corotational"
        and resolved_corotational_tangent == "consistent"
    )
    start_time = time.time()
    working_model = _copy_model_with_imperfection(model, imperfection)
    exact_guard(working_model, context="arc-length imperfection observation")
    if _arc_contact is not None and _arc_contact.manager.model is not working_model:
        raise ValueError("arc contact contributor must bind the actual working model")
    restart_analysis_contract = _arc_restart_analysis_contract(
        load_case=load_case,
        constant_load_case=constant_load_case,
        settings=settings,
        max_iterations=max_iterations,
        tolerance=tolerance,
        arc_tolerance=arc_tolerance,
        num_layers=num_layers,
        resource_config=resource_config,
        kinematics=kinematics,
        resolved_corotational_tangent=resolved_corotational_tangent,
    )
    if _arc_contact is not None:
        restart_analysis_contract["schema"] = ("ANYSOLVER_ARC_CONTACT_RESTART_CONTRACT_PRELOAD_V2"
            if _arc_contact.preload_enabled else "ANYSOLVER_ARC_CONTACT_RESTART_CONTRACT_V1")
        restart_analysis_contract["general_contact"] = _arc_contact.restart_contract()
    validated_restart = None
    restored_arc_path: Optional[Dict[str, Any]] = None
    if parsed_restart_checkpoint is not None:
        validated_restart = validate_nonlinear_checkpoint(
            parsed_restart_checkpoint,
            analysis_kind="arc_length",
            model=working_model,
            analysis_contract=restart_analysis_contract,
            num_layers=num_layers,
        )
        exact_guard(working_model, context="arc-length checkpoint validation")
        contact_path = dict(validated_restart.path_state)
        if _arc_contact is not None:
            if "contact_state" not in contact_path:
                raise ValueError("arc contact checkpoint history missing")
            _arc_contact.restore(contact_path.pop("contact_state"), contact_path,
                                 validated_restart.displacements)
        restored_arc_path = _restore_arc_path_state(
            contact_path,
            total_dofs=int(working_model.mesh.dof_manager.total_dofs),
        )
        if restored_arc_path["terminal_status"] != "maximum_steps_reached":
            raise ValueError(
                "arc-length checkpoint does not end at a continuable maximum-step boundary"
            )
        if validated_restart.deleted_element_ids:
            raise ValueError("arc-length checkpoints cannot contain deletion state")
        if validated_restart.activity is not None:
            working_model.set_element_activity(validated_restart.activity)
            exact_guard(working_model, context="arc-length activity restoration")
        initial_element_states = validated_restart.element_states
    working_model.apply_boundary_conditions()
    exact_guard(working_model, context="arc-length boundary conditions")

    K0, stiffness_info = assemble_stiffness_matrix(working_model)
    exact_guard(working_model, context="arc-length stiffness assembly")
    F_prop, load_info = assemble_load_vector(working_model, load_case)
    exact_guard(working_model, context="arc-length proportional-load assembly")
    if constant_load_case is None:
        F_const = np.zeros_like(F_prop)
        constant_load_info = None
    else:
        F_const, constant_load_info = assemble_load_vector(working_model, constant_load_case)
        exact_guard(working_model, context="arc-length constant-load assembly")

    _, _, T, u0, _, constraint_info = build_constraint_transformation(K0, F_prop, working_model)
    exact_guard(working_model, context="arc-length constraint transformation")
    prescribed_offset = np.asarray(u0, dtype=float).reshape(-1)
    zero_prescribed_offset = not bool(np.any(prescribed_offset))
    prescribed_path_active = bool(
        prescribed_offset.size
        and float(np.max(np.abs(prescribed_offset))) > _SMALL
    )
    n_red = int(T.shape[1])
    if restored_arc_path is not None:
        restored_arc_path = _restore_arc_path_state(
            contact_path,
            n_red=n_red,
            total_dofs=int(working_model.mesh.dof_manager.total_dofs),
        )
    assembly_info = {
        "stiffness": stiffness_info,
        "load": load_info,
        "constant_load": constant_load_info,
        "constraint_info": constraint_info,
        "total_dofs": int(working_model.mesh.dof_manager.total_dofs),
        "reduced_dofs": n_red,
    }

    info: Dict[str, Any] = {
        **assembly_info,
        "control": settings.to_dict(),
        "num_layers": int(num_layers),
        "formulation": "crisfield_spherical_block_elimination",
        "kinematics": kinematics,
        "corotational_tangent_requested": str(corotational_tangent).lower(),
        "corotational_tangent": resolved_corotational_tangent,
        "follower_pressure": follower_active,
        "equilibrium_tangent": "K_internal-K_external" if follower_active else "K_internal",
        "prescribed_displacement_path": {
            "mode": "proportional_to_load_factor",
            "active": prescribed_path_active,
            "target_max_abs": (
                float(np.max(np.abs(prescribed_offset)))
                if prescribed_offset.size else 0.0
            ),
        },
    }
    if validated_restart is not None:
        info["restart"] = {
            "schema": validated_restart.payload["schema"],
            "checkpoint_sha256": validated_restart.payload["checkpoint_sha256"],
            "resumed_load_factor": float(restored_arc_path["load_factor"]),
            "resumed_step_index": int(restored_arc_path["step_index"]),
        }
    if imperfection is not None:
        info["imperfection"] = getattr(working_model, "imperfection_metadata", [])

    committed_states: Dict[int, Any] = _copy_initial_states(initial_element_states)

    if n_red == 0:
        final_u = (
            np.asarray(validated_restart.displacements, dtype=np.float64).copy()
            if validated_restart is not None
            else np.asarray(u0, dtype=np.float64).copy()
        )
        committed_states = _prepare_qualified_q4_states_for_nonlinear_solve(
            working_model,
            final_u,
            committed_states,
            num_layers,
            info,
            supplied_element_ids=tuple(
                sorted(int(value) for value in committed_states)
            ),
            ordinary_restart=validated_restart is not None,
            allow_explicit_initial_material_states=bool(
                validated_restart is None and initial_element_states is not None
            ),
        )
        committed_states = _finalize_arc_element_states(
            working_model,
            final_u,
            committed_states,
            num_layers,
            info,
            kinematics=kinematics,
        )
        info["failure_reason"] = "empty_reduced_system"
        return ArcLengthResult(
            [],
            "empty_reduced_system",
            final_u,
            0.0,
            0.0,
            None,
            committed_states,
            info,
        )

    F_prop_red = np.asarray(T.T @ F_prop, dtype=float).reshape(-1)
    F_const_red = np.asarray(T.T @ F_const, dtype=float).reshape(-1)
    if float(np.linalg.norm(F_prop_red)) <= _SMALL and not prescribed_path_active:
        final_u = np.asarray(u0, dtype=np.float64).copy()
        committed_states = _prepare_qualified_q4_states_for_nonlinear_solve(
            working_model,
            final_u,
            committed_states,
            num_layers,
            info,
            supplied_element_ids=tuple(
                sorted(int(value) for value in committed_states)
            ),
            ordinary_restart=False,
            allow_explicit_initial_material_states=bool(
                initial_element_states is not None
            ),
        )
        committed_states = _finalize_arc_element_states(
            working_model,
            final_u,
            committed_states,
            num_layers,
            info,
            kinematics=kinematics,
        )
        info["failure_reason"] = "zero_reduced_reference_load"
        return ArcLengthResult(
            [],
            "zero_reference_load",
            final_u,
            0.0,
            0.0,
            None,
            committed_states,
            info,
        )

    exact_base_offset = np.zeros(
        int(working_model.mesh.dof_manager.total_dofs), dtype=np.float64
    )
    if restored_arc_path is not None:
        lam = float(restored_arc_path["load_factor"])
        q = np.asarray(restored_arc_path["reduced_coordinates"], dtype=float).copy()
        preload_info = copy.deepcopy(restored_arc_path["preload_info"])
        preload_path_info = copy.deepcopy(preload_info)
        stored_base_offset = restored_arc_path.get("exact_base_offset")
        if stored_base_offset is None:
            # Backward-compatible migration of a pre-offset checkpoint. Its
            # independently hash-bound full displacement supplies the exact
            # constant remainder omitted by the older path schema.
            projected = np.asarray(
                T @ q + lam * prescribed_offset, dtype=np.float64
            ).reshape(-1)
            supplied = np.asarray(
                validated_restart.displacements, dtype=np.float64
            ).reshape(-1)
            exact_base_offset = supplied - projected
            for _pass in range(4):
                reconstructed = projected + exact_base_offset
                if np.array_equal(reconstructed, supplied):
                    break
                exact_base_offset += supplied - reconstructed
            if not np.array_equal(projected + exact_base_offset, supplied):
                raise ValueError(
                    "legacy arc-length checkpoint displacement cannot be "
                    "represented bit-exactly"
                )
        else:
            exact_base_offset = np.asarray(
                stored_base_offset, dtype=np.float64
            ).copy()
    else:
        q = np.zeros(n_red, dtype=float)
        lam = 0.0
        preload_info = None
        preload_path_info = None

    if constant_load_case is not None and prescribed_path_active and _arc_contact is None:
        raise NotImplementedError(
            "arc-length continuation does not yet combine a constant preload "
            "with a proportional prescribed-displacement path"
        )
    if (
        restored_arc_path is None
        and constant_load_case is not None
        and (float(np.linalg.norm(F_const_red)) > _SMALL or _arc_contact is not None)
    ):
        preload_solver = _solve_static_nonlinear_under_lease
        preload_extra = {}
        if _arc_contact is not None:
            preload_solver = _arc_contact.preload
            preload_extra = {"summary": _arc_preload_summary}
        preload = preload_solver(
            working_model,
            **preload_extra,
            load_case=constant_load_case,
            max_load_factor=1.0,
            num_steps=settings.preload_steps,
            max_iterations=max_iterations,
            tolerance=tolerance,
            num_layers=num_layers,
            initial_element_states=committed_states,
            kinematics=kinematics,
            corotational_tangent=corotational_tangent,
            cancellation_token=cancellation_token,
            **({} if _arc_contact is not None else {"record_increment_snapshots": record_increment_snapshots}),
            _qualified_runtime_guard=exact_guard,
        )
        exact_guard(working_model, context="arc-length preload result")
        # Retain the established rich runtime diagnostic for a fresh solve,
        # but bind only a stable, canonical summary into the restart path.
        preload_info = preload.to_dict()
        preload_path_info = _arc_preload_summary(preload)
        if preload.status != "completed":
            info["preload"] = preload_info
            info["failure_reason"] = "constant_preload_not_converged"
            return ArcLengthResult(
                [],
                "preload_failed",
                preload.displacements,
                0.0,
                0.0,
                None,
                preload.element_states,
                info,
            )
        q = _recover_reduced_coordinates(T, u0, preload.displacements)
        committed_states = copy.deepcopy(preload.element_states)
        projected_preload = np.asarray(T @ q, dtype=np.float64).reshape(-1)
        supplied_preload = np.asarray(
            preload.displacements, dtype=np.float64
        ).reshape(-1)
        exact_base_offset = supplied_preload - projected_preload
        for _pass in range(4):
            reconstructed = projected_preload + exact_base_offset
            if np.array_equal(reconstructed, supplied_preload):
                break
            exact_base_offset += supplied_preload - reconstructed
        if not np.array_equal(
            projected_preload + exact_base_offset, supplied_preload
        ):
            raise ValueError(
                "constant-preload displacement cannot be represented bit-"
                "exactly on the arc-length base path"
            )

    if zero_prescribed_offset:

        def full_displacement(
            q_reduced: np.ndarray, path_factor: float
        ) -> np.ndarray:
            del path_factor
            return np.asarray(
                T @ q_reduced + exact_base_offset, dtype=float
            ).reshape(-1)

    else:

        def full_displacement(
            q_reduced: np.ndarray, path_factor: float
        ) -> np.ndarray:
            return np.asarray(
                T @ q_reduced
                + exact_base_offset
                + float(path_factor) * prescribed_offset,
                dtype=float,
            ).reshape(-1)

    if restored_arc_path is not None and not np.array_equal(
        full_displacement(q, lam),
        validated_restart.displacements,
    ):
        raise ValueError(
            "arc-length checkpoint reduced coordinates do not exactly reconstruct "
            "its committed displacement"
        )

    initial_committed_displacement = full_displacement(q, lam)
    committed_states = _prepare_qualified_q4_states_for_nonlinear_solve(
        working_model,
        initial_committed_displacement,
        committed_states,
        num_layers,
        info,
        supplied_element_ids=tuple(
            sorted(int(value) for value in committed_states)
        ),
        ordinary_restart=bool(
            restored_arc_path is not None or preload_info is not None
        ),
        allow_explicit_initial_material_states=bool(
            restored_arc_path is None
            and preload_info is None
            and initial_element_states is not None
        ),
    )

    rotation_scale = (
        float(restored_arc_path["rotation_length_scale"])
        if restored_arc_path is not None
        else settings.rotation_length_scale or _characteristic_length(working_model)
    )
    W = _reduced_metric(working_model, T, rotation_scale)
    full_weights = _full_metric_weights(working_model, rotation_scale)
    metric_cross = np.asarray(
        T.T @ (full_weights * prescribed_offset), dtype=float
    ).reshape(-1)
    prescribed_metric = float(
        prescribed_offset @ (full_weights * prescribed_offset)
    )
    zero_load_tangent = sparse.csr_matrix(K0.shape, dtype=float)

    def external_system(
        displacements: np.ndarray,
        load_factor: float,
        *,
        tangent: bool,
    ) -> Tuple[
        np.ndarray,
        Optional[sparse.csr_matrix],
        np.ndarray,
        Optional[sparse.csr_matrix],
    ]:
        if not follower_active:
            return (
                F_const,
                zero_load_tangent if tangent else None,
                F_prop,
                zero_load_tangent if tangent else None,
            )
        constant_force, constant_tangent = _weighted_external_load_system(
            working_model,
            [(constant_load_case, 1.0)],
            displacements,
            tangent=tangent,
        )
        proportional_force, proportional_tangent = _weighted_external_load_system(
            working_model,
            [(load_case, 1.0)],
            displacements,
            tangent=tangent,
        )
        exact_guard(working_model, context="arc-length external-load system")
        return constant_force, constant_tangent, proportional_force, proportional_tangent

    if _arc_contact is not None and _arc_contact.preload_enabled:
        _arc_contact.validate_base_inventory()
        base = _arc_contact.base
        if base['summary'] != preload_path_info:
            raise ValueError('arc preload base differs from structural preload summary')
        # The admitted material profile is elastic: replay every accepted preload
        # pose from the bound material origin without committing a new state.
        for row in base['history']:
            pose, factor = row['displacements'], row['load_factor']
            imposed = factor*prescribed_offset
            try:
                base_q = _recover_reduced_coordinates(T, imposed, pose)
            except RuntimeError as exc:
                raise ValueError('arc preload base violates prescribed constraints') from exc
            guard = 512*np.finfo(float).eps*max(1.,float(np.linalg.norm(pose)))
            if np.linalg.norm(T @ base_q + imposed-pose) > guard:
                raise ValueError('arc preload base violates prescribed constraints')
            base_force, _, base_trial = _assemble_nonlinear_system(working_model,
                pose,base['element_states'],num_layers,tangent=False,
                kinematics=kinematics,corotational_tangent=resolved_corotational_tangent)
            _discard_nonlinear_state_candidate(base['element_states'])
            if np.linalg.norm(T.T @ (factor*F_const-base_force)) > tolerance*max(1.,np.linalg.norm(T.T @ (factor*F_const))):
                raise ValueError('arc preload base is not in equilibrium')

    # Establish the first tangent direction and derive fixed path-space radius
    # limits from the user-facing load-increment settings.
    u = initial_committed_displacement
    committed_states = _activate_nonlinear_state_storage(
        working_model,
        committed_states,
        num_layers,
        info,
        kinematics=kinematics,
        committed_displacements=u,
    )
    F_int, K_T, _trial_states = _assemble_nonlinear_system(
        working_model,
        u,
        committed_states,
        num_layers,
        tangent=True,
        kinematics=kinematics,
        corotational_tangent=resolved_corotational_tangent,
    )
    exact_guard(working_model, context="arc-length initial tangent assembly")
    # The initial tangent is diagnostic at the already committed base.  It is
    # not an accepted continuation increment and must never advance either
    # material history or node-shared multiplicative rotations.
    _discard_nonlinear_state_candidate(committed_states)
    F_const_current, K_const, F_prop_current, K_prop = external_system(u, lam, tangent=True)
    exact_guard(working_model, context="arc-length initial external load")
    F_prop_red = np.asarray(T.T @ F_prop_current, dtype=float).reshape(-1)
    residual0 = (
        np.asarray(T.T @ (F_const_current + lam * F_prop_current), dtype=float).reshape(-1)
        - np.asarray(T.T @ F_int, dtype=float).reshape(-1)
    )
    reference0 = max(
        float(np.linalg.norm(np.asarray(T.T @ (F_const_current + F_prop_current), dtype=float))),
        1.0,
    )
    if float(np.linalg.norm(residual0)) > 10.0 * tolerance * reference0:
        info["failure_reason"] = "initial_state_not_in_equilibrium"
        info["initial_residual_norm"] = float(np.linalg.norm(residual0))
        committed_states = _finalize_failed_arc_element_states(
            working_model,
            u,
            committed_states,
            num_layers,
            info,
            kinematics=kinematics,
            failure_reason="initial_state_not_in_equilibrium",
        )
        return ArcLengthResult(
            [],
            "initial_equilibrium_failed",
            u,
            lam,
            lam,
            None,
            committed_states,
            info,
        )

    K_red = (
        (T.T @ K_T @ T)
        - (T.T @ K_const @ T)
        - lam * (T.T @ K_prop @ T)
    ).tocsr()
    if prescribed_path_active:
        K_effective = K_T - K_const - lam * K_prop
        path_direction_red = np.asarray(
            T.T @ (F_prop_current - K_effective @ prescribed_offset),
            dtype=float,
        ).reshape(-1)
    else:
        path_direction_red = F_prop_red
    if float(np.linalg.norm(path_direction_red)) <= _SMALL:
        info["failure_reason"] = "zero_reduced_reference_load"
        committed_states = _finalize_arc_element_states(
            working_model,
            u,
            committed_states,
            num_layers,
            info,
            kinematics=kinematics,
        )
        return ArcLengthResult(
            [],
            "zero_reference_load",
            u.copy(),
            0.0,
            0.0,
            None,
            committed_states,
            info,
        )
    try:
        tangent_direction = _factorized_solve(
            K_red,
            path_direction_red,
            "arc_length.initial_tangent",
            matrix_class=(
                MatrixClass.GENERAL
                if general_tangent
                else MatrixClass.SYMMETRIC_INDEFINITE
            ),
        )
    except Exception as exc:
        info["failure_reason"] = "initial_tangent_factorization_failed"
        info["factorization_error"] = str(exc)
        committed_states = _finalize_failed_arc_element_states(
            working_model,
            u,
            committed_states,
            num_layers,
            info,
            kinematics=kinematics,
            failure_reason="initial_tangent_factorization_failed",
        )
        return ArcLengthResult(
            [],
            "initial_tangent_failed",
            u,
            lam,
            lam,
            None,
            committed_states,
            info,
        )

    physical_direction_norm_sq = (
        _metric_dot(W, tangent_direction, tangent_direction)
        + 2.0 * float(metric_cross @ tangent_direction)
        + prescribed_metric
    )
    physical_direction_norm = float(
        np.sqrt(max(physical_direction_norm_sq, 0.0))
    )
    load_scaling = (
        float(restored_arc_path["load_scaling"])
        if restored_arc_path is not None
        else (
            float(settings.load_scaling)
            if settings.load_scaling is not None
            else max(physical_direction_norm, 1.0e-12)
        )
    )

    if not zero_prescribed_offset:

        def path_metric_dot(
            dq_left: np.ndarray,
            dlambda_left: float,
            dq_right: np.ndarray,
            dlambda_right: float,
        ) -> float:
            """Full affine-displacement metric plus the load weight."""

            return float(
                _metric_dot(W, dq_left, dq_right)
                + float(dlambda_right) * float(metric_cross @ dq_left)
                + float(dlambda_left) * float(metric_cross @ dq_right)
                + float(dlambda_left)
                * float(dlambda_right)
                * (prescribed_metric + load_scaling * load_scaling)
            )

    else:

        def path_metric_dot(
            dq_left: np.ndarray,
            dlambda_left: float,
            dq_right: np.ndarray,
            dlambda_right: float,
        ) -> float:
            """Mature force-driven metric without zero affine products."""

            return float(
                _metric_dot(W, dq_left, dq_right)
                + float(dlambda_left)
                * float(dlambda_right)
                * load_scaling
                * load_scaling
            )

    predictor_norm = float(
        np.sqrt(
            max(path_metric_dot(tangent_direction, 1.0, tangent_direction, 1.0), 0.0)
        )
    )
    if _arc_contact is not None and restored_arc_path is not None:
        def reduce_contact_pose(pose, path_factor):
            offset = exact_base_offset + float(path_factor)*prescribed_offset
            return _recover_reduced_coordinates(T, offset, pose)
        _arc_contact.validate_path(validated_restart.path_state, reduce_contact_pose,
                                   full_displacement, path_metric_dot, arc_tolerance)
    if restored_arc_path is not None:
        radius = float(restored_arc_path["radius"])
        min_radius = float(restored_arc_path["minimum_radius"])
        max_radius = float(restored_arc_path["maximum_radius"])
        initial_radius = float(restored_arc_path["initial_radius"])
        steps = list(restored_arc_path["steps"])
        previous_dq = (
            None
            if restored_arc_path["previous_dq"] is None
            else np.asarray(restored_arc_path["previous_dq"], dtype=float).copy()
        )
        previous_dlambda = restored_arc_path["previous_dlambda"]
        peak_load_factor = float(restored_arc_path["peak_load_factor"])
        peak_step_index = restored_arc_path["peak_step_index"]
        peak_step = (
            None
            if peak_step_index is None
            else steps[int(peak_step_index) - 1]
        )
        max_translation = float(restored_arc_path["max_translation"])
    else:
        radius = settings.initial_load_increment * predictor_norm
        min_radius = radius * settings.minimum_load_increment / settings.initial_load_increment
        max_radius = radius * settings.maximum_load_increment / settings.initial_load_increment
        initial_radius = float(radius)
        steps: List[ArcLengthStep] = []
        previous_dq: Optional[np.ndarray] = None
        previous_dlambda: Optional[float] = None
        peak_load_factor = float(lam)
        peak_step_index: Optional[int] = None
        peak_step: Optional[ArcLengthStep] = None
        max_translation = 0.0
    track_step_translation = (
        progress_callback is not None or settings.max_translation is not None
    )
    descending_steps = (
        int(restored_arc_path["descending_steps"])
        if restored_arc_path is not None
        else 0
    )
    status = "maximum_steps_reached"
    failure_reason: Optional[str] = None
    total_iterations = (
        int(restored_arc_path["total_iterations"])
        if restored_arc_path is not None
        else 0
    )
    total_retries = (
        int(restored_arc_path["total_retries"])
        if restored_arc_path is not None
        else 0
    )
    corrector_solve_many_count = (
        int(restored_arc_path["corrector_solve_many_count"])
        if restored_arc_path is not None
        else 0
    )
    corrector_tangent_projection_count = (
        int(restored_arc_path["corrector_tangent_projection_count"])
        if restored_arc_path is not None
        else 0
    )
    reaction_force_reuse_count = (
        int(restored_arc_path["reaction_force_reuse_count"])
        if restored_arc_path is not None
        else 0
    )
    reaction_force_reassembly_count = (
        int(restored_arc_path["reaction_force_reassembly_count"])
        if restored_arc_path is not None
        else 0
    )
    adaptation_history: List[Dict[str, Any]] = (
        copy.deepcopy(restored_arc_path["adaptation_history"])
        if restored_arc_path is not None
        else []
    )
    snapshots: List[NonlinearIncrementSnapshot] = []
    support_reaction_history: List[Dict[str, Any]] = (
        copy.deepcopy(restored_arc_path["support_reaction_history"])
        if restored_arc_path is not None
        else []
    )
    support_reaction_dof_plan = _support_reaction_dof_plan(working_model)
    exact_guard(
        working_model,
        context="arc-length support-reaction plan observation",
    )
    corrector_rhs = np.empty((n_red, 2), dtype=float)
    predictor_reuse_exclusion = _arc_predictor_reuse_exclusion(
        model=working_model, load_cases=(load_case, constant_load_case),
        contact=_arc_contact, follower_active=follower_active,
        kinematics=kinematics, cancellation_token=cancellation_token,
        initial_state_supplied=(
            initial_element_states is not None or parsed_restart_checkpoint is not None
        ),
    )
    exact_guard(working_model, context="arc-length predictor eligibility observation")
    predictor_reuse_count = 0
    predictor_solve_count = 0

    step_index_offset = len(steps)
    for local_step_index in range(1, settings.max_steps + 1):
        step_index = step_index_offset + local_step_index
        cancellation_safe_point(
            cancellation_token,
            f"arc_length.step:{step_index}",
        )
        exact_guard(
            working_model,
            context="arc-length step cancellation",
        )
        q_base = q.copy()
        lambda_base = float(lam)
        states_base = (
            committed_states
            if isinstance(committed_states, NonlinearStateStore)
            else copy.deepcopy(committed_states)
        )
        accepted = False
        step_failure = "unknown"
        # The committed base is fixed across radius retries, but never across
        # accepted steps. Retain values, not a backend handle or trial state.
        retry_predictor = None

        for retry in range(settings.max_retries_per_step + 1):
            cancellation_safe_point(
                cancellation_token,
                f"arc_length.step:{step_index}.retry:{retry}",
            )
            exact_guard(
                working_model,
                context="arc-length retry cancellation",
            )
            total_retries += int(retry > 0)
            predictor_revision = (
                working_model.revision_signature()
                if predictor_reuse_exclusion is None else None
            )
            if predictor_reuse_exclusion is None:
                exact_guard(working_model, context="arc-length predictor revision observation")
            if (retry_predictor is not None
                    and retry_predictor[0] == predictor_revision):
                _discard_nonlinear_state_candidate(committed_states)
                load_direction, direction_norm = retry_predictor[1:]
                # Keep lifecycle observations at the same retry boundaries.
                exact_guard(working_model, context="arc-length predictor assembly")
                exact_guard(working_model, context="arc-length predictor load")
                predictor_reuse_count += 1
            else:
                retry_predictor = None
                u_base = full_displacement(q_base, lambda_base)
                F_base, K_base, _ = _assemble_nonlinear_system(
                    working_model,
                    u_base,
                    states_base,
                    num_layers,
                    tangent=True,
                    kinematics=kinematics,
                    corotational_tangent=resolved_corotational_tangent,
                )
                exact_guard(working_model, context="arc-length predictor assembly")
                F_const_base, K_const_base, F_prop_base, K_prop_base = external_system(
                    u_base,
                    lambda_base,
                    tangent=True,
                )
                exact_guard(working_model, context="arc-length predictor load")
                K_base_red = (
                    (T.T @ K_base @ T)
                    - (T.T @ K_const_base @ T)
                    - lambda_base * (T.T @ K_prop_base @ T)
                ).tocsr()
                if prescribed_path_active:
                    K_base_effective = (
                        K_base - K_const_base - lambda_base * K_prop_base
                    )
                    path_direction_base_red = np.asarray(
                        T.T @ (F_prop_base - K_base_effective @ prescribed_offset),
                        dtype=float,
                    ).reshape(-1)
                else:
                    path_direction_base_red = np.asarray(
                        T.T @ F_prop_base, dtype=float
                    ).reshape(-1)
                try:
                    predictor_solve_count += 1
                    load_direction = _factorized_solve(
                        K_base_red,
                        path_direction_base_red,
                        f"arc_length.predictor:{step_index}:{retry}",
                        matrix_class=(
                            MatrixClass.GENERAL
                            if general_tangent
                            else MatrixClass.SYMMETRIC_INDEFINITE
                        ),
                    )
                except Exception:
                    step_failure = "singular_predictor_tangent"
                    _discard_nonlinear_state_candidate(committed_states)
                    radius *= settings.cutback_factor
                    if radius < min_radius:
                        break
                    continue
            sign = 1.0
            if previous_dq is not None and previous_dlambda is not None:
                orientation = path_metric_dot(
                    previous_dq,
                    previous_dlambda,
                    load_direction,
                    1.0,
                )
                sign = 1.0 if orientation >= 0.0 else -1.0

            if retry_predictor is None:
                direction_norm = float(
                    np.sqrt(
                        max(
                            path_metric_dot(
                                load_direction, 1.0, load_direction, 1.0
                            ),
                            0.0,
                        )
                    )
                )
                if predictor_reuse_exclusion is None:
                    retry_predictor = (
                        predictor_revision, load_direction.copy(), direction_norm,
                    )
            if direction_norm <= _SMALL:
                step_failure = "zero_predictor_direction"
                _discard_nonlinear_state_candidate(committed_states)
                break

            dlambda_total = sign * radius / direction_norm
            dq_total = dlambda_total * load_direction
            q_trial = q_base + dq_total
            lambda_trial = lambda_base + dlambda_total
            contact_previous_trial = full_displacement(q_base, lambda_base)
            residual_norm = float("inf")
            arc_residual = float("inf")
            trial_states = states_base

            for iteration in range(1, max_iterations + 1):
                cancellation_safe_point(
                    cancellation_token,
                    f"arc_length.step:{step_index}.iteration:{iteration}",
                )
                exact_guard(
                    working_model,
                    context="arc-length iteration cancellation",
                )
                total_iterations += 1
                u_trial = full_displacement(q_trial, lambda_trial)
                if _arc_contact is not None:
                    origin = full_displacement(q_base, lambda_base)
                    if (not _arc_contact.admissible(origin, u_trial) or
                            not _arc_contact.admissible(contact_previous_trial, u_trial)):
                        step_failure = "uncertified_contact_swept_path"
                        break
                    contact_previous_trial = u_trial.copy()
                F_internal, K_trial, states_candidate = _assemble_nonlinear_system(
                    working_model,
                    u_trial,
                    states_base,
                    num_layers,
                    tangent=True,
                    kinematics=kinematics,
                    corotational_tangent=resolved_corotational_tangent,
                )
                exact_guard(working_model, context="arc-length corrector assembly")
                F_const_trial, K_const_trial, F_prop_trial, K_prop_trial = external_system(
                    u_trial,
                    lambda_trial,
                    tangent=True,
                )
                exact_guard(working_model, context="arc-length corrector load")
                if prescribed_path_active:
                    K_trial_effective = (
                        K_trial - K_const_trial - lambda_trial * K_prop_trial
                    )
                    path_direction_trial_red = np.asarray(
                        T.T
                        @ (F_prop_trial - K_trial_effective @ prescribed_offset),
                        dtype=float,
                    ).reshape(-1)
                else:
                    path_direction_trial_red = np.asarray(
                        T.T @ F_prop_trial, dtype=float
                    ).reshape(-1)
                residual = (
                    np.asarray(
                        T.T @ (F_const_trial + lambda_trial * F_prop_trial),
                        dtype=float,
                    ).reshape(-1)
                    - np.asarray(T.T @ F_internal, dtype=float).reshape(-1)
                )
                residual_norm = float(np.linalg.norm(residual))
                arc_residual = float(
                    path_metric_dot(
                        dq_total, dlambda_total, dq_total, dlambda_total
                    )
                    - radius * radius
                )
                force_reference = max(
                    float(
                        np.linalg.norm(
                            np.asarray(
                                T.T @ (F_const_trial + lambda_trial * F_prop_trial),
                                dtype=float,
                            )
                        )
                    ),
                    float(np.linalg.norm(path_direction_trial_red)),
                    1.0,
                )
                arc_reference = max(radius * radius, 1.0e-24)

                if (
                    residual_norm <= tolerance * force_reference
                    and abs(arc_residual) <= arc_tolerance * arc_reference
                ):
                    if _arc_contact is not None:
                        contact_candidate = _arc_contact.prepare(u_trial, step_index, lambda_trial)
                        cancellation_safe_point(cancellation_token,"arc_length.contact_before_commit")
                    trial_states = states_candidate
                    accepted = True
                    break

                # Match the mature correction path: tangent reduction is only
                # needed when a Newton correction will actually be solved.
                # Keeping it below the convergence check avoids three sparse
                # projections and two sparse subtractions on every accepted
                # iteration while leaving the prescribed-path direction above
                # exact and unchanged.
                K_trial_red = (
                    (T.T @ K_trial @ T)
                    - (T.T @ K_const_trial @ T)
                    - lambda_trial * (T.T @ K_prop_trial @ T)
                ).tocsr()
                corrector_tangent_projection_count += 1
                try:
                    handle = factorize(
                        K_trial_red,
                        (
                            MatrixClass.GENERAL
                            if general_tangent
                            else MatrixClass.SYMMETRIC_INDEFINITE
                        ),
                        signature=f"arc_length.corrector:{step_index}:{retry}:{iteration}",
                    )
                    corrector_rhs[:, 0] = residual
                    corrector_rhs[:, 1] = path_direction_trial_red
                    corrections = np.asarray(
                        handle.solve_many(corrector_rhs), dtype=float
                    )
                    correction_at_fixed_load = corrections[:, 0]
                    correction_per_load = corrections[:, 1]
                    corrector_solve_many_count += 1
                except Exception:
                    step_failure = "singular_corrector_tangent"
                    break
                if (
                    np.any(~np.isfinite(correction_at_fixed_load))
                    or np.any(~np.isfinite(correction_per_load))
                ):
                    step_failure = "nonfinite_corrector_solution"
                    break

                denominator = 2.0 * path_metric_dot(
                    dq_total,
                    dlambda_total,
                    correction_per_load,
                    1.0,
                )
                denominator_scale = max(
                    2.0 * radius * max(_metric_norm(W, correction_per_load), load_scaling),
                    1.0,
                )
                if abs(denominator) <= 1.0e-14 * denominator_scale:
                    step_failure = "singular_arc_constraint_linearization"
                    break

                numerator = -arc_residual - 2.0 * path_metric_dot(
                    dq_total,
                    dlambda_total,
                    correction_at_fixed_load,
                    0.0,
                )
                dlambda_correction = numerator / denominator
                dq_correction = correction_at_fixed_load + correction_per_load * dlambda_correction
                if (
                    np.any(~np.isfinite(dq_correction))
                    or not np.isfinite(dlambda_correction)
                ):
                    step_failure = "nonfinite_arc_correction"
                    break

                q_trial += dq_correction
                lambda_trial += float(dlambda_correction)
                dq_total = q_trial - q_base
                dlambda_total = lambda_trial - lambda_base
            else:
                step_failure = "maximum_iterations_reached"

            if accepted:
                q = q_trial
                lam = float(lambda_trial)
                committed_states = _commit_nonlinear_state_candidate(
                    committed_states,
                    trial_states,
                    model=working_model,
                    accepted_displacements=full_displacement(
                        q_trial,
                        lambda_trial,
                    ),
                )
                exact_guard(
                    working_model,
                    context="arc-length committed-state observation",
                )
                if _arc_contact is not None:
                    _arc_contact.commit(contact_candidate)
                u = full_displacement(q, lam)
                path_increment_norm = float(
                    np.sqrt(
                        max(
                            path_metric_dot(
                                dq_total,
                                dlambda_total,
                                dq_total,
                                dlambda_total,
                            ),
                            0.0,
                        )
                    )
                )
                is_new_peak = lam > peak_load_factor
                if is_new_peak:
                    peak_load_factor = float(lam)
                    peak_step_index = step_index
                    descending_steps = 0
                    if peak_step is not None:
                        peak_step.is_peak = False
                else:
                    required_drop = settings.peak_drop_tolerance * max(abs(peak_load_factor), 1.0)
                    if peak_step_index is not None and lam < peak_load_factor - required_drop:
                        descending_steps += 1
                    else:
                        descending_steps = 0

                # The converged Newton evaluation already contains the exact
                # accepted internal and external forces.  Reuse them for
                # support reactions.  Direct reduced assembly carries a lazy,
                # generation-qualified view of its local force buffers, so it
                # needs only one full scatter here—not a second constitutive
                # evaluation.  A stale/unrecognized payload conservatively
                # falls back to the original full-coordinate recovery.
                from .nonlinear_performance_batch_c import (
                    materialize_full_internal_force,
                )

                reaction_internal = materialize_full_internal_force(
                    F_internal,
                    working_model.mesh.dof_manager.total_dofs,
                )
                exact_guard(
                    working_model,
                    context="arc-length reaction-force observation",
                )
                if reaction_internal is None:
                    reaction_internal, _unused, _reaction_states = (
                        _assemble_nonlinear_system(
                            working_model,
                            u,
                            committed_states,
                            num_layers,
                            tangent=False,
                            kinematics=kinematics,
                            corotational_tangent=resolved_corotational_tangent,
                            require_full_coordinates=True,
                        )
                    )
                    exact_guard(
                        working_model,
                        context="arc-length reaction assembly",
                    )
                    # Reaction recovery is diagnostic-only; do not leave its
                    # trial state active for the next continuation step.
                    _discard_nonlinear_state_candidate(committed_states)
                    reaction_force_reassembly_count += 1
                    (
                        reaction_constant,
                        _unused,
                        reaction_proportional,
                        _unused,
                    ) = external_system(u, lam, tangent=False)
                    exact_guard(working_model, context="arc-length reaction load")
                else:
                    reaction_force_reuse_count += 1
                    reaction_constant = F_const_trial
                    reaction_proportional = F_prop_trial
                support_reactions = _support_reaction_resultants_from_forces(
                    working_model,
                    reaction_internal,
                    reaction_constant,
                    reaction_proportional,
                    lam,
                    dof_plan=support_reaction_dof_plan,
                )
                exact_guard(
                    working_model,
                    context="arc-length support reactions",
                )
                support_reaction_history.append(
                    {
                        "step_index": int(step_index),
                        "load_factor": float(lam),
                        "support_reactions": {
                            name: list(values)
                            for name, values in support_reactions.items()
                        },
                    }
                )

                displacement_norm = float(np.linalg.norm(u))
                step = ArcLengthStep(
                    step_index=step_index,
                    load_factor=float(lam),
                    iterations=iteration,
                    retries=retry,
                    arc_radius=float(radius),
                    residual_norm=float(residual_norm),
                    arc_residual=float(arc_residual),
                    displacement_norm=displacement_norm,
                    load_increment=float(dlambda_total),
                    path_increment_norm=path_increment_norm,
                    max_equivalent_plastic_strain=_max_plastic_strain(committed_states),
                    is_peak=is_new_peak,
                    support_reactions=support_reactions,
                )
                steps.append(step)
                if is_new_peak:
                    peak_step = step
                if record_increment_snapshots:
                    snapshots.append(
                        _increment_snapshot(
                            step_index,
                            lam,
                            u,
                            committed_states,
                            control_value=float(lam),
                        )
                    )
                    exact_guard(
                        working_model,
                        context="arc-length increment snapshot",
                    )
                previous_dq = dq_total.copy()
                previous_dlambda = float(dlambda_total)
                if track_step_translation:
                    max_translation = _max_nodal_translation(working_model, u)
                    exact_guard(
                        working_model,
                        context="arc-length translation observation",
                    )
                if progress_callback is not None:
                    emit_progress(
                        progress_callback,
                        "nonlinear_static_step",
                        "arc_length.continuation",
                        completed=local_step_index,
                        total=settings.max_steps,
                        iteration=iteration,
                        control="arc length",
                        step_index=int(step_index),
                        load_factor=float(lam),
                        peak_load_factor=float(peak_load_factor),
                        displacement_norm=displacement_norm,
                        max_translation=float(max_translation),
                        iterations=int(iteration),
                        max_equivalent_plastic_strain=float(
                            step.max_equivalent_plastic_strain
                        ),
                        support_reactions={
                            name: list(values)
                            for name, values in support_reactions.items()
                        },
                    )
                    exact_guard(
                        working_model,
                        context="arc-length progress callback",
                    )

                old_radius = radius
                if iteration <= max(settings.target_iterations // 2, 1):
                    radius = min(radius * settings.growth_factor, max_radius)
                    action = "grow"
                elif iteration > settings.target_iterations:
                    radius = max(
                        radius * max(settings.cutback_factor, np.sqrt(settings.target_iterations / iteration)),
                        min_radius,
                    )
                    action = "shrink_after_slow_convergence"
                else:
                    action = "keep"
                adaptation_history.append(
                    {
                        "step_index": step_index,
                        "iterations": int(iteration),
                        "retries": int(retry),
                        "accepted_radius": float(old_radius),
                        "next_radius": float(radius),
                        "action": action,
                    }
                )
                break

            _discard_nonlinear_state_candidate(committed_states)
            radius *= settings.cutback_factor
            adaptation_history.append(
                {
                    "step_index": step_index,
                    "retry": int(retry),
                    "accepted": False,
                    "next_radius": float(radius),
                    "action": "cutback_after_nonconvergence",
                    "failure_reason": step_failure,
                }
            )
            if radius < min_radius:
                break

        if not accepted:
            q = q_base
            lam = lambda_base
            committed_states = states_base
            status = "stopped_at_limit" if steps else "diverged"
            failure_reason = step_failure if radius >= min_radius else "minimum_arc_radius_reached"
            break

        if (
            settings.post_peak_load_fraction is not None
            and peak_step_index is not None
            and step_index > peak_step_index
            and lam <= settings.post_peak_load_fraction * peak_load_factor
        ):
            # Automatic post-buckling stop: the descending branch has shed
            # the requested fraction of the peak load, so the post-buckling
            # response is traced and further continuation adds no insight.
            status = "post_buckling_traced"
            break
        if settings.max_translation is not None and max_translation > settings.max_translation:
            status = "displacement_limit_reached"
            break
        if descending_steps >= settings.stop_after_peak_steps:
            status = "peak_confirmed"
            break
        if (
            settings.maximum_absolute_load_factor is not None
            and abs(lam) >= settings.maximum_absolute_load_factor
        ):
            status = "load_factor_limit_reached"
            break
    else:
        status = "maximum_steps_reached"

    u_final = full_displacement(q, lam)
    if not track_step_translation and steps:
        max_translation = _max_nodal_translation(working_model, u_final)
        exact_guard(working_model, context="arc-length final translation observation")
    if _arc_contact is not None and failure_reason is not None:
        committed_states = _finalize_failed_arc_element_states(
            working_model,u_final,committed_states,num_layers,info,
            kinematics=kinematics,failure_reason=failure_reason)
    else:
        committed_states = _finalize_arc_element_states(
            working_model,
            u_final,
            committed_states,
            num_layers,
            info,
            kinematics=kinematics,
        )
    exact_guard(working_model, context="arc-length final state recovery")
    info["failure_reason"] = failure_reason
    info["last_converged_load_factor"] = float(lam)
    info["peak_load_factor"] = float(peak_load_factor)
    info["peak_step_index"] = peak_step_index
    info["descending_steps_after_peak"] = int(descending_steps)
    info["final_max_translation"] = float(max_translation)
    info["load_scaling"] = float(load_scaling)
    info["rotation_length_scale"] = float(rotation_scale)
    info["initial_arc_radius"] = float(initial_radius)
    info["minimum_arc_radius"] = float(min_radius)
    info["maximum_arc_radius"] = float(max_radius)
    info["adaptation_history"] = adaptation_history
    info["support_reaction_history"] = support_reaction_history
    info["strain_summary"] = _nonlinear_state_summary(committed_states)
    info["preload"] = preload_info
    info["total_newton_iterations"] = int(total_iterations)
    info["total_retries"] = int(total_retries)
    info["predictor_reuse"] = {
        "reuse_count": predictor_reuse_count,
        "solve_count": predictor_solve_count,
        "exclusion_reason": predictor_reuse_exclusion,
    }
    info["corrector_solve_many_count"] = int(corrector_solve_many_count)
    info["corrector_tangent_projection_count"] = int(
        corrector_tangent_projection_count
    )
    info["reaction_force_recovery"] = {
        "accepted_force_reuse_count": int(reaction_force_reuse_count),
        "full_reassembly_count": int(reaction_force_reassembly_count),
    }
    info["solve_time"] = float(time.time() - start_time)
    info["constraint_postcheck"] = constraint_residual_summary(
        working_model, u_final, affine_scale=lam
    )
    exact_guard(working_model, context="arc-length constraint postcheck")
    info["result_case"] = make_result_case(
        name="nonlinear_static_arc_length",
        analysis_type="nonlinear_static",
        load_cases=tuple(
            case
            for case in (constant_load_case, load_case)
            if case is not None
        ),
        assembly_info=assembly_info,
        solver_info={"convergence_info": {"status": status, "failure_reason": failure_reason}},
        recovery={
            "displacements": True,
            "element_states": True,
            "force_displacement_history": True,
            "arc_length_history": True,
        },
        settings={
            "control": "arc_length",
            "arc_length": settings.to_dict(),
            "max_iterations": int(max_iterations),
            "tolerance": float(tolerance),
            "arc_tolerance": float(arc_tolerance),
            "num_layers": int(num_layers),
            "kinematics": kinematics,
            "corotational_tangent": resolved_corotational_tangent,
        },
    ).to_dict()

    if _arc_contact is not None:
        info["general_contact_profile"] = _arc_contact.restart_contract()["profile"]
        info["general_contact_history"] = copy.deepcopy(_arc_contact.history)
        if _arc_contact.history:
            info["general_contact"] = copy.deepcopy(_arc_contact.history[-1]["contact"])
    restart_payload = None
    if emit_restart_checkpoint:
        def arc_path_payload(**kwargs):
            payload = _arc_restart_path_payload(**kwargs)
            if _arc_contact is not None:
                payload['contact_state'] = _arc_contact.encoded_state()
            return payload
        restart_payload = create_nonlinear_checkpoint(
            analysis_kind="arc_length",
            model=working_model,
            analysis_contract=restart_analysis_contract,
            displacements=u_final,
            element_states=committed_states,
            deleted_element_ids=(),
            path_state=arc_path_payload(
                load_factor=float(lam),
                steps=steps,
                reduced_coordinates=q,
                exact_base_offset=exact_base_offset,
                radius=float(radius),
                minimum_radius=float(min_radius),
                maximum_radius=float(max_radius),
                initial_radius=float(initial_radius),
                previous_dq=previous_dq,
                previous_dlambda=previous_dlambda,
                peak_load_factor=float(peak_load_factor),
                peak_step_index=peak_step_index,
                descending_steps=int(descending_steps),
                max_translation=float(max_translation),
                load_scaling=float(load_scaling),
                rotation_length_scale=float(rotation_scale),
                adaptation_history=adaptation_history,
                support_reaction_history=support_reaction_history,
                total_iterations=int(total_iterations),
                total_retries=int(total_retries),
                corrector_solve_many_count=int(corrector_solve_many_count),
                corrector_tangent_projection_count=int(
                    corrector_tangent_projection_count
                ),
                reaction_force_reuse_count=int(reaction_force_reuse_count),
                reaction_force_reassembly_count=int(
                    reaction_force_reassembly_count
                ),
                preload_info=preload_path_info,
                terminal_status=status,
                failure_reason=failure_reason,
            ),
        )
        exact_guard(working_model, context="arc-length checkpoint output")

    return ArcLengthResult(
        steps=steps,
        status=status,
        displacements=u_final,
        load_factor=float(lam),
        peak_load_factor=float(peak_load_factor),
        peak_step_index=peak_step_index,
        element_states=committed_states,
        info=info,
        snapshots=tuple(snapshots),
        restart_checkpoint=restart_payload,
    )


def solve_static_arc_length(
    model: "FEModel",
    load_case: Optional["LoadCase"],
    *,
    constant_load_case: Optional["LoadCase"] = None,
    control: Optional[ArcLengthControl] = None,
    max_iterations: int = 25,
    tolerance: float = 1.0e-6,
    arc_tolerance: float = 1.0e-6,
    num_layers: int = 5,
    imperfection: Optional[Any] = None,
    initial_element_states: Optional[Mapping[int, Any]] = None,
    kinematics: str = "von_karman",
    corotational_tangent: str = "auto",
    progress_callback: Optional[ProgressCallback] = None,
    resource_config: Optional[ResourceConfig] = None,
    cancellation_token: Optional[CancellationToken] = None,
    record_increment_snapshots: bool = False,
    restart_checkpoint: Optional[Any] = None,
    emit_restart_checkpoint: bool = False,
    contact: Any = None,
    contact_settings: Any = None,
) -> ArcLengthResult:
    """Run arc-length analysis under one non-renewable family lease."""

    if contact is None and contact_settings is not None:
        raise ValueError('contact_settings requires contact')
    if contact is not None:
        if kinematics != 'von_karman' or corotational_tangent != 'auto':
            raise ValueError('arc contact requires von_karman/auto physical skin chart')
        if imperfection is not None or initial_element_states is not None:
            raise ValueError('arc contact imperfection/seeded history is not integrated')
        if _has_follower_pressure(load_case, model=model) or _has_follower_pressure(constant_load_case, model=model):
            raise ValueError('arc fixed-tool profile requires dead proportional load')
        if getattr(model,'initial_fields',None):
            raise ValueError('arc contact initial fields are unsupported')
    exact_guard = _EXACT_QUALIFIED_LIFECYCLE_GUARD
    run_under_lease = _run_with_qualified_assembly_runtime_lease
    own_resource_config = _owned_resource_config_snapshot
    solve_under_lease = _solve_static_arc_length_under_lease
    exact_guard(
        model,
        context="solve_static_arc_length preflight",
    )

    def operation(lease: Any) -> ArcLengthResult:
        def post_observation() -> None:
            exact_guard(
                model,
                context="solve_static_arc_length resource configuration",
            )
            lease(
                model,
                context="solve_static_arc_length resource configuration",
            )

        owned_resource_config = own_resource_config(
            resource_config,
            post_observation=post_observation,
        )
        def run_adapter(adapter=None):
            return _run_with_nonlinear_state_cleanup(
                solve_under_lease,
                model,
                load_case,
                constant_load_case=constant_load_case,
                control=control,
                max_iterations=max_iterations,
                tolerance=tolerance,
                arc_tolerance=arc_tolerance,
                num_layers=num_layers,
                imperfection=imperfection,
                initial_element_states=initial_element_states,
                kinematics=kinematics,
                corotational_tangent=corotational_tangent,
                progress_callback=progress_callback,
                resource_config=owned_resource_config,
                cancellation_token=cancellation_token,
                record_increment_snapshots=record_increment_snapshots,
                restart_checkpoint=restart_checkpoint,
                emit_restart_checkpoint=emit_restart_checkpoint,
                _qualified_runtime_guard=lease,
                _arc_contact=adapter,
            )
        if contact is None:
            return run_adapter()
        from .general_contact.arc_contact import arc_contact_scope
        cancellation_safe_point(cancellation_token, 'arc_length.contact_preflight')
        with arc_contact_scope(model, contact, contact_settings, preload=constant_load_case is not None) as adapter:
            return run_adapter(adapter)

    return run_under_lease(
        model,
        context="solve_static_arc_length",
        operation=operation,
    )
