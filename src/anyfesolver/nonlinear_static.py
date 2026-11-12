"""Incremental Newton-Raphson solver with geometric and material nonlinearity.

Geometric nonlinearity: total-Lagrangian von Karman kinematics in the shell
elements (membrane-bending coupling from transverse-deflection gradients,
initial-stress stiffness from the current membrane resultants) and a
consistent beam-column axial coupling in the 2-node beam.

Material nonlinearity: layered J2 plane-stress plasticity in the shells with
the isotropic hardening curve attached to the material
(``Material.hardening_curve``, e.g. a DNV-RP-C208 curve from
:mod:`anyfesolver.material_curves`).  Materials without a curve stay elastic.

Solution strategy (chosen for speed):

* full Newton-Raphson per load increment (quadratic-ish convergence, one
  sparse factorization per iteration),
* vectorized element kernels with cached reference geometry,
* COO-triplet assembly of tangent and internal force in a single element loop,
* adaptive load stepping: the increment halves on non-convergence and grows
  again after fast steps, so the run survives limit points gracefully and
  reports the last converged load factor as the capacity estimate.

The external load is ``F = F_constant + lambda * F_proportional`` so dead
loads or imperfection loads can be held while the proportional part ramps.
"""

from __future__ import annotations

import math

import copy
import os
import threading
import time
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from .assembly import build_constraint_transformation
from .cases import make_result_case
from .constraint_audit import constraint_residual_summary
from .control import (
    CancellationToken,
    ProgressCallback,
    cancellation_safe_point,
    emit_progress,
)
from .current_state_tangent import (
    _capture_qualified_v2d_trusted_authority as _CAPTURE_QUALIFIED_V2D_TRUSTED_AUTHORITY,
    _guarded_owned_input_snapshot as _guarded_owned_nonlinear_snapshot,
    require_exact_qualified_component_lifecycle_api as _EXACT_QUALIFIED_LIFECYCLE_GUARD,
)
from .e4_pl_element import QualifiedE4PLShellElement as _QualifiedE4PLShellElement
from .e4_pl_s3_element import (
    QualifiedE4PLS3ShellElement as _QualifiedE4PLS3ShellElement,
)
from .e4_pl_s3_v2d_element import (
    NativeParityE4PLS3V2DShellElement as _NativeParityE4PLS3V2DShellElement,
)
from .element_capabilities import (
    require_model_element_capabilities,
    require_model_nonlinear_workflow_capabilities,
)
from .fracture import (
    DeletedElementRecord,
    FractureConfig,
    detect_new_deletions,
    deleted_pressure_load_resultant,
    element_fracture_category,
    filtered_load_case_for_deleted_elements,
    fracture_summary,
    mpc_warning_for_deleted_shells,
)
from .linalg import MatrixClass, factorize
from .initial_field_state import state_has_active_initial_fields
from .jit_compiler import numba_thread_scope
from .matrix_assembly import (
    _run_with_qualified_assembly_runtime_lease,
    _scatter_element_matrix,
    _triplets_to_csr,
    assemble_external_load_system,
    assemble_load_vector,
    assemble_stiffness_matrix,
)
from .nonlinear_analysis_diagnostics import (
    capture_nonlinear_analysis_diagnostics,
    record_nonlinear_assembly_execution,
    record_nonlinear_solver_event,
)
from .plasticity import PlaneStressConvergenceError
from . import _q4_assumed_strain_plasticity as _q4_assumed_strain
from .nonlinear_state import (
    _register_nonlinear_state_cleanup,
    _run_with_nonlinear_state_cleanup,
    NonlinearStateStore,
    StateMaterializationPolicy,
    StateTransactionError,
    commit_state_candidate,
    discard_active_state_candidate,
    materialize_state_mapping,
)
from .nonlinear_restart import (
    canonical_checkpoint_json_bytes,
    create_nonlinear_checkpoint,
    load_case_descriptor,
    load_nonlinear_checkpoint,
    validate_nonlinear_checkpoint,
)
from .recovery import ResourceConfig, _owned_resource_config_snapshot
from .threading_policy import resource_threaded, thread_policy_diagnostics

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


_DOF_INDEX = {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5}

# Import-time identities for the solver's own cancellation checkpoint.  A
# replaced checkpoint, a foreign/sub-classed token or a token whose class,
# event or instance data changed remains a hostile observation boundary.
_EXACT_CANCELLATION_SAFE_POINT = cancellation_safe_point
_EXACT_CANCELLATION_TOKEN_TYPE = CancellationToken
_EXACT_CANCELLATION_TOKEN_NAMESPACE = tuple(
    type.__getattribute__(CancellationToken, "__dict__").items()
)
_EXACT_CANCELLATION_EVENT_TYPE = threading.Event
_EXACT_CANCELLATION_EVENT_NAMESPACE = tuple(
    type.__getattribute__(threading.Event, "__dict__").items()
)
_EXACT_CANCELLATION_LOCK_TYPE = type(threading.Lock())
_EXACT_CANCELLATION_CONDITION_TYPE = threading.Condition


def _exact_namespace_unchanged(owner: type, expected: tuple) -> bool:
    actual = type.__getattribute__(owner, "__dict__")
    return len(actual) == len(expected) and all(
        name in actual and actual[name] is value for name, value in expected
    )


def _is_known_cancellation_checkpoint(token: Any) -> bool:
    """Whether a checkpoint can only have run the solver's own token code."""

    if cancellation_safe_point is not _EXACT_CANCELLATION_SAFE_POINT:
        return False
    if token is None:
        return True
    if type(token) is not _EXACT_CANCELLATION_TOKEN_TYPE:
        return False
    try:
        namespace = object.__getattribute__(token, "__dict__")
    except AttributeError:
        return False
    if type(namespace) is not dict or set(namespace) != {
        "_event",
        "_lock",
        "_reason",
    }:
        return False
    event = dict.__getitem__(namespace, "_event")
    if (
        type(event) is not _EXACT_CANCELLATION_EVENT_TYPE
        or type(dict.__getitem__(namespace, "_lock"))
        is not _EXACT_CANCELLATION_LOCK_TYPE
        or type(dict.__getitem__(namespace, "_reason")) is not str
    ):
        return False
    try:
        event_namespace = object.__getattribute__(event, "__dict__")
    except AttributeError:
        return False
    if type(event_namespace) is not dict or set(event_namespace) != {
        "_cond",
        "_flag",
    }:
        return False
    # ``Event.is_set`` returns ``_flag`` and the checkpoint truth-tests it, so
    # a non-bool flag would run its own ``__bool__`` code at every checkpoint.
    if (
        type(dict.__getitem__(event_namespace, "_flag")) is not bool
        or type(dict.__getitem__(event_namespace, "_cond"))
        is not _EXACT_CANCELLATION_CONDITION_TYPE
    ):
        return False
    return _exact_namespace_unchanged(
        _EXACT_CANCELLATION_TOKEN_TYPE, _EXACT_CANCELLATION_TOKEN_NAMESPACE
    ) and _exact_namespace_unchanged(
        _EXACT_CANCELLATION_EVENT_TYPE, _EXACT_CANCELLATION_EVENT_NAMESPACE
    )


_FAST_NL_BOOTSTRAPPED = False
_FAST_NL_BOOTSTRAP_ERROR: Optional[str] = None
_FAST_NL_BOOTSTRAP_LOCK = threading.RLock()
_INITIAL_FIELD_STATE_KEYS = (
    "initial_membrane_stress",
    "initial_bending_stress",
    "initial_membrane_prestrain",
    "initial_curvature_prestrain",
    "initial_fiber_stress",
    "initial_fiber_prestrain",
)
_Q4_COMMITTED_BINDING_KEY = "qualified_q4_committed_binding"
_Q4_COMMITTED_INTEGRITY_KEY = "state_integrity_sha256"
_Q4_ACTIVITY_DISPOSITION_KEY = "qualified_q4_activity_disposition"
_Q4_COMMITTED_STATE_LIFECYCLE_SCHEMA = (
    "ANYSOLVER_Q4_COMMITTED_STATE_LIFECYCLE_V1"
)
_S3_ACTIVITY_DISPOSITION_KEY = "qualified_s3_activity_disposition"
_S3_COMMITTED_STATE_LIFECYCLE_SCHEMA = (
    "ANYSOLVER_S3_COMMITTED_STATE_LIFECYCLE_V1"
)


def _ensure_nonlinear_acceleration() -> None:
    """Install optional nonlinear acceleration on first nonlinear use."""

    global _FAST_NL_BOOTSTRAPPED, _FAST_NL_BOOTSTRAP_ERROR
    if _FAST_NL_BOOTSTRAPPED:
        return
    with _FAST_NL_BOOTSTRAP_LOCK:
        if _FAST_NL_BOOTSTRAPPED:
            return
        _FAST_NL_BOOTSTRAP_ERROR = None
        if os.environ.get("FE_SOLVER_DISABLE_FAST_NL", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _FAST_NL_BOOTSTRAPPED = True
            return
        try:
            from .nonlinear_performance_bootstrap import (
                install_nonlinear_performance_optimizations,
            )

            install_nonlinear_performance_optimizations()
        except Exception as exc:  # Optional acceleration must not disable the solver.
            _FAST_NL_BOOTSTRAP_ERROR = f"{type(exc).__name__}: {exc}"
        finally:
            # Publish completion only after the composite install attempt has
            # finished, so another first-use caller cannot observe a partially
            # patched nonlinear stack.
            _FAST_NL_BOOTSTRAPPED = True


@dataclass(frozen=True)
class NonlinearLoadStage:
    """One ordered load stage in a nonlinear load program."""

    name: str
    load_case: "LoadCase"
    target_factor: float = 1.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("NonlinearLoadStage name must not be empty")
        factor = float(self.target_factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("target_factor must be finite and positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "target_factor", factor)


@dataclass(frozen=True)
class NonlinearLoadProgram:
    """Ordered nonlinear load path, e.g. permanent then environmental load."""

    stages: Sequence[NonlinearLoadStage]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("NonlinearLoadProgram requires at least one stage")
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(
                "NonlinearLoadProgram stage names must be unique; duplicates: "
                + ", ".join(duplicates)
            )
        object.__setattr__(self, "stages", stages)

    @property
    def total_factor(self) -> float:
        return float(sum(stage.target_factor for stage in self.stages))

    def stage_factors(self, path_factor: float) -> Dict[str, float]:
        remaining = max(float(path_factor), 0.0)
        factors: Dict[str, float] = {}
        for stage in self.stages:
            value = min(remaining, stage.target_factor)
            factors[stage.name] = max(value, 0.0)
            remaining -= value
        return factors

    def active_stage(self, path_factor: float) -> str:
        remaining = max(float(path_factor), 0.0)
        for stage in self.stages:
            if remaining <= stage.target_factor + 1.0e-12:
                return stage.name
            remaining -= stage.target_factor
        return self.stages[-1].name


@dataclass(frozen=True)
class ShellInitialField:
    """Prescribed shell residual stress and/or eigenstrain field.

    Each stress or strain component uses shell-local engineering-vector order
    ``[xx, yy, xy]``.  A three-value vector is uniform over the element;
    arrays with one row per integration point are also accepted.

    ``membrane_stress`` is the through-thickness-uniform stress.  The
    ``bending_stress`` value is the antisymmetric stress at the positive
    surface, varying linearly from its negative at ``-t/2``.  Prestrain is
    subtracted from the kinematic strain; ``curvature_prestrain`` has units
    of inverse length.
    """

    membrane_stress: Optional[Any] = None
    bending_stress: Optional[Any] = None
    membrane_prestrain: Optional[Any] = None
    curvature_prestrain: Optional[Any] = None
    source: str = "user"

    def __post_init__(self) -> None:
        if all(
            value is None
            for value in (
                self.membrane_stress,
                self.bending_stress,
                self.membrane_prestrain,
                self.curvature_prestrain,
            )
        ):
            raise ValueError("ShellInitialField requires at least one stress or prestrain component")
        if not str(self.source).strip():
            raise ValueError("ShellInitialField source must not be empty")

    def state_values(self) -> Dict[str, Any]:
        values = {
            "initial_membrane_stress": self.membrane_stress,
            "initial_bending_stress": self.bending_stress,
            "initial_membrane_prestrain": self.membrane_prestrain,
            "initial_curvature_prestrain": self.curvature_prestrain,
        }
        return {
            key: np.asarray(value, dtype=float).copy()
            for key, value in values.items()
            if value is not None
        }


@dataclass(frozen=True)
class BeamInitialField:
    """Prescribed beam-fiber residual stress and/or eigenstrain distribution.

    Values may be scalar, one value per section fiber, or one value per
    Gauss-point/fiber pair.  Arbitrary self-equilibrated distributions are
    therefore representable.  Beam fiber fields require the element's
    ``cross_section["fiber_plasticity"]`` configuration.
    """

    fiber_stress: Optional[Any] = None
    fiber_prestrain: Optional[Any] = None
    source: str = "user"

    def __post_init__(self) -> None:
        if self.fiber_stress is None and self.fiber_prestrain is None:
            raise ValueError("BeamInitialField requires fiber_stress or fiber_prestrain")
        if not str(self.source).strip():
            raise ValueError("BeamInitialField source must not be empty")

    def state_values(self) -> Dict[str, Any]:
        values = {
            "initial_fiber_stress": self.fiber_stress,
            "initial_fiber_prestrain": self.fiber_prestrain,
        }
        return {
            key: np.asarray(value, dtype=float).copy()
            for key, value in values.items()
            if value is not None
        }


@dataclass(frozen=True)
class DisplacementControl:
    """Scalar displacement constraint used with load-factor continuation."""

    node_id: Optional[int] = None
    dof: Optional[Union[str, int]] = None
    target_displacement: float = 0.0
    weighted_dofs: Optional[Mapping[Any, float]] = None

    def full_row(self, model: "FEModel") -> np.ndarray:
        row = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
        if self.weighted_dofs:
            for key, weight in self.weighted_dofs.items():
                if isinstance(key, tuple):
                    node_id, dof = key
                    dof_index = _local_dof_index(dof)
                    node = model.mesh.get_node(int(node_id))
                    if node is None:
                        raise ValueError(f"Displacement control references missing node {node_id}")
                    row[node.dofs[dof_index]] += float(weight)
                else:
                    row[int(key)] += float(weight)
        else:
            if self.node_id is None or self.dof is None:
                raise ValueError("DisplacementControl requires node_id and dof, or weighted_dofs")
            node = model.mesh.get_node(int(self.node_id))
            if node is None:
                raise ValueError(f"Displacement control references missing node {self.node_id}")
            row[node.dofs[_local_dof_index(self.dof)]] = 1.0
        if float(np.linalg.norm(row)) <= 0.0:
            raise ValueError("Displacement control row is empty")
        return row


@dataclass(frozen=True)
class NonlinearConvergenceSettings:
    """Automatic convergence controls for force-control nonlinear static solves.

    The settings do not change element theory.  They tune globalization and load
    increment adaptation: line-search usage, step growth after fast convergence,
    and cutback after difficult increments.
    """

    profile: str = "auto"
    line_search: str = "rescue"
    fast_iterations: int = 4
    slow_iterations: int = 9
    growth_factor: float = 1.5
    cutback_factor: float = 0.5
    max_step_factor: float = 2.0
    min_step_fraction: Optional[float] = None
    max_line_search_cuts: int = 16
    line_search_reduction: float = 0.5
    characteristic_length: Optional[float] = None

    def __post_init__(self) -> None:
        profile = str(self.profile).lower()
        line_search = str(self.line_search).lower()
        if profile not in {"legacy", "auto", "balanced", "fast", "robust"}:
            raise ValueError("profile must be one of 'legacy', 'auto', 'balanced', 'fast', or 'robust'")
        if line_search not in {"never", "rescue", "auto", "always", "armijo"}:
            raise ValueError(
                "line_search must be one of 'never', 'rescue', 'auto', "
                "'always', or 'armijo'"
            )
        if self.fast_iterations <= 0 or self.slow_iterations <= 0:
            raise ValueError("iteration thresholds must be positive")
        if self.growth_factor < 1.0:
            raise ValueError("growth_factor must be at least 1.0")
        if not (0.0 < self.cutback_factor < 1.0):
            raise ValueError("cutback_factor must be between 0 and 1")
        if self.max_step_factor <= 0.0:
            raise ValueError("max_step_factor must be positive")
        if self.min_step_fraction is not None and self.min_step_fraction <= 0.0:
            raise ValueError("min_step_fraction must be positive when supplied")
        if self.max_line_search_cuts <= 0:
            raise ValueError("max_line_search_cuts must be positive")
        if not (0.0 < self.line_search_reduction < 1.0):
            raise ValueError("line_search_reduction must be between 0 and 1")
        if self.characteristic_length is not None and (
            not np.isfinite(float(self.characteristic_length))
            or float(self.characteristic_length) <= 0.0
        ):
            raise ValueError(
                "characteristic_length must be finite and positive when supplied"
            )

    @staticmethod
    def for_profile(profile: str) -> "NonlinearConvergenceSettings":
        name = str(profile).lower()
        if name in {"auto", "balanced"}:
            return NonlinearConvergenceSettings(profile=name)
        if name == "fast":
            return NonlinearConvergenceSettings(
                profile="fast",
                line_search="auto",
                fast_iterations=3,
                slow_iterations=8,
                growth_factor=2.0,
                cutback_factor=0.5,
                max_step_factor=4.0,
                max_line_search_cuts=10,
            )
        if name == "robust":
            return NonlinearConvergenceSettings(
                profile="robust",
                line_search="always",
                fast_iterations=5,
                slow_iterations=7,
                growth_factor=1.25,
                cutback_factor=0.5,
                max_step_factor=1.0,
                max_line_search_cuts=20,
            )
        if name == "legacy":
            return NonlinearConvergenceSettings(
                profile="legacy",
                line_search="rescue",
                fast_iterations=5,
                slow_iterations=25,
                growth_factor=2.0,
                cutback_factor=0.5,
                max_step_factor=1.0,
                max_line_search_cuts=16,
            )
        raise ValueError("Unknown nonlinear convergence profile")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "line_search": self.line_search,
            "fast_iterations": int(self.fast_iterations),
            "slow_iterations": int(self.slow_iterations),
            "growth_factor": float(self.growth_factor),
            "cutback_factor": float(self.cutback_factor),
            "max_step_factor": float(self.max_step_factor),
            "min_step_fraction": self.min_step_fraction,
            "max_line_search_cuts": int(self.max_line_search_cuts),
            "line_search_reduction": float(self.line_search_reduction),
            "characteristic_length": self.characteristic_length,
        }


def _armijo_characteristic_length(
    model: "FEModel",
    requested: Optional[float],
    *,
    rotations_present: bool = True,
) -> float:
    """Resolve one frozen force/moment scaling length for Armijo merit."""

    if requested is not None:
        return float(requested)
    coordinates = np.asarray(model.mesh.get_node_coordinates(), dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] == 0
        or coordinates.shape[1] != 3
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError(
            "Armijo characteristic length requires finite reference node coordinates"
        )
    centered = coordinates - np.mean(coordinates, axis=0)
    value = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(value) or value <= np.finfo(float).eps:
        if not rotations_present:
            # A length scale cancels out when every independent residual row
            # is translational.  Keep a finite, restart-stable placeholder so
            # a one-node translational model does not need an irrelevant
            # user-supplied length.
            return 1.0
        raise ValueError(
            "Armijo characteristic length is degenerate; supply an explicit "
            "positive characteristic_length"
        )
    return value


def _armijo_independent_rotations_present(model: "FEModel") -> bool:
    """Return whether the validated constraint basis retains a rotation."""

    from .constraint_audit import require_valid_constraints

    audit = require_valid_constraints(model)
    dependent_dofs = {
        int(equation.dependent_dof) for equation in audit.equations
    }
    manager = model.mesh.dof_manager
    for full_dof in range(int(manager.total_dofs)):
        if full_dof in dependent_dofs:
            continue
        _node_id, local_index, _name = manager.get_dof_info(full_dof)
        if local_index >= 3:
            return True
    return False


def _armijo_residual_weights(
    model: "FEModel",
    independent_dofs: np.ndarray,
    characteristic_length: float,
) -> np.ndarray:
    """Map force rows to one and moment rows to inverse length."""

    weights = np.ones(int(independent_dofs.size), dtype=float)
    inverse_length = 1.0 / float(characteristic_length)
    manager = model.mesh.dof_manager
    for reduced_dof, full_dof in enumerate(
        np.asarray(independent_dofs, dtype=np.intp)
    ):
        _node_id, local_index, _name = manager.get_dof_info(int(full_dof))
        if local_index < 0:
            raise ValueError("Armijo scaling encountered an unknown independent DOF")
        if local_index >= 3:
            weights[reduced_dof] = inverse_length
    return weights


def _resolved_corotational_line_search_settings(
    settings: NonlinearConvergenceSettings,
    kinematics: str,
) -> Tuple[NonlinearConvergenceSettings, Optional[str]]:
    """Resolve the runtime policy without changing legacy V1 fingerprints."""

    if kinematics != "corotational" or str(settings.line_search).lower() not in {
        "auto",
        "rescue",
    }:
        return settings, None
    if str(settings.line_search) in {"auto", "rescue"}:
        # Preserve the established lowercase restart contract, which has
        # recorded the resolved plain-Newton policy since V1.
        return dataclass_replace(settings, line_search="never"), None
    # Older V1 contracts retained valid non-lowercase spelling.  Keep that
    # spelling in the fingerprint while applying the same case-insensitive
    # plain-Newton runtime behavior.
    return settings, "never"


def _coerce_convergence_settings(
    value: Optional[
        Union[str, Mapping[str, Any], NonlinearConvergenceSettings]
    ],
    *,
    _post_observation: Optional[Callable[[], Any]] = None,
) -> NonlinearConvergenceSettings:
    def observed() -> None:
        if _post_observation is not None:
            _post_observation()

    if value is None:
        return NonlinearConvergenceSettings.for_profile("auto")
    if isinstance(value, NonlinearConvergenceSettings):
        observed()
        return value
    if isinstance(value, str):
        result = NonlinearConvergenceSettings.for_profile(value)
        observed()
        return result
    if isinstance(value, Mapping):
        data = dict(value)
        observed()
        profile = str(data.pop("profile", "auto")).lower()
        observed()
        base = NonlinearConvergenceSettings.for_profile(profile).to_dict()
        base.update(data)
        observed()
        result = NonlinearConvergenceSettings(**base)
        observed()
        return result
    raise TypeError("convergence_settings must be None, a profile string, a mapping, or NonlinearConvergenceSettings")


@dataclass
class NonlinearStaticStep:
    """One converged load increment."""

    step_index: int
    load_factor: float
    iterations: int
    residual_norm: float
    displacement_norm: float
    max_equivalent_plastic_strain: float
    control_value: Optional[float] = None
    active_stage: Optional[str] = None
    deleted_element_count: int = 0
    max_fracture_utilization: float = 0.0
    support_reactions: Dict[str, Tuple[float, ...]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "load_factor": self.load_factor,
            "iterations": self.iterations,
            "residual_norm": self.residual_norm,
            "displacement_norm": self.displacement_norm,
            "max_equivalent_plastic_strain": self.max_equivalent_plastic_strain,
            "control_value": self.control_value,
            "active_stage": self.active_stage,
            "deleted_element_count": int(self.deleted_element_count),
            "max_fracture_utilization": float(self.max_fracture_utilization),
            "support_reactions": {
                str(name): [float(value) for value in values]
                for name, values in self.support_reactions.items()
            },
        }


@dataclass(frozen=True)
class NonlinearIncrementSnapshot:
    """Opt-in committed state at one converged nonlinear increment.

    Snapshots own defensive copies of the displacement vector and element
    state map.  They are therefore suitable for true step animation and
    material-history-aware recovery without reconstructing intermediate
    states from the final solution.
    """

    step_index: int
    load_factor: float
    displacements: np.ndarray
    element_states: Mapping[int, Any]
    control_value: Optional[float] = None

    def __post_init__(self) -> None:
        values = np.asarray(self.displacements, dtype=float).reshape(-1).copy()
        values.setflags(write=False)
        object.__setattr__(self, "displacements", values)
        object.__setattr__(self, "element_states", copy.deepcopy(dict(self.element_states)))

    def to_dict(self, *, include_displacements: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "step_index": int(self.step_index),
            "load_factor": float(self.load_factor),
            "control_value": self.control_value,
            "num_dofs": int(self.displacements.size),
            "element_state_ids": sorted(int(value) for value in self.element_states),
        }
        if include_displacements:
            payload["displacements"] = self.displacements.tolist()
        return payload


def _increment_snapshot(
    step_index: int,
    load_factor: float,
    displacements: np.ndarray,
    element_states: Mapping[int, Any],
    *,
    control_value: Optional[float] = None,
) -> NonlinearIncrementSnapshot:
    if isinstance(element_states, NonlinearStateStore):
        element_states = element_states.materialize_owned(
            policy=StateMaterializationPolicy.SAVED_STATE
        )
    return NonlinearIncrementSnapshot(
        step_index=int(step_index),
        load_factor=float(load_factor),
        displacements=displacements,
        element_states=element_states,
        control_value=control_value,
    )


@dataclass
class NonlinearStaticResult:
    """Result of the incremental geometric/material nonlinear solve."""

    steps: List[NonlinearStaticStep]
    status: str
    displacements: np.ndarray
    load_factor: float
    element_states: Dict[int, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    snapshots: Tuple[NonlinearIncrementSnapshot, ...] = field(default_factory=tuple)
    restart_checkpoint: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def to_contact_archive_bytes(self, *, max_bytes=16 * 1024 * 1024) -> bytes:
        """Export inert bounded contact data; this is not a restart operation."""
        from .general_contact.static_results_archive import serialize_static_contact_results
        return serialize_static_contact_results(self, max_bytes=max_bytes)

    @property
    def converged(self) -> bool:
        return self.status in {"completed", "stopped_at_limit"}

    @property
    def capacity_estimate(self) -> float:
        """Last converged proportional load factor."""
        return self.load_factor

    @property
    def peak_load_factor(self) -> float:
        return float(self.info.get("peak_load_factor", max((step.load_factor for step in self.steps), default=self.load_factor)))

    @property
    def last_converged_load_factor(self) -> float:
        return float(self.info.get("last_converged_load_factor", self.load_factor))

    @property
    def failure_reason(self) -> Optional[str]:
        return self.info.get("failure_reason")

    @property
    def status_category(self) -> str:
        return str(self.info.get("status_category", self.status))

    @property
    def stop_reason(self) -> Optional[str]:
        return self.info.get("stop_reason", self.failure_reason)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "status_category": self.status_category,
            "converged": self.converged,
            "load_factor": self.load_factor,
            "peak_load_factor": self.peak_load_factor,
            "last_converged_load_factor": self.last_converged_load_factor,
            "failure_reason": self.failure_reason,
            "stop_reason": self.stop_reason,
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

    def to_restart_checkpoint(self) -> Dict[str, Any]:
        """Return an owned complete checkpoint of the last committed step."""

        if self.restart_checkpoint is None:
            raise ValueError("this result does not contain a nonlinear restart checkpoint")
        return copy.deepcopy(self.restart_checkpoint)

    def restart_checkpoint_bytes(self) -> bytes:
        """Return the complete checkpoint as strict canonical JSON."""

        return canonical_checkpoint_json_bytes(self.to_restart_checkpoint())

    @property
    def quantity_metadata(self) -> Tuple[Any, ...]:
        from .quantities import describe_result_quantities

        return describe_result_quantities(self)


def _static_restart_analysis_contract(
    *,
    model: "FEModel",
    load_case: Optional["LoadCase"],
    constant_load_case: Optional["LoadCase"],
    load_program: Optional[NonlinearLoadProgram],
    control_name: str,
    displacement_control: Optional[DisplacementControl],
    num_layers: int,
    max_iterations: int,
    tolerance: float,
    effective_min_step_fraction: float,
    settings: NonlinearConvergenceSettings,
    fracture_config: Optional[FractureConfig],
    resource_config: Optional[ResourceConfig],
    kinematics: str,
    resolved_corotational_tangent: str,
) -> Dict[str, Any]:
    """Return invariant inputs required to continue the same static path.

    Target load/displacement and the number of additional increments are
    intentionally excluded.  The checkpoint owns the accepted path position
    and adaptive increment state; a resumed call may extend that same path.
    """

    if load_program is None:
        stages: List[Dict[str, Any]] = []
    else:
        stages = [
            {
                "name": stage.name,
                "target_factor": float(stage.target_factor),
                "load_case": load_case_descriptor(stage.load_case),
            }
            for stage in load_program.stages
        ]
    control_descriptor: Optional[Dict[str, Any]] = None
    if control_name == "displacement":
        if displacement_control is None:
            raise ValueError("displacement_control is required when control='displacement'")
        control_descriptor = {
            # The row defines the physical continuation coordinate.  Its new
            # target is deliberately not part of the invariant contract.
            "full_row": displacement_control.full_row(model).tolist(),
        }
    convergence_contract = settings.to_dict()
    if str(settings.line_search).lower() != "armijo":
        # Preserve the V1 legacy restart contract byte-for-byte when the new
        # globalization option is unused.
        convergence_contract.pop("characteristic_length", None)
    contract = {
        "schema": "ANYSOLVER_STATIC_RESTART_CONTRACT_V1",
        "control": control_name,
        "load_case": load_case_descriptor(load_case),
        "constant_load_case": load_case_descriptor(constant_load_case),
        "load_program": stages,
        "displacement_control": control_descriptor,
        "num_layers": int(num_layers),
        "max_iterations": int(max_iterations),
        "tolerance": float(tolerance),
        "effective_min_step_fraction": float(effective_min_step_fraction),
        "convergence_settings": convergence_contract,
        "fracture_config": (
            None if fracture_config is None else fracture_config.to_dict()
        ),
        "resource_config": (
            None if resource_config is None else resource_config.to_dict()
        ),
        "kinematics": str(kinematics),
        "corotational_tangent": str(resolved_corotational_tangent),
    }
    from .general_contact.runtime import active_contact_manager

    contact_manager = active_contact_manager()
    if contact_manager is not None:
        contract["schema"] = "ANYSOLVER_STATIC_RESTART_CONTRACT_V2"
        contract["general_contact"] = contact_manager.restart_contract()
    return contract


def _static_step_from_restart(value: Any) -> NonlinearStaticStep:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint static step must be a mapping")
    expected = {
        "step_index",
        "load_factor",
        "iterations",
        "residual_norm",
        "displacement_norm",
        "max_equivalent_plastic_strain",
        "control_value",
        "active_stage",
        "deleted_element_count",
        "max_fracture_utilization",
        "support_reactions",
    }
    if set(value) != expected:
        raise ValueError("checkpoint static step schema is incompatible")
    reactions = value["support_reactions"]
    if not isinstance(reactions, Mapping):
        raise ValueError("checkpoint static step reactions must be a mapping")
    return NonlinearStaticStep(
        step_index=int(value["step_index"]),
        load_factor=float(value["load_factor"]),
        iterations=int(value["iterations"]),
        residual_norm=float(value["residual_norm"]),
        displacement_norm=float(value["displacement_norm"]),
        max_equivalent_plastic_strain=float(value["max_equivalent_plastic_strain"]),
        control_value=(
            None if value["control_value"] is None else float(value["control_value"])
        ),
        active_stage=(
            None if value["active_stage"] is None else str(value["active_stage"])
        ),
        deleted_element_count=int(value["deleted_element_count"]),
        max_fracture_utilization=float(value["max_fracture_utilization"]),
        support_reactions={
            str(name): tuple(float(item) for item in components)
            for name, components in reactions.items()
        },
    )


def _restore_static_path_state(
    value: Mapping[str, Any],
    *,
    control_name: str,
    n_red: Optional[int] = None,
    total_dofs: Optional[int] = None,
    normal_multiplier_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate static continuation state before solver assembly."""

    if not isinstance(value, Mapping):
        raise ValueError("static checkpoint path_state must be a mapping")
    common = {
        "mode",
        "load_factor",
        "step_index",
        "total_iterations",
        "steps",
        "force_displacement_history",
        "reduced_coordinates",
        "terminal_status",
        "failure_reason",
    }
    expected = (
        common
        | {
            "base_step",
            "minimum_step",
            "maximum_step",
            "next_step_size",
            "force_line_search_next",
            "convergence_adaptation",
            "deletion_records",
            "fracture_warnings",
            "max_fracture_utilization",
            "affine_path_mode",
            "exact_base_offset",
        }
        if control_name == "force"
        else common
    )
    legacy_expected = (
        expected - {"affine_path_mode", "exact_base_offset"}
        if control_name == "force"
        else expected
    )
    if normal_multiplier_count is not None:
        expected = expected | {"normal_multipliers"}
        admitted_keys = {frozenset(expected)}
    else:
        admitted_keys = {frozenset(expected), frozenset(legacy_expected)}
    if frozenset(value) not in admitted_keys or value.get("mode") != control_name:
        raise ValueError("static checkpoint path schema is incompatible")
    steps_raw = value["steps"]
    history = value["force_displacement_history"]
    if not isinstance(steps_raw, list) or not isinstance(history, list):
        raise ValueError("static checkpoint histories must be lists")
    steps = [_static_step_from_restart(item) for item in steps_raw]
    step_index = int(value["step_index"])
    if step_index < 0 or len(steps) != step_index:
        raise ValueError("static checkpoint step count is inconsistent")
    if len(history) != step_index:
        raise ValueError("static checkpoint force/displacement history is incomplete")
    if [step.step_index for step in steps] != list(range(1, step_index + 1)):
        raise ValueError("static checkpoint step ordering is inconsistent")
    load_factor = float(value["load_factor"])
    if steps and steps[-1].load_factor != load_factor:
        raise ValueError("static checkpoint load factor differs from its last step")
    restored: Dict[str, Any] = {
        "mode": control_name,
        "load_factor": load_factor,
        "step_index": step_index,
        "total_iterations": int(value["total_iterations"]),
        "steps": steps,
        "force_displacement_history": copy.deepcopy(history),
        "terminal_status": str(value["terminal_status"]),
        "failure_reason": (
            None if value["failure_reason"] is None else str(value["failure_reason"])
        ),
    }
    reduced_coordinates = np.asarray(value["reduced_coordinates"], dtype=float)
    if (
        reduced_coordinates.ndim != 1
        or not np.all(np.isfinite(reduced_coordinates))
        or (n_red is not None and reduced_coordinates.shape != (int(n_red),))
    ):
        raise ValueError("static checkpoint reduced coordinates are incompatible")
    restored["reduced_coordinates"] = reduced_coordinates.copy()
    if normal_multiplier_count is not None:
        normal_multipliers = np.asarray(value["normal_multipliers"], dtype=float)
        if (normal_multipliers.shape != (normal_multiplier_count,)
                or not np.all(np.isfinite(normal_multipliers))
                or np.any(normal_multipliers < 0.0)):
            raise ValueError("static checkpoint normal multipliers are incompatible")
        restored["normal_multipliers"] = normal_multipliers.copy()
    if restored["total_iterations"] < 0:
        raise ValueError("static checkpoint iteration count is invalid")
    if control_name == "force":
        from .fracture import DeletedElementRecord

        deletion_records_raw = value["deletion_records"]
        if not isinstance(deletion_records_raw, list):
            raise ValueError("static checkpoint deletion records must be a list")
        deletion_records = [
            DeletedElementRecord(**record) for record in deletion_records_raw
        ]
        deletion_ids = [int(record.element_id) for record in deletion_records]
        if len(deletion_ids) != len(set(deletion_ids)):
            raise ValueError("static checkpoint deletion records contain duplicate IDs")
        for record in deletion_records:
            record_step = int(record.step_index)
            if record_step < 1 or record_step > len(steps):
                raise ValueError(
                    "static checkpoint deletion step lies outside its history"
                )
            if float(record.load_factor) != float(
                steps[record_step - 1].load_factor
            ):
                raise ValueError(
                    "static checkpoint deletion load factor differs from its step"
                )
        for step in steps:
            cumulative_deleted = sum(
                int(record.step_index) <= int(step.step_index)
                for record in deletion_records
            )
            if int(step.deleted_element_count) != cumulative_deleted:
                raise ValueError(
                    "static checkpoint cumulative deletion history is inconsistent"
                )
        if steps and steps[-1].deleted_element_count != len(deletion_ids):
            raise ValueError("static checkpoint deletion count differs from its last step")
        restored.update(
            {
                "base_step": float(value["base_step"]),
                "minimum_step": float(value["minimum_step"]),
                "maximum_step": float(value["maximum_step"]),
                "next_step_size": float(value["next_step_size"]),
                "force_line_search_next": bool(value["force_line_search_next"]),
                "convergence_adaptation": copy.deepcopy(value["convergence_adaptation"]),
                "deletion_records": deletion_records,
                "fracture_warnings": [str(item) for item in value["fracture_warnings"]],
                "max_fracture_utilization": float(value["max_fracture_utilization"]),
                "affine_path_mode": value.get("affine_path_mode"),
                "exact_base_offset": (
                    None
                    if value.get("exact_base_offset") is None
                    else np.asarray(
                        value["exact_base_offset"], dtype=np.float64
                    ).copy()
                ),
            }
        )
        affine_mode = restored["affine_path_mode"]
        exact_base = restored["exact_base_offset"]
        if affine_mode is not None and affine_mode not in {
            "PROPORTIONAL_PRESCRIBED_FIELD",
            "FIXED_RESTART_AFFINE_STATE",
        }:
            raise ValueError("static checkpoint affine path mode is incompatible")
        if exact_base is not None and (
            exact_base.ndim != 1
            or not np.all(np.isfinite(exact_base))
            or (
                total_dofs is not None
                and exact_base.shape != (int(total_dofs),)
            )
        ):
            raise ValueError("static checkpoint exact base offset is incompatible")
        if (affine_mode is None) != (exact_base is None):
            raise ValueError("static checkpoint affine path state is incomplete")
        positive_names = ("base_step", "minimum_step", "maximum_step", "next_step_size")
        if any(restored[name] <= 0.0 for name in positive_names):
            raise ValueError("static checkpoint adaptive step sizes must be positive")
        if restored["minimum_step"] > restored["maximum_step"]:
            raise ValueError("static checkpoint adaptive step bounds are inconsistent")
        if not isinstance(restored["convergence_adaptation"], list):
            raise ValueError("static checkpoint convergence history must be a list")
    return restored


def _static_restart_path_payload(
    *,
    control_name: str,
    load_factor: float,
    step_index: int,
    total_iterations: int,
    steps: Sequence[NonlinearStaticStep],
    force_displacement_history: Sequence[Mapping[str, Any]],
    reduced_coordinates: np.ndarray,
    terminal_status: str,
    failure_reason: Optional[str],
    **path_values: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "mode": str(control_name),
        "load_factor": float(load_factor),
        "step_index": int(step_index),
        "total_iterations": int(total_iterations),
        "steps": [step.to_dict() for step in steps],
        "force_displacement_history": copy.deepcopy(list(force_displacement_history)),
        "reduced_coordinates": np.asarray(
            reduced_coordinates, dtype=float
        ).reshape(-1).tolist(),
        "terminal_status": str(terminal_status),
        "failure_reason": None if failure_reason is None else str(failure_reason),
    }
    if control_name == "force":
        payload.update(path_values)
    elif "normal_multipliers" in path_values:
        payload["normal_multipliers"] = path_values["normal_multipliers"]
    return payload


def _augmented_contact_increment_summary(certificate: Any) -> Dict[str, Any]:
    """Keep accepted contact history dimensional and bounded in size."""
    summary = {
        "profile": str(certificate.profile),
        "minimum_gap": float(certificate.minimum_gap),
        "gap_tolerance_met": bool(certificate.gap_tolerance_met),
        "active_pressure_dofs": int(certificate.active_pressure_dofs),
        "integration_force": list(map(float, certificate.integration_force)),
        "target_force": list(map(float, certificate.target_force)),
        "integration_moment": list(map(float, certificate.integration_moment)),
        "target_moment": list(map(float, certificate.target_moment)),
    }
    origin = getattr(certificate, "wrench_origin_m", None)
    if origin is not None:
        summary["wrench_origin_m"] = list(map(float, origin))
    return summary


def _local_dof_index(dof: Union[str, int]) -> int:
    if isinstance(dof, str):
        key = dof.lower()
        if key not in _DOF_INDEX:
            raise ValueError(f"Unknown DOF {dof!r}; use one of {sorted(_DOF_INDEX)}")
        return _DOF_INDEX[key]
    index = int(dof)
    if not (0 <= index < 6):
        raise ValueError("DOF index must be in [0, 5]")
    return index


def _has_follower_pressure(
    load_case: Optional["LoadCase"],
    *,
    model: Optional["FEModel"] = None,
    _exact_guard: Any = None,
) -> bool:
    """Whether a load case contains current-area pressure.

    A custom mapping or descriptor may execute caller code while the policy is
    observed.  Qualified orchestration passes its original non-renewable guard
    so no subsequent load-case or model observation can occur after a
    transient authority mutation.
    """

    def observed(context: str) -> None:
        if _exact_guard is not None and model is not None:
            _exact_guard(model, context=context)

    if load_case is None:
        return False
    pressure_loads = getattr(load_case, "pressure_loads", None)
    observed("nonlinear follower-pressure mapping observation")
    has_pressure = bool(pressure_loads)
    observed("nonlinear follower-pressure mapping disposition")
    if not has_pressure:
        return False
    follower_pressure = getattr(load_case, "follower_pressure", False)
    observed("nonlinear follower-pressure policy observation")
    result = bool(follower_pressure)
    observed("nonlinear follower-pressure policy disposition")
    return result


def _weighted_external_load_system(
    model: "FEModel",
    weighted_load_cases: Sequence[Tuple[Optional["LoadCase"], float]],
    displacements: np.ndarray,
    *,
    tangent: bool,
) -> Tuple[np.ndarray, Optional[sparse.csr_matrix]]:
    """Assemble a weighted external force and its displacement derivative."""
    total_dofs = model.mesh.dof_manager.total_dofs
    force = np.zeros(total_dofs, dtype=float)
    load_tangent = sparse.csr_matrix((total_dofs, total_dofs), dtype=float) if tangent else None
    for load_case, raw_factor in weighted_load_cases:
        factor = float(raw_factor)
        if load_case is None or factor == 0.0:
            continue
        vector, case_tangent, _info = assemble_external_load_system(
            model,
            load_case,
            displacements,
            tangent=tangent,
        )
        force += factor * vector
        if tangent and case_tangent is not None and case_tangent.nnz:
            load_tangent = load_tangent + factor * case_tangent
    return force, load_tangent


def _assemble_nonlinear_system_unchecked(
    model: "FEModel",
    displacements: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    tangent: bool = True,
    deleted_element_ids: Optional[Sequence[int]] = None,
    residual_stiffness_fraction: float = 1.0,
    element_stiffness_scales: Optional[Mapping[int, float]] = None,
    kinematics: str = "von_karman",
    corotational_tangent: str = "rotated",
    require_full_coordinates: bool = False,
    stable_shell_rotation: bool = False,
) -> Tuple[np.ndarray, Any, Dict[int, Any]]:
    """Assemble F_int (and the tangent K_T when requested) at a state.

    ``kinematics`` selects between the default von Karman element response and
    the element-independent corotational formulation for large rigid
    rotations. ``corotational_tangent`` is already resolved to either
    ``"rotated"`` or ``"consistent"`` by the public solver.
    """
    assembly_start = time.perf_counter()
    if require_full_coordinates:
        assembly_fallback_reason = "full_coordinates_explicitly_required"
    elif str(kinematics) != "von_karman":
        assembly_fallback_reason = "kinematics_not_von_karman"
    elif tuple(deleted_element_ids or ()):
        assembly_fallback_reason = "deleted_elements_require_reference_assembly"
    elif element_stiffness_scales:
        assembly_fallback_reason = "element_stiffness_scales_require_reference_assembly"
    else:
        assembly_fallback_reason = "persistent_assembly_plan_not_selected"
    del require_full_coordinates  # consumed by the direct-reduction adapter
    from .nonlinear_state import begin_state_evaluation, finish_state_evaluation

    state_token = begin_state_evaluation(
        committed_states,
        model=model,
        displacements=displacements,
    )
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    F_int = np.zeros(total_dofs, dtype=float)
    data: list = []
    trial_states: Dict[int, Any] = {}
    deleted_set = {int(element_id) for element_id in (deleted_element_ids or ())}
    residual_fraction = float(residual_stiffness_fraction)
    freeze_deleted = getattr(committed_states, "freeze_deleted", None)
    if deleted_set and callable(freeze_deleted):
        freeze_deleted(tuple(sorted(deleted_set)))
    element_scales = {
        int(element_id): min(max(float(scale), 0.0), 1.0)
        for element_id, scale in (element_stiffness_scales or {}).items()
    }

    if tangent:
        from .matrix_assembly import _get_cached_sparsity_pattern
        rows_concat, cols_concat = _get_cached_sparsity_pattern(mesh, "tangent_stiffness")

    from .elements import ShellElement
    from .materials import is_isotropic_material
    from .nonlinear_element_evaluation import evaluate_nonlinear_element
    from .vectorized_nonlinear import (
        batch_shell_nonlinear_response, shell_nonlinear_batch_eligible,
    )

    groups = {}
    for elem_id, element in mesh.elements.items():
        if kinematics == "corotational":
            break
        # Initial stress/eigenstrain fields are deliberately evaluated by the
        # scalar element kernel.  The ordinary batched path stays untouched
        # and fast, while field-bearing elements can retain their immutable
        # offsets alongside the committed material history.
        if (
            isinstance(element, ShellElement)
            and shell_nonlinear_batch_eligible(element)
            and is_isotropic_material(model.get_material(element.material_name))
            and not _state_has_initial_field(committed_states.get(elem_id))
        ):
            key = (
                element.num_nodes,
                element.thickness,
                element.drilling_stabilization,
                element.reduced_integration,
                element.hourglass_stabilization,
                element.material_name,
                _q4_assumed_strain.route(
                    element, mesh, model.get_material(element.material_name)
                ),
            )
            if key not in groups:
                groups[key] = []
            groups[key].append((elem_id, element))

    precomputed_F = {}
    precomputed_K = {}

    for key, elem_list in groups.items():
        (num_nodes, thickness, drilling_stabilization, _reduced_integration,
         _hourglass_stabilization, material_name, assumed_strain_group) = key
        material = model.get_material(material_name)
        E = float(material.elastic_modulus)
        nu = float(material.poisson_ratio)
        G_mod = float(material.shear_modulus)
        curve = getattr(material, "hardening_curve", None)

        n_elem = len(elem_list)
        first_element = elem_list[0][1]
        n_dof = first_element.total_dofs

        # We need to extract the caches
        cache_first = first_element._nonlinear_geometry(mesh)
        n_gp = cache_first["detw_all"].shape[0]
        n_shear = cache_first["detw_shear_all"].shape[0]

        u_elem_batch = np.zeros((n_elem, n_dof))
        T0_batch = np.zeros((n_elem, n_dof, n_dof))
        B_m_all_batch = np.zeros((n_elem, n_gp, 3, n_dof))
        B_b_all_batch = np.zeros((n_elem, n_gp, 3, n_dof))
        B_d_all_batch = np.zeros((n_elem, n_gp, 1, n_dof))
        Gw_all_batch = np.zeros((n_elem, n_gp, 2, n_dof))
        detw_all_batch = np.zeros((n_elem, n_gp))
        B_s_all_batch = np.zeros((n_elem, n_shear, 2, n_dof))
        detw_shear_all_batch = np.zeros((n_elem, n_shear))

        plastic_strain_batch = np.zeros((n_elem, n_gp * num_layers, 3))
        alpha_batch = np.zeros((n_elem, n_gp * num_layers))

        dof_mappings = []
        for idx, (elem_id, element) in enumerate(elem_list):
            dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
            dof_mappings.append(dof_mapping)
            if dof_mapping.size > 0:
                u_elem_batch[idx] = displacements[dof_mapping]

            cache = element._nonlinear_geometry(mesh)
            T0_batch[idx] = cache["T0"]
            B_m_all_batch[idx] = cache["B_m_all"]
            B_b_all_batch[idx] = cache["B_b_all"]
            B_d_all_batch[idx] = cache["B_d_all"]
            Gw_all_batch[idx] = cache["Gw_all"]
            detw_all_batch[idx] = cache["detw_all"]
            B_s_all_batch[idx] = cache["B_s_all"]
            detw_shear_all_batch[idx] = cache["detw_shear_all"]

            state = committed_states.get(elem_id)
            if state is None:
                state = element.init_nonlinear_state(num_layers)
            plastic_strain_batch[idx] = state["plastic_strain"]
            alpha_batch[idx] = state["alpha"]

        # Qualified plastic Q4 shells evaluate the layered return map at the
        # hybrid assumed strain (see _q4_assumed_strain_plasticity).
        batch_kernel = batch_shell_nonlinear_response
        if assumed_strain_group:
            offsets = np.asarray([
                _q4_assumed_strain.offset_operator(element, mesh, material)
                for _elem_id, element in elem_list
            ])

            def batch_kernel(*arguments):
                return _q4_assumed_strain.kernel_batch(*arguments, offsets)[:5]

        F_int_batch, K_T_batch, ep_new, alpha_new, layer_strain_batch = batch_kernel(
            u_elem_batch,
            T0_batch,
            B_m_all_batch,
            B_b_all_batch,
            B_d_all_batch,
            Gw_all_batch,
            detw_all_batch,
            B_s_all_batch,
            detw_shear_all_batch,
            E,
            nu,
            G_mod,
            thickness,
            drilling_stabilization,
            tangent,
            curve,
            plastic_strain_batch,
            alpha_batch,
            num_layers,
        )

        correction_cache: Dict[tuple[Any, ...], np.ndarray] = {}
        for idx, (elem_id, element) in enumerate(elem_list):
            correction_builder = getattr(element, "_qualified_linear_correction", None)
            if callable(correction_builder):
                cache_key_builder = getattr(
                    element, "_qualified_stiffness_cache_key", None
                )
                correction_key = (
                    int(num_layers),
                    cache_key_builder(mesh, material)
                    if callable(cache_key_builder)
                    else (int(elem_id),),
                )
                correction = correction_cache.get(correction_key)
                if correction is None:
                    correction = np.asarray(
                        correction_builder(mesh, material, int(num_layers)),
                        dtype=float,
                    )
                    correction_cache[correction_key] = correction
                F_int_batch[idx] += correction @ u_elem_batch[idx]
                if tangent:
                    K_T_batch[idx] += correction
                direct_builder = (element._elastic_increment_response
                                  if type(element) is _QualifiedE4PLShellElement else None)
                direct_admission = (element._elastic_increment_admitted
                                    if type(element) is _QualifiedE4PLShellElement else None)
                if (callable(direct_builder) and callable(direct_admission)
                        and not assumed_strain_group
                        and direct_admission(mesh, material, committed_states.get(elem_id))):
                    # Prepare both independent baselines inside the guarded
                    # response, under its original pre-preparation snapshot.
                    direct = direct_builder(
                        mesh, material, u_elem_batch[idx],
                        committed_states.get(elem_id), tangent,
                        batch_baseline=True,
                    )
                    if direct is not None:
                        F_int_batch[idx] = direct[0]
                        if tangent:
                            K_T_batch[idx] = direct[1]
            precomputed_F[elem_id] = F_int_batch[idx]
            if tangent:
                precomputed_K[elem_id] = K_T_batch[idx]

            # Reconstruct trial state to be compatible with single-element API
            trial_state = {
                "plastic_strain": ep_new[idx],
                "alpha": alpha_new[idx],
                "layer_strain": layer_strain_batch[idx * n_gp * num_layers : (idx + 1) * n_gp * num_layers].copy(),
            }
            attach_algorithmic_origin = getattr(
                element,
                "attach_current_tangent_algorithmic_origin",
                None,
            )
            if callable(attach_algorithmic_origin):
                trial_state = attach_algorithmic_origin(
                    material,
                    {
                        "plastic_strain": plastic_strain_batch[idx],
                        "alpha": alpha_batch[idx],
                    },
                    trial_state,
                    int(num_layers),
                    tangent_evaluated=bool(tangent),
                )
            if curve is not None:
                # We need layer_stress for outputs if people use it, let's keep it simple: we can omit it if not strictly required,
                # but let's see if elements.py returned it. We didn't return it from batch_shell_nonlinear_response.
                pass
            if elem_id in deleted_set:
                trial_state = committed_states.get(elem_id, trial_state)
            trial_states[elem_id] = trial_state

    for elem_id, element in mesh.elements.items():
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue
        evaluated_deleted_noncurrent = False

        if elem_id in precomputed_F:
            f_elem = precomputed_F[elem_id]
            k_elem = precomputed_K.get(elem_id) if tangent else None
            # trial_state already in trial_states dict
        else:
            f_elem = None
            k_elem = None
            if kinematics == "corotational":
                from .corotational import corotational_element_response

                f_elem, k_elem, cr_trial_state = corotational_element_response(
                    model,
                    int(elem_id),
                    element,
                    displacements[dof_mapping],
                    tangent,
                    committed_state=committed_states.get(elem_id),
                    num_layers=num_layers,
                    tangent_mode=corotational_tangent,
                    stable_shell_rotation=stable_shell_rotation,
                )
                if f_elem is not None and cr_trial_state is not None:
                    trial_states[elem_id] = cr_trial_state
            if f_elem is None:
                material = model.get_material(element.material_name)
                u_elem = displacements[dof_mapping]
                deleted_operator = (
                    getattr(
                        element,
                        "compute_noncurrent_deleted_residual_operator",
                        None,
                    )
                    if elem_id in deleted_set
                    else None
                )
                if callable(deleted_operator):
                    deleted_mode = (
                        {"von_karman": kinematics == "von_karman"}
                        if type(element) is _NativeParityE4PLS3V2DShellElement
                        else {}
                    )
                    f_elem, k_elem = deleted_operator(
                        mesh,
                        material,
                        u_elem,
                        committed_states.get(elem_id),
                        int(num_layers),
                        tangent=bool(tangent),
                        **deleted_mode,
                    )
                    evaluated_deleted_noncurrent = True
                else:
                    f_elem, k_elem, trial_state = evaluate_nonlinear_element(
                        element,
                        mesh,
                        material,
                        u_elem,
                        committed_states.get(elem_id),
                        num_layers,
                        tangent,
                        committed_states=committed_states,
                        state_token=state_token,
                        element_id=elem_id,
                        kinematics=kinematics,
                    )
                    if trial_state is not None:
                        trial_states[elem_id] = trial_state

        if elem_id in deleted_set:
            f_elem = residual_fraction * np.asarray(f_elem, dtype=float)
            if tangent and k_elem is not None:
                k_elem = residual_fraction * np.asarray(k_elem, dtype=float)
            if elem_id in committed_states and (
                not evaluated_deleted_noncurrent
                or not isinstance(committed_states, NonlinearStateStore)
            ):
                trial_states[elem_id] = committed_states[elem_id]
        elif elem_id in element_scales:
            scale = float(element_scales[elem_id])
            f_elem = scale * np.asarray(f_elem, dtype=float)
            if tangent and k_elem is not None:
                k_elem = scale * np.asarray(k_elem, dtype=float)

        np.add.at(F_int, dof_mapping, np.asarray(f_elem, dtype=float))
        if tangent and k_elem is not None:
            data.append(np.asarray(k_elem, dtype=float).ravel())

    if tangent:
        if not data:
            K_T = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
        else:
            K_T = sparse.coo_matrix(
                (np.concatenate(data), (rows_concat, cols_concat)),
                shape=(total_dofs, total_dofs),
                dtype=float,
            ).tocsr()
    else:
        K_T = None

    state_payload = finish_state_evaluation(
        committed_states,
        state_token,
        trial_states,
    )
    record_nonlinear_assembly_execution(
        path="reference_full_coordinate",
        tangent=bool(tangent),
        elapsed_seconds=time.perf_counter() - assembly_start,
        fallback_reason=assembly_fallback_reason,
    )
    return F_int, K_T, state_payload


def _assemble_nonlinear_system(
    model: "FEModel",
    displacements: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    tangent: bool = True,
    deleted_element_ids: Optional[Sequence[int]] = None,
    residual_stiffness_fraction: float = 1.0,
    element_stiffness_scales: Optional[Mapping[int, float]] = None,
    kinematics: str = "von_karman",
    corotational_tangent: str = "rotated",
    require_full_coordinates: bool = False,
    stable_shell_rotation: bool = False,
) -> Tuple[np.ndarray, Any, Dict[int, Any]]:
    """Run one reference assembly as an all-or-nothing state evaluation.

    The reference assembler opens a persistent trial transaction before it
    invokes any element.  An element exception must not strand that token:
    retries, cutbacks, and cancellation cleanup all require the committed
    state to be immediately reusable.
    """

    try:
        F_int, K_T, states = _assemble_nonlinear_system_unchecked(
            model,
            displacements,
            committed_states,
            num_layers,
            tangent=tangent,
            deleted_element_ids=deleted_element_ids,
            residual_stiffness_fraction=residual_stiffness_fraction,
            element_stiffness_scales=element_stiffness_scales,
            kinematics=kinematics,
            corotational_tangent=corotational_tangent,
            require_full_coordinates=require_full_coordinates,
            stable_shell_rotation=stable_shell_rotation,
        )
        from .general_contact.runtime import active_contact_manager

        contact_manager = active_contact_manager()
        if contact_manager is not None:
            contact_force, contact_tangent, _ = contact_manager.evaluate(
                displacements, tangent=tangent
            )
            F_int = np.asarray(F_int, dtype=float) + contact_force
            if tangent:
                K_T = K_T + contact_tangent
        return F_int, K_T, states
    except BaseException:
        _discard_nonlinear_state_candidate(committed_states)
        raise


def _support_reaction_dof_plan(
    model: "FEModel",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Precompute named support DOFs and their six result components."""

    manager = model.mesh.dof_manager
    entries: Dict[str, Tuple[List[int], List[int]]] = {}
    for index, boundary in enumerate(getattr(model, "boundary_conditions", ()) or ()):
        name = str(getattr(boundary, "name", "") or f"support {index + 1}")
        dofs, components = entries.setdefault(name, ([], []))
        indices = getattr(boundary, "_dof_indices", {})
        constraints = getattr(boundary, "dof_constraints", {})
        for node_id in getattr(boundary, "node_ids", ()) or ():
            node_dofs = manager.get_node_dofs(int(node_id))
            for component in constraints:
                local = indices.get(component)
                if local is None or local >= len(node_dofs):
                    continue
                dof = int(node_dofs[local])
                if dof >= 0:
                    dofs.append(dof)
                    components.append(int(local))
    return {
        name: (
            np.asarray(dofs, dtype=np.intp),
            np.asarray(components, dtype=np.intp),
        )
        for name, (dofs, components) in entries.items()
    }


def _support_reaction_resultants(
    model: "FEModel",
    imbalance: np.ndarray,
    *,
    dof_plan: Optional[Mapping[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Tuple[float, ...]]:
    """Sum nonlinear residual forces by named support boundary condition."""

    residual = np.asarray(imbalance, dtype=float).reshape(-1)
    plan = _support_reaction_dof_plan(model) if dof_plan is None else dof_plan
    result: Dict[str, Tuple[float, ...]] = {}
    for name, (dofs, components) in plan.items():
        if dofs.size <= 32:
            values = [0.0] * 6
            for entry_index in range(int(dofs.size)):
                dof = int(dofs[entry_index])
                if dof < residual.size:
                    component = int(components[entry_index])
                    values[component] += float(residual[dof])
            result[name] = tuple(values)
            continue
        valid = dofs < residual.size
        if np.all(valid):
            selected_dofs = dofs
            selected_components = components
        else:
            selected_dofs = dofs[valid]
            selected_components = components[valid]
        vector = np.bincount(
            selected_components,
            weights=residual[selected_dofs],
            minlength=6,
        )[:6]
        result[name] = tuple(float(value) for value in vector)
    return result


def _support_reaction_resultants_from_forces(
    model: "FEModel",
    internal_force: np.ndarray,
    constant_force: np.ndarray,
    proportional_force: np.ndarray,
    load_factor: float,
    *,
    dof_plan: Optional[Mapping[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Tuple[float, ...]]:
    """Recover named support reactions without full residual temporaries."""

    internal = np.asarray(internal_force, dtype=float).reshape(-1)
    constant = np.asarray(constant_force, dtype=float).reshape(-1)
    proportional = np.asarray(proportional_force, dtype=float).reshape(-1)
    available = min(internal.size, constant.size, proportional.size)
    factor = float(load_factor)
    plan = _support_reaction_dof_plan(model) if dof_plan is None else dof_plan
    result: Dict[str, Tuple[float, ...]] = {}
    for name, (dofs, components) in plan.items():
        if dofs.size <= 32:
            values = [0.0] * 6
            for entry_index in range(int(dofs.size)):
                dof = int(dofs[entry_index])
                if dof < available:
                    component = int(components[entry_index])
                    values[component] += float(
                        internal[dof]
                        - (constant[dof] + factor * proportional[dof])
                    )
            result[name] = tuple(values)
            continue
        valid = dofs < available
        if np.all(valid):
            selected_dofs = dofs
            selected_components = components
        else:
            selected_dofs = dofs[valid]
            selected_components = components[valid]
        weights = internal[selected_dofs] - (
            constant[selected_dofs] + factor * proportional[selected_dofs]
        )
        vector = np.bincount(
            selected_components,
            weights=weights,
            minlength=6,
        )[:6]
        result[name] = tuple(float(value) for value in vector)
    return result


def _max_plastic_strain(states: Mapping[int, Any]) -> float:
    if isinstance(states, NonlinearStateStore):
        return states.max_equivalent_plastic_strain()
    alpha_parts: List[np.ndarray] = []
    for state in states.values():
        if not isinstance(state, dict):
            continue
        alpha = np.asarray(state.get("alpha", ()), dtype=float).reshape(-1)
        if alpha.size:
            alpha_parts.append(alpha)
    if not alpha_parts:
        return 0.0
    return float(np.max(np.concatenate(alpha_parts)))


def state_von_mises_envelope(state: Any, E: float, nu: float) -> Optional[float]:
    """Max von Mises stress reconstructed from a committed layered shell state.

    ``sigma = C_el (eps_layer - eps_p)`` reproduces the return-mapped stress
    at the last commit exactly, so the value respects the material hardening
    curve -- unlike an elastic recovery from total displacements, which keeps
    growing with plastic strain.  Returns ``None`` for states without layered
    strain data (e.g. beam fiber states) so callers can fall back to elastic
    recovery.
    """
    if not isinstance(state, Mapping):
        return None
    fiber_stress = np.asarray(state.get("fiber_stress", ()), dtype=float).reshape(-1)
    if fiber_stress.size:
        # Beam fiber sections store the return-mapped uniaxial stress
        # directly; its magnitude is the von Mises equivalent.
        return float(np.max(np.abs(fiber_stress)))
    layer = np.asarray(state.get("layer_strain", ()), dtype=float)
    plastic = np.asarray(state.get("plastic_strain", ()), dtype=float)
    if layer.size == 0 or layer.shape != plastic.shape:
        return None
    layer = layer.reshape(-1, layer.shape[-1]) if layer.ndim > 1 else layer.reshape(-1, 1)
    if layer.shape[-1] != 3:
        return None
    plastic = plastic.reshape(layer.shape)
    elastic = layer - plastic
    factor = float(E) / (1.0 - float(nu) ** 2)
    sxx = factor * (elastic[:, 0] + float(nu) * elastic[:, 1])
    syy = factor * (elastic[:, 1] + float(nu) * elastic[:, 0])
    sxy = factor * (1.0 - float(nu)) / 2.0 * elastic[:, 2]
    von_mises = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
    return float(np.max(von_mises)) if von_mises.size else None


def states_von_mises_map(model: Any, states: Mapping[int, Any]) -> Dict[int, float]:
    """Per-element von Mises envelope from committed plastic states.

    Layered shell states are batched per (material, layer-count) group and
    reduced with one vectorized pass; this runs once per saved transient step
    over every element state, so per-element numpy calls dominate on large
    models.  Values match :func:`state_von_mises_envelope` exactly.
    """
    constants: Dict[str, Tuple[float, float]] = {}
    values: Dict[int, float] = {}
    layered_groups: Dict[Tuple[str, int], Tuple[List[int], List[np.ndarray], List[np.ndarray]]] = {}
    for element_id, state in states.items():
        element = model.mesh.elements.get(int(element_id))
        if element is None or not isinstance(state, Mapping):
            continue
        fiber_stress = np.asarray(state.get("fiber_stress", ()), dtype=float).reshape(-1)
        if fiber_stress.size:
            values[int(element_id)] = float(np.max(np.abs(fiber_stress)))
            continue
        stored_stress = np.asarray(state.get("layer_stress", ()), dtype=float)
        if stored_stress.size:
            try:
                stored_stress = stored_stress.reshape(-1, 3)
            except ValueError:
                stored_stress = np.empty((0, 3), dtype=float)
            if stored_stress.size:
                sxx = stored_stress[:, 0]
                syy = stored_stress[:, 1]
                sxy = stored_stress[:, 2]
                von_mises = np.sqrt(
                    np.maximum(
                        sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy,
                        0.0,
                    )
                )
                values[int(element_id)] = float(np.max(von_mises))
                continue
        layer = np.asarray(state.get("layer_strain", ()), dtype=float)
        plastic = np.asarray(state.get("plastic_strain", ()), dtype=float)
        if layer.size == 0 or layer.shape != plastic.shape:
            continue
        layer = layer.reshape(-1, layer.shape[-1]) if layer.ndim > 1 else layer.reshape(-1, 1)
        if layer.shape[-1] != 3:
            continue
        name = str(element.material_name)
        from .materials import is_isotropic_material

        # Modern orthotropic states always retain converged physical layer
        # stress.  An incomplete legacy/restart state must not be reconstructed
        # with fictitious isotropic constants.
        if not is_isotropic_material(model.get_material(name)):
            continue
        ids, layers, plastics = layered_groups.setdefault((name, layer.shape[0]), ([], [], []))
        ids.append(int(element_id))
        layers.append(layer)
        plastics.append(plastic.reshape(layer.shape))
    for (name, rows), (ids, layers, plastics) in layered_groups.items():
        if name not in constants:
            material = model.get_material(name)
            constants[name] = (float(material.elastic_modulus), float(material.poisson_ratio))
        E, nu = constants[name]
        elastic = np.concatenate(layers) - np.concatenate(plastics)
        factor = E / (1.0 - nu**2)
        sxx = factor * (elastic[:, 0] + nu * elastic[:, 1])
        syy = factor * (elastic[:, 1] + nu * elastic[:, 0])
        sxy = factor * (1.0 - nu) / 2.0 * elastic[:, 2]
        von_mises = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
        per_element = von_mises.reshape(len(ids), rows).max(axis=1)
        for element_id, value in zip(ids, per_element):
            values[element_id] = float(value)
    return values


def _nonlinear_state_summary(states: Dict[int, Any]) -> Dict[str, Any]:
    """Summarize plastic strain and layer/fiber strain data from element states.

    Vectorized: the per-state arrays are gathered once and reduced with a
    handful of concatenated numpy calls.  The previous per-element reductions
    ran once per element per saved transient step and rivalled the cost of
    the Newton solves on large impact models.
    """
    alpha_parts: List[np.ndarray] = []
    plastic_parts: List[np.ndarray] = []
    layer_parts: List[np.ndarray] = []
    fiber_parts: List[np.ndarray] = []
    # (alpha, compression-measure) pairs with row alignment, grouped by the
    # number of in-plane strain columns participating in the compression test.
    comp_layer_pairs: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    comp_fiber_alpha: List[np.ndarray] = []
    comp_fiber_measure: List[np.ndarray] = []

    for state in states.values():
        if not isinstance(state, dict):
            continue
        alpha = np.asarray(state.get("alpha", ()), dtype=float).reshape(-1)
        if alpha.size:
            alpha_parts.append(alpha)
        plastic = np.asarray(state.get("plastic_strain", ()), dtype=float)
        if plastic.size:
            plastic_parts.append(plastic.reshape(-1))
        layer = np.asarray(state.get("layer_strain", ()), dtype=float)
        if layer.size:
            layer_2d = layer.reshape((-1, layer.shape[-1] if layer.ndim > 1 else 1))
            layer_parts.append(layer_2d.reshape(-1))
            if alpha.size == layer_2d.shape[0]:
                width = min(2, layer_2d.shape[1])
                alphas, measures = comp_layer_pairs.setdefault(width, ([], []))
                alphas.append(alpha)
                measures.append(layer_2d[:, :width])
        fiber = np.asarray(state.get("fiber_strain", ()), dtype=float).reshape(-1)
        if fiber.size:
            fiber_parts.append(fiber)
            if alpha.size == fiber.size:
                comp_fiber_alpha.append(alpha)
                comp_fiber_measure.append(fiber)

    max_alpha = 0.0
    yielded = 0
    if alpha_parts:
        alpha_all = np.concatenate(alpha_parts)
        max_alpha = float(np.max(alpha_all))
        sizes = np.asarray([part.size for part in alpha_parts], dtype=np.intp)
        offsets = np.concatenate(([0], np.cumsum(sizes[:-1])))
        yielded = int(np.count_nonzero(np.maximum.reduceat(alpha_all, offsets) > 0.0))
    max_plastic_component = float(np.max(np.abs(np.concatenate(plastic_parts)))) if plastic_parts else 0.0
    layer_min = float("inf")
    layer_max = float("-inf")
    if layer_parts:
        layer_all = np.concatenate(layer_parts)
        layer_min = float(np.min(layer_all))
        layer_max = float(np.max(layer_all))
    fiber_min = float("inf")
    fiber_max = float("-inf")
    if fiber_parts:
        fiber_all = np.concatenate(fiber_parts)
        fiber_min = float(np.min(fiber_all))
        fiber_max = float(np.max(fiber_all))
    max_compressed_alpha = 0.0
    for _width, (alphas, measures) in comp_layer_pairs.items():
        alpha_all = np.concatenate(alphas)
        compression = np.min(np.concatenate(measures), axis=1) < 0.0
        if np.any(compression):
            max_compressed_alpha = max(max_compressed_alpha, float(np.max(alpha_all[compression])))
    if comp_fiber_alpha:
        alpha_all = np.concatenate(comp_fiber_alpha)
        compression = np.concatenate(comp_fiber_measure) < 0.0
        if np.any(compression):
            max_compressed_alpha = max(max_compressed_alpha, float(np.max(alpha_all[compression])))

    return {
        "max_equivalent_plastic_strain": max_alpha,
        "max_plastic_strain_component": max_plastic_component,
        "max_compressed_side_plastic_strain": max_compressed_alpha,
        "layer_strain_min": None if layer_min == float("inf") else layer_min,
        "layer_strain_max": None if layer_max == float("-inf") else layer_max,
        "fiber_strain_min": None if fiber_min == float("inf") else fiber_min,
        "fiber_strain_max": None if fiber_max == float("-inf") else fiber_max,
        "yielded_element_count": yielded,
    }


def _owned_initial_element_states(
    model: "FEModel",
    initial_element_states: Optional[Mapping[int, Any]],
    *,
    _exact_guard: Any,
) -> Optional[Dict[int, Any]]:
    """Detach caller restart states before any model/mechanics observation."""

    if initial_element_states is None:
        return None
    if not isinstance(initial_element_states, Mapping):
        raise TypeError("initial_element_states must be a mapping or None")
    observed_items = tuple(initial_element_states.items())
    _exact_guard(model, context="nonlinear static initial-state mapping observation")
    owned: Dict[int, Any] = {}
    for raw_element_id, state in observed_items:
        if isinstance(raw_element_id, (bool, np.bool_)):
            raise ValueError("initial_element_states IDs must be canonical integers")
        element_id = int(raw_element_id)
        _exact_guard(model, context="nonlinear static initial-state ID observation")
        if element_id in owned:
            raise ValueError("initial_element_states contains duplicate element IDs")
        element = model.mesh.elements.get(element_id)
        if type(element).__module__ == "anyfesolver._ge_beam3_native_generalized_element":
            from ._ge_beam3_native_generalized_restart import capture_solver_state

            owned[element_id] = capture_solver_state(
                model, element_id, state, exact_guard=_exact_guard
            )
            continue
        if type(element).__module__ == "anyfesolver._ge_beam3_native_distributed_element":
            from ._ge_beam3_native_distributed_restart import capture_solver_state

            owned[element_id] = capture_solver_state(
                model, element_id, state, exact_guard=_exact_guard
            )
            continue
        if type(element).__module__ == "anyfesolver._ge_beam3_native_line_static_element":
            from ._ge_beam3_native_line_restart import capture_solver_state

            owned[element_id] = capture_solver_state(
                model, element_id, state, exact_guard=_exact_guard
            )
            continue
        if type(element).__module__ == "anyfesolver._ge_beam3_native_fibre_static_element":
            # Private typed candidate only; the helper checks the exact class.
            # Keep the closed generic copier unchanged for all existing elements.
            from ._ge_beam3_native_fibre_restart import capture_solver_state

            owned[element_id] = capture_solver_state(
                model, element_id, state, exact_guard=_exact_guard
            )
            continue
        owned[element_id] = _guarded_owned_nonlinear_snapshot(
            model,
            state,
            path=f"initial_element_states[{element_id}]",
            _exact_guard=_exact_guard,
        )
    return owned


def _owned_initial_displacements(
    model: "FEModel",
    initial_displacements: Any,
    *,
    _exact_guard: Any,
) -> Optional[np.ndarray]:
    """Capture a bit-exact caller displacement vector under the operation lease."""

    if initial_displacements is None:
        return None
    observed = np.asarray(initial_displacements, dtype=np.float64)
    _exact_guard(model, context="nonlinear static initial-displacement observation")
    contiguous = np.ascontiguousarray(observed, dtype=np.float64)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(contiguous.shape)


def _copy_initial_states(initial_element_states: Optional[Mapping[int, Any]]) -> Dict[int, Any]:
    return copy.deepcopy(dict(initial_element_states or {}))


def _qualified_q4_committed_state_hooks(element: Any) -> Optional[Tuple[Any, Any]]:
    """Return the closed Q4 state hooks without importing formulation code.

    The nonlinear driver deliberately discovers this private, formulation-
    versioned protocol from the element.  That keeps legacy Q4 and qualified
    S3 state lifecycles unchanged and avoids putting changing displacement
    seals into Newton's packed constitutive-state buffers.
    """

    schema = str(getattr(element, "current_state_binding_schema_id", ""))
    seal = getattr(element, "seal_committed_current_tangent_state", None)
    validate = getattr(element, "validate_committed_current_tangent_state", None)
    if (
        schema.startswith("E4_PL_Q4_COMMITTED_STATE_")
        and callable(seal)
        and callable(validate)
    ):
        return seal, validate
    return None


def _qualified_s3_committed_state_validator(element: Any) -> Optional[Any]:
    """Return the formulation-owned S3 committed-state validator."""

    formulation = str(getattr(element, "formulation_id", ""))
    validator = getattr(element, "validate_model_bound_nonlinear_state", None)
    if formulation in {
        "E4_PL_QUALIFIED_S3_COMPANION_V1",
        "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
    } and callable(validator):
        return validator
    return None


def _is_explicit_q4_initial_material_seed(state: Mapping[str, Any]) -> bool:
    """Whether an unbound payload is a material seed rather than a restart.

    The public virgin layered state consists only of plastic history arrays;
    callers have historically been allowed to seed those arrays before a
    zero-displacement analysis.  Recovery/kinematic/origin fields identify a
    previously evaluated committed result and therefore still require an
    exact displacement binding.
    """

    allowed = {
        "plastic_strain",
        "alpha",
        *_INITIAL_FIELD_STATE_KEYS,
        "initial_field_provenance",
    }
    keys = {str(key) for key in state}
    return {"plastic_strain", "alpha"}.issubset(keys) and keys <= allowed


def _prepare_qualified_q4_states_for_nonlinear_solve(
    model: "FEModel",
    displacements: np.ndarray,
    element_states: Mapping[int, Any],
    num_layers: int,
    info: Dict[str, Any],
    *,
    supplied_element_ids: Sequence[int],
    ordinary_restart: bool,
    kinematics: str = "von_karman",
    allow_explicit_initial_material_states: bool = False,
    expected_deleted_element_ids: Sequence[int] = (),
    deletion_records: Sequence[Any] = (),
    residual_stiffness_fraction: Optional[float] = None,
    deleted_dispositions: Optional[Dict[int, Any]] = None,
) -> Dict[int, Any]:
    """Validate supplied Q4 seals, then remove them from an owned copy.

    A Q4 seal binds a *committed* constitutive payload to one exact owner
    displacement/configuration (reference deformation for corotational Q4).
    It must therefore be checked before the
    first nonlinear element evaluation, but it must not travel through the
    packed trial-state machinery.  Historical Q4 states that predate the seal
    are accepted only on the ordinary restart path where an exact committed
    displacement is supplied.  They are identified explicitly and receive a
    fresh seal on the returned final state.
    """

    full = np.asarray(displacements, dtype=np.float64)
    if (
        full.ndim != 1
        or full.size != int(model.mesh.dof_manager.total_dofs)
        or not np.all(np.isfinite(full))
    ):
        raise ValueError(
            "qualified Q4 state preflight requires the complete finite "
            "committed displacement"
        )
    supplied = {int(value) for value in supplied_element_ids}
    expected_deleted = {int(value) for value in expected_deleted_element_ids}
    records_by_id = {
        int(record.element_id): record
        for record in deletion_records
    }
    if set(records_by_id) != expected_deleted:
        raise ValueError(
            "qualified Q4 deleted IDs and deletion records are inconsistent"
        )
    # This second deep copy is intentional.  _prepare_initial_states currently
    # owns its result too, but the binding boundary must remain safe if that
    # implementation changes or a future state-store adapter calls us directly.
    prepared = copy.deepcopy(dict(element_states))
    validated: List[int] = []
    migrated: List[int] = []
    stripped: List[int] = []
    fresh: List[int] = []
    explicit_seeds: List[int] = []
    restored_deleted: List[int] = []

    for raw_element_id, element in sorted(
        model.mesh.elements.items(), key=lambda item: int(item[0])
    ):
        hooks = _qualified_q4_committed_state_hooks(element)
        if hooks is None:
            continue
        element_id = int(raw_element_id)
        if element_id not in prepared:
            continue
        state = prepared[element_id]
        if state is None:
            # A caller-supplied None is a virgin state, not historical material
            # history.  The final materializer will create and seal it.
            continue
        if not isinstance(state, Mapping):
            raise TypeError(
                f"qualified Q4 state for element {element_id} must be a mapping"
            )
        has_binding = _Q4_COMMITTED_BINDING_KEY in state
        has_integrity = _Q4_COMMITTED_INTEGRITY_KEY in state
        has_disposition = _Q4_ACTIVITY_DISPOSITION_KEY in state
        if has_binding != has_integrity:
            raise ValueError(
                f"qualified Q4 state for element {element_id} has a partial "
                "committed-state binding"
            )

        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        if dofs.shape != (24,):
            raise ValueError(
                f"qualified Q4 element {element_id} must map exactly 24 DOFs"
            )
        local_u = np.asarray(full[dofs], dtype=np.float64)
        if has_binding and kinematics == "corotational":
            from .corotational import _corotational_cache, _corotational_deformation_state

            reference = _corotational_cache(model)[element_id]
            local_u, _, _ = _corotational_deformation_state(reference, element, local_u)
        if has_disposition:
            if not has_binding or element_id not in expected_deleted:
                raise ValueError(
                    f"qualified Q4 noncurrent disposition for element "
                    f"{element_id} disagrees with checkpoint deletion state"
                )
            validate_deleted = getattr(
                element,
                "validate_noncurrent_deleted_state",
                None,
            )
            if not callable(validate_deleted):
                raise ValueError(
                    f"qualified Q4 element {element_id} cannot validate its "
                    "deleted/frozen disposition"
                )
            record = records_by_id[element_id]
            validate_deleted(
                model.mesh,
                model.get_material(element.material_name),
                state,
                int(num_layers),
                expected_deletion_step_index=int(record.step_index),
                expected_deletion_load_factor=float(record.load_factor),
                expected_residual_stiffness_fraction=(
                    residual_stiffness_fraction
                ),
                expected_trigger_name=str(record.trigger_name),
            )
            if deleted_dispositions is not None:
                deleted_dispositions[element_id] = copy.deepcopy(
                    state[_Q4_ACTIVITY_DISPOSITION_KEY]
                )
            restored_deleted.append(element_id)
        elif element_id in expected_deleted:
            raise ValueError(
                f"deleted qualified Q4 element {element_id} lacks its "
                "frozen noncurrent disposition"
            )
        elif has_binding:
            _seal, validate = hooks
            validate(
                model.mesh,
                model.get_material(element.material_name),
                local_u,
                state,
                int(num_layers),
            )
            validated.append(element_id)
        elif element_id in supplied:
            if (
                allow_explicit_initial_material_states
                and _is_explicit_q4_initial_material_seed(state)
            ):
                explicit_seeds.append(element_id)
            elif not ordinary_restart:
                raise ValueError(
                    f"historical unbound qualified Q4 state for element "
                    f"{element_id} requires an exact ordinary-restart "
                    "displacement and equilibrate_initial_state=False"
                )
            else:
                migrated.append(element_id)
        else:
            # State freshly constructed from an initial-field declaration.
            fresh.append(element_id)

        owned_state = copy.deepcopy(dict(state))
        if has_binding:
            owned_state.pop(_Q4_COMMITTED_BINDING_KEY, None)
            owned_state.pop(_Q4_COMMITTED_INTEGRITY_KEY, None)
            stripped.append(element_id)
        owned_state.pop(_Q4_ACTIVITY_DISPOSITION_KEY, None)
        # Older experimental records sometimes mirrored the whole-state
        # digest under this cache name.  It is never trial-state input, even
        # when the two closed binding fields are absent during migration.
        owned_state.pop("state_digest", None)
        prepared[element_id] = owned_state

    s3_validated: List[int] = []
    s3_restored_deleted: List[int] = []
    s3_elements_present = False
    for raw_element_id, element in sorted(
        model.mesh.elements.items(), key=lambda item: int(item[0])
    ):
        validate_active = _qualified_s3_committed_state_validator(element)
        if validate_active is None:
            continue
        s3_elements_present = True
        element_id = int(raw_element_id)
        if element_id not in prepared:
            if element_id in expected_deleted:
                raise ValueError(
                    f"deleted qualified S3 element {element_id} lacks its "
                    "frozen noncurrent state"
                )
            continue
        state = prepared[element_id]
        if not isinstance(state, Mapping):
            raise TypeError(
                f"qualified S3 state for element {element_id} must be a mapping"
            )
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        if dofs.shape != (18,):
            raise ValueError(
                f"qualified S3 element {element_id} must map exactly 18 DOFs"
            )
        local_u = np.asarray(full[dofs], dtype=np.float64)
        has_disposition = state.get(_S3_ACTIVITY_DISPOSITION_KEY) is not None
        if has_disposition:
            if element_id not in expected_deleted:
                raise ValueError(
                    f"qualified S3 noncurrent disposition for element "
                    f"{element_id} disagrees with checkpoint deletion state"
                )
            record = records_by_id[element_id]
            validate_deleted = getattr(
                element, "validate_noncurrent_deleted_state", None
            )
            restore_deleted = getattr(
                element,
                "restore_noncurrent_deleted_state_for_internal_use",
                None,
            )
            if not callable(validate_deleted) or not callable(restore_deleted):
                raise ValueError(
                    f"qualified S3 element {element_id} cannot restore its "
                    "deleted/frozen state"
                )
            material = model.get_material(element.material_name)
            expected_keywords = {
                "expected_deletion_step_index": int(record.step_index),
                "expected_deletion_load_factor": float(record.load_factor),
                "expected_residual_stiffness_fraction": (
                    residual_stiffness_fraction
                ),
                "expected_trigger_name": str(record.trigger_name),
            }
            validate_deleted(
                model.mesh,
                material,
                state,
                int(num_layers),
                **expected_keywords,
            )
            if deleted_dispositions is not None:
                deleted_dispositions[element_id] = copy.deepcopy(
                    state[_S3_ACTIVITY_DISPOSITION_KEY]
                )
            prepared[element_id] = restore_deleted(
                model.mesh,
                material,
                state,
                int(num_layers),
                **expected_keywords,
            )
            s3_restored_deleted.append(element_id)
        elif element_id in expected_deleted:
            raise ValueError(
                f"deleted qualified S3 element {element_id} lacks its "
                "frozen noncurrent disposition"
            )
        else:
            validate_active(
                model.mesh,
                model.get_material(element.material_name),
                state,
                int(num_layers),
                expected_committed_total_u=local_u,
            )
            s3_validated.append(element_id)

    previous = info.get("qualified_q4_committed_state_lifecycle", {})

    def cumulative_ids(name: str, values: Sequence[int]) -> List[int]:
        prior = previous.get(name, ()) if isinstance(previous, Mapping) else ()
        return sorted({int(value) for value in (*prior, *values)})

    info["qualified_q4_committed_state_lifecycle"] = {
        "schema": _Q4_COMMITTED_STATE_LIFECYCLE_SCHEMA,
        "ordinary_restart": bool(ordinary_restart),
        "preflight_count": int(previous.get("preflight_count", 0)) + 1
        if isinstance(previous, Mapping)
        else 1,
        "validated_bound_element_ids": cumulative_ids(
            "validated_bound_element_ids", validated
        ),
        "migrated_historical_unbound_element_ids": cumulative_ids(
            "migrated_historical_unbound_element_ids", migrated
        ),
        "stripped_internal_binding_element_ids": cumulative_ids(
            "stripped_internal_binding_element_ids", stripped
        ),
        "fresh_unbound_element_ids": cumulative_ids(
            "fresh_unbound_element_ids", fresh
        ),
        "explicit_initial_material_state_element_ids": cumulative_ids(
            "explicit_initial_material_state_element_ids", explicit_seeds
        ),
        "restored_deleted_frozen_element_ids": cumulative_ids(
            "restored_deleted_frozen_element_ids", restored_deleted
        ),
        "input_states_deep_copied": True,
        "final_state_policy": "DETERMINISTIC_RESEAL_AFTER_MATERIALIZATION",
    }
    if s3_elements_present:
        previous_s3 = info.get("qualified_s3_committed_state_lifecycle", {})

        def cumulative_s3_ids(name: str, values: Sequence[int]) -> List[int]:
            prior = (
                previous_s3.get(name, ())
                if isinstance(previous_s3, Mapping)
                else ()
            )
            return sorted({int(value) for value in (*prior, *values)})

        info["qualified_s3_committed_state_lifecycle"] = {
            "schema": _S3_COMMITTED_STATE_LIFECYCLE_SCHEMA,
            "ordinary_restart": bool(ordinary_restart),
            "preflight_count": int(previous_s3.get("preflight_count", 0)) + 1
            if isinstance(previous_s3, Mapping)
            else 1,
            "validated_active_element_ids": cumulative_s3_ids(
                "validated_active_element_ids", s3_validated
            ),
            "restored_deleted_frozen_element_ids": cumulative_s3_ids(
                "restored_deleted_frozen_element_ids", s3_restored_deleted
            ),
            "input_states_deep_copied": True,
            "final_state_policy": (
                "ACTIVE_ONLY_OR_EXPLICIT_NONCURRENT_DISPOSITION"
            ),
        }
    return prepared


def _seal_final_qualified_q4_states(
    model: "FEModel",
    displacements: np.ndarray,
    element_states: Mapping[int, Any],
    num_layers: int,
    info: Optional[Dict[str, Any]] = None,
    *,
    kinematics: str = "von_karman",
    seal_corotational_checkpoint: bool = False,
    deleted_element_ids: Sequence[int] = (),
    deleted_dispositions: Optional[Mapping[int, Any]] = None,
    deletion_records: Sequence[Any] = (),
    residual_stiffness_fraction: Optional[float] = None,
) -> Dict[int, Any]:
    """Seal finalized Q4 states in their constitutive owner's coordinates.

    Corotational sealing is opt-in at the displacement checkpoint boundary;
    ordinary returned histories retain their established lifecycle.
    """

    states = dict(element_states)
    normalized_kinematics = str(kinematics)
    has_solver_integrated_s3 = any(
        _qualified_s3_committed_state_validator(element) is not None
        and callable(
            getattr(element, "seal_solver_integrated_nonlinear_state", None)
        )
        for element in model.mesh.elements.values()
    )
    corotational_checkpoint = (
        normalized_kinematics == "corotational" and seal_corotational_checkpoint
    )
    if normalized_kinematics != "von_karman" and not has_solver_integrated_s3 and not corotational_checkpoint:
        return states
    full = np.asarray(displacements, dtype=np.float64)
    if (
        full.ndim != 1
        or full.size != int(model.mesh.dof_manager.total_dofs)
        or not np.all(np.isfinite(full))
    ):
        raise ValueError(
            "qualified Q4 finalization requires the complete finite displacement"
        )
    sealed_ids: List[int] = []
    deleted_sealed_ids: List[int] = []
    origin_materialized_ids: List[int] = []
    digests: List[Dict[str, Any]] = []
    s3_sealed_ids: List[int] = []
    s3_deleted_sealed_ids: List[int] = []
    s3_digests: List[Dict[str, Any]] = []
    s3_elements_present = False
    deleted = {int(value) for value in deleted_element_ids}
    disposition_by_id = {
        int(key): copy.deepcopy(value)
        for key, value in (deleted_dispositions or {}).items()
    }
    record_by_id = {int(record.element_id): record for record in deletion_records}
    for raw_element_id, element in sorted(
        model.mesh.elements.items(), key=lambda item: int(item[0])
    ):
        if normalized_kinematics != "von_karman" and not corotational_checkpoint:
            break
        hooks = _qualified_q4_committed_state_hooks(element)
        if hooks is None:
            continue
        element_id = int(raw_element_id)
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        if dofs.shape != (24,):
            raise ValueError(
                f"qualified Q4 element {element_id} must map exactly 24 DOFs"
            )
        local_u = np.asarray(full[dofs], dtype=np.float64)
        if corotational_checkpoint:
            from .corotational import _corotational_cache, _corotational_deformation_state

            reference = _corotational_cache(model)[element_id]
            local_u, _, _ = _corotational_deformation_state(reference, element, local_u)
        state = states.get(element_id)
        if element_id in deleted:
            if state is None or element_id not in disposition_by_id:
                raise ValueError(
                    f"deleted qualified Q4 element {element_id} lacks its "
                    "frozen-state lifecycle record"
                )
            record = record_by_id.get(element_id)
            if record is None:
                raise ValueError(
                    f"deleted qualified Q4 element {element_id} lacks its "
                    "deletion authority record"
                )
            disposition = disposition_by_id[element_id]
            deletion_u = np.asarray(
                disposition.get("accepted_local_u", ()), dtype=np.float64
            )
            seal_deleted = getattr(element, "seal_noncurrent_deleted_state", None)
            validate_deleted = getattr(
                element, "validate_noncurrent_deleted_state", None
            )
            if not callable(seal_deleted) or not callable(validate_deleted):
                raise ValueError(
                    f"qualified Q4 element {element_id} lacks deleted-state hooks"
                )
            material = model.get_material(element.material_name)
            sealed = seal_deleted(
                model.mesh,
                material,
                deletion_u,
                state,
                int(num_layers),
                deletion_step_index=int(record.step_index),
                deletion_load_factor=float(record.load_factor),
                residual_stiffness_fraction=float(
                    residual_stiffness_fraction
                ),
                trigger_name=str(record.trigger_name),
            )
            digest = validate_deleted(
                model.mesh,
                material,
                sealed,
                int(num_layers),
                expected_deletion_step_index=int(record.step_index),
                expected_deletion_load_factor=float(record.load_factor),
                expected_residual_stiffness_fraction=float(
                    residual_stiffness_fraction
                ),
                expected_trigger_name=str(record.trigger_name),
            )
            states[element_id] = sealed
            deleted_sealed_ids.append(element_id)
            digests.append(
                {"element_id": element_id, "state_integrity_sha256": digest}
            )
            continue
        if state is None:
            # Fully constrained and zero-step/stopped solves can legitimately
            # have no Newton candidate.  Recover their result state once at
            # the accepted displacement; this is outside the iteration path.
            # Request the tangent too so plastic-capable Q4 records the virgin
            # discrete-update origin required by the committed-state seal.
            material = model.get_material(element.material_name)
            _force, _tangent, recovered = element.compute_nonlinear_response(
                model.mesh,
                material,
                local_u,
                None,
                int(num_layers),
                True,
            )
            state = recovered
        if not isinstance(state, Mapping):
            raise TypeError(
                f"qualified Q4 finalized state for element {element_id} "
                "must be a mapping"
            )
        seal, validate = hooks
        material = model.get_material(element.material_name)
        requires_origin = getattr(
            element, "_requires_algorithmic_return_map_origin", None
        )
        if (
            "qualified_q4_algorithmic_origin" not in state
            and callable(requires_origin)
            and bool(requires_origin(material))
        ):
            if not _is_explicit_q4_initial_material_seed(state):
                raise ValueError(
                    f"qualified Q4 finalized plastic state for element "
                    f"{element_id} lacks its accepted algorithmic origin and "
                    "is not an unevaluated initial material parent"
                )
            # The only safe reconstruction is from an unevaluated explicit
            # parent.  A residual-only output contains derived/recovery fields
            # and is deliberately rejected above: replaying from that output
            # would perform a second return map from converged history.
            _force, tangent_matrix, tangent_state = (
                element.compute_nonlinear_response(
                    model.mesh,
                    material,
                    local_u,
                    state,
                    int(num_layers),
                    True,
                )
            )
            if tangent_matrix is None or not isinstance(tangent_state, Mapping):
                raise ValueError(
                    f"qualified Q4 initial material parent for element "
                    f"{element_id} did not produce a tangent-evaluated state"
                )
            state = tangent_state
            origin_materialized_ids.append(element_id)
        solver_finalizer = getattr(
            element,
            "_seal_solver_finalized_current_tangent_state",
            None,
        )
        sealing_hook = solver_finalizer if callable(solver_finalizer) else seal
        sealed = sealing_hook(
            model.mesh,
            material,
            local_u,
            state,
            int(num_layers),
        )
        digest = validate(
            model.mesh,
            material,
            local_u,
            sealed,
            int(num_layers),
        )
        states[element_id] = sealed
        sealed_ids.append(element_id)
        digests.append({"element_id": element_id, "state_integrity_sha256": digest})

    for raw_element_id, element in sorted(
        model.mesh.elements.items(), key=lambda item: int(item[0])
    ):
        validate_active = _qualified_s3_committed_state_validator(element)
        if validate_active is None:
            continue
        s3_elements_present = True
        element_id = int(raw_element_id)
        solver_sealer = getattr(
            element, "seal_solver_integrated_nonlinear_state", None
        )
        if normalized_kinematics != "von_karman" and not callable(
            solver_sealer
        ):
            continue
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        if dofs.shape != (18,):
            raise ValueError(
                f"qualified S3 element {element_id} must map exactly 18 DOFs"
            )
        local_u = np.asarray(full[dofs], dtype=np.float64)
        state = states.get(element_id)
        material = model.get_material(element.material_name)
        if (
            state is None
            and element_id not in deleted
            and normalized_kinematics == "von_karman"
            and type(element) is _NativeParityE4PLS3V2DShellElement
        ):
            _force, _tangent, state = element.compute_nonlinear_response(
                model.mesh, material, local_u, None, int(num_layers), True,
                von_karman=True,
            )
        if element_id in deleted:
            if state is None or element_id not in disposition_by_id:
                raise ValueError(
                    f"deleted qualified S3 element {element_id} lacks its "
                    "frozen-state lifecycle record"
                )
            record = record_by_id.get(element_id)
            if record is None:
                raise ValueError(
                    f"deleted qualified S3 element {element_id} lacks its "
                    "deletion authority record"
                )
            disposition = disposition_by_id[element_id]
            deletion_u = np.asarray(
                disposition.get("accepted_local_u", ()), dtype=np.float64
            )
            seal_deleted = getattr(element, "seal_noncurrent_deleted_state", None)
            validate_deleted = getattr(
                element, "validate_noncurrent_deleted_state", None
            )
            if not callable(seal_deleted) or not callable(validate_deleted):
                raise ValueError(
                    f"qualified S3 element {element_id} lacks deleted-state hooks"
                )
            sealed = seal_deleted(
                model.mesh,
                material,
                deletion_u,
                state,
                int(num_layers),
                deletion_step_index=int(record.step_index),
                deletion_load_factor=float(record.load_factor),
                residual_stiffness_fraction=float(
                    residual_stiffness_fraction
                ),
                trigger_name=str(record.trigger_name),
            )
            digest = validate_deleted(
                model.mesh,
                material,
                sealed,
                int(num_layers),
                expected_deletion_step_index=int(record.step_index),
                expected_deletion_load_factor=float(record.load_factor),
                expected_residual_stiffness_fraction=float(
                    residual_stiffness_fraction
                ),
                expected_trigger_name=str(record.trigger_name),
            )
            states[element_id] = sealed
            s3_deleted_sealed_ids.append(element_id)
            s3_digests.append(
                {"element_id": element_id, "state_integrity_sha256": digest}
            )
            continue
        if not isinstance(state, Mapping):
            raise TypeError(
                f"qualified S3 finalized state for element {element_id} "
                "must be a mapping"
            )
        if callable(solver_sealer):
            validated = solver_sealer(
                model.mesh,
                material,
                state,
                int(num_layers),
                local_u,
                kinematics=str(kinematics),
            )
        else:
            validated = validate_active(
                model.mesh,
                material,
                state,
                int(num_layers),
                expected_committed_total_u=local_u,
            )
        states[element_id] = validated
        s3_sealed_ids.append(element_id)
        s3_digests.append(
            {
                "element_id": element_id,
                "state_integrity_sha256": str(
                    validated["state_integrity_sha256"]
                ),
            }
        )

    if info is not None:
        lifecycle = dict(
            info.get(
                "qualified_q4_committed_state_lifecycle",
                {
                    "schema": _Q4_COMMITTED_STATE_LIFECYCLE_SCHEMA,
                    "ordinary_restart": False,
                    "preflight_count": 0,
                    "validated_bound_element_ids": [],
                    "migrated_historical_unbound_element_ids": [],
                    "stripped_internal_binding_element_ids": [],
                    "fresh_unbound_element_ids": [],
                    "input_states_deep_copied": True,
                    "final_state_policy": (
                        "DETERMINISTIC_RESEAL_AFTER_MATERIALIZATION"
                    ),
                },
            )
        )
        lifecycle["sealed_final_element_ids"] = sealed_ids
        lifecycle["sealed_deleted_frozen_element_ids"] = deleted_sealed_ids
        lifecycle[
            "origin_materialized_from_unevaluated_parent_element_ids"
        ] = origin_materialized_ids
        lifecycle["sealed_final_state_digests"] = digests
        info["qualified_q4_committed_state_lifecycle"] = lifecycle
        if s3_elements_present:
            s3_lifecycle = dict(
                info.get(
                    "qualified_s3_committed_state_lifecycle",
                    {
                        "schema": _S3_COMMITTED_STATE_LIFECYCLE_SCHEMA,
                        "ordinary_restart": False,
                        "preflight_count": 0,
                        "validated_active_element_ids": [],
                        "restored_deleted_frozen_element_ids": [],
                        "input_states_deep_copied": True,
                        "final_state_policy": (
                            "ACTIVE_ONLY_OR_EXPLICIT_NONCURRENT_DISPOSITION"
                        ),
                    },
                )
            )
            s3_lifecycle["sealed_active_element_ids"] = s3_sealed_ids
            s3_lifecycle[
                "sealed_deleted_frozen_element_ids"
            ] = s3_deleted_sealed_ids
            s3_lifecycle["sealed_final_state_digests"] = s3_digests
            info["qualified_s3_committed_state_lifecycle"] = s3_lifecycle
    return states


def _mark_failed_qualified_q4_states(
    model: "FEModel",
    displacements: np.ndarray,
    element_states: Mapping[int, Any],
    num_layers: int,
    info: Dict[str, Any],
    *,
    failure_reason: str,
    kinematics: str,
) -> Dict[int, Any]:
    """Own failed Q4 output while denying it ACTIVE current-state authority."""

    states = dict(element_states)
    normalized_kinematics = str(kinematics)
    has_solver_integrated_s3 = any(
        _qualified_s3_committed_state_validator(element) is not None
        and callable(getattr(element, "mark_noncurrent_failed_state", None))
        for element in model.mesh.elements.values()
    )
    if normalized_kinematics != "von_karman" and not has_solver_integrated_s3:
        return states
    full = np.asarray(displacements, dtype=np.float64)
    marked_ids: List[int] = []
    s3_marked_ids: List[int] = []
    for raw_element_id, element in sorted(
        model.mesh.elements.items(), key=lambda item: int(item[0])
    ):
        if normalized_kinematics != "von_karman":
            break
        if _qualified_q4_committed_state_hooks(element) is None:
            continue
        element_id = int(raw_element_id)
        state = states.get(element_id)
        if state is None:
            initializer = getattr(element, "init_nonlinear_state", None)
            if callable(initializer):
                state = initializer(int(num_layers))
        if not isinstance(state, Mapping):
            continue
        marker = getattr(element, "mark_noncurrent_failed_state", None)
        if not callable(marker):
            raise ValueError(
                f"qualified Q4 element {element_id} lacks failed-state lifecycle"
            )
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        states[element_id] = marker(
            model.mesh,
            model.get_material(element.material_name),
            np.asarray(full[dofs], dtype=np.float64),
            state,
            int(num_layers),
            failure_reason=str(failure_reason),
        )
        marked_ids.append(element_id)
    for raw_element_id, element in sorted(
        model.mesh.elements.items(), key=lambda item: int(item[0])
    ):
        if _qualified_s3_committed_state_validator(element) is None:
            continue
        element_id = int(raw_element_id)
        material = model.get_material(element.material_name)
        state = states.get(element_id)
        if state is None:
            initializer = getattr(
                element, "init_model_bound_nonlinear_state", None
            )
            if callable(initializer):
                state = initializer(
                    model.mesh,
                    material,
                    int(num_layers),
                )
        if not isinstance(state, Mapping):
            continue
        marker = getattr(element, "mark_noncurrent_failed_state", None)
        if not callable(marker):
            if normalized_kinematics != "von_karman":
                continue
            raise ValueError(
                f"qualified S3 element {element_id} lacks failed-state lifecycle"
            )
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        if dofs.shape != (18,):
            raise ValueError(
                f"qualified S3 element {element_id} must map exactly 18 DOFs"
            )
        states[element_id] = marker(
            model.mesh,
            material,
            np.asarray(full[dofs], dtype=np.float64),
            state,
            int(num_layers),
            failure_reason=str(failure_reason),
        )
        s3_marked_ids.append(element_id)
    lifecycle = dict(
        info.get(
            "qualified_q4_committed_state_lifecycle",
            {"schema": _Q4_COMMITTED_STATE_LIFECYCLE_SCHEMA},
        )
    )
    lifecycle["failed_nonauthoritative_element_ids"] = marked_ids
    lifecycle["final_state_policy"] = "FAILED_NONAUTHORITATIVE_NO_ACTIVE_SEAL"
    info["qualified_q4_committed_state_lifecycle"] = lifecycle
    if s3_marked_ids:
        s3_lifecycle = dict(
            info.get(
                "qualified_s3_committed_state_lifecycle",
                {"schema": _S3_COMMITTED_STATE_LIFECYCLE_SCHEMA},
            )
        )
        s3_lifecycle["failed_nonauthoritative_element_ids"] = s3_marked_ids
        s3_lifecycle[
            "final_state_policy"
        ] = "FAILED_NONAUTHORITATIVE_NO_ACTIVE_SEAL"
        info["qualified_s3_committed_state_lifecycle"] = s3_lifecycle
    return states


def _state_has_initial_field(state: Any) -> bool:
    return state_has_active_initial_fields(state, _INITIAL_FIELD_STATE_KEYS)


def _initial_value_summary(value: Any) -> Dict[str, Any]:
    array = np.asarray(value, dtype=float)
    return {
        "shape": list(array.shape),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _coerce_initial_field(value: Any) -> Union[ShellInitialField, BeamInitialField]:
    if isinstance(value, (ShellInitialField, BeamInitialField)):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Initial fields must be ShellInitialField, BeamInitialField, or mappings")
    data = dict(value)
    kind = str(data.pop("kind", "")).strip().lower()
    shell_names = {"membrane_stress", "bending_stress", "membrane_prestrain", "curvature_prestrain"}
    beam_names = {"fiber_stress", "fiber_prestrain"}
    supplied = set(data) - {"source"}
    if kind == "shell" or (not kind and supplied and supplied <= shell_names):
        return ShellInitialField(**data)
    if kind == "beam" or (not kind and supplied and supplied <= beam_names):
        return BeamInitialField(**data)
    raise ValueError(
        "Initial-field mapping requires kind='shell' or kind='beam', or unambiguous field names"
    )


def _initialize_missing_model_bound_states(
    model: "FEModel",
    states: Dict[int, Any],
    num_layers: int,
) -> None:
    """Materialize virgin native states before the shared-SO(3) store.

    Legacy elements deliberately retain their established lazy ``state=None``
    path.  A formulation-native element whose state is model-bound cannot do
    that: the node-shared rotation store must bind its reference directors and
    identity before the first trial begins.
    """

    for raw_element_id, element in model.mesh.elements.items():
        element_id = int(raw_element_id)
        if element_id in states:
            continue
        initializer = getattr(
            element,
            "init_model_bound_nonlinear_state",
            None,
        )
        if not (
            bool(
                getattr(
                    element,
                    "formulation_native_total_lagrangian",
                    False,
                )
            )
            and callable(initializer)
        ):
            continue
        initialized = initializer(
            model.mesh,
            model.get_material(element.material_name),
            num_layers,
        )
        if not isinstance(initialized, Mapping):
            raise TypeError(
                f"Model-bound initializer for element {element_id} must "
                "return a mapping"
            )
        states[element_id] = initialized


def _prepare_initial_states(
    model: "FEModel",
    initial_element_states: Optional[Mapping[int, Any]],
    initial_fields: Optional[Mapping[int, Any]],
    num_layers: int,
) -> Tuple[Dict[int, Any], List[Dict[str, Any]]]:
    """Merge prescribed residual fields into committed element state.

    Initial fields remain immutable state offsets while plastic strain and
    hardening variables are committed normally.  This separation prevents a
    geometric imperfection from being mislabeled as residual stress and lets
    restart states retain the manufacturing-field provenance.
    """
    states = _copy_initial_states(initial_element_states)
    provenance: List[Dict[str, Any]] = []
    for raw_element_id, state in states.items():
        if not _state_has_initial_field(state):
            continue
        stored = state.get("initial_field_provenance", {}) if isinstance(state, Mapping) else {}
        components = {
            key.removeprefix("initial_"): _initial_value_summary(state[key])
            for key in _INITIAL_FIELD_STATE_KEYS
            if key in state
        }
        provenance.append(
            {
                "element_id": int(raw_element_id),
                "kind": str(stored.get("kind", "unknown")),
                "source": str(stored.get("source", "restart_state")),
                "components": components,
                "restored_from_committed_state": True,
            }
        )
    if not initial_fields:
        _initialize_missing_model_bound_states(model, states, num_layers)
        return states, provenance

    from .elements import (
        BeamElement,
        QuadraticBeamElement,
        ShellElement,
        validate_initial_field_state,
    )

    for raw_element_id, raw_field in initial_fields.items():
        element_id = int(raw_element_id)
        provenance = [
            entry for entry in provenance
            if int(entry["element_id"]) != element_id
        ]
        element = model.mesh.get_element(element_id)
        if element is None:
            raise ValueError(f"Initial field references missing element {element_id}")
        field_value = _coerce_initial_field(raw_field)
        previous_state = states.get(element_id)
        if isinstance(previous_state, Mapping):
            previous_plastic = np.asarray(
                previous_state.get("plastic_strain", ()),
                dtype=float,
            )
            previous_alpha = np.asarray(
                previous_state.get("alpha", ()),
                dtype=float,
            )
            if (
                previous_plastic.size
                and np.any(np.abs(previous_plastic) > 1.0e-14)
            ) or (
                previous_alpha.size
                and np.any(previous_alpha > 1.0e-14)
            ):
                raise ValueError(
                    f"Cannot superpose a new initial field on nonzero plastic history for "
                    f"element {element_id}; restart with the already-committed field state instead."
                )
        if isinstance(field_value, ShellInitialField):
            if not isinstance(element, ShellElement):
                raise TypeError(f"ShellInitialField requires a shell element; {element_id} is {type(element).__name__}")
            state = states.get(element_id)
            if state is not None and not isinstance(state, Mapping):
                raise TypeError(f"Initial state for shell element {element_id} must be a mapping")
            values = field_value.state_values()
            field_type = "shell"
        else:
            if not isinstance(element, (BeamElement, QuadraticBeamElement)):
                raise TypeError(f"BeamInitialField requires a beam element; {element_id} is {type(element).__name__}")
            if getattr(element, "_fiber_plasticity_config")(model.get_material(element.material_name)) is None:
                raise ValueError(
                    f"Beam initial fiber fields require fiber plasticity on element {element_id}"
                )
            state = states.get(element_id)
            if state is None:
                state = {}
            elif not isinstance(state, Mapping):
                raise TypeError(f"Initial state for beam element {element_id} must be a mapping")
            state = copy.deepcopy(dict(state))
            values = field_value.state_values()
            field_type = "beam"
        field_provenance = {
            "kind": field_type,
            "source": field_value.source,
            "components": sorted(values),
        }
        for key, array in values.items():
            if array.size == 0 or np.any(~np.isfinite(array)):
                raise ValueError(
                    f"{key} for element {element_id} must contain finite values"
                )
        initialized_atomically = False
        if isinstance(field_value, ShellInitialField):
            model_bound_initializer = getattr(
                element,
                "init_model_bound_nonlinear_state",
                None,
            )
            if callable(model_bound_initializer):
                if state is not None:
                    raise ValueError(
                        "Cannot replace qualified model-bound initial fields on an "
                        f"existing state for element {element_id}; rebuild the virgin "
                        "state atomically or continue the stored restart history."
                    )
                state = model_bound_initializer(
                    model.mesh,
                    model.get_material(element.material_name),
                    num_layers,
                    initial_fields=values,
                    initial_field_provenance=field_provenance,
                )
                if not isinstance(state, Mapping):
                    raise TypeError(
                        f"Model-bound initializer for element {element_id} must return a mapping"
                    )
                initialized_atomically = True
            elif state is None:
                state = element.init_nonlinear_state(num_layers)
        if not isinstance(state, Mapping):
            raise TypeError(
                f"Initial state for element {element_id} must be a mapping"
            )
        if not initialized_atomically:
            state = copy.deepcopy(dict(state))
        # Supplying a field for an element replaces its complete prior field
        # definition. Mixing old components with a new source would make both
        # the constitutive input and its provenance ambiguous; callers that
        # want multiple components must provide them together in one field.
        if not initialized_atomically:
            for key in (*_INITIAL_FIELD_STATE_KEYS, "initial_field_provenance"):
                state.pop(key, None)
            for key, array in values.items():
                state[key] = array
            state["initial_field_provenance"] = field_provenance
        validate_initial_field_state(
            element,
            model.get_material(element.material_name),
            state,
            num_layers,
            model.mesh,
        )
        states[element_id] = state
        provenance.append(
            {
                "element_id": element_id,
                "kind": field_type,
                "source": field_value.source,
                "components": {
                    key.removeprefix("initial_"): _initial_value_summary(array)
                    for key, array in values.items()
                },
            }
        )
    _initialize_missing_model_bound_states(model, states, num_layers)
    return states, provenance


def _finalize_nonlinear_element_states(
    model: "FEModel",
    displacements: np.ndarray,
    element_states: Dict[int, Any],
    num_layers: int,
    *,
    kinematics: str = "von_karman",
) -> Dict[int, Any]:
    """Return result-ready element states.

    The default assembly paths already return complete recovery state. Optional
    acceleration layers may replace this no-op with a final-displacement
    recovery pass for data deliberately omitted inside Newton iterations.
    """

    del model, displacements, num_layers, kinematics
    return element_states


def _activate_nonlinear_state_storage(
    model: "FEModel",
    committed_states: Mapping[int, Any],
    num_layers: int,
    info: Dict[str, Any],
    *,
    kinematics: str,
    committed_displacements: Optional[np.ndarray] = None,
    noncurrent_native_element_ids: Sequence[int] = (),
) -> Mapping[int, Any]:
    """Create one solver-owned material/native-rotation transaction store."""

    previous_diagnostic = info.get("nonlinear_state_storage")
    noncurrent_native = {
        int(value) for value in noncurrent_native_element_ids
    }
    native_required = any(
        bool(getattr(element, "formulation_native_total_lagrangian", False))
        and int(element_id) not in noncurrent_native
        for element_id, element in model.mesh.elements.items()
    )
    if native_required and not isinstance(committed_states, NonlinearStateStore):
        initialized_states = copy.deepcopy(dict(committed_states))
        _initialize_missing_model_bound_states(
            model,
            initialized_states,
            num_layers,
        )
        committed_states = initialized_states
    diagnostic: Dict[str, Any] = {
        "activated": False,
        "eligible_batch_count": 0,
        "fallback_reason": None,
    }
    if native_required:
        diagnostic.update(
            {
                "native_rotation_required": True,
                "native_rotation_activated": False,
                "noncurrent_native_element_ids": sorted(noncurrent_native),
            }
        )
    info["nonlinear_state_storage"] = diagnostic
    if isinstance(committed_states, NonlinearStateStore):
        if isinstance(previous_diagnostic, Mapping):
            for key in (
                "eligible_batch_count",
                "batch_eligibility",
                "array_batch_fallback_reason",
            ):
                if key in previous_diagnostic:
                    diagnostic[key] = copy.deepcopy(previous_diagnostic[key])
        if native_required and not committed_states.has_native_rotations:
            if committed_displacements is None:
                raise StateTransactionError(
                    "Native state storage requires committed displacements"
                )
            from .nonlinear_state import create_model_native_rotation_store

            rotation_store = create_model_native_rotation_store(
                model,
                committed_states,
                committed_displacements,
                noncurrent_element_ids=tuple(sorted(noncurrent_native)),
            )
            if rotation_store is None:
                raise StateTransactionError(
                    "Native elements were present but no rotation store was created"
                )
            committed_states.attach_native_rotation_store(rotation_store)
        diagnostic["activated"] = True
        if native_required:
            diagnostic["native_rotation_activated"] = (
                committed_states.has_native_rotations
            )
        diagnostic.update(committed_states.diagnostics())
        return _register_nonlinear_state_cleanup(committed_states)
    if str(kinematics) != "von_karman":
        if native_required:
            raise NotImplementedError(
                "Formulation-native total-Lagrangian elements cannot use the "
                f"generic {kinematics!r} nonlinear kinematics path"
            )
        diagnostic["fallback_reason"] = "kinematics_not_von_karman"
        return committed_states
    persistent_disabled = os.environ.get(
        "FE_SOLVER_DISABLE_PERSISTENT_STATE", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    acceleration_disabled = os.environ.get(
        "FE_SOLVER_DISABLE_FAST_NL", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    allow_array_batches = not persistent_disabled and not acceleration_disabled
    if not native_required and persistent_disabled:
        diagnostic["fallback_reason"] = "persistent_state_storage_disabled"
        return committed_states
    if not native_required and acceleration_disabled:
        diagnostic["fallback_reason"] = "nonlinear_acceleration_disabled"
        return committed_states
    try:
        plastic_batches: tuple[Any, ...] = ()
        if allow_array_batches:
            from .nonlinear_performance_bootstrap import get_nonlinear_assembly_plan

            plan = get_nonlinear_assembly_plan(model, int(num_layers))
            plastic_batches = tuple(
                batch for batch in plan.shell_batches if batch.has_plasticity
            )
        store = NonlinearStateStore.from_shell_layouts(
            tuple(batch.state_layout for batch in plastic_batches),
            committed_states,
        )
        eligibility = []
        for batch in plastic_batches:
            state_batch = store.shell_batch_for_layout(batch.state_layout)
            eligible, reason = batch.persistent_state_eligibility(state_batch)
            eligibility.append(
                {
                    "element_ids": list(batch.state_layout.element_ids),
                    "eligible": bool(eligible),
                    "fallback_reason": reason,
                }
            )
        eligible_count = sum(int(item["eligible"]) for item in eligibility)
        diagnostic.update(
            {
                "eligible_batch_count": int(eligible_count),
                "batch_eligibility": eligibility,
            }
        )
        if not native_required and eligible_count == 0:
            if persistent_disabled:
                diagnostic["fallback_reason"] = "persistent_state_storage_disabled"
            elif acceleration_disabled:
                diagnostic["fallback_reason"] = "nonlinear_acceleration_disabled"
            elif not plastic_batches:
                diagnostic["fallback_reason"] = "no_plastic_shell_batch"
            else:
                diagnostic["fallback_reason"] = "no_persistent_state_batch_eligible"
            return committed_states
        if native_required:
            if committed_displacements is None:
                raise StateTransactionError(
                    "Native state storage requires committed displacements"
                )
            from .nonlinear_state import create_model_native_rotation_store

            rotation_store = create_model_native_rotation_store(
                model,
                committed_states,
                committed_displacements,
                noncurrent_element_ids=tuple(sorted(noncurrent_native)),
            )
            if rotation_store is None:
                raise StateTransactionError(
                    "Native elements were present but no rotation store was created"
                )
            store.attach_native_rotation_store(rotation_store)
            diagnostic["native_rotation_activated"] = True
            if not allow_array_batches:
                diagnostic["array_batch_fallback_reason"] = (
                    "persistent_state_storage_disabled"
                    if persistent_disabled
                    else "nonlinear_acceleration_disabled"
                )
        diagnostic["activated"] = True
        diagnostic.update(store.diagnostics())
        return _register_nonlinear_state_cleanup(store)
    except Exception as exc:
        if native_required:
            diagnostic["fallback_reason"] = (
                f"native_state_store_setup_failed:{type(exc).__name__}:{exc}"
            )
            raise
        diagnostic["fallback_reason"] = (
            f"state_store_setup_failed:{type(exc).__name__}:{exc}"
        )
        return committed_states


def _commit_nonlinear_state_candidate(
    committed_states: Mapping[int, Any],
    candidate_states: Mapping[int, Any],
    *,
    model: Any = None,
    accepted_displacements: Any = None,
) -> Mapping[int, Any]:
    return commit_state_candidate(
        committed_states,
        candidate_states,
        model=model,
        accepted_full_displacement=accepted_displacements,
    )


def _discard_nonlinear_state_candidate(
    committed_states: Mapping[int, Any],
) -> None:
    discard_active_state_candidate(committed_states)


def _materialize_final_nonlinear_states(
    committed_states: Mapping[int, Any],
    info: Dict[str, Any],
) -> Dict[int, Any]:
    store = committed_states if isinstance(committed_states, NonlinearStateStore) else None
    result = materialize_state_mapping(
        committed_states,
        policy=StateMaterializationPolicy.FINAL_RESULT,
    )
    if store is not None:
        activation = dict(info.get("nonlinear_state_storage", {}))
        activation.update(store.diagnostics())
        activation["activated"] = True
        info["nonlinear_state_storage"] = activation
    return result


def _nonlinear_status_category(status: str, failure_reason: Optional[str]) -> str:
    if status == "completed":
        return "converged"
    if status == "empty_reduced_system":
        return "invalid_model"
    reason = str(failure_reason or "")
    if reason.startswith("fracture_") or "deleted_fraction" in reason:
        return "fracture_limit"
    if "singular" in reason or "factorization" in reason:
        return "singular_tangent"
    if "nonfinite" in reason:
        return "numerical_instability"
    if "maximum_iterations" in reason:
        return "iteration_failure"
    if "minimum_load_increment" in reason:
        return "limit_point_or_nonconvergence"
    if status == "stopped_at_limit":
        return "limit_point_or_nonconvergence"
    return "failed"


def _equilibrate_initial_fields_candidate(
    *,
    model: "FEModel",
    T: sparse.csr_matrix,
    u0: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    max_iterations: int,
    tolerance: float,
    kinematics: str,
    corotational_tangent: str,
    general_tangent: bool,
    initial_reduced_displacements: Optional[np.ndarray] = None,
    cancellation_token: Optional[CancellationToken] = None,
    _exact_guard: Any = None,
) -> Tuple[np.ndarray, Dict[int, Any], List[Dict[str, Any]], Optional[str]]:
    """Equilibrate prescribed residual fields before any external load.

    The residual field is held fixed while compatible displacements and the
    material history are solved from ``F_int = 0``.  Only a converged trial
    state is committed.  This is intentionally a separate zero-load phase so
    permanent/environmental loading cannot silently absorb an unbalanced
    manufacturing field in its first increment.
    """
    exact_guard = _exact_guard or _EXACT_QUALIFIED_LIFECYCLE_GUARD
    exact_guard(
        model,
        context="nonlinear static initial equilibration preflight",
    )
    n_red = int(T.shape[1])
    if initial_reduced_displacements is None:
        q = np.zeros(n_red, dtype=float)
    else:
        q = np.asarray(initial_reduced_displacements, dtype=float).reshape(-1).copy()
        exact_guard(
            model,
            context="nonlinear static initial displacement observation",
        )
        if q.size != n_red:
            raise ValueError(
                f"initial_reduced_displacements has {q.size} entries; expected {n_red}"
            )
    committed_q = q.copy()
    history: List[Dict[str, Any]] = []
    reference: Optional[float] = None
    for iteration in range(1, max_iterations + 1):
        cancellation_safe_point(
            cancellation_token,
            f"nonlinear_static.initial_equilibration.iteration:{iteration}",
        )
        exact_guard(
            model,
            context="nonlinear static initial equilibration cancellation",
        )
        u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
        F_int, K_T, trial_states = _assemble_nonlinear_system(
            model,
            u,
            committed_states,
            num_layers,
            tangent=True,
            kinematics=kinematics,
            corotational_tangent=corotational_tangent,
        )
        exact_guard(
            model,
            context="nonlinear static initial equilibration assembly",
        )
        residual = -np.asarray(T.T @ F_int, dtype=float).reshape(-1)
        residual_norm = float(np.linalg.norm(residual))
        if reference is None:
            reference = max(residual_norm, 1.0)
        history.append(
            {
                "iteration": iteration,
                "residual_norm": residual_norm,
                "displacement_norm": float(np.linalg.norm(u)),
            }
        )
        if not np.isfinite(residual_norm):
            return committed_q, committed_states, history, "nonfinite_initial_state_residual"
        if residual_norm <= tolerance * reference:
            return q, trial_states, history, None
        K_red = (T.T @ K_T @ T).tocsr()
        try:
            with np.errstate(all="ignore"):
                handle = factorize(
                    K_red,
                    MatrixClass.GENERAL if general_tangent else MatrixClass.SYMMETRIC_INDEFINITE,
                    signature=f"nonlinear.initial_state:{iteration}",
                )
                dq = np.asarray(handle.solve(residual), dtype=float).reshape(-1)
        except Exception:
            return committed_q, committed_states, history, "singular_initial_state_tangent"
        if np.any(~np.isfinite(dq)):
            return committed_q, committed_states, history, "nonfinite_initial_state_increment"

        # Residual-norm backtracking makes non-equilibrated user fields
        # predictable without affecting the ordinary load-step fast path.
        accepted = False
        scale = 1.0
        for _ in range(16):
            q_candidate = q + scale * dq
            u_candidate = np.asarray(T @ q_candidate + u0, dtype=float).reshape(-1)
            F_candidate, _unused, _candidate_states = _assemble_nonlinear_system(
                model,
                u_candidate,
                committed_states,
                num_layers,
                tangent=False,
                kinematics=kinematics,
                corotational_tangent=corotational_tangent,
            )
            exact_guard(
                model,
                context="nonlinear static initial equilibration trial assembly",
            )
            candidate_norm = float(
                np.linalg.norm(-np.asarray(T.T @ F_candidate, dtype=float).reshape(-1))
            )
            if np.isfinite(candidate_norm) and candidate_norm < residual_norm:
                q = q_candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return committed_q, committed_states, history, "initial_state_line_search_failed"
    return committed_q, committed_states, history, "maximum_initial_state_iterations_reached"


def _equilibrate_initial_fields(
    *,
    model: "FEModel",
    T: sparse.csr_matrix,
    u0: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    max_iterations: int,
    tolerance: float,
    kinematics: str,
    corotational_tangent: str,
    general_tangent: bool,
    initial_reduced_displacements: Optional[np.ndarray] = None,
    cancellation_token: Optional[CancellationToken] = None,
    _exact_guard: Any = None,
) -> Tuple[np.ndarray, Dict[int, Any], List[Dict[str, Any]], Optional[str]]:
    """Commit exactly one accepted zero-load state and discard every other trial."""

    try:
        q, candidate_states, history, failure = (
            _equilibrate_initial_fields_candidate(
                model=model,
                T=T,
                u0=u0,
                committed_states=committed_states,
                num_layers=num_layers,
                max_iterations=max_iterations,
                tolerance=tolerance,
                kinematics=kinematics,
                corotational_tangent=corotational_tangent,
                general_tangent=general_tangent,
                initial_reduced_displacements=initial_reduced_displacements,
                cancellation_token=cancellation_token,
                _exact_guard=_exact_guard,
            )
        )
        if failure is not None:
            return q, committed_states, history, failure
        accepted_states = _commit_nonlinear_state_candidate(
            committed_states,
            candidate_states,
            model=model,
            accepted_displacements=np.asarray(T @ q + u0, dtype=float).reshape(-1),
        )
        return q, accepted_states, history, None
    finally:
        # This is a no-op after a successful persistent commit and for legacy
        # dictionaries.  Every failure return, cancellation, and exception
        # therefore leaves the input state reusable without duplicating
        # cleanup at each exit site in the Newton/backtracking implementation.
        _discard_nonlinear_state_candidate(committed_states)


def _reduced_coordinates(
    T: sparse.csr_matrix,
    u0: np.ndarray,
    displacements: Optional[np.ndarray],
) -> np.ndarray:
    """Recover reduced coordinates for a compatible full displacement state."""
    if displacements is None:
        return np.zeros(int(T.shape[1]), dtype=float)
    full = np.asarray(displacements, dtype=float).reshape(-1)
    if full.size != T.shape[0]:
        raise ValueError(f"initial_displacements has {full.size} entries; expected {T.shape[0]}")
    if np.any(~np.isfinite(full)):
        raise ValueError("initial_displacements must contain only finite values")
    rhs = full - np.asarray(u0, dtype=float).reshape(-1)
    if int(T.shape[1]) == 0:
        error = float(np.linalg.norm(rhs))
        scale = max(float(np.linalg.norm(full)), 1.0)
        if error > 1.0e-9 * scale:
            raise ValueError(
                "initial_displacements is incompatible with the fully "
                f"constrained state (residual {error:.3e})"
            )
        return np.zeros(0, dtype=float)
    # Constraint elimination normally retains an exact unit row for each
    # independent coordinate. Invert those rows directly: a least-squares
    # stopping error can otherwise perturb a valid supplied equilibrium.
    # General transformations without complete retained rows keep LSQR.
    singleton_rows = np.flatnonzero(np.diff(T.indptr) == 1)
    singleton_offsets = T.indptr[singleton_rows]
    unit_rows = singleton_rows[T.data[singleton_offsets] == 1.0]
    unit_columns = T.indices[T.indptr[unit_rows]]
    columns, first = np.unique(unit_columns, return_index=True)
    if columns.size == int(T.shape[1]):
        q = np.empty(int(T.shape[1]), dtype=float)
        q[columns] = rhs[unit_rows[first]]
    else:
        solution = sparse.linalg.lsqr(T, rhs, atol=1.0e-12, btol=1.0e-12)
        q = np.asarray(solution[0], dtype=float).reshape(-1)
    if np.any(~np.isfinite(q)):
        raise ValueError("initial_displacements could not be reduced to finite coordinates")
    error = float(np.linalg.norm(np.asarray(T @ q, dtype=float).reshape(-1) - rhs))
    scale = max(float(np.linalg.norm(rhs)), 1.0)
    if error > 1.0e-9 * scale:
        raise ValueError(
            "initial_displacements is incompatible with the active supports/MPC constraints "
            f"(projection residual {error:.3e})"
        )
    return q


def _reduced_coordinates_with_affine_scale(
    T: sparse.csr_matrix,
    u0: np.ndarray,
    displacements: Optional[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Recover a restart state and its proportional affine-constraint scale.

    A force-control restart must preserve the supplied physical support/MPC
    state while a new proportional load path starts at zero.  Solving the
    augmented compatible representation ``u = T q + s u0`` recovers both the
    free coordinates and the already-committed affine scale ``s`` without
    guessing it from one selected support row.
    """

    if displacements is None:
        return np.zeros(int(T.shape[1]), dtype=float), 0.0
    full = np.asarray(displacements, dtype=float).reshape(-1)
    if full.size != T.shape[0]:
        raise ValueError(
            f"initial_displacements has {full.size} entries; expected {T.shape[0]}"
        )
    if np.any(~np.isfinite(full)):
        raise ValueError("initial_displacements must contain only finite values")
    affine = np.asarray(u0, dtype=float).reshape(-1)
    if float(np.linalg.norm(affine)) <= 1.0e-30:
        return _reduced_coordinates(T, np.zeros_like(affine), full), 0.0

    augmented = sparse.hstack(
        (T, sparse.csr_matrix(affine.reshape(-1, 1))),
        format="csr",
    )
    solution = sparse.linalg.lsqr(
        augmented,
        full,
        atol=1.0e-12,
        btol=1.0e-12,
    )
    values = np.asarray(solution[0], dtype=float).reshape(-1)
    q = values[:-1]
    affine_scale = float(values[-1])
    if np.any(~np.isfinite(q)) or not np.isfinite(affine_scale):
        raise ValueError(
            "initial_displacements could not be reduced to finite coordinates"
        )
    reconstructed = np.asarray(augmented @ values, dtype=float).reshape(-1)
    error = float(np.linalg.norm(reconstructed - full))
    scale = max(float(np.linalg.norm(full)), 1.0)
    if error > 1.0e-9 * scale:
        raise ValueError(
            "initial_displacements is incompatible with the active "
            "supports/MPC affine path "
            f"(projection residual {error:.3e})"
        )
    return q, affine_scale


def _exact_binary64_additive_offset(
    target: np.ndarray,
    projected: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Return an offset whose binary64 addition reconstructs ``target``.

    The reduced-coordinate solve is a compatibility proof, not serialization
    authority: sparse projections may differ from the caller/checkpoint vector
    by a few ulps.  Retaining the exact additive residual keeps the public
    committed displacement bit-identical without changing the established
    reduced coordinates or projecting the state a second time.
    """

    expected = np.asarray(target, dtype=np.float64).reshape(-1)
    base = np.asarray(projected, dtype=np.float64).reshape(-1)
    if expected.shape != base.shape or not (
        np.all(np.isfinite(expected)) and np.all(np.isfinite(base))
    ):
        raise ValueError(f"{context} requires compatible finite vectors")
    offset = expected - base
    for _pass in range(4):
        reconstructed = base + offset
        if np.array_equal(reconstructed, expected):
            return offset
        offset += expected - reconstructed
    raise ValueError(f"{context} cannot be reconstructed bit-exactly")


def _solve_static_displacement_control(
    *,
    model: "FEModel",
    T: sparse.csr_matrix,
    u0: np.ndarray,
    F_const: np.ndarray,
    F_prop: np.ndarray,
    stage_vectors: Sequence[np.ndarray],
    load_case: Optional["LoadCase"],
    constant_load_case: Optional["LoadCase"],
    load_program: Optional[NonlinearLoadProgram],
    displacement_control: DisplacementControl,
    committed_states: Dict[int, Any],
    num_layers: int,
    num_steps: int,
    max_iterations: int,
    tolerance: float,
    info: Dict[str, Any],
    start_time: float,
    resource_config: Optional[ResourceConfig] = None,
    kinematics: str = "von_karman",
    corotational_tangent: str = "not_applicable",
    initial_reduced_displacements: Optional[np.ndarray] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    record_increment_snapshots: bool = False,
    initial_load_factor: float = 0.0,
    recover_initial_load_factor: bool = False,
    step_index_offset: int = 0,
    initial_steps: Sequence[NonlinearStaticStep] = (),
    initial_history: Sequence[Mapping[str, Any]] = (),
    initial_total_iterations: int = 0,
    restart_analysis_contract: Optional[Mapping[str, Any]] = None,
    rotating_mpc_map: Optional[Any] = None,
    _exact_guard: Any = None,
) -> NonlinearStaticResult:
    """Displacement-control Newton solve with load factor as an unknown."""
    exact_guard = _exact_guard or _EXACT_QUALIFIED_LIFECYCLE_GUARD
    exact_guard(
        model,
        context="nonlinear static displacement-control preflight",
    )
    if load_program is not None:
        if len(load_program.stages) == 1:
            constant_terms = [(constant_load_case, 1.0)]
            proportional_term = (
                load_program.stages[0].load_case,
                load_program.stages[0].target_factor,
            )
            F_const_static = F_const
            F_prop_static = load_program.stages[0].target_factor * stage_vectors[0]
            active_stage = load_program.stages[0].name
        else:
            constant_terms = [(constant_load_case, 1.0)] + [
                (stage.load_case, stage.target_factor)
                for stage in load_program.stages[:-1]
            ]
            proportional_term = (
                load_program.stages[-1].load_case,
                load_program.stages[-1].target_factor,
            )
            F_const_static = F_const.copy()
            for stage, vector in zip(load_program.stages[:-1], stage_vectors[:-1]):
                F_const_static += stage.target_factor * vector
            F_prop_static = load_program.stages[-1].target_factor * stage_vectors[-1]
            active_stage = load_program.stages[-1].name
        info["displacement_control_load_split"] = {
            "constant_stages": [stage.name for stage in load_program.stages[:-1]],
            "proportional_stage": active_stage,
        }
    else:
        constant_terms = [(constant_load_case, 1.0)]
        proportional_term = (load_case, 1.0)
        F_const_static = F_const
        F_prop_static = F_prop
        active_stage = "displacement_control"

    n_red = int(T.shape[1])
    if initial_reduced_displacements is None:
        q = np.zeros(n_red, dtype=float)
    else:
        q = np.asarray(initial_reduced_displacements, dtype=float).reshape(-1).copy()
        if q.size != n_red:
            raise ValueError(
                f"initial_reduced_displacements has {q.size} entries; expected {n_red}"
            )
    lam = float(initial_load_factor)
    steps: List[NonlinearStaticStep] = list(initial_steps)
    snapshots: List[NonlinearIncrementSnapshot] = []
    history: List[Dict[str, Any]] = copy.deepcopy(list(initial_history))
    status = "completed"
    failure_reason: Optional[str] = None
    total_iterations = int(initial_total_iterations)

    from .corotational import SO3MeanDomainError

    def full_u(q_vec: np.ndarray) -> np.ndarray:
        u_lin = np.asarray(T @ q_vec + u0, dtype=float).reshape(-1)
        if rotating_mpc_map is not None:
            return rotating_mpc_map.apply_kinematics(u_lin)
        return u_lin

    row_full = displacement_control.full_row(model)
    exact_guard(model, context="nonlinear static displacement-control row")
    row_red = np.asarray(row_full @ T, dtype=float).reshape(-1)
    row_u0 = float(row_full @ u0)
    # Admission uses the actual initial mapped state and its Jacobian: a
    # nonlinear-only control coordinate (zero linear row, nonzero
    # finite-state derivative) is usable at a finite restart while a truly
    # zero Jacobian is still refused.  Without a rotating map the linear
    # row remains the admission criterion.
    initial_u = full_u(q)
    if rotating_mpc_map is not None:
        row_jac_initial = np.asarray(
            row_full @ rotating_mpc_map.jacobian(initial_u), dtype=float
        ).reshape(-1)
        if float(np.linalg.norm(row_jac_initial)) <= 0.0:
            raise ValueError(
                "Displacement control target has a zero control Jacobian at "
                "the initial state and cannot be used as an unknown"
            )
    elif float(np.linalg.norm(row_red)) <= 0.0:
        raise ValueError("Displacement control target is fixed or dependent and cannot be used as an unknown")
    follower_active = any(
        _has_follower_pressure(
            case,
            model=model,
            _exact_guard=exact_guard,
        )
        for case, _factor in [*constant_terms, proportional_term]
    )
    if follower_active:
        F_prop_initial, _ = _weighted_external_load_system(
            model,
            [proportional_term],
            initial_u,
            tangent=False,
        )
        exact_guard(
            model,
            context="nonlinear static displacement-control initial load",
        )
    else:
        F_prop_initial = F_prop_static
    F_prop_red = (
        rotating_mpc_map.reduce_force(initial_u, F_prop_initial)
        if rotating_mpc_map is not None
        else np.asarray(T.T @ F_prop_initial, dtype=float).reshape(-1)
    )
    if float(np.linalg.norm(F_prop_red)) <= 0.0:
        raise ValueError("Displacement control requires a non-zero proportional load vector")

    if recover_initial_load_factor and rotating_mpc_map is not None:
        # Ordinary supplied initial-state resume without re-equilibration:
        # the copied state is an accepted equilibrium, so recover its load
        # scalar by least squares on the reduced forces as the Newton
        # initial guess.  This is a predictor only: it changes no success
        # criterion and no public contract, and the diagnostic trial
        # assembly below is never committed.
        f_int_initial, _k0, _initial_trial = _assemble_nonlinear_system(
            model,
            initial_u,
            committed_states,
            num_layers,
            tangent=False,
            kinematics=kinematics,
            corotational_tangent=corotational_tangent,
        )
        exact_guard(
            model,
            context="nonlinear static displacement-control initial load recovery",
        )
        _discard_nonlinear_state_candidate(committed_states)
        if follower_active:
            f_const_initial, _unused_k = _weighted_external_load_system(
                model, constant_terms, initial_u, tangent=False,
            )
        else:
            f_const_initial = F_const_static
        f_red_initial = rotating_mpc_map.reduce_force(
            initial_u, f_int_initial - f_const_initial
        )
        _p_norm = float(F_prop_red @ F_prop_red)
        if _p_norm > 0.0:
            lam = float((F_prop_red @ f_red_initial) / _p_norm)
        info["displacement_control_recovered_initial_load_factor"] = float(lam)
    zero_load_tangent = sparse.csr_matrix(
        (model.mesh.dof_manager.total_dofs, model.mesh.dof_manager.total_dofs),
        dtype=float,
    )

    target_total = float(displacement_control.target_displacement)
    initial_control = (
        float(row_full @ full_u(q))
        if rotating_mpc_map is not None
        else float(row_red @ q + row_u0)
    )
    target_increment = target_total - initial_control
    target_scale = max(abs(target_increment), abs(target_total), 1.0e-9)
    info["displacement_control_initial_value"] = initial_control

    from .general_contact.runtime import active_contact_manager

    contact_manager = active_contact_manager()
    augmented_contact_manager = (
        contact_manager if getattr(contact_manager, "augmented", False) else None
    )
    moving_tool = bool(
        contact_manager is not None
        and hasattr(contact_manager, "path_translation")
        and np.any(contact_manager.path_translation)
    )
    motion_factor = 0.0
    if moving_tool:
        if target_total == 0.0:
            raise ValueError("moving rigid-tool displacement control needs a nonzero target")
        motion_factor = initial_control / target_total
        if not -1.0e-12 <= motion_factor <= 1.0 + 1.0e-12:
            raise ValueError("initial controlled displacement is outside the rigid-tool path")
        motion_factor = min(1.0, max(0.0, motion_factor))
        contact_manager.set_path_factor(motion_factor)
        initial_contact = contact_manager.evaluate(
            full_u(q), tangent=False,
        )[2]
        if not initial_contact.gap_tolerance_met:
            raise ValueError("committed moving-tool pose exceeds the contact gap tolerance")
        info["contact_path_parameter"] = "controlled_displacement_fraction"

    # Bounded adaptive control-step subdivision for the rotating-MPC,
    # contact-free displacement-control route: a failed fixed control
    # interval is discarded and only that interval is retried subdivided,
    # preserving every scheduled control milestone.  Legacy and contact
    # routes keep their fixed-increment behaviour unchanged.
    adaptive_control_subdivision = (
        rotating_mpc_map is not None and contact_manager is None
    )
    max_control_cutbacks = 8
    control_attempts_total = 0
    control_cutbacks_total = 0
    control_line_search_reductions = 0
    control_material_event_steps = 0
    control_material_event_evaluations = 0
    control_seed_solves = 0
    control_seed_history = []
    previous_accepted_control_state = None
    seeded_control_direction = 0.0
    min_control_subincrement = (
        abs(target_increment) / max(num_steps, 1) / (2.0 ** max_control_cutbacks)
    )
    record_step_index = int(step_index_offset)
    assembly_threads = None if resource_config is None else resource_config.assembly_threads
    with numba_thread_scope(assembly_threads):
        for local_step_index in range(1, num_steps + 1):
            step_index = int(step_index_offset) + local_step_index
            cancellation_safe_point(
                cancellation_token,
                f"nonlinear_static.displacement.step:{step_index}",
            )
            exact_guard(
                model,
                context="nonlinear static displacement-control step cancellation",
            )
            milestone = initial_control + target_increment * local_step_index / num_steps
            attempt_fraction = 1.0
            attempts_this_step = 0
            cutbacks_this_step = 0
            interval_cutbacks = 0
            accepted_subincrements = 0
            step_iterations = 0
            failed_seed_at_accepted_origin = None
            while True:
                attempts_this_step += 1
                control_attempts_total += 1
                q_attempt_start = q.copy()
                lam_attempt_start = float(lam)
                attempt_start_control = (
                    float(row_full @ full_u(q))
                    if rotating_mpc_map is not None
                    else float(row_red @ q + row_u0)
                )
                target = attempt_start_control + (
                    milestone - attempt_start_control
                ) * attempt_fraction
                residual_norm = float("inf")
                constraint_error = float("inf")
                states_new = committed_states

                next_motion_factor = target / target_total if moving_tool else motion_factor
                if moving_tool:
                    start_u = full_u(q_attempt_start)
                    if not contact_manager.swept_increment_admissible(
                        start_u, start_u, motion_factor, next_motion_factor,
                    ):
                        failure_reason = "contact_gap_exceeded"
                        status = "contact_gap_exceeded"
                        break
                    contact_manager.set_path_factor(next_motion_factor)

                start_normal_multipliers = (
                    augmented_contact_manager.normal_multipliers.copy()
                    if augmented_contact_manager is not None else None
                )
                augmentation_limit = (
                    augmented_contact_manager.settings.maximum_augmentations
                    if augmented_contact_manager is not None else 1
                )
                augmentation_iterations = 0
                normal_multiplier_change = 0.0
                attempt_seed_solves = 0
                attempt_used_equilibrated_seed = False
                pending_control_direction = float(np.sign(target-attempt_start_control))
                if pending_control_direction * seeded_control_direction < 0.0:
                    seeded_control_direction = 0.0
                if (
                    adaptive_control_subdivision and not follower_active
                    and previous_accepted_control_state is not None
                    and (seeded_control_direction != 0.0
                         or abs(lam_attempt_start) < abs(previous_accepted_control_state[1]))
                    and ((attempt_start_control-previous_accepted_control_state[2])
                         * (target-attempt_start_control) > 0.0)
                    and max_iterations > 1 and target != attempt_start_control
                    and np.isfinite(_max_plastic_strain(committed_states))
                    and _max_plastic_strain(committed_states) > 0.0
                ):
                    from ._nonlinear_seed import secant_plane_seed

                    def seed_evaluate(seed_q, seed_lam, tangent):
                        cancellation_safe_point(
                            cancellation_token,
                            "nonlinear_static.displacement.seed.probe",
                        )
                        exact_guard(model, context="displacement seed probe")
                        seed_u = full_u(seed_q)
                        # No plane trial owns the accepted state transaction.
                        seed_history = _copy_initial_states(committed_states)
                        try:
                            seed_f, seed_k, _seed_states = _assemble_nonlinear_system(
                                model, seed_u, seed_history, num_layers,
                                tangent=tangent, kinematics=kinematics,
                                corotational_tangent=corotational_tangent,
                            )
                            exact_guard(model, context="displacement seed assembly")
                            external = F_const_static + seed_lam * F_prop_static
                            seed_r = rotating_mpc_map.reduce_force(
                                seed_u, seed_f-external,
                            )
                            seed_reference = max(float(np.linalg.norm(
                                rotating_mpc_map.reduce_force(seed_u,
                                    F_const_static + max(abs(seed_lam), 1.0) * F_prop_static)
                            )), 1.0)
                            seed_tangent = (
                                rotating_mpc_map.reduce_tangent(
                                    seed_u, seed_k, seed_f-external,
                                ).toarray() if tangent else None
                            )
                            seed_load_column = -rotating_mpc_map.reduce_force(
                                seed_u, F_prop_static,
                            )
                            cancellation_safe_point(
                                cancellation_token,
                                "nonlinear_static.displacement.seed.probe_complete",
                            )
                            exact_guard(model, context="displacement seed observation")
                            return dict(residual=seed_r, reference=seed_reference,
                                        tangent=seed_tangent, load_column=seed_load_column)
                        finally:
                            _discard_nonlinear_state_candidate(seed_history)

                    def seed_solve(matrix, seed_rhs):
                        nonlocal attempt_seed_solves, total_iterations
                        cancellation_safe_point(
                            cancellation_token,
                            "nonlinear_static.displacement.seed.solve",
                        )
                        exact_guard(model, context="displacement seed solve")
                        attempt_seed_solves += 1
                        total_iterations += 1
                        seed_handle = factorize(
                            sparse.csr_matrix(matrix), MatrixClass.GENERAL,
                            signature=(f"nonlinear.displacement_seed:"
                                       f"{step_index}:{attempt_seed_solves}"),
                        )
                        if getattr(seed_handle, "status", "ok") != "ok":
                            return None
                        # Unrelated errors/cancellation must propagate; a
                        # factorization's declared failure is the only fallback.
                        return np.asarray(seed_handle.solve(seed_rhs), dtype=float).reshape(-1)

                    # Fixed work metric: translations/1m, rotations/1rad and
                    # lambda/1. The static load-pattern L1 work scale is fixed
                    # before probing; it is not fitted to residual rows.
                    seed_force_scale = max(
                        float(np.linalg.norm(F_const_static, ord=1)),
                        float(np.linalg.norm(F_prop_static, ord=1)), 1.0,
                    )
                    try:
                        if failed_seed_at_accepted_origin is not None:
                            # The plane contains no pending DC target. Between
                            # cutbacks q/lambda/H/the accepted pair, map, static
                            # loads and tolerance stay unchanged. Reuse only its
                            # failed observation, never a failed trial iterate.
                            seed_x = np.r_[q_attempt_start, lam_attempt_start]
                            seed_info = dict(
                                failed_seed_at_accepted_origin,
                                solves=0, evaluations=0, cached_failure=True,
                            )
                        else:
                            seed_x, seed_info = secant_plane_seed(
                                q_attempt_start, lam_attempt_start,
                                previous_accepted_control_state,
                                seed_evaluate, seed_solve,
                                maximum_solves=min(20, max_iterations-1),
                                tolerance=tolerance, force_scale=seed_force_scale,
                                domain_error=SO3MeanDomainError,
                            )
                            if seed_info["status"] in (
                                "seed_domain_refused", "seed_solve_budget",
                                "seed_merit_no_improvement", "invalid_accepted_secant",
                            ):
                                failed_seed_at_accepted_origin = dict(seed_info)
                    finally:
                        _discard_nonlinear_state_candidate(committed_states)
                    assert attempt_seed_solves == seed_info["solves"]
                    control_seed_solves += attempt_seed_solves
                    step_iterations += attempt_seed_solves
                    control_seed_history.append(dict(
                        seed_info, target=float(target),
                        accepted_load_factor=float(lam_attempt_start),
                        previous_accepted_load_factor=float(previous_accepted_control_state[1]),
                        accepted_control=float(attempt_start_control),
                        previous_accepted_control=float(previous_accepted_control_state[2]),
                        strategy_engaged=bool(seeded_control_direction != 0.0),
                        force_scale=seed_force_scale,
                        remaining_iterations=max_iterations-attempt_seed_solves,
                    ))
                    cancellation_safe_point(
                        cancellation_token,
                        "nonlinear_static.displacement.seed.complete",
                    )
                    exact_guard(model, context="displacement seed completion")
                    # Only an initial guess; no plane state/history is committed.
                    if seed_info["status"] == "equilibrated_uncommitted_seed":
                        q = seed_x[:-1].copy()
                        lam = float(seed_x[-1])
                        attempt_used_equilibrated_seed = True
                for augmentation in range(1, augmentation_limit + 1):
                    augmentation_iterations = augmentation
                    for iteration in range(1, max_iterations-attempt_seed_solves + 1):
                        cancellation_safe_point(
                            cancellation_token,
                            f"nonlinear_static.displacement.step:{step_index}.iteration:{iteration}",
                        )
                        exact_guard(
                            model,
                            context=(
                                "nonlinear static displacement-control iteration cancellation"
                            ),
                        )
                        total_iterations += 1
                        u = full_u(q)
                        F_int, K_T, trial_states = _assemble_nonlinear_system(
                            model,
                            u,
                            committed_states,
                            num_layers,
                            kinematics=kinematics,
                            corotational_tangent=corotational_tangent,
                        )
                        exact_guard(
                            model,
                            context="nonlinear static displacement-control assembly",
                        )
                        if follower_active:
                            F_const_current, K_const = _weighted_external_load_system(
                                model,
                                constant_terms,
                                u,
                                tangent=True,
                            )
                            F_prop_current, K_prop = _weighted_external_load_system(
                                model,
                                [proportional_term],
                                u,
                                tangent=True,
                            )
                            exact_guard(
                                model,
                                context="nonlinear static displacement-control loads",
                            )
                        else:
                            F_const_current, F_prop_current = F_const_static, F_prop_static
                            K_const = K_prop = zero_load_tangent
                        F_external = F_const_current + lam * F_prop_current
                        if rotating_mpc_map is not None:
                            residual = rotating_mpc_map.reduce_force(
                                u, F_external - F_int
                            )
                            current = float(row_full @ u)
                            ref_vec = rotating_mpc_map.reduce_force(
                                u, F_const_current + max(abs(lam), 1.0) * F_prop_current
                            )
                        else:
                            residual = (
                                np.asarray(T.T @ F_external, dtype=float).reshape(-1)
                                - np.asarray(T.T @ F_int, dtype=float).reshape(-1)
                            )
                            current = float(row_red @ q + row_u0)
                            ref_vec = np.asarray(
                                T.T @ (F_const_current + max(abs(lam), 1.0) * F_prop_current),
                                dtype=float,
                            )
                        residual_norm = float(np.linalg.norm(residual))
                        constraint = target - current
                        constraint_error = abs(constraint)
                        reference = max(float(np.linalg.norm(ref_vec)), 1.0)

                        if residual_norm <= tolerance * reference and constraint_error <= tolerance * target_scale:
                            from .general_contact.runtime import active_contact_manager

                            contact_manager = active_contact_manager()
                            if (contact_manager is not None
                                    and augmented_contact_manager is None
                                    and not contact_manager.evaluate(
                                        u, tangent=False
                                    )[2].gap_tolerance_met):
                                failure_reason = "contact_gap_exceeded"
                                break
                            states_new = trial_states
                            break

                        if rotating_mpc_map is not None:
                            K_ext_total = (
                                (K_const + lam * K_prop) if follower_active else None
                            )
                            K_red = rotating_mpc_map.reduce_tangent(
                                u, K_T, F_int - F_external, K_ext_total
                            )
                            F_prop_red = rotating_mpc_map.reduce_force(u, F_prop_current)
                            row_jac = np.asarray(
                                row_full @ rotating_mpc_map.jacobian(u), dtype=float
                            ).reshape(-1)
                        else:
                            K_red = (
                                (T.T @ K_T @ T)
                                - (T.T @ K_const @ T)
                                - lam * (T.T @ K_prop @ T)
                            ).tocsr()
                            F_prop_red = np.asarray(T.T @ F_prop_current, dtype=float).reshape(-1)
                            row_jac = row_red
                        aug = sparse.bmat(
                            [
                                [K_red, sparse.csr_matrix((-F_prop_red).reshape(-1, 1))],
                                [sparse.csr_matrix(row_jac.reshape(1, -1)), sparse.csr_matrix((1, 1))],
                            ],
                            format="csr",
                        )
                        rhs = np.concatenate([residual, np.array([constraint], dtype=float)])
                        try:
                            with np.errstate(all="ignore"):
                                # The augmented control system [K, -F; row, 0]
                                # is nonsymmetric even for a symmetric tangent:
                                # the proportional-load column and the control
                                # row are independent, and a symmetric
                                # factorization would consume only the upper
                                # triangle.  Always factorize as GENERAL.
                                handle = factorize(
                                    aug,
                                    MatrixClass.GENERAL,
                                    signature=(
                                        f"nonlinear.displacement_control:{step_index}:{iteration}"
                                        if augmented_contact_manager is None else
                                        f"nonlinear.displacement_control:{step_index}:"
                                        f"{augmentation}:{iteration}"
                                    ),
                                )
                                delta = np.asarray(handle.solve(rhs), dtype=float).reshape(-1)
                        except Exception:
                            failure_reason = "singular_augmented_tangent"
                            break
                        if np.any(~np.isfinite(delta)):
                            failure_reason = "nonfinite_augmented_solution"
                            break
                        full_step_admissible = False
                        if adaptive_control_subdivision and np.any(delta):
                            residual_merit_globalization = any(
                                np.isfinite(value) and value > 0.0
                                for value in (
                                    _max_plastic_strain(committed_states),
                                    _max_plastic_strain(trial_states),
                                )
                            )
                            # Bounded residual line search on the rotating-MPC
                            # contact-free route: from a converged state the
                            # first postyield Newton step can overshoot the
                            # load factor catastrophically.  Backtrack on the
                            # augmented merit (reduced residual and control
                            # constraint, both normalized like the
                            # convergence criterion).  Every trial evaluation
                            # only replaces the pending state candidate and
                            # is never committed, so committed histories are
                            # retained.  A trial whose master rotations
                            # leave the supported SO(3) chart is refused
                            # (SO3MeanDomainError) and the search backtracks;
                            # a refused trial is never committed and never
                            # mutates constitutive state.  While the control
                            # row is not yet satisfied the full Newton step is
                            # kept (unchanged behavior); once the control
                            # constraint holds, a step with no admissible
                            # sufficient decrease is not committed: the
                            # attempt fails with a classified reason and
                            # bounded subdivision retries, because there the
                            # committed full step sustains a Newton limit
                            # cycle (measured period-3 cycle on the direct
                            # EP route).
                            merit_current = (
                                residual_norm / (tolerance * reference)
                                + constraint_error / (tolerance * target_scale)
                            )
                            line_search_alpha = 1.0
                            line_search_accepted = False
                            any_trial_admissible = False
                            event_base = None
                            event_trial_states = None
                            event_trial_alpha = None
                            if (residual_merit_globalization
                                    and residual_norm * target_scale >= constraint_error * reference):
                                from ._nonlinear_events import material_events
                                event_base = material_events(model, committed_states, trial_states)
                                exact_guard(model, context="nonlinear static material event observation")
                            if merit_current > 0.0:
                                q_line_start = q.copy()
                                lam_line_start = float(lam)
                                alpha = 1.0
                                for line_reduction in range(8):
                                    q_trial = q_line_start + alpha * delta[:-1]
                                    lam_trial = (
                                        lam_line_start + alpha * float(delta[-1])
                                    )
                                    try:
                                        u_trial = full_u(q_trial)
                                    except SO3MeanDomainError:
                                        # Bounded trial-domain rejection:
                                        # the trial state left the supported
                                        # rotation chart.  It is inadmissible:
                                        # backtrack to the next halving and
                                        # never commit or assemble it.
                                        alpha *= 0.5
                                        continue
                                    if line_reduction == 0:
                                        full_step_admissible = True
                                    any_trial_admissible = True
                                    if not residual_merit_globalization:
                                        # Preserve the elastic full-Newton
                                        # correction. Domain refusals still
                                        # backtrack through the same eight
                                        # chart-admissibility trials above.
                                        line_search_alpha = alpha
                                        line_search_accepted = True
                                        break
                                    f_int_trial, _k_trial, _s_trial = (
                                        _assemble_nonlinear_system(
                                            model,
                                            u_trial,
                                            committed_states,
                                            num_layers,
                                            tangent=False,
                                            kinematics=kinematics,
                                            corotational_tangent=( 
                                                corotational_tangent
                                            ),
                                        )
                                    )
                                    exact_guard(
                                        model,
                                        context=( 
                                            "nonlinear static displacement-control "
                                            "line search"
                                        ),
                                    )
                                    if follower_active:
                                        f_const_trial, _unused_k = (
                                            _weighted_external_load_system(
                                                model, constant_terms, u_trial,
                                                tangent=False,
                                            )
                                        )
                                        f_prop_trial, _unused_k2 = (
                                            _weighted_external_load_system(
                                                model, [proportional_term],
                                                u_trial, tangent=False,
                                            )
                                        )
                                    else:
                                        f_const_trial = F_const_static
                                        f_prop_trial = F_prop_static
                                    residual_trial = (
                                        rotating_mpc_map.reduce_force(
                                            u_trial,
                                            f_const_trial
                                            + lam_trial * f_prop_trial
                                            - f_int_trial,
                                        )
                                    )
                                    constraint_trial = abs(
                                        target - float(row_full @ u_trial)
                                    )
                                    merit_trial = (
                                        float(np.linalg.norm(residual_trial))
                                        / (tolerance * reference)
                                        + constraint_trial
                                        / (tolerance * target_scale)
                                    )
                                    event_trial_states = _s_trial
                                    event_trial_alpha = alpha
                                    if merit_trial <= (
                                        1.0 - 1.0e-4 * alpha
                                    ) * merit_current:
                                        line_search_alpha = alpha
                                        line_search_accepted = True
                                        break
                                    # For the actual augmented Newton solve,
                                    # with reference scales frozen in this
                                    # search, merit'(0) = -merit_current.
                                    # A quadratic through the rejected trial
                                    # therefore proposes this next alpha.
                                    # Safeguards retain descent backtracking
                                    # while allowing a genuinely small Newton
                                    # step within the same eight-trial budget.
                                    denominator = (
                                        merit_trial - merit_current
                                        + alpha * merit_current
                                    )
                                    if (
                                        residual_norm * target_scale
                                        >= constraint_error * reference
                                        and merit_trial >= merit_current
                                    ):
                                        # A nearby constitutive branch switch
                                        # can create a flat or rising shoulder
                                        # even though the local Newton slope
                                        # is exact.  Interpolating that trial
                                        # approaches a half step and can miss
                                        # existing small descent within eight
                                        # trials. Contract that shoulder more
                                        # strongly without accepting it.
                                        alpha *= 0.1
                                    elif (
                                        residual_norm * target_scale
                                        >= constraint_error * reference
                                        and np.isfinite(denominator)
                                        and denominator > 0.0
                                    ):
                                        proposed_alpha = (
                                            0.5 * alpha * alpha * merit_current
                                            / denominator
                                        )
                                        alpha = float(np.clip(
                                            proposed_alpha, 0.1 * alpha, 0.5 * alpha
                                        ))
                                    else:
                                        # Preserve the control predictor's
                                        # existing halving/full-step fallback.
                                        # Tiny accepted control advances can
                                        # otherwise replace that predictor
                                        # with many unnecessary subdivisions.
                                        alpha *= 0.5
                            if (not line_search_accepted and event_base is not None
                                    and event_trial_alpha is not None):
                                from ._nonlinear_events import crossed_material_event
                                event_upper = material_events(model, committed_states, event_trial_states)
                                exact_guard(model, context="nonlinear static material event observation")

                                def evaluate_material_event(event_alpha):
                                    nonlocal control_material_event_evaluations
                                    cancellation_safe_point(
                                        cancellation_token,
                                        f"nonlinear_static.displacement.material_event:{iteration}",
                                    )
                                    exact_guard(model, context="nonlinear static material event cancellation")
                                    event_q = q_line_start + event_alpha * delta[:-1]
                                    event_lam = lam_line_start + event_alpha * float(delta[-1])
                                    try:
                                        event_u = full_u(event_q)
                                    except SO3MeanDomainError:
                                        return None
                                    event_f, _event_k, event_states = _assemble_nonlinear_system(
                                        model, event_u, committed_states, num_layers, tangent=False,
                                        kinematics=kinematics, corotational_tangent=corotational_tangent,
                                    )
                                    control_material_event_evaluations += 1
                                    exact_guard(model, context="nonlinear static material event trial")
                                    if follower_active:
                                        event_const, _unused_k = _weighted_external_load_system(
                                            model, constant_terms, event_u, tangent=False)
                                        event_prop, _unused_k2 = _weighted_external_load_system(
                                            model, [proportional_term], event_u, tangent=False)
                                    else:
                                        event_const, event_prop = F_const_static, F_prop_static
                                    event_external = event_const + event_lam * event_prop
                                    event_r = rotating_mpc_map.reduce_force(event_u, event_external - event_f)
                                    if not np.all(np.isfinite(event_f)) or not np.all(np.isfinite(event_r)):
                                        return None
                                    event_constraint = abs(target - float(row_full @ event_u))
                                    event_merit = (float(np.linalg.norm(event_r)) / (tolerance * reference)
                                                   + event_constraint / (tolerance * target_scale))
                                    observation = material_events(model, committed_states, event_states)
                                    exact_guard(model, context="nonlinear static material event observation")
                                    return {"events": observation, "merit": event_merit,
                                            "residual_norm": float(np.linalg.norm(event_r)),
                                            "force_scale": max(float(np.linalg.norm(rotating_mpc_map.reduce_force(event_u, event_f))),
                                                               float(np.linalg.norm(rotating_mpc_map.reduce_force(event_u, event_external))), 1.0)}

                                event_step = crossed_material_event(
                                    event_base, event_upper, event_trial_alpha,
                                    q_line_start, delta[:-1], evaluate_material_event,
                                )
                                if event_step is not None:
                                    noise_bound = 128.0 * np.finfo(float).eps * max(
                                        event_step["force_scale"], residual_norm, reference, 1.0)
                                    if (event_step["merit"] <= (1.0 - 1.0e-4 * event_step["alpha"]) * merit_current
                                            and residual_norm - event_step["residual_norm"] > noise_bound):
                                        line_search_alpha = event_step["alpha"]
                                        line_search_accepted = True
                                        control_material_event_steps += 1
                            if merit_current > 0.0 and not line_search_accepted:
                                if not any_trial_admissible:
                                    failure_reason = "trial_state_domain_refused"
                                    break
                                if (
                                    residual_norm * target_scale
                                    >= constraint_error * reference
                                ):
                                    # No admissible improving step was found
                                    # in the bounded trial set while the residual term dominates the
                                    # augmented merit (equilibrium
                                    # correction): committing the
                                    # full step has no demonstrated descent.
                                    # Classify the
                                    # attempt failed; the rollback below
                                    # restores the committed state and
                                    # bounded subdivision retries.
                                    failure_reason = "line_search_no_improvement"
                                    break
                                # Control row not yet satisfied: keep the
                                # full Newton step (unchanged behavior).
                            if line_search_accepted and line_search_alpha < 1.0:
                                delta = delta * line_search_alpha
                                control_line_search_reductions += 1
                        if rotating_mpc_map is not None and not full_step_admissible:
                            try:
                                full_u(q + delta[:-1])
                            except SO3MeanDomainError:
                                # The step endpoint is outside the supported
                                # rotation chart: refuse the commit, roll
                                # back to the attempt start and classify the
                                # failure instead of aborting the solve from
                                # an otherwise valid committed state.
                                failure_reason = "trial_state_domain_refused"
                                break
                        from .general_contact.runtime import active_contact_manager

                        contact_manager = active_contact_manager()
                        if contact_manager is not None:
                            candidate_u = full_u(q + delta[:-1])
                            fraction = contact_manager.admissible_step_fraction(u, candidate_u)
                            if fraction <= 1.0e-12:
                                failure_reason = "contact_gap_exceeded"
                                break
                            delta *= fraction
                        q += delta[:-1]
                        lam += float(delta[-1])
                    else:
                        failure_reason = "maximum_iterations_reached"

                    step_iterations += iteration
                    if failure_reason is not None or augmented_contact_manager is None:
                        break
                    candidate_u = full_u(q)
                    certificate = augmented_contact_manager.evaluate(
                        candidate_u, tangent=False
                    )[2]
                    projected = augmented_contact_manager.projected_normal_multipliers(
                        candidate_u
                    )
                    normal_multiplier_change = float(np.max(np.abs(
                        projected - augmented_contact_manager.normal_multipliers
                    )))
                    if (normal_multiplier_change <=
                            augmented_contact_manager.settings.pressure_tolerance
                            and certificate.gap_tolerance_met):
                        break
                    augmented_contact_manager.set_normal_multipliers(projected)
                else:
                    failure_reason = "maximum_augmentations_reached"

                if failure_reason is not None:
                    _discard_nonlinear_state_candidate(committed_states)
                    q = q_attempt_start
                    lam = lam_attempt_start
                    if augmented_contact_manager is not None:
                        augmented_contact_manager.set_normal_multipliers(
                            start_normal_multipliers
                        )
                    if moving_tool:
                        contact_manager.set_path_factor(motion_factor)
                    # DC1: subdivision requires a strictly positive, finite,
                    # representable remaining control span and an explicit
                    # cutback-count bound.  A zero-span request (restart at
                    # the current controlled value, or a zero target
                    # increment) must not retry forever: it runs its bounded
                    # Newton attempt and exits through the finite failure
                    # path below.
                    attempted_control_span = (
                        abs(milestone - attempt_start_control) * attempt_fraction
                    )
                    if (
                        failure_reason in (
                            "maximum_iterations_reached",
                            "line_search_no_improvement",
                            "trial_state_domain_refused",
                        )
                        and adaptive_control_subdivision
                        and np.isfinite(attempted_control_span)
                        and attempted_control_span > 0.0
                        and attempted_control_span * 0.5
                            >= min_control_subincrement
                        and interval_cutbacks < max_control_cutbacks
                    ):
                        # Discard the failed trial and subdivide only this
                        # control interval; the scheduled milestone and all
                        # previously accepted increments are preserved.
                        failure_reason = None
                        attempt_fraction *= 0.5
                        control_cutbacks_total += 1
                        cutbacks_this_step += 1
                        interval_cutbacks += 1
                        continue
                    if (
                        failure_reason == "maximum_iterations_reached"
                        and adaptive_control_subdivision
                    ):
                        failure_reason = "minimum_control_step_reached"
                    status = (
                        "contact_gap_exceeded"
                        if failure_reason == "contact_gap_exceeded"
                        else "stopped_at_limit" if steps else "diverged"
                    )
                    break

                if adaptive_control_subdivision and attempt_fraction < 1.0:
                    accepted_control = float(row_full @ full_u(q))
                    control_progress = accepted_control - attempt_start_control
                    # Only an accepted interval may reset its retry counter.
                    # A subdivided interval must make representable progress
                    # toward its milestone. Normally the positive minimum
                    # span bounds accepted subdivisions by 2**max_cutbacks
                    # (up to the convergence tolerance); retain an explicit
                    # count bound as well, so loose tolerances or roundoff
                    # cannot repeatedly accept the same control state.
                    if (
                        not np.isfinite(control_progress)
                        or (control_progress <= 0.0 if milestone > attempt_start_control
                            else control_progress >= 0.0)
                        or accepted_subincrements >= 2 ** max_control_cutbacks + 1
                    ):
                        _discard_nonlinear_state_candidate(committed_states)
                        q = q_attempt_start
                        lam = lam_attempt_start
                        failure_reason = "minimum_control_step_reached"
                        status = "stopped_at_limit" if steps else "diverged"
                        break

                contact_tool_work_increment = 0.0
                if moving_tool:
                    end_u = full_u(q)
                    start_u = full_u(q_attempt_start)
                    if not contact_manager.swept_increment_admissible(
                        start_u, end_u, motion_factor, next_motion_factor,
                    ):
                        _discard_nonlinear_state_candidate(committed_states)
                        q = q_attempt_start
                        lam = lam_attempt_start
                        if augmented_contact_manager is not None:
                            augmented_contact_manager.set_normal_multipliers(
                                start_normal_multipliers
                            )
                        contact_manager.set_path_factor(motion_factor)
                        failure_reason = "contact_gap_exceeded"
                        status = "contact_gap_exceeded"
                        break
                    contact_tool_work_increment = contact_manager.tool_work_increment(
                        start_u, end_u, motion_factor, next_motion_factor,
                    )

                u = full_u(q)
                committed_states = _commit_nonlinear_state_candidate(
                    committed_states,
                    states_new,
                    model=model,
                    accepted_displacements=u,
                )
                previous_accepted_control_state = (
                    q_attempt_start.copy(), float(lam_attempt_start),
                    float(attempt_start_control),
                )
                failed_seed_at_accepted_origin = None
                if attempt_used_equilibrated_seed:
                    # Keep the family-seed strategy when the newly reached
                    # branch reloads; only original DC acceptance activates it.
                    seeded_control_direction = pending_control_direction
                motion_factor = next_motion_factor
                current = (
                    float(row_full @ u)
                    if rotating_mpc_map is not None
                    else float(row_red @ q + row_u0)
                )
                reaction_internal, _unused, _reaction_states = _assemble_nonlinear_system(
                    model,
                    u,
                    committed_states,
                    num_layers,
                    tangent=False,
                    kinematics=kinematics,
                    corotational_tangent=corotational_tangent,
                    require_full_coordinates=True,
                )
                exact_guard(
                    model,
                    context="nonlinear static displacement-control reaction assembly",
                )
                # Reaction recovery is diagnostic-only; do not leave its trial
                # constitutive state active for the next accepted increment.
                _discard_nonlinear_state_candidate(committed_states)
                if follower_active:
                    reaction_constant, _unused = _weighted_external_load_system(
                        model, constant_terms, u, tangent=False
                    )
                    reaction_proportional, _unused = _weighted_external_load_system(
                        model, [proportional_term], u, tangent=False
                    )
                    exact_guard(
                        model,
                        context="nonlinear static displacement-control reaction loads",
                    )
                else:
                    reaction_constant = F_const_static
                    reaction_proportional = F_prop_static
                reaction_imbalance = (
                    reaction_internal - (reaction_constant + lam * reaction_proportional)
                )
                if rotating_mpc_map is not None:
                    reaction_imbalance = (
                        rotating_mpc_map.transfer_slave_imbalance_to_masters(
                            u, reaction_imbalance
                        )
                    )
                support_reactions = _support_reaction_resultants(
                    model,
                    reaction_imbalance,
                )
                record_step_index += 1
                steps.append(
                    NonlinearStaticStep(
                        step_index=record_step_index,
                        load_factor=float(lam),
                        iterations=step_iterations,
                        residual_norm=residual_norm,
                        displacement_norm=float(np.linalg.norm(u)),
                        max_equivalent_plastic_strain=_max_plastic_strain(committed_states),
                        control_value=current,
                        active_stage=active_stage,
                        support_reactions=support_reactions,
                    )
                )
                history.append(
                    {
                        "step_index": record_step_index,
                        "load_factor": float(lam),
                        "control_value": current,
                        "target_displacement": target,
                        "residual_norm": residual_norm,
                        "constraint_error": constraint_error,
                        "iterations": step_iterations,
                        **({"control_attempts": attempts_this_step,
                            "control_cutbacks": cutbacks_this_step}
                           if adaptive_control_subdivision else {}),
                        **({
                            "augmentation_iterations": augmentation_iterations,
                            "normal_multiplier_change": normal_multiplier_change,
                            "contact": _augmented_contact_increment_summary(certificate),
                            "normal_multipliers": (
                                augmented_contact_manager.normal_multipliers.tolist()
                            ),
                        } if augmented_contact_manager is not None else {}),
                        "active_stage": active_stage,
                        **({"contact_path_factor": motion_factor,
                            "contact_tool_work_increment": contact_tool_work_increment}
                           if moving_tool else {}),
                        "support_reactions": {
                            name: list(values)
                            for name, values in support_reactions.items()
                        },
                    }
                )
                if record_increment_snapshots:
                    snapshots.append(
                        _increment_snapshot(
                            record_step_index,
                            lam,
                            u,
                            committed_states,
                            control_value=current,
                        )
                    )
                if progress_callback is not None:
                    emit_progress(
                        progress_callback,
                        "nonlinear_static_step",
                        "nonlinear_static.displacement",
                        completed=local_step_index,
                        total=num_steps,
                        iteration=step_iterations,
                        control="displacement",
                        step_index=int(record_step_index),
                        load_factor=float(lam),
                        control_value=float(current),
                        displacement_norm=float(np.linalg.norm(u)),
                        max_equivalent_plastic_strain=float(_max_plastic_strain(committed_states)),
                        support_reactions={
                            name: list(values)
                            for name, values in support_reactions.items()
                        },
                    )
                    exact_guard(
                        model,
                        context="nonlinear static displacement-control progress callback",
                    )
                if attempt_fraction >= 1.0:
                    break
                # The accepted increment was a subdivision of the current
                # control interval; continue with the remaining interval.
                # DC2: the published per-record iteration count describes
                # only this accepted increment; reset it before continuing
                # the remaining interval so later accepted subdivisions do
                # not double-count earlier accepted work.  Solve-wide totals
                # stay in total_iterations and the cumulative attempt fields.
                step_iterations = 0
                attempt_fraction = 1.0
                # Earlier accepted subdivisions no longer belong to the
                # failed interval being retried. Preserve cumulative evidence,
                # but give the remaining interval its own bounded cutbacks.
                interval_cutbacks = 0
                accepted_subincrements += 1
            if failure_reason is not None:
                break

    u_final = full_u(q)
    if (not steps and not committed_states and (
            augmented_contact_manager is not None
            or any(_qualified_s3_committed_state_validator(element) is not None
                   for element in model.mesh.elements.values()))):
        # A rejected first trial has no committed candidate. Recover the
        # accepted origin through the normal structural owner before sealing;
        # do not substitute the failed trial or mask its failure status.
        _initial_force, _initial_tangent, reference_states = (
            _assemble_nonlinear_system(
                model, u_final, committed_states, num_layers,
                tangent=False, kinematics=kinematics,
                corotational_tangent=corotational_tangent,
            )
        )
        committed_states = _commit_nonlinear_state_candidate(
            committed_states, reference_states, model=model,
            accepted_displacements=u_final,
        )
    committed_states = _materialize_final_nonlinear_states(committed_states, info)
    committed_states = _finalize_nonlinear_element_states(
        model,
        u_final,
        committed_states,
        num_layers,
        kinematics=kinematics,
    )
    committed_states = _seal_final_qualified_q4_states(
        model,
        u_final,
        committed_states,
        num_layers,
        info,
        kinematics=kinematics,
    )
    if adaptive_control_subdivision:
        info["displacement_control_adaptation"] = {
            "attempts": int(control_attempts_total),
            "cutbacks": int(control_cutbacks_total),
            "max_cutbacks_per_interval": int(max_control_cutbacks),
            "min_subincrement": float(min_control_subincrement),
            "line_search_reductions": int(control_line_search_reductions),
            "material_event_steps": int(control_material_event_steps),
            "material_event_evaluations": int(control_material_event_evaluations),
            "uncommitted_seed_solves": int(control_seed_solves),
            "uncommitted_seeds": control_seed_history,
            "applied": control_cutbacks_total > 0,
        }
    info["failure_reason"] = failure_reason
    info["stop_reason"] = "target_displacement_reached" if failure_reason is None else failure_reason
    info["status_category"] = _nonlinear_status_category(status, failure_reason)
    info["last_converged_load_factor"] = float(lam)
    info["peak_load_factor"] = max((step.load_factor for step in steps), default=float(lam))
    info["force_displacement_history"] = history
    if moving_tool:
        info["contact_path_factor"] = motion_factor
    info["strain_summary"] = _nonlinear_state_summary(committed_states)
    info["total_newton_iterations"] = total_iterations
    info["solve_time"] = time.time() - start_time
    u_postcheck = (
        rotating_mpc_map.to_linear_displacement(u_final)
        if rotating_mpc_map is not None
        else u_final
    )
    info["constraint_postcheck"] = constraint_residual_summary(model, u_postcheck)
    if rotating_mpc_map is not None:
        info["constraint_postcheck"]["nonlinear_rotating_mpc_count"] = len(
            rotating_mpc_map.entries
        )
    exact_guard(
        model,
        context="nonlinear static displacement-control constraint postcheck",
    )
    info["result_case"] = make_result_case(
        name="nonlinear_static_displacement_control",
        analysis_type="nonlinear_static",
        load_cases=tuple(stage.load_case for stage in load_program.stages) if load_program is not None else (),
        assembly_info={"load": {"vector_type": "load_program" if load_program is not None else "load"}, **info},
        solver_info={"convergence_info": {"status": status}},
        recovery={"displacements": True, "element_states": True, "force_displacement_history": True},
        settings={
            "control": "displacement",
            "num_steps": num_steps,
            "num_layers": num_layers,
            "kinematics": kinematics,
            "corotational_tangent": corotational_tangent,
        },
    ).to_dict()
    restart_payload = None
    if restart_analysis_contract is not None:
        # Seal the accepted constitutive record at the same reference
        # deformational coordinates used by assembly, before validation.
        committed_states = _seal_final_qualified_q4_states(
            model, u_final, committed_states, num_layers, info,
            kinematics=kinematics, seal_corotational_checkpoint=True,
        )
        restart_payload = create_nonlinear_checkpoint(
            analysis_kind="static",
            model=model,
            analysis_contract=restart_analysis_contract,
            displacements=u_final,
            element_states=committed_states,
            deleted_element_ids=(),
            path_state=_static_restart_path_payload(
                control_name="displacement",
                load_factor=float(lam),
                step_index=len(steps),
                total_iterations=total_iterations,
                steps=steps,
                force_displacement_history=history,
                reduced_coordinates=q,
                terminal_status=status,
                failure_reason=failure_reason,
                **({"normal_multipliers":
                    augmented_contact_manager.normal_multipliers.tolist()}
                   if augmented_contact_manager is not None else {}),
            ),
        )
        exact_guard(
            model,
            context="nonlinear static displacement-control checkpoint output",
        )
    return NonlinearStaticResult(
        steps,
        status,
        u_final,
        float(lam),
        committed_states,
        info,
        tuple(snapshots),
        restart_payload,
    )


@dataclass(frozen=True)
class _NonlinearStaticOperationConfig:
    max_load_factor: float
    num_steps: int
    max_iterations: int
    tolerance: float
    num_layers: int
    min_step_fraction: float
    control: str
    kinematics: str
    corotational_tangent: str
    equilibrate_initial_state: Optional[bool]
    record_increment_snapshots: bool
    emit_restart_checkpoint: bool


_NONLINEAR_CONFIG_BUILTIN_CONVERTERS = (bool, int, float, str)
_NONLINEAR_CONFIG_INERT_TYPES = (bool, int, float, str, bytes, type(None))
_NONLINEAR_CONFIG_STRING_TYPE = str


def _inert_nonlinear_config_scalar(value: Any) -> bool:
    """Exact builtin scalar conversion cannot invoke caller-defined methods.

    Use identity, not isinstance or type equality: subclasses (including NumPy
    scalars) and custom metaclasses must retain the hostile-observation guard.
    """
    value_type = type(value)
    return any(
        value_type is builtin for builtin in _NONLINEAR_CONFIG_INERT_TYPES
    )


def _owned_nonlinear_static_operation_config(
    model: "FEModel",
    *,
    max_load_factor: Any,
    num_steps: Any,
    max_iterations: Any,
    tolerance: Any,
    num_layers: Any,
    min_step_fraction: Any,
    control: Any,
    kinematics: Any,
    corotational_tangent: Any,
    equilibrate_initial_state: Any,
    record_increment_snapshots: Any,
    emit_restart_checkpoint: Any,
    _exact_guard: Any,
) -> _NonlinearStaticOperationConfig:
    """Own scalar nonlinear policy before model/capability observation."""

    def converted(value: Any, converter: Any, name: str) -> Any:
        inert = _inert_nonlinear_config_scalar(value) and any(
            converter is builtin for builtin in _NONLINEAR_CONFIG_BUILTIN_CONVERTERS
        )
        made = converter(value)
        if not inert:
            _exact_guard(model, context=f"nonlinear static {name} conversion")
        return made

    def canonical_int(value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(f"{name} must be an integer")
        return converted(value, int, name)

    def normalized_string(value: Any, name: str) -> str:
        text = converted(value, str, name)
        # __str__ can return a str subclass whose lower() runs caller code.
        # Guard that second observation before converting any later option.
        inert = type(text) is _NONLINEAR_CONFIG_STRING_TYPE
        lowered = text.lower()
        if not inert:
            _exact_guard(model, context=f"nonlinear static {name} normalization")
        return lowered

    owned_equilibrate = (
        None
        if equilibrate_initial_state is None
        else converted(
            equilibrate_initial_state,
            bool,
            "equilibrate_initial_state",
        )
    )
    owned = _NonlinearStaticOperationConfig(
        max_load_factor=converted(
            max_load_factor,
            float,
            "max_load_factor",
        ),
        num_steps=canonical_int(num_steps, "num_steps"),
        max_iterations=canonical_int(max_iterations, "max_iterations"),
        tolerance=converted(tolerance, float, "tolerance"),
        num_layers=canonical_int(num_layers, "num_layers"),
        min_step_fraction=converted(
            min_step_fraction,
            float,
            "min_step_fraction",
        ),
        control=normalized_string(control, "control"),
        kinematics=normalized_string(kinematics, "kinematics"),
        corotational_tangent=converted(
            corotational_tangent,
            str,
            "corotational_tangent",
        ),
        equilibrate_initial_state=owned_equilibrate,
        record_increment_snapshots=converted(
            record_increment_snapshots,
            bool,
            "record_increment_snapshots",
        ),
        emit_restart_checkpoint=converted(
            emit_restart_checkpoint,
            bool,
            "emit_restart_checkpoint",
        ),
    )
    _exact_guard(model, context="nonlinear static owned configuration")
    return owned


def _guarded_config_sequence(
    model: "FEModel",
    value: Any,
    *,
    context: str,
    _exact_guard: Any,
) -> Tuple[Any, ...]:
    iterator = iter(value)
    _exact_guard(model, context=f"{context} iterator")
    observed = []
    while True:
        try:
            member = next(iterator)
        except StopIteration:
            _exact_guard(model, context=f"{context} end")
            break
        _exact_guard(model, context=f"{context} member")
        observed.append(member)
    return tuple(observed)


def _guarded_plain_config_snapshot(
    model: "FEModel",
    value: Any,
    *,
    context: str,
    _exact_guard: Any,
) -> Any:
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    if isinstance(value, np.ndarray):
        detached = np.array(value, copy=True)
        _exact_guard(model, context=f"{context} array observation")
        return detached
    if isinstance(value, np.generic):
        detached = value.item()
        _exact_guard(model, context=f"{context} scalar observation")
        return detached
    if isinstance(value, Mapping):
        iterator = iter(value)
        _exact_guard(model, context=f"{context} mapping iterator")
        detached_mapping: Dict[Any, Any] = {}
        while True:
            try:
                key = next(iterator)
            except StopIteration:
                _exact_guard(model, context=f"{context} mapping end")
                break
            _exact_guard(model, context=f"{context} mapping key")
            detached_key = _guarded_plain_config_snapshot(
                model,
                key,
                context=f"{context}.key",
                _exact_guard=_exact_guard,
            )
            member = value[key]
            _exact_guard(model, context=f"{context} mapping value")
            detached_member = _guarded_plain_config_snapshot(
                model,
                member,
                context=f"{context}[{key!r}]",
                _exact_guard=_exact_guard,
            )
            detached_mapping[detached_key] = detached_member
            _exact_guard(model, context=f"{context} mapping insertion")
        return detached_mapping
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _guarded_plain_config_snapshot(
                model,
                member,
                context=f"{context}[{index}]",
                _exact_guard=_exact_guard,
            )
            for index, member in enumerate(
                _guarded_config_sequence(
                    model,
                    value,
                    context=context,
                    _exact_guard=_exact_guard,
                )
            )
        )
    detached = copy.deepcopy(value)
    _exact_guard(model, context=f"{context} copy observation")
    return detached


def _owned_nonlinear_load_program(
    model: "FEModel",
    value: Any,
    *,
    _exact_guard: Any,
) -> Optional[NonlinearLoadProgram]:
    if value is None:
        return None
    raw_stages = getattr(value, "stages")
    _exact_guard(model, context="nonlinear load-program stages observation")
    stages = []
    for index, raw_stage in enumerate(
        _guarded_config_sequence(
            model,
            raw_stages,
            context="nonlinear load-program stages",
            _exact_guard=_exact_guard,
        )
    ):
        raw_name = getattr(raw_stage, "name")
        _exact_guard(
            model,
            context=f"nonlinear load-program stage {index} name observation",
        )
        name = str(raw_name)
        _exact_guard(
            model,
            context=f"nonlinear load-program stage {index} name conversion",
        )
        load_case = getattr(raw_stage, "load_case")
        _exact_guard(
            model,
            context=f"nonlinear load-program stage {index} load-case observation",
        )
        raw_factor = getattr(raw_stage, "target_factor")
        _exact_guard(
            model,
            context=f"nonlinear load-program stage {index} factor observation",
        )
        target_factor = float(raw_factor)
        _exact_guard(
            model,
            context=f"nonlinear load-program stage {index} factor conversion",
        )
        stages.append(
            NonlinearLoadStage(
                name=name,
                load_case=load_case,
                target_factor=target_factor,
            )
        )
        _exact_guard(
            model,
            context=f"nonlinear load-program stage {index} construction",
        )
    owned = NonlinearLoadProgram(tuple(stages))
    _exact_guard(model, context="nonlinear load-program construction")
    return owned


def _owned_displacement_control(
    model: "FEModel",
    value: Any,
    *,
    _exact_guard: Any,
) -> Optional[DisplacementControl]:
    if value is None:
        return None

    def observed(name: str) -> Any:
        member = getattr(value, name)
        _exact_guard(
            model,
            context=f"nonlinear displacement control {name} observation",
        )
        return member

    raw_node_id = observed("node_id")
    node_id = None if raw_node_id is None else int(raw_node_id)
    if raw_node_id is not None:
        _exact_guard(
            model,
            context="nonlinear displacement control node_id conversion",
        )
    raw_dof = observed("dof")
    if raw_dof is None:
        dof: Optional[Union[str, int]] = None
    elif isinstance(raw_dof, str):
        dof = str(raw_dof)
        _exact_guard(
            model,
            context="nonlinear displacement control dof conversion",
        )
    else:
        dof = int(raw_dof)
        _exact_guard(
            model,
            context="nonlinear displacement control dof conversion",
        )
    target = float(observed("target_displacement"))
    _exact_guard(
        model,
        context="nonlinear displacement control target conversion",
    )
    raw_weighted = observed("weighted_dofs")
    weighted: Optional[Dict[Any, float]] = None
    if raw_weighted is not None:
        if not isinstance(raw_weighted, Mapping):
            raise TypeError("weighted_dofs must be a mapping when supplied")
        weighted = {}
        iterator = iter(raw_weighted)
        _exact_guard(
            model,
            context="nonlinear displacement control weights iterator",
        )
        while True:
            try:
                raw_key = next(iterator)
            except StopIteration:
                _exact_guard(
                    model,
                    context="nonlinear displacement control weights end",
                )
                break
            _exact_guard(
                model,
                context="nonlinear displacement control weight key observation",
            )
            raw_weight = raw_weighted[raw_key]
            _exact_guard(
                model,
                context="nonlinear displacement control weight observation",
            )
            if isinstance(raw_key, tuple):
                if len(raw_key) != 2:
                    raise ValueError(
                        "weighted displacement-control node keys must have two members"
                    )
                owned_node = int(raw_key[0])
                _exact_guard(
                    model,
                    context="nonlinear displacement control weight node conversion",
                )
                raw_key_dof = raw_key[1]
                owned_dof = (
                    str(raw_key_dof)
                    if isinstance(raw_key_dof, str)
                    else int(raw_key_dof)
                )
                _exact_guard(
                    model,
                    context="nonlinear displacement control weight dof conversion",
                )
                key: Any = (owned_node, owned_dof)
            else:
                key = int(raw_key)
                _exact_guard(
                    model,
                    context="nonlinear displacement control weight key conversion",
                )
            weight = float(raw_weight)
            _exact_guard(
                model,
                context="nonlinear displacement control weight conversion",
            )
            weighted[key] = weight
            _exact_guard(
                model,
                context="nonlinear displacement control weight insertion",
            )
    owned = DisplacementControl(
        node_id=node_id,
        dof=dof,
        target_displacement=target,
        weighted_dofs=weighted,
    )
    _exact_guard(model, context="nonlinear displacement control construction")
    return owned


def _owned_fracture_config(
    model: "FEModel",
    value: Any,
    *,
    _exact_guard: Any,
) -> Optional[FractureConfig]:
    if value is None:
        return None
    if not isinstance(value, FractureConfig):
        raise TypeError("fracture_config must be a FractureConfig or None")

    def observed(name: str) -> Any:
        member = getattr(value, name)
        _exact_guard(model, context=f"fracture config {name} observation")
        return member

    def converted(name: str, converter: Any) -> Any:
        made = converter(observed(name))
        _exact_guard(model, context=f"fracture config {name} conversion")
        return made

    raw_scope = observed("element_scope")
    scope = tuple(
        str(member)
        for member in _guarded_config_sequence(
            model,
            raw_scope,
            context="fracture config element_scope",
            _exact_guard=_exact_guard,
        )
    )
    _exact_guard(model, context="fracture config element_scope conversion")
    owned = FractureConfig(
        threshold=converted("threshold", float),
        residual_stiffness_fraction=converted(
            "residual_stiffness_fraction",
            float,
        ),
        max_deleted_fraction=converted("max_deleted_fraction", float),
        min_load_factor=converted("min_load_factor", float),
        element_scope=scope,
        delete_after_converged_increment=converted(
            "delete_after_converged_increment",
            bool,
        ),
        record_history=converted("record_history", bool),
    )
    _exact_guard(model, context="fracture config construction")
    return owned


def _owned_imperfection_input(
    model: "FEModel",
    value: Any,
    *,
    _exact_guard: Any,
) -> Any:
    if value is None:
        return None
    from .imperfections import ImperfectionField

    candidate = value
    if not isinstance(candidate, (ImperfectionField, Mapping)):
        converter = getattr(candidate, "to_field", None)
        _exact_guard(model, context="imperfection converter observation")
        if converter is None:
            raise TypeError(
                f"Cannot convert {type(candidate).__name__} to ImperfectionField"
            )
        candidate = converter(model)
        _exact_guard(model, context="imperfection converter callback")

    if isinstance(candidate, Mapping):
        raw_offsets = candidate
        raw_name: Any = "imperfection"
        raw_metadata: Any = {}
    elif isinstance(candidate, ImperfectionField):
        raw_offsets = getattr(candidate, "offsets")
        _exact_guard(model, context="imperfection offsets observation")
        raw_name = getattr(candidate, "name")
        _exact_guard(model, context="imperfection name observation")
        raw_metadata = getattr(candidate, "metadata")
        _exact_guard(model, context="imperfection metadata observation")
    else:
        arrays_provider = getattr(candidate, "as_arrays")
        _exact_guard(model, context="imperfection array provider observation")
        raw_offsets = arrays_provider()
        _exact_guard(model, context="imperfection array provider callback")
        raw_name = getattr(candidate, "name", "imperfection")
        _exact_guard(model, context="imperfection name observation")
        raw_metadata = getattr(candidate, "metadata", {})
        _exact_guard(model, context="imperfection metadata observation")

    if not isinstance(raw_offsets, Mapping):
        raise TypeError("imperfection offsets must be a mapping")
    offsets: Dict[int, np.ndarray] = {}
    iterator = iter(raw_offsets)
    _exact_guard(model, context="imperfection offsets iterator")
    while True:
        try:
            raw_node_id = next(iterator)
        except StopIteration:
            _exact_guard(model, context="imperfection offsets end")
            break
        _exact_guard(model, context="imperfection node observation")
        node_id = int(raw_node_id)
        _exact_guard(model, context="imperfection node conversion")
        raw_offset = raw_offsets[raw_node_id]
        _exact_guard(model, context="imperfection offset observation")
        offset = np.asarray(raw_offset, dtype=np.float64).reshape(-1).copy()
        _exact_guard(model, context="imperfection offset conversion")
        offsets[node_id] = offset
    name = str(raw_name)
    _exact_guard(model, context="imperfection name conversion")
    metadata = _guarded_plain_config_snapshot(
        model,
        raw_metadata,
        context="imperfection metadata",
        _exact_guard=_exact_guard,
    )
    if not isinstance(metadata, Mapping):
        raise TypeError("imperfection metadata must be a mapping")
    owned = ImperfectionField(offsets, name=name, metadata=metadata)
    _exact_guard(model, context="imperfection construction")
    return owned


@resource_threaded
@capture_nonlinear_analysis_diagnostics
def _solve_static_nonlinear_under_lease(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    constant_load_case: Optional["LoadCase"] = None,
    max_load_factor: float = 1.0,
    num_steps: int = 10,
    max_iterations: int = 25,
    tolerance: float = 1.0e-6,
    num_layers: int = 5,
    min_step_fraction: float = 1.0 / 1024.0,
    imperfection: Optional[Any] = None,
    load_program: Optional[NonlinearLoadProgram] = None,
    control: str = "force",
    displacement_control: Optional[DisplacementControl] = None,
    initial_element_states: Optional[Mapping[int, Any]] = None,
    convergence_settings: Optional[Union[str, Mapping[str, Any], NonlinearConvergenceSettings]] = None,
    resource_config: Optional[ResourceConfig] = None,
    fracture_config: Optional[FractureConfig] = None,
    kinematics: str = "von_karman",
    corotational_tangent: str = "auto",
    status_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    initial_fields: Optional[Mapping[int, Any]] = None,
    initial_displacements: Optional[np.ndarray] = None,
    equilibrate_initial_state: Optional[bool] = None,
    cancellation_token: Optional[CancellationToken] = None,
    record_increment_snapshots: bool = False,
    restart_checkpoint: Optional[Any] = None,
    emit_restart_checkpoint: bool = False,
    _qualified_runtime_guard: Any = None,
    stability_control: Optional[str] = None,
    _plate_stability_binding: Any = None,
    _plate_stability_authority: Any = None,
) -> NonlinearStaticResult:
    """Incremental nonlinear static solve with adaptive load stepping.

    The proportional load case is ramped from 0 to ``max_load_factor`` while
    ``constant_load_case`` (if given) is applied in full from the first
    increment.  Plastic state is committed per element only on increment
    convergence, so every Newton iteration return-maps from the last
    converged state (standard backward-Euler incremental plasticity).

    ``kinematics="corotational"`` accepts ``corotational_tangent="rotated"``
    for the lower-cost historical approximation or ``"consistent"`` for the
    full generally nonsymmetric chain-rule Jacobian.  ``"auto"`` selects the
    consistent tangent whenever a supplied load case uses current-area
    follower pressure and otherwise preserves the rotated tangent.

    ``initial_fields`` maps element IDs to :class:`ShellInitialField` or
    :class:`BeamInitialField` values. New fields are equilibrated at zero
    external load by default. A true restart with field-bearing
    ``initial_element_states`` must also supply the matching
    ``initial_displacements``; alternatively pass
    ``equilibrate_initial_state=True`` to deliberately re-equilibrate the
    restored field from the supplied (or zero) displacement state.

    ``restart_checkpoint`` resumes the last committed solver transaction.  It
    is mutually exclusive with the lower-level initial-state arguments and is
    fingerprint-validated before stiffness or load assembly.
    """
    raw_exact_guard = _EXACT_QUALIFIED_LIFECYCLE_GUARD
    lease_model = model

    def exact_guard(
        observed_model: "FEModel",
        *,
        context: str,
    ) -> Dict[str, Any]:
        result = raw_exact_guard(observed_model, context=context)
        if stability_control is not None:
            _plate_stability_authority()
            from ._plate_stability_continuation import admit
            if admit(observed_model) != _plate_stability_binding:
                raise ValueError("plate_energy admitted model changed during solve")
        # Imperfection handling may replace ``model`` with a solver-owned
        # copy.  The non-renewable lease remains bound to the caller's model
        # and global authority generation, while the exact lifecycle guard
        # validates whichever owned model is being observed.
        _qualified_runtime_guard(lease_model, context=context)
        return result

    native_generalized_model = any(
        type(e).__module__ == "anyfesolver._ge_beam3_native_generalized_element"
        for e in model.mesh.elements.values()
    )
    if native_generalized_model:
        from ._ge_beam3_native_generalized_program import require_active, require_solver_initial

        require_active(model)
        require_solver_initial(model, initial_element_states, initial_displacements)
        if (str(control).lower() != "force" or load_program is not None or initial_fields
                or fracture_config is not None or imperfection is not None
                or restart_checkpoint is not None or emit_restart_checkpoint):
            raise ValueError("private generalized statics require their authenticated force programme")

    lease_namespace = getattr(_qualified_runtime_guard, "__dict__", {})
    owned_items_provider = (
        dict.get(lease_namespace, "_qualified_owned_element_items")
        if type(lease_namespace) is dict
        else None
    )
    trusted_runtime_guard = (
        dict.get(lease_namespace, "_qualified_trusted_require")
        if type(lease_namespace) is dict
        else None
    )
    owned_items = (
        owned_items_provider()
        if callable(owned_items_provider)
        else None
    )
    exact_qualified_internal_fast_path = bool(
        type(owned_items) is tuple
        and owned_items
        and callable(trusted_runtime_guard)
        and all(
            type(element)
            in {_QualifiedE4PLShellElement, _QualifiedE4PLS3ShellElement}
            for element_id, element in owned_items
            if type(element_id) is int
        )
        and all(type(element_id) is int for element_id, _element in owned_items)
    )
    # Models that also own exact S3 V2D elements pair the lease's trusted
    # state check with a V2D formulation/instance capture.  Any other element
    # family, or an ineligible V2D instance, keeps the complete scan.
    trusted_state_guard = (
        dict.get(lease_namespace, "_qualified_trusted_state_require")
        if type(lease_namespace) is dict
        else None
    )
    v2d_trusted_guard = (
        _CAPTURE_QUALIFIED_V2D_TRUSTED_AUTHORITY(
            owned_items,
            context="nonlinear static solve",
        )
        if (
            not exact_qualified_internal_fast_path
            and type(owned_items) is tuple
            and owned_items
            and callable(trusted_state_guard)
            and all(type(element_id) is int for element_id, _element in owned_items)
            and all(
                type(element)
                in {
                    _QualifiedE4PLShellElement,
                    _QualifiedE4PLS3ShellElement,
                    _NativeParityE4PLS3V2DShellElement,
                }
                for _element_id, element in owned_items
            )
        )
        else None
    )
    if (
        v2d_trusted_guard is None
        and not exact_qualified_internal_fast_path
        and type(owned_items) is tuple
        and any(
            type(element) is _NativeParityE4PLS3V2DShellElement
            for _element_id, element in owned_items
        )
    ):
        # Not an error: the complete lifecycle scan is used instead.  Report it
        # so a solve that cannot use the constant-time path is visible.
        record_nonlinear_solver_event("qualified_v2d_trusted_unavailable")

    def internal_guard(
        observed_model: "FEModel",
        *,
        context: str,
    ) -> Dict[str, Any]:
        """Use the captured lease only across exact built-in solver work.

        Caller-controlled observations retain ``exact_guard``.  Generic and
        imperfection-owned models also fail closed to that full lifecycle
        scan.  The lease is non-renewable and still rejects every Q4, S3,
        assembly, numerical, and model-input generation change; owned S3 V2D
        elements additionally re-validate their formulation authority and
        class epoch.
        """

        if observed_model is lease_model:
            if exact_qualified_internal_fast_path:
                trusted_runtime_guard(lease_model, context=context)
                record_nonlinear_solver_event("qualified_guard_trusted")
                return {}
            if v2d_trusted_guard is not None:
                trusted_state_guard(lease_model, context=context)
                v2d_trusted_guard(context=context)
                record_nonlinear_solver_event("qualified_guard_trusted")
                return {}
        record_nonlinear_solver_event("qualified_guard_complete_scan")
        return exact_guard(observed_model, context=context)

    def cancellation_checkpoint(stage: str, *, context: str) -> Dict[str, Any]:
        """Run one cancellation checkpoint, then the matching lifecycle guard.

        An absent token or an exact, unmodified ``CancellationToken`` can only
        run the solver's own checkpoint code.  The token is verified *before*
        that code runs and again afterwards: checking only afterwards would let
        a token repair itself inside the checkpoint and still be trusted.  A
        replaced checkpoint and any other token remain hostile observation
        boundaries and keep the complete lifecycle scan.
        """

        known_before = _is_known_cancellation_checkpoint(cancellation_token)
        cancellation_safe_point(cancellation_token, stage)
        if known_before and _is_known_cancellation_checkpoint(cancellation_token):
            return internal_guard(model, context=context)
        if cancellation_token is not None:
            record_nonlinear_solver_event("qualified_guard_untrusted_token")
        return exact_guard(model, context=context)

    exact_guard(model, context="nonlinear static solve preflight")
    cancellation_safe_point(cancellation_token, "nonlinear_static.start")
    exact_guard(model, context="nonlinear static start cancellation")
    owned_config = _owned_nonlinear_static_operation_config(
        model,
        max_load_factor=max_load_factor,
        num_steps=num_steps,
        max_iterations=max_iterations,
        tolerance=tolerance,
        num_layers=num_layers,
        min_step_fraction=min_step_fraction,
        control=control,
        kinematics=kinematics,
        corotational_tangent=corotational_tangent,
        equilibrate_initial_state=equilibrate_initial_state,
        record_increment_snapshots=record_increment_snapshots,
        emit_restart_checkpoint=emit_restart_checkpoint,
        _exact_guard=exact_guard,
    )
    max_load_factor = owned_config.max_load_factor
    num_steps = owned_config.num_steps
    max_iterations = owned_config.max_iterations
    tolerance = owned_config.tolerance
    num_layers = owned_config.num_layers
    min_step_fraction = owned_config.min_step_fraction
    control = owned_config.control
    kinematics = owned_config.kinematics
    corotational_tangent = owned_config.corotational_tangent
    equilibrate_initial_state = owned_config.equilibrate_initial_state
    record_increment_snapshots = owned_config.record_increment_snapshots
    emit_restart_checkpoint = owned_config.emit_restart_checkpoint
    load_program = _owned_nonlinear_load_program(
        model,
        load_program,
        _exact_guard=exact_guard,
    )
    displacement_control = _owned_displacement_control(
        model,
        displacement_control,
        _exact_guard=exact_guard,
    )
    fracture_config = _owned_fracture_config(
        model,
        fracture_config,
        _exact_guard=exact_guard,
    )
    imperfection = _owned_imperfection_input(
        model,
        imperfection,
        _exact_guard=exact_guard,
    )
    if any(type(e).__module__ == "anyfesolver._ge_beam3_native_fibre_static_element" for e in model.mesh.elements.values()):
        from ._ge_beam3_native_load_admission import assemble_nodal_forces

        native_load_cases = (load_case, constant_load_case) + (
            tuple(stage.load_case for stage in load_program.stages)
            if load_program is not None else ()
        )
        for native_load_case in native_load_cases:
            if native_load_case is not None:
                assemble_nodal_forces(
                    native_load_case, model.mesh, model.mesh.dof_manager,
                    guard=lambda *, stage: exact_guard(model, context=stage),
                    activity=model.mesh.element_activity,
                )
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    control_name = control
    if control_name not in {"force", "displacement"}:
        raise ValueError("control must be 'force' or 'displacement'")
    if kinematics not in {"von_karman", "corotational"}:
        raise ValueError("kinematics must be 'von_karman' or 'corotational'")
    parsed_restart_checkpoint: Optional[Dict[str, Any]] = None
    if restart_checkpoint is not None:
        emit_restart_checkpoint = True
        if initial_element_states is not None:
            raise ValueError(
                "restart_checkpoint cannot be combined with initial_element_states"
            )
        if initial_displacements is not None:
            raise ValueError(
                "restart_checkpoint cannot be combined with initial_displacements"
            )
        if initial_fields is not None:
            raise ValueError("restart_checkpoint cannot be combined with initial_fields")
        if equilibrate_initial_state not in {None, False}:
            raise ValueError(
                "restart_checkpoint cannot request initial-state re-equilibration"
            )
        parsed_restart_checkpoint = load_nonlinear_checkpoint(
            restart_checkpoint,
            _exact_guard=exact_guard,
            _guard_model=model,
        )
        require_model_element_capabilities(
            model,
            "static_restart_history",
            context="solve_static_nonlinear",
        )
    elif emit_restart_checkpoint:
        require_model_element_capabilities(
            model,
            "static_restart_history",
            context="solve_static_nonlinear",
        )
    initial_element_states = _owned_initial_element_states(
        model,
        initial_element_states,
        _exact_guard=exact_guard,
    )
    initial_displacements = _owned_initial_displacements(
        model,
        initial_displacements,
        _exact_guard=exact_guard,
    )
    if initial_fields is not None:
        initial_fields = dict(initial_fields)
        exact_guard(model, context="nonlinear static initial-fields ownership")
    initial_fields_present = initial_fields is not None and len(initial_fields) > 0
    exact_guard(model, context="nonlinear static initial-fields disposition")
    if imperfection is not None:
        require_model_element_capabilities(
            model,
            "initial_fields",
            context="solve_static_nonlinear",
        )
    elif initial_fields_present:
        require_model_element_capabilities(
            model,
            "initial_fields",
            context="solve_static_nonlinear",
            element_ids=initial_fields,
        )
    if initial_element_states is not None:
        require_model_element_capabilities(
            model,
            "static_restart_history",
            context="solve_static_nonlinear",
            element_ids=initial_element_states,
        )
    require_model_nonlinear_workflow_capabilities(
        model,
        context="solve_static_nonlinear",
    )
    specified_load_cases = [
        case
        for case in (
            [load_case, constant_load_case]
            + ([] if load_program is None else [stage.load_case for stage in load_program.stages])
        )
        if case is not None
    ]
    follower_active = any(
        _has_follower_pressure(
            case,
            model=model,
            _exact_guard=exact_guard,
        )
        for case in specified_load_cases
    )
    settings = _coerce_convergence_settings(
        convergence_settings,
        _post_observation=lambda: exact_guard(
            model,
            context="nonlinear static convergence-settings observation",
        ),
    )
    exact_guard(model, context="nonlinear static convergence settings")
    armijo_requested = str(settings.line_search).lower() == "armijo"
    if armijo_requested:
        if control_name != "force":
            raise ValueError("line_search='armijo' currently requires force control")
        if fracture_config is not None:
            raise ValueError(
                "line_search='armijo' does not yet support fracture or element deletion"
            )
        if any(
            type(element).__module__.startswith("anyfesolver._ge_beam3_native")
            for element in model.mesh.elements.values()
        ):
            raise ValueError(
                "line_search='armijo' is outside the private GE beam workflow scope"
            )
        if (
            kinematics == "corotational"
            and str(corotational_tangent).lower() == "rotated"
        ):
            raise ValueError(
                "line_search='armijo' with corotational kinematics requires "
                "corotational_tangent='auto' or 'consistent'"
            )
    from .corotational import (
        build_nonlinear_rotating_mpc_map,
        has_nonlinear_rotating_mpc,
        resolve_corotational_tangent_mode,
    )

    from .mesh_gen import InterpolatedBeamShellMPCElement
    if kinematics != "corotational" and any(
        isinstance(e, InterpolatedBeamShellMPCElement)
        and getattr(e, "connection_trace", "linear") == "director_integrated"
        for e in model.mesh.elements.values()
    ):
        raise ValueError("director_integrated nonlinear solves require corotational kinematics")

    rotating_mpc_present = (
        kinematics == "corotational" and has_nonlinear_rotating_mpc(model)
    )
    if rotating_mpc_present:
        requested_tangent = str(corotational_tangent).strip().lower()
        if requested_tangent == "auto":
            # The exact rotating-map route requires the work-conjugate
            # internal force.  Auto selects it with or without follower
            # loads; the external follower tangent is retained separately.
            tangent_request = "work_conjugate"
        elif requested_tangent in ("rotated", "consistent"):
            raise ValueError(
                "corotational_tangent='rotated' and 'consistent' change "
                "the internal force law on the rotating-MPC route; use "
                "corotational_tangent='auto' or 'work_conjugate'."
            )
        else:
            tangent_request = corotational_tangent
    else:
        tangent_request = (
            "consistent"
            if armijo_requested
            and kinematics == "corotational"
            and str(corotational_tangent).lower() == "auto"
            else corotational_tangent
        )
    resolved_corotational_tangent = resolve_corotational_tangent_mode(
        kinematics,
        tangent_request,
        follower_pressure=follower_active,
    )
    if kinematics == "corotational":
        from .corotational import validate_corotational_scope

        validate_corotational_scope(model)
        if fracture_config is not None:
            raise ValueError("Corotational kinematics v1 does not support fracture/erosion")
        if (
            follower_active
            and resolved_corotational_tangent != "consistent"
            and not (
                rotating_mpc_present
                and resolved_corotational_tangent == "work_conjugate"
            )
        ):
            raise NotImplementedError(
                "Follower pressure with corotational kinematics requires "
                "corotational_tangent='consistent'."
            )
    if max_load_factor <= 0.0:
        raise ValueError("max_load_factor must be positive")
    if kinematics == "corotational" and str(settings.line_search).lower() in {
        "auto",
        "rescue",
    }:
        # Corotational Newton necessarily passes through a large intermediate
        # residual while the element frames rotate toward the new state;
        # residual-norm backtracking rejects that excursion and grinds the
        # increment adaptation.  Plain Newton converges in a few iterations.
        settings, corotational_line_search_override = (
            _resolved_corotational_line_search_settings(settings, kinematics)
        )
    else:
        corotational_line_search_override = None
    effective_min_step_fraction = settings.min_step_fraction if settings.min_step_fraction is not None else min_step_fraction

    start_time = time.time()
    if imperfection is not None:
        from .imperfections import apply_imperfection

        model = apply_imperfection(model, imperfection, copy_model=True)
        exact_guard(model, context="nonlinear static imperfection observation")
    if armijo_requested:
        settings = dataclass_replace(
            settings,
            characteristic_length=_armijo_characteristic_length(
                model,
                settings.characteristic_length,
                rotations_present=_armijo_independent_rotations_present(model),
            ),
        )
        exact_guard(model, context="nonlinear static Armijo scaling")
    restart_analysis_contract = _static_restart_analysis_contract(
        model=model,
        load_case=load_case,
        constant_load_case=constant_load_case,
        load_program=load_program,
        control_name=control_name,
        displacement_control=displacement_control,
        num_layers=num_layers,
        max_iterations=max_iterations,
        tolerance=tolerance,
        effective_min_step_fraction=effective_min_step_fraction,
        settings=settings,
        fracture_config=fracture_config,
        resource_config=resource_config,
        kinematics=kinematics,
        resolved_corotational_tangent=resolved_corotational_tangent,
    )
    from .general_contact.runtime import active_contact_manager

    active_manager = active_contact_manager()
    # Numerical evaluation policy only: retain legacy owner/checkpoint arithmetic.
    stable_shell_rotation = bool(
        kinematics == "corotational" and resolved_corotational_tangent == "work_conjugate"
        and control_name == "force" and active_manager is None
        and initial_element_states is None and initial_displacements is None
        and initial_fields is None and imperfection is None
        and restart_checkpoint is None and not emit_restart_checkpoint
        and fracture_config is None and bool(model.mesh.elements)
        and all(type(element) is _QualifiedE4PLShellElement
                and element.shell_section is None
                and model.get_material(element.material_name).yield_stress == 0.
                and model.get_material(element.material_name).hardening_curve is None
                for element in model.mesh.elements.values())
    )
    augmented_contact_manager = (
        active_manager if getattr(active_manager, "augmented", False) else None
    )
    normal_multiplier_count = (
        len(augmented_contact_manager.normal_multipliers)
        if augmented_contact_manager is not None else None
    )
    restored_static_path: Optional[Dict[str, Any]] = None
    validated_restart = None
    if parsed_restart_checkpoint is not None:
        validated_restart = validate_nonlinear_checkpoint(
            parsed_restart_checkpoint,
            analysis_kind="static",
            model=model,
            analysis_contract=restart_analysis_contract,
            num_layers=num_layers,
        )
        exact_guard(model, context="nonlinear static checkpoint validation")
        if active_manager is not None:
            from .general_contact.finite_rigid_patch import PROFILE as finite_rigid_profile
            if active_manager.profile == finite_rigid_profile:
                active_manager.admit_validated_committed_state(validated_restart)
        restored_static_path = _restore_static_path_state(
            validated_restart.path_state,
            control_name=control_name,
            total_dofs=int(model.mesh.dof_manager.total_dofs),
            normal_multiplier_count=normal_multiplier_count,
        )
        if augmented_contact_manager is not None:
            augmented_contact_manager.set_normal_multipliers(
                restored_static_path["normal_multipliers"]
            )
        if restored_static_path["terminal_status"] != "completed":
            raise ValueError(
                "restart checkpoint does not end at a continuable completed static target"
            )
        if control_name == "force":
            requested_target = (
                load_program.total_factor
                if load_program is not None and max_load_factor == 1.0
                else float(max_load_factor)
            )
            if restored_static_path["load_factor"] > requested_target + 1.0e-12:
                raise ValueError(
                    "restart checkpoint is beyond the requested maximum load factor"
                )
            recorded_deleted = {
                int(record.element_id)
                for record in restored_static_path["deletion_records"]
            }
            if recorded_deleted != validated_restart.deleted_element_ids:
                raise ValueError(
                    "restart checkpoint deletion records and deleted IDs disagree"
                )
            if fracture_config is None and recorded_deleted:
                raise ValueError(
                    "restart checkpoint contains fracture history without fracture_config"
                )
        if validated_restart.activity is not None:
            model.set_element_activity(validated_restart.activity)
        initial_element_states = validated_restart.element_states
        initial_displacements = validated_restart.displacements
        equilibrate_initial_state = False
    _ensure_nonlinear_acceleration()
    model.apply_boundary_conditions()
    exact_guard(model, context="nonlinear static boundary conditions")

    # The constraint transformation only depends on supports/MPCs; the
    # elastic stiffness is assembled once to build it (and warms the element
    # caches used by the nonlinear kernels).
    K0, stiffness_info = assemble_stiffness_matrix(model)
    exact_guard(model, context="nonlinear static stiffness assembly")
    stage_vectors: List[np.ndarray] = []
    stage_infos: List[Dict[str, Any]] = []
    if load_program is not None:
        for stage in load_program.stages:
            vector, stage_info = assemble_load_vector(model, stage.load_case)
            exact_guard(model, context="nonlinear static stage-load assembly")
            stage_vectors.append(vector)
            stage_infos.append({"name": stage.name, "target_factor": stage.target_factor, **stage_info})
        F_prop = np.sum(np.vstack(stage_vectors), axis=0) if stage_vectors else np.zeros(K0.shape[0], dtype=float)
        load_info = {"vector_type": "load_program", "stages": stage_infos}
    else:
        F_prop, load_info = assemble_load_vector(model, load_case)
        exact_guard(model, context="nonlinear static proportional-load assembly")

    if constant_load_case is not None:
        F_const, constant_load_info = assemble_load_vector(model, constant_load_case)
        exact_guard(model, context="nonlinear static constant-load assembly")
    else:
        F_const = np.zeros_like(F_prop)
        constant_load_info = None
    _, _, T, u0, independent_dofs, constraint_info = build_constraint_transformation(
        K0,
        F_prop,
        model,
    )
    if (active_manager is not None
            and hasattr(active_manager, "restrict_tangent_to_reduced_variations")):
        active_manager.restrict_tangent_to_reduced_variations(T)
    exact_guard(model, context="nonlinear static constraint transformation")
    rotating_mpc_map = (
        build_nonlinear_rotating_mpc_map(model, T)
        if kinematics == "corotational"
        else None
    )
    armijo_weights = (
        _armijo_residual_weights(
            model,
            independent_dofs,
            float(settings.characteristic_length),
        )
        if armijo_requested
        else None
    )
    if restored_static_path is not None:
        restored_static_path = _restore_static_path_state(
            validated_restart.path_state,
            control_name=control_name,
            n_red=int(T.shape[1]),
            total_dofs=int(model.mesh.dof_manager.total_dofs),
            normal_multiplier_count=normal_multiplier_count,
        )

    info: Dict[str, Any] = {
        "stiffness": stiffness_info,
        "load": load_info,
        "constant_load": constant_load_info,
        "constraint_info": constraint_info,
        "num_layers": int(num_layers),
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "reduced_dofs": int(T.shape[1]),
        "control": str(control),
        "kinematics": kinematics,
        "corotational_tangent_requested": str(corotational_tangent).lower(),
        "corotational_tangent": resolved_corotational_tangent,
        "follower_pressure": follower_active,
        "equilibrium_tangent": "K_internal-K_external" if follower_active else "K_internal",
        "convergence_settings": settings.to_dict(),
        "resource_config": None if resource_config is None else resource_config.to_dict(),
        "thread_policy": thread_policy_diagnostics(resource_config),
    }
    info["corotational_shell_rotation_stable"] = stable_shell_rotation
    info["corotational_shell_rotation_policy"] = (
        "cancellation_safe_virgin_force_q4" if stable_shell_rotation
        else "historical_owner_policy"
    )
    if stability_control is not None:
        info["stability_control"] = {
            "method": "plate_energy", "max_free_coordinates": 512,
            "initial_radius_over_edge": .1, "maximum_radius_over_edge": 1.,
            "acceptance_ratio": .1, "shrink": .25, "growth": 2.,
            "curvature_rule": "lambda_min > 100*eps*n*norm_inf(scaled_K)",
            "eigensolves": 0, "energy_proposals": 0, "rejected_proposals": 0,
            "accepted_curvatures": [], "attempts": [],
        }
    if validated_restart is not None:
        info["restart"] = {
            "schema": validated_restart.payload["schema"],
            "checkpoint_sha256": validated_restart.payload["checkpoint_sha256"],
            "resumed_load_factor": float(restored_static_path["load_factor"]),
            "resumed_step_index": int(restored_static_path["step_index"]),
        }
    general_tangent = follower_active or (
        kinematics == "corotational"
        and resolved_corotational_tangent == "consistent"
    ) or augmented_contact_manager is not None or getattr(active_manager, "projected_work", False)
    native_spatial_couples = False
    native_combined_couples = False
    native_generalized_combined_couples = False
    if native_generalized_model:
        from ._ge_beam3_native_generalized_program import require_active, require_solver_initial

        require_active(model)
        require_solver_initial(model, initial_element_states, initial_displacements)
        if follower_active:
            raise ValueError("generalized distributed statics do not admit follower pressure")
        general_tangent = True
        info["equilibrium_tangent"] = "GENERAL_GENERALIZED_SECTION_DISTRIBUTED_STATIC_SCHUR"
        info["native_generalized_general_matrix"] = True
        from ._ge_beam3_native_generalized_combined_couples import active_for

        native_generalized_combined_couples = active_for(model)
        if native_generalized_combined_couples:
            info['equilibrium_tangent'] = 'K_generalized_static_Schur-K_spatial_nodal_chart_external'
            info['native_generalized_combined_couple_general_matrix'] = True
    native_distributed_model = any(
        type(e).__module__ == 'anyfesolver._ge_beam3_native_distributed_element'
        for e in model.mesh.elements.values()
    )
    if native_distributed_model:
        from ._ge_beam3_native_distributed_program import require_active, require_solver_initial

        require_active(model)
        require_solver_initial(model, initial_element_states, initial_displacements)
        if (control_name != 'force' or load_program is not None or follower_active or fracture_config is not None
                or initial_fields):
            raise ValueError('private distributed couples require authenticated standalone force control')
        general_tangent = True
        info['equilibrium_tangent'] = 'GENERAL_DISTRIBUTED_COUPLE_STATIC_SCHUR'
        info['native_distributed_couple_general_matrix'] = True
        from ._ge_beam3_native_combined_couples import active_for

        native_combined_couples = active_for(model)
        if native_combined_couples:
            info['equilibrium_tangent'] = 'K_distributed_static_Schur-K_spatial_nodal_chart_external'
            info['native_combined_couple_general_matrix'] = True
    native_line_model = any(
        type(e).__module__ == 'anyfesolver._ge_beam3_native_line_static_element'
        for e in model.mesh.elements.values()
    )
    if native_line_model:
        from ._ge_beam3_native_spatial_couples import active_for

        native_spatial_couples = active_for(model)
        if native_spatial_couples:
            if control_name != 'force' or load_program is not None or follower_active or fracture_config is not None:
                raise ValueError('private spatial couples require standalone force control')
            general_tangent = True
            info['equilibrium_tangent'] = 'K_internal-K_spatial_couple_chart_external'
            info['native_spatial_couple_general_matrix'] = True
    dead_load_projection_eligible = not (
        follower_active
        or fracture_config is not None
        or rotating_mpc_map is not None
        or native_line_model
        or native_distributed_model
        or native_generalized_model
        or native_spatial_couples
        or native_combined_couples
        or native_generalized_combined_couples
    )
    if dead_load_projection_eligible:
        F_const_reduced = np.asarray(T.T @ F_const, dtype=float).reshape(-1)
        F_prop_reduced = np.asarray(T.T @ F_prop, dtype=float).reshape(-1)
        stage_vectors_reduced = [
            np.asarray(T.T @ vector, dtype=float).reshape(-1)
            for vector in stage_vectors
        ]
    else:
        F_const_reduced = None
        F_prop_reduced = None
        stage_vectors_reduced = []
    info["external_load_reduction"] = {
        "preprojected": bool(dead_load_projection_eligible),
        "constant_vector": bool(dead_load_projection_eligible),
        "proportional_vector": bool(
            dead_load_projection_eligible and load_program is None
        ),
        "stage_vector_count": (
            len(stage_vectors_reduced) if dead_load_projection_eligible else 0
        ),
    }
    info["armijo"] = {
        "enabled": bool(armijo_requested),
        "characteristic_length": (
            float(settings.characteristic_length) if armijo_requested else None
        ),
        "residual_scaling": (
            "translation_force_rotation_moment_over_length"
            if armijo_requested
            else None
        ),
        "sufficient_decrease": 1.0e-4 if armijo_requested else None,
    }
    imperfection_provenance: List[Dict[str, Any]] = []
    if imperfection is not None:
        imperfection_provenance = list(getattr(model, "imperfection_metadata", []))
        info["imperfection"] = imperfection_provenance

    committed_states, residual_field_provenance = _prepare_initial_states(
        model,
        initial_element_states,
        initial_fields,
        num_layers,
    )
    if kinematics == "corotational" and any(
        _state_has_initial_field(state) for state in committed_states.values()
    ):
        raise NotImplementedError(
            "Initial stress/prestrain fields are currently qualified only for "
            "kinematics='von_karman' in element-local reference coordinates."
        )
    info["initial_condition_provenance"] = {
        "geometric_imperfection": imperfection_provenance,
        "residual_stress_or_prestrain": residual_field_provenance,
        "coordinate_system": {
            "geometric_imperfection": "global nodal coordinates",
            "residual_stress_or_prestrain": "element-local reference coordinates",
        },
    }

    if fracture_config is not None and control_name != "force":
        raise ValueError("fracture_config is currently supported only with force control")

    n_red = int(T.shape[1])
    checkpoint_resume = validated_restart is not None
    force_restart = (
        control_name == "force"
        and initial_displacements is not None
        and not checkpoint_resume
    )
    initial_linear_displacements = (
        rotating_mpc_map.to_linear_displacement(initial_displacements)
        if rotating_mpc_map is not None and initial_displacements is not None
        else initial_displacements
    )
    if control_name == "force" and checkpoint_resume:
        resumed_lam = float(restored_static_path["load_factor"])
        q = np.asarray(
            restored_static_path["reduced_coordinates"], dtype=float
        ).copy()
        supplied_full = np.asarray(
            initial_displacements, dtype=np.float64
        ).reshape(-1)
        supplied_linear = np.asarray(
            initial_linear_displacements, dtype=np.float64
        ).reshape(-1)
        restored_mode = restored_static_path.get("affine_path_mode")
        restored_base = restored_static_path.get("exact_base_offset")
        projected_free = np.asarray(T @ q, dtype=np.float64).reshape(-1)
        proportional_affine = np.asarray(
            resumed_lam * np.asarray(u0, dtype=np.float64).reshape(-1),
            dtype=np.float64,
        )
        if restored_mode is None:
            # Pre-schema checkpoints are migrated from their independently
            # hash-bound committed displacement.  Exact equality identifies
            # the proportional path; every other compatible legacy state is
            # retained as a fixed affine restart rather than projected onto a
            # different path.
            proportional_projection = projected_free + proportional_affine
            if np.array_equal(proportional_projection, supplied_linear):
                force_affine_path_mode = "PROPORTIONAL_PRESCRIBED_FIELD"
                force_exact_base_offset = np.zeros_like(supplied_linear)
            else:
                force_affine_path_mode = "FIXED_RESTART_AFFINE_STATE"
                force_exact_base_offset = _exact_binary64_additive_offset(
                    supplied_linear,
                    projected_free,
                    context="legacy static restart affine state",
                )
            info_migration = "MIGRATED_FROM_BOUND_COMMITTED_DISPLACEMENT"
        else:
            force_affine_path_mode = str(restored_mode)
            force_exact_base_offset = np.asarray(
                restored_base, dtype=np.float64
            ).reshape(-1).copy()
            info_migration = "CURRENT_EXACT_BASE_SCHEMA"
        path_projection = projected_free.copy()
        if force_affine_path_mode == "PROPORTIONAL_PRESCRIBED_FIELD":
            path_projection = path_projection + proportional_affine
            initial_affine_scale = resumed_lam
        else:
            # The scale is diagnostic-only for constraint reporting.  The
            # exact fixed base below remains the sole reconstruction authority.
            _unused_q, initial_affine_scale = (
                _reduced_coordinates_with_affine_scale(T, u0, supplied_linear)
            )
        initial_affine_offset = force_exact_base_offset
        initial_committed_displacement = np.asarray(
            path_projection + force_exact_base_offset, dtype=np.float64
        ).reshape(-1)
        if rotating_mpc_map is not None:
            initial_committed_displacement = np.asarray(
                rotating_mpc_map.apply_kinematics(initial_committed_displacement),
                dtype=np.float64,
            ).reshape(-1)
        if not np.array_equal(initial_committed_displacement, supplied_full):
            raise ValueError(
                "restart checkpoint affine path does not exactly reconstruct "
                "its committed displacement"
            )
    elif control_name == "force":
        if initial_displacements is not None and model.mesh.elements and all(
            type(element).__module__ == "anyfesolver._ge_beam3_native_fibre_static_element"
            for element in model.mesh.elements.values()
        ):
            from ._ge_beam3_native_fibre_restart import supported_solver_coordinates

            q, initial_affine_scale = supported_solver_coordinates(
                model, T, u0, initial_displacements
            )
        elif initial_displacements is not None and model.mesh.elements and all(
            type(element).__module__ == "anyfesolver._ge_beam3_native_line_static_element"
            for element in model.mesh.elements.values()
        ):
            from ._ge_beam3_native_line_restart import supported_solver_coordinates

            q, initial_affine_scale = supported_solver_coordinates(
                model, T, u0, initial_displacements
            )
        elif initial_displacements is not None and model.mesh.elements and all(
            type(element).__module__ == "anyfesolver._ge_beam3_native_generalized_element"
            for element in model.mesh.elements.values()
        ):
            from ._ge_beam3_native_generalized_restart import supported_solver_coordinates

            q, initial_affine_scale = supported_solver_coordinates(
                model, T, u0, initial_displacements
            )
        elif initial_displacements is not None and model.mesh.elements and all(
            type(element).__module__ == "anyfesolver._ge_beam3_native_distributed_element"
            for element in model.mesh.elements.values()
        ):
            from ._ge_beam3_native_distributed_restart import supported_solver_coordinates

            q, initial_affine_scale = supported_solver_coordinates(
                model, T, u0, initial_displacements
            )
        else:
            q, initial_affine_scale = _reduced_coordinates_with_affine_scale(
                T,
                u0,
                initial_linear_displacements,
            )
        if initial_displacements is None:
            force_affine_path_mode = "PROPORTIONAL_PRESCRIBED_FIELD"
            force_exact_base_offset = np.zeros(int(T.shape[0]), dtype=float)
            initial_affine_offset = force_exact_base_offset
            initial_committed_displacement = np.asarray(
                T @ q, dtype=np.float64
            ).reshape(-1)
            if rotating_mpc_map is not None:
                initial_committed_displacement = np.asarray(
                    rotating_mpc_map.apply_kinematics(initial_committed_displacement),
                    dtype=np.float64,
                ).reshape(-1)
            info_migration = "NOT_A_RESTART"
        else:
            # Retain the supplied committed vector bit for bit.  LSQR proves
            # compatibility but its reconstructed affine scale can differ by
            # a few ulps; binding a state to that rounded projection would
            # reject a checkpoint/result produced by this same solver.  The
            # exact residual is a constant restart offset, so
            # ``T @ q + offset`` reconstructs the supplied state exactly while
            # keeping the established fixed-affine restart path.
            supplied_full = np.asarray(
                initial_displacements, dtype=np.float64
            ).reshape(-1)
            supplied_linear = np.asarray(
                initial_linear_displacements, dtype=np.float64
            ).reshape(-1)
            projected_free = np.asarray(T @ q, dtype=np.float64).reshape(-1)
            force_affine_path_mode = "FIXED_RESTART_AFFINE_STATE"
            force_exact_base_offset = _exact_binary64_additive_offset(
                supplied_linear,
                projected_free,
                context="initial_displacements fixed-affine restart path",
            )
            initial_affine_offset = force_exact_base_offset
            initial_committed_displacement = np.asarray(
                projected_free + force_exact_base_offset,
                dtype=np.float64,
            ).reshape(-1)
            if rotating_mpc_map is not None:
                initial_committed_displacement = np.asarray(
                    rotating_mpc_map.apply_kinematics(initial_committed_displacement),
                    dtype=np.float64,
                ).reshape(-1)
            info_migration = "DIRECT_EXACT_INITIAL_DISPLACEMENT"
    else:
        q = (
            np.asarray(restored_static_path["reduced_coordinates"], dtype=float).copy()
            if checkpoint_resume
            else _reduced_coordinates(T, u0, initial_linear_displacements)
        )
        initial_affine_scale = 1.0
        initial_affine_offset = np.asarray(u0, dtype=float).reshape(-1)
        initial_committed_displacement = np.asarray(
            T @ q + np.asarray(u0, dtype=float).reshape(-1),
            dtype=float,
        ).reshape(-1)
        if rotating_mpc_map is not None:
            initial_committed_displacement = rotating_mpc_map.apply_kinematics(
                initial_committed_displacement
            )
    if checkpoint_resume and not np.array_equal(
        initial_committed_displacement,
        np.asarray(initial_displacements, dtype=float).reshape(-1),
    ):
        raise ValueError(
            "restart checkpoint reduced coordinates do not exactly reconstruct "
            "its committed displacement"
        )
    supplied_state_ids = tuple(
        sorted(int(value) for value in (initial_element_states or {}))
    )
    preflight_deleted_ids = (
        tuple(sorted(int(value) for value in validated_restart.deleted_element_ids))
        if validated_restart is not None
        else ()
    )
    preflight_deletion_records = (
        tuple(restored_static_path["deletion_records"])
        if restored_static_path is not None and control_name == "force"
        else ()
    )
    deleted_qualified_dispositions: Dict[int, Any] = {}
    committed_states = _prepare_qualified_q4_states_for_nonlinear_solve(
        model,
        initial_committed_displacement,
        committed_states,
        num_layers,
        info,
        supplied_element_ids=supplied_state_ids,
        kinematics=kinematics,
        ordinary_restart=bool(
            initial_element_states is not None
            and initial_displacements is not None
            and equilibrate_initial_state is not True
        ),
        allow_explicit_initial_material_states=bool(
            validated_restart is None
            and initial_element_states is not None
            and initial_displacements is None
        ),
        expected_deleted_element_ids=preflight_deleted_ids,
        deletion_records=preflight_deletion_records,
        residual_stiffness_fraction=(
            None
            if fracture_config is None
            else float(fracture_config.residual_stiffness_fraction)
        ),
        deleted_dispositions=deleted_qualified_dispositions,
    )
    # Native multiplicative rotations are solver-owned and must exist before
    # the first possible constitutive evaluation (initial-field equilibrium or
    # a fully constrained prescribed state).  Q4-only models retain the prior
    # dictionary/array activation behavior.
    committed_states = _activate_nonlinear_state_storage(
        model,
        committed_states,
        num_layers,
        info,
        kinematics=kinematics,
        committed_displacements=initial_committed_displacement,
        noncurrent_native_element_ids=preflight_deleted_ids,
    )
    if (
        initial_fields_present
        and initial_displacements is not None
        and equilibrate_initial_state is not False
    ):
        raise ValueError(
            "initial_fields combined with initial_displacements is ambiguous during "
            "automatic equilibration; pass equilibrate_initial_state=False for an "
            "already-equilibrated restart."
        )
    if n_red == 0:
        # The rotating map corrects only its own slave translations, so the
        # fully constrained return must use the mapped physical state in
        # every branch, mapping each still-linear vector exactly once.  A
        # checkpoint resume and an ordinary force restart already hold the
        # mapped committed state (for n_red == 0 the force restart's
        # projected free motion is zero, so it equals map(offset)).  The
        # fresh prescribed-state path must expand the raw offset u0: fresh
        # force control commits only T @ q, which is zero here and would
        # silently discard the prescribed master motion.
        if checkpoint_resume or force_restart:
            fully_constrained_displacement = np.asarray(
                initial_committed_displacement, dtype=float
            ).reshape(-1).copy()
        else:
            fully_constrained_displacement = np.asarray(
                u0, dtype=float
            ).reshape(-1)
            if rotating_mpc_map is not None:
                fully_constrained_displacement = np.asarray(
                    rotating_mpc_map.apply_kinematics(
                        fully_constrained_displacement
                    ),
                    dtype=float,
                ).reshape(-1)
            fully_constrained_displacement = np.asarray(
                fully_constrained_displacement, dtype=float
            ).reshape(-1).copy()
        has_initial_fields = any(
            _state_has_initial_field(state) for state in committed_states.values()
        )
        requires_native_tl_evaluation = any(
            bool(
                getattr(
                    element,
                    "formulation_native_total_lagrangian",
                    False,
                )
            )
            for element in model.mesh.elements.values()
        )
        requested_equilibration = (
            initial_fields_present
            if equilibrate_initial_state is None
            else bool(equilibrate_initial_state)
        )
        info["initial_state_equilibration"] = {
            "requested": requested_equilibration,
            "converged": True,
            "iterations": 0,
            "history": [],
            "failure_reason": None,
            "free_dof_residual_norm": 0.0,
            "fully_constrained": True,
        }
        if has_initial_fields or requires_native_tl_evaluation:
            # There are no admissible displacement corrections, so the
            # prescribed field/native kinematics are already equilibrated in
            # the free-DOF sense; any imbalance is carried by reactions.
            # Native-TL elements still require one real evaluation so their
            # accepted element-owned kinematic state is not silently omitted.
            internal_force, _unused_tangent, trial_states = (
                _assemble_nonlinear_system(
                    model,
                    fully_constrained_displacement,
                    committed_states,
                    num_layers,
                    # A fully constrained plastic Q4 still needs the exact
                    # accepted return-map parent for its committed tangent.
                    # Evaluate that one accepted candidate with its tangent;
                    # never promote an origin-less residual-only output.
                    tangent=True,
                    kinematics=kinematics,
                    corotational_tangent=resolved_corotational_tangent,
                    stable_shell_rotation=stable_shell_rotation,
                )
            )
            internal_guard(
                model,
                context="nonlinear static fully-constrained assembly",
            )
            committed_states = _commit_nonlinear_state_candidate(
                committed_states,
                trial_states,
                model=model,
                accepted_displacements=fully_constrained_displacement,
            )
            info["initial_state_equilibration"][
                "constrained_internal_force_norm"
            ] = float(np.linalg.norm(internal_force))
            if requires_native_tl_evaluation:
                info["initial_state_equilibration"]["native_tl_evaluated"] = True
            info["strain_summary"] = _nonlinear_state_summary(committed_states)
        info["failure_reason"] = "empty_reduced_system"
        info["stop_reason"] = "empty_reduced_system"
        info["status_category"] = _nonlinear_status_category("empty_reduced_system", "empty_reduced_system")
        info["result_case"] = make_result_case(
            name="nonlinear_static",
            analysis_type="nonlinear_static",
            load_cases=tuple(stage.load_case for stage in load_program.stages) if load_program is not None else (() if load_case is None else (load_case,)),
            assembly_info={"stiffness": stiffness_info, "load": load_info},
            solver_info={"convergence_info": {"status": "empty_reduced_system"}},
            recovery={"displacements": True, "element_states": True},
            settings={
                "control": control_name,
                "num_steps": num_steps,
                "num_layers": num_layers,
                "kinematics": kinematics,
                "corotational_tangent": resolved_corotational_tangent,
            },
        ).to_dict()
        final_committed_states = _materialize_final_nonlinear_states(
            committed_states,
            info,
        )
        final_committed_states = _finalize_nonlinear_element_states(
            model,
            fully_constrained_displacement,
            final_committed_states,
            num_layers,
            kinematics=kinematics,
        )
        final_committed_states = _seal_final_qualified_q4_states(
            model,
            fully_constrained_displacement,
            final_committed_states,
            num_layers,
            info,
            kinematics=kinematics,
        )
        return NonlinearStaticResult(
            [],
            "empty_reduced_system",
            fully_constrained_displacement.copy(),
            0.0,
            final_committed_states,
            info,
        )

    has_initial_fields = any(_state_has_initial_field(state) for state in committed_states.values())
    restored_initial_fields = has_initial_fields and not initial_fields_present
    if (
        restored_initial_fields
        and initial_displacements is None
        and equilibrate_initial_state is not True
    ):
        raise ValueError(
            "Field-bearing initial_element_states require matching "
            "initial_displacements for a restart. Pass "
            "equilibrate_initial_state=True only when re-equilibration from "
            "zero displacement is intended."
        )
    should_equilibrate = (
        initial_fields_present
        if equilibrate_initial_state is None
        else bool(equilibrate_initial_state)
    )
    if should_equilibrate and has_initial_fields:
        q, committed_states, initialization_history, initialization_failure = _equilibrate_initial_fields(
            model=model,
            T=T,
            u0=initial_affine_offset,
            committed_states=committed_states,
            num_layers=num_layers,
            max_iterations=max_iterations,
            tolerance=tolerance,
            kinematics=kinematics,
            corotational_tangent=resolved_corotational_tangent,
            general_tangent=general_tangent,
            initial_reduced_displacements=q,
            cancellation_token=cancellation_token,
            _exact_guard=exact_guard,
        )
        exact_guard(model, context="nonlinear static initial equilibration")
        info["initial_state_equilibration"] = {
            "requested": True,
            "converged": initialization_failure is None,
            "iterations": len(initialization_history),
            "history": initialization_history,
            "failure_reason": initialization_failure,
        }
        if initialization_failure is not None:
            u_failed = np.asarray(
                T @ q + initial_affine_offset,
                dtype=float,
            ).reshape(-1)
            info["failure_reason"] = initialization_failure
            info["stop_reason"] = initialization_failure
            info["status_category"] = _nonlinear_status_category("diverged", initialization_failure)
            committed_states = _materialize_final_nonlinear_states(
                committed_states,
                info,
            )
            # The installed result-recovery finalizer may evaluate mechanics
            # and add displacement-derived fields.  No candidate was accepted
            # on this failure path, so the last committed materialized pair is
            # the only honest output and is marked nonauthoritative below.
            committed_states = _mark_failed_qualified_q4_states(
                model,
                u_failed,
                committed_states,
                num_layers,
                info,
                failure_reason=initialization_failure,
                kinematics=kinematics,
            )
            info["strain_summary"] = _nonlinear_state_summary(committed_states)
            info["solve_time"] = time.time() - start_time
            return NonlinearStaticResult([], "diverged", u_failed, 0.0, committed_states, info)
    else:
        info["initial_state_equilibration"] = {
            "requested": should_equilibrate,
            "converged": True,
            "iterations": 0,
            "history": [],
            "failure_reason": None,
        }
    steps: List[NonlinearStaticStep] = (
        list(restored_static_path["steps"])
        if restored_static_path is not None
        else []
    )
    status = "completed"
    deleted_element_ids: set[int] = (
        set(validated_restart.deleted_element_ids)
        if validated_restart is not None
        else set()
    )
    deletion_records: List[DeletedElementRecord] = (
        list(restored_static_path["deletion_records"])
        if restored_static_path is not None and control_name == "force"
        else []
    )
    fracture_warnings: List[str] = (
        list(restored_static_path["fracture_warnings"])
        if restored_static_path is not None and control_name == "force"
        else []
    )
    max_fracture_utilization = (
        float(restored_static_path["max_fracture_utilization"])
        if restored_static_path is not None and control_name == "force"
        else 0.0
    )

    if load_program is not None and max_load_factor == 1.0:
        target_load_factor = load_program.total_factor
    else:
        target_load_factor = float(max_load_factor)

    def _active_load_case(case: Optional["LoadCase"]) -> Optional["LoadCase"]:
        if not deleted_element_ids:
            return case
        return filtered_load_case_for_deleted_elements(case, deleted_element_ids)

    def external_load_at(
        path_factor: float,
        displacements: np.ndarray,
        *,
        tangent: bool,
    ) -> Tuple[np.ndarray, Optional[sparse.csr_matrix], Dict[str, float], Optional[str]]:
        if not follower_active and not deleted_element_ids:
            if load_program is None:
                factors = {"proportional": float(path_factor)}
                force = F_const + float(path_factor) * F_prop
                active_stage = None
            else:
                factors = load_program.stage_factors(path_factor)
                force = F_const.copy()
                for stage, vector in zip(load_program.stages, stage_vectors):
                    force += factors[stage.name] * vector
                active_stage = load_program.active_stage(path_factor)
            zero_tangent = (
                sparse.csr_matrix((force.size, force.size), dtype=float)
                if tangent
                else None
            )
            if native_spatial_couples or native_combined_couples or native_generalized_combined_couples:
                if native_generalized_combined_couples:
                    from ._ge_beam3_native_generalized_combined_couples import external_at
                elif native_combined_couples:
                    from ._ge_beam3_native_combined_couples import external_at
                else:
                    from ._ge_beam3_native_spatial_couples import external_at

                extra, extra_tangent = external_at(
                    model, committed_states, displacements, path_factor, tangent=tangent
                )
                force = force + extra
                if tangent:
                    zero_tangent = zero_tangent + extra_tangent
            if native_line_model or native_distributed_model or native_generalized_model:
                with np.errstate(over='ignore', invalid='ignore'):
                    load_norm = float(np.linalg.norm(force))
                if not np.all(np.isfinite(force)) or not np.isfinite(load_norm):
                    raise ValueError('native external load exceeds finite norm range')
            return force, zero_tangent, factors, active_stage

        weighted_cases: List[Tuple[Optional["LoadCase"], float]] = [
            (_active_load_case(constant_load_case), 1.0)
        ]
        if load_program is None:
            factors = {"proportional": float(path_factor)}
            weighted_cases.append((_active_load_case(load_case), float(path_factor)))
            active_stage = None
        else:
            factors = load_program.stage_factors(path_factor)
            weighted_cases.extend(
                (_active_load_case(stage.load_case), factors[stage.name])
                for stage in load_program.stages
            )
            active_stage = load_program.active_stage(path_factor)
        force, load_tangent = _weighted_external_load_system(
            model,
            weighted_cases,
            displacements,
            tangent=tangent,
        )
        return force, load_tangent, factors, active_stage

    last_external_force_full: Optional[np.ndarray] = None

    def reduced_external_load_at(
        path_factor: float,
        displacements: np.ndarray,
        *,
        tangent: bool,
    ) -> Tuple[
        np.ndarray,
        Optional[sparse.csr_matrix],
        Dict[str, float],
        Optional[str],
    ]:
        """Return reduced loads, reusing fixed dead-load projections."""
        nonlocal last_external_force_full

        if dead_load_projection_eligible and not deleted_element_ids:
            record_nonlinear_solver_event("dead_load_projection_reuse")
            assert F_const_reduced is not None
            if load_program is None:
                assert F_prop_reduced is not None
                factors = {"proportional": float(path_factor)}
                force_reduced = (
                    F_const_reduced + float(path_factor) * F_prop_reduced
                )
                active_stage = None
            else:
                factors = load_program.stage_factors(path_factor)
                force_reduced = F_const_reduced.copy()
                for stage, vector in zip(
                    load_program.stages,
                    stage_vectors_reduced,
                ):
                    force_reduced += factors[stage.name] * vector
                active_stage = load_program.active_stage(path_factor)
            return force_reduced, None, factors, active_stage

        force, load_tangent, factors, active_stage = external_load_at(
            path_factor,
            displacements,
            tangent=tangent,
        )
        last_external_force_full = force
        if rotating_mpc_map is not None:
            force_reduced = rotating_mpc_map.reduce_force(displacements, force)
            reduced_tangent = None
            if tangent and load_tangent is not None and load_tangent.nnz:
                J_mpc = rotating_mpc_map.jacobian(displacements)
                reduced_tangent = (J_mpc.T @ load_tangent @ J_mpc).tocsr()
            return force_reduced, reduced_tangent, factors, active_stage
        force_reduced = np.asarray(T.T @ force, dtype=float).reshape(-1)
        reduced_tangent = None
        if tangent and load_tangent is not None and load_tangent.nnz:
            reduced_tangent = (T.T @ load_tangent @ T).tocsr()
        return force_reduced, reduced_tangent, factors, active_stage

    def external_load_guard(*, context: str) -> Dict[str, Any]:
        """Use the lease for the preassembled affine load path only."""

        guard = (
            internal_guard
            if not follower_active and not deleted_element_ids
            else exact_guard
        )
        return guard(model, context=context)

    if control_name == "displacement":
        if displacement_control is None:
            raise ValueError("displacement_control is required when control='displacement'")
        preload_result: Optional[NonlinearStaticResult] = None
        if (
            load_program is not None
            and len(load_program.stages) > 1
            and not checkpoint_resume
        ):
            # Each permanent stage is first solved by force control and its
            # displacement/material state committed.  The final stage then
            # starts from that exact checkpoint under displacement control;
            # the earlier loads remain constant in the controlled equilibrium.
            preload_program = NonlinearLoadProgram(tuple(load_program.stages[:-1]))
            preload_result = _solve_static_nonlinear_under_lease(
                model,
                constant_load_case=constant_load_case,
                max_load_factor=preload_program.total_factor,
                num_steps=max(num_steps, len(preload_program.stages)),
                max_iterations=max_iterations,
                tolerance=tolerance,
                num_layers=num_layers,
                min_step_fraction=min_step_fraction,
                load_program=preload_program,
                control="force",
                initial_element_states=committed_states,
                initial_displacements=initial_committed_displacement.copy(),
                equilibrate_initial_state=False,
                convergence_settings=settings,
                resource_config=resource_config,
                kinematics=kinematics,
                corotational_tangent=(
                    "auto" if kinematics == "von_karman" else resolved_corotational_tangent
                ),
                status_callback=status_callback,
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
                record_increment_snapshots=record_increment_snapshots,
                _qualified_runtime_guard=_qualified_runtime_guard,
            )
            exact_guard(model, context="nonlinear static preload result")
            if not preload_result.converged or (
                preload_result.load_factor < preload_program.total_factor - 1.0e-10
            ):
                preload_result.info["requested_control"] = "displacement"
                preload_result.info["failed_phase"] = "load_program_preload"
                preload_result.info["initial_condition_provenance"] = info[
                    "initial_condition_provenance"
                ]
                return preload_result
            committed_states = _copy_initial_states(preload_result.element_states)
            preload_linear_displacements = (
                rotating_mpc_map.to_linear_displacement(preload_result.displacements)
                if rotating_mpc_map is not None
                else preload_result.displacements
            )
            q = _reduced_coordinates(T, u0, preload_linear_displacements)
            committed_states = _prepare_qualified_q4_states_for_nonlinear_solve(
                model,
                np.asarray(preload_result.displacements, dtype=float).reshape(-1),
                committed_states,
                num_layers,
                info,
                supplied_element_ids=tuple(sorted(int(value) for value in committed_states)),
                ordinary_restart=True,
                kinematics=kinematics,
            )
            info["load_program_preload"] = {
                "status": preload_result.status,
                "load_factor": preload_result.load_factor,
                "stage_factors": preload_result.info.get("load_program_stage_factors", {}),
                "steps": len(preload_result.steps),
                "total_newton_iterations": preload_result.info.get("total_newton_iterations", 0),
                "strain_summary": preload_result.info.get("strain_summary", {}),
                "force_displacement_history": preload_result.info.get(
                    "force_displacement_history",
                    [],
                ),
            }
            info["material_history_reused_from_preload"] = True

        displacement_committed_u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
        if rotating_mpc_map is not None:
            displacement_committed_u = rotating_mpc_map.apply_kinematics(
                displacement_committed_u
            )
        committed_states = _activate_nonlinear_state_storage(
            model,
            committed_states,
            num_layers,
            info,
            kinematics=kinematics,
            committed_displacements=displacement_committed_u,
        )
        controlled_result = _solve_static_displacement_control(
            model=model,
            T=T,
            u0=u0,
            F_const=F_const,
            F_prop=F_prop,
            stage_vectors=stage_vectors,
            load_case=load_case,
            constant_load_case=constant_load_case,
            load_program=load_program,
            displacement_control=displacement_control,
            committed_states=committed_states,
            num_layers=num_layers,
            num_steps=num_steps,
            max_iterations=max_iterations,
            tolerance=tolerance,
            kinematics=kinematics,
            corotational_tangent=resolved_corotational_tangent,
            info=info,
            start_time=start_time,
            resource_config=resource_config,
            initial_reduced_displacements=q,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
            record_increment_snapshots=record_increment_snapshots,
            initial_load_factor=(
                float(restored_static_path["load_factor"])
                if restored_static_path is not None
                else 0.0
            ),
            recover_initial_load_factor=(
                restored_static_path is None
                and rotating_mpc_map is not None
                and initial_displacements is not None
                and initial_element_states is not None
                and equilibrate_initial_state is False
            ),
            step_index_offset=(
                int(restored_static_path["step_index"])
                if restored_static_path is not None
                else 0
            ),
            initial_steps=(
                restored_static_path["steps"]
                if restored_static_path is not None
                else ()
            ),
            initial_history=(
                restored_static_path["force_displacement_history"]
                if restored_static_path is not None
                else ()
            ),
            initial_total_iterations=(
                int(restored_static_path["total_iterations"])
                if restored_static_path is not None
                else 0
            ),
            restart_analysis_contract=(
                restart_analysis_contract if emit_restart_checkpoint else None
            ),
            rotating_mpc_map=rotating_mpc_map,
            _exact_guard=exact_guard,
        )
        exact_guard(model, context="nonlinear static displacement-control result")
        controlled_result.element_states = _seal_final_qualified_q4_states(
            model,
            controlled_result.displacements,
            controlled_result.element_states,
            num_layers,
            controlled_result.info,
            kinematics=kinematics,
        )
        if controlled_result.restart_checkpoint is not None:
            # An accelerated displacement-control implementation may have
            # built its checkpoint immediately before the public lifecycle
            # seam. Recreate it over the sealed states without changing its
            # already canonical path transaction.
            prior_checkpoint = controlled_result.restart_checkpoint
            controlled_result.restart_checkpoint = create_nonlinear_checkpoint(
                analysis_kind="static",
                model=model,
                analysis_contract=restart_analysis_contract,
                displacements=controlled_result.displacements,
                element_states=controlled_result.element_states,
                deleted_element_ids=tuple(
                    int(value)
                    for value in prior_checkpoint.get("deleted_element_ids", ())
                ),
                path_state=prior_checkpoint["path_state"],
            )
        if load_program is not None:
            stage_factors = {
                stage.name: stage.target_factor
                for stage in load_program.stages[:-1]
            }
            stage_factors[load_program.stages[-1].name] = (
                controlled_result.load_factor * load_program.stages[-1].target_factor
            )
            controlled_result.info["load_program_stage_factors"] = stage_factors
        if preload_result is not None:
            preload_steps = [
                dataclass_replace(step, step_index=index)
                for index, step in enumerate(preload_result.steps, start=1)
            ]
            offset = len(preload_steps)
            controlled_steps = [
                dataclass_replace(step, step_index=offset + index)
                for index, step in enumerate(controlled_result.steps, start=1)
            ]
            controlled_result.steps = preload_steps + controlled_steps
            preload_snapshots = [
                dataclass_replace(snapshot, step_index=index)
                for index, snapshot in enumerate(preload_result.snapshots, start=1)
            ]
            controlled_snapshots = [
                dataclass_replace(snapshot, step_index=offset + index)
                for index, snapshot in enumerate(controlled_result.snapshots, start=1)
            ]
            controlled_result.snapshots = tuple(preload_snapshots + controlled_snapshots)
            controlled_result.info["total_newton_iterations"] = int(
                controlled_result.info.get("total_newton_iterations", 0)
            ) + int(preload_result.info.get("total_newton_iterations", 0))
            if emit_restart_checkpoint:
                checkpoint_history = copy.deepcopy(
                    preload_result.info.get("force_displacement_history", [])
                ) + copy.deepcopy(
                    controlled_result.info.get("force_displacement_history", [])
                )
                for index, item in enumerate(checkpoint_history, start=1):
                    if isinstance(item, dict):
                        item["step_index"] = index
                controlled_result.info[
                    "force_displacement_history"
                ] = checkpoint_history
                controlled_result.restart_checkpoint = create_nonlinear_checkpoint(
                    analysis_kind="static",
                    model=model,
                    analysis_contract=restart_analysis_contract,
                    displacements=controlled_result.displacements,
                    element_states=controlled_result.element_states,
                    deleted_element_ids=(),
                    path_state=_static_restart_path_payload(
                        control_name="displacement",
                        load_factor=controlled_result.load_factor,
                        step_index=len(controlled_result.steps),
                        total_iterations=int(
                            controlled_result.info.get("total_newton_iterations", 0)
                        ),
                        steps=controlled_result.steps,
                        force_displacement_history=checkpoint_history,
                        reduced_coordinates=np.asarray(
                            controlled_result.restart_checkpoint["path_state"][
                                "reduced_coordinates"
                            ],
                            dtype=float,
                        ),
                        terminal_status=controlled_result.status,
                        failure_reason=controlled_result.failure_reason,
                        **({"normal_multipliers":
                            augmented_contact_manager.normal_multipliers.tolist()}
                           if augmented_contact_manager is not None else {}),
                    ),
                )
        return controlled_result

    committed_states = _activate_nonlinear_state_storage(
        model,
        committed_states,
        num_layers,
        info,
        kinematics=kinematics,
        committed_displacements=initial_committed_displacement,
        noncurrent_native_element_ids=preflight_deleted_ids,
    )
    if restored_static_path is not None:
        base_step = float(restored_static_path["base_step"])
        min_step = float(restored_static_path["minimum_step"])
        max_step = float(restored_static_path["maximum_step"])
        step_size = float(restored_static_path["next_step_size"])
        lam = float(restored_static_path["load_factor"])
        step_index = int(restored_static_path["step_index"])
        total_iterations = int(restored_static_path["total_iterations"])
    else:
        base_step = target_load_factor / num_steps
        min_step = max(float(effective_min_step_fraction) * base_step, 1.0e-12)
        max_step = max(base_step * settings.max_step_factor, min_step)
        step_size = base_step
        lam = 0.0
        step_index = 0
        total_iterations = 0
    stage_boundaries = (
        np.cumsum([stage.target_factor for stage in load_program.stages], dtype=float)
        if load_program is not None
        else np.empty(0, dtype=float)
    )

    prescribed_offset = np.asarray(u0, dtype=float).reshape(-1)
    prescribed_base = np.asarray(
        force_exact_base_offset, dtype=np.float64
    ).reshape(-1).copy()
    prescribed_slope = (
        prescribed_offset
        if force_affine_path_mode == "PROPORTIONAL_PRESCRIBED_FIELD"
        else np.zeros_like(prescribed_offset)
    )
    info["prescribed_displacement_path"] = {
        "mode": (
            "proportional_to_load_factor"
            if force_affine_path_mode == "PROPORTIONAL_PRESCRIBED_FIELD"
            else "restart_fixed_affine_state"
        ),
        "schema_mode": force_affine_path_mode,
        "restart_schema_disposition": info_migration,
        "initial_affine_scale": float(initial_affine_scale),
        "affine_scale_slope": (
            1.0
            if force_affine_path_mode == "PROPORTIONAL_PRESCRIBED_FIELD"
            else 0.0
        ),
        "target_max_abs": float(np.max(np.abs(prescribed_offset)))
        if prescribed_offset.size
        else 0.0,
    }

    def full_displacement(q_reduced: np.ndarray, path_factor: float) -> np.ndarray:
        """Expand a force-control state on its affine constraint path."""
        u_lin = np.asarray(
            T @ q_reduced + prescribed_base + float(path_factor) * prescribed_slope,
            dtype=float,
        ).reshape(-1)
        if rotating_mpc_map is not None:
            return rotating_mpc_map.apply_kinematics(u_lin)
        return u_lin

    def _reduce_internal_force(u_state: np.ndarray, f_int: np.ndarray) -> np.ndarray:
        if rotating_mpc_map is not None:
            return rotating_mpc_map.reduce_force(u_state, f_int)
        return np.asarray(T.T @ f_int, dtype=float).reshape(-1)

    def _reduce_equilibrium_tangent(
        u_state: np.ndarray,
        k_int: sparse.csr_matrix,
        f_int: np.ndarray,
        k_ext_red: Optional[sparse.csr_matrix],
    ) -> sparse.csr_matrix:
        if rotating_mpc_map is not None:
            f_ext_full = (
                np.zeros_like(f_int)
                if last_external_force_full is None
                else last_external_force_full
            )
            k_red = rotating_mpc_map.reduce_tangent(
                u_state,
                k_int,
                np.asarray(f_int, dtype=float).reshape(-1)
                - np.asarray(f_ext_full, dtype=float).reshape(-1),
                K_ext=None,
            )
        else:
            k_red = (T.T @ k_int @ T).tocsr()
        if k_ext_red is not None:
            k_red = (k_red - k_ext_red).tocsr()
        return k_red

    def assemble_force_system(path_factor, *args, **kwargs):
        # Private load-aware candidate only; all existing formulations retain
        # their original assembler and load semantics.
        from .general_contact.runtime import active_contact_manager

        contact_manager = active_contact_manager()
        if contact_manager is not None and hasattr(contact_manager, "set_path_factor"):
            contact_manager.set_path_factor(path_factor)
        if native_generalized_model:
            from ._ge_beam3_native_generalized_program import assemble_at
            return assemble_at(path_factor, *args, **kwargs)
        if native_distributed_model:
            from ._ge_beam3_native_distributed_program import assemble_at
            return assemble_at(path_factor, *args, **kwargs)
        if any(type(e).__module__ == 'anyfesolver._ge_beam3_native_line_static_element'
               for e in model.mesh.elements.values()):
            from ._ge_beam3_native_line_program import assemble_at
            return assemble_at(path_factor, *args, **kwargs)
        if stable_shell_rotation:
            kwargs["stable_shell_rotation"] = True
        return _assemble_nonlinear_system(*args, **kwargs)

    last_contact_equilibrium_tangent = None

    def _newton_increment(q_start, path_factor, reference, line_search_mode,
                         *, swept_origin_q=None, augmentation=0):
        """Solve one load increment with plain, decrease, or Armijo Newton."""
        from .corotational import SO3MeanDomainError
        nonlocal total_iterations, last_contact_equilibrium_tangent
        last_contact_equilibrium_tangent = None
        cancellation_checkpoint(
            f"nonlinear_static.force.step:{step_index}.start",
            context="nonlinear static force increment cancellation",
        )
        q_trial = q_start.copy()
        u = full_displacement(q_trial, path_factor)
        from .general_contact.runtime import active_contact_manager

        contact_manager = active_contact_manager()
        origin_u = (None if contact_manager is None else full_displacement(
            q_start if swept_origin_q is None else swept_origin_q, lam
        ))
        coupled_seed_pending = False
        if (contact_manager is not None
                and hasattr(contact_manager, "swept_increment_admissible")
                and not contact_manager.swept_increment_admissible(
                    origin_u, u, lam, path_factor,
                )):
            active_origin_query = getattr(
                contact_manager, "coupled_load_seed_active_origin", None
            )
            coupled_seed_pending = bool(
                active_origin_query is not None
                and active_origin_query(origin_u, lam)
            )
            if (not coupled_seed_pending
                    or not contact_manager.coupled_load_seed_pose_admissible(
                        u, path_factor,
                    )):
                return (
                    False, q_start, committed_states, float("inf"), 0,
                    "contact_gap_exceeded", None,
                )
        try:
            F_int, K_T, trial_states = assemble_force_system(
                path_factor,
                model,
                u,
                committed_states,
                num_layers,
                kinematics=kinematics,
                corotational_tangent=resolved_corotational_tangent,
                deleted_element_ids=tuple(deleted_element_ids),
                residual_stiffness_fraction=(
                    fracture_config.residual_stiffness_fraction if fracture_config is not None else 1.0
                ),
            )
            internal_guard(model, context="nonlinear static force tangent assembly")
            F_ext_red, K_ext_red, _stage_factors, _active_stage = (
                reduced_external_load_at(
                    path_factor,
                    u,
                    tangent=True,
                )
            )
            external_load_guard(context="nonlinear static force external load")
            residual = F_ext_red - _reduce_internal_force(u, F_int)
            residual_norm = float(np.linalg.norm(residual))
        finally:
            if coupled_seed_pending:
                _discard_nonlinear_state_candidate(committed_states)
                contact_manager.set_path_factor(lam)
        if coupled_seed_pending:
            contact_manager.set_path_factor(path_factor)

        for iteration in range(1, max_iterations + 1):
            cancellation_checkpoint(
                f"nonlinear_static.force.step:{step_index}.iteration:{iteration}",
                context="nonlinear static force iteration cancellation",
            )
            if status_callback is not None:
                # ``num_steps`` defines the initial load partition, not a hard
                # count: adaptive cutbacks/growth may produce far more or
                # fewer converged increments.  A fraction such as 1459/10 was
                # therefore numerically valid but semantically misleading.
                status_callback(
                    f"Increment trial {step_index + 1} | "
                    f"load factor {path_factor:.6g} / {target_load_factor:.6g} | "
                    f"increment {path_factor - lam:.3g} | "
                    f"Newton iteration {iteration} | residual {residual_norm:.3e}"
                )
                exact_guard(
                    model,
                    context="nonlinear static status callback",
                )
            total_iterations += 1
            if residual_norm <= tolerance * reference and not coupled_seed_pending:
                from .general_contact.runtime import active_contact_manager

                contact_manager = active_contact_manager()
                if (contact_manager is not None
                        and augmented_contact_manager is None
                        and not contact_manager.evaluate(
                            u, tangent=False
                        )[2].gap_tolerance_met):
                    return (
                        False,
                        q_start,
                        committed_states,
                        residual_norm,
                        iteration,
                        "contact_gap_exceeded",
                        None,
                    )
                if (contact_manager is not None
                        and hasattr(contact_manager, "swept_increment_admissible")
                        and not contact_manager.swept_increment_admissible(
                            full_displacement(
                                q_start if swept_origin_q is None else swept_origin_q,
                                lam,
                            ), u,
                            lam, path_factor,
                        )):
                    return (
                        False,
                        q_start,
                        committed_states,
                        residual_norm,
                        iteration,
                        "contact_gap_exceeded",
                        None,
                    )
                last_contact_equilibrium_tangent = K_T
                return (
                    True,
                    q_trial,
                    trial_states,
                    residual_norm,
                    iteration,
                    None,
                    F_int,
                )

            K_red = _reduce_equilibrium_tangent(u, K_T, F_int, K_ext_red)
            record_nonlinear_solver_event("linear_factorization")
            with np.errstate(all="ignore"):
                handle = factorize(
                    K_red,
                    (
                        MatrixClass.GENERAL
                        if general_tangent
                        else MatrixClass.SYMMETRIC_INDEFINITE
                    ),
                    signature=(
                        f"nonlinear.static_newton:{lam:.16g}:{iteration}"
                        if augmented_contact_manager is None else
                        f"nonlinear.static_newton:{lam:.16g}:{path_factor:.16g}:"
                        f"{augmentation}:{iteration}"
                    ),
                )
            if handle.status != "ok":
                detail = (
                    f"{handle.backend_name}:"
                    f"{handle.failure_reason or 'factorization_failed'}"
                )[:240]
                record_nonlinear_solver_event(
                    "linear_failure",
                    failure_reason=detail,
                )
                return (
                    False,
                    q_start,
                    committed_states,
                    residual_norm,
                    iteration,
                    "singular_tangent_factorization",
                    None,
                )
            record_nonlinear_solver_event("linear_solve")
            try:
                dq = np.asarray(handle.solve(residual), dtype=float).reshape(-1)
            except RuntimeError as exc:
                detail = f"{handle.backend_name}:{type(exc).__name__}:{exc}"[:240]
                record_nonlinear_solver_event(
                    "linear_failure",
                    failure_reason=detail,
                )
                return (
                    False,
                    q_start,
                    committed_states,
                    residual_norm,
                    iteration,
                    "singular_tangent_factorization",
                    None,
                )
            if np.any(~np.isfinite(dq)):
                return (
                    False,
                    q_start,
                    committed_states,
                    residual_norm,
                    iteration,
                    "nonfinite_newton_increment",
                    None,
                )

            from .general_contact.runtime import active_contact_manager

            contact_manager = active_contact_manager()
            coupled_seed_iteration = coupled_seed_pending
            if coupled_seed_pending:
                # This correction uses the same complete constrained Newton
                # system and iteration budget as ordinary equilibrium. Its path
                # starts at the committed tool pose, never the invalid preflight.
                candidate_u = full_displacement(q_trial + dq, path_factor)
                if not contact_manager.swept_increment_admissible(
                    origin_u, candidate_u, lam, path_factor,
                ):
                    return (
                        False, q_start, committed_states, residual_norm, iteration,
                        "contact_gap_exceeded", None,
                    )
                coupled_seed_pending = False
                record_nonlinear_solver_event("coupled_load_seed")
            elif contact_manager is not None:
                candidate_u = full_displacement(q_trial + dq, path_factor)
                fraction = contact_manager.admissible_step_fraction(u, candidate_u)
                if fraction <= 1.0e-12:
                    return (
                        False,
                        q_start,
                        committed_states,
                        residual_norm,
                        iteration,
                        "contact_gap_exceeded",
                        None,
                    )
                dq *= fraction

            if line_search_mode == "none":
                q_trial = q_trial + dq
                try:
                    u = full_displacement(q_trial, path_factor)
                except SO3MeanDomainError:
                    _discard_nonlinear_state_candidate(committed_states)
                    return (False, q_start, committed_states, residual_norm,
                            iteration, "trial_state_domain_refused", None)
                F_int, K_T, trial_states = assemble_force_system(
                    path_factor,
                    model,
                    u,
                    committed_states,
                    num_layers,
                    kinematics=kinematics,
                    corotational_tangent=resolved_corotational_tangent,
                    deleted_element_ids=tuple(deleted_element_ids),
                    residual_stiffness_fraction=(
                        fracture_config.residual_stiffness_fraction
                        if fracture_config is not None
                        else 1.0
                    ),
                )
                internal_guard(
                    model,
                    context="nonlinear static force tangent assembly",
                )
                F_ext_red, K_ext_red, _stage_factors, _active_stage = (
                    reduced_external_load_at(
                        path_factor,
                        u,
                        tangent=True,
                    )
                )
                external_load_guard(context="nonlinear static force external load")
                residual = F_ext_red - _reduce_internal_force(u, F_int)
                residual_norm = float(np.linalg.norm(residual))
                if not np.isfinite(residual_norm):
                    return (
                        False,
                        q_start,
                        committed_states,
                        residual_norm,
                        iteration,
                        "nonfinite_residual",
                        None,
                    )
                continue

            armijo = line_search_mode == "armijo"
            if line_search_mode not in {"decrease", "armijo"}:
                raise ValueError("unknown nonlinear line-search mode")
            merit = None
            slope = None
            if armijo:
                assert armijo_weights is not None
                weighted_residual = armijo_weights * residual
                weighted_model_step = armijo_weights * np.asarray(
                    K_red @ dq,
                    dtype=float,
                ).reshape(-1)
                merit = 0.5 * float(weighted_residual @ weighted_residual)
                slope = -float(weighted_residual @ weighted_model_step)
                if not np.isfinite(merit) or not np.isfinite(slope) or slope >= 0.0:
                    record_nonlinear_solver_event(
                        "armijo_non_descent",
                        failure_reason="armijo_non_descent_direction",
                    )
                    return (
                        False,
                        q_start,
                        committed_states,
                        residual_norm,
                        iteration,
                        "armijo_non_descent_direction",
                        None,
                    )

            def acceptable(candidate_residual: np.ndarray, candidate_norm: float, alpha: float) -> bool:
                if not np.isfinite(candidate_norm):
                    return False
                if not armijo:
                    return candidate_norm < residual_norm
                assert armijo_weights is not None and merit is not None and slope is not None
                weighted = armijo_weights * candidate_residual
                candidate_merit = 0.5 * float(weighted @ weighted)
                return bool(
                    np.isfinite(candidate_merit)
                    and candidate_merit <= merit + 1.0e-4 * alpha * slope
                )

            accepted = False
            scale = 1.0
            domain_refusals = 0
            for trial in range(settings.max_line_search_cuts):
                cancellation_checkpoint(
                    f"nonlinear_static.force.step:{step_index}.line_search:{trial + 1}",
                    context="nonlinear static line-search cancellation",
                )
                q_candidate = q_trial + scale * dq
                try:
                    u = full_displacement(q_candidate, path_factor)
                except SO3MeanDomainError:
                    domain_refusals += 1
                    _discard_nonlinear_state_candidate(committed_states)
                    record_nonlinear_solver_event(
                        "recoverable_trial_failure", failure_reason="trial_state_domain_refused"
                    )
                    record_nonlinear_solver_event("backtrack")
                    scale *= settings.line_search_reduction
                    continue
                if (coupled_seed_iteration
                        and not contact_manager.swept_increment_admissible(
                            origin_u, u, lam, path_factor,
                        )):
                    record_nonlinear_solver_event("backtrack")
                    scale *= settings.line_search_reduction
                    continue
                with_tangent = trial == 0
                try:
                    F_c, K_c, states_c = assemble_force_system(
                        path_factor,
                        model,
                        u,
                        committed_states,
                        num_layers,
                        tangent=with_tangent,
                        kinematics=kinematics,
                        corotational_tangent=resolved_corotational_tangent,
                        deleted_element_ids=tuple(deleted_element_ids),
                        residual_stiffness_fraction=(
                            fracture_config.residual_stiffness_fraction
                            if fracture_config is not None
                            else 1.0
                        ),
                    )
                except (PlaneStressConvergenceError, SO3MeanDomainError) as exc:
                    if isinstance(exc, PlaneStressConvergenceError) and not armijo:
                        raise
                    domain_refusals += int(isinstance(exc, SO3MeanDomainError))
                    _discard_nonlinear_state_candidate(committed_states)
                    record_nonlinear_solver_event(
                        "recoverable_trial_failure",
                        failure_reason=(
                            f"{type(exc).__name__}:{exc}"
                        )[:240],
                    )
                    if trial == 0:
                        record_nonlinear_solver_event("rejected_full_step")
                    record_nonlinear_solver_event("backtrack")
                    scale *= settings.line_search_reduction
                    continue
                internal_guard(model, context="nonlinear static line-search assembly")
                F_ext_c_red, K_ext_c_red, _stage_factors, _active_stage = (
                    reduced_external_load_at(
                        path_factor,
                        u,
                        tangent=with_tangent,
                    )
                )
                external_load_guard(context="nonlinear static line-search load")
                r_c = F_ext_c_red - _reduce_internal_force(u, F_c)
                rn_c = float(np.linalg.norm(r_c))
                if acceptable(r_c, rn_c, scale):
                    if not with_tangent:
                        record_nonlinear_solver_event("promotion_evaluation")
                        residual_only = np.asarray(r_c, dtype=float).copy()
                        try:
                            F_c, K_c, states_c = assemble_force_system(
                                path_factor,
                                model,
                                u,
                                committed_states,
                                num_layers,
                                tangent=True,
                                kinematics=kinematics,
                                corotational_tangent=resolved_corotational_tangent,
                                deleted_element_ids=tuple(deleted_element_ids),
                                residual_stiffness_fraction=(
                                    fracture_config.residual_stiffness_fraction
                                    if fracture_config is not None
                                    else 1.0
                                ),
                            )
                        except (PlaneStressConvergenceError, SO3MeanDomainError) as exc:
                            if isinstance(exc, PlaneStressConvergenceError) and not armijo:
                                raise
                            domain_refusals += int(isinstance(exc, SO3MeanDomainError))
                            _discard_nonlinear_state_candidate(committed_states)
                            record_nonlinear_solver_event(
                                "recoverable_trial_failure",
                                failure_reason=(
                                    f"{type(exc).__name__}:{exc}"
                                )[:240],
                            )
                            record_nonlinear_solver_event("promotion_rejected")
                            record_nonlinear_solver_event("backtrack")
                            scale *= settings.line_search_reduction
                            continue
                        internal_guard(
                            model,
                            context="nonlinear static accepted line-search assembly",
                        )
                        (
                            F_ext_c_red,
                            K_ext_c_red,
                            _stage_factors,
                            _active_stage,
                        ) = reduced_external_load_at(
                            path_factor,
                            u,
                            tangent=True,
                        )
                        external_load_guard(
                            context="nonlinear static accepted line-search load",
                        )
                        r_c = F_ext_c_red - _reduce_internal_force(u, F_c)
                        rn_c = float(np.linalg.norm(r_c))
                        if not np.allclose(
                            r_c,
                            residual_only,
                            rtol=1.0e-10,
                            atol=1.0e-12 * max(float(reference), 1.0),
                        ):
                            record_nonlinear_solver_event(
                                "promotion_residual_mismatch",
                                failure_reason="promotion_residual_mismatch",
                            )
                            _discard_nonlinear_state_candidate(committed_states)
                            record_nonlinear_solver_event("promotion_rejected")
                            record_nonlinear_solver_event("backtrack")
                            scale *= settings.line_search_reduction
                            continue
                        if not acceptable(r_c, rn_c, scale):
                            _discard_nonlinear_state_candidate(committed_states)
                            record_nonlinear_solver_event("promotion_rejected")
                            record_nonlinear_solver_event("backtrack")
                            scale *= settings.line_search_reduction
                            continue
                    q_trial = q_candidate
                    F_int, K_T, trial_states = F_c, K_c, states_c
                    K_ext_red = K_ext_c_red
                    residual, residual_norm = r_c, rn_c
                    accepted = True
                    break

                if not np.isfinite(rn_c):
                    record_nonlinear_solver_event(
                        "recoverable_trial_failure",
                        failure_reason="nonfinite_trial_residual",
                    )
                if trial == 0:
                    record_nonlinear_solver_event("rejected_full_step")
                _discard_nonlinear_state_candidate(committed_states)
                record_nonlinear_solver_event("backtrack")
                scale *= settings.line_search_reduction
            if not accepted:
                reason = ("trial_state_domain_refused"
                          if domain_refusals == settings.max_line_search_cuts
                          else "armijo_line_search_failed" if armijo else "line_search_failed")
                record_nonlinear_solver_event(
                    "line_search_failure",
                    failure_reason=reason,
                )
                return (
                    False,
                    q_start,
                    committed_states,
                    residual_norm,
                    iteration,
                    reason,
                    None,
                )

        return (
            False,
            q_start,
            committed_states,
            residual_norm,
            max_iterations,
            "maximum_iterations_reached",
            None,
        )

    def plate_energy_increment(q_start, path_factor, reference, line_search_mode, **unused):
        from ._plate_stability_continuation import segment_work, solve_increment
        nonlocal total_iterations, last_contact_equilibrium_tangent
        diagnostics = info["stability_control"]
        length = _plate_stability_binding[1]
        scale = np.ones(len(q_start))
        # With admitted simple prescribed DOFs, T is a coordinate selection.
        rotation_dofs = {node.dofs[j] for node in model.mesh.nodes.values() for j in (3,4,5)}
        for j, dof in enumerate(independent_dofs):
            if int(dof) in rotation_dofs: scale[j] = 1. / length
        def checkpoint():
            cancellation_checkpoint("nonlinear_static.plate_energy",
                context="plate energy continuation cancellation")
        def evaluate(candidate):
            checkpoint()
            u = full_displacement(candidate, path_factor)
            force, tangent, states = assemble_force_system(
                path_factor, model, u, committed_states, num_layers,
                kinematics=kinematics, corotational_tangent=resolved_corotational_tangent)
            internal_guard(model, context="plate energy trial assembly")
            residual = -_reduce_internal_force(u, force)
            if status_callback is not None:
                status_callback(f"Plate energy trial | load factor {path_factor:.6g} | residual {np.linalg.norm(residual):.3e}")
                exact_guard(model, context="plate energy status callback")
                checkpoint()
            return residual, _reduce_equilibrium_tangent(u, tangent, force, None), (force,tangent,states)
        def work(a,b):
            return segment_work(model,full_displacement(a,path_factor),
                full_displacement(b,path_factor),checkpoint,num_layers)
        def small_step_work(a,b):
            # Fixed prescribed coordinates: only free-coordinate work contributes.
            # Quadrature agreement is an error indicator, not a rigorous bound.
            delta = b-a
            estimates = []
            absolute = []
            for order in (3,5):
                points, weights = np.polynomial.legendre.leggauss(order)
                terms = []
                for point, weight in zip(points,weights):
                    checkpoint()
                    u = full_displacement(a+.5*(point+1)*delta,path_factor)
                    virgin = {}
                    try:
                        force, _, _ = assemble_force_system(
                            path_factor,model,u,virgin,num_layers,tangent=False,
                            kinematics=kinematics,corotational_tangent=resolved_corotational_tangent)
                        internal_guard(model,context="plate energy work assembly")
                        terms.append(.5*weight*float(_reduce_internal_force(u,force)@delta))
                        diagnostics["global_work_probes"] = diagnostics.get("global_work_probes",0)+1
                        checkpoint()
                    finally:
                        _discard_nonlinear_state_candidate(virgin)
                estimates.append(math.fsum(terms))
                absolute.append(math.fsum(abs(term) for term in terms))
            screen = abs(estimates[1]-estimates[0])+64*np.finfo(float).eps*sum(absolute)
            if not np.isfinite(estimates[1]+screen):
                raise ValueError("nonfinite plate energy quadrature")
            return estimates[1],screen
        def discard():
            _discard_nonlinear_state_candidate(committed_states)
        try:
            success, q_result, payload, residual_norm, iterations, reason = solve_increment(
                q_start,evaluate,work,checkpoint,discard,scale,length,tolerance,
                reference,max_iterations,diagnostics,small_step_work=small_step_work)
            total_iterations += iterations
            diagnostics["attempts"].append({"factor":float(path_factor),"success":bool(success),
                "iterations":iterations,"reason":reason})
            if success:
                force,tangent,states = payload
                last_contact_equilibrium_tangent = tangent
                return True,q_result,states,residual_norm,iterations,None,force
            discard()
            return False,q_start,committed_states,residual_norm,iterations,reason,None
        except BaseException:
            discard()
            raise

    def newton_increment(*args, **kwargs):
        if stability_control is not None:
            return plate_energy_increment(*args, **kwargs)
        # Only the opt-in contributor owns disposable load seeds. Restore its
        # physical-origin pose on every rejected increment or exception, not
        # just on an exception during the initial force evaluation.
        from .general_contact.runtime import active_contact_manager
        manager = active_contact_manager()
        if manager is None or not hasattr(manager, "coupled_load_seed_active_origin"):
            return _newton_increment(*args, **kwargs)
        converged = False
        try:
            result = _newton_increment(*args, **kwargs)
            converged = bool(result[0])
            return result
        finally:
            if not converged:
                _discard_nonlinear_state_candidate(committed_states)
                manager.set_path_factor(lam)

    force_displacement_history: List[Dict[str, Any]] = (
        copy.deepcopy(restored_static_path["force_displacement_history"])
        if restored_static_path is not None
        else []
    )
    convergence_adaptation: List[Dict[str, Any]] = (
        copy.deepcopy(restored_static_path["convergence_adaptation"])
        if restored_static_path is not None
        else []
    )
    snapshots: List[NonlinearIncrementSnapshot] = []
    force_line_search_next = (
        bool(restored_static_path["force_line_search_next"])
        if restored_static_path is not None
        else False
    )
    reaction_force_reuse_count = 0
    reaction_force_reassembly_count = 0

    assembly_threads = None if resource_config is None else resource_config.assembly_threads
    with numba_thread_scope(assembly_threads):
        while lam < target_load_factor - 1.0e-12:
            cancellation_checkpoint(
                f"nonlinear_static.force.step:{step_index + 1}",
                context="nonlinear static force step cancellation",
            )
            step_size = min(step_size, max(target_load_factor - lam, min_step))
            next_stage_boundary = target_load_factor
            for boundary in stage_boundaries:
                if boundary > lam + 1.0e-12:
                    next_stage_boundary = min(float(boundary), target_load_factor)
                    break
            # Never jump across a cumulative stage boundary.  Converging and
            # committing that endpoint is what makes the permanent-to-
            # environmental material path independent of adaptive step growth.
            lam_trial = min(lam + step_size, target_load_factor, next_stage_boundary)
            attempted_step_size = lam_trial - lam
            u_start = full_displacement(q, lam_trial)
            F_ext_red, _K_ext_red, stage_factors, active_stage = (
                reduced_external_load_at(
                    lam_trial,
                    u_start,
                    tangent=False,
                )
            )
            external_load_guard(context="nonlinear static step external load")
            reference = max(float(np.linalg.norm(F_ext_red)), 1.0)

            policy = (
                corotational_line_search_override
                or str(settings.line_search).lower()
            )
            line_search_first = policy in {"always", "armijo"} or (
                policy == "auto"
                and (
                    force_line_search_next
                    or attempted_step_size > base_step * 1.000001
                )
            )
            first_mode = (
                "armijo"
                if policy == "armijo"
                else "decrease"
                if line_search_first
                else "none"
            )
            start_normal_multipliers = (
                augmented_contact_manager.normal_multipliers.copy()
                if augmented_contact_manager is not None else None
            )
            augmentation_limit = (
                augmented_contact_manager.settings.maximum_augmentations
                if augmented_contact_manager is not None else 1
            )
            augmentation_iterations = 0
            normal_multiplier_change = 0.0
            contact_augmentation_trace: list[dict[str, object]] = []
            iterations_used = 0
            adaptation_iterations = 0
            line_search_used = bool(line_search_first)
            augmentation_start_q = q
            schur_active_signature = None
            schur_epoch_attempts = 0
            schur_backup = None
            for augmentation in range(1, augmentation_limit + 1):
                augmentation_iterations = augmentation
                (
                    converged,
                    q_new,
                    states_new,
                    residual_norm,
                    newton_iterations,
                    failure_reason,
                    accepted_force_payload,
                ) = newton_increment(
                    augmentation_start_q,
                    lam_trial,
                    reference,
                    line_search_mode=first_mode,
                    swept_origin_q=q,
                    augmentation=augmentation,
                )
                iterations_used += newton_iterations
                equilibrium_iterations = newton_iterations
                if (not converged and not line_search_first
                        and policy in {"rescue", "auto", "always"}):
                    # Rescue at the same fixed multiplier estimate.
                    (
                        converged,
                        q_new,
                        states_new,
                        residual_norm,
                        extra,
                        failure_reason,
                        accepted_force_payload,
                    ) = newton_increment(
                        augmentation_start_q,
                        lam_trial,
                        reference,
                        line_search_mode="decrease",
                        swept_origin_q=q,
                        augmentation=augmentation,
                    )
                    iterations_used += extra
                    equilibrium_iterations += extra
                    line_search_used = True
                adaptation_iterations = max(
                    adaptation_iterations, equilibrium_iterations,
                )
                if not converged or augmented_contact_manager is None:
                    if augmented_contact_manager is not None:
                        failed_row = {
                            "augmentation": augmentation,
                            "equilibrium_converged": bool(converged),
                            "failure_reason": failure_reason,
                            "equilibrium_residual_norm": float(residual_norm),
                        }
                        if schur_backup is not None:
                            fallback_q, fallback_pressure, _prior_change, _active = (
                                schur_backup
                            )
                            augmented_contact_manager.set_normal_multipliers(
                                fallback_pressure
                            )
                            _discard_nonlinear_state_candidate(committed_states)
                            augmentation_start_q = fallback_q
                            schur_backup = None
                            failed_row["schur_rejected"] = "equilibrium_failure"
                            contact_augmentation_trace.append(failed_row)
                            continue
                        contact_augmentation_trace.append(failed_row)
                    break
                candidate_u = full_displacement(q_new, lam_trial)
                certificate = augmented_contact_manager.evaluate(
                    candidate_u, tangent=False
                )[2]
                projected = augmented_contact_manager.projected_normal_multipliers(
                    candidate_u
                )
                normal_multiplier_change = float(np.max(np.abs(
                    projected - augmented_contact_manager.normal_multipliers
                )))
                contact_augmentation_trace.append({
                    "augmentation": augmentation,
                    "equilibrium_converged": True,
                    "equilibrium_residual_norm": float(residual_norm),
                    "normal_multiplier_change_pa": normal_multiplier_change,
                    "active_pressure_dofs": int(certificate.active_pressure_dofs),
                    "minimum_gap_m": float(certificate.minimum_gap),
                    "gap_tolerance_met": bool(certificate.gap_tolerance_met),
                    "contact_force_z_n": float(certificate.integration_force[2]),
                })
                row = contact_augmentation_trace[-1]
                active_signature = tuple(int(index) for index in np.flatnonzero(
                    np.asarray(certificate.numerical_pressure_pa) > 0.0,
                ))
                if schur_backup is not None:
                    fallback_q, fallback_pressure, prior_change, predicted_active = (
                        schur_backup
                    )
                    schur_backup = None
                    if (not certificate.gap_tolerance_met
                            or active_signature != predicted_active
                            or normal_multiplier_change >= 0.75*prior_change):
                        augmented_contact_manager.set_normal_multipliers(
                            fallback_pressure
                        )
                        _discard_nonlinear_state_candidate(committed_states)
                        augmentation_start_q = fallback_q
                        row["schur_rejected"] = "nonlinear_verification"
                        row["schur_prior_pressure_change_pa"] = prior_change
                        continue
                    row["schur_verified"] = True
                if (augmented_contact_manager.settings.augmentation_acceleration
                        == "supported_schur_dev"
                        and active_signature != schur_active_signature):
                    if schur_active_signature is not None:
                        row["schur_active_epoch_reset"] = {
                            "prior_active_indices": list(schur_active_signature),
                            "current_active_indices": list(active_signature),
                            "prior_epoch_attempts": schur_epoch_attempts,
                        }
                    schur_active_signature = active_signature
                    schur_epoch_attempts = 0
                if (normal_multiplier_change <=
                        augmented_contact_manager.settings.pressure_tolerance
                        and certificate.gap_tolerance_met):
                    break
                if (augmented_contact_manager.settings.augmentation_acceleration
                        == "supported_schur_dev"
                        and active_signature
                        and schur_epoch_attempts
                        < augmented_contact_manager.settings.augmentation_schur_maximum_attempts
                        and augmentation >= 2
                        and augmentation <= augmentation_limit - 2):
                    from .general_contact.linearized_pivot import (
                        refined_patch_supported_schur,
                    )

                    schur_epoch_attempts += 1
                    if last_contact_equilibrium_tangent is None:
                        raise RuntimeError("equilibrated contact tangent is missing")
                    try:
                        schur = refined_patch_supported_schur(
                            augmented_contact_manager, candidate_u,
                            last_contact_equilibrium_tangent, T,
                            np.zeros((0, T.shape[1])),
                            difference_step=(augmented_contact_manager.settings
                                             .augmentation_schur_difference_step_m),
                            maximum_pressure=(augmented_contact_manager.settings
                                              .augmentation_schur_maximum_pressure_pa),
                            maximum_pressure_step=(augmented_contact_manager.settings
                                                   .augmentation_schur_maximum_step_pa),
                            maximum_modes=(augmented_contact_manager.settings
                                           .augmentation_schur_maximum_modes),
                            diagnostics=row,
                            check_cancel=lambda: cancellation_checkpoint(
                                f"nonlinear_static.force.step:{step_index}."
                                f"augmentation:{augmentation}.schur",
                                context="nonlinear static augmentation Schur cancellation",
                            ),
                        )
                    except (ValueError, MemoryError, np.linalg.LinAlgError) as exc:
                        row["schur_ineligible"] = str(exc)
                        schur = None
                    row["schur_proposed"] = schur is not None
                    if schur is not None:
                        schur_backup = (
                            q_new.copy(), projected.copy(), normal_multiplier_change,
                            schur.active_indices,
                        )
                        row["schur_predicted_active_indices"] = list(
                            schur.active_indices
                        )
                        row["schur_predicted_residual_pa"] = (
                            schur.predicted_residual_after
                        )
                        augmented_contact_manager.set_normal_multipliers(
                            schur.multipliers
                        )
                        augmentation_start_q = q_new
                        continue
                augmented_contact_manager.set_normal_multipliers(projected)
                augmentation_start_q = q_new
            else:
                converged = False
                failure_reason = "maximum_augmentations_reached"
            if augmented_contact_manager is not None:
                info["last_contact_augmentation_trace"] = contact_augmentation_trace
                info.setdefault("contact_augmentation_attempts", []).append({
                    "step_index": step_index + 1,
                    "target_load_factor": float(lam_trial),
                    "equilibrium_converged": bool(converged),
                    "failure_reason": failure_reason,
                    "trace": contact_augmentation_trace,
                })
            if not converged and augmented_contact_manager is not None:
                augmented_contact_manager.set_normal_multipliers(
                    start_normal_multipliers
                )

            if converged:
                from .general_contact.runtime import active_contact_manager

                contact_manager = active_contact_manager()
                contact_tool_work_increment = 0.0
                if (contact_manager is not None
                        and hasattr(contact_manager, "tool_work_increment")):
                    try:
                        contact_tool_work_increment = contact_manager.tool_work_increment(
                            full_displacement(q, lam),
                            full_displacement(q_new, lam_trial),
                            lam, lam_trial,
                        )
                    except BaseException:
                        if hasattr(contact_manager, "coupled_load_seed_active_origin"):
                            _discard_nonlinear_state_candidate(committed_states)
                            contact_manager.set_path_factor(lam)
                        raise

                q = q_new
                lam = lam_trial
                committed_states = _commit_nonlinear_state_candidate(
                    committed_states,
                    states_new,
                    model=model,
                    accepted_displacements=full_displacement(q_new, lam_trial),
                )
                internal_guard(
                    model,
                    context="nonlinear static committed-state observation",
                )
                force_line_search_next = False
                step_index += 1
                u = full_displacement(q, lam)
                control_value = float(np.linalg.norm(u))
                from .nonlinear_performance_batch_c import (
                    materialize_full_internal_force,
                )

                reaction_internal = materialize_full_internal_force(
                    accepted_force_payload,
                    model.mesh.dof_manager.total_dofs,
                )
                internal_guard(
                    model,
                    context="nonlinear static accepted reaction force",
                )
                if reaction_internal is None:
                    reaction_internal, _unused, _reaction_states = (
                        assemble_force_system(
                            lam,
                            model,
                            u,
                            committed_states,
                            num_layers,
                            tangent=False,
                            deleted_element_ids=tuple(deleted_element_ids),
                            residual_stiffness_fraction=(
                                fracture_config.residual_stiffness_fraction
                                if fracture_config is not None
                                else 1.0
                            ),
                            kinematics=kinematics,
                            corotational_tangent=resolved_corotational_tangent,
                            require_full_coordinates=True,
                        )
                    )
                    internal_guard(
                        model,
                        context="nonlinear static reaction assembly",
                    )
                    # Reaction recovery is diagnostic-only; do not leave its
                    # trial constitutive state active for the next increment.
                    _discard_nonlinear_state_candidate(committed_states)
                    reaction_force_reassembly_count += 1
                    record_nonlinear_solver_event("reaction_force_reassembly")
                else:
                    reaction_force_reuse_count += 1
                    record_nonlinear_solver_event("reaction_force_reuse")
                reaction_external, _unused_tangent, _unused_factors, _unused_stage = (
                    external_load_at(lam, u, tangent=False)
                )
                external_load_guard(context="nonlinear static reaction load")
                reaction_imbalance = reaction_internal - reaction_external
                if rotating_mpc_map is not None:
                    reaction_imbalance = (
                        rotating_mpc_map.transfer_slave_imbalance_to_masters(
                            u, reaction_imbalance
                        )
                    )
                support_reactions = _support_reaction_resultants(
                    model, reaction_imbalance
                )
                exact_guard(model, context="nonlinear static support reactions")
                new_records: Tuple[DeletedElementRecord, ...] = ()
                if fracture_config is not None:
                    new_records, step_fracture_utilization = detect_new_deletions(
                        model,
                        committed_states,
                        fracture_config,
                        deleted_element_ids,
                        step_index=step_index,
                        load_factor=float(lam),
                    )
                    exact_guard(
                        model,
                        context="nonlinear static fracture observation",
                    )
                    max_fracture_utilization = max(max_fracture_utilization, step_fracture_utilization)
                    if new_records:
                        for record in new_records:
                            deleted_element = model.mesh.get_element(
                                int(record.element_id)
                            )
                            if (
                                deleted_element is None
                                or (
                                    _qualified_q4_committed_state_hooks(
                                        deleted_element
                                    )
                                    is None
                                    and _qualified_s3_committed_state_validator(
                                        deleted_element
                                    )
                                    is None
                                )
                            ):
                                continue
                            deleted_dofs = np.asarray(
                                deleted_element.get_dof_mapping(model.mesh),
                                dtype=np.intp,
                            )
                            deleted_qualified_dispositions[int(record.element_id)] = {
                                "accepted_local_u": np.asarray(
                                    u[deleted_dofs], dtype=np.float64
                                ).tolist(),
                                "deletion_step_index": int(record.step_index),
                                "deletion_load_factor": float(record.load_factor),
                                "residual_stiffness_fraction": float(
                                    fracture_config.residual_stiffness_fraction
                                ),
                                "trigger_name": str(record.trigger_name),
                            }
                        deletion_records.extend(new_records)
                        deleted_element_ids.update(record.element_id for record in new_records)
                        warning = mpc_warning_for_deleted_shells(model, (record.element_id for record in new_records))
                        if warning is not None and warning not in fracture_warnings:
                            fracture_warnings.append(warning)
                steps.append(
                    NonlinearStaticStep(
                        step_index=step_index,
                        load_factor=float(lam),
                        iterations=iterations_used,
                        residual_norm=residual_norm,
                        displacement_norm=float(np.linalg.norm(u)),
                        max_equivalent_plastic_strain=_max_plastic_strain(committed_states),
                        control_value=control_value,
                        active_stage=active_stage,
                        deleted_element_count=len(deleted_element_ids),
                        max_fracture_utilization=max_fracture_utilization,
                        support_reactions=support_reactions,
                    )
                )
                removed_load = np.zeros(3, dtype=float)
                if fracture_config is not None and deleted_element_ids:
                    if load_program is None:
                        removed_load += float(lam) * deleted_pressure_load_resultant(
                            model,
                            load_case,
                            deleted_element_ids,
                            u,
                        )
                    else:
                        for stage in load_program.stages:
                            removed_load += stage_factors[stage.name] * deleted_pressure_load_resultant(
                                model,
                                stage.load_case,
                                deleted_element_ids,
                                u,
                            )
                    if constant_load_case is not None:
                        removed_load += deleted_pressure_load_resultant(
                            model,
                            constant_load_case,
                            deleted_element_ids,
                            u,
                        )
                force_displacement_history.append(
                    {
                        "step_index": step_index,
                        "load_factor": float(lam),
                        "control_value": control_value,
                        "displacement_norm": float(np.linalg.norm(u)),
                        "residual_norm": residual_norm,
                        "iterations": iterations_used,
                        "step_size": float(attempted_step_size),
                        "line_search_used": line_search_used,
                        **({
                            "augmentation_iterations": augmentation_iterations,
                            "normal_multiplier_change": normal_multiplier_change,
                            "contact": _augmented_contact_increment_summary(certificate),
                            "normal_multipliers": (
                                augmented_contact_manager.normal_multipliers.tolist()
                            ),
                        } if augmented_contact_manager is not None else {}),
                        "stage_factors": stage_factors,
                        "active_stage": active_stage,
                        "stage_endpoint_committed": bool(
                            np.any(np.isclose(lam, stage_boundaries, rtol=0.0, atol=1.0e-12))
                        ),
                        "deleted_element_count": len(deleted_element_ids),
                        "newly_deleted_element_ids": [record.element_id for record in new_records],
                        "max_fracture_utilization": max_fracture_utilization,
                        "deleted_pressure_force_resultant": removed_load.tolist(),
                        "support_reactions": {
                            name: list(values)
                            for name, values in support_reactions.items()
                        },
                        **({"contact_tool_work_increment": contact_tool_work_increment}
                           if contact_manager is not None
                           and hasattr(contact_manager, "tool_work_increment")
                           and np.any(contact_manager.path_translation) else {}),
                    }
                )
                if record_increment_snapshots:
                    snapshots.append(
                        _increment_snapshot(
                            step_index,
                            lam,
                            u,
                            committed_states,
                            control_value=control_value,
                        )
                    )
                    internal_guard(
                        model,
                        context="nonlinear static increment snapshot",
                    )
                if progress_callback is not None:
                    max_translation = 0.0
                    size = int(u.size)
                    for node in model.mesh.nodes.values():
                        dofs = np.asarray(node.dofs[:3], dtype=np.intp)
                        if dofs.size == 0 or int(dofs.max()) >= size:
                            continue
                        value = float(np.linalg.norm(u[dofs]))
                        if value > max_translation:
                            max_translation = value
                    exact_guard(
                        model,
                        context="nonlinear static progress-state observation",
                    )
                    emit_progress(
                        progress_callback,
                        "nonlinear_static_step",
                        "nonlinear_static.force",
                        completed=float(lam),
                        total=float(target_load_factor),
                        iteration=int(iterations_used),
                        control="force",
                        step_index=int(step_index),
                        load_factor=float(lam),
                        displacement_norm=float(np.linalg.norm(u)),
                        max_translation=float(max_translation),
                        iterations=int(iterations_used),
                        max_equivalent_plastic_strain=float(_max_plastic_strain(committed_states)),
                        nominal_increment_count=int(num_steps),
                        load_increment=float(attempted_step_size),
                        support_reactions={
                            name: list(values)
                            for name, values in support_reactions.items()
                        },
                    )
                    exact_guard(
                        model,
                        context="nonlinear static progress callback",
                    )
                if fracture_config is not None and deleted_element_ids:
                    scoped_total = sum(
                        1
                        for element in model.mesh.elements.values()
                        if element_fracture_category(element) in fracture_config.element_scope
                    )
                    deleted_fraction = len(deleted_element_ids) / max(scoped_total, 1)
                    if deleted_fraction > fracture_config.max_deleted_fraction + 1.0e-12:
                        status = "stopped_at_limit"
                        info["failure_reason"] = "max_deleted_fraction_reached"
                        info["deleted_fraction"] = float(deleted_fraction)
                        break
                next_step = step_size
                action = "keep"
                if adaptation_iterations <= settings.fast_iterations and step_size < max_step:
                    next_step = min(step_size * settings.growth_factor, max_step)
                    action = "grow"
                elif adaptation_iterations >= settings.slow_iterations:
                    next_step = max(step_size * settings.cutback_factor, min_step)
                    action = "shrink_after_slow_convergence"
                    if policy == "auto":
                        force_line_search_next = True
                convergence_adaptation.append(
                    {
                        "step_index": step_index,
                        "load_factor": float(lam),
                        "iterations": int(iterations_used),
                        **({"adaptation_iterations": int(adaptation_iterations)}
                           if augmented_contact_manager is not None else {}),
                        "line_search_used": line_search_used,
                        "attempted_step_size": float(attempted_step_size),
                        "next_step_size": float(next_step),
                        "action": action,
                    }
                )
                step_size = next_step
            else:
                _discard_nonlinear_state_candidate(committed_states)
                record_nonlinear_solver_event(
                    "failed_increment",
                    failure_reason=(
                        "nonlinear_increment_failed"
                        if failure_reason is None
                        else str(failure_reason)
                    ),
                )
                record_nonlinear_solver_event(
                    "failed_increment_iteration",
                    count=max(int(iterations_used), 1),
                )
                if fracture_config is not None and deleted_element_ids and failure_reason in {
                    "singular_tangent_factorization",
                    "maximum_iterations_reached",
                    "line_search_failed",
                }:
                    failure_reason = "fracture_instability"
                    status = "stopped_at_limit" if steps else "diverged"
                    info["failure_reason"] = failure_reason
                    info["first_failed_load_factor"] = float(lam_trial)
                    info["first_failed_step_size"] = float(attempted_step_size)
                    break
                previous_step_size = step_size
                step_size *= 0.5
                force_line_search_next = True
                convergence_adaptation.append(
                    {
                        "step_index": step_index + 1,
                        "load_factor": float(lam_trial),
                        "iterations": int(iterations_used),
                        "line_search_used": line_search_used,
                        "attempted_step_size": float(attempted_step_size),
                        "next_step_size": float(step_size),
                        "action": "cutback_after_nonconvergence",
                        "previous_step_size": float(previous_step_size),
                        # The swept preflight rejects before equilibrium assembly.
                        # Infinity is its internal failure sentinel, not a measured
                        # residual, and must not enter a canonical checkpoint.
                        "residual_norm": (
                            None if failure_reason == "contact_gap_exceeded"
                            and iterations_used == 0 and np.isinf(residual_norm)
                            else float(residual_norm)
                        ),
                        **({"residual_evaluated": False,
                            "failure_reason": "contact_gap_exceeded"}
                           if failure_reason == "contact_gap_exceeded"
                           and iterations_used == 0 and np.isinf(residual_norm)
                           else {}),
                    }
                )
                if step_size < min_step:
                    contact_gap_failure = failure_reason == "contact_gap_exceeded"
                    augmentation_failure = (
                        failure_reason == "maximum_augmentations_reached"
                    )
                    status = (
                        "contact_gap_exceeded"
                        if contact_gap_failure
                        else "stopped_at_limit" if steps else "diverged"
                    )
                    info["failure_reason"] = (
                        "contact_gap_exceeded"
                        if contact_gap_failure else
                        "maximum_augmentations_reached"
                        if augmentation_failure else
                        "minimum_load_increment_reached"
                    )
                    info["first_failed_load_factor"] = float(lam_trial)
                    info["first_failed_step_size"] = float(attempted_step_size)
                    info["first_failed_iteration_reason"] = failure_reason
                    break

    u_final = full_displacement(q, lam)
    if (augmented_contact_manager is not None and not steps
            and not committed_states):
        # A rejected first increment leaves the physical state at its reference
        # configuration. Recover that result state without accepting any trial.
        _initial_force, _initial_tangent, reference_states = (
            assemble_force_system(
                lam, model, u_final, committed_states, num_layers,
                tangent=False, kinematics=kinematics,
                corotational_tangent=resolved_corotational_tangent,
                deleted_element_ids=tuple(deleted_element_ids),
            )
        )
        committed_states = _commit_nonlinear_state_candidate(
            committed_states, reference_states, model=model,
            accepted_displacements=u_final,
        )
    committed_states = _materialize_final_nonlinear_states(committed_states, info)
    committed_states = _finalize_nonlinear_element_states(
        model,
        u_final,
        committed_states,
        num_layers,
        kinematics=kinematics,
    )
    committed_states = _seal_final_qualified_q4_states(
        model,
        u_final,
        committed_states,
        num_layers,
        info,
        kinematics=kinematics,
        deleted_element_ids=tuple(sorted(deleted_element_ids)),
        deleted_dispositions=deleted_qualified_dispositions,
        deletion_records=deletion_records,
        residual_stiffness_fraction=(
            None
            if fracture_config is None
            else float(fracture_config.residual_stiffness_fraction)
        ),
    )
    exact_guard(model, context="nonlinear static final state recovery")
    if "failure_reason" not in info and status == "completed":
        info["failure_reason"] = None
    failure_reason = info.get("failure_reason")
    info["stop_reason"] = "target_load_factor_reached" if failure_reason is None else failure_reason
    info["status_category"] = _nonlinear_status_category(status, failure_reason)
    info["last_converged_load_factor"] = float(lam)
    info["peak_load_factor"] = max((step.load_factor for step in steps), default=float(lam))
    info["force_displacement_history"] = force_displacement_history
    info["convergence_adaptation"] = convergence_adaptation
    info["reaction_force_recovery"] = {
        "accepted_force_reuse_count": int(reaction_force_reuse_count),
        "full_reassembly_count": int(reaction_force_reassembly_count),
    }
    info["strain_summary"] = _nonlinear_state_summary(committed_states)
    if fracture_config is not None:
        info["fracture_summary"] = fracture_summary(
            model,
            fracture_config,
            deletion_records,
            deleted_element_ids,
            max_utilization=max_fracture_utilization,
            warnings=fracture_warnings,
        )
        exact_guard(model, context="nonlinear static fracture summary")
    if load_program is not None:
        info["load_program_stage_factors"] = load_program.stage_factors(lam)
        exact_guard(model, context="nonlinear static load-program output")
    info["total_newton_iterations"] = total_iterations
    info["solve_time"] = time.time() - start_time
    final_affine_scale = (
        float(lam)
        if force_affine_path_mode == "PROPORTIONAL_PRESCRIBED_FIELD"
        else float(initial_affine_scale)
    )
    info["constraint_postcheck"] = constraint_residual_summary(
        model,
        rotating_mpc_map.to_linear_displacement(u_final)
        if rotating_mpc_map is not None
        else u_final,
        affine_scale=final_affine_scale,
    )
    exact_guard(model, context="nonlinear static constraint postcheck")
    info["result_case"] = make_result_case(
        name="nonlinear_static",
        analysis_type="nonlinear_static",
        load_cases=tuple(stage.load_case for stage in load_program.stages) if load_program is not None else (() if load_case is None else (load_case,)),
        assembly_info={"stiffness": stiffness_info, "load": load_info},
        solver_info={"convergence_info": {"status": status}},
        recovery={"displacements": True, "element_states": True, "force_displacement_history": True},
        settings={
            "control": control_name,
            "max_load_factor": max_load_factor,
            "num_steps": num_steps,
            "num_layers": num_layers,
            "kinematics": kinematics,
            "corotational_tangent": resolved_corotational_tangent,
            "convergence_settings": settings.to_dict(),
            "fracture": None if fracture_config is None else fracture_config.to_dict(),
        },
    ).to_dict()
    restart_payload = None
    if emit_restart_checkpoint:
        restart_payload = create_nonlinear_checkpoint(
            analysis_kind="static",
            model=model,
            analysis_contract=restart_analysis_contract,
            displacements=u_final,
            element_states=committed_states,
            deleted_element_ids=tuple(sorted(deleted_element_ids)),
            path_state=_static_restart_path_payload(
                control_name="force",
                load_factor=float(lam),
                step_index=int(step_index),
                total_iterations=int(total_iterations),
                steps=steps,
                force_displacement_history=force_displacement_history,
                reduced_coordinates=q,
                terminal_status=status,
                failure_reason=failure_reason,
                base_step=float(base_step),
                minimum_step=float(min_step),
                maximum_step=float(max_step),
                next_step_size=float(step_size),
                force_line_search_next=bool(force_line_search_next),
                convergence_adaptation=convergence_adaptation,
                deletion_records=[record.to_dict() for record in deletion_records],
                fracture_warnings=list(fracture_warnings),
                max_fracture_utilization=float(max_fracture_utilization),
                affine_path_mode=force_affine_path_mode,
                exact_base_offset=np.asarray(
                    prescribed_base, dtype=np.float64
                ).reshape(-1).tolist(),
                **({"normal_multipliers":
                    augmented_contact_manager.normal_multipliers.tolist()}
                   if augmented_contact_manager is not None else {}),
            ),
        )
        exact_guard(model, context="nonlinear static checkpoint output")
    return NonlinearStaticResult(
        steps,
        status,
        u_final,
        float(lam),
        committed_states,
        info,
        tuple(snapshots),
        restart_payload,
    )


def solve_static_nonlinear(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    constant_load_case: Optional["LoadCase"] = None,
    max_load_factor: float = 1.0,
    num_steps: int = 10,
    max_iterations: int = 25,
    tolerance: float = 1.0e-6,
    num_layers: int = 5,
    min_step_fraction: float = 1.0 / 1024.0,
    imperfection: Optional[Any] = None,
    load_program: Optional[NonlinearLoadProgram] = None,
    control: str = "force",
    displacement_control: Optional[DisplacementControl] = None,
    initial_element_states: Optional[Mapping[int, Any]] = None,
    convergence_settings: Optional[
        Union[str, Mapping[str, Any], NonlinearConvergenceSettings]
    ] = None,
    resource_config: Optional[ResourceConfig] = None,
    fracture_config: Optional[FractureConfig] = None,
    kinematics: str = "von_karman",
    corotational_tangent: str = "auto",
    status_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    initial_fields: Optional[Mapping[int, Any]] = None,
    initial_displacements: Optional[np.ndarray] = None,
    equilibrate_initial_state: Optional[bool] = None,
    cancellation_token: Optional[CancellationToken] = None,
    record_increment_snapshots: bool = False,
    restart_checkpoint: Optional[Any] = None,
    emit_restart_checkpoint: bool = False,
    contact: Optional[Any] = None,
    contact_settings: Optional[Any] = None,
    contact_path_program: Optional[Any] = None,
    contact_friction: Optional[Any] = None,
    stability_control: Optional[str] = None,
) -> NonlinearStaticResult:
    """Run nonlinear static analysis under one non-renewable family lease."""

    plate_stability_binding = None
    plate_stability_authority = None
    if stability_control is not None:
        if type(stability_control) is not str or stability_control != "plate_energy":
            raise ValueError("unknown stability_control")
        if (control != "force" or kinematics != "corotational"
                or corotational_tangent != "work_conjugate"
                or emit_restart_checkpoint or equilibrate_initial_state is not None
                or any(value is not None for value in (
                    load_case, constant_load_case, imperfection, load_program,
                    displacement_control, initial_element_states, initial_fields,
                    initial_displacements, fracture_config, restart_checkpoint,
                    contact, contact_settings, contact_path_program, contact_friction,
                    convergence_settings))):
            raise ValueError("plate_energy requires virgin unloaded work_conjugate force control")
        from ._plate_stability_continuation import admit, require_authority
        plate_stability_authority = require_authority
        plate_stability_authority()
        plate_stability_binding = admit(model)

    if contact_friction is not None and contact_path_program is None:
        raise ValueError("development shell friction requires an explicit contact path program")
    if contact_path_program is None and contact_settings is not None:
        from .general_contact import ContactSolverSettings
        if (type(contact_settings) is ContactSolverSettings
                and contact_settings.finite_frame_seed != 'common_reference_frame_dev'):
            raise ValueError('finite_frame_seed requires contact_path_program')
        if (type(contact_settings) is ContactSolverSettings
                and contact_settings.shell_skin_kinematics != "reference_linear"):
            # Only the exact fixed-planar-rigid penalty profile composes the
            # ordinary corotational force/displacement owner. All existing
            # finite shell-pair profiles retain their contact-path guard.
            from .general_contact.finite_rigid_patch import admitted_definition
            if contact is None or not admitted_definition(model, contact, contact_settings):
                raise ValueError("shell_skin_kinematics requires contact_path_program")
        if (type(contact_settings) is ContactSolverSettings
                and contact_settings.shell_pair_orientation != "legacy"):
            raise ValueError("shell_pair_orientation requires contact_path_program")
        if (type(contact_settings) is ContactSolverSettings
                and contact_settings.path_predictor != "none"):
            raise ValueError("path_predictor requires contact_path_program")
    if contact_path_program is not None:
        from .general_contact import ContactSolverSettings
        from .general_contact.static_path import (
            ContactPathProgram,
            solve_static_contact_path,
        )

        if type(contact_path_program) is not ContactPathProgram:
            raise TypeError("contact_path_program must be ContactPathProgram")
        if contact is None or contact_settings is None:
            raise ValueError("contact path requires contact and contact_settings")
        finite_contact_path = (type(contact_settings) is ContactSolverSettings
                               and contact_settings.shell_skin_kinematics != "reference_linear")
        matching_kinematics = ((kinematics == "corotational" and corotational_tangent == "consistent")
                               if finite_contact_path else
                               (kinematics == "von_karman" and corotational_tangent == "auto"))
        if any(value is not None for value in (
            load_case, constant_load_case, imperfection, load_program,
            displacement_control, initial_element_states, convergence_settings,
            resource_config, fracture_config, status_callback, progress_callback,
            initial_fields, initial_displacements,
        )) or control not in {"force", "contact_path"} or (
            equilibrate_initial_state is not None
            or num_steps != 10 or min_step_fraction != 1.0 / 1024.0
            or not matching_kinematics
        ):
            raise ValueError("contact path has its own prescribed controls and admitted settings")
        return solve_static_contact_path(
            model, contact, contact_settings, contact_path_program,
            max_path_fraction=max_load_factor,
            max_iterations=max_iterations, tolerance=tolerance,
            num_layers=num_layers, restart_checkpoint=restart_checkpoint,
            emit_restart_checkpoint=emit_restart_checkpoint,
            record_increment_snapshots=record_increment_snapshots,
            cancellation_token=cancellation_token,
            friction=contact_friction,
        )

    if contact is not None:
        from .general_contact import ContactSolverSettings

        if (type(contact_settings) is ContactSolverSettings
                and contact_settings.enforcement_solver == "monolithic_active_set"):
            from .general_contact.static_path import (
                _program_from_dead_nodal_load_case,
                _program_from_simple_displacement_control,
                solve_static_contact_path,
            )

            if (control not in {"force", "displacement"}
                    or any(value is not None for value in (
                    constant_load_case, imperfection, load_program,
                    initial_element_states,
                    convergence_settings, resource_config, fracture_config,
                    status_callback, progress_callback, initial_fields,
                    initial_displacements,
            )) or equilibrate_initial_state is not None
                    or min_step_fraction != 1.0 / 1024.0
                    or kinematics != "von_karman"
                    or corotational_tangent != "auto"):
                raise ValueError(
                    "monolithic ordinary contact admits one dead nodal load "
                    "or displacement control with default structural options"
                )
            if control == "force":
                if displacement_control is not None:
                    raise ValueError("ordinary force contact rejects displacement_control")
                owned_program = _program_from_dead_nodal_load_case(load_case, num_steps)
            else:
                if load_case is not None:
                    raise ValueError("ordinary displacement contact rejects load_case")
                owned_program = _program_from_simple_displacement_control(
                    displacement_control, num_steps,
                )
            return solve_static_contact_path(
                model, contact, contact_settings, owned_program,
                max_path_fraction=max_load_factor,
                max_iterations=max_iterations, tolerance=tolerance,
                num_layers=num_layers, restart_checkpoint=restart_checkpoint,
                emit_restart_checkpoint=emit_restart_checkpoint,
                record_increment_snapshots=record_increment_snapshots,
                cancellation_token=cancellation_token,
            )

    exact_guard = _EXACT_QUALIFIED_LIFECYCLE_GUARD
    run_under_lease = _run_with_qualified_assembly_runtime_lease
    own_resource_config = _owned_resource_config_snapshot
    solve_under_lease = _solve_static_nonlinear_under_lease
    if contact is not None:
        from .general_contact import ContactDefinition, ContactSolverSettings

        if type(contact) is not ContactDefinition:
            raise TypeError("contact must be a ContactDefinition")
        if contact_settings is not None and type(contact_settings) is not ContactSolverSettings:
            raise TypeError("contact_settings must be ContactSolverSettings")
        if (contact_settings is not None
                and contact_settings.enforcement_solver == "monolithic_active_set"):
            raise ValueError(
                "monolithic_active_set currently requires contact_path_program"
            )
        if contact_settings is not None and contact_settings.shell_pair_orientation != "legacy":
            raise ValueError("shell_pair_orientation requires contact_path_program")
        if contact_settings is not None and contact_settings.path_predictor != "none":
            raise ValueError("path_predictor requires contact_path_program")
        if (contact_settings is not None
                and contact_settings.augmentation_acceleration != "none"):
            if contact_settings.augmentation_acceleration != "supported_schur_dev":
                raise ValueError(
                    "augmentation acceleration requires the contact path analysis owner"
                )
            if (len(contact.interactions) != 1
                    or contact.interactions[0].normal_law != "augmented"):
                raise ValueError(
                    "ordinary supported Schur development requires one augmented contact interaction"
                )
            from .boundary import LoadCase
            from .fe_core import Material

            if (control != "force" or kinematics != "von_karman"
                    or type(load_case) is not LoadCase
                    or not load_case.nodal_loads or load_case.element_loads
                    or load_case.pressure_loads or load_case.gravity is not None
                    or load_case.follower_pressure
                    or any(value is not None for value in (
                        constant_load_case, load_program, displacement_control,
                        fracture_config, imperfection,
                    ))
                    or any(type(material) is not Material
                           or float(getattr(material, "yield_stress", 0.0)) != 0.0
                           or getattr(material, "hardening_curve", None) is not None
                           for material in model.materials.values())):
                raise ValueError(
                    "ordinary supported Schur development requires elastic "
                    "von Karman force control with one dead nodal load case"
                )
        if (any(item.normal_law == "projected_penalty" for item in contact.interactions)
                and (str(kinematics).lower() != "von_karman"
                     or corotational_tangent != "auto")):
            raise ValueError("projected penalty contact currently requires von_karman small-rotation kinematics")
        if (any(item.normal_law == "projected_penalty" for item in contact.interactions)
                and str(control).lower() not in {"force", "displacement"}):
            raise ValueError("projected penalty contact requires force or displacement control")
        augmented_contact_requested = any(
            interaction.normal_law == "augmented"
            for interaction in contact.interactions
        )
        if augmented_contact_requested and str(control).lower() not in {
            "force", "displacement",
        }:
            raise ValueError("ordinary augmented contact requires force or displacement control")
        if (contact_settings is not None and contact_settings.warped_q4_plane
                and (str(kinematics).lower() != "von_karman"
                     or corotational_tangent != "auto")):
            raise ValueError("warped Q4 plane contact currently requires von_karman small-rotation kinematics")
        if (contact_settings is not None and contact_settings.warped_q4_plane
                and (initial_element_states is not None or initial_fields is not None)):
            raise ValueError("warped Q4 plane contact does not admit seeded material or initial-field history")
        if (contact_settings is not None
                and contact_settings.pair_motion == "general"
                and not contact.rigid_bodies
                and all(surface.beam_radius is None for surface in contact.surfaces)
                and (initial_element_states is not None or initial_fields is not None)):
            raise ValueError("general S3 pair does not admit seeded material or initial-field history")
        if any(surface.beam_radius is not None for surface in contact.surfaces):
            if str(kinematics).lower() != "von_karman":
                raise ValueError("round classical beam-plane contact currently requires von_karman kinematics")
        if (len(contact.rigid_bodies) == 1
                and contact.rigid_bodies[0].kind in {"sphere", "cylinder"}
                and (kinematics != "von_karman"
                     or corotational_tangent != "auto")):
            raise ValueError("radial rigid contact currently admits von_karman kinematics only")
        from .general_contact.triangulated_rigid import TriangulatedRigidContactBody
        finite_rigid_contact = False
        if any(type(body) is TriangulatedRigidContactBody for body in contact.rigid_bodies):
            from .general_contact.finite_rigid_patch import admitted_definition as admitted_finite_rigid
            finite_rigid_contact = admitted_finite_rigid(model, contact, contact_settings)
            if finite_rigid_contact and (kinematics != 'corotational' or corotational_tangent != 'consistent'):
                raise ValueError('finite rigid V2D penalty contact requires consistent corotational mechanics')
            if not finite_rigid_contact and (kinematics != 'von_karman' or corotational_tangent != 'auto'):
                raise ValueError('triangulated rigid contact requires von_karman affine small-rotation skins')
            if initial_element_states is not None or initial_fields is not None or fracture_config is not None:
                raise ValueError('triangulated rigid contact does not admit seeded material/initial fields or fracture')
        if any(body.path_translation is not None for body in contact.rigid_bodies):
            if str(control).lower() not in {"force", "displacement"} or load_program is not None:
                raise ValueError("prescribed rigid-tool translation requires force or displacement control without a load program")
            if str(kinematics).lower() != "von_karman":
                raise ValueError("prescribed rigid-tool translation currently requires von_karman kinematics")
            if str(control).lower() == "displacement" and initial_displacements is not None:
                raise ValueError("moving rigid-tool displacement control requires a fresh or checkpointed initial state")
        if fracture_config is not None or imperfection is not None:
            raise ValueError("contact with fracture or imperfection is not yet admitted")
        if not contact.rigid_bodies:
            if str(kinematics).lower() != "von_karman":
                raise ValueError("guided shell-pair contact requires von_karman kinematics")
        if str(kinematics).lower() == "corotational" and not finite_rigid_contact:
            if any(surface.side != "midsurface" for surface in contact.surfaces):
                raise ValueError("offset shell contact is admitted with von_karman kinematics only")
            if restart_checkpoint is not None or emit_restart_checkpoint:
                raise ValueError("corotational contact restart is not yet admitted")
            if corotational_tangent == "rotated":
                raise ValueError("contact requires the consistent corotational tangent")
            corotational_tangent = "consistent"
    elif contact_settings is not None:
        raise ValueError("contact_settings requires contact")
    exact_guard(
        model,
        context="solve_static_nonlinear preflight",
    )

    def operation(lease: Any) -> NonlinearStaticResult:
        def post_observation() -> None:
            exact_guard(
                model,
                context="solve_static_nonlinear resource configuration",
            )
            lease(
                model,
                context="solve_static_nonlinear resource configuration",
            )

        owned_resource_config = own_resource_config(
            resource_config,
            post_observation=post_observation,
        )

        def run_solver() -> NonlinearStaticResult:
            return _run_with_nonlinear_state_cleanup(
                solve_under_lease,
                model,
                load_case=load_case,
                constant_load_case=constant_load_case,
                max_load_factor=max_load_factor,
                num_steps=num_steps,
                max_iterations=max_iterations,
                tolerance=tolerance,
                num_layers=num_layers,
                min_step_fraction=min_step_fraction,
                imperfection=imperfection,
                load_program=load_program,
                control=control,
                displacement_control=displacement_control,
                initial_element_states=initial_element_states,
                convergence_settings=convergence_settings,
                resource_config=owned_resource_config,
                fracture_config=fracture_config,
                kinematics=kinematics,
                corotational_tangent=corotational_tangent,
                status_callback=status_callback,
                progress_callback=progress_callback,
                initial_fields=initial_fields,
                initial_displacements=initial_displacements,
                equilibrate_initial_state=equilibrate_initial_state,
                cancellation_token=cancellation_token,
                record_increment_snapshots=record_increment_snapshots,
                restart_checkpoint=restart_checkpoint,
                emit_restart_checkpoint=emit_restart_checkpoint,
                _qualified_runtime_guard=lease,
                stability_control=stability_control,
                _plate_stability_binding=plate_stability_binding,
                _plate_stability_authority=plate_stability_authority,
            )

        if contact is not None:
            from .general_contact.core import contact_scope
            if finite_rigid_contact and restart_checkpoint is not None:
                from .nonlinear_restart import load_nonlinear_checkpoint
                from .general_contact.finite_rigid_patch import validated_restart_scope
                payload = load_nonlinear_checkpoint(restart_checkpoint,
                    _exact_guard=exact_guard, _guard_model=model)
                scope = validated_restart_scope(model, contact, contact_settings, payload,
                    analysis_control=str(control).lower())
            else:
                scope = contact_scope(model, contact, contact_settings, initial_displacements,
                    allow_augmented=augmented_contact_requested,
                    analysis_control=str(control).lower())
            with scope as contact_manager:
                if augmented_contact_requested:
                    from .general_contact.s3_patch_static import (
                        AUGMENTED_PROFILE, AUGMENTED_MIXED_PROFILE,
                        REFINED_AUGMENTED_PROFILE, SOURCE_Q4_AUGMENTED_PROFILE,
                        SOURCE_Q4_PROJECTED_AUGMENTED_PROFILE,
                        Q4_PAIR_AUGMENTED_PROFILE,
                        Q4_PAIR_PROJECTED_AUGMENTED_PROFILE,
                        REFINED_Q4_S3_AUGMENTED_PROFILE,
                        REFINED_Q4_S3_PROJECTED_AUGMENTED_PROFILE,
                    )

                    if contact_manager.profile not in {
                        AUGMENTED_PROFILE, AUGMENTED_MIXED_PROFILE,
                        REFINED_AUGMENTED_PROFILE, SOURCE_Q4_AUGMENTED_PROFILE,
                        SOURCE_Q4_PROJECTED_AUGMENTED_PROFILE,
                        Q4_PAIR_AUGMENTED_PROFILE,
                        Q4_PAIR_PROJECTED_AUGMENTED_PROFILE,
                        REFINED_Q4_S3_AUGMENTED_PROFILE,
                        REFINED_Q4_S3_PROJECTED_AUGMENTED_PROFILE,
                    }:
                        if "RESTRICTED_SLIDING" in contact_manager.profile:
                            raise ValueError(
                                "restricted sliding augmented contact requires the contact path analysis owner"
                            )
                        raise ValueError("ordinary augmented contact requires an admitted general shell patch")
                result = run_solver()
                if hasattr(contact_manager, "set_path_factor"):
                    contact_manager.set_path_factor(
                        result.info.get("contact_path_factor", result.load_factor)
                    )
                summary = contact_manager.evaluate(result.displacements, tangent=False)[2]
                summary_payload = summary.to_dict()
                pair_settings = getattr(contact_manager, "summary_settings", None)
                if pair_settings is not None:
                    summary_payload.update(pair_settings())
                else:
                    summary_payload.update(
                        normal_law=contact.interactions[0].normal_law,
                        gap_tolerance=contact_manager.gap_tolerance,
                    )
                summary_payload.update(
                    maximum_candidate_pairs=(
                        contact_manager.settings.maximum_candidate_pairs
                    ),
                    swept_trial_cutbacks=contact_manager.swept_trial_cutbacks,
                )
                if hasattr(contact_manager, "last_gap_certificate"):
                    certificate = contact_manager.last_gap_certificate
                    summary_payload.update(
                        independent_gap_sample=certificate.minimum_sampled_gap,
                        independent_gap_lower_bound=certificate.conservative_lower_gap,
                        independent_gap_cells=certificate.cells_checked,
                        independent_gap_accuracy=certificate.accuracy,
                    )
                    if hasattr(contact_manager.law, "integration_relative_error"):
                        summary_payload["contact_integration_relative_error"] = (
                            contact_manager.law.integration_relative_error
                        )
                if hasattr(contact_manager, "last_projection_evaluations"):
                    summary_payload["projection_evaluations"] = (
                        contact_manager.last_projection_evaluations
                    )
                if not hasattr(contact_manager, "end_caps") and pair_settings is None:
                    summary_payload["normal_stiffness"] = contact_manager.stiffness
                if (hasattr(contact_manager, "path_translation")
                        and np.any(contact_manager.path_translation)):
                    increments = result.info.get("force_displacement_history", [])
                    tool_work = float(sum(
                        float(item.get("contact_tool_work_increment", 0.0))
                        for item in increments
                    ))
                    summary_payload.update(
                        rigid_tool_position=(
                            contact_manager.center.tolist()
                            if hasattr(contact_manager, "center") else
                            contact_manager.point.tolist()
                        ),
                        rigid_contact_work_trapezoid=tool_work,
                        prescribed_actuator_work_trapezoid=-tool_work,
                    )
                    if hasattr(contact_manager, "tool_position_report"):
                        summary_payload.update(contact_manager.tool_position_report())
                result.info["general_contact"] = summary_payload
                return result
        from .general_contact.runtime import suspend_contact

        with suspend_contact():
            return run_solver()

    return run_under_lease(
        model,
        context="solve_static_nonlinear",
        operation=operation,
    )
