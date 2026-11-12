"""Linear eigenvalue buckling helpers.

The first production buckling path is intentionally modest: element routines
assemble a reference geometric stiffness matrix ``KG`` and this module solves
the constrained generalized problem

    K phi = lambda KG phi

Positive eigenvalues are critical load multipliers relative to the supplied
reference compression state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .assembly import build_constraint_transformation, build_reduced_rigid_body_modes
from .cases import make_result_case
from .constraint_audit import constraint_residual_summary
from .control import CancellationToken, ProgressCallback, cancellation_safe_point, emit_progress
from .current_state_tangent import (
    _assemble_committed_current_tangent_components_implementation as _EXACT_CURRENT_STATE_COMPONENT_ASSEMBLER,
    _snapshot_committed_current_tangent_inputs as _EXACT_CURRENT_STATE_INPUT_SNAPSHOT,
    _validate_committed_current_tangent_inputs_implementation as _EXACT_CURRENT_STATE_INPUT_VALIDATOR,
    require_active_current_state_eigen_lifecycle as _EXACT_ACTIVE_CURRENT_STATE_LIFECYCLE_GUARD,
    require_committed_tangent_component_api as _EXACT_COMMITTED_TANGENT_ROUTE_GUARD,
    require_exact_qualified_component_lifecycle_api as _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
)
from .element_capabilities import require_model_element_capabilities
from .e4_pl_element import QualifiedE4PLShellElement
from .e4_pl_s3_element import QualifiedE4PLS3ShellElement
from .linalg import FactorizationCache, MatrixClass, cached_inverse_operator
from .matrix_assembly import (
    _assemble_geometric_stiffness_matrix_under_lease,
    _pack_owned_geometric_states,
    _run_with_qualified_assembly_runtime_lease,
    assemble_external_load_tangent,
    assemble_geometric_stiffness_matrix,
    assemble_stiffness_matrix,
)
from .recovery import ResourceConfig, _owned_resource_config_snapshot
from .threading_policy import resource_threaded

# Keep a private import-time identity for the genuine optional-token checkpoint.
# A replacement callback always requires a full lifecycle validation.
_EXACT_CANCELLATION_SAFE_POINT = cancellation_safe_point
_EXACT_PUBLIC_GEOMETRIC_ASSEMBLER = assemble_geometric_stiffness_matrix
_EXACT_PUBLIC_EXTERNAL_LOAD_TANGENT = assemble_external_load_tangent

if TYPE_CHECKING:
    from .analysis_session import AnalysisSession
    from .boundary import LoadCase
    from .fe_core import FEModel


REFERENCE_ELASTIC_BUCKLING_POLICY_ID = (
    "REFERENCE_ELASTIC_BUBBLE_CONDENSED_INITIAL_STRESS_V1"
)
from .modal import (
    QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID as _QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID,
    _normalize_prestress_states as _normalize_reference_prestress_states,
    _require_exact_eigen_runtime_owners,
    _require_qualified_prestress_operator_authority as _require_reference_prestress_authority,
)
CURRENT_STATE_BUCKLING_POLICY_ID = (
    "COMMITTED_MATERIAL_VS_NEGATIVE_STRESS_HESSIAN_V1"
)
QUALIFIED_Q4_S3_CURRENT_STATE_BUCKLING_POLICY_ID = (
    "Q4_ADDITIVE_VON_KARMAN_S3_NATIVE_TL_COMMITTED_PENCIL_V2"
)
_QUALIFIED_Q4_FORMULATION_ID = "E4_PL_QUALIFIED_Q4_HYBRID_V2"
_QUALIFIED_S3_FORMULATION_ID = "E4_PL_QUALIFIED_S3_COMPANION_V1"
_REFERENCE_ELASTIC_BUBBLE_POLICY_ID = (
    "REFERENCE_ELASTIC_BUBBLE_SCHUR_DERIVATIVE_V1"
)


def _static_formulation_id(element: Any) -> Optional[str]:
    """Read the class formulation declaration without descriptor execution."""

    owner = type(element)
    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if "formulation_id" in namespace:
            value = namespace["formulation_id"]
            return value if type(value) is str else None
    return None


def _has_dynamic_attribute_semantics(owner: type) -> bool:
    """Return whether normal instance lookup can execute caller-owned code."""

    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if "__getattr__" in namespace:
            return True
        attribute_reader = namespace.get("__getattribute__")
        if (
            attribute_reader is not None
            and attribute_reader is not object.__getattribute__
        ):
            return True
    return False


def _try_exact_dead_load_zero_tangent(
    model: "FEModel",
    load_case: Optional["LoadCase"],
    displacements: Optional[np.ndarray],
    *,
    exact_guard: Any,
    trusted_guard: Any,
) -> Optional[tuple[sparse.csr_matrix, Dict[str, Any]]]:
    """Return the exact zero tangent for a callback-free dead-load route.

    The public assembly path remains authoritative for follower pressure,
    qualified-S3 pressure-surface diagnostics, activity controllers, custom
    load cases and array-like displacement providers.  This narrow route
    avoids opening a second complete qualified lease when the already-open
    spectral lease proves that the tangent is identically zero.
    """

    from .boundary import LoadCase as ExactLoadCase

    exact_guard(model, context="buckling exact dead-load tangent eligibility")

    if load_case is None:
        load_name = None
    elif type(load_case) is ExactLoadCase:
        class_namespace = type.__getattribute__(ExactLoadCase, "__dict__")
        if (
            _has_dynamic_attribute_semantics(ExactLoadCase)
            or "name" in class_namespace
            or "pressure_loads" in class_namespace
            or class_namespace.get("follower_pressure") is not False
        ):
            # A class-level descriptor or changed default can alter the
            # authoritative public attribute semantics. Preserve that route.
            return None
        instance_namespace = object.__getattribute__(load_case, "__dict__")
        if (
            type(instance_namespace) is not dict
            or type(dict.get(instance_namespace, "name")) is not str
            or type(dict.get(instance_namespace, "follower_pressure")) is not bool
            or type(dict.get(instance_namespace, "pressure_loads")) is not dict
        ):
            return None
        load_name = object.__getattribute__(load_case, "name")
        follower = object.__getattribute__(load_case, "follower_pressure")
        pressure_loads = object.__getattribute__(load_case, "pressure_loads")
        if type(follower) is not bool or follower or bool(pressure_loads):
            return None
    else:
        return None

    mesh = object.__getattribute__(model, "mesh")
    if object.__getattribute__(mesh, "element_activity") is not None:
        return None
    elements = object.__getattribute__(mesh, "elements")
    from .fe_core import _QualifiedStateMapping

    if type(elements) not in {dict, _QualifiedStateMapping}:
        return None
    element_values = tuple(dict.values(elements))
    for element in element_values:
        owner = type(element)
        if _has_dynamic_attribute_semantics(owner):
            return None
        instance_namespace = object.__getattribute__(element, "__dict__")
        if type(instance_namespace) is not dict:
            return None
        if (
            "formulation_id" in instance_namespace
            or "pressure_surface_policy_id" in instance_namespace
        ):
            return None
        for base in type.__getattribute__(owner, "__mro__"):
            namespace = type.__getattribute__(base, "__dict__")
            formulation = namespace.get("formulation_id")
            if "formulation_id" in namespace and type(formulation) is not str:
                return None
            if "pressure_surface_policy_id" in namespace:
                return None
    if any(
        type(element) is QualifiedE4PLS3ShellElement
        or _static_formulation_id(element) == _QUALIFIED_S3_FORMULATION_ID
        or type.__getattribute__(type(element), "__module__")
        == "anyfesolver._ge_beam3_native_fibre_static_element"
        for element in element_values
    ):
        return None

    dof_manager = object.__getattribute__(mesh, "dof_manager")
    total_dofs = int(object.__getattribute__(dof_manager, "total_dofs"))
    if displacements is not None:
        if type(displacements) is not np.ndarray:
            return None
        if displacements.shape != (total_dofs,) or not np.all(
            np.isfinite(displacements)
        ):
            return None

    started = time.perf_counter()
    trusted_guard(model, context="buckling exact dead-load tangent preflight")
    tangent = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
    trusted_guard(model, context="buckling exact dead-load tangent output")
    return tangent, {
        "matrix_type": "external_load_tangent",
        "load_case": load_name,
        "total_dofs": total_dofs,
        "num_pressure_elements": 0,
        "pressure_element_ids": [],
        "pressure_configuration": "reference",
        "diagnostics": {
            "assembled_symmetry_error": 0.0,
            "element_activity": None,
            "dead_load_zero_tangent_fast_path": True,
            "spectral_lease_reused": True,
        },
        "assembly_time": float(time.perf_counter() - started),
    }


def _require_reference_elastic_s3_states(
    model: "FEModel",
    element_states: Any,
) -> None:
    """Validate the narrow S3 reference-elastic buckling authority boundary."""

    s3_ids = tuple(
        int(element_id)
        for element_id, element in sorted(model.mesh.elements.items())
        if _static_formulation_id(element) == _QUALIFIED_S3_FORMULATION_ID
    )
    if not s3_ids:
        return
    if not isinstance(element_states, Mapping):
        raise ValueError(
            "reference-elastic qualified S3 buckling requires an explicit "
            "element-state mapping"
        )
    for element_id in s3_ids:
        state = element_states.get(element_id)
        if not isinstance(state, Mapping):
            raise ValueError(
                "reference-elastic qualified S3 buckling requires a state "
                f"mapping for element {element_id}"
            )
        if (
            state.get("bubble_linearization_policy")
            != _REFERENCE_ELASTIC_BUBBLE_POLICY_ID
        ):
            raise ValueError(
                "reference-elastic qualified S3 buckling requires "
                "bubble_linearization_policy="
                f"{_REFERENCE_ELASTIC_BUBBLE_POLICY_ID} for element {element_id}"
            )


@dataclass
class BucklingMode:
    """One positive eigenvalue buckling mode."""

    mode_number: int
    load_factor: float
    eigenvalue: float
    mode_shape: np.ndarray
    reduced_mode_shape: np.ndarray
    modal_stiffness: float
    modal_geometric_stiffness: float
    residual_norm: float = 0.0
    rigid_body_correlation: float = 0.0
    validity_status: str = "ok"
    repeated_group: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_number": self.mode_number,
            "load_factor": self.load_factor,
            "eigenvalue": self.eigenvalue,
            "modal_stiffness": self.modal_stiffness,
            "modal_geometric_stiffness": self.modal_geometric_stiffness,
            "residual_norm": self.residual_norm,
            "rigid_body_correlation": self.rigid_body_correlation,
            "validity_status": self.validity_status,
            "repeated_group": self.repeated_group,
            "mode_shape": self.mode_shape.tolist(),
        }


@dataclass
class BucklingResult:
    """Result bundle from the linear eigenvalue buckling solve."""

    modes: List[BucklingMode]
    num_modes_requested: int
    solver_status: str
    constraint_info: Dict[str, Any]
    assembly_info: Dict[str, Any]
    result_case: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None

    @property
    def num_modes_returned(self) -> int:
        return len(self.modes)

    @property
    def quantity_metadata(self) -> Tuple[Any, ...]:
        from .quantities import describe_result_quantities

        return describe_result_quantities(self)

    @property
    def critical_load_factor(self) -> Optional[float]:
        if not self.modes:
            return None
        return self.modes[0].load_factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_status": self.solver_status,
            "num_modes_requested": self.num_modes_requested,
            "num_modes_returned": self.num_modes_returned,
            "critical_load_factor": self.critical_load_factor,
            "constraint_info": self.constraint_info,
            "assembly_info": self.assembly_info,
            "result_case": self.result_case,
            "diagnostics": self.diagnostics or {},
            "modes": [mode.to_dict() for mode in self.modes],
        }


def _as_symmetric_dense(matrix: sparse.spmatrix) -> np.ndarray:
    dense = np.asarray(matrix.toarray(), dtype=float)
    return 0.5 * (dense + dense.T)


_FLOAT64_EPS = 2.220446049250313e-16
_FLOAT64_SMALLEST_NORMAL = float(np.finfo(np.float64).tiny)
_RIGID_METRIC_VERSION = "dimensionless_full_dof_bbox_v1"


def _rank_threshold(matrix: np.ndarray) -> float:
    """Return the frozen scale-aware float64 rank threshold."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.size == 0:
        return 0.0
    singular_values = np.linalg.svd(array, compute_uv=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    if sigma_max == 0.0:
        return 0.0
    return float(64.0 * max(array.shape) * _FLOAT64_EPS * sigma_max)


def _relative_frobenius_action(matrix: np.ndarray, basis: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(matrix, ord="fro")), _FLOAT64_SMALLEST_NORMAL)
    return float(np.linalg.norm(matrix @ basis, ord="fro") / denominator)


def _relative_frobenius_cross(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(matrix, ord="fro")), _FLOAT64_SMALLEST_NORMAL)
    return float(np.linalg.norm(left.T @ matrix @ right, ord="fro") / denominator)


def _relative_symmetry_residual(matrix: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(matrix, ord="fro")), _FLOAT64_SMALLEST_NORMAL)
    return float(np.linalg.norm(matrix - matrix.T, ord="fro") / denominator)


def _rigid_projection_diagnostics(
    *,
    applied: bool,
    characteristic_length: Optional[float],
    original_dofs: int,
    projected_dofs: Optional[int],
    rigid_rank: int,
) -> Dict[str, Any]:
    """Create the stable diagnostics schema for the rigid quotient."""
    return {
        "applied": bool(applied),
        "metric_version": _RIGID_METRIC_VERSION,
        "characteristic_length": characteristic_length,
        "original_dofs": int(original_dofs),
        "projected_dofs": None if projected_dofs is None else int(projected_dofs),
        "rigid_rank": int(rigid_rank),
        "orthonormality_residual": None,
        "elastic_null_residual": None,
        "geometric_null_residual": None,
        "elastic_cross_residual": None,
        "geometric_cross_residual": None,
        "elastic_min_eigenvalue": None,
        "elastic_max_eigenvalue": None,
        "spd_tolerance": None,
        "spd_sensitivity": {},
    }


def _normalize_mode(full_mode: np.ndarray, reduced_mode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = float(np.max(np.abs(full_mode))) if full_mode.size else 0.0
    if scale <= 0.0:
        return full_mode, reduced_mode
    return full_mode / scale, reduced_mode / scale


def _mode_residual(
    K: sparse.spmatrix,
    KG: sparse.spmatrix,
    reduced_mode: np.ndarray,
    load_factor: float,
) -> float:
    residual = np.asarray(K @ reduced_mode - float(load_factor) * (KG @ reduced_mode), dtype=float).reshape(-1)
    denominator = max(
        float(np.linalg.norm(K @ reduced_mode))
        + abs(float(load_factor)) * float(np.linalg.norm(KG @ reduced_mode)),
        1.0,
    )
    return float(np.linalg.norm(residual) / denominator)


def _factor_in_range(value: float, load_factor_range: Optional[Tuple[Optional[float], Optional[float]]]) -> bool:
    if load_factor_range is None:
        return True
    lower, upper = load_factor_range
    if lower is not None and value < float(lower):
        return False
    if upper is not None and value > float(upper):
        return False
    return True


def _sort_key(value: float, shift_load_factor: Optional[float]) -> Tuple[float, float]:
    if shift_load_factor is None:
        return (float(value), 0.0)
    return (abs(float(value) - float(shift_load_factor)), float(value))


def _assign_repeated_groups(modes: List[BucklingMode], tolerance: float) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    if not modes:
        return groups
    current = [modes[0]]
    group_index = 1
    for mode in modes[1:]:
        reference = current[0].load_factor
        relative = abs(mode.load_factor - reference) / max(abs(reference), 1.0)
        if relative <= tolerance:
            current.append(mode)
        else:
            if len(current) > 1:
                for item in current:
                    item.repeated_group = group_index
                groups.append(
                    {
                        "group": group_index,
                        "mode_numbers": [int(item.mode_number) for item in current],
                        "load_factors": [float(item.load_factor) for item in current],
                        "relative_spread": float(
                            (max(item.load_factor for item in current) - min(item.load_factor for item in current))
                            / max(abs(reference), 1.0)
                        ),
                    }
                )
                group_index += 1
            current = [mode]
    if len(current) > 1:
        reference = current[0].load_factor
        for item in current:
            item.repeated_group = group_index
        groups.append(
            {
                "group": group_index,
                "mode_numbers": [int(item.mode_number) for item in current],
                "load_factors": [float(item.load_factor) for item in current],
                "relative_spread": float(
                    (max(item.load_factor for item in current) - min(item.load_factor for item in current))
                    / max(abs(reference), 1.0)
                ),
            }
        )
    return groups


@dataclass(frozen=True)
class _BucklingOperationConfig:
    num_modes: int
    eigen_tolerance: float
    dense_size_limit: int
    shift_load_factor: Optional[float]
    load_factor_range: Optional[Tuple[Optional[float], Optional[float]]]
    search_factor: int
    repeated_tolerance: float
    allow_dense_fallback: bool
    allow_free_mechanisms: bool
    follower_symmetry_tolerance: float
    reference_elastic_only: bool
    current_state_num_layers: int
    current_state_load_scale: float


def _owned_buckling_operation_config(
    model: "FEModel",
    *,
    num_modes: Any,
    eigen_tolerance: Any,
    dense_size_limit: Any,
    shift_load_factor: Any,
    load_factor_range: Any,
    search_factor: Any,
    repeated_tolerance: Any,
    allow_dense_fallback: Any,
    allow_free_mechanisms: Any,
    follower_symmetry_tolerance: Any,
    reference_elastic_only: Any,
    current_state_num_layers: Any,
    current_state_load_scale: Any,
    _exact_guard: Any,
    _trusted_guard: Any,
) -> _BucklingOperationConfig:
    """Detach all scalar/range policy before any eigensolver mechanics."""

    def converted(value: Any, converter: Any, name: str) -> Any:
        made = converter(value)
        guard = _trusted_guard if type(value) in {bool, int, float} else _exact_guard
        guard(model, context=f"buckling {name} conversion")
        return made

    def canonical_int(value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(f"{name} must be an integer")
        return converted(value, int, name)

    shift = (
        None
        if shift_load_factor is None
        else converted(shift_load_factor, float, "shift_load_factor")
    )
    if shift is not None and not math.isfinite(shift):
        raise ValueError("shift_load_factor must be finite when supplied")
    owned_range: Optional[Tuple[Optional[float], Optional[float]]]
    if load_factor_range is None:
        owned_range = None
    else:
        iterator = iter(load_factor_range)
        _exact_guard(model, context="buckling load_factor_range iterator")
        observed: list[Any] = []
        while True:
            try:
                member = next(iterator)
            except StopIteration:
                _exact_guard(model, context="buckling load_factor_range end")
                break
            _exact_guard(model, context="buckling load_factor_range observation")
            observed.append(member)
        if len(observed) != 2:
            raise ValueError("load_factor_range must contain exactly two bounds")
        bounds: list[Optional[float]] = []
        for index, member in enumerate(observed):
            if member is None:
                bounds.append(None)
                continue
            bound = converted(
                member,
                float,
                f"load_factor_range[{index}]",
            )
            if not math.isfinite(bound):
                raise ValueError("load_factor_range bounds must be finite")
            bounds.append(bound)
        owned_range = (bounds[0], bounds[1])
    owned = _BucklingOperationConfig(
        num_modes=canonical_int(num_modes, "num_modes"),
        eigen_tolerance=converted(
            eigen_tolerance,
            float,
            "eigen_tolerance",
        ),
        dense_size_limit=canonical_int(
            dense_size_limit,
            "dense_size_limit",
        ),
        shift_load_factor=shift,
        load_factor_range=owned_range,
        search_factor=canonical_int(search_factor, "search_factor"),
        repeated_tolerance=converted(
            repeated_tolerance,
            float,
            "repeated_tolerance",
        ),
        allow_dense_fallback=converted(
            allow_dense_fallback,
            bool,
            "allow_dense_fallback",
        ),
        allow_free_mechanisms=converted(
            allow_free_mechanisms,
            bool,
            "allow_free_mechanisms",
        ),
        follower_symmetry_tolerance=converted(
            follower_symmetry_tolerance,
            float,
            "follower_symmetry_tolerance",
        ),
        reference_elastic_only=converted(
            reference_elastic_only,
            bool,
            "reference_elastic_only",
        ),
        current_state_num_layers=canonical_int(
            current_state_num_layers,
            "current_state_num_layers",
        ),
        current_state_load_scale=converted(
            current_state_load_scale,
            float,
            "current_state_load_scale",
        ),
    )
    _exact_guard(model, context="buckling owned configuration")
    return owned


@resource_threaded
def _solve_eigenvalue_buckling_under_lease(
    model: "FEModel",
    element_states: Optional[Any] = None,
    num_modes: int = 3,
    eigen_tolerance: float = 1.0e-8,
    dense_size_limit: int = 200,
    shift_load_factor: Optional[float] = None,
    load_factor_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    search_factor: int = 4,
    repeated_tolerance: float = 1.0e-3,
    allow_dense_fallback: bool = False,
    allow_free_mechanisms: bool = False,
    factorization_cache: Optional[FactorizationCache] = None,
    reference_load_case: Optional["LoadCase"] = None,
    reference_displacements: Optional[np.ndarray] = None,
    follower_symmetry_tolerance: float = 1.0e-10,
    resource_config: Optional[ResourceConfig] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    session: Optional["AnalysisSession"] = None,
    reference_elastic_only: bool = False,
    current_state_displacements: Optional[Any] = None,
    current_state_element_states: Optional[Any] = None,
    current_state_num_layers: int = 5,
    current_state_load_scale: float = 1.0,
    _qualified_runtime_guard: Any = None,
    _exact_cancellation_safe_point: Any = _EXACT_CANCELLATION_SAFE_POINT,
) -> BucklingResult:
    """Solve ``K phi = lambda (KG + Kload) phi`` for positive factors.

    ``element_states`` is passed to
    :func:`anyfesolver.matrix_assembly.assemble_geometric_stiffness_matrix`.
    Beam elements accept axial reference compression and shell elements accept
    membrane resultant prestress (compression positive, or tension-positive
    ``membrane_forces``).

    When ``reference_load_case`` carries follower pressure, its exact
    current-area external-load stiffness is added to the destabilizing
    right-hand operator.  This symmetric eigensolver accepts that extension
    only when the *constrained* follower tangent is symmetric to
    ``follower_symmetry_tolerance`` (for example, a conservative closed
    pressure surface or boundary-restrained equivalent).  Open,
    nonconservative pressure patches require a general complex eigenanalysis
    and return an explicit unsupported status.
    """
    operation_started = time.perf_counter()
    phase_timings: Dict[str, float] = {}
    guard_counts = {"full": 0, "trusted": 0}
    raw_exact_guard = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD

    def exact_guard(
        observed_model: "FEModel",
        *,
        context: str,
    ) -> Dict[str, Any]:
        guard_counts["full"] += 1
        result = raw_exact_guard(observed_model, context=context)
        _qualified_runtime_guard(observed_model, context=context)
        return result

    runtime_namespace = object.__getattribute__(
        _qualified_runtime_guard,
        "__dict__",
    )
    trusted_runtime_guard = (
        dict.get(runtime_namespace, "_qualified_trusted_require")
        if type(runtime_namespace) is dict
        else None
    )
    if trusted_runtime_guard is None and type(runtime_namespace) is dict:
        trusted_runtime_guard = dict.get(
            runtime_namespace,
            "_qualified_trusted_input_require",
        )

    def trusted_guard(
        observed_model: "FEModel",
        *,
        context: str,
    ) -> Dict[str, Any]:
        if trusted_runtime_guard is None or observed_model is not model:
            return exact_guard(observed_model, context=context)
        guard_counts["trusted"] += 1
        trusted_runtime_guard(observed_model, context=context)
        return qualified_lifecycle_authority

    def cancellation_checkpoint(stage: str) -> bool:
        """Run a checkpoint and report whether it was the known no-op path."""

        checkpoint = cancellation_safe_point
        checkpoint(cancellation_token, stage)
        return (
            cancellation_token is None
            and checkpoint is _exact_cancellation_safe_point
        )

    normalize_reference_states = _normalize_reference_prestress_states
    reference_authority_guard = _require_reference_prestress_authority
    reference_authority_policy_id = (
        _QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID
    )
    snapshot_current_state = _EXACT_CURRENT_STATE_INPUT_SNAPSHOT
    current_state_route_guard = _EXACT_COMMITTED_TANGENT_ROUTE_GUARD
    current_state_activity_guard = _EXACT_ACTIVE_CURRENT_STATE_LIFECYCLE_GUARD
    current_state_input_validator = _EXACT_CURRENT_STATE_INPUT_VALIDATOR
    current_state_assembler = _EXACT_CURRENT_STATE_COMPONENT_ASSEMBLER
    qualified_lifecycle_authority = exact_guard(
        model,
        context="solve_eigenvalue_buckling preflight",
    )
    if cancellation_checkpoint("buckling.start"):
        trusted_guard(model, context="solve_eigenvalue_buckling cancellation start")
    else:
        exact_guard(model, context="solve_eigenvalue_buckling cancellation start")
    owned_config = _owned_buckling_operation_config(
        model,
        num_modes=num_modes,
        eigen_tolerance=eigen_tolerance,
        dense_size_limit=dense_size_limit,
        shift_load_factor=shift_load_factor,
        load_factor_range=load_factor_range,
        search_factor=search_factor,
        repeated_tolerance=repeated_tolerance,
        allow_dense_fallback=allow_dense_fallback,
        allow_free_mechanisms=allow_free_mechanisms,
        follower_symmetry_tolerance=follower_symmetry_tolerance,
        reference_elastic_only=reference_elastic_only,
        current_state_num_layers=current_state_num_layers,
        current_state_load_scale=current_state_load_scale,
        _exact_guard=exact_guard,
        _trusted_guard=trusted_guard,
    )
    num_modes = owned_config.num_modes
    eigen_tolerance = owned_config.eigen_tolerance
    dense_size_limit = owned_config.dense_size_limit
    shift_load_factor = owned_config.shift_load_factor
    load_factor_range = owned_config.load_factor_range
    search_factor = owned_config.search_factor
    repeated_tolerance = owned_config.repeated_tolerance
    allow_dense_fallback = owned_config.allow_dense_fallback
    allow_free_mechanisms = owned_config.allow_free_mechanisms
    follower_symmetry_tolerance = owned_config.follower_symmetry_tolerance
    reference_elastic_only = owned_config.reference_elastic_only
    current_state_num_layers = owned_config.current_state_num_layers
    current_state_load_scale = owned_config.current_state_load_scale
    _require_exact_eigen_runtime_owners(
        qualified_element_ids=qualified_lifecycle_authority[
            "qualified_element_ids"
        ],
        session=session,
        factorization_cache=factorization_cache,
        context="solve_eigenvalue_buckling",
    )
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")
    if session is not None:
        session._require_model(model)
    if session is not None:
        exact_guard(
            model,
            context="solve_eigenvalue_buckling session ownership",
        )
    if follower_symmetry_tolerance < 0.0:
        raise ValueError("follower_symmetry_tolerance must be non-negative")
    current_state = (
        current_state_displacements is not None
        or current_state_element_states is not None
    )
    qualified_reference_family_present = any(
        type(element)
        in {QualifiedE4PLShellElement, QualifiedE4PLS3ShellElement}
        or _static_formulation_id(element)
        in {_QUALIFIED_Q4_FORMULATION_ID, _QUALIFIED_S3_FORMULATION_ID}
        for element in model.mesh.elements.values()
    )
    qualified_reference_state = bool(
        not current_state
        and element_states is not None
        and qualified_reference_family_present
    )
    if (current_state_displacements is None) != (
        current_state_element_states is None
    ):
        raise ValueError(
            "current-state buckling requires both committed displacements and "
            "element states"
        )
    if current_state and element_states is not None:
        raise ValueError(
            "current-state buckling and reference element_states are mutually "
            "exclusive"
        )
    if current_state and bool(reference_elastic_only):
        raise ValueError(
            "current-state buckling and reference_elastic_only are mutually "
            "exclusive"
        )
    if current_state and (
        reference_load_case is not None or reference_displacements is not None
    ):
        raise ValueError(
            "current-state buckling and reference follower-load inputs are "
            "mutually exclusive"
        )
    if current_state and factorization_cache is not None:
        raise ValueError(
            "current-state buckling and a persistent factorization_cache are "
            "mutually exclusive"
        )
    if qualified_reference_state and factorization_cache is not None:
        raise ValueError(
            "qualified reference-prestress buckling and a persistent "
            "factorization_cache are mutually exclusive"
        )
    if isinstance(current_state_load_scale, (bool, np.bool_)):
        raise ValueError("current_state_load_scale must be finite and positive")
    load_scale = float(current_state_load_scale)
    if not math.isfinite(load_scale) or load_scale <= 0.0:
        raise ValueError("current_state_load_scale must be finite and positive")
    if not current_state and load_scale != 1.0:
        raise ValueError(
            "current_state_load_scale is available only with committed "
            "current-state inputs"
        )
    if not current_state and (
        isinstance(current_state_num_layers, (bool, np.bool_))
        or not isinstance(current_state_num_layers, (int, np.integer))
        or int(current_state_num_layers) != 5
    ):
        raise ValueError(
            "current_state_num_layers is available only with committed "
            "current-state inputs"
        )

    current_state_route: Optional[Dict[str, Any]] = None
    current_state_activity_authority: Optional[Dict[str, Any]] = None
    current_state_capability_profiles: list[tuple[str, tuple[int, ...]]] = []
    active_current_state_policy_id = CURRENT_STATE_BUCKLING_POLICY_ID
    if current_state:
        current_state_route = current_state_route_guard(
            model,
            context="solve_eigenvalue_buckling current-state path",
        )
        current_state_activity_authority = (
            current_state_activity_guard(
                model,
                current_state_route,
                context="current-state buckling analysis",
            )
        )
        if str(current_state_route["route"]) not in {
            "qualified_s3",
            "qualified_s3_v2d",
        }:
            active_current_state_policy_id = (
                QUALIFIED_Q4_S3_CURRENT_STATE_BUCKLING_POLICY_ID
            )
        qualified_q4_ids = tuple(
            int(element_id)
            for element_id, profile in sorted(
                current_state_route["element_profiles"].items()
            )
            if str(profile["family"]) == "qualified_q4"
        )
        qualified_s3_ids = tuple(
            int(element_id)
            for element_id, profile in sorted(
                current_state_route["element_profiles"].items()
            )
            if str(profile["family"]) in {"qualified_s3", "qualified_s3_v2d"}
        )
        if qualified_q4_ids:
            current_state_capability_profiles.append(
                ("current_state_buckling", qualified_q4_ids)
            )
        if qualified_s3_ids:
            current_state_capability_profiles.append(
                ("current_state_buckling_s3", qualified_s3_ids)
            )
        if str(current_state_route["route"]) == "mixed_qualified_q4_s3":
            current_state_capability_profiles.append(
                (
                    "mixed_current_state_buckling",
                    tuple(sorted((*qualified_q4_ids, *qualified_s3_ids))),
                )
            )
        required_capability = None
    elif bool(reference_elastic_only):
        required_capability = "reference_elastic_buckling"
    else:
        required_capability = "buckling"
    if current_state:
        for capability, element_ids in current_state_capability_profiles:
            require_model_element_capabilities(
                model,
                capability,
                context="solve_eigenvalue_buckling",
                element_ids=element_ids,
            )
        (
            current_state_displacements,
            current_state_element_states,
        ) = snapshot_current_state(
            model,
            current_state_displacements,
            current_state_element_states,
            _exact_guard=exact_guard,
        )
        exact_guard(
            model,
            context="solve_eigenvalue_buckling current-state snapshot",
        )
        current_state_input_validator(
            model,
            current_state_displacements,
            current_state_element_states,
            current_state_num_layers,
            context="solve_eigenvalue_buckling current-state inputs",
            _exact_guard=exact_guard,
        )
    else:
        assert required_capability is not None
        require_model_element_capabilities(
            model,
            required_capability,
            context="solve_eigenvalue_buckling",
        )

    reference_state_input_info: Optional[Dict[str, Any]] = None
    reference_operator_authority_policy_id: Optional[str] = None
    qualified_reference_operator = any(
        type(element)
        in {QualifiedE4PLShellElement, QualifiedE4PLS3ShellElement}
        or _static_formulation_id(element)
        in {_QUALIFIED_Q4_FORMULATION_ID, _QUALIFIED_S3_FORMULATION_ID}
        for element in model.mesh.elements.values()
    )
    if not current_state and element_states is not None:
        element_states, reference_state_input_info = normalize_reference_states(
            model,
            element_states,
            _exact_guard=exact_guard,
            _qualified_runtime_guard=_qualified_runtime_guard,
        )
        exact_guard(
            model,
            context="solve_eigenvalue_buckling reference-state snapshot",
        )
        reference_authority_guard(
            model,
            element_states,
            include_mass_and_descriptor=False,
            _exact_guard=trusted_guard,
        )
        reference_operator_authority_policy_id = (
            reference_authority_policy_id
        )
    elif not current_state and qualified_reference_operator:
        reference_authority_guard(
            model,
            {int(element_id): None for element_id in model.mesh.elements},
            include_mass_and_descriptor=False,
            _exact_guard=trusted_guard,
        )
        reference_operator_authority_policy_id = (
            reference_authority_policy_id
        )
    if not current_state and bool(reference_elastic_only):
        _require_reference_elastic_s3_states(model, element_states)

    packed_element_states = (
        _pack_owned_geometric_states(element_states)
        if type(element_states) is dict
        else element_states
    )

    phase_timings["validation"] = float(time.perf_counter() - operation_started)

    # Make the solve independent of whether a caller happened to run another
    # constrained analysis first.  Rigid-mode filtering reads the active DOF
    # constraints, so boundary conditions must be applied here explicitly.
    operator_assembly_started = time.perf_counter()
    model.apply_boundary_conditions()
    exact_guard(
        model,
        context="solve_eigenvalue_buckling boundary conditions",
    )
    current_state_info = None
    phase_timings["stiffness"] = 0.0
    phase_timings["geometric_stiffness"] = 0.0
    if current_state:
        component_started = time.perf_counter()
        K, internal_geometric, current_total, current_state_info = (
            current_state_assembler(
                model,
                current_state_displacements,
                current_state_element_states,
                current_state_num_layers,
                _exact_guard=exact_guard,
            )
        )
        phase_timings["stiffness"] = float(
            time.perf_counter() - component_started
        )
        exact_guard(
            model,
            context="solve_eigenvalue_buckling current-state assembly",
        )
        KG = (-load_scale * internal_geometric).tocsr()
        stiffness_info = {
            "matrix_type": "committed_current_material_tangent",
            "policy_id": active_current_state_policy_id,
            "relative_symmetry_error": current_state_info[
                "relative_symmetry_error"
            ],
            "matrix_persistence": "none",
        }
        geometric_info = {
            "matrix_type": "negative_committed_stress_hessian",
            "policy_id": active_current_state_policy_id,
            "sign_convention": "compression_positive_equals_negative_internal_geometric",
            "load_scale": load_scale,
            "relative_symmetry_error": current_state_info[
                "relative_symmetry_error"
            ],
            "matrix_persistence": "none",
        }
        total_closure = current_total - K - internal_geometric
        current_state_info = dict(current_state_info)
        current_state_info["buckling_total_closure_relative_error"] = float(
            sparse_linalg.norm(total_closure)
            / max(float(sparse_linalg.norm(current_total)), 1.0)
        )
        stiffness_plan = None
    elif session is None:
        stiffness_started = time.perf_counter()
        K, stiffness_info = assemble_stiffness_matrix(model)
        phase_timings["stiffness"] = float(
            time.perf_counter() - stiffness_started
        )
        exact_guard(model, context="solve_eigenvalue_buckling stiffness assembly")
        stiffness_plan = None
        geometric_started = time.perf_counter()
        if assemble_geometric_stiffness_matrix is _EXACT_PUBLIC_GEOMETRIC_ASSEMBLER:
            KG, geometric_info = _assemble_geometric_stiffness_matrix_under_lease(
                model,
                packed_element_states,
                qualified_runtime_guard=_qualified_runtime_guard,
            )
        else:
            KG, geometric_info = assemble_geometric_stiffness_matrix(
                model,
                element_states,
            )
        phase_timings["geometric_stiffness"] = float(
            time.perf_counter() - geometric_started
        )
        exact_guard(model, context="solve_eigenvalue_buckling geometric assembly")
    else:
        stiffness_started = time.perf_counter()
        stiffness_plan = session.stiffness_plan(model)
        phase_timings["stiffness"] = float(
            time.perf_counter() - stiffness_started
        )
        exact_guard(model, context="solve_eigenvalue_buckling session stiffness plan")
        K = stiffness_plan.matrix
        stiffness_info = dict(stiffness_plan.info)
        geometric_started = time.perf_counter()
        if assemble_geometric_stiffness_matrix is _EXACT_PUBLIC_GEOMETRIC_ASSEMBLER:
            KG, geometric_info = _assemble_geometric_stiffness_matrix_under_lease(
                model,
                packed_element_states,
                qualified_runtime_guard=_qualified_runtime_guard,
            )
        else:
            KG, geometric_info = assemble_geometric_stiffness_matrix(
                model,
                element_states,
            )
        phase_timings["geometric_stiffness"] = float(
            time.perf_counter() - geometric_started
        )
        exact_guard(model, context="solve_eigenvalue_buckling geometric assembly")
    load_tangent_started = time.perf_counter()
    dead_load_tangent = None
    if assemble_external_load_tangent is _EXACT_PUBLIC_EXTERNAL_LOAD_TANGENT:
        dead_load_tangent = _try_exact_dead_load_zero_tangent(
            model,
            reference_load_case,
            reference_displacements,
            exact_guard=exact_guard,
            trusted_guard=trusted_guard,
        )
    if dead_load_tangent is None:
        K_load, load_tangent_info = assemble_external_load_tangent(
            model,
            reference_load_case,
            reference_displacements,
        )
    else:
        K_load, load_tangent_info = dead_load_tangent
    exact_guard(model, context="solve_eigenvalue_buckling external-load tangent")
    phase_timings["external_load_tangent"] = float(
        time.perf_counter() - load_tangent_started
    )
    phase_timings["operator_assembly_total"] = float(
        time.perf_counter() - operator_assembly_started
    )
    if cancellation_checkpoint("buckling.after_assembly"):
        trusted_guard(
            model,
            context="solve_eigenvalue_buckling cancellation after assembly",
        )
    else:
        exact_guard(
            model,
            context="solve_eigenvalue_buckling cancellation after assembly",
        )
    zero_load = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)

    reduction_started = time.perf_counter()
    if session is None or current_state:
        K_red, _, T, _, independent_dofs, constraint_info = (
            build_constraint_transformation(K, zero_load, model)
        )
        exact_guard(
            model,
            context="solve_eigenvalue_buckling constraint transformation",
        )
        constraint_plan = None
    else:
        constraint_plan = session.constraint_plan(stiffness_plan, model)
        exact_guard(model, context="solve_eigenvalue_buckling session constraint plan")
        K_red = constraint_plan.K_red
        T = constraint_plan.T
        independent_dofs = constraint_plan.independent_dofs
        constraint_info = dict(constraint_plan.info)
    KG_red = (T.T @ KG @ T).tocsr()
    K_load_red = (T.T @ K_load @ T).tocsr()
    follower_symmetry_error = float(
        sparse.linalg.norm(K_load_red - K_load_red.T)
        / max(float(sparse.linalg.norm(K_load_red)), 1.0)
    )
    KG_red = (KG_red + K_load_red).tocsr()
    phase_timings["constraint_reduction"] = float(
        time.perf_counter() - reduction_started
    )

    assembly_info = {
        "stiffness": stiffness_info,
        "geometric_stiffness": geometric_info,
        "external_load_tangent": load_tangent_info,
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "reduced_dofs": int(K_red.shape[0]),
    }
    if reference_state_input_info is not None:
        assembly_info["reference_prestress_input"] = reference_state_input_info
        if session is not None and qualified_reference_state:
            assembly_info["analysis_session_state_dependent_bypass_reason"] = (
                "qualified_reference_prestress_matrices_and_factors_are_not_cacheable"
            )
            assembly_info["analysis_session_elastic_inverse_policy"] = (
                "immutable_elastic_stiffness_factor_may_persist;"
                "prestress_and_geometric_operators_never_persist"
            )
    if reference_operator_authority_policy_id is not None:
        assembly_info["reference_operator_authority_policy_id"] = (
            reference_operator_authority_policy_id
        )
        if reference_state_input_info is not None:
            assembly_info["prestress_operator_authority_policy_id"] = (
                reference_operator_authority_policy_id
            )
    if current_state_info is not None:
        assembly_info["current_state_tangent_components"] = current_state_info
        assembly_info["current_state_buckling_policy_id"] = (
            active_current_state_policy_id
        )
        assembly_info["current_state_activity_authority"] = (
            current_state_activity_authority
        )
        if current_state_route is not None:
            assembly_info["current_state_route"] = {
                "route": str(current_state_route["route"]),
                "route_policy_id": str(current_state_route["route_policy_id"]),
                "families": list(current_state_route["families"]),
                "formulation_counts": dict(
                    current_state_route["formulation_counts"]
                ),
                "native_rotation_required": bool(
                    current_state_route["native_rotation_required"]
                ),
                "kinematic_scope": dict(
                    sorted(current_state_route["kinematic_scope"].items())
                ),
                "reference_surface_offset_scope": dict(
                    sorted(
                        current_state_route[
                            "reference_surface_offset_scope"
                        ].items()
                    )
                ),
            }
        if session is not None:
            assembly_info["analysis_session_bypass_reason"] = (
                "committed_current_state_matrices_and_factors_are_not_cacheable"
            )
    if session is not None:
        assembly_info["analysis_session"] = session.diagnostics()
        exact_guard(model, context="solve_eigenvalue_buckling session diagnostics")

    settings = {
        "num_modes": num_modes,
        "dense_size_limit": dense_size_limit,
        "eigen_tolerance": eigen_tolerance,
        "shift_load_factor": shift_load_factor,
        "load_factor_range": None if load_factor_range is None else list(load_factor_range),
        "search_factor": search_factor,
        "repeated_tolerance": repeated_tolerance,
        "allow_dense_fallback": allow_dense_fallback,
        "allow_free_mechanisms": allow_free_mechanisms,
        "factorization_cache": (
            (
                None
                if current_state or qualified_reference_state
                else (session.factorization_cache.name if session is not None else None)
            )
            if factorization_cache is None
            else factorization_cache.name
        ),
        "reference_load_case": None if reference_load_case is None else reference_load_case.name,
        "follower_symmetry_tolerance": float(follower_symmetry_tolerance),
        "resource_config": None if resource_config is None else resource_config.to_dict(),
    }
    if bool(reference_elastic_only):
        settings["reference_elastic_buckling_policy_id"] = (
            REFERENCE_ELASTIC_BUCKLING_POLICY_ID
        )
        assembly_info["reference_elastic_buckling_policy_id"] = (
            REFERENCE_ELASTIC_BUCKLING_POLICY_ID
        )
    if qualified_reference_state and reference_state_input_info is not None:
        settings["reference_prestress_input_schema_id"] = (
            reference_state_input_info["schema_id"]
        )
        settings["reference_prestress_matrix_persistence"] = "none"
        settings["reference_prestress_factorization_persistence"] = "none"
    if current_state:
        settings.update(
            {
                "current_state_buckling_policy_id": (
                    active_current_state_policy_id
                ),
                "current_state_num_layers": int(current_state_num_layers),
                "current_state_load_scale": load_scale,
                "current_state_matrix_persistence": "none",
                "current_state_factorization_persistence": "none",
                "current_state_route": (
                    None
                    if current_state_route is None
                    else str(current_state_route["route"])
                ),
            }
        )
    prestress_state_source = (
        type(current_state_element_states).__name__
        if current_state
        else (
            "none"
            if element_states is None
            else (
                str(reference_state_input_info["source_kind"])
                if reference_state_input_info is not None
                else type(element_states).__name__
            )
        )
    )

    if follower_symmetry_error > float(follower_symmetry_tolerance):
        diagnostics = {
            "status": "unsupported_nonsymmetric_follower_pencil",
            "reason": (
                "The constrained follower-pressure load tangent is nonsymmetric; "
                "general complex nonconservative eigenanalysis is not implemented."
            ),
            "follower_tangent_symmetry_error": follower_symmetry_error,
            "follower_symmetry_tolerance": float(follower_symmetry_tolerance),
        }
        result_case = make_result_case(
            name="linear_buckling",
            analysis_type="linear_buckling",
            load_cases=() if reference_load_case is None else (reference_load_case,),
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
            metadata={"prestress_state_source": prestress_state_source},
        ).to_dict()
        exact_guard(model, context="solve_eigenvalue_buckling output")
        return BucklingResult(
            [],
            num_modes,
            "unsupported_nonsymmetric_follower_pencil",
            constraint_info,
            assembly_info,
            result_case,
            diagnostics,
        )

    if K_red.shape[0] == 0:
        diagnostics = {"status": "empty_reduced_system"}
        result_case = make_result_case(
            name="linear_buckling",
            analysis_type="linear_buckling",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
            metadata={"prestress_state_source": prestress_state_source},
        ).to_dict()
        exact_guard(model, context="solve_eigenvalue_buckling output")
        return BucklingResult([], num_modes, "empty_reduced_system", constraint_info, assembly_info, result_case, diagnostics)
    if KG_red.nnz == 0:
        diagnostics = {"status": "zero_geometric_stiffness"}
        result_case = make_result_case(
            name="linear_buckling",
            analysis_type="linear_buckling",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
            metadata={"prestress_state_source": prestress_state_source},
        ).to_dict()
        exact_guard(model, context="solve_eigenvalue_buckling output")
        return BucklingResult([], num_modes, "zero_geometric_stiffness", constraint_info, assembly_info, result_case, diagnostics)

    rigid_started = time.perf_counter()
    if session is None or current_state:
        Q_rigid, nullspace_info = build_reduced_rigid_body_modes(
            model,
            independent_dofs,
            int(K.shape[0]),
            transformation=T,
        )
        exact_guard(model, context="solve_eigenvalue_buckling rigid-body basis")
    else:
        Q_rigid, nullspace_info = session.rigid_body_modes(constraint_plan, model)
        exact_guard(
            model,
            context="solve_eigenvalue_buckling session rigid-body basis",
        )
    phase_timings["rigid_mode_preparation"] = float(
        time.perf_counter() - rigid_started
    )
    assembly_info["nullspace"] = nullspace_info

    # Work with the inverted symmetric pencil KG phi = mu K phi.  Constrained
    # systems without an analytic rigid space retain the existing sparse/dense
    # selection.  Free systems are first reduced to the dimensionless rigid
    # quotient so a known singular elastic pencil is never sent to an
    # eigensolver.
    n_red = int(K_red.shape[0])
    K_sym = (0.5 * (K_red + K_red.T)).tocsr()
    KG_sym = (0.5 * (KG_red + KG_red.T)).tocsr()
    k = min(max(num_modes * max(int(search_factor), 1), num_modes + 2), n_red - 1)

    eigenvectors = None
    projection_y_vectors = None
    projected_rigid_basis = None
    projection_diagnostics = None
    solver_kind = "not_started"
    sparse_error = None
    shift_invert_diagnostics: Dict[str, Any] = {"shift_invert": False}
    elastic_inverse_diagnostics: Dict[str, Any] = {"elastic_inverse": False}
    phase_timings["factorization"] = 0.0
    eigen_started = time.perf_counter()

    def _projection_failure(status: str, reason: str) -> BucklingResult:
        diagnostics = {
            "status": status,
            "reason": reason,
            "solver": "dense_scipy_eigh_rigid_quotient",
            "follower_load_stiffness_included": bool(K_load_red.nnz),
            "follower_tangent_symmetry_error": follower_symmetry_error,
            "nullspace_rank": int(Q_rigid.shape[1]),
            "nullspace_info": nullspace_info,
            "free_mechanism_handling": (
                "retained" if allow_free_mechanisms else "rigid_body_roots_filtered"
            ),
            "rigid_body_handling": "projected",
            "rigid_projection": projection_diagnostics,
        }
        result_case = make_result_case(
            name="linear_buckling",
            analysis_type="linear_buckling",
            load_cases=() if reference_load_case is None else (reference_load_case,),
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes, "num_modes_returned": 0},
            settings=settings,
            metadata={"prestress_state_source": prestress_state_source},
        ).to_dict()
        exact_guard(model, context="solve_eigenvalue_buckling output")
        return BucklingResult(
            [],
            num_modes,
            status,
            constraint_info,
            assembly_info,
            result_case,
            diagnostics,
        )

    if Q_rigid.shape[1]:
        coordinates = np.asarray(model.mesh.get_node_coordinates(), dtype=np.float64)
        characteristic_length: Optional[float] = None
        if coordinates.ndim == 2 and coordinates.shape[0] > 0 and coordinates.shape[1] == 3:
            coordinate_extent = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
            characteristic_length = float(np.linalg.norm(coordinate_extent))
        projection_diagnostics = _rigid_projection_diagnostics(
            applied=False,
            characteristic_length=characteristic_length,
            original_dofs=n_red,
            projected_dofs=None,
            rigid_rank=int(Q_rigid.shape[1]),
        )
        if (
            not np.all(np.isfinite(coordinates))
            or characteristic_length is None
            or not np.isfinite(characteristic_length)
            or characteristic_length <= 0.0
        ):
            return _projection_failure(
                "invalid_rigid_quotient",
                "The dimensionless rigid metric requires a finite, positive bounding-box diagonal.",
            )
        if n_red > dense_size_limit and not allow_dense_fallback:
            return _projection_failure(
                "rigid_projection_requires_dense",
                "The analytic rigid space requires the dense quotient, but dense fallback is disabled for this reduced size.",
            )

        transformation = np.asarray(T.toarray(), dtype=np.float64)
        full_scale = np.ones(transformation.shape[0], dtype=np.float64)
        for node in model.mesh.nodes.values():
            node_dofs = np.asarray(node.dofs[:3], dtype=np.intp)
            full_scale[node_dofs] = 1.0 / characteristic_length
        scaled_transformation = full_scale[:, None] * transformation
        if not (
            np.all(np.isfinite(transformation))
            and np.all(np.isfinite(scaled_transformation))
            and np.all(np.isfinite(Q_rigid))
        ):
            return _projection_failure(
                "invalid_rigid_quotient",
                "The constraint transformation or rigid basis contains non-finite values.",
            )

        _Q_transformation, R_transformation = linalg.qr(
            scaled_transformation,
            mode="economic",
            check_finite=True,
        )
        transformation_singular_values = np.linalg.svd(R_transformation, compute_uv=False)
        transformation_threshold = _rank_threshold(R_transformation)
        if (
            R_transformation.shape != (n_red, n_red)
            or transformation_singular_values.size != n_red
            or float(transformation_singular_values[-1]) <= transformation_threshold
        ):
            return _projection_failure(
                "invalid_rigid_quotient",
                "The dimensionless constraint transformation is rank deficient.",
            )

        identity = np.eye(n_red, dtype=np.float64)
        R_inverse = linalg.solve_triangular(
            R_transformation,
            identity,
            lower=False,
            check_finite=True,
        )
        K_dense = _as_symmetric_dense(K_red)
        KG_dense = _as_symmetric_dense(KG_red)
        K_y_raw = R_inverse.T @ K_dense @ R_inverse
        KG_y_raw = R_inverse.T @ KG_dense @ R_inverse
        dimension_tolerance = float(4096.0 * n_red * _FLOAT64_EPS)
        if (
            not np.all(np.isfinite(K_y_raw))
            or not np.all(np.isfinite(KG_y_raw))
            or _relative_symmetry_residual(K_y_raw) > dimension_tolerance
            or _relative_symmetry_residual(KG_y_raw) > dimension_tolerance
        ):
            return _projection_failure(
                "invalid_rigid_quotient",
                "A transformed operator is non-finite or violates the frozen symmetry residual.",
            )
        K_y = 0.5 * (K_y_raw + K_y_raw.T)
        KG_y = 0.5 * (KG_y_raw + KG_y_raw.T)

        rigid_y = R_transformation @ np.asarray(Q_rigid, dtype=np.float64)
        rigid_singular_values = np.linalg.svd(rigid_y, compute_uv=False)
        rigid_threshold = _rank_threshold(rigid_y)
        rigid_rank = int(np.count_nonzero(rigid_singular_values > rigid_threshold))
        if rigid_rank != int(Q_rigid.shape[1]):
            return _projection_failure(
                "invalid_rigid_quotient",
                "The transformed analytic rigid basis is rank deficient.",
            )
        Q_complete, _R_rigid = linalg.qr(rigid_y, mode="full", check_finite=True)
        projected_rigid_basis = np.asarray(Q_complete[:, :rigid_rank], dtype=np.float64)
        flexible_basis = np.asarray(Q_complete[:, rigid_rank:], dtype=np.float64)
        projection_diagnostics["applied"] = True
        projection_diagnostics["projected_dofs"] = int(flexible_basis.shape[1])
        if flexible_basis.shape[1] == 0:
            return _projection_failure(
                "empty_flexible_subspace",
                "The analytic rigid space spans the complete reduced system.",
            )

        orthonormality_residual = float(
            np.linalg.norm(Q_complete.T @ Q_complete - identity, ord="fro")
        )
        elastic_null_residual = _relative_frobenius_action(K_y, projected_rigid_basis)
        geometric_null_residual = _relative_frobenius_action(KG_y, projected_rigid_basis)
        elastic_cross_residual = _relative_frobenius_cross(
            K_y,
            projected_rigid_basis,
            flexible_basis,
        )
        geometric_cross_residual = _relative_frobenius_cross(
            KG_y,
            projected_rigid_basis,
            flexible_basis,
        )
        projection_diagnostics.update(
            {
                "orthonormality_residual": orthonormality_residual,
                "elastic_null_residual": elastic_null_residual,
                "geometric_null_residual": geometric_null_residual,
                "elastic_cross_residual": elastic_cross_residual,
                "geometric_cross_residual": geometric_cross_residual,
            }
        )
        if orthonormality_residual > dimension_tolerance:
            return _projection_failure(
                "invalid_rigid_quotient",
                "The complete dimensionless quotient basis is not orthonormal under the frozen residual.",
            )
        if (
            elastic_null_residual > dimension_tolerance
            or geometric_null_residual > dimension_tolerance
            or elastic_cross_residual > dimension_tolerance
            or geometric_cross_residual > dimension_tolerance
        ):
            return _projection_failure(
                "invalid_rigid_quotient",
                "The elastic or combined geometric operator does not descend to the analytic rigid quotient.",
            )

        K_f_raw = flexible_basis.T @ K_y @ flexible_basis
        KG_f_raw = flexible_basis.T @ KG_y @ flexible_basis
        if (
            _relative_symmetry_residual(K_f_raw) > dimension_tolerance
            or _relative_symmetry_residual(KG_f_raw) > dimension_tolerance
        ):
            return _projection_failure(
                "invalid_rigid_quotient",
                "A quotient operator violates the frozen symmetry residual.",
            )
        K_f = 0.5 * (K_f_raw + K_f_raw.T)
        KG_f = 0.5 * (KG_f_raw + KG_f_raw.T)
        elastic_eigenvalues = np.linalg.eigvalsh(K_f)
        elastic_minimum = float(elastic_eigenvalues[0])
        elastic_maximum = float(elastic_eigenvalues[-1])
        spd_tolerance = _rank_threshold(K_f)
        spd_sensitivity = {
            key: {
                "threshold": float(multiplier * spd_tolerance),
                "positive_definite": bool(elastic_minimum > multiplier * spd_tolerance),
            }
            for key, multiplier in (("0.25", 0.25), ("1", 1.0), ("4", 4.0))
        }
        projection_diagnostics.update(
            {
                "elastic_min_eigenvalue": elastic_minimum,
                "elastic_max_eigenvalue": elastic_maximum,
                "spd_tolerance": float(spd_tolerance),
                "spd_sensitivity": spd_sensitivity,
            }
        )
        positive_definite_values = {
            bool(item["positive_definite"])
            for item in spd_sensitivity.values()
        }
        if positive_definite_values != {True}:
            return _projection_failure(
                "singular_projected_stiffness",
                "The quotient elastic operator is not stably positive definite at all frozen sensitivity multipliers.",
            )

        try:
            _inverted_eigenvalues, flexible_eigenvectors = linalg.eigh(
                KG_f,
                K_f,
                check_finite=True,
            )
        except linalg.LinAlgError:
            return _projection_failure(
                "singular_projected_stiffness",
                "The symmetric quotient pencil rejected the verified elastic operator.",
            )
        projection_y_vectors = flexible_basis @ flexible_eigenvectors
        eigenvectors = R_inverse @ projection_y_vectors
        solver_kind = "dense_scipy_eigh_rigid_quotient"

    if not Q_rigid.shape[1] and n_red > dense_size_limit and 1 <= k < n_red:
        try:
            if shift_load_factor is None:
                # ARPACK otherwise constructs and discards its own inverse of
                # the elastic metric.  Make that inverse explicit so an
                # immutable AnalysisSession can retain the *elastic* factor
                # across changed prestress states.  Current-state tangents stay
                # solve-local, and no geometric/prestress factor is persisted.
                elastic_cache = (
                    (
                        None
                        if current_state
                        else (
                            factorization_cache
                            or (
                                session.factorization_cache
                                if session is not None
                                else None
                            )
                        )
                    )
                    or FactorizationCache(
                        name=(
                            "buckling_current_state_elastic_inverse_transient"
                            if current_state
                            else "buckling_elastic_inverse"
                        ),
                        max_entries=1,
                    )
                )
                factor_started = time.perf_counter()
                elastic_inverse, elastic_handle = cached_inverse_operator(
                    K_sym.tocsc(),
                    MatrixClass.SPD,
                    cache=elastic_cache,
                )
                phase_timings["factorization"] = float(
                    time.perf_counter() - factor_started
                )
                exact_guard(
                    model,
                    context="solve_eigenvalue_buckling elastic factorization",
                )
                _, eigenvectors = sparse_linalg.eigsh(
                    KG_sym.tocsc(),
                    k=k,
                    M=K_sym.tocsc(),
                    Minv=elastic_inverse,
                    which="LA",
                )
                elastic_inverse_diagnostics = {
                    "elastic_inverse": True,
                    "elastic_inverse_factorization": elastic_handle.diagnostics(),
                    "elastic_inverse_cache": elastic_cache.diagnostics(),
                    "elastic_inverse_persistence": (
                        "solve_local"
                        if current_state
                        else (
                            "caller_owned_elastic_only"
                            if factorization_cache is not None
                            else (
                                "analysis_session_elastic_only"
                                if session is not None
                                else "solve_local"
                            )
                        )
                    ),
                }
            else:
                sigma = 1.0 / float(shift_load_factor)
                shift_matrix = (KG_sym - sigma * K_sym).tocsc()
                cache = (
                    factorization_cache
                    or (
                        session.factorization_cache
                        if session is not None
                        and not current_state
                        and not qualified_reference_state
                        else None
                    )
                    or FactorizationCache(
                        name=(
                            "buckling_current_state_transient"
                            if current_state
                            else (
                                "buckling_qualified_prestress_transient"
                                if qualified_reference_state
                                else "buckling_shift_invert"
                            )
                        ),
                        max_entries=2,
                    )
                )
                factor_started = time.perf_counter()
                operator, handle = cached_inverse_operator(
                    shift_matrix,
                    MatrixClass.SYMMETRIC_INDEFINITE,
                    cache=cache,
                )
                phase_timings["factorization"] = float(
                    time.perf_counter() - factor_started
                )
                exact_guard(
                    model,
                    context="solve_eigenvalue_buckling shift factorization",
                )
                _, eigenvectors = sparse_linalg.eigsh(
                    KG_sym.tocsc(),
                    k=k,
                    M=K_sym.tocsc(),
                    sigma=sigma,
                    which="LM",
                    OPinv=operator,
                )
                shift_invert_diagnostics = {
                    "shift_invert": True,
                    "shift_inverse_sigma": float(sigma),
                    "shift_factorization": handle.diagnostics(),
                    "factorization_cache": cache.diagnostics(),
                }
            solver_kind = "sparse_scipy_eigsh_inverted_pencil"
        except Exception as exc:
            sparse_error = str(exc)
            eigenvectors = None
    if not Q_rigid.shape[1] and eigenvectors is None and (n_red <= dense_size_limit or allow_dense_fallback):
        K_dense = _as_symmetric_dense(K_red)
        KG_dense = _as_symmetric_dense(KG_red)
        try:
            _, eigenvectors = linalg.eigh(KG_dense, K_dense)
            solver_kind = "dense_scipy_eigh_inverted_pencil"
        except linalg.LinAlgError:
            # Singular K (e.g. unconstrained model): fall back to the general
            # nonsymmetric pencil and let the Rayleigh filtering sort it out.
            _, eigenvectors = linalg.eig(K_dense, KG_dense)
            solver_kind = "dense_scipy_eig_general_pencil"
    if eigenvectors is None:
        diagnostics = {
            "status": "failed",
            "solver": solver_kind,
            "sparse_error": sparse_error,
            "reason": "sparse eigensolve failed and dense fallback is disabled for this reduced size",
        }
        result_case = make_result_case(
            name="linear_buckling",
            analysis_type="linear_buckling",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
            metadata={"prestress_state_source": prestress_state_source},
        ).to_dict()
        exact_guard(model, context="solve_eigenvalue_buckling output")
        return BucklingResult([], num_modes, "failed", constraint_info, assembly_info, result_case, diagnostics)

    phase_timings["eigen_iterations"] = max(
        float(time.perf_counter() - eigen_started)
        - phase_timings["factorization"],
        0.0,
    )
    if cancellation_checkpoint("buckling.after_eigensolve"):
        trusted_guard(
            model,
            context="solve_eigenvalue_buckling cancellation after eigensolve",
        )
    else:
        exact_guard(
            model,
            context="solve_eigenvalue_buckling cancellation after eigensolve",
        )

    residual_started = time.perf_counter()
    real_eigenvectors = np.asarray(np.real(eigenvectors), dtype=float)
    residual_batch_started = time.perf_counter()
    trusted_guard(model, context="solve_eigenvalue_buckling residual batch preflight")
    stiffness_vectors = np.asarray(K_sym @ real_eigenvectors, dtype=float)
    geometric_vectors = np.asarray(KG_sym @ real_eigenvectors, dtype=float)
    trusted_guard(model, context="solve_eigenvalue_buckling residual batch output")
    residual_batch_seconds = float(time.perf_counter() - residual_batch_started)
    phase_timings["residual_batch"] = residual_batch_seconds
    candidates: List[
        tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]
    ] = []
    rejected: List[Dict[str, Any]] = []
    for i in range(real_eigenvectors.shape[1]):
        recovery_checkpoint_is_exact_noop = cancellation_checkpoint(
            f"buckling.root:{i + 1}"
        )
        if not recovery_checkpoint_is_exact_noop:
            exact_guard(
                model,
                context="solve_eigenvalue_buckling cancellation during recovery",
            )
        else:
            trusted_guard(
                model,
                context="solve_eigenvalue_buckling cancellation during recovery",
            )
        reduced_mode = real_eigenvectors[:, i].copy()
        stiffness_vector = stiffness_vectors[:, i].copy()
        geometric_vector = geometric_vectors[:, i].copy()
        mode_norm = float(np.linalg.norm(reduced_mode))
        if not np.isfinite(mode_norm) or mode_norm <= 0.0:
            rejected.append({"root_index": int(i), "reason": "invalid_or_zero_norm"})
            continue
        reduced_mode = reduced_mode / mode_norm
        stiffness_vector = stiffness_vector / mode_norm
        geometric_vector = geometric_vector / mode_norm
        if projection_y_vectors is not None:
            y_mode = np.asarray(np.real(projection_y_vectors[:, i]), dtype=np.float64)
            y_mode_norm = float(np.linalg.norm(y_mode))
            if not np.isfinite(y_mode_norm) or y_mode_norm <= 0.0:
                rejected.append({"root_index": int(i), "reason": "invalid_or_zero_dimensionless_norm"})
                continue
            y_mode = y_mode / y_mode_norm
            rigid_body_correlation = float(
                np.max(np.abs(projected_rigid_basis.T @ y_mode))
            )
        else:
            rigid_body_correlation = (
                float(np.max(np.abs(Q_rigid.T @ reduced_mode)))
                if Q_rigid.shape[1]
                else 0.0
            )
        if projection_y_vectors is None and rigid_body_correlation > 0.90 and not allow_free_mechanisms:
            rejected.append(
                {
                    "root_index": int(i),
                    "reason": "rigid_body_mode",
                    "rigid_body_correlation": rigid_body_correlation,
                }
            )
            continue
        modal_geometric = float(reduced_mode @ geometric_vector)
        modal_stiffness = float(reduced_mode @ stiffness_vector)
        if modal_geometric <= eigen_tolerance or modal_stiffness <= 0.0:
            rejected.append(
                {
                    "root_index": int(i),
                    "reason": "nonpositive_modal_terms",
                    "modal_stiffness": float(modal_stiffness),
                    "modal_geometric_stiffness": float(modal_geometric),
                }
            )
            continue
        rayleigh_value = modal_stiffness / modal_geometric
        if rayleigh_value <= eigen_tolerance or not np.isfinite(rayleigh_value):
            rejected.append({"root_index": int(i), "reason": "invalid_load_factor", "load_factor": float(rayleigh_value)})
            continue
        if not _factor_in_range(rayleigh_value, load_factor_range):
            rejected.append({"root_index": int(i), "reason": "outside_load_factor_range", "load_factor": float(rayleigh_value)})
            continue
        residual_vector = stiffness_vector - rayleigh_value * geometric_vector
        residual_norm = float(
            np.linalg.norm(residual_vector)
            / max(
                float(np.linalg.norm(stiffness_vector))
                + abs(rayleigh_value) * float(np.linalg.norm(geometric_vector)),
                1.0,
            )
        )
        candidates.append(
            (
                rayleigh_value,
                reduced_mode,
                stiffness_vector,
                geometric_vector,
                modal_stiffness,
                modal_geometric,
                residual_norm,
                rigid_body_correlation,
            )
        )

    candidates.sort(key=lambda item: _sort_key(item[0], shift_load_factor))
    modes: List[BucklingMode] = []
    for mode_number, (
        value,
        reduced_mode,
        stiffness_vector,
        geometric_vector,
        _modal_stiffness,
        _modal_geometric,
        _residual_norm,
        rigid_body_correlation,
    ) in enumerate(
        candidates[:num_modes],
        start=1,
    ):
        full_mode = np.asarray(T @ reduced_mode, dtype=float).reshape(-1)
        scale = float(np.max(np.abs(full_mode))) if full_mode.size else 0.0
        if scale > 0.0:
            full_mode = full_mode / scale
            reduced_mode = reduced_mode / scale
            stiffness_vector = stiffness_vector / scale
            geometric_vector = geometric_vector / scale
        modal_stiffness = float(reduced_mode @ stiffness_vector)
        modal_geometric = float(reduced_mode @ geometric_vector)
        residual_vector = stiffness_vector - value * geometric_vector
        residual_norm = float(
            np.linalg.norm(residual_vector)
            / max(
                float(np.linalg.norm(stiffness_vector))
                + abs(value) * float(np.linalg.norm(geometric_vector)),
                1.0,
            )
        )
        validity = "ok" if residual_norm <= max(1.0e-6, 100.0 * eigen_tolerance) else "high_residual"
        modes.append(
            BucklingMode(
                mode_number=mode_number,
                load_factor=float(value),
                eigenvalue=float(value),
                mode_shape=full_mode,
                reduced_mode_shape=reduced_mode,
                modal_stiffness=modal_stiffness,
                modal_geometric_stiffness=modal_geometric,
                residual_norm=residual_norm,
                rigid_body_correlation=rigid_body_correlation,
                validity_status=validity,
            )
        )

    repeated_groups = _assign_repeated_groups(modes, repeated_tolerance)
    phase_timings["residual_checks_and_mode_recovery"] = float(
        time.perf_counter() - residual_started
    )
    phase_timings["total"] = float(time.perf_counter() - operation_started)
    status = "ok" if modes else "no_positive_modes"
    diagnostics = {
        "status": status,
        "solver": solver_kind,
        "follower_load_stiffness_included": bool(K_load_red.nnz),
        "follower_tangent_symmetry_error": follower_symmetry_error,
        **shift_invert_diagnostics,
        **elastic_inverse_diagnostics,
        "sparse_error": sparse_error,
        "phase_timings_seconds": phase_timings,
        "spectral_guard_diagnostics": {
            "full_guard_count": int(guard_counts["full"]),
            "trusted_guard_count": int(guard_counts["trusted"]),
            "residual_batch_columns": int(real_eigenvectors.shape[1]),
        },
        "matrix_diagnostics": {
            "elastic_reduced_shape": [int(value) for value in K_sym.shape],
            "elastic_reduced_nnz": int(K_sym.nnz),
            "geometric_reduced_nnz": int(KG_sym.nnz),
            "elastic_reduced_storage_bytes": int(
                K_sym.data.nbytes + K_sym.indices.nbytes + K_sym.indptr.nbytes
            ),
            "geometric_reduced_storage_bytes": int(
                KG_sym.data.nbytes + KG_sym.indices.nbytes + KG_sym.indptr.nbytes
            ),
        },
        "nullspace_rank": int(Q_rigid.shape[1]),
        "nullspace_info": nullspace_info,
        "free_mechanism_handling": "retained" if allow_free_mechanisms else "rigid_body_roots_filtered",
        "num_roots_considered": int(eigenvectors.shape[1]),
        "num_rejected_roots": int(len(rejected)),
        "rejected_roots": rejected[:50],
        "max_residual_norm": max((mode.residual_norm for mode in modes), default=0.0),
        "repeated_mode_groups": repeated_groups,
        "num_repeated_mode_groups": int(len(repeated_groups)),
        "sorting": "nearest_shift" if shift_load_factor is not None else "ascending_load_factor",
        "constraint_postcheck": constraint_residual_summary(
            model,
            np.column_stack([mode.mode_shape for mode in modes])
            if modes
            else np.zeros((model.mesh.dof_manager.total_dofs, 0), dtype=float),
            homogeneous_variation=True,
        ),
    }
    if projection_diagnostics is not None:
        diagnostics["rigid_body_handling"] = "projected"
        diagnostics["rigid_projection"] = projection_diagnostics
    if session is not None:
        diagnostics["analysis_session"] = session.diagnostics()
        exact_guard(model, context="solve_eigenvalue_buckling session diagnostics")
    result_case = make_result_case(
        name="linear_buckling",
        analysis_type="linear_buckling",
        load_cases=() if reference_load_case is None else (reference_load_case,),
        assembly_info=assembly_info,
        solver_info={"convergence_info": diagnostics},
        recovery={"modes": num_modes, "num_modes_returned": len(modes)},
        settings=settings,
        metadata={"prestress_state_source": prestress_state_source},
    ).to_dict()
    emit_progress(
        progress_callback,
        "buckling_complete",
        "buckling.complete",
        completed=len(modes),
        total=num_modes,
        status=status,
        num_modes_returned=len(modes),
    )
    exact_guard(model, context="solve_eigenvalue_buckling output")
    return BucklingResult(modes, num_modes, status, constraint_info, assembly_info, result_case, diagnostics)


def solve_eigenvalue_buckling(
    model: "FEModel",
    element_states: Optional[Any] = None,
    num_modes: int = 3,
    eigen_tolerance: float = 1.0e-8,
    dense_size_limit: int = 200,
    shift_load_factor: Optional[float] = None,
    load_factor_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    search_factor: int = 4,
    repeated_tolerance: float = 1.0e-3,
    allow_dense_fallback: bool = False,
    allow_free_mechanisms: bool = False,
    factorization_cache: Optional[FactorizationCache] = None,
    reference_load_case: Optional["LoadCase"] = None,
    reference_displacements: Optional[np.ndarray] = None,
    follower_symmetry_tolerance: float = 1.0e-10,
    resource_config: Optional[ResourceConfig] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    session: Optional["AnalysisSession"] = None,
    reference_elastic_only: bool = False,
    current_state_displacements: Optional[Any] = None,
    current_state_element_states: Optional[Any] = None,
    current_state_num_layers: int = 5,
    current_state_load_scale: float = 1.0,
) -> BucklingResult:
    """Solve buckling under one non-renewable qualified-family lease."""

    exact_guard = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD
    run_under_lease = _run_with_qualified_assembly_runtime_lease
    own_resource_config = _owned_resource_config_snapshot
    solve_under_lease = _solve_eigenvalue_buckling_under_lease
    exact_guard(
        model,
        context="solve_eigenvalue_buckling preflight",
    )

    def operation(lease: Any) -> BucklingResult:
        def post_observation() -> None:
            exact_guard(
                model,
                context="solve_eigenvalue_buckling resource configuration",
            )
            lease(
                model,
                context="solve_eigenvalue_buckling resource configuration",
            )

        owned_resource_config = own_resource_config(
            resource_config,
            post_observation=post_observation,
        )
        return solve_under_lease(
            model,
            element_states=element_states,
            num_modes=num_modes,
            eigen_tolerance=eigen_tolerance,
            dense_size_limit=dense_size_limit,
            shift_load_factor=shift_load_factor,
            load_factor_range=load_factor_range,
            search_factor=search_factor,
            repeated_tolerance=repeated_tolerance,
            allow_dense_fallback=allow_dense_fallback,
            allow_free_mechanisms=allow_free_mechanisms,
            factorization_cache=factorization_cache,
            reference_load_case=reference_load_case,
            reference_displacements=reference_displacements,
            follower_symmetry_tolerance=follower_symmetry_tolerance,
            resource_config=owned_resource_config,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
            session=session,
            reference_elastic_only=reference_elastic_only,
            current_state_displacements=current_state_displacements,
            current_state_element_states=current_state_element_states,
            current_state_num_layers=current_state_num_layers,
            current_state_load_scale=current_state_load_scale,
            _qualified_runtime_guard=lease,
        )

    return run_under_lease(
        model,
        context="solve_eigenvalue_buckling",
        operation=operation,
    )
