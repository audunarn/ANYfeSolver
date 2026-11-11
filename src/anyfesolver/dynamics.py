"""Linear transient dynamics and prescribed pressure-patch loading.

The v1 transient path is intentionally conservative: it reuses the existing
linear stiffness, mass, load-vector and constraint-transformation machinery,
then advances the reduced system with the unconditionally stable Newmark
average-acceleration method by default.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from .algebraic_dynamics import (
    AlgebraicDynamicsError,
    AlgebraicStaticReduction,
    DESCRIPTOR_TRANSIENT_CONSTRAINED_POLICY_ID,
    DESCRIPTOR_TRANSIENT_FIRST_ORDER_POLICY_ID,
    DESCRIPTOR_TRANSIENT_POLICY_ID,
    DESCRIPTOR_TRANSIENT_STATIC_POLICY_ID,
    build_algebraic_static_reduction,
    build_declared_algebraic_basis,
    certify_descriptor_effective_operator,
    declared_algebraic_mass_elements,
)
from .assembly import build_constraint_transformation, reconstruct_full_solution
from .cases import make_result_case
from .constraint_audit import constraint_residual_summary
from .control import CancellationToken, ProgressCallback, cancellation_safe_point, emit_progress
from .element_capabilities import ElementCapabilityError, require_model_element_capabilities
from .e4_pl_s3_state import (
    require_exact_numpy_runtime_authority as _EXACT_NUMERICAL_RUNTIME_GUARD,
)
from .linalg import MatrixClass, factorize
from .boundary import LoadCase
from .matrix_assembly import (
    _run_with_qualified_assembly_runtime_lease,
    assemble_load_vector,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
)
from .modal import (
    QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID as _EXACT_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID,
    _require_qualified_prestress_operator_authority as _EXACT_PRESTRESS_OPERATOR_AUTHORITY_GUARD,
)
from .recovery import RecoveryConfig, ResourceConfig, enforce_memory_limit, estimate_model_memory, recovery_metadata
from .validation import load_vector_resultant
from .threading_policy import resource_threaded

if TYPE_CHECKING:
    from .analysis_session import AnalysisSession
    from .fe_core import FEModel


QUALIFIED_REFERENCE_TRANSIENT_AUTHORITY_POLICY_ID = (
    "EXACT_Q4_S3_REFERENCE_OPERATORS_ACTIVE_STATE_ONLY_V1"
)
_QUALIFIED_Q4_FORMULATION_ID = "E4_PL_QUALIFIED_Q4_HYBRID_V2"
_QUALIFIED_S3_FORMULATION_ID = "E4_PL_QUALIFIED_S3_COMPANION_V1"


def _bind_transient_numerical_guard(numerical_guard: Any) -> Any:
    def require(*, context: str) -> None:
        try:
            numerical_guard(context=context)
        except ValueError as exc:
            raise ElementCapabilityError(str(exc)) from exc

    return require


_TRANSIENT_NUMERICAL_GUARD = _bind_transient_numerical_guard(
    _EXACT_NUMERICAL_RUNTIME_GUARD
)


def _static_mro_attribute(owner: type[Any], name: str) -> Any:
    """Return a class member without invoking a caller-controlled descriptor."""

    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            value = namespace[name]
            if isinstance(value, (classmethod, staticmethod)):
                return value.__func__
            return value
    return None


def _static_formulation_id(element: Any) -> Optional[str]:
    value = _static_mro_attribute(type(element), "formulation_id")
    return value if type(value) is str else None


def _require_qualified_reference_transient_authority(
    model: "FEModel",
    *,
    _prestress_guard: Any = _EXACT_PRESTRESS_OPERATOR_AUTHORITY_GUARD,
    _prestress_policy_id: str = _EXACT_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID,
) -> Dict[str, Any]:
    """Validate qualified reference operators and ACTIVE lifecycle pre-mechanics.

    Linear Newmark advances the reference stiffness and consistent mass.  It
    therefore rejects softened or hard-deleted model activity.  This route has
    no element-state input and makes no claim about detached failed-result
    records.  The exact operator guard is shared with the reference-prestress
    modal/buckling paths; this wrapper adds only the full-activity condition.
    """

    from .activity import ElementActivity
    from .e4_pl_element import (
        IMPLEMENTATION_ID,
        Q4_QUADRATURE_AUTHORITY_ID,
        QualifiedE4PLShellElement,
    )
    from .e4_pl_s3_element import (
        ALGEBRAIC_COORDINATE_POLICY_ID,
        S3_QUADRATURE_AUTHORITY_ID,
        QualifiedE4PLS3ShellElement,
    )
    qualified_ids: list[int] = []
    q4_ids: list[int] = []
    s3_ids: list[int] = []
    for raw_element_id, element in sorted(model.mesh.elements.items()):
        element_id = int(raw_element_id)
        formulation_id = _static_formulation_id(element)
        if (
            type(element) is QualifiedE4PLShellElement
            or formulation_id == _QUALIFIED_Q4_FORMULATION_ID
        ):
            qualified_ids.append(element_id)
            q4_ids.append(element_id)
        elif (
            type(element) is QualifiedE4PLS3ShellElement
            or formulation_id == _QUALIFIED_S3_FORMULATION_ID
        ):
            qualified_ids.append(element_id)
            s3_ids.append(element_id)

    if not qualified_ids:
        return {
            "policy_id": QUALIFIED_REFERENCE_TRANSIENT_AUTHORITY_POLICY_ID,
            "active": False,
            "qualified_element_ids": [],
            "formulation_counts": {},
            "activity_disposition": "NOT_APPLICABLE",
        }

    # ``None`` is the exact reference-elastic state for both qualified
    # families.  The shared guard binds concrete type/formulation routing,
    # stiffness and mass operators, inherited helpers, quadrature, S3
    # algebraic-coordinate descriptors, and exact nodal DOF mapping.
    reference_states = {
        int(element_id): None for element_id in model.mesh.elements
    }
    _prestress_guard(
        model,
        reference_states,
        include_mass_and_descriptor=True,
    )

    failures: list[tuple[int, str]] = []
    descriptor_names = (
        "dynamic_algebraic_nullity",
        "dynamic_algebraic_policy",
        "dynamic_algebraic_mass_witness",
        "dynamic_algebraic_local_zero_indices",
    )
    for element_id in q4_ids:
        element = model.mesh.elements[element_id]
        if (
            "formulation_id" in vars(element)
            or "implementation_id" in vars(element)
            or "quadrature_authority_id" in vars(element)
            or _static_mro_attribute(type(element), "formulation_id")
            != _QUALIFIED_Q4_FORMULATION_ID
            or _static_mro_attribute(type(element), "implementation_id")
            != IMPLEMENTATION_ID
            or _static_mro_attribute(type(element), "quadrature_authority_id")
            != Q4_QUADRATURE_AUTHORITY_ID
        ):
            failures.append((element_id, "qualified_q4:identity"))
        if any(
            name in vars(element)
            or _static_mro_attribute(type(element), name) is not None
            for name in descriptor_names
        ):
            failures.append(
                (element_id, "qualified_q4:unexpected_algebraic_descriptor")
            )
    for element_id in s3_ids:
        element = model.mesh.elements[element_id]
        if (
            "formulation_id" in vars(element)
            or "quadrature_authority_id" in vars(element)
            or _static_mro_attribute(type(element), "formulation_id")
            != _QUALIFIED_S3_FORMULATION_ID
            or _static_mro_attribute(type(element), "quadrature_authority_id")
            != S3_QUADRATURE_AUTHORITY_ID
        ):
            failures.append((element_id, "qualified_s3:identity"))
        class_namespace = vars(type(element))
        dynamic_identity = (
            class_namespace.get("dynamic_algebraic_nullity"),
            class_namespace.get("dynamic_algebraic_policy"),
            class_namespace.get("dynamic_algebraic_mass_witness"),
            class_namespace.get("dynamic_algebraic_local_zero_indices"),
        )
        if dynamic_identity != (
            3,
            ALGEBRAIC_COORDINATE_POLICY_ID,
            "S3_LOCAL_DRILL_ROWS_EXACT_ZERO_V1",
            (5, 11, 17),
        ) or any(name in vars(element) for name in descriptor_names):
            failures.append((element_id, "qualified_s3:algebraic_descriptor"))
    if failures:
        detail = "; ".join(
            f"{element_id} ({reason})" for element_id, reason in failures[:8]
        )
        raise ElementCapabilityError(
            "linear transient analysis requires exact qualified formulation "
            f"identity and algebraic descriptors; incompatible element IDs {detail}"
        )

    activity = getattr(model.mesh, "element_activity", None)
    activity_sequence = 0
    if activity is not None:
        if type(activity) is not ElementActivity:
            raise ElementCapabilityError(
                "linear transient analysis requires exact ElementActivity "
                "ownership for qualified Q4/S3 elements"
            )
        raw_activity_ids = np.asarray(activity.element_ids)
        model_ids = tuple(sorted(int(value) for value in model.mesh.elements))
        if (
            raw_activity_ids.ndim != 1
            or raw_activity_ids.dtype.kind not in "iu"
            or len(set(int(value) for value in raw_activity_ids))
            != raw_activity_ids.size
            or tuple(sorted(int(value) for value in raw_activity_ids)) != model_ids
        ):
            raise ElementCapabilityError(
                "linear transient ElementActivity is not bound to the exact FE model"
            )
        values = np.asarray(activity.activity, dtype=np.float64)
        hard_deleted = np.asarray(activity.hard_deleted_mask, dtype=bool)
        if (
            values.shape != raw_activity_ids.shape
            or hard_deleted.shape != raw_activity_ids.shape
            or not np.all(np.isfinite(values))
        ):
            raise ElementCapabilityError(
                "linear transient ElementActivity state is malformed"
            )
        index = {
            int(element_id): position
            for position, element_id in enumerate(raw_activity_ids)
        }
        inactive = [
            element_id
            for element_id in qualified_ids
            if values[index[element_id]] != 1.0
            or bool(hard_deleted[index[element_id]])
        ]
        if inactive:
            raise ElementCapabilityError(
                "linear transient qualified Q4/S3 reference operators require "
                "exact full model activity; softened or hard-deleted element "
                "IDs "
                + ", ".join(str(value) for value in inactive[:8])
            )
        activity_sequence = int(activity.sequence)

    _prestress_guard(
        model,
        reference_states,
        include_mass_and_descriptor=True,
    )
    return {
        "policy_id": QUALIFIED_REFERENCE_TRANSIENT_AUTHORITY_POLICY_ID,
        "active": True,
        "qualified_element_ids": qualified_ids,
        "formulation_counts": {
            _QUALIFIED_Q4_FORMULATION_ID: len(q4_ids),
            _QUALIFIED_S3_FORMULATION_ID: len(s3_ids),
        },
        "activity_disposition": "ACTIVE_REFERENCE_ONLY",
        "activity_sequence": activity_sequence,
        "operator_authority_policy_id": (
            _prestress_policy_id
        ),
    }


PressureTime = Union[float, int, Sequence[Tuple[float, float]], Callable[[float], float]]
PatchSelector = Callable[[int, Any, np.ndarray], bool]


def _as_axes(axes: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(int(axis) for axis in axes)
    if not result:
        raise ValueError("axes must contain at least one coordinate index")
    if any(axis < 0 or axis > 2 for axis in result):
        raise ValueError("axes entries must be 0, 1 or 2")
    return result


def _time_value(value: PressureTime, time: float) -> float:
    numerical_guard = _TRANSIENT_NUMERICAL_GUARD
    numerical_guard(context="transient pressure history")
    if callable(value):
        observed = value(float(time))
        numerical_guard(context="transient pressure history callback")
        return float(observed)
    if isinstance(value, (int, float, np.number)):
        return float(value)

    observed_rows = list(value)
    numerical_guard(context="transient pressure history iterable")
    owned_rows = [tuple(row) for row in observed_rows]
    numerical_guard(context="transient pressure history rows")
    table = np.asarray(owned_rows, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] == 0:
        raise ValueError("pressure_time table must contain (time, pressure) pairs")
    order = np.argsort(table[:, 0])
    table = table[order]
    return float(np.interp(float(time), table[:, 0], table[:, 1]))


def _guarded_pressure_value(
    model: "FEModel",
    patch: "PressurePatch",
    time: float,
    operation_guard: Callable[["FEModel"], Any],
) -> float:
    """Evaluate one retained pressure callback and stop at its first mutation."""

    pressure_time = patch.pressure_time
    operation_guard(model)
    if callable(pressure_time):
        observed = pressure_time(float(time))
        operation_guard(model)
        magnitude = float(observed)
        operation_guard(model)
    else:
        magnitude = _time_value(pressure_time, float(time))
        operation_guard(model)
    pressure_scale = patch.pressure_scale
    operation_guard(model)
    scale = float(pressure_scale)
    operation_guard(model)
    return scale * magnitude


@dataclass(frozen=True)
class PressurePatch:
    """Prescribed shell pressure over a selected area.

    Element selection is centroid-based in v1.  Supply explicit ``element_ids``
    for exact control, or use ``center`` with ``box_size`` and/or ``radius``.
    Positive pressure follows the shell element normal, matching
    ``LoadCase.add_pressure_load``.
    """

    name: str
    pressure_time: PressureTime
    element_ids: Optional[Sequence[int]] = None
    selector: Optional[PatchSelector] = None
    center: Optional[Sequence[float]] = None
    box_size: Optional[Sequence[float]] = None
    radius: Optional[float] = None
    axes: Sequence[int] = (0, 1)
    pressure_scale: float = 1.0
    normal_mode: str = "element_normal"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def rectangular(
        cls,
        name: str,
        pressure_time: PressureTime,
        center: Sequence[float],
        size: Sequence[float],
        axes: Sequence[int] = (0, 1),
        **kwargs: Any,
    ) -> "PressurePatch":
        """Create a centroid-selected rectangular patch."""
        return cls(name=name, pressure_time=pressure_time, center=center, box_size=size, axes=axes, **kwargs)

    @classmethod
    def circular(
        cls,
        name: str,
        pressure_time: PressureTime,
        center: Sequence[float],
        radius: float,
        axes: Sequence[int] = (0, 1),
        **kwargs: Any,
    ) -> "PressurePatch":
        """Create a centroid-selected circular patch."""
        return cls(name=name, pressure_time=pressure_time, center=center, radius=radius, axes=axes, **kwargs)

    @classmethod
    def rectangular_pulse(
        cls,
        name: str,
        pressure: float,
        start_time: float,
        end_time: float,
        **kwargs: Any,
    ) -> "PressurePatch":
        """Create a constant pressure pulse with zero load before/after."""
        if end_time < start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        start = float(start_time)
        end = float(end_time)
        magnitude = float(pressure)

        def pulse(time: float) -> float:
            return magnitude if start <= float(time) < end else 0.0

        return cls(name=name, pressure_time=pulse, **kwargs)

    def pressure_at(self, time: float) -> float:
        """Pressure magnitude at ``time`` in Pa."""
        return float(self.pressure_scale) * _time_value(self.pressure_time, time)

    def selected_element_ids(self, model: "FEModel") -> Tuple[int, ...]:
        """Return selected shell-like element ids in stable model order."""
        authority_guard = _require_qualified_reference_transient_authority
        authority_guard(model)

        def operation(lease: Any) -> Tuple[int, ...]:
            def combined_guard(observed_model: "FEModel") -> None:
                authority_guard(observed_model)
                lease(
                    observed_model,
                    context="pressure-patch selection observation",
                )

            return _selected_pressure_patch_element_ids(
                self,
                model,
                combined_guard,
            )

        return _run_with_qualified_assembly_runtime_lease(
            model,
            context="PressurePatch.selected_element_ids",
            operation=operation,
        )


def _selected_pressure_patch_element_ids(
    patch: PressurePatch,
    model: "FEModel",
    operation_guard: Callable[["FEModel"], Any],
) -> Tuple[int, ...]:
    """Select one patch while retaining its caller's operation generation."""

    normal_mode = patch.normal_mode
    operation_guard(model)
    if normal_mode != "element_normal":
        raise NotImplementedError(
            "PressurePatch v1 supports normal_mode='element_normal' only"
        )

    element_ids = patch.element_ids
    operation_guard(model)
    explicit = (
        None
        if element_ids is None
        else {int(element_id) for element_id in element_ids}
    )
    operation_guard(model)
    axes = _as_axes(patch.axes)
    operation_guard(model)
    center_value = patch.center
    operation_guard(model)
    center = (
        None
        if center_value is None
        else np.asarray(center_value, dtype=float)
    )
    operation_guard(model)
    if center is not None:
        center = center.reshape(-1)
    if center is not None and center.size < 3:
        padded = np.zeros(3, dtype=float)
        padded[: center.size] = center
        center = padded
    box_value = patch.box_size
    operation_guard(model)
    box = (
        None
        if box_value is None
        else np.asarray(box_value, dtype=float)
    )
    operation_guard(model)
    if box is not None:
        box = box.reshape(-1)
        if box.size not in (len(axes), 3):
            raise ValueError(
                "box_size must have either len(axes) entries or 3 entries"
            )
        if np.any(box <= 0.0):
            raise ValueError("box_size entries must be positive")
    radius_value = patch.radius
    operation_guard(model)
    radius = None if radius_value is None else float(radius_value)
    operation_guard(model)
    selector = patch.selector
    operation_guard(model)
    element_items = tuple(model.mesh.elements.items())
    operation_guard(model)

    selected = []
    for element_id, element in element_items:
        if explicit is not None:
            if int(element_id) in explicit:
                selected.append(int(element_id))
            continue
        coordinate_getter = getattr(element, "get_node_coordinates", None)
        operation_guard(model)
        shape_getter = getattr(element, "compute_shape_functions", None)
        operation_guard(model)
        if not callable(coordinate_getter) or not callable(shape_getter):
            continue
        observed_coords = coordinate_getter(model.mesh)
        operation_guard(model)
        coords = np.asarray(observed_coords, dtype=float)
        operation_guard(model)
        centroid = np.mean(coords, axis=0)
        include = True
        if selector is not None:
            selected_by_callback = selector(int(element_id), element, centroid)
            operation_guard(model)
            include = bool(selected_by_callback)
            operation_guard(model)
        if include and center is not None and box is not None:
            half_size = (
                box[list(axes)] / 2.0
                if box.size == 3
                else box / 2.0
            )
            include = bool(
                np.all(
                    np.abs(centroid[list(axes)] - center[list(axes)])
                    <= half_size + 1.0e-12
                )
            )
        if include and center is not None and radius is not None:
            include = (
                float(
                    np.linalg.norm(
                        centroid[list(axes)] - center[list(axes)]
                    )
                )
                <= radius + 1.0e-12
            )
        if include:
            selected.append(int(element_id))
    operation_guard(model)
    return tuple(selected)


@dataclass(frozen=True)
class TransientConfig:
    """Configuration for linear Newmark/HHT-alpha transient analysis.

    ``hht_alpha`` activates Hilber-Hughes-Taylor numerical dissipation.  The
    valid range is ``-1/3 <= hht_alpha <= 0``; ``hht_alpha = 0`` reproduces the
    plain Newmark method with the configured ``beta``/``gamma``.  For a
    non-zero ``hht_alpha`` with ``beta``/``gamma`` left at their defaults, the
    solvers use the second-order-accurate HHT-optimal parameters
    ``gamma = 1/2 - alpha`` and ``beta = (1 - alpha)^2 / 4``.
    """

    dt: float
    t_end: float
    beta: float = 0.25
    gamma: float = 0.5
    hht_alpha: float = 0.0
    rayleigh_alpha: float = 0.0
    rayleigh_beta: float = 0.0
    save_every: int = 1
    output_nodes: Optional[Sequence[int]] = None
    output_elements: Optional[Sequence[int]] = None
    initial_displacement: Optional[np.ndarray] = None
    initial_velocity: Optional[np.ndarray] = None
    include_stress_history: bool = False
    recovery: Optional[RecoveryConfig] = None
    resource_config: Optional[ResourceConfig] = None

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.t_end < 0.0:
            raise ValueError("t_end must be non-negative")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if not (-1.0 / 3.0 - 1.0e-12 <= float(self.hht_alpha) <= 0.0):
            raise ValueError("hht_alpha must be in [-1/3, 0]")
        if self.save_every <= 0:
            raise ValueError("save_every must be a positive integer")

    def integration_parameters(self) -> Tuple[float, float, float]:
        """Return the ``(hht_alpha, beta, gamma)`` used by the time integrators.

        With a non-zero ``hht_alpha`` and default ``beta``/``gamma``, the
        HHT-optimal Newmark parameters are derived from ``hht_alpha``;
        explicitly configured non-default ``beta``/``gamma`` are respected.
        """
        alpha = float(self.hht_alpha)
        if alpha == 0.0:
            return 0.0, float(self.beta), float(self.gamma)
        if float(self.beta) == 0.25 and float(self.gamma) == 0.5:
            return alpha, 0.25 * (1.0 - alpha) ** 2, 0.5 - alpha
        return alpha, float(self.beta), float(self.gamma)


def _guarded_iterable_snapshot(
    model: "FEModel",
    value: Any,
    operation_guard: Callable[["FEModel"], Any],
) -> Tuple[Any, ...]:
    """Own one iterable while checking after every caller observation."""

    iterator = iter(value)
    operation_guard(model)
    owned: list[Any] = []
    while True:
        try:
            member = next(iterator)
        except StopIteration:
            operation_guard(model)
            break
        operation_guard(model)
        owned.append(member)
    return tuple(owned)


def _guarded_plain_snapshot(
    model: "FEModel",
    value: Any,
    operation_guard: Callable[["FEModel"], Any],
) -> Any:
    """Recursively detach configuration metadata and sequence leaves."""

    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, np.ndarray):
        observed = np.asarray(value)
        operation_guard(model)
        contiguous = np.ascontiguousarray(observed)
        return np.frombuffer(
            contiguous.tobytes(order="C"),
            dtype=contiguous.dtype,
        ).reshape(contiguous.shape)
    if isinstance(value, np.generic):
        observed = value.item()
        operation_guard(model)
        return observed
    if isinstance(value, Mapping):
        observed_items = value.items()
        operation_guard(model)
        items = _guarded_iterable_snapshot(
            model,
            observed_items,
            operation_guard,
        )
        result: Dict[Any, Any] = {}
        for observed_item in items:
            pair = _guarded_iterable_snapshot(
                model,
                observed_item,
                operation_guard,
            )
            if len(pair) != 2:
                raise ValueError("configuration mapping item must contain two values")
            raw_key, raw_value = pair
            key = _guarded_plain_snapshot(model, raw_key, operation_guard)
            member = _guarded_plain_snapshot(model, raw_value, operation_guard)
            if key in result:
                raise ValueError("configuration mapping contains duplicate keys")
            result[key] = member
        return result
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _guarded_plain_snapshot(model, member, operation_guard)
            for member in _guarded_iterable_snapshot(
                model,
                value,
                operation_guard,
            )
        )
    observed = copy.deepcopy(value)
    operation_guard(model)
    return observed


def _guarded_attribute(
    model: "FEModel",
    owner: Any,
    name: str,
    operation_guard: Callable[["FEModel"], Any],
) -> Any:
    value = getattr(owner, name)
    operation_guard(model)
    return value


def _guarded_convert(
    model: "FEModel",
    value: Any,
    converter: Callable[[Any], Any],
    operation_guard: Callable[["FEModel"], Any],
) -> Any:
    converted = converter(value)
    operation_guard(model)
    return converted


def _guarded_optional_sequence(
    model: "FEModel",
    value: Any,
    converter: Callable[[Any], Any],
    operation_guard: Callable[["FEModel"], Any],
) -> Optional[Tuple[Any, ...]]:
    if value is None:
        return None
    return tuple(
        _guarded_convert(model, member, converter, operation_guard)
        for member in _guarded_iterable_snapshot(
            model,
            value,
            operation_guard,
        )
    )


def _owned_resource_config(
    model: "FEModel",
    value: Any,
    operation_guard: Callable[["FEModel"], Any],
) -> Optional[ResourceConfig]:
    if value is None:
        return None

    def optional_int(name: str) -> Optional[int]:
        raw = _guarded_attribute(model, value, name, operation_guard)
        return (
            None
            if raw is None
            else _guarded_convert(model, raw, int, operation_guard)
        )

    deterministic = _guarded_convert(
        model,
        _guarded_attribute(model, value, "deterministic", operation_guard),
        bool,
        operation_guard,
    )
    raw_metadata = _guarded_attribute(
        model,
        value,
        "metadata",
        operation_guard,
    )
    metadata = _guarded_plain_snapshot(model, raw_metadata, operation_guard)
    owned = ResourceConfig(
        solver_threads=optional_int("solver_threads"),
        assembly_threads=optional_int("assembly_threads"),
        recovery_threads=optional_int("recovery_threads"),
        process_workers=optional_int("process_workers"),
        deterministic=deterministic,
        memory_limit_bytes=optional_int("memory_limit_bytes"),
        metadata=metadata,
    )
    operation_guard(model)
    return owned


def _owned_recovery_config(
    model: "FEModel",
    value: Any,
    operation_guard: Callable[["FEModel"], Any],
) -> Optional[RecoveryConfig]:
    if value is None:
        return None
    node_ids = _guarded_optional_sequence(
        model,
        _guarded_attribute(model, value, "node_ids", operation_guard),
        int,
        operation_guard,
    )
    element_ids = _guarded_optional_sequence(
        model,
        _guarded_attribute(model, value, "element_ids", operation_guard),
        int,
        operation_guard,
    )
    components = _guarded_optional_sequence(
        model,
        _guarded_attribute(model, value, "components", operation_guard),
        str,
        operation_guard,
    )
    include_displacements = _guarded_convert(
        model,
        _guarded_attribute(
            model,
            value,
            "include_displacements",
            operation_guard,
        ),
        bool,
        operation_guard,
    )
    include_stresses = _guarded_convert(
        model,
        _guarded_attribute(model, value, "include_stresses", operation_guard),
        bool,
        operation_guard,
    )
    include_reactions = _guarded_convert(
        model,
        _guarded_attribute(model, value, "include_reactions", operation_guard),
        bool,
        operation_guard,
    )
    history_mode = _guarded_convert(
        model,
        _guarded_attribute(model, value, "history_mode", operation_guard),
        str,
        operation_guard,
    )
    store_full_histories = _guarded_convert(
        model,
        _guarded_attribute(
            model,
            value,
            "store_full_histories",
            operation_guard,
        ),
        bool,
        operation_guard,
    )
    metadata = _guarded_plain_snapshot(
        model,
        _guarded_attribute(model, value, "metadata", operation_guard),
        operation_guard,
    )
    owned = RecoveryConfig(
        node_ids=node_ids,
        element_ids=element_ids,
        components=components,
        include_displacements=include_displacements,
        include_stresses=include_stresses,
        include_reactions=include_reactions,
        history_mode=history_mode,
        store_full_histories=store_full_histories,
        metadata=metadata,
    )
    operation_guard(model)
    return owned


def _owned_pressure_time(
    model: "FEModel",
    value: Any,
    operation_guard: Callable[["FEModel"], Any],
) -> PressureTime:
    if callable(value):
        return value
    if isinstance(value, (int, float, np.number)):
        return _guarded_convert(model, value, float, operation_guard)
    rows = _guarded_iterable_snapshot(model, value, operation_guard)
    owned_rows = []
    for raw_row in rows:
        row = _guarded_iterable_snapshot(model, raw_row, operation_guard)
        if len(row) != 2:
            raise ValueError(
                "pressure_time table must contain (time, pressure) pairs"
            )
        owned_rows.append(
            tuple(
                _guarded_convert(model, member, float, operation_guard)
                for member in row
            )
        )
    return tuple(owned_rows)


def _owned_pressure_patch(
    model: "FEModel",
    patch: Any,
    operation_guard: Callable[["FEModel"], Any],
) -> PressurePatch:
    name = _guarded_convert(
        model,
        _guarded_attribute(model, patch, "name", operation_guard),
        str,
        operation_guard,
    )
    pressure_time = _owned_pressure_time(
        model,
        _guarded_attribute(model, patch, "pressure_time", operation_guard),
        operation_guard,
    )
    element_ids = _guarded_optional_sequence(
        model,
        _guarded_attribute(model, patch, "element_ids", operation_guard),
        int,
        operation_guard,
    )
    selector = _guarded_attribute(model, patch, "selector", operation_guard)
    center = _guarded_optional_sequence(
        model,
        _guarded_attribute(model, patch, "center", operation_guard),
        float,
        operation_guard,
    )
    box_size = _guarded_optional_sequence(
        model,
        _guarded_attribute(model, patch, "box_size", operation_guard),
        float,
        operation_guard,
    )
    raw_radius = _guarded_attribute(model, patch, "radius", operation_guard)
    radius = (
        None
        if raw_radius is None
        else _guarded_convert(model, raw_radius, float, operation_guard)
    )
    axes = _guarded_optional_sequence(
        model,
        _guarded_attribute(model, patch, "axes", operation_guard),
        int,
        operation_guard,
    )
    pressure_scale = _guarded_convert(
        model,
        _guarded_attribute(model, patch, "pressure_scale", operation_guard),
        float,
        operation_guard,
    )
    normal_mode = _guarded_convert(
        model,
        _guarded_attribute(model, patch, "normal_mode", operation_guard),
        str,
        operation_guard,
    )
    metadata = _guarded_plain_snapshot(
        model,
        _guarded_attribute(model, patch, "metadata", operation_guard),
        operation_guard,
    )
    owned = PressurePatch(
        name=name,
        pressure_time=pressure_time,
        element_ids=element_ids,
        selector=selector,
        center=center,
        box_size=box_size,
        radius=radius,
        axes=() if axes is None else axes,
        pressure_scale=pressure_scale,
        normal_mode=normal_mode,
        metadata=metadata,
    )
    operation_guard(model)
    return owned


def _owned_transient_configuration(
    model: "FEModel",
    config: Any,
    pressure_patches: Optional[Sequence[PressurePatch]],
    operation_guard: Callable[["FEModel"], Any],
) -> Tuple[TransientConfig, Tuple[float, float, float], Tuple[PressurePatch, ...]]:
    """Own all transient configuration before matrix or element mechanics."""

    def scalar(name: str, converter: Callable[[Any], Any]) -> Any:
        return _guarded_convert(
            model,
            _guarded_attribute(model, config, name, operation_guard),
            converter,
            operation_guard,
        )

    output_nodes = _guarded_optional_sequence(
        model,
        _guarded_attribute(model, config, "output_nodes", operation_guard),
        int,
        operation_guard,
    )
    output_elements = _guarded_optional_sequence(
        model,
        _guarded_attribute(model, config, "output_elements", operation_guard),
        int,
        operation_guard,
    )

    def owned_array(name: str) -> Optional[np.ndarray]:
        raw = _guarded_attribute(model, config, name, operation_guard)
        if raw is None:
            return None
        observed = np.asarray(raw, dtype=np.float64)
        operation_guard(model)
        contiguous = np.ascontiguousarray(observed, dtype=np.float64)
        return np.frombuffer(
            contiguous.tobytes(order="C"),
            dtype=np.float64,
        ).reshape(contiguous.shape)

    recovery = _owned_recovery_config(
        model,
        _guarded_attribute(model, config, "recovery", operation_guard),
        operation_guard,
    )
    resources = _owned_resource_config(
        model,
        _guarded_attribute(model, config, "resource_config", operation_guard),
        operation_guard,
    )
    owned = TransientConfig(
        dt=scalar("dt", float),
        t_end=scalar("t_end", float),
        beta=scalar("beta", float),
        gamma=scalar("gamma", float),
        hht_alpha=scalar("hht_alpha", float),
        rayleigh_alpha=scalar("rayleigh_alpha", float),
        rayleigh_beta=scalar("rayleigh_beta", float),
        save_every=scalar("save_every", int),
        output_nodes=output_nodes,
        output_elements=output_elements,
        initial_displacement=owned_array("initial_displacement"),
        initial_velocity=owned_array("initial_velocity"),
        include_stress_history=scalar("include_stress_history", bool),
        recovery=recovery,
        resource_config=resources,
    )
    operation_guard(model)
    integration = tuple(float(value) for value in owned.integration_parameters())
    operation_guard(model)
    if len(integration) != 3 or not all(math.isfinite(value) for value in integration):
        raise ValueError("transient integration parameters must be three finite values")
    raw_patches = (
        ()
        if pressure_patches is None
        else _guarded_iterable_snapshot(
            model,
            pressure_patches,
            operation_guard,
        )
    )
    patches = tuple(
        _owned_pressure_patch(model, patch, operation_guard)
        for patch in raw_patches
    )
    operation_guard(model)
    return owned, integration, patches


@dataclass(frozen=True)
class TransientResult:
    """Saved transient response histories and diagnostics."""

    times: np.ndarray
    displacements: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    node_histories: Dict[int, np.ndarray]
    load_impulse: np.ndarray
    force_impulse: np.ndarray
    moment_impulse: np.ndarray
    peak_displacement: float
    peak_displacement_node: Optional[int]
    peak_von_mises_stress: float
    stress_history: Optional[Tuple[Dict[int, Dict[str, np.ndarray]], ...]]
    status: str
    diagnostics: Dict[str, Any]
    result_case: Optional[Dict[str, Any]] = None
    history_storage_mode: str = "full"
    history_dof_indices: Optional[np.ndarray] = None
    displacement_envelope: Optional[np.ndarray] = None
    velocity_envelope: Optional[np.ndarray] = None
    acceleration_envelope: Optional[np.ndarray] = None

    def node_displacement_history(self, model: "FEModel", node_id: int) -> np.ndarray:
        """Return saved 6-DOF displacement history for one node."""
        node = model.mesh.get_node(int(node_id))
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        if int(node_id) in self.node_histories:
            return self.node_histories[int(node_id)]
        if self.history_storage_mode != "full":
            raise ValueError(f"Node {node_id} was not saved in {self.history_storage_mode!r} transient history storage")
        return self.displacements[:, node.dofs]

    @property
    def quantity_metadata(self) -> Tuple[Any, ...]:
        from .quantities import describe_result_quantities

        return describe_result_quantities(self)


def _time_grid(config: TransientConfig) -> np.ndarray:
    if config.t_end == 0.0:
        return np.array([0.0], dtype=float)
    n_steps = int(np.ceil(config.t_end / config.dt - 1.0e-12))
    times = np.arange(n_steps + 1, dtype=float) * config.dt
    times[-1] = float(config.t_end)
    return times


def _full_initial_vector(
    value: Optional[np.ndarray],
    size: int,
    *,
    post_observation: Optional[Callable[[], Any]] = None,
) -> np.ndarray:
    if value is None:
        return np.zeros(size, dtype=float)
    vector = np.asarray(value, dtype=float)
    if post_observation is not None:
        post_observation()
    vector = vector.reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"initial vector has shape {vector.shape}; expected {(size,)}")
    return vector


def _sampled_grid_derivative(
    times: Sequence[float],
    values: Sequence[np.ndarray],
    order: int,
    *,
    evaluation_index: int,
) -> np.ndarray:
    """Differentiate a deterministic local polynomial through sampled values."""

    if not values:
        raise ValueError("load derivative requires at least one sampled value")
    reference = np.asarray(values[0], dtype=float)
    if order not in {1, 2}:
        raise ValueError("load derivative order must be one or two")
    count_all = min(len(values), len(times))
    index = int(evaluation_index)
    if index < 0:
        index += count_all
    if index < 0 or index >= count_all:
        raise ValueError("load derivative evaluation index is out of range")
    required = order + 1
    count = min(3, count_all)
    if count < required:
        return np.zeros_like(reference)
    start = min(max(index - count + 1, 0), count_all - count)
    stop = start + count
    local_times = np.asarray(times[start:stop], dtype=float)
    offsets = local_times - float(times[index])
    powers = np.arange(count, dtype=int)
    system = offsets[None, :] ** powers[:, None]
    rhs = np.zeros(count, dtype=float)
    rhs[order] = float(math.factorial(order))
    try:
        weights = np.linalg.solve(system, rhs)
    except Exception as error:
        raise ValueError("could not construct deterministic load-derivative weights") from error
    result = np.zeros_like(reference)
    derivative_origin = np.asarray(values[index], dtype=float)
    for weight, value in zip(weights, values[start:stop]):
        result += float(weight) * (
            np.asarray(value, dtype=float) - derivative_origin
        )
    if np.any(~np.isfinite(result)):
        raise ValueError("sampled load derivative is non-finite")
    return result


def _forward_grid_derivative(
    times: np.ndarray,
    values: Sequence[np.ndarray],
    order: int,
) -> np.ndarray:
    return _sampled_grid_derivative(
        times,
        values,
        order,
        evaluation_index=0,
    )


def _translation_peak_index(model: "FEModel") -> Tuple[np.ndarray, np.ndarray]:
    """Cached (node_ids, translation-DOF-index) arrays for peak scans."""
    mesh = model.mesh
    signature = mesh.revision_signature()
    cached = getattr(mesh, "_translation_peak_index_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]
    node_ids = np.fromiter((int(node_id) for node_id in mesh.nodes), dtype=np.int64, count=len(mesh.nodes))
    dof_index = np.asarray([node.dofs[:3] for node in mesh.nodes.values()], dtype=np.intp).reshape(-1, 3)
    mesh._translation_peak_index_cache = (signature, node_ids, dof_index)
    return node_ids, dof_index


def _translation_peak(model: "FEModel", displacement: np.ndarray) -> Tuple[float, Optional[int]]:
    node_ids, dof_index = _translation_peak_index(model)
    if node_ids.size == 0:
        return 0.0, None
    translations = np.asarray(displacement, dtype=float)[dof_index]
    magnitudes_sq = np.einsum("ij,ij->i", translations, translations)
    best = int(np.argmax(magnitudes_sq))
    return float(np.sqrt(magnitudes_sq[best])), int(node_ids[best])


def _saved_step_count(times: np.ndarray, save_every: int) -> int:
    if times.size == 0:
        return 0
    count = 1
    for step_index in range(1, len(times)):
        if step_index % int(save_every) == 0 or step_index == len(times) - 1:
            count += 1
    return count


def _node_dof_indices(model: "FEModel", node_ids: Sequence[int]) -> np.ndarray:
    indices = []
    for node_id in node_ids:
        node = model.mesh.get_node(int(node_id))
        if node is None:
            raise ValueError(f"Configured output node {node_id} not found")
        indices.extend(int(dof) for dof in node.dofs)
    return np.asarray(indices, dtype=np.intp)


def _make_pressure_patch_load_assembler(
    authority_guard: Callable[["FEModel"], Dict[str, Any]],
) -> Callable[["FEModel", PressurePatch, float], Tuple[np.ndarray, Dict[str, Any]]]:
    """Capture the exact guard across a caller-owned selector callback."""

    def guarded_assembler(
        model: "FEModel",
        patch: PressurePatch,
        pressure: float,
        operation_guard: Callable[["FEModel"], Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        observed_name = patch.name
        operation_guard(model)
        patch_name = str(observed_name)
        operation_guard(model)
        observed_element_ids = patch.element_ids
        operation_guard(model)
        selection_mode = (
            "explicit" if observed_element_ids is not None else "centroid"
        )
        magnitude = float(pressure)
        operation_guard(model)
        selected = _selected_pressure_patch_element_ids(
            patch,
            model,
            operation_guard,
        )
        # Selection may execute an application callback.  Re-establish the
        # same non-renewable operation generation before constructing any
        # pressure load or evaluating element shape mechanics.
        operation_guard(model)
        load_case = LoadCase(name=f"pressure_patch:{patch_name}")
        for element_id in selected:
            load_case.add_pressure_load(element_id, magnitude)
        vector, info = assemble_load_vector(model, load_case)
        operation_guard(model)
        resultant = load_vector_resultant(model, vector)
        operation_guard(model)
        info.update(
            {
                "patch_name": patch_name,
                "selected_element_ids": list(selected),
                "num_selected_elements": len(selected),
                "pressure": magnitude,
                "selection_mode": selection_mode,
                "resultant_force": resultant.force.tolist(),
                "resultant_moment": resultant.moment.tolist(),
            }
        )
        return vector, info

    def assemble_pressure_patch_load_vector(
        model: "FEModel",
        patch: PressurePatch,
        pressure: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Assemble the consistent global load vector for a unit pressure patch."""

        authority_guard(model)

        def operation(lease: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
            def combined_guard(observed_model: "FEModel") -> None:
                authority_guard(observed_model)
                lease(
                    observed_model,
                    context="pressure-patch callback observation",
                )

            return guarded_assembler(
                model,
                patch,
                pressure,
                combined_guard,
            )

        return _run_with_qualified_assembly_runtime_lease(
            model,
            context="assemble_pressure_patch_load_vector",
            operation=operation,
        )

    assemble_pressure_patch_load_vector._guarded_implementation = guarded_assembler
    return assemble_pressure_patch_load_vector


assemble_pressure_patch_load_vector = _make_pressure_patch_load_assembler(
    _require_qualified_reference_transient_authority
)
_ASSEMBLE_PRESSURE_PATCH_LOAD_VECTOR_GUARDED = (
    assemble_pressure_patch_load_vector._guarded_implementation
)


def _reduced_load(
    T: sparse.csr_matrix,
    K: sparse.csr_matrix,
    u0: np.ndarray,
    full_load: np.ndarray,
) -> np.ndarray:
    return np.asarray(T.T @ (full_load - K @ u0), dtype=float).reshape(-1)


def _selected_stresses(
    model: "FEModel",
    displacement: np.ndarray,
    output_elements: Optional[Sequence[int]],
    recovery: Optional[RecoveryConfig],
) -> Dict[int, Dict[str, np.ndarray]]:
    from .recovery import RecoveryConfig, recover_element_stresses

    if recovery is not None:
        element_ids = output_elements if output_elements is not None else recovery.element_ids
        scoped = RecoveryConfig(
            node_ids=recovery.node_ids,
            element_ids=element_ids,
            components=recovery.components,
            include_displacements=recovery.include_displacements,
            include_stresses=recovery.include_stresses,
            include_reactions=recovery.include_reactions,
            history_mode=recovery.history_mode,
            store_full_histories=recovery.store_full_histories,
            metadata=recovery.metadata,
        )
    else:
        scoped = RecoveryConfig(element_ids=output_elements)
    return recover_element_stresses(model, displacement, scoped)


@resource_threaded
def _solve_transient_newmark_under_lease(
    model: "FEModel",
    config: TransientConfig,
    pressure_patches: Optional[Sequence[PressurePatch]] = None,
    base_load_case: Optional[LoadCase] = None,
    *,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    session: Optional["AnalysisSession"] = None,
    _owned_integration_parameters: Optional[
        Tuple[float, float, float]
    ] = None,
    _start_cancellation_checked: bool = False,
    _qualified_runtime_guard: Any = None,
) -> TransientResult:
    """Solve linear transient response with Newmark time integration.

    The equation advanced in reduced coordinates is:

        M qdd + C qd + K q = F(t)

    where fixed DOFs and MPCs are eliminated by the same transformation used in
    the static solver.  ``C`` is Rayleigh damping
    ``alpha * M + beta * K``.
    """
    raw_reference_authority_guard = (
        _require_qualified_reference_transient_authority
    )

    def reference_authority_guard(
        observed_model: "FEModel",
    ) -> Dict[str, Any]:
        result = raw_reference_authority_guard(observed_model)
        _qualified_runtime_guard(
            observed_model,
            context="solve_transient_newmark runtime authority",
        )
        return result
    if not _start_cancellation_checked:
        cancellation_safe_point(cancellation_token, "transient.start")
        reference_authority_guard(model)
    if _owned_integration_parameters is None:
        config, integration_parameters, patches = (
            _owned_transient_configuration(
                model,
                config,
                pressure_patches,
                reference_authority_guard,
            )
        )
        reference_authority_guard(model)
    else:
        integration_parameters = _owned_integration_parameters
        patches = () if pressure_patches is None else tuple(pressure_patches)
    if session is not None:
        # Ownership and liveness must be established before boundary-condition
        # application, load evaluation, or any matrix/cache access.
        session._require_model(model)
        reference_authority_guard(model)
    qualified_reference_authority = reference_authority_guard(model)
    require_model_element_capabilities(
        model,
        "transient_algebraic_dynamics",
        context="linear transient analysis",
    )
    model.apply_boundary_conditions()
    reference_authority_guard(model)
    total_dofs = model.mesh.dof_manager.total_dofs
    base_load, base_load_info = assemble_load_vector(model, base_load_case)
    reference_authority_guard(model)
    if session is None:
        K, stiffness_info = assemble_stiffness_matrix(model)
        reference_authority_guard(model)
        M, mass_info = assemble_mass_matrix(model)
        reference_authority_guard(model)
        zero_load = np.zeros(total_dofs, dtype=float)
        K_red, _zero_red, T, u0, independent_dofs, constraint_info = (
            build_constraint_transformation(K, zero_load, model)
        )
        reference_authority_guard(model)
        M_red = (T.T @ M @ T).tocsr()
        constraint_plan = None
    else:
        stiffness_plan = session.stiffness_plan(model)
        reference_authority_guard(model)
        constraint_plan = session.constraint_plan(stiffness_plan, model)
        reference_authority_guard(model)
        mass_plan = session.mass_plan(model)
        reference_authority_guard(model)
        K = stiffness_plan.matrix
        M = mass_plan.matrix
        stiffness_info = dict(stiffness_plan.info)
        mass_info = dict(mass_plan.info)
        K_red = constraint_plan.K_red
        M_red, _ = session.reduced_mass(constraint_plan, model)
        reference_authority_guard(model)
        T = constraint_plan.T
        u0 = constraint_plan.u0
        independent_dofs = constraint_plan.independent_dofs
        constraint_info = dict(constraint_plan.info)
    descriptor_elements = declared_algebraic_mass_elements(model)
    descriptor_guarded = bool(descriptor_elements)
    descriptor_reduction: Optional[AlgebraicStaticReduction] = None
    descriptor_certificate: Optional[Dict[str, Any]] = None
    descriptor_formulations = [
        {
            "element_id": int(element_id),
            "formulation_id": str(
                _static_formulation_id(model.mesh.elements[element_id]) or ""
            ),
            "algebraic_coordinate_policy": str(
                getattr(
                    model.mesh.elements[element_id],
                    "dynamic_algebraic_policy",
                    "",
                )
            ),
        }
        for element_id in descriptor_elements
    ]
    if descriptor_elements:
        descriptor_basis = build_declared_algebraic_basis(
            model,
            M,
            M_red,
            T,
            independent_dofs,
            dense_size_limit=512,
        )
        if descriptor_basis.reduced_basis.shape[1]:
            descriptor_reduction = build_algebraic_static_reduction(
                K_red,
                M_red,
                descriptor_basis.reduced_basis,
                dense_size_limit=512,
            )
        descriptor_certificate = {"mass_basis": descriptor_basis.diagnostics}
        if descriptor_reduction is not None:
            descriptor_certificate["static_reduction"] = (
                descriptor_reduction.diagnostics
            )
            M_dynamic = descriptor_reduction.mass_physical
        else:
            M_dynamic = M_red
    else:
        M_dynamic = M_red
    C_red = (
        config.rayleigh_alpha * M_red + config.rayleigh_beta * K_red
    ).tocsr()
    if descriptor_guarded:
        rayleigh_coefficients = np.asarray(
            (config.rayleigh_alpha, config.rayleigh_beta),
            dtype=float,
        )
        if np.any(~np.isfinite(rayleigh_coefficients)) or np.any(
            rayleigh_coefficients < 0.0
        ):
            raise AlgebraicDynamicsError(
                "descriptor transient analysis requires finite non-negative "
                "Rayleigh damping"
            )

    if float(np.linalg.norm(M_dynamic.diagonal())) <= 0.0 and M_dynamic.nnz == 0:
        raise ValueError("Transient analysis requires a non-zero mass matrix; set material density values.")

    patch_vectors = []
    patch_infos = []
    for patch in patches:
        vector, info = _ASSEMBLE_PRESSURE_PATCH_LOAD_VECTOR_GUARDED(
            model,
            patch,
            1.0,
            reference_authority_guard,
        )
        reference_authority_guard(model)
        patch_vectors.append(vector)
        patch_infos.append(info)
    base_load_red = _reduced_load(T, K, u0, base_load)
    patch_vectors_red = [
        np.asarray(T.T @ vector, dtype=float).reshape(-1)
        for vector in patch_vectors
    ]
    times = _time_grid(config)
    time_index = {
        float(value).hex(): index for index, value in enumerate(times)
    }
    # Observe application callbacks into plain Python-owned rows first.  The
    # captured guard must run before NumPy conversion or finite-value checks,
    # because either operation is part of the exact numerical authority.
    pressure_rows = []
    for patch in patches:
        row = []
        for value in times:
            row.append(
                _guarded_pressure_value(
                    model,
                    patch,
                    float(value),
                    reference_authority_guard,
                )
            )
        pressure_rows.append(row)
    pressure_samples = np.asarray(pressure_rows, dtype=float).reshape(
        len(patches), len(times)
    )
    if pressure_samples.size and not np.all(np.isfinite(pressure_samples)):
        if descriptor_guarded:
            raise AlgebraicDynamicsError(
                "descriptor transient full load is non-finite"
            )
        raise ValueError("pressure history must contain only finite values")
    def pressure_values_at(time: float) -> np.ndarray:
        index = time_index.get(float(time).hex())
        if index is None:
            raise ValueError("transient load requested outside the owned time grid")
        return pressure_samples[:, index]

    def full_load_at(time: float) -> np.ndarray:
        load = base_load.copy()
        for pressure, vector in zip(pressure_values_at(time), patch_vectors):
            load += float(pressure) * vector
        if descriptor_guarded and np.any(~np.isfinite(load)):
            raise AlgebraicDynamicsError(
                "descriptor transient full load is non-finite"
            )
        return load

    def reduced_load_at(time: float) -> np.ndarray:
        load = base_load_red.copy()
        for pressure, vector in zip(pressure_values_at(time), patch_vectors_red):
            load += float(pressure) * vector
        if descriptor_guarded and np.any(~np.isfinite(load)):
            raise AlgebraicDynamicsError(
                "descriptor transient reduced load is non-finite"
            )
        return load

    observed_recovery = config.recovery
    reference_authority_guard(model)
    if observed_recovery is None:
        recovery = None
    else:
        observed_recovery_node_ids = observed_recovery.node_ids
        reference_authority_guard(model)
        recovery_node_ids = (
            None
            if observed_recovery_node_ids is None
            else tuple(int(node_id) for node_id in observed_recovery_node_ids)
        )
        reference_authority_guard(model)
        observed_recovery_element_ids = observed_recovery.element_ids
        reference_authority_guard(model)
        recovery_element_ids = (
            None
            if observed_recovery_element_ids is None
            else tuple(
                int(element_id)
                for element_id in observed_recovery_element_ids
            )
        )
        reference_authority_guard(model)
        observed_recovery_components = observed_recovery.components
        reference_authority_guard(model)
        recovery_components = (
            None
            if observed_recovery_components is None
            else tuple(str(component) for component in observed_recovery_components)
        )
        reference_authority_guard(model)
        observed_recovery_metadata = observed_recovery.metadata
        reference_authority_guard(model)
        owned_recovery_metadata = dict(observed_recovery_metadata)
        reference_authority_guard(model)
        recovery_include_displacements = bool(
            observed_recovery.include_displacements
        )
        reference_authority_guard(model)
        recovery_include_stresses = bool(observed_recovery.include_stresses)
        reference_authority_guard(model)
        recovery_include_reactions = bool(observed_recovery.include_reactions)
        reference_authority_guard(model)
        recovery_history_mode = str(observed_recovery.history_mode)
        reference_authority_guard(model)
        recovery_store_full_histories = bool(
            observed_recovery.store_full_histories
        )
        reference_authority_guard(model)
        recovery = RecoveryConfig(
            node_ids=recovery_node_ids,
            element_ids=recovery_element_ids,
            components=recovery_components,
            include_displacements=recovery_include_displacements,
            include_stresses=recovery_include_stresses,
            include_reactions=recovery_include_reactions,
            history_mode=recovery_history_mode,
            store_full_histories=recovery_store_full_histories,
            metadata=owned_recovery_metadata,
        )
        reference_authority_guard(model)
    configured_output_nodes = config.output_nodes
    reference_authority_guard(model)
    output_node_ids = (
        ()
        if configured_output_nodes is None
        else tuple(int(node_id) for node_id in configured_output_nodes)
    )
    reference_authority_guard(model)
    if recovery is not None and not output_node_ids and recovery.node_ids is not None:
        output_node_ids = tuple(int(node_id) for node_id in recovery.node_ids)
        reference_authority_guard(model)
    output_element_ids: Optional[Tuple[int, ...]]
    configured_output_elements = config.output_elements
    reference_authority_guard(model)
    if configured_output_elements is not None:
        output_element_ids = tuple(
            int(element_id) for element_id in configured_output_elements
        )
        reference_authority_guard(model)
    elif recovery is not None and recovery.element_ids is not None:
        output_element_ids = tuple(int(element_id) for element_id in recovery.element_ids)
        reference_authority_guard(model)
    else:
        output_element_ids = None
    include_stress_history = bool(config.include_stress_history)
    if recovery is not None:
        include_stress_history = include_stress_history and bool(recovery.include_stresses)
    history_storage_mode = "full" if recovery is None else str(recovery.history_mode)
    if recovery is not None and history_storage_mode == "full" and not recovery.store_full_histories:
        history_storage_mode = "selected"
    if history_storage_mode == "selected" and not output_node_ids and (recovery is None or recovery.include_displacements):
        output_node_ids = tuple(int(node_id) for node_id in model.mesh.nodes)
    history_dof_indices = _node_dof_indices(model, output_node_ids) if history_storage_mode == "selected" else None
    selected_output_plan = (
        session.output_selection_plan(history_dof_indices, constraint_plan, model)
        if session is not None and history_dof_indices is not None
        else None
    )
    if session is not None and history_dof_indices is not None:
        reference_authority_guard(model)
    selected_T = (
        T[np.asarray(history_dof_indices, dtype=np.intp)].tocsr()
        if selected_output_plan is None and history_dof_indices is not None
        else None
    )
    selected_u0 = (
        u0[np.asarray(history_dof_indices, dtype=np.intp)]
        if selected_output_plan is None and history_dof_indices is not None
        else None
    )
    node_history_slices = {
        int(node_id): slice(6 * index, 6 * (index + 1))
        for index, node_id in enumerate(output_node_ids)
    }
    peak_node_ids, peak_dof_index = _translation_peak_index(model)
    peak_flat_dofs = peak_dof_index.reshape(-1)
    peak_rows_required = not (
        history_storage_mode in {"full", "envelope"} or include_stress_history
    )
    peak_output_plan = (
        session.output_selection_plan(peak_flat_dofs, constraint_plan, model)
        if session is not None and peak_rows_required
        else None
    )
    if session is not None and peak_rows_required:
        reference_authority_guard(model)
    peak_T = (
        T[peak_flat_dofs].tocsr()
        if peak_output_plan is None and peak_rows_required
        else None
    )
    peak_u0 = (
        u0[peak_flat_dofs]
        if peak_output_plan is None and peak_rows_required
        else None
    )
    estimated_saved_steps = _saved_step_count(times, config.save_every)
    preflight_memory = estimate_model_memory(
        model,
        transient_saved_steps=estimated_saved_steps,
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(preflight_memory, config.resource_config, context="solve_transient_newmark")
    q = _full_initial_vector(
        config.initial_displacement,
        total_dofs,
        post_observation=lambda: reference_authority_guard(model),
    )
    v_full = _full_initial_vector(
        config.initial_velocity,
        total_dofs,
        post_observation=lambda: reference_authority_guard(model),
    )
    q_red = np.asarray((q - u0)[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    v_red = np.asarray(v_full[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    if descriptor_guarded and (
        np.any(~np.isfinite(q_red)) or np.any(~np.isfinite(v_red))
    ):
        raise AlgebraicDynamicsError(
            "descriptor transient initial state is non-finite"
        )

    F0_red = reduced_load_at(float(times[0]))
    algebraic_displacement: Optional[np.ndarray] = None
    algebraic_velocity: Optional[np.ndarray] = None
    algebraic_acceleration: Optional[np.ndarray] = None
    algebraic_load_equilibria: list[np.ndarray] = []
    descriptor_initial_dae_residual = 0.0
    descriptor_initial_algebraic_residual = 0.0
    descriptor_partition_displacement: Optional[np.ndarray] = None
    descriptor_partition_velocity: Optional[np.ndarray] = None
    descriptor_partition_acceleration: Optional[np.ndarray] = None
    if descriptor_reduction is not None:
        sampled_loads = [
            reduced_load_at(float(time)) for time in times[: min(3, len(times))]
        ]
        algebraic_load_equilibria = [
            descriptor_reduction.algebraic_load_equilibrium(load)
            for load in sampled_loads
        ]
        load_equilibrium = algebraic_load_equilibria[0]
        load_equilibrium_rate = _forward_grid_derivative(
            times,
            algebraic_load_equilibria,
            1,
        )
        load_equilibrium_acceleration = _forward_grid_derivative(
            times,
            algebraic_load_equilibria,
            2,
        )
        physical_displacement, initial_algebraic_displacement = (
            descriptor_reduction.split_k_orthogonal_state(q_red)
        )
        physical_velocity, _initial_algebraic_velocity = (
            descriptor_reduction.split_k_orthogonal_state(v_red)
        )
        damping_time = float(config.rayleigh_beta)
        if damping_time == 0.0:
            algebraic_displacement = load_equilibrium.copy()
            algebraic_velocity = load_equilibrium_rate.copy()
            algebraic_acceleration = load_equilibrium_acceleration.copy()
        else:
            algebraic_displacement = initial_algebraic_displacement.copy()
            algebraic_velocity = (
                load_equilibrium - algebraic_displacement
            ) / damping_time
            algebraic_acceleration = (
                load_equilibrium_rate - algebraic_velocity
            ) / damping_time
        q_red = descriptor_reduction.reconstruct_k_orthogonal_state(
            physical_displacement,
            algebraic_displacement,
        )
        v_red = descriptor_reduction.reconstruct_k_orthogonal_state(
            physical_velocity,
            algebraic_velocity,
        )
        condensed_force = descriptor_reduction.condensed_load(F0_red)
        stiffness_force = descriptor_reduction.stiffness_action(
            physical_displacement
        )
        damping_force = (
            float(config.rayleigh_alpha)
            * np.asarray(
                descriptor_reduction.mass_physical @ physical_velocity,
                dtype=float,
            ).reshape(-1)
            + float(config.rayleigh_beta)
            * descriptor_reduction.stiffness_action(physical_velocity).reshape(-1)
        )
        try:
            mass_handle = factorize(
                descriptor_reduction.mass_physical,
                MatrixClass.SPD,
                signature="transient.initial_descriptor_physical_mass",
            )
        except Exception as exc:
            raise AlgebraicDynamicsError(
                "descriptor physical mass factorization failed"
            ) from exc
        if mass_handle.status != "ok":
            raise AlgebraicDynamicsError(
                "descriptor physical mass factorization failed"
            )
        try:
            physical_acceleration = np.asarray(
                mass_handle.solve(
                    condensed_force - damping_force - stiffness_force
                ),
                dtype=float,
            ).reshape(-1)
        except Exception as exc:
            raise AlgebraicDynamicsError(
                "descriptor physical mass solve failed"
            ) from exc
        a_red = descriptor_reduction.reconstruct_k_orthogonal_state(
            physical_acceleration,
            algebraic_acceleration,
        )
        descriptor_partition_displacement = np.concatenate(
            (
                physical_displacement,
                descriptor_reduction.partition_coordinate_from_k_orthogonal(
                    physical_displacement,
                    algebraic_displacement,
                ),
            )
        )
        descriptor_partition_velocity = np.concatenate(
            (
                physical_velocity,
                descriptor_reduction.partition_coordinate_from_k_orthogonal(
                    physical_velocity,
                    algebraic_velocity,
                ),
            )
        )
        descriptor_partition_acceleration = np.concatenate(
            (
                physical_acceleration,
                descriptor_reduction.partition_coordinate_from_k_orthogonal(
                    physical_acceleration,
                    algebraic_acceleration,
                ),
            )
        )
        initial_residual = np.asarray(
            M_red @ a_red + C_red @ v_red + K_red @ q_red - F0_red,
            dtype=float,
        ).reshape(-1)
        initial_scale = np.asarray(
            abs(M_red) @ np.abs(a_red)
            + abs(C_red) @ np.abs(v_red)
            + abs(K_red) @ np.abs(q_red)
            + np.abs(F0_red),
            dtype=float,
        ).reshape(-1)
        initial_relative = float(
            np.linalg.norm(initial_residual)
            / max(np.linalg.norm(initial_scale), 1.0)
        )
        initial_algebraic = np.asarray(
            descriptor_reduction.basis.T
            @ (K_red @ q_red + C_red @ v_red - F0_red),
            dtype=float,
        ).reshape(-1)
        initial_algebraic_scale = np.asarray(
            abs(descriptor_reduction.basis.T)
            @ (
                abs(K_red) @ np.abs(q_red)
                + abs(C_red) @ np.abs(v_red)
                + np.abs(F0_red)
            ),
            dtype=float,
        ).reshape(-1)
        initial_algebraic_relative = float(
            np.linalg.norm(initial_algebraic)
            / max(np.linalg.norm(initial_algebraic_scale), 1.0)
        )
        descriptor_initial_dae_residual = initial_relative
        descriptor_initial_algebraic_residual = initial_algebraic_relative
        if (
            not np.isfinite(initial_relative)
            or not np.isfinite(initial_algebraic_relative)
            or initial_relative > 2.0e-9
            or initial_algebraic_relative > 2.0e-9
        ):
            raise AlgebraicDynamicsError(
                "descriptor initial state failed the DAE residual gate"
            )
    else:
        if descriptor_guarded:
            constrained_mass_certificate = certify_descriptor_effective_operator(
                M_red,
                dense_size_limit=512,
            )
            assert descriptor_certificate is not None
            descriptor_certificate["constrained_physical_mass"] = (
                constrained_mass_certificate
            )
            try:
                mass_handle = factorize(
                    M_red,
                    MatrixClass.SPD,
                    signature="transient.initial_descriptor_constrained_mass",
                )
            except Exception as exc:
                raise AlgebraicDynamicsError(
                    "descriptor constrained physical mass factorization failed"
                ) from exc
            if mass_handle.status != "ok":
                raise AlgebraicDynamicsError(
                    "descriptor constrained physical mass factorization failed"
                )
            try:
                a_red = np.asarray(
                    mass_handle.solve(F0_red - C_red @ v_red - K_red @ q_red),
                    dtype=float,
                ).reshape(-1)
            except Exception as exc:
                raise AlgebraicDynamicsError(
                    "descriptor constrained physical mass solve failed"
                ) from exc
            if a_red.shape != (M_red.shape[0],) or np.any(~np.isfinite(a_red)):
                raise AlgebraicDynamicsError(
                    "descriptor constrained physical mass solve returned an invalid result"
                )
        else:
            try:
                mass_handle = factorize(
                    M_red,
                    MatrixClass.SYMMETRIC_SEMIDEFINITE,
                    signature="transient.initial_mass",
                )
                a_red = np.asarray(
                    mass_handle.solve(F0_red - C_red @ v_red - K_red @ q_red),
                    dtype=float,
                ).reshape(-1)
            except Exception as exc:
                raise ValueError(
                    f"Could not compute initial acceleration: {exc}"
                ) from exc

    saved_times = []
    saved_u = []
    saved_v = []
    saved_a = []
    node_history_values: Dict[int, list[np.ndarray]] = {int(node_id): [] for node_id in output_node_ids}
    displacement_envelope: Optional[np.ndarray] = None
    velocity_envelope: Optional[np.ndarray] = None
    acceleration_envelope: Optional[np.ndarray] = None
    stress_history = [] if include_stress_history else None
    peak_displacement = 0.0
    peak_displacement_node = None
    peak_von_mises = 0.0
    energy_kinetic = []
    energy_strain = []
    descriptor_algebraic_energy: list[float] = []
    full_vector_reconstruction_count = 0
    selected_output_reconstruction_count = 0

    def save_state(time: float, q_state: np.ndarray, v_state: np.ndarray, a_state: np.ndarray) -> None:
        nonlocal peak_displacement, peak_displacement_node, peak_von_mises
        nonlocal displacement_envelope, velocity_envelope, acceleration_envelope
        nonlocal full_vector_reconstruction_count, selected_output_reconstruction_count
        needs_full = history_storage_mode in {"full", "envelope"} or stress_history is not None
        full_u: Optional[np.ndarray]
        full_v: Optional[np.ndarray]
        full_a: Optional[np.ndarray]
        if needs_full:
            full_u = reconstruct_full_solution(T, q_state, u0)
            full_v = np.asarray(T @ v_state, dtype=float).reshape(-1)
            full_a = np.asarray(T @ a_state, dtype=float).reshape(-1)
            full_vector_reconstruction_count += 1
        else:
            full_u = None
            full_v = None
            full_a = None
        if selected_output_plan is not None:
            selected_u = selected_output_plan.reconstruct(q_state)
            selected_v = selected_output_plan.reconstruct(v_state, affine=False)
            selected_a = selected_output_plan.reconstruct(a_state, affine=False)
            reference_authority_guard(model)
            selected_output_reconstruction_count += 1
        elif selected_T is not None:
            selected_u = np.asarray(selected_T @ q_state, dtype=float).reshape(-1) + np.asarray(selected_u0, dtype=float)
            selected_v = np.asarray(selected_T @ v_state, dtype=float).reshape(-1)
            selected_a = np.asarray(selected_T @ a_state, dtype=float).reshape(-1)
            selected_output_reconstruction_count += 1
        else:
            selected_u = np.zeros(0, dtype=float)
            selected_v = np.zeros(0, dtype=float)
            selected_a = np.zeros(0, dtype=float)
        saved_times.append(float(time))
        if history_storage_mode == "full":
            assert full_u is not None and full_v is not None and full_a is not None
            saved_u.append(full_u)
            saved_v.append(full_v)
            saved_a.append(full_a)
        elif history_storage_mode == "selected":
            saved_u.append(selected_u)
            saved_v.append(selected_v)
            saved_a.append(selected_a)
        elif history_storage_mode == "envelope":
            assert full_u is not None and full_v is not None and full_a is not None
            abs_u = np.abs(full_u)
            abs_v = np.abs(full_v)
            abs_a = np.abs(full_a)
            displacement_envelope = abs_u if displacement_envelope is None else np.maximum(displacement_envelope, abs_u)
            velocity_envelope = abs_v if velocity_envelope is None else np.maximum(velocity_envelope, abs_v)
            acceleration_envelope = abs_a if acceleration_envelope is None else np.maximum(acceleration_envelope, abs_a)
        for node_id in output_node_ids:
            if full_u is not None:
                node = model.mesh.get_node(int(node_id))
                if node is not None:
                    node_history_values[int(node_id)].append(full_u[np.asarray(node.dofs, dtype=np.intp)])
            else:
                node_history_values[int(node_id)].append(
                    selected_u[node_history_slices[int(node_id)]]
                )
        if full_u is not None:
            current_peak, current_node = _translation_peak(model, full_u)
        elif peak_node_ids.size:
            if peak_output_plan is not None:
                translations = peak_output_plan.reconstruct(q_state).reshape(-1, 3)
                reference_authority_guard(model)
            else:
                assert peak_T is not None and peak_u0 is not None
                translations = (
                    np.asarray(peak_T @ q_state, dtype=float).reshape(-1) + peak_u0
                ).reshape(-1, 3)
            magnitudes_sq = np.einsum("ij,ij->i", translations, translations)
            best = int(np.argmax(magnitudes_sq))
            current_peak = float(np.sqrt(magnitudes_sq[best]))
            current_node = int(peak_node_ids[best])
        else:
            current_peak, current_node = 0.0, None
        if current_peak > peak_displacement:
            peak_displacement = current_peak
            peak_displacement_node = current_node
        energy_kinetic.append(0.5 * float(v_state @ (M_red @ v_state)))
        energy_strain.append(0.5 * float(q_state @ (K_red @ q_state)))
        if descriptor_reduction is not None:
            _physical_state, algebraic_state = (
                descriptor_reduction.split_k_orthogonal_state(q_state)
            )
            descriptor_algebraic_energy.append(
                0.5
                * float(
                    algebraic_state
                    @ (
                        descriptor_reduction.stiffness_algebraic
                        @ algebraic_state
                    )
                )
            )
        if stress_history is not None:
            assert full_u is not None
            stresses = _selected_stresses(model, full_u, output_element_ids, recovery)
            reference_authority_guard(model)
            stress_history.append(stresses)
            for element_stresses in stresses.values():
                if "von_mises" in element_stresses:
                    peak_von_mises = max(
                        peak_von_mises,
                        float(np.max(np.abs(np.asarray(element_stresses["von_mises"], dtype=float)))),
                    )

    save_state(float(times[0]), q_red, v_red, a_red)
    emit_progress(
        progress_callback,
        "transient_step",
        "transient.integration",
        completed=0,
        total=max(len(times) - 1, 0),
        step_index=0,
        time_s=float(times[0]),
        saved=True,
    )
    if progress_callback is not None:
        reference_authority_guard(model)
    load_prev = full_load_at(float(times[0]))
    impulse = np.zeros(total_dofs, dtype=float)

    factorization_count = 0
    solve_count = 0
    factorization_reused = False
    cached_dt = None
    cached_solver = None
    cached_solver_diagnostics: Dict[str, Any] = {}

    alpha_h, beta, gamma = integration_parameters
    one_plus_alpha = 1.0 + alpha_h
    F_red_prev = F0_red
    descriptor_max_dae_residual = descriptor_initial_dae_residual
    descriptor_max_algebraic_residual = descriptor_initial_algebraic_residual
    descriptor_external_work = 0.0
    descriptor_algebraic_external_work = 0.0
    descriptor_algebraic_load_norm_max = 0.0
    descriptor_effective_certificates: list[Dict[str, Any]] = []
    descriptor_abs_mass = abs(M_red) if descriptor_reduction is not None else None
    descriptor_abs_damping = (
        abs(C_red) if descriptor_reduction is not None else None
    )
    descriptor_abs_stiffness = (
        abs(K_red) if descriptor_reduction is not None else None
    )
    descriptor_abs_basis_transpose = (
        abs(descriptor_reduction.basis.T)
        if descriptor_reduction is not None
        else None
    )
    descriptor_previous_coordinate = (
        descriptor_reduction.algebraic_partition_coordinate(q_red)
        if descriptor_reduction is not None
        else None
    )
    descriptor_previous_load = (
        np.asarray(
            descriptor_reduction.basis.T @ F0_red,
            dtype=float,
        ).reshape(-1)
        if descriptor_reduction is not None
        else None
    )
    if descriptor_previous_load is not None:
        descriptor_algebraic_load_norm_max = float(
            np.linalg.norm(descriptor_previous_load)
        )
    for step_index in range(1, len(times)):
        cancellation_safe_point(
            cancellation_token,
            f"transient.step:{step_index}",
        )
        reference_authority_guard(model)
        dt = float(times[step_index] - times[step_index - 1])
        if descriptor_guarded:
            final_step_index = len(times) - 1
            nominal_end = float(final_step_index) * float(config.dt)
            end_time = float(config.t_end)
            nominal_roundoff = 8.0 * max(
                math.ulp(end_time),
                math.ulp(nominal_end),
                float(final_step_index) * math.ulp(float(config.dt)),
            )
            nominal_uniform_end = (
                abs(end_time - nominal_end) <= nominal_roundoff
            )
            if step_index < final_step_index or nominal_uniform_end:
                dt = float(config.dt)
            else:
                dt = end_time - float(
                    final_step_index - 1
                ) * float(config.dt)
        if dt <= 0.0:
            continue
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        descriptor_dt_changed = (
            descriptor_guarded
            and cached_dt is not None
            and dt != cached_dt
        )
        ordinary_dt_changed = (
            not descriptor_guarded
            and cached_dt is not None
            and not np.isclose(dt, cached_dt)
        )
        if (
            cached_solver is None
            or cached_dt is None
            or descriptor_dt_changed
            or ordinary_dt_changed
        ):
            if descriptor_reduction is not None:
                stiffness_coefficient = one_plus_alpha * (
                    1.0 + a1 * float(config.rayleigh_beta)
                )
                mass_coefficient = a0 + one_plus_alpha * a1 * float(
                    config.rayleigh_alpha
                )
                K_eff = descriptor_reduction.effective_block_matrix(
                    stiffness_coefficient=stiffness_coefficient,
                    mass_coefficient=mass_coefficient,
                )
                effective_certificate = certify_descriptor_effective_operator(
                    K_eff,
                    dense_size_limit=512,
                )
                effective_certificate.update(
                    {
                        "dt": dt,
                        "stiffness_coefficient": stiffness_coefficient,
                        "mass_coefficient": mass_coefficient,
                    }
                )
                descriptor_effective_certificates.append(effective_certificate)
                factorization_signature = (
                    f"transient.effective.descriptor:{dt:.16g}"
                )
            else:
                K_eff = (
                    one_plus_alpha * K_red
                    + a0 * M_red
                    + one_plus_alpha * a1 * C_red
                ).tocsr()
                if descriptor_guarded:
                    effective_certificate = certify_descriptor_effective_operator(
                        K_eff,
                        dense_size_limit=512,
                    )
                    effective_certificate.update(
                        {
                            "dt": dt,
                            "stiffness_coefficient": one_plus_alpha
                            * (1.0 + a1 * float(config.rayleigh_beta)),
                            "mass_coefficient": a0
                            + one_plus_alpha
                            * a1
                            * float(config.rayleigh_alpha),
                        }
                    )
                    descriptor_effective_certificates.append(
                        effective_certificate
                    )
                    factorization_signature = (
                        f"transient.effective.descriptor_constrained:{dt:.16g}"
                    )
                else:
                    factorization_signature = f"transient.effective:{dt:.16g}"
            try:
                cached_solver = factorize(
                    K_eff,
                    (
                        MatrixClass.SPD
                        if descriptor_guarded
                        else MatrixClass.SYMMETRIC_INDEFINITE
                    ),
                    signature=factorization_signature,
                )
                if (
                    descriptor_guarded
                    and cached_solver.status != "ok"
                ):
                    raise AlgebraicDynamicsError(
                        "descriptor effective operator factorization failed"
                    )
                cached_solver_diagnostics = cached_solver.diagnostics()
                if descriptor_guarded:
                    cached_solver_diagnostics["coordinate_system"] = (
                        "CERTIFIED_PHYSICAL_ALGEBRAIC_PARTITION"
                        if descriptor_reduction is not None
                        else "CONSTRAINT_ELIMINATED_ALGEBRAIC_COORDINATES"
                    )
            except Exception as exc:
                if descriptor_guarded:
                    raise AlgebraicDynamicsError(
                        "descriptor effective operator failed its SPD factorization"
                    ) from exc
                cached_solver = None
                cached_solver_diagnostics = {"failure_reason": "effective_stiffness_factorization_failed"}
            cached_dt = dt
            factorization_count += 1

        load_next = full_load_at(float(times[step_index]))
        impulse += 0.5 * (load_prev + load_next) * dt
        load_prev = load_next
        F_red = reduced_load_at(float(times[step_index]))
        rhs = (
            one_plus_alpha * F_red
            - alpha_h * F_red_prev
            + M_red @ (a0 * q_red + a2 * v_red + a3 * a_red)
            + one_plus_alpha * (C_red @ (a1 * q_red + a4 * v_red + a5 * a_red))
        )
        if alpha_h != 0.0:
            rhs += alpha_h * (K_red @ q_red) + alpha_h * (C_red @ v_red)
        descriptor_partition_next: Optional[np.ndarray] = None
        if cached_solver is not None:
            effective_rhs = (
                descriptor_reduction.effective_rhs(rhs)
                if descriptor_reduction is not None
                else rhs
            )
            try:
                effective_solution = np.asarray(
                    cached_solver.solve(effective_rhs), dtype=float
                ).reshape(-1)
            except Exception as exc:
                if descriptor_guarded:
                    raise AlgebraicDynamicsError(
                        "descriptor effective operator solve failed"
                    ) from exc
                raise
            if descriptor_guarded and (
                effective_solution.shape != (K_eff.shape[0],)
                or np.any(~np.isfinite(effective_solution))
            ):
                raise AlgebraicDynamicsError(
                    "descriptor effective operator solve returned an invalid result"
                )
            if descriptor_reduction is not None:
                descriptor_partition_next = effective_solution.copy()
                q_next = descriptor_reduction.reconstruct_partition_state(
                    effective_solution[: descriptor_reduction.physical_dimension],
                    effective_solution[descriptor_reduction.physical_dimension :],
                )
            else:
                q_next = effective_solution
            factorization_reused = True
        else:
            if descriptor_guarded:
                raise AlgebraicDynamicsError(
                    "descriptor effective operator has no certified solver"
                )
            fallback_handle = factorize(
                (one_plus_alpha * K_red + a0 * M_red + one_plus_alpha * a1 * C_red).tocsr(),
                MatrixClass.GENERAL,
                signature=f"transient.effective.fallback:{dt:.16g}:{step_index}",
            )
            q_next = np.asarray(fallback_handle.solve(rhs), dtype=float).reshape(-1)
            cached_solver_diagnostics = fallback_handle.diagnostics()
        solve_count += 1
        if descriptor_reduction is not None:
            assert descriptor_partition_next is not None
            assert descriptor_partition_displacement is not None
            assert descriptor_partition_velocity is not None
            assert descriptor_partition_acceleration is not None
            damping_time = float(config.rayleigh_beta)
            if damping_time == 0.0:
                current_equilibrium = (
                    descriptor_reduction.algebraic_load_equilibrium(F_red)
                )
                physical_q = descriptor_partition_next[
                    : descriptor_reduction.physical_dimension
                ]
                descriptor_partition_next[
                    descriptor_reduction.physical_dimension :
                ] = descriptor_reduction.partition_coordinate_from_k_orthogonal(
                    physical_q,
                    current_equilibrium,
                )
            descriptor_partition_a_next = (
                a0
                * (
                    descriptor_partition_next
                    - descriptor_partition_displacement
                )
                - a2 * descriptor_partition_velocity
                - a3 * descriptor_partition_acceleration
            )
            descriptor_partition_v_next = (
                descriptor_partition_velocity
                + dt
                * (
                    (1.0 - gamma) * descriptor_partition_acceleration
                    + gamma * descriptor_partition_a_next
                )
            )
            q_next = descriptor_reduction.reconstruct_partition_state(
                descriptor_partition_next[
                    : descriptor_reduction.physical_dimension
                ],
                descriptor_partition_next[
                    descriptor_reduction.physical_dimension :
                ],
            )
            v_next = descriptor_reduction.reconstruct_partition_state(
                descriptor_partition_v_next[
                    : descriptor_reduction.physical_dimension
                ],
                descriptor_partition_v_next[
                    descriptor_reduction.physical_dimension :
                ],
            )
            a_next = descriptor_reduction.reconstruct_partition_state(
                descriptor_partition_a_next[
                    : descriptor_reduction.physical_dimension
                ],
                descriptor_partition_a_next[
                    descriptor_reduction.physical_dimension :
                ],
            )
            descriptor_partition_displacement = descriptor_partition_next
            descriptor_partition_velocity = descriptor_partition_v_next
            descriptor_partition_acceleration = descriptor_partition_a_next
        else:
            a_next = a0 * (q_next - q_red) - a2 * v_red - a3 * a_red
            v_next = v_red + dt * (
                (1.0 - gamma) * a_red + gamma * a_next
            )
        if descriptor_reduction is not None:
            current_static_residual = np.asarray(
                C_red @ v_next + K_red @ q_next - F_red,
                dtype=float,
            ).reshape(-1)
            previous_static_residual = np.asarray(
                C_red @ v_red + K_red @ q_red - F_red_prev,
                dtype=float,
            ).reshape(-1)
            dae_residual = np.asarray(M_red @ a_next, dtype=float).reshape(-1)
            dae_residual += one_plus_alpha * current_static_residual
            dae_residual -= alpha_h * previous_static_residual
            assert descriptor_abs_mass is not None
            assert descriptor_abs_damping is not None
            assert descriptor_abs_stiffness is not None
            assert descriptor_abs_basis_transpose is not None
            dae_scale = np.asarray(
                descriptor_abs_mass @ np.abs(a_next)
                + abs(one_plus_alpha)
                * (
                    descriptor_abs_damping @ np.abs(v_next)
                    + descriptor_abs_stiffness @ np.abs(q_next)
                    + np.abs(F_red)
                )
                + abs(alpha_h)
                * (
                    descriptor_abs_damping @ np.abs(v_red)
                    + descriptor_abs_stiffness @ np.abs(q_red)
                    + np.abs(F_red_prev)
                ),
                dtype=float,
            ).reshape(-1)
            dae_relative = float(
                np.linalg.norm(dae_residual)
                / max(np.linalg.norm(dae_scale), 1.0)
            )
            algebraic_residual = np.asarray(
                descriptor_reduction.basis.T
                @ (K_red @ q_next + C_red @ v_next - F_red),
                dtype=float,
            ).reshape(-1)
            algebraic_scale = np.asarray(
                descriptor_abs_basis_transpose
                @ (
                    descriptor_abs_stiffness @ np.abs(q_next)
                    + descriptor_abs_damping @ np.abs(v_next)
                    + np.abs(F_red)
                ),
                dtype=float,
            ).reshape(-1)
            algebraic_relative = float(
                np.linalg.norm(algebraic_residual)
                / max(np.linalg.norm(algebraic_scale), 1.0)
            )
            descriptor_max_dae_residual = max(
                descriptor_max_dae_residual, dae_relative
            )
            descriptor_max_algebraic_residual = max(
                descriptor_max_algebraic_residual, algebraic_relative
            )
            descriptor_external_work += 0.5 * float(
                (F_red_prev + F_red) @ (q_next - q_red)
            )
            algebraic_coordinate = (
                descriptor_reduction.algebraic_partition_coordinate(q_next)
            )
            algebraic_load = np.asarray(
                descriptor_reduction.basis.T @ F_red,
                dtype=float,
            ).reshape(-1)
            assert descriptor_previous_coordinate is not None
            assert descriptor_previous_load is not None
            descriptor_algebraic_external_work += 0.5 * float(
                (descriptor_previous_load + algebraic_load)
                @ (algebraic_coordinate - descriptor_previous_coordinate)
            )
            descriptor_previous_coordinate = algebraic_coordinate
            descriptor_previous_load = algebraic_load
            descriptor_algebraic_load_norm_max = max(
                descriptor_algebraic_load_norm_max,
                float(np.linalg.norm(algebraic_load)),
            )
            if (
                not np.isfinite(dae_relative)
                or not np.isfinite(algebraic_relative)
                or dae_relative > 2.0e-8
                or algebraic_relative > 2.0e-9
            ):
                raise AlgebraicDynamicsError(
                    "descriptor transient state failed the DAE residual gate"
                )
        q_red, v_red, a_red = q_next, v_next, a_next
        F_red_prev = F_red

        if step_index % int(config.save_every) == 0 or step_index == len(times) - 1:
            save_state(float(times[step_index]), q_red, v_red, a_red)
        emit_progress(
            progress_callback,
            "transient_step",
            "transient.integration",
            completed=step_index,
            total=max(len(times) - 1, 0),
            step_index=int(step_index),
            time_s=float(times[step_index]),
            saved=bool(step_index % int(config.save_every) == 0 or step_index == len(times) - 1),
        )
        if progress_callback is not None:
            reference_authority_guard(model)

    impulse_resultant = load_vector_resultant(model, impulse)
    total_energy = np.asarray(energy_kinetic, dtype=float) + np.asarray(energy_strain, dtype=float)
    nonzero_energy = total_energy[np.abs(total_energy) > 1.0e-30]
    if nonzero_energy.size:
        energy_drift = float((np.max(nonzero_energy) - np.min(nonzero_energy)) / max(abs(nonzero_energy[0]), 1.0e-30))
    else:
        energy_drift = 0.0

    history_width = total_dofs if history_storage_mode == "full" else int(0 if history_dof_indices is None else len(history_dof_indices))
    if history_storage_mode == "envelope":
        history_width = 0
    saved_u_array = np.vstack(saved_u) if saved_u else np.zeros((0, history_width), dtype=float)
    saved_v_array = np.vstack(saved_v) if saved_v else np.zeros((0, history_width), dtype=float)
    saved_a_array = np.vstack(saved_a) if saved_a else np.zeros((0, history_width), dtype=float)
    node_histories: Dict[int, np.ndarray] = {}
    for node_id in output_node_ids:
        node = model.mesh.get_node(int(node_id))
        if node is None:
            raise ValueError(f"Configured output node {node_id} not found")
        values = node_history_values.get(int(node_id), [])
        node_histories[int(node_id)] = np.vstack(values) if values else np.zeros((0, 6), dtype=float)

    recovery_memory = estimate_model_memory(
        model,
        transient_saved_steps=len(saved_times),
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(recovery_memory, config.resource_config, context="solve_transient_newmark.recovery")
    policy_metadata = recovery_metadata(recovery, config.resource_config, recovery_memory)

    diagnostics: Dict[str, Any] = {
        "method": "newmark" if alpha_h == 0.0 else "hht_alpha",
        "hht_alpha": alpha_h,
        "beta": beta,
        "gamma": gamma,
        "rayleigh_alpha": float(config.rayleigh_alpha),
        "rayleigh_beta": float(config.rayleigh_beta),
        "num_steps": max(int(len(times) - 1), 0),
        "num_saved_steps": len(saved_times),
        "num_reduced_dofs": int(K_red.shape[0]),
        "factorization_count": int(factorization_count),
        "factorization_reused": bool(factorization_reused and factorization_count <= max(solve_count, 1)),
        "solve_count": int(solve_count),
        "initial_mass_factorization": mass_handle.diagnostics(),
        "effective_stiffness_factorization": cached_solver_diagnostics,
        "constraint_info": constraint_info,
        "stiffness": stiffness_info,
        "mass": mass_info,
        "base_load": base_load_info,
        "pressure_patches": patch_infos,
        "preprojected_load_basis_count": int(1 + len(patch_vectors_red)),
        "full_vector_reconstruction_count": int(full_vector_reconstruction_count),
        "selected_output_reconstruction_count": int(selected_output_reconstruction_count),
        "output_nodes": [int(node_id) for node_id in output_node_ids],
        "output_elements": [] if output_element_ids is None else [int(element_id) for element_id in output_element_ids],
        "history_storage_mode": history_storage_mode,
        "history_dof_indices": None if history_dof_indices is None else [int(dof) for dof in history_dof_indices],
        "session_output_plans_active": bool(
            selected_output_plan is not None or peak_output_plan is not None
        ),
        "recovery_policy": policy_metadata,
        "kinetic_energy": energy_kinetic,
        "strain_energy": energy_strain,
        "max_relative_energy_drift": energy_drift,
        "constraint_postcheck": constraint_residual_summary(
            model,
            np.asarray(T @ q_red + u0, dtype=float).reshape(-1),
        ),
        "qualified_reference_transient_authority": (
            qualified_reference_authority
        ),
    }
    descriptor_transient_provenance: Optional[Dict[str, Any]] = None
    if descriptor_elements:
        if descriptor_reduction is not None:
            algebraic_endpoint_policy = (
                DESCRIPTOR_TRANSIENT_STATIC_POLICY_ID
                if float(config.rayleigh_beta) == 0.0
                else DESCRIPTOR_TRANSIENT_FIRST_ORDER_POLICY_ID
            )
            descriptor_disposition = "ACTIVE_COMPATIBLE_ALGEBRAIC_REDUCTION"
        else:
            algebraic_endpoint_policy = (
                DESCRIPTOR_TRANSIENT_CONSTRAINED_POLICY_ID
            )
            descriptor_disposition = (
                "COMPATIBLE_ALGEBRAIC_NULLITY_ZERO_ORDINARY_REDUCED_SYSTEM"
            )
        assert descriptor_certificate is not None
        descriptor_transient_provenance = {
            "policy_id": DESCRIPTOR_TRANSIENT_POLICY_ID,
            "algebraic_endpoint_policy": algebraic_endpoint_policy,
            "disposition": descriptor_disposition,
            "declared_algebraic_element_ids": [
                int(element_id) for element_id in descriptor_elements
            ],
            "declared_algebraic_formulations": descriptor_formulations,
        }
    if descriptor_reduction is not None:
        assert descriptor_transient_provenance is not None
        assert descriptor_certificate is not None
        diagnostics.update(
            {
                "descriptor_transient": True,
                **descriptor_transient_provenance,
                "declared_algebraic_mass_certificate": descriptor_certificate,
                "initial_dae_relative_residual": float(
                    descriptor_initial_dae_residual
                ),
                "initial_algebraic_relative_residual": float(
                    descriptor_initial_algebraic_residual
                ),
                "max_dae_relative_residual": float(
                    descriptor_max_dae_residual
                ),
                "max_algebraic_relative_residual": float(
                    descriptor_max_algebraic_residual
                ),
                "algebraic_external_work_after_initial_consistency": float(
                    descriptor_algebraic_external_work
                ),
                "total_external_work_after_initial_consistency": float(
                    descriptor_external_work
                ),
                "algebraic_strain_energy": descriptor_algebraic_energy,
                "max_algebraic_generalized_load_norm": float(
                    descriptor_algebraic_load_norm_max
                ),
                "effective_operator_certificates": (
                    descriptor_effective_certificates
                ),
                "load_impulse_policy": "FULL_UNPROJECTED_LOAD_VECTOR_TRAPEZOIDAL_V1",
                "reaction_recovery_policy": (
                    "FULL_LOAD_RETAINED_DYNAMIC_REACTION_HISTORY_NOT_EXPOSED"
                ),
            }
        )
    elif descriptor_transient_provenance is not None:
        assert descriptor_certificate is not None
        diagnostics.update(
            {
                "descriptor_transient": False,
                **descriptor_transient_provenance,
                "declared_algebraic_mass_certificate": descriptor_certificate,
                "effective_operator_certificates": (
                    descriptor_effective_certificates
                ),
            }
        )
    if session is not None:
        diagnostics["analysis_session"] = session.diagnostics()
        reference_authority_guard(model)
    assembly_info = {
        "stiffness": stiffness_info,
        "mass": mass_info,
        "load": base_load_info,
    }
    result_metadata: Dict[str, Any] = {
        "pressure_patches": [info.get("patch_name") for info in patch_infos],
        "resources": policy_metadata.get("resources"),
        "memory_estimate": policy_metadata.get("memory_estimate"),
    }
    if descriptor_transient_provenance is not None:
        result_metadata["descriptor_transient_provenance"] = (
            descriptor_transient_provenance
        )
    result_case = make_result_case(
        name="linear_transient_newmark",
        analysis_type="linear_transient",
        load_cases=() if base_load_case is None else (base_load_case,),
        assembly_info=assembly_info,
        solver_info={"backend": cached_solver_diagnostics, "convergence_info": {"status": "completed"}},
        recovery={
            "displacement_history": True,
            "velocity_history": True,
            "acceleration_history": True,
            "history_storage_mode": history_storage_mode,
            "history_dof_indices": None if history_dof_indices is None else [int(dof) for dof in history_dof_indices],
            "stress_history": include_stress_history,
            "output_nodes": [int(node_id) for node_id in output_node_ids],
            "output_elements": None if output_element_ids is None else [int(element_id) for element_id in output_element_ids],
            **policy_metadata["recovery"],
        },
        settings={
            "dt": config.dt,
            "t_end": config.t_end,
            "beta": beta,
            "gamma": gamma,
            "hht_alpha": alpha_h,
            "rayleigh_alpha": config.rayleigh_alpha,
            "rayleigh_beta": config.rayleigh_beta,
            "save_every": config.save_every,
        },
        metadata=result_metadata,
    ).to_dict()
    diagnostics["result_case"] = result_case
    reference_authority_guard(model)
    return TransientResult(
        times=np.asarray(saved_times, dtype=float),
        displacements=saved_u_array,
        velocities=saved_v_array,
        accelerations=saved_a_array,
        node_histories=node_histories,
        load_impulse=impulse,
        force_impulse=impulse_resultant.force,
        moment_impulse=impulse_resultant.moment,
        peak_displacement=float(peak_displacement),
        peak_displacement_node=peak_displacement_node,
        peak_von_mises_stress=float(peak_von_mises),
        stress_history=None if stress_history is None else tuple(stress_history),
        status="completed",
        diagnostics=diagnostics,
        result_case=result_case,
        history_storage_mode=history_storage_mode,
        history_dof_indices=None if history_dof_indices is None else np.asarray(history_dof_indices, dtype=np.intp),
        displacement_envelope=displacement_envelope,
        velocity_envelope=velocity_envelope,
        acceleration_envelope=acceleration_envelope,
    )


def solve_transient_newmark(
    model: "FEModel",
    config: TransientConfig,
    pressure_patches: Optional[Sequence[PressurePatch]] = None,
    base_load_case: Optional[LoadCase] = None,
    *,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    session: Optional["AnalysisSession"] = None,
) -> TransientResult:
    """Solve Newmark dynamics under one qualified-family operation lease."""

    raw_authority_guard = _require_qualified_reference_transient_authority
    raw_authority_guard(model)

    def operation(lease: Any) -> TransientResult:
        def combined_guard(observed_model: "FEModel") -> None:
            raw_authority_guard(observed_model)
            lease(
                observed_model,
                context="solve_transient_newmark configuration observation",
            )

        cancellation_safe_point(cancellation_token, "transient.start")
        combined_guard(model)
        owned_config, integration_parameters, owned_patches = (
            _owned_transient_configuration(
                model,
                config,
                pressure_patches,
                combined_guard,
            )
        )
        return _solve_transient_newmark_under_lease(
            model,
            owned_config,
            pressure_patches=owned_patches,
            base_load_case=base_load_case,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
            session=session,
            _owned_integration_parameters=integration_parameters,
            _start_cancellation_checked=True,
            _qualified_runtime_guard=lease,
        )

    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="solve_transient_newmark",
        operation=operation,
    )
