"""Sparse/dense free-vibration modal analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
from json.encoder import encode_basestring as _JSON_ENCODE_BASESTRING
import math
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from .analysis_session import AnalysisSession
from .assembly import build_constraint_transformation, build_reduced_rigid_body_modes
from .algebraic_dynamics import (
    AlgebraicDynamicsError,
    DESCRIPTOR_MODAL_POLICY_ID,
    build_declared_algebraic_basis,
    declared_algebraic_mass_elements,
    solve_descriptor_spectrum,
)
from .cases import make_result_case
from .constraint_audit import constraint_residual_summary
from .control import CancellationToken, ProgressCallback, cancellation_safe_point, emit_progress
from .current_state_tangent import (
    _assemble_committed_current_tangent_components_implementation as _EXACT_CURRENT_STATE_COMPONENT_IMPLEMENTATION,
    _snapshot_committed_current_tangent_inputs as _EXACT_CURRENT_STATE_INPUT_SNAPSHOT,
    _validate_committed_current_tangent_inputs_implementation as _EXACT_CURRENT_STATE_INPUT_VALIDATOR,
    require_active_current_state_eigen_lifecycle as _EXACT_ACTIVE_CURRENT_STATE_LIFECYCLE_GUARD,
    require_committed_tangent_component_api as _EXACT_COMMITTED_TANGENT_ROUTE_GUARD,
    require_exact_qualified_component_lifecycle_api as _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
)
from .element_capabilities import require_model_element_capabilities
from .element_capabilities import ElementCapabilityError
from .e4_pl_element import (
    QualifiedE4PLShellElement,
    _QUALIFIED_Q4_BASE_LOCAL_FRAME_KERNEL,
    _QUALIFIED_Q4_BASE_MITC4_SHEAR_KERNEL,
    _QUALIFIED_Q4_BASE_SERIALIZATION_KERNEL,
    _QUALIFIED_Q4_BASE_STIFFNESS_KERNEL,
)
from .e4_pl_s3_element import (
    ALGEBRAIC_COORDINATE_POLICY_ID,
    REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
    QualifiedE4PLS3ShellElement,
)
from .e4_pl_s3_v2d_element import (
    DYNAMIC_ALGEBRAIC_MASS_WITNESS as S3_V2D_DYNAMIC_MASS_WITNESS,
    DYNAMIC_ALGEBRAIC_POLICY_ID as S3_V2D_DYNAMIC_POLICY_ID,
    NativeParityE4PLS3V2DShellElement,
)
from .elements import ShellElement
from .linalg import FactorizationCache, MatrixClass, cached_inverse_operator
from .matrix_assembly import (
    _run_with_qualified_assembly_runtime_lease,
    _assemble_geometric_stiffness_matrix_under_lease,
    _pack_owned_geometric_states,
    assemble_geometric_stiffness_matrix,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
)
from .recovery import ResourceConfig, _owned_resource_config_snapshot
from .threading_policy import resource_threaded, thread_policy_diagnostics

# Keep a private import-time identity for the genuine optional-token checkpoint.
# A replacement callback is an observation boundary even when its token is None.
_EXACT_CANCELLATION_SAFE_POINT = cancellation_safe_point
_EXACT_PUBLIC_GEOMETRIC_ASSEMBLER = assemble_geometric_stiffness_matrix

if TYPE_CHECKING:
    from .fe_core import FEModel


PRESTRESSED_MODAL_POLICY_ID = (
    "MATERIAL_TANGENT_MINUS_COMPRESSION_POSITIVE_GEOMETRIC_V1"
)
PRESTRESS_INPUT_SCHEMA_ID = "CANONICAL_COMPLETE_ELEMENT_PRESTRESS_MAP_V1"
QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID = (
    "EXACT_TRANSITIVE_Q4_S3_REFERENCE_OPERATOR_GUARD_BEFORE_MECHANICS_V1"
)
CURRENT_STATE_MODAL_POLICY_ID = (
    "COMMITTED_Q4_S3_EXACT_TOTAL_TANGENT_WITH_REFERENCE_CONSISTENT_MASS_V2"
)
_ANALYSIS_SESSION_METHOD_AUTHORITY = {
    name: vars(AnalysisSession)[name]
    for name in (
        "_require_model",
        "constraint_plan",
        "diagnostics",
        "mass_plan",
        "reduced_mass",
        "rigid_body_modes",
        "stiffness_plan",
    )
}
_FACTORIZATION_CACHE_METHOD_AUTHORITY = {
    name: vars(FactorizationCache)[name]
    for name in (
        "clear",
        "diagnostics",
        "factorize",
        "key",
        "linear_operator",
        "set_backend_if_absent",
    )
}
_CURRENT_MODAL_MASS_APIS = {
    "qualified_q4": {
        "compute_mass_matrix": QualifiedE4PLShellElement.compute_mass_matrix,
        "get_node_coordinates": QualifiedE4PLShellElement.get_node_coordinates,
        "compute_shape_functions": ShellElement.compute_shape_functions,
        "gauss_points": ShellElement.gauss_points,
        "gauss_weights": ShellElement.gauss_weights,
        "shear_gauss_points": ShellElement.shear_gauss_points,
        "shear_gauss_weights": ShellElement.shear_gauss_weights,
        "_local_frame_and_derivatives": QualifiedE4PLShellElement._local_frame_and_derivatives,
        "_local_dof_transform": ShellElement._local_dof_transform,
        "validate_quadrature_authority": QualifiedE4PLShellElement.validate_quadrature_authority,
    },
    "qualified_s3": {
        "compute_mass_matrix": QualifiedE4PLS3ShellElement.compute_mass_matrix,
        "get_node_coordinates": QualifiedE4PLS3ShellElement.get_node_coordinates,
        "compute_mass_components": QualifiedE4PLS3ShellElement.compute_mass_components,
        "_compute_mass_components": QualifiedE4PLS3ShellElement._compute_mass_components,
        "_compute_stiffness_components": QualifiedE4PLS3ShellElement._compute_stiffness_components,
        "dynamic_algebraic_directions": QualifiedE4PLS3ShellElement.dynamic_algebraic_directions,
        "gauss_points": QualifiedE4PLS3ShellElement.gauss_points,
        "gauss_weights": QualifiedE4PLS3ShellElement.gauss_weights,
        "shear_gauss_points": QualifiedE4PLS3ShellElement.shear_gauss_points,
        "shear_gauss_weights": QualifiedE4PLS3ShellElement.shear_gauss_weights,
        "validate_quadrature_authority": QualifiedE4PLS3ShellElement.validate_quadrature_authority,
    },
    "qualified_s3_v2d": {
        "compute_mass_matrix": NativeParityE4PLS3V2DShellElement.compute_mass_matrix,
        "get_node_coordinates": ShellElement.get_node_coordinates,
        "compute_mass_components": NativeParityE4PLS3V2DShellElement.compute_mass_components,
        "dynamic_algebraic_directions": NativeParityE4PLS3V2DShellElement.dynamic_algebraic_directions,
        "gauss_points": NativeParityE4PLS3V2DShellElement.gauss_points,
        "gauss_weights": NativeParityE4PLS3V2DShellElement.gauss_weights,
        "shear_gauss_points": NativeParityE4PLS3V2DShellElement.shear_gauss_points,
        "shear_gauss_weights": NativeParityE4PLS3V2DShellElement.shear_gauss_weights,
        "validate_quadrature_authority": NativeParityE4PLS3V2DShellElement.validate_quadrature_authority,
    },
}
_QUALIFIED_PRESTRESS_OPERATOR_APIS = {
    "qualified_q4": {
        "compute_stiffness_matrix": QualifiedE4PLShellElement.compute_stiffness_matrix,
        "compute_stiffness_components": QualifiedE4PLShellElement.compute_stiffness_components,
        "_constitutive_and_drill_stiffness": QualifiedE4PLShellElement._constitutive_and_drill_stiffness,
        "_qualified_stiffness_cache_key": QualifiedE4PLShellElement._qualified_stiffness_cache_key,
        "_bind_qualified_component_guard": QualifiedE4PLShellElement._bind_qualified_component_guard,
        "_adopt_qualified_components": QualifiedE4PLShellElement._adopt_qualified_components,
        "_warped_generalized_drilling_correction": QualifiedE4PLShellElement._warped_generalized_drilling_correction,
        "_generalized_section_in_frame": QualifiedE4PLShellElement._generalized_section_in_frame,
        "_physical_director_context": QualifiedE4PLShellElement._physical_director_context,
        "_material_angle": ShellElement._material_angle,
        "_build_drilling_b_matrix": ShellElement._build_drilling_b_matrix,
        "compute_geometric_stiffness_matrix": QualifiedE4PLShellElement.compute_geometric_stiffness_matrix,
        "_membrane_compression_samples": ShellElement._membrane_compression_samples.__func__,
        "_bending_compression_samples": ShellElement._bending_compression_samples.__func__,
        "_stress_second_moment_samples": ShellElement._stress_second_moment_samples.__func__,
        "_membrane_compression_from_state": ShellElement._membrane_compression_from_state,
        "_resultant_samples": ShellElement._resultant_samples,
        "compute_mass_matrix": QualifiedE4PLShellElement.compute_mass_matrix,
        "get_node_coordinates": QualifiedE4PLShellElement.get_node_coordinates,
        "compute_shape_functions": ShellElement.compute_shape_functions,
        "_compute_3node_shape_functions": ShellElement._compute_3node_shape_functions,
        "_compute_4node_shape_functions": ShellElement._compute_4node_shape_functions,
        "_compute_6node_shape_functions": ShellElement._compute_6node_shape_functions,
        "_compute_8node_shape_functions": ShellElement._compute_8node_shape_functions,
        "gauss_points": ShellElement.gauss_points,
        "gauss_weights": ShellElement.gauss_weights,
        "shear_gauss_points": ShellElement.shear_gauss_points,
        "shear_gauss_weights": ShellElement.shear_gauss_weights,
        "compute_jacobian": ShellElement.compute_jacobian,
        "_fallback_edge_direction": ShellElement._fallback_edge_direction,
        "_normalize": ShellElement._normalize,
        "_local_frame_and_derivatives": QualifiedE4PLShellElement._local_frame_and_derivatives,
        "_local_dof_transform": ShellElement._local_dof_transform,
        "get_dof_mapping": ShellElement.get_dof_mapping,
        "validate_quadrature_authority": QualifiedE4PLShellElement.validate_quadrature_authority,
    },
    "qualified_s3": {
        "compute_stiffness_matrix": QualifiedE4PLS3ShellElement.compute_stiffness_matrix,
        "compute_stiffness_components": QualifiedE4PLS3ShellElement.compute_stiffness_components,
        "_compute_stiffness_components": QualifiedE4PLS3ShellElement._compute_stiffness_components,
        "_cache_key": QualifiedE4PLS3ShellElement._cache_key,
        "_constitutive": QualifiedE4PLS3ShellElement._constitutive,
        "_director_generalized_transform": QualifiedE4PLS3ShellElement._director_generalized_transform,
        "_generalized_section_in_frame": ShellElement._generalized_section_in_frame,
        "_material_angle": ShellElement._material_angle,
        "get_node_coordinates": QualifiedE4PLS3ShellElement.get_node_coordinates,
        "compute_geometric_stiffness_matrix": QualifiedE4PLS3ShellElement.compute_geometric_stiffness_matrix,
        "compute_geometric_stiffness_components": QualifiedE4PLS3ShellElement.compute_geometric_stiffness_components,
        "_compute_geometric_stiffness_components": QualifiedE4PLS3ShellElement._compute_geometric_stiffness_components,
        "_membrane_compression_samples": ShellElement._membrane_compression_samples.__func__,
        "_bending_compression_samples": ShellElement._bending_compression_samples.__func__,
        "_stress_second_moment_samples": ShellElement._stress_second_moment_samples.__func__,
        "_membrane_compression_from_state": ShellElement._membrane_compression_from_state,
        "_resultant_samples": ShellElement._resultant_samples,
        "_local_dof_transform": ShellElement._local_dof_transform,
        "compute_mass_matrix": QualifiedE4PLS3ShellElement.compute_mass_matrix,
        "compute_mass_components": QualifiedE4PLS3ShellElement.compute_mass_components,
        "_compute_mass_components": QualifiedE4PLS3ShellElement._compute_mass_components,
        "dynamic_algebraic_directions": QualifiedE4PLS3ShellElement.dynamic_algebraic_directions,
        "gauss_points": QualifiedE4PLS3ShellElement.gauss_points,
        "gauss_weights": QualifiedE4PLS3ShellElement.gauss_weights,
        "shear_gauss_points": QualifiedE4PLS3ShellElement.shear_gauss_points,
        "shear_gauss_weights": QualifiedE4PLS3ShellElement.shear_gauss_weights,
        "get_dof_mapping": ShellElement.get_dof_mapping,
        "validate_quadrature_authority": QualifiedE4PLS3ShellElement.validate_quadrature_authority,
    },
}
_QUALIFIED_Q4_BASE_OPERATOR_APIS = {
    "_local_frame_and_derivatives": _QUALIFIED_Q4_BASE_LOCAL_FRAME_KERNEL,
    "_mitc4_shear_b_matrix": _QUALIFIED_Q4_BASE_MITC4_SHEAR_KERNEL,
    "to_dict": _QUALIFIED_Q4_BASE_SERIALIZATION_KERNEL,
}

_S3_DYNAMIC_DESCRIPTOR_AUTHORITY = {
    "dynamic_algebraic_nullity": 3,
    "dynamic_algebraic_policy": ALGEBRAIC_COORDINATE_POLICY_ID,
    "dynamic_algebraic_mass_witness": "S3_LOCAL_DRILL_ROWS_EXACT_ZERO_V1",
    "dynamic_algebraic_local_zero_indices": (5, 11, 17),
}
_S3_V2D_DYNAMIC_DESCRIPTOR_AUTHORITY = {
    "dynamic_algebraic_nullity": 9,
    "dynamic_algebraic_policy": S3_V2D_DYNAMIC_POLICY_ID,
    "dynamic_algebraic_mass_witness": S3_V2D_DYNAMIC_MASS_WITNESS,
    "dynamic_algebraic_local_zero_indices": (3, 4, 5, 9, 10, 11, 15, 16, 17),
}
_S3_DYNAMIC_DESCRIPTOR_AUTHORITIES = {
    "qualified_s3": _S3_DYNAMIC_DESCRIPTOR_AUTHORITY,
    "qualified_s3_v2d": _S3_V2D_DYNAMIC_DESCRIPTOR_AUTHORITY,
}

_QUALIFIED_NONCURRENT_ACTIVITY_DISPOSITION_KEYS = frozenset(
    {
        "qualified_q4_activity_disposition",
        "qualified_s3_activity_disposition",
    }
)


def _static_mro_attribute(owner: type[Any], name: str) -> Any:
    """Return a class member without executing a descriptor."""

    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            value = namespace[name]
            if isinstance(value, (classmethod, staticmethod)):
                return value.__func__
            return value
    return None


def _static_formulation_id(element: Any) -> Optional[str]:
    """Read a formulation declaration without invoking a descriptor."""

    value = _static_mro_attribute(type(element), "formulation_id")
    return value if type(value) is str else None


def _has_no_dynamic_descriptor_declarations(element: Any) -> bool:
    names = tuple(
        sorted(
            {
                name
                for authority in _S3_DYNAMIC_DESCRIPTOR_AUTHORITIES.values()
                for name in authority
            }
        )
    )
    try:
        instance_namespace = object.__getattribute__(element, "__dict__")
    except AttributeError:
        instance_namespace = {}
    owner = type(element)
    return not any(name in instance_namespace for name in names) and not any(
        name in type.__getattribute__(base, "__dict__")
        for base in type.__getattribute__(owner, "__mro__")
        for name in names
    )


def _has_exact_s3_dynamic_descriptor_authority(
    element: Any,
    authority: Mapping[str, Any] = _S3_DYNAMIC_DESCRIPTOR_AUTHORITY,
) -> bool:
    """Check static class data without invoking caller-controlled descriptors."""

    try:
        instance_namespace = object.__getattribute__(element, "__dict__")
    except AttributeError:
        instance_namespace = {}
    class_namespace = type.__getattribute__(type(element), "__dict__")
    return all(
        name not in instance_namespace
        and type(class_namespace.get(name)) is type(expected)
        and class_namespace.get(name) == expected
        for name, expected in authority.items()
    )


def _require_current_state_modal_mass_authority(
    model: "FEModel", route: Mapping[str, Any]
) -> None:
    """Bind exact mass and algebraic-coordinate APIs before tangent mechanics."""

    failures: list[tuple[int, str]] = []
    profiles = route["element_profiles"]
    for raw_element_id, element in sorted(model.mesh.elements.items()):
        element_id = int(raw_element_id)
        family = str(profiles[element_id]["family"])
        expected_apis = _CURRENT_MODAL_MASS_APIS.get(family, {})
        instance_namespace = vars(element) if hasattr(element, "__dict__") else {}
        for name, expected in expected_apis.items():
            if (
                name in instance_namespace
                or _static_mro_attribute(type(element), name) is not expected
            ):
                failures.append((element_id, f"{family}:{name}"))
        if family in _S3_DYNAMIC_DESCRIPTOR_AUTHORITIES:
            if not _has_exact_s3_dynamic_descriptor_authority(
                element, _S3_DYNAMIC_DESCRIPTOR_AUTHORITIES[family]
            ):
                failures.append((element_id, f"{family}:dynamic_identity"))
        elif not _has_no_dynamic_descriptor_declarations(element):
            failures.append((element_id, f"{family}:unexpected_dynamic_identity"))
    if failures:
        detail = "; ".join(
            f"{element_id} ({reason})" for element_id, reason in failures[:8]
        )
        raise ElementCapabilityError(
            "current-state modal analysis requires exact formulation mass and "
            f"algebraic-coordinate authority; incompatible element IDs {detail}"
        )


def _canonical_prestress_element_id(
    raw: Any,
    _bool_types: tuple[type[Any], ...] = (bool, np.bool_),
    _exact_int: Any = int,
    _exact_isinstance: Any = isinstance,
    _integer_types: tuple[type[Any], ...] = (int, np.integer),
    _string_type: type[Any] = str,
) -> int:
    if _exact_isinstance(raw, _bool_types):
        raise ValueError("prestress element-state IDs must be canonical integers")
    if _exact_isinstance(raw, _integer_types):
        return _exact_int(raw)
    if _exact_isinstance(raw, _string_type) and raw == _string_type(
        _exact_int(raw)
    ):
        return _exact_int(raw)
    raise ValueError("prestress element-state IDs must be canonical integers")


def _evaluate_prestress_provider(provider: Any, element_id: int, element: Any) -> Any:
    try:
        signature = inspect.signature(provider)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "prestress state provider must expose a deterministic inspectable signature"
        ) from exc
    selected: Optional[tuple[Any, ...]] = None
    for arguments in ((element_id, element), (element_id,)):
        try:
            signature.bind(*arguments)
        except TypeError:
            continue
        selected = arguments
        break
    if selected is None:
        raise TypeError(
            "prestress state provider must accept (element_id, element) or element_id"
        )
    return provider(*selected)


def _guarded_prestress_snapshot(
    model: "FEModel",
    value: Any,
    *,
    path: str,
    _exact_guard: Any,
    _internal_guard: Any,
    _dict_items: Any = dict.items,
    _dict_type: type[Any] = dict,
    _exact_isinstance: Any = isinstance,
    _exact_iter: Any = iter,
    _exact_next: Any = next,
    _exact_type: Any = type,
    _list_type: type[Any] = list,
    _mapping_type: Any = Mapping,
    _numpy_array_type: type[Any] = np.ndarray,
    _numpy_asarray: Any = np.asarray,
    _numpy_generic_type: type[Any] = np.generic,
    _primitive_types: frozenset[type[Any]] = frozenset(
        {str, bool, int, float}
    ),
    _sequence_exclusions: tuple[type[Any], ...] = (
        str,
        bytes,
        bytearray,
    ),
    _sequence_type: Any = Sequence,
    _stop_iteration: type[BaseException] = StopIteration,
    _string_type: type[Any] = str,
    _snapshotter: Any = None,
    _tuple_type: type[Any] = tuple,
) -> Any:
    """Detach provider/mapping data before canonical prestress processing.

    Exact built-in containers cannot execute caller code while they are being
    traversed.  Their checkpoints may therefore use the operation's
    non-renewable, ABA-sensitive lease.  NumPy objects and arbitrary
    ``Mapping``/``Sequence`` implementations retain the complete lifecycle
    scan after every observation.
    """

    if _snapshotter is None:
        _snapshotter = _guarded_prestress_snapshot

    def snapshot(member: Any, member_path: str) -> Any:
        """Recurse without consulting mutable helper keyword defaults."""

        return _snapshotter(
            model,
            member,
            path=member_path,
            _exact_guard=_exact_guard,
            _internal_guard=_internal_guard,
            _dict_items=_dict_items,
            _dict_type=_dict_type,
            _exact_isinstance=_exact_isinstance,
            _exact_iter=_exact_iter,
            _exact_next=_exact_next,
            _exact_type=_exact_type,
            _list_type=_list_type,
            _mapping_type=_mapping_type,
            _numpy_array_type=_numpy_array_type,
            _numpy_asarray=_numpy_asarray,
            _numpy_generic_type=_numpy_generic_type,
            _primitive_types=_primitive_types,
            _sequence_exclusions=_sequence_exclusions,
            _sequence_type=_sequence_type,
            _stop_iteration=_stop_iteration,
            _string_type=_string_type,
            _snapshotter=_snapshotter,
            _tuple_type=_tuple_type,
        )

    context = f"qualified reference-prestress input observation at {path}"
    value_type = _exact_type(value)
    if value is None or value_type in _primitive_types:
        return value
    if value_type is _dict_type:
        observed_iterator = _exact_iter(_dict_items(value))
        _internal_guard(model, context=context)
        result: Dict[str, Any] = {}
        while True:
            try:
                observed_item = _exact_next(observed_iterator)
            except _stop_iteration:
                _internal_guard(model, context=context)
                break
            _internal_guard(model, context=context)
            key, member = observed_item
            _internal_guard(model, context=context)
            if _exact_type(key) is not _string_type:
                raise ValueError(f"prestress state has a non-string key at {path}")
            if key in result:
                raise ValueError(
                    f"prestress state has duplicate key {key!r} at {path}"
                )
            result[key] = snapshot(member, f"{path}.{key}")
        return result
    if value_type in {_list_type, _tuple_type}:
        observed_iterator = _exact_iter(value)
        _internal_guard(model, context=context)
        result = []
        index = 0
        while True:
            try:
                member = _exact_next(observed_iterator)
            except _stop_iteration:
                _internal_guard(model, context=context)
                break
            _internal_guard(model, context=context)
            result.append(snapshot(member, f"{path}[{index}]"))
            index += 1
        return result
    if _exact_isinstance(value, _numpy_array_type):
        observed = _numpy_asarray(value)
        _exact_guard(model, context=context)
        observed_list = observed.tolist()
        _exact_guard(model, context=context)
        return snapshot(observed_list, path)
    if _exact_isinstance(value, _numpy_generic_type):
        observed = value.item()
        _exact_guard(model, context=context)
        return snapshot(observed, path)
    if _exact_isinstance(value, _mapping_type):
        observed_items = value.items()
        _exact_guard(model, context=context)
        observed_iterator = _exact_iter(observed_items)
        _exact_guard(model, context=context)
        result: Dict[str, Any] = {}
        while True:
            try:
                observed_item = _exact_next(observed_iterator)
            except _stop_iteration:
                _exact_guard(model, context=context)
                break
            _exact_guard(model, context=context)
            key, member = observed_item
            _exact_guard(model, context=context)
            if _exact_type(key) is not _string_type:
                raise ValueError(f"prestress state has a non-string key at {path}")
            if key in result:
                raise ValueError(
                    f"prestress state has duplicate key {key!r} at {path}"
                )
            result[key] = snapshot(member, f"{path}.{key}")
        return result
    if _exact_isinstance(value, _sequence_type) and not _exact_isinstance(
        value, _sequence_exclusions
    ):
        observed_iterator = _exact_iter(value)
        _exact_guard(model, context=context)
        result = []
        index = 0
        while True:
            try:
                member = _exact_next(observed_iterator)
            except _stop_iteration:
                _exact_guard(model, context=context)
                break
            _exact_guard(model, context=context)
            result.append(snapshot(member, f"{path}[{index}]"))
            index += 1
        return result
    return value


def _normalize_prestress_states(
    model: "FEModel",
    source: Any,
    *,
    _exact_guard: Any = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
    _qualified_runtime_guard: Any = None,
    _dict_get: Any = dict.get,
    _dict_items: Any = dict.items,
    _dict_type: type[Any] = dict,
    _bool_type: type[Any] = bool,
    _bool_types: tuple[type[Any], ...] = (bool, np.bool_),
    _builtin_key_types: tuple[type[Any], ...] = (int, str),
    _exact_callable: Any = callable,
    _exact_enumerate: Any = enumerate,
    _exact_float_isfinite: Any = math.isfinite,
    _exact_float_repr: Any = repr,
    _exact_int: Any = int,
    _exact_isinstance: Any = isinstance,
    _exact_iter: Any = iter,
    _exact_len: Any = len,
    _exact_list: Any = list,
    _exact_next: Any = next,
    _exact_object_getattribute: Any = object.__getattribute__,
    _exact_set: Any = set,
    _exact_sorted: Any = sorted,
    _exact_str_encode: Any = str.encode,
    _exact_str_join: Any = str.join,
    _exact_type: Any = type,
    _exact_tuple: Any = tuple,
    _float_type: type[Any] = float,
    _inspect_signature: Any = inspect.signature,
    _integer_type: type[Any] = int,
    _json_encode_string: Any = _JSON_ENCODE_BASESTRING,
    _mapping_type: Any = Mapping,
    _numpy_array_type: type[Any] = np.ndarray,
    _numpy_asarray: Any = np.asarray,
    _numpy_generic_type: type[Any] = np.generic,
    _integer_types: tuple[type[Any], ...] = (int, np.integer),
    _primitive_types: frozenset[type[Any]] = frozenset(
        {str, bool, int, float}
    ),
    _sequence_exclusions: tuple[type[Any], ...] = (
        str,
        bytes,
        bytearray,
    ),
    _sequence_type: Any = Sequence,
    _schema_id: str = PRESTRESS_INPUT_SCHEMA_ID,
    _sha256: Any = hashlib.sha256,
    _stop_iteration: type[BaseException] = StopIteration,
    _string_type: type[Any] = str,
    _type_error: type[BaseException] = TypeError,
    _value_error: type[BaseException] = ValueError,
) -> tuple[Dict[int, Any], Dict[str, Any]]:
    """Evaluate once and bind a complete canonical element-state map.

    The complete lifecycle guard still owns both ends of this operation and
    every caller-controlled observation.  An all-qualified solver lease may
    amortize only traversal of already-observed exact built-in containers.
    """

    _exact_guard(
        model,
        context="qualified reference-prestress normalization preflight",
    )
    runtime_namespace = (
        _exact_object_getattribute(_qualified_runtime_guard, "__dict__")
        if _qualified_runtime_guard is not None
        else None
    )
    trusted_runtime_guard = None
    if _exact_type(runtime_namespace) is _dict_type:
        trusted_runtime_guard = _dict_get(
            runtime_namespace,
            "_qualified_trusted_input_require",
        )
        if trusted_runtime_guard is None:
            trusted_runtime_guard = _dict_get(
                runtime_namespace,
                "_qualified_trusted_require",
            )

    def internal_guard(
        observed_model: "FEModel",
        *,
        context: str,
    ) -> Any:
        if trusted_runtime_guard is None or observed_model is not model:
            return _exact_guard(observed_model, context=context)
        return trusted_runtime_guard(observed_model, context=context)

    def snapshot_state(value: Any, *, path: str) -> Any:
        """Detach input with an operation-private recursive helper."""

        context = f"qualified reference-prestress input observation at {path}"
        value_type = _exact_type(value)
        if value is None or value_type in _primitive_types:
            return value
        if value_type is _dict_type:
            observed_iterator = _exact_iter(_dict_items(value))
            internal_guard(model, context=context)
            result = {}
            while True:
                try:
                    observed_item = _exact_next(observed_iterator)
                except _stop_iteration:
                    internal_guard(model, context=context)
                    break
                internal_guard(model, context=context)
                key, member = observed_item
                internal_guard(model, context=context)
                if _exact_type(key) is not _string_type:
                    raise _value_error(
                        f"prestress state has a non-string key at {path}"
                    )
                if key in result:
                    raise _value_error(
                        f"prestress state has duplicate key {key!r} at {path}"
                    )
                result[key] = snapshot_state(member, path=f"{path}.{key}")
            return result
        if value_type in {_exact_list, _exact_tuple}:
            observed_iterator = _exact_iter(value)
            internal_guard(model, context=context)
            result = []
            index = 0
            while True:
                try:
                    member = _exact_next(observed_iterator)
                except _stop_iteration:
                    internal_guard(model, context=context)
                    break
                internal_guard(model, context=context)
                result.append(snapshot_state(member, path=f"{path}[{index}]"))
                index += 1
            return result
        if _exact_isinstance(value, _numpy_array_type):
            observed = _numpy_asarray(value)
            _exact_guard(model, context=context)
            observed_list = observed.tolist()
            _exact_guard(model, context=context)
            return snapshot_state(observed_list, path=path)
        if _exact_isinstance(value, _numpy_generic_type):
            observed = value.item()
            _exact_guard(model, context=context)
            return snapshot_state(observed, path=path)
        if _exact_isinstance(value, _mapping_type):
            observed_items = value.items()
            _exact_guard(model, context=context)
            observed_iterator = _exact_iter(observed_items)
            _exact_guard(model, context=context)
            result = {}
            while True:
                try:
                    observed_item = _exact_next(observed_iterator)
                except _stop_iteration:
                    _exact_guard(model, context=context)
                    break
                _exact_guard(model, context=context)
                key, member = observed_item
                _exact_guard(model, context=context)
                if _exact_type(key) is not _string_type:
                    raise _value_error(
                        f"prestress state has a non-string key at {path}"
                    )
                if key in result:
                    raise _value_error(
                        f"prestress state has duplicate key {key!r} at {path}"
                    )
                result[key] = snapshot_state(member, path=f"{path}.{key}")
            return result
        if _exact_isinstance(value, _sequence_type) and not _exact_isinstance(
            value, _sequence_exclusions
        ):
            observed_iterator = _exact_iter(value)
            _exact_guard(model, context=context)
            result = []
            index = 0
            while True:
                try:
                    member = _exact_next(observed_iterator)
                except _stop_iteration:
                    _exact_guard(model, context=context)
                    break
                _exact_guard(model, context=context)
                result.append(snapshot_state(member, path=f"{path}[{index}]"))
                index += 1
            return result
        return value

    def canonical_element_id(raw: Any) -> int:
        """Canonicalize one key without callback-reachable helper authority."""

        if _exact_isinstance(raw, _bool_types):
            raise _value_error(
                "prestress element-state IDs must be canonical integers"
            )
        if _exact_isinstance(raw, _integer_types):
            return _exact_int(raw)
        if _exact_isinstance(raw, _string_type) and raw == _string_type(
            _exact_int(raw)
        ):
            return _exact_int(raw)
        raise _value_error(
            "prestress element-state IDs must be canonical integers"
        )

    def canonical_owned(value: Any, *, path: str = "$") -> Any:
        """Copy an observed built-in tree under the canonical JSON rules."""

        value_type = _exact_type(value)
        if (
            value is None
            or value_type is _string_type
            or value_type is _bool_type
            or value_type is _integer_type
        ):
            return value
        if value_type is _float_type:
            if not _exact_float_isfinite(value):
                raise _value_error(f"nonfinite canonical value at {path}")
            return 0.0 if value == 0.0 else value
        if value_type is _dict_type:
            result = {}
            for key, member in _dict_items(value):
                if _exact_type(key) is not _string_type:
                    raise _value_error(f"non-string canonical key at {path}")
                if key in result:
                    raise _value_error(
                        f"duplicate canonical key {key!r} at {path}"
                    )
                result[key] = canonical_owned(member, path=f"{path}.{key}")
            return result
        if value_type is _exact_list:
            return [
                canonical_owned(member, path=f"{path}[{index}]")
                for index, member in _exact_enumerate(value)
            ]
        raise _value_error(
            f"unsupported canonical value {value_type.__name__} at {path}"
        )

    def canonical_fragment(value: Any) -> str:
        """Encode an already-owned canonical tree without shared Python helpers."""

        value_type = _exact_type(value)
        if value is None:
            return "null"
        if value_type is _bool_type:
            return "true" if value else "false"
        if value_type is _integer_type:
            return _string_type(value)
        if value_type is _float_type:
            return _exact_float_repr(value)
        if value_type is _string_type:
            return _json_encode_string(value)
        if value_type is _exact_list:
            return "[" + _exact_str_join(
                ",", [canonical_fragment(member) for member in value]
            ) + "]"
        if value_type is _dict_type:
            members = [
                _json_encode_string(key) + ":" + canonical_fragment(member)
                for key, member in _exact_sorted(_dict_items(value))
            ]
            return "{" + _exact_str_join(",", members) + "}"
        raise _value_error(
            f"unsupported canonical value {value_type.__name__} at $"
        )

    def canonical_json_owned(value: Any) -> bytes:
        return _exact_str_encode(canonical_fragment(value) + "\n", "utf-8")

    model_ids = _exact_tuple(
        _exact_sorted(_exact_int(value) for value in model.mesh.elements)
    )
    model_id_set = _exact_set(model_ids)
    supplied: Dict[int, Any] = {}
    if _exact_callable(source):
        source_kind = "callable_evaluated_once"
        provider_argument_count = 2
        if model_ids:
            signature_context = (
                "qualified reference-prestress provider signature observation"
            )
            try:
                signature = _inspect_signature(source)
            except (_type_error, _value_error) as exc:
                _exact_guard(model, context=signature_context)
                raise _type_error(
                    "prestress state provider must expose a deterministic "
                    "inspectable signature"
                ) from exc
            _exact_guard(model, context=signature_context)
            sample_element_id = model_ids[0]
            sample_element = model.mesh.elements[sample_element_id]
            provider_argument_count = 0
            for arguments in (
                (sample_element_id, sample_element),
                (sample_element_id,),
            ):
                try:
                    signature.bind(*arguments)
                except _type_error:
                    _exact_guard(model, context=signature_context)
                    continue
                _exact_guard(model, context=signature_context)
                provider_argument_count = 2 if _exact_len(arguments) == 2 else 1
                break
            if provider_argument_count == 0:
                raise _type_error(
                    "prestress state provider must accept "
                    "(element_id, element) or element_id"
                )
        for element_id in model_ids:
            raw_state = (
                source(element_id, model.mesh.elements[element_id])
                if provider_argument_count == 2
                else source(element_id)
            )
            _exact_guard(
                model,
                context=(
                    "qualified reference-prestress provider observation for "
                    f"element {element_id}"
                ),
            )
            try:
                snapshot = snapshot_state(
                    raw_state,
                    path=f"prestress_states[{element_id}]",
                )
                supplied[element_id] = canonical_owned(snapshot)
                internal_guard(
                    model,
                    context=(
                        "qualified reference-prestress owned provider state for "
                        f"element {element_id}"
                    ),
                )
            except (_type_error, _value_error) as exc:
                raise _value_error(
                    f"prestress state for element {element_id} is not strict "
                    "canonical data"
                ) from exc
    elif _exact_isinstance(source, _mapping_type):
        source_kind = "mapping"
        exact_builtin_mapping = _exact_type(source) is _dict_type
        mapping_guard = internal_guard if exact_builtin_mapping else _exact_guard
        observed_items = (
            _dict_items(source) if exact_builtin_mapping else source.items()
        )
        mapping_guard(
            model,
            context="qualified reference-prestress mapping observation",
        )
        observed_iterator = _exact_iter(observed_items)
        mapping_guard(
            model,
            context="qualified reference-prestress mapping observation",
        )
        while True:
            try:
                observed_item = _exact_next(observed_iterator)
            except _stop_iteration:
                mapping_guard(
                    model,
                    context="qualified reference-prestress mapping observation",
                )
                break
            mapping_guard(
                model,
                context="qualified reference-prestress mapping observation",
            )
            raw_element_id, state = observed_item
            mapping_guard(
                model,
                context="qualified reference-prestress mapping item observation",
            )
            element_id = canonical_element_id(raw_element_id)
            key_guard = (
                mapping_guard
                if _exact_type(raw_element_id) in _builtin_key_types
                else _exact_guard
            )
            key_guard(
                model,
                context="qualified reference-prestress element-ID observation",
            )
            if element_id in supplied:
                raise _value_error(
                    "prestress element-state IDs are duplicate or ambiguous"
                )
            if element_id not in model_id_set:
                raise _value_error(
                    f"prestress element-state ID {element_id} is not in the model"
                )
            try:
                snapshot = snapshot_state(
                    state,
                    path=f"prestress_states[{element_id}]",
                )
                supplied[element_id] = canonical_owned(snapshot)
                internal_guard(
                    model,
                    context=(
                        "qualified reference-prestress owned mapping state for "
                        f"element {element_id}"
                    ),
                )
            except (_type_error, _value_error) as exc:
                raise _value_error(
                    f"prestress state for element {element_id} is not strict "
                    "canonical data"
                ) from exc
    else:
        raise _type_error(
            "prestress_states must be a mapping or deterministic callable"
        )

    complete: Dict[int, Any] = {}
    for element_id in model_ids:
        raw_state = supplied.get(element_id)
        try:
            complete[element_id] = canonical_owned(raw_state)
        except (_type_error, _value_error) as exc:
            raise _value_error(
                f"prestress state for element {element_id} is not strict canonical data"
            ) from exc
    state_hashes: Dict[str, str] = {}
    for element_id, state in _dict_items(complete):
        try:
            payload = canonical_json_owned(state)
        except (_type_error, _value_error) as exc:
            raise _value_error(
                f"prestress state for element {element_id} is not strict canonical data"
            ) from exc
        state_hashes[_string_type(element_id)] = _sha256(
            payload
        ).hexdigest().upper()
    provenance = {
        "schema_id": _schema_id,
        "source_kind": source_kind,
        "element_ids": _exact_list(model_ids),
        "supplied_element_ids": _exact_sorted(
            _exact_int(value) for value in supplied
        ),
        "explicitly_unstressed_element_ids": [
            element_id for element_id, state in complete.items() if state is None
        ],
        "state_sha256": state_hashes,
    }
    _exact_guard(
        model,
        context="qualified reference-prestress normalization output",
    )
    return complete, provenance


def _require_exact_eigen_runtime_owners(
    *,
    qualified_element_ids: Any,
    session: Optional[AnalysisSession],
    factorization_cache: Optional[FactorizationCache],
    context: str,
) -> None:
    """Close session/cache method dispatch for qualified eigen routes."""

    if not tuple(qualified_element_ids):
        return

    def require_cache(cache: FactorizationCache, label: str) -> None:
        if type(cache) is not FactorizationCache:
            raise ElementCapabilityError(
                f"{context} requires an exact FactorizationCache for {label}"
            )
        namespace = vars(cache)
        changed = [
            name
            for name, expected in _FACTORIZATION_CACHE_METHOD_AUTHORITY.items()
            if name in namespace
            or vars(FactorizationCache).get(name) is not expected
        ]
        if changed:
            raise ElementCapabilityError(
                f"{context} found incompatible FactorizationCache methods for "
                f"{label}: {', '.join(sorted(changed))}"
            )

    if session is not None:
        if type(session) is not AnalysisSession:
            raise ElementCapabilityError(
                f"{context} requires the exact AnalysisSession type"
            )
        namespace = vars(session)
        changed = [
            name
            for name, expected in _ANALYSIS_SESSION_METHOD_AUTHORITY.items()
            if name in namespace
            or vars(AnalysisSession).get(name) is not expected
        ]
        if changed:
            raise ElementCapabilityError(
                f"{context} found incompatible AnalysisSession methods: "
                + ", ".join(sorted(changed))
            )
        owned_cache = namespace.get("factorization_cache")
        require_cache(owned_cache, "AnalysisSession")
    if factorization_cache is not None:
        require_cache(factorization_cache, "solver input")


def _require_qualified_prestress_operator_authority(
    model: "FEModel",
    states: Mapping[int, Any],
    *,
    include_mass_and_descriptor: bool,
    _exact_guard: Any = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
) -> None:
    """Reject qualified-family API or prestress-policy spoofing pre-mechanics."""

    q4_id = "E4_PL_QUALIFIED_Q4_HYBRID_V2"
    s3_id = "E4_PL_QUALIFIED_S3_COMPANION_V1"
    _exact_guard(
        model,
        context="qualified reference-prestress analysis",
    )
    # Reference-prestress inputs are a separate operator route, not a way to
    # reinterpret a nonlinear result whose lifecycle explicitly denies ACTIVE
    # authority.  Reject the mere presence of either family marker before any
    # stiffness, mass, or geometric operator can be evaluated.  This also
    # fails closed for malformed/tampered dispositions whose internal status
    # or hash could not safely be trusted.
    for raw_element_id, element in sorted(model.mesh.elements.items()):
        element_id = int(raw_element_id)
        formulation_id = _static_formulation_id(element)
        if formulation_id not in {q4_id, s3_id}:
            continue
        state = states[element_id]
        if not isinstance(state, Mapping):
            continue
        markers = sorted(
            _QUALIFIED_NONCURRENT_ACTIVITY_DISPOSITION_KEYS.intersection(state)
        )
        if markers:
            raise ValueError(
                "qualified reference-prestress analysis rejects noncurrent "
                "activity dispositions before mechanics for element "
                f"{element_id}: {', '.join(markers)}"
            )
    _exact_guard(
        model,
        context="qualified reference-prestress state observation",
    )
    failures: list[tuple[int, str]] = []
    state_failures: list[int] = []
    total_dofs = int(model.mesh.dof_manager.total_dofs)
    for raw_element_id, element in sorted(model.mesh.elements.items()):
        element_id = int(raw_element_id)
        formulation_id = _static_formulation_id(element)
        if type(element) is QualifiedE4PLShellElement or formulation_id == q4_id:
            family = "qualified_q4"
            expected_type = QualifiedE4PLShellElement
            node_count = 4
        elif type(element) is QualifiedE4PLS3ShellElement or formulation_id == s3_id:
            family = "qualified_s3"
            expected_type = QualifiedE4PLS3ShellElement
            node_count = 3
        else:
            continue
        expected_formulation_id = (
            q4_id if family == "qualified_q4" else s3_id
        )
        instance_namespace = vars(element) if hasattr(element, "__dict__") else {}
        if (
            type(element) is not expected_type
            or formulation_id != expected_formulation_id
            or "formulation_id" in instance_namespace
        ):
            failures.append((element_id, f"{family}:type_or_formulation"))
            continue
        if (
            family == "qualified_q4"
            and ShellElement.compute_stiffness_matrix
            is not _QUALIFIED_Q4_BASE_STIFFNESS_KERNEL
        ):
            failures.append((element_id, f"{family}:base_stiffness_kernel"))
        if family == "qualified_q4":
            for name, expected in _QUALIFIED_Q4_BASE_OPERATOR_APIS.items():
                if _static_mro_attribute(ShellElement, name) is not expected:
                    failures.append((element_id, f"{family}:base_{name}"))
        if any(callable(value) for value in instance_namespace.values()):
            failures.append((element_id, f"{family}:callable_instance_override"))
        expected_apis = _QUALIFIED_PRESTRESS_OPERATOR_APIS[family]
        if not include_mass_and_descriptor:
            expected_apis = {
                name: expected
                for name, expected in expected_apis.items()
                if name
                not in {
                    "compute_mass_matrix",
                    "compute_mass_components",
                    "_compute_mass_components",
                    "dynamic_algebraic_directions",
                }
            }
        for name, expected in expected_apis.items():
            if (
                name in instance_namespace
                or _static_mro_attribute(type(element), name) is not expected
            ):
                failures.append((element_id, f"{family}:{name}"))
        try:
            expected_apis["validate_quadrature_authority"](element)
        except (AttributeError, TypeError, ValueError):
            failures.append((element_id, f"{family}:quadrature_authority"))
        actual_dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        expected_dofs = np.asarray(
            [
                int(dof)
                for node_id in tuple(int(value) for value in element.node_ids)
                for dof in model.mesh.dof_manager.get_node_dofs(node_id)
            ],
            dtype=np.intp,
        )
        if (
            actual_dofs.shape != (6 * node_count,)
            or expected_dofs.shape != (6 * node_count,)
            or not np.array_equal(actual_dofs, expected_dofs)
            or np.any(actual_dofs < 0)
            or np.any(actual_dofs >= total_dofs)
        ):
            failures.append((element_id, f"{family}:dof_mapping"))
        if family == "qualified_s3":
            state = states[element_id]
            if state is not None and (
                not isinstance(state, Mapping)
                or state.get("bubble_linearization_policy")
                != REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
            ):
                state_failures.append(element_id)
            if include_mass_and_descriptor:
                if not _has_exact_s3_dynamic_descriptor_authority(element):
                    failures.append((element_id, f"{family}:dynamic_identity"))
        elif include_mass_and_descriptor and not (
            _has_no_dynamic_descriptor_declarations(element)
        ):
            failures.append((element_id, f"{family}:unexpected_dynamic_identity"))
    if state_failures:
        if include_mass_and_descriptor:
            failures.extend(
                (element_id, "qualified_s3:bubble_policy")
                for element_id in state_failures
            )
        else:
            raise ValueError(
                "qualified S3 reference prestress requires exact "
                "bubble_linearization_policy for element IDs "
                + ", ".join(str(value) for value in state_failures)
            )
    if failures:
        detail = "; ".join(
            f"{element_id} ({reason})" for element_id, reason in failures[:8]
        )
        raise ElementCapabilityError(
            "qualified reference-prestress analysis requires exact "
            f"operator/state authority; incompatible element IDs {detail}"
        )
    _exact_guard(
        model,
        context="qualified reference-prestress analysis",
    )


def _assemble_committed_current_tangent(
    model: "FEModel",
    displacements: Any,
    element_states: Any,
    num_layers: int,
    *,
    _exact_guard: Any,
) -> tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Evaluate the exact read-only committed total tangent used by Q4/S3."""

    _material, _geometric, total, component_info = (
        _EXACT_CURRENT_STATE_COMPONENT_IMPLEMENTATION(
            model,
            displacements,
            element_states,
            num_layers,
            _exact_guard=_exact_guard,
        )
    )
    info = dict(component_info)
    info.update(
        {
            "matrix_type": "committed_current_total_tangent",
            "modal_policy_id": CURRENT_STATE_MODAL_POLICY_ID,
            "matrix_persistence": "none",
            "factorization_persistence": "none",
        }
    )
    return total, info


@dataclass
class ModalMode:
    """One free-vibration mode."""

    mode_number: int
    eigenvalue: float
    angular_frequency: float
    frequency_hz: float
    period: Optional[float]
    mode_shape: np.ndarray
    reduced_mode_shape: np.ndarray
    modal_mass: float
    modal_stiffness: float
    residual_norm: float
    rigid_body_correlation: float
    is_rigid_body: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_number": int(self.mode_number),
            "eigenvalue": float(self.eigenvalue),
            "angular_frequency": float(self.angular_frequency),
            "frequency_hz": float(self.frequency_hz),
            "period": None if self.period is None else float(self.period),
            "mode_shape": self.mode_shape.tolist(),
            "modal_mass": float(self.modal_mass),
            "modal_stiffness": float(self.modal_stiffness),
            "residual_norm": float(self.residual_norm),
            "rigid_body_correlation": float(self.rigid_body_correlation),
            "is_rigid_body": bool(self.is_rigid_body),
        }


@dataclass
class ModalResult:
    """Result bundle from modal analysis."""

    modes: List[ModalMode]
    num_modes_requested: int
    solver_status: str
    constraint_info: Dict[str, Any]
    nullspace_info: Dict[str, Any]
    assembly_info: Dict[str, Any]
    diagnostics: Dict[str, Any]
    result_case: Optional[Dict[str, Any]] = None

    @property
    def num_modes_returned(self) -> int:
        return len(self.modes)

    @property
    def quantity_metadata(self) -> Tuple[Any, ...]:
        from .quantities import describe_result_quantities

        return describe_result_quantities(self)

    @property
    def frequencies_hz(self) -> np.ndarray:
        return np.asarray([mode.frequency_hz for mode in self.modes], dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_status": self.solver_status,
            "num_modes_requested": int(self.num_modes_requested),
            "num_modes_returned": int(self.num_modes_returned),
            "frequencies_hz": self.frequencies_hz.tolist(),
            "constraint_info": self.constraint_info,
            "nullspace_info": self.nullspace_info,
            "assembly_info": self.assembly_info,
            "diagnostics": self.diagnostics,
            "result_case": self.result_case,
            "modes": [mode.to_dict() for mode in self.modes],
        }


def _sym(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    return (0.5 * (matrix + matrix.T)).tocsr()


def _dense_eigensolve(K: sparse.spmatrix, M: sparse.spmatrix) -> Tuple[np.ndarray, np.ndarray]:
    Kd = np.asarray(K.toarray(), dtype=float)
    Md = np.asarray(M.toarray(), dtype=float)
    Kd = 0.5 * (Kd + Kd.T)
    Md = 0.5 * (Md + Md.T)
    return linalg.eigh(Kd, Md)


def _sparse_eigensolve(
    K: sparse.spmatrix,
    M: sparse.spmatrix,
    num_modes: int,
    shift: Optional[float],
    factorization_cache: Optional[FactorizationCache] = None,
    *,
    _post_factorization_guard: Any = None,
    use_zero_shift_inverse: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n = int(K.shape[0])
    k = min(max(num_modes + 4, num_modes), n - 1)
    if shift is None and not use_zero_shift_inverse:
        values, vectors = sparse_linalg.eigsh(K.tocsc(), k=k, M=M.tocsc(), which="SM")
        return values, vectors, {"shift_invert": False}
    effective_shift = 0.0 if shift is None else float(shift)
    shift_matrix = (K - effective_shift * M).tocsc()
    cache = factorization_cache or FactorizationCache(name="modal_shift_invert", max_entries=2)
    factorization_started = time.perf_counter()
    try:
        operator, handle = cached_inverse_operator(
            shift_matrix,
            (
                MatrixClass.SPD
                if shift is None and use_zero_shift_inverse
                else MatrixClass.SYMMETRIC_INDEFINITE
            ),
            cache=cache,
        )
    except RuntimeError as exc:
        factorization_lookup_seconds = float(
            time.perf_counter() - factorization_started
        )
        if shift is not None or not use_zero_shift_inverse:
            raise
        if _post_factorization_guard is not None:
            _post_factorization_guard()
        # Absence of geometric rigid modes does not prove that K has no
        # internal mechanism.  Preserve the former singular-safe route when
        # the optional zero-shift inverse cannot be constructed.
        values, vectors = sparse_linalg.eigsh(
            K.tocsc(),
            k=k,
            M=M.tocsc(),
            which="SM",
        )
        return values, vectors, {
            "shift_invert": False,
            "zero_shift_inverse": False,
            "zero_shift_inverse_fallback": True,
            "zero_shift_inverse_failure": str(exc),
            "factorization_lookup_seconds": factorization_lookup_seconds,
            "factorization_cache": cache.diagnostics(),
        }
    factorization_lookup_seconds = float(
        time.perf_counter() - factorization_started
    )
    if _post_factorization_guard is not None:
        _post_factorization_guard()
    values, vectors = sparse_linalg.eigsh(
        K.tocsc(),
        k=k,
        M=M.tocsc(),
        sigma=effective_shift,
        which="LM",
        OPinv=operator,
    )
    return values, vectors, {
        "shift_invert": True,
        "zero_shift_inverse": bool(shift is None and use_zero_shift_inverse),
        "factorization_lookup_seconds": factorization_lookup_seconds,
        "shift_factorization": handle.diagnostics(),
        "factorization_cache": cache.diagnostics(),
    }


def _deterministic_sign(vector: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(vector))) if vector.size else 0
    if vector.size and vector[idx] < 0.0:
        return -vector
    return vector


def _orthogonality_error(modes: List[ModalMode], M_red: sparse.spmatrix) -> float:
    if not modes:
        return 0.0
    Phi = np.column_stack([mode.reduced_mode_shape for mode in modes])
    gram = np.asarray(Phi.T @ (M_red @ Phi), dtype=float)
    return float(np.max(np.abs(gram - np.eye(gram.shape[0]))))


@dataclass(frozen=True)
class _ModalOperationConfig:
    num_modes: int
    shift: Optional[float]
    dense_size_limit: int
    eigen_tolerance: float
    rigid_body_frequency_tolerance: float
    current_state_num_layers: int


def _owned_modal_operation_config(
    model: "FEModel",
    *,
    num_modes: Any,
    shift: Any,
    dense_size_limit: Any,
    eigen_tolerance: Any,
    rigid_body_frequency_tolerance: Any,
    current_state_num_layers: Any,
    _exact_guard: Any,
    _trusted_guard: Any,
) -> _ModalOperationConfig:
    """Detach scalar modal policy before session or matrix observation."""

    def converted(value: Any, converter: Any, name: str) -> Any:
        made = converter(value)
        # Exact built-in scalars cannot execute caller code during conversion.
        # Keep the complete lifecycle guard for subclasses/NumPy scalars, but
        # use the solve lease's constant-time epoch check for the ordinary hot
        # path.  The complete guard still brackets the owned configuration.
        guard = _trusted_guard if type(value) in {bool, int, float} else _exact_guard
        guard(model, context=f"modal {name} conversion")
        return made

    def canonical_int(value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(f"{name} must be an integer")
        return converted(value, int, name)

    owned_shift = (
        None
        if shift is None
        else converted(shift, float, "shift")
    )
    if owned_shift is not None and not np.isfinite(owned_shift):
        raise ValueError("shift must be finite when supplied")
    owned = _ModalOperationConfig(
        num_modes=canonical_int(num_modes, "num_modes"),
        shift=owned_shift,
        dense_size_limit=canonical_int(
            dense_size_limit,
            "dense_size_limit",
        ),
        eigen_tolerance=converted(
            eigen_tolerance,
            float,
            "eigen_tolerance",
        ),
        rigid_body_frequency_tolerance=converted(
            rigid_body_frequency_tolerance,
            float,
            "rigid_body_frequency_tolerance",
        ),
        current_state_num_layers=canonical_int(
            current_state_num_layers,
            "current_state_num_layers",
        ),
    )
    _exact_guard(model, context="modal owned configuration")
    return owned


@resource_threaded
def _solve_free_vibration_under_lease(
    model: "FEModel",
    num_modes: int = 6,
    shift: Optional[float] = None,
    dense_size_limit: int = 200,
    eigen_tolerance: float = 1.0e-9,
    rigid_body_frequency_tolerance: float = 1.0e-6,
    factorization_cache: Optional[FactorizationCache] = None,
    resource_config: Optional[ResourceConfig] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    session: Optional["AnalysisSession"] = None,
    prestress_states: Optional[Any] = None,
    current_state_displacements: Optional[Any] = None,
    current_state_element_states: Optional[Any] = None,
    current_state_num_layers: int = 5,
    _qualified_runtime_guard: Any = None,
    _exact_cancellation_safe_point: Any = _EXACT_CANCELLATION_SAFE_POINT,
) -> ModalResult:
    """Solve ``K phi = omega^2 M phi`` with the common constraint transform.

    ``prestress_states`` activates the stress-stiffened tangent
    ``K_material - K_G``.  ``K_G`` uses the same compression-positive
    convention as linear buckling.  Element operators own bubble/internal
    condensation; this solver only assembles their final nodal matrices.

    ``current_state_displacements`` plus ``current_state_element_states``
    instead evaluate the formulation-native committed total tangent through
    a read-only zero-increment state transaction.  That path retains the
    current material, geometric, bubble-Schur and objective-PL tangent while
    continuing to use the formulation's consistent reference mass.
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

    prestress_authority_guard = _require_qualified_prestress_operator_authority
    snapshot_current_state = _EXACT_CURRENT_STATE_INPUT_SNAPSHOT
    current_state_route_guard = _EXACT_COMMITTED_TANGENT_ROUTE_GUARD
    current_state_activity_guard = _EXACT_ACTIVE_CURRENT_STATE_LIFECYCLE_GUARD
    current_state_input_validator = _EXACT_CURRENT_STATE_INPUT_VALIDATOR
    current_state_assembler = _assemble_committed_current_tangent
    qualified_lifecycle_authority = exact_guard(
        model,
        context="solve_free_vibration preflight",
    )
    if cancellation_checkpoint("modal.start"):
        trusted_guard(model, context="solve_free_vibration cancellation start")
    else:
        exact_guard(model, context="solve_free_vibration cancellation start")
    owned_config = _owned_modal_operation_config(
        model,
        num_modes=num_modes,
        shift=shift,
        dense_size_limit=dense_size_limit,
        eigen_tolerance=eigen_tolerance,
        rigid_body_frequency_tolerance=rigid_body_frequency_tolerance,
        current_state_num_layers=current_state_num_layers,
        _exact_guard=exact_guard,
        _trusted_guard=trusted_guard,
    )
    num_modes = owned_config.num_modes
    shift = owned_config.shift
    dense_size_limit = owned_config.dense_size_limit
    eigen_tolerance = owned_config.eigen_tolerance
    rigid_body_frequency_tolerance = (
        owned_config.rigid_body_frequency_tolerance
    )
    current_state_num_layers = owned_config.current_state_num_layers
    _require_exact_eigen_runtime_owners(
        qualified_element_ids=qualified_lifecycle_authority[
            "qualified_element_ids"
        ],
        session=session,
        factorization_cache=factorization_cache,
        context="solve_free_vibration",
    )
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")
    if session is not None:
        session._require_model(model)
        exact_guard(model, context="solve_free_vibration session ownership")
    current_state = (
        current_state_displacements is not None
        or current_state_element_states is not None
    )
    if (current_state_displacements is None) != (
        current_state_element_states is None
    ):
        raise ValueError(
            "current-state modal analysis requires both committed "
            "displacements and element states"
        )
    if current_state and prestress_states is not None:
        raise ValueError(
            "current-state modal tangent and reference-elastic prestress_states "
            "are mutually exclusive"
        )
    if current_state and factorization_cache is not None:
        raise ValueError(
            "current-state modal analysis and a persistent factorization_cache "
            "are mutually exclusive"
        )
    if prestress_states is not None and factorization_cache is not None:
        raise ValueError(
            "reference-prestressed modal analysis and a persistent "
            "factorization_cache are mutually exclusive"
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
    if current_state:
        current_state_route = current_state_route_guard(
            model,
            context="solve_free_vibration current-state route",
        )
        current_state_activity_authority = (
            current_state_activity_guard(
                model,
                current_state_route,
                context="current-state modal analysis",
            )
        )
        _require_current_state_modal_mass_authority(model, current_state_route)
        q4_ids = tuple(
            int(element_id)
            for element_id, profile in sorted(
                current_state_route["element_profiles"].items()
            )
            if str(profile["family"]) == "qualified_q4"
        )
        s3_ids = tuple(
            int(element_id)
            for element_id, profile in sorted(
                current_state_route["element_profiles"].items()
            )
            if str(profile["family"]) in {"qualified_s3", "qualified_s3_v2d"}
        )
        if q4_ids:
            require_model_element_capabilities(
                model,
                "current_state_modal",
                context="solve_free_vibration",
                element_ids=q4_ids,
            )
        if s3_ids:
            require_model_element_capabilities(
                model,
                "current_state_modal",
                context="solve_free_vibration",
                element_ids=s3_ids,
            )
        if str(current_state_route["route"]) == "mixed_qualified_q4_s3":
            require_model_element_capabilities(
                model,
                "mixed_current_state_modal",
                context="solve_free_vibration",
                element_ids=tuple(sorted((*q4_ids, *s3_ids))),
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
            context="solve_free_vibration current-state snapshot",
        )
        # Complete all state/binding guards before boundary-condition or matrix
        # evaluation.  The component assembler repeats this validation at its
        # own public boundary so direct callers receive the same fail-closed
        # contract.
        current_state_input_validator(
            model,
            current_state_displacements,
            current_state_element_states,
            current_state_num_layers,
            context="solve_free_vibration current-state inputs",
            _exact_guard=exact_guard,
        )
    normalized_prestress_states: Optional[Dict[int, Any]] = None
    packed_prestress_states: Any = None
    prestress_input_info: Optional[Dict[str, Any]] = None
    qualified_formulation_ids = {
        "E4_PL_QUALIFIED_Q4_HYBRID_V2",
        "E4_PL_QUALIFIED_S3_COMPANION_V1",
    }
    has_qualified_reference_elements = any(
        type(element)
        in {QualifiedE4PLShellElement, QualifiedE4PLS3ShellElement}
        or _static_formulation_id(element) in qualified_formulation_ids
        for element in model.mesh.elements.values()
    )
    reference_operator_authority = False
    if prestress_states is not None:
        require_model_element_capabilities(
            model,
            "reference_elastic_prestressed_modal",
            context="solve_free_vibration",
        )
        normalized_prestress_states, prestress_input_info = (
            _normalize_prestress_states(
                model,
                prestress_states,
                _exact_guard=exact_guard,
                _qualified_runtime_guard=_qualified_runtime_guard,
            )
        )
        exact_guard(
            model,
            context="solve_free_vibration prestress snapshot",
        )
        prestress_authority_guard(
            model,
            normalized_prestress_states,
            include_mass_and_descriptor=True,
            _exact_guard=trusted_guard,
        )
        packed_prestress_states = _pack_owned_geometric_states(
            normalized_prestress_states
        )
        reference_operator_authority = has_qualified_reference_elements
    elif not current_state:
        if has_qualified_reference_elements:
            prestress_authority_guard(
                model,
                {int(element_id): None for element_id in model.mesh.elements},
                include_mass_and_descriptor=True,
                _exact_guard=trusted_guard,
            )
            reference_operator_authority = True
    phase_timings["validation"] = float(time.perf_counter() - operation_started)
    model.apply_boundary_conditions()
    exact_guard(model, context="solve_free_vibration boundary conditions")
    current_state_info = None
    phase_timings["stiffness"] = 0.0
    phase_timings["mass"] = 0.0
    phase_timings["geometric_stiffness"] = 0.0
    phase_timings["constraint_reduction"] = 0.0
    if session is None or current_state:
        stiffness_started = time.perf_counter()
        if current_state:
            K, current_state_info = current_state_assembler(
                model,
                current_state_displacements,
                current_state_element_states,
                current_state_num_layers,
                _exact_guard=exact_guard,
            )
            exact_guard(model, context="solve_free_vibration current-state assembly")
            stiffness_info = dict(current_state_info)
        else:
            K, stiffness_info = assemble_stiffness_matrix(model)
            exact_guard(model, context="solve_free_vibration stiffness assembly")
        phase_timings["stiffness"] = float(
            time.perf_counter() - stiffness_started
        )
        mass_started = time.perf_counter()
        M, mass_info = assemble_mass_matrix(model)
        phase_timings["mass"] = float(time.perf_counter() - mass_started)
        exact_guard(model, context="solve_free_vibration mass assembly")
        geometric_info = None
        if prestress_states is not None:
            geometric_started = time.perf_counter()
            if assemble_geometric_stiffness_matrix is _EXACT_PUBLIC_GEOMETRIC_ASSEMBLER:
                geometric, geometric_info = _assemble_geometric_stiffness_matrix_under_lease(
                    model,
                    packed_prestress_states,
                    qualified_runtime_guard=_qualified_runtime_guard,
                )
            else:
                geometric, geometric_info = assemble_geometric_stiffness_matrix(
                    model,
                    normalized_prestress_states,
                )
            phase_timings["geometric_stiffness"] = float(
                time.perf_counter() - geometric_started
            )
            exact_guard(model, context="solve_free_vibration geometric assembly")
            K = (K - geometric).tocsr()
        reduction_started = time.perf_counter()
        zero = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
        K_red, _, T, _, independent_dofs, constraint_info = (
            build_constraint_transformation(K, zero, model)
        )
        exact_guard(model, context="solve_free_vibration constraint transformation")
        M_red = (T.T @ M @ T).tocsr()
        Q, nullspace_info = build_reduced_rigid_body_modes(
            model,
            independent_dofs,
            int(K.shape[0]),
            transformation=T,
        )
        phase_timings["constraint_reduction"] = float(
            time.perf_counter() - reduction_started
        )
        exact_guard(model, context="solve_free_vibration rigid-body basis")
    else:
        stiffness_started = time.perf_counter()
        stiffness_plan = session.stiffness_plan(model)
        phase_timings["stiffness"] = float(
            time.perf_counter() - stiffness_started
        )
        exact_guard(model, context="solve_free_vibration session stiffness plan")
        reduction_started = time.perf_counter()
        constraint_plan = session.constraint_plan(stiffness_plan, model)
        exact_guard(model, context="solve_free_vibration session constraint plan")
        phase_timings["constraint_reduction"] = float(
            time.perf_counter() - reduction_started
        )
        mass_started = time.perf_counter()
        mass_plan = session.mass_plan(model)
        exact_guard(model, context="solve_free_vibration session mass plan")
        K = stiffness_plan.matrix
        M = mass_plan.matrix
        stiffness_info = dict(stiffness_plan.info)
        mass_info = dict(mass_plan.info)
        M_red, _ = session.reduced_mass(constraint_plan, model)
        phase_timings["mass"] = float(time.perf_counter() - mass_started)
        exact_guard(model, context="solve_free_vibration session reduced mass")
        geometric_info = None
        if prestress_states is None:
            K_red = constraint_plan.K_red
        else:
            geometric_started = time.perf_counter()
            if assemble_geometric_stiffness_matrix is _EXACT_PUBLIC_GEOMETRIC_ASSEMBLER:
                geometric, geometric_info = _assemble_geometric_stiffness_matrix_under_lease(
                    model,
                    packed_prestress_states,
                    qualified_runtime_guard=_qualified_runtime_guard,
                )
            else:
                geometric, geometric_info = assemble_geometric_stiffness_matrix(
                    model,
                    normalized_prestress_states,
                )
            phase_timings["geometric_stiffness"] = float(
                time.perf_counter() - geometric_started
            )
            exact_guard(model, context="solve_free_vibration geometric assembly")
            K = (K - geometric).tocsr()
            K_red = (constraint_plan.T.T @ K @ constraint_plan.T).tocsr()
        T = constraint_plan.T
        independent_dofs = constraint_plan.independent_dofs
        constraint_info = dict(constraint_plan.info)
        Q, nullspace_info = session.rigid_body_modes(constraint_plan, model)
        exact_guard(model, context="solve_free_vibration session rigid-body basis")
    if cancellation_checkpoint("modal.after_assembly"):
        trusted_guard(
            model,
            context="solve_free_vibration cancellation after assembly",
        )
    else:
        exact_guard(
            model,
            context="solve_free_vibration cancellation after assembly",
        )

    assembly_info = {
        "stiffness": stiffness_info,
        "mass": mass_info,
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "reduced_dofs": int(K_red.shape[0]),
    }
    if geometric_info is not None:
        assembly_info["geometric_stiffness"] = geometric_info
        assembly_info["prestressed_modal_policy_id"] = (
            PRESTRESSED_MODAL_POLICY_ID
        )
        if reference_operator_authority:
            assembly_info["prestress_operator_authority_policy_id"] = (
                QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID
            )
        assembly_info["prestress_input"] = prestress_input_info
        if session is not None:
            assembly_info["analysis_session_state_dependent_bypass_reason"] = (
                "prestress_combined_matrix_and_factors_are_not_cacheable"
            )
    if reference_operator_authority:
        assembly_info["reference_operator_authority_policy_id"] = (
            QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID
        )
    if current_state_info is not None:
        assembly_info["current_state_tangent"] = current_state_info
        assembly_info["current_state_modal_policy_id"] = (
            CURRENT_STATE_MODAL_POLICY_ID
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
        exact_guard(model, context="solve_free_vibration session diagnostics")
    settings = {
        "num_modes": int(num_modes),
        "shift": None if shift is None else float(shift),
        "dense_size_limit": int(dense_size_limit),
        "eigen_tolerance": float(eigen_tolerance),
        "rigid_body_frequency_tolerance": float(rigid_body_frequency_tolerance),
        "factorization_cache": (
            (
                None
                if current_state or prestress_states is not None
                else (session.factorization_cache.name if session is not None else None)
            )
            if factorization_cache is None
            else factorization_cache.name
        ),
        "resource_config": None if resource_config is None else resource_config.to_dict(),
    }
    if prestress_states is not None:
        settings.update(
            {
                "prestress_state_source": type(prestress_states).__name__,
                "prestressed_modal_policy_id": PRESTRESSED_MODAL_POLICY_ID,
                "prestress_input_schema_id": PRESTRESS_INPUT_SCHEMA_ID,
                "prestress_matrix_persistence": "none",
                "prestress_factorization_persistence": "none",
            }
        )
    if current_state:
        settings.update(
            {
                "current_state_modal_policy_id": CURRENT_STATE_MODAL_POLICY_ID,
                "current_state_num_layers": int(current_state_num_layers),
                "current_state_matrix_persistence": "none",
                "current_state_factorization_persistence": "none",
                "current_state_route": (
                    None
                    if current_state_route is None
                    else str(current_state_route["route"])
                ),
            }
        )
    descriptor_formulations = [
        {
            "element_id": int(element_id),
            "formulation_id": str(_static_formulation_id(element) or ""),
            "algebraic_coordinate_policy": str(
                getattr(element, "dynamic_algebraic_policy", "")
            ),
        }
        for element_id, element in sorted(model.mesh.elements.items())
        if str(getattr(element, "dynamic_algebraic_policy", ""))
    ]

    if K_red.shape[0] == 0:
        diagnostics = {"status": "empty_reduced_system"}
        result_case = make_result_case(
            name="modal",
            analysis_type="modal",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
            metadata=(
                {
                    "descriptor_modal_provenance": {
                        "policy_id": DESCRIPTOR_MODAL_POLICY_ID,
                        "elements": descriptor_formulations,
                    }
                }
                if descriptor_formulations
                else None
            ),
        ).to_dict()
        exact_guard(model, context="solve_free_vibration output")
        return ModalResult([], num_modes, "empty_reduced_system", constraint_info, nullspace_info, assembly_info, diagnostics, result_case)

    K_sym = _sym(K_red)
    M_sym = _sym(M_red)
    n_red = int(K_sym.shape[0])
    descriptor_elements: Tuple[int, ...] = ()
    descriptor_modal = False
    descriptor_certificate = None
    state_dependent_transient_cache = (
        FactorizationCache(name="modal_state_dependent_transient", max_entries=2)
        if current_state or prestress_states is not None
        else None
    )
    eigen_started = time.perf_counter()
    try:
        sparse_diagnostics: Dict[str, Any] = {}
        descriptor_elements = declared_algebraic_mass_elements(model)
        descriptor_modal = bool(descriptor_elements)
        if descriptor_modal:
            descriptor_basis = build_declared_algebraic_basis(
                model,
                M,
                M_sym,
                T,
                independent_dofs,
                dense_size_limit=dense_size_limit,
            )
            descriptor_certificate = descriptor_basis.diagnostics
            descriptor = solve_descriptor_spectrum(
                K_sym,
                M_sym,
                num_modes=num_modes,
                dense_size_limit=dense_size_limit,
                algebraic_nullity=int(descriptor_basis.reduced_basis.shape[1]),
                algebraic_basis=descriptor_basis.reduced_basis,
                target_shift=shift,
                factorization_cache=(
                    state_dependent_transient_cache
                    if current_state or prestress_states is not None
                    else (
                        factorization_cache
                        or (
                            session.factorization_cache
                            if session is not None
                            else None
                        )
                    )
                ),
            )
            eigenvalues = descriptor.eigenvalues
            eigenvectors = descriptor.eigenvectors
            sparse_diagnostics = dict(descriptor.diagnostics)
            sparse_diagnostics["declared_algebraic_element_ids"] = list(
                descriptor_elements
            )
            sparse_diagnostics["declared_algebraic_formulations"] = (
                descriptor_formulations
            )
            sparse_diagnostics["declared_algebraic_mass_certificate"] = (
                descriptor_certificate
            )
            solver_kind = str(descriptor.diagnostics["solver"])
        elif n_red <= dense_size_limit or n_red <= num_modes + 1:
            eigenvalues, eigenvectors = _dense_eigensolve(K_sym, M_sym)
            solver_kind = "dense_scipy_eigh"
        else:
            eigenvalues, eigenvectors, sparse_diagnostics = _sparse_eigensolve(
                K_sym,
                M_sym,
                num_modes,
                shift,
                factorization_cache=(
                    state_dependent_transient_cache
                    if current_state or prestress_states is not None
                    else (
                        factorization_cache
                        or (
                            session.factorization_cache
                            if session is not None
                            else None
                        )
                    )
                ),
                _post_factorization_guard=lambda: exact_guard(
                    model,
                    context="solve_free_vibration shift factorization",
                ),
                use_zero_shift_inverse=bool(
                    shift is None
                    and not current_state
                    and prestress_states is None
                    and not Q.shape[1]
                ),
            )
            solver_kind = "sparse_scipy_eigsh"
    except Exception as exc:
        diagnostics = {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, AlgebraicDynamicsError):
            diagnostics.update(
                {
                    "error_code": "ALGEBRAIC_DESCRIPTOR_INVALID",
                    "policy_id": DESCRIPTOR_MODAL_POLICY_ID,
                    "declared_algebraic_element_ids": list(descriptor_elements),
                    "declared_algebraic_formulations": descriptor_formulations,
                }
            )
        result_case = make_result_case(
            name="modal",
            analysis_type="modal",
            assembly_info=assembly_info,
            solver_info={"convergence_info": diagnostics},
            recovery={"modes": num_modes},
            settings=settings,
            metadata=(
                {
                    "descriptor_modal_provenance": {
                        "policy_id": DESCRIPTOR_MODAL_POLICY_ID,
                        "elements": descriptor_formulations,
                    }
                }
                if descriptor_formulations
                else None
            ),
        ).to_dict()
        exact_guard(model, context="solve_free_vibration output")
        return ModalResult([], num_modes, "failed", constraint_info, nullspace_info, assembly_info, diagnostics, result_case)

    eigensolve_checkpoint_is_exact_noop = cancellation_checkpoint(
        "modal.after_eigensolve"
    )
    eigensolve_seconds = float(time.perf_counter() - eigen_started)
    phase_timings["factorization"] = float(
        sparse_diagnostics.get("factorization_lookup_seconds", 0.0)
    )
    phase_timings["eigen_iterations"] = max(
        eigensolve_seconds - phase_timings["factorization"],
        0.0,
    )
    if eigensolve_checkpoint_is_exact_noop:
        trusted_guard(
            model,
            context="solve_free_vibration cancellation after eigensolve",
        )
    else:
        exact_guard(
            model,
            context="solve_free_vibration cancellation after eigensolve",
        )

    recovery_started = time.perf_counter()
    order = np.argsort(np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    residual_batch_started = time.perf_counter()
    trusted_guard(model, context="solve_free_vibration residual batch preflight")
    stiffness_vectors = np.asarray(K_sym @ eigenvectors, dtype=float)
    mass_vectors = np.asarray(M_sym @ eigenvectors, dtype=float)
    trusted_guard(model, context="solve_free_vibration residual batch output")
    residual_batch_seconds = float(time.perf_counter() - residual_batch_started)
    phase_timings["residual_batch"] = residual_batch_seconds
    stiffness_operator_norm = (
        float(sparse_linalg.norm(K_sym)) if descriptor_modal else 0.0
    )
    mass_operator_norm = float(sparse_linalg.norm(M_sym)) if descriptor_modal else 0.0
    if Q.shape[1]:
        mass_times_rigid = np.asarray(M_sym @ Q, dtype=float)
        rigid_mass_gram = np.asarray(Q.T @ mass_times_rigid, dtype=float)
        rigid_mass_inverse = np.linalg.pinv(
            0.5 * (rigid_mass_gram + rigid_mass_gram.T), rcond=1.0e-12
        )
    else:
        mass_times_rigid = np.zeros((n_red, 0), dtype=float)
        rigid_mass_inverse = np.zeros((0, 0), dtype=float)

    modes: List[ModalMode] = []
    descriptor_backward_errors: List[float] = []
    for vector_index, (value, vector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        recovery_checkpoint_is_exact_noop = cancellation_checkpoint(
            f"modal.recovery:{len(modes) + 1}"
        )
        if not recovery_checkpoint_is_exact_noop:
            exact_guard(
                model,
                context="solve_free_vibration cancellation during recovery",
            )
        else:
            trusted_guard(
                model,
                context="solve_free_vibration cancellation during recovery",
            )
        if len(modes) >= num_modes:
            break
        if not np.isfinite(value):
            continue
        reduced = np.asarray(vector, dtype=float).reshape(-1)
        stiffness_vector = stiffness_vectors[:, vector_index]
        mass_vector = mass_vectors[:, vector_index]
        modal_mass = float(reduced @ mass_vector)
        if modal_mass <= eigen_tolerance:
            continue
        normalization = 1.0 / np.sqrt(modal_mass)
        reduced = reduced * normalization
        stiffness_vector = stiffness_vector * normalization
        mass_vector = mass_vector * normalization
        signed = _deterministic_sign(reduced)
        if signed is not reduced:
            reduced = signed
            stiffness_vector = -stiffness_vector
            mass_vector = -mass_vector
        modal_mass = float(reduced @ mass_vector)
        raw_modal_stiffness = float(reduced @ stiffness_vector)
        if descriptor_modal:
            # The descriptor solver may obtain the certified finite value from
            # a statically condensed quotient.  Recomputing x^T K x in a
            # strongly sheared algebraic coordinate system can catastrophically
            # cancel even when the full residual is at componentwise roundoff.
            modal_stiffness = float(value)
        else:
            modal_stiffness = raw_modal_stiffness
        eig = (
            float(modal_stiffness)
            if descriptor_modal
            else (
                max(float(value), 0.0)
                if abs(float(value)) <= eigen_tolerance
                else float(value)
            )
        )
        reduced_norm = float(np.linalg.norm(reduced))
        # Normalized whole-rigid-subspace measure: the fraction of modal mass
        # captured by the mass-orthogonal projection onto span(Q).  Unlike a
        # per-coefficient test against a Euclidean basis, it is invariant under
        # orthogonal re-mixing of Q and under mass-matrix rescaling.
        if Q.shape[1]:
            coefficients = rigid_mass_inverse @ (mass_times_rigid.T @ reduced)
            projected = Q @ coefficients
            projected_mass = float(projected @ (M_sym @ projected))
            rigid_corr = float(
                np.sqrt(max(projected_mass, 0.0) / max(modal_mass, np.finfo(float).tiny))
            )
        else:
            rigid_corr = 0.0
        rigid_corr = min(max(rigid_corr, 0.0), 1.0)
        omega = float(np.sqrt(max(eig, 0.0)))
        frequency = omega / (2.0 * np.pi)
        residual = np.asarray(
            stiffness_vector - eig * mass_vector,
            dtype=float,
        ).reshape(-1)
        denominator = max(
            float(np.linalg.norm(stiffness_vector))
            + abs(eig) * float(np.linalg.norm(mass_vector)),
            1.0,
        )
        if descriptor_modal:
            backward_denominator = max(
                (stiffness_operator_norm + abs(eig) * mass_operator_norm)
                * reduced_norm,
                1.0,
            )
            descriptor_backward_errors.append(
                float(np.linalg.norm(residual) / backward_denominator)
            )
        residual_norm = float(np.linalg.norm(residual) / denominator)
        is_rigid = bool(frequency <= rigid_body_frequency_tolerance or rigid_corr > 0.90)
        full = np.asarray(T @ reduced, dtype=float).reshape(-1)
        modes.append(
            ModalMode(
                mode_number=len(modes) + 1,
                eigenvalue=eig,
                angular_frequency=omega,
                frequency_hz=frequency,
                period=None if frequency <= 0.0 else 1.0 / frequency,
                mode_shape=full,
                reduced_mode_shape=reduced,
                modal_mass=modal_mass,
                modal_stiffness=modal_stiffness,
                residual_norm=residual_norm,
                rigid_body_correlation=rigid_corr,
                is_rigid_body=is_rigid,
            )
        )

    status = "ok" if modes else "no_modes"
    phase_timings["residual_checks_and_mode_recovery"] = float(
        time.perf_counter() - recovery_started
    )
    phase_timings["total"] = float(time.perf_counter() - operation_started)
    diagnostics = {
        "status": status,
        "thread_policy": thread_policy_diagnostics(resource_config),
        "solver": solver_kind,
        **sparse_diagnostics,
        "phase_timings_seconds": phase_timings,
        "spectral_guard_diagnostics": {
            "full_guard_count": int(guard_counts["full"]),
            "trusted_guard_count": int(guard_counts["trusted"]),
            "residual_batch_columns": int(eigenvectors.shape[1]),
        },
        "matrix_diagnostics": {
            "stiffness_reduced_shape": [int(value) for value in K_sym.shape],
            "stiffness_reduced_nnz": int(K_sym.nnz),
            "mass_reduced_nnz": int(M_sym.nnz),
            "stiffness_reduced_storage_bytes": int(
                K_sym.data.nbytes + K_sym.indices.nbytes + K_sym.indptr.nbytes
            ),
            "mass_reduced_storage_bytes": int(
                M_sym.data.nbytes + M_sym.indices.nbytes + M_sym.indptr.nbytes
            ),
        },
        "max_residual_norm": max((mode.residual_norm for mode in modes), default=0.0),
        "mass_orthogonality_error": _orthogonality_error(modes, M_sym),
        "num_rigid_body_modes": int(sum(1 for mode in modes if mode.is_rigid_body)),
        "constraint_postcheck": constraint_residual_summary(
            model,
            np.column_stack([mode.mode_shape for mode in modes])
            if modes
            else np.zeros((model.mesh.dof_manager.total_dofs, 0), dtype=float),
            homogeneous_variation=True,
        ),
    }
    if descriptor_modal:
        diagnostics["descriptor_modal"] = True
        diagnostics["max_normwise_backward_error"] = max(
            descriptor_backward_errors, default=0.0
        )
    if session is not None:
        diagnostics["analysis_session"] = session.diagnostics()
        exact_guard(model, context="solve_free_vibration session diagnostics")
    result_case = make_result_case(
        name="modal",
        analysis_type="modal",
        assembly_info=assembly_info,
        solver_info={"convergence_info": diagnostics},
        recovery={"modes": num_modes, "num_modes_returned": len(modes)},
        settings=settings,
        metadata=(
            {
                "descriptor_modal_provenance": {
                    "policy_id": DESCRIPTOR_MODAL_POLICY_ID,
                    "elements": descriptor_formulations,
                }
            }
            if descriptor_modal
            else None
        ),
    ).to_dict()
    emit_progress(
        progress_callback,
        "modal_complete",
        "modal.complete",
        completed=len(modes),
        total=num_modes,
        status=status,
        num_modes_returned=len(modes),
    )
    exact_guard(model, context="solve_free_vibration output")
    return ModalResult(modes, num_modes, status, constraint_info, nullspace_info, assembly_info, diagnostics, result_case)


def solve_free_vibration(
    model: "FEModel",
    num_modes: int = 6,
    shift: Optional[float] = None,
    dense_size_limit: int = 200,
    eigen_tolerance: float = 1.0e-9,
    rigid_body_frequency_tolerance: float = 1.0e-6,
    factorization_cache: Optional[FactorizationCache] = None,
    resource_config: Optional[ResourceConfig] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[ProgressCallback] = None,
    session: Optional["AnalysisSession"] = None,
    prestress_states: Optional[Any] = None,
    current_state_displacements: Optional[Any] = None,
    current_state_element_states: Optional[Any] = None,
    current_state_num_layers: int = 5,
) -> ModalResult:
    """Solve free vibration under one non-renewable qualified-family lease."""

    exact_guard = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD
    run_under_lease = _run_with_qualified_assembly_runtime_lease
    own_resource_config = _owned_resource_config_snapshot
    solve_under_lease = _solve_free_vibration_under_lease
    exact_guard(
        model,
        context="solve_free_vibration preflight",
    )

    def operation(lease: Any) -> ModalResult:
        def post_observation() -> None:
            exact_guard(
                model,
                context="solve_free_vibration resource configuration",
            )
            lease(
                model,
                context="solve_free_vibration resource configuration",
            )

        owned_resource_config = own_resource_config(
            resource_config,
            post_observation=post_observation,
        )
        return solve_under_lease(
            model,
            num_modes=num_modes,
            shift=shift,
            dense_size_limit=dense_size_limit,
            eigen_tolerance=eigen_tolerance,
            rigid_body_frequency_tolerance=rigid_body_frequency_tolerance,
            factorization_cache=factorization_cache,
            resource_config=owned_resource_config,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
            session=session,
            prestress_states=prestress_states,
            current_state_displacements=current_state_displacements,
            current_state_element_states=current_state_element_states,
            current_state_num_layers=current_state_num_layers,
            _qualified_runtime_guard=lease,
        )

    return run_under_lease(
        model,
        context="solve_free_vibration",
        operation=operation,
    )
