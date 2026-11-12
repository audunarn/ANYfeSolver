"""Read-only assembly of exact qualified committed tangent components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import math
import sys
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from scipy import linalg as scipy_linalg
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

import anymaterial.contract as _anymaterial_contract_module
import anymaterial.curves as _anymaterial_curves_module
import anymaterial.yield_criteria as _anymaterial_yield_module

from . import _native_rotation_state as _native_rotation_state_module
from . import _q4_assumed_strain_plasticity as _q4_assumed_strain_module
from . import corotational as _corotational_module
from . import e4_pl_s3_initial_fields as _s3_initial_fields_module
from . import e4_pl_s3_state as _s3_state_module
from . import e4_pl_s3_v2d_element as _s3_v2d_element_module
from . import e4_pl_s3_v2d_state as _s3_v2d_state_module
from . import elements as _elements_module
from . import material_curves as _material_curves_module
from . import materials as _materials_module
from . import plasticity as _plasticity_module
from . import shell_sections as _shell_sections_module

from .e4_pl_element import (
    IMPLEMENTATION_ID as QUALIFIED_Q4_IMPLEMENTATION_ID,
    Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID,
    Q4_CURRENT_STATE_BINDING_SCHEMA_ID,
    Q4_CURRENT_STATE_PROJECTION_POLICY_ID,
    Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID,
    Q4_QUADRATURE_AUTHORITY_ID,
    QualifiedE4PLShellElement,
    _QUALIFIED_Q4_BASE_LOCAL_FRAME_KERNEL,
    _QUALIFIED_Q4_BASE_MITC4_SHEAR_KERNEL,
    _QUALIFIED_Q4_BASE_NONLINEAR_KERNEL,
    _QUALIFIED_Q4_BASE_SERIALIZATION_KERNEL,
    _QUALIFIED_Q4_BASE_STIFFNESS_KERNEL,
    _QUALIFIED_Q4_CLASS_NAMESPACE_AUTHORITY,
    _QUALIFIED_Q4_MODULE_DATA_AUTHORITY,
    _QUALIFIED_Q4_MODULE_FUNCTION_AUTHORITY,
    _validate_q4_serialization_authority,
)
from .e4_pl_s3_element import (
    CURRENT_STATE_BUBBLE_PROJECTION_POLICY_ID,
    CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID,
    S3_QUADRATURE_AUTHORITY_ID,
    QualifiedE4PLS3ShellElement,
    _S3_BASE_SERIALIZATION_KERNEL,
    _QUALIFIED_S3_CLASS_NAMESPACE_AUTHORITY,
    _QUALIFIED_S3_MODULE_DATA_AUTHORITY,
    _QUALIFIED_S3_MODULE_FUNCTION_AUTHORITY,
    _validate_s3_serialization_authority,
)
from .e4_pl_s3_state import (
    _module_authority_signature,
    canonical_json_bytes,
)
from .e4_pl_s3_v2d_element import (
    FORMULATION_ID as _S3_V2D_DECLARED_FORMULATION_ID,
    IMPLEMENTATION_ID as S3_V2D_IMPLEMENTATION_ID,
    QUADRATURE_AUTHORITY_ID as S3_V2D_QUADRATURE_AUTHORITY_ID,
    V2D_DERIVED_CACHE_NAMES as _S3_V2D_DERIVED_CACHE_NAMES,
    NativeParityE4PLS3V2DShellElement,
    _V2D_LEASE_EPOCHS_KEY,
    _s3_v2d_runtime_epoch_manager as _S3_V2D_RUNTIME_EPOCH_MANAGER,
    _subscribe_v2d_lease_epoch,
    _V2DLeaseEpoch,
)
from .e4_pl_s3_v2d_state import (
    canonical_json_bytes as _v2d_canonical_json_bytes,
)
from .element_capabilities import ElementCapabilityError
from .elements import ShellElement
from .matrix_assembly import (
    _activity_scales,
    _run_with_qualified_assembly_runtime_lease,
)
from .nonlinear_state import (
    NonlinearStateStore,
    begin_state_evaluation,
    create_model_native_rotation_store,
    discard_active_state_candidate,
)

if TYPE_CHECKING:
    from .fe_core import FEModel


QUALIFIED_Q4_FORMULATION_ID = "E4_PL_QUALIFIED_Q4_HYBRID_V2"
QUALIFIED_S3_FORMULATION_ID = "E4_PL_QUALIFIED_S3_COMPANION_V1"
S3_V2D_FORMULATION_ID = "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"
if _S3_V2D_DECLARED_FORMULATION_ID != S3_V2D_FORMULATION_ID:
    raise RuntimeError("S3 V2D current-state formulation authority mismatch")
S3_V2D_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID = (
    "S3_V2D_EICR_SYMMETRIC_MATERIAL_FORCE_GEOMETRIC_HESSIAN_SPLIT_V1"
)
S3_V2D_CURRENT_STATE_PROJECTION_POLICY_ID = (
    "S3_V2D_EICR_NO_INTERNAL_BUBBLE_PROJECTION_V1"
)
COMMITTED_CURRENT_TANGENT_ASSEMBLY_POLICY_ID = (
    "Q4_ADDITIVE_VON_KARMAN_S3_NATIVE_TL_COMMITTED_COMPONENT_ASSEMBLY_V1"
)
COMMITTED_CURRENT_TANGENT_ROUTE_POLICY_ID = (
    "EXACT_FORMULATION_ID_AND_NATIVE_COMPONENT_API_ROUTE_V1"
)
COMMITTED_CURRENT_TANGENT_INPUT_OWNERSHIP_POLICY_ID = (
    "SINGLE_DETACHED_BINARY64_DISPLACEMENT_AND_CANONICAL_STATE_SNAPSHOT_V1"
)
CURRENT_STATE_EIGEN_ACTIVITY_POLICY_ID = (
    "Q4_S3_CURRENT_STATE_EIGEN_EXACT_ACTIVE_LIFECYCLE_ONLY_V1"
)

# ``copy.deepcopy`` asks ``copyreg`` for a class's slot layout.  CPython
# memoizes that derived layout by adding exactly ``__slotnames__`` to the
# class namespace, including an empty list for ordinary dataclasses.  The
# cache is not executable mechanics authority and may appear at any point in
# an otherwise read-only process, so exclude only that exact stdlib cache key
# from both sides of every class-namespace comparison.
_IGNORED_STDLIB_CLASS_CACHE_NAME = "__slotnames__"


def _capture_numerical_module_authority(
    module: ModuleType,
    names: tuple[str, ...],
) -> Mapping[str, Any]:
    """Freeze the shallow numerical namespace used by exact shell routes."""

    namespace = vars(module)
    return MappingProxyType(
        {
            "module": module,
            "module_name": str(module.__name__),
            "bindings": MappingProxyType(
                {
                    str(name): namespace[name]
                    for name in names
                }
            ),
        }
    )


_QUALIFIED_NUMERICAL_MODULE_AUTHORITY = tuple(
    _capture_numerical_module_authority(module, names)
    for module, names in (
        (
            np,
            (
                "abs", "add", "all", "allclose", "any", "arange", "arccos",
                "arctan2", "argmax", "argsort", "array", "array_equal",
                "asarray", "ascontiguousarray", "average", "block",
                "broadcast_to", "ceil", "clip", "column_stack",
                "concatenate", "cos", "count_nonzero", "cross", "deg2rad",
                "degrees", "diag", "dot", "dtype", "einsum", "empty",
                "empty_like", "errstate", "eye", "finfo", "flatnonzero",
                "frombuffer", "fromiter", "full", "interp", "isclose",
                "isfinite", "isinf", "isnan", "issubdtype", "ix_",
                "lexsort", "linspace", "max", "linalg", "maximum",
                "mean", "median", "meshgrid", "min", "moveaxis",
                "nextafter", "ones", "ones_like", "outer", "power",
                "radians", "real", "repeat", "sin", "sort", "sqrt", "sum",
                "swapaxes", "tile", "trace", "unique", "vstack", "where",
                "zeros", "zeros_like",
            ),
        ),
        (
            np.linalg,
            (
                "cholesky", "cond", "det", "eigh", "eigvalsh", "inv",
                "lstsq", "norm", "pinv", "qr", "solve", "svd",
                "LinAlgError",
            ),
        ),
        (
            scipy_linalg,
            (
                "LinAlgError", "cholesky", "eig", "eigh", "eigvalsh",
                "qr", "solve", "solve_triangular", "svdvals",
            ),
        ),
        (
            sparse,
            (
                "bmat", "coo_matrix", "csc_matrix", "csr_matrix", "diags",
                "issparse", "linalg", "spmatrix",
            ),
        ),
        (
            sparse_linalg,
            ("LinearOperator", "eigsh", "lobpcg", "norm", "splu"),
        ),
    )
)


def _capture_dependency_module_authority(module: ModuleType) -> Mapping[str, Any]:
    """Freeze bounded solver/material dependencies used by qualified mechanics."""

    namespace = vars(module)
    callable_bindings = MappingProxyType(
        {
            name: value
            for name, value in tuple(namespace.items())
            if callable(value) or isinstance(value, ModuleType)
        }
    )
    data_bindings = MappingProxyType(
        {
            name: (type(value), _module_authority_signature(value))
            for name, value in tuple(namespace.items())
            if name.lstrip("_").isupper()
        }
    )
    class_namespaces = MappingProxyType(
        {
            value: MappingProxyType(
                {
                    name: member
                    for name, member in dict(
                        type.__getattribute__(value, "__dict__")
                    ).items()
                    if name != _IGNORED_STDLIB_CLASS_CACHE_NAME
                }
            )
            for value in tuple(namespace.values())
            if isinstance(value, type)
            and type.__getattribute__(value, "__module__") == module.__name__
        }
    )
    return MappingProxyType(
        {
            "module": module,
            "callable_bindings": callable_bindings,
            "data_bindings": data_bindings,
            "class_namespaces": class_namespaces,
        }
    )


_Q4_DEPENDENCY_MODULE_AUTHORITY = tuple(
    _capture_dependency_module_authority(module)
    for module in (
        _elements_module,
        _q4_assumed_strain_module,
        _plasticity_module,
        _shell_sections_module,
        _materials_module,
        _material_curves_module,
        _s3_state_module,
        _s3_initial_fields_module,
        _anymaterial_contract_module,
        _anymaterial_curves_module,
        _anymaterial_yield_module,
    )
)
_S3_DEPENDENCY_MODULE_AUTHORITY = tuple(
    _capture_dependency_module_authority(module)
    for module in (
        _elements_module,
        _native_rotation_state_module,
        _plasticity_module,
        _shell_sections_module,
        _materials_module,
        _material_curves_module,
        _s3_state_module,
        _s3_initial_fields_module,
        _anymaterial_contract_module,
        _anymaterial_curves_module,
        _anymaterial_yield_module,
    )
)

_S3_V2D_MODULE_AUTHORITY = _capture_dependency_module_authority(
    _s3_v2d_element_module
)
_S3_V2D_DEPENDENCY_MODULE_AUTHORITY = tuple(
    _capture_dependency_module_authority(module)
    for module in (
        _corotational_module,
        _s3_v2d_state_module,
    )
)


def _validate_s3_v2d_serialization_authority(
    element: Any,
    *,
    expected_class: type[Any],
) -> None:
    if type(element) is not expected_class:
        raise TypeError("S3 V2D serialization requires its exact class")
    element._validate_configuration()
    payload = element.to_dict()
    restored = expected_class.from_dict(payload)
    if type(restored) is not expected_class or restored.to_dict() != payload:
        raise ValueError("S3 V2D serialization round trip changed its fingerprint")


def _s3_v2d_committed_current_tangent_components(
    element: Any,
    model: "FEModel",
    material: Any,
    committed_u_elem: Any,
    committed_state: Mapping[str, Any],
    num_layers: int = 5,
) -> Mapping[str, Any]:
    """Evaluate one immutable V2D EICR energy-Hessian decomposition."""

    if type(element) is not NativeParityE4PLS3V2DShellElement:
        raise TypeError("S3 V2D current tangent requires its exact class")
    if not isinstance(committed_state, Mapping):
        raise TypeError("S3 V2D current tangent requires a committed mapping")
    if committed_state.get("qualified_s3_activity_disposition") is not None:
        raise ElementCapabilityError(
            "S3 V2D current-state eigen analysis requires ACTIVE state"
        )
    displacement = np.asarray(committed_u_elem, dtype=np.float64)
    if displacement.shape != (18,) or not np.all(np.isfinite(displacement)):
        raise ValueError("S3 V2D current tangent requires a finite 18-vector")
    validated = element.validate_model_bound_nonlinear_state(
        model.mesh,
        material,
        committed_state,
        int(num_layers),
        expected_committed_total_u=displacement,
    )
    before = _v2d_canonical_json_bytes(committed_state)
    if validated.get("experimental_von_karman_law") is not None:
        raise ElementCapabilityError(
            "Experimental S3 V2D von Karman current-state eigen decomposition "
            "is unavailable; use loading tangents with the fixed predecessor"
        )
    from .corotational import corotational_element_response_components

    force, material_tangent, geometric_tangent, _discarded_state = (
        corotational_element_response_components(
            model,
            int(element.element_id),
            element,
            displacement,
            committed_state,
            int(num_layers),
        )
    )
    after = _v2d_canonical_json_bytes(committed_state)
    if before != after:
        raise RuntimeError("S3 V2D current tangent mutated committed state")
    material_array = np.asarray(material_tangent, dtype=np.float64)
    geometric_array = np.asarray(geometric_tangent, dtype=np.float64)
    total_array = material_array + geometric_array
    force_array = np.asarray(force, dtype=np.float64)
    if (
        force_array.shape != (18,)
        or material_array.shape != (18, 18)
        or geometric_array.shape != (18, 18)
        or total_array.shape != (18, 18)
        or not np.all(np.isfinite(force_array))
        or not np.all(np.isfinite(total_array))
    ):
        raise ValueError("S3 V2D current tangent returned incompatible arrays")
    scale = max(float(np.linalg.norm(total_array, ord="fro")), 1.0)
    decomposition_error = float(
        np.linalg.norm(total_array - material_array - geometric_array, ord="fro")
        / scale
    )
    symmetry_error = max(
        float(np.linalg.norm(value - value.T, ord="fro") / scale)
        for value in (material_array, geometric_array, total_array)
    )
    if (
        decomposition_error > 256.0 * np.finfo(np.float64).eps
        or symmetry_error > 512.0 * np.finfo(np.float64).eps
    ):
        raise ValueError("S3 V2D current tangent violates decomposition symmetry")
    readonly: Dict[str, Any] = {
        "decomposition_policy_id": (
            S3_V2D_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
        ),
        "force": force_array.copy(),
        "geometric": geometric_array.copy(),
        "material": material_array.copy(),
        "projection_policy_id": S3_V2D_CURRENT_STATE_PROJECTION_POLICY_ID,
        "relative_decomposition_error": decomposition_error,
        "relative_symmetry_error": symmetry_error,
        "state_digest": str(validated["state_integrity_sha256"]),
        "state_storage": "transient_read_only_not_persisted",
        "total": total_array.copy(),
    }
    for name in ("force", "geometric", "material", "total"):
        readonly[name].setflags(write=False)
    return MappingProxyType(readonly)


_QUALIFIED_PROFILES = MappingProxyType(
    {
        QUALIFIED_Q4_FORMULATION_ID: MappingProxyType(
            {
                "formulation_id": QUALIFIED_Q4_FORMULATION_ID,
                "family": "qualified_q4",
                "node_count": 4,
                "native_rotation_required": False,
                "kinematics": "additive_rotation_von_karman",
                "reference_surface_offset_scope": "q4_zero_offset_only",
                "element_type": QualifiedE4PLShellElement,
                "component_api": (
                    QualifiedE4PLShellElement.compute_committed_current_tangent_components
                ),
                "component_api_receives_model": False,
                "serialization_validator": _validate_q4_serialization_authority,
                "state_validator": (
                    QualifiedE4PLShellElement.validate_committed_current_tangent_state
                ),
                "implementation_id": QUALIFIED_Q4_IMPLEMENTATION_ID,
                "class_identity": MappingProxyType(
                    {
                        "implementation_id": QUALIFIED_Q4_IMPLEMENTATION_ID,
                        "current_state_binding_schema_id": Q4_CURRENT_STATE_BINDING_SCHEMA_ID,
                        "current_state_tangent_decomposition_policy_id": Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID,
                        "current_state_projection_policy_id": Q4_CURRENT_STATE_PROJECTION_POLICY_ID,
                        "quadrature_authority_id": Q4_QUADRATURE_AUTHORITY_ID,
                    }
                ),
                "decomposition_policy_id": (
                    Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
                ),
                "projection_policy_id": Q4_CURRENT_STATE_PROJECTION_POLICY_ID,
                "state_binding_schema_id": Q4_CURRENT_STATE_BINDING_SCHEMA_ID,
                "algorithmic_origin_schema_id": (
                    Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
                ),
                "state_displacement_binding": "qualified_q4_nested_seal",
                "component_binding_flag_required": True,
                "component_algorithmic_origin_flag_required": True,
                "quadrature_authority_id": Q4_QUADRATURE_AUTHORITY_ID,
                "critical_apis": MappingProxyType(
                    {
                        "get_dof_mapping": ShellElement.get_dof_mapping,
                        "compute_committed_current_tangent_components": QualifiedE4PLShellElement.compute_committed_current_tangent_components,
                        "_compute_committed_current_tangent_components_unchecked": QualifiedE4PLShellElement._compute_committed_current_tangent_components_unchecked,
                        "validate_committed_current_tangent_binding": QualifiedE4PLShellElement.validate_committed_current_tangent_binding,
                        "validate_committed_current_tangent_semantics": QualifiedE4PLShellElement.validate_committed_current_tangent_semantics,
                        "validate_committed_current_tangent_state": QualifiedE4PLShellElement.validate_committed_current_tangent_state,
                        "_validate_committed_state_at_configuration": QualifiedE4PLShellElement._validate_committed_state_at_configuration,
                        "_validate_committed_current_kinematics": QualifiedE4PLShellElement._validate_committed_current_kinematics,
                        "_replay_accepted_algorithmic_response": QualifiedE4PLShellElement._replay_accepted_algorithmic_response,
                        "_qualified_linear_correction": QualifiedE4PLShellElement._qualified_linear_correction,
                        "_current_state_cache_transaction": QualifiedE4PLShellElement._current_state_cache_transaction,
                        "_committed_current_binding_payload": QualifiedE4PLShellElement._committed_current_binding_payload,
                        "_accepted_algorithmic_update_fingerprint": QualifiedE4PLShellElement._accepted_algorithmic_update_fingerprint,
                        "_stable_state_identity_payload": QualifiedE4PLShellElement._stable_state_identity_payload,
                        "_validated_algorithmic_origin": QualifiedE4PLShellElement._validated_algorithmic_origin,
                        "_requires_algorithmic_return_map_origin": QualifiedE4PLShellElement._requires_algorithmic_return_map_origin,
                        "_reject_noncurrent_activity_disposition": QualifiedE4PLShellElement._reject_noncurrent_activity_disposition,
                        "validate_noncurrent_deleted_state": QualifiedE4PLShellElement.validate_noncurrent_deleted_state,
                        "_deleted_frozen_disposition_payload": QualifiedE4PLShellElement._deleted_frozen_disposition_payload,
                        "compute_stiffness_matrix": QualifiedE4PLShellElement.compute_stiffness_matrix,
                        "compute_stiffness_components": QualifiedE4PLShellElement.compute_stiffness_components,
                        "_constitutive_and_drill_stiffness": QualifiedE4PLShellElement._constitutive_and_drill_stiffness,
                        "_qualified_stiffness_cache_key": QualifiedE4PLShellElement._qualified_stiffness_cache_key,
                        "_bind_qualified_component_guard": QualifiedE4PLShellElement._bind_qualified_component_guard,
                        "_warped_generalized_drilling_correction": QualifiedE4PLShellElement._warped_generalized_drilling_correction,
                        "_generalized_section_in_frame": QualifiedE4PLShellElement._generalized_section_in_frame,
                        "_physical_director_context": QualifiedE4PLShellElement._physical_director_context,
                        "_material_angle": ShellElement._material_angle,
                        "compute_mass_matrix": QualifiedE4PLShellElement.compute_mass_matrix,
                        "compute_geometric_stiffness_matrix": QualifiedE4PLShellElement.compute_geometric_stiffness_matrix,
                        "compute_nonlinear_response": QualifiedE4PLShellElement.compute_nonlinear_response,
                        "_nonlinear_geometry": QualifiedE4PLShellElement._nonlinear_geometry,
                        "get_node_coordinates": QualifiedE4PLShellElement.get_node_coordinates,
                        "to_dict": QualifiedE4PLShellElement.to_dict,
                        "compute_shape_functions": ShellElement.compute_shape_functions,
                        "_compute_3node_shape_functions": ShellElement._compute_3node_shape_functions,
                        "_compute_4node_shape_functions": ShellElement._compute_4node_shape_functions,
                        "_compute_6node_shape_functions": ShellElement._compute_6node_shape_functions,
                        "_compute_8node_shape_functions": ShellElement._compute_8node_shape_functions,
                        "compute_jacobian": ShellElement.compute_jacobian,
                        "_local_frame_and_derivatives": QualifiedE4PLShellElement._local_frame_and_derivatives,
                        "_inverse_planar_jacobian": QualifiedE4PLShellElement._inverse_planar_jacobian,
                        "_local_dof_transform": ShellElement._local_dof_transform,
                        "_center_frame": ShellElement._center_frame,
                        "_reference_center": ShellElement._reference_center,
                        "_build_shell_b_matrices": ShellElement._build_shell_b_matrices,
                        "_build_drilling_b_matrix": ShellElement._build_drilling_b_matrix,
                        "_mitc4_shear_samples": ShellElement._mitc4_shear_samples,
                        "_mitc4_shear_b_matrix": QualifiedE4PLShellElement._mitc4_shear_b_matrix,
                        "validate_quadrature_authority": QualifiedE4PLShellElement.validate_quadrature_authority,
                        "gauss_points": ShellElement.gauss_points,
                        "gauss_weights": ShellElement.gauss_weights,
                        "shear_gauss_points": ShellElement.shear_gauss_points,
                        "shear_gauss_weights": ShellElement.shear_gauss_weights,
                    }
                ),
                "base_critical_apis": MappingProxyType(
                    {
                        "_local_frame_and_derivatives": _QUALIFIED_Q4_BASE_LOCAL_FRAME_KERNEL,
                        "_mitc4_shear_b_matrix": _QUALIFIED_Q4_BASE_MITC4_SHEAR_KERNEL,
                        "to_dict": _QUALIFIED_Q4_BASE_SERIALIZATION_KERNEL,
                        "compute_nonlinear_response": _QUALIFIED_Q4_BASE_NONLINEAR_KERNEL,
                    }
                ),
                "module_function_authority": _QUALIFIED_Q4_MODULE_FUNCTION_AUTHORITY,
                "module_data_authority": _QUALIFIED_Q4_MODULE_DATA_AUTHORITY,
                "class_namespace_authority": _QUALIFIED_Q4_CLASS_NAMESPACE_AUTHORITY,
                "module_name": QualifiedE4PLShellElement.__module__,
                "dependency_module_authority": _Q4_DEPENDENCY_MODULE_AUTHORITY,
            }
        ),
        QUALIFIED_S3_FORMULATION_ID: MappingProxyType(
            {
                "formulation_id": QUALIFIED_S3_FORMULATION_ID,
                "family": "qualified_s3",
                "node_count": 3,
                "native_rotation_required": True,
                "kinematics": "native_multiplicative_total_lagrangian",
                "reference_surface_offset_scope": "s3_native_signed_offset",
                "element_type": QualifiedE4PLS3ShellElement,
                "component_api": (
                    QualifiedE4PLS3ShellElement.compute_committed_current_tangent_components
                ),
                "component_api_receives_model": False,
                "serialization_validator": _validate_s3_serialization_authority,
                "state_validator": (
                    QualifiedE4PLS3ShellElement.validate_model_bound_nonlinear_state
                ),
                "implementation_id": None,
                "class_identity": MappingProxyType(
                    {
                        "formulation_native_total_lagrangian": True,
                        "quadrature_authority_id": S3_QUADRATURE_AUTHORITY_ID,
                    }
                ),
                "decomposition_policy_id": (
                    CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
                ),
                "projection_policy_id": (
                    CURRENT_STATE_BUBBLE_PROJECTION_POLICY_ID
                ),
                "state_binding_schema_id": None,
                "algorithmic_origin_schema_id": None,
                "state_displacement_binding": "committed_total_u",
                "component_binding_flag_required": False,
                "component_algorithmic_origin_flag_required": False,
                "quadrature_authority_id": S3_QUADRATURE_AUTHORITY_ID,
                "critical_apis": MappingProxyType(
                    {
                        "get_dof_mapping": ShellElement.get_dof_mapping,
                        "compute_committed_current_tangent_components": QualifiedE4PLS3ShellElement.compute_committed_current_tangent_components,
                        "compute_nonlinear_response": QualifiedE4PLS3ShellElement.compute_nonlinear_response,
                        "_validated_native_nonlinear_trial": QualifiedE4PLS3ShellElement._validated_native_nonlinear_trial,
                        "init_model_bound_nonlinear_state": QualifiedE4PLS3ShellElement.init_model_bound_nonlinear_state,
                        "validate_model_bound_nonlinear_state": QualifiedE4PLS3ShellElement.validate_model_bound_nonlinear_state,
                        "_validate_model_bound_nonlinear_state_core": QualifiedE4PLS3ShellElement._validate_model_bound_nonlinear_state_core,
                        "_reject_noncurrent_activity_disposition": QualifiedE4PLS3ShellElement._reject_noncurrent_activity_disposition,
                        "validate_noncurrent_deleted_state": QualifiedE4PLS3ShellElement.validate_noncurrent_deleted_state,
                        "_validated_activity_core": QualifiedE4PLS3ShellElement._validated_activity_core,
                        "_activity_core_identity_payload": QualifiedE4PLS3ShellElement._activity_core_identity_payload,
                        "_deleted_frozen_disposition_payload": QualifiedE4PLS3ShellElement._deleted_frozen_disposition_payload,
                        "_model_bound_nonlinear_context": QualifiedE4PLS3ShellElement._model_bound_nonlinear_context,
                        "_model_bound_generalized_nonlinear_context": QualifiedE4PLS3ShellElement._model_bound_generalized_nonlinear_context,
                        "_initial_fields_in_physical_director_convention": QualifiedE4PLS3ShellElement._initial_fields_in_physical_director_convention,
                        "_gap": QualifiedE4PLS3ShellElement._gap,
                        "native_reference_directors": QualifiedE4PLS3ShellElement.native_reference_directors,
                        "compute_stiffness_components": QualifiedE4PLS3ShellElement.compute_stiffness_components,
                        "_compute_stiffness_components": QualifiedE4PLS3ShellElement._compute_stiffness_components,
                        "compute_mass_matrix": QualifiedE4PLS3ShellElement.compute_mass_matrix,
                        "compute_mass_components": QualifiedE4PLS3ShellElement.compute_mass_components,
                        "_compute_mass_components": QualifiedE4PLS3ShellElement._compute_mass_components,
                        "dynamic_algebraic_directions": QualifiedE4PLS3ShellElement.dynamic_algebraic_directions,
                        "_local_dof_transform": QualifiedE4PLS3ShellElement._local_dof_transform,
                        "_constitutive": QualifiedE4PLS3ShellElement._constitutive,
                        "_director_generalized_transform": QualifiedE4PLS3ShellElement._director_generalized_transform,
                        "_generalized_section_in_frame": ShellElement._generalized_section_in_frame,
                        "_material_angle": ShellElement._material_angle,
                        "get_node_coordinates": QualifiedE4PLS3ShellElement.get_node_coordinates,
                        "to_dict": QualifiedE4PLS3ShellElement.to_dict,
                        "capability_gaps": QualifiedE4PLS3ShellElement.capability_gaps,
                        "capability_restrictions": QualifiedE4PLS3ShellElement.capability_restrictions,
                        "validate_quadrature_authority": QualifiedE4PLS3ShellElement.validate_quadrature_authority,
                        "gauss_points": QualifiedE4PLS3ShellElement.gauss_points,
                        "gauss_weights": QualifiedE4PLS3ShellElement.gauss_weights,
                        "shear_gauss_points": QualifiedE4PLS3ShellElement.shear_gauss_points,
                        "shear_gauss_weights": QualifiedE4PLS3ShellElement.shear_gauss_weights,
                    }
                ),
                "base_critical_apis": MappingProxyType(
                    {"to_dict": _S3_BASE_SERIALIZATION_KERNEL}
                ),
                "module_function_authority": _QUALIFIED_S3_MODULE_FUNCTION_AUTHORITY,
                "module_data_authority": _QUALIFIED_S3_MODULE_DATA_AUTHORITY,
                "class_namespace_authority": _QUALIFIED_S3_CLASS_NAMESPACE_AUTHORITY,
                "module_name": QualifiedE4PLS3ShellElement.__module__,
                "dependency_module_authority": _S3_DEPENDENCY_MODULE_AUTHORITY,
            }
        ),
        S3_V2D_FORMULATION_ID: MappingProxyType(
            {
                "formulation_id": S3_V2D_FORMULATION_ID,
                "family": "qualified_s3_v2d",
                "node_count": 3,
                "native_rotation_required": False,
                "kinematics": "element_independent_corotational_energy_hessian",
                "reference_surface_offset_scope": "s3_v2d_signed_offset",
                "element_type": NativeParityE4PLS3V2DShellElement,
                "component_api": _s3_v2d_committed_current_tangent_components,
                "component_api_receives_model": True,
                "serialization_validator": _validate_s3_v2d_serialization_authority,
                "state_validator": (
                    NativeParityE4PLS3V2DShellElement.validate_model_bound_nonlinear_state
                ),
                "implementation_id": S3_V2D_IMPLEMENTATION_ID,
                "class_identity": MappingProxyType(
                    {
                        "formulation_native_total_lagrangian": False,
                        "implementation_id": S3_V2D_IMPLEMENTATION_ID,
                        "quadrature_authority_id": S3_V2D_QUADRATURE_AUTHORITY_ID,
                    }
                ),
                "decomposition_policy_id": (
                    S3_V2D_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
                ),
                "projection_policy_id": S3_V2D_CURRENT_STATE_PROJECTION_POLICY_ID,
                "state_binding_schema_id": None,
                "algorithmic_origin_schema_id": None,
                "state_displacement_binding": "committed_total_u",
                "component_binding_flag_required": False,
                "component_algorithmic_origin_flag_required": False,
                "quadrature_authority_id": S3_V2D_QUADRATURE_AUTHORITY_ID,
                "critical_apis": MappingProxyType(
                    {
                        "get_dof_mapping": ShellElement.get_dof_mapping,
                        "compute_nonlinear_response": NativeParityE4PLS3V2DShellElement.compute_nonlinear_response,
                        "init_model_bound_nonlinear_state": NativeParityE4PLS3V2DShellElement.init_model_bound_nonlinear_state,
                        "validate_model_bound_nonlinear_state": NativeParityE4PLS3V2DShellElement.validate_model_bound_nonlinear_state,
                        "compute_stiffness_components": NativeParityE4PLS3V2DShellElement.compute_stiffness_components,
                        "compute_mass_matrix": NativeParityE4PLS3V2DShellElement.compute_mass_matrix,
                        "compute_mass_components": NativeParityE4PLS3V2DShellElement.compute_mass_components,
                        "dynamic_algebraic_directions": NativeParityE4PLS3V2DShellElement.dynamic_algebraic_directions,
                        "get_node_coordinates": ShellElement.get_node_coordinates,
                        "to_dict": NativeParityE4PLS3V2DShellElement.to_dict,
                        "from_dict": NativeParityE4PLS3V2DShellElement.from_dict.__func__,
                        "validate_quadrature_authority": NativeParityE4PLS3V2DShellElement.validate_quadrature_authority,
                        "gauss_points": NativeParityE4PLS3V2DShellElement.gauss_points,
                        "gauss_weights": NativeParityE4PLS3V2DShellElement.gauss_weights,
                        "shear_gauss_points": NativeParityE4PLS3V2DShellElement.shear_gauss_points,
                        "shear_gauss_weights": NativeParityE4PLS3V2DShellElement.shear_gauss_weights,
                    }
                ),
                "base_critical_apis": MappingProxyType({}),
                "module_function_authority": _S3_V2D_MODULE_AUTHORITY[
                    "callable_bindings"
                ],
                "module_data_authority": _S3_V2D_MODULE_AUTHORITY["data_bindings"],
                "class_namespace_authority": _S3_V2D_MODULE_AUTHORITY[
                    "class_namespaces"
                ],
                "module_name": NativeParityE4PLS3V2DShellElement.__module__,
                "dependency_module_authority": (
                    _S3_V2D_DEPENDENCY_MODULE_AUTHORITY
                ),
            }
        ),
    }
)

_QUALIFIED_PROFILE_CAPTURED_CLASS_NAMES = MappingProxyType(
    {
        formulation_id: frozenset().union(
            *(
                frozenset(namespace)
                for namespace in profile["class_namespace_authority"].values()
            )
        )
        for formulation_id, profile in _QUALIFIED_PROFILES.items()
    }
)


def _static_mro_attribute(owner: type[Any], name: str) -> Any:
    """Return a class attribute without invoking a runtime-replaced descriptor."""

    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            value = namespace[name]
            if isinstance(value, (classmethod, staticmethod)):
                return value.__func__
            return value
    return None


def _changed_critical_apis(
    owner: type[Any],
    profile: Mapping[str, Any],
    static_lookup: Any,
) -> tuple[str, ...]:
    """Critical APIs whose class-level definition is not the frozen one."""

    return tuple(
        sorted(
            str(name)
            for name, expected in profile["critical_apis"].items()
            if static_lookup(owner, str(name)) is not expected
        )
    )


def _changed_base_critical_apis(
    base_critical_apis: Mapping[str, Any],
    static_lookup: Any,
) -> tuple[str, ...]:
    """Base-class critical APIs that are not the frozen ones."""

    return tuple(
        sorted(
            str(name)
            for name, expected in base_critical_apis.items()
            if static_lookup(ShellElement, str(name)) is not expected
        )
    )


def _capture_serialization_module_guards(
    profiles: Mapping[str, Mapping[str, Any]],
) -> Mapping[Any, Any]:
    """Freeze the module-level guard each serialization validator runs itself.

    The Q4 and S3 validators re-run one element-independent module guard for
    every element; the same guard object they bind by default is captured here
    so one lifecycle validation call can run it once.  A validator without such
    a keyword (S3 V2D) has no entry and keeps its own per-element behaviour.
    """

    guards: dict[Any, Any] = {}
    for profile in profiles.values():
        validator = profile["serialization_validator"]
        defaults = getattr(validator, "__kwdefaults__", None)
        guard = defaults.get("_module_guard") if type(defaults) is dict else None
        if callable(guard):
            guards[validator] = guard
    return MappingProxyType(guards)


_SERIALIZATION_MODULE_GUARDS = _capture_serialization_module_guards(
    _QUALIFIED_PROFILES
)


def _run_guard_once_per_call(guard: Any) -> Any:
    """Wrap an element-independent guard so one call runs it once.

    A failing guard is not marked done: every later element runs it again and
    fails the same way, as before.
    """

    done = False

    def once(*args: Any, **kwargs: Any) -> None:
        nonlocal done
        if not done:
            guard(*args, **kwargs)
            done = True

    return once


class _ScanClassSnapshot:
    """Class- and module-level facts read once per lifecycle validation call.

    One instance lives for exactly one call of the exact lifecycle guard and is
    never stored.  Every instance-level check and every read that is specific
    to one element (its namespace, values, callables, the element-specific
    validators) still runs for every element, and each class-level read stays
    at its original position inside the element check.  What the snapshot
    reuses within one call is what all elements of one class share: attribute
    lookups on that class, the two critical-API comparisons, and the
    serialization module guard.  A change to a *base* class made after the
    first element of a class was checked is therefore reported by the next
    call, not by the remaining elements of this one; a change to the Q4 or S3
    class itself is still reported by the remaining elements through their own
    final-class authority.  Only the qualified Q4 and S3 families take a
    snapshot; S3 V2D elements and custom profile-failure callbacks keep the
    per-element boundary.

    The memos key on ``id()`` (a hostile metaclass ``__hash__`` must not run
    here) and keep the keyed class alive, so an id cannot be reused within one
    call.
    """

    __slots__ = (
        "_lookup",
        "_lookup_cache",
        "_critical",
        "_base_critical",
        "_guards",
    )

    def __init__(self, lookup: Any) -> None:
        self._lookup = lookup
        self._lookup_cache: dict[tuple[int, str], tuple[Any, Any]] = {}
        self._critical: dict[tuple[int, int], tuple[Any, Any, Any]] = {}
        self._base_critical: dict[int, tuple[Any, Any]] = {}
        self._guards: dict[Any, Any] = {}

    def lookup(self, owner: type[Any], name: str) -> Any:
        key = (id(owner), name)
        entry = self._lookup_cache.get(key)
        if entry is None:
            entry = self._lookup_cache[key] = (owner, self._lookup(owner, name))
        return entry[1]

    def changed_critical(
        self,
        owner: type[Any],
        profile: Mapping[str, Any],
    ) -> tuple[str, ...]:
        key = (id(owner), id(profile))
        entry = self._critical.get(key)
        if entry is None:
            entry = self._critical[key] = (
                owner,
                profile,
                _changed_critical_apis(owner, profile, self.lookup),
            )
        return entry[2]

    def changed_base_critical(
        self,
        profile: Mapping[str, Any],
    ) -> tuple[str, ...]:
        key = id(profile)
        entry = self._base_critical.get(key)
        if entry is None:
            entry = self._base_critical[key] = (
                profile,
                _changed_base_critical_apis(
                    profile["base_critical_apis"], self.lookup
                ),
            )
        return entry[1]

    def serialization_module_guard(self, validator: Any) -> Any:
        """The once-per-call module guard for *validator*, or ``None``."""

        guard = self._guards.get(validator)
        if guard is None:
            real = _SERIALIZATION_MODULE_GUARDS.get(validator)
            guard = self._guards[validator] = (
                False if real is None else _run_guard_once_per_call(real)
            )
        return guard or None


def _qualified_profile_api_failure(
    element: Any,
    profile: Mapping[str, Any],
    *,
    _captured_class_names: frozenset[str] | None = None,
    _static_lookup: Any = _static_mro_attribute,
    _serialization_static_lookup: Any = None,
    _snapshot: _ScanClassSnapshot | None = None,
) -> str | None:
    """Return one exact API-authority failure without evaluating mechanics."""

    expected_formulation_id = str(profile["formulation_id"])
    owner = type(element)
    static_formulation_id = _static_lookup(
        owner, "formulation_id"
    )
    if (
        owner is not profile["element_type"]
        or type(static_formulation_id) is not str
        or static_formulation_id != expected_formulation_id
    ):
        return f"{expected_formulation_id}:FORMULATION_ID_CLASS_MISMATCH"
    try:
        instance_namespace = object.__getattribute__(element, "__dict__")
    except AttributeError:
        instance_namespace = {}
    if type(instance_namespace) is not dict:
        return f"{expected_formulation_id}:INSTANCE_NAMESPACE_MISMATCH"
    captured_class_names = (
        set().union(
            *(
                set(namespace)
                for namespace in profile["class_namespace_authority"].values()
            )
        )
        if _captured_class_names is None
        else _captured_class_names
    )
    class_data_shadows = tuple(
        sorted(set(instance_namespace).intersection(captured_class_names))
    )
    if class_data_shadows:
        return (
            f"{expected_formulation_id}:CLASS_NAMESPACE_INSTANCE_SHADOW="
            + ",".join(class_data_shadows)
        )
    critical_apis = profile["critical_apis"]
    base_critical_apis = profile["base_critical_apis"]
    for name in ("element_id", "node_ids", "material_name"):
        if name not in instance_namespace:
            return f"{expected_formulation_id}:MISSING_INSTANCE_DATA={name}"
        if _static_lookup(type(element), name) is not None:
            return f"{expected_formulation_id}:INSTANCE_DATA_CLASS_SHADOW={name}"
    raw_element_id = instance_namespace["element_id"]
    raw_node_ids = instance_namespace["node_ids"]
    raw_material_name = instance_namespace["material_name"]
    if (
        type(raw_element_id) is not int
        or type(raw_node_ids) is not tuple
        or len(raw_node_ids) != int(profile["node_count"])
        or not all(type(value) is int for value in raw_node_ids)
        or len(set(raw_node_ids)) != len(raw_node_ids)
        or type(raw_material_name) is not str
    ):
        return f"{expected_formulation_id}:INSTANCE_DATA_VALUE_MISMATCH"
    if _static_lookup(type(element), "reference_surface_offset") is not None:
        return f"{expected_formulation_id}:OFFSET_SCOPE_MISMATCH"
    if profile["family"] in {"qualified_s3", "qualified_s3_v2d"}:
        if "reference_surface_offset" not in instance_namespace:
            return f"{expected_formulation_id}:OFFSET_SCOPE_MISMATCH"
        raw_offset = instance_namespace["reference_surface_offset"]
        if type(raw_offset) is not float or not math.isfinite(raw_offset):
            return f"{expected_formulation_id}:OFFSET_SCOPE_MISMATCH"
    elif "reference_surface_offset" in instance_namespace:
        raw_offset = instance_namespace["reference_surface_offset"]
        if type(raw_offset) is not float or raw_offset != 0.0:
            return f"{expected_formulation_id}:OFFSET_SCOPE_MISMATCH"
    for name, expected in profile["class_identity"].items():
        actual = _static_lookup(type(element), str(name))
        if (
            name in instance_namespace
            or type(actual) is not type(expected)
            or actual != expected
        ):
            if name == "implementation_id":
                label = "IMPLEMENTATION_MISMATCH"
            elif name == "current_state_binding_schema_id":
                label = "BINDING_SCHEMA_MISMATCH"
            elif name in {
                "current_state_tangent_decomposition_policy_id",
                "current_state_projection_policy_id",
            }:
                label = "COMPONENT_POLICY_MISMATCH"
            else:
                label = f"CLASS_IDENTITY_MISMATCH={name}"
            return f"{expected_formulation_id}:{label}"
    if "formulation_id" in instance_namespace:
        return f"{expected_formulation_id}:FORMULATION_ID_INSTANCE_SHADOW"
    shadowed_critical = tuple(
        sorted(set(instance_namespace).intersection(critical_apis))
    )
    if shadowed_critical:
        return (
            f"{expected_formulation_id}:CRITICAL_INSTANCE_SHADOW="
            + ",".join(shadowed_critical)
        )
    callable_instance_overrides = tuple(
        sorted(
            str(name)
            for name, value in instance_namespace.items()
            if callable(value)
        )
    )
    if callable_instance_overrides:
        return (
            f"{expected_formulation_id}:CALLABLE_INSTANCE_OVERRIDE="
            + ",".join(callable_instance_overrides)
        )
    changed_critical = (
        _changed_critical_apis(type(element), profile, _static_lookup)
        if _snapshot is None
        else _snapshot.changed_critical(type(element), profile)
    )
    if changed_critical:
        return (
            f"{expected_formulation_id}:CRITICAL_API_MISMATCH="
            + ",".join(changed_critical)
        )
    changed_base_critical = (
        _changed_base_critical_apis(base_critical_apis, _static_lookup)
        if _snapshot is None
        else _snapshot.changed_base_critical(profile)
    )
    if changed_base_critical:
        return (
            f"{expected_formulation_id}:BASE_CRITICAL_API_MISMATCH="
            + ",".join(changed_base_critical)
        )
    try:
        validator = profile["serialization_validator"]
        validator_options: dict[str, Any] = {}
        if _serialization_static_lookup is not None:
            validator_options["_static_lookup"] = _serialization_static_lookup
        if _snapshot is not None:
            module_guard = _snapshot.serialization_module_guard(validator)
            if module_guard is not None:
                validator_options["_module_guard"] = module_guard
        validator(
            element,
            expected_class=profile["element_type"],
            **validator_options,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return (
            f"{expected_formulation_id}:CONFIGURATION_AUTHORITY={exc}"
        )
    try:
        critical_apis["validate_quadrature_authority"](element)
    except (AttributeError, TypeError, ValueError) as exc:
        return f"{expected_formulation_id}:QUADRATURE_AUTHORITY={exc}"
    return None


def _call_local_static_lookup() -> Any:
    """Reuse live namespace views, never previously validated member values.

    Each lookup still reads the current MRO and current dictionary contents.
    Mapping proxies are live views: additions, removals and replacements remain
    visible immediately. A changed MRO rebuilds the views. Nothing survives
    the enclosing lifecycle validation call, and descriptors are never invoked.
    The exact lifecycle scan may put a per-call ``_ScanClassSnapshot`` in front
    of this lookup for the qualified Q4 and S3 families (see its docstring).
    """
    views: dict[int, tuple[Any, tuple[Any, ...]]] = {}

    def lookup(owner: type[Any], name: str) -> Any:
        mro = type.__getattribute__(owner, "__mro__")
        key = id(owner)
        record = views.get(key)
        if record is None or record[0] is not mro:
            record = (mro, tuple(type.__getattribute__(base, "__dict__") for base in mro))
            views[key] = record
        for namespace in record[1]:
            if name in namespace:
                value = namespace[name]
                if isinstance(value, (classmethod, staticmethod)):
                    return value.__func__
                return value
        return None

    return lookup


def _require_exact_qualified_component_lifecycle_api_implementation(
    model: "FEModel",
    *,
    context: str,
    _profiles: Mapping[str, Mapping[str, Any]] = _QUALIFIED_PROFILES,
    _static_lookup: Any = _static_mro_attribute,
    _profile_failure: Any = _qualified_profile_api_failure,
    _preparable_profile_failure: Any = _qualified_profile_api_failure,
    _default_profiles: Mapping[str, Mapping[str, Any]] = _QUALIFIED_PROFILES,
    _lookup_factory: Any = _call_local_static_lookup,
    _captured_class_names_by_formulation: Mapping[str, frozenset[str]] = (
        _QUALIFIED_PROFILE_CAPTURED_CLASS_NAMES
    ),
    _authority_signer: Any = _module_authority_signature,
    _mapping_type: Any = Mapping,
    _capability_error: Any = ElementCapabilityError,
    _sys_module: Any = sys,
    _numerical_authority: tuple[Mapping[str, Any], ...] = (
        _QUALIFIED_NUMERICAL_MODULE_AUTHORITY
    ),
) -> Dict[str, Any]:
    """Guard exact Q4/S3 APIs while admitting unrelated generic elements."""

    elements = getattr(getattr(model, "mesh", None), "elements", None)
    if not isinstance(elements, _mapping_type):
        raise _capability_error(
            f"{context} requires a model element mapping"
        )
    profile_by_type = {
        profile["element_type"]: profile
        for profile in _profiles.values()
    }
    qualified_ids: list[int] = []
    failures: list[tuple[int, str]] = []
    admitted_profiles: Dict[str, tuple[int, Mapping[str, Any]]] = {}
    candidates: list[tuple[int, Any, Mapping[str, Any]]] = []
    for raw_element_id, element in tuple(elements.items()):
        static_formulation_id = _static_lookup(
            type(element), "formulation_id"
        )
        profile = (
            _profiles.get(static_formulation_id)
            if type(static_formulation_id) is str
            else None
        )
        if profile is None:
            profile = profile_by_type.get(type(element))
        if profile is None:
            owner_mro = type.__getattribute__(type(element), "__mro__")
            profile = next(
                (
                    candidate
                    for candidate in _profiles.values()
                    if candidate["element_type"] in owner_mro
                ),
                None,
            )
        if profile is None:
            continue
        if type(raw_element_id) is not int:
            failures.append((-1, "QUALIFIED_ELEMENT_MAPPING_KEY_MISMATCH"))
            continue
        element_id = raw_element_id
        qualified_ids.append(element_id)
        admitted_profiles.setdefault(
            str(profile["formulation_id"]), (element_id, profile)
        )
        candidates.append((element_id, element, profile))
    qualified_ids.sort()
    candidates.sort(key=lambda item: item[0])
    changed_numerical: list[str] = []
    if candidates:
        for authority in _numerical_authority:
            numerical_module = authority["module"]
            numerical_name = str(authority["module_name"])
            if _sys_module.modules.get(numerical_name) is not numerical_module:
                changed_numerical.append(f"{numerical_name}.__module__")
                continue
            numerical_namespace = vars(numerical_module)
            changed_numerical.extend(
                f"{numerical_name}.{name}"
                for name, expected in authority["bindings"].items()
                if numerical_namespace.get(name) is not expected
            )
    if changed_numerical:
        first_id = int(candidates[0][0])
        failures.append(
            (
                first_id,
                "NUMERICAL_AUTHORITY_MISMATCH="
                + ",".join(sorted(changed_numerical)),
            )
        )
    for _formulation_id, (element_id, profile) in admitted_profiles.items():
        if failures:
            break
        family = str(profile["family"])
        authority_module = _sys_module.modules.get(str(profile["module_name"]))
        authority_namespace = (
            {} if authority_module is None else vars(authority_module)
        )
        changed_helpers = tuple(
            sorted(
                str(name)
                for name, expected in profile[
                    "module_function_authority"
                ].items()
                if authority_namespace.get(str(name)) is not expected
            )
        )
        if changed_helpers:
            failures.append(
                (
                    element_id,
                    f"{family}:MODULE_HELPER_MISMATCH="
                    + ",".join(changed_helpers),
                )
            )
            continue
        changed_data = tuple(
            sorted(
                str(name)
                for name, (expected_type, expected) in profile["module_data_authority"].items()
                if name not in authority_namespace
                or type(authority_namespace[name]) is not expected_type
                or _authority_signer(authority_namespace[name])
                != expected
            )
        )
        if changed_data:
            failures.append(
                (
                    element_id,
                    f"{family}:MODULE_DATA_MISMATCH="
                    + ",".join(changed_data),
                )
            )
            continue
        changed_dependencies: list[str] = []
        for dependency in profile["dependency_module_authority"]:
            dependency_module = dependency["module"]
            dependency_name = str(dependency_module.__name__)
            if _sys_module.modules.get(dependency_name) is not dependency_module:
                changed_dependencies.append(f"{dependency_name}.__module__")
                continue
            dependency_namespace = vars(dependency_module)
            changed_dependencies.extend(
                f"{dependency_name}.{name}"
                for name, expected in dependency["callable_bindings"].items()
                if dependency_namespace.get(name) is not expected
            )
            changed_dependencies.extend(
                f"{dependency_name}.{name}"
                for name, (expected_type, expected_signature) in dependency[
                    "data_bindings"
                ].items()
                if name not in dependency_namespace
                or type(dependency_namespace[name]) is not expected_type
                or _authority_signer(dependency_namespace[name])
                != expected_signature
            )
            for owner, expected_namespace in dependency[
                "class_namespaces"
            ].items():
                actual_namespace = type.__getattribute__(owner, "__dict__")
                owner_name = str(type.__getattribute__(owner, "__qualname__"))
                actual_names = set(actual_namespace) - {
                    _IGNORED_STDLIB_CLASS_CACHE_NAME
                }
                expected_names = set(expected_namespace) - {
                    _IGNORED_STDLIB_CLASS_CACHE_NAME
                }
                changed_members = sorted(
                    actual_names.symmetric_difference(expected_names)
                    | {
                        name
                        for name, expected in expected_namespace.items()
                        if name != _IGNORED_STDLIB_CLASS_CACHE_NAME
                        if name in actual_namespace
                        and actual_namespace[name] is not expected
                    }
                )
                changed_dependencies.extend(
                    f"{dependency_name}.{owner_name}.{name}"
                    for name in changed_members
                )
        if changed_dependencies:
            failures.append(
                (
                    element_id,
                    f"{family}:DEPENDENCY_AUTHORITY_MISMATCH="
                    + ",".join(sorted(changed_dependencies)),
                )
            )
            continue
        changed_namespaces: list[str] = []
        for owner, expected_namespace in profile[
            "class_namespace_authority"
        ].items():
            actual_namespace = type.__getattribute__(owner, "__dict__")
            owner_name = str(type.__getattribute__(owner, "__qualname__"))
            actual_names = set(actual_namespace) - {
                _IGNORED_STDLIB_CLASS_CACHE_NAME
            }
            expected_names = set(expected_namespace) - {
                _IGNORED_STDLIB_CLASS_CACHE_NAME
            }
            changed_members = sorted(
                actual_names.symmetric_difference(expected_names)
                | {
                    name
                    for name, expected in expected_namespace.items()
                    if name != _IGNORED_STDLIB_CLASS_CACHE_NAME
                    if name in actual_namespace
                    and actual_namespace[name] is not expected
                }
            )
            changed_namespaces.extend(
                f"{owner_name}.{name}" for name in changed_members
            )
        if changed_namespaces:
            labels = ["CLASS_NAMESPACE_MISMATCH", "CRITICAL_API_MISMATCH"]
            if any(
                name.endswith(".formulation_id")
                for name in changed_namespaces
            ):
                labels.append("FORMULATION_ID_CLASS_MISMATCH")
            if any(
                name.endswith(".implementation_id")
                for name in changed_namespaces
            ):
                labels.append("IMPLEMENTATION_MISMATCH")
            if any(
                name.startswith("ShellElement.")
                for name in changed_namespaces
            ):
                labels.append("BASE_CRITICAL_API_MISMATCH")
            failures.append(
                (
                    element_id,
                    f"{family}:" + ":".join(labels) + "="
                    + ",".join(sorted(changed_namespaces)),
                )
            )
    if not failures:
        call_static_lookup = _lookup_factory()
        # Only the exact default profile check takes a snapshot, and only for
        # the qualified Q4 and S3 families.  Any other callback and every S3
        # V2D element keep the per-element boundary and their own signature.
        snapshot = (
            _ScanClassSnapshot(call_static_lookup)
            if _profile_failure is _qualified_profile_api_failure
            else None
        )
        for element_id, element, profile in candidates:
            formulation_id = str(profile["formulation_id"])
            if (
                _profile_failure is _preparable_profile_failure
                and profile is _default_profiles.get(formulation_id)
            ):
                element_snapshot = (
                    snapshot
                    if profile["family"] in {"qualified_q4", "qualified_s3"}
                    else None
                )
                element_lookup = (
                    call_static_lookup
                    if element_snapshot is None
                    else element_snapshot.lookup
                )
                snapshot_options = (
                    {}
                    if element_snapshot is None
                    else {"_snapshot": element_snapshot}
                )
                failure = _profile_failure(
                    element,
                    profile,
                    _captured_class_names=(
                        _captured_class_names_by_formulation[formulation_id]
                    ),
                    _static_lookup=element_lookup,
                    _serialization_static_lookup=(
                        element_lookup
                        if profile["family"] in {"qualified_q4", "qualified_s3"}
                        else None
                    ),
                    **snapshot_options,
                )
            else:
                failure = _profile_failure(element, profile)
            if failure is None:
                instance_namespace = object.__getattribute__(
                    element,
                    "__dict__",
                )
                if instance_namespace.get("element_id") != element_id:
                    failure = (
                        f"{profile['formulation_id']}:"
                        "ELEMENT_MAPPING_ID_MISMATCH"
                    )
            if failure is not None:
                failures.append((element_id, failure))
    if failures:
        detail = "; ".join(
            f"{element_id} ({reason})"
            for element_id, reason in failures[:8]
        )
        raise _capability_error(
            f"{context} requires exact qualified component/lifecycle APIs; "
            f"incompatible element IDs {detail}"
        )
    return {
        "qualified_element_ids": qualified_ids,
        "guarded": bool(qualified_ids),
    }


def _bind_exact_qualified_component_lifecycle_guard(
    implementation: Any,
) -> Any:
    """Expose the guard without a mutable module-lookup dependency."""

    def guard(
        model: "FEModel",
        *,
        context: str,
    ) -> Dict[str, Any]:
        return implementation(model, context=context)

    guard.__name__ = "require_exact_qualified_component_lifecycle_api"
    guard.__qualname__ = "require_exact_qualified_component_lifecycle_api"
    guard.__doc__ = implementation.__doc__
    guard.__module__ = __name__
    guard.__annotations__ = {
        "model": "FEModel",
        "context": str,
        "return": Dict[str, Any],
    }
    guard.__signature__ = inspect.Signature(
        parameters=(
            inspect.Parameter(
                "model", inspect.Parameter.POSITIONAL_OR_KEYWORD
            ),
            inspect.Parameter(
                "context", inspect.Parameter.KEYWORD_ONLY, annotation=str
            ),
        ),
        return_annotation=Dict[str, Any],
    )
    return guard


require_exact_qualified_component_lifecycle_api = (
    _bind_exact_qualified_component_lifecycle_guard(
        _require_exact_qualified_component_lifecycle_api_implementation
    )
)
_EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD = (
    require_exact_qualified_component_lifecycle_api
)


class _QualifiedLifecycleProbeMesh:
    __slots__ = ("elements",)

    def __init__(self, elements: Mapping[int, Any]) -> None:
        self.elements = elements


class _QualifiedLifecycleProbeModel:
    """A private element subset observed by the exact lifecycle guard."""

    __slots__ = ("mesh",)

    def __init__(self, elements: Mapping[int, Any]) -> None:
        self.mesh = _QualifiedLifecycleProbeMesh(elements)


_V2D_TRUSTED_BOOKKEEPING_NAMES = frozenset(
    {
        "_qualified_direct_state_token",
        "_qualified_direct_state_tokens",
        _V2D_LEASE_EPOCHS_KEY,
    }
)
_V2D_TRUSTED_SCALAR_TYPES = frozenset({type(None), bool, int, float, str})


def _v2d_trusted_immutable_value(value: Any, *, depth: int = 0) -> bool:
    kind = type(value)
    if kind in _V2D_TRUSTED_SCALAR_TYPES:
        return True
    if kind is tuple:
        return depth < 2 and all(
            _v2d_trusted_immutable_value(item, depth=depth + 1)
            for item in value
        )
    if kind is np.ndarray:
        current: Any = value
        while type(current) is np.ndarray:
            if current.flags.writeable:
                return False
            current = current.base
        return type(current) is bytes
    return False


def _v2d_trusted_instance_eligible(element: Any) -> bool:
    """Whether only ordinary (counted) writes can change the instance.

    Every value the exact guard can observe must be immutable, so an in-place
    mutation cannot alter a validated fact without an attribute write.  The
    generic derived caches are data the exact guard never reads; their
    data-only resets deliberately are not counted as modifications.
    """

    try:
        namespace = object.__getattribute__(element, "__dict__")
    except AttributeError:
        return False
    if type(namespace) is not dict:
        return False
    for name, value in tuple(dict.items(namespace)):
        if type(name) is not str:
            return False
        if name in _V2D_TRUSTED_BOOKKEEPING_NAMES:
            continue
        if name in _S3_V2D_DERIVED_CACHE_NAMES:
            if value is None or type(value) is np.ndarray:
                continue
            return False
        if not _v2d_trusted_immutable_value(value):
            return False
    return True


def _capture_qualified_v2d_trusted_authority(
    element_items: Any,
    *,
    context: str,
    _exact_guard: Any = _require_exact_qualified_component_lifecycle_api_implementation,
    _v2d_type: type[Any] = NativeParityE4PLS3V2DShellElement,
    _v2d_manager: Any = _S3_V2D_RUNTIME_EPOCH_MANAGER,
) -> Any:
    """Capture a constant-time lifecycle check for owned S3 V2D elements.

    The exact guard validates each admitted formulation in two parts: module,
    data, dependency and class authority once per formulation, then each
    element instance.  This capture runs the complete guard over every V2D
    element once.  The returned check re-runs the complete guard over one
    representative element, which re-validates the whole formulation-level
    authority with the unchanged exact code, and requires the V2D class epoch
    to be unchanged.  The remaining per-instance facts are protected by a write
    counter private to this capture: every V2D element is subscribed to it, any
    ordinary write or deletion on a subscribed element advances it, and
    eligibility requires every observed value to be immutable.  Callers must
    separately require the owning mesh epoch (the qualified assembly lease's
    trusted state check does) for the element mapping and the nodes.

    Returns ``None`` when no V2D element is present, any V2D instance is not
    eligible or the capture scan fails; callers then keep the complete
    lifecycle scan.
    """

    if type(element_items) is not tuple:
        return None
    v2d_items = tuple(
        (element_id, element)
        for element_id, element in element_items
        if type(element) is _v2d_type
    )
    if not v2d_items or not all(
        type(element_id) is int and _v2d_trusted_instance_eligible(element)
        for element_id, element in v2d_items
    ):
        return None
    # Subscribe and capture first, then validate: a write that lands between
    # subscription and the end of the scan is counted and rejected below.
    epoch = _V2DLeaseEpoch()
    _subscribe_v2d_lease_epoch((element for _id, element in v2d_items), epoch)
    generation = _v2d_manager.capture_generation()
    try:
        _exact_guard(
            _QualifiedLifecycleProbeModel(dict(v2d_items)),
            context=f"{context} S3 V2D trusted capture",
        )
    except (AttributeError, ElementCapabilityError, TypeError, ValueError):
        # Declining keeps the complete scan, which then reports the failure
        # at the caller's own preflight boundary with its usual context.
        return None
    representative = _QualifiedLifecycleProbeModel(dict(v2d_items[:1]))

    def require_unchanged(context: str) -> None:
        try:
            _v2d_manager.require_generation(generation)
        except ValueError as exc:
            raise ElementCapabilityError(
                f"{context} requires exact qualified component/lifecycle APIs; "
                "S3 V2D class authority changed during the trusted lease"
            ) from exc
        if epoch.value != 0:
            raise ElementCapabilityError(
                f"{context} requires exact qualified component/lifecycle APIs; "
                "an S3 V2D element was modified during the trusted lease"
            )

    def require(*, context: str) -> None:
        require_unchanged(context)
        _exact_guard(representative, context=context)
        require_unchanged(context)

    require_unchanged(f"{context} S3 V2D trusted capture")
    return require


def _qualified_route(
    model: "FEModel",
    *,
    context: str,
    _exact_guard: Any = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
) -> Dict[str, Any]:
    """Classify the exact admitted family before evaluating element mechanics."""

    _exact_guard(model, context=context)
    elements = getattr(getattr(model, "mesh", None), "elements", None)
    if not isinstance(elements, Mapping) or not elements:
        raise ElementCapabilityError(
            f"{context} requires a nonempty qualified Q4/S3 element mapping"
        )
    element_profiles: Dict[int, Mapping[str, Any]] = {}
    element_data: Dict[int, Mapping[str, Any]] = {}
    unsupported: list[tuple[int, str]] = []
    for raw_element_id, element in sorted(elements.items()):
        element_id = int(raw_element_id)
        static_formulation_id = _static_mro_attribute(
            type(element), "formulation_id"
        )
        formulation_id = (
            static_formulation_id
            if type(static_formulation_id) is str
            else ""
        )
        profile = _QUALIFIED_PROFILES.get(formulation_id)
        if profile is None:
            unsupported.append((element_id, formulation_id or "UNDECLARED"))
            continue
        api_failure = _qualified_profile_api_failure(element, profile)
        if api_failure is not None:
            unsupported.append((element_id, api_failure))
            continue
        critical_apis = profile["critical_apis"]
        instance_namespace = vars(element)
        raw_owned_element_id = instance_namespace["element_id"]
        if (
            type(raw_owned_element_id) is not int
            or raw_owned_element_id != element_id
        ):
            unsupported.append((element_id, f"{formulation_id}:ID_MISMATCH"))
            continue
        raw_node_ids = instance_namespace["node_ids"]
        if type(raw_node_ids) is not tuple:
            unsupported.append((element_id, f"{formulation_id}:BAD_CONNECTIVITY"))
            continue
        node_ids = raw_node_ids
        if (
            len(node_ids) != int(profile["node_count"])
            or not all(type(value) is int for value in node_ids)
            or len(set(node_ids)) != len(node_ids)
        ):
            unsupported.append((element_id, f"{formulation_id}:BAD_CONNECTIVITY"))
            continue
        declared_native = bool(profile["native_rotation_required"])
        if declared_native != bool(profile["native_rotation_required"]):
            unsupported.append(
                (element_id, f"{formulation_id}:NATIVE_ROTATION_MISMATCH")
            )
            continue
        if profile["family"] == "qualified_q4":
            if (
                ShellElement.compute_nonlinear_response
                is not _QUALIFIED_Q4_BASE_NONLINEAR_KERNEL
                or ShellElement.compute_stiffness_matrix
                is not _QUALIFIED_Q4_BASE_STIFFNESS_KERNEL
            ):
                unsupported.append(
                    (element_id, f"{formulation_id}:BASE_NONLINEAR_API_MISMATCH")
                )
                continue
            if _static_mro_attribute(type(element), "reference_surface_offset") is not None:
                unsupported.append(
                    (element_id, f"{formulation_id}:OFFSET_SCOPE_MISMATCH")
                )
                continue
            raw_offset = instance_namespace.get("reference_surface_offset", 0.0)
            if type(raw_offset) is not float or raw_offset != 0.0:
                unsupported.append(
                    (element_id, f"{formulation_id}:OFFSET_SCOPE_MISMATCH")
                )
                continue
        raw_material_name = instance_namespace["material_name"]
        if type(raw_material_name) is not str:
            unsupported.append(
                (element_id, f"{formulation_id}:MATERIAL_NAME_MISMATCH")
            )
            continue
        reference_surface_offset = 0.0
        if profile["family"] in {"qualified_s3", "qualified_s3_v2d"}:
            if "reference_surface_offset" not in instance_namespace:
                unsupported.append(
                    (element_id, f"{formulation_id}:OFFSET_SCOPE_MISMATCH")
                )
                continue
            raw_s3_offset = instance_namespace["reference_surface_offset"]
            if type(raw_s3_offset) is not float or not math.isfinite(
                raw_s3_offset
            ):
                unsupported.append(
                    (element_id, f"{formulation_id}:OFFSET_SCOPE_MISMATCH")
                )
                continue
            reference_surface_offset = raw_s3_offset
        element_profiles[element_id] = profile
        element_data[element_id] = MappingProxyType(
            {
                "element_id": element_id,
                "node_ids": node_ids,
                "material_name": raw_material_name,
                "reference_surface_offset": reference_surface_offset,
            }
        )
    if unsupported:
        details = "; ".join(
            f"{element_id} ({formulation_id})"
            for element_id, formulation_id in unsupported[:8]
        )
        if len(unsupported) > 8:
            details += f"; and {len(unsupported) - 8} more"
        raise ElementCapabilityError(
            f"{context} admits only explicitly routed qualified Q4/S3 committed "
            f"component APIs; unsupported element IDs {details}"
        )

    _exact_guard(model, context=context)

    families = tuple(
        sorted({str(profile["family"]) for profile in element_profiles.values()})
    )
    route = (
        families[0]
        if len(families) == 1
        else "mixed_qualified_q4_s3"
    )
    formulation_counts = {
        formulation_id: sum(
            1
            for profile in element_profiles.values()
            if profile["formulation_id"] == formulation_id
        )
        for formulation_id in sorted(_QUALIFIED_PROFILES)
        if any(
            profile["formulation_id"] == formulation_id
            for profile in element_profiles.values()
        )
    }
    return {
        "route": route,
        "route_policy_id": COMMITTED_CURRENT_TANGENT_ROUTE_POLICY_ID,
        "families": list(families),
        "formulation_counts": formulation_counts,
        "element_profiles": element_profiles,
        "element_data": element_data,
        "native_rotation_required": any(
            bool(profile["native_rotation_required"])
            for profile in element_profiles.values()
        ),
        "kinematic_scope": {
            str(profile["family"]): str(profile["kinematics"])
            for profile in element_profiles.values()
        },
        "reference_surface_offset_scope": {
            str(profile["family"]): str(
                profile["reference_surface_offset_scope"]
            )
            for profile in element_profiles.values()
        },
        "quadrature_authority": {
            str(profile["family"]): str(profile["quadrature_authority_id"])
            for profile in element_profiles.values()
        },
    }


def _positive_layer_count(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("current_state_num_layers must be a positive integer")
    made = int(value)
    if made <= 0:
        raise ValueError("current_state_num_layers must be a positive integer")
    return made


def _guarded_owned_input_snapshot(
    model: "FEModel",
    value: Any,
    *,
    path: str,
    _exact_guard: Any,
) -> Any:
    """Detach one caller-owned tree and recheck authority after observation.

    Mapping iteration, sequence iteration, and NumPy's array protocol may run
    caller code.  No subsequent canonicalization or numerical operation may
    run until the exact qualified-family boundary has been re-established.
    """

    context = f"committed current tangent input observation at {path}"
    if isinstance(value, np.ndarray):
        observed = np.asarray(value)
        _exact_guard(model, context=context)
        made = np.ascontiguousarray(observed)
        return np.frombuffer(
            made.tobytes(order="C"), dtype=made.dtype
        ).reshape(made.shape)
    if isinstance(value, np.generic):
        observed = value.item()
        _exact_guard(model, context=context)
        return observed
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Mapping):
        observed_items = tuple(value.items())
        _exact_guard(model, context=context)
        result: dict[str, Any] = {}
        for key, member in observed_items:
            if type(key) is not str:
                raise TypeError(f"committed current tangent has a non-string key at {path}")
            if key in result:
                raise ValueError(
                    f"committed current tangent has duplicate key {key!r} at {path}"
                )
            result[key] = _guarded_owned_input_snapshot(
                model,
                member,
                path=f"{path}.{key}",
                _exact_guard=_exact_guard,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        observed_members = tuple(value)
        _exact_guard(model, context=context)
        return [
            _guarded_owned_input_snapshot(
                model,
                member,
                path=f"{path}[{index}]",
                _exact_guard=_exact_guard,
            )
            for index, member in enumerate(observed_members)
        ]
    raise TypeError(
        "committed current tangent has unsupported input type "
        f"{type(value).__name__} at {path}"
    )


def _normalized_exact_states(
    model: "FEModel",
    element_states: Any,
    *,
    _exact_guard: Any = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
) -> Dict[int, Mapping[str, Any]]:
    if not isinstance(element_states, Mapping):
        raise TypeError(
            "committed current tangent requires an element-state mapping"
        )
    normalized: Dict[int, Mapping[str, Any]] = {}
    observed_items = tuple(element_states.items())
    _exact_guard(
        model,
        context="committed current tangent element-state mapping observation",
    )
    for raw_element_id, state in observed_items:
        if isinstance(raw_element_id, (bool, np.bool_)):
            raise ValueError(
                "committed current tangent element-state IDs must be canonical integers"
            )
        if isinstance(raw_element_id, (int, np.integer)):
            element_id = int(raw_element_id)
        elif isinstance(raw_element_id, str) and raw_element_id:
            try:
                parsed_element_id = int(raw_element_id)
            except ValueError:
                parsed_element_id = None
            if (
                parsed_element_id is None
                or raw_element_id != str(parsed_element_id)
            ):
                raise ValueError(
                    "committed current tangent element-state IDs must be "
                    "canonical integers"
                )
            element_id = parsed_element_id
        else:
            raise ValueError(
                "committed current tangent element-state IDs must be canonical integers"
            )
        if element_id in normalized:
            raise ValueError(
                "committed current tangent element-state IDs are ambiguous"
            )
        if not isinstance(state, Mapping):
            raise TypeError(
                "committed current tangent requires a state mapping for "
                f"element {element_id}"
            )
        snapshot = _guarded_owned_input_snapshot(
            model,
            state,
            path=f"element_states[{element_id}]",
            _exact_guard=_exact_guard,
        )
        if type(snapshot) is not dict:
            raise TypeError(
                "committed current tangent requires a canonical object state for "
                f"element {element_id}"
            )
        canonical_json_bytes(snapshot)
        normalized[element_id] = snapshot
    expected_ids = {int(value) for value in model.mesh.elements}
    supplied_ids = set(normalized)
    if supplied_ids != expected_ids:
        missing = sorted(expected_ids - supplied_ids)
        unknown = sorted(supplied_ids - expected_ids)
        raise ValueError(
            "committed current tangent requires exactly one model-bound state "
            f"per element; missing={missing}, unknown={unknown}"
        )
    return normalized


def _snapshot_committed_current_tangent_inputs(
    model: "FEModel",
    displacements: Any,
    element_states: Any,
    *,
    _exact_guard: Any = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
) -> tuple[np.ndarray, Dict[int, Mapping[str, Any]]]:
    """Consume caller-controlled current-state inputs exactly once.

    Modal and buckling perform several independent authority checks before
    assembly.  They must all observe one detached displacement/state snapshot,
    never repeated views of a mutable caller mapping.
    """

    _exact_guard(model, context="committed current tangent input snapshot")
    total_dofs = int(model.mesh.dof_manager.total_dofs)
    observed = np.asarray(displacements, dtype=np.float64)
    _exact_guard(model, context="committed current tangent displacement observation")
    if observed.shape != (total_dofs,) or not np.all(np.isfinite(observed)):
        raise ValueError(
            "committed current tangent requires the complete finite committed "
            "displacement vector"
        )
    contiguous = np.ascontiguousarray(observed, dtype=np.float64)
    owned = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.float64
    ).reshape(contiguous.shape)
    normalized = _normalized_exact_states(
        model,
        element_states,
        _exact_guard=_exact_guard,
    )
    _exact_guard(model, context="committed current tangent input snapshot")
    return owned, normalized


def _validate_owned_committed_current_tangent_inputs(
    model: "FEModel",
    displacements: np.ndarray,
    element_states: Mapping[int, Mapping[str, Any]],
    num_layers: int,
    *,
    context: str,
    _exact_guard: Any = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
) -> Dict[str, Any]:
    """Validate a detached snapshot without consulting caller state again."""

    _exact_guard(model, context=context)
    route = _qualified_route(
        model,
        context=context,
        _exact_guard=_exact_guard,
    )
    layers = _positive_layer_count(num_layers)
    total_dofs = int(model.mesh.dof_manager.total_dofs)
    full = np.asarray(displacements, dtype=np.float64)
    if full.shape != (total_dofs,) or not np.all(np.isfinite(full)):
        raise ValueError(
            "committed current tangent requires the complete finite committed "
            "displacement vector"
        )
    _validate_exact_state_pairing(
        model,
        full,
        element_states,
        route,
        layers,
        _exact_guard=_exact_guard,
    )
    _exact_guard(model, context=context)
    return route


def require_committed_tangent_component_api(
    model: "FEModel", *, context: str
) -> Dict[str, Any]:
    """Fail before mechanics unless every element has explicit family authority."""

    return _qualified_route(model, context=context)


def require_active_current_state_eigen_lifecycle(
    model: "FEModel",
    route: Mapping[str, Any],
    *,
    context: str,
) -> Dict[str, Any]:
    """Require exact ACTIVE lifecycle state before modal/buckling mechanics."""

    from .activity import ElementActivity

    activity = getattr(model.mesh, "element_activity", None)
    qualified_ids = tuple(sorted(int(value) for value in route["element_profiles"]))
    if activity is None:
        return {
            "policy_id": CURRENT_STATE_EIGEN_ACTIVITY_POLICY_ID,
            "disposition": "ACTIVE_DEFAULT",
            "qualified_element_ids": list(qualified_ids),
            "activity_sequence": 0,
        }
    if type(activity) is not ElementActivity:
        raise ElementCapabilityError(
            f"{context} requires exact ElementActivity ownership"
        )
    raw_ids = np.asarray(activity.element_ids)
    _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD(
        model,
        context=f"{context} ElementActivity ID observation",
    )
    model_ids = tuple(sorted(int(value) for value in model.mesh.elements))
    if (
        raw_ids.ndim != 1
        or raw_ids.dtype.kind not in "iu"
        or len(set(int(value) for value in raw_ids)) != raw_ids.size
        or tuple(sorted(int(value) for value in raw_ids)) != model_ids
    ):
        raise ElementCapabilityError(
            f"{context} ElementActivity is not bound to the exact FE model"
        )
    values = np.asarray(activity.activity, dtype=np.float64)
    _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD(
        model,
        context=f"{context} ElementActivity value observation",
    )
    hard_deleted = np.asarray(activity.hard_deleted_mask, dtype=bool)
    _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD(
        model,
        context=f"{context} ElementActivity deletion observation",
    )
    if (
        values.shape != raw_ids.shape
        or hard_deleted.shape != raw_ids.shape
        or not np.all(np.isfinite(values))
    ):
        raise ElementCapabilityError(f"{context} ElementActivity is malformed")
    index = {
        int(element_id): position
        for position, element_id in enumerate(raw_ids)
    }
    nonactive = [
        element_id
        for element_id in qualified_ids
        if values[index[element_id]] != 1.0
        or bool(hard_deleted[index[element_id]])
    ]
    if nonactive:
        raise ElementCapabilityError(
            f"{context} requires exact ACTIVE qualified Q4/S3 lifecycle state; "
            "nonactive element IDs "
            + ", ".join(str(value) for value in nonactive[:8])
        )
    return {
        "policy_id": CURRENT_STATE_EIGEN_ACTIVITY_POLICY_ID,
        "disposition": "ACTIVE_EXPLICIT",
        "qualified_element_ids": list(qualified_ids),
        "activity_sequence": int(activity.sequence),
    }


def _state_digest(state: Mapping[str, Any], *, element_id: int) -> str:
    raw = state.get("state_integrity_sha256", state.get("state_digest"))
    if not isinstance(raw, str):
        raise ValueError(
            "committed current tangent requires an integrity-bound state for "
            f"element {element_id}"
        )
    digest = raw.strip()
    normalized = digest.lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(
            "committed current tangent requires a canonical SHA-256 state "
            f"digest for element {element_id}"
        )
    return digest


def _validate_exact_state_pairing(
    model: "FEModel",
    full: np.ndarray,
    normalized_states: Mapping[int, Mapping[str, Any]],
    route: Mapping[str, Any],
    num_layers: int,
    *,
    _exact_guard: Any = _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
) -> None:
    """Validate all raw bindings before any constitutive/kinematic replay."""

    total_dofs = int(model.mesh.dof_manager.total_dofs)
    profiles = route["element_profiles"]
    route_data = route["element_data"]
    prepared: Dict[int, Dict[str, Any]] = {}

    # Phase 1a is deliberately closed-world and nonmechanical.  A malformed
    # later element must reject before a valid earlier element can evaluate a
    # constitutive response.
    for raw_element_id, element in sorted(model.mesh.elements.items()):
        element_id = int(raw_element_id)
        profile = profiles[element_id]
        dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
        expected_local_dofs = 6 * int(profile["node_count"])
        expected_dofs = np.asarray(
            [
                int(dof)
                for node_id in route_data[element_id]["node_ids"]
                for dof in model.mesh.dof_manager.get_node_dofs(node_id)
            ],
            dtype=np.intp,
        )
        if (
            dofs.shape != (expected_local_dofs,)
            or expected_dofs.shape != (expected_local_dofs,)
            or not np.array_equal(dofs, expected_dofs)
            or np.unique(dofs).size != dofs.size
            or np.any(dofs < 0)
            or np.any(dofs >= total_dofs)
        ):
            raise ElementCapabilityError(
                "committed current tangent requires six distinct valid nodal "
                "DOFs in the exact immutable-connectivity order per qualified "
                f"element; element {element_id} is incompatible"
            )
        state = normalized_states[element_id]
        if profile["state_displacement_binding"] == "qualified_q4_nested_seal":
            binding = state.get("qualified_q4_committed_binding")
            if not isinstance(binding, Mapping) or binding.get("schema_id") != (
                profile["state_binding_schema_id"]
            ):
                raise ValueError(
                    "committed current tangent requires a sealed qualified Q4 "
                    f"state binding for element {element_id}"
                )
            raw_committed_local = binding.get("committed_total_u", ())
        else:
            raw_committed_local = state.get("committed_total_u", ())
        committed_local = np.asarray(raw_committed_local, dtype=np.float64)
        if (
            committed_local.shape != (expected_local_dofs,)
            or not np.all(np.isfinite(committed_local))
            or not np.array_equal(committed_local, full[dofs])
        ):
            raise ValueError(
                "committed current tangent requires exact displacement/state "
                f"pairing for element {element_id}"
            )
        source_digest = _state_digest(state, element_id=element_id)
        # Canonicalization is both the immutability snapshot and a strict
        # nonfinite/type check.  Perform it for every state before mechanics.
        before = canonical_json_bytes(state)
        prepared[element_id] = {
            "element": element,
            "profile": profile,
            "state": state,
            "committed_local": committed_local,
            "source_digest": source_digest,
            "before": before,
        }

    # Phase 1b validates every formulation identity/hash/configuration seal.
    # Q4 exposes an explicitly nonmechanical binding API; S3's model-bound
    # validator reconstructs only immutable frame/constitutive identity and
    # does not evaluate an element force or tangent.
    for element_id, item in sorted(prepared.items()):
        element = item["element"]
        profile = item["profile"]
        material = model.get_material(route_data[element_id]["material_name"])
        _exact_guard(
            model,
            context="committed current tangent state binding validation",
        )
        if profile["family"] == "qualified_q4":
            validated_digest = element.validate_committed_current_tangent_binding(
                model.mesh,
                material,
                item["committed_local"],
                item["state"],
                int(num_layers),
            )
            if str(validated_digest).lower() != str(
                item["source_digest"]
            ).lower():
                raise ValueError(
                    "qualified Q4 state binding validator returned a mismatched "
                    f"digest for element {element_id}"
                )
        else:
            profile["state_validator"](
                element,
                model.mesh,
                material,
                item["state"],
                int(num_layers),
                expected_committed_total_u=item["committed_local"],
            )
        if canonical_json_bytes(item["state"]) != item["before"]:
            raise RuntimeError(
                "committed current tangent binding validation mutated element "
                f"{element_id}"
            )

    # Phase 2 may reconstruct formulation semantics only after every element
    # has passed the complete raw/binding guard phase.
    for element_id, item in sorted(prepared.items()):
        element = item["element"]
        profile = item["profile"]
        state = item["state"]
        committed_local = item["committed_local"]
        source_digest = item["source_digest"]
        before = item["before"]
        material = model.get_material(route_data[element_id]["material_name"])
        _exact_guard(
            model,
            context="committed current tangent state semantic validation",
        )
        if profile["family"] == "qualified_q4":
            validated_digest = element.validate_committed_current_tangent_semantics(
                model.mesh,
                material,
                committed_local,
                state,
                int(num_layers),
            )
            if str(validated_digest).lower() != source_digest.lower():
                raise ValueError(
                    "qualified Q4 state semantic validator returned a mismatched digest "
                    f"for element {element_id}"
                )
        if canonical_json_bytes(state) != before:
            raise RuntimeError(
                "committed current tangent state validation mutated element "
                f"{element_id}"
            )
    _exact_guard(
        model,
        context="committed current tangent state validation",
    )


def _validate_committed_current_tangent_inputs_implementation(
    model: "FEModel",
    displacements: Any,
    element_states: Any,
    num_layers: int,
    *,
    context: str,
    _exact_guard: Any,
) -> Dict[str, Any]:
    """Validate the complete authority profile without evaluating mechanics."""

    _exact_guard(model, context=context)
    full, normalized_states = _snapshot_committed_current_tangent_inputs(
        model,
        displacements,
        element_states,
        _exact_guard=_exact_guard,
    )
    return _validate_owned_committed_current_tangent_inputs(
        model,
        full,
        normalized_states,
        num_layers,
        context=context,
        _exact_guard=_exact_guard,
    )


def _relative_sparse_error(left: sparse.spmatrix, right: sparse.spmatrix) -> float:
    denominator = max(float(sparse_linalg.norm(right)), 1.0)
    return float(sparse_linalg.norm(left) / denominator)


def _assemble_committed_current_tangent_components_implementation(
    model: "FEModel",
    displacements: Any,
    element_states: Any,
    num_layers: int = 5,
    *,
    _exact_guard: Any,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, Dict[str, Any]]:
    """Assemble ``Kmaterial``, ``Kgeometric`` and their consistent total.

    ``Kgeometric`` is the internal-force stress/resultant Hessian and is
    tension-positive.  Consequently ``-Kgeometric`` is the compression-positive
    destabilizing operator used by current-state buckling.  S3 is evaluated in
    a disposable native rotation transaction; additive-von-Karman Q4 neither
    creates nor receives one.  Input states are never updated.  Matrices and
    internal-coordinate sensitivities are transient and are not attached to
    the model, element, state store, or an analysis session.
    """

    _exact_guard(
        model,
        context="assemble_committed_current_tangent_components",
    )
    full, normalized_states = _snapshot_committed_current_tangent_inputs(
        model,
        displacements,
        element_states,
        _exact_guard=_exact_guard,
    )
    route = _validate_owned_committed_current_tangent_inputs(
        model,
        full,
        normalized_states,
        num_layers,
        context="assemble_committed_current_tangent_components",
        _exact_guard=_exact_guard,
    )
    layers = _positive_layer_count(num_layers)
    total_dofs = int(model.mesh.dof_manager.total_dofs)
    state_bytes_before = {
        element_id: canonical_json_bytes(state)
        for element_id, state in normalized_states.items()
    }
    _activity, activity_scales, activity_info = _activity_scales(
        model, "stiffness"
    )
    _exact_guard(
        model,
        context="assemble_committed_current_tangent_components activity",
    )

    store = NonlinearStateStore.from_shell_layouts((), normalized_states)
    token = None
    if bool(route["native_rotation_required"]):
        rotation_store = create_model_native_rotation_store(model, store, full)
        if rotation_store is None:
            raise ElementCapabilityError(
                "committed current tangent route requires formulation-native "
                "rotation state"
            )
        store.attach_native_rotation_store(rotation_store)
        token = begin_state_evaluation(
            store,
            model=model,
            displacements=full,
        )
        if token is None:
            raise RuntimeError(
                "committed current tangent native rotation transaction did not start"
            )

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    material_data: list[np.ndarray] = []
    geometric_data: list[np.ndarray] = []
    total_data: list[np.ndarray] = []
    element_info: Dict[str, Any] = {}
    decomposition_policy_ids: Dict[str, str] = {}
    projection_policy_ids: Dict[str, str] = {}
    bubble_projection_policy_ids: set[str] = set()
    try:
        for element_id, element in sorted(model.mesh.elements.items()):
            element_id = int(element_id)
            profile = route["element_profiles"][element_id]
            owned_data = route["element_data"][element_id]
            dofs = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
            local_dofs = int(dofs.size)
            native_kwargs: Dict[str, Any] = {}
            if bool(profile["native_rotation_required"]):
                if token is None:
                    raise RuntimeError(
                        "committed current tangent native element has no transaction"
                    )
                reference_directors = np.asarray(
                    element.native_reference_directors(model.mesh),
                    dtype=np.float64,
                )
                native_kwargs["native_rotation_trial"] = (
                    store.native_element_rotation_view(
                        token,
                        element_id,
                        owned_data["node_ids"],
                        reference_directors,
                    )
                )
            material = model.get_material(owned_data["material_name"])
            _exact_guard(
                model,
                context=(
                    "assemble_committed_current_tangent_components material"
                ),
            )
            component_context = (
                model
                if bool(profile["component_api_receives_model"])
                else model.mesh
            )
            components = profile["component_api"](
                element,
                component_context,
                material,
                full[dofs],
                normalized_states[element_id],
                layers,
                **native_kwargs,
            )
            if not isinstance(components, Mapping):
                raise TypeError(
                    f"committed tangent components for element {element_id} "
                    "must be a mapping"
                )
            material = np.asarray(components.get("material", ()), dtype=np.float64)
            geometric = np.asarray(components.get("geometric", ()), dtype=np.float64)
            total = np.asarray(components.get("total", ()), dtype=np.float64)
            if (
                material.shape != (local_dofs, local_dofs)
                or geometric.shape != (local_dofs, local_dofs)
                or total.shape != (local_dofs, local_dofs)
                or not np.all(np.isfinite(material))
                or not np.all(np.isfinite(geometric))
                or not np.all(np.isfinite(total))
            ):
                raise ValueError(
                    f"committed tangent components for element {element_id} are incompatible"
                )
            local_decomposition_error = float(
                np.linalg.norm(total - material - geometric, ord="fro")
                / max(float(np.linalg.norm(total, ord="fro")), 1.0)
            )
            local_symmetry_error = max(
                float(
                    np.linalg.norm(matrix - matrix.T, ord="fro")
                    / max(float(np.linalg.norm(matrix, ord="fro")), 1.0)
                )
                for matrix in (material, geometric, total)
            )
            claimed_decomposition_error = float(
                components.get("relative_decomposition_error", math.inf)
            )
            claimed_symmetry_error = float(
                components.get("relative_symmetry_error", math.inf)
            )
            component_limit = 512.0 * np.finfo(np.float64).eps
            if (
                not math.isfinite(local_decomposition_error)
                or not math.isfinite(local_symmetry_error)
                or not math.isfinite(claimed_decomposition_error)
                or not math.isfinite(claimed_symmetry_error)
                or local_decomposition_error > component_limit
                or local_symmetry_error > component_limit
                or claimed_decomposition_error > component_limit
                or claimed_symmetry_error > component_limit
            ):
                raise ValueError(
                    f"committed tangent components for element {element_id} "
                    "violate decomposition or symmetry"
                )
            source_digest = _state_digest(
                normalized_states[element_id], element_id=element_id
            )
            returned_digest = str(components.get("state_digest", "")).strip()
            if returned_digest.lower() != source_digest.lower():
                raise ValueError(
                    f"committed tangent components for element {element_id} "
                    "are not bound to the supplied committed state"
                )
            decomposition_policy_id = str(
                components.get("decomposition_policy_id", "")
            )
            projection_key = (
                "projection_policy_id"
                if "projection_policy_id" in components
                else "bubble_projection_policy_id"
            )
            projection_policy_id = str(components.get(projection_key, ""))
            expected_decomposition_policy = str(
                profile["decomposition_policy_id"]
            )
            expected_projection_policy = str(profile["projection_policy_id"])
            binding_verified = components.get("state_binding_verified") is True
            expected_algorithmic_origin_schema = profile[
                "algorithmic_origin_schema_id"
            ]
            algorithmic_origin_schema_id = components.get(
                "algorithmic_origin_schema_id"
            )
            algorithmic_origin_verified = (
                components.get("algorithmic_origin_verified") is True
            )
            if (
                decomposition_policy_id != expected_decomposition_policy
                or projection_policy_id != expected_projection_policy
                or (
                    bool(profile["component_binding_flag_required"])
                    and not binding_verified
                )
                or (
                    bool(
                        profile[
                            "component_algorithmic_origin_flag_required"
                        ]
                    )
                    and (
                        str(algorithmic_origin_schema_id)
                        != str(expected_algorithmic_origin_schema)
                        or not algorithmic_origin_verified
                    )
                )
            ):
                raise ValueError(
                    f"committed tangent components for element {element_id} "
                    "lack exact decomposition/projection/state-binding/"
                    "algorithmic-origin authority"
                )
            formulation_id = str(profile["formulation_id"])
            prior_decomposition_policy = decomposition_policy_ids.get(
                formulation_id
            )
            prior_projection_policy = projection_policy_ids.get(formulation_id)
            if (
                prior_decomposition_policy is not None
                and prior_decomposition_policy != decomposition_policy_id
            ) or (
                prior_projection_policy is not None
                and prior_projection_policy != projection_policy_id
            ):
                raise ValueError(
                    "committed tangent components disagree on formulation "
                    f"policy authority for {formulation_id}"
                )
            decomposition_policy_ids[formulation_id] = decomposition_policy_id
            projection_policy_ids[formulation_id] = projection_policy_id
            if projection_key == "bubble_projection_policy_id":
                bubble_projection_policy_ids.add(projection_policy_id)
            activity_scale = float(activity_scales.get(element_id, 1.0))
            rows.append(np.repeat(dofs, dofs.size))
            columns.append(np.tile(dofs, dofs.size))
            material_data.append((activity_scale * material).ravel())
            geometric_data.append((activity_scale * geometric).ravel())
            total_data.append((activity_scale * total).ravel())
            element_info[str(element_id)] = {
                "formulation_id": formulation_id,
                "implementation_id": (
                    None
                    if profile["implementation_id"] is None
                    else str(profile["implementation_id"])
                ),
                "family": str(profile["family"]),
                "local_dofs": local_dofs,
                "native_rotation_required": bool(
                    profile["native_rotation_required"]
                ),
                "kinematics": str(profile["kinematics"]),
                "reference_surface_offset_scope": str(
                    profile["reference_surface_offset_scope"]
                ),
                "state_digest": source_digest,
                "state_binding_verified": (
                    binding_verified
                    if bool(profile["component_binding_flag_required"])
                    else True
                ),
                "algorithmic_origin_schema_id": (
                    str(algorithmic_origin_schema_id)
                    if expected_algorithmic_origin_schema is not None
                    else None
                ),
                "algorithmic_origin_verified": (
                    algorithmic_origin_verified
                    if bool(
                        profile[
                            "component_algorithmic_origin_flag_required"
                        ]
                    )
                    else None
                ),
                "relative_decomposition_error": local_decomposition_error,
                "relative_symmetry_error": local_symmetry_error,
                "decomposition_policy_id": decomposition_policy_id,
                "projection_policy_id": projection_policy_id,
                "activity_stiffness_scale": activity_scale,
                "matrix_persistence": "none",
            }
    finally:
        discard_active_state_candidate(store)

    state_bytes_after = {
        element_id: canonical_json_bytes(state)
        for element_id, state in normalized_states.items()
    }
    if state_bytes_after != state_bytes_before:
        raise RuntimeError("committed current tangent mutated an input state")

    if rows:
        row = np.concatenate(rows)
        column = np.concatenate(columns)

        def assembled(values: list[np.ndarray]) -> sparse.csr_matrix:
            return sparse.coo_matrix(
                (np.concatenate(values), (row, column)),
                shape=(total_dofs, total_dofs),
                dtype=np.float64,
            ).tocsr()

        material_matrix = assembled(material_data)
        geometric_matrix = assembled(geometric_data)
        total_matrix = assembled(total_data)
        if activity_info is not None and activity_info["zero_contribution_count"]:
            material_matrix.eliminate_zeros()
            geometric_matrix.eliminate_zeros()
            total_matrix.eliminate_zeros()
    else:
        material_matrix = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
        geometric_matrix = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
        total_matrix = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)

    decomposition_error = _relative_sparse_error(
        total_matrix - material_matrix - geometric_matrix,
        total_matrix,
    )
    symmetry_error = max(
        _relative_sparse_error(matrix - matrix.T, matrix)
        for matrix in (material_matrix, geometric_matrix, total_matrix)
    )
    if (
        not math.isfinite(decomposition_error)
        or not math.isfinite(symmetry_error)
        or decomposition_error > 512.0 * np.finfo(np.float64).eps
        or symmetry_error > 512.0 * np.finfo(np.float64).eps
    ):
        raise ValueError(
            "assembled committed tangent violates its decomposition or symmetry bound"
        )
    material_matrix = (0.5 * (material_matrix + material_matrix.T)).tocsr()
    geometric_matrix = (0.5 * (geometric_matrix + geometric_matrix.T)).tocsr()
    total_matrix = (0.5 * (total_matrix + total_matrix.T)).tocsr()
    state_digests = {
        str(element_id): _state_digest(state, element_id=element_id)
        for element_id, state in sorted(normalized_states.items())
    }
    route_info = {
        "route": str(route["route"]),
        "route_policy_id": str(route["route_policy_id"]),
        "families": list(route["families"]),
        "formulation_counts": dict(route["formulation_counts"]),
        "native_rotation_required": bool(route["native_rotation_required"]),
        "kinematic_scope": dict(sorted(route["kinematic_scope"].items())),
        "reference_surface_offset_scope": dict(
            sorted(route["reference_surface_offset_scope"].items())
        ),
    }
    info: Dict[str, Any] = {
        "matrix_type": "committed_current_tangent_components",
        "policy_id": COMMITTED_CURRENT_TANGENT_ASSEMBLY_POLICY_ID,
        "input_ownership_policy_id": (
            COMMITTED_CURRENT_TANGENT_INPUT_OWNERSHIP_POLICY_ID
        ),
        "route": route_info,
        "decomposition_policy_ids": dict(sorted(decomposition_policy_ids.items())),
        "projection_policy_ids": dict(sorted(projection_policy_ids.items())),
        "geometric_sign_convention": (
            "internal_tension_positive_stress_hessian; negative_is_"
            "compression_positive_destabilizing"
        ),
        "matrix_persistence": "none",
        "factorization_persistence": "none",
        "state_digests": state_digests,
        "state_immutability_verified": True,
        "relative_decomposition_error": decomposition_error,
        "relative_symmetry_error": symmetry_error,
        "num_layers": layers,
        "element_components": element_info,
        "state_storage": store.diagnostics(),
        "element_activity": activity_info,
    }
    # Retain the homogeneous-S3 diagnostic alias without mislabelling a Q4
    # projection or a heterogeneous route as a bubble operation.
    if (
        route_info["route"] == "qualified_s3"
        and len(bubble_projection_policy_ids) == 1
    ):
        info["bubble_projection_policy_id"] = next(
            iter(bubble_projection_policy_ids)
        )
    return material_matrix, geometric_matrix, total_matrix, info


def _bind_committed_current_tangent_boundaries(
    validate_implementation: Any,
    assemble_implementation: Any,
    exact_guard: Any,
) -> tuple[Any, Any]:
    """Bind one immutable authority guard across every input/output boundary."""

    def leased_guard(lease: Any) -> Any:
        def guard(
            model: "FEModel",
            *,
            context: str,
        ) -> Dict[str, Any]:
            result = exact_guard(model, context=context)
            lease(model, context=context)
            return result

        return guard

    def validate(
        model: "FEModel",
        displacements: Any,
        element_states: Any,
        num_layers: int,
        *,
        context: str,
    ) -> Dict[str, Any]:
        exact_guard(model, context=context)
        return _run_with_qualified_assembly_runtime_lease(
            model,
            context=context,
            operation=lambda lease: validate_implementation(
                model,
                displacements,
                element_states,
                num_layers,
                context=context,
                _exact_guard=leased_guard(lease),
            ),
        )

    def assemble(
        model: "FEModel",
        displacements: Any,
        element_states: Any,
        num_layers: int = 5,
    ) -> tuple[
        sparse.csr_matrix,
        sparse.csr_matrix,
        sparse.csr_matrix,
        Dict[str, Any],
    ]:
        context = "assemble_committed_current_tangent_components"
        exact_guard(model, context=context)
        return _run_with_qualified_assembly_runtime_lease(
            model,
            context=context,
            operation=lambda lease: assemble_implementation(
                model,
                displacements,
                element_states,
                num_layers,
                _exact_guard=leased_guard(lease),
            ),
        )

    for function, name, implementation in (
        (validate, "validate_committed_current_tangent_inputs", validate_implementation),
        (assemble, "assemble_committed_current_tangent_components", assemble_implementation),
    ):
        function.__name__ = name
        function.__qualname__ = name
        function.__doc__ = implementation.__doc__
        function.__module__ = __name__
    return validate, assemble


(
    validate_committed_current_tangent_inputs,
    assemble_committed_current_tangent_components,
) = _bind_committed_current_tangent_boundaries(
    _validate_committed_current_tangent_inputs_implementation,
    _assemble_committed_current_tangent_components_implementation,
    _EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD,
)


__all__ = [
    "COMMITTED_CURRENT_TANGENT_ASSEMBLY_POLICY_ID",
    "COMMITTED_CURRENT_TANGENT_INPUT_OWNERSHIP_POLICY_ID",
    "COMMITTED_CURRENT_TANGENT_ROUTE_POLICY_ID",
    "CURRENT_STATE_EIGEN_ACTIVITY_POLICY_ID",
    "QUALIFIED_Q4_FORMULATION_ID",
    "QUALIFIED_S3_FORMULATION_ID",
    "S3_V2D_FORMULATION_ID",
    "S3_V2D_CURRENT_STATE_PROJECTION_POLICY_ID",
    "S3_V2D_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID",
    "assemble_committed_current_tangent_components",
    "require_committed_tangent_component_api",
    "require_active_current_state_eigen_lifecycle",
    "validate_committed_current_tangent_inputs",
]
