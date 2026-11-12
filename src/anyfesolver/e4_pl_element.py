"""Dormant production implementation of the qualified E4-PL four-node shell.

The class reuses the mature :class:`~anyfesolver.elements.ShellElement`
infrastructure for mass, geometric stiffness, state, nonlinear, dynamics,
contact and serialization behavior.  Planar facets use the qualified 35+3
stationary E4-PL formulation for both tangent and physical recovery.  Genuinely
warped facets use the established varying-frame Q4 surface kernel explicitly,
because a single projected plane does not retain the six physical rigid modes
on a warped bilinear surface.

The physical condensed tangent, centre-PL term and retained drilling
hourglass term are exposed separately so numerical fields cannot silently
enter physical recovery or reaction reporting.
"""

from __future__ import annotations

from abc import ABC
import copy
import inspect
import math
import sys
import warnings
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from operator import is_ as _operator_is, itemgetter
from types import FunctionType as _FunctionType, MappingProxyType, ModuleType
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from . import e4_pl_s3_state as _s3_state_module
from . import elements as _elements_module
from . import fe_core as _fe_core_module
from . import plasticity as _plasticity_module
from . import shell_sections as _shell_sections_module
from . import vectorized_nonlinear as _elastic_increment_module
from . import _compensated_contractions as _local_contractions
from . import _q4_assumed_strain_plasticity as _q4_assumed_strain
from ._qualified_authority_epoch import (
    AuthorityEpochMeta,
    make_authority_epoch_manager,
)
from .elements import (
    Element,
    ShellElement,
    _generalized_shell_section_cache_fingerprint,
    _guarded_observe_attribute,
    _guarded_observe_call,
    _guarded_owned_generalized_shell_section,
    _guarded_owned_mapping,
    _guarded_owned_plain_value,
    _shell_elastic_material_cache_fingerprint,
    _shell_material_matrices,
)
from .fe_core import Node
from .e4_pl_s3_state import (
    _capture_authority_array_metadata,
    _module_authority_signature,
    _require_authority_array_metadata,
    _require_exact_numpy_runtime_module_identity,
    _require_immutable_authority_data,
    _watch_exact_numpy_runtime_epoch,
    canonical_json_bytes,
    canonical_sha256,
    require_exact_numpy_runtime_authority,
    resolved_material_descriptor,
)
from .plasticity import hill48_plane_stress_equivalent_stress, lobatto_layers
from .shell_sections import (
    GeneralizedShellSection,
    SHELL_MEMBRANE_VOIGT_ORDER,
    SHELL_TRANSVERSE_SHEAR_ORDER,
    coerce_generalized_shell_section,
)


FORMULATION_ID = "E4_PL_QUALIFIED_Q4_HYBRID_V2"
IMPLEMENTATION_ID = "E4_PL_Q4_HYBRID_STATIONARY_RECOVERY_DIRECTOR_RUIZ_V7"
RECOVERY_POLICY_ID = (
    "Q4_HYBRID_PLANAR_STATIONARY_WARPED_VARYING_FRAME_"
    "PHYSICAL_DIRECTOR_RECOVERY_V3"
)
STATIONARY_SOLVE_POLICY_ID = (
    "Q4_SYMMETRIC_RUIZ_8_ORIGINAL_BACKWARD_ERROR_V2"
)
DIRECTOR_POLARITY_POLICY_ID = (
    "Q4_ELEMENT_OWNED_PHYSICAL_DIRECTOR_INDEPENDENT_OF_D4_NUMBERING_V1"
)
DIRECTOR_REVERSAL_TRANSFORM_ID = (
    "Q4_EPS_S_KAPPA_SIGN_S_SHEAR_SIGN_P_ABD_CONGRUENCE_V1"
)
Q4_CURRENT_STATE_BINDING_SCHEMA_ID = (
    "E4_PL_Q4_COMMITTED_STATE_ALGORITHMIC_TANGENT_BINDING_V2"
)
Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID = (
    "Q4_VON_KARMAN_ACCEPTED_ALGORITHMIC_MATERIAL_PLUS_MEMBRANE_STRESS_HESSIAN_V2"
)
Q4_CURRENT_STATE_PROJECTION_POLICY_ID = (
    "Q4_NO_INTERNAL_BUBBLE_IDENTITY_PROJECTION_V1"
)
Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID = (
    "E4_PL_Q4_ACCEPTED_DISCRETE_RETURN_MAP_ORIGIN_V1"
)
Q4_ACTIVITY_DISPOSITION_SCHEMA_ID = "E4_PL_Q4_ACTIVITY_DISPOSITION_V1"
Q4_DELETED_FROZEN_POLICY_ID = (
    "Q4_DELETED_FROZEN_CONSTITUTIVE_HISTORY_RESIDUAL_OPERATOR_V1"
)
Q4_FAILED_STATE_POLICY_ID = "Q4_FAILED_NONAUTHORITATIVE_RESULT_STATE_V1"
Q4_QUADRATURE_AUTHORITY_ID = (
    "Q4_ORDERED_STIFFNESS_2X2_MASS_2X2_MITC_SHEAR_CLASS_2X2_IMMUTABLE_EXACT_V1"
)
_Q4_CURRENT_STATE_BINDING_KEY = "qualified_q4_committed_binding"
_Q4_CURRENT_STATE_DIGEST_KEY = "state_integrity_sha256"
_Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_KEY = (
    "qualified_q4_algorithmic_origin"
)
_Q4_ACTIVITY_DISPOSITION_KEY = "qualified_q4_activity_disposition"
_Q4_COMPACT_REPLAY_MAX_ULPS = 262144
_FOREIGN_S3_ACTIVITY_DISPOSITION_KEY = "qualified_s3_activity_disposition"
_V5_IMPLEMENTATION_ID = (
    "E4_PL_Q4_HYBRID_STATIONARY_RECOVERY_DIRECTOR_RUIZ_V5"
)
_V6_IMPLEMENTATION_ID = (
    "E4_PL_Q4_HYBRID_STATIONARY_RECOVERY_DIRECTOR_RUIZ_V6"
)
_V6_CURRENT_STATE_BINDING_SCHEMA_ID = (
    "E4_PL_Q4_COMMITTED_STATE_DISPLACEMENT_BINDING_V1"
)
_V6_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID = (
    "Q4_VON_KARMAN_ALGORITHMIC_MATERIAL_PLUS_MEMBRANE_STRESS_HESSIAN_V1"
)
_PLANAR_FORMULATION_ID = "E4_PL_QUALIFIED_PLANAR_LINEAR_V1"
_WARPED_FORMULATIONS = frozenset({"varying_frame", "reject"})
_STATIONARY_RUIZ_ITERATIONS = 8
_STATIONARY_RUIZ_ROW_NORM_RATIO_LIMIT = 2.0
_STATIONARY_BACKWARD_ERROR_LIMIT = 1.0e-10
_Q4_INITIAL_STATE_KEYS = (
    "initial_membrane_stress",
    "initial_bending_stress",
    "initial_membrane_prestrain",
    "initial_curvature_prestrain",
    "initial_field_provenance",
)


def _q4_compact_replay_last_place_equivalent(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    max_ulps: int = _Q4_COMPACT_REPLAY_MAX_ULPS,
) -> bool:
    """Return whether finite binary64 arrays differ only in bounded last places.

    Production batch assembly and scalar final-state replay evaluate identical
    Q4 kinematics with different BLAS operation shapes.  That can move a small
    number of values by a bounded count of representable binary64 steps.  This
    predicate is deliberately narrower than an engineering tolerance: it has
    no absolute or relative floor and is used only for compact batch-produced
    constitutive state after all state and origin hashes have been validated.
    """

    lhs = np.asarray(actual, dtype=np.float64)
    rhs = np.asarray(expected, dtype=np.float64)
    if lhs.shape != rhs.shape or max_ulps < 0:
        return False
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        return False
    if np.array_equal(lhs, rhs):
        return True
    scale = np.maximum(np.abs(lhs), np.abs(rhs))
    spacing = np.spacing(scale)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(lhs - rhs)
        limit = float(max_ulps) * spacing
    return bool(np.all(np.isfinite(difference)) and np.all(difference <= limit))


_QUALIFIED_Q4_COORDINATE_SCALAR_TYPES = frozenset(
    {
        int,
        float,
        *(np.dtype(code).type for code in "bBhHiIlLqQefdg"),
    }
)
_QUALIFIED_Q4_PROTECTED_CACHE_NAMES = frozenset(
    {
        "_nl_cache",
        "_nl_cache_key",
        "_qualified_cache_key",
        "_qualified_component_guard",
        "_qualified_components",
    }
)
_Q4_STIFFNESS_STATION_AUTHORITY = tuple(
    (r, s)
    for r, s in (
        (-1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
        (1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
        (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
        (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
    )
)
_GAUSS = _Q4_STIFFNESS_STATION_AUTHORITY
_Q4_QUADRATURE_ARRAY_AUTHORITY = MappingProxyType(
    {
        "gauss_points": (
            (4, 2),
            np.ascontiguousarray(
                np.asarray(
                    [
                        [-1.0, -1.0],
                        [1.0, -1.0],
                        [-1.0, 1.0],
                        [1.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                / math.sqrt(3.0)
            ).tobytes(order="C"),
        ),
        "gauss_weights": (
            (4,),
            np.ones(4, dtype=np.float64).tobytes(order="C"),
        ),
        "shear_gauss_points": (
            (1, 2),
            np.zeros((1, 2), dtype=np.float64).tobytes(order="C"),
        ),
        "shear_gauss_weights": (
            (1,),
            np.asarray([4.0], dtype=np.float64).tobytes(order="C"),
        ),
    }
)
_Q4_QUADRATURE_PROPERTY_AUTHORITY = MappingProxyType(
    {
        "gauss_points": ShellElement.gauss_points,
        "gauss_weights": ShellElement.gauss_weights,
        "shear_gauss_points": ShellElement.shear_gauss_points,
        "shear_gauss_weights": ShellElement.shear_gauss_weights,
    }
)


def _owned_q4_quadrature_array(value: np.ndarray) -> np.ndarray:
    made = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return np.frombuffer(
        made.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(made.shape)


_Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY = MappingProxyType(
    {
        "GAUSS_POINTS_2x2": _owned_q4_quadrature_array(
            ShellElement.GAUSS_POINTS_2x2
        ),
        "GAUSS_WEIGHTS_2x2": _owned_q4_quadrature_array(
            ShellElement.GAUSS_WEIGHTS_2x2
        ),
        "GAUSS_POINTS_1x1": _owned_q4_quadrature_array(
            ShellElement.GAUSS_POINTS_1x1
        ),
        "GAUSS_WEIGHTS_1x1": _owned_q4_quadrature_array(
            ShellElement.GAUSS_WEIGHTS_1x1
        ),
    }
)


def _static_mro_attribute(owner: type[Any], name: str) -> Any:
    """Return a class member without invoking a mutable descriptor."""

    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            value = namespace[name]
            if isinstance(value, (classmethod, staticmethod)):
                return value.__func__
            return value
    return None


def _require_exact_readonly_float64_array(
    label: str,
    value: Any,
    expected_shape: tuple[int, ...],
    expected_bytes: bytes,
) -> None:
    if type(value) is not np.ndarray:
        raise ValueError(f"qualified Q4 {label} must be an exact numpy array")
    if (
        value.dtype != np.dtype(np.float64)
        or value.shape != expected_shape
        or not value.flags.c_contiguous
        or value.flags.writeable
        or value.tobytes(order="C") != expected_bytes
    ):
        raise ValueError(f"qualified Q4 {label} authority is incompatible")


def _validate_q4_quadrature_authority_exact(
    element: Any,
    _array_authority: Mapping[str, tuple[tuple[int, ...], bytes]] = (
        _Q4_QUADRATURE_ARRAY_AUTHORITY
    ),
    _property_authority: Mapping[str, Any] = _Q4_QUADRATURE_PROPERTY_AUTHORITY,
    _class_array_authority: Mapping[str, np.ndarray] = (
        _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY
    ),
    _station_authority: tuple[tuple[float, float], ...] = (
        _Q4_STIFFNESS_STATION_AUTHORITY
    ),
    _gauss_signature: tuple[Any, ...] = _module_authority_signature(_GAUSS),
    _signature: Any = _module_authority_signature,
    _static_lookup: Any = _static_mro_attribute,
    _array_checker: Any = _require_exact_readonly_float64_array,
    _base_class: type[Any] = ShellElement,
    _authority_id: str = Q4_QUADRATURE_AUTHORITY_ID,
) -> str:
    namespace = object.__getattribute__(element, "__dict__")
    property_values: Dict[str, Any] = {}
    for name, expected in _property_authority.items():
        if (
            name in namespace
            or _static_lookup(type(element), name) is not expected
        ):
            raise ValueError(
                f"qualified Q4 {name} property authority is incompatible"
            )
        property_values[name] = expected.__get__(element, type(element))
    for name, expected in _class_array_authority.items():
        if (
            name in namespace
            or _static_lookup(type(element), name) is not expected
        ):
            raise ValueError(
                f"qualified Q4 {name} class authority is incompatible"
            )
    for name, (shape, expected_bytes) in _array_authority.items():
        _array_checker(
            name,
            property_values[name],
            shape,
            expected_bytes,
        )
    if (
        _signature(globals().get("_GAUSS")) != _gauss_signature
        or _signature(_station_authority) != _gauss_signature
    ):
        raise ValueError(
            "qualified Q4 stiffness station-table authority is incompatible"
        )
    return _authority_id


def _require_q4_quadrature_instance_authority(
    element: Any,
    _property_authority: Mapping[str, Any] = _Q4_QUADRATURE_PROPERTY_AUTHORITY,
    _class_array_authority: Mapping[str, np.ndarray] = (
        _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY
    ),
    _metadata_authority: Mapping[
        str, tuple[np.ndarray, str, tuple[int, ...], tuple[int, ...]]
    ] = MappingProxyType(
        {
            "gauss_points": (
                _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY["GAUSS_POINTS_2x2"],
                _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                    "GAUSS_POINTS_2x2"
                ].dtype.str,
                tuple(
                    _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                        "GAUSS_POINTS_2x2"
                    ].shape
                ),
                tuple(
                    _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                        "GAUSS_POINTS_2x2"
                    ].strides
                ),
            ),
            "gauss_weights": (
                _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY["GAUSS_WEIGHTS_2x2"],
                _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                    "GAUSS_WEIGHTS_2x2"
                ].dtype.str,
                tuple(
                    _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                        "GAUSS_WEIGHTS_2x2"
                    ].shape
                ),
                tuple(
                    _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                        "GAUSS_WEIGHTS_2x2"
                    ].strides
                ),
            ),
            "shear_gauss_points": (
                _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY["GAUSS_POINTS_1x1"],
                _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                    "GAUSS_POINTS_1x1"
                ].dtype.str,
                tuple(
                    _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                        "GAUSS_POINTS_1x1"
                    ].shape
                ),
                tuple(
                    _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                        "GAUSS_POINTS_1x1"
                    ].strides
                ),
            ),
            "shear_gauss_weights": (
                _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY["GAUSS_WEIGHTS_1x1"],
                _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                    "GAUSS_WEIGHTS_1x1"
                ].dtype.str,
                tuple(
                    _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                        "GAUSS_WEIGHTS_1x1"
                    ].shape
                ),
                tuple(
                    _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
                        "GAUSS_WEIGHTS_1x1"
                    ].strides
                ),
            ),
        }
    ),
    _static_lookup: Any = _static_mro_attribute,
    _base_class: type[Any] = ShellElement,
) -> None:
    """Check the element-specific quadrature surface on every invocation."""

    namespace = object.__getattribute__(element, "__dict__")
    if type(namespace) is not dict:
        raise ValueError("qualified Q4 instance namespace is incompatible")
    if not all(type(name) is str for name in namespace):
        raise ValueError("qualified Q4 instance keys must be exact strings")
    for name, expected in _property_authority.items():
        if name in namespace or _static_lookup(type(element), name) is not expected:
            raise ValueError(
                f"qualified Q4 {name} property authority is incompatible"
            )
        value = expected.__get__(element, type(element))
        expected_value, dtype_string, shape, strides = _metadata_authority[name]
        if (
            value is not expected_value
            or type(value) is not np.ndarray
            or value.dtype.str != dtype_string
            or value.shape != shape
            or value.strides != strides
            or not value.flags.c_contiguous
            or value.flags.writeable
        ):
            raise ValueError(
                f"qualified Q4 {name} array metadata is incompatible"
            )
        current: Any = value
        while type(current) is np.ndarray:
            if current.flags.writeable:
                raise ValueError(
                    f"qualified Q4 {name} array base is writeable"
                )
            current = current.base
        if not (
            type(current) is bytes
            or isinstance(current, memoryview) and current.readonly
        ):
            raise ValueError(
                f"qualified Q4 {name} array base is incompatible"
            )
    for name, expected in _class_array_authority.items():
        if (
            name in namespace
            or _static_lookup(type(element), name) is not expected
        ):
            raise ValueError(
                f"qualified Q4 {name} class authority is incompatible"
            )


_q4_quadrature_epoch_manager = make_authority_epoch_manager(
    "qualified Q4 quadrature"
)
_q4_quadrature_epoch_manager.watch_module(
    sys.modules[__name__],
    (
        "__name__",
        "_GAUSS",
        "_Q4_STIFFNESS_STATION_AUTHORITY",
        "_Q4_QUADRATURE_ARRAY_AUTHORITY",
        "_Q4_QUADRATURE_PROPERTY_AUTHORITY",
        "_Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY",
    ),
)
for _q4_quadrature_name, _q4_quadrature_value in (
    ("_GAUSS", _GAUSS),
    ("_Q4_STIFFNESS_STATION_AUTHORITY", _Q4_STIFFNESS_STATION_AUTHORITY),
    ("_Q4_QUADRATURE_ARRAY_AUTHORITY", _Q4_QUADRATURE_ARRAY_AUTHORITY),
    ("_Q4_QUADRATURE_PROPERTY_AUTHORITY", _Q4_QUADRATURE_PROPERTY_AUTHORITY),
    ("_Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY", _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY),
):
    _require_immutable_authority_data(
        _q4_quadrature_value,
        label=f"qualified Q4 {_q4_quadrature_name}",
    )
del _q4_quadrature_name, _q4_quadrature_value
_q4_quadrature_epoch_guard = _q4_quadrature_epoch_manager.bind_argument(
    _validate_q4_quadrature_authority_exact
)


def _validate_q4_quadrature_authority(element: Any) -> str:
    """Validate exact per-element facts plus dirty global quadrature state."""

    _require_q4_quadrature_instance_authority(element)
    _q4_quadrature_epoch_guard(element)
    return Q4_QUADRATURE_AUTHORITY_ID


_QUALIFIED_Q4_BASE_NONLINEAR_KERNEL = ShellElement.compute_nonlinear_response
_QUALIFIED_Q4_BASE_NONLINEAR_GEOMETRY = ShellElement._nonlinear_geometry
_QUALIFIED_Q4_BASE_STIFFNESS_KERNEL = ShellElement.compute_stiffness_matrix
_QUALIFIED_Q4_BASE_MASS_KERNEL = ShellElement.compute_mass_matrix
_QUALIFIED_Q4_BASE_GEOMETRIC_KERNEL = ShellElement.compute_geometric_stiffness_matrix
_QUALIFIED_Q4_BASE_NODE_COORDINATES_KERNEL = ShellElement.get_node_coordinates
_QUALIFIED_Q4_BASE_LOCAL_FRAME_KERNEL = ShellElement._local_frame_and_derivatives
_QUALIFIED_Q4_BASE_MITC4_SHEAR_KERNEL = ShellElement._mitc4_shear_b_matrix
_QUALIFIED_Q4_BASE_SERIALIZATION_KERNEL = ShellElement.to_dict
_QUALIFIED_Q4_BASE_ELEMENT_SERIALIZATION_KERNEL = Element.to_dict


def _capture_q4_fast_array_authority(
    metadata: tuple[
        tuple[Any, str, tuple[int, ...], tuple[int, ...], bool], ...
    ],
) -> tuple[
    tuple[
        Any,
        str,
        tuple[int, ...],
        tuple[int, ...],
        bool,
        tuple[Any, ...],
    ], ...
]:
    made = []
    for array, dtype_string, shape, strides, c_contiguous in metadata:
        bases = []
        current: Any = array
        while type(current) is np.ndarray:
            current = current.base
            bases.append(current)
        made.append(
            (
                array,
                dtype_string,
                shape,
                strides,
                c_contiguous,
                tuple(bases),
            )
        )
    return tuple(made)


def _require_q4_fast_array_authority(
    authority: tuple[
        tuple[
            Any,
            str,
            tuple[int, ...],
            tuple[int, ...],
            bool,
            tuple[Any, ...],
        ], ...
    ],
    *,
    label: str,
) -> None:
    for array, dtype_string, shape, strides, c_contiguous, bases in authority:
        if (
            type(array) is not np.ndarray
            or array.dtype.str != dtype_string
            or array.shape != shape
            or array.strides != strides
            or bool(array.flags.c_contiguous) is not c_contiguous
            or array.flags.writeable
        ):
            raise ValueError(f"{label} ndarray metadata changed")
        current: Any = array
        for expected_base in bases:
            if current.base is not expected_base:
                raise ValueError(f"{label} ndarray base changed")
            current = expected_base
        if not (
            type(current) is bytes
            or isinstance(current, memoryview) and current.readonly
        ):
            raise ValueError(f"{label} ndarray base changed")


_Q4_FAST_BASE_CLASS_ENTRY_NAMES = MappingProxyType(
    {
        Element: (
            "get_node_coordinates",
            "total_dofs",
        ),
        _fe_core_module.FEMesh: (
            "get_node",
            "revision_signature",
        ),
        ShellElement: (
            "compute_stiffness_matrix",
            "num_nodes",
            "dofs_per_node",
            "_local_frame_and_derivatives",
            "compute_jacobian",
            "_normalize",
            "_fallback_edge_direction",
            "_mitc4_shear_b_matrix",
            "_mitc4_shear_samples",
            "_center_frame",
            "_reference_center",
            "compute_shape_functions",
            "_compute_4node_shape_functions",
            "_build_shell_b_matrices",
            "_build_drilling_b_matrix",
            "_local_dof_transform",
            "_material_angle",
            "_hourglass_stabilization_matrix",
            "_rigid_body_mode_matrix",
            "gauss_points",
            "gauss_weights",
            "shear_gauss_points",
            "shear_gauss_weights",
            "GAUSS_POINTS_2x2",
            "GAUSS_WEIGHTS_2x2",
            "GAUSS_POINTS_1x1",
            "GAUSS_WEIGHTS_1x1",
            "_MITC4_SAMPLE_POINTS",
        ),
    }
)
_q4_fast_base_class_authority = tuple(
    (
        owner,
        names,
        itemgetter(*names),
        tuple(
            type.__getattribute__(owner, "__dict__")[name]
            for name in names
        ),
    )
    for owner, names in _Q4_FAST_BASE_CLASS_ENTRY_NAMES.items()
)
(
    _q4_fast_mesh_class_owner,
    _q4_fast_mesh_class_names,
    _q4_fast_mesh_class_getter,
    _q4_fast_mesh_class_values,
) = next(
    authority
    for authority in _q4_fast_base_class_authority
    if authority[0] is _fe_core_module.FEMesh
)
_q4_fast_mutable_class_authority = tuple(
    (
        f"{owner.__name__}.{name}",
        value,
        tuple(value.items()),
    )
    for owner, names, _getter, values in _q4_fast_base_class_authority
    for name, value in zip(names, values)
    if type(value) is dict
)
_q4_fast_base_array_authority = _capture_q4_fast_array_authority(
    _capture_authority_array_metadata(
        tuple(_Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY.values())
    )
)
_Q4_FAST_MESH_TYPE = _fe_core_module.FEMesh
_Q4_FAST_NODE_TYPE = Node
_Q4_FAST_STATE_MAPPING_TYPE = _fe_core_module._QualifiedStateMapping
_Q4_FAST_TOKEN_TYPE = _fe_core_module._QualifiedMutationEpoch
_Q4_FAST_MATERIAL_TYPE = _fe_core_module.Material
_Q4_FAST_PUBLIC_ARRAY_CONSTRUCTOR = np.ndarray
# Exact immutable scalar only: production builders retain NumPy coordinates.
# This binding is covered by the existing formulation and NumPy authorities.
_Q4_FAST_COORDINATE_FLOAT64_TYPE = np.float64
_Q4_FAST_ELEMENT_INPUT_NAMES = (
    "element_id",
    "node_ids",
    "material_name",
    "thickness",
    "drilling_stabilization",
    "hourglass_stabilization",
    "reduced_integration",
    "_is_3node",
    "_is_4node",
    "_is_6node",
    "_is_8node",
    "_is_triangular",
    "_is_quadrilateral",
    "material_angle_deg",
    "material_direction",
    "shell_section",
    "reference_normal",
    "director_polarity",
    "pl_stabilization",
    "planar_tolerance",
    "warped_formulation",
    "_qualified_plan_state_revision",
)
_Q4_FAST_FORBIDDEN_INSTANCE_SHADOWS = (
    "formulation_id",
    "implementation_id",
    "legacy_stiffness_batch_eligible",
    "legacy_nonlinear_batch_eligible",
    "quadrature_authority_id",
    "recovery_policy_id",
    "compute_stiffness_components",
    "compute_stiffness_matrix",
    "get_node_coordinates",
    "compute_shape_functions",
    "compute_jacobian",
    "_compute_4node_shape_functions",
    "_local_frame_and_derivatives",
    "_local_dof_transform",
    "_build_shell_b_matrices",
    "_build_drilling_b_matrix",
    "_material_angle",
    "_center_frame",
    "_reference_center",
    "_mitc4_shear_samples",
    "_mitc4_shear_b_matrix",
    "_hourglass_stabilization_matrix",
    "_rigid_body_mode_matrix",
    "_qualified_stiffness_cache_key",
    "_stationary_blocks",
    "_constitutive_and_drill_stiffness",
    "_physical_director_context",
    "_warped_generalized_drilling_correction",
    "_inverse_planar_jacobian",
)
_Q4_FAST_FORBIDDEN_MESH_SHADOWS = (
    "get_node",
    "revision_signature",
)
_Q4_FAST_OWNED_MESH_DERIVED_KEYS = frozenset(
    {
        "_sparsity_cache",
        "_topology_signature_cache",
        "_qualified_s3_reference_stiffness_plan",
        "_recovery_batch_plan",
    }
)
_Q4_FAST_FORBIDDEN_MATERIAL_SHADOWS = (
    "elastic_symmetry",
    "shear_modulus",
    "is_nonlinear",
    "elastic_compliance_matrix",
)
_Q4_FAST_MATERIAL_INPUT_NAMES = (
    "name",
    "elastic_modulus",
    "poisson_ratio",
    "density",
    "yield_stress",
    "hardening_curve",
)
_Q4_FAST_ELEMENT_ITEMGETTER = itemgetter(*_Q4_FAST_ELEMENT_INPUT_NAMES)
_Q4_FAST_MATERIAL_ITEMGETTER = itemgetter(*_Q4_FAST_MATERIAL_INPUT_NAMES)
_Q4_FAST_NODE_ITEMGETTER = itemgetter("x", "y", "z", "_coordinate_revision")
_Q4_FAST_MATERIAL_CLASS_NAMES = (
    "elastic_symmetry",
    "shear_modulus",
    "is_nonlinear",
    "elastic_compliance_matrix",
)
_Q4_FAST_MATERIAL_CLASS_ITEMGETTER = itemgetter(
    *_Q4_FAST_MATERIAL_CLASS_NAMES
)
_Q4_FAST_MATERIAL_CLASS_VALUES = _Q4_FAST_MATERIAL_CLASS_ITEMGETTER(
    type.__getattribute__(_Q4_FAST_MATERIAL_TYPE, "__dict__")
)


class _Q4FastInputSnapshot(NamedTuple):
    element_namespace: dict[str, Any]
    element_namespace_keys: tuple[str, ...]
    element_namespace_key_types: tuple[type[Any], ...]
    element_values: tuple[Any, ...]
    mesh: Any
    mesh_namespace: dict[str, Any]
    mesh_namespace_keys: tuple[str, ...]
    mesh_namespace_key_types: tuple[type[Any], ...]
    nodes: Any
    nodes_namespace: dict[str, Any]
    nodes_namespace_keys: tuple[str, ...]
    nodes_namespace_key_types: tuple[type[Any], ...]
    token: Any
    token_value: int
    nodes_snapshot: tuple[
        tuple[
            int,
            Any,
            dict[str, Any],
            tuple[type[Any], ...],
            tuple[Any, ...],
        ], ...
    ]
    material: Any
    material_namespace: dict[str, Any]
    material_namespace_keys: tuple[str, ...]
    material_namespace_key_types: tuple[type[Any], ...]
    material_value_types: tuple[type[Any], ...]
    material_values: tuple[Any, ...]
    vector_metadata: Any


class _PreparedQ4LinearCorrection:
    """Owner-local immutable preparation, never a checkpoint/material state."""
    __slots__ = ("__weakref__",)


def _make_prepared_q4_correction_provenance():
    records = weakref.WeakKeyDictionary()

    def prepare(element, mesh, material, post_observation):
        correction = element._qualified_linear_correction(
            mesh, material, 5, post_observation=post_observation)
        if element._qualified_component_guard is None:
            return None
        element._validate_qualified_component_guard(post_observation=post_observation)
        inputs = _capture_q4_fast_input_snapshot(element, mesh, material)
        if inputs is None:
            return None
        record = _PreparedQ4LinearCorrection()
        # Payload and original binding stay closure-owned. There is no arbitrary
        # bytes registrar, and no record attribute is trusted during evaluation.
        records[record] = (weakref.ref(record), element, mesh, material,
            element._qualified_component_guard, inputs,
            np.ascontiguousarray(correction, dtype=float).tobytes())
        return record

    def lookup(record):
        value = records.get(record)
        if value is None or value[0]() is not record:
            return None
        return value[1:]

    return prepare, lookup


_prepare_q4_correction_payload, _lookup_prepared_q4_correction = (
    _make_prepared_q4_correction_provenance())


def _make_elastic_batch_baseline_preparation():
    active = ContextVar("qualified_q4_batch_baselines", default=None)

    owners = weakref.WeakKeyDictionary()

    class Scope:
        __slots__ = ("__weakref__",)

    @contextmanager
    def scope():
        # A solve owns its bindings, including mesh graphs. Nested solves have
        # separate dictionaries; exit/cancellation releases all preparation.
        owned = Scope()
        records = {}
        owners[owned] = records
        token = active.set(owned)
        try:
            yield
        finally:
            owners.pop(owned, None)
            records.clear()
            active.reset(token)

    def prepare(element, mesh, material, snapshot, post_observation):
        owned = active.get()
        records = owners.get(owned) if type(owned) is Scope else None
        if records is None:
            return None
        _require_exact_q4_runtime_authority(
            element, context="qualified Q4 batch baseline preparation")
        # The factory obtains authoritative inputs itself; callers cannot
        # publish arbitrary matrices or geometry through a preparation token.
        geometry = element._nonlinear_geometry(mesh)
        qualified = np.asarray(element.compute_stiffness_matrix(mesh, material), dtype=float)
        if not element._defining_class_input_snapshot_matches(mesh, material, snapshot):
            raise RuntimeError("qualified Q4 elastic increment inputs changed during preparation")
        key = id(element)
        guard = element._qualified_component_guard

        def valid(binding):
            return (binding[0] is element and binding[1] is mesh
                    and binding[2] is material and binding[3] is guard
                    and binding[5] is geometry
                    and element._qualified_component_guard is guard
                    and element._defining_class_input_snapshot_matches(
                        mesh, material, binding[4])
                    and element._defining_class_input_snapshot_matches(
                        mesh, material, snapshot))

        def observation():
            if post_observation is not None:
                post_observation()
            _require_exact_q4_runtime_authority(
                element, context="qualified Q4 batch baseline observation")
            if not element._defining_class_input_snapshot_matches(mesh, material, snapshot):
                raise RuntimeError("qualified Q4 elastic baseline inputs changed during observation")
            element._validate_qualified_component_guard()

        binding = records.pop(key, None)
        if binding is None or not valid(binding):
            batch = _elastic_increment_module.compensated_elastic_batch_zero(
                geometry, float(material.elastic_modulus), float(material.poisson_ratio),
                float(material.shear_modulus), element.thickness,
                element.drilling_stabilization)
            element._hourglass_stiffness_matrix = None
            scalar = _elements_module._compensated_elastic_zero_tangent(
                element, mesh, material, observation)
            observation()
            offset = _local_contractions.sum_arrays((batch, -scalar))
            baseline = _local_contractions.sum_arrays((qualified, offset))
            binding = (element, mesh, material, guard, snapshot, geometry,
                       np.ascontiguousarray(baseline, dtype=float).tobytes())
        else:
            element._hourglass_stiffness_matrix = None

        def publish():
            # No arbitrary-payload registrar: only this call's internally
            # computed immutable bytes can be published, after final guards.
            observation()
            if not valid(binding):
                raise RuntimeError("qualified Q4 elastic baseline preparation is stale")
            if (records is not None and owners.get(owned) is records
                    and active.get() is owned):
                records[key] = binding

        return np.frombuffer(binding[6], dtype=float).reshape(24, 24), publish

    return prepare, scope


_prepare_elastic_batch_baseline, _elastic_batch_baseline_scope = (
    _make_elastic_batch_baseline_preparation())


class _Q4FastCacheRecord(NamedTuple):
    reference: Any
    components: Any
    cache_key: Any
    guard: Any
    total: Any
    total_bytes: bytes
    total_metadata: Any
    fast_array_authority: Any
    total_public_validated: bool
    total_prevalidated: bool
    component_items: tuple[tuple[str, Any], ...]
    component_metadata: tuple[
        tuple[Any, str, tuple[int, ...], tuple[int, ...], bool], ...
    ]
    inputs: Optional[_Q4FastInputSnapshot]


def _require_q4_fast_base_authority() -> None:
    """Check bounded builtin mesh/material cached authority.

    Every inherited stiffness kernel/property is shadowed onto the protected
    exact final class before its authority freeze.  Base-class monkeypatches
    therefore cannot influence qualified cold or warm mechanics and do not
    belong on the steady cached path.
    """

    mesh_namespace = type.__getattribute__(
        _q4_fast_mesh_class_owner,
        "__dict__",
    )
    try:
        current_mesh_values = _q4_fast_mesh_class_getter(mesh_namespace)
    except KeyError as error:
        raise ValueError(
            "qualified Q4 builtin mesh class authority changed"
        ) from error
    if (
        current_mesh_values[0] is not _q4_fast_mesh_class_values[0]
        or current_mesh_values[1] is not _q4_fast_mesh_class_values[1]
    ):
        raise ValueError(
            "qualified Q4 builtin mesh class authority changed"
        )

    material_namespace = type.__getattribute__(
        _Q4_FAST_MATERIAL_TYPE,
        "__dict__",
    )
    try:
        current = _Q4_FAST_MATERIAL_CLASS_ITEMGETTER(material_namespace)
    except KeyError as error:
        raise ValueError(
            "qualified Q4 builtin material class authority changed"
        ) from error
    if not all(map(_operator_is, current, _Q4_FAST_MATERIAL_CLASS_VALUES)):
        raise ValueError(
            "qualified Q4 builtin material class authority changed"
        )


def _capture_q4_fast_input_snapshot(
    element: Any,
    mesh: Any,
    material: Any,
) -> Optional[_Q4FastInputSnapshot]:
    """Capture raw builtin/NumPy-float64 inputs without numeric conversion."""

    if vars(np).get("float64") is not _Q4_FAST_COORDINATE_FLOAT64_TYPE:
        raise ValueError("qualified Q4 coordinate scalar authority changed")
    if type(mesh) is not _Q4_FAST_MESH_TYPE or type(material) is not _Q4_FAST_MATERIAL_TYPE:
        return None
    element_namespace = object.__getattribute__(element, "__dict__")
    mesh_namespace = object.__getattribute__(mesh, "__dict__")
    material_namespace = object.__getattribute__(material, "__dict__")
    if (
        type(element_namespace) is not dict
        or type(mesh_namespace) is not dict
        or type(material_namespace) is not dict
        or not all(type(name) is str for name in element_namespace)
        or not all(type(name) is str for name in mesh_namespace)
        or not all(type(name) is str for name in material_namespace)
        or dict.get(element_namespace, "shell_section") is not None
        or any(
            name in element_namespace
            for name in _Q4_FAST_FORBIDDEN_INSTANCE_SHADOWS
        )
        or any(name in mesh_namespace for name in _Q4_FAST_FORBIDDEN_MESH_SHADOWS)
        or any(
            name in material_namespace
            for name in _Q4_FAST_FORBIDDEN_MATERIAL_SHADOWS
        )
    ):
        return None
    try:
        element_values = _Q4_FAST_ELEMENT_ITEMGETTER(element_namespace)
    except KeyError:
        return None
    node_ids = dict.get(element_namespace, "node_ids")
    nodes = dict.get(mesh_namespace, "nodes")
    token = dict.get(mesh_namespace, "_qualified_direct_state_token")
    nodes_namespace = (
        object.__getattribute__(nodes, "__dict__")
        if type(nodes) is _Q4_FAST_STATE_MAPPING_TYPE
        else None
    )
    if (
        type(node_ids) is not tuple
        or len(node_ids) != 4
        or not all(type(node_id) is int for node_id in node_ids)
        or type(nodes) is not _Q4_FAST_STATE_MAPPING_TYPE
        or "get"
        in type.__getattribute__(
            _Q4_FAST_STATE_MAPPING_TYPE,
            "__dict__",
        )
        or type(nodes_namespace) is not dict
        or not all(type(name) is str for name in nodes_namespace)
        or "get" in nodes_namespace
        or type(token) is not _Q4_FAST_TOKEN_TYPE
        or len(token) != 1
        or type(token[0]) is not int
        or dict.get(nodes_namespace, "_qualified_token") is not token
        or dict.get(nodes_namespace, "_qualified_kind") != "node"
    ):
        return None
    node_records = []
    for node_id in node_ids:
        node = dict.get(nodes, node_id)
        if type(node) is not _Q4_FAST_NODE_TYPE:
            return None
        namespace = object.__getattribute__(node, "__dict__")
        if type(namespace) is not dict or not all(
            type(name) is str for name in namespace
        ):
            return None
        try:
            values = _Q4_FAST_NODE_ITEMGETTER(namespace)
        except KeyError:
            return None
        if (
            any(type(value) not in {int, float, _Q4_FAST_COORDINATE_FLOAT64_TYPE}
                or type(value) is bool for value in values[:3])
            or type(values[3]) is not int
        ):
            return None
        node_records.append(
            (
                node_id,
                node,
                namespace,
                tuple(type(value) for value in values),
                values,
            )
        )
    try:
        raw_material_values = _Q4_FAST_MATERIAL_ITEMGETTER(material_namespace)
    except KeyError:
        return None
    for name, value in zip(_Q4_FAST_MATERIAL_INPUT_NAMES, raw_material_values):
        if name == "hardening_curve":
            if value is not None:
                return None
        elif name == "name":
            if type(value) is not str:
                return None
        elif type(value) not in {int, float} or type(value) is bool:
            return None
    vectors = tuple(
        value
        for name, value in zip(_Q4_FAST_ELEMENT_INPUT_NAMES, element_values)
        if name in {"material_direction", "reference_normal"}
        and value is not None
    )
    try:
        vector_metadata = _capture_q4_fast_array_authority(
            _capture_authority_array_metadata(vectors)
        )
        _require_q4_fast_array_authority(
            vector_metadata,
            label="qualified Q4 warm input authority",
        )
    except (TypeError, ValueError):
        return None
    return _Q4FastInputSnapshot(
        element_namespace=element_namespace,
        element_namespace_keys=tuple(element_namespace),
        element_namespace_key_types=tuple(map(type, element_namespace)),
        element_values=element_values,
        mesh=mesh,
        mesh_namespace=mesh_namespace,
        mesh_namespace_keys=tuple(
            name
            for name in mesh_namespace
            if name not in _Q4_FAST_OWNED_MESH_DERIVED_KEYS
        ),
        mesh_namespace_key_types=tuple(
            type(name)
            for name in mesh_namespace
            if name not in _Q4_FAST_OWNED_MESH_DERIVED_KEYS
        ),
        nodes=nodes,
        nodes_namespace=nodes_namespace,
        nodes_namespace_keys=tuple(nodes_namespace),
        nodes_namespace_key_types=tuple(map(type, nodes_namespace)),
        token=token,
        token_value=int(token[0]),
        nodes_snapshot=tuple(node_records),
        material=material,
        material_namespace=material_namespace,
        material_namespace_keys=tuple(material_namespace),
        material_namespace_key_types=tuple(map(type, material_namespace)),
        material_value_types=tuple(type(value) for value in raw_material_values),
        material_values=raw_material_values,
        vector_metadata=vector_metadata,
    )


def _q4_fast_input_snapshot_matches(
    element: Any,
    mesh: Any,
    material: Any,
    snapshot: _Q4FastInputSnapshot,
) -> bool:
    """Compare only raw owned builtin state; never invoke a descriptor."""

    element_keys = snapshot.element_namespace
    nodes_keys = snapshot.nodes_namespace
    material_keys = snapshot.material_namespace

    if (
        vars(np).get("float64") is not _Q4_FAST_COORDINATE_FLOAT64_TYPE
        or type(element) is not QualifiedE4PLShellElement
        or type(mesh) is not _Q4_FAST_MESH_TYPE
        or type(material) is not _Q4_FAST_MATERIAL_TYPE
        or type(snapshot.nodes) is not _Q4_FAST_STATE_MAPPING_TYPE
        or "get"
        in type.__getattribute__(
            _Q4_FAST_STATE_MAPPING_TYPE,
            "__dict__",
        )
        or type(snapshot.token) is not _Q4_FAST_TOKEN_TYPE
        or mesh is not snapshot.mesh
        or material is not snapshot.material
        or object.__getattribute__(element, "__dict__")
        is not snapshot.element_namespace
        or object.__getattribute__(mesh, "__dict__") is not snapshot.mesh_namespace
        or object.__getattribute__(material, "__dict__")
        is not snapshot.material_namespace
        or len(element_keys) != len(snapshot.element_namespace_keys)
        or not all(
            map(
                _operator_is,
                element_keys,
                snapshot.element_namespace_keys,
            )
        )
        or any(
            name in snapshot.mesh_namespace
            for name in _Q4_FAST_FORBIDDEN_MESH_SHADOWS
        )
        or len(material_keys) != len(snapshot.material_namespace_keys)
        or not all(
            map(
                _operator_is,
                material_keys,
                snapshot.material_namespace_keys,
            )
        )
        or dict.get(snapshot.mesh_namespace, "nodes") is not snapshot.nodes
        or dict.get(snapshot.mesh_namespace, "_qualified_direct_state_token")
        is not snapshot.token
        or len(snapshot.token) != 1
        or type(snapshot.token[0]) is not int
        or int(snapshot.token[0]) != snapshot.token_value
        or object.__getattribute__(snapshot.nodes, "__dict__")
        is not snapshot.nodes_namespace
        or len(nodes_keys) != len(snapshot.nodes_namespace_keys)
        or not all(
            map(
                _operator_is,
                nodes_keys,
                snapshot.nodes_namespace_keys,
            )
        )
        or dict.get(snapshot.nodes_namespace, "_qualified_token")
        is not snapshot.token
        or dict.get(snapshot.nodes_namespace, "_qualified_kind") != "node"
    ):
        return False
    try:
        current_element_values = _Q4_FAST_ELEMENT_ITEMGETTER(
            snapshot.element_namespace
        )
    except KeyError:
        return False
    if not all(
        map(
            _operator_is,
            current_element_values,
            snapshot.element_values,
        )
    ):
        return False
    for node_id, node, namespace, value_types, values in snapshot.nodes_snapshot:
        if (
            type(node) is not _Q4_FAST_NODE_TYPE
            or object.__getattribute__(node, "__dict__") is not namespace
            or dict.get(snapshot.nodes, node_id) is not node
        ):
            return False
        try:
            current_values = _Q4_FAST_NODE_ITEMGETTER(namespace)
        except KeyError:
            return False
        if (
            len(current_values) != len(values)
            or not all(map(_operator_is, current_values, values))
        ):
            return False
    try:
        current_material_values = _Q4_FAST_MATERIAL_ITEMGETTER(
            snapshot.material_namespace
        )
    except KeyError:
        return False
    if not all(
        map(
            _operator_is,
            current_material_values,
            snapshot.material_values,
        )
    ):
        return False
    return True


def _make_q4_component_cache_provenance() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Keep warm component provenance outside caller-reachable instances."""

    records: Dict[int, _Q4FastCacheRecord] = {}

    def bind(element: Any, guard: Any, mesh: Any, material: Any) -> None:
        namespace = object.__getattribute__(element, "__dict__")
        components = dict.get(namespace, "_qualified_components")
        cache_key = dict.get(namespace, "_qualified_cache_key")
        total = (
            components.get("total")
            if type(components) is MappingProxyType
            else None
        )
        total_metadata = _capture_q4_fast_array_authority(
            _capture_authority_array_metadata(total)
        )
        inputs = _capture_q4_fast_input_snapshot(element, mesh, material)
        fast_array_authority = (
            _q4_fast_base_array_authority
            + total_metadata
            + (() if inputs is None else inputs.vector_metadata)
        )
        identity = id(element)

        def discard(reference: Any, *, expected_identity: int = identity) -> None:
            current = records.get(expected_identity)
            if current is not None and current.reference is reference:
                records.pop(expected_identity, None)

        records[identity] = _Q4FastCacheRecord(
            reference=weakref.ref(element, discard),
            components=components,
            cache_key=cache_key,
            guard=guard,
            total=total,
            total_bytes=(
                np.ascontiguousarray(total, dtype=np.float64).tobytes(order="C")
                if type(total) is np.ndarray and total.shape == (24, 24)
                else b""
            ),
            total_metadata=total_metadata,
            fast_array_authority=fast_array_authority,
            total_public_validated=bool(
                type(total) is np.ndarray
                and total.shape == (24, 24)
                and total.dtype == np.dtype(np.float64)
                and np.all(np.isfinite(total))
            ),
            total_prevalidated=bool(
                type(total) is np.ndarray
                and total.shape == (24, 24)
                and total.dtype == np.dtype(np.float64)
                and np.all(np.isfinite(total))
                and np.array_equal(total, total.T)
            ),
            component_items=tuple(components.items()),
            component_metadata=_capture_authority_array_metadata(components),
            inputs=inputs,
        )

    def clear(element: Any) -> None:
        records.pop(id(element), None)

    def retire_private(element: Any, mesh: Any) -> None:
        # Only the explicitly owned mesh may retire its record.  Check closure
        # provenance, not a caller-mutable instance guard, before any removal.
        record = records.get(id(element))
        if record is not None and (
            record.reference() is not element or record.guard[0] is not mesh
        ):
            raise ValueError("Q4 private retirement mesh does not own provenance")
        if record is not None:
            records.pop(id(element))

    def require_identity(element: Any, components: Any, cache_key: Any, guard: Any) -> None:
        record = records.get(id(element))
        if (
            record is None
            or record.reference() is not element
            or record.components is not components
            or record.cache_key is not cache_key
            or record.guard is not guard
            or type(components) is not MappingProxyType
            or type(cache_key) is not tuple
            or type(guard) is not tuple
        ):
            raise RuntimeError(
                "qualified Q4 component cache closure provenance changed"
            )
        actual_items = tuple(components.items())
        if (
            len(actual_items) != len(record.component_items)
            or any(
                type(actual_name) is not str
                or actual_name != expected_name
                or actual_value is not expected_value
                for (actual_name, actual_value), (
                    expected_name,
                    expected_value,
                ) in zip(actual_items, record.component_items)
            )
        ):
            raise RuntimeError(
                "qualified Q4 component cache mapping provenance changed"
            )
        _require_authority_array_metadata(
            record.component_metadata,
            label="qualified Q4 component cache authority",
        )

    def try_cached(element: Any, mesh: Any, material: Any) -> Optional[np.ndarray]:
        if type(element) is not QualifiedE4PLShellElement:
            raise TypeError(
                "qualified Q4 cached stiffness requires the exact final class"
            )
        namespace = object.__getattribute__(element, "__dict__")
        components = dict.get(namespace, "_qualified_components")
        if components is None:
            return None
        cache_key = dict.get(namespace, "_qualified_cache_key")
        guard = dict.get(namespace, "_qualified_component_guard")
        record = records.get(id(element))
        if (
            record is None
            or record.reference() is not element
            or record.components is not components
            or record.cache_key is not cache_key
            or record.guard is not guard
            or type(components) is not MappingProxyType
            or type(cache_key) is not tuple
            or type(guard) is not tuple
            or components.get("total") is not record.total
        ):
            raise RuntimeError(
                "qualified Q4 cached stiffness closure provenance changed"
            )
        _require_q4_fast_array_authority(
            record.fast_array_authority,
            label="qualified Q4 cached stiffness authority",
        )
        if not record.total_prevalidated:
            # The established varying-frame warped kernel is symmetric to
            # binary64 roundoff, but its accumulation is not necessarily
            # bitwise symmetric.  That legitimate result remains ineligible
            # for the total-only fast path; route it through the fully guarded
            # public component path instead of misclassifying it as damaged
            # closure provenance.
            if record.total_public_validated:
                return None
            raise RuntimeError(
                "qualified Q4 cached stiffness closure provenance changed"
            )
        snapshot = record.inputs
        if snapshot is None:
            return None
        _require_q4_fast_base_authority()
        if type(record.total) is not np.ndarray:
            raise RuntimeError("qualified Q4 cached stiffness is not an exact array")
        if not _q4_fast_input_snapshot_matches(
            element,
            mesh,
            material,
            snapshot,
        ):
            records.pop(id(element), None)
            for name in (
                "_hourglass_stiffness_matrix",
                "_qualified_cache_key",
                "_qualified_component_guard",
                "_qualified_components",
                "_stiffness_matrix",
            ):
                object.__setattr__(element, name, None)
            return None
        if len(record.total_bytes) != 24 * 24 * 8:
            raise RuntimeError("qualified Q4 cached stiffness bytes are incomplete")
        # The authoritative bytes never escape the closure.  Every public
        # matrix result is a disposable readonly view, so caller-controlled
        # ndarray metadata cannot race or poison later warm operations.
        return _Q4_FAST_PUBLIC_ARRAY_CONSTRUCTOR(
            (24, 24),
            dtype=np.float64,
            buffer=record.total_bytes,
        )

    def public_view(element: Any) -> np.ndarray:
        """Return one disposable matrix view from closure-owned exact bytes."""

        namespace = object.__getattribute__(element, "__dict__")
        components = dict.get(namespace, "_qualified_components")
        cache_key = dict.get(namespace, "_qualified_cache_key")
        guard = dict.get(namespace, "_qualified_component_guard")
        require_identity(element, components, cache_key, guard)
        record = records.get(id(element))
        if (
            record is None
            or components.get("total") is not record.total
            or not record.total_public_validated
            or len(record.total_bytes) != 24 * 24 * 8
        ):
            raise RuntimeError(
                "qualified Q4 public stiffness closure provenance changed"
            )
        _require_q4_fast_array_authority(
            record.total_metadata,
            label="qualified Q4 public stiffness authority",
        )
        return _Q4_FAST_PUBLIC_ARRAY_CONSTRUCTOR(
            (24, 24),
            dtype=np.float64,
            buffer=record.total_bytes,
        )

    def try_assembly_cached(
        element: Any,
        mesh: Any,
        material: Any,
    ) -> Optional[np.ndarray]:
        """Return a total-only view after bounded shared-input checks.

        The assembly lease separately binds the exact model/material map,
        every material scalar, all node coordinates through the monotonic mesh
        token, and the formulation generation.  This accessor therefore
        verifies only per-element cache provenance and the raw element inputs
        that cannot be amortized across a batch.
        """

        if type(element) is not QualifiedE4PLShellElement:
            return None
        namespace = object.__getattribute__(element, "__dict__")
        components = dict.get(namespace, "_qualified_components")
        cache_key = dict.get(namespace, "_qualified_cache_key")
        guard = dict.get(namespace, "_qualified_component_guard")
        record = records.get(id(element))
        snapshot = None if record is None else record.inputs
        if (
            record is None
            or snapshot is None
            or record.reference() is not element
            or record.components is not components
            or record.cache_key is not cache_key
            or record.guard is not guard
            or type(components) is not MappingProxyType
            or type(cache_key) is not tuple
            or type(guard) is not tuple
            or components.get("total") is not record.total
            or not record.total_prevalidated
            or len(record.total_bytes) != 24 * 24 * 8
            or mesh is not snapshot.mesh
            or material is not snapshot.material
            or type(material) is not _Q4_FAST_MATERIAL_TYPE
            or object.__getattribute__(material, "__dict__")
            is not snapshot.material_namespace
            or namespace is not snapshot.element_namespace
            or len(namespace) != len(snapshot.element_namespace_keys)
            or not all(
                map(
                    _operator_is,
                    namespace,
                    snapshot.element_namespace_keys,
                )
            )
            or type(mesh) is not _Q4_FAST_MESH_TYPE
            or object.__getattribute__(mesh, "__dict__")
            is not snapshot.mesh_namespace
            or dict.get(snapshot.mesh_namespace, "nodes") is not snapshot.nodes
            or dict.get(
                snapshot.mesh_namespace,
                "_qualified_direct_state_token",
            )
            is not snapshot.token
            or type(snapshot.token) is not _Q4_FAST_TOKEN_TYPE
            or len(snapshot.token) != 1
            or type(snapshot.token[0]) is not int
            or int(snapshot.token[0]) != snapshot.token_value
        ):
            return None
        try:
            current_values = _Q4_FAST_ELEMENT_ITEMGETTER(namespace)
            current_material_values = _Q4_FAST_MATERIAL_ITEMGETTER(
                snapshot.material_namespace
            )
        except KeyError:
            return None
        if (
            not all(map(_operator_is, current_values, snapshot.element_values))
            or not all(
                map(
                    _operator_is,
                    current_material_values,
                    snapshot.material_values,
                )
            )
        ):
            return None
        _require_q4_fast_array_authority(
            record.fast_array_authority,
            label="qualified Q4 assembly cached stiffness authority",
        )
        # Assembly consumes the closure-owned immutable payload, never the
        # public ndarray view whose metadata a caller may legally reshape.
        return record.total_bytes

    return (
        bind,
        clear,
        require_identity,
        try_cached,
        public_view,
        try_assembly_cached,
        retire_private,
    )


(
    _bind_q4_component_cache_provenance,
    _clear_q4_component_cache_provenance,
    _require_q4_component_cache_provenance,
    _try_q4_fast_cached_stiffness,
    _q4_public_stiffness_view,
    _try_q4_fast_assembly_cached_stiffness,
    _retire_q4_private_component_provenance,
) = _make_q4_component_cache_provenance()
_Q4_GENERALIZED_SECTION_NAMESPACE_AUTHORITY = MappingProxyType(
    dict(type.__getattribute__(GeneralizedShellSection, "__dict__"))
)
_Q4_SERIALIZATION_CLASS_IDENTITY = MappingProxyType(
    {
        "formulation_id": FORMULATION_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "recovery_policy_id": RECOVERY_POLICY_ID,
        "stationary_solve_policy_id": STATIONARY_SOLVE_POLICY_ID,
        "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
        "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
        "current_state_binding_schema_id": Q4_CURRENT_STATE_BINDING_SCHEMA_ID,
        "current_state_algorithmic_origin_schema_id": (
            Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
        ),
        "current_state_tangent_decomposition_policy_id": (
            Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
        ),
        "current_state_projection_policy_id": Q4_CURRENT_STATE_PROJECTION_POLICY_ID,
        "activity_disposition_schema_id": Q4_ACTIVITY_DISPOSITION_SCHEMA_ID,
        "deleted_frozen_policy_id": Q4_DELETED_FROZEN_POLICY_ID,
        "failed_state_policy_id": Q4_FAILED_STATE_POLICY_ID,
        "quadrature_authority_id": Q4_QUADRATURE_AUTHORITY_ID,
    }
)
_Q4_SERIALIZATION_GLOBAL_IDENTITY = MappingProxyType(
    {
        "np": np,
        "math": math,
        "Element": Element,
        "ShellElement": ShellElement,
        "GeneralizedShellSection": GeneralizedShellSection,
        "FORMULATION_ID": FORMULATION_ID,
        "_PLANAR_FORMULATION_ID": _PLANAR_FORMULATION_ID,
        "IMPLEMENTATION_ID": IMPLEMENTATION_ID,
        "RECOVERY_POLICY_ID": RECOVERY_POLICY_ID,
        "STATIONARY_SOLVE_POLICY_ID": STATIONARY_SOLVE_POLICY_ID,
        "DIRECTOR_POLARITY_POLICY_ID": DIRECTOR_POLARITY_POLICY_ID,
        "DIRECTOR_REVERSAL_TRANSFORM_ID": DIRECTOR_REVERSAL_TRANSFORM_ID,
        "Q4_CURRENT_STATE_BINDING_SCHEMA_ID": Q4_CURRENT_STATE_BINDING_SCHEMA_ID,
        "Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID": (
            Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
        ),
        "Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID": (
            Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
        ),
        "Q4_CURRENT_STATE_PROJECTION_POLICY_ID": (
            Q4_CURRENT_STATE_PROJECTION_POLICY_ID
        ),
        "Q4_ACTIVITY_DISPOSITION_SCHEMA_ID": Q4_ACTIVITY_DISPOSITION_SCHEMA_ID,
        "Q4_DELETED_FROZEN_POLICY_ID": Q4_DELETED_FROZEN_POLICY_ID,
        "Q4_FAILED_STATE_POLICY_ID": Q4_FAILED_STATE_POLICY_ID,
        "Q4_QUADRATURE_AUTHORITY_ID": Q4_QUADRATURE_AUTHORITY_ID,
        "_V5_IMPLEMENTATION_ID": _V5_IMPLEMENTATION_ID,
        "_V6_IMPLEMENTATION_ID": _V6_IMPLEMENTATION_ID,
        "_V6_CURRENT_STATE_BINDING_SCHEMA_ID": (
            _V6_CURRENT_STATE_BINDING_SCHEMA_ID
        ),
        "_V6_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID": (
            _V6_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
        ),
    }
)


def _require_q4_serialization_module_authority(
    expected_class: type[Any],
    *,
    _global_identity: Mapping[str, Any] = _Q4_SERIALIZATION_GLOBAL_IDENTITY,
    _class_identity: Mapping[str, Any] = _Q4_SERIALIZATION_CLASS_IDENTITY,
    _base_serializer: Any = _QUALIFIED_Q4_BASE_SERIALIZATION_KERNEL,
    _base_element_serializer: Any = (
        _QUALIFIED_Q4_BASE_ELEMENT_SERIALIZATION_KERNEL
    ),
    _section_class: type[Any] = GeneralizedShellSection,
    _section_namespace: Mapping[str, Any] = (
        _Q4_GENERALIZED_SECTION_NAMESPACE_AUTHORITY
    ),
    _static_lookup: Any = _static_mro_attribute,
    _base_class: type[Any] = ShellElement,
    _element_class: type[Any] = Element,
) -> None:
    if globals().get("QualifiedE4PLShellElement") is not expected_class:
        raise ValueError("qualified Q4 serialization requires the exact class")
    for name, expected in _global_identity.items():
        actual = globals().get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"qualified Q4 serialization global {name} authority is incompatible"
            )
    for name, expected in _class_identity.items():
        actual = _static_lookup(expected_class, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"qualified Q4 serialization {name} authority is incompatible"
            )
    if _static_lookup(_base_class, "to_dict") is not _base_serializer:
        raise ValueError(
            "qualified Q4 base serialization authority is incompatible"
        )
    if _static_lookup(_element_class, "to_dict") is not _base_element_serializer:
        raise ValueError(
            "qualified Q4 root serialization authority is incompatible"
        )
    actual_section_namespace = type.__getattribute__(_section_class, "__dict__")
    changed_section_members = set(actual_section_namespace).symmetric_difference(
        _section_namespace
    ) | {
        name
        for name, expected in _section_namespace.items()
        if name in actual_section_namespace
        and actual_section_namespace[name] is not expected
    }
    if changed_section_members:
        raise ValueError(
            "qualified Q4 generalized-section serialization authority is incompatible"
        )


def _validate_q4_serialization_authority(
    element: Any,
    *,
    expected_class: type[Any],
    _global_identity: Mapping[str, Any] = _Q4_SERIALIZATION_GLOBAL_IDENTITY,
    _class_identity: Mapping[str, Any] = _Q4_SERIALIZATION_CLASS_IDENTITY,
    _base_serializer: Any = _QUALIFIED_Q4_BASE_SERIALIZATION_KERNEL,
    _base_element_serializer: Any = (
        _QUALIFIED_Q4_BASE_ELEMENT_SERIALIZATION_KERNEL
    ),
    _section_class: type[Any] = GeneralizedShellSection,
    _section_namespace: Mapping[str, Any] = (
        _Q4_GENERALIZED_SECTION_NAMESPACE_AUTHORITY
    ),
    _module_guard: Any = _require_q4_serialization_module_authority,
    _static_lookup: Any = _static_mro_attribute,
    _numpy: Any = np,
    _isfinite: Any = math.isfinite,
    _warped_formulations: frozenset[str] = frozenset(_WARPED_FORMULATIONS),
) -> None:
    """Reject identity laundering before Q4 serialization or reconstruction."""

    _module_guard(
        expected_class,
        _global_identity=_global_identity,
        _class_identity=_class_identity,
        _base_serializer=_base_serializer,
        _base_element_serializer=_base_element_serializer,
        _section_class=_section_class,
        _section_namespace=_section_namespace,
        _static_lookup=_static_lookup,
    )
    if type(element) is not expected_class:
        raise ValueError("qualified Q4 serialization requires the exact class")
    namespace = object.__getattribute__(element, "__dict__")
    # The final-class certificate binds the installed guarded
    # ``__getattribute__`` descriptor exactly; it intentionally supersedes
    # the pre-hardening direct ``object.__getattribute__`` slot.
    _require_q4_final_class_authority()
    for name in (
        "element_id",
        "node_ids",
        "material_name",
        "thickness",
        "material_angle_deg",
        "material_direction",
        "shell_section",
        "drilling_stabilization",
        "reduced_integration",
        "hourglass_stabilization",
        "pl_stabilization",
        "planar_tolerance",
        "director_polarity",
        "reference_normal",
        "warped_formulation",
    ):
        if name not in namespace or _static_lookup(type(element), name) is not None:
            raise ValueError(
                f"qualified Q4 serialization {name} authority is incompatible"
            )
    for name, expected in _class_identity.items():
        actual = _static_lookup(type(element), name)
        if (
            name in namespace
            or type(actual) is not type(expected)
            or actual != expected
        ):
            raise ValueError(
                f"qualified Q4 serialization {name} authority is incompatible"
            )
    if (
        type(namespace["element_id"]) is not int
        or type(namespace["node_ids"]) is not tuple
        or len(namespace["node_ids"]) != 4
        or not all(type(value) is int for value in namespace["node_ids"])
        or type(namespace["material_name"]) is not str
        or type(namespace["thickness"]) is not float
        or type(namespace["material_angle_deg"]) is not float
        or type(namespace["drilling_stabilization"]) is not float
        or type(namespace["reduced_integration"]) is not bool
        or type(namespace["hourglass_stabilization"]) is not float
        or type(namespace["pl_stabilization"]) is not float
        or type(namespace["planar_tolerance"]) is not float
        or type(namespace["director_polarity"]) is not int
        or namespace["director_polarity"] not in {-1, 1}
        or type(namespace["warped_formulation"]) is not str
        or namespace["warped_formulation"] not in _warped_formulations
        or namespace["thickness"] <= 0.0
        or any(
            namespace[name] < 0.0
            for name in (
                "drilling_stabilization",
                "hourglass_stabilization",
                "pl_stabilization",
                "planar_tolerance",
            )
        )
        or (
            namespace["director_polarity"] != 1
            and namespace["reference_normal"] is None
        )
        or not all(
            _isfinite(namespace[name])
            for name in (
                "thickness",
                "material_angle_deg",
                "drilling_stabilization",
                "hourglass_stabilization",
                "pl_stabilization",
                "planar_tolerance",
            )
        )
        or (
            namespace["shell_section"] is not None
            and type(namespace["shell_section"]) is not _section_class
        )
    ):
        raise ValueError(
            "qualified Q4 serialization instance-data authority is incompatible"
        )
    for name in ("material_direction", "reference_normal"):
        vector = namespace[name]
        if vector is None:
            continue
        if (
            type(vector) is not _numpy.ndarray
            or vector.dtype != _numpy.dtype(_numpy.float64)
            or vector.shape != (3,)
            or not vector.flags.c_contiguous
            or vector.flags.writeable
            or not _numpy.all(_numpy.isfinite(vector))
        ):
            raise ValueError(
                f"qualified Q4 serialization {name} authority is incompatible"
            )
        if name == "reference_normal":
            norm = float(_numpy.linalg.norm(vector))
            if norm <= 0.0 or not _numpy.array_equal(vector / norm, vector):
                raise ValueError(
                    "qualified Q4 serialization reference_normal must be canonical"
                )


def _q4_serialized_output_boundary(method: Any) -> Any:
    """Capture immutable Q4 serialization guards without changing its API."""

    serialization_guard = _validate_q4_serialization_authority
    quadrature_guard = _validate_q4_quadrature_authority
    class_cell = dict(zip(method.__code__.co_freevars, method.__closure__ or ())).get(
        "__class__"
    )
    if class_cell is None:
        raise RuntimeError("qualified Q4 serialization method lacks class authority")

    def guarded(self: Any) -> Dict[str, Any]:
        expected_class = class_cell.cell_contents
        if type(self) is not expected_class:
            raise ValueError("qualified Q4 serialization requires the exact class")
        serialization_guard(self, expected_class=expected_class)
        quadrature_guard(self)
        payload = method(self)
        serialization_guard(self, expected_class=expected_class)
        quadrature_guard(self)
        return payload

    guarded.__name__ = method.__name__
    guarded.__qualname__ = method.__qualname__
    guarded.__doc__ = method.__doc__
    guarded.__annotations__ = dict(method.__annotations__)
    return guarded


def _q4_serialized_input_boundary(method: Any) -> Any:
    """Guard Q4 deserialization before parse and again before return."""

    module_guard = _require_q4_serialization_module_authority
    serialization_guard = _validate_q4_serialization_authority
    quadrature_guard = _validate_q4_quadrature_authority
    numerical_guard = require_exact_numpy_runtime_authority
    authority_signer = _module_authority_signature
    class_cell = dict(zip(method.__code__.co_freevars, method.__closure__ or ())).get(
        "__class__"
    )
    if class_cell is None:
        raise RuntimeError("qualified Q4 deserialization method lacks class authority")

    def guarded(cls: type[Any], payload: Mapping[str, Any]) -> Any:
        expected_class = class_cell.cell_contents
        if cls is not expected_class:
            raise ValueError("qualified Q4 deserialization requires the exact class")
        module_guard(expected_class)
        numerical_guard(context="qualified Q4 deserialization")
        owned_payload = dict(payload)
        module_guard(expected_class)
        numerical_guard(context="qualified Q4 deserialization")
        candidate = method(cls, owned_payload)
        serialization_guard(candidate, expected_class=expected_class)
        quadrature_guard(candidate)
        return candidate

    guarded.__name__ = method.__name__
    guarded.__qualname__ = method.__qualname__
    guarded.__doc__ = method.__doc__
    guarded.__annotations__ = dict(method.__annotations__)
    return guarded


def _freeze_qualified_component_cache(
    value: Any,
    frozen_arrays: Optional[Dict[int, np.ndarray]] = None,
) -> Any:
    """Recursively freeze every mechanics-bearing Q4 cache value."""

    arrays = {} if frozen_arrays is None else frozen_arrays
    if isinstance(value, np.ndarray):
        identity = id(value)
        frozen = arrays.get(identity)
        if frozen is None:
            contiguous = np.ascontiguousarray(value)
            frozen = np.frombuffer(
                contiguous.tobytes(order="C"), dtype=contiguous.dtype
            ).reshape(contiguous.shape)
            arrays[identity] = frozen
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_qualified_component_cache(item, arrays)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_qualified_component_cache(item, arrays) for item in value
        )
    return value


def _make_q4_nonlinear_cache_provenance() -> tuple[Any, Any, Any]:
    """Keep Q4 nonlinear cache identity outside caller-reachable instances."""

    records: Dict[int, tuple[Any, Any, Any]] = {}

    def bind(element: Any, cache: Any, cache_key: Any) -> None:
        identity = id(element)

        def discard(reference: Any, *, expected_identity: int = identity) -> None:
            current = records.get(expected_identity)
            if current is not None and current[0] is reference:
                records.pop(expected_identity, None)

        records[identity] = (
            weakref.ref(element, discard),
            cache,
            cache_key,
        )

    def clear(element: Any) -> None:
        records.pop(id(element), None)

    def require(element: Any, cache: Any, cache_key: Any) -> None:
        record = records.get(id(element))
        if (
            record is None
            or record[0]() is not element
            or record[1] is not cache
            or record[2] is not cache_key
            or type(cache) is not MappingProxyType
            or type(cache_key) is not tuple
        ):
            raise RuntimeError(
                "qualified Q4 nonlinear cache provenance changed"
            )

    return bind, clear, require


(
    _bind_q4_nonlinear_cache_provenance,
    _clear_q4_nonlinear_cache_provenance,
    _require_q4_nonlinear_cache_provenance,
) = _make_q4_nonlinear_cache_provenance()


class QualifiedQ4MigrationWarning(UserWarning):
    """Warn that a safe uncoupled pre-policy Q4 record was migrated."""


def _q4_state_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the constitutive payload covered by the Q4 committed seal."""

    return {
        str(key): value
        for key, value in state.items()
        if str(key)
        not in {
            _Q4_CURRENT_STATE_BINDING_KEY,
            _Q4_CURRENT_STATE_DIGEST_KEY,
            "state_digest",
        }
    }


def _qualified_q4_layer_count(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(
            "qualified Q4 num_layers must be one of [3, 5, 7, 9, 11]"
        )
    count = int(value)
    if count not in {3, 5, 7, 9, 11}:
        raise ValueError(
            "qualified Q4 num_layers must be one of [3, 5, 7, 9, 11]"
        )
    return count


def _binary64_vector_fingerprint(values: Any, shape: tuple[int, ...], label: str) -> str:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"qualified Q4 {label} must be a finite {shape} array")
    return canonical_sha256(
        {
            "layout": f"Q4_{label.upper()}_BINARY64_HEX_V1",
            "shape": list(shape),
            "values": [float(value).hex() for value in array.reshape(-1)],
        }
    )


def _shape(r: float, s: float) -> np.ndarray:
    return np.asarray(
        ((1.0 - r) * (1.0 - s), (1.0 + r) * (1.0 - s),
         (1.0 + r) * (1.0 + s), (1.0 - r) * (1.0 + s)),
        dtype=float,
    ) / 4.0


def _shape_derivatives(r: float, s: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((-(1.0 - s), 1.0 - s, 1.0 + s, -(1.0 + s)), dtype=float) / 4.0,
        np.asarray((-(1.0 - r), -(1.0 + r), 1.0 + r, 1.0 - r), dtype=float) / 4.0,
    )


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    component_scale = float(np.max(np.abs(values)))
    if not math.isfinite(component_scale) or component_scale <= 0.0:
        raise ValueError(f"cannot normalize E4-PL {label}")
    scaled = values / component_scale
    norm = float(np.linalg.norm(scaled))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"cannot normalize E4-PL {label}")
    return scaled / norm


def _characteristic_length(coordinates: np.ndarray) -> float:
    """Return a translation- and scale-covariant nodal diameter."""

    values = np.asarray(coordinates, dtype=float)
    length = max(
        (
            float(np.linalg.norm(values[first] - values[second]))
            for first in range(len(values))
            for second in range(first)
        ),
        default=0.0,
    )
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("E4-PL nodes must have positive finite diameter")
    return length


def _local_jacobian_scale(local: np.ndarray) -> float:
    """Return the squared local diameter for dimensionless Jacobian guards."""

    length = _characteristic_length(np.asarray(local, dtype=float))
    scale = length * length
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("E4-PL local Jacobian scale must be positive and finite")
    return scale


def equation7_frame(nodes: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the frozen numbered-frame basis, local nodes and warpage ratio."""

    coordinates = np.asarray(nodes, dtype=float)
    if coordinates.shape != (4, 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("E4-PL nodes must be a finite 4x3 array")
    diagonal_1 = coordinates[2] - coordinates[0]
    diagonal_2 = coordinates[1] - coordinates[3]
    length = _characteristic_length(coordinates)
    angular_floor = 64.0 * np.finfo(float).eps
    if (
        float(np.linalg.norm(diagonal_1)) <= angular_floor * length
        or float(np.linalg.norm(diagonal_2)) <= angular_floor * length
    ):
        raise ValueError("E4-PL diagonals are degenerate relative to the facet size")
    normalized_1 = _normalize(diagonal_1, "first diagonal")
    normalized_2 = _normalize(diagonal_2, "second diagonal")
    tangent_1_source = normalized_1 + normalized_2
    tangent_2_source = normalized_1 - normalized_2
    if (
        float(np.linalg.norm(tangent_1_source)) <= angular_floor
        or float(np.linalg.norm(tangent_2_source)) <= angular_floor
    ):
        raise ValueError("E4-PL diagonals cannot establish two stable tangents")
    tangent_1 = _normalize(tangent_1_source, "first tangent")
    tangent_2 = _normalize(tangent_2_source, "second tangent")
    normal = _normalize(np.cross(tangent_1, tangent_2), "normal")
    tangent_2 = _normalize(np.cross(normal, tangent_1), "orthogonal tangent")
    tangent_1 = _normalize(np.cross(tangent_2, normal), "renormalized tangent")
    frame = np.column_stack((tangent_1, tangent_2, normal))
    centre = np.mean(coordinates, axis=0)
    relative = coordinates - centre
    local = relative @ frame[:, :2]
    warpage = float(np.max(np.abs(relative @ normal)) / length)
    return frame, local, warpage


def _coefficients(local: np.ndarray) -> Dict[str, float]:
    modal = np.asarray(
        ((1, 1, 1, 1), (-1, 1, 1, -1), (-1, -1, 1, 1), (1, -1, 1, -1)),
        dtype=float,
    ) / 4.0
    x0, xr, xs, xrs = modal @ local[:, 0]
    y0, yr, ys, yrs = modal @ local[:, 1]
    return {
        "x0": float(x0), "xr": float(xr), "xs": float(xs), "xrs": float(xrs),
        "y0": float(y0), "yr": float(yr), "ys": float(ys), "yrs": float(yrs),
        "jc": float(xr * ys - xs * yr),
        "jr": float(xr * yrs - xrs * yr),
        "js": float(xrs * ys - xs * yrs),
    }


def _jacobian(c: Mapping[str, float], r: float, s: float) -> tuple[float, float, float, float, float]:
    xr = c["xr"] + c["xrs"] * s
    xs = c["xs"] + c["xrs"] * r
    yr = c["yr"] + c["yrs"] * s
    ys = c["ys"] + c["yrs"] * r
    return xr, xs, yr, ys, xr * ys - xs * yr


def _natural_shear(local: np.ndarray, r: float, s: float, direction: int) -> np.ndarray:
    shape = _shape(r, s)
    nr, ns = _shape_derivatives(r, s)
    derivative = nr if direction == 0 else ns
    x_direction = float(local[:, 0] @ derivative)
    y_direction = float(local[:, 1] @ derivative)
    row = np.zeros(20, dtype=float)
    for index in range(4):
        base = 5 * index
        row[base + 2] = derivative[index]
        row[base + 3] = -y_direction * shape[index]
        row[base + 4] = x_direction * shape[index]
    return row


def _compatible(local: np.ndarray, c: Mapping[str, float], r: float, s: float) -> np.ndarray:
    nr, ns = _shape_derivatives(r, s)
    xr, xs, yr, ys, determinant = _jacobian(c, r, s)
    nx = (ys * nr - yr * ns) / determinant
    ny = (-xs * nr + xr * ns) / determinant
    result = np.zeros((8, 20), dtype=float)
    for index in range(4):
        base = 5 * index
        result[0, base] = nx[index]
        result[1, base + 1] = ny[index]
        result[2, base] = ny[index]
        result[2, base + 1] = nx[index]
        result[3, base + 4] = nx[index]
        result[4, base + 3] = -ny[index]
        result[5, base + 4] = ny[index]
        result[5, base + 3] = -nx[index]
    row_r_minus = _natural_shear(local, 0.0, -1.0, 0)
    row_r_plus = _natural_shear(local, 0.0, 1.0, 0)
    row_s_plus = _natural_shear(local, 1.0, 0.0, 1)
    row_s_minus = _natural_shear(local, -1.0, 0.0, 1)
    row_r = 0.5 * (1.0 - s) * row_r_minus + 0.5 * (1.0 + s) * row_r_plus
    row_s = 0.5 * (1.0 + r) * row_s_plus + 0.5 * (1.0 - r) * row_s_minus
    result[6] = (ys * row_r - yr * row_s) / determinant
    result[7] = (-xs * row_r + xr * row_s) / determinant
    return result


def _tensor_transform(xr: float, xs: float, yr: float, ys: float, a: float, b: float) -> np.ndarray:
    return np.asarray(
        (
            (xr * xr, xs * xs, a * xr * xs),
            (yr * yr, ys * ys, a * yr * ys),
            (b * xr * yr, b * xs * ys, xr * ys + yr * xs),
        ),
        dtype=float,
    )


def _source_fields(c: Mapping[str, float], r: float, s: float) -> tuple[np.ndarray, np.ndarray]:
    jc, jr, js = c["jc"], c["jr"], c["js"]
    r_bar, s_bar = jr / (3.0 * jc), js / (3.0 * jc)
    stress_transform = _tensor_transform(c["xr"], c["xs"], c["yr"], c["ys"], 2.0, 1.0)
    strain_transform = _tensor_transform(c["xr"], c["xs"], c["yr"], c["ys"], 1.0, 2.0)
    shear_transform = np.asarray(((c["xr"], c["xs"]), (c["yr"], c["ys"])), dtype=float)
    n_sigma = np.zeros((8, 14), dtype=float)
    n_epsilon = np.zeros((8, 21), dtype=float)
    n_sigma[:, :8] = np.eye(8)
    n_epsilon[:, :8] = np.eye(8)
    seed = np.asarray(((s - s_bar, 0.0), (0.0, r - r_bar), (0.0, 0.0)), dtype=float)
    stress_vary = stress_transform @ seed
    strain_vary = strain_transform @ seed
    for row, column in ((0, 8), (3, 10)):
        n_sigma[row : row + 3, column : column + 2] = stress_vary
        n_epsilon[row : row + 3, column : column + 2] = strain_vary
    shear_seed = np.asarray(((s - s_bar, 0.0), (0.0, r - r_bar)), dtype=float)
    n_sigma[6:8, 12:14] = shear_transform @ shear_seed
    n_epsilon[6:8, 12:14] = shear_transform @ shear_seed
    enrichment = np.asarray(
        ((r, 0, 0, 0, r * s, 0, 0), (0, s, 0, 0, 0, r * s, 0), (0, 0, r, s, 0, 0, r * s)),
        dtype=float,
    )
    determinant = _jacobian(c, r, s)[4]
    n_epsilon[:3, 14:21] = (jc / determinant) * (strain_transform @ enrichment)
    return n_sigma, n_epsilon


def _centre_taylor(c: Mapping[str, float]) -> np.ndarray:
    f0 = np.ones(4, dtype=float) / 4.0
    fr = np.asarray((-1, 1, 1, -1), dtype=float) / 4.0
    fs = np.asarray((-1, -1, 1, 1), dtype=float) / 4.0
    frs = np.asarray((1, -1, 1, -1), dtype=float) / 4.0
    result = np.zeros((3, 24), dtype=float)
    jc, jr, js = c["jc"], c["jr"], c["js"]
    for coordinate in range(24):
        node, component = divmod(coordinate, 6)
        ur = fr[node] if component == 0 else 0.0
        us = fs[node] if component == 0 else 0.0
        urs = frs[node] if component == 0 else 0.0
        vr = fr[node] if component == 1 else 0.0
        vs = fs[node] if component == 1 else 0.0
        vrs = frs[node] if component == 1 else 0.0
        d0 = f0[node] if component == 5 else 0.0
        dr = fr[node] if component == 5 else 0.0
        ds = fs[node] if component == 5 else 0.0
        n0 = -c["xs"] * ur + c["xr"] * us - c["ys"] * vr + c["yr"] * vs
        nr = -c["xrs"] * ur + c["xr"] * urs - c["yrs"] * vr + c["yr"] * vrs
        ns = -c["xs"] * urs + c["xrs"] * us - c["ys"] * vrs + c["yrs"] * vs
        result[0, coordinate] = d0 + n0 / (2.0 * jc)
        result[1, coordinate] = dr + (nr * jc - n0 * jr) / (2.0 * jc * jc)
        result[2, coordinate] = ds + (ns * jc - n0 * js) / (2.0 * jc * jc)
    return result


def _residual_mode(local: np.ndarray, c: Mapping[str, float]) -> np.ndarray:
    x = local[:, 0]
    y = local[:, 1]
    centred_x = x - c["x0"]
    centred_y = y - c["y0"]
    xi = np.asarray((-1, 1, 1, -1), dtype=float)
    eta = np.asarray((-1, -1, 1, 1), dtype=float)
    alternating = np.asarray((1, -1, 1, -1), dtype=float)
    area = 4.0 * c["jc"]
    b1 = ((eta @ centred_y) * xi - (xi @ centred_y) * eta) / (4.0 * area)
    b2 = (-(eta @ centred_x) * xi + (xi @ centred_x) * eta) / (4.0 * area)
    return (alternating - (alternating @ centred_x) * b1 - (alternating @ centred_y) * b2) / 4.0


def _global_transform(frame: np.ndarray) -> np.ndarray:
    transform = np.zeros((24, 24), dtype=float)
    for node in range(4):
        transform[6 * node : 6 * node + 3, 6 * node : 6 * node + 3] = frame
        transform[6 * node + 3 : 6 * node + 6, 6 * node + 3 : 6 * node + 6] = frame
    return transform


def _stationary_blocks(
    local: np.ndarray,
    c: Mapping[str, float],
    constitutive: np.ndarray,
    *, compensated: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the frozen 35-field stationary system and its physical coupling.

    This is the single floating-point implementation of the Q1A/Q1Y mixed
    blocks.  Both stiffness condensation and physical recovery call this
    helper so recovery cannot silently drift back to the inherited compatible
    MITC4 fields.
    """

    f_matrix = np.zeros((21, 14), dtype=float)
    coupling_20 = np.zeros((14, 20), dtype=float)
    strain_gram = np.zeros((21, 21), dtype=float)
    gram = np.zeros((3, 3), dtype=float)
    f_terms, coupling_terms, strain_terms, gram_terms = [], [], [], []
    weights = []
    for r, s in _GAUSS:
        determinant = _jacobian(c, r, s)[4]
        n_sigma, n_epsilon = _source_fields(c, r, s)
        compatible = _compatible(local, c, r, s)
        polynomial = np.asarray((1.0, r, s), dtype=float)
        if compensated:
            product = _local_contractions.product
            f_terms.append(-product(n_epsilon.T, n_sigma))
            coupling_terms.append(product(n_sigma.T, compatible))
            strain_terms.append(product(product(n_epsilon.T, constitutive), n_epsilon))
            gram_terms.append(np.outer(polynomial, polynomial))
            weights.append(determinant)
        else:
            f_matrix -= determinant * (n_epsilon.T @ n_sigma)
            coupling_20 += determinant * (n_sigma.T @ compatible)
            strain_gram += determinant * (n_epsilon.T @ constitutive @ n_epsilon)
            gram += determinant * np.outer(polynomial, polynomial)
    if compensated:
        f_matrix = _local_contractions.weighted_sum(f_terms, weights)
        coupling_20 = _local_contractions.weighted_sum(coupling_terms, weights)
        strain_gram = _local_contractions.weighted_sum(strain_terms, weights)
        gram = _local_contractions.weighted_sum(gram_terms, weights)
    stationary = np.zeros((35, 35), dtype=float)
    stationary[:14, 14:] = f_matrix.T
    stationary[14:, :14] = f_matrix
    stationary[14:, 14:] = strain_gram
    coupling = np.zeros((24, 35), dtype=float)
    physical_coupling = np.zeros((20, 35), dtype=float)
    physical_coupling[:, :14] = coupling_20.T
    for node in range(4):
        coupling[6 * node : 6 * node + 5] = physical_coupling[
            5 * node : 5 * node + 5
        ]
    return stationary, coupling, gram


def _invariant_generalized_drilling_scale(membrane_matrix: np.ndarray) -> float:
    """Return the physical, basis-invariant generalized-section drill scale.

    This is the same generalized eigenvalue invariant used by the qualified
    S3 companion.  It equals ``A66`` for isotropy and, unlike a numbered
    ``A[2, 2]`` lookup, is unchanged by proper or reflected in-plane frame
    re-expression.
    """

    membrane = np.asarray(membrane_matrix, dtype=float)
    if membrane.shape != (3, 3) or not np.all(np.isfinite(membrane)):
        raise ValueError("qualified Q4 membrane matrix A must be finite 3x3")
    membrane = 0.5 * (membrane + membrane.T)
    if float(np.linalg.eigvalsh(membrane)[0]) <= 0.0:
        raise ValueError("qualified Q4 membrane matrix A must be positive definite")
    projector = np.asarray(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)), dtype=float)
    inverse_metric_sqrt = np.diag((1.0 / math.sqrt(2.0), math.sqrt(2.0)))
    restricted = projector.T @ membrane @ projector
    canonical = inverse_metric_sqrt @ restricted @ inverse_metric_sqrt
    value = 0.5 * float(np.linalg.eigvalsh(0.5 * (canonical + canonical.T))[0])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("qualified Q4 invariant drilling scale must be positive")
    return value


def _symmetric_ruiz_congruence(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Balance a symmetric mixed system through a fixed Ruiz congruence.

    Q4 stationary coordinates combine stress and generalized-strain fields
    with different physical units.  At ``t/L=1e-6`` the raw 35-coordinate
    matrix can be ill-conditioned solely because of those units.  The fixed
    congruence preserves the exact equation while making the binary64 solve
    insensitive to length and stiffness units::

        H_eq = D H D,  rhs_eq = D rhs,  x = D x_eq.

    Exactly eight max-row Ruiz steps are used.  The fixed count is deterministic,
    while the final row-norm certificate prevents a partially equilibrated
    system from reaching the solve.  The six-step bound was established over
    all registered Q4 geometries at coordinate scales ``1e-6``, ``1`` and
    ``1e6``.  Eight steps also retain the row-norm certificate for ordinary
    thick facets at extreme coordinate scales; further iterations do not
    materially improve the condition bound but do regress the cold element
    path.
    """

    made = np.asarray(matrix, dtype=np.float64)
    if made.ndim != 2 or made.shape[0] != made.shape[1]:
        raise ValueError("E4-PL stationary system must be square")
    if not np.all(np.isfinite(made)):
        raise ValueError("E4-PL stationary system must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        input_norm = float(np.linalg.norm(made, ord=np.inf))
        asymmetry = float(np.linalg.norm(made - made.T, ord=np.inf))
    if not math.isfinite(input_norm) or not math.isfinite(asymmetry):
        raise ValueError("E4-PL stationary system norm is non-finite")
    if asymmetry > 64.0 * np.finfo(np.float64).eps * input_norm:
        raise ValueError("E4-PL stationary system is not symmetric")
    equilibrated = 0.5 * (made + made.T)
    accumulated = np.ones(made.shape[0], dtype=np.float64)
    for _iteration in range(_STATIONARY_RUIZ_ITERATIONS):
        row_norms = np.max(np.abs(equilibrated), axis=1)
        if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= 0.0):
            raise ValueError("E4-PL stationary system is singular")
        step = 1.0 / np.sqrt(row_norms)
        accumulated *= step
        equilibrated = step[:, None] * equilibrated * step[None, :]
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    final_row_norms = np.max(np.abs(equilibrated), axis=1)
    if (
        np.any(~np.isfinite(accumulated))
        or np.any(accumulated <= 0.0)
        or np.any(~np.isfinite(equilibrated))
        or np.any(~np.isfinite(final_row_norms))
        or np.any(final_row_norms <= 0.0)
    ):
        raise ValueError("E4-PL stationary equilibration is unresolved")
    row_norm_ratio = float(np.max(final_row_norms) / np.min(final_row_norms))
    if (
        not math.isfinite(row_norm_ratio)
        or row_norm_ratio > _STATIONARY_RUIZ_ROW_NORM_RATIO_LIMIT
    ):
        raise ValueError(
            "E4-PL stationary equilibration exceeded the row-norm ratio limit"
        )
    return equilibrated, accumulated, {
        "id": STATIONARY_SOLVE_POLICY_ID,
        "iterations": _STATIONARY_RUIZ_ITERATIONS,
        "input_asymmetry_relative": asymmetry / max(input_norm, np.finfo(float).tiny),
        "row_norm_ratio": row_norm_ratio,
        "row_norm_ratio_limit": _STATIONARY_RUIZ_ROW_NORM_RATIO_LIMIT,
        "scale_max": float(np.max(accumulated)),
        "scale_min": float(np.min(accumulated)),
    }


def _stationary_backward_error(
    matrix: np.ndarray,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Return a fail-closed normwise backward error in original coordinates."""

    made_matrix = np.asarray(matrix, dtype=np.float64)
    made_solution = np.asarray(solution, dtype=np.float64)
    made_rhs = np.asarray(rhs, dtype=np.float64)
    if (
        np.any(~np.isfinite(made_matrix))
        or np.any(~np.isfinite(made_solution))
        or np.any(~np.isfinite(made_rhs))
    ):
        return math.inf
    with np.errstate(over="ignore", invalid="ignore"):
        residual = made_matrix @ made_solution - made_rhs
        matrix_norm = float(np.linalg.norm(made_matrix, ord=np.inf))
        solution_norm = float(np.linalg.norm(made_solution, ord=np.inf))
        rhs_norm = float(np.linalg.norm(made_rhs, ord=np.inf))
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        denominator = matrix_norm * solution_norm + rhs_norm
    if not all(
        math.isfinite(value)
        for value in (
            matrix_norm,
            solution_norm,
            rhs_norm,
            residual_norm,
            denominator,
        )
    ):
        return math.inf
    if denominator <= 0.0:
        return 0.0 if residual_norm == 0.0 else math.inf
    return residual_norm / denominator


def _solve_stationary_system(
    stationary: np.ndarray,
    coupling: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Solve the Q4 mixed condensation with deterministic certification."""

    made_stationary = np.asarray(stationary, dtype=np.float64)
    made_coupling = np.asarray(coupling, dtype=np.float64)
    if (
        made_stationary.ndim != 2
        or made_stationary.shape[0] != made_stationary.shape[1]
        or made_coupling.ndim != 2
        or made_coupling.shape[1] != made_stationary.shape[0]
    ):
        raise ValueError("E4-PL stationary coupling dimensions are incompatible")
    if not np.all(np.isfinite(made_coupling)):
        raise ValueError("E4-PL stationary coupling must be finite")
    equilibrated, scaling, diagnostics = _symmetric_ruiz_congruence(
        made_stationary
    )
    rhs = made_coupling.T
    scaled_rhs = scaling[:, None] * rhs
    try:
        equilibrated_solution = np.linalg.solve(equilibrated, scaled_rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("E4-PL stationary system is singular") from exc
    solution = scaling[:, None] * equilibrated_solution
    backward_error = _stationary_backward_error(
        made_stationary,
        solution,
        rhs,
    )
    if (
        not np.all(np.isfinite(solution))
        or not math.isfinite(backward_error)
        or backward_error > _STATIONARY_BACKWARD_ERROR_LIMIT
    ):
        raise ValueError(
            "E4-PL stationary solve has uncertified original-system accuracy"
        )
    return solution, {
        **diagnostics,
        "relative_backward_error": backward_error,
        "relative_backward_error_limit": _STATIONARY_BACKWARD_ERROR_LIMIT,
        "disposition": "CERTIFIED",
    }



def _initial_offsets_or_none(element, state, cache, layers, material):
    offsets = _q4_assumed_strain.layer_offsets(
        state, len(cache["gp"]), int(layers), float(element.thickness),
        _plasticity_module.plane_stress_elastic_matrix(
            float(material.elastic_modulus), float(material.poisson_ratio)),
    )
    return None if offsets is None else offsets[None]

class QualifiedE4PLShellElement(
    ShellElement,
    metaclass=AuthorityEpochMeta,
):
    """Dormant qualified E4-PL element for four-node shell facets."""

    formulation_id = FORMULATION_ID
    GAUSS_POINTS_2x2 = _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
        "GAUSS_POINTS_2x2"
    ]
    GAUSS_WEIGHTS_2x2 = _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
        "GAUSS_WEIGHTS_2x2"
    ]
    GAUSS_POINTS_1x1 = _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
        "GAUSS_POINTS_1x1"
    ]
    GAUSS_WEIGHTS_1x1 = _Q4_QUADRATURE_CLASS_ARRAY_AUTHORITY[
        "GAUSS_WEIGHTS_1x1"
    ]
    _MITC4_SAMPLE_POINTS = MappingProxyType(
        dict(ShellElement._MITC4_SAMPLE_POINTS)
    )

    def get_node_coordinates(
        self,
        mesh: Any,
        *,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> np.ndarray:
        """Read mesh coordinates and recheck exact authority before mechanics."""

        # Do not delegate the provider and node callbacks as one opaque base
        # call.  A mesh callback can replace ``node.coords`` (or any later
        # authority) before the base kernel invokes it.  Observe one provider
        # result/property at a time and recheck the captured runtime authority
        # before consuming that observation.
        post_observation = _qualified_runtime_post_observation
        coordinates = np.empty((4, 3), dtype=np.float64)
        for index, node_id in enumerate(self.node_ids):
            node = mesh.get_node(node_id)
            if post_observation is not None:
                post_observation()
            if node is None:
                raise ValueError(f"Node {node_id} not found")
            if type(node) is not Node:
                raise ValueError(
                    f"Node {node_id} is not an exact ANYfeSolver Node"
                )
            namespace = object.__getattribute__(node, "__dict__")
            if type(namespace) is not dict or not all(
                type(name) is str for name in namespace
            ):
                raise ValueError(f"Node {node_id} coordinate state is incompatible")
            for component, name in enumerate(("x", "y", "z")):
                if name not in namespace:
                    raise ValueError(f"Node {node_id} lacks coordinate {name}")
                value = dict.__getitem__(namespace, name)
                if post_observation is not None:
                    post_observation()
                if type(value) not in _QUALIFIED_Q4_COORDINATE_SCALAR_TYPES:
                    raise ValueError(
                        f"Node {node_id} coordinate {name} is not an exact real scalar"
                    )
                made_value = float(value)
                if post_observation is not None:
                    post_observation()
                if not math.isfinite(made_value):
                    raise ValueError(
                        f"Node {node_id} coordinate {name} must be finite"
                    )
                coordinates[index, component] = made_value
        return coordinates
    implementation_id = IMPLEMENTATION_ID
    recovery_policy_id = RECOVERY_POLICY_ID
    stationary_solve_policy_id = STATIONARY_SOLVE_POLICY_ID
    director_polarity_policy_id = DIRECTOR_POLARITY_POLICY_ID
    director_reversal_transform_id = DIRECTOR_REVERSAL_TRANSFORM_ID
    current_state_binding_schema_id = Q4_CURRENT_STATE_BINDING_SCHEMA_ID
    current_state_algorithmic_origin_schema_id = (
        Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
    )
    current_state_tangent_decomposition_policy_id = (
        Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
    )
    current_state_projection_policy_id = Q4_CURRENT_STATE_PROJECTION_POLICY_ID
    activity_disposition_schema_id = Q4_ACTIVITY_DISPOSITION_SCHEMA_ID
    deleted_frozen_policy_id = Q4_DELETED_FROZEN_POLICY_ID
    failed_state_policy_id = Q4_FAILED_STATE_POLICY_ID
    quadrature_authority_id = Q4_QUADRATURE_AUTHORITY_ID
    legacy_stiffness_batch_eligible = False
    legacy_nonlinear_batch_eligible = True
    _plan_invalidating_attributes = frozenset(
        {
            "director_polarity",
            "drilling_stabilization",
            "element_id",
            "formulation_id",
            "hourglass_stabilization",
            "legacy_nonlinear_batch_eligible",
            "legacy_stiffness_batch_eligible",
            "material_angle_deg",
            "material_direction",
            "material_name",
            "node_ids",
            "planar_tolerance",
            "pl_stabilization",
            "reduced_integration",
            "reference_normal",
            "reference_surface_offset",
            "shell_section",
            "thickness",
            "warped_formulation",
            "GAUSS_POINTS_2x2",
            "GAUSS_WEIGHTS_2x2",
            "GAUSS_POINTS_1x1",
            "GAUSS_WEIGHTS_1x1",
            "gauss_points",
            "gauss_weights",
            "shear_gauss_points",
            "shear_gauss_weights",
        }
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            self.__dict__.get("_qualified_plan_state_revision") is not None
            and name in _QUALIFIED_Q4_PROTECTED_CACHE_NAMES
        ):
            raise AttributeError(
                f"qualified Q4 derived cache {name} is internally managed"
            )
        if name == "node_ids":
            made_node_ids = tuple(value)
            if not all(type(node_id) is int for node_id in made_node_ids):
                raise TypeError(
                    "qualified Q4 node_ids must contain exact non-boolean integers"
                )
            value = made_node_ids
        elif name == "reference_normal" and value is None:
            if (
                self.__dict__.get("_qualified_plan_state_revision") is not None
                and self.__dict__.get("director_polarity", 1) != 1
            ):
                raise ValueError(
                    "qualified Q4 reversed director requires reference_normal"
                )
        elif name in {"material_direction", "reference_normal"} and value is not None:
            numerical_guard = require_exact_numpy_runtime_authority
            numerical_guard(context=f"qualified Q4 {name} assignment")
            if type(value) in {list, tuple} and any(
                isinstance(component, (bool, np.bool_))
                or not isinstance(
                    component, (int, float, np.integer, np.floating)
                )
                for component in value
            ):
                raise ValueError(
                    f"qualified Q4 {name} must be a finite real 3-vector"
                )
            observed = np.asarray(value)
            numerical_guard(context=f"qualified Q4 {name} assignment")
            if (
                observed.shape != (3,)
                or observed.dtype.kind not in "fiu"
                or observed.dtype.kind == "b"
            ):
                raise ValueError(
                    f"qualified Q4 {name} must be a finite real 3-vector"
                )
            vector = np.ascontiguousarray(observed, dtype=np.float64)
            if not np.all(np.isfinite(vector)):
                raise ValueError(
                    f"qualified Q4 {name} must be a finite real 3-vector"
                )
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                raise ValueError(f"qualified Q4 {name} must be non-zero")
            if name == "reference_normal":
                vector = np.ascontiguousarray(vector / norm, dtype=np.float64)
            value = np.frombuffer(
                vector.tobytes(order="C"), dtype=np.float64
            ).reshape(vector.shape)
        elif name == "shell_section" and value is not None:
            if self.__dict__.get("_qualified_plan_state_revision") is None:
                numerical_guard = require_exact_numpy_runtime_authority

                def post_observation() -> None:
                    numerical_guard(
                        context="qualified Q4 shell-section construction",
                    )

            else:
                runtime_guard = _require_exact_q4_runtime_authority

                def post_observation() -> None:
                    runtime_guard(
                        self,
                        context="qualified Q4 shell-section assignment",
                    )

            post_observation()
            value = _guarded_owned_generalized_shell_section(
                value,
                post_observation=post_observation,
            )
        elif name in {
            "drilling_stabilization",
            "hourglass_stabilization",
            "material_angle_deg",
            "planar_tolerance",
            "pl_stabilization",
            "thickness",
        }:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"qualified Q4 {name} must be a finite real scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"qualified Q4 {name} must be a finite real scalar")
            if name == "thickness" and value <= 0.0:
                raise ValueError("qualified Q4 thickness must be strictly positive")
            if name in {
                "drilling_stabilization",
                "hourglass_stabilization",
                "planar_tolerance",
                "pl_stabilization",
            } and value < 0.0:
                raise ValueError(f"qualified Q4 {name} must be nonnegative")
        elif name == "reduced_integration" and type(value) is not bool:
            raise TypeError("qualified Q4 reduced_integration must be an exact bool")
        elif name == "director_polarity":
            if type(value) is not int or value not in {-1, 1}:
                raise ValueError(
                    "qualified Q4 director_polarity must be the integer -1 or +1"
                )
            if (
                value != 1
                and self.__dict__.get("_qualified_plan_state_revision") is not None
                and self.__dict__.get("reference_normal") is None
            ):
                raise ValueError(
                    "qualified Q4 director_polarity requires reference_normal"
                )
        revision = self.__dict__.get("_qualified_plan_state_revision")
        if revision is not None and name in self._plan_invalidating_attributes:
            object.__setattr__(
                self,
                "_qualified_plan_state_revision",
                int(revision) + 1,
            )
            tokens = self.__dict__.get("_qualified_direct_state_tokens")
            if tokens is None:
                token = self.__dict__.get("_qualified_direct_state_token")
                tokens = () if token is None else (token,)
            for token in tokens:
                token[0] = int(token[0]) + 1
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if (
            self.__dict__.get("_qualified_plan_state_revision") is not None
            and name in _QUALIFIED_Q4_PROTECTED_CACHE_NAMES
        ):
            raise AttributeError(
                f"qualified Q4 derived cache {name} is internally managed"
            )
        super().__delattr__(name)

    def __init__(
        self,
        element_id: int,
        node_ids: list[int],
        material_name: str = "default",
        thickness: float = 0.01,
        drilling_stabilization: float = 1.0e-3,
        reduced_integration: bool = False,
        hourglass_stabilization: float = 1.0e-3,
        material_direction: Optional[np.ndarray] = None,
        material_angle_deg: float = 0.0,
        shell_section: Optional[Any] = None,
        *,
        pl_stabilization: float = 1.0,
        planar_tolerance: float = 1.0e-10,
        warped_formulation: str = "varying_frame",
        legacy_warped_fallback: Optional[bool] = None,
        reference_normal: Optional[Sequence[float]] = None,
        director_polarity: int = 1,
    ) -> None:
        numerical_guard = require_exact_numpy_runtime_authority
        numerical_guard(context="qualified Q4 construction")
        owned_node_ids = tuple(node_ids)
        numerical_guard(context="qualified Q4 connectivity")
        if len(owned_node_ids) != 4:
            raise ValueError("QualifiedE4PLShellElement requires exactly four nodes")
        if not all(type(node_id) is int for node_id in owned_node_ids):
            raise TypeError(
                "qualified Q4 node_ids must contain exact non-boolean integers"
            )
        node_ids = owned_node_ids
        scalar_inputs = {
            "thickness": thickness,
            "material_angle_deg": material_angle_deg,
            "drilling_stabilization": drilling_stabilization,
            "hourglass_stabilization": hourglass_stabilization,
            "pl_stabilization": pl_stabilization,
            "planar_tolerance": planar_tolerance,
        }
        for label, raw_value in scalar_inputs.items():
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
                raw_value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    f"qualified Q4 {label} must be a finite real scalar"
                )
            if not math.isfinite(float(raw_value)):
                raise ValueError(
                    f"qualified Q4 {label} must be a finite real scalar"
                )
        if float(thickness) <= 0.0:
            raise ValueError("qualified Q4 thickness must be strictly positive")
        for label in (
            "drilling_stabilization",
            "hourglass_stabilization",
            "pl_stabilization",
            "planar_tolerance",
        ):
            if float(scalar_inputs[label]) < 0.0:
                raise ValueError(f"qualified Q4 {label} must be nonnegative")
        if type(reduced_integration) is not bool:
            raise TypeError("qualified Q4 reduced_integration must be an exact bool")
        if legacy_warped_fallback is not None and type(
            legacy_warped_fallback
        ) is not bool:
            raise TypeError(
                "qualified Q4 legacy_warped_fallback must be an exact bool"
            )
        if type(warped_formulation) is not str:
            raise TypeError("qualified Q4 warped_formulation must be a string")
        if material_direction is not None:
            if type(material_direction) in {list, tuple} and any(
                isinstance(component, (bool, np.bool_))
                or not isinstance(
                    component, (int, float, np.integer, np.floating)
                )
                for component in material_direction
            ):
                raise ValueError(
                    "qualified Q4 material_direction must be a finite real 3-vector"
                )
            raw_direction = np.asarray(material_direction)
            numerical_guard(context="qualified Q4 material direction")
            if (
                raw_direction.shape != (3,)
                or raw_direction.dtype.kind not in "fiu"
                or raw_direction.dtype.kind == "b"
            ):
                raise ValueError(
                    "qualified Q4 material_direction must be a finite real 3-vector"
                )
            material_direction = np.asarray(raw_direction, dtype=np.float64)
            if not np.all(np.isfinite(material_direction)):
                raise ValueError(
                    "qualified Q4 material_direction must be a finite real 3-vector"
                )
        super().__init__(
            element_id,
            node_ids,
            material_name,
            thickness,
            drilling_stabilization,
            reduced_integration,
            hourglass_stabilization,
            material_direction,
            material_angle_deg,
            shell_section,
        )
        if isinstance(director_polarity, (bool, np.bool_)) or not isinstance(
            director_polarity, (int, np.integer)
        ) or int(director_polarity) not in (-1, 1):
            raise ValueError(
                "qualified Q4 director_polarity must be the integer -1 or +1"
            )
        self.director_polarity = int(director_polarity)
        if reference_normal is None:
            self.reference_normal = None
            if self.director_polarity != 1:
                raise ValueError(
                    "qualified Q4 director_polarity requires an authoritative "
                    "reference_normal"
                )
        else:
            if type(reference_normal) in {list, tuple} and any(
                isinstance(component, (bool, np.bool_))
                or not isinstance(
                    component, (int, float, np.integer, np.floating)
                )
                for component in reference_normal
            ):
                raise ValueError(
                    "qualified Q4 reference_normal must be a finite 3-vector"
                )
            normal = np.asarray(reference_normal)
            numerical_guard(context="qualified Q4 reference normal")
            if normal.dtype.kind not in "fiu" or normal.dtype.kind == "b":
                raise ValueError(
                    "qualified Q4 reference_normal must be a finite 3-vector"
                )
            normal = np.asarray(normal, dtype=float).reshape(-1)
            if normal.size != 3 or not np.all(np.isfinite(normal)):
                raise ValueError(
                    "qualified Q4 reference_normal must be a finite 3-vector"
                )
            self.reference_normal = _normalize(normal, "reference normal").copy()
        if (
            self.shell_section is not None
            and np.any(np.asarray(self.shell_section.B, dtype=float) != 0.0)
            and self.reference_normal is None
        ):
            raise ValueError(
                "B-coupled qualified Q4 sections require an authoritative "
                "reference_normal; connectivity winding is not a physical director"
            )
        self.pl_stabilization = float(pl_stabilization)
        self.planar_tolerance = float(planar_tolerance)
        if legacy_warped_fallback is not None:
            warped_formulation = (
                "varying_frame" if bool(legacy_warped_fallback) else "reject"
            )
        self.warped_formulation = str(warped_formulation).strip().lower()
        if not math.isfinite(self.pl_stabilization) or self.pl_stabilization < 0.0:
            raise ValueError("pl_stabilization must be finite and nonnegative")
        if not math.isfinite(self.planar_tolerance) or self.planar_tolerance < 0.0:
            raise ValueError("planar_tolerance must be finite and nonnegative")
        if self.warped_formulation not in _WARPED_FORMULATIONS:
            raise ValueError(
                "warped_formulation must be one of "
                f"{sorted(_WARPED_FORMULATIONS)}"
            )
        self._qualified_components: Optional[Mapping[str, Any]] = None
        self._qualified_cache_key: Optional[tuple[Any, ...]] = None
        self._qualified_component_guard: Optional[tuple[Any, ...]] = None
        self._qualified_plan_state_revision = 0
        # Cold preparation replaces owned derived values, not the input
        # namespace captured by an ORIGINAL response snapshot. Deserialization
        # already establishes these same empty fields in __setstate__.
        object.__setattr__(self, "_nl_cache", None)
        object.__setattr__(self, "_nl_cache_key", None)
        object.__setattr__(self, "_hourglass_stiffness_matrix", None)

    def validate_quadrature_authority(
        self,
    ) -> str:
        """Fail closed unless every Q4 quadrature input retains exact authority."""

        return _validate_q4_quadrature_authority(self)

    def __deepcopy__(
        self,
        memo: Dict[int, Any],
    ) -> "QualifiedE4PLShellElement":
        """Copy owned inputs while dropping every mesh/material-derived cache."""

        made = type(self).__new__(type(self))
        memo[id(self)] = made
        derived = {
            "_hourglass_stiffness_matrix",
            "_internal_forces",
            "_mass_matrix",
            "_nl_cache",
            "_nl_cache_key",
            "_qualified_component_guard",
            "_qualified_cache_key",
            "_qualified_components",
            "_stiffness_matrix",
        }
        for name, value in self.__dict__.items():
            if name in {
                "_qualified_direct_state_token",
                "_qualified_direct_state_tokens",
            }:
                continue
            if name in derived:
                object.__setattr__(made, name, None)
                continue
            if name in {"material_direction", "reference_normal"} and value is not None:
                vector = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
                value = np.frombuffer(
                    vector.tobytes(order="C"), dtype=np.float64
                ).reshape(vector.shape)
            else:
                value = copy.deepcopy(value, memo)
            object.__setattr__(made, name, value)
        return made

    def __getstate__(self) -> Dict[str, Any]:
        """Serialize owned Q4 inputs without exporting derived mechanics caches."""

        state = dict(self.__dict__)
        state.pop("_qualified_direct_state_token", None)
        state.pop("_qualified_direct_state_tokens", None)
        for name in {
            "_hourglass_stiffness_matrix",
            "_internal_forces",
            "_mass_matrix",
            "_nl_cache",
            "_nl_cache_key",
            "_qualified_component_guard",
            "_qualified_cache_key",
            "_qualified_components",
            "_stiffness_matrix",
        }:
            state[name] = None
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        restored = dict(state)
        node_ids = restored.get("node_ids")
        if type(node_ids) in {list, tuple} and all(
            type(value) is int for value in node_ids
        ):
            restored["node_ids"] = tuple(node_ids)
        restored.setdefault("_qualified_plan_state_revision", 0)
        if type(restored["_qualified_plan_state_revision"]) is not int:
            restored["_qualified_plan_state_revision"] = 0
        for name in {
            "_hourglass_stiffness_matrix",
            "_internal_forces",
            "_mass_matrix",
            "_nl_cache",
            "_nl_cache_key",
            "_qualified_component_guard",
            "_qualified_cache_key",
            "_qualified_components",
            "_stiffness_matrix",
        }:
            restored[name] = None
        self.__dict__.update(restored)
        for name in ("material_direction", "reference_normal"):
            value = self.__dict__.get(name)
            if value is None:
                continue
            vector = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
            object.__setattr__(
                self,
                name,
                np.frombuffer(
                    vector.tobytes(order="C"), dtype=np.float64
                ).reshape(vector.shape),
            )

    @contextmanager
    def _current_state_cache_transaction(self):
        """Restore every mutable mechanics cache after evidence evaluation."""

        names = (
            "_nl_cache",
            "_nl_cache_key",
            "_qualified_component_guard",
            "_qualified_components",
            "_qualified_cache_key",
            "_hourglass_stiffness_matrix",
            "_stiffness_matrix",
        )
        snapshot = {
            name: (hasattr(self, name), getattr(self, name, None))
            for name in names
        }
        try:
            yield
        finally:
            for name, (present, value) in snapshot.items():
                if present:
                    object.__setattr__(self, name, value)
                elif hasattr(self, name):
                    object.__delattr__(self, name)
            restored_cache = object.__getattribute__(self, "__dict__").get(
                "_nl_cache"
            )
            restored_key = object.__getattribute__(self, "__dict__").get(
                "_nl_cache_key"
            )
            if restored_cache is None or restored_key is None:
                _clear_q4_nonlinear_cache_provenance(self)
            else:
                _bind_q4_nonlinear_cache_provenance(
                    self,
                    restored_cache,
                    restored_key,
                )
            restored_component_guard = object.__getattribute__(
                self,
                "__dict__",
            ).get("_qualified_component_guard")
            if (
                type(restored_component_guard) is tuple
                and len(restored_component_guard) == 10
                and object.__getattribute__(self, "__dict__").get(
                    "_qualified_components"
                )
                is restored_component_guard[8]
                and object.__getattribute__(self, "__dict__").get(
                    "_qualified_cache_key"
                )
                is restored_component_guard[9]
            ):
                _bind_q4_component_cache_provenance(
                    self,
                    restored_component_guard,
                    restored_component_guard[0],
                    restored_component_guard[4],
                )
            else:
                _clear_q4_component_cache_provenance(self)

    def _local_frame_and_derivatives(
        self,
        coords: np.ndarray,
        derivative_xi: np.ndarray,
        derivative_eta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Scale-invariant varying-frame geometry for the qualified Q4.

        The inherited shell guard is expressed in absolute square-length
        units.  Q4 uses the same frame and derivatives, but judges every
        degeneracy against the nodal diameter so geometrically similar facets
        at micrometre, metre and megametre scales receive the same decision.
        """

        coordinates = np.asarray(coords, dtype=float)
        length = _characteristic_length(coordinates)
        length_floor = 64.0 * np.finfo(float).eps * length
        area_floor = 64.0 * np.finfo(float).eps * length * length
        jacobian = self.compute_jacobian(
            coordinates,
            np.asarray(derivative_xi, dtype=float),
            np.asarray(derivative_eta, dtype=float),
        )
        tangent_xi = jacobian[0]
        tangent_eta = jacobian[1]
        cross = np.cross(tangent_xi, tangent_eta)
        cross_scale = float(np.max(np.abs(cross)))
        determinant = (
            0.0
            if cross_scale <= 0.0
            else cross_scale * float(np.linalg.norm(cross / cross_scale))
        )
        inherited: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float]]
        try:
            inherited = _QUALIFIED_Q4_BASE_LOCAL_FRAME_KERNEL(
                self,
                coordinates,
                derivative_xi,
                derivative_eta,
            )
        except ValueError:
            inherited = None
        if inherited is not None and math.isfinite(determinant) and determinant > area_floor:
            inherited_frame = np.asarray(inherited[0], dtype=float)
            inherited_first_source = (
                tangent_xi
                - float(tangent_xi @ inherited_frame[:, 2]) * inherited_frame[:, 2]
            )
            inherited_local_jacobian = np.asarray(
                (
                    (
                        float(tangent_xi @ inherited_frame[:, 0]),
                        float(tangent_xi @ inherited_frame[:, 1]),
                    ),
                    (
                        float(tangent_eta @ inherited_frame[:, 0]),
                        float(tangent_eta @ inherited_frame[:, 1]),
                    ),
                ),
                dtype=float,
            )
            inherited_local_determinant = float(
                np.linalg.det(inherited_local_jacobian)
            )
            if (
                float(np.linalg.norm(inherited_first_source)) > length_floor
                and math.isfinite(inherited_local_determinant)
                and abs(inherited_local_determinant) > area_floor
            ):
                # Preserve the established admitted response byte-for-byte,
                # but only after the dimensionless certificate has passed.
                return inherited
        if not math.isfinite(determinant) or determinant <= area_floor:
            raise ValueError(
                f"Shell element {self.element_id} has a dimensionless near-zero "
                "surface Jacobian"
            )
        normal = _normalize(cross, "varying-frame normal")
        first_source = tangent_xi - float(tangent_xi @ normal) * normal
        first_norm = float(np.linalg.norm(first_source))
        if not math.isfinite(first_norm) or first_norm <= length_floor:
            first = None
            for begin, end in (
                (0, 1),
                (0, 2),
                (1, 2),
                (3, 2),
                (0, 3),
                (1, 3),
            ):
                edge = coordinates[end] - coordinates[begin]
                projected = edge - float(edge @ normal) * normal
                if float(np.linalg.norm(projected)) > length_floor:
                    first = _normalize(projected, "fallback in-plane direction")
                    break
            if first is None:
                raise ValueError(
                    f"Shell element {self.element_id} has no stable in-plane direction"
                )
        else:
            first = _normalize(first_source, "varying-frame first tangent")
        second_source = np.cross(normal, first)
        if float(np.linalg.norm(second_source)) <= 64.0 * np.finfo(float).eps:
            raise ValueError(
                f"Shell element {self.element_id} has an invalid local y direction"
            )
        second = _normalize(second_source, "varying-frame second tangent")
        first = _normalize(
            np.cross(second, normal),
            "varying-frame renormalized first tangent",
        )
        frame = np.column_stack((first, second, normal))
        local_jacobian = np.asarray(
            (
                (
                    float(tangent_xi @ first),
                    float(tangent_xi @ second),
                ),
                (
                    float(tangent_eta @ first),
                    float(tangent_eta @ second),
                ),
            ),
            dtype=float,
        )
        local_determinant = float(np.linalg.det(local_jacobian))
        if (
            not math.isfinite(local_determinant)
            or abs(local_determinant) <= area_floor
        ):
            raise ValueError(
                f"Shell element {self.element_id} has a dimensionless singular "
                "local Jacobian"
            )
        inverse = np.linalg.inv(local_jacobian)
        dshape_x = inverse[0, 0] * derivative_xi + inverse[0, 1] * derivative_eta
        dshape_y = inverse[1, 0] * derivative_xi + inverse[1, 1] * derivative_eta
        return frame, dshape_x, dshape_y, determinant

    def _inverse_planar_jacobian(
        self,
        planar: np.ndarray,
        jacobian: np.ndarray,
        label: str,
    ) -> tuple[np.ndarray, float]:
        length = _characteristic_length(np.asarray(planar, dtype=float))
        determinant = float(np.linalg.det(np.asarray(jacobian, dtype=float)))
        if (
            not math.isfinite(determinant)
            or abs(determinant)
            <= 64.0 * np.finfo(float).eps * length * length
        ):
            raise ValueError(
                f"Shell element {self.element_id} has a dimensionless singular {label}"
            )
        return np.linalg.inv(np.asarray(jacobian, dtype=float)), determinant

    def _mitc4_shear_b_matrix(
        self,
        planar: np.ndarray,
        samples: Dict[str, tuple[np.ndarray, np.ndarray]],
        xi: float,
        eta: float,
    ) -> tuple[np.ndarray, float]:
        """Qualified scale-invariant form of the established MITC4 map."""

        _shape, derivative_xi, derivative_eta = self.compute_shape_functions(
            float(xi), float(eta)
        )
        jacobian = np.asarray(
            (
                (
                    float(derivative_xi @ planar[:, 0]),
                    float(derivative_xi @ planar[:, 1]),
                ),
                (
                    float(derivative_eta @ planar[:, 0]),
                    float(derivative_eta @ planar[:, 1]),
                ),
            ),
            dtype=float,
        )
        try:
            inherited = _QUALIFIED_Q4_BASE_MITC4_SHEAR_KERNEL(
                self,
                planar,
                samples,
                xi,
                eta,
            )
        except ValueError:
            inherited = None
        if inherited is not None:
            length = _characteristic_length(np.asarray(planar, dtype=float))
            inherited_determinant = float(inherited[1])
            if (
                math.isfinite(inherited_determinant)
                and abs(inherited_determinant)
                > 64.0 * np.finfo(float).eps * length * length
            ):
                return inherited
        inverse, determinant = self._inverse_planar_jacobian(
            planar,
            jacobian,
            "MITC4 in-plane Jacobian",
        )
        covariant = np.vstack(
            (
                0.5 * (1.0 - eta) * samples["A"][0]
                + 0.5 * (1.0 + eta) * samples["C"][0],
                0.5 * (1.0 - xi) * samples["D"][1]
                + 0.5 * (1.0 + xi) * samples["B"][1],
            )
        )
        return inverse @ covariant, determinant

    def _nonlinear_geometry(self, mesh: Any) -> Dict[str, Any]:
        """Qualified Q4 nonlinear geometry with dimensionless 2x2 guards."""

        _validate_q4_quadrature_authority(self)
        coordinates = self.get_node_coordinates(mesh)
        relative = coordinates - coordinates[0]
        cache_key = (
            id(mesh),
            tuple(int(value) for value in self.node_ids),
            np.ascontiguousarray(relative, dtype=np.float64).tobytes(order="C"),
            None
            if self.reference_normal is None
            else tuple(np.asarray(self.reference_normal, dtype=np.float64)),
            int(self.director_polarity),
            Q4_QUADRATURE_AUTHORITY_ID,
        )
        namespace = object.__getattribute__(self, "__dict__")
        cache = dict.get(namespace, "_nl_cache")
        current_cache_key = dict.get(namespace, "_nl_cache_key")
        if cache is not None:
            _require_q4_nonlinear_cache_provenance(
                self,
                cache,
                current_cache_key,
            )
            if current_cache_key == cache_key:
                return cache
        object.__setattr__(self, "_nl_cache", None)
        object.__setattr__(self, "_nl_cache_key", None)
        _clear_q4_nonlinear_cache_provenance(self)
        try:
            inherited = _QUALIFIED_Q4_BASE_NONLINEAR_GEOMETRY(self, mesh)
            frozen_inherited = _freeze_qualified_component_cache(inherited)
            object.__setattr__(self, "_nl_cache", frozen_inherited)
            object.__setattr__(self, "_nl_cache_key", cache_key)
            _bind_q4_nonlinear_cache_provenance(
                self,
                frozen_inherited,
                cache_key,
            )
            return frozen_inherited
        except np.linalg.LinAlgError:
            # The inherited 2x2 helper alone rejected a dimensionally small,
            # otherwise regular geometry.  Rebuild the same arrays using the
            # qualified dimensionless inverse below.
            pass
        frame = self._center_frame(coordinates)
        transform = self._local_dof_transform(frame)
        planar = coordinates @ frame[:, :2]
        gauss_data = []
        for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
            shape, derivative_xi, derivative_eta = self.compute_shape_functions(
                float(xi), float(eta)
            )
            jacobian = np.asarray(
                (
                    (
                        float(derivative_xi @ planar[:, 0]),
                        float(derivative_xi @ planar[:, 1]),
                    ),
                    (
                        float(derivative_eta @ planar[:, 0]),
                        float(derivative_eta @ planar[:, 1]),
                    ),
                ),
                dtype=float,
            )
            inverse, determinant = self._inverse_planar_jacobian(
                planar,
                jacobian,
                "nonlinear in-plane Jacobian",
            )
            derivative_x = (
                inverse[0, 0] * derivative_xi + inverse[0, 1] * derivative_eta
            )
            derivative_y = (
                inverse[1, 0] * derivative_xi + inverse[1, 1] * derivative_eta
            )
            membrane, bending, shear = self._build_shell_b_matrices(
                shape,
                derivative_x,
                derivative_y,
            )
            drilling = self._build_drilling_b_matrix(
                shape,
                derivative_x,
                derivative_y,
            )
            transverse_gradient = np.zeros((2, self.total_dofs), dtype=float)
            transverse_gradient[0, 2::6] = derivative_x
            transverse_gradient[1, 2::6] = derivative_y
            gauss_data.append(
                {
                    "B_m": membrane,
                    "B_b": bending,
                    "B_s": shear,
                    "B_d": drilling,
                    "Gw": transverse_gradient,
                    "detw": abs(determinant) * float(weight),
                }
            )

        _planar, samples = self._mitc4_shear_samples(coordinates, frame)
        shear_data = []
        for (xi, eta), weight in zip(
            self.GAUSS_POINTS_2x2,
            self.GAUSS_WEIGHTS_2x2,
        ):
            shear, determinant = self._mitc4_shear_b_matrix(
                planar,
                samples,
                float(xi),
                float(eta),
            )
            shear_data.append(
                {
                    "B_s": shear,
                    "detw": abs(determinant) * float(weight),
                }
            )

        count = len(gauss_data)
        membrane_all = np.zeros((count, 3, self.total_dofs), dtype=float)
        bending_all = np.zeros_like(membrane_all)
        drilling_all = np.zeros((count, 1, self.total_dofs), dtype=float)
        gradient_all = np.zeros((count, 2, self.total_dofs), dtype=float)
        determinant_all = np.zeros(count, dtype=float)
        for index, data in enumerate(gauss_data):
            membrane_all[index] = data["B_m"]
            bending_all[index] = data["B_b"]
            drilling_all[index] = data["B_d"]
            gradient_all[index] = data["Gw"]
            determinant_all[index] = data["detw"]
        shear_all = np.asarray([data["B_s"] for data in shear_data], dtype=float)
        shear_determinant_all = np.asarray(
            [data["detw"] for data in shear_data],
            dtype=float,
        )
        cache = {
            "R0": frame,
            "T0": transform,
            "gp": gauss_data,
            "shear": shear_data,
            "B_m_all": membrane_all,
            "B_b_all": bending_all,
            "B_d_all": drilling_all,
            "Gw_all": gradient_all,
            "detw_all": determinant_all,
            "B_s_all": shear_all,
            "detw_shear_all": shear_determinant_all,
        }
        frozen_cache = _freeze_qualified_component_cache(cache)
        object.__setattr__(self, "_nl_cache", frozen_cache)
        object.__setattr__(self, "_nl_cache_key", cache_key)
        _bind_q4_nonlinear_cache_provenance(self, frozen_cache, cache_key)
        return frozen_cache

    @_q4_serialized_output_boundary
    def to_dict(self) -> Dict[str, Any]:
        if type(self) is not __class__:
            raise ValueError("qualified Q4 serialization requires the exact class")
        payload = super().to_dict()
        payload.update(
            {
                "drilling_stabilization": float(self.drilling_stabilization),
                "implementation_id": IMPLEMENTATION_ID,
                "formulation_id": FORMULATION_ID,
                "hourglass_stabilization": float(self.hourglass_stabilization),
                "planar_tolerance": self.planar_tolerance,
                "pl_stabilization": self.pl_stabilization,
                "reduced_integration": bool(self.reduced_integration),
                "recovery_policy_id": RECOVERY_POLICY_ID,
                "stationary_solve_policy_id": STATIONARY_SOLVE_POLICY_ID,
                "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
                "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
                "current_state_binding_schema_id": (
                    Q4_CURRENT_STATE_BINDING_SCHEMA_ID
                ),
                "current_state_algorithmic_origin_schema_id": (
                    Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
                ),
                "current_state_tangent_decomposition_policy_id": (
                    Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
                ),
                "current_state_projection_policy_id": (
                    Q4_CURRENT_STATE_PROJECTION_POLICY_ID
                ),
                "activity_disposition_schema_id": (
                    Q4_ACTIVITY_DISPOSITION_SCHEMA_ID
                ),
                "deleted_frozen_policy_id": Q4_DELETED_FROZEN_POLICY_ID,
                "failed_state_policy_id": Q4_FAILED_STATE_POLICY_ID,
                "quadrature_authority_id": Q4_QUADRATURE_AUTHORITY_ID,
                "director_polarity": int(self.director_polarity),
                "reference_normal": (
                    None
                    if self.reference_normal is None
                    else np.asarray(self.reference_normal, dtype=float).tolist()
                ),
                "warped_formulation": self.warped_formulation,
            }
        )
        return payload

    @classmethod
    @_q4_serialized_input_boundary
    def from_dict(cls, payload: Mapping[str, Any]) -> "QualifiedE4PLShellElement":
        """Reconstruct a candidate from its lossless JSON-compatible record."""

        if cls is not __class__:
            raise ValueError("qualified Q4 deserialization requires the exact class")
        data = dict(payload)
        if data.get("formulation_id") not in {FORMULATION_ID, _PLANAR_FORMULATION_ID}:
            raise ValueError("serialized E4-PL formulation_id is missing or incompatible")
        if data.get("type") not in {cls.__name__, "e4-pl"}:
            raise ValueError("serialized E4-PL type is incompatible")
        identity = {
            "implementation_id": IMPLEMENTATION_ID,
            "recovery_policy_id": RECOVERY_POLICY_ID,
            "stationary_solve_policy_id": STATIONARY_SOLVE_POLICY_ID,
            "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
            "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
            "current_state_binding_schema_id": (
                Q4_CURRENT_STATE_BINDING_SCHEMA_ID
            ),
            "current_state_algorithmic_origin_schema_id": (
                Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
            ),
            "current_state_tangent_decomposition_policy_id": (
                Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
            ),
            "current_state_projection_policy_id": (
                Q4_CURRENT_STATE_PROJECTION_POLICY_ID
            ),
            "activity_disposition_schema_id": (
                Q4_ACTIVITY_DISPOSITION_SCHEMA_ID
            ),
            "deleted_frozen_policy_id": Q4_DELETED_FROZEN_POLICY_ID,
            "failed_state_policy_id": Q4_FAILED_STATE_POLICY_ID,
            "quadrature_authority_id": Q4_QUADRATURE_AUTHORITY_ID,
        }
        v7_pre_quadrature_identity = {
            name: expected
            for name, expected in identity.items()
            if name != "quadrature_authority_id"
        }
        v7_pre_activity_identity = {
            name: expected
            for name, expected in identity.items()
            if name
            not in {
                "activity_disposition_schema_id",
                "deleted_frozen_policy_id",
                "failed_state_policy_id",
                "quadrature_authority_id",
            }
        }
        v5_identity = {
            "implementation_id": _V5_IMPLEMENTATION_ID,
            "recovery_policy_id": RECOVERY_POLICY_ID,
            "stationary_solve_policy_id": STATIONARY_SOLVE_POLICY_ID,
            "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
            "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
        }
        v6_identity = {
            "implementation_id": _V6_IMPLEMENTATION_ID,
            "recovery_policy_id": RECOVERY_POLICY_ID,
            "stationary_solve_policy_id": STATIONARY_SOLVE_POLICY_ID,
            "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
            "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
            "current_state_binding_schema_id": (
                _V6_CURRENT_STATE_BINDING_SCHEMA_ID
            ),
            "current_state_tangent_decomposition_policy_id": (
                _V6_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
            ),
            "current_state_projection_policy_id": (
                Q4_CURRENT_STATE_PROJECTION_POLICY_ID
            ),
        }
        present = {name for name in identity if name in data}
        current_identity = present == set(identity)
        v7_pre_quadrature_identity_present = present == set(
            v7_pre_quadrature_identity
        )
        v7_pre_activity_identity_present = present == set(
            v7_pre_activity_identity
        )
        v5_identity_present = present == set(v5_identity)
        v6_identity_present = present == set(v6_identity)
        if present and data.get("formulation_id") != FORMULATION_ID:
            raise ValueError(
                "serialized current E4-PL Q4 formulation_id is incompatible"
            )
        if (
            present
            and not current_identity
            and not v7_pre_quadrature_identity_present
            and not v7_pre_activity_identity_present
            and not v5_identity_present
            and not v6_identity_present
        ):
            raise ValueError(
                "serialized E4-PL Q4 implementation/recovery/director/current-"
                "state identity is incomplete"
            )
        if current_identity:
            for name, expected in identity.items():
                if data.get(name) != expected:
                    raise ValueError(
                        f"serialized E4-PL Q4 {name} is incompatible"
                    )
        elif v7_pre_quadrature_identity_present:
            for name, expected in v7_pre_quadrature_identity.items():
                if data.get(name) != expected:
                    raise ValueError(
                        f"serialized E4-PL Q4 {name} is incompatible"
                    )
            warnings.warn(
                "Migrated an exact qualified Q4 V7 element configuration to "
                "the immutable exact quadrature authority; mechanics are unchanged.",
                QualifiedQ4MigrationWarning,
                stacklevel=2,
            )
        elif v7_pre_activity_identity_present:
            for name, expected in v7_pre_activity_identity.items():
                if data.get(name) != expected:
                    raise ValueError(
                        f"serialized E4-PL Q4 {name} is incompatible"
                    )
            warnings.warn(
                "Migrated an exact qualified Q4 V7 element configuration to "
                "the closed activity-disposition identity; mechanics and the "
                "active current-tangent schema are unchanged.",
                QualifiedQ4MigrationWarning,
                stacklevel=2,
            )
        elif v5_identity_present:
            for name, expected in v5_identity.items():
                if data.get(name) != expected:
                    raise ValueError(
                        f"serialized E4-PL Q4 {name} is incompatible"
                    )
            warnings.warn(
                "Migrated an exact qualified Q4 V5 element configuration to V7; "
                "the frozen mechanics are unchanged and finalized nonlinear "
                "states now require the V7 algorithmic-tangent binding before "
                "current-state modal or buckling use.",
                QualifiedQ4MigrationWarning,
                stacklevel=2,
            )
        elif v6_identity_present:
            for name, expected in v6_identity.items():
                if data.get(name) != expected:
                    raise ValueError(
                        f"serialized E4-PL Q4 {name} is incompatible"
                    )
            warnings.warn(
                "Migrated an exact qualified Q4 V6 element configuration to V7; "
                "the frozen mechanics are unchanged, but old V1 displacement-"
                "only state seals are not current-tangent evidence.",
                QualifiedQ4MigrationWarning,
                stacklevel=2,
            )
        if (
            current_identity
            or v7_pre_quadrature_identity_present
            or v7_pre_activity_identity_present
            or v5_identity_present
            or v6_identity_present
        ):
            required_director_state = {"director_polarity", "reference_normal"}
            retained_director_state = required_director_state & data.keys()
            if retained_director_state != required_director_state:
                raise ValueError(
                    "serialized E4-PL Q4 current director state is incomplete"
                )
        else:
            legacy_section = coerce_generalized_shell_section(data.get("shell_section"))
            if legacy_section is not None and np.any(legacy_section.B != 0.0):
                raise ValueError(
                    "pre-policy B-coupled qualified Q4 records cannot be migrated: "
                    "their physical director is not authoritative"
                )
            if any(
                name in data
                for name in ("reference_normal", "director_polarity")
            ):
                raise ValueError(
                    "pre-policy qualified Q4 records cannot contain partial director data"
                )
            warnings.warn(
                "Migrated a pre-policy uncoupled qualified Q4 record with its "
                "connectivity-relative director behavior preserved; B-coupled "
                "records require an authoritative reference_normal.",
                QualifiedQ4MigrationWarning,
                stacklevel=2,
            )
        if data.get("formulation_id") in {FORMULATION_ID, _PLANAR_FORMULATION_ID}:
            expected_current_keys = {
                "activity_disposition_schema_id",
                "current_state_algorithmic_origin_schema_id",
                "current_state_binding_schema_id",
                "current_state_projection_policy_id",
                "current_state_tangent_decomposition_policy_id",
                "deleted_frozen_policy_id",
                "director_polarity",
                "director_polarity_policy_id",
                "director_reversal_transform_id",
                "drilling_stabilization",
                "element_id",
                "failed_state_policy_id",
                "formulation_id",
                "hourglass_stabilization",
                "implementation_id",
                "material_angle_deg",
                "material_direction",
                "material_name",
                "node_ids",
                "planar_tolerance",
                "pl_stabilization",
                "quadrature_authority_id",
                "recovery_policy_id",
                "reduced_integration",
                "reference_normal",
                "shell_section",
                "stationary_solve_policy_id",
                "thickness",
                "type",
                "warped_formulation",
            }
            expected_closed_keys = (
                (expected_current_keys - set(identity)) | present
                if present
                else expected_current_keys
                - set(identity)
                - {"director_polarity", "reference_normal"}
            )
            if set(data) != expected_closed_keys:
                missing = sorted(expected_closed_keys - set(data))
                extra = sorted(set(data) - expected_closed_keys)
                raise ValueError(
                    "serialized exact E4-PL Q4 keys are incompatible; "
                    f"missing={missing}, extra={extra}"
                )
            raw_node_ids = data["node_ids"]
            raw_direction = data["material_direction"]
            raw_reference_normal = data.get("reference_normal")
            raw_section = data["shell_section"]
            if raw_section is not None:
                section_keys = {
                    "A",
                    "As",
                    "B",
                    "D",
                    "mass_per_area",
                    "name",
                    "rotary_inertia_per_area",
                }
                if type(raw_section) is not dict or set(raw_section) != section_keys:
                    raise ValueError(
                        "serialized exact E4-PL Q4 shell_section keys are incompatible"
                    )
                if type(raw_section["name"]) is not str:
                    raise ValueError(
                        "serialized exact E4-PL Q4 shell_section name is incompatible"
                    )
                for matrix_name, shape in (
                    ("A", (3, 3)),
                    ("B", (3, 3)),
                    ("D", (3, 3)),
                    ("As", (2, 2)),
                ):
                    matrix = raw_section[matrix_name]
                    if (
                        type(matrix) not in {list, tuple}
                        or len(matrix) != shape[0]
                        or any(type(row) not in {list, tuple} for row in matrix)
                        or any(len(row) != shape[1] for row in matrix)
                        or any(
                            type(component) is not float
                            or not math.isfinite(component)
                            for row in matrix
                            for component in row
                        )
                    ):
                        raise ValueError(
                            "serialized exact E4-PL Q4 shell_section matrix is incompatible"
                        )
                for mass_name in ("mass_per_area", "rotary_inertia_per_area"):
                    mass = raw_section[mass_name]
                    if mass is not None and (
                        type(mass) is not float
                        or not math.isfinite(mass)
                    ):
                        raise ValueError(
                            "serialized exact E4-PL Q4 shell_section mass is incompatible"
                        )
                for symmetric_name in ("A", "D", "As"):
                    symmetric = raw_section[symmetric_name]
                    if any(
                        symmetric[row][column] != symmetric[column][row]
                        for row in range(len(symmetric))
                        for column in range(len(symmetric))
                    ):
                        raise ValueError(
                            "serialized exact E4-PL Q4 shell_section symmetry is incompatible"
                        )
            exact_vector = lambda value: (
                type(value) in {list, tuple}
                and len(value) == 3
                and all(type(component) is float for component in value)
                and all(math.isfinite(component) for component in value)
            )
            exact_floats = (
                "thickness",
                "material_angle_deg",
                "drilling_stabilization",
                "hourglass_stabilization",
                "pl_stabilization",
                "planar_tolerance",
            )
            exact_unit_vector = lambda value: (
                exact_vector(value)
                and (norm := math.sqrt(sum(component * component for component in value)))
                > 0.0
                and tuple(component / norm for component in value) == tuple(value)
            )
            if (
                type(data["element_id"]) is not int
                or type(raw_node_ids) not in {list, tuple}
                or len(raw_node_ids) != 4
                or not all(type(value) is int for value in raw_node_ids)
                or type(data["material_name"]) is not str
                or any(
                    type(data[name]) is not float
                    or not math.isfinite(data[name])
                    for name in exact_floats
                )
                or data["thickness"] <= 0.0
                or data["pl_stabilization"] < 0.0
                or data["planar_tolerance"] < 0.0
                or data["drilling_stabilization"] < 0.0
                or data["hourglass_stabilization"] < 0.0
                or type(data["reduced_integration"]) is not bool
                or type(data["warped_formulation"]) is not str
                or data["warped_formulation"] not in _WARPED_FORMULATIONS
                or (
                    bool(present)
                    and (
                        type(data["director_polarity"]) is not int
                        or data["director_polarity"] not in {-1, 1}
                    )
                )
                or (
                    raw_direction is not None
                    and not exact_vector(raw_direction)
                )
                or (
                    raw_reference_normal is not None
                    and not exact_unit_vector(raw_reference_normal)
                )
                or (
                    bool(present)
                    and data["director_polarity"] != 1
                    and raw_reference_normal is None
                )
            ):
                raise ValueError(
                    "serialized current E4-PL Q4 configuration uses noncanonical types"
                )
            element_id = data["element_id"]
            node_ids = list(raw_node_ids)
            material_name = data["material_name"]
            thickness = data["thickness"]
            reduced_integration = data["reduced_integration"]
            director_polarity = data.get("director_polarity", 1)
        candidate = cls(
            element_id=element_id,
            node_ids=node_ids,
            material_name=material_name,
            thickness=thickness,
            drilling_stabilization=float(data.get("drilling_stabilization", 1.0e-3)),
            reduced_integration=reduced_integration,
            hourglass_stabilization=float(data.get("hourglass_stabilization", 1.0e-3)),
            material_direction=data.get("material_direction"),
            material_angle_deg=float(data.get("material_angle_deg", 0.0)),
            shell_section=data.get("shell_section"),
            pl_stabilization=float(data.get("pl_stabilization", 1.0)),
            planar_tolerance=float(data.get("planar_tolerance", 1.0e-10)),
            warped_formulation=str(
                data.get(
                    "warped_formulation",
                    "varying_frame"
                    if bool(data.get("legacy_warped_fallback", True))
                    else "reject",
                )
            ),
            reference_normal=data.get("reference_normal"),
            director_polarity=director_polarity,
        )
        return candidate

    @property
    def physical_reference_director(self) -> Optional[np.ndarray]:
        """Return the persisted physical director authority, when supplied."""

        if self.reference_normal is None:
            return None
        return (
            float(self.director_polarity)
            * np.asarray(self.reference_normal, dtype=float)
        ).copy()

    def _physical_director_context(
        self,
        numbered_frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Map numbered Equation-7 fields to a physical-director frame.

        A reflected D4 numbering reverses the Equation-7 normal and one local
        tangent.  With ``s = sign(n_numbered . d_physical)``, the exact
        numbered-to-physical engineering maps are

        ``E = diag(1, 1, s)``, ``K = s E`` and
        ``H = s diag(1, s)``.

        ``E`` acts on membrane fields, ``K`` on curvatures/moments and ``H``
        on transverse shear.  All three are orthogonal involutions, so the
        same matrices map conjugate resultants in the reverse direction.
        """

        frame = np.asarray(numbered_frame, dtype=float).reshape(3, 3)
        if self.reference_normal is None:
            return (
                frame.copy(),
                np.eye(3, dtype=float),
                np.eye(3, dtype=float),
                np.eye(2, dtype=float),
                1,
            )
        director = self.physical_reference_director
        assert director is not None
        alignment = float(frame[:, 2] @ director)
        if not math.isfinite(alignment) or abs(alignment) <= 1.0e-8:
            raise ValueError(
                "qualified Q4 reference_normal is tangential to the local facet "
                "and cannot establish physical director polarity"
            )
        sign = 1 if alignment > 0.0 else -1
        physical_frame = frame.copy()
        physical_frame[:, 1] *= float(sign)
        physical_frame[:, 2] *= float(sign)
        membrane = np.diag((1.0, 1.0, float(sign)))
        curvature = float(sign) * membrane
        shear = float(sign) * np.diag((1.0, float(sign)))
        return physical_frame, membrane, curvature, shear, sign

    def init_nonlinear_state(self, num_layers: int) -> Dict[str, Any]:
        """Create a Q4 state only for one admitted Lobatto rule."""

        _validate_q4_quadrature_authority(self)
        layers = _qualified_q4_layer_count(num_layers)
        return super().init_nonlinear_state(layers)

    def _requires_algorithmic_return_map_origin(self, material: Any) -> bool:
        """Return whether the layered constitutive update is path dependent."""

        if self.shell_section is not None:
            return False
        return bool(
            getattr(material, "hardening_curve", None) is not None
            or getattr(material, "hill_yield", None) is not None
        )

    def _algorithmic_origin_payload(
        self,
        material: Any,
        prior_state: Optional[Mapping[str, Any]],
        num_layers: int,
    ) -> Optional[Dict[str, Any]]:
        """Capture the exact parent history used by one return-map call.

        The accepted tangent is a derivative of the discrete update from the
        parent history to the accepted layer strain.  The converged plastic
        state is the *output* of that update and is not a valid replacement
        for its parent.  Persisting the parent makes the production update
        exactly replayable without storing a dense element matrix.
        """

        layers = _qualified_q4_layer_count(num_layers)
        if not self._requires_algorithmic_return_map_origin(material):
            return None
        source: Mapping[str, Any]
        if prior_state is None:
            source = super().init_nonlinear_state(layers)
        elif isinstance(prior_state, Mapping):
            source = prior_state
        else:
            raise TypeError(
                "qualified Q4 plastic return-map parent state must be a mapping"
            )
        points = len(self.gauss_points) * layers
        plastic = np.asarray(source.get("plastic_strain", ()), dtype=np.float64)
        alpha = np.asarray(source.get("alpha", ()), dtype=np.float64)
        if (
            plastic.shape != (points, 3)
            or alpha.shape != (points,)
            or not np.all(np.isfinite(plastic))
            or not np.all(np.isfinite(alpha))
        ):
            raise ValueError(
                "qualified Q4 plastic return-map parent state has incompatible "
                "plastic_strain or alpha"
            )
        return {
            "schema_id": Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID,
            "kind": "LAYERED_DISCRETE_RETURN_MAP_PARENT_STATE",
            "num_layers": layers,
            # Persist the descriptor in canonical JSON-native form.  The
            # constitutive evaluator materializes owned binary64 arrays only
            # after schema/hash validation.
            "parent_plastic_strain": plastic.tolist(),
            "parent_alpha": alpha.tolist(),
        }

    def attach_current_tangent_algorithmic_origin(
        self,
        material: Any,
        prior_state: Optional[Mapping[str, Any]],
        trial_state: Mapping[str, Any],
        num_layers: int,
        *,
        tangent_evaluated: bool,
    ) -> Dict[str, Any]:
        """Return an owned trial state carrying its exact update origin.

        Batch and scalar assembly call this at the same accepted constitutive
        evaluation boundary.  Residual-only candidates deliberately carry no
        origin because they did not evaluate an algorithmic tangent and must
        not become current-state tangent evidence.
        """

        if not isinstance(trial_state, Mapping):
            raise TypeError("qualified Q4 trial state must be a mapping")
        made = copy.deepcopy(dict(trial_state))
        made.pop(_Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_KEY, None)
        if not tangent_evaluated:
            return made
        origin = self._algorithmic_origin_payload(
            material,
            prior_state,
            num_layers,
        )
        if origin is not None:
            made[_Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_KEY] = origin
        return made

    def _validated_algorithmic_origin(
        self,
        material: Any,
        state: Mapping[str, Any],
        num_layers: int,
    ) -> Optional[Dict[str, Any]]:
        """Validate and return an owned exact discrete-update parent."""

        layers = _qualified_q4_layer_count(num_layers)
        raw = state.get(_Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_KEY)
        required = self._requires_algorithmic_return_map_origin(material)
        if not required:
            if raw is not None:
                raise ValueError(
                    "qualified Q4 elastic/generalized state must not carry a "
                    "plastic algorithmic origin"
                )
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(
                "qualified Q4 plastic committed state lacks the accepted "
                "algorithmic return-map origin"
            )
        expected_keys = {
            "schema_id",
            "kind",
            "num_layers",
            "parent_plastic_strain",
            "parent_alpha",
        }
        if set(raw) != expected_keys:
            raise ValueError(
                "qualified Q4 algorithmic return-map origin schema is incomplete"
            )
        if (
            raw.get("schema_id")
            != Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
            or raw.get("kind")
            != "LAYERED_DISCRETE_RETURN_MAP_PARENT_STATE"
            or raw.get("num_layers") != layers
        ):
            raise ValueError(
                "qualified Q4 algorithmic return-map origin identity is incompatible"
            )
        points = len(self.gauss_points) * layers
        plastic = np.asarray(
            raw.get("parent_plastic_strain", ()), dtype=np.float64
        )
        alpha = np.asarray(raw.get("parent_alpha", ()), dtype=np.float64)
        if (
            plastic.shape != (points, 3)
            or alpha.shape != (points,)
            or not np.all(np.isfinite(plastic))
            or not np.all(np.isfinite(alpha))
        ):
            raise ValueError(
                "qualified Q4 algorithmic return-map origin arrays are incompatible"
            )
        return {
            "plastic_strain": plastic.copy(),
            "alpha": alpha.copy(),
            **{
                key: copy.deepcopy(state[key])
                for key in _Q4_INITIAL_STATE_KEYS
                if key in state
            },
        }

    def _validate_committed_current_kinematics(
        self,
        mesh: Any,
        committed_u_elem: np.ndarray,
        state: Mapping[str, Any],
        num_layers: int,
    ) -> None:
        """Reject a seal over stored kinematics from another displacement."""

        tracked = {
            "membrane_strain",
            "curvature",
            "transverse_shear_strain",
            "kinematic_layer_strain",
        }
        if not tracked.intersection(state):
            return
        cache = self._nonlinear_geometry(mesh)
        local = np.asarray(cache["T0"], dtype=np.float64) @ committed_u_elem
        count = len(cache["gp"])
        membrane = np.zeros((count, 3), dtype=np.float64)
        curvature = np.zeros_like(membrane)
        for index, data in enumerate(cache["gp"]):
            gradient = np.asarray(data["Gw"], dtype=np.float64) @ local
            membrane[index] = np.asarray(data["B_m"], dtype=np.float64) @ local + np.asarray(
                (
                    0.5 * gradient[0] ** 2,
                    0.5 * gradient[1] ** 2,
                    gradient[0] * gradient[1],
                ),
                dtype=np.float64,
            )
            curvature[index] = np.asarray(data["B_b"], dtype=np.float64) @ local
        shear = np.einsum(
            "gij,j->gi",
            np.asarray(cache["B_s_all"], dtype=np.float64),
            local,
        )
        expected = {
            "membrane_strain": membrane,
            "curvature": curvature,
            "transverse_shear_strain": shear,
        }
        if "kinematic_layer_strain" in state:
            z_layers, _weights = lobatto_layers(
                int(num_layers), float(self.thickness)
            )
            expected["kinematic_layer_strain"] = (
                membrane[:, None, :]
                + z_layers[None, :, None] * curvature[:, None, :]
            ).reshape(count * int(num_layers), 3)
        for name, values in expected.items():
            if name not in state:
                continue
            stored = np.asarray(state[name], dtype=np.float64)
            if (
                stored.shape != values.shape
                or not np.all(np.isfinite(stored))
                or not np.array_equal(stored, values)
            ):
                raise ValueError(
                    "qualified Q4 committed state kinematics disagree with the "
                    f"bound displacement field {name!r}"
                )

    def _accepted_algorithmic_update_fingerprint(
        self,
        material: Any,
        state: Mapping[str, Any],
        num_layers: int,
    ) -> Optional[str]:
        """Hash the atomically retained parent and accepted constitutive core."""

        layers = _qualified_q4_layer_count(num_layers)
        parent = self._validated_algorithmic_origin(material, state, layers)
        if parent is None:
            return None
        points = len(self.gauss_points) * layers
        core: Dict[str, Any] = {}
        for name, shape in (
            ("plastic_strain", (points, 3)),
            ("alpha", (points,)),
            ("layer_strain", (points, 3)),
        ):
            values = np.asarray(state.get(name, ()), dtype=np.float64)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    "qualified Q4 accepted algorithmic state has incompatible "
                    f"{name}"
                )
            core[name] = values
        return canonical_sha256(
            {
                "layout": "Q4_ACCEPTED_DISCRETE_CONSTITUTIVE_UPDATE_V1",
                "algorithmic_origin": state[
                    _Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_KEY
                ],
                "accepted_core": core,
            }
        )

    def _is_compact_batch_constitutive_state(
        self,
        material: Any,
        state: Mapping[str, Any],
    ) -> bool:
        compact_keys = {
            "plastic_strain",
            "alpha",
            "layer_strain",
            _Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_KEY,
            _Q4_CURRENT_STATE_BINDING_KEY,
            _Q4_CURRENT_STATE_DIGEST_KEY,
            "state_digest",
            _Q4_ACTIVITY_DISPOSITION_KEY,
        }
        return bool(
            set(state).issubset(compact_keys)
            and self.shell_section is None
            and getattr(material, "hardening_curve", None) is not None
            and getattr(material, "hill_yield", None) is None
        )

    def _assumed_strain_kernel(
        self,
        mesh: Any,
        material: Any,
        vector: np.ndarray,
        state: Optional[Mapping[str, Any]],
        layers: int,
        tangent: bool,
        *,
        return_components: bool = False,
    ) -> Any:
        """Assumed-strain layered plastic kernel, or ``None`` when not applicable.

        Supported plastic elements (see :mod:`._q4_assumed_strain_plasticity`)
        replace the inherited compatible kernel by the same batch kernel the
        assembly plans use, so scalar, reference and planned evaluations share
        one floating-point implementation.  The qualified ``Cr`` delta is
        added by the caller exactly as on the historical route.
        """

        if _q4_assumed_strain.require_supported(self, mesh, material):
            _q4_assumed_strain.require_state_supported(self, state)
        elif not _q4_assumed_strain.elastic_route(self, mesh, material):
            return None
        cache = self._nonlinear_geometry(mesh)
        points = len(cache["gp"]) * int(layers)
        if state is None:
            plastic = np.zeros((points, 3), dtype=np.float64)
            alpha = np.zeros(points, dtype=np.float64)
        else:
            plastic = np.asarray(state["plastic_strain"], dtype=np.float64)
            alpha = np.asarray(state["alpha"], dtype=np.float64)
        if (
            plastic.shape != (points, 3)
            or alpha.shape != (points,)
            or not np.all(np.isfinite(plastic))
            or not np.all(np.isfinite(alpha))
        ):
            raise ValueError("qualified Q4 plastic state arrays are incompatible")
        offsets = _q4_assumed_strain.offset_operator(self, mesh, material)

        def one(name: str) -> np.ndarray:
            return np.asarray(cache[name], dtype=np.float64)[None]

        force, stiffness, plastic_new, alpha_new, layer_strain, extras = (
            _q4_assumed_strain.kernel_batch(
                np.asarray(vector, dtype=np.float64).reshape(1, self.total_dofs),
                one("T0"),
                one("B_m_all"),
                one("B_b_all"),
                one("B_d_all"),
                one("Gw_all"),
                one("detw_all"),
                one("B_s_all"),
                one("detw_shear_all"),
                float(material.elastic_modulus),
                float(material.poisson_ratio),
                float(material.shear_modulus),
                float(self.thickness),
                float(self.drilling_stabilization),
                bool(tangent),
                material.hardening_curve,
                plastic[None],
                alpha[None],
                int(layers),
                offsets[None],
                return_geometric=bool(return_components and tangent),
                initial_layer_offsets=(
                    None
                    if material.hardening_curve is not None
                    else _initial_offsets_or_none(self, state, cache, layers, material)
                ),
            )
        )
        trial_state = {
            "plastic_strain": plastic_new[0].copy(),
            "alpha": alpha_new[0].copy(),
            "layer_strain": np.asarray(layer_strain, dtype=np.float64).copy(),
            "layer_strain_material": np.asarray(layer_strain, dtype=np.float64).copy(),
            "kinematic_layer_strain": extras["kinematic_layer_strain"][0].copy(),
            "layer_stress": extras["layer_stress"][0].copy(),
        }
        if state is not None:
            # Inactive (all-zero, provenance-free) initial fields are carried
            # unchanged, exactly as the inherited kernel carries them.
            for key in _Q4_INITIAL_STATE_KEYS:
                if key in state:
                    trial_state[key] = copy.deepcopy(state[key])
        total = None if not tangent else np.asarray(stiffness[0], dtype=np.float64)
        if not return_components:
            return np.asarray(force[0], dtype=np.float64), total, trial_state
        if total is None:
            raise RuntimeError("qualified Q4 components require a tangent")
        geometric = np.asarray(extras["geometric"][0], dtype=np.float64)
        geometric = 0.5 * (geometric + geometric.T)
        return (
            np.asarray(force[0], dtype=np.float64),
            total,
            trial_state,
            {"material": total - geometric, "geometric": geometric},
        )

    def _replay_accepted_algorithmic_response(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: np.ndarray,
        committed_state: Mapping[str, Any],
        num_layers: int,
        *,
        return_components: bool,
        validated_compact_core: bool = False,
    ) -> Any:
        """Replay the accepted update from its parent, never from its output."""

        _validate_q4_quadrature_authority(self)
        layers = _qualified_q4_layer_count(num_layers)
        parent = self._validated_algorithmic_origin(
            material,
            committed_state,
            layers,
        )
        evaluation_state: Optional[Mapping[str, Any]]
        if parent is None:
            # Elastic and generalized tangents are uniquely determined by the
            # current kinematics and frozen section; no path descriptor exists.
            evaluation_state = committed_state
        else:
            evaluation_state = parent
        # Qualified Q4 caches its separately recoverable hourglass matrix on
        # the element.  The inherited four-node nonlinear kernel must start
        # without that qualified cache; the exact baseline delta is applied
        # below by the Q4 wrapper, just as in ordinary Newton evaluation.
        self._hourglass_stiffness_matrix = None
        inherited = self._assumed_strain_kernel(
            mesh,
            material,
            np.asarray(committed_u_elem, dtype=np.float64),
            evaluation_state,
            layers,
            True,
            return_components=return_components,
        )
        if inherited is None:
            inherited = ShellElement.compute_nonlinear_response(
                self,
                mesh,
                material,
                np.asarray(committed_u_elem, dtype=np.float64),
                evaluation_state,
                layers,
                True,
                _return_tangent_components=return_components,
            )
        candidate = inherited[2]
        if parent is not None:
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    "qualified Q4 accepted return-map replay produced no state"
                )
            compact_batch_state = self._is_compact_batch_constitutive_state(
                material,
                committed_state,
            )
            points = len(self.gauss_points) * layers
            for name, shape in (
                ("plastic_strain", (points, 3)),
                ("alpha", (points,)),
                ("layer_strain", (points, 3)),
            ):
                expected = np.asarray(
                    committed_state.get(name, ()), dtype=np.float64
                )
                actual = np.asarray(candidate.get(name, ()), dtype=np.float64)
                exact = (
                    expected.shape == shape
                    and actual.shape == shape
                    and np.all(np.isfinite(expected))
                    and np.array_equal(actual, expected)
                )
                bounded_last_place = (
                    compact_batch_state
                    and expected.shape == shape
                    and actual.shape == shape
                    and _q4_compact_replay_last_place_equivalent(
                        actual,
                        expected,
                    )
                )
                trusted_solver_core = bool(
                    validated_compact_core
                    and compact_batch_state
                )
                if (
                    expected.shape != shape
                    or actual.shape != shape
                    or not np.all(np.isfinite(expected))
                    or (
                        not exact
                        and not bounded_last_place
                        and not trusted_solver_core
                    )
                ):
                    detail = "shape_or_nonfinite"
                    if (
                        expected.shape == shape
                        and actual.shape == shape
                        and np.all(np.isfinite(expected))
                        and np.all(np.isfinite(actual))
                    ):
                        differing = np.flatnonzero(actual.ravel() != expected.ravel())
                        maximum = float(np.max(np.abs(actual - expected)))
                        scale = np.maximum(np.abs(actual), np.abs(expected))
                        ratios = np.abs(actual - expected) / np.spacing(scale)
                        maximum_ulps = float(np.max(ratios))
                        first = int(differing[0])
                        detail = (
                            f"different_values={differing.size}, "
                            f"max_abs_difference={maximum:.17g}, "
                            f"max_ulp_distance={maximum_ulps:.17g}, "
                            f"first_flat_index={first}, "
                            f"first_actual={actual.ravel()[first]:.17g}, "
                            f"first_expected={expected.ravel()[first]:.17g}"
                        )
                    raise ValueError(
                        "qualified Q4 accepted algorithmic origin does not "
                        f"reproduce committed {name} for element "
                        f"{self.element_id} ({detail})"
                    )
        # Scalar/generalized evaluations retain richer deterministic recovery
        # fields than the compact batch state.  When present, each such field
        # is part of the accepted constitutive result and must match the replay
        # exactly; otherwise a valid core plus stale stress/recovery arrays
        # could be blessed by a newly computed outer hash.
        if isinstance(candidate, Mapping):
            replayable = (
                "layer_strain_material",
                "kinematic_layer_strain",
                "layer_stress",
                "layer_stress_material",
                "equivalent_stress_measure",
                "generalized_section",
                "geometric_nonlinearity",
                "membrane_strain",
                "curvature",
                "transverse_shear_strain",
                "membrane_resultants",
                "bending_resultants",
                "transverse_shear_resultants",
                "membrane_resultant_order",
                "transverse_shear_resultant_order",
                "recovery_scope",
            )
            for name in replayable:
                if name not in committed_state:
                    continue
                if name not in candidate:
                    raise ValueError(
                        "qualified Q4 accepted response replay omitted committed "
                        f"field {name!r}"
                    )
                stored = committed_state[name]
                reproduced = candidate[name]
                if isinstance(stored, np.ndarray) or isinstance(
                    reproduced, np.ndarray
                ):
                    try:
                        stored_array = np.asarray(stored)
                        reproduced_array = np.asarray(reproduced)
                        equal = bool(
                            stored_array.shape == reproduced_array.shape
                            and (
                                not np.issubdtype(stored_array.dtype, np.number)
                                or (
                                    np.all(np.isfinite(stored_array))
                                    and np.all(np.isfinite(reproduced_array))
                                )
                            )
                            and np.array_equal(stored_array, reproduced_array)
                        )
                    except (TypeError, ValueError):
                        equal = False
                else:
                    equal = canonical_json_bytes(stored) == canonical_json_bytes(
                        reproduced
                    )
                if not equal:
                    raise ValueError(
                        "qualified Q4 accepted response replay disagrees with "
                        f"committed field {name!r}"
                    )
        return inherited

    def _canonicalize_compact_batch_constitutive_core(
        self,
        material: Any,
        state: Dict[str, Any],
        replayed: Any,
    ) -> None:
        """Replace an admitted compact batch core with its scalar replay.

        Packed batch assembly and scalar finalization can reach the same
        constitutive update through different binary64 contraction orders.
        The bounded replay check above proves that the compact state is the
        admitted update.  Canonicalizing its three constitutive-core arrays at
        the one-time sealing boundary makes every later replay byte-exact and
        prevents rounding drift from accumulating across restart boundaries.
        Rich/generalized states are deliberately excluded and remain exact.
        """

        compact_batch_state = self._is_compact_batch_constitutive_state(
            material,
            state,
        )
        if not compact_batch_state:
            return
        try:
            candidate = replayed[2]
        except (IndexError, TypeError) as exc:
            raise ValueError(
                "qualified Q4 compact replay produced no canonical state"
            ) from exc
        if not isinstance(candidate, Mapping):
            raise ValueError(
                "qualified Q4 compact replay produced no canonical state"
            )
        for name in ("plastic_strain", "alpha", "layer_strain"):
            values = np.asarray(candidate.get(name, ()), dtype=np.float64)
            if values.shape != np.asarray(state.get(name, ())).shape:
                raise ValueError(
                    "qualified Q4 compact replay canonical state has an "
                    f"incompatible {name} shape"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "qualified Q4 compact replay canonical state has "
                    f"nonfinite {name}"
                )
            state[name] = values.copy()

    def _stable_state_identity_payload(
        self,
        mesh: Any,
        material: Any,
        num_layers: int,
    ) -> Dict[str, Any]:
        """Return stable element/material identity shared by state dispositions."""

        coordinates = np.asarray(self.get_node_coordinates(mesh), dtype=np.float64)
        element_descriptor = self.to_dict()
        material_descriptor = resolved_material_descriptor(material)
        return {
            "formulation_id": FORMULATION_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "element_id": int(self.element_id),
            "node_ids": [int(value) for value in self.node_ids],
            "num_layers": int(num_layers),
            "reference_coordinates_sha256": _binary64_vector_fingerprint(
                coordinates,
                (4, 3),
                "reference_coordinates",
            ),
            "element_configuration_sha256": canonical_sha256(
                {
                    "layout": "Q4_COMPLETE_STABLE_ELEMENT_CONFIGURATION_V1",
                    "descriptor": element_descriptor,
                }
            ),
            "material_or_section_sha256": canonical_sha256(
                {
                    "layout": "Q4_RESOLVED_MATERIAL_AND_SECTION_V1",
                    "material": material_descriptor,
                    "section": element_descriptor.get("shell_section"),
                }
            ),
        }

    def _committed_current_binding_payload(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: np.ndarray,
        state: Mapping[str, Any],
        num_layers: int,
    ) -> Dict[str, Any]:
        identity = self._stable_state_identity_payload(
            mesh,
            material,
            num_layers,
        )
        return {
            "schema_id": Q4_CURRENT_STATE_BINDING_SCHEMA_ID,
            "algorithmic_origin_schema_id": (
                Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
            ),
            **identity,
            "decomposition_policy_id": (
                Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
            ),
            "projection_policy_id": Q4_CURRENT_STATE_PROJECTION_POLICY_ID,
            "committed_total_u": committed_u_elem.tolist(),
            "committed_total_u_sha256": _binary64_vector_fingerprint(
                committed_u_elem,
                (24,),
                "committed_total_u",
            ),
            "state_payload_sha256": canonical_sha256(_q4_state_payload(state)),
            "accepted_algorithmic_update_sha256": (
                self._accepted_algorithmic_update_fingerprint(
                    material,
                    state,
                    num_layers,
                )
            ),
        }

    def _deleted_frozen_disposition_payload(
        self,
        mesh: Any,
        material: Any,
        accepted_local_u: np.ndarray,
        num_layers: int,
        *,
        deletion_step_index: int,
        deletion_load_factor: float,
        residual_stiffness_fraction: float,
        trigger_name: str,
    ) -> Dict[str, Any]:
        step = int(deletion_step_index)
        load_factor = float(deletion_load_factor)
        residual = float(residual_stiffness_fraction)
        trigger = str(trigger_name)
        if step <= 0 or not math.isfinite(load_factor):
            raise ValueError("qualified Q4 deletion coordinates are invalid")
        if not math.isfinite(residual) or not 0.0 <= residual <= 1.0:
            raise ValueError(
                "qualified Q4 deletion residual stiffness fraction is invalid"
            )
        if not trigger:
            raise ValueError("qualified Q4 deletion trigger must not be empty")
        displacement = np.asarray(accepted_local_u, dtype=np.float64)
        if displacement.shape != (24,) or not np.all(np.isfinite(displacement)):
            raise ValueError(
                "qualified Q4 deletion disposition requires a finite 24-vector"
            )
        payload = {
            "schema_id": Q4_ACTIVITY_DISPOSITION_SCHEMA_ID,
            "policy_id": Q4_DELETED_FROZEN_POLICY_ID,
            "status": "DELETED_FROZEN_NONCURRENT",
            **self._stable_state_identity_payload(mesh, material, num_layers),
            "deletion_step_index": step,
            "deletion_load_factor": load_factor,
            "accepted_local_u": displacement.tolist(),
            "accepted_local_u_sha256": _binary64_vector_fingerprint(
                displacement,
                (24,),
                "deleted_accepted_local_u",
            ),
            "residual_stiffness_fraction": residual,
            "trigger_name": trigger,
            "constitutive_history_semantics": (
                "FROZEN_AT_DELETION_ACCEPTED_STATE"
            ),
            "residual_operator_semantics": (
                "CURRENT_CONFIGURATION_FORCE_AND_TANGENT_SCALED_"
                "WITHOUT_CONSTITUTIVE_STATE_UPDATE"
            ),
            "operator_semantics": (
                "CONSTITUTIVE_HISTORY_FROZEN;"
                "FORCE_AND_TANGENT_REEVALUATED_AT_CURRENT_U_THEN_SCALED"
            ),
        }
        payload["disposition_sha256"] = canonical_sha256(payload)
        return payload

    def _failed_state_disposition_payload(
        self,
        mesh: Any,
        material: Any,
        failed_local_u: np.ndarray,
        state: Mapping[str, Any],
        num_layers: int,
        *,
        failure_reason: str,
    ) -> Dict[str, Any]:
        displacement = np.asarray(failed_local_u, dtype=np.float64)
        reason = str(failure_reason)
        if displacement.shape != (24,) or not np.all(np.isfinite(displacement)):
            raise ValueError(
                "qualified Q4 failed disposition requires a finite 24-vector"
            )
        if not reason:
            raise ValueError(
                "qualified Q4 failed disposition requires a failure reason"
            )
        payload = {
            "schema_id": Q4_ACTIVITY_DISPOSITION_SCHEMA_ID,
            "policy_id": Q4_FAILED_STATE_POLICY_ID,
            "status": "FAILED_NONAUTHORITATIVE",
            **self._stable_state_identity_payload(mesh, material, num_layers),
            "failed_local_u": displacement.tolist(),
            "failed_local_u_sha256": _binary64_vector_fingerprint(
                displacement,
                (24,),
                "failed_local_u",
            ),
            "failure_reason": reason,
            "state_payload_sha256": canonical_sha256(_q4_state_payload(state)),
            "semantics": (
                "MATERIALIZED_RESULT_ONLY_NOT_ACCEPTED_CURRENT_STATE_EVIDENCE"
            ),
        }
        payload["disposition_sha256"] = canonical_sha256(payload)
        return payload

    def _reject_noncurrent_activity_disposition(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if _FOREIGN_S3_ACTIVITY_DISPOSITION_KEY in state:
            raise ValueError(
                "qualified Q4 state carries a foreign S3 activity disposition"
            )
        disposition = state.get(_Q4_ACTIVITY_DISPOSITION_KEY)
        if disposition is not None:
            status = (
                str(disposition.get("status", "UNKNOWN"))
                if isinstance(disposition, Mapping)
                else "MALFORMED"
            )
            raise ValueError(
                "qualified Q4 state is noncurrent and cannot supply an active "
                f"current-state tangent ({status})"
            )

    def seal_committed_current_tangent_state(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
    ) -> Dict[str, Any]:
        """Bind one accepted active Q4 state without retaining matrix caches."""

        if isinstance(state, Mapping):
            self._reject_noncurrent_activity_disposition(state)
        with self._current_state_cache_transaction():
            return self._seal_committed_state_at_configuration(
                mesh,
                material,
                committed_u_elem,
                state,
                num_layers,
                allow_noncurrent=False,
            )

    def _seal_solver_finalized_current_tangent_state(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
    ) -> Dict[str, Any]:
        """Seal a state owned by the solver's accepted Newton transaction.

        This private hook is the sole authority for converting the packed
        vectorized constitutive core to the scalar canonical representation.
        Arbitrary callers continue through the public fail-closed replay path.
        """

        if isinstance(state, Mapping):
            self._reject_noncurrent_activity_disposition(state)
        with self._current_state_cache_transaction():
            return self._seal_committed_state_at_configuration(
                mesh,
                material,
                committed_u_elem,
                state,
                num_layers,
                allow_noncurrent=False,
                trusted_solver_compact_batch=True,
            )

    def _seal_committed_state_at_configuration(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        allow_noncurrent: bool,
        trusted_solver_compact_batch: bool = False,
    ) -> Dict[str, Any]:
        """Bind a finalized Q4 constitutive state to one exact configuration.

        This hook is intended for the solver's committed-result finalizer, not
        for Newton trial updates.  Keeping the changing seal outside the trial
        loop preserves packed Q4 constitutive-state storage.
        """

        if not isinstance(state, Mapping):
            raise TypeError("qualified Q4 committed state must be a mapping")
        if _FOREIGN_S3_ACTIVITY_DISPOSITION_KEY in state:
            raise ValueError(
                "qualified Q4 state carries a foreign S3 activity disposition"
            )
        if not allow_noncurrent:
            self._reject_noncurrent_activity_disposition(state)
        layers = _qualified_q4_layer_count(num_layers)
        displacement = np.asarray(committed_u_elem, dtype=np.float64)
        if displacement.shape != (24,) or not np.all(np.isfinite(displacement)):
            raise ValueError(
                "qualified Q4 committed state requires a finite 24-vector"
            )
        before = canonical_json_bytes(state)
        made = copy.deepcopy(dict(state))
        made.pop(_Q4_CURRENT_STATE_BINDING_KEY, None)
        made.pop(_Q4_CURRENT_STATE_DIGEST_KEY, None)
        made.pop("state_digest", None)
        self._validate_committed_current_kinematics(
            mesh, displacement, made, layers
        )
        # A plastic state is sealable only when its retained parent exactly
        # reproduces the accepted trial core.  This is the final-result
        # lifecycle proof that rejected line-search candidates or a second
        # return map from converged history were not substituted.
        validated_compact_core = bool(
            trusted_solver_compact_batch
            and self._is_compact_batch_constitutive_state(material, made)
        )
        replayed = self._replay_accepted_algorithmic_response(
            mesh,
            material,
            displacement,
            made,
            layers,
            return_components=False,
            validated_compact_core=validated_compact_core,
        )
        self._canonicalize_compact_batch_constitutive_core(
            material,
            made,
            replayed,
        )
        binding = self._committed_current_binding_payload(
            mesh,
            material,
            displacement,
            made,
            layers,
        )
        binding[_Q4_CURRENT_STATE_DIGEST_KEY] = canonical_sha256(binding)
        made[_Q4_CURRENT_STATE_BINDING_KEY] = binding
        made[_Q4_CURRENT_STATE_DIGEST_KEY] = binding[
            _Q4_CURRENT_STATE_DIGEST_KEY
        ]
        if canonical_json_bytes(state) != before:
            raise RuntimeError("qualified Q4 state sealing mutated its input")
        return made

    def validate_committed_current_tangent_binding(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
    ) -> str:
        """Run strict current-state identity/hash guards without mechanics."""

        if isinstance(state, Mapping):
            self._reject_noncurrent_activity_disposition(state)
        return self._validate_committed_state_at_configuration(
            mesh,
            material,
            committed_u_elem,
            state,
            num_layers,
            allow_noncurrent=False,
        )

    def validate_committed_current_tangent_semantics(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
    ) -> str:
        """Replay one already guard-valid active state transactionally."""

        _validate_q4_quadrature_authority(self)
        layers = _qualified_q4_layer_count(num_layers)
        displacement = np.asarray(committed_u_elem, dtype=np.float64)
        digest = self.validate_committed_current_tangent_binding(
            mesh,
            material,
            displacement,
            state,
            layers,
        )
        with self._current_state_cache_transaction():
            self._validate_committed_current_kinematics(
                mesh,
                displacement,
                state,
                layers,
            )
            self._replay_accepted_algorithmic_response(
                mesh,
                material,
                displacement,
                state,
                layers,
                return_components=False,
            )
        return digest

    def validate_committed_current_tangent_state(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
    ) -> str:
        """Validate both binding guards and accepted constitutive semantics."""

        return self.validate_committed_current_tangent_semantics(
            mesh,
            material,
            committed_u_elem,
            state,
            num_layers,
        )

    def _validate_committed_state_at_configuration(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        allow_noncurrent: bool,
    ) -> str:
        """Validate the closed binding only; never evaluate constitutive mechanics."""

        if not isinstance(state, Mapping):
            raise TypeError("qualified Q4 committed state must be a mapping")
        if not allow_noncurrent:
            self._reject_noncurrent_activity_disposition(state)
        layers = _qualified_q4_layer_count(num_layers)
        displacement = np.asarray(committed_u_elem, dtype=np.float64)
        if displacement.shape != (24,) or not np.all(np.isfinite(displacement)):
            raise ValueError(
                "qualified Q4 committed state requires a finite 24-vector"
            )
        binding = state.get(_Q4_CURRENT_STATE_BINDING_KEY)
        if not isinstance(binding, Mapping):
            raise ValueError(
                "qualified Q4 committed state lacks its configuration binding"
            )
        stored_displacement = np.asarray(
            binding.get("committed_total_u", ()), dtype=np.float64
        )
        if (
            stored_displacement.shape != (24,)
            or not np.all(np.isfinite(stored_displacement))
            or not np.array_equal(stored_displacement, displacement)
        ):
            raise ValueError(
                "qualified Q4 committed displacement disagrees with its state seal"
            )
        expected_payload = self._committed_current_binding_payload(
            mesh,
            material,
            displacement,
            state,
            layers,
        )
        expected_digest = canonical_sha256(expected_payload)
        expected = {
            **expected_payload,
            _Q4_CURRENT_STATE_DIGEST_KEY: expected_digest,
        }
        if canonical_json_bytes(binding) != canonical_json_bytes(expected):
            raise ValueError(
                "qualified Q4 committed state/configuration binding is incompatible"
            )
        if state.get(_Q4_CURRENT_STATE_DIGEST_KEY) != expected_digest or (
            "state_digest" in state and state.get("state_digest") != expected_digest
        ):
            raise ValueError("qualified Q4 committed state integrity hash is invalid")
        return expected_digest

    def seal_noncurrent_deleted_state(
        self,
        mesh: Any,
        material: Any,
        accepted_local_u: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        deletion_step_index: int,
        deletion_load_factor: float,
        residual_stiffness_fraction: float,
        trigger_name: str,
    ) -> Dict[str, Any]:
        """Seal frozen history at deletion time as explicitly noncurrent."""

        if not isinstance(state, Mapping):
            raise TypeError("qualified Q4 deleted state must be a mapping")
        if _FOREIGN_S3_ACTIVITY_DISPOSITION_KEY in state:
            raise ValueError(
                "qualified Q4 deleted state carries a foreign S3 disposition"
            )
        before = canonical_json_bytes(state)
        layers = _qualified_q4_layer_count(num_layers)
        displacement = np.asarray(accepted_local_u, dtype=np.float64)
        made = copy.deepcopy(dict(state))
        for key in (
            _Q4_ACTIVITY_DISPOSITION_KEY,
            _Q4_CURRENT_STATE_BINDING_KEY,
            _Q4_CURRENT_STATE_DIGEST_KEY,
            "state_digest",
        ):
            made.pop(key, None)
        made[_Q4_ACTIVITY_DISPOSITION_KEY] = (
            self._deleted_frozen_disposition_payload(
                mesh,
                material,
                displacement,
                layers,
                deletion_step_index=deletion_step_index,
                deletion_load_factor=deletion_load_factor,
                residual_stiffness_fraction=residual_stiffness_fraction,
                trigger_name=trigger_name,
            )
        )
        with self._current_state_cache_transaction():
            sealed = self._seal_committed_state_at_configuration(
                mesh,
                material,
                displacement,
                made,
                layers,
                allow_noncurrent=True,
            )
        if canonical_json_bytes(state) != before:
            raise RuntimeError("qualified Q4 deletion sealing mutated its input")
        return sealed

    def validate_noncurrent_deleted_state(
        self,
        mesh: Any,
        material: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        expected_deletion_step_index: Optional[int] = None,
        expected_deletion_load_factor: Optional[float] = None,
        expected_residual_stiffness_fraction: Optional[float] = None,
        expected_trigger_name: Optional[str] = None,
    ) -> str:
        """Validate deletion-time seal, disposition, and frozen semantics."""

        if not isinstance(state, Mapping):
            raise TypeError("qualified Q4 deleted state must be a mapping")
        if _FOREIGN_S3_ACTIVITY_DISPOSITION_KEY in state:
            raise ValueError(
                "qualified Q4 deleted state carries a foreign S3 disposition"
            )
        layers = _qualified_q4_layer_count(num_layers)
        raw = state.get(_Q4_ACTIVITY_DISPOSITION_KEY)
        if not isinstance(raw, Mapping):
            raise ValueError(
                "qualified Q4 deleted state lacks its activity disposition"
            )
        try:
            displacement = np.asarray(
                raw.get("accepted_local_u", ()), dtype=np.float64
            )
            step = int(raw.get("deletion_step_index"))
            load_factor = float(raw.get("deletion_load_factor"))
            residual = float(raw.get("residual_stiffness_fraction"))
            trigger = str(raw.get("trigger_name"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "qualified Q4 deleted disposition values are malformed"
            ) from exc
        expected = self._deleted_frozen_disposition_payload(
            mesh,
            material,
            displacement,
            layers,
            deletion_step_index=step,
            deletion_load_factor=load_factor,
            residual_stiffness_fraction=residual,
            trigger_name=trigger,
        )
        if canonical_json_bytes(raw) != canonical_json_bytes(expected):
            raise ValueError(
                "qualified Q4 deleted activity disposition is incompatible"
            )
        if (
            expected_deletion_step_index is not None
            and step != int(expected_deletion_step_index)
        ):
            raise ValueError(
                "qualified Q4 deletion step disagrees with checkpoint history"
            )
        if (
            expected_deletion_load_factor is not None
            and load_factor != float(expected_deletion_load_factor)
        ):
            raise ValueError(
                "qualified Q4 deletion load factor disagrees with checkpoint history"
            )
        if (
            expected_residual_stiffness_fraction is not None
            and residual != float(expected_residual_stiffness_fraction)
        ):
            raise ValueError(
                "qualified Q4 residual fraction disagrees with checkpoint policy"
            )
        if expected_trigger_name is not None and trigger != str(
            expected_trigger_name
        ):
            raise ValueError(
                "qualified Q4 deletion trigger disagrees with checkpoint history"
            )
        digest = self._validate_committed_state_at_configuration(
            mesh,
            material,
            displacement,
            state,
            layers,
            allow_noncurrent=True,
        )
        with self._current_state_cache_transaction():
            self._validate_committed_current_kinematics(
                mesh,
                displacement,
                state,
                layers,
            )
            self._replay_accepted_algorithmic_response(
                mesh,
                material,
                displacement,
                state,
                layers,
                return_components=False,
            )
        return digest

    def mark_noncurrent_failed_state(
        self,
        mesh: Any,
        material: Any,
        failed_local_u: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        failure_reason: str,
    ) -> Dict[str, Any]:
        """Return owned, hash-closed failure output without ACTIVE authority."""

        if not isinstance(state, Mapping):
            raise TypeError("qualified Q4 failed state must be a mapping")
        if _FOREIGN_S3_ACTIVITY_DISPOSITION_KEY in state:
            raise ValueError(
                "qualified Q4 failed state carries a foreign S3 disposition"
            )
        before = canonical_json_bytes(state)
        layers = _qualified_q4_layer_count(num_layers)
        displacement = np.asarray(failed_local_u, dtype=np.float64)
        made = copy.deepcopy(dict(state))
        for key in (
            _Q4_ACTIVITY_DISPOSITION_KEY,
            _Q4_CURRENT_STATE_BINDING_KEY,
            _Q4_CURRENT_STATE_DIGEST_KEY,
            "state_digest",
        ):
            made.pop(key, None)
        made[_Q4_ACTIVITY_DISPOSITION_KEY] = (
            self._failed_state_disposition_payload(
                mesh,
                material,
                displacement,
                made,
                layers,
                failure_reason=failure_reason,
            )
        )
        if canonical_json_bytes(state) != before:
            raise RuntimeError("qualified Q4 failure marking mutated its input")
        return made

    def validate_noncurrent_failed_state(
        self,
        mesh: Any,
        material: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
    ) -> str:
        """Validate a nonauthoritative failed-result marker without mechanics."""

        if not isinstance(state, Mapping):
            raise TypeError("qualified Q4 failed state must be a mapping")
        if _FOREIGN_S3_ACTIVITY_DISPOSITION_KEY in state:
            raise ValueError(
                "qualified Q4 failed state carries a foreign S3 disposition"
            )
        raw = state.get(_Q4_ACTIVITY_DISPOSITION_KEY)
        if not isinstance(raw, Mapping):
            raise ValueError("qualified Q4 failed state lacks its disposition")
        forbidden_active = {
            _Q4_CURRENT_STATE_BINDING_KEY,
            _Q4_CURRENT_STATE_DIGEST_KEY,
            "state_digest",
        }.intersection(state)
        if forbidden_active:
            raise ValueError(
                "qualified Q4 failed state must not carry an ACTIVE "
                "committed-state seal"
            )
        displacement = np.asarray(raw.get("failed_local_u", ()), dtype=np.float64)
        reason = str(raw.get("failure_reason", ""))
        base = copy.deepcopy(dict(state))
        base.pop(_Q4_ACTIVITY_DISPOSITION_KEY, None)
        expected = self._failed_state_disposition_payload(
            mesh,
            material,
            displacement,
            base,
            _qualified_q4_layer_count(num_layers),
            failure_reason=reason,
        )
        if canonical_json_bytes(raw) != canonical_json_bytes(expected):
            raise ValueError(
                "qualified Q4 failed activity disposition is incompatible"
            )
        return str(expected["disposition_sha256"])

    def _generalized_section_in_frame(
        self,
        local_frame: np.ndarray,
    ) -> Optional[GeneralizedShellSection]:
        """Return ABD/As in numbered axes with physical-director covariance."""

        if self.shell_section is None:
            return None
        if (
            np.any(np.asarray(self.shell_section.B, dtype=float) != 0.0)
            and self.reference_normal is None
        ):
            raise ValueError(
                "B-coupled qualified Q4 sections require an authoritative "
                "reference_normal; connectivity winding is not a physical director"
            )
        physical_frame, membrane, curvature, shear, _sign = (
            self._physical_director_context(local_frame)
        )
        physical = self.shell_section.rotated(self._material_angle(physical_frame))
        physical_abd = physical.ABD
        generalized = np.block(
            [
                [membrane, np.zeros((3, 3), dtype=float)],
                [np.zeros((3, 3), dtype=float), curvature],
            ]
        )
        numbered_abd = generalized.T @ physical_abd @ generalized
        numbered_abd = 0.5 * (numbered_abd + numbered_abd.T)
        numbered_shear = shear.T @ physical.As @ shear
        numbered_shear = 0.5 * (numbered_shear + numbered_shear.T)
        return GeneralizedShellSection(
            A=numbered_abd[:3, :3],
            B=numbered_abd[:3, 3:],
            D=numbered_abd[3:, 3:],
            As=numbered_shear,
            name=physical.name,
            mass_per_area=physical.mass_per_area,
            rotary_inertia_per_area=physical.rotary_inertia_per_area,
        )

    def _constitutive_and_drill_stiffness(
        self,
        material: Any,
        frame: np.ndarray,
        *,
        post_observation: Optional[Callable[[], None]] = None,
    ) -> tuple[np.ndarray, float]:
        runtime_guard = _require_exact_q4_runtime_authority

        def recheck() -> None:
            if post_observation is not None:
                post_observation()
            else:
                runtime_guard(
                    self,
                    context="qualified Q4 constitutive observation",
                )

        constitutive = np.zeros((8, 8), dtype=float)
        if self.shell_section is not None:
            section = self._generalized_section_in_frame(frame)
            recheck()
            assert section is not None
            constitutive[:3, :3] = section.A
            constitutive[:3, 3:6] = section.B
            constitutive[3:6, :3] = section.B.T
            constitutive[3:6, 3:6] = section.D
            constitutive[6:, 6:] = section.As
            drill_stiffness = float(section.A[2, 2])
        else:
            membrane, shear, _strain_transform, _stress_transform = _shell_material_matrices(
                material,
                self._material_angle(frame),
                recheck,
            )
            constitutive[:3, :3] = self.thickness * membrane
            constitutive[3:6, 3:6] = self.thickness**3 / 12.0 * membrane
            constitutive[6:, 6:] = (5.0 / 6.0) * self.thickness * shear
            drill_stiffness = float(self.thickness * membrane[2, 2])
        if not np.all(np.isfinite(constitutive)) or drill_stiffness <= 0.0:
            raise ValueError("E4-PL constitutive matrix must be finite with positive in-plane shear")
        return constitutive, drill_stiffness

    def _qualified_stiffness_cache_key(
        self,
        mesh: Any,
        material: Any,
        coordinates: Optional[np.ndarray] = None,
        *,
        post_observation: Optional[Callable[[], None]] = None,
    ) -> tuple[Any, ...]:
        runtime_guard = _require_exact_q4_runtime_authority

        def recheck() -> None:
            if post_observation is not None:
                post_observation()
            else:
                runtime_guard(
                    self,
                    context="qualified Q4 cache input observation",
                )

        coordinates = (
            self.get_node_coordinates(mesh)
            if coordinates is None
            else np.asarray(coordinates, dtype=float)
        )
        recheck()
        revision_reader = _guarded_observe_attribute(
            mesh,
            "revision_signature",
            default=lambda: {},
            post_observation=recheck,
        )
        revision_result = _guarded_observe_call(
            revision_reader,
            post_observation=recheck,
        )
        revisions = _guarded_owned_mapping(
            revision_result,
            label="qualified Q4 mesh revision signature",
            post_observation=recheck,
        )
        if any(type(value) is not int for value in revisions.values()):
            raise TypeError(
                "qualified Q4 mesh revision values must be exact integers"
            )
        material_fingerprint = _shell_elastic_material_cache_fingerprint(
            material,
            recheck,
        )
        section_fingerprint = _generalized_shell_section_cache_fingerprint(
            self.shell_section,
            recheck,
        )
        relative = coordinates - np.mean(coordinates, axis=0)
        return (
            id(mesh),
            id(material),
            material_fingerprint,
            int(revisions.get("geometry", 0)),
            int(revisions.get("material", 0)),
            np.ascontiguousarray(relative, dtype=float).tobytes(),
            float(self.thickness),
            float(self.drilling_stabilization),
            float(self.hourglass_stabilization),
            float(self.material_angle_deg),
            None
            if self.material_direction is None
            else tuple(np.asarray(self.material_direction, dtype=float)),
            section_fingerprint,
            None
            if self.reference_normal is None
            else tuple(np.asarray(self.reference_normal, dtype=float)),
            int(self.director_polarity),
            IMPLEMENTATION_ID,
            RECOVERY_POLICY_ID,
            DIRECTOR_POLARITY_POLICY_ID,
            DIRECTOR_REVERSAL_TRANSFORM_ID,
            float(self.pl_stabilization),
            float(self.planar_tolerance),
            self.warped_formulation,
        )

    def _bind_qualified_component_guard(
        self,
        mesh: Any,
        material: Any,
        *,
        post_observation: Optional[Callable[[], None]] = None,
    ) -> None:
        runtime_guard = _require_exact_q4_runtime_authority

        def recheck() -> None:
            if post_observation is not None:
                post_observation()
            else:
                runtime_guard(
                    self,
                    context="qualified Q4 component binding observation",
                )

        token = _guarded_observe_attribute(
            mesh,
            "_qualified_direct_state_token",
            default=None,
            post_observation=recheck,
        )
        token_value = (
            int(token[0])
            if isinstance(token, list) and len(token) == 1
            else None
        )
        guard = (
            mesh,
            int(self.__dict__.get("_qualified_plan_state_revision", 0)),
            token,
            token_value,
            material,
            _shell_elastic_material_cache_fingerprint(material, recheck),
            _generalized_shell_section_cache_fingerprint(
                self.shell_section,
                recheck,
            ),
            Q4_QUADRATURE_AUTHORITY_ID,
            self._qualified_components,
            self._qualified_cache_key,
        )
        object.__setattr__(self, "_qualified_component_guard", guard)
        _bind_q4_component_cache_provenance(self, guard, mesh, material)

    def _validate_qualified_component_cache_identity(self) -> tuple[Any, ...]:
        guard = self._qualified_component_guard
        if guard is None or type(guard) is not tuple or len(guard) != 10:
            raise RuntimeError(
                "qualified Q4 component cache lacks its exact input guard"
            )
        if (
            self._qualified_components is not guard[8]
            or self._qualified_cache_key is not guard[9]
            or type(self._qualified_cache_key) is not tuple
        ):
            raise RuntimeError(
                "qualified Q4 component cache provenance changed"
            )
        _require_q4_component_cache_provenance(
            self,
            self._qualified_components,
            self._qualified_cache_key,
            guard,
        )
        return guard

    def _validate_qualified_component_guard(
        self,
        *,
        post_observation: Optional[Callable[[], None]] = None,
    ) -> None:
        _validate_q4_quadrature_authority(self)
        guard = self._validate_qualified_component_cache_identity()
        (
            mesh,
            element_revision,
            token,
            token_value,
            material,
            material_fingerprint,
            section_fingerprint,
            quadrature_id,
            _components,
            expected_cache_key,
        ) = guard
        current_token_value = (
            int(token[0])
            if isinstance(token, list) and len(token) == 1
            else None
        )
        coordinates = self.get_node_coordinates(mesh)
        current_cache_key = self._qualified_stiffness_cache_key(
            mesh,
            material,
            coordinates,
            post_observation=post_observation,
        )
        if (
            int(self.__dict__.get("_qualified_plan_state_revision", 0))
            != int(element_revision)
            or current_token_value != token_value
            or _shell_elastic_material_cache_fingerprint(
                material,
                post_observation,
            )
            != material_fingerprint
            or _generalized_shell_section_cache_fingerprint(
                self.shell_section,
                post_observation,
            )
            != section_fingerprint
            or quadrature_id != Q4_QUADRATURE_AUTHORITY_ID
            or current_cache_key != expected_cache_key
        ):
            raise RuntimeError(
                "qualified Q4 component cache is stale for current model inputs"
            )

    def _adopt_qualified_components(
        self,
        cache_key: tuple[Any, ...],
        components: Mapping[str, Any],
        mesh: Any = None,
        material: Any = None,
    ) -> np.ndarray:
        copied = _freeze_qualified_component_cache(dict(components))
        object.__setattr__(self, "_qualified_components", copied)
        object.__setattr__(self, "_qualified_cache_key", cache_key)
        self._hourglass_stiffness_matrix = np.asarray(copied["hourglass"], dtype=float)
        self._stiffness_matrix = np.asarray(copied["total"], dtype=float)
        if mesh is None or material is None:
            object.__setattr__(self, "_qualified_component_guard", None)
            _clear_q4_component_cache_provenance(self)
        else:
            self._bind_qualified_component_guard(mesh, material)
        return self._stiffness_matrix

    def _warped_generalized_drilling_correction(
        self,
        mesh: Any,
        coordinates: np.ndarray,
    ) -> np.ndarray:
        """Replace numbered ``A66`` drilling by its physical invariant."""

        correction = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        if (
            self.shell_section is None
            or self.reference_normal is None
            or float(self.drilling_stabilization) == 0.0
        ):
            return correction
        coords = np.asarray(coordinates, dtype=float)
        for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
            shape, derivative_xi, derivative_eta = self.compute_shape_functions(
                float(xi), float(eta)
            )
            frame, derivative_x, derivative_y, determinant = (
                self._local_frame_and_derivatives(
                    coords,
                    derivative_xi,
                    derivative_eta,
                )
            )
            section = self._generalized_section_in_frame(frame)
            assert section is not None
            numbered_scale = float(section.A[2, 2])
            invariant_scale = _invariant_generalized_drilling_scale(section.A)
            delta = float(self.drilling_stabilization) * (
                invariant_scale - numbered_scale
            )
            if delta == 0.0:
                continue
            drilling = self._build_drilling_b_matrix(
                shape,
                derivative_x,
                derivative_y,
            )
            local = (drilling.T @ (delta * np.eye(1)) @ drilling) * (
                float(determinant) * float(weight)
            )
            transform = self._local_dof_transform(frame)
            correction += transform.T @ local @ transform
        correction[:] = 0.5 * (correction + correction.T)
        return correction

    def compute_stiffness_components(
        self,
        mesh: Any,
        material: Any,
        *,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> Mapping[str, Any]:
        _validate_q4_quadrature_authority(self)
        coordinates = self.get_node_coordinates(mesh)
        cache_key = self._qualified_stiffness_cache_key(
            mesh,
            material,
            coordinates,
            post_observation=_qualified_runtime_post_observation,
        )
        if (
            self._qualified_components is not None
            and self._qualified_cache_key == cache_key
        ):
            self._validate_qualified_component_cache_identity()
            self._bind_qualified_component_guard(
                mesh,
                material,
                post_observation=_qualified_runtime_post_observation,
            )
            return self._qualified_components
        frame, local, warpage = equation7_frame(coordinates)
        if warpage > self.planar_tolerance:
            if self.warped_formulation == "reject":
                raise ValueError(
                    f"E4-PL element {self.element_id} is warped by {warpage:.6e}, "
                    f"above planar_tolerance={self.planar_tolerance:.6e}"
                )
            physical = _QUALIFIED_Q4_BASE_STIFFNESS_KERNEL(self, mesh, material)
            director_drilling_correction = self._warped_generalized_drilling_correction(
                mesh,
                coordinates,
            )
            if np.any(director_drilling_correction != 0.0):
                physical = (
                    np.asarray(physical, dtype=float)
                    + director_drilling_correction
                )
                physical = 0.5 * (physical + physical.T)
            zero = np.zeros_like(physical)
            result = {
                "core": physical.copy(),
                "physical": physical.copy(),
                "pl": zero.copy(),
                "hourglass": zero.copy(),
                "numerical": zero.copy(),
                "total": physical.copy(),
                "frame": frame,
                "jacobian_centre": math.nan,
                "mixed_condensed": False,
                "legacy_fallback": False,
                "warped_direct": True,
                "warped_formulation": "varying_frame",
                "warpage_ratio": warpage,
                "director_drilling_correction": director_drilling_correction,
            }
            object.__setattr__(
                self,
                "_qualified_components",
                _freeze_qualified_component_cache(result),
            )
            object.__setattr__(self, "_qualified_cache_key", cache_key)
            self._bind_qualified_component_guard(
                mesh,
                material,
                post_observation=_qualified_runtime_post_observation,
            )
            return self._qualified_components

        c = _coefficients(local)
        determinants = [c["jc"], *(_jacobian(c, r, s)[4] for r, s in _GAUSS)]
        jacobian_scale = _local_jacobian_scale(local)
        if min(determinants) <= 1.0e-12 * jacobian_scale:
            raise ValueError(f"E4-PL element {self.element_id} has a nonpositive local Jacobian")
        constitutive, drill_stiffness = self._constitutive_and_drill_stiffness(
            material,
            frame,
            post_observation=_qualified_runtime_post_observation,
        )
        compensated = self._elastic_increment_admitted(mesh, material, None)
        stationary, coupling, gram = _stationary_blocks(
            local, c, constitutive, compensated=compensated)
        solution, stationary_solve_diagnostics = _solve_stationary_system(
            stationary,
            coupling,
        )
        core_local = (-_local_contractions.product(coupling, solution)
                      if compensated else -coupling @ solution)
        core_local = 0.5 * (core_local + core_local.T)
        centre = _centre_taylor(c)
        pl_core = (_local_contractions.product(_local_contractions.product(centre.T, gram), centre)
                   if compensated else centre.T @ gram @ centre)
        pl_local = self.pl_stabilization * drill_stiffness * pl_core
        gamma = _residual_mode(local, c)
        gamma_24 = np.zeros(24, dtype=float)
        gamma_24[5::6] = gamma
        area = 4.0 * c["jc"]
        hourglass_local = (
            2.0
            * float(self.hourglass_stabilization)
            * drill_stiffness
            * area
            * np.outer(gamma_24, gamma_24)
        )
        transform = _global_transform(frame)
        if compensated:
            product = _local_contractions.product
            core = product(product(transform, core_local), transform.T)
            pl = product(product(transform, pl_local), transform.T)
            hourglass = product(product(transform, hourglass_local), transform.T)
        else:
            core = transform @ core_local @ transform.T
            pl = transform @ pl_local @ transform.T
            hourglass = transform @ hourglass_local @ transform.T
        for matrix in (core, pl, hourglass):
            matrix[:] = 0.5 * (matrix + matrix.T)
        numerical = (_local_contractions.sum_arrays((pl, hourglass))
                     if compensated else pl + hourglass)
        total = (_local_contractions.sum_arrays((core, numerical))
                 if compensated else core + numerical)
        result = {
            "core": core,
            "physical": core,
            "pl": pl,
            "hourglass": hourglass,
            "numerical": numerical,
            "total": total,
            "frame": frame,
            "jacobian_centre": c["jc"],
            "mixed_condensed": True,
            "legacy_fallback": False,
            "warped_direct": False,
            "warped_formulation": "planar_e4_pl",
            "warpage_ratio": warpage,
            "stationary_solve_policy_id": STATIONARY_SOLVE_POLICY_ID,
            "stationary_solve_diagnostics": stationary_solve_diagnostics,
        }
        object.__setattr__(
            self,
            "_qualified_components",
            _freeze_qualified_component_cache(result),
        )
        object.__setattr__(self, "_qualified_cache_key", cache_key)
        self._bind_qualified_component_guard(
            mesh,
            material,
            post_observation=_qualified_runtime_post_observation,
        )
        return self._qualified_components

    def compute_stiffness_matrix(self, mesh: Any, material: Any) -> np.ndarray:
        components = self.compute_stiffness_components(mesh, material)
        self._hourglass_stiffness_matrix = np.asarray(components["hourglass"], dtype=float)
        self._stiffness_matrix = np.asarray(components["total"], dtype=float)
        # The component binder ran before the two derived cache assignments
        # above.  Rebind once so the canonical instance-key snapshot includes
        # those owned fields and the very next call is a true warm hit.
        self._bind_qualified_component_guard(mesh, material)
        return _q4_public_stiffness_view(self)

    def compute_mass_matrix(
        self,
        mesh: Any,
        material: Any,
        *,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> np.ndarray:
        """Evaluate the inherited consistent mass under exact Q4 authority."""

        _validate_q4_quadrature_authority(self)
        return _QUALIFIED_Q4_BASE_MASS_KERNEL(
            self,
            mesh,
            material,
            _qualified_runtime_post_observation=(
                _qualified_runtime_post_observation
            ),
        )

    def compute_geometric_stiffness_matrix(
        self,
        mesh: Any,
        material: Any,
        state: Optional[Any] = None,
    ) -> np.ndarray:
        """Evaluate reference stress stiffness under exact Q4 authority."""

        _validate_q4_quadrature_authority(self)
        return _QUALIFIED_Q4_BASE_GEOMETRIC_KERNEL(self, mesh, material, state)

    def compute_internal_forces(
        self,
        mesh: Any,
        displacements: np.ndarray,
        material: Any,
    ) -> np.ndarray:
        """Return the qualified linear internal force for local or global input."""

        vector = self._get_element_displacements(mesh, displacements)
        stiffness = self.compute_stiffness_matrix(mesh, material)
        if self._elastic_increment_admitted(mesh, material, None):
            return _local_contractions.product(stiffness, vector)
        return stiffness @ vector

    def _qualified_linear_correction(
        self,
        mesh: Any,
        material: Any,
        num_layers: int,
        *,
        post_observation: Optional[Callable[[], None]] = None,
    ) -> np.ndarray:
        """Difference between the qualified and inherited elastic tangents.

        The mature ``ShellElement`` nonlinear implementation supplies the
        geometric, material-state and generalized-section increments.  Its
        zero-displacement tangent is the legacy elastic shell, however.  On a
        planar element the constant correction below replaces that baseline
        with E4-PL without disturbing the established nonlinear/state
        algorithms.

        Warped elements without a physical director authority deliberately
        retain the established varying-frame nonlinear mechanics byte for
        byte.  An authoritative generalized section instead receives the
        complete physical-director elastic baseline: the inherited nonlinear
        increments already call ``_generalized_section_in_frame``, while this
        delta removes the numbered zero-state baseline (including its A66
        drill) and installs the covariant varying-frame tangent.

        This remains the RAW inherited-kernel correction, including when the
        eligible direct route reports it as diagnostic component metadata.
        That route replaces the total response; it must not apply this delta
        to its independently compensated scalar baseline a second time.
        """

        coordinates = self.get_node_coordinates(mesh)
        _frame, _local, warpage = equation7_frame(coordinates)
        if warpage > self.planar_tolerance and (
            self.shell_section is None or self.reference_normal is None
        ):
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        # The correction is an elastic tangent delta and is independent of
        # the caller's through-thickness integration count.  Use a fixed valid
        # Lobatto rule for the inherited zero-state nonlinear tangent: this
        # preserves the generalized-section baseline while permitting plan
        # cache bookkeeping probes to use arbitrary layer identifiers.
        self._hourglass_stiffness_matrix = None
        _force, legacy, _state = _QUALIFIED_Q4_BASE_NONLINEAR_KERNEL(
            self,
            mesh,
            material,
            np.zeros(self.total_dofs, dtype=float),
            None,
            5,
            True,
            _qualified_runtime_post_observation=post_observation,
        )
        if legacy is None:
            raise RuntimeError("ShellElement returned no zero-state tangent")
        qualified = np.asarray(self.compute_stiffness_matrix(mesh, material), dtype=float)
        return qualified - np.asarray(legacy, dtype=float)

    def compute_nonlinear_response(
        self,
        mesh: Any,
        material: Any,
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
        *,
        _return_tangent_components: bool = False,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> Any:
        """Use mature nonlinear/state mechanics with the qualified baseline.

        This additive construction is exact at zero displacement and retains
        the existing von-Karman, plasticity, orthotropy, initial-field and
        generalized-section increments.  Numerical PL/hourglass contributions
        remain a constant separately recoverable part of the tangent.
        """

        return self._nonlinear_response_core(
            mesh, material, u_elem, state, num_layers, tangent,
            _return_tangent_components=_return_tangent_components,
            post_observation=_qualified_runtime_post_observation,
        )

    def _defining_class_input_snapshot_matches(
        self: Any,
        mesh: Any,
        material: Any,
        snapshot: _Q4FastInputSnapshot,
    ) -> bool:
        """Compare only raw owned builtin state; never invoke a descriptor."""

        if type(self) is not __class__:
            raise ValueError('qualified Q4 defining class identity changed')
        element = self
        element_keys = snapshot.element_namespace
        nodes_keys = snapshot.nodes_namespace
        material_keys = snapshot.material_namespace

        if (
            vars(np).get("float64") is not _Q4_FAST_COORDINATE_FLOAT64_TYPE
            or type(element) is not __class__
            or type(mesh) is not _Q4_FAST_MESH_TYPE
            or type(material) is not _Q4_FAST_MATERIAL_TYPE
            or type(snapshot.nodes) is not _Q4_FAST_STATE_MAPPING_TYPE
            or "get"
            in type.__getattribute__(
                _Q4_FAST_STATE_MAPPING_TYPE,
                "__dict__",
            )
            or type(snapshot.token) is not _Q4_FAST_TOKEN_TYPE
            or mesh is not snapshot.mesh
            or material is not snapshot.material
            or object.__getattribute__(element, "__dict__")
            is not snapshot.element_namespace
            or object.__getattribute__(mesh, "__dict__") is not snapshot.mesh_namespace
            or object.__getattribute__(material, "__dict__")
            is not snapshot.material_namespace
            or len(element_keys) != len(snapshot.element_namespace_keys)
            or not all(
                map(
                    _operator_is,
                    element_keys,
                    snapshot.element_namespace_keys,
                )
            )
            or any(
                name in snapshot.mesh_namespace
                for name in _Q4_FAST_FORBIDDEN_MESH_SHADOWS
            )
            or len(material_keys) != len(snapshot.material_namespace_keys)
            or not all(
                map(
                    _operator_is,
                    material_keys,
                    snapshot.material_namespace_keys,
                )
            )
            or dict.get(snapshot.mesh_namespace, "nodes") is not snapshot.nodes
            or dict.get(snapshot.mesh_namespace, "_qualified_direct_state_token")
            is not snapshot.token
            or len(snapshot.token) != 1
            or type(snapshot.token[0]) is not int
            or int(snapshot.token[0]) != snapshot.token_value
            or object.__getattribute__(snapshot.nodes, "__dict__")
            is not snapshot.nodes_namespace
            or len(nodes_keys) != len(snapshot.nodes_namespace_keys)
            or not all(
                map(
                    _operator_is,
                    nodes_keys,
                    snapshot.nodes_namespace_keys,
                )
            )
            or dict.get(snapshot.nodes_namespace, "_qualified_token")
            is not snapshot.token
            or dict.get(snapshot.nodes_namespace, "_qualified_kind") != "node"
        ):
            return False
        try:
            current_element_values = _Q4_FAST_ELEMENT_ITEMGETTER(
                snapshot.element_namespace
            )
        except KeyError:
            return False
        if not all(
            map(
                _operator_is,
                current_element_values,
                snapshot.element_values,
            )
        ):
            return False
        for node_id, node, namespace, value_types, values in snapshot.nodes_snapshot:
            if (
                type(node) is not _Q4_FAST_NODE_TYPE
                or object.__getattribute__(node, "__dict__") is not namespace
                or dict.get(snapshot.nodes, node_id) is not node
            ):
                return False
            try:
                current_values = _Q4_FAST_NODE_ITEMGETTER(namespace)
            except KeyError:
                return False
            if (
                len(current_values) != len(values)
                or not all(map(_operator_is, current_values, values))
            ):
                return False
        try:
            current_material_values = _Q4_FAST_MATERIAL_ITEMGETTER(
                snapshot.material_namespace
            )
        except KeyError:
            return False
        if not all(
            map(
                _operator_is,
                current_material_values,
                snapshot.material_values,
            )
        ):
            return False
        return True

    def _elastic_increment_admitted(self, mesh, material, state):
        """Narrow builtin planar elastic route; no state or result cache."""
        if type(self) is not __class__:
            raise ValueError('qualified Q4 defining class identity changed')
        if (
                type(mesh) is not _Q4_FAST_MESH_TYPE or
                type(material) is not _Q4_FAST_MATERIAL_TYPE or
                self.shell_section is not None or self.reduced_integration or
                material.hardening_curve is not None or material.yield_stress != 0.0):
            return False
        if state is not None and (
                type(state) is not dict or
                any(key in state for key in _Q4_INITIAL_STATE_KEYS)):
            return False
        _require_q4_fast_base_authority()
        snapshot = _capture_q4_fast_input_snapshot(self, mesh, material)
        if snapshot is None:
            return False
        _frame, _local, warpage = equation7_frame(self.get_node_coordinates(mesh))
        if not self._defining_class_input_snapshot_matches(mesh, material, snapshot):
            raise RuntimeError("qualified Q4 elastic increment inputs changed")
        return bool(warpage <= self.planar_tolerance)

    def _elastic_increment_response(
        self, mesh, material, vector, state, tangent, *, batch_linear=None,
        batch_baseline=False,
        _qualified_runtime_post_observation=None,
    ):
        """Replace cancelled elastic output sums, preserving the batch offset.

        The caller first runs its existing state/constitutive path. Preparation
        reuses immutable batch baselines only while original input bindings hold. The batch
        baseline is not presumed identical to the inherited scalar tangent.
        """
        # Keep the ORIGINAL call input binding across admission and every
        # preparation observation, including the batch caller (no callback).
        # Never replace this snapshot with a newer post-preparation snapshot.
        snapshot = _capture_q4_fast_input_snapshot(self, mesh, material)
        if snapshot is None or not self._elastic_increment_admitted(mesh, material, state):
            return None
        geometry = self._nonlinear_geometry(mesh)
        qualified = np.asarray(self.compute_stiffness_matrix(mesh, material), dtype=float)
        if not self._defining_class_input_snapshot_matches(mesh, material, snapshot):
            raise RuntimeError("qualified Q4 elastic increment inputs changed during preparation")
        baseline = qualified
        publish_baseline = None
        prepared_batch = (batch_baseline and batch_linear is None
                          and type(snapshot) is _Q4FastInputSnapshot)
        if prepared_batch:
            prepared = _prepare_elastic_batch_baseline(
                self, mesh, material, snapshot, _qualified_runtime_post_observation)
            prepared_batch = prepared is not None
            if prepared_batch:
                baseline, publish_baseline = prepared
        if batch_baseline and not prepared_batch:
            if batch_linear is not None:
                raise ValueError("ambiguous qualified Q4 batch baseline")
            batch_linear = _elastic_increment_module.compensated_elastic_batch_zero(
                geometry, float(material.elastic_modulus), float(material.poisson_ratio),
                float(material.shear_modulus), self.thickness, self.drilling_stabilization)
        if batch_linear is not None:
            self._hourglass_stiffness_matrix = None
            scalar_linear = _elements_module._compensated_elastic_zero_tangent(
                self, mesh, material, _qualified_runtime_post_observation)
            offset = _local_contractions.sum_arrays((batch_linear, -scalar_linear))
            baseline = _local_contractions.sum_arrays((qualified, offset))
        increment, material_increment, geometric = (
            _elastic_increment_module.elastic_shell_nonlinear_increment(
                vector, geometry, float(material.elastic_modulus),
                float(material.poisson_ratio), self.thickness, tangent=tangent,
            )
        )
        force = _local_contractions.sum_arrays((
            _local_contractions.product(baseline, vector), increment))
        material_tangent = (_local_contractions.sum_arrays((baseline, material_increment))
                            if tangent else None)
        total = (_local_contractions.sum_arrays((material_tangent, geometric))
                 if tangent else None)
        if _qualified_runtime_post_observation is not None:
            _qualified_runtime_post_observation()
        if not self._defining_class_input_snapshot_matches(mesh, material, snapshot):
            raise RuntimeError("qualified Q4 elastic increment inputs changed")
        self._validate_qualified_component_guard(
            post_observation=_qualified_runtime_post_observation)
        if publish_baseline is not None:
            publish_baseline()
        return force, total, material_tangent, geometric

    def _prepare_nonlinear_linear_correction(
        self, mesh: Any, material: Any, *,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> Optional[_PreparedQ4LinearCorrection]:
        """Prepare only exact builtin, nonyielding scalar-section Q4 inputs."""
        if type(self) is not __class__:
            raise ValueError('qualified Q4 defining class identity changed')
        if (
                self.shell_section is not None or
                type(mesh) is not _Q4_FAST_MESH_TYPE or
                type(material) is not _Q4_FAST_MATERIAL_TYPE or
                material.hardening_curve is not None or material.yield_stress != 0.):
            return None
        if _capture_q4_fast_input_snapshot(self, mesh, material) is None:
            return None
        return _prepare_q4_correction_payload(
            self, mesh, material, _qualified_runtime_post_observation)

    def _compute_prepared_nonlinear_response(
        self, mesh: Any, material: Any, u_elem: np.ndarray,
        state: Optional[Any], num_layers: int, tangent: bool, *,
        prepared: _PreparedQ4LinearCorrection,
        _return_tangent_components: bool = False,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> Any:
        """Same nonlinear update with an originally bound elastic correction."""
        def require_preparation() -> bytes:
            if type(prepared) is not _PreparedQ4LinearCorrection:
                raise TypeError("qualified Q4 correction preparation required")
            binding = _lookup_prepared_q4_correction(prepared)
            if (binding is None or binding[0] is not self or
                    binding[1] is not mesh or binding[2] is not material or
                    self._qualified_component_guard is not binding[3] or
                    not self._defining_class_input_snapshot_matches(mesh, material, binding[4])):
                raise RuntimeError("qualified Q4 correction preparation is stale")
            self._validate_qualified_component_guard(
                post_observation=_qualified_runtime_post_observation)
            return binding[5]

        correction = np.frombuffer(require_preparation(), dtype=float).reshape(24, 24)
        result = self._nonlinear_response_core(
            mesh, material, u_elem, state, num_layers, tangent,
            _return_tangent_components=_return_tangent_components,
            post_observation=_qualified_runtime_post_observation,
            prepared_correction=correction)
        # User-supplied arrays/state may execute observation callbacks. Recheck
        # the ORIGINAL binding as well as the ordinary runtime boundary.
        require_preparation()
        return result

    def _nonlinear_response_core(
        self, mesh, material, u_elem, state, num_layers, tangent, *,
        _return_tangent_components=False, post_observation=None,
        prepared_correction=None,
    ):
        _qualified_runtime_post_observation = post_observation

        _validate_q4_quadrature_authority(self)
        layers = _qualified_q4_layer_count(num_layers)
        coordinates_before = np.array(self.get_node_coordinates(mesh), dtype=float, copy=True)
        observed_vector = np.asarray(u_elem, dtype=float)
        if _qualified_runtime_post_observation is not None:
            _qualified_runtime_post_observation()
        if not np.array_equal(
            np.asarray(self.get_node_coordinates(mesh), dtype=float), coordinates_before
        ):
            raise RuntimeError(
                "qualified Q4 node coordinates changed during input observation"
            )
        vector = observed_vector.reshape(self.total_dofs)
        direct_state_admissible = state is None or type(state) is dict
        if state is not None:
            if not isinstance(state, Mapping):
                raise TypeError("qualified Q4 nonlinear state must be a mapping")
            state = _guarded_owned_plain_value(
                state,
                label="qualified Q4 nonlinear state",
                post_observation=_qualified_runtime_post_observation,
            )
        self._hourglass_stiffness_matrix = None
        inherited = self._assumed_strain_kernel(
            mesh, material, vector, state, layers, tangent,
            return_components=_return_tangent_components,
        )
        assumed_route = inherited is not None
        if inherited is None:
            inherited = _QUALIFIED_Q4_BASE_NONLINEAR_KERNEL(
                self,
                mesh,
                material,
                vector,
                state,
                layers,
                tangent,
                _return_tangent_components=_return_tangent_components,
                _qualified_runtime_post_observation=(
                    _qualified_runtime_post_observation
                ),
            )
        if _return_tangent_components:
            force, inherited_tangent, trial_state, inherited_components = inherited
        else:
            force, inherited_tangent, trial_state = inherited
        if not isinstance(trial_state, Mapping):
            raise TypeError("qualified Q4 nonlinear trial state must be a mapping")
        trial_state = self.attach_current_tangent_algorithmic_origin(
            material,
            state,
            trial_state,
            layers,
            tangent_evaluated=bool(tangent),
        )
        correction = prepared_correction if prepared_correction is not None else self._qualified_linear_correction(
            mesh,
            material,
            layers,
            post_observation=_qualified_runtime_post_observation,
        )
        direct = (self._elastic_increment_response(
            mesh, material, vector, state, tangent,
        ) if direct_state_admissible and not assumed_route else None)
        force = (np.asarray(force, dtype=float) + correction @ vector
                 if direct is None else direct[0])
        if not tangent:
            return force, None, trial_state
        if inherited_tangent is None:
            raise RuntimeError("ShellElement returned no tangent with tangent=True")
        total = (np.asarray(inherited_tangent, dtype=float) + correction
                 if direct is None else direct[1])
        if _return_tangent_components:
            material_tangent = np.asarray(
                inherited_components["material"], dtype=np.float64
            ) + correction
            geometric_tangent = np.asarray(
                inherited_components["geometric"], dtype=np.float64
            )
            if direct is not None:
                material_tangent, geometric_tangent = direct[2:]
            return force, total, trial_state, {
                "material": material_tangent,
                "geometric": geometric_tangent,
                "qualified_linear_material_correction": correction.copy(),
            }
        return force, total, trial_state

    def compute_committed_current_tangent_components(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: Any,
        committed_state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        native_rotation_trial: Any = None,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> Mapping[str, Any]:
        """Evaluate committed components without retaining mutable caches."""

        observed_displacement = np.asarray(committed_u_elem, dtype=np.float64)
        if _qualified_runtime_post_observation is not None:
            _qualified_runtime_post_observation()
        if not isinstance(committed_state, Mapping):
            raise TypeError("qualified Q4 committed state must be a mapping")
        observed_state_items = tuple(committed_state.items())
        if _qualified_runtime_post_observation is not None:
            _qualified_runtime_post_observation()
        owned_state = dict(observed_state_items)
        with self._current_state_cache_transaction():
            return self._compute_committed_current_tangent_components_unchecked(
                mesh,
                material,
                observed_displacement,
                owned_state,
                num_layers,
                native_rotation_trial=native_rotation_trial,
            )

    def _compute_committed_current_tangent_components_unchecked(
        self,
        mesh: Any,
        material: Any,
        committed_u_elem: Any,
        committed_state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        native_rotation_trial: Any = None,
    ) -> Mapping[str, Any]:
        """Return the qualified Q4 additive-von-Karman tangent split.

        Q4 retains its frozen additive-rotation von-Karman mechanics; it is not
        reinterpreted as the S3 multiplicative/director total-Lagrangian model.
        The material and stress-Hessian matrices are integrated independently
        by the production nonlinear kernel.  The input must be a finalized,
        configuration-sealed state and remains read-only throughout evaluation.
        """

        if native_rotation_trial is not None:
            raise TypeError(
                "qualified Q4 committed tangent uses additive von-Karman "
                "coordinates and does not accept a native rotation trial"
            )
        displacement = np.asarray(committed_u_elem, dtype=np.float64)
        if displacement.shape != (24,) or not np.all(np.isfinite(displacement)):
            raise ValueError(
                "qualified Q4 committed current tangent requires a finite 24-vector"
            )
        layers = _qualified_q4_layer_count(num_layers)
        before = canonical_json_bytes(committed_state)
        state_digest = self.validate_committed_current_tangent_state(
            mesh,
            material,
            displacement,
            committed_state,
            layers,
        )
        force, inherited_total, _candidate_state, inherited_components = (
            self._replay_accepted_algorithmic_response(
                mesh,
                material,
                displacement,
                committed_state,
                layers,
                return_components=True,
            )
        )
        if inherited_total is None:
            raise RuntimeError("qualified Q4 committed current tangent is missing")
        correction = self._qualified_linear_correction(mesh, material, layers)
        force = np.asarray(force, dtype=np.float64) + correction @ displacement
        actual_total = np.asarray(inherited_total, dtype=np.float64) + correction
        components = {
            "material": np.asarray(
                inherited_components["material"], dtype=np.float64
            )
            + correction,
            "geometric": np.asarray(
                inherited_components["geometric"], dtype=np.float64
            ),
            "qualified_linear_material_correction": correction.copy(),
        }
        direct = None if _q4_assumed_strain.elastic_route(
            self, mesh, material
        ) else self._elastic_increment_response(
            mesh, material, displacement, committed_state, True)
        if direct is not None:
            force, actual_total, components["material"], components["geometric"] = direct
        if canonical_json_bytes(committed_state) != before:
            raise RuntimeError(
                "qualified Q4 committed tangent mutated its input state"
            )
        material_tangent = np.asarray(
            components["material"], dtype=np.float64
        )
        geometric_tangent = np.asarray(
            components["geometric"], dtype=np.float64
        )
        total_tangent = np.asarray(actual_total, dtype=np.float64)
        matrices = (material_tangent, geometric_tangent, total_tangent)
        if any(
            matrix.shape != (24, 24) or not np.all(np.isfinite(matrix))
            for matrix in matrices
        ) or (
            np.asarray(force, dtype=np.float64).shape != (24,)
            or not np.all(np.isfinite(np.asarray(force, dtype=np.float64)))
        ):
            raise ValueError(
                "qualified Q4 committed tangent components are incompatible"
            )
        scale = max(float(np.linalg.norm(total_tangent, ord="fro")), 1.0)
        relative_decomposition_error = float(
            np.linalg.norm(
                total_tangent - material_tangent - geometric_tangent,
                ord="fro",
            )
            / scale
        )
        relative_symmetry_error = max(
            float(
                np.linalg.norm(matrix - matrix.T, ord="fro")
                / max(float(np.linalg.norm(matrix, ord="fro")), 1.0)
            )
            for matrix in matrices
        )
        if (
            not math.isfinite(relative_decomposition_error)
            or not math.isfinite(relative_symmetry_error)
            or relative_decomposition_error > 512.0 * np.finfo(np.float64).eps
            or relative_symmetry_error > 512.0 * np.finfo(np.float64).eps
        ):
            raise ValueError(
                "qualified Q4 committed tangent violates its production "
                "decomposition or symmetry bound"
            )
        readonly: Dict[str, Any] = {
            "state_digest": state_digest,
            "state_binding_verified": True,
            "state_storage": (
                "sealed_accepted_algorithmic_origin_plus_transient_components"
            ),
            "algorithmic_origin_schema_id": (
                Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID
            ),
            "algorithmic_origin_verified": True,
            "decomposition_policy_id": (
                Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
            ),
            "projection_policy_id": Q4_CURRENT_STATE_PROJECTION_POLICY_ID,
            "geometric_sign_convention": (
                "internal_tension_positive_membrane_resultant_hessian"
            ),
            "relative_decomposition_error": relative_decomposition_error,
            "relative_symmetry_error": relative_symmetry_error,
            "force": np.asarray(force, dtype=np.float64).copy(),
            "material": material_tangent.copy(),
            "geometric": geometric_tangent.copy(),
            "total": total_tangent.copy(),
            "qualified_linear_material_correction": np.asarray(
                components["qualified_linear_material_correction"],
                dtype=np.float64,
            ).copy(),
        }
        for name in (
            "force",
            "material",
            "geometric",
            "total",
            "qualified_linear_material_correction",
        ):
            readonly[name].setflags(write=False)
        return MappingProxyType(readonly)

    def _recover_planar_mixed_fields(
        self,
        mesh: Any,
        displacements: np.ndarray,
        material: Any,
        natural_points: Sequence[Sequence[float]],
        *,
        post_observation: Optional[Callable[[], None]] = None,
    ) -> Dict[str, np.ndarray]:
        """Evaluate the planar mixed fields at arbitrary natural coordinates.

        This private entry point is intentionally point-agnostic so bounded
        research checks can compare Q4 and S3 resultants at common physical
        locations. Public recovery defaults to the established four Gauss
        records and optionally admits explicit bounded planar homogeneous points.
        """

        element_displacements = self._get_element_displacements(mesh, displacements)
        if not np.all(np.isfinite(element_displacements)):
            raise ValueError("qualified Q4 recovery requires finite displacements")
        points = np.asarray(tuple(tuple(point) for point in natural_points), dtype=float)
        if points.size == 0:
            points = np.empty((0, 2), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("qualified Q4 recovery points must be a finite Nx2 array")

        coordinates = self.get_node_coordinates(mesh)
        frame, local, warpage = equation7_frame(coordinates)
        if warpage > self.planar_tolerance:
            raise ValueError("planar mixed recovery is unavailable for a warped-direct Q4")
        compensated = self._elastic_increment_admitted(mesh, material, None)
        transform = _global_transform(frame)
        local_displacement = (_local_contractions.product(transform.T, element_displacements)
                              if compensated else transform.T @ element_displacements)
        c = _coefficients(local)
        determinants = [c["jc"], *(_jacobian(c, r, s)[4] for r, s in _GAUSS)]
        jacobian_scale = _local_jacobian_scale(local)
        if min(determinants) <= 1.0e-12 * jacobian_scale:
            raise ValueError(f"E4-PL element {self.element_id} has a nonpositive local Jacobian")
        constitutive = self._constitutive_and_drill_stiffness(
            material,
            frame,
            post_observation=post_observation,
        )[0]
        stationary, coupling, _gram = _stationary_blocks(
            local, c, constitutive,
            compensated=compensated)
        solution, _stationary_solve_diagnostics = _solve_stationary_system(
            stationary,
            coupling,
        )
        stationary_parameters = (-_local_contractions.product(solution, local_displacement)
                                 if compensated else -solution @ local_displacement)
        stress_parameters = stationary_parameters[:14]
        strain_parameters = stationary_parameters[14:]
        stationarity_residual = (_local_contractions.sum_arrays((
            _local_contractions.product(stationary, stationary_parameters),
            _local_contractions.product(coupling.T, local_displacement))) if compensated else (
                stationary @ stationary_parameters + coupling.T @ local_displacement))
        physical_displacement = np.concatenate(
            tuple(local_displacement[6 * node : 6 * node + 5] for node in range(4))
        )

        compatible = np.zeros((len(points), 8), dtype=float)
        independent = np.zeros_like(compatible)
        resultants = np.zeros_like(compatible)
        point_determinants = np.zeros(len(points), dtype=float)
        for index, (r, s) in enumerate(points):
            determinant = _jacobian(c, float(r), float(s))[4]
            if determinant <= 1.0e-12 * jacobian_scale:
                raise ValueError(
                    f"E4-PL element {self.element_id} has a nonpositive recovery Jacobian"
                )
            n_sigma, n_epsilon = _source_fields(c, float(r), float(s))
            compatible_matrix = _compatible(local, c, float(r), float(s))
            if compensated:
                compatible[index] = _local_contractions.product(compatible_matrix, physical_displacement)
                independent[index] = _local_contractions.product(n_epsilon, strain_parameters)
                resultants[index] = _local_contractions.product(n_sigma, stress_parameters)
            else:
                compatible[index] = compatible_matrix @ physical_displacement
                independent[index] = n_epsilon @ strain_parameters
                resultants[index] = n_sigma @ stress_parameters
            point_determinants[index] = determinant

        recovered = {
            "frame": frame,
            "local_nodes": local,
            "natural_points": points,
            "local_displacement": local_displacement,
            "physical_displacement": physical_displacement,
            "constitutive": constitutive,
            "stationary_matrix": stationary,
            "stationary_coupling": coupling,
            "stationary_parameters": stationary_parameters,
            "stress_parameters": stress_parameters,
            "strain_parameters": strain_parameters,
            "stationarity_residual": stationarity_residual,
            "compatible": compatible,
            "independent": independent,
            "resultants": resultants,
            "jacobian_determinants": point_determinants,
        }
        for name, values in recovered.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"qualified Q4 mixed recovery produced non-finite field {name!r}"
                )
        return recovered

    def _recover_warped_generalized_section(
        self,
        mesh: Any,
        displacements: np.ndarray,
        *,
        return_global: bool,
    ) -> Dict[str, Any]:
        """Physicalize inherited varying-frame generalized Q4 recovery."""

        raw = ShellElement._compute_generalized_section_results(
            self,
            mesh,
            displacements,
            return_global=False,
        )
        coordinates = self.get_node_coordinates(mesh)
        center_numbered = self._center_frame(coordinates)
        center_frame, _center_membrane, _center_curvature, center_shear, center_sign = (
            self._physical_director_context(center_numbered)
        )
        membrane_strain = np.asarray(raw["membrane_strain"], dtype=float).copy()
        curvature = np.asarray(raw["curvature"], dtype=float).copy()
        membrane_resultants = np.asarray(raw["membrane_resultants"], dtype=float).copy()
        bending_resultants = np.asarray(raw["bending_resultants"], dtype=float).copy()
        transverse_shear_strain = (
            np.asarray(raw["transverse_shear_strain"], dtype=float)
            @ center_shear.T
        )
        transverse_shear_resultants = (
            np.asarray(raw["transverse_shear_resultants"], dtype=float)
            @ center_shear.T
        )
        physical_frames = np.zeros((len(self.gauss_points), 3, 3), dtype=float)
        director_signs = np.zeros(len(self.gauss_points), dtype=int)
        for index, (xi, eta) in enumerate(self.gauss_points):
            _shape, derivative_xi, derivative_eta = self.compute_shape_functions(
                float(xi), float(eta)
            )
            numbered, _dx, _dy, _determinant = self._local_frame_and_derivatives(
                coordinates,
                derivative_xi,
                derivative_eta,
            )
            physical, membrane_map, curvature_map, _shear_map, sign = (
                self._physical_director_context(numbered)
            )
            membrane_strain[index] = membrane_map @ membrane_strain[index]
            curvature[index] = curvature_map @ curvature[index]
            membrane_resultants[index] = membrane_map @ membrane_resultants[index]
            bending_resultants[index] = curvature_map @ bending_resultants[index]
            physical_frames[index] = physical
            director_signs[index] = sign

        recovered: Dict[str, Any] = dict(raw)
        recovered.update(
            {
                "membrane_strain": membrane_strain,
                "curvature": curvature,
                "transverse_shear_strain": transverse_shear_strain,
                "membrane_resultants": membrane_resultants,
                "bending_resultants": bending_resultants,
                "transverse_shear_resultants": transverse_shear_resultants,
                "implementation_id": IMPLEMENTATION_ID,
                "recovery_policy_id": RECOVERY_POLICY_ID,
                "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
                "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
                "physical_director_authoritative": True,
                "physical_director": center_frame[:, 2].copy(),
                "physical_directors": physical_frames[:, :, 2].copy(),
                "numbered_frame_director_sign": int(center_sign),
                "numbered_frame_director_signs": director_signs,
                "warped_direct": True,
            }
        )
        if return_global:
            global_membrane = np.zeros((len(self.gauss_points), 3, 3), dtype=float)
            global_bending = np.zeros_like(global_membrane)
            for index, frame in enumerate(physical_frames):
                membrane = membrane_resultants[index]
                bending = bending_resultants[index]
                membrane_tensor = np.asarray(
                    (
                        (membrane[0], membrane[2], 0.0),
                        (membrane[2], membrane[1], 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                    dtype=float,
                )
                bending_tensor = np.asarray(
                    (
                        (bending[0], bending[2], 0.0),
                        (bending[2], bending[1], 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                    dtype=float,
                )
                global_membrane[index] = frame @ membrane_tensor @ frame.T
                global_bending[index] = frame @ bending_tensor @ frame.T
            global_shear = (
                transverse_shear_resultants[:, :1] * center_frame[:, 0][None, :]
                + transverse_shear_resultants[:, 1:] * center_frame[:, 1][None, :]
            )
            recovered.update(
                {
                    "global_membrane_resultant_tensors": global_membrane,
                    "global_bending_resultant_tensors": global_bending,
                    "global_transverse_shear_resultants": global_shear,
                }
            )
        for name, values in recovered.items():
            if isinstance(values, np.ndarray) and not np.all(np.isfinite(values)):
                raise ValueError(
                    f"qualified warped Q4 recovery produced non-finite field {name!r}"
                )
        return recovered

    def _recover_warped_homogeneous_section(
        self,
        mesh: Any,
        displacements: np.ndarray,
        material: Any,
        *,
        return_global: bool,
    ) -> Dict[str, Any]:
        """Return warped homogeneous stresses in physical-director frames."""

        raw = ShellElement.compute_stresses(
            self,
            mesh,
            displacements,
            material,
            return_global=False,
        )
        coordinates = self.get_node_coordinates(mesh)
        center_numbered = self._center_frame(coordinates)
        center_frame, _center_membrane, _center_curvature, center_shear, center_sign = (
            self._physical_director_context(center_numbered)
        )
        numbered_membrane = np.column_stack(
            (raw["membrane_xx"], raw["membrane_yy"], raw["membrane_xy"])
        )
        numbered_bending = np.column_stack(
            (raw["bending_xx"], raw["bending_yy"], raw["bending_xy"])
        )
        numbered_shear = np.column_stack((raw["shear_xz"], raw["shear_yz"]))
        membrane = numbered_membrane.copy()
        bending = numbered_bending.copy()
        shear = numbered_shear @ center_shear.T
        physical_frames = np.zeros((len(self.gauss_points), 3, 3), dtype=float)
        director_signs = np.zeros(len(self.gauss_points), dtype=int)
        for index, (xi, eta) in enumerate(self.gauss_points):
            _shape, derivative_xi, derivative_eta = self.compute_shape_functions(
                float(xi), float(eta)
            )
            numbered, _dx, _dy, _determinant = self._local_frame_and_derivatives(
                coordinates,
                derivative_xi,
                derivative_eta,
            )
            physical, membrane_map, curvature_map, _shear_map, sign = (
                self._physical_director_context(numbered)
            )
            membrane[index] = membrane_map @ numbered_membrane[index]
            bending[index] = curvature_map @ numbered_bending[index]
            physical_frames[index] = physical
            director_signs[index] = sign

        recovered: Dict[str, Any] = dict(raw)
        recovered.update(
            {
                "membrane_xx": membrane[:, 0].copy(),
                "membrane_yy": membrane[:, 1].copy(),
                "membrane_xy": membrane[:, 2].copy(),
                "bending_xx": bending[:, 0].copy(),
                "bending_yy": bending[:, 1].copy(),
                "bending_xy": bending[:, 2].copy(),
                "shear_xz": shear[:, 0].copy(),
                "shear_yz": shear[:, 1].copy(),
                "implementation_id": IMPLEMENTATION_ID,
                "recovery_policy_id": RECOVERY_POLICY_ID,
                "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
                "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
                "physical_director_authoritative": True,
                "physical_director": center_frame[:, 2].copy(),
                "physical_directors": physical_frames[:, :, 2].copy(),
                "numbered_frame_director_sign": int(center_sign),
                "numbered_frame_director_signs": director_signs,
                "warped_direct": True,
            }
        )
        von_mises = np.zeros(len(self.gauss_points), dtype=float)
        equivalent = np.zeros_like(von_mises)
        utilization = np.zeros_like(von_mises)
        hill_yield = getattr(material, "hill_yield", None)
        global_shear = (
            shear[:, :1] * center_frame[:, 0][None, :]
            + shear[:, 1:] * center_frame[:, 1][None, :]
        )
        for index, frame in enumerate(physical_frames):
            tangent_shear = global_shear[index] - (
                float(global_shear[index] @ frame[:, 2]) * frame[:, 2]
            )
            local_shear = np.asarray(
                (
                    float(tangent_shear @ frame[:, 0]),
                    float(tangent_shear @ frame[:, 1]),
                ),
                dtype=float,
            )
            top = membrane[index] + bending[index]
            bottom = membrane[index] - bending[index]
            vm_top = math.sqrt(
                top[0] * top[0]
                - top[0] * top[1]
                + top[1] * top[1]
                + 3.0
                * (
                    top[2] * top[2]
                    + local_shear[0] * local_shear[0]
                    + local_shear[1] * local_shear[1]
                )
            )
            vm_bottom = math.sqrt(
                bottom[0] * bottom[0]
                - bottom[0] * bottom[1]
                + bottom[1] * bottom[1]
                + 3.0
                * (
                    bottom[2] * bottom[2]
                    + local_shear[0] * local_shear[0]
                    + local_shear[1] * local_shear[1]
                )
            )
            von_mises[index] = max(vm_top, vm_bottom)
            if hill_yield is None:
                equivalent[index] = von_mises[index]
            else:
                _membrane, _shear, _strain_to_material, stress_to_local = (
                    _shell_material_matrices(material, self._material_angle(frame))
                )
                material_stresses = np.linalg.solve(
                    stress_to_local,
                    np.vstack((top, bottom)).T,
                ).T
                values = hill48_plane_stress_equivalent_stress(
                    material_stresses,
                    hill_yield,
                )
                equivalent[index] = float(np.max(values))
                utilization[index] = equivalent[index] / max(
                    float(hill_yield.X),
                    np.finfo(float).tiny,
                )
            if return_global:
                for surface, values in (("top", top), ("bot", bottom)):
                    local_tensor = np.asarray(
                        (
                            (values[0], values[2], local_shear[0]),
                            (values[2], values[1], local_shear[1]),
                            (local_shear[0], local_shear[1], 0.0),
                        ),
                        dtype=float,
                    )
                    global_tensor = frame @ local_tensor @ frame.T
                    for first, second, label in (
                        (0, 0, "xx"),
                        (1, 1, "yy"),
                        (2, 2, "zz"),
                        (0, 1, "xy"),
                        (1, 2, "yz"),
                        (0, 2, "xz"),
                    ):
                        recovered[f"local_{label}_{surface}"] = recovered.get(
                            f"local_{label}_{surface}",
                            np.zeros(len(self.gauss_points), dtype=float),
                        )
                        recovered[f"global_{label}_{surface}"] = recovered.get(
                            f"global_{label}_{surface}",
                            np.zeros(len(self.gauss_points), dtype=float),
                        )
                        recovered[f"local_{label}_{surface}"][index] = local_tensor[
                            first, second
                        ]
                        recovered[f"global_{label}_{surface}"][index] = global_tensor[
                            first, second
                        ]
        recovered["von_mises"] = von_mises
        recovered["equivalent_stress"] = equivalent
        recovered["hill_utilization"] = utilization
        for name, values in recovered.items():
            if isinstance(values, np.ndarray) and not np.all(np.isfinite(values)):
                raise ValueError(
                    f"qualified warped Q4 recovery produced non-finite field {name!r}"
                )
        return recovered

    def compute_stresses(
        self,
        mesh: Any,
        displacements: np.ndarray,
        material: Any,
        return_global: bool = False,
        *,
        natural_points=None,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Recover formulation-native planar fields at four default Gauss points.

        Explicit natural_points samples the unchanged stationary field for a
        planar homogeneous section. It adds development recovery sampling only;
        default quadrature and qualification authority remain unchanged.

        Planar resultants come from the same 35-field stationary system used
        by the condensed tangent.  PL and drilling-hourglass fields are not
        present in this recovery.  The established varying-frame implementation
        remains authoritative for genuinely warped facets.
        """

        _validate_q4_quadrature_authority(self)
        recovery_points = _GAUSS if natural_points is None else np.asarray(natural_points,dtype=float)
        if natural_points is not None:
            if recovery_points.ndim!=2 or recovery_points.shape[1]!=2 or len(recovery_points)==0 or not np.all(np.isfinite(recovery_points)) or np.any(np.abs(recovery_points)>1):
                raise ValueError("natural stress points must be nonempty finite Nx2 in [-1,1]")
            if self.shell_section is not None:
                raise ValueError("custom stress points are unsupported for generalized shell sections")
            if _qualified_runtime_post_observation is not None:
                _qualified_runtime_post_observation()
            _validate_q4_quadrature_authority(self)
        observed_displacements = np.asarray(displacements, dtype=float)
        if _qualified_runtime_post_observation is not None:
            _qualified_runtime_post_observation()
        displacements = observed_displacements
        coordinates = self.get_node_coordinates(mesh)
        _frame, _local, warpage = equation7_frame(coordinates)
        if warpage > self.planar_tolerance:
            if natural_points is not None:
                raise ValueError("custom stress points are unsupported for warped Q4 recovery")
            if self.warped_formulation == "reject":
                raise ValueError(
                    f"E4-PL element {self.element_id} is warped by {warpage:.6e}, "
                    f"above planar_tolerance={self.planar_tolerance:.6e}"
                )
            if self.reference_normal is None:
                return ShellElement.compute_stresses(
                    self,
                    mesh,
                    displacements,
                    material,
                    return_global=return_global,
                )
            if self.shell_section is not None:
                return self._recover_warped_generalized_section(
                    mesh,
                    displacements,
                    return_global=return_global,
                )
            return self._recover_warped_homogeneous_section(
                mesh,
                displacements,
                material,
                return_global=return_global,
            )

        mixed = self._recover_planar_mixed_fields(
            mesh,
            displacements,
            material,
            recovery_points,
            post_observation=_qualified_runtime_post_observation,
        )
        numbered_frame = mixed["frame"]
        frame, membrane_map, curvature_map, shear_map, director_sign = (
            self._physical_director_context(numbered_frame)
        )
        numbered_independent = mixed["independent"]
        numbered_compatible = mixed["compatible"]
        numbered_resultants = mixed["resultants"]
        independent = np.column_stack(
            (
                numbered_independent[:, :3] @ membrane_map.T,
                numbered_independent[:, 3:6] @ curvature_map.T,
                numbered_independent[:, 6:] @ shear_map.T,
            )
        )
        compatible = np.column_stack(
            (
                numbered_compatible[:, :3] @ membrane_map.T,
                numbered_compatible[:, 3:6] @ curvature_map.T,
                numbered_compatible[:, 6:] @ shear_map.T,
            )
        )
        resultants = np.column_stack(
            (
                numbered_resultants[:, :3] @ membrane_map.T,
                numbered_resultants[:, 3:6] @ curvature_map.T,
                numbered_resultants[:, 6:] @ shear_map.T,
            )
        )
        recovered: Dict[str, Any] = {
            "natural_point_coordinates": np.asarray(recovery_points,dtype=float).copy(),
            "recovery_scope": (
                "section_resultants_only"
                if self.shell_section is not None
                else "qualified_q4_local_and_global_physical"
                if return_global
                else "qualified_q4_local_physical_only"
            ),
            "physical_stress_available": self.shell_section is None,
            "membrane_resultant_order": SHELL_MEMBRANE_VOIGT_ORDER,
            "transverse_shear_resultant_order": SHELL_TRANSVERSE_SHEAR_ORDER,
            "membrane_strain": independent[:, :3].copy(),
            "curvature": independent[:, 3:6].copy(),
            "transverse_shear_strain": independent[:, 6:].copy(),
            "compatible_membrane_strain": compatible[:, :3].copy(),
            "compatible_curvature": compatible[:, 3:6].copy(),
            "compatible_transverse_shear_strain": compatible[:, 6:].copy(),
            "membrane_resultants": resultants[:, :3].copy(),
            "bending_resultants": resultants[:, 3:6].copy(),
            "transverse_shear_resultants": resultants[:, 6:].copy(),
            "numerical_fields_excluded": True,
            "implementation_id": IMPLEMENTATION_ID,
            "recovery_policy_id": RECOVERY_POLICY_ID,
            "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
            "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
            "physical_director_authoritative": self.reference_normal is not None,
            "physical_director": frame[:, 2].copy(),
            "numbered_frame_director_sign": int(director_sign),
        }
        if self.shell_section is not None:
            recovered["generalized_stress_scope"] = "section_resultants_only"

        if return_global:
            global_membrane = np.zeros((len(recovery_points), 3, 3), dtype=float)
            global_bending = np.zeros_like(global_membrane)
            global_shear = np.zeros((len(recovery_points), 3), dtype=float)
            for index in range(len(recovery_points)):
                membrane = resultants[index, :3]
                bending = resultants[index, 3:6]
                membrane_tensor = np.asarray(
                    (
                        (membrane[0], membrane[2], 0.0),
                        (membrane[2], membrane[1], 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                    dtype=float,
                )
                bending_tensor = np.asarray(
                    (
                        (bending[0], bending[2], 0.0),
                        (bending[2], bending[1], 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                    dtype=float,
                )
                global_membrane[index] = frame @ membrane_tensor @ frame.T
                global_bending[index] = frame @ bending_tensor @ frame.T
                global_shear[index] = (
                    resultants[index, 6] * frame[:, 0]
                    + resultants[index, 7] * frame[:, 1]
                )
            recovered.update(
                {
                    "global_membrane_resultant_tensors": global_membrane,
                    "global_bending_resultant_tensors": global_bending,
                    "global_transverse_shear_resultants": global_shear,
                }
            )

        if self.shell_section is None:
            thickness = float(self.thickness)
            if not math.isfinite(thickness) or thickness <= 0.0:
                raise ValueError("qualified Q4 recovery requires positive finite thickness")
            membrane_stress = resultants[:, :3] / thickness
            bending_stress = 6.0 * resultants[:, 3:6] / (thickness * thickness)
            transverse = resultants[:, 6:] / thickness
            recovered.update(
                {
                    "membrane_xx": membrane_stress[:, 0].copy(),
                    "membrane_yy": membrane_stress[:, 1].copy(),
                    "membrane_xy": membrane_stress[:, 2].copy(),
                    "bending_xx": bending_stress[:, 0].copy(),
                    "bending_yy": bending_stress[:, 1].copy(),
                    "bending_xy": bending_stress[:, 2].copy(),
                    "shear_xz": transverse[:, 0].copy(),
                    "shear_yz": transverse[:, 1].copy(),
                }
            )
            top = membrane_stress + bending_stress
            bottom = membrane_stress - bending_stress
            vm_top = np.sqrt(
                top[:, 0] ** 2
                - top[:, 0] * top[:, 1]
                + top[:, 1] ** 2
                + 3.0
                * (top[:, 2] ** 2 + transverse[:, 0] ** 2 + transverse[:, 1] ** 2)
            )
            vm_bottom = np.sqrt(
                bottom[:, 0] ** 2
                - bottom[:, 0] * bottom[:, 1]
                + bottom[:, 1] ** 2
                + 3.0
                * (
                    bottom[:, 2] ** 2
                    + transverse[:, 0] ** 2
                    + transverse[:, 1] ** 2
                )
            )
            recovered["von_mises"] = np.maximum(vm_top, vm_bottom)
            recovered["hill_utilization"] = np.zeros(len(recovery_points), dtype=float)
            hill_yield = getattr(material, "hill_yield", None)
            if hill_yield is not None:
                _membrane, _shear, _strain_to_material, stress_to_local = (
                    _shell_material_matrices(material, self._material_angle(frame))
                )
                top_material = np.linalg.solve(stress_to_local, top.T).T
                bottom_material = np.linalg.solve(stress_to_local, bottom.T).T
                hill_top = hill48_plane_stress_equivalent_stress(
                    top_material,
                    hill_yield,
                )
                hill_bottom = hill48_plane_stress_equivalent_stress(
                    bottom_material,
                    hill_yield,
                )
                equivalent = np.maximum(hill_top, hill_bottom)
                recovered["equivalent_stress"] = equivalent
                recovered["hill_utilization"] = equivalent / max(
                    float(hill_yield.X),
                    np.finfo(float).tiny,
                )
                recovered["equivalent_stress_measure"] = "hill48"
            else:
                recovered["equivalent_stress"] = recovered["von_mises"].copy()
                recovered["equivalent_stress_measure"] = "von_mises"

            if return_global:
                for surface, values in (("top", top), ("bot", bottom)):
                    local_tensors = np.zeros((len(recovery_points), 3, 3), dtype=float)
                    local_tensors[:, 0, 0] = values[:, 0]
                    local_tensors[:, 1, 1] = values[:, 1]
                    local_tensors[:, 0, 1] = values[:, 2]
                    local_tensors[:, 1, 0] = values[:, 2]
                    local_tensors[:, 0, 2] = transverse[:, 0]
                    local_tensors[:, 2, 0] = transverse[:, 0]
                    local_tensors[:, 1, 2] = transverse[:, 1]
                    local_tensors[:, 2, 1] = transverse[:, 1]
                    global_tensors = np.asarray(
                        [frame @ tensor @ frame.T for tensor in local_tensors],
                        dtype=float,
                    )
                    for first, second, label in (
                        (0, 0, "xx"),
                        (1, 1, "yy"),
                        (2, 2, "zz"),
                        (0, 1, "xy"),
                        (1, 2, "yz"),
                        (0, 2, "xz"),
                    ):
                        recovered[f"local_{label}_{surface}"] = local_tensors[
                            :, first, second
                        ].copy()
                        recovered[f"global_{label}_{surface}"] = global_tensors[
                            :, first, second
                        ].copy()

        for name, values in recovered.items():
            if isinstance(values, np.ndarray) and not np.all(np.isfinite(values)):
                raise ValueError(
                    f"qualified Q4 recovery produced non-finite field {name!r}"
                )
        return recovered

    def numerical_internal_force(
        self,
        displacement: np.ndarray,
        *,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return PL/hourglass forces separately from physical recovery."""

        if self._qualified_components is None:
            raise RuntimeError("compute_stiffness_matrix must run before numerical force recovery")
        self._validate_qualified_component_guard(
            post_observation=_qualified_runtime_post_observation,
        )
        observed_vector = np.asarray(displacement, dtype=float)
        if _qualified_runtime_post_observation is not None:
            _qualified_runtime_post_observation()
        self._validate_qualified_component_guard(
            post_observation=_qualified_runtime_post_observation,
        )
        components = dict.get(
            object.__getattribute__(self, "__dict__"),
            "_qualified_components",
        )
        snapshots: Dict[str, np.ndarray] = {}
        for name in ("pl", "hourglass", "numerical"):
            value = components[name]
            if (
                type(value) is not np.ndarray
                or value.dtype != np.dtype(np.float64)
                or value.shape != (24, 24)
                or value.nbytes != 24 * 24 * 8
            ):
                raise RuntimeError(
                    f"qualified Q4 {name} component has invalid exact array metadata"
                )
            try:
                raw = memoryview(value).cast("B").tobytes()
            except (BufferError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"qualified Q4 {name} component raw-byte snapshot failed"
                ) from exc
            if len(raw) != 24 * 24 * 8:
                raise RuntimeError(
                    f"qualified Q4 {name} component raw-byte snapshot is incomplete"
                )
            snapshots[name] = np.frombuffer(raw, dtype=np.float64).reshape(24, 24)

        # No shared cache ndarray participates in the operation.  A second
        # validation rejects any cache mutation that occurred while its raw
        # bytes were copied; later changes cannot affect the private views.
        self._validate_qualified_component_guard(
            post_observation=_qualified_runtime_post_observation,
        )
        vector = observed_vector.reshape(24)
        result = {
            "pl": snapshots["pl"] @ vector,
            "hourglass": snapshots["hourglass"] @ vector,
            "numerical": snapshots["numerical"] @ vector,
        }
        self._validate_qualified_component_guard(
            post_observation=_qualified_runtime_post_observation,
        )
        return result

def _make_q4_final_class_authority() -> tuple[Any, Any]:
    authority: list[tuple[Any, ...]] = []

    def initialize(owner: type[Any]) -> None:
        if authority:
            raise RuntimeError("qualified Q4 class authority is already frozen")
        namespace = dict(type.__getattribute__(owner, "__dict__"))
        authority.append(
            (
                owner,
                type(owner),
                type.__getattribute__(owner, "__name__"),
                type.__getattribute__(owner, "__bases__"),
                namespace,
            )
        )

    def require() -> None:
        if len(authority) != 1:
            raise RuntimeError("qualified Q4 class authority is not frozen")
        owner, expected_type, expected_name, expected_bases, expected = authority[0]
        actual = type.__getattribute__(owner, "__dict__")
        if len(actual) != len(expected) or any(
            name not in actual or actual[name] is not value
            for name, value in expected.items()
        ):
            raise ValueError("qualified Q4 concrete class authority changed")
        actual_name = type.__getattribute__(owner, "__name__")
        actual_bases = type.__getattribute__(owner, "__bases__")
        if (
            type(owner) is not expected_type
            or type(actual_name) is not str
            or actual_name != expected_name
            or type(actual_bases) is not tuple
            or len(actual_bases) != len(expected_bases)
            or any(
                actual_base is not expected_base
                for actual_base, expected_base in zip(actual_bases, expected_bases)
            )
        ):
            raise ValueError("qualified Q4 concrete class identity changed")

    return initialize, require


(
    _initialize_q4_final_class_authority,
    _require_q4_final_class_authority,
) = _make_q4_final_class_authority()


def _make_q4_runtime_boundary_holder() -> tuple[Any, Any]:
    authority: list[Any] = []

    def install(guard: Any) -> None:
        if authority:
            raise RuntimeError("qualified Q4 runtime boundary is already bound")
        authority.append(guard)

    def require(element: Any, *, context: str) -> None:
        if len(authority) != 1:
            raise RuntimeError("qualified Q4 runtime boundary is not bound")
        authority[0](element, context=context)

    return install, require


(
    _install_q4_runtime_boundary,
    _require_exact_q4_runtime_authority,
) = _make_q4_runtime_boundary_holder()


def _make_prepared_q4_input_validator() -> Any:
    # Capture the already-bound authority and original preparation registry.
    # This checks reuse without observing a new constitutive trial.
    boundary = _require_exact_q4_runtime_authority
    lookup = _lookup_prepared_q4_correction
    matches = _q4_fast_input_snapshot_matches
    prepared_type = _PreparedQ4LinearCorrection

    def require(element: Any, mesh: Any, material: Any, prepared: Any) -> None:
        boundary(element, context="qualified Q4 same-parent trial reuse")
        if type(prepared) is not prepared_type:
            raise TypeError("qualified Q4 correction preparation required")
        binding = lookup(prepared)
        if (binding is None or binding[0] is not element or
                binding[1] is not mesh or binding[2] is not material or
                element._qualified_component_guard is not binding[3] or
                not matches(element, mesh, material, binding[4])):
            raise RuntimeError("qualified Q4 correction preparation is stale")
        element._validate_qualified_component_guard()
        boundary(element, context="qualified Q4 same-parent trial reuse")

    return require


_validate_prepared_q4_correction_inputs = _make_prepared_q4_input_validator()


def _make_q4_cached_stiffness_epoch_holder() -> tuple[Any, Any]:
    """Expose only the closure-bound runtime epoch check used by warm Q4."""

    authority: list[Any] = []

    def install(guard: Any) -> None:
        if authority:
            raise RuntimeError(
                "qualified Q4 cached-stiffness epoch guard is already bound"
            )
        authority.append(guard)

    def require() -> None:
        if len(authority) != 1:
            raise RuntimeError(
                "qualified Q4 cached-stiffness epoch guard is not bound"
            )
        authority[0]()

    return install, require


(
    _install_q4_cached_stiffness_runtime_epoch_authority,
    _require_q4_cached_stiffness_runtime_epoch_authority,
) = _make_q4_cached_stiffness_epoch_holder()


_q4_runtime_epoch_manager = make_authority_epoch_manager(
    "qualified Q4 runtime"
)


def _make_q4_trusted_operation_scope() -> tuple[Any, Any]:
    """Create a context-local fast boundary for fully validated operations.

    Entry and exit retain the complete element/runtime validation.  Calls made
    between those brackets only check the monotonic authority generation and
    exact instance identity.  The token class and ContextVar are closure-owned,
    so there is no process-wide guard-disable switch.  The token carries an
    explicit live flag revoked before exit validation/reset on every exit, so
    a context copied inside the scope cannot retain the operation lease after
    the scope has ended.
    """

    active: ContextVar[Any] = ContextVar("qualified_q4_operation", default=None)
    manager = _q4_runtime_epoch_manager

    class Token:
        __slots__ = ("generation", "records", "live")

        def __init__(self, generation: int, records: dict[int, tuple[Any, Any]]) -> None:
            self.generation = generation
            self.records = records
            self.live = True

    @contextmanager
    def operation(elements: Sequence[Any]) -> Any:
        owned = tuple(elements)
        for element in owned:
            _require_exact_q4_runtime_authority(
                element,
                context="qualified Q4 trusted operation preflight",
            )
        generation = manager.capture_generation()
        records = {
            id(element): (element, object.__getattribute__(element, "__dict__"))
            for element in owned
        }
        token = Token(generation, records)
        marker = active.set(token)
        try:
            yield
        finally:
            # Revoke the shared token before exit validation/reset so a
            # context copied inside the scope cannot retain the operation
            # lease after any exit (normal, error or cancellation).
            token.live = False
            active.reset(marker)
            try:
                manager.require_generation(generation)
                for element in owned:
                    _require_exact_q4_runtime_authority(
                        element,
                        context="qualified Q4 trusted operation output",
                    )
                    record = records.get(id(element))
                    if (
                        record is None
                        or record[0] is not element
                        or record[1]
                        is not object.__getattribute__(element, "__dict__")
                    ):
                        raise ValueError(
                            "qualified Q4 trusted operation identity changed"
                        )
            except BaseException:
                for element in owned:
                    _invalidate_q4_guarded_call_caches(element)
                raise
    def accepts(element: Any) -> bool:
        token = active.get()
        if type(token) is not Token:
            return False
        if not token.live:
            return False
        record = token.records.get(id(element))
        if (
            record is None
            or record[0] is not element
            or record[1] is not object.__getattribute__(element, "__dict__")
        ):
            return False
        manager.require_generation(token.generation)
        return True

    return operation, accepts


(
    _q4_trusted_operation_scope,
    _q4_trusted_operation_accepts,
) = _make_q4_trusted_operation_scope()


def _invalidate_q4_guarded_call_caches(element: Any) -> None:
    """Drop every derived cache after a rejected authority lease."""

    namespace = object.__getattribute__(element, "__dict__")
    if type(namespace) is not dict:
        return
    for name in (
        "_hourglass_stiffness_matrix",
        "_internal_forces",
        "_mass_matrix",
        "_nl_cache",
        "_nl_cache_key",
        "_qualified_cache_key",
        "_qualified_component_guard",
        "_qualified_components",
        "_stiffness_matrix",
    ):
        if name in namespace:
            object.__setattr__(element, name, None)
    _clear_q4_component_cache_provenance(element)
    _clear_q4_nonlinear_cache_provenance(element)


def _retire_q4_private_mesh_caches(element: Any, mesh: Any) -> None:
    """Retire an exclusively owned Q4 after its private compiler is finished.

    This is a lifecycle boundary, not a live authority lease or global purge.
    The closure check runs before cache mutation, including for a rebound
    external element.  Removing the record releases both its strong guard and
    input snapshot; removing derived instance guards breaks the private cycle.
    """
    _retire_q4_private_component_provenance(element, mesh)
    _invalidate_q4_guarded_call_caches(element)


def _bind_q4_exact_quadrature_boundary(method: Any) -> Any:
    """Bind immutable quadrature/numerical authority around a Q4 operation."""

    quadrature_guard = _validate_q4_quadrature_authority
    numerical_guard = require_exact_numpy_runtime_authority
    numerical_module_guard = _require_exact_numpy_runtime_module_identity
    authority_signer = _module_authority_signature
    runtime_manager = _q4_runtime_epoch_manager
    runtime_module = sys.modules[__name__]
    module_bindings = tuple(
        (name, value)
        for name, value in tuple(globals().items())
        if callable(value) or isinstance(value, ModuleType)
    )
    module_data = tuple(
        (name, value, type(value), authority_signer(value))
        for name, value in tuple(globals().items())
        if name.lstrip("_").isupper()
    )
    ignored_dependency_class_cache_name = "__slotnames__"
    dependency_namespaces = tuple(
        (
            owner,
            {
                name: value
                for name, value in dict(
                    type.__getattribute__(owner, "__dict__")
                ).items()
                if name != ignored_dependency_class_cache_name
            },
        )
        for owner in (
            Element,
            ShellElement,
            _fe_core_module.FEMesh,
            _fe_core_module.Material,
            GeneralizedShellSection,
        )
    )
    def capture_dependency_namespace(module: ModuleType) -> dict[str, Any]:
        captured: dict[str, Any] = {}
        for name, value in tuple(vars(module).items()):
            if callable(value) or isinstance(value, ModuleType):
                captured[name] = value
                continue
            if not name.lstrip("_").isupper():
                continue
            try:
                _require_immutable_authority_data(
                    value,
                    label=f"qualified Q4 {module.__name__}.{name}",
                )
            except TypeError:
                continue
            captured[name] = value
        return captured

    dependency_modules = tuple(
        (
            module,
            str(module.__name__),
            capture_dependency_namespace(module),
        )
        for module in (
            _s3_state_module,
            _elements_module,
            _fe_core_module,
            _plasticity_module,
            _shell_sections_module,
            _elastic_increment_module,
            _local_contractions,
            _local_contractions._native,
            _q4_assumed_strain,
        )
    )
    expected_class = QualifiedE4PLShellElement
    expected_formulation_id = FORMULATION_ID
    class_data_names = frozenset().union(
        *(
            frozenset(type.__getattribute__(owner, "__dict__"))
            for owner in type.__getattribute__(expected_class, "__mro__")
        )
    )
    dependency_class_metadata = tuple(
        (
            owner,
            type(owner),
            owner.__name__,
            owner.__qualname__,
            owner.__module__,
            type.__getattribute__(owner, "__bases__"),
        )
        for owner, _namespace in dependency_namespaces
    )
    method_signature = inspect.signature(method)
    post_observation_name = "_qualified_runtime_post_observation"
    accepts_post_observation = post_observation_name in method_signature.parameters
    fast_cached_stiffness = _try_q4_fast_cached_stiffness
    if method.__name__ == "compute_stiffness_matrix":
        def private_function_clone(
            function: _FunctionType,
            global_overrides: Optional[dict[str, Any]] = None,
        ) -> _FunctionType:
            private_globals = dict(function.__globals__)
            if global_overrides is not None:
                private_globals.update(global_overrides)
            cloned = _FunctionType(
                function.__code__,
                private_globals,
                function.__name__,
                function.__defaults__,
                function.__closure__,
            )
            if function.__kwdefaults__ is not None:
                cloned.__kwdefaults__ = dict(function.__kwdefaults__)
            return cloned

        private_fast_array_authority = private_function_clone(
            _require_q4_fast_array_authority,
        )
        private_fast_base_authority = private_function_clone(
            _require_q4_fast_base_authority,
        )
        private_fast_input_snapshot = private_function_clone(
            _q4_fast_input_snapshot_matches,
        )
        fast_cached_stiffness = private_function_clone(
            fast_cached_stiffness,
            {
                "_require_q4_fast_array_authority": (
                    private_fast_array_authority
                ),
                "_require_q4_fast_base_authority": private_fast_base_authority,
                "_q4_fast_input_snapshot_matches": private_fast_input_snapshot,
            },
        )
        numerical_module_guard = private_function_clone(
            numerical_module_guard,
        )
    class_mutable_mappings = tuple(
        (
            f"{owner.__name__}.{name}",
            value,
            tuple(value.items()),
        )
        for owner, expected_namespace in dependency_namespaces
        for name, value in expected_namespace.items()
        if type(value) is dict
    )
    for label, mapping, _items in class_mutable_mappings:
        if not all(type(key) is str for key in mapping):
            raise TypeError(f"{label} authority keys must be exact strings")
    authority_array_metadata = _capture_authority_array_metadata(
        tuple(value for _name, value, _kind, _signature in module_data),
        tuple(
            tuple(expected_namespace.values())
            for _module, _module_name, expected_namespace in dependency_modules
        ),
        tuple(
            tuple(expected_namespace.values())
            for _owner, expected_namespace in dependency_namespaces
        ),
    )

    for name, value, _expected_type, _expected_signature in module_data:
        if name == "_Q4_GENERALIZED_SECTION_NAMESPACE_AUTHORITY":
            continue
        _require_immutable_authority_data(
            value,
            label=f"qualified Q4 {name}",
        )
    for module, module_name, expected_namespace in dependency_modules:
        for name, value in expected_namespace.items():
            if not name.lstrip("_").isupper():
                continue
            _require_immutable_authority_data(
                value,
                label=f"qualified Q4 {module_name}.{name}",
            )

    runtime_manager.watch_module(
        runtime_module,
        (
            "__name__",
            *(name for name, _expected in module_bindings),
            *(name for name, *_expected in module_data),
        ),
    )
    for dependency_module, _module_name, expected_namespace in dependency_modules:
        runtime_manager.watch_module(
            dependency_module,
            ("__name__", *expected_namespace),
        )
    _watch_exact_numpy_runtime_epoch(runtime_manager)

    def formulation_module_guard() -> None:
        namespace = globals()
        changed = [
            name
            for name, expected in module_bindings
            if namespace.get(name) is not expected
        ]
        changed.extend(
            name
            for name, _value, expected_type, expected_signature in module_data
            if type(namespace.get(name)) is not expected_type
            or authority_signer(namespace.get(name))
            != expected_signature
        )
        for module, module_name, expected_namespace in dependency_modules:
            actual_namespace = vars(module)
            changed.extend(
                f"{module_name}.{name}"
                for name, expected in expected_namespace.items()
                if actual_namespace.get(name) is not expected
            )
        if changed:
            raise ValueError(
                "qualified Q4 formulation runtime authority changed: "
                + ", ".join(sorted(set(changed)))
            )

    def formulation_class_guard() -> None:
        for (
            owner,
            expected_namespace,
        ), metadata in zip(dependency_namespaces, dependency_class_metadata):
            actual_namespace = type.__getattribute__(owner, "__dict__")
            if (
                len(actual_namespace)
                - int(ignored_dependency_class_cache_name in actual_namespace)
                != len(expected_namespace)
                or any(
                    name not in actual_namespace
                    or actual_namespace[name] is not expected
                    for name, expected in expected_namespace.items()
                )
            ):
                raise ValueError(
                    f"qualified Q4 {metadata[2]} class namespace changed"
                )
        for (
            owner,
            expected_type,
            expected_name,
            _expected_qualname,
            _expected_module,
            expected_bases,
        ) in dependency_class_metadata:
            actual_name = type.__getattribute__(owner, "__name__")
            actual_bases = type.__getattribute__(owner, "__bases__")
            if (
                type(owner) is not expected_type
                or type(actual_name) is not str
                or actual_name != expected_name
                or type(actual_bases) is not tuple
                or len(actual_bases) != len(expected_bases)
                or any(
                    actual_base is not expected_base
                    for actual_base, expected_base in zip(
                        actual_bases, expected_bases
                    )
                )
            ):
                raise ValueError(
                    f"qualified Q4 {expected_name} class identity changed"
                )
        for label, mapping, expected_items in class_mutable_mappings:
            if (
                type(mapping) is not dict
                or not all(type(key) is str for key in mapping)
                or len(mapping) != len(expected_items)
                or any(
                    name not in mapping or mapping[name] is not expected
                    for name, expected in expected_items
                )
            ):
                raise ValueError(
                    f"qualified Q4 {label} mutable class authority changed"
                )
        _require_authority_array_metadata(
            authority_array_metadata,
            label="qualified Q4 runtime authority",
        )

    def slow_runtime_guard() -> None:
        numerical_guard(context=f"qualified Q4 {method.__name__}")
        formulation_module_guard()
        _require_q4_final_class_authority()

    runtime_epoch_guard = runtime_manager.bind(slow_runtime_guard)
    if method.__name__ == "compute_stiffness_matrix":
        _install_q4_cached_stiffness_runtime_epoch_authority(
            runtime_epoch_guard
        )

    def runtime_guard() -> None:
        numerical_module_guard(context=f"qualified Q4 {method.__name__}")
        runtime_epoch_guard()
        formulation_class_guard()

    def instance_guard(self: Any) -> None:
        if type(self) is not expected_class:
            raise ValueError("qualified Q4 requires the exact element class")
        namespace = object.__getattribute__(self, "__dict__")
        if type(namespace) is not dict:
            raise ValueError("qualified Q4 instance namespace is incompatible")
        if not all(type(name) is str for name in namespace):
            raise ValueError("qualified Q4 instance keys must be exact strings")
        shadowed = class_data_names.intersection(namespace)
        if shadowed:
            raise ValueError(
                "qualified Q4 class authority has instance shadows: "
                + ", ".join(sorted(shadowed))
            )
        if any(callable(value) for value in namespace.values()):
            raise ValueError(
                "qualified Q4 instance contains a callable authority override"
            )
        element_id = namespace.get("element_id")
        node_ids = namespace.get("node_ids")
        material_name = namespace.get("material_name")
        if (
            type(element_id) is not int
            or type(node_ids) is not tuple
            or len(node_ids) != 4
            or not all(type(node_id) is int for node_id in node_ids)
            or len(set(node_ids)) != 4
            or type(material_name) is not str
            or _static_mro_attribute(type(self), "element_id") is not None
            or _static_mro_attribute(type(self), "node_ids") is not None
            or _static_mro_attribute(type(self), "material_name") is not None
            or _static_mro_attribute(type(self), "formulation_id")
            != expected_formulation_id
        ):
            raise ValueError(
                "qualified Q4 instance connectivity/material authority is incompatible"
            )
        expected_topology = {
            "reduced_integration": False,
            "_is_3node": False,
            "_is_4node": True,
            "_is_6node": False,
            "_is_8node": False,
            "_is_triangular": False,
            "_is_quadrilateral": True,
        }
        if any(
            dict.get(namespace, name) is not expected
            for name, expected in expected_topology.items()
        ):
            raise ValueError(
                "qualified Q4 fixed topology authority is incompatible"
            )

        for name in ("reference_normal", "material_direction"):
            vector = dict.get(namespace, name)
            if vector is None:
                continue
            if (
                type(vector) is not np.ndarray
                or vector.dtype != np.dtype(np.float64)
                or vector.shape != (3,)
                or vector.strides != (8,)
                or not vector.flags.c_contiguous
                or vector.flags.writeable
            ):
                raise ValueError(
                    f"qualified Q4 {name} vector authority is incompatible"
                )
            current: Any = vector
            while type(current) is np.ndarray:
                if current.flags.writeable:
                    raise ValueError(
                        f"qualified Q4 {name} vector base is writeable"
                    )
                current = current.base
            if not (
                type(current) is bytes
                or isinstance(current, memoryview) and current.readonly
            ):
                raise ValueError(
                    f"qualified Q4 {name} vector base is incompatible"
                )

    def boundary_guard(self: Any, *, context: str) -> None:
        del context
        runtime_guard()
        instance_guard(self)
        quadrature_guard(self)

    if method.__name__ == "get_node_coordinates":
        _install_q4_runtime_boundary(boundary_guard)

    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        if post_observation_name in kwargs:
            raise TypeError(
                f"{method.__name__} does not accept a caller-supplied runtime guard"
            )
        if method.__name__ == "__init__":
            runtime_guard()
            call_generation = runtime_manager.capture_generation()
            completed = False
            try:
                result = method(self, *args, **kwargs)
                completed = True
                return result
            finally:
                try:
                    runtime_guard()
                    runtime_manager.require_generation(call_generation)
                    if completed:
                        instance_guard(self)
                        quadrature_guard(self)
                except BaseException:
                    _invalidate_q4_guarded_call_caches(self)
                    raise
        if _q4_trusted_operation_accepts(self):
            call_kwargs = dict(kwargs)

            def trusted_post_observation_guard() -> None:
                if not _q4_trusted_operation_accepts(self):
                    raise ValueError("qualified Q4 trusted operation ended")

            if accepts_post_observation:
                call_kwargs[post_observation_name] = trusted_post_observation_guard
            try:
                return method(self, *args, **call_kwargs)
            finally:
                if not _q4_trusted_operation_accepts(self):
                    _invalidate_q4_guarded_call_caches(self)
                    raise ValueError("qualified Q4 trusted operation changed")
        if method.__name__ == "compute_stiffness_matrix":
            mesh = None
            material = None
            fast_arguments_are_exact = False
            if len(args) == 2 and not kwargs:
                mesh, material = args
                fast_arguments_are_exact = True
            elif not args and set(kwargs) == {"mesh", "material"}:
                mesh = kwargs["mesh"]
                material = kwargs["material"]
                fast_arguments_are_exact = True
            if fast_arguments_are_exact:
                numerical_module_guard(
                    context="qualified Q4 cached stiffness"
                )
                runtime_epoch_guard()
                call_generation = runtime_manager.capture_generation()
                try:
                    cached = fast_cached_stiffness(self, mesh, material)
                    runtime_manager.require_generation(call_generation)
                except BaseException:
                    _invalidate_q4_guarded_call_caches(self)
                    runtime_manager.require_generation(call_generation)
                    raise
                if cached is not None:
                    return cached
        boundary_guard(self, context=f"qualified Q4 {method.__name__}")
        call_generation = runtime_manager.capture_generation()

        def post_observation_guard() -> None:
            boundary_guard(self, context=f"qualified Q4 {method.__name__}")
            runtime_manager.require_generation(call_generation)

        call_kwargs = dict(kwargs)
        if accepts_post_observation:
            call_kwargs[post_observation_name] = post_observation_guard

        try:
            return method(self, *args, **call_kwargs)
        finally:
            try:
                boundary_guard(self, context=f"qualified Q4 {method.__name__}")
                runtime_manager.require_generation(call_generation)
            except BaseException:
                _invalidate_q4_guarded_call_caches(self)
                raise

    guarded.__name__ = method.__name__
    guarded.__qualname__ = method.__qualname__
    guarded.__doc__ = method.__doc__
    guarded.__annotations__ = dict(method.__annotations__)
    guarded.__signature__ = method_signature.replace(
        parameters=[
            parameter
            for name, parameter in method_signature.parameters.items()
            if name != post_observation_name
        ]
    )
    return guarded


def _install_q4_class_access_guard(names: Sequence[str]) -> None:
    """Guard direct public descriptor/method lookup on the exact final class."""

    exact_names = frozenset(str(name) for name in names)
    class_epoch_guard = _q4_runtime_epoch_manager.bind(
        _require_q4_final_class_authority
    )
    raw_getattribute = object.__getattribute__

    def guarded_getattribute(self: Any, name: str) -> Any:
        if type(name) is str and name in exact_names:
            class_epoch_guard()
        return raw_getattribute(self, name)

    guarded_getattribute.__name__ = "__getattribute__"
    guarded_getattribute.__qualname__ = (
        "QualifiedE4PLShellElement.__getattribute__"
    )
    setattr(QualifiedE4PLShellElement, "__getattribute__", guarded_getattribute)


_q4_stiffness_inherited_shadow_sources = (
    (Element, "total_dofs"),
    (ShellElement, "num_nodes"),
    (ShellElement, "dofs_per_node"),
    (ShellElement, "compute_shape_functions"),
    (ShellElement, "_compute_4node_shape_functions"),
    (ShellElement, "compute_jacobian"),
    (ShellElement, "_normalize"),
    (ShellElement, "_fallback_edge_direction"),
    (ShellElement, "_local_dof_transform"),
    (ShellElement, "_build_shell_b_matrices"),
    (ShellElement, "_build_drilling_b_matrix"),
    (ShellElement, "_material_angle"),
    (ShellElement, "_center_frame"),
    (ShellElement, "_reference_center"),
    (ShellElement, "_mitc4_shear_samples"),
    (ShellElement, "_hourglass_stabilization_matrix"),
    (ShellElement, "_rigid_body_mode_matrix"),
    (ShellElement, "gauss_points"),
    (ShellElement, "gauss_weights"),
    (ShellElement, "shear_gauss_points"),
    (ShellElement, "shear_gauss_weights"),
)
_q4_stiffness_inherited_shadow_names = tuple(
    name for _owner, name in _q4_stiffness_inherited_shadow_sources
)
for _q4_shadow_owner, _q4_shadow_name in _q4_stiffness_inherited_shadow_sources:
    setattr(
        QualifiedE4PLShellElement,
        _q4_shadow_name,
        type.__getattribute__(_q4_shadow_owner, "__dict__")[_q4_shadow_name],
    )
del _q4_shadow_owner, _q4_shadow_name


_q4_guarded_public_names = (
    "__init__",
    "get_node_coordinates",
    "init_nonlinear_state",
    "attach_current_tangent_algorithmic_origin",
    "seal_committed_current_tangent_state",
    "validate_committed_current_tangent_binding",
    "validate_committed_current_tangent_semantics",
    "validate_committed_current_tangent_state",
    "seal_noncurrent_deleted_state",
    "validate_noncurrent_deleted_state",
    "mark_noncurrent_failed_state",
    "validate_noncurrent_failed_state",
    "compute_stiffness_components",
    "compute_stiffness_matrix",
    "compute_mass_matrix",
    "compute_geometric_stiffness_matrix",
    "compute_internal_forces",
    "compute_nonlinear_response",
    "_prepare_nonlinear_linear_correction",
    "_compute_prepared_nonlinear_response",
    "_elastic_increment_admitted",
    "_elastic_increment_response",
    "compute_committed_current_tangent_components",
    "compute_stresses",
    "numerical_internal_force",
)
for _q4_public_name in _q4_guarded_public_names:
    setattr(
        QualifiedE4PLShellElement,
        _q4_public_name,
        _bind_q4_exact_quadrature_boundary(
            type.__getattribute__(QualifiedE4PLShellElement, "__dict__")[
                _q4_public_name
            ]
        ),
    )
del _q4_public_name

_q4_guarded_property_names = ("physical_reference_director",)
for _q4_property_name in _q4_guarded_property_names:
    _q4_property = type.__getattribute__(
        QualifiedE4PLShellElement,
        "__dict__",
    )[_q4_property_name]
    setattr(
        QualifiedE4PLShellElement,
        _q4_property_name,
        property(
            _bind_q4_exact_quadrature_boundary(_q4_property.fget),
            _q4_property.fset,
            _q4_property.fdel,
            _q4_property.__doc__,
        ),
    )
del _q4_property_name, _q4_property

_initialize_q4_final_class_authority(QualifiedE4PLShellElement)
_q4_runtime_epoch_manager.protect_type_entries(
    QualifiedE4PLShellElement,
    (
        *_q4_guarded_public_names,
        *_q4_guarded_property_names,
        "__setattr__",
        "__delattr__",
        "__getattribute__",
        "from_dict",
        "to_dict",
        "validate_quadrature_authority",
        *_q4_stiffness_inherited_shadow_names,
        "GAUSS_POINTS_2x2",
        "GAUSS_WEIGHTS_2x2",
        "GAUSS_POINTS_1x1",
        "GAUSS_WEIGHTS_1x1",
        "_MITC4_SAMPLE_POINTS",
    ),
)
_q4_runtime_epoch_manager.watch_type(QualifiedE4PLShellElement, None)


def _capture_q4_class_namespace_authority() -> MappingProxyType:
    namespaces = {}
    for owner in type.__getattribute__(QualifiedE4PLShellElement, "__mro__"):
        # ABC's lazy annotations can be observed by unrelated Python 3.14
        # tooling after import. Capture its exact metadata before sealing.
        if owner is ABC:
            type.__getattribute__(owner, "__annotations__")
        namespaces[owner] = MappingProxyType(
            dict(type.__getattribute__(owner, "__dict__"))
        )
    return MappingProxyType(namespaces)


_QUALIFIED_Q4_MODULE_FUNCTION_AUTHORITY = MappingProxyType(
    {
        name: value
        for name, value in tuple(globals().items())
        if callable(value) or isinstance(value, ModuleType)
    }
)
_QUALIFIED_Q4_CLASS_NAMESPACE_AUTHORITY = _capture_q4_class_namespace_authority()
_QUALIFIED_Q4_MODULE_DATA_AUTHORITY = MappingProxyType(
    {
        name: (type(value), _module_authority_signature(value))
        for name, value in tuple(globals().items())
        if name.lstrip("_").isupper()
    }
)


__all__ = [
    "DIRECTOR_POLARITY_POLICY_ID",
    "DIRECTOR_REVERSAL_TRANSFORM_ID",
    "FORMULATION_ID",
    "IMPLEMENTATION_ID",
    "Q4_ACTIVITY_DISPOSITION_SCHEMA_ID",
    "Q4_CURRENT_STATE_ALGORITHMIC_ORIGIN_SCHEMA_ID",
    "Q4_CURRENT_STATE_BINDING_SCHEMA_ID",
    "Q4_CURRENT_STATE_PROJECTION_POLICY_ID",
    "Q4_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID",
    "Q4_DELETED_FROZEN_POLICY_ID",
    "Q4_FAILED_STATE_POLICY_ID",
    "Q4_QUADRATURE_AUTHORITY_ID",
    "QualifiedE4PLShellElement",
    "QualifiedQ4MigrationWarning",
    "RECOVERY_POLICY_ID",
    "STATIONARY_SOLVE_POLICY_ID",
    "equation7_frame",
]
