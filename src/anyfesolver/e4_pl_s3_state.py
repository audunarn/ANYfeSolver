"""Committed-state primitives for the qualified E4-PL S3 companion.

This module deliberately contains no element or solver integration.  It
freezes the identity, shape, and canonical-fingerprint boundary needed before
the formulation-native incremental-director implementation can consume or
persist nonlinear state.

The physical bubble orientation is the fourth reconstructed director triad.
``bubble_rotation_last_increment`` is a reserved exact-zero slot: every new
inner solve starts from zero and no converged bubble increment is persisted.
It is intentionally distinct from ``alpha``, the material-hardening
coordinate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import fields as dataclass_fields, is_dataclass
import hashlib
import json
import math
import re
import sys
from types import MappingProxyType, ModuleType
from typing import Any, Callable, Optional

import numpy as np

from anymaterial import (
    DNVC208MaterialCurve,
    Hill48Yield,
    IsotropicMaterial,
    LinearHardeningCurve,
    OrthotropicMaterial,
    PiecewiseLinearCurve,
    PowerLawHardeningCurve,
    StructuralMaterial,
    elastic_compliance_matrix,
    material_symmetry,
)

from ._qualified_authority_epoch import make_authority_epoch_manager
from .e4_pl_s3_initial_fields import (
    GENERALIZED_INITIAL_FIELD_POLICY_ID,
    integrate_generalized_initial_fields,
)
from .shell_sections import validate_generalized_shell_section


_NUMPY_RUNTIME_ROOT_NAMES = (
    "abs", "add", "all", "allclose", "any", "arange", "arccos", "arctan2",
    "argmax", "argsort", "array", "array_equal", "asarray",
    "ascontiguousarray", "average", "block", "broadcast_to", "ceil",
    "clip", "column_stack", "concatenate", "cos", "count_nonzero",
    "cross", "deg2rad", "degrees", "diag", "dot", "dtype", "einsum",
    "empty", "empty_like", "errstate", "eye", "finfo", "flatnonzero",
    "float64",
    "frombuffer", "fromiter", "full", "interp", "isclose", "isfinite",
    "inf", "int64", "int8", "intp", "isinf", "isnan", "issubdtype",
    "ix_", "lexsort", "linspace",
    "max", "maximum", "mean", "median", "meshgrid", "min", "moveaxis",
    "nan", "newaxis", "nextafter", "number", "ones", "ones_like", "outer",
    "pi", "power", "radians", "real",
    "repeat", "sin", "sort", "sqrt", "sum", "swapaxes", "tile", "trace",
    "unique", "vstack", "where", "zeros", "zeros_like", "ndarray", "generic",
    "bool_", "integer", "floating",
)
_NUMPY_RUNTIME_LINALG_NAMES = (
    "cholesky", "cond", "det", "eigh", "eigvalsh", "inv", "lstsq",
    "norm", "pinv", "qr", "solve", "svd", "LinAlgError",
)


def _bind_exact_numpy_runtime_authority(
    numpy_module: Any,
    sys_module: Any,
) -> tuple[Any, Any]:
    """Capture the bounded NumPy operations used by qualified shell mechanics."""

    root_names = _NUMPY_RUNTIME_ROOT_NAMES
    linalg_names = _NUMPY_RUNTIME_LINALG_NAMES
    linalg_module = numpy_module.linalg
    numpy_name = str(numpy_module.__name__)
    linalg_name = str(linalg_module.__name__)
    sys_modules = sys_module.modules
    root_bindings = tuple(
        (name, vars(numpy_module)[name]) for name in root_names
    )
    linalg_bindings = tuple(
        (name, vars(linalg_module)[name]) for name in linalg_names
    )

    def require_exact(*, context: str) -> None:
        changed: list[str] = []
        if sys_modules.get(numpy_name) is not numpy_module:
            changed.append("numpy.__module__")
        root_namespace = vars(numpy_module)
        if root_namespace.get("__name__") != numpy_name:
            changed.append("numpy.__name__")
        if root_namespace.get("linalg") is not linalg_module:
            changed.append("numpy.linalg")
        changed.extend(
            f"numpy.{name}"
            for name, expected in root_bindings
            if root_namespace.get(name) is not expected
        )
        if sys_modules.get(linalg_name) is not linalg_module:
            changed.append("numpy.linalg.__module__")
        linalg_namespace = vars(linalg_module)
        if linalg_namespace.get("__name__") != linalg_name:
            changed.append("numpy.linalg.__name__")
        changed.extend(
            f"numpy.linalg.{name}"
            for name, expected in linalg_bindings
            if linalg_namespace.get(name) is not expected
        )
        if changed:
            raise ValueError(
                f"{context} requires exact numerical runtime operations; changed "
                + ", ".join(sorted(changed))
            )

    def require_module_identity(*, context: str) -> None:
        """Check the unobservable ``sys.modules`` mapping every call."""

        sys_namespace = vars(sys_module)
        if (
            sys_namespace.get("modules") is sys_modules
            and sys_modules.get(numpy_name) is numpy_module
            and sys_modules.get(linalg_name) is linalg_module
        ):
            return
        changed: list[str] = []
        if sys_namespace.get("modules") is not sys_modules:
            changed.append("sys.modules")
        if sys_modules.get(numpy_name) is not numpy_module:
            changed.append("numpy.__module__")
        if sys_modules.get(linalg_name) is not linalg_module:
            changed.append("numpy.linalg.__module__")
        if changed:
            raise ValueError(
                f"{context} requires exact numerical runtime modules; changed "
                + ", ".join(sorted(changed))
            )

    manager = make_authority_epoch_manager(
        "qualified NumPy runtime"
    )
    manager.watch_module(
        numpy_module,
        (*root_names, "__name__", "linalg"),
    )
    manager.watch_module(linalg_module, (*linalg_names, "__name__"))
    epoch_guard = manager.bind_context(require_exact)

    def require(*, context: str) -> None:
        require_module_identity(context=context)
        epoch_guard(context=context)

    return require, require_module_identity


(
    require_exact_numpy_runtime_authority,
    _require_exact_numpy_runtime_module_identity,
) = _bind_exact_numpy_runtime_authority(np, sys)


def _watch_exact_numpy_runtime_epoch(manager: Any) -> None:
    """Subscribe one private manager to the frozen NumPy attribute surface."""

    manager.watch_module(
        np,
        (*_NUMPY_RUNTIME_ROOT_NAMES, "__name__", "linalg"),
    )
    manager.watch_module(
        np.linalg,
        (*_NUMPY_RUNTIME_LINALG_NAMES, "__name__"),
    )


def _make_module_authority_signer(
    numpy_module: Any,
    mapping_proxy_type: type[Any],
) -> Any:
    """Capture signing dependencies so a guarded module cannot replace them."""

    numpy_scalar_types = frozenset(
        value
        for value in numpy_module.sctypeDict.values()
        if type(value) is type and issubclass(value, numpy_module.generic)
    )
    original_asarray = numpy_module.asarray
    original_float64 = numpy_module.float64
    # Cache only small, recursively inert immutable containers. Strong owner
    # references prevent identity reuse; mutable authority is always signed live.
    immutable_cache: list[tuple[Any, tuple[Any, ...]] | None] = [None] * 256

    def cached_immutable(value: Any) -> tuple[Any, ...] | None:
        if (numpy_module.asarray is not original_asarray
                or numpy_module.float64 is not original_float64):
            return None
        entry = immutable_cache[(id(value) >> 4) % 256]
        return entry[1] if entry is not None and entry[0] is value else None

    def retain_immutable(value: Any, result: tuple[Any, ...]) -> None:
        if (numpy_module.asarray is not original_asarray
                or numpy_module.float64 is not original_float64):
            return
        pending = [value]
        nodes = 0
        payload_bytes = 0
        while pending:
            member = pending.pop()
            nodes += 1
            kind = type(member)
            if kind in {tuple, frozenset}:
                if len(member) + len(pending) > 256:
                    return
                pending.extend(member)
            elif kind in {str, bytes}:
                payload_bytes += len(member) * (4 if kind is str else 1)
            elif member is None or kind in {bool, int, float}:
                payload_bytes += (member.bit_length() // 8 + 1
                                  if kind is int else 8)
            else:
                return
            if nodes > 256 or payload_bytes > 4096 or len(pending) > 256:
                return
        # A fixed slot assignment cannot race eviction or grow the cache.
        immutable_cache[(id(value) >> 4) % 256] = (value, result)

    def signature(value: Any) -> tuple[Any, ...]:
        """Return an immutable exact signature for frozen module authority data."""

        if type(value) is numpy_module.ndarray:
            array = numpy_module.asarray(value)
            return (
                "ndarray",
                array.dtype.str,
                tuple(int(size) for size in array.shape),
                tuple(int(stride) for stride in array.strides),
                bool(array.flags.c_contiguous),
                bool(array.flags.writeable),
                array.tobytes(order="A"),
            )
        if type(value) in numpy_scalar_types:
            scalar = numpy_module.asarray(value)
            return ("numpy_scalar", scalar.dtype.str, scalar.tobytes())
        if type(value) in {dict, mapping_proxy_type}:
            items = [
                (signature(key), signature(item))
                for key, item in value.items()
            ]
            items.sort(key=repr)
            return (
                "mapping",
                type(value).__module__,
                type(value).__qualname__,
                tuple(items),
            )
        if type(value) in {tuple, list}:
            if type(value) is tuple:
                cached = cached_immutable(value)
                if cached is not None:
                    return cached
            result = (
                type(value).__name__,
                tuple(signature(item) for item in value),
            )
            if type(value) is tuple:
                retain_immutable(value, result)
            return result
        if type(value) in {set, frozenset}:
            if type(value) is frozenset:
                cached = cached_immutable(value)
                if cached is not None:
                    return cached
            items = sorted((signature(item) for item in value), key=repr)
            result = (type(value).__name__, tuple(items))
            if type(value) is frozenset:
                retain_immutable(value, result)
            return result
        if type(value) is bytes:
            return ("bytes", value)
        if value is None or type(value) in {bool, int, str}:
            return (type(value).__name__, value)
        if type(value) is float:
            return (
                "float",
                numpy_module.asarray(
                    value, dtype=numpy_module.float64
                ).tobytes(),
            )
        return (
            "identity",
            type(value).__module__,
            type(value).__qualname__,
            id(value),
        )

    return signature


_module_authority_signature = _make_module_authority_signer(
    np, MappingProxyType
)
del _make_module_authority_signer


def _require_immutable_authority_data(
    value: Any,
    *,
    label: str,
    _seen: set[int] | None = None,
) -> None:
    """Reject mutable leaves that an attribute epoch cannot observe."""

    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return
    if isinstance(value, np.generic):
        return
    if isinstance(value, re.Pattern):
        return
    if callable(value) or isinstance(
        value,
        (ModuleType, type, property, classmethod, staticmethod),
    ):
        return
    if type(value) in {tuple, frozenset}:
        for index, item in enumerate(value):
            _require_immutable_authority_data(
                item,
                label=f"{label}[{index}]",
                _seen=seen,
            )
        return
    if type(value) is MappingProxyType:
        for key, item in value.items():
            _require_immutable_authority_data(
                key,
                label=f"{label}.key",
                _seen=seen,
            )
            _require_immutable_authority_data(
                item,
                label=f"{label}[{key!r}]",
                _seen=seen,
            )
        return
    if type(value) is np.ndarray:
        current: Any = value
        while type(current) is np.ndarray:
            if current.flags.writeable:
                raise TypeError(f"{label} contains a writeable authority array")
            current = current.base
        if isinstance(current, memoryview):
            if not current.readonly:
                raise TypeError(
                    f"{label} contains a writeable authority buffer"
                )
            return
        if type(current) is bytes:
            return
        raise TypeError(
            f"{label} authority array is not backed by immutable bytes"
        )
    raise TypeError(
        f"{label} contains unsupported mutable authority data "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _capture_authority_array_metadata(
    *roots: Any,
    _array_type: type[Any] = np.ndarray,
    _mapping_proxy_type: type[Any] = MappingProxyType,
) -> tuple[
    tuple[Any, str, tuple[int, ...], tuple[int, ...], bool], ...
]:
    """Capture immutable ndarray identities and their mutable metadata."""

    arrays: list[
        tuple[Any, str, tuple[int, ...], tuple[int, ...], bool]
    ] = []
    seen: set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if type(value) is _array_type:
            arrays.append(
                (
                    value,
                    value.dtype.str,
                    tuple(int(size) for size in value.shape),
                    tuple(int(stride) for stride in value.strides),
                    bool(value.flags.c_contiguous),
                )
            )
            return
        if type(value) in {tuple, frozenset, list, set}:
            for member in value:
                visit(member)
            return
        if type(value) in {dict, _mapping_proxy_type}:
            for key, member in value.items():
                visit(key)
                visit(member)

    for root in roots:
        visit(root)
    return tuple(arrays)


def _require_authority_array_metadata(
    metadata: tuple[
        tuple[Any, str, tuple[int, ...], tuple[int, ...], bool], ...
    ],
    *,
    label: str,
    _array_type: type[Any] = np.ndarray,
    _memoryview_type: type[Any] = memoryview,
) -> None:
    """Reject in-place ndarray metadata/base changes on every boundary."""

    try:
        for array, dtype_string, shape, strides, c_contiguous in metadata:
            if (
                type(array) is not _array_type
                or array.dtype.str != dtype_string
                or array.shape != shape
                or array.strides != strides
                or bool(array.flags.c_contiguous) is not c_contiguous
                or array.flags.writeable
            ):
                raise ValueError(f"{label} ndarray metadata changed")
            current: Any = array
            while type(current) is _array_type:
                if current.flags.writeable:
                    raise ValueError(f"{label} ndarray base is writeable")
                current = current.base
            if not (
                type(current) is bytes
                or isinstance(current, _memoryview_type) and current.readonly
            ):
                raise ValueError(f"{label} ndarray base changed")
    except (AttributeError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(label):
            raise
        raise ValueError(f"{label} ndarray metadata changed") from exc


FORMULATION_ID = "E4_PL_QUALIFIED_S3_COMPANION_V1"
FORMULATION_SCHEMA = "anysolver.e4_pl_s3.linear.v1"
MITC3_PLUS_SOURCE_URL = (
    "https://web.mit.edu/kjb/www/Principal_Publications/"
    "The_MITC3%2B_shell_element_and_its_performance.pdf"
)
MITC3_PLUS_SOURCE_BYTES = 1_146_142
MITC3_PLUS_SOURCE_SHA256 = (
    "182F52217277B55E17627B8C41A3A4626ED91ED5378399088E0EA1748AD93EF0"
)
MITC3_PLUS_EQUATION_MAP = MappingProxyType(
    {
        "bubble_interpolation": "equations_6_to_8",
        "internal_tying_directions": "equations_12_to_15",
        "assumed_covariant_shear": "equations_16_and_17",
    }
)
MITC3_PLUS_NONLINEAR_SOURCE_URL = (
    "https://web.mit.edu/kjb/www/Principal_Publications/"
    "The_MITC3%2B_shell_element_in_geometric_nonlinear_analysis.pdf"
)
MITC3_PLUS_NONLINEAR_SOURCE_BYTES = 2_312_312
MITC3_PLUS_NONLINEAR_SOURCE_SHA256 = (
    "8006F7EBB9A8A3D72F29C356F6C9D76668A38CC5DC80A9B48D489FC2A87E082A"
)
MITC3_PLUS_NONLINEAR_EQUATION_MAP = MappingProxyType(
    {
        "current_geometry": "equations_7_to_9",
        "director_update": "equations_10_to_15",
        "quadratic_director_increment": "equations_16_to_21",
        "incremental_green_lagrange": "equations_22_to_28",
        "assumed_nonlinear_covariant_shear": "equations_29_to_31",
    }
)
EXTERNAL_COORDINATE_LAYOUT_ID = "S3_EXTERNAL18_BUBBLE2_PL3_V1"
NONLINEAR_STATE_SCHEMA = "anysolver.e4_pl_s3.committed_state.v2"
NONLINEAR_STATE_VERSION = 2
NONLINEAR_STATE_LAYOUT_ID = (
    "S3_TL_Q18_NODE_SO3_PL3_TRIADS4_BUBBLE2_STATION7_LAYERED_V2"
)
NONLINEAR_KINEMATICS_ID = (
    "MITC3_PLUS_TOTAL_LAGRANGIAN_SHARED_SO3_CORNERS_EQ7_31_V2"
)
DIRECTOR_GAUGE_ID = "MITC3_PLUS_EQ11_GLOBAL_EY_WITH_EZ_PARALLEL_FALLBACK_V1"
EXTERNAL_ROTATION_MAP_ID = (
    "GLOBAL_COORDINATES_WITH_SOLVER_OWNED_LEFT_MULTIPLICATIVE_SO3_V1"
)
DIRECTOR_INCREMENT_MAP_ID = (
    "CORNER_FROM_SOLVER_SHARED_SO3_BUBBLE_EQ14_HIERARCHICAL_V1"
)
SURFACE_ROTATION_POLICY_ID = "SURFACE_RIGHT_POLAR_FRAME_3X2_V1"
PL_TWIST_POLICY_ID = "RELATIVE_SWING_TWIST_ATAN2_UNWRAPPED_V1"
NODAL_ROTATION_UPDATE_POLICY_ID = "LEFT_MULTIPLICATIVE_SPATIAL_ROTATION_UPDATE_V1"
PL_PHASE_POLICY_ID = "NEAREST_COMMITTED_PHASE_UNIQUE_PI_MARGIN_V1"
NONLINEAR_PL_ENERGY_POLICY_ID = "BARYCENTRIC_PL_ENERGY_EXACT_HESSIAN_V1"
PL_PHASE_MARGIN_ID = "PI_MINUS_64_SQRT_BINARY64_EPSILON_V1"
PL_MINIMUM_TWIST_DENOMINATOR_ID = "64_SQRT_BINARY64_EPSILON_V1"
PL_PHASE_MARGIN = 64.0 * math.sqrt(np.finfo(np.float64).eps)
PL_MINIMUM_TWIST_DENOMINATOR = 64.0 * math.sqrt(np.finfo(np.float64).eps)
BUBBLE_STATE_ROLE = "RESERVED_ZERO_NEW_INCREMENT_PREDICTOR_ONLY"
BUBBLE_PREDICTOR_COMMIT_POLICY_ID = "RESET_TO_ZERO_AFTER_EVERY_ACCEPTED_STEP_V1"
COMMIT_STATUS = "committed_converged"
STATE_MODE = "layered_material"

# Pre-integrated generalized sections have no layer/material-point history and
# must never be serialized with the layered V2 schema.  Their state is a
# separately identified contract so restart cannot silently reinterpret
# section resultants as layer stresses (or vice versa).
GENERALIZED_NONLINEAR_STATE_SCHEMA = (
    "anysolver.e4_pl_s3.committed_state.generalized_section.v4"
)
GENERALIZED_NONLINEAR_STATE_VERSION = 4
GENERALIZED_NONLINEAR_STATE_LAYOUT_ID = (
    "S3_TL_Q18_NODE_SO3_PL3_TRIADS4_BUBBLE2_STATION7_"
    "GENERALIZED_STATELESS_INITIAL_FIELDS_V4"
)
GENERALIZED_STATE_MODE = "stateless_generalized_section"
GENERALIZED_ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA = (
    "anysolver.e4_pl_s3.element_configuration.generalized_section.v3"
)
GENERALIZED_SECTION_DESCRIPTOR_SCHEMA = (
    "anysolver.e4_pl_s3.generalized_section.binary64.v1"
)
GENERALIZED_STATE_FIELD_MANIFEST_ID = (
    "S3_GENERALIZED_STATELESS_STATION7_SHARED_SO3_INITIAL_FIELDS_V2"
)
GENERALIZED_SECTION_HISTORY_POLICY_ID = (
    "STATELESS_PREINTEGRATED_GENERALIZED_SECTION_ONLY_V1"
)

BUBBLE_CONVENTION = "hierarchical_rotation_relative_to_corner_average"
QUADRATURE_ID = "dunavant_degree5_7point"
NONLINEAR_POLICY_ID = (
    "TOTAL_LAGRANGIAN_MITC3_PLUS_SHARED_SO3_CORNER_BUBBLE_SCHUR_V2"
)
RECOVERY_POLICY_ID = "S3_NATIVE_PHYSICAL_DIRECTOR_POLARITY_RECOVERY_V2"
DIRECTOR_POLARITY_POLICY_ID = (
    "ELEMENT_OWNED_PHYSICAL_DIRECTOR_INDEPENDENT_OF_SHEET_OWNER_NORMAL_V1"
)
DIRECTOR_REVERSAL_TRANSFORM_ID = (
    "GENERALIZED_DIAG_I3_POLARITY_I5_WITH_LAYER_ORDER_REVERSAL_V1"
)
REFERENCE_SURFACE_OFFSET_POLICY_ID = (
    "SIGNED_SECTION_ORIGIN_TO_NODAL_REFERENCE_ALONG_PHYSICAL_DIRECTOR_V1"
)
REFERENCE_SURFACE_STRAIN_TRANSFORM_ID = (
    "EPSILON_SECTION_EQ_EPSILON_REFERENCE_MINUS_OFFSET_KAPPA_V1"
)
REFERENCE_SURFACE_MASS_SHIFT_ID = (
    "M1_REFERENCE_EQ_MINUS_OFFSET_M0_AND_M2_REFERENCE_EQ_M2_SECTION_PLUS_OFFSET_SQUARED_M0_V1"
)
NONLINEAR_SOURCE_SHA256 = MITC3_PLUS_NONLINEAR_SOURCE_SHA256
BUBBLE_OFFSET_D = 1.0e-4
BUBBLE_OFFSET_EXACT = "1/10000"
BUBBLE_POLYNOMIAL_SCALE = 27.0
TYING_POINTS = MappingProxyType(
    {
        "A": (1.0 / 6.0, 2.0 / 3.0),
        "B": (2.0 / 3.0, 1.0 / 6.0),
        "C": (1.0 / 6.0, 1.0 / 6.0),
        "D": (1.0 / 3.0 + BUBBLE_OFFSET_D, 1.0 / 3.0 - 2.0 * BUBBLE_OFFSET_D),
        "E": (1.0 / 3.0 - 2.0 * BUBBLE_OFFSET_D, 1.0 / 3.0 + BUBBLE_OFFSET_D),
        "F": (1.0 / 3.0 + BUBBLE_OFFSET_D, 1.0 / 3.0 + BUBBLE_OFFSET_D),
    }
)
TYING_POINT_DEFINITIONS = MappingProxyType(
    {
        "A": ("1/6", "2/3"),
        "B": ("2/3", "1/6"),
        "C": ("1/6", "1/6"),
        "D": ("1/3+d", "1/3-2*d"),
        "E": ("1/3-2*d", "1/3+d"),
        "F": ("1/3+d", "1/3+d"),
    }
)
PL_BASIS_ID = "BARYCENTRIC_L1_L2_L3_V1"
PL_CONSTRAINT_ID = "THETA_D_MINUS_ONE_HALF_V_X_MINUS_U_Y_AT_NODES_V1"
PL_GRAM_SCALE_ID = "AREA_OVER_12_V1"
PL_GRAM_NUMERATOR = ((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0))
PL_BLOCK_SIGN_ID = "KQT_EQ_C_TRANSPOSE_M_KTT_EQ_MINUS_M_OVER_KD_V1"
PL_CONDENSATION_ID = "KD_EQ_KD_C_TRANSPOSE_M_C_V1"
DRILL_SCALE_POLICY_ID = "HALF_MIN_GENERALIZED_EIGENVALUE_PTAP_AGAINST_G_V1"
DRILL_SCALE_PROJECTOR = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))
DRILL_SCALE_METRIC = ((2.0, 0.0), (0.0, 0.5))
DRILL_SCALE_INVERSE_METRIC_SQRT = (
    (1.0 / math.sqrt(2.0), 0.0),
    (0.0, math.sqrt(2.0)),
)
BUBBLE_MAX_ITERATIONS = 32
BUBBLE_RELATIVE_TOLERANCE = 5.0e-11
BUBBLE_STEP_TOLERANCE = 5.0e-13
BUBBLE_CONDITION_LIMIT = 1.0e14
BUBBLE_LINE_SEARCH_MIN_FACTOR = 1.0 / 128.0
BUBBLE_LINE_SEARCH_REDUCTION = 0.5
BUBBLE_FORCE_CONDENSATION_ID = "FQ_MINUS_KQA_SOLVE_KAA_FA_V1"
ISOTROPIC_CONSTITUTIVE_INTEGRATION_ID = "PLANE_STRESS_RETURN_MAP_CONSISTENT_V1"
ORTHOTROPIC_CONSTITUTIVE_INTEGRATION_ID = (
    "HILL48_PLANE_STRESS_RETURN_MAP_CONSISTENT_V1"
)
GENERALIZED_SECTION_INTEGRATION_ID = (
    "STATELESS_KINEMATIC_MINUS_INITIAL_EIGENSTRAIN_PLUS_INITIAL_RESULTANT_V2"
)

CANONICALIZATION_ID = "RFC8259_SORTED_UTF8_COMPACT_LF_SIGNED_ZERO_NORMALIZED_V1"
ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA = (
    "anysolver.e4_pl_s3.element_configuration.layered.v3"
)
MATERIAL_DESCRIPTOR_SCHEMA = "anysolver.e4_pl_s3.material.dataclass.v1"
MATERIAL_DESCRIPTOR_VALIDATION_ID = (
    "REGISTERED_ANYMATERIAL_DATACLASS_COMPLIANCE_CROSSCHECK_V1"
)
THICKNESS_QUADRATURE_ID = "GAUSS_LOBATTO_3_5_7_9_11_V1"
SUPPORTED_LOBATTO_LAYER_COUNTS = (3, 5, 7, 9, 11)
STATE_ARRAY_LAYOUT_ID = "S3_STATION_MAJOR_LAYER_BOTTOM_TO_TOP_MINOR_V1"
THICKNESS_COORDINATE_SIGN_ID = (
    "ZETA_MINUS_ONE_BOTTOM_OPPOSITE_DIRECTOR_TO_PLUS_ONE_TOP_ALONG_DIRECTOR_V1"
)
STATE_INTEGRITY_ID = "SHA256_CANONICAL_COMPLETE_STATE_EXCLUDING_DIGEST_V1"
STATE_REDUNDANCY_TOLERANCE_FACTOR = 128.0
STATE_REDUNDANCY_VALIDATION_ID = (
    "KINEMATIC_FROM_GENERALIZED_AND_INPLANE_RESULTANT_FROM_LAYER_STRESS_V1"
)
REFERENCE_GEOMETRY_VALIDATION_ID = (
    "QUALIFIED_ADMISSION_DERIVED_NUMBERED_FRAME_EXACT_IDENTITY_V1"
)
DIRECTOR_ORTHONORMALITY_TOLERANCE = 1.0e-12
DIRECTOR_GAUGE_SWITCH_TOLERANCE = 1.0e-12
DIRECTOR_GAUGE_PRIMARY_AXIS = (0.0, 1.0, 0.0)
DIRECTOR_GAUGE_FALLBACK_AXIS = (0.0, 0.0, 1.0)
MINIMUM_OWNER_NORMAL_ALIGNMENT = 1.0e-8
MINIMUM_NORMALIZED_TWICE_AREA = max(64.0 * np.finfo(np.float64).eps, 1.0e-14)
MINIMUM_ANGLE_DEG = 30.0
MAXIMUM_ANGLE_DEG = 150.0
MAXIMUM_EDGE_RATIO = 4.0
MINIMUM_CORNER_SCALED_JACOBIAN = 0.20
MINIMUM_NORMALIZED_AREA = 0.60
QUALITY_COMPARISON_TOLERANCE = 1.0e-12
REFERENCE_FRAME_MATCH_TOLERANCE = 1.0e-12
NUM_INTEGRATION_STATIONS = 7
GENERALIZED_COMPONENTS = 8
GENERALIZED_STRAIN_COMPONENT_ORDER = (
    "membrane_xx",
    "membrane_yy",
    "membrane_xy_engineering",
    "curvature_xx",
    "curvature_yy",
    "curvature_xy_engineering",
    "transverse_shear_xz_engineering",
    "transverse_shear_yz_engineering",
)
GENERALIZED_RESULTANT_COMPONENT_ORDER = (
    "membrane_force_xx",
    "membrane_force_yy",
    "membrane_force_xy_tensor",
    "bending_moment_xx",
    "bending_moment_yy",
    "bending_moment_xy_tensor",
    "transverse_shear_force_xz_tensor",
    "transverse_shear_force_yz_tensor",
)
LAYER_STRAIN_COMPONENT_ORDER = (
    "inplane_xx",
    "inplane_yy",
    "inplane_xy_engineering",
)
LAYER_STRESS_COMPONENT_ORDER = (
    "inplane_xx",
    "inplane_yy",
    "inplane_xy_tensor",
)
# Compatibility aliases are retained for research callers, but the fingerprint
# and field manifest below always use the unambiguous strain/resultant names.
GENERALIZED_COMPONENT_ORDER = GENERALIZED_STRAIN_COMPONENT_ORDER
LAYER_COMPONENT_ORDER = LAYER_STRAIN_COMPONENT_ORDER

EXTERNAL_NODE_COORDINATE_ORDER = (
    "translation_global_x",
    "translation_global_y",
    "translation_global_z",
    "additive_rotation_global_x",
    "additive_rotation_global_y",
    "additive_rotation_global_z",
)
EXTERNAL_NODE_FORCE_ORDER = (
    "force_global_x",
    "force_global_y",
    "force_global_z",
    "moment_global_x",
    "moment_global_y",
    "moment_global_z",
)

INITIAL_FIELD_NAMES = (
    "initial_membrane_stress",
    "initial_bending_stress",
    "initial_membrane_prestrain",
    "initial_curvature_prestrain",
)

STIFFNESS_STATION_TABLE = (
    (1.0 / 3.0, 1.0 / 3.0, 0.1125),
    (0.470142064105115, 0.470142064105115, 0.066197076394253),
    (0.059715871789770, 0.470142064105115, 0.066197076394253),
    (0.470142064105115, 0.059715871789770, 0.066197076394253),
    (0.101286507323456, 0.101286507323456, 0.062969590272414),
    (0.797426985353087, 0.101286507323456, 0.062969590272414),
    (0.101286507323456, 0.797426985353087, 0.062969590272414),
)

LOBATTO_NORMALIZED_TABLES = (
    (
        3,
        (-1.0, 0.0, 1.0),
        (0.3333333333333333, 1.3333333333333333, 0.3333333333333333),
    ),
    (
        5,
        (-1.0, -0.6546536707079771, 0.0, 0.6546536707079771, 1.0),
        (0.1, 0.5444444444444444, 0.7111111111111111, 0.5444444444444444, 0.1),
    ),
    (
        7,
        (-1.0, -0.830223896278567, -0.468848793470714, 0.0, 0.468848793470714, 0.830223896278567, 1.0),
        (0.047619047619048, 0.276826047361566, 0.431745381209863, 0.487619047619048, 0.431745381209863, 0.276826047361566, 0.047619047619048),
    ),
    (
        9,
        (-1.0, -0.89975799541146, -0.677186279510738, -0.363117463826178, 0.0, 0.363117463826178, 0.677186279510738, 0.89975799541146, 1.0),
        (0.027777777777778, 0.165495361560806, 0.274538712500162, 0.346428510973046, 0.371519274376417, 0.346428510973046, 0.274538712500162, 0.165495361560806, 0.027777777777778),
    ),
    (
        11,
        (-1.0, -0.934001430408059, -0.784483473663144, -0.565235326996205, -0.295758135586939, 0.0, 0.295758135586939, 0.565235326996205, 0.784483473663144, 0.934001430408059, 1.0),
        (0.018181818181818, 0.109612273266995, 0.187169881780305, 0.248048104264028, 0.286879124779008, 0.300217595455691, 0.286879124779008, 0.248048104264028, 0.187169881780305, 0.109612273266995, 0.018181818181818),
    ),
)

def _freeze_authority_tree(value: Any) -> Any:
    """Recursively freeze exact scientific mappings at module construction."""

    if type(value) is dict:
        return MappingProxyType(
            {
                key: _freeze_authority_tree(member)
                for key, member in value.items()
            }
        )
    if type(value) is tuple:
        return tuple(_freeze_authority_tree(member) for member in value)
    if type(value) is frozenset:
        return frozenset(_freeze_authority_tree(member) for member in value)
    return value


STATE_FIELD_MANIFEST = _freeze_authority_tree({
    "reference_corner_directors": {
        "role": "immutable_element_owned_reference_with_strict_polarity",
        "component_frame": "global",
        "component_order": ("x", "y", "z"),
        "point_order": "ordered_connectivity_nodes",
    },
    "committed_total_u": {
        "role": "authoritative",
        "component_frame": "global",
        "component_order": EXTERNAL_NODE_COORDINATE_ORDER,
        "point_order": "ordered_connectivity_nodes",
    },
    "committed_nodal_rotation_matrices": {
        "role": "authoritative_node_shared_redundant_copy",
        "component_frame": "global_operator_rows_columns",
        "component_order": ("global_x", "global_y", "global_z"),
        "point_order": "ordered_connectivity_nodes",
    },
    "committed_pl_twist": {
        "role": "authoritative_unwrapped_phase",
        "component_frame": "relative_surface_nodal_twist",
        "component_order": ("scalar_angle_radians",),
        "point_order": "ordered_connectivity_nodes",
    },
    "committed_pl_turn_count": {
        "role": "authoritative_unwrapped_phase_integer",
        "component_frame": "relative_surface_nodal_twist",
        "component_order": ("integer_turn_count",),
        "point_order": "ordered_connectivity_nodes",
    },
    "committed_pl_multiplier": {
        "role": "numerical_diagnostic",
        "component_frame": "barycentric_pl_constraint",
        "component_order": ("tau_equals_kd_times_twist",),
        "point_order": "ordered_connectivity_nodes",
    },
    "committed_pl_internal_force": {
        "role": "numerical_diagnostic",
        "component_frame": "global",
        "component_order": EXTERNAL_NODE_FORCE_ORDER,
        "point_order": "ordered_connectivity_nodes",
    },
    "committed_pl_energy": {
        "role": "numerical_diagnostic",
        "component_frame": "scalar",
        "component_order": ("energy",),
        "point_order": "single_element",
    },
    "committed_director_triads": {
        "role": "authoritative",
        "component_frame": "global_xyz_rows",
        "component_order": ("V1", "V2", "VN"),
        "point_order": "source_nodes_corner_1_2_3_then_bubble_4",
    },
    "bubble_rotation_last_increment": {
        "role": "reserved_zero",
        "component_frame": "current_source_node_4_eq11_tangent_gauge",
        "component_order": ("A4_hierarchical", "B4_hierarchical"),
        "point_order": "single_source_bubble_node",
    },
    "committed_internal_force": {
        "role": "derived_integrity_protected",
        "component_frame": "global",
        "component_order": EXTERNAL_NODE_FORCE_ORDER,
        "point_order": "ordered_connectivity_nodes",
    },
    "station_generalized_strain": {
        "role": "authoritative",
        "component_frame": "numbered_reference_engineering_strain",
        "component_order": GENERALIZED_STRAIN_COMPONENT_ORDER,
        "point_order": "ordered_surface_stations",
    },
    "station_generalized_resultant": {
        "role": "derived_integrity_protected",
        "component_frame": "numbered_reference_tensor_resultant",
        "component_order": GENERALIZED_RESULTANT_COMPONENT_ORDER,
        "point_order": "ordered_surface_stations",
    },
    "plastic_strain": {
        "role": "authoritative_material_history",
        "component_frame": "physical_material_engineering_strain",
        "component_order": LAYER_STRAIN_COMPONENT_ORDER,
        "point_order": "station_major_layer_minor_physical_director_bottom_to_top",
    },
    "alpha": {
        "role": "authoritative_material_history",
        "component_frame": "scalar",
        "component_order": ("equivalent_plastic_strain",),
        "point_order": "station_major_layer_minor_physical_director_bottom_to_top",
    },
    "kinematic_layer_strain": {
        "role": "authoritative_redundant_validated",
        "component_frame": "numbered_reference_engineering_strain",
        "component_order": LAYER_STRAIN_COMPONENT_ORDER,
        "point_order": "station_major_layer_minor_physical_director_bottom_to_top",
    },
    "layer_strain": {
        "role": "derived_integrity_protected",
        "component_frame": "numbered_reference_engineering_strain",
        "component_order": LAYER_STRAIN_COMPONENT_ORDER,
        "point_order": "station_major_layer_minor_physical_director_bottom_to_top",
    },
    "layer_strain_material": {
        "role": "derived_integrity_protected",
        "component_frame": "physical_material_engineering_strain",
        "component_order": LAYER_STRAIN_COMPONENT_ORDER,
        "point_order": "station_major_layer_minor_physical_director_bottom_to_top",
    },
    "layer_stress": {
        "role": "derived_integrity_protected",
        "component_frame": "numbered_reference_tensor_stress",
        "component_order": LAYER_STRESS_COMPONENT_ORDER,
        "point_order": "station_major_layer_minor_physical_director_bottom_to_top",
    },
    "layer_stress_material": {
        "role": "derived_integrity_protected",
        "component_frame": "physical_material_tensor_stress",
        "component_order": LAYER_STRESS_COMPONENT_ORDER,
        "point_order": "station_major_layer_minor_physical_director_bottom_to_top",
    },
    "initial_membrane_stress": {
        "role": "authoritative_initial_field",
        "component_frame": "numbered_reference_tensor_stress",
        "component_order": LAYER_STRESS_COMPONENT_ORDER,
        "point_order": "ordered_surface_stations",
    },
    "initial_bending_stress": {
        "role": "authoritative_initial_field",
        "component_frame": "numbered_reference_tensor_stress",
        "component_order": LAYER_STRESS_COMPONENT_ORDER,
        "point_order": "ordered_surface_stations",
    },
    "initial_membrane_prestrain": {
        "role": "authoritative_initial_field",
        "component_frame": "numbered_reference_engineering_strain",
        "component_order": LAYER_STRAIN_COMPONENT_ORDER,
        "point_order": "ordered_surface_stations",
    },
    "initial_curvature_prestrain": {
        "role": "authoritative_initial_field",
        "component_frame": "numbered_reference_engineering_curvature",
        "component_order": LAYER_STRAIN_COMPONENT_ORDER,
        "point_order": "ordered_surface_stations",
    },
    "initial_field_provenance": {
        "role": "authoritative_initial_field_metadata",
        "component_frame": "canonical_metadata",
        "component_order": ("sorted_string_keys",),
        "point_order": "not_applicable",
    },
})

# This manifest intentionally has no layer, ply-stress, plastic-strain, or
# alpha entries.  The four public shell initial fields are authoritative
# station inputs, while their analytically integrated generalized offsets are
# redundant and strictly validated.  No manufacturing layers are reconstructed.
GENERALIZED_STATE_FIELD_MANIFEST = _freeze_authority_tree({
    "reference_corner_directors": STATE_FIELD_MANIFEST[
        "reference_corner_directors"
    ],
    "committed_total_u": STATE_FIELD_MANIFEST["committed_total_u"],
    "committed_nodal_rotation_matrices": STATE_FIELD_MANIFEST[
        "committed_nodal_rotation_matrices"
    ],
    "committed_pl_twist": STATE_FIELD_MANIFEST["committed_pl_twist"],
    "committed_pl_turn_count": STATE_FIELD_MANIFEST[
        "committed_pl_turn_count"
    ],
    "committed_pl_multiplier": STATE_FIELD_MANIFEST[
        "committed_pl_multiplier"
    ],
    "committed_pl_internal_force": STATE_FIELD_MANIFEST[
        "committed_pl_internal_force"
    ],
    "committed_pl_energy": STATE_FIELD_MANIFEST["committed_pl_energy"],
    "committed_director_triads": STATE_FIELD_MANIFEST[
        "committed_director_triads"
    ],
    "bubble_rotation_last_increment": STATE_FIELD_MANIFEST[
        "bubble_rotation_last_increment"
    ],
    "committed_internal_force": STATE_FIELD_MANIFEST[
        "committed_internal_force"
    ],
    "station_generalized_strain": STATE_FIELD_MANIFEST[
        "station_generalized_strain"
    ],
    "station_generalized_resultant": STATE_FIELD_MANIFEST[
        "station_generalized_resultant"
    ],
    "initial_membrane_stress": STATE_FIELD_MANIFEST[
        "initial_membrane_stress"
    ],
    "initial_bending_stress": STATE_FIELD_MANIFEST[
        "initial_bending_stress"
    ],
    "initial_membrane_prestrain": STATE_FIELD_MANIFEST[
        "initial_membrane_prestrain"
    ],
    "initial_curvature_prestrain": STATE_FIELD_MANIFEST[
        "initial_curvature_prestrain"
    ],
    "initial_field_provenance": STATE_FIELD_MANIFEST[
        "initial_field_provenance"
    ],
    "initial_generalized_prestrain": {
        "role": "derived_initial_field_redundant_validated",
        "component_frame": "numbered_reference_engineering_strain",
        "component_order": GENERALIZED_STRAIN_COMPONENT_ORDER,
        "point_order": "ordered_surface_stations",
    },
    "initial_generalized_resultant": {
        "role": "derived_initial_field_redundant_validated",
        "component_frame": "numbered_reference_tensor_resultant",
        "component_order": GENERALIZED_RESULTANT_COMPONENT_ORDER,
        "point_order": "ordered_surface_stations",
    },
})

_ROOT_MATERIAL_FIELDS = _freeze_authority_tree({
    "anymaterial.isotropic.IsotropicMaterial": frozenset(
        {
            "name",
            "elastic_modulus",
            "poisson_ratio",
            "density",
            "yield_stress",
            "hardening_curve",
        }
    ),
    "anymaterial.orthotropic.OrthotropicMaterial": frozenset(
        {
            "name",
            "elastic_modulus_1",
            "elastic_modulus_2",
            "elastic_modulus_3",
            "poisson_ratio_12",
            "poisson_ratio_13",
            "poisson_ratio_23",
            "shear_modulus_12",
            "shear_modulus_13",
            "shear_modulus_23",
            "density",
            "hill_yield",
            "hardening_curve",
        }
    ),
})
_MATERIAL_SYMMETRY_BY_TYPE = _freeze_authority_tree({
    "anymaterial.isotropic.IsotropicMaterial": "isotropic",
    "anymaterial.orthotropic.OrthotropicMaterial": "orthotropic",
})
_HARDENING_CURVE_FIELDS = _freeze_authority_tree({
    "anymaterial.curves.LinearHardeningCurve": frozenset(
        {"sigma_yield", "hardening_modulus_value"}
    ),
    "anymaterial.curves.PiecewiseLinearCurve": frozenset(
        {"plastic_strain", "flow_stress_values"}
    ),
    "anymaterial.curves.PowerLawHardeningCurve": frozenset({"K", "n", "eps_0"}),
    "anymaterial.curves.DNVC208MaterialCurve": frozenset(
        {"sigma_prop", "sigma_yield", "sigma_yield_2", "eps_p_y1", "eps_p_y2", "K", "n"}
    ),
})
_HILL48_TYPE_ID = "anymaterial.yield_criteria.Hill48Yield"
_HILL48_FIELDS = frozenset({"X", "Y", "Z", "S12", "S13", "S23"})

_HEX_SHA256 = re.compile(r"[0-9A-F]{64}\Z")
_IDENTITY_KEYS = frozenset(
    {
        "formulation_fingerprint",
        "element_id",
        "element_configuration_fingerprint",
        "node_ids",
        "node_order_fingerprint",
        "reference_geometry_fingerprint",
        "reference_frame_fingerprint",
        "reference_corner_directors_fingerprint",
        "director_polarity",
        "material_fingerprint",
        "initial_fields_fingerprint",
        "state_mode",
        "thickness_quadrature_id",
        "thickness",
        "num_layers",
        "material_symmetry",
        "equivalent_stress_measure",
    }
)
_ARRAY_SHAPES_FIXED = _freeze_authority_tree({
    "reference_corner_directors": (3, 3),
    "committed_total_u": (18,),
    "committed_nodal_rotation_matrices": (3, 3, 3),
    "committed_pl_twist": (3,),
    "committed_pl_multiplier": (3,),
    "committed_pl_internal_force": (18,),
    "committed_director_triads": (4, 3, 3),
    "bubble_rotation_last_increment": (2,),
    "committed_internal_force": (18,),
    "station_generalized_strain": (NUM_INTEGRATION_STATIONS, GENERALIZED_COMPONENTS),
    "station_generalized_resultant": (NUM_INTEGRATION_STATIONS, GENERALIZED_COMPONENTS),
    **{name: (NUM_INTEGRATION_STATIONS, 3) for name in INITIAL_FIELD_NAMES},
})
_LAYER_VECTOR_FIELDS = (
    "plastic_strain",
    "layer_strain",
    "layer_strain_material",
    "kinematic_layer_strain",
    "layer_stress",
    "layer_stress_material",
)
_STATE_KEYS = frozenset(
    {
        "state_schema",
        "state_version",
        "commit_status",
        "state_mode",
        "formulation_id",
        "formulation_schema",
        "external_coordinate_layout_id",
        "nonlinear_state_layout_id",
        "nonlinear_kinematics_id",
        "director_gauge_id",
        "external_rotation_map_id",
        "director_increment_map_id",
        "surface_rotation_policy_id",
        "pl_twist_policy_id",
        "nodal_rotation_update_policy_id",
        "pl_phase_policy_id",
        "pl_phase_margin_id",
        "pl_minimum_twist_denominator_id",
        "nonlinear_pl_energy_policy_id",
        "bubble_convention",
        "bubble_state_role",
        "bubble_predictor_commit_policy_id",
        "quadrature_id",
        "nonlinear_policy_id",
        "recovery_policy_id",
        "canonicalization_id",
        "state_integrity_id",
        "state_array_layout_id",
        "thickness_coordinate_sign_id",
        "stiffness_station_table_sha256",
        "lobatto_table_sha256",
        "state_field_manifest_sha256",
        "formulation_fingerprint",
        "state_integrity_sha256",
        "element_id",
        "element_configuration_fingerprint",
        "node_ids",
        "node_order_fingerprint",
        "reference_geometry_fingerprint",
        "reference_corner_directors_fingerprint",
        "director_polarity",
        "material_fingerprint",
        "initial_fields_fingerprint",
        "reference_frame_fingerprint",
        "thickness_quadrature_id",
        "thickness",
        "num_layers",
        "material_symmetry",
        "equivalent_stress_measure",
        "committed_total_u",
        "reference_corner_directors",
        "committed_nodal_rotation_matrices",
        "committed_pl_twist",
        "committed_pl_turn_count",
        "committed_pl_multiplier",
        "committed_pl_internal_force",
        "committed_pl_energy",
        "committed_director_triads",
        "bubble_rotation_last_increment",
        "committed_internal_force",
        "station_generalized_strain",
        "station_generalized_resultant",
        "plastic_strain",
        "alpha",
        "layer_strain",
        "layer_strain_material",
        "kinematic_layer_strain",
        "layer_stress",
        "layer_stress_material",
        *INITIAL_FIELD_NAMES,
        "initial_field_provenance",
    }
)

_GENERALIZED_IDENTITY_KEYS = frozenset(
    {
        "generalized_formulation_fingerprint",
        "element_id",
        "element_configuration_fingerprint",
        "node_ids",
        "node_order_fingerprint",
        "reference_geometry_fingerprint",
        "reference_frame_fingerprint",
        "reference_corner_directors_fingerprint",
        "director_polarity",
        "section_fingerprint",
        "initial_fields_fingerprint",
        "thickness",
        "state_mode",
    }
)
_GENERALIZED_ARRAY_SHAPES = _freeze_authority_tree({
    "reference_corner_directors": (3, 3),
    "committed_total_u": (18,),
    "committed_nodal_rotation_matrices": (3, 3, 3),
    "committed_pl_twist": (3,),
    "committed_pl_multiplier": (3,),
    "committed_pl_internal_force": (18,),
    "committed_director_triads": (4, 3, 3),
    "bubble_rotation_last_increment": (2,),
    "committed_internal_force": (18,),
    "station_generalized_strain": (
        NUM_INTEGRATION_STATIONS,
        GENERALIZED_COMPONENTS,
    ),
    "station_generalized_resultant": (
        NUM_INTEGRATION_STATIONS,
        GENERALIZED_COMPONENTS,
    ),
    "initial_generalized_prestrain": (
        NUM_INTEGRATION_STATIONS,
        GENERALIZED_COMPONENTS,
    ),
    "initial_generalized_resultant": (
        NUM_INTEGRATION_STATIONS,
        GENERALIZED_COMPONENTS,
    ),
    **{name: (NUM_INTEGRATION_STATIONS, 3) for name in INITIAL_FIELD_NAMES},
})
_GENERALIZED_STATE_KEYS = frozenset(
    {
        "state_schema",
        "state_version",
        "commit_status",
        "state_mode",
        "formulation_id",
        "formulation_schema",
        "external_coordinate_layout_id",
        "nonlinear_state_layout_id",
        "nonlinear_kinematics_id",
        "director_gauge_id",
        "external_rotation_map_id",
        "director_increment_map_id",
        "surface_rotation_policy_id",
        "pl_twist_policy_id",
        "nodal_rotation_update_policy_id",
        "pl_phase_policy_id",
        "pl_phase_margin_id",
        "pl_minimum_twist_denominator_id",
        "nonlinear_pl_energy_policy_id",
        "bubble_convention",
        "bubble_state_role",
        "bubble_predictor_commit_policy_id",
        "quadrature_id",
        "nonlinear_policy_id",
        "generalized_section_integration_id",
        "generalized_section_history_policy_id",
        "generalized_initial_field_policy_id",
        "recovery_scope",
        "physical_layer_recovery_available",
        "canonicalization_id",
        "state_integrity_id",
        "stiffness_station_table_sha256",
        "generalized_state_field_manifest_sha256",
        "generalized_formulation_fingerprint",
        "state_integrity_sha256",
        "element_id",
        "element_configuration_fingerprint",
        "node_ids",
        "node_order_fingerprint",
        "reference_geometry_fingerprint",
        "reference_frame_fingerprint",
        "reference_corner_directors_fingerprint",
        "director_polarity",
        "section_fingerprint",
        "initial_fields_fingerprint",
        "thickness",
        "reference_corner_directors",
        "committed_total_u",
        "committed_nodal_rotation_matrices",
        "committed_pl_twist",
        "committed_pl_turn_count",
        "committed_pl_multiplier",
        "committed_pl_internal_force",
        "committed_pl_energy",
        "committed_director_triads",
        "bubble_rotation_last_increment",
        "committed_internal_force",
        "station_generalized_strain",
        "station_generalized_resultant",
        "initial_generalized_prestrain",
        "initial_generalized_resultant",
        *INITIAL_FIELD_NAMES,
        "initial_field_provenance",
    }
)


class S3CommittedStateError(ValueError):
    """Raised when qualified-S3 committed state is malformed or incompatible."""


def _key_labels(values: set[Any] | frozenset[Any]) -> list[str]:
    return sorted(repr(value) for value in values)


def _canonical_value(value: Any, *, path: str = "$") -> Any:
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _canonical_value(value.item(), path=path)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise S3CommittedStateError(f"nonfinite canonical value at {path}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, member in value.items():
            if not isinstance(key, str):
                raise S3CommittedStateError(f"non-string canonical key at {path}")
            if key in result:
                raise S3CommittedStateError(f"duplicate canonical key {key!r} at {path}")
            result[key] = _canonical_value(member, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_value(member, path=f"{path}[{index}]")
            for index, member in enumerate(value)
        ]
    raise S3CommittedStateError(
        f"unsupported canonical value {type(value).__name__} at {path}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes with one terminal LF."""

    normalized = _canonical_value(value)
    text = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def canonical_plain_data(value: Any) -> Any:
    """Materialize one owned built-in JSON tree under the canonical rules.

    Consumers must evaluate mechanics from this snapshot rather than retain a
    caller-controlled ``Mapping`` whose ``items`` and ``get`` views can
    disagree or change after evidence hashing.
    """

    return _canonical_value(value)


def owned_plain_snapshot(value: Any, *, path: str = "$") -> Any:
    """Detach a caller-owned tree without changing any scalar or array bits.

    Unlike canonical evidence normalization, committed mechanics identities
    may bind signed zero and other exact binary64 payloads.  This snapshot
    consumes each mapping/sequence once, rejects duplicate keys, and copies
    arrays into immutable byte-backed storage while preserving dtype, shape,
    and bytes exactly.  Call :func:`canonical_json_bytes` separately to apply
    the canonical schema/nonfinite checks.
    """

    if isinstance(value, np.ndarray):
        made = np.ascontiguousarray(np.asarray(value))
        return np.frombuffer(
            made.tobytes(order="C"), dtype=made.dtype
        ).reshape(made.shape)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, member in value.items():
            if not isinstance(key, str):
                raise S3CommittedStateError(
                    f"non-string snapshot key at {path}"
                )
            if key in result:
                raise S3CommittedStateError(
                    f"duplicate snapshot key {key!r} at {path}"
                )
            result[key] = owned_plain_snapshot(member, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            owned_plain_snapshot(member, path=f"{path}[{index}]")
            for index, member in enumerate(value)
        ]
    raise S3CommittedStateError(
        f"unsupported snapshot value {type(value).__name__} at {path}"
    )


def canonical_sha256(value: Any) -> str:
    """Return the uppercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def _binary64_rows(rows: Sequence[Sequence[float]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(float(value).hex() for value in row) for row in rows)


def stiffness_station_table_fingerprint() -> str:
    """Bind exact ordered binary64 ``(r, s, weight)`` surface stations."""

    return canonical_sha256(
        {
            "quadrature_id": QUADRATURE_ID,
            "ordered_r_s_weight_binary64": _binary64_rows(STIFFNESS_STATION_TABLE),
        }
    )


def lobatto_table_fingerprint() -> str:
    """Bind every ordered normalized Lobatto point and weight."""

    return canonical_sha256(
        {
            "quadrature_id": THICKNESS_QUADRATURE_ID,
            "order": STATE_ARRAY_LAYOUT_ID,
            "thickness_sign": THICKNESS_COORDINATE_SIGN_ID,
            "rules": tuple(
                {
                    "num_layers": count,
                    "zeta_binary64": tuple(float(value).hex() for value in points),
                    "weight_binary64": tuple(float(value).hex() for value in weights),
                }
                for count, points, weights in LOBATTO_NORMALIZED_TABLES
            ),
        }
    )


def state_field_manifest_fingerprint() -> str:
    return canonical_sha256(
        {"layout_id": STATE_ARRAY_LAYOUT_ID, "fields": STATE_FIELD_MANIFEST}
    )


def generalized_state_field_manifest_fingerprint() -> str:
    """Hash the closed field set for stateless generalized-section history."""

    return canonical_sha256(
        {
            "layout_id": GENERALIZED_STATE_FIELD_MANIFEST_ID,
            "fields": GENERALIZED_STATE_FIELD_MANIFEST,
        }
    )


def qualified_s3_lobatto_layers(
    num_layers: Any, thickness: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen S3 Lobatto rule with strict count and thickness types."""

    layers = _layer_count(num_layers, "num_layers")
    if isinstance(thickness, (bool, np.bool_)) or not isinstance(
        thickness, (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError("thickness must be a finite positive scalar")
    made_thickness = float(thickness)
    if not math.isfinite(made_thickness) or made_thickness <= 0.0:
        raise S3CommittedStateError("thickness must be a finite positive scalar")
    for count, normalized_points, normalized_weights in LOBATTO_NORMALIZED_TABLES:
        if count == layers:
            points = 0.5 * made_thickness * np.asarray(
                normalized_points, dtype=np.float64
            )
            weights = 0.5 * made_thickness * np.asarray(
                normalized_weights, dtype=np.float64
            )
            return points, weights
    raise AssertionError("supported S3 Lobatto table is missing")


def _state_integrity_sha256(state: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in state.items() if key != "state_integrity_sha256"}
    return canonical_sha256({"integrity_id": STATE_INTEGRITY_ID, "state": payload})


def seal_committed_s3_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return an owned state carrying a digest of every persisted value."""

    if not isinstance(state, Mapping):
        raise S3CommittedStateError("qualified S3 committed state must be a mapping")
    sealed = copy.deepcopy(dict(state))
    sealed.pop("state_integrity_sha256", None)
    if "bubble_rotation_last_increment" in sealed:
        predictor = _finite_state_array(
            sealed["bubble_rotation_last_increment"],
            (2,),
            "bubble_rotation_last_increment",
        )
        if np.any(predictor != 0.0):
            raise S3CommittedStateError(
                "committed bubble predictor must reset to exact zero"
            )
        sealed["bubble_rotation_last_increment"] = predictor
    sealed["state_integrity_sha256"] = _state_integrity_sha256(sealed)
    return sealed


def strict_canonical_json_loads(raw: bytes) -> Any:
    """Decode canonical bytes while rejecting duplicate or alternate JSON.

    This is a codec primitive, not a checkpoint loader.  Callers must still
    pass a decoded state through :func:`validate_committed_s3_state` with the
    current model identity.
    """

    if not isinstance(raw, bytes):
        raise S3CommittedStateError("canonical JSON input must be bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise S3CommittedStateError("canonical JSON must not contain a UTF-8 BOM")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise S3CommittedStateError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise S3CommittedStateError(f"nonfinite JSON constant {value}")

    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
        canonical = canonical_json_bytes(value)
    except S3CommittedStateError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise S3CommittedStateError("invalid canonical JSON") from exc
    if raw != canonical:
        raise S3CommittedStateError("JSON bytes are not in canonical form")
    return value


def _closed_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    non_string = {key for key in value if not isinstance(key, str)}
    if non_string:
        raise S3CommittedStateError(
            f"{label} keys must be strings: " + ", ".join(_key_labels(non_string))
        )
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise S3CommittedStateError(
            f"{label} keys mismatch; missing={_key_labels(missing)}, "
            f"unknown={_key_labels(unknown)}"
        )


def _qualified_dataclass_descriptor(
    value: Any,
    *,
    path: str,
    _post_observation: Optional[Callable[[], None]] = None,
) -> Any:
    """Return an exact structural descriptor without object stringification."""

    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        made = _canonical_value(value, path=path)
        if _post_observation is not None:
            _post_observation()
        return made
    if isinstance(value, np.ndarray):
        made = _canonical_value(value, path=path)
        if _post_observation is not None:
            _post_observation()
        return made
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        observed = tuple(value)
        if _post_observation is not None:
            _post_observation()
        return [
            _qualified_dataclass_descriptor(
                member,
                path=f"{path}[{index}]",
                _post_observation=_post_observation,
            )
            for index, member in enumerate(observed)
        ]
    if isinstance(value, Mapping):
        canonical = _canonical_value(value, path=path)
        if _post_observation is not None:
            _post_observation()
        assert isinstance(canonical, dict)
        return canonical
    if not is_dataclass(value) or isinstance(value, type):
        raise S3CommittedStateError(
            f"unsupported state-bearing object {type(value).__module__}."
            f"{type(value).__qualname__} at {path}"
        )
    fields = tuple(dataclass_fields(value))
    if _post_observation is not None:
        _post_observation()
    value_type = type(value)
    type_id = f"{value_type.__module__}.{value_type.__qualname__}"
    members: dict[str, Any] = {}
    for field in fields:
        observed = getattr(value, field.name)
        if _post_observation is not None:
            _post_observation()
        members[field.name] = _qualified_dataclass_descriptor(
            observed,
            path=f"{path}.{field.name}",
            _post_observation=_post_observation,
        )
    return {"type_id": type_id, "fields": members}


def _validate_nested_dataclass_descriptor(
    value: Any,
    allowed: Mapping[str, frozenset[str]],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S3CommittedStateError(f"{label} must be a registered dataclass descriptor")
    _closed_keys(value, frozenset({"type_id", "fields"}), label)
    type_id = value["type_id"]
    fields_value = value["fields"]
    if not isinstance(type_id, str) or type_id not in allowed:
        raise S3CommittedStateError(f"unsupported {label} type_id")
    if not isinstance(fields_value, Mapping):
        raise S3CommittedStateError(f"{label} fields must be a mapping")
    _closed_keys(fields_value, allowed[type_id], f"{label} fields")
    canonical = _canonical_value(dict(value), path=f"$.{label}")
    assert isinstance(canonical, dict)
    return canonical


def _validate_resolved_material_tree(value: Any) -> tuple[dict[str, Any], str, str]:
    root = _validate_nested_dataclass_descriptor(
        value,
        _ROOT_MATERIAL_FIELDS,
        label="resolved_material",
    )
    type_id = root["type_id"]
    fields_value = root["fields"]
    assert isinstance(fields_value, dict)
    hardening = fields_value["hardening_curve"]
    if hardening is not None:
        fields_value["hardening_curve"] = _validate_nested_dataclass_descriptor(
            hardening,
            _HARDENING_CURVE_FIELDS,
            label="hardening_curve",
        )
    symmetry = _MATERIAL_SYMMETRY_BY_TYPE[type_id]
    if symmetry == "orthotropic":
        hill = fields_value["hill_yield"]
        if hill is not None:
            fields_value["hill_yield"] = _validate_nested_dataclass_descriptor(
                hill,
                {_HILL48_TYPE_ID: _HILL48_FIELDS},
                label="hill_yield",
            )
        if hardening is not None and hill is None:
            raise S3CommittedStateError(
                "orthotropic hardening requires a registered Hill48 yield descriptor"
            )
        measure = "hill48" if hill is not None else "von_mises"
    else:
        measure = "von_mises"
    return root, symmetry, measure


def _owned_registered_hardening_curve(value: Any) -> Any:
    """Rebuild one registered hardening law from canonical owned fields."""

    if value is None:
        return None
    if type(value) is not dict:
        raise S3CommittedStateError(
            "hardening_curve must be a canonical registered descriptor"
        )
    type_id = value["type_id"]
    fields_value = value["fields"]
    if type(fields_value) is not dict:
        raise S3CommittedStateError("hardening_curve fields must be canonical")
    if type_id == "anymaterial.curves.LinearHardeningCurve":
        return LinearHardeningCurve(
            sigma_yield=float(fields_value["sigma_yield"]),
            hardening_modulus_value=float(
                fields_value["hardening_modulus_value"]
            ),
        )
    if type_id == "anymaterial.curves.PiecewiseLinearCurve":
        return PiecewiseLinearCurve(
            plastic_strain=tuple(
                float(member) for member in fields_value["plastic_strain"]
            ),
            flow_stress_values=tuple(
                float(member) for member in fields_value["flow_stress_values"]
            ),
        )
    if type_id == "anymaterial.curves.PowerLawHardeningCurve":
        return PowerLawHardeningCurve(
            K=float(fields_value["K"]),
            n=float(fields_value["n"]),
            eps_0=float(fields_value["eps_0"]),
        )
    if type_id == "anymaterial.curves.DNVC208MaterialCurve":
        return DNVC208MaterialCurve(
            sigma_prop=float(fields_value["sigma_prop"]),
            sigma_yield=float(fields_value["sigma_yield"]),
            sigma_yield_2=float(fields_value["sigma_yield_2"]),
            eps_p_y1=float(fields_value["eps_p_y1"]),
            eps_p_y2=float(fields_value["eps_p_y2"]),
            K=float(fields_value["K"]),
            n=float(fields_value["n"]),
        )
    raise S3CommittedStateError("unsupported hardening_curve type_id")


def _owned_registered_hill_yield(value: Any) -> Any:
    """Rebuild the optional Hill-48 authority from canonical owned fields."""

    if value is None:
        return None
    if type(value) is not dict or type(value.get("fields")) is not dict:
        raise S3CommittedStateError(
            "hill_yield must be a canonical registered descriptor"
        )
    if value.get("type_id") != _HILL48_TYPE_ID:
        raise S3CommittedStateError("unsupported hill_yield type_id")
    fields_value = value["fields"]
    return Hill48Yield(
        X=float(fields_value["X"]),
        Y=float(fields_value["Y"]),
        Z=float(fields_value["Z"]),
        S12=float(fields_value["S12"]),
        S13=float(fields_value["S13"]),
        S23=float(fields_value["S23"]),
    )


def _owned_material_from_resolved_descriptor(value: Any) -> StructuralMaterial:
    """Rebuild a solver-owned registered material from its closed descriptor."""

    resolved, _symmetry, _measure = _validate_resolved_material_tree(
        copy.deepcopy(value)
    )
    type_id = resolved["type_id"]
    fields_value = resolved["fields"]
    assert type(fields_value) is dict
    hardening_curve = _owned_registered_hardening_curve(
        fields_value["hardening_curve"]
    )
    if type_id == "anymaterial.isotropic.IsotropicMaterial":
        return IsotropicMaterial(
            name=str(fields_value["name"]),
            elastic_modulus=float(fields_value["elastic_modulus"]),
            poisson_ratio=float(fields_value["poisson_ratio"]),
            density=float(fields_value["density"]),
            yield_stress=float(fields_value["yield_stress"]),
            hardening_curve=hardening_curve,
        )
    if type_id == "anymaterial.orthotropic.OrthotropicMaterial":
        return OrthotropicMaterial(
            name=str(fields_value["name"]),
            elastic_modulus_1=float(fields_value["elastic_modulus_1"]),
            elastic_modulus_2=float(fields_value["elastic_modulus_2"]),
            elastic_modulus_3=float(fields_value["elastic_modulus_3"]),
            poisson_ratio_12=float(fields_value["poisson_ratio_12"]),
            poisson_ratio_13=float(fields_value["poisson_ratio_13"]),
            poisson_ratio_23=float(fields_value["poisson_ratio_23"]),
            shear_modulus_12=float(fields_value["shear_modulus_12"]),
            shear_modulus_13=float(fields_value["shear_modulus_13"]),
            shear_modulus_23=float(fields_value["shear_modulus_23"]),
            density=float(fields_value["density"]),
            hill_yield=_owned_registered_hill_yield(fields_value["hill_yield"]),
            hardening_curve=hardening_curve,
        )
    raise S3CommittedStateError("unsupported resolved material type_id")


def _validated_compliance(value: Any) -> np.ndarray:
    matrix = _finite_array(
        value,
        (6, 6),
        "elastic_compliance_engineering_6x6",
    )
    scale = max(float(np.max(np.abs(matrix))), np.finfo(np.float64).tiny)
    if float(np.max(np.abs(matrix - matrix.T))) > 1.0e-12 * scale:
        raise S3CommittedStateError("elastic compliance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if not np.all(np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) <= 0.0:
        raise S3CommittedStateError("elastic compliance must be positive definite")
    return matrix


def _compliance_from_registered_fields(resolved: Mapping[str, Any]) -> np.ndarray:
    type_id = resolved["type_id"]
    fields_value = resolved["fields"]
    assert isinstance(type_id, str) and isinstance(fields_value, Mapping)
    if type_id == "anymaterial.isotropic.IsotropicMaterial":
        elastic_modulus = float(fields_value["elastic_modulus"])
        poisson_ratio = float(fields_value["poisson_ratio"])
        shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
        matrix = np.asarray(
            (
                (1.0 / elastic_modulus, -poisson_ratio / elastic_modulus, -poisson_ratio / elastic_modulus, 0.0, 0.0, 0.0),
                (-poisson_ratio / elastic_modulus, 1.0 / elastic_modulus, -poisson_ratio / elastic_modulus, 0.0, 0.0, 0.0),
                (-poisson_ratio / elastic_modulus, -poisson_ratio / elastic_modulus, 1.0 / elastic_modulus, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0 / shear_modulus, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 1.0 / shear_modulus, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / shear_modulus),
            ),
            dtype=np.float64,
        )
    elif type_id == "anymaterial.orthotropic.OrthotropicMaterial":
        elastic_1 = float(fields_value["elastic_modulus_1"])
        elastic_2 = float(fields_value["elastic_modulus_2"])
        elastic_3 = float(fields_value["elastic_modulus_3"])
        poisson_12 = float(fields_value["poisson_ratio_12"])
        poisson_13 = float(fields_value["poisson_ratio_13"])
        poisson_23 = float(fields_value["poisson_ratio_23"])
        matrix = np.zeros((6, 6), dtype=np.float64)
        matrix[0, 0] = 1.0 / elastic_1
        matrix[1, 1] = 1.0 / elastic_2
        matrix[2, 2] = 1.0 / elastic_3
        matrix[0, 1] = matrix[1, 0] = -poisson_12 / elastic_1
        matrix[0, 2] = matrix[2, 0] = -poisson_13 / elastic_1
        matrix[1, 2] = matrix[2, 1] = -poisson_23 / elastic_2
        matrix[3, 3] = 1.0 / float(fields_value["shear_modulus_23"])
        matrix[4, 4] = 1.0 / float(fields_value["shear_modulus_13"])
        matrix[5, 5] = 1.0 / float(fields_value["shear_modulus_12"])
    else:
        raise S3CommittedStateError("unsupported resolved material type_id")
    return _validated_compliance(matrix)


def resolved_material_descriptor(
    material: Any,
    *,
    _post_observation: Optional[Callable[[], None]] = None,
) -> dict[str, Any]:
    """Describe every dataclass field of one supported material exactly."""

    if (
        not is_dataclass(material)
        or isinstance(material, type)
        or not isinstance(material, StructuralMaterial)
    ):
        raise S3CommittedStateError(
            "qualified S3 layered state requires a registered dataclass "
            "StructuralMaterial"
        )
    descriptor = _qualified_dataclass_descriptor(
        material,
        path="$.material",
        _post_observation=_post_observation,
    )
    assert isinstance(descriptor, dict)
    descriptor, registered_symmetry, measure = _validate_resolved_material_tree(
        descriptor
    )
    try:
        actual_symmetry = material_symmetry(material)
        if _post_observation is not None:
            _post_observation()
        owned_material = _owned_material_from_resolved_descriptor(descriptor)
        if _post_observation is not None:
            _post_observation()
        compliance = _validated_compliance(
            elastic_compliance_matrix(owned_material)
        )
        if _post_observation is not None:
            _post_observation()
    except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise S3CommittedStateError(
            "qualified S3 material does not provide valid structural compliance"
        ) from exc
    if actual_symmetry != registered_symmetry:
        raise S3CommittedStateError(
            "material symmetry does not match its registered descriptor type"
        )
    return {
        "descriptor_schema": MATERIAL_DESCRIPTOR_SCHEMA,
        "material_symmetry": registered_symmetry,
        "equivalent_stress_measure": measure,
        "elastic_compliance_engineering_6x6": compliance.tolist(),
        "resolved_material": descriptor,
    }


_GENERALIZED_SECTION_HISTORY_MARKERS = frozenset(
    {
        "committed_state",
        "history_schema",
        "init_nonlinear_state",
        "initialize_state",
        "requires_history",
        "state_layout_id",
        "state_schema",
        "update_nonlinear_state",
        "update_state",
    }
)


def require_stateless_generalized_section(section: Any) -> None:
    """Reject section-owned history until an authoritative protocol exists.

    ``GeneralizedShellSection`` is a frozen constitutive snapshot.  Structural
    input objects and mappings are accepted by the linear API, but a history
    marker must not be silently discarded by coercion when nonlinear state is
    enabled.  There is deliberately no fallback to invented layer history.
    """

    if section is None:
        return
    if isinstance(section, Mapping):
        markers = sorted(
            str(key)
            for key in section
            if str(key) in _GENERALIZED_SECTION_HISTORY_MARKERS
        )
    else:
        markers = sorted(
            name
            for name in _GENERALIZED_SECTION_HISTORY_MARKERS
            if hasattr(section, name)
        )
    if markers:
        raise S3CommittedStateError(
            "qualified S3 generalized nonlinear response supports only a "
            "stateless pre-integrated section; unsupported section-owned "
            "history markers: "
            + ", ".join(markers)
        )


def generalized_section_descriptor(section: Any) -> dict[str, Any]:
    """Return the exact binary64 constitutive identity of a stateless section."""

    require_stateless_generalized_section(section)
    try:
        made = validate_generalized_shell_section(section)
    except (TypeError, ValueError) as exc:
        raise S3CommittedStateError(
            "qualified S3 generalized section is incompatible"
        ) from exc
    return {
        "descriptor_schema": GENERALIZED_SECTION_DESCRIPTOR_SCHEMA,
        "history_policy_id": GENERALIZED_SECTION_HISTORY_POLICY_ID,
        "name": str(made.name),
        "A": np.asarray(made.A, dtype=np.float64).tolist(),
        "B": np.asarray(made.B, dtype=np.float64).tolist(),
        "D": np.asarray(made.D, dtype=np.float64).tolist(),
        "As": np.asarray(made.As, dtype=np.float64).tolist(),
        "mass_per_area": (
            None if made.mass_per_area is None else float(made.mass_per_area)
        ),
        "rotary_inertia_per_area": (
            None
            if made.rotary_inertia_per_area is None
            else float(made.rotary_inertia_per_area)
        ),
    }


def _validate_generalized_section_descriptor(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S3CommittedStateError(
            "generalized_section_descriptor must be a mapping"
        )
    expected = frozenset(
        {
            "descriptor_schema",
            "history_policy_id",
            "name",
            "A",
            "B",
            "D",
            "As",
            "mass_per_area",
            "rotary_inertia_per_area",
        }
    )
    _closed_keys(value, expected, "generalized_section_descriptor")
    if value["descriptor_schema"] != GENERALIZED_SECTION_DESCRIPTOR_SCHEMA:
        raise S3CommittedStateError(
            "incompatible generalized section descriptor schema"
        )
    if value["history_policy_id"] != GENERALIZED_SECTION_HISTORY_POLICY_ID:
        raise S3CommittedStateError(
            "incompatible generalized section history policy"
        )
    rebuilt = generalized_section_descriptor(
        {
            "name": value["name"],
            "A": value["A"],
            "B": value["B"],
            "D": value["D"],
            "As": value["As"],
            "mass_per_area": value["mass_per_area"],
            "rotary_inertia_per_area": value["rotary_inertia_per_area"],
        }
    )
    if _canonical_value(dict(value)) != rebuilt:
        raise S3CommittedStateError(
            "generalized section descriptor is not normalized"
        )
    return rebuilt


def generalized_section_fingerprint(section_descriptor: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "layout": "QUALIFIED_S3_GENERALIZED_SECTION_IDENTITY_V1",
            "descriptor": _validate_generalized_section_descriptor(
                section_descriptor
            ),
        }
    )


def build_generalized_element_configuration_descriptor(
    *,
    thickness: float,
    reference_normal: Any,
    director_polarity: Any,
    material_direction: Any | None,
    material_angle_deg: float,
    shell_section: Any,
    reference_surface_offset: float = 0.0,
) -> dict[str, Any]:
    """Build the closed configuration identity for generalized-section state."""

    if shell_section is None:
        raise S3CommittedStateError(
            "generalized qualified S3 state requires a shell section"
        )
    if isinstance(thickness, (bool, np.bool_)) or not isinstance(
        thickness, (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError("thickness must be a finite positive scalar")
    thickness_value = float(thickness)
    if not math.isfinite(thickness_value) or thickness_value <= 0.0:
        raise S3CommittedStateError("thickness must be a finite positive scalar")
    normal = _finite_array(reference_normal, (3,), "reference_normal")
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= np.finfo(float).tiny:
        raise S3CommittedStateError("reference_normal must have positive norm")
    normal /= normal_norm
    if material_direction is None:
        direction: list[float] | None = None
    else:
        made_direction = _finite_array(
            material_direction, (3,), "material_direction"
        )
        tangent = made_direction - float(made_direction @ normal) * normal
        tangent_norm = float(np.linalg.norm(tangent))
        if not math.isfinite(tangent_norm) or tangent_norm <= np.finfo(float).tiny:
            raise S3CommittedStateError(
                "material_direction must define a surface tangent"
            )
        direction = (tangent / tangent_norm).tolist()
    if isinstance(material_angle_deg, (bool, np.bool_)) or not isinstance(
        material_angle_deg, (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError("material_angle_deg must be finite")
    angle = float(material_angle_deg)
    if not math.isfinite(angle):
        raise S3CommittedStateError("material_angle_deg must be finite")
    if direction is None and angle != 0.0:
        raise S3CommittedStateError(
            "material_angle_deg requires a physical material_direction"
        )
    section_descriptor = generalized_section_descriptor(shell_section)
    polarity = _director_polarity(director_polarity)
    offset = _reference_surface_offset(reference_surface_offset)
    return {
        "descriptor_schema": GENERALIZED_ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA,
        "thickness": thickness_value,
        "reference_surface_offset": offset,
        "reference_surface_offset_policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
        "reference_normal": normal.tolist(),
        "director_polarity": polarity,
        "material_direction": direction,
        "material_angle_deg": angle,
        "section_mode": GENERALIZED_STATE_MODE,
        "generalized_section": section_descriptor,
    }


def _validate_generalized_element_descriptor(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S3CommittedStateError("element_descriptor must be a mapping")
    expected = frozenset(
        {
            "descriptor_schema",
            "thickness",
            "reference_surface_offset",
            "reference_surface_offset_policy_id",
            "reference_normal",
            "director_polarity",
            "material_direction",
            "material_angle_deg",
            "section_mode",
            "generalized_section",
        }
    )
    _closed_keys(value, expected, "generalized element_descriptor")
    if value["descriptor_schema"] != GENERALIZED_ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA:
        raise S3CommittedStateError(
            "incompatible generalized element descriptor schema"
        )
    if value["section_mode"] != GENERALIZED_STATE_MODE:
        raise S3CommittedStateError("incompatible generalized element section mode")
    if value["reference_surface_offset_policy_id"] != REFERENCE_SURFACE_OFFSET_POLICY_ID:
        raise S3CommittedStateError(
            "incompatible generalized reference-surface offset policy"
        )
    section = _validate_generalized_section_descriptor(
        value["generalized_section"]
    )
    rebuilt = build_generalized_element_configuration_descriptor(
        thickness=value["thickness"],
        reference_normal=value["reference_normal"],
        director_polarity=value["director_polarity"],
        material_direction=value["material_direction"],
        material_angle_deg=value["material_angle_deg"],
        shell_section=section,
        reference_surface_offset=value["reference_surface_offset"],
    )
    if _canonical_value(dict(value)) != rebuilt:
        raise S3CommittedStateError(
            "generalized element descriptor is not normalized"
        )
    return rebuilt


def generalized_element_configuration_fingerprint(
    element_id: int,
    node_ids: Sequence[int],
    element_descriptor: Mapping[str, Any],
) -> str:
    if isinstance(element_id, (bool, np.bool_)) or not isinstance(
        element_id, (int, np.integer)
    ):
        raise S3CommittedStateError("element_id must be an integer")
    descriptor = _validate_generalized_element_descriptor(element_descriptor)
    return canonical_sha256(
        {
            "layout": "QUALIFIED_S3_GENERALIZED_ELEMENT_CONFIGURATION_V3",
            "element_id": int(element_id),
            "node_ids": _node_ids(node_ids),
            "descriptor": descriptor,
        }
    )


def build_element_configuration_descriptor(
    *,
    thickness: float,
    reference_normal: Any,
    director_polarity: Any,
    material_direction: Any | None,
    material_angle_deg: float,
    shell_section: Any | None,
    reference_surface_offset: float = 0.0,
) -> dict[str, Any]:
    """Build the complete layered-state configuration descriptor.

    A pre-integrated generalized section has a different state mode and is
    intentionally rejected by this V1 layered-state builder.
    """

    if shell_section is not None:
        raise S3CommittedStateError(
            "layered qualified S3 state cannot describe a generalized section"
        )
    if isinstance(thickness, (bool, np.bool_)) or not isinstance(
        thickness, (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError("thickness must be a finite positive scalar")
    thickness_value = float(thickness)
    if not math.isfinite(thickness_value) or thickness_value <= 0.0:
        raise S3CommittedStateError("thickness must be a finite positive scalar")
    normal = _finite_array(reference_normal, (3,), "reference_normal")
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= np.finfo(float).tiny:
        raise S3CommittedStateError("reference_normal must have positive norm")
    normal /= normal_norm
    if material_direction is None:
        direction: list[float] | None = None
    else:
        made_direction = _finite_array(
            material_direction, (3,), "material_direction"
        )
        tangent = made_direction - float(made_direction @ normal) * normal
        tangent_norm = float(np.linalg.norm(tangent))
        if not math.isfinite(tangent_norm) or tangent_norm <= np.finfo(float).tiny:
            raise S3CommittedStateError(
                "material_direction must define a surface tangent"
            )
        direction = (tangent / tangent_norm).tolist()
    if isinstance(material_angle_deg, (bool, np.bool_)) or not isinstance(
        material_angle_deg, (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError("material_angle_deg must be finite")
    angle = float(material_angle_deg)
    if not math.isfinite(angle):
        raise S3CommittedStateError("material_angle_deg must be finite")
    if direction is None and angle != 0.0:
        raise S3CommittedStateError(
            "material_angle_deg requires a physical material_direction"
        )
    polarity = _director_polarity(director_polarity)
    offset = _reference_surface_offset(reference_surface_offset)
    return {
        "descriptor_schema": ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA,
        "thickness": thickness_value,
        "reference_surface_offset": offset,
        "reference_surface_offset_policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
        "reference_normal": normal.tolist(),
        "director_polarity": polarity,
        "material_direction": direction,
        "material_angle_deg": angle,
        "section_mode": "homogeneous_layered_material",
    }


def _validate_element_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S3CommittedStateError("element_descriptor must be a mapping")
    expected = frozenset(
        {
            "descriptor_schema",
            "thickness",
            "reference_surface_offset",
            "reference_surface_offset_policy_id",
            "reference_normal",
            "director_polarity",
            "material_direction",
            "material_angle_deg",
            "section_mode",
        }
    )
    _closed_keys(value, expected, "element_descriptor")
    if value["reference_surface_offset_policy_id"] != REFERENCE_SURFACE_OFFSET_POLICY_ID:
        raise S3CommittedStateError(
            "incompatible layered reference-surface offset policy"
        )
    rebuilt = build_element_configuration_descriptor(
        thickness=value["thickness"],
        reference_normal=value["reference_normal"],
        director_polarity=value["director_polarity"],
        material_direction=value["material_direction"],
        material_angle_deg=value["material_angle_deg"],
        shell_section=None,
        reference_surface_offset=value["reference_surface_offset"],
    )
    if value["descriptor_schema"] != ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA:
        raise S3CommittedStateError("incompatible element descriptor schema")
    if value["section_mode"] != "homogeneous_layered_material":
        raise S3CommittedStateError("incompatible element section mode")
    if _canonical_value(dict(value)) != rebuilt:
        raise S3CommittedStateError("element descriptor is not normalized")
    return rebuilt


def _validate_material_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S3CommittedStateError("material_descriptor must be a mapping")
    expected = frozenset(
        {
            "descriptor_schema",
            "material_symmetry",
            "equivalent_stress_measure",
            "elastic_compliance_engineering_6x6",
            "resolved_material",
        }
    )
    _closed_keys(value, expected, "material_descriptor")
    if value["descriptor_schema"] != MATERIAL_DESCRIPTOR_SCHEMA:
        raise S3CommittedStateError("incompatible material descriptor schema")
    resolved, symmetry, measure = _validate_resolved_material_tree(
        value["resolved_material"]
    )
    compliance = _validated_compliance(value["elastic_compliance_engineering_6x6"])
    try:
        expected_compliance = _compliance_from_registered_fields(resolved)
    except S3CommittedStateError:
        raise
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise S3CommittedStateError(
            "registered material fields cannot define structural compliance"
        ) from exc
    if not np.array_equal(compliance, expected_compliance):
        raise S3CommittedStateError(
            "elastic compliance does not match the registered material fields"
        )
    if value["material_symmetry"] != symmetry:
        raise S3CommittedStateError(
            "material_symmetry does not match the registered material descriptor"
        )
    if value["equivalent_stress_measure"] != measure:
        raise S3CommittedStateError(
            "equivalent_stress_measure does not match the registered material descriptor"
        )
    _constitutive_identity(symmetry, measure)
    canonical = _canonical_value(dict(value), path="$.material_descriptor")
    assert isinstance(canonical, dict)
    canonical["resolved_material"] = resolved
    return canonical


def formulation_mechanics_contract_payload() -> dict[str, Any]:
    """Return the exact formulation mechanics authority used by production."""

    return {
        "physical_director_reversal": {
            "polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
            "transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
            "admitted_polarities": (-1, 1),
            "sheet_owner_normal_is_independent": True,
            "generalized_component_transform": (
                "diag(1,1,1,p,p,p,p,p)"
            ),
            "generalized_section_B_transform": "B_physical=p*B_owner",
            "layer_order": "bottom_to_top_along_physical_director",
            "equivalent_offset_transform": "(polarity,offset)->(-polarity,-offset)",
        },
        "reference_surface_offset": {
            "policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
            "strain_transform_id": REFERENCE_SURFACE_STRAIN_TRANSFORM_ID,
            "mass_shift_id": REFERENCE_SURFACE_MASS_SHIFT_ID,
            "positive_direction": "section_origin_to_nodal_reference_along_physical_director",
            "section_strain": "epsilon_section=epsilon_reference-offset*kappa",
            "layer_coordinate_from_reference": "xi=z-offset",
            "material_surfaces_from_reference": (
                "bottom=-thickness/2-offset",
                "top=thickness/2-offset",
            ),
        },
        "linear_source": {
            "url": MITC3_PLUS_SOURCE_URL,
            "byte_count": MITC3_PLUS_SOURCE_BYTES,
            "sha256": MITC3_PLUS_SOURCE_SHA256,
            "equation_map": MITC3_PLUS_EQUATION_MAP,
        },
        "nonlinear_source": {
            "url": MITC3_PLUS_NONLINEAR_SOURCE_URL,
            "byte_count": MITC3_PLUS_NONLINEAR_SOURCE_BYTES,
            "sha256": MITC3_PLUS_NONLINEAR_SOURCE_SHA256,
            "equation_map": MITC3_PLUS_NONLINEAR_EQUATION_MAP,
        },
        "bubble": {
            "polynomial": "27*L1*L2*L3",
            "polynomial_scale": BUBBLE_POLYNOMIAL_SCALE,
            "offset_d_exact": BUBBLE_OFFSET_EXACT,
            "offset_d_binary64": BUBBLE_OFFSET_D,
            "tying_point_definitions": TYING_POINT_DEFINITIONS,
            "tying_point_binary64": TYING_POINTS,
        },
        "pl_completion": {
            "basis_id": PL_BASIS_ID,
            "constraint_id": PL_CONSTRAINT_ID,
            "gram_scale_id": PL_GRAM_SCALE_ID,
            "gram_numerator": PL_GRAM_NUMERATOR,
            "block_sign_id": PL_BLOCK_SIGN_ID,
            "condensation_id": PL_CONDENSATION_ID,
            "surface_rotation_policy_id": SURFACE_ROTATION_POLICY_ID,
            "twist_policy_id": PL_TWIST_POLICY_ID,
            "rotation_update_policy_id": NODAL_ROTATION_UPDATE_POLICY_ID,
            "phase_policy_id": PL_PHASE_POLICY_ID,
            "phase_margin_id": PL_PHASE_MARGIN_ID,
            "phase_margin_binary64": PL_PHASE_MARGIN,
            "minimum_twist_denominator_id": (
                PL_MINIMUM_TWIST_DENOMINATOR_ID
            ),
            "minimum_twist_denominator_binary64": (
                PL_MINIMUM_TWIST_DENOMINATOR
            ),
            "energy_policy_id": NONLINEAR_PL_ENERGY_POLICY_ID,
        },
        "drilling_scale": {
            "policy_id": DRILL_SCALE_POLICY_ID,
            "projector": DRILL_SCALE_PROJECTOR,
            "metric": DRILL_SCALE_METRIC,
            "inverse_metric_sqrt": DRILL_SCALE_INVERSE_METRIC_SQRT,
            "factor": 0.5,
            "material_state": "elastic_membrane_section_matrix_A",
            "required": "finite_strictly_positive",
        },
    }


def formulation_mechanics_fingerprint() -> str:
    """Return the canonical hash of exact state-relevant mechanics authority."""

    return canonical_sha256(formulation_mechanics_contract_payload())


def formulation_fingerprint_payload() -> dict[str, Any]:
    """Return the frozen state-relevant qualified-S3 formulation identity."""

    return {
        "formulation_id": FORMULATION_ID,
        "formulation_schema": FORMULATION_SCHEMA,
        "external_coordinate_layout_id": EXTERNAL_COORDINATE_LAYOUT_ID,
        "nonlinear_state_schema": NONLINEAR_STATE_SCHEMA,
        "nonlinear_state_version": NONLINEAR_STATE_VERSION,
        "nonlinear_state_layout_id": NONLINEAR_STATE_LAYOUT_ID,
        "nonlinear_kinematics_id": NONLINEAR_KINEMATICS_ID,
        "director_gauge_id": DIRECTOR_GAUGE_ID,
        "director_gauge_primary_axis": DIRECTOR_GAUGE_PRIMARY_AXIS,
        "director_gauge_fallback_axis": DIRECTOR_GAUGE_FALLBACK_AXIS,
        "director_gauge_switch_tolerance": DIRECTOR_GAUGE_SWITCH_TOLERANCE,
        "external_rotation_map_id": EXTERNAL_ROTATION_MAP_ID,
        "director_increment_map_id": DIRECTOR_INCREMENT_MAP_ID,
        "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
        "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
        "reference_surface_offset_policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
        "reference_surface_strain_transform_id": (
            REFERENCE_SURFACE_STRAIN_TRANSFORM_ID
        ),
        "reference_surface_mass_shift_id": REFERENCE_SURFACE_MASS_SHIFT_ID,
        "surface_rotation_policy_id": SURFACE_ROTATION_POLICY_ID,
        "pl_twist_policy_id": PL_TWIST_POLICY_ID,
        "nodal_rotation_update_policy_id": NODAL_ROTATION_UPDATE_POLICY_ID,
        "pl_phase_policy_id": PL_PHASE_POLICY_ID,
        "pl_phase_margin_id": PL_PHASE_MARGIN_ID,
        "pl_phase_margin_binary64": PL_PHASE_MARGIN,
        "pl_minimum_twist_denominator_id": (
            PL_MINIMUM_TWIST_DENOMINATOR_ID
        ),
        "pl_minimum_twist_denominator_binary64": (
            PL_MINIMUM_TWIST_DENOMINATOR
        ),
        "nonlinear_pl_energy_policy_id": NONLINEAR_PL_ENERGY_POLICY_ID,
        "bubble_state_role": BUBBLE_STATE_ROLE,
        "bubble_predictor_commit_policy_id": BUBBLE_PREDICTOR_COMMIT_POLICY_ID,
        "bubble_convention": BUBBLE_CONVENTION,
        "quadrature_id": QUADRATURE_ID,
        "nonlinear_policy_id": NONLINEAR_POLICY_ID,
        "recovery_policy_id": RECOVERY_POLICY_ID,
        "nonlinear_source_sha256": NONLINEAR_SOURCE_SHA256,
        "formulation_mechanics": formulation_mechanics_contract_payload(),
        "formulation_mechanics_sha256": formulation_mechanics_fingerprint(),
        "bubble_equilibrium_policy": {
            "max_iterations": BUBBLE_MAX_ITERATIONS,
            "relative_tolerance": BUBBLE_RELATIVE_TOLERANCE,
            "step_tolerance": BUBBLE_STEP_TOLERANCE,
            "condition_limit": BUBBLE_CONDITION_LIMIT,
            "line_search_min_factor": BUBBLE_LINE_SEARCH_MIN_FACTOR,
            "line_search_reduction": BUBBLE_LINE_SEARCH_REDUCTION,
            "force_condensation_id": BUBBLE_FORCE_CONDENSATION_ID,
        },
        "constitutive_integration_ids": {
            "isotropic": ISOTROPIC_CONSTITUTIVE_INTEGRATION_ID,
            "orthotropic": ORTHOTROPIC_CONSTITUTIVE_INTEGRATION_ID,
            "generalized_section": GENERALIZED_SECTION_INTEGRATION_ID,
        },
        "canonicalization_id": CANONICALIZATION_ID,
        "state_integrity_id": STATE_INTEGRITY_ID,
        "state_redundancy_tolerance_factor": STATE_REDUNDANCY_TOLERANCE_FACTOR,
        "state_redundancy_validation_id": STATE_REDUNDANCY_VALIDATION_ID,
        "reference_geometry_validation_id": REFERENCE_GEOMETRY_VALIDATION_ID,
        "state_mode": STATE_MODE,
        "element_configuration_descriptor_schema": (
            ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA
        ),
        "material_descriptor_schema": MATERIAL_DESCRIPTOR_SCHEMA,
        "material_descriptor_validation_id": MATERIAL_DESCRIPTOR_VALIDATION_ID,
        "thickness_quadrature_id": THICKNESS_QUADRATURE_ID,
        "supported_lobatto_layer_counts": SUPPORTED_LOBATTO_LAYER_COUNTS,
        "stiffness_station_table_sha256": stiffness_station_table_fingerprint(),
        "lobatto_table_sha256": lobatto_table_fingerprint(),
        "state_array_layout_id": STATE_ARRAY_LAYOUT_ID,
        "thickness_coordinate_sign_id": THICKNESS_COORDINATE_SIGN_ID,
        "state_field_manifest_sha256": state_field_manifest_fingerprint(),
        "director_orthonormality_tolerance": DIRECTOR_ORTHONORMALITY_TOLERANCE,
        "geometry_admission": {
            "minimum_owner_normal_alignment": MINIMUM_OWNER_NORMAL_ALIGNMENT,
            "minimum_normalized_twice_area": MINIMUM_NORMALIZED_TWICE_AREA,
            "minimum_angle_deg": MINIMUM_ANGLE_DEG,
            "maximum_angle_deg": MAXIMUM_ANGLE_DEG,
            "maximum_edge_ratio": MAXIMUM_EDGE_RATIO,
            "minimum_corner_scaled_jacobian": MINIMUM_CORNER_SCALED_JACOBIAN,
            "minimum_normalized_area": MINIMUM_NORMALIZED_AREA,
            "comparison_tolerance": QUALITY_COMPARISON_TOLERANCE,
            "reference_frame_match_tolerance": REFERENCE_FRAME_MATCH_TOLERANCE,
        },
        "generalized_strain_component_order": GENERALIZED_STRAIN_COMPONENT_ORDER,
        "generalized_resultant_component_order": (
            GENERALIZED_RESULTANT_COMPONENT_ORDER
        ),
        "layer_strain_component_order": LAYER_STRAIN_COMPONENT_ORDER,
        "layer_stress_component_order": LAYER_STRESS_COMPONENT_ORDER,
        "registered_material_descriptor_fields": {
            "root": {
                key: tuple(sorted(value))
                for key, value in sorted(_ROOT_MATERIAL_FIELDS.items())
            },
            "hardening": {
                key: tuple(sorted(value))
                for key, value in sorted(_HARDENING_CURVE_FIELDS.items())
            },
            "hill48": tuple(sorted(_HILL48_FIELDS)),
        },
        "initial_field_names": INITIAL_FIELD_NAMES,
        "fixed_array_shapes": {
            key: shape for key, shape in sorted(_ARRAY_SHAPES_FIXED.items())
        },
        "layer_vector_fields": _LAYER_VECTOR_FIELDS,
        "state_keys": tuple(sorted(_STATE_KEYS)),
        "identity_keys": tuple(sorted(_IDENTITY_KEYS)),
    }


def formulation_fingerprint() -> str:
    """Return the canonical fingerprint of the frozen state formulation."""

    return canonical_sha256(formulation_fingerprint_payload())


def generalized_formulation_fingerprint_payload() -> dict[str, Any]:
    """Return the independently identified stateless-section state contract."""

    return {
        "formulation_id": FORMULATION_ID,
        "formulation_schema": FORMULATION_SCHEMA,
        "state_schema": GENERALIZED_NONLINEAR_STATE_SCHEMA,
        "state_version": GENERALIZED_NONLINEAR_STATE_VERSION,
        "state_layout_id": GENERALIZED_NONLINEAR_STATE_LAYOUT_ID,
        "state_mode": GENERALIZED_STATE_MODE,
        "external_coordinate_layout_id": EXTERNAL_COORDINATE_LAYOUT_ID,
        "nonlinear_kinematics_id": NONLINEAR_KINEMATICS_ID,
        "director_gauge_id": DIRECTOR_GAUGE_ID,
        "external_rotation_map_id": EXTERNAL_ROTATION_MAP_ID,
        "director_increment_map_id": DIRECTOR_INCREMENT_MAP_ID,
        "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
        "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
        "reference_surface_offset_policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
        "reference_surface_strain_transform_id": (
            REFERENCE_SURFACE_STRAIN_TRANSFORM_ID
        ),
        "reference_surface_mass_shift_id": REFERENCE_SURFACE_MASS_SHIFT_ID,
        "surface_rotation_policy_id": SURFACE_ROTATION_POLICY_ID,
        "pl_twist_policy_id": PL_TWIST_POLICY_ID,
        "nodal_rotation_update_policy_id": NODAL_ROTATION_UPDATE_POLICY_ID,
        "pl_phase_policy_id": PL_PHASE_POLICY_ID,
        "pl_phase_margin_id": PL_PHASE_MARGIN_ID,
        "pl_phase_margin_binary64": PL_PHASE_MARGIN,
        "pl_minimum_twist_denominator_id": (
            PL_MINIMUM_TWIST_DENOMINATOR_ID
        ),
        "pl_minimum_twist_denominator_binary64": (
            PL_MINIMUM_TWIST_DENOMINATOR
        ),
        "nonlinear_pl_energy_policy_id": NONLINEAR_PL_ENERGY_POLICY_ID,
        "bubble_state_role": BUBBLE_STATE_ROLE,
        "bubble_predictor_commit_policy_id": BUBBLE_PREDICTOR_COMMIT_POLICY_ID,
        "bubble_convention": BUBBLE_CONVENTION,
        "quadrature_id": QUADRATURE_ID,
        "nonlinear_policy_id": NONLINEAR_POLICY_ID,
        "generalized_section_integration_id": GENERALIZED_SECTION_INTEGRATION_ID,
        "generalized_initial_field_policy": {
            "id": GENERALIZED_INITIAL_FIELD_POLICY_ID,
            "source_convention": "SHELL_INITIAL_FIELD_LOCAL_SURFACE_STRESS_V1",
            "membrane_resultant": "N0_EQUALS_H_TIMES_SIGMA_MEMBRANE",
            "bending_resultant": "M0_EQUALS_H_SQUARED_OVER_6_TIMES_SIGMA_BENDING_SURFACE",
            "constitutive_relation": (
                "RESULTANT_EQUALS_C_TIMES_KINEMATIC_MINUS_PRESTRAIN_PLUS_INITIAL_RESULTANT"
            ),
            "station_count": NUM_INTEGRATION_STATIONS,
            "physical_layer_reconstruction": "FORBIDDEN",
        },
        "generalized_section_history_policy_id": (
            GENERALIZED_SECTION_HISTORY_POLICY_ID
        ),
        "generalized_initial_field_policy_id": (
            GENERALIZED_INITIAL_FIELD_POLICY_ID
        ),
        "rejected_section_history_markers": tuple(
            sorted(_GENERALIZED_SECTION_HISTORY_MARKERS)
        ),
        "generalized_section_descriptor_schema": (
            GENERALIZED_SECTION_DESCRIPTOR_SCHEMA
        ),
        "element_configuration_descriptor_schema": (
            GENERALIZED_ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA
        ),
        "canonicalization_id": CANONICALIZATION_ID,
        "state_integrity_id": STATE_INTEGRITY_ID,
        "stiffness_station_table_sha256": stiffness_station_table_fingerprint(),
        "state_field_manifest_sha256": (
            generalized_state_field_manifest_fingerprint()
        ),
        "recovery_scope": "section_resultants_only",
        "physical_layer_recovery_available": False,
        "state_keys": tuple(sorted(_GENERALIZED_STATE_KEYS)),
        "identity_keys": tuple(sorted(_GENERALIZED_IDENTITY_KEYS)),
        "fixed_array_shapes": {
            key: shape
            for key, shape in sorted(_GENERALIZED_ARRAY_SHAPES.items())
        },
        "formulation_mechanics_sha256": formulation_mechanics_fingerprint(),
    }


def generalized_formulation_fingerprint() -> str:
    return canonical_sha256(generalized_formulation_fingerprint_payload())


def _node_ids(value: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes, bytearray)):
        raise S3CommittedStateError("qualified S3 state requires exactly three node IDs")
    try:
        size = len(value)
    except TypeError as exc:
        raise S3CommittedStateError(
            "qualified S3 state requires exactly three node IDs"
        ) from exc
    if size != 3:
        raise S3CommittedStateError("qualified S3 state requires exactly three node IDs")
    result: list[int] = []
    for raw in value:
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
            raise S3CommittedStateError("qualified S3 node IDs must be integers")
        result.append(int(raw))
    if len(set(result)) != 3:
        raise S3CommittedStateError("qualified S3 state requires three distinct node IDs")
    return tuple(result)  # type: ignore[return-value]


def _layer_count(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise S3CommittedStateError(
            f"{label} must be one of {list(SUPPORTED_LOBATTO_LAYER_COUNTS)}"
        )
    made = int(value)
    if made not in SUPPORTED_LOBATTO_LAYER_COUNTS:
        raise S3CommittedStateError(
            f"{label} must be one of {list(SUPPORTED_LOBATTO_LAYER_COUNTS)}"
        )
    return made


def _director_polarity(value: Any, label: str = "director_polarity") -> int:
    """Return the strict element-owned physical-director polarity."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise S3CommittedStateError(f"{label} must be the integer -1 or +1")
    polarity = int(value)
    if polarity not in (-1, 1):
        raise S3CommittedStateError(f"{label} must be the integer -1 or +1")
    return polarity


def _reference_surface_offset(
    value: Any,
    label: str = "reference_surface_offset",
) -> float:
    """Return the strict signed physical-director reference-surface offset."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError(f"{label} must be a finite real scalar")
    offset = float(value)
    if not math.isfinite(offset):
        raise S3CommittedStateError(f"{label} must be a finite real scalar")
    return 0.0 if offset == 0.0 else offset


def _constitutive_identity(
    material_symmetry: Any,
    equivalent_stress_measure: Any,
) -> tuple[str, str]:
    if not isinstance(material_symmetry, str) or material_symmetry not in {
        "isotropic",
        "orthotropic",
    }:
        raise S3CommittedStateError(
            "material_symmetry must be 'isotropic' or 'orthotropic'"
        )
    if not isinstance(equivalent_stress_measure, str) or (
        equivalent_stress_measure not in {"von_mises", "hill48"}
    ):
        raise S3CommittedStateError("unsupported equivalent_stress_measure")
    if material_symmetry == "isotropic" and equivalent_stress_measure != "von_mises":
        raise S3CommittedStateError("isotropic state requires von_mises")
    return material_symmetry, equivalent_stress_measure


def _validate_numeric_values(value: Any, label: str) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in {"i", "u", "f"}:
            raise S3CommittedStateError(
                f"{label} must contain real numeric values, not {value.dtype}"
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for member in value:
            _validate_numeric_values(member, label)
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError(f"{label} must contain real numeric values")


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    _validate_numeric_values(value, label)
    try:
        result = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise S3CommittedStateError(f"{label} must contain numeric values") from exc
    if result.shape != shape:
        raise S3CommittedStateError(
            f"{label} has shape {result.shape}; expected {shape}"
        )
    if not np.all(np.isfinite(result)):
        raise S3CommittedStateError(f"{label} contains nonfinite values")
    result = np.array(result, dtype=np.float64, order="C", copy=True)
    result[result == 0.0] = 0.0
    return result


def _validate_state_float_values(value: Any, label: str) -> None:
    """Require canonical binary64 spellings for persisted state arrays."""

    if isinstance(value, np.ndarray):
        if value.dtype != np.dtype(np.float64):
            raise S3CommittedStateError(
                f"{label} must contain canonical binary64 floating values"
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for member in value:
            _validate_state_float_values(member, label)
        return
    if type(value) is not float:
        raise S3CommittedStateError(
            f"{label} must contain canonical binary64 floating values"
        )


def _finite_state_array(
    value: Any, shape: tuple[int, ...], label: str
) -> np.ndarray:
    _validate_state_float_values(value, label)
    return _finite_array(value, shape, label)


def _integer_state_array(
    value: Any, shape: tuple[int, ...], label: str
) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype != np.dtype(np.int64):
            raise S3CommittedStateError(
                f"{label} must contain canonical signed 64-bit integers"
            )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for member in value:
            if type(member) is not int:
                raise S3CommittedStateError(
                    f"{label} must contain canonical integer values"
                )
    else:
        raise S3CommittedStateError(f"{label} must be an integer array")
    result = np.asarray(value, dtype=np.int64)
    if result.shape != shape:
        raise S3CommittedStateError(
            f"{label} has shape {result.shape}; expected {shape}"
        )
    return np.array(result, dtype=np.int64, order="C", copy=True)


def _finite_state_scalar(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise S3CommittedStateError(
            f"{label} must be a finite canonical binary64 scalar"
        )
    return float(value)


def qualified_s3_triangle_frame(
    coordinates: Any,
    owner_normal: Any | None,
    *,
    enforce_admission: bool = True,
    enforce_positive_winding: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Derive the unique numbered reference frame and qualified quality data."""

    nodes = _finite_array(coordinates, (3, 3), "reference_coordinates")
    edge_r = nodes[1] - nodes[0]
    edge_s = nodes[2] - nodes[0]
    normal_raw = np.cross(edge_r, edge_s)
    twice_area = float(np.linalg.norm(normal_raw))
    lengths = np.asarray(
        (
            np.linalg.norm(nodes[1] - nodes[0]),
            np.linalg.norm(nodes[2] - nodes[1]),
            np.linalg.norm(nodes[0] - nodes[2]),
        ),
        dtype=np.float64,
    )
    maximum_edge = float(np.max(lengths))
    minimum_edge = float(np.min(lengths))
    if (
        not math.isfinite(maximum_edge)
        or maximum_edge <= 0.0
        or minimum_edge <= 0.0
    ):
        raise S3CommittedStateError("qualified S3 requires three distinct nodes")
    normalized_twice_area = twice_area / (maximum_edge * maximum_edge)
    if (
        not math.isfinite(normalized_twice_area)
        or normalized_twice_area <= MINIMUM_NORMALIZED_TWICE_AREA
    ):
        raise S3CommittedStateError(
            "qualified S3 has a zero or near-zero signed area"
        )

    first = edge_r / float(np.linalg.norm(edge_r))
    normal = normal_raw / twice_area
    connectivity_sign = 1.0
    owner_alignment = 1.0
    if owner_normal is not None:
        supplied = _finite_array(owner_normal, (3,), "reference_normal")
        supplied_norm = float(np.linalg.norm(supplied))
        if supplied_norm <= np.finfo(np.float64).tiny:
            raise S3CommittedStateError(
                "qualified S3 reference_normal must have positive norm"
            )
        supplied /= supplied_norm
        signed_alignment = float(normal @ supplied)
        if abs(signed_alignment) <= MINIMUM_OWNER_NORMAL_ALIGNMENT:
            raise S3CommittedStateError(
                "qualified S3 reference_normal is tangential to the facet"
            )
        if signed_alignment < 0.0:
            normal = -normal
            connectivity_sign = -1.0
        owner_alignment = abs(signed_alignment)
    second = np.cross(normal, first)
    second /= float(np.linalg.norm(second))
    first = np.cross(second, normal)
    first /= float(np.linalg.norm(first))
    frame = np.column_stack((first, second, normal))
    local = (nodes - nodes[0]) @ frame[:, :2]

    cosines = np.empty(3, dtype=np.float64)
    for index in range(3):
        left = nodes[(index + 1) % 3] - nodes[index]
        right = nodes[(index - 1) % 3] - nodes[index]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosines[index] = float(np.clip(float(left @ right) / denominator, -1.0, 1.0))
    angles = np.degrees(np.arccos(cosines))
    corner_scaled_jacobian = float(np.min(np.sin(np.radians(angles))))
    normalized_area = float(
        2.0 * math.sqrt(3.0) * twice_area / float(lengths @ lengths)
    )
    quality = {
        "area": 0.5 * twice_area,
        "normalized_twice_area": normalized_twice_area,
        "minimum_angle_deg": float(np.min(angles)),
        "maximum_angle_deg": float(np.max(angles)),
        "edge_ratio": maximum_edge / minimum_edge,
        "minimum_scaled_jacobian": corner_scaled_jacobian,
        "normalized_area": normalized_area,
        "connectivity_sign": connectivity_sign,
        "reference_normal_alignment": owner_alignment,
    }
    if enforce_admission:
        require_qualified_s3_quality(
            quality,
            enforce_positive_winding=enforce_positive_winding,
        )
    return frame, local, quality


def require_qualified_s3_quality(
    quality: Mapping[str, Any],
    *,
    enforce_positive_winding: bool = True,
) -> None:
    """Fail closed unless a triangle satisfies the frozen admission envelope."""

    required = frozenset(
        {
            "area",
            "normalized_twice_area",
            "minimum_angle_deg",
            "maximum_angle_deg",
            "edge_ratio",
            "minimum_scaled_jacobian",
            "normalized_area",
            "connectivity_sign",
            "reference_normal_alignment",
        }
    )
    if not isinstance(quality, Mapping):
        raise S3CommittedStateError("qualified S3 quality data is incomplete")
    _closed_keys(quality, required, "qualified S3 quality data")
    values: dict[str, float] = {}
    for key in required:
        raw = quality[key]
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, float, np.integer, np.floating)
        ):
            raise S3CommittedStateError(
                f"qualified S3 quality metric {key!r} must be a real scalar"
            )
        made = float(raw)
        if not math.isfinite(made):
            raise S3CommittedStateError(
                f"qualified S3 quality metric {key!r} must be finite"
            )
        values[key] = made
    failures: list[str] = []
    tolerance = QUALITY_COMPARISON_TOLERANCE
    if values["area"] <= 0.0:
        failures.append("area must be strictly positive")
    if values["normalized_twice_area"] <= MINIMUM_NORMALIZED_TWICE_AREA:
        failures.append("normalized signed area is below the admitted threshold")
    if values["reference_normal_alignment"] <= MINIMUM_OWNER_NORMAL_ALIGNMENT:
        failures.append("owner-normal alignment is below the admitted threshold")
    if values["reference_normal_alignment"] > 1.0 + tolerance:
        failures.append("owner-normal alignment exceeds one")
    if values["connectivity_sign"] not in (-1.0, 1.0):
        failures.append("connectivity sign must be exactly minus or plus one")
    if enforce_positive_winding and values["connectivity_sign"] <= 0.0:
        failures.append("connectivity winding opposes the authoritative owner normal")
    if values["minimum_angle_deg"] < MINIMUM_ANGLE_DEG - tolerance:
        failures.append("minimum angle is below 30 degrees")
    if values["maximum_angle_deg"] > MAXIMUM_ANGLE_DEG + tolerance:
        failures.append("maximum angle exceeds 150 degrees")
    if values["edge_ratio"] > MAXIMUM_EDGE_RATIO + tolerance:
        failures.append("edge ratio exceeds 4.0")
    if (
        values["minimum_scaled_jacobian"]
        < MINIMUM_CORNER_SCALED_JACOBIAN - tolerance
    ):
        failures.append("minimum scaled Jacobian is below 0.20")
    if values["normalized_area"] < MINIMUM_NORMALIZED_AREA - tolerance:
        failures.append("normalized area is below 0.60")
    if failures:
        raise S3CommittedStateError(
            "qualified S3 quality admission failed: " + "; ".join(failures)
        )


def reconstruct_director_triad(normal: Any) -> np.ndarray:
    """Return the frozen MITC3+ Eq. (11) tangent gauge for ``normal``.

    The published construction uses global ``e_y``.  Bathe's stated parallel
    special case uses global ``e_z``; the frozen switch only covers the
    numerically singular neighbourhood of exact parallelism.  The returned
    columns are ``(V1, V2, Vn)`` with ``V1 = axis x Vn`` and
    ``V2 = Vn x V1``.
    """

    made = _finite_array(normal, (3,), "director_normal")
    norm = float(np.linalg.norm(made))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        raise S3CommittedStateError("director_normal must have positive norm")
    unit = made / norm
    primary = np.asarray(DIRECTOR_GAUGE_PRIMARY_AXIS, dtype=np.float64)
    first = np.cross(primary, unit)
    first_norm = float(np.linalg.norm(first))
    if first_norm <= DIRECTOR_GAUGE_SWITCH_TOLERANCE:
        fallback = np.asarray(DIRECTOR_GAUGE_FALLBACK_AXIS, dtype=np.float64)
        first = np.cross(fallback, unit)
        first_norm = float(np.linalg.norm(first))
    if not math.isfinite(first_norm) or first_norm <= np.finfo(np.float64).tiny:
        raise S3CommittedStateError("director Eq. (11) gauge is singular")
    first /= first_norm
    second = np.cross(unit, first)
    second /= float(np.linalg.norm(second))
    result = np.column_stack((first, second, unit))
    result[result == 0.0] = 0.0
    return result


def _validate_right_handed_triad(triad: np.ndarray, label: str) -> None:
    identity = np.eye(3, dtype=np.float64)
    gram_error = float(np.max(np.abs(triad.T @ triad - identity)))
    determinant = float(np.linalg.det(triad))
    if (
        not math.isfinite(gram_error)
        or gram_error > DIRECTOR_ORTHONORMALITY_TOLERANCE
        or not math.isfinite(determinant)
        or determinant <= 0.0
        or abs(determinant - 1.0) > DIRECTOR_ORTHONORMALITY_TOLERANCE
    ):
        raise S3CommittedStateError(
            f"{label} is not a right-handed orthonormal triad"
        )


def _validate_triads(triads: np.ndarray) -> None:
    for index, triad in enumerate(triads):
        _validate_right_handed_triad(
            triad, f"committed_director_triads[{index}]"
        )
        expected = reconstruct_director_triad(triad[:, 2])
        gauge_error = float(np.max(np.abs(triad - expected)))
        if (
            not math.isfinite(gauge_error)
            or gauge_error > DIRECTOR_ORTHONORMALITY_TOLERANCE
        ):
            raise S3CommittedStateError(
                f"committed_director_triads[{index}] does not satisfy the "
                "frozen MITC3+ Eq. (11) gauge"
            )


def _validate_nodal_rotation_state(
    reference_directors: np.ndarray,
    rotations: np.ndarray,
    triads: np.ndarray,
) -> None:
    identity = np.eye(3, dtype=np.float64)
    for index in range(3):
        reference = reference_directors[index]
        norm_error = abs(float(np.linalg.norm(reference)) - 1.0)
        if norm_error > DIRECTOR_ORTHONORMALITY_TOLERANCE:
            raise S3CommittedStateError(
                f"reference_corner_directors[{index}] is not a unit physical director"
            )
        rotation = rotations[index]
        gram_error = float(np.max(np.abs(rotation.T @ rotation - identity)))
        determinant = float(np.linalg.det(rotation))
        if (
            gram_error > DIRECTOR_ORTHONORMALITY_TOLERANCE
            or determinant <= 0.0
            or abs(determinant - 1.0) > DIRECTOR_ORTHONORMALITY_TOLERANCE
        ):
            raise S3CommittedStateError(
                f"committed_nodal_rotation_matrices[{index}] is not a proper SO(3) operator"
            )
        expected_normal = rotation @ reference
        normal_error = float(
            np.max(np.abs(triads[index, :, 2] - expected_normal))
        )
        if normal_error > DIRECTOR_ORTHONORMALITY_TOLERANCE:
            raise S3CommittedStateError(
                f"committed director {index} contradicts its node-shared rotation operator"
            )


def _fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise S3CommittedStateError(f"{label} must be an uppercase SHA-256")
    return value


def node_order_fingerprint(node_ids: Sequence[int]) -> str:
    nodes = _node_ids(node_ids)
    return canonical_sha256(
        {"layout": "ORDERED_S3_CONNECTIVITY_V1", "node_ids": nodes}
    )


def reference_geometry_fingerprint(
    node_ids: Sequence[int], reference_coordinates: Any
) -> str:
    nodes = _node_ids(node_ids)
    coordinates = _finite_array(
        reference_coordinates, (3, 3), "reference_coordinates"
    )
    return canonical_sha256(
        {
            "layout": "ORDERED_S3_REFERENCE_XYZ_GLOBAL_V1",
            "node_ids": nodes,
            "coordinates": coordinates,
        }
    )


def reference_frame_fingerprint(reference_frame: Any) -> str:
    frame = _finite_array(reference_frame, (3, 3), "reference_frame")
    _validate_right_handed_triad(frame, "reference_frame")
    return canonical_sha256(
        {
            "layout": "QUALIFIED_S3_REFERENCE_FRAME_GLOBAL_COLUMNS_V1",
            "frame": frame,
        }
    )


def reference_corner_directors_fingerprint(
    node_ids: Sequence[int],
    reference_corner_directors: Any,
) -> str:
    """Bind the element-owned physical director at every ordered corner."""

    nodes = _node_ids(node_ids)
    directors = _finite_array(
        reference_corner_directors,
        (3, 3),
        "reference_corner_directors",
    )
    for index, director in enumerate(directors):
        norm_error = abs(float(np.linalg.norm(director)) - 1.0)
        if norm_error > DIRECTOR_ORTHONORMALITY_TOLERANCE:
            raise S3CommittedStateError(
                f"reference_corner_directors[{index}] is not a unit physical director"
            )
    return canonical_sha256(
        {
            "layout": "ORDERED_S3_ELEMENT_OWNED_REFERENCE_DIRECTORS_GLOBAL_V1",
            "node_ids": nodes,
            "directors": directors,
        }
    )


def material_fingerprint(material_descriptor: Mapping[str, Any]) -> str:
    descriptor = _validate_material_descriptor(material_descriptor)
    return canonical_sha256(
        {
            "layout": "QUALIFIED_S3_MATERIAL_OR_SECTION_DESCRIPTOR_V1",
            "descriptor": descriptor,
        }
    )


def element_configuration_fingerprint(
    element_id: int,
    node_ids: Sequence[int],
    element_descriptor: Mapping[str, Any],
) -> str:
    """Bind every state-relevant element option supplied by the caller."""

    if isinstance(element_id, (bool, np.bool_)) or not isinstance(
        element_id, (int, np.integer)
    ):
        raise S3CommittedStateError("element_id must be an integer")
    descriptor = _validate_element_descriptor(element_descriptor)
    return canonical_sha256(
        {
            "layout": "QUALIFIED_S3_ELEMENT_CONFIGURATION_V3",
            "element_id": int(element_id),
            "node_ids": _node_ids(node_ids),
            "descriptor": descriptor,
        }
    )


def build_state_identity(
    *,
    element_id: int,
    node_ids: Sequence[int],
    reference_coordinates: Any,
    reference_frame: Any,
    element_descriptor: Mapping[str, Any],
    material_descriptor: Mapping[str, Any],
    num_layers: int,
    material_symmetry: str,
    equivalent_stress_measure: str,
    initial_fields: Mapping[str, Any] | None = None,
    initial_field_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact model-bound identity consumed by state validation."""

    nodes = _node_ids(node_ids)
    if isinstance(element_id, (bool, np.bool_)) or not isinstance(
        element_id, (int, np.integer)
    ):
        raise S3CommittedStateError("element_id must be an integer")
    layers = _layer_count(num_layers, "num_layers")
    symmetry, measure = _constitutive_identity(
        material_symmetry, equivalent_stress_measure
    )
    frame = _finite_array(reference_frame, (3, 3), "reference_frame")
    _validate_right_handed_triad(frame, "reference_frame")
    descriptor = _validate_element_descriptor(element_descriptor)
    material_description = _validate_material_descriptor(material_descriptor)
    if material_description["material_symmetry"] != symmetry:
        raise S3CommittedStateError(
            "material_symmetry does not match the resolved material descriptor"
        )
    if material_description["equivalent_stress_measure"] != measure:
        raise S3CommittedStateError(
            "equivalent_stress_measure does not match the resolved material descriptor"
        )
    coordinates = _finite_array(
        reference_coordinates,
        (3, 3),
        "reference_coordinates",
    )
    descriptor_normal = np.asarray(descriptor["reference_normal"], dtype=np.float64)
    polarity = _director_polarity(descriptor["director_polarity"])
    derived_frame, _local, _quality = qualified_s3_triangle_frame(
        coordinates,
        descriptor_normal,
        enforce_admission=True,
        enforce_positive_winding=True,
    )
    frame_error = float(np.max(np.abs(frame - derived_frame)))
    if frame_error > REFERENCE_FRAME_MATCH_TOLERANCE:
        raise S3CommittedStateError(
            "reference_frame does not match the admitted numbered geometry"
        )
    fields = _normalized_initial_fields(initial_fields)
    provenance_value = {} if initial_field_provenance is None else initial_field_provenance
    if not isinstance(provenance_value, Mapping):
        raise S3CommittedStateError("initial_field_provenance must be a mapping")
    provenance = _canonical_value(provenance_value, path="$.initial_field_provenance")
    assert isinstance(provenance, dict)
    return {
        "formulation_fingerprint": formulation_fingerprint(),
        "element_id": int(element_id),
        "element_configuration_fingerprint": element_configuration_fingerprint(
            int(element_id), nodes, element_descriptor
        ),
        "node_ids": nodes,
        "node_order_fingerprint": node_order_fingerprint(nodes),
        "reference_geometry_fingerprint": reference_geometry_fingerprint(
            nodes, coordinates
        ),
        "reference_frame_fingerprint": reference_frame_fingerprint(frame),
        "reference_corner_directors_fingerprint": (
            reference_corner_directors_fingerprint(
                nodes,
                np.repeat((polarity * frame[:, 2])[None, :], 3, axis=0),
            )
        ),
        "director_polarity": polarity,
        "material_fingerprint": material_fingerprint(material_descriptor),
        "initial_fields_fingerprint": _initial_fields_fingerprint(fields, provenance),
        "state_mode": STATE_MODE,
        "thickness_quadrature_id": THICKNESS_QUADRATURE_ID,
        "thickness": descriptor["thickness"],
        "num_layers": layers,
        "material_symmetry": symmetry,
        "equivalent_stress_measure": measure,
    }


def build_generalized_state_identity(
    *,
    element_id: int,
    node_ids: Sequence[int],
    reference_coordinates: Any,
    reference_frame: Any,
    element_descriptor: Mapping[str, Any],
    initial_fields: Mapping[str, Any] | None = None,
    initial_field_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one V4 generalized state to geometry, section, and initial fields."""

    nodes = _node_ids(node_ids)
    if isinstance(element_id, (bool, np.bool_)) or not isinstance(
        element_id, (int, np.integer)
    ):
        raise S3CommittedStateError("element_id must be an integer")
    frame = _finite_array(reference_frame, (3, 3), "reference_frame")
    _validate_right_handed_triad(frame, "reference_frame")
    descriptor = _validate_generalized_element_descriptor(element_descriptor)
    coordinates = _finite_array(
        reference_coordinates,
        (3, 3),
        "reference_coordinates",
    )
    descriptor_normal = np.asarray(
        descriptor["reference_normal"], dtype=np.float64
    )
    polarity = _director_polarity(descriptor["director_polarity"])
    derived_frame, _local, _quality = qualified_s3_triangle_frame(
        coordinates,
        descriptor_normal,
        enforce_admission=True,
        enforce_positive_winding=True,
    )
    frame_error = float(np.max(np.abs(frame - derived_frame)))
    if frame_error > REFERENCE_FRAME_MATCH_TOLERANCE:
        raise S3CommittedStateError(
            "reference_frame does not match the admitted numbered geometry"
        )
    fields = _normalized_initial_fields(initial_fields)
    provenance_value = (
        {} if initial_field_provenance is None else initial_field_provenance
    )
    if not isinstance(provenance_value, Mapping):
        raise S3CommittedStateError("initial_field_provenance must be a mapping")
    provenance = _canonical_value(
        provenance_value, path="$.initial_field_provenance"
    )
    assert isinstance(provenance, dict)
    return {
        "generalized_formulation_fingerprint": (
            generalized_formulation_fingerprint()
        ),
        "element_id": int(element_id),
        "element_configuration_fingerprint": (
            generalized_element_configuration_fingerprint(
                int(element_id), nodes, descriptor
            )
        ),
        "node_ids": nodes,
        "node_order_fingerprint": node_order_fingerprint(nodes),
        "reference_geometry_fingerprint": reference_geometry_fingerprint(
            nodes, coordinates
        ),
        "reference_frame_fingerprint": reference_frame_fingerprint(frame),
        "reference_corner_directors_fingerprint": (
            reference_corner_directors_fingerprint(
                nodes,
                np.repeat((polarity * frame[:, 2])[None, :], 3, axis=0),
            )
        ),
        "director_polarity": polarity,
        "section_fingerprint": generalized_section_fingerprint(
            descriptor["generalized_section"]
        ),
        "initial_fields_fingerprint": _initial_fields_fingerprint(
            fields, provenance
        ),
        "thickness": descriptor["thickness"],
        "state_mode": GENERALIZED_STATE_MODE,
    }


def _normalized_initial_fields(
    initial_fields: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    supplied = {} if initial_fields is None else dict(initial_fields)
    non_string = {key for key in supplied if not isinstance(key, str)}
    if non_string:
        raise S3CommittedStateError(
            "qualified S3 initial-field keys must be strings: "
            + ", ".join(_key_labels(non_string))
        )
    unknown = sorted(set(supplied) - set(INITIAL_FIELD_NAMES))
    if unknown:
        raise S3CommittedStateError(
            "unknown qualified S3 initial fields: " + ", ".join(unknown)
        )
    normalized: dict[str, np.ndarray] = {}
    for name in INITIAL_FIELD_NAMES:
        value = supplied.get(name, np.zeros((NUM_INTEGRATION_STATIONS, 3)))
        _validate_numeric_values(value, name)
        try:
            array = np.asarray(value, dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise S3CommittedStateError(f"{name} must contain numeric values") from exc
        if array.shape == (3,):
            array = np.broadcast_to(array, (NUM_INTEGRATION_STATIONS, 3))
        elif array.shape == (1, 3):
            array = np.broadcast_to(array, (NUM_INTEGRATION_STATIONS, 3))
        normalized[name] = _finite_array(
            array, (NUM_INTEGRATION_STATIONS, 3), name
        )
    return normalized


def _initial_fields_fingerprint(
    fields: Mapping[str, np.ndarray], provenance: Mapping[str, Any]
) -> str:
    return canonical_sha256(
        {
            "layout": "S3_INITIAL_FIELDS_STATION7_V1",
            "fields": {name: fields[name] for name in INITIAL_FIELD_NAMES},
            "provenance": provenance,
        }
    )


def initialize_zero_committed_s3_state(
    *,
    element_id: int,
    node_ids: Sequence[int],
    reference_coordinates: Any,
    reference_frame: Any,
    element_descriptor: Mapping[str, Any],
    material_descriptor: Mapping[str, Any],
    num_layers: int,
    material_symmetry: str = "isotropic",
    equivalent_stress_measure: str | None = None,
    initial_fields: Mapping[str, Any] | None = None,
    initial_field_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return owned, validated zero-history committed state.

    ``reference_frame`` contains global ``(V1, V2, Vn)`` as columns and is
    repeated for the three corner directors and the internal bubble director.
    """

    layers = _layer_count(num_layers, "num_layers")
    measure = equivalent_stress_measure
    if measure is None:
        measure = "von_mises"
    symmetry, measure = _constitutive_identity(material_symmetry, measure)

    frame = _finite_array(reference_frame, (3, 3), "reference_frame")
    _validate_right_handed_triad(frame, "reference_frame")
    descriptor = _validate_element_descriptor(element_descriptor)
    polarity = _director_polarity(descriptor["director_polarity"])
    reference_triad = reconstruct_director_triad(polarity * frame[:, 2])
    triads = np.repeat(reference_triad[np.newaxis, :, :], 4, axis=0)
    _validate_triads(triads)
    fields = _normalized_initial_fields(initial_fields)
    provenance_value = {} if initial_field_provenance is None else initial_field_provenance
    if not isinstance(provenance_value, Mapping):
        raise S3CommittedStateError("initial_field_provenance must be a mapping")
    provenance = _canonical_value(provenance_value, path="$.initial_field_provenance")
    assert isinstance(provenance, dict)
    identity = build_state_identity(
        element_id=element_id,
        node_ids=node_ids,
        reference_coordinates=reference_coordinates,
        reference_frame=frame,
        element_descriptor=element_descriptor,
        material_descriptor=material_descriptor,
        num_layers=layers,
        material_symmetry=symmetry,
        equivalent_stress_measure=measure,
        initial_fields=fields,
        initial_field_provenance=provenance,
    )

    points = NUM_INTEGRATION_STATIONS * layers
    zero_vectors = lambda: np.zeros((points, 3), dtype=np.float64)
    state: dict[str, Any] = {
        "state_schema": NONLINEAR_STATE_SCHEMA,
        "state_version": NONLINEAR_STATE_VERSION,
        "commit_status": COMMIT_STATUS,
        "state_mode": STATE_MODE,
        "formulation_id": FORMULATION_ID,
        "formulation_schema": FORMULATION_SCHEMA,
        "external_coordinate_layout_id": EXTERNAL_COORDINATE_LAYOUT_ID,
        "nonlinear_state_layout_id": NONLINEAR_STATE_LAYOUT_ID,
        "nonlinear_kinematics_id": NONLINEAR_KINEMATICS_ID,
        "director_gauge_id": DIRECTOR_GAUGE_ID,
        "external_rotation_map_id": EXTERNAL_ROTATION_MAP_ID,
        "director_increment_map_id": DIRECTOR_INCREMENT_MAP_ID,
        "surface_rotation_policy_id": SURFACE_ROTATION_POLICY_ID,
        "pl_twist_policy_id": PL_TWIST_POLICY_ID,
        "nodal_rotation_update_policy_id": NODAL_ROTATION_UPDATE_POLICY_ID,
        "pl_phase_policy_id": PL_PHASE_POLICY_ID,
        "pl_phase_margin_id": PL_PHASE_MARGIN_ID,
        "pl_minimum_twist_denominator_id": (
            PL_MINIMUM_TWIST_DENOMINATOR_ID
        ),
        "nonlinear_pl_energy_policy_id": NONLINEAR_PL_ENERGY_POLICY_ID,
        "bubble_convention": BUBBLE_CONVENTION,
        "bubble_state_role": BUBBLE_STATE_ROLE,
        "bubble_predictor_commit_policy_id": BUBBLE_PREDICTOR_COMMIT_POLICY_ID,
        "quadrature_id": QUADRATURE_ID,
        "nonlinear_policy_id": NONLINEAR_POLICY_ID,
        "recovery_policy_id": RECOVERY_POLICY_ID,
        "canonicalization_id": CANONICALIZATION_ID,
        "state_integrity_id": STATE_INTEGRITY_ID,
        "state_array_layout_id": STATE_ARRAY_LAYOUT_ID,
        "thickness_coordinate_sign_id": THICKNESS_COORDINATE_SIGN_ID,
        "stiffness_station_table_sha256": stiffness_station_table_fingerprint(),
        "lobatto_table_sha256": lobatto_table_fingerprint(),
        "state_field_manifest_sha256": state_field_manifest_fingerprint(),
        "thickness_quadrature_id": THICKNESS_QUADRATURE_ID,
        **identity,
        "reference_corner_directors": np.repeat(
            (polarity * frame[:, 2])[None, :], 3, axis=0
        ),
        "director_polarity": polarity,
        "committed_total_u": np.zeros(18, dtype=np.float64),
        "committed_nodal_rotation_matrices": np.repeat(
            np.eye(3, dtype=np.float64)[None, :, :], 3, axis=0
        ),
        "committed_pl_twist": np.zeros(3, dtype=np.float64),
        "committed_pl_turn_count": np.zeros(3, dtype=np.int64),
        "committed_pl_multiplier": np.zeros(3, dtype=np.float64),
        "committed_pl_internal_force": np.zeros(18, dtype=np.float64),
        "committed_pl_energy": 0.0,
        "committed_director_triads": triads,
        "bubble_rotation_last_increment": np.zeros(2, dtype=np.float64),
        "committed_internal_force": np.zeros(18, dtype=np.float64),
        "station_generalized_strain": np.zeros(
            (NUM_INTEGRATION_STATIONS, GENERALIZED_COMPONENTS), dtype=np.float64
        ),
        "station_generalized_resultant": np.zeros(
            (NUM_INTEGRATION_STATIONS, GENERALIZED_COMPONENTS), dtype=np.float64
        ),
        "plastic_strain": zero_vectors(),
        "alpha": np.zeros(points, dtype=np.float64),
        "layer_strain": zero_vectors(),
        "layer_strain_material": zero_vectors(),
        "kinematic_layer_strain": zero_vectors(),
        "layer_stress": zero_vectors(),
        "layer_stress_material": zero_vectors(),
        **fields,
        "initial_field_provenance": provenance,
    }
    return validate_committed_s3_state(
        seal_committed_s3_state(state),
        expected_identity=identity,
    )


def _validate_generalized_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise S3CommittedStateError("expected_identity must be a mapping")
    unknown = set(identity) - _GENERALIZED_IDENTITY_KEYS
    missing = _GENERALIZED_IDENTITY_KEYS - set(identity)
    if unknown or missing:
        raise S3CommittedStateError(
            "expected generalized identity keys mismatch; "
            f"missing={_key_labels(missing)}, unknown={_key_labels(unknown)}"
        )
    nodes = _node_ids(identity["node_ids"])
    result = {
        "generalized_formulation_fingerprint": _fingerprint(
            identity["generalized_formulation_fingerprint"],
            "generalized_formulation_fingerprint",
        ),
        "element_id": identity["element_id"],
        "element_configuration_fingerprint": _fingerprint(
            identity["element_configuration_fingerprint"],
            "element_configuration_fingerprint",
        ),
        "node_ids": nodes,
        "node_order_fingerprint": _fingerprint(
            identity["node_order_fingerprint"], "node_order_fingerprint"
        ),
        "reference_geometry_fingerprint": _fingerprint(
            identity["reference_geometry_fingerprint"],
            "reference_geometry_fingerprint",
        ),
        "reference_frame_fingerprint": _fingerprint(
            identity["reference_frame_fingerprint"],
            "reference_frame_fingerprint",
        ),
        "reference_corner_directors_fingerprint": _fingerprint(
            identity["reference_corner_directors_fingerprint"],
            "reference_corner_directors_fingerprint",
        ),
        "director_polarity": _director_polarity(
            identity["director_polarity"],
            "expected_identity director_polarity",
        ),
        "section_fingerprint": _fingerprint(
            identity["section_fingerprint"], "section_fingerprint"
        ),
        "initial_fields_fingerprint": _fingerprint(
            identity["initial_fields_fingerprint"],
            "initial_fields_fingerprint",
        ),
        "thickness": identity["thickness"],
        "state_mode": identity["state_mode"],
    }
    if isinstance(result["element_id"], (bool, np.bool_)) or not isinstance(
        result["element_id"], (int, np.integer)
    ):
        raise S3CommittedStateError(
            "expected_identity element_id must be an integer"
        )
    result["element_id"] = int(result["element_id"])
    if result["state_mode"] != GENERALIZED_STATE_MODE:
        raise S3CommittedStateError(
            "expected_identity generalized state_mode mismatch"
        )
    if isinstance(result["thickness"], (bool, np.bool_)) or not isinstance(
        result["thickness"], (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError(
            "expected_identity thickness must be finite and positive"
        )
    result["thickness"] = float(result["thickness"])
    if not math.isfinite(result["thickness"]) or result["thickness"] <= 0.0:
        raise S3CommittedStateError(
            "expected_identity thickness must be finite and positive"
        )
    if (
        result["generalized_formulation_fingerprint"]
        != generalized_formulation_fingerprint()
    ):
        raise S3CommittedStateError(
            "expected generalized formulation fingerprint mismatch"
        )
    if result["node_order_fingerprint"] != node_order_fingerprint(nodes):
        raise S3CommittedStateError(
            "expected_identity node-order fingerprint mismatch"
        )
    return result


def initialize_zero_committed_s3_generalized_state(
    *,
    element_id: int,
    node_ids: Sequence[int],
    reference_coordinates: Any,
    reference_frame: Any,
    element_descriptor: Mapping[str, Any],
    constitutive: Any,
    drilling_scale: Any,
    initial_fields: Mapping[str, Any] | None = None,
    initial_field_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete zero-history V4 stateless-section state."""

    frame = _finite_array(reference_frame, (3, 3), "reference_frame")
    _validate_right_handed_triad(frame, "reference_frame")
    descriptor = _validate_generalized_element_descriptor(element_descriptor)
    polarity = _director_polarity(descriptor["director_polarity"])
    coordinates = _finite_array(
        reference_coordinates, (3, 3), "reference_coordinates"
    )
    section = _finite_array(constitutive, (8, 8), "constitutive")
    if not np.array_equal(section, section.T):
        raise S3CommittedStateError(
            "generalized section constitutive matrix must be exactly symmetric"
        )
    scale = _finite_state_scalar(drilling_scale, "drilling_scale")
    if scale <= 0.0:
        raise S3CommittedStateError("drilling_scale must be strictly positive")
    fields = _normalized_initial_fields(initial_fields)
    provenance_value = (
        {} if initial_field_provenance is None else initial_field_provenance
    )
    if not isinstance(provenance_value, Mapping):
        raise S3CommittedStateError("initial_field_provenance must be a mapping")
    provenance = _canonical_value(
        provenance_value, path="$.initial_field_provenance"
    )
    assert isinstance(provenance, dict)
    try:
        initial_prestrain, initial_resultant = (
            integrate_generalized_initial_fields(
                fields,
                descriptor["thickness"],
                station_count=NUM_INTEGRATION_STATIONS,
            )
        )
    except (TypeError, ValueError) as exc:
        raise S3CommittedStateError(str(exc)) from exc
    identity = build_generalized_state_identity(
        element_id=element_id,
        node_ids=node_ids,
        reference_coordinates=coordinates,
        reference_frame=frame,
        element_descriptor=element_descriptor,
        initial_fields=fields,
        initial_field_provenance=provenance,
    )
    reference_triad = reconstruct_director_triad(polarity * frame[:, 2])
    triads = np.repeat(reference_triad[np.newaxis, :, :], 4, axis=0)
    _validate_triads(triads)
    zero_station_strain = np.zeros(
        (NUM_INTEGRATION_STATIONS, GENERALIZED_COMPONENTS),
        dtype=np.float64,
    )
    zero_station_resultant = (
        (zero_station_strain - initial_prestrain) @ section.T
        + initial_resultant
    )
    state: dict[str, Any] = {
        "state_schema": GENERALIZED_NONLINEAR_STATE_SCHEMA,
        "state_version": GENERALIZED_NONLINEAR_STATE_VERSION,
        "commit_status": COMMIT_STATUS,
        "state_mode": GENERALIZED_STATE_MODE,
        "formulation_id": FORMULATION_ID,
        "formulation_schema": FORMULATION_SCHEMA,
        "external_coordinate_layout_id": EXTERNAL_COORDINATE_LAYOUT_ID,
        "nonlinear_state_layout_id": GENERALIZED_NONLINEAR_STATE_LAYOUT_ID,
        "nonlinear_kinematics_id": NONLINEAR_KINEMATICS_ID,
        "director_gauge_id": DIRECTOR_GAUGE_ID,
        "external_rotation_map_id": EXTERNAL_ROTATION_MAP_ID,
        "director_increment_map_id": DIRECTOR_INCREMENT_MAP_ID,
        "surface_rotation_policy_id": SURFACE_ROTATION_POLICY_ID,
        "pl_twist_policy_id": PL_TWIST_POLICY_ID,
        "nodal_rotation_update_policy_id": NODAL_ROTATION_UPDATE_POLICY_ID,
        "pl_phase_policy_id": PL_PHASE_POLICY_ID,
        "pl_phase_margin_id": PL_PHASE_MARGIN_ID,
        "pl_minimum_twist_denominator_id": (
            PL_MINIMUM_TWIST_DENOMINATOR_ID
        ),
        "nonlinear_pl_energy_policy_id": NONLINEAR_PL_ENERGY_POLICY_ID,
        "bubble_convention": BUBBLE_CONVENTION,
        "bubble_state_role": BUBBLE_STATE_ROLE,
        "bubble_predictor_commit_policy_id": BUBBLE_PREDICTOR_COMMIT_POLICY_ID,
        "quadrature_id": QUADRATURE_ID,
        "nonlinear_policy_id": NONLINEAR_POLICY_ID,
        "generalized_section_integration_id": GENERALIZED_SECTION_INTEGRATION_ID,
        "generalized_section_history_policy_id": (
            GENERALIZED_SECTION_HISTORY_POLICY_ID
        ),
        "generalized_initial_field_policy_id": (
            GENERALIZED_INITIAL_FIELD_POLICY_ID
        ),
        "recovery_scope": "section_resultants_only",
        "physical_layer_recovery_available": False,
        "canonicalization_id": CANONICALIZATION_ID,
        "state_integrity_id": STATE_INTEGRITY_ID,
        "stiffness_station_table_sha256": stiffness_station_table_fingerprint(),
        "generalized_state_field_manifest_sha256": (
            generalized_state_field_manifest_fingerprint()
        ),
        **identity,
        "reference_corner_directors": np.repeat(
            (polarity * frame[:, 2])[None, :], 3, axis=0
        ),
        "director_polarity": polarity,
        "committed_total_u": np.zeros(18, dtype=np.float64),
        "committed_nodal_rotation_matrices": np.repeat(
            np.eye(3, dtype=np.float64)[None, :, :], 3, axis=0
        ),
        "committed_pl_twist": np.zeros(3, dtype=np.float64),
        "committed_pl_turn_count": np.zeros(3, dtype=np.int64),
        "committed_pl_multiplier": np.zeros(3, dtype=np.float64),
        "committed_pl_internal_force": np.zeros(18, dtype=np.float64),
        "committed_pl_energy": 0.0,
        "committed_director_triads": triads,
        "bubble_rotation_last_increment": np.zeros(2, dtype=np.float64),
        "committed_internal_force": np.zeros(18, dtype=np.float64),
        "station_generalized_strain": zero_station_strain,
        "station_generalized_resultant": zero_station_resultant,
        "initial_generalized_prestrain": initial_prestrain,
        "initial_generalized_resultant": initial_resultant,
        **fields,
        "initial_field_provenance": provenance,
    }
    area = 0.5 * float(
        np.linalg.norm(
            np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
        )
    )
    return validate_committed_s3_generalized_state(
        seal_committed_s3_state(state),
        expected_identity=identity,
        expected_constitutive=section,
        expected_drilling_scale=scale,
        expected_reference_area=area,
    )


def validate_committed_s3_generalized_state(
    state: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_constitutive: Any,
    expected_drilling_scale: Any,
    expected_reference_area: Any,
    expected_committed_total_u: Any | None = None,
) -> dict[str, Any]:
    """Strictly validate and own one stateless generalized-section state."""

    if not isinstance(state, Mapping):
        raise S3CommittedStateError(
            "qualified S3 generalized committed state must be a mapping"
        )
    non_string = {key for key in state if not isinstance(key, str)}
    if non_string:
        raise S3CommittedStateError(
            "committed state keys must be strings: "
            + ", ".join(_key_labels(non_string))
        )
    missing = _GENERALIZED_STATE_KEYS - set(state)
    unknown = set(state) - _GENERALIZED_STATE_KEYS
    if missing or unknown:
        raise S3CommittedStateError(
            "generalized committed state keys mismatch; "
            f"missing={_key_labels(missing)}, unknown={_key_labels(unknown)}"
        )
    expected_constants: dict[str, Any] = {
        "state_schema": GENERALIZED_NONLINEAR_STATE_SCHEMA,
        "state_version": GENERALIZED_NONLINEAR_STATE_VERSION,
        "commit_status": COMMIT_STATUS,
        "state_mode": GENERALIZED_STATE_MODE,
        "formulation_id": FORMULATION_ID,
        "formulation_schema": FORMULATION_SCHEMA,
        "external_coordinate_layout_id": EXTERNAL_COORDINATE_LAYOUT_ID,
        "nonlinear_state_layout_id": GENERALIZED_NONLINEAR_STATE_LAYOUT_ID,
        "nonlinear_kinematics_id": NONLINEAR_KINEMATICS_ID,
        "director_gauge_id": DIRECTOR_GAUGE_ID,
        "external_rotation_map_id": EXTERNAL_ROTATION_MAP_ID,
        "director_increment_map_id": DIRECTOR_INCREMENT_MAP_ID,
        "surface_rotation_policy_id": SURFACE_ROTATION_POLICY_ID,
        "pl_twist_policy_id": PL_TWIST_POLICY_ID,
        "nodal_rotation_update_policy_id": NODAL_ROTATION_UPDATE_POLICY_ID,
        "pl_phase_policy_id": PL_PHASE_POLICY_ID,
        "pl_phase_margin_id": PL_PHASE_MARGIN_ID,
        "pl_minimum_twist_denominator_id": (
            PL_MINIMUM_TWIST_DENOMINATOR_ID
        ),
        "nonlinear_pl_energy_policy_id": NONLINEAR_PL_ENERGY_POLICY_ID,
        "bubble_convention": BUBBLE_CONVENTION,
        "bubble_state_role": BUBBLE_STATE_ROLE,
        "bubble_predictor_commit_policy_id": BUBBLE_PREDICTOR_COMMIT_POLICY_ID,
        "quadrature_id": QUADRATURE_ID,
        "nonlinear_policy_id": NONLINEAR_POLICY_ID,
        "generalized_section_integration_id": GENERALIZED_SECTION_INTEGRATION_ID,
        "generalized_section_history_policy_id": (
            GENERALIZED_SECTION_HISTORY_POLICY_ID
        ),
        "generalized_initial_field_policy_id": (
            GENERALIZED_INITIAL_FIELD_POLICY_ID
        ),
        "recovery_scope": "section_resultants_only",
        "physical_layer_recovery_available": False,
        "canonicalization_id": CANONICALIZATION_ID,
        "state_integrity_id": STATE_INTEGRITY_ID,
        "stiffness_station_table_sha256": stiffness_station_table_fingerprint(),
        "generalized_state_field_manifest_sha256": (
            generalized_state_field_manifest_fingerprint()
        ),
    }
    for key, expected_value in expected_constants.items():
        actual = state[key]
        if key == "state_version":
            matches = (
                isinstance(actual, (int, np.integer))
                and not isinstance(actual, (bool, np.bool_))
                and int(actual) == int(expected_value)
            )
        elif key == "physical_layer_recovery_available":
            matches = isinstance(actual, (bool, np.bool_)) and not bool(actual)
        else:
            matches = isinstance(actual, str) and actual == expected_value
        if not matches:
            raise S3CommittedStateError(f"incompatible {key}")

    nodes = _node_ids(state["node_ids"])
    fingerprints = {
        name: _fingerprint(state[name], name)
        for name in (
            "generalized_formulation_fingerprint",
            "element_configuration_fingerprint",
            "node_order_fingerprint",
            "reference_geometry_fingerprint",
            "reference_frame_fingerprint",
            "reference_corner_directors_fingerprint",
            "section_fingerprint",
            "initial_fields_fingerprint",
            "state_integrity_sha256",
        )
    }
    element_id_value = state["element_id"]
    if isinstance(element_id_value, (bool, np.bool_)) or not isinstance(
        element_id_value, (int, np.integer)
    ):
        raise S3CommittedStateError("element_id must be an integer")
    element_id = int(element_id_value)
    if fingerprints["node_order_fingerprint"] != node_order_fingerprint(nodes):
        raise S3CommittedStateError("node-order fingerprint mismatch")
    if (
        fingerprints["generalized_formulation_fingerprint"]
        != generalized_formulation_fingerprint()
    ):
        raise S3CommittedStateError(
            "generalized formulation fingerprint mismatch"
        )
    normalized: dict[str, Any] = dict(expected_constants)
    normalized.update(
        {
            "generalized_formulation_fingerprint": fingerprints[
                "generalized_formulation_fingerprint"
            ],
            "element_id": element_id,
            "element_configuration_fingerprint": fingerprints[
                "element_configuration_fingerprint"
            ],
            "node_ids": nodes,
            "node_order_fingerprint": fingerprints["node_order_fingerprint"],
            "reference_geometry_fingerprint": fingerprints[
                "reference_geometry_fingerprint"
            ],
            "reference_frame_fingerprint": fingerprints[
                "reference_frame_fingerprint"
            ],
            "reference_corner_directors_fingerprint": fingerprints[
                "reference_corner_directors_fingerprint"
            ],
            "director_polarity": _director_polarity(
                state["director_polarity"]
            ),
            "section_fingerprint": fingerprints["section_fingerprint"],
            "initial_fields_fingerprint": fingerprints[
                "initial_fields_fingerprint"
            ],
            "thickness": _finite_state_scalar(state["thickness"], "thickness"),
            "state_integrity_sha256": fingerprints["state_integrity_sha256"],
        }
    )
    for key, shape in _GENERALIZED_ARRAY_SHAPES.items():
        normalized[key] = _finite_state_array(state[key], shape, key)
    normalized["committed_pl_turn_count"] = _integer_state_array(
        state["committed_pl_turn_count"], (3,), "committed_pl_turn_count"
    )
    normalized["committed_pl_energy"] = _finite_state_scalar(
        state["committed_pl_energy"], "committed_pl_energy"
    )
    if normalized["committed_pl_energy"] < 0.0:
        raise S3CommittedStateError("committed PL energy must be nonnegative")
    if normalized["thickness"] <= 0.0:
        raise S3CommittedStateError("generalized state thickness must be positive")
    provenance_value = state["initial_field_provenance"]
    if not isinstance(provenance_value, Mapping):
        raise S3CommittedStateError("initial_field_provenance must be a mapping")
    provenance = _canonical_value(
        provenance_value, path="$.initial_field_provenance"
    )
    assert isinstance(provenance, dict)
    normalized["initial_field_provenance"] = provenance
    fields = {name: normalized[name] for name in INITIAL_FIELD_NAMES}
    current_initial_fingerprint = _initial_fields_fingerprint(
        fields, provenance
    )
    if current_initial_fingerprint != normalized["initial_fields_fingerprint"]:
        raise S3CommittedStateError("initial-fields fingerprint mismatch")
    try:
        expected_initial_prestrain, expected_initial_resultant = (
            integrate_generalized_initial_fields(
                fields,
                normalized["thickness"],
                station_count=NUM_INTEGRATION_STATIONS,
            )
        )
    except (TypeError, ValueError) as exc:
        raise S3CommittedStateError(str(exc)) from exc
    if not np.array_equal(
        normalized["initial_generalized_prestrain"],
        expected_initial_prestrain,
    ):
        raise S3CommittedStateError(
            "initial generalized prestrain contradicts the public shell field"
        )
    if not np.array_equal(
        normalized["initial_generalized_resultant"],
        expected_initial_resultant,
    ):
        raise S3CommittedStateError(
            "initial generalized resultant contradicts the public shell stress field"
        )
    _validate_triads(normalized["committed_director_triads"])
    _validate_nodal_rotation_state(
        normalized["reference_corner_directors"],
        normalized["committed_nodal_rotation_matrices"],
        normalized["committed_director_triads"],
    )
    actual_reference_fingerprint = reference_corner_directors_fingerprint(
        nodes, normalized["reference_corner_directors"]
    )
    if (
        actual_reference_fingerprint
        != normalized["reference_corner_directors_fingerprint"]
    ):
        raise S3CommittedStateError(
            "reference corner directors contradict their model-bound fingerprint"
        )
    principal_twist = normalized["committed_pl_twist"] - (
        2.0 * math.pi * normalized["committed_pl_turn_count"]
    )
    if np.any(principal_twist <= -math.pi) or np.any(principal_twist > math.pi):
        raise S3CommittedStateError(
            "committed PL turn count is incompatible with its unwrapped twist"
        )
    canonical_turn_count = np.asarray(
        [
            math.ceil((float(twist) - math.pi) / (2.0 * math.pi))
            for twist in normalized["committed_pl_twist"]
        ],
        dtype=np.int64,
    )
    if not np.array_equal(
        normalized["committed_pl_turn_count"], canonical_turn_count
    ):
        raise S3CommittedStateError(
            "committed PL turn count is not the canonical (-pi, pi] phase"
        )
    if np.any(normalized["bubble_rotation_last_increment"] != 0.0):
        raise S3CommittedStateError(
            "committed bubble predictor must reset to exact zero"
        )

    # Reject a state from another geometry/section before interpreting its
    # redundant resultants with the current section matrix.
    state_identity = {
        "generalized_formulation_fingerprint": normalized[
            "generalized_formulation_fingerprint"
        ],
        "element_id": normalized["element_id"],
        "element_configuration_fingerprint": normalized[
            "element_configuration_fingerprint"
        ],
        "node_ids": nodes,
        "node_order_fingerprint": normalized["node_order_fingerprint"],
        "reference_geometry_fingerprint": normalized[
            "reference_geometry_fingerprint"
        ],
        "reference_frame_fingerprint": normalized[
            "reference_frame_fingerprint"
        ],
        "reference_corner_directors_fingerprint": normalized[
            "reference_corner_directors_fingerprint"
        ],
        "director_polarity": normalized["director_polarity"],
        "section_fingerprint": normalized["section_fingerprint"],
        "initial_fields_fingerprint": normalized[
            "initial_fields_fingerprint"
        ],
        "thickness": normalized["thickness"],
        "state_mode": normalized["state_mode"],
    }
    expected = _validate_generalized_identity(expected_identity)
    if state_identity != expected:
        mismatched = sorted(
            key
            for key in _GENERALIZED_IDENTITY_KEYS
            if state_identity[key] != expected[key]
        )
        raise S3CommittedStateError(
            "generalized state identity does not match current model: "
            + ", ".join(mismatched)
        )

    section = _finite_array(
        expected_constitutive, (8, 8), "expected_constitutive"
    )
    if not np.array_equal(section, section.T):
        raise S3CommittedStateError(
            "expected generalized constitutive matrix must be exactly symmetric"
        )
    expected_resultants = (
        (
            normalized["station_generalized_strain"]
            - normalized["initial_generalized_prestrain"]
        )
        @ section.T
        + normalized["initial_generalized_resultant"]
    )
    if not np.array_equal(
        normalized["station_generalized_resultant"], expected_resultants
    ):
        raise S3CommittedStateError(
            "station generalized resultants contradict the bound section"
        )
    drilling_scale = _finite_state_scalar(
        expected_drilling_scale, "expected_drilling_scale"
    )
    if drilling_scale <= 0.0:
        raise S3CommittedStateError(
            "expected_drilling_scale must be strictly positive"
        )
    expected_multiplier = drilling_scale * normalized["committed_pl_twist"]
    if not np.array_equal(
        normalized["committed_pl_multiplier"], expected_multiplier
    ):
        raise S3CommittedStateError(
            "committed PL multiplier contradicts the bound drill scale"
        )
    area = _finite_state_scalar(
        expected_reference_area, "expected_reference_area"
    )
    if area <= 0.0:
        raise S3CommittedStateError(
            "expected_reference_area must be strictly positive"
        )
    gram = (area / 12.0) * np.asarray(PL_GRAM_NUMERATOR, dtype=np.float64)
    mixed_force = np.asarray(
        [
            sum(
                float(gram[row, column])
                * float(normalized["committed_pl_multiplier"][column])
                for column in range(3)
            )
            for row in range(3)
        ],
        dtype=np.float64,
    )
    expected_energy = sum(
        0.5 * float(normalized["committed_pl_twist"][row]) * float(mixed_force[row])
        for row in range(3)
    )
    if normalized["committed_pl_energy"] != expected_energy:
        raise S3CommittedStateError(
            "committed PL energy contradicts twist, multiplier, and Gram matrix"
        )

    if expected_committed_total_u is not None:
        expected_u = _finite_array(
            expected_committed_total_u, (18,), "expected_committed_total_u"
        )
        if not np.array_equal(normalized["committed_total_u"], expected_u):
            raise S3CommittedStateError(
                "committed_total_u does not match the current global displacement slice"
            )
    actual_integrity = _state_integrity_sha256(normalized)
    if actual_integrity != normalized["state_integrity_sha256"]:
        raise S3CommittedStateError(
            "committed generalized state integrity fingerprint mismatch"
        )
    return normalized


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise S3CommittedStateError("expected_identity must be a mapping")
    unknown = set(identity) - _IDENTITY_KEYS
    missing = _IDENTITY_KEYS - set(identity)
    if unknown or missing:
        raise S3CommittedStateError(
            f"expected_identity keys mismatch; missing={_key_labels(missing)}, "
            f"unknown={_key_labels(unknown)}"
        )
    nodes = _node_ids(identity["node_ids"])
    result = {
        "formulation_fingerprint": _fingerprint(
            identity["formulation_fingerprint"], "formulation_fingerprint"
        ),
        "element_id": identity["element_id"],
        "element_configuration_fingerprint": _fingerprint(
            identity["element_configuration_fingerprint"],
            "element_configuration_fingerprint",
        ),
        "node_ids": nodes,
        "node_order_fingerprint": _fingerprint(
            identity["node_order_fingerprint"], "node_order_fingerprint"
        ),
        "reference_geometry_fingerprint": _fingerprint(
            identity["reference_geometry_fingerprint"],
            "reference_geometry_fingerprint",
        ),
        "reference_frame_fingerprint": _fingerprint(
            identity["reference_frame_fingerprint"],
            "reference_frame_fingerprint",
        ),
        "reference_corner_directors_fingerprint": _fingerprint(
            identity["reference_corner_directors_fingerprint"],
            "reference_corner_directors_fingerprint",
        ),
        "director_polarity": _director_polarity(
            identity["director_polarity"],
            "expected_identity director_polarity",
        ),
        "material_fingerprint": _fingerprint(
            identity["material_fingerprint"], "material_fingerprint"
        ),
        "initial_fields_fingerprint": _fingerprint(
            identity["initial_fields_fingerprint"], "initial_fields_fingerprint"
        ),
        "state_mode": identity["state_mode"],
        "thickness_quadrature_id": identity["thickness_quadrature_id"],
        "thickness": identity["thickness"],
        "num_layers": identity["num_layers"],
        "material_symmetry": identity["material_symmetry"],
        "equivalent_stress_measure": identity["equivalent_stress_measure"],
    }
    if isinstance(result["element_id"], (bool, np.bool_)) or not isinstance(
        result["element_id"], (int, np.integer)
    ):
        raise S3CommittedStateError("expected_identity element_id must be an integer")
    result["element_id"] = int(result["element_id"])
    if result["state_mode"] != STATE_MODE:
        raise S3CommittedStateError("expected_identity state_mode mismatch")
    if result["thickness_quadrature_id"] != THICKNESS_QUADRATURE_ID:
        raise S3CommittedStateError(
            "expected_identity thickness_quadrature_id mismatch"
        )
    if isinstance(result["thickness"], (bool, np.bool_)) or not isinstance(
        result["thickness"], (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError("expected_identity thickness must be finite and positive")
    result["thickness"] = float(result["thickness"])
    if not math.isfinite(result["thickness"]) or result["thickness"] <= 0.0:
        raise S3CommittedStateError("expected_identity thickness must be finite and positive")
    result["num_layers"] = _layer_count(
        result["num_layers"], "expected_identity num_layers"
    )
    result["material_symmetry"], result["equivalent_stress_measure"] = (
        _constitutive_identity(
            result["material_symmetry"], result["equivalent_stress_measure"]
        )
    )
    if result["node_order_fingerprint"] != node_order_fingerprint(nodes):
        raise S3CommittedStateError("expected_identity node-order fingerprint mismatch")
    return result


def validate_committed_s3_state(
    state: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_num_layers: int | None = None,
    expected_committed_total_u: Any | None = None,
) -> dict[str, Any]:
    """Strictly validate state and return a normalized, deeply owned mapping.

    Model compatibility for geometry and material is established only when the
    caller supplies ``expected_identity`` built from the current model.
    """

    if not isinstance(state, Mapping):
        raise S3CommittedStateError("qualified S3 committed state must be a mapping")
    non_string = {key for key in state if not isinstance(key, str)}
    if non_string:
        raise S3CommittedStateError(
            "committed state keys must be strings: "
            + ", ".join(_key_labels(non_string))
        )
    missing = _STATE_KEYS - set(state)
    unknown = set(state) - _STATE_KEYS
    if missing or unknown:
        raise S3CommittedStateError(
            f"committed state keys mismatch; missing={_key_labels(missing)}, "
            f"unknown={_key_labels(unknown)}"
        )

    expected_constants = {
        "state_schema": NONLINEAR_STATE_SCHEMA,
        "state_version": NONLINEAR_STATE_VERSION,
        "commit_status": COMMIT_STATUS,
        "state_mode": STATE_MODE,
        "formulation_id": FORMULATION_ID,
        "formulation_schema": FORMULATION_SCHEMA,
        "external_coordinate_layout_id": EXTERNAL_COORDINATE_LAYOUT_ID,
        "nonlinear_state_layout_id": NONLINEAR_STATE_LAYOUT_ID,
        "nonlinear_kinematics_id": NONLINEAR_KINEMATICS_ID,
        "director_gauge_id": DIRECTOR_GAUGE_ID,
        "external_rotation_map_id": EXTERNAL_ROTATION_MAP_ID,
        "director_increment_map_id": DIRECTOR_INCREMENT_MAP_ID,
        "surface_rotation_policy_id": SURFACE_ROTATION_POLICY_ID,
        "pl_twist_policy_id": PL_TWIST_POLICY_ID,
        "nodal_rotation_update_policy_id": NODAL_ROTATION_UPDATE_POLICY_ID,
        "pl_phase_policy_id": PL_PHASE_POLICY_ID,
        "pl_phase_margin_id": PL_PHASE_MARGIN_ID,
        "pl_minimum_twist_denominator_id": (
            PL_MINIMUM_TWIST_DENOMINATOR_ID
        ),
        "nonlinear_pl_energy_policy_id": NONLINEAR_PL_ENERGY_POLICY_ID,
        "bubble_convention": BUBBLE_CONVENTION,
        "bubble_state_role": BUBBLE_STATE_ROLE,
        "bubble_predictor_commit_policy_id": BUBBLE_PREDICTOR_COMMIT_POLICY_ID,
        "quadrature_id": QUADRATURE_ID,
        "nonlinear_policy_id": NONLINEAR_POLICY_ID,
        "recovery_policy_id": RECOVERY_POLICY_ID,
        "canonicalization_id": CANONICALIZATION_ID,
        "state_integrity_id": STATE_INTEGRITY_ID,
        "state_array_layout_id": STATE_ARRAY_LAYOUT_ID,
        "thickness_coordinate_sign_id": THICKNESS_COORDINATE_SIGN_ID,
        "stiffness_station_table_sha256": stiffness_station_table_fingerprint(),
        "lobatto_table_sha256": lobatto_table_fingerprint(),
        "state_field_manifest_sha256": state_field_manifest_fingerprint(),
        "thickness_quadrature_id": THICKNESS_QUADRATURE_ID,
    }
    for key, expected in expected_constants.items():
        actual = state[key]
        if key == "state_version":
            matches = isinstance(actual, (int, np.integer)) and not isinstance(
                actual, (bool, np.bool_)
            ) and int(actual) == expected
        else:
            matches = isinstance(actual, str) and actual == expected
        if not matches:
            raise S3CommittedStateError(f"incompatible {key}")

    nodes = _node_ids(state["node_ids"])
    fingerprint_values = {
        "formulation_fingerprint": _fingerprint(
            state["formulation_fingerprint"], "formulation_fingerprint"
        ),
        "element_configuration_fingerprint": _fingerprint(
            state["element_configuration_fingerprint"],
            "element_configuration_fingerprint",
        ),
        "node_order_fingerprint": _fingerprint(
            state["node_order_fingerprint"], "node_order_fingerprint"
        ),
        "reference_geometry_fingerprint": _fingerprint(
            state["reference_geometry_fingerprint"],
            "reference_geometry_fingerprint",
        ),
        "reference_frame_fingerprint": _fingerprint(
            state["reference_frame_fingerprint"],
            "reference_frame_fingerprint",
        ),
        "reference_corner_directors_fingerprint": _fingerprint(
            state["reference_corner_directors_fingerprint"],
            "reference_corner_directors_fingerprint",
        ),
        "material_fingerprint": _fingerprint(
            state["material_fingerprint"], "material_fingerprint"
        ),
        "initial_fields_fingerprint": _fingerprint(
            state["initial_fields_fingerprint"], "initial_fields_fingerprint"
        ),
        "state_integrity_sha256": _fingerprint(
            state["state_integrity_sha256"], "state_integrity_sha256"
        ),
    }
    element_id_value = state["element_id"]
    if isinstance(element_id_value, (bool, np.bool_)) or not isinstance(
        element_id_value, (int, np.integer)
    ):
        raise S3CommittedStateError("element_id must be an integer")
    element_id = int(element_id_value)
    if fingerprint_values["node_order_fingerprint"] != node_order_fingerprint(nodes):
        raise S3CommittedStateError("node-order fingerprint mismatch")
    if fingerprint_values["formulation_fingerprint"] != formulation_fingerprint():
        raise S3CommittedStateError("formulation fingerprint mismatch")

    layers = _layer_count(state["num_layers"], "num_layers")
    thickness_value = state["thickness"]
    if isinstance(thickness_value, (bool, np.bool_)) or not isinstance(
        thickness_value, (int, float, np.integer, np.floating)
    ):
        raise S3CommittedStateError("thickness must be finite and positive")
    thickness = float(thickness_value)
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise S3CommittedStateError("thickness must be finite and positive")
    if expected_num_layers is not None:
        expected_layers = _layer_count(
            expected_num_layers, "expected_num_layers"
        )
        if layers != expected_layers:
            raise S3CommittedStateError("num_layers does not match the current analysis")

    symmetry, measure = _constitutive_identity(
        state["material_symmetry"], state["equivalent_stress_measure"]
    )

    normalized: dict[str, Any] = dict(expected_constants)
    normalized.update(
        {
            "formulation_fingerprint": fingerprint_values["formulation_fingerprint"],
            "element_id": element_id,
            "element_configuration_fingerprint": fingerprint_values[
                "element_configuration_fingerprint"
            ],
            "node_ids": nodes,
            "node_order_fingerprint": fingerprint_values["node_order_fingerprint"],
            "reference_geometry_fingerprint": fingerprint_values[
                "reference_geometry_fingerprint"
            ],
            "reference_frame_fingerprint": fingerprint_values[
                "reference_frame_fingerprint"
            ],
            "reference_corner_directors_fingerprint": fingerprint_values[
                "reference_corner_directors_fingerprint"
            ],
            "director_polarity": _director_polarity(
                state["director_polarity"]
            ),
            "material_fingerprint": fingerprint_values["material_fingerprint"],
            "initial_fields_fingerprint": fingerprint_values[
                "initial_fields_fingerprint"
            ],
            "state_integrity_sha256": fingerprint_values[
                "state_integrity_sha256"
            ],
            "num_layers": layers,
            "thickness": thickness,
            "material_symmetry": symmetry,
            "equivalent_stress_measure": measure,
        }
    )
    for key, shape in _ARRAY_SHAPES_FIXED.items():
        normalized[key] = _finite_state_array(state[key], shape, key)
    normalized["committed_pl_turn_count"] = _integer_state_array(
        state["committed_pl_turn_count"],
        (3,),
        "committed_pl_turn_count",
    )
    normalized["committed_pl_energy"] = _finite_state_scalar(
        state["committed_pl_energy"],
        "committed_pl_energy",
    )
    if normalized["committed_pl_energy"] < 0.0:
        raise S3CommittedStateError("committed PL energy must be nonnegative")
    _validate_triads(normalized["committed_director_triads"])
    _validate_nodal_rotation_state(
        normalized["reference_corner_directors"],
        normalized["committed_nodal_rotation_matrices"],
        normalized["committed_director_triads"],
    )
    actual_reference_directors_fingerprint = (
        reference_corner_directors_fingerprint(
            nodes,
            normalized["reference_corner_directors"],
        )
    )
    if actual_reference_directors_fingerprint != normalized[
        "reference_corner_directors_fingerprint"
    ]:
        raise S3CommittedStateError(
            "reference corner directors contradict their model-bound fingerprint"
        )
    principal_twist = normalized["committed_pl_twist"] - (
        2.0 * math.pi * normalized["committed_pl_turn_count"]
    )
    if np.any(principal_twist <= -math.pi) or np.any(principal_twist > math.pi):
        raise S3CommittedStateError(
            "committed PL turn count is incompatible with its unwrapped twist"
        )
    canonical_turn_count = np.asarray(
        [
            math.ceil((float(twist) - math.pi) / (2.0 * math.pi))
            for twist in normalized["committed_pl_twist"]
        ],
        dtype=np.int64,
    )
    if not np.array_equal(
        normalized["committed_pl_turn_count"], canonical_turn_count
    ):
        raise S3CommittedStateError(
            "committed PL turn count is not the canonical (-pi, pi] phase"
        )
    if np.any(normalized["bubble_rotation_last_increment"] != 0.0):
        raise S3CommittedStateError(
            "committed bubble predictor must reset to exact zero in the frozen gauge policy"
        )

    points = NUM_INTEGRATION_STATIONS * layers
    for key in _LAYER_VECTOR_FIELDS:
        normalized[key] = _finite_state_array(state[key], (points, 3), key)
    normalized["alpha"] = _finite_state_array(
        state["alpha"], (points,), "alpha"
    )
    if np.any(normalized["alpha"] < 0.0):
        raise S3CommittedStateError("material hardening alpha must be nonnegative")

    z_layers, layer_weights = qualified_s3_lobatto_layers(layers, thickness)
    station_strain = normalized["station_generalized_strain"]
    expected_kinematic = (
        station_strain[:, None, :3]
        + z_layers[None, :, None] * station_strain[:, None, 3:6]
    ).reshape(points, 3)
    redundancy_scale = max(
        1.0,
        float(np.max(np.abs(expected_kinematic))),
        float(np.max(np.abs(normalized["kinematic_layer_strain"]))),
    )
    redundancy_tolerance = (
        STATE_REDUNDANCY_TOLERANCE_FACTOR
        * np.finfo(np.float64).eps
        * redundancy_scale
    )
    if not np.allclose(
        normalized["kinematic_layer_strain"],
        expected_kinematic,
        rtol=0.0,
        atol=redundancy_tolerance,
    ):
        raise S3CommittedStateError(
            "kinematic layer strain contradicts station generalized strain"
        )

    local_stress = normalized["layer_stress"].reshape(
        NUM_INTEGRATION_STATIONS,
        layers,
        3,
    )
    expected_membrane = np.einsum("l,gli->gi", layer_weights, local_stress)
    expected_bending = np.einsum(
        "l,l,gli->gi",
        layer_weights,
        z_layers,
        local_stress,
    )
    expected_inplane_resultants = np.concatenate(
        (expected_membrane, expected_bending),
        axis=1,
    )
    stored_inplane_resultants = normalized["station_generalized_resultant"][:, :6]
    resultant_scale = max(
        1.0,
        float(np.max(np.abs(expected_inplane_resultants))),
        float(np.max(np.abs(stored_inplane_resultants))),
    )
    resultant_tolerance = (
        STATE_REDUNDANCY_TOLERANCE_FACTOR
        * np.finfo(np.float64).eps
        * resultant_scale
    )
    if not np.allclose(
        stored_inplane_resultants,
        expected_inplane_resultants,
        rtol=0.0,
        atol=resultant_tolerance,
    ):
        raise S3CommittedStateError(
            "station generalized resultants contradict stored layer stress"
        )

    provenance_value = state["initial_field_provenance"]
    if not isinstance(provenance_value, Mapping):
        raise S3CommittedStateError("initial_field_provenance must be a mapping")
    provenance = _canonical_value(
        provenance_value, path="$.initial_field_provenance"
    )
    assert isinstance(provenance, dict)
    normalized["initial_field_provenance"] = provenance
    current_initial_fingerprint = _initial_fields_fingerprint(normalized, provenance)
    if current_initial_fingerprint != normalized["initial_fields_fingerprint"]:
        raise S3CommittedStateError("initial-fields fingerprint mismatch")

    state_identity = {
        "formulation_fingerprint": normalized["formulation_fingerprint"],
        "element_id": normalized["element_id"],
        "element_configuration_fingerprint": normalized[
            "element_configuration_fingerprint"
        ],
        "node_ids": nodes,
        "node_order_fingerprint": normalized["node_order_fingerprint"],
        "reference_geometry_fingerprint": normalized[
            "reference_geometry_fingerprint"
        ],
        "reference_frame_fingerprint": normalized["reference_frame_fingerprint"],
        "reference_corner_directors_fingerprint": normalized[
            "reference_corner_directors_fingerprint"
        ],
        "director_polarity": normalized["director_polarity"],
        "material_fingerprint": normalized["material_fingerprint"],
        "initial_fields_fingerprint": normalized["initial_fields_fingerprint"],
        "state_mode": normalized["state_mode"],
        "thickness_quadrature_id": normalized["thickness_quadrature_id"],
        "thickness": normalized["thickness"],
        "num_layers": normalized["num_layers"],
        "material_symmetry": normalized["material_symmetry"],
        "equivalent_stress_measure": normalized["equivalent_stress_measure"],
    }
    expected = _validate_identity(expected_identity)
    if state_identity != expected:
        mismatched = sorted(
            key for key in _IDENTITY_KEYS if state_identity[key] != expected[key]
        )
        raise S3CommittedStateError(
            "state identity does not match current model: " + ", ".join(mismatched)
        )

    if expected_committed_total_u is not None:
        expected_u = _finite_array(
            expected_committed_total_u, (18,), "expected_committed_total_u"
        )
        if not np.array_equal(normalized["committed_total_u"], expected_u):
            raise S3CommittedStateError(
                "committed_total_u does not match the current global displacement slice"
            )

    actual_integrity = _state_integrity_sha256(normalized)
    if actual_integrity != normalized["state_integrity_sha256"]:
        raise S3CommittedStateError("committed state integrity fingerprint mismatch")

    return normalized


def _publish_s3_state_public_signatures() -> None:
    """Keep private observation hooks outside the stable public API."""

    from inspect import signature

    public = signature(resolved_material_descriptor)
    resolved_material_descriptor.__signature__ = public.replace(
        parameters=tuple(
            parameter
            for parameter in public.parameters.values()
            if parameter.name != "_post_observation"
        )
    )


_publish_s3_state_public_signatures()
del _publish_s3_state_public_signatures


__all__ = [
    "BUBBLE_CONVENTION",
    "BUBBLE_CONDITION_LIMIT",
    "BUBBLE_FORCE_CONDENSATION_ID",
    "BUBBLE_LINE_SEARCH_MIN_FACTOR",
    "BUBBLE_LINE_SEARCH_REDUCTION",
    "BUBBLE_MAX_ITERATIONS",
    "BUBBLE_OFFSET_D",
    "BUBBLE_OFFSET_EXACT",
    "BUBBLE_POLYNOMIAL_SCALE",
    "BUBBLE_PREDICTOR_COMMIT_POLICY_ID",
    "BUBBLE_RELATIVE_TOLERANCE",
    "BUBBLE_STATE_ROLE",
    "BUBBLE_STEP_TOLERANCE",
    "CANONICALIZATION_ID",
    "COMMIT_STATUS",
    "DIRECTOR_GAUGE_FALLBACK_AXIS",
    "DIRECTOR_GAUGE_ID",
    "DIRECTOR_INCREMENT_MAP_ID",
    "DIRECTOR_POLARITY_POLICY_ID",
    "DIRECTOR_REVERSAL_TRANSFORM_ID",
    "DIRECTOR_GAUGE_PRIMARY_AXIS",
    "DIRECTOR_GAUGE_SWITCH_TOLERANCE",
    "DIRECTOR_ORTHONORMALITY_TOLERANCE",
    "DRILL_SCALE_INVERSE_METRIC_SQRT",
    "DRILL_SCALE_METRIC",
    "DRILL_SCALE_POLICY_ID",
    "DRILL_SCALE_PROJECTOR",
    "ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA",
    "EXTERNAL_COORDINATE_LAYOUT_ID",
    "EXTERNAL_ROTATION_MAP_ID",
    "FORMULATION_ID",
    "FORMULATION_SCHEMA",
    "GENERALIZED_ELEMENT_CONFIGURATION_DESCRIPTOR_SCHEMA",
    "GENERALIZED_INITIAL_FIELD_POLICY_ID",
    "GENERALIZED_COMPONENTS",
    "GENERALIZED_COMPONENT_ORDER",
    "GENERALIZED_NONLINEAR_STATE_LAYOUT_ID",
    "GENERALIZED_NONLINEAR_STATE_SCHEMA",
    "GENERALIZED_NONLINEAR_STATE_VERSION",
    "GENERALIZED_RESULTANT_COMPONENT_ORDER",
    "GENERALIZED_SECTION_DESCRIPTOR_SCHEMA",
    "GENERALIZED_SECTION_HISTORY_POLICY_ID",
    "GENERALIZED_SECTION_INTEGRATION_ID",
    "GENERALIZED_STATE_FIELD_MANIFEST",
    "GENERALIZED_STATE_FIELD_MANIFEST_ID",
    "GENERALIZED_STATE_MODE",
    "GENERALIZED_STRAIN_COMPONENT_ORDER",
    "INITIAL_FIELD_NAMES",
    "ISOTROPIC_CONSTITUTIVE_INTEGRATION_ID",
    "LAYER_COMPONENT_ORDER",
    "LAYER_STRAIN_COMPONENT_ORDER",
    "LAYER_STRESS_COMPONENT_ORDER",
    "LOBATTO_NORMALIZED_TABLES",
    "MATERIAL_DESCRIPTOR_SCHEMA",
    "MATERIAL_DESCRIPTOR_VALIDATION_ID",
    "MINIMUM_OWNER_NORMAL_ALIGNMENT",
    "MITC3_PLUS_EQUATION_MAP",
    "MITC3_PLUS_NONLINEAR_EQUATION_MAP",
    "MITC3_PLUS_NONLINEAR_SOURCE_BYTES",
    "MITC3_PLUS_NONLINEAR_SOURCE_SHA256",
    "MITC3_PLUS_NONLINEAR_SOURCE_URL",
    "MITC3_PLUS_SOURCE_BYTES",
    "MITC3_PLUS_SOURCE_SHA256",
    "MITC3_PLUS_SOURCE_URL",
    "NONLINEAR_KINEMATICS_ID",
    "NODAL_ROTATION_UPDATE_POLICY_ID",
    "NONLINEAR_POLICY_ID",
    "NONLINEAR_PL_ENERGY_POLICY_ID",
    "NONLINEAR_SOURCE_SHA256",
    "NONLINEAR_STATE_LAYOUT_ID",
    "NONLINEAR_STATE_SCHEMA",
    "NONLINEAR_STATE_VERSION",
    "NUM_INTEGRATION_STATIONS",
    "ORTHOTROPIC_CONSTITUTIVE_INTEGRATION_ID",
    "PL_BASIS_ID",
    "PL_BLOCK_SIGN_ID",
    "PL_CONDENSATION_ID",
    "PL_CONSTRAINT_ID",
    "PL_GRAM_NUMERATOR",
    "PL_GRAM_SCALE_ID",
    "PL_MINIMUM_TWIST_DENOMINATOR",
    "PL_MINIMUM_TWIST_DENOMINATOR_ID",
    "PL_PHASE_MARGIN",
    "PL_PHASE_MARGIN_ID",
    "PL_PHASE_POLICY_ID",
    "PL_TWIST_POLICY_ID",
    "QUADRATURE_ID",
    "RECOVERY_POLICY_ID",
    "REFERENCE_GEOMETRY_VALIDATION_ID",
    "REFERENCE_SURFACE_MASS_SHIFT_ID",
    "REFERENCE_SURFACE_OFFSET_POLICY_ID",
    "REFERENCE_SURFACE_STRAIN_TRANSFORM_ID",
    "S3CommittedStateError",
    "STATE_ARRAY_LAYOUT_ID",
    "STATE_FIELD_MANIFEST",
    "STATE_INTEGRITY_ID",
    "STATE_MODE",
    "STATE_REDUNDANCY_VALIDATION_ID",
    "STIFFNESS_STATION_TABLE",
    "SUPPORTED_LOBATTO_LAYER_COUNTS",
    "SURFACE_ROTATION_POLICY_ID",
    "THICKNESS_COORDINATE_SIGN_ID",
    "THICKNESS_QUADRATURE_ID",
    "TYING_POINTS",
    "TYING_POINT_DEFINITIONS",
    "build_element_configuration_descriptor",
    "build_generalized_element_configuration_descriptor",
    "build_generalized_state_identity",
    "build_state_identity",
    "canonical_json_bytes",
    "canonical_plain_data",
    "canonical_sha256",
    "element_configuration_fingerprint",
    "formulation_fingerprint",
    "formulation_fingerprint_payload",
    "formulation_mechanics_contract_payload",
    "formulation_mechanics_fingerprint",
    "generalized_element_configuration_fingerprint",
    "generalized_formulation_fingerprint",
    "generalized_formulation_fingerprint_payload",
    "generalized_section_descriptor",
    "generalized_section_fingerprint",
    "generalized_state_field_manifest_fingerprint",
    "initialize_zero_committed_s3_generalized_state",
    "initialize_zero_committed_s3_state",
    "lobatto_table_fingerprint",
    "material_fingerprint",
    "node_order_fingerprint",
    "owned_plain_snapshot",
    "reference_frame_fingerprint",
    "reference_corner_directors_fingerprint",
    "reference_geometry_fingerprint",
    "reconstruct_director_triad",
    "qualified_s3_lobatto_layers",
    "qualified_s3_triangle_frame",
    "require_qualified_s3_quality",
    "require_stateless_generalized_section",
    "resolved_material_descriptor",
    "seal_committed_s3_state",
    "state_field_manifest_fingerprint",
    "stiffness_station_table_fingerprint",
    "strict_canonical_json_loads",
    "validate_committed_s3_generalized_state",
    "validate_committed_s3_state",
]
