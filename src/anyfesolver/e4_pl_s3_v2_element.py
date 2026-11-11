"""Strict flat-linear E4-PL S3 V2 production candidate.

This module deliberately implements only the bounded flat-linear surface
recorded in ``docs/reference_cases/e4_pl_s3_v2_dkmt_equation_map.md``: a flat,
small-strain, homogeneous isotropic elastic triangle with CST membrane,
published DKMT bending/shear, exact three-point Hammer integration and the
barycentric PL drill completion.  Its only mixed-model admission is an exact,
globally coplanar qualified-Q4/V2A mesh with one positively aligned physical
director.  Every inherited shell capability outside that surface fails closed
rather than falling back to legacy TRI3 mechanics.
"""

from __future__ import annotations

import math
import threading
import weakref
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .element_capabilities import ElementCapabilityError
from .elements import ShellElement
from .e4_pl_element import (
    FORMULATION_ID as _QUALIFIED_Q4_FORMULATION_ID,
    QualifiedE4PLShellElement,
)
from .fe_core import FEMesh, Material, Node
from .e4_pl_s3_state import require_exact_numpy_runtime_authority


SELECTOR = "e4-pl-s3-v2"
FORMULATION_ID = "CANDIDATE_E4_PL_S3_V2A_FLAT_LINEAR_V1"
IMPLEMENTATION_ID = "E4_PL_S3_V2_DKMT_EQ12_41_CST_PL_HAMMER3_V1"
FORMULATION_SCHEMA = "anysolver.e4-pl-s3-v2-strict-flat-linear-element-v1"
SOURCE_CONTRACT_SCHEMA = "anysolver.e4-pl-s3-v2-source-equation-contract-v1"
SOURCE_CONTRACT_SHA256 = (
    "754A31C2B03FA3785274F30BF4F2A2FC8C66DF66A76C5D39CD9736E81679513A"
)
EQUATION_MAP_SHA256 = (
    "B527729C2F3AF482722ECB2D4635FB0FB165FB35F2EE952833D06740A68E0C4A"
)
PRIMARY_SOURCE_SHA256 = (
    "CAACAB9E0DF9B4F8B55887E84FAFA923899677F7D0F285EB2EEB46FBDFF8D17A"
)
QUADRATURE_AUTHORITY_ID = "S3_V2_DKMT_HAMMER3_DEGREE2_EXACT_V1"
PL_COMPLETION_POLICY_ID = "S3_V2_BARYCENTRIC_EXACT_SCHUR_KD_EQUALS_A66_V1"
RESULTANT_POLICY_ID = "SHELL_VARIATIONAL_RESULTANTS_V1"
DIRECTOR_POLICY_ID = "S3_V2_FIXED_PHYSICAL_DIRECTOR_D3_BLOCK_TRANSPORT_V1"
SECTION_POLICY_ID = "S3_V2_HOMOGENEOUS_ISOTROPIC_UNCOUPLED_ZERO_OFFSET_V1"
SHEAR_CORRECTION = 5.0 / 6.0

_NORMAL_TOLERANCE = 1.0e-10
_DEGENERACY_FACTOR = 64.0 * np.finfo(np.float64).eps


class StrictFlatLinearCapabilityError(ElementCapabilityError):
    """The requested operation is outside the authorized S3 V2 surface."""


def _immutable_float64_array(value: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(contiguous.shape)


# Natural coordinates are (xi, eta), with lambda = 1 - xi - eta.  The same
# three positive physical weights A/3 are used for membrane, bending and shear.
HAMMER_POINTS = _immutable_float64_array(
    (
        (1.0 / 6.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0),
    )
)
HAMMER_REFERENCE_WEIGHTS = _immutable_float64_array(
    (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
)

SUPPORTED_OPERATIONS = frozenset(
    {
        "linear_stiffness",
        "linear_internal_force",
        "raw_variational_resultants",
        "dead_transverse_pressure",
        "flat_qualified_q4_v2a_mixed_mesh",
    }
)
BLOCKED_OPERATIONS = frozenset(
    {
        "consistent_mass",
        "geometric_stiffness",
        "nonlinear_geometry",
        "material_nonlinearity",
        "unqualified_mixed_element_mesh",
        "nonlinear_state",
        "qualified_recovery",
        "serialization",
        "restart",
        "follower_pressure",
        "distributed_couple",
        "offset_load",
        "generalized_section",
        "curved_shell",
        "nonplanar_mixed_shell_mesh",
    }
)
CAPABILITY_MATRIX = MappingProxyType(
    {
        **{name: "SUPPORTED_STRICT_FLAT_LINEAR" for name in SUPPORTED_OPERATIONS},
        **{name: "BLOCKED_OUTSIDE_STAGE1_AUTHORITY" for name in BLOCKED_OPERATIONS},
    }
)


def _real_scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"strict-flat S3 V2 {label} must be a finite real scalar")
    made = float(value)
    if not math.isfinite(made):
        raise ValueError(f"strict-flat S3 V2 {label} must be finite")
    return made


def _unit_vector(value: Any, label: str) -> np.ndarray:
    if type(value) in {list, tuple} and any(
        isinstance(component, (bool, np.bool_))
        or not isinstance(component, (int, float, np.integer, np.floating))
        for component in value
    ):
        raise ValueError(f"strict-flat S3 V2 {label} must be a finite 3-vector")
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"strict-flat S3 V2 {label} must be a finite 3-vector"
        ) from exc
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError(f"strict-flat S3 V2 {label} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"strict-flat S3 V2 {label} must be nonzero")
    return _immutable_float64_array(vector / norm)


def _block_permutation(order: Tuple[int, int, int]) -> np.ndarray:
    permutation = np.zeros((18, 18), dtype=np.float64)
    identity = np.eye(6, dtype=np.float64)
    for internal, external in enumerate(order):
        permutation[
            6 * internal : 6 * internal + 6,
            6 * external : 6 * external + 6,
        ] = identity
    return permutation


def _local_component_transform(frame: np.ndarray) -> np.ndarray:
    transform = np.zeros((18, 18), dtype=np.float64)
    local_from_global = np.asarray(frame, dtype=np.float64).T
    for node in range(3):
        base = 6 * node
        transform[base : base + 3, base : base + 3] = local_from_global
        transform[base + 3 : base + 6, base + 3 : base + 6] = (
            local_from_global
        )
    return transform


def _plate_embedding() -> np.ndarray:
    """Map local shell q to printed [w,beta_x,beta_y] node blocks."""

    embedding = np.zeros((9, 18), dtype=np.float64)
    for node in range(3):
        shell = 6 * node
        plate = 3 * node
        embedding[plate, shell + 2] = 1.0
        embedding[plate + 1, shell + 4] = 1.0
        embedding[plate + 2, shell + 3] = -1.0
    return embedding


_PLATE_EMBEDDING = _immutable_float64_array(_plate_embedding())
_LAYOUT_SENTINEL = object()
_MIXED_SCOPE_CACHE_SENTINEL = object()
_MIXED_SCOPE_CACHE_NAME = "_strict_flat_v2_mixed_scope_cache_v1"
_COMPONENT_CACHE_SCHEMA = "strict-flat-s3-v2-exact-component-cache-v1"
_COMPONENT_CACHE_ENTRY_SCHEMA = (
    "strict-flat-s3-v2-exact-component-cache-entry-v1"
)
_COMPONENT_CACHE_CAPACITY = 512


def _make_exact_component_compute_wrapper() -> Any:
    """Create the sole cache surface with all mutable state closure-owned."""

    entries: Dict[Tuple[Any, ...], Tuple[Any, ...]] = {}
    missing = object()
    lock = threading.RLock()
    float64 = np.dtype(np.float64)
    float64_name = float64.str
    matrix_names = (
        "membrane",
        "bending",
        "shear",
        "physical",
        "pl",
        "total",
    )
    packed_shapes = {
        **{name: (18, 18) for name in matrix_names},
        "frame": (3, 3),
        "phi": (3,),
    }
    packed_names = (*matrix_names, "frame", "phi")
    output_names = (
        "membrane",
        "bending",
        "shear",
        "physical",
        "pl",
        "numerical",
        "hourglass",
        "total",
        "frame",
        "area",
        "phi",
        "quadrature_authority_id",
        "pl_completion_policy_id",
    )
    zero_matrix_payload = np.zeros((18, 18), dtype=np.float64).tobytes(order="C")

    def fail(label: str) -> None:
        raise StrictFlatLinearCapabilityError(
            f"strict-flat S3 V2 exact component cache {label}"
        )

    def array_identity(
        value: Any,
        shape: Tuple[int, ...],
        label: str,
    ) -> Tuple[Any, ...]:
        if (
            type(value) is not np.ndarray
            or value.dtype != float64
            or value.shape != shape
            or not np.all(np.isfinite(value))
        ):
            fail(f"{label} array is malformed")
        return (float64_name, shape, value.tobytes(order="C"))

    def scalar_identity(value: Any, label: str) -> bytes:
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            fail(f"{label} scalar is malformed")
        return np.float64(value).tobytes()

    def cache_key(
        geometry: Mapping[str, Any],
        constitutive: Mapping[str, Any],
    ) -> Tuple[Any, ...]:
        return (
            _COMPONENT_CACHE_SCHEMA,
            FORMULATION_ID,
            IMPLEMENTATION_ID,
            SOURCE_CONTRACT_SHA256,
            EQUATION_MAP_SHA256,
            QUADRATURE_AUTHORITY_ID,
            PL_COMPLETION_POLICY_ID,
            array_identity(
                geometry["local_coordinates"], (3, 2), "local-coordinate"
            ),
            array_identity(
                geometry["shape_gradients"], (3, 2), "shape-gradient"
            ),
            scalar_identity(geometry["area"], "area"),
            array_identity(geometry["frame"], (3, 3), "frame"),
            array_identity(
                geometry["local_from_external"],
                (18, 18),
                "external transform",
            ),
            array_identity(constitutive["A"], (3, 3), "membrane constitutive"),
            array_identity(constitutive["D"], (3, 3), "bending constitutive"),
            array_identity(constitutive["H"], (2, 2), "shear constitutive"),
            scalar_identity(constitutive["bending_scalar"], "bending"),
            scalar_identity(constitutive["shear_scalar"], "shear"),
            scalar_identity(constitutive["drill_scale"], "drill"),
            array_identity(HAMMER_POINTS, (3, 2), "Hammer point"),
            array_identity(HAMMER_REFERENCE_WEIGHTS, (3,), "Hammer weight"),
            array_identity(_PLATE_EMBEDDING, (9, 18), "plate embedding"),
        )

    def pack_result(result: Mapping[str, Any]) -> Tuple[Any, ...]:
        if type(result) is not dict or tuple(result) != output_names:
            fail("result schema or order differs")
        packed = tuple(
            (
                name,
                *array_identity(result[name], packed_shapes[name], name),
            )
            for name in packed_names
        )
        pl_payload = packed[packed_names.index("pl")][-1]
        numerical = array_identity(result["numerical"], (18, 18), "numerical")
        hourglass = array_identity(result["hourglass"], (18, 18), "hourglass")
        if numerical[-1] != pl_payload or hourglass[-1] != zero_matrix_payload:
            fail("numerical component identity differs")
        if (
            result["quadrature_authority_id"] != QUADRATURE_AUTHORITY_ID
            or result["pl_completion_policy_id"] != PL_COMPLETION_POLICY_ID
        ):
            fail("policy identity differs")
        return (
            _COMPONENT_CACHE_ENTRY_SCHEMA,
            packed,
            scalar_identity(result["area"], "result area"),
        )

    def unpack_result(record: Tuple[Any, ...]) -> Dict[str, Any]:
        if (
            type(record) is not tuple
            or len(record) != 3
            or record[0] != _COMPONENT_CACHE_ENTRY_SCHEMA
            or type(record[1]) is not tuple
            or len(record[1]) != len(packed_names)
            or type(record[2]) is not bytes
            or len(record[2]) != float64.itemsize
        ):
            fail("packed entry schema differs")
        arrays: Dict[str, np.ndarray] = {}
        for expected_name, expected_shape, packed in zip(
            packed_names,
            (packed_shapes[name] for name in packed_names),
            record[1],
        ):
            if (
                type(packed) is not tuple
                or len(packed) != 4
                or packed[0] != expected_name
                or packed[1] != float64_name
                or packed[2] != expected_shape
                or type(packed[3]) is not bytes
                or len(packed[3]) != math.prod(expected_shape) * float64.itemsize
            ):
                fail("packed array schema differs")
            array = np.frombuffer(packed[3], dtype=float64).reshape(expected_shape)
            if not np.all(np.isfinite(array)):
                fail("packed array is nonfinite")
            arrays[expected_name] = array.copy()
        area = float(np.frombuffer(record[2], dtype=float64, count=1)[0])
        if not math.isfinite(area) or area <= 0.0:
            fail("packed area is malformed")
        pl = arrays["pl"]
        return {
            "membrane": arrays["membrane"],
            "bending": arrays["bending"],
            "shear": arrays["shear"],
            "physical": arrays["physical"],
            "pl": pl,
            "numerical": pl.copy(),
            "hourglass": np.zeros((18, 18), dtype=np.float64),
            "total": arrays["total"],
            "frame": arrays["frame"],
            "area": area,
            "phi": arrays["phi"],
            "quadrature_authority_id": QUADRATURE_AUTHORITY_ID,
            "pl_completion_policy_id": PL_COMPLETION_POLICY_ID,
        }

    def calculate(
        self: Any,
        geometry: Mapping[str, Any],
        constitutive: Mapping[str, Any],
    ) -> Dict[str, Any]:
        operators = self._operators(geometry, constitutive)
        area = float(geometry["area"])
        physical_weight = area / 3.0
        membrane_local = (
            operators["B_m"].T
            @ constitutive["A"]
            @ operators["B_m"]
            * area
        )
        bending_local = np.zeros((18, 18), dtype=np.float64)
        shear_local = np.zeros((18, 18), dtype=np.float64)
        for bending_operator, shear_operator in zip(
            operators["B_b"],
            operators["B_s"],
        ):
            bending_local += (
                bending_operator.T
                @ constitutive["D"]
                @ bending_operator
                * physical_weight
            )
            shear_local += (
                shear_operator.T
                @ constitutive["H"]
                @ shear_operator
                * physical_weight
            )
        barycentric_mass = (area / 12.0) * np.asarray(
            ((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0)),
            dtype=np.float64,
        )
        constraint = operators["C"]
        pl_local = (
            float(constitutive["drill_scale"])
            * constraint.T
            @ barycentric_mass
            @ constraint
        )
        physical_local = membrane_local + bending_local + shear_local
        total_local = physical_local + pl_local
        transform = np.asarray(geometry["local_from_external"], dtype=np.float64)

        def globalize(matrix: np.ndarray) -> np.ndarray:
            made = transform.T @ matrix @ transform
            return 0.5 * (made + made.T)

        membrane = globalize(membrane_local)
        bending = globalize(bending_local)
        shear = globalize(shear_local)
        physical = globalize(physical_local)
        pl = globalize(pl_local)
        total = globalize(total_local)
        return {
            "membrane": membrane,
            "bending": bending,
            "shear": shear,
            "physical": physical,
            "pl": pl,
            "numerical": pl.copy(),
            "hourglass": np.zeros((18, 18), dtype=np.float64),
            "total": total,
            "frame": np.asarray(geometry["frame"], dtype=np.float64).copy(),
            "area": area,
            "phi": np.asarray(operators["phi"], dtype=np.float64).copy(),
            "quadrature_authority_id": QUADRATURE_AUTHORITY_ID,
            "pl_completion_policy_id": PL_COMPLETION_POLICY_ID,
        }

    def compute_stiffness_components(
        self: Any,
        mesh: FEMesh,
        material: Material,
    ) -> Mapping[str, Any]:
        # Admission and all live instance/model/material guards deliberately run
        # before any cache observation.  Only exact validated numeric inputs
        # enter the closure-owned key.
        geometry = self._geometry(mesh)
        constitutive = self._constitutive(material)
        key = cache_key(geometry, constitutive)
        with lock:
            cached = entries.get(key, missing)
        if cached is not missing:
            return unpack_result(cached)

        components = calculate(self, geometry, constitutive)
        packed = pack_result(components)
        with lock:
            existing = entries.get(key, missing)
            if existing is not missing:
                if existing != packed:
                    fail("concurrent result disagreement")
            else:
                entries[key] = packed
                if len(entries) > _COMPONENT_CACHE_CAPACITY:
                    # Retain the lexicographically smallest exact content keys;
                    # the final cache set is independent of insertion/thread order.
                    del entries[max(entries)]
        return components

    compute_stiffness_components.__qualname__ = (
        "StrictFlatLinearE4PLS3V2ShellElement.compute_stiffness_components"
    )
    return compute_stiffness_components


def _validate_module_authority(
    _expected_hammer_points: np.ndarray = HAMMER_POINTS,
    _expected_hammer_weights: np.ndarray = HAMMER_REFERENCE_WEIGHTS,
    _expected_plate_embedding: np.ndarray = _PLATE_EMBEDDING,
    _expected_supported: frozenset[str] = SUPPORTED_OPERATIONS,
    _expected_blocked: frozenset[str] = BLOCKED_OPERATIONS,
    _expected_capabilities: Mapping[str, str] = CAPABILITY_MATRIX,
    _expected_shear_correction: float = SHEAR_CORRECTION,
    _expected_normal_tolerance: float = _NORMAL_TOLERANCE,
    _expected_degeneracy_factor: float = _DEGENERACY_FACTOR,
    _expected_selector: str = SELECTOR,
    _expected_formulation_id: str = FORMULATION_ID,
    _expected_implementation_id: str = IMPLEMENTATION_ID,
    _expected_formulation_schema: str = FORMULATION_SCHEMA,
    _expected_source_schema: str = SOURCE_CONTRACT_SCHEMA,
    _expected_source_hash: str = SOURCE_CONTRACT_SHA256,
    _expected_equation_hash: str = EQUATION_MAP_SHA256,
    _expected_primary_hash: str = PRIMARY_SOURCE_SHA256,
    _expected_quadrature_id: str = QUADRATURE_AUTHORITY_ID,
    _expected_pl_id: str = PL_COMPLETION_POLICY_ID,
    _expected_resultant_id: str = RESULTANT_POLICY_ID,
    _expected_director_id: str = DIRECTOR_POLICY_ID,
    _expected_section_id: str = SECTION_POLICY_ID,
    _expected_numpy: Any = np,
    _expected_math: Any = math,
    _expected_threading: Any = threading,
    _expected_shell_class: Any = ShellElement,
    _expected_mesh_class: Any = FEMesh,
    _expected_material_class: Any = Material,
    _expected_capability_error: Any = ElementCapabilityError,
    _expected_candidate_error: Any = StrictFlatLinearCapabilityError,
    _expected_real_scalar: Any = _real_scalar,
    _expected_immutable_array: Any = _immutable_float64_array,
    _expected_unit_vector: Any = _unit_vector,
    _expected_block_permutation: Any = _block_permutation,
    _expected_component_transform: Any = _local_component_transform,
    _expected_numpy_guard: Any = require_exact_numpy_runtime_authority,
    _expected_layout_sentinel: Any = _LAYOUT_SENTINEL,
    _expected_mixed_scope_cache_sentinel: Any = _MIXED_SCOPE_CACHE_SENTINEL,
    _expected_mixed_scope_cache_name: str = _MIXED_SCOPE_CACHE_NAME,
    _expected_component_cache_schema: str = _COMPONENT_CACHE_SCHEMA,
    _expected_component_cache_entry_schema: str = _COMPONENT_CACHE_ENTRY_SCHEMA,
    _expected_component_cache_capacity: int = _COMPONENT_CACHE_CAPACITY,
    _expected_qualified_q4_formulation_id: str = _QUALIFIED_Q4_FORMULATION_ID,
    _expected_qualified_q4_class: Any = QualifiedE4PLShellElement,
    _expected_node_class: Any = Node,
) -> None:
    """Fail closed if a frozen Stage-1 module binding was rebound at runtime."""

    identity_bindings = (
        (HAMMER_POINTS, _expected_hammer_points),
        (HAMMER_REFERENCE_WEIGHTS, _expected_hammer_weights),
        (_PLATE_EMBEDDING, _expected_plate_embedding),
        (SUPPORTED_OPERATIONS, _expected_supported),
        (BLOCKED_OPERATIONS, _expected_blocked),
        (CAPABILITY_MATRIX, _expected_capabilities),
        (np, _expected_numpy),
        (math, _expected_math),
        (threading, _expected_threading),
        (ShellElement, _expected_shell_class),
        (QualifiedE4PLShellElement, _expected_qualified_q4_class),
        (FEMesh, _expected_mesh_class),
        (Material, _expected_material_class),
        (Node, _expected_node_class),
        (ElementCapabilityError, _expected_capability_error),
        (StrictFlatLinearCapabilityError, _expected_candidate_error),
        (_real_scalar, _expected_real_scalar),
        (_immutable_float64_array, _expected_immutable_array),
        (_unit_vector, _expected_unit_vector),
        (_block_permutation, _expected_block_permutation),
        (_local_component_transform, _expected_component_transform),
        (require_exact_numpy_runtime_authority, _expected_numpy_guard),
        (_LAYOUT_SENTINEL, _expected_layout_sentinel),
        (_MIXED_SCOPE_CACHE_SENTINEL, _expected_mixed_scope_cache_sentinel),
    )
    scalar_bindings = (
        (SHEAR_CORRECTION, _expected_shear_correction),
        (_NORMAL_TOLERANCE, _expected_normal_tolerance),
        (_DEGENERACY_FACTOR, _expected_degeneracy_factor),
        (SELECTOR, _expected_selector),
        (FORMULATION_ID, _expected_formulation_id),
        (IMPLEMENTATION_ID, _expected_implementation_id),
        (FORMULATION_SCHEMA, _expected_formulation_schema),
        (SOURCE_CONTRACT_SCHEMA, _expected_source_schema),
        (SOURCE_CONTRACT_SHA256, _expected_source_hash),
        (EQUATION_MAP_SHA256, _expected_equation_hash),
        (PRIMARY_SOURCE_SHA256, _expected_primary_hash),
        (QUADRATURE_AUTHORITY_ID, _expected_quadrature_id),
        (PL_COMPLETION_POLICY_ID, _expected_pl_id),
        (RESULTANT_POLICY_ID, _expected_resultant_id),
        (DIRECTOR_POLICY_ID, _expected_director_id),
        (SECTION_POLICY_ID, _expected_section_id),
        (_MIXED_SCOPE_CACHE_NAME, _expected_mixed_scope_cache_name),
        (_COMPONENT_CACHE_SCHEMA, _expected_component_cache_schema),
        (_COMPONENT_CACHE_ENTRY_SCHEMA, _expected_component_cache_entry_schema),
        (_COMPONENT_CACHE_CAPACITY, _expected_component_cache_capacity),
        (
            _QUALIFIED_Q4_FORMULATION_ID,
            _expected_qualified_q4_formulation_id,
        ),
    )
    if any(actual is not expected for actual, expected in identity_bindings) or any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in scalar_bindings
    ):
        raise StrictFlatLinearCapabilityError(
            "strict-flat S3 V2 frozen module authority was modified"
        )
    _expected_numpy_guard(context="strict-flat S3 V2")


_FROZEN_INSTANCE_STATE = frozenset(
    {
        "_internal_forces",
        "_is_3node",
        "_is_4node",
        "_is_6node",
        "_is_8node",
        "_is_quadrilateral",
        "_is_triangular",
        "_mass_matrix",
        "_stiffness_matrix",
        "_strict_flat_v2_frozen",
        "drilling_stabilization",
        "element_id",
        "hourglass_stabilization",
        "material_angle_deg",
        "material_direction",
        "material_name",
        "node_ids",
        "reduced_integration",
        "reference_normal",
        "shell_section",
        "thickness",
    }
)
_MESH_AUTHORITY_STATE = frozenset(
    {"_qualified_direct_state_token", "_qualified_direct_state_tokens"}
)


def _make_instance_authority_registry() -> Tuple[Any, Any]:
    registry: "weakref.WeakKeyDictionary[Any, Tuple[Any, ...]]" = (
        weakref.WeakKeyDictionary()
    )

    def register(element: Any, signature: Tuple[Any, ...]) -> None:
        registry[element] = signature

    def require(element: Any, signature: Tuple[Any, ...]) -> None:
        expected = registry.get(element)
        if expected is None or expected != signature:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 construction authority changed"
            )

    return register, require


(
    _register_instance_authority,
    _require_instance_authority,
) = _make_instance_authority_registry()


def _make_final_class_authority() -> Tuple[Any, Any]:
    authority: Optional[Dict[str, Any]] = None

    def initialize(cls: type) -> None:
        nonlocal authority
        if authority is not None:
            raise RuntimeError("strict-flat S3 V2 class authority already initialized")
        authority = dict(type.__getattribute__(cls, "__dict__"))

    def require(cls: type) -> None:
        if authority is None:
            raise RuntimeError("strict-flat S3 V2 class authority is not initialized")
        namespace = type.__getattribute__(cls, "__dict__")
        if set(namespace) != set(authority) or any(
            namespace[name] is not expected for name, expected in authority.items()
        ):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 candidate class namespace changed"
            )

    return initialize, require


(
    _initialize_final_class_authority,
    _require_final_class_authority,
) = _make_final_class_authority()


class _StrictFlatCandidateMeta(type(ShellElement)):
    """Prevent normal runtime mutation of the reviewed candidate class."""

    def __setattr__(cls, name: str, value: Any) -> None:
        namespace = type.__getattribute__(cls, "__dict__")
        if namespace.get("_strict_flat_v2_class_frozen") is True:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 candidate class authority is frozen"
            )
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        namespace = type.__getattribute__(cls, "__dict__")
        if namespace.get("_strict_flat_v2_class_frozen") is True:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 candidate class authority is frozen"
            )
        super().__delattr__(name)


class StrictFlatLinearE4PLS3V2ShellElement(
    ShellElement, metaclass=_StrictFlatCandidateMeta
):
    """Opt-in S3 V2 for the strict flat, linear, isotropic Stage-1 scope."""

    __slots__ = ("_strict_flat_v2_layout_sentinel",)

    formulation_id = FORMULATION_ID
    selector = SELECTOR
    supported_operations = SUPPORTED_OPERATIONS
    blocked_operations = BLOCKED_OPERATIONS
    _legacy_shell_dispatch_forbidden = FORMULATION_ID
    _strict_flat_v2_class_frozen = False
    _register_instance_authority = staticmethod(_register_instance_authority)
    _require_instance_authority = staticmethod(_require_instance_authority)
    _module_authority_guard = staticmethod(_validate_module_authority)

    # Do not inherit the legacy S3 one-point shear rule.  Both physical blocks
    # use the published three-point Hammer rule.
    TRI_GAUSS_POINTS_3 = HAMMER_POINTS
    TRI_GAUSS_WEIGHTS_3 = HAMMER_REFERENCE_WEIGHTS

    def __getattribute__(
        self,
        name: str,
        _frozen_state: frozenset[str] = _FROZEN_INSTANCE_STATE,
        _mesh_state: frozenset[str] = _MESH_AUTHORITY_STATE,
    ) -> Any:
        # The inherited Element hierarchy owns an instance ``__dict__``.  A
        # caller could otherwise shadow a frozen operator (for example
        # ``_constitutive``) without touching the reviewed class.  Reject such
        # shadows before Python dispatches them; the exact state-set check in
        # ``_validate_configuration`` catches every other injected name.
        if name not in {"__dict__", "__class__"}:
            namespace = object.__getattribute__(self, "__dict__")
            if (
                namespace.get("_strict_flat_v2_frozen") is True
                and name in namespace
                and name not in (_frozen_state | _mesh_state)
            ):
                raise StrictFlatLinearCapabilityError(
                    "strict-flat S3 V2 forbids instance operator shadowing"
                )
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        namespace = object.__getattribute__(self, "__dict__")
        if namespace.get("_strict_flat_v2_frozen") is True:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 candidate state is immutable after construction"
            )
        super().__setattr__(name, value)

    def __init__(
        self,
        element_id: int,
        node_ids: Sequence[int],
        material_name: str = "default",
        *,
        thickness: float = 0.01,
        reference_normal: Sequence[float],
    ) -> None:
        __class__._module_authority_guard()
        if type(element_id) is not int:
            raise TypeError("strict-flat S3 V2 element_id must be an exact integer")
        if type(material_name) is not str:
            raise TypeError("strict-flat S3 V2 material_name must be a string")
        owned_node_ids = tuple(node_ids)
        if len(owned_node_ids) != 3:
            raise ValueError(
                "StrictFlatLinearE4PLS3V2ShellElement requires exactly three nodes"
            )
        if not all(type(node_id) is int for node_id in owned_node_ids):
            raise TypeError(
                "strict-flat S3 V2 node_ids must contain exact non-boolean integers"
            )
        if len(set(owned_node_ids)) != 3:
            raise ValueError("strict-flat S3 V2 node_ids must be distinct")
        thickness_value = _real_scalar(thickness, "thickness")
        if thickness_value <= 0.0:
            raise ValueError("strict-flat S3 V2 thickness must be strictly positive")
        normal = _unit_vector(reference_normal, "reference_normal")

        super().__init__(
            element_id,
            owned_node_ids,
            material_name,
            thickness=thickness_value,
            drilling_stabilization=0.0,
            reduced_integration=False,
            hourglass_stabilization=0.0,
            material_direction=None,
            material_angle_deg=0.0,
            shell_section=None,
        )
        self.reference_normal = normal
        object.__setattr__(self, "_strict_flat_v2_layout_sentinel", _LAYOUT_SENTINEL)
        __class__._register_instance_authority(
            self,
            (
                element_id,
                owned_node_ids,
                material_name,
                thickness_value,
                normal.tobytes(order="C"),
            ),
        )
        object.__setattr__(self, "_strict_flat_v2_frozen", True)

    @property
    def gauss_points(self) -> np.ndarray:
        self._validate_configuration()
        return HAMMER_POINTS

    @property
    def gauss_weights(self) -> np.ndarray:
        self._validate_configuration()
        return HAMMER_REFERENCE_WEIGHTS

    @property
    def shear_gauss_points(self) -> np.ndarray:
        self._validate_configuration()
        return HAMMER_POINTS

    @property
    def shear_gauss_weights(self) -> np.ndarray:
        self._validate_configuration()
        return HAMMER_REFERENCE_WEIGHTS

    @property
    def physical_reference_director(self) -> np.ndarray:
        self._validate_configuration()
        return np.asarray(self.reference_normal, dtype=np.float64).copy()

    def capability_matrix(self) -> Dict[str, str]:
        self._validate_configuration()
        return dict(CAPABILITY_MATRIX)

    @staticmethod
    def _unsupported(operation: str) -> None:
        raise StrictFlatLinearCapabilityError(
            f"strict-flat S3 V2 capability {operation!r} is outside the "
            "authorized Stage-1 flat-linear surface"
        )

    def _validate_configuration(
        self,
        _module_guard: Any = _validate_module_authority,
        _class_guard: Any = _require_final_class_authority,
        _frozen_state: frozenset[str] = _FROZEN_INSTANCE_STATE,
        _mesh_state: frozenset[str] = _MESH_AUTHORITY_STATE,
        _layout_sentinel: Any = _LAYOUT_SENTINEL,
    ) -> None:
        _module_guard()
        _class_guard(__class__)
        if type(self) is not __class__:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 mechanics require the exact production class"
            )
        if (
            __class__.formulation_id != FORMULATION_ID
            or __class__.selector != SELECTOR
            or __class__.supported_operations is not SUPPORTED_OPERATIONS
            or __class__.blocked_operations is not BLOCKED_OPERATIONS
            or __class__._legacy_shell_dispatch_forbidden != FORMULATION_ID
            or __class__.TRI_GAUSS_POINTS_3 is not HAMMER_POINTS
            or __class__.TRI_GAUSS_WEIGHTS_3 is not HAMMER_REFERENCE_WEIGHTS
        ):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 frozen class authority was modified"
            )
        namespace = object.__getattribute__(self, "__dict__")
        state_names = set(namespace)
        if state_names != set(_frozen_state) and state_names != set(
            _frozen_state | _mesh_state
        ):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 instance authority contains unregistered state"
            )
        if namespace.get("_strict_flat_v2_frozen") is not True:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 instance authority is not frozen"
            )
        if (
            object.__getattribute__(self, "_strict_flat_v2_layout_sentinel")
            is not _layout_sentinel
        ):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 class-layout authority changed"
            )
        if _mesh_state <= state_names:
            token = namespace["_qualified_direct_state_token"]
            tokens = namespace["_qualified_direct_state_tokens"]
            if (
                not isinstance(token, list)
                or len(token) != 1
                or type(token[0]) is not int
                or type(tokens) is not list
                or not tokens
                or not any(candidate is token for candidate in tokens)
                or any(
                    not isinstance(candidate, list)
                    or len(candidate) != 1
                    or type(candidate[0]) is not int
                    for candidate in tokens
                )
            ):
                raise StrictFlatLinearCapabilityError(
                    "strict-flat S3 V2 mesh authority token is malformed"
                )
        if type(self.element_id) is not int:
            raise ValueError("strict-flat S3 V2 element_id authority changed")
        if type(self.material_name) is not str:
            raise ValueError("strict-flat S3 V2 material_name authority changed")
        if (
            type(self.node_ids) is not tuple
            or len(self.node_ids) != 3
            or not all(type(node_id) is int for node_id in self.node_ids)
            or len(set(self.node_ids)) != 3
        ):
            raise ValueError("strict-flat S3 V2 connectivity authority changed")
        if (
            type(self.thickness) is not float
            or not math.isfinite(self.thickness)
            or self.thickness <= 0.0
        ):
            raise ValueError("strict-flat S3 V2 thickness authority changed")
        if (
            self.drilling_stabilization != 0.0
            or self.hourglass_stabilization != 0.0
            or type(self.reduced_integration) is not bool
            or self.reduced_integration
            or self.material_direction is not None
            or self.material_angle_deg != 0.0
            or self.shell_section is not None
        ):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 forbids legacy integration, stabilization, "
                "orientation and generalized-section controls"
            )
        if (
            self._stiffness_matrix is not None
            or self._mass_matrix is not None
            or self._internal_forces is not None
            or self._is_3node is not True
            or self._is_triangular is not True
            or self._is_4node is not False
            or self._is_6node is not False
            or self._is_8node is not False
            or self._is_quadrilateral is not False
        ):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 inherited runtime state changed"
            )
        __class__._require_instance_authority(
            self,
            (
                self.element_id,
                self.node_ids,
                self.material_name,
                self.thickness,
                self.reference_normal.tobytes(order="C"),
            ),
        )
        normal = self.reference_normal
        if (
            type(normal) is not np.ndarray
            or normal.shape != (3,)
            or normal.dtype != np.dtype(np.float64)
            or normal.flags.writeable
            or not np.all(np.isfinite(normal))
            or not math.isclose(
                float(np.linalg.norm(normal)),
                1.0,
                rel_tol=0.0,
                abs_tol=8.0 * np.finfo(np.float64).eps,
            )
        ):
            raise ValueError("strict-flat S3 V2 reference_normal authority changed")

    def _coordinates(self, mesh: FEMesh) -> np.ndarray:
        if type(mesh) is not FEMesh:
            raise TypeError("strict-flat S3 V2 requires an exact FEMesh")
        coordinates = np.empty((3, 3), dtype=np.float64)
        for index, node_id in enumerate(self.node_ids):
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(
                    f"strict-flat S3 V2 references missing node {node_id}"
                )
            value = np.asarray(node.coords(), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"strict-flat S3 V2 node {node_id} has invalid coordinates"
                )
            coordinates[index] = value
        return coordinates

    def _validate_model_scope(self, mesh: FEMesh) -> None:
        """Admit only one exact, coplanar qualified-Q4/V2A model boundary.

        The complete registry is checked once per mesh mutation epoch.  A
        successful check is cached with the mesh's solver-owned direct-state
        token, mapping identities and topology/geometry revisions.  This keeps
        formal mixed-mesh assembly linear in the element count while any node,
        element or registry mutation forces a complete revalidation.
        """

        if type(mesh) is not FEMesh:
            raise TypeError("strict-flat S3 V2 requires an exact FEMesh")

        mesh_namespace = object.__getattribute__(mesh, "__dict__")
        elements = dict.get(mesh_namespace, "elements")
        nodes = dict.get(mesh_namespace, "nodes")
        token = dict.get(mesh_namespace, "_qualified_direct_state_token")
        revisions = dict.get(mesh_namespace, "revisions")
        if (
            not isinstance(elements, dict)
            or not isinstance(nodes, dict)
            or not isinstance(token, list)
            or len(token) != 1
            or type(token[0]) is not int
            or type(revisions) is not dict
            or type(revisions.get("topology")) is not int
            or type(revisions.get("geometry")) is not int
        ):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 mesh registry authority is malformed"
            )
        if not elements:
            # Standalone element evaluation remains part of the local
            # formulation gate.  A nonempty model registry, however, must own
            # the exact element instance evaluated below.
            return
        if elements.get(self.element_id) is not self:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 element registry identity must register the "
                "evaluated V2A instance exactly"
            )

        q4_class_namespace = type.__getattribute__(
            QualifiedE4PLShellElement,
            "__dict__",
        )
        if (
            q4_class_namespace.get("formulation_id")
            != _QUALIFIED_Q4_FORMULATION_ID
        ):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 qualified Q4 formulation identity changed"
            )

        common_normal = np.asarray(self.reference_normal, dtype=np.float64)
        cache = dict.get(mesh_namespace, _MIXED_SCOPE_CACHE_NAME)
        if (
            type(cache) is tuple
            and len(cache) == 8
            and cache[0] is _MIXED_SCOPE_CACHE_SENTINEL
            and cache[1] is token
            and cache[2] == token[0]
            and cache[3] is nodes
            and cache[4] is elements
            and cache[5] == revisions["topology"]
            and cache[6] == revisions["geometry"]
            and cache[7] == common_normal.tobytes(order="C")
        ):
            return

        try:
            registered_items = list(elements.items())
            registered_nodes = list(nodes.items())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 mesh registry authority is malformed"
            ) from exc

        allowed_types = (__class__, QualifiedE4PLShellElement)
        registered_elements: list[Any] = []
        for key, element in registered_items:
            if type(key) is not int or type(element) not in allowed_types:
                raise StrictFlatLinearCapabilityError(
                    "strict-flat S3 V2 mixed element registry permits only exact "
                    "qualified Q4 and V2A shell elements"
                )
            namespace = object.__getattribute__(element, "__dict__")
            element_id = dict.get(namespace, "element_id")
            if type(element_id) is not int or key != element_id:
                raise StrictFlatLinearCapabilityError(
                    "strict-flat S3 V2 element registry identity is malformed"
                )
            registered_elements.append(element)
        if (
            len({id(element) for element in registered_elements})
            != len(registered_elements)
            or len(
                {
                    dict.get(object.__getattribute__(element, "__dict__"), "element_id")
                    for element in registered_elements
                }
            )
            != len(registered_elements)
        ):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 element registry contains duplicate identities"
            )

        node_coordinates: dict[int, np.ndarray] = {}
        observed_nodes: list[Node] = []
        for key, node in registered_nodes:
            if type(key) is not int or type(node) is not Node:
                raise StrictFlatLinearCapabilityError(
                    "strict-flat S3 V2 node registry identity is malformed"
                )
            namespace = object.__getattribute__(node, "__dict__")
            if type(dict.get(namespace, "id")) is not int or namespace["id"] != key:
                raise StrictFlatLinearCapabilityError(
                    "strict-flat S3 V2 node registry identity is malformed"
                )
            try:
                coordinate = np.asarray(
                    (
                        _real_scalar(dict.get(namespace, "x"), f"node {key} x"),
                        _real_scalar(dict.get(namespace, "y"), f"node {key} y"),
                        _real_scalar(dict.get(namespace, "z"), f"node {key} z"),
                    ),
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as exc:
                raise StrictFlatLinearCapabilityError(
                    f"strict-flat S3 V2 node {key} coordinates are invalid"
                ) from exc
            node_coordinates[key] = coordinate
            observed_nodes.append(node)
        if len({id(node) for node in observed_nodes}) != len(observed_nodes):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 node registry contains duplicate identities"
            )

        for element in registered_elements:
            if type(element) is __class__:
                element._validate_configuration()
                director = np.asarray(element.reference_normal, dtype=np.float64)
                element_coordinates = element._coordinates(mesh)
            else:
                namespace = object.__getattribute__(element, "__dict__")
                if "formulation_id" in namespace:
                    raise StrictFlatLinearCapabilityError(
                        "strict-flat S3 V2 qualified Q4 formulation identity is "
                        "shadowed"
                    )
                # This accessor runs the independently frozen Q4 runtime and
                # instance guards before returning any coordinate observation.
                element_coordinates = np.asarray(
                    element.get_node_coordinates(mesh),
                    dtype=np.float64,
                )
                reference_normal = dict.get(namespace, "reference_normal")
                polarity = dict.get(namespace, "director_polarity")
                if reference_normal is None:
                    raise StrictFlatLinearCapabilityError(
                        "strict-flat S3 V2 mixed qualified Q4 requires an "
                        "authoritative reference_normal"
                    )
                if type(polarity) is not int or polarity not in {-1, 1}:
                    raise StrictFlatLinearCapabilityError(
                        "strict-flat S3 V2 qualified Q4 physical-director "
                        "authority is malformed"
                    )
                director = float(polarity) * np.asarray(
                    reference_normal,
                    dtype=np.float64,
                )
            if (
                director.shape != (3,)
                or not np.all(np.isfinite(director))
                or not math.isclose(
                    float(np.linalg.norm(director)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=8.0 * np.finfo(np.float64).eps,
                )
                or float(director @ common_normal)
                < 1.0 - _NORMAL_TOLERANCE
            ):
                raise StrictFlatLinearCapabilityError(
                    "strict-flat S3 V2 mixed shell physical directors are not "
                    "positively aligned"
                )
            if (
                element_coordinates.shape != (len(element.node_ids), 3)
                or not np.all(np.isfinite(element_coordinates))
                or any(node_id not in node_coordinates for node_id in element.node_ids)
            ):
                raise StrictFlatLinearCapabilityError(
                    "strict-flat S3 V2 mixed shell connectivity is malformed"
                )

        if not node_coordinates:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 nonempty element registry has no nodes"
            )
        all_coordinates = np.asarray(
            [node_coordinates[key] for key in sorted(node_coordinates)],
            dtype=np.float64,
        )
        common_origin = all_coordinates[0]
        relative = all_coordinates - common_origin
        scale = max(1.0, float(np.max(np.linalg.norm(relative, axis=1))))
        offsets = relative @ common_normal
        if float(np.max(np.abs(offsets))) > _NORMAL_TOLERANCE * scale:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 shell nodes are not globally coplanar"
            )

        object.__setattr__(
            mesh,
            _MIXED_SCOPE_CACHE_NAME,
            (
                _MIXED_SCOPE_CACHE_SENTINEL,
                token,
                token[0],
                nodes,
                elements,
                revisions["topology"],
                revisions["geometry"],
                common_normal.tobytes(order="C"),
            ),
        )

    def _geometry(self, mesh: FEMesh) -> Dict[str, Any]:
        self._validate_configuration()
        self._validate_model_scope(mesh)
        external = self._coordinates(mesh)
        normal = np.asarray(self.reference_normal, dtype=np.float64)
        edge_12 = external[1] - external[0]
        edge_13 = external[2] - external[0]
        edge_scale = max(
            float(edge_12 @ edge_12),
            float(edge_13 @ edge_13),
            float((external[2] - external[1]) @ (external[2] - external[1])),
            np.finfo(np.float64).tiny,
        )
        cross = np.cross(edge_12, edge_13)
        doubled_area = float(np.linalg.norm(cross))
        if (
            not math.isfinite(doubled_area)
            or doubled_area <= _DEGENERACY_FACTOR * edge_scale
        ):
            raise ValueError(
                "strict-flat S3 V2 requires a numerically nondegenerate triangle"
            )
        tangency = max(abs(float(edge_12 @ normal)), abs(float(edge_13 @ normal)))
        if tangency > _NORMAL_TOLERANCE * math.sqrt(edge_scale):
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 reference_normal is not normal to the flat facet"
            )
        signed = float(cross @ normal)
        if abs(signed) < (1.0 - _NORMAL_TOLERANCE) * doubled_area:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 reference_normal is incompatible with the facet"
            )
        order: Tuple[int, int, int] = (0, 1, 2) if signed > 0.0 else (0, 2, 1)
        coordinates = external[np.asarray(order, dtype=np.intp)]

        x_axis = coordinates[1] - coordinates[0]
        x_axis /= float(np.linalg.norm(x_axis))
        y_axis = np.cross(normal, x_axis)
        y_axis /= float(np.linalg.norm(y_axis))
        frame = np.column_stack((x_axis, y_axis, normal))
        relative = coordinates - coordinates[0]
        local = np.column_stack((relative @ x_axis, relative @ y_axis))
        jacobian = np.array(
            (
                (local[1, 0] - local[0, 0], local[2, 0] - local[0, 0]),
                (local[1, 1] - local[0, 1], local[2, 1] - local[0, 1]),
            ),
            dtype=np.float64,
        )
        determinant = float(np.linalg.det(jacobian))
        if not math.isfinite(determinant) or determinant <= 0.0:
            raise RuntimeError(
                "strict-flat S3 V2 failed to construct a positive element chart"
            )
        natural_gradients = np.asarray(
            ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)),
            dtype=np.float64,
        )
        gradients = natural_gradients @ np.linalg.inv(jacobian)
        permutation = _block_permutation(order)
        local_from_external = _local_component_transform(frame) @ permutation
        return {
            "external_coordinates": external,
            "internal_order": order,
            "local_coordinates": local,
            "shape_gradients": gradients,
            "area": 0.5 * determinant,
            "frame": frame,
            "local_from_external": local_from_external,
        }

    def _constitutive(self, material: Material) -> Dict[str, Any]:
        if type(material) is not Material:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 requires the exact homogeneous isotropic Material"
            )
        if getattr(material, "hardening_curve", None) is not None:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 material nonlinearity is outside Stage-1 authority"
            )
        yield_stress = getattr(material, "yield_stress", 0.0)
        if yield_stress is not None and _real_scalar(yield_stress, "yield_stress") != 0.0:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 requires a purely elastic material descriptor"
            )
        elastic_modulus = _real_scalar(
            getattr(material, "elastic_modulus", None),
            "elastic_modulus",
        )
        poisson_ratio = _real_scalar(
            getattr(material, "poisson_ratio", None),
            "poisson_ratio",
        )
        if elastic_modulus <= 0.0:
            raise ValueError("strict-flat S3 V2 elastic_modulus must be positive")
        if not -1.0 < poisson_ratio < 0.5:
            raise ValueError(
                "strict-flat S3 V2 poisson_ratio must satisfy -1 < nu < 0.5"
            )
        shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
        membrane_scale = (
            elastic_modulus * self.thickness / (1.0 - poisson_ratio**2)
        )
        bending_scale = (
            elastic_modulus
            * self.thickness**3
            / (12.0 * (1.0 - poisson_ratio**2))
        )
        isotropic = np.asarray(
            (
                (1.0, poisson_ratio, 0.0),
                (poisson_ratio, 1.0, 0.0),
                (0.0, 0.0, 0.5 * (1.0 - poisson_ratio)),
            ),
            dtype=np.float64,
        )
        membrane = membrane_scale * isotropic
        bending = bending_scale * isotropic
        shear_scalar = SHEAR_CORRECTION * shear_modulus * self.thickness
        shear = shear_scalar * np.eye(2, dtype=np.float64)
        if not all(
            np.all(np.isfinite(matrix)) for matrix in (membrane, bending, shear)
        ) or shear_scalar <= 0.0:
            raise ValueError("strict-flat S3 V2 constitutive data are invalid")
        return {
            "A": membrane,
            "D": bending,
            "H": shear,
            "bending_scalar": bending_scale,
            "shear_scalar": shear_scalar,
            "drill_scale": shear_modulus * self.thickness,
        }

    @staticmethod
    def _membrane_operator(gradients: np.ndarray) -> np.ndarray:
        operator = np.zeros((3, 18), dtype=np.float64)
        for node, (derivative_x, derivative_y) in enumerate(gradients):
            base = 6 * node
            operator[0, base] = derivative_x
            operator[1, base + 1] = derivative_y
            operator[2, base] = derivative_y
            operator[2, base + 1] = derivative_x
        return operator

    @staticmethod
    def _pl_constraint(gradients: np.ndarray) -> np.ndarray:
        constraint = np.zeros((3, 18), dtype=np.float64)
        for row in range(3):
            for node, (derivative_x, derivative_y) in enumerate(gradients):
                base = 6 * node
                constraint[row, base] = 0.5 * derivative_y
                constraint[row, base + 1] = -0.5 * derivative_x
            constraint[row, 6 * row + 5] = 1.0
        return constraint

    @staticmethod
    def _edge_data(local: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        directed = ((0, 1), (1, 2), (2, 0))
        cosine = np.empty(3, dtype=np.float64)
        sine = np.empty(3, dtype=np.float64)
        length = np.empty(3, dtype=np.float64)
        for edge, (first, second) in enumerate(directed):
            delta = local[second] - local[first]
            edge_length = float(np.linalg.norm(delta))
            if not math.isfinite(edge_length) or edge_length <= 0.0:
                raise ValueError("strict-flat S3 V2 has an invalid directed edge")
            length[edge] = edge_length
            cosine[edge] = delta[0] / edge_length
            sine[edge] = delta[1] / edge_length
        return cosine, sine, length

    @staticmethod
    def _side_kinematics(
        cosine: np.ndarray,
        sine: np.ndarray,
        length: np.ndarray,
    ) -> np.ndarray:
        operator = np.zeros((3, 9), dtype=np.float64)
        for edge, (first, second) in enumerate(((0, 1), (1, 2), (2, 0))):
            first_base = 3 * first
            second_base = 3 * second
            operator[edge, first_base] = -1.0 / length[edge]
            operator[edge, second_base] = 1.0 / length[edge]
            operator[edge, first_base + 1] = 0.5 * cosine[edge]
            operator[edge, first_base + 2] = 0.5 * sine[edge]
            operator[edge, second_base + 1] = 0.5 * cosine[edge]
            operator[edge, second_base + 2] = 0.5 * sine[edge]
        return operator

    @staticmethod
    def _side_shear_projection(
        shape: np.ndarray,
        cosine: np.ndarray,
        sine: np.ndarray,
    ) -> np.ndarray:
        c12, c23, c31 = cosine
        s12, s23, s31 = sine
        a1 = c12 * s31 - c31 * s12
        a2 = c23 * s12 - c12 * s23
        a3 = c31 * s23 - c23 * s31
        scale = max(abs(a1), abs(a2), abs(a3), 1.0)
        if min(abs(a1), abs(a2), abs(a3)) <= _DEGENERACY_FACTOR * scale:
            raise ValueError(
                "strict-flat S3 V2 side-shear projection is singular"
            )
        n1, n2, n3 = shape
        return np.asarray(
            (
                (
                    s31 * n1 / a1 - s23 * n2 / a2,
                    s12 * n2 / a2 - s31 * n3 / a3,
                    s23 * n3 / a3 - s12 * n1 / a1,
                ),
                (
                    -(c31 * n1 / a1 - c23 * n2 / a2),
                    -(c12 * n2 / a2 - c31 * n3 / a3),
                    -(c23 * n3 / a3 - c12 * n1 / a1),
                ),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _bending_beta_operator(gradients: np.ndarray) -> np.ndarray:
        operator = np.zeros((3, 9), dtype=np.float64)
        for node, (derivative_x, derivative_y) in enumerate(gradients):
            base = 3 * node
            operator[0, base + 1] = derivative_x
            operator[1, base + 2] = derivative_y
            operator[2, base + 1] = derivative_y
            operator[2, base + 2] = derivative_x
        return operator

    @staticmethod
    def _hierarchical_gradients(
        shape: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        pairs = ((0, 1), (1, 2), (2, 0))
        made = np.empty((3, 2), dtype=np.float64)
        for edge, (first, second) in enumerate(pairs):
            made[edge] = 4.0 * (
                shape[first] * gradients[second]
                + shape[second] * gradients[first]
            )
        return made

    @staticmethod
    def _bending_delta_operator(
        hierarchical_gradients: np.ndarray,
        cosine: np.ndarray,
        sine: np.ndarray,
    ) -> np.ndarray:
        operator = np.empty((3, 3), dtype=np.float64)
        for edge, (derivative_x, derivative_y) in enumerate(
            hierarchical_gradients
        ):
            operator[:, edge] = (
                derivative_x * cosine[edge],
                derivative_y * sine[edge],
                derivative_y * cosine[edge] + derivative_x * sine[edge],
            )
        return operator

    def _operators(
        self,
        geometry: Mapping[str, Any],
        constitutive: Mapping[str, Any],
    ) -> Dict[str, Any]:
        local = np.asarray(geometry["local_coordinates"], dtype=np.float64)
        gradients = np.asarray(geometry["shape_gradients"], dtype=np.float64)
        cosine, sine, length = self._edge_data(local)
        side = self._side_kinematics(cosine, sine, length)
        phi = (
            12.0
            * float(constitutive["bending_scalar"])
            / (float(constitutive["shear_scalar"]) * length**2)
        )
        if np.any(~np.isfinite(phi)) or np.any(phi <= 0.0):
            raise ValueError("strict-flat S3 V2 DKMT phi factors are invalid")
        a_delta = -(2.0 / 3.0) * np.diag(1.0 + phi)
        a_phi = -(2.0 / 3.0) * np.diag(phi)
        try:
            delta_map = np.linalg.solve(a_delta, side)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "strict-flat S3 V2 DKMT elimination is singular"
            ) from exc
        beta_operator = self._bending_beta_operator(gradients)
        membrane = self._membrane_operator(gradients)
        bending_all = []
        shear_all = []
        shapes = []
        for xi, eta in HAMMER_POINTS:
            shape = np.asarray((1.0 - xi - eta, xi, eta), dtype=np.float64)
            hierarchical = self._hierarchical_gradients(shape, gradients)
            delta_operator = self._bending_delta_operator(
                hierarchical,
                cosine,
                sine,
            )
            bending_plate = beta_operator + delta_operator @ delta_map
            shear_plate = (
                self._side_shear_projection(shape, cosine, sine)
                @ a_phi
                @ delta_map
            )
            bending_all.append(bending_plate @ _PLATE_EMBEDDING)
            shear_all.append(shear_plate @ _PLATE_EMBEDDING)
            shapes.append(shape)
        return {
            "B_m": membrane,
            "B_b": np.asarray(bending_all, dtype=np.float64),
            "B_s": np.asarray(shear_all, dtype=np.float64),
            "shape": np.asarray(shapes, dtype=np.float64),
            "C": self._pl_constraint(gradients),
            "phi": phi,
        }

    compute_stiffness_components = _make_exact_component_compute_wrapper()

    def compute_stiffness_matrix(
        self,
        mesh: FEMesh,
        material: Material,
    ) -> np.ndarray:
        return np.asarray(
            self.compute_stiffness_components(mesh, material)["total"],
            dtype=np.float64,
        ).copy()

    def compute_internal_forces(
        self,
        mesh: FEMesh,
        displacements: np.ndarray,
        material: Material,
    ) -> np.ndarray:
        vector = self._get_element_displacements(mesh, displacements)
        if vector.shape != (18,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                "strict-flat S3 V2 displacements must resolve to a finite 18-vector"
            )
        return self.compute_stiffness_matrix(mesh, material) @ vector

    def compute_variational_resultants(
        self,
        mesh: FEMesh,
        displacements: np.ndarray,
        material: Material,
    ) -> Dict[str, Any]:
        geometry = self._geometry(mesh)
        constitutive = self._constitutive(material)
        operators = self._operators(geometry, constitutive)
        vector = self._get_element_displacements(mesh, displacements)
        if vector.shape != (18,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                "strict-flat S3 V2 displacements must resolve to a finite 18-vector"
            )
        local_vector = geometry["local_from_external"] @ vector
        membrane_strain = np.broadcast_to(
            operators["B_m"] @ local_vector,
            (3, 3),
        ).copy()
        curvature = np.einsum("gij,j->gi", operators["B_b"], local_vector)
        transverse_shear = np.einsum(
            "gij,j->gi",
            operators["B_s"],
            local_vector,
        )
        membrane_resultants = membrane_strain @ constitutive["A"].T
        bending_resultants = curvature @ constitutive["D"].T
        shear_resultants = transverse_shear @ constitutive["H"].T
        # Operators are evaluated in the positive internal D3 chart.  Restore
        # station rows to the external connectivity order so the canonical
        # Hammer coordinates below always describe the returned row.  The
        # component frame remains the explicitly reported physical frame.
        order = tuple(int(value) for value in geometry["internal_order"])

        def external_station_order(values: np.ndarray) -> np.ndarray:
            made = np.empty_like(values)
            for internal_index, external_index in enumerate(order):
                made[external_index] = values[internal_index]
            return made

        membrane_strain = external_station_order(membrane_strain)
        curvature = external_station_order(curvature)
        transverse_shear = external_station_order(transverse_shear)
        membrane_resultants = external_station_order(membrane_resultants)
        bending_resultants = external_station_order(bending_resultants)
        shear_resultants = external_station_order(shear_resultants)
        barycentric = np.column_stack(
            (
                1.0 - HAMMER_POINTS[:, 0] - HAMMER_POINTS[:, 1],
                HAMMER_POINTS[:, 0],
                HAMMER_POINTS[:, 1],
            )
        )
        physical_stations = barycentric @ np.asarray(
            geometry["external_coordinates"], dtype=np.float64
        )
        area = float(geometry["area"])
        return {
            "recovery_scope": RESULTANT_POLICY_ID,
            "qualified_recovery": False,
            "physical_stress_available": False,
            "numerical_fields_excluded": True,
            "membrane_strain": membrane_strain,
            "curvature": curvature,
            "transverse_shear_strain": transverse_shear,
            "membrane_resultants": membrane_resultants,
            "bending_resultants": bending_resultants,
            "transverse_shear_resultants": shear_resultants,
            "hammer_points": np.asarray(HAMMER_POINTS, dtype=np.float64).copy(),
            "external_barycentric_coordinates": barycentric,
            "physical_station_coordinates": physical_stations,
            "internal_order": order,
            "physical_weights": np.full(3, area / 3.0, dtype=np.float64),
            "frame": np.asarray(geometry["frame"], dtype=np.float64).copy(),
            "resultant_policy_id": RESULTANT_POLICY_ID,
        }

    def compute_dead_transverse_pressure_load(
        self,
        mesh: FEMesh,
        pressure: Any,
    ) -> np.ndarray:
        geometry = self._geometry(mesh)
        area = float(geometry["area"])
        if isinstance(pressure, (bool, np.bool_)):
            raise TypeError("strict-flat S3 V2 pressure must be real")
        if isinstance(pressure, (int, float, np.integer, np.floating)):
            value = _real_scalar(pressure, "pressure")
            nodal_work = np.full(3, value * area / 3.0, dtype=np.float64)
        else:
            try:
                values = np.asarray(pressure, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "strict-flat S3 V2 pressure must be uniform or three nodal values"
                ) from exc
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "strict-flat S3 V2 pressure must be uniform or three nodal values"
                )
            nodal_work = (area / 12.0) * np.asarray(
                ((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0)),
                dtype=np.float64,
            ) @ values
        load = np.zeros(18, dtype=np.float64)
        director = np.asarray(self.reference_normal, dtype=np.float64)
        for node, scalar in enumerate(nodal_work):
            load[6 * node : 6 * node + 3] = scalar * director
        return load

    # Explicitly seal every inherited mechanics or persistence route not
    # authorized by the Stage-1 equation map.
    def compute_mass_matrix(self, mesh: FEMesh, material: Material, **kwargs: Any) -> np.ndarray:
        self._unsupported("consistent_mass")

    def compute_geometric_stiffness_matrix(
        self,
        mesh: FEMesh,
        material: Material,
        state: Optional[Any] = None,
    ) -> np.ndarray:
        self._unsupported("geometric_stiffness")

    def init_nonlinear_state(self, num_layers: int) -> Dict[str, Any]:
        self._unsupported("nonlinear_state")

    def compute_nonlinear_response(
        self,
        mesh: FEMesh,
        material: Material,
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        self._unsupported("nonlinear_geometry")

    def compute_stresses(
        self,
        mesh: FEMesh,
        displacements: np.ndarray,
        material: Material,
        return_global: bool = False,
    ) -> Dict[str, Any]:
        self._unsupported("qualified_recovery")

    def to_dict(self) -> Dict[str, Any]:
        self._unsupported("serialization")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "StrictFlatLinearE4PLS3V2ShellElement":
        cls._unsupported("serialization")

    def __getstate__(self) -> Dict[str, Any]:
        self._unsupported("restart")

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self._unsupported("restart")

    def __reduce_ex__(self, protocol: int) -> Any:
        self._unsupported("restart")

    def __copy__(self) -> "StrictFlatLinearE4PLS3V2ShellElement":
        self._unsupported("restart")

    def __deepcopy__(
        self, memo: Dict[int, Any]
    ) -> "StrictFlatLinearE4PLS3V2ShellElement":
        del memo
        self._unsupported("restart")


# The live cache can be reached only through the reviewed public compute
# wrapper installed above; its construction primitive is not a module surface.
del _make_exact_component_compute_wrapper


_ALLOWED_INHERITED_CALLABLES = frozenset(
    {
        "_get_element_displacements",
        "get_dof_mapping",
        "get_node_coordinates",
    }
)
_SEALED_INHERITED_CALLABLES = frozenset(
    {
        "_bending_compression_samples",
        "_build_drilling_b_matrix",
        "_build_shell_b_matrices",
        "_center_frame",
        "_compute_3node_shape_functions",
        "_compute_4node_shape_functions",
        "_compute_6node_shape_functions",
        "_compute_8node_shape_functions",
        "_compute_generalized_section_results",
        "_fallback_edge_direction",
        "_generalized_section_in_frame",
        "_generalized_section_nonlinear_response",
        "_hourglass_stabilization_matrix",
        "_local_dof_transform",
        "_local_frame_and_derivatives",
        "_material_angle",
        "_membrane_compression_from_state",
        "_membrane_compression_samples",
        "_mitc4_shear_b_matrix",
        "_mitc4_shear_samples",
        "_nonlinear_geometry",
        "_normalize",
        "_reference_center",
        "_resultant_samples",
        "_rigid_body_mode_matrix",
        "_stress_second_moment_samples",
        "_tri3_assumed_shear_b_matrix",
        "compute_jacobian",
        "compute_shape_functions",
    }
)


def _sealed_inherited_operator(name: str) -> Any:
    def sealed(self: StrictFlatLinearE4PLS3V2ShellElement, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        self._unsupported(f"legacy_inherited_operator:{name}")

    sealed.__name__ = name
    sealed.__qualname__ = f"StrictFlatLinearE4PLS3V2ShellElement.{name}"
    return sealed


for _sealed_name in _SEALED_INHERITED_CALLABLES:
    setattr(
        StrictFlatLinearE4PLS3V2ShellElement,
        _sealed_name,
        _sealed_inherited_operator(_sealed_name),
    )


type.__setattr__(
    StrictFlatLinearE4PLS3V2ShellElement,
    "_strict_flat_v2_class_frozen",
    True,
)
_initialize_final_class_authority(StrictFlatLinearE4PLS3V2ShellElement)


__all__ = [
    "BLOCKED_OPERATIONS",
    "CAPABILITY_MATRIX",
    "DIRECTOR_POLICY_ID",
    "EQUATION_MAP_SHA256",
    "FORMULATION_ID",
    "FORMULATION_SCHEMA",
    "HAMMER_POINTS",
    "HAMMER_REFERENCE_WEIGHTS",
    "IMPLEMENTATION_ID",
    "PL_COMPLETION_POLICY_ID",
    "PRIMARY_SOURCE_SHA256",
    "QUADRATURE_AUTHORITY_ID",
    "RESULTANT_POLICY_ID",
    "SECTION_POLICY_ID",
    "SELECTOR",
    "_ALLOWED_INHERITED_CALLABLES",
    "_SEALED_INHERITED_CALLABLES",
    "SHEAR_CORRECTION",
    "SOURCE_CONTRACT_SCHEMA",
    "SOURCE_CONTRACT_SHA256",
    "SUPPORTED_OPERATIONS",
    "StrictFlatLinearCapabilityError",
    "StrictFlatLinearE4PLS3V2ShellElement",
]
