"""Opt-in S3 V2D native-parity successor.

V2D preserves the accepted V2C MIN3/CST/PL small-strain operator and adds
native generalized-section integration, model-bound constitutive state, and
an element-independent corotational response.  Activity, contact, and the
bounded stiffness-plan path are formulation-owned; activation remains
fail-closed until its later reviewed gates are complete.

No legacy TRI3 or qualified-Q4 mechanics are dispatched from this module.
"""

from __future__ import annotations

import math
import threading
import weakref
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._qualified_authority_epoch import (
    AuthorityEpochMeta,
    make_authority_epoch_manager,
)
from .element_capabilities import ElementCapabilityError
from .elements import ShellElement
from .e4_pl_element import QualifiedE4PLShellElement
from .e4_pl_s3_v2c_element import (
    HAMMER_POINTS,
    HAMMER_REFERENCE_WEIGHTS,
    PL_COMPLETION_POLICY_ID,
    RELAXATION_AUTHORITY_SHA256,
    RESULTANT_POLICY_ID,
    StrictFlatLinearE4PLS3V2CShellElement,
)
from .e4_pl_s3_initial_fields import integrate_generalized_initial_fields
from .e4_pl_s3_state import (
    DIRECTOR_POLARITY_POLICY_ID,
    DIRECTOR_REVERSAL_TRANSFORM_ID,
    REFERENCE_SURFACE_OFFSET_POLICY_ID,
    REFERENCE_SURFACE_STRAIN_TRANSFORM_ID,
    qualified_s3_lobatto_layers,
)
from .e4_pl_s3_v2d_state import (
    ACTIVITY_DISPOSITION_KEY,
    STATE_LAYOUT_ID,
    STATE_SCHEMA,
    V2DStateError,
    canonical_json_bytes,
    canonical_sha256,
    deserialize_v2d_state,
    initialize_v2d_state,
    seal_v2d_state,
    serialize_v2d_state,
    validate_v2d_state,
)
from .fe_core import FEMesh, Material
from .plasticity import plane_stress_return_map


SELECTOR = "e4-pl-s3-v2d"
FORMULATION_ID = "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"
IMPLEMENTATION_ID = "E4_PL_S3_V2D_RECOVERY_CURRENT_EIGEN_GATE_V1"
FORMULATION_SCHEMA = "anysolver.e4-pl-s3-v2d-native-parity-element-v3"
SOURCE_SELECTION_SHA256 = (
    "DB5750539FB87CA4E4DDA1B37ECEACD65B76DF9A64969808712A4BDD44A45E3D"
)
V2C_OPERATOR_SHA256 = (
    "84FB0B881F0F795BB9FC315A27FF53998BADE58CBAA1EF0A48785A5BE5E086F4"
)
NATIVE_SECTION_POLICY_ID = "S3_V2D_NATIVE_MIN3_GENERALIZED_SECTION_STATIONS_V1"
SERIALIZATION_POLICY_ID = "S3_V2D_ELEMENT_AND_STATE_FINGERPRINT_V3"
NATIVE_STATE_SCHEMA_ID = STATE_SCHEMA
NATIVE_STATE_LAYOUT_ID = STATE_LAYOUT_ID
COROTATIONAL_POLICY_ID = "S3_V2D_EICR_ELEMENT_INDEPENDENT_PULLBACK_V1"
MATERIAL_LIFECYCLE_POLICY_ID = "S3_V2D_HAMMER3_LOBATTO_TRIAL_COMMIT_REVERT_V1"
PRESSURE_SURFACE_POLICY_ID = "ELEMENT_NODAL_REFERENCE_SURFACE_V1"
ACTIVITY_POLICY_ID = "S3_V2D_CLOSED_ACTIVITY_LIFECYCLE_V1"
BATCH_POLICY_ID = "S3_V2D_EXACT_REVISION_BOUND_STIFFNESS_PLAN_V1"
DRILL_SCALE_POLICY_ID = "S3_BASIS_INVARIANT_GENERALIZED_EIGENVALUE_V1"
MASS_POLICY_ID = "S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1"
QUADRATURE_AUTHORITY_ID = "S3_V2D_MIN3_HAMMER3_V1"
GEOMETRIC_STIFFNESS_POLICY_ID = (
    "S3_V2D_CST_UNIFORM_MEMBRANE_COMPRESSION_INITIAL_STRESS_V1"
)
DYNAMIC_ALGEBRAIC_POLICY_ID = "S3_V2D_ALL_NODAL_ROTATIONS_ZERO_INERTIA_V1"
DYNAMIC_ALGEBRAIC_MASS_WITNESS = "S3_V2C_LOCAL_ROTATION_ROWS_EXACT_ZERO_V1"
CBMIN3 = 2.0

_NORMAL_TOLERANCE = 1.0e-10
_DEGENERACY_FACTOR = 64.0 * np.finfo(np.float64).eps
_ROTATIONAL_INDICES = np.asarray((3, 4, 9, 10, 15, 16), dtype=np.intp)
_DRILL_PROJECTOR = np.asarray(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)))
_DRILL_INVERSE_METRIC_SQRT = np.diag((1.0 / math.sqrt(2.0), math.sqrt(2.0)))


class NativeParityCapabilityError(ElementCapabilityError):
    """An operation is outside the currently accepted V2D gate."""


# Generic derived caches that shared solver utilities reset with ``setattr``.
# Data-only writes to them cannot change any instance fact the exact lifecycle
# guard validates, so they never count as a modification (see below).
V2D_DERIVED_CACHE_NAMES = frozenset(
    {
        "_hourglass_stiffness_matrix",
        "_internal_forces",
        "_mass_matrix",
        "_nl_cache",
        "_stiffness_matrix",
    }
)

# Modification counters for live trusted leases.  A nonlinear solve that owns
# V2D elements captures one counter per capture and subscribes it to every V2D
# element.  Any ordinary write or deletion on a subscribed element then advances
# the counter, so the constant-time trusted check can tell that the element
# changed between two complete scans.  Nothing is subscribed outside a live
# capture: writes to V2D elements behave exactly as before, and the shared mesh
# epoch is never touched.  Direct ``__dict__`` writes and ``object.__setattr__``
# remain interpreter-level bypasses outside the supported mutation surface.
_V2D_LEASE_EPOCHS_KEY = "_qualified_v2d_lease_epochs"
_V2D_LEASE_LOCK = threading.Lock()


class _V2DLeaseEpoch:
    """Count of ordinary writes to the V2D elements of one trusted capture.

    Elements hold only weak references, so the subscription ends by itself when
    the capture that owns the counter is released and nothing needs to be
    unsubscribed.
    """

    __slots__ = ("value", "__weakref__")

    def __init__(self) -> None:
        self.value = 0


def _subscribe_v2d_lease_epoch(
    elements: Iterable[Any],
    epoch: _V2DLeaseEpoch,
    *,
    _key: str = _V2D_LEASE_EPOCHS_KEY,
    _lock: Any = _V2D_LEASE_LOCK,
) -> None:
    """Subscribe *epoch* to every element and drop released subscriptions."""

    reference = weakref.ref(epoch)
    with _lock:
        for element in elements:
            namespace = object.__getattribute__(element, "__dict__")
            references = dict.get(namespace, _key)
            if type(references) is not list:
                references = []
                dict.__setitem__(namespace, _key, references)
            references[:] = [item for item in references if item() is not None]
            references.append(reference)


def _advance_v2d_lease_epochs(
    element: Any,
    name: str,
    value: Any,
    *,
    deleting: bool = False,
    _cache_names: frozenset[str] = V2D_DERIVED_CACHE_NAMES,
    _ndarray: type = np.ndarray,
    _namespace_of: Any = object.__getattribute__,
    _key: str = _V2D_LEASE_EPOCHS_KEY,
    _lock: Any = _V2D_LEASE_LOCK,
) -> None:
    """Count an ordinary write for every live capture that covers *element*.

    Increments run under one lock, so concurrent writers cannot lose or reject
    one another's advance.  Solver-owned derived-cache resets are not writes to
    an authority fact and are not counted.
    """

    if (
        not deleting
        and type(name) is str
        and name in _cache_names
        and (value is None or type(value) is _ndarray)
    ):
        return
    references = _namespace_of(element, "__dict__").get(_key)
    if not references:
        return
    for reference in tuple(references):
        epoch = reference()
        if epoch is not None:
            with _lock:
                epoch.value += 1


SUPPORTED_OPERATIONS = frozenset(
    {
        "accepted_v2c_linear_stiffness",
        "accepted_v2c_linear_internal_force",
        "native_linear_generalized_section",
        "native_variational_resultants",
        "dead_uniform_pressure",
        "lumped_translational_mass",
        "stateless_serialization",
        "model_bound_material_state",
        "initial_generalized_fields",
        "layered_plane_stress_material_update",
        "same_formulation_state_serialization",
        "element_independent_corotational_response",
        "director_polarity_reversal",
        "reference_surface_offset",
        "follower_pressure",
        "distributed_couple",
        "solver_integrated_hot_restart",
        "contact_state",
        "activity_state",
        "qualified_batch_path",
        "descriptor_modal",
        "rayleigh_damping",
        "reference_elastic_buckling",
        "local_resultant_recovery",
        "package_round_trip",
    }
)
BLOCKED_OPERATIONS = frozenset(
    {
        "default_activation",
        "python_pickle_restart",
        "v2_or_earlier_state_hot_migration",
    }
)
CAPABILITY_MATRIX = MappingProxyType(
    {
        **{name: "SUPPORTED" for name in SUPPORTED_OPERATIONS},
        **{name: "BLOCKED_PENDING_SUCCESSOR_GATE" for name in BLOCKED_OPERATIONS},
    }
)


def _real_scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"S3 V2D {label} must be a finite real scalar")
    made = float(value)
    if not math.isfinite(made):
        raise ValueError(f"S3 V2D {label} must be finite")
    return made


def _binary64_vector_sha256(value: Any, label: str) -> str:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(18)
    except (TypeError, ValueError) as exc:
        raise V2DStateError(f"S3 V2D {label} must be a finite 18-vector") from exc
    if not np.all(np.isfinite(vector)):
        raise V2DStateError(f"S3 V2D {label} must be a finite 18-vector")
    return canonical_sha256(
        {
            "binary64_little_endian_hex": np.asarray(
                vector, dtype="<f8"
            ).tobytes(order="C").hex().upper(),
            "shape": [18],
        }
    )


def _immutable_vector(value: Any, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"S3 V2D {label} must be a finite 3-vector") from exc
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError(f"S3 V2D {label} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"S3 V2D {label} must be nonzero")
    contiguous = np.ascontiguousarray(vector / norm, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)


def _immutable_serialized_unit_vector(value: Any, label: str) -> np.ndarray:
    """Validate and preserve a serialized unit vector byte-for-byte.

    Qualified restart identity is expressed in binary64.  Re-normalizing an
    already normalized vector is not generally bit-idempotent, so a validated
    serialized vector must be restored without another arithmetic transform.
    """

    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"S3 V2D {label} must be a finite 3-vector") from exc
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError(f"S3 V2D {label} must be a finite 3-vector")
    if not math.isclose(
        float(np.linalg.norm(vector)),
        1.0,
        rel_tol=0.0,
        abs_tol=8.0 * np.finfo(np.float64).eps,
    ):
        raise ValueError(f"S3 V2D serialized {label} must be unit length")
    contiguous = np.ascontiguousarray(vector, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)


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
        transform[base + 3 : base + 6, base + 3 : base + 6] = local_from_global
    return transform


def _invariant_drill_scale(membrane: np.ndarray) -> float:
    matrix = np.asarray(membrane, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("S3 V2D membrane A must be a finite 3x3 matrix")
    matrix = 0.5 * (matrix + matrix.T)
    if float(np.linalg.eigvalsh(matrix)[0]) <= 0.0:
        raise ValueError("S3 V2D membrane A must be positive definite")
    restricted = _DRILL_PROJECTOR.T @ matrix @ _DRILL_PROJECTOR
    canonical = (
        _DRILL_INVERSE_METRIC_SQRT
        @ restricted
        @ _DRILL_INVERSE_METRIC_SQRT
    )
    value = 0.5 * float(np.linalg.eigvalsh(0.5 * (canonical + canonical.T))[0])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("S3 V2D drill scale must be finite and positive")
    return value


class NativeParityE4PLS3V2DShellElement(
    ShellElement,
    metaclass=AuthorityEpochMeta,
):
    """Explicit V2D candidate; no public alias or default selects it."""

    formulation_id = FORMULATION_ID
    implementation_id = IMPLEMENTATION_ID
    selector = SELECTOR
    native_state_schema_id = NATIVE_STATE_SCHEMA_ID
    native_state_layout_id = NATIVE_STATE_LAYOUT_ID
    corotational_policy_id = COROTATIONAL_POLICY_ID
    director_polarity_policy_id = DIRECTOR_POLARITY_POLICY_ID
    director_reversal_transform_id = DIRECTOR_REVERSAL_TRANSFORM_ID
    reference_surface_offset_policy_id = REFERENCE_SURFACE_OFFSET_POLICY_ID
    reference_surface_strain_transform_id = REFERENCE_SURFACE_STRAIN_TRANSFORM_ID
    pressure_surface_policy_id = PRESSURE_SURFACE_POLICY_ID
    activity_policy_id = ACTIVITY_POLICY_ID
    qualified_batch_policy_id = BATCH_POLICY_ID
    quadrature_authority_id = QUADRATURE_AUTHORITY_ID
    dynamic_algebraic_nullity = 9
    dynamic_algebraic_policy = DYNAMIC_ALGEBRAIC_POLICY_ID
    dynamic_algebraic_mass_witness = DYNAMIC_ALGEBRAIC_MASS_WITNESS
    dynamic_algebraic_local_zero_indices = (3, 4, 5, 9, 10, 11, 15, 16, 17)
    legacy_stiffness_batch_eligible = False
    legacy_nonlinear_batch_eligible = False
    formulation_native_total_lagrangian = False
    _legacy_shell_dispatch_forbidden = FORMULATION_ID
    TRI_GAUSS_POINTS_3 = HAMMER_POINTS
    TRI_GAUSS_WEIGHTS_3 = HAMMER_REFERENCE_WEIGHTS

    # These functions are the reviewed V2C source functions themselves, not
    # copied/re-authored equation implementations.
    _membrane_operator = staticmethod(
        StrictFlatLinearE4PLS3V2CShellElement._membrane_operator
    )
    _pl_constraint = staticmethod(
        StrictFlatLinearE4PLS3V2CShellElement._pl_constraint
    )
    _operators = StrictFlatLinearE4PLS3V2CShellElement._operators

    # The counting helper is bound at class creation so a temporarily replaced
    # module global cannot suppress the count; both hooks are protected class
    # entries (see ``_s3_v2d_runtime_epoch_manager``).
    def __setattr__(
        self,
        name: str,
        value: Any,
        _advance: Any = _advance_v2d_lease_epochs,
    ) -> None:
        _advance(self, name, value)
        super().__setattr__(name, value)

    def __delattr__(
        self,
        name: str,
        _advance: Any = _advance_v2d_lease_epochs,
    ) -> None:
        _advance(self, name, None, deleting=True)
        super().__delattr__(name)

    def __init__(
        self,
        element_id: int,
        node_ids: Sequence[int],
        material_name: str = "default",
        *,
        thickness: float = 0.01,
        reference_normal: Sequence[float],
        director_polarity: int = 1,
        reference_surface_offset: float = 0.0,
        material_direction: Optional[Sequence[float]] = None,
        material_angle_deg: float = 0.0,
        shell_section: Optional[Any] = None,
    ) -> None:
        if type(element_id) is not int:
            raise TypeError("S3 V2D element_id must be an exact integer")
        owned_nodes = tuple(node_ids)
        if (
            len(owned_nodes) != 3
            or not all(type(node_id) is int for node_id in owned_nodes)
            or len(set(owned_nodes)) != 3
        ):
            raise ValueError("S3 V2D requires three distinct exact-integer nodes")
        if type(director_polarity) is not int or director_polarity not in (-1, 1):
            raise ValueError("S3 V2D director_polarity must be the integer -1 or +1")
        offset_value = _real_scalar(
            reference_surface_offset, "reference_surface_offset"
        )
        thickness_value = _real_scalar(thickness, "thickness")
        if thickness_value <= 0.0:
            raise ValueError("S3 V2D thickness must be positive")
        normal = _immutable_vector(reference_normal, "reference_normal")
        super().__init__(
            element_id,
            owned_nodes,
            material_name,
            thickness=thickness_value,
            drilling_stabilization=0.0,
            reduced_integration=False,
            hourglass_stabilization=0.0,
            material_direction=(
                None
                if material_direction is None
                else np.asarray(material_direction, dtype=np.float64)
            ),
            material_angle_deg=material_angle_deg,
            shell_section=shell_section,
        )
        self.node_ids = owned_nodes
        self.reference_normal = normal
        self.director_polarity = director_polarity
        self.reference_surface_offset = 0.0 if offset_value == 0.0 else offset_value

    @property
    def physical_reference_director(self) -> np.ndarray:
        return (
            float(self.director_polarity)
            * np.asarray(self.reference_normal, dtype=np.float64)
        ).copy()

    @property
    def material_mid_surface_offset_from_reference(self) -> float:
        return -float(self.reference_surface_offset)

    def material_surface_offset_from_reference(
        self, normalized_thickness_coordinate: Any
    ) -> float:
        coordinate = _real_scalar(
            normalized_thickness_coordinate, "normalized_thickness_coordinate"
        )
        if coordinate < -1.0 or coordinate > 1.0:
            raise ValueError("S3 V2D surface coordinate must be in [-1,1]")
        return 0.5 * self.thickness * coordinate - self.reference_surface_offset

    def native_reference_directors(self, mesh: FEMesh) -> np.ndarray:
        """Return the physical facet directors consumed by contact routing."""

        geometry = self._geometry(mesh)
        director = (
            float(self.director_polarity)
            * np.asarray(geometry["frame"], dtype=np.float64)[:, 2]
        )
        return np.repeat(director[None, :], 3, axis=0)

    @property
    def gauss_points(self) -> np.ndarray:
        return HAMMER_POINTS

    @property
    def gauss_weights(self) -> np.ndarray:
        return HAMMER_REFERENCE_WEIGHTS

    @property
    def shear_gauss_points(self) -> np.ndarray:
        return HAMMER_POINTS

    @property
    def shear_gauss_weights(self) -> np.ndarray:
        return HAMMER_REFERENCE_WEIGHTS

    def capability_matrix(self) -> Dict[str, str]:
        return dict(CAPABILITY_MATRIX)

    def validate_quadrature_authority(self) -> None:
        """Fail closed unless the frozen MIN3/Hammer-3 station rule is exact."""

        self._validate_configuration()
        if (
            self.quadrature_authority_id != QUADRATURE_AUTHORITY_ID
            or not np.array_equal(self.gauss_points, HAMMER_POINTS)
            or not np.array_equal(self.gauss_weights, HAMMER_REFERENCE_WEIGHTS)
            or not np.array_equal(self.shear_gauss_points, HAMMER_POINTS)
            or not np.array_equal(self.shear_gauss_weights, HAMMER_REFERENCE_WEIGHTS)
        ):
            raise NativeParityCapabilityError("S3 V2D quadrature authority changed")

    def corotational_reference_frame(self, coordinates: Any) -> np.ndarray:
        """Return the objective triangle frame used only by the EICR wrapper."""

        made = np.asarray(coordinates, dtype=np.float64)
        if made.shape != (3, 3) or not np.all(np.isfinite(made)):
            raise ValueError("S3 V2D corotational coordinates must be finite (3,3)")
        first = made[1] - made[0]
        second = made[2] - made[0]
        first_norm = float(np.linalg.norm(first))
        cross = np.cross(first, second)
        cross_norm = float(np.linalg.norm(cross))
        scale = max(
            float(first @ first),
            float(second @ second),
            float((made[2] - made[1]) @ (made[2] - made[1])),
            np.finfo(np.float64).tiny,
        )
        if (
            not math.isfinite(first_norm)
            or not math.isfinite(cross_norm)
            or first_norm <= 0.0
            or cross_norm <= _DEGENERACY_FACTOR * scale
        ):
            raise ValueError("S3 V2D corotational triangle is degenerate")
        x_axis = first / first_norm
        z_axis = cross / cross_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= float(np.linalg.norm(y_axis))
        return np.column_stack((x_axis, y_axis, z_axis))

    def _material_angle(self, local_frame: np.ndarray) -> float:
        """Resolve the physical material direction without legacy mechanics."""

        angle = math.radians(float(self.material_angle_deg))
        if self.material_direction is None:
            return angle
        frame = np.asarray(local_frame, dtype=np.float64).reshape(3, 3)
        direction = np.asarray(self.material_direction, dtype=np.float64)
        components = np.asarray(
            (float(direction @ frame[:, 0]), float(direction @ frame[:, 1])),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(components))
        if norm <= 1.0e-10 * max(float(np.linalg.norm(direction)), 1.0):
            raise ValueError("S3 V2D material_direction is parallel to the normal")
        return float(math.atan2(components[1], components[0]) + angle)

    def _generalized_section_in_frame(self, local_frame: np.ndarray) -> Any:
        if self.shell_section is None:
            return None
        return self.shell_section.rotated(self._material_angle(local_frame))

    @staticmethod
    def _unsupported(operation: str) -> None:
        raise NativeParityCapabilityError(
            f"S3 V2D capability {operation!r} is pending a successor gate"
        )

    def _validate_configuration(self) -> None:
        if type(self) is not NativeParityE4PLS3V2DShellElement:
            raise NativeParityCapabilityError("S3 V2D requires its exact production class")
        if (
            self.formulation_id != FORMULATION_ID
            or self.selector != SELECTOR
            or self._legacy_shell_dispatch_forbidden != FORMULATION_ID
        ):
            raise NativeParityCapabilityError("S3 V2D class identity changed")
        if (
            type(self.node_ids) is not tuple
            or len(self.node_ids) != 3
            or len(set(self.node_ids)) != 3
        ):
            raise ValueError("S3 V2D connectivity authority changed")
        if self.drilling_stabilization != 0.0 or self.hourglass_stabilization != 0.0:
            raise NativeParityCapabilityError("S3 V2D forbids legacy stabilizations")
        if self.reduced_integration is not False:
            raise NativeParityCapabilityError("S3 V2D quadrature authority changed")
        normal = self.reference_normal
        if (
            type(normal) is not np.ndarray
            or normal.shape != (3,)
            or normal.dtype != np.dtype(np.float64)
            or normal.flags.writeable
            or not np.all(np.isfinite(normal))
        ):
            raise ValueError("S3 V2D reference-normal authority changed")
        if not math.isclose(
            float(np.linalg.norm(normal)),
            1.0,
            rel_tol=0.0,
            abs_tol=8.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("S3 V2D reference normal is not unit length")
        if type(self.director_polarity) is not int or self.director_polarity not in (
            -1,
            1,
        ):
            raise ValueError("S3 V2D director-polarity authority changed")
        if (
            isinstance(self.reference_surface_offset, (bool, np.bool_))
            or not isinstance(
                self.reference_surface_offset,
                (int, float, np.integer, np.floating),
            )
            or not math.isfinite(float(self.reference_surface_offset))
        ):
            raise ValueError("S3 V2D reference-surface offset authority changed")

    def _coordinates(self, mesh: FEMesh) -> np.ndarray:
        if type(mesh) is not FEMesh:
            raise TypeError("S3 V2D requires an exact FEMesh")
        coordinates = np.empty((3, 3), dtype=np.float64)
        for index, node_id in enumerate(self.node_ids):
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"S3 V2D references missing node {node_id}")
            value = np.asarray(node.coords(), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"S3 V2D node {node_id} has invalid coordinates")
            coordinates[index] = value
        return coordinates

    def _validate_model_scope(self, mesh: FEMesh) -> None:
        if type(mesh) is not FEMesh:
            raise TypeError("S3 V2D requires an exact FEMesh")
        elements = object.__getattribute__(mesh, "__dict__").get("elements")
        if not elements:
            return
        if not isinstance(elements, dict) or elements.get(self.element_id) is not self:
            raise NativeParityCapabilityError(
                "S3 V2D evaluation requires exact model-registry ownership"
            )
        allowed = (
            NativeParityE4PLS3V2DShellElement,
            StrictFlatLinearE4PLS3V2CShellElement,
            QualifiedE4PLShellElement,
        )
        if any(
            isinstance(element, ShellElement) and type(element) not in allowed
            for element in elements.values()
        ):
            raise NativeParityCapabilityError(
                "S3 V2D mixed shell scope admits only exact V2D, accepted V2C and qualified Q4"
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
        if doubled_area <= _DEGENERACY_FACTOR * edge_scale:
            raise ValueError("S3 V2D requires a nondegenerate flat triangle")
        tangency = max(abs(float(edge_12 @ normal)), abs(float(edge_13 @ normal)))
        if tangency > _NORMAL_TOLERANCE * math.sqrt(edge_scale):
            raise NativeParityCapabilityError("S3 V2D reference normal is not facet-normal")
        signed = float(cross @ normal)
        if abs(signed) < (1.0 - _NORMAL_TOLERANCE) * doubled_area:
            raise NativeParityCapabilityError("S3 V2D reference normal is incompatible")
        order: Tuple[int, int, int] = (0, 1, 2) if signed > 0.0 else (0, 2, 1)
        coordinates = external[np.asarray(order, dtype=np.intp)]
        x_axis = coordinates[1] - coordinates[0]
        x_axis /= float(np.linalg.norm(x_axis))
        y_axis = np.cross(normal, x_axis)
        y_axis /= float(np.linalg.norm(y_axis))
        frame = np.column_stack((x_axis, y_axis, normal))
        relative = coordinates - coordinates[0]
        local = np.column_stack((relative @ x_axis, relative @ y_axis))
        jacobian = np.asarray(
            (
                (local[1, 0] - local[0, 0], local[2, 0] - local[0, 0]),
                (local[1, 1] - local[0, 1], local[2, 1] - local[0, 1]),
            ),
            dtype=np.float64,
        )
        determinant = float(np.linalg.det(jacobian))
        if not math.isfinite(determinant) or determinant <= 0.0:
            raise RuntimeError("S3 V2D failed to construct a positive local chart")
        natural_gradients = np.asarray(
            ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64
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
        if self.shell_section is not None:
            raise NativeParityCapabilityError(
                "S3 V2D generalized sections use the native station integrator"
            )
        # The exact reviewed V2C function supplies the elastic matrices and
        # binary64 operation ordering required by the overlap certificate.
        return StrictFlatLinearE4PLS3V2CShellElement._constitutive(self, material)

    def _elastic_baseline_constitutive(self, material: Material) -> Dict[str, Any]:
        if type(material) is not Material:
            raise NativeParityCapabilityError(
                "S3 V2D layered material currently requires exact isotropic Material"
            )
        elastic_modulus = _real_scalar(
            getattr(material, "elastic_modulus", None), "elastic_modulus"
        )
        poisson_ratio = _real_scalar(
            getattr(material, "poisson_ratio", None), "poisson_ratio"
        )
        if elastic_modulus <= 0.0 or not -1.0 < poisson_ratio < 0.5:
            raise ValueError("S3 V2D isotropic elastic constants are inadmissible")
        scale = elastic_modulus / (1.0 - poisson_ratio**2)
        plane = scale * np.asarray(
            (
                (1.0, poisson_ratio, 0.0),
                (poisson_ratio, 1.0, 0.0),
                (0.0, 0.0, 0.5 * (1.0 - poisson_ratio)),
            ),
            dtype=np.float64,
        )
        shear_scalar = (
            (5.0 / 6.0)
            * elastic_modulus
            * self.thickness
            / (2.0 * (1.0 + poisson_ratio))
        )
        return {
            "A": self.thickness * plane,
            "B": np.zeros((3, 3), dtype=np.float64),
            "D": self.thickness**3 / 12.0 * plane,
            "H": shear_scalar * np.eye(2, dtype=np.float64),
            "E": elastic_modulus,
            "nu": poisson_ratio,
            "drill_scale": self.thickness
            * elastic_modulus
            / (2.0 * (1.0 + poisson_ratio)),
        }

    @staticmethod
    def _is_v2c_elastic_overlap(material: Material) -> bool:
        if type(material) is not Material:
            return False
        if getattr(material, "hardening_curve", None) is not None:
            return False
        raw_yield = getattr(material, "yield_stress", 0.0)
        try:
            return float(raw_yield or 0.0) == 0.0
        except (TypeError, ValueError):
            return False

    def _native_section(self, geometry: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        section = self._generalized_section_in_frame(
            np.asarray(geometry["frame"], dtype=np.float64)
        )
        if section is None:
            raise NativeParityCapabilityError("S3 V2D native section is absent")
        matrices = {
            "A": np.asarray(section.A, dtype=np.float64),
            "B": np.asarray(section.B, dtype=np.float64),
            "D": np.asarray(section.D, dtype=np.float64),
            "H": np.asarray(section.As, dtype=np.float64),
        }
        if not all(np.all(np.isfinite(value)) for value in matrices.values()):
            raise ValueError("S3 V2D generalized section is nonfinite")
        return matrices

    def _director_generalized_transform(self) -> np.ndarray:
        transform = np.eye(8, dtype=np.float64)
        transform[3:, 3:] *= float(self.director_polarity)
        return transform

    def _reference_surface_strain_transform(self) -> np.ndarray:
        transform = np.eye(8, dtype=np.float64)
        offset = float(self.reference_surface_offset)
        if offset != 0.0:
            transform[:3, 3:6] = -offset * np.eye(3, dtype=np.float64)
        return transform

    def _director_adjusted_section(
        self, section: Mapping[str, Any]
    ) -> Dict[str, Any]:
        adjusted = dict(section)
        adjusted["A"] = np.asarray(section["A"], dtype=np.float64)
        adjusted["B"] = (
            float(self.director_polarity)
            * np.asarray(section["B"], dtype=np.float64)
        )
        adjusted["D"] = np.asarray(section["D"], dtype=np.float64)
        adjusted["H"] = np.asarray(section["H"], dtype=np.float64)
        return adjusted

    def _effective_station_operators(
        self, operators: Mapping[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        polarity = float(self.director_polarity)
        curvature = polarity * np.asarray(operators["B_b"], dtype=np.float64)
        shear = polarity * np.asarray(operators["B_s"], dtype=np.float64)
        membrane = np.broadcast_to(
            np.asarray(operators["B_m"], dtype=np.float64),
            curvature.shape,
        ).copy()
        offset = float(self.reference_surface_offset)
        if offset != 0.0:
            membrane -= offset * curvature
        return membrane, curvature, shear

    @staticmethod
    def _generalized_constitutive(
        section: Mapping[str, Any], phi_squared: float
    ) -> np.ndarray:
        constitutive = np.zeros((8, 8), dtype=np.float64)
        constitutive[:3, :3] = np.asarray(section["A"], dtype=np.float64)
        constitutive[:3, 3:6] = np.asarray(section["B"], dtype=np.float64)
        constitutive[3:6, :3] = np.asarray(section["B"], dtype=np.float64).T
        constitutive[3:6, 3:6] = np.asarray(section["D"], dtype=np.float64)
        constitutive[6:, 6:] = (
            float(phi_squared) * np.asarray(section["H"], dtype=np.float64)
        )
        return constitutive

    def _state_material_mode(self) -> str:
        return (
            "GENERALIZED_SECTION"
            if self.shell_section is not None
            else "LAYERED_PLANE_STRESS"
        )

    @staticmethod
    def _validated_num_layers(value: Any) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise V2DStateError("S3 V2D num_layers must be an integer")
        layers = int(value)
        if layers <= 0:
            raise V2DStateError("S3 V2D num_layers must be positive")
        return layers

    @staticmethod
    def _curve_descriptor(curve: Any) -> Any:
        if curve is None:
            return None
        fields = (
            "sigma_prop",
            "sigma_yield",
            "sigma_yield_2",
            "eps_p_y1",
            "eps_p_y2",
            "K",
            "n",
            "_power_offset",
        )
        descriptor: Dict[str, float] = {}
        for name in fields:
            if not hasattr(curve, name):
                raise V2DStateError(
                    f"S3 V2D hardening curve lacks required field {name}"
                )
            descriptor[name] = _real_scalar(getattr(curve, name), name)
        return descriptor

    def _state_identity(
        self,
        mesh: FEMesh,
        material: Material,
        num_layers: int,
    ) -> str:
        geometry = self._geometry(mesh)
        material_descriptor: Dict[str, Any]
        if self.shell_section is None:
            material_descriptor = {
                "type": type(material).__name__,
                "elastic_modulus": _real_scalar(
                    getattr(material, "elastic_modulus", None), "elastic_modulus"
                ),
                "poisson_ratio": _real_scalar(
                    getattr(material, "poisson_ratio", None), "poisson_ratio"
                ),
                "yield_stress": _real_scalar(
                    getattr(material, "yield_stress", 0.0), "yield_stress"
                ),
                "hardening_curve": self._curve_descriptor(
                    getattr(material, "hardening_curve", None)
                ),
            }
        else:
            section = self._native_section(geometry)
            material_descriptor = {
                "type": "GeneralizedShellSection",
                "A": section["A"],
                "B": section["B"],
                "D": section["D"],
                "As": section["H"],
            }
        return canonical_sha256(
            {
                "identity_id": "S3_V2D_MODEL_BOUND_STATE_IDENTITY_V1",
                "formulation_id": FORMULATION_ID,
                "element_id": int(self.element_id),
                "node_ids": tuple(int(value) for value in self.node_ids),
                "reference_coordinates": geometry["external_coordinates"],
                "reference_normal": self.reference_normal,
                "director_polarity": int(self.director_polarity),
                "reference_surface_offset": float(self.reference_surface_offset),
                "thickness": float(self.thickness),
                "material_direction": self.material_direction,
                "material_angle_deg": float(self.material_angle_deg),
                "material_mode": self._state_material_mode(),
                "material": material_descriptor,
                "num_layers": int(num_layers),
                "source_selection_sha256": SOURCE_SELECTION_SHA256,
            }
        )

    @staticmethod
    def _initial_station_rows(value: Any, label: str) -> np.ndarray:
        try:
            made = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise V2DStateError(f"S3 V2D {label} must be numeric") from exc
        if made.shape == (3,):
            made = np.broadcast_to(made, (3, 3)).copy()
        elif made.shape == (1, 3):
            made = np.broadcast_to(made, (3, 3)).copy()
        elif made.shape == (3, 3):
            made = made.copy()
        else:
            raise V2DStateError(
                f"S3 V2D {label} must have shape (3,), (1,3), or (3,3)"
            )
        if not np.all(np.isfinite(made)):
            raise V2DStateError(f"S3 V2D {label} must be finite")
        return made

    def _initial_generalized_fields(
        self, initial_fields: Optional[Mapping[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        names = (
            "initial_membrane_stress",
            "initial_bending_stress",
            "initial_membrane_prestrain",
            "initial_curvature_prestrain",
        )
        normalized = {
            name: np.zeros((3, 3), dtype=np.float64) for name in names
        }
        if initial_fields is not None:
            if not isinstance(initial_fields, Mapping):
                raise V2DStateError("S3 V2D initial fields must be a mapping")
            unknown = set(initial_fields) - set(names)
            if unknown:
                raise V2DStateError(
                    f"S3 V2D initial-field keys are unknown: {sorted(unknown)}"
                )
            for name, value in initial_fields.items():
                normalized[name] = self._initial_station_rows(value, name)
        polarity = float(self.director_polarity)
        normalized["initial_bending_stress"] *= polarity
        normalized["initial_curvature_prestrain"] *= polarity
        return integrate_generalized_initial_fields(
            normalized,
            self.thickness,
            station_count=3,
        )

    def init_model_bound_nonlinear_state(
        self,
        mesh: FEMesh,
        material: Material,
        num_layers: int,
        *,
        initial_fields: Optional[Mapping[str, Any]] = None,
        initial_field_provenance: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._validate_configuration()
        layers = self._validated_num_layers(num_layers)
        prestrain, resultant = self._initial_generalized_fields(initial_fields)
        return initialize_v2d_state(
            element_id=self.element_id,
            node_ids=self.node_ids,
            element_identity_sha256=self._state_identity(
                mesh, material, layers
            ),
            num_layers=layers,
            material_mode=self._state_material_mode(),
            initial_generalized_prestrain=prestrain,
            initial_generalized_resultant=resultant,
            initial_field_provenance=initial_field_provenance,
        )

    def validate_model_bound_nonlinear_state(
        self,
        mesh: FEMesh,
        material: Material,
        state: Mapping[str, Any],
        num_layers: int,
        *,
        expected_committed_total_u: Any | None = None,
    ) -> Dict[str, Any]:
        layers = self._validated_num_layers(num_layers)
        validated = validate_v2d_state(
            state,
            element_id=self.element_id,
            node_ids=self.node_ids,
            element_identity_sha256=self._state_identity(
                mesh, material, layers
            ),
            num_layers=layers,
            material_mode=self._state_material_mode(),
            expected_committed_total_u=expected_committed_total_u,
        )
        if validated[ACTIVITY_DISPOSITION_KEY] is not None:
            raise V2DStateError(
                "S3 V2D noncurrent activity state cannot supply ACTIVE mechanics"
            )
        return validated

    def _validated_activity_core(
        self,
        mesh: FEMesh,
        material: Material,
        state: Mapping[str, Any],
        num_layers: int,
        *,
        expected_committed_total_u: Any | None = None,
    ) -> Dict[str, Any]:
        layers = self._validated_num_layers(num_layers)
        validated = validate_v2d_state(
            state,
            element_id=self.element_id,
            node_ids=self.node_ids,
            element_identity_sha256=self._state_identity(
                mesh, material, layers
            ),
            num_layers=layers,
            material_mode=self._state_material_mode(),
        )
        made = dict(validated)
        made[ACTIVITY_DISPOSITION_KEY] = None
        core = seal_v2d_state(made)
        return self.validate_model_bound_nonlinear_state(
            mesh,
            material,
            core,
            layers,
            expected_committed_total_u=expected_committed_total_u,
        )

    def _activity_core_identity(
        self, state: Mapping[str, Any], num_layers: int
    ) -> Dict[str, Any]:
        return {
            "activity_policy_id": ACTIVITY_POLICY_ID,
            "core_committed_total_u_sha256": _binary64_vector_sha256(
                state["committed_total_u"], "activity committed_total_u"
            ),
            "core_state_integrity_sha256": str(
                state["state_integrity_sha256"]
            ),
            "element_id": int(state["element_id"]),
            "element_identity_sha256": str(
                state["element_identity_sha256"]
            ),
            "formulation_id": FORMULATION_ID,
            "formulation_schema": FORMULATION_SCHEMA,
            "material_mode": str(state["material_mode"]),
            "node_ids": [int(value) for value in state["node_ids"]],
            "solver_kinematics": str(state["solver_kinematics"]),
            "solver_num_layers": int(num_layers),
            "state_layout_id": NATIVE_STATE_LAYOUT_ID,
            "state_schema": NATIVE_STATE_SCHEMA_ID,
        }

    def _deleted_disposition(
        self,
        core: Mapping[str, Any],
        accepted_local_u: Any,
        num_layers: int,
        *,
        deletion_step_index: int,
        deletion_load_factor: float,
        residual_stiffness_fraction: float,
        trigger_name: str,
    ) -> Dict[str, Any]:
        try:
            displacement = np.asarray(
                accepted_local_u, dtype=np.float64
            ).reshape(18)
        except (TypeError, ValueError) as exc:
            raise V2DStateError(
                "S3 V2D deletion requires a finite accepted 18-vector"
            ) from exc
        if not np.all(np.isfinite(displacement)):
            raise V2DStateError(
                "S3 V2D deletion requires a finite accepted 18-vector"
            )
        step = int(deletion_step_index)
        load_factor = float(deletion_load_factor)
        residual = float(residual_stiffness_fraction)
        trigger = str(trigger_name)
        if step <= 0 or not math.isfinite(load_factor):
            raise V2DStateError("S3 V2D deletion coordinates are invalid")
        if not math.isfinite(residual) or not 0.0 <= residual <= 1.0:
            raise V2DStateError("S3 V2D deletion residual fraction is invalid")
        if not trigger:
            raise V2DStateError("S3 V2D deletion trigger must not be empty")
        identity = self._activity_core_identity(core, num_layers)
        displacement_sha = _binary64_vector_sha256(
            displacement, "deletion accepted_local_u"
        )
        if displacement_sha != identity["core_committed_total_u_sha256"]:
            raise V2DStateError(
                "S3 V2D deletion displacement differs from frozen history"
            )
        payload = {
            **identity,
            "accepted_local_u": displacement.tolist(),
            "accepted_local_u_sha256": displacement_sha,
            "constitutive_history_semantics": "FROZEN_AT_DELETION_ACCEPTED_STATE",
            "deletion_load_factor": load_factor,
            "deletion_step_index": step,
            "operator_semantics": "CURRENT_FORCE_AND_TANGENT_FROM_FROZEN_PARENT_THEN_SCALED",
            "residual_stiffness_fraction": residual,
            "status": "DELETED_FROZEN_NONCURRENT",
            "trigger_name": trigger,
        }
        payload["disposition_sha256"] = canonical_sha256(payload)
        return payload

    def _failed_disposition(
        self,
        core: Mapping[str, Any],
        failed_local_u: Any,
        num_layers: int,
        *,
        failure_reason: str,
    ) -> Dict[str, Any]:
        try:
            displacement = np.asarray(
                failed_local_u, dtype=np.float64
            ).reshape(18)
        except (TypeError, ValueError) as exc:
            raise V2DStateError(
                "S3 V2D failed state requires a finite 18-vector"
            ) from exc
        if not np.all(np.isfinite(displacement)):
            raise V2DStateError(
                "S3 V2D failed state requires a finite 18-vector"
            )
        reason = str(failure_reason)
        if not reason:
            raise V2DStateError("S3 V2D failed state requires a reason")
        payload = {
            **self._activity_core_identity(core, num_layers),
            "failed_local_u": displacement.tolist(),
            "failed_local_u_sha256": _binary64_vector_sha256(
                displacement, "failed_local_u"
            ),
            "failure_reason": reason,
            "semantics": "MATERIALIZED_RESULT_ONLY_NOT_ACTIVE_EVIDENCE",
            "status": "FAILED_NONAUTHORITATIVE",
        }
        payload["disposition_sha256"] = canonical_sha256(payload)
        return payload

    def seal_noncurrent_deleted_state(
        self,
        mesh: FEMesh,
        material: Material,
        accepted_local_u: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        deletion_step_index: int,
        deletion_load_factor: float,
        residual_stiffness_fraction: float,
        trigger_name: str,
    ) -> Dict[str, Any]:
        before = canonical_json_bytes(state)
        core = self.validate_model_bound_nonlinear_state(
            mesh,
            material,
            state,
            num_layers,
            expected_committed_total_u=accepted_local_u,
        )
        marker = self._deleted_disposition(
            core,
            accepted_local_u,
            int(num_layers),
            deletion_step_index=deletion_step_index,
            deletion_load_factor=deletion_load_factor,
            residual_stiffness_fraction=residual_stiffness_fraction,
            trigger_name=trigger_name,
        )
        made = dict(core)
        made[ACTIVITY_DISPOSITION_KEY] = marker
        sealed = seal_v2d_state(made)
        if canonical_json_bytes(state) != before:
            raise RuntimeError("S3 V2D deletion sealing mutated its input")
        return sealed

    def validate_noncurrent_deleted_state(
        self,
        mesh: FEMesh,
        material: Material,
        state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        expected_deletion_step_index: int | None = None,
        expected_deletion_load_factor: float | None = None,
        expected_residual_stiffness_fraction: float | None = None,
        expected_trigger_name: str | None = None,
    ) -> str:
        layers = self._validated_num_layers(num_layers)
        validated = validate_v2d_state(
            state,
            element_id=self.element_id,
            node_ids=self.node_ids,
            element_identity_sha256=self._state_identity(
                mesh, material, layers
            ),
            num_layers=layers,
            material_mode=self._state_material_mode(),
        )
        raw = validated[ACTIVITY_DISPOSITION_KEY]
        if not isinstance(raw, Mapping) or raw.get("status") != (
            "DELETED_FROZEN_NONCURRENT"
        ):
            raise V2DStateError("S3 V2D deleted disposition is absent")
        core = self._validated_activity_core(
            mesh,
            material,
            validated,
            layers,
            expected_committed_total_u=raw.get("accepted_local_u"),
        )
        expected = self._deleted_disposition(
            core,
            raw.get("accepted_local_u"),
            layers,
            deletion_step_index=raw.get("deletion_step_index"),
            deletion_load_factor=raw.get("deletion_load_factor"),
            residual_stiffness_fraction=raw.get(
                "residual_stiffness_fraction"
            ),
            trigger_name=raw.get("trigger_name"),
        )
        if canonical_json_bytes(raw) != canonical_json_bytes(expected):
            raise V2DStateError("S3 V2D deleted disposition is incompatible")
        if expected_deletion_step_index is not None and int(
            raw["deletion_step_index"]
        ) != int(expected_deletion_step_index):
            raise V2DStateError("S3 V2D deletion step is incompatible")
        if expected_deletion_load_factor is not None and float(
            raw["deletion_load_factor"]
        ) != float(expected_deletion_load_factor):
            raise V2DStateError("S3 V2D deletion load factor is incompatible")
        if expected_residual_stiffness_fraction is not None and float(
            raw["residual_stiffness_fraction"]
        ) != float(expected_residual_stiffness_fraction):
            raise V2DStateError("S3 V2D deletion residual fraction is incompatible")
        if expected_trigger_name is not None and str(
            raw["trigger_name"]
        ) != str(expected_trigger_name):
            raise V2DStateError("S3 V2D deletion trigger is incompatible")
        rebuilt = dict(core)
        rebuilt[ACTIVITY_DISPOSITION_KEY] = expected
        rebuilt = seal_v2d_state(rebuilt)
        if canonical_json_bytes(rebuilt) != canonical_json_bytes(validated):
            raise V2DStateError("S3 V2D deleted state integrity is incompatible")
        return str(rebuilt["state_integrity_sha256"])

    def restore_noncurrent_deleted_state_for_internal_use(
        self,
        mesh: FEMesh,
        material: Material,
        state: Mapping[str, Any],
        num_layers: int = 5,
        **expected: Any,
    ) -> Dict[str, Any]:
        self.validate_noncurrent_deleted_state(
            mesh, material, state, num_layers, **expected
        )
        return self._validated_activity_core(
            mesh, material, state, num_layers
        )

    def mark_noncurrent_failed_state(
        self,
        mesh: FEMesh,
        material: Material,
        failed_local_u: Any,
        state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        failure_reason: str,
    ) -> Dict[str, Any]:
        before = canonical_json_bytes(state)
        core = self.validate_model_bound_nonlinear_state(
            mesh, material, state, num_layers
        )
        marker = self._failed_disposition(
            core,
            failed_local_u,
            int(num_layers),
            failure_reason=failure_reason,
        )
        made = dict(core)
        made[ACTIVITY_DISPOSITION_KEY] = marker
        sealed = seal_v2d_state(made)
        if canonical_json_bytes(state) != before:
            raise RuntimeError("S3 V2D failure marking mutated its input")
        return sealed

    def validate_noncurrent_failed_state(
        self,
        mesh: FEMesh,
        material: Material,
        state: Mapping[str, Any],
        num_layers: int = 5,
    ) -> str:
        layers = self._validated_num_layers(num_layers)
        validated = validate_v2d_state(
            state,
            element_id=self.element_id,
            node_ids=self.node_ids,
            element_identity_sha256=self._state_identity(
                mesh, material, layers
            ),
            num_layers=layers,
            material_mode=self._state_material_mode(),
        )
        raw = validated[ACTIVITY_DISPOSITION_KEY]
        if not isinstance(raw, Mapping) or raw.get("status") != (
            "FAILED_NONAUTHORITATIVE"
        ):
            raise V2DStateError("S3 V2D failed disposition is absent")
        core = self._validated_activity_core(mesh, material, validated, layers)
        expected = self._failed_disposition(
            core,
            raw.get("failed_local_u"),
            layers,
            failure_reason=raw.get("failure_reason"),
        )
        if canonical_json_bytes(raw) != canonical_json_bytes(expected):
            raise V2DStateError("S3 V2D failed disposition is incompatible")
        rebuilt = dict(core)
        rebuilt[ACTIVITY_DISPOSITION_KEY] = expected
        rebuilt = seal_v2d_state(rebuilt)
        if canonical_json_bytes(rebuilt) != canonical_json_bytes(validated):
            raise V2DStateError("S3 V2D failed state integrity is incompatible")
        return str(rebuilt["state_integrity_sha256"])

    def serialize_nonlinear_state(
        self,
        mesh: FEMesh,
        material: Material,
        state: Mapping[str, Any],
        num_layers: int,
    ) -> bytes:
        validated = self.validate_model_bound_nonlinear_state(
            mesh, material, state, num_layers
        )
        return serialize_v2d_state(validated)

    def deserialize_nonlinear_state(
        self,
        mesh: FEMesh,
        material: Material,
        raw: bytes,
        num_layers: int,
        *,
        expected_committed_total_u: Any | None = None,
    ) -> Dict[str, Any]:
        decoded = deserialize_v2d_state(raw)
        return self.validate_model_bound_nonlinear_state(
            mesh,
            material,
            decoded,
            num_layers,
            expected_committed_total_u=expected_committed_total_u,
        )

    def seal_solver_integrated_nonlinear_state(
        self,
        mesh: FEMesh,
        material: Material,
        state: Mapping[str, Any],
        num_layers: int,
        committed_total_u: Any,
        *,
        kinematics: str,
    ) -> Dict[str, Any]:
        validated = self.validate_model_bound_nonlinear_state(
            mesh, material, state, num_layers
        )
        try:
            global_u = np.asarray(committed_total_u, dtype=np.float64).reshape(18)
        except (TypeError, ValueError) as exc:
            raise V2DStateError(
                "S3 V2D solver state requires a finite committed 18-vector"
            ) from exc
        if not np.all(np.isfinite(global_u)):
            raise V2DStateError(
                "S3 V2D solver state requires a finite committed 18-vector"
            )
        normalized = str(kinematics).strip().replace("-", "_").upper()
        if normalized not in {"LINEAR", "VON_KARMAN", "COROTATIONAL"}:
            raise V2DStateError("S3 V2D solver kinematics is unsupported")
        if normalized == "VON_KARMAN" and validated["solver_kinematics"] == "UNBOUND":
            # A stopped/fully constrained solve may never have evaluated a
            # trial. Recover once from its virgin parent, with explicit mode.
            _force, _tangent, validated = self.compute_nonlinear_response(
                mesh, material, global_u, validated, num_layers, True,
                von_karman=True,
            )
        made = dict(validated)
        made["committed_total_u"] = global_u.copy()
        made["solver_kinematics"] = normalized
        sealed = seal_v2d_state(made)
        return self.validate_model_bound_nonlinear_state(
            mesh,
            material,
            sealed,
            num_layers,
            expected_committed_total_u=global_u,
        )

    @staticmethod
    def _globalize(matrix: np.ndarray, transform: np.ndarray) -> np.ndarray:
        made = transform.T @ matrix @ transform
        return 0.5 * (made + made.T)

    def _native_generalized_components(
        self, mesh: FEMesh, material: Material
    ) -> Dict[str, Any]:
        geometry = self._geometry(mesh)
        if self.shell_section is None:
            raw_section = self._elastic_baseline_constitutive(material)
        else:
            raw_section = self._native_section(geometry)
        section = self._director_adjusted_section(raw_section)
        operators = self._operators(geometry, raw_section)
        membrane_operators, curvature_operators, shear_operators = (
            self._effective_station_operators(operators)
        )
        area = float(geometry["area"])
        weight = area / 3.0
        membrane_local = np.zeros((18, 18), dtype=np.float64)
        bending_local = np.zeros((18, 18), dtype=np.float64)
        coupling_local = np.zeros((18, 18), dtype=np.float64)
        shear_local = np.zeros((18, 18), dtype=np.float64)
        unrelaxed_shear = np.zeros((18, 18), dtype=np.float64)
        for membrane, bending, shear in zip(
            membrane_operators, curvature_operators, shear_operators
        ):
            membrane_local += membrane.T @ section["A"] @ membrane * weight
            bending_local += bending.T @ section["D"] @ bending * weight
            coupling_local += (
                membrane.T @ section["B"] @ bending
                + bending.T @ section["B"].T @ membrane
            ) * weight
            unrelaxed_shear += shear.T @ section["H"] @ shear * weight
        bending_sum = float(sum(bending_local[i, i] for i in _ROTATIONAL_INDICES))
        shear_sum = float(
            sum(unrelaxed_shear[i, i] for i in _ROTATIONAL_INDICES)
        )
        if bending_sum <= 0.0 or shear_sum <= 0.0:
            raise ValueError("S3 V2D native MIN3 relaxation traces must be positive")
        psi_hat = bending_sum / shear_sum
        phi_squared = CBMIN3 * psi_hat / (1.0 + CBMIN3 * psi_hat)
        if not math.isfinite(phi_squared) or not 0.0 < phi_squared <= 1.0:
            raise ValueError("S3 V2D native MIN3 relaxation is invalid")
        shear_local = phi_squared * unrelaxed_shear
        barycentric_mass = (area / 12.0) * np.asarray(
            ((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0)),
            dtype=np.float64,
        )
        drill_scale = float(
            raw_section.get(
                "drill_scale", _invariant_drill_scale(raw_section["A"])
            )
        )
        constraint = operators["C"]
        pl_local = drill_scale * constraint.T @ barycentric_mass @ constraint
        physical_local = membrane_local + coupling_local + bending_local + shear_local
        total_local = physical_local + pl_local
        transform = np.asarray(geometry["local_from_external"], dtype=np.float64)
        membrane = self._globalize(membrane_local, transform)
        coupling = self._globalize(coupling_local, transform)
        bending = self._globalize(bending_local, transform)
        shear = self._globalize(shear_local, transform)
        physical = self._globalize(physical_local, transform)
        pl = self._globalize(pl_local, transform)
        total = self._globalize(total_local, transform)
        return {
            "membrane": membrane,
            "membrane_bending_coupling": coupling,
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
            "phi_squared": phi_squared,
            "drill_scale": drill_scale,
            "quadrature_authority_id": QUADRATURE_AUTHORITY_ID,
            "pl_completion_policy_id": PL_COMPLETION_POLICY_ID,
            "native_section_policy_id": (
                NATIVE_SECTION_POLICY_ID
                if self.shell_section is not None
                else "S3_V2D_LAYERED_ELASTIC_BASELINE_V1"
            ),
            "relaxation_authority_sha256": RELAXATION_AUTHORITY_SHA256,
            "director_polarity": int(self.director_polarity),
            "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
            "reference_surface_offset": float(self.reference_surface_offset),
            "reference_surface_offset_policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
            "reference_surface_strain_transform": (
                self._reference_surface_strain_transform()
            ),
            "reference_surface_strain_transform_id": (
                REFERENCE_SURFACE_STRAIN_TRANSFORM_ID
            ),
        }

    def compute_stiffness_components(
        self, mesh: FEMesh, material: Material
    ) -> Mapping[str, Any]:
        self._validate_configuration()
        if (
            self.shell_section is None
            and self._is_v2c_elastic_overlap(material)
            and self.director_polarity == 1
            and self.reference_surface_offset == 0.0
        ):
            return StrictFlatLinearE4PLS3V2CShellElement.compute_stiffness_components(
                self, mesh, material
            )
        return self._native_generalized_components(mesh, material)

    def compute_stiffness_matrix(self, mesh: FEMesh, material: Material) -> np.ndarray:
        return np.asarray(
            self.compute_stiffness_components(mesh, material)["total"],
            dtype=np.float64,
        ).copy()

    def compute_internal_forces(
        self, mesh: FEMesh, displacements: np.ndarray, material: Material
    ) -> np.ndarray:
        vector = self._get_element_displacements(mesh, displacements)
        if vector.shape != (18,) or not np.all(np.isfinite(vector)):
            raise ValueError("S3 V2D displacement must resolve to a finite 18-vector")
        return self.compute_stiffness_matrix(mesh, material) @ vector

    def compute_variational_resultants(
        self, mesh: FEMesh, displacements: np.ndarray, material: Material
    ) -> Dict[str, Any]:
        if (
            self.shell_section is None
            and self._is_v2c_elastic_overlap(material)
            and self.director_polarity == 1
            and self.reference_surface_offset == 0.0
        ):
            return StrictFlatLinearE4PLS3V2CShellElement.compute_variational_resultants(
                self, mesh, displacements, material
            )
        geometry = self._geometry(mesh)
        raw_section = (
            self._native_section(geometry)
            if self.shell_section is not None
            else self._elastic_baseline_constitutive(material)
        )
        section = self._director_adjusted_section(raw_section)
        operators = self._operators(geometry, raw_section)
        membrane_operators, curvature_operators, shear_operators = (
            self._effective_station_operators(operators)
        )
        phi_squared = float(self.compute_stiffness_components(mesh, material)["phi_squared"])
        vector = self._get_element_displacements(mesh, displacements)
        local_vector = np.asarray(geometry["local_from_external"]) @ vector
        epsilon = np.einsum("gij,j->gi", membrane_operators, local_vector)
        kappa = np.einsum("gij,j->gi", curvature_operators, local_vector)
        gamma = np.einsum("gij,j->gi", shear_operators, local_vector)
        membrane_resultants = epsilon @ section["A"].T + kappa @ section["B"].T
        bending_resultants = epsilon @ section["B"] + kappa @ section["D"].T
        shear_resultants = phi_squared * gamma @ section["H"].T
        order = tuple(int(value) for value in geometry["internal_order"])

        def external_order(values: np.ndarray) -> np.ndarray:
            made = np.empty_like(values)
            for internal, external in enumerate(order):
                made[external] = values[internal]
            return made

        barycentric = np.column_stack(
            (
                1.0 - HAMMER_POINTS[:, 0] - HAMMER_POINTS[:, 1],
                HAMMER_POINTS[:, 0],
                HAMMER_POINTS[:, 1],
            )
        )
        return {
            "recovery_scope": RESULTANT_POLICY_ID,
            "qualified_recovery": False,
            "physical_stress_available": False,
            "numerical_fields_excluded": True,
            "membrane_strain": external_order(epsilon),
            "curvature": external_order(kappa),
            "transverse_shear_strain": external_order(gamma),
            "membrane_resultants": external_order(membrane_resultants),
            "bending_resultants": external_order(bending_resultants),
            "transverse_shear_resultants": external_order(shear_resultants),
            "min3_relaxation_phi_squared": phi_squared,
            "hammer_points": np.asarray(HAMMER_POINTS, dtype=np.float64).copy(),
            "external_barycentric_coordinates": barycentric,
            "physical_station_coordinates": barycentric
            @ np.asarray(geometry["external_coordinates"], dtype=np.float64),
            "internal_order": order,
            "physical_weights": np.full(3, float(geometry["area"]) / 3.0),
            "frame": np.asarray(geometry["frame"], dtype=np.float64).copy(),
            "resultant_policy_id": RESULTANT_POLICY_ID,
            "native_section_policy_id": (
                NATIVE_SECTION_POLICY_ID
                if self.shell_section is not None
                else "S3_V2D_LAYERED_ELASTIC_BASELINE_V1"
            ),
            "director_polarity": int(self.director_polarity),
            "reference_surface_offset": float(self.reference_surface_offset),
            "section_origin_offset_from_reference": (
                self.material_mid_surface_offset_from_reference
            ),
            "physical_bottom_offset_from_reference": (
                self.material_surface_offset_from_reference(-1.0)
            ),
            "physical_top_offset_from_reference": (
                self.material_surface_offset_from_reference(1.0)
            ),
        }

    def compute_stresses(
        self,
        mesh: FEMesh,
        displacements: np.ndarray,
        material: Material,
        return_global: bool = False,
    ) -> Dict[str, Any]:
        recovered = self.compute_variational_resultants(
            mesh,
            displacements,
            material,
        )
        result: Dict[str, Any] = {
            **recovered,
            "formulation_id": FORMULATION_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "recovery_scope": "PHYSICAL_LOCAL_RESULTANTS_ONLY",
        }
        if self.shell_section is None:
            thickness = float(self.thickness)
            if not math.isfinite(thickness) or thickness <= np.finfo(np.float64).tiny:
                raise ValueError("S3 V2D recovery thickness must be positive")
            membrane_stress = np.asarray(
                recovered["membrane_resultants"], dtype=np.float64
            ).reshape(3, 3) / thickness
            bending_stress = 6.0 * np.asarray(
                recovered["bending_resultants"], dtype=np.float64
            ).reshape(3, 3) / (thickness * thickness)
            owner_shear = np.asarray(
                recovered["transverse_shear_resultants"], dtype=np.float64
            ).reshape(3, 2) / thickness
            result.update(
                {
                    "membrane_xx": membrane_stress[:, 0].copy(),
                    "membrane_yy": membrane_stress[:, 1].copy(),
                    "membrane_xy": membrane_stress[:, 2].copy(),
                    "bending_xx": bending_stress[:, 0].copy(),
                    "bending_yy": bending_stress[:, 1].copy(),
                    "bending_xy": bending_stress[:, 2].copy(),
                    "shear_xz": owner_shear[:, 0].copy(),
                    "shear_yz": owner_shear[:, 1].copy(),
                }
            )
            top = membrane_stress + bending_stress
            bottom = membrane_stress - bending_stress
            shear_square = owner_shear[:, 0] ** 2 + owner_shear[:, 1] ** 2

            def von_mises(in_plane: np.ndarray) -> np.ndarray:
                return np.sqrt(
                    in_plane[:, 0] ** 2
                    - in_plane[:, 0] * in_plane[:, 1]
                    + in_plane[:, 1] ** 2
                    + 3.0 * (in_plane[:, 2] ** 2 + shear_square)
                )

            result["von_mises"] = np.maximum(von_mises(top), von_mises(bottom))
            result["equivalent_stress"] = result["von_mises"].copy()
            result["equivalent_stress_measure"] = "von_mises"
            result["physical_stress_available"] = True
            result["physical_layer_recovery_available"] = True
            result["recovery_scope"] = "PHYSICAL_LOCAL_SURFACE_STRESSES"
        else:
            result["physical_stress_available"] = False
            result["physical_layer_recovery_available"] = False
        if not return_global:
            return result

        frame = np.asarray(recovered["frame"], dtype=np.float64).reshape(3, 3)
        membrane = np.asarray(
            recovered["membrane_resultants"], dtype=np.float64
        ).reshape(3, 3)
        bending = np.asarray(
            recovered["bending_resultants"], dtype=np.float64
        ).reshape(3, 3)
        shear_resultant = np.asarray(
            recovered["transverse_shear_resultants"], dtype=np.float64
        ).reshape(3, 2)

        def tensor_rows(rows: np.ndarray) -> np.ndarray:
            tensors = np.zeros((3, 3, 3), dtype=np.float64)
            tensors[:, 0, 0] = rows[:, 0]
            tensors[:, 1, 1] = rows[:, 1]
            tensors[:, 0, 1] = rows[:, 2]
            tensors[:, 1, 0] = rows[:, 2]
            return np.asarray(
                [frame @ tensor @ frame.T for tensor in tensors],
                dtype=np.float64,
            )

        reference_coordinates = self._coordinates(mesh)
        result.update(
            {
                "global_membrane_resultant_tensors": tensor_rows(membrane),
                "global_bending_resultant_tensors": tensor_rows(bending),
                "global_transverse_shear_resultants": (
                    shear_resultant[:, 0, None] * frame[:, 0][None, :]
                    + shear_resultant[:, 1, None] * frame[:, 1][None, :]
                ),
                "_reference_surface_recovery_coordinates": (
                    reference_coordinates.copy()
                ),
                "_recovery_coordinates": (
                    reference_coordinates
                    - float(self.reference_surface_offset)
                    * self.physical_reference_director[None, :]
                ),
                "_recovery_stress_frame": frame.copy(),
            }
        )
        if self.shell_section is not None:
            result["recovery_scope"] = "PHYSICAL_GLOBAL_RESULTANTS_ONLY"
            result["physical_stress_available"] = False
            result["physical_layer_recovery_available"] = False
            return result

        for surface, sign in (("bot", -1.0), ("top", 1.0)):
            in_plane = membrane_stress + sign * bending_stress
            local_tensors = np.zeros((3, 3, 3), dtype=np.float64)
            local_tensors[:, 0, 0] = in_plane[:, 0]
            local_tensors[:, 1, 1] = in_plane[:, 1]
            local_tensors[:, 0, 1] = in_plane[:, 2]
            local_tensors[:, 1, 0] = in_plane[:, 2]
            local_tensors[:, 0, 2] = owner_shear[:, 0]
            local_tensors[:, 2, 0] = owner_shear[:, 0]
            local_tensors[:, 1, 2] = owner_shear[:, 1]
            local_tensors[:, 2, 1] = owner_shear[:, 1]
            global_tensors = np.asarray(
                [frame @ tensor @ frame.T for tensor in local_tensors],
                dtype=np.float64,
            )
            for first, second, label in (
                (0, 0, "xx"),
                (1, 1, "yy"),
                (2, 2, "zz"),
                (0, 1, "xy"),
                (1, 2, "yz"),
                (0, 2, "xz"),
            ):
                result[f"local_{label}_{surface}"] = local_tensors[
                    :, first, second
                ].copy()
                result[f"global_{label}_{surface}"] = global_tensors[
                    :, first, second
                ].copy()
        result["recovery_scope"] = "PHYSICAL_GLOBAL_SURFACE_TENSORS"
        result["physical_stress_available"] = True
        result["physical_layer_recovery_available"] = True
        return result

    def compute_dead_transverse_pressure_load(self, mesh: FEMesh, pressure: Any) -> np.ndarray:
        return StrictFlatLinearE4PLS3V2CShellElement.compute_dead_transverse_pressure_load(
            self, mesh, pressure
        )

    def compute_native_dead_pressure_load(
        self, mesh: FEMesh, pressure: Any
    ) -> np.ndarray:
        return self.compute_dead_transverse_pressure_load(mesh, pressure)

    # Shared (not duplicated) native pressure virtual-work density; V2D delegates its pressure routes to V2C by
    # unbound call, so the same static helper must be bound on this class.
    _pressure_virtual_work_density_local = staticmethod(
        StrictFlatLinearE4PLS3V2CShellElement._pressure_virtual_work_density_local
    )

    def compute_native_pressure_density(self, mesh: FEMesh, points: Any) -> Tuple[np.ndarray, float]:
        return StrictFlatLinearE4PLS3V2CShellElement.compute_native_pressure_density(
            self, mesh, points
        )

    @staticmethod
    def compute_native_surface_shape_functions(
        xi: Any, eta: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = _real_scalar(xi, "surface_xi")
        s = _real_scalar(eta, "surface_eta")
        return (
            np.asarray((1.0 - r - s, r, s), dtype=np.float64),
            np.asarray((-1.0, 1.0, 0.0), dtype=np.float64),
            np.asarray((-1.0, 0.0, 1.0), dtype=np.float64),
        )

    def sheet_area_orientation_sign(self, mesh: FEMesh) -> float:
        coordinates = self._coordinates(mesh)
        raw = np.cross(
            coordinates[1] - coordinates[0],
            coordinates[2] - coordinates[0],
        )
        signed = float(raw @ np.asarray(self.reference_normal, dtype=np.float64))
        scale = float(np.linalg.norm(raw))
        if (
            not math.isfinite(signed)
            or not math.isfinite(scale)
            or scale <= np.finfo(np.float64).tiny
            or abs(signed) <= _NORMAL_TOLERANCE * scale
        ):
            raise ValueError("S3 V2D sheet orientation is unresolved")
        return 1.0 if signed > 0.0 else -1.0

    @staticmethod
    def _pressure_scalar(pressure: Any) -> float:
        if isinstance(pressure, (bool, np.bool_)) or not isinstance(
            pressure, (int, float, np.integer, np.floating)
        ):
            raise NativeParityCapabilityError(
                "S3 V2D pressure authority admits a uniform scalar only"
            )
        return _real_scalar(pressure, "pressure")

    def compute_native_current_pressure_load(
        self,
        mesh: FEMesh,
        pressure: Any,
        current_coordinates: Any,
    ) -> np.ndarray:
        value = self._pressure_scalar(pressure)
        coordinates = np.asarray(current_coordinates, dtype=np.float64)
        if coordinates.shape != (3, 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("S3 V2D current pressure coordinates must be finite (3,3)")
        orientation = self.sheet_area_orientation_sign(mesh)
        load = np.zeros(18, dtype=np.float64)
        for (xi, eta), weight in zip(HAMMER_POINTS, HAMMER_REFERENCE_WEIGHTS):
            shape, derivative_xi, derivative_eta = (
                self.compute_native_surface_shape_functions(xi, eta)
            )
            tangent_xi = coordinates.T @ derivative_xi
            tangent_eta = coordinates.T @ derivative_eta
            area_vector = orientation * np.cross(tangent_xi, tangent_eta)
            if float(np.linalg.norm(area_vector)) <= np.finfo(np.float64).tiny:
                raise ValueError("S3 V2D current pressure surface is degenerate")
            for node in range(3):
                load[6 * node : 6 * node + 3] += (
                    float(shape[node])
                    * value
                    * area_vector
                    * float(weight)
                )
        return load

    def compute_native_current_pressure_tangent(
        self,
        mesh: FEMesh,
        pressure: Any,
        current_coordinates: Any,
    ) -> np.ndarray:
        value = self._pressure_scalar(pressure)
        coordinates = np.asarray(current_coordinates, dtype=np.float64)
        if coordinates.shape != (3, 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("S3 V2D current pressure coordinates must be finite (3,3)")
        orientation = self.sheet_area_orientation_sign(mesh)
        tangent = np.zeros((18, 18), dtype=np.float64)

        def skew(vector: np.ndarray) -> np.ndarray:
            x, y, z = np.asarray(vector, dtype=np.float64)
            return np.asarray(
                ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
                dtype=np.float64,
            )

        for (xi, eta), weight in zip(HAMMER_POINTS, HAMMER_REFERENCE_WEIGHTS):
            shape, derivative_xi, derivative_eta = (
                self.compute_native_surface_shape_functions(xi, eta)
            )
            tangent_xi = coordinates.T @ derivative_xi
            tangent_eta = coordinates.T @ derivative_eta
            if float(np.linalg.norm(np.cross(tangent_xi, tangent_eta))) <= np.finfo(
                np.float64
            ).tiny:
                raise ValueError("S3 V2D current pressure surface is degenerate")
            skew_xi = skew(tangent_xi)
            skew_eta = skew(tangent_eta)
            scale = orientation * value * float(weight)
            for row_node in range(3):
                row = slice(6 * row_node, 6 * row_node + 3)
                for column_node in range(3):
                    column = slice(6 * column_node, 6 * column_node + 3)
                    tangent[row, column] += (
                        scale
                        * float(shape[row_node])
                        * (
                            -float(derivative_xi[column_node]) * skew_eta
                            + float(derivative_eta[column_node]) * skew_xi
                        )
                    )
        return tangent

    def compute_mass_matrix(self, mesh: FEMesh, material: Material, **kwargs: Any) -> np.ndarray:
        if kwargs:
            raise NativeParityCapabilityError("S3 V2D mass gate accepts no options")
        geometry = self._geometry(mesh)
        section = self._generalized_section_in_frame(geometry["frame"])
        if section is None or section.mass_per_area is None:
            density = _real_scalar(getattr(material, "density", None), "density")
            mass_per_area = density * self.thickness
        else:
            mass_per_area = _real_scalar(section.mass_per_area, "mass_per_area")
        if mass_per_area <= 0.0:
            raise ValueError("S3 V2D mass_per_area must be positive")
        nodal_mass = mass_per_area * float(geometry["area"]) / 3.0
        local = np.zeros((18, 18), dtype=np.float64)
        for node in range(3):
            for axis in range(3):
                local[6 * node + axis, 6 * node + axis] = nodal_mass
        transform = np.asarray(geometry["local_from_external"], dtype=np.float64)
        return self._globalize(local, transform)

    def compute_mass_components(
        self,
        mesh: FEMesh,
        material: Material,
    ) -> Dict[str, Any]:
        """Return the exact zero-rotation mass witness used by descriptor solvers."""

        global_mass = self.compute_mass_matrix(mesh, material)
        geometry = self._geometry(mesh)
        section = self._generalized_section_in_frame(geometry["frame"])
        if section is None:
            density = _real_scalar(getattr(material, "density", None), "density")
            mass_per_area = density * self.thickness
        elif section.mass_per_area is None:
            density = _real_scalar(getattr(material, "density", None), "density")
            mass_per_area = density * self.thickness
        else:
            mass_per_area = _real_scalar(section.mass_per_area, "mass_per_area")
        if mass_per_area <= 0.0:
            raise ValueError("S3 V2D mass_per_area must be positive")
        nodal_mass = mass_per_area * float(geometry["area"]) / 3.0
        local = np.zeros((18, 18), dtype=np.float64)
        for node in range(3):
            for axis in range(3):
                local[6 * node + axis, 6 * node + axis] = nodal_mass
        return {
            "condensed_local": local,
            "condensed_rank": 9,
            "formulation_id": FORMULATION_ID,
            "global": np.asarray(global_mass, dtype=np.float64),
            "mass_per_area": mass_per_area,
            "mass_policy_id": MASS_POLICY_ID,
            "rotary_inertia_per_area": 0.0,
            "zero_drill_inertia": True,
            "zero_rotational_inertia": True,
        }

    def dynamic_algebraic_directions(
        self,
        mesh: FEMesh,
        material: Material,
    ) -> np.ndarray:
        """Return the nine orthonormal global rotation directions with zero mass."""

        self.compute_mass_components(mesh, material)
        directions = np.zeros((18, 9), dtype=np.float64)
        for node in range(3):
            for axis in range(3):
                directions[6 * node + 3 + axis, 3 * node + axis] = 1.0
        return directions

    def compute_geometric_stiffness_matrix(
        self,
        mesh: FEMesh,
        material: Material,
        state: Optional[Any] = None,
    ) -> np.ndarray:
        """Assemble the frozen V2C uniform membrane initial-stress operator."""

        self._validate_configuration()
        geometry = self._geometry(mesh)
        # Evaluate the formulation-owned section once so material and section
        # authority are validated before consuming the prestress state.
        self.compute_stiffness_components(mesh, material)
        if state is None:
            return np.zeros((18, 18), dtype=np.float64)
        if not isinstance(state, Mapping):
            raise TypeError("S3 V2D geometric state must be a mapping")
        allowed = {
            "membrane_compression",
            "bending_compression",
            "stress_second_moment",
        }
        if not set(state) <= allowed or "membrane_compression" not in state:
            raise NativeParityCapabilityError(
                "S3 V2D reference buckling admits only the frozen uniform "
                "membrane-compression policy"
            )
        compression = np.asarray(state["membrane_compression"], dtype=np.float64)
        if compression.shape != (3,) or not np.all(np.isfinite(compression)):
            raise ValueError("S3 V2D membrane_compression must be a finite 3-vector")
        for optional in ("bending_compression", "stress_second_moment"):
            if optional not in state:
                continue
            values = np.asarray(state[optional], dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"S3 V2D {optional} must be a finite 3-vector")
            if np.any(values):
                raise NativeParityCapabilityError(
                    "S3 V2D source authority does not admit nonzero " + optional
                )
        nx, ny, nxy = (float(value) for value in compression)
        stress = np.asarray(((nx, nxy), (nxy, ny)), dtype=np.float64)
        gradients = np.asarray(geometry["shape_gradients"], dtype=np.float64)
        scalar = float(geometry["area"]) * gradients @ stress @ gradients.T
        local = np.zeros((18, 18), dtype=np.float64)
        for first in range(3):
            for second in range(3):
                for axis in range(3):
                    local[6 * first + axis, 6 * second + axis] = scalar[first, second]
        transform = np.asarray(geometry["local_from_external"], dtype=np.float64)
        made = self._globalize(local, transform)
        return 0.5 * (made + made.T)

    @staticmethod
    def _external_station_order(
        values: np.ndarray, order: Tuple[int, int, int]
    ) -> np.ndarray:
        made = np.empty_like(values)
        for internal, external in enumerate(order):
            made[external] = values[internal]
        return made

    @staticmethod
    def _internal_station_order(
        values: np.ndarray, order: Tuple[int, int, int]
    ) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)[np.asarray(order, dtype=np.intp)]

    def _native_trial_response(
        self,
        mesh: FEMesh,
        material: Material,
        total_u: np.ndarray,
        committed: Mapping[str, Any],
        num_layers: int,
        *,
        von_karman: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        geometry = self._geometry(mesh)
        raw_baseline = (
            self._native_section(geometry)
            if self.shell_section is not None
            else self._elastic_baseline_constitutive(material)
        )
        baseline = self._director_adjusted_section(raw_baseline)
        operators = self._operators(geometry, raw_baseline)
        membrane_operators, curvature_operators, shear_operators = (
            self._effective_station_operators(operators)
        )
        components = self.compute_stiffness_components(mesh, material)
        phi_squared = float(components["phi_squared"])
        transform = np.asarray(geometry["local_from_external"], dtype=np.float64)
        local_u = transform @ total_u
        epsilon = np.einsum("gij,j->gi", membrane_operators, local_u)
        kappa = np.einsum("gij,j->gi", curvature_operators, local_u)
        gamma = np.einsum("gij,j->gi", shear_operators, local_u)
        if von_karman:
            # Reference-surface transverse displacement is affine on this
            # triangle.  Engineering shear strain contains wx*wy (not half).
            gradient = np.zeros((2, 18), dtype=np.float64)
            gradient[:, 2::6] = np.asarray(geometry["shape_gradients"]).T
            wx, wy = gradient @ local_u
            epsilon = epsilon + np.asarray((0.5 * wx * wx, 0.5 * wy * wy, wx * wy))
            membrane_operators = membrane_operators.copy()
            membrane_operators[:, 0, :] += wx * gradient[0]
            membrane_operators[:, 1, :] += wy * gradient[1]
            membrane_operators[:, 2, :] += wy * gradient[0] + wx * gradient[1]
        generalized_strain = np.concatenate((epsilon, kappa, gamma), axis=1)
        order = tuple(int(value) for value in geometry["internal_order"])
        initial_prestrain = self._internal_station_order(
            np.asarray(
                committed["initial_generalized_prestrain"], dtype=np.float64
            ),
            order,
        )
        initial_resultant = self._internal_station_order(
            np.asarray(
                committed["initial_generalized_resultant"], dtype=np.float64
            ),
            order,
        )
        area_weight = float(geometry["area"]) / 3.0
        force_local = np.zeros(18, dtype=np.float64)
        tangent_local = np.zeros((18, 18), dtype=np.float64)
        station_resultant = np.zeros((3, 8), dtype=np.float64)

        trial = dict(committed)
        if self.shell_section is not None:
            constitutive = self._generalized_constitutive(
                baseline, phi_squared
            )
            for station in range(3):
                effective = generalized_strain[station] - initial_prestrain[station]
                resultant = constitutive @ effective + initial_resultant[station]
                station_resultant[station] = resultant
                combined = np.vstack(
                    (
                        membrane_operators[station],
                        curvature_operators[station],
                        shear_operators[station],
                    )
                )
                force_local += combined.T @ resultant * area_weight
                tangent_local += combined.T @ constitutive @ combined * area_weight
            trial["plastic_strain"] = np.zeros((0, 3), dtype=np.float64)
            trial["alpha"] = np.zeros(0, dtype=np.float64)
            trial["layer_strain"] = np.zeros((0, 3), dtype=np.float64)
            trial["layer_stress"] = np.zeros((0, 3), dtype=np.float64)
        else:
            layer_z, layer_weight = qualified_s3_lobatto_layers(
                int(num_layers), self.thickness
            )
            plastic = np.asarray(
                committed["plastic_strain"], dtype=np.float64
            ).reshape(3, int(num_layers), 3)
            alpha = np.asarray(committed["alpha"], dtype=np.float64).reshape(
                3, int(num_layers)
            )
            material_strain = np.empty((3, int(num_layers), 3), dtype=np.float64)
            for station in range(3):
                for layer, z_value in enumerate(layer_z):
                    material_strain[station, layer] = (
                        epsilon[station]
                        + float(z_value) * kappa[station]
                        - initial_prestrain[station, :3]
                        - float(z_value) * initial_prestrain[station, 3:6]
                    )
            stress, algorithmic, plastic_new, alpha_new = plane_stress_return_map(
                material_strain.reshape(-1, 3),
                plastic.reshape(-1, 3),
                alpha.reshape(-1),
                float(baseline["E"]),
                float(baseline["nu"]),
                getattr(material, "hardening_curve", None),
                compute_tangent=True,
            )
            stress = np.asarray(stress, dtype=np.float64).reshape(
                3, int(num_layers), 3
            )
            algorithmic = np.asarray(algorithmic, dtype=np.float64).reshape(
                3, int(num_layers), 3, 3
            )
            for station in range(3):
                membrane = np.zeros(3, dtype=np.float64)
                bending = np.zeros(3, dtype=np.float64)
                tangent_mb = np.zeros((6, 6), dtype=np.float64)
                for layer, (z_value, weight_value) in enumerate(
                    zip(layer_z, layer_weight)
                ):
                    z = float(z_value)
                    weight = float(weight_value)
                    membrane += weight * stress[station, layer]
                    bending += weight * z * stress[station, layer]
                    layer_map = np.hstack(
                        (np.eye(3, dtype=np.float64), z * np.eye(3, dtype=np.float64))
                    )
                    tangent_mb += (
                        layer_map.T
                        @ algorithmic[station, layer]
                        @ layer_map
                        * weight
                    )
                shear = (
                    phi_squared
                    * baseline["H"]
                    @ (gamma[station] - initial_prestrain[station, 6:])
                )
                resultant = np.concatenate((membrane, bending, shear))
                resultant += initial_resultant[station]
                station_resultant[station] = resultant
                membrane_bending_operator = np.vstack(
                    (
                        membrane_operators[station],
                        curvature_operators[station],
                    )
                )
                force_local += (
                    membrane_bending_operator.T @ resultant[:6]
                    + shear_operators[station].T @ resultant[6:]
                ) * area_weight
                tangent_local += (
                    membrane_bending_operator.T
                    @ tangent_mb
                    @ membrane_bending_operator
                    + shear_operators[station].T
                    @ (phi_squared * baseline["H"])
                    @ shear_operators[station]
                ) * area_weight
            trial["plastic_strain"] = np.asarray(
                plastic_new, dtype=np.float64
            ).reshape(-1, 3)
            trial["alpha"] = np.asarray(alpha_new, dtype=np.float64).reshape(-1)
            trial["layer_strain"] = material_strain.reshape(-1, 3)
            trial["layer_stress"] = stress.reshape(-1, 3)

        if von_karman:
            # The second variation includes initial resultants and returned
            # material stress.  The existing return map supplies dN/dstrain.
            for resultant in station_resultant:
                membrane_stress = np.asarray(
                    ((resultant[0], resultant[2]), (resultant[2], resultant[1]))
                )
                tangent_local += gradient.T @ membrane_stress @ gradient * area_weight

        physical_force = transform.T @ force_local
        physical_tangent = self._globalize(tangent_local, transform)
        pl = np.asarray(components["pl"], dtype=np.float64)
        total_force = physical_force + pl @ total_u
        total_tangent = physical_tangent + pl
        total_tangent = 0.5 * (total_tangent + total_tangent.T)
        if not np.all(np.isfinite(total_force)) or not np.all(np.isfinite(total_tangent)):
            raise ValueError("S3 V2D native trial response is nonfinite")
        trial["committed_total_u"] = total_u.copy()
        trial["committed_constitutive_u"] = total_u.copy()
        trial["solver_kinematics"] = "VON_KARMAN" if von_karman else "ELEMENT_LOCAL"
        if von_karman:
            trial["experimental_von_karman_law"] = "S3_V2D_VK_R1"
        trial["station_generalized_strain"] = self._external_station_order(
            generalized_strain, order
        )
        trial["station_generalized_resultant"] = self._external_station_order(
            station_resultant, order
        )
        sealed = seal_v2d_state(trial)
        return total_force, total_tangent, sealed

    def von_karman_directional_curvature(
        self,
        mesh: FEMesh,
        material: Material,
        total_u: np.ndarray,
        direction: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Exact F''(u)[v,v] of the von Karman force, or None if not elastic.

        The force is cubic in u (strain quadratic, section linear), so with
        J(u)=B+P(u) the combined operator and e2(v)=(vx^2/2, vy^2/2, vx*vy):
        F(u+hv) = sum_s w J(u+hv)^T C (e(u+hv)-e0) and
        F''[v,v] = 2 w sum_s [ P(v)^T C J(u) v + J(u)^T C e2(v) ].
        Plastic layered sections return None (no closed form).
        """
        self._validate_configuration()
        if self.shell_section is None and not self._is_v2c_elastic_overlap(material):
            return None
        geometry = self._geometry(mesh)
        raw_baseline = (
            self._native_section(geometry)
            if self.shell_section is not None
            else self._elastic_baseline_constitutive(material)
        )
        baseline = self._director_adjusted_section(raw_baseline)
        operators = self._operators(geometry, raw_baseline)
        membrane, curvature, shear = self._effective_station_operators(operators)
        phi_squared = float(self.compute_stiffness_components(mesh, material)["phi_squared"])
        constitutive = self._generalized_constitutive(baseline, phi_squared)
        transform = np.asarray(geometry["local_from_external"], dtype=np.float64)
        local_u = transform @ np.asarray(total_u, dtype=np.float64)
        local_v = transform @ np.asarray(direction, dtype=np.float64)
        gradient = np.zeros((2, 18), dtype=np.float64)
        gradient[:, 2::6] = np.asarray(geometry["shape_gradients"]).T
        wx, wy = gradient @ local_u
        vx, vy = gradient @ local_v
        vk_operator = np.zeros((8, 18), dtype=np.float64)
        vk_operator[0] = wx * gradient[0]
        vk_operator[1] = wy * gradient[1]
        vk_operator[2] = wy * gradient[0] + wx * gradient[1]
        direction_operator = np.zeros((8, 18), dtype=np.float64)
        direction_operator[0] = vx * gradient[0]
        direction_operator[1] = vy * gradient[1]
        direction_operator[2] = vy * gradient[0] + vx * gradient[1]
        second_strain = np.zeros(8, dtype=np.float64)
        second_strain[:3] = (0.5 * vx * vx, 0.5 * vy * vy, vx * vy)
        area_weight = float(geometry["area"]) / 3.0
        local = np.zeros(18, dtype=np.float64)
        for station in range(3):
            combined = np.vstack((membrane[station], curvature[station], shear[station]))
            combined = combined + vk_operator
            local += (
                direction_operator.T @ (constitutive @ (combined @ local_v))
                + combined.T @ (constitutive @ second_strain)
            )
        return transform.T @ (2.0 * area_weight * local)

    def compute_nonlinear_response(
        self,
        mesh: FEMesh,
        material: Material,
        u_elem: np.ndarray,
        state: Optional[Mapping[str, Any]] = None,
        num_layers: int = 5,
        tangent: bool = True,
        *,
        von_karman: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        self._validate_configuration()
        try:
            total_u = np.asarray(u_elem, dtype=np.float64).reshape(18)
        except (TypeError, ValueError) as exc:
            raise ValueError("S3 V2D nonlinear displacement must be a finite 18-vector") from exc
        if not np.all(np.isfinite(total_u)):
            raise ValueError("S3 V2D nonlinear displacement must be a finite 18-vector")
        layers = self._validated_num_layers(num_layers)
        committed = (
            self.init_model_bound_nonlinear_state(mesh, material, layers)
            if state is None
            else self.validate_model_bound_nonlinear_state(
                mesh, material, state, layers
            )
        )
        if not von_karman and "experimental_von_karman_law" in committed:
            raise V2DStateError("V2D experimental von Karman history cannot change kinematics")
        if von_karman and committed["solver_kinematics"] not in {"UNBOUND", "VON_KARMAN"}:
            raise V2DStateError("V2D von Karman requires virgin or matching experimental history")
        force, made_tangent, trial = self._native_trial_response(
            mesh, material, total_u, committed, layers, von_karman=von_karman
        )
        validated = self.validate_model_bound_nonlinear_state(
            mesh,
            material,
            trial,
            layers,
            expected_committed_total_u=total_u,
        )
        return force, made_tangent if bool(tangent) else None, validated

    def compute_noncurrent_deleted_residual_operator(
        self,
        mesh: FEMesh,
        material: Material,
        current_u_elem: Any,
        frozen_state: Mapping[str, Any],
        num_layers: int = 5,
        *,
        tangent: bool = True,
        von_karman: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate from frozen parent history and discard the trial state."""

        core = self.validate_model_bound_nonlinear_state(
            mesh, material, frozen_state, num_layers
        )
        force, stiffness, _discarded = self.compute_nonlinear_response(
            mesh,
            material,
            np.asarray(current_u_elem, dtype=np.float64),
            core,
            num_layers,
            bool(tangent),
            von_karman=von_karman,
        )
        return force, stiffness

    def init_nonlinear_state(self, num_layers: int) -> Dict[str, Any]:
        del num_layers
        raise NativeParityCapabilityError(
            "S3 V2D state initialization is model-bound; use "
            "init_model_bound_nonlinear_state(mesh, material, num_layers)"
        )

    def to_dict(self) -> Dict[str, Any]:
        self._validate_configuration()
        return {
            "element_id": int(self.element_id),
            "formulation_id": FORMULATION_ID,
            "formulation_schema": FORMULATION_SCHEMA,
            "implementation_id": IMPLEMENTATION_ID,
            "material_name": self.material_name,
            "node_ids": [int(value) for value in self.node_ids],
            "reference_normal": np.asarray(self.reference_normal).tolist(),
            "director_polarity": int(self.director_polarity),
            "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
            "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
            "reference_surface_offset": float(self.reference_surface_offset),
            "reference_surface_offset_policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
            "reference_surface_strain_transform_id": (
                REFERENCE_SURFACE_STRAIN_TRANSFORM_ID
            ),
            "material_direction": (
                None
                if self.material_direction is None
                else np.asarray(self.material_direction).tolist()
            ),
            "material_angle_deg": float(self.material_angle_deg),
            "shell_section": (
                None if self.shell_section is None else self.shell_section.to_dict()
            ),
            "selector": SELECTOR,
            "serialization_policy_id": SERIALIZATION_POLICY_ID,
            "native_state_schema_id": NATIVE_STATE_SCHEMA_ID,
            "native_state_layout_id": NATIVE_STATE_LAYOUT_ID,
            "corotational_policy_id": COROTATIONAL_POLICY_ID,
            "pressure_surface_policy_id": PRESSURE_SURFACE_POLICY_ID,
            "activity_policy_id": ACTIVITY_POLICY_ID,
            "qualified_batch_policy_id": BATCH_POLICY_ID,
            "dynamic_algebraic_policy_id": DYNAMIC_ALGEBRAIC_POLICY_ID,
            "geometric_stiffness_policy_id": GEOMETRIC_STIFFNESS_POLICY_ID,
            "mass_policy_id": MASS_POLICY_ID,
            "source_selection_sha256": SOURCE_SELECTION_SHA256,
            "thickness": float(self.thickness),
            "type": type(self).__name__,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeParityE4PLS3V2DShellElement":
        if cls is not NativeParityE4PLS3V2DShellElement:
            raise NativeParityCapabilityError("S3 V2D deserialization requires exact class")
        if not isinstance(payload, Mapping):
            raise TypeError("S3 V2D serialized payload must be a mapping")
        data = dict(payload)
        required = {
            "element_id",
            "formulation_id",
            "formulation_schema",
            "implementation_id",
            "material_name",
            "node_ids",
            "reference_normal",
            "director_polarity",
            "director_polarity_policy_id",
            "director_reversal_transform_id",
            "reference_surface_offset",
            "reference_surface_offset_policy_id",
            "reference_surface_strain_transform_id",
            "material_direction",
            "material_angle_deg",
            "shell_section",
            "selector",
            "serialization_policy_id",
            "native_state_schema_id",
            "native_state_layout_id",
            "corotational_policy_id",
            "pressure_surface_policy_id",
            "activity_policy_id",
            "qualified_batch_policy_id",
            "dynamic_algebraic_policy_id",
            "geometric_stiffness_policy_id",
            "mass_policy_id",
            "source_selection_sha256",
            "thickness",
            "type",
        }
        if set(data) != required:
            raise ValueError("S3 V2D serialized schema keys mismatch")
        identities = {
            "formulation_id": FORMULATION_ID,
            "formulation_schema": FORMULATION_SCHEMA,
            "implementation_id": IMPLEMENTATION_ID,
            "selector": SELECTOR,
            "serialization_policy_id": SERIALIZATION_POLICY_ID,
            "native_state_schema_id": NATIVE_STATE_SCHEMA_ID,
            "native_state_layout_id": NATIVE_STATE_LAYOUT_ID,
            "corotational_policy_id": COROTATIONAL_POLICY_ID,
            "pressure_surface_policy_id": PRESSURE_SURFACE_POLICY_ID,
            "activity_policy_id": ACTIVITY_POLICY_ID,
            "qualified_batch_policy_id": BATCH_POLICY_ID,
            "dynamic_algebraic_policy_id": DYNAMIC_ALGEBRAIC_POLICY_ID,
            "geometric_stiffness_policy_id": GEOMETRIC_STIFFNESS_POLICY_ID,
            "mass_policy_id": MASS_POLICY_ID,
            "director_polarity_policy_id": DIRECTOR_POLARITY_POLICY_ID,
            "director_reversal_transform_id": DIRECTOR_REVERSAL_TRANSFORM_ID,
            "reference_surface_offset_policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
            "reference_surface_strain_transform_id": (
                REFERENCE_SURFACE_STRAIN_TRANSFORM_ID
            ),
            "source_selection_sha256": SOURCE_SELECTION_SHA256,
            "type": "NativeParityE4PLS3V2DShellElement",
        }
        if any(data[name] != value for name, value in identities.items()):
            raise ValueError("S3 V2D serialized fingerprint mismatch")
        serialized_normal = _immutable_serialized_unit_vector(
            data["reference_normal"], "reference_normal"
        )
        restored = cls(
            data["element_id"],
            data["node_ids"],
            data["material_name"],
            thickness=data["thickness"],
            reference_normal=serialized_normal,
            director_polarity=data["director_polarity"],
            reference_surface_offset=data["reference_surface_offset"],
            material_direction=data["material_direction"],
            material_angle_deg=data["material_angle_deg"],
            shell_section=data["shell_section"],
        )
        restored.reference_normal = serialized_normal
        restored._validate_configuration()
        return restored

    def __getstate__(self) -> Dict[str, Any]:
        self._unsupported("python_pickle_restart")

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        del state
        self._unsupported("python_pickle_restart")

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        self._unsupported("python_pickle_restart")


# Replacing a mutation hook installs a raising stub, so a swap-and-restore of
# either hook cannot silently suppress the write count; any ordinary class
# mutation also advances this generation.
_s3_v2d_runtime_epoch_manager = make_authority_epoch_manager(
    "qualified S3 V2D runtime"
)
_s3_v2d_runtime_epoch_manager.protect_type_entries(
    NativeParityE4PLS3V2DShellElement,
    ("__setattr__", "__delattr__"),
)
_s3_v2d_runtime_epoch_manager.watch_type(NativeParityE4PLS3V2DShellElement, None)


__all__ = [
    "ACTIVITY_POLICY_ID",
    "BATCH_POLICY_ID",
    "BLOCKED_OPERATIONS",
    "CAPABILITY_MATRIX",
    "DYNAMIC_ALGEBRAIC_MASS_WITNESS",
    "DYNAMIC_ALGEBRAIC_POLICY_ID",
    "DRILL_SCALE_POLICY_ID",
    "FORMULATION_ID",
    "FORMULATION_SCHEMA",
    "GEOMETRIC_STIFFNESS_POLICY_ID",
    "IMPLEMENTATION_ID",
    "MASS_POLICY_ID",
    "MATERIAL_LIFECYCLE_POLICY_ID",
    "NATIVE_STATE_LAYOUT_ID",
    "NATIVE_STATE_SCHEMA_ID",
    "NATIVE_SECTION_POLICY_ID",
    "NativeParityCapabilityError",
    "NativeParityE4PLS3V2DShellElement",
    "SELECTOR",
    "SERIALIZATION_POLICY_ID",
    "PRESSURE_SURFACE_POLICY_ID",
    "QUADRATURE_AUTHORITY_ID",
    "SOURCE_SELECTION_SHA256",
    "SUPPORTED_OPERATIONS",
    "V2C_OPERATOR_SHA256",
]
