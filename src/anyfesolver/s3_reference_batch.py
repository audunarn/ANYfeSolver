"""Private bounded batches for the qualified S3 reference-elastic path.

The qualified companion must never enter the legacy TRI3 or qualified-Q4
kernels.  This module therefore provides two deliberately narrow
optimizations: translation-equivalent S3 elements share one formulation-native
component construction, and a warm mesh-owned plan reuses each remaining
element's own exact, revision-bound matrix.  The latter never rounds geometry
or substitutes a neighbouring element's matrix.  Recovery continues through
the element's public native recovery routine, so its quality, rank, bubble and
provenance guards remain the numerical authority.

Only homogeneous, isotropic, reference-elastic, zero-reference-offset,
positive-winding candidates are admitted.  Generalized sections,
anisotropy/Hill data, material history, nonzero reference-surface offsets and
non-positive winding remain on the scalar path.  Cold small groups are scalar;
their own exact formulation caches may be reused by later warm assemblies.
"""

from __future__ import annotations

import copy
import math
import weakref
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .e4_pl_s3_element import (
    QualifiedE4PLS3ShellElement,
    _invalidate_s3_guarded_call_caches as _INVALIDATE_S3_GUARDED_CACHES,
    _s3_runtime_epoch_manager as _S3_RUNTIME_EPOCH_MANAGER,
    _require_exact_s3_runtime_authority as _EXACT_S3_RUNTIME_GUARD,
)
from .e4_pl_s3_state import (
    require_exact_numpy_runtime_authority as _EXACT_NUMPY_RUNTIME_GUARD,
)
from .materials import is_isotropic_material
from .shell_sections import (
    SHELL_MEMBRANE_VOIGT_ORDER,
    SHELL_TRANSVERSE_SHEAR_ORDER,
)

if TYPE_CHECKING:
    from .fe_core import FEModel


REFERENCE_S3_FORMULATION_ID = "E4_PL_QUALIFIED_S3_COMPANION_V1"
REFERENCE_S3_BATCH_POLICY_ID = (
    "QUALIFIED_S3_REFERENCE_ELASTIC_EXACT_CACHE_PLAN_V2"
)
MIN_REFERENCE_S3_STIFFNESS_GROUP = 8
MIN_REFERENCE_S3_RECOVERY_GROUP = 128


def _bind_reference_s3_runtime_authority(
    numerical_guard: Any,
    element_guard: Any,
    exact_type: type[Any],
) -> Any:
    def require(model: "FEModel", *, context: str) -> None:
        numerical_guard(context=context)
        exact_elements = tuple(
            element
            for element in tuple(model.mesh.elements.values())
            if type(element) is exact_type
        )
        for element in exact_elements:
            element_guard(element, context=context)

    return require


_REQUIRE_REFERENCE_S3_RUNTIME_AUTHORITY = _bind_reference_s3_runtime_authority(
    _EXACT_NUMPY_RUNTIME_GUARD,
    _EXACT_S3_RUNTIME_GUARD,
    QualifiedE4PLS3ShellElement,
)


def _capture_reference_s3_runtime_lease(
    model: "FEModel",
    *,
    context: str,
) -> Any:
    """Capture one non-renewable S3 generation for a direct batch call."""

    runtime_guard = _REQUIRE_REFERENCE_S3_RUNTIME_AUTHORITY
    generation = _S3_RUNTIME_EPOCH_MANAGER.capture_generation()
    runtime_guard(model, context=context)
    _S3_RUNTIME_EPOCH_MANAGER.require_generation(generation)
    exact_elements = tuple(
        element
        for element in tuple(model.mesh.elements.values())
        if type(element) is QualifiedE4PLS3ShellElement
    )

    def invalidate() -> None:
        for element in exact_elements:
            _INVALIDATE_S3_GUARDED_CACHES(element)
        namespace = object.__getattribute__(model.mesh, "__dict__")
        if type(namespace) is dict:
            dict.pop(namespace, "_qualified_s3_reference_stiffness_plan", None)

    def require(expected_model: "FEModel", *, context: str) -> None:
        try:
            if expected_model is not model:
                raise ValueError("qualified S3 batch lease model changed")
            _S3_RUNTIME_EPOCH_MANAGER.require_generation(generation)
            runtime_guard(model, context=context)
            _S3_RUNTIME_EPOCH_MANAGER.require_generation(generation)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            invalidate()
            raise

    return require


def _run_with_reference_s3_runtime_lease(
    model: "FEModel",
    *,
    context: str,
    operation: Any,
) -> Any:
    """Run a direct S3 batch operation under one immutable generation."""

    lease = _capture_reference_s3_runtime_lease(
        model,
        context=f"{context} preflight",
    )
    try:
        result = operation(lease)
    except BaseException:
        lease(model, context=f"{context} exceptional output")
        raise
    lease(model, context=f"{context} output")
    return result


def _readonly(values: np.ndarray) -> np.ndarray:
    made = np.ascontiguousarray(values)
    return np.frombuffer(
        made.tobytes(order="C"), dtype=made.dtype
    ).reshape(made.shape)


def _revision_key(model: "FEModel") -> Tuple[int, int, int]:
    revisions = model.mesh.revision_signature()
    return (
        int(revisions.get("topology", 0)),
        int(revisions.get("geometry", 0)),
        int(revisions.get("material", 0)),
    )


def _direct_state_key(model: "FEModel") -> Tuple[int, int, int, int]:
    token = getattr(model.mesh, "_qualified_direct_state_token", (-1,))
    return (
        id(model.mesh.nodes),
        id(model.mesh.elements),
        id(token),
        int(token[0]),
    )


def _is_exact_qualified_s3(element: Any) -> bool:
    return (
        type(element) is QualifiedE4PLS3ShellElement
        and getattr(element, "formulation_id", None)
        == REFERENCE_S3_FORMULATION_ID
    )


def reference_s3_candidate(element: Any) -> bool:
    """Return whether an element belongs to the exact qualified S3 class."""

    return _is_exact_qualified_s3(element)


def reference_s3_eligibility(
    model: "FEModel",
    element: Any,
) -> Tuple[bool, str]:
    """Classify the narrow reference-elastic S3 batch contract."""

    if not _is_exact_qualified_s3(element):
        return False, "not_exact_qualified_s3"
    if bool(getattr(element, "legacy_stiffness_batch_eligible", True)):
        return False, "legacy_stiffness_flag_not_false"
    if bool(getattr(element, "legacy_nonlinear_batch_eligible", True)):
        return False, "legacy_nonlinear_flag_not_false"
    if getattr(element, "shell_section", None) is not None:
        return False, "generalized_section"
    material = model.get_material(element.material_name)
    if not is_isotropic_material(material):
        return False, "orthotropic_or_anisotropic_material"
    if getattr(material, "hill_yield", None) is not None:
        return False, "hill_material"
    if getattr(material, "hardening_curve", None) is not None:
        return False, "material_history"
    if getattr(element, "material_direction", None) is not None:
        return False, "oriented_material"
    if float(getattr(element, "material_angle_deg", 0.0)) != 0.0:
        return False, "material_angle"
    reference_surface_offset = float(
        getattr(element, "reference_surface_offset", 0.0)
    )
    if not math.isfinite(reference_surface_offset):
        return False, "invalid_reference_surface_offset"
    if reference_surface_offset != 0.0:
        return False, "nonzero_reference_surface_offset"
    thickness = float(getattr(element, "thickness", math.nan))
    if not math.isfinite(thickness) or thickness <= 0.0:
        return False, "invalid_thickness"

    coordinates = np.asarray(
        element.get_node_coordinates(model.mesh), dtype=float
    )
    normal = np.asarray(getattr(element, "reference_normal", ()), dtype=float)
    if (
        coordinates.shape != (3, 3)
        or normal.shape != (3,)
        or not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(normal))
    ):
        return False, "invalid_geometry_or_reference_normal"
    signed_area_twice = float(
        np.dot(
            np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]),
            normal,
        )
    )
    if not math.isfinite(signed_area_twice) or signed_area_twice <= 0.0:
        return False, "nonpositive_winding"
    return True, "eligible_reference_elastic_isotropic_positive_winding"


def _component_key(model: "FEModel", element: Any) -> Tuple[Any, ...]:
    """Return the formulation's own exact, revision-bound component key."""

    material = model.get_material(element.material_name)
    coordinates = np.asarray(
        element.get_node_coordinates(model.mesh), dtype=float
    )
    # ``True`` is the public production path's positive-winding guard and is
    # part of the QualifiedE4PLS3ShellElement cache identity.
    return (*element._cache_key(model.mesh, material, coordinates), True)


def _plan_validation_signature(
    model: "FEModel",
    material_names: Sequence[str],
) -> Tuple[Any, ...]:
    """Bind every input that can change eligibility or the S3 component key.

    The mesh revision catches supported topology/geometry/material mutations.
    A mesh-owned direct-state revision catches coordinate or qualified-S3
    attribute replacement without an element-by-element coordinate scan.
    Qualified S3 connectivity and vector inputs are immutable, so every
    supported element-state change also advances that shared revision.
    Equality of this stronger preimage therefore
    guarantees that both ``reference_s3_eligibility`` and ``_component_key``
    have the same result as when the plan was built, without repeated frame
    construction or centered-coordinate arithmetic.
    """

    from .elements import _shell_elastic_material_cache_fingerprint

    material_by_name = {
        str(material_name): model.get_material(str(material_name))
        for material_name in material_names
    }
    material_state = tuple(
        (
            material_name,
            type(material).__module__,
            type(material).__qualname__,
            id(material),
            str(getattr(material, "elastic_symmetry", "")),
            bool(is_isotropic_material(material)),
            _shell_elastic_material_cache_fingerprint(material),
            getattr(material, "hill_yield", None) is None,
            getattr(material, "hardening_curve", None) is None,
        )
        for material_name, material in sorted(material_by_name.items())
    )
    return (
        id(model.mesh),
        _revision_key(model),
        _direct_state_key(model),
        material_state,
    )


def _bind_plan_state_sources(
    model: "FEModel",
    items: Sequence[Tuple[int, Any]],
) -> None:
    """Normalize direct public-mapping replacements before plan creation."""

    from .fe_core import (
        _bind_qualified_direct_state_token,
        _ensure_qualified_state_mappings,
        _freeze_qualified_element_vector_inputs,
    )

    _ensure_qualified_state_mappings(model.mesh)
    token = model.mesh._qualified_direct_state_token
    for _raw_element_id, element in items:
        if not reference_s3_candidate(element):
            continue
        _freeze_qualified_element_vector_inputs(element)
        _bind_qualified_direct_state_token(element, token)
        for node_id in element.node_ids:
            node = model.mesh.nodes.get(int(node_id))
            if node is not None:
                _bind_qualified_direct_state_token(node, token)


def _immutable_component_copy(
    value: Any,
    frozen_arrays: Optional[Dict[int, np.ndarray]] = None,
) -> Any:
    arrays = {} if frozen_arrays is None else frozen_arrays
    if isinstance(value, np.ndarray):
        identity = id(value)
        frozen = arrays.get(identity)
        if frozen is None:
            frozen = _readonly(value)
            arrays[identity] = frozen
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _immutable_component_copy(item, arrays)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_component_copy(item, arrays) for item in value)
    return copy.deepcopy(value)


def _copy_components(components: Mapping[str, Any]) -> Mapping[str, Any]:
    return _immutable_component_copy(components)


def _adopt_components(
    element: Any,
    cache_key: Tuple[Any, ...],
    components: Mapping[str, Any],
    mesh: Any,
    material: Any,
) -> np.ndarray:
    copied = _copy_components(components)
    # These assignments adopt derived caches produced from already-bound
    # inputs.  They must not advance the model-input mutation epoch.
    object.__setattr__(element, "_qualified_components", copied)
    object.__setattr__(element, "_qualified_cache_key", cache_key)
    element._hourglass_stiffness_matrix = np.asarray(
        copied["hourglass"], dtype=float
    )
    element._stiffness_matrix = np.asarray(copied["total"], dtype=float)
    # Bind only after every derived cache member exists.  The exact warm-total
    # provenance includes the complete owned instance namespace, so binding
    # before the first hourglass assignment would make an adopted cache stale
    # immediately even though its mechanics inputs were unchanged.
    element._bind_qualified_component_guard(mesh, material)
    return element._stiffness_matrix


@dataclass(frozen=True)
class PreparedReferenceS3Components:
    matrices: Mapping[int, np.ndarray]
    element_cache_keys: Mapping[int, Tuple[Any, ...]]
    batched_element_ids: Tuple[int, ...]
    cached_element_ids: Tuple[int, ...]
    group_element_ids: Tuple[Tuple[int, ...], ...]
    candidate_element_ids: Tuple[int, ...]
    complete_eligible_coverage: bool
    fallback_reasons: Mapping[str, Tuple[int, ...]]
    component_evaluation_count: int
    revision_key: Tuple[int, int, int]
    minimum_group_size: int
    material_names: Tuple[str, ...]
    validation_signature: Tuple[Any, ...]
    matrices_prevalidated: bool

    def diagnostics(self) -> Dict[str, Any]:
        if self.batched_element_ids and self.cached_element_ids:
            path = "formulation_native_shared_components_and_exact_cache_reuse"
        elif self.cached_element_ids:
            path = "formulation_native_exact_cache_reuse"
        elif self.batched_element_ids:
            path = "formulation_native_shared_components"
        else:
            path = "formulation_native_scalar_fallback"
        return {
            "policy_id": REFERENCE_S3_BATCH_POLICY_ID,
            "formulation_id": REFERENCE_S3_FORMULATION_ID,
            "scope": "reference_elastic_isotropic_positive_winding",
            "path": path,
            "candidate_element_count": len(self.candidate_element_ids),
            "element_count": len(self.matrices),
            "translation_group_element_count": len(self.batched_element_ids),
            "exact_element_cache_reuse_count": len(self.cached_element_ids),
            "exact_translation_group_count": len(self.group_element_ids),
            "component_evaluation_count": int(self.component_evaluation_count),
            "matrix_shape_finite_symmetry_prevalidated": bool(
                self.matrices_prevalidated
            ),
            "element_ids": list(self.matrices),
            "group_element_ids": [list(group) for group in self.group_element_ids],
            "fallback_reasons": {
                reason: list(element_ids)
                for reason, element_ids in self.fallback_reasons.items()
            },
            "revision_key": list(self.revision_key),
            "parallel_kernel": False,
            "legacy_stiffness_batch_eligible": False,
            "legacy_nonlinear_batch_eligible": False,
            "speedup_claimed": False,
        }


def _make_prepared_s3_matrix_authority() -> tuple[Any, Any]:
    """Keep exact retained-matrix provenance outside the public plan object."""

    records: Dict[int, tuple[Any, ...]] = {}
    reference_factory = weakref.ref
    float64_dtype = np.dtype(np.float64)

    def array_authority(
        matrix: Any,
    ) -> tuple[Any, str, tuple[int, ...], tuple[int, ...], tuple[Any, ...], bytes] | None:
        if (
            type(matrix) is not np.ndarray
            or matrix.dtype != float64_dtype
            or matrix.shape != (18, 18)
            or matrix.strides != (144, 8)
            or not matrix.flags.c_contiguous
            or matrix.flags.writeable
        ):
            return None
        bases: list[Any] = []
        current: Any = matrix
        seen: set[int] = set()
        while type(current) is np.ndarray:
            if id(current) in seen or current.flags.writeable:
                return None
            seen.add(id(current))
            base = current.base
            if base is None:
                return None
            bases.append(base)
            current = base
        if not (
            type(current) is bytes
            or (
                type(current) is memoryview
                and current.readonly
                and type(current.obj) is bytes
            )
        ):
            return None
        payload = memoryview(matrix).cast("B").tobytes()
        if len(payload) != 18 * 18 * 8:
            return None
        return (
            matrix,
            matrix.dtype.str,
            matrix.shape,
            matrix.strides,
            tuple(bases),
            payload,
        )

    def bind(plan: PreparedReferenceS3Components) -> None:
        if type(plan) is not PreparedReferenceS3Components:
            return
        namespace = object.__getattribute__(plan, "__dict__")
        matrices = dict.get(namespace, "matrices")
        if type(matrices) is not MappingProxyType:
            return
        made: list[tuple[Any, ...]] = []
        for element_id, matrix in matrices.items():
            authority = array_authority(matrix)
            if type(element_id) is not int or authority is None:
                return
            made.append((element_id, *authority))
        if not made or dict.get(namespace, "matrices_prevalidated") is not True:
            return
        identity = id(plan)

        def discard(reference: Any, *, expected_identity: int = identity) -> None:
            current = records.get(expected_identity)
            if current is not None and current[0] is reference:
                records.pop(expected_identity, None)

        reference = reference_factory(plan, discard)
        records[identity] = (
            reference,
            namespace,
            matrices,
            tuple(matrices.items()),
            tuple(made),
        )

    def require(plan: PreparedReferenceS3Components) -> bool:
        identity = id(plan)
        record = records.get(identity)

        def reject() -> bool:
            records.pop(identity, None)
            return False

        if record is None or type(plan) is not PreparedReferenceS3Components:
            return False
        reference, namespace, matrices, matrix_items, authorities = record
        if (
            reference() is not plan
            or object.__getattribute__(plan, "__dict__") is not namespace
            or dict.get(namespace, "matrices") is not matrices
            or dict.get(namespace, "matrices_prevalidated") is not True
            or type(matrices) is not MappingProxyType
        ):
            return reject()
        current_items = tuple(matrices.items())
        if (
            len(current_items) != len(matrix_items)
            or any(
                current_id != expected_id or current_matrix is not expected_matrix
                for (current_id, current_matrix), (
                    expected_id,
                    expected_matrix,
                ) in zip(current_items, matrix_items)
            )
        ):
            return reject()
        for (
            element_id,
            matrix,
            dtype_string,
            shape,
            strides,
            bases,
            payload,
        ) in authorities:
            if (
                matrices[element_id] is not matrix
                or type(matrix) is not np.ndarray
                or matrix.dtype.str != dtype_string
                or matrix.shape != shape
                or matrix.strides != strides
                or not matrix.flags.c_contiguous
                or matrix.flags.writeable
            ):
                return reject()
            current: Any = matrix
            for expected_base in bases:
                if current.base is not expected_base:
                    return reject()
                current = expected_base
            if not (
                type(current) is bytes
                or (
                    type(current) is memoryview
                    and current.readonly
                    and type(current.obj) is bytes
                )
            ):
                return reject()
            if memoryview(matrix).cast("B").tobytes() != payload:
                return reject()
        return True

    return bind, require


(
    _bind_prepared_s3_matrix_authority,
    _require_prepared_s3_matrix_authority,
) = _make_prepared_s3_matrix_authority()


def _prepare_reference_s3_components_under_lease(
    model: "FEModel",
    items: Sequence[Tuple[int, Any]],
    *,
    minimum_group_size: int = MIN_REFERENCE_S3_STIFFNESS_GROUP,
    allow_exact_element_cache_reuse: bool = True,
    _runtime_lease: Any,
) -> PreparedReferenceS3Components:
    """Prepare one validated component evaluation per exact cache-key group."""

    _bind_plan_state_sources(model, items)
    runtime_guard = _runtime_lease
    runtime_guard(model, context="qualified S3 reference batch state binding")
    minimum = max(1, int(minimum_group_size))
    candidate_ids: list[int] = []
    eligible_ids: list[int] = []
    candidate_material_names: set[str] = set()
    groups: Dict[Tuple[Any, ...], list[Tuple[int, Any]]] = {}
    fallback: Dict[str, list[int]] = {}
    for raw_element_id, element in items:
        element_id = int(raw_element_id)
        if not reference_s3_candidate(element):
            continue
        candidate_ids.append(element_id)
        candidate_material_names.add(str(element.material_name))
        eligible, reason = reference_s3_eligibility(model, element)
        if not eligible:
            fallback.setdefault(reason, []).append(element_id)
            continue
        eligible_ids.append(element_id)
        groups.setdefault(_component_key(model, element), []).append(
            (element_id, element)
        )

    matrices: Dict[int, np.ndarray] = {}
    element_cache_keys: Dict[int, Tuple[Any, ...]] = {}
    admitted_groups: list[Tuple[int, ...]] = []
    cached_element_ids: list[int] = []
    evaluation_count = 0
    for cache_key, group in groups.items():
        ordered_group = tuple(int(element_id) for element_id, _element in group)
        if len(group) < minimum:
            # A warm production assembly has already populated each element's
            # formulation-native, exact cache.  Retain those exact matrices in
            # the mesh-owned plan even when binary64 coordinate subtraction
            # prevents nominally translated elements from forming a large
            # byte-identical group.  No geometry is rounded and no matrix is
            # shared between distinct cache keys.
            cached_group: Dict[int, np.ndarray] = {}
            if allow_exact_element_cache_reuse:
                for element_id, element in group:
                    current = getattr(element, "_qualified_components", None)
                    current_key = getattr(element, "_qualified_cache_key", None)
                    if (
                        current is None
                        or current_key != cache_key
                        or current.get("formulation_id")
                        != REFERENCE_S3_FORMULATION_ID
                        or bool(current.get("legacy_fallback", True))
                    ):
                        cached_group = {}
                        break
                    element._validate_qualified_component_cache_identity()
                    cached_group[int(element_id)] = np.asarray(
                        current["total"], dtype=float
                    )
                    element._bind_qualified_component_guard(
                        model.mesh,
                        model.get_material(element.material_name),
                    )
            if cached_group:
                matrices.update(cached_group)
                cached_element_ids.extend(cached_group)
                element_cache_keys.update(
                    {int(element_id): cache_key for element_id in cached_group}
                )
                continue
            fallback.setdefault("group_below_minimum_size", []).extend(ordered_group)
            continue
        first_id, first = group[0]
        material = model.get_material(first.material_name)
        components = first.compute_stiffness_components(model.mesh, material)
        evaluation_count += 1
        if (
            components.get("formulation_id") != REFERENCE_S3_FORMULATION_ID
            or bool(components.get("legacy_fallback", True))
        ):
            raise ValueError(
                "qualified S3 reference batch received incompatible component provenance"
            )
        if tuple(first._qualified_cache_key) != tuple(cache_key):
            raise ValueError(
                "qualified S3 reference batch component cache identity changed during evaluation"
            )
        matrices[int(first_id)] = np.asarray(components["total"], dtype=float)
        element_cache_keys[int(first_id)] = cache_key
        for element_id, element in group[1:]:
            current = getattr(element, "_qualified_components", None)
            current_key = getattr(element, "_qualified_cache_key", None)
            if current is not None and current_key == cache_key:
                element._validate_qualified_component_cache_identity()
                element._bind_qualified_component_guard(
                    model.mesh,
                    model.get_material(element.material_name),
                )
                matrices[int(element_id)] = np.asarray(
                    current["total"], dtype=float
                )
            else:
                matrices[int(element_id)] = _adopt_components(
                    element,
                    cache_key,
                    components,
                    model.mesh,
                    model.get_material(element.material_name),
                )
            element_cache_keys[int(element_id)] = cache_key
        admitted_groups.append(ordered_group)

    frozen_matrices = MappingProxyType(
        {
            int(element_id): _readonly(np.asarray(matrix, dtype=float).copy())
            for element_id, matrix in matrices.items()
        }
    )
    matrices_prevalidated = bool(frozen_matrices) and all(
        matrix.shape == (18, 18)
        and bool(np.all(np.isfinite(matrix)))
        and float(
            np.linalg.norm(matrix - matrix.T)
            / max(float(np.linalg.norm(matrix)), 1.0)
        )
        <= 1.0e-8
        for matrix in frozen_matrices.values()
    )
    frozen_element_cache_keys = MappingProxyType(
        {
            int(element_id): tuple(cache_key)
            for element_id, cache_key in element_cache_keys.items()
        }
    )
    frozen_fallback = MappingProxyType(
        {
            reason: tuple(sorted(int(element_id) for element_id in element_ids))
            for reason, element_ids in sorted(fallback.items())
        }
    )
    admitted_element_ids = {
        int(element_id)
        for admitted_group in admitted_groups
        for element_id in admitted_group
    }
    cached_element_id_set = set(cached_element_ids)
    runtime_guard(
        model,
        context="qualified S3 reference batch signature preflight",
    )
    validation_signature = _plan_validation_signature(
        model,
        tuple(sorted(candidate_material_names)),
    )
    runtime_guard(
        model,
        context="qualified S3 reference batch signature observation",
    )
    result = PreparedReferenceS3Components(
        matrices=frozen_matrices,
        element_cache_keys=frozen_element_cache_keys,
        batched_element_ids=tuple(
            int(element_id)
            for element_id, _element in items
            if int(element_id) in admitted_element_ids
        ),
        cached_element_ids=tuple(
            int(element_id)
            for element_id, _element in items
            if int(element_id) in cached_element_id_set
        ),
        group_element_ids=tuple(admitted_groups),
        candidate_element_ids=tuple(candidate_ids),
        complete_eligible_coverage=(set(eligible_ids) == set(matrices)),
        fallback_reasons=frozen_fallback,
        component_evaluation_count=int(evaluation_count),
        revision_key=_revision_key(model),
        minimum_group_size=minimum,
        material_names=tuple(sorted(candidate_material_names)),
        validation_signature=validation_signature,
        matrices_prevalidated=matrices_prevalidated,
    )
    _bind_prepared_s3_matrix_authority(result)
    runtime_guard(model, context="qualified S3 reference batch output")
    return result


def prepare_reference_s3_components(
    model: "FEModel",
    items: Sequence[Tuple[int, Any]],
    *,
    minimum_group_size: int = MIN_REFERENCE_S3_STIFFNESS_GROUP,
    allow_exact_element_cache_reuse: bool = True,
) -> PreparedReferenceS3Components:
    """Prepare exact S3 components under one direct-call authority lease."""

    frozen_items = tuple(items)
    return _run_with_reference_s3_runtime_lease(
        model,
        context="qualified S3 reference batch",
        operation=lambda lease: _prepare_reference_s3_components_under_lease(
            model,
            frozen_items,
            minimum_group_size=minimum_group_size,
            allow_exact_element_cache_reuse=allow_exact_element_cache_reuse,
            _runtime_lease=lease,
        ),
    )


def _stiffness_plan_is_current(
    model: "FEModel",
    cached: PreparedReferenceS3Components,
) -> bool:
    """Revalidate the exact key preimage, provenance, and retained matrices."""

    try:
        runtime_guard = _REQUIRE_REFERENCE_S3_RUNTIME_AUTHORITY
        runtime_guard(
            model,
            context="qualified S3 stiffness signature preflight",
        )
        current_signature = _plan_validation_signature(
            model,
            cached.material_names,
        )
        runtime_guard(
            model,
            context="qualified S3 stiffness signature observation",
        )
        if (
            current_signature != cached.validation_signature
            or not _require_prepared_s3_matrix_authority(cached)
        ):
            return False
        # A cold scalar fallback populates its element cache after plan
        # construction.  Rebuild once rather than perpetually omitting it.
        if not cached.complete_eligible_coverage:
            return False
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return True


def _get_reference_s3_stiffness_components_under_lease(
    model: "FEModel",
    items: Sequence[Tuple[int, Any]],
    *,
    minimum_group_size: int = MIN_REFERENCE_S3_STIFFNESS_GROUP,
    complete_candidate_items: bool = False,
    _runtime_lease: Any,
) -> Tuple[PreparedReferenceS3Components, bool]:
    """Return a mesh-owned plan; reuse requires the caller's complete set."""

    runtime_guard = _runtime_lease
    revision = _revision_key(model)
    minimum = max(1, int(minimum_group_size))
    cached = getattr(
        model.mesh,
        "_qualified_s3_reference_stiffness_plan",
        None,
    )
    if (
        isinstance(cached, PreparedReferenceS3Components)
        and bool(complete_candidate_items)
        and cached.revision_key == revision
        and cached.minimum_group_size == minimum
        and bool(cached.matrices)
        and _stiffness_plan_is_current(model, cached)
    ):
        runtime_guard(model, context="qualified S3 stiffness plan reuse")
        return cached, True
    prepared = _prepare_reference_s3_components_under_lease(
        model,
        items,
        minimum_group_size=minimum,
        _runtime_lease=runtime_guard,
    )
    if prepared.matrices:
        model.mesh._qualified_s3_reference_stiffness_plan = prepared
    elif hasattr(model.mesh, "_qualified_s3_reference_stiffness_plan"):
        delattr(model.mesh, "_qualified_s3_reference_stiffness_plan")
    runtime_guard(model, context="qualified S3 stiffness plan output")
    return prepared, False


def get_reference_s3_stiffness_components(
    model: "FEModel",
    items: Sequence[Tuple[int, Any]],
    *,
    minimum_group_size: int = MIN_REFERENCE_S3_STIFFNESS_GROUP,
    complete_candidate_items: bool = False,
) -> Tuple[PreparedReferenceS3Components, bool]:
    """Return a mesh-owned S3 plan under one direct-call authority lease."""

    frozen_items = tuple(items)
    return _run_with_reference_s3_runtime_lease(
        model,
        context="qualified S3 stiffness plan",
        operation=lambda lease: _get_reference_s3_stiffness_components_under_lease(
            model,
            frozen_items,
            minimum_group_size=minimum_group_size,
            complete_candidate_items=complete_candidate_items,
            _runtime_lease=lease,
        ),
    )


@dataclass(frozen=True)
class ReferenceS3RecoveryKernel:
    frame: np.ndarray
    local_transform: np.ndarray
    bubble_map: np.ndarray
    strain_operators: np.ndarray
    constitutive: np.ndarray
    membrane_material: np.ndarray
    shear_material: np.ndarray
    thickness: float
    director_polarity: int

    @property
    def retained_bytes(self) -> int:
        return int(
            self.frame.nbytes
            + self.local_transform.nbytes
            + self.bubble_map.nbytes
            + self.strain_operators.nbytes
            + self.constitutive.nbytes
            + self.membrane_material.nbytes
            + self.shear_material.nbytes
        )


@dataclass(frozen=True)
class ReferenceS3RecoveryBatch:
    revision_key: Tuple[int, int, int]
    direct_state_key: Tuple[int, int, int, int]
    element_ids: np.ndarray
    index_by_id: Mapping[int, int]
    dof_mappings: np.ndarray
    kernel_index_by_row: np.ndarray
    kernels: Tuple[ReferenceS3RecoveryKernel, ...]
    group_element_ids: Tuple[Tuple[int, ...], ...]
    component_evaluation_count: int

    @property
    def retained_bytes(self) -> int:
        return int(
            self.element_ids.nbytes
            + self.dof_mappings.nbytes
            + self.kernel_index_by_row.nbytes
            + sum(kernel.retained_bytes for kernel in self.kernels)
        )

    def is_valid(self, model: "FEModel") -> bool:
        return (
            self.revision_key == _revision_key(model)
            and self.direct_state_key == _direct_state_key(model)
        )

    def select_indices(self, element_ids: Iterable[int]) -> np.ndarray:
        return np.asarray(
            [
                self.index_by_id[int(element_id)]
                for element_id in element_ids
                if int(element_id) in self.index_by_id
            ],
            dtype=np.intp,
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "policy_id": REFERENCE_S3_BATCH_POLICY_ID,
            "formulation_id": REFERENCE_S3_FORMULATION_ID,
            "scope": "reference_elastic_isotropic_positive_winding",
            "path": "formulation_native_shared_components_recovery",
            "element_count": int(self.element_ids.size),
            "exact_translation_group_count": len(self.group_element_ids),
            "component_evaluation_count": int(self.component_evaluation_count),
            "element_ids": [int(value) for value in self.element_ids],
            "group_element_ids": [list(group) for group in self.group_element_ids],
            "revision_key": list(self.revision_key),
            "parallel_kernel": False,
            "legacy_stiffness_batch_eligible": False,
            "legacy_nonlinear_batch_eligible": False,
            "speedup_claimed": False,
        }


def _build_reference_s3_recovery_batch_under_lease(
    model: "FEModel",
    items: Sequence[Tuple[int, Any, np.ndarray]],
    *,
    minimum_group_size: int = MIN_REFERENCE_S3_RECOVERY_GROUP,
    _runtime_lease: Any,
) -> Tuple[ReferenceS3RecoveryBatch | None, PreparedReferenceS3Components]:
    """Build an immutable recovery layout for admitted large S3 groups."""

    element_items = tuple(
        (int(element_id), element)
        for element_id, element, _mapping in items
    )
    prepared = _prepare_reference_s3_components_under_lease(
        model,
        element_items,
        minimum_group_size=minimum_group_size,
        _runtime_lease=_runtime_lease,
    )
    if not prepared.batched_element_ids:
        return None, prepared
    item_by_id = {
        int(element_id): (element, np.asarray(mapping, dtype=np.intp).reshape(-1))
        for element_id, element, mapping in items
    }
    ordered_ids = tuple(prepared.batched_element_ids)
    mappings = np.asarray(
        [item_by_id[element_id][1] for element_id in ordered_ids],
        dtype=np.intp,
    )
    if mappings.shape != (len(ordered_ids), 18):
        raise ValueError("qualified S3 recovery batch requires 18 DOFs per element")
    from .e4_pl_s3_element import TRIANGLE_QUADRATURE, _kinematic_matrix
    from .elements import _shell_material_matrices

    row_by_id = {element_id: index for index, element_id in enumerate(ordered_ids)}
    kernel_index_by_row = np.empty(len(ordered_ids), dtype=np.intp)
    kernels: list[ReferenceS3RecoveryKernel] = []
    for kernel_index, group_ids in enumerate(prepared.group_element_ids):
        first = item_by_id[int(group_ids[0])][0]
        components = getattr(first, "_qualified_components", None)
        if not isinstance(components, Mapping):
            raise ValueError("qualified S3 recovery batch lacks validated components")
        frame = np.asarray(components["frame"], dtype=float)
        generalized_transform = first._director_generalized_transform()
        strain_operators = np.asarray(
            [
                generalized_transform
                @ _kinematic_matrix(
                    components["local_nodes"],
                    r,
                    s,
                    components["assumed_shear_samples"],
                )
                for r, s, _weight in TRIANGLE_QUADRATURE
            ],
            dtype=float,
        )
        material = model.get_material(first.material_name)
        membrane, shear, _strain_transform, _stress_transform = (
            _shell_material_matrices(material, first._material_angle(frame))
        )
        kernels.append(
            ReferenceS3RecoveryKernel(
                frame=_readonly(frame),
                local_transform=_readonly(first._local_dof_transform(frame)),
                bubble_map=_readonly(
                    np.asarray(components["bubble_map"], dtype=float)
                ),
                strain_operators=_readonly(strain_operators),
                constitutive=_readonly(
                    np.asarray(components["constitutive"], dtype=float)
                ),
                membrane_material=_readonly(np.asarray(membrane, dtype=float)),
                shear_material=_readonly(np.asarray(shear, dtype=float)),
                thickness=float(first.thickness),
                director_polarity=int(first.director_polarity),
            )
        )
        for element_id in group_ids:
            kernel_index_by_row[row_by_id[int(element_id)]] = kernel_index
    batch = ReferenceS3RecoveryBatch(
        revision_key=_revision_key(model),
        direct_state_key=_direct_state_key(model),
        element_ids=_readonly(np.asarray(ordered_ids, dtype=np.int64)),
        index_by_id=MappingProxyType(
            {element_id: index for index, element_id in enumerate(ordered_ids)}
        ),
        dof_mappings=_readonly(mappings),
        kernel_index_by_row=_readonly(kernel_index_by_row),
        kernels=tuple(kernels),
        group_element_ids=prepared.group_element_ids,
        component_evaluation_count=prepared.component_evaluation_count,
    )
    return batch, prepared


def build_reference_s3_recovery_batch(
    model: "FEModel",
    items: Sequence[Tuple[int, Any, np.ndarray]],
    *,
    minimum_group_size: int = MIN_REFERENCE_S3_RECOVERY_GROUP,
) -> Tuple[ReferenceS3RecoveryBatch | None, PreparedReferenceS3Components]:
    """Build a recovery batch under one direct-call authority lease."""

    frozen_items = tuple(items)
    return _run_with_reference_s3_runtime_lease(
        model,
        context="qualified S3 recovery batch",
        operation=lambda lease: _build_reference_s3_recovery_batch_under_lease(
            model,
            frozen_items,
            minimum_group_size=minimum_group_size,
            _runtime_lease=lease,
        ),
    )


def _recover_with_kernel(
    kernel: ReferenceS3RecoveryKernel,
    element_displacements: np.ndarray,
    *,
    return_global: bool,
) -> Dict[str, Any]:
    """Evaluate the frozen homogeneous reference-elastic S3 recovery map."""

    from .e4_pl_s3_element import (
        HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID,
        PHYSICAL_EXTERNAL_INDICES,
        REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
    )
    from .e4_pl_s3_state import REFERENCE_SURFACE_OFFSET_POLICY_ID

    vector = np.asarray(element_displacements, dtype=float).reshape(18)
    if not np.all(np.isfinite(vector)):
        raise ValueError("qualified S3 recovery requires finite displacements")
    frame = np.asarray(kernel.frame, dtype=float)
    local_external = np.asarray(kernel.local_transform, dtype=float) @ vector
    physical_external = local_external[PHYSICAL_EXTERNAL_INDICES]
    bubble = np.asarray(kernel.bubble_map, dtype=float) @ physical_external
    coordinates = np.concatenate((physical_external, bubble))
    station_count = int(kernel.strain_operators.shape[0])
    strains = np.zeros((station_count, 8), dtype=float)
    resultants = np.zeros_like(strains)
    for station in range(station_count):
        strains[station] = kernel.strain_operators[station] @ coordinates
        resultants[station] = kernel.constitutive @ strains[station]

    recovered: Dict[str, Any] = {
        "recovery_scope": (
            "qualified_s3_local_and_global_physical"
            if return_global
            else "qualified_s3_local_physical_only"
        ),
        "physical_stress_available": True,
        "membrane_resultant_order": SHELL_MEMBRANE_VOIGT_ORDER,
        "transverse_shear_resultant_order": SHELL_TRANSVERSE_SHEAR_ORDER,
        "membrane_strain": strains[:, :3],
        "curvature": strains[:, 3:6],
        "transverse_shear_strain": strains[:, 6:],
        "membrane_resultants": resultants[:, :3],
        "bending_resultants": resultants[:, 3:6],
        "transverse_shear_resultants": resultants[:, 6:],
        "bubble_rotations": bubble.copy(),
        "numerical_fields_excluded": True,
        # Eligibility rejects nonzero reference-surface offsets.  Preserve the
        # scalar qualified-S3 recovery schema exactly for the admitted zero-
        # offset path so batching never changes provenance or key order.
        "reference_surface_offset": 0.0,
        "reference_surface_offset_policy_id": REFERENCE_SURFACE_OFFSET_POLICY_ID,
        "section_origin_offset_from_reference": 0.0,
        "physical_bottom_offset_from_reference": -0.5 * kernel.thickness,
        "physical_top_offset_from_reference": 0.5 * kernel.thickness,
        "bubble_linearization_policy": REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
        "through_thickness_stress_profile": HOMOGENEOUS_ELASTIC_STRESS_PROFILE_ID,
    }
    polarity = float(kernel.director_polarity)
    if return_global:
        global_membrane = np.zeros((station_count, 3, 3), dtype=float)
        global_bending = np.zeros_like(global_membrane)
        global_shear = np.zeros((station_count, 3), dtype=float)
        for station in range(station_count):
            membrane = resultants[station, :3]
            bending = resultants[station, 3:6]
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
            global_membrane[station] = frame @ membrane_tensor @ frame.T
            global_bending[station] = polarity * (
                frame @ bending_tensor @ frame.T
            )
            global_shear[station] = polarity * (
                resultants[station, 6] * frame[:, 0]
                + resultants[station, 7] * frame[:, 1]
            )
        recovered.update(
            {
                "global_membrane_resultant_tensors": global_membrane,
                "global_bending_resultant_tensors": global_bending,
                "global_transverse_shear_resultants": global_shear,
            }
        )

    membrane_stress = strains[:, :3] @ kernel.membrane_material.T
    moment = resultants[:, 3:6]
    bending_stress = 6.0 * moment / (kernel.thickness * kernel.thickness)
    transverse = strains[:, 6:] @ ((5.0 / 6.0) * kernel.shear_material).T
    recovered.update(
        {
            "membrane_xx": membrane_stress[:, 0],
            "membrane_yy": membrane_stress[:, 1],
            "membrane_xy": membrane_stress[:, 2],
            "bending_xx": bending_stress[:, 0],
            "bending_yy": bending_stress[:, 1],
            "bending_xy": bending_stress[:, 2],
            "shear_xz": transverse[:, 0],
            "shear_yz": transverse[:, 1],
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
    recovered["hill_utilization"] = np.zeros(station_count, dtype=float)
    recovered["equivalent_stress"] = recovered["von_mises"].copy()
    recovered["equivalent_stress_measure"] = "von_mises"
    if return_global:
        for surface, values in (("top", top), ("bot", bottom)):
            local_tensors = np.zeros((station_count, 3, 3), dtype=float)
            local_tensors[:, 0, 0] = values[:, 0]
            local_tensors[:, 1, 1] = values[:, 1]
            local_tensors[:, 0, 1] = values[:, 2]
            local_tensors[:, 1, 0] = values[:, 2]
            owner_shear = polarity * transverse
            local_tensors[:, 0, 2] = owner_shear[:, 0]
            local_tensors[:, 2, 0] = owner_shear[:, 0]
            local_tensors[:, 1, 2] = owner_shear[:, 1]
            local_tensors[:, 2, 1] = owner_shear[:, 1]
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
                f"qualified S3 recovery produced non-finite field {name!r}"
            )
    return recovered


def _recover_reference_s3_under_lease(
    model: "FEModel",
    batch: ReferenceS3RecoveryBatch,
    selected_indices: np.ndarray,
    displacements: np.ndarray,
    *,
    return_global: bool,
    _runtime_lease: Any,
) -> Mapping[int, Dict[str, Any]]:
    """Recover selected S3 rows through the frozen formulation-native kernel."""

    _runtime_lease(model, context="qualified S3 recovery validity preflight")
    if not batch.is_valid(model):
        raise ValueError("qualified S3 recovery batch is stale after a model revision")
    _runtime_lease(model, context="qualified S3 recovery validity observation")
    vector = np.asarray(displacements, dtype=float).reshape(-1)
    indices = np.asarray(selected_indices, dtype=np.intp).reshape(-1)
    recovered: Dict[int, Dict[str, Any]] = {}
    for raw_index in indices:
        index = int(raw_index)
        element_id = int(batch.element_ids[index])
        mapping = np.asarray(batch.dof_mappings[index], dtype=np.intp)
        if (
            mapping.size != 18
            or int(mapping.min()) < 0
            or int(mapping.max()) >= vector.size
        ):
            raise ValueError(
                "fail-closed element recovery requires a complete in-range DOF mapping"
            )
        kernel_index = int(batch.kernel_index_by_row[index])
        recovered[element_id] = _recover_with_kernel(
            batch.kernels[kernel_index],
            vector[mapping],
            return_global=bool(return_global),
        )
    return recovered


def recover_reference_s3(
    model: "FEModel",
    batch: ReferenceS3RecoveryBatch,
    selected_indices: np.ndarray,
    displacements: np.ndarray,
    *,
    return_global: bool,
) -> Mapping[int, Dict[str, Any]]:
    """Recover S3 rows under one direct-call authority lease."""

    return _run_with_reference_s3_runtime_lease(
        model,
        context="qualified S3 recovery",
        operation=lambda lease: _recover_reference_s3_under_lease(
            model,
            batch,
            selected_indices,
            displacements,
            return_global=return_global,
            _runtime_lease=lease,
        ),
    )
