"""Exact revision-bound stiffness plans for the bounded S3 V2C candidate.

The plan is deliberately narrower than a mechanics batch kernel.  A cold
capture evaluates every V2C element through its public, guarded formulation
and retains the resulting matrix bytes.  A warm capture reuses those bytes
only while the exact model, mesh, node, element, material, runtime and matrix
authorities remain unchanged.  No geometry is rounded and no matrix is shared
between distinct element IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from .e4_pl_s3_state import require_exact_numpy_runtime_authority
from .e4_pl_s3_v2c_element import (
    FORMULATION_ID,
    StrictFlatLinearE4PLS3V2CShellElement,
)
from .e4_pl_element import QualifiedE4PLShellElement
from .fe_core import FEModel, FEMesh, Material, Node


V2C_FAST_ASSEMBLY_POLICY_ID = "S3_V2C_EXACT_REVISION_BOUND_STIFFNESS_PLAN_V1"


class V2CFastAssemblyError(RuntimeError):
    """A V2C fast-plan input or retained authority is incompatible."""


def _readonly_matrix(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.float64).reshape((18, 18))


def _revision_key(mesh: FEMesh) -> Tuple[int, int, int]:
    revisions = mesh.revision_signature()
    return tuple(int(revisions.get(name, 0)) for name in ("topology", "geometry", "material"))


def _scalar_authority(value: Any) -> tuple[type[Any], Any]:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V2CFastAssemblyError("V2C fast-plan scalar authority is invalid")
    made = float(value)
    if not math.isfinite(made):
        raise V2CFastAssemblyError("V2C fast-plan scalar authority is nonfinite")
    return type(value), value


def _candidate(element: Any) -> bool:
    return (
        type(element) is StrictFlatLinearE4PLS3V2CShellElement
        and element.formulation_id == FORMULATION_ID
    )


def v2c_fast_candidate(element: Any) -> bool:
    """Return whether *element* is exactly the frozen V2C candidate."""

    return _candidate(element)


def _node_signature(mesh: FEMesh, node_ids: set[int]) -> tuple[Any, ...]:
    made = []
    for node_id in sorted(node_ids):
        node = mesh.nodes.get(node_id)
        if type(node) is not Node:
            raise V2CFastAssemblyError("V2C fast-plan node authority is absent")
        namespace = object.__getattribute__(node, "__dict__")
        coordinates = tuple(_scalar_authority(namespace.get(name)) for name in ("x", "y", "z"))
        made.append(
            (
                node_id,
                id(node),
                id(namespace),
                tuple(namespace),
                tuple(map(type, namespace)),
                namespace.get("id"),
                coordinates,
                namespace.get("_coordinate_revision"),
                tuple(namespace.get("dofs", ())),
            )
        )
    return tuple(made)


def _material_signature(material: Material) -> tuple[Any, ...]:
    if type(material) is not Material:
        raise V2CFastAssemblyError("V2C fast-plan requires an exact Material")
    namespace = object.__getattribute__(material, "__dict__")
    values = tuple(
        (name, type(namespace.get(name)), namespace.get(name))
        for name in (
            "elastic_modulus",
            "poisson_ratio",
            "density",
            "yield_stress",
            "hardening_curve",
        )
    )
    return (
        id(material),
        id(namespace),
        tuple(namespace),
        tuple(map(type, namespace)),
        values,
    )


def _element_signature(element: Any) -> tuple[Any, ...]:
    if not _candidate(element):
        raise V2CFastAssemblyError("V2C fast-plan element authority changed")
    namespace = object.__getattribute__(element, "__dict__")
    normal = namespace.get("reference_normal")
    if (
        type(normal) is not np.ndarray
        or normal.dtype != np.dtype(np.float64)
        or normal.shape != (3,)
        or not normal.flags.c_contiguous
        or normal.flags.writeable
    ):
        raise V2CFastAssemblyError("V2C fast-plan director authority is invalid")
    return (
        id(element),
        id(namespace),
        tuple(namespace),
        tuple(map(type, namespace)),
        namespace.get("element_id"),
        namespace.get("node_ids"),
        namespace.get("material_name"),
        _scalar_authority(namespace.get("thickness")),
        normal.tobytes(order="C"),
        namespace.get("_strict_flat_v2_frozen"),
        id(namespace.get("_qualified_direct_state_token")),
    )


def _validation_signature(
    model: FEModel,
    items: Sequence[Tuple[int, Any]],
) -> tuple[Any, ...]:
    if type(model) is not FEModel or type(model.mesh) is not FEMesh:
        raise V2CFastAssemblyError("V2C fast-plan requires exact FEModel/FEMesh inputs")
    mesh = model.mesh
    namespace = object.__getattribute__(mesh, "__dict__")
    token = namespace.get("_qualified_direct_state_token")
    if not isinstance(token, list) or len(token) != 1 or type(token[0]) is not int:
        raise V2CFastAssemblyError("V2C fast-plan mesh epoch authority is invalid")
    candidate_items = tuple((int(element_id), element) for element_id, element in items if _candidate(element))
    if not candidate_items:
        raise V2CFastAssemblyError("V2C fast-plan has no candidate elements")
    if len({element_id for element_id, _element in candidate_items}) != len(candidate_items):
        raise V2CFastAssemblyError("V2C fast-plan element IDs are not unique")
    node_ids: set[int] = set()
    elements = []
    materials = []
    for element_id, element in candidate_items:
        element_signature = _element_signature(element)
        if element_signature[4] != element_id or mesh.elements.get(element_id) is not element:
            raise V2CFastAssemblyError("V2C fast-plan registry identity changed")
        node_ids.update(int(node_id) for node_id in element.node_ids)
        material = model.get_material(element.material_name)
        elements.append((element_id, element_signature, id(material)))
        materials.append((element.material_name, _material_signature(material)))
    first = candidate_items[0][1]
    first._validate_configuration()
    first._validate_model_scope(mesh)
    return (
        id(model),
        id(object.__getattribute__(model, "__dict__")),
        id(mesh),
        id(namespace),
        id(mesh.elements),
        id(mesh.nodes),
        id(token),
        token[0],
        _revision_key(mesh),
        tuple(elements),
        tuple(materials),
        _node_signature(mesh, node_ids),
    )


def _fast_validation_signature(
    model: FEModel,
    items: Sequence[Tuple[int, Any]],
) -> tuple[Any, ...]:
    """Bind the supported mutation epoch without rebuilding numeric inputs."""

    if type(model) is not FEModel or type(model.mesh) is not FEMesh:
        raise V2CFastAssemblyError("V2C fast-plan requires exact FEModel/FEMesh inputs")
    mesh = model.mesh
    namespace = object.__getattribute__(mesh, "__dict__")
    token = namespace.get("_qualified_direct_state_token")
    revisions = namespace.get("revisions")
    if (
        not isinstance(token, list)
        or len(token) != 1
        or type(token[0]) is not int
        or type(revisions) is not dict
    ):
        raise V2CFastAssemblyError("V2C fast-plan epoch authority is invalid")
    candidate_items = tuple(
        (int(element_id), id(element))
        for element_id, element in items
        if _candidate(element)
    )
    if not candidate_items:
        raise V2CFastAssemblyError("V2C fast-plan has no candidate elements")
    first = next(element for _element_id, element in items if _candidate(element))
    first._module_authority_guard()
    material_signatures = tuple(
        (name, _material_signature(model.get_material(name)))
        for name in sorted(
            {
                element.material_name
                for _element_id, element in items
                if _candidate(element)
            }
        )
    )
    return (
        id(model),
        id(object.__getattribute__(model, "__dict__")),
        id(mesh),
        id(namespace),
        id(mesh.elements),
        id(mesh.nodes),
        id(token),
        token[0],
        tuple(int(revisions.get(name, 0)) for name in ("topology", "geometry", "material")),
        candidate_items,
        material_signatures,
    )


@dataclass(frozen=True)
class PreparedV2CStiffness:
    matrices: Mapping[int, np.ndarray]
    element_ids: Tuple[int, ...]
    matrix_payload_sha256: str
    matrices_prevalidated: bool
    revision_key: Tuple[int, int, int]
    validation_signature: tuple[Any, ...]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "element_count": len(self.element_ids),
            "element_ids": list(self.element_ids),
            "formulation_id": FORMULATION_ID,
            "matrix_payload_sha256": self.matrix_payload_sha256,
            "matrix_shape_finite_symmetry_prevalidated": self.matrices_prevalidated,
            "path": "exact_revision_bound_matrix_plan",
            "policy_id": V2C_FAST_ASSEMBLY_POLICY_ID,
            "revision_key": list(self.revision_key),
            "speedup_claimed": False,
        }


@dataclass(frozen=True)
class PreparedV2CGlobalStiffness:
    data_bytes: bytes
    indices_bytes: bytes
    indices_dtype: str
    indptr_bytes: bytes
    indptr_dtype: str
    info_bytes: bytes
    shape: Tuple[int, int]


def _global_fast_signature(
    model: FEModel,
    items: Sequence[Tuple[int, Any]],
) -> tuple[Any, ...] | None:
    if type(model) is not FEModel or type(model.mesh) is not FEMesh:
        return None
    frozen = tuple(items)
    if not frozen or not any(_candidate(element) for _element_id, element in frozen):
        return None
    if not all(
        type(element) in {QualifiedE4PLShellElement, StrictFlatLinearE4PLS3V2CShellElement}
        for _element_id, element in frozen
    ):
        return None
    mesh = model.mesh
    namespace = object.__getattribute__(mesh, "__dict__")
    if namespace.get("element_activity") is not None:
        return None
    token = namespace.get("_qualified_direct_state_token")
    revisions = namespace.get("revisions")
    if not isinstance(token, list) or len(token) != 1 or type(token[0]) is not int or type(revisions) is not dict:
        return None
    first = next(element for _element_id, element in frozen if _candidate(element))
    first._module_authority_guard()
    material_signatures = tuple(
        (name, _material_signature(model.get_material(name)))
        for name in sorted(
            {
                element.material_name
                for _element_id, element in frozen
                if _candidate(element)
            }
        )
    )
    return (
        id(model),
        id(object.__getattribute__(model, "__dict__")),
        id(mesh),
        id(namespace),
        id(mesh.elements),
        id(mesh.nodes),
        id(token),
        token[0],
        tuple(int(revisions.get(name, 0)) for name in ("topology", "geometry", "material")),
        tuple((int(element_id), id(element), type(element)) for element_id, element in frozen),
        material_signatures,
    )


def _make_global_plan_manager() -> tuple[Any, Any]:
    records: dict[int, tuple[Any, tuple[Any, ...], PreparedV2CGlobalStiffness]] = {}

    def discard(mesh_id: int, reference: Any) -> None:
        current = records.get(mesh_id)
        if current is not None and current[0] is reference:
            records.pop(mesh_id, None)

    def lookup(
        model: FEModel,
        items: Sequence[Tuple[int, Any]],
    ) -> PreparedV2CGlobalStiffness | None:
        signature = _global_fast_signature(model, items)
        if signature is None:
            return None
        current = records.get(id(model.mesh))
        if current is None:
            return None
        reference, expected_signature, plan = current
        if reference() is not model.mesh or signature != expected_signature or type(plan) is not PreparedV2CGlobalStiffness:
            records.pop(id(model.mesh), None)
            return None
        if not all(
            type(value) is bytes
            for value in (plan.data_bytes, plan.indices_bytes, plan.indptr_bytes, plan.info_bytes)
        ):
            records.pop(id(model.mesh), None)
            return None
        return plan

    def bind(
        model: FEModel,
        items: Sequence[Tuple[int, Any]],
        matrix: Any,
        info: Mapping[str, Any],
    ) -> None:
        signature = _global_fast_signature(model, items)
        if signature is None:
            return
        data = getattr(matrix, "data", None)
        indices = getattr(matrix, "indices", None)
        indptr = getattr(matrix, "indptr", None)
        shape = getattr(matrix, "shape", None)
        if (
            type(data) is not np.ndarray
            or data.dtype != np.dtype(np.float64)
            or data.ndim != 1
            or not np.all(np.isfinite(data))
            or type(indices) is not np.ndarray
            or indices.ndim != 1
            or indices.dtype.kind != "i"
            or type(indptr) is not np.ndarray
            or indptr.ndim != 1
            or indptr.dtype.kind != "i"
            or type(shape) is not tuple
            or len(shape) != 2
            or shape[0] != shape[1]
        ):
            raise V2CFastAssemblyError("V2C global stiffness output is incompatible")
        owned_info = dict(info)
        owned_info["assembly_time"] = 0.0
        owned_info["element_times"] = {
            str(element_id): 0.0 for element_id, _element in items
        }
        info_bytes = (
            json.dumps(owned_info, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("ascii")
        plan = PreparedV2CGlobalStiffness(
            data_bytes=np.ascontiguousarray(data).tobytes(order="C"),
            indices_bytes=np.ascontiguousarray(indices).tobytes(order="C"),
            indices_dtype=indices.dtype.str,
            indptr_bytes=np.ascontiguousarray(indptr).tobytes(order="C"),
            indptr_dtype=indptr.dtype.str,
            info_bytes=info_bytes,
            shape=(int(shape[0]), int(shape[1])),
        )
        mesh_id = id(model.mesh)
        reference = weakref.ref(model.mesh, lambda observed, expected=mesh_id: discard(expected, observed))
        records[mesh_id] = (reference, signature, plan)

    return lookup, bind


lookup_v2c_global_stiffness_plan, bind_v2c_global_stiffness_plan = _make_global_plan_manager()
del _make_global_plan_manager


def _make_plan_manager() -> Any:
    records: dict[
        int,
        tuple[
            Any,
            PreparedV2CStiffness,
            tuple[tuple[int, bytes], ...],
            tuple[Any, ...],
            str,
        ],
    ] = {}

    def discard(mesh_id: int, reference: Any) -> None:
        current = records.get(mesh_id)
        if current is not None and current[0] is reference:
            records.pop(mesh_id, None)

    def require_plan(
        plan: PreparedV2CStiffness,
        payloads: tuple[tuple[int, bytes], ...],
        expected_digest: str,
    ) -> bool:
        if type(plan) is not PreparedV2CStiffness or type(plan.matrices) is not MappingProxyType:
            return False
        if (
            plan.matrix_payload_sha256 != expected_digest
            or plan.element_ids != tuple(element_id for element_id, _payload in payloads)
            or plan.matrices_prevalidated is not True
        ):
            return False
        current = tuple(plan.matrices.items())
        if len(current) != len(payloads):
            return False
        for (current_id, matrix), (expected_id, payload) in zip(current, payloads):
            if current_id != expected_id or type(payload) is not bytes:
                return False
            if (
                type(matrix) is not np.ndarray
                or matrix.dtype != np.dtype(np.float64)
                or matrix.shape != (18, 18)
                or matrix.flags.writeable
            ):
                return False
            base: Any = matrix
            seen: set[int] = set()
            while type(base) is np.ndarray:
                if id(base) in seen or base.flags.writeable:
                    return False
                seen.add(id(base))
                base = base.base
            if base is not payload:
                return False
        return True

    def get(
        model: FEModel,
        items: Sequence[Tuple[int, Any]],
    ) -> tuple[PreparedV2CStiffness, bool]:
        require_exact_numpy_runtime_authority(context="S3 V2C fast stiffness plan")
        frozen_items = tuple(items)
        fast_signature = _fast_validation_signature(model, frozen_items)
        mesh = model.mesh
        mesh_id = id(mesh)
        current = records.get(mesh_id)
        if current is not None:
            reference, plan, payloads, expected_fast_signature, expected_digest = current
            if (
                reference() is mesh
                and fast_signature == expected_fast_signature
                and require_plan(plan, payloads, expected_digest)
            ):
                return plan, True
            records.pop(mesh_id, None)

        signature = _validation_signature(model, frozen_items)

        matrices: dict[int, np.ndarray] = {}
        payload_rows: list[tuple[int, bytes]] = []
        for element_id, element in frozen_items:
            if not _candidate(element):
                continue
            material = model.get_material(element.material_name)
            matrix = np.asarray(element.compute_stiffness_matrix(mesh, material), dtype=np.float64)
            scale = max(float(np.linalg.norm(matrix)), 1.0)
            if (
                matrix.shape != (18, 18)
                or not np.all(np.isfinite(matrix))
                or float(np.linalg.norm(matrix - matrix.T)) / scale > 1.0e-8
            ):
                raise V2CFastAssemblyError("V2C fast-plan matrix is incompatible")
            payload = np.ascontiguousarray(matrix).tobytes(order="C")
            payload_rows.append((int(element_id), payload))
            matrices[int(element_id)] = _readonly_matrix(payload)
        payloads = tuple(payload_rows)
        if not payloads:
            raise V2CFastAssemblyError("V2C fast-plan captured no matrices")
        digest = hashlib.sha256(b"".join(payload for _element_id, payload in payloads)).hexdigest().upper()
        plan = PreparedV2CStiffness(
            matrices=MappingProxyType(matrices),
            element_ids=tuple(element_id for element_id, _payload in payloads),
            matrix_payload_sha256=digest,
            matrices_prevalidated=True,
            revision_key=_revision_key(mesh),
            validation_signature=signature,
        )
        reference = weakref.ref(mesh, lambda observed, expected=mesh_id: discard(expected, observed))
        records[mesh_id] = (reference, plan, payloads, fast_signature, digest)
        if not require_plan(plan, payloads, digest):
            records.pop(mesh_id, None)
            raise V2CFastAssemblyError("V2C fast-plan retained authority is incompatible")
        return plan, False

    return get


get_v2c_stiffness_plan = _make_plan_manager()
del _make_plan_manager


__all__ = [
    "PreparedV2CGlobalStiffness",
    "PreparedV2CStiffness",
    "V2C_FAST_ASSEMBLY_POLICY_ID",
    "V2CFastAssemblyError",
    "bind_v2c_global_stiffness_plan",
    "get_v2c_stiffness_plan",
    "lookup_v2c_global_stiffness_plan",
    "v2c_fast_candidate",
]
