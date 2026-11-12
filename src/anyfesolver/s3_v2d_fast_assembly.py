"""Exact revision-bound stiffness plans for the S3 V2D candidate.

This is a cache/batch orchestration path, not a second mechanics kernel.  A
cold plan calls every candidate's public stiffness method and retains its
individual matrix bytes.  A warm plan reuses those bytes only while the full
bounded eligibility and identity signature remains exactly unchanged.
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

from .e4_pl_s3_v2d_element import (
    BATCH_POLICY_ID,
    FORMULATION_ID,
    NativeParityE4PLS3V2DShellElement,
)
from .e4_pl_s3_v2d_state import canonical_sha256
from .e4_pl_element import QualifiedE4PLShellElement
from .fe_core import FEModel, FEMesh, Material


V2D_GLOBAL_ASSEMBLY_POLICY_ID = "S3_V2D_EXACT_REVISION_BOUND_GLOBAL_CSR_PLAN_V1"


class V2DFastAssemblyError(RuntimeError):
    """A V2D stiffness-plan input or retained plan is incompatible."""


def _candidate(element: Any) -> bool:
    return bool(
        type(element) is NativeParityE4PLS3V2DShellElement
        and element.formulation_id == FORMULATION_ID
        and element.qualified_batch_policy_id == BATCH_POLICY_ID
        and element.legacy_stiffness_batch_eligible is False
        and element.legacy_nonlinear_batch_eligible is False
    )


def v2d_fast_candidate(element: Any) -> bool:
    return _candidate(element)


def v2d_batch_eligibility(model: FEModel, element: Any) -> tuple[bool, str]:
    if not _candidate(element):
        return False, "not_exact_v2d_candidate"
    if element.shell_section is not None:
        return False, "generalized_section"
    if element.material_direction is not None or float(element.material_angle_deg) != 0.0:
        return False, "oriented_material"
    if int(element.director_polarity) != 1:
        return False, "director_reversal"
    if float(element.reference_surface_offset) != 0.0:
        return False, "reference_surface_offset"
    material = model.get_material(element.material_name)
    if type(material) is not Material:
        return False, "nonexact_material"
    if getattr(material, "hardening_curve", None) is not None:
        return False, "material_history"
    if getattr(material, "hill_yield", None) is not None:
        return False, "anisotropic_yield"
    return True, "eligible_v2d_reference_elastic"


def _finite_scalar(value: Any, label: str) -> tuple[str, float]:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V2DFastAssemblyError(f"V2D {label} is not a scalar")
    made = float(value)
    if not math.isfinite(made):
        raise V2DFastAssemblyError(f"V2D {label} is nonfinite")
    return type(value).__qualname__, made


def _signature(
    model: FEModel, items: Sequence[Tuple[int, Any]]
) -> tuple[Any, ...]:
    if type(model) is not FEModel or type(model.mesh) is not FEMesh:
        raise V2DFastAssemblyError("V2D plan requires exact FEModel/FEMesh")
    mesh = model.mesh
    namespace = object.__getattribute__(mesh, "__dict__")
    token = namespace.get("_qualified_direct_state_token")
    if not isinstance(token, list) or len(token) != 1 or type(token[0]) is not int:
        raise V2DFastAssemblyError("V2D plan epoch authority is invalid")
    records: list[Any] = []
    for raw_id, element in items:
        if not _candidate(element):
            continue
        eligible, reason = v2d_batch_eligibility(model, element)
        if not eligible:
            raise V2DFastAssemblyError(f"V2D plan candidate is ineligible: {reason}")
        element_id = int(raw_id)
        if mesh.elements.get(element_id) is not element or element_id != element.element_id:
            raise V2DFastAssemblyError("V2D plan registry identity changed")
        coordinates = np.asarray(element.get_node_coordinates(mesh), dtype=np.float64)
        if coordinates.shape != (3, 3) or not np.all(np.isfinite(coordinates)):
            raise V2DFastAssemblyError("V2D plan geometry is invalid")
        material = model.get_material(element.material_name)
        material_record = tuple(
            (name, _finite_scalar(getattr(material, name), name))
            for name in (
                "elastic_modulus",
                "poisson_ratio",
                "density",
                "yield_stress",
            )
        )
        records.append(
            (
                element_id,
                id(element),
                tuple(int(value) for value in element.node_ids),
                np.ascontiguousarray(coordinates).tobytes(order="C"),
                _finite_scalar(element.thickness, "thickness"),
                np.asarray(element.reference_normal, dtype=np.float64).tobytes(order="C"),
                id(material),
                material_record,
                canonical_sha256(element.to_dict()),
            )
        )
    if not records:
        raise V2DFastAssemblyError("V2D plan has no eligible candidates")
    revisions = mesh.revision_signature()
    return (
        id(model),
        id(mesh),
        id(mesh.nodes),
        id(mesh.elements),
        id(token),
        token[0],
        tuple(int(revisions.get(name, 0)) for name in ("topology", "geometry", "material")),
        tuple(records),
    )


def _readonly(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.float64).reshape(18, 18)


@dataclass(frozen=True)
class PreparedV2DStiffness:
    matrices: Mapping[int, np.ndarray]
    element_ids: tuple[int, ...]
    matrix_payload_sha256: str
    matrices_prevalidated: bool
    revision_key: tuple[int, int, int]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "element_count": len(self.element_ids),
            "element_ids": list(self.element_ids),
            "formulation_id": FORMULATION_ID,
            "matrix_payload_sha256": self.matrix_payload_sha256,
            "matrix_shape_finite_symmetry_prevalidated": self.matrices_prevalidated,
            "path": "exact_revision_bound_per_element_stiffness_plan",
            "policy_id": BATCH_POLICY_ID,
            "revision_key": list(self.revision_key),
            "speedup_claimed": False,
        }


@dataclass(frozen=True)
class PreparedV2DGlobalStiffness:
    data_bytes: bytes
    indices_bytes: bytes
    indices_dtype: str
    indptr_bytes: bytes
    indptr_dtype: str
    info_bytes: bytes
    shape: tuple[int, int]


def _global_fast_signature(
    model: FEModel, items: Sequence[Tuple[int, Any]]
) -> tuple[Any, ...] | None:
    if type(model) is not FEModel or type(model.mesh) is not FEMesh:
        return None
    frozen = tuple(items)
    if not frozen or not any(_candidate(element) for _element_id, element in frozen):
        return None
    if not all(
        type(element)
        in {QualifiedE4PLShellElement, NativeParityE4PLS3V2DShellElement}
        for _element_id, element in frozen
    ):
        return None
    mesh = model.mesh
    namespace = object.__getattribute__(mesh, "__dict__")
    if namespace.get("element_activity") is not None:
        return None
    token = namespace.get("_qualified_direct_state_token")
    revisions = namespace.get("revisions")
    if (
        not isinstance(token, list)
        or len(token) != 1
        or type(token[0]) is not int
        or type(revisions) is not dict
    ):
        return None
    candidate_items = tuple(
        (int(element_id), element)
        for element_id, element in frozen
        if _candidate(element)
    )
    if any(not v2d_batch_eligibility(model, element)[0] for _, element in candidate_items):
        return None
    material_signatures = []
    for name in sorted({element.material_name for _, element in candidate_items}):
        material = model.get_material(name)
        material_signatures.append(
            (
                name,
                id(material),
                tuple(
                    (field, _finite_scalar(getattr(material, field), field))
                    for field in (
                        "elastic_modulus",
                        "poisson_ratio",
                        "density",
                        "yield_stress",
                    )
                ),
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
        tuple(
            int(revisions.get(name, 0))
            for name in ("topology", "geometry", "material")
        ),
        tuple(
            (int(element_id), id(element), type(element))
            for element_id, element in frozen
        ),
        tuple(material_signatures),
    )


def _make_global_plan_manager() -> tuple[Any, Any]:
    records: dict[int, tuple[Any, tuple[Any, ...], PreparedV2DGlobalStiffness]] = {}

    def discard(mesh_id: int, observed: Any) -> None:
        current = records.get(mesh_id)
        if current is not None and current[0] is observed:
            records.pop(mesh_id, None)

    def lookup(
        model: FEModel, items: Sequence[Tuple[int, Any]]
    ) -> PreparedV2DGlobalStiffness | None:
        signature = _global_fast_signature(model, items)
        if signature is None:
            return None
        current = records.get(id(model.mesh))
        if current is None:
            return None
        reference, expected_signature, plan = current
        if (
            reference() is not model.mesh
            or signature != expected_signature
            or type(plan) is not PreparedV2DGlobalStiffness
            or not all(
                type(value) is bytes
                for value in (
                    plan.data_bytes,
                    plan.indices_bytes,
                    plan.indptr_bytes,
                    plan.info_bytes,
                )
            )
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
            raise V2DFastAssemblyError("V2D global stiffness output is incompatible")
        owned_info = dict(info)
        owned_info["assembly_time"] = 0.0
        owned_info["element_times"] = {
            str(element_id): 0.0 for element_id, _element in items
        }
        info_bytes = (
            json.dumps(
                owned_info,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        plan = PreparedV2DGlobalStiffness(
            data_bytes=np.ascontiguousarray(data).tobytes(order="C"),
            indices_bytes=np.ascontiguousarray(indices).tobytes(order="C"),
            indices_dtype=indices.dtype.str,
            indptr_bytes=np.ascontiguousarray(indptr).tobytes(order="C"),
            indptr_dtype=indptr.dtype.str,
            info_bytes=info_bytes,
            shape=(int(shape[0]), int(shape[1])),
        )
        mesh_id = id(model.mesh)
        reference = weakref.ref(
            model.mesh,
            lambda observed, expected=mesh_id: discard(expected, observed),
        )
        records[mesh_id] = (reference, signature, plan)

    return lookup, bind


lookup_v2d_global_stiffness_plan, bind_v2d_global_stiffness_plan = (
    _make_global_plan_manager()
)
del _make_global_plan_manager


def _make_plan_manager() -> Any:
    records: dict[int, tuple[Any, tuple[Any, ...], PreparedV2DStiffness, tuple[tuple[int, bytes], ...]]] = {}

    def discard(mesh_id: int, observed: Any) -> None:
        current = records.get(mesh_id)
        if current is not None and current[0] is observed:
            records.pop(mesh_id, None)

    def get(
        model: FEModel, items: Sequence[Tuple[int, Any]]
    ) -> tuple[PreparedV2DStiffness, bool]:
        frozen = tuple(items)
        signature = _signature(model, frozen)
        mesh_id = id(model.mesh)
        current = records.get(mesh_id)
        if current is not None:
            reference, expected, plan, payloads = current
            if reference() is model.mesh and signature == expected:
                if tuple(plan.element_ids) == tuple(value[0] for value in payloads):
                    return plan, True
            records.pop(mesh_id, None)

        matrices: dict[int, np.ndarray] = {}
        payload_rows: list[tuple[int, bytes]] = []
        for raw_id, element in frozen:
            if not _candidate(element):
                continue
            eligible, _reason = v2d_batch_eligibility(model, element)
            if not eligible:
                continue
            matrix = np.asarray(
                element.compute_stiffness_matrix(
                    model.mesh, model.get_material(element.material_name)
                ),
                dtype=np.float64,
            )
            scale = max(float(np.linalg.norm(matrix, ord="fro")), 1.0)
            if (
                matrix.shape != (18, 18)
                or not np.all(np.isfinite(matrix))
                or float(np.linalg.norm(matrix - matrix.T, ord="fro")) / scale > 1.0e-12
            ):
                raise V2DFastAssemblyError("V2D plan matrix is incompatible")
            payload = np.ascontiguousarray(matrix).tobytes(order="C")
            element_id = int(raw_id)
            payload_rows.append((element_id, payload))
            matrices[element_id] = _readonly(payload)
        payloads = tuple(payload_rows)
        if not payloads:
            raise V2DFastAssemblyError("V2D plan captured no matrices")
        digest = hashlib.sha256(
            b"".join(payload for _element_id, payload in payloads)
        ).hexdigest().upper()
        revisions = model.mesh.revision_signature()
        plan = PreparedV2DStiffness(
            matrices=MappingProxyType(matrices),
            element_ids=tuple(element_id for element_id, _payload in payloads),
            matrix_payload_sha256=digest,
            matrices_prevalidated=True,
            revision_key=tuple(
                int(revisions.get(name, 0))
                for name in ("topology", "geometry", "material")
            ),
        )
        reference = weakref.ref(
            model.mesh,
            lambda observed, expected=mesh_id: discard(expected, observed),
        )
        records[mesh_id] = (reference, signature, plan, payloads)
        return plan, False

    return get


get_v2d_stiffness_plan = _make_plan_manager()
del _make_plan_manager


__all__ = [
    "PreparedV2DGlobalStiffness",
    "PreparedV2DStiffness",
    "V2D_GLOBAL_ASSEMBLY_POLICY_ID",
    "V2DFastAssemblyError",
    "bind_v2d_global_stiffness_plan",
    "get_v2d_stiffness_plan",
    "lookup_v2d_global_stiffness_plan",
    "v2d_batch_eligibility",
    "v2d_fast_candidate",
]
