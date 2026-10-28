"""Explicit stiffness, mass and load assembly APIs.

This module is the step-3 public assembly interface.  It keeps K, M and F
assembly separate so modal, buckling and nonlinear solvers can choose exactly
which matrices they need without side effects.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from operator import itemgetter
from types import FunctionType, MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from weakref import WeakKeyDictionary, ref

import numpy as np
from scipy import sparse
from scipy.sparse._sparsetools import (
    coo_tocsr as _SCIPY_COO_TO_CSR,
    csr_has_canonical_format as _SCIPY_CSR_HAS_CANONICAL_FORMAT,
    csr_has_sorted_indices as _SCIPY_CSR_HAS_SORTED_INDICES,
    csr_sort_indices as _SCIPY_CSR_SORT_INDICES,
    csr_sum_duplicates as _SCIPY_CSR_SUM_DUPLICATES,
)

from ._qualified_authority_epoch import make_authority_epoch_manager

from .e4_pl_element import (
    QualifiedE4PLShellElement as _QualifiedE4PLShellElement,
    _invalidate_q4_guarded_call_caches as _INVALIDATE_Q4_GUARDED_CACHES,
    _q4_runtime_epoch_manager as _Q4_RUNTIME_EPOCH_MANAGER,
    _q4_trusted_operation_scope as _Q4_TRUSTED_OPERATION_SCOPE,
    _require_q4_fast_base_authority as _EXACT_Q4_FAST_BASE_AUTHORITY,
    _require_q4_cached_stiffness_runtime_epoch_authority as _EXACT_Q4_CACHED_STIFFNESS_EPOCH_GUARD,
    _require_exact_q4_runtime_authority as _EXACT_Q4_RUNTIME_GUARD,
    _try_q4_fast_cached_stiffness as _TRY_Q4_FAST_CACHED_STIFFNESS,
    _try_q4_fast_assembly_cached_stiffness as _TRY_Q4_FAST_ASSEMBLY_CACHED_STIFFNESS,
    _validate_q4_quadrature_authority as _EXACT_Q4_QUADRATURE_GUARD,
)
from .fe_core import (
    DOFManager as _DOFManager,
    FEModel as _FEModel,
    FEMesh as _FEMesh,
    Material as _Material,
    Node as _Node,
    _QualifiedMutationEpoch as _QualifiedMutationEpoch,
    _QualifiedStateMapping as _QualifiedStateMapping,
)
from .e4_pl_s3_element import (
    QualifiedE4PLS3ShellElement as _QualifiedE4PLS3ShellElement,
    _invalidate_s3_guarded_call_caches as _INVALIDATE_S3_GUARDED_CACHES,
    _s3_runtime_epoch_manager as _S3_RUNTIME_EPOCH_MANAGER,
    _s3_trusted_operation_scope as _S3_TRUSTED_OPERATION_SCOPE,
    _require_s3_cached_stiffness_runtime_epoch_authority as _EXACT_S3_CACHED_STIFFNESS_EPOCH_GUARD,
    _require_exact_s3_runtime_authority as _EXACT_S3_RUNTIME_GUARD,
    _require_s3_fast_base_authority as _EXACT_S3_FAST_BASE_AUTHORITY,
    _try_s3_fast_assembly_cached_stiffness as _TRY_S3_FAST_ASSEMBLY_CACHED_STIFFNESS,
    _validate_s3_quadrature_values as _EXACT_S3_QUADRATURE_GUARD,
)
from .e4_pl_s3_state import (
    require_exact_numpy_runtime_authority as _EXACT_NUMPY_RUNTIME_GUARD,
)
from .s3_reference_batch import (
    PreparedReferenceS3Components as _PreparedReferenceS3Components,
    REFERENCE_S3_BATCH_POLICY_ID as _REFERENCE_S3_BATCH_POLICY_ID,
    REFERENCE_S3_FORMULATION_ID as _REFERENCE_S3_FORMULATION_ID,
)
from .element_capabilities import ElementCapabilityError
from .elements import BeamElement as _BeamElement, Element as _Element
if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


class AssemblyError(ValueError):
    """Raised when an element returns an invalid matrix or load contribution."""


_ASSEMBLY_NUMERICAL_EPOCH_MANAGER = make_authority_epoch_manager(
    "qualified sparse assembly runtime"
)
_ASSEMBLY_RUNTIME_MODULE = sys.modules[__name__]
_ASSEMBLY_SPARSE_LINALG = sparse.linalg
_ASSEMBLY_MODULE_ALIASES = {
    "hashlib": hashlib,
    "json": json,
    "np": np,
    "sparse": sparse,
    "time": time,
    "_DOFManager": _DOFManager,
    "_FEModel": _FEModel,
    "_FEMesh": _FEMesh,
    "_Material": _Material,
    "_Node": _Node,
    "_PreparedReferenceS3Components": _PreparedReferenceS3Components,
    "_QualifiedStateMapping": _QualifiedStateMapping,
    "_BeamElement": _BeamElement,
    "_Element": _Element,
}

_EXACT_BEAM_GEOMETRIC_STIFFNESS = type.__getattribute__(
    _BeamElement,
    "__dict__",
)["compute_geometric_stiffness_matrix"]
_ASSEMBLY_MODULE_ALIASES["_EXACT_BEAM_GEOMETRIC_STIFFNESS"] = (
    _EXACT_BEAM_GEOMETRIC_STIFFNESS
)
_EXACT_BEAM_DYNAMIC_LOOKUP_AUTHORITY = (
    *(
        (
            _BeamElement,
            name,
            type.__getattribute__(_BeamElement, "__dict__")[name],
        )
        for name in (
            "_axial_compression_from_state",
            "get_node_coordinates",
            "_beam_frame_and_transform",
            "_geometric_polar_radius_squared",
            "num_nodes",
            "dofs_per_node",
        )
    ),
    (
        _Element,
        "total_dofs",
        type.__getattribute__(_Element, "__dict__")["total_dofs"],
    ),
)
_ASSEMBLY_MODULE_ALIASES["_EXACT_BEAM_DYNAMIC_LOOKUP_AUTHORITY"] = (
    _EXACT_BEAM_DYNAMIC_LOOKUP_AUTHORITY
)
_ASSEMBLY_SPARSE_ALIASES = {
    name: vars(sparse)[name]
    for name in ("coo_matrix", "csr_matrix", "diags", "issparse", "linalg")
}
_ASSEMBLY_SPARSE_LINALG_ALIASES = {
    "norm": vars(_ASSEMBLY_SPARSE_LINALG)["norm"],
}
_EXACT_COO_TO_CSR = next(
    type.__getattribute__(base, "__dict__")["tocsr"]
    for base in type.__getattribute__(
        _ASSEMBLY_SPARSE_ALIASES["coo_matrix"],
        "__mro__",
    )
    if "tocsr" in type.__getattribute__(base, "__dict__")
)
_EXACT_COO_INIT = next(
    type.__getattribute__(base, "__dict__")["__init__"]
    for base in type.__getattribute__(
        _ASSEMBLY_SPARSE_ALIASES["coo_matrix"],
        "__mro__",
    )
    if "__init__" in type.__getattribute__(base, "__dict__")
)
_EXACT_CSR_ELIMINATE_ZEROS = next(
    type.__getattribute__(base, "__dict__")["eliminate_zeros"]
    for base in type.__getattribute__(
        _ASSEMBLY_SPARSE_ALIASES["csr_matrix"],
        "__mro__",
    )
    if "eliminate_zeros" in type.__getattribute__(base, "__dict__")
)
_ASSEMBLY_NUMERICAL_EPOCH_MANAGER.watch_module(
    _ASSEMBLY_RUNTIME_MODULE,
    _ASSEMBLY_MODULE_ALIASES,
)
_ASSEMBLY_NUMERICAL_EPOCH_MANAGER.watch_module(
    sparse,
    _ASSEMBLY_SPARSE_ALIASES,
)
_ASSEMBLY_NUMERICAL_EPOCH_MANAGER.watch_module(
    _ASSEMBLY_SPARSE_LINALG,
    _ASSEMBLY_SPARSE_LINALG_ALIASES,
)


def _require_exact_assembly_numerical_authority() -> None:
    for name, expected in _ASSEMBLY_MODULE_ALIASES.items():
        if vars(_ASSEMBLY_RUNTIME_MODULE).get(name) is not expected:
            raise ValueError(
                f"qualified assembly module authority changed: {name}"
            )
    for name, expected in _ASSEMBLY_SPARSE_ALIASES.items():
        if vars(sparse).get(name) is not expected:
            raise ValueError(
                f"qualified scipy.sparse authority changed: {name}"
            )
    for name, expected in _ASSEMBLY_SPARSE_LINALG_ALIASES.items():
        if vars(_ASSEMBLY_SPARSE_LINALG).get(name) is not expected:
            raise ValueError(
                f"qualified scipy.sparse.linalg authority changed: {name}"
            )
    if (
        _static_mro_attribute(
            _ASSEMBLY_SPARSE_ALIASES["coo_matrix"],
            "tocsr",
        )
        is not _EXACT_COO_TO_CSR
    ):
        raise ValueError("qualified scipy COO conversion authority changed")
    if (
        _static_mro_attribute(
            _ASSEMBLY_SPARSE_ALIASES["csr_matrix"],
            "eliminate_zeros",
        )
        is not _EXACT_CSR_ELIMINATE_ZEROS
    ):
        raise ValueError("qualified scipy CSR authority changed")


_EXACT_ASSEMBLY_NUMERICAL_GUARD = _ASSEMBLY_NUMERICAL_EPOCH_MANAGER.bind(
    _require_exact_assembly_numerical_authority
)


_EXACT_FE_MODEL_GET_MATERIAL = type.__getattribute__(
    _FEModel,
    "__dict__",
)["get_material"]
_EXACT_FE_MESH_REVISION_SIGNATURE = type.__getattribute__(
    _FEMesh,
    "__dict__",
)["revision_signature"]
_EXACT_FE_MESH_GET_NODE = type.__getattribute__(_FEMesh, "__dict__")["get_node"]
_EXACT_FE_MESH_NUM_NODES = type.__getattribute__(_FEMesh, "__dict__")[
    "num_nodes"
]
_EXACT_DOF_MANAGER_TOTAL_DOFS = type.__getattribute__(
    _DOFManager,
    "__dict__",
)["total_dofs"]
_EXACT_DOF_MANAGER_GET_NODE_DOFS = type.__getattribute__(
    _DOFManager,
    "__dict__",
)["get_node_dofs"]
_EXACT_PREPARED_S3_CLASS_NAMESPACE = tuple(
    type.__getattribute__(_PreparedReferenceS3Components, "__dict__").items()
)
_Q4_WARM_ASSEMBLY_OWNED_MESH_CACHE_KEYS = frozenset(
    {
        "_sparsity_cache",
        "_topology_signature_cache",
        "_qualified_s3_reference_stiffness_plan",
        "_recovery_batch_plan",
        "_strict_flat_v2c_mixed_scope_cache_v1",
    }
)


def _require_exact_prepared_s3_class_authority() -> None:
    namespace = type.__getattribute__(_PreparedReferenceS3Components, "__dict__")
    current = tuple(namespace.items())
    if (
        len(current) != len(_EXACT_PREPARED_S3_CLASS_NAMESPACE)
        or any(
            type(name) is not str
            or name != expected_name
            or value is not expected_value
            for (name, value), (expected_name, expected_value) in zip(
                current,
                _EXACT_PREPARED_S3_CLASS_NAMESPACE,
            )
        )
    ):
        raise ValueError("qualified S3 reference-plan authority changed")


def _exact_mapping_items(value: Any) -> tuple[tuple[Any, Any], ...] | None:
    """Return callback-free exact-dict items for one canonical mapping."""

    if type(value) not in {dict, _QualifiedStateMapping}:
        return None
    return tuple(dict.items(value))


def _static_mro_attribute(owner: type[Any], name: str) -> Any:
    """Resolve one descriptor without invoking user-controlled lookup."""

    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return None


def _bind_qualified_assembly_runtime_authority(
    numerical_guard: Any,
    q4_guard: Any,
    s3_guard: Any,
    q4_type: type[Any],
    s3_type: type[Any],
) -> Any:
    """Bind exact runtime authority across warm and scalar shell routes."""

    def require(model: "FEModel", *, context: str) -> None:
        numerical_guard(context=context)
        for element in tuple(model.mesh.elements.values()):
            try:
                if type(element) is q4_type:
                    q4_guard(element, context=context)
                elif type(element) is s3_type:
                    s3_guard(element, context=context)
            except (AttributeError, TypeError, ValueError) as exc:
                raise AssemblyError(
                    f"{context} found incompatible qualified shell authority"
                ) from exc

    return require


_REQUIRE_QUALIFIED_ASSEMBLY_RUNTIME_AUTHORITY = (
    _bind_qualified_assembly_runtime_authority(
        _EXACT_NUMPY_RUNTIME_GUARD,
        _EXACT_Q4_RUNTIME_GUARD,
        _EXACT_S3_RUNTIME_GUARD,
        _QualifiedE4PLShellElement,
        _QualifiedE4PLS3ShellElement,
    )
)


def _make_assembly_operation_authority_holder() -> tuple[Any, Any]:
    """Keep the exact qualified assembly dispatcher outside module state."""

    authority: list[Any] = []

    def install(guard: Any) -> None:
        if authority:
            raise RuntimeError("qualified assembly operation is already bound")
        authority.append(guard)

    def require() -> None:
        if len(authority) != 1:
            raise RuntimeError("qualified assembly operation is not bound")
        authority[0]()

    return install, require


(
    _INSTALL_EXACT_ASSEMBLY_OPERATION_AUTHORITY,
    _REQUIRE_EXACT_ASSEMBLY_OPERATION_AUTHORITY,
) = _make_assembly_operation_authority_holder()


def _make_assembly_execution_plan_registry() -> tuple[Any, Any]:
    """Keep warm execution plans inaccessible from caller-facing leases."""

    plans: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()

    def register(lease: Any, plan: Any) -> None:
        if lease in plans:
            raise RuntimeError("qualified assembly execution plan is already bound")
        plans[lease] = plan

    def lookup(lease: Any) -> Any:
        return plans.get(lease)

    return register, lookup


(
    _REGISTER_QUALIFIED_ASSEMBLY_EXECUTION_PLAN,
    _LOOKUP_QUALIFIED_ASSEMBLY_EXECUTION_PLAN,
) = _make_assembly_execution_plan_registry()


def _bind_qualified_assembly_runtime_lease(
    numerical_guard: Any,
    q4_guard: Any,
    s3_guard: Any,
    q4_type: type[Any],
    s3_type: type[Any],
    q4_manager: Any,
    s3_manager: Any,
    invalidate_q4: Any,
    invalidate_s3: Any,
    q4_cached_epoch_guard: Any,
    q4_fast_base_guard: Any,
    q4_assembly_cached_stiffness: Any,
    s3_cached_epoch_guard: Any,
    s3_fast_base_guard: Any,
    s3_assembly_cached_stiffness: Any,
    assembly_epoch_manager: Any,
    assembly_numerical_guard: Any,
    assembly_operation_guard: Any,
    exact_model_type: type[Any],
    execution_plan_register: Any,
) -> Any:
    """Bind one immutable authority generation across a whole assembly call."""

    cached_accessor_authority = tuple(
        (
            accessor,
            accessor.__code__,
            accessor.__defaults__,
            accessor.__kwdefaults__,
            (
                ()
                if accessor.__kwdefaults__ is None
                else tuple(accessor.__kwdefaults__.items())
            ),
        )
        for accessor in (
            q4_assembly_cached_stiffness,
            s3_assembly_cached_stiffness,
        )
    )
    exact_execution_plan_register = FunctionType(
        execution_plan_register.__code__,
        execution_plan_register.__globals__,
        execution_plan_register.__name__,
        execution_plan_register.__defaults__,
        execution_plan_register.__closure__,
    )
    owned_routing_by_mesh: dict[int, tuple[Any, dict[str, Any]]] = {}
    prepared_s3_by_mesh: dict[int, tuple[Any, dict[str, Any]]] = {}
    exact_assembly_error = AssemblyError
    exact_attribute_error = AttributeError
    exact_bool = bool
    exact_dict_contains = dict.__contains__
    exact_dict_get = dict.get
    exact_dict_pop = dict.pop
    exact_dict_type = dict
    exact_globals = globals
    exact_id = id
    exact_int = int
    exact_len = len
    exact_list_getitem = list.__getitem__
    exact_numpy_isfinite = np.isfinite
    exact_object_getattribute = object.__getattribute__
    exact_runtime_error = RuntimeError
    exact_type = type
    exact_type_error = TypeError
    exact_value_error = ValueError
    module_namespace = exact_globals()
    trusted_element_builtin_shadow_names = (
        "dict",
        "id",
        "int",
        "len",
        "list",
        "object",
        "type",
    )

    def require_no_trusted_element_builtin_shadows() -> None:
        changed = tuple(
            name
            for name in trusted_element_builtin_shadow_names
            if exact_dict_contains(module_namespace, name)
        )
        if changed:
            raise exact_assembly_error(
                "qualified trusted-element builtin authority changed: "
                + ", ".join(changed)
            )
    exact_numpy_coordinate_types = frozenset(
        {
            np.float16,
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.intp,
            np.uintp,
        }
    )

    class PreparedS3PlanDataChanged(ValueError):
        """An owned plan changed and may be safely dropped before execution."""

    class PreparedS3MaterialChanged(ValueError):
        """A supported material edit invalidated a prepared plan preflight."""

    class PreparedS3ExecutionChanged(ValueError):
        """A valid derived component cache replaced a prepared execution record."""

    def capture_coordinate_scalar(value: Any) -> tuple[Any, ...] | None:
        """Capture one exact finite built-in or NumPy coordinate scalar."""

        value_type = type(value)
        if value_type is int:
            return (int, value)
        if value_type is float:
            if not exact_bool(exact_numpy_isfinite(value)):
                return None
            return (float, value)
        if value_type not in exact_numpy_coordinate_types:
            return None
        if not exact_bool(exact_numpy_isfinite(value)):
            return None
        dtype_text = value.dtype.str
        payload = value.tobytes()
        if type(dtype_text) is not str or type(payload) is not bytes:
            return None
        return (value_type, dtype_text, payload)

    def coordinate_scalar_matches(
        current: Any,
        authority: tuple[Any, ...],
    ) -> bool:
        """Compare one coordinate without invoking foreign numeric protocols."""

        expected_type = authority[0]
        if type(current) is not expected_type:
            return False
        if expected_type is int:
            return current == authority[1]
        if expected_type is float:
            return current == authority[1]
        return (
            current.dtype.str == authority[1]
            and current.tobytes() == authority[2]
        )

    def discard_mesh_records(reference: Any, *, mesh_identity: int) -> None:
        current_routing = owned_routing_by_mesh.get(mesh_identity)
        if current_routing is not None and current_routing[0] is reference:
            owned_routing_by_mesh.pop(mesh_identity, None)
        current_s3 = prepared_s3_by_mesh.get(mesh_identity)
        if current_s3 is not None and current_s3[0] is reference:
            prepared_s3_by_mesh.pop(mesh_identity, None)

    def mesh_reference(mesh: Any) -> Any:
        identity = id(mesh)
        return ref(
            mesh,
            lambda reference, *, mesh_identity=identity: discard_mesh_records(
                reference,
                mesh_identity=mesh_identity,
            ),
        )

    def require_cached_accessor_authority() -> None:
        for (
            accessor,
            expected_code,
            expected_defaults,
            expected_kwdefaults,
            expected_kw_items,
        ) in cached_accessor_authority:
            current_kwdefaults = accessor.__kwdefaults__
            if (
                accessor.__code__ is not expected_code
                or accessor.__defaults__ is not expected_defaults
                or current_kwdefaults is not expected_kwdefaults
                or (
                    expected_kwdefaults is not None
                    and (
                        type(current_kwdefaults) is not dict
                        or len(current_kwdefaults) != len(expected_kw_items)
                        or any(
                            name not in current_kwdefaults
                            or current_kwdefaults[name] is not expected
                            for name, expected in expected_kw_items
                        )
                    )
                )
            ):
                raise ValueError(
                    "qualified assembly cached-stiffness accessor authority changed"
                )

    def capture(
        model: "FEModel",
        *,
        context: str,
        allow_q4_cached_stiffness: bool = False,
    ) -> Any:
        require_no_trusted_element_builtin_shadows()
        # Both family generations precede every model/mesh observation.  A
        # provider that mutates either authority while exposing the model can
        # therefore never redefine the lease's starting generation.
        assembly_start_generation = assembly_epoch_manager.capture_generation()
        q4_start_generation = q4_manager.capture_generation()
        s3_start_generation = s3_manager.capture_generation()
        try:
            numerical_guard(context=context)
            assembly_numerical_guard()
            assembly_operation_guard()
            require_cached_accessor_authority()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise AssemblyError(
                f"{context} found incompatible qualified shell authority"
            ) from exc
        elements: tuple[Any, ...] = ()
        q4_elements: tuple[Any, ...] = ()
        s3_elements: tuple[Any, ...] = ()
        q4_fast_records: tuple[tuple[Any, Any, bytes], ...] = ()
        s3_fast_records: tuple[tuple[Any, Any, bytes], ...] = ()
        s3_plan_fallback_records: tuple[
            tuple[Any, Any, tuple[Any, ...], bytes], ...
        ] = ()
        s3_fast_reference_plan: Any = None
        s3_fast_reference_snapshot: dict[str, Any] | None = None
        s3_prepared_authority: dict[str, Any] | None = None
        s3_fast_candidate_seen = False
        q4_fast_mesh_token: tuple[Any, type[Any], int] | None = None
        q4_fast_plan: dict[str, Any] | None = None
        qualified_input_plan: dict[str, Any] | None = None
        owned_execution_authority: dict[str, Any] | None = None

        def invalidate_s3_reference_plan() -> None:
            mesh = (
                q4_fast_plan["mesh"]
                if q4_fast_plan is not None
                else qualified_input_plan["mesh"]
                if qualified_input_plan is not None
                else None
            )
            if exact_type(mesh) is _FEMesh:
                namespace = exact_object_getattribute(mesh, "__dict__")
                if exact_type(namespace) is exact_dict_type:
                    exact_dict_pop(
                        namespace,
                        "_qualified_s3_reference_stiffness_plan",
                        None,
                    )
                exact_dict_pop(prepared_s3_by_mesh, exact_id(mesh), None)

        def canonical_q4_total_bytes(value: Any) -> bytes:
            if type(value) is bytes:
                if len(value) != 24 * 24 * 8:
                    raise ValueError(
                        "qualified Q4 cached assembly bytes are incompatible"
                    )
                return value
            if (
                type(value) is not np.ndarray
                or value.dtype != np.dtype(np.float64)
                or value.shape != (24, 24)
                or value.strides != (192, 8)
                or not value.flags.c_contiguous
                or value.flags.writeable
            ):
                raise ValueError(
                    "qualified Q4 cached assembly total is incompatible"
                )
            payload = memoryview(value).cast("B").tobytes()
            if len(payload) != 24 * 24 * 8:
                raise ValueError(
                    "qualified Q4 cached assembly bytes are incompatible"
                )
            return payload

        def canonical_s3_total_bytes(value: Any) -> bytes:
            if type(value) is not bytes or len(value) != 18 * 18 * 8:
                raise ValueError(
                    "qualified S3 cached assembly bytes are incompatible"
                )
            return value

        def capture_owned_routing(
            mesh: Any,
            mesh_namespace: dict[str, Any],
            element_items: tuple[tuple[int, Any], ...],
            token: Any,
        ) -> dict[str, Any] | None:
            """Capture callback-free node/DOF routing for an all-qualified mesh."""

            if not element_items or not all(
                type(element_id) is int
                and type(element) in {q4_type, s3_type}
                for element_id, element in element_items
            ):
                return None
            mesh_identity = id(mesh)
            cached_entry = owned_routing_by_mesh.get(mesh_identity)
            if cached_entry is not None:
                cached_reference, cached_routing = cached_entry
                if cached_reference() is mesh:
                    cached_plan = {
                        "routing": cached_routing,
                        "mesh_namespace": mesh_namespace,
                        "token": token,
                    }
                    try:
                        require_owned_routing(cached_plan)
                        cached_elements = cached_routing["element_records"]
                        if (
                            len(element_items) != len(cached_elements)
                            or any(
                                current_id != expected[0]
                                or current_element is not expected[1]
                                for (current_id, current_element), expected in zip(
                                    element_items,
                                    cached_elements,
                                )
                            )
                        ):
                            raise ValueError(
                                "qualified assembly element mapping changed"
                            )
                    except ValueError:
                        if (
                            token is cached_routing["token"]
                            and type(token) is _QualifiedMutationEpoch
                            and len(token) == 1
                            and int(list.__getitem__(token, 0))
                            == cached_routing["token_value"]
                        ):
                            # An unchanged monotonic token cannot authorize a
                            # new baseline.  This is an untracked raw mutation,
                            # so fail closed instead of normalizing it into the
                            # persistent execution record.
                            raise
                        owned_routing_by_mesh.pop(mesh_identity, None)
                        prepared_s3_by_mesh.pop(mesh_identity, None)
                    else:
                        return cached_routing
                else:
                    owned_routing_by_mesh.pop(mesh_identity, None)
                    prepared_s3_by_mesh.pop(mesh_identity, None)
            nodes = dict.get(mesh_namespace, "nodes")
            dof_manager = dict.get(mesh_namespace, "dof_manager")
            if (
                type(nodes) is not _QualifiedStateMapping
                or type(dof_manager) is not _DOFManager
            ):
                return None
            nodes_namespace = object.__getattribute__(nodes, "__dict__")
            dof_namespace = object.__getattribute__(dof_manager, "__dict__")
            if (
                type(nodes_namespace) is not dict
                or not all(type(name) is str for name in nodes_namespace)
                or any(name in nodes_namespace for name in ("get", "items", "values"))
                or dict.get(nodes_namespace, "_qualified_token") is not token
                or dict.get(nodes_namespace, "_qualified_kind") != "node"
                or _static_mro_attribute(_QualifiedStateMapping, "get")
                is not dict.get
                or _static_mro_attribute(_QualifiedStateMapping, "items")
                is not dict.items
                or _static_mro_attribute(_QualifiedStateMapping, "values")
                is not dict.values
                or type(dof_namespace) is not dict
                or not all(type(name) is str for name in dof_namespace)
                or "total_dofs" in dof_namespace
                or "get_node_dofs" in dof_namespace
                or type.__getattribute__(_DOFManager, "__dict__").get(
                    "total_dofs"
                )
                is not _EXACT_DOF_MANAGER_TOTAL_DOFS
                or type.__getattribute__(_DOFManager, "__dict__").get(
                    "get_node_dofs"
                )
                is not _EXACT_DOF_MANAGER_GET_NODE_DOFS
            ):
                return None

            node_to_dof = dict.get(dof_namespace, "_node_to_dof")
            dof_to_node = dict.get(dof_namespace, "_dof_to_node")
            dof_to_local = dict.get(dof_namespace, "_dof_to_local")
            total_dofs = dict.get(dof_namespace, "_total_dofs")
            constrained_dofs = dict.get(dof_namespace, "_constrained_dofs")
            if (
                type(node_to_dof) is not dict
                or type(dof_to_node) is not dict
                or type(dof_to_local) is not dict
                or type(total_dofs) is not int
                or total_dofs < 0
                or type(constrained_dofs) is not set
            ):
                return None

            node_records: list[tuple[Any, ...]] = []
            node_dofs: dict[int, tuple[int, ...]] = {}
            node_items = tuple(dict.items(nodes))
            for node_id, node in node_items:
                if type(node_id) is not int or type(node) is not _Node:
                    return None
                namespace = object.__getattribute__(node, "__dict__")
                if (
                    type(namespace) is not dict
                    or not all(type(name) is str for name in namespace)
                    or "coords" in namespace
                ):
                    return None
                stored_id = dict.get(namespace, "id")
                x = dict.get(namespace, "x")
                y = dict.get(namespace, "y")
                z = dict.get(namespace, "z")
                x_authority = capture_coordinate_scalar(x)
                y_authority = capture_coordinate_scalar(y)
                z_authority = capture_coordinate_scalar(z)
                revision = dict.get(namespace, "_coordinate_revision")
                dofs = dict.get(namespace, "dofs")
                if (
                    type(stored_id) is not int
                    or stored_id != node_id
                    or x_authority is None
                    or y_authority is None
                    or z_authority is None
                    or type(revision) is not int
                    or type(dofs) is not list
                    or len(dofs) != 6
                    or not all(type(dof) is int for dof in dofs)
                ):
                    return None
                coordinates_are_builtin = (
                    x_authority[0] in {int, float}
                    and y_authority[0] in {int, float}
                    and z_authority[0] in {int, float}
                )
                dof_values = tuple(dofs)
                manager_dofs = dict.get(node_to_dof, node_id)
                if (
                    type(manager_dofs) is not list
                    or len(manager_dofs) != len(dof_values)
                    or tuple(manager_dofs) != dof_values
                ):
                    return None
                node_dofs[node_id] = tuple(manager_dofs)
                node_records.append(
                    (
                        node_id,
                        node,
                        namespace,
                        tuple(namespace),
                        stored_id,
                        x_authority,
                        y_authority,
                        z_authority,
                        coordinates_are_builtin,
                        revision,
                        dofs,
                        dof_values,
                    )
                )

            node_to_dof_items = tuple(
                (node_id, dofs, tuple(dofs))
                for node_id, dofs in dict.items(node_to_dof)
                if type(node_id) is int and type(dofs) is list
            )
            if (
                len(node_to_dof_items) != len(node_to_dof)
                or tuple(node_dofs) != tuple(node_id for node_id, *_ in node_to_dof_items)
                or any(node_dofs[node_id] != values for node_id, _dofs, values in node_to_dof_items)
                or not all(
                    type(dof) is int
                    and type(node_id) is int
                    and type(dict.get(dof_to_local, dof)) is int
                    and dict.get(dof_to_node, dof) == node_id
                    for node_id, values in node_dofs.items()
                    for dof in values
                )
                or len(dof_to_node) != total_dofs
                or len(dof_to_local) != total_dofs
                or set(dof_to_node) != set(range(total_dofs))
                or set(dof_to_local) != set(range(total_dofs))
                or not all(type(dof) is int for dof in constrained_dofs)
            ):
                return None

            element_records: list[tuple[Any, ...]] = []
            rows: list[int] = []
            cols: list[int] = []
            signature_elements: list[dict[str, Any]] = []
            for element_id, element in element_items:
                namespace = object.__getattribute__(element, "__dict__")
                if type(namespace) is not dict or not all(
                    type(name) is str for name in namespace
                ):
                    return None
                stored_id = dict.get(namespace, "element_id")
                node_ids = dict.get(namespace, "node_ids")
                material_name = dict.get(namespace, "material_name")
                expected_nodes = 4 if type(element) is q4_type else 3
                if (
                    type(stored_id) is not int
                    or stored_id != element_id
                    or type(node_ids) is not tuple
                    or len(node_ids) != expected_nodes
                    or not all(type(node_id) is int for node_id in node_ids)
                    or type(material_name) is not str
                ):
                    return None
                node_id_values = tuple(node_ids)
                if any(node_id not in node_dofs for node_id in node_id_values):
                    return None
                dof_values = tuple(
                    dof
                    for node_id in node_id_values
                    for dof in node_dofs[node_id]
                )
                local_size = len(dof_values)
                rows.extend(
                    dof
                    for dof in dof_values
                    for _ in range(local_size)
                )
                cols.extend(dof_values * local_size)
                element_records.append(
                    (
                        element_id,
                        element,
                        namespace,
                        stored_id,
                        node_ids,
                        node_id_values,
                        material_name,
                        dof_values,
                    )
                )
                signature_elements.append(
                    {
                        "id": element_id,
                        "class": type(element).__name__,
                        "node_ids": list(node_id_values),
                        "dofs": list(dof_values),
                    }
                )

            revisions = dict.get(mesh_namespace, "revisions")
            if type(revisions) is not dict or not all(
                type(name) is str and type(value) is int
                for name, value in dict.items(revisions)
            ):
                return None
            revision_items = tuple(dict.items(revisions))
            signature_payload = {
                "matrix_type": "stiffness",
                "topology_revision": dict.get(revisions, "topology", 0),
                "mpc_revision": dict.get(revisions, "mpc", 0),
                "elements": signature_elements,
            }
            signature = hashlib.sha256(
                json.dumps(
                    signature_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            rows_values = tuple(rows)
            cols_values = tuple(cols)
            routing = {
                "token": token,
                "token_value": int(list.__getitem__(token, 0)),
                "nodes": nodes,
                "nodes_namespace": nodes_namespace,
                "nodes_mapping_keys": tuple(nodes_namespace),
                "node_items": node_items,
                "node_records": tuple(node_records),
                "dof_manager": dof_manager,
                "dof_namespace": dof_namespace,
                "dof_keys": tuple(dof_namespace),
                "node_to_dof": node_to_dof,
                "node_to_dof_items": node_to_dof_items,
                "dof_to_node": dof_to_node,
                "dof_to_node_items": tuple(dict.items(dof_to_node)),
                "dof_to_local": dof_to_local,
                "dof_to_local_items": tuple(dict.items(dof_to_local)),
                "constrained_dofs": constrained_dofs,
                "constrained_values": frozenset(constrained_dofs),
                "total_dofs": total_dofs,
                "element_records": tuple(element_records),
                "rows": rows_values,
                "cols": cols_values,
                "rows_bytes": np.asarray(
                    rows_values,
                    dtype=np.intp,
                ).tobytes(order="C"),
                "cols_bytes": np.asarray(
                    cols_values,
                    dtype=np.intp,
                ).tobytes(order="C"),
                "entry_count": len(rows_values),
                "revisions": revisions,
                "revision_items": revision_items,
                "signature": signature,
                "has_s3": any(
                    type(element) is s3_type
                    for _element_id, element in element_items
                ),
            }
            owned_routing_by_mesh[mesh_identity] = (
                mesh_reference(mesh),
                routing,
            )
            return routing

        def require_owned_routing(plan: dict[str, Any]) -> None:
            routing = plan.get("routing")
            if routing is None:
                return
            mesh_namespace = plan["mesh_namespace"]
            nodes = routing["nodes"]
            nodes_namespace = routing["nodes_namespace"]
            dof_manager = routing["dof_manager"]
            dof_namespace = routing["dof_namespace"]
            if (
                dict.get(mesh_namespace, "nodes") is not nodes
                or type(nodes) is not _QualifiedStateMapping
                or object.__getattribute__(nodes, "__dict__") is not nodes_namespace
                or tuple(nodes_namespace) != routing["nodes_mapping_keys"]
                or any(name in nodes_namespace for name in ("get", "items", "values"))
                or dict.get(nodes_namespace, "_qualified_token") is not plan["token"]
                or dict.get(nodes_namespace, "_qualified_kind") != "node"
                or _static_mro_attribute(_QualifiedStateMapping, "get") is not dict.get
                or _static_mro_attribute(_QualifiedStateMapping, "items") is not dict.items
                or _static_mro_attribute(_QualifiedStateMapping, "values") is not dict.values
                or dict.get(mesh_namespace, "dof_manager") is not dof_manager
                or type(dof_manager) is not _DOFManager
                or object.__getattribute__(dof_manager, "__dict__") is not dof_namespace
                or tuple(dof_namespace) != routing["dof_keys"]
                or "total_dofs" in dof_namespace
                or "get_node_dofs" in dof_namespace
                or type.__getattribute__(_DOFManager, "__dict__").get("total_dofs")
                is not _EXACT_DOF_MANAGER_TOTAL_DOFS
                or type.__getattribute__(_DOFManager, "__dict__").get("get_node_dofs")
                is not _EXACT_DOF_MANAGER_GET_NODE_DOFS
                or dict.get(dof_namespace, "_node_to_dof") is not routing["node_to_dof"]
                or dict.get(dof_namespace, "_dof_to_node") is not routing["dof_to_node"]
                or dict.get(dof_namespace, "_dof_to_local") is not routing["dof_to_local"]
                or dict.get(dof_namespace, "_constrained_dofs") is not routing["constrained_dofs"]
                or dict.get(dof_namespace, "_total_dofs") != routing["total_dofs"]
            ):
                raise ValueError("qualified assembly owned routing changed")
            current_node_items = tuple(dict.items(nodes))
            if (
                len(current_node_items) != len(routing["node_items"])
                or any(
                    current_id != expected_id or current_node is not expected_node
                    for (current_id, current_node), (expected_id, expected_node) in zip(
                        current_node_items,
                        routing["node_items"],
                    )
                )
            ):
                raise ValueError("qualified assembly node mapping changed")
            for (
                node_id,
                node,
                namespace,
                namespace_keys,
                stored_id,
                x_authority,
                y_authority,
                z_authority,
                coordinates_are_builtin,
                revision,
                dofs,
                dof_values,
            ) in routing["node_records"]:
                current_x = dict.get(namespace, "x")
                current_y = dict.get(namespace, "y")
                current_z = dict.get(namespace, "z")
                coordinates_match = (
                    type(current_x) is x_authority[0]
                    and current_x == x_authority[1]
                    and type(current_y) is y_authority[0]
                    and current_y == y_authority[1]
                    and type(current_z) is z_authority[0]
                    and current_z == z_authority[1]
                    if coordinates_are_builtin
                    else coordinate_scalar_matches(current_x, x_authority)
                    and coordinate_scalar_matches(current_y, y_authority)
                    and coordinate_scalar_matches(current_z, z_authority)
                )
                if (
                    type(node) is not _Node
                    or object.__getattribute__(node, "__dict__") is not namespace
                    or tuple(namespace) != namespace_keys
                    or dict.get(namespace, "id") != stored_id
                    or stored_id != node_id
                    or not coordinates_match
                    or dict.get(namespace, "_coordinate_revision") != revision
                    or dict.get(namespace, "dofs") is not dofs
                    or tuple(dofs) != dof_values
                ):
                    raise ValueError("qualified assembly node routing changed")
            if any(
                dict.get(routing["node_to_dof"], node_id) is not dofs
                or tuple(dofs) != values
                for node_id, dofs, values in routing["node_to_dof_items"]
            ):
                raise ValueError("qualified assembly DOF routing changed")
            for element_id, element, namespace, stored_id, node_ids, node_values, material_name, _dofs in routing["element_records"]:
                if (
                    type(element) not in {q4_type, s3_type}
                    or object.__getattribute__(element, "__dict__") is not namespace
                    or dict.get(namespace, "element_id") != stored_id
                    or stored_id != element_id
                    or dict.get(namespace, "node_ids") is not node_ids
                    or tuple(node_ids) != node_values
                    or dict.get(namespace, "material_name") != material_name
                ):
                    raise ValueError("qualified assembly element routing changed")

        def capture_s3_reference_snapshot(
            candidate: Any,
            s3_ids: tuple[int, ...],
        ) -> dict[str, Any] | None:
            _require_exact_prepared_s3_class_authority()
            if type(candidate) is not _PreparedReferenceS3Components:
                return None
            namespace = object.__getattribute__(candidate, "__dict__")
            if type(namespace) is not dict or not all(
                type(name) is str for name in namespace
            ):
                return None
            matrices = dict.get(namespace, "matrices")
            cache_keys = dict.get(namespace, "element_cache_keys")
            batched_ids = dict.get(namespace, "batched_element_ids")
            cached_ids = dict.get(namespace, "cached_element_ids")
            group_ids = dict.get(namespace, "group_element_ids")
            candidate_ids = dict.get(namespace, "candidate_element_ids")
            complete = dict.get(namespace, "complete_eligible_coverage")
            fallback = dict.get(namespace, "fallback_reasons")
            evaluations = dict.get(namespace, "component_evaluation_count")
            revision_key = dict.get(namespace, "revision_key")
            prevalidated = dict.get(namespace, "matrices_prevalidated")
            if (
                type(matrices) is not MappingProxyType
                or type(cache_keys) is not MappingProxyType
                or type(batched_ids) is not tuple
                or type(cached_ids) is not tuple
                or type(group_ids) is not tuple
                or type(candidate_ids) is not tuple
                or candidate_ids != s3_ids
                or complete is not True
                or type(fallback) is not MappingProxyType
                or type(evaluations) is not int
                or type(revision_key) is not tuple
                or type(prevalidated) is not bool
                or not all(element_id in matrices for element_id in s3_ids)
            ):
                return None
            matrix_items = tuple(matrices.items())
            matrix_records: list[tuple[int, Any, bytes]] = []
            for element_id, matrix in matrix_items:
                if (
                    type(element_id) is not int
                    or type(matrix) is not np.ndarray
                    or matrix.dtype.str != "<f8"
                    or matrix.shape != (18, 18)
                    or matrix.strides != (144, 8)
                    or not matrix.flags.c_contiguous
                    or matrix.flags.writeable
                ):
                    return None
                base = matrix
                seen: set[int] = set()
                while type(base) is np.ndarray:
                    if id(base) in seen or base.flags.writeable:
                        return None
                    seen.add(id(base))
                    base = base.base
                    if base is None:
                        return None
                if not (
                    type(base) is bytes
                    or (
                        type(base) is memoryview
                        and base.readonly
                        and type(base.obj) is bytes
                    )
                ):
                    return None
                payload = memoryview(matrix).cast("B").tobytes()
                if (
                    len(payload) != 18 * 18 * 8
                    or matrix.dtype.str != "<f8"
                    or matrix.shape != (18, 18)
                    or matrix.strides != (144, 8)
                    or matrix.flags.writeable
                ):
                    return None
                matrix_records.append((element_id, matrix, payload))
            cache_key_items = tuple(cache_keys.items())
            fallback_items = tuple(
                (reason, tuple(element_ids))
                for reason, element_ids in fallback.items()
                if type(reason) is str and type(element_ids) is tuple
            )
            if len(fallback_items) != len(fallback):
                return None
            if batched_ids and cached_ids:
                path = "formulation_native_shared_components_and_exact_cache_reuse"
            elif cached_ids:
                path = "formulation_native_exact_cache_reuse"
            elif batched_ids:
                path = "formulation_native_shared_components"
            else:
                path = "formulation_native_scalar_fallback"
            diagnostics = (
                ("policy_id", _REFERENCE_S3_BATCH_POLICY_ID),
                ("formulation_id", _REFERENCE_S3_FORMULATION_ID),
                ("scope", "reference_elastic_isotropic_positive_winding"),
                ("path", path),
                ("candidate_element_count", len(candidate_ids)),
                ("element_count", len(matrix_items)),
                ("translation_group_element_count", len(batched_ids)),
                ("exact_element_cache_reuse_count", len(cached_ids)),
                ("exact_translation_group_count", len(group_ids)),
                ("component_evaluation_count", evaluations),
                ("matrix_shape_finite_symmetry_prevalidated", prevalidated),
                ("element_ids", tuple(element_id for element_id, _ in matrix_items)),
                ("group_element_ids", tuple(tuple(group) for group in group_ids)),
                ("fallback_reasons", fallback_items),
                ("revision_key", tuple(revision_key)),
                ("parallel_kernel", False),
                ("legacy_stiffness_batch_eligible", False),
                ("legacy_nonlinear_batch_eligible", False),
                ("speedup_claimed", False),
            )
            return {
                "candidate": candidate,
                "namespace": namespace,
                "namespace_keys": tuple(namespace),
                "matrices": matrices,
                "matrix_items": matrix_items,
                "matrix_records": tuple(matrix_records),
                "matrix_payloads": MappingProxyType(
                    {
                        element_id: payload
                        for element_id, _matrix, payload in matrix_records
                    }
                ),
                "cache_keys": cache_keys,
                "cache_key_items": cache_key_items,
                "batched_ids": batched_ids,
                "cached_ids": cached_ids,
                "group_ids": group_ids,
                "candidate_ids": candidate_ids,
                "complete": complete,
                "fallback": fallback,
                "fallback_items": fallback_items,
                "evaluations": evaluations,
                "revision_key": revision_key,
                "prevalidated": prevalidated,
                "diagnostics": diagnostics,
            }

        def require_s3_reference_snapshot(snapshot: dict[str, Any]) -> None:
            _require_exact_prepared_s3_class_authority()
            candidate = snapshot["candidate"]
            namespace = snapshot["namespace"]
            current_matrix_items = tuple(snapshot["matrices"].items())
            current_cache_key_items = tuple(snapshot["cache_keys"].items())
            matrix_items_changed = (
                len(current_matrix_items) != len(snapshot["matrix_items"])
                or any(
                    current_id != expected_id or current_matrix is not expected_matrix
                    for (current_id, current_matrix), (expected_id, expected_matrix) in zip(
                        current_matrix_items,
                        snapshot["matrix_items"],
                    )
                )
            )
            cache_key_items_changed = (
                len(current_cache_key_items) != len(snapshot["cache_key_items"])
                or any(
                    current_id != expected_id or current_key is not expected_key
                    for (current_id, current_key), (expected_id, expected_key) in zip(
                        current_cache_key_items,
                        snapshot["cache_key_items"],
                    )
                )
            )
            if (
                type(candidate) is not _PreparedReferenceS3Components
                or object.__getattribute__(candidate, "__dict__") is not namespace
                or tuple(namespace) != snapshot["namespace_keys"]
                or dict.get(namespace, "matrices") is not snapshot["matrices"]
                or matrix_items_changed
                or any(
                    type(current) is not np.ndarray
                    or current is not expected
                    or current.dtype.str != "<f8"
                    or current.shape != (18, 18)
                    or current.strides != (144, 8)
                    or not current.flags.c_contiguous
                    or current.flags.writeable
                    or memoryview(current).cast("B").tobytes() != payload
                    for _element_id, expected, payload in snapshot[
                        "matrix_records"
                    ]
                    for current in (expected,)
                )
                or dict.get(namespace, "element_cache_keys") is not snapshot["cache_keys"]
                or cache_key_items_changed
                or dict.get(namespace, "batched_element_ids") is not snapshot["batched_ids"]
                or dict.get(namespace, "cached_element_ids") is not snapshot["cached_ids"]
                or dict.get(namespace, "group_element_ids") is not snapshot["group_ids"]
                or dict.get(namespace, "candidate_element_ids") is not snapshot["candidate_ids"]
                or dict.get(namespace, "complete_eligible_coverage") is not snapshot["complete"]
                or dict.get(namespace, "fallback_reasons") is not snapshot["fallback"]
                or tuple(
                    (reason, tuple(element_ids))
                    for reason, element_ids in snapshot["fallback"].items()
                )
                != snapshot["fallback_items"]
                or dict.get(namespace, "component_evaluation_count") != snapshot["evaluations"]
                or dict.get(namespace, "revision_key") is not snapshot["revision_key"]
                or dict.get(namespace, "matrices_prevalidated") is not snapshot["prevalidated"]
            ):
                raise ValueError("qualified S3 reference-plan state changed")

        s3_plan_input_names = (
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
            "reference_surface_offset",
            "_qualified_plan_state_revision",
        )

        def capture_s3_plan_value(value: Any) -> tuple[Any, ...] | None:
            if type(value) is np.ndarray:
                if (
                    value.dtype.str != "<f8"
                    or not value.flags.c_contiguous
                    or value.flags.writeable
                ):
                    return None
                payload = memoryview(value).cast("B").tobytes()
                return (
                    "array",
                    value,
                    value.dtype.str,
                    value.shape,
                    value.strides,
                    payload,
                )
            if type(value) in {type(None), bool, int, float, str, tuple}:
                if type(value) is tuple and not all(
                    type(member) is int for member in value
                ):
                    return None
                return ("value", type(value), value)
            return None

        def capture_s3_plan_element_authority(
            element_items: tuple[tuple[int, Any], ...],
        ) -> tuple[tuple[Any, ...], ...] | None:
            records: list[tuple[Any, ...]] = []
            for element_id, element in element_items:
                if type(element) is not s3_type:
                    continue
                namespace = object.__getattribute__(element, "__dict__")
                if type(namespace) is not dict or not all(
                    type(name) is str for name in namespace
                ):
                    return None
                values: list[tuple[str, tuple[Any, ...]]] = []
                for name in s3_plan_input_names:
                    if name not in namespace:
                        return None
                    authority = capture_s3_plan_value(dict.get(namespace, name))
                    if authority is None:
                        return None
                    values.append((name, authority))
                records.append(
                    (
                        element_id,
                        element,
                        namespace,
                        tuple(namespace),
                        tuple(values),
                    )
                )
            return tuple(records)

        def require_s3_plan_element_authority(
            records: tuple[tuple[Any, ...], ...],
        ) -> None:
            for element_id, element, namespace, namespace_keys, values in records:
                if (
                    type(element_id) is not int
                    or type(element) is not s3_type
                    or object.__getattribute__(element, "__dict__") is not namespace
                    or tuple(namespace) != namespace_keys
                    or dict.get(namespace, "element_id") != element_id
                ):
                    raise ValueError("qualified S3 prepared-plan input changed")
                for name, authority in values:
                    current = dict.get(namespace, name)
                    if authority[0] == "array":
                        (
                            _kind,
                            expected,
                            dtype,
                            shape,
                            strides,
                            payload,
                        ) = authority
                        if (
                            type(current) is not np.ndarray
                            or current.dtype.str != dtype
                            or current.shape != shape
                            or current.strides != strides
                            or not current.flags.c_contiguous
                            or current.flags.writeable
                            or memoryview(current).cast("B").tobytes() != payload
                        ):
                            raise ValueError(
                                "qualified S3 prepared-plan vector input changed"
                            )
                    else:
                        _kind, expected_type, expected = authority
                        if type(current) is not expected_type or current != expected:
                            raise ValueError(
                                "qualified S3 prepared-plan scalar input changed"
                            )

        q4_execution_input_names = (
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
        material_execution_input_names = (
            "name",
            "elastic_modulus",
            "poisson_ratio",
            "density",
            "yield_stress",
            "hardening_curve",
        )
        s3_execution_scalar_names = tuple(
            name
            for name in s3_plan_input_names
            if name
            not in {
                "element_id",
                "node_ids",
                "material_name",
                "reference_normal",
            }
        )
        s3_execution_scalar_getter = itemgetter(*s3_execution_scalar_names)
        material_execution_getter = itemgetter(
            *material_execution_input_names
        )

        def execution_scalar_is_owned(value: Any) -> bool:
            if type(value) in {type(None), bool, int, float, str}:
                return True
            return type(value) is tuple and all(
                execution_scalar_is_owned(member) for member in value
            )

        def owned_immutable_values_match(current: Any, expected: Any) -> bool:
            """Compare exact cache fingerprints without foreign equality hooks."""

            current_type = type(current)
            if current_type is not type(expected):
                return False
            if current_type in {type(None), bool, int, float, str, bytes}:
                return current == expected
            if current_type is tuple:
                return len(current) == len(expected) and all(
                    owned_immutable_values_match(member, expected_member)
                    for member, expected_member in zip(current, expected)
                )
            return current is expected

        def component_guards_match(current: Any, expected: Any) -> bool:
            """Accept a freshly sealed, value-equal exact S3 guard tuple."""

            if current is expected:
                return True
            return bool(
                type(current) is tuple
                and type(expected) is tuple
                and len(current) == len(expected) == 12
                and current[0] is expected[0]
                and type(current[1]) is type(expected[1]) is int
                and current[1] == expected[1]
                and current[2] is expected[2]
                and type(current[3]) is type(expected[3])
                and current[3] == expected[3]
                and current[4] is expected[4]
                and owned_immutable_values_match(current[5], expected[5])
                and owned_immutable_values_match(current[6], expected[6])
                and owned_immutable_values_match(current[7], expected[7])
                and current[8] is expected[8]
                and current[9] is expected[9]
                and current[10] is expected[10]
                and current[11] is expected[11]
            )

        def capture_execution_array(value: Any) -> tuple[Any, ...] | None:
            if (
                type(value) is not np.ndarray
                or value.dtype.str != "<f8"
                or not value.flags.c_contiguous
                or value.flags.writeable
            ):
                return None
            base = value
            bases: list[Any] = []
            seen: set[int] = set()
            while type(base) is np.ndarray:
                if id(base) in seen or base.flags.writeable:
                    return None
                seen.add(id(base))
                base = base.base
                if base is None:
                    return None
                bases.append(base)
            if not (
                type(base) is bytes
                or (
                    type(base) is memoryview
                    and base.readonly
                    and type(base.obj) is bytes
                )
            ):
                return None
            return (
                value,
                value.dtype.str,
                value.shape,
                value.strides,
                tuple(bases),
                base,
            )

        def require_execution_array(
            current: Any,
            authority: tuple[Any, ...],
            *,
            payload: bytes | None = None,
        ) -> None:
            expected, dtype, shape, strides, bases, terminal_base = authority
            if (
                type(current) is not np.ndarray
                or current is not expected
                or current.dtype.str != dtype
                or current.shape != shape
                or current.strides != strides
                or not current.flags.c_contiguous
                or current.flags.writeable
            ):
                raise ValueError("qualified cached execution array changed")
            base: Any = current
            for expected_base in bases:
                if (
                    base.base is not expected_base
                    or (
                        type(expected_base) is np.ndarray
                        and expected_base.flags.writeable
                    )
                ):
                    raise ValueError(
                        "qualified cached execution array base changed"
                    )
                base = expected_base
            if base is not terminal_base:
                raise ValueError("qualified cached execution array base changed")
            if payload is not None and memoryview(current).cast("B").tobytes() != payload:
                raise ValueError("qualified cached execution array bytes changed")

        def capture_execution_authority(
            records: tuple[tuple[Any, Any, bytes], ...],
            *,
            family: str,
        ) -> tuple[tuple[Any, ...], ...]:
            if family != "s3":
                raise ValueError("prepared execution authority is S3-only")
            captured: list[tuple[Any, ...]] = []
            for element, material, expected_bytes in records:
                if type(element) is not s3_type or type(material) is not _Material:
                    raise ValueError("qualified cached execution type changed")
                namespace = object.__getattribute__(element, "__dict__")
                material_namespace = object.__getattribute__(material, "__dict__")
                if (
                    type(namespace) is not dict
                    or type(material_namespace) is not dict
                    or not all(type(name) is str for name in namespace)
                    or not all(type(name) is str for name in material_namespace)
                ):
                    raise ValueError("qualified cached execution namespace changed")
                if any(
                    name not in namespace
                    for name in s3_execution_scalar_names
                ) or "reference_normal" not in namespace:
                    raise ValueError("qualified cached execution input is absent")
                scalar_values = s3_execution_scalar_getter(namespace)
                if not all(
                    execution_scalar_is_owned(value)
                    for value in scalar_values
                ):
                    raise ValueError(
                        "qualified cached execution input is mutable"
                    )
                scalar_types = tuple(map(type, scalar_values))
                normal_authority = capture_execution_array(
                    dict.get(namespace, "reference_normal")
                )
                if normal_authority is None:
                    raise ValueError(
                        "qualified cached execution normal is mutable"
                    )
                normal_payload = memoryview(normal_authority[0]).cast(
                    "B"
                ).tobytes()
                if any(
                    name not in material_namespace
                    for name in material_execution_input_names
                ):
                    raise ValueError("qualified cached material input is absent")
                material_values = material_execution_getter(
                    material_namespace
                )
                if not all(
                    execution_scalar_is_owned(value)
                    for value in material_values
                ):
                    raise ValueError("qualified cached material input is mutable")
                components = dict.get(namespace, "_qualified_components")
                cache_key = dict.get(namespace, "_qualified_cache_key")
                guard = dict.get(namespace, "_qualified_component_guard")
                total = (
                    components.get("total")
                    if type(components) is MappingProxyType
                    else None
                )
                if (
                    type(cache_key) is not tuple
                    or type(guard) is not tuple
                    or type(total) is not np.ndarray
                    or total.dtype.str != "<f8"
                    or total.shape != (18, 18)
                    or total.strides != (144, 8)
                    or not total.flags.c_contiguous
                    or total.flags.writeable
                    or memoryview(total).cast("B").tobytes() != expected_bytes
                ):
                    raise ValueError("qualified cached execution total changed")
                total_authority = capture_execution_array(total)
                if total_authority is None:
                    raise ValueError("qualified cached execution total changed")
                captured.append(
                    (
                        element,
                        namespace,
                        frozenset(namespace),
                        scalar_values,
                        scalar_types,
                        normal_authority,
                        normal_payload,
                        components,
                        cache_key,
                        guard,
                        total_authority,
                        expected_bytes,
                        material,
                        material_namespace,
                        frozenset(material_namespace),
                        material_values,
                        tuple(map(type, material_values)),
                    )
                )
            return tuple(captured)

        def require_execution_authority(
            authority: tuple[tuple[Any, ...], ...],
            *,
            allow_equivalent_guard_rebind: bool = False,
        ) -> tuple[tuple[Any, ...], ...]:
            rebound: list[tuple[Any, ...]] | None = None
            for record_index, record in enumerate(authority):
                (
                element,
                namespace,
                namespace_keys,
                scalar_values,
                scalar_types,
                normal_authority,
                normal_payload,
                components,
                cache_key,
                guard,
                total_authority,
                expected_bytes,
                material,
                material_namespace,
                material_namespace_keys,
                material_values,
                material_types,
                ) = record
                subscriptions = dict.get(
                    namespace,
                    "_qualified_direct_state_tokens",
                )
                has_multiple_owners = (
                    type(subscriptions) is list
                    and len(subscriptions) > 1
                    and all(
                        type(token) is _QualifiedMutationEpoch
                        and len(token) == 1
                        and type(list.__getitem__(token, 0)) is int
                        for token in subscriptions
                    )
                    and len({id(token) for token in subscriptions})
                    == len(subscriptions)
                )
                if (
                    type(element) is not s3_type
                    or object.__getattribute__(element, "__dict__") is not namespace
                    or namespace.keys() != namespace_keys
                    or type(material) is not _Material
                    or object.__getattribute__(material, "__dict__")
                    is not material_namespace
                    or material_namespace.keys() != material_namespace_keys
                ):
                    raise ValueError("qualified cached execution authority changed")
                if not has_multiple_owners:
                    current_components = dict.get(
                        namespace,
                        "_qualified_components",
                    )
                    current_cache_key = dict.get(
                        namespace,
                        "_qualified_cache_key",
                    )
                    current_guard = dict.get(
                        namespace,
                        "_qualified_component_guard",
                    )
                    guard_matches = component_guards_match(current_guard, guard)
                    if (
                        current_components is not components
                        or current_cache_key is not cache_key
                        or not guard_matches
                        or type(components) is not MappingProxyType
                        or components.get("total") is not total_authority[0]
                    ):
                        raise PreparedS3ExecutionChanged(
                            "qualified cached execution cache changed"
                        )
                    if current_guard is not guard:
                        if not allow_equivalent_guard_rebind:
                            raise ValueError(
                                "qualified cached execution guard changed"
                            )
                        if rebound is None:
                            rebound = list(authority)
                        updated = list(record)
                        updated[9] = current_guard
                        rebound[record_index] = tuple(updated)
                current_scalars = s3_execution_scalar_getter(namespace)
                if (
                    tuple(map(type, current_scalars)) != scalar_types
                    or current_scalars != scalar_values
                ):
                    raise ValueError(
                        "qualified cached execution scalar changed"
                    )
                current_normal = dict.get(namespace, "reference_normal")
                if has_multiple_owners:
                    current_normal_authority = capture_execution_array(
                        current_normal
                    )
                    if (
                        current_normal_authority is None
                        or current_normal_authority[1:4]
                        != normal_authority[1:4]
                        or memoryview(current_normal).cast("B").tobytes()
                        != normal_payload
                    ):
                        raise ValueError(
                            "qualified cached execution normal changed"
                        )
                else:
                    require_execution_array(
                        current_normal,
                        normal_authority,
                    )
                if not has_multiple_owners:
                    require_execution_array(
                        total_authority[0],
                        total_authority,
                    )
                current_material_values = material_execution_getter(
                    material_namespace
                )
                if (
                    tuple(map(type, current_material_values))
                    != material_types
                    or current_material_values != material_values
                ):
                    raise PreparedS3MaterialChanged(
                        "qualified cached material input changed"
                    )
            return authority if rebound is None else tuple(rebound)

        def require_execution_node_authority(routing: dict[str, Any]) -> None:
            for (
                node_id,
                node,
                namespace,
                _namespace_keys,
                stored_id,
                x_authority,
                y_authority,
                z_authority,
                coordinates_are_builtin,
                revision,
                _dofs,
                _dof_values,
            ) in routing["node_records"]:
                current_x = dict.get(namespace, "x")
                current_y = dict.get(namespace, "y")
                current_z = dict.get(namespace, "z")
                coordinates_match = (
                    type(current_x) is x_authority[0]
                    and current_x == x_authority[1]
                    and type(current_y) is y_authority[0]
                    and current_y == y_authority[1]
                    and type(current_z) is z_authority[0]
                    and current_z == z_authority[1]
                    if coordinates_are_builtin
                    else coordinate_scalar_matches(current_x, x_authority)
                    and coordinate_scalar_matches(current_y, y_authority)
                    and coordinate_scalar_matches(current_z, z_authority)
                )
                if (
                    type(node) is not _Node
                    or object.__getattribute__(node, "__dict__") is not namespace
                    or dict.get(namespace, "id") != stored_id
                    or stored_id != node_id
                    or not coordinates_match
                    or dict.get(namespace, "_coordinate_revision") != revision
                ):
                    raise ValueError("qualified cached execution node input changed")

        def bind_prepared_s3_snapshot(
            mesh: Any,
            candidate: Any,
            snapshot: dict[str, Any],
            token: Any,
            element_items: tuple[tuple[int, Any], ...],
        ) -> None:
            element_authority = capture_s3_plan_element_authority(element_items)
            plan = q4_fast_plan
            routing = None if plan is None else plan.get("routing")
            execution_records: list[tuple[Any, Any, bytes]] = []
            fallback_records: list[tuple[Any, Any, tuple[Any, ...], bytes]] = []
            cache_keys = snapshot["cache_keys"]
            payloads = snapshot["matrix_payloads"]
            for element_id, element in element_items:
                if type(element) is not s3_type:
                    continue
                material = builtin_material(element)
                expected_key = cache_keys.get(element_id)
                payload = payloads.get(element_id)
                if (
                    material is None
                    or type(expected_key) is not tuple
                    or type(payload) is not bytes
                ):
                    execution_records = []
                    break
                execution_records.append((element, material, payload))
                fallback_records.append(
                    (element, material, expected_key, payload)
                )
            if (
                element_authority is None
                or routing is None
                or len(execution_records) != len(snapshot["candidate_ids"])
            ):
                prepared_s3_by_mesh.pop(id(mesh), None)
                return
            execution_authority = capture_execution_authority(
                tuple(execution_records),
                family="s3",
            )
            mesh_identity = id(mesh)
            prepared_s3_by_mesh[mesh_identity] = (
                mesh_reference(mesh),
                {
                    "candidate": candidate,
                    "snapshot": snapshot,
                    "token": token,
                    "token_value": int(list.__getitem__(token, 0)),
                    "element_authority": element_authority,
                    "routing": routing,
                    "execution_records": tuple(execution_records),
                    "fallback_records": tuple(fallback_records),
                    "execution_authority": execution_authority,
                },
            )

        def lookup_prepared_s3_authority(
            mesh: Any,
            candidate: Any,
            token: Any,
            *,
            routing_prevalidated: bool = False,
            preflight: bool = False,
        ) -> dict[str, Any] | None:
            record = prepared_s3_by_mesh.get(id(mesh))
            if record is None:
                return None
            reference, authority = record
            if reference() is not mesh:
                prepared_s3_by_mesh.pop(id(mesh), None)
                return None
            if (
                candidate is not authority["candidate"]
                or token is not authority["token"]
                or type(token) is not _QualifiedMutationEpoch
                or len(token) != 1
                or int(list.__getitem__(token, 0)) != authority["token_value"]
            ):
                return None
            snapshot = authority["snapshot"]
            try:
                require_s3_reference_snapshot(snapshot)
            except ValueError as exc:
                raise PreparedS3PlanDataChanged(
                    "qualified S3 prepared matrix authority changed"
                ) from exc
            if q4_fast_plan is None or q4_fast_plan.get("routing") is not authority["routing"]:
                raise ValueError("qualified S3 prepared routing changed")
            if not routing_prevalidated:
                require_owned_routing(q4_fast_plan)
            current_execution_authority = require_execution_authority(
                authority["execution_authority"],
                allow_equivalent_guard_rebind=preflight,
            )
            if current_execution_authority is not authority["execution_authority"]:
                authority["execution_authority"] = current_execution_authority
            return authority

        def lookup_prepared_s3_snapshot(
            mesh: Any,
            candidate: Any,
            token: Any,
            *,
            routing_prevalidated: bool = False,
        ) -> dict[str, Any] | None:
            authority = lookup_prepared_s3_authority(
                mesh,
                candidate,
                token,
                routing_prevalidated=routing_prevalidated,
            )
            return None if authority is None else authority["snapshot"]

        def prepared_s3_snapshot_is_current(
            mesh: Any,
            candidate: Any,
            token: Any,
        ) -> bool:
            record = prepared_s3_by_mesh.get(id(mesh))
            if record is None:
                return False
            reference, authority = record
            return bool(
                reference() is mesh
                and candidate is authority["candidate"]
                and token is authority["token"]
                and type(token) is _QualifiedMutationEpoch
                and len(token) == 1
                and int(list.__getitem__(token, 0))
                == authority["token_value"]
            )

        def qualified_builtin_inputs() -> dict[str, Any] | None:
            """Capture callback-free inputs for an exact qualified model.

            Exact FEModel/FEMesh instances remain supported with the ordinary
            providers defined by their classes.  Once a qualified element is
            present, however, an instance or class provider replacement must
            fail closed; falling back through the changed provider would let
            routing differ between qualification discovery and assembly.
            """

            if type(model) is not exact_model_type:
                return None
            model_namespace = object.__getattribute__(model, "__dict__")
            if type(model_namespace) is not dict or not all(
                type(name) is str for name in model_namespace
            ):
                return None
            mesh = dict.get(model_namespace, "mesh")
            if type(mesh) is not _FEMesh:
                return None
            mesh_namespace = object.__getattribute__(mesh, "__dict__")
            if type(mesh_namespace) is not dict or not all(
                type(name) is str for name in mesh_namespace
            ):
                return None
            mapping = dict.get(mesh_namespace, "elements")
            if type(mapping) is not _QualifiedStateMapping:
                return None
            mapping_namespace = object.__getattribute__(mapping, "__dict__")
            if type(mapping_namespace) is not dict or not all(
                type(name) is str for name in mapping_namespace
            ):
                return None
            element_items = tuple(dict.items(mapping))
            if not all(
                type(element_id) is int
                for element_id, _element in element_items
            ):
                return None
            qualified_elements = tuple(
                element
                for _element_id, element in element_items
                if type(element) in {q4_type, s3_type}
            )
            if not qualified_elements:
                return None

            materials = dict.get(model_namespace, "materials")
            current_material = dict.get(model_namespace, "current_material")
            token = dict.get(mesh_namespace, "_qualified_direct_state_token")
            provider_changed = (
                "get_material" in model_namespace
                or type.__getattribute__(exact_model_type, "__dict__").get(
                    "get_material"
                )
                is not _EXACT_FE_MODEL_GET_MATERIAL
                or any(
                    name in mesh_namespace
                    for name in ("get_node", "revision_signature", "num_nodes")
                )
                or type.__getattribute__(_FEMesh, "__dict__").get("get_node")
                is not _EXACT_FE_MESH_GET_NODE
                or type.__getattribute__(_FEMesh, "__dict__").get(
                    "revision_signature"
                )
                is not _EXACT_FE_MESH_REVISION_SIGNATURE
                or type.__getattribute__(_FEMesh, "__dict__").get("num_nodes")
                is not _EXACT_FE_MESH_NUM_NODES
                or any(
                    name in mapping_namespace
                    for name in ("get", "items", "values")
                )
                or _static_mro_attribute(_QualifiedStateMapping, "get")
                is not dict.get
                or _static_mro_attribute(_QualifiedStateMapping, "items")
                is not dict.items
                or _static_mro_attribute(_QualifiedStateMapping, "values")
                is not dict.values
            )
            if provider_changed:
                raise ValueError(
                    "qualified assembly input-provider authority changed"
                )
            if (
                type(materials) is not dict
                or not all(type(name) is str for name in materials)
                or type(current_material) is not str
                or "default" not in materials
                or type(token) is not _QualifiedMutationEpoch
                or len(token) != 1
                or type(list.__getitem__(token, 0)) is not int
                or dict.get(mapping_namespace, "_qualified_token") is not token
                or dict.get(mapping_namespace, "_qualified_kind") != "element"
            ):
                raise ValueError("qualified assembly owned inputs are invalid")

            material_items = tuple(dict.items(materials))
            material_by_element: dict[int, tuple[Any, Any]] = {}
            for element in qualified_elements:
                element_namespace = object.__getattribute__(element, "__dict__")
                if type(element_namespace) is not dict:
                    raise ValueError(
                        "qualified assembly element namespace is invalid"
                    )
                material_name = dict.get(element_namespace, "material_name")
                if type(material_name) is not str:
                    raise ValueError(
                        "qualified assembly material name is invalid"
                    )
                resolved_name = material_name or current_material
                default = dict.get(materials, "default")
                material = dict.get(materials, resolved_name, default)
                if material is None:
                    raise ValueError(
                        "qualified assembly material authority is absent"
                    )
                material_by_element[id(element)] = (element, material)

            return {
                "model_namespace": model_namespace,
                "mesh": mesh,
                "mesh_namespace": mesh_namespace,
                "mapping": mapping,
                "mapping_namespace": mapping_namespace,
                "element_items": element_items,
                "materials": materials,
                "material_items": material_items,
                "current_material": current_material,
                "material_by_element": material_by_element,
                "token": token,
                "token_value": int(list.__getitem__(token, 0)),
            }

        def require_qualified_builtin_inputs() -> None:
            plan = qualified_input_plan
            if plan is None:
                return
            model_namespace = plan["model_namespace"]
            mesh = plan["mesh"]
            mesh_namespace = plan["mesh_namespace"]
            mapping = plan["mapping"]
            mapping_namespace = plan["mapping_namespace"]
            current_items = tuple(dict.items(mapping))
            current_material_items = tuple(dict.items(plan["materials"]))
            if (
                type(model) is not exact_model_type
                or object.__getattribute__(model, "__dict__")
                is not model_namespace
                or "get_material" in model_namespace
                or type.__getattribute__(exact_model_type, "__dict__").get(
                    "get_material"
                )
                is not _EXACT_FE_MODEL_GET_MATERIAL
                or dict.get(model_namespace, "mesh") is not mesh
                or dict.get(model_namespace, "materials")
                is not plan["materials"]
                or dict.get(model_namespace, "current_material")
                != plan["current_material"]
                or type(mesh) is not _FEMesh
                or object.__getattribute__(mesh, "__dict__") is not mesh_namespace
                or any(
                    name in mesh_namespace
                    for name in ("get_node", "revision_signature", "num_nodes")
                )
                or type.__getattribute__(_FEMesh, "__dict__").get("get_node")
                is not _EXACT_FE_MESH_GET_NODE
                or type.__getattribute__(_FEMesh, "__dict__").get(
                    "revision_signature"
                )
                is not _EXACT_FE_MESH_REVISION_SIGNATURE
                or type.__getattribute__(_FEMesh, "__dict__").get("num_nodes")
                is not _EXACT_FE_MESH_NUM_NODES
                or dict.get(mesh_namespace, "elements") is not mapping
                or dict.get(mesh_namespace, "_qualified_direct_state_token")
                is not plan["token"]
                or type(mapping) is not _QualifiedStateMapping
                or object.__getattribute__(mapping, "__dict__")
                is not mapping_namespace
                or any(
                    name in mapping_namespace
                    for name in ("get", "items", "values")
                )
                or _static_mro_attribute(_QualifiedStateMapping, "get")
                is not dict.get
                or _static_mro_attribute(_QualifiedStateMapping, "items")
                is not dict.items
                or _static_mro_attribute(_QualifiedStateMapping, "values")
                is not dict.values
                or dict.get(mapping_namespace, "_qualified_token")
                is not plan["token"]
                or dict.get(mapping_namespace, "_qualified_kind") != "element"
                or int(list.__getitem__(plan["token"], 0))
                != plan["token_value"]
                or len(current_items) != len(plan["element_items"])
                or any(
                    current_id != expected_id
                    or current_element is not expected_element
                    for (current_id, current_element), (
                        expected_id,
                        expected_element,
                    ) in zip(current_items, plan["element_items"])
                )
                or len(current_material_items) != len(plan["material_items"])
                or any(
                    current_name != expected_name
                    or current_material is not expected_material
                    for (current_name, current_material), (
                        expected_name,
                        expected_material,
                    ) in zip(current_material_items, plan["material_items"])
                )
            ):
                raise ValueError("qualified assembly owned inputs changed")

        def raw_plan_candidate() -> dict[str, Any] | None:
            """Capture provider-free exact builtin model state for Q4 reuse."""

            if type(model) is not exact_model_type:
                return None
            model_namespace = object.__getattribute__(model, "__dict__")
            if (
                type(model_namespace) is not dict
                or not all(type(name) is str for name in model_namespace)
                or "get_material" in model_namespace
                or type.__getattribute__(exact_model_type, "__dict__").get(
                    "get_material"
                )
                is not _EXACT_FE_MODEL_GET_MATERIAL
            ):
                return None
            mesh = dict.get(model_namespace, "mesh")
            materials = dict.get(model_namespace, "materials")
            current_material = dict.get(model_namespace, "current_material")
            if (
                type(mesh) is not _FEMesh
                or type(materials) is not dict
                or not all(type(name) is str for name in materials)
                or type(current_material) is not str
                or "default" not in materials
            ):
                return None
            mesh_namespace = object.__getattribute__(mesh, "__dict__")
            if (
                type(mesh_namespace) is not dict
                or not all(type(name) is str for name in mesh_namespace)
                or any(
                    name in mesh_namespace
                    for name in ("get_node", "revision_signature", "num_nodes")
                )
                or (
                    dict.get(mesh_namespace, "_sparsity_cache") is not None
                    and type(dict.get(mesh_namespace, "_sparsity_cache")) is not dict
                )
                or (
                    dict.get(mesh_namespace, "_topology_signature_cache") is not None
                    and type(
                        dict.get(mesh_namespace, "_topology_signature_cache")
                    )
                    is not dict
                )
                or type.__getattribute__(_FEMesh, "__dict__").get("get_node")
                is not _EXACT_FE_MESH_GET_NODE
                or type.__getattribute__(_FEMesh, "__dict__").get(
                    "revision_signature"
                )
                is not _EXACT_FE_MESH_REVISION_SIGNATURE
                or type.__getattribute__(_FEMesh, "__dict__").get("num_nodes")
                is not _EXACT_FE_MESH_NUM_NODES
            ):
                return None
            element_mapping = dict.get(mesh_namespace, "elements")
            token = dict.get(mesh_namespace, "_qualified_direct_state_token")
            mapping_namespace = (
                object.__getattribute__(element_mapping, "__dict__")
                if type(element_mapping) is _QualifiedStateMapping
                else None
            )
            if (
                type(element_mapping) is not _QualifiedStateMapping
                or type(mapping_namespace) is not dict
                or not all(type(name) is str for name in mapping_namespace)
                or any(
                    name in mapping_namespace
                    for name in ("get", "items", "values")
                )
                or _static_mro_attribute(_QualifiedStateMapping, "get")
                is not dict.get
                or _static_mro_attribute(_QualifiedStateMapping, "items")
                is not dict.items
                or _static_mro_attribute(_QualifiedStateMapping, "values")
                is not dict.values
                or type(token) is not _QualifiedMutationEpoch
                or len(token) != 1
                or type(list.__getitem__(token, 0)) is not int
                or dict.get(mapping_namespace, "_qualified_token") is not token
                or dict.get(mapping_namespace, "_qualified_kind") != "element"
            ):
                return None
            element_items = tuple(dict.items(element_mapping))
            if not all(
                type(element_id) is int
                for element_id, _element in element_items
            ):
                return None
            material_items = tuple(dict.items(materials))
            if not all(
                type(name) is str for name, _material in material_items
            ):
                return None
            routing = capture_owned_routing(
                mesh,
                mesh_namespace,
                element_items,
                token,
            )
            if (
                element_items
                and all(
                    type(element) in {q4_type, s3_type}
                    for _element_id, element in element_items
                )
                and routing is None
            ):
                raise ValueError(
                    "qualified assembly node/DOF routing is incompatible"
                )
            # Ordinary mixed assembly populates these owned containers after
            # lease capture.  Establish only missing containers NOW, after
            # exact mesh/container admission and before binding their identity.
            # Use the admitted exact namespace directly: no provider/setattr
            # callback, no cache clearing, and no acceptance of later replacement.
            for cache_name in ("_sparsity_cache", "_topology_signature_cache"):
                if dict.get(mesh_namespace, cache_name) is None:
                    mesh_namespace[cache_name] = {}
            return {
                "model_namespace": model_namespace,
                "model_keys": tuple(model_namespace),
                "model_key_types": tuple(map(type, model_namespace)),
                "mesh": mesh,
                "mesh_namespace": mesh_namespace,
                "mesh_keys": tuple(
                    name
                    for name in mesh_namespace
                    if name not in _Q4_WARM_ASSEMBLY_OWNED_MESH_CACHE_KEYS
                ),
                "mesh_key_types": tuple(
                    type(name)
                    for name in mesh_namespace
                    if name not in _Q4_WARM_ASSEMBLY_OWNED_MESH_CACHE_KEYS
                ),
                "sparsity_cache": dict.get(mesh_namespace, "_sparsity_cache"),
                "topology_cache": dict.get(
                    mesh_namespace,
                    "_topology_signature_cache",
                ),
                "element_mapping": element_mapping,
                "mapping_namespace": mapping_namespace,
                "mapping_keys": tuple(mapping_namespace),
                "mapping_key_types": tuple(map(type, mapping_namespace)),
                "element_items": element_items,
                "materials": materials,
                "material_items": material_items,
                "current_material": current_material,
                "token": token,
                "token_type": type(token),
                "token_value": int(list.__getitem__(token, 0)),
                "routing": routing,
            }

        def builtin_material(element: Any) -> Any:
            plan = q4_fast_plan
            if plan is None:
                return None
            element_namespace = object.__getattribute__(element, "__dict__")
            if type(element_namespace) is not dict:
                return None
            material_name = dict.get(element_namespace, "material_name")
            if type(material_name) is not str:
                return None
            if not material_name:
                material_name = plan["current_material"]
            materials = plan["materials"]
            default = dict.get(materials, "default")
            if default is None:
                return None
            return dict.get(materials, material_name, default)

        def s3_prepared_cache_key_matches(
            element: Any,
            material: Any,
            expected_key: Any,
        ) -> bool:
            """Reconstruct the exact S3 plan preimage without providers."""

            plan = q4_fast_plan
            if (
                plan is None
                or type(element) is not s3_type
                or type(material) is not _Material
                or type(expected_key) is not tuple
                or len(expected_key) != 14
            ):
                return False
            element_namespace = object.__getattribute__(element, "__dict__")
            material_namespace = object.__getattribute__(material, "__dict__")
            if (
                type(element_namespace) is not dict
                or type(material_namespace) is not dict
                or any(
                    name in material_namespace
                    for name in (
                        "elastic_symmetry",
                        "shear_modulus",
                        "is_nonlinear",
                        "elastic_compliance_matrix",
                    )
                )
            ):
                return False
            node_ids = dict.get(element_namespace, "node_ids")
            if (
                type(node_ids) is not tuple
                or len(node_ids) != 3
                or not all(type(node_id) is int for node_id in node_ids)
            ):
                return False
            nodes = dict.get(plan["mesh_namespace"], "nodes")
            coordinate_rows: list[tuple[float, float, float]] = []
            for node_id in node_ids:
                node = dict.get(nodes, node_id)
                if type(node) is not _Node:
                    return False
                namespace = object.__getattribute__(node, "__dict__")
                if type(namespace) is not dict:
                    return False
                x = dict.get(namespace, "x")
                y = dict.get(namespace, "y")
                z = dict.get(namespace, "z")
                if any(type(value) not in {int, float} for value in (x, y, z)):
                    return False
                coordinate_rows.append((float(x), float(y), float(z)))
            coordinates = np.asarray(coordinate_rows, dtype=np.float64)
            relative_bytes = np.ascontiguousarray(
                coordinates - np.mean(coordinates, axis=0),
                dtype=np.float64,
            ).tobytes(order="C")

            elastic_modulus = dict.get(material_namespace, "elastic_modulus")
            poisson_ratio = dict.get(material_namespace, "poisson_ratio")
            hardening_curve = dict.get(material_namespace, "hardening_curve")
            if (
                type(elastic_modulus) not in {int, float}
                or type(poisson_ratio) not in {int, float}
                or hardening_curve is not None
            ):
                return False
            elastic_modulus = float(elastic_modulus)
            poisson_ratio = float(poisson_ratio)
            material_fingerprint = (
                "isotropic_scalar_path",
                elastic_modulus,
                poisson_ratio,
                elastic_modulus / (2.0 * (1.0 + poisson_ratio)),
            )
            revisions = dict.get(plan["mesh_namespace"], "revisions")
            if type(revisions) is not dict:
                return False
            normal = dict.get(element_namespace, "reference_normal")
            if (
                type(normal) is not np.ndarray
                or normal.dtype.str != "<f8"
                or normal.shape != (3,)
                or normal.strides != (8,)
                or normal.flags.writeable
            ):
                return False
            direction = dict.get(element_namespace, "material_direction")
            section = dict.get(element_namespace, "shell_section")
            if direction is not None or section is not None:
                return False
            thickness = dict.get(element_namespace, "thickness")
            angle = dict.get(element_namespace, "material_angle_deg")
            polarity = dict.get(element_namespace, "director_polarity")
            offset = dict.get(element_namespace, "reference_surface_offset")
            if (
                type(thickness) not in {int, float}
                or type(angle) not in {int, float}
                or type(polarity) is not int
                or type(offset) not in {int, float}
            ):
                return False
            current_key = (
                id(plan["mesh"]),
                id(material),
                material_fingerprint,
                int(dict.get(revisions, "geometry", 0)),
                int(dict.get(revisions, "material", 0)),
                relative_bytes,
                float(thickness),
                float(angle),
                None,
                None,
                tuple(normal),
                int(polarity),
                float(offset),
                True,
            )
            return current_key == expected_key

        def require_raw_plan() -> None:
            plan = q4_fast_plan
            if plan is None:
                raise ValueError("qualified Q4 warm assembly plan is absent")
            model_namespace = plan["model_namespace"]
            mesh = plan["mesh"]
            mesh_namespace = plan["mesh_namespace"]
            mapping = plan["element_mapping"]
            mapping_namespace = plan["mapping_namespace"]
            current_mesh_keys = tuple(
                name
                for name in mesh_namespace
                if name not in _Q4_WARM_ASSEMBLY_OWNED_MESH_CACHE_KEYS
            )
            current_items = tuple(dict.items(mapping))
            current_material_items = tuple(dict.items(plan["materials"]))
            if (
                type(model) is not exact_model_type
                or object.__getattribute__(model, "__dict__")
                is not model_namespace
                or tuple(model_namespace) != plan["model_keys"]
                or tuple(map(type, model_namespace)) != plan["model_key_types"]
                or "get_material" in model_namespace
                or type.__getattribute__(exact_model_type, "__dict__").get(
                    "get_material"
                )
                is not _EXACT_FE_MODEL_GET_MATERIAL
                or dict.get(model_namespace, "mesh") is not mesh
                or dict.get(model_namespace, "materials") is not plan["materials"]
                or dict.get(model_namespace, "current_material")
                != plan["current_material"]
                or type(mesh) is not _FEMesh
                or object.__getattribute__(mesh, "__dict__") is not mesh_namespace
                or current_mesh_keys != plan["mesh_keys"]
                or tuple(map(type, current_mesh_keys)) != plan["mesh_key_types"]
                or any(
                    name in mesh_namespace
                    for name in ("get_node", "revision_signature", "num_nodes")
                )
                or type.__getattribute__(_FEMesh, "__dict__").get("get_node")
                is not _EXACT_FE_MESH_GET_NODE
                or type.__getattribute__(_FEMesh, "__dict__").get(
                    "revision_signature"
                )
                is not _EXACT_FE_MESH_REVISION_SIGNATURE
                or type.__getattribute__(_FEMesh, "__dict__").get("num_nodes")
                is not _EXACT_FE_MESH_NUM_NODES
                or dict.get(mesh_namespace, "elements") is not mapping
                or dict.get(mesh_namespace, "_sparsity_cache")
                is not plan["sparsity_cache"]
                or dict.get(mesh_namespace, "_topology_signature_cache")
                is not plan["topology_cache"]
                or dict.get(mesh_namespace, "_qualified_direct_state_token")
                is not plan["token"]
                or type(mapping) is not _QualifiedStateMapping
                or object.__getattribute__(mapping, "__dict__")
                is not mapping_namespace
                or tuple(mapping_namespace) != plan["mapping_keys"]
                or tuple(map(type, mapping_namespace))
                != plan["mapping_key_types"]
                or any(
                    name in mapping_namespace
                    for name in ("get", "items", "values")
                )
                or dict.get(mapping_namespace, "_qualified_token")
                is not plan["token"]
                or dict.get(mapping_namespace, "_qualified_kind") != "element"
                or type(plan["token"]) is not plan["token_type"]
                or int(list.__getitem__(plan["token"], 0))
                != plan["token_value"]
                or len(current_items) != len(plan["element_items"])
                or any(
                    current_id != expected_id
                    or current_element is not expected_element
                    for (current_id, current_element), (
                        expected_id,
                        expected_element,
                    ) in zip(current_items, plan["element_items"])
                )
                or len(current_material_items) != len(plan["material_items"])
                or any(
                    current_name != expected_name
                    or current_material is not expected_material
                    for (current_name, current_material), (
                        expected_name,
                        expected_material,
                    ) in zip(current_material_items, plan["material_items"])
                )
            ):
                raise ValueError("qualified Q4 warm assembly raw inputs changed")
            require_owned_routing(plan)

        def require_raw_token() -> None:
            """Check only state that can influence the provider-free body."""

            plan = q4_fast_plan
            if (
                plan is None
                or type(model) is not exact_model_type
                or object.__getattribute__(model, "__dict__")
                is not plan["model_namespace"]
                or dict.get(plan["model_namespace"], "mesh") is not plan["mesh"]
                or type(plan["mesh"]) is not _FEMesh
                or object.__getattribute__(plan["mesh"], "__dict__")
                is not plan["mesh_namespace"]
                or dict.get(
                    plan["mesh_namespace"],
                    "_qualified_direct_state_token",
                )
                is not plan["token"]
                or type(plan["token"]) is not plan["token_type"]
                or int(list.__getitem__(plan["token"], 0))
                != plan["token_value"]
            ):
                raise ValueError("qualified Q4 warm assembly inputs changed")
            require_owned_routing(plan)

        try:
            if allow_q4_cached_stiffness:
                q4_fast_plan = raw_plan_candidate()
            # The warm-stiffness plan and the callback-free input plan have
            # different scopes.  Mixed shell/beam models can own a warm Q4
            # plan while still needing the narrower input-traversal
            # capability during prestress normalization and geometric-state
            # lookup.  Capture that authority independently; otherwise the
            # cached mixed route silently loses it and falls back to a full
            # O(N) lifecycle scan for every state observation.
            qualified_input_plan = qualified_builtin_inputs()
            if q4_fast_plan is not None:
                elements = tuple(
                    element
                    for _element_id, element in q4_fast_plan["element_items"]
                )
            elif qualified_input_plan is not None:
                elements = tuple(
                    element
                    for _element_id, element in qualified_input_plan[
                        "element_items"
                    ]
                )
            else:
                elements = tuple(model.mesh.elements.values())
            q4_elements = tuple(
                element for element in elements if type(element) is q4_type
            )
            s3_elements = tuple(
                element for element in elements if type(element) is s3_type
            )
            # Capture first, then validate the exact state, then require the
            # same generation.  Capturing after validation would allow a
            # concurrent persistent mutation in the narrow return-to-capture
            # window to become the lease's starting authority.
            q4_generation = q4_start_generation if q4_elements else None
            s3_generation = s3_start_generation if s3_elements else None
            numerical_guard(context=context)
            assembly_numerical_guard()
            assembly_operation_guard()
            require_cached_accessor_authority()
            if q4_elements and allow_q4_cached_stiffness:
                q4_cached_epoch_guard()
                q4_fast_base_guard()
                provisional = []
                for element in q4_elements:
                    material = builtin_material(element)
                    cached = (
                        None
                        if material is None
                        else q4_assembly_cached_stiffness(
                            element,
                            q4_fast_plan["mesh"],
                            material,
                        )
                    )
                    if cached is None:
                        provisional = []
                        break
                    provisional.append(
                        (
                            element,
                            material,
                            canonical_q4_total_bytes(cached),
                        )
                    )
                q4_fast_records = tuple(provisional)
                if len(q4_fast_records) == len(q4_elements):
                    if q4_fast_plan is None:
                        q4_fast_records = ()
                    else:
                        token = q4_fast_plan["token"]
                        q4_fast_mesh_token = (
                            token,
                            type(token),
                            int(list.__getitem__(token, 0)),
                        )
            if len(q4_fast_records) != len(q4_elements):
                if qualified_input_plan is None:
                    qualified_input_plan = qualified_builtin_inputs()
                for element in q4_elements:
                    q4_guard(element, context=context)
            if s3_elements and allow_q4_cached_stiffness:
                s3_cached_epoch_guard()
                s3_fast_base_guard()
                # A changed prepared-plan class is an authority violation,
                # not a cache miss.  Reject it before observing any candidate
                # fields or falling through to the generic reference-batch
                # route, where a replaced descriptor/method could run.
                _require_exact_prepared_s3_class_authority()
                candidate_plan = (
                    None
                    if q4_fast_plan is None
                    else dict.get(
                        q4_fast_plan["mesh_namespace"],
                        "_qualified_s3_reference_stiffness_plan",
                    )
                )
                s3_ids = tuple(
                    int(element_id)
                    for element_id, element in (
                        ()
                        if q4_fast_plan is None
                        else q4_fast_plan["element_items"]
                    )
                    if type(element) is s3_type
                )
                candidate_rejected = False
                try:
                    prior_candidate_authority = lookup_prepared_s3_authority(
                        q4_fast_plan["mesh"],
                        candidate_plan,
                        q4_fast_plan["token"],
                        routing_prevalidated=True,
                        preflight=True,
                    )
                except (
                    PreparedS3PlanDataChanged,
                    PreparedS3MaterialChanged,
                    PreparedS3ExecutionChanged,
                ):
                    prior_candidate_authority = None
                    candidate_rejected = True
                prior_candidate_snapshot = (
                    None
                    if prior_candidate_authority is None
                    else prior_candidate_authority["snapshot"]
                )
                candidate_snapshot = (
                    None
                    if candidate_rejected
                    else prior_candidate_snapshot
                    if prior_candidate_snapshot is not None
                    else capture_s3_reference_snapshot(
                        candidate_plan,
                        s3_ids,
                    )
                )
                if (
                    type(candidate_plan) is _PreparedReferenceS3Components
                    and candidate_snapshot is None
                ):
                    # A known prepared plan that fails the stronger exact
                    # snapshot must never fall through to a weaker plan-reuse
                    # predicate in the generic reference-batch path.
                    s3_fast_candidate_seen = True
                if candidate_snapshot is not None:
                    s3_fast_candidate_seen = True
                    provisional_s3 = (
                        []
                        if prior_candidate_authority is None
                        else list(
                            prior_candidate_authority["execution_records"]
                        )
                    )
                    provisional_fallback = (
                        []
                        if prior_candidate_authority is None
                        else list(
                            prior_candidate_authority["fallback_records"]
                        )
                    )
                    prepared_snapshot: dict[str, Any] | None = None
                    prepared_checked = False
                    s3_items = tuple(
                        (int(element_id), element)
                        for element_id, element in q4_fast_plan["element_items"]
                        if type(element) is s3_type
                    )
                    if prior_candidate_authority is None:
                        for element_id, element in s3_items:
                            material = builtin_material(element)
                            cached = (
                                None
                                if material is None
                                else s3_assembly_cached_stiffness(
                                    element,
                                    q4_fast_plan["mesh"],
                                    material,
                                )
                            )
                            if cached is None:
                                if not prepared_checked:
                                    prepared_snapshot = lookup_prepared_s3_snapshot(
                                        q4_fast_plan["mesh"],
                                        candidate_plan,
                                        q4_fast_plan["token"],
                                        routing_prevalidated=True,
                                    )
                                    prepared_checked = True
                                expected_key = None
                                if prepared_snapshot is not None:
                                    expected_key = next(
                                        (
                                            key
                                            for current_id, key in prepared_snapshot[
                                                "cache_key_items"
                                            ]
                                            if current_id == element_id
                                        ),
                                        None,
                                    )
                                payload = (
                                    None
                                    if prepared_snapshot is None
                                    else prepared_snapshot["matrix_payloads"].get(
                                        element_id
                                    )
                                )
                                if (
                                    material is None
                                    or type(expected_key) is not tuple
                                    or type(payload) is not bytes
                                    or not s3_prepared_cache_key_matches(
                                        element,
                                        material,
                                        expected_key,
                                    )
                                ):
                                    provisional_s3 = []
                                    provisional_fallback = []
                                    break
                                cached = payload
                                provisional_fallback.append(
                                    (element, material, expected_key, payload)
                                )
                            provisional_s3.append(
                                (
                                    element,
                                    material,
                                    canonical_s3_total_bytes(cached),
                                )
                            )
                    s3_fast_records = tuple(provisional_s3)
                    s3_plan_fallback_records = tuple(provisional_fallback)
                    if len(s3_fast_records) == len(s3_elements):
                        s3_fast_reference_plan = candidate_plan
                        s3_fast_reference_snapshot = candidate_snapshot
                        s3_prepared_authority = prior_candidate_authority
            if len(s3_fast_records) != len(s3_elements):
                if s3_elements and allow_q4_cached_stiffness and s3_fast_candidate_seen:
                    invalidate_s3_reference_plan()
                if s3_elements and qualified_input_plan is None:
                    qualified_input_plan = qualified_builtin_inputs()
                for element in s3_elements:
                    s3_guard(element, context=context)
            require_qualified_builtin_inputs()
            if q4_generation is not None:
                q4_manager.require_generation(q4_generation)
            if s3_generation is not None:
                s3_manager.require_generation(s3_generation)
            assembly_epoch_manager.require_generation(
                assembly_start_generation
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            for element in q4_elements:
                invalidate_q4(element)
            for element in s3_elements:
                invalidate_s3(element)
            invalidate_s3_reference_plan()
            raise AssemblyError(
                f"{context} found incompatible qualified shell authority"
            ) from exc

        def require(
            expected_model: "FEModel",
            *,
            context: str,
            final: bool = False,
        ) -> None:
            try:
                require_no_trusted_element_builtin_shadows()
                if expected_model is not model:
                    raise ValueError("qualified assembly lease model changed")
                assembly_epoch_manager.require_generation(
                    assembly_start_generation
                )
                if q4_generation is not None:
                    q4_manager.require_generation(q4_generation)
                if s3_generation is not None:
                    s3_manager.require_generation(s3_generation)
                numerical_guard(context=context)
                assembly_numerical_guard()
                assembly_operation_guard()
                require_cached_accessor_authority()
                require_qualified_builtin_inputs()
                if q4_fast_records:
                    q4_cached_epoch_guard()
                    if final:
                        require_raw_plan()
                    else:
                        require_raw_token()
                    if q4_fast_mesh_token is None:
                        raise ValueError(
                            "qualified Q4 warm assembly token is absent"
                        )
                    token, token_type, token_value = q4_fast_mesh_token
                    if (
                        type(token) is not token_type
                        or int(list.__getitem__(token, 0)) != token_value
                    ):
                        raise ValueError(
                            "qualified Q4 warm assembly inputs changed"
                        )
                    if final:
                        for element, material, expected_bytes in q4_fast_records:
                            current_total_bytes = q4_assembly_cached_stiffness(
                                element,
                                q4_fast_plan["mesh"],
                                material,
                            )
                            if (
                                builtin_material(element) is not material
                                or current_total_bytes is None
                                or canonical_q4_total_bytes(current_total_bytes)
                                != expected_bytes
                            ):
                                raise ValueError(
                                    "qualified Q4 warm assembly authority changed"
                                )
                else:
                    for element in q4_elements:
                        q4_guard(element, context=context)
                if s3_fast_records:
                    s3_cached_epoch_guard()
                    if final:
                        require_raw_plan()
                    else:
                        require_raw_token()
                    if (
                        q4_fast_plan is None
                        or s3_fast_reference_snapshot is None
                        or dict.get(
                            q4_fast_plan["mesh_namespace"],
                            "_qualified_s3_reference_stiffness_plan",
                        )
                        is not s3_fast_reference_plan
                    ):
                        raise ValueError(
                            "qualified S3 warm assembly plan changed"
                        )
                    if final and s3_prepared_authority is not None:
                        current_authority = lookup_prepared_s3_authority(
                            q4_fast_plan["mesh"],
                            s3_fast_reference_plan,
                            q4_fast_plan["token"],
                            routing_prevalidated=True,
                        )
                        if current_authority is not s3_prepared_authority:
                            raise ValueError(
                                "qualified S3 prepared execution authority changed"
                            )
                    else:
                        require_s3_reference_snapshot(
                            s3_fast_reference_snapshot
                        )
                    if final and s3_prepared_authority is None:
                        for element, material, expected_bytes in s3_fast_records:
                            fallback_record = next(
                                (
                                    record
                                    for record in s3_plan_fallback_records
                                    if record[0] is element
                                ),
                                None,
                            )
                            if fallback_record is None:
                                current_total_bytes = s3_assembly_cached_stiffness(
                                    element,
                                    q4_fast_plan["mesh"],
                                    material,
                                )
                                valid = (
                                    current_total_bytes is not None
                                    and canonical_s3_total_bytes(
                                        current_total_bytes
                                    )
                                    == expected_bytes
                                )
                            else:
                                (
                                    _fallback_element,
                                    _fallback_material,
                                    expected_key,
                                    fallback_bytes,
                                ) = fallback_record
                                persistent = lookup_prepared_s3_snapshot(
                                    q4_fast_plan["mesh"],
                                    s3_fast_reference_plan,
                                    q4_fast_plan["token"],
                                    routing_prevalidated=True,
                                )
                                valid = (
                                    persistent is not None
                                    and persistent["matrix_payloads"].get(
                                        int(
                                            dict.get(
                                                object.__getattribute__(
                                                    element,
                                                    "__dict__",
                                                ),
                                                "element_id",
                                            )
                                        )
                                    )
                                    == fallback_bytes
                                    and fallback_bytes == expected_bytes
                                    and s3_prepared_cache_key_matches(
                                        element,
                                        material,
                                        expected_key,
                                    )
                                )
                            if builtin_material(element) is not material or not valid:
                                raise ValueError(
                                    "qualified S3 warm assembly authority changed"
                                )
                else:
                    for element in s3_elements:
                        s3_guard(element, context=context)
                if final and s3_elements and q4_fast_plan is not None:
                    current_candidate = dict.get(
                        q4_fast_plan["mesh_namespace"],
                        "_qualified_s3_reference_stiffness_plan",
                    )
                    if not prepared_s3_snapshot_is_current(
                        q4_fast_plan["mesh"],
                        current_candidate,
                        q4_fast_plan["token"],
                    ):
                        current_s3_ids = tuple(
                            int(element_id)
                            for element_id, element in q4_fast_plan[
                                "element_items"
                            ]
                            if type(element) is s3_type
                        )
                        current_snapshot = capture_s3_reference_snapshot(
                            current_candidate,
                            current_s3_ids,
                        )
                        if current_snapshot is None:
                            prepared_s3_by_mesh.pop(
                                id(q4_fast_plan["mesh"]),
                                None,
                            )
                        else:
                            require_owned_routing(q4_fast_plan)
                            bind_prepared_s3_snapshot(
                                q4_fast_plan["mesh"],
                                current_candidate,
                                current_snapshot,
                                q4_fast_plan["token"],
                                q4_fast_plan["element_items"],
                            )
                if q4_generation is not None:
                    q4_manager.require_generation(q4_generation)
                if s3_generation is not None:
                    s3_manager.require_generation(s3_generation)
                assembly_epoch_manager.require_generation(
                    assembly_start_generation
                )
                assembly_operation_guard()
                require_cached_accessor_authority()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                for element in q4_elements:
                    invalidate_q4(element)
                for element in s3_elements:
                    invalidate_s3(element)
                invalidate_s3_reference_plan()
                raise AssemblyError(
                    f"{context} found incompatible qualified shell authority"
                ) from exc

        if q4_fast_records and q4_fast_plan is not None:
            fast_by_identity = {
                id(element): (element, material, total_bytes)
                for element, material, total_bytes in q4_fast_records
            }

            def raw_items() -> tuple[tuple[int, Any], ...]:
                return q4_fast_plan["element_items"]

            def raw_material(element: Any) -> Any:
                record = fast_by_identity.get(id(element))
                if record is None or record[0] is not element:
                    return None
                return record[1]

            def raw_total(element: Any) -> Any:
                record = fast_by_identity.get(id(element))
                if record is None or record[0] is not element:
                    return None
                return np.frombuffer(record[2], dtype=np.float64).reshape(
                    (24, 24)
                )

            require._qualified_q4_raw_element_items = raw_items
            require._qualified_q4_raw_material = raw_material
            require._qualified_q4_cached_total = raw_total
            require._qualified_q4_raw_mesh = q4_fast_plan["mesh"]
            require._qualified_q4_only = len(q4_elements) == len(elements)

        if s3_fast_records and q4_fast_plan is not None:
            s3_fast_by_identity = {
                id(element): (element, material, total_bytes)
                for element, material, total_bytes in s3_fast_records
            }

            def s3_raw_items() -> tuple[tuple[int, Any], ...]:
                return q4_fast_plan["element_items"]

            def s3_raw_material(element: Any) -> Any:
                record = s3_fast_by_identity.get(id(element))
                if record is None or record[0] is not element:
                    return None
                return record[1]

            def s3_raw_total(element: Any) -> Any:
                record = s3_fast_by_identity.get(id(element))
                if record is None or record[0] is not element:
                    return None
                return np.ndarray(
                    (18, 18),
                    dtype=np.float64,
                    buffer=record[2],
                )

            require._qualified_fast_element_items = s3_raw_items
            require._qualified_s3_raw_material = s3_raw_material
            require._qualified_s3_cached_total = s3_raw_total
            require._qualified_s3_reference_plan = s3_fast_reference_plan
            require._qualified_s3_only = len(s3_elements) == len(elements)

        if (
            q4_fast_plan is not None
            and (q4_fast_records or s3_fast_records)
            and len(q4_fast_records) + len(s3_fast_records) == len(elements)
            and q4_fast_plan["routing"] is not None
            and dict.get(q4_fast_plan["mesh_namespace"], "element_activity")
            is None
        ):
            def exact_cached_items() -> tuple[tuple[int, Any], ...]:
                return q4_fast_plan["element_items"]

            require._qualified_fast_element_items = exact_cached_items
            require._qualified_exact_cached_stiffness_only = True
            q4_by_identity = {
                id(element): (element, material, total_bytes)
                for element, material, total_bytes in q4_fast_records
            }
            s3_by_identity = {
                id(element): (element, material, total_bytes)
                for element, material, total_bytes in s3_fast_records
            }
            execution_records: list[tuple[Any, ...]] = []
            for element_id, element in q4_fast_plan["element_items"]:
                family = "q4" if type(element) is q4_type else "s3"
                record = (
                    q4_by_identity.get(id(element))
                    if family == "q4"
                    else s3_by_identity.get(id(element))
                )
                if record is None or record[0] is not element:
                    raise AssemblyError(
                        f"{context} could not bind exact cached assembly data"
                    )
                execution_records.append(
                    (
                        int(element_id),
                        element,
                        record[1],
                        record[2],
                        family,
                    )
                )
            routing = q4_fast_plan["routing"]
            s3_snapshot = s3_fast_reference_snapshot
            data_bytes = b"".join(
                record[3] for record in execution_records
            )
            if len(data_bytes) != routing["entry_count"] * 8:
                raise AssemblyError(
                    f"{context} found incompatible cached assembly payloads"
                )
            execution_plan = MappingProxyType(
                {
                    "mesh": q4_fast_plan["mesh"],
                    "element_items": q4_fast_plan["element_items"],
                    "records": tuple(execution_records),
                    "rows_bytes": routing["rows_bytes"],
                    "cols_bytes": routing["cols_bytes"],
                    "data_bytes": data_bytes,
                    "entry_count": routing["entry_count"],
                    "total_dofs": routing["total_dofs"],
                    "num_nodes": len(routing["node_records"]),
                    "revision_signature": routing["revision_items"],
                    "sparsity_signature": routing["signature"],
                    "s3_diagnostics": (
                        None if s3_snapshot is None else s3_snapshot["diagnostics"]
                    ),
                    "s3_batched_ids": (
                        () if s3_snapshot is None else s3_snapshot["batched_ids"]
                    ),
                    "s3_cached_ids": (
                        () if s3_snapshot is None else s3_snapshot["cached_ids"]
                    ),
                    "s3_group_ids": (
                        () if s3_snapshot is None else s3_snapshot["group_ids"]
                    ),
                    "s3_cache_key_items": (
                        () if s3_snapshot is None else s3_snapshot["cache_key_items"]
                    ),
                    "s3_evaluations": (
                        0 if s3_snapshot is None else s3_snapshot["evaluations"]
                    ),
                }
            )
            exact_execution_plan_register(
                require,
                execution_plan,
            )

        if qualified_input_plan is not None:
            bound_by_identity = qualified_input_plan["material_by_element"]

            def owned_items() -> tuple[tuple[int, Any], ...]:
                return qualified_input_plan["element_items"]

            def owned_material(element: Any) -> Any:
                record = bound_by_identity.get(id(element))
                if record is None or record[0] is not element:
                    return None
                return record[1]

            def owned_material_name(name: Any) -> Any:
                if name is not None and type(name) is not str:
                    return None
                resolved_name = name or qualified_input_plan["current_material"]
                materials = qualified_input_plan["materials"]
                return dict.get(
                    materials,
                    resolved_name,
                    dict.get(materials, "default"),
                )

            require._qualified_owned_element_items = owned_items
            require._qualified_owned_material = owned_material
            require._qualified_owned_material_name = owned_material_name
            require._qualified_owned_mesh = qualified_input_plan["mesh"]

            trusted_plan = qualified_input_plan
            trusted_token = trusted_plan["token"]
            trusted_token_value = trusted_plan["token_value"]

            def trusted_element_require(
                expected_model: "FEModel",
                element: Any,
                material: Any,
                *,
                context: str,
            ) -> None:
                """Check one exact qualified owned element in constant time.

                Mixed models cannot use the complete all-qualified loop fast
                path because generic elements remain callback boundaries.  A
                qualified element and its already-bound material are still
                safe to observe between those boundaries: the exact capture
                owns their providers, while monotonic family/assembly epochs
                and the shared mesh token reject supported mutation and ABA.
                """

                try:
                    require_no_trusted_element_builtin_shadows()
                    assembly_epoch_manager.require_generation(
                        assembly_start_generation
                    )
                    if q4_generation is not None:
                        q4_manager.require_generation(q4_generation)
                    if s3_generation is not None:
                        s3_manager.require_generation(s3_generation)
                    record = exact_dict_get(
                        bound_by_identity,
                        exact_id(element),
                    )
                    if (
                        expected_model is not model
                        or exact_type(model) is not exact_model_type
                        or exact_type(element) not in {q4_type, s3_type}
                        or record is None
                        or record[0] is not element
                        or record[1] is not material
                        or exact_object_getattribute(model, "__dict__")
                        is not trusted_plan["model_namespace"]
                        or exact_dict_get(
                            trusted_plan["model_namespace"], "mesh"
                        )
                        is not trusted_plan["mesh"]
                        or exact_dict_get(
                            trusted_plan["model_namespace"], "materials"
                        )
                        is not trusted_plan["materials"]
                        or exact_dict_get(
                            trusted_plan["model_namespace"],
                            "current_material",
                        )
                        != trusted_plan["current_material"]
                        or exact_type(trusted_plan["mesh"]) is not _FEMesh
                        or exact_object_getattribute(
                            trusted_plan["mesh"], "__dict__"
                        )
                        is not trusted_plan["mesh_namespace"]
                        or exact_dict_get(
                            trusted_plan["mesh_namespace"], "elements"
                        )
                        is not trusted_plan["mapping"]
                        or exact_dict_get(
                            trusted_plan["mesh_namespace"],
                            "_qualified_direct_state_token",
                        )
                        is not trusted_token
                        or exact_type(trusted_plan["mapping"])
                        is not _QualifiedStateMapping
                        or exact_object_getattribute(
                            trusted_plan["mapping"], "__dict__"
                        )
                        is not trusted_plan["mapping_namespace"]
                        or exact_dict_get(
                            trusted_plan["mapping_namespace"],
                            "_qualified_token",
                        )
                        is not trusted_token
                        or exact_dict_get(
                            trusted_plan["mapping_namespace"],
                            "_qualified_kind",
                        )
                        != "element"
                        or exact_type(trusted_token) is not _QualifiedMutationEpoch
                        or exact_len(trusted_token) != 1
                        or exact_int(exact_list_getitem(trusted_token, 0))
                        != trusted_token_value
                    ):
                        raise exact_value_error(
                            "qualified trusted-element inputs changed"
                        )
                    if q4_generation is not None:
                        q4_manager.require_generation(q4_generation)
                    if s3_generation is not None:
                        s3_manager.require_generation(s3_generation)
                    assembly_epoch_manager.require_generation(
                        assembly_start_generation
                    )
                except (
                    exact_attribute_error,
                    exact_runtime_error,
                    exact_type_error,
                    exact_value_error,
                ) as exc:
                    for current in q4_elements:
                        invalidate_q4(current)
                    for current in s3_elements:
                        invalidate_s3(current)
                    invalidate_s3_reference_plan()
                    raise exact_assembly_error(
                        f"{context} found incompatible qualified shell authority"
                    ) from exc

            require._qualified_trusted_element_require = (
                trusted_element_require
            )

            qualified_input_anchors = q4_elements + s3_elements
            if qualified_input_anchors:
                trusted_input_element = qualified_input_anchors[0]
                trusted_input_record = exact_dict_get(
                    bound_by_identity,
                    exact_id(trusted_input_element),
                )
                if (
                    trusted_input_record is None
                    or trusted_input_record[0] is not trusted_input_element
                ):
                    raise exact_assembly_error(
                        "qualified input-traversal authority is absent"
                    )
                trusted_input_material = trusted_input_record[1]

                def trusted_input_require(
                    expected_model: "FEModel",
                    *,
                    context: str,
                ) -> None:
                    """Guard callback-free owned input traversal in mixed models.

                    This authority is deliberately narrower than the
                    all-qualified mechanics-loop lease.  It may be used only
                    between observations of exact built-in input containers;
                    arbitrary element/provider calls remain complete-guard
                    boundaries.
                    """

                    trusted_element_require(
                        expected_model,
                        trusted_input_element,
                        trusted_input_material,
                        context=context,
                    )

                require._qualified_trusted_input_require = (
                    trusted_input_require
                )

            def trusted_require(
                expected_model: "FEModel",
                *,
                context: str,
            ) -> None:
                """Check a fully qualified trusted loop in constant time.

                Also exported as ``_qualified_trusted_state_require`` for
                mixed models.  There it validates only the owned inputs, the
                mesh token and the Q4/S3/assembly epochs; it says nothing
                about any other element family, which the caller must cover
                with its own authority.

                The exact capture and every caller-controlled boundary use
                the complete lease.  Between those boundaries, monotonic
                authority generations and the mesh mutation token reject
                supported mutation, including mutate-then-restore ABA.
                """

                try:
                    require_no_trusted_element_builtin_shadows()
                    assembly_epoch_manager.require_generation(
                        assembly_start_generation
                    )
                    if q4_generation is not None:
                        q4_manager.require_generation(q4_generation)
                    if s3_generation is not None:
                        s3_manager.require_generation(s3_generation)
                    if (
                        expected_model is not model
                        or exact_type(model) is not exact_model_type
                        or exact_object_getattribute(model, "__dict__")
                        is not trusted_plan["model_namespace"]
                        or exact_dict_get(
                            trusted_plan["model_namespace"], "mesh"
                        )
                        is not trusted_plan["mesh"]
                        or exact_dict_get(
                            trusted_plan["model_namespace"], "materials"
                        )
                        is not trusted_plan["materials"]
                        or exact_dict_get(
                            trusted_plan["model_namespace"],
                            "current_material",
                        )
                        != trusted_plan["current_material"]
                        or exact_type(trusted_plan["mesh"]) is not _FEMesh
                        or exact_object_getattribute(
                            trusted_plan["mesh"], "__dict__"
                        )
                        is not trusted_plan["mesh_namespace"]
                        or exact_dict_get(
                            trusted_plan["mesh_namespace"], "elements"
                        )
                        is not trusted_plan["mapping"]
                        or exact_dict_get(
                            trusted_plan["mesh_namespace"],
                            "_qualified_direct_state_token",
                        )
                        is not trusted_token
                        or exact_type(trusted_plan["mapping"])
                        is not _QualifiedStateMapping
                        or exact_object_getattribute(
                            trusted_plan["mapping"], "__dict__"
                        )
                        is not trusted_plan["mapping_namespace"]
                        or exact_dict_get(
                            trusted_plan["mapping_namespace"],
                            "_qualified_token",
                        )
                        is not trusted_token
                        or exact_dict_get(
                            trusted_plan["mapping_namespace"],
                            "_qualified_kind",
                        )
                        != "element"
                        or exact_type(trusted_token)
                        is not _QualifiedMutationEpoch
                        or exact_len(trusted_token) != 1
                        or exact_int(exact_list_getitem(trusted_token, 0))
                        != trusted_token_value
                    ):
                        raise exact_value_error(
                            "qualified trusted-loop inputs changed"
                        )
                    if q4_generation is not None:
                        q4_manager.require_generation(q4_generation)
                    if s3_generation is not None:
                        s3_manager.require_generation(s3_generation)
                    assembly_epoch_manager.require_generation(
                        assembly_start_generation
                    )
                except (
                    exact_attribute_error,
                    exact_runtime_error,
                    exact_type_error,
                    exact_value_error,
                ) as exc:
                    for element in q4_elements:
                        invalidate_q4(element)
                    for element in s3_elements:
                        invalidate_s3(element)
                    invalidate_s3_reference_plan()
                    raise exact_assembly_error(
                        f"{context} found incompatible qualified shell authority"
                    ) from exc

            if len(q4_elements) + len(s3_elements) == len(elements):
                require._qualified_trusted_require = trusted_require
            # Owned-input, token and Q4/S3/assembly epoch state only.  It
            # validates no other element family; a caller may use it only
            # together with its own authority for every remaining element
            # (nonlinear statics pairs it with the S3 V2D capture).
            require._qualified_trusted_state_require = trusted_require

        return require

    return capture


_CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE = (
    _bind_qualified_assembly_runtime_lease(
        _EXACT_NUMPY_RUNTIME_GUARD,
        _EXACT_Q4_RUNTIME_GUARD,
        _EXACT_S3_RUNTIME_GUARD,
        _QualifiedE4PLShellElement,
        _QualifiedE4PLS3ShellElement,
        _Q4_RUNTIME_EPOCH_MANAGER,
        _S3_RUNTIME_EPOCH_MANAGER,
        _INVALIDATE_Q4_GUARDED_CACHES,
        _INVALIDATE_S3_GUARDED_CACHES,
        _EXACT_Q4_CACHED_STIFFNESS_EPOCH_GUARD,
        _EXACT_Q4_FAST_BASE_AUTHORITY,
        _TRY_Q4_FAST_ASSEMBLY_CACHED_STIFFNESS,
        _EXACT_S3_CACHED_STIFFNESS_EPOCH_GUARD,
        _EXACT_S3_FAST_BASE_AUTHORITY,
        _TRY_S3_FAST_ASSEMBLY_CACHED_STIFFNESS,
        _ASSEMBLY_NUMERICAL_EPOCH_MANAGER,
        _EXACT_ASSEMBLY_NUMERICAL_GUARD,
        _REQUIRE_EXACT_ASSEMBLY_OPERATION_AUTHORITY,
        _FEModel,
        _REGISTER_QUALIFIED_ASSEMBLY_EXECUTION_PLAN,
    )
)


def _scoped_operation_error_has_precedence(
    operation_error: BaseException,
    scope_error: BaseException,
) -> bool:
    """Keep the first failure unless cleanup raises a termination signal."""

    return (
        not isinstance(operation_error, Exception)
        or isinstance(scope_error, Exception)
    )


def _run_with_qualified_assembly_runtime_lease(
    model: "FEModel",
    *,
    context: str,
    operation: Callable[[Any], Any],
    allow_q4_cached_stiffness: bool = False,
) -> Any:
    """Run an assembly operation under one non-renewable runtime lease."""

    lease = _CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context=f"{context} preflight",
        allow_q4_cached_stiffness=allow_q4_cached_stiffness,
    )
    try:
        mesh = getattr(model, "mesh", None)
        elements = getattr(mesh, "elements", None)
        if not isinstance(elements, dict):
            # Test doubles and deliberately rejected inputs still exercise the
            # operation/exception precedence owned by the captured lease.
            result = operation(lease)
        else:
            qualified_elements = tuple(elements.values())
            q4_elements = tuple(
                element
                for element in qualified_elements
                if type(element) is _QualifiedE4PLShellElement
            )
            s3_elements = tuple(
                element
                for element in qualified_elements
                if type(element) is _QualifiedE4PLS3ShellElement
            )
            scoped_operation_error: BaseException | None = None
            try:
                with ExitStack() as stack:
                    stack.enter_context(_Q4_TRUSTED_OPERATION_SCOPE(q4_elements))
                    stack.enter_context(_S3_TRUSTED_OPERATION_SCOPE(s3_elements))
                    try:
                        result = operation(lease)
                    except BaseException as exc:
                        scoped_operation_error = exc
                        raise
            except BaseException as scope_error:
                # Trusted-scope finalization must still run and may detect the
                # same adversarial mutation.  It must not, however, replace a
                # more specific operation-time rejection with a generic
                # cleanup AssemblyError.  The outer lease still adjudicates
                # exceptional output and may wrap that original rejection.
                if (
                    scoped_operation_error is not None
                    and _scoped_operation_error_has_precedence(
                        scoped_operation_error,
                        scope_error,
                    )
                ):
                    if (
                        scope_error is not scoped_operation_error
                        and hasattr(scoped_operation_error, "add_note")
                    ):
                        scoped_operation_error.add_note(
                            "qualified trusted scope also rejected cleanup: "
                            f"{type(scope_error).__name__}: {scope_error}"
                        )
                    raise scoped_operation_error
                raise
    except BaseException as operation_error:
        # A mutation followed by restoration must invalidate the failed call
        # and every derived qualified cache written while it was in flight.
        lease_error: BaseException | None = None
        try:
            lease(
                model,
                context=f"{context} exceptional output",
                final=True,
            )
        except BaseException as exc:
            lease_error = exc
        if isinstance(operation_error, ElementCapabilityError):
            if lease_error is not None and hasattr(operation_error, "add_note"):
                operation_error.add_note(
                    "qualified assembly lease also rejected exceptional output: "
                    f"{type(lease_error).__name__}: {lease_error}"
                )
            raise
        if lease_error is not None:
            raise lease_error from operation_error
        raise
    lease(model, context=f"{context} output", final=True)
    return result


def _element_activity(model: "FEModel") -> Any | None:
    return getattr(model.mesh, "element_activity", None)


def _activity_scales(
    model: "FEModel", quantity: str
) -> tuple[Any | None, Dict[int, float], Dict[str, Any] | None]:
    activity = _element_activity(model)
    if activity is None:
        return None, {}, None
    from .current_state_tangent import (
        require_exact_qualified_component_lifecycle_api,
    )

    exact_guard = require_exact_qualified_component_lifecycle_api
    exact_guard(model, context=f"{quantity} activity-scale preflight")
    element_ids = tuple(int(element_id) for element_id in model.mesh.elements)
    try:
        observed_values = activity.scales(quantity, element_ids)
    except Exception as error:
        raise AssemblyError(
            f"element activity cannot provide {quantity} scales for the FE mesh: {error}"
        ) from error
    exact_guard(
        model,
        context=f"{quantity} activity-scale provider observation",
    )
    try:
        values = np.asarray(observed_values, dtype=float)
    except Exception as error:
        raise AssemblyError(
            f"element activity cannot provide {quantity} scales for the FE mesh: {error}"
        ) from error
    exact_guard(
        model,
        context=f"{quantity} activity-scale array observation",
    )
    values = values.reshape(-1)
    if values.shape != (len(element_ids),) or not np.all(np.isfinite(values)):
        raise AssemblyError(f"element activity returned invalid {quantity} scales")
    scales = dict(zip(element_ids, (float(value) for value in values)))
    return activity, scales, {
        "quantity": str(quantity),
        "sequence": int(getattr(activity, "sequence", 0)),
        "element_count": len(element_ids),
        "scaled_element_count": int(np.count_nonzero(values != 1.0)),
        "zero_contribution_count": int(np.count_nonzero(values == 0.0)),
        "minimum_scale": float(np.min(values)) if len(values) else 1.0,
        "maximum_scale": float(np.max(values)) if len(values) else 1.0,
    }


def _base_info(model: "FEModel", matrix_type: str) -> Dict[str, Any]:
    mesh = model.mesh
    return {
        "matrix_type": matrix_type,
        "num_elements": 0,
        "num_nodes": mesh.num_nodes,
        "total_dofs": mesh.dof_manager.total_dofs,
        "assembly_time": 0.0,
        "element_times": {},
        "skipped_elements": [],
        "diagnostics": {},
        "revision_signature": getattr(mesh, "revision_signature", lambda: {})(),
    }


def _check_element_matrix_shape(element_id: int, matrix_name: str, matrix: np.ndarray, expected_size: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    expected_shape = (expected_size, expected_size)
    if matrix.shape != expected_shape:
        raise AssemblyError(
            f"Element {element_id} returned {matrix_name} with shape {matrix.shape}; "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise AssemblyError(f"Element {element_id} returned non-finite values in {matrix_name}.")
    return matrix


def _relative_symmetry_error(
    matrix: sparse.spmatrix | np.ndarray,
    _issparse: Any = _ASSEMBLY_SPARSE_ALIASES["issparse"],
    _sparse_norm: Any = _ASSEMBLY_SPARSE_LINALG_ALIASES["norm"],
) -> float:
    if _issparse(matrix):
        diff = matrix - matrix.T
        numerator = float(_sparse_norm(diff))
        denominator = max(float(_sparse_norm(matrix)), 1.0)
        return numerator / denominator
    dense = np.asarray(matrix, dtype=float)
    return float(np.linalg.norm(dense - dense.T) / max(np.linalg.norm(dense), 1.0))


def _topology_signature(
    mesh: Any,
    matrix_type: str,
    *,
    element_items: tuple[tuple[Any, Any], ...] | None = None,
) -> str:
    exact_mesh = type(mesh) is _FEMesh
    mesh_namespace = (
        object.__getattribute__(mesh, "__dict__") if exact_mesh else None
    )
    revisions = (
        _EXACT_FE_MESH_REVISION_SIGNATURE(mesh)
        if exact_mesh
        and type(mesh_namespace) is dict
        and "revision_signature" not in mesh_namespace
        else getattr(mesh, "revision_signature", lambda: {})()
    )
    direct_token = (
        dict.get(mesh_namespace, "_qualified_direct_state_token")
        if exact_mesh and type(mesh_namespace) is dict
        else getattr(mesh, "_qualified_direct_state_token", None)
    )
    direct_revision = (
        int(direct_token[0])
        if isinstance(direct_token, list) and len(direct_token) == 1
        else -1
    )
    cache_key = (
        str(matrix_type),
        int(revisions.get("topology", 0)),
        int(revisions.get("mpc", 0)),
        direct_revision,
    )
    cache = (
        dict.get(mesh_namespace, "_topology_signature_cache")
        if exact_mesh and type(mesh_namespace) is dict
        else getattr(mesh, "_topology_signature_cache", None)
    )
    if cache is None:
        cache = {}
        if exact_mesh:
            object.__setattr__(mesh, "_topology_signature_cache", cache)
        else:
            mesh._topology_signature_cache = cache
    cached = cache.get(cache_key)
    if cached is not None:
        return str(cached)

    owned_element_items = (
        element_items
        if element_items is not None
        else tuple(mesh.elements.items())
    )
    payload = {
        "matrix_type": matrix_type,
        "topology_revision": revisions.get("topology", 0),
        "mpc_revision": revisions.get("mpc", 0),
        "elements": [
            {
                "id": int(elem_id),
                "class": element.__class__.__name__,
                "node_ids": [int(node_id) for node_id in getattr(element, "node_ids", [])],
                "dofs": [int(dof) for dof in element.get_dof_mapping(mesh)],
            }
            for elem_id, element in owned_element_items
        ],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for stale_key in tuple(cache):
        if stale_key != cache_key and stale_key[0] == str(matrix_type):
            cache.pop(stale_key, None)
    cache[cache_key] = signature
    return signature


def _scatter_element_matrix(
    element_matrix: np.ndarray,
    dof_mapping: np.ndarray,
    rows: list,
    cols: list,
    data: list,
) -> None:
    """Append element matrix entries to COO triplet buffers (vectorized)."""
    n_local = dof_mapping.size
    values = element_matrix.ravel()
    mask = values != 0.0
    if not np.any(mask):
        return
    rows.append(np.repeat(dof_mapping, n_local)[mask])
    cols.append(np.tile(dof_mapping, n_local)[mask])
    data.append(values[mask])


def _triplets_to_csr(
    rows: list,
    cols: list,
    data: list,
    total_dofs: int,
    _coo_constructor: Any = _ASSEMBLY_SPARSE_ALIASES["coo_matrix"],
    _csr_constructor: Any = _ASSEMBLY_SPARSE_ALIASES["csr_matrix"],
    _coo_to_csr: Any = _EXACT_COO_TO_CSR,
    _eliminate_zeros: Any = _EXACT_CSR_ELIMINATE_ZEROS,
) -> sparse.csr_matrix:
    """Build a CSR matrix from COO triplet buffers; duplicates are summed."""
    if not data:
        return _csr_constructor((total_dofs, total_dofs), dtype=float)
    coo = _coo_constructor(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(total_dofs, total_dofs),
        dtype=float,
    )
    return _coo_to_csr(coo)


def _get_cached_sparsity_pattern(
    mesh: "FEMesh",
    matrix_type: str,
    *,
    element_items: tuple[tuple[Any, Any], ...] | None = None,
    _topology_signature_kernel: Any = _topology_signature,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve or build the cached row and column indices for global matrix COO assembly."""
    exact_mesh = type(mesh) is _FEMesh
    mesh_namespace = (
        object.__getattribute__(mesh, "__dict__") if exact_mesh else None
    )
    cache = (
        dict.get(mesh_namespace, "_sparsity_cache")
        if exact_mesh and type(mesh_namespace) is dict
        else getattr(mesh, "_sparsity_cache", None)
    )
    if cache is None:
        cache = {}
        if exact_mesh:
            object.__setattr__(mesh, "_sparsity_cache", cache)
        else:
            mesh._sparsity_cache = cache

    signature = _topology_signature_kernel(
        mesh,
        matrix_type,
        element_items=element_items,
    )

    if matrix_type in cache:
        cached = cache[matrix_type]
        if cached.get("signature") == signature:
            return cached["rows"], cached["cols"]

    rows_list = []
    cols_list = []
    owned_element_items = (
        element_items
        if element_items is not None
        else tuple(mesh.elements.items())
    )
    for _, element in owned_element_items:
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue
        n_local = dof_mapping.size
        rows_list.append(np.repeat(dof_mapping, n_local))
        cols_list.append(np.tile(dof_mapping, n_local))

    rows_concat = np.concatenate(rows_list) if rows_list else np.empty(0, dtype=np.intp)
    cols_concat = np.concatenate(cols_list) if cols_list else np.empty(0, dtype=np.intp)

    cache[matrix_type] = {
        "rows": rows_concat,
        "cols": cols_concat,
        "signature": signature,
    }
    return rows_concat, cols_concat


def _exact_quadrature_array_identity(value: Any) -> tuple[str, tuple[int, ...], bytes]:
    """Return the dtype-, shape-, and byte-exact identity of one rule array."""

    array = np.ascontiguousarray(np.asarray(value))
    return (
        array.dtype.str,
        tuple(int(size) for size in array.shape),
        array.tobytes(order="C"),
    )


def _shell_quadrature_batch_identity(
    element: Any,
    *,
    include_shear: bool,
) -> tuple[Any, ...]:
    """Bind every quadrature input consumed by one legacy batch kernel.

    A concrete shell class may legally expose a formulation-specific rule,
    and two instances of the same custom class may expose different rules.
    Batch grouping must therefore retain the exact arrays rather than infer
    compatibility from topology and material inputs alone.
    """

    identity: tuple[Any, ...] = (
        _exact_quadrature_array_identity(element.gauss_points),
        _exact_quadrature_array_identity(element.gauss_weights),
    )
    if include_shear:
        identity += (
            _exact_quadrature_array_identity(element.shear_gauss_points),
            _exact_quadrature_array_identity(element.shear_gauss_weights),
        )
    return identity


def _assemble_element_matrix_under_lease_impl(
    model: "FEModel",
    matrix_type: str,
    element_matrix_getter: Callable[[Any, Any, Any], np.ndarray],
    qualified_runtime_guard: Any,
    _owned_execution_plan: Any,
    *,
    activity_quantity: str | None = None,
    _activity_scales_kernel: Any = _activity_scales,
    _base_info_kernel: Any = _base_info,
    _check_matrix_kernel: Any = _check_element_matrix_shape,
    _relative_symmetry_kernel: Any = _relative_symmetry_error,
    _sparsity_kernel: Any = _get_cached_sparsity_pattern,
    _topology_signature_kernel: Any = _topology_signature,
    _coo_constructor: Any = _ASSEMBLY_SPARSE_ALIASES["coo_matrix"],
    _csr_constructor: Any = _ASSEMBLY_SPARSE_ALIASES["csr_matrix"],
    _coo_to_csr: Any = _EXACT_COO_TO_CSR,
    _eliminate_zeros: Any = _EXACT_CSR_ELIMINATE_ZEROS,
    _object_new: Any = object.__new__,
    _object_setattr: Any = object.__setattr__,
    _object_getattribute: Any = object.__getattribute__,
    _ndarray_constructor: Any = np.ndarray,
    _empty_constructor: Any = np.empty,
    _intp_dtype: Any = np.dtype(np.intp),
    _float64_dtype: Any = np.dtype(np.float64),
    _coo_to_csr_kernel: Any = _SCIPY_COO_TO_CSR,
    _csr_sort_indices_kernel: Any = _SCIPY_CSR_SORT_INDICES,
    _csr_sum_duplicates_kernel: Any = _SCIPY_CSR_SUM_DUPLICATES,
    _csr_has_sorted_indices_kernel: Any = _SCIPY_CSR_HAS_SORTED_INDICES,
    _csr_has_canonical_format_kernel: Any = _SCIPY_CSR_HAS_CANONICAL_FORMAT,
    _isfinite_kernel: Any = np.isfinite,
    _all_kernel: Any = np.all,
    _exact_type: Any = type,
    _exact_len: Any = len,
    _exact_tuple: Any = tuple,
    _exact_dict: Any = dict,
    _exact_list: Any = list,
    _exact_int: Any = int,
    _float_type: Any = float,
    _mapping_proxy_type: Any = MappingProxyType,
    _assembly_error_type: Any = AssemblyError,
    _clock: Any = time.perf_counter,
    _reference_s3_formulation_id: Any = _REFERENCE_S3_FORMULATION_ID,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    if _owned_execution_plan is not None:
        qualified_runtime_guard(
            model,
            context="qualified cached stiffness assembly mechanics",
        )
        if _exact_type(_owned_execution_plan) is not _mapping_proxy_type:
            raise _assembly_error_type(
                "qualified assembly execution plan is incompatible"
            )
        if matrix_type != "stiffness":
            raise _assembly_error_type(
                "qualified cached execution plans are stiffness-only"
            )
        start_time = _clock()
        mesh = _owned_execution_plan["mesh"]
        records = _owned_execution_plan["records"]
        total_dofs = _owned_execution_plan["total_dofs"]
        entry_count = _owned_execution_plan["entry_count"]
        rows = _ndarray_constructor(
            (entry_count,),
            dtype=_intp_dtype,
            buffer=_owned_execution_plan["rows_bytes"],
        )
        cols = _ndarray_constructor(
            (entry_count,),
            dtype=_intp_dtype,
            buffer=_owned_execution_plan["cols_bytes"],
        )
        data = _ndarray_constructor(
            (entry_count,),
            dtype=_float64_dtype,
            buffer=_owned_execution_plan["data_bytes"],
        )
        if rows.size != data.size or cols.size != data.size:
            raise _assembly_error_type(
                "qualified owned sparsity does not match matrices"
            )
        matrix_indptr = _empty_constructor(
            (total_dofs + 1,),
            dtype=_intp_dtype,
        )
        unsummed_indices = _empty_constructor(
            (entry_count,),
            dtype=_intp_dtype,
        )
        unsummed_data = _empty_constructor(
            (entry_count,),
            dtype=_float64_dtype,
        )
        _coo_to_csr_kernel(
            total_dofs,
            total_dofs,
            entry_count,
            rows,
            cols,
            data,
            matrix_indptr,
            unsummed_indices,
            unsummed_data,
        )
        _csr_sort_indices_kernel(
            total_dofs,
            matrix_indptr,
            unsummed_indices,
            unsummed_data,
        )
        _csr_sum_duplicates_kernel(
            total_dofs,
            total_dofs,
            matrix_indptr,
            unsummed_indices,
            unsummed_data,
        )
        unique_count = _exact_int(matrix_indptr[-1])
        matrix_indices = unsummed_indices[:unique_count]
        matrix_data = unsummed_data[:unique_count]
        if (
            not _csr_has_sorted_indices_kernel(
                total_dofs,
                matrix_indptr,
                matrix_indices,
            )
            or not _csr_has_canonical_format_kernel(
                total_dofs,
                matrix_indptr,
                matrix_indices,
            )
        ):
            raise _assembly_error_type(
                "qualified sparse assembly output is not canonical"
            )
        matrix = _object_new(_csr_constructor)
        _object_setattr(matrix, "_shape", (total_dofs, total_dofs))
        _object_setattr(matrix, "maxprint", 50)
        _object_setattr(matrix, "indptr", matrix_indptr)
        _object_setattr(matrix, "indices", matrix_indices)
        _object_setattr(matrix, "data", matrix_data)
        _object_setattr(matrix, "_has_canonical_format", True)
        _object_setattr(matrix, "_has_sorted_indices", True)
        matrix_namespace = _object_getattribute(matrix, "__dict__")
        matrix_data = _exact_dict.get(matrix_namespace, "data")
        matrix_indices = _exact_dict.get(matrix_namespace, "indices")
        matrix_indptr = _exact_dict.get(matrix_namespace, "indptr")
        if (
            _exact_type(matrix) is not _csr_constructor
            or _exact_type(matrix_namespace) is not _exact_dict
            or _exact_dict.get(matrix_namespace, "_shape")
            != (total_dofs, total_dofs)
            or _exact_type(matrix_data) is not _ndarray_constructor
            or matrix_data.dtype != _float64_dtype
            or matrix_data.ndim != 1
            or not matrix_data.flags.c_contiguous
            or not _all_kernel(_isfinite_kernel(matrix_data))
            or _exact_type(matrix_indices) is not _ndarray_constructor
            or matrix_indices.ndim != 1
            or matrix_indices.dtype.kind != "i"
            or matrix_indices.dtype.itemsize not in (4, 8)
            or not matrix_indices.flags.c_contiguous
            or matrix_indices.size != matrix_data.size
            or (
                matrix_indices.size
                and (
                    not _all_kernel(matrix_indices >= 0)
                    or not _all_kernel(matrix_indices < total_dofs)
                )
            )
            or _exact_type(matrix_indptr) is not _ndarray_constructor
            or matrix_indptr.ndim != 1
            or matrix_indptr.dtype.kind != "i"
            or matrix_indptr.dtype.itemsize not in (4, 8)
            or not matrix_indptr.flags.c_contiguous
            or matrix_indptr.size != total_dofs + 1
            or matrix_indptr[0] != 0
            or matrix_indptr[-1] != matrix_data.size
            or not _all_kernel(matrix_indptr[1:] >= matrix_indptr[:-1])
            or _exact_dict.get(
                matrix_namespace,
                "_has_canonical_format",
            )
            is not True
            or _exact_dict.get(
                matrix_namespace,
                "_has_sorted_indices",
            )
            is not True
        ):
            raise _assembly_error_type(
                "qualified sparse assembly output is incompatible"
            )
        s3_records = _exact_tuple(
            record for record in records if record[4] == "s3"
        )
        q4_records = _exact_tuple(
            record for record in records if record[4] == "q4"
        )
        vectorized_shell_groups: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {
            "assembled_symmetry_error": 0.0,
        }
        owned_s3_diagnostics = _owned_execution_plan["s3_diagnostics"]
        if owned_s3_diagnostics is not None:
            s3_diagnostics = _exact_dict(owned_s3_diagnostics)
            s3_diagnostics["element_ids"] = _exact_list(
                s3_diagnostics["element_ids"]
            )
            s3_diagnostics["group_element_ids"] = [
                _exact_list(group)
                for group in s3_diagnostics["group_element_ids"]
            ]
            s3_diagnostics["fallback_reasons"] = {
                reason: _exact_list(element_ids)
                for reason, element_ids in s3_diagnostics["fallback_reasons"]
            }
            s3_diagnostics["revision_key"] = _exact_list(
                s3_diagnostics["revision_key"]
            )
            s3_diagnostics["plan_reused"] = True
            diagnostics["qualified_s3_reference_elastic_stiffness"] = (
                s3_diagnostics
            )
            batched_ids = _owned_execution_plan["s3_batched_ids"]
            cached_ids = _owned_execution_plan["s3_cached_ids"]
            group_ids = _owned_execution_plan["s3_group_ids"]
            if batched_ids:
                vectorized_shell_groups.append(
                    {
                        "shell_order": "S3",
                        "num_elements": _exact_len(batched_ids),
                        "kernel": "qualified_s3_reference_elastic_shared_components",
                        "parallel_kernel": False,
                        "unique_geometry_count": _exact_len(group_ids),
                        "component_evaluation_count": _owned_execution_plan[
                            "s3_evaluations"
                        ],
                        "formulation_id": _reference_s3_formulation_id,
                        "speedup_claimed": False,
                    }
                )
            if cached_ids:
                cache_key_by_id = _exact_dict(
                    _owned_execution_plan["s3_cache_key_items"]
                )
                vectorized_shell_groups.append(
                    {
                        "shell_order": "S3",
                        "num_elements": _exact_len(cached_ids),
                        "kernel": "qualified_s3_exact_element_cache_reuse",
                        "parallel_kernel": False,
                        "unique_geometry_count": _exact_len(
                            {cache_key_by_id[element_id] for element_id in cached_ids}
                        ),
                        "component_evaluation_count": 0,
                        "formulation_id": _reference_s3_formulation_id,
                        "speedup_claimed": False,
                    }
                )
        if q4_records:
            unique_q4 = _exact_len({record[3] for record in q4_records})
            vectorized_shell_groups.append(
                {
                    "shell_order": "S4",
                    "num_elements": len(q4_records),
                    "kernel": "e4_pl_shared_geometry_cache",
                    "parallel_kernel": False,
                    "unique_geometry_count": unique_q4,
                }
            )
            diagnostics["qualified_e4_pl_stiffness"] = {
                "path": "shared_geometry_cache",
                "element_count": _exact_len(q4_records),
                "unique_geometry_count": unique_q4,
            }
        diagnostics["vectorized_shell_groups"] = vectorized_shell_groups
        diagnostics["vectorized_shell_element_count"] = _exact_len(records)
        diagnostics["scalar_shell_element_count"] = 0
        info = {
            "matrix_type": matrix_type,
            "num_elements": _exact_len(records),
            "num_nodes": _owned_execution_plan["num_nodes"],
            "total_dofs": total_dofs,
            "assembly_time": _clock() - start_time,
            "element_times": {record[0]: 0.0 for record in records},
            "skipped_elements": [],
            "diagnostics": diagnostics,
            "revision_signature": _exact_dict(
                _owned_execution_plan["revision_signature"]
            ),
            "sparsity_signature": _owned_execution_plan["sparsity_signature"],
        }
        return matrix, info

    owned_mesh = getattr(qualified_runtime_guard, "_qualified_owned_mesh", None)
    raw_mesh = getattr(qualified_runtime_guard, "_qualified_q4_raw_mesh", None)
    mesh = (
        raw_mesh
        if raw_mesh is not None
        else owned_mesh
        if owned_mesh is not None
        else model.mesh
    )
    owned_items_provider = getattr(
        qualified_runtime_guard,
        "_qualified_owned_element_items",
        None,
    )
    owned_material_provider = getattr(
        qualified_runtime_guard,
        "_qualified_owned_material",
        None,
    )
    owned_material_name_provider = getattr(
        qualified_runtime_guard,
        "_qualified_owned_material_name",
        None,
    )
    fast_items_provider = getattr(
        qualified_runtime_guard,
        "_qualified_fast_element_items",
        None,
    )
    raw_items_provider = getattr(
        qualified_runtime_guard,
        "_qualified_q4_raw_element_items",
        None,
    )
    raw_material_provider = getattr(
        qualified_runtime_guard,
        "_qualified_q4_raw_material",
        None,
    )
    raw_total_provider = getattr(
        qualified_runtime_guard,
        "_qualified_q4_cached_total",
        None,
    )
    raw_s3_material_provider = getattr(
        qualified_runtime_guard,
        "_qualified_s3_raw_material",
        None,
    )
    raw_s3_total_provider = getattr(
        qualified_runtime_guard,
        "_qualified_s3_cached_total",
        None,
    )
    raw_s3_reference_plan = getattr(
        qualified_runtime_guard,
        "_qualified_s3_reference_plan",
        None,
    )
    raw_q4_only = bool(
        getattr(qualified_runtime_guard, "_qualified_q4_only", False)
    )
    raw_exact_cached_only = bool(
        getattr(
            qualified_runtime_guard,
            "_qualified_exact_cached_stiffness_only",
            False,
        )
    )
    fast_plan_started = time.time()
    element_items = (
        fast_items_provider()
        if callable(fast_items_provider)
        else raw_items_provider()
        if callable(raw_items_provider)
        else owned_items_provider()
        if callable(owned_items_provider)
        else tuple(mesh.elements.items())
    )

    if matrix_type == "stiffness":
        from .s3_v2c_fast_assembly import lookup_v2c_global_stiffness_plan
        from .s3_v2d_fast_assembly import (
            V2D_GLOBAL_ASSEMBLY_POLICY_ID,
            lookup_v2d_global_stiffness_plan,
        )

        global_v2c_plan = lookup_v2c_global_stiffness_plan(model, element_items)
        if global_v2c_plan is not None:
            qualified_runtime_guard(
                model,
                context="stiffness assembly V2C global-plan preflight",
            )
            data = np.frombuffer(global_v2c_plan.data_bytes, dtype=np.float64)
            indices = np.frombuffer(
                global_v2c_plan.indices_bytes,
                dtype=np.dtype(global_v2c_plan.indices_dtype),
            )
            indptr = np.frombuffer(
                global_v2c_plan.indptr_bytes,
                dtype=np.dtype(global_v2c_plan.indptr_dtype),
            )
            matrix = _csr_constructor(
                (data, indices, indptr),
                shape=global_v2c_plan.shape,
                copy=False,
            )
            info = json.loads(global_v2c_plan.info_bytes)
            info["element_times"] = {
                int(element_id): float(value)
                for element_id, value in info["element_times"].items()
            }
            info["diagnostics"]["s3_v2c_exact_stiffness"][
                "plan_reused"
            ] = True
            info["assembly_time"] = float(time.time() - fast_plan_started)
            return matrix, info

        global_v2d_plan = lookup_v2d_global_stiffness_plan(model, element_items)
        if global_v2d_plan is not None:
            qualified_runtime_guard(
                model,
                context="stiffness assembly V2D global-plan preflight",
            )
            data = np.frombuffer(global_v2d_plan.data_bytes, dtype=np.float64)
            indices = np.frombuffer(
                global_v2d_plan.indices_bytes,
                dtype=np.dtype(global_v2d_plan.indices_dtype),
            )
            indptr = np.frombuffer(
                global_v2d_plan.indptr_bytes,
                dtype=np.dtype(global_v2d_plan.indptr_dtype),
            )
            matrix = _csr_constructor(
                (data, indices, indptr),
                shape=global_v2d_plan.shape,
                copy=False,
            )
            info = json.loads(global_v2d_plan.info_bytes)
            info["element_times"] = {
                int(element_id): float(value)
                for element_id, value in info["element_times"].items()
            }
            info["diagnostics"]["s3_v2d_exact_stiffness"][
                "plan_reused"
            ] = True
            info["diagnostics"]["s3_v2d_global_stiffness"] = {
                "plan_reused": True,
                "policy_id": V2D_GLOBAL_ASSEMBLY_POLICY_ID,
            }
            info["assembly_time"] = float(time.time() - fast_plan_started)
            return matrix, info

    def observed_material(name: Any, *, context: str) -> Any:
        if callable(owned_material_name_provider):
            material = owned_material_name_provider(name)
            if material is None:
                raise AssemblyError(
                    f"{context} found no exact material authority"
                )
            return material
        material = model.get_material(name)
        qualified_runtime_guard(model, context=context)
        return material

    total_dofs = mesh.dof_manager.total_dofs
    info = _base_info_kernel(model, matrix_type)
    start_time = time.time()
    quantity = activity_quantity or (
        "stiffness" if matrix_type == "geometric_stiffness" else matrix_type
    )
    raw_mesh_namespace = (
        object.__getattribute__(mesh, "__dict__")
        if raw_exact_cached_only
        else None
    )
    if (
        raw_exact_cached_only
        and type(raw_mesh_namespace) is dict
        and dict.get(raw_mesh_namespace, "element_activity") is None
    ):
        _activity, activity_scales, activity_info = None, {}, None
    else:
        _activity, activity_scales, activity_info = _activity_scales_kernel(
            model,
            quantity,
        )
        qualified_runtime_guard(
            model,
            context=f"{matrix_type} assembly activity",
        )
    if activity_info is not None:
        info["diagnostics"]["element_activity"] = activity_info

    # Precompute shell matrices in a JIT-compiled batch for stiffness and mass assembly
    precomputed = {}
    prevalidated_element_ids: set[int] = set()
    vectorized_shell_groups = []
    if matrix_type in {"stiffness", "mass"}:
        from .elements import ShellElement
        from .e4_pl_element import (
            FORMULATION_ID as QUALIFIED_Q4_FORMULATION_ID,
            QualifiedE4PLShellElement,
        )
        from .jit_compiler import JIT_ENABLED, JIT_DISABLED_REASON, jit_diagnostics
        from .materials import is_isotropic_material
        from .vectorized_stiffness import compute_shell_mass_matrices_jit, compute_shell_stiffness_matrices_jit
        from .vectorized_generalized_shell import (
            prepare_s4_generalized_stiffness_batch,
            prepare_s4_section_mass_batch,
        )
        from .s3_reference_batch import (
            get_reference_s3_stiffness_components,
            reference_s3_candidate,
        )
        from .s3_v2c_fast_assembly import (
            get_v2c_stiffness_plan,
            v2c_fast_candidate,
        )
        from .s3_v2d_fast_assembly import (
            get_v2d_stiffness_plan,
            v2d_batch_eligibility,
            v2d_fast_candidate,
        )

        groups = {}
        reference_s3_items = []
        v2c_stiffness_items = []
        v2d_stiffness_items = []
        cached_s3_stiffness_items = []
        qualified_stiffness_items = []
        advanced_stiffness_items = []
        section_mass_items = []
        constitutive_fallback_ids = []
        generalized_section_fallback_ids = []
        generalized_mass_fallback_ids = []
        for elem_id, element in element_items:
            if (
                matrix_type == "stiffness"
                and type(element) is _QualifiedE4PLS3ShellElement
                and callable(raw_s3_total_provider)
            ):
                cached_s3_total = raw_s3_total_provider(element)
                if cached_s3_total is not None:
                    cached_s3_stiffness_items.append((int(elem_id), element))
                    precomputed[int(elem_id)] = cached_s3_total
                    prevalidated_element_ids.add(int(elem_id))
                    continue
            if (
                matrix_type == "stiffness"
                and type(element) is QualifiedE4PLShellElement
                and callable(raw_total_provider)
                and raw_total_provider(element) is not None
            ):
                # The outer lease already bound exact routing, material,
                # quadrature, total bytes and the full model input snapshot.
                # Do not re-enter public descriptors while classifying this
                # provider-free warm record.
                qualified_stiffness_items.append((int(elem_id), element))
                continue
            formulation_id = str(getattr(element, "formulation_id", ""))
            if (
                type(element) is QualifiedE4PLShellElement
                or formulation_id == QUALIFIED_Q4_FORMULATION_ID
            ):
                if (
                    type(element) is not QualifiedE4PLShellElement
                    or formulation_id != QUALIFIED_Q4_FORMULATION_ID
                ):
                    raise AssemblyError(
                        f"Element {elem_id} has incompatible qualified Q4 authority"
                    )
            material = (
                raw_material_provider(element)
                if callable(raw_material_provider)
                and type(element) is QualifiedE4PLShellElement
                else owned_material_provider(element)
                if callable(owned_material_provider)
                and type(element)
                in {QualifiedE4PLShellElement, _QualifiedE4PLS3ShellElement}
                else observed_material(
                    element.material_name,
                    context=f"{matrix_type} assembly material observation",
                )
            )
            if material is None:
                raise AssemblyError(
                    f"Element {elem_id} has no exact material authority"
                )
            shell_section = getattr(element, "shell_section", None)
            has_section_mass = bool(
                shell_section is not None
                and (
                    getattr(shell_section, "mass_per_area", None) is not None
                    or getattr(shell_section, "rotary_inertia_per_area", None) is not None
                )
            )
            if (
                matrix_type == "stiffness"
                and v2c_fast_candidate(element)
            ):
                # V2C owns a revision-bound exact-matrix plan.  It remains
                # distinct from both legacy TRI3 and the qualified S3 V1
                # shared-component path.
                v2c_stiffness_items.append((int(elem_id), element))
                continue
            if (
                matrix_type == "stiffness"
                and v2d_fast_candidate(element)
                and v2d_batch_eligibility(model, element)[0]
            ):
                v2d_stiffness_items.append((int(elem_id), element))
                continue
            if (
                matrix_type == "stiffness"
                and reference_s3_candidate(element)
            ):
                # Qualified S3 has a formulation-native reference-elastic
                # batch.  It must never enter either the legacy TRI3 or the
                # qualified-Q4 kernels below, including on scalar fallback.
                reference_s3_items.append((int(elem_id), element))
                continue
            if (
                matrix_type == "stiffness"
                and isinstance(element, ShellElement)
                and not bool(getattr(element, "legacy_stiffness_batch_eligible", True))
                and hasattr(element, "_qualified_stiffness_cache_key")
                and hasattr(element, "_adopt_qualified_components")
            ):
                qualified_stiffness_items.append((int(elem_id), element))
                continue
            if (
                matrix_type == "stiffness"
                and isinstance(element, ShellElement)
                and bool(getattr(element, "legacy_stiffness_batch_eligible", True))
                and bool(getattr(element, "_is_4node", False))
                and (
                    shell_section is not None
                    or not is_isotropic_material(material)
                )
            ):
                advanced_stiffness_items.append((int(elem_id), element))
                continue
            if (
                matrix_type == "mass"
                and isinstance(element, ShellElement)
                and bool(getattr(element, "_is_4node", False))
                and has_section_mass
            ):
                section_mass_items.append((int(elem_id), element))
                continue
            if (
                isinstance(element, ShellElement)
                and getattr(element, "_is_quadrilateral", False)
                and not (getattr(element, "_is_8node", False) and bool(getattr(element, "reduced_integration", False)))
                and (
                    (matrix_type == "mass" and not has_section_mass)
                    or (
                        matrix_type == "stiffness"
                        and bool(getattr(element, "legacy_stiffness_batch_eligible", True))
                        and shell_section is None
                        and is_isotropic_material(material)
                    )
                )
            ):
                primary_quadrature = _shell_quadrature_batch_identity(
                    element,
                    include_shear=matrix_type == "stiffness",
                )
                key = (
                    element.num_nodes,
                    element.thickness,
                    element.drilling_stabilization,
                    element.reduced_integration,
                    element.hourglass_stabilization,
                    element.material_name,
                    type(element),
                    str(getattr(element, "formulation_id", "")),
                    primary_quadrature,
                )
                if key not in groups:
                    groups[key] = []
                groups[key].append((elem_id, element))
            elif (
                matrix_type == "stiffness"
                and isinstance(element, ShellElement)
                and shell_section is None
                and not is_isotropic_material(material)
            ):
                constitutive_fallback_ids.append(int(elem_id))
            elif (
                matrix_type == "stiffness"
                and isinstance(element, ShellElement)
                and shell_section is not None
            ):
                generalized_section_fallback_ids.append(int(elem_id))
            elif (
                matrix_type == "mass"
                and isinstance(element, ShellElement)
                and has_section_mass
            ):
                generalized_mass_fallback_ids.append(int(elem_id))

        if constitutive_fallback_ids:
            info["diagnostics"]["constitutive_fallback"] = {
                "path": "general_element",
                "reason": "orthotropic_material",
                "element_ids": sorted(constitutive_fallback_ids),
            }
        if generalized_section_fallback_ids:
            info["diagnostics"]["generalized_shell_section_fallback"] = {
                "path": "general_element",
                "reason": "preintegrated_generalized_shell_section",
                "element_ids": sorted(generalized_section_fallback_ids),
            }
        if generalized_mass_fallback_ids:
            info["diagnostics"]["generalized_shell_section_mass_fallback"] = {
                "path": "general_element",
                "reason": "unsupported_shell_topology",
                "element_ids": sorted(generalized_mass_fallback_ids),
            }

        if cached_s3_stiffness_items:
            prepared_s3 = raw_s3_reference_plan
            s3_diagnostics = prepared_s3.diagnostics()
            s3_diagnostics["plan_reused"] = True
            info["diagnostics"]["qualified_s3_reference_elastic_stiffness"] = (
                s3_diagnostics
            )
            if prepared_s3.batched_element_ids:
                vectorized_shell_groups.append(
                    {
                        "shell_order": "S3",
                        "num_elements": len(prepared_s3.batched_element_ids),
                        "kernel": (
                            "qualified_s3_reference_elastic_shared_components"
                        ),
                        "parallel_kernel": False,
                        "unique_geometry_count": len(
                            prepared_s3.group_element_ids
                        ),
                        "component_evaluation_count": (
                            prepared_s3.component_evaluation_count
                        ),
                        "formulation_id": s3_diagnostics["formulation_id"],
                        "speedup_claimed": False,
                    }
                )
            if prepared_s3.cached_element_ids:
                vectorized_shell_groups.append(
                    {
                        "shell_order": "S3",
                        "num_elements": len(prepared_s3.cached_element_ids),
                        "kernel": "qualified_s3_exact_element_cache_reuse",
                        "parallel_kernel": False,
                        "unique_geometry_count": len(
                            {
                                prepared_s3.element_cache_keys[element_id]
                                for element_id in prepared_s3.cached_element_ids
                            }
                        ),
                        "component_evaluation_count": 0,
                        "formulation_id": s3_diagnostics["formulation_id"],
                        "speedup_claimed": False,
                    }
                )

        if reference_s3_items:
            prepared_s3, s3_plan_reused = get_reference_s3_stiffness_components(
                model,
                reference_s3_items,
                complete_candidate_items=True,
            )
            precomputed.update(prepared_s3.matrices)
            if prepared_s3.matrices_prevalidated:
                # The plan owns bytes-backed immutable matrices and binds the
                # complete S3 eligibility/component-key preimage.  Shape,
                # finiteness and symmetry were checked once when those exact
                # arrays entered the plan, so warm assembly need not repeat
                # three dense validations for every unchanged element.
                prevalidated_element_ids.update(prepared_s3.matrices)
            s3_diagnostics = prepared_s3.diagnostics()
            s3_diagnostics["plan_reused"] = bool(s3_plan_reused)
            info["diagnostics"]["qualified_s3_reference_elastic_stiffness"] = (
                s3_diagnostics
            )
            if prepared_s3.batched_element_ids:
                vectorized_shell_groups.append(
                    {
                        "shell_order": "S3",
                        "num_elements": len(prepared_s3.batched_element_ids),
                        "kernel": (
                            "qualified_s3_reference_elastic_shared_components"
                        ),
                        "parallel_kernel": False,
                        "unique_geometry_count": len(
                            prepared_s3.group_element_ids
                        ),
                        "component_evaluation_count": (
                            prepared_s3.component_evaluation_count
                        ),
                        "formulation_id": s3_diagnostics["formulation_id"],
                        "speedup_claimed": False,
                    }
                )
            if prepared_s3.cached_element_ids:
                vectorized_shell_groups.append(
                    {
                        "shell_order": "S3",
                        "num_elements": len(prepared_s3.cached_element_ids),
                        "kernel": "qualified_s3_exact_element_cache_reuse",
                        "parallel_kernel": False,
                        "unique_geometry_count": len(
                            {
                                prepared_s3.element_cache_keys[element_id]
                                for element_id in prepared_s3.cached_element_ids
                            }
                        ),
                        "component_evaluation_count": 0,
                        "formulation_id": s3_diagnostics["formulation_id"],
                        "speedup_claimed": False,
                    }
                )

        if v2c_stiffness_items:
            prepared_v2c, v2c_plan_reused = get_v2c_stiffness_plan(
                model,
                v2c_stiffness_items,
            )
            precomputed.update(prepared_v2c.matrices)
            if prepared_v2c.matrices_prevalidated:
                prevalidated_element_ids.update(prepared_v2c.matrices)
            v2c_diagnostics = prepared_v2c.diagnostics()
            v2c_diagnostics["plan_reused"] = bool(v2c_plan_reused)
            info["diagnostics"]["s3_v2c_exact_stiffness"] = v2c_diagnostics
            vectorized_shell_groups.append(
                {
                    "shell_order": "S3",
                    "num_elements": len(prepared_v2c.element_ids),
                    "kernel": "s3_v2c_exact_revision_bound_matrix_plan",
                    "parallel_kernel": False,
                    "unique_geometry_count": len(prepared_v2c.element_ids),
                    "component_evaluation_count": (
                        0 if v2c_plan_reused else len(prepared_v2c.element_ids)
                    ),
                    "formulation_id": v2c_diagnostics["formulation_id"],
                    "speedup_claimed": False,
                }
            )

        if v2d_stiffness_items:
            prepared_v2d, v2d_plan_reused = get_v2d_stiffness_plan(
                model,
                v2d_stiffness_items,
            )
            precomputed.update(prepared_v2d.matrices)
            if prepared_v2d.matrices_prevalidated:
                prevalidated_element_ids.update(prepared_v2d.matrices)
            v2d_diagnostics = prepared_v2d.diagnostics()
            v2d_diagnostics["plan_reused"] = bool(v2d_plan_reused)
            info["diagnostics"]["s3_v2d_exact_stiffness"] = v2d_diagnostics
            vectorized_shell_groups.append(
                {
                    "shell_order": "S3",
                    "num_elements": len(prepared_v2d.element_ids),
                    "kernel": "s3_v2d_exact_revision_bound_stiffness_plan",
                    "parallel_kernel": False,
                    "unique_geometry_count": len(prepared_v2d.element_ids),
                    "component_evaluation_count": (
                        0 if v2d_plan_reused else len(prepared_v2d.element_ids)
                    ),
                    "formulation_id": v2d_diagnostics["formulation_id"],
                    "speedup_claimed": False,
                }
            )

        if qualified_stiffness_items:
            shared_components = {}
            for element_id, element in qualified_stiffness_items:
                material = (
                    raw_material_provider(element)
                    if callable(raw_material_provider)
                    else owned_material_provider(element)
                    if callable(owned_material_provider)
                    else observed_material(
                        element.material_name,
                        context=(
                            f"{matrix_type} assembly qualified material "
                            "observation"
                        ),
                    )
                )
                cached_total = (
                    raw_total_provider(element)
                    if callable(raw_total_provider)
                    else _TRY_Q4_FAST_CACHED_STIFFNESS(
                        element,
                        mesh,
                        material,
                    )
                )
                if cached_total is not None:
                    namespace = object.__getattribute__(element, "__dict__")
                    cache_key = dict.get(namespace, "_qualified_cache_key")
                    current_components = dict.get(
                        namespace,
                        "_qualified_components",
                    )
                    shared_components.setdefault(cache_key, current_components)
                    precomputed[element_id] = cached_total
                    prevalidated_element_ids.add(int(element_id))
                    continue
                cache_key = element._qualified_stiffness_cache_key(mesh, material)
                current_components = getattr(element, "_qualified_components", None)
                if (
                    current_components is not None
                    and getattr(element, "_qualified_cache_key", None) == cache_key
                ):
                    element._validate_qualified_component_cache_identity()
                    shared_components.setdefault(cache_key, current_components)
                    element._bind_qualified_component_guard(mesh, material)
                    precomputed[element_id] = np.asarray(
                        current_components["total"], dtype=float
                    )
                    continue
                components = shared_components.get(cache_key)
                if components is None:
                    precomputed[element_id] = element.compute_stiffness_matrix(
                        mesh, material
                    )
                    components = element._qualified_components
                    if components is None:
                        raise RuntimeError(
                            "Qualified E4-PL stiffness did not populate its component cache"
                        )
                    shared_components[cache_key] = components
                else:
                    precomputed[element_id] = element._adopt_qualified_components(
                        cache_key,
                        components,
                        mesh,
                        material,
                    )
            vectorized_shell_groups.append(
                {
                    "shell_order": "S4",
                    "num_elements": int(len(qualified_stiffness_items)),
                    "kernel": "e4_pl_shared_geometry_cache",
                    "parallel_kernel": False,
                    "unique_geometry_count": int(len(shared_components)),
                }
            )
            info["diagnostics"]["qualified_e4_pl_stiffness"] = {
                "path": "shared_geometry_cache",
                "element_count": int(len(qualified_stiffness_items)),
                "unique_geometry_count": int(len(shared_components)),
            }

        if not raw_exact_cached_only:
            qualified_runtime_guard(
                model,
                context=f"{matrix_type} assembly prepared operators",
            )

        for key, elem_list in groups.items():
            (
                num_nodes,
                thickness,
                drilling_stabilization,
                _reduced_integration,
                _hourglass_stabilization,
                material_name,
                _concrete_type,
                _formulation_id,
                _quadrature_identity,
            ) = key
            material = observed_material(
                material_name,
                context=f"{matrix_type} assembly batch material observation",
            )

            n_elem = len(elem_list)
            coords_all = np.zeros((n_elem, num_nodes, 3))
            for idx, (elem_id, element) in enumerate(elem_list):
                coords_all[idx] = element.get_node_coordinates(mesh)

            first_element = elem_list[0][1]
            is_4node = first_element._is_4node
            gauss_points = first_element.gauss_points
            gauss_weights = first_element.gauss_weights

            if matrix_type == "mass":
                kernel_name = "compute_shell_mass_matrices_jit"
                batched = compute_shell_mass_matrices_jit(
                    coords_all,
                    is_4node,
                    thickness,
                    float(material.density),
                    gauss_points,
                    gauss_weights,
                )
            else:
                kernel_name = "compute_shell_stiffness_matrices_jit"
                E = float(material.elastic_modulus)
                nu = float(material.poisson_ratio)
                G = float(material.shear_modulus)
                if is_4node:
                    shear_points = np.empty((0, 2))
                    shear_weights = np.empty(0)
                else:
                    shear_points = first_element.shear_gauss_points
                    shear_weights = first_element.shear_gauss_weights
                batched = compute_shell_stiffness_matrices_jit(
                    coords_all,
                    is_4node,
                    thickness,
                    drilling_stabilization,
                    E,
                    nu,
                    G,
                    gauss_points,
                    gauss_weights,
                    shear_points,
                    shear_weights,
                )

            for idx, (elem_id, element) in enumerate(elem_list):
                precomputed[elem_id] = batched[idx]
            jit_info = jit_diagnostics()
            vectorized_shell_groups.append(
                {
                    "shell_order": "S4" if is_4node else "Q8",
                    "num_elements": int(n_elem),
                    "num_nodes": int(num_nodes),
                    "material": str(material_name),
                    "thickness": float(thickness),
                    "jit_enabled": bool(JIT_ENABLED),
                    "jit_disabled_reason": JIT_DISABLED_REASON,
                    "kernel": kernel_name,
                    "parallel_kernel": True,
                    "parallel_threads": jit_info.get("num_threads"),
                    "backend": jit_info.get("backend"),
                }
            )

        if advanced_stiffness_items:
            advanced_groups: Dict[tuple[Any, ...], list[tuple[int, Any]]] = {}
            for element_id, element in advanced_stiffness_items:
                key = (
                    type(element),
                    str(getattr(element, "formulation_id", "")),
                    _shell_quadrature_batch_identity(
                        element,
                        include_shear=True,
                    ),
                )
                advanced_groups.setdefault(key, []).append(
                    (int(element_id), element)
                )
            advanced_group_records: list[tuple[int, float, Dict[str, int]]] = []
            advanced_counts: Dict[str, int] = {
                "orthotropic_element_count": 0,
                "generalized_element_count": 0,
            }
            for advanced_group in advanced_groups.values():
                group_start = time.perf_counter()
                advanced_matrices, group_counts = (
                    prepare_s4_generalized_stiffness_batch(
                        model,
                        [
                            element
                            for _element_id, element in advanced_group
                        ],
                    )
                )
                group_seconds = time.perf_counter() - group_start
                for index, (element_id, _element) in enumerate(advanced_group):
                    precomputed[element_id] = advanced_matrices[index]
                for name, value in group_counts.items():
                    advanced_counts[name] = advanced_counts.get(name, 0) + int(
                        value
                    )
                advanced_group_records.append(
                    (len(advanced_group), group_seconds, group_counts)
                )
            jit_info = jit_diagnostics()
            for group_size, group_seconds, group_counts in advanced_group_records:
                vectorized_shell_groups.append(
                    {
                        "shell_order": "S4",
                        "num_elements": int(group_size),
                        "jit_enabled": bool(JIT_ENABLED),
                        "jit_disabled_reason": JIT_DISABLED_REASON,
                        "kernel": "compute_s4_generalized_stiffness_matrices_jit",
                        "parallel_kernel": True,
                        "parallel_threads": jit_info.get("num_threads"),
                        "backend": jit_info.get("backend"),
                        "kernel_seconds": float(group_seconds),
                        **group_counts,
                    }
                )
            info["diagnostics"]["advanced_s4_stiffness"] = {
                "path": "compiled_batch",
                **advanced_counts,
            }

        if section_mass_items:
            section_mass_groups: Dict[tuple[Any, ...], list[tuple[int, Any]]] = {}
            for element_id, element in section_mass_items:
                key = (
                    type(element),
                    str(getattr(element, "formulation_id", "")),
                    _shell_quadrature_batch_identity(
                        element,
                        include_shear=False,
                    ),
                )
                section_mass_groups.setdefault(key, []).append(
                    (int(element_id), element)
                )
            section_mass_group_records: list[tuple[int, float]] = []
            for section_mass_group in section_mass_groups.values():
                group_start = time.perf_counter()
                section_mass_matrices = prepare_s4_section_mass_batch(
                    model,
                    [
                        element
                        for _element_id, element in section_mass_group
                    ],
                )
                for index, (element_id, _element) in enumerate(
                    section_mass_group
                ):
                    precomputed[element_id] = section_mass_matrices[index]
                section_mass_group_records.append(
                    (
                        len(section_mass_group),
                        time.perf_counter() - group_start,
                    )
                )
            jit_info = jit_diagnostics()
            for group_size, group_seconds in section_mass_group_records:
                vectorized_shell_groups.append(
                    {
                        "shell_order": "S4",
                        "num_elements": int(group_size),
                        "jit_enabled": bool(JIT_ENABLED),
                        "jit_disabled_reason": JIT_DISABLED_REASON,
                        "kernel": "compute_s4_section_mass_matrices_jit",
                        "parallel_kernel": True,
                        "parallel_threads": jit_info.get("num_threads"),
                        "backend": jit_info.get("backend"),
                        "kernel_seconds": float(group_seconds),
                        "generalized_section_mass_element_count": int(
                            group_size
                        ),
                    }
                )
            info["diagnostics"]["generalized_s4_section_mass"] = {
                "path": "compiled_batch",
                "element_count": int(len(section_mass_items)),
            }

    # Retrieve or build cached sparsity pattern
    rows_concat, cols_concat = _sparsity_kernel(
        mesh,
        matrix_type,
        element_items=element_items,
    )

    data_list = []
    for elem_id, element in element_items:
        elem_start = time.time()
        raw_q4_material = (
            raw_material_provider(element)
            if callable(raw_material_provider)
            and type(element) is _QualifiedE4PLShellElement
            else None
        )
        raw_s3_material = (
            raw_s3_material_provider(element)
            if callable(raw_s3_material_provider)
            and type(element) is _QualifiedE4PLS3ShellElement
            else None
        )
        owned_qualified_material = (
            owned_material_provider(element)
            if callable(owned_material_provider)
            and type(element)
            in {_QualifiedE4PLShellElement, _QualifiedE4PLS3ShellElement}
            else None
        )
        material = (
            raw_q4_material
            if raw_q4_material is not None
            else raw_s3_material
            if raw_s3_material is not None
            else owned_qualified_material
            if owned_qualified_material is not None
            else observed_material(
                element.material_name,
                context=f"{matrix_type} assembly element material observation",
            )
        )
        raw_q4_prevalidated = (
            raw_q4_material is not None
            and int(elem_id) in prevalidated_element_ids
            and callable(raw_total_provider)
        )
        raw_s3_prevalidated = (
            raw_s3_material is not None
            and int(elem_id) in prevalidated_element_ids
            and callable(raw_s3_total_provider)
        )
        dof_mapping = (
            None
            if raw_q4_prevalidated or raw_s3_prevalidated
            else np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        )
        dof_size = (
            24
            if raw_q4_prevalidated
            else 18
            if raw_s3_prevalidated
            else int(dof_mapping.size)
        )
        if dof_size == 0:
            info["skipped_elements"].append(int(elem_id))
            continue

        if elem_id in precomputed:
            element_matrix = precomputed[elem_id]
        else:
            element_matrix = element_matrix_getter(element, mesh, material)

        matrix_prevalidated = (
            int(elem_id) in prevalidated_element_ids
            and dof_size in {18, 24}
        )
        if not matrix_prevalidated:
            element_matrix = _check_matrix_kernel(
                int(elem_id),
                matrix_type,
                element_matrix,
                dof_size,
            )
        if (
            matrix_type in {"stiffness", "mass", "geometric_stiffness"}
            and not matrix_prevalidated
        ):
            local_symmetry = _relative_symmetry_kernel(element_matrix)
            if local_symmetry > 1.0e-8:
                raise AssemblyError(
                    f"Element {elem_id} returned nonsymmetric {matrix_type}; "
                    f"relative symmetry error {local_symmetry:.3e}."
                )
        scale = activity_scales.get(int(elem_id), 1.0)
        data_list.append(
            (scale * np.asarray(element_matrix, dtype=float)).ravel()
        )

        info["element_times"][int(elem_id)] = time.time() - elem_start
        info["num_elements"] += 1

    if not raw_exact_cached_only:
        qualified_runtime_guard(
            model,
            context=f"{matrix_type} assembly completed elements",
        )

    if not data_list:
        matrix = _csr_constructor((total_dofs, total_dofs), dtype=float)
        info["diagnostics"]["assembled_symmetry_error"] = 0.0
        info["sparsity_signature"] = _topology_signature_kernel(
            mesh,
            matrix_type,
            element_items=element_items,
        )
        info["assembly_time"] = time.time() - start_time
        return matrix, info

    data_concat = np.concatenate(data_list)
    coo = _coo_constructor(
        (data_concat, (rows_concat, cols_concat)),
        shape=(total_dofs, total_dofs),
        dtype=float,
    )
    matrix = _coo_to_csr(coo)
    if activity_info is not None and activity_info["zero_contribution_count"]:
        _eliminate_zeros(matrix)
    info["diagnostics"]["assembled_symmetry_error"] = _relative_symmetry_kernel(matrix)
    if matrix_type in {"stiffness", "mass"}:
        info["diagnostics"]["vectorized_shell_groups"] = vectorized_shell_groups
        info["diagnostics"]["vectorized_shell_element_count"] = int(len(precomputed))
        info["diagnostics"]["scalar_shell_element_count"] = int(info["num_elements"] - len(precomputed))
    info["sparsity_signature"] = _topology_signature_kernel(
        mesh,
        matrix_type,
        element_items=element_items,
    )
    info["assembly_time"] = time.time() - start_time
    if matrix_type == "stiffness" and v2c_stiffness_items:
        from .s3_v2c_fast_assembly import bind_v2c_global_stiffness_plan

        bind_v2c_global_stiffness_plan(model, element_items, matrix, info)
    if matrix_type == "stiffness" and v2d_stiffness_items:
        from .s3_v2d_fast_assembly import bind_v2d_global_stiffness_plan

        bind_v2d_global_stiffness_plan(model, element_items, matrix, info)
    return matrix, info


def _make_exact_assembly_operation(implementation: Any) -> tuple[Any, Any]:
    """Bind kernel routing in a closure, not mutable function defaults.

    The published implementation object remains inspectable for diagnostics,
    but qualified calls execute a fresh private function made from the exact
    captured code and receive every kernel explicitly.  Mutating ``__code__``,
    ``__defaults__`` or ``__kwdefaults__`` can therefore neither dispatch an
    attacker nor silently redefine an already-authorized assembly.
    """

    expected_code = implementation.__code__
    expected_defaults = implementation.__defaults__
    expected_kwdefaults = implementation.__kwdefaults__
    if type(expected_kwdefaults) is not dict:
        raise TypeError("qualified assembly implementation defaults are absent")
    expected_kw_items = tuple(expected_kwdefaults.items())
    exact_globals = dict(implementation.__globals__)
    exact_globals["__builtins__"] = dict(implementation.__builtins__)
    exact_name = implementation.__name__
    function_type = FunctionType
    plan_lookup = _LOOKUP_QUALIFIED_ASSEMBLY_EXECUTION_PLAN
    plan_lookup_code = plan_lookup.__code__
    plan_lookup_globals = dict(plan_lookup.__globals__)
    plan_lookup_globals["__builtins__"] = dict(plan_lookup.__builtins__)
    plan_lookup_name = plan_lookup.__name__
    plan_lookup_defaults = plan_lookup.__defaults__
    plan_lookup_closure = plan_lookup.__closure__
    runtime_module = sys.modules[__name__]
    published: list[Any] = []

    def private_function(function: Any) -> Any:
        if type(function) is not function_type:
            return function
        private_globals = dict(function.__globals__)
        private_globals["__builtins__"] = dict(function.__builtins__)
        clone = function_type(
            function.__code__,
            private_globals,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        clone.__kwdefaults__ = (
            None
            if function.__kwdefaults__ is None
            else dict(function.__kwdefaults__)
        )
        return clone

    exact_activity_scales = private_function(
        expected_kwdefaults["_activity_scales_kernel"]
    )
    exact_base_info = private_function(expected_kwdefaults["_base_info_kernel"])
    exact_check_matrix = private_function(
        expected_kwdefaults["_check_matrix_kernel"]
    )
    exact_relative_symmetry = private_function(
        expected_kwdefaults["_relative_symmetry_kernel"]
    )
    exact_sparsity = private_function(expected_kwdefaults["_sparsity_kernel"])
    exact_topology_signature = private_function(
        expected_kwdefaults["_topology_signature_kernel"]
    )
    exact_coo_constructor = expected_kwdefaults["_coo_constructor"]
    exact_csr_constructor = expected_kwdefaults["_csr_constructor"]
    exact_coo_to_csr = private_function(expected_kwdefaults["_coo_to_csr"])
    exact_eliminate_zeros = private_function(
        expected_kwdefaults["_eliminate_zeros"]
    )
    exact_object_new = expected_kwdefaults["_object_new"]
    exact_object_setattr = expected_kwdefaults["_object_setattr"]
    exact_object_getattribute = expected_kwdefaults["_object_getattribute"]
    exact_ndarray_constructor = expected_kwdefaults["_ndarray_constructor"]
    exact_empty_constructor = expected_kwdefaults["_empty_constructor"]
    exact_intp_dtype = expected_kwdefaults["_intp_dtype"]
    exact_float64_dtype = expected_kwdefaults["_float64_dtype"]
    exact_coo_to_csr_kernel = expected_kwdefaults["_coo_to_csr_kernel"]
    exact_csr_sort_indices_kernel = expected_kwdefaults[
        "_csr_sort_indices_kernel"
    ]
    exact_csr_sum_duplicates_kernel = expected_kwdefaults[
        "_csr_sum_duplicates_kernel"
    ]
    exact_csr_has_sorted_indices_kernel = expected_kwdefaults[
        "_csr_has_sorted_indices_kernel"
    ]
    exact_csr_has_canonical_format_kernel = expected_kwdefaults[
        "_csr_has_canonical_format_kernel"
    ]
    exact_isfinite = expected_kwdefaults["_isfinite_kernel"]
    exact_all = expected_kwdefaults["_all_kernel"]
    exact_type = expected_kwdefaults["_exact_type"]
    exact_len = expected_kwdefaults["_exact_len"]
    exact_tuple = expected_kwdefaults["_exact_tuple"]
    exact_dict = expected_kwdefaults["_exact_dict"]
    exact_list = expected_kwdefaults["_exact_list"]
    exact_int = expected_kwdefaults["_exact_int"]
    exact_float_type = expected_kwdefaults["_float_type"]
    exact_mapping_proxy_type = expected_kwdefaults["_mapping_proxy_type"]
    exact_assembly_error = expected_kwdefaults["_assembly_error_type"]
    exact_clock = expected_kwdefaults["_clock"]
    exact_reference_s3_formulation_id = expected_kwdefaults[
        "_reference_s3_formulation_id"
    ]

    def require() -> None:
        current_kwdefaults = implementation.__kwdefaults__
        if len(published) != 1:
            raise ValueError("qualified assembly operation authority changed")
        (
            published_call,
            published_code,
            published_defaults,
            published_kwdefaults,
            published_kw_items,
        ) = published[0]
        current_published_kwdefaults = published_call.__kwdefaults__
        if (
            implementation.__code__ is not expected_code
            or implementation.__defaults__ is not expected_defaults
            or current_kwdefaults is not expected_kwdefaults
            or len(current_kwdefaults) != len(expected_kw_items)
            or any(
                name not in current_kwdefaults
                or current_kwdefaults[name] is not expected
                for name, expected in expected_kw_items
            )
            or published_call.__code__ is not published_code
            or published_call.__defaults__ is not published_defaults
            or current_published_kwdefaults is not published_kwdefaults
            or type(current_published_kwdefaults) is not dict
            or len(current_published_kwdefaults) != len(published_kw_items)
            or any(
                name not in current_published_kwdefaults
                or current_published_kwdefaults[name] is not expected
                for name, expected in published_kw_items
            )
            or vars(runtime_module).get("_assemble_element_matrix_under_lease")
            is not published_call
        ):
            raise ValueError("qualified assembly operation authority changed")

    def call(
        model: "FEModel",
        matrix_type: str,
        element_matrix_getter: Callable[[Any, Any, Any], np.ndarray],
        qualified_runtime_guard: Any,
        *,
        activity_quantity: str | None = None,
    ) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
        require()
        exact_function = function_type(
            expected_code,
            exact_globals,
            exact_name,
            expected_defaults,
        )
        exact_plan_lookup = function_type(
            plan_lookup_code,
            plan_lookup_globals,
            plan_lookup_name,
            plan_lookup_defaults,
            plan_lookup_closure,
        )
        owned_execution_plan = exact_plan_lookup(qualified_runtime_guard)
        return exact_function(
            model,
            matrix_type,
            element_matrix_getter,
            qualified_runtime_guard,
            owned_execution_plan,
            activity_quantity=activity_quantity,
            _activity_scales_kernel=exact_activity_scales,
            _base_info_kernel=exact_base_info,
            _check_matrix_kernel=exact_check_matrix,
            _relative_symmetry_kernel=exact_relative_symmetry,
            _sparsity_kernel=exact_sparsity,
            _topology_signature_kernel=exact_topology_signature,
            _coo_constructor=exact_coo_constructor,
            _csr_constructor=exact_csr_constructor,
            _coo_to_csr=exact_coo_to_csr,
            _eliminate_zeros=exact_eliminate_zeros,
            _object_new=exact_object_new,
            _object_setattr=exact_object_setattr,
            _object_getattribute=exact_object_getattribute,
            _ndarray_constructor=exact_ndarray_constructor,
            _empty_constructor=exact_empty_constructor,
            _intp_dtype=exact_intp_dtype,
            _float64_dtype=exact_float64_dtype,
            _coo_to_csr_kernel=exact_coo_to_csr_kernel,
            _csr_sort_indices_kernel=exact_csr_sort_indices_kernel,
            _csr_sum_duplicates_kernel=exact_csr_sum_duplicates_kernel,
            _csr_has_sorted_indices_kernel=exact_csr_has_sorted_indices_kernel,
            _csr_has_canonical_format_kernel=exact_csr_has_canonical_format_kernel,
            _isfinite_kernel=exact_isfinite,
            _all_kernel=exact_all,
            _exact_type=exact_type,
            _exact_len=exact_len,
            _exact_tuple=exact_tuple,
            _exact_dict=exact_dict,
            _exact_list=exact_list,
            _exact_int=exact_int,
            _float_type=exact_float_type,
            _mapping_proxy_type=exact_mapping_proxy_type,
            _assembly_error_type=exact_assembly_error,
            _clock=exact_clock,
            _reference_s3_formulation_id=exact_reference_s3_formulation_id,
        )

    call_kwdefaults = call.__kwdefaults__
    if type(call_kwdefaults) is not dict:
        raise TypeError("qualified assembly dispatcher defaults are absent")
    published.append(
        (
            call,
            call.__code__,
            call.__defaults__,
            call_kwdefaults,
            tuple(call_kwdefaults.items()),
        )
    )
    return call, require


(
    _assemble_element_matrix_under_lease,
    _require_exact_assembly_operation_metadata,
) = _make_exact_assembly_operation(_assemble_element_matrix_under_lease_impl)
_INSTALL_EXACT_ASSEMBLY_OPERATION_AUTHORITY(
    _require_exact_assembly_operation_metadata
)
_ASSEMBLY_NUMERICAL_EPOCH_MANAGER.watch_module(
    _ASSEMBLY_RUNTIME_MODULE,
    ("_assemble_element_matrix_under_lease",),
)
del _REGISTER_QUALIFIED_ASSEMBLY_EXECUTION_PLAN
del _LOOKUP_QUALIFIED_ASSEMBLY_EXECUTION_PLAN


def _assemble_element_matrix(
    model: "FEModel",
    matrix_type: str,
    element_matrix_getter: Callable[[Any, Any, Any], np.ndarray],
    *,
    activity_quantity: str | None = None,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    return _run_with_qualified_assembly_runtime_lease(
        model,
        context=f"{matrix_type} assembly",
        operation=lambda lease: _assemble_element_matrix_under_lease(
            model,
            matrix_type,
            element_matrix_getter,
            lease,
            activity_quantity=activity_quantity,
        ),
        allow_q4_cached_stiffness=matrix_type == "stiffness",
    )


def assemble_stiffness_matrix(model: "FEModel") -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble the global stiffness matrix K only."""
    return _assemble_element_matrix(
        model,
        "stiffness",
        lambda element, mesh, material: element.compute_stiffness_matrix(mesh, material),
    )


def assemble_mass_matrix(model: "FEModel") -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble the global mass matrix M only, including any added point masses."""
    def assemble(lease: Any) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
        matrix, info = _assemble_element_matrix_under_lease(
            model,
            "mass",
            lambda element, mesh, material: element.compute_mass_matrix(
                mesh, material
            ),
            lease,
        )
        matrix = _add_point_masses_to_matrix(model, matrix)
        info["diagnostics"]["point_mass_count"] = int(
            len(getattr(model.mesh, "point_masses", {}) or {})
        )
        return matrix, info

    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="mass assembly",
        operation=assemble,
    )


def _add_point_masses_to_matrix(model: "FEModel", matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Add lumped point masses to the translational-DOF diagonal of ``matrix``."""
    from .concentrated_mass import add_to_matrix
    matrix = add_to_matrix(model.mesh, matrix)
    point_masses = getattr(model.mesh, "point_masses", None)
    if not point_masses:
        return matrix
    total_dofs = model.mesh.dof_manager.total_dofs
    diagonal = np.zeros(total_dofs, dtype=float)
    for node_id, mass in point_masses.items():
        node = model.mesh.get_node(int(node_id))
        if node is None or float(mass) == 0.0:
            continue
        for axis in range(3):
            diagonal[node.dofs[axis]] += float(mass)
    if not diagonal.any():
        return matrix
    return (matrix + sparse.diags(diagonal, 0, shape=(total_dofs, total_dofs), format="csr")).tocsr()


def _get_element_state(
    element_states: Optional[Any],
    element_id: int,
    element: Any,
    *,
    _post_observation: Optional[Callable[[str], None]] = None,
) -> Any:
    def observed(label: str) -> None:
        if _post_observation is not None:
            _post_observation(label)

    if element_states is None:
        return None
    if callable(element_states):
        try:
            state = element_states(element_id, element)
        except TypeError:
            observed("provider signature fallback")
            state = element_states(element_id)
        observed("provider return")
        return state
    if isinstance(element_states, Mapping):
        has_numeric_id = element_id in element_states
        observed("mapping numeric-ID lookup")
        if has_numeric_id:
            state = element_states[element_id]
            observed("mapping numeric-ID value")
            return state
        element_id_text = str(element_id)
        has_text_id = element_id_text in element_states
        observed("mapping text-ID lookup")
        if has_text_id:
            state = element_states[element_id_text]
            observed("mapping text-ID value")
            return state
    return None


def _guarded_geometric_state_snapshot(
    model: "FEModel",
    state: Any,
    *,
    element_id: int,
    _exact_guard: Any,
    path: str = "state",
) -> Any:
    """Detach qualified prestress data before an element helper consumes it."""

    context = f"geometric state observation for element {element_id} at {path}"
    if isinstance(state, np.ndarray):
        observed = np.asarray(state)
        _exact_guard(model, context=context)
        return np.frombuffer(
            np.ascontiguousarray(observed).tobytes(order="C"),
            dtype=observed.dtype,
        ).reshape(observed.shape)
    if isinstance(state, np.generic):
        observed = state.item()
        _exact_guard(model, context=context)
        return observed
    if state is None or type(state) in {str, bool, int, float}:
        return state
    if isinstance(state, Mapping):
        observed_items = tuple(state.items())
        _exact_guard(model, context=context)
        result: Dict[str, Any] = {}
        for key, member in observed_items:
            if type(key) is not str:
                raise AssemblyError(
                    f"qualified geometric state for element {element_id} "
                    f"contains a non-string key at {path}"
                )
            if key in result:
                raise AssemblyError(
                    f"qualified geometric state for element {element_id} "
                    f"contains duplicate key {key!r} at {path}"
                )
            result[key] = _guarded_geometric_state_snapshot(
                model,
                member,
                element_id=element_id,
                _exact_guard=_exact_guard,
                path=f"{path}.{key}",
            )
        return result
    if isinstance(state, Sequence) and not isinstance(
        state, (str, bytes, bytearray)
    ):
        observed_members = tuple(state)
        _exact_guard(model, context=context)
        return [
            _guarded_geometric_state_snapshot(
                model,
                member,
                element_id=element_id,
                _exact_guard=_exact_guard,
                path=f"{path}[{index}]",
            )
            for index, member in enumerate(observed_members)
        ]
    raise AssemblyError(
        f"qualified geometric state for element {element_id} has unsupported "
        f"type {type(state).__name__} at {path}"
    )


@dataclass(frozen=True)
class _OwnedGeometricStates:
    """Solver-private immutable envelope for detached prestress states."""

    values: Mapping[int, Any]
    element_ids: tuple[int, ...]


def _pack_owned_geometric_states(
    states: Mapping[int, Any],
) -> _OwnedGeometricStates:
    """Bind an already-detached canonical state map for one spectral solve."""

    copied = {int(element_id): state for element_id, state in states.items()}
    return _OwnedGeometricStates(
        values=MappingProxyType(copied),
        element_ids=tuple(sorted(copied)),
    )


def _assemble_geometric_stiffness_matrix_under_lease(
    model: "FEModel",
    element_states: Optional[Any] = None,
    *,
    qualified_runtime_guard: Any,
    _assembly_error_type: type[BaseException] = AssemblyError,
    _beam_dynamic_lookup_authority: tuple[tuple[Any, str, Any], ...] = (
        _EXACT_BEAM_DYNAMIC_LOOKUP_AUTHORITY
    ),
    _beam_element_type: type[Any] = _BeamElement,
    _beam_geometric_stiffness: Any = _EXACT_BEAM_GEOMETRIC_STIFFNESS,
    _dict_contains: Any = dict.__contains__,
    _dict_get: Any = dict.get,
    _dict_getitem: Any = dict.__getitem__,
    _dict_items: Any = dict.items,
    _dict_type: type[Any] = dict,
    _exact_any: Any = any,
    _exact_callable: Any = callable,
    _exact_getattr: Any = getattr,
    _exact_isinstance: Any = isinstance,
    _exact_iter: Any = iter,
    _exact_len: Any = len,
    _exact_next: Any = next,
    _exact_object_getattribute: Any = object.__getattribute__,
    _exact_type: Any = type,
    _exact_type_getattribute: Any = type.__getattribute__,
    _list_type: type[Any] = list,
    _integer_type: type[Any] = int,
    _mapping_type: Any = Mapping,
    _numpy_array_type: type[Any] = np.ndarray,
    _numpy_asarray: Any = np.asarray,
    _numpy_ascontiguousarray: Any = np.ascontiguousarray,
    _numpy_frombuffer: Any = np.frombuffer,
    _numpy_generic_type: type[Any] = np.generic,
    _primitive_types: frozenset[type[Any]] = frozenset(
        {str, bool, int, float}
    ),
    _beam_scalar_types: frozenset[type[Any]] = frozenset({int, float}),
    _sequence_exclusions: tuple[type[Any], ...] = (
        str,
        bytes,
        bytearray,
    ),
    _sequence_type: Any = Sequence,
    _stop_iteration: type[BaseException] = StopIteration,
    _string_type: type[Any] = str,
    _tuple_type: type[Any] = tuple,
    _type_error: type[BaseException] = TypeError,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble the global geometric stiffness matrix KG only.

    ``element_states`` supplies the reference stress/resultant state for each
    element.  Beams accept a numeric value or a mapping with
    ``axial_compression`` positive in compression.  Shell resultants act
    through the Mindlin field ``[u+z*ry, v-z*rx, w]``; drilling rotation and
    stress components normal to the midsurface are outside this operator.
    """
    from .current_state_tangent import (
        require_exact_qualified_component_lifecycle_api,
    )

    lifecycle_guard = require_exact_qualified_component_lifecycle_api

    def exact_qualified_guard(
        expected_model: "FEModel",
        *,
        context: str,
    ) -> None:
        lifecycle_guard(expected_model, context=context)
        qualified_runtime_guard(expected_model, context=context)

    runtime_namespace = _exact_object_getattribute(
        qualified_runtime_guard,
        "__dict__",
    )
    trusted_input_guard = (
        _dict_get(
            runtime_namespace,
            "_qualified_trusted_input_require",
        )
        if _exact_type(runtime_namespace) is _dict_type
        else None
    )

    def internal_input_guard(*, context: str) -> None:
        if trusted_input_guard is None:
            exact_qualified_guard(model, context=context)
            return
        trusted_input_guard(model, context=context)

    def observed_element_state(
        source: Any,
        element_id: int,
        element: Any,
    ) -> Any:
        """Read one state; only exact dict operations use the narrow lease."""

        if source is None:
            return None
        if type(source) is _OwnedGeometricStates:
            state = source.values.get(element_id)
            internal_input_guard(
                context=(
                    "geometric stiffness packed-state lookup for element "
                    f"{element_id}"
                )
            )
            return state
        if _exact_callable(source):
            try:
                try:
                    state = source(element_id, element)
                except _type_error:
                    exact_qualified_guard(
                        model,
                        context=(
                            "geometric stiffness assembly state provider "
                            f"signature fallback for element {element_id}"
                        ),
                    )
                    state = source(element_id)
            finally:
                exact_qualified_guard(
                    model,
                    context=(
                        "geometric stiffness assembly state provider return "
                        f"for element {element_id}"
                    ),
                )
            return state

        exact_mapping = _exact_type(source) is _dict_type

        def post_lookup(label: str) -> None:
            context = (
                "geometric stiffness assembly state "
                f"{label} for element {element_id}"
            )
            if exact_mapping:
                internal_input_guard(context=context)
            else:
                exact_qualified_guard(model, context=context)

        if exact_mapping:
            has_numeric_id = _dict_contains(source, element_id)
            post_lookup("mapping numeric-ID lookup")
            if has_numeric_id:
                state = _dict_getitem(source, element_id)
                post_lookup("mapping numeric-ID value")
                return state
            element_id_text = _string_type(element_id)
            has_text_id = _dict_contains(source, element_id_text)
            post_lookup("mapping text-ID lookup")
            if has_text_id:
                state = _dict_getitem(source, element_id_text)
                post_lookup("mapping text-ID value")
                return state
            return None
        if _exact_isinstance(source, _mapping_type):
            try:
                has_numeric_id = element_id in source
            finally:
                post_lookup("mapping numeric-ID lookup")
            if has_numeric_id:
                try:
                    state = source[element_id]
                finally:
                    post_lookup("mapping numeric-ID value")
                return state
            element_id_text = _string_type(element_id)
            try:
                has_text_id = element_id_text in source
            finally:
                post_lookup("mapping text-ID lookup")
            if has_text_id:
                try:
                    state = source[element_id_text]
                finally:
                    post_lookup("mapping text-ID value")
                return state
        return None

    def snapshot_geometric_state(
        state: Any,
        *,
        element_id: int,
        path: str = "state",
    ) -> Any:
        """Detach state with private callback-aware traversal authority."""

        if type(element_states) is _OwnedGeometricStates:
            # The solver detached this state from caller ownership before
            # constructing the private immutable envelope.  Re-copying every
            # nested station value here was the dominant remaining prestress
            # assembly overhead.
            internal_input_guard(
                context=(
                    "geometric packed-state use for element "
                    f"{element_id}"
                )
            )
            return state

        context = (
            f"geometric state observation for element {element_id} at {path}"
        )
        state_type = _exact_type(state)
        if state is None or state_type in _primitive_types:
            return state
        if state_type is _dict_type:
            observed_iterator = _exact_iter(_dict_items(state))
            internal_input_guard(context=context)
            result = {}
            while True:
                try:
                    observed_item = _exact_next(observed_iterator)
                except _stop_iteration:
                    internal_input_guard(context=context)
                    break
                internal_input_guard(context=context)
                key, member = observed_item
                internal_input_guard(context=context)
                if _exact_type(key) is not _string_type:
                    raise _assembly_error_type(
                        f"qualified geometric state for element {element_id} "
                        f"contains a non-string key at {path}"
                    )
                if key in result:
                    raise _assembly_error_type(
                        f"qualified geometric state for element {element_id} "
                        f"contains duplicate key {key!r} at {path}"
                    )
                result[key] = snapshot_geometric_state(
                    member,
                    element_id=element_id,
                    path=f"{path}.{key}",
                )
            return result
        if state_type in {_list_type, _tuple_type}:
            observed_iterator = _exact_iter(state)
            internal_input_guard(context=context)
            result = []
            index = 0
            while True:
                try:
                    member = _exact_next(observed_iterator)
                except _stop_iteration:
                    internal_input_guard(context=context)
                    break
                internal_input_guard(context=context)
                result.append(
                    snapshot_geometric_state(
                        member,
                        element_id=element_id,
                        path=f"{path}[{index}]",
                    )
                )
                index += 1
            return result
        if _exact_isinstance(state, _numpy_array_type):
            observed = _numpy_asarray(state)
            exact_qualified_guard(model, context=context)
            observed_bytes = _numpy_ascontiguousarray(observed).tobytes(
                order="C"
            )
            exact_qualified_guard(model, context=context)
            copied = _numpy_frombuffer(
                observed_bytes,
                dtype=observed.dtype,
            ).reshape(observed.shape)
            exact_qualified_guard(model, context=context)
            return copied
        if _exact_isinstance(state, _numpy_generic_type):
            observed = state.item()
            exact_qualified_guard(model, context=context)
            return observed
        if _exact_isinstance(state, _mapping_type):
            observed_items = state.items()
            exact_qualified_guard(model, context=context)
            observed_iterator = _exact_iter(observed_items)
            exact_qualified_guard(model, context=context)
            result = {}
            while True:
                try:
                    observed_item = _exact_next(observed_iterator)
                except _stop_iteration:
                    exact_qualified_guard(model, context=context)
                    break
                exact_qualified_guard(model, context=context)
                key, member = observed_item
                exact_qualified_guard(model, context=context)
                if _exact_type(key) is not _string_type:
                    raise _assembly_error_type(
                        f"qualified geometric state for element {element_id} "
                        f"contains a non-string key at {path}"
                    )
                if key in result:
                    raise _assembly_error_type(
                        f"qualified geometric state for element {element_id} "
                        f"contains duplicate key {key!r} at {path}"
                    )
                result[key] = snapshot_geometric_state(
                    member,
                    element_id=element_id,
                    path=f"{path}.{key}",
                )
            return result
        if _exact_isinstance(state, _sequence_type) and not _exact_isinstance(
            state,
            _sequence_exclusions,
        ):
            observed_iterator = _exact_iter(state)
            exact_qualified_guard(model, context=context)
            result = []
            index = 0
            while True:
                try:
                    member = _exact_next(observed_iterator)
                except _stop_iteration:
                    exact_qualified_guard(model, context=context)
                    break
                exact_qualified_guard(model, context=context)
                result.append(
                    snapshot_geometric_state(
                        member,
                        element_id=element_id,
                        path=f"{path}[{index}]",
                    )
                )
                index += 1
            return result
        raise _assembly_error_type(
            f"qualified geometric state for element {element_id} has "
            f"unsupported type {state_type.__name__} at {path}"
        )

    def has_exact_builtin_beam_profile(element: Any) -> bool:
        """Prove every BeamElement lookup used by the direct fast path."""

        if _exact_type(element) is not _beam_element_type:
            return False
        element_namespace = _exact_object_getattribute(element, "__dict__")
        if _exact_type(element_namespace) is not _dict_type:
            return False
        for name in (
            "compute_geometric_stiffness_matrix",
            "_axial_compression_from_state",
            "get_node_coordinates",
            "_beam_frame_and_transform",
            "_geometric_polar_radius_squared",
            "total_dofs",
            "num_nodes",
            "dofs_per_node",
        ):
            if _dict_contains(element_namespace, name):
                return False
        for owner, name, descriptor in _beam_dynamic_lookup_authority:
            if (
                _exact_type_getattribute(owner, "__dict__").get(name)
                is not descriptor
            ):
                return False
        if (
            _exact_type_getattribute(
                _beam_element_type,
                "__dict__",
            ).get("compute_geometric_stiffness_matrix")
            is not _beam_geometric_stiffness
            or _exact_type(_dict_get(element_namespace, "element_id"))
            is not _integer_type
            or _exact_type(_dict_get(element_namespace, "material_name"))
            is not _string_type
            or _exact_type(_dict_get(element_namespace, "node_ids"))
            is not _list_type
            or _exact_len(_dict_get(element_namespace, "node_ids")) != 2
            or _exact_any(
                _exact_type(node_id) is not _integer_type
                for node_id in _dict_get(element_namespace, "node_ids")
            )
            or _exact_type(_dict_get(element_namespace, "cross_section"))
            is not _dict_type
            or _dict_get(element_namespace, "generalized_section") is not None
            or _exact_any(
                _exact_type(_dict_get(element_namespace, name))
                not in _beam_scalar_types
                for name in ("_A", "_Iy", "_Iz")
            )
        ):
            return False
        orientation = _dict_get(element_namespace, "_orientation")
        return orientation is None or _exact_type(orientation) is _numpy_array_type

    exact_qualified_guard(
        model,
        context="geometric stiffness assembly exact preflight",
    )
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    info = _base_info(model, "geometric_stiffness")
    start_time = time.time()
    _activity, activity_scales, activity_info = _activity_scales(
        model, "stiffness"
    )
    qualified_runtime_guard(
        model,
        context="geometric stiffness assembly activity",
    )
    if activity_info is not None:
        info["diagnostics"]["element_activity"] = activity_info

    # Retrieve or build cached sparsity pattern
    rows_concat, cols_concat = _get_cached_sparsity_pattern(mesh, "geometric_stiffness")

    # S4 geometric stiffness is especially expensive in the scalar element
    # loop because every element repeatedly reconstructs the same reference
    # derivatives and coordinate transforms.  Keep the stress/resultant
    # sampling in the element contract, but evaluate the common matrix
    # operator in one compiled batch and cache its immutable geometry.
    from .elements import ShellElement
    from .e4_pl_element import (
        FORMULATION_ID as QUALIFIED_Q4_FORMULATION_ID,
        QualifiedE4PLShellElement,
    )
    from .jit_compiler import JIT_ENABLED, JIT_DISABLED_REASON, jit_diagnostics
    from .vectorized_stiffness import (
        compute_s4_geometric_stiffness_matrices_jit,
        prepare_s4_geometric_kinematics_jit,
    )

    eligible_groups: Dict[Tuple[bytes, bytes], list[Tuple[int, Any]]] = {}
    for elem_id, element in mesh.elements.items():
        if isinstance(element, ShellElement) and bool(getattr(element, "_is_4node", False)):
            formulation_id = str(getattr(element, "formulation_id", ""))
            if (
                type(element) is QualifiedE4PLShellElement
                or formulation_id == QUALIFIED_Q4_FORMULATION_ID
            ):
                if (
                    type(element) is not QualifiedE4PLShellElement
                    or formulation_id != QUALIFIED_Q4_FORMULATION_ID
                ):
                    raise AssemblyError(
                        f"Element {elem_id} has incompatible qualified Q4 authority"
                    )
                try:
                    _EXACT_Q4_QUADRATURE_GUARD(element)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise AssemblyError(
                        f"Element {elem_id} has incompatible qualified Q4 quadrature authority"
                    ) from exc
            points = np.ascontiguousarray(element.gauss_points, dtype=float)
            weights = np.ascontiguousarray(element.gauss_weights, dtype=float)
            key = (points.tobytes(), weights.tobytes())
            eligible_groups.setdefault(key, []).append((int(elem_id), element))

    precomputed: Dict[int, np.ndarray] = {}
    geometry_cache = getattr(mesh, "_s4_geometric_kinematics_cache", None)
    if geometry_cache is None:
        geometry_cache = {}
        mesh._s4_geometric_kinematics_cache = geometry_cache
    revisions = getattr(mesh, "revision_signature", lambda: {})()
    direct_token = getattr(mesh, "_qualified_direct_state_token", None)
    direct_revision = (
        int(direct_token[0])
        if isinstance(direct_token, list) and len(direct_token) == 1
        else -1
    )
    geometry_revision = (
        int(revisions.get("topology", 0)),
        int(revisions.get("geometry", 0)),
        direct_revision,
    )
    stale_geometry_keys = [
        key for key in geometry_cache if not key or key[0] != geometry_revision
    ]
    for stale_key in stale_geometry_keys:
        del geometry_cache[stale_key]
    geometry_setup_seconds = 0.0
    kernel_seconds = 0.0
    cache_hits = 0
    batched_ids: list[int] = []
    for group_index, elem_list in enumerate(eligible_groups.values()):
        first = elem_list[0][1]
        points = np.ascontiguousarray(first.gauss_points, dtype=float)
        weights = np.ascontiguousarray(first.gauss_weights, dtype=float)
        element_ids = tuple(elem_id for elem_id, _element in elem_list)
        coords = np.ascontiguousarray(
            [element.get_node_coordinates(mesh) for _elem_id, element in elem_list],
            dtype=float,
        )
        cache_key = (
            geometry_revision,
            element_ids,
            coords.tobytes(order="C"),
            points.tobytes(order="C"),
            weights.tobytes(order="C"),
        )
        geometry = geometry_cache.get(cache_key)
        if geometry is None:
            geometry_start = time.perf_counter()
            geometry = prepare_s4_geometric_kinematics_jit(coords, points, weights)
            geometry_setup_seconds += time.perf_counter() - geometry_start
            geometry_cache[cache_key] = geometry
        else:
            cache_hits += 1

        count = len(elem_list)
        gp_count = points.shape[0]
        membrane = np.zeros((count, gp_count, 3), dtype=float)
        bending = np.zeros_like(membrane)
        second_moment = np.zeros_like(membrane)
        for index, (elem_id, element) in enumerate(elem_list):
            state = observed_element_state(
                element_states,
                elem_id,
                element,
            )
            state = snapshot_geometric_state(
                state,
                element_id=elem_id,
            )
            membrane[index] = element._membrane_compression_samples(state, gp_count)
            bending[index] = element._bending_compression_samples(state, gp_count)
            second_moment[index] = element._stress_second_moment_samples(
                state,
                gp_count,
                membrane[index],
                element.thickness,
            )
        kernel_start = time.perf_counter()
        matrices = compute_s4_geometric_stiffness_matrices_jit(
            *geometry,
            membrane,
            bending,
            second_moment,
        )
        kernel_seconds += time.perf_counter() - kernel_start
        for index, (elem_id, _element) in enumerate(elem_list):
            precomputed[elem_id] = matrices[index]
            batched_ids.append(elem_id)

    info["diagnostics"]["vectorized_s4_geometric_stiffness"] = {
        "element_count": len(batched_ids),
        "group_count": len(eligible_groups),
        "element_ids": sorted(batched_ids),
        "geometry_cache_hits": cache_hits,
        "geometry_setup_seconds": geometry_setup_seconds,
        "kernel_seconds": kernel_seconds,
        "jit_enabled": bool(JIT_ENABLED),
        "jit_disabled_reason": JIT_DISABLED_REASON,
        "jit": jit_diagnostics(),
    }
    qualified_runtime_guard(
        model,
        context="geometric stiffness assembly prepared operators",
    )

    data_list = []
    for elem_id, element in mesh.elements.items():
        elem_start = time.time()
        material = model.get_material(element.material_name)
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            info["skipped_elements"].append(int(elem_id))
            continue

        if int(elem_id) in precomputed:
            element_matrix = precomputed[int(elem_id)]
        else:
            state = observed_element_state(
                element_states,
                int(elem_id),
                element,
            )
            if type(element) in {
                _QualifiedE4PLShellElement,
                _QualifiedE4PLS3ShellElement,
            }:
                state = snapshot_geometric_state(
                    state,
                    element_id=int(elem_id),
                )
            exact_builtin_beam = has_exact_builtin_beam_profile(element)
            if exact_builtin_beam:
                # The exact built-in beam implementation is a fixed operator,
                # not an arbitrary callback.  Keep its per-element boundary
                # checks constant-time so mixed shell/beam assembly remains
                # O(N), and detach exact built-in state containers before the
                # direct authority-bound call.
                state = snapshot_geometric_state(
                    state,
                    element_id=int(elem_id),
                )
                internal_input_guard(
                    context=(
                        "geometric stiffness exact built-in beam preflight "
                        f"for element {int(elem_id)}"
                    ),
                )
                element_matrix = _beam_geometric_stiffness(
                    element,
                    mesh,
                    material,
                    state,
                )
                internal_input_guard(
                    context=(
                        "geometric stiffness exact built-in beam return "
                        f"for element {int(elem_id)}"
                    ),
                )
            else:
                # Attribute lookup and invocation can both execute arbitrary
                # user code for a generic/custom element.  Exact-dict input
                # traversal authority must never cross either callback
                # boundary, even when the state source itself is an exact
                # built-in dictionary.
                exact_qualified_guard(
                    model,
                    context=(
                        "geometric stiffness custom getter lookup preflight "
                        f"for element {int(elem_id)}"
                    ),
                )
                try:
                    getter = _exact_getattr(
                        element,
                        "compute_geometric_stiffness_matrix",
                        None,
                    )
                finally:
                    exact_qualified_guard(
                        model,
                        context=(
                            "geometric stiffness custom getter lookup return "
                            f"for element {int(elem_id)}"
                        ),
                    )
                if getter is None:
                    element_matrix = np.zeros(
                        (dof_mapping.size, dof_mapping.size),
                        dtype=float,
                    )
                else:
                    exact_qualified_guard(
                        model,
                        context=(
                            "geometric stiffness custom getter call preflight "
                            f"for element {int(elem_id)}"
                        ),
                    )
                    try:
                        element_matrix = getter(mesh, material, state)
                    finally:
                        exact_qualified_guard(
                            model,
                            context=(
                                "geometric stiffness custom getter call return "
                                f"for element {int(elem_id)}"
                            ),
                        )
        element_matrix = _check_element_matrix_shape(
            int(elem_id),
            "geometric_stiffness",
            element_matrix,
            int(dof_mapping.size),
        )
        scale = activity_scales.get(int(elem_id), 1.0)
        data_list.append(
            (scale * np.asarray(element_matrix, dtype=float)).ravel()
        )

        info["element_times"][int(elem_id)] = time.time() - elem_start
        info["num_elements"] += 1

    qualified_runtime_guard(
        model,
        context="geometric stiffness assembly completed elements",
    )
    exact_qualified_guard(
        model,
        context="geometric stiffness assembly exact output",
    )

    info["state_source"] = "none" if element_states is None else type(element_states).__name__
    info["diagnostics"]["scalar_element_count"] = int(info["num_elements"] - len(batched_ids))
    info["diagnostics"]["packed_prestress_state"] = bool(
        type(element_states) is _OwnedGeometricStates
    )
    info["diagnostics"]["shell_initial_stress_scope"] = (
        "mindlin_translations_and_director_gradients; no_drilling_or_transverse_normal_stress_terms"
    )

    if not data_list:
        matrix = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
        info["diagnostics"]["assembled_symmetry_error"] = 0.0
        info["sparsity_signature"] = _topology_signature(mesh, "geometric_stiffness")
        info["assembly_time"] = time.time() - start_time
        return matrix, info

    data_concat = np.concatenate(data_list)
    coo = sparse.coo_matrix(
        (data_concat, (rows_concat, cols_concat)),
        shape=(total_dofs, total_dofs),
        dtype=float,
    )
    matrix = coo.tocsr()
    if activity_info is not None and activity_info["zero_contribution_count"]:
        matrix.eliminate_zeros()
    info["diagnostics"]["assembled_symmetry_error"] = _relative_symmetry_error(matrix)
    info["sparsity_signature"] = _topology_signature(mesh, "geometric_stiffness")
    info["assembly_time"] = time.time() - start_time
    return matrix, info


def assemble_geometric_stiffness_matrix(
    model: "FEModel",
    element_states: Optional[Any] = None,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble global geometric stiffness under one authority lease."""

    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="geometric stiffness assembly",
        operation=lambda lease: _assemble_geometric_stiffness_matrix_under_lease(
            model,
            element_states,
            qualified_runtime_guard=lease,
        ),
    )


def _qualified_s3_pressure_surface_records(
    model: "FEModel",
    load_case: Optional["LoadCase"],
    *,
    qualified_runtime_guard: Any,
    foreign_code_free_preamble: bool = False,
) -> list[Dict[str, Any]]:
    """Identify the exact surface carrying qualified-S3 pressure work.

    ``foreign_code_free_preamble`` is the caller's attestation that nothing
    outside owned code ran since the last guard (an untouched ``LoadCase``, no
    activity provider); the opening observation guard is then the lease's
    constant-time trusted guard rather than a complete scan.  A lease without
    that capability keeps the complete guard.

    The S3 section origin may be offset from its nodal interpolation surface.
    Pressure is intentionally conjugate to the latter; reporting that choice
    prevents force/reaction post-processing from silently treating the material
    midsurface as the pressure surface.  Other shell formulations retain their
    existing diagnostics unchanged.
    """

    if load_case is None:
        return []
    from .current_state_tangent import (
        require_exact_qualified_component_lifecycle_api,
    )

    lifecycle_guard = require_exact_qualified_component_lifecycle_api

    def exact_qualified_guard(*, context: str) -> None:
        lifecycle_guard(model, context=context)
        qualified_runtime_guard(model, context=context)

    runtime_namespace = object.__getattribute__(
        qualified_runtime_guard,
        "__dict__",
    )
    trusted_input_guard = (
        dict.get(runtime_namespace, "_qualified_trusted_input_require")
        if type(runtime_namespace) is dict
        else None
    )

    def internal_input_guard(*, context: str) -> None:
        # Capture and finalization retain the complete closed-world guard.  An
        # exact qualified model also exposes a non-renewable constant-time
        # token/epoch guard for callback-free input traversal.  Re-running the
        # complete model-wide lifecycle scan for every S3 pressure record made
        # this diagnostic path quadratic in mesh size without adding authority.
        if trusted_input_guard is None:
            exact_qualified_guard(context=context)
            return
        trusted_input_guard(model, context=context)

    pressure_ids = tuple(getattr(load_case, "pressure_loads", {}))
    if foreign_code_free_preamble and trusted_input_guard is not None:
        trusted_input_guard(
            model,
            context="qualified S3 pressure-surface mapping observation",
        )
    else:
        exact_qualified_guard(
            context="qualified S3 pressure-surface mapping observation",
        )
    records: list[Dict[str, Any]] = []
    for raw_element_id in pressure_ids:
        element_id = int(raw_element_id)
        element = model.mesh.get_element(element_id)
        if element is None:
            continue
        formulation = str(getattr(element, "formulation_id", ""))
        pressure_policy = str(
            getattr(element, "pressure_surface_policy_id", "")
        )
        if (
            formulation != "E4_PL_QUALIFIED_S3_COMPANION_V1"
            and pressure_policy != "ELEMENT_NODAL_REFERENCE_SURFACE_V1"
        ):
            continue
        offset = float(getattr(element, "reference_surface_offset", 0.0))
        internal_input_guard(
            context=(
                "qualified S3 pressure-surface offset observation for "
                f"element {element_id}"
            ),
        )
        if not np.isfinite(offset):
            raise AssemblyError(
                f"Qualified S3 element {element_id} has a non-finite reference-surface offset."
            )
        records.append(
            {
                "element_id": element_id,
                "pressure_surface_id": "ELEMENT_NODAL_REFERENCE_SURFACE_V1",
                "reference_surface_offset": offset,
                "resultant_and_reaction_reference": (
                    "GLOBAL_NODAL_REFERENCE_COORDINATES"
                ),
                "section_origin_offset_from_reference": -offset,
                "virtual_work": "TRANSLATIONAL_NODAL_REFERENCE_SURFACE_ONLY",
            }
        )
    return records


def _qualified_trusted_state_guard(qualified_runtime_guard: Any) -> Any:
    """Return the lease's constant-time trusted-state guard, or ``None``.

    The guard exists only for an exact qualified model.  It checks the owned
    inputs, the mesh mutation token and the Q4/S3/assembly epochs, and says
    nothing about other element families or about class facts outside those
    epochs.  A caller may therefore substitute it for a complete guard only
    across a span in which no code outside this package's owned kernels ran;
    where it is absent the complete guard stays in force.
    """

    try:
        namespace = object.__getattribute__(qualified_runtime_guard, "__dict__")
    except AttributeError:
        return None
    if type(namespace) is not dict:
        return None
    return dict.get(namespace, "_qualified_trusted_state_require")


def _assemble_load_vector_under_lease(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    displacements: Optional[np.ndarray] = None,
    *,
    qualified_runtime_guard: Any,
    preflight_verified: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Assemble the global external load vector ``F_external``.

    ``displacements`` is ignored by ordinary dead loads.  A load case with
    ``follower_pressure=True`` uses it to evaluate pressure on the current
    shell nodal interpolation surface.

    Every point where code outside this package's owned kernels may have run
    (a ``LoadCase`` subclass or replaced method, a generic element, an
    activity provider, an array-like) keeps the complete lifecycle-plus-lease
    guard.  A point reached with provably no such code since the previous
    complete guard uses the lease's constant-time trusted guard instead, so
    the cost of an unchanged model is paid once per lease and not once per
    observation.  ``preflight_verified`` is the caller's attestation that a
    complete guard has passed since the last foreign code ran.
    """
    from .boundary import (
        _exact_load_case_instance_namespace,
        _load_model_access_is_builtin,
        _plain_numeric_array,
        _numpy_error_routes_are_builtin,
        _run_foreign_code_free_load_vector,
    )
    from .current_state_tangent import (
        require_exact_qualified_component_lifecycle_api,
    )

    lifecycle_guard = require_exact_qualified_component_lifecycle_api
    trusted_state_guard = _qualified_trusted_state_guard(qualified_runtime_guard)
    routing_foreign_code_free = (_load_model_access_is_builtin(model)
                                 and _numpy_error_routes_are_builtin())

    def exact_qualified_guard(
        expected_model: "FEModel",
        *,
        context: str,
    ) -> None:
        lifecycle_guard(expected_model, context=context)
        qualified_runtime_guard(expected_model, context=context)

    def observation_guard(*, context: str, foreign_code_free: bool) -> None:
        nonlocal routing_foreign_code_free
        # Once a callback could have run, a class/instance fact may have
        # changed outside the mesh epochs. Never revive a prior routing proof.
        if not foreign_code_free:
            routing_foreign_code_free = False
        if routing_foreign_code_free and foreign_code_free and trusted_state_guard is not None:
            trusted_state_guard(model, context=context)
        else:
            exact_qualified_guard(model, context=context)

    if not preflight_verified:
        exact_qualified_guard(model, context="load-vector assembly preflight")
    total_dofs = model.mesh.dof_manager.total_dofs
    if not routing_foreign_code_free:
        exact_qualified_guard(model, context="load-vector DOF observation")
    start_time = time.time()
    case_state = (
        None
        if load_case is None
        else _exact_load_case_instance_namespace(load_case)
    )
    if load_case is None:
        load_name = None
        follower_pressure = False
    elif case_state is not None:
        # An untouched LoadCase: both reads are plain dictionary reads.
        load_name = case_state["name"]
        observation_guard(
            context="load-vector LoadCase name observation",
            foreign_code_free=True,
        )
        follower_pressure = case_state["follower_pressure"]
        observation_guard(
            context="load-vector LoadCase pressure-policy observation",
            foreign_code_free=True,
        )
    else:
        routing_foreign_code_free = False
        load_name = load_case.name
        exact_qualified_guard(
            model,
            context="load-vector LoadCase name observation",
        )
        follower_pressure = bool(
            getattr(load_case, "follower_pressure", False)
        )
        exact_qualified_guard(
            model,
            context="load-vector LoadCase pressure-policy observation",
        )
    if displacements is not None:
        displacements_exact = _plain_numeric_array(displacements)
        displacements = np.asarray(displacements, dtype=float)
        observation_guard(
            context="load-vector displacement observation",
            foreign_code_free=displacements_exact,
        )
        displacements = displacements.reshape(-1)
        if displacements.shape != (total_dofs,):
            raise AssemblyError(
                f"Displacement vector shape {displacements.shape} does not match total DOFs {(total_dofs,)}."
            )
        if not np.all(np.isfinite(displacements)):
            raise AssemblyError("Displacement vector contains non-finite values.")
    owned_kernel = False
    if load_case is None:
        load_vector = np.zeros(total_dofs, dtype=float)
    else:
        load_vector = None
        if routing_foreign_code_free and case_state is not None and trusted_state_guard is not None:

            def owned_stage_guard(*, stage: str) -> None:
                trusted_state_guard(model, context=stage)

            load_vector = _run_foreign_code_free_load_vector(
                load_case,
                model.mesh,
                model.mesh.dof_manager,
                model.get_material,
                displacements,
                _element_activity(model),
                stage_guard=owned_stage_guard,
            )
            owned_kernel = load_vector is not None
        if load_vector is None:
            routing_foreign_code_free = False
            load_vector = load_case.get_load_vector(
                model.mesh,
                model.mesh.dof_manager,
                model.get_material,
                displacements=displacements,
                element_activity=_element_activity(model),
            )
        observation_guard(
            context="load-vector LoadCase observation",
            foreign_code_free=owned_kernel,
        )
        array_exact = _plain_numeric_array(load_vector)
        load_vector = np.asarray(load_vector, dtype=float)
        observation_guard(
            context="load-vector array observation",
            foreign_code_free=array_exact,
        )
        load_vector = load_vector.reshape(-1)

    if load_vector.shape != (total_dofs,):
        raise AssemblyError(f"Load vector shape {load_vector.shape} does not match total DOFs {(total_dofs,)}.")
    if not np.all(np.isfinite(load_vector)):
        raise AssemblyError(f"Load case {load_name!r} produced non-finite load vector values.")

    activity = _element_activity(model)
    info = {
        "vector_type": "load",
        "load_case": load_name,
        "num_nodes": model.mesh.num_nodes,
        "total_dofs": total_dofs,
        "assembly_time": time.time() - start_time,
        "load_norm": float(np.linalg.norm(load_vector)),
        "pressure_configuration": (
            "current"
            if follower_pressure
            else "reference"
        ),
        "element_activity": (
            None
            if activity is None
            else {
                "quantity": "load",
                "sequence": int(getattr(activity, "sequence", 0)),
            }
        ),
    }
    # No foreign code has run since the previous guard when the load case is
    # untouched and no activity provider was consulted for the info record.
    preamble_foreign_code_free = (
        load_case is None or (case_state is not None and activity is None)
    )
    pressure_surfaces = _qualified_s3_pressure_surface_records(
        model,
        load_case,
        # Suppress all trusted metadata capabilities when ordinary routing
        # observations could have invoked foreign code.
        qualified_runtime_guard=(qualified_runtime_guard if routing_foreign_code_free else exact_qualified_guard),
        foreign_code_free_preamble=preamble_foreign_code_free,
    )
    if pressure_surfaces:
        info["qualified_s3_pressure_surfaces"] = pressure_surfaces
    # The pressure-surface walk touches only elements the owned kernel already
    # admitted as exact qualified shells; for any other route it may read
    # generic elements, so the complete guard closes the vector.
    observation_guard(
        context="load-vector assembly output",
        foreign_code_free=owned_kernel or load_case is None,
    )
    return load_vector, info


def assemble_load_vector(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    displacements: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Assemble the external load vector under one authority lease."""

    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="load-vector assembly",
        operation=lambda lease: _assemble_load_vector_under_lease(
            model,
            load_case,
            displacements,
            qualified_runtime_guard=lease,
        ),
    )


def _assemble_external_load_tangent_under_lease(
    model: "FEModel",
    load_case: Optional["LoadCase"],
    displacements: Optional[np.ndarray] = None,
    *,
    qualified_runtime_guard: Any,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble ``dF_external / du`` for current-area follower pressure.

    Dead loads return an exact zero matrix.  The follower tangent is generally
    nonsymmetric for an open pressure patch; callers must therefore use a
    general sparse factorization for ``K_internal - K_external``.
    """
    from .current_state_tangent import (
        require_exact_qualified_component_lifecycle_api,
    )

    lifecycle_guard = require_exact_qualified_component_lifecycle_api

    def exact_qualified_guard(
        expected_model: "FEModel",
        *,
        context: str,
    ) -> None:
        lifecycle_guard(expected_model, context=context)
        qualified_runtime_guard(expected_model, context=context)

    exact_qualified_guard(
        model,
        context="external-load tangent assembly preflight",
    )
    if load_case is not None and any(
        type(e).__module__ == "anysolver._ge_beam3_native_fibre_static_element"
        for e in model.mesh.elements.values()
    ):
        from ._ge_beam3_native_load_admission import assemble_nodal_forces

        assemble_nodal_forces(
            load_case, model.mesh, model.mesh.dof_manager,
            guard=lambda *, stage: exact_qualified_guard(model, context=stage),
            activity=_element_activity(model),
        )
    total_dofs = model.mesh.dof_manager.total_dofs
    start_time = time.time()
    if load_case is None:
        load_name = None
        follower_pressure = False
    else:
        load_name = load_case.name
        exact_qualified_guard(
            model,
            context="external-load tangent LoadCase name observation",
        )
        follower_pressure = bool(
            getattr(load_case, "follower_pressure", False)
        )
        exact_qualified_guard(
            model,
            context="external-load tangent pressure-policy observation",
        )
    if displacements is None:
        u = np.zeros(total_dofs, dtype=float)
    else:
        u = np.asarray(displacements, dtype=float)
        exact_qualified_guard(
            model,
            context="external-load tangent displacement observation",
        )
        u = u.reshape(-1)
    if u.shape != (total_dofs,):
        raise AssemblyError(f"Displacement vector shape {u.shape} does not match total DOFs {(total_dofs,)}.")
    if not np.all(np.isfinite(u)):
        raise AssemblyError("Displacement vector contains non-finite values.")

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    element_ids: list[int] = []
    _activity, activity_scales, activity_info = _activity_scales(model, "load")
    if load_case is not None and follower_pressure:
        pressure_items = tuple(
            getattr(load_case, "pressure_loads", {}).items()
        )
        exact_qualified_guard(
            model,
            context="external-load tangent pressure mapping observation",
        )
        for raw_element_id, pressure in pressure_items:
            element_id = int(raw_element_id)
            element = model.mesh.get_element(element_id)
            if element is None:
                continue
            if not hasattr(element, "node_ids"):
                raise AssemblyError(f"Follower pressure element {element_id} has no nodal interpolation.")
            load_case._reject_strict_flat_s3_v2_follower_pressure(element)
            dof_mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
            coords = load_case._current_element_coordinates(element, model.mesh, u)
            exact_qualified_guard(
                model,
                context=(
                    "external-load tangent current-coordinate observation for "
                    f"element {element_id}"
                ),
            )
            try:
                element_tangent = load_case._consistent_pressure_tangent(
                    element,
                    model.mesh,
                    pressure,
                    coords,
                )
            except ValueError as exc:
                raise AssemblyError(str(exc)) from exc
            exact_qualified_guard(
                model,
                context=(
                    "external-load tangent pressure-kernel observation for "
                    f"element {element_id}"
                ),
            )
            element_tangent = _check_element_matrix_shape(
                element_id,
                "external_load_tangent",
                element_tangent,
                int(dof_mapping.size),
            )
            element_tangent = (
                activity_scales.get(element_id, 1.0) * element_tangent
            )
            row_grid, col_grid = np.meshgrid(dof_mapping, dof_mapping, indexing="ij")
            rows.append(row_grid.ravel())
            cols.append(col_grid.ravel())
            data.append(element_tangent.ravel())
            element_ids.append(element_id)

    if data:
        tangent = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(total_dofs, total_dofs),
            dtype=float,
        ).tocsr()
        tangent.eliminate_zeros()
    else:
        tangent = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)

    info = {
        "matrix_type": "external_load_tangent",
        "load_case": load_name,
        "total_dofs": total_dofs,
        "num_pressure_elements": len(element_ids),
        "pressure_element_ids": element_ids,
        "pressure_configuration": (
            "current"
            if follower_pressure
            else "reference"
        ),
        "diagnostics": {
            "assembled_symmetry_error": _relative_symmetry_error(tangent),
            "element_activity": activity_info,
        },
        "assembly_time": time.time() - start_time,
    }
    pressure_surfaces = _qualified_s3_pressure_surface_records(
        model,
        load_case,
        qualified_runtime_guard=qualified_runtime_guard,
    )
    if pressure_surfaces:
        info["qualified_s3_pressure_surfaces"] = pressure_surfaces
    exact_qualified_guard(
        model,
        context="external-load tangent assembly output",
    )
    return tangent, info


def assemble_external_load_tangent(
    model: "FEModel",
    load_case: Optional["LoadCase"],
    displacements: Optional[np.ndarray] = None,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble the external-load tangent under one authority lease."""

    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="external-load tangent assembly",
        operation=lambda lease: _assemble_external_load_tangent_under_lease(
            model,
            load_case,
            displacements,
            qualified_runtime_guard=lease,
        ),
    )


def _assemble_external_load_system_under_lease(
    model: "FEModel",
    load_case: Optional["LoadCase"],
    displacements: Optional[np.ndarray] = None,
    *,
    tangent: bool = True,
    qualified_runtime_guard: Any,
) -> Tuple[np.ndarray, Optional[sparse.csr_matrix], Dict[str, Any]]:
    """Assemble external force and, optionally, its configuration tangent."""
    vector, vector_info = _assemble_load_vector_under_lease(
        model,
        load_case,
        displacements,
        qualified_runtime_guard=qualified_runtime_guard,
    )
    if tangent:
        load_tangent, tangent_info = _assemble_external_load_tangent_under_lease(
            model,
            load_case,
            displacements,
            qualified_runtime_guard=qualified_runtime_guard,
        )
    else:
        load_tangent = None
        tangent_info = None
    return vector, load_tangent, {"load": vector_info, "external_load_tangent": tangent_info}


def assemble_external_load_system(
    model: "FEModel",
    load_case: Optional["LoadCase"],
    displacements: Optional[np.ndarray] = None,
    *,
    tangent: bool = True,
) -> Tuple[np.ndarray, Optional[sparse.csr_matrix], Dict[str, Any]]:
    """Assemble the complete external-load system under one authority lease."""

    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="external-load system assembly",
        operation=lambda lease: _assemble_external_load_system_under_lease(
            model,
            load_case,
            displacements,
            tangent=tangent,
            qualified_runtime_guard=lease,
        ),
    )


def _assemble_load_matrix_under_lease(
    model: "FEModel",
    load_cases: Sequence[Optional["LoadCase"]],
    *,
    qualified_runtime_guard: Any,
    annotate_case_errors: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Assemble a dense load matrix with one column per load case."""
    from .current_state_tangent import (
        require_exact_qualified_component_lifecycle_api,
    )

    start = time.time()
    vectors = []
    infos = []
    names = []
    for index, load_case in enumerate(load_cases):
        # One complete preflight covers the whole batch.  Every case closes
        # with a guard that is complete after any foreign code it ran, and
        # the loop between cases runs none, so a later case needs no second
        # preflight.  The column name is the one the case itself reported;
        # re-reading ``load_case.name`` here would run another unguarded
        # attribute observation on arbitrary objects.
        if index == 0 or type(load_cases) not in (list, tuple):
            require_exact_qualified_component_lifecycle_api(
                model,
                context="load-matrix assembly preflight",
            )
            qualified_runtime_guard(
                model,
                context="load-matrix assembly preflight",
            )
        try:
            vector, info = _assemble_load_vector_under_lease(
                model,
                load_case,
                qualified_runtime_guard=qualified_runtime_guard,
                preflight_verified=True,
            )
        except Exception as error:
            if not annotate_case_errors:
                raise
            from .control import SolveCancelled
            if isinstance(error, SolveCancelled):
                raise
            raise LoadMatrixCaseError(index, error) from error
        vectors.append(vector)
        infos.append(info)
        names.append(info["load_case"])
    total_dofs = model.mesh.dof_manager.total_dofs
    matrix = np.column_stack(vectors) if vectors else np.zeros((total_dofs, 0), dtype=float)
    return matrix, {
        "vector_type": "load_matrix",
        "load_cases": names,
        "num_load_cases": len(names),
        "total_dofs": total_dofs,
        "assembly_time": time.time() - start,
        "columns": infos,
        "load_norms": [float(np.linalg.norm(matrix[:, idx])) for idx in range(matrix.shape[1])],
        "revision_signature": getattr(model.mesh, "revision_signature", lambda: {})(),
    }


def assemble_load_matrix(
    model: "FEModel",
    load_cases: Sequence[Optional["LoadCase"]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Assemble all load columns under one authority lease."""

    frozen_load_cases = tuple(load_cases)
    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="load-matrix assembly",
        operation=lambda lease: _assemble_load_matrix_under_lease(
            model,
            frozen_load_cases,
            qualified_runtime_guard=lease,
        ),
    )


class LoadMatrixCaseError(AssemblyError):
    """Internal case-position annotation without evaluating a failed provider."""
    def __init__(self, case_index: int, original: Exception):
        super().__init__(str(original))
        self.case_index = int(case_index)


def _assemble_load_matrix_annotated(model: "FEModel", load_cases: tuple[Any, ...]) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _run_with_qualified_assembly_runtime_lease(
        model, context="static stream load-matrix assembly",
        operation=lambda lease: _assemble_load_matrix_under_lease(
            model, load_cases, qualified_runtime_guard=lease, annotate_case_errors=True),
    )


def _assemble_damping_matrix_under_lease(
    model: "FEModel",
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
    *,
    qualified_runtime_guard: Any,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble Rayleigh damping C = alpha M + beta K."""
    start = time.time()
    if _element_activity(model) is None:
        M, mass_info = _assemble_element_matrix_under_lease(
            model,
            "mass",
            lambda element, mesh, material: element.compute_mass_matrix(
                mesh, material
            ),
            qualified_runtime_guard,
        )
        M = _add_point_masses_to_matrix(model, M)
        mass_info["diagnostics"]["point_mass_count"] = int(
            len(getattr(model.mesh, "point_masses", {}) or {})
        )
        K, stiffness_info = _assemble_element_matrix_under_lease(
            model,
            "stiffness",
            lambda element, mesh, material: element.compute_stiffness_matrix(
                mesh, material
            ),
            qualified_runtime_guard,
        )
    else:
        M, mass_info = _assemble_element_matrix_under_lease(
            model,
            "mass",
            lambda element, mesh, material: element.compute_mass_matrix(
                mesh, material
            ),
            qualified_runtime_guard,
            activity_quantity="damping",
        )
        M = _add_point_masses_to_matrix(model, M)
        K, stiffness_info = _assemble_element_matrix_under_lease(
            model,
            "stiffness",
            lambda element, mesh, material: element.compute_stiffness_matrix(
                mesh, material
            ),
            qualified_runtime_guard,
            activity_quantity="damping",
        )
    C = (float(rayleigh_alpha) * M + float(rayleigh_beta) * K).tocsr()
    return C, {
        "matrix_type": "damping",
        "rayleigh_alpha": float(rayleigh_alpha),
        "rayleigh_beta": float(rayleigh_beta),
        "mass": mass_info,
        "stiffness": stiffness_info,
        "assembly_time": time.time() - start,
        "diagnostics": {"assembled_symmetry_error": _relative_symmetry_error(C)},
        "revision_signature": getattr(model.mesh, "revision_signature", lambda: {})(),
    }


def assemble_damping_matrix(
    model: "FEModel",
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble Rayleigh damping under one authority lease."""

    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="damping assembly",
        operation=lambda lease: _assemble_damping_matrix_under_lease(
            model,
            rayleigh_alpha,
            rayleigh_beta,
            qualified_runtime_guard=lease,
        ),
    )


def _assemble_system_under_lease(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    include_mass: bool = False,
    *,
    qualified_runtime_guard: Any,
) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[str, Any]]:
    """Compatibility wrapper returning K, F and assembly metadata.

    The mass matrix is assembled separately and returned in info["mass_matrix"]
    only when include_mass is true.  It is never added to stiffness.
    """
    start_time = time.time()
    K, stiffness_info = _assemble_element_matrix_under_lease(
        model,
        "stiffness",
        lambda element, mesh, material: element.compute_stiffness_matrix(
            mesh, material
        ),
        qualified_runtime_guard,
    )
    F, load_info = _assemble_load_vector_under_lease(
        model,
        load_case,
        qualified_runtime_guard=qualified_runtime_guard,
    )

    info: Dict[str, Any] = {
        "num_elements": stiffness_info["num_elements"],
        "num_nodes": model.mesh.num_nodes,
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "includes_mass_matrix": bool(include_mass),
        "assembly_time": 0.0,
        "stiffness": stiffness_info,
        "load": load_info,
        # Backwards-compatible keys used by older diagnostics/tests.
        "element_times": stiffness_info.get("element_times", {}),
    }

    if include_mass:
        M, mass_info = _assemble_element_matrix_under_lease(
            model,
            "mass",
            lambda element, mesh, material: element.compute_mass_matrix(
                mesh, material
            ),
            qualified_runtime_guard,
        )
        M = _add_point_masses_to_matrix(model, M)
        mass_info["diagnostics"]["point_mass_count"] = int(
            len(getattr(model.mesh, "point_masses", {}) or {})
        )
        info["mass_matrix"] = M
        info["mass"] = mass_info

    info["assembly_time"] = time.time() - start_time
    return K, F, info


def assemble_system(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    include_mass: bool = False,
) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[str, Any]]:
    """Assemble the compatibility system under one authority lease."""

    return _run_with_qualified_assembly_runtime_lease(
        model,
        context="system assembly",
        operation=lambda lease: _assemble_system_under_lease(
            model,
            load_case,
            include_mass,
            qualified_runtime_guard=lease,
        ),
    )
