"""Bounded, revision-aware plans for deterministic stress recovery chunks.

The scalar element recovery routines remain the numerical oracle.  This module
removes per-call topology work and partitions that oracle into a small number
of formulation-aware chunks, avoiding one ``Future`` allocation per element.
The plan is stored as one mesh-owned entry and therefore cannot grow with
selection masks or result-output cadence.
"""

from __future__ import annotations

import math
import sys
import time
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .elements import (
    BeamElement,
    Element,
    QuadraticBeamElement,
    ShellElement,
    _shell_material_matrices,
)
from .fe_core import FEMesh as _FEMesh
from .fe_core import Material as _BuiltinMaterial
from .fe_core import Node as _Node
from .fe_core import _QualifiedStateMapping
from .materials import is_isotropic_material
from .materials import elastic_compliance_matrix, material_symmetry
from .matrix_assembly import (
    AssemblyError,
    _CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE as _CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE,
)
from .e4_pl_element import (
    QualifiedE4PLShellElement as _QualifiedE4PLShellElement,
    _q4_trusted_operation_scope as _Q4_TRUSTED_OPERATION_SCOPE,
)
from .e4_pl_s3_element import (
    QualifiedE4PLS3ShellElement as _QualifiedE4PLS3ShellElement,
    _s3_trusted_operation_scope as _S3_TRUSTED_OPERATION_SCOPE,
)
from .s3_reference_batch import (
    MIN_REFERENCE_S3_RECOVERY_GROUP,
    ReferenceS3RecoveryBatch,
    build_reference_s3_recovery_batch,
    reference_s3_candidate,
)

if TYPE_CHECKING:
    from .fe_core import FEModel


_BEAM_RECOVERY_IGNORED_INSTANCE_KEYS = frozenset(
    {
        "_stiffness_matrix",
        "_mass_matrix",
        "_internal_forces",
        "_nl_cache",
        "_nl_cache_key",
        "_qualified_direct_state_token",
        "_qualified_direct_state_tokens",
        "eccentricity",
    }
)
_BEAM_RECOVERY_UNSUPPORTED = object()
_BEAM_RECOVERY_CALLBACK_UNSAFE = object()


def _bind_exact_builtin_beam_recovery_authority() -> Callable[..., Any]:
    """Bind the exact elastic built-in beam recovery surface once."""

    exact_all = all
    exact_any = any
    exact_bool = bool
    exact_callable = callable
    exact_dict = dict
    exact_dict_get = dict.get
    exact_dict_items = dict.items
    exact_dict_setitem = dict.__setitem__
    exact_getattr = getattr
    exact_id = id
    exact_int = int
    exact_isinstance = isinstance
    exact_len = len
    exact_list = list
    exact_list_getitem = list.__getitem__
    exact_list_len = list.__len__
    exact_object_getattribute = object.__getattribute__
    exact_object_new = object.__new__
    exact_range = range
    exact_set = set
    exact_str = str
    exact_float = float
    exact_tuple = tuple
    exact_tuple_getitem = tuple.__getitem__
    exact_type = type
    exact_type_getattribute = type.__getattribute__
    exact_vars = vars
    exact_assembly_error = AssemblyError
    exact_beam_element = BeamElement
    exact_element = Element
    exact_femesh = _FEMesh
    exact_material = _BuiltinMaterial
    exact_node = _Node
    exact_quadratic_beam_element = QuadraticBeamElement
    exact_state_mapping = _QualifiedStateMapping
    exact_callback_unsafe = _BEAM_RECOVERY_CALLBACK_UNSAFE
    exact_ignored_instance_keys = _BEAM_RECOVERY_IGNORED_INSTANCE_KEYS
    exact_unsupported = _BEAM_RECOVERY_UNSUPPORTED
    elements_module = sys.modules[exact_beam_element.__module__]
    exact_isfinite = math.isfinite
    class_namespace_get = exact_type(
        exact_type_getattribute(exact_element, "__dict__")
    ).get
    ndarray_type = np.ndarray
    ndarray_tobytes = np.ndarray.tobytes
    numpy_frombuffer = np.frombuffer
    missing_class_attribute = object()
    real_numpy_scalar_types = frozenset(
        np.dtype(code).type
        for code in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
        )
    )
    beam_recovery_fields = (
        "element_id",
        "node_ids",
        "material_name",
        "cross_section",
        "_A",
        "_Iy",
        "_Iz",
        "_J",
        "_ky",
        "_kz",
        "_orientation",
        "_fiber_plasticity",
        "generalized_section",
        "_geometric_nonlinearity",
        "_c_y",
        "_c_z",
        "_torsion_modulus",
    )
    material_recovery_fields = (
        "name",
        "elastic_modulus",
        "poisson_ratio",
        "density",
        "yield_stress",
        "hardening_curve",
    )
    class_entries = (
        (
            exact_element,
            tuple(
                (
                    name,
                    class_namespace_get(
                        exact_type_getattribute(exact_element, "__dict__"),
                        name,
                    ),
                )
                for name in (
                    "get_dof_mapping",
                    "_get_element_displacements",
                    "total_dofs",
                )
            ),
        ),
        (
            exact_beam_element,
            tuple(
                (
                    name,
                    class_namespace_get(
                        exact_type_getattribute(exact_beam_element, "__dict__"),
                        name,
                    ),
                )
                for name in (
                    "get_node_coordinates",
                    "_beam_frame_and_transform",
                    "_fiber_distances",
                    "_torsion_section_modulus",
                    "_end_displacements",
                    "compute_stresses",
                    "num_nodes",
                    "dofs_per_node",
                )
            ),
        ),
        (
            exact_quadratic_beam_element,
            tuple(
                (
                    name,
                    class_namespace_get(
                        exact_type_getattribute(
                            exact_quadratic_beam_element,
                            "__dict__",
                        ),
                        name,
                    ),
                )
                for name in (
                    "get_node_coordinates",
                    "_beam_frame_and_transform",
                    "_end_displacements",
                    "compute_shape_functions",
                    "num_nodes",
                    "dofs_per_node",
                    "total_dofs",
                    "GAUSS_POINTS",
                    "GAUSS_WEIGHTS",
                )
            ),
        ),
        (
            exact_node,
            (
                (
                    "coords",
                    class_namespace_get(
                        exact_type_getattribute(exact_node, "__dict__"),
                        "coords",
                    ),
                ),
            ),
        ),
        (
            exact_femesh,
            (
                (
                    "get_node",
                    class_namespace_get(
                        exact_type_getattribute(exact_femesh, "__dict__"),
                        "get_node",
                    ),
                ),
            ),
        ),
        (
            exact_material,
            tuple(
                (
                    name,
                    class_namespace_get(
                        exact_type_getattribute(exact_material, "__dict__"),
                        name,
                    ),
                )
                for name in ("elastic_symmetry", "shear_modulus")
            ),
        ),
    )
    module_entries = tuple(
        (name, exact_dict_get(exact_vars(elements_module), name))
        for name in (
            "np",
            "_SMALL",
            "_cross3",
            "_beam_rotation_matrix",
            "_elastic_symmetry",
            "_beam_material_properties",
            "_canonical_beam_material_properties",
            "QuadraticBeamElement",
        )
    )
    gauss_arrays = tuple(
        (
            name,
            value,
            value.dtype.str,
            value.shape,
            value.strides,
            value.tobytes(order="C"),
        )
        for name, value in (
            (
                "GAUSS_POINTS",
                exact_type_getattribute(
                    exact_quadratic_beam_element,
                    "GAUSS_POINTS",
                ),
            ),
            (
                "GAUSS_WEIGHTS",
                exact_type_getattribute(
                    exact_quadratic_beam_element,
                    "GAUSS_WEIGHTS",
                ),
            ),
        )
    )
    attribute_dispatch_entries = tuple(
        (
            owner,
            tuple(
                (
                    name,
                    class_namespace_get(
                        exact_type_getattribute(owner, "__dict__"),
                        name,
                        missing_class_attribute,
                    ),
                )
                for name in names
            ),
        )
        for owner, names in (
            (
                exact_element,
                (
                    "__getattribute__",
                    "__getattr__",
                    "element_id",
                    "node_ids",
                    "material_name",
                ),
            ),
            (
                exact_beam_element,
                (
                    "__getattribute__",
                    "__getattr__",
                    "cross_section",
                    "_A",
                    "_Iy",
                    "_Iz",
                    "_J",
                    "_ky",
                    "_kz",
                    "_orientation",
                    "_c_y",
                    "_c_z",
                    "_torsion_modulus",
                    "generalized_section",
                    "_fiber_plasticity",
                ),
            ),
            (
                exact_quadratic_beam_element,
                ("__getattribute__", "__getattr__", "eccentricity"),
            ),
            (
                exact_node,
                (
                    "__getattribute__",
                    "__getattr__",
                    "id",
                    "x",
                    "y",
                    "z",
                    "dofs",
                ),
            ),
            (
                exact_state_mapping,
                ("__getattribute__", "__getattr__"),
            ),
            (
                exact_femesh,
                ("__getattribute__", "__getattr__", "nodes", "elements"),
            ),
            (
                exact_material,
                (
                    "__getattribute__",
                    "__getattr__",
                    "name",
                    "elastic_modulus",
                    "poisson_ratio",
                    "density",
                    "yield_stress",
                    "hardening_curve",
                    "hill_yield",
                ),
            ),
        )
    )

    def class_authority_is_exact() -> bool:
        for owner, entries in class_entries:
            namespace = exact_type_getattribute(owner, "__dict__")
            if exact_any(
                class_namespace_get(namespace, name) is not expected
                for name, expected in entries
            ):
                return False
        if exact_any(
            exact_dict_get(exact_vars(elements_module), name) is not expected
            for name, expected in module_entries
        ):
            return False
        if (
            exact_type_getattribute(exact_state_mapping, "get")
            is not exact_dict_get
        ):
            return False
        for owner, entries in attribute_dispatch_entries:
            namespace = exact_type_getattribute(owner, "__dict__")
            for name, expected in entries:
                current = class_namespace_get(
                    namespace,
                    name,
                    missing_class_attribute,
                )
                if current is not expected:
                    return False
        for name, expected, dtype, shape, strides, payload in gauss_arrays:
            current = class_namespace_get(
                exact_type_getattribute(
                    exact_quadratic_beam_element,
                    "__dict__",
                ),
                name,
            )
            if (
                current is not expected
                or exact_type(current) is not ndarray_type
                or current.dtype.str != dtype
                or current.shape != shape
                or current.strides != strides
                or ndarray_tobytes(current, order="C") != payload
            ):
                return False
        return True

    def frozen_value(value: Any, depth: int = 0) -> Any:
        if depth > 4:
            return exact_unsupported
        value_type = exact_type(value)
        if value is None or value_type in {exact_bool, exact_int, exact_str}:
            return (value_type.__name__, value)
        if value_type is exact_float:
            return (
                "float",
                value,
            ) if exact_isfinite(value) else exact_unsupported
        if value_type in {exact_tuple, exact_list}:
            if exact_len(value) > 32:
                return exact_unsupported
            getter = (
                exact_tuple_getitem
                if value_type is exact_tuple
                else exact_list_getitem
            )
            members = exact_tuple(
                frozen_value(getter(value, index), depth + 1)
                for index in exact_range(exact_len(value))
            )
            if exact_any(member is exact_callback_unsafe for member in members):
                return exact_callback_unsafe
            if exact_any(member is exact_unsupported for member in members):
                return exact_unsupported
            return (value_type.__name__, exact_id(value), members)
        if value_type is exact_dict:
            if exact_len(value) > 32 or not exact_all(
                exact_type(key) is exact_str for key in value
            ):
                return exact_unsupported
            members = exact_tuple(
                (key, frozen_value(member, depth + 1))
                for key, member in exact_dict_items(value)
            )
            if exact_any(
                member is exact_callback_unsafe
                for _key, member in members
            ):
                return exact_callback_unsafe
            if exact_any(
                member is exact_unsupported
                for _key, member in members
            ):
                return exact_unsupported
            return ("dict", exact_id(value), members)
        if value_type is ndarray_type:
            dtype = value.dtype
            # Object arrays can dispatch arbitrary caller code from ordinary
            # NumPy arithmetic.  Structured, subarray, textual, temporal and
            # complex dtypes are not part of the built-in elastic beam input
            # contract either.  Inspect dtype metadata before touching the
            # payload so rejection itself cannot invoke an object callback.
            if dtype.hasobject:
                return exact_callback_unsafe
            if (
                dtype.fields is not None
                or dtype.subdtype is not None
                or dtype.kind not in "biuf"
                or value.size > 32
                or not value.flags.c_contiguous
            ):
                return exact_unsupported
            return (
                "ndarray",
                exact_id(value),
                value.dtype.str,
                value.shape,
                value.strides,
                exact_bool(value.flags.writeable),
                ndarray_tobytes(value, order="C"),
            )
        return exact_unsupported

    def clone_exact_value(value: Any, depth: int = 0) -> Any:
        """Clone only callback-free exact built-ins used by beam recovery."""

        if depth > 4:
            return exact_unsupported
        if value is None or exact_type(value) in {
            exact_bool,
            exact_int,
            exact_str,
        }:
            return value
        if exact_type(value) is exact_float:
            return value if exact_isfinite(value) else exact_unsupported
        if exact_type(value) is exact_tuple:
            if exact_len(value) > 32:
                return exact_unsupported
            members = exact_tuple(
                clone_exact_value(exact_tuple_getitem(value, index), depth + 1)
                for index in exact_range(exact_len(value))
            )
            if exact_any(
                member is exact_unsupported
                or member is exact_callback_unsafe
                for member in members
            ):
                return exact_unsupported
            return members
        if exact_type(value) is exact_list:
            if exact_len(value) > 32:
                return exact_unsupported
            members = [
                clone_exact_value(exact_list_getitem(value, index), depth + 1)
                for index in exact_range(exact_len(value))
            ]
            if exact_any(
                member is exact_unsupported
                or member is exact_callback_unsafe
                for member in members
            ):
                return exact_unsupported
            return members
        if exact_type(value) is exact_dict:
            if exact_len(value) > 32 or not exact_all(
                exact_type(key) is exact_str for key in value
            ):
                return exact_unsupported
            made: dict[str, Any] = {}
            for key, member in exact_dict_items(value):
                cloned = clone_exact_value(member, depth + 1)
                if (
                    cloned is exact_unsupported
                    or cloned is exact_callback_unsafe
                ):
                    return exact_unsupported
                exact_dict_setitem(made, key, cloned)
            return made
        if exact_type(value) is ndarray_type:
            dtype = value.dtype
            if dtype.hasobject:
                return exact_callback_unsafe
            if (
                dtype.fields is not None
                or dtype.subdtype is not None
                or dtype.kind not in "biuf"
                or value.size > 32
                or not value.flags.c_contiguous
            ):
                return exact_unsupported
            payload = ndarray_tobytes(value, order="C")
            return numpy_frombuffer(payload, dtype=dtype).reshape(value.shape)
        return exact_unsupported

    def semantic_exact_value(value: Any, depth: int = 0) -> Any:
        """Identity-free representation of the exact mechanics value."""

        if depth > 4:
            return exact_unsupported
        if value is None or exact_type(value) in {
            exact_bool,
            exact_int,
            exact_str,
        }:
            return (exact_type(value).__name__, value)
        if exact_type(value) is exact_float:
            return (
                ("float", value)
                if exact_isfinite(value)
                else exact_unsupported
            )
        if exact_type(value) in {exact_tuple, exact_list}:
            if exact_len(value) > 32:
                return exact_unsupported
            getter = (
                exact_tuple_getitem
                if exact_type(value) is exact_tuple
                else exact_list_getitem
            )
            members = exact_tuple(
                semantic_exact_value(getter(value, index), depth + 1)
                for index in exact_range(exact_len(value))
            )
            if exact_any(member is exact_unsupported for member in members):
                return exact_unsupported
            return (exact_type(value).__name__, members)
        if exact_type(value) is exact_dict:
            if exact_len(value) > 32 or not exact_all(
                exact_type(key) is exact_str for key in value
            ):
                return exact_unsupported
            members = exact_tuple(
                (key, semantic_exact_value(member, depth + 1))
                for key, member in exact_dict_items(value)
            )
            if exact_any(
                member is exact_unsupported
                for _key, member in members
            ):
                return exact_unsupported
            return ("dict", members)
        if exact_type(value) is ndarray_type:
            dtype = value.dtype
            if (
                dtype.hasobject
                or dtype.fields is not None
                or dtype.subdtype is not None
                or dtype.kind not in "biuf"
                or value.size > 32
                or not value.flags.c_contiguous
            ):
                return exact_unsupported
            return (
                "ndarray",
                dtype.str,
                value.shape,
                ndarray_tobytes(value, order="C"),
            )
        return exact_unsupported

    def clone_exact_record(
        source: Any,
        expected_type: type,
        field_names: tuple[str, ...],
    ) -> Any:
        if exact_type(source) is not expected_type:
            return exact_unsupported
        namespace = exact_object_getattribute(source, "__dict__")
        if exact_type(namespace) is not exact_dict or exact_any(
            name not in namespace for name in field_names
        ):
            return exact_unsupported
        before = exact_tuple(
            (name, semantic_exact_value(exact_dict_get(namespace, name)))
            for name in field_names
        )
        if exact_any(
            value is exact_unsupported for _name, value in before
        ):
            return exact_unsupported
        made = exact_object_new(expected_type)
        made_namespace = exact_object_getattribute(made, "__dict__")
        for name in field_names:
            cloned = clone_exact_value(exact_dict_get(namespace, name))
            if (
                cloned is exact_unsupported
                or cloned is exact_callback_unsafe
            ):
                return exact_unsupported
            exact_dict_setitem(made_namespace, name, cloned)
        after = exact_tuple(
            (name, semantic_exact_value(exact_dict_get(namespace, name)))
            for name in field_names
        )
        isolated = exact_tuple(
            (name, semantic_exact_value(exact_dict_get(made_namespace, name)))
            for name in field_names
        )
        if before != after or before != isolated:
            return exact_unsupported
        return made

    def frozen_real_scalar(value: Any) -> Any:
        if exact_type(value) is exact_int:
            return ("int", value)
        if exact_type(value) is exact_float:
            return (
                ("float", value)
                if exact_isfinite(value)
                else exact_unsupported
            )
        value_type = exact_type(value)
        if value_type in real_numpy_scalar_types:
            dtype = value.dtype
            if (
                value_type is not dtype.type
                or dtype.hasobject
                or dtype.fields is not None
                or dtype.subdtype is not None
                or dtype.kind not in "iuf"
            ):
                return exact_callback_unsafe
            numeric = exact_float(value)
            if not exact_isfinite(numeric):
                return exact_unsupported
            return (
                "numpy-scalar",
                value_type,
                dtype.str,
                value.tobytes(),
            )
        return exact_callback_unsafe

    def node_fingerprint(node: Any) -> Any:
        if exact_type(node) is not exact_node:
            return exact_callback_unsafe
        namespace = exact_object_getattribute(node, "__dict__")
        if exact_type(namespace) is not exact_dict or "coords" in namespace:
            return exact_callback_unsafe
        node_id = exact_dict_get(namespace, "id", exact_callback_unsafe)
        coordinates = exact_tuple(
            frozen_real_scalar(
                exact_dict_get(namespace, name, exact_callback_unsafe)
            )
            for name in ("x", "y", "z")
        )
        dofs = exact_dict_get(namespace, "dofs", exact_callback_unsafe)
        revision = exact_dict_get(
            namespace,
            "_coordinate_revision",
            exact_callback_unsafe,
        )
        if (
            exact_any(value is exact_callback_unsafe for value in coordinates)
            or exact_any(value is exact_unsupported for value in coordinates)
            or exact_type(node_id) is not exact_int
            or exact_type(revision) is not exact_int
            or exact_type(dofs) is not exact_list
            or exact_len(dofs) != 6
            or exact_any(
                exact_type(exact_list_getitem(dofs, index)) is not exact_int
                or exact_list_getitem(dofs, index) < 0
                for index in exact_range(6)
            )
        ):
            return exact_callback_unsafe
        return (
            exact_id(namespace),
            node_id,
            coordinates,
            revision,
            exact_id(dofs),
            exact_tuple(
                exact_list_getitem(dofs, index) for index in exact_range(6)
            ),
        )

    def element_node_authority(
        element: Any,
        nodes: Any,
    ) -> Any:
        namespace = exact_object_getattribute(element, "__dict__")
        node_ids = exact_dict_get(namespace, "node_ids")
        expected_count = 2 if exact_type(element) is exact_beam_element else 3
        if (
            exact_type(node_ids) not in {exact_list, exact_tuple}
            or exact_len(node_ids) != expected_count
            or exact_any(
                exact_type(node_id) is not exact_int for node_id in node_ids
            )
        ):
            return exact_callback_unsafe
        records = []
        for node_id in node_ids:
            node = exact_dict_get(nodes, node_id)
            fingerprint = node_fingerprint(node)
            if (
                fingerprint is exact_callback_unsafe
                or fingerprint is exact_unsupported
            ):
                return fingerprint
            records.append((node_id, node, fingerprint))
        return exact_tuple(records)

    def node_authority_is_exact(nodes: Any, records: Any) -> bool:
        if exact_type(nodes) is not exact_state_mapping:
            return False
        for node_id, node, fingerprint in records:
            if (
                exact_dict_get(nodes, node_id) is not node
                or node_fingerprint(node) != fingerprint
            ):
                return False
        return True

    def instance_fingerprint(element: Any) -> Any:
        namespace = exact_object_getattribute(element, "__dict__")
        if exact_type(namespace) is not exact_dict or not exact_all(
            exact_type(name) is exact_str for name in namespace
        ):
            return exact_unsupported
        shadowed = {
            "compute_stresses",
            "get_dof_mapping",
            "get_node_coordinates",
            "_beam_frame_and_transform",
            "_get_element_displacements",
            "_end_displacements",
            "_fiber_distances",
            "_torsion_section_modulus",
            "compute_shape_functions",
            "total_dofs",
            "num_nodes",
            "dofs_per_node",
        }.intersection(namespace)
        if shadowed:
            return exact_unsupported
        if (
            exact_dict_get(namespace, "generalized_section") is not None
            or exact_dict_get(namespace, "_fiber_plasticity")
            not in {None, False}
        ):
            return exact_unsupported
        values = []
        for name, value in exact_dict_items(namespace):
            if name in exact_ignored_instance_keys:
                continue
            frozen = frozen_value(value)
            if frozen is exact_callback_unsafe:
                return exact_callback_unsafe
            if frozen is exact_unsupported:
                return exact_unsupported
            values.append((name, frozen))
        return (exact_id(namespace), exact_tuple(values))

    def material_fingerprint(material: Any) -> Any:
        if exact_type(material) is not exact_material:
            return exact_unsupported
        namespace = exact_object_getattribute(material, "__dict__")
        if (
            exact_type(namespace) is not exact_dict
            or exact_set(namespace)
            != {
                "name",
                "elastic_modulus",
                "poisson_ratio",
                "density",
                "yield_stress",
                "hardening_curve",
            }
            or exact_dict_get(namespace, "hardening_curve") is not None
        ):
            return exact_unsupported
        frozen = frozen_value(namespace)
        if frozen is exact_callback_unsafe:
            return frozen
        if frozen is exact_unsupported:
            return frozen
        return (exact_id(namespace), frozen)

    def capture(model: "FEModel", lease: Any) -> Any:
        namespace = exact_getattr(lease, "__dict__", {})
        item_provider = (
            exact_dict_get(namespace, "_qualified_owned_element_items")
            if exact_type(namespace) is exact_dict
            else None
        )
        material_provider = (
            exact_dict_get(namespace, "_qualified_owned_material_name")
            if exact_type(namespace) is exact_dict
            else None
        )
        mesh = (
            exact_dict_get(namespace, "_qualified_owned_mesh")
            if exact_type(namespace) is exact_dict
            else None
        )
        model_namespace = exact_object_getattribute(model, "__dict__")
        if (
            not exact_callable(item_provider)
            or not exact_callable(material_provider)
            or exact_type(mesh) is not exact_femesh
            or exact_type(model_namespace) is not exact_dict
            or exact_dict_get(model_namespace, "mesh") is not mesh
        ):
            return None
        if not class_authority_is_exact():
            unsafe_elements = {
                exact_id(element): element
                for _element_id, element in item_provider()
                if exact_type(element)
                in {exact_beam_element, exact_quadratic_beam_element}
            }
            if not unsafe_elements:
                return None

            def unsafe_candidate(element: Any) -> bool:
                return (
                    exact_dict_get(unsafe_elements, exact_id(element))
                    is element
                )

            def unsafe_require(element: Any, material: Any) -> bool:
                del material
                if unsafe_candidate(element):
                    raise exact_assembly_error(
                        "exact built-in beam recovery attribute authority changed"
                    )
                return False

            def unsafe_dof_mapping(element: Any, material: Any) -> Any:
                unsafe_require(element, material)
                return None

            def unsafe_recover(
                element: Any,
                mesh: Any,
                displacements: Any,
                material: Any,
                return_global: bool,
            ) -> Any:
                del mesh, displacements, return_global
                unsafe_require(element, material)
                return None

            return (
                unsafe_candidate,
                unsafe_require,
                unsafe_dof_mapping,
                unsafe_recover,
            )
        mesh_namespace = exact_object_getattribute(mesh, "__dict__")
        materials = exact_dict_get(model_namespace, "materials")
        mapping = exact_dict_get(mesh_namespace, "elements")
        nodes = exact_dict_get(mesh_namespace, "nodes")
        token = exact_dict_get(
            mesh_namespace,
            "_qualified_direct_state_token",
        )
        if (
            exact_type(model_namespace) is not exact_dict
            or exact_type(materials) is not exact_dict
            or exact_type(mesh_namespace) is not exact_dict
            or "get_node" in mesh_namespace
            or exact_type(nodes) is not exact_state_mapping
            or not exact_isinstance(token, exact_list)
            or exact_list_len(token) != 1
            or exact_type(exact_list_getitem(token, 0)) is not exact_int
        ):
            return None
        token_value = exact_int(exact_list_getitem(token, 0))
        records: dict[int, tuple[Any, ...]] = {}
        unsafe_records: dict[int, tuple[Any, ...]] = {}
        for element_id, element in item_provider():
            if exact_type(element) not in {
                exact_beam_element,
                exact_quadratic_beam_element,
            }:
                continue
            element_namespace = exact_object_getattribute(
                element,
                "__dict__",
            )
            material_name = exact_dict_get(
                element_namespace,
                "material_name",
            )
            if exact_type(material_name) is not exact_str:
                continue
            material = material_provider(material_name)
            element_fingerprint = instance_fingerprint(element)
            exact_material_fingerprint = material_fingerprint(material)
            if (
                element_fingerprint is exact_callback_unsafe
                or exact_material_fingerprint is exact_callback_unsafe
            ):
                unsafe_records[exact_id(element)] = (
                    exact_int(element_id),
                    element,
                )
                continue
            node_authority = element_node_authority(element, nodes)
            if node_authority is exact_callback_unsafe:
                unsafe_records[exact_id(element)] = (
                    exact_int(element_id),
                    element,
                )
                continue
            if (
                element_fingerprint is exact_unsupported
                or exact_material_fingerprint is exact_unsupported
                or node_authority is exact_unsupported
            ):
                continue
            isolated_element = clone_exact_record(
                element,
                exact_type(element),
                beam_recovery_fields,
            )
            isolated_material = clone_exact_record(
                material,
                exact_material,
                material_recovery_fields,
            )
            if (
                isolated_element is exact_unsupported
                or isolated_material is exact_unsupported
                or instance_fingerprint(isolated_element) is exact_unsupported
                or instance_fingerprint(isolated_element)
                is exact_callback_unsafe
                or exact_type(isolated_material) is not exact_material
                or material_fingerprint(isolated_material) is exact_unsupported
                or material_fingerprint(isolated_material)
                is exact_callback_unsafe
            ):
                continue
            dof_mapping = exact_tuple(
                dof
                for _node_id, _node, fingerprint in node_authority
                for dof in fingerprint[5]
            )
            records[exact_id(element)] = (
                exact_int(element_id),
                element,
                material,
                element_fingerprint,
                exact_material_fingerprint,
                node_authority,
                isolated_element,
                isolated_material,
                dof_mapping,
            )
        if not records and not unsafe_records:
            return None

        def candidate(element: Any) -> bool:
            record = exact_dict_get(records, exact_id(element))
            unsafe = exact_dict_get(unsafe_records, exact_id(element))
            return exact_bool(
                record is not None and record[1] is element
                or unsafe is not None and unsafe[1] is element
            )

        def require(element: Any, material: Any) -> bool:
            unsafe = exact_dict_get(unsafe_records, exact_id(element))
            if unsafe is not None and unsafe[1] is element:
                raise exact_assembly_error(
                    "exact built-in beam recovery input can dispatch caller code"
                )
            record = exact_dict_get(records, exact_id(element))
            if record is None or record[1] is not element:
                return False
            if (
                not class_authority_is_exact()
                or exact_object_getattribute(model, "__dict__")
                is not model_namespace
                or exact_dict_get(model_namespace, "mesh") is not mesh
                or exact_dict_get(model_namespace, "materials") is not materials
                or exact_object_getattribute(mesh, "__dict__")
                is not mesh_namespace
                or exact_dict_get(mesh_namespace, "elements") is not mapping
                or exact_dict_get(mesh_namespace, "nodes") is not nodes
                or "get_node" in mesh_namespace
                or exact_dict_get(
                    mesh_namespace,
                    "_qualified_direct_state_token",
                )
                is not token
                or exact_int(exact_list_getitem(token, 0)) != token_value
                or exact_dict_get(mapping, record[0]) is not element
                or material is not record[2]
                or instance_fingerprint(element) != record[3]
                or material_fingerprint(material) != record[4]
                or not node_authority_is_exact(nodes, record[5])
            ):
                raise exact_assembly_error(
                    "exact built-in beam recovery authority changed"
                )
            return True

        def captured_dof_mapping(element: Any, material: Any) -> Any:
            if not require(element, material):
                return None
            return exact_dict_get(records, exact_id(element))[8]

        def recover(
            element: Any,
            observed_mesh: Any,
            displacements: Any,
            material: Any,
            return_global: bool,
        ) -> Any:
            if observed_mesh is not mesh or not require(element, material):
                raise exact_assembly_error(
                    "exact built-in beam recovery authority changed"
                )
            record = exact_dict_get(records, exact_id(element))
            isolated_element = record[6]
            isolated_material = record[7]
            return isolated_element.compute_stresses(
                mesh,
                displacements,
                isolated_material,
                return_global=return_global,
            )

        return candidate, require, captured_dof_mapping, recover

    capture._qualified_class_authority_is_exact = class_authority_is_exact
    return capture


_CAPTURE_EXACT_BUILTIN_BEAM_RECOVERY_AUTHORITY = (
    _bind_exact_builtin_beam_recovery_authority()
)


def _bind_clear_recovery_plan_without_callbacks() -> Any:
    """Bind raw namespace access used while unwinding a hostile callback."""

    exact_dict = dict
    exact_dict_get = dict.get
    exact_dict_pop = dict.pop
    exact_object_getattribute = object.__getattribute__
    exact_type = type

    def clear(model: "FEModel") -> None:
        model_namespace = exact_object_getattribute(model, "__dict__")
        if exact_type(model_namespace) is not exact_dict:
            return
        mesh = exact_dict_get(model_namespace, "mesh")
        mesh_namespace = exact_object_getattribute(mesh, "__dict__")
        if exact_type(mesh_namespace) is exact_dict:
            exact_dict_pop(mesh_namespace, "_recovery_batch_plan", None)

    return clear


_clear_recovery_plan_without_callbacks = (
    _bind_clear_recovery_plan_without_callbacks()
)


def _require_exact_builtin_beam_recovery_class_authority(
    model: "FEModel",
) -> None:
    """Reject a changed exact built-in beam recovery attribute authority."""

    capture_namespace = getattr(
        _CAPTURE_EXACT_BUILTIN_BEAM_RECOVERY_AUTHORITY,
        "__dict__",
        {},
    )
    class_authority_check = (
        dict.get(
            capture_namespace,
            "_qualified_class_authority_is_exact",
        )
        if type(capture_namespace) is dict
        else None
    )
    if callable(class_authority_check) and not class_authority_check():
        model_namespace = object.__getattribute__(model, "__dict__")
        raw_mesh = (
            dict.get(model_namespace, "mesh")
            if type(model_namespace) is dict
            else None
        )
        raw_mesh_namespace = (
            object.__getattribute__(raw_mesh, "__dict__")
            if type(raw_mesh) is _FEMesh
            else None
        )
        raw_elements = (
            dict.get(raw_mesh_namespace, "elements")
            if type(raw_mesh_namespace) is dict
            else None
        )
        if isinstance(raw_elements, dict) and any(
            type(element) in {BeamElement, QuadraticBeamElement}
            for element in dict.values(raw_elements)
        ):
            raise AssemblyError(
                "exact built-in beam recovery attribute authority changed"
            )


def _build_qualified_recovery_require(
    model: "FEModel",
    lease: Any,
    exact_beam_authority: Any,
    *,
    context: str,
) -> Callable[..., Any]:
    """Build the per-operation guard for one captured recovery lease."""

    def require(*, stage: str) -> None:
        try:
            lease(model, context=f"{context} {stage}")
        except BaseException:
            _clear_recovery_plan_without_callbacks(model)
            raise

    lease_namespace = getattr(lease, "__dict__", {})
    trusted_lease_require = (
        dict.get(lease_namespace, "_qualified_trusted_require")
        if type(lease_namespace) is dict
        else None
    )
    trusted_element_lease_require = (
        dict.get(lease_namespace, "_qualified_trusted_element_require")
        if type(lease_namespace) is dict
        else None
    )
    owned_material_provider = (
        dict.get(lease_namespace, "_qualified_owned_material")
        if type(lease_namespace) is dict
        else None
    )
    if callable(owned_material_provider):
        def exact_qualified_element_candidate(element: Any) -> bool:
            return owned_material_provider(element) is not None

        require._qualified_exact_element_candidate = (
            exact_qualified_element_candidate
        )
    if callable(trusted_lease_require):
        def trusted_require(*, stage: str) -> None:
            """Check an all-qualified owned recovery loop in constant time."""

            try:
                trusted_lease_require(
                    model,
                    context=f"{context} {stage}",
                )
            except BaseException:
                _clear_recovery_plan_without_callbacks(model)
                raise

        # Keep the cheap check private to the complete recovery lease.  The
        # outer preflight/output checks remain full, while exact qualified
        # element loops can reject generation or mesh-token ABA without
        # rescanning every element for every observation.
        require._qualified_trusted_recovery_require = trusted_require

    if (
        callable(trusted_element_lease_require)
        and callable(owned_material_provider)
    ):
        def trusted_element_require(
            element: Any,
            material: Any,
            *,
            stage: str,
        ) -> bool:
            """Check one exact qualified element inside a mixed-model loop."""

            owned_material = owned_material_provider(element)
            if owned_material is None:
                return False
            try:
                trusted_element_lease_require(
                    model,
                    element,
                    material,
                    context=f"{context} {stage}",
                )
            except BaseException:
                _clear_recovery_plan_without_callbacks(model)
                raise
            return True

        require._qualified_trusted_recovery_element_require = (
            trusted_element_require
        )

    if exact_beam_authority is not None:
        (
            exact_beam_candidate,
            exact_beam_require,
            exact_beam_dof_mapping,
            exact_beam_recover,
        ) = exact_beam_authority

        def trusted_beam_require(
            element: Any,
            material: Any,
            *,
            stage: str,
        ) -> bool:
            """Check one exact elastic built-in beam inside its full bracket."""

            del stage
            try:
                return bool(exact_beam_require(element, material))
            except BaseException:
                _clear_recovery_plan_without_callbacks(model)
                raise

        require._qualified_exact_builtin_beam_candidate = (
            exact_beam_candidate
        )
        require._qualified_trusted_recovery_beam_require = (
            trusted_beam_require
        )

        def captured_beam_dof_mapping(
            element: Any,
            material: Any,
        ) -> Any:
            try:
                return exact_beam_dof_mapping(element, material)
            except BaseException:
                _clear_recovery_plan_without_callbacks(model)
                raise

        def recover_captured_beam(
            element: Any,
            mesh: Any,
            displacements: Any,
            material: Any,
            return_global: bool,
        ) -> Any:
            try:
                return exact_beam_recover(
                    element,
                    mesh,
                    displacements,
                    material,
                    return_global,
                )
            except BaseException:
                _clear_recovery_plan_without_callbacks(model)
                raise

        require._qualified_captured_beam_dof_mapping = (
            captured_beam_dof_mapping
        )
        require._qualified_recover_captured_beam = recover_captured_beam

    return require


def _qualified_recovery_trusted_elements(
    model: "FEModel",
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Collect the qualified Q4/S3 elements entering trusted scopes."""

    qualified_elements = tuple(model.mesh.elements.values())
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
    return q4_elements, s3_elements


_ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE: ContextVar[Any] = ContextVar(
    "qualified_recovery_runtime_lease_scope",
    default=None,
)


@contextmanager
def _qualified_recovery_runtime_lease_scope(
    model: "FEModel",
    *,
    context: str,
) -> Any:
    """Lend one captured recovery lease to package-owned block operations.

    The lease, the exact built-in beam authority and the Q4/S3 trusted
    operation scopes are captured once per block.  Every borrowing
    operation still runs the complete class-authority, preflight and
    output guards under its own context, and any guard failure clears
    the retained recovery plan.  The scope is private to the package: it
    never crosses a yield, progress callback or input iteration
    boundary, and a borrowing operation must present the same model
    object the lease was captured for.  Leaving the scope revokes the
    published borrow explicitly, so a context copied while the scope
    was live cannot retain a usable borrowed lease once the scope has
    closed; dispatch in such a copied context falls back to the
    ordinary fresh lease capture below.
    """

    _require_exact_builtin_beam_recovery_class_authority(model)
    lease = _CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE(
        model,
        context=f"{context} preflight",
    )
    exact_beam_authority = (
        _CAPTURE_EXACT_BUILTIN_BEAM_RECOVERY_AUTHORITY(model, lease)
    )
    entry_guard = _build_qualified_recovery_require(
        model,
        lease,
        exact_beam_authority,
        context=context,
    )
    live = [True]

    def borrow(
        borrowed_model: "FEModel",
        *,
        operation_context: str,
        operation: Callable[[Any], Any],
    ) -> Any:
        """Run one operation under full pre/post borrowed guards."""

        if not live[0]:
            raise AssemblyError(
                "borrowed recovery lease scope is no longer active"
            )
        if borrowed_model is not model:
            raise AssemblyError(
                "borrowed recovery lease model identity changed"
            )
        _require_exact_builtin_beam_recovery_class_authority(model)
        guard = _build_qualified_recovery_require(
            model,
            lease,
            exact_beam_authority,
            context=operation_context,
        )
        guard(stage="preflight")
        try:
            result = operation(guard)
        except BaseException:
            guard(stage="exceptional output")
            raise
        guard(stage="output")
        return result

    stack = ExitStack()
    marker = None
    published = False
    try:
        try:
            q4_elements, s3_elements = _qualified_recovery_trusted_elements(
                model
            )
            stack.enter_context(_Q4_TRUSTED_OPERATION_SCOPE(q4_elements))
            stack.enter_context(_S3_TRUSTED_OPERATION_SCOPE(s3_elements))
        except BaseException:
            entry_guard(stage="exceptional output")
            raise
        marker = _ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE.set(
            (model, borrow, live)
        )
        published = True
        yield borrow
    except BaseException as primary:
        # Error, cancellation and close all revoke the published borrow,
        # clear the context marker and close the trusted scopes.  A
        # cleanup failure is noted on the primary failure instead of
        # replacing it, so the case failure identity survives.
        live[0] = False
        if published:
            _ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE.reset(marker)
        try:
            stack.close()
        except BaseException as cleanup_error:
            BaseException.add_note(
                primary,
                "recovery lease scope cleanup failed: " + repr(cleanup_error),
            )
        raise
    live[0] = False
    _ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE.reset(marker)
    stack.close()


def _run_with_qualified_recovery_runtime_lease(
    model: "FEModel",
    *,
    context: str,
    operation: Callable[[Any], Any],
) -> Any:
    """Run plan/recovery work under one non-renewable family lease."""

    active_scope = _ACTIVE_QUALIFIED_RECOVERY_RUNTIME_LEASE_SCOPE.get()
    if (
        active_scope is not None
        and active_scope[0] is model
        and active_scope[2][0]
    ):
        return active_scope[1](
            model,
            operation_context=context,
            operation=operation,
        )
    # A scope published for another model, or one retained by a context
    # copied before the live scope closed, takes the ordinary fresh
    # capture path below instead of reusing the expired closure.

    _require_exact_builtin_beam_recovery_class_authority(model)
    lease = _CAPTURE_QUALIFIED_RECOVERY_RUNTIME_LEASE(
        model,
        context=f"{context} preflight",
    )
    exact_beam_authority = (
        _CAPTURE_EXACT_BUILTIN_BEAM_RECOVERY_AUTHORITY(model, lease)
    )
    require = _build_qualified_recovery_require(
        model,
        lease,
        exact_beam_authority,
        context=context,
    )
    try:
        q4_elements, s3_elements = _qualified_recovery_trusted_elements(model)
        with ExitStack() as stack:

            stack.enter_context(_Q4_TRUSTED_OPERATION_SCOPE(q4_elements))
            stack.enter_context(_S3_TRUSTED_OPERATION_SCOPE(s3_elements))
            result = operation(require)
    except BaseException:
        require(stage="exceptional output")
        raise
    require(stage="output")
    return result


def _qualified_recovery_observation_guard(
    qualified_runtime_guard: Any,
) -> Callable[..., None]:
    """Route owned qualified observations through bounded lease checks."""

    namespace = getattr(qualified_runtime_guard, "__dict__", {})
    trusted = (
        dict.get(namespace, "_qualified_trusted_recovery_require")
        if type(namespace) is dict
        else None
    )
    trusted_element = (
        dict.get(
            namespace,
            "_qualified_trusted_recovery_element_require",
        )
        if type(namespace) is dict
        else None
    )
    exact_beam_candidate = (
        dict.get(namespace, "_qualified_exact_builtin_beam_candidate")
        if type(namespace) is dict
        else None
    )
    exact_qualified_candidate = (
        dict.get(namespace, "_qualified_exact_element_candidate")
        if type(namespace) is dict
        else None
    )
    trusted_beam = (
        dict.get(namespace, "_qualified_trusted_recovery_beam_require")
        if type(namespace) is dict
        else None
    )

    def require(
        *,
        stage: str,
        element: Any = None,
        material: Any = None,
    ) -> None:
        if (
            callable(trusted_element)
            and element is not None
            and material is not None
        ):
            if trusted_element(
                element,
                material,
                stage=stage,
            ):
                return
        if (
            callable(trusted_beam)
            and element is not None
            and material is not None
            and trusted_beam(
                element,
                material,
                stage=stage,
            )
        ):
            return
        if callable(trusted):
            trusted(stage=stage)
            return
        qualified_runtime_guard(stage=stage)

    if callable(exact_beam_candidate):
        require._qualified_exact_builtin_beam_candidate = (
            exact_beam_candidate
        )
    if callable(exact_beam_candidate) or callable(exact_qualified_candidate):
        def trusted_segment_candidate(element: Any) -> bool:
            return bool(
                callable(exact_beam_candidate)
                and exact_beam_candidate(element)
                or callable(exact_qualified_candidate)
                and exact_qualified_candidate(element)
            )

        require._qualified_recovery_trusted_segment_candidate = (
            trusted_segment_candidate
        )
    return require


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    return np.frombuffer(
        result.tobytes(order="C"), dtype=result.dtype
    ).reshape(result.shape)


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


def _material_state_key(model: "FEModel") -> Tuple[Tuple[object, ...], ...]:
    names = sorted(str(name) for name in model.materials)
    rows = []
    for name in names:
        material = model.get_material(name)
        if is_isotropic_material(material):
            elastic = (
                "isotropic",
                float(material.elastic_modulus),
                float(material.poisson_ratio),
            )
        else:
            compliance = np.ascontiguousarray(
                np.asarray(elastic_compliance_matrix(material), dtype=np.float64)
            )
            elastic = (
                str(material_symmetry(material)),
                compliance.shape,
                compliance.tobytes(order="C"),
            )
        hill = getattr(material, "hill_yield", None)
        hardening = getattr(material, "hardening_curve", None)
        rows.append(
            (
                name,
                type(material).__module__,
                type(material).__qualname__,
                id(material),
                bool(is_isotropic_material(material)),
                elastic,
                hill is None,
                id(hill),
                hardening is None,
                id(hardening),
            )
        )
    return tuple(rows)


def _formulation_name(model: "FEModel", element: object) -> str:
    if isinstance(element, ShellElement):
        node_count = len(element.node_ids)
        topology = {3: "t3", 4: "s4", 6: "t6", 8: "q8"}.get(
            node_count, f"shell{node_count}"
        )
        if bool(getattr(element, "reduced_integration", False)):
            topology += "r"
        if getattr(element, "shell_section", None) is not None:
            constitutive = "generalized"
        else:
            material = model.get_material(element.material_name)
            if is_isotropic_material(material):
                constitutive = (
                    "isotropic_hill"
                    if getattr(material, "hill_yield", None) is not None
                    else "isotropic"
                )
            else:
                constitutive = "orthotropic"
        return f"shell_{topology}_{constitutive}"
    if isinstance(element, QuadraticBeamElement):
        return "beam3"
    if isinstance(element, BeamElement):
        return "beam2"
    return f"scalar_{type(element).__name__}"


@dataclass(frozen=True)
class RecoveryPlanItem:
    element_id: int
    formulation: str
    dof_mapping: np.ndarray


@dataclass(frozen=True)
class RecoveryBatchPlan:
    """Immutable layout shared by repeated recovery calls for one mesh."""

    revision_key: Tuple[int, int, int]
    direct_state_key: Tuple[int, int, int, int]
    material_state_key: Tuple[Tuple[object, ...], ...]
    items: Tuple[RecoveryPlanItem, ...]
    item_by_id: Mapping[int, RecoveryPlanItem]
    setup_seconds: float
    retained_bytes: int
    isotropic_s4: "RecoveryS4Batch | None"
    reference_s3: "ReferenceS3RecoveryBatch | None"
    reference_s3_candidate_ids: Tuple[int, ...]
    reference_s3_fallback_reasons: Mapping[str, Tuple[int, ...]]

    @classmethod
    def build(cls, model: "FEModel") -> "RecoveryBatchPlan":
        from .fe_core import _ensure_qualified_state_mappings

        # A wholesale public mapping replacement invalidates the prior plan
        # by identity.  Normalize the replacement before capturing a new key
        # so subsequent direct node/element mutations remain observable.
        _ensure_qualified_state_mappings(model.mesh)
        start = time.perf_counter()
        items = []
        s4_element_ids = []
        s4_coords = []
        s4_dof_mappings = []
        s4_q_local = []
        s4_g_local = []
        s4_thickness = []
        reference_s3_items = []
        retained_bytes = 0
        for element_id, element in model.mesh.elements.items():
            mapping = _readonly(
                np.asarray(
                    element.get_dof_mapping(model.mesh), dtype=np.intp
                ).reshape(-1)
            )
            retained_bytes += int(mapping.nbytes)
            items.append(
                RecoveryPlanItem(
                    element_id=int(element_id),
                    formulation=_formulation_name(model, element),
                    dof_mapping=mapping,
                )
            )
            if reference_s3_candidate(element):
                reference_s3_items.append(
                    (int(element_id), element, mapping)
                )
            if items[-1].formulation in {"shell_s4_isotropic", "shell_s4r_isotropic"}:
                material = model.get_material(element.material_name)
                q_local, g_local, _strain_transform, _stress_transform = (
                    _shell_material_matrices(material, 0.0)
                )
                s4_element_ids.append(int(element_id))
                s4_coords.append(
                    np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
                )
                s4_dof_mappings.append(mapping)
                s4_q_local.append(np.asarray(q_local, dtype=float))
                s4_g_local.append(np.asarray(g_local, dtype=float))
                s4_thickness.append(float(element.thickness))
        item_tuple = tuple(items)
        isotropic_s4 = None
        if s4_element_ids:
            isotropic_s4 = RecoveryS4Batch(
                element_ids=_readonly(np.asarray(s4_element_ids, dtype=np.int64)),
                index_by_id=MappingProxyType({
                    int(element_id): index
                    for index, element_id in enumerate(s4_element_ids)
                }),
                coords=_readonly(np.asarray(s4_coords, dtype=float)),
                dof_mappings=_readonly(np.asarray(s4_dof_mappings, dtype=np.intp)),
                q_local=_readonly(np.asarray(s4_q_local, dtype=float)),
                g_local=_readonly(np.asarray(s4_g_local, dtype=float)),
                thickness=_readonly(np.asarray(s4_thickness, dtype=float)),
                gauss_points=_readonly(
                    np.asarray(ShellElement.GAUSS_POINTS_2x2, dtype=float)
                ),
            )
            retained_bytes += isotropic_s4.retained_bytes
        reference_s3 = None
        reference_s3_candidate_ids: Tuple[int, ...] = ()
        reference_s3_fallback_reasons: Mapping[str, Tuple[int, ...]] = (
            MappingProxyType({})
        )
        # Keep small selections on the existing scalar oracle with no retained
        # batch state.  This is intentionally checked before component
        # preparation so ordinary small models pay no S3 batch setup cost.
        if len(reference_s3_items) >= MIN_REFERENCE_S3_RECOVERY_GROUP:
            reference_s3, prepared_s3 = build_reference_s3_recovery_batch(
                model,
                reference_s3_items,
            )
            reference_s3_candidate_ids = prepared_s3.candidate_element_ids
            reference_s3_fallback_reasons = prepared_s3.fallback_reasons
            if reference_s3 is not None:
                retained_bytes += reference_s3.retained_bytes
        return cls(
            revision_key=_revision_key(model),
            direct_state_key=_direct_state_key(model),
            material_state_key=_material_state_key(model),
            items=item_tuple,
            item_by_id=MappingProxyType(
                {item.element_id: item for item in item_tuple}
            ),
            setup_seconds=float(time.perf_counter() - start),
            retained_bytes=int(retained_bytes),
            isotropic_s4=isotropic_s4,
            reference_s3=reference_s3,
            reference_s3_candidate_ids=reference_s3_candidate_ids,
            reference_s3_fallback_reasons=reference_s3_fallback_reasons,
        )

    def is_valid(self, model: "FEModel") -> bool:
        return (
            self.revision_key == _revision_key(model)
            and self.direct_state_key == _direct_state_key(model)
            and self.material_state_key == _material_state_key(model)
        )

    def select(self, element_ids: Iterable[int]) -> Tuple[RecoveryPlanItem, ...]:
        wanted = {int(element_id) for element_id in element_ids}
        return tuple(item for item in self.items if item.element_id in wanted)


def _get_recovery_batch_plan_under_lease(
    model: "FEModel",
    *,
    qualified_runtime_guard: Any,
) -> Tuple[RecoveryBatchPlan, bool]:
    """Return the mesh-owned plan and whether it was reused."""

    qualified_runtime_guard(stage="plan lookup")
    cached = getattr(model.mesh, "_recovery_batch_plan", None)
    cached_is_current = False
    if isinstance(cached, RecoveryBatchPlan):
        cached_is_current = cached.is_valid(model)
        qualified_runtime_guard(stage="plan validation observation")
    if cached_is_current:
        return cached, True
    plan = RecoveryBatchPlan.build(model)
    qualified_runtime_guard(stage="plan build observation")
    model.mesh._recovery_batch_plan = plan
    qualified_runtime_guard(stage="plan publication")
    return plan, False


def get_recovery_batch_plan(model: "FEModel") -> Tuple[RecoveryBatchPlan, bool]:
    """Return the mesh-owned plan under one authority lease."""

    return _run_with_qualified_recovery_runtime_lease(
        model,
        context="recovery batch plan",
        operation=lambda guard: _get_recovery_batch_plan_under_lease(
            model,
            qualified_runtime_guard=guard,
        ),
    )


@dataclass(frozen=True)
class RecoveryS4Batch:
    element_ids: np.ndarray
    index_by_id: Mapping[int, int]
    coords: np.ndarray
    dof_mappings: np.ndarray
    q_local: np.ndarray
    g_local: np.ndarray
    thickness: np.ndarray
    gauss_points: np.ndarray

    @property
    def retained_bytes(self) -> int:
        return int(
            self.element_ids.nbytes
            + self.coords.nbytes
            + self.dof_mappings.nbytes
            + self.q_local.nbytes
            + self.g_local.nbytes
            + self.thickness.nbytes
            + self.gauss_points.nbytes
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


def clear_recovery_batch_plan(model: "FEModel") -> None:
    if hasattr(model.mesh, "_recovery_batch_plan"):
        delattr(model.mesh, "_recovery_batch_plan")


def formulation_counts(
    items: Sequence[RecoveryPlanItem],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        counts[item.formulation] = counts.get(item.formulation, 0) + 1
    return counts


def _split_evenly(
    items: Sequence[RecoveryPlanItem], chunk_count: int
) -> Tuple[Tuple[RecoveryPlanItem, ...], ...]:
    if not items:
        return ()
    count = max(1, min(int(chunk_count), len(items)))
    chunk_size = int(math.ceil(len(items) / count))
    return tuple(
        tuple(items[start : start + chunk_size])
        for start in range(0, len(items), chunk_size)
    )


def build_recovery_chunks(
    items: Sequence[RecoveryPlanItem],
    worker_count: int,
    *,
    chunks_per_worker: int = 3,
) -> Tuple[Tuple[RecoveryPlanItem, ...], ...]:
    """Build coarse formulation-homogeneous chunks in deterministic order."""

    if not items:
        return ()
    groups: "OrderedDict[str, list[RecoveryPlanItem]]" = OrderedDict()
    for item in items:
        groups.setdefault(item.formulation, []).append(item)
    target = min(
        len(items),
        max(len(groups), max(int(worker_count), 1) * int(chunks_per_worker)),
    )
    chunks = []
    total = len(items)
    for group_items in groups.values():
        proportional = max(
            1,
            int(round(target * len(group_items) / total)),
        )
        chunks.extend(_split_evenly(group_items, proportional))
    return tuple(chunks)
