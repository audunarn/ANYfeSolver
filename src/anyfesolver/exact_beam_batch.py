"""Batched linear stiffness and stress recovery for exact built-in beams.

Scope (owner decisions D1 and D2, 2026-11-24).  When *every* element of a
model is an admitted exact :class:`~anyfesolver.elements.BeamElement`, the
linear stiffness assembly and the stress recovery evaluate all elements in one
batch instead of one Python call per element:

* **Guard bracket (D1).**  The per-element lease guards of the scalar loops
  (two per element in stiffness assembly, two per element in stress recovery)
  are replaced by one complete ``qualified_runtime_guard`` call before the
  batch reads anything and one after its last output.  This relies on the
  existing contract that changing a model *during* a request -- including from
  another thread -- is outside the supported contract.  Single-threaded calls
  accept and reject exactly the inputs the scalar loops accept and reject:
  admission runs no caller code, and any irregularity routes the whole call to
  the unchanged scalar loop before any side effect.
* **Bit-identical batch kernel (D2).**  Every BLAS-backed operation of the
  scalar code (``np.dot`` of 3-vectors, ``np.linalg.norm``, the 12x12 ``@``
  products) is evaluated with stacked ``np.matmul``, which issues the same BLAS
  call per element.  Elementwise ``+ - * /``, ``sqrt``, ``abs`` and comparisons
  are vectorised with the scalar operation order and operand values.  Scalar
  ``**`` stays a per-element Python/NumPy-scalar expression of the same type as
  the scalar code.  The batch runs with every floating-point error raising, so a
  computation that would warn or raise in the scalar code falls back to it.
* **Dispatch authority.**  Every dispatch the scalar loop performs for an
  admitted element is either verified first, with pure reads only, to be the
  original object (class namespaces along the MRO, module globals, instance
  namespaces free of shadowing names, the classification predicates of the
  scalar loop), or performed as the identical call: ``model.get_material`` is
  the verified original reading exact dicts, and ``_beam_material_properties``
  is called once per element, in element order, with the element's own
  ``cross_section`` -- exactly the calls ``compute_stiffness_matrix`` and
  ``compute_stresses`` make.  Order: authority, then admission (pure reads),
  then the calls and the computation.
* **Runtime probe.**  Once per process the batch is compared bitwise with the
  scalar methods on a fixed probe set covering every branch; each batch then
  re-checks its first and last element against the scalar methods.  Any
  mismatch disables the batch for the process and the call falls back.  The
  probe and the endpoint checks only detect a broken kernel or platform; they
  do not prove parity for the interior elements of a batch.  That parity rests
  on the batch performing the identical floating-point operations (above).

* **Trust boundary (owner decision B, 2026-11-24).**  In scope -- the batch
  behaves exactly like the scalar loops: hostile *data*, instance attributes
  and subclasses of elements, models, materials, meshes and mappings; caches
  held on model/mesh instances (``_sparsity_cache``,
  ``_topology_signature_cache``); and the kernels' inputs.  Every value the
  batch reads, looks up, hashes, compares, iterates or writes before its
  decision first passes an exact-type check made with pure reads
  (``type(x) is ...``; namespaces through ``object.__getattribute__``);
  anything non-exact declines before it is touched.  A declined batch has no
  side effect: no element or mesh cache is written before the decision (the
  endpoint stiffness checks do not store ``_stiffness_matrix``; the sparsity
  kernel runs once, after the decision), and the recovery bracket resolves
  the first material only for an exact ``str`` name.  The classification
  predicates, the recovery observation-guard factory and the element-matrix
  check kernels are identity-checked against the objects bound at import.
  Trusted and *not* defended: (1) library internals -- NumPy, ANYmaterial,
  Python builtins, finalisers/``__del__`` order, ``__code__``/``__closure__``
  swaps, frame inspection, and stateful dependencies that observe repeated
  identical calls (the probe, the endpoint checks, or a declined batch
  followed by the scalar loop); (2) ANYfeSolver's own guard, lease and check
  machinery -- lease and require builders (e.g.
  ``_run_with_qualified_assembly_runtime_lease``,
  ``recovery_batches._build_qualified_recovery_require``), guard factories,
  internal caches as module/class machinery, module imports and aliases.

The scalar element methods remain the numerical oracle.  This module keeps no
model, mesh, material or displacement data beyond one call.
"""

from __future__ import annotations

import abc
import builtins
import math
import time
from itertools import chain, repeat
from operator import is_
from typing import Any, Callable

import numpy as np

from . import e4_pl_element as _q4_module
from . import elements as _elements_module
from . import fe_core as _fe_core_module
from . import s3_reference_batch as _s3_reference_module
from . import s3_v2c_fast_assembly as _s3_v2c_module
from . import s3_v2d_fast_assembly as _s3_v2d_module
from .elements import BeamElement, Element, QuadraticBeamElement
from .fe_core import FEMesh, FEModel, Node, _QualifiedStateMapping
from .fe_core import Material as _BuiltinMaterial

#: Smallest element count for which the batch is attempted.  Smaller models
#: take the scalar loop, whose cost is negligible there.
MIN_BATCH_ELEMENTS = 8

#: Elements per stacked kernel call; bounds the temporary memory.
CHUNK_ELEMENTS = 4096


def _bind_exact_beam_batch() -> Any:
    """Bind the batch, its authority snapshot and its probe once."""

    exact_type = type
    exact_len = len
    exact_int = int
    exact_float = float
    exact_str = str
    exact_bool = bool
    exact_dict = dict
    exact_list = list
    exact_tuple = tuple
    exact_all = all
    exact_any = any
    exact_map = map
    exact_zip = zip
    exact_range = range
    exact_id = id
    exact_min = min
    exact_max = max
    chunk_elements = CHUNK_ELEMENTS
    exact_enumerate = enumerate
    exact_is = is_
    exact_repeat = repeat
    exact_chain = chain
    exact_isfinite = math.isfinite
    exact_object_getattribute = object.__getattribute__
    exact_type_getattribute = type.__getattribute__
    exact_dict_get = dict.get
    exact_dict_items = dict.items
    exact_dict_values = dict.values
    exact_dict_keys = dict.keys
    exact_dict_len = dict.__len__
    exact_isinstance = isinstance
    exact_frozenset = frozenset
    exact_set = set
    exact_exception = Exception
    clock = time.perf_counter
    none_type = type(None)

    beam_type = BeamElement
    element_type = Element
    node_type = Node
    mesh_type = FEMesh
    model_type = FEModel
    material_type = _BuiltinMaterial
    state_mapping_type = _QualifiedStateMapping
    float64_type = np.float64
    ndarray_type = np.ndarray
    float64_dtype = np.dtype(np.float64)
    intp_dtype = np.dtype(np.intp)

    np_array = np.array
    np_empty = np.empty
    np_zeros = np.zeros
    np_ones = np.ones
    np_eye = np.eye
    np_matmul = np.matmul
    np_sqrt = np.sqrt
    np_abs = np.abs
    np_where = np.where
    np_isfinite = np.isfinite
    np_all = np.all
    np_any = np.any
    np_nonzero = np.flatnonzero
    np_errstate = np.errstate
    ndarray_tobytes = np.ndarray.tobytes

    elements_namespace = vars(_elements_module)
    fe_core_namespace = vars(_fe_core_module)
    builtins_namespace = vars(builtins)
    small = elements_namespace["_SMALL"]
    beam_material_properties = elements_namespace["_beam_material_properties"]
    shell_element_type = elements_namespace["ShellElement"]
    abc_meta_type = abc.ABCMeta
    model_get_material = vars(FEModel)["get_material"]
    missing = object()
    max_exact_int = 2**53

    def class_namespace(cls: Any) -> Any:
        return exact_type_getattribute(cls, "__dict__")

    mappingproxy_get = exact_type(class_namespace(element_type)).get

    instance_fields = (
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
        "_c_y",
        "_c_z",
        "_torsion_modulus",
        "generalized_section",
        "_fiber_plasticity",
        "_geometric_nonlinearity",
        "_linear_nodal_offsets",
        "_stiffness_matrix",
        "_mass_matrix",
        "_internal_forces",
        "recovery_errors_fail_closed",
        "formulation_id",
        "shell_section",
    )
    dispatch_names = (
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__bool__",
        "__len__",
    )
    class_spec = (
        (
            element_type,
            dispatch_names
            + instance_fields
            + (
                "__dict__",
                "get_dof_mapping",
                "_get_element_displacements",
                "total_dofs",
                "compute_stiffness_matrix",
                "compute_stresses",
            ),
        ),
        (
            beam_type,
            dispatch_names
            + instance_fields
            + (
                "num_nodes",
                "dofs_per_node",
                "total_dofs",
                "get_dof_mapping",
                "_get_element_displacements",
                "get_node_coordinates",
                "_beam_frame_and_transform",
                "_local_linear_stiffness",
                "_linear_rigid_arm_map",
                "compute_stiffness_matrix",
                "_fiber_distances",
                "_torsion_section_modulus",
                "_end_displacements",
                "compute_stresses",
            ),
        ),
        (abc.ABC, dispatch_names + instance_fields),
        # ``isinstance(element, ShellElement)`` in the scalar classification.
        (abc_meta_type, ("__instancecheck__", "__subclasscheck__")),
        (
            node_type,
            dispatch_names + ("coords", "id", "x", "y", "z", "dofs"),
        ),
        (
            mesh_type,
            dispatch_names + ("get_node", "nodes", "elements"),
        ),
        (
            model_type,
            dispatch_names
            + ("get_material", "mesh", "materials", "current_material"),
        ),
        (
            material_type,
            dispatch_names
            + (
                "elastic_symmetry",
                "shear_modulus",
                "elastic_compliance_matrix",
                "hill_yield",
                "name",
                "elastic_modulus",
                "poisson_ratio",
                "density",
                "yield_stress",
                "hardening_curve",
            ),
        ),
        (
            state_mapping_type,
            dispatch_names
            + (
                "get",
                "items",
                "values",
                "keys",
                "__iter__",
                "__getitem__",
                "__contains__",
            ),
        ),
    )
    class_entries = exact_tuple(
        (
            cls,
            exact_tuple(
                (name, mappingproxy_get(class_namespace(cls), name, missing))
                for name in names
            ),
        )
        for cls, names in class_spec
    )
    mro_entries = exact_tuple(
        (cls, exact_type_getattribute(cls, "__mro__"))
        for cls in (
            beam_type,
            node_type,
            mesh_type,
            model_type,
            material_type,
            state_mapping_type,
            shell_element_type,
        )
    )
    builtin_names = (
        "abs",
        "bool",
        "dict",
        "enumerate",
        "float",
        "getattr",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "range",
        "str",
        "super",
        "tuple",
        "type",
        "IndexError",
        "ValueError",
    )
    module_entries = exact_tuple(
        (namespace, name, exact_dict_get(namespace, name, missing))
        for namespace, names in (
            (
                elements_namespace,
                (
                    "np",
                    "_SMALL",
                    "_cross3",
                    "_beam_rotation_matrix",
                    "_contiguous_vector_norm",
                    "_frame_columns",
                    "_elastic_symmetry",
                    # Called by ``_elastic_symmetry`` once per reduction.
                    "_run_post_observation_guard",
                    "_beam_material_properties",
                    "_canonical_beam_material_properties",
                    "_canonical_material_symmetry",
                    "QuadraticBeamElement",
                    "BeamElement",
                    "Element",
                    "ShellElement",
                )
                + builtin_names,
            ),
            (fe_core_namespace, ("np",) + builtin_names),
            (builtins_namespace, builtin_names),
        )
        for name in names
    )
    numpy_entries = exact_tuple(
        (owner, name, getattr(owner, name))
        for owner, names in (
            (
                np,
                (
                    "array",
                    "zeros",
                    "empty",
                    "eye",
                    "sqrt",
                    "dot",
                    "asarray",
                    "column_stack",
                    "any",
                    "all",
                    "isfinite",
                    "linalg",
                    "ndarray",
                    "float64",
                    "intp",
                ),
            ),
            (np.linalg, ("norm",)),
        )
        for name in names
    )
    # A module state that is already unexpected when this module is bound
    # disables the batch for the process instead of failing the import.
    def resolve_class_attribute(cls: Any, name: str) -> Any:
        # Pure MRO walk over class namespaces (no descriptor is invoked).
        for owner in exact_type_getattribute(cls, "__mro__"):
            value = mappingproxy_get(class_namespace(owner), name, missing)
            if value is not missing:
                return value
        return missing

    # ``str(getattr(element, "formulation_id", ""))`` and
    # ``getattr(element, "shell_section", None)`` of the scalar
    # classification resolve to these class-level values for an admitted
    # element (its instance namespace cannot hold either name).
    beam_formulation_id = resolve_class_attribute(beam_type, "formulation_id")
    if beam_formulation_id is missing:
        beam_formulation_id = ""
    bound_state_is_expected = (
        elements_namespace.get("np") is np
        and fe_core_namespace.get("np") is np
        and elements_namespace.get("QuadraticBeamElement") is QuadraticBeamElement
        and elements_namespace.get("BeamElement") is BeamElement
        and elements_namespace.get("Element") is Element
        and exact_type(shell_element_type) is abc_meta_type
        and exact_type(beam_formulation_id) is exact_str
        and resolve_class_attribute(beam_type, "shell_section") is missing
        and resolve_class_attribute(beam_type, "_linear_nodal_offsets") is missing
    )

    def authority_is_exact() -> bool:
        if not bound_state_is_expected:
            return False
        for cls, entries in class_entries:
            namespace = class_namespace(cls)
            for name, expected in entries:
                if mappingproxy_get(namespace, name, missing) is not expected:
                    return False
        for cls, mro in mro_entries:
            current = exact_type_getattribute(cls, "__mro__")
            # Identity only: tuple ``==`` could reach a metaclass ``__eq__``.
            if exact_type(current) is not exact_tuple or exact_len(current) != exact_len(
                mro
            ) or not exact_all(exact_map(exact_is, current, mro)):
                return False
        for namespace, name, expected in module_entries:
            if exact_dict_get(namespace, name, missing) is not expected:
                return False
        for owner, name, expected in numpy_entries:
            if exact_dict_get(exact_object_getattribute(owner, "__dict__"), name, missing) is not expected:
                return False
        return True

    # ------------------------------------------------------------------
    # The scalar classification loop of the stiffness assembly imports, at
    # call time, ``QualifiedE4PLShellElement`` and ``FORMULATION_ID`` of the
    # Q4 module, ``ShellElement`` and the predicates ``v2c_fast_candidate``,
    # ``v2d_fast_candidate`` and ``reference_s3_candidate``, and calls the
    # predicates for every element.  The originals are
    # ``<inner>(element)`` -> ``type(element) is <class> and ...``, i.e.
    # ``False`` for an exact ``BeamElement`` without further dispatch.  Each
    # object the scalar loop would use is compared by identity, with pure
    # reads only, against the object captured here when this module is bound:
    # the classifier objects themselves, and the inner function and class
    # global each predicate reads from its module namespace.  No attribute of
    # a caller-supplied object is read and none of its methods (``__eq__``,
    # ``__ne__``, ``__name__``) is invoked.  Anything else falls back.
    # ------------------------------------------------------------------

    q4_classifiers = (
        vars(_q4_module)["QualifiedE4PLShellElement"],
        vars(_q4_module)["FORMULATION_ID"],
        shell_element_type,
    )
    predicate_bindings = exact_tuple(
        (
            vars(module)[name],
            vars(module),
            inner_name,
            vars(module)[inner_name],
            class_name,
            vars(module)[class_name],
        )
        for module, name, inner_name, class_name in (
            (
                _s3_v2c_module,
                "v2c_fast_candidate",
                "_candidate",
                "StrictFlatLinearE4PLS3V2CShellElement",
            ),
            (
                _s3_v2d_module,
                "v2d_fast_candidate",
                "_candidate",
                "NativeParityE4PLS3V2DShellElement",
            ),
            (
                _s3_reference_module,
                "reference_s3_candidate",
                "_is_exact_qualified_s3",
                "QualifiedE4PLS3ShellElement",
            ),
        )
    )
    classifiers_are_beam_free = (
        exact_type(q4_classifiers[1]) is exact_str
        and exact_type(beam_formulation_id) is exact_str
        and q4_classifiers[1] != beam_formulation_id
        and q4_classifiers[0] is not beam_type
        and exact_all(binding[5] is not beam_type for binding in predicate_bindings)
    )

    def classification_is_noop(classifiers: Any, element_items: Any) -> bool:
        """The scalar classification loop leaves every admitted beam alone.

        ``classifiers`` are the objects the scalar loop imports at call time:
        ``(QualifiedE4PLShellElement, Q4 FORMULATION_ID, ShellElement,
        v2c_fast_candidate, v2d_fast_candidate, reference_s3_candidate)``.
        """

        if (
            not classifiers_are_beam_free
            or exact_type(classifiers) is not exact_tuple
            or exact_len(classifiers) != 6
        ):
            return False
        for supplied, expected in exact_zip(classifiers, q4_classifiers):
            if supplied is not expected:
                return False
        for supplied, binding in exact_zip(classifiers[3:], predicate_bindings):
            function, namespace, inner_name, inner, class_name, cls = binding
            if supplied is not function:
                return False
            # Module namespace keys are read through the C dict storage; a
            # non-``str`` key could run caller code during a lookup.
            if not all_str_keys(namespace):
                return False
            if (
                exact_dict_get(namespace, inner_name, missing) is not inner
                or exact_dict_get(namespace, class_name, missing) is not cls
            ):
                return False
        # ``isinstance(element, ShellElement)``: the identical call of the
        # scalar loop (ABCMeta hooks verified by ``authority_is_exact``; the
        # ABC cache makes repeated calls return the first answer).
        for item in element_items:
            if exact_isinstance(item[1], shell_element_type):
                return False
        return True

    allowed_element_keys = exact_frozenset(
        {
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
            "_c_y",
            "_c_z",
            "_torsion_modulus",
            "generalized_section",
            "_fiber_plasticity",
            "_geometric_nonlinearity",
            "_stiffness_matrix",
            "_mass_matrix",
            "_internal_forces",
            "_nl_cache",
            "_nl_cache_key",
            "_qualified_direct_state_token",
            "_qualified_direct_state_tokens",
        }
    )
    required_element_keys = exact_frozenset(
        {
            "node_ids",
            "material_name",
            "cross_section",
            "_A",
            "_Iy",
            "_Iz",
            "_ky",
            "_kz",
            "_orientation",
            "_c_y",
            "_c_z",
            "_torsion_modulus",
            "generalized_section",
            "_fiber_plasticity",
        }
    )
    material_keys = exact_frozenset(
        {
            "name",
            "elastic_modulus",
            "poisson_ratio",
            "density",
            "yield_stress",
            "hardening_curve",
        }
    )
    plain_section_value_types = (
        float,
        int,
        bool,
        str,
        none_type,
        float64_type,
        list,
        tuple,
    )

    def type_is_one_of(value_type: Any, candidates: tuple[Any, ...]) -> bool:
        # Identity only: `in` would compare with `==`, which a metaclass
        # could implement.
        return exact_any(exact_map(exact_is, exact_repeat(value_type), candidates))

    # Keys are always read through ``dict.keys`` (the C dict storage), so a
    # replaced ``__iter__`` of a dict subclass is never invoked.
    def all_str_keys(mapping: Any) -> bool:
        return exact_all(
            exact_map(
                exact_is,
                exact_map(exact_type, exact_dict_keys(mapping)),
                exact_repeat(exact_str),
            )
        )

    def all_int_keys(mapping: Any) -> bool:
        return exact_all(
            exact_map(
                exact_is,
                exact_map(exact_type, exact_dict_keys(mapping)),
                exact_repeat(exact_int),
            )
        )

    def plain_number(value: Any) -> bool:
        value_type = exact_type(value)
        if value_type is exact_float or value_type is float64_type:
            return exact_isfinite(value)
        if value_type is exact_int:
            return -max_exact_int <= value <= max_exact_int
        return False

    def optional_plain_number(value: Any) -> bool:
        return value is None or plain_number(value)

    # The private state of a ``_QualifiedStateMapping`` instance.  Any other
    # instance attribute could shadow ``get``/``items``/``keys``/``values``,
    # which the scalar path calls through attribute lookup.
    state_mapping_instance_keys = exact_frozenset({"_qualified_token", "_qualified_kind"})

    def plain_mapping(value: Any, key_check: Callable[[Any], bool]) -> bool:
        if exact_type(value) is state_mapping_type:
            namespace = exact_object_getattribute(value, "__dict__")
            if (
                exact_type(namespace) is not exact_dict
                or not all_str_keys(namespace)
                or not exact_dict_keys(namespace) <= state_mapping_instance_keys
            ):
                return False
        elif exact_type(value) is not exact_dict:
            return False
        return key_check(value)

    def node_record(node: Any, dof_limit: int) -> Any:
        if exact_type(node) is not node_type:
            return None
        namespace = exact_object_getattribute(node, "__dict__")
        if (
            exact_type(namespace) is not exact_dict
            or not all_str_keys(namespace)
            or "coords" in namespace
        ):
            return None
        x = exact_dict_get(namespace, "x", missing)
        y = exact_dict_get(namespace, "y", missing)
        z = exact_dict_get(namespace, "z", missing)
        dofs = exact_dict_get(namespace, "dofs", missing)
        if (
            not plain_numbers((x, y, z))
            or exact_type(dofs) is not exact_list
            or exact_len(dofs) != 6
            or not exact_all(
                exact_map(exact_is, exact_map(exact_type, dofs), exact_repeat(exact_int))
            )
            or exact_min(dofs) < 0
            or exact_max(dofs) >= dof_limit
        ):
            # Every DOF is an exact int in [0, dof_limit): no conversion of
            # the batch can raise where the scalar path would raise later.
            return None
        return (exact_float(x), exact_float(y), exact_float(z)), exact_tuple(dofs)

    def material_record(material: Any) -> Any:
        if exact_type(material) is not material_type:
            return None
        namespace = exact_object_getattribute(material, "__dict__")
        if (
            exact_type(namespace) is not exact_dict
            or not all_str_keys(namespace)
            or exact_dict_keys(namespace) != material_keys
            or namespace["hardening_curve"] is not None
            or exact_type(namespace["name"]) is not exact_str
        ):
            return None
        for name in ("elastic_modulus", "poisson_ratio", "density", "yield_stress"):
            value = namespace[name]
            if exact_type(value) is not exact_float and exact_type(value) is not exact_int:
                return None
            if exact_type(value) is exact_float and not exact_isfinite(value):
                return None
        return namespace

    plain_section_type_ids = exact_frozenset(
        exact_map(exact_id, plain_section_value_types)
    )
    float_type_id = exact_id(exact_float)

    def plain_numbers(values: tuple[Any, ...]) -> bool:
        # Fast path: exact finite Python floats, decided in C.
        if exact_all(
            exact_map(exact_is, exact_map(exact_type, values), exact_repeat(exact_float))
        ) and exact_all(exact_map(exact_isfinite, values)):
            return True
        return exact_all(exact_map(plain_number, values))

    def gather(
        element_items: Any,
        nodes: Any,
        resolve_material: Callable[[str], Any],
        dof_limit: Any,
    ) -> Any:
        """Read every input of the batch without running caller code.

        Returns ``None`` when anything is outside the admitted exact surface.
        """

        if exact_type(dof_limit) is not exact_int:
            return None
        count = exact_len(element_items)
        node_index: dict[int, int] = {}
        node_coordinates: list[Any] = []
        node_dofs: list[Any] = []
        material_by_name: dict[str, int] = {}
        material_list: list[Any] = []
        rows: list[tuple[Any, ...]] = []
        append_row = rows.append
        orientation_rows: list[int] = []
        orientation_values: list[Any] = []
        str_type = exact_str
        int_type = exact_int
        for position, item in exact_enumerate(element_items):
            if exact_type(item) is not exact_tuple or exact_len(item) != 2:
                return None
            element_id, element = item
            if exact_type(element) is not beam_type or exact_type(element_id) is not int_type:
                return None
            namespace = exact_object_getattribute(element, "__dict__")
            if exact_type(namespace) is not exact_dict or not exact_all(
                exact_map(exact_is, exact_map(exact_type, namespace), exact_repeat(str_type))
            ):
                return None
            keys = exact_dict_keys(namespace)
            if not (keys <= allowed_element_keys and keys >= required_element_keys):
                return None
            if (
                namespace["generalized_section"] is not None
                or namespace["_fiber_plasticity"] is not None
            ):
                return None
            node_ids = namespace["node_ids"]
            if (
                exact_type(node_ids) is not exact_list
                and exact_type(node_ids) is not exact_tuple
            ) or exact_len(node_ids) != 2:
                return None
            first_id, second_id = node_ids
            if exact_type(first_id) is not int_type or exact_type(second_id) is not int_type:
                return None
            material_name = namespace["material_name"]
            if exact_type(material_name) is not str_type:
                return None
            cross_section = namespace["cross_section"]
            if exact_type(cross_section) is not exact_dict or not exact_all(
                exact_map(exact_is, exact_map(exact_type, cross_section), exact_repeat(str_type))
            ):
                return None
            if not exact_all(
                exact_map(
                    plain_section_type_ids.__contains__,
                    exact_map(exact_id, exact_map(exact_type, exact_dict_values(cross_section))),
                )
            ):
                for value in exact_dict_values(cross_section):
                    if not type_is_one_of(exact_type(value), plain_section_value_types):
                        if not (
                            exact_type(value) is ndarray_type
                            and value.dtype.kind in "biuf"
                        ):
                            return None
            torsion_constant = exact_dict_get(cross_section, "J", 1.0e-8)
            area = namespace["_A"]
            iy = namespace["_Iy"]
            iz = namespace["_Iz"]
            ky = namespace["_ky"]
            kz = namespace["_kz"]
            c_y = namespace["_c_y"]
            c_z = namespace["_c_z"]
            torsion_modulus = namespace["_torsion_modulus"]
            if not plain_numbers((torsion_constant, area, iy, iz, ky, kz)):
                return None
            if (
                c_y is not None
                and not plain_number(c_y)
                or c_z is not None
                and not plain_number(c_z)
                or torsion_modulus is not None
                and not plain_number(torsion_modulus)
            ):
                return None
            orientation = namespace["_orientation"]
            if orientation is not None:
                if (
                    exact_type(orientation) is not ndarray_type
                    or orientation.dtype != float64_dtype
                    or orientation.shape != (3,)
                    or not orientation.flags.c_contiguous
                ):
                    return None
                values = orientation.tolist()
                if not exact_all(exact_map(exact_isfinite, values)):
                    return None
                orientation_rows.append(position)
                orientation_values.append(values)
            first_index = exact_dict_get(node_index, first_id)
            if first_index is None:
                record = node_record(exact_dict_get(nodes, first_id), dof_limit)
                if record is None:
                    return None
                first_index = node_index[first_id] = exact_len(node_coordinates)
                node_coordinates.append(record[0])
                node_dofs.append(record[1])
            second_index = exact_dict_get(node_index, second_id)
            if second_index is None:
                record = node_record(exact_dict_get(nodes, second_id), dof_limit)
                if record is None:
                    return None
                second_index = node_index[second_id] = exact_len(node_coordinates)
                node_coordinates.append(record[0])
                node_dofs.append(record[1])
            material_position = exact_dict_get(material_by_name, material_name)
            if material_position is None:
                material = resolve_material(material_name)
                if material is missing:
                    return None
                for known_position, known in exact_enumerate(material_list):
                    if known is material:
                        material_position = known_position
                        break
                else:
                    if material_record(material) is None:
                        return None
                    material_position = exact_len(material_list)
                    material_list.append(material)
                material_by_name[material_name] = material_position
            append_row(
                (
                    element_id,
                    element,
                    namespace,
                    (first_index, second_index),
                    material_position,
                    area,
                    iy,
                    iz,
                    ky,
                    kz,
                    exact_float(torsion_constant),
                    c_y,
                    c_z,
                    torsion_modulus,
                )
            )
        if exact_len(rows) != count or count == 0:
            return None
        (
            element_ids,
            element_list,
            namespaces,
            pair_index,
            material_index,
            areas,
            inertia_y,
            inertia_z,
            shear_y,
            shear_z,
            torsion_constants,
            fiber_y,
            fiber_z,
            torsion_moduli,
        ) = exact_zip(*rows)
        return {
            "element_ids": element_ids,
            "elements": element_list,
            "namespaces": namespaces,
            "node_coordinates": node_coordinates,
            "node_dofs": node_dofs,
            "pair_index": pair_index,
            "materials": material_list,
            "material_index": material_index,
            "areas": areas,
            "inertia_y": inertia_y,
            "inertia_z": inertia_z,
            "shear_y": shear_y,
            "shear_z": shear_z,
            "torsion_constants": torsion_constants,
            "fiber_y": fiber_y,
            "fiber_z": fiber_z,
            "torsion_moduli": torsion_moduli,
            "orientation_rows": orientation_rows,
            "orientation_values": orientation_values,
        }

    # ------------------------------------------------------------------
    # Kernel.  Every comment names the scalar expression it reproduces.
    # ------------------------------------------------------------------

    rigid_arm = np_eye(12)
    # BeamElement._linear_rigid_arm_map without offsets.
    rigid_arm[(0, 1, 2, 6, 7, 8), (5, 3, 4, 11, 9, 10)] = -0.0
    rigid_arm.flags.writeable = False
    trial_y = np_array([0.0, 1.0, 0.0])
    trial_z = np_array([0.0, 0.0, 1.0])

    def dot3(first: Any, second: Any) -> Any:
        """``np.dot(a_i, b_i)`` per row: one BLAS dot per element."""

        return np_matmul(first[:, None, :], second[:, :, None])[:, 0, 0]

    def cross3(first: Any, second: Any) -> Any:
        """``_cross3(a_i, b_i)`` per row, same component operation order."""

        made = np_empty(first.shape, dtype=float64_dtype)
        made[:, 0] = first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1]
        made[:, 1] = first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2]
        made[:, 2] = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        return made

    def scalar_max(values: Any, floor: float) -> Any:
        """Python ``max(value, floor)``: ``floor`` only when it is larger."""

        return np_where(floor > values, floor, values)

    def prepared_arrays(plan: dict[str, Any]) -> Any:
        """Element coordinates and orientations as arrays, once per batch."""

        count = exact_len(plan["element_ids"])
        node_xyz = np_array(plan["node_coordinates"], dtype=float64_dtype)
        pairs = np_array(plan["pair_index"], dtype=intp_dtype)
        # get_node_coordinates: zeros((2, 3)) filled per node, then ``+ 0``.
        coordinates = node_xyz[pairs] + 0
        oriented = np_zeros(count, dtype=bool)
        orientation = np_zeros((count, 3), dtype=float64_dtype)
        rows = np_array(plan["orientation_rows"], dtype=intp_dtype)
        if rows.size:
            oriented[rows] = True
            orientation[rows] = np_array(
                plan["orientation_values"], dtype=float64_dtype
            )
        return coordinates, oriented, orientation

    def chunks(count: int) -> Any:
        for begin in exact_range(0, count, chunk_elements):
            yield begin, exact_min(count, begin + chunk_elements)

    def frames(coordinates: Any, oriented: Any, all_orientations: Any) -> Any:
        """Lengths and transforms of ``BeamElement._beam_frame_and_transform``."""

        count = coordinates.shape[0]
        start = coordinates[:, 0, :]
        end = coordinates[:, 1, :]
        # length = float(_contiguous_vector_norm(coords[1] - coords[0]))
        difference = end - start
        lengths = np_sqrt(dot3(difference, difference))
        if not exact_bool(np_all(lengths >= small)):
            return None
        # e1 = (coords[1] - coords[0]) / length
        axis = (end - start) / lengths[:, None]
        second = np_empty((count, 3), dtype=float64_dtype)
        third = np_empty((count, 3), dtype=float64_dtype)
        heuristic = np_ones(count, dtype=bool)
        rows = np_nonzero(oriented)
        if rows.size:
            orientation = all_orientations[rows]
            oriented_axis = axis[rows]
            # candidate = orientation - np.dot(orientation, e1) * e1
            projection = dot3(orientation, oriented_axis)
            candidate = orientation - projection[:, None] * oriented_axis
            # norm = float(_contiguous_vector_norm(candidate))
            candidate_norm = np_sqrt(dot3(candidate, candidate))
            # norm > 1.0e-6 * float(np.linalg.norm(orientation))
            orientation_norm = np_sqrt(dot3(orientation, orientation))
            accepted = candidate_norm > 1.0e-6 * orientation_norm
            if exact_bool(np_any(accepted)):
                used = np_nonzero(accepted)
                target = rows[used]
                # e3 = candidate / norm
                local_z = candidate[used] / candidate_norm[used][:, None]
                # e2 = _cross3(e3, e1); e2 /= _contiguous_vector_norm(e2)
                local_y = cross3(local_z, oriented_axis[used])
                local_y = local_y / np_sqrt(dot3(local_y, local_y))[:, None]
                second[target] = local_y
                third[target] = local_z
                heuristic[target] = False
        fallback_rows = np_nonzero(heuristic)
        if fallback_rows.size:
            fallback_axis = axis[fallback_rows]
            trial = np_empty((fallback_rows.size, 3), dtype=float64_dtype)
            trial[:] = trial_y
            # abs(float(np.dot(e1, trial))) > 0.95
            switched = np_abs(dot3(fallback_axis, trial)) > 0.95
            trial[switched] = trial_z
            # e2 = trial - np.dot(trial, e1) * e1; e2 /= norm(e2)
            local_y = trial - dot3(trial, fallback_axis)[:, None] * fallback_axis
            local_y = local_y / np_sqrt(dot3(local_y, local_y))[:, None]
            # e3 = _cross3(e1, e2); e3 /= norm(e3)
            local_z = cross3(fallback_axis, local_y)
            local_z = local_z / np_sqrt(dot3(local_z, local_z))[:, None]
            second[fallback_rows] = local_y
            third[fallback_rows] = local_z
        # R = _frame_columns(e1, e2, e3); T blocks hold R.T
        rotation_t = np_empty((count, 3, 3), dtype=float64_dtype)
        rotation_t[:, 0, :] = axis
        rotation_t[:, 1, :] = second
        rotation_t[:, 2, :] = third
        transform = np_zeros((count, 12, 12), dtype=float64_dtype)
        for base in (0, 3, 6, 9):
            transform[:, base : base + 3, base : base + 3] = rotation_t
        # T @ self._linear_rigid_arm_map()
        transform = np_matmul(transform, rigid_arm)
        return lengths, transform

    def lengths_admitted(prepared: Any) -> bool:
        """Every member is at least ``_SMALL`` long (the scalar methods' test).

        Checked before any material reduction: for a shorter member the
        scalar ``compute_stiffness_matrix`` returns zeros (and
        ``compute_stresses`` an empty record) without reducing the material.
        """

        coordinates = prepared[0]
        difference = coordinates[:, 1, :] - coordinates[:, 0, :]
        lengths = np_sqrt(dot3(difference, difference))
        return exact_bool(np_all(lengths >= small))

    def material_constants(plan: dict[str, Any]) -> Any:
        """``E, G12, G13, GJ`` of every element, in element order.

        The identical call of ``_local_linear_stiffness`` and
        ``compute_stresses``: ``_beam_material_properties(material,
        self.cross_section)`` through the elements-module global, once per
        element, with the element's own ``cross_section`` dict.  Library
        results are not memoised.  Runs in the caller's floating-point error
        state, as the scalar methods do.
        """

        reduce = exact_dict_get(elements_namespace, "_beam_material_properties")
        if reduce is not beam_material_properties:
            return None
        materials = plan["materials"]
        reduced_values = []
        append = reduced_values.append
        for namespace, material_position in exact_zip(
            plan["namespaces"], plan["material_index"]
        ):
            reduced = reduce(materials[material_position], namespace["cross_section"])
            if exact_type(reduced) is not exact_tuple or exact_len(reduced) != 4:
                return None
            append(reduced)
        if not exact_all(
            exact_map(
                exact_is,
                exact_map(exact_type, exact_chain.from_iterable(reduced_values)),
                exact_repeat(exact_float),
            )
        ):
            return None
        table = np_array(reduced_values, dtype=float64_dtype)
        # Columns: E, G12, G13 and GJ = G12 * float(cross_section.get("J", 1e-8)).
        return exact_tuple(table[:, index].copy() for index in exact_range(4))

    def section_arrays(plan: dict[str, Any]) -> Any:
        return (
            np_array(plan["areas"], dtype=float64_dtype),
            np_array(plan["inertia_y"], dtype=float64_dtype),
            np_array(plan["inertia_z"], dtype=float64_dtype),
            np_array(plan["shear_y"], dtype=float64_dtype),
            np_array(plan["shear_z"], dtype=float64_dtype),
        )

    def stiffness_kernel(plan: dict[str, Any], prepared: Any, constants: Any) -> Any:
        """``BeamElement.compute_stiffness_matrix`` for every element."""

        sections = section_arrays(plan)
        coordinates, oriented, orientation = prepared
        count = coordinates.shape[0]
        stiffness = np_empty((count, 12, 12), dtype=float64_dtype)
        for begin, end in chunks(count):
            part = stiffness_chunk(
                coordinates[begin:end],
                oriented[begin:end],
                orientation[begin:end],
                exact_tuple(values[begin:end] for values in constants),
                exact_tuple(values[begin:end] for values in sections),
            )
            if part is None:
                return None
            stiffness[begin:end] = part
        return stiffness

    def stiffness_chunk(
        coordinates: Any,
        oriented: Any,
        orientation: Any,
        constants: Any,
        sections: Any,
    ) -> Any:
        framed = frames(coordinates, oriented, orientation)
        if framed is None:
            return None
        lengths, transform = framed
        modulus, g12, g13, torsion = constants
        area, iy, iz, ky, kz = sections
        length_values = lengths.tolist()
        # L**2 and L**3 on the Python float L = float(length).
        length_2 = np_array([value**2 for value in length_values], dtype=float64_dtype)
        length_3 = np_array([value**3 for value in length_values], dtype=float64_dtype)
        count = lengths.size
        L = lengths
        EA = modulus * area
        EIy = modulus * iy
        EIz = modulus * iz
        K = np_zeros((count, 12, 12), dtype=float64_dtype)
        K[:, 0, 0] = K[:, 6, 6] = EA / L
        K[:, 0, 6] = K[:, 6, 0] = -EA / L
        phi_w = 12.0 * EIy / scalar_max(g13 * area * kz * length_2, small)
        K[:, 2, 2] = K[:, 8, 8] = 12.0 * EIy / (length_3 * (1.0 + phi_w))
        K[:, 2, 8] = K[:, 8, 2] = -K[:, 2, 2]
        K[:, 2, 4] = K[:, 4, 2] = -6.0 * EIy / (length_2 * (1.0 + phi_w))
        K[:, 2, 10] = K[:, 10, 2] = K[:, 2, 4]
        K[:, 8, 4] = K[:, 4, 8] = -K[:, 2, 4]
        K[:, 8, 10] = K[:, 10, 8] = -K[:, 2, 4]
        K[:, 4, 4] = K[:, 10, 10] = (4.0 + phi_w) * EIy / (L * (1.0 + phi_w))
        K[:, 4, 10] = K[:, 10, 4] = (2.0 - phi_w) * EIy / (L * (1.0 + phi_w))
        phi_v = 12.0 * EIz / scalar_max(g12 * area * ky * length_2, small)
        K[:, 1, 1] = K[:, 7, 7] = 12.0 * EIz / (length_3 * (1.0 + phi_v))
        K[:, 1, 7] = K[:, 7, 1] = -K[:, 1, 1]
        K[:, 1, 5] = K[:, 5, 1] = 6.0 * EIz / (length_2 * (1.0 + phi_v))
        K[:, 1, 11] = K[:, 11, 1] = 6.0 * EIz / (length_2 * (1.0 + phi_v))
        K[:, 7, 5] = K[:, 5, 7] = -6.0 * EIz / (length_2 * (1.0 + phi_v))
        K[:, 7, 11] = K[:, 11, 7] = -6.0 * EIz / (length_2 * (1.0 + phi_v))
        K[:, 5, 5] = K[:, 11, 11] = (4.0 + phi_v) * EIz / (L * (1.0 + phi_v))
        K[:, 5, 11] = K[:, 11, 5] = (2.0 - phi_v) * EIz / (L * (1.0 + phi_v))
        K[:, 3, 3] = K[:, 9, 9] = torsion / L
        K[:, 3, 9] = K[:, 9, 3] = -torsion / L
        # K_global = T.T @ K @ T
        stiffness = np_matmul(np_matmul(transform.transpose(0, 2, 1), K), transform)
        # _check_element_matrix_shape and _relative_symmetry_error <= 1e-8.
        if not exact_bool(np_all(np_isfinite(stiffness))):
            return None
        flat = stiffness.reshape(count, 144)
        asymmetric = (stiffness - stiffness.transpose(0, 2, 1)).reshape(count, 144)
        numerator = np_sqrt(dot3(asymmetric, asymmetric))
        denominator = scalar_max(np_sqrt(dot3(flat, flat)), 1.0)
        if exact_bool(np_any(numerator / denominator > 1.0e-8)):
            return None
        return stiffness

    def recovery_kernel(
        plan: dict[str, Any], prepared: Any, constants: Any, local_displacements: Any
    ) -> Any:
        """Non-generalized branch of ``BeamElement.compute_stresses``."""

        sections = section_arrays(plan)
        coordinates, oriented, orientation = prepared
        count = coordinates.shape[0]
        # None -> 0.0 selects the estimate exactly as ``None or <= 0.0`` does.
        fiber = exact_tuple(
            np_array(
                [0.0 if value is None else exact_float(value) for value in plan[name]],
                dtype=float64_dtype,
            )
            for name in ("fiber_y", "fiber_z", "torsion_moduli")
        )
        columns = exact_tuple(
            np_empty(count, dtype=float64_dtype) for _index in exact_range(7)
        )
        for begin, end in chunks(count):
            part = recovery_chunk(
                coordinates[begin:end],
                oriented[begin:end],
                orientation[begin:end],
                exact_tuple(values[begin:end] for values in constants),
                exact_tuple(values[begin:end] for values in sections),
                exact_tuple(values[begin:end] for values in fiber),
                local_displacements[begin:end],
            )
            if part is None:
                return None
            for column, values in exact_zip(columns, part):
                column[begin:end] = values
        return columns

    def recovery_chunk(
        coordinates: Any,
        oriented: Any,
        orientation: Any,
        constants: Any,
        sections: Any,
        fiber: Any,
        local_displacements: Any,
    ) -> Any:
        framed = frames(coordinates, oriented, orientation)
        if framed is None:
            return None
        lengths, transform = framed
        E, G12, G13, GJ = constants
        area, iy, iz, ky, kz = sections
        c_y, c_z, wt = fiber
        L = lengths
        # u_local = T @ u
        u_local = np_matmul(transform, local_displacements[:, :, None])[:, :, 0]
        u1, v1, w1, rx1, ry1, rz1 = (u_local[:, index] for index in exact_range(6))
        u2, v2, w2, rx2, ry2, rz2 = (u_local[:, index] for index in exact_range(6, 12))
        sigma_axial = E * (u2 - u1) / L
        kappa_z = (rz2 - rz1) / L
        kappa_y = (ry2 - ry1) / L
        M_z = E * iz * kappa_z
        M_y = E * iy * kappa_y
        # _fiber_distances
        safe_area = scalar_max(area, small)
        estimate_y = np_sqrt(np_abs(iz) / safe_area) * 2.0
        estimate_z = np_sqrt(np_abs(iy) / safe_area) * 2.0
        c_y = scalar_max(np_where(c_y <= 0.0, estimate_y, c_y), small)
        c_z = scalar_max(np_where(c_z <= 0.0, estimate_z, c_z), small)
        # _torsion_section_modulus
        wt = scalar_max(np_where(wt <= 0.0, 2.0 * area, wt), small)
        sigma_bending_y = M_y * c_z / scalar_max(iy, small)
        sigma_bending_z = M_z * c_y / scalar_max(iz, small)
        gamma_xy = (v2 - v1) / L - 0.5 * (rz1 + rz2)
        gamma_xz = (w2 - w1) / L + 0.5 * (ry1 + ry2)
        tau_y = G12 * ky * gamma_xy
        tau_z = G13 * kz * gamma_xz
        tau_torsion = GJ * (rx2 - rx1) / L / wt
        sigma_x = sigma_axial + sigma_bending_y + sigma_bending_z
        # ``**2`` on the np.float64 scalars of the scalar code.
        squared = [
            np_array([value**2 for value in values], dtype=float64_dtype)
            for values in (sigma_x, tau_y, tau_z, tau_torsion)
        ]
        von_mises = np_sqrt(
            squared[0] + 3.0 * (squared[1] + squared[2] + squared[3])
        )
        columns = (
            sigma_axial,
            sigma_bending_y,
            sigma_bending_z,
            tau_y,
            tau_z,
            tau_torsion,
            von_mises,
        )
        for column in columns:
            if not exact_bool(np_all(np_isfinite(column))):
                return None
        return columns

    def stress_records(columns: Any) -> list[dict[str, Any]]:
        records = []
        append = records.append
        for (
            sigma_axial,
            sigma_bending_y,
            sigma_bending_z,
            tau_y,
            tau_z,
            tau_torsion,
            von_mises,
        ) in exact_zip(*(exact_list(column) for column in columns)):
            append(
                {
                    "axial_stress": sigma_axial,
                    "bending_stress_y": sigma_bending_y,
                    "bending_stress_z": sigma_bending_z,
                    "shear_stress_y": tau_y,
                    "shear_stress_z": tau_z,
                    "torsional_stress": tau_torsion,
                    "von_mises": von_mises,
                    "equivalent_stress": von_mises,
                    "equivalent_stress_measure": "von_mises",
                }
            )
        return records

    beam_node_coordinates = resolve_class_attribute(beam_type, "get_node_coordinates")
    beam_frame_and_transform = resolve_class_attribute(
        beam_type, "_beam_frame_and_transform"
    )
    beam_local_linear_stiffness = resolve_class_attribute(
        beam_type, "_local_linear_stiffness"
    )

    def stiffness_oracle(element: Any, mesh: Any, material: Any) -> Any:
        """``BeamElement.compute_stiffness_matrix`` without its store.

        The verified original methods (``authority_is_exact``; an admitted
        element shadows none of them), called with the operands and in the
        order of ``compute_stiffness_matrix``, so the result is the scalar
        method's value while no ``_stiffness_matrix`` is written before the
        batch has decided to succeed.  Its zero-matrix branch for a short
        member cannot be reached (``lengths_admitted``); a ``ValueError``
        propagates and declines the batch.
        """

        coordinates = beam_node_coordinates(element, mesh)
        length, transform = beam_frame_and_transform(element, coordinates)
        local = beam_local_linear_stiffness(element, length, material)
        return transform.T @ local @ transform

    def same_matrix(oracle: Any, batched: Any) -> bool:
        return (
            exact_type(oracle) is ndarray_type
            and oracle.dtype == float64_dtype
            and oracle.shape == (12, 12)
            and ndarray_tobytes(oracle, order="C") == ndarray_tobytes(batched, order="C")
        )

    def same_record(oracle: Any, batched: dict[str, Any]) -> bool:
        if exact_type(oracle) is not exact_dict or exact_list(oracle) != exact_list(batched):
            return False
        for key, value in exact_dict_items(batched):
            expected = oracle[key]
            if exact_type(expected) is not exact_type(value):
                return False
            if exact_type(value) is exact_str:
                if expected != value:
                    return False
            elif ndarray_tobytes(np_array(expected)) != ndarray_tobytes(np_array(value)):
                return False
        return oracle["equivalent_stress"] is oracle["von_mises"]

    # ------------------------------------------------------------------
    # Process-wide state and the one-time probe.
    # ------------------------------------------------------------------

    state = {"probe": None, "disabled_reason": None, "failed": None}
    counters = {"stiffness_batches": 0, "recovery_batches": 0, "fallbacks": 0}
    last_fallback: list[Any] = [None]

    def fallback(reason: str) -> None:
        counters["fallbacks"] += 1
        last_fallback[0] = reason

    def disable(reason: str) -> None:
        state["failed"] = reason
        state["disabled_reason"] = reason
        fallback(reason)

    class _ProbeMesh:
        def __init__(self, nodes: dict[int, Any]) -> None:
            self.nodes = nodes

        def get_node(self, node_id: int) -> Any:
            return self.nodes.get(node_id)

    def probe_inputs() -> Any:
        materials = {
            "steel": material_type("probe-steel", 2.1e11, 0.3, 7850.0),
            "alu": material_type("probe-alu", 7.0e10, 0.33, 2700.0, 1.0e8),
        }
        generator = np.random.default_rng(20261124)
        coordinates = [
            ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
            ((0.0, 2.0, 0.0), (0.0, -1.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 3.0)),
            ((-0.0, -0.0, -0.0), (-0.0, 2.0, -0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0)),
            ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0)),
            ((-3.0, 0.5, 2.0), (-3.0, 0.5, -0.0)),
            ((0.25, -0.0, 0.0), (0.3, 1.0, 0.0)),
            ((5, 0, 0), (5, 0, 2)),
        ]
        for _ in exact_range(14):
            start = generator.normal(size=3) * 3.0
            end = start + generator.normal(size=3) * generator.uniform(0.05, 4.0)
            coordinates.append((exact_tuple(start.tolist()), exact_tuple(end.tolist())))
        base = {"area": 0.02, "Iy": 1.6667e-5, "Iz": 6.6667e-5, "J": 2.29e-5}
        sections = [
            dict(base),
            dict(base),
            dict(base, orientation=[1.0, 0.0, 0.0]),
            dict(base, orientation=[0.0, 0.0, 1.0]),
            dict(base, orientation=[0.0, 0.0, 2.0]),
            dict(base, orientation=[1.0, 1.0e-7, 0.0]),
            dict(base, orientation=[1.0, 1.0e-5, 0.0]),
            dict(base, web_direction=(0.0, 1.0, 0.0), c_y=0.2, torsion_modulus=1.0e-3),
            {key: float64_type(value) for key, value in base.items()},
            {"area": 1, "Iy": 2, "Iz": 3, "J": 1, "shear_factor_y": 1, "c_y": 0, "c_z": -1.0},
        ]
        for index in exact_range(14):
            section = dict(base)
            section["area"] = 0.01 + 0.001 * index
            if index % 2 == 0:
                section["orientation"] = generator.normal(size=3).tolist()
            if index % 3 == 0:
                section["shear_factor_z"] = 0.5
                section["c_z"] = 0.15
            if index % 5 == 0:
                section.pop("J")
                section["torsion_modulus"] = 0.0
            sections.append(section)
        nodes: dict[int, Any] = {}
        items = []
        names = []
        for index, ((start, end), section) in exact_enumerate(
            exact_zip(coordinates, sections)
        ):
            first = 2 * index + 1
            second = first + 1
            nodes[first] = node_type(first, *start, dofs=exact_list(exact_range(12 * index, 12 * index + 6)))
            nodes[second] = node_type(second, *end, dofs=exact_list(exact_range(12 * index + 6, 12 * index + 12)))
            name = "alu" if index % 4 == 1 else "steel"
            names.append(name)
            items.append((index + 1, beam_type(index + 1, [first, second], name, section)))
        displacements = generator.normal(size=(exact_len(items), 12)) * 1.0e-3
        return materials, nodes, exact_tuple(items), names, displacements

    def run_probe() -> bool:
        try:
            materials, nodes, items, names, displacements = probe_inputs()
            plan = gather(
                items,
                nodes,
                lambda name: exact_dict_get(materials, name, missing),
                12 * exact_len(items),
            )
            if plan is None:
                return False
            prepared = prepared_arrays(plan)
            with np_errstate(all="raise"):
                if not lengths_admitted(prepared):
                    return False
            constants = material_constants(plan)
            if constants is None:
                return False
            with np_errstate(all="raise"):
                stiffness = stiffness_kernel(plan, prepared, constants)
                columns = recovery_kernel(plan, prepared, constants, displacements)
            if stiffness is None or columns is None:
                return False
            records = stress_records(columns)
            mesh = _ProbeMesh(nodes)
            for index, (_element_id, element) in exact_enumerate(items):
                material = materials[names[index]]
                if not same_matrix(
                    element.compute_stiffness_matrix(mesh, material),
                    stiffness[index],
                ):
                    return False
                if not same_record(
                    element.compute_stresses(mesh, displacements[index].copy(), material),
                    records[index],
                ):
                    return False
            return True
        except exact_exception:
            return False

    def ready() -> bool:
        if state["disabled_reason"] is not None:
            return False
        if state["probe"] is None:
            state["probe"] = run_probe()
            if not state["probe"]:
                disable("process probe mismatch")
                return False
        return exact_bool(state["probe"])

    # ------------------------------------------------------------------
    # Entry points.  Order in every call: authority (pure reads), admission
    # (pure reads), then calls and computation; every decline returns
    # ``None`` before any side effect of the batch.
    # ------------------------------------------------------------------

    def model_inputs(model: Any, mesh: Any) -> Any:
        """Exact model/mesh namespaces, or ``None``.  Pure reads only.

        Call only after :func:`authority_is_exact`.
        """

        if exact_type(model) is not model_type or exact_type(mesh) is not mesh_type:
            return None
        model_namespace = exact_object_getattribute(model, "__dict__")
        mesh_namespace = exact_object_getattribute(mesh, "__dict__")
        if (
            exact_type(model_namespace) is not exact_dict
            or exact_type(mesh_namespace) is not exact_dict
            or not all_str_keys(model_namespace)
            or not all_str_keys(mesh_namespace)
            or "get_material" in model_namespace
            or "get_node" in mesh_namespace
            or "num_nodes" in mesh_namespace
            or exact_dict_get(model_namespace, "mesh") is not mesh
        ):
            return None
        materials = exact_dict_get(model_namespace, "materials")
        current = exact_dict_get(model_namespace, "current_material", missing)
        if current is missing:
            current = mappingproxy_get(class_namespace(model_type), "current_material")
        nodes = exact_dict_get(mesh_namespace, "nodes")
        elements = exact_dict_get(mesh_namespace, "elements")
        # Caller-held mesh caches: the scalar sparsity and topology-signature
        # kernels call their mapping protocol.  Only ``None`` (absent) or an
        # exact dict is admitted; anything else takes the scalar loop before
        # the batch touches it.
        sparsity_cache = exact_dict_get(mesh_namespace, "_sparsity_cache")
        signature_cache = exact_dict_get(mesh_namespace, "_topology_signature_cache")
        if (
            sparsity_cache is not None
            and exact_type(sparsity_cache) is not exact_dict
            or signature_cache is not None
            and exact_type(signature_cache) is not exact_dict
        ):
            return None
        if (
            exact_type(materials) is not exact_dict
            or not all_str_keys(materials)
            or exact_type(current) is not exact_str
            or not plain_mapping(nodes, all_int_keys)
            or not plain_mapping(elements, all_int_keys)
        ):
            return None
        return nodes, elements

    def material_resolver(model: Any) -> Callable[[str], Any]:
        # ``FEModel.get_material`` (verified original; it reads only the
        # exact ``materials`` dict and ``current_material``), so one call per
        # distinct name returns what the scalar path's per-element calls do.
        def resolve(name: str) -> Any:
            try:
                return model_get_material(model, name)
            except exact_exception:
                return missing

        return resolve

    def all_exact_beams(element_items: Any) -> bool:
        return exact_all(
            exact_type(item) is exact_tuple
            and exact_len(item) == 2
            and exact_type(item[1]) is beam_type
            for item in element_items
        )

    def stiffness_candidate(model: Any, mesh: Any, element_items: Any) -> bool:
        """Cheap pure pre-check deciding whether to open a guard bracket."""

        return (
            state["disabled_reason"] is None
            and exact_type(element_items) is exact_tuple
            and exact_len(element_items) >= MIN_BATCH_ELEMENTS
            and exact_type(model) is model_type
            and exact_type(mesh) is mesh_type
            and all_exact_beams(element_items)
        )

    def stiffness_batch(
        model: Any,
        mesh: Any,
        element_items: Any,
        total_dofs: Any,
        classifiers: Any,
        prepare_sparsity: Callable[[], Any],
    ) -> Any:
        """Batch stiffness for an all-exact-beam model, or ``None``.

        Called inside the caller's guard bracket.  Every decline happens
        before any write and before any caller-held cache is touched.
        ``prepare_sparsity`` is the scalar path's pre-loop sparsity call; it
        runs exactly once, after the batch has decided to succeed and before
        any element side effect, and its exceptions propagate exactly as in
        the scalar path.  On success every element's ``_stiffness_matrix``
        holds an owned copy of its matrix, written in element order after
        everything else succeeded.
        Returns ``(matrices, element_ids, seconds, sparsity)``.
        """

        if not authority_is_exact():
            fallback("class or module authority changed")
            return None
        inputs = model_inputs(model, mesh)
        if inputs is None:
            fallback("model inputs outside the exact surface")
            return None
        plan = gather(element_items, inputs[0], material_resolver(model), total_dofs)
        if plan is None:
            fallback("element inputs outside the exact surface")
            return None
        try:
            classified = classification_is_noop(classifiers, element_items)
        except exact_exception:
            classified = False
        if not classified:
            fallback("scalar classification would not leave the beams alone")
            return None
        if not ready():
            return None
        started = clock()
        prepared = prepared_arrays(plan)
        try:
            with np_errstate(all="raise"):
                admitted = lengths_admitted(prepared)
        except exact_exception:
            admitted = False
        if not admitted:
            fallback("member shorter than the scalar length threshold")
            return None
        try:
            constants = material_constants(plan)
        except exact_exception:
            constants = None
        if constants is None:
            fallback("material reduction declined")
            return None
        try:
            with np_errstate(all="raise"):
                stiffness = stiffness_kernel(plan, prepared, constants)
        except exact_exception:
            stiffness = None
        if stiffness is None:
            fallback("stiffness batch declined")
            return None
        count = stiffness.shape[0]
        materials = plan["materials"]
        material_index = plan["material_index"]
        elements = plan["elements"]
        namespaces = plan["namespaces"]
        # Endpoint checks without the scalar method's store: nothing is
        # written before the batch has decided to succeed.
        for position in (0, count - 1):
            try:
                matched = same_matrix(
                    stiffness_oracle(
                        elements[position],
                        mesh,
                        materials[material_index[position]],
                    ),
                    stiffness[position],
                )
            except exact_exception:
                matched = None
            if not matched:
                if matched is None:
                    fallback("stiffness spot check raised")
                else:
                    disable("stiffness spot check mismatch")
                return None
        # Decided.  The scalar path's pre-loop sparsity step runs once, here,
        # before any element side effect; it may raise exactly as there.
        sparsity = prepare_sparsity()
        for namespace, matrix in exact_zip(namespaces, stiffness):
            namespace["_stiffness_matrix"] = matrix.copy()
        counters["stiffness_batches"] += 1
        return stiffness, plan["element_ids"], clock() - started, sparsity

    def recovery_candidate(model: Any, mesh: Any, selected: Any) -> Any:
        """``(first_id, first_element, first_material, last_id)``, or ``None``.

        Authority first, then pure reads; the ids name the first and last
        per-element observation guards of the scalar loop that the bracket
        replaces, and the first element (with its material) is the one the
        scalar loop observes first.
        """

        if state["disabled_reason"] is not None or not authority_is_exact():
            return None
        inputs = model_inputs(model, mesh)
        if inputs is None:
            return None
        elements = inputs[1]
        if exact_dict_len(elements) < MIN_BATCH_ELEMENTS or not exact_all(
            exact_type(element) is beam_type
            for element in exact_dict_values(elements)
        ):
            return None
        if selected is None:
            element_ids = exact_list(exact_dict_keys(elements))
        elif exact_type(selected) is exact_set and exact_all(
            exact_type(value) is exact_int for value in selected
        ):
            element_ids = [
                element_id
                for element_id in exact_dict_keys(elements)
                if element_id in selected
            ]
        else:
            return None
        if not element_ids:
            return None
        first_id = element_ids[0]
        first_element = exact_dict_get(elements, first_id)
        # The first element's material, obtained as the scalar loop's first
        # iteration obtains it (``model.get_material(element.material_name)``,
        # the verified original), but only for an exact ``str`` name read
        # from an exact instance namespace: a hostile name declines here,
        # before anything hashes or compares it.
        namespace = exact_object_getattribute(first_element, "__dict__")
        if exact_type(namespace) is not exact_dict or not all_str_keys(namespace):
            return None
        material_name = exact_dict_get(namespace, "material_name", missing)
        if exact_type(material_name) is not exact_str:
            return None
        first_material = material_resolver(model)(material_name)
        if first_material is missing:
            return None
        return first_id, first_element, first_material, element_ids[-1]

    def recovery_batch(
        model: Any,
        mesh: Any,
        displacements: Any,
        selected: Any,
    ) -> Any:
        """``(records, last_element, last_material)``, or ``None``.

        ``records`` maps every selected element id to its stress record in
        mesh order; the last element and its material are the arguments of
        the scalar loop's last observation guard.
        """

        if not authority_is_exact():
            fallback("class or module authority changed")
            return None
        inputs = model_inputs(model, mesh)
        if inputs is None:
            fallback("model inputs outside the exact surface")
            return None
        nodes, element_mapping = inputs
        if (
            exact_type(displacements) is not ndarray_type
            or displacements.dtype != float64_dtype
            or displacements.ndim != 1
        ):
            fallback("displacements outside the exact surface")
            return None
        if selected is None:
            element_items = exact_tuple(exact_dict_items(element_mapping))
        else:
            if exact_type(selected) is not exact_set or not exact_all(
                exact_type(value) is exact_int for value in selected
            ):
                fallback("selection outside the exact surface")
                return None
            element_items = exact_tuple(
                item
                for item in exact_dict_items(element_mapping)
                if item[0] in selected
            )
        if not element_items:
            fallback("empty selection")
            return None
        # Every DOF must index the displacement vector (the scalar loop skips
        # an element otherwise); a declined element takes the scalar loop.
        plan = gather(
            element_items,
            nodes,
            material_resolver(model),
            exact_int(displacements.size),
        )
        if plan is None:
            fallback("element inputs outside the exact surface")
            return None
        if not ready():
            return None
        node_dofs = plan["node_dofs"]
        dof_rows = np_array(
            [node_dofs[first] + node_dofs[second] for first, second in plan["pair_index"]],
            dtype=intp_dtype,
        )
        local_displacements = displacements[dof_rows]
        prepared = prepared_arrays(plan)
        try:
            with np_errstate(all="raise"):
                admitted = lengths_admitted(prepared)
        except exact_exception:
            admitted = False
        if not admitted:
            fallback("member shorter than the scalar length threshold")
            return None
        try:
            constants = material_constants(plan)
        except exact_exception:
            constants = None
        if constants is None:
            fallback("material reduction declined")
            return None
        try:
            with np_errstate(all="raise"):
                columns = recovery_kernel(plan, prepared, constants, local_displacements)
        except exact_exception:
            columns = None
        if columns is None:
            fallback("recovery batch declined")
            return None
        records = stress_records(columns)
        count = exact_len(records)
        materials = plan["materials"]
        material_index = plan["material_index"]
        elements = plan["elements"]
        for position in (0, count - 1):
            try:
                oracle = elements[position].compute_stresses(
                    mesh,
                    local_displacements[position].copy(),
                    materials[material_index[position]],
                )
            except exact_exception:
                fallback("recovery spot check raised")
                return None
            if not same_record(oracle, records[position]):
                disable("recovery spot check mismatch")
                return None
        counters["recovery_batches"] += 1
        return (
            exact_dict(exact_zip(plan["element_ids"], records)),
            elements[count - 1],
            materials[material_index[count - 1]],
        )

    def status() -> dict[str, Any]:
        return {
            "probe": state["probe"],
            "disabled_reason": state["disabled_reason"],
            "last_fallback": last_fallback[0],
            **counters,
        }

    def disable_for_process(reason: str = "disabled by caller") -> None:
        """Route every later call to the scalar loops (tests, diagnostics)."""

        state["disabled_reason"] = exact_str(reason)

    def enable_for_process() -> None:
        """Re-enable after :func:`disable_for_process`.

        A failed probe or spot check keeps the batch disabled for the process.
        """

        if state["failed"] is None:
            state["disabled_reason"] = None

    return (
        stiffness_candidate,
        stiffness_batch,
        recovery_candidate,
        recovery_batch,
        status,
        disable_for_process,
        enable_for_process,
    )


(
    exact_beam_stiffness_candidate,
    exact_beam_stiffness_batch,
    exact_beam_recovery_candidate,
    exact_beam_recovery_batch,
    exact_beam_batch_status,
    disable_exact_beam_batch,
    enable_exact_beam_batch,
) = _bind_exact_beam_batch()

__all__ = [
    "MIN_BATCH_ELEMENTS",
    "disable_exact_beam_batch",
    "enable_exact_beam_batch",
    "exact_beam_batch_status",
    "exact_beam_recovery_batch",
    "exact_beam_recovery_candidate",
    "exact_beam_stiffness_batch",
    "exact_beam_stiffness_candidate",
]
