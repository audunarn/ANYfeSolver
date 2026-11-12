"""Canonical load maps for the prepared static evaluator.

Compilation happens once against the live model during preparation.  The
compiled maps own every container and back every array with immutable buffers,
so evaluating a case never observes the model, the mesh, an element, a
material or the caller's ``LoadCase`` object again.

The maps reproduce the exact builtin ``LoadCase`` load families:

``nodal``
    Direct scatter of six-component nodal values at the node's DOFs.
``element``
    Scatter of element load vectors at the element DOF mapping, scaled by
    the frozen element activity load scale.
``pressure``
    Consistent element pressure kernels evaluated in the reference
    configuration (the linear-static treatment of both dead and follower
    pressure), including the strict-flat S3 V2 follower refusal.  Scalar
    and per-node tuple pressure values are supported where the element
    kernel admits them.
``acceleration``
    Consistent body loads ``M_element @ a`` plus structural point masses,
    with per-case added node masses applied at translational DOFs.

Every compiled family is verified against the live assembly oracle with
deterministic probe cases before preparation succeeds; any mismatch fails
the preparation closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import types

import numpy as np

if TYPE_CHECKING:
    from .fe_core import FEModel

SUPPORTED_LOAD_FAMILIES: Tuple[str, ...] = ("nodal", "element", "pressure", "acceleration")

# Hooks whose exact identity is required before an element may contribute to
# a compiled element/pressure/acceleration map.  A patched hook or a foreign
# subclass is refused with a typed reason instead of being probed.
_ELEMENT_AUTHORITY_HOOKS: Tuple[str, ...] = (
    "get_dof_mapping",
    "compute_stresses",
    "compute_mass_matrix",
    "get_node_coordinates",
    "compute_shape_functions",
    "sheet_area_orientation_sign",
    "compute_native_dead_pressure_load",
)


def _bind_exact_element_authority() -> Tuple[
    Dict[type, Dict[str, Any]],
    Dict[type, str],
]:
    """Bind the exact native element classes and their reviewed hooks once."""

    from .elements import (
        BeamElement,
        CoupledBeamShellElement,
        QuadraticBeamElement,
        ShellElement,
    )
    from .e4_pl_element import QualifiedE4PLShellElement
    from .e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from .e4_pl_s3_v2d_element import (
        FORMULATION_ID as S3_V2D_FORMULATION_ID,
        NativeParityE4PLS3V2DShellElement,
    )

    hooks: Dict[type, Dict[str, Any]] = {}
    formulation_ids: Dict[type, str] = {}
    for cls in (
        ShellElement,
        BeamElement,
        CoupledBeamShellElement,
        QuadraticBeamElement,
        QualifiedE4PLShellElement,
        QualifiedE4PLS3ShellElement,
        NativeParityE4PLS3V2DShellElement,
    ):
        hooks[cls] = {name: getattr(cls, name, None) for name in _ELEMENT_AUTHORITY_HOOKS}
    formulation_ids[NativeParityE4PLS3V2DShellElement] = str(S3_V2D_FORMULATION_ID)
    return hooks, formulation_ids


_EXACT_ELEMENT_HOOKS, _EXACT_ELEMENT_FORMULATION_IDS = _bind_exact_element_authority()

# Capture executable metadata and referenced native helper bindings once, before
# user preparation/probes. Function identity alone misses in-place __code__ edits.
from .boundary import _callable_authority, _callable_authority_matches
from . import boundary as _boundary_module
from .static_batch import _class_state
from anymaterial import IsotropicMaterial, OrthotropicMaterial, Hill48Yield, DNVC208MaterialCurve
from .beam_sections import GeneralizedBeamSection
from .shell_sections import GeneralizedShellSection
from .fe_core import FEModel, FEMesh, Node, DOFManager, _QualifiedStateMapping

# M31c/M31d: run-time checks shared by the identical-result shortcuts (M30/M31). Trust
# root: the code as loaded when these modules are imported (the model of the response
# authority, captured at import). M31e (owner decision): replacing library or module
# functions after import (monkeypatching NumPy, SciPy, copy, or ANY* module functions,
# or their code) is outside the supported contract; the binding checks below turn a
# shortcut off when they see such a replacement, as defence in depth, not as a promise
# of call-for-call equality with the original code. M31f: the floating-point
# environment of the contract is the rounding mode and the denormal handling
# (flush-to-zero, denormals-are-zero); hardware floating-point exception traps stay
# masked, as NumPy reports floating-point conditions through status flags and its
# error state (the probes below themselves round, so they would trap).
_NOT_ORIGINAL = object()
_DEFAULT_ERROR_ITEMS = (('divide', 'warn'), ('invalid', 'warn'), ('over', 'warn'), ('under', 'ignore'))
_FP_ONE, _FP_THREE_QUARTER_ULP = np.float64(1.0), np.float64(3.0 * 2.0 ** -54)
_FP_SMALLEST_NORMAL, _FP_SMALLEST_SUBNORMAL = np.float64(2.0 ** -1022), np.float64(2.0 ** -1074)
_FP_HALF = np.float64(0.5)


def _binding(owner, name):
    if isinstance(owner, types.ModuleType):
        return vars(owner).get(name)
    return getattr(owner, name, None)


def _capture_originals(*groups):
    """(owner, name, object) for each binding, as loaded; an object that is not the
    library's own (package of its module) is recorded as never matching."""
    made = []
    for owner, names in groups:
        package = (getattr(owner, '__module__', None) if not isinstance(owner, types.ModuleType)
                   else owner.__name__).split('.')[0]
        for name in names:
            value = _binding(owner, name)
            module = getattr(value, '__module__', None)
            genuine = type(value) is np.ufunc or (type(module) is str and module.split('.')[0] == package)
            made.append((owner, name, value if genuine else _NOT_ORIGINAL))
    return tuple(made)


def _originals_unchanged(captured):
    """Every captured binding is still the object loaded at import (no code runs)."""
    for owner, name, value in captured:
        if _binding(owner, name) is not value:
            return False
    return True


def _default_error_state():
    """NumPy's default error state in this thread (compared with an immutable copy)."""
    return tuple(sorted(np.geterr().items())) == _DEFAULT_ERROR_ITEMS


def _standard_float_environment():
    """Round to nearest, no flush-to-zero and no denormals-are-zero in this thread,
    checked at run time (each probe is exact or differs between the modes)."""
    with np.errstate(all='ignore'):
        up = _FP_ONE + _FP_THREE_QUARTER_ULP           # 1 + 2**-52 when rounding to nearest or up
        down = -_FP_ONE - _FP_THREE_QUARTER_ULP        # -(1 + 2**-52) when rounding to nearest or down
        halved = _FP_SMALLEST_NORMAL * _FP_HALF        # 2**-1023 unless flushed to zero
        kept = _FP_SMALLEST_SUBNORMAL * _FP_ONE        # 2**-1074 unless read as zero
    return bool(up != _FP_ONE and down != -_FP_ONE and halved != 0.0 and kept != 0.0)


_NATIVE_DATA_CLASSES = (IsotropicMaterial, OrthotropicMaterial, Hill48Yield, DNVC208MaterialCurve,
                        GeneralizedBeamSection, GeneralizedShellSection)
_NATIVE_CLASS_FACTS = {cls: _class_state(cls) for cls in (*_EXACT_ELEMENT_HOOKS, *_NATIVE_DATA_CLASSES, FEModel, FEMesh, Node, DOFManager, _QualifiedStateMapping)}

from ._native_authority_dag import matches as _native_authority_matches, intern_facts as _intern_native_facts


def _capture_native_helpers():
    records = []
    seen = set()
    def visit(function):
        if type(function) is not types.FunctionType or id(function) in seen:
            return
        if not function.__module__.startswith(("anyfesolver", "anymaterial")):
            return
        seen.add(id(function))
        if function is _native_authority_matches:
            # A missing global is also a fact for the new matcher: injecting
            # id/type/all/etc must not shadow a builtin and alias memo keys.
            def names(code):
                return set(code.co_names).union(*(names(child) for child in code.co_consts if type(child) is types.CodeType))
            bindings = tuple((name,name in function.__globals__,function.__globals__.get(name))
                             for name in sorted(names(function.__code__)))
        else:
            bindings = tuple((name, function.__globals__[name])
                             for name in function.__code__.co_names if name in function.__globals__)
        records.append((function, _callable_authority(function), bindings))
        for binding in bindings:
            visit(binding[-1])
        for cell in function.__closure__ or ():
            try:
                visit(cell.cell_contents)
            except ValueError:
                pass
    # The optimized matcher is itself covered by the unchanged literal
    # predicate before it may consume any expected nodes.
    visit(_native_authority_matches)
    for helper_name in ("_load_model_access_is_builtin", "_load_routing_classes_are_untouched", "_numpy_error_routes_are_builtin", "_builtin_acceleration_inputs", "_exact_load_case_instance_namespace"):
        visit(getattr(_boundary_module, helper_name))
    for cls in _NATIVE_CLASS_FACTS:
        for base in cls.__mro__:
            for member in vars(base).values():
                if type(member) in (staticmethod, classmethod):
                    visit(member.__func__)
                elif type(member) is property:
                    for accessor in (member.fget, member.fset, member.fdel):
                        visit(accessor)
                else:
                    visit(member)
    return _intern_native_facts(tuple(records))

_NATIVE_HELPER_FACTS = _capture_native_helpers()
_PRESSURE_PROOF_IDENTITIES = tuple((name, getattr(_boundary_module, name)) for name in ("_load_model_access_is_builtin", "_load_routing_classes_are_untouched", "_numpy_error_routes_are_builtin", "_builtin_acceleration_inputs", "_exact_load_case_instance_namespace"))
_EXACT_HOOK_FACTS = {cls: {name: _callable_authority(value) for name, value in hooks.items()} for cls, hooks in _EXACT_ELEMENT_HOOKS.items()}

from . import _native_guard_dispatch as _native_guard_dispatch_module
from .static_batch import _callable_state as _native_literal_callable_state
def _native_cpp_literal_bindings():
    # This body is not called. The native plan captures these original global
    # observations and declines to the complete literal path if any changes.
    return _class_state,_callable_authority_matches,_native_authority_matches
_native_guard_dispatch_module.initialise(_NATIVE_CLASS_FACTS,_NATIVE_HELPER_FACTS,
    _native_authority_matches.__globals__['_MISSING'],
    (_class_state,_native_literal_callable_state,_callable_authority_matches,_native_authority_matches,_native_cpp_literal_bindings))
_native_cpp_function=_native_guard_dispatch_module.try_compare
def _native_cpp_names(code):
    return set(code.co_names).union(*(_native_cpp_names(child) for child in code.co_consts if type(child) is types.CodeType))
_NATIVE_CPP_BOOTSTRAP=(_native_cpp_function,_callable_authority(_native_cpp_function),
    tuple((name,name in _native_cpp_function.__globals__,_native_cpp_function.__globals__.get(name))
          for name in sorted(_native_cpp_names(_native_cpp_function.__code__))))

def require_native_source_authority():
    if any(getattr(_boundary_module, name, None) is not expected for name, expected in _PRESSURE_PROOF_IDENTITIES):
        raise PreparedStaticCapabilityError("native runtime guard binding authority changed")
    function,facts,bindings=_NATIVE_CPP_BOOTSTRAP
    namespace=function.__globals__
    cpp_valid=(_native_cpp_function is function and _native_guard_dispatch_module.try_compare is function
        and _callable_authority_matches(function,facts) and all(type(key) is str for key in namespace)
        and not any((name in namespace) is not present or (present and namespace[name] is not value)
                    for name,present,value in bindings))
    native_result=function(_NATIVE_CLASS_FACTS,_NATIVE_HELPER_FACTS) if cpp_valid else 2
    if native_result is True:
        return
    for cls, expected in _NATIVE_CLASS_FACTS.items():
        if _class_state(cls) != expected:
            raise PreparedStaticCapabilityError("native class executable authority changed", diagnostics={"stage": "native admission", "class": cls.__qualname__})
    if not cpp_valid or (type(native_result) is int and native_result==2):
        raise PreparedStaticCapabilityError('native helper executable authority changed',diagnostics={'stage':'native admission','function':'native guard binding'})
    matcher, facts, bindings = _NATIVE_HELPER_FACTS[0]
    if (_native_authority_matches is not matcher or not _callable_authority_matches(matcher,facts)
            or any((name in matcher.__globals__) is not present or
                   (present and matcher.__globals__[name] is not value) for name,present,value in bindings)):
        raise PreparedStaticCapabilityError("native helper executable authority changed", diagnostics={"stage":"native admission","function":matcher.__qualname__})
    # Fresh on every original safe-point/callback guard; never retain actual
    # function metadata or proof results across caller observations.
    memo = {}
    for function, authority, bindings in _NATIVE_HELPER_FACTS[1:]:
        if not matcher(function, authority, memo) or any(function.__globals__.get(name) is not value for name, value in bindings):
            raise PreparedStaticCapabilityError("native helper executable authority changed", diagnostics={"stage": "native admission", "function": function.__qualname__})


def _plain_numeric_state(value, depth=0):
    if depth > 16:
        return False
    if value is None or type(value) in (bool, int, float, str, bytes):
        return True
    if type(value) is np.ndarray:
        return value.dtype.kind in "biufc" and not value.dtype.hasobject
    if type(value) in set(np.sctypeDict.values()):
        return type(value).__module__.startswith("numpy") and value.dtype.kind in "biufc"
    if type(value) in (list, tuple):
        return all(_plain_numeric_state(item, depth+1) for item in value)
    if type(value) is dict:
        return all(type(key) in (str, int) and _plain_numeric_state(item, depth+1) for key, item in value.items())
    if type(value) in _NATIVE_DATA_CLASSES:
        import dataclasses
        return all(_plain_numeric_state(object.__getattribute__(value, field.name), depth+1) for field in dataclasses.fields(value))
    return False


def _instance_authority(value, *, allow_internal_callbacks=False):
    namespace = object.__getattribute__(value, "__dict__")
    if type(namespace) is not dict:
        return False
    for base in type(value).__mro__:
        for name, member in vars(base).items():
            if name in namespace and (callable(member) or type(member) in (property, staticmethod, classmethod)):
                return False
    return allow_internal_callbacks or not any(callable(item) for item in namespace.values())


def require_preparation_model_authority(model):
    if type(model) is not FEModel or not _instance_authority(model, allow_internal_callbacks=True):
        raise PreparedStaticCapabilityError("preparation requires exact native FEModel providers without instance shadows")
    mesh = object.__getattribute__(model, "mesh")
    if type(mesh) is not FEMesh or not _instance_authority(mesh, allow_internal_callbacks=True):
        raise PreparedStaticCapabilityError("preparation requires exact native FEMesh providers without instance shadows")
    manager = object.__getattribute__(mesh, "dof_manager")
    if type(manager) is not DOFManager or not _instance_authority(manager, allow_internal_callbacks=True):
        raise PreparedStaticCapabilityError("preparation requires an exact native DOF provider")
    for mapping in (model.materials, mesh.nodes, mesh.elements):
        if type(mapping) not in (dict, _QualifiedStateMapping):
            raise PreparedStaticCapabilityError("preparation requires native structural mappings")
    for node in mesh.nodes.values():
        if type(node) is not Node or not _instance_authority(node, allow_internal_callbacks=True):
            raise PreparedStaticCapabilityError("preparation requires exact native node providers")
    require_native_source_authority()


def native_material_section_authority(model, element):
    material = model.materials.get(element.material_name)
    if type(material) not in (IsotropicMaterial, OrthotropicMaterial):
        return False, "material is not an exact reviewed builtin elastic material"
    if not _instance_authority(material):
        return False, "material instance shadows reviewed native executable attributes"
    for key, value in vars(material).items():
        if not _plain_numeric_state(value):
            return False, "material data contains a custom numeric/container provider"
    for name, expected_type in (("generalized_section", GeneralizedBeamSection), ("shell_section", GeneralizedShellSection)):
        section = getattr(element, name, None)
        if section is not None and (type(section) is not expected_type or not _plain_numeric_state(section)):
            return False, name + " is not an exact reviewed builtin section"
    return True, ""



def exact_element_authority(element: Any) -> Tuple[bool, str]:
    """Return ``(admitted, typed_reason)`` for one element's load hooks.

    Only exact native classes are admitted; subclasses, foreign elements and
    patched class hooks are refused.  String or module-name matching is
    deliberately not used: identity of the class object and of the bound
    hook functions is the authority.
    """

    cls = type(element)
    if cls not in _EXACT_ELEMENT_HOOKS:
        return False, "element type is not an admitted exact native class"
    if not _instance_authority(element):
        return False, "element instance shadows a reviewed native method/helper"
    for name in ("node_ids", "cross_section", "material_name", "reference_normal", "material_direction", "eccentricity", "thickness", "_thickness", "reference_surface_offset", "material_angle_deg", "_area", "_Iy", "_Iz", "_J", "_ky", "_kz"):
        if name in vars(element) and not _plain_numeric_state(vars(element)[name]):
            return False, "element data contains a custom numeric/container provider: " + name
    expected = _EXACT_ELEMENT_HOOKS.get(cls)
    if expected is None:
        return False, (
            f"element type {cls.__module__}.{cls.__name__} is not an admitted "
            "exact native class"
        )
    for name, function in expected.items():
        if name in vars(element) or getattr(cls, name, None) is not function or not _callable_authority_matches(function, _EXACT_HOOK_FACTS[cls][name]):
            return False, f"class hook {name!r} is not the exact reviewed function"
    formulation_id = _EXACT_ELEMENT_FORMULATION_IDS.get(cls)
    if formulation_id is not None and str(
        getattr(element, "formulation_id", "")
    ) != formulation_id:
        return False, "element formulation_id does not match the admitted native class"
    return True, ""


class PreparedStaticError(RuntimeError):
    """A prepared static route failed; earlier yielded blocks remain valid."""

    def __init__(self, message: str, *, diagnostics: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.diagnostics: Dict[str, Any] = {} if diagnostics is None else dict(diagnostics)


class PreparedStaticCapabilityError(PreparedStaticError):
    """An explicitly refused capability, never a silent live-stream fallback."""


def _normalized_families(requested: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if requested is None:
        return SUPPORTED_LOAD_FAMILIES
    made = tuple(str(name) for name in requested)
    unknown = [name for name in made if name not in SUPPORTED_LOAD_FAMILIES]
    if unknown:
        raise ValueError(
            f"unsupported load families requested: {sorted(set(unknown))}; "
            f"supported families are {list(SUPPORTED_LOAD_FAMILIES)}"
        )
    if not made:
        raise ValueError("load_families must be None or a non-empty subset of supported families")
    return tuple(dict.fromkeys(made))


@dataclass(frozen=True)
class ElementLoadMap:
    """One element's DOF mapping with its frozen activity load scale."""

    dof_mapping: np.ndarray
    activity_scale: float


@dataclass(frozen=True)
class PressureLoadMap:
    """Reference-configuration pressure basis for one element.

    ``scalar_basis`` is the consistent load for unit scalar pressure.
    ``tuple_basis`` holds one column per per-node pressure value where the
    kernel admits nodal pressure; ``tuple_length`` records the admitted
    length.  ``follower_rejected`` mirrors the strict-flat S3 V2 follower
    refusal so follower cases fail exactly like the live oracle.
    ``refusal_reason`` records the typed reason when neither basis could be
    compiled; using pressure on such an element then fails with that reason
    instead of a silent unsupported-load skip.
    """

    dof_mapping: np.ndarray
    activity_scale: float
    scalar_basis: Optional[np.ndarray]
    tuple_basis: Optional[np.ndarray]
    tuple_length: int
    follower_rejected: bool
    kernel: str
    refusal_reason: str = ""


@dataclass(frozen=True)
class CanonicalLoadMaps:
    """Immutable per-family load compilation for one fixed structure."""

    total_dofs: int
    families: Tuple[str, ...]
    node_dofs: Mapping[int, np.ndarray]
    element_maps: Mapping[int, ElementLoadMap]
    pressure_maps: Mapping[int, PressureLoadMap]
    acceleration_available: bool
    acceleration_note: str
    retained_bytes: int
    verification: Tuple[Tuple[str, float], ...]
    refused_elements: Mapping[int, Tuple[Tuple[str, str], ...]] = ()
    pressure_recipes: Mapping[int, Any] = ()
    mass_maps: tuple = ()
    point_masses: tuple = ()

    def family_enabled(self, family: str) -> bool:
        return family in self.families


@dataclass(frozen=True)
class OwnedCaseLoads:
    """One case's ingested numeric load content; no live objects remain."""

    name: Optional[str]
    nodal: Tuple[Tuple[int, np.ndarray], ...]
    element: Tuple[Tuple[int, np.ndarray], ...]
    pressure: Tuple[Tuple[int, Tuple[str, Any]], ...]
    gravity: Optional[np.ndarray]
    added_masses: Tuple[Tuple[int, float], ...]
    follower: bool


def _finite_array(values: Any, *, what: str) -> np.ndarray:
    made = np.array(values, dtype=float, copy=True).reshape(-1)
    if not np.all(np.isfinite(made)):
        raise PreparedStaticError(f"{what} contains non-finite values")
    # Own the content: the source container may be reused and mutated by the
    # caller between the ingestion of successive cases.
    return made


def load_packet_storage_bytes(case: Any) -> int:
    """Bound owned copies before allocating them; reject callback-bearing input."""
    from .boundary import _exact_load_case_instance_namespace
    from .static_batch import StaticCase
    if type(case) is not StaticCase:
        raise PreparedStaticCapabilityError("cases must be exact StaticCase records")
    if case.load_case is None:
        return 256
    state = _exact_load_case_instance_namespace(case.load_case)
    if state is None:
        raise PreparedStaticCapabilityError("prepared static evaluator accepts only exact builtin LoadCase objects")
    def size(value):
        if type(value) is np.ndarray:
            if value.dtype.kind not in "biuf" or value.dtype.hasobject:
                raise PreparedStaticCapabilityError("load arrays must contain plain real numbers")
            return 256 + int(value.size) * 16
        if type(value) in (list, tuple):
            return 64 + sum(size(item) for item in value)
        if type(value) is dict:
            # M31: an int-keyed dict of floats (the usual load maps) has the same
            # size as below in closed form: each entry 128 + 64 + 64.
            if all(type(key) is int for key in value) and all(type(item) is float for item in value.values()):
                return 256 + 256 * len(value)
            return 256 + sum(128 + size(key) + size(item) for key, item in value.items())
        if value is None or type(value) in (bool, int, float, str):
            return 64 + (4 * len(value) if type(value) is str else 0)
        if type(value) in set(np.sctypeDict.values()) and type(value).__module__.startswith("numpy") and value.dtype.kind in "biuf":
            return 64
        raise PreparedStaticCapabilityError("load input contains a custom numeric/container provider")
    return 256 + sum(size(state[name]) for name in ("nodal_loads", "element_loads", "pressure_loads", "gravity", "added_node_masses", "name"))


def ingest_static_case(case: Any) -> Tuple[str, Optional[float], OwnedCaseLoads]:
    """Validate one StaticCase and own its numeric load content.

    Only an exact builtin ``LoadCase`` instance is accepted.  Foreign load
    objects are refused explicitly: after preparation the evaluator must not
    run provider code against a live model.
    """

    from .boundary import _exact_load_case_instance_namespace
    from .static_batch import StaticCase

    if type(case) is not StaticCase or type(case.case_id) is not str:
        raise PreparedStaticError(
            "cases must be StaticCase records with string case_id",
            diagnostics={"stage": "case validation"},
        )
    case_time = None if case.time is None else float(case.time)
    if case_time is not None and not np.isfinite(case_time):
        raise PreparedStaticError(
            "case time must be finite when supplied",
            diagnostics={"stage": "case validation", "case_id": case.case_id},
        )
    load = case.load_case
    if load is None:
        return case.case_id, case_time, OwnedCaseLoads(None, (), (), (), None, (), False)
    state = _exact_load_case_instance_namespace(load)
    if state is None:
        raise PreparedStaticCapabilityError(
            "prepared static evaluator accepts only exact builtin LoadCase objects; "
            "foreign load providers are refused",
            diagnostics={"stage": "load ingestion", "case_id": case.case_id},
        )
    try:
        nodal = tuple(
            (int(node_id), _finite_array(values, what="nodal load"))
            for node_id, values in state["nodal_loads"].items()
        )
        element = tuple(
            (int(element_id), _finite_array(values, what="element load"))
            for element_id, values in state["element_loads"].items()
        )
        pressure: List[Tuple[int, Tuple[str, Any]]] = []
        for element_id, values in state["pressure_loads"].items():
            if isinstance(values, (int, float, np.integer, np.floating)):
                value = float(values)
                if not np.isfinite(value):
                    raise PreparedStaticError("pressure load contains non-finite values")
                pressure.append((int(element_id), ("scalar", value)))
            elif type(values) in (tuple, list):
                made = tuple(float(item) for item in values)
                if not all(np.isfinite(item) for item in made):
                    raise PreparedStaticError("pressure load contains non-finite values")
                pressure.append((int(element_id), ("tuple", made)))
            else:
                raise PreparedStaticCapabilityError(
                    "pressure load must be a scalar or a per-node tuple of scalars",
                    diagnostics={"stage": "load ingestion", "case_id": case.case_id,
                                 "element_id": int(element_id)},
                )
        gravity = None if state["gravity"] is None else _finite_array(
            state["gravity"], what="gravity"
        )
        if gravity is not None and gravity.shape != (3,):
            raise PreparedStaticError(
                "gravity/acceleration field must have three components",
                diagnostics={"stage": "load ingestion", "case_id": case.case_id},
            )
        added = tuple(
            (int(node_id), float(mass))
            for node_id, mass in state["added_node_masses"].items()
        )
        if not all(np.isfinite(mass) for _node, mass in added):
            raise PreparedStaticError("added node masses contain non-finite values")
    except PreparedStaticError:
        raise
    except Exception as error:
        raise PreparedStaticError(
            "load case ingestion failed",
            diagnostics={"stage": "load ingestion", "case_id": case.case_id,
                         "error": str(error), "original_exception_type": type(error).__name__},
        ) from error
    owned = OwnedCaseLoads(
        name=state["name"],
        nodal=nodal,
        element=element,
        pressure=tuple(pressure),
        gravity=gravity,
        added_masses=added,
        follower=bool(state["follower_pressure"]),
    )
    return case.case_id, case_time, owned


def _require_family(maps: CanonicalLoadMaps, family: str, case_id: str) -> None:
    if not maps.family_enabled(family):
        raise PreparedStaticCapabilityError(
            f"case uses the {family!r} load family which this evaluator did not prepare",
            diagnostics={"stage": "load evaluation", "case_id": case_id, "family": family},
        )


def _refused_element(
    maps: CanonicalLoadMaps,
    element_id: int,
    family: str,
    case_id: str,
) -> None:
    reasons = maps.refused_elements.get(int(element_id))
    if reasons is None:
        return
    matched = tuple(reason for name, reason in reasons if name == family)
    if not matched:
        return
    raise PreparedStaticCapabilityError(
        f"case uses the {family!r} load family on element {int(element_id)} "
        "whose compiled map was refused at preparation",
        diagnostics={
            "stage": "load evaluation",
            "case_id": case_id,
            "family": family,
            "element_id": int(element_id),
            "typed_refusal_reasons": list(matched),
        },
    )


def evaluate_load_vector(maps: CanonicalLoadMaps, owned: OwnedCaseLoads) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evaluate one case's exact load vector from the compiled maps."""

    total_dofs = int(maps.total_dofs)
    load = np.zeros(total_dofs, dtype=float)
    if owned.nodal:
        _require_family(maps, "nodal", owned.name or "?")
        for node_id, values in owned.nodal:
            dofs = maps.node_dofs.get(int(node_id))
            if dofs is None:
                continue
            count = min(int(dofs.size), int(values.size))
            for i in range(count):
                load[dofs[i]] += values[i]
    if owned.element:
        _require_family(maps, "element", owned.name or "?")
        for element_id, values in owned.element:
            _refused_element(maps, int(element_id), "element", owned.name or "?")
            mapping = maps.element_maps.get(int(element_id))
            if mapping is None:
                continue
            count = min(int(mapping.dof_mapping.size), int(values.size))
            for i in range(count):
                load[mapping.dof_mapping[i]] += mapping.activity_scale * values[i]
    if owned.pressure:
        _require_family(maps, "pressure", owned.name or "?")
        from .boundary import LoadCase, _load_routing_classes_are_untouched
        from .prepared_static_recovery import owned_recovery_scope
        # M17: routing authority is checked once before and once after the whole
        # pressure loop instead of around every element. No caller code runs
        # inside the loop, scatter order is unchanged and a failed check still
        # discards the vector, so results are bitwise identical and fail closed.
        if not _load_routing_classes_are_untouched():
            raise PreparedStaticCapabilityError("native load routing authority changed")
        # M22: one probe per evaluation; the pressure helpers never write to it.
        probe = LoadCase("prepared_owned_pressure")
        for element_id, (kind, values) in owned.pressure:
            _refused_element(maps, int(element_id), "pressure", owned.name or "?")
            pressure_map = maps.pressure_maps.get(int(element_id))
            if pressure_map is None:
                if int(element_id) not in maps.element_maps:
                    continue
                raise PreparedStaticCapabilityError(
                    "pressure loads are not admitted for this element's kernel",
                    diagnostics={"stage": "load evaluation", "case_id": owned.name or "?",
                                 "element_id": int(element_id)},
                )
            if pressure_map.scalar_basis is None and pressure_map.tuple_basis is None:
                raise PreparedStaticCapabilityError(
                    "pressure loads are not admitted for this element's kernel: "
                    + pressure_map.refusal_reason,
                    diagnostics={"stage": "load evaluation", "case_id": owned.name or "?",
                                 "element_id": int(element_id),
                                 "typed_refusal_reason": pressure_map.refusal_reason},
                )
            if owned.follower and pressure_map.follower_rejected:
                raise PreparedStaticCapabilityError(
                    "strict-flat S3 V2 capability 'follower_pressure' is outside the "
                    "admitted linear-static pressure map",
                    diagnostics={"stage": "load evaluation", "case_id": owned.name or "?",
                                 "element_id": int(element_id)},
                )
            # Preserve native pressure multiplication/quadrature order. A unit
            # pressure vector scaled afterward changes cancellation-sensitive F.
            recipe = maps.pressure_recipes.get(int(element_id))
            if recipe is None:
                raise PreparedStaticCapabilityError("canonical pressure recipe unavailable")
            element, mesh, _material = recipe.execution_owner
            with owned_recovery_scope((recipe,)):
                coords = probe._current_element_coordinates(element, mesh, None) if owned.follower else None
                f_elem = probe._consistent_pressure_load(element, mesh, float(values) if kind == "scalar" else tuple(values), coords)
            for i, dof in enumerate(pressure_map.dof_mapping):
                if i < len(f_elem):
                    load[dof] += pressure_map.activity_scale * f_elem[i]
        if not _load_routing_classes_are_untouched():
            raise PreparedStaticCapabilityError("native load routing authority changed")
    if owned.gravity is not None:
        _require_family(maps, "acceleration", owned.name or "?")
        if not maps.acceleration_available:
            raise PreparedStaticCapabilityError(
                "the acceleration load family could not be compiled for this structure",
                diagnostics={"stage": "load evaluation", "case_id": owned.name or "?",
                             "note": maps.acceleration_note},
            )
        # Canonical staged M @ repeated acceleration, then element-order scatter.
        for mapping, scale, mass_matrix in maps.mass_maps:
            acceleration = np.zeros(mass_matrix.shape[1])
            for offset in range(0, acceleration.size, 6):
                acceleration[offset:offset + 3] = owned.gravity
            f_elem = np.asarray(mass_matrix @ acceleration, dtype=float).reshape(-1)
            for i, dof in enumerate(mapping):
                if i < f_elem.size:
                    load[dof] += scale * f_elem[i]
        combined = dict(maps.point_masses)
        for node_id, mass in owned.added_masses:
            combined[int(node_id)] = combined.get(int(node_id), 0.) + float(mass)
        for node_id, mass in combined.items():
            dofs = maps.node_dofs.get(int(node_id))
            if dofs is None or mass == 0.0:
                continue
            for axis in range(3):
                load[dofs[axis]] += float(mass) * owned.gravity[axis]
    if not np.all(np.isfinite(load)):
        raise PreparedStaticError(
            f"load case {owned.name!r} produced non-finite load vector values",
            diagnostics={"stage": "load evaluation", "case_id": owned.name},
        )
    info = {
        "vector_type": "load",
        "load_case": owned.name,
        "total_dofs": total_dofs,
        "assembly_time": 0.0,
        "load_norm": float(np.linalg.norm(load)),
        "pressure_configuration": "current" if owned.follower else "reference",
        "prepared_load_evaluation": True,
    }
    return load, info


def _affine_probe_residual(basis: np.ndarray, values: np.ndarray, observed: np.ndarray) -> float:
    expected = np.asarray(basis @ np.asarray(values, dtype=float), dtype=float).reshape(-1)
    made = np.asarray(observed, dtype=float).reshape(-1)
    scale = max(float(np.linalg.norm(expected)), float(np.linalg.norm(made)), 1.0)
    return float(np.linalg.norm(made - expected) / scale)


def _pressure_probes_are_package_only(model, probe=None) -> bool:
    """Reuse the canonical load proof, conservatively limiting the fast span.

    The acceleration preflight proves plain built-in material/element state,
    exact routing/mass helper classes and native model providers. Although this
    span only probes pressure, requiring that stronger proof avoids introducing
    a separate, weaker definition of callback-free element inputs.
    """
    if any(getattr(_boundary_module, name) is not function for name, function in _PRESSURE_PROOF_IDENTITIES):
        return False
    from .boundary import (LoadCase, _exact_load_case_instance_namespace, _load_model_access_is_builtin,
                           _load_routing_classes_are_untouched,
                           _numpy_error_routes_are_builtin,
                           _builtin_acceleration_inputs)
    if probe is None:
        probe = LoadCase("prepared_static_pressure_proof")
    if _exact_load_case_instance_namespace(probe) is None:
        return False
    if not (_load_model_access_is_builtin(model)
            and _load_routing_classes_are_untouched()
            and _numpy_error_routes_are_builtin()):
        return False
    mesh = model.mesh
    if mesh.element_activity is not None:
        return False
    mesh_state = object.__getattribute__(mesh, "__dict__")
    for mapping in (mesh.nodes, mesh.elements):
        if type(mapping) is not dict:
            namespace = object.__getattribute__(mapping, "__dict__")
            if any(name in namespace for name in vars(dict)):
                return False
    nodes = set()
    state = {"gravity": np.zeros(3), "added_node_masses": {}}
    if not _builtin_acceleration_inputs(state, mesh_state, model.get_material, mesh.elements, nodes):
        return False
    for node_id in nodes:
        node = dict.get(mesh.nodes, node_id)
        if node is None:
            continue
        if type(node) is not Node:
            return False
        data = object.__getattribute__(node, "__dict__")
        if ("coords" in data or type(data.get("dofs")) is not list
                or not all(type(dof) is int for dof in data["dofs"])
                or not all(type(data.get(axis)) in (float, int) for axis in ("x", "y", "z"))):
            return False
    return True


def compile_canonical_load_maps(
    model: "FEModel",
    *,
    families: Sequence[str],
) -> CanonicalLoadMaps:
    """Compile and verify the canonical load maps against the live oracle."""

    from .boundary import LoadCase
    from .matrix_assembly import _activity_scales
    from .analysis_session import _immutable_array

    require_native_source_authority()
    enabled = _normalized_families(families)
    mesh = model.mesh
    dof_manager = mesh.dof_manager
    total_dofs = int(dof_manager.total_dofs)
    retained = 0

    node_dofs: Dict[int, np.ndarray] = {}
    for node_id, node in mesh.nodes.items():
        dofs = _immutable_array(np.asarray(node.dofs, dtype=np.intp).reshape(-1))
        node_dofs[int(node_id)] = dofs
        retained += int(dofs.nbytes)

    _activity, activity_scales, _activity_info = _activity_scales(model, "load")

    refused_elements: Dict[int, Tuple[Tuple[str, str], ...]] = {}
    element_maps: Dict[int, ElementLoadMap] = {}
    for element_id, element in mesh.elements.items():
        admitted, reason = exact_element_authority(element)
        if admitted:
            admitted, reason = native_material_section_authority(model, element)
        if not admitted:
            refused_elements[int(element_id)] = (
                ("element", reason),
                ("pressure", reason),
                ("acceleration", reason),
            )
            continue
        mapping = _immutable_array(
            np.asarray(element.get_dof_mapping(mesh), dtype=np.intp).reshape(-1)
        )
        element_maps[int(element_id)] = ElementLoadMap(
            dof_mapping=mapping,
            activity_scale=float(activity_scales.get(int(element_id), 1.0)),
        )
        retained += int(mapping.nbytes)

    pressure_maps: Dict[int, PressureLoadMap] = {}
    if "pressure" in enabled:
        probe = LoadCase("prepared_static_pressure_probe")
        def compile_pressure(_lease=None):
            nonlocal retained
            for element_id, element in dict.items(mesh.elements):
                if int(element_id) not in element_maps:
                    continue
                if not hasattr(element, "node_ids"):
                    continue
                mapping = element_maps[int(element_id)].dof_mapping
                scale = element_maps[int(element_id)].activity_scale
                kernel = "consistent_pressure"
                scalar_basis = None
                scalar_failure = ""
                try:
                    unit = np.asarray(
                        probe._consistent_pressure_load(element, mesh, 1.0, None), dtype=float
                    ).reshape(-1)
                    half = np.asarray(
                        probe._consistent_pressure_load(element, mesh, 0.5, None), dtype=float
                    ).reshape(-1)
                    if (unit.size == mapping.size and half.size == mapping.size
                            and _affine_probe_residual(unit[:, None], [0.5], half) <= 1e-12):
                        scalar_basis = _immutable_array(unit)
                    else:
                        scalar_failure = "scalar pressure kernel is not affine in the pressure value"
                except Exception as error:  # typed record, never a silent skip
                    from .control import SolveCancelled
                    if isinstance(error, SolveCancelled):
                        raise
                    scalar_failure = (
                        f"scalar pressure kernel probe failed: {type(error).__name__}: {error}"
                    )
                tuple_basis = None
                tuple_failure = ""
                tuple_length = int(len(element.node_ids))
                try:
                    if not callable(getattr(element, "compute_native_dead_pressure_load", None)):
                        raise ValueError("affine nodal pressure requires a formulation-specific dead-pressure kernel")
                    columns = []
                    for index in range(tuple_length):
                        values = tuple(1.0 if j == index else 0.0 for j in range(tuple_length))
                        column = np.asarray(
                            probe._consistent_pressure_load(element, mesh, values, None),
                            dtype=float,
                        ).reshape(-1)
                        columns.append(column)
                    basis = _immutable_array(np.column_stack(columns))
                    rng = np.random.default_rng(20261008)
                    random_values = rng.uniform(-1.0, 1.0, tuple_length)
                    observed = np.asarray(
                        probe._consistent_pressure_load(element, mesh, tuple(random_values), None),
                        dtype=float,
                    ).reshape(-1)
                    if (basis.shape[0] == mapping.size
                            and _affine_probe_residual(basis, random_values, observed) <= 1e-12):
                        tuple_basis = basis
                    else:
                        tuple_failure = (
                            "per-node tuple pressure kernel is not affine in the "
                            "per-node pressure values"
                        )
                except Exception as error:  # typed record, never a silent skip
                    from .control import SolveCancelled
                    if isinstance(error, SolveCancelled):
                        raise
                    tuple_failure = (
                        f"tuple pressure kernel probe failed: {type(error).__name__}: {error}"
                    )
                try:
                    probe._reject_strict_flat_s3_v2_follower_pressure(element)
                    follower_rejected = False
                except Exception:
                    follower_rejected = True
                if scalar_basis is None and tuple_basis is None:
                    reason = "; ".join(
                        part for part in (scalar_failure, tuple_failure) if part
                    ) or "pressure kernel admitted no affine basis"
                    refused_elements.setdefault(int(element_id), ())
                    refused_elements[int(element_id)] = tuple(
                        list(refused_elements[int(element_id)]) + [("pressure", reason)]
                    )
                    pressure_maps[int(element_id)] = PressureLoadMap(
                        dof_mapping=mapping,
                        activity_scale=scale,
                        scalar_basis=None,
                        tuple_basis=None,
                        tuple_length=tuple_length,
                        follower_rejected=follower_rejected,
                        kernel=kernel,
                        refusal_reason=reason,
                    )
                    continue
                pressure_maps[int(element_id)] = PressureLoadMap(
                    dof_mapping=mapping,
                    activity_scale=scale,
                    scalar_basis=scalar_basis,
                    tuple_basis=tuple_basis,
                    tuple_length=tuple_length,
                    follower_rejected=follower_rejected,
                    kernel=kernel,
                )
                for basis in (scalar_basis, tuple_basis):
                    if basis is not None:
                        retained += int(basis.nbytes)

        if _pressure_probes_are_package_only(model, probe):
            from .matrix_assembly import _run_with_qualified_assembly_runtime_lease
            _run_with_qualified_assembly_runtime_lease(
                model, context="prepared static package-only pressure probes",
                operation=compile_pressure,
            )
        else:
            compile_pressure()

    acceleration_available = False
    acceleration_note = ""
    mass_maps = []
    if "acceleration" in enabled:
        from .control import SolveCancelled

        refused_mass = [int(eid) for eid, element in mesh.elements.items()
                        if hasattr(element, "node_ids") and int(eid) not in element_maps]
        if refused_mass:
            acceleration_note = "acceleration family refused for non-admitted element ids: " + ", ".join(map(str, sorted(refused_mass)))
        else:
            try:
                # Runtime preserves each native M @ a and element-order scatter.
                # Capture that data once; the former three unit assemblies only
                # constructed an unused global acceleration basis. Independent
                # full-F verification below is deliberately unchanged.
                for eid, element in dict.items(mesh.elements):
                    if not hasattr(element, "node_ids"):
                        continue
                    mass = element.compute_mass_matrix(mesh, model.get_material(element.material_name))
                    mass = _immutable_array(np.asarray(mass, dtype=float))
                    entry = element_maps[int(eid)]
                    mass_maps.append((entry.dof_mapping, entry.activity_scale, mass))
                # Retain the previous unit-load capability boundary without
                # recomputing native mass matrices. In particular a nonfinite
                # M or overflow during canonical scatter must leave other load
                # families usable, rather than escaping later oracle verification.
                for axis in range(3):
                    gravity = np.eye(3, dtype=float)[axis]
                    unit_load = np.zeros(total_dofs, dtype=float)
                    for mapping, scale, mass in mass_maps:
                        acceleration = np.zeros(len(mapping), dtype=float)
                        for offset in range(0, acceleration.size, 6):
                            acceleration[offset:offset + 3] = gravity
                        force = np.asarray(mass @ acceleration, dtype=float).reshape(-1)
                        for index, dof in enumerate(mapping):
                            if index < force.size:
                                unit_load[dof] += scale * force[index]
                    combined_masses = {}
                    for node_id, value in mesh.point_masses.items():
                        node_id = int(node_id)
                        combined_masses[node_id] = combined_masses.get(node_id, 0.) + float(value)
                    for node_id, mass in combined_masses.items():
                        dofs = node_dofs.get(node_id)
                        if dofs is not None and mass != 0.:
                            for component in range(3):
                                unit_load[dofs[component]] += mass * gravity[component]
                    if not np.all(np.isfinite(unit_load)):
                        raise ValueError('captured mass produces a nonfinite canonical unit-acceleration load')
                acceleration_available = True
            except SolveCancelled:
                raise
            except (ValueError, TypeError, NotImplementedError, AttributeError) as error:
                # A model may be useful for nodal/pressure loading even when a
                # generalized section deliberately provides no structural mass.
                mass_maps.clear()
                acceleration_note = f"canonical acceleration mass capture unavailable: {type(error).__name__}: {error}"
        retained += sum(mass.nbytes for _mapping, _scale, mass in mass_maps)

    from .prepared_static_recovery import _capture_owned_local_recovery, _authenticate_trusted_capture
    _authenticate_trusted_capture()
    pressure_recipes = {}
    for eid, pressure_map in pressure_maps.items():
        if pressure_map.scalar_basis is not None or pressure_map.tuple_basis is not None:
            recipe = _capture_owned_local_recovery(model, mesh.elements[eid], warm_stress=False,
                pressure_probe=1.0 if pressure_map.scalar_basis is not None else (1.0,) * pressure_map.tuple_length)
            pressure_recipes[eid] = recipe
            retained += recipe.retained_bytes
    point_masses = tuple((int(node), float(mass)) for node, mass in mesh.point_masses.items())
    retained += len(point_masses) * 80
    maps = CanonicalLoadMaps(
        total_dofs=total_dofs,
        families=enabled,
        node_dofs=node_dofs,
        element_maps=element_maps,
        pressure_maps=pressure_maps,
        acceleration_available=acceleration_available,
        acceleration_note=acceleration_note,
        retained_bytes=int(retained),
        verification=(),
        refused_elements=MappingProxyType(refused_elements),
        pressure_recipes=MappingProxyType(pressure_recipes), mass_maps=tuple(mass_maps), point_masses=point_masses,
    )
    verification = _verify_maps_against_oracle(model, maps)
    return CanonicalLoadMaps(
        total_dofs=maps.total_dofs,
        families=maps.families,
        node_dofs=maps.node_dofs,
        element_maps=maps.element_maps,
        pressure_maps=maps.pressure_maps,
        acceleration_available=maps.acceleration_available,
        acceleration_note=maps.acceleration_note,
        retained_bytes=maps.retained_bytes,
        verification=verification,
        refused_elements=maps.refused_elements,
        pressure_recipes=maps.pressure_recipes, mass_maps=maps.mass_maps, point_masses=maps.point_masses,
    )


def _verify_maps_against_oracle(
    model: "FEModel",
    maps: CanonicalLoadMaps,
) -> Tuple[Tuple[str, float], ...]:
    """Compare each compiled family with the live assembly oracle."""

    from .boundary import LoadCase
    from .matrix_assembly import assemble_load_vector

    rng = np.random.default_rng(20261008)
    results: List[Tuple[str, float]] = []

    def probe_case(name: str, **fields: Any) -> Any:
        case = LoadCase(name)
        for key, value in fields.items():
            setattr(case, key, value)
        return case

    def compare(family: str, case: Any) -> None:
        oracle, _info = assemble_load_vector(model, case)
        if family == "pressure":
            # The canonical oracle remains outside this span. Only freshly
            # ingested numeric input and privately captured native recipes enter;
            # no provider, iterator, callback or yield is observed here.
            from . import prepared_static_recovery as recovery
            if (_pressure_static_batch.StaticCase is not _PRESSURE_STATIC_CASE
                    or _class_state(_PRESSURE_STATIC_CASE) != _PRESSURE_STATIC_CASE_FACTS):
                raise PreparedStaticCapabilityError('pressure verification case authority changed')
            for name, function, authority, bindings in _PRESSURE_EVALUATION_FACTS:
                if (globals().get(name) is not function or not _callable_authority_matches(function, authority)
                        or any(function.__globals__.get(key) is not value for key, value in bindings)):
                    raise PreparedStaticCapabilityError('pressure verification helper authority changed')
            for name, cls, facts in _PRESSURE_RECORD_FACTS:
                if globals().get(name) is not cls or _class_state(cls) != facts:
                    raise PreparedStaticCapabilityError('pressure verification record authority changed')
            if (recovery.owned_recovery_scope is not recovery._OWNED_SCOPE_FACTORY
                    or not _callable_authority_matches(recovery.owned_recovery_scope, recovery._OWNED_SCOPE_FACTORY_FACTS)):
                raise PreparedStaticCapabilityError('pressure verification scope authority changed')
            if (recovery.OwnedLocalRecovery is not recovery._OWNED_RECIPE_CLASS
                    or _class_state(recovery.OwnedLocalRecovery) != recovery._OWNED_RECIPE_CLASS_FACTS
                    or any(type(recipe) is not recovery._OWNED_RECIPE_CLASS for recipe in maps.pressure_recipes.values())):
                raise PreparedStaticCapabilityError('pressure verification recipe authority changed')
            _probe_id, _probe_time, owned = ingest_static_case(_static_case_probe(case))
            with recovery._OWNED_SCOPE_FACTORY(tuple(maps.pressure_recipes.values())):
                evaluated, _info2 = evaluate_load_vector(maps, owned)
        else:
            _probe_id, _probe_time, owned = ingest_static_case(_static_case_probe(case))
            evaluated, _info2 = evaluate_load_vector(maps, owned)
        scale = max(float(np.linalg.norm(oracle)), 1.0)
        residual = float(np.linalg.norm(np.asarray(oracle) - np.asarray(evaluated)) / scale)
        results.append((family, residual))
        if not np.isfinite(residual) or residual > 1e-11:
            raise PreparedStaticCapabilityError(
                f"compiled {family} load map does not reproduce the live assembly oracle",
                diagnostics={"stage": "load map verification", "family": family,
                             "relative_residual": residual if np.isfinite(residual) else None},
            )

    if maps.family_enabled("nodal") and maps.node_dofs:
        nodal = {
            node_id: rng.uniform(-1.0, 1.0, 6)
            for node_id in list(maps.node_dofs)[: max(len(maps.node_dofs), 0)]
        }
        compare("nodal", probe_case("prepared_nodal_probe", nodal_loads=nodal))
    if maps.family_enabled("element") and maps.element_maps:
        element = {
            element_id: rng.uniform(-1.0, 1.0, int(item.dof_mapping.size))
            for element_id, item in maps.element_maps.items()
        }
        compare("element", probe_case("prepared_element_probe", element_loads=element))
    if maps.family_enabled("pressure") and maps.pressure_maps:
        pressure: Dict[int, Any] = {}
        for element_id, item in maps.pressure_maps.items():
            if item.tuple_basis is not None:
                pressure[int(element_id)] = tuple(
                    rng.uniform(-1.0, 1.0, item.tuple_length).tolist()
                )
            elif item.scalar_basis is not None:
                pressure[int(element_id)] = float(rng.uniform(-1.0, 1.0))
        if pressure:
            compare("pressure", probe_case("prepared_pressure_probe_case", pressure_loads=pressure))
    if maps.family_enabled("acceleration") and maps.acceleration_available:
        gravity = rng.uniform(-1.0, 1.0, 3)
        added = {
            node_id: float(rng.uniform(0.0, 2.0))
            for node_id in list(maps.node_dofs)[: min(len(maps.node_dofs), 4)]
        }
        compare(
            "acceleration",
            probe_case("prepared_acceleration_probe", gravity=gravity,
                       added_node_masses=added),
        )
    return tuple(results)


def _static_case_probe(load_case: Any) -> Any:
    from .static_batch import StaticCase

    return StaticCase("prepared_verification_probe", load_case, 0.0)


# Pin the executable span before any caller can affect verification inputs.
from . import static_batch as _pressure_static_batch
_PRESSURE_STATIC_CASE = _pressure_static_batch.StaticCase
_PRESSURE_STATIC_CASE_FACTS = _class_state(_PRESSURE_STATIC_CASE)
_PRESSURE_EVALUATION_FACTS = tuple((name, function, _callable_authority(function),
    tuple((key, function.__globals__[key]) for key in function.__code__.co_names if key in function.__globals__))
    for name in ('ingest_static_case', '_static_case_probe',
                 'evaluate_load_vector', '_require_family', '_refused_element')
    for function in (globals()[name],))
_PRESSURE_RECORD_FACTS = tuple((name, cls, _class_state(cls))
    for name in ('CanonicalLoadMaps', 'OwnedCaseLoads', 'PressureLoadMap', 'ElementLoadMap')
    for cls in (globals()[name],))
