"""Recovery and reaction diagnostics for the prepared static evaluator.

Stress recovery is compiled at preparation time by canonical zero and
local-unit-basis probes of the package's own recovery oracle.  Admission is
by exact native class identity plus reviewed-hook identity (never by
formulation strings); the numerical affinity probe that follows is
verification, not admission.

Components are classified into three groups:

- frozen constants (plain scalars, strings, booleans, nested tuples of plain
  values, and arrays that do not respond to any local probe);
- canonical linear fields, compiled affinely as ``baseline + matrix @ u``;
- derived measures (von Mises and equivalent-stress fields), which are
  never compiled affinely: they are computed afterward from the compiled
  canonical linear fields with the reviewed measure formulas and verified
  against the oracle at a random probe.

Any component outside these groups, or any verification mismatch, refuses
the element explicitly.

Reaction and constraint-force diagnostics are reproduced exactly from the
immutable snapshot: the raw residual convention ``K u - F`` on the unreduced
global system with fixed/slave/master separation and complete residual
records, matching the live oracle's diagnostics structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import copy
import sys
from types import MappingProxyType
from time import perf_counter_ns as _recovery_preparation_clock_ns
from contextlib import contextmanager, ExitStack, nullcontext
from contextvars import ContextVar

_OWNED_RECOVERY_SCOPE = ContextVar("prepared_owned_recovery", default=None)
# M25f (owner decision): the trusted capture (review item S3) is opt-in. Capture is
# fully guarded (the pre-M25 route) unless a preparation that the caller opted in
# sets this for the duration of its own snapshot capture.
class _TrustedCaptureToken:
    """Revocable opt-in lease; a context copied during preparation shares the token,
    so revoking it at the end of the scope disables the opt-in there too."""
    __slots__ = ("live",)

    def __init__(self):
        self.live = True


_TRUSTED_CAPTURE = ContextVar("prepared_trusted_capture", default=None)


@contextmanager
def trusted_capture_scope(enabled):
    """Enable (True) or keep disabled (False) the opt-in trusted capture for the
    duration of one preparation's snapshot capture."""
    if type(enabled) is not bool:
        raise ValueError("trusted_capture must be True or False")
    token = _TrustedCaptureToken() if enabled else None
    marker = _TRUSTED_CAPTURE.set(token)
    try:
        yield
    finally:
        if token is not None:
            token.live = False  # revoked on every exit, before the reset
        _TRUSTED_CAPTURE.reset(marker)


def _trusted_capture_active():
    token = _TRUSTED_CAPTURE.get()
    return type(token) is _TrustedCaptureToken and token.live is True


def _authenticate_trusted_capture():
    """Capture callers authenticate the trusted-capture helpers only when it is enabled."""
    if _trusted_capture_active():
        _require_capture_span_authority()

import numpy as np
from .prepared_static_measures import PRINCIPAL_COMPONENTS, shell_principal_stresses
from .recovery import (_compute_one_element_stress as _canonical_one_element_stress,
                       estimate_model_memory, enforce_memory_limit, MemoryEstimate,
                       ResourcePolicyError)
from .boundary import _callable_authority, _callable_authority_matches

_PROBE_HELPER_AUTHORITY = _callable_authority(_canonical_one_element_stress)

# Scope factories receive private native elements. Pin their module bindings,
# code/defaults/closure executables and closure-function global dependencies
# before lending them those owners.
from . import e4_pl_element as _q4_scope_module
from . import e4_pl_s3_element as _s3_scope_module
# M25 review fix: one table pairs each factory with its element class; the
# factory and class tables are derived from it, so their order cannot drift.
_SCOPE_PAIRS = tuple((module, name, getattr(module, name), _callable_authority(getattr(module, name)),
                      class_name, getattr(module, class_name)) for module, name, class_name in (
        (_q4_scope_module, '_q4_trusted_operation_scope', 'QualifiedE4PLShellElement'),
        (_s3_scope_module, '_s3_trusted_operation_scope', 'QualifiedE4PLS3ShellElement')))
_SCOPE_FACTORIES = tuple((module, name, factory, authority) for module, name, factory, authority, _cname, _cls in _SCOPE_PAIRS)
_SCOPE_CLASSES = tuple((module, class_name, cls) for module, _name, _factory, _authority, class_name, cls in _SCOPE_PAIRS)
_SCOPE_ELEMENT_TYPES = tuple(cls for *_rest, cls in _SCOPE_PAIRS)

def _capture_scope_globals():
    import types
    seen = set()
    captured = []
    def visit(function):
        if type(function) is not types.FunctionType or id(function) in seen:
            return
        seen.add(id(function))
        captured.append((function, tuple((name, function.__globals__[name])
            for name in function.__code__.co_names if name in function.__globals__)))
        for cell in function.__closure__ or ():
            visit(cell.cell_contents)
    for _module, _name, factory, _authority in _SCOPE_FACTORIES:
        visit(factory)
    return tuple(captured)

_SCOPE_GLOBALS = _capture_scope_globals()

def _require_owned_scope_factory_authority():
    # The pre-M25 check, unchanged (it runs on the default route through
    # owned_recovery_scope). M25o (opt-in review round 9): the table-binding check
    # added in M25b runs only on the opt-in path (_require_capture_scope_tables).
    if (any(getattr(module, name, None) is not factory or not _callable_authority_matches(factory, authority)
            for module, name, factory, authority in _SCOPE_FACTORIES)
            or any(getattr(module, name, None) is not cls for module, name, cls in _SCOPE_CLASSES)
            or any(function.__globals__.get(name) is not value
                   for function, bindings in _SCOPE_GLOBALS for name, value in bindings)):
        raise PreparedStaticCapabilityError('native trusted scope factory authority changed')


def _make_capture_scope_tables_check():
    # Opt-in path only (M25b, moved in M25o): the tables live in this closure and
    # this module's bindings must still be those objects, so a rebound table cannot
    # pass the factory check that reads them.
    namespace = globals()
    tables = (('_SCOPE_PAIRS', _SCOPE_PAIRS), ('_SCOPE_FACTORIES', _SCOPE_FACTORIES), ('_SCOPE_CLASSES', _SCOPE_CLASSES),
              ('_SCOPE_ELEMENT_TYPES', _SCOPE_ELEMENT_TYPES), ('_SCOPE_GLOBALS', _SCOPE_GLOBALS))

    def require():
        if any(namespace.get(name) is not table for name, table in tables):
            raise PreparedStaticCapabilityError('native trusted scope factory authority changed')

    require.__name__ = require.__qualname__ = '_require_capture_scope_tables'
    return require


_require_capture_scope_tables = _make_capture_scope_tables_check()

from .prepared_static_loads import (
    PreparedStaticCapabilityError,
    PreparedStaticError,
    exact_element_authority,
    native_material_section_authority,
    require_native_source_authority,
)

from .boundary import (_numpy_error_routes_are_builtin as _capture_numpy_routes_are_builtin,
                       _load_routing_classes_are_untouched as _capture_routing_classes_are_untouched,
                       _exact_load_case_instance_namespace as _capture_exact_load_case_namespace)
from .prepared_static_loads import (_plain_numeric_state as _capture_plain_state,
                                    _instance_authority as _capture_instance_authority,
                                    _NATIVE_CLASS_FACTS as _capture_native_class_facts)
from .static_batch import _class_state as _capture_class_state
from anymaterial import IsotropicMaterial as _CaptureIsotropic, OrthotropicMaterial as _CaptureOrthotropic
from .beam_sections import GeneralizedBeamSection as _CaptureBeamSection
from .shell_sections import GeneralizedShellSection as _CaptureShellSection
# Immutable import-time facts for the materials and sections a capture span may use.


def _make_capture_material_facts():
    # M25n (opt-in review round 8): read on the first opt-in admission, not at import.
    facts_source = _capture_native_class_facts
    cache = []

    def facts():
        if not cache:
            cache.append(tuple((cls, facts_source[cls]) for cls in (_CaptureIsotropic, _CaptureOrthotropic)))
        return cache[0]

    facts.__name__ = facts.__qualname__ = '_capture_material_facts'
    return facts


_capture_material_facts = _make_capture_material_facts()
_CAPTURE_SECTION_CLASSES = (('generalized_section', _CaptureBeamSection), ('shell_section', _CaptureShellSection))
del _capture_native_class_facts


def _capture_state_table():
    # M25c review fix: the classes and module namespaces whose contents a callback
    # during admission could change to carry code into the span and that the
    # native factory does not re-check at entry. M25m (opt-in review round 7): built
    # on the first opt-in admission, after the module check has established that
    # warnings, sys and builtins are plain modules (never at import).
    import builtins
    import sys
    import warnings
    from numpy._core import _ufunc_config
    from abc import ABCMeta
    from .boundary import LoadCase
    from .fe_core import FEModel, FEMesh, Node, DOFManager, _QualifiedStateMapping
    classes = (LoadCase, FEModel, FEMesh, Node, DOFManager, _QualifiedStateMapping, ABCMeta,
               _CaptureIsotropic, _CaptureOrthotropic)
    namespaces = tuple(object.__getattribute__(module, '__dict__') for module in (warnings, sys, builtins))
    executables = []
    for cls in classes:
        for value in type.__getattribute__(cls, '__dict__').values():
            if type(value) is property:
                executables.extend(accessor for accessor in (value.fget, value.fset, value.fdel) if accessor is not None)
            elif type(value) in (staticmethod, classmethod):
                executables.append(value.__func__)
            else:
                executables.append(value)
    executables = tuple(value for value in executables if type(value) is _CaptureFunctionTypeEarly)
    return classes, namespaces, _ufunc_config._get_extobj_dict, executables


from types import FunctionType as _CaptureFunctionTypeEarly
from operator import attrgetter as _capture_attrgetter


def _make_capture_state_tables():
    cache = []

    def tables():
        if not cache:
            cache.append(_capture_state_table())
        return cache[0]

    tables.__name__ = tables.__qualname__ = '_capture_state_tables'
    return tables


_capture_state_tables = _make_capture_state_tables()
_capture_code_of = _capture_attrgetter('__code__')
_capture_defaults_of = _capture_attrgetter('__defaults__')

from . import prepared_static_q4 as _prepared_q4
_Q4_STAGE_REQUIRE = _prepared_q4.require_q4_kernel_authority
_Q4_STAGE_REQUIRE_AUTHORITY = _callable_authority(_Q4_STAGE_REQUIRE)

def _require_prepared_q4_authority():
    if (_prepared_q4.require_q4_kernel_authority is not _Q4_STAGE_REQUIRE
            or not _callable_authority_matches(_Q4_STAGE_REQUIRE, _Q4_STAGE_REQUIRE_AUTHORITY)):
        raise PreparedStaticCapabilityError('prepared Q4 authority guard changed')
    _Q4_STAGE_REQUIRE()

if TYPE_CHECKING:
    from .fe_core import FEModel
    from .recovery import RecoveryConfig

_AFFINITY_RELATIVE_TOLERANCE = 1e-9
_AFFINITY_ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class OwnedLocalRecovery:
    """Native private execution owner plus immutable numerical recipe identity."""
    numeric_identity: Any
    retained_bytes: int
    execution_owner: Any
    staged_kernel: Any = None
    staged_fallback_reason: str = ""


def _capture_plain_arguments(pressure_probe, warm_stress):
    """M25b: the span consumes these values, so they must be plain (no __bool__ or
    __float__ callback inside the span); type checks only."""
    if type(warm_stress) is not bool:
        return False
    if pressure_probe is None or type(pressure_probe) is float or type(pressure_probe) is int:
        return True
    return type(pressure_probe) is tuple and all(type(value) is float for value in pressure_probe)


def _capture_material_is_native(owned_element, owned_material):
    """M25b: callback-free material/section proof on the private clones that run
    inside the span: an exact reviewed elastic material class whose executables
    are unchanged since import, plain instance data without executable shadows,
    and exact plain sections. No mapping or attribute provider is called."""
    cls = type(owned_material)
    facts = None
    for native, native_facts in _capture_material_facts():
        if cls is native:
            facts = native_facts
    if facts is None or _capture_class_state(cls) != facts or not _capture_instance_authority(owned_material):
        return False
    namespace = object.__getattribute__(owned_material, "__dict__")
    if type(namespace) is not dict or not all(type(key) is str and _capture_plain_state(value)
                                               for key, value in dict.items(namespace)):
        return False
    element_namespace = object.__getattribute__(owned_element, "__dict__")
    if type(element_namespace) is not dict:
        return False
    for name, expected in _CAPTURE_SECTION_CLASSES:
        section = dict.get(element_namespace, name)
        if section is not None and (type(section) is not expected or not _capture_plain_state(section)):
            return False
    return True


def _capture_trust_state(owned_material=None, probe_case=None):
    """M25c/M25e: callback-free identity snapshot of the state admission must not
    change: LoadCase, routing and material class namespaces, the warnings, sys,
    builtins and boundary namespaces and warning filters, NumPy's error state, the
    probe LoadCase instance and (for admission) the material clone. The snapshot
    keeps the objects alive, so equal ids mean the same objects; NumPy's error
    modes compare by exact value. The owner's numerical inputs are checked by value
    after every capture (_capture_owner_matches_identity)."""
    made = []
    state_classes, state_namespaces, numpy_error_state, state_executables = _capture_state_tables()
    if probe_case is not None:
        namespace = object.__getattribute__(probe_case, '__dict__')
        made.append(namespace)
        made.extend(dict.keys(namespace))
        for value in dict.values(namespace):
            made.append(value)
            if type(value) is dict:
                made.extend(dict.keys(value))
                made.extend(dict.values(value))
    for cls in state_classes:
        namespace = type.__getattribute__(cls, '__dict__')
        made.extend(namespace.keys())
        made.extend(namespace.values())
    for namespace in state_namespaces:
        made.extend(dict.keys(namespace))
        made.extend(dict.values(namespace))
    # In-place executable changes of the classes' methods (code and defaults objects).
    made.extend(map(_capture_code_of, state_executables))
    made.extend(map(_capture_defaults_of, state_executables))
    filters = dict.get(state_namespaces[0], 'filters')
    made.append(filters)
    if type(filters) is list:
        made.extend(list.__iter__(filters))
    errors = numpy_error_state()
    modes = tuple(value if type(value) is str or type(value) is int else id(value) for value in dict.values(errors))
    material = None
    if owned_material is not None:
        namespace = object.__getattribute__(owned_material, '__dict__')
        material = (namespace, tuple(dict.keys(namespace)), tuple(dict.values(namespace)))
        material = (material, tuple(map(id, material[1])) + tuple(map(id, material[2])))
    return (made, tuple(map(id, made)), modes, material, errors)  # errors keeps the 'call' object alive


def _capture_same_state(before, after, *, material=True):
    """Shared part by ids and NumPy modes; the material part when requested."""
    if before[1] != after[1] or before[2] != after[2]:
        return False
    if material and (before[3] is None or after[3] is None or before[3][1] != after[3][1]):
        return False
    return True


def _capture_frozen_equal(left, right):
    """Equality of frozen plain values (tuples, scalars, bytes, str); NaN equals NaN."""
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        return len(left) == len(right) and all(_capture_frozen_equal(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return left == right or (left != left and right != right)
    return left == right


def _capture_owner_matches_identity(owned_element, owned_material, owned_mesh, freeze, frozen_inputs,
                                    frozen_material, geometry, ignored, inputs):
    """M25e: the private owner still holds exactly the inputs, material data and
    geometry frozen into its recipe identity (by value)."""
    namespace = object.__getattribute__(owned_element, '__dict__')
    if type(namespace) is not dict:
        return False
    current = {key: value for key, value in dict.items(namespace) if key not in ignored}
    if set(current) != set(inputs) or not _capture_frozen_equal(freeze(current), frozen_inputs):
        return False
    material_namespace = object.__getattribute__(owned_material, '__dict__')
    if type(material_namespace) is not dict or not _capture_frozen_equal(freeze(dict(material_namespace)), frozen_material):
        return False
    nodes = dict.get(object.__getattribute__(owned_mesh, '__dict__'), 'nodes')
    if type(nodes) is not dict and type(nodes) is not _CaptureStateMapping:
        return False
    for node_id, x, y, z in geometry:
        node = dict.get(nodes, node_id)
        state = None if node is None else object.__getattribute__(node, '__dict__')
        if type(state) is not dict or not all(
                type(dict.get(state, axis)) in (float, int) and _capture_frozen_equal(float(dict.get(state, axis)), value)
                for axis, value in (('x', x), ('y', y), ('z', z))):
            return False
    return True


def _capture_span_proofs_hold(owned_element, owned_material, probe_case, pressure_probe, warm_stress):
    """M25 (review S3, fixed three times): plain span arguments, builtin NumPy error
    and warning routes, native material/section (on the clones), untouched
    model/mesh/node routing classes and an exact LoadCase for the pressure probe
    (the proofs of boundary._run_foreign_code_free_load_vector, in its order). The
    caller brackets the proofs with a module-hook check and a state snapshot, so
    a callback reached during the proofs cannot carry code into the span."""
    return bool(_capture_plain_arguments(pressure_probe, warm_stress)
                and _capture_numpy_routes_are_builtin()
                and _capture_material_is_native(owned_element, owned_material)
                and _capture_routing_classes_are_untouched()
                and (probe_case is None or (_capture_exact_load_case_namespace(probe_case) is not None
                                            and _capture_pressure_helpers_authentic())))


def _capture_span_factory(owned_element, owned_material, probe_case, pressure_probe, warm_stress, span_authority):
    """(native trusted scope factory, admission snapshot) for a freshly cloned private
    Q4/S3 element, or (None, None) to keep the per-call guarded route (the pre-M25
    behaviour). The scope itself checks the element's runtime authority at entry
    and exit.

    M25e order: the module check first (so attribute lookups below do not reach a
    substituted module hook), the scope tables, a snapshot, then the proofs (an
    exception in a proof means "not admitted"). The final block before the span
    repeats the module check, compares the snapshot and re-runs the helper
    authentication; it is intended to be callback-free, but the round-five review
    found that it is not in every case (see the M25 report; opt-in path only). If
    anything changed during admission the capture is refused rather than continued
    on possibly changed owners."""
    if not _capture_modules_unhooked():
        return None, None
    _require_capture_scope_tables()
    _require_owned_scope_factory_authority()
    factory = None
    for _module, _name, candidate, _authority, _class_name, cls in _SCOPE_PAIRS:
        if type(owned_element) is cls:
            factory = candidate
    if factory is None:
        return None, None
    before = _capture_trust_state(owned_material, probe_case)
    try:
        admitted = _capture_span_proofs_hold(owned_element, owned_material, probe_case, pressure_probe, warm_stress)
    except Exception:
        admitted = False
    if not _capture_modules_unhooked() or not _capture_same_state(before, _capture_trust_state(owned_material, probe_case)):
        raise PreparedStaticCapabilityError("state changed during trusted-capture admission")
    span_authority()
    return (factory, before) if admitted else (None, None)


def _capture_owned_local_recovery(model, element, *, warm_stress=True, pressure_probe=None, _span_authority=None):
    from .fe_core import FEMesh, Node, _ELEMENT_LOCAL_CACHE_NAMES
    from .e4_pl_element import QualifiedE4PLShellElement
    from .e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from .prepared_static_loads import _plain_numeric_state
    import dataclasses
    import sys
    ignored = set(_ELEMENT_LOCAL_CACHE_NAMES) | {"_qualified_direct_state_token", "_qualified_direct_state_tokens", "_qualified_component_guard"}
    inputs = {key: value for key, value in vars(element).items() if key not in ignored}
    if not all(_plain_numeric_state(value) for value in inputs.values()):
        raise PreparedStaticCapabilityError("local canonical recipe contains nonplain element inputs")
    material = model.get_material(element.material_name)
    source_q4_policy = None
    if warm_stress and type(element) is QualifiedE4PLShellElement:
        # Authenticate the new helper before fetching it from its module. Keep
        # only its builtin bool, never the original mesh/scalar providers.
        _require_prepared_q4_authority()
        try:
            source_q4_policy = _prepared_q4._capture_original_q4_policy(element, model.mesh, material)
        except ValueError:
            # Native policy capture checks the Q4 module before the opt-in
            # prepared span is entered. Preserve its typed refusal when a
            # trusted factory was rebound, without probing hooked modules.
            if _trusted_capture_active() and _capture_modules_unhooked():
                _require_capture_scope_tables()
                _require_owned_scope_factory_authority()
            raise
    def freeze(value):
        if type(value) is np.ndarray:
            return (value.dtype.str, tuple(value.shape), value.tobytes())
        if isinstance(value, np.generic):
            return value.item()
        if dataclasses.is_dataclass(value):
            return (type(value).__module__, type(value).__qualname__, tuple((field.name, freeze(getattr(value, field.name))) for field in dataclasses.fields(value)))
        if type(value) is dict:
            return tuple((key, freeze(item)) for key, item in sorted(value.items()))
        if type(value) in (list, tuple):
            return tuple(freeze(item) for item in value)
        if value is None or type(value) in (bool, int, float, str, bytes):
            return value
        raise PreparedStaticCapabilityError("local canonical recipe contains opaque input")
    geometry = tuple((int(node_id), float(model.mesh.nodes[node_id].x), float(model.mesh.nodes[node_id].y), float(model.mesh.nodes[node_id].z)) for node_id in element.node_ids)
    identity = ((type(element).__module__, type(element).__qualname__), freeze(inputs), freeze(vars(material)), geometry)
    frozen_inputs, frozen_material = identity[1], identity[2]  # the pre-M25 expression, unchanged
    def clone_plain(value):
        # Avoid copyreg: generic deepcopy can mutate native dataclass classes
        # by creating __slotnames__, and V2D intentionally refuses restart.
        if type(value) is np.ndarray:
            return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        if dataclasses.is_dataclass(value):
            made = object.__new__(type(value))
            for field in dataclasses.fields(value):
                object.__setattr__(made, field.name, clone_plain(getattr(value, field.name)))
            return made
        if type(value) is dict:
            return {key: clone_plain(item) for key, item in value.items()}
        if type(value) in (tuple, list):
            return type(value)(clone_plain(item) for item in value)
        return value
    owned_element = object.__new__(type(element))
    for key, value in vars(element).items():
        if key in {"_qualified_direct_state_token", "_qualified_direct_state_tokens"}:
            continue
        object.__setattr__(owned_element, key, None if key in ignored else clone_plain(value))
    owned_material = object.__new__(type(material))
    for key, value in vars(material).items():
        object.__setattr__(owned_material, key, clone_plain(value))
    owned_mesh = FEMesh()
    for node_id, x, y, z in geometry:
        owned_mesh.add_node(node_id, x, y, z)
    owned_mesh.add_element(int(element.element_id), owned_element)
    owner = (owned_element, owned_mesh, owned_material)
    staged = None
    fallback_reason = 'pressure_only_recipe'
    if not _trusted_capture_active():
        # Default (M25f): the pre-M25 fully guarded capture, unchanged.
        if pressure_probe is not None:
            from .boundary import LoadCase
            LoadCase("prepared_owned_pressure_warmup")._consistent_pressure_load(owned_element, owned_mesh, pressure_probe, None)
        if warm_stress:
            _require_prepared_q4_authority()
            staged, fallback_reason = _prepared_q4.compile_q4_stages(
                owned_element, owned_mesh, owned_material, _source_compensated=source_q4_policy)
            if staged is not None:
                identity = (identity, staged)
            else:
                # Materialize fixed geometry/material caches before accounting retained
                # bytes. M21: a staged compile already evaluates this native zero-
                # displacement recovery, so the warm-up runs only without a staged kernel.
                owned_element.compute_stresses(owned_mesh, np.zeros(6 * len(geometry)), owned_material, return_global=False)
    else:
        # Opt-in trusted capture (M25, review item S3), admission-checked.
        probe_case = None
        if pressure_probe is not None:
            from .boundary import LoadCase
            probe_case = LoadCase("prepared_owned_pressure_warmup")
        if warm_stress:
            _require_prepared_q4_authority()
        # M25 (S3): a trusted native span only for exact Q4/S3 clones that pass the
        # admission checks (not a proof that no foreign code can run; see the M25
        # report); everything else keeps per-call guards.
        factory = admitted = None
        scoped_type = any(type(owned_element) is cls for cls in _SCOPE_ELEMENT_TYPES)
        if scoped_type:
            _span_authority()  # checker made at import, facts built on first use; held as a keyword default (M25b)
            factory, admitted = _capture_span_factory(owned_element, owned_material, probe_case, pressure_probe,
                                                      warm_stress, _span_authority)
        with (nullcontext() if factory is None else factory((owned_element,))):
            if probe_case is not None:
                probe_case._consistent_pressure_load(owned_element, owned_mesh, pressure_probe, None)
            if warm_stress:
                staged, fallback_reason = _prepared_q4.compile_q4_stages(
                    owned_element, owned_mesh, owned_material, _source_compensated=source_q4_policy)
                if staged is not None:
                    identity = (identity, staged)
                else:
                    # Materialize fixed geometry/material caches before accounting retained
                    # bytes. M21: a staged compile already evaluates this native zero-
                    # displacement recovery, so the warm-up runs only without a staged kernel.
                    owned_element.compute_stresses(owned_mesh, np.zeros(6 * len(geometry)), owned_material, return_global=False)
        if factory is not None and (not _capture_modules_unhooked() or not _capture_same_state(
                admitted, _capture_trust_state(None, probe_case), material=False)):
            # The span may cache on the private material clone; everything else it
            # relies on must be unchanged after the span.
            raise PreparedStaticCapabilityError("capture trusted span route changed during capture")
        if scoped_type and not _capture_owner_matches_identity(
                owned_element, owned_material, owned_mesh, freeze, frozen_inputs, frozen_material, geometry, ignored, inputs):
            # M25e: whichever route ran, the private owner must still hold exactly the
            # inputs its recipe identity records (a change during admission or the span
            # would otherwise yield a recipe that disagrees with its identity).
            raise PreparedStaticCapabilityError("captured owner state differs from its recipe identity")
    seen = set()
    def storage(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        size = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            return size + value.nbytes
        if isinstance(value, dict):
            return (size + sum(storage(key) + storage(item) for key, item in dict.items(value))
                    + (storage(vars(value)) if hasattr(value, "__dict__") else 0))
        if type(value) in (tuple, list):
            return size + sum(storage(item) for item in value)
        if hasattr(value, "__dict__") and not callable(value) and not isinstance(value, type):
            return size + storage(vars(value))
        return size
    if source_q4_policy is not None:
        _require_prepared_q4_authority()
        if _prepared_q4._capture_original_q4_policy(element, model.mesh, material) is not source_q4_policy:
            raise PreparedStaticCapabilityError('source Q4 arithmetic policy changed during capture')
    return OwnedLocalRecovery(identity, storage(owner) + storage(identity), owner, staged, fallback_reason)


@contextmanager
def owned_recovery_scope(recipes):
    """Authority-bounded package-only span; never crosses caller observation."""
    from .boundary import _numpy_error_routes_are_builtin
    owners = tuple(recipe.execution_owner[0] for recipe in recipes)
    current = _OWNED_RECOVERY_SCOPE.get()
    if current is not None and current[0] and all(id(owner) in current[1] for owner in owners):
        yield
        return
    require_native_source_authority()
    _require_owned_scope_factory_authority()
    _require_prepared_q4_authority()
    if not _numpy_error_routes_are_builtin():
        raise PreparedStaticCapabilityError("native NumPy error routes changed")
    owners = tuple(recipe.execution_owner[0] for recipe in recipes)
    live = [True, frozenset(id(element) for element in owners)]
    marker = None
    try:
        with ExitStack() as stack:
            stack.enter_context(_SCOPE_FACTORIES[0][2](tuple(e for e in owners if type(e) is _SCOPE_CLASSES[0][2])))
            stack.enter_context(_SCOPE_FACTORIES[1][2](tuple(e for e in owners if type(e) is _SCOPE_CLASSES[1][2])))
            marker = _OWNED_RECOVERY_SCOPE.set(live)
            yield
    finally:
        live[0] = False
        if marker is not None:
            _OWNED_RECOVERY_SCOPE.reset(marker)
        import sys
        primary = sys.exc_info()[1]
        try:
            require_native_source_authority()
            _require_owned_scope_factory_authority()
            _require_prepared_q4_authority()
            if not _numpy_error_routes_are_builtin():
                raise PreparedStaticCapabilityError("native NumPy error routes changed")
        except BaseException as authority_error:
            if primary is None:
                raise
            primary.add_note("owned recovery output authority also failed: " + str(authority_error))


_OWNED_SCOPE_FACTORY = owned_recovery_scope
_OWNED_SCOPE_FACTORY_FACTS = _callable_authority(owned_recovery_scope)
from .static_batch import _class_state as _prepared_class_state
_OWNED_RECIPE_CLASS = OwnedLocalRecovery
_OWNED_RECIPE_CLASS_FACTS = _prepared_class_state(OwnedLocalRecovery)


def _recover_owned_local_columns(recipe, columns):
    """M21: staged Q4 recovery of several displacement columns in one pass, else None."""
    scope = _OWNED_RECOVERY_SCOPE.get()
    element = recipe.execution_owner[0]
    if scope is None or not scope[0] or id(element) not in scope[1]:
        require_native_source_authority()
        _require_prepared_q4_authority()
    if recipe.staged_kernel is None:
        return None
    return _prepared_q4.recover_q4_stage_columns(recipe.staged_kernel, columns)


def _recover_owned_local(recipe, local):
    element, mesh, material = recipe.execution_owner
    scope = _OWNED_RECOVERY_SCOPE.get()
    if scope is None or not scope[0] or id(element) not in scope[1]:
        require_native_source_authority()
        _require_prepared_q4_authority()
    if recipe.staged_kernel is not None:
        return _prepared_q4.recover_q4_stages(recipe.staged_kernel, local)
    return element.compute_stresses(mesh, local, material, return_global=False)


@dataclass(frozen=True)
class CompiledElementRecovery:
    """One element's compiled recovery: constants, linear maps, derived names."""

    element_id: int
    formulation: str
    admitted_by: str
    dof_mapping: np.ndarray
    array_components: Tuple[Tuple[str, np.ndarray, np.ndarray, Tuple[int, ...]], ...]
    constant_components: Tuple[Tuple[str, Any], ...]
    derived_components: Tuple[Tuple[str, str], ...]
    verification_relative_residual: float
    local_recipe: Any = None


@dataclass(frozen=True)
class PreparedRecoveryCompilation:
    """All compiled recovery maps for the selected elements."""

    elements: Mapping[int, CompiledElementRecovery]
    retained_bytes: int
    probe_count: int
    setup_seconds: float


def _recovery_admission(model: "FEModel", element: Any, recovery: Any) -> Tuple[bool, str, str]:
    """Return (admitted, formulation label, authority) for one element."""

    from .recovery_batches import _formulation_name

    formulation = _formulation_name(model, element)
    admitted, reason = exact_element_authority(element)
    if not admitted:
        return False, formulation, reason
    admitted, reason = native_material_section_authority(model, element)
    if not admitted:
        return False, formulation, reason
    material = model.get_material(element.material_name)
    if getattr(material, "hill_yield", None) is not None and (recovery.components is None or set(recovery.components) & {"hill_utilization", "equivalent_stress"}):
        return False, formulation, (
            "Hill48 material measures are outside the admitted elastic "
            "prepared recovery"
        )
    cls = type(element).__name__
    if cls == "NativeParityE4PLS3V2DShellElement":
        authority = "source_reviewed_native_s3_v2d"
    elif cls == "QualifiedE4PLS3ShellElement":
        authority = "source_reviewed_native_s3"
    elif cls == "QualifiedE4PLShellElement":
        authority = "source_reviewed_native_q4"
    elif cls == "QuadraticBeamElement":
        authority = "source_reviewed_beam3"
    elif cls == "BeamElement":
        authority = "source_reviewed_beam2"
    else:
        authority = "source_reviewed_classic_shell"
    return True, formulation, authority


def _probe_element_stress(
    model: "FEModel",
    displacements: np.ndarray,
    element_id: int,
    recovery: "RecoveryConfig",
    *, _publication_authority=None,
) -> Dict[str, Any]:
    # A held checker is installed at module initialization. Caller-owned
    # RecoveryConfig observations must not change delayed private publication.
    if not _callable_authority_matches(_publication_authority[0],_publication_authority[1]):
        raise PreparedStaticCapabilityError('recovery publication checker authority changed')
    _publication_authority[0]()
    item = _canonical_one_element_stress(
        model, displacements, int(element_id), return_global=False,
    )
    recovered = None if item is None else item[1]
    if not isinstance(recovered, dict):
        raise PreparedStaticError(
            "stress recovery oracle returned no result for a selected element",
            diagnostics={"stage": "recovery compilation", "element_id": int(element_id)},
        )
    requested = recovery.selected_components()
    if not _callable_authority_matches(_publication_authority[0],_publication_authority[1]):
        raise PreparedStaticCapabilityError('recovery publication checker authority changed')
    _publication_authority[0]()
    if requested is not None:
        for name in set(requested).intersection(PRINCIPAL_COMPONENTS):
            try:
                recovered[name] = shell_principal_stresses(recovered, name.removeprefix("principal_stresses_"))
            except ValueError as error:
                raise PreparedStaticCapabilityError(str(error), diagnostics={"stage": "recovery compilation", "element_id": element_id, "component": name}) from error
        missing = set(requested) - set(recovered)
        if missing:
            raise PreparedStaticCapabilityError("requested recovery components are unavailable: " + ", ".join(sorted(missing)), diagnostics={"stage": "recovery compilation", "element_id": element_id, "components": sorted(missing)})
        wanted = set(requested)
        if wanted & {"von_mises", "equivalent_stress", "hill_utilization", *PRINCIPAL_COMPONENTS}:
            wanted.update({"von_mises", "membrane_xx", "membrane_yy", "membrane_xy", "bending_xx", "bending_yy", "bending_xy", "shear_xz", "shear_yz", "axial_stress", "bending_stress_y", "bending_stress_z", "shear_stress_y", "shear_stress_z", "torsional_stress"})
        recovered = {name: value for name, value in recovered.items() if name in wanted}
    if not _callable_authority_matches(_publication_authority[0],_publication_authority[1]):
        raise PreparedStaticCapabilityError('recovery publication checker authority changed')
    _publication_authority[0]()
    return recovered


def _plain_constant(value: Any, *, depth: int = 0) -> Any:
    """Return a frozen plain copy, or raise for a live/nested object."""

    if depth > 6:
        raise ValueError("recovery component nesting too deep")
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, np.generic) and value.dtype.kind in "biuf":
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain_constant(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        from .analysis_session import _immutable_array
        return _immutable_array(value)
    if type(value) in (tuple, list):
        return tuple(_plain_constant(item, depth=depth + 1) for item in value)
    raise ValueError(f"component of type {type(value).__name__} is not plain")


def _constant_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_constant_equal(left[key], right[key]) for key in left)
    if type(left) in (tuple, list) or type(right) in (tuple, list):
        try:
            return len(left) == len(right) and all(
                _constant_equal(a, b) for a, b in zip(left, right)
            )
        except TypeError:
            return False
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right))
        except Exception:
            return False
    try:
        return bool(left == right)
    except Exception:
        return False


def _array_value(value: Any) -> Optional[np.ndarray]:
    """Return the float content of an array/scalar component, else None."""

    if isinstance(value, np.ndarray):
        if value.dtype.kind != "f":
            return None
        return np.asarray(value, dtype=float)
    if type(value) in (int, float):
        return np.asarray(float(value), dtype=float).reshape(())
    if isinstance(value, np.generic) and value.dtype.kind in "biuf":
        return np.asarray(float(value), dtype=float).reshape(())
    return None


def _shell_surface_von_mises(record: Dict[str, Any]) -> Any:
    """Reviewed shell top/bottom surface von Mises from canonical fields.

    Source: ``elements.ShellElement.compute_stresses`` and
    ``e4_pl_s3_element.QualifiedE4PLS3ShellElement`` elastic recovery: the
    measure is the max of the top/bottom plane-stress von Mises with the
    shared transverse shear.  The formula is shared with the offline capsule
    runtime through :mod:`anyfesolver_results.formulas`.
    """

    from anyfesolver_results.formulas import shell_surface_von_mises
    return shell_surface_von_mises(record)


def _beam_von_mises(record: Dict[str, Any]) -> Any:
    """Reviewed beam von Mises from canonical section fields.

    Source: ``elements.BeamElement.compute_stresses`` elastic path.  The
    formula is shared with the offline capsule runtime through
    :mod:`anyfesolver_results.formulas`.
    """

    from anyfesolver_results.formulas import beam_von_mises
    return beam_von_mises(record)


def _derived_measure(name: str, record: Dict[str, Any]) -> Any:
    """Compute one derived measure from the compiled canonical linear fields."""

    if name in PRINCIPAL_COMPONENTS:
        return shell_principal_stresses(record, name.removeprefix("principal_stresses_"))
    if name == "von_mises":
        if all(key in record for key in (
            "membrane_xx", "membrane_yy", "membrane_xy",
            "bending_xx", "bending_yy", "bending_xy",
            "shear_xz", "shear_yz",
        )):
            return _shell_surface_von_mises(record)
        if all(key in record for key in (
            "axial_stress", "bending_stress_y", "bending_stress_z",
            "shear_stress_y", "shear_stress_z", "torsional_stress",
        )):
            return _beam_von_mises(record)
        raise KeyError("von_mises")
    if name == "equivalent_stress":
        return np.asarray(record["von_mises"], dtype=float).copy()
    if name == "hill_utilization":
        return np.zeros_like(np.asarray(record["von_mises"], dtype=float))
    raise KeyError(name)


def _derived_measure_inputs(name: str) -> Tuple[str, ...]:
    if name == "von_mises":
        return (
            "membrane_xx", "membrane_yy", "membrane_xy",
            "bending_xx", "bending_yy", "bending_xy",
            "shear_xz", "shear_yz",
            "axial_stress", "bending_stress_y", "bending_stress_z",
            "shear_stress_y", "shear_stress_z", "torsional_stress",
        )
    if name == "equivalent_stress":
        return ("von_mises",)
    if name == "hill_utilization":
        return ("von_mises",)
    return ()


# Derived measures are evaluated in this order so that dependents
# (equivalent_stress, hill_utilization) see the already computed von Mises.
_DERIVED_ORDER: Tuple[str, ...] = ("von_mises", "equivalent_stress", "hill_utilization", *PRINCIPAL_COMPONENTS)


def _ordered_derived(names: Mapping[str, str]) -> Tuple[str, ...]:
    return tuple(name for name in _DERIVED_ORDER if name in names)


_VERIFICATION_RECIPE_TILE = 64
_VERIFICATION_PENDING_BYTES = 4 * 1024**2


def _verification_storage_bytes(value):
    """Conservative incoming copy/storage bound, without making that copy."""
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:
            raise PreparedStaticCapabilityError('verification record contains object arrays')
        return sys.getsizeof(value)+2*value.nbytes+128
    if isinstance(value,Mapping):
        size=0
        count=0
        for key,item in value.items():
            size+=_verification_storage_bytes(key)+_verification_storage_bytes(item)
            count+=1
        # A Mapping subclass can be smaller than its normalized plain dict.
        return max(sys.getsizeof(value)+40,sys.getsizeof({})+128*count)+size
    if type(value) in (tuple,list):
        return sys.getsizeof(value)+sum(_verification_storage_bytes(item) for item in value)
    if value is None or type(value) in (bool,int,float,str,bytes) or isinstance(value,np.generic):
        return sys.getsizeof(value)
    raise PreparedStaticCapabilityError('verification record contains nonplain values')


def _freeze_verification_value(value):
    """Alias-free source dtype/shape/bytes; never retain mutable native caches."""
    if isinstance(value,np.ndarray):
        array=np.asarray(value)
        return np.frombuffer(array.tobytes(order='C'),dtype=array.dtype).reshape(array.shape)
    if isinstance(value,Mapping):
        return MappingProxyType({key:_freeze_verification_value(item) for key,item in value.items()})
    if type(value) in (tuple,list):
        return tuple(_freeze_verification_value(item) for item in value)
    return value


def _check_verification_memory(memory_base,extra,resource_config,timing):
    required=int(memory_base.total_bytes_estimate)+int(extra)
    timing['pending_copy_peak_bytes']=max(timing['pending_copy_peak_bytes'],int(extra))
    estimate=MemoryEstimate(**{**vars(memory_base),'total_bytes_estimate':required})
    enforce_memory_limit(estimate,resource_config,context='prepared_static.recovery_verification_staging')


def _compare_owner_scale(recipe,local_check,oracle_check,element_id,amplitude):
    """Original source comparisons and tolerances, on frozen native operands."""
    owned_check = _recover_owned_local(recipe,local_check)
    for component,expected_value in oracle_check.items():
        if component in PRINCIPAL_COMPONENTS:
            owned_check[component] = _derived_measure(component,owned_check)
        actual_value = owned_check[component]
        expected_array = _array_value(expected_value)
        if expected_array is not None:
            if not np.allclose(actual_value,expected_array,rtol=1e-12,atol=1e-12):
                raise PreparedStaticCapabilityError('owned canonical recovery failed multiple-scale verification',diagnostics={'element_id':int(element_id),'component':component,'scale':amplitude})
        elif not _constant_equal(actual_value,expected_value):
            raise PreparedStaticCapabilityError('owned canonical recovery metadata changed')


def _verify_staged_recipes(pending,compiled,refused,timing,*,_publication_authority=None):
    """No live model/config/caller observation; publish only after full scope exit."""
    accepted=[]
    if not _callable_authority_matches(_publication_authority[0],_publication_authority[1]):
        raise PreparedStaticCapabilityError('recovery publication checker authority changed')
    _publication_authority[0]()
    started=_recovery_preparation_clock_ns()
    with owned_recovery_scope(tuple(entry[0].local_recipe for entry in pending)):
        entered=_recovery_preparation_clock_ns()
        timing['verification_scope_entry_ns']+=entered-started
        for candidate,checks,_size in pending:
            try:
                for amplitude,local,oracle in checks:
                    _compare_owner_scale(candidate.local_recipe,local,oracle,candidate.element_id,amplitude)
                    timing['owner_verification_calls']+=1
                accepted.append(candidate)
            except PreparedStaticCapabilityError as error:
                refused.append(dict(element_id=candidate.element_id,formulation=candidate.formulation,reason=str(error)))
            except Exception as error:
                from .control import SolveCancelled
                from .recovery import ResourcePolicyError
                if isinstance(error,(SolveCancelled,ResourcePolicyError)):
                    raise
                refused.append(dict(element_id=candidate.element_id,formulation=candidate.formulation,
                    reason='recovery oracle probe failed: '+str(error),original_exception_type=type(error).__name__))
        finished=_recovery_preparation_clock_ns()
        timing['owner_verification_ns']+=finished-entered
    timing['verification_scope_exit_ns']+=_recovery_preparation_clock_ns()-finished
    timing['verification_scope_count']+=1
    timing['verification_recipe_max']=max(timing['verification_recipe_max'],len(pending))
    if not _callable_authority_matches(_publication_authority[0],_publication_authority[1]):
        raise PreparedStaticCapabilityError('recovery publication checker authority changed')
    _publication_authority[0]()
    for candidate in accepted:
        compiled[candidate.element_id]=candidate
        timing['admitted_elements']+=1


def compile_affine_recovery(
    model: "FEModel",
    recovery: "RecoveryConfig",
    selected_element_ids: Sequence[int],
    *, _preparation_timings=None, _resource_config=None, _publication_authority=None,
) -> PreparedRecoveryCompilation:
    """Compile recovery maps for the selected, admitted elements."""

    import time

    from .analysis_session import _immutable_array

    started = time.perf_counter()
    if not _callable_authority_matches(_publication_authority[0],_publication_authority[1]):
        raise PreparedStaticCapabilityError('recovery publication checker authority changed')
    _publication_authority[0]()
    if _preparation_timings is not None and type(_preparation_timings) is not dict:
        raise PreparedStaticCapabilityError('preparation timing sink must be a source-owned plain dict')
    timing = dict(admission_ns=0, global_probe_allocation_ns=0, native_probe_ns=0,
                  schema_ns=0, affine_validation_ns=0, owner_capture_ns=0,
                  owner_verification_ns=0, publication_ns=0, initial_authority_ns=0,
                  admitted_elements=0, native_probe_calls=0, owner_verification_calls=0,
                  verification_scope_entry_ns=0,verification_scope_exit_ns=0,
                  verification_scope_count=0,verification_recipe_max=0,
                  pending_copy_peak_bytes=0,original_verification_fallback_elements=0)
    memory_base=estimate_model_memory(model,num_rhs=0,recovery_config=recovery)
    pending=[]
    pending_bytes=0
    flush_failed=False
    def flush_pending():
        nonlocal pending_bytes,flush_failed
        if pending:
            try:
                _verify_staged_recipes(pending,compiled,refused,timing)
            except Exception:
                flush_failed=True
                raise
            finally:
                pending.clear()
                pending_bytes=0
    total_dofs = int(model.mesh.dof_manager.total_dofs)
    compiled: Dict[int, CompiledElementRecovery] = {}
    retained = 0
    probes = 0
    refused: List[Dict[str, Any]] = []
    phase_started = _recovery_preparation_clock_ns()
    require_native_source_authority()
    if not _callable_authority_matches(_canonical_one_element_stress, _PROBE_HELPER_AUTHORITY):
        raise PreparedStaticCapabilityError("canonical element recovery helper authority changed")
    _authenticate_trusted_capture()
    timing['initial_authority_ns'] = _recovery_preparation_clock_ns()-phase_started
    for element_id in selected_element_ids:
        element = model.mesh.elements.get(int(element_id))
        if element is None:
            continue
        phase_started = _recovery_preparation_clock_ns()
        admitted, formulation, authority = _recovery_admission(model, element, recovery)
        timing['admission_ns'] += _recovery_preparation_clock_ns()-phase_started
        if not admitted:
            refused.append({
                "element_id": int(element_id),
                "formulation": formulation,
                "reason": authority,
            })
            continue
        dof_mapping = _immutable_array(
            np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp).reshape(-1)
        )
        local_size = int(dof_mapping.size)
        try:
            phase_started = _recovery_preparation_clock_ns()
            baseline_probe = np.zeros(total_dofs, dtype=float)
            timing['global_probe_allocation_ns'] += _recovery_preparation_clock_ns()-phase_started
            phase_started = _recovery_preparation_clock_ns()
            baseline = _probe_element_stress(model, baseline_probe,int(element_id), recovery)
            del baseline_probe
            timing['native_probe_ns'] += _recovery_preparation_clock_ns()-phase_started
            timing['native_probe_calls'] += 1
            probes += 1
            phase_started = _recovery_preparation_clock_ns()
            names = sorted(baseline)
            constants: Dict[str, Any] = {}
            arrays: Dict[str, np.ndarray] = {}
            shapes: Dict[str, Tuple[int, ...]] = {}
            unclassifiable: List[str] = []
            for name in names:
                value = baseline[name]
                content = _array_value(value)
                if content is not None:
                    arrays[name] = content.reshape(-1)
                    shapes[name] = tuple(int(size) for size in content.shape)
                    continue
                try:
                    constants[name] = _plain_constant(value)
                except ValueError:
                    unclassifiable.append(name)
            if unclassifiable:
                raise PreparedStaticCapabilityError(
                    "recovery components cannot be represented as plain "
                    "constants or float fields: " + ", ".join(unclassifiable),
                    diagnostics={"stage": "recovery compilation",
                                 "element_id": int(element_id),
                                 "components": unclassifiable},
                )
            columns: Dict[str, List[np.ndarray]] = {name: [] for name in arrays}
            timing['schema_ns'] += _recovery_preparation_clock_ns()-phase_started
            for local_index in range(local_size):
                phase_started = _recovery_preparation_clock_ns()
                probe = np.zeros(total_dofs, dtype=float)
                probe[dof_mapping[local_index]] = 1.0
                timing['global_probe_allocation_ns'] += _recovery_preparation_clock_ns()-phase_started
                phase_started = _recovery_preparation_clock_ns()
                observed = _probe_element_stress(model, probe, int(element_id), recovery)
                timing['native_probe_ns'] += _recovery_preparation_clock_ns()-phase_started
                timing['native_probe_calls'] += 1
                probes += 1
                phase_started = _recovery_preparation_clock_ns()
                if set(observed) != set(baseline):
                    raise PreparedStaticCapabilityError(
                        "recovery oracle changed its component set under local probes",
                        diagnostics={"stage": "recovery compilation",
                                     "element_id": int(element_id)},
                    )
                for name, value in constants.items():
                    if not _constant_equal(value, observed[name]):
                        raise PreparedStaticCapabilityError(
                            "recovery component is not constant under local probes",
                            diagnostics={"stage": "recovery compilation",
                                         "element_id": int(element_id),
                                         "component": str(name)},
                        )
                for name in arrays:
                    content = _array_value(observed[name])
                    if (content is None
                            or tuple(int(size) for size in content.shape) != shapes[name]):
                        raise PreparedStaticCapabilityError(
                            "recovery component changed shape under local probes",
                            diagnostics={"stage": "recovery compilation",
                                         "element_id": int(element_id),
                                         "component": str(name)},
                        )
                    columns[name].append(content.reshape(-1) - arrays[name])
                timing['schema_ns'] += _recovery_preparation_clock_ns()-phase_started
            rng = np.random.default_rng(int(element_id))
            random_local = rng.uniform(-1.0, 1.0, local_size)
            phase_started = _recovery_preparation_clock_ns()
            random_probe = np.zeros(total_dofs, dtype=float)
            random_probe[dof_mapping] = random_local
            timing['global_probe_allocation_ns'] += _recovery_preparation_clock_ns()-phase_started
            phase_started = _recovery_preparation_clock_ns()
            observed = _probe_element_stress(model, random_probe, int(element_id), recovery)
            timing['native_probe_ns'] += _recovery_preparation_clock_ns()-phase_started
            timing['native_probe_calls'] += 1
            probes += 1
            phase_started = _recovery_preparation_clock_ns()
            linear: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            frozen_constants = dict(constants)
            derived: Dict[str, str] = {}
            verification = 0.0
            for name in sorted(arrays):
                matrix = (
                    np.column_stack(columns[name]) if columns[name]
                    else np.zeros((arrays[name].size, 0))
                )
                expected = arrays[name] + (
                    matrix @ random_local if matrix.size else 0.0
                )
                made = _array_value(observed[name])
                if made is None or made.shape != arrays[name].reshape(shapes[name]).shape:
                    raise PreparedStaticCapabilityError(
                        "recovery component changed shape under the random probe",
                        diagnostics={"stage": "recovery compilation",
                                     "element_id": int(element_id),
                                     "component": str(name)},
                    )
                made = made.reshape(-1)
                scale = max(
                    float(np.linalg.norm(expected)), float(np.linalg.norm(made)), 1.0
                )
                residual = float(np.linalg.norm(made - expected) / scale)
                if name in {"von_mises", "equivalent_stress", "hill_utilization", *PRINCIPAL_COMPONENTS}:
                    derived[name] = "derived"
                    continue
                if np.isfinite(residual) and residual <= _AFFINITY_RELATIVE_TOLERANCE:
                    if not matrix.size or not np.any(matrix):
                        frozen_constants[name] = _immutable_array(
                            arrays[name].reshape(shapes[name])
                        )
                    else:
                        linear[name] = (_immutable_array(arrays[name]),
                                        _immutable_array(matrix))
                    verification = max(verification, residual)
                    continue
                derived[name] = "derived"
            record_probe: Dict[str, Any] = {}
            for name in sorted(arrays):
                if name in linear:
                    base, matrix = linear[name]
                    record_probe[name] = (base + matrix @ random_local).reshape(
                        shapes[name]
                    )
                elif name in frozen_constants and name not in derived:
                    record_probe[name] = frozen_constants[name]
            for name in _ordered_derived(derived):
                try:
                    value = _derived_measure(name, record_probe)
                except KeyError:
                    refused.append({
                        "element_id": int(element_id),
                        "formulation": formulation,
                        "reason": (
                            f"derived measure {name!r} has no reviewed canonical "
                            "computation from the compiled linear fields"
                        ),
                    })
                    derived = {}
                    break
                record_probe[name] = value
                made = _array_value(observed[name])
                if made is None:
                    refused.append({
                        "element_id": int(element_id),
                        "formulation": formulation,
                        "reason": f"derived measure {name!r} changed type",
                    })
                    derived = {}
                    break
                expected = np.asarray(value, dtype=float)
                scale = max(
                    float(np.linalg.norm(expected)),
                    float(np.linalg.norm(made)),
                    1.0,
                )
                residual = float(
                    np.linalg.norm(np.asarray(made, dtype=float).reshape(-1)
                                   - expected.reshape(-1)) / scale
                )
                if not np.isfinite(residual) or residual > _AFFINITY_RELATIVE_TOLERANCE:
                    refused.append({
                        "element_id": int(element_id),
                        "formulation": formulation,
                        "reason": (
                            f"derived measure {name!r} does not reproduce the "
                            "recovery oracle"
                        ),
                        "verification_relative_residual": residual,
                    })
                    derived = {}
                    break
                verification = max(verification, residual)
            if not derived and any(name not in linear and name not in frozen_constants
                                   for name in arrays):
                continue
            array_components = []
            for name in sorted(linear):
                base, matrix = linear[name]
                array_components.append((name, base, matrix, shapes[name]))
                retained += int(matrix.nbytes + base.nbytes)
            constant_items = []
            for name in sorted(frozen_constants):
                if name in linear:
                    continue
                value = frozen_constants[name]
                if isinstance(value, np.ndarray):
                    retained += int(value.nbytes)
                constant_items.append((name, value))
            timing['affine_validation_ns'] += _recovery_preparation_clock_ns()-phase_started
            phase_started = _recovery_preparation_clock_ns()
            local_recipe = _capture_owned_local_recovery(model, element)
            timing['owner_capture_ns'] += _recovery_preparation_clock_ns()-phase_started
            scale_checks=[]
            scale_bytes=0
            immediate=False
            # Check owned canonical arithmetic at distinct scales and under a
            # rigid displacement offset; affinity probes alone are insufficient.
            for amplitude in (1e-8, 1e-3):
                local_check = random_local * amplitude
                local_check[0::6] += .125
                phase_started = _recovery_preparation_clock_ns()
                global_check = np.zeros(total_dofs)
                global_check[dof_mapping] = local_check
                timing['global_probe_allocation_ns'] += _recovery_preparation_clock_ns()-phase_started
                phase_started = _recovery_preparation_clock_ns()
                oracle_check = _probe_element_stress(model, global_check, int(element_id), recovery)
                timing['native_probe_ns'] += _recovery_preparation_clock_ns()-phase_started
                timing['native_probe_calls'] += 1
                incoming=_verification_storage_bytes(oracle_check)+_verification_storage_bytes(local_check)+128
                if not _callable_authority_matches(_publication_authority[0],_publication_authority[1]):
                    raise PreparedStaticCapabilityError('recovery publication checker authority changed')
                _publication_authority[0]()
                # Account scratch while older immutable entries remain live. Freeze
                # before any private verification can mutate a native shared cache.
                _check_verification_memory(memory_base,pending_bytes+scale_bytes+2*incoming,_resource_config,timing)
                saved_oracle=_freeze_verification_value(oracle_check)
                saved_local=_freeze_verification_value(local_check)
                if not _callable_authority_matches(_publication_authority[0],_publication_authority[1]):
                    raise PreparedStaticCapabilityError('recovery publication checker authority changed')
                _publication_authority[0]()
                if pending_bytes+scale_bytes+incoming>_VERIFICATION_PENDING_BYTES:
                    flush_pending()
                if immediate or scale_bytes+incoming>_VERIFICATION_PENDING_BYTES:
                    # Preserve unbounded/default caller admission for a single
                    # oversized source record using original per-call authority.
                    if not immediate:
                        timing['original_verification_fallback_elements']+=1
                        for prior_amplitude,prior_local,prior_oracle in scale_checks:
                            _compare_owner_scale(local_recipe,prior_local,prior_oracle,element_id,prior_amplitude)
                            timing['owner_verification_calls']+=1
                        scale_checks.clear()
                        scale_bytes=0
                    phase_started=_recovery_preparation_clock_ns()
                    _compare_owner_scale(local_recipe,saved_local,saved_oracle,element_id,amplitude)
                    timing['owner_verification_ns']+=_recovery_preparation_clock_ns()-phase_started
                    timing['owner_verification_calls']+=1
                    immediate=True
                else:
                    scale_checks.append((amplitude,saved_local,saved_oracle))
                    scale_bytes+=incoming
                probes += 1
            phase_started = _recovery_preparation_clock_ns()
            candidate = CompiledElementRecovery(
                element_id=int(element_id),
                formulation=formulation,
                admitted_by=authority,
                dof_mapping=dof_mapping,
                array_components=tuple(array_components),
                constant_components=tuple(constant_items),
                derived_components=tuple(
                    (name, derived[name]) for name in _ordered_derived(derived)
                ),
                verification_relative_residual=float(verification),
                local_recipe=local_recipe,
            )
            if immediate:
                compiled[int(element_id)]=candidate
                timing['admitted_elements']+=1
            else:
                pending.append((candidate,tuple(scale_checks),scale_bytes))
                pending_bytes+=scale_bytes
                if len(pending)>=_VERIFICATION_RECIPE_TILE:
                    flush_pending()
            timing['publication_ns'] += _recovery_preparation_clock_ns()-phase_started
        except PreparedStaticCapabilityError as error:
            if flush_failed:
                raise
            refused.append({
                "element_id": int(element_id),
                "formulation": formulation,
                "reason": str(error),
            })
            continue
        except Exception as error:
            from .control import SolveCancelled

            if flush_failed or isinstance(error, (SolveCancelled,ResourcePolicyError)):
                raise
            refused.append({
                "element_id": int(element_id),
                "formulation": formulation,
                "reason": "recovery oracle probe failed: " + str(error),
                "original_exception_type": type(error).__name__,
            })
            continue
    flush_pending()
    if not _callable_authority_matches(_publication_authority[0],_publication_authority[1]):
        raise PreparedStaticCapabilityError('recovery publication checker authority changed')
    _publication_authority[0]()
    if refused:
        raise PreparedStaticCapabilityError(
            "stress recovery was requested for elements outside the source-reviewed "
            "affine families; restrict RecoveryConfig.element_ids or disable "
            "include_stresses",
            diagnostics={"stage": "recovery compilation", "refused": refused},
        )
    retained += sum(item.local_recipe.retained_bytes for item in compiled.values())
    if _preparation_timings is not None:
        _preparation_timings.update(timing)
    return PreparedRecoveryCompilation(
        elements=compiled,
        retained_bytes=int(retained),
        probe_count=int(probes),
        setup_seconds=float(time.perf_counter() - started),
    )


def apply_compiled_recovery(
    compilation: PreparedRecoveryCompilation,
    displacements: np.ndarray,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Apply the compiled maps and derived measures to one displacement field."""

    made = np.asarray(displacements, dtype=float).reshape(-1)
    stresses: Dict[int, Dict[str, Any]] = {}
    worst = 0.0
    refined = []
    for element_id, element in compilation.elements.items():
        local = made[element.dof_mapping]
        record: Dict[str, Any] = {}
        for name, baseline, matrix, shape in element.array_components:
            values = baseline + matrix @ local
            record[name] = values.reshape(shape) if shape else float(values.reshape(()))
        for name, value in element.constant_components:
            record[name] = (
                np.asarray(value).copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
            )
        principal_requested = any(name in PRINCIPAL_COMPONENTS for name, _ in element.derived_components)
        cancellation_sensitive = any(np.any(64 * np.finfo(float).eps * (np.abs(matrix) @ np.abs(local)) > 1e-12 + 1e-12 * np.abs(np.asarray(record[name]).reshape(-1))) for name, _base, matrix, _shape in element.array_components)
        canonical = None
        if element.local_recipe is not None:
            canonical = _recover_owned_local(element.local_recipe, local)
            refined.append(int(element_id))
            # Preserve only the admitted requested/dependency schema.
            for name in tuple(record):
                if name in canonical:
                    record[name] = copy.deepcopy(canonical[name])
        for name, _kind in element.derived_components:
            record[name] = (copy.deepcopy(canonical[name]) if canonical is not None and name in canonical
                            else _derived_measure(name, record))
        stresses[int(element_id)] = record
        worst = max(worst, element.verification_relative_residual)
    provenance = {
        "method": "prepared_affine_zero_local_basis_with_derived_measures",
        "compiled_elements": len(compilation.elements),
        "probe_count": int(compilation.probe_count),
        "worst_verification_relative_residual": float(worst),
        "setup_seconds": float(compilation.setup_seconds),
    }
    execution = {
        "prepared_affine_recovery": True,
        "compiled_elements": len(compilation.elements),
        "oracle_probes_at_preparation": int(compilation.probe_count),
        "canonical_local_refined_elements": refined,
        "prepared_q4_staged_elements": [eid for eid, item in compilation.elements.items() if item.local_recipe.staged_kernel is not None],
    }
    return stresses, provenance, execution


@dataclass(frozen=True)
class ReactionMaps:
    """Immutable data needed to reproduce constraint-force diagnostics."""

    fixed_dofs: Tuple[int, ...]
    mpc_constraints: Tuple[Mapping[str, Any], ...]
    dof_info: Mapping[int, Tuple[int, int, str]]
    retained_bytes: int


def capture_reaction_maps(model: "FEModel") -> ReactionMaps:
    """Capture fixed DOFs, MPC records and DOF-to-node info."""

    from .assembly import _collect_mpc_constraints

    mesh = model.mesh
    dof_manager = mesh.dof_manager
    fixed = sorted(int(dof) for dof in getattr(dof_manager, "_constrained_dofs", set()))
    constraints: List[Mapping[str, Any]] = []
    retained = 0
    for constraint in _collect_mpc_constraints(model):
        made = {
            "slave": int(constraint["slave"]),
            "masters": {int(dof): float(value) for dof, value in constraint["masters"].items()},
            "value": float(constraint["value"]),
            "label": str(constraint["label"]),
            "kind": str(constraint.get("kind", "mpc")),
        }
        constraints.append(made)
        retained += 64 + 24 * len(made["masters"])
    needed = set(fixed)
    for constraint in constraints:
        needed.add(int(constraint["slave"]))
        needed.update(int(dof) for dof in constraint["masters"])
    dof_info: Dict[int, Tuple[int, int, str]] = {}
    for dof in sorted(needed):
        node_id, local_index, component = dof_manager.get_dof_info(int(dof))
        dof_info[int(dof)] = (int(node_id), int(local_index), str(component))
        retained += 64
    return ReactionMaps(
        fixed_dofs=tuple(fixed),
        mpc_constraints=tuple(constraints),
        dof_info=dof_info,
        retained_bytes=int(retained),
    )


def _add_dof_force(
    force_map: Dict[int, np.ndarray],
    dof_info: Mapping[int, Tuple[int, int, str]],
    dof: int,
    value: float,
) -> None:
    node_id, local_index, _component = dof_info[int(dof)]
    if node_id < 0 or local_index < 0:
        return
    force_map.setdefault(int(node_id), np.zeros(6, dtype=float))[int(local_index)] += float(value)


def _compact_force_map(force_map: Mapping[int, np.ndarray]) -> Dict[int, np.ndarray]:
    compact: Dict[int, np.ndarray] = {}
    for node_id, values in force_map.items():
        vector = np.asarray(values, dtype=float).reshape(6)
        if np.any(vector != 0.0):
            compact[int(node_id)] = vector
    return compact


def constraint_force_diagnostics(
    reaction_maps: ReactionMaps,
    stiffness: Any,
    transformation: Any,
    rigid_modes: np.ndarray,
    displacements: np.ndarray,
    external_load: np.ndarray,
    *,
    constraint_info: Mapping[str, Any],
    nullspace_info: Mapping[str, Any],
    metadata_copies: Any = None,
) -> Dict[str, Any]:
    """Reproduce the live constraint-force diagnostics from the snapshot.

    ``metadata_copies`` (M20) may supply a callable returning fresh independent
    copies of (constraint_info, nullspace_info); the default deep-copies them.
    """

    u = np.asarray(displacements, dtype=float).reshape(-1)
    F_ext = np.asarray(external_load, dtype=float).reshape(-1)
    residual = np.asarray(stiffness @ u - F_ext, dtype=float).reshape(-1)
    fixed_dofs = [int(dof) for dof in reaction_maps.fixed_dofs]

    support_reactions: Dict[int, np.ndarray] = {}
    mpc_slave_forces: Dict[int, np.ndarray] = {}
    mpc_master_equivalent_forces: Dict[int, np.ndarray] = {}
    mpc_constraint_forces: List[Dict[str, Any]] = []

    for dof in fixed_dofs:
        _add_dof_force(support_reactions, reaction_maps.dof_info, dof, residual[dof])

    for index, constraint in enumerate(reaction_maps.mpc_constraints):
        slave = int(constraint["slave"])
        slave_force = float(residual[slave])
        slave_node, slave_local, slave_component = reaction_maps.dof_info[slave]
        _add_dof_force(mpc_slave_forces, reaction_maps.dof_info, slave, slave_force)
        master_entries: List[Dict[str, Any]] = []
        for master, coefficient in constraint["masters"].items():
            master_dof = int(master)
            value = float(coefficient) * slave_force
            _add_dof_force(mpc_master_equivalent_forces, reaction_maps.dof_info, master_dof, value)
            master_node, master_local, master_component = reaction_maps.dof_info[master_dof]
            master_entries.append({
                "dof": master_dof,
                "node_id": int(master_node),
                "local_index": int(master_local),
                "component": str(master_component),
                "coefficient": float(coefficient),
                "equivalent_force": value,
            })
        mpc_constraint_forces.append({
            "index": int(index),
            "label": str(constraint["label"]),
            "slave_dof": slave,
            "slave_node_id": int(slave_node),
            "slave_local_index": int(slave_local),
            "slave_component": str(slave_component),
            "slave_force": slave_force,
            "master_equivalent_forces": master_entries,
            "master_equivalent_norm": float(
                np.linalg.norm([entry["equivalent_force"] for entry in master_entries])
            ),
        })

    reduced_residual = np.asarray(transformation.T @ residual, dtype=float).reshape(-1)
    if rigid_modes.shape[1] > 0:
        nullspace_generalized_forces = np.asarray(
            rigid_modes.T @ reduced_residual, dtype=float
        ).reshape(-1)
    else:
        nullspace_generalized_forces = np.zeros(0, dtype=float)

    return {
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "reduced_residual_norm": float(np.linalg.norm(reduced_residual)),
        "fixed_dofs": fixed_dofs,
        "mpc_slave_dofs": sorted(
            int(constraint["slave"]) for constraint in reaction_maps.mpc_constraints
        ),
        "support_reactions": _compact_force_map(support_reactions),
        "mpc_slave_forces": _compact_force_map(mpc_slave_forces),
        "mpc_master_equivalent_forces": _compact_force_map(mpc_master_equivalent_forces),
        "mpc_constraint_forces": mpc_constraint_forces,
        "support_reaction_norm": float(
            np.linalg.norm(np.concatenate(list(support_reactions.values())))
            if support_reactions else 0.0
        ),
        "mpc_slave_force_norm": float(
            np.linalg.norm(np.concatenate(list(mpc_slave_forces.values())))
            if mpc_slave_forces else 0.0
        ),
        "mpc_master_equivalent_force_norm": float(
            np.linalg.norm(np.concatenate(list(mpc_master_equivalent_forces.values())))
            if mpc_master_equivalent_forces else 0.0
        ),
        "nullspace_generalized_forces": nullspace_generalized_forces,
        "nullspace_generalized_force_norm": float(np.linalg.norm(nullspace_generalized_forces)),
        **_metadata_entries(constraint_info, nullspace_info, metadata_copies),
    }


def _metadata_entries(constraint_info, nullspace_info, metadata_copies):
    if metadata_copies is None:
        return {"constraint_info": copy.deepcopy(dict(constraint_info)),
                "nullspace_info": copy.deepcopy(dict(nullspace_info))}
    made_constraint, made_nullspace = metadata_copies()
    return {"constraint_info": made_constraint, "nullspace_info": made_nullspace}


def constraint_force_maps(reaction_maps: ReactionMaps, stiffness: Any, displacements: np.ndarray,
                          external_load: np.ndarray) -> Dict[str, Dict[int, np.ndarray]]:
    """M20: only the three force maps, with the arithmetic of constraint_force_diagnostics."""
    u = np.asarray(displacements, dtype=float).reshape(-1)
    F_ext = np.asarray(external_load, dtype=float).reshape(-1)
    residual = np.asarray(stiffness @ u - F_ext, dtype=float).reshape(-1)
    support: Dict[int, np.ndarray] = {}
    slave_forces: Dict[int, np.ndarray] = {}
    master_forces: Dict[int, np.ndarray] = {}
    for dof in reaction_maps.fixed_dofs:
        dof = int(dof)
        _add_dof_force(support, reaction_maps.dof_info, dof, residual[dof])
    for constraint in reaction_maps.mpc_constraints:
        slave = int(constraint["slave"])
        slave_force = float(residual[slave])
        _add_dof_force(slave_forces, reaction_maps.dof_info, slave, slave_force)
        for master, coefficient in constraint["masters"].items():
            _add_dof_force(master_forces, reaction_maps.dof_info, int(master), float(coefficient) * slave_force)
    return {"support_reactions": _compact_force_map(support),
            "mpc_slave_forces": _compact_force_map(slave_forces),
            "mpc_master_equivalent_forces": _compact_force_map(master_forces)}



class ReactionPlan:
    """M22: index arrays and record templates for constraint_force_diagnostics.

    planned_constraint_force_diagnostics reproduces constraint_force_diagnostics
    exactly: same residual, the same per-slot accumulation order (ufunc.at is
    unbuffered and applies repeated indices in order), first-touch node order,
    identical record key order and Python float values, and the same norm calls.
    """

    def __init__(self, reaction_maps):
        dof_info = reaction_maps.dof_info

        def slots(dofs):
            order, position, rows, cols, keep = [], {}, [], [], []
            for dof in dofs:
                node, local, _component = dof_info[int(dof)]
                if node < 0 or local < 0:
                    keep.append(False)
                    continue
                if int(node) not in position:
                    position[int(node)] = len(order)
                    order.append(int(node))
                rows.append(position[int(node)])
                cols.append(int(local))
                keep.append(True)
            return (tuple(order), np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp),
                    np.asarray(keep, dtype=bool))

        self.fixed_dofs = tuple(int(dof) for dof in reaction_maps.fixed_dofs)
        self.fixed_index = np.asarray(self.fixed_dofs, dtype=np.intp)
        self.support = slots(self.fixed_dofs)
        slaves = [int(constraint["slave"]) for constraint in reaction_maps.mpc_constraints]
        self.slave_index = np.asarray(slaves, dtype=np.intp)
        self.slave = slots(slaves)
        self.mpc_slave_dofs = tuple(sorted(slaves))
        master_dofs, coefficients, owners, templates, ranges = [], [], [], [], []
        for index, constraint in enumerate(reaction_maps.mpc_constraints):
            slave = int(constraint["slave"])
            slave_node, slave_local, slave_component = dof_info[slave]
            start = len(master_dofs)
            entries = []
            for master, coefficient in constraint["masters"].items():
                master_dof = int(master)
                master_node, master_local, master_component = dof_info[master_dof]
                master_dofs.append(master_dof)
                coefficients.append(float(coefficient))
                owners.append(index)
                entries.append({"dof": master_dof, "node_id": int(master_node), "local_index": int(master_local),
                                "component": str(master_component), "coefficient": float(coefficient)})
            ranges.append((start, len(master_dofs)))
            templates.append(({"index": int(index), "label": str(constraint["label"]), "slave_dof": slave,
                               "slave_node_id": int(slave_node), "slave_local_index": int(slave_local),
                               "slave_component": str(slave_component)}, tuple(entries)))
        self.master = slots(master_dofs)
        self.master_coefficients = np.asarray(coefficients, dtype=float)
        self.master_owners = np.asarray(owners, dtype=np.intp)
        self.templates = tuple(templates)
        self.ranges = tuple(ranges)


def _planned_map(spec, values):
    order, rows, cols, keep = spec
    made = np.zeros((len(order), 6), dtype=float)
    np.add.at(made, (rows, cols), np.asarray(values, dtype=float)[keep])
    return order, made


def _planned_norm(order, made):
    if not order:
        return 0.0
    return float(np.linalg.norm(np.concatenate([made[i] for i in range(len(order))])))


def _planned_compact(order, made):
    # M30: one vectorised non-zero test per map instead of one per node; the same
    # rows are copied in the same order.
    keep = np.any(made != 0.0, axis=1)
    return {node: made[i].copy() for i, node in enumerate(order) if keep[i]}


def planned_constraint_force_diagnostics(plan, stiffness, transformation, rigid_modes, displacements, external_load,
                                         *, constraint_info, nullspace_info, metadata_copies=None, maps_only=False):
    u = np.asarray(displacements, dtype=float).reshape(-1)
    F_ext = np.asarray(external_load, dtype=float).reshape(-1)
    residual = np.asarray(stiffness @ u - F_ext, dtype=float).reshape(-1)
    support = _planned_map(plan.support, residual[plan.fixed_index])
    slave_values = residual[plan.slave_index]
    slave = _planned_map(plan.slave, slave_values)
    master_values = plan.master_coefficients * slave_values[plan.master_owners]
    master = _planned_map(plan.master, master_values)
    maps = {"support_reactions": _planned_compact(*support),
            "mpc_slave_forces": _planned_compact(*slave),
            "mpc_master_equivalent_forces": _planned_compact(*master)}
    if maps_only:
        return maps
    slave_floats = slave_values.tolist()
    master_floats = master_values.tolist()
    records = []
    for (head, entries), (start, stop), slave_force in zip(plan.templates, plan.ranges, slave_floats):
        master_entries = [dict(entry, equivalent_force=value)
                          for entry, value in zip(entries, master_floats[start:stop])]
        records.append({**head, "slave_force": slave_force, "master_equivalent_forces": master_entries,
                        "master_equivalent_norm": float(np.linalg.norm(
                            [entry["equivalent_force"] for entry in master_entries]))})
    reduced_residual = np.asarray(transformation.T @ residual, dtype=float).reshape(-1)
    if rigid_modes.shape[1] > 0:
        nullspace_generalized_forces = np.asarray(rigid_modes.T @ reduced_residual, dtype=float).reshape(-1)
    else:
        nullspace_generalized_forces = np.zeros(0, dtype=float)
    return {
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "reduced_residual_norm": float(np.linalg.norm(reduced_residual)),
        "fixed_dofs": list(plan.fixed_dofs),
        "mpc_slave_dofs": list(plan.mpc_slave_dofs),
        **maps,
        "mpc_constraint_forces": records,
        "support_reaction_norm": _planned_norm(*support),
        "mpc_slave_force_norm": _planned_norm(*slave),
        "mpc_master_equivalent_force_norm": _planned_norm(*master),
        "nullspace_generalized_forces": nullspace_generalized_forces,
        "nullspace_generalized_force_norm": float(np.linalg.norm(nullspace_generalized_forces)),
        **_metadata_entries(constraint_info, nullspace_info, metadata_copies),
    }


# M25e review fix: the module check, built after every module-level import of this
# module (so prepared_static_q4 and the other span modules are included).
from types import FunctionType as _CaptureFunctionType
from itertools import repeat as _capture_repeat
from .fe_core import _QualifiedStateMapping as _CaptureStateMapping


def _make_capture_module_check():
    import sys
    import types
    from importlib.machinery import ModuleSpec
    families = ('numpy', 'math', 'cmath', 'warnings', 'sys', 'types', 'io', 'dataclasses', 'builtins',
                'functools', 'operator', 'contextlib', 'contextvars', 'copy', 'linecache', 'anyfesolver',
                'anymaterial', 'numba')  # scipy is used by neither the proofs nor the span code
    missing = object()
    held = [None]    # sys.modules, captured on the first opt-in admission
    unsafe = [False]  # sys.modules holds a key that is not an exact str
    fixed = []       # (name, module, kind, namespace, hook)
    late = []        # [name, module, kind, namespace, pinned hook or None, authority]
    uninspectable = []  # (name, object): namespace not readable without a Python-level descriptor
    seen = [0]

    def safe_namespace(obj):
        # M25i (opt-in review round 3): read an instance namespace without invoking any
        # Python-level descriptor. This runs on the first opt-in admission (M25k: never
        # at import) and must not call code from arbitrary sys.modules entries: every class in the MRO must
        # have the plain metaclass and define __dict__ only as a builtin descriptor and
        # __getattribute__ only as a builtin slot wrapper. None when that does not hold.
        # M25j (round 4): identity comparisons only (no __eq__ of a foreign metaclass),
        # and each builtin descriptor must belong to the class that defines it.
        kind = type(obj)
        if type(kind) is not type:
            return None
        for cls in type.__getattribute__(kind, '__mro__'):
            if type(cls) is not type:
                return None
            members = type.__getattribute__(cls, '__dict__')
            for key in members:  # M25k: exact str keys before any keyed lookup (no foreign __eq__)
                if type(key) is not str:
                    return None
            member = members.get('__dict__')
            if member is not None and not (
                    (type(member) is types.MemberDescriptorType or type(member) is types.GetSetDescriptorType)
                    and member.__objclass__ is cls):
                return None
            member = members.get('__getattribute__')
            if member is not None and not (type(member) is types.WrapperDescriptorType and member.__objclass__ is cls):
                return None
        try:
            namespace = object.__getattribute__(obj, '__dict__')
        except Exception:  # only builtin descriptors can run here (checked above)
            return None
        if type(namespace) is not dict:
            return None
        for key in namespace:  # M25k: exact str keys before any keyed lookup (no foreign __eq__)
            if type(key) is not str:
                return None
        return namespace

    def initializing(namespace):
        spec = dict.get(namespace, '__spec__')
        if type(spec) is not ModuleSpec:
            return False
        state = safe_namespace(spec)
        return type(state) is dict and dict.get(state, '_initializing') is True

    def track(name, module):
        namespace = safe_namespace(module)
        if namespace is None:
            # Not inspectable without running its own code: never read, and the opt-in
            # keeps the guarded route while this object is registered under that name.
            uninspectable.append((name, module))
            return
        if initializing(namespace):
            late.append([name, module, type(module), namespace, None, None])
        else:
            fixed.append((name, module, type(module), namespace, dict.get(namespace, '__getattr__', missing)))

    def current_modules():
        # M25k: sys.modules, read without a keyed lookup in a namespace that holds a
        # non-str key (such a key's __eq__ could run during the lookup).
        if type(sys) is not types.ModuleType:
            return None
        namespace = object.__getattribute__(sys, '__dict__')
        if type(namespace) is not dict:
            return None
        for key in namespace:
            if type(key) is not str:
                return None
        found = dict.get(namespace, 'modules')
        return found if type(found) is dict else None

    def scan(modules):
        tracked = ({id(entry[1]) for entry in fixed} | {id(entry[1]) for entry in late}
                   | {id(entry[1]) for entry in uninspectable})
        unsafe[0] = False
        for name, module in tuple(dict.items(modules)):
            if type(name) is not str:
                unsafe[0] = True
                continue
            if name.split('.', 1)[0] in families and module is not None and id(module) not in tracked:
                track(name, module)
        seen[0] = len(modules)
        return (tuple(entry[0] for entry in fixed), tuple(entry[1] for entry in fixed),
                tuple(entry[3] for entry in fixed), tuple(id(entry[1]) for entry in fixed),
                tuple(id(entry[2]) for entry in fixed), tuple(id(entry[4]) for entry in fixed))

    # M25k (opt-in review round 5): built on the first opt-in admission, not at import,
    # so the default path never inspects sys.modules.
    tables = [None]

    def canonical_hook(hook, namespace):
        file = dict.get(namespace, '__file__')
        return (type(hook) is _CaptureFunctionType and hook.__globals__ is namespace and type(file) is str
                and type(hook.__code__.co_filename) is str and hook.__code__.co_filename == file
                and hook.__code__.co_name == '__getattr__')

    def check():
        modules = current_modules()
        if modules is None:
            return False
        if held[0] is None:
            held[0] = modules
        if modules is not held[0]:
            return False
        if tables[0] is None or len(modules) != seen[0]:
            tables[0] = scan(modules)  # first opt-in use, then newly imported modules (pinned on first sight)
        if unsafe[0]:
            return False
        names, objects, namespaces, object_ids, type_ids, hook_ids = tables[0]
        for name, obj in uninspectable:
            if dict.get(modules, name) is obj:
                return False
        if (tuple(map(id, map(dict.get, _capture_repeat(modules), names))) != object_ids
                or tuple(map(id, map(type, objects))) != type_ids
                or tuple(map(id, map(dict.get, namespaces, _capture_repeat('__getattr__'),
                                     _capture_repeat(missing)))) != hook_ids):
            return False
        for entry in late:
            name, module, kind, namespace, pinned, authority = entry
            if dict.get(modules, name) is not module or type(module) is not kind:
                return False
            current = dict.get(namespace, '__getattr__', missing)
            if pinned is not None:
                if current is not pinned or (authority is not None and not _callable_authority_matches(current, authority)):
                    return False
                continue
            if current is not missing and not canonical_hook(current, namespace):
                return False
            if not initializing(namespace):
                # Finished executing: pin its hook (or its absence) with the code fingerprint.
                entry[4] = current
                entry[5] = None if current is missing else _callable_authority(current)
        return True

    check.__name__ = check.__qualname__ = '_capture_modules_unhooked'
    return check


_capture_modules_unhooked = _make_capture_module_check()


# M25e review fix: the LoadCase pressure helper the span runs, and the boundary
# functions it calls, with code and global bindings. A change keeps the guarded
# route (as a substituted helper did before M25) instead of refusing.
def _make_pressure_helper_check():
    # M25m: objects from import statements; the boundary module's globals dict (not
    # the module object) is the owner of its functions. M25n (opt-in review round 8):
    # the facts are built on the first check (opt-in only), never at import.
    from .boundary import LoadCase as load_case
    built = []

    def build():
        helper = type.__getattribute__(load_case, '__dict__')['_consistent_pressure_load']
        boundary_globals = helper.__globals__
        boundary_name = dict.get(boundary_globals, '__name__')
        made = [(load_case, '_consistent_pressure_load', helper)]
        for name in helper.__code__.co_names:
            value = dict.get(boundary_globals, name)
            if type(value) is _CaptureFunctionType and value.__module__ == boundary_name:
                made.append((boundary_globals, name, value))
        return tuple((owner, name, function, _callable_authority(function),
                      tuple((key, function.__globals__[key]) for key in function.__code__.co_names
                            if key in function.__globals__))
                     for owner, name, function in made)

    def check():
        if not built:
            built.append(build())
        for owner, name, function, authority, bindings in built[0]:
            namespace = type.__getattribute__(owner, '__dict__') if type(owner) is type else owner
            if (namespace.get(name) is not function or not _callable_authority_matches(function, authority)
                    or any(function.__globals__.get(key) is not value for key, value in bindings)):
                return False
        return True

    check.__name__ = check.__qualname__ = '_capture_pressure_helpers_authentic'
    return check


_capture_pressure_helpers_authentic = _make_pressure_helper_check()


# M25b review fix: the executables a capture span relies on (the capture itself,
# its admission helpers, the proof functions, this checker) with their global
# bindings. The facts live only in this closure, so no module binding can empty
# them; every capture caller runs the check before capturing.
def _make_capture_span_authority():
    # M25m (opt-in review round 7): the proof functions come from import statements and
    # their modules' globals dicts are the owners, so nothing here reads a module object.
    from .boundary import (_numpy_error_routes_are_builtin, _load_routing_classes_are_untouched,
                           _exact_load_case_instance_namespace)
    from .prepared_static_loads import _plain_numeric_state, _instance_authority
    from .static_batch import _class_state
    # M25l (opt-in review round 6): this module's own globals, not a sys.modules lookup.
    namespace = globals()
    built = []
    code = None

    def require():
        # The checker verifies its own module binding and code, then the facts.
        # M25n (opt-in review round 8): the facts are recorded on the first check
        # (only an opted-in preparation calls this), never at import.
        if not built:
            built.append(build())
        facts = built[0]
        if (dict.get(namespace, '_require_capture_span_authority') is not require
                or require.__code__ is not code or require.__defaults__ is not None
                or any((dict.get(owner, name) if type(owner) is dict else getattr(owner, name, None)) is not function
                       or not _callable_authority_matches(function, authority)
                       or any(function.__globals__.get(key) is not value for key, value in bindings)
                       for owner, name, function, authority, bindings in facts)):
            raise PreparedStaticCapabilityError('capture trusted span authority changed')

    require.__name__ = require.__qualname__ = '_require_capture_span_authority'
    code = require.__code__
    # Capture calls this checker through a keyword default, so rebinding the module
    # name cannot replace it; set before the capture's own facts are recorded.
    _capture_owned_local_recovery.__kwdefaults__['_span_authority'] = require

    def build():
        made = []

        def record(owner, name, function):
            if any(entry[2] is function for entry in made):
                return
            made.append((owner, name, function, _callable_authority(function),
                         tuple((key, function.__globals__[key]) for key in function.__code__.co_names
                               if key in function.__globals__)))

        for name in ('_capture_owned_local_recovery', '_capture_span_factory', '_capture_span_proofs_hold',
                     '_capture_plain_arguments', '_capture_material_is_native', '_capture_modules_unhooked',
                     '_capture_trust_state', '_capture_same_state', '_capture_owner_matches_identity',
                     '_capture_frozen_equal', '_capture_pressure_helpers_authentic', '_capture_state_tables',
                     '_capture_material_facts', '_require_owned_scope_factory_authority',
                     '_require_capture_scope_tables', 'compile_affine_recovery'):
            record(namespace, name, namespace[name])
        for name, function in (('_numpy_error_routes_are_builtin', _numpy_error_routes_are_builtin),
                               ('_load_routing_classes_are_untouched', _load_routing_classes_are_untouched),
                               ('_exact_load_case_instance_namespace', _exact_load_case_instance_namespace),
                               ('_plain_numeric_state', _plain_numeric_state), ('_instance_authority', _instance_authority),
                               ('_class_state', _class_state)):
            record(function.__globals__, name, function)
        return tuple(made)

    return require


_require_capture_span_authority = _make_capture_span_authority()


def _make_verification_publication_authority():
    """Pin delayed publication before any live model/config observation."""
    namespace=globals()
    matches=_callable_authority_matches
    records=[]
    def require():
        for name,function,authority,bindings in records:
            if (dict.get(namespace,name) is not function or not matches(function,authority)
                    or any(dict.get(function.__globals__,key) is not value for key,value in bindings)):
                raise PreparedStaticCapabilityError('recovery verification publication authority changed')
        if (sys.getsizeof is not storage_size or np.frombuffer is not frombuffer
                or np.asarray is not asarray or np.allclose is not allclose):
            raise PreparedStaticCapabilityError('recovery verification operation authority changed')
    storage_size=sys.getsizeof
    frombuffer=np.frombuffer
    asarray=np.asarray
    allclose=np.allclose
    proof=(require,_callable_authority(require))
    for name in ('_probe_element_stress','compile_affine_recovery','_verify_staged_recipes'):
        namespace[name].__kwdefaults__['_publication_authority']=proof
    for name in ('_probe_element_stress','compile_affine_recovery','_verify_staged_recipes',
                 '_compare_owner_scale','_freeze_verification_value','_verification_storage_bytes',
                 '_check_verification_memory','_recover_owned_local','_array_value',
                 '_constant_equal','_derived_measure'):
        function=namespace[name]
        bindings=tuple((key,function.__globals__[key]) for key in function.__code__.co_names
                       if key in function.__globals__)
        records.append((name,function,_callable_authority(function),bindings))
    return require


_require_verification_publication_authority=_make_verification_publication_authority()
