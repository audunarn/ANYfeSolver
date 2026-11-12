"""M25: S3 trusted native scope around capture-time calls on private element clones.

M25f (owner decision): the trusted capture is opt-in (solver option
trusted_capture, default False). Most tests below exercise the opt-in path and
enable it through the autouse fixture; the default (fully guarded, pre-M25) path
has its own tests at the end.

After the M25 reviews: on the opt-in path the span is admitted only for exact Q4/S3
clones after admission checks (the boundary's trusted-span proofs, a module-hook
check and a before/after state snapshot), otherwise the pre-M25 per-call guarded
route; the capture authenticates its helpers through an import-time checker it
holds itself; and a trusted capture must equal a guarded capture. The admission is
not proven callback-free: the round-five review findings in the M25 report remain
open for this opt-in path. Milestone evidence is written by the opt-in
test_prepared_static_response_m25_assessment.py, not here.
"""
import cProfile
import dataclasses
from types import FunctionType as types_FunctionType
import hashlib
import json
import os
import pstats
import subprocess
import sys

import numpy as np
import pytest
from anyfesolver import PreparedStaticCapabilityError
from anyfesolver import e4_pl_element as q4_module
from anyfesolver import e4_pl_s3_element as s3_module
from anyfesolver import prepared_static_recovery as recovery
from anyfesolver.boundary import LoadCase
from anyfesolver.control import SolveCancelled
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import _QualifiedStateMapping
from anyfesolver.prepared_static_recovery import (_capture_owned_local_recovery, _recover_owned_local,
                                                owned_recovery_scope)
import prepared_static_response_m16_assessment  # noqa: F401  (adds scripts/ to sys.path)
from benchmark_static_batches import mixed_free_model
from test_prepared_static_evaluator import _bar, _shell_model
import prepared_static_response_m16_assessment as assessment

ELEMENT_FILES = None  # set below from the imported modules (exact paths)
Q4, S3 = q4_module.QualifiedE4PLShellElement, s3_module.QualifiedE4PLS3ShellElement
ELEMENT_FILES = {os.path.normcase(os.path.abspath(q4_module.__file__)), os.path.normcase(os.path.abspath(s3_module.__file__))}
PRESSURE = {'warm_stress': False, 'pressure_probe': 1.0}
KINDS = ('q4_staged', 'q4_warped', 's3', 'q4_pressure', 's3_pressure', 'beam')


@pytest.fixture(autouse=True)
def _trusted_capture_enabled():
    """The tests in this module exercise the opt-in trusted capture directly."""
    with recovery.trusted_capture_scope(True):
        yield


def _warped_mixed_model():
    model = mixed_free_model()[0]
    first = min(eid for eid, element in model.mesh.elements.items() if type(element) is Q4)
    model.mesh.nodes[model.mesh.elements[first].node_ids[2]].z += 0.05
    return model


def _case(kind):
    """(model, element, capture options, expected staged kernel) for one capture kind."""
    builders = {'q4_staged': (lambda: mixed_free_model()[0], Q4, {}, True),
                'q4_warped': (_warped_mixed_model, Q4, {}, False),
                's3': (_shell_model, S3, {}, False),
                'q4_pressure': (lambda: mixed_free_model()[0], Q4, PRESSURE, False),
                's3_pressure': (_shell_model, S3, PRESSURE, False),
                'beam': (_bar, BeamElement, {}, False)}
    build, cls, options, staged = builders[kind]
    model = build()
    eid = min(eid for eid, element in model.mesh.elements.items() if type(element) is cls)
    return model, model.mesh.elements[eid], options, staged


def _accepts(element):
    accepts = q4_module._q4_trusted_operation_accepts if type(element) is Q4 else s3_module._s3_trusted_operation_accepts
    return accepts(element)


def _profiled(call):
    profile = cProfile.Profile()
    profile.enable()
    try:
        result = call()
    finally:
        profile.disable()
    counts = {}
    for (path, _line, name), (_primitive, calls, *_rest) in pstats.Stats(profile).stats.items():
        key = (os.path.normcase(os.path.abspath(path)) if os.path.isabs(path) else path, name)
        counts[key] = counts.get(key, 0) + calls
    return result, counts


def _calls(counts, name, files=None):
    files = ELEMENT_FILES if files is None else files
    def matches(path):
        return path in files or os.path.basename(path) in files
    return sum(calls for (path, function), calls in counts.items() if function == name and matches(path))


def _foreign_error_handler():
    """A NumPy error callback: the capture span must then keep per-call guards."""
    return np.errstate(all='call', call=lambda kind, flag: None)


def _digest(value, digest=None, seen=None, labels=None):
    """Content digest. References to the private owner objects and their id() values
    (native cache keys embed ids) become stable labels; a repeated object becomes a
    reference to its first visit, so aliasing stays visible."""
    digest = hashlib.sha256() if digest is None else digest
    seen = [] if seen is None else seen
    labels = {} if labels is None else labels
    label = labels.get(id(value))
    if label is not None and label[0] is value:
        digest.update(('<owner:' + label[1] + '>').encode())
        return digest
    if type(value) is int and value in labels:
        digest.update(('<id:' + labels[value][1] + '>').encode())
        return digest
    if not isinstance(value, (bool, int, float, str, bytes, type(None), np.generic)):
        for index, prior in enumerate(seen):
            if value is prior:
                digest.update(('<seen:%d>' % index).encode())
                return digest
        seen.append(value)
    if isinstance(value, np.ndarray):
        digest.update(repr((value.dtype.str, value.shape)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode())
            _digest(value[key], digest, seen, labels)
    elif isinstance(value, (list, tuple)):
        digest.update(repr((type(value).__name__, len(value))).encode())
        for item in value:
            _digest(item, digest, seen, labels)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        digest.update(type(value).__qualname__.encode())
        for field in dataclasses.fields(value):
            _digest(getattr(value, field.name), digest, seen, labels)
    elif value is None or isinstance(value, (bool, int, float, str, bytes, np.generic)):
        digest.update(repr(value).encode())
    elif callable(value) or isinstance(value, type):
        digest.update(('callable:' + getattr(value, '__qualname__', type(value).__qualname__)).encode())
    elif hasattr(value, '__dict__'):
        digest.update(type(value).__qualname__.encode())
        _digest(dict(vars(value)), digest, seen, labels)
    else:
        digest.update(type(value).__qualname__.encode())
    return digest


def _recipe_fingerprint(recipe, displacement=None, pressure=None):
    """Numeric identity, staged kernel, the warmed private element, mesh and material
    contents, and an output: a recovery for stress recipes, a pressure load for
    pressure-only recipes. Retained bytes depend on earlier work in the process
    (shared-object accounting), so they are compared in fresh processes below."""
    element, mesh, material = recipe.execution_owner
    labels = {id(owned): (owned, name) for owned, name in zip(recipe.execution_owner, ('element', 'mesh', 'material'))}
    made = dict(identity=_digest(recipe.numeric_identity, labels=labels).hexdigest(),
                staged=_digest(recipe.staged_kernel, labels=labels).hexdigest(),
                reason=recipe.staged_fallback_reason)
    for name, owned in (('element', element), ('mesh', mesh), ('material', material)):
        made[name] = _digest(dict(vars(owned)), labels=labels).hexdigest()
    with owned_recovery_scope((recipe,)):
        if displacement is not None:
            made['recovery'] = _digest(_recover_owned_local(recipe, displacement)).hexdigest()
        if pressure is not None:
            load = LoadCase('m25-pressure-check')._consistent_pressure_load(element, mesh, pressure, None)
            made['pressure'] = _digest(load).hexdigest()
    return made


def _clone_from_traceback(excinfo):
    tb = excinfo.tb
    while tb is not None:
        if tb.tb_frame.f_code.co_name == '_capture_owned_local_recovery':
            return tb.tb_frame.f_locals.get('owned_element')
        tb = tb.tb_next
    return None


def _capture_with_guard_counts(model, element, **options):
    # Original-input policy admission precedes the private clone/trusted span,
    # and final policy revalidation follows it. Account for both real guard
    # costs separately; every clone warm-up guard is still counted. This does
    # not claim the complete capture is guard-free.
    policy_code = recovery._prepared_q4._capture_original_q4_policy.__code__
    counts, policy_counts = {}, {}
    policy_depth = policy_calls = 0
    def observe(frame, event, _arg):
        nonlocal policy_depth, policy_calls
        if event == 'call':
            if frame.f_code is policy_code:
                policy_depth += 1
                policy_calls += 1
            path = frame.f_code.co_filename
            key = (os.path.normcase(os.path.abspath(path)) if os.path.isabs(path) else path,
                   frame.f_code.co_name)
            bucket = policy_counts if policy_depth else counts
            bucket[key] = bucket.get(key, 0) + 1
        elif event == 'return' and frame.f_code is policy_code:
            policy_depth -= 1
    previous_profile = sys.getprofile()
    sys.setprofile(observe)
    try:
        recipe = _capture_owned_local_recovery(model, element, **options)
    finally:
        sys.setprofile(previous_profile)
    assert sys.getprofile() is previous_profile
    assert policy_depth == 0
    assert policy_calls == 2 * int(type(element) is Q4 and options.get('warm_stress', True))
    return recipe, counts, policy_counts


@pytest.mark.parametrize('kind', ['q4_staged', 'q4_warped', 's3', 'q4_pressure', 's3_pressure'])
def test_trusted_capture_runs_warmups_on_native_trusted_path(kind):
    model, element, options, staged = _case(kind)
    recipe, counts, policy_counts = _capture_with_guard_counts(model, element, **options)
    if kind == 'q4_staged':
        assert _calls(policy_counts, 'post_observation_guard') > 0
    assert _calls(counts, 'post_observation_guard') == 0, counts
    assert _calls(counts, 'trusted_post_observation_guard') > 0
    assert (recipe.staged_kernel is not None) is staged
    if 'pressure_probe' in options:
        assert _calls(counts, '_consistent_pressure_load', {'boundary.py'}) == 1
    else:
        assert _calls(counts, 'compile_q4_stages', {'prepared_static_q4.py'}) == 1
        if not staged:  # the native warm-up ran on the element itself
            assert _calls(counts, 'compute_stresses') >= 1
    owned = recipe.execution_owner[0]
    assert owned is not element and not _accepts(owned)  # the scope ended with the capture


@pytest.mark.parametrize('kind', KINDS)
def test_trusted_capture_equals_guarded_capture(kind, monkeypatch):
    model, element, options, _staged = _case(kind)
    trusted, trusted_counts = _profiled(lambda: _capture_owned_local_recovery(model, element, **options))
    with recovery.trusted_capture_scope(False):  # the default: the pre-M25 guarded route
        guarded, guarded_counts = _profiled(lambda: _capture_owned_local_recovery(model, element, **options))
    assert _calls(guarded_counts, 'trusted_post_observation_guard') == 0
    if kind == 'beam':
        assert _calls(trusted_counts, 'trusted_post_observation_guard') == 0
    else:
        assert _calls(trusted_counts, 'trusted_post_observation_guard') > 0
        assert _calls(guarded_counts, 'post_observation_guard') > 0
    if 'pressure_probe' in options:
        arguments = dict(pressure=0.75)
    else:
        arguments = dict(displacement=np.random.default_rng(25).uniform(-1e-3, 1e-3, 6 * len(element.node_ids)))
    assert _recipe_fingerprint(trusted, **arguments) == _recipe_fingerprint(guarded, **arguments)


_FRESH_CAPTURES = """
import json, sys
sys.path[:0] = json.loads(sys.argv[2])
import test_prepared_static_response_m25 as m25
from anyfesolver import prepared_static_recovery as recovery
made = []
with recovery.trusted_capture_scope(sys.argv[1] == 'trusted'):
    for kind in m25.KINDS:
        model, element, options, _staged = m25._case(kind)
        made.append(int(recovery._capture_owned_local_recovery(model, element, **options).retained_bytes))
print(json.dumps(made))
"""


def test_retained_bytes_equal_in_fresh_processes():
    paths = json.dumps([entry for entry in sys.path if entry])
    made = {}
    for route in ('trusted', 'guarded'):
        done = subprocess.run([sys.executable, '-c', _FRESH_CAPTURES, route, paths], capture_output=True,
                              text=True, timeout=120, check=True)
        made[route] = json.loads(done.stdout.strip().splitlines()[-1])
    assert made['trusted'] == made['guarded'], made


def _self_restoring_substitute(observed, original):
    def substitute(self, owned, mesh, pressure, coords=None):
        LoadCase._consistent_pressure_load = original
        namespace = object.__getattribute__(owned, '__dict__')
        namespace['injected_callable'] = lambda *args: None
        try:
            owned.get_node_coordinates(mesh)
            observed.append('accepted')
        except ValueError as error:
            observed.append(str(error))
        finally:
            namespace.pop('injected_callable', None)
        return original(self, owned, mesh, pressure, coords)
    return substitute


def test_substituted_pressure_helper_keeps_guarded_route(monkeypatch):
    """First review: a substituted LoadCase helper must not run inside a trusted span."""
    model, element, options, _staged = _case('q4_pressure')
    observed = []
    monkeypatch.setattr(LoadCase, '_consistent_pressure_load',
                        _self_restoring_substitute(observed, LoadCase._consistent_pressure_load))
    _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element, **options))
    assert observed == ['qualified Q4 instance contains a callable authority override']
    assert _calls(counts, 'trusted_post_observation_guard') == 0


@pytest.mark.parametrize('trigger', ['admission', 'any'])
def test_materials_mapping_cannot_plant_code_after_the_proofs(trigger, monkeypatch):
    """Second review: a materials mapping with an instance `get` plants a self-restoring
    substitute. Admission must call no mapping method, and a substitute planted
    before the proofs (by get_material) is caught by them."""
    model, element, options, _staged = _case('q4_pressure')
    original = LoadCase._consistent_pressure_load
    monkeypatch.setattr(LoadCase, '_consistent_pressure_load', original)  # restored after the test
    observed, callers = [], []
    materials = _QualifiedStateMapping(dict(model.materials))

    def planting_get(name, default=None):
        caller = sys._getframe(1).f_code.co_name
        callers.append(caller)
        if trigger == 'any' or caller != 'get_material':
            LoadCase._consistent_pressure_load = _self_restoring_substitute(observed, original)
        return dict.get(materials, name, default)

    object.__getattribute__(materials, '__dict__')['get'] = planting_get
    model.materials = materials
    _capture_owned_local_recovery(model, element, **options)
    assert 'accepted' not in observed
    assert all(caller == 'get_material' for caller in callers), callers  # admission calls no mapping method
    if trigger == 'any':
        assert observed == ['qualified Q4 instance contains a callable authority override']


def test_foreign_numpy_error_handler_keeps_guarded_route():
    model, element, _options, _staged = _case('q4_staged')
    owned, _mesh, material = _capture_owned_local_recovery(model, element).execution_owner
    checker = recovery._capture_owned_local_recovery.__kwdefaults__['_span_authority']
    with _foreign_error_handler():
        assert recovery._capture_span_factory(owned, material, None, None, True, checker) == (None, None)
    assert recovery._capture_span_factory(owned, material, None, None, True, checker)[0] is q4_module._q4_trusted_operation_scope


@pytest.mark.parametrize('delete', [True, False])
def test_module_getattr_hook_keeps_guarded_route(delete, monkeypatch):
    """Third review (Codex probe): a numpy module __getattr__ reached through a deleted
    attribute during admission plants a self-restoring substitute; a hook installed
    without deleting anything also removes trust."""
    model, element, options, _staged = _case('q4_pressure')
    original = LoadCase._consistent_pressure_load
    monkeypatch.setattr(LoadCase, '_consistent_pressure_load', original)
    geterr, observed = np.geterr, []

    prior = vars(np).get('__getattr__')

    def hook(name):
        if name == 'geterr' and sys._getframe(1).f_code.co_name == '_numpy_error_routes_are_builtin':
            np.geterr = geterr
            observed.append('hook ran in the NumPy route proof')
            LoadCase._consistent_pressure_load = _self_restoring_substitute(observed, original)
            return geterr
        if name == 'geterr':
            return geterr
        return prior(name)

    monkeypatch.setattr(np, '__getattr__', hook, raising=False)
    if delete:
        monkeypatch.delattr(np, 'geterr')
    _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element, **options))
    assert 'accepted' not in observed
    assert _calls(counts, 'trusted_post_observation_guard') == 0
    # The module check runs before any proof, so the hook never reaches the proof.
    assert 'hook ran in the NumPy route proof' not in observed


class _ValuesHook(dict):
    def __init__(self, base, action):
        super().__init__(base)
        self.action = action

    def values(self):
        self.action()
        return dict.values(self)


def test_callback_during_proofs_is_caught_by_the_snapshot(monkeypatch):
    """A replaced numpy value reached by the material proof installs a NumPy error
    callback after the NumPy route proof has passed; the admission snapshot must
    keep the guarded route."""
    model, element, _options, _staged = _case('q4_staged')
    material = model.get_material(element.material_name)
    monkeypatch.setattr(material, 'density', np.float64(material.density))  # numpy scalar: the plain-state proof reads sctypeDict
    saved = np.geterr(), np.geterrcall()
    fired = []

    def action():
        frame, inside = sys._getframe(1), False
        while frame is not None and not inside:
            inside = frame.f_code.co_name == '_capture_material_is_native'
            frame = frame.f_back
        if inside and not fired:  # only inside the admission proofs, after the NumPy route proof
            fired.append(True)
            np.seterrcall(lambda kind, flag: None)
            np.seterr(under='call')

    monkeypatch.setattr(np, 'sctypeDict', _ValuesHook(np.sctypeDict, action))
    try:
        # M25e: a change during admission is refused rather than continued on the guarded route.
        with pytest.raises(PreparedStaticCapabilityError, match='state changed during trusted-capture admission'):
            _capture_owned_local_recovery(model, element)
    finally:
        np.seterr(**saved[0])
        np.seterrcall(saved[1])
    assert fired == [True]


def test_state_change_inside_span_is_refused_at_exit():
    model, element, _options, _staged = _case('q4_staged')
    saved, previous = np.geterr(), sys.gettrace()
    np.seterr(under='ignore')  # known precondition, so the change below is a change

    def tracer(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == 'compile_q4_stages':
            np.seterr(under='warn')  # changes NumPy's error state inside the span
        return None

    sys.settrace(tracer)
    try:
        with pytest.raises(PreparedStaticCapabilityError, match='route changed during capture'):
            _capture_owned_local_recovery(model, element)
    finally:
        sys.settrace(previous)
        np.seterr(**saved)


class _FloatLike:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return self.value


@pytest.mark.parametrize('arguments', [{'warm_stress': False, 'pressure_probe': _FloatLike(1.0)},
                                       {'warm_stress': 1}, {'warm_stress': np.bool_(True)}])
def test_nonplain_span_arguments_keep_guarded_route(arguments):
    model, element, _options, _staged = _case('q4_staged')
    _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element, **arguments))
    assert _calls(counts, 'trusted_post_observation_guard') == 0
    assert _calls(counts, 'post_observation_guard') > 0
    assert recovery._capture_plain_arguments(1.0, True) and recovery._capture_plain_arguments((1.0, 2.0), False)
    assert not recovery._capture_plain_arguments(np.float64(1.0), True)


@pytest.mark.parametrize('kind', ['q4_staged', 's3'])
def test_capture_refuses_rebound_scope_factory(kind, monkeypatch):
    model, element, _options, _staged = _case(kind)
    module, name = (q4_module, '_q4_trusted_operation_scope') if kind == 'q4_staged' else (s3_module, '_s3_trusted_operation_scope')
    monkeypatch.setattr(module, name, lambda elements: pytest.fail('replaced scope factory entered'))
    with pytest.raises(PreparedStaticCapabilityError, match='scope factory authority'):
        _capture_owned_local_recovery(model, element)


def test_beam_capture_does_not_depend_on_scope_factories(monkeypatch):
    model, element, _options, _staged = _case('beam')
    monkeypatch.setattr(q4_module, '_q4_trusted_operation_scope', lambda elements: pytest.fail('factory entered'))
    monkeypatch.setattr(recovery, '_SCOPE_PAIRS', ())
    recipe = _capture_owned_local_recovery(model, element)
    assert recipe.staged_kernel is None


@pytest.mark.parametrize('table', ['_SCOPE_PAIRS', '_SCOPE_FACTORIES', '_SCOPE_CLASSES', '_SCOPE_ELEMENT_TYPES'])
def test_rebound_scope_tables_are_refused_by_the_opt_in_capture(table, monkeypatch):
    model, element, _options, _staged = _case('q4_staged')
    recipe = _capture_owned_local_recovery(model, element)
    monkeypatch.setattr(recovery, table, tuple(list(getattr(recovery, table))))  # equal content, new object
    with pytest.raises(PreparedStaticCapabilityError, match='scope factory authority'):
        recovery._require_capture_scope_tables()
    with pytest.raises(PreparedStaticCapabilityError, match='(scope factory|capture trusted span) authority'):
        _capture_owned_local_recovery(model, element)
    # Opt-in review round 9: the default route keeps the pre-M25 owned-scope check,
    # which reads the tables it uses and does not inspect their module bindings.
    recovery._require_owned_scope_factory_authority()
    with owned_recovery_scope((recipe,)):
        pass


_DEFAULT_SCOPE_WITH_PLANTED_KEY = """
import json, sys
sys.path[:0] = json.loads(sys.argv[1])
import test_prepared_static_response_m25 as m25
from anyfesolver import prepared_static_recovery as recovery
model, element, _options, _staged = m25._case('q4_staged')
with recovery.trusted_capture_scope(False):
    recipe = recovery._capture_owned_local_recovery(model, element)
calls = []
class Key(str):
    def __eq__(self, other):
        calls.append(sys._getframe(1).f_code.co_name)
        return str.__eq__(self, other)
    __hash__ = str.__hash__
namespace = vars(recovery)
for name in ('_SCOPE_PAIRS', '_SCOPE_ELEMENT_TYPES'):
    namespace[Key(name)] = namespace.pop(name)
with recovery.trusted_capture_scope(False):
    with recovery.owned_recovery_scope((recipe,)):
        pass
print(json.dumps({'calls': calls}))
"""


def test_default_owned_scope_makes_no_keyed_reads_of_recovery_globals():
    """Opt-in review round 9: owned_recovery_scope (reached on the default route by
    envelope compilation) runs only the pre-M25 checks; it does not look up the
    scope-table bindings by key (a planted str-subclass key would run its __eq__)."""
    paths = json.dumps([entry for entry in sys.path if entry])
    done = subprocess.run([sys.executable, '-c', _DEFAULT_SCOPE_WITH_PLANTED_KEY, paths], capture_output=True,
                          text=True, timeout=180, check=True)
    made = json.loads(done.stdout.strip().splitlines()[-1])
    assert made == {'calls': []}


def _admit_factory(owned_element, owned_material, probe_case, pressure_probe, warm_stress, span_authority):
    return probe_case, None


def _admit_proofs(owned_element, owned_material, probe_case, pressure_probe, warm_stress):
    return True


def _admit_material(owned_element, owned_material):
    return True


def _admit_arguments(pressure_probe, warm_stress):
    return True


@pytest.mark.parametrize('target, replacement', [('_capture_span_factory', _admit_factory),
                                                 ('_capture_span_proofs_hold', _admit_proofs),
                                                 ('_capture_material_is_native', _admit_material),
                                                 ('_capture_plain_arguments', _admit_arguments)])
def test_capture_span_helper_code_swap_is_refused(target, replacement, monkeypatch):
    model, element, _options, _staged = _case('q4_staged')
    monkeypatch.setattr(recovery, '_CAPTURE_SPAN_FACTS', (), raising=False)  # second-review scenario
    monkeypatch.setattr(getattr(recovery, target), '__code__', replacement.__code__)
    with pytest.raises(PreparedStaticCapabilityError, match='capture trusted span authority'):
        _capture_owned_local_recovery(model, element)


def test_rebound_checker_is_refused(monkeypatch):
    model, element, _options, _staged = _case('q4_staged')
    monkeypatch.setattr(recovery, '_require_capture_span_authority', lambda: None)
    monkeypatch.setattr(recovery._capture_span_proofs_hold, '__code__', _admit_proofs.__code__)
    with pytest.raises(PreparedStaticCapabilityError, match='capture trusted span authority'):
        _capture_owned_local_recovery(model, element)


@pytest.mark.parametrize('kind', ['q4_staged', 's3'])
def test_error_inside_capture_span_matches_guarded_route_and_revokes(kind):
    model, element, _options, _staged = _case(kind)
    first, third = model.mesh.nodes[element.node_ids[0]], model.mesh.nodes[element.node_ids[2]]
    saved = (third.x, third.y, third.z)
    third.x, third.y, third.z = first.x, first.y, first.z
    try:
        with pytest.raises(ValueError) as trusted_error:
            _capture_owned_local_recovery(model, element)
        clone = _clone_from_traceback(trusted_error)
        if kind == 'q4_staged':
            # Original-input Q4 policy rejects collapsed diagonals before cloning.
            assert clone is None and 'diagonals are degenerate' in str(trusted_error.value)
        else:
            assert clone is not None and not _accepts(clone)
        with _foreign_error_handler():
            with pytest.raises(ValueError) as guarded_error:
                _capture_owned_local_recovery(model, element)
        assert (type(trusted_error.value), str(trusted_error.value)) == (type(guarded_error.value), str(guarded_error.value))
    finally:
        third.x, third.y, third.z = saved
    _recipe, counts, _policy_counts = _capture_with_guard_counts(model, element)
    assert _calls(counts, 'trusted_post_observation_guard') > 0 and _calls(counts, 'post_observation_guard') == 0


@pytest.mark.parametrize('kind', ['q4_staged', 'q4_warped', 's3'])
@pytest.mark.parametrize('failure', [SolveCancelled, ValueError])
def test_cancellation_inside_capture_span_propagates_and_revokes(kind, failure):
    """An exception raised inside the span (injected at the stage compile) propagates
    unchanged, the clone's lease is revoked, and a later capture is trusted again."""
    model, element, _options, _staged = _case(kind)
    previous = sys.gettrace()
    inside = []

    def tracer(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == 'compile_q4_stages':
            caller = frame.f_back
            owned = caller.f_locals.get('owned_element') if caller is not None else None
            inside.append(owned is not None and _accepts(owned))
            raise failure('injected capture failure')
        return None

    sys.settrace(tracer)
    try:
        with pytest.raises(failure, match='injected capture failure') as cancelled:
            _capture_owned_local_recovery(model, element)
    finally:
        sys.settrace(previous)
    assert inside == [True]  # the cancellation was raised inside the trusted span
    clone = _clone_from_traceback(cancelled)
    assert clone is not None and not _accepts(clone)
    _recipe, counts, _policy_counts = _capture_with_guard_counts(model, element)
    assert _calls(counts, 'trusted_post_observation_guard') > 0 and _calls(counts, 'post_observation_guard') == 0


def test_lean_holdouts_meet_frozen_criteria_after_s3():
    lean = {'selected_reaction_diagnostics': 'lean', 'trusted_capture': True}
    mode = 'certified_vectorized'
    summaries = [assessment.holdout_a1(mode, native_loads=True, route_options=lean),
                 assessment.holdout_a3(mode, native_loads=True, route_options=lean),
                 assessment.holdout_a5(mode, native_loads=True, route_options=lean),
                 assessment.holdout_a4(mode, route_options=lean)]
    for summary in summaries:
        assert summary['failure_count'] == 0, summary
    assert summaries[2]['master_entries_checked'] > 0



# ---------------------------------------------------------------------------
# M25e: third Codex review (blockers 1, 2, 4, should-fix 6, 8, 9)

def _admission_frame(name):
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == name:
            return frame
        frame = frame.f_back
    return None


def _numpy_scalar_material(model, element, monkeypatch):
    material = model.get_material(element.material_name)
    monkeypatch.setattr(material, 'density', np.float64(material.density))  # the material proof reads sctypeDict


@pytest.mark.parametrize('target', ['material_clone', 'element_clone'])
def test_owner_change_during_admission_is_refused(target, monkeypatch):
    """Blocker 1: a callback during admission changes a private owner; the capture
    must not continue with it (snapshot for the material, identity check for the element)."""
    model, element, _options, _staged = _case('q4_staged')
    _numpy_scalar_material(model, element, monkeypatch)
    done = []

    def action():
        frame = _admission_frame('_capture_material_is_native')
        if frame is None or done:
            return
        done.append(True)
        if target == 'material_clone':
            owned = frame.f_locals['owned_material']
            object.__getattribute__(owned, '__dict__')['poisson_ratio'] = 0.25
        else:
            owned = frame.f_locals['owned_element']
            namespace = object.__getattribute__(owned, '__dict__')
            key = next(k for k, v in sorted(namespace.items()) if type(v) is float and v != 0.0)
            namespace[key] = namespace[key] * 1.5

    monkeypatch.setattr(np, 'sctypeDict', _ValuesHook(np.sctypeDict, action))
    match = 'state changed during trusted-capture admission' if target == 'material_clone' else 'differs from its recipe identity'
    with pytest.raises(PreparedStaticCapabilityError, match=match):
        _capture_owned_local_recovery(model, element)
    assert done == [True]


def test_hook_installed_during_proofs_is_refused(monkeypatch):
    """Post-proof module check: a callback during the proofs installs a module hook."""
    model, element, _options, _staged = _case('q4_staged')
    _numpy_scalar_material(model, element, monkeypatch)
    prior = vars(np).get('__getattr__')
    monkeypatch.setattr(np, '__getattr__', prior, raising=False)  # restored after the test

    def action():
        if _admission_frame('_capture_material_is_native') is not None:
            np.__getattr__ = lambda name: prior(name)

    monkeypatch.setattr(np, 'sctypeDict', _ValuesHook(np.sctypeDict, action))
    with pytest.raises(PreparedStaticCapabilityError, match='state changed during trusted-capture admission'):
        _capture_owned_local_recovery(model, element)


def test_hook_installed_inside_span_is_refused_at_exit(monkeypatch):
    model, element, _options, _staged = _case('q4_staged')
    prior, previous = vars(np).get('__getattr__'), sys.gettrace()
    monkeypatch.setattr(np, '__getattr__', prior, raising=False)

    def tracer(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == 'compile_q4_stages':
            np.__getattr__ = lambda name: prior(name)
        return None

    sys.settrace(tracer)
    try:
        with pytest.raises(PreparedStaticCapabilityError, match='route changed during capture'):
            _capture_owned_local_recovery(model, element)
    finally:
        sys.settrace(previous)


def test_late_module_hook_is_pinned(monkeypatch):
    """Blocker 2: the anyfesolver package's own __getattr__ is pinned after first use; a
    replacement that merely looks canonical (same globals, file name and function
    name) removes trust, and the module __file__ is never compared through __eq__."""
    import types
    import anyfesolver
    model, element, options, _staged = _case('q4_pressure')
    _capture_owned_local_recovery(model, element, **options)  # pins the late hooks
    namespace = vars(anyfesolver)
    code = compile('def __getattr__(name):\n    raise AttributeError(name)\n', namespace['__file__'], 'exec')
    lookalike = types.FunctionType(next(c for c in code.co_consts if isinstance(c, types.CodeType)), namespace, '__getattr__')
    compared = []

    class FileName(str):
        def __eq__(self, other):
            compared.append(other)
            return True
        __hash__ = str.__hash__

    monkeypatch.setitem(namespace, '__getattr__', lookalike)
    monkeypatch.setitem(namespace, '__file__', FileName(namespace['__file__']))
    _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element, **options))
    assert _calls(counts, 'trusted_post_observation_guard') == 0 and compared == []


@pytest.mark.parametrize('change', ['hook', 'sys_modules_entry'])
def test_span_module_prepared_static_q4_is_tracked(change, monkeypatch):
    """Blocker 4: the stage-compile module (imported after the old table) is tracked,
    for module hooks and for its sys.modules binding."""
    import types
    from anyfesolver import prepared_static_q4 as q4_kernels
    model, element, _options, _staged = _case('q4_staged')
    if change == 'hook':
        def hook(name):
            raise AttributeError(name)
        monkeypatch.setitem(vars(q4_kernels), '__getattr__', hook)
    else:
        monkeypatch.setitem(sys.modules, q4_kernels.__name__, types.ModuleType(q4_kernels.__name__))
    _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element))
    assert _calls(counts, 'trusted_post_observation_guard') == 0


def test_module_hook_cannot_run_during_the_table_check(monkeypatch):
    """Should-fix 6: the module check runs before the scope tables are read, so a
    restoring module hook on the element module is never reached by admission."""
    model, element, _options, _staged = _case('q4_staged')
    calls = []
    factory = q4_module._q4_trusted_operation_scope

    def hook(name):
        calls.append(name)
        if name == '_q4_trusted_operation_scope':
            return factory
        raise AttributeError(name)

    monkeypatch.setitem(vars(q4_module), '__getattr__', hook)
    monkeypatch.delitem(vars(q4_module), '_q4_trusted_operation_scope')
    _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element))
    # Python's import machinery may probe __path__ at the capture's lazy imports
    # (before admission); admission itself must never reach the hook.
    assert '_q4_trusted_operation_scope' not in calls
    assert _calls(counts, 'trusted_post_observation_guard') == 0


def test_admission_exception_falls_back_to_guarded_route(monkeypatch):
    """Should-fix 8: a proof that raises (np.geterr removed, NumPy's own hook intact)
    keeps the guarded route instead of failing the capture."""
    model, element, options, _staged = _case('q4_pressure')
    monkeypatch.delattr(np, 'geterr')
    _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element, **options))
    assert _calls(counts, 'trusted_post_observation_guard') == 0 and _calls(counts, 'post_observation_guard') > 0


def test_material_class_code_change_keeps_guarded_route(monkeypatch):
    """Should-fix 9: the material proof is necessary (an in-place code change of a
    material class method, behaviour unchanged, removes trust)."""
    model, element, _options, _staged = _case('q4_staged')
    material_class = type(model.get_material(element.material_name))
    name, function = next((n, v) for n, v in sorted(vars(material_class).items()) if type(v) is types_FunctionType)
    # A different code object with the same behaviour (renamed); value comparison sees the change.
    monkeypatch.setattr(function, '__code__', function.__code__.replace(co_name=function.__code__.co_name + '_changed'))
    _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element))
    assert _calls(counts, 'trusted_post_observation_guard') == 0 and _calls(counts, 'post_observation_guard') > 0


def test_rebinding_only_the_checker_name_is_refused(monkeypatch):
    model, element, _options, _staged = _case('q4_staged')
    monkeypatch.setattr(recovery, '_require_capture_span_authority', lambda: None)
    with pytest.raises(PreparedStaticCapabilityError, match='capture trusted span authority'):
        _capture_owned_local_recovery(model, element)


def test_lease_revoked_in_a_context_copied_inside_the_span():
    import contextvars
    model, element, _options, _staged = _case('q4_staged')
    previous, copied = sys.gettrace(), []

    def tracer(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == 'compile_q4_stages' and not copied:
            owned = frame.f_back.f_locals.get('owned_element')
            copied.append((owned, contextvars.copy_context()))
        return None

    sys.settrace(tracer)
    try:
        _capture_owned_local_recovery(model, element)
    finally:
        sys.settrace(previous)
    owned, context = copied[0]
    assert context.run(_accepts, owned) is False



# ---------------------------------------------------------------------------
# M25f: opt-in (default off)

ADMISSION_FUNCTIONS = {'_capture_span_factory', '_capture_span_proofs_hold', '_capture_trust_state',
                       '_capture_owner_matches_identity', '_capture_material_is_native', '_capture_plain_arguments'}


@pytest.mark.parametrize('kind', KINDS)
def test_default_capture_is_the_guarded_route_without_admission(kind):
    model, element, options, _staged = _case(kind)
    with recovery.trusted_capture_scope(False):
        _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element, **options))
    assert _calls(counts, 'trusted_post_observation_guard') == 0
    assert not any(name in ADMISSION_FUNCTIONS for (_path, name) in counts)
    if kind != 'beam':
        assert _calls(counts, 'post_observation_guard') > 0


def test_context_default_is_off():
    import contextvars
    assert contextvars.Context().run(recovery._trusted_capture_active) is False


@pytest.mark.parametrize('outcome', ['success', 'failure'])
def test_opt_in_is_revoked_in_contexts_copied_during_preparation(outcome):
    """Opt-in review: a context copied during an opted-in preparation must not keep
    the opt-in after that preparation ends (normally or by an exception)."""
    import contextvars
    from anyfesolver import prepare_static_evaluator, RecoveryConfig
    every = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)
    copied = []

    class Stop(Exception):
        pass

    def progress(event):
        if not copied:
            copied.append((recovery._trusted_capture_active(), contextvars.copy_context()))
            if outcome == 'failure':
                raise Stop()

    with recovery.trusted_capture_scope(False):  # no outer opt-in for this test
        if outcome == 'failure':
            from anyfesolver import PreparedStaticError
            with pytest.raises((Stop, PreparedStaticError)):  # snapshot capture wraps the callback's error
                prepare_static_evaluator(mixed_free_model()[0], recovery_config=every, envelope_recovery_config=every,
                                         trusted_capture=True, progress_callback=progress)
        else:
            prepare_static_evaluator(mixed_free_model()[0], recovery_config=every, envelope_recovery_config=every,
                                     trusted_capture=True, progress_callback=progress).close()
    active_inside, context = copied[0]
    assert active_inside is True
    assert context.run(recovery._trusted_capture_active) is False
    model, element, options, _staged = _case('q4_staged')
    _recipe, counts = _profiled(lambda: context.run(_capture_owned_local_recovery, model, element))
    assert _calls(counts, 'trusted_post_observation_guard') == 0


def test_evaluator_option_controls_trusted_capture_and_keeps_the_snapshot():
    from anyfesolver import prepare_static_evaluator, RecoveryConfig
    every = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)
    made = {}
    for flag in (None, False, True):
        options = {} if flag is None else {'trusted_capture': flag}
        owner, counts = _profiled(lambda: prepare_static_evaluator(mixed_free_model()[0], recovery_config=every,
                                                                   envelope_recovery_config=every, **options))
        with owner:
            # Preparation's verification steps use the existing trusted recovery scopes in
            # either mode, so the opt-in is observed through capture admission.
            made[flag] = (owner.diagnostics()['trusted_capture'], owner._snapshot.content_digest,
                          _calls(counts, '_capture_span_factory', {'prepared_static_recovery.py'}) > 0)
    assert made[None][0] is False and made[False][0] is False and made[True][0] is True
    assert made[None][2] is False and made[False][2] is False and made[True][2] is True
    assert made[None][1] == made[False][1] == made[True][1]


@pytest.mark.parametrize('value', [1, 0, 'yes', None, np.bool_(True)])
def test_trusted_capture_option_must_be_bool(value):
    from anyfesolver import prepare_static_evaluator
    with pytest.raises(ValueError, match='trusted_capture must be True or False'):
        prepare_static_evaluator(mixed_free_model()[0], trusted_capture=value)


def test_response_evaluator_accepts_the_option():
    from anyfesolver import PatternCase, prepare_static_response_evaluator, RecoveryConfig
    patterns = assessment._shell_patterns()
    stress = RecoveryConfig(include_stresses=True, include_reactions=True, components=assessment.SHELL_COMPONENTS)
    for flag in (False, True):
        with prepare_static_response_evaluator(_shell_model(), patterns, recovery_config=stress,
                                               stress_execution='certified_vectorized', trusted_capture=flag) as owner:
            assert owner.diagnostics()['trusted_capture'] is flag
            next(owner.iter_batches([PatternCase('a', {name: 1.0 for name in patterns})]))


# ---------------------------------------------------------------------------
# M25i-M25l (opt-in review rounds 3-6): import inspects no sys.modules entry (the
# module table is built on the first opt-in admission), and that inspection must
# never run code of arbitrary sys.modules entries.

_HOSTILE_IMPORT = """
import json, sys
calls = []
kind = sys.argv[2]
if kind == 'property':
    class Hostile:
        @property
        def __dict__(self):
            calls.append('property')
            raise RuntimeError('hostile __dict__ descriptor')
elif kind == 'metaclass_eq':
    class Meta(type):
        def __eq__(cls, other):
            calls.append('metaclass __eq__')
            return True
        __hash__ = type.__hash__
    class Descriptor(metaclass=Meta):
        def __get__(self, instance, owner=None):
            calls.append('descriptor __get__')
            return {}
    class Hostile:
        __dict__ = Descriptor()
elif kind == 'foreign_builtin_descriptor':  # belongs to another class
    class Hostile:
        __dict__ = int.real
else:  # hash_collision_key: a namespace key whose __eq__ would run during a keyed lookup
    class Key:
        def __hash__(self):
            return hash('__spec__')
        def __eq__(self, other):
            calls.append('key __eq__')
            raise RuntimeError('hostile key equality')
    class Hostile:
        pass
hostile = Hostile()
if kind == 'hash_collision_key':
    vars(hostile)[Key()] = 0
sys.modules['numpy.capture_probe'] = hostile
sys.path[:0] = json.loads(sys.argv[1])
import anyfesolver.prepared_static_response
from anyfesolver import prepared_static_recovery as recovery
after_import = list(calls)  # the default path (import) inspects nothing
admissible = recovery._capture_modules_unhooked()  # first opt-in admission builds the table
print(json.dumps({'calls_after_import': after_import, 'calls': calls, 'opt_in_admissible': admissible}))
"""


@pytest.mark.parametrize('kind', ['property', 'metaclass_eq', 'foreign_builtin_descriptor', 'hash_collision_key'])
def test_import_never_runs_descriptors_of_sys_modules_entries(kind):
    paths = json.dumps([entry for entry in sys.path if entry])
    done = subprocess.run([sys.executable, '-c', _HOSTILE_IMPORT, paths, kind], capture_output=True, text=True,
                          timeout=120, check=True)
    made = json.loads(done.stdout.strip().splitlines()[-1])
    assert made == {'calls_after_import': [], 'calls': [], 'opt_in_admissible': False}


def test_uninspectable_module_entry_keeps_guarded_route_without_running_it(monkeypatch):
    calls = []

    class Hostile:
        @property
        def __dict__(self):
            calls.append(1)
            raise RuntimeError('hostile __dict__ descriptor')

    model, element, _options, _staged = _case('q4_staged')
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'numpy.capture_probe', Hostile())
        _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element))
        assert _calls(counts, 'trusted_post_observation_guard') == 0 and _calls(counts, 'post_observation_guard') > 0
    assert calls == []
    _recipe, counts = _profiled(lambda: _capture_owned_local_recovery(model, element))  # entry removed: trusted again
    assert _calls(counts, 'trusted_post_observation_guard') > 0


_COLLIDING_SYS_MODULES_KEY = """
import json, sys
calls = []
target = 'anyfesolver.prepared_static_recovery'
class Key(str):
    def __hash__(self):
        return hash(target)
    def __eq__(self, other):
        caller = sys._getframe(1)
        if caller.f_code.co_filename.endswith('prepared_static_recovery.py'):
            calls.append(caller.f_code.co_name)
        return False
    __ne__ = str.__ne__
sys.modules[Key('capture-probe-colliding-key')] = None
sys.path[:0] = json.loads(sys.argv[1])
import anyfesolver.prepared_static_response
print(json.dumps({'calls_from_this_module': calls}))
"""


def test_import_makes_no_sys_modules_lookup_of_its_own():
    """Opt-in review round 6: this module's import-time code performs no keyed
    sys.modules lookup (a colliding str-subclass key could run code there)."""
    paths = json.dumps([entry for entry in sys.path if entry])
    done = subprocess.run([sys.executable, '-c', _COLLIDING_SYS_MODULES_KEY, paths], capture_output=True, text=True,
                          timeout=120, check=True)
    made = json.loads(done.stdout.strip().splitlines()[-1])
    assert made == {'calls_from_this_module': []}


_HOSTILE_WARNINGS_MODULE = """
import json, sys, types, warnings
calls = []
plain = types.ModuleType.__dict__['__dict__']
class HostileModule(types.ModuleType):
    @property
    def __dict__(self):
        caller = sys._getframe(1)
        if caller.f_code.co_filename.endswith('prepared_static_recovery.py'):
            calls.append(caller.f_code.co_name)
        return plain.__get__(self)
warnings.__class__ = HostileModule
sys.path[:0] = json.loads(sys.argv[1])
import anyfesolver.prepared_static_response
from anyfesolver import prepared_static_recovery as recovery
after_import = list(calls)
admissible = recovery._capture_modules_unhooked()
during_check = list(calls)
print(json.dumps({'after_import': after_import, 'during_check': during_check, 'opt_in_admissible': admissible}))
"""


def test_capture_state_inspection_is_not_done_at_import():
    """Opt-in review round 7: the capture state table (warnings, sys, builtins
    namespaces) is built on the first opt-in admission, after the module check, and
    never through a module's own __dict__ descriptor."""
    paths = json.dumps([entry for entry in sys.path if entry])
    done = subprocess.run([sys.executable, '-c', _HOSTILE_WARNINGS_MODULE, paths], capture_output=True, text=True,
                          timeout=120, check=True)
    made = json.loads(done.stdout.strip().splitlines()[-1])
    assert made == {'after_import': [], 'during_check': [], 'opt_in_admissible': False}


_RELOAD_WITH_PLANTED_KEY = """
import dis, importlib, json, sys
sys.path[:0] = json.loads(sys.argv[1])
import anyfesolver.prepared_static_response
from anyfesolver import boundary, prepared_static_recovery as recovery
calls, import_statements = [], []
def opname(frame):
    for instruction in dis.get_instructions(frame.f_code):
        if instruction.offset == frame.f_lasti:
            return instruction.opname
    return None
class Key(str):
    def __eq__(self, other):
        caller = sys._getframe(1)
        if caller.f_code.co_filename.endswith('prepared_static_recovery.py'):
            # The import system's own reads during an import statement (Python 3.12
            # reads the source module's __name__ for every `from .boundary import`)
            # are outside the scope; only the change's own keyed reads count.
            if opname(caller) in ('IMPORT_NAME', 'IMPORT_FROM'):
                import_statements.append(caller.f_code.co_name)
            else:
                calls.append(caller.f_code.co_name)
        return str.__eq__(self, other)
    __hash__ = str.__hash__
namespace = vars(boundary)
value = namespace.pop('__name__')
namespace[Key('__name__')] = value
importlib.reload(recovery)
print(json.dumps({'calls_during_reload': calls, 'import_statement_reads': import_statements}))
"""


def test_initialization_makes_no_keyed_reads_of_other_modules_globals():
    """Opt-in review round 8: recovery's (re)initialization reads no other module's
    globals by key (a planted str-subclass key could run its __eq__ there); the
    capture facts are built on the first opt-in check."""
    paths = json.dumps([entry for entry in sys.path if entry])
    done = subprocess.run([sys.executable, '-c', _RELOAD_WITH_PLANTED_KEY, paths], capture_output=True, text=True,
                          timeout=120, check=True)
    made = json.loads(done.stdout.strip().splitlines()[-1])
    assert made['calls_during_reload'] == []
