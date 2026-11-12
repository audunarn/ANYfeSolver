"""Unexecuted numerical regression proposal; requires separately adopted runtime."""
import ast
import ctypes
import builtins
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.general_contact import native_replay_arithmetic as arithmetic
from anyfesolver.general_contact import native_curved_s3_work as contact
from test_general_contact_native_curved_s3_work import _case, _corners

REFERENCE = Path(__file__).parent / 'fixtures' / 'native_gap_jet_reference'

def original(filename, name, namespace):
    tree = ast.parse((REFERENCE / filename).read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    module = ast.Module(body=matches, type_ignores=[])
    scope = dict(namespace)
    exec(compile(module, str(REFERENCE / filename), 'exec'), scope)
    return scope[name]

def matrix_original():
    return original('native_replay_arithmetic.py', 'gap_matrix_product', vars(arithmetic))

def test_native_preparation_is_exercised_when_available():
    # Runtime admission must record platform coverage; this is not an invented native hook.
    if arithmetic._native_fma is None:
        assert not arithmetic._native_preparation_allowed()
        return
    assert arithmetic._native_preparation_allowed()

@pytest.mark.parametrize('kind', ['cancellation', 'subnormal', 'signed_zero', 'extreme'])
def test_prepared_against_frozen_original_bytes(kind, monkeypatch):
    e = 2.**-27
    cases = {
        'cancellation': ([[-1., 1.+e, 0.]], [[1.], [1.-e], [0.]]),
        'subnormal': ([[2.**-1074]*3], [[.5], [.5], [.5]]),
        'signed_zero': ([[-0., 0., -0.]], [[-0., 1.], [0., -1.], [-0., 0.]]),
        'extreme': ([[2.**1023, -2.**1023, 1.]], [[1., 2.], [1., 0.], [2.**-1022, 1.]])}
    left, right = map(np.array, cases[kind])
    left = np.tile(left, (9, 1))
    right = np.tile(right, (1, 9))[:, :9]
    assert left.shape == (9, 3) and right.shape == (3, 9)
    oracle = matrix_original()
    for native in (True, False):
        with monkeypatch.context() as patch:
            if not native: patch.setattr(arithmetic, '_fma', arithmetic._portable_fma)
            # Overflow behavior remains the same as the original selected primitive.
            oracle = matrix_original()
            try:
                expected = oracle(left, right)
            except (OverflowError, ValueError) as error:
                with pytest.raises(type(error)): arithmetic.gap_matrix_product(left, right)
            else:
                assert arithmetic.gap_matrix_product(left, right).tobytes() == expected.tobytes()


def test_each_call_reads_mutated_and_rebound_right():
    left = np.arange(27, dtype=float).reshape(9, 3)
    right = np.arange(27, dtype=float).reshape(3, 9)
    for replacement in (False, True):
        arithmetic.gap_matrix_product(left, right)
        right[0, 0] = -0.
        right[2, 4] = -13.
        if replacement: right = right.copy()[:, ::-1]
        assert arithmetic.gap_matrix_product(left, right).tobytes() == matrix_original()(left, right).tobytes()


def test_python_hook_in_call_right_mutation_and_rebinding(monkeypatch):
    primitive = arithmetic._portable_fma
    def run(function):
        right = np.ones((3, 9)); calls = []
        def replacement(a, b, c):
            calls.append('replacement')
            return primitive(a, b, c)
        def hook(a, b, c):
            calls.append('initial')
            right[0, 1] = 7.
            arithmetic._fma = replacement
            return primitive(a, b, c)
        monkeypatch.setattr(arithmetic, '_fma', hook)
        assert not arithmetic._native_preparation_allowed()
        # Build frozen oracle after the hook is installed; preserve its own global lookup.
        if function == 'original':
            oracle = matrix_original()
            # Its globals must be the live namespace for rebinding behavior.
            def original_hook(a, b, c):
                calls.append('initial'); right[0, 1] = 7.
                oracle.__globals__['_fma'] = replacement
                return primitive(a, b, c)
            oracle.__globals__['_fma'] = original_hook
            value = oracle(np.ones((9, 3)), right)
        else: value = arithmetic.gap_matrix_product(np.ones((9, 3)), right)
        return value.tobytes(), calls
    assert run('candidate') == run('original')


def test_python_hook_cancellation_propagates(monkeypatch):
    class Cancelled(BaseException): pass
    def hook(*args): raise Cancelled()
    monkeypatch.setattr(arithmetic, '_fma', hook)
    assert not arithmetic._native_preparation_allowed()
    with pytest.raises(Cancelled): arithmetic.gap_matrix_product(np.ones((9, 3)), np.ones((3, 9)))


def test_native_errcheck_mutation_uses_original_loop(monkeypatch):
    native = arithmetic._native_fma
    if native is None:
        assert not arithmetic._native_preparation_allowed()
        return
    def run(oracle):
        right = np.ones((3, 9)); calls = []
        def errcheck(result, function, arguments):
            calls.append(1); right[0, 1] = 7.
            return result
        native.errcheck = errcheck
        try:
            assert not arithmetic._native_preparation_allowed()
            fn = matrix_original() if oracle else arithmetic.gap_matrix_product
            return fn(np.ones((9, 3)), right).tobytes(), len(calls)
        finally: del native.errcheck
    assert run(False) == run(True)


def test_native_configuration_mutations_decline():
    native = arithmetic._native_fma
    if native is None: return
    saved = native.argtypes, native.restype
    try:
        native.argtypes = (ctypes.c_float, ctypes.c_double, ctypes.c_double)
        assert not arithmetic._native_preparation_allowed()
        native.argtypes = saved[0]; native.restype = ctypes.c_float
        assert not arithmetic._native_preparation_allowed()
    finally: native.argtypes, native.restype = saved

@pytest.mark.parametrize('right', [np.ones((9, 3)), np.full((3, 9), np.nan), np.full((3, 9), np.inf)])
def test_invalid_right_retains_refusal(right):
    with pytest.raises(ValueError): arithmetic.gap_matrix_product(np.ones((9, 3)), right)
    with pytest.raises(ValueError): matrix_original()(np.ones((9, 3)), right)

@pytest.mark.parametrize('tilt,shift,arm', [
    (0., 0., (.001, .002, -.03)),
    (.01, 0., (-.03, -.0, .002)),
    (.01, 2.**48, (.001, .002, -.03))])
def test_raw_point_full_jet_matches_frozen_original(tilt, shift, arm, monkeypatch):
    work, high, low, rotations, u = _case()
    corners = _corners(work, u); corners[1, 2] += tilt
    corners[:, 0] += shift; high[:, 0] += shift
    for fallback in (False, True):
        with monkeypatch.context() as patch:
            if fallback: patch.setattr(arithmetic, '_fma', arithmetic._portable_fma)
            candidate = contact._S3CandidateIntegration(work, corners)
            namespace = dict(vars(contact)); namespace['gap_matrix_product'] = matrix_original()
            oracle = original('native_curved_s3_work.py', '_gap_work_jet', namespace)
            for station in (0., .1, 1.):
                for repeat in range(2):
                    args = (-.09, high, low, 1, station, np.array(arm))
                    actual = candidate._gap_work_jet(*args); expected = oracle(candidate, *args)
                    assert actual.value == expected.value
                    assert actual.gradient.tobytes() == expected.gradient.tobytes()
                    assert actual.hessian.tobytes() == expected.hessian.tobytes()
                candidate.dn = candidate.dn.copy(); candidate.dn[0, 0] = -0.
                actual = candidate._gap_work_jet(*args); expected = oracle(candidate, *args)
                assert actual.gradient.tobytes() == expected.gradient.tobytes()
                assert actual.hessian.tobytes() == expected.hessian.tobytes()


def test_custom_scalar_conversion_declines(monkeypatch):
    original = ctypes.c_double.from_param
    monkeypatch.setattr(ctypes.c_double, 'from_param', classmethod(lambda cls, value: original(value)))
    assert not arithmetic._native_preparation_allowed()

@pytest.mark.parametrize('left', [np.ones((2, 2)), np.full((9, 3), np.nan), np.full((9, 3), np.inf)])
def test_invalid_left_retains_refusal(left):
    with pytest.raises(ValueError): arithmetic.gap_matrix_product(left, np.ones((3, 9)))
    with pytest.raises(ValueError): matrix_original()(left, np.ones((3, 9)))


def test_target_shape_conversion_count_and_legacy_hook(monkeypatch):
    calls = []
    def mapping(function, values):
        calls.append(1)
        return builtins.map(function, values)
    monkeypatch.setattr(arithmetic, 'map', mapping, raising=False)
    left = np.ones((9, 3)); right = np.ones((3, 9))
    eligible = arithmetic._native_preparation_allowed()
    actual = arithmetic.gap_matrix_product(left, right)
    assert len(calls) == (18 if eligible else 90)
    assert actual.tobytes() == matrix_original()(left, right).tobytes()
    calls.clear()
    primitive = arithmetic._fma
    def hook(a, b, c): return primitive(a, b, c)
    monkeypatch.setattr(arithmetic, '_fma', hook)
    assert not arithmetic._native_preparation_allowed()
    arithmetic.gap_matrix_product(left, right)
    assert len(calls) == 90
