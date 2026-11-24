"""Coupled route: regression tests for review round 6 (GPT 6.2 Astra) under owner decision OD14: the whole
certificate runs in the isolated child process (docs/SOLID_COUPLING.md, "Certificate process"). Each
test names its finding; the fails-on-3d1227284 evidence is in the commit message and the handover."""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import types

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

from anyfesolver import _coupled_certificate as cc  # noqa: E402
from anyfesolver import solid_coupling as sc  # noqa: E402
from anyfesolver.boundary import LoadCase  # noqa: E402
from test_coupled_review_r1 import refused, small, solve  # noqa: E402
from test_coupled_review_r5 import _alive, _all_gone, _descendants  # noqa: E402

_ANYFESOLVER = os.path.dirname(os.path.abspath(cc.__file__))
_PROBES = frozenset(('_fp_environment_reason', '_subnormals_preserved', '_nearest_rounding'))


# --- P1: FTZ/DAZ toggled by a parent hook around single lines cannot change the certificate --------------------

def _controlfp():
    if sys.platform != 'win32':
        pytest.skip('denormal flushing is switched through the MSVC runtime')
    crt = ctypes.CDLL('ucrtbase')
    crt._controlfp.restype = ctypes.c_uint
    crt._controlfp.argtypes = [ctypes.c_uint, ctypes.c_uint]
    return crt._controlfp


_DN_MASK, _DN_FLUSH = 0x03000000, 0x01000000


def _counterexample():
    """The reviewer's matrix: diagonal d = 2^-1022, off-diagonal o = d - 2^-1074 (a subnormal); exact
    condition number (d + o) / (d - o) = 2^53 - 1."""
    d = cc._float_from_bits(1 << 52)
    o = cc._float_from_bits((1 << 52) - 1)
    return sparse.csr_matrix(np.array([[d, o], [o, d]]))


@contextlib.contextmanager
def _flush_on_parent_lines():
    """A trace hook that switches FTZ/DAZ on before every line of anyfesolver code in this process and
    off whenever an FP-environment probe runs (so the probes never see it): r6 P1, generalised from
    the reviewer's np.ldexp line to every line of the parent's route."""
    controlfp = _controlfp()
    old = controlfp(0, 0)
    state = {'flushed_lines': 0}

    def in_probe(frame):
        while frame is not None:
            if frame.f_code.co_name in _PROBES:
                return True
            frame = frame.f_back
        return False

    def local(frame, event, arg):
        if event == 'line':
            if in_probe(frame):
                controlfp(old, _DN_MASK)
            else:
                controlfp(_DN_FLUSH, _DN_MASK)
                state['flushed_lines'] += 1
        return local

    def tracer(frame, event, arg):
        if frame.f_code.co_name in _PROBES:
            controlfp(old, _DN_MASK)
            return None
        if frame.f_code.co_filename.startswith(_ANYFESOLVER):
            return local
        return None
    sys.settrace(tracer)
    try:
        yield state
    finally:
        sys.settrace(None)
        controlfp(old, _DN_MASK)


def test_r6_p1_flush_on_parent_lines_cannot_certify_the_wrong_matrix():
    """On 3d1227284 the parent equilibrated: FTZ/DAZ around its np.ldexp dropped the subnormal
    off-diagonals and the certificate accepted kappa_up 1.0 for exact kappa 2^53 - 1. The child now
    equilibrates the stored bytes: the solve refuses condition_above_limit."""
    S = _counterexample()
    reference = refused('COUPLED_UNCERTIFIED', cc.certified_solve, S, np.zeros(2), raw_load_zero=True)
    assert reference.reason in ('condition_above_limit', 'singular_within_svd_error'), reference.reason
    with _flush_on_parent_lines() as state:
        with pytest.raises(cc.CoupledSolveError) as info:
            cc.certified_solve(S, np.zeros(2), raw_load_zero=True)
    assert state['flushed_lines'] > 20, state
    assert (info.value.code, info.value.reason) == (reference.code, reference.reason), \
        (info.value.code, info.value.reason)
    certificate = info.value.diagnostics['certificate']
    assert certificate == reference.diagnostics['certificate']
    assert certificate['accepted'] is False and certificate['condition_upper_bound'] >= 2.0 ** 53 - 1


def test_r6_p1_flush_on_parent_lines_keeps_an_accepted_solve_bit_identical():
    """The same hook on an ordinary certified solve: either a coded refusal from the parent's own
    probes or the bit-identical certified result (here: the result; the parent sends bytes only)."""
    S = sparse.csr_matrix(np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]))
    b = np.array([1.0, 2.0, 3.0])
    reference = cc.certified_solve(S, b, raw_load_zero=False)
    with _flush_on_parent_lines() as state:
        try:
            result = cc.certified_solve(S, b, raw_load_zero=False)
        except cc.CoupledSolveError as exc:
            assert exc.code == 'COUPLED_UNCERTIFIED', (exc.code, exc.reason)
            return
    assert state['flushed_lines'] > 20
    assert np.array_equal(result.x.view(np.int64), reference.x.view(np.int64))
    assert result.forward_error == reference.forward_error
    assert result.certificate == reference.certificate


# --- P1: the child imports nothing from a caller-controlled directory ----------------------------------------

def _shadow_dir(tmp_path, numbers_body):
    """A directory holding a genuine copy of threadpoolctl.py next to a numbers.py (NumPy imports
    ``numbers`` during initialisation)."""
    import threadpoolctl
    shutil.copy(threadpoolctl.__file__, tmp_path / 'threadpoolctl.py')
    (tmp_path / 'numbers.py').write_text(numbers_body)
    module = types.ModuleType('threadpoolctl')
    module.__file__ = str(tmp_path / 'threadpoolctl.py')
    return module


def test_r6_p1_numbers_next_to_threadpoolctl_is_not_imported(monkeypatch, tmp_path):
    marker = tmp_path / 'ran.txt'
    body = f"open({str(marker)!r}, 'a').write('numbers\\n')\nfrom fractions import *\n"
    monkeypatch.setitem(sys.modules, 'threadpoolctl', _shadow_dir(tmp_path, body))
    command = cc._svd_worker_command()
    certificate = cc._certify_solve_matrix(sparse.csr_matrix(np.diag([4.0, 2.0, 1.0])))
    assert certificate['accepted'] is True
    assert not marker.exists(), (command, marker.read_text())


def test_r6_p1_worker_path_is_the_standard_library_only():
    """The child's sys.path holds only the standard library; only numpy, threadpoolctl and the
    certificate arithmetic are imported outside it."""
    decoded, report = cc._run_worker(sparse.csr_matrix(np.eye(3)), None, mode='certificate', raw_load_zero=False,
                                     memory_limit=None)
    assert decoded['certificate']['accepted'] is True
    assert not any('site-packages' in p for p in report['sys_path']), report['sys_path']
    outside = set(report['modules']) - set(sys.stdlib_module_names) - {'__main__', '_coupled_certificate_math'}
    assert outside <= {'numpy', 'threadpoolctl'}, outside
    assert '_coupled_certificate_math' in report['modules']


# --- P2: Popen cleanup, bounded waits, MemoryError mapping ------------------------------------------------------

def _is_popen_init(frame):
    return frame.f_code.co_name == '__init__' and frame.f_code.co_filename.endswith('subprocess.py') \
        and isinstance(frame.f_locals.get('self'), subprocess.Popen)


def test_r6_p2_hook_after_popen_init_kills_the_child(monkeypatch):
    """An exception on return from Popen.__init__ (after the OS created the process): the refusal is
    start_failed and the child is gone while the refusal is still held.

    Review r8: Popen now runs on the launch lane (a ``_thread`` thread), which a profile hook set on the
    solving thread does not reach (and in-process hooks are out of scope, OD15). The same failure is
    therefore raised by a test-side wrapper of Popen.__init__ after the process exists."""
    import subprocess
    seen, state = [], {'fired': False}
    original = subprocess.Popen.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not state['fired']:
            state['fired'] = True
            seen.extend(_descendants(os.getpid()))
            raise PermissionError('denied after creation')
    monkeypatch.setattr(subprocess.Popen, '__init__', init)
    with pytest.raises(cc.CoupledSolveError) as info:
        cc._certify_solve_matrix(sparse.csr_matrix(np.diag([4.0, 2.0, 1.0])))
    assert state['fired'] and seen, seen
    assert (info.value.code, info.value.reason) == ('COUPLED_UNCERTIFIED', 'svd_worker')
    assert info.value.diagnostics['worker_failure'] == 'start_failed'
    held = info.value                            # the refusal and its traceback stay referenced
    assert _all_gone(seen, wait=5.0), [p for p in seen if _alive(p)]
    assert held is info.value


def test_r6_p2_lane_start_does_not_wait_on_the_thread_registry():
    """The reviewer's cycle: the thread registry lock held (by a parent hook, here by this test) while
    the solve starts its transfer lanes. Thread.start() waited on it; the lanes do not."""
    lock = getattr(threading, '_active_limbo_lock', None)
    if lock is None:
        pytest.skip('this Python has no thread registry lock')
    go, outcome = threading.Event(), {}

    def run():
        go.wait(30)
        try:
            outcome['certificate'] = cc._certify_solve_matrix(sparse.csr_matrix(np.diag([4.0, 2.0, 1.0])))
        except BaseException as exc:  # noqa: BLE001
            outcome['error'] = exc
        finally:
            outcome['done'] = time.monotonic()
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    deadline = time.monotonic() + 30.0
    with lock:
        go.set()
        while 'done' not in outcome and time.monotonic() < deadline:
            time.sleep(0.05)
        finished_under_lock = 'done' in outcome
    worker.join(120)
    assert finished_under_lock, 'the certificate waited on the thread registry lock'
    assert outcome.get('certificate', {}).get('accepted') is True, outcome


_GRANDCHILD = textwrap.dedent('''
    import subprocess, sys
    subprocess.Popen([sys.executable, '-I', '-S', '-c', 'import time; time.sleep(25)'],
                     stdout=sys.stdout, stderr=sys.stderr, close_fds=False)
    sys.exit(7)
''')


def test_r6_p2_teardown_is_bounded_when_a_pipe_stays_open(monkeypatch):
    """The worker exits while a process it started keeps its stdout pipe open, so the reader lane
    stays blocked: the refusal comes within the bounded IO wait (no close() on a blocked stream)."""
    monkeypatch.setattr(cc, '_svd_worker_command', lambda *args: [sys.executable, '-I', '-S', '-c', _GRANDCHILD])
    monkeypatch.setattr(cc, '_SVD_WORKER_IO_SECONDS', 1.0)
    if hasattr(cc, '_SVD_WORKER_STOP_SECONDS'):
        monkeypatch.setattr(cc, '_SVD_WORKER_STOP_SECONDS', 1.0)
    before = set(_descendants(os.getpid()))
    started = time.monotonic()
    try:
        with pytest.raises(cc.CoupledSolveError) as info:
            cc._certify_solve_matrix(sparse.csr_matrix(np.diag([4.0, 2.0, 1.0])))
        elapsed = time.monotonic() - started
    finally:
        extra = set(_descendants(os.getpid())) - before
        _all_gone(extra, wait=30.0)              # the grandchild ends by itself after 25 s
    assert (info.value.code, info.value.reason) == ('COUPLED_UNCERTIFIED', 'svd_worker')
    assert elapsed < 15.0, elapsed


def test_r6_p2_reader_memory_error_is_too_large(monkeypatch):
    original = cc._WorkerIO._read

    def failing(self, stream, sink, limit):
        if sink is self.stdout:
            raise MemoryError('reader buffer')
        return original(self, stream, sink, limit)
    monkeypatch.setattr(cc._WorkerIO, '_read', failing)
    refused('COUPLED_TOO_LARGE', cc._certify_solve_matrix, sparse.csr_matrix(np.diag([4.0, 2.0, 1.0])),
            reason='memory_error')


def test_r6_p2_child_blas_probe_memory_error_is_too_large(monkeypatch, tmp_path):
    """The child's BLAS probe treated MemoryError as a probe failure (blas_threads)."""
    # Review r8: the seam is the threadpoolctl bytes the parent sends (sys.modules no longer selects them).
    from _coupled_worker_patch import patch_threadpoolctl
    patch_threadpoolctl(monkeypatch, cc, "def threadpool_info():\n    raise MemoryError('probe')\n")
    refused('COUPLED_TOO_LARGE', cc._certify_solve_matrix, sparse.csr_matrix(np.diag([4.0, 2.0, 1.0])),
            reason='memory_error')


# --- P2: coded refusals while the cancellation refusal is built and in the top-level translation ---------------

def _cancel_after(calls):
    state = {'n': 0}

    def cancel():
        state['n'] += 1
        return state['n'] > calls
    return cancel


@pytest.mark.parametrize('where', ['refuse_call', 'check_return'])
def test_r6_p2_hook_while_the_cancellation_is_built_still_cancels(where):
    """A one-shot profile exception while a genuine cancellation is built (the reviewer's case: at
    the refusal's construction; also on the unwinding return of the check) refuses COUPLED_CANCELLED."""
    model, grid, removed, prepared, case, box = small()
    state = {'fired': False}

    def profile(frame, event, arg):
        if state['fired']:
            return
        if where == 'refuse_call' and event == 'call' and frame.f_code.co_name == '_refuse' \
                and frame.f_locals.get('code') == 'COUPLED_CANCELLED':
            state['fired'] = True
            raise PermissionError('denied by policy')
        if where == 'check_return' and event == 'return' and frame.f_code.co_name == 'check' \
                and frame.f_code.co_filename.endswith('solid_coupling.py') and frame.f_locals.get('value') is True:
            state['fired'] = True
            raise PermissionError('denied by policy')
    sys.setprofile(profile)
    try:
        error = refused('COUPLED_CANCELLED', solve, model, prepared, removed, box, case, cancel=_cancel_after(3))
    finally:
        sys.setprofile(None)
    assert state['fired'], where
    assert error.reason is None


def _memory_error_route(monkeypatch):
    def out_of_memory(*args):
        raise MemoryError('route')
    monkeypatch.setattr(sc, '_check_load_route', out_of_memory)
    return small()


def _sweep_hook(kind, index):
    """A one-shot hook raising PermissionError at the index-th event after the route's MemoryError was
    raised and before the route's entry function returns: kind 'trace' (every line and exception event
    in solid_coupling and _coupled_certificate frames) or 'profile' (every profile event). A hook
    raising at the return of the entry function itself acts in the caller and is not counted."""
    state = {'armed': False, 'count': 0, 'fired': False}

    def entry_returns(frame, event):
        if event == 'return' and frame.f_code.co_name == 'solve_shell_solid_coupled':
            state['armed'] = False
            return True
        return False

    def should_fire():
        if not state['armed'] or state['fired']:
            return False
        state['count'] += 1
        if state['count'] > index:
            state['fired'] = True
            return True
        return False

    def ours(frame):
        return frame.f_code.co_filename.endswith(('solid_coupling.py', '_coupled_certificate.py'))

    def local(frame, event, arg):
        if entry_returns(frame, event):
            return local
        if event == 'exception' and isinstance(arg[1], MemoryError) and frame.f_code.co_name == '_solve':
            state['armed'] = True
        elif event in ('line', 'exception') and should_fire():
            state['event'] = event
            raise PermissionError('trace hook')
        return local

    def tracer(frame, event, arg):
        return local if ours(frame) else None

    def profile(frame, event, arg):
        if entry_returns(frame, event):
            return
        if event == 'return' and frame.f_code.co_name == '_solve' and frame.f_code.co_filename.endswith(
                'solid_coupling.py'):
            state['armed'] = True
            return
        if should_fire():
            state['event'] = event
            raise PermissionError('profile hook')
    return state, (tracer if kind == 'trace' else profile)


@pytest.mark.parametrize('kind', ['trace', 'profile'])
def test_r6_p2_hook_anywhere_in_the_translation_keeps_the_code(monkeypatch, kind):
    """The reviewer's case (a one-shot trace exception at entry to the top-level translation while
    translating MemoryError) as a sweep: the hook raises at the k-th event of the translation, for
    every k until it no longer fires. Every run refuses COUPLED_TOO_LARGE (memory_error)."""
    model, grid, removed, prepared, case, box = _memory_error_route(monkeypatch)
    outcomes = []
    for index in range(80):
        state, hook = _sweep_hook(kind, index)
        setter = sys.settrace if kind == 'trace' else sys.setprofile
        setter(hook)
        try:
            try:
                solve(model, prepared, removed, box, case)
                outcomes.append((index, None, 'accepted'))
            except cc.CoupledSolveError as exc:
                outcomes.append((index, state.get('event'), exc.code, exc.reason))
            except BaseException as exc:  # noqa: BLE001
                outcomes.append((index, state.get('event'), 'raw', type(exc).__name__))
        finally:
            setter(None)
        if not state['fired']:
            break
    assert not state['fired'] and len(outcomes) > 3, 'the sweep did not cover the whole translation'
    # A trace function that raises on an 'exception' event replaces the propagating exception (CPython
    # drops it, with no __context__): the hook's own exception is then translated (coded, unexpected:).
    replaced = ('COUPLED_NATIVE_INVALID', 'unexpected:PermissionError')
    bad = [o for o in outcomes if o[2:] != ('COUPLED_TOO_LARGE', 'memory_error')
           and not (o[1] == 'exception' and o[2:] == replaced)]
    assert not bad, bad


# --- P2: caller protocols never run while containers are checked or the model is copied ---------------------------

class _Meta(type):
    log: list = []

    def __hash__(cls):
        _Meta.log.append('__hash__')
        return type.__hash__(cls)

    def __eq__(cls, other):
        _Meta.log.append('__eq__')
        return type.__eq__(cls, other)


class _Key(int, metaclass=_Meta):
    pass


@pytest.mark.parametrize('field', ['nodal_loads', 'pressure_loads', 'added_node_masses', 'element_loads'])
def test_r6_p2_metaclass_hooks_do_not_run_in_the_container_check(field):
    """An integer-subclass key whose metaclass defines __hash__ and __eq__, in an untouched LoadCase:
    the exact-type check refuses container_type without hashing or comparing the key's class."""
    model, grid, removed, prepared, case, box = small()
    containers = {'pressure_loads': dict(case.pressure_loads)}
    containers[field] = dict(containers.get(field, {}))
    containers[field][_Key(10 ** 6)] = 0.0 if field != 'nodal_loads' else [0.0, 0.0, 0.0]
    new = LoadCase(name='r6', **containers)
    _Meta.log.clear()
    error = refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, new)
    assert _Meta.log == [], _Meta.log
    assert error.reason == 'container_type', error.reason


class _Watched(dict):
    calls: list = []

    def __deepcopy__(self, memo):
        _Watched.calls.append('__deepcopy__')
        return dict(self)

    def __reduce_ex__(self, protocol):
        _Watched.calls.append('__reduce_ex__')
        return dict.__reduce_ex__(self, protocol)


def test_r6_p2_registered_load_cases_are_not_copied():
    """The reviewer's case: a second load case registered through model.add_load_case whose mapping
    defines __deepcopy__. The route never copies registered load cases, and the result is unchanged."""
    model, grid, removed, prepared, case, box = small()
    reference = solve(model, prepared, removed, box, case)
    other = LoadCase(name='other', nodal_loads=_Watched({grid[1, 1]: np.zeros(6)}))
    model.add_load_case(other) if hasattr(model, 'add_load_case') else model.load_cases.append(other)
    _Watched.calls.clear()
    result = solve(model, prepared, removed, box, case)
    assert _Watched.calls == [], _Watched.calls
    assert np.array_equal(result.shell_displacements.values.view(np.int64),
                          reference.shell_displacements.values.view(np.int64))
    assert np.array_equal(result.shell_displacements.bounds.view(np.int64),
                          reference.shell_displacements.bounds.view(np.int64))


class _Name(str):
    calls: list = []

    def __deepcopy__(self, memo):
        _Name.calls.append('__deepcopy__')
        return str(self)


def test_r6_p2_load_case_name_is_an_exact_str():
    """Class member: the LoadCase copies ran a str subclass's __deepcopy__; the name must be an exact str."""
    model, grid, removed, prepared, case, box = small()
    new = LoadCase(name=_Name('r6'), pressure_loads=dict(case.pressure_loads))
    _Name.calls.clear()
    error = refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, new)
    assert _Name.calls == [], _Name.calls
    assert error.reason == 'container_type'


def test_r6_p2_registered_materials_are_not_copied():
    """Class member: a registered material whose class defines __deepcopy__ is shared, not copied."""
    model, grid, removed, prepared, case, box = small()
    reference = solve(model, prepared, removed, box, case)
    base = type(model.materials['default'])
    calls = []
    Watched = type('WatchedMaterial', (base,), {'__deepcopy__': lambda self, memo: calls.append('copy') or self})
    import copy as _copy
    extra = _copy.copy(model.materials['default'])
    extra.__class__ = Watched
    extra.name = 'watched'
    model.materials['watched'] = extra
    result = solve(model, prepared, removed, box, case)
    assert calls == [], calls
    assert np.array_equal(result.shell_displacements.values.view(np.int64),
                          reference.shell_displacements.values.view(np.int64))


# --- OD14: the parent decodes an exact schema; the result is the child's ---------------------------------------

def test_od14_result_values_are_the_childs_copy(monkeypatch):
    """The returned x is the worker's echo of the candidate it verified: changing the parent's
    candidate array after it was sent does not reach the result."""
    S = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = np.array([1.0, 2.0])
    reference = cc.certified_solve(S, b, raw_load_zero=False)
    original = cc._Run.post_candidate

    def post_then_change(self, x):
        original(self, x)
        if x is not None:
            x[:] = 123.0
    monkeypatch.setattr(cc._Run, 'post_candidate', post_then_change)
    result = cc.certified_solve(S, b, raw_load_zero=False)
    assert np.array_equal(result.x.view(np.int64), reference.x.view(np.int64))


def test_od14_refusal_precedence_is_unchanged(monkeypatch):
    """A certificate refusal wins over a candidate failure (as when the certificate ran first)."""
    def singular(*args, **kwargs):
        raise RuntimeError('Factor is exactly singular')
    monkeypatch.setattr(cc.sparse_linalg, 'splu', singular)
    S = sparse.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    refused('COUPLED_UNCERTIFIED', cc.certified_solve, S, np.array([1.0, 2.0]), raw_load_zero=False,
            reason='singular_within_svd_error')
    S = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    refused('COUPLED_UNCERTIFIED', cc.certified_solve, S, np.array([1.0, 2.0]), raw_load_zero=False,
            reason='candidate_failed')


def test_od14_coupled_solve_is_bit_identical_through_the_child():
    """The bounded recovery values and bounds computed by the child equal an in-process evaluation of
    the same maps with the same exact arithmetic (the pre-OD14 parent formula)."""
    from fractions import Fraction
    from anyfesolver import _coupled_certificate_math as m
    model, grid, removed, prepared, case, box = small()
    captured = {}
    original = sc.certified_solve

    def capture(system, rhs, **kwargs):
        solved = original(system, rhs, **kwargs)
        captured['solved'], captured['exports'] = solved, kwargs['exports']
        return solved
    sc_certified = sc.certified_solve
    try:
        sc.certified_solve = capture
        result = solve(model, prepared, removed, box, case)
    finally:
        sc.certified_solve = sc_certified
    solved = captured['solved']
    for k, export in enumerate(captured['exports']):
        coefficients = [m._exact(c) for c in export.coefficients] if type(export.coefficients) is np.ndarray \
            else list(export.coefficients)
        offsets = None if export.offsets is None else [m._exact(v) for v in export.offsets]
        values, bounds = m.bounded_rows(export.indptr, export.columns, coefficients, offsets, solved.x,
                                        solved.shifts, solved.forward_error)
        assert np.array_equal(values.view(np.int64), solved.exports[k][0].view(np.int64)), k
        assert np.array_equal(bounds.view(np.int64), solved.exports[k][1].view(np.int64)), k
    assert isinstance(solved.forward_error, Fraction)
    assert result.verification['forward_error'] == m._out_up(solved.forward_error)
