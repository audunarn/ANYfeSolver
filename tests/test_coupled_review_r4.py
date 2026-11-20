"""Coupled route: regression tests for Codex code review round 4 (1 P1 + 4 P2, owner decision OD12).
Each test names its finding; the fails-on-6ac863fa0 evidence is in the commit message."""

from __future__ import annotations

import contextlib
import contextvars
import os
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

from anyfesolver import _coupled_certificate as cc  # noqa: E402
from test_coupled_review_r1 import refused, small, solve  # noqa: E402
from test_coupled_review_r2 import rounding  # noqa: E402


def _blas_threads():
    from threadpoolctl import threadpool_info
    return [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']


# --- P1: caller code inside the SVD window cannot re-enter the coordinator and change the pools ----------

def _in_certificate_svd(frame):
    return (frame.f_code.co_name == 'svd' and 'linalg' in frame.f_code.co_filename
            and frame.f_back is not None and frame.f_back.f_code.co_name == '_svd_values')


@pytest.mark.parametrize('mode', ['propagate', 'swallow', 'fresh_context'])
def test_r4_p1_same_thread_callback_reenters_native_thread_scope_in_the_svd_window(mode):
    """The reviewer's schedule: a same-thread profile callback enters native_thread_scope(4) when NumPy's
    SVD wrapper is called from the certificate and closes it when the wrapper returns. The solve must
    refuse (blas_threads) or the SVD must see one thread; acceptance on four threads must not happen.
    'swallow' catches whatever the re-entry raises; 'fresh_context' re-enters from an empty context."""
    from anyfesolver.threading_policy import native_thread_scope
    stack = contextlib.ExitStack()
    state = {'fired': 0, 'entered': 0, 'svd_threads': [], 'errors': []}

    def enter():
        stack.enter_context(native_thread_scope(4, phase='r4-reentrant'))

    def profile(frame, event, arg):
        if event == 'call' and _in_certificate_svd(frame):
            state['fired'] += 1
            try:
                if mode == 'fresh_context':
                    contextvars.Context().run(enter)
                else:
                    enter()
                state['entered'] += 1
            except Exception as exc:  # noqa: BLE001
                state['errors'].append(type(exc).__name__)
                if mode == 'propagate':
                    raise
            state['svd_threads'].append(_blas_threads())
        elif event == 'return' and _in_certificate_svd(frame):
            stack.close()
    model, grid, removed, prepared, case, box = small()
    before = _blas_threads()
    sys.setprofile(profile)
    try:
        try:
            result = solve(model, prepared, removed, box, case)
        except cc.CoupledSolveError as exc:
            assert exc.code == 'COUPLED_UNCERTIFIED' and exc.reason == 'blas_threads', (exc.code, exc.reason)
            result = None
    finally:
        sys.setprofile(None)
        stack.close()
    assert state['fired'], 'the certificate SVD was not reached'
    assert _blas_threads() == before
    if result is not None:
        assert state['svd_threads']
        assert all(set(t) == {1} for t in state['svd_threads']), state


def test_r4_p1_same_limit_reentry_inside_the_window_is_harmless():
    """Control: a callback that re-enters native_thread_scope(1) (the window's own limit) changes nothing;
    the solve accepts with the reference result."""
    from anyfesolver.threading_policy import native_thread_scope
    model, grid, removed, prepared, case, box = small()
    reference = solve(model, prepared, removed, box, case)
    seen = []

    def profile(frame, event, arg):
        if event == 'call' and _in_certificate_svd(frame):
            with native_thread_scope(1, phase='r4-same-limit'):
                seen.append(_blas_threads())
    sys.setprofile(profile)
    try:
        result = solve(model, prepared, removed, box, case)
    finally:
        sys.setprofile(None)
    assert seen and all(set(t) == {1} for t in seen)
    assert np.array_equal(result.shell_displacements.values, reference.shell_displacements.values)


def test_r4_p1_reentry_outside_the_window_still_works():
    """Control: after a refused re-entry the coordinator is free; scopes on this and another thread run."""
    from anyfesolver.threading_policy import native_thread_scope
    with native_thread_scope(4, phase='r4-after'):
        assert set(_blas_threads()) == {4}
    ran = threading.Event()

    def other():
        with native_thread_scope(2, phase='r4-after-other'):
            ran.set()
    worker = threading.Thread(target=other, daemon=True)
    worker.start()
    worker.join(30)
    assert ran.is_set()


# --- P2: a parent holding a default reader and joining a child solve must not hang ---------------------------

def _child_solve(outcome, *args, **kwargs):
    try:
        outcome['result'] = solve(*args, **kwargs)
    except cc.CoupledSolveError as exc:
        outcome['error'] = exc
    except BaseException as exc:  # noqa: BLE001
        outcome['other'] = exc


def test_r4_p2_parent_reader_joining_a_child_solve_refuses_or_proceeds(monkeypatch):
    """The reviewer's schedule: the parent holds native_thread_scope(None) and joins a child solve. The child
    must refuse with a code or finish; it must not wait for the parent's reader (hard timeout 60 s). The
    coordinator wait bound is shortened for the test; the production bound is checked separately."""
    from anyfesolver.threading_policy import native_thread_scope
    monkeypatch.setattr(cc, '_COORDINATOR_WAIT_SECONDS', 2.0, raising=False)
    model, grid, removed, prepared, case, box = small()
    outcome = {}
    child = threading.Thread(target=_child_solve, args=(outcome, model, prepared, removed, box, case), daemon=True)
    with native_thread_scope(None, phase='r4-parent'):
        child.start()
        child.join(60)
        hung = child.is_alive()
    child.join(120)
    assert not hung, 'the child solve waited for the parent reader'
    assert 'other' not in outcome, outcome
    if 'error' in outcome:
        assert (outcome['error'].code, outcome['error'].reason) == ('COUPLED_UNCERTIFIED', 'blas_threads')
    else:
        assert 'result' in outcome


def test_r4_p2_bounded_writer_wait_times_out_and_restores_the_reader():
    """Coordinator unit check: a bounded writer request from a thread that holds a reader, while another
    thread holds a reader too, times out and leaves the caller's reader registered and releasable."""
    from anyfesolver import threading_policy as tp
    holding, release = threading.Event(), threading.Event()

    def other():
        with tp.native_thread_scope(None, phase='r4-other-reader'):
            holding.set()
            release.wait(30)
    worker = threading.Thread(target=other, daemon=True)
    worker.start()
    assert holding.wait(10)
    try:
        with tp.native_thread_scope(None, phase='r4-own-reader'):
            with pytest.raises(tp.NativeThreadScopeTimeout):
                with tp.native_thread_scope(3, phase='r4-bounded', wait_timeout=0.2):
                    pass
            assert tp._NATIVE_SCOPE_COORDINATOR._reader_depth.get(threading.get_ident()) == 1
    finally:
        release.set()
        worker.join(30)
    assert tp._NATIVE_SCOPE_COORDINATOR._reader_count == 0
    with tp.native_thread_scope(3, phase='r4-free', wait_timeout=1.0) as report:
        assert report['coordination'] == 'writer'


def test_r4_p1_seal_counts_a_swallowed_reentry():
    """Coordinator unit check: inside a sealed window a nested explicit scope raises; the violation is
    reported after the window although the caller caught it; outside the window nesting is reentrant."""
    from anyfesolver import threading_policy as tp
    with tp.native_thread_scope(2, phase='r4-outer'):
        with tp._sealed_native_window() as seal:
            with tp.native_thread_scope(2, phase='r4-same-limit') as same:
                assert same['status'] == 'inherited_limit'
            try:
                with tp.native_thread_scope(4, phase='r4-nested'):
                    pass
            except tp.NativeThreadScopeSealed:
                pass
        assert seal['violated'] is True
        with tp._sealed_native_window() as clean:
            pass
        assert clean['violated'] is False
        with tp.native_thread_scope(4, phase='r4-nested-after') as nested:
            assert nested['coordination'] == 'writer_reentrant'


def test_r4_p2_the_coordinator_wait_is_bounded():
    """The production wait bound is finite and long enough for another solve's SVD window (<= 45 s)."""
    assert 60.0 <= cc._COORDINATOR_WAIT_SECONDS <= 600.0


def test_r4_p2_a_short_reader_on_another_thread_is_waited_for():
    """Control: a default reader on another thread that ends after 0.5 s delays the certificate; the solve
    then accepts with the reference result."""
    from anyfesolver.threading_policy import native_thread_scope
    model, grid, removed, prepared, case, box = small()
    reference = solve(model, prepared, removed, box, case)
    holding, release = threading.Event(), threading.Event()

    def reader():
        with native_thread_scope(None, phase='r4-short-reader'):
            holding.set()
            release.wait(0.5)
    worker = threading.Thread(target=reader, daemon=True)
    worker.start()
    assert holding.wait(10)
    result = solve(model, prepared, removed, box, case)
    worker.join(30)
    assert np.array_equal(result.shell_displacements.values, reference.shell_displacements.values)


# --- P2: any error in the coordinator's pool snapshot refuses blas_threads ---------------------------------

@pytest.mark.parametrize('failing_call', [1, 2, 3], ids=['before', 'active', 'after'])
@pytest.mark.parametrize('error', [PermissionError, OSError, RuntimeError])
def test_r4_p2_coordinator_snapshot_error_refuses_blas_threads(monkeypatch, failing_call, error):
    """The coordinator's own threadpoolctl snapshot (pools before, active, after) raises inside the
    certificate's scope: the route refuses COUPLED_UNCERTIFIED (blas_threads), never unexpected:*, and the
    coordinator and the pools are released afterwards."""
    from anyfesolver import threading_policy as tp
    from threadpoolctl import threadpool_info
    calls = {'n': 0}

    def info():
        if any(f.f_code.co_name == '_svd_values' for f in _frames()):
            calls['n'] += 1
            if calls['n'] == failing_call:
                raise error('denied by policy')
        return threadpool_info()
    monkeypatch.setattr(tp, 'threadpool_info', info)
    model, grid, removed, prepared, case, box = small()
    before = _blas_threads()
    refused('COUPLED_UNCERTIFIED', solve, model, prepared, removed, box, case, reason='blas_threads')
    assert calls['n'] >= failing_call
    assert _blas_threads() == before
    test_r4_p1_reentry_outside_the_window_still_works()


def _frames():
    frame = sys._getframe(1)
    while frame is not None:
        yield frame
        frame = frame.f_back


_AUDIT_CHILD = textwrap.dedent('''
    import sys
    sys.path[:0] = {paths!r}
    from test_coupled_review_r1 import small, solve
    from anyfesolver._coupled_certificate import CoupledSolveError
    model, grid, removed, prepared, case, box = small()
    armed = [False]

    def hook(event, args):
        if armed[0] and event == 'ctypes.dlopen':
            raise PermissionError('explicit DLL handles denied by policy')
    sys.addaudithook(hook)
    armed[0] = True
    try:
        solve(model, prepared, removed, box, case)
        print('OUTCOME accepted')
    except CoupledSolveError as exc:
        print('OUTCOME', exc.code, exc.reason)
''')


@pytest.mark.skipif(sys.platform != 'win32', reason='threadpoolctl opens DLL handles through ctypes on Windows')
def test_r4_p2_audit_policy_denying_dll_handles_refuses_blas_threads():
    """The reviewer's reproduction in a child process: an audit hook denies ctypes.dlopen (threadpoolctl's
    explicit BLAS handles) once the fixture is built. The route refuses blas_threads."""
    paths = [p for p in sys.path if p]
    code = _AUDIT_CHILD.format(paths=paths)
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300,
                          env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'})
    lines = [line for line in done.stdout.splitlines() if line.startswith('OUTCOME')]
    assert lines == ['OUTCOME COUPLED_UNCERTIFIED blas_threads'], (done.stdout[-2000:], done.stderr[-2000:])


# --- P2: no truthiness (or other implicit protocol) call on the caller's clock --------------------------------

class _ProtocolClock:
    """A valid clock whose implicit protocol methods record (and switch the rounding mode, as in the
    reviewer's schedule) if they are ever called."""

    def __init__(self, stack):
        self.stack, self.protocol, self.calls = stack, [], 0

    def __call__(self):
        self.calls += 1
        return 0.0

    def _record(self, name):
        self.protocol.append(name)
        self.stack.enter_context(rounding(0x100))

    def __bool__(self):
        self._record('__bool__')
        return False

    def __len__(self):
        self._record('__len__')
        return 0

    def __eq__(self, other):
        self._record('__eq__')
        return False

    __hash__ = object.__hash__

    def __index__(self):
        self._record('__index__')
        return 0


def test_r4_p2_route_never_calls_the_clock_truthiness():
    stack = contextlib.ExitStack()
    clock = _ProtocolClock(stack)
    model, grid, removed, prepared, case, box = small()
    try:
        result = solve(model, prepared, removed, box, case, clock=clock)
    finally:
        stack.close()
    assert clock.protocol == []
    assert clock.calls > 0, 'the caller clock was not used'
    assert result.timings['total'] == 0.0


def test_r4_p2_certified_solve_never_calls_the_clock_truthiness():
    stack = contextlib.ExitStack()
    clock = _ProtocolClock(stack)
    S = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    try:
        solved = cc.certified_solve(S, np.array([1.0, 2.0]), raw_load_zero=False, clock=clock)
    finally:
        stack.close()
    assert clock.protocol == []
    assert clock.calls > 0
    assert set(solved.timings.values()) == {0.0}


@pytest.mark.parametrize('flag', [np.True_, 1, None], ids=['numpy_bool', 'int', 'none'])
def test_r4_p2_certified_solve_raw_load_zero_must_be_a_bool(flag):
    """Class search: certified_solve's raw_load_zero is decided by truthiness; only an exact bool is admitted."""
    S = sparse.csr_matrix(np.eye(2))
    with pytest.raises(cc.CoupledSolveError) as info:
        cc.certified_solve(S, np.zeros(2), raw_load_zero=flag)
    assert (info.value.code, info.value.reason) == ('COUPLED_UNCERTIFIED', 'invalid_arguments')


# --- P2: the oracle at exactly kappa = 1e7 ---------------------------------------------------------------

def test_r4_oracle_raises_inside_the_accuracy_band_at_the_cutoff():
    """S = [[10000001, 9999999], [9999999, 10000001]], b = S[:, 0]: both Ruiz shifts are equal, kappa = 1e7
    exactly, x = (1, 0) has zero residual. The computed kappa lies within the oracle's accuracy of the
    cutoff, so the oracle must raise (undecided), not return False."""
    from fractions import Fraction
    import test_coupled_r11_classes as r11
    S = sparse.csr_matrix(np.array([[10000001.0, 9999999.0], [9999999.0, 10000001.0]]))
    b = np.ascontiguousarray(S.toarray()[:, 0])
    shifts = cc._certify_solve_matrix(S)['shifts']
    assert shifts[0] == shifts[1]
    assert (Fraction(10000001) + 9999999) / (Fraction(10000001) - 9999999) == 10 ** 7
    with pytest.raises(AssertionError, match='accuracy band'):
        r11.acceptance_warranted(S, b)
