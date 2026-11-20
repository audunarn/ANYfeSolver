"""Coupled route: regression tests for review round 5 (GPT 6.2 Astra) under owner decision OD13: the
certificate SVD runs in a short-lived child process (docs/SOLID_COUPLING.md, "Certificate SVD process").
Each test names its finding; the fails-on-a884ef8f8 evidence is in the commit message."""

from __future__ import annotations

import contextlib
import ctypes
import os
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

from anyfesolver import BoundaryCondition  # noqa: E402
from anyfesolver import _coupled_certificate as cc  # noqa: E402
from anyfesolver import solid_coupling as sc  # noqa: E402
from anyfesolver.boundary import LoadCase  # noqa: E402
from test_coupled_review_r1 import flushing, refused, small, solve  # noqa: E402
from test_coupled_review_r2 import rounding  # noqa: E402


def _blas_threads():
    from threadpoolctl import threadpool_info
    return [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']


def _matrix(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return np.ascontiguousarray(a + a.T)


# --- Process helpers (Windows: Toolhelp snapshot; the PIDs are this test process's own children) ----------

def _children(pid):
    """PIDs whose parent is ``pid`` (Windows Toolhelp snapshot)."""
    if sys.platform != 'win32':
        pytest.skip('process inspection uses the Windows Toolhelp API')
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD), ('th32ProcessID', wintypes.DWORD),
                    ('th32DefaultHeapID', ctypes.c_void_p), ('th32ModuleID', wintypes.DWORD),
                    ('cntThreads', wintypes.DWORD), ('th32ParentProcessID', wintypes.DWORD),
                    ('pcPriClassBase', ctypes.c_long), ('dwFlags', wintypes.DWORD), ('szExeFile', ctypes.c_char * 260)]
    kernel = ctypes.windll.kernel32
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    entry = Entry()
    entry.dwSize = ctypes.sizeof(Entry)
    found = []
    ok = kernel.Process32First(snapshot, ctypes.byref(entry))
    while ok:
        if entry.th32ParentProcessID == pid:
            found.append(int(entry.th32ProcessID))
        ok = kernel.Process32Next(snapshot, ctypes.byref(entry))
    kernel.CloseHandle(snapshot)
    return found


def _descendants(pid):
    out, todo = [], [pid]
    while todo:
        for child in _children(todo.pop()):
            out.append(child)
            todo.append(child)
    return out


def _alive(pid):
    from ctypes import wintypes
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = wintypes.DWORD()
    kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return code.value == 259                                   # STILL_ACTIVE


def _all_gone(pids, wait=10.0):
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.1)
    return False


@contextlib.contextmanager
def _capture_reports(monkeypatch):
    """Record the worker's diagnostic report of every certificate SVD (the production function is
    wrapped, not replaced)."""
    reports = []
    original = getattr(cc, '_svd_in_worker', None)
    if original is not None:
        def wrapped(*args, **kwargs):
            sigma, failure, report = original(*args, **kwargs)
            reports.append(report)
            return sigma, failure, report
        monkeypatch.setattr(cc, '_svd_in_worker', wrapped)
    yield reports


def _one_thread(report):
    return report is not None and report.get('threads_before') == [1] and report.get('threads_after') == [1] \
        and report.get('fp_before') is None and report.get('fp_after') is None


# --- r5 #1 (P1): parent hooks and thread scopes cannot reach the certificate SVD -----------------------------

def _in_numpy_svd_from_certificate(frame):
    return (frame.f_code.co_name == 'svd' and 'linalg' in frame.f_code.co_filename
            and frame.f_back is not None and frame.f_back.f_code.co_name == '_svd_values')


def test_r5_1_profile_hook_four_threads_at_the_svd_does_not_reach_it(monkeypatch):
    """The reviewer's schedule, reduced: a same-thread profile hook sets both BLAS pools to four threads
    (threadpoolctl directly) when the certificate's SVD is entered (NumPy's svd called from the
    certificate in process, or the worker run) and restores them when it returns. Any SVD the parent
    runs there must see one thread, and the worker must report one thread."""
    from threadpoolctl import threadpool_limits
    stack = contextlib.ExitStack()
    parent_svd_threads = []

    def at_svd(frame):
        return frame.f_code.co_name == '_svd_in_worker' or _in_numpy_svd_from_certificate(frame)

    def profile(frame, event, arg):
        if event == 'call' and at_svd(frame):
            stack.enter_context(threadpool_limits(limits=4, user_api='blas'))
            if frame.f_code.co_name == 'svd':
                parent_svd_threads.append(_blas_threads())
        elif event == 'return' and at_svd(frame):
            stack.close()
    model, grid, removed, prepared, case, box = small()
    reference = solve(model, prepared, removed, box, case)
    before = _blas_threads()
    with _capture_reports(monkeypatch) as reports:
        sys.setprofile(profile)
        try:
            result = solve(model, prepared, removed, box, case)
        finally:
            sys.setprofile(None)
            stack.close()
    assert all(set(t) == {1} for t in parent_svd_threads), parent_svd_threads
    assert reports and all(_one_thread(r) for r in reports), reports
    assert _blas_threads() == before
    assert np.array_equal(result.shell_displacements.values, reference.shell_displacements.values)


def test_r5_1_parent_four_thread_scopes_do_not_reach_the_worker(monkeypatch):
    """A four-thread limit in the parent (threadpoolctl on this thread, native_thread_scope(4) held on
    another thread for the whole solve) leaves the worker's verified environment and its result unchanged."""
    from threadpoolctl import threadpool_limits
    from anyfesolver.threading_policy import native_thread_scope
    A = _matrix(200)
    baseline, failure, base_report = cc._svd_in_worker(A)
    assert failure is None and _one_thread(base_report)
    with threadpool_limits(limits=4, user_api='blas'):
        assert set(_blas_threads()) == {4}
        sigma, failure, report = cc._svd_in_worker(A)
    assert failure is None and _one_thread(report)
    assert np.array_equal(sigma.view(np.int64), baseline.view(np.int64))
    model, grid, removed, prepared, case, box = small()
    reference = solve(model, prepared, removed, box, case)
    holding, release = threading.Event(), threading.Event()

    def other():
        with native_thread_scope(4, phase='r5-other-thread'):
            holding.set()
            release.wait(120)
    worker = threading.Thread(target=other, daemon=True)
    worker.start()
    assert holding.wait(30)
    try:
        with _capture_reports(monkeypatch) as reports:
            with threadpool_limits(limits=4, user_api='blas'):
                result = solve(model, prepared, removed, box, case)
    finally:
        release.set()
        worker.join(30)
    assert reports and all(_one_thread(r) for r in reports), reports
    assert np.array_equal(result.shell_displacements.values, reference.shell_displacements.values)


@pytest.mark.parametrize('mode', ['down', 'flush'])
def test_r5_1_parent_fp_environment_does_not_reach_the_worker(mode):
    """Directed rounding or FTZ/DAZ in the parent thread does not reach the worker: it verifies its own
    environment and returns bit-identical singular values."""
    A = _matrix(200, seed=1)
    baseline, failure, base_report = cc._svd_in_worker(A)
    assert failure is None
    with (rounding(0x100) if mode == 'down' else flushing()):
        sigma, failure, report = cc._svd_in_worker(A)
    assert failure is None and _one_thread(report), report
    def verified(r):                        # the report without its timings and memory figures
        return {k: v for k, v in r.items() if not k.startswith(('seconds_', 'peak_'))}
    assert verified(report) == verified(base_report)
    assert np.array_equal(sigma.view(np.int64), baseline.view(np.int64))


def test_r5_1_parent_reader_joining_a_child_solve_proceeds():
    """Formerly r4 P2: a parent holding a default native scope and joining a solve on another thread. The
    certificate takes no coordinator scope any more, so the solve finishes (hard timeout 60 s)."""
    from anyfesolver.threading_policy import native_thread_scope
    model, grid, removed, prepared, case, box = small()
    outcome = {}

    def child():
        try:
            outcome['result'] = solve(model, prepared, removed, box, case)
        except BaseException as exc:  # noqa: BLE001
            outcome['error'] = exc
    thread = threading.Thread(target=child, daemon=True)
    with native_thread_scope(None, phase='r5-parent'):
        thread.start()
        thread.join(60)
        hung = thread.is_alive()
    thread.join(180)
    assert not hung, 'the child solve waited for the parent reader'
    assert 'result' in outcome, outcome


# --- OD13: the worker verifies its own environment and starts isolated --------------------------------------

def test_od13_worker_refuses_more_than_one_blas_thread(monkeypatch):
    """The worker's own check: with OPENBLAS_NUM_THREADS=4 in its environment it refuses blas_threads."""
    original = cc._svd_worker_environment
    monkeypatch.setattr(cc, '_svd_worker_environment', lambda: dict(original(), OPENBLAS_NUM_THREADS='4'))
    S = sparse.csr_matrix(np.diag([4.0, 2.0, 1.0]))
    error = refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, S, reason='blas_threads')
    report = error.diagnostics.get('worker_report')
    assert report is not None and report.get('threads_start') == [4], report


def test_od13_worker_refuses_without_threadpoolctl(monkeypatch, tmp_path):
    """threadpoolctl not importable in the worker: the thread count cannot be verified, so it refuses."""
    (tmp_path / 'threadpoolctl.py').write_text("raise ImportError('blocked for the test')\n")
    original = cc._svd_worker_command
    monkeypatch.setattr(cc, '_svd_worker_command', lambda: original()[:5] + [str(tmp_path)] + original()[5:])
    S = sparse.csr_matrix(np.diag([4.0, 2.0, 1.0]))
    refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, S, reason='blas_threads')


def test_od13_worker_refuses_another_numpy(monkeypatch, tmp_path):
    original = cc._svd_worker_command
    monkeypatch.setattr(cc, '_svd_worker_command', lambda: original()[:4] + [str(tmp_path)] + original()[5:])
    S = sparse.csr_matrix(np.diag([4.0, 2.0, 1.0]))
    error = refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, S, reason='svd_worker')
    assert error.diagnostics['worker_failure'] == 'environment'


def test_od13_worker_loads_no_site_customisation_or_python_variables(monkeypatch, tmp_path):
    """sitecustomize, usercustomize, a .pth file and PYTHONSTARTUP on the parent's PYTHONPATH and in its
    environment never run in the worker; it starts isolated with the minimal environment."""
    marker = tmp_path / 'ran.txt'
    payload = f"open({str(marker)!r}, 'a').write(__name__ + '\\n')\n"
    for name in ('sitecustomize.py', 'usercustomize.py', 'startup.py'):
        (tmp_path / name).write_text(payload)
    (tmp_path / 'evil.pth').write_text(f"import os; open({str(marker)!r}, 'a').write('pth\\n')\n")
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    monkeypatch.setenv('PYTHONSTARTUP', str(tmp_path / 'startup.py'))
    monkeypatch.setenv('PYTHONUSERBASE', str(tmp_path))
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '4')
    sigma, failure, report = cc._svd_in_worker(_matrix(50))
    assert failure is None and _one_thread(report)
    assert not marker.exists(), marker.read_text()
    assert report['isolated'] == 1 and report['no_site'] == 1 and report['site_loaded'] is False
    assert report['sitecustomize_loaded'] is False
    allowed = {'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'SYSTEMROOT', '__PYVENV_LAUNCHER__'}
    assert set(report['environment_keys']) <= allowed, report['environment_keys']
    command = cc._svd_worker_command()
    assert command[0] == sys.executable and command[1:3] == ['-I', '-S']


# --- OD13: cancellation terminates this solve's own worker ------------------------------------------------------

def test_od13_cancel_during_the_worker_terminates_it_and_leaves_no_process():
    """The route's cancellation flag turns True once the worker exists (it is polled while the worker
    runs). The solve refuses COUPLED_CANCELLED; the worker and the interpreter it started are gone."""
    model, grid, removed, prepared, case, box = small()
    seen = []

    def cancel():
        children = _descendants(os.getpid())
        if children:
            seen.extend(children)
            return True
        return False
    refused('COUPLED_CANCELLED', solve, model, prepared, removed, box, case, cancel=cancel)
    assert seen, 'the cancellation was never polled while the worker ran'
    assert _all_gone(seen), [p for p in seen if _alive(p)]


def test_od13_cancel_mid_svd_is_prompt():
    """A cancellation 1 s into a 2000 x 2000 worker SVD refuses within a few seconds; no process remains."""
    A = _matrix(2000, seed=2)
    seen, started = [], time.monotonic()

    def cancel():
        if time.monotonic() - started > 1.0:
            seen.extend(_descendants(os.getpid()))
            raise cc._refuse('COUPLED_CANCELLED', None, 'test cancellation')
    with pytest.raises(cc.CoupledSolveError) as info:
        cc._svd_in_worker(A, cancel)
    elapsed = time.monotonic() - started
    assert info.value.code == 'COUPLED_CANCELLED'
    assert seen and _all_gone(seen), seen
    assert elapsed < 10.0, elapsed


def test_od13_worker_timeout_refuses_and_terminates(monkeypatch):
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 1.0)
    monkeypatch.setattr(cc, '_svd_worker_command',
                        lambda: [sys.executable, '-I', '-S', '-c', 'import time; time.sleep(60)'])
    seen = []
    original_poll = cc._poll

    def cancel():
        seen.extend(_descendants(os.getpid()))
    S = sparse.csr_matrix(np.diag([4.0, 2.0, 1.0]))
    error = refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, S, cancel=cancel, reason='svd_worker')
    assert error.diagnostics['worker_failure'] == 'timeout'
    assert seen and _all_gone(seen)
    assert cc._poll is original_poll


# --- OD13: a crash or bad output in the worker refuses -----------------------------------------------------------

_SCRIPT = textwrap.dedent('''
    import struct, sys
    out = sys.stdout.buffer
    def ok(values, n=3, tail=b''):
        out.write(b'OK' + struct.pack('<Q', n) + struct.pack('<%dd' % len(values), *values) + tail)
        out.flush()
    {body}
''')

_CASES = {
    'exit7': ('sys.exit(7)', ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'crash': ("raise RuntimeError('worker crash')", ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'silent_ok_exit': ('sys.exit(0)', ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'short': ('ok([4.0, 2.0])', ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'long': ("ok([4.0, 2.0, 1.0], tail=b'x')", ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'wrong_n': ('ok([4.0, 2.0, 1.0], n=4)', ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'nan': ("ok([4.0, float('nan'), 1.0])", ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'inf': ("ok([float('inf'), 2.0, 1.0])", ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'negative': ('ok([4.0, 2.0, -1.0])', ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'ascending': ('ok([1.0, 2.0, 4.0])', ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'ok_nonzero_exit': ('ok([4.0, 2.0, 1.0]); sys.exit(1)', ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'garbage': ("out.write(b'\\x00' * 34); out.flush()", ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'unknown_environment': ("out.write(b'ENsomething'); out.flush(); sys.exit(3)", ('COUPLED_UNCERTIFIED', 'svd_worker')),
    'memory': ("out.write(b'MM'); out.flush(); sys.exit(4)", ('COUPLED_TOO_LARGE', 'memory_error')),
    'killed': ('import os; os._exit(9)', ('COUPLED_UNCERTIFIED', 'svd_worker')),
}


@pytest.mark.parametrize('case', sorted(_CASES))
def test_od13_bad_worker_output_refuses(monkeypatch, case):
    body, (code, reason) = _CASES[case]
    script = _SCRIPT.format(body=body)
    monkeypatch.setattr(cc, '_svd_worker_command', lambda: [sys.executable, '-I', '-S', '-c', script])
    S = sparse.csr_matrix(np.diag([4.0, 2.0, 1.0]))
    refused(code, cc._certify_solve_matrix, S, reason=reason)


def test_od13_worker_linalg_failure_is_a_numerical_failure(monkeypatch):
    script = _SCRIPT.format(body="out.write(b'LA'); out.flush(); sys.exit(5)")
    monkeypatch.setattr(cc, '_svd_worker_command', lambda: [sys.executable, '-I', '-S', '-c', script])
    certificate = cc._certify_solve_matrix(sparse.csr_matrix(np.diag([4.0, 2.0, 1.0])))
    assert certificate['accepted'] is False and certificate['reason'] == 'numerical_failure:LinAlgError'


def test_od13_valid_worker_record_is_accepted(monkeypatch):
    """Control for the cases above: a well-formed record from the same harness is used as is."""
    script = _SCRIPT.format(body='ok([1.0, 1.0, 1.0])')
    monkeypatch.setattr(cc, '_svd_worker_command', lambda: [sys.executable, '-I', '-S', '-c', script])
    certificate = cc._certify_solve_matrix(sparse.csr_matrix(np.eye(3)))
    assert certificate['accepted'] is True and certificate['sigma_max'] == 1.0 == certificate['sigma_min']


def test_od13_worker_matches_in_process_svd_bits():
    """The worker's singular values are bit-identical to NumPy's in-process one-thread SVD here."""
    from threadpoolctl import threadpool_limits
    A = _matrix(300, seed=3)
    sigma, failure, _ = cc._svd_in_worker(A)
    with threadpool_limits(limits=1, user_api='blas'):
        local = np.linalg.svd(A, compute_uv=False)
    assert failure is None and np.array_equal(sigma.view(np.int64), local.view(np.int64))


_AUDIT_CHILD = textwrap.dedent('''
    import sys
    sys.path[:0] = {paths!r}
    from test_coupled_review_r1 import small, solve
    from anyfesolver._coupled_certificate import CoupledSolveError
    model, grid, removed, prepared, case, box = small()

    def hook(event, args):
        if event == 'subprocess.Popen':
            raise PermissionError('process creation denied by policy')
    sys.addaudithook(hook)
    try:
        solve(model, prepared, removed, box, case)
        print('OUTCOME accepted')
    except CoupledSolveError as exc:
        print('OUTCOME', exc.code, exc.reason, exc.diagnostics.get('worker_failure'))
''')


def test_od13_audit_policy_denying_process_creation_refuses():
    """In a separate interpreter (an audit hook cannot be removed): an audit policy that denies
    subprocess.Popen refuses svd_worker (start_failed)."""
    paths = [p for p in sys.path if p]
    done = subprocess.run([sys.executable, '-c', _AUDIT_CHILD.format(paths=paths)], capture_output=True, text=True,
                          timeout=300, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'})
    lines = [line for line in done.stdout.splitlines() if line.startswith('OUTCOME')]
    assert lines == ['OUTCOME COUPLED_UNCERTIFIED svd_worker start_failed'], (done.stdout[-2000:], done.stderr[-2000:])


# --- r5 #4: caller containers are checked by exact type before any protocol call ---------------------------------

_PROTOCOL = ('__len__', '__bool__', '__iter__', '__contains__', '__getitem__', 'items', 'keys', 'values', 'get',
             '__eq__', '__ne__', '__hash__', '__reversed__', 'pop', 'copy', '__deepcopy__', '__reduce_ex__')


def _recording(base, log):
    def make(name):
        def method(self, *args, **kwargs):
            if log.get('armed'):
                log.setdefault('calls', []).append(name)
            return getattr(base, name)(self, *args, **kwargs)
        return method
    namespace = {name: make(name) for name in _PROTOCOL if hasattr(base, name)}
    if base is dict or base is list:
        namespace['__hash__'] = None
    return type('Recording' + base.__name__.title(), (base,), namespace)


class _Flag:
    def __init__(self, log):
        self.log = log

    def __bool__(self):
        if self.log.get('armed'):
            self.log.setdefault('calls', []).append('__bool__')
        return False


def _case_with(field, log):
    model, grid, removed, prepared, case, box = small()
    RecDict = _recording(dict, log)
    if field == 'follower_pressure':
        new = LoadCase(name='r5', pressure_loads=dict(case.pressure_loads), follower_pressure=_Flag(log))
    else:
        kwargs = {'pressure_loads': dict(case.pressure_loads)}
        kwargs[field] = RecDict(kwargs.get(field, {}))
        new = LoadCase(name='r5', **kwargs)
    return model, removed, prepared, new, box


@pytest.mark.parametrize('field', ['element_loads', 'nodal_loads', 'pressure_loads', 'added_node_masses',
                                   'follower_pressure'])
def test_r5_4_load_case_containers_are_exact_types(field):
    """The reviewer's construction: an untouched LoadCase built with a dict subclass (here every one of
    its containers in turn, and a non-bool follower flag). The route refuses container_type before any
    of the caller's protocol methods runs."""
    log = {}
    model, removed, prepared, case, box = _case_with(field, log)
    log['armed'] = True
    error = refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case)
    log['armed'] = False
    assert log.get('calls', []) == [], log
    assert error.reason == 'container_type', error.reason


def _model_with(target, log):
    RecDict, RecList = _recording(dict, log), _recording(list, log)

    def hook(model, grid):
        if target == 'support_node_ids':
            model.add_boundary_condition(BoundaryCondition(name='r5', node_ids=RecList([grid[0, 1]]),
                                                           dof_constraints={'ux': 0.0}))
        elif target == 'support_dofs':
            model.add_boundary_condition(BoundaryCondition(name='r5', node_ids=[grid[0, 1]],
                                                           dof_constraints=RecDict({'ux': 0.0})))
    model, grid, removed, prepared, case, box = small(hook=hook)
    if target == 'boundary_conditions':
        model.boundary_conditions = RecList(model.boundary_conditions)
    elif target == 'constraint_equations':
        model.constraint_equations = RecList(model.constraint_equations)
    elif target == 'materials':
        model.materials = RecDict(model.materials)
    elif target == 'point_masses':
        model.mesh.point_masses = RecDict(model.mesh.point_masses)
    elif target == 'node_dofs':
        node = next(iter(model.mesh.nodes.values()))
        node.dofs = RecList(node.dofs)
    return model, removed, prepared, case, box


@pytest.mark.parametrize('target', ['support_node_ids', 'support_dofs', 'boundary_conditions', 'constraint_equations',
                                    'materials', 'point_masses', 'node_dofs'])
def test_r5_4_model_containers_are_exact_types(target):
    """Class search over the FEModel containers the route reads: each subclass refuses container_type
    and none of its protocol methods runs inside the solve."""
    log = {}
    model, removed, prepared, case, box = _model_with(target, log)
    log['armed'] = True
    error = refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case)
    log['armed'] = False
    assert log.get('calls', []) == [], log
    assert error.reason == 'container_type', error.reason


def test_r5_4_exact_containers_are_admitted():
    """Control: the same values in exact dicts (and integer numpy keys, as the fixture grid gives) are accepted."""
    model, grid, removed, prepared, case, box = small()
    copy_case = LoadCase(name='r5', pressure_loads=dict(case.pressure_loads), nodal_loads={np.int64(grid[1, 1]): [0.0, 0.0, 0.0]})
    result = solve(model, prepared, removed, box, copy_case)
    reference = solve(model, prepared, removed, box, case)
    assert np.array_equal(result.shell_displacements.values, reference.shell_displacements.values)


# --- r5 #3: hooks raising inside the FP probe or the translation end in a coded refusal --------------------------

@contextlib.contextmanager
def _raise_once_at(name, make, after=0):
    """A profile hook that raises make() once, at the call of the function ``name`` (after ``after``
    earlier calls). The interpreter unsets a profile function that raises."""
    state = {'seen': 0, 'fired': False}

    def profile(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == name and not state['fired']:
            state['seen'] += 1
            if state['seen'] > after:
                state['fired'] = True
                raise make()
    sys.setprofile(profile)
    try:
        yield state
    finally:
        sys.setprofile(None)


@pytest.mark.parametrize('probe', ['_nearest_rounding', '_subnormals_preserved'])
@pytest.mark.parametrize('error', ['permission', 'forged_refusal'])
def test_r5_3_hook_raising_inside_the_fp_probe_refuses_fp_probe(probe, error):
    def make():
        if error == 'permission':
            return PermissionError('denied by policy')
        return cc.CoupledSolveError('forged', code='COUPLED_CANCELLED', reason=None)
    model, grid, removed, prepared, case, box = small()
    with _raise_once_at(probe, make, after=2) as state:
        caught = refused('COUPLED_UNCERTIFIED', solve, model, prepared, removed, box, case)
    assert state['fired']
    assert caught.reason == 'fp_probe', (caught.code, caught.reason)


def test_r5_3_hook_raising_while_a_native_refusal_is_built(monkeypatch):
    def broken(*args):
        raise ValueError('native refusal')
    monkeypatch.setattr(sc, '_holed_model', broken)
    model, grid, removed, prepared, case, box = small()
    with _raise_once_at('_type_name', lambda: PermissionError('denied by policy')) as state:
        refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case, reason='ValueError')
    assert state['fired']


def test_r5_3_hook_raising_in_the_top_level_translation():
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        raise ValueError('callback failure')
    model, grid, removed, prepared, case, box = small()
    with _raise_once_at('_type_name', lambda: PermissionError('denied by policy')) as state:
        refused('COUPLED_NATIVE_INVALID', solve, model, prepared, removed, box, case, cancel=cancel,
                reason='unexpected:ValueError')
    assert state['fired'] and calls['n'] == 1


def test_r5_3_hook_raising_while_the_candidate_refusal_is_built(monkeypatch):
    def singular(*args, **kwargs):
        raise RuntimeError('Factor is exactly singular')
    monkeypatch.setattr(cc.sparse_linalg, 'splu', singular)
    S = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    with _raise_once_at('_type_name', lambda: PermissionError('denied by policy')) as state:
        with pytest.raises(cc.CoupledSolveError) as info:
            cc.certified_solve(S, np.array([1.0, 2.0]), raw_load_zero=False)
    assert state['fired']
    assert (info.value.code, info.value.reason) == ('COUPLED_UNCERTIFIED', 'candidate_failed')


# --- r5 #5: the oracle uses a private mpmath context ------------------------------------------------------------

def test_r5_5_oracle_is_independent_of_the_global_mpmath_precision():
    """The reviewer's schedule: another thread enters mpmath.workdps(3) while the oracle computes. Here a
    trace function sets the shared context to 3 digits on every line of the oracle's functions (the
    effect of that thread). At kappa = 1e7 exactly the oracle must still raise (undecided)."""
    import mpmath
    import test_coupled_r11_classes as r11
    S = sparse.csr_matrix(np.array([[10000001.0, 9999999.0], [9999999.0, 10000001.0]]))
    b = np.ascontiguousarray(S.toarray()[:, 0])
    names = {'kappa_within_cutoff', 'independent_condition', 'triplet', 'reference_solution'}
    saved = mpmath.mp.dps

    def local(frame, event, arg):
        if event == 'line':
            mpmath.mp.dps = 3
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code.co_name in names else None
    sys.settrace(tracer)
    try:
        with pytest.raises(AssertionError, match='accuracy band'):
            r11.acceptance_warranted(S, b)
    finally:
        sys.settrace(None)
        mpmath.mp.dps = saved
