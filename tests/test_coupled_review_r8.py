"""Coupled route: regression tests for review round 8 (GPT 6.2 Astra), owner decision OD16 (2026-11-21).

In-process hooks and caller-defined built-in subclasses stay out of scope (OD15); these tests use genuine
inputs, legal environment state (an earlier ``sys.path`` entry, a slow file system, a suspended parent,
a thread-resource limit, a reader out of memory) or test-side seams around the real child. Production
code is never patched by these tests except through such test-side seams.

Sections: (1) the provenance of every module the child trusts; (2) the certificate-process state machine
of docs/SOLID_COUPLING.md, one test per transition; (3) the request wire and the export structure.
The fails-on-3601a0dcb evidence is in the commit message and the handover."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import os
import subprocess
import sys
import threading
import time
import types
from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

from anyfesolver import _coupled_certificate as cc  # noqa: E402
from anyfesolver import _coupled_certificate_math as cm  # noqa: E402
from anyfesolver import solid_coupling as sc  # noqa: E402
from _coupled_worker_patch import patch_worker  # noqa: E402
from test_coupled_review_r1 import refused  # noqa: E402
from test_coupled_review_r5 import _all_gone, _descendants  # noqa: E402


def _diag(*values):
    return sparse.csr_matrix(np.diag(np.asarray(values, dtype=float)))


def _map(indptr, columns, coefficients, offsets=None):
    return cc.ExportMap(np.asarray(indptr, dtype=np.int64), np.asarray(columns, dtype=np.int64), coefficients,
                        None if offsets is None else np.asarray(offsets, dtype=np.float64))


def _offset_map(value=7.0):
    return _map([0, 0], [], np.zeros(0), offsets=[value])


# === (1) provenance ===========================================================================================

def _fake_threadpoolctl(tmp_path, marker):
    """A same-name module in an earlier sys.path entry: it records that it ran and reports one BLAS
    thread for NumPy's own library whatever the pool really has (the reviewer's case)."""
    numpy_libs = os.path.join(os.path.dirname(np.__file__) + '.libs', 'openblas.dll')
    path = tmp_path / 'threadpoolctl.py'
    path.write_text(f"open({str(marker)!r}, 'a').write('fake\\n')\n"
                    f"def threadpool_info():\n"
                    f"    return [{{'user_api': 'blas', 'filepath': {numpy_libs!r}, 'num_threads': 1}}]\n")
    module = types.ModuleType('threadpoolctl')
    module.__file__ = str(path)
    return module


def test_r8_p1_threadpoolctl_from_an_earlier_path_entry_is_not_trusted(monkeypatch, tmp_path):
    """r8 P1: the parent imported a same-name threadpoolctl from an earlier sys.path entry before the
    solve; it hashed that module's own file and the child ran it. Now the child runs the installed
    distribution's file (RECORD in NumPy's directory), whatever sys.modules and sys.path hold."""
    marker = tmp_path / 'ran.txt'
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, 'threadpoolctl', _fake_threadpoolctl(tmp_path, marker))
    certificate = cc._certify_solve_matrix(_diag(4.0, 2.0, 1.0))
    assert certificate['accepted'] is True
    assert not marker.exists(), marker.read_text()


def _record_hash(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')


def _fake_site(tmp_path, *, numpy_version='9.9.9', numpy_init=b'# numpy\n', numpy_record_init=None,
               tp_source=b'def threadpool_info():\n    return []\n', tp_record=None, tp_dist=True):
    """A site directory with a numpy package and threadpoolctl.py, each with a dist-info RECORD."""
    site = tmp_path / 'site'
    (site / 'numpy').mkdir(parents=True)
    (site / 'numpy' / '__init__.py').write_bytes(numpy_init)
    info = site / f'numpy-{numpy_version}.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: numpy\nVersion: {numpy_version}\n')
    record_init = numpy_record_init if numpy_record_init is not None else _record_hash(numpy_init)
    (info / 'RECORD').write_text(f'numpy/__init__.py,sha256={record_init},{len(numpy_init)}\n'
                                 f'numpy-{numpy_version}.dist-info/RECORD,,\n')
    (site / 'threadpoolctl.py').write_bytes(tp_source)
    if tp_dist:
        tp_info = site / 'threadpoolctl-3.0.0.dist-info'
        tp_info.mkdir()
        (tp_info / 'METADATA').write_text('Metadata-Version: 2.1\nName: threadpoolctl\nVersion: 3.0.0\n')
        hashed = tp_record if tp_record is not None else _record_hash(tp_source)
        (tp_info / 'RECORD').write_text(f'threadpoolctl.py,sha256={hashed},{len(tp_source)}\n'
                                        'threadpoolctl-3.0.0.dist-info/RECORD,,\n')
    numpy_module = types.ModuleType('numpy')
    numpy_module.__file__ = str(site / 'numpy' / '__init__.py')
    numpy_module.__version__ = numpy_version
    return numpy_module


def test_r8_p1_runtime_comes_from_the_installed_records():
    """Control: the genuine environment resolves NumPy and threadpoolctl through their RECORDs, and the
    bytes the child will run are exactly the RECORD-checked file."""
    runtime = cc._trusted_runtime()
    import threadpoolctl
    site = os.path.dirname(os.path.dirname(np.__file__))
    assert os.path.normcase(runtime.threadpoolctl_path) == os.path.normcase(os.path.join(site, 'threadpoolctl.py'))
    with open(runtime.threadpoolctl_path, 'rb') as handle:
        assert handle.read() == runtime.threadpoolctl_source
    assert runtime.threadpoolctl_digest == hashlib.sha256(runtime.threadpoolctl_source).hexdigest()
    assert runtime.numpy_version == np.__version__ and threadpoolctl is not None


@pytest.mark.parametrize('case, reason', [
    ('control', None),
    ('tp_hash', 'threadpoolctl_mismatch'),
    ('tp_no_distribution', 'threadpoolctl_mismatch'),
    ('numpy_hash', 'numpy_mismatch'),
    ('numpy_version', 'numpy_mismatch'),
])
def test_r8_p1_runtime_refuses_anything_its_records_do_not_vouch_for(tmp_path, case, reason):
    kwargs = {'tp_hash': dict(tp_record=_record_hash(b'other')), 'tp_no_distribution': dict(tp_dist=False),
              'numpy_hash': dict(numpy_record_init=_record_hash(b'other'))}.get(case, {})
    numpy_module = _fake_site(tmp_path, **kwargs)
    if case == 'numpy_version':
        numpy_module.__version__ = '9.9.8'
    if reason is None:
        runtime = cc._trusted_runtime(numpy_module)
        assert runtime.threadpoolctl_source == b'def threadpool_info():\n    return []\n'
        return
    error = refused('COUPLED_UNCERTIFIED', cc._trusted_runtime, numpy_module)
    assert error.reason == reason, error.reason


def test_r8_p1_child_runs_exactly_the_bytes_sent(monkeypatch, tmp_path):
    """The child executes the bytes the parent read and checked, from the request (not a file it opens):
    a runtime whose threadpoolctl bytes also write a marker shows the marker; the file on disk is the
    genuine one."""
    marker = tmp_path / 'sent.txt'
    real = cc._trusted_runtime()
    # Appended: threadpoolctl 3.8 opens with ``from __future__``, which no statement may precede.
    source = real.threadpoolctl_source + f"\nopen({str(marker)!r}, 'a').write('sent\\n')\n".encode('ascii')
    runtime = dataclasses.replace(real, threadpoolctl_source=source,
                                  threadpoolctl_digest=hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(cc, '_trusted_runtime', lambda *args: runtime)
    assert cc._certify_solve_matrix(_diag(4.0, 2.0, 1.0))['accepted'] is True
    assert marker.read_text() == 'sent\n'


def test_r8_p1_provenance_refusal_starts_no_child(monkeypatch):
    calls = []
    monkeypatch.setattr(cc, '_trusted_runtime', lambda *args: (_ for _ in ()).throw(
        cc._refuse('COUPLED_UNCERTIFIED', 'threadpoolctl_mismatch', 'test')))
    original = subprocess.Popen.__init__

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(subprocess.Popen, '__init__', spy)
    refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(4.0), np.ones(1), raw_load_zero=False,
            reason='threadpoolctl_mismatch')
    assert calls == []


# === (2) the state machine (docs/SOLID_COUPLING.md, "Certificate process: states") ===========================

def _slow_provenance(monkeypatch, seconds):
    """A provenance read that blocks (a hung network mount): on 3601a0dcb the digest read in argv."""
    name = '_trusted_runtime' if hasattr(cc, '_trusted_runtime') else '_file_digest'
    original = getattr(cc, name)

    def slow(*args):
        time.sleep(seconds)
        return original(*args)
    monkeypatch.setattr(cc, name, slow)


def test_r8_state_prepare_to_timeout_when_the_provenance_read_blocks(monkeypatch):
    """PREPARE -> TIMEOUT: the deadline starts before any child-related I/O, the digest read included
    (r8 P2), and the read is waited for within it."""
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 1.5)
    _slow_provenance(monkeypatch, 5.0)
    started = time.monotonic()
    error = refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, _diag(4.0, 2.0, 1.0), reason='svd_worker')
    assert error.diagnostics['worker_failure'] == 'timeout', error.diagnostics
    assert time.monotonic() - started < 4.0


def test_r8_state_prepare_to_launch_failure(monkeypatch):
    """LAUNCH -> refusal: the interpreter cannot be started (start_failed); nothing is left running."""
    original = cc._svd_worker_command

    def missing(*args):
        return [os.path.join(os.path.dirname(sys.executable), 'no-such-python.exe')] + original(*args)[1:]
    monkeypatch.setattr(cc, '_svd_worker_command', missing)
    error = refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, _diag(4.0, 2.0, 1.0), reason='svd_worker')
    assert error.diagnostics['worker_failure'] == 'start_failed', error.diagnostics


def test_r8_state_partial_lane_start_tears_everything_down_at_once(monkeypatch):
    """START -> TEARDOWN: the stdout lane cannot start (a thread-resource limit) after the writer has sent
    the request and waits for the candidate (r8 P2). The refusal comes at once, the child is killed and
    reaped by its own handle, and the writer is released (it waited to the deadline on 3601a0dcb)."""
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 60.0)
    writers, processes = [], []
    original_lane = cc._Lane.__init__

    def lane(self, target, *args):
        name = getattr(target, '__name__', '')
        if name == '_read' and not getattr(lane, 'failed', False):
            lane.failed = True
            raise RuntimeError("can't start new thread")
        original_lane(self, target, *args)
        if name == '_write':
            writers.append(self)
    monkeypatch.setattr(cc._Lane, '__init__', lane)
    original_io = cc._WorkerIO.__init__

    def io(self, process, *args, **kwargs):
        processes.append(process)
        return original_io(self, process, *args, **kwargs)
    monkeypatch.setattr(cc._WorkerIO, '__init__', io)
    started = time.monotonic()
    refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(4.0, 2.0, 1.0), np.ones(3), raw_load_zero=False,
            reason='svd_worker')
    assert time.monotonic() - started < 15.0
    assert processes and processes[0].poll() is not None
    assert writers and writers[0].finished(5.0), 'the writer still waits for the candidate'


def test_r8_state_collect_after_a_suspension_keeps_a_timely_refusal(monkeypatch):
    """RUNNING -> COLLECT: the child refuses zero_row and exits in time; the parent is suspended past the
    deadline before it looks (here: in its first poll of the result). The exited child's reply decides
    (r8 P2: it was discarded as a timeout)."""
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 3.0)
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        if calls['n'] == 4:                     # the first poll of the result, after the candidate
            time.sleep(4.0)
    error = refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(0.0), np.ones(1), raw_load_zero=False,
                    cancel=cancel)
    assert error.reason == 'zero_row', (error.reason, error.diagnostics.get('worker_failure'))
    assert calls['n'] >= 4


def test_r8_state_collect_after_a_suspension_keeps_an_acceptance(monkeypatch):
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 3.0)
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        if calls['n'] == 4:
            time.sleep(4.0)
    result = cc.certified_solve(_diag(4.0, 2.0, 1.0), np.ones(3), raw_load_zero=False, cancel=cancel)
    assert result.x.tolist() == [0.25, 0.5, 1.0]


def test_r8_state_expired_with_a_reader_memory_error_is_too_large(monkeypatch):
    """RUNNING -> EXPIRED: the stdout reader runs out of memory just as the deadline passes, after the
    last lane check (r8, r7 item 5 still open). Lane failures are checked after the final wait, so the
    MemoryError is COUPLED_TOO_LARGE, never a timeout."""
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 3.0)
    patch_worker(monkeypatch, cc, 'import time\ntime.sleep(20)\n')       # still running at the deadline
    release = threading.Event()
    original_read = cc._WorkerIO._read

    def read(self, stream, sink, limit):
        if sink is self.stdout:
            release.wait(30.0)
            raise MemoryError('reader buffer')
        return original_read(self, stream, sink, limit)
    monkeypatch.setattr(cc._WorkerIO, '_read', read)
    runs = []
    original_run = cc._Run.__init__

    def run(self, *args, **kwargs):
        original_run(self, *args, **kwargs)
        runs.append(self)
    monkeypatch.setattr(cc._Run, '__init__', run)

    def inject(process):
        """Past the deadline, with the child still running: the reader fails now (once)."""
        if release.is_set() or not runs or runs[0].__dict__.get('process') is not process \
                or time.monotonic() < runs[0].deadline:
            return
        release.set()
        limit = time.monotonic() + 10.0
        while time.monotonic() < limit and not any(isinstance(e, MemoryError) for e in runs[0].io.errors):
            time.sleep(0.01)
    real_wait, real_poll = subprocess.Popen.wait, subprocess.Popen.poll

    def wait(self, timeout=None):
        try:
            return real_wait(self, timeout)
        except subprocess.TimeoutExpired:
            inject(self)
            raise

    def poll(self):
        code = real_poll(self)
        if code is None:
            inject(self)
        return code
    monkeypatch.setattr(subprocess.Popen, 'wait', wait)
    monkeypatch.setattr(subprocess.Popen, 'poll', poll)
    refused('COUPLED_TOO_LARGE', cc._certify_solve_matrix, _diag(4.0, 2.0, 1.0), reason='memory_error')
    assert release.is_set()


def test_r8_state_expired_without_a_lane_failure_is_a_timeout(monkeypatch):
    """RUNNING -> EXPIRED -> TIMEOUT (control): a child still running at the deadline is killed and the
    solve refuses timeout; no process remains."""
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 2.0)
    patch_worker(monkeypatch, cc, 'import time\ntime.sleep(30)\n')
    seen = []
    error = refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, _diag(4.0, 2.0, 1.0),
                    cancel=lambda: seen.extend(_descendants(os.getpid())), reason='svd_worker')
    assert error.diagnostics['worker_failure'] == 'timeout'
    assert seen and _all_gone(seen)


def test_r8_state_collect_with_a_memory_error_at_the_final_read_is_too_large(monkeypatch):
    """COLLECT: the reader fails after the child exited (its final read): COUPLED_TOO_LARGE (control)."""
    original_read = cc._WorkerIO._read

    def read(self, stream, sink, limit):
        original_read(self, stream, sink, limit)
        if sink is self.stdout:
            raise MemoryError('final read')
    monkeypatch.setattr(cc._WorkerIO, '_read', read)
    refused('COUPLED_TOO_LARGE', cc._certify_solve_matrix, _diag(4.0, 2.0, 1.0), reason='memory_error')


# === (3) wire and structure ===================================================================================

def _child(request: bytes, command=None) -> int:
    process = subprocess.run(command or cc._svd_worker_command(), input=request, capture_output=True,
                             env=cc._svd_worker_environment(), timeout=120)
    return process.returncode


def _request(system, b, *, mode='solve', raw_load_zero=False, memory_limit=None, candidate=None):
    chunks = cc._request(system, b, mode=mode, raw_load_zero=raw_load_zero, memory_limit=memory_limit, exports=())
    out = b''.join(bytes(c) for c in chunks)
    if candidate is not None:
        out += candidate
    return out


def _xv(*values):
    return b'XV' + np.asarray(values, dtype='<f8').tobytes()


_EARLY = {
    # name: (request builder, exit code without a trailing byte)
    'zero_row': (lambda: _request(_diag(0.0), np.ones(1), candidate=_xv(0.0)), 0),
    'memory_limit_solve': (lambda: _request(_diag(4.0), np.ones(1), memory_limit=1, candidate=_xv(0.25)), 0),
    'load_stored_as_zero': (lambda: _request(_diag(4.0), np.zeros(1), candidate=_xv(0.0)), 0),
    'memory_limit_certificate': (lambda: _request(_diag(4.0), None, mode='certificate', memory_limit=1), 0),
    'no_candidate': (lambda: _request(_diag(4.0), np.ones(1), candidate=b'XF'), 0),
    'candidate_not_finite': (lambda: _request(_diag(4.0), np.ones(1), candidate=_xv(np.inf)), 0),
}


@pytest.mark.parametrize('name', sorted(_EARLY))
def test_r8_wire_trailing_request_byte_refused_on_every_path(name):
    """r8 (r7 item 3 still open): with S = [0], b = [1], an XV candidate and one trailing byte the child
    returned refuse/zero_row (exit 0); the singular branch returned before reading the candidate. Every
    path now reads and checks the whole request before its verdict: exit 2 with the byte, the verdict
    without it."""
    build, expected = _EARLY[name]
    request = build()
    assert _child(request) == expected
    assert _child(request + b'X') == 2


def test_r8_wire_trailing_request_byte_refused_before_an_environment_verdict():
    """The environment verdict (here: a threadpoolctl digest the bytes do not match) also comes only
    after the whole request was read: exit 3 without the trailing byte, 2 with it."""
    command = cc._svd_worker_command()
    command[-1] = '0' * 64
    request = _request(_diag(4.0), None, mode='certificate')
    assert _child(request, command) == 3
    assert _child(request + b'X', command) == 2


_OVERFLOW = np.array([0, 2 ** 62, -2 ** 63, -2 ** 62, 0], dtype=np.int64)


def test_r8_wire_row_pointer_overflow_is_refused_everywhere():
    """r8 P2: np.diff of [0, 2^62, -2^63, -2^62, 0] is four positive int64 values (overflow), so the
    pointer passed the structure check and bounded_rows raised IndexError. Every check now decides by
    comparisons (no subtraction) and refuses with a code."""
    empty_i, empty_f = np.zeros(0, dtype=np.int64), np.zeros(0)
    assert cm.export_structure_reason(_OVERFLOW, empty_i, 1) is not None
    assert cm.canonical_csr_reason(4, _OVERFLOW, empty_i, empty_f) is not None
    with pytest.raises(cm.CertificateRefusal) as info:
        cm.bounded_rows(_OVERFLOW, empty_i, [], None, np.zeros(1), np.zeros(1, dtype=np.int64), Fraction(0))
    assert info.value.reason == 'invalid_export'
    error = refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(4.0), np.ones(1), raw_load_zero=False,
                    exports=[cc.ExportMap(_OVERFLOW, empty_i, np.zeros(0))])
    assert error.reason == 'invalid_arguments', (error.reason, error.diagnostics)
    matrix = types.SimpleNamespace(shape=(4, 4), indptr=_OVERFLOW, indices=empty_i, data=empty_f)
    error = refused('COUPLED_FORMATION_RANGE', sc._check_csr_structure, matrix, 'K')
    assert error.reason == 'csr_structure'


def test_r8_wire_row_pointer_control():
    """Control: valid pointers, including empty rows, pass unchanged."""
    indptr = np.array([0, 0, 2, 2, 3], dtype=np.int64)
    columns = np.array([0, 1, 1], dtype=np.int64)
    assert cm.export_structure_reason(indptr, columns, 2) is None
    assert cm.canonical_csr_reason(4, indptr, np.array([0, 1, 2]), np.ones(3)) is None


def test_r8_wire_map_count_limit_is_the_childs_and_checked_first(monkeypatch):
    """r8 P2: 65 maps passed the parent and the child exited 2 (svd_worker/exit). The parent now checks
    the child's own limit (one constant in _coupled_certificate_math) before starting the child."""
    assert cm.MAX_EXPORTS == 64
    result = cc.certified_solve(_diag(1.0), np.ones(1), raw_load_zero=False, exports=[_offset_map()] * 64)
    assert len(result.exports) == 64
    starts = []
    original = cc._start_worker

    def spy(*args, **kwargs):
        starts.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cc, '_start_worker', spy)
    error = refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(1.0), np.ones(1), raw_load_zero=False,
                    exports=[_offset_map()] * 65)
    assert error.reason == 'export_limit', (error.reason, error.diagnostics)
    assert starts == []


def test_r8_wire_entry_limits_are_shared_with_the_child(monkeypatch):
    """The per-map entry limit is the same constant: the parent refuses above it before the child."""
    monkeypatch.setattr(cm, 'MAX_EXPORT_ENTRIES', 3)
    error = refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(1.0), np.ones(1), raw_load_zero=False,
                    exports=[_map([0, 0, 0, 0, 0], [], np.zeros(0))])
    assert error.reason == 'export_limit'
