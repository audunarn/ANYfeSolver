"""Coupled route: regression tests for the non-hook findings of review round 7 (GPT 6.2 Astra). In-process
hooks and caller-defined built-in subclasses are out of scope (owner decision OD15, 2026-11-21); these
tests use genuine inputs, legal environment state, or test-side seams around the real child. Each test
names its finding; the fails-on-6f7e67ce2 evidence is in the commit message and the handover."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import time
import types
from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('skfem')
pytest.importorskip('anysolid')

from anyfesolver import _coupled_certificate as cc  # noqa: E402
from _coupled_worker_patch import patch_worker  # noqa: E402
from test_coupled_review_r1 import refused  # noqa: E402

_REPLY_MAGIC = b'ANYCRR02'


def _diag(*values):
    return sparse.csr_matrix(np.diag(np.asarray(values, dtype=float)))


def _map(indptr, columns, coefficients, offsets=None):
    return cc.ExportMap(np.asarray(indptr, dtype=np.int64), np.asarray(columns, dtype=np.int64), coefficients,
                        None if offsets is None else np.asarray(offsets, dtype=np.float64))


# --- P1: a repeated column in a recovery-map row -----------------------------------------------------------------

def test_r7_p1_repeated_column_is_refused_not_bounded():
    """Codex's case: S = [3], b = [1], one row with two unit coefficients on column 0. The exact value is
    2/3; the bound sqrt(1 + 1) E treated them as two coordinates (the row norm is 2) and fell below the
    actual error. The map is refused."""
    export = _map([0, 2], [0, 0], np.ones(2))
    error = refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(3.0), np.ones(1), raw_load_zero=False,
                    exports=[export])
    assert error.reason == 'invalid_arguments', error.reason


@pytest.mark.parametrize('kind', ['float', 'rational'])
def test_r7_p1_child_refuses_a_repeated_column_on_its_own(monkeypatch, kind):
    """The child decides on its own copy: with the parent's structure check switched off in this test
    process, the child still refuses the map (protocol error, no reply), never bounds it."""
    monkeypatch.setattr(cc._m, 'export_structure_reason', lambda *args: None, raising=False)
    coefficients = np.ones(2) if kind == 'float' else (Fraction(1), Fraction(1))
    error = refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(3.0), np.ones(1), raw_load_zero=False,
                    exports=[_map([0, 2], [0, 0], coefficients)])
    assert error.reason == 'svd_worker' and error.diagnostics['worker_failure'] == 'exit', error.diagnostics


def test_r7_p1_distinct_unsorted_columns_are_bounded():
    """Control: distinct columns in any order are a valid row (the bound does not assume an order)."""
    result = cc.certified_solve(_diag(4.0, 2.0), np.array([1.0, 1.0]), raw_load_zero=False,
                                exports=[_map([0, 2], [1, 0], np.ones(2))])
    (values, bounds), = result.exports
    assert values[0] == 0.75 and bounds[0] >= 0.0


# --- P1: threadpoolctl and NumPy identity --------------------------------------------------------------------

def _package_next_to_genuine_file(tmp_path, marker):
    """A directory holding a genuine copy of threadpoolctl.py and a same-name package that reports one
    BLAS thread for NumPy's own library and records that it ran."""
    import threadpoolctl
    shutil.copy(threadpoolctl.__file__, tmp_path / 'threadpoolctl.py')
    package = tmp_path / 'threadpoolctl'
    package.mkdir()
    numpy_libs = os.path.join(os.path.dirname(np.__file__) + '.libs', 'openblas.dll')
    (package / '__init__.py').write_text(
        f"open({str(marker)!r}, 'a').write('package\\n')\n"
        f"def threadpool_info():\n"
        f"    return [{{'user_api': 'blas', 'filepath': {numpy_libs!r}, 'num_threads': 1}}]\n")
    module = types.ModuleType('threadpoolctl')
    module.__file__ = str(tmp_path / 'threadpoolctl.py')
    return module


def test_r7_p1_same_name_package_next_to_threadpoolctl_is_not_loaded(monkeypatch, tmp_path):
    marker = tmp_path / 'ran.txt'
    monkeypatch.setitem(sys.modules, 'threadpoolctl', _package_next_to_genuine_file(tmp_path, marker))
    certificate = cc._certify_solve_matrix(_diag(4.0, 2.0, 1.0))
    assert certificate['accepted'] is True
    assert not marker.exists(), marker.read_text()


def test_r7_p1_threadpoolctl_with_another_digest_refuses(monkeypatch):
    command = cc._svd_worker_command()
    assert len(command) == 8, command
    command[7] = '0' * 64
    monkeypatch.setattr(cc, '_svd_worker_command', lambda: command)
    error = refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, _diag(4.0, 2.0, 1.0), reason='svd_worker')
    assert error.diagnostics['worker_failure'] == 'environment', error.diagnostics


def test_r7_p1_numpy_with_another_version_refuses(monkeypatch):
    command = cc._svd_worker_command()
    assert len(command) == 8, command
    command[6] = '0.0.0'
    monkeypatch.setattr(cc, '_svd_worker_command', lambda: command)
    error = refused('COUPLED_UNCERTIFIED', cc._certify_solve_matrix, _diag(4.0, 2.0, 1.0), reason='svd_worker')
    assert error.diagnostics['worker_failure'] == 'environment', error.diagnostics


# --- P2: the reply schema ------------------------------------------------------------------------------------

def _captured(monkeypatch, call):
    """Run ``call`` with the real child and return the (code, output, size, mode, export_rows) it decoded."""
    seen = []
    original = cc._decode

    def capture(code, output, size, mode, export_rows):
        seen.append((code, bytes(output), size, mode, list(export_rows)))
        return original(code, output, size, mode, export_rows)
    monkeypatch.setattr(cc, '_decode', capture)
    call()
    monkeypatch.setattr(cc, '_decode', original)
    assert len(seen) == 1
    return seen[0]


def _split(output):
    length = struct.unpack('<Q', output[8:16])[0]
    return json.loads(output[16:16 + length].decode('ascii')), output[16 + length:]


def _join(reply, payload, text=None):
    body = (text if text is not None else json.dumps(reply, sort_keys=True, separators=(',', ':'))).encode('ascii')
    return _REPLY_MAGIC + struct.pack('<Q', len(body)) + body + payload


def _accepted(monkeypatch):
    return _captured(monkeypatch, lambda: cc.certified_solve(_diag(4.0, 2.0, 1.0), np.ones(3), raw_load_zero=False,
                                                              exports=[_map([0, 1], [0], np.ones(1))]))


def _malformed(code, output, size, mode, rows):
    with pytest.raises(cc._Malformed):
        cc._decode(code, output, size, mode, rows)


def test_r7_p2_shift_minimum_int64_refuses(monkeypatch):
    code, output, size, mode, rows = _captured(monkeypatch, lambda: cc._certify_solve_matrix(_diag(4.0, 2.0, 1.0)))
    cc._decode(code, output, size, mode, rows)                              # control
    reply, payload = _split(output)
    shifts = np.frombuffer(payload, dtype='<i8').copy()
    shifts[0] = np.iinfo(np.int64).min
    _malformed(code, _join(reply, shifts.tobytes()), size, mode, rows)


def test_r7_p2_negative_forward_error_up_refuses(monkeypatch):
    code, output, size, mode, rows = _accepted(monkeypatch)
    cc._decode(code, output, size, mode, rows)                              # control: the genuine reply decodes
    reply, payload = _split(output)
    reply['forward_error_up'] = reply['forward_error_up'] | (1 << 63)
    _malformed(code, _join(reply, payload), size, mode, rows)


@pytest.mark.parametrize('key', ['svd_error_alpha', 'svd_error_bound', 'sigma_min', 'condition_number'])
def test_r7_p2_accepted_certificate_fields_are_non_null_finite_non_negative(monkeypatch, key):
    code, output, size, mode, rows = _accepted(monkeypatch)
    reply, payload = _split(output)
    for bad in (None, 0x7FF8_0000_0000_0000, reply['certificate'][key] | (1 << 63)):
        mutated = json.loads(json.dumps(reply))
        mutated['certificate'][key] = bad
        _malformed(code, _join(mutated, payload), size, mode, rows)


def test_r7_p2_repeated_json_key_refuses(monkeypatch):
    code, output, size, mode, rows = _accepted(monkeypatch)
    reply, payload = _split(output)
    text = json.dumps(reply, sort_keys=True, separators=(',', ':'))
    _malformed(code, _join(None, payload, text='{"verdict":"accept",' + text[1:]), size, mode, rows)


def _refusal_reply(reason, code='COUPLED_UNCERTIFIED', **extra):
    return _join(dict({'verdict': 'refuse', 'code': code, 'reason': reason, 'message': 'm', 'sections': []}, **extra),
                 b'')


def test_r7_p2_refusal_reason_is_checked_per_mode():
    """A certificate-mode run cannot refuse for a solve-only reason; a refusal must carry exactly its
    diagnostics."""
    _malformed(0, _refusal_reply('forward_error_above_limit'), 3, 'certificate', [])
    _malformed(0, _refusal_reply('candidate_failed'), 3, 'certificate', [])
    _malformed(0, _refusal_reply('backward_error_above_limit'), 3, 'solve', [])    # no certificate/backward
    decoded = cc._decode(0, _refusal_reply('equation_cap', code='COUPLED_TOO_LARGE'), 3, 'certificate', [])
    assert decoded['reason'] == 'equation_cap'                               # control


def test_r7_p2_trailing_reply_byte_refuses(monkeypatch):
    """Control (already closed on 6f7e67ce2): a byte after the listed sections is malformed."""
    code, output, size, mode, rows = _accepted(monkeypatch)
    _malformed(code, output + b'\0', size, mode, rows)


# --- P2: the request schema, decided by the real child -----------------------------------------------------

def _child(request: bytes) -> int:
    process = subprocess.run(cc._svd_worker_command(), input=request, capture_output=True,
                             env=cc._svd_worker_environment(), timeout=120)
    return process.returncode


def _request_bytes(header_edit=None, tail=b''):
    chunks = cc._request(_diag(4.0, 2.0, 1.0), None, mode='certificate', raw_load_zero=False, memory_limit=None,
                         exports=())
    head = chunks[0]
    length = struct.unpack('<Q', head[8:16])[0]
    text = head[16:16 + length].decode('ascii')
    if header_edit is not None:
        text = header_edit(text)
    body = text.encode('ascii')
    return b''.join([head[:8] + struct.pack('<Q', len(body)) + body] + chunks[1:]) + tail


def test_r7_p2_request_control_is_accepted():
    assert _child(_request_bytes()) == 0


def test_r7_p2_request_float_constant_refuses():
    assert '"cap":5000' in _request_bytes().decode('latin-1')
    assert _child(_request_bytes(lambda t: t.replace('"cap":5000', '"cap":5000.0'))) == 2


def test_r7_p2_request_repeated_key_refuses():
    assert _child(_request_bytes(lambda t: '{"version":2,' + t[1:])) == 2


def test_r7_p2_request_trailing_byte_refuses():
    assert _child(_request_bytes(tail=b'X')) == 2


# --- P2: empty rational coefficient list, and N = 0 through the child -----------------------------------------

def test_r7_p2_empty_rational_map_is_accepted():
    """S = [1], one offset-only row with an empty rational coefficient tuple: the child exited 2."""
    result = cc.certified_solve(_diag(1.0), np.ones(1), raw_load_zero=False,
                                exports=[_map([0, 0], [], (), offsets=[7.0])])
    (values, bounds), = result.exports
    assert values.tolist() == [7.0] and bounds.tolist() == [0.0]


def test_r7_p2_empty_system_evaluates_its_maps_in_the_child():
    """N = 0 with a one-row offset-only map: the value is the offset (it was returned as [])."""
    result = cc.certified_solve(sparse.csr_matrix((0, 0)), np.zeros(0), raw_load_zero=True,
                                exports=[_map([0, 0], [], np.zeros(0), offsets=[7.0])])
    (values, bounds), = result.exports
    assert values.tolist() == [7.0] and bounds.tolist() == [0.0]
    assert result.certificate['method'] == 'no_free_dofs' and result.certificate['accepted'] is True
    assert result.x.shape == (0,) and result.forward_error == 0 and 'certificate' in result.timings


# --- r5 #2: the deadline starts at the child's start; r6 e: reader MemoryError on a large reply ------------------

def test_r7_deadline_counts_from_the_child_start(monkeypatch):
    """The child takes 3.5 s, the candidate 2 s, the limit is 3 s: past the limit from the child's start,
    so the solve refuses timeout (the deadline started after the candidate and the solve succeeded)."""
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 3.0)
    patch_worker(monkeypatch, cc, 'import time\ntime.sleep(3.5)\n')
    calls = {'n': 0}

    def cancel():
        calls['n'] += 1
        if calls['n'] == 2:                     # the first poll inside the candidate
            time.sleep(2.0)
    error = refused('COUPLED_UNCERTIFIED', cc.certified_solve, _diag(4.0, 2.0, 1.0), np.ones(3),
                    raw_load_zero=False, cancel=cancel, reason='svd_worker')
    assert error.diagnostics['worker_failure'] == 'timeout', error.diagnostics


def test_r7_reader_memory_error_on_a_large_reply_is_too_large(monkeypatch):
    """The stdout reader runs out of memory while a reply larger than the pipe is written: the child
    blocks on the full pipe. The solve refuses COUPLED_TOO_LARGE at once, not a timeout."""
    monkeypatch.setattr(cc, '_SVD_WORKER_TIMEOUT_SECONDS', 30.0)
    original = cc._WorkerIO._read

    def failing(self, stream, sink, limit):
        if sink is self.stdout:
            raise MemoryError('reader buffer')
        return original(self, stream, sink, limit)
    monkeypatch.setattr(cc._WorkerIO, '_read', failing)
    rows = 20000
    export = _map(np.arange(rows + 1), np.zeros(rows), np.ones(rows))
    started = time.monotonic()
    refused('COUPLED_TOO_LARGE', cc.certified_solve, _diag(4.0, 2.0, 1.0), np.ones(3), raw_load_zero=False,
            exports=[export], reason='memory_error')
    assert time.monotonic() - started < 20.0
