"""Certificate SVD in a short-lived child process (private; owner decision OD13, 2026-11-20).

The coupled route's certificate relies on one LAPACK call, the singular values of the dense
equilibrated matrix (premise A1). Inside the solving process that call cannot be shielded from
same-process caller code (trace, profile and audit hooks, other threads changing the BLAS pools or
the floating-point environment). It therefore runs in a fresh interpreter that this module is the
whole program of:

    <sys.executable> -I -S _coupled_svd_worker.py <numpy package dir> <path> ...

``-I`` ignores every ``PYTHON*`` variable and the user site and keeps the script and working
directories off ``sys.path``; ``-S`` skips ``site``, so no ``.pth`` file, ``sitecustomize`` or
``usercustomize`` runs. ``sys.path`` is the standard library plus the given paths (the directory that
holds NumPy, and threadpoolctl's when it lies elsewhere). The environment is minimal (one BLAS, OpenMP
and MKL thread). The module imports nothing from anyfesolver, so no caller code runs in the child.

The parent (``_coupled_certificate``) imports the floating-point probes from here, so both sides
use the same code.

Protocol (little endian):
  stdin   MAGIC (8 bytes) | n (uint64) | n*n float64 in C order: the equilibrated matrix
  stdout  b'OK' | n (uint64) | n float64 singular values, descending        exit 0
          b'EN' | reason (ascii): the child's own environment check failed  exit 3
                  (blas_threads, subnormal_flush, rounding_mode, numpy_mismatch)
          b'MM'  MemoryError                                                exit 4
          b'LA'  LinAlgError (the SVD did not converge)                     exit 5
  stderr  one line 'ANYSVD-REPORT {json}' (diagnostics, never decided on); a traceback otherwise.
"""

from __future__ import annotations

import os
import struct
import sys
import time

_STARTED = time.perf_counter()

MAGIC = b'ANYSVD01'
REPORT_PREFIX = 'ANYSVD-REPORT '
EXIT_OK, EXIT_ENVIRONMENT, EXIT_MEMORY, EXIT_LINALG = 0, 3, 4, 5
ENVIRONMENT_REASONS = ('blas_threads', 'subnormal_flush', 'rounding_mode', 'numpy_mismatch')

_SIGN_BIT = 1 << 63
_ONE_BITS = 0x3FF0_0000_0000_0000
_HALF_ULP_BITS = 0x3CA0_0000_0000_0000          # 2^-53, half an ulp of 1.0


# --- Floating-point environment probes (shared with the parent) -----------------------------------------

def _float_from_bits(bits: int) -> float:
    return struct.unpack('<d', struct.pack('<Q', bits & 0xFFFF_FFFF_FFFF_FFFF))[0]


def _bits_of(value: float) -> int:
    """The signed int64 bit pattern of one binary64 value (a byte copy)."""
    return struct.unpack('<q', struct.pack('<d', value))[0]


def _float_bits(values):
    """int64 bit patterns of a float64 array (a byte copy; callers pass float64 only)."""
    import numpy as np
    return np.ascontiguousarray(values, dtype=np.float64).view(np.int64)


def _subnormals_preserved() -> bool:
    """False when this thread flushes subnormal operands or results (DAZ/FTZ), for Python floats
    or for NumPy's vector loops. The certificate premises (A2) and every float comparison of the
    route assume IEEE gradual underflow."""
    import numpy as np
    tiny = _float_from_bits(1)
    if _bits_of(tiny * 3.0) != 3 or not tiny != 0.0:
        return False
    vector = np.full(64, tiny)
    return bool((_float_bits(vector * 3.0) == 3).all()) and bool((vector != 0.0).all())


def _nearest_rounding() -> bool:
    """False unless this thread rounds to nearest-even, for Python floats and NumPy's vector loops
    (review r2 (a)). 1 + 2^-53 is a tie that rounds down to 1 and (1 + 2^-52) + 2^-53 a tie that
    rounds up to 1 + 2^-51; the negated pair rules out rounding toward zero. Operands come from bit
    patterns at run time, so nothing is folded at compile time."""
    import numpy as np
    one, half = _float_from_bits(_ONE_BITS), _float_from_bits(_HALF_ULP_BITS)
    odd = _float_from_bits(_ONE_BITS + 1)
    expected = (_ONE_BITS, _ONE_BITS + 2, _ONE_BITS | _SIGN_BIT, (_ONE_BITS + 2) | _SIGN_BIT)
    scalar = (one + half, odd + half, -one - half, -odd - half)
    if tuple(_bits_of(v) & 0xFFFF_FFFF_FFFF_FFFF for v in scalar) != expected:
        return False
    a = np.array([one, odd, -one, -odd] * 16)
    b = np.array([half, half, -half, -half] * 16)
    got = _float_bits(a + b).view(np.uint64).tolist()
    return got == list(expected) * 16


def _fp_environment_reason():
    if not _subnormals_preserved():
        return 'subnormal_flush'
    if not _nearest_rounding():
        return 'rounding_mode'
    return None


def _numpy_blas_dirs() -> tuple:
    import numpy as np
    root = os.path.normcase(os.path.dirname(os.path.abspath(np.__file__)))
    return (root + os.sep, root + '.libs' + os.sep)


def _blas_thread_counts():
    """(ok, counts): ok iff threadpoolctl is importable, NumPy's own BLAS (a library under NumPy's
    package directory or numpy.libs) is among the pools and every BLAS pool reports one thread."""
    try:
        from threadpoolctl import threadpool_info
        pools = [p for p in threadpool_info() if p.get('user_api') == 'blas']
    except Exception:  # noqa: BLE001  (missing or failing threadpoolctl: not verifiable)
        return False, None
    dirs = _numpy_blas_dirs()
    numpy_pools = [p for p in pools if os.path.normcase(str(p.get('filepath', ''))).startswith(dirs)]
    counts = [p.get('num_threads') for p in pools]
    return bool(numpy_pools) and all(c == 1 for c in counts), counts


# --- The child program ----------------------------------------------------------------------------------

def _read_exact(stream, count: int):
    buffer = bytearray(count)
    view, done = memoryview(buffer), 0
    while done < count:
        got = stream.readinto(view[done:])
        if not got:
            return None
        done += got
    return buffer


def _environment(report: dict, stage: str):
    """The child's own checks: round-to-nearest, gradual underflow, one BLAS thread."""
    reason = _fp_environment_reason()
    ok, counts = _blas_thread_counts()
    report[f'fp_{stage}'] = reason
    report[f'threads_{stage}'] = counts
    if reason is not None:
        return reason
    return None if ok else 'blas_threads'


def _main(argv) -> int:
    out = sys.stdout.buffer
    report = {'isolated': sys.flags.isolated, 'no_site': sys.flags.no_site,
              'ignore_environment': sys.flags.ignore_environment, 'no_user_site': sys.flags.no_user_site,
              'site_loaded': 'site' in sys.modules, 'sitecustomize_loaded': 'sitecustomize' in sys.modules,
              'environment_keys': sorted(os.environ)}

    def finish(tag: bytes, payload: bytes, code: int) -> int:
        out.write(tag)
        out.write(payload)
        out.flush()
        try:
            report['seconds_total'] = time.perf_counter() - _STARTED
            report['peak_commit_bytes'] = _peak_commit_bytes()
            sys.stderr.write(REPORT_PREFIX + _json(report) + '\n')
            sys.stderr.flush()
        except Exception:  # noqa: BLE001  (the report is diagnostics only)
            pass
        return code

    if len(argv) < 3:
        return 2
    sys.path[:0] = argv[2:]
    try:
        import numpy as np
        report['numpy'] = np.__version__
        report['seconds_startup'] = time.perf_counter() - _STARTED
        if os.path.normcase(os.path.dirname(os.path.abspath(np.__file__))) != os.path.normcase(argv[1]):
            return finish(b'EN', b'numpy_mismatch', EXIT_ENVIRONMENT)
        reason = _environment(report, 'start')
        if reason is not None:
            return finish(b'EN', reason.encode('ascii'), EXIT_ENVIRONMENT)
        head = _read_exact(sys.stdin.buffer, 16)
        if head is None or bytes(head[:8]) != MAGIC:
            return 2
        n = struct.unpack('<Q', bytes(head[8:]))[0]
        started = time.perf_counter()
        data = _read_exact(sys.stdin.buffer, 8 * n * n)
        if data is None or n == 0:
            return 2
        report['seconds_read'] = time.perf_counter() - started
        matrix = np.frombuffer(data, dtype='<f8').reshape(n, n)
        reason = _environment(report, 'before')
        if reason is not None:
            return finish(b'EN', reason.encode('ascii'), EXIT_ENVIRONMENT)
        started = time.perf_counter()
        try:
            sigma = np.linalg.svd(matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            return finish(b'LA', b'', EXIT_LINALG)
        report['seconds_svd'] = time.perf_counter() - started
        reason = _environment(report, 'after')
        if reason is not None:
            return finish(b'EN', reason.encode('ascii'), EXIT_ENVIRONMENT)
        values = np.ascontiguousarray(sigma, dtype='<f8')
        return finish(b'OK', struct.pack('<Q', n) + values.tobytes(), EXIT_OK)
    except MemoryError:
        return finish(b'MM', b'', EXIT_MEMORY)


def _peak_commit_bytes():
    """This process's peak commit (Windows; None elsewhere), for the memory measurement only."""
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakPagefileUsage)


def _json(value) -> str:
    import json
    return json.dumps(value, sort_keys=True, default=str)


if __name__ == '__main__':
    sys.exit(_main(sys.argv))
