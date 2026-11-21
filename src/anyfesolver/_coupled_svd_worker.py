"""The coupled route's whole certificate in a short-lived child process (private; owner decisions OD13,
2026-11-20, and OD14, 2026-11-21).

Inside the solving process the certificate cannot be shielded from same-process caller code (trace,
profile and audit hooks, other threads changing the BLAS pools or the floating-point environment):
review r6 showed a trace hook toggling FTZ/DAZ around the parent's equilibration. So every certificate
decision runs in a fresh interpreter that this module is the whole program of:

    <sys.executable> -I -S _coupled_svd_worker.py <numpy package dir> <threadpoolctl file>
                                                  <numpy version> <threadpoolctl sha256>

``-I`` ignores every ``PYTHON*`` variable and the user site and keeps the script and working
directories off ``sys.path``; ``-S`` skips ``site``, so no ``.pth`` file, ``sitecustomize`` or
``usercustomize`` runs. ``sys.path`` stays the standard library only. A finder appended to
``sys.meta_path`` (after the standard ones) resolves exactly one top-level name, ``numpy``, from the
directory that holds the given NumPy package; the loaded package must be that directory and have the
parent's version. ``threadpoolctl`` is executed from the bytes of exactly the given file, whose SHA-256
must equal the parent's (review r7 P1: a same-name package next to the file no longer wins). No other
module of those directories can be imported and no standard-library module can be shadowed (review r6
P1). The arithmetic module ``_coupled_certificate_math.py`` next to this file is loaded by path; nothing
else from anyfesolver is imported. The environment is minimal (one BLAS, OpenMP and MKL thread).

The child checks its environment (one BLAS thread, round-to-nearest, gradual underflow) at start,
after reading the request, between the equilibration and the SVD, after the SVD, after reading the
candidate and before replying. Protocol (little endian; docs/SOLID_COUPLING.md, "Certificate process"):

  stdin   b'ANYCRT02' | u64 L | header (L bytes of JSON) | arrays in header order
          [then, for candidate 'follows': b'XV' | n float64  or  b'XF' (no candidate)]
  stdout  b'ANYCRR02' | u64 L | reply (L bytes of JSON) | the sections the reply lists
  exit    0 verdict (accept, refuse, certificate, no_candidate); 3 environment; 4 memory; 2 protocol
  stderr  one line 'ANYSVD-REPORT {json}' (diagnostics and timings, never decided on).
"""

from __future__ import annotations

import os
import struct
import sys
import time

_STARTED = time.perf_counter()

REQUEST_MAGIC = b'ANYCRT02'
REPLY_MAGIC = b'ANYCRR02'
REPORT_PREFIX = 'ANYSVD-REPORT '
EXIT_OK, EXIT_PROTOCOL, EXIT_ENVIRONMENT, EXIT_MEMORY = 0, 2, 3, 4
ENVIRONMENT_REASONS = ('blas_threads', 'subnormal_flush', 'rounding_mode', 'numpy_mismatch',
                       'threadpoolctl_mismatch')
# The digest the parent sends when it found no readable threadpoolctl file (the child then has none).
NO_DIGEST = 'none'
MATH_FILE = '_coupled_certificate_math.py'
# Upper bounds on the request, checked before any allocation (the parent sends far less).
MAX_HEADER_BYTES = 1 << 24
MAX_EXPORTS = 64
MAX_EXPORT_ENTRIES = 1 << 26


class _ProtocolError(Exception):
    """The request is malformed (exit 2, no reply)."""


class _Environment(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --- import isolation ------------------------------------------------------------------------------------

class _ExactFinder:
    """Resolves only the listed top-level names, each from its one directory (appended to
    sys.meta_path, so built-in, frozen and standard-library modules always win). Since review r7 only
    ``numpy`` is listed (a package directory takes precedence over same-name files there, and the
    loaded package's directory and version are checked)."""

    def __init__(self, allowed: dict) -> None:
        self._allowed = dict(allowed)

    def find_spec(self, name, path=None, target=None):
        if path is not None or name not in self._allowed:
            return None
        from importlib.machinery import PathFinder
        return PathFinder.find_spec(name, [self._allowed[name]])

    def invalidate_caches(self) -> None:
        pass


def _install_finder(numpy_dir: str) -> None:
    sys.meta_path.append(_ExactFinder({'numpy': os.path.dirname(os.path.abspath(numpy_dir))}))


def _load_threadpoolctl(path: str, digest: str) -> None:
    """Execute threadpoolctl from the bytes of exactly ``path`` (one read: the bytes hashed are the
    bytes run) after checking their SHA-256 against the parent's ``digest`` (review r7 P1). Without a
    readable file, or when the module fails to execute, ``threadpoolctl`` stays unimportable and the
    thread probe refuses ``blas_threads`` (as before). A different file refuses
    ``threadpoolctl_mismatch``. A MemoryError propagates."""
    import hashlib
    from importlib.util import module_from_spec, spec_from_file_location
    sys.modules['threadpoolctl'] = None                      # unimportable unless loaded below
    path = os.path.abspath(path)
    if digest == NO_DIGEST:
        return
    try:
        with open(path, 'rb') as handle:
            source = handle.read()
    except OSError:
        return
    if hashlib.sha256(source).hexdigest() != digest:
        raise _Environment('threadpoolctl_mismatch')
    package = os.path.basename(path) == '__init__.py'
    spec = spec_from_file_location('threadpoolctl', path,
                                   submodule_search_locations=[os.path.dirname(path)] if package else None)
    module = module_from_spec(spec)
    sys.modules['threadpoolctl'] = module
    try:
        exec(compile(source, path, 'exec'), module.__dict__)
    except MemoryError:
        raise
    except Exception:  # noqa: BLE001  (a failing threadpoolctl: not verifiable, blas_threads)
        sys.modules['threadpoolctl'] = None


def _load_math():
    from importlib.util import module_from_spec, spec_from_file_location
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MATH_FILE)
    spec = spec_from_file_location('_coupled_certificate_math', path)
    module = module_from_spec(spec)
    sys.modules['_coupled_certificate_math'] = module
    spec.loader.exec_module(module)
    return module


# --- environment ---------------------------------------------------------------------------------------

def _numpy_blas_dirs(np) -> tuple:
    root = os.path.normcase(os.path.dirname(os.path.abspath(np.__file__)))
    return (root + os.sep, root + '.libs' + os.sep)


def _blas_thread_counts(np):
    """(ok, counts): ok iff threadpoolctl is importable, NumPy's own BLAS is among the pools and every
    BLAS pool reports one thread. A MemoryError propagates (review r6: not a probe failure)."""
    try:
        from threadpoolctl import threadpool_info
        pools = [p for p in threadpool_info() if p.get('user_api') == 'blas']
    except MemoryError:
        raise
    except Exception:  # noqa: BLE001  (missing or failing threadpoolctl: not verifiable)
        return False, None
    dirs = _numpy_blas_dirs(np)
    numpy_pools = [p for p in pools if os.path.normcase(str(p.get('filepath', ''))).startswith(dirs)]
    counts = [p.get('num_threads') for p in pools]
    return bool(numpy_pools) and all(c == 1 for c in counts), counts


def _check(math_module, np, report: dict, stage: str) -> None:
    reason = math_module._fp_environment_reason()
    ok, counts = _blas_thread_counts(np)
    report[f'fp_{stage}'] = reason
    report[f'threads_{stage}'] = counts
    if reason is not None:
        raise _Environment(reason)
    if not ok:
        raise _Environment('blas_threads')


# --- request -------------------------------------------------------------------------------------------

def _read_exact(stream, count: int) -> bytearray:
    buffer = bytearray(count)
    view, done = memoryview(buffer), 0
    while done < count:
        got = stream.readinto(view[done:])
        if not got:
            raise _ProtocolError('short read')
        done += got
    return buffer


def _array(stream, np, dtype: str, count: int):
    width = 8
    return np.frombuffer(_read_exact(stream, width * count), dtype=dtype).astype(dtype[1:], copy=True) \
        if count else np.zeros(0, dtype=dtype[1:])


def _is_int(value) -> bool:
    return type(value) is int


def _expected_constants(m) -> dict:
    return {'cap': m.COUPLED_MAX_EQUATIONS, 'condition_limit_bits': m._bits_of(m.CONDITION_LIMIT) & 0xFFFF_FFFF_FFFF_FFFF,
            'backward': m.encode_fraction(m.BACKWARD_TOLERANCE), 'forward': m.encode_fraction(m.FORWARD_TOLERANCE),
            'max_sweeps': m.EQUILIBRATION_MAX_SWEEPS, 'svd_constant_min': m.SVD_ERROR_CONSTANT_MIN,
            'dense_copies': m.DENSE_COPIES, 'sqrt_bits': m.SQRT_BITS}


_HEADER_KEYS = {'version', 'mode', 'n', 'nnz', 'raw_load_zero', 'candidate', 'memory_limit', 'constants', 'exports'}
_EXPORT_KEYS = {'rows', 'nnz', 'coefficients', 'text_bytes', 'offsets'}


def _strict_json(raw: bytes):
    """JSON with no float or constant literal and no repeated key in any object (review r7 P2)."""
    import json

    def no_float(text):
        raise ValueError('no float literal')

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('repeated key')
            out[key] = value
        return out
    return json.loads(raw.decode('ascii'), parse_float=no_float, parse_constant=no_float, object_pairs_hook=unique)


def _same(value, expected) -> bool:
    """value == expected with the exact type of every element (review r7: 2.0 is not the constant 2)."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(_same(value[k], expected[k]) for k in expected)
    if type(expected) is list:
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected))
    return value == expected


def _header(m, raw: bytes) -> dict:
    try:
        header = _strict_json(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise _ProtocolError('header') from exc
    if type(header) is not dict or set(header) != _HEADER_KEYS or header['version'] != 2 \
            or type(header['version']) is not int:
        raise _ProtocolError('header keys')
    n, nnz = header['n'], header['nnz']
    if not (_is_int(n) and _is_int(nnz) and 0 <= n and 0 <= nnz <= n * n):
        raise _ProtocolError('sizes')
    if header['mode'] not in ('certificate', 'solve') or type(header['mode']) is not str:
        raise _ProtocolError('mode')
    if type(header['raw_load_zero']) is not bool or type(header['candidate']) is not str:
        raise _ProtocolError('flags')
    limit = header['memory_limit']
    if limit is not None and not _is_int(limit):
        raise _ProtocolError('memory_limit')
    if not _same(header['constants'], _expected_constants(m)):
        raise _ProtocolError('constants')
    exports = header['exports']
    if type(exports) is not list or len(exports) > MAX_EXPORTS:
        raise _ProtocolError('exports')
    if header['mode'] == 'certificate':
        if header['candidate'] != 'none' or exports or header['raw_load_zero']:
            raise _ProtocolError('certificate mode')
    elif header['candidate'] != ('zero' if header['raw_load_zero'] else 'follows'):
        raise _ProtocolError('candidate')
    for item in exports:
        if type(item) is not dict or set(item) != _EXPORT_KEYS or not _is_int(item['rows']) \
                or not _is_int(item['nnz']) or not _is_int(item['text_bytes']) or type(item['offsets']) is not bool \
                or item['coefficients'] not in ('float', 'rational') or type(item['coefficients']) is not str \
                or not 0 <= item['rows'] <= MAX_EXPORT_ENTRIES or not 0 <= item['nnz'] <= MAX_EXPORT_ENTRIES \
                or not 0 <= item['text_bytes'] <= 64 * MAX_EXPORT_ENTRIES \
                or (item['coefficients'] == 'float' and item['text_bytes'] != 0) \
                or (item['coefficients'] == 'rational' and (item['text_bytes'] == 0) != (item['nnz'] == 0)):
            raise _ProtocolError('export')
    return header


def _read_export(stream, m, np, item: dict, n: int) -> tuple:
    indptr = _array(stream, np, '<i8', item['rows'] + 1)
    columns = _array(stream, np, '<i8', item['nnz'])
    if m.export_structure_reason(indptr, columns, n) is not None:     # incl. a repeated column (review r7 P1)
        raise _ProtocolError('export structure')
    if item['coefficients'] == 'float':
        data = _array(stream, np, '<f8', item['nnz'])
        if bool(((m._float_bits(data) & m._ABS_MASK) >= m._INF_BITS).any()):
            raise _ProtocolError('export coefficient')
        coefficients = [m._exact(v) for v in data]
    else:
        text = bytes(_read_exact(stream, item['text_bytes'])).decode('ascii') if item['text_bytes'] else ''
        parts = text.split(',') if text else []
        if len(parts) != item['nnz']:
            raise _ProtocolError('export text')
        try:
            coefficients = [m.decode_fraction(p) for p in parts]
        except ValueError as exc:
            raise _ProtocolError('export fraction') from exc
    offsets = None
    if item['offsets']:
        raw = _array(stream, np, '<f8', item['rows'])
        if bool(((m._float_bits(raw) & m._ABS_MASK) >= m._INF_BITS).any()):
            raise _ProtocolError('export offset')
        offsets = [m._exact(v) for v in raw]
    return indptr, columns, coefficients, offsets


# --- the certificate --------------------------------------------------------------------------------------

def _encode_certificate(m, certificate: dict) -> dict:
    out = {}
    for key, value in certificate.items():
        if key == 'shifts':
            continue
        if isinstance(value, float):              # Python float and np.float64
            out[key] = m._bits_of(float(value)) & 0xFFFF_FFFF_FFFF_FFFF
        else:
            out[key] = value
    return out


def _end_of_request(stdin) -> None:
    """The parent closes stdin after the last byte it sends: anything more is malformed (review r7)."""
    if stdin.read(1):
        raise _ProtocolError('trailing bytes')


def _run(m, np, stdin, report: dict) -> tuple:
    """Returns (reply dict, sections list of (name, bytes))."""
    head = _read_exact(stdin, 16)
    if bytes(head[:8]) != REQUEST_MAGIC:
        raise _ProtocolError('magic')
    length = struct.unpack('<Q', bytes(head[8:]))[0]
    if length > MAX_HEADER_BYTES:
        raise _ProtocolError('header length')
    header = _header(m, bytes(_read_exact(stdin, length)))
    n, nnz, solve = header['n'], header['nnz'], header['mode'] == 'solve'
    started = time.perf_counter()
    indptr = _array(stdin, np, '<i8', n + 1)
    indices = _array(stdin, np, '<i8', nnz)
    data = _array(stdin, np, '<f8', nnz)
    b = _array(stdin, np, '<f8', n) if solve else None
    exports = [_read_export(stdin, m, np, item, n) for item in header['exports']]
    if header['candidate'] != 'follows':
        _end_of_request(stdin)
    report['seconds_read'] = time.perf_counter() - started
    _check(m, np, report, 'request')

    def refuse(code, reason, message, **diagnostics):
        return {'verdict': 'refuse', 'code': code, 'reason': reason, 'message': message, **diagnostics}, []

    # Size, memory and system checks (the parent made them too; the child decides on its own copy).
    if n > m.COUPLED_MAX_EQUATIONS:
        return refuse('COUPLED_TOO_LARGE', 'equation_cap', f'the dense certificate is capped at '
                      f'{m.COUPLED_MAX_EQUATIONS} equations (system has {n})')
    if header['memory_limit'] is not None and m.dense_bytes(n) > header['memory_limit']:
        return refuse('COUPLED_TOO_LARGE', 'memory_limit', 'the dense certificate exceeds the memory limit')
    reason = m.canonical_csr_reason(n, indptr, indices, data)
    if reason is not None:
        return refuse('COUPLED_UNCERTIFIED', reason, f'the stored system is not certifiable ({reason})')
    system = m.CSR(n, indptr, indices, data)
    if solve:
        if bool(((m._float_bits(b) & m._ABS_MASK) >= m._INF_BITS).any()):
            return refuse('COUPLED_UNCERTIFIED', 'nonfinite_entry', 'the right-hand side is not finite')
        b_zero = not m._any_nonzero(b)
        if header['raw_load_zero'] and not b_zero:
            return refuse('COUPLED_UNCERTIFIED', 'zero_load_inconsistent',
                          'the zero-load path requires an exactly zero stored right-hand side')
        if not header['raw_load_zero'] and b_zero:
            return refuse('COUPLED_FORMATION_LOSS', 'load_stored_as_zero',
                          'a nonzero raw load was stored as an all-zero right-hand side')

    # Equilibration and SVD (sections 2.1-2.2).
    _check(m, np, report, 'before')
    started = time.perf_counter()
    if n == 0:
        # The 0 x 0 system (review r7): nothing to equilibrate or decompose; x = () is its solution.
        certificate, A, shifts = m.empty_certificate(), None, np.zeros(0, dtype=np.int64)
    else:
        certificate = m.new_certificate(n)
        A, shifts = m.equilibrate(m.dense_from_csr(system), certificate)
    report['seconds_equilibration'] = time.perf_counter() - started
    if A is not None:
        _check(m, np, report, 'between')
        started = time.perf_counter()
        try:
            sigma = np.linalg.svd(A, compute_uv=False)
        except np.linalg.LinAlgError:
            sigma = None
            certificate['reason'] = 'numerical_failure:LinAlgError'
        del A
        report['seconds_svd'] = time.perf_counter() - started
        _check(m, np, report, 'after_svd')
        if sigma is not None:
            m.singular_value_certificate(sigma, n, certificate)
    public = _encode_certificate(m, certificate)
    if not solve:
        sections = [('shifts', np.ascontiguousarray(shifts, dtype='<i8').tobytes())] if shifts is not None else []
        _check(m, np, report, 'after')
        return {'verdict': 'certificate', 'certificate': public}, sections
    if not certificate['accepted']:
        return refuse('COUPLED_UNCERTIFIED', certificate['reason'],
                      f"the coupled system is not certified ({certificate['reason']})", certificate=public)
    sigma_lower = m._sigma_lower(certificate['sigma_max'], certificate['sigma_min'], n) if n else m.Fraction(1)
    if not sigma_lower > 0:
        return refuse('COUPLED_UNCERTIFIED', 'singular_within_svd_error',
                      'the exact singular-value lower bound is not positive', certificate=public)

    # The candidate: zero on the zero-load path, else the parent's (it carries no claim).
    if header['candidate'] == 'zero':
        x = np.zeros(n)
    else:
        tag = bytes(_read_exact(stdin, 2))
        if tag == b'XF':
            _end_of_request(stdin)
            _check(m, np, report, 'after')
            return {'verdict': 'no_candidate', 'certificate': public}, []
        if tag != b'XV':
            raise _ProtocolError('candidate tag')
        x = _array(stdin, np, '<f8', n)
        _end_of_request(stdin)
        if bool(((m._float_bits(x) & m._ABS_MASK) >= m._INF_BITS).any()):
            return refuse('COUPLED_UNCERTIFIED', 'candidate_failed', 'the candidate solve is not finite',
                          certificate=public)
    _check(m, np, report, 'candidate')

    # Exact residual, V1-V3, V6 (section 2.3) and the bounded recovery values (section 5).
    started = time.perf_counter()
    rows = m._exact_rows(system, x, b)
    report['seconds_exact_residual'] = time.perf_counter() - started
    _check(m, np, report, 'residual')
    started = time.perf_counter()
    backward = m._backward_check(rows, system, x, b)
    if not backward['accepted']:
        backward = dict(backward, max_ratio=m._bits_of(backward['max_ratio']) & 0xFFFF_FFFF_FFFF_FFFF)
        return refuse('COUPLED_UNCERTIFIED', 'backward_error_above_limit',
                      'the exact residual exceeds 1e-10 of its row reference', certificate=public, backward=backward)
    error = m._scaled_residual_norm_hi(rows, shifts) / sigma_lower
    error_up = m._bits_of(m._out_up(error)) & 0xFFFF_FFFF_FFFF_FFFF
    if not m._forward_error_accepts(error, x, shifts):
        return refuse('COUPLED_UNCERTIFIED', 'forward_error_above_limit',
                      'the certified forward error exceeds 1e-6 of the solution scale', certificate=public,
                      forward_error_up=error_up)
    report['seconds_verification'] = time.perf_counter() - started
    started = time.perf_counter()
    sections = [('shifts', np.ascontiguousarray(shifts, dtype='<i8').tobytes()),
                ('x', np.ascontiguousarray(x, dtype='<f8').tobytes())]
    for k, (e_indptr, e_columns, coefficients, offsets) in enumerate(exports):
        values, bounds = m.bounded_rows(e_indptr, e_columns, coefficients, offsets, x, shifts, error)
        sections.append((f'values{k}', np.ascontiguousarray(values, dtype='<f8').tobytes()))
        sections.append((f'bounds{k}', np.ascontiguousarray(bounds, dtype='<f8').tobytes()))
    report['seconds_exports'] = time.perf_counter() - started
    _check(m, np, report, 'after')
    backward = dict(backward, max_ratio=m._bits_of(backward['max_ratio']) & 0xFFFF_FFFF_FFFF_FFFF)
    return {'verdict': 'accept', 'certificate': public, 'backward': backward,
            'forward_error': m.encode_fraction(error), 'forward_error_up': error_up,
            'sigma_lower': m.encode_fraction(sigma_lower)}, sections


def _write_reply(out, reply: dict, sections: list) -> None:
    import json
    reply = dict(reply, sections=[[name, len(payload)] for name, payload in sections])
    body = json.dumps(reply, sort_keys=True, separators=(',', ':')).encode('ascii')
    out.write(REPLY_MAGIC + struct.pack('<Q', len(body)) + body)
    for _, payload in sections:
        out.write(payload)
    out.flush()


def _main(argv) -> int:
    out = sys.stdout.buffer
    report = {'isolated': sys.flags.isolated, 'no_site': sys.flags.no_site,
              'ignore_environment': sys.flags.ignore_environment, 'no_user_site': sys.flags.no_user_site,
              'site_loaded': 'site' in sys.modules, 'sitecustomize_loaded': 'sitecustomize' in sys.modules,
              'environment_keys': sorted(os.environ), 'sys_path': list(sys.path)}

    def finish(reply: dict, sections: list, code: int) -> int:
        _write_reply(out, reply, sections)
        try:
            report['seconds_total'] = time.perf_counter() - _STARTED
            report['peak_commit_bytes'] = _peak_commit_bytes()
            sys.stderr.write(REPORT_PREFIX + _json(report) + '\n')
            sys.stderr.flush()
        except Exception:  # noqa: BLE001  (the report is diagnostics only)
            pass
        return code

    if len(argv) != 5:
        return EXIT_PROTOCOL
    try:
        _install_finder(argv[1])
        try:
            import numpy as np
        except ImportError:
            return finish({'verdict': 'environment', 'reason': 'numpy_mismatch'}, [], EXIT_ENVIRONMENT)
        report['numpy'] = np.__version__
        report['seconds_startup'] = time.perf_counter() - _STARTED
        if os.path.normcase(os.path.dirname(os.path.abspath(np.__file__))) != os.path.normcase(os.path.abspath(argv[1])) \
                or np.__version__ != argv[3]:
            return finish({'verdict': 'environment', 'reason': 'numpy_mismatch'}, [], EXIT_ENVIRONMENT)
        _load_threadpoolctl(argv[2], argv[4])
        m = _load_math()
        report['modules'] = sorted(name for name in sys.modules if '.' not in name)
        _check(m, np, report, 'start')
        reply, sections = _run(m, np, sys.stdin.buffer, report)
        report['modules'] = sorted(name for name in sys.modules if '.' not in name)
        return finish(reply, sections, EXIT_OK)
    except _Environment as exc:
        return finish({'verdict': 'environment', 'reason': exc.reason}, [], EXIT_ENVIRONMENT)
    except MemoryError:
        return finish({'verdict': 'memory'}, [], EXIT_MEMORY)
    except _ProtocolError:
        return EXIT_PROTOCOL
    except Exception as exc:  # noqa: BLE001
        refusal = sys.modules.get('_coupled_certificate_math')
        if refusal is not None and isinstance(exc, refusal.CertificateRefusal):
            return finish({'verdict': 'refuse', 'code': exc.code, 'reason': exc.reason, 'message': exc.message},
                          [], EXIT_OK)
        raise


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
