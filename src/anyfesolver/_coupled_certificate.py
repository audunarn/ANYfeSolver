"""Dense certificate and exact verification of the coupled shell-solid KKT solve (private).

Ported from the PR #67 solid port (b94c9179f, ``solids.py``) with the cap raised to 5000 equations,
the ``COUPLED_*`` codes and no row normalisation: the certified matrix is exactly the stored system
``S``. Added here: the exact inverse-norm bound (spec section 2.2), the exact integer residual, the
backward checks V1-V3 and the forward-error decision V6 (section 2.3), the zero-load path (2.4) and
the SuperLU candidate (2.5). See ``docs/SOLID_COUPLING.md``.

Owner decision OD14 (2026-11-21): the whole certificate runs in an isolated child process
(``_coupled_svd_worker``): equilibration, SVD, delta and kappa_up, the exact residual, V1-V3, E and V6
and the bounded recovery values. This process forms the system, computes the SuperLU candidate (which
carries no claim), sends the stored system, b, the candidate and the export maps as raw bytes, and
decodes the child's verdict against an exact schema. It does no certificate arithmetic.

The only premises are (A1) LAPACK's SVD backward error and (A2) one rounding per equilibrated entry,
inexact only for a subnormal result. Every other decision is exact rational arithmetic.

Not exported from :mod:`anyfesolver`; the sole caller is :mod:`anyfesolver.solid_coupling`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import functools
import os
import struct
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from . import _coupled_certificate_math as _m
from . import _coupled_svd_worker as _worker
from ._coupled_certificate_math import (  # noqa: F401  (re-exported for solid_coupling and the tests)
    BACKWARD_TOLERANCE,
    CONDITION_LIMIT,
    COUPLED_MAX_EQUATIONS,
    ExactRows,
    FORWARD_TOLERANCE,
    _ABS_MASK,
    _FRACTION_BITS,
    _INF_BITS,
    _SIGN_BIT,
    _abs_max_exact,
    _any_nonzero,
    _backward_check,
    _bits_of,
    _condition_upper_bound,
    _decompose,
    _exact,
    _exact_rows,
    _float_bits,
    _float_from_bits,
    _forward_error_accepts,
    _forward_scale,
    _fp_environment_reason,
    _int_to_float,
    _nearest_rounding,
    _nonzero_mask,
    _out_up,
    _ratio_to_float,
    _round_down,
    _round_up,
    _row_abs_sums,
    _scaled,
    _scaled_residual_norm_hi,
    _sigma_lower,
    _sqrt_hi,
    _subnormals_preserved,
    _to_float,
)

__all__: list[str] = []

_EQUILIBRATION_MAX_SWEEPS = _m.EQUILIBRATION_MAX_SWEEPS
_SVD_ERROR_CONSTANT_MIN = _m.SVD_ERROR_CONSTANT_MIN
_DENSE_COPIES = _m.DENSE_COPIES
_EPS = _m.EPS
_REFINEMENT_STEPS = 5   # owner decision OD12 (2026-11-20): 2 -> 5
# Nonzero |components| in this range cannot over- or underflow when squared and summed (n < 2^23).
_PLAIN_NORM_RANGE = (2.0 ** -500, 2.0 ** 500)


class CoupledSolveError(Exception):
    """A coded refusal of the coupled route. ``code`` is one of the ``COUPLED_*`` codes, ``reason``
    a stable sub-reason; callers match on these, never on the message."""

    def __init__(self, message: str, *, code: str, reason: Optional[str] = None,
                 diagnostics: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.diagnostics = dict(diagnostics or {})


def _refuse(code: str, reason: Optional[str], message: str, **diagnostics: Any) -> CoupledSolveError:
    return CoupledSolveError(message, code=code, reason=reason, diagnostics=diagnostics)


_TYPE_NAME = type.__dict__['__name__']          # reads tp_name: no metaclass hook runs
_EXCEPTION_ARGS = BaseException.__dict__['args']


def _type_name(exc: BaseException) -> str:
    """The class name of an exception, read without running any user-defined code (review r2 (d))."""
    try:
        name = _TYPE_NAME.__get__(type(exc), type)
    except Exception:  # noqa: BLE001
        return 'Exception'
    return name if type(name) is str else 'Exception'


def _exception_text(exc: BaseException) -> str:
    """'TypeName: message' without calling any __str__, __repr__ or __format__ (review r2 (d))."""
    try:
        args = _EXCEPTION_ARGS.__get__(exc, BaseException)
    except Exception:  # noqa: BLE001
        args = ()
    parts = [a for a in args if type(a) is str] if type(args) is tuple else []
    text = '; '.join(parts) if parts else '<not shown>'
    return _type_name(exc) + ': ' + text


# Review r5 #3: a refusal is built inside a bounded retry. A trace or profile function that raises is
# unset by the interpreter after that one exception, so a hook raising while a refusal is built costs
# one attempt, not the refusal's code.
_TRANSLATION_ATTEMPTS = 4


def _built(make: Callable[[], BaseException]) -> BaseException:
    """``make()`` in a bounded retry (review r5 #3, r6 P2): a one-shot hook raising while a refusal is
    built costs one attempt. After the attempts, the last exception raised by the hook propagates."""
    failure: Optional[BaseException] = None
    for _ in range(_TRANSLATION_ATTEMPTS):
        try:
            return make()
        except Exception as exc:  # noqa: BLE001
            failure = exc
    raise failure  # type: ignore[misc]


def _require_fp_environment() -> None:
    """Gradual underflow and round-to-nearest-even on this thread, else a coded refusal.

    Review r5 #3: the probe itself never raises, so any exception from inside it was raised by caller
    code running there; it refuses ``COUPLED_UNCERTIFIED`` (``fp_probe``)."""
    try:
        reason = _fp_environment_reason()
        failure = None
    except Exception as exc:  # noqa: BLE001
        reason, failure = 'fp_probe', exc
    if reason is None:
        return
    for _ in range(_TRANSLATION_ATTEMPTS):
        try:
            if reason == 'subnormal_flush':
                refusal = _refuse('COUPLED_UNCERTIFIED', 'subnormal_flush',
                                  'the floating-point environment flushes subnormals (FTZ/DAZ); the certificate '
                                  'premise A2 and exact float comparisons do not hold')
            elif reason == 'rounding_mode':
                refusal = _refuse('COUPLED_UNCERTIFIED', 'rounding_mode',
                                  'the floating-point environment does not round to nearest-even; the '
                                  'certificate premises A1/A2 do not hold')
            else:
                refusal = _refuse('COUPLED_UNCERTIFIED', 'fp_probe',
                                  'the floating-point environment probe was interrupted by caller code '
                                  f'({_type_name(failure)}); the environment is not verified')
            break
        except Exception:  # noqa: BLE001  (a hook raised while the refusal was built)
            continue
    else:
        raise _refuse('COUPLED_UNCERTIFIED', 'fp_probe', 'the floating-point environment is not verified')
    raise refusal from failure


# --- Ported helpers (b94c9179f solids.py, unchanged; formation and candidate only) ---------------------

def _plain_norm_safe(vector: np.ndarray) -> bool:
    magnitude = np.abs(vector[vector != 0])
    return bool(magnitude.size == 0 or (magnitude.min() >= _PLAIN_NORM_RANGE[0]
                                        and magnitude.max() <= _PLAIN_NORM_RANGE[1]))


def _ignore_fp(function):
    """Run under a local np.errstate(all='ignore'): a caller's 'raise' cannot surface an uncoded
    FloatingPointError. IEEE results do not depend on errstate, so values are unchanged."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with np.errstate(all='ignore'):
            return function(*args, **kwargs)
    return wrapped


@_ignore_fp
def _scaled_norm(vector: Any) -> float:
    """Euclidean norm without overflow or underflow loss; bit-identical to np.linalg.norm when every
    nonzero |component| lies in [2^-500, 2^500]. nan for a non-finite vector, inf beyond binary64."""
    v = np.asarray(vector, dtype=float).ravel()
    if not np.isfinite(v).all():
        return float('nan')
    if _plain_norm_safe(v):
        return float(np.linalg.norm(v))
    exponent = int(np.frexp(np.max(np.abs(v)))[1])
    root = float(np.linalg.norm(np.ldexp(v, -exponent)))  # in [0.5, sqrt(n)]
    if int(np.frexp(root)[1]) + exponent > np.finfo(float).maxexp:
        return float('inf')
    return float(np.ldexp(root, exponent))


@_ignore_fp
def _scaled_unit(vector: Any) -> Optional[np.ndarray]:
    """vector / ||vector|| for a finite nonzero vector, or None (the caller refuses with its code)."""
    v = np.asarray(vector, dtype=float)
    if not np.isfinite(v).all() or not np.any(v):
        return None
    if not _plain_norm_safe(v.ravel()):
        v = np.ldexp(v, -int(np.frexp(np.max(np.abs(v)))[1]))
    unit = v / np.linalg.norm(v)
    if not (np.isfinite(unit).all() and abs(float(np.linalg.norm(unit)) - 1.0) <= 1e-12):
        return None
    return unit


def _asymmetric(stiffness: sparse.csr_matrix) -> bool:
    """Symmetry check ||K - K^T||_F > 1e-10 ||K||_F (entries finite), robust to over/underflow."""
    with np.errstate(all='ignore'):
        norm = sparse.linalg.norm(stiffness)
        asymmetry = sparse.linalg.norm(stiffness - stiffness.T)
        if np.isfinite(norm) and np.isfinite(asymmetry) and norm >= 2.0 ** -400:
            return bool(asymmetry > 1e-10 * max(norm, np.finfo(float).tiny))
        largest = float(np.max(np.abs(stiffness.data), initial=0.0))
        if largest == 0.0:
            return False
        scaled = stiffness.copy()
        scaled.data = np.ldexp(scaled.data, -int(np.frexp(largest)[1]))
        return bool(sparse.linalg.norm(scaled - scaled.T) > 1e-10 * sparse.linalg.norm(scaled))


def _dense_bytes(size: int) -> int:
    return _m.dense_bytes(size)


def _require_system(system: Any) -> sparse.csr_matrix:
    if type(system) is not sparse.csr_matrix or system.dtype != np.float64 or system.ndim != 2 \
            or system.shape[0] != system.shape[1]:
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_system',
                      'the coupled system must be a square float64 scipy.sparse.csr_matrix')
    if not system.has_canonical_format:
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_system', 'the coupled system must be canonical CSR')
    return system


# --- The certificate process (owner decisions OD13, OD14; review r8: docs "Certificate process: states") --

_SVD_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_coupled_svd_worker.py')
# Hard bound on one worker run (review r8: ONE deadline, started before any child-related I/O, the
# provenance read included, and carried through the candidate to the verdict; N = 5000 takes about 30 s).
_SVD_WORKER_TIMEOUT_SECONDS = 600.0
# Interval of the cancellation polls while the worker runs.
_SVD_WORKER_POLL_SECONDS = 0.05
# After the child has exited, its lanes get at most this long: an exited child's unread output is at
# most one pipe buffer per stream, so this drain does not depend on the deadline (review r8).
_SVD_WORKER_IO_SECONDS = 10.0
# Teardown (kill, reap, lanes) on every path, after the deadline too.
_SVD_WORKER_STOP_SECONDS = 30.0
_SVD_WORKER_STDERR_LIMIT = 1 << 16
_REPLY_JSON_LIMIT = 1 << 22
_SHIFT_LIMIT = 1 << 20
_SVD_WORKER_ENVIRONMENT_REASONS = ('blas_threads', 'subnormal_flush', 'rounding_mode')


def _remaining(deadline: float) -> float:
    import time
    return deadline - time.monotonic()


# --- Provenance of the modules the child trusts (review r8 P1) ---------------------------------------------

@dataclass(frozen=True)
class _Runtime:
    """What the child is started with: NumPy's package directory and version, and threadpoolctl's
    RECORD-checked bytes (sent in the request) with their SHA-256 (in argv)."""

    numpy_dir: str
    numpy_version: str
    threadpoolctl_path: str
    threadpoolctl_source: bytes
    threadpoolctl_digest: str


def _provenance_refusal(reason: str, message: str) -> CoupledSolveError:
    return _refuse('COUPLED_UNCERTIFIED', reason, f'the certificate cannot vouch for its modules: {message}')


def _installed(site: str, name: str, reason: str) -> Any:
    """The one installed distribution ``name`` whose dist-info lies in ``site`` (importlib.metadata)."""
    from importlib import metadata
    try:
        found = list(metadata.distributions(name=name, path=[site]))
    except (OSError, ValueError, UnicodeDecodeError):
        found = None
    if found is None or len(found) != 1:
        raise _provenance_refusal(reason, f'no single installed {name} distribution next to NumPy')
    return found[0]


def _recorded_bytes(distribution: Any, site: str, relative: str, reason: str,
                    expected: Optional[str] = None) -> tuple:
    """(path, bytes) of the file ``relative`` (POSIX, relative to ``site``) of an installed distribution,
    read once and checked against the SHA-256 that the distribution's RECORD lists for it. With
    ``expected`` the file must also be that path (the module this process loaded)."""
    import base64
    import hashlib
    try:
        entries = [f for f in (distribution.files or ()) if str(f) == relative]
    except (OSError, ValueError, UnicodeDecodeError):
        entries = []
    if len(entries) != 1 or entries[0].hash is None or entries[0].hash.mode != 'sha256':
        raise _provenance_refusal(reason, f'{relative} has no SHA-256 entry in its RECORD')
    path = os.path.join(site, *relative.split('/'))
    if expected is not None and os.path.normcase(os.path.abspath(expected)) != os.path.normcase(os.path.abspath(path)):
        raise _provenance_refusal(reason, f'the loaded module is not the installed {relative}')
    try:
        with open(path, 'rb') as handle:
            data = handle.read()
    except OSError:
        raise _provenance_refusal(reason, f'{relative} cannot be read') from None
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')
    if digest != entries[0].hash.value:
        raise _provenance_refusal(reason, f'{relative} does not match its RECORD')
    return path, data


def _trusted_runtime(numpy_module: Any = None) -> _Runtime:
    """Resolve every module the child trusts from the installed distributions, never from
    ``sys.modules`` or ``sys.path`` (review r8 P1; inventory in docs/SOLID_COUPLING.md):

    - NumPy: the package this process runs must be the installed ``numpy`` distribution of its own
      directory: one dist-info there, the same version, ``numpy/__init__.py`` that very file and equal
      to its RECORD hash (``numpy_mismatch`` otherwise). The child imports NumPy from that directory
      only and checks directory and version again.
    - threadpoolctl: the one installed distribution in NumPy's directory; ``threadpoolctl.py`` is read
      once and checked against its RECORD hash (``threadpoolctl_mismatch`` otherwise), and exactly
      those bytes are sent to the child, which runs them after checking their SHA-256.
    - The standard library is the interpreter's own (``-I -S``); the worker script and the arithmetic
      module are this package's files, loaded by path. Files on disk are trusted (OD9).

    Blocking file I/O: :func:`_start_worker` runs this on a lane inside the run's deadline."""
    import hashlib
    module = np if numpy_module is None else numpy_module
    numpy_file, version = module.__dict__.get('__file__'), module.__dict__.get('__version__')
    if type(numpy_file) is not str or type(version) is not str:
        raise _provenance_refusal('numpy_mismatch', 'NumPy has no file or version')
    numpy_dir = os.path.dirname(os.path.abspath(numpy_file))
    site = os.path.dirname(numpy_dir)
    if os.path.normcase(os.path.basename(numpy_dir)) != os.path.normcase('numpy'):
        raise _provenance_refusal('numpy_mismatch', 'NumPy is not a numpy package directory')
    numpy_dist = _installed(site, 'numpy', 'numpy_mismatch')
    try:
        recorded_version = numpy_dist.version
    except (OSError, ValueError, KeyError, UnicodeDecodeError):
        recorded_version = None
    if type(recorded_version) is not str or recorded_version != version:
        raise _provenance_refusal('numpy_mismatch', 'the loaded NumPy is not the installed version')
    _recorded_bytes(numpy_dist, site, 'numpy/__init__.py', 'numpy_mismatch', expected=numpy_file)
    path, source = _recorded_bytes(_installed(site, 'threadpoolctl', 'threadpoolctl_mismatch'), site,
                                   'threadpoolctl.py', 'threadpoolctl_mismatch')
    if not 0 < len(source) <= _m.MAX_THREADPOOLCTL_BYTES:
        raise _provenance_refusal('threadpoolctl_mismatch', 'threadpoolctl.py has an unexpected size')
    return _Runtime(numpy_dir, version, path, source, hashlib.sha256(source).hexdigest())


def _svd_worker_command(runtime: Optional[_Runtime] = None) -> list:
    """argv of the worker: this interpreter, isolated (-I) and without site (-S), the worker script,
    NumPy's package directory and version, and the SHA-256 of the threadpoolctl bytes sent in the
    request (review r6 P1, r7 P1, r8 P1). Without ``runtime`` it is resolved here (tests)."""
    runtime = _trusted_runtime() if runtime is None else runtime
    return [sys.executable, '-I', '-S', _SVD_WORKER_PATH, runtime.numpy_dir, runtime.numpy_version,
            runtime.threadpoolctl_digest]


def _svd_worker_environment() -> dict:
    """The worker's whole environment: one BLAS, OpenMP and MKL thread, plus SYSTEMROOT on Windows."""
    environment = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    if sys.platform == 'win32':
        root = os.environ.get('SYSTEMROOT')
        if type(root) is str and root:
            environment['SYSTEMROOT'] = root
    return environment


def _worker_refusal(detail: str, message: str, **diagnostics: Any) -> CoupledSolveError:
    return _refuse('COUPLED_UNCERTIFIED', 'svd_worker', f'the certificate worker failed: {message}',
                   worker_failure=detail, **diagnostics)


def _timeout() -> CoupledSolveError:
    return _worker_refusal('timeout', f'no result within {_SVD_WORKER_TIMEOUT_SECONDS:.0f} s')


# --- Lanes and the run ----------------------------------------------------------------------------------------

class _Lane:
    """One thread, started with ``_thread.start_new_thread`` (no wait on the thread registry or on a
    'started' event: review r6 P2) and joined through its own lock with a bounded wait. A start that
    fails raises from the constructor (the caller tears the run down)."""

    def __init__(self, target: Callable, *args: Any) -> None:
        import _thread
        self.error: Optional[BaseException] = None
        self.result: Any = None
        self._done = _thread.allocate_lock()
        self._done.acquire()
        _thread.start_new_thread(self._run, (target, args))

    def _run(self, target: Callable, args: tuple) -> None:
        try:
            self.result = target(*args)
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
        finally:
            self._done.release()

    def finished(self, timeout: float) -> bool:
        if self._done.acquire(timeout=max(0.0, timeout)):
            self._done.release()
            return True
        return False


def _write_all(stream: Any, data: Any) -> None:
    view = memoryview(data).cast('B')
    while len(view):
        written = stream.write(view[:1 << 24])
        if written is None or written <= 0:
            raise OSError('the worker pipe accepted no bytes')
        view = view[written:]


def _close(stream: Any) -> None:
    try:
        if stream is not None:
            stream.close()
    except Exception:  # noqa: BLE001
        pass


class _WorkerIO:
    """Feeds the request (and, later, the candidate) to the worker's stdin and drains stdout and
    stderr on three lanes, so the calling thread stays free to compute the candidate and to poll.
    Each lane closes its own pipe end when it ends; the coordinator closes only pipe ends whose lane
    has finished or never started (a close never races a lane blocked on the stream)."""

    def __init__(self, process: Any, request: list, candidate_follows: bool, stdout_limit: int,
                 deadline: float) -> None:
        import _thread
        self.deadline = deadline
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.overflow = False
        self._candidate: Optional[bytes] = None
        self._aborted = False
        self._ready = _thread.allocate_lock()
        self._ready.acquire()
        self._streams = (process.stdin, process.stdout, process.stderr)
        self._plan = ((self._write, process.stdin, request, candidate_follows),
                      (self._read, process.stdout, self.stdout, stdout_limit),
                      (self._read, process.stderr, self.stderr, _SVD_WORKER_STDERR_LIMIT))
        self._lanes: list = []

    def start(self) -> None:
        """START: one lane per stream, each started only within the deadline. A lane that cannot start
        raises; the caller then tears the whole run down at once (review r8 P2)."""
        for target, *args in self._plan:
            if _remaining(self.deadline) <= 0:
                raise _timeout()
            self._lanes.append(_Lane(target, *args))

    def post_candidate(self, payload: bytes) -> None:
        if self._candidate is None and not self._aborted:
            self._candidate = payload
            self._ready.release()

    def abort(self) -> None:
        """Release a writer still waiting for the candidate (teardown): it writes nothing more."""
        if self._candidate is None and not self._aborted:
            self._aborted = True
            self._ready.release()

    def _write(self, stream: Any, request: list, candidate_follows: bool) -> None:
        try:
            for chunk in request:
                _write_all(stream, chunk)
            if candidate_follows:
                if not self._ready.acquire(timeout=max(0.0, _remaining(self.deadline))) or self._aborted:
                    return
                _write_all(stream, self._candidate)
        except (BrokenPipeError, OSError, ValueError):
            pass                                  # the worker exited early; its exit status decides
        finally:
            _close(stream)

    def _read(self, stream: Any, sink: bytearray, limit: int) -> None:
        try:
            while True:
                chunk = stream.read(1 << 16)
                if not chunk:
                    break
                room = limit - len(sink)
                if room > 0:
                    sink += chunk[:room]
                if len(chunk) > room and sink is self.stdout:
                    self.overflow = True
        finally:
            _close(stream)

    def finished(self, timeout: float) -> bool:
        import time
        deadline = time.monotonic() + timeout
        return all(lane.finished(deadline - time.monotonic()) for lane in self._lanes)

    @property
    def errors(self) -> list:
        return [lane.error for lane in self._lanes if lane.error is not None]

    def close_finished(self) -> None:
        """Close the parent's pipe ends whose lane has finished or never started."""
        for k, stream in enumerate(self._streams):
            if k >= len(self._lanes) or self._lanes[k].finished(0.0):
                _close(stream)


def _reap(process: Any) -> None:
    """Kill and reap a process by its own handle (never by image name); every wait is bounded. Each
    step is retried, so a one-shot failure cannot leave the child running (review r6 P2)."""
    if process is None:
        return
    for _ in range(_TRANSLATION_ATTEMPTS):
        try:
            if process.__dict__.get('pid') is None:
                break                             # Popen.__init__ never created the process
            if process.poll() is None:
                process.kill()
            process.wait(timeout=_SVD_WORKER_STOP_SECONDS)
            break
        except Exception:  # noqa: BLE001
            continue


def _worker_report(stderr: bytes) -> Optional[dict]:
    import json
    for line in bytes(stderr).decode('utf-8', 'replace').splitlines():
        if line.startswith(_worker.REPORT_PREFIX):
            try:
                value = json.loads(line[len(_worker.REPORT_PREFIX):])
            except ValueError:
                return None
            return value if type(value) is dict else None
    return None


class _Run:
    """One worker run through the states of docs/SOLID_COUPLING.md ("Certificate process: states"):
    PREPARE and LAUNCH (on a lane), START, RUNNING, then COLLECT or EXPIRED, and TEARDOWN on every
    path. ``deadline`` (monotonic) is the one deadline, set before any child-related I/O."""

    def __init__(self, deadline: float) -> None:
        import _thread
        self.deadline = deadline
        self.process: Any = None
        self.io: Optional[_WorkerIO] = None
        self._abandoned = False
        self._lock = _thread.allocate_lock()

    def adopt(self, process: Any) -> bool:
        """LAUNCH hands the created process over; False once the run was abandoned (the launching lane
        then kills and reaps it itself)."""
        with self._lock:
            if self._abandoned:
                return False
            self.process = process
            return True

    @property
    def abandoned(self) -> bool:
        return self._abandoned

    def post_candidate(self, x: Optional[np.ndarray]) -> None:
        if x is None:
            self.io.post_candidate(b'XF')
        else:
            self.io.post_candidate(b'XV' + np.ascontiguousarray(x, dtype='<f8').tobytes())

    def result(self, cancel: Optional[Callable[[], None]]):
        """RUNNING until the child exits (COLLECT) or the deadline passes with the child still running
        (EXPIRED). Verdict precedence (review r8): once the child has exited, its complete reply
        decides, whatever the clock says when the parent looks (a suspended parent does not turn a
        timely verdict into a timeout). Lane failures are checked at every poll, after the final read
        and after an expiry, so a reader MemoryError is COUPLED_TOO_LARGE, never a timeout."""
        import subprocess
        while True:
            if cancel is not None:
                _poll(cancel)                   # caller code; anything it raises propagates (stop follows)
            self._lane_failure(None)
            if self.process.poll() is not None:
                break                           # COLLECT
            remaining = _remaining(self.deadline)
            if remaining <= 0:
                self._expire()                  # EXPIRED: raises
            try:
                self.process.wait(timeout=min(_SVD_WORKER_POLL_SECONDS, remaining))
                break                           # COLLECT
            except subprocess.TimeoutExpired:
                continue
        complete = self.io.finished(_SVD_WORKER_IO_SECONDS)
        code = self.process.returncode
        report = _worker_report(self.io.stderr)
        self._lane_failure(code, report)        # after the final read (review r8)
        if not complete:
            raise _worker_refusal('io', 'its output could not be read', exit_code=code)
        if self.io.overflow:
            raise _worker_refusal('bad_output', 'its output is too long', exit_code=code)
        return code, bytes(self.io.stdout), report

    def _expire(self) -> None:
        """EXPIRED: the child was still running at the deadline. TEARDOWN first; then a lane failure
        (MemoryError: COUPLED_TOO_LARGE) wins over the timeout."""
        self.stop()
        self._lane_failure(None)
        raise _timeout()

    def _lane_failure(self, code: Optional[int], report: Optional[dict] = None) -> None:
        """Refuse if a transfer lane raised: MemoryError -> COUPLED_TOO_LARGE, else svd_worker / io."""
        errors = self.io.errors if self.io is not None else []
        if any(isinstance(e, MemoryError) for e in errors):
            raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory reading the certificate worker',
                          worker_report=report)
        if errors:
            raise _worker_refusal('io', 'its output could not be read', exit_code=code)

    def stop(self) -> None:
        """TEARDOWN (idempotent, every path): abandon the run (a launch still in progress reaps its own
        process), release the writer, kill and reap the child by its own handle, wait for the lanes
        (bounded) and close the pipe ends no lane uses."""
        with self._lock:
            self._abandoned = True
            process = self.process
        io = self.io
        if io is not None:
            io.abort()
        _reap(process)
        for _ in range(_TRANSLATION_ATTEMPTS):
            try:
                if io is not None:
                    io.finished(_SVD_WORKER_STOP_SECONDS)
                    io.close_finished()
                elif process is not None:
                    for name in ('stdin', 'stdout', 'stderr'):
                        _close(process.__dict__.get(name))
                break
            except Exception:  # noqa: BLE001
                continue


def _launch(run: _Run, header: dict, chunks: list) -> Optional[list]:
    """PREPARE and LAUNCH, on a lane inside the deadline: resolve the trusted modules (file I/O that may
    block), build argv and the request, create the process and hand it to the run. Returns the
    request chunks, or None when the run was abandoned meanwhile (any process is then reaped here)."""
    import subprocess
    runtime = _trusted_runtime()
    command, environment = _svd_worker_command(runtime), _svd_worker_environment()
    request = _encode_request(header, chunks, runtime)
    if run.abandoned:
        return None
    flags = 0x08000000 if sys.platform == 'win32' else 0          # CREATE_NO_WINDOW
    process = subprocess.Popen.__new__(subprocess.Popen)
    try:
        process.__init__(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         bufsize=0, env=environment, cwd=os.path.dirname(_SVD_WORKER_PATH), close_fds=True,
                         creationflags=flags)
        if process.__dict__.get('pid') is None:
            raise OSError('the worker process was not created')
    except BaseException:
        _reap(process)
        raise
    if not run.adopt(process):
        _reap(process)
        for name in ('stdin', 'stdout', 'stderr'):
            _close(process.__dict__.get(name))
        return None
    return request


def _start_worker(header: dict, chunks: list, candidate_follows: bool, stdout_limit: int,
                  deadline: float) -> _Run:
    """PREPARE, LAUNCH and START. Every wait is bounded by the deadline; any failure (a provenance
    refusal, a process that cannot be created, a lane that cannot start) tears the run down at once,
    the child killed and reaped and the writer released, before the refusal propagates (review r8)."""
    run = _Run(deadline)
    try:
        launch = _Lane(_launch, run, header, chunks)
        if not launch.finished(_remaining(deadline)):
            raise _timeout()
        if launch.error is not None:
            raise launch.error
        if run.process is None or launch.result is None:
            raise OSError('the worker process was not created')
        run.io = _WorkerIO(run.process, launch.result, candidate_follows, stdout_limit, deadline)
        run.io.start()
        return run
    except BaseException as exc:
        run.stop()
        if not isinstance(exc, Exception) or isinstance(exc, CoupledSolveError):
            raise
        if isinstance(exc, MemoryError):
            raise _built(lambda: _refuse('COUPLED_TOO_LARGE', 'memory_error',
                                         'out of memory starting the certificate worker')) from exc
        detail = 'start_failed' if run.process is None else 'io'
        raise _built(lambda: _worker_refusal(detail, f'it could not be started ({_type_name(exc)})')) from exc


# --- Wire format (docs/SOLID_COUPLING.md, "Certificate process") -----------------------------------------

@dataclass(frozen=True)
class ExportMap:
    """A linear recovery map M (CSR rows over the solution vector) the worker evaluates exactly with
    its bound: ``coefficients`` is float64 (exact binary64 values) or a tuple of Fractions;
    ``offsets`` (float64, one per row) or None."""

    indptr: np.ndarray
    columns: np.ndarray
    coefficients: Any
    offsets: Optional[np.ndarray] = None


def _constants() -> dict:
    return {'cap': COUPLED_MAX_EQUATIONS, 'condition_limit_bits': _bits_of(CONDITION_LIMIT) & 0xFFFF_FFFF_FFFF_FFFF,
            'backward': _m.encode_fraction(BACKWARD_TOLERANCE), 'forward': _m.encode_fraction(FORWARD_TOLERANCE),
            'max_sweeps': _m.EQUILIBRATION_MAX_SWEEPS, 'svd_constant_min': _m.SVD_ERROR_CONSTANT_MIN,
            'dense_copies': _m.DENSE_COPIES, 'sqrt_bits': _m.SQRT_BITS}


def _i8(values: Any) -> bytes:
    return np.ascontiguousarray(values, dtype='<i8').tobytes()


def _f8(values: Any) -> bytes:
    return np.ascontiguousarray(values, dtype='<f8').tobytes()


def _export_limit(message: str) -> CoupledSolveError:
    return _refuse('COUPLED_UNCERTIFIED', 'export_limit', f'the certificate request exceeds its limits: {message}')


def _request_parts(system: sparse.csr_matrix, b: Optional[np.ndarray], *, mode: str, raw_load_zero: bool,
                   memory_limit: Optional[int], exports: Sequence[ExportMap]) -> tuple:
    """(header, byte chunks after the threadpoolctl section): validated, with copies taken now (later
    changes to the arrays do not reach it). Review r8: every request limit of the child (the same
    ``_coupled_certificate_math`` constants) is checked here, before any child starts."""
    import json
    size = int(system.shape[0])
    if len(exports) > _m.MAX_EXPORTS:
        raise _export_limit(f'{len(exports)} export maps (at most {_m.MAX_EXPORTS})')
    items, blobs = [], []
    for export in exports:
        if type(export) is not ExportMap:
            raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', 'exports must be ExportMap objects')
        indptr, columns = export.indptr, export.columns
        if type(indptr) is not np.ndarray or type(columns) is not np.ndarray or indptr.ndim != 1 \
                or columns.ndim != 1 or indptr.dtype.kind != 'i' or columns.dtype.kind != 'i' or indptr.size < 1:
            raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', 'an export map has invalid index arrays')
        rows, count = int(indptr.size) - 1, int(columns.size)
        if rows > _m.MAX_EXPORT_ENTRIES or count > _m.MAX_EXPORT_ENTRIES:
            raise _export_limit(f'an export map has {rows} rows and {count} entries '
                                f'(at most {_m.MAX_EXPORT_ENTRIES} each)')
        reason = _m.export_structure_reason(indptr, columns, size)
        if reason is not None:                   # incl. a repeated column in a row (review r7 P1)
            raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', f'an export map is invalid ({reason})')
        coefficients = export.coefficients
        if type(coefficients) is np.ndarray and coefficients.dtype == np.float64 and coefficients.shape == (count,):
            kind, text = 'float', b''
            data = _f8(coefficients)
        elif type(coefficients) is tuple and len(coefficients) == count \
                and all(type(c) is Fraction for c in coefficients):
            kind, data = 'rational', b''
            text = ','.join(_m.encode_fraction(c) for c in coefficients).encode('ascii')
            if len(text) > _m.MAX_EXPORT_TEXT_BYTES:
                raise _export_limit(f'an export map has {len(text)} coefficient bytes '
                                    f'(at most {_m.MAX_EXPORT_TEXT_BYTES})')
        else:
            raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', 'an export map has invalid coefficients')
        offsets = export.offsets
        if offsets is not None and not (type(offsets) is np.ndarray and offsets.dtype == np.float64
                                        and offsets.shape == (rows,)):
            raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', 'an export map has invalid offsets')
        items.append({'rows': rows, 'nnz': count, 'coefficients': kind, 'text_bytes': len(text),
                      'offsets': offsets is not None})
        blobs += [_i8(indptr), _i8(columns), data if kind == 'float' else text]
        if offsets is not None:
            blobs.append(_f8(offsets))
    candidate = 'none' if mode == 'certificate' else ('zero' if raw_load_zero else 'follows')
    header = {'version': _worker.REQUEST_VERSION, 'mode': mode, 'n': size, 'nnz': int(system.nnz),
              'raw_load_zero': raw_load_zero, 'candidate': candidate, 'memory_limit': memory_limit,
              'constants': _constants(), 'exports': items}
    if len(_header_body(header, _m.MAX_THREADPOOLCTL_BYTES)) > _m.MAX_HEADER_BYTES:
        raise _export_limit(f'the request header exceeds {_m.MAX_HEADER_BYTES} bytes')
    chunks = [_i8(system.indptr), _i8(system.indices), _f8(system.data)]
    if b is not None:
        chunks.append(_f8(b))
    return header, chunks + blobs


def _header_body(header: dict, threadpoolctl_bytes: int) -> bytes:
    import json
    return json.dumps(dict(header, threadpoolctl_bytes=threadpoolctl_bytes), sort_keys=True,
                      separators=(',', ':')).encode('ascii')


def _encode_request(header: dict, chunks: list, runtime: _Runtime) -> list:
    """The request bytes: magic, header, threadpoolctl's RECORD-checked bytes (review r8 P1), arrays."""
    body = _header_body(header, len(runtime.threadpoolctl_source))
    return [_worker.REQUEST_MAGIC + struct.pack('<Q', len(body)) + body, runtime.threadpoolctl_source] + chunks


def _request(system: sparse.csr_matrix, b: Optional[np.ndarray], *, mode: str, raw_load_zero: bool,
             memory_limit: Optional[int], exports: Sequence[ExportMap], runtime: Optional[_Runtime] = None) -> list:
    """The whole request as byte chunks (tests and evidence scripts; the solve builds it on its launch
    lane). Without ``runtime`` the trusted modules are resolved here."""
    header, chunks = _request_parts(system, b, mode=mode, raw_load_zero=raw_load_zero, memory_limit=memory_limit,
                                    exports=exports)
    return _encode_request(header, chunks, _trusted_runtime() if runtime is None else runtime)


class _Malformed(Exception):
    """The worker's reply does not match the schema."""


def _expect(condition: bool) -> None:
    if not condition:
        raise _Malformed()


def _int(value: Any, low: int, high: int) -> int:
    _expect(type(value) is int and low <= value <= high)
    return value


def _f64(value: Any) -> float:
    _expect(type(value) is int and 0 <= value < 1 << 64)
    return _float_from_bits(value)


def _finite_bits(value: Any) -> float:
    _expect(type(value) is int and 0 <= value < 1 << 64 and (value & 0x7FFF_FFFF_FFFF_FFFF) < _INF_BITS)
    return _float_from_bits(value)


def _non_negative_bits(value: Any) -> float:
    """A finite binary64 with the sign bit clear (+0.0 included, -0.0 not) (review r7)."""
    _expect(type(value) is int and 0 <= value < _INF_BITS)
    return _float_from_bits(value)


def _fraction(value: Any) -> Fraction:
    try:
        return _m.decode_fraction(value)
    except ValueError:
        raise _Malformed() from None


def _text(value: Any) -> str:
    _expect(type(value) is str and value.isascii() and value.isprintable() and len(value) <= 400)
    return value


_NUMERICAL_FAILURES = frozenset(f'numerical_failure:{name}' for name in
                                ('LinAlgError', 'ValueError', 'FloatingPointError', 'OverflowError'))
_CERTIFICATE_REASONS = frozenset(('nonfinite_entry', 'zero_row', 'equilibration_not_converged',
                                  'nonfinite_singular_values', 'invalid_singular_values', 'svd_error_bound_invalid',
                                  'singular_within_svd_error', 'condition_above_limit')) | _NUMERICAL_FAILURES
# Review r7 P2: every refusal the child can send, per mode, with the diagnostic keys it carries and
# whether its certificate record is accepted (None: no record). Anything else is malformed.
_BEFORE = frozenset()
_REFUSALS_BEFORE_CERTIFICATE = {
    ('COUPLED_TOO_LARGE', 'equation_cap'): {(_BEFORE, None)},
    ('COUPLED_TOO_LARGE', 'memory_limit'): {(_BEFORE, None)},
    ('COUPLED_UNCERTIFIED', 'invalid_system'): {(_BEFORE, None)},
    ('COUPLED_UNCERTIFIED', 'nonfinite_entry'): {(_BEFORE, None)},
}
_CERT = frozenset(('certificate',))
_REFUSALS = {
    'certificate': _REFUSALS_BEFORE_CERTIFICATE,
    'solve': {
        **_REFUSALS_BEFORE_CERTIFICATE,
        ('COUPLED_UNCERTIFIED', 'nonfinite_entry'): {(_BEFORE, None), (_CERT, False)},
        ('COUPLED_UNCERTIFIED', 'zero_load_inconsistent'): {(_BEFORE, None)},
        ('COUPLED_FORMATION_LOSS', 'load_stored_as_zero'): {(_BEFORE, None)},
        **{('COUPLED_UNCERTIFIED', r): {(_CERT, False)} for r in _CERTIFICATE_REASONS if r != 'nonfinite_entry'},
        ('COUPLED_UNCERTIFIED', 'singular_within_svd_error'): {(_CERT, False), (_CERT, True)},
        ('COUPLED_UNCERTIFIED', 'candidate_failed'): {(_CERT, True)},
        ('COUPLED_UNCERTIFIED', 'backward_error_above_limit'): {(_CERT | {'backward'}, True)},
        ('COUPLED_UNCERTIFIED', 'forward_error_above_limit'): {(_CERT | {'forward_error_up'}, True)},
        ('COUPLED_UNCERTIFIED', 'internal_residual_inconsistency'): {(_BEFORE, None)},
        ('COUPLED_UNCERTIFIED', 'bound_not_finite'): {(_BEFORE, None)},
        ('COUPLED_UNCERTIFIED', 'invalid_export'): {(_BEFORE, None)},
    },
}
_CERTIFICATE_FLOATS = ('condition_number', 'condition_upper_bound')
_CERTIFICATE_REQUIRED = frozenset(('matrix', 'size', 'method', 'condition_limit', 'condition_number',
                                   'condition_upper_bound', 'equilibration_sweeps', 'accepted', 'reason'))


_EMPTY_CERTIFICATE_BITS = {k: (_bits_of(float(v)) & 0xFFFF_FFFF_FFFF_FFFF if isinstance(v, float) else v)
                           for k, v in _m.empty_certificate().items()}


def _decode_certificate(value: Any, size: int) -> dict:
    """The certificate record, key by key against the schema; floats arrive as their bit patterns.
    The 0 x 0 system has exactly the empty record (review r7: decided in the child)."""
    _expect(type(value) is dict)
    if size == 0:
        _expect(set(value) == set(_EMPTY_CERTIFICATE_BITS)
                and all(type(value[k]) is type(v) and value[k] == v for k, v in _EMPTY_CERTIFICATE_BITS.items()))
        return _m.empty_certificate()
    keys = set(value)
    _expect(_CERTIFICATE_REQUIRED <= keys)
    sigma = {'sigma_max', 'sigma_min'} & keys
    svd = {'svd_error_constant', 'svd_error_alpha', 'svd_error_bound'} & keys
    _expect(keys == _CERTIFICATE_REQUIRED | sigma | svd and len(sigma) in (0, 2) and len(svd) in (0, 3))
    _expect(not svd or sigma)
    _expect(value['matrix'] == 'kkt_stored' and type(value['matrix']) is str)
    _expect(value['method'] == 'power_of_two_ruiz_dense_svd' and type(value['method']) is str)
    out = {'matrix': 'kkt_stored', 'size': _int(value['size'], size, size), 'method': 'power_of_two_ruiz_dense_svd'}
    _expect(value['condition_limit'] == _bits_of(CONDITION_LIMIT) & 0xFFFF_FFFF_FFFF_FFFF)
    out['condition_limit'] = CONDITION_LIMIT
    for key in _CERTIFICATE_FLOATS:
        out[key] = _f64(value[key])
    out['equilibration_sweeps'] = _int(value['equilibration_sweeps'], 0, _m.EQUILIBRATION_MAX_SWEEPS)
    accepted, reason = value['accepted'], value['reason']
    _expect(type(accepted) is bool and (reason is None) == accepted)
    _expect(reason is None or (type(reason) is str and reason in _CERTIFICATE_REASONS))
    out['accepted'], out['reason'] = accepted, reason
    if sigma:
        out['sigma_max'], out['sigma_min'] = _finite_bits(value['sigma_max']), _finite_bits(value['sigma_min'])
    if svd:
        constant = value['svd_error_constant']
        _expect(constant is None or (type(constant) is int and constant == max(_m.SVD_ERROR_CONSTANT_MIN, size)))
        out['svd_error_constant'] = constant
        alpha = value['svd_error_alpha']
        out['svd_error_alpha'] = None if alpha is None else _f64(alpha)
        out['svd_error_bound'] = _f64(value['svd_error_bound'])
    if accepted:
        # An accepted record has the full set with every field non-null, finite and non-negative, and an
        # upper bound at most the limit (bit order of non-negative binary64 values; no arithmetic).
        _expect(sigma and svd and value['svd_error_constant'] is not None and value['svd_error_alpha'] is not None)
        for key in ('condition_number', 'condition_upper_bound', 'sigma_max', 'sigma_min', 'svd_error_alpha',
                    'svd_error_bound'):
            _non_negative_bits(value[key])
        upper = value['condition_upper_bound']
        _expect(upper <= _bits_of(CONDITION_LIMIT) and (upper >> 63) == 0)
    return out


def _decode_backward(value: Any, size: int, accepted: bool) -> dict:
    _expect(type(value) is dict and set(value) == {'accepted', 'failing_rows', 'failing_count', 'exempt_rows',
                                                    'max_ratio', 'reference'})
    _expect(value['accepted'] is accepted and value['reference'] == 'row_normwise_od7'
            and type(value['reference']) is str)
    rows = value['failing_rows']
    _expect(type(rows) is list and len(rows) <= 20)
    failing = [_int(r, 0, size - 1) for r in rows]
    count = _int(value['failing_count'], len(failing), size)
    _expect((count == 0) == accepted and (accepted or failing))
    return {'accepted': accepted, 'failing_rows': failing, 'failing_count': count,
            'exempt_rows': _int(value['exempt_rows'], 0, size), 'max_ratio': _non_negative_bits(value['max_ratio']),
            'reference': 'row_normwise_od7'}


def _decode(code: Any, output: bytes, size: int, mode: str, export_rows: Sequence[int]) -> dict:
    """The worker's reply against the exact schema (owner decision OD14): magic, length, JSON keys and
    types, the section list and byte lengths, exit code per verdict. Raises _Malformed otherwise."""
    import json
    _expect(len(output) >= 16 and output[:8] == _worker.REPLY_MAGIC)
    length = struct.unpack('<Q', output[8:16])[0]
    _expect(length <= _REPLY_JSON_LIMIT and 16 + length <= len(output))

    def no_float(text: str) -> None:
        raise _Malformed()

    def unique(pairs: list) -> dict:            # review r7: a repeated key is malformed, not collapsed
        out = {}
        for key, item in pairs:
            _expect(key not in out)
            out[key] = item
        return out
    try:
        reply = json.loads(output[16:16 + length].decode('ascii'), parse_float=no_float, parse_constant=no_float,
                           object_pairs_hook=unique)
    except (UnicodeDecodeError, ValueError):
        raise _Malformed() from None
    _expect(type(reply) is dict and type(reply.get('verdict')) is str and type(reply.get('sections')) is list)
    verdict, sections = reply['verdict'], reply['sections']
    for item in sections:
        _expect(type(item) is list and len(item) == 2 and type(item[0]) is str and type(item[1]) is int)
    names = [item[0] for item in sections]
    payload = output[16 + length:]
    _expect(sum(item[1] for item in sections) == len(payload))
    blobs, offset = {}, 0
    for name, count in sections:
        blobs[name] = payload[offset:offset + count]
        offset += count
    out: dict = {'verdict': verdict}
    if verdict == 'environment':
        _expect(code == _worker.EXIT_ENVIRONMENT and set(reply) == {'verdict', 'reason', 'sections'} and not sections)
        _expect(type(reply['reason']) is str and reply['reason'] in _worker.ENVIRONMENT_REASONS)
        out['reason'] = reply['reason']
        return out
    if verdict == 'memory':
        _expect(code == _worker.EXIT_MEMORY and set(reply) == {'verdict', 'sections'} and not sections)
        return out
    _expect(code == _worker.EXIT_OK)
    if verdict == 'refuse':
        _expect({'verdict', 'code', 'reason', 'message', 'sections'} <= set(reply) <= {
            'verdict', 'code', 'reason', 'message', 'sections', 'certificate', 'backward', 'forward_error_up'})
        _expect(not sections and type(reply['code']) is str and type(reply['reason']) is str)
        allowed = _REFUSALS[mode].get((reply['code'], reply['reason']))
        _expect(allowed is not None)
        extra = frozenset(set(reply) - {'verdict', 'code', 'reason', 'message', 'sections'})
        out.update(code=reply['code'], reason=reply['reason'], message=_text(reply['message']))
        diagnostics = {}
        record = None
        if 'certificate' in reply:
            diagnostics['certificate'] = _decode_certificate(reply['certificate'], size)
            record = diagnostics['certificate']['accepted']
            _expect(record or diagnostics['certificate']['reason'] == reply['reason'])
        _expect((extra, record) in allowed)
        if 'backward' in reply:
            diagnostics['backward'] = _decode_backward(reply['backward'], size, False)
        if 'forward_error_up' in reply:
            diagnostics['forward_error'] = _non_negative_bits(reply['forward_error_up'])
        out['diagnostics'] = diagnostics
        return out
    if verdict == 'certificate':
        _expect(mode == 'certificate' and set(reply) == {'verdict', 'certificate', 'sections'})
        out['certificate'] = _decode_certificate(reply['certificate'], size)
        _expect(names in ([], ['shifts']) and (names or not out['certificate']['accepted']))
        out['shifts'] = _shifts(blobs['shifts'], size) if names else None
        return out
    if verdict == 'no_candidate':
        _expect(mode == 'solve' and set(reply) == {'verdict', 'certificate', 'sections'} and not sections)
        out['certificate'] = _decode_certificate(reply['certificate'], size)
        _expect(out['certificate']['accepted'])
        return out
    _expect(verdict == 'accept' and mode == 'solve')
    _expect(set(reply) == {'verdict', 'certificate', 'backward', 'forward_error', 'forward_error_up', 'sigma_lower',
                           'sections'})
    out['certificate'] = _decode_certificate(reply['certificate'], size)
    _expect(out['certificate']['accepted'])
    out['backward'] = _decode_backward(reply['backward'], size, True)
    out['forward_error'] = _fraction(reply['forward_error'])
    out['forward_error_up'] = _non_negative_bits(reply['forward_error_up'])
    out['sigma_lower'] = _fraction(reply['sigma_lower'])
    _expect(out['forward_error'] >= 0 and out['sigma_lower'] > 0)
    expected = [['shifts', 8 * size], ['x', 8 * size]]
    for k, rows in enumerate(export_rows):
        expected += [[f'values{k}', 8 * rows], [f'bounds{k}', 8 * rows]]
    _expect(sections == expected)
    out['shifts'] = _shifts(blobs['shifts'], size)
    out['x'] = _finite_array(blobs['x'], size)
    exports = []
    for k, rows in enumerate(export_rows):
        values = _finite_array(blobs[f'values{k}'], rows)
        bounds = _finite_array(blobs[f'bounds{k}'], rows)
        _expect(not bool((bounds.view(np.int64) < 0).any()))
        exports.append((values, bounds))
    out['exports'] = tuple(exports)
    return out


def _shifts(blob: bytes, size: int) -> np.ndarray:
    _expect(len(blob) == 8 * size)
    shifts = np.frombuffer(blob, dtype='<i8').astype(np.int64, copy=True)
    # Two comparisons, no np.abs: abs(-2^63) overflows to -2^63 and passed (review r7).
    _expect(not bool(((shifts < -_SHIFT_LIMIT) | (shifts > _SHIFT_LIMIT)).any()))
    return shifts


def _finite_array(blob: bytes, count: int) -> np.ndarray:
    _expect(len(blob) == 8 * count)
    values = np.frombuffer(blob, dtype='<f8').astype(np.float64, copy=True)
    _expect(bool(((values.view(np.int64) & _ABS_MASK) < _INF_BITS).all()))
    return values


def _verdict(run: _Run, cancel: Optional[Callable[[], None]], size: int, mode: str,
             export_rows: Sequence[int]) -> tuple:
    """Wait for the worker and decode its reply: returns (decoded, report) or raises the refusal."""
    code, output, report = run.result(cancel)
    try:
        decoded = _decode(code, output, size, mode, export_rows)
    except _Malformed:
        raise _worker_refusal('exit' if code not in (_worker.EXIT_OK, _worker.EXIT_ENVIRONMENT, _worker.EXIT_MEMORY)
                              else 'bad_output', f'its reply is malformed (exit code {code})', exit_code=code,
                              output_bytes=len(output), worker_report=report) from None
    if decoded['verdict'] == 'environment':
        reason = decoded['reason']
        if reason in _SVD_WORKER_ENVIRONMENT_REASONS:
            raise _refuse('COUPLED_UNCERTIFIED', reason,
                          f'the certificate worker could not verify its environment ({reason})', worker_report=report)
        raise _worker_refusal('environment', f'its environment check failed ({reason})', worker_report=report)
    if decoded['verdict'] == 'memory':
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory in the certificate worker', size=size)
    if decoded['verdict'] == 'refuse':
        raise CoupledSolveError(decoded['message'], code=decoded['code'], reason=decoded['reason'],
                                diagnostics=dict(decoded['diagnostics'], worker_report=report))
    return decoded, report


def _run_worker(system: sparse.csr_matrix, b: Optional[np.ndarray], *, mode: str, raw_load_zero: bool,
                memory_limit: Optional[int], exports: Sequence[ExportMap] = (),
                cancel: Optional[Callable[[], None]] = None,
                candidate: Optional[Callable[[], Optional[np.ndarray]]] = None) -> tuple:
    """One certificate run: start the worker, compute the candidate here meanwhile (``candidate()``,
    None when it failed), send it, wait and decode. Returns (decoded, report). The worker is stopped
    on every path."""
    import time
    size = int(system.shape[0])
    try:
        header, chunks = _request_parts(system, b, mode=mode, raw_load_zero=raw_load_zero,
                                        memory_limit=memory_limit, exports=exports)
    except MemoryError as exc:
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory building the certificate request') from exc
    export_rows = [int(e.indptr.size) - 1 for e in exports]
    limit = 16 + _REPLY_JSON_LIMIT + 16 * size + 16 * sum(export_rows) + 1
    follows = mode == 'solve' and not raw_load_zero
    # The one deadline (review r8): before any child-related I/O, the provenance read included.
    deadline = time.monotonic() + _SVD_WORKER_TIMEOUT_SECONDS
    run = _start_worker(header, chunks, follows, limit, deadline)
    try:
        if follows:
            run.post_candidate(candidate())
        return _verdict(run, cancel, size, mode, export_rows)
    finally:
        run.stop()


# --- The certificate and the certified solve ------------------------------------------------------------

def _size_checks(system: sparse.csr_matrix, memory_limit_bytes: Optional[int]) -> int:
    size = int(system.shape[0])
    if size > COUPLED_MAX_EQUATIONS:
        raise _refuse('COUPLED_TOO_LARGE', 'equation_cap',
                      f'coupled solve certificate is dense and capped at {COUPLED_MAX_EQUATIONS} equations '
                      f'(KKT system has {size})', size=size, cap=COUPLED_MAX_EQUATIONS)
    if memory_limit_bytes is not None and _dense_bytes(size) > int(memory_limit_bytes):
        raise _refuse('COUPLED_TOO_LARGE', 'memory_limit',
                      f'the dense certificate needs about {_dense_bytes(size)} bytes, above the '
                      f'memory limit {int(memory_limit_bytes)}', size=size)
    return size


def _certify_solve_matrix(system: Any, *, memory_limit_bytes: Optional[int] = None,
                          cancel: Optional[Callable[[], None]] = None) -> dict:
    """Condition certificate of the stored system S alone, computed by the worker (its 'certificate'
    mode; used by the tests and the evidence scripts). Refuses COUPLED_TOO_LARGE above
    COUPLED_MAX_EQUATIONS, above the memory limit or on MemoryError. Returns the decision numbers,
    including the exact power-of-two shifts ``shifts`` (D = diag(2^shifts)), with ``accepted``."""
    system = _require_system(system)
    _size_checks(system, memory_limit_bytes)
    _require_fp_environment()                 # N = 0 too: the child decides it (review r7)
    decoded, _report = _run_worker(system, None, mode='certificate', raw_load_zero=False,
                                   memory_limit=memory_limit_bytes, cancel=cancel)
    return dict(decoded['certificate'], shifts=decoded['shifts'])


def _candidate(system: sparse.csr_matrix, b: np.ndarray) -> np.ndarray:
    """SuperLU candidate with up to five refinement steps on the exact residual. Carries no claim: the
    worker certifies or refuses it."""
    try:
        with np.errstate(all='ignore'):
            lu = sparse_linalg.splu(system.tocsc())
            x = lu.solve(b)
            for _ in range(_REFINEMENT_STEPS):
                if not np.isfinite(x).all():
                    break
                r = _exact_rows(system, x, b).residual_float()
                if not r.any():
                    break
                x = x + lu.solve(r)
    except Exception as exc:  # includes SuperLU's exact-singular RuntimeError
        for _ in range(_TRANSLATION_ATTEMPTS):          # review r5 #3: a hook raising while it is built
            try:
                if isinstance(exc, MemoryError):
                    refusal = _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory in the candidate solve')
                else:
                    refusal = _refuse('COUPLED_UNCERTIFIED', 'candidate_failed',
                                      f'the candidate solve failed ({_type_name(exc)})')
                break
            except Exception:  # noqa: BLE001
                continue
        else:
            raise
        raise refusal from exc
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise _refuse('COUPLED_UNCERTIFIED', 'candidate_failed', 'the candidate solve is not finite')
    return x


@dataclass(frozen=True)
class CertifiedSolve:
    """x within d_i * forward_error of the exact solution x* of S x = b (claim under A1/A2). Every
    number here was decided by the certificate worker (OD14): ``x`` is the worker's copy of the
    candidate it verified, ``exports`` the (values, bounds) of each export map."""

    x: np.ndarray
    shifts: np.ndarray
    forward_error: Fraction
    sigma_lower: Fraction
    certificate: Mapping[str, Any]
    backward: Mapping[str, Any]
    zero_load: bool
    timings: Mapping[str, float] = field(default_factory=dict)
    exports: tuple = ()
    forward_error_up: float = 0.0

    def bound(self, i: int) -> Fraction:
        return self.forward_error * (Fraction(2) ** int(self.shifts[i]))


def _poll(cancel: Optional[Callable[[], None]]) -> None:
    """Run the caller's cancellation check, then re-check the FP environment."""
    if cancel is not None:
        cancel()
    _require_fp_environment()


def _checked_clock(clock: Callable[[], Any]) -> Callable[[], Any]:
    """The caller's timing callback, followed by the FP-environment probe on every call (review r3 P2).
    The value must be an exact ``float`` or ``int``."""
    def tick():
        value = clock()
        _require_fp_environment()
        if type(value) is not float and type(value) is not int:
            raise _refuse('COUPLED_NATIVE_INVALID', 'clock_invalid', 'the clock must return a float or an int')
        return value
    return tick


def _held_candidate(system: sparse.csr_matrix, b: np.ndarray, held: dict) -> Optional[np.ndarray]:
    """The candidate, or None with its refusal kept in `held['refusal']` (raised only once the
    worker's certificate is accepted: the refusal precedence of certified_solve)."""
    try:
        return _candidate(system, b)
    except CoupledSolveError as exc:
        held['refusal'] = exc
        return None


def certified_solve(system: Any, rhs: Any, *, raw_load_zero: bool,
                    memory_limit_bytes: Optional[int] = None,
                    cancel: Optional[Callable[[], None]] = None,
                    clock: Optional[Callable[[], float]] = None,
                    exports: Sequence[ExportMap] = ()) -> CertifiedSolve:
    """Certify S, solve S x = b, and verify V1-V3 and V6 exactly; refuse with a code otherwise.

    ``raw_load_zero`` is True iff the native load, the offset u0 and every consumed pressure are
    exactly zero (the zero-load path, spec 2.4); then b must be exactly zero and x = 0. A nonzero raw
    load whose stored b is all zero refuses COUPLED_FORMATION_LOSS.

    OD14: the worker starts first and certifies S while the SuperLU candidate is computed here; the
    candidate is then sent and the worker verifies it. Refusal precedence is unchanged: a certificate
    refusal wins over a candidate failure, which wins over the V1-V3 and V6 refusals.
    """
    import time
    clock = _checked_clock(time.perf_counter if clock is None else clock)
    if type(raw_load_zero) is not bool:
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', 'raw_load_zero must be a bool')
    if memory_limit_bytes is not None and type(memory_limit_bytes) is not int:
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', 'memory_limit_bytes must be None or an int')
    if type(exports) is not tuple and type(exports) is not list:
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', 'exports must be a tuple or list')
    exports = tuple(exports)
    timings: dict[str, float] = {}
    system = _require_system(system)
    b = rhs
    if type(b) is not np.ndarray or b.dtype != np.float64 or b.shape != (system.shape[0],):
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_rhs', 'the right-hand side must be float64 of length N')
    if not np.isfinite(system.data).all() or not np.isfinite(b).all():
        raise _refuse('COUPLED_UNCERTIFIED', 'nonfinite_entry', 'the coupled system is not finite')
    _require_fp_environment()
    b_zero = not _any_nonzero(b)
    if raw_load_zero and not b_zero:
        raise _refuse('COUPLED_UNCERTIFIED', 'zero_load_inconsistent',
                      'the zero-load path requires an exactly zero stored right-hand side')
    if not raw_load_zero and b_zero:
        raise _refuse('COUPLED_FORMATION_LOSS', 'load_stored_as_zero',
                      'a nonzero raw load was stored as an all-zero right-hand side')
    _poll(cancel)
    _size_checks(system, memory_limit_bytes)
    # N = 0 goes through the child like every other size (review r7): its recovery maps (offset-only
    # rows) are evaluated and bounded there, never returned empty here.
    held: dict = {}

    def candidate() -> Optional[np.ndarray]:
        """The SuperLU candidate, computed while the worker certifies S; a candidate refusal is held
        back until the worker's certificate verdict is known (precedence above)."""
        _poll(cancel)
        t0 = clock()
        x = _held_candidate(system, b, held)
        timings['candidate'] = clock() - t0
        _poll(cancel)
        return x

    t0 = clock()
    decoded, _report = _run_worker(system, b, mode='solve', raw_load_zero=raw_load_zero,
                                  memory_limit=memory_limit_bytes, exports=exports, cancel=cancel,
                                  candidate=candidate)
    timings['certificate'] = clock() - t0
    if raw_load_zero:
        timings['candidate'] = 0.0
    # The worker's own stage times are in its report (diagnostics); timings use the caller's clock only.

    if decoded['verdict'] == 'no_candidate':
        refusal = held.get('refusal')
        if refusal is None:
            raise _worker_refusal('bad_output', 'it reported no candidate for a sent candidate')
        raise refusal
    if 'refusal' in held:
        raise _worker_refusal('bad_output', 'it verified a candidate that was not sent')
    # Review r1 #8 (class 7): a cancellation after the verification never returns success.
    _poll(cancel)
    return CertifiedSolve(decoded['x'], decoded['shifts'], decoded['forward_error'], decoded['sigma_lower'],
                          decoded['certificate'], decoded['backward'], raw_load_zero, timings, decoded['exports'],
                          decoded['forward_error_up'])
