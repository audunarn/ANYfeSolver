"""Dense certificate and exact verification of the coupled shell-solid KKT solve (private).

Ported from the PR #67 solid port (b94c9179f, ``solids.py``) with the cap raised to 5000 equations,
the ``COUPLED_*`` codes and no row normalisation: the certified matrix is exactly the stored system
``S``. Added here: the exact inverse-norm bound (spec section 2.2), the exact integer residual, the
backward checks V1-V3 and the forward-error decision V6 (section 2.3), the zero-load path (2.4) and
the SuperLU candidate (2.5). See ``docs/SOLID_COUPLING.md``.

The only premises are (A1) LAPACK's SVD backward error and (A2) one rounding per equilibrated entry,
inexact only for a subnormal result. Every other decision is exact rational arithmetic.

Not exported from :mod:`anyfesolver`; the sole caller is :mod:`anyfesolver.solid_coupling`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import functools
import math
import os
import struct
import sys
from typing import Any, Callable, Mapping, Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from . import _coupled_svd_worker as _worker
from ._coupled_svd_worker import (  # noqa: F401  (re-exported for solid_coupling and the tests)
    _bits_of,
    _float_bits,
    _float_from_bits,
    _fp_environment_reason,
    _nearest_rounding,
    _subnormals_preserved,
)

__all__: list[str] = []


# --- Exact float <-> rational conversion by IEEE bit extraction (review r1 #1, class 1) ---------------
# frexp, float.as_integer_ratio, Fraction(float), float(Fraction), ldexp, comparisons and any() all
# change their result for subnormals when the thread flushes denormals (FTZ/DAZ). These helpers read
# and write the IEEE binary64 bit pattern with integer operations only, so their results do not
# depend on the floating-point environment.

_SIGN_BIT = 1 << 63
_FRACTION_BITS = (1 << 52) - 1
_INF_BITS = 0x7FF << 52
_ABS_MASK = np.int64(0x7FFF_FFFF_FFFF_FFFF)
# _float_from_bits, _bits_of and _float_bits come from the SVD worker module (one source for the
# parent and the child, owner decision OD13).


def _nonzero_mask(values: Any) -> np.ndarray:
    """values != 0 decided on the bit pattern (+-0 are zero; a subnormal is nonzero)."""
    return (_float_bits(values) & _ABS_MASK) != 0


def _any_nonzero(values: Any) -> bool:
    return bool(_nonzero_mask(values).any())


def _decompose(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Exact (integer mantissa, exponent) with value = m * 2^e; m is a signed integer below 2^53.

    Integer bit extraction (no frexp): a subnormal keeps its bits under FTZ/DAZ. Callers pass
    finite float64 arrays."""
    bits = _float_bits(values)
    biased = (bits >> 52) & 0x7FF
    fraction = bits & _FRACTION_BITS
    normal = biased != 0
    mantissa = np.where(normal, fraction | (1 << 52), fraction)
    exponent = np.where(normal, biased - 1075, -1074)
    mantissa = np.where(bits < 0, -mantissa, mantissa)
    return mantissa.astype(np.int64), exponent.astype(np.int64)


def _exact(value: Any) -> Fraction:
    """The exact rational value of one finite binary64 value, by bit extraction."""
    bits = _bits_of(float(value))
    biased = (bits >> 52) & 0x7FF
    if biased == 0x7FF:
        raise ValueError('a non-finite value has no exact rational value')
    mantissa = (bits & _FRACTION_BITS) | (1 << 52) if biased else bits & _FRACTION_BITS
    exponent = biased - 1075 if biased else -1074
    if bits < 0:
        mantissa = -mantissa
    return _scaled(mantissa, exponent)


def _ratio_to_float(numerator: int, denominator: int, *, up: bool = False) -> float:
    """numerator / denominator (denominator > 0) as binary64, rounded to nearest-even or, with
    ``up``, toward +inf; integer arithmetic and bit assembly only (no floating-point operation)."""
    if numerator == 0:
        return 0.0
    negative = numerator < 0
    n = -numerator if negative else numerator
    k = n.bit_length() - denominator.bit_length()
    if (n < (denominator << k)) if k >= 0 else ((n << -k) < denominator):
        k -= 1                                    # now 2^k <= n / d < 2^(k+1)
    unit = max(k - 52, -1074)                     # the exponent of one ulp at this magnitude
    if unit >= 0:
        scale = denominator << unit
        q, r = divmod(n, scale)
    else:
        scale = denominator
        q, r = divmod(n << -unit, denominator)
    if r:
        if up:
            if not negative:
                q += 1
        elif 2 * r > scale or (2 * r == scale and q & 1):
            q += 1
    bits = ((unit + 1074) << 52) + q              # continuous across the subnormal/normal boundary
    if bits >= _INF_BITS:
        if negative and up:
            bits = _INF_BITS - 1                  # toward +inf, a negative overflow stays finite
        else:
            return -math.inf if negative else math.inf
    return _float_from_bits(bits | _SIGN_BIT if negative else bits)


def _to_float(value: Fraction) -> float:
    """value rounded to nearest-even binary64 (independent of the FP environment)."""
    value = Fraction(value)
    return _ratio_to_float(value.numerator, value.denominator)


def _out_up(value: Fraction) -> float:
    """The smallest binary64 >= value, or inf (independent of the FP environment)."""
    value = Fraction(value)
    return _ratio_to_float(value.numerator, value.denominator, up=True)


# Review r5 #3: a refusal is built inside a bounded retry. A trace or profile function that raises is
# unset by the interpreter after that one exception, so a hook raising while a refusal is built (on
# the call or return of a helper) costs one attempt, not the refusal's code.
_TRANSLATION_ATTEMPTS = 4


def _require_fp_environment() -> None:
    """Gradual underflow and round-to-nearest-even on this thread, else a coded refusal.

    Review r5 #3: the probe itself never raises, so any exception from inside it was raised by caller
    code running there (a trace, profile or audit hook), including a forged CoupledSolveError; it
    refuses ``COUPLED_UNCERTIFIED`` (``fp_probe``): the environment is not verified."""
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


# --- The certificate SVD in a child process (owner decision OD13, 2026-11-20) --------------------------
# The in-process single-thread window could not be closed against same-process hooks and threads
# (review r5). The SVD now runs in a short-lived child interpreter (_coupled_svd_worker): started from
# sys.executable with -I -S and a minimal environment, it verifies its own environment (one BLAS
# thread, round-to-nearest, gradual underflow) before and after the call and returns the singular
# values only. Equilibration, the kappa bound, delta and the exact residual stay in this process.

_SVD_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_coupled_svd_worker.py')
# Hard bound on one worker run (start-up, transfer and the SVD; N = 5000 takes about 30 s on one thread).
_SVD_WORKER_TIMEOUT_SECONDS = 600.0
# Interval of the cancellation polls while the worker runs.
_SVD_WORKER_POLL_SECONDS = 0.05
# Longest wait for the transfer threads once the worker has exited.
_SVD_WORKER_IO_SECONDS = 60.0
_SVD_WORKER_STDERR_LIMIT = 1 << 16
_SVD_WORKER_ENVIRONMENT_REASONS = ('blas_threads', 'subnormal_flush', 'rounding_mode')


def _svd_worker_command() -> list:
    """argv of the worker: this interpreter, isolated (-I: no PYTHON* variables, no user site, no script
    or working directory on sys.path) and without site (-S: no .pth, sitecustomize or usercustomize),
    then the worker script, NumPy's package directory and the sys.path entries NumPy and threadpoolctl
    need (the directory that holds NumPy; threadpoolctl's, when it is loaded from elsewhere)."""
    numpy_dir = os.path.dirname(os.path.abspath(np.__file__))
    paths = [os.path.dirname(numpy_dir)]
    loaded = sys.modules.get('threadpoolctl')
    location = getattr(loaded, '__file__', None) if loaded is not None else None
    if type(location) is str and os.path.dirname(os.path.abspath(location)) not in paths:
        paths.append(os.path.dirname(os.path.abspath(location)))
    return [sys.executable, '-I', '-S', _SVD_WORKER_PATH, numpy_dir, *paths]


def _svd_worker_environment() -> dict:
    """The worker's whole environment: one BLAS, OpenMP and MKL thread, plus SYSTEMROOT on Windows
    (the interpreter needs it to start)."""
    environment = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    if sys.platform == 'win32':
        root = os.environ.get('SYSTEMROOT')
        if type(root) is str and root:
            environment['SYSTEMROOT'] = root
    return environment


def _worker_refusal(detail: str, message: str, **diagnostics: Any) -> CoupledSolveError:
    return _refuse('COUPLED_UNCERTIFIED', 'svd_worker', f'the certificate SVD worker failed: {message}',
                   worker_failure=detail, **diagnostics)


class _WorkerIO:
    """Feeds the matrix to the worker's stdin and drains its stdout and stderr on three threads, so
    the calling thread stays free to poll the cancellation and the timeout."""

    def __init__(self, process: Any, header: bytes, payload: memoryview, stdout_limit: int) -> None:
        import threading
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.overflow = False
        self.errors: list = []
        self._threads = [threading.Thread(target=self._write, args=(process.stdin, header, payload), daemon=True),
                         threading.Thread(target=self._read, args=(process.stdout, self.stdout, stdout_limit),
                                          daemon=True),
                         threading.Thread(target=self._read, args=(process.stderr, self.stderr,
                                                                  _SVD_WORKER_STDERR_LIMIT), daemon=True)]
        for thread in self._threads:
            thread.start()

    def _write(self, stream: Any, header: bytes, payload: memoryview) -> None:
        try:
            stream.write(header)
            step = 1 << 24
            for start in range(0, len(payload), step):
                stream.write(payload[start:start + step])
        except (BrokenPipeError, OSError, ValueError):
            pass                                  # the worker exited early; its exit status decides
        except BaseException as exc:  # noqa: BLE001
            self.errors.append(exc)
        finally:
            try:
                stream.close()
            except BaseException:  # noqa: BLE001
                pass

    def _read(self, stream: Any, sink: bytearray, limit: int) -> None:
        try:
            while True:
                chunk = stream.read1(1 << 16)
                if not chunk:
                    break
                room = limit - len(sink)
                if room > 0:
                    sink += chunk[:room]
                if len(chunk) > room:
                    self.overflow = self.overflow or sink is self.stdout
        except BaseException as exc:  # noqa: BLE001
            self.errors.append(exc)

    def join(self, timeout: float) -> bool:
        import time
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)


def _stop(process: Any) -> None:
    """Terminate this solve's own worker, by the handle it was started with (never by image name)."""
    try:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)
    except BaseException:  # noqa: BLE001
        pass
    for stream in (process.stdin, process.stdout, process.stderr):
        try:
            if stream is not None:
                stream.close()
        except BaseException:  # noqa: BLE001
            pass


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


def _checked_singular_values(output: bytearray, size: int) -> Optional[np.ndarray]:
    """The singular values from the worker's 'OK' record, or None when the record is malformed: exact
    byte length, the echoed size, float64 of shape (N,), every value finite and non-negative, in
    descending order. Decided on the bit patterns (no float comparison)."""
    if len(output) != 10 + 8 * size or bytes(output[:2]) != b'OK':
        return None
    if struct.unpack('<Q', bytes(output[2:10]))[0] != size:
        return None
    sigma = np.frombuffer(bytes(output[10:]), dtype='<f8').astype(np.float64, copy=True)
    if sigma.dtype != np.float64 or sigma.shape != (size,):
        return None
    bits = sigma.view(np.int64)
    magnitude = bits & _ABS_MASK
    if not bool((magnitude < _INF_BITS).all()):            # inf and nan
        return None
    if bool(((bits < 0) & (magnitude != 0)).any()):         # negative (a -0.0 is zero)
        return None
    if size > 1 and not bool((magnitude[:-1] >= magnitude[1:]).all()):
        return None
    return magnitude.view(np.float64).copy()               # -0.0 -> +0.0, exactly


def _svd_in_worker(matrix: np.ndarray, cancel: Optional[Callable[[], None]] = None):
    """Singular values of ``matrix`` computed by the worker: returns (sigma, failure, report).

    ``failure`` is None or the certificate reason 'numerical_failure:LinAlgError' (the worker's SVD did
    not converge). Refusals: the worker's own environment check -> COUPLED_UNCERTIFIED with its reason
    (blas_threads, subnormal_flush, rounding_mode); a MemoryError in the worker -> COUPLED_TOO_LARGE
    (memory_error); a start failure, crash, nonzero exit, timeout or malformed output ->
    COUPLED_UNCERTIFIED (svd_worker). While the worker runs, ``cancel`` (when given) is polled, then
    this thread's FP environment, since the callback may have changed it; whatever they raise
    propagates after this solve's worker has been terminated. This thread's FP environment does not
    reach the worker, which starts with the process defaults and verifies its own.
    ``report`` is the worker's diagnostic record (never decided on)."""
    import subprocess
    import time
    size = int(matrix.shape[0])
    try:
        payload = memoryview(np.ascontiguousarray(matrix, dtype='<f8')).cast('B')
        header = _worker.MAGIC + struct.pack('<Q', size)
        command, environment = _svd_worker_command(), _svd_worker_environment()
        flags = 0x08000000 if sys.platform == 'win32' else 0          # CREATE_NO_WINDOW
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=environment, cwd=os.path.dirname(_SVD_WORKER_PATH), close_fds=True,
                                   creationflags=flags)
    except MemoryError as exc:
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory starting the certificate SVD worker') from exc
    except Exception as exc:  # noqa: BLE001  (including an audit hook that denies process creation)
        raise _worker_refusal('start_failed', f'it could not be started ({_type_name(exc)})') from exc
    try:
        try:
            io = _WorkerIO(process, header, payload, 10 + 8 * size + 1)
        except MemoryError as exc:
            raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory feeding the certificate SVD worker') from exc
        except Exception as exc:  # noqa: BLE001
            raise _worker_refusal('io', f'its transfer could not be started ({_type_name(exc)})') from exc
        deadline = time.monotonic() + _SVD_WORKER_TIMEOUT_SECONDS
        while True:
            if cancel is not None:
                _poll(cancel)                   # caller code; anything it raises propagates (finally: stop)
            try:
                process.wait(timeout=_SVD_WORKER_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() > deadline:
                    raise _worker_refusal('timeout', f'no result within {_SVD_WORKER_TIMEOUT_SECONDS:.0f} s') from None
        complete = io.join(_SVD_WORKER_IO_SECONDS)
        code = process.returncode
    finally:
        _stop(process)
    report = _worker_report(io.stderr)
    if not complete or io.errors:
        raise _worker_refusal('io', 'its output could not be read', exit_code=code)
    output = io.stdout
    tag = bytes(output[:2])
    if code == _worker.EXIT_OK and not io.overflow:
        sigma = _checked_singular_values(output, size)
        if sigma is None:
            raise _worker_refusal('bad_output', 'its result is malformed', exit_code=code, output_bytes=len(output))
        return sigma, None, report
    if code == _worker.EXIT_ENVIRONMENT and tag == b'EN':
        reason = bytes(output[2:66]).decode('ascii', 'replace')
        if reason in _SVD_WORKER_ENVIRONMENT_REASONS and len(output) == 2 + len(reason):
            raise _refuse('COUPLED_UNCERTIFIED', reason,
                          f'the certificate SVD worker could not verify its environment ({reason})',
                          worker_report=report)
        raise _worker_refusal('environment', f'its environment check failed ({reason[:32]})', exit_code=code)
    if code == _worker.EXIT_MEMORY and bytes(output) == b'MM':
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory in the certificate SVD worker', size=size)
    if code == _worker.EXIT_LINALG and bytes(output) == b'LA':
        return None, 'numerical_failure:LinAlgError', report
    raise _worker_refusal('exit' if code != 0 else 'bad_output', f'it ended with exit code {code}', exit_code=code,
                          output_bytes=len(output))


COUPLED_MAX_EQUATIONS = 5000
# Accept only kappa_up * eps <= 1e-6 of the equilibrated stored matrix (kappa_up <= ~4.5e9).
CONDITION_LIMIT = 1e-6 / np.finfo(float).eps
_EQUILIBRATION_MAX_SWEEPS = 64
# SVD error constant: alpha = c N eps with c = max(1000, N) (assumption A1).
_SVD_ERROR_CONSTANT_MIN = 1000
_SMALLEST_SUBNORMAL = _float_from_bits(1)
_EPS = Fraction(1, 2 ** 52)
# Bytes per dense working copy of an N x N float64 matrix, times the copies held at once.
_DENSE_COPIES = 4
# Backward (V1-V3) and forward (V6) tolerances, decided exactly as rationals.
BACKWARD_TOLERANCE = Fraction(1, 10 ** 10)
FORWARD_TOLERANCE = Fraction(1, 10 ** 6)
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
    """'TypeName: message' without calling any __str__, __repr__ or __format__ (review r2 (d)): only
    arguments that are exact ``str`` objects are shown; anything else is '<not shown>'."""
    try:
        args = _EXCEPTION_ARGS.__get__(exc, BaseException)
    except Exception:  # noqa: BLE001
        args = ()
    parts = [a for a in args if type(a) is str] if type(args) is tuple else []
    text = '; '.join(parts) if parts else '<not shown>'
    return _type_name(exc) + ': ' + text


# --- Ported helpers (b94c9179f solids.py, unchanged) -------------------------------------------------

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


def _round_up(value: float) -> float:
    """An upper bound of the exact result x of one correctly rounded operation whose result is value."""
    return float(np.nextafter(value, np.inf))


def _round_down(value: float) -> float:
    return float(np.nextafter(value, -np.inf))


def _condition_upper_bound(sigma_max: float, sigma_min: float, size: int) -> dict:
    """Rigorous upper bound of kappa(A*) from computed singular values (pure function).

    beta = alpha / (1 - alpha), delta = beta * sigma_max + N 2^-1074,
    kappa(A*) <= (sigma_max + delta) / (sigma_min - delta) if sigma_min > delta,
    every operation rounded outward. Refuses for alpha >= 1/2, sigma_min <= delta or invalid inputs.

    The outward rounding (nextafter, the subnormal tau) assumes IEEE gradual underflow; the only
    caller, certified_solve, refuses first when the thread flushes subnormals (review r1 class 1).
    """
    with np.errstate(all='ignore'):
        result = {'svd_error_constant': None, 'svd_error_alpha': None, 'svd_error_bound': float('inf'),
                  'condition_upper_bound': float('inf'), 'accepted': False, 'reason': None}
        sigma_max, sigma_min, size = float(sigma_max), float(sigma_min), int(size)
        if not (size >= 1 and np.isfinite(sigma_max) and np.isfinite(sigma_min)
                and sigma_max >= sigma_min >= 0.0 and sigma_max > 0.0):
            result['reason'] = 'invalid_singular_values'
            return result
        constant = max(_SVD_ERROR_CONSTANT_MIN, size)
        eps = float(np.finfo(float).eps)
        alpha = float(constant * size) * eps  # exact: an integer below 2^53 times a power of two
        result.update(svd_error_constant=constant, svd_error_alpha=alpha)
        if not alpha < 0.5:
            result['reason'] = 'svd_error_bound_invalid'
            return result
        tau = size * _SMALLEST_SUBNORMAL  # exact: an integer multiple of 2^-1074 below 2^-1022
        beta = _round_up(alpha / _round_down(1.0 - alpha))
        delta = _round_up(_round_up(beta * sigma_max) + tau)
        result['svd_error_bound'] = delta
        denominator = _round_down(sigma_min - delta)
        if not denominator > 0.0:
            result['reason'] = 'singular_within_svd_error'
            return result
        upper = _round_up(_round_up(sigma_max + delta) / denominator)
        result['condition_upper_bound'] = upper
        result['accepted'] = bool(upper <= CONDITION_LIMIT)
        result['reason'] = None if result['accepted'] else 'condition_above_limit'
        return result


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


# --- Certificate (port step 1-6; cap 5000; stored matrix, no row normalisation) ----------------------

def _require_system(system: Any) -> sparse.csr_matrix:
    if type(system) is not sparse.csr_matrix or system.dtype != np.float64 or system.ndim != 2 \
            or system.shape[0] != system.shape[1]:
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_system',
                      'the coupled system must be a square float64 scipy.sparse.csr_matrix')
    if not system.has_canonical_format:
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_system', 'the coupled system must be canonical CSR')
    return system


def _svd_values(matrix: np.ndarray, cancel: Optional[Callable[[], None]] = None) -> np.ndarray:
    """Singular values of the dense equilibrated matrix (the one LAPACK call the certificate relies on),
    computed in the SVD worker process on one verified BLAS thread under verified round-to-nearest and
    gradual underflow (owner decision OD13). Raises _SvdNotConverged (a LinAlgError) when the worker's
    SVD did not converge."""
    sigma, failure, _report = _svd_in_worker(matrix, cancel)
    if failure is not None:
        raise _SvdNotConverged('the certificate SVD did not converge')
    return sigma


class _SvdNotConverged(np.linalg.LinAlgError):
    """The worker's SVD did not converge (certificate reason numerical_failure:LinAlgError)."""


def _dense_bytes(size: int) -> int:
    return _DENSE_COPIES * size * size * 8


def _certify_solve_matrix(system: Any, *, memory_limit_bytes: Optional[int] = None,
                          cancel: Optional[Callable[[], None]] = None) -> dict:
    """Condition certificate of the stored system S (the matrix the candidate factorises).

    Refuses COUPLED_TOO_LARGE above COUPLED_MAX_EQUATIONS, above the memory limit or on MemoryError,
    always before any dense allocation for the size checks. Returns the decision numbers, including
    the exact power-of-two shifts ``shifts`` (D = diag(2^shifts)), with ``accepted``.
    """
    system = _require_system(system)
    size = int(system.shape[0])
    if size > COUPLED_MAX_EQUATIONS:
        raise _refuse('COUPLED_TOO_LARGE', 'equation_cap',
                      f'coupled solve certificate is dense and capped at {COUPLED_MAX_EQUATIONS} equations '
                      f'(KKT system has {size})', size=size, cap=COUPLED_MAX_EQUATIONS)
    if memory_limit_bytes is not None and _dense_bytes(size) > int(memory_limit_bytes):
        raise _refuse('COUPLED_TOO_LARGE', 'memory_limit',
                      f'the dense certificate needs about {_dense_bytes(size)} bytes, above the '
                      f'memory limit {int(memory_limit_bytes)}', size=size)
    if size == 0:
        return {'matrix': 'empty', 'size': 0, 'method': 'no_free_dofs', 'condition_limit': CONDITION_LIMIT,
                'condition_number': 1.0, 'condition_upper_bound': 1.0, 'equilibration_sweeps': 0,
                'shifts': np.zeros(0, dtype=np.int64), 'accepted': True, 'reason': None}
    try:
        with np.errstate(all='ignore'):
            return _certificate_numbers(system, size, cancel)
    except MemoryError as exc:
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error',
                      'out of memory in the dense certificate', size=size) from exc


def _certificate_numbers(system: sparse.csr_matrix, size: int, cancel: Optional[Callable[[], None]] = None) -> dict:
    certificate = {'matrix': 'kkt_stored', 'size': size, 'method': 'power_of_two_ruiz_dense_svd',
                   'condition_limit': CONDITION_LIMIT, 'condition_number': float('inf'),
                   'condition_upper_bound': float('inf'), 'equilibration_sweeps': 0,
                   'shifts': None, 'accepted': False, 'reason': None}
    _require_fp_environment()            # the equilibration's ldexp is premise A2 (review r2 (a))
    try:
        original = np.array(system.toarray(), dtype=float)
        if not np.isfinite(original).all():
            certificate['reason'] = 'nonfinite_entry'
            return certificate
        A = original
        total = np.zeros(size, dtype=np.int64)
        for sweep in range(_EQUILIBRATION_MAX_SWEEPS + 1):
            largest = np.max(np.abs(A), axis=1, initial=0.0)
            if np.any(largest == 0.0):
                certificate['reason'] = 'zero_row'
                return certificate
            shift = -(np.frexp(largest)[1] // 2)
            if not shift.any():
                break
            if sweep == _EQUILIBRATION_MAX_SWEEPS:
                certificate['reason'] = 'equilibration_not_converged'
                return certificate
            # Cumulative shifts on the original entries: each entry is rounded at most once (A2).
            total = total + shift
            A = np.ldexp(original, total[:, None] + total[None, :])
            certificate['equilibration_sweeps'] = sweep + 1
        del original
        certificate['shifts'] = total
    except MemoryError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError) as exc:
        certificate['reason'] = f'numerical_failure:{type(exc).__name__}'
        return certificate
    # OD13: the SVD runs in the worker process; the caller's cancellation is polled while it runs and
    # whatever the cancellation raises propagates (outside the numerical-failure mapping above).
    try:
        sigma = _svd_values(A, cancel)
    except _SvdNotConverged:
        certificate['reason'] = 'numerical_failure:LinAlgError'
        return certificate
    del A
    sigma = np.asarray(sigma, dtype=float)
    if not sigma.size or not np.isfinite(sigma).all():
        certificate['reason'] = 'nonfinite_singular_values'
        return certificate
    sigma_max, sigma_min = float(sigma[0]), float(sigma[-1])
    certificate.update(sigma_max=sigma_max, sigma_min=sigma_min)
    if sigma_min > 0.0:
        certificate['condition_number'] = sigma_max / sigma_min  # reported only, never decided on
    certificate.update(_condition_upper_bound(sigma_max, sigma_min, size))
    _require_fp_environment()            # the outward rounding above assumed it
    return certificate


# --- Section 2.2: exact inverse-norm bound --------------------------------------------------------------

def _sigma_lower(sigma_max: float, sigma_min: float, size: int) -> Fraction:
    """sigma^- = s_N - (beta s_1 + N 2^-1074), beta = alpha / (1 - alpha), alpha = c N eps, exactly.

    Under A1/A2, sigma_N(A*) >= sigma^-, so ||A*^-1||_2 <= 1 / sigma^- whenever sigma^- > 0."""
    constant = max(_SVD_ERROR_CONSTANT_MIN, int(size))
    alpha = Fraction(constant * int(size)) * _EPS
    beta = alpha / (1 - alpha)
    return _exact(sigma_min) - (beta * _exact(sigma_max) + int(size) * Fraction(1, 2 ** 1074))


# --- Section 2.3: exact residual -------------------------------------------------------------------------

@dataclass(frozen=True)
class ExactRows:
    """Per row i, ``residual[i] * 2^exponent[i]`` is exactly r_i = b_i - (S x)_i and
    ``reference[i] * 2^exponent[i]`` exactly ref_i = sum_j |S_ij x_j| + |b_i| (Python integers)."""

    residual: list
    reference: list
    exponent: np.ndarray

    def residual_fraction(self, i: int) -> Fraction:
        return _scaled(self.residual[i], int(self.exponent[i]))

    def residual_float(self) -> np.ndarray:
        """r rounded to nearest float64 (correctly rounded: Python integer true division)."""
        return np.array([_int_to_float(r, int(e)) for r, e in zip(self.residual, self.exponent)], dtype=float)


def _scaled(value: int, exponent: int) -> Fraction:
    return Fraction(value << exponent) if exponent >= 0 else Fraction(value, 1 << -exponent)


def _int_to_float(value: int, exponent: int) -> float:
    """value * 2^exponent rounded to nearest binary64 (bit assembly, FP-environment independent)."""
    if value == 0:
        return 0.0
    if exponent >= 0:
        return _ratio_to_float(value << exponent, 1)
    return _ratio_to_float(value, 1 << -exponent)


def _exact_rows(system: sparse.csr_matrix, x: np.ndarray, b: np.ndarray) -> ExactRows:
    """r = b - S x and ref = sum |S_ij x_j| + |b_i|, each row exactly, in Python integers.

    Every float is a dyadic rational, so each product is an exact integer times a power of two; the
    terms of a row are shifted to the row's smallest exponent and summed. The result does not depend
    on the floating-point environment, the summation order or the term count."""
    size = system.shape[0]
    indptr = system.indptr
    counts = np.diff(indptr)
    ms, es = _decompose(system.data)
    mx, ex = _decompose(x)
    mb, eb = _decompose(b)
    cols = system.indices
    product = ms.astype(object) * mx[cols].astype(object)
    product_exponent = es + ex[cols]
    rows = np.concatenate([np.repeat(np.arange(size), counts), np.arange(size)])
    mantissa = np.concatenate([-product, mb.astype(object)])
    exponent = np.concatenate([product_exponent, eb])
    nonzero = np.concatenate([(ms != 0) & (mx[cols] != 0), mb != 0])
    order = np.argsort(rows, kind='stable')
    rows, mantissa, exponent, nonzero = rows[order], mantissa[order], exponent[order], nonzero[order]
    starts = np.concatenate([[0], np.cumsum(counts + 1)[:-1]])
    sentinel = np.iinfo(np.int64).max
    masked = np.where(nonzero, exponent, sentinel)
    row_min = np.minimum.reduceat(masked, starts)
    row_min = np.where(row_min == sentinel, 0, row_min)
    shift = np.where(nonzero, exponent - row_min[rows], 0).astype(object)
    shifted = np.left_shift(mantissa, shift)
    residual = np.add.reduceat(shifted, starts)
    reference = np.add.reduceat(np.abs(shifted), starts)
    return ExactRows(residual=[int(v) for v in residual], reference=[int(v) for v in reference],
                     exponent=row_min.astype(np.int64))


def _sqrt_hi(numerator: int, exponent: int) -> Fraction:
    """An upper bound of sqrt(numerator * 2^exponent) (numerator >= 0), via an exact integer root."""
    if numerator == 0:
        return Fraction(0)
    if exponent % 2:
        numerator <<= 1
        exponent -= 1
    root = math.isqrt(numerator)
    if root * root < numerator:
        root += 1
    return _scaled(root, exponent // 2)


def _scaled_residual_norm_hi(rows: ExactRows, shifts: np.ndarray) -> Fraction:
    """sqrt_hi(sum_i (d_i r_i)^2) with d_i = 2^shifts[i], exactly then rounded up by the root."""
    terms = [(r, 2 * (int(e) + int(t))) for r, e, t in zip(rows.residual, rows.exponent, shifts) if r]
    if not terms:
        return Fraction(0)
    low = min(k for _, k in terms)
    total = sum((r * r) << (k - low) for r, k in terms)
    return _sqrt_hi(total, low)


def _row_abs_sums(system: sparse.csr_matrix) -> list:
    """sum_j |S_ij| per row, exactly, as Fractions (dyadic rationals)."""
    ms, es = _decompose(system.data)
    ms = np.abs(ms)                     # integer magnitude: no float abs of a subnormal
    out = []
    for i in range(system.shape[0]):
        lo, hi = system.indptr[i], system.indptr[i + 1]
        mant, expo = ms[lo:hi], es[lo:hi]
        keep = mant != 0
        if not keep.any():
            out.append(Fraction(0))
            continue
        mant, expo = mant[keep], expo[keep]
        low = int(expo.min())
        out.append(_scaled(sum(int(m) << (int(e) - low) for m, e in zip(mant, expo)), low))
    return out


def _backward_check(rows: ExactRows, system: sparse.csr_matrix, x: np.ndarray, b: np.ndarray) -> dict:
    """V1-V3 (owner decision OD7, 2026-11-19): accept iff |r_i| <= 1e-10 ref_i for every row, exactly,
    with the per-row normwise reference ref_i = sum_j |S_ij| * ||x||_inf + |b_i|. A row with ref_i = 0
    (a zero row or x = 0, and b_i = 0) has r_i = 0 and is exempt; r_i != 0 with ref_i = 0 cannot occur."""
    failing, exempt, worst = [], 0, 0.0
    x_inf = _abs_max_exact(x)
    abs_sums = _row_abs_sums(system)
    for i, (r, row_sum, bi) in enumerate(zip(rows.residual, abs_sums, b)):
        ref = row_sum * x_inf + abs(_exact(bi))
        residual = rows.residual_fraction(i)
        if ref == 0:
            if residual != 0:
                raise _refuse('COUPLED_UNCERTIFIED', 'internal_residual_inconsistency',
                              'a nonzero exact residual with a zero reference')
            exempt += 1
            continue
        if abs(residual) > BACKWARD_TOLERANCE * ref:
            failing.append(i)
        if r:
            worst = max(worst, _to_float(abs(residual) / ref))
    return {'accepted': not failing, 'failing_rows': failing[:20], 'failing_count': len(failing),
            'exempt_rows': exempt, 'max_ratio': worst, 'reference': 'row_normwise_od7'}


def _forward_scale(x: np.ndarray, shifts: np.ndarray) -> Fraction:
    """max_i |x_i / d_i|, exactly."""
    best = Fraction(0)
    nonzero = _nonzero_mask(x)
    for value, t, keep in zip(x, shifts, nonzero):
        if keep:
            candidate = abs(_exact(value)) * (Fraction(2) ** -int(t))
            if candidate > best:
                best = candidate
    return best


def _abs_max_exact(x: np.ndarray) -> Fraction:
    """max_i |x_i| exactly: binary64 magnitudes order as their bit patterns without the sign bit."""
    bits = _float_bits(x) & _ABS_MASK
    if not bits.size:
        return Fraction(0)
    return _exact(_float_from_bits(int(bits.max())))


def _forward_error_accepts(error: Fraction, x: np.ndarray, shifts: np.ndarray) -> bool:
    """V6: E <= 1e-6 max_i |x_i / d_i|, decided exactly."""
    return error <= FORWARD_TOLERANCE * _forward_scale(x, shifts)


# --- Section 2.5: candidate; the whole certified solve --------------------------------------------------

def _candidate(system: sparse.csr_matrix, b: np.ndarray) -> np.ndarray:
    """SuperLU candidate with up to two refinement steps on the exact residual. Carries no claim."""
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
    """x within d_i * forward_error of the exact solution x* of S x = b (claim under A1/A2)."""

    x: np.ndarray
    shifts: np.ndarray
    forward_error: Fraction
    sigma_lower: Fraction
    certificate: Mapping[str, Any]
    backward: Mapping[str, Any]
    zero_load: bool
    timings: Mapping[str, float] = field(default_factory=dict)

    def bound(self, i: int) -> Fraction:
        return self.forward_error * (Fraction(2) ** int(self.shifts[i]))


def _poll(cancel: Optional[Callable[[], None]]) -> None:
    """Run the caller's cancellation check, then re-check the FP environment (a callback may have
    changed it)."""
    if cancel is not None:
        cancel()
    _require_fp_environment()


def _checked_clock(clock: Callable[[], Any]) -> Callable[[], Any]:
    """The caller's timing callback, followed by the FP-environment probe on every call (review r3 P2):
    a clock that switches the rounding mode or FTZ/DAZ before a stage and back at the next timing call
    refuses at the first call, before the stage runs. The value must be an exact ``float`` or ``int``,
    so the route's own subtraction runs no caller code."""
    def tick():
        value = clock()
        _require_fp_environment()
        if type(value) is not float and type(value) is not int:
            raise _refuse('COUPLED_NATIVE_INVALID', 'clock_invalid', 'the clock must return a float or an int')
        return value
    return tick


def certified_solve(system: Any, rhs: Any, *, raw_load_zero: bool,
                    memory_limit_bytes: Optional[int] = None,
                    cancel: Optional[Callable[[], None]] = None,
                    clock: Optional[Callable[[], float]] = None) -> CertifiedSolve:
    """Certify S, solve S x = b, and verify V1-V3 and V6 exactly; refuse with a code otherwise.

    ``raw_load_zero`` is True iff the native load, the offset u0 and every consumed pressure are
    exactly zero (the zero-load path, spec 2.4); then b must be exactly zero and x = 0. A nonzero raw
    load whose stored b is all zero refuses COUPLED_FORMATION_LOSS.
    """
    import time
    # Review r4 P2: only identity tests on the caller's arguments before the checks (no __bool__).
    clock = _checked_clock(time.perf_counter if clock is None else clock)
    if type(raw_load_zero) is not bool:
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', 'raw_load_zero must be a bool')
    if memory_limit_bytes is not None and type(memory_limit_bytes) is not int:
        raise _refuse('COUPLED_UNCERTIFIED', 'invalid_arguments', 'memory_limit_bytes must be None or an int')
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
    t0 = clock()
    certificate = _certify_solve_matrix(system, memory_limit_bytes=memory_limit_bytes, cancel=cancel)
    timings['certificate'] = clock() - t0
    public = {k: v for k, v in certificate.items() if k != 'shifts'}
    if not certificate['accepted']:
        raise _refuse('COUPLED_UNCERTIFIED', certificate['reason'],
                      f"the coupled system is not certified ({certificate['reason']})", certificate=public)
    size = system.shape[0]
    shifts = np.asarray(certificate['shifts'], dtype=np.int64)
    if size == 0:
        return CertifiedSolve(np.zeros(0), shifts, Fraction(0), Fraction(1), public,
                              {'accepted': True, 'exempt_rows': 0, 'max_ratio': 0.0, 'failing_count': 0},
                              raw_load_zero, timings)
    sigma_lower = _sigma_lower(certificate['sigma_max'], certificate['sigma_min'], size)
    if not sigma_lower > 0:
        raise _refuse('COUPLED_UNCERTIFIED', 'singular_within_svd_error',
                      'the exact singular-value lower bound is not positive', certificate=public)
    _poll(cancel)
    t0 = clock()
    x = np.zeros(size) if raw_load_zero else _candidate(system, b)
    timings['candidate'] = clock() - t0
    _poll(cancel)
    t0 = clock()
    _require_fp_environment()            # immediately around the exact residual (review r2 (a))
    rows = _exact_rows(system, x, b)
    _require_fp_environment()
    timings['exact_residual'] = clock() - t0
    t0 = clock()
    backward = _backward_check(rows, system, x, b)
    if not backward['accepted']:
        raise _refuse('COUPLED_UNCERTIFIED', 'backward_error_above_limit',
                      'the exact residual exceeds 1e-10 of its row reference', certificate=public,
                      backward=backward)
    error = _scaled_residual_norm_hi(rows, shifts) / sigma_lower
    if not _forward_error_accepts(error, x, shifts):
        raise _refuse('COUPLED_UNCERTIFIED', 'forward_error_above_limit',
                      'the certified forward error exceeds 1e-6 of the solution scale', certificate=public,
                      forward_error=_out_up(error))
    timings['verification'] = clock() - t0
    # Review r1 #8 (class 7): a cancellation during verification never returns success, and the
    # environment must still preserve subnormals after any callback.
    _poll(cancel)
    return CertifiedSolve(x, shifts, error, sigma_lower, public, backward, raw_load_zero, timings)
