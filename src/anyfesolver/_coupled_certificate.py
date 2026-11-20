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

import contextlib
from dataclasses import dataclass, field
from fractions import Fraction
import functools
import math
import os
import struct
from typing import Any, Callable, Mapping, Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

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


def _float_from_bits(bits: int) -> float:
    return struct.unpack('<d', struct.pack('<Q', bits & 0xFFFF_FFFF_FFFF_FFFF))[0]


def _bits_of(value: float) -> int:
    """The signed int64 bit pattern of one binary64 value (a byte copy)."""
    return struct.unpack('<q', struct.pack('<d', value))[0]


def _float_bits(values: Any) -> np.ndarray:
    """int64 bit patterns of a float64 array (a byte copy; callers pass float64 only)."""
    return np.ascontiguousarray(values, dtype=np.float64).view(np.int64)


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


def _subnormals_preserved() -> bool:
    """False when this thread flushes subnormal operands or results (DAZ/FTZ), for Python floats
    or for NumPy's vector loops. The certificate premises (A2) and every float comparison of the
    route assume IEEE gradual underflow."""
    tiny = _float_from_bits(1)
    if _bits_of(tiny * 3.0) != 3 or not tiny != 0.0:
        return False
    vector = np.full(64, tiny)
    return bool((_float_bits(vector * 3.0) == 3).all()) and bool((vector != 0.0).all())


_ONE_BITS = 0x3FF0_0000_0000_0000
_HALF_ULP_BITS = 0x3CA0_0000_0000_0000          # 2^-53, half an ulp of 1.0


def _nearest_rounding() -> bool:
    """False unless this thread rounds to nearest-even, for Python floats and NumPy's vector loops
    (review r2 (a)). 1 + 2^-53 is a tie that rounds down to 1 and (1 + 2^-52) + 2^-53 a tie that
    rounds up to 1 + 2^-51; the negated pair rules out rounding toward zero. Operands come from bit
    patterns at run time, so nothing is folded at compile time."""
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


def _fp_environment_reason() -> Optional[str]:
    if not _subnormals_preserved():
        return 'subnormal_flush'
    if not _nearest_rounding():
        return 'rounding_mode'
    return None


def _require_fp_environment() -> None:
    """Gradual underflow and round-to-nearest-even on this thread, else a coded refusal."""
    reason = _fp_environment_reason()
    if reason == 'subnormal_flush':
        raise _refuse('COUPLED_UNCERTIFIED', 'subnormal_flush',
                      'the floating-point environment flushes subnormals (FTZ/DAZ); the certificate '
                      'premise A2 and exact float comparisons do not hold')
    if reason is not None:
        raise _refuse('COUPLED_UNCERTIFIED', 'rounding_mode',
                      'the floating-point environment does not round to nearest-even; the certificate '
                      'premises A1/A2 do not hold')


@contextlib.contextmanager
def _single_thread_blas():
    """Run the enclosed LAPACK work on the calling thread only (review r2 (a), owner decision OD9):
    threadpoolctl limits every BLAS pool to one thread, and a probe confirms that NumPy's own BLAS
    is among the controlled libraries and that every BLAS pool reports exactly one thread. With one
    thread the BLAS runs inline on the caller, whose FP environment the route verifies before and
    after. Anything that cannot be established refuses ``COUPLED_UNCERTIFIED`` (``blas_threads``)."""
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError as exc:
        raise _refuse('COUPLED_UNCERTIFIED', 'blas_threads',
                      'threadpoolctl is required to run the certificate SVD on one thread') from exc
    with threadpool_limits(limits=1, user_api='blas'):
        _require_single_thread_blas(threadpool_info)
        yield functools.partial(_require_single_thread_blas, threadpool_info)


def _numpy_blas_dirs() -> tuple:
    root = os.path.normcase(os.path.dirname(os.path.abspath(np.__file__)))
    return (root + os.sep, root + '.libs' + os.sep)


def _require_single_thread_blas(threadpool_info) -> None:
    try:
        pools = [p for p in threadpool_info() if p.get('user_api') == 'blas']
    except Exception as exc:  # noqa: BLE001  (an introspection failure is a refusal)
        raise _refuse('COUPLED_UNCERTIFIED', 'blas_threads', 'the BLAS thread pools cannot be inspected') from exc
    dirs = _numpy_blas_dirs()
    numpy_pools = [p for p in pools if os.path.normcase(str(p.get('filepath', ''))).startswith(dirs)]
    if not numpy_pools or any(p.get('num_threads') != 1 for p in pools):
        raise _refuse('COUPLED_UNCERTIFIED', 'blas_threads',
                      'the certificate SVD cannot be verified to run on one thread '
                      f'({len(numpy_pools)} NumPy BLAS pool(s), threads {[p.get("num_threads") for p in pools]})')


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
_REFINEMENT_STEPS = 2
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


def _svd_values(matrix: np.ndarray) -> np.ndarray:
    """Singular values of the dense equilibrated matrix (the one LAPACK call the certificate relies on),
    on one verified BLAS thread under verified round-to-nearest and gradual underflow, checked
    immediately before and after the call (review r2 (a))."""
    with _single_thread_blas() as recheck:
        _require_fp_environment()
        sigma = np.linalg.svd(matrix, compute_uv=False)
        _require_fp_environment()
        recheck()
    return sigma


def _dense_bytes(size: int) -> int:
    return _DENSE_COPIES * size * size * 8


def _certify_solve_matrix(system: Any, *, memory_limit_bytes: Optional[int] = None) -> dict:
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
            return _certificate_numbers(system, size)
    except MemoryError as exc:
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error',
                      'out of memory in the dense certificate', size=size) from exc


def _certificate_numbers(system: sparse.csr_matrix, size: int) -> dict:
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
        sigma = _svd_values(A)
        del A
    except MemoryError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError) as exc:
        certificate['reason'] = f'numerical_failure:{type(exc).__name__}'
        return certificate
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
    except MemoryError as exc:
        raise _refuse('COUPLED_TOO_LARGE', 'memory_error', 'out of memory in the candidate solve') from exc
    except Exception as exc:  # includes SuperLU's exact-singular RuntimeError
        raise _refuse('COUPLED_UNCERTIFIED', 'candidate_failed',
                      f'the candidate solve failed ({_type_name(exc)})') from exc
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
    clock = clock or time.perf_counter
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
    certificate = _certify_solve_matrix(system, memory_limit_bytes=memory_limit_bytes)
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
