"""Exact certificate arithmetic of the coupled shell-solid route (private; owner decisions OD13/OD14).

One source for the parent process and the certificate worker (``_coupled_svd_worker``). Since OD14
(2026-11-21) every certificate decision runs in the worker: the power-of-two equilibration, the SVD,
delta and kappa_up, the exact inverse-norm bound, the exact integer residual, the backward checks
V1-V3, the forward error E and V6, and the bounded recovery values. The parent imports these
functions for its candidate refinement (which carries no claim) and for the tests.

The module imports only the standard library and NumPy, and nothing from anyfesolver: the worker
loads it by file path. Pure functions; a refusal inside the worker is a :class:`CertificateRefusal`.
See ``docs/SOLID_COUPLING.md``.
"""

from __future__ import annotations

from fractions import Fraction
import math
import struct

import numpy as np

# --- Constants (the worker checks the parent's copy against these) --------------------------------------

COUPLED_MAX_EQUATIONS = 5000
# Accept only kappa_up * eps <= 1e-6 of the equilibrated stored matrix (kappa_up <= ~4.5e9).
CONDITION_LIMIT = 1e-6 / np.finfo(float).eps
EQUILIBRATION_MAX_SWEEPS = 64
# SVD error constant: alpha = c N eps with c = max(1000, N) (assumption A1).
SVD_ERROR_CONSTANT_MIN = 1000
# Backward (V1-V3) and forward (V6) tolerances, decided exactly as rationals.
BACKWARD_TOLERANCE = Fraction(1, 10 ** 10)
FORWARD_TOLERANCE = Fraction(1, 10 ** 6)
# Bytes per dense working copy of an N x N float64 matrix, times the copies held at once.
DENSE_COPIES = 4
# Fraction bits of the recovery-bound square root.
SQRT_BITS = 120
EPS = Fraction(1, 2 ** 52)

_SIGN_BIT = 1 << 63
_FRACTION_BITS = (1 << 52) - 1
_INF_BITS = 0x7FF << 52
_ABS_MASK = np.int64(0x7FFF_FFFF_FFFF_FFFF)
_ONE_BITS = 0x3FF0_0000_0000_0000
_HALF_ULP_BITS = 0x3CA0_0000_0000_0000          # 2^-53, half an ulp of 1.0


class CertificateRefusal(Exception):
    """A coded refusal decided by the certificate (code, reason, message, JSON-safe diagnostics)."""

    def __init__(self, code: str, reason: str, message: str, **diagnostics) -> None:
        super().__init__(message)
        self.code, self.reason, self.message, self.diagnostics = code, reason, message, diagnostics


def dense_bytes(size: int) -> int:
    return DENSE_COPIES * size * size * 8


# --- IEEE bit helpers (review r1 #1, class 1) -------------------------------------------------------------
# frexp, float.as_integer_ratio, Fraction(float), float(Fraction), ldexp, comparisons and any() change
# their result for subnormals when the thread flushes denormals (FTZ/DAZ). These read and write the
# IEEE binary64 bit pattern with integer operations only.

def _float_from_bits(bits: int) -> float:
    return struct.unpack('<d', struct.pack('<Q', bits & 0xFFFF_FFFF_FFFF_FFFF))[0]


def _bits_of(value: float) -> int:
    """The signed int64 bit pattern of one binary64 value (a byte copy)."""
    return struct.unpack('<q', struct.pack('<d', value))[0]


def _float_bits(values):
    """int64 bit patterns of a float64 array (a byte copy; callers pass float64 only)."""
    return np.ascontiguousarray(values, dtype=np.float64).view(np.int64)


def _nonzero_mask(values) -> np.ndarray:
    """values != 0 decided on the bit pattern (+-0 are zero; a subnormal is nonzero)."""
    return (_float_bits(values) & _ABS_MASK) != 0


def _any_nonzero(values) -> bool:
    return bool(_nonzero_mask(values).any())


def _decompose(values):
    """Exact (integer mantissa, exponent) with value = m * 2^e; m is a signed integer below 2^53."""
    bits = _float_bits(values)
    biased = (bits >> 52) & 0x7FF
    fraction = bits & _FRACTION_BITS
    normal = biased != 0
    mantissa = np.where(normal, fraction | (1 << 52), fraction)
    exponent = np.where(normal, biased - 1075, -1074)
    mantissa = np.where(bits < 0, -mantissa, mantissa)
    return mantissa.astype(np.int64), exponent.astype(np.int64)


def _scaled(value: int, exponent: int) -> Fraction:
    return Fraction(value << exponent) if exponent >= 0 else Fraction(value, 1 << -exponent)


def _exact(value) -> Fraction:
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
    ``up``, toward +inf; integer arithmetic and bit assembly only."""
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


def _int_to_float(value: int, exponent: int) -> float:
    """value * 2^exponent rounded to nearest binary64 (bit assembly, FP-environment independent)."""
    if value == 0:
        return 0.0
    if exponent >= 0:
        return _ratio_to_float(value << exponent, 1)
    return _ratio_to_float(value, 1 << -exponent)


# --- Floating-point environment probes ----------------------------------------------------------------------

def _subnormals_preserved() -> bool:
    """False when this thread flushes subnormal operands or results (DAZ/FTZ), for Python floats
    or for NumPy's vector loops."""
    tiny = _float_from_bits(1)
    if _bits_of(tiny * 3.0) != 3 or not tiny != 0.0:
        return False
    vector = np.full(64, tiny)
    return bool((_float_bits(vector * 3.0) == 3).all()) and bool((vector != 0.0).all())


def _nearest_rounding() -> bool:
    """False unless this thread rounds to nearest-even, for Python floats and NumPy's vector loops
    (review r2 (a)). Operands come from bit patterns at run time, so nothing is folded."""
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


# --- Condition certificate (sections 2.1-2.2) -----------------------------------------------------------

def _round_up(value: float) -> float:
    """An upper bound of the exact result x of one correctly rounded operation whose result is value."""
    return float(np.nextafter(value, np.inf))


def _round_down(value: float) -> float:
    return float(np.nextafter(value, -np.inf))


_SMALLEST_SUBNORMAL = _float_from_bits(1)


def _condition_upper_bound(sigma_max: float, sigma_min: float, size: int) -> dict:
    """Rigorous upper bound of kappa(A*) from computed singular values (pure function).

    beta = alpha / (1 - alpha), delta = beta * sigma_max + N 2^-1074,
    kappa(A*) <= (sigma_max + delta) / (sigma_min - delta) if sigma_min > delta,
    every operation rounded outward (IEEE gradual underflow and round-to-nearest assumed; the worker
    verifies both). Refuses for alpha >= 1/2, sigma_min <= delta or invalid inputs."""
    with np.errstate(all='ignore'):
        result = {'svd_error_constant': None, 'svd_error_alpha': None, 'svd_error_bound': float('inf'),
                  'condition_upper_bound': float('inf'), 'accepted': False, 'reason': None}
        sigma_max, sigma_min, size = float(sigma_max), float(sigma_min), int(size)
        if not (size >= 1 and np.isfinite(sigma_max) and np.isfinite(sigma_min)
                and sigma_max >= sigma_min >= 0.0 and sigma_max > 0.0):
            result['reason'] = 'invalid_singular_values'
            return result
        constant = max(SVD_ERROR_CONSTANT_MIN, size)
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


def _sigma_lower(sigma_max: float, sigma_min: float, size: int) -> Fraction:
    """sigma^- = s_N - (beta s_1 + N 2^-1074), beta = alpha / (1 - alpha), alpha = c N eps, exactly."""
    constant = max(SVD_ERROR_CONSTANT_MIN, int(size))
    alpha = Fraction(constant * int(size)) * EPS
    beta = alpha / (1 - alpha)
    return _exact(sigma_min) - (beta * _exact(sigma_max) + int(size) * Fraction(1, 2 ** 1074))


class CSR:
    """The stored system as plain arrays (shape, indptr, indices, data): what the exact residual reads."""

    def __init__(self, size: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray) -> None:
        self.shape = (size, size)
        self.indptr, self.indices, self.data = indptr, indices, data


def canonical_csr_reason(size: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray):
    """None for a canonical square CSR of finite float64 entries (exact integer checks), else a reason."""
    if indptr.shape != (size + 1,) or indices.shape != data.shape or indices.ndim != 1:
        return 'invalid_system'
    if int(indptr[0]) != 0 or int(indptr[-1]) != indices.shape[0] or bool((np.diff(indptr) < 0).any()):
        return 'invalid_system'
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= size):
        return 'invalid_system'
    if indices.size > 1:
        step = np.diff(indices)
        row_start = np.zeros(indices.size, dtype=bool)
        row_start[indptr[1:-1][indptr[1:-1] < indices.size]] = True
        if bool(((step <= 0) & ~row_start[1:]).any()):
            return 'invalid_system'                       # unsorted or duplicate columns in a row
    if bool(((_float_bits(data) & _ABS_MASK) >= _INF_BITS).any()):
        return 'nonfinite_entry'
    return None


def dense_from_csr(system: CSR) -> np.ndarray:
    """The dense matrix exactly as scipy's toarray builds it from canonical CSR (0.0 + entry)."""
    size = system.shape[0]
    dense = np.zeros((size, size))
    rows = np.repeat(np.arange(size), np.diff(system.indptr))
    with np.errstate(all='ignore'):
        dense[rows, system.indices] = 0.0 + system.data
    return dense


def equilibrate(original: np.ndarray, certificate: dict):
    """Power-of-two Ruiz equilibration of the dense stored matrix: returns (A*, shifts) or (None, None)
    with ``certificate['reason']`` set. Cumulative shifts on the original entries: each entry is
    rounded at most once (A2)."""
    size = original.shape[0]
    try:
        with np.errstate(all='ignore'):
            if not np.isfinite(original).all():
                certificate['reason'] = 'nonfinite_entry'
                return None, None
            A = original
            total = np.zeros(size, dtype=np.int64)
            for sweep in range(EQUILIBRATION_MAX_SWEEPS + 1):
                largest = np.max(np.abs(A), axis=1, initial=0.0)
                if np.any(largest == 0.0):
                    certificate['reason'] = 'zero_row'
                    return None, None
                shift = -(np.frexp(largest)[1] // 2)
                if not shift.any():
                    break
                if sweep == EQUILIBRATION_MAX_SWEEPS:
                    certificate['reason'] = 'equilibration_not_converged'
                    return None, None
                total = total + shift
                A = np.ldexp(original, total[:, None] + total[None, :])
                certificate['equilibration_sweeps'] = sweep + 1
    except MemoryError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError) as exc:
        certificate['reason'] = f'numerical_failure:{type(exc).__name__}'
        return None, None
    return A, total


def singular_value_certificate(sigma: np.ndarray, size: int, certificate: dict) -> None:
    """The kappa decision from the computed singular values (into ``certificate``)."""
    sigma = np.asarray(sigma, dtype=np.float64)
    bits = sigma.view(np.int64) if sigma.size else np.zeros(0, dtype=np.int64)
    magnitude = bits & _ABS_MASK
    if not sigma.size or not bool((magnitude < _INF_BITS).all()):
        certificate['reason'] = 'nonfinite_singular_values'
        return
    if bool(((bits < 0) & (magnitude != 0)).any()) or \
            (sigma.size > 1 and not bool((magnitude[:-1] >= magnitude[1:]).all())):
        certificate['reason'] = 'invalid_singular_values'
        return
    sigma = magnitude.view(np.float64)                     # -0.0 -> +0.0, exactly
    sigma_max, sigma_min = float(sigma[0]), float(sigma[-1])
    certificate.update(sigma_max=sigma_max, sigma_min=sigma_min)
    with np.errstate(all='ignore'):
        if sigma_min > 0.0:
            certificate['condition_number'] = sigma_max / sigma_min  # reported only, never decided on
    certificate.update(_condition_upper_bound(sigma_max, sigma_min, size))


def new_certificate(size: int) -> dict:
    return {'matrix': 'kkt_stored', 'size': size, 'method': 'power_of_two_ruiz_dense_svd',
            'condition_limit': CONDITION_LIMIT, 'condition_number': float('inf'),
            'condition_upper_bound': float('inf'), 'equilibration_sweeps': 0,
            'accepted': False, 'reason': None}


def empty_certificate() -> dict:
    """The certificate of the 0 x 0 system (no free DOFs): S x = b has the unique solution x = (),
    there is no matrix entry, singular value or residual to decide on (review r7: decided in the
    worker like every other size, so the recovery maps are evaluated there too)."""
    return {'matrix': 'empty', 'size': 0, 'method': 'no_free_dofs', 'condition_limit': CONDITION_LIMIT,
            'condition_number': 1.0, 'condition_upper_bound': 1.0, 'equilibration_sweeps': 0,
            'accepted': True, 'reason': None}


def export_structure_reason(indptr: np.ndarray, columns: np.ndarray, size: int):
    """None for a valid recovery-map structure, else a reason (review r7 P1).

    ``bounded_rows`` bounds row r by sqrt(sum_k (M_rk d_j(k))^2) E, the 2-norm of the row taken over
    its entries; that is the row's norm only if no column repeats in the row (two entries 1 on one
    column are the coefficient 2, norm 2, not sqrt(2)). So every row must have distinct columns, in
    [0, size); the order within a row does not matter. Exact integer checks only."""
    if indptr.ndim != 1 or columns.ndim != 1 or indptr.shape[0] < 1:
        return 'structure'
    if int(indptr[0]) != 0 or int(indptr[-1]) != columns.shape[0] or bool((np.diff(indptr) < 0).any()):
        return 'structure'
    if columns.size and (int(columns.min()) < 0 or int(columns.max()) >= size):
        return 'column_range'
    if columns.size > 1:
        rows = np.repeat(np.arange(indptr.shape[0] - 1, dtype=np.int64), np.diff(indptr))
        order = np.lexsort((columns, rows))
        r, c = rows[order], columns[order]
        if bool(((r[1:] == r[:-1]) & (c[1:] == c[:-1])).any()):
            return 'repeated_column'
    return None


# --- Section 2.3: exact residual and V1-V3, V6 -------------------------------------------------------------

class ExactRows:
    """Per row i, ``residual[i] * 2^exponent[i]`` is exactly r_i = b_i - (S x)_i and
    ``reference[i] * 2^exponent[i]`` exactly sum_j |S_ij x_j| + |b_i| (Python integers)."""

    __slots__ = ('residual', 'reference', 'exponent')

    def __init__(self, residual: list, reference: list, exponent: np.ndarray) -> None:
        self.residual, self.reference, self.exponent = residual, reference, exponent

    def residual_fraction(self, i: int) -> Fraction:
        return _scaled(self.residual[i], int(self.exponent[i]))

    def residual_float(self) -> np.ndarray:
        """r rounded to nearest float64 (correctly rounded: Python integer arithmetic)."""
        return np.array([_int_to_float(r, int(e)) for r, e in zip(self.residual, self.exponent)], dtype=float)


def _exact_rows(system, x: np.ndarray, b: np.ndarray) -> ExactRows:
    """r = b - S x and ref = sum |S_ij x_j| + |b_i|, each row exactly, in Python integers; does not
    depend on the FP environment, the summation order or the term count. ``system`` is a canonical
    CSR (scipy or :class:`CSR`)."""
    size = system.shape[0]
    if size == 0:
        return ExactRows(residual=[], reference=[], exponent=np.zeros(0, dtype=np.int64))
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


def _row_abs_sums(system) -> list:
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


def _abs_max_exact(x: np.ndarray) -> Fraction:
    """max_i |x_i| exactly: binary64 magnitudes order as their bit patterns without the sign bit."""
    bits = _float_bits(x) & _ABS_MASK
    if not bits.size:
        return Fraction(0)
    return _exact(_float_from_bits(int(bits.max())))


def _backward_check(rows: ExactRows, system, x: np.ndarray, b: np.ndarray) -> dict:
    """V1-V3 (owner decision OD7): accept iff |r_i| <= 1e-10 ref_i for every row, exactly, with
    ref_i = sum_j |S_ij| * ||x||_inf + |b_i|. A row with ref_i = 0 has r_i = 0 and is exempt."""
    failing, exempt, worst = [], 0, 0.0
    x_inf = _abs_max_exact(x)
    abs_sums = _row_abs_sums(system)
    for i, (r, row_sum, bi) in enumerate(zip(rows.residual, abs_sums, b)):
        ref = row_sum * x_inf + abs(_exact(bi))
        residual = rows.residual_fraction(i)
        if ref == 0:
            if residual != 0:
                raise CertificateRefusal('COUPLED_UNCERTIFIED', 'internal_residual_inconsistency',
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


def _forward_error_accepts(error: Fraction, x: np.ndarray, shifts: np.ndarray) -> bool:
    """V6: E <= 1e-6 max_i |x_i / d_i|, decided exactly."""
    return error <= FORWARD_TOLERANCE * _forward_scale(x, shifts)


# --- Section 5: bounded recovery values ------------------------------------------------------------------

def _sqrt_hi_fraction(value: Fraction) -> Fraction:
    """An upper bound of sqrt(value), value >= 0 (SQRT_BITS fraction bits)."""
    if value <= 0:
        return Fraction(0)
    numerator = value.numerator << (2 * SQRT_BITS)
    scaled = -(-numerator // value.denominator)
    root = math.isqrt(scaled)
    if root * root < scaled:
        root += 1
    return Fraction(root, 1 << SQRT_BITS)


def bounded_rows(indptr: np.ndarray, columns: np.ndarray, coefficients, offsets, z: np.ndarray,
                 shifts: np.ndarray, error: Fraction):
    """values_i = sum_j M_ij z_j (+ offset_i) exactly, rounded once; bound
    |float - exact| + sqrt_hi(sum_j (M_ij d_j)^2) E. ``coefficients`` are exact Fractions in CSR
    order, ``offsets`` exact Fractions or None. Returns (values, bounds) float64 arrays. The structure
    must satisfy :func:`export_structure_reason` (distinct columns per row); a repeated column is
    refused here as well, never bounded."""
    reason = export_structure_reason(indptr, columns, z.shape[0])
    if reason is not None:
        raise CertificateRefusal('COUPLED_UNCERTIFIED', 'invalid_export', f'a recovery map is invalid ({reason})')
    values, bounds = [], []
    zq = {}
    for r in range(indptr.shape[0] - 1):
        exact = offsets[r] if offsets is not None else Fraction(0)
        square = Fraction(0)
        for k in range(int(indptr[r]), int(indptr[r + 1])):
            j, coefficient = int(columns[k]), coefficients[k]
            if j not in zq:
                zq[j] = _exact(z[j])
            exact += coefficient * zq[j]
            square += (coefficient * Fraction(2) ** int(shifts[j])) ** 2
        value = _to_float(exact)
        values.append(value)
        rounding = abs(_exact(value) - exact) if math.isfinite(value) else Fraction(0)
        bounds.append(_out_up(rounding + _sqrt_hi_fraction(square) * error))
    values = np.asarray(values, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if not (np.isfinite(values).all() and np.isfinite(bounds).all()):
        raise CertificateRefusal('COUPLED_UNCERTIFIED', 'bound_not_finite', 'a recovered value or bound is not finite')
    return values, bounds


# --- Exact rational wire encoding (hexadecimal: no int/str digit limit) ----------------------------------

def encode_fraction(value: Fraction) -> str:
    value = Fraction(value)
    sign = '-' if value < 0 else ''
    return f'{sign}{abs(value.numerator):x}/{value.denominator:x}'


def decode_fraction(text: str) -> Fraction:
    """The exact inverse of encode_fraction; ValueError on anything else."""
    if type(text) is not str or not text.isascii():
        raise ValueError('not an encoded fraction')
    body = text[1:] if text.startswith('-') else text
    numerator, sep, denominator = body.partition('/')
    hexdigits = set('0123456789abcdef')
    if sep != '/' or not numerator or not denominator or not set(numerator) <= hexdigits \
            or not set(denominator) <= hexdigits:
        raise ValueError('not an encoded fraction')
    n, d = int(numerator, 16), int(denominator, 16)
    if d == 0 or math.gcd(n, d) != 1 or (n == 0 and (d != 1 or text.startswith('-'))):
        raise ValueError('not a reduced fraction')
    value = Fraction(-n if text.startswith('-') else n, d)
    if encode_fraction(value) != text:
        raise ValueError('not canonical')
    return value
