"""Deterministic scalar reductions, not a fully portable replay route.

Selected native geometry and contact work kernels use these reductions.
Complete archived checkpoint portability remains unresolved; its original
gates are unchanged.

`np.linalg.norm` (ord=None) and small vector dot products dispatch to OpenBLAS
ddot, and ord=2 matrix norms dispatch to LAPACK SVD; both accumulation orders
depend on the host micro-architecture and library build. Traced small-vector
norm inputs matched an ascending fused-multiply-add chain on the recording
host; this candidate specifies that fixed order for scalar reductions. These
kernels specify a fixed input-independent chain, one correctly rounded step
per element. They remove dispatch dependence from those reductions; they do
not establish portability of complete records containing other calculations.

The spectral 2-norm is computed as the SAME mathematical quantity, not a
surrogate bound: the correctly rounded largest singular value, obtained from
the exactly accumulated Gram matrix eigenvalue via exact PSD tests (all
principal minors, supported for at most three columns/rows) and adjacent-
binary64 midpoint comparisons, initialized from a relative-precision rational
approximation. There is no BLAS/LAPACK fallback on this path; non-finite
inputs, more than three Gram dimensions, results outside the binary64 range
and bracketing failures are refused by exception.

Contract distinction: `deterministic_dot`/`deterministic_norm` reproduce the
recorded finite ddot chain (each step rounded once, including its
overflow/underflow behavior for extreme scales); `deterministic_spectral_norm`
is a correctly rounded norm of finite input that refuses rather than
silently underflowing, overflowing or returning non-finite values.
"""
import struct
import math
from fractions import Fraction
from itertools import combinations
from math import isqrt, nextafter, sqrt

import numpy as np

DETERMINISTIC_SPECTRAL_MAXIMUM_DIMENSION = 3
_BRACKET_MAXIMUM_STEPS = 16


def _fma(product_left, product_right, addend):
    # The operands are finite. An infinite accumulator is an earlier rounded
    # overflow, not an independently rounded infinite product to cancel.
    if math.isinf(addend):
        return addend
    native_fma = getattr(math, 'fma', None)
    if native_fma is not None:
        try:
            return native_fma(product_left, product_right, addend)
        except OverflowError:
            pass
    exact = Fraction(product_left) * Fraction(product_right) + Fraction(addend)
    if exact == 0:
        negative_product_zero = (product_left == 0 or product_right == 0) and (
            math.copysign(1., product_left) != math.copysign(1., product_right))
        return -0. if negative_product_zero and addend == 0 and math.copysign(1., addend) < 0 else 0.
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


def deterministic_dot(x, y):
    """Ascending single-rounding FMA chain; the recorded ddot accumulation order."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError('deterministic dot requires matching shapes')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('deterministic dot requires finite input')
    total = 0.0
    for a, b in zip(x, y):
        total = _fma(float(a), float(b), total)
    return total


def deterministic_norm(x):
    """2-norm/Frobenius norm over C-ravelled input, recorded ddot order."""
    return sqrt(deterministic_dot(x, x))


def _det(M):
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0]*M[1][1]-M[0][1]*M[1][0]
    return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
            -M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
            +M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))


def _psd(M):
    """Exact PSD test for symmetric matrices of dimension at most three."""
    n = len(M)
    for k in range(1, n+1):
        for S in combinations(range(n), k):
            if _det([[M[i][j] for j in S] for i in S]) < 0:
                return False
    return True


def _sqrt_fraction(x):
    """Relative-precision square root of a nonnegative exact fraction.

    The input is rescaled by a power of four into [1, 4) using bit lengths,
    so the fixed 320-bit integer square root delivers about 2^-160 relative
    precision at every exponent scale; the rescaling loops are bounded by
    construction (the bit-length exponent is within one step of the target
    binade pair).
    """
    if x == 0:
        return Fraction(0)
    if x < 0:
        raise ValueError('square root requires a nonnegative value')
    exponent = (x.numerator.bit_length()-x.denominator.bit_length())//2
    scaled = x / (1 << (2*exponent)) if exponent >= 0 else x*(1 << (-2*exponent))
    while scaled >= 4:
        scaled /= 4
        exponent += 1
    while scaled < 1:
        scaled *= 4
        exponent -= 1
    root = Fraction(isqrt(scaled.numerator*(1 << 320)//scaled.denominator), 1 << 160)
    return root*(1 << exponent) if exponent >= 0 else root/(1 << -exponent)


def _gram(flat, rows, cols):
    """Exactly accumulated Gram matrix of the smaller dimension."""
    if cols <= rows:
        return [[sum(Fraction(flat[r*cols+c1])*Fraction(flat[r*cols+c2])
                     for r in range(rows)) for c2 in range(cols)]
                for c1 in range(cols)]
    return [[sum(Fraction(flat[r1*cols+c])*Fraction(flat[r2*cols+c])
                 for c in range(cols)) for r2 in range(rows)]
            for r1 in range(rows)]


def _lambda_max_estimate(G):
    """High-precision rational estimate of the largest Gram eigenvalue."""
    dim = len(G)
    if dim == 1:
        return G[0][0]
    if dim == 2:
        trace = G[0][0]+G[1][1]
        discriminant = (G[0][0]-G[1][1])**2+4*G[0][1]*G[1][0]
        return (trace+_sqrt_fraction(discriminant))/2
    low = max(G[i][i] for i in range(3))
    high = sum(G[i][i] for i in range(3))
    if low == high:
        return low
    for _ in range(220):
        middle = (low+high)/2
        if _psd([[(middle if i == j else 0)-G[i][j] for j in range(3)]
                 for i in range(3)]):
            high = middle
        else:
            low = middle
    return (low+high)/2


def _even_binary64(value):
    return struct.unpack('<Q', struct.pack('<d', value))[0] % 2 == 0


def _estimate_float(value):
    try:
        estimate = float(value)
    except OverflowError:
        raise ValueError('deterministic spectral norm result outside binary64 range') from None
    return estimate


def _round_sqrt(dominates, exact_hit, estimate):
    """Correctly rounded binary64 square root from exact dominance tests.

    ``dominates(x)`` is an exact predicate for ``x*x >= target`` on a
    nonnegative Fraction and ``exact_hit(x)`` for equality. The relative-
    precision ``estimate`` is within a few binary64 steps of the true root,
    so the adjacent bracket is found in at most ``_BRACKET_MAXIMUM_STEPS``
    nextafter steps per direction; any failure to bracket is refused by
    exception instead of looping.
    """
    if estimate == float('inf') or not (estimate >= 0.0):
        raise ValueError('deterministic spectral norm result outside binary64 range')
    high = estimate
    for _ in range(_BRACKET_MAXIMUM_STEPS):
        if high == float('inf'):
            raise ValueError('deterministic spectral norm result outside binary64 range')
        if dominates(Fraction(high)):
            break
        high = nextafter(high, float('inf'))
    else:
        raise ValueError('deterministic spectral norm bracketing failed')
    if exact_hit(Fraction(high)):
        return high
    low = nextafter(high, 0.0)
    for _ in range(_BRACKET_MAXIMUM_STEPS):
        if not dominates(Fraction(low)):
            break
        if exact_hit(Fraction(low)):
            return low
        high = low
        low = nextafter(low, 0.0)
    else:
        raise ValueError('deterministic spectral norm bracketing failed')
    middle = (Fraction(low)+Fraction(high))/2
    if dominates(middle):
        if exact_hit(middle):
            # The exact norm is the non-representable midpoint: half-to-even.
            return high if _even_binary64(high) else low
        return low
    return high


def deterministic_spectral_norm(a):
    """Correctly rounded spectral 2-norm (largest singular value).

    The exact Gram matrix eigenvalue is bracketed between adjacent binary64
    values of the norm using exact PSD tests of x*x*I-G (all principal
    minors), initialized from a high-precision rational approximation; the
    neighboring-value midpoint test then selects the correctly rounded result.
    Vectors use the exactly accumulated sum of squares, so tiny or huge
    finite inputs cannot underflow or overflow the way the recorded ddot
    chain does; non-finite input and results outside the binary64 range are
    refused.
    """
    array = np.asarray(a, dtype=float)
    if array.ndim > 2:
        raise ValueError('deterministic spectral norm requires a vector or matrix')
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        raise ValueError('deterministic spectral norm requires finite input')
    if array.ndim == 1:
        total = sum((Fraction(value)*Fraction(value)
                     for value in array.ravel().tolist()), Fraction(0))
        if total == 0:
            return 0.0
        return _round_sqrt(lambda x: x*x >= total,
                           lambda x: x*x == total,
                           _estimate_float(_sqrt_fraction(total)))
    rows, cols = array.shape
    if min(rows, cols) > DETERMINISTIC_SPECTRAL_MAXIMUM_DIMENSION:
        raise ValueError('deterministic spectral norm supports at most %d columns/rows'
                         % DETERMINISTIC_SPECTRAL_MAXIMUM_DIMENSION)
    flat = array.ravel().tolist()
    G = _gram(flat, rows, cols)
    dim = len(G)
    if not any(any(value != 0 for value in row) for row in G):
        return 0.0

    def dominates(x):
        return _psd([[(x*x if i == j else 0)-G[i][j] for j in range(dim)]
                     for i in range(dim)])

    def exact_hit(x):
        M = [[(x*x if i == j else 0)-G[i][j] for j in range(dim)] for i in range(dim)]
        return _psd(M) and _det(M) == 0

    return _round_sqrt(dominates, exact_hit,
                       _estimate_float(_sqrt_fraction(_lambda_max_estimate(G))))
