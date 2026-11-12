"""Canonical near-real roots of the binary64 B3 projection polynomials.

The caller retains coefficient construction, normalization and degree trimming.
Here coefficients are exact dyadic numbers, not inputs to an eigenvalue backend.
Sturm isolation and integer signs certify rounding of every returned real part.
Nonreal conjugates are admitted by the *rounded* imaginary-part threshold too.
Only distinct real parts are needed: both projection callers deduplicate stations.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isfinite, lcm, nextafter, inf
from collections import OrderedDict
import struct
from sys import getsizeof
from threading import Lock


class _RootResultCache:
    """Process-local LRU of exact keys and immutable successful results only.

    The byte limit covers the dictionary, keys, result tuples and their floats
    (shared float objects are conservatively counted per occurrence). The fixed
    cache/lock objects and transient caller computations are not cache storage.
    Concurrent misses may compute independently; only access/publication locks.
    """

    def __init__(self, max_entries=1024, max_bytes=256 * 1024):
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries = OrderedDict()
        self._payload_bytes = 0
        self._lock = Lock()

    @staticmethod
    def _size(key, result):
        return getsizeof(key) + getsizeof(result) + sum(map(getsizeof, result))

    def get(self, key):
        with self._lock:
            result = self._entries.get(key)
            if result is not None:
                self._entries.move_to_end(key)
            return result

    def put(self, key, result):
        size = self._size(key, result)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return
            if self._max_entries <= 0 or size + getsizeof(OrderedDict()) > self._max_bytes:
                return
            self._entries[key] = result
            self._payload_bytes += size
            while (len(self._entries) > self._max_entries or
                   self._payload_bytes + getsizeof(self._entries) > self._max_bytes):
                old_key, old_result = self._entries.popitem(last=False)
                self._payload_bytes -= self._size(old_key, old_result)
                if not self._entries:
                    self._entries.clear()  # Release a grown, now-empty table.
                    break


_RESULT_CACHE = _RootResultCache()
_MAX_CACHE_KEY_BYTES = 4096


def _trim(values):
    values = list(values)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _primitive(values):
    """Clear denominators with a positive scale, preserving all signs."""
    values = _trim(tuple(Fraction(x) for x in values))
    denominator = lcm(*(x.denominator for x in values))
    integers = tuple(x.numerator * (denominator // x.denominator) for x in values)
    divisor = gcd(*integers)
    return tuple(x // divisor for x in integers) if divisor else (0,)


def _divrem(a, b):
    a, b = list(map(Fraction, a)), _trim(tuple(map(Fraction, b)))
    quotient = [Fraction(0)] * max(1, len(a) - len(b) + 1)
    while len(a) >= len(b) and any(a):
        offset = len(a) - len(b)
        value = a[-1] / b[-1]
        quotient[offset] = value
        for i, coefficient in enumerate(b):
            a[offset + i] -= value * coefficient
        a = list(_trim(a))
    return _trim(quotient), _trim(a)


def _derivative(p):
    return tuple(i * p[i] for i in range(1, len(p))) or (0,)


def _gcd(a, b):
    while b != (0,):
        _, remainder = _divrem(a, b)
        a, b = b, _primitive(remainder)
    return _primitive(a)


def _sign(p, x):
    """Sign of p(n/d), using integer Horner arithmetic only."""
    n, d = x.numerator, x.denominator
    result, power = p[-1], d
    for coefficient in reversed(p[:-1]):
        result = result * n + coefficient * power
        power *= d
    return (result > 0) - (result < 0)


def _sturm(p):
    sequence = [p, _primitive(_derivative(p))]
    while sequence[-1] != (0,):
        _, remainder = _divrem(sequence[-2], sequence[-1])
        remainder = _primitive(tuple(-x for x in remainder))
        if remainder == (0,):
            break
        sequence.append(remainder)
    return tuple(sequence)


def _variations(sequence, x):
    previous = changes = 0
    for p in sequence:
        sign = _sign(p, x)
        if sign:
            changes += bool(previous and sign != previous)
            previous = sign
    return changes


class _Root:
    """One simple real root in an interval with nonroot endpoints."""

    def __init__(self, polynomial, low, high):
        self.polynomial, self.low, self.high = polynomial, low, high
        self.low_sign = _sign(polynomial, low)

    def refine(self):
        middle = (self.low + self.high) / 2
        sign = _sign(self.polynomial, middle)
        if not sign:
            self.low = self.high = middle
        elif sign == self.low_sign:
            self.low = middle
        else:
            self.high = middle


def _isolate(p):
    """Distinct real roots; rational split roots are factored out exactly."""
    if len(p) <= 1:
        return []
    if len(p) == 2:
        root = Fraction(-p[0], p[1])
        return [_Root(p, root, root)]
    sequence = _sturm(p)
    bound = Fraction(2 + max(abs(x) // abs(p[-1]) for x in p[:-1]))
    vl, vh = _variations(sequence, -bound), _variations(sequence, bound)
    pending = [(-bound, bound, vl, vh)]
    result = []
    while pending:
        low, high, vl, vh = pending.pop()
        if vl == vh:
            continue
        if vl - vh == 1:
            result.append(_Root(p, low, high))
            continue
        middle = (low + high) / 2
        if not _sign(p, middle):
            quotient, remainder = _divrem(p, (-middle, Fraction(1)))
            assert remainder == (0,)
            return [_Root(p, middle, middle)] + _isolate(_primitive(quotient))
        vm = _variations(sequence, middle)
        pending.extend(((middle, high, vm, vh), (low, middle, vl, vm)))
    return result


def _as_float(value):
    try:
        return float(value)
    except OverflowError:
        return inf if value > 0 else -inf


def _rounding_boundary(low, high):
    if nextafter(low, inf) != high:
        return None
    if high == inf:
        return Fraction(low) + Fraction(2) ** 970
    if low == -inf:
        return Fraction(high) - Fraction(2) ** 970
    return (Fraction(low) + Fraction(high)) / 2


def _round_affine(root, scale=Fraction(1), shift=Fraction(0)):
    """Correct round-to-nearest/even of scale*root+shift, including ties."""
    while True:
        low, high = sorted((scale * root.low + shift, scale * root.high + shift))
        if low == high:
            return _as_float(low)
        if low <= 0 <= high and not _sign(root.polynomial, -shift / scale):
            # A rational root need not be dyadic. Its affine image can be
            # exactly zero even when dyadic bisection never hits the root.
            return 0.0
        fl, fh = _as_float(low), _as_float(high)
        if fl.hex() == fh.hex():
            return fl
        boundary = _rounding_boundary(fl, fh)
        if boundary is not None and not _sign(root.polynomial, (boundary - shift) / scale):
            return _as_float(boundary)
        root.refine()


def _value(p, x):
    value = Fraction(p[-1])
    for coefficient in reversed(p[:-1]):
        value = value * x + coefficient
    return value


def _algebraic_sign(root, q):
    """Sign of quadratic q at the sole real root of a square-free cubic.

    The GCD detects exact equality before interval refinement, including a
    cutoff tie that would otherwise keep straddling zero indefinitely.
    """
    common = _gcd(root.polynomial, _primitive(q))
    if len(common) > 1 and _isolate(common):
        return 0
    while True:
        samples = [_value(q, root.low), _value(q, root.high)]
        vertex = -q[1] / (2 * q[2])
        if root.low < vertex < root.high:
            samples.append(_value(q, vertex))
        if min(samples) > 0:
            return 1
        if max(samples) < 0:
            return -1
        root.refine()


def near_real_roots(coefficients, imaginary_tolerance=1.0e-10):
    """Return canonical real parts admitted by rounded |imag| <= tolerance.

    Coefficients are ascending, already normalized and trimmed by the caller.
    Degree is at most three. The real components are correctly rounded roots
    of these exact binary64 coefficients. This intentionally does not retain
    a particular LAPACK implementation's approximate root bit patterns.
    """
    values = tuple(float(x) for x in coefficients)
    if not values or not all(isfinite(x) for x in values):
        raise ValueError('polynomial coefficients must be finite and nonempty')
    p = _primitive(values)
    if len(p) > 4:
        raise ValueError('B3 projection polynomial degree exceeds three')
    tolerance = float(imaginary_tolerance)
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError('imaginary tolerance must be finite and nonnegative')
    # Keep all conversions, primitive construction, degree and tolerance checks
    # above the lookup, in their original order, even for a warm result. Key the
    # untrimmed converted sequence: length and every signed-zero bit matter.
    key = None
    if 8 * (len(values) + 1) <= _MAX_CACHE_KEY_BYTES:
        key = b''.join(struct.pack('>d', value) for value in (*values, tolerance))
        cached = _RESULT_CACHE.get(key)
        if cached is not None:
            return cached
    result = _near_real_roots_uncached(p, tolerance)
    if key is not None:
        _RESULT_CACHE.put(key, result)
    return result


def _near_real_roots_uncached(p, tolerance):
    """The unchanged canonical algorithm, after the public input checks."""
    if len(p) <= 1:
        return ()
    next_tolerance = nextafter(tolerance, inf)
    cutoff = _rounding_boundary(tolerance, next_tolerance)
    inclusive = not (struct.unpack('>Q', struct.pack('>d', tolerance))[0] & 1)
    divisor = _gcd(p, _derivative(p))
    square_free, remainder = _divrem(p, divisor)
    assert remainder == (0,)
    square_free = _primitive(square_free)
    real = _isolate(square_free)
    result = [_round_affine(root) for root in real]
    if len(p) == 3 and not real:
        # a*x^2+b*x+c: u=-b/(2a), v^2=(4ac-b^2)/(4a^2).
        c, b, a = p
        imaginary_square = Fraction(4*a*c-b*b, 4*a*a)
        delta = imaginary_square - cutoff*cutoff
        if delta < 0 or (not delta and inclusive):
            result.append(_as_float(Fraction(-b, 2*a)))
    elif len(p) == 4 and len(square_free) == 4 and len(real) == 1:
        # x^3+B*x^2+C*x+D=(x-r)*((x-u)^2+v^2):
        # u=-(B+r)/2 and 4v^2=3r^2+2Br+4C-B^2.
        b, c = Fraction(p[2], p[3]), Fraction(p[1], p[3])
        q = (4*c-b*b-4*cutoff*cutoff, 2*b, Fraction(3))
        sign = _algebraic_sign(real[0], q)
        if sign < 0 or (not sign and inclusive):
            result.append(_round_affine(real[0], Fraction(-1, 2), -b/2))
    return tuple(sorted(set(result)))
