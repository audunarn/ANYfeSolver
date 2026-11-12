"""Exact inertia of a represented binary64 symmetric shifted pencil.

This is arithmetic authority for H - shift*M as supplied, not an interval
certificate for continuum mechanics or the pre-rounding source operator.
"""
from time import monotonic
import numpy as np
from ._native_dyadic_interval_inertia import interval_inertia


def exact_inertia(h, mass, shift, check=lambda: None, *, dimension_limit=64):
    started=monotonic()
    def guard():
        check()
        if monotonic()-started>30.: raise TimeoutError('exact shifted inertia deadline')
    guard()
    for precision in (256,512):
        answer=interval_inertia(h,mass,shift,guard,precision=precision,dimension_limit=dimension_limit)
        guard()
        if answer is not None: return answer
    return integer_inertia(h,mass,shift,guard,dimension_limit=dimension_limit)


def integer_inertia(h, mass, shift, check=lambda: None, *, dimension_limit=64):
    started = monotonic()
    def guard():
        check()
        if monotonic()-started > 30.:
            raise TimeoutError('exact shifted inertia deadline')
    guard()
    if type(dimension_limit) is not int or dimension_limit not in (64,96,160):
        raise ValueError('explicit admitted exact-inertia dimension limit required')
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= len(h) <= dimension_limit
            or mass.shape != h.shape or not np.isfinite(h).all()
            or not np.isfinite(mass).all() or not np.isfinite(shift)
            or not np.array_equal(h, h.T) or not np.array_equal(mass, mass.T)):
        raise ValueError('bounded finite symmetric shifted pencil required')
    # Binary64 inputs are dyadic. Clear their denominators with ONE positive
    # scale, preserving exactly H - shift*M (not its rounded subtraction).
    sn, sd = float(shift).as_integer_ratio()
    ratios = []
    denominator = 1
    for row, other in zip(h, mass):
        guard(); converted = []
        for x, y in zip(row, other):
            xn, xd = float(x).as_integer_ratio()
            yn, yd = float(y).as_integer_ratio()
            d = max(xd, sd*yd)  # all denominators are powers of two
            converted.append((xn*(d//xd)-sn*yn*(d//(sd*yd)), d))
            denominator = max(denominator, d)
        ratios.append(converted)
    a = [[v*(denominator//d) for v, d in row] for row in ratios]
    previous = 1
    positive = negative = zeros = 0
    while a:
        guard(); n = len(a)
        pivot = max(range(n), key=lambda i: abs(a[i][i]))
        if not a[pivot][pivot]:
            # All diagonals vanish. A unimodular congruence e_i <- e_i+e_j
            # makes diagonal 2*a_ij nonzero without dividing by a zero pivot.
            # Congruence preserves inertia and integer-minor divisibility.
            pair = next(((i, j) for i in range(n) for j in range(i+1, n) if a[i][j]), None)
            if pair is None:
                zeros += n; break
            pivot, j = pair
            a[pivot] = [x+y for x, y in zip(a[pivot], a[j])]
            for row in a: row[pivot] += row[j]
        order = [pivot]+[i for i in range(n) if i != pivot]
        a = [[a[i][j] for j in order] for i in order]
        d = a[0][0]
        # Invariant: a/previous is the remaining exact Schur complement
        # of the positively scaled input, after congruences. Thus the sign
        # is d/previous, NOT d alone. previous is allowed to be negative.
        positive += int((d > 0) == (previous > 0))
        negative += int((d > 0) != (previous > 0))
        reduced = [[0]*(n-1) for _ in range(n-1)]
        for i in range(1, n):
            guard()
            for j in range(i, n):
                value, remainder = divmod(d*a[i][j]-a[i][0]*a[0][j], previous)
                if remainder:
                    raise ArithmeticError('fraction-free inertia lost exact divisibility')
                reduced[i-1][j-1] = reduced[j-1][i-1] = value
        a = reduced; previous = d
    guard()
    return positive, negative, zeros
