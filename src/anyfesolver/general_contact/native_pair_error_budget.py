"""Exact nonrenewable opt-in whole-pair quadrature error accounting."""
from fractions import Fraction
from math import isfinite, nextafter, inf


def within_reserve(used, bounds, limits, *, half=False):
    """Compare prospective sums exactly, including positive sub-ulp charges."""
    if any(len(row) != 3 for row in (used, bounds, limits)):
        raise ValueError('three complete error components required')
    if any(not isfinite(float(v)) or float(v) < 0 for row in (used, bounds) for v in row):
        raise ValueError('finite nonnegative error accounting required')
    if any(not isfinite(float(v)) or float(v) <= 0 for v in limits):
        raise ValueError('finite positive error limits required')
    divisor = 2 if half else 1
    return all(Fraction(float(a))+Fraction(float(b)) <= Fraction(float(t))/divisor
               for a,b,t in zip(used,bounds,limits))


class PooledQuadratureBudget:
    def __init__(self, limits):
        if len(limits) != 3 or any(not isfinite(float(v)) or float(v) <= 0 for v in limits):
            raise ValueError('finite positive complete error limits required')
        self.limits = tuple(Fraction(float(v))/2 for v in limits)
        self.used = (Fraction(0),)*3
        self.charges = 0

    def _values(self, bounds):
        if len(bounds) != 3 or any(not isfinite(float(v)) or float(v) < 0 for v in bounds):
            raise ValueError('finite nonnegative complete error bounds required')
        return tuple(Fraction(float(v)) for v in bounds)

    def admits(self, bounds):
        return all(u+v <= limit for u,v,limit in zip(self.used,self._values(bounds),self.limits))

    def charge(self, bounds):
        values = self._values(bounds)
        if not self.admits(bounds):
            raise MemoryError('pooled whole-pair quadrature reserve exhausted')
        self.used = tuple(u+v for u,v in zip(self.used,values)); self.charges += 1

    def remaining(self):
        return tuple(nextafter(float(limit-used),-inf)
            if Fraction(float(limit-used)) > limit-used else float(limit-used)
            for limit,used in zip(self.limits,self.used))

    def report(self):
        return dict(charges=self.charges,used=tuple(nextafter(float(v),inf)
            if Fraction(float(v)) < v else float(v) for v in self.used),
            remaining=self.remaining(),semantics='ONE_HALF_GLOBAL_PHYSICAL30_E_F_K_POOL_EXACT_ONCE')
