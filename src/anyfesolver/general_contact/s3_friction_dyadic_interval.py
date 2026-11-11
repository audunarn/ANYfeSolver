"""Opt-in outward rational interval compression; no floating endpoints."""
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction

_BITS=ContextVar('s3_material_interval_significand_bits',default=None)
METHOD='outward_dyadic_significand160_v1'


def _bounds(value,bits):
    value=Fraction(value)
    if value==0:return value,value
    numerator=value.numerator;denominator=value.denominator
    absolute=abs(numerator);exponent=absolute.bit_length()-denominator.bit_length()
    below=(absolute < (denominator << exponent)) if exponent>=0 else ((absolute << -exponent)<denominator)
    if below:exponent-=1
    shift=exponent-(bits-1)
    if shift>=0:
        scaled_denominator=denominator << shift
        low=numerator//scaled_denominator;high=-((-numerator)//scaled_denominator)
        return Fraction(low << shift),Fraction(high << shift)
    scaled_numerator=numerator << -shift
    low=scaled_numerator//denominator;high=-((-scaled_numerator)//denominator)
    unit=1 << -shift
    return Fraction(low,unit),Fraction(high,unit)


def compress(interval):
    bits=_BITS.get()
    if bits is None:return interval
    if interval[0]>interval[1]:raise ValueError('reversed rational interval')
    return _bounds(interval[0],bits)[0],_bounds(interval[1],bits)[1]


@contextmanager
def outward_dyadic_160():
    token=_BITS.set(160)
    try:yield
    finally:_BITS.reset(token)
