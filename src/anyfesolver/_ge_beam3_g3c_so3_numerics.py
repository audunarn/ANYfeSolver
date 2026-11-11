"""Unrouted private G3c SO(3) evaluation successor, not graph qualification.

Same Exp/Log maps and Jet class; stable scalar evaluation only. No existing
caller imports this module. The shared arithmetic source remains immutable.
"""
from math import acos, factorial, isfinite, sqrt as scalar_sqrt

from ._ge_beam3_mixed_ad import (
    Jet2, RotationDomainError, identity, matmul, math_cos_limit, sin,
    skew, sqrt, unary,
)

POLICY = 'GE_BEAM3_G3C_ISOLATED_SO3_NUMERICS_V1'
SINC = tuple((-1.0)**n / factorial(2*n+1) for n in range(17))
COSC = tuple((-1.0)**n / factorial(2*n+2) for n in range(17))


def _log_coefficients():
    values = [1.0]
    for n in range(1, 41):
        values.append(values[-1] * n / (2*n+1))
    return tuple(values)


LOG = _log_coefficients()


def _polynomial(coefficients, x):
    """One polynomial and its two exact algebraic derivatives, via Horner."""
    value, first, second = coefficients[-1], 0.0, 0.0
    for coefficient in reversed(coefficients[:-1]):
        second = x*second + 2.0*first
        first = x*first + value
        value = x*value + coefficient
    return value, first, second


def _exp_coefficients(argument):
    if not isfinite(argument.value) or argument.value < 0.0:
        raise ValueError('finite nonnegative squared rotation required')
    if argument.value <= 1.0:
        return (unary(argument, *_polynomial(SINC, argument.value)),
                unary(argument, *_polynomial(COSC, argument.value)))
    theta = sqrt(argument)
    half_sinc = sin(theta/2.0)/theta
    return sin(theta)/theta, 2.0*half_sinc*half_sinc


def _log_factor(cosine):
    c = cosine.value
    if not isfinite(c) or not math_cos_limit() < c:
        raise RotationDomainError('relative rotation must remain below 0.9*pi; refine or cut back')
    if c >= 0.5:
        value, first, second = _polynomial(LOG, 1.0-c)
        return unary(cosine, value, -first, second)
    angle = acos(c)
    sine = scalar_sqrt((1.0-c)*(1.0+c))
    return unary(cosine, angle/sine,
                 -1.0/sine**2 + angle*c/sine**3,
                 -3.0*c/sine**4 + angle/sine**3 + 3.0*angle*c*c/sine**5)


def so3_exp(vector):
    size = vector[0].gradient.size
    argument = sum((entry*entry for entry in vector), start=Jet2.constant(0.0, size))
    a, b = _exp_coefficients(argument)
    cross = skew(vector)
    square = matmul(cross, cross)
    result = identity(size)
    return [[result[i][j] + a*cross[i][j] + b*square[i][j]
             for j in range(3)] for i in range(3)]


def so3_log(matrix):
    cosine = (matrix[0][0] + matrix[1][1] + matrix[2][2] - 1.0)/2.0
    factor = _log_factor(cosine)
    return [factor*entry for entry in (
        (matrix[2][1]-matrix[1][2])/2.0,
        (matrix[0][2]-matrix[2][0])/2.0,
        (matrix[1][0]-matrix[0][1])/2.0,
    )]
