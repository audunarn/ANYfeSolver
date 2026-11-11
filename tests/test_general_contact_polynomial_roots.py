"""Independent exact/factored and high-precision root-rounding examples."""

from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
import pytest

from anyfesolver.general_contact.polynomial_roots import near_real_roots


def _decimal_real_root(coefficients, initial):
    # Independent high-precision Newton oracle for well-separated simple roots.
    # These examples converge far beyond the precision needed for binary64.
    with localcontext() as context:
        context.prec = 180
        p = [Decimal.from_float(float(x)) for x in coefficients]
        x = Decimal(initial)
        for _ in range(80):
            value, derivative = p[-1], Decimal(0)
            for coefficient in reversed(p[:-1]):
                derivative = derivative*x + value
                value = value*x + coefficient
            update = x - value/derivative
            if update == x:
                break
            x = update
        assert abs(value) < Decimal('1e-150')
        return float(x)


def _decimal_quadratic(coefficients):
    with localcontext() as context:
        context.prec = 180
        c, b, a = [Decimal.from_float(float(x)) for x in coefficients]
        discriminant = (b*b-4*a*c).sqrt()
        return tuple(sorted((float((-b-discriminant)/(2*a)),
                             float((-b+discriminant)/(2*a)))))


@pytest.mark.parametrize(('coefficients', 'expected'), [
    ([0.], ()), ([3.], ()), ([0., 0., 0., 0.], ()),
    ([-.25, 1., 0., 0.], (.25,)),
    ([0., .5, -1.5, 1.], (0., .5, 1.)),
    ([2., -3., 0., 1.], (-2., 1.)),
    ([-.125, .75, -1.5, 1.], (.5,)),
    ([1., -2., 1.], (1.,)),
    ([-2., 1., 1.], (-2., 1.)),
    ([1., 0., 1.], ()),
])
def test_exact_factored_and_degree_reduced_polynomials(coefficients, expected):
    assert near_real_roots(coefficients) == expected


def test_close_distinct_cubic_roots_are_not_merged_before_caller_filtering():
    delta = 2.**-25
    coefficients = [-.125 + .5*delta*delta, .75-delta*delta, -1.5, 1.]
    assert near_real_roots(coefficients) == (.5-delta, .5, .5+delta)


@pytest.mark.parametrize(('coefficients', 'initials'), [
    ([-2., 0., 0., 1.], ('1',)),
    ([1., -3., 0., 1.], ('-2', '0', '2')),
])
def test_irrational_cubic_roots_against_independent_decimal_oracle(coefficients, initials):
    expected = tuple(sorted(_decimal_real_root(coefficients, x) for x in initials))
    assert near_real_roots(coefficients) == expected


def test_quadratic_cancellation_preserves_the_small_root():
    coefficients = [1., 1.e12, 1.]
    assert near_real_roots(coefficients) == _decimal_quadratic(coefficients)


def test_binary64_rounding_ties_and_signed_underflow():
    tiny = math.ulp(0.)
    # Exact linear roots at half-subnormal ties, including the odd-to-even tie.
    assert near_real_roots([-tiny, 2.])[0].hex() == '0x0.0p+0'
    assert near_real_roots([tiny, 2.])[0].hex() == '-0x0.0p+0'
    assert near_real_roots([-3*tiny, 2.])[0] == 2*tiny
    # Roots that round to the two physical endpoints remain exactly endpoints.
    assert near_real_roots([-tiny, 2.])[0] == 0.
    assert near_real_roots([-1., 1.])[0] == 1.


@pytest.mark.parametrize('degree', [2, 3])
def test_near_real_conjugates_on_both_sides_of_rounded_imaginary_cutoff(degree):
    threshold = 1.e-10
    midpoint = (Fraction(threshold) + Fraction(math.nextafter(threshold, math.inf))) / 2
    square = midpoint*midpoint
    near = float(square)
    below = math.nextafter(near, -math.inf)
    above = math.nextafter(near, math.inf)
    assert Fraction(below) < square < Fraction(above)
    for imaginary_square, admitted in ((below, True), (above, False)):
        coefficients = ([imaginary_square, 0., 1.] if degree == 2 else
                        [-imaginary_square, imaginary_square, -1., 1.])
        expected = ((0.,) if admitted else ()) if degree == 2 else ((0., 1.) if admitted else (1.,))
        assert near_real_roots(coefficients) == expected


def test_near_real_cubic_conjugate_real_part_is_not_the_real_root():
    # (x-1)*(x^2+2^-80) has a real root at 1 and an admitted conjugate pair at ±i*2^-40.
    square = 2.**-80
    assert near_real_roots([-square, square, -1., 1.]) == (0., 1.)


@pytest.mark.parametrize(('leading', 'real_root'), [(3., Fraction(1, 3)), (.75, Fraction(4, 3))])
def test_zero_conjugate_real_part_with_nondyadic_real_root(leading, real_root):
    square = 2.**-80
    # (leading*x-1)*(x^2+2^-80); .75 also keeps coefficients normalized to max1.
    assert near_real_roots([-square, leading*square, -1., leading]) == (0., float(real_root))


def test_small_nonzero_real_part_of_admitted_conjugates():
    # (x-2^-39)*((x-2^-40)^2+2^-82), all coefficients exactly representable.
    coefficients = [-5*2.**-121, 21*2.**-82, -2.**-38, 1.]
    assert near_real_roots(coefficients) == (2.**-40, 2.**-39)


def test_algebraic_cutoff_equality_and_nonreal_common_factor():
    from anyfesolver.general_contact.polynomial_roots import _algebraic_sign, _isolate
    # p=(x-1)(x^2+1), q=(x-1)(3x+1) share the sole real root exactly.
    root = _isolate((-1, 1, -1, 1))[0]
    assert _algebraic_sign(root, (Fraction(-1), Fraction(-2), Fraction(3))) == 0
    # A common nonreal factor must not be mistaken for equality at the real root.
    assert _algebraic_sign(root, (Fraction(1), Fraction(0), Fraction(1))) == 1


def test_exact_imaginary_threshold_and_rejected_complex_pair():
    # Optional thresholds exercise the same admission rule without cancellation.
    assert near_real_roots([1., 0., 1.], imaginary_tolerance=1.) == (0.,)
    assert near_real_roots([-1., 1., -1., 1.], imaginary_tolerance=1.) == (0., 1.)
    assert near_real_roots([-1., 1., -1., 1.], imaginary_tolerance=.5) == (1.,)


def test_captured_cross_host_coefficients_and_projection(monkeypatch):
    from anyfesolver.general_contact.beam_curve_pair import closest_on_b3
    coefficients = [float.fromhex(x) for x in (
        '0x1.f204e4472b959p-1', '-0x1.0000000000000p+0',
        '-0x1.c43308c4b6156p-25', '-0x1.a3debef364c7ep-15')]
    expected = _decimal_real_root(coefficients, '1')
    assert near_real_roots(coefficients) == (expected,)
    point = np.array([float.fromhex(x) for x in (
        '0x1.f8a9d57f3aa24p+0', '-0x1.af49ed0f0915ap-9', '0x0.0p+0')])
    target = np.array([[float.fromhex(x) for x in row] for row in (
        ('0x1.000068da61013p+0', '0x1.ac32614bc4a48p-4', '0x0.0p+0'),
        ('0x1.7ffffb4948291p+0', '0x1.a087e3d370a06p-4', '0x0.0p+0'),
        ('0x1.ffff07cdebad7p+0', '0x1.8a9ff922ebcc8p-4', '0x0.0p+0'))])
    def forbidden(*args, **kwargs):
        raise AssertionError('projection must not dispatch to Polynomial.roots')
    monkeypatch.setattr(np.polynomial.Polynomial, 'roots', forbidden)
    assert closest_on_b3(point, target).minimizers == (expected,)


@pytest.mark.parametrize('coefficients', [[], [math.inf, 1.], [1., math.nan], [1., 0., 0., 0., 1.]])
def test_invalid_coefficients_are_refused(coefficients):
    with pytest.raises(ValueError):
        near_real_roots(coefficients)
