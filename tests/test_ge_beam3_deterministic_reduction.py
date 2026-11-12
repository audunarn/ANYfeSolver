"""Independent arithmetic regression tests for the deterministic reductions.

Oracles are deliberately independent of the module under test: decimal
libmpdec arithmetic for correctly rounded square roots and sequential FMA
chains, exact Fraction targets for midpoint ties, and hand-derived exact
spectral values for rank/zero and dyadic cases. The review counterexamples
(tiny/large dyadics, nonfinite, scale extremes) are executed here only after
the bounded-bracket repair; each case is fast and cannot hang.
"""
import struct
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from math import fsum, nextafter

import numpy as np
import pytest

from anyfesolver._ge_beam3_deterministic_reduction import (
    _round_sqrt,
    _sqrt_fraction,
    deterministic_dot,
    deterministic_norm,
    deterministic_spectral_norm)

@pytest.fixture(autouse=True)
def isolated_decimal_context():
    with localcontext() as context:
        context.prec = 80
        yield


def test_fma_overflow_accumulator_and_signed_zero():
    from math import copysign, inf
    from anyfesolver._ge_beam3_deterministic_reduction import _fma
    assert deterministic_dot([2.**1023, 2.**1023], [2., -2.]) == inf
    assert deterministic_dot([-2.**1023, 2.**1023], [2., 2.]) == -inf
    assert copysign(1., _fma(-0., 1., -0.)) == -1.
    assert copysign(1., _fma(-1., 1., 1.)) == 1.
    with pytest.raises(ValueError, match='finite'):
        deterministic_dot([inf], [1.])


def decimal_fma_chain(x, y):
    """Sequential correctly rounded FMA chain via decimal exact products."""
    total = Decimal(0)
    for a, b in zip(x, y):
        total = Decimal(float(Decimal(float(a))*Decimal(float(b)) + total))
    return float(total)


def decimal_vector_norm(x):
    """Correctly rounded 2-norm oracle: exact decimal sum of squares."""
    total = sum((Decimal(float(v))**2 for v in np.asarray(x).ravel()), Decimal(0))
    if total == 0:
        return 0.0
    return float(total.sqrt())


def decimal_matrix_norm_2x2(a):
    """Independent 2x2 spectral norm: exact decimal Gram + closed form."""
    a = np.asarray(a, dtype=float)
    rows, cols = a.shape
    flat = a.ravel().tolist()
    if cols <= rows:
        g = [[sum(Decimal(flat[r*cols+c1])*Decimal(flat[r*cols+c2])
                  for r in range(rows)) for c2 in range(cols)] for c1 in range(cols)]
    else:
        g = [[sum(Decimal(flat[r1*cols+c])*Decimal(flat[r2*cols+c])
                  for c in range(cols)) for r2 in range(rows)] for r1 in range(rows)]
    dim = len(g)
    if dim == 1:
        lam = g[0][0]
    else:
        trace = g[0][0]+g[1][1]
        disc = (g[0][0]-g[1][1])**2+Decimal(4)*g[0][1]*g[1][0]
        lam = (trace+disc.sqrt())/2
    if lam == 0:
        return 0.0
    return float(lam.sqrt())


def rounded_sqrt_of_fraction(target):
    """Public tie path exercised through the module's own bracket helper."""
    target = Fraction(target)
    return _round_sqrt(lambda x: x*x >= target,
                       lambda x: x*x == target,
                       float(_sqrt_fraction(target)))


@pytest.mark.parametrize('exponent', [-200, -100, -1, 0, 1, 100, 200, 600])
def test_dyadic_single_entry_matrices(exponent):
    value = 2.0**exponent
    assert deterministic_spectral_norm([[value]]) == abs(value)
    assert deterministic_spectral_norm([[value, 0.0], [0.0, 0.0]]) == abs(value)


def test_review_counterexample_discriminant_scale():
    # Gram eigenvalue 2^-199; the norm sqrt(2)*2^-100 is not representable.
    a = [[2.0**-100, 2.0**-100], [0.0, 0.0]]
    expected = float(Decimal(2).sqrt()*Decimal(2)**-100)
    assert deterministic_spectral_norm(a) == expected


@pytest.mark.parametrize('exponent', [-600, -300, 300, 600])
def test_vector_extreme_dyadics_do_not_underflow_or_overflow(exponent):
    value = 2.0**exponent
    # The recorded ddot chain would underflow to 0.0 or overflow to inf.
    assert deterministic_spectral_norm([value]) == abs(value)
    assert deterministic_spectral_norm([value, 0.0, 0.0]) == abs(value)


def test_vector_pair_extreme_dyadic_rounding():
    expected = float((Decimal(2)*Decimal(2.0**600)**2).sqrt())
    assert deterministic_spectral_norm([2.0**600, 2.0**600]) == expected
    expected = float((Decimal(2)*Decimal(2.0**-600)**2).sqrt())
    assert deterministic_spectral_norm([2.0**-600, 2.0**-600]) == expected


def test_rank_and_zero_matrices():
    assert deterministic_spectral_norm(np.zeros((2, 2))) == 0.0
    assert deterministic_spectral_norm(np.zeros((3, 3))) == 0.0
    assert deterministic_spectral_norm(np.zeros(4)) == 0.0
    assert deterministic_spectral_norm([[3.0, 0.0], [4.0, 0.0]]) == 5.0
    assert deterministic_spectral_norm([[1.0, 2.0], [2.0, 4.0]]) == 5.0
    assert deterministic_spectral_norm(np.eye(3)) == 1.0
    assert deterministic_spectral_norm([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]) == 1.0
    # Rank one outer product with exact dyadic factor norms 3 and 1.
    assert deterministic_spectral_norm([[1.0, 0.0, 0.0],
                                        [2.0, 0.0, 0.0],
                                        [2.0, 0.0, 0.0]]) == 3.0
    assert deterministic_spectral_norm([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]) == 1.0


def test_nonfinite_input_refused():
    for bad in ([[float('inf')]], [[float('nan')]], [float('inf')],
                [float('nan')], [[1.0, float('inf')], [0.0, 1.0]]):
        with pytest.raises(ValueError):
            deterministic_spectral_norm(bad)


def test_dimension_and_shape_contracts():
    with pytest.raises(ValueError):
        deterministic_spectral_norm(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        deterministic_spectral_norm(np.zeros((2, 2, 2)))
    assert deterministic_spectral_norm([]) == 0.0
    assert deterministic_spectral_norm(np.zeros((0, 3))) == 0.0
    # Wide and tall flat matrices only need the one-dimensional Gram.
    assert deterministic_spectral_norm(np.zeros((1, 8))) == 0.0
    assert deterministic_spectral_norm([[3.0, 4.0]]) == 5.0
    assert deterministic_spectral_norm([[3.0], [4.0]]) == 5.0


def test_finite_upper_bound_exact_and_refusal():
    largest = sys.float_info.max
    assert deterministic_spectral_norm([[largest]]) == largest
    assert deterministic_spectral_norm([[largest], [0.0]]) == largest
    # sqrt(2)*max overflows binary64: refused, never inf.
    with pytest.raises(ValueError):
        deterministic_spectral_norm([[largest, largest], [0.0, 0.0]])
    with pytest.raises(ValueError):
        deterministic_spectral_norm([largest, largest])


def test_subnormal_results_round_correctly():
    tiny = nextafter(0.0, 1.0)  # 2^-1074
    assert deterministic_spectral_norm([[tiny]]) == tiny
    expected = float((Decimal(2)*Decimal(tiny)**2).sqrt())
    assert deterministic_spectral_norm([tiny, tiny]) == expected
    assert deterministic_spectral_norm([[tiny, tiny], [0.0, 0.0]]) == expected


def test_midpoint_tie_rounds_half_to_even():
    # Midpoint of 1.0 (even) and 1+2^-52 (odd): tie selects 1.0.
    low, high = 1.0, nextafter(1.0, float('inf'))
    target = ((Fraction(low)+Fraction(high))/2)**2
    assert rounded_sqrt_of_fraction(target) == low
    # Midpoint of 1+2^-52 (odd) and 1+2^-51 (even): tie selects the even high.
    low = nextafter(1.0, float('inf'))
    high = nextafter(low, float('inf'))
    target = ((Fraction(low)+Fraction(high))/2)**2
    assert rounded_sqrt_of_fraction(target) == high
    # Subnormal tie between 0.0 (even) and the minimum subnormal.
    tiny = nextafter(0.0, 1.0)
    target = (Fraction(tiny)/2)**2
    assert rounded_sqrt_of_fraction(target) == 0.0


def test_sqrt_fraction_relative_precision_across_scales():
    for exponent in (-1200, -199, -1, 0, 1, 199, 1200):
        target = Fraction(2)**(2*exponent)
        root = _sqrt_fraction(target)
        relative = abs(root - Fraction(2)**exponent)/Fraction(2)**exponent
        assert relative < Fraction(1, 2**100)
    # The review's lost-discriminant case keeps its scale.
    root = _sqrt_fraction(Fraction(1, 2**199))
    assert abs(root*root - Fraction(1, 2**199)) < Fraction(1, 2**300)


def test_recorded_fma_chain_matches_decimal_oracle():
    rng = np.random.default_rng(20261012)
    for _ in range(25):
        n = int(rng.integers(1, 65))
        x = rng.standard_normal(n)*10.0**rng.integers(-8, 9)
        y = rng.standard_normal(n)*10.0**rng.integers(-8, 9)
        assert deterministic_dot(x, y) == decimal_fma_chain(x, y)
    assert deterministic_dot([3.0, 4.0], [3.0, 4.0]) == 25.0
    assert deterministic_norm([3.0, 4.0]) == 5.0
    with pytest.raises(ValueError):
        deterministic_dot([1.0, 2.0], [1.0])


def test_recorded_chain_extreme_scale_semantics():
    # Recorded-chain contract: extreme scales underflow/overflow like ddot.
    assert deterministic_dot([2.0**-600], [2.0**-600]) == 0.0
    assert deterministic_dot([2.0**600], [2.0**600]) == float('inf')
    assert deterministic_norm([2.0**-600]) == 0.0
    assert deterministic_norm([2.0**600]) == float('inf')


def test_random_matrices_agree_with_independent_oracles():
    rng = np.random.default_rng(20261012)
    for _ in range(40):
        rows = int(rng.integers(1, 4))
        cols = int(rng.integers(1, 4))
        a = rng.standard_normal((rows, cols))*10.0**rng.integers(-6, 7)
        got = deterministic_spectral_norm(a)
        if min(rows, cols) <= 2:
            assert got == decimal_matrix_norm_2x2(a)
        else:
            reference = float(np.linalg.norm(a, 2))
            assert abs(got-reference) <= 4*np.spacing(abs(reference))
        vector = a.ravel()
        assert deterministic_spectral_norm(vector) == decimal_vector_norm(vector)
