"""Exact cancellation and refusal controls for the recorded gap work maps."""
from fractions import Fraction
import math

import numpy as np
import pytest

from anyfesolver.general_contact import native_replay_arithmetic as arithmetic


@pytest.mark.parametrize('fallback', [False, True])
def test_vector_block_and_tail_keep_distinct_exact_cancellation(monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(arithmetic, '_fma', arithmetic._portable_fma)
    epsilon = 2.**-27
    matrix = np.tile((1.+epsilon, -1., 0.), (9, 1))
    vector = np.array((1.-epsilon, 1., .5))
    result = arithmetic.gap_vector_product(matrix, vector)
    # (1+e)(1-e)-1 is exactly -e**2 when fused; rounding its
    # product before the subtraction instead gives +0 (ties to even).
    assert [float(x).hex() for x in result[:8]] == [(-epsilon**2).hex()]*8
    assert result[8].hex() == (0.).hex()
    np.testing.assert_array_equal(matrix, np.tile((1.+epsilon, -1., 0.), (9, 1)))
    np.testing.assert_array_equal(vector, (1.-epsilon, 1., .5))


@pytest.mark.parametrize('fallback', [False, True])
def test_matrix_chain_cancellation_subnormals_and_signed_zero(monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(arithmetic, '_fma', arithmetic._portable_fma)
    epsilon = 2.**-27
    tiny = float.fromhex('0x0.0000000000001p-1022')
    left = np.array(((-1., 1.+epsilon, 0.), (tiny, tiny, tiny), (-0., -0., -0.)))
    right = np.array(((1., .5), (1.-epsilon, .5), (0., .5)))
    result = arithmetic.gap_matrix_product(left, right)
    expected = np.empty(result.shape)
    for i, row in enumerate(left):
        for j, column in enumerate(right.T):
            accumulator = 0.
            for a, b in zip(row, column):
                accumulator = float(Fraction(float(a))*Fraction(float(b)) + Fraction(accumulator))
            expected[i, j] = accumulator
    assert result.tobytes() == expected.tobytes()
    assert result[0, 0] == -epsilon**2
    assert result[1, 1].hex() == (0.).hex()
    assert math.copysign(1., result[2, 0]) == 1.


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_operands_refused(bad):
    matrix = np.ones((9, 3)); matrix[0, 0] = bad
    with pytest.raises(ValueError, match='finite'):
        arithmetic.gap_vector_product(matrix, np.ones(3))
    with pytest.raises(ValueError, match='finite'):
        arithmetic.gap_matrix_product(matrix, np.ones((3, 2)))
    with pytest.raises(ValueError, match='finite'):
        arithmetic.gap_vector_product(np.ones((9, 3)), np.array((bad, 1., 1.)))
    with pytest.raises(ValueError, match='finite'):
        arithmetic.gap_matrix_product(np.ones((2, 3)), np.full((3, 2), bad))


def test_wrong_work_map_shape_refused():
    with pytest.raises(ValueError, match='9x3'):
        arithmetic.gap_vector_product(np.ones((8, 3)), np.ones(3))
    with pytest.raises(ValueError, match='9x3'):
        arithmetic.gap_vector_product(np.ones((9, 3)), np.ones(4))
    with pytest.raises(ValueError, match='three inner'):
        arithmetic.gap_matrix_product(np.ones((3, 2)), np.ones((2, 3)))


def test_missing_runtime_symbol_keeps_exact_fallback(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError('unavailable runtime')
    monkeypatch.delattr(arithmetic.math, 'fma', raising=False)
    monkeypatch.setattr(arithmetic.ctypes, 'CDLL', unavailable)
    assert arithmetic._load_fma() is arithmetic._portable_fma


@pytest.mark.parametrize('station', [0., .37, 1.])
def test_curve_position_and_derivatives_use_fixed_fused_chain(station):
    from anyfesolver.general_contact.beam_curve_pair import _curve
    epsilon = 2.**-27
    positions = np.array(((1.+epsilon, -1., 0.),
                          (-1., 1.-epsilon, 2.**-1022),
                          (epsilon, -epsilon, -0.)))
    before = positions.tobytes()
    t = station
    # Independent Q2 position, tangent and second-derivative work maps.
    weights = ((2*t*t-3*t+1, 4*t-4*t*t, 2*t*t-t),
               (4*t-3, 4-8*t, 4*t-1), (4., -8., 4.))
    expected = []
    for row in weights:
        values = []
        for column in positions.T:
            value = 0.
            for a, b in zip(row, column):
                value = float(Fraction(float(a))*Fraction(float(b)) + Fraction(value))
            values.append(value)
        expected.append(values)
    assert np.asarray(_curve(positions, station)).tobytes() == np.asarray(expected).tobytes()
    assert positions.tobytes() == before


def test_endpoint_projection_cuts_use_fixed_coefficient_reductions():
    from decimal import Decimal, localcontext
    from anyfesolver.general_contact.beam_curve_pair import _curve, _endpoint_projection_cuts
    source = np.array(((0., .11, .01), (1., .1101, .011), (2., .11, .01)))
    target = np.array(((1., 0., 0.), (2., .01, .002), (3., 0., 0.)))
    linear = -3*source[0]+4*source[1]-source[2]
    quadratic = 2*source[0]-4*source[1]+2*source[2]
    expected = []
    for station in (0., 1.):
        point, tangent, _ = _curve(target, station)
        coefficients = []
        for vector in (source[0]-point, linear, quadratic):
            value = 0.
            for a, b in zip(vector, tangent):
                value = float(Fraction(float(a))*Fraction(float(b)) + Fraction(value))
            coefficients.append(value)
        coefficients = np.asarray(coefficients)
        polynomial = np.polynomial.Polynomial(coefficients/np.max(np.abs(coefficients)))
        trimmed = polynomial.trim(tol=64*np.finfo(float).eps).coef
        # Independent root oracle: exact float-to-Decimal inputs and a
        # high-precision quadratic formula, not the production isolator or LAPACK.
        with localcontext() as context:
            context.prec = 160
            c, b, a = [Decimal.from_float(float(x)) for x in trimmed]
            discriminant = b*b-4*a*c
            assert discriminant > 0
            radical = discriminant.sqrt()
            roots = (float((-b-radical)/(2*a)), float((-b+radical)/(2*a)))
        expected.extend(root for root in roots if 0 < root < 1)
    assert expected
    assert _endpoint_projection_cuts(source, target) == tuple(sorted(set(expected)))
