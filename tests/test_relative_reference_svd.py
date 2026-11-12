"""Driver controls, Decimal accuracy and explicit failure propagation."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

import anyfesolver._relative_reference_svd as driver
from anyfesolver._native_relative_reference_factor_spectrum import solve_relative_reference_factor_spectrum


@pytest.mark.parametrize('amplitude', [1., 1e8, 1e16])
def test_decimal_two_by_two_singular_values(amplitude):
    matrix = np.array([[amplitude, .5*amplitude], [0., 1/amplitude]])
    _, values, _ = driver.relative_reference_svd(matrix)
    with localcontext() as context:
        context.prec = 100
        a, b, c, d = map(Decimal.from_float, matrix.flat)
        trace = a*a+b*b+c*c+d*d; determinant = a*d-b*c
        largest = ((trace+(trace*trace-4*determinant*determinant).sqrt())/2).sqrt()
        expected = [float(largest), float(abs(determinant)/largest)]
    np.testing.assert_allclose(values, expected, rtol=1e-13, atol=0.)


def test_exact_zero_and_tiny_positive_are_distinct():
    _, values, vectors = driver.relative_reference_svd(np.diag([0., 1e-20, 1., 1e20]))
    np.testing.assert_allclose(values, [1e20, 1., 1e-20, 0.], rtol=1e-14, atol=0.)
    np.testing.assert_allclose(vectors@vectors.T, np.eye(4), rtol=1e-11, atol=1e-11)
    _, zero, _ = driver.relative_reference_svd(np.zeros((4, 4)))
    np.testing.assert_array_equal(zero, np.zeros(4))


@pytest.mark.parametrize('mutation', ['info', 'warning', 'scale', 'nan_vectors', 'negative', 'order', 'reconstruction'])
def test_driver_failure_or_mutation_never_retries(monkeypatch, mutation):
    original = driver.lapack.dgejsv; calls = []
    def changed(matrix, **kwargs):
        calls.append(kwargs)
        assert kwargs == dict(joba=3, jobu=0, jobv=1, jobr=0, jobt=0, jobp=0, overwrite_a=0)
        s, u, v, work, iw, info = original(matrix, **kwargs)
        if mutation == 'info': info = 1
        elif mutation == 'warning': iw[2] = 1
        elif mutation == 'scale': work[1] = 0.
        elif mutation == 'nan_vectors': v[0, 0] = np.nan
        elif mutation == 'negative': s[-1] = -1.
        elif mutation == 'order': s = s[::-1]
        elif mutation == 'reconstruction': u[0, 0] += 1.
        return s, u, v, work, iw, info
    monkeypatch.setattr(driver.lapack, 'dgejsv', changed)
    with pytest.raises(ValueError): driver.relative_reference_svd(np.diag([1., 2., 3.]))
    assert len(calls) == 1


def test_squared_values_scale_without_normal_equation_cancellation():
    e = np.diag([1e-8, 1., 1e8]); g = np.eye(3)
    result = solve_relative_reference_factor_spectrum(e, g, (0, 1, 2), (), num_modes=3)
    np.testing.assert_allclose(result.eigenvalues, [1e-16, 1., 1e16], rtol=1e-13, atol=0.)
    assert not result.production_qualified and not result.prestressed_tangent_authorized


def test_unrepresentable_squared_frequency_is_not_reported_as_zero():
    with pytest.raises(ValueError, match='unrepresentable squared'):
        solve_relative_reference_factor_spectrum(np.diag([1e-200, 1.]), np.eye(2),
            (0, 1), (), num_modes=2)


@pytest.mark.parametrize('matrix', [np.zeros((2, 3)), np.zeros((1, 257)), np.full((3, 3), np.inf), np.ones(4)])
def test_invalid_shapes_and_values_are_rejected(matrix):
    with pytest.raises(ValueError): driver.relative_reference_svd(matrix)
