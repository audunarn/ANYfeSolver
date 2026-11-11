"""Tiny driver accuracy checks; do not constitute a slenderness campaign."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from docs.reference_cases.ge_beam3_relative_svd_probe import jacobi_svd, beam_comparison
from anyfesolver._ge_beam3_p5_loads.core import canonical


@pytest.mark.parametrize('amplitude', [1., 1e8, 1e16])
def test_small_singular_value_against_decimal_closed_form(amplitude):
    matrix = np.array([[amplitude, .5*amplitude], [0., 1./amplitude]])
    _, s, _ = jacobi_svd(matrix)
    with localcontext() as context:
        context.prec = 100
        a, b, c, d = [Decimal.from_float(float(x)) for x in matrix.flat]
        trace = a*a+b*b+c*c+d*d; determinant = a*d-b*c
        largest = ((trace+(trace*trace-4*determinant*determinant).sqrt())/2).sqrt()
        smallest = abs(determinant)/largest
    np.testing.assert_allclose(s, [float(largest), float(smallest)], rtol=1e-13, atol=0.)


def test_zero_and_tiny_positive_values_are_not_rank_truncated():
    _, s, v = jacobi_svd(np.diag([0., 1e-20, 1., 1e20]))
    np.testing.assert_allclose(s, [1e20, 1., 1e-20, 0.], rtol=1e-14, atol=0.)
    np.testing.assert_allclose(v@v.T, np.eye(4), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('slenderness', [100., 10000., 1000000.])
def test_beam_degenerate_pairs_remain_equal(slenderness, tmp_path):
    values = beam_comparison(slenderness)
    with (tmp_path/'jacobi-beam.json').open('xb') as stream:
        stream.write(canonical(dict(slenderness=slenderness, eigenvalues=values,
            production_qualified=False)))
    print(canonical(dict(slenderness=slenderness, eigenvalues=values)).decode('ascii'), flush=True)
    for pair in ((0, 1), (2, 3)):
        assert abs(values[pair[0]]-values[pair[1]]) <= 1e-11*max(1., values[pair[1]])
    if slenderness == 1000000.:
        np.testing.assert_allclose(values[:4], [36/49, 36/49, 36., 36.], rtol=1e-9, atol=0.)
