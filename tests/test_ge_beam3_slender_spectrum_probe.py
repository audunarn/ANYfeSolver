"""Expose spectral conditioning without a large slenderness campaign."""

import numpy as np
import pytest

from fixtures.reference_cases.ge_beam3_slender_spectrum_probe import comparison
from anyfesolver._ge_beam3_p5_loads.core import canonical


@pytest.mark.parametrize('slenderness', [100., 10000., 1000000.])
def test_factor_successor_retains_inextensible_discrete_bending_limits(slenderness, tmp_path):
    row = comparison(slenderness)
    with (tmp_path/'comparison.json').open('xb') as stream: stream.write(canonical(row))
    print(canonical(row).decode('ascii'), flush=True)
    ratio = row['successor_squared_frequencies'][:4]/np.array([36./49, 36./49, 36., 36.])
    assert np.all(ratio > 0.)
    assert np.max(np.abs(np.sqrt(ratio)-1.)) < .02
    assert row['successor_normalized_residual'] <= 1e-11
    assert not row['production_qualified'] and not row['prestressed_tangent_authorized']


def test_rational_inextensible_discrete_reference():
    from fractions import Fraction as F
    # Two unit cells, clamp left trace. Condensing the two free traces from
    # complementary bending energy gives these two displacement coordinates.
    # Reference linear translation mass (zero rotary inertia thin limit).
    # q=(v_mid,v_tip,theta_mid,theta_tip); cell spin equals the unit-cell
    # transverse slope. H^-1 for linear endpoint moments is [[4,-2],[-2,4]].
    jumps = (np.array([[1, 0, 0, 0], [-1, 0, 1, 0]], dtype=object),
             np.array([[-1, 1, -1, 0], [1, -1, 0, 1]], dtype=object))
    inverse_h = np.array([[F(4), F(-2)], [F(-2), F(4)]], dtype=object)
    stiffness = sum(d.T@inverse_h@d for d in jumps)
    a = stiffness[2:, 2:]; determinant = a[0, 0]*a[1, 1]-a[0, 1]*a[1, 0]
    inverse_a = np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]], dtype=object)/determinant
    reduced = stiffness[:2, :2]-stiffness[:2, 2:]@inverse_a@stiffness[2:, :2]
    k11, k12, k22 = reduced[0, 0], reduced[0, 1], reduced[1, 1]
    assert (k11, k12, k22) == (F(96, 7), F(-30, 7), F(12, 7))
    m11, m12, m22 = F(2, 3), F(1, 6), F(1, 3)
    for eigenvalue in (F(36, 49), F(36)):
        assert (k11-eigenvalue*m11)*(k22-eigenvalue*m22)-(k12-eigenvalue*m12)**2 == 0
