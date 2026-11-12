"""Virgin native B3 kinematic bridge for a conditional broken-trace bound.

This checks the actual retained operator, not material coercivity or F03
acceptance. Generalized resultants remain retained in the evaluated Hessian.
"""

import numpy as np
import pytest
from scipy import linalg

from anyfesolver._ge_beam3_generalized_ellipsoid_section import (
    EllipsoidalGeneralizedSection,
)
from anyfesolver._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from scripts._b3_e4_f03_native_fixture import SECTION_DIAGONAL, build


def _skew(vector):
    x, y, z = vector
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


@pytest.mark.parametrize("half_cells", (4, 8, 16))
def test_actual_native_virgin_strains_have_half_cell_difference_structure(
        half_cells):
    fixture = build(half_cells, z_offset=.0675, assemble=False)
    operator = fixture.context.elements[0][1].operator
    reference = operator.reference
    result = operator.evaluate(
        reference.coordinates, np.zeros((3, 3)),
        reference.nodal_triads, np.tile(np.eye(3), (2, 1, 1)),
        np.zeros(18))
    actual = result.hessian[24:, :24]
    expected = np.zeros((18, 24))
    for cell in (0, 1):
        left, right = cell, cell+1
        chord = reference.coordinates[right]-reference.coordinates[left]
        z = 3*cell
        expected[z:z+3, 6*left:6*left+3] = -np.eye(3)
        expected[z:z+3, 6*right:6*right+3] = np.eye(3)
        expected[z:z+3, 18+3*cell:21+3*cell] = _skew(chord)
        left_log = 6+6*cell
        right_log = left_log+3
        expected[left_log:left_log+3, 6*left+3:6*left+6] = -np.eye(3)
        expected[left_log:left_log+3, 18+3*cell:21+3*cell] = np.eye(3)
        expected[right_log:right_log+3, 6*right+3:6*right+6] = np.eye(3)
        expected[right_log:right_log+3, 18+3*cell:21+3*cell] = -np.eye(3)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=3e-14)
    assert np.linalg.matrix_rank(actual, tol=1e-12) == 18


def test_coupled_spd_section_preserves_straight_virgin_energy_lower_order():
    fixture = build(4, z_offset=.0675, assemble=False)
    reference = fixture.context.elements[0][1].operator.reference
    diagonal = np.diag(SECTION_DIAGONAL)
    root = np.sqrt(SECTION_DIAGONAL)
    coupling = np.array((1., -.4, .6, .5, -.3, .8))
    coupled = root[:, None]*(.8*np.eye(6)+.15*np.outer(coupling, coupling))*root
    coupled = np.triu(coupled)+np.triu(coupled, 1).T
    alpha = linalg.eigvalsh(coupled, diagonal)[0]
    assert alpha == pytest.approx(.8, abs=2e-15)

    def tangent(section):
        operator = RetainedGeneralizedOperator(reference, section, order=4)
        result = operator.evaluate(
            reference.coordinates, np.zeros((3, 3)),
            reference.nodal_triads, np.tile(np.eye(3), (2, 1, 1)),
            np.zeros(18))
        mixed = result.hessian
        jacobian = mixed[24:, 6:24]
        compliance = -mixed[24:, 24:]
        return jacobian.T@linalg.cho_solve(
            linalg.cho_factor(compliance, lower=True), jacobian)

    reference_section = EllipsoidalGeneralizedSection(
        diagonal, np.eye(6), 1e20, 1e9)
    coupled_section = EllipsoidalGeneralizedSection(
        coupled, np.eye(6), 1e20, 1e9)
    reference_energy = tangent(reference_section)
    coupled_energy = tangent(coupled_section)
    eigenvalues = linalg.eigvalsh(coupled_energy, reference_energy)
    assert eigenvalues.min() >= alpha-2e-11
    assert eigenvalues.max() > 1.
