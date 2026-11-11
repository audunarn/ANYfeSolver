"""Independent polynomial integrals for partial Q4/S3 pressure support."""

import numpy as np
import pytest

from anyfesolver.general_contact.pressure_space import supported_pressure_gram
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty, S3PatchFace,
)


_RECTANGLE = (0.07, 0.73, 0.08, 0.67)
_PENETRATION_M = 1.0e-5
_STIFFNESS_PA_PER_M = 1.0e6


def _line_integrals(cell_low, cell_high, lower, upper):
    """Exact integrals of the two linear basis functions and their products."""
    width = cell_high-cell_low
    lower = max(cell_low, lower)
    upper = min(cell_high, upper)
    if upper <= lower:
        return np.zeros(2), np.zeros((2, 2))
    start, end = (lower-cell_low)/width, (upper-cell_low)/width
    first = width*((end-end**2/2)-(start-start**2/2))
    second = width*(end**2-start**2)/2
    ll = width*((end-end**2+end**3/3)
                -(start-start**2+start**3/3))
    lr = width*((end**2/2-end**3/3)
                -(start**2/2-start**3/3))
    rr = width*(end**3-start**3)/3
    return np.array((first, second)), np.array(((ll, lr), (lr, rr)))


def _case(divisions):
    def node(column, row):
        return 1+row*(divisions+1)+column

    source_faces = []
    for row in range(divisions):
        for column in range(divisions):
            ids = (node(column, row), node(column+1, row),
                   node(column+1, row+1), node(column, row+1))
            source_faces.append(S3PatchFace(
                ids, np.array([((index-1) % (divisions+1)/divisions,
                                (index-1) // (divisions+1)/divisions, 0.0)
                               for index in ids]),
            ))
    left, right, bottom, top = _RECTANGLE
    corners = np.array(((left, bottom, 0.0), (right, bottom, 0.0),
                        (right, top, 0.0), (left, top, 0.0)))
    target_faces = (
        S3PatchFace((10001, 10002, 10004), corners[[0, 1, 3]]),
        S3PatchFace((10002, 10003, 10004), corners[[1, 2, 3]]),
    )
    geometry = MovingS3PatchPrimalMortarPenalty(
        tuple(source_faces), target_faces, _STIFFNESS_PA_PER_M,
        difference_step=1.0e-6,
        maximum_energy_evaluations=2*divisions**2*884,
        q4_source_area_tolerance=1.0e-10,
        q4_source_weighted_gap_tolerance=1.0e-12,
        maximum_q4_source_integration_triangles=300,
        differentiation="pair_local",
        maximum_candidate_pairs=2*divisions**2,
    )
    source = geometry.reference_source.copy()
    source[:, 2] -= _PENETRATION_M
    return geometry, source, source_faces


def _independent_pressure_integrals(divisions):
    left, right, bottom, top = _RECTANGLE
    count = (divisions+1)**2
    support = np.zeros(count)
    gram = np.zeros((count, count))
    sides = ((0, 0), (1, 0), (1, 1), (0, 1))
    for row in range(divisions):
        y0, y1 = row/divisions, (row+1)/divisions
        y_basis, y_gram = _line_integrals(y0, y1, bottom, top)
        for column in range(divisions):
            x0, x1 = column/divisions, (column+1)/divisions
            x_basis, x_gram = _line_integrals(x0, x1, left, right)
            nodes = (row*(divisions+1)+column,
                     row*(divisions+1)+column+1,
                     (row+1)*(divisions+1)+column+1,
                     (row+1)*(divisions+1)+column)
            for i, (ix, iy) in enumerate(sides):
                support[nodes[i]] += x_basis[ix]*y_basis[iy]
                for j, (jx, jy) in enumerate(sides):
                    gram[nodes[i], nodes[j]] += x_gram[ix, jx]*y_gram[iy, jy]
    return support, gram


@pytest.mark.parametrize("divisions", (4, 8, 16))
def test_partial_nonmatching_patch_matches_exact_q1_pressure_integrals(divisions):
    geometry, source, _faces = _case(divisions)
    state = geometry.evaluate(source, geometry.reference_target)
    support, gram = _independent_pressure_integrals(divisions)
    exact_area = (_RECTANGLE[1]-_RECTANGLE[0])*(_RECTANGLE[3]-_RECTANGLE[2])
    exact_pressure = _STIFFNESS_PA_PER_M*_PENETRATION_M
    assert 0 < np.count_nonzero(support) < len(support)
    np.testing.assert_allclose(state.reference_support, support,
                               rtol=1.0e-4, atol=1.0e-10)
    np.testing.assert_allclose(state.reference_weighted_gap,
                               -_PENETRATION_M*support,
                               rtol=1.0e-4, atol=1.0e-15)
    np.testing.assert_allclose(state.pressure_gram, gram,
                               rtol=1.0e-4, atol=1.0e-10)
    assert state.overlap_area == pytest.approx(exact_area, rel=1.0e-4)
    assert state.reference_support.sum() == pytest.approx(exact_area, rel=1.0e-4)
    assert state.active_pressure_dofs == np.count_nonzero(support)
    np.testing.assert_allclose(state.pressure[support > 0.0], exact_pressure,
                               rtol=1.0e-4)
    assert state.pressure @ state.reference_support == pytest.approx(
        exact_pressure*exact_area, rel=1.0e-4,
    )
    pressure_space = supported_pressure_gram(
        state.pressure_gram, state.reference_support,
    )
    assert pressure_space.scaled_rank == pressure_space.supported_modes
    assert np.isfinite(pressure_space.scaled_condition)
