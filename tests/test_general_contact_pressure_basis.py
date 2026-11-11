"""Topology-only controls for the optional macro-Q1 pressure basis."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from anyfesolver.general_contact.pressure_basis import (
    MACRO_Q1_PAIR_VERSION,
    SELECTED_Q1_KNOTS_VERSION,
    PressureBasis,
    build_q4_source_macro_q1_pair_basis,
    build_q4_source_selected_q1_basis,
)


def _strip_faces(cells):
    stride = cells+1
    return tuple((1+j, 2+j, stride+2+j, stride+1+j)
                 for j in range(cells))


def _direct_two_face_integrals():
    """Integrate fine and macro shapes independently on [0,2]×[0,1]."""
    faces = _strip_faces(2)
    fine_support = np.zeros(6)
    fine_weighted_gap = np.zeros(6)
    fine_gram = np.zeros((6, 6))
    fine_ds = np.zeros((6, 2))
    fine_dw = np.zeros((6, 2))
    macro_support = np.zeros(4)
    macro_weighted_gap = np.zeros(4)
    macro_gram = np.zeros((4, 4))
    macro_ds = np.zeros((4, 2))
    macro_dw = np.zeros((4, 2))
    points, weights = np.polynomial.legendre.leggauss(4)
    for cell, face in enumerate(faces):
        indices = np.asarray(face, dtype=int)-1
        for u, wu in zip(points, weights):
            x = cell+0.5*(u+1.0)
            local_x = 0.5*(u+1.0)
            for v, wv in zip(points, weights):
                y = 0.5*(v+1.0)
                weight = 0.25*wu*wv
                local = np.array([
                    (1.0-local_x)*(1.0-y), local_x*(1.0-y),
                    local_x*y, (1.0-local_x)*y,
                ])
                # Direct Q1 interpolation on the two-cell macro domain.
                macro = np.array([
                    (1.0-x/2.0)*(1.0-y), (x/2.0)*(1.0-y),
                    (1.0-x/2.0)*y, (x/2.0)*y,
                ])
                gap = 0.15+0.25*x-0.4*y+0.12*x*y+0.03*x*x
                gap_derivative = np.array([
                    0.2*x*y-0.1*y*y, -0.3*x+0.05*x*y,
                ])
                jacobian_derivative = np.array([x, y])
                weighted_derivative = gap_derivative+gap*jacobian_derivative
                fine_support[indices] += weight*local
                fine_weighted_gap[indices] += weight*local*gap
                fine_gram[np.ix_(indices, indices)] += weight*np.outer(local, local)
                fine_ds[indices] += weight*local[:, None]*jacobian_derivative
                fine_dw[indices] += weight*local[:, None]*weighted_derivative
                macro_support += weight*macro
                macro_weighted_gap += weight*macro*gap
                macro_gram += weight*np.outer(macro, macro)
                macro_ds += weight*macro[:, None]*jacobian_derivative
                macro_dw += weight*macro[:, None]*weighted_derivative
    return ((fine_support, fine_weighted_gap, fine_gram, fine_ds, fine_dw),
            (macro_support, macro_weighted_gap, macro_gram, macro_ds, macro_dw))


@pytest.mark.parametrize("cells", (16, 24))
def test_pairwise_macro_q1_is_positive_shared_and_partition_unity(cells):
    basis = build_q4_source_macro_q1_pair_basis(_strip_faces(cells))
    B = basis.as_array()
    assert basis.version == MACRO_Q1_PAIR_VERSION
    assert B.shape == (2*(cells+1), 2*(cells//2+1))
    assert basis.fine_source_node_ids == tuple(range(1, 2*(cells+1)+1))
    assert basis.mode_ids == tuple(
        [1+j for j in range(0, cells+1, 2)]
        + [cells+2+j for j in range(0, cells+1, 2)]
    )
    assert np.all(B >= 0.0)
    np.testing.assert_array_equal(B @ np.ones(len(basis.mode_ids)), 1.0)
    assert np.linalg.matrix_rank(B) == len(basis.mode_ids)
    for lower_offset in (0, cells+1):
        for column in range(cells+1):
            row = B[lower_offset+column]
            if column % 2 == 0:
                assert np.count_nonzero(row) == 1
                assert row.max() == 1.0
            else:
                assert np.count_nonzero(row) == 2
                np.testing.assert_array_equal(row[row > 0.0], [0.5, 0.5])


def test_transformed_integrals_match_direct_macro_quadrature():
    basis = build_q4_source_macro_q1_pair_basis(_strip_faces(2))
    fine, direct = _direct_two_face_integrals()
    support, weighted_gap, gram = basis.transform_integrals(*fine[:3])
    ds, dw = basis.transform_derivatives(*fine[3:])
    for actual, expected in zip((support, weighted_gap, gram, ds, dw), direct):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)
    assert np.linalg.matrix_rank(gram) == len(basis.mode_ids)
    assert np.ptp(weighted_gap/support) > 0.1


def test_transform_rejects_invalid_integrals_and_gradient_shapes():
    basis = build_q4_source_macro_q1_pair_basis(_strip_faces(2))
    fine, _direct = _direct_two_face_integrals()
    support, weighted_gap, gram, ds, dw = fine
    for inputs in (
        (support[:-1], weighted_gap, gram),
        (support, np.full_like(weighted_gap, np.nan), gram),
        (-support, weighted_gap, gram),
        (support, weighted_gap, gram+np.triu(np.ones_like(gram), 1)),
        (support, weighted_gap, -gram),
    ):
        with pytest.raises(ValueError):
            basis.transform_integrals(*inputs)
    for inputs in (
        (ds[:-1], dw),
        (ds, dw[:, :1]),
        (ds, np.full_like(dw, np.inf)),
    ):
        with pytest.raises(ValueError):
            basis.transform_derivatives(*inputs)


def test_basis_identity_does_not_depend_on_current_coordinates():
    faces = _strip_faces(16)
    original = build_q4_source_macro_q1_pair_basis(faces)
    coordinates = {node: np.array([float(node), 0.0, 0.0])
                   for node in original.fine_source_node_ids}
    for node in coordinates:
        coordinates[node] += np.array([0.031, -0.004, 0.002])
    # The builder consumes material incidence, so motion cannot change IDs/B.
    moved = build_q4_source_macro_q1_pair_basis(faces)
    assert moved == original
    with pytest.raises(FrozenInstanceError):
        original.mode_ids = ()
    with pytest.raises(TypeError):
        original.B[0][0] = 0.0
    detached = original.as_array()
    detached[0, :] = 0.0
    assert original.B[0][0] == 1.0


@pytest.mark.parametrize("faces", (
    (),
    _strip_faces(3),
    _strip_faces(34),
    ((1, 2, 5, 4), (7, 8, 9, 10)),
    ((1, 2, 5, 4), (5, 6, 7, 2)),
    ((1, 2, 5, 4), (2, 3, 1, 5)),
    ((1, 2, 5, 4), (2, 3, 6, 5), (3, 4, 7, 6), (4, 1, 8, 7)),
    ((1, 2, 5, 4), (2, 3, 3, 5)),
    ((0, 2, 5, 4), (2, 3, 6, 5)),
))
def test_unsupported_strip_topology_is_rejected(faces):
    with pytest.raises(ValueError):
        build_q4_source_macro_q1_pair_basis(faces)


def test_pressure_basis_constructor_rejects_nonpositive_or_unanchored_rows():
    with pytest.raises(ValueError):
        PressureBasis(MACRO_Q1_PAIR_VERSION, (1, 2), (1,), ((1.0,), (-0.1,)))
    with pytest.raises(ValueError):
        PressureBasis(MACRO_Q1_PAIR_VERSION, (1, 2), (1, 2),
                      ((0.5, 0.5), (0.5, 0.5)))
    mutable = [[1.0], [1.0]]
    basis = PressureBasis(MACRO_Q1_PAIR_VERSION, [1, 2], [1], mutable)
    mutable[0][0] = 0.0
    assert basis.B == ((1.0,), (1.0,))


def test_malformed_face_values_fail_with_explicit_error():
    with pytest.raises(ValueError, match="four distinct"):
        build_q4_source_macro_q1_pair_basis(
            ((1, 2, 5, 4), (2, [3], 6, 5)),
        )


def test_saved_weak_three_node_pattern_has_small_euclidean_macro_projection():
    # A heuristic on the *scaled* saved mode, not a physical Schur or
    # response qualification. The physical metric and column scaling still
    # need to be restored before any acceptance claim.
    basis = build_q4_source_macro_q1_pair_basis(_strip_faces(24))
    B = basis.as_array()
    weak = np.zeros(len(basis.fine_source_node_ids))
    for node, value in ((42, -0.422), (43, 0.813), (44, -0.400)):
        weak[basis.fine_source_node_ids.index(node)] = value
    projection = B @ np.linalg.lstsq(B, weak, rcond=None)[0]
    assert np.linalg.norm(projection)/np.linalg.norm(weak) < 0.03


def test_selected_knots_match_direct_piecewise_q1_work_on_each_fine_face():
    cells = 8
    knots = (0, 1, 3, 6, 8)
    basis = build_q4_source_selected_q1_basis(_strip_faces(cells), knots)
    B = basis.as_array()
    assert basis.version == SELECTED_Q1_KNOTS_VERSION
    assert B.shape == (18, 10)
    assert np.linalg.matrix_rank(B) == 10
    assert np.all(B >= 0.0)
    np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-15)
    points, _weights = np.polynomial.legendre.leggauss(4)
    mode_order = {node: index for index, node in enumerate(basis.mode_ids)}
    for cell, face in enumerate(_strip_faces(cells)):
        fine_indices = [basis.fine_source_node_ids.index(node) for node in face]
        for xi in points:
            x = cell+0.5*(xi+1.0)
            local_x = x-cell
            left = max(knot for knot in knots if knot <= x)
            right = min(knot for knot in knots if knot >= x)
            if left == right:
                left = knots[knots.index(left)-1] if left else left
                right = knots[knots.index(right)+1] if right < cells else right
            fraction = (x-left)/(right-left)
            for eta in points:
                y = 0.5*(eta+1.0)
                fine_shape = np.array((
                    (1-local_x)*(1-y), local_x*(1-y),
                    local_x*y, (1-local_x)*y,
                ))
                direct = np.zeros(len(basis.mode_ids))
                for row, y_weight in ((0, 1-y), (1, y)):
                    offset = row*(cells+1)
                    direct[mode_order[1+offset+left]] = (1-fraction)*y_weight
                    direct[mode_order[1+offset+right]] = fraction*y_weight
                np.testing.assert_allclose(fine_shape @ B[fine_indices],
                                           direct, atol=2e-15, rtol=0.0)


def test_selected_nested_front_knots_reduce_modes_without_changing_macro_pair():
    cells = 128
    knots = [0]
    for cell in range(16):
        parts = 8 if 9 <= cell < 13 else 1
        knots.extend((cell*8+part*8//parts) for part in range(1, parts+1))
    selected = build_q4_source_selected_q1_basis(_strip_faces(cells), knots)
    assert len(selected.mode_ids) == 90
    assert selected.B[0][selected.mode_ids.index(1)] == 1.0
    paired = build_q4_source_macro_q1_pair_basis(_strip_faces(16))
    equivalent = build_q4_source_selected_q1_basis(
        _strip_faces(16), range(0, 17, 2),
    )
    assert equivalent.B == paired.B
    assert equivalent.mode_ids == paired.mode_ids


@pytest.mark.parametrize("knots", (
    (), (0,), (1, 8), (0, 7), (0, 4, 4, 8),
    (0, 9), (0, 3.0, 8), (0, True, 8),
))
def test_selected_knots_reject_invalid_material_columns(knots):
    with pytest.raises(ValueError):
        build_q4_source_selected_q1_basis(_strip_faces(8), knots)
