"""Physical varying-frame warped Q4 skin geometry, before contact admission."""

import numpy as np
import pytest

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.warped_q4_skin import WarpedQ4Skin


_WARPED = np.array(((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.08),
                    (1.0, 1.0, -0.04),
                    (0.0, 1.0, 0.03)))


def _model(corners, *, reference_normal=(0.0, 0.0, 1.0), polarity=1,
           warped_formulation="varying_frame", thickness=0.08):
    model = FEModel("warped_q4_contact_geometry")
    model.add_material("elastic", 1.0e5, 0.3)
    for node, point in enumerate(corners, 1):
        model.add_node(node, *point)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=thickness,
        warped_formulation=warped_formulation,
        reference_normal=reference_normal,
        director_polarity=polarity,
    ))
    return model


def _independent_position(corners, xi, eta, offset, physical_normal):
    shape = 0.25 * np.array(((1 - xi) * (1 - eta),
                             (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta),
                             (1 - xi) * (1 + eta)))
    derivative_xi = 0.25 * np.array((-(1 - eta), 1 - eta,
                                     1 + eta, -(1 + eta)))
    derivative_eta = 0.25 * np.array((-(1 - xi), -(1 + xi),
                                      1 + xi, 1 - xi))
    normal = np.cross(derivative_xi @ corners, derivative_eta @ corners)
    normal /= np.linalg.norm(normal)
    if normal @ physical_normal < 0:
        normal = -normal
    return shape @ corners + offset * normal


def test_warped_q4_skin_follows_bilinear_varying_frame_and_derivatives():
    skin = WarpedQ4Skin(_model(_WARPED), 1, "top")
    bottom = WarpedQ4Skin(_model(_WARPED), 1, "bottom")
    h = 1.0e-6
    for xi, eta in ((-0.65, -0.45), (0.0, 0.0), (0.37, 0.71)):
        sample = skin.sample(xi, eta)
        expected = _independent_position(_WARPED, xi, eta, 0.04,
                                         np.array((0.0, 0.0, 1.0)))
        np.testing.assert_allclose(sample.position, expected, rtol=0, atol=2e-14)
        derivative_xi = (
            _independent_position(_WARPED, xi + h, eta, 0.04, np.array((0, 0, 1)))
            - _independent_position(_WARPED, xi - h, eta, 0.04, np.array((0, 0, 1)))
        ) / (2 * h)
        derivative_eta = (
            _independent_position(_WARPED, xi, eta + h, 0.04, np.array((0, 0, 1)))
            - _independent_position(_WARPED, xi, eta - h, 0.04, np.array((0, 0, 1)))
        ) / (2 * h)
        np.testing.assert_allclose(sample.tangent_xi, derivative_xi,
                                   rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(sample.tangent_eta, derivative_eta,
                                   rtol=1e-8, atol=1e-9)
        assert sample.reference_measure == pytest.approx(
            np.linalg.norm(np.cross(derivative_xi, derivative_eta)), rel=1e-8,
        )
        np.testing.assert_allclose(
            sample.position - bottom.sample(xi, eta).position,
            0.08 * sample.director, rtol=0, atol=2e-14,
        )


def test_warped_q4_skin_is_equivariant_and_bounds_nonnodal_extrema():
    original = WarpedQ4Skin(_model(_WARPED), 1, "top")
    rotation = np.array(((0.0, -1.0, 0.0),
                         (0.8, 0.0, -0.6),
                         (0.6, 0.0, 0.8)))
    translation = np.array((2.0, -3.0, 0.4))
    moved = WarpedQ4Skin(_model(
        _WARPED @ rotation.T + translation,
        reference_normal=rotation @ np.array((0.0, 0.0, 1.0)),
    ), 1, "top")
    reversed_data = WarpedQ4Skin(_model(
        _WARPED, reference_normal=(0.0, 0.0, -1.0), polarity=-1,
    ), 1, "top")
    lower, upper = original.conservative_bounds()
    for xi in np.linspace(-1.0, 1.0, 41):
        for eta in np.linspace(-1.0, 1.0, 41):
            sample = original.sample(xi, eta)
            assert np.all(sample.position >= lower - 1e-14)
            assert np.all(sample.position <= upper + 1e-14)
            transformed = moved.sample(xi, eta)
            np.testing.assert_allclose(transformed.position,
                                       rotation @ sample.position + translation,
                                       atol=1e-12)
            np.testing.assert_allclose(transformed.director,
                                       rotation @ sample.director, atol=1e-12)
            np.testing.assert_allclose(reversed_data.sample(xi, eta).position,
                                       sample.position, atol=1e-12)


def test_warped_q4_skin_rejects_unsupported_route_and_parameter():
    model = _model(_WARPED, warped_formulation="reject")
    with pytest.raises(ValueError, match="varying-frame"):
        WarpedQ4Skin(model, 1, "top")
    skin = WarpedQ4Skin(_model(_WARPED), 1, "top")
    with pytest.raises(ValueError, match="inside"):
        skin.sample(1.1, 0.0)


def test_warped_q4_reference_plane_gap_catches_nonnodal_edge_minimum():
    corners = np.column_stack((
        np.array((0.0, 1.0, 1.0, 0.0)),
        np.array((0.0, 0.0, 1.0, 1.0)),
        np.array((0.8223682702671189, -0.030550590109171383,
                  0.506803038611159, -0.339761771934596)),
    ))
    normal = np.array((-0.8961944507236381, -0.380450358369687,
                       -0.2282389785039647))
    skin = WarpedQ4Skin(_model(corners, thickness=1.0), 1, "top")
    corner_minimum = min(float(normal @ _independent_position(
        corners, xi, eta, 0.5, np.array((0.0, 0.0, 1.0)),
    )) for xi in (-1.0, 1.0) for eta in (-1.0, 1.0))
    independent_minimum = min(float(normal @ _independent_position(
        corners, float(xi), float(eta), 0.5, np.array((0.0, 0.0, 1.0)),
    )) for xi in np.linspace(-1.0, 1.0, 121)
        for eta in np.linspace(-1.0, 1.0, 121))
    assert corner_minimum - independent_minimum > 0.01
    certificate = skin.certify_reference_plane_gap(
        np.zeros(3), normal, accuracy=0.002,
        maximum_cells=2_000, maximum_depth=14,
    )
    assert certificate.conservative_lower_gap <= independent_minimum
    assert certificate.minimum_sampled_gap - certificate.conservative_lower_gap <= 0.002
    assert certificate.minimum_sampled_gap < corner_minimum - 0.01
    moved = skin.certify_reference_plane_gap(
        0.15 * normal, normal, accuracy=0.002,
        maximum_cells=2_000, maximum_depth=14,
    )
    assert moved.cells_checked == certificate.cells_checked
    assert moved.conservative_lower_gap == pytest.approx(
        certificate.conservative_lower_gap - 0.15, abs=1e-11,
    )
    with pytest.raises(MemoryError, match="cell budget exhausted"):
        skin.certify_reference_plane_gap(
            np.zeros(3), normal, accuracy=0.002,
            maximum_cells=10, maximum_depth=14,
        )


def test_warped_q4_gap_lower_bound_contains_independent_random_samples():
    rng = np.random.default_rng(227)
    for _case in range(6):
        corners = np.column_stack((
            np.array((0.0, 1.0, 1.0, 0.0)),
            np.array((0.0, 0.0, 1.0, 1.0)),
            rng.uniform(-0.15, 0.15, 4),
        ))
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        skin = WarpedQ4Skin(_model(corners), 1, "bottom")
        certificate = skin.certify_reference_plane_gap(
            np.zeros(3), direction, accuracy=0.01,
            maximum_cells=5_000, maximum_depth=12,
        )
        independent = min(float(direction @ _independent_position(
            corners, float(xi), float(eta), -0.04,
            np.array((0.0, 0.0, 1.0)),
        )) for xi in np.linspace(-1.0, 1.0, 31)
            for eta in np.linspace(-1.0, 1.0, 31))
        assert certificate.conservative_lower_gap <= independent + 1e-12
        assert (certificate.minimum_sampled_gap
                - certificate.conservative_lower_gap <= 0.01 + 1e-12)


def test_warped_q4_linearized_current_gap_matches_independent_surface():
    skin = WarpedQ4Skin(_model(_WARPED), 1, "top")
    nodal = np.array((
        (0.002, -0.001, 0.003, 0.011, -0.009, 0.004),
        (-0.001, 0.002, -0.004, 0.006, 0.012, -0.008),
        (0.003, 0.001, 0.002, -0.007, 0.005, 0.009),
        (-0.002, -0.003, 0.001, 0.008, -0.006, -0.003),
    ))
    normal = np.array((0.2, -0.3, np.sqrt(0.87)))
    point = np.array((0.0, 0.0, 0.05))
    certificate = skin.certify_linearized_plane_gap(
        nodal, point, normal, accuracy=0.0005,
        maximum_cells=5_000, maximum_depth=14,
    )
    sampled = []
    for xi in np.linspace(-1.0, 1.0, 61):
        for eta in np.linspace(-1.0, 1.0, 61):
            shape = 0.25 * np.array(((1 - xi) * (1 - eta),
                                     (1 + xi) * (1 - eta),
                                     (1 + xi) * (1 + eta),
                                     (1 - xi) * (1 + eta)))
            reference_position = _independent_position(
                _WARPED, float(xi), float(eta), 0.04,
                np.array((0.0, 0.0, 1.0)),
            )
            director = (reference_position - shape @ _WARPED) / 0.04
            current = (reference_position + shape @ nodal[:, :3]
                       + 0.04 * np.cross(shape @ nodal[:, 3:], director))
            sampled.append(float(normal @ (current - point)))
    independent = min(sampled)
    assert certificate.conservative_lower_gap <= independent + 1e-12
    assert (certificate.minimum_sampled_gap
            - certificate.conservative_lower_gap <= 0.0005 + 1e-12)
    reference = skin.certify_reference_plane_gap(
        point, normal, accuracy=0.0005,
        maximum_cells=5_000, maximum_depth=14,
    )
    zeros = skin.certify_linearized_plane_gap(
        np.zeros((4, 6)), point, normal, accuracy=0.0005,
        maximum_cells=5_000, maximum_depth=14,
    )
    assert zeros == reference

    omega = np.array((0.001, -0.002, 0.0015))
    rigid = np.column_stack((np.cross(omega, _WARPED),
                             np.broadcast_to(omega, (4, 3))))
    xi, eta = 0.37, -0.22
    shape = 0.25 * np.array(((1 - xi) * (1 - eta),
                             (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta),
                             (1 - xi) * (1 + eta)))
    reference_position = skin.sample(xi, eta).position
    linearized = (reference_position + shape @ rigid[:, :3]
                  + 0.04 * np.cross(shape @ rigid[:, 3:],
                                    skin.sample(xi, eta).director))
    np.testing.assert_allclose(linearized,
                               reference_position + np.cross(omega, reference_position),
                               rtol=0, atol=1e-12)


def test_warped_q4_linearized_swept_bound_contains_interior_path_samples():
    skin = WarpedQ4Skin(_model(_WARPED), 1, "top")
    start = np.zeros((4, 6))
    end = np.array((
        (0.003, 0.001, -0.009, 0.005, 0.004, 0.002),
        (0.001, -0.002, -0.008, -0.006, 0.007, 0.001),
        (-0.002, 0.002, -0.012, 0.004, -0.003, -0.005),
        (0.002, -0.001, -0.011, 0.003, 0.005, -0.002),
    ))
    normal = np.array((0.0, 0.0, 1.0))
    first_point = np.array((0.0, 0.0, -0.1))
    last_point = first_point + np.array((0.0, 0.0, 0.015))
    swept = skin.certify_linearized_swept_plane_gap(
        start, end, first_point, last_point, normal,
        accuracy=0.001, maximum_cells=3_000, maximum_depth=12,
    )
    assert swept.conservative_lower_gap == min(
        swept.start.conservative_lower_gap, swept.end.conservative_lower_gap,
    )
    for time in np.linspace(0.0, 1.0, 9):
        coordinates = (1.0 - time) * start + time * end
        plane_point = (1.0 - time) * first_point + time * last_point
        for xi in np.linspace(-1.0, 1.0, 13):
            for eta in np.linspace(-1.0, 1.0, 13):
                shape = 0.25 * np.array(((1 - xi) * (1 - eta),
                                         (1 + xi) * (1 - eta),
                                         (1 + xi) * (1 + eta),
                                         (1 - xi) * (1 + eta)))
                reference_position = _independent_position(
                    _WARPED, float(xi), float(eta), 0.04,
                    np.array((0.0, 0.0, 1.0)),
                )
                director = (reference_position - shape @ _WARPED) / 0.04
                current = (reference_position + shape @ coordinates[:, :3]
                           + 0.04 * np.cross(shape @ coordinates[:, 3:], director))
                assert float(normal @ (current - plane_point)) >= (
                    swept.conservative_lower_gap - 1e-12
                )
