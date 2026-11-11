from __future__ import annotations

import json

import numpy as np
import pytest

from anyfesolver.ge_beam3_curved_reference import (
    CurvedBeam3ReferenceGeometry,
    GE_BEAM3_CURVED_REFERENCE_INTERPOLATION_ID,
    GE_BEAM3_CURVED_REFERENCE_SCHEMA_ID,
    GeBeam3CurvedFrameError,
    GeBeam3CurvedGeometryError,
    GeBeam3CurvedReferenceError,
    p2_shape_derivatives,
    p2_shape_functions,
    reversal_frame_map,
)


def _triad(tangent: np.ndarray, physical_second_hint: np.ndarray) -> np.ndarray:
    first = np.asarray(tangent, dtype=float)
    first /= np.linalg.norm(first)
    second = np.asarray(physical_second_hint, dtype=float)
    second = second - float(first @ second) * first
    second /= np.linalg.norm(second)
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def _nodal_triads(
    coordinates: np.ndarray,
    hints: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if hints is None:
        hint = np.array((0.0, 0.0, 1.0))
        hints = (hint, hint, hint)
    return np.asarray(
        [
            _triad(p2_shape_derivatives(xi) @ coordinates, hints[index])
            for index, xi in enumerate((-1.0, 0.0, 1.0))
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=float)
    unit /= np.linalg.norm(unit)
    cross = np.array(
        (
            (0.0, -unit[2], unit[1]),
            (unit[2], 0.0, -unit[0]),
            (-unit[1], unit[0], 0.0),
        )
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _planar_geometry() -> CurvedBeam3ReferenceGeometry:
    coordinates = np.array(((-1.0, 0.0, 0.0), (0.0, 0.4, 0.0), (1.0, 0.0, 0.0)))
    return CurvedBeam3ReferenceGeometry(coordinates, _nodal_triads(coordinates))


def test_quadratic_lagrange_basis_is_nodal_and_has_affine_derivative() -> None:
    np.testing.assert_array_equal(p2_shape_functions(-1.0), (1.0, 0.0, 0.0))
    np.testing.assert_array_equal(p2_shape_functions(0.0), (0.0, 1.0, 0.0))
    np.testing.assert_array_equal(p2_shape_functions(1.0), (0.0, 0.0, 1.0))
    np.testing.assert_array_equal(p2_shape_derivatives(-1.0), (-1.5, 2.0, -0.5))
    np.testing.assert_array_equal(p2_shape_derivatives(0.0), (-0.5, 0.0, 0.5))
    np.testing.assert_array_equal(p2_shape_derivatives(1.0), (0.5, -2.0, 1.5))
    with pytest.raises(GeBeam3CurvedReferenceError, match="closed interval"):
        p2_shape_functions(1.00001)


def test_analytic_interval_regularity_matches_affine_tangent_minimum() -> None:
    geometry = _planar_geometry()
    certificate = geometry.regularity
    a = np.asarray(certificate.affine_constant)
    b = np.asarray(certificate.affine_linear)
    expected_xi = float(np.clip(-float(a @ b) / float(b @ b), -1.0, 1.0))
    assert certificate.minimizer_xi == expected_xi == 0.0
    np.testing.assert_allclose(
        certificate.minimum_jacobian_squared,
        np.dot(a + expected_xi * b, a + expected_xi * b),
        rtol=0.0,
        atol=0.0,
    )
    sampled = [geometry.jacobian(xi) for xi in np.linspace(-1.0, 1.0, 1001)]
    assert certificate.minimum_jacobian <= min(sampled) + 1.0e-15
    assert certificate.minimum_jacobian > certificate.minimum_admissible_jacobian


def test_station_frame_is_proper_and_tangent_constrained_for_planar_and_spatial_curves() -> None:
    planar = _planar_geometry()
    spatial_coordinates = np.array(
        ((-1.2, -0.2, 0.4), (0.1, 0.7, 0.9), (1.4, 0.3, -0.1))
    )
    spatial_hint = np.array((0.2, -0.6, 1.0))
    spatial = CurvedBeam3ReferenceGeometry(
        spatial_coordinates,
        _nodal_triads(
            spatial_coordinates,
            (spatial_hint, spatial_hint, spatial_hint),
        ),
    )
    for geometry in (planar, spatial):
        for xi in np.linspace(-1.0, 1.0, 41):
            station = geometry.station(float(xi))
            np.testing.assert_allclose(
                station.frame[:, 0],
                station.derivative / station.jacobian,
                rtol=0.0,
                atol=2.0e-15,
            )
            np.testing.assert_allclose(
                station.frame.T @ station.frame,
                np.eye(3),
                rtol=0.0,
                atol=3.0e-15,
            )
            assert np.linalg.det(station.frame) == pytest.approx(1.0, abs=3.0e-15)
    assert spatial.frame_interpolation_id == GE_BEAM3_CURVED_REFERENCE_INTERPOLATION_ID


def test_authoritative_nodal_triads_are_matched_and_frame_is_continuous() -> None:
    geometry = _planar_geometry()
    supplied = geometry.nodal_triads
    for index, xi in enumerate((-1.0, 0.0, 1.0)):
        np.testing.assert_array_equal(geometry.frame(xi), supplied[index])
    epsilon = 1.0e-9
    np.testing.assert_allclose(
        geometry.frame(-epsilon),
        geometry.frame(epsilon),
        rtol=0.0,
        atol=3.0e-9,
    )


def test_geometry_rejects_degenerate_folded_and_scale_relative_near_singular_curves() -> None:
    folded = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    arbitrary = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    with pytest.raises(GeBeam3CurvedGeometryError, match="interval regularity"):
        CurvedBeam3ReferenceGeometry(folded, arbitrary)

    near_singular = np.array(
        ((0.0, 0.0, 0.0), (0.25 + 1.0e-15, 0.0, 0.0), (1.0, 0.0, 0.0))
    )
    for scale in (1.0, 1.0e8, 1.0e-8):
        coordinates = scale * near_singular
        with pytest.raises(GeBeam3CurvedGeometryError, match="scale-aware"):
            CurvedBeam3ReferenceGeometry(coordinates, arbitrary)


def test_nodal_frames_must_be_proper_and_follow_oriented_tangents() -> None:
    coordinates = np.array(((0.0, 0.0, 0.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0)))
    triads = _nodal_triads(coordinates)
    reflected = triads.copy()
    reflected[1, :, 2] *= -1.0
    with pytest.raises(GeBeam3CurvedFrameError, match=r"proper SO\(3\)"):
        CurvedBeam3ReferenceGeometry(coordinates, reflected)
    wrong_tangent = triads.copy()
    wrong_tangent[0] = triads[2]
    with pytest.raises(GeBeam3CurvedFrameError, match="oriented reference tangent"):
        CurvedBeam3ReferenceGeometry(coordinates, wrong_tangent)


def test_callers_cannot_weaken_frozen_reference_admission_tolerances() -> None:
    geometry = _planar_geometry()
    coordinates = geometry.coordinates
    triads = geometry.nodal_triads
    with pytest.raises(GeBeam3CurvedReferenceError, match="regularity_relative_tolerance"):
        CurvedBeam3ReferenceGeometry(
            coordinates,
            triads,
            regularity_relative_tolerance=1.0e-15,
        )
    with pytest.raises(GeBeam3CurvedReferenceError, match="rotation_tolerance"):
        CurvedBeam3ReferenceGeometry(coordinates, triads, rotation_tolerance=1.0e-6)
    with pytest.raises(GeBeam3CurvedReferenceError, match="frame_tolerance"):
        CurvedBeam3ReferenceGeometry(coordinates, triads, frame_tolerance=1.0e-6)

    reflected = triads.copy()
    reflected[1, :, 2] *= -1.0
    with pytest.raises(GeBeam3CurvedReferenceError, match="rotation_tolerance"):
        CurvedBeam3ReferenceGeometry(
            coordinates,
            reflected,
            rotation_tolerance=3.0,
        )


def test_admitted_nodal_tolerance_is_not_rejected_during_frame_evaluation() -> None:
    coordinates = np.array(
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    )
    admitted = np.eye(3)
    admitted[0, 0] += 4.0e-12
    geometry = CurvedBeam3ReferenceGeometry(
        coordinates,
        np.repeat(admitted[None, :, :], 3, axis=0),
    )
    frame = geometry.frame(-0.5)
    assert np.linalg.norm(frame[:, 0] - np.array((1.0, 0.0, 0.0))) < 1.0e-11


def test_shortest_transport_residual_roll_branch_fails_closed() -> None:
    coordinates = np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)))
    positive = _triad(np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)))
    negative = _triad(np.array((1.0, 0.0, 0.0)), np.array((0.0, -1.0, 0.0)))
    triads = np.asarray((positive, negative, negative))
    with pytest.raises(GeBeam3CurvedFrameError, match="residual roll"):
        CurvedBeam3ReferenceGeometry(coordinates, triads)


def test_shortest_transport_tangent_turn_branch_fails_closed() -> None:
    angle = 0.91 * np.pi
    left_derivative = np.array((np.cos(angle), np.sin(angle), 0.0))
    middle_derivative = np.array((1.0, 0.0, 0.0))
    affine_linear = middle_derivative - left_derivative
    coordinates = np.asarray(
        (
            np.zeros(3),
            0.5 * (middle_derivative + left_derivative),
            2.0 * middle_derivative,
        )
    )
    np.testing.assert_allclose(
        coordinates[0] - 2.0 * coordinates[1] + coordinates[2],
        affine_linear,
        rtol=0.0,
        atol=2.0e-16,
    )
    with pytest.raises(GeBeam3CurvedFrameError, match="tangent turn"):
        CurvedBeam3ReferenceGeometry(coordinates, _nodal_triads(coordinates))


def test_rigid_transformation_is_objective_at_every_station() -> None:
    geometry = _planar_geometry()
    rotation = _axis_angle(np.array((0.3, -0.5, 0.8)), 0.73)
    translation = np.array((4.0, -2.0, 1.5))
    transformed = geometry.rigidly_transformed(rotation, translation)
    for xi in np.linspace(-1.0, 1.0, 29):
        source = geometry.station(float(xi))
        target = transformed.station(float(xi))
        np.testing.assert_allclose(
            target.position,
            rotation @ source.position + translation,
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            target.derivative,
            rotation @ source.derivative,
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(target.frame, rotation @ source.frame, rtol=0.0, atol=3.0e-15)
    with pytest.raises(GeBeam3CurvedFrameError, match=r"proper SO\(3\)"):
        geometry.rigidly_transformed(np.diag((1.0, 1.0, -1.0)), translation)


def test_reversal_is_stationwise_covariant_and_preserves_physical_second_director() -> None:
    coordinates = np.array(
        ((-1.2, -0.2, 0.4), (0.1, 0.7, 0.9), (1.4, 0.3, -0.1))
    )
    hints = (
        np.array((0.4, -0.2, 1.0)),
        np.array((0.1, -0.7, 1.1)),
        np.array((-0.2, -0.4, 1.0)),
    )
    geometry = CurvedBeam3ReferenceGeometry(coordinates, _nodal_triads(coordinates, hints))
    reversed_geometry = geometry.reversed()
    transform = reversal_frame_map()
    assert np.linalg.det(transform) == pytest.approx(1.0)
    for xi in np.linspace(-1.0, 1.0, 37):
        source = geometry.station(float(-xi))
        target = reversed_geometry.station(float(xi))
        np.testing.assert_allclose(target.position, source.position, rtol=0.0, atol=3.0e-15)
        np.testing.assert_allclose(target.derivative, -source.derivative, rtol=0.0, atol=3.0e-15)
        np.testing.assert_allclose(target.frame, source.frame @ transform, rtol=0.0, atol=4.0e-15)
        np.testing.assert_allclose(target.frame[:, 1], source.frame[:, 1], rtol=0.0, atol=4.0e-15)
    np.testing.assert_allclose(
        reversed_geometry.reversed().coordinates,
        geometry.coordinates,
        rtol=0.0,
        atol=0.0,
    )


def test_straight_limit_is_exactly_constant_and_scale_independent() -> None:
    frame = _triad(np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)))
    for scale in (1.0e-8, 1.0, 1.0e8):
        coordinates = scale * np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)))
        geometry = CurvedBeam3ReferenceGeometry(
            coordinates,
            np.repeat(frame[None, :, :], 3, axis=0),
        )
        assert geometry.regularity.minimizer_xi == 0.0
        assert geometry.regularity.minimum_jacobian == pytest.approx(0.5 * scale)
        for xi in np.linspace(-1.0, 1.0, 17):
            np.testing.assert_allclose(geometry.frame(float(xi)), frame, rtol=0.0, atol=0.0)


def test_canonical_data_and_fingerprint_are_deterministic_and_owned() -> None:
    geometry = _planar_geometry()
    coordinates = geometry.coordinates
    triads = geometry.nodal_triads
    duplicate = CurvedBeam3ReferenceGeometry(coordinates, triads)
    assert duplicate.canonical_bytes() == geometry.canonical_bytes()
    assert duplicate.fingerprint() == geometry.fingerprint()
    assert len(geometry.fingerprint()) == 64
    assert geometry.fingerprint() == geometry.fingerprint().upper()
    decoded = json.loads(geometry.canonical_bytes())
    assert decoded["schema"] == GE_BEAM3_CURVED_REFERENCE_SCHEMA_ID
    assert decoded["frame_interpolation"]["authority"] == "INDEPENDENT_DERIVATION"
    coordinates[:] = 99.0
    triads[:] = 0.0
    assert duplicate.canonical_bytes() == geometry.canonical_bytes()


def test_analytic_frame_derivative_and_intrinsic_curvature_are_objective() -> None:
    geometry = _planar_geometry()
    rotation = _axis_angle(np.array((0.3, -0.5, 0.8)), 0.73)
    transformed = geometry.rigidly_transformed(rotation, np.array((4.0, -2.0, 1.5)))
    for xi in (-0.8, -0.3, 0.3, 0.8):
        step = 1.0e-7
        diagnostic = (
            geometry.frame(xi + step) - geometry.frame(xi - step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            geometry.frame_derivative(xi),
            diagnostic,
            rtol=0.0,
            atol=2.0e-9,
        )
        np.testing.assert_allclose(
            transformed.frame_derivative(xi),
            rotation @ geometry.frame_derivative(xi),
            rtol=0.0,
            atol=4.0e-15,
        )
        np.testing.assert_allclose(
            transformed.intrinsic_curvature(xi),
            geometry.intrinsic_curvature(xi),
            rtol=0.0,
            atol=4.0e-15,
        )


def test_intrinsic_curvature_reversal_and_midpoint_traces_are_explicit() -> None:
    geometry = _planar_geometry()
    reversed_geometry = geometry.reversed()
    transform = reversal_frame_map()
    for xi in (-0.8, -0.3, 0.3, 0.8):
        np.testing.assert_allclose(
            reversed_geometry.intrinsic_curvature(xi),
            (-transform) @ geometry.intrinsic_curvature(-xi),
            rtol=0.0,
            atol=5.0e-15,
        )
    with pytest.raises(GeBeam3CurvedReferenceError, match="two analytic"):
        geometry.frame_derivative(0.0)
    for trace in ("LEFT", "RIGHT"):
        derivative = geometry.frame_derivative(0.0, trace=trace)
        spin = geometry.frame(0.0).T @ derivative
        np.testing.assert_allclose(spin + spin.T, np.zeros((3, 3)), rtol=0.0, atol=2.0e-15)
