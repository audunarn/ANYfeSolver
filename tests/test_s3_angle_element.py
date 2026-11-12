"""Reduced-angle element evidence for the current-policy (V2D) S3 route.

These are the fast regression checks of the S3 minimum-angle investigation
(docs/S3_ANGLE_EXTENSION.md).  They exercise the *public* three-node selector,
not a named historical class, and verify element properties that must hold on
any admitted triangle: symmetry, rigid-body invariance, exact membrane and
Kirchhoff bending patches, absence of spurious mechanisms, node-order
covariance, frame objectivity and scale covariance.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from _s3_angle_fixtures import (
    ANGLE_BANDS_DEG,
    CANDIDATE_ENVELOPES,
    DIAGNOSTIC_BANDS_DEG,
    NU,
    build_default_s3,
    bending_patch_field,
    dimensionless_stiffness,
    elastic_spectrum,
    envelope_violations,
    independent_quality,
    isotropic_plane_stress,
    membrane_patch_field,
    rigid_body_basis,
    seeded_rotations,
    shape_family,
    triangle_from_angles,
)
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.e4_pl_s3_state import qualified_s3_triangle_frame
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.elements import DEFAULT_S3_FORMULATION, create_shell_element
from anyfesolver.fe_core import FEMesh


BANDS = ANGLE_BANDS_DEG + DIAGNOSTIC_BANDS_DEG[-1:]
SHAPES = [
    (band, name, angles)
    for band in BANDS
    for name, angles in shape_family(band).items()
]
SHAPE_IDS = [f"{band:g}deg-{name}" for band, name, _ in SHAPES]


# ---------------------------------------------------------------------------
# Geometry fixtures and envelope arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("angles", "expected_q"),
    [
        ((30.0, 30.0), 0.600000),
        ((20.0, 80.0), 0.558702),
        ((20.0, 20.0), 0.402503),
        ((15.0, 82.5), 0.433516),
        ((15.0, 15.0), 0.302169),
    ],
)
def test_normalized_area_table(angles: tuple[float, float], expected_q: float) -> None:
    quality = independent_quality(triangle_from_angles(*angles))
    assert quality.normalized_area == pytest.approx(expected_q, abs=5.0e-7)


@pytest.mark.parametrize(("band", "name", "angles"), SHAPES, ids=SHAPE_IDS)
def test_generator_and_solver_quality_agree(band, name, angles) -> None:
    coordinates = triangle_from_angles(*angles, length=2.5)
    quality = independent_quality(coordinates)
    assert quality.minimum_angle_deg == pytest.approx(band, abs=1.0e-9)
    assert sum(quality.angles_deg) == pytest.approx(180.0, abs=1.0e-9)
    _frame, _local, solver = qualified_s3_triangle_frame(
        coordinates, (0.0, 0.0, 1.0), enforce_admission=False
    )
    assert solver["minimum_angle_deg"] == pytest.approx(quality.minimum_angle_deg, abs=1.0e-9)
    assert solver["maximum_angle_deg"] == pytest.approx(quality.maximum_angle_deg, abs=1.0e-9)
    assert solver["edge_ratio"] == pytest.approx(quality.edge_ratio, rel=1.0e-12)
    assert solver["normalized_area"] == pytest.approx(quality.normalized_area, rel=1.0e-12)
    assert solver["minimum_scaled_jacobian"] == pytest.approx(
        quality.minimum_scaled_jacobian, rel=1.0e-12
    )


@pytest.mark.parametrize(
    ("envelope", "admitted_bands", "rejected_bands"),
    [
        ("historical_30", (30.0,), (25.0, 20.0, 15.0)),
        ("candidate_20", (30.0, 25.0, 20.0), (17.5, 15.0, 14.0)),
        ("candidate_15", (30.0, 25.0, 20.0, 17.5, 15.0), (14.0, 12.0, 10.0)),
    ],
)
def test_candidate_envelopes_admit_whole_shape_families(
    envelope, admitted_bands, rejected_bands
) -> None:
    """Every family shape at/above the floor is admitted; below it none is.

    This is the "coherent joint envelope" requirement: retaining the
    historical ``q >= 0.60`` with a lower angle floor would silently exclude
    obtuse shapes above the floor.
    """

    limits = CANDIDATE_ENVELOPES[envelope]
    for band in admitted_bands:
        for name, angles in shape_family(band).items():
            quality = independent_quality(triangle_from_angles(*angles))
            assert envelope_violations(quality, limits) == [], (envelope, band, name)
    for band in rejected_bands:
        for name, angles in shape_family(band).items():
            quality = independent_quality(triangle_from_angles(*angles))
            assert "minimum_angle" in envelope_violations(quality, limits), (band, name)


def test_candidate_normalized_area_floor_is_implied_by_angle_floor() -> None:
    """Numerically, ``q`` is minimized by the obtuse isosceles shape.

    For every triangle with minimum angle >= a and maximum angle <= 150,
    q >= q(a, a, 180 - 2a).  A dense sweep confirms the candidate q floors are
    never the binding constraint once the angle floor holds.
    """

    for floor, q_floor in ((20.0, 0.40), (15.0, 0.30)):
        worst = math.inf
        for alpha in np.linspace(floor, 60.0, 161):
            for beta in np.linspace(floor, 180.0 - 2.0 * floor, 241):
                gamma = 180.0 - alpha - beta
                if gamma < floor - 1e-12 or max(alpha, beta, gamma) > 150.0 + 1e-12:
                    continue
                worst = min(
                    worst,
                    independent_quality(triangle_from_angles(alpha, beta)).normalized_area,
                )
        bound = independent_quality(triangle_from_angles(floor, floor)).normalized_area
        assert worst == pytest.approx(bound, abs=1.0e-12)
        assert worst >= q_floor


# ---------------------------------------------------------------------------
# Live dispatch/admission map (G0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band", (20.0, 15.0))
def test_public_s3_route_is_v2d_and_admits_reduced_angles(band: float) -> None:
    assert DEFAULT_S3_FORMULATION == "e4-pl-s3-v2d"
    for name, angles in shape_family(band).items():
        coordinates = triangle_from_angles(*angles)
        for selector in (None, "default", "e4-pl-s3", "qualified-s3", "e4-pl-s3-v2d"):
            mesh = FEMesh()
            for node_id, xyz in enumerate(coordinates, start=1):
                mesh.add_node(node_id, *xyz)
            element = create_shell_element(
                1,
                [1, 2, 3],
                "steel",
                formulation=selector,
                thickness=0.01,
                reference_normal=(0.0, 0.0, 1.0),
            )
            assert type(element) is NativeParityE4PLS3V2DShellElement, (selector, name)


def test_historical_qualified_s3_guard_is_unchanged() -> None:
    """The MITC3+-class historical element still fails closed below 30 deg.

    It is reachable only by direct construction or by deserializing its own
    formulation identity, and its fingerprint binds the 30 deg envelope.  The
    angle investigation must not transfer V2D evidence to it.
    """

    for band, expected_ok in ((30.0, True), (20.0, False), (15.0, False)):
        coordinates = triangle_from_angles(band, 0.5 * (180.0 - band))
        mesh, _element, material = build_default_s3(coordinates, thickness=0.01)
        historical = QualifiedE4PLS3ShellElement(
            1, [1, 2, 3], "steel", thickness=0.01, reference_normal=(0.0, 0.0, 1.0)
        )
        if expected_ok:
            historical.compute_stiffness_matrix(mesh, material)
        else:
            with pytest.raises(ValueError, match="minimum angle is below 30 degrees"):
                historical.compute_stiffness_matrix(mesh, material)


# ---------------------------------------------------------------------------
# Element mechanics across angle bands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thickness_ratio", (1.0e-1, 1.0e-2, 1.0e-4))
@pytest.mark.parametrize(("band", "name", "angles"), SHAPES, ids=SHAPE_IDS)
def test_reference_elastic_contract(band, name, angles, thickness_ratio) -> None:
    length = 1.7
    coordinates = triangle_from_angles(*angles, length=length)
    thickness = thickness_ratio * length
    mesh, element, material = build_default_s3(coordinates, thickness=thickness)
    components = element.compute_stiffness_components(mesh, material)
    K = np.asarray(components["total"])
    area = independent_quality(coordinates).area

    Ks = dimensionless_stiffness(K, length)
    assert np.max(np.abs(Ks - Ks.T)) <= 1.0e-13

    rigid = rigid_body_basis(coordinates, length)
    diagonal = float(np.max(np.abs(np.diag(K))))
    assert np.max(np.abs(K @ rigid)) <= 1.0e-10 * diagonal

    # Twelve elastic modes; bending modes scale with (t/L)^2, so normalize.
    spectrum = elastic_spectrum(K, coordinates, length)
    assert spectrum.shape == (12,)
    assert float(spectrum[0]) / thickness_ratio**2 > 1.0e-3

    # Exact linear membrane patch with matching drill: PL energy vanishes.
    gradient = np.asarray(((1.0e-3, 3.0e-4), (-2.0e-4, 5.0e-4)))
    membrane = membrane_patch_field(coordinates, gradient)
    strain = np.asarray((gradient[0, 0], gradient[1, 1], gradient[0, 1] + gradient[1, 0]))
    exact = area * thickness * strain @ isotropic_plane_stress() @ strain
    assert membrane @ K @ membrane == pytest.approx(exact, rel=1.0e-10)
    assert abs(membrane @ components["pl"] @ membrane) <= 1.0e-12 * exact

    # Exact Kirchhoff constant-curvature patch: no transverse shear energy.
    hessian = np.asarray(((2.0e-3, 7.0e-4), (7.0e-4, -1.0e-3)))
    bending = bending_patch_field(coordinates, hessian)
    curvature = np.asarray((hessian[0, 0], hessian[1, 1], 2.0 * hessian[0, 1]))
    exact_b = area * thickness**3 / 12.0 * curvature @ isotropic_plane_stress() @ curvature
    assert bending @ K @ bending == pytest.approx(exact_b, rel=1.0e-10)
    assert abs(bending @ components["shear"] @ bending) <= 1.0e-10 * exact_b


@pytest.mark.parametrize("band", (20.0, 15.0))
def test_node_order_covariance(band: float) -> None:
    for name, angles in shape_family(band).items():
        coordinates = triangle_from_angles(*angles)
        mesh, base, material = build_default_s3(coordinates, thickness=0.02)
        reference = base.compute_stiffness_matrix(mesh, material)
        scale = float(np.max(np.abs(reference)))
        for order in itertools.permutations((1, 2, 3)):
            element = create_shell_element(
                1, list(order), "steel", thickness=0.02, reference_normal=(0.0, 0.0, 1.0)
            )
            made = element.compute_stiffness_matrix(mesh, material)
            permutation = np.zeros((18, 18))
            for external, node in enumerate(order):
                permutation[
                    6 * external : 6 * external + 6, 6 * (node - 1) : 6 * node
                ] = np.eye(6)
            np.testing.assert_allclose(
                made,
                permutation @ reference @ permutation.T,
                rtol=0.0,
                atol=1.0e-12 * scale,
                err_msg=f"{band} {name} {order}",
            )


@pytest.mark.parametrize("band", (20.0, 15.0))
def test_global_frame_objectivity_and_scale_covariance(band: float) -> None:
    for name, angles in shape_family(band).items():
        coordinates = triangle_from_angles(*angles)
        mesh, element, material = build_default_s3(coordinates, thickness=0.02)
        reference = element.compute_stiffness_matrix(mesh, material)
        scale = float(np.max(np.abs(reference)))
        for rotation in seeded_rotations(seed=int(band * 10), count=3):
            moved = coordinates @ rotation.T + np.asarray((3.0, -1.0, 7.0))
            normal = rotation @ np.asarray((0.0, 0.0, 1.0))
            mesh_r, element_r, _ = build_default_s3(moved, thickness=0.02, normal=normal)
            rotated = element_r.compute_stiffness_matrix(mesh_r, material)
            block = np.kron(np.eye(6), rotation)
            np.testing.assert_allclose(
                block.T @ rotated @ block,
                reference,
                rtol=0.0,
                atol=1.0e-11 * scale,
                err_msg=f"{band} {name}",
            )
        # Uniform scaling of geometry and thickness by s: translational
        # stiffness scales by s, rotational coupling by s^2 and s^3.
        s = 1.0e3
        mesh_s, element_s, _ = build_default_s3(coordinates * s, thickness=0.02 * s)
        scaled = element_s.compute_stiffness_matrix(mesh_s, material)
        weights = np.tile(np.asarray((1.0, 1.0, 1.0, s, s, s)), 3)
        expected = s * reference * weights[:, None] * weights[None, :]
        np.testing.assert_allclose(
            scaled, expected, rtol=0.0, atol=1.0e-11 * float(np.max(np.abs(expected)))
        )


def test_conditioning_degrades_smoothly_with_angle() -> None:
    """Record the geometry-caused conditioning trend; no cliff below 30 deg."""

    previous = None
    for band in (30.0, 25.0, 20.0, 17.5, 15.0):
        coordinates = triangle_from_angles(band, band)
        mesh, element, material = build_default_s3(coordinates, thickness=0.01)
        spectrum = elastic_spectrum(
            element.compute_stiffness_matrix(mesh, material), coordinates, 1.0
        )
        condition = float(spectrum[-1] / spectrum[0])
        if previous is not None:
            assert 1.0 < condition / previous < 2.0
        previous = condition
    # Obtuse 15/15/150 is the worst admitted candidate shape: < 8x the 30 deg
    # obtuse isosceles at t/L = 1e-2.
    assert previous < 8.0 * 1.4e6
