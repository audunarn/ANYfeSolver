"""Derivative qualification for the private native S3 TL strain kernel.

The oracle in this module is deliberately written from the incremental
MITC3+ equations rather than from the production derivative implementation.
It specializes the current configuration to a flat reference facet with
identity director triads.  In that configuration the eight physical strain
coordinates are a quadratic polynomial in the 20 uncondensed increments.

The 2015 nonlinear MITC3+ formulation supplies the independent identities:

* its Eq. (17) gives the linear and quadratic director increments;
* Eqs. (24)--(28) give the consistently truncated incremental
  Green--Lagrange strain; and
* Eqs. (29)--(31) apply the A--F assumed-covariant-shear interpolation to
  both the linear and quadratic terms.

No public API is exercised or established here.  The helper under test is a
private formulation kernel used to build the later condensed response.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import anyfesolver
import anyfesolver.e4_pl_s3_element as s3
from _e4_pl_s3_native_trial import native_trial_for_increment


_D = 1.0e-4
_TYING_POINTS = {
    "A": (1.0 / 6.0, 2.0 / 3.0),
    "B": (2.0 / 3.0, 1.0 / 6.0),
    "C": (1.0 / 6.0, 1.0 / 6.0),
    "D": (1.0 / 3.0 + _D, 1.0 / 3.0 - 2.0 * _D),
    "E": (1.0 / 3.0 - 2.0 * _D, 1.0 / 3.0 + _D),
    "F": (1.0 / 3.0 + _D, 1.0 / 3.0 + _D),
}


def _reference_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a skew flat triangle, identity triads, and its local nodes."""

    nodes = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.3, 0.0, 0.0),
            (0.37, 1.41, 0.0),
        ),
        dtype=float,
    )
    triads = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    return nodes, triads, nodes[:, :2].copy()


def _shape_data(
    local: np.ndarray,
    r: float,
    s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return corner/bubble fields and physical derivatives.

    The four director fields are ``(L1-b/3, L2-b/3, L3-b/3, b)``.  They
    sum to one, while defining source-node four as the corner mean plus the
    hierarchical bubble coordinate recovers ``sum(L_i theta_i) + b alpha``.
    """

    corner = np.asarray((1.0 - r - s, r, s), dtype=float)
    corner_r = np.asarray((-1.0, 1.0, 0.0), dtype=float)
    corner_s = np.asarray((-1.0, 0.0, 1.0), dtype=float)
    bubble = 27.0 * r * s * (1.0 - r - s)
    bubble_r = 27.0 * s * (1.0 - 2.0 * r - s)
    bubble_s = 27.0 * r * (1.0 - r - 2.0 * s)

    director = np.concatenate((corner - bubble / 3.0, (bubble,)))
    director_r = np.concatenate((corner_r - bubble_r / 3.0, (bubble_r,)))
    director_s = np.concatenate((corner_s - bubble_s / 3.0, (bubble_s,)))

    jacobian = np.asarray(
        (
            (local[1, 0] - local[0, 0], local[1, 1] - local[0, 1]),
            (local[2, 0] - local[0, 0], local[2, 1] - local[0, 1]),
        ),
        dtype=float,
    )
    inverse = np.linalg.inv(jacobian)

    def physical(reference_r: np.ndarray, reference_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            inverse[0, 0] * reference_r + inverse[0, 1] * reference_s,
            inverse[1, 0] * reference_r + inverse[1, 1] * reference_s,
        )

    corner_x, corner_y = physical(corner_r, corner_s)
    director_x, director_y = physical(director_r, director_s)
    return corner, corner_x, corner_y, director, director_x, director_y


def _increment_fields(
    local: np.ndarray,
    triads: np.ndarray,
    r: float,
    s: float,
    increment: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate Eq. (17)'s independent flat-reference increment fields."""

    (
        _corner,
        corner_x,
        corner_y,
        director,
        director_x,
        director_y,
    ) = _shape_data(local, r, s)
    made = np.asarray(increment, dtype=float).reshape(20)
    translations = made[:18].reshape(3, 6)[:, :3]
    rotations = made[:18].reshape(3, 6)[:, 3:6]

    # Pull the additive global rotation vector back to Eq. (14)'s minimal
    # director increment through the formulation's retained quadratic order.
    # A pure drill remains physical-null; its mixed tilt term is required for
    # second-order rigid-body objectivity.
    a_corner = np.einsum("ij,ij->i", rotations, triads[:3, :, 0])
    b_corner = np.einsum("ij,ij->i", rotations, triads[:3, :, 1])
    drill_corner = np.einsum("ij,ij->i", rotations, triads[:3, :, 2])
    a_second_corner = -0.5 * drill_corner * b_corner
    b_second_corner = 0.5 * drill_corner * a_corner
    a = np.concatenate((a_corner, (float(np.mean(a_corner) + made[18]),)))
    b = np.concatenate((b_corner, (float(np.mean(b_corner) + made[19]),)))
    a_second = np.concatenate(
        (a_second_corner, (float(np.mean(a_second_corner)),))
    )
    b_second = np.concatenate(
        (b_second_corner, (float(np.mean(b_second_corner)),))
    )

    # Eq. (17), including the second-order coordinate pullback.  Only the
    # first-order a/b fields are squared so the oracle is exactly quadratic.
    director_linear_sources = (
        -a[:, None] * triads[:, :, 1] + b[:, None] * triads[:, :, 0]
    )
    director_quadratic_sources = (
        -a_second[:, None] * triads[:, :, 1]
        + b_second[:, None] * triads[:, :, 0]
        - 0.5 * (a * a + b * b)[:, None] * triads[:, :, 2]
    )
    return {
        "translation_x": corner_x @ translations,
        "translation_y": corner_y @ translations,
        "director_linear": director @ director_linear_sources,
        "director_linear_x": director_x @ director_linear_sources,
        "director_linear_y": director_y @ director_linear_sources,
        "director_quadratic": director @ director_quadratic_sources,
        "director_quadratic_x": director_x @ director_quadratic_sources,
        "director_quadratic_y": director_y @ director_quadratic_sources,
    }


def _compatible_engineering_shear(
    local: np.ndarray,
    triads: np.ndarray,
    point: tuple[float, float],
    increment: np.ndarray,
) -> np.ndarray:
    """Evaluate twice the incremental ``(x,z)`` and ``(y,z)`` strains."""

    fields = _increment_fields(local, triads, *point, increment)
    ux = fields["translation_x"]
    uy = fields["translation_y"]
    linear = fields["director_linear"]
    quadratic = fields["director_quadratic"]
    e1 = np.asarray((1.0, 0.0, 0.0))
    e2 = np.asarray((0.0, 1.0, 0.0))
    normal = np.asarray((0.0, 0.0, 1.0))
    return np.asarray(
        (
            ux @ normal + e1 @ linear + ux @ linear + e1 @ quadratic,
            uy @ normal + e2 @ linear + uy @ linear + e2 @ quadratic,
        )
    )


def _independent_reference_strain(
    local: np.ndarray,
    triads: np.ndarray,
    r: float,
    s: float,
    increment: np.ndarray,
) -> np.ndarray:
    """Evaluate the eight consistently quadratic reference strains."""

    fields = _increment_fields(local, triads, r, s, increment)
    ux = fields["translation_x"]
    uy = fields["translation_y"]
    lx = fields["director_linear_x"]
    ly = fields["director_linear_y"]
    qx = fields["director_quadratic_x"]
    qy = fields["director_quadratic_y"]
    e1 = np.asarray((1.0, 0.0, 0.0))
    e2 = np.asarray((0.0, 1.0, 0.0))

    # Mid-surface coefficients of Eq. (24), in engineering Voigt order.
    membrane = np.asarray(
        (
            e1 @ ux + 0.5 * (ux @ ux),
            e2 @ uy + 0.5 * (uy @ uy),
            e1 @ uy + e2 @ ux + ux @ uy,
        )
    )

    # Linear-through-thickness coefficients after the consistently retained
    # terms in Eqs. (25)--(28).  Terms containing the current director
    # gradients vanish in this flat identity-triad reference fixture.
    curvature = np.asarray(
        (
            e1 @ lx + ux @ lx + e1 @ qx,
            e2 @ ly + uy @ ly + e2 @ qy,
            e1 @ ly + e2 @ lx + ux @ ly + lx @ uy + e1 @ qy + e2 @ qx,
        )
    )

    jacobian = np.asarray(
        (
            (local[1, 0] - local[0, 0], local[1, 1] - local[0, 1]),
            (local[2, 0] - local[0, 0], local[2, 1] - local[0, 1]),
        ),
        dtype=float,
    )
    inverse = np.linalg.inv(jacobian)
    covariant_samples = {
        name: jacobian
        @ _compatible_engineering_shear(local, triads, point, increment)
        for name, point in _TYING_POINTS.items()
    }
    constant_r = (
        (2.0 / 3.0)
        * (covariant_samples["B"][0] - 0.5 * covariant_samples["B"][1])
        + (1.0 / 3.0)
        * (covariant_samples["C"][0] + covariant_samples["C"][1])
    )
    constant_s = (
        (2.0 / 3.0)
        * (covariant_samples["A"][1] - 0.5 * covariant_samples["A"][0])
        + (1.0 / 3.0)
        * (covariant_samples["C"][0] + covariant_samples["C"][1])
    )
    twisting = (
        covariant_samples["F"][0]
        - covariant_samples["D"][0]
        - covariant_samples["F"][1]
        + covariant_samples["E"][1]
    )
    assumed_covariant = np.asarray(
        (
            constant_r + (twisting / 3.0) * (3.0 * s - 1.0),
            constant_s + (twisting / 3.0) * (1.0 - 3.0 * r),
        )
    )
    shear = inverse @ assumed_covariant
    return np.concatenate((membrane, curvature, shear))


def _polarize_quadratic(
    function: Callable[[np.ndarray], np.ndarray],
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover exact polynomial coefficients without differentiating code."""

    zero = np.zeros(dimension, dtype=float)
    constant = function(zero)
    basis = np.eye(dimension, dtype=float)
    plus = np.asarray([function(vector) for vector in basis])
    minus = np.asarray([function(-vector) for vector in basis])
    jacobian = (plus - minus).T / 2.0
    hessian = np.zeros((constant.size, dimension, dimension), dtype=float)
    diagonal = plus + minus - 2.0 * constant
    for index in range(dimension):
        hessian[:, index, index] = diagonal[index]
        for other in range(index):
            cross = (
                function(basis[index] + basis[other])
                - function(basis[index])
                - function(basis[other])
                + constant
            )
            hessian[:, index, other] = cross
            hessian[:, other, index] = cross
    return constant, jacobian, hessian


@pytest.mark.parametrize("point", [(0.23, 0.31), (1.0 / 3.0, 1.0 / 3.0)])
def test_reference_jacobian_is_the_existing_linear_mitc3_plus_operator(
    point: tuple[float, float],
) -> None:
    nodes, triads, local = _reference_fixture()
    native_trial, increment, _store = native_trial_for_increment(
        nodes, triads, np.zeros(20)
    )
    values, jacobian, hessian = s3._native_incremental_strain_jets(
        nodes,
        triads,
        *point,
        increment,
        reference_nodes=nodes,
        reference_frame=np.eye(3),
        native_rotation_trial=native_trial,
    )

    expected = np.zeros((8, 20), dtype=float)
    old_operator = s3._kinematic_matrix(local, *point)
    expected[:, s3.PHYSICAL_EXTERNAL_INDICES] = old_operator[:, :15]
    expected[:, 18:] = old_operator[:, 15:]

    np.testing.assert_allclose(values, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(jacobian, expected, rtol=2.0e-13, atol=2.0e-13)
    assert hessian.shape == (8, 20, 20)
    np.testing.assert_allclose(hessian, np.swapaxes(hessian, 1, 2), atol=2.0e-14)
    np.testing.assert_array_equal(jacobian[:, (5, 11, 17)], 0.0)
    drill = np.asarray((5, 11, 17))
    np.testing.assert_array_equal(hessian[:, drill, drill], 0.0)
    assert np.linalg.norm(hessian[:, drill, :]) > 0.0


def test_values_jacobian_and_hessian_match_directional_finite_differences() -> None:
    nodes, triads, _local = _reference_fixture()
    point = (0.271, 0.417)
    rng = np.random.default_rng(20260825)
    increment = 0.035 * rng.standard_normal(20)
    increment[[5, 11, 17]] = rng.standard_normal(3)  # arbitrary drilling

    def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        native_trial, exact, _store = native_trial_for_increment(
            nodes, triads, candidate
        )
        return s3._native_incremental_strain_jets(
            nodes,
            triads,
            *point,
            exact,
            reference_nodes=nodes,
            reference_frame=np.eye(3),
            native_rotation_trial=native_trial,
        )

    values, jacobian, hessian = evaluate(increment)
    direction = rng.standard_normal(20)
    direction /= np.linalg.norm(direction)
    step = 2.0e-5
    plus_value, plus_jacobian, _plus_hessian = evaluate(increment + step * direction)
    minus_value, minus_jacobian, _minus_hessian = evaluate(increment - step * direction)
    np.testing.assert_allclose(
        (plus_value - minus_value) / (2.0 * step),
        jacobian @ direction,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        (plus_jacobian - minus_jacobian) / (2.0 * step),
        np.einsum("ijk,k->ij", hessian, direction),
        rtol=4.0e-7,
        atol=4.0e-8,
    )
    assert np.all(np.isfinite(values))


def test_directional_value_and_derivative_identities_include_assumed_shear() -> None:
    nodes, triads, _local = _reference_fixture()
    point = (0.19, 0.52)
    rng = np.random.default_rng(310729)
    increment = 0.11 * rng.standard_normal(20)
    direction = 0.23 * rng.standard_normal(20)
    direction /= np.linalg.norm(direction)
    step = 1.0e-5

    def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        native_trial, exact, _store = native_trial_for_increment(
            nodes, triads, candidate
        )
        return s3._native_incremental_strain_jets(
            nodes,
            triads,
            *point,
            exact,
            reference_nodes=nodes,
            reference_frame=np.eye(3),
            native_rotation_trial=native_trial,
        )

    value, jacobian, hessian = evaluate(increment)
    plus_value, plus_jacobian, plus_hessian = evaluate(increment + step * direction)
    minus_value, minus_jacobian, minus_hessian = evaluate(increment - step * direction)
    first_directional = jacobian @ direction

    # Compare all eight fields, then repeat the assertions explicitly on the
    # two assumed-shear rows so an accidental membrane-only implementation
    # cannot satisfy this derivative qualification unnoticed.
    np.testing.assert_allclose(
        (plus_value - minus_value) / (2.0 * step),
        first_directional,
        rtol=3.0e-7,
        atol=3.0e-8,
    )
    np.testing.assert_allclose(
        (plus_jacobian - minus_jacobian) / (2.0 * step),
        np.einsum("ijk,k->ij", hessian, direction),
        rtol=5.0e-7,
        atol=5.0e-8,
    )
    np.testing.assert_allclose(
        ((plus_value - minus_value) / (2.0 * step))[6:],
        first_directional[6:],
        rtol=3.0e-7,
        atol=3.0e-8,
    )
    assert np.all(np.isfinite(plus_hessian))
    assert np.all(np.isfinite(minus_hessian))
    assert np.linalg.norm(hessian[6:]) > 1.0e-6


def test_private_kernel_is_not_exposed_as_a_supported_api() -> None:
    assert "_native_incremental_strain_jets" not in s3.__all__
    assert not hasattr(anyfesolver, "_native_incremental_strain_jets")
