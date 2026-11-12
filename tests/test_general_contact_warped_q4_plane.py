"""Independent local work checks for the unadmitted warped Q4 plane mortar."""

import numpy as np
import pytest

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.warped_q4_plane import WarpedQ4PlaneMortar
from anyfesolver.general_contact.warped_q4_skin import WarpedQ4Skin


_WARPED = np.array(((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.08),
                    (1.0, 1.0, -0.04),
                    (0.0, 1.0, 0.03)))


def _skin(corners=_WARPED, side="top"):
    model = FEModel("warped_q4_local_work")
    model.add_material("elastic", 1.0e5, 0.3)
    for node, position in enumerate(corners, 1):
        model.add_node(node, *position)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.08,
        reference_normal=(0.0, 0.0, 1.0),
    ))
    return WarpedQ4Skin(model, 1, side)


def _manual_point(corners, xi, eta, offset):
    shape = 0.25 * np.array(((1 - xi) * (1 - eta),
                             (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta),
                             (1 - xi) * (1 + eta)))
    derivative_xi = 0.25 * np.array((-(1 - eta), 1 - eta,
                                     1 + eta, -(1 + eta)))
    derivative_eta = 0.25 * np.array((-(1 - xi), -(1 + xi),
                                      1 + xi, 1 - xi))
    cross = np.cross(derivative_xi @ corners, derivative_eta @ corners)
    director = cross / np.linalg.norm(cross)
    position = shape @ corners + offset * director
    return shape, position, director


def _manual_integrals(corners, offset, point, normal, order=20):
    abscissae, weights = np.polynomial.legendre.leggauss(order)
    support = np.zeros(4)
    weighted = np.zeros(4)
    gradient = np.zeros((4, 4, 6))
    samples = []
    h = 1.0e-6
    for xi, weight_xi in zip(abscissae, weights):
        for eta, weight_eta in zip(abscissae, weights):
            shape, position, director = _manual_point(corners, xi, eta, offset)
            tangent_xi = (_manual_point(corners, xi + h, eta, offset)[1]
                          - _manual_point(corners, xi - h, eta, offset)[1]) / (2 * h)
            tangent_eta = (_manual_point(corners, xi, eta + h, offset)[1]
                           - _manual_point(corners, xi, eta - h, offset)[1]) / (2 * h)
            measure = float(weight_xi * weight_eta
                            * np.linalg.norm(np.cross(tangent_xi, tangent_eta)))
            gap = float(normal @ (position - point))
            work = np.r_[normal, offset * np.cross(director, normal)]
            support += measure * shape
            weighted += measure * shape * gap
            gradient += measure * shape[:, None, None] * shape[None, :, None] * work
            samples.append((shape, position, measure))
    return support, weighted, gradient.reshape(4, 24), samples


def test_warped_q4_local_work_matches_independent_surface_integrals_and_tangent():
    skin = _skin()
    point = np.array((0.0, 0.0, 0.2))
    normal = np.array((0.0, 0.0, 1.0))
    law = WarpedQ4PlaneMortar(
        skin, point, normal, 1.0e6,
        quadrature_order=6, control_order=12,
        integration_relative_tolerance=1.0e-6,
    )
    support, weighted, gradient, samples = _manual_integrals(
        _WARPED, 0.04, point, normal,
    )
    np.testing.assert_allclose(law.support, support, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(law.reference_weighted_gap, weighted,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(law.gap_gradient, gradient,
                               rtol=1e-7, atol=1e-9)
    zero = np.zeros((4, 6))
    state = law.evaluate(zero)
    gap = weighted / support
    energy = 0.5 * 1.0e6 * float(support @ (np.minimum(gap, 0.0) ** 2))
    assert state.energy == pytest.approx(energy, rel=1e-7)
    direction = np.array((
        (0.12, -0.13, 0.24, 0.01, -0.02, 0.03),
        (-0.08, 0.10, -0.21, -0.04, 0.03, -0.01),
        (0.05, 0.02, 0.15, 0.02, 0.01, -0.02),
        (-0.11, 0.03, -0.09, -0.03, -0.02, 0.01),
    ))
    h = 1.0e-7
    plus, minus = law.evaluate(h * direction), law.evaluate(-h * direction)
    numerical_work = (plus.energy - minus.energy) / (2 * h)
    assert numerical_work == pytest.approx(
        float(np.sum(state.residual * direction)), rel=1e-7,
    )
    numerical_tangent = (plus.residual - minus.residual).reshape(24) / (2 * h)
    np.testing.assert_allclose(numerical_tangent,
                               state.tangent @ direction.reshape(24),
                               rtol=1e-6, atol=1e-5)

    force_oracle = np.zeros(3)
    moment_oracle = np.zeros(3)
    for shape, position, measure in samples:
        traction = float(shape @ state.pressure_coefficients) * normal
        force_oracle += measure * traction
        moment_oracle += measure * np.cross(position, traction)
    force_from_work = -np.sum(state.residual[:, :3], axis=0)
    moment_from_work = -(
        np.sum(np.cross(_WARPED, state.residual[:, :3]), axis=0)
        + np.sum(state.residual[:, 3:], axis=0)
    )
    np.testing.assert_allclose(force_from_work, force_oracle, rtol=1e-7, atol=1e-5)
    np.testing.assert_allclose(moment_from_work, moment_oracle,
                               rtol=1e-7, atol=1e-5)
    omega = np.array((0.07, -0.03, 0.05))
    rigid_mode = np.column_stack((np.cross(omega, _WARPED),
                                  np.broadcast_to(omega, (4, 3))))
    assert float(np.sum(state.residual * rigid_mode)) == pytest.approx(
        -float(omega @ moment_oracle), rel=1e-7,
    )


def test_warped_q4_local_drill_and_integration_budget_are_explicit():
    skin = _skin()
    point = np.array((0.0, 0.0, 0.2))
    normal = np.array((0.0, 0.0, 1.0))
    sample = skin.sample(0.3, -0.2)
    assert float(sample.director @ np.cross(sample.director, normal)) == pytest.approx(
        0.0, abs=1e-15,
    )
    midsurface = _skin(side="midsurface")
    law = WarpedQ4PlaneMortar(
        midsurface, point, normal, 1.0e6,
        quadrature_order=6, control_order=12,
        integration_relative_tolerance=1e-6,
    )
    np.testing.assert_array_equal(law.gap_gradient.reshape(4, 4, 6)[:, :, 3:],
                                  0.0)
    with pytest.raises(ValueError, match="integration tolerance exceeded"):
        WarpedQ4PlaneMortar(
            skin, point, normal, 1.0e6,
            quadrature_order=1, control_order=2,
            integration_relative_tolerance=1e-12,
        )


def test_top_and_bottom_skins_have_opposite_eccentric_contact_couples():
    planar = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                       (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    normal = np.array((0.2, 0.0, np.sqrt(0.96)))
    top_point = np.array((0.0, 0.0, 0.2))
    bottom_point = top_point - 0.08 * normal[2] * normal
    top = WarpedQ4PlaneMortar(
        _skin(planar, "top"), top_point, normal, 1.0e6,
        quadrature_order=4, control_order=8,
        integration_relative_tolerance=1e-8,
    ).evaluate(np.zeros((4, 6)))
    bottom = WarpedQ4PlaneMortar(
        _skin(planar, "bottom"), bottom_point, normal, 1.0e6,
        quadrature_order=4, control_order=8,
        integration_relative_tolerance=1e-8,
    ).evaluate(np.zeros((4, 6)))
    assert top.energy == pytest.approx(bottom.energy, rel=1e-12)
    np.testing.assert_allclose(top.residual[:, :3], bottom.residual[:, :3],
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(top.residual[:, 3:], -bottom.residual[:, 3:],
                               rtol=1e-12, atol=1e-9)
    assert np.linalg.norm(top.residual[:, 3:]) > 0.0
