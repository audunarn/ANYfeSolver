"""Independent scalar and pressure-space checks for moving S3 mortar."""

import numpy as np
import pytest

from anyfesolver.general_contact.triangle_mortar_energy import (
    MovingS3PrimalMortarPenalty,
)


def _triangles():
    source = np.array(((0., 0., 0.08), (1., 0., 0.08),
                       (0., 1., 0.08)))
    target = np.array(((0.05, 0.05, 0.), (0.45, 0.05, 0.),
                       (0.05, 0.45, 0.)))
    return source, target


def test_moving_s3_mortar_full_support_constant_pressure_and_rank():
    source, _ = _triangles()
    target = source.copy()
    target[:, 2] = 0.0
    law = MovingS3PrimalMortarPenalty(
        source, target, 1000.0, difference_step=1e-5,
    )
    compressed = source.copy()
    compressed[:, 2] = -0.02
    state = law.evaluate(compressed, target)
    np.testing.assert_allclose(state.reference_support, 0.5/3.0, atol=1e-14)
    np.testing.assert_allclose(state.normalized_gap, -0.02, atol=1e-14)
    np.testing.assert_allclose(state.pressure, 20.0, atol=1e-12)
    assert state.energy == pytest.approx(0.1)
    assert state.active_pressure_dofs == 3
    np.testing.assert_allclose(np.linalg.eigvalsh(state.pressure_gram),
                               (0.5/12.0, 0.5/12.0, 0.5/3.0), atol=1e-14)
    assert np.sum(state.reference_support*state.pressure) == pytest.approx(10.0)


def test_moving_s3_mortar_partial_overlap_reference_weights_are_exact():
    source, target = _triangles()
    law = MovingS3PrimalMortarPenalty(
        source, target, 1000.0, difference_step=1e-5,
    )
    compressed = source.copy()
    compressed[:, 2] = -0.02
    state = law.evaluate(compressed, target)
    area = 0.5*0.4*0.4
    expected = area*np.array((1.0-0.05-0.05-0.4/3.0-0.4/3.0,
                              0.05+0.4/3.0, 0.05+0.4/3.0))
    np.testing.assert_allclose(state.reference_support, expected, atol=1e-14)
    np.testing.assert_allclose(state.reference_weighted_gap,
                               -0.02*expected, atol=1e-14)
    assert state.reference_overlap_area == pytest.approx(area)
    assert state.energy == pytest.approx(0.5*1000.0*area*0.02**2)
    assert np.min(np.linalg.eigvalsh(state.pressure_gram)) > 0.0
    assert state.pressure.shape == (3,)


def test_moving_s3_mortar_reference_measure_survives_in_plane_stretch():
    source, _ = _triangles()
    target = np.array(((0., 0., 0.), (3., 0., 0.), (0., 3., 0.)))
    law = MovingS3PrimalMortarPenalty(
        source, target, 1000.0, difference_step=1e-5,
    )
    stretched = source.copy()
    stretched[:, :2] *= 2.0
    stretched[:, 2] = -0.02
    state = law.evaluate(stretched, target)
    assert state.overlap_area == pytest.approx(2.0)
    assert state.reference_overlap_area == pytest.approx(0.5)
    np.testing.assert_allclose(state.reference_support, 0.5/3.0, atol=1e-14)
    np.testing.assert_allclose(state.pressure, 20.0, atol=1e-12)
    assert state.energy == pytest.approx(0.1)


def test_moving_s3_mortar_gradient_obeys_independent_directional_work():
    source, target = _triangles()
    law = MovingS3PrimalMortarPenalty(
        source, target, 1000.0, difference_step=2e-5,
    )
    current = source.copy()
    current[:, 2] = (-0.015, -0.02, -0.025)
    current[1, 0] += 0.01
    work = law.gradient(current, target)
    assert work.state.active_pressure_dofs == 3
    assert work.energy_evaluations == 37
    direction = np.zeros(18)
    direction[0], direction[4], direction[8] = 0.1, -0.07, 0.11
    direction[11], direction[15] = -0.04, 0.06
    step = 5e-6
    positions = np.concatenate((current, target)).reshape(-1)
    def energy_at(sign):
        trial = (positions + sign*step*direction).reshape(6, 3)
        return law.evaluate(trial[:3], trial[3:]).energy
    measured = (energy_at(1.0)-energy_at(-1.0))/(2*step)
    assert measured == pytest.approx(float(work.gradient @ direction),
                                     rel=1e-5, abs=1e-7)
    np.testing.assert_allclose(
        np.sum(work.gradient.reshape(6, 3), axis=0),
        0.0, atol=1e-8,
    )
    tangent = law.work(current, target, tangent=True)
    assert tangent.energy_evaluations <= law.maximum_energy_evaluations
    plus = (positions + step*direction).reshape(6, 3)
    minus = (positions - step*direction).reshape(6, 3)
    measured_tangent = (
        law.gradient(plus[:3], plus[3:]).gradient
        - law.gradient(minus[:3], minus[3:]).gradient
    )/(2*step)
    relative = float(np.linalg.norm(measured_tangent-tangent.tangent@direction)
                     / np.linalg.norm(measured_tangent))
    assert relative <= 1e-5


def test_moving_s3_mortar_opening_and_budget_failure():
    source, target = _triangles()
    law = MovingS3PrimalMortarPenalty(
        source, target, 1000.0, difference_step=1e-5,
        maximum_energy_evaluations=1,
    )
    opening = law.gradient(source, target)
    assert opening.state.energy == 0.0
    np.testing.assert_array_equal(opening.gradient, 0.0)
    closed = source.copy()
    closed[:, 2] = -0.02
    with pytest.raises(MemoryError, match="budget exhausted"):
        law.gradient(closed, target)


def test_moving_s3_mortar_pressure_activation_has_bounded_generalized_tangent():
    source, _ = _triangles()
    target = np.array(((-1., -1., 0.), (3., -1., 0.),
                       (-1., 3., 0.)))
    law = MovingS3PrimalMortarPenalty(
        source, target, 1000.0, difference_step=1e-5,
    )
    current = source.copy()
    current[:, 2] = (-0.03, 0.01, 0.01)
    work = law.work(current, target, tangent=True)
    assert work.state.energy > 0.0
    assert max(abs(work.state.normalized_gap[1:])) < 1e-12
    assert work.energy_evaluations <= law.maximum_energy_evaluations
    assert np.all(np.isfinite(work.residual))
    assert np.all(np.isfinite(work.tangent))
    np.testing.assert_allclose(work.tangent, work.tangent.T, atol=0.0)
