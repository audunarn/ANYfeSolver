"""Independent checks for original-Q1 Q4 pressure-side overlap integrals."""

import numpy as np
import pytest

from anyfesolver.general_contact.q4_source_overlap import q4_source_overlap_integrals
from anyfesolver.general_contact.q4_source_mortar_energy import MovingQ4PrimalMortarPenalty


def _faces():
    reference = np.array(((0., 0., .1), (1., 0., .1),
                          (1., 1., .1), (0., 1., .1)))
    target = np.array(((-2., -2., 0.), (5., -2., 0.),
                       (-2., 5., 0.)))
    return reference, target


def _evaluate(source, *, area_tolerance=1e-11,
              weighted_gap_tolerance=1e-11,
              maximum_triangles=500):
    reference, target = _faces()
    return q4_source_overlap_integrals(
        reference, source, target,
        area_tolerance=area_tolerance,
        weighted_gap_tolerance=weighted_gap_tolerance,
        maximum_triangles=maximum_triangles,
    )


def test_q4_source_uniform_gap_reproduces_constant_pressure():
    reference, _ = _faces()
    current = reference.copy()
    current[:, 2] = -.02
    result = _evaluate(current)
    assert result.overlap_vertices == 4
    assert result.overlap_area == pytest.approx(1.)
    assert result.reference_overlap_area == pytest.approx(1.)
    np.testing.assert_allclose(result.support, .25, atol=1e-12)
    np.testing.assert_allclose(result.weighted_gap, -.005, atol=1e-12)
    gram = np.array(((1/9, 1/18, 1/36, 1/18),
                     (1/18, 1/9, 1/18, 1/36),
                     (1/36, 1/18, 1/9, 1/18),
                     (1/18, 1/36, 1/18, 1/9)))
    np.testing.assert_allclose(result.pressure_gram, gram, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(result.pressure_gram)) > 0.
    assert result.integration_triangles <= 500


def test_q4_source_warped_trial_matches_independent_gauss_integral():
    reference, _ = _faces()
    current = reference.copy()
    current[:, 2] = -.02
    current[2, 2] += .03
    result = _evaluate(current)
    abscissas, weights = np.polynomial.legendre.leggauss(16)
    abscissas = (abscissas+1.)/2.
    weights = weights/2.
    expected = np.zeros(4)
    for x, wx in zip(abscissas, weights):
        for y, wy in zip(abscissas, weights):
            shape = np.array(((1-x)*(1-y), x*(1-y), x*y, (1-x)*y))
            z = -.02+.03*x*y
            normal_z = 1./np.sqrt(1.+.03**2*(x*x+y*y))
            expected += wx*wy*shape*z*normal_z
    np.testing.assert_allclose(result.support, .25, atol=1e-11)
    np.testing.assert_allclose(result.weighted_gap, expected, atol=2e-11)
    assert result.weighted_gap_error <= 1e-11
    assert result.area_error <= 1e-11


def test_q4_source_fails_closed_on_inversion_and_resource_limit():
    reference, _ = _faces()
    current = reference.copy()
    current[:, 2] = -.02
    with pytest.raises(ValueError, match="inverted|folded|polarity"):
        _evaluate(current[[0, 3, 2, 1]])
    with pytest.raises(MemoryError, match="budget"):
        _evaluate(current, maximum_triangles=1)


def test_five_vertex_overlap_fan_sliver_keeps_global_error_budget():
    # Captured from a finite-slide Q4/S3 candidate. Clipping leaves one
    # 3.84e-10 m2 fan triangle beside two 0.045 m2 triangles. Its area-scaled
    # budget was smaller than floating-point quadrature noise.
    reference = np.array((
        (.85, .1, .01504), (.975, .1, .01504),
        (.975, 1.1, .01504), (.85, 1.1, .01504),
    ))
    current = np.array((
        (.8999999991968173, .1, .015000876007947613),
        (1.0249999990234167, .1, .014993759922196752),
        (1.024999999017355, 1.1, .014993693383378352),
        (.8999999991918018, 1.1, .015000817844115952),
    ))
    target = np.array((
        (1., 0., .015), (1., 1., .015), (0., 1., .015),
    ))
    result = q4_source_overlap_integrals(
        reference, current, target,
        area_tolerance=1e-9, weighted_gap_tolerance=1e-10,
        maximum_triangles=300,
    )
    assert result.overlap_vertices == 5
    assert result.overlap_area == pytest.approx(.09, abs=1e-8)
    assert np.sum(result.support) == pytest.approx(result.reference_overlap_area,
                                                   abs=1e-9)
    assert result.area_error <= 1e-9
    assert result.weighted_gap_error <= 1e-10
    assert result.integration_triangles <= 300


def test_q4_source_pressure_law_has_constant_traction_and_balanced_work():
    reference, target = _faces()
    current = reference.copy()
    current[:, 2] = -.02
    law = MovingQ4PrimalMortarPenalty(
        reference, target, 1000., difference_step=2e-5,
        area_tolerance=1e-10, weighted_gap_tolerance=1e-10,
        maximum_integration_triangles=100,
    )
    state = law.evaluate(current, target)
    np.testing.assert_allclose(state.normalized_gap, -.02, atol=1e-12)
    np.testing.assert_allclose(state.pressure, 20., atol=1e-10)
    assert state.energy == pytest.approx(.2, abs=1e-10)
    work = law.work(current, target, tangent=True)
    assert work.residual.shape == (21,)
    assert work.tangent.shape == (21, 21)
    np.testing.assert_allclose(work.residual.reshape(7, 3).sum(axis=0),
                               0., atol=1e-8*np.linalg.norm(work.residual))
    np.testing.assert_allclose(work.residual[:12].reshape(4, 3)[:, 2],
                               -5., atol=1e-7)
    np.testing.assert_allclose(work.tangent, work.tangent.T, atol=1e-8)


def test_q4_source_warped_trial_tangent_matches_independent_directional_step():
    reference, target = _faces()
    current = reference.copy()
    current[:, 2] = -.02
    current[2, 2] += .03
    law = MovingQ4PrimalMortarPenalty(
        reference, target, 1000., difference_step=2e-5,
        area_tolerance=1e-9, weighted_gap_tolerance=1e-10,
        maximum_integration_triangles=300,
    )
    work = law.work(current, target, tangent=True)
    direction = np.zeros(21)
    direction[0] = .2
    direction[8] = 1.
    direction[15] = -.1
    h = 1e-4
    def gradient(sign):
        trial = np.concatenate((current, target)).reshape(-1)+sign*h*direction
        return law.work(trial[:12].reshape(4, 3),
                        trial[12:].reshape(3, 3), tangent=False).residual
    control = (gradient(1.)-gradient(-1.))/(2.*h)
    actual = work.tangent@direction
    error = np.linalg.norm(actual-control)/max(1., np.linalg.norm(control))
    assert error <= 1e-5


def test_q4_q4_bilinear_target_matches_independent_gap_integral_and_work():
    reference, _ = _faces()
    target_reference = np.array(((-2., -2., 0.), (3., -2., 0.),
                                 (3., 3., 0.), (-2., 3., 0.)))
    current = reference.copy()
    current[:, 2] = -.02
    target = target_reference.copy()
    target[2, 2] = .03
    law = MovingQ4PrimalMortarPenalty(
        reference, target_reference, 1000., difference_step=2e-5,
        area_tolerance=1e-10, weighted_gap_tolerance=1e-11,
        maximum_integration_triangles=300,
    )
    state = law.evaluate(current, target)
    abscissas, weights = np.polynomial.legendre.leggauss(12)
    abscissas = (abscissas+1.)/2.
    weights = weights/2.
    expected = np.zeros(4)
    for x, wx in zip(abscissas, weights):
        for y, wy in zip(abscissas, weights):
            shape = np.array(((1-x)*(1-y), x*(1-y), x*y, (1-x)*y))
            target_height = .03*((x+2.)/5.)*((y+2.)/5.)
            expected += wx*wy*shape*(-.02-target_height)
    np.testing.assert_allclose(state.reference_support, .25, atol=1e-10)
    np.testing.assert_allclose(state.reference_weighted_gap, expected, atol=1e-10)
    assert state.overlap_area == pytest.approx(1.)
    work = law.work(current, target, tangent=False)
    assert work.residual.shape == (24,)
    assert abs(work.residual[12+2*3+2]) > 0.
    np.testing.assert_allclose(work.residual.reshape(8, 3).sum(axis=0),
                               0., atol=1e-8*np.linalg.norm(work.residual))
