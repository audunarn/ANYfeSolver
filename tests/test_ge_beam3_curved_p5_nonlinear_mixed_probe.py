"""Integrated nonlinear density/condensation research, not beam qualification."""

import numpy as np
import pytest

from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import (
    mixed_energy, reduced_energy, reference_hessians, rotation, T6, T3,
)
from docs.reference_cases.ge_beam3_curved_p5_finite_probe import CurvedFiniteProbe
from docs.reference_cases.ge_beam3_curved_p5_section_probe import DirectedHardeningSectionProbe, SectionHistory
from docs.reference_cases.ge_beam3_curved_p5_nonlinear_mixed_probe import NonlinearMixedBeamProbe, NonlinearLocalError
from test_ge_beam3_curved_p5_algebra_probe import reference, section, perturbed


A = np.array([1., .2, -.1, .3, -.4, .5])


def law(yield_force=.02):
    return DirectedHardeningSectionProbe(section(), A, yield_force, .4)


def mask(result):
    return tuple(station.response.plastic_active for station in result.stations)


def test_zero_state_full_hessian_matches_existing_elastic_reference():
    ref = reference(.4)
    model = NonlinearMixedBeamProbe(ref, law())
    made = model.evaluate(ref.coordinates, ref.nodal_triads, np.tile(np.eye(3), (2, 1, 1)), np.zeros((2, 2, 3)))
    expected = reference_hessians(ref, section(), 24)[0]
    assert np.linalg.norm(made.residual) <= 1e-11
    assert abs(made.potential) <= 1e-11
    assert np.linalg.norm(made.hessian-expected) <= 1e-11*np.linalg.norm(expected)
    assert len(made.stations) == 48 and not any(mask(made))
    assert [(s.cell, s.index) for s in made.stations] == [(c, i) for c in (0, 1) for i in range(24)]


@pytest.mark.parametrize('force', [None, [.1, -.2, .3]])
def test_elastic_limit_moment_and_rotation_condensation_match_original(force):
    ref = reference(.6, .2, .15)
    x, q, u = perturbed(ref)
    model = NonlinearMixedBeamProbe(ref, law(1000.), order=8, line_force=force)
    _, moments = reduced_energy(ref, section(), x, q, u, 8)
    uncondensed = model.evaluate(x, q, u, moments)
    h = uncondensed.hessian
    reduced = h[:24, :24]-h[:24, 24:] @ np.linalg.solve(h[24:, 24:], h[24:, :24])
    original_model = CurvedFiniteProbe(ref, section(), order=8, line_force=force)
    original = original_model._jet(x, q, u, external=True)
    assert abs(uncondensed.potential-original.value) <= 1e-11
    assert np.linalg.norm(uncondensed.residual[:24]-original.gradient) <= 1e-11
    assert np.linalg.norm(uncondensed.residual[24:]) <= 1e-11
    assert np.linalg.norm(reduced-original.hessian) <= 1e-11*np.linalg.norm(original.hessian)
    new_solution = model.solve(x, q)
    old_solution = original_model.evaluate(x, q)
    assert abs(new_solution.potential-old_solution.potential) <= 1e-11
    assert np.linalg.norm(new_solution.residual-old_solution.residual) <= 1e-11
    assert np.linalg.norm(new_solution.tangent-old_solution.tangent) <= 1e-11*np.linalg.norm(old_solution.tangent)


def test_elastic_uneliminated_scalar_matches_station_reconstruction():
    ref = reference(.4)
    x, q, u = perturbed(ref)
    m = np.cos(np.arange(12)).reshape(2, 2, 3)/7
    made = NonlinearMixedBeamProbe(ref, law(1000.), order=8).evaluate(x, q, u, m)
    assert abs(made.potential-mixed_energy(ref, section(), x, q, u, m, 8)) <= 1e-11


def test_plastic_full_gradient_hessian_and_geometric_chain_rule():
    ref = reference(.4)
    x, q, u = perturbed(ref)
    moments = np.cos(np.arange(12)).reshape(2, 2, 3)/7
    model = NonlinearMixedBeamProbe(ref, law(), order=8, line_force=[.1, -.2, .3])
    base = model.evaluate(x, q, u, moments)
    direction = np.sin(np.arange(36)+.4)/9
    step = 1e-6
    plus = model.evaluate(x, q, u, moments, increment=step*direction)
    minus = model.evaluate(x, q, u, moments, increment=-step*direction)
    assert any(mask(base)) and mask(base) == mask(plus) == mask(minus)
    assert abs((plus.potential-minus.potential)/(2*step)-base.residual @ direction) <= 1e-7
    assert np.linalg.norm((plus.residual-minus.residual)/(2*step)-base.hessian @ direction) <= 1e-7
    assert np.linalg.norm(base.hessian-base.hessian.T) <= 1e-11


def test_missing_geometric_second_variation_is_detected(monkeypatch):
    import docs.reference_cases.ge_beam3_curved_p5_nonlinear_mixed_probe as module
    from anyfesolver._ge_beam3_mixed_ad import Jet2
    ref = reference(.4)
    x, q, u = perturbed(ref)
    moments = np.cos(np.arange(12)).reshape(2, 2, 3)/7
    model = NonlinearMixedBeamProbe(ref, law(), order=8)
    original = model.evaluate(x, q, u, moments)
    def omitted_curvature(inputs, density):
        jacobian = np.array([item.gradient for item in inputs])
        return Jet2(density.potential, jacobian.T @ density.gradient, jacobian.T @ density.hessian @ jacobian)
    monkeypatch.setattr(module, 'compose_density', omitted_curvature)
    mutated = model.evaluate(x, q, u, moments)
    assert np.linalg.norm(mutated.residual-original.residual) <= 1e-11
    assert np.linalg.norm(mutated.hessian-original.hessian) > .01


def test_stationary_plastic_response_recovery_and_fixed_origins():
    ref = reference(.4)
    x, q, _ = perturbed(ref)
    origins = tuple(SectionHistory(.0001*i, .0002*i) for i in range(16))
    model = NonlinearMixedBeamProbe(ref, law(), order=8, origins=origins)
    made = model.solve(x, q)
    assert made.local_residual_norm <= 1e-11 and any(mask(made))
    assert model.origins == origins
    assert made.evaluations <= 64 and made.iterations <= 25
    points, _ = np.polynomial.legendre.leggauss(8)
    for index, station in enumerate(made.stations):
        response = station.response
        assert response.origin == origins[index]
        assert response.history.accumulated >= response.origin.accumulated
        assert response.dissipation_increment >= 0
        t = (points[station.index]+1)/2
        moment = (1-t)*made.moments[station.cell, 0]+t*made.moments[station.cell, 1]
        assert np.linalg.norm(response.resultants[3:]-moment) <= 1e-11
        full_return = law().strain_response(response.strain, response.origin)
        assert np.linalg.norm(full_return.resultants-response.resultants) <= 1e-11


def test_condensed_plastic_tangent_uses_common_external_increment_chart():
    ref = reference(.4)
    x, q, _ = perturbed(ref)
    model = NonlinearMixedBeamProbe(ref, law(), order=8)
    base = model.solve(x, q)
    direction = np.sin(np.arange(18)+.2)/8
    step = 1e-6
    evaluations = []
    for sign in (1., -1.):
        delta = sign*step*direction.reshape(3, 6)
        made_x = x+delta[:, :3]
        made_q = np.array([rotation(delta[n, 3:]) @ q[n] for n in range(3)])
        solved = model.solve(made_x, made_q)
        increment = np.r_[delta.ravel(), np.zeros(18)]
        evaluations.append(model.evaluate(x, q, solved.local_rotations, solved.moments, increment=increment))
    plus, minus = evaluations
    assert mask(plus) == mask(minus) == mask(base)
    assert abs((plus.potential-minus.potential)/(2*step)-base.residual @ direction) <= 1e-7
    assert np.linalg.norm((plus.residual[:18]-minus.residual[:18])/(2*step)-base.tangent @ direction) <= 1e-7
    assert np.linalg.norm(base.tangent-base.tangent.T) <= 1e-11


def test_finite_superposed_motion_and_reference_coordinate_covariance():
    ref = reference(.4)
    x, q, u = perturbed(ref)
    moments = np.cos(np.arange(12)).reshape(2, 2, 3)/7
    g, shift = rotation([1.8, -1.7, 2.1]), np.array([2., -3., 4.])
    original = NonlinearMixedBeamProbe(ref, law(), order=8).evaluate(x, q, u, moments)
    moved_x, moved_q = x @ g.T+shift, np.einsum('ij,njk->nik', g, q)
    transformation = np.eye(36)
    transformation[:24, :24] = np.kron(np.eye(8), g)
    for moved_ref, moved_u in ((ref, np.einsum('ij,njk->nik', g, u)),
                               (ref.rigidly_transformed(g, shift), np.einsum('ij,njk,kl->nil', g, u, g.T))):
        other = NonlinearMixedBeamProbe(moved_ref, law(), order=8).evaluate(moved_x, moved_q, moved_u, moments)
        assert abs(other.potential-original.potential) <= 1e-11
        assert np.linalg.norm(other.residual-transformation @ original.residual) <= 1e-11
        assert np.linalg.norm(other.hessian-transformation @ original.hessian @ transformation.T) <= 1e-11*np.linalg.norm(original.hessian)


def test_reversal_transports_all_station_histories_and_material_moments():
    ref = reference(.4)
    x, q, u = perturbed(ref)
    moments = np.cos(np.arange(12)).reshape(2, 2, 3)/7
    origins = tuple(SectionHistory(.0001*i, .0002*i) for i in range(16))
    original = NonlinearMixedBeamProbe(ref, law(), order=8, origins=origins).evaluate(x, q, u, moments)
    other_law = DirectedHardeningSectionProbe(T6 @ section() @ T6.T, T6 @ A, .02, .4)
    flip = np.diag([-1., 1., -1.])
    other = NonlinearMixedBeamProbe(ref.reversed(), other_law, order=8, origins=origins[::-1]).evaluate(
        x[::-1], q[::-1] @ flip, u[::-1], moments[::-1, ::-1] @ T3.T)
    assert abs(other.potential-original.potential) <= 1e-11
    for made, expected in zip(other.stations, original.stations[::-1]):
        assert made.response.origin == expected.response.origin
        assert abs(made.response.history.plastic_coordinate-expected.response.history.plastic_coordinate) <= 1e-11
        assert np.linalg.norm(made.response.resultants-T6 @ expected.response.resultants) <= 1e-11


def test_budgets_invalid_inputs_and_failed_trial_preserve_origins():
    ref = reference(.4)
    x, q, u = perturbed(ref)
    model = NonlinearMixedBeamProbe(ref, law(), order=8)
    before = model.origins
    for kwargs in ({'max_iterations': 0}, {'max_evaluations': 0}, {'max_evaluations': 1}):
        with pytest.raises(NonlinearLocalError, match='budget'):
            model.solve(x, q, **kwargs)
        assert model.origins == before
    for kwargs in ({'max_iterations': 26}, {'max_evaluations': 65}, {'max_iterations': True}):
        with pytest.raises(ValueError, match='bounds'):
            model.solve(x, q, **kwargs)
    with pytest.raises(ValueError, match='origin'):
        NonlinearMixedBeamProbe(ref, law(), order=8, origins=[SectionHistory()])
    with pytest.raises(ValueError, match='quadrature'):
        NonlinearMixedBeamProbe(ref, law(), order=64)
    bad = np.zeros(36)
    bad[18] = 2*np.pi
    with pytest.raises(ValueError, match='cutback'):
        model.evaluate(x, q, u, np.zeros((2, 2, 3)), increment=bad)
    with pytest.raises(ValueError):
        model.evaluate(x, q, u, np.full((2, 2, 3), np.nan))
    assert model.origins == before
