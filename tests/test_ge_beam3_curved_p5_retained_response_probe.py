"""Retained-coordinate elastic response tests; no production qualification."""

from dataclasses import replace

import numpy as np
import pytest

from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation, T6
from docs.reference_cases.ge_beam3_curved_p5_finite_probe import CurvedFiniteProbe, LocalStationarityError
from docs.reference_cases.ge_beam3_curved_p5_retained_response_probe import RetainedFiniteResponseProbe
from test_ge_beam3_curved_p5_algebra_probe import reference, section, perturbed, assert_scaled_close
from test_ge_beam3_curved_p5_spin_split_probe import coupled_state


def test_retained_straight_limit_matches_accepted_p3_potential_and_derivatives():
    from anyfesolver.ge_beam3_mixed_element import GeometricallyExactBeam3D3NElement as P3
    ref = reference(0.)
    x, q, _ = perturbed(ref)
    made = RetainedFiniteResponseProbe(ref, section(), x, q).evaluate()
    local = P3._solve_local(x, q, 1., section())
    expected = P3._potential(x, q, 1., section(), local.rotations, local.moments,
                             include_external=True)
    tangent = expected.hessian[:18, :18]-expected.hessian[:18, 18:] @ np.linalg.solve(
        expected.hessian[18:, 18:], expected.hessian[18:, :18])
    assert_scaled_close(made.potential, expected.value)
    assert_scaled_close(made.residual, expected.gradient[:18])
    assert_scaled_close(made.tangent, tangent)


@pytest.mark.parametrize("force", [None, [.1, -.2, .3]])
def test_potential_residual_tangent_and_recovery_match_original_moderate_response(force):
    ref = reference(.6, .2, .15)
    x, q, _ = perturbed(ref)
    original = CurvedFiniteProbe(ref, section(), line_force=force)
    expected = original.evaluate(x, q)
    model = RetainedFiniteResponseProbe(ref, section(), x, q, line_force=force)
    actual = model.evaluate()
    assert actual.physical_residual_norm <= 1e-11
    assert_scaled_close(actual.potential, expected.potential)
    assert_scaled_close(actual.residual, expected.residual)
    assert_scaled_close(actual.tangent, expected.tangent)
    for left, right in zip(model.recover(actual), original.recover(x, q, expected)):
        for field in ("strain", "resultant", "current_frame", "spatial_force", "spatial_moment"):
            assert_scaled_close(left[field], right[field])


@pytest.mark.parametrize("nonzero_chart", [False, True])
def test_directional_energy_residual_and_tangent_in_one_fixed_external_chart(nonzero_chart):
    ref = reference(.4, .1, .15)
    x, q, _ = perturbed(ref)
    model = RetainedFiniteResponseProbe(ref, section(), x, q, line_force=[.1, -.2, .3])
    chart = .2*np.sin(np.arange(18)+.2) if nonzero_chart else np.zeros(18)
    direction = np.cos(np.arange(18)+.7)/5
    step = 2e-5
    result = model.evaluate(increment=chart)
    plus = model.evaluate(increment=chart+step*direction)
    minus = model.evaluate(increment=chart-step*direction)
    assert_scaled_close((plus.potential-minus.potential)/(2*step), result.residual @ direction, 1e-7)
    assert_scaled_close((plus.residual-minus.residual)/(2*step), result.tangent @ direction, 1e-7)
    assert_scaled_close(result.tangent, result.tangent.T)
    original = CurvedFiniteProbe(ref, section(), line_force=[.1, -.2, .3]).evaluate(x, q, increment=chart)
    assert_scaled_close(result.residual, original.residual)
    assert_scaled_close(result.tangent, original.tangent)
    assert len(model.recover(result)) == 4


@pytest.mark.parametrize("rho", [10000., 1000000.])
def test_uncoupled_high_contrast_response_and_normalized_work_balance(rho):
    ref = reference(.6, .2, .15)
    x, q, _ = perturbed(ref)
    elastic = np.diag([rho*rho]*3+[1., 2., 3.])
    model = RetainedFiniteResponseProbe(ref, elastic, x, q)
    result = model.evaluate()
    assert result.physical_residual_norm <= 1e-11
    assert np.isfinite(result.residual).all() and np.isfinite(result.tangent).all()
    force = result.residual.reshape(3, 6)
    force_scale = max(1., np.linalg.norm(force[:, :3]))
    moment_terms = force[:, 3:]+np.cross(x, force[:, :3])
    moment_scale = max(1., np.linalg.norm(force[:, 3:]), np.linalg.norm(np.cross(x, force[:, :3])))
    assert np.linalg.norm(force[:, :3].sum(axis=0)) <= 1e-11*force_scale
    assert np.linalg.norm(moment_terms.sum(axis=0)) <= 1e-11*moment_scale
    for row in model.recover(result):
        assert_scaled_close(row["resultant"], elastic @ row["strain"])
    direction = np.sin(np.arange(18)+.8)/5
    step = 2e-5
    plus, minus = model.evaluate(increment=step*direction), model.evaluate(increment=-step*direction)
    assert_scaled_close((plus.residual-minus.residual)/(2*step), result.tangent @ direction, 1e-7)


@pytest.mark.parametrize("rho", [10000., 1000000.])
def test_additional_small_amplitude_coupled_states_have_bound_response_and_recovery(rho):
    ref = reference(.6, .2, .15)
    elastic, x, q = coupled_state(ref, rho, 1/rho)
    model = RetainedFiniteResponseProbe(ref, elastic, x, q)
    result = model.evaluate()
    assert result.physical_residual_norm <= 1e-11
    assert np.isfinite(result.residual).all() and np.isfinite(result.tangent).all()
    assert len(model.recover(result)) == 4


def test_finite_superposition_transports_response_and_recovery():
    ref = reference(.6, .2, .15)
    x, q, _ = perturbed(ref)
    first = RetainedFiniteResponseProbe(ref, section(), x, q)
    before = first.evaluate()
    g = rotation([2.1, -1.8, 2.5])
    second = RetainedFiniteResponseProbe(ref, section(), x @ g.T+[2., -4., 7.],
                                        np.einsum("ij,njk->nik", g, q))
    after = second.evaluate()
    transform = np.kron(np.eye(6), g)
    assert_scaled_close(after.potential, before.potential)
    assert_scaled_close(after.residual, transform @ before.residual)
    assert_scaled_close(after.tangent, transform @ before.tangent @ transform.T)
    for a, b in zip(first.recover(before), second.recover(after)):
        assert_scaled_close(a["strain"], b["strain"])
        assert_scaled_close(b["spatial_force"], g @ a["spatial_force"])


def test_reference_covariance_and_reversal_congruence():
    ref = reference(.6, .2, .15)
    x, q, _ = perturbed(ref)
    before = RetainedFiniteResponseProbe(ref, section(), x, q).evaluate()
    g = rotation([.6, -.8, 1.1])
    moved = RetainedFiniteResponseProbe(ref.rigidly_transformed(g, np.array([3., -2., 1.])),
                                       section(), x @ g.T+[3., -2., 1.],
                                       np.einsum("ij,njk->nik", g, q)).evaluate()
    transform = np.kron(np.eye(6), g)
    assert_scaled_close(moved.residual, transform @ before.residual)
    assert_scaled_close(moved.tangent, transform @ before.tangent @ transform.T)
    flip = np.diag([-1., 1., -1.])
    reversed_result = RetainedFiniteResponseProbe(ref.reversed(), T6 @ section() @ T6.T,
                                                 x[::-1], q[::-1] @ flip).evaluate()
    permutation = np.arange(18).reshape(3, 6)[::-1].ravel()
    assert_scaled_close(reversed_result.residual, before.residual[permutation])
    assert_scaled_close(reversed_result.tangent, before.tangent[np.ix_(permutation, permutation)])


def test_fingerprint_mutation_configuration_rejection_and_repeatability():
    ref = reference(.6, .2, .15)
    x, q, _ = perturbed(ref)
    model = RetainedFiniteResponseProbe(ref, section(), x, q)
    response, second = model.evaluate(), model.evaluate()
    assert response.state_sha256 == second.state_sha256
    assert np.array_equal(response.residual, second.residual)
    assert np.array_equal(response.tangent, second.tangent)
    for field in ("parameters", "increment"):
        changed = getattr(response, field).copy()
        changed[0] += .001
        with pytest.raises(ValueError, match="fingerprint"):
            model.recover(replace(response, **{field: changed}))
    changed = response.rotations.copy()
    changed[0, 0, 0] += .001
    with pytest.raises(ValueError, match="rotations mismatch"):
        model.recover(replace(response, rotations=changed))
    other = RetainedFiniteResponseProbe(ref, section()*2, x, q)
    with pytest.raises(ValueError, match="fingerprint"):
        other.recover(response)
    with pytest.raises(ValueError, match="station"):
        model.recover(response, stations=[np.nan])


def test_extreme_coupled_failure_creates_no_response():
    ref = reference(.6, .2, .15)
    elastic, x, q = coupled_state(ref, 1000000., 1.)
    model = RetainedFiniteResponseProbe(ref, elastic, x, q)
    with pytest.raises(LocalStationarityError):
        model.evaluate()
