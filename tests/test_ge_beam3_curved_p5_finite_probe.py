"""Bounded unit diagnostics for the P5 finite functional; no formal claims."""

import numpy as np
import pytest

from docs.reference_cases import ge_beam3_curved_p5_algebra_probe as algebra
from docs.reference_cases.ge_beam3_curved_p5_finite_probe import (
    CurvedFiniteProbe, LocalStationarityError,
)
from test_ge_beam3_curved_p5_algebra_probe import (
    reference, section, perturbed, assert_scaled_close,
)


def test_reference_response_matches_analytic_algebra_probe():
    ref = reference(.6, .2, .15)
    model = CurvedFiniteProbe(ref, section())
    result = model.evaluate(ref.coordinates, ref.nodal_triads)
    _, reduced, condensed = algebra.reference_hessians(ref, section())
    assert result.potential < 1e-25
    assert_scaled_close(result.residual, np.zeros(18))
    assert_scaled_close(result.full_hessian, reduced)
    assert_scaled_close(result.tangent, condensed)
    assert result.local_iterations == 0


def test_finite_stationarity_symmetry_and_physical_recovery():
    ref = reference(.6, .2, .15)
    model = CurvedFiniteProbe(ref, section())
    x, q, _ = perturbed(ref)
    result = model.evaluate(x, q)
    assert 0 < result.local_iterations < 25
    assert result.local_residual_norm < 1e-11
    assert_scaled_close(result.tangent, result.tangent.T)
    total, moments = algebra.reduced_energy(ref, section(), x, q, result.local_rotations)
    assert_scaled_close(result.potential, total)
    assert_scaled_close(result.moments, moments)
    recovered = model.recover(x, q, result)
    for row in recovered:
        assert_scaled_close(row["resultant"], section() @ row["strain"])
        assert_scaled_close(row["spatial_moment"], row["current_frame"] @ row["resultant"][3:])
    # Moment compliance recovery carries bending; the interior rotation
    # gradient relative to the initial curve is identically zero.
    assert max(np.linalg.norm(row["strain"][3:]) for row in recovered) > 1e-3


@pytest.mark.parametrize("nonzero_chart", [False, True])
def test_condensed_residual_and_tangent_are_derivatives_in_same_chart(nonzero_chart):
    ref = reference(.4, .1, .15)
    model = CurvedFiniteProbe(ref, section())
    x, q, _ = perturbed(ref)
    chart = .2*np.sin(np.arange(18)+.2) if nonzero_chart else np.zeros(18)
    result = model.evaluate(x, q, increment=chart)
    direction = np.cos(np.arange(18)+.7)/5
    step = 2e-5
    plus = model.evaluate(x, q, increment=chart+step*direction, initial=result.local_rotations)
    minus = model.evaluate(x, q, increment=chart-step*direction, initial=result.local_rotations)
    assert_scaled_close((plus.potential-minus.potential)/(2*step), result.residual @ direction, 1e-7)
    assert_scaled_close((plus.residual-minus.residual)/(2*step), result.tangent @ direction, 1e-7)


def test_finite_superposed_motion_transports_residual_tangent_and_recovery():
    ref = reference(.6, .2, .15)
    model = CurvedFiniteProbe(ref, section())
    x, q, _ = perturbed(ref)
    before = model.evaluate(x, q)
    g = algebra.rotation((2.1, -1.8, 2.5))
    x2, q2 = x @ g.T+(2., -4., 7.), np.einsum("ij,njk->nik", g, q)
    after = model.evaluate(x2, q2)
    change = np.kron(np.eye(6), g)
    assert_scaled_close(after.potential, before.potential)
    assert_scaled_close(after.residual, change @ before.residual)
    assert_scaled_close(after.tangent, change @ before.tangent @ change.T)
    assert_scaled_close(after.local_rotations, np.einsum("ij,njk->nik", g, before.local_rotations))
    first, second = model.recover(x, q, before), model.recover(x2, q2, after)
    for original, moved in zip(first, second):
        assert_scaled_close(moved["strain"], original["strain"])
        assert_scaled_close(moved["resultant"], original["resultant"])
        assert_scaled_close(moved["spatial_force"], g @ original["spatial_force"])


def test_large_noncommuting_bend_twist_solves_and_has_consistent_tangent():
    ref = reference(.6, .2, .15)
    model = CurvedFiniteProbe(ref, section())
    x = ref.coordinates+np.array(((.1, -.15, .2), (-.15, .1, -.2), (.2, .2, .1)))
    vectors = ((-.5, .7, .6), (.8, -.2, .3), (.4, .6, -.7))
    q = np.array([algebra.rotation(vector) @ ref.nodal_triads[n]
                  for n, vector in enumerate(vectors)])
    result = model.evaluate(x, q)
    assert result.local_iterations < 25
    assert result.local_residual_norm < 1e-11
    direction, step = np.sin(np.arange(18)+.8)/4, 2e-5
    plus = model.evaluate(x, q, increment=step*direction, initial=result.local_rotations)
    minus = model.evaluate(x, q, increment=-step*direction, initial=result.local_rotations)
    assert_scaled_close((plus.residual-minus.residual)/(2*step), result.tangent @ direction, 1e-7)


def test_loaded_response_is_covariant_under_reference_coordinate_change():
    ref = reference(.6, .2, .15)
    force = np.array((.1, -.2, .3))
    x, q, _ = perturbed(ref)
    before = CurvedFiniteProbe(ref, section(), line_force=force).evaluate(x, q)
    g = algebra.rotation((.5, -.8, .3))
    moved = ref.rigidly_transformed(g, (2., -3., 4.))
    x2, q2 = x @ g.T+(2., -3., 4.), np.einsum("ij,njk->nik", g, q)
    after = CurvedFiniteProbe(moved, section(), line_force=g @ force).evaluate(x2, q2)
    change = np.kron(np.eye(6), g)
    assert_scaled_close(after.potential, before.potential)
    assert_scaled_close(after.residual, change @ before.residual)
    assert_scaled_close(after.tangent, change @ before.tangent @ change.T)


def test_finite_reversal_transports_condensed_tangent_and_moment_work():
    ref = reference(.6, .2, .15)
    x, q, _ = perturbed(ref)
    original = CurvedFiniteProbe(ref, section()).evaluate(x, q)
    reversed_model = CurvedFiniteProbe(ref.reversed(), algebra.T6 @ section() @ algebra.T6.T)
    reversed_result = reversed_model.evaluate(x[::-1], q[::-1] @ algebra.S)
    permutation = np.r_[12:18, 6:12, 0:6]
    assert_scaled_close(reversed_result.potential, original.potential)
    assert_scaled_close(reversed_result.residual, original.residual[permutation])
    assert_scaled_close(reversed_result.tangent, original.tangent[np.ix_(permutation, permutation)])
    assert_scaled_close(reversed_result.moments,
                        np.einsum("ij,nej->nei", algebra.T3, original.moments[::-1, ::-1]))


def test_finite_straight_limit_matches_p3_local_solve_and_response():
    from anyfesolver.ge_beam3_mixed_element import GeometricallyExactBeam3D3NElement as P3
    ref = reference(0.)
    x, q, _ = perturbed(ref)
    made = CurvedFiniteProbe(ref, section()).evaluate(x, q)
    old_local = P3._solve_local(x, q, 1., section())
    old = P3._potential(x, q, 1., section(), old_local.rotations, old_local.moments,
                       include_external=True)
    old_tangent = old.hessian[:18, :18]-old.hessian[:18, 18:] @ np.linalg.solve(
        old.hessian[18:, 18:], old.hessian[18:, :18])
    assert_scaled_close(made.potential, old.value)
    assert_scaled_close(made.residual, old.gradient[:18])
    assert_scaled_close(made.tangent, old_tangent)
    assert_scaled_close(made.moments, old_local.moments)


def test_spatial_dead_force_load_includes_lift_rotation_work_and_total_balance():
    ref = reference(.6, .2, .15)
    force = np.array((.1, -.2, .3))
    model = CurvedFiniteProbe(ref, section(), line_force=force)
    x, q, _ = perturbed(ref)
    result = model.evaluate(x, q)
    stations, weights = np.polynomial.legendre.leggauss(32)
    work, force_total, torque = 0., np.zeros(3), np.zeros(3)
    for cell in (0, 1):
        for station, weight in zip(stations, weights):
            t = (station+1)/2
            xi = cell-1+t
            point = algebra.lift_position(ref, x, result.local_rotations[cell], cell, t)
            measure = weight*ref.jacobian(xi)/2
            work += measure*force @ (point-ref.position(xi))
            force_total += measure*force
            torque += measure*np.cross(point, force)
    internal_energy, _ = algebra.reduced_energy(ref, section(), x, q, result.local_rotations)
    assert_scaled_close(result.potential, internal_energy-work)
    nodal_residual = result.residual.reshape(3, 6)
    assert_scaled_close(nodal_residual[:, :3].sum(axis=0), -force_total)
    nodal_torque = (np.cross(x, nodal_residual[:, :3])+nodal_residual[:, 3:]).sum(axis=0)
    assert_scaled_close(nodal_torque, -torque)
    # Omitting the rotation load would change local stationarity.
    unloaded = CurvedFiniteProbe(ref, section())
    missed = unloaded._jet(x, q, result.local_rotations, external=False).gradient
    assert np.linalg.norm(missed) > 1e-3


def test_loaded_condensed_tangent_matches_directional_residual():
    ref = reference(.4, .1, .15)
    model = CurvedFiniteProbe(ref, section(), line_force=(.1, -.2, .3))
    x, q, _ = perturbed(ref)
    result = model.evaluate(x, q)
    direction, step = np.sin(np.arange(18)+.5)/4, 2e-5
    plus = model.evaluate(x, q, increment=step*direction, initial=result.local_rotations)
    minus = model.evaluate(x, q, increment=-step*direction, initial=result.local_rotations)
    assert_scaled_close((plus.residual-minus.residual)/(2*step), result.tangent @ direction, 1e-7)


def test_failure_and_repeated_evaluation_do_not_mutate_caller_inputs_or_probe():
    ref = reference(.6, .2, .15)
    c = section()
    model = CurvedFiniteProbe(ref, c)
    x, q, _ = perturbed(ref)
    saved = [a.copy() for a in (x, q, c)]
    with pytest.raises(LocalStationarityError, match="budget exhausted"):
        model.evaluate(x, q, max_iterations=0)
    before = model.evaluate(x, q)
    before.moments[:] = -999
    after = model.evaluate(x, q)
    assert after.moments.min() > -999
    assert before.potential == after.potential
    np.testing.assert_array_equal(before.residual, after.residual)
    for value, original in zip((x, q, c), saved):
        np.testing.assert_array_equal(value, original)
    c[:] = 0.
    np.testing.assert_array_equal(model.section, saved[2])


def test_recovery_rejects_mismatched_configuration_or_corrupted_moments():
    ref = reference(.4, .1, .15)
    model = CurvedFiniteProbe(ref, section())
    x, q, _ = perturbed(ref)
    result = model.evaluate(x, q)
    with pytest.raises(LocalStationarityError, match="configuration"):
        model.recover(x+np.eye(3)*.01, q, result)
    result.moments[0, 0, 0] += .1
    with pytest.raises(LocalStationarityError, match="moments"):
        model.recover(x, q, result)


@pytest.mark.parametrize("mutation", ["position", "frame", "reflection", "increment", "cutoff"])
def test_malformed_configuration_and_relative_branch_fail_closed(mutation):
    ref = reference()
    model = CurvedFiniteProbe(ref, section())
    x, q = ref.coordinates, ref.nodal_triads
    increment = np.zeros(18)
    if mutation == "position":
        x[0, 0] = np.nan
    elif mutation == "frame":
        q[0, 0, 0] += .2
    elif mutation == "reflection":
        q[0, :, 1] *= -1
    elif mutation == "increment":
        increment[0] = np.inf
    else:
        q[2] = algebra.rotation((.91*np.pi, 0., 0.)) @ q[2]
    with pytest.raises(ValueError):
        model.evaluate(x, q, increment=increment)
