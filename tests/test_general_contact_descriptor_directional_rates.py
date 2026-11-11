"""Independent affine DAE derivative oracles and public S3 integration."""

from itertools import product
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    ContactNodalLoadHistory, ContactTransientSolverControls, LoadCase,
    TransientConfig, solve_transient_contact,
)
from anyfesolver.general_contact.descriptor_rates import directional_descriptor_rate
from anyfesolver.general_contact.core import active_contact_manager
import anyfesolver.general_contact.transient as driver
from test_general_contact_transient_plane import _guided_s3_descriptor_impact, _uniform_s3_normal_load


@pytest.mark.parametrize("physical_rate", [-2., 0., 2.])
def test_two_coordinate_touch_matches_exact_right_derivative(physical_rate):
    k, c = 2., 3.
    rate = directional_descriptor_rate(
        np.diag([0., k]), np.array([[0.], [1.]]), np.array([physical_rate, 0.]),
        np.array([[1., 1.]]), np.array([c]), [False], [True],
    )
    exact = np.array([physical_rate, -c*physical_rate/(k+c) if physical_rate < 0 else 0.])
    np.testing.assert_allclose(rate, exact, atol=1e-15)
    # Direct right difference of the independently defined algebraic force.
    h = 1e-5
    q = h*rate
    algebraic_force = k*q[1]+c*min(q[0]+q[1], 0.)
    assert abs(algebraic_force/h) < 1e-14
    # The same oracle applies to acceleration at stationary touch; algebraic
    # coordinates have no inertia and do not obey full-coordinate Newmark.
    q = .5*h*h*rate
    assert abs((k*q[1]+c*min(q[0]+q[1], 0.))/(.5*h*h)) < 1e-14


def _enumerated_oracle(K, B, weights, fixed, switching, physical, offset):
    """Enumerate scalar inequalities and solve two algebraic unknowns directly."""
    for choices in product((False, True), repeat=int(np.count_nonzero(switching))):
        active = fixed.copy()
        active[switching] = choices
        A = K+B[active].T @ np.diag(weights[active]) @ B[active]
        result = np.r_[physical, np.linalg.solve(A[1:, 1:], -A[1:, 0]*physical-offset[1:])]
        gap_rate = B @ result
        if np.all(gap_rate[switching & active] <= 1e-13) and np.all(gap_rate[switching & ~active] >= -1e-13):
            return result
    raise AssertionError("oracle has no consistent branch")


@pytest.mark.parametrize("physical", [-1.3, .7])
def test_multiple_constraints_basis_scaling_and_offset_match_enumerated_oracle(physical):
    K = np.diag([0., 2., 3.])
    B = np.array([[1., 1., 0.], [1., 0., -1.], [0., 1., 1.]])
    weights = np.array([5., 7., 11.])
    fixed = np.array([True, False, False])
    switching = np.array([False, True, True])
    offset = np.array([0., 2., -3.])
    expected = _enumerated_oracle(K, B, weights, fixed, switching, physical, offset)
    for Z in (np.eye(3)[:, 1:], np.array([[0., 0.], [2., 0.], [0., 3.]])):
        seed = np.array([physical, 10., -8.])
        original = seed.copy()
        actual = directional_descriptor_rate(K, Z, seed, B, weights, fixed, switching, offset=offset)
        np.testing.assert_allclose(actual, expected, atol=2e-14)
        np.testing.assert_array_equal(seed, original)


def test_directional_iteration_exhaustion_is_explicit_and_does_not_mutate_seed():
    seed = np.array([1., 0.])
    args = (np.diag([0., 2.]), np.array([[0.], [1.]]), seed,
            np.array([[1., 1.]]), np.array([3.]), [False], [True])
    with pytest.raises(ValueError, match="limit exhausted"):
        directional_descriptor_rate(*args, offset=np.array([0., 10.]), maximum_iterations=1)
    np.testing.assert_array_equal(seed, [1., 0.])
    result = directional_descriptor_rate(*args, offset=np.array([0., 10.]), maximum_iterations=2)
    np.testing.assert_allclose(result, [1., -2.6])


def test_directional_s3_tilted_impact_restart_preserves_physical_newmark():
    controls = ContactTransientSolverControls(allow_s3_descriptor_dev=True, s3_one_sided_rates_dev=True)
    def solve(end, checkpoint=None, emit=False, supplied_controls=controls):
        model, contact, velocity = _guided_s3_descriptor_impact(tilt=.002, span=.1)
        return model, solve_transient_contact(
            model, contact, TransientConfig(.0002, end, initial_velocity=velocity),
            solver_controls=supplied_controls, restart_checkpoint=checkpoint,
            emit_restart_checkpoint=emit,
        )
    model, direct = solve(.012)
    assert direct.profile.endswith("NEWMARK_DIRECTIONAL_RATES_DEV_V1")
    assert any(c.active_pressure_dofs for c in direct.contact)
    # Match the actual accepted partition. A separately rounded .011 end
    # trims dt by a few ulps and is not the identical discrete trajectory.
    _, prefix = solve(float(direct.times[55]), emit=True)
    _, resumed = solve(.012, prefix.restart_checkpoint)
    for field in ("displacements", "velocities", "accelerations", "external_nodal_work_j"):
        np.testing.assert_allclose(
            np.concatenate((getattr(prefix, field), getattr(resumed, field)[1:])),
            getattr(direct, field), rtol=0., atol=1e-11,
        )
    with pytest.raises(ValueError, match="checkpoint analysis is incompatible"):
        solve(.012, prefix.restart_checkpoint, supplied_controls=replace(controls, s3_one_sided_rates_dev=False))
    physical = [model.mesh.dof_manager.get_node_dofs(node)[2] for node in (1, 2, 3)]
    for i in range(1, len(direct.times)):
        dt = direct.times[i]-direct.times[i-1]
        q0, q1 = direct.displacements[i-1:i+1, physical]
        v0, v1 = direct.velocities[i-1:i+1, physical]
        a0, a1 = direct.accelerations[i-1:i+1, physical]
        np.testing.assert_allclose(q1-q0-dt*v0-dt*dt*(a0+a1)/4, 0., atol=1e-12)
        np.testing.assert_allclose(v1-v0-dt*(a0+a1)/2, 0., atol=1e-10)
    assert active_contact_manager() is None


def test_directional_controls_require_explicit_s3_opt_in():
    with pytest.raises(ValueError, match="descriptor opt-in"):
        ContactTransientSolverControls(s3_one_sided_rates_dev=True)


def test_directional_history_profile_keeps_physical_loading_and_massless_rejection():
    controls = ContactTransientSolverControls(allow_s3_descriptor_dev=True, s3_one_sided_rates_dev=True)
    model, contact, velocity = _guided_s3_descriptor_impact()
    velocity[:] = 0.
    history = ContactNodalLoadHistory(((0., 0.), (.002, 1.)))
    result = solve_transient_contact(
        model, contact, TransientConfig(.001, .002, initial_velocity=velocity),
        base_load_case=_uniform_s3_normal_load(), nodal_load_history=history,
        solver_controls=controls, emit_restart_checkpoint=True,
    )
    assert result.profile.endswith("NEWMARK_DIRECTIONAL_HISTORY_DEV_V1")
    assert result.restart_checkpoint["version"] == 2
    dof = model.mesh.dof_manager.get_node_dofs(1)[2]
    assert result.displacements[-1, dof] == pytest.approx(-6e-7, abs=2e-12)
    assert result.velocities[-1, dof] == pytest.approx(-.0008, abs=2e-12)
    load = LoadCase("tiny_algebraic_moment")
    load.add_nodal_load(1, moments=(0., 1e-12, 0.))
    with pytest.raises(ValueError, match="base load has unresolved algebraic force"):
        solve_transient_contact(
            model, contact, TransientConfig(.001, .002, initial_velocity=velocity),
            base_load_case=load, nodal_load_history=history, solver_controls=controls,
        )


@pytest.mark.parametrize("V,A", [(-2., 3.), (2., -3.), (0., -3.), (0., 3.)])
def test_integrated_touch_acceleration_matches_independent_force_expansion(monkeypatch, V, A):
    # Manufactured descriptor law: F_a=k*a+d*x**2+c*min(x+a,0).
    # This isolates caller branch/curvature wiring from any FE implementation.
    k, c, d = 2., 3., .7
    B = sparse.csr_matrix([[1., 1.]])
    Kcontact = c*(B.T @ B)
    manager = SimpleNamespace(
        support=np.ones(1), gap_gradient=B, stiffness=c,
        reference_weighted_gap=np.zeros(1), point=np.zeros(3),
        reference_point=np.zeros(3), normal=np.array([0., 0., 1.]),
        evaluate=lambda q, tangent: (np.zeros(2), Kcontact, None),
    )

    def manufactured_structure(model, q, state, *args, **kwargs):
        return (np.array([0., k*q[1]+d*q[0]**2]),
                sparse.csr_matrix([[0., 0.], [2*d*q[0], k]]), None)

    monkeypatch.setattr(driver, "_assemble_nonlinear_system", manufactured_structure)
    velocity, acceleration = driver._consistent_s3_descriptor_rates(
        None, sparse.eye(2, format="csr"), sparse.diags([1., 0.]),
        SimpleNamespace(reduced_basis=sparse.csr_matrix([[0.], [1.]])),
        np.zeros(2), np.array([V, 11.]), np.array([A, -13.]),
        sparse.diags([0., k])+Kcontact, {}, 1e-5,
        one_sided_manager=manager,
    )
    expected_v = -c*V/(k+c) if V < 0 else 0.
    expected_a = (-(c*A+2*d*V*V)/(k+c) if V < 0
                  else -2*d*V*V/k if V > 0
                  else -c*A/(k+c) if A < 0 else 0.)
    np.testing.assert_allclose(velocity, [V, expected_v], atol=2e-14)
    np.testing.assert_allclose(acceleration, [A, expected_a], atol=2e-14)
    # Independently substitute the right path into the unilateral force.
    h = 1e-5
    q = h*velocity+.5*h*h*acceleration
    force = k*q[1]+d*q[0]**2+c*min(q.sum(), 0.)
    remainder = d*(h**3*V*A+.25*h**4*A*A)
    assert abs((force-remainder)/(.5*h*h)) < 1e-8


def test_public_rate_failure_unwinds_without_physical_commit(monkeypatch):
    controls = ContactTransientSolverControls(allow_s3_descriptor_dev=True, s3_one_sided_rates_dev=True)
    model, contact, velocity = _guided_s3_descriptor_impact()
    commits = []
    original_commit = driver.commit_state_candidate

    def record_commit(*args, **kwargs):
        commits.append(True)
        return original_commit(*args, **kwargs)

    def exhausted(*args, **kwargs):
        raise ValueError("injected directional iteration limit exhausted")

    with monkeypatch.context() as patch:
        patch.setattr(driver, "commit_state_candidate", record_commit)
        patch.setattr(driver, "directional_descriptor_rate", exhausted)
        with pytest.raises(ValueError, match="injected directional iteration limit exhausted"):
            solve_transient_contact(
                model, contact, TransientConfig(.001, .002, initial_velocity=velocity),
                solver_controls=controls, emit_restart_checkpoint=True,
            )
    assert len(commits) == 1  # Initial equilibrium only; failed endpoint discarded.
    assert active_contact_manager() is None
    resumed = solve_transient_contact(
        model, contact, TransientConfig(.001, .002, initial_velocity=velocity), solver_controls=controls,
    )
    fresh_model, fresh_contact, fresh_velocity = _guided_s3_descriptor_impact()
    fresh = solve_transient_contact(
        fresh_model, fresh_contact, TransientConfig(.001, .002, initial_velocity=fresh_velocity),
        solver_controls=controls,
    )
    for field in ("displacements", "velocities", "accelerations", "external_nodal_work_j"):
        np.testing.assert_array_equal(getattr(resumed, field), getattr(fresh, field))
