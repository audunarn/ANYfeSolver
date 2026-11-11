"""Private signed mode extraction correctness; not production qualification."""

import numpy as np
import pytest
from scipy import linalg
from anyfesolver._native_signed_factor_modes import solve_signed_factor_modes
from anyfesolver.control import CancellationToken, SolveCancelled


def test_negative_high_contrast_mode_and_mass_normalized_eigenvector():
    f = np.diag([1., 1e12]); g = np.array([[-3., 1e12], [1e12, 0.]]); m = np.eye(2)
    result = solve_signed_factor_modes(f, g, m, (0, 1), (), bounds=(-5., 0.), num_modes=1)
    assert abs(result.eigenvalues[0]+3.) < 1e-10
    v = result.full_modes[:, 0]
    assert abs(v[1]/v[0]+1e-12) < 1e-23
    assert abs(v@m@v-1.) < 1e-11
    assert result.spectral_residual < 1e-11 and result.full_backward_residual < 1e-11
    assert result.negative_eigenvalues_retained and not result.production_qualified
    for a in (result.eigenvalues, result.full_modes, result.numerical_brackets):
        with pytest.raises(ValueError): a.setflags(write=True)


def test_repeated_zero_cluster_has_complete_orthogonal_modes():
    f = np.diag([1., 1., 1e12]); g = np.diag([-1., -1., 0.])
    result = solve_signed_factor_modes(f, g, np.eye(3), (0, 1, 2), (), bounds=(-1., 1.), num_modes=2)
    np.testing.assert_allclose(result.eigenvalues, 0., atol=1e-11)
    np.testing.assert_allclose(result.full_modes.T@result.full_modes, np.eye(2), atol=1e-11)
    with pytest.raises(ValueError, match='truncate'):
        solve_signed_factor_modes(f, g, np.eye(3), (0, 1, 2), (), bounds=(-1., 1.), num_modes=1)


def test_unstressed_exact_zero_and_positive_diagonal_modes_are_not_confused():
    result = solve_signed_factor_modes(np.diag([0., 0., 1., 2.]), np.zeros((4, 4)),
        np.eye(4), (0, 1, 2, 3), (), bounds=(-1., 5.), num_modes=4)
    np.testing.assert_allclose(result.eigenvalues, [0., 0., 1., 4.], atol=1e-11)
    np.testing.assert_allclose(result.full_modes.T@result.full_modes, np.eye(4), atol=1e-11)


def test_signed_trace_elimination_and_modes_match_moderate_dense_pencil():
    rng = np.random.default_rng(53); f = rng.normal(size=(12, 6))
    g = rng.normal(size=(6, 6)); g = .1*(g+g.T); mass = np.diag([1., 2., 3., 4., 0., 0.])
    k = f.T@f+g; condensed = k[:4, :4]-k[:4, 4:]@linalg.solve(k[4:, 4:], k[4:, :4])
    expected = linalg.eigvalsh(condensed, mass[:4, :4])
    result = solve_signed_factor_modes(f, g, mass, tuple(range(6)), (4, 5), bounds=(-10., 100.), num_modes=4)
    np.testing.assert_allclose(result.eigenvalues, expected, atol=1e-10, rtol=1e-11)
    np.testing.assert_allclose(k@result.full_modes-(mass@result.full_modes)*result.eigenvalues, 0., atol=1e-10)


def test_cancelled_input_is_rejected_before_numerical_work(monkeypatch):
    token = CancellationToken(); token.cancel('test cancellation')
    import anyfesolver._native_signed_factor_modes as kernel
    monkeypatch.setattr(kernel, '_reduce', lambda *a: pytest.fail('numerics after cancellation'))
    with pytest.raises(SolveCancelled):
        kernel.solve_signed_factor_modes(np.eye(2), np.zeros((2, 2)), np.eye(2), (0, 1), (),
            bounds=(-1., 2.), num_modes=2, cancellation_token=token)


@pytest.mark.parametrize('mutation', ['geometric', 'mass', 'bounds', 'dofs'])
def test_bad_inputs_fail_closed(mutation):
    f = np.eye(2); g = np.zeros((2, 2)); m = np.eye(2); bounds = (-1., 2.); dofs = (0, 1)
    if mutation == 'geometric': g[0, 0] = np.nan
    if mutation == 'mass': m[0, 0] = -1.
    if mutation == 'bounds': bounds = (1.5, 2.)
    if mutation == 'dofs': dofs = (0, 0)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        solve_signed_factor_modes(f, g, m, dofs, (), bounds=bounds, num_modes=2)


def test_cancellation_inside_bracketing_stops_before_mode_extraction(monkeypatch):
    import anyfesolver._native_signed_factor_modes as kernel
    token = CancellationToken(); real = kernel._inertia; calls = []
    def cancel(*a):
        calls.append(1)
        if len(calls) == 3: token.cancel('during shift')
        return real(*a)
    monkeypatch.setattr(kernel, '_inertia', cancel)
    with pytest.raises(SolveCancelled):
        kernel.solve_signed_factor_modes(np.diag([1., 2.]), np.zeros((2, 2)), np.eye(2),
            (0, 1), (), bounds=(-1., 5.), num_modes=2, cancellation_token=token)
    assert len(calls) == 3


def test_cooperative_time_limit_stops_before_reduction(monkeypatch):
    import anyfesolver._native_signed_factor_modes as kernel
    times = iter((0., 601.)); monkeypatch.setattr(kernel, 'monotonic', lambda: next(times))
    monkeypatch.setattr(kernel, '_reduce', lambda *a: pytest.fail('reduction after deadline'))
    with pytest.raises(ValueError, match='time limit'):
        kernel.solve_signed_factor_modes(np.eye(2), np.zeros((2, 2)), np.eye(2),
            (0, 1), (), bounds=(-1., 2.), num_modes=2)


def test_caller_mutation_does_not_change_owned_snapshot(monkeypatch):
    import anyfesolver._native_signed_factor_modes as kernel
    f = np.diag([1., 2.]); g = np.zeros((2, 2)); m = np.eye(2); real = kernel._reduce
    def mutate(*a):
        f[:] = 100.; g[:] = -100.; m[:] = 0.
        return real(*a)
    monkeypatch.setattr(kernel, '_reduce', mutate)
    result = kernel.solve_signed_factor_modes(f, g, m, (0, 1), (), bounds=(-1., 5.), num_modes=2)
    np.testing.assert_allclose(result.eigenvalues, [1., 4.], atol=1e-11)


@pytest.mark.parametrize('kwargs', [dict(root_width=True), dict(bounds=(False, 5.))])
def test_boolean_numerical_controls_are_not_accepted(kwargs):
    args = dict(bounds=(-1., 5.), num_modes=2); args.update(kwargs)
    with pytest.raises(ValueError, match='explicit root'):
        solve_signed_factor_modes(np.diag([1., 2.]), np.zeros((2, 2)), np.eye(2), (0, 1), (), **args)
