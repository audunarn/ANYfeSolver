"""Factor numerical kernel checks; no nonlinear/indefinite interpretation."""

import numpy as np
import pytest

from anyfesolver._native_reference_factor_spectrum import solve_reference_factor_spectrum as solve
from anyfesolver._native_stationary_spectrum import solve_stationary_spectrum
from anyfesolver.control import CancellationToken, SolveCancelled
from docs.reference_cases.ge_beam3_slender_spectrum_probe import factors


def test_moderate_reference_factors_match_dense_without_changing_signed_solver():
    e, g = factors(10.)
    free = tuple(range(6, 24)); algebraic = (9, 10, 11, 15, 16, 17)
    made = solve(e, g, free, algebraic)
    dense = solve_stationary_spectrum(e.T@e, g.T@g, free, algebraic)
    np.testing.assert_allclose(made.eigenvalues, dense.eigenvalues, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose((g@made.full_modes).T@(g@made.full_modes), np.eye(6), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(made.dynamic_map, dense.dynamic_map, rtol=1e-11, atol=1e-11)
    assert not made.production_qualified and not made.prestressed_tangent_authorized
    for value in (made.eigenvalues, made.full_modes, made.dynamic_map):
        with pytest.raises(ValueError): value.setflags(write=True)


def test_zero_reference_modes_not_given_stiffness_or_inertia():
    e = np.diag([0., 0., 1., 3.]); g = np.eye(4)
    result = solve(e, g, (0, 1, 2, 3), (), num_modes=4)
    np.testing.assert_array_equal(result.eigenvalues, [0., 0., 1., 9.])
    # The preserved signed solver must still retain a negative mode.
    signed = solve_stationary_spectrum(np.diag([-1., 0., 1., 9.]), g, (0, 1, 2, 3), (), num_modes=4)
    np.testing.assert_array_equal(signed.eigenvalues, [-1., 0., 1., 9.])


def test_factor_row_orthogonal_change_and_repeat_are_invariant():
    e, g = factors(30.); free = tuple(range(6, 24)); a = (9, 10, 11, 15, 16, 17)
    first = solve(e, g, free, a); repeated = solve(e, g, free, a)
    np.testing.assert_array_equal(first.eigenvalues, repeated.eigenvalues)
    np.testing.assert_array_equal(first.full_modes, repeated.full_modes)
    # Row reversal/signs leave both energies exactly the same mathematical form.
    moved = solve(-e[::-1], g[::-1], free, a)
    np.testing.assert_allclose(first.eigenvalues, moved.eigenvalues, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('kind', ['shape', 'nonfinite', 'slots', 'duplicate', 'algebraic_mass',
    'algebraic_rank', 'kinetic_rank', 'modes'])
def test_invalid_reference_factors_fail_closed(kind):
    e, g = np.eye(4), np.eye(4); free = (0, 1, 2, 3); a = (); modes = 2
    if kind == 'shape': e = np.ones((3, 5))
    elif kind == 'nonfinite': e[0, 0] = np.nan
    elif kind == 'slots': free = (0, 1, 2, True)
    elif kind == 'duplicate': free = (0, 1, 1, 3)
    elif kind == 'algebraic_mass': a = (0,)
    elif kind == 'algebraic_rank': a = (0,); e[:, 0] = 0.; g[:, 0] = 0.
    elif kind == 'kinetic_rank': g[:, 0] = 0.
    elif kind == 'modes': modes = True
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        solve(e, g, free, a, num_modes=modes)


def test_pre_cancel_does_not_evaluate_factors():
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled): solve(None, None, (), (), cancellation_token=token)
