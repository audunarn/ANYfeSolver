"""Exact sign arithmetic, not a claim of certified continuum spectra."""
from fractions import Fraction
import numpy as np
import pytest
from anyfesolver._native_exact_shift_inertia import exact_inertia
from anyfesolver._native_exact_shift_chain_modes import inertia, brackets


@pytest.mark.parametrize('diagonal', [
    (2., -3., 0., 5.),
    (1.e-200, -1.e200, 2., 0.),
    (0., 0., 0., 0.),
    (1., 2., 3., 4.),
])
def test_exact_congruence_inertia(diagonal):
    # Unit upper triangular integer map is exactly invertible; reconstruct
    # all entries as Fractions to avoid a rounded test congruence claim.
    d = list(map(Fraction, diagonal))
    t = [[Fraction(int(i == j or j == i+1)) for j in range(4)] for i in range(4)]
    h = [[sum((t[k][i]*d[k]*t[k][j] for k in range(4)), Fraction()) for j in range(4)] for i in range(4)]
    # For the contrast fixture use its diagonal directly, so binary64
    # rounding does not erase small terms during this fixture conversion.
    a = np.diag(diagonal) if max(abs(v) for v in diagonal) > 1.e100 else np.array(h, dtype=float)
    answer = exact_inertia(a, np.zeros((4, 4)), 0.)
    assert answer == (sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d))


def test_hyperbolic_pivot_and_exact_zero():
    a = np.array([[0., 2., 0.], [2., 0., 0.], [0., 0., 0.]])
    assert exact_inertia(a, np.zeros_like(a), 0.) == (1, 1, 1)


def test_near_root_fallback_and_repeated_signed_roots():
    h = np.diag([-2., 1., 1., 22036.])
    mass = np.eye(4)
    shift = float(np.nextafter(22036., -np.inf))
    assert inertia(h, mass, shift) == 3
    assert inertia(h, mass, float(np.nextafter(22036., np.inf))) == 4
    found = brackets(h, mass, (-10., 30000.), 4, 1e-10, lambda: None)
    for interval, value in zip(found, [-2., 1., 1., 22036.]):
        assert interval[0] <= value <= interval[1]
        assert interval[1]-interval[0] <= 1e-10


def test_cancellation_and_nonsymmetric_inputs_fail():
    def stop(): raise RuntimeError('cancel exact sign')
    with pytest.raises(RuntimeError, match='cancel exact sign'):
        exact_inertia(np.eye(2), np.eye(2), 0., stop)
    with pytest.raises(ValueError):
        exact_inertia(np.array([[1., 2.], [0., 1.]]), np.eye(2), 0.)
