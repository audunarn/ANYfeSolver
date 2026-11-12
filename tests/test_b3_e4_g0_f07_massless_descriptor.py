"""G0 F07 analytical check for a constrained, massless descriptor.

These are small formulation and existing-kernel checks. They do not qualify
B3/E4 mass operators, the mixed owner, free-free modes, or coupled buckling.
"""

import numpy as np
from scipy import linalg

from anyfesolver._ge_beam3_native_buckling import factor_buckling
from anyfesolver._native_paired_factor_chain_modes import solve_paired_factor_chain_modes


def test_massless_retained_coordinate_and_multiplier_are_algebraic():
    # U = 12 us^2/2 + 20 (ub-r)^2/2 + 30 r^2/2.
    # The last row enforces us=ub; r and its multiplier have no inertia.
    stiffness = np.array(
        [[12., 0., 0., 1.],
         [0., 20., -20., -1.],
         [0., -20., 50., 0.],
         [1., -1., 0., 0.]],
    )
    mass = np.diag([2., 3., 0., 0.])
    homogeneous, vectors = linalg.eig(
        stiffness, mass, homogeneous_eigvals=True, check_finite=True,
    )
    alpha, beta = homogeneous
    finite = np.flatnonzero(np.abs(beta) > 1e-12 * np.max(np.abs(homogeneous)))
    assert finite.size == 1
    assert np.count_nonzero(beta == 0.) == 3
    assert np.array_equal(np.diag(mass), [2., 3., 0., 0.])

    value = alpha[finite[0]] / beta[finite[0]]
    # Static elimination gives r=(20/50) ub and us=ub, hence
    # K_eff=12+20*30/50=24 and M_eff=2+3=5.
    assert abs(value.imag) <= 1e-13
    np.testing.assert_allclose(value.real, 24. / 5., rtol=1e-12, atol=0.)
    vector = vectors[:, finite[0]]
    equation = stiffness @ vector - value * (mass @ vector)
    normalizer = (np.linalg.norm(stiffness, 2) +
                  abs(value) * np.linalg.norm(mass, 2)) * np.linalg.norm(vector)
    assert np.linalg.norm(equation) / normalizer <= 1e-12
    assert abs(vector[0] - vector[1]) / np.linalg.norm(vector[:3]) <= 1e-12
    assert abs(vector[2] - .4 * vector[1]) / np.linalg.norm(vector[:3]) <= 1e-12


def test_massless_coordinate_is_condensed_for_modes_but_retained_for_buckling():
    """F07 bounded kernel case: exact joint lift, zero retained inertia.

    With us=ub=u, K=[[32,-20],[-20,50]], M=diag(5,0).
    The finite vibration eigenvalue is 24/5. Under geometric stiffness
    G=diag(0,-25), the full buckling multiplier is 3/2; incorrectly
    reusing the vibration static lift r=2u/5 gives multiplier 6.
    This tests the existing spectral kernels, not a coupled B3/E4 owner.
    """
    material = np.array([[32., -20.], [-20., 50.]])
    factor = np.linalg.cholesky(material).T
    geometric = np.diag([0., -25.])
    kinetic = np.array([[np.sqrt(5.), 0.]])

    modes = solve_paired_factor_chain_modes(
        np.eye(2), factor, np.zeros((2, 2)), kinetic,
        (0, 1), (1,), bounds=(0., 10.), num_modes=1,
    )
    np.testing.assert_allclose(modes.eigenvalues, [24./5.], rtol=1e-10)
    assert abs(modes.high_modes[1, 0]-.4*modes.high_modes[0, 0]) <= 1e-10
    assert modes.full_backward_residual <= 1e-11

    buckling = factor_buckling(
        np.eye(2), factor, geometric, (0, 1),
        bounds=(0., 10.), num_modes=1,
    )
    np.testing.assert_allclose(buckling.multipliers, [1.5], rtol=1e-11)
    np.testing.assert_allclose(
        buckling.multipliers,
        [v.real for v in linalg.eigvals(material, -geometric)
         if np.isfinite(v) and v.real > 0.], rtol=1e-11,
    )
    assert abs(buckling.modes[1, 0]-.4*buckling.modes[0, 0]) > .1
    wrong_condensed_multiplier = 24./(25.*(.4**2))
    assert abs(wrong_condensed_multiplier-buckling.multipliers[0]) > 1.
    assert max(buckling.original_factor_residuals) <= 1e-11
