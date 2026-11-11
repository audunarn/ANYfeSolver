from __future__ import annotations

from typing import Optional

import numpy as np
import pytest
from scipy import linalg, sparse

from anyfesolver.algebraic_dynamics import (
    AlgebraicDynamicsError,
    DESCRIPTOR_RAYLEIGH_REFINEMENT_POLICY_ID,
    _deterministic_block,
    solve_descriptor_spectrum,
)


def _congruent_descriptor(
    finite_eigenvalues: np.ndarray,
    *,
    algebraic_nullity: int,
    condition: float = 2.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a symmetric descriptor with a prescribed finite spectrum."""

    finite = np.asarray(finite_eigenvalues, dtype=float).reshape(-1)
    nullity = int(algebraic_nullity)
    size = int(finite.size + nullity)
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(size, size)))
    right, _ = np.linalg.qr(rng.normal(size=(size, size)))
    transform = (
        left
        @ np.diag(np.geomspace(1.0, float(condition), size))
        @ right.T
    )
    inverse = np.linalg.inv(transform)
    mass_diagonal = np.concatenate((np.ones(finite.size), np.zeros(nullity)))
    algebraic_stiffness = np.arange(
        float(np.max(finite)) + 10.0,
        float(np.max(finite)) + 10.0 + nullity,
    )
    stiffness_diagonal = np.concatenate((finite, algebraic_stiffness))
    mass = inverse.T @ np.diag(mass_diagonal) @ inverse
    stiffness = inverse.T @ np.diag(stiffness_diagonal) @ inverse
    return 0.5 * (stiffness + stiffness.T), 0.5 * (mass + mass.T)


def _normwise_backward_errors(
    stiffness: np.ndarray,
    mass: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    stiffness_norm = float(np.linalg.norm(stiffness))
    mass_norm = float(np.linalg.norm(mass))
    errors = []
    for eigenvalue, vector in zip(eigenvalues, eigenvectors.T):
        made = np.asarray(vector, dtype=float).reshape(-1)
        residual = stiffness @ made - float(eigenvalue) * (mass @ made)
        denominator = (
            stiffness_norm + abs(float(eigenvalue)) * mass_norm
        ) * float(np.linalg.norm(made))
        errors.append(float(np.linalg.norm(residual) / denominator))
    return np.asarray(errors, dtype=float)


def _sparse_spectrum(
    stiffness: np.ndarray,
    mass: np.ndarray,
    *,
    num_modes: int,
    algebraic_nullity: int,
    algebraic_basis: Optional[sparse.spmatrix] = None,
    target_shift: Optional[float] = None,
):
    return solve_descriptor_spectrum(
        stiffness,
        mass,
        num_modes=num_modes,
        dense_size_limit=1,
        algebraic_nullity=algebraic_nullity,
        algebraic_basis=algebraic_basis,
        target_shift=target_shift,
        static_condensation_limit=512,
    )


def _repeated_descriptor() -> tuple[np.ndarray, np.ndarray]:
    finite = np.concatenate(
        (
            np.full(8, 1.0),
            np.full(8, 2.0),
            np.linspace(3.0, 40.0, 32),
        )
    )
    return _congruent_descriptor(finite, algebraic_nullity=2)


def test_sparse_descriptor_preserves_repeated_eigenvalue_multiplicity() -> None:
    stiffness, mass = _repeated_descriptor()

    spectrum = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=8,
        algebraic_nullity=2,
    )
    order = np.argsort(spectrum.eigenvalues, kind="stable")[:8]

    np.testing.assert_allclose(
        spectrum.eigenvalues[order],
        np.ones(8),
        rtol=5.0e-11,
        atol=5.0e-12,
    )


def test_sparse_repeated_eigenspace_is_mass_orthonormal() -> None:
    stiffness, mass = _repeated_descriptor()

    spectrum = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=8,
        algebraic_nullity=2,
    )
    order = np.argsort(spectrum.eigenvalues, kind="stable")[:8]
    modes = spectrum.eigenvectors[:, order]
    gram = modes.T @ mass @ modes

    np.testing.assert_allclose(gram, np.eye(8), rtol=0.0, atol=2.0e-10)


def test_sparse_target_candidates_match_oracle_and_backward_residual() -> None:
    finite = np.linspace(1.0, 100.0, 49)
    stiffness, mass = _congruent_descriptor(
        finite,
        algebraic_nullity=1,
        condition=100.0,
        seed=913,
    )
    target = float(finite[24] + 1.0e-10)

    spectrum = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=3,
        algebraic_nullity=1,
        target_shift=target,
    )
    expected_indices = np.lexsort((finite, np.abs(finite - target)))[:3]
    expected = finite[expected_indices]
    errors = _normwise_backward_errors(
        stiffness,
        mass,
        spectrum.eigenvalues,
        spectrum.eigenvectors,
    )

    np.testing.assert_allclose(
        spectrum.eigenvalues,
        expected,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    assert float(np.max(errors)) <= 2.0e-9
    for eigenvalue, vector in zip(spectrum.eigenvalues, spectrum.eigenvectors.T):
        modal_mass = float(vector @ (mass @ vector))
        rayleigh = float(vector @ (stiffness @ vector)) / modal_mass
        assert float(eigenvalue) == pytest.approx(rayleigh, rel=2.0e-12, abs=2.0e-12)


def test_sparse_exact_target_has_a_repeatable_safe_outcome() -> None:
    size = 30
    mass = np.diag(np.concatenate((np.ones(size - 1), (0.0,))))
    stiffness = np.diag(np.arange(1.0, size + 1.0))

    outcomes = []
    for _ in range(2):
        try:
            spectrum = _sparse_spectrum(
                stiffness,
                mass,
                num_modes=2,
                algebraic_nullity=1,
                target_shift=10.0,
            )
        except AlgebraicDynamicsError as error:
            outcomes.append(("error", str(error)))
        else:
            errors = _normwise_backward_errors(
                stiffness,
                mass,
                spectrum.eigenvalues,
                spectrum.eigenvectors,
            )
            outcomes.append(
                (
                    "result",
                    spectrum.eigenvalues.copy(),
                    spectrum.eigenvectors.copy(),
                    errors,
                )
            )

    assert outcomes[0][0] == outcomes[1][0]
    if outcomes[0][0] == "error":
        assert outcomes[0][1] == outcomes[1][1]
        assert "target shift" in outcomes[0][1].lower()
    else:
        for outcome in outcomes:
            np.testing.assert_allclose(outcome[1][0], 10.0, rtol=0.0, atol=2.0e-10)
            assert float(np.max(outcome[3])) <= 2.0e-9
        np.testing.assert_array_equal(outcomes[0][1], outcomes[1][1])
        np.testing.assert_array_equal(outcomes[0][2], outcomes[1][2])


def test_sparse_descriptor_rejects_a_small_physical_negative_mode() -> None:
    size = 30
    mass = np.diag(np.concatenate((np.ones(size - 1), (0.0,))))
    stiffness = np.diag(
        np.concatenate(((-2.5e-7,), np.ones(size - 2), (1.0,)))
    )

    with pytest.raises(AlgebraicDynamicsError, match="negative|unstable"):
        _sparse_spectrum(
            stiffness,
            mass,
            num_modes=2,
            algebraic_nullity=1,
        )


def test_sparse_descriptor_clamps_only_roundoff_negative_rigid_value() -> None:
    size = 30
    mass = np.diag(np.concatenate((np.ones(size - 1), (0.0,))))
    stiffness = np.diag(
        np.concatenate(((-64.0 * np.finfo(float).eps,), np.ones(size - 2), (1.0,)))
    )

    spectrum = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=2,
        algebraic_nullity=1,
    )

    assert float(np.min(spectrum.eigenvalues)) == 0.0


def test_massless_algebraic_stiffness_cannot_hide_physical_instability() -> None:
    size = 30
    mass = np.diag(np.concatenate((np.ones(size - 1), (0.0,))))
    stiffness = np.diag(
        np.concatenate(((-1.0,), np.full(size - 2, 10.0), (1.0e16,)))
    )

    with pytest.raises(AlgebraicDynamicsError, match="negative|positive definite"):
        _sparse_spectrum(
            stiffness,
            mass,
            num_modes=2,
            algebraic_nullity=1,
        )


@pytest.mark.parametrize("coupling", np.logspace(3.0, 8.0, 6))
def test_algebraic_coordinate_mixing_preserves_finite_spectrum_and_sign(
    coupling: float,
) -> None:
    size = 80
    algebraic_basis = sparse.csc_matrix(
        (np.asarray((1.0,)), (np.asarray((size - 1,)), np.asarray((0,)))),
        shape=(size, 1),
    )
    mass = sparse.diags(
        np.concatenate((np.ones(size - 1), (0.0,))), format="csr"
    )

    stiffness = sparse.diags(
        np.concatenate(((10.0,), np.arange(20.0, 20.0 + size - 2), (1.0,))),
        format="lil",
    )
    # This is the congruent re-expression a = a_hat - C*q of the
    # uncoupled block diag(10, 1).  Its finite generalized eigenvalue is
    # exactly 10 for every C, while the algebraic coordinate grows as C*q.
    stiffness[0, 0] = 10.0 + coupling * coupling
    stiffness[0, size - 1] = -coupling
    stiffness[size - 1, 0] = -coupling

    spectrum = _sparse_spectrum(
        stiffness.tocsr(),
        mass,
        num_modes=3,
        algebraic_nullity=1,
        algebraic_basis=algebraic_basis,
    )

    assert np.all(spectrum.eigenvalues > 0.0)
    np.testing.assert_allclose(
        spectrum.eigenvalues[:3],
        np.asarray((10.0, 20.0, 21.0)),
        rtol=2.0e-8,
        atol=2.0e-8,
    )


@pytest.mark.parametrize("coupling", (1.0e6, 1.0e8))
def test_bounded_static_condensation_catches_trial_orthogonal_coordinate_shear(
    coupling: float,
) -> None:
    size = 80
    finite_dimension = size - 1
    trial_physical = _deterministic_block(size, 8)[:finite_dimension, :]
    # Use the unique null direction of an 8x9 leading block rather than an
    # arbitrary vector from the full 71-dimensional nullspace.  This keeps the
    # adversarial fixture stable across LAPACK implementations.
    local_shear = linalg.null_space(trial_physical[:9, :].T)[:, 0]
    shear_direction = np.zeros(finite_dimension, dtype=float)
    shear_direction[:9] = local_shear
    assert float(np.linalg.norm(trial_physical.T @ shear_direction)) < 2.0e-15

    physical_stiffness = np.diag(
        np.arange(10.0, 10.0 + finite_dimension)
    ) + coupling * coupling * np.outer(shear_direction, shear_direction)
    algebraic_coupling = -coupling * shear_direction
    stiffness = np.zeros((size, size), dtype=float)
    stiffness[:finite_dimension, :finite_dimension] = physical_stiffness
    stiffness[:finite_dimension, -1] = algebraic_coupling
    stiffness[-1, :finite_dimension] = algebraic_coupling
    stiffness[-1, -1] = 1.0
    mass = sparse.diags(
        np.concatenate((np.ones(finite_dimension), (0.0,))), format="csr"
    )
    algebraic_basis = sparse.csc_matrix(
        (np.asarray((1.0,)), (np.asarray((size - 1,)), np.asarray((0,)))),
        shape=(size, 1),
    )
    represented_schur = physical_stiffness - np.outer(
        algebraic_coupling, algebraic_coupling
    )
    expected = np.linalg.eigvalsh(
        0.5 * (represented_schur + represented_schur.T)
    )[:3]

    spectrum = _sparse_spectrum(
        sparse.csr_matrix(stiffness),
        mass,
        num_modes=3,
        algebraic_nullity=1,
        algebraic_basis=algebraic_basis,
    )

    # The former eight-vector activation probe is intentionally blind to this
    # direction.  Bounded descriptors must therefore select condensation by
    # policy, not by the sampled correction magnitude.
    assert spectrum.diagnostics["algebraic_static_correction_ratio"] < 1.0e-6
    assert spectrum.diagnostics["solver"] == "dense_algebraic_static_condensation_eigh"
    assert (
        spectrum.diagnostics["sparse_mode"]
        == "coordinate_invariant_static_condensation_fallback"
    )
    assert np.all(spectrum.eigenvalues[:3] > 0.0)
    assert spectrum.diagnostics["candidate_eigenvalue_policy_id"] == (
        DESCRIPTOR_RAYLEIGH_REFINEMENT_POLICY_ID
    )
    assert spectrum.diagnostics["candidate_eigenvalue_disposition"] == (
        "CONDENSED_INPUT_PRESERVED"
    )
    assert spectrum.diagnostics["candidate_rayleigh_refined_count"] == 0
    assert spectrum.diagnostics["candidate_condensed_preserved_count"] > 0
    assert "COORDINATE_SHEAR" in spectrum.diagnostics[
        "candidate_rayleigh_refinement_rejection_reasons"
    ]
    np.testing.assert_allclose(
        spectrum.eigenvalues[:3], expected, rtol=2.0e-10, atol=2.0e-9
    )


@pytest.mark.parametrize("target_shift", (100.0, 1.0e6))
def test_target_outside_finite_spectrum_excludes_algebraic_modes(
    target_shift: float,
) -> None:
    finite = np.arange(1.0, 31.0)
    nullity = 2
    size = int(finite.size + nullity)
    mass = sparse.diags(
        np.concatenate((np.ones(finite.size), np.zeros(nullity))), format="csr"
    )
    stiffness = sparse.diags(
        np.concatenate((finite, np.asarray((200.0, 201.0)))), format="csr"
    )
    algebraic_basis = sparse.csc_matrix(
        (
            np.ones(nullity),
            (
                np.arange(size - nullity, size),
                np.arange(nullity),
            ),
        ),
        shape=(size, nullity),
    )

    spectrum = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=2,
        algebraic_nullity=nullity,
        algebraic_basis=algebraic_basis,
        target_shift=target_shift,
    )

    np.testing.assert_allclose(
        spectrum.eigenvalues,
        np.asarray((30.0, 29.0)),
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    assert spectrum.diagnostics["target_excluded_algebraic_ritz_vectors"] >= 1


@pytest.mark.parametrize(
    ("isolated_nearest", "crowded_side"),
    (
        (90.0, np.linspace(110.1, 112.4, 20)),
        (110.0, np.linspace(89.9, 87.6, 20)),
    ),
)
def test_nonlinear_target_distance_crowding_selects_true_nearest_on_both_sides(
    isolated_nearest: float,
    crowded_side: np.ndarray,
) -> None:
    target = 100.0
    finite = np.concatenate(
        (
            np.asarray((isolated_nearest,)),
            np.asarray(crowded_side, dtype=float),
            np.linspace(150.0, 190.0, 12),
        )
    )
    stiffness, mass = _congruent_descriptor(
        finite,
        algebraic_nullity=2,
        condition=3.0,
        seed=818,
    )

    spectrum = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=1,
        algebraic_nullity=2,
        target_shift=target,
    )

    np.testing.assert_allclose(
        spectrum.eigenvalues,
        np.asarray((isolated_nearest,)),
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def _large_sparse_diagonal_descriptor():
    size = 540
    nullity = 2
    finite = np.arange(1.0, size - nullity + 1.0)
    mass = sparse.diags(
        np.concatenate((np.ones(finite.size), np.zeros(nullity))), format="csr"
    )
    stiffness = sparse.diags(
        np.concatenate((finite, np.asarray((700.0, 701.0)))), format="csr"
    )
    basis = sparse.csc_matrix(
        (
            np.ones(nullity),
            (np.arange(size - nullity, size), np.arange(nullity)),
        ),
        shape=(size, nullity),
    )
    return stiffness, mass, basis


def test_large_sparse_swapped_solver_returns_lowest_physical_modes() -> None:
    stiffness, mass, basis = _large_sparse_diagonal_descriptor()

    spectrum = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=3,
        algebraic_nullity=2,
        algebraic_basis=basis,
    )

    np.testing.assert_allclose(spectrum.eigenvalues[:3], (1.0, 2.0, 3.0))
    assert spectrum.diagnostics["sparse_mode"] == (
        "lowest_symmetric_swapped_block_lobpcg"
    )


def test_default_bounded_descriptor_extent_uses_static_condensation() -> None:
    stiffness, mass, basis = _large_sparse_diagonal_descriptor()

    spectrum = solve_descriptor_spectrum(
        stiffness,
        mass,
        num_modes=3,
        dense_size_limit=1,
        algebraic_nullity=2,
        algebraic_basis=basis,
    )

    np.testing.assert_allclose(spectrum.eigenvalues[:3], (1.0, 2.0, 3.0))
    assert spectrum.diagnostics["static_condensation_limit"] == 3072
    assert spectrum.diagnostics["solver"] == (
        "dense_algebraic_static_condensation_eigh"
    )


@pytest.mark.parametrize("limit", (0, -1, True, 1.5, "3072"))
def test_static_condensation_extent_rejects_invalid_values(limit: object) -> None:
    stiffness = sparse.diags((1.0, 2.0, 3.0), format="csr")
    mass = sparse.diags((1.0, 1.0, 0.0), format="csr")
    basis = sparse.csc_matrix(([1.0], ([2], [0])), shape=(3, 1))

    with pytest.raises(ValueError, match="positive integer"):
        solve_descriptor_spectrum(
            stiffness,
            mass,
            num_modes=1,
            dense_size_limit=1,
            algebraic_nullity=1,
            algebraic_basis=basis,
            static_condensation_limit=limit,
        )


def test_large_sparse_swapped_solver_preserves_repeated_multiplicity_deterministically() -> None:
    size = 540
    nullity = 2
    finite = np.concatenate(
        (np.ones(8), np.full(8, 2.0), np.arange(3.0, size - nullity - 13.0))
    )
    assert finite.size == size - nullity
    mass = sparse.diags(np.concatenate((np.ones(finite.size), np.zeros(nullity))))
    stiffness = sparse.diags(np.concatenate((finite, (700.0, 701.0))))
    basis = sparse.csc_matrix(
        (
            np.ones(nullity),
            (np.arange(size - nullity, size), np.arange(nullity)),
        ),
        shape=(size, nullity),
    )

    first = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=8,
        algebraic_nullity=nullity,
        algebraic_basis=basis,
    )
    second = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=8,
        algebraic_nullity=nullity,
        algebraic_basis=basis,
    )

    np.testing.assert_allclose(first.eigenvalues[:8], np.ones(8), atol=2.0e-10)
    np.testing.assert_allclose(
        first.eigenvectors[:, :8].T @ mass @ first.eigenvectors[:, :8],
        np.eye(8),
        atol=2.0e-10,
    )
    np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
    np.testing.assert_array_equal(first.eigenvectors, second.eigenvectors)


@pytest.mark.parametrize(
    ("target_shift", "expected", "minimum_excluded"),
    (
        (300.25, (300.0, 301.0), 0),
        (1.0e6, (538.0, 537.0), 2),
    ),
)
def test_large_sparse_two_sided_target_filters_algebraic_ritz_vectors(
    target_shift: float,
    expected: tuple[float, float],
    minimum_excluded: int,
) -> None:
    stiffness, mass, basis = _large_sparse_diagonal_descriptor()

    spectrum = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=2,
        algebraic_nullity=2,
        algebraic_basis=basis,
        target_shift=target_shift,
    )

    np.testing.assert_allclose(spectrum.eigenvalues, expected)
    assert spectrum.diagnostics["sparse_mode"] == (
        "targeted_two_sided_projected_swapped_eigsh"
    )
    assert (
        spectrum.diagnostics["target_excluded_algebraic_ritz_vectors"]
        >= minimum_excluded
    )


@pytest.mark.parametrize(
    ("isolated_nearest", "crowded_side"),
    (
        (90.0, np.linspace(110.1, 112.4, 20)),
        (110.0, np.linspace(89.9, 87.6, 20)),
    ),
)
def test_large_sparse_two_sided_target_uses_lambda_distance_not_mu_distance(
    isolated_nearest: float,
    crowded_side: np.ndarray,
) -> None:
    size = 540
    nullity = 2
    head = np.concatenate(((isolated_nearest,), crowded_side))
    tail = np.linspace(150.0, 1000.0, size - nullity - head.size)
    finite = np.concatenate((head, tail))
    mass = sparse.diags(np.concatenate((np.ones(finite.size), np.zeros(nullity))))
    stiffness = sparse.diags(np.concatenate((finite, (1200.0, 1201.0))))
    basis = sparse.csc_matrix(
        (
            np.ones(nullity),
            (np.arange(size - nullity, size), np.arange(nullity)),
        ),
        shape=(size, nullity),
    )

    spectrum = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=1,
        algebraic_nullity=nullity,
        algebraic_basis=basis,
        target_shift=100.0,
    )

    np.testing.assert_allclose(spectrum.eigenvalues, (isolated_nearest,))
    assert spectrum.diagnostics["sparse_mode"] == (
        "targeted_two_sided_projected_swapped_eigsh"
    )


@pytest.mark.parametrize("target_shift", [None, 17.25])
def test_sparse_descriptor_repeats_are_byte_identical(
    target_shift: Optional[float],
) -> None:
    finite = np.linspace(1.0, 60.0, 38)
    stiffness, mass = _congruent_descriptor(
        finite,
        algebraic_nullity=2,
        condition=3.0,
        seed=111,
    )

    first = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=8,
        algebraic_nullity=2,
        target_shift=target_shift,
    )
    second = _sparse_spectrum(
        stiffness,
        mass,
        num_modes=8,
        algebraic_nullity=2,
        target_shift=target_shift,
    )

    np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
    np.testing.assert_array_equal(first.eigenvectors, second.eigenvectors)
