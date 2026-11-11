from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from anyfesolver import (
    AnalysisSession,
    BoundaryCondition,
    FEModel,
    FixedSupport,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
    solve_free_vibration,
)
from anyfesolver.algebraic_dynamics import (
    DESCRIPTOR_MODAL_POLICY_ID,
    DESCRIPTOR_RAYLEIGH_REFINEMENT_POLICY_ID,
    solve_descriptor_spectrum,
)
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.matrix_assembly import assemble_mass_matrix, assemble_stiffness_matrix
from anyfesolver.shell_sections import GeneralizedShellSection


def _supported_model(*, inclined: bool = False) -> FEModel:
    model = FEModel("qualified-s3-modal")
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)), dtype=float
    )
    if inclined:
        angle = np.deg2rad(31.0)
        rotation = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, np.cos(angle), -np.sin(angle)),
                (0.0, np.sin(angle), np.cos(angle)),
            )
        )
        nodes = nodes @ rotation.T
    owner = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(nodes, start=1):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.02,
            reference_normal=owner,
        ),
    )
    model.add_boundary_condition(FixedSupport("fixed-edge", [1, 2]))
    model.add_boundary_condition(
        BoundaryCondition("node-3-in-plane", [3], {"ux": 0.0, "uy": 0.0})
    )
    return model


def _reduced_matrices(model: FEModel) -> tuple[np.ndarray, np.ndarray]:
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    mass, _ = assemble_mass_matrix(model)
    zero = np.zeros(stiffness.shape[0])
    reduced_stiffness, _, transform, _, _, _ = build_constraint_transformation(
        stiffness, zero, model
    )
    reduced_mass = transform.T @ mass @ transform
    return reduced_stiffness.toarray(), reduced_mass.toarray()


def _explicit_finite_eigenvalues(stiffness: np.ndarray, mass: np.ndarray) -> np.ndarray:
    # Independent dense descriptor oracle: homogeneous generalized eigenvalues
    # represent infinite algebraic modes with beta == 0.
    alpha_beta, _vectors = linalg.eig(
        stiffness,
        mass,
        homogeneous_eigvals=True,
        right=True,
    )
    alpha, beta = alpha_beta
    scale = max(float(np.max(np.abs(beta))), np.finfo(float).tiny)
    finite = np.abs(beta) > np.finfo(float).eps * len(beta) * scale
    values = np.real_if_close(alpha[finite] / beta[finite], tol=1000)
    values = np.asarray(np.real(values), dtype=float)
    return np.sort(values[np.isfinite(values)])


@pytest.mark.parametrize("inclined", [False, True])
def test_supported_s3_modal_excludes_three_infinite_drill_modes(
    inclined: bool,
) -> None:
    model = _supported_model(inclined=inclined)
    stiffness, mass = _reduced_matrices(model)
    oracle = _explicit_finite_eigenvalues(stiffness, mass)

    result = solve_free_vibration(model, num_modes=len(oracle))

    assert result.solver_status == "ok"
    assert result.num_modes_returned == len(oracle)
    assert result.diagnostics["descriptor_modal"] is True
    assert result.diagnostics["policy_id"] == DESCRIPTOR_MODAL_POLICY_ID
    assert result.diagnostics["declared_algebraic_element_ids"] == [1]
    assert result.diagnostics["declared_algebraic_mass_certificate"][
        "compatible_global_nullity"
    ] == 1
    np.testing.assert_allclose(
        np.asarray([mode.eigenvalue for mode in result.modes]),
        oracle,
        rtol=2.0e-10,
        atol=2.0e-7,
    )
    assert result.diagnostics["max_residual_norm"] < 2.0e-10
    assert result.diagnostics["mass_orthogonality_error"] < 2.0e-10
    assert all(mode.mode_shape.shape == (18,) for mode in result.modes)


def test_swapped_pencil_matches_explicit_static_condensation() -> None:
    stiffness = np.asarray(
        (
            (9.0, 1.0, 2.0),
            (1.0, 5.0, -1.0),
            (2.0, -1.0, 4.0),
        )
    )
    mass = np.diag((3.0, 2.0, 0.0))
    schur = stiffness[:2, :2] - np.outer(stiffness[:2, 2], stiffness[2, :2]) / stiffness[2, 2]
    expected = linalg.eigvalsh(schur, mass[:2, :2])

    spectrum = solve_descriptor_spectrum(
        stiffness,
        mass,
        num_modes=2,
        dense_size_limit=10,
        algebraic_nullity=1,
    )

    np.testing.assert_allclose(np.sort(spectrum.eigenvalues), expected, rtol=1.0e-13)
    assert spectrum.diagnostics["excluded_algebraic_candidates"] == 1


def test_descriptor_modal_rejects_common_mass_and_stiffness_mechanism() -> None:
    stiffness = np.diag((2.0, 0.0))
    mass = np.diag((1.0, 0.0))

    with pytest.raises(ValueError, match="not positive definite"):
        solve_descriptor_spectrum(
            stiffness,
            mass,
            num_modes=1,
            dense_size_limit=10,
            algebraic_nullity=1,
        )


def test_free_s3_retains_six_rigid_modes_without_drill_inertia() -> None:
    model = _supported_model()
    model.boundary_conditions.clear()

    result = solve_free_vibration(model, num_modes=9)

    assert result.solver_status == "ok"
    assert result.num_modes_returned == 9
    assert result.diagnostics["descriptor_modal"] is True
    assert result.diagnostics["num_rigid_body_modes"] == 6
    assert np.max(result.frequencies_hz[:6]) < 2.0e-3
    assert np.min(result.frequencies_hz[6:]) > 1.0
    assert result.nullspace_info["rank"] == 6


def test_bounded_descriptor_and_target_shift_match_dense_oracle() -> None:
    size = 30
    mass = np.diag(np.concatenate((np.linspace(1.0, 2.0, size - 2), (0.0, 0.0))))
    base = np.diag(np.concatenate((np.linspace(10.0, 280.0, size - 2), (320.0, 350.0))))
    coupling = np.zeros((size, size))
    coupling[-2, 2] = coupling[2, -2] = 0.4
    coupling[-1, 5] = coupling[5, -1] = -0.3
    stiffness = base + coupling
    oracle = _explicit_finite_eigenvalues(stiffness, mass)

    lowest = solve_descriptor_spectrum(
        stiffness,
        mass,
        num_modes=4,
        dense_size_limit=1,
        algebraic_nullity=2,
    )
    np.testing.assert_allclose(
        np.sort(lowest.eigenvalues)[:4], oracle[:4], rtol=2.0e-11, atol=1.0e-12
    )
    assert (
        lowest.diagnostics["sparse_mode"]
        == "coordinate_invariant_static_condensation_fallback"
    )

    target = 100.0
    targeted = solve_descriptor_spectrum(
        stiffness,
        mass,
        num_modes=2,
        dense_size_limit=1,
        algebraic_nullity=2,
        target_shift=target,
    )
    nearest = np.argsort(np.abs(oracle - target))[: len(targeted.eigenvalues)]
    np.testing.assert_allclose(
        np.sort(targeted.eigenvalues),
        np.sort(oracle[nearest]),
        rtol=2.0e-10,
        atol=1.0e-11,
    )
    assert (
        targeted.diagnostics["sparse_mode"]
        == "coordinate_invariant_static_condensation_fallback"
    )


def test_descriptor_retains_extremely_small_but_finite_physical_mass() -> None:
    stiffness = np.eye(3)
    mass = np.diag((1.0, 1.0e-20, 0.0))

    spectrum = solve_descriptor_spectrum(
        stiffness,
        mass,
        num_modes=2,
        dense_size_limit=10,
        algebraic_nullity=1,
    )

    np.testing.assert_allclose(
        np.sort(spectrum.eigenvalues), np.asarray((1.0, 1.0e20)), rtol=2.0e-14
    )


def test_sparse_metric_spd_certificate_rejects_indefinite_stiffness() -> None:
    size = 14
    stiffness = np.diag(np.concatenate(((-2.0,), np.full(size - 1, 2.0))))
    mass = np.diag(np.concatenate((np.ones(size - 1), (0.0,))))

    with pytest.raises(ValueError, match="not positive definite"):
        solve_descriptor_spectrum(
            stiffness,
            mass,
            num_modes=2,
            dense_size_limit=1,
            algebraic_nullity=1,
        )


def test_sparse_descriptor_is_repeatable_from_frozen_start_vector() -> None:
    size = 24
    mass = np.diag(np.concatenate((np.linspace(1.0, 2.0, size - 1), (0.0,))))
    stiffness = np.diag(np.linspace(3.0, 80.0, size))

    first = solve_descriptor_spectrum(
        stiffness,
        mass,
        num_modes=5,
        dense_size_limit=1,
        algebraic_nullity=1,
    )
    second = solve_descriptor_spectrum(
        stiffness,
        mass,
        num_modes=5,
        dense_size_limit=1,
        algebraic_nullity=1,
    )

    np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
    np.testing.assert_array_equal(first.eigenvectors, second.eigenvectors)


def test_inclined_declared_drill_directions_are_exact_mass_nulls() -> None:
    model = _supported_model(inclined=True)
    element = model.mesh.elements[1]
    material = model.get_material("steel")
    directions = element.dynamic_algebraic_directions(model.mesh, material)
    mass = element.compute_mass_matrix(model.mesh, material)

    np.testing.assert_allclose(directions.T @ directions, np.eye(3), atol=2.0e-15)
    assert np.linalg.norm(mass @ directions, ord=np.inf) / np.linalg.norm(
        mass, ord=np.inf
    ) < 2.0e-16
    assert any(
        np.count_nonzero(np.abs(directions[6 * node + 3 : 6 * node + 6, node]) > 1.0e-12)
        > 1
        for node in range(3)
    )


def test_descriptor_modal_fails_closed_for_zero_areal_inertia() -> None:
    model = _supported_model()
    model.materials["steel"].density = 0.0

    result = solve_free_vibration(model, num_modes=1)

    assert result.solver_status == "failed"
    assert "positive areal mass and rotary inertia" in result.diagnostics["error"]


def test_descriptor_modal_session_matches_uncached_path() -> None:
    model = _supported_model(inclined=True)
    expected = solve_free_vibration(model, num_modes=3)

    with AnalysisSession(model) as session:
        actual = solve_free_vibration(model, num_modes=3, session=session)
        repeated = solve_free_vibration(model, num_modes=3, session=session)

    np.testing.assert_allclose(actual.frequencies_hz, expected.frequencies_hz, rtol=2.0e-12)
    np.testing.assert_allclose(repeated.frequencies_hz, actual.frequencies_hz, rtol=2.0e-12)
    assert actual.diagnostics["descriptor_modal"] is True
    assert actual.diagnostics["analysis_session"]["plan_reused"] is True
    assert repeated.diagnostics["analysis_session"]["plan_reused"] is True


def test_mixed_qualified_q4_s3_modal_matches_homogeneous_pencil() -> None:
    model = FEModel("mixed-qualified-shell-modal")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        QualifiedE4PLShellElement(1, [1, 2, 3, 4], "steel", thickness=0.02),
    )
    model.add_element(
        2,
        QualifiedE4PLS3ShellElement(
            2,
            [2, 5, 3],
            "steel",
            thickness=0.02,
            reference_normal=np.asarray((0.0, 0.0, 1.0)),
        ),
    )
    model.add_boundary_condition(FixedSupport("left-edge", [1, 4]))
    stiffness, mass = _reduced_matrices(model)
    oracle = _explicit_finite_eigenvalues(stiffness, mass)

    result = solve_free_vibration(model, num_modes=4)

    assert result.solver_status == "ok"
    assert result.diagnostics["declared_algebraic_element_ids"] == [2]
    assert result.diagnostics["candidate_eigenvalue_policy_id"] == (
        DESCRIPTOR_RAYLEIGH_REFINEMENT_POLICY_ID
    )
    assert result.diagnostics["candidate_eigenvalue_disposition"] == (
        "FULL_SYSTEM_RAYLEIGH_REFINED"
    )
    assert result.diagnostics["candidate_rayleigh_refined_count"] > 0
    assert result.diagnostics["candidate_condensed_preserved_count"] == 0
    assert not result.diagnostics[
        "candidate_rayleigh_refinement_rejection_reasons"
    ]
    np.testing.assert_allclose(
        np.asarray([mode.eigenvalue for mode in result.modes]),
        oracle[:4],
        rtol=3.0e-9,
        atol=1.0e-7,
    )


def test_mixed_model_rejects_undeclared_q4_mass_nullspace() -> None:
    model = FEModel("mixed-undeclared-mass-null")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    section = GeneralizedShellSection(
        A=np.asarray(((2.0, 1.0, 0.0), (1.0, 2.0, 0.0), (0.0, 0.0, 0.5))),
        B=np.zeros((3, 3)),
        D=np.asarray(((0.2, 0.1, 0.0), (0.1, 0.2, 0.0), (0.0, 0.0, 0.05))),
        As=np.eye(2),
        mass_per_area=1.0,
        rotary_inertia_per_area=0.0,
    )
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1, [1, 2, 3, 4], "steel", thickness=0.02, shell_section=section
        ),
    )
    model.add_element(
        2,
        QualifiedE4PLS3ShellElement(
            2,
            [2, 5, 3],
            "steel",
            thickness=0.02,
            reference_normal=np.asarray((0.0, 0.0, 1.0)),
        ),
    )
    # Leave node 4's Q4-only zero-rotary-inertia coordinates unconstrained so
    # the assembled nullspace is strictly larger than the declared S3 drills.
    model.add_boundary_condition(FixedSupport("left-edge", [1]))

    result = solve_free_vibration(model, num_modes=2)

    assert result.solver_status == "failed"
    assert result.diagnostics["error_code"] == "ALGEBRAIC_DESCRIPTOR_INVALID"
    assert "mass plus declared algebraic projector" in result.diagnostics["error"]
