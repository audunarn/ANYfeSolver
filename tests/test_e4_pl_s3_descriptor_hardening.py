from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    FEModel,
    FixedSupport,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
)
from anyfesolver.algebraic_dynamics import (
    AlgebraicDynamicsError,
    _certify_spd,
    _sparse_nullspace_basis,
    build_declared_algebraic_basis,
    solve_descriptor_spectrum,
)
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.matrix_assembly import assemble_mass_matrix, assemble_stiffness_matrix
from anyfesolver.shell_sections import GeneralizedShellSection


def _q4_section(*, rotary_inertia_per_area: float) -> GeneralizedShellSection:
    return GeneralizedShellSection(
        A=np.asarray(((2.0, 1.0, 0.0), (1.0, 2.0, 0.0), (0.0, 0.0, 0.5))),
        B=np.zeros((3, 3)),
        D=np.asarray(((0.2, 0.1, 0.0), (0.1, 0.2, 0.0), (0.0, 0.0, 0.05))),
        As=np.eye(2),
        mass_per_area=1.0,
        rotary_inertia_per_area=rotary_inertia_per_area,
    )


def _mixed_model(
    *, rotary_inertia_per_area: float, reverse_element_insertion: bool = False
) -> FEModel:
    model = FEModel("mixed-qualified-shell-descriptor-hardening")
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
    q4 = QualifiedE4PLShellElement(
        1,
        [1, 2, 3, 4],
        "steel",
        thickness=0.02,
        shell_section=_q4_section(
            rotary_inertia_per_area=rotary_inertia_per_area
        ),
    )
    s3 = QualifiedE4PLS3ShellElement(
        2,
        [2, 5, 3],
        "steel",
        thickness=0.02,
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
    )
    ordered = ((2, s3), (1, q4)) if reverse_element_insertion else ((1, q4), (2, s3))
    for element_id, element in ordered:
        model.add_element(element_id, element)
    model.add_boundary_condition(FixedSupport("left-edge", [1, 4]))
    return model


def _assembled_basis(model: FEModel):
    model.apply_boundary_conditions()
    stiffness, _stiffness_info = assemble_stiffness_matrix(model)
    mass, _mass_info = assemble_mass_matrix(model)
    zero = np.zeros(stiffness.shape[0], dtype=float)
    _stiffness_reduced, _load_reduced, transform, _offset, independent, _info = (
        build_constraint_transformation(stiffness, zero, model)
    )
    reduced_mass = (transform.T @ mass @ transform).tocsr()
    basis = build_declared_algebraic_basis(
        model,
        mass,
        reduced_mass,
        transform,
        independent,
        dense_size_limit=200,
    )
    return basis, reduced_mass


def _two_s3_facet_model(*, fold_height: float) -> tuple[FEModel, np.ndarray, np.ndarray]:
    model = FEModel("two-qualified-s3-facets")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    first_edge = np.asarray((0.1, 0.1, 0.5), dtype=float)
    second_edge = np.asarray((0.1, 0.5, 0.1), dtype=float)
    owner = np.cross(first_edge, second_edge)
    owner_unit = owner / np.linalg.norm(owner)
    fourth = first_edge + second_edge + float(fold_height) * owner_unit
    coordinates = (
        np.zeros(3, dtype=float),
        first_edge,
        second_edge,
        fourth,
    )
    for node_id, coordinate in enumerate(coordinates, start=1):
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
    model.add_element(
        2,
        QualifiedE4PLS3ShellElement(
            2,
            [2, 4, 3],
            "steel",
            thickness=0.02,
            reference_normal=owner,
        ),
    )
    first_normal = owner_unit
    second_owner = np.cross(fourth - first_edge, second_edge - first_edge)
    second_normal = second_owner / np.linalg.norm(second_owner)
    return model, first_normal, second_normal


def _unconstrained_declared_basis(model: FEModel):
    full_mass, _mass_info = assemble_mass_matrix(model)
    size = int(full_mass.shape[0])
    identity = sparse.eye(size, format="csr")
    return build_declared_algebraic_basis(
        model,
        full_mass,
        full_mass,
        identity,
        np.arange(size, dtype=np.intp),
        dense_size_limit=200,
    )


def test_sparse_spd_certificate_rejects_an_exact_singular_matrix() -> None:
    size = 64
    singular = sparse.eye(size, format="lil", dtype=float)
    singular[-1, -1] = 0.0

    with pytest.raises(AlgebraicDynamicsError, match="positive definite|certify"):
        _certify_spd(
            singular.tocsr(),
            dense_size_limit=1,
            label="exactly singular sparse regression",
        )


@pytest.mark.parametrize("dense_size_limit", (1, 100))
def test_spd_certificate_rejects_rank_deficient_gram_matrix(
    dense_size_limit: int,
) -> None:
    rectangular = np.asarray(
        (
            (-3, 0, 2, -2, 2, 3, 0, 2),
            (-1, -2, -2, 1, -2, 0, -1, 1),
            (0, -3, -2, -1, 1, -1, 2, 2),
            (-1, 0, 2, -3, -1, -1, 1, -3),
            (0, -3, 0, 2, 2, -3, 1, -1),
            (1, 2, 3, 3, 1, 2, 3, -1),
            (3, 2, 3, -2, 3, 3, 1, 1),
        ),
        dtype=float,
    )
    singular = sparse.csr_matrix(rectangular.T @ rectangular)

    with pytest.raises(AlgebraicDynamicsError, match="positive definite|certify"):
        _certify_spd(
            singular,
            dense_size_limit=dense_size_limit,
            label="rank-deficient Gram regression",
        )


def test_sparse_rref_nullspace_is_invariant_to_independent_row_scaling() -> None:
    equations = sparse.csr_matrix(
        np.asarray(((1.0, -1.0, 0.0), (0.0, 1.0, -1.0)))
    )
    scaled = sparse.diags((1.0e-13, 3.0e11), format="csr") @ equations

    baseline = _sparse_nullspace_basis(equations)
    actual = _sparse_nullspace_basis(scaled)

    assert baseline.shape == actual.shape == (3, 1)
    np.testing.assert_allclose(
        baseline.toarray(), np.ones((3, 1)), rtol=0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(actual.toarray(), baseline.toarray(), rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        (scaled @ actual).toarray(),
        np.zeros((2, 1)),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_sparse_rref_rejects_an_ambiguous_small_constraint_coefficient() -> None:
    equations = sparse.csr_matrix(np.asarray(((1.0, 1.0e-13),)))

    with pytest.raises(AlgebraicDynamicsError, match="ambiguous nonzero"):
        _sparse_nullspace_basis(equations)


def test_sparse_rref_uses_stable_pivots_for_hierarchical_chain() -> None:
    size = 24
    equations = sparse.diags(
        (np.full(size - 1, 1.0e-10), np.ones(size - 1)),
        (0, 1),
        shape=(size - 1, size),
        format="csr",
    )

    basis = _sparse_nullspace_basis(equations)

    assert basis.shape == (size, 1)
    assert np.all(np.isfinite(basis.data))
    residual = (equations @ basis).toarray().reshape(-1)
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1.0e-14)
    assert float(np.max(np.abs(basis.data))) <= 1.0


@pytest.mark.parametrize("rotary_inertia_per_area", (1.0e-20, 1.0e-6))
def test_any_positive_q4_rotary_inertia_removes_shared_s3_drills(
    rotary_inertia_per_area: float,
) -> None:
    basis, reduced_mass = _assembled_basis(
        _mixed_model(rotary_inertia_per_area=rotary_inertia_per_area)
    )

    certificate = basis.diagnostics
    assert certificate["candidate_node_ids"] == [5]
    assert certificate["mass_removed_node_ids"] == [2, 3]
    assert certificate["compatible_global_nullity"] == 1
    assert basis.full_basis.shape[1] == basis.reduced_basis.shape[1] == 1
    action = reduced_mass @ basis.reduced_basis
    np.testing.assert_allclose(action.toarray(), 0.0, rtol=0.0, atol=2.0e-13)


def test_mixed_basis_and_certificate_are_insertion_order_deterministic() -> None:
    first, _first_mass = _assembled_basis(
        _mixed_model(rotary_inertia_per_area=0.75)
    )
    second, _second_mass = _assembled_basis(
        _mixed_model(
            rotary_inertia_per_area=0.75,
            reverse_element_insertion=True,
        )
    )

    assert first.diagnostics == second.diagnostics
    for left, right in (
        (first.full_basis, second.full_basis),
        (first.reduced_basis, second.reduced_basis),
    ):
        np.testing.assert_array_equal(left.indptr, right.indptr)
        np.testing.assert_array_equal(left.indices, right.indices)
        np.testing.assert_array_equal(left.data, right.data)


def test_one_ulp_coplanar_facet_normals_share_structural_drill_coordinates() -> None:
    model, first_normal, second_normal = _two_s3_facet_model(fold_height=0.0)

    # The component in which normalization rounds by one ULP is permitted to
    # vary with the platform/libm.  The contract under test is that the two
    # independently constructed normals differ by no more than one ULP, not
    # that a particular component owns that rounding difference.
    np.testing.assert_array_max_ulp(first_normal, second_normal, maxulp=1)
    assert np.any(first_normal != second_normal)
    basis = _unconstrained_declared_basis(model)

    assert basis.diagnostics["noncoplanar_node_ids"] == []
    assert basis.diagnostics["candidate_node_ids"] == [1, 2, 3, 4]
    assert basis.diagnostics["compatible_global_nullity"] == 4


def test_genuine_near_fold_rejects_shared_structural_drill_coordinates() -> None:
    model, first_normal, second_normal = _two_s3_facet_model(fold_height=1.0e-4)
    separation = float(np.linalg.norm(first_normal - second_normal))

    assert 256.0 * np.finfo(float).eps < separation < 1.0e-3
    basis = _unconstrained_declared_basis(model)

    assert basis.diagnostics["noncoplanar_node_ids"] == [2, 3]
    assert basis.diagnostics["candidate_node_ids"] == [1, 4]
    assert basis.diagnostics["compatible_global_nullity"] == 2


def test_sparse_descriptor_returns_every_requested_finite_mode_near_dimension() -> None:
    size = 20
    finite_dimension = size - 1
    stiffness = sparse.diags(np.arange(1.0, size + 1.0), format="csr")
    mass = sparse.diags(np.concatenate((np.ones(finite_dimension), (0.0,))), format="csr")

    spectrum = solve_descriptor_spectrum(
        stiffness,
        mass,
        num_modes=finite_dimension,
        dense_size_limit=1,
        algebraic_nullity=1,
    )

    assert spectrum.eigenvalues.shape == (finite_dimension,)
    np.testing.assert_allclose(
        np.sort(spectrum.eigenvalues),
        np.arange(1.0, size),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
