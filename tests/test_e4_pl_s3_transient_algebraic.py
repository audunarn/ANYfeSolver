from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import (
    AnalysisSession,
    BoundaryCondition,
    FEModel,
    FixedSupport,
    LoadCase,
    PressurePatch,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
    TransientConfig,
    solve_transient_newmark,
)
from anyfesolver.algebraic_dynamics import (
    AlgebraicDynamicsError,
    DESCRIPTOR_TRANSIENT_CONSTRAINED_POLICY_ID,
    DESCRIPTOR_TRANSIENT_FIRST_ORDER_POLICY_ID,
    DESCRIPTOR_TRANSIENT_POLICY_ID,
    DESCRIPTOR_TRANSIENT_STATIC_POLICY_ID,
    DeclaredAlgebraicBasis,
    build_algebraic_static_reduction,
    build_declared_algebraic_basis,
    certify_descriptor_effective_operator,
)
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.dynamics import _sampled_grid_derivative
from anyfesolver.matrix_assembly import (
    assemble_load_vector,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
)
from anyfesolver.linalg import FactorizationHandle, MatrixClass
from anyfesolver.recovery import RecoveryConfig


@dataclass(frozen=True)
class _AssembledTransient:
    stiffness: sparse.csr_matrix
    mass: sparse.csr_matrix
    load: np.ndarray
    transform: sparse.csr_matrix
    affine_offset: np.ndarray
    basis: DeclaredAlgebraicBasis


def _single_s3_model(*, coupled: bool) -> FEModel:
    model = FEModel("qualified-s3-transient-algebraic")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.02,
            reference_normal=np.asarray((0.0, 0.0, 1.0)),
        ),
    )
    if coupled:
        # This leaves in-plane coordinates active, so the PL drill block is
        # strongly coupled to mass-carrying coordinates.  It catches a solver
        # that chooses the Euclidean gauge N.T*a=0 instead of differentiating
        # the algebraic equilibrium equation.
        model.add_boundary_condition(FixedSupport("node-1", [1]))
        model.add_boundary_condition(
            BoundaryCondition("node-2", [2], {"uy": 0.0, "uz": 0.0})
        )
    else:
        # A free node-3 drill is exactly decoupled from the remaining physical
        # coordinates, which gives a compact direct-work oracle.
        model.add_boundary_condition(FixedSupport("edge-12", [1, 2]))
        model.add_boundary_condition(
            BoundaryCondition("node-3-in-plane", [3], {"ux": 0.0, "uy": 0.0})
        )
    return model


def _mixed_q4_s3_model() -> FEModel:
    model = FEModel("mixed-q4-s3-transient-algebraic")
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
        QualifiedE4PLShellElement(
            1, [1, 2, 3, 4], "steel", thickness=0.02
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
    model.add_boundary_condition(FixedSupport("left-edge", [1, 4]))
    return model


def _mpc_tied_s3_model() -> FEModel:
    model = FEModel("mpc-tied-s3-transient-algebraic")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.2, 0.9, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.2, 0.9, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    for element_id, node_ids in ((1, [1, 2, 3]), (2, [4, 5, 6])):
        model.add_element(
            element_id,
            QualifiedE4PLS3ShellElement(
                element_id,
                node_ids,
                "steel",
                thickness=0.02,
                reference_normal=np.asarray((0.0, 0.0, 1.0)),
            ),
        )
    model.add_boundary_condition(FixedSupport("edge-12", [1, 2]))
    model.add_boundary_condition(FixedSupport("edge-45", [4, 5]))
    model.add_boundary_condition(
        BoundaryCondition("node-3-in-plane", [3], {"ux": 0.0, "uy": 0.0})
    )
    model.add_boundary_condition(
        BoundaryCondition("node-6-in-plane", [6], {"ux": 0.0, "uy": 0.0})
    )
    rz3 = int(model.mesh.get_node(3).dofs[5])
    rz6 = int(model.mesh.get_node(6).dofs[5])
    model.add_constraint_equation(
        terms=((rz6, 1.0), (rz3, -1.0)),
        rhs=0.0,
        source_id="tie-drill-6-to-3",
        dependent_dof=rz6,
    )
    return model


def _assembled(model: FEModel, load_case: LoadCase) -> _AssembledTransient:
    model.apply_boundary_conditions()
    stiffness, _stiffness_info = assemble_stiffness_matrix(model)
    mass, _mass_info = assemble_mass_matrix(model)
    load, _load_info = assemble_load_vector(model, load_case)
    zero = np.zeros(stiffness.shape[0], dtype=float)
    _stiffness_reduced, _zero_reduced, transform, affine_offset, independent, _info = (
        build_constraint_transformation(stiffness, zero, model)
    )
    reduced_mass = (transform.T @ mass @ transform).tocsr()
    basis = build_declared_algebraic_basis(
        model,
        mass,
        reduced_mass,
        transform,
        independent,
        dense_size_limit=512,
    )
    return _AssembledTransient(
        stiffness=stiffness.tocsr(),
        mass=mass.tocsr(),
        load=np.asarray(load, dtype=float),
        transform=transform.tocsr(),
        affine_offset=np.asarray(affine_offset, dtype=float),
        basis=basis,
    )


def _relative_residual(value: np.ndarray, *scales: np.ndarray) -> float:
    denominator = max(
        (float(np.linalg.norm(np.asarray(scale, dtype=float))) for scale in scales),
        default=1.0,
    )
    return float(np.linalg.norm(np.asarray(value, dtype=float)) / max(denominator, 1.0))


def _assert_constant_load_dae(
    assembled: _AssembledTransient,
    result,
    *,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
) -> None:
    damping = (
        float(rayleigh_alpha) * assembled.mass
        + float(rayleigh_beta) * assembled.stiffness
    ).tocsr()
    algebraic = assembled.basis.full_basis
    for displacement, velocity, acceleration in zip(
        result.displacements, result.velocities, result.accelerations
    ):
        inertia = assembled.mass @ acceleration
        damping_force = damping @ velocity
        stiffness_force = assembled.stiffness @ displacement
        residual = inertia + damping_force + stiffness_force - assembled.load
        reduced_residual = assembled.transform.T @ residual
        assert _relative_residual(
            reduced_residual,
            assembled.transform.T @ inertia,
            assembled.transform.T @ damping_force,
            assembled.transform.T @ stiffness_force,
            assembled.transform.T @ assembled.load,
        ) <= 3.0e-9

        if algebraic.shape[1]:
            algebraic_equilibrium = algebraic.T @ (
                stiffness_force + damping_force - assembled.load
            )
            assert _relative_residual(
                algebraic_equilibrium,
                algebraic.T @ stiffness_force,
                algebraic.T @ damping_force,
                algebraic.T @ assembled.load,
            ) <= 3.0e-10

            # The load is constant.  Differentiating the algebraic equation
            # gives N.T*(K*v + C*a)=0.  In the undamped case its second
            # derivative additionally requires N.T*K*a=0.  These checks catch
            # an arbitrary Euclidean acceleration gauge.
            if rayleigh_beta == 0.0:
                differentiated = algebraic.T @ (
                    assembled.stiffness @ velocity + damping @ acceleration
                )
                assert _relative_residual(
                    differentiated,
                    assembled.stiffness @ velocity,
                    damping @ acceleration,
                    algebraic.T @ (assembled.stiffness @ velocity),
                    algebraic.T @ (damping @ acceleration),
                ) <= 3.0e-10
                second_derivative = algebraic.T @ (
                    assembled.stiffness @ acceleration
                )
                assert _relative_residual(
                    second_derivative,
                    assembled.stiffness @ acceleration,
                    algebraic.T @ (assembled.stiffness @ acceleration),
                ) <= 3.0e-10


def _assert_hht_constant_load_dae(
    assembled: _AssembledTransient,
    result,
    *,
    hht_alpha: float,
    rayleigh_beta: float,
) -> None:
    damping = float(rayleigh_beta) * assembled.stiffness
    reduced_load = np.asarray(
        assembled.transform.T @ assembled.load,
        dtype=float,
    ).reshape(-1)
    algebraic = assembled.basis.full_basis
    for index in range(1, len(result.times)):
        current_static = (
            damping @ result.velocities[index]
            + assembled.stiffness @ result.displacements[index]
            - assembled.load
        )
        previous_static = (
            damping @ result.velocities[index - 1]
            + assembled.stiffness @ result.displacements[index - 1]
            - assembled.load
        )
        residual = (
            assembled.mass @ result.accelerations[index]
            + (1.0 + hht_alpha) * current_static
            - hht_alpha * previous_static
        )
        reduced_residual = np.asarray(
            assembled.transform.T @ residual,
            dtype=float,
        ).reshape(-1)
        assert _relative_residual(
            reduced_residual,
            assembled.transform.T
            @ (assembled.mass @ result.accelerations[index]),
            assembled.transform.T @ current_static,
            assembled.transform.T @ previous_static,
            reduced_load,
        ) <= 4.0e-9
        algebraic_equilibrium = algebraic.T @ current_static
        assert _relative_residual(
            algebraic_equilibrium,
            algebraic.T @ (damping @ result.velocities[index]),
            algebraic.T
            @ (assembled.stiffness @ result.displacements[index]),
            algebraic.T @ assembled.load,
        ) <= 4.0e-10


def _assert_discrete_newmark_kinematics(result, config: TransientConfig) -> None:
    _alpha, beta, gamma = config.integration_parameters()
    for index in range(1, len(result.times)):
        dt = float(result.times[index] - result.times[index - 1])
        displacement_increment = (
            result.displacements[index]
            - result.displacements[index - 1]
            - dt * result.velocities[index - 1]
            - dt**2
            * (
                (0.5 - beta) * result.accelerations[index - 1]
                + beta * result.accelerations[index]
            )
        )
        velocity_increment = (
            result.velocities[index]
            - result.velocities[index - 1]
            - dt
            * (
                (1.0 - gamma) * result.accelerations[index - 1]
                + gamma * result.accelerations[index]
            )
        )
        assert _relative_residual(
            displacement_increment,
            result.displacements[index],
            result.displacements[index - 1],
            dt * result.velocities[index - 1],
        ) <= 3.0e-11
        assert _relative_residual(
            velocity_increment,
            result.velocities[index],
            result.velocities[index - 1],
            dt * result.accelerations[index],
        ) <= 3.0e-11


def test_newmark_beta_zero_has_an_explicit_rejection_policy() -> None:
    with pytest.raises(ValueError, match="beta must be positive"):
        TransientConfig(dt=1.0e-4, t_end=1.0e-3, beta=0.0)


def test_sampled_load_derivatives_are_exact_for_local_quadratics() -> None:
    times = np.asarray((0.0, 0.2, 0.7), dtype=float)
    values = [
        np.asarray((time**2 + 2.0 * time + 3.0, -4.0 * time**2 + time))
        for time in times
    ]

    np.testing.assert_allclose(
        _sampled_grid_derivative(times, values, 1, evaluation_index=0),
        (2.0, 1.0),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        _sampled_grid_derivative(times, values, 1, evaluation_index=2),
        (3.4, -4.6),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        _sampled_grid_derivative(times, values, 2, evaluation_index=1),
        (2.0, -8.0),
        rtol=0.0,
        atol=3.0e-14,
    )


def test_static_reduction_round_trips_a_sheared_mass_null_partition() -> None:
    basis = np.asarray(
        (
            (1.0, 0.0),
            (0.0, 1.0),
            (7.0, 0.5),
            (-3.0, 0.25),
            (0.2, 5.0),
            (2.0, -4.0),
        ),
        dtype=float,
    )
    orthogonal, _upper = np.linalg.qr(basis, mode="complete")
    complement = orthogonal[:, 2:]
    mass = complement @ np.diag((1.0, 2.0, 3.0, 4.0)) @ complement.T
    generator = np.asarray(
        (
            (2.0, -1.0, 0.0, 0.5, 0.0, 0.0),
            (0.0, 3.0, 1.0, 0.0, -0.5, 0.0),
            (1.0, 0.0, 4.0, 0.0, 0.0, 0.5),
            (0.0, 0.5, 0.0, 2.0, 1.0, 0.0),
            (0.0, 0.0, 0.5, 0.0, 3.0, 1.0),
            (1.0, 0.0, 0.0, 0.5, 0.0, 2.0),
        ),
        dtype=float,
    )
    stiffness = generator.T @ generator + np.eye(6)
    reduction = build_algebraic_static_reduction(
        sparse.csr_matrix(stiffness),
        sparse.csr_matrix(mass),
        sparse.csr_matrix(basis),
        dense_size_limit=512,
    )
    state = np.asarray((0.3, -0.2, 0.7, 0.1, -0.4, 0.6), dtype=float)

    physical, algebraic = reduction.split_k_orthogonal_state(state)
    reconstructed = reduction.reconstruct_k_orthogonal_state(
        physical,
        algebraic,
    )

    np.testing.assert_allclose(reconstructed, state, rtol=0.0, atol=3.0e-13)
    np.testing.assert_allclose(
        basis.T @ stiffness @ reconstructed,
        np.asarray(reduction.stiffness_algebraic @ algebraic).reshape(-1),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    stiffness_coefficient = 1.3
    mass_coefficient = 4.2
    effective = stiffness_coefficient * stiffness + mass_coefficient * mass
    reduced_rhs = np.asarray((1.0, -2.0, 0.5, 0.3, -0.4, 0.8), dtype=float)
    block_solution = np.linalg.solve(
        reduction.effective_block_matrix(
            stiffness_coefficient=stiffness_coefficient,
            mass_coefficient=mass_coefficient,
        ).toarray(),
        reduction.effective_rhs(reduced_rhs),
    )
    partition_solution = reduction.reconstruct_partition_state(
        block_solution[: reduction.physical_dimension],
        block_solution[reduction.physical_dimension :],
    )
    np.testing.assert_allclose(
        partition_solution,
        np.linalg.solve(effective, reduced_rhs),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


class _AdversarialSparseFactor:
    def __init__(self, result=None, *, fail: bool = False) -> None:
        self._result = result
        self._fail = fail

    def solve(self, _rhs):
        if self._fail:
            raise RuntimeError("backend-private-detail-must-not-leak")
        return self._result


def _small_static_reduction():
    return build_algebraic_static_reduction(
        sparse.diags((2.0, 3.0), format="csr"),
        sparse.diags((0.0, 1.0), format="csr"),
        sparse.csr_matrix(np.asarray(((1.0,), (0.0,)), dtype=float)),
        dense_size_limit=512,
    )


@pytest.mark.parametrize(
    ("factor_attribute", "operation", "message"),
    (
        (
            "algebraic_factor",
            lambda reduction: reduction.solve_algebraic(np.asarray((1.0,))),
            "algebraic stiffness solve failed",
        ),
        (
            "pivot_factor",
            lambda reduction: reduction.algebraic_partition_coordinate(
                np.asarray((1.0, 2.0))
            ),
            "algebraic pivot solve failed",
        ),
    ),
)
def test_static_reduction_wraps_backend_factor_failures(
    factor_attribute: str,
    operation,
    message: str,
) -> None:
    reduction = _small_static_reduction()
    setattr(reduction, factor_attribute, _AdversarialSparseFactor(fail=True))

    with pytest.raises(AlgebraicDynamicsError, match=message) as caught:
        operation(reduction)

    assert "backend-private-detail" not in str(caught.value)


@pytest.mark.parametrize(
    ("factor_attribute", "operation", "message"),
    (
        (
            "algebraic_factor",
            lambda reduction: reduction.solve_algebraic(np.asarray((1.0,))),
            "algebraic stiffness solve returned an invalid result",
        ),
        (
            "pivot_factor",
            lambda reduction: reduction.algebraic_partition_coordinate(
                np.asarray((1.0, 2.0))
            ),
            "algebraic pivot solve returned an invalid result",
        ),
    ),
)
def test_static_reduction_rejects_invalid_backend_factor_results(
    factor_attribute: str,
    operation,
    message: str,
) -> None:
    reduction = _small_static_reduction()
    setattr(
        reduction,
        factor_attribute,
        _AdversarialSparseFactor(np.asarray((float("nan"),))),
    )

    with pytest.raises(AlgebraicDynamicsError, match=message):
        operation(reduction)


def test_static_reduction_uses_sparse_matching_above_the_bounded_dense_limit() -> None:
    size = 520
    pivot_rows = np.asarray((0, 17, 511), dtype=np.intp)
    basis = sparse.csc_matrix(
        (
            np.ones(3, dtype=float),
            (pivot_rows, np.arange(3, dtype=np.intp)),
        ),
        shape=(size, 3),
    )
    mass_diagonal = np.ones(size, dtype=float)
    mass_diagonal[pivot_rows] = 0.0
    reduction = build_algebraic_static_reduction(
        sparse.diags(np.linspace(1.0, 2.0, size), format="csr"),
        sparse.diags(mass_diagonal, format="csr"),
        basis,
        dense_size_limit=512,
    )

    assert reduction.diagnostics["pivot_row_method"] == (
        "sparse_maximum_bipartite_matching"
    )
    np.testing.assert_array_equal(reduction.pivot_rows, pivot_rows)


def test_static_reduction_rejects_excessive_pivot_coordinate_shear() -> None:
    sheared_basis = sparse.csc_matrix(
        np.asarray(
            (
                (1.0, 1.0),
                (0.0, 3.0e-6),
                (0.0, 0.0),
                (0.0, 0.0),
            ),
            dtype=float,
        )
    )

    with pytest.raises(AlgebraicDynamicsError, match="coordinate shear"):
        build_algebraic_static_reduction(
            sparse.eye(4, format="csr"),
            sparse.diags((0.0, 0.0, 1.0, 1.0), format="csr"),
            sheared_basis,
            dense_size_limit=512,
        )


def test_static_reduction_rejects_excessive_k_orthogonal_coordinate_shear() -> None:
    # The coupling norm itself is below the frozen 256 limit, but the
    # forward/inverse triangular coordinate maps have condition ~402.
    static_shear = 20.0
    stiffness = sparse.csr_matrix(
        np.asarray(
            (
                (1.0, static_shear),
                (static_shear, static_shear**2 + 1.0),
            ),
            dtype=float,
        )
    )

    with pytest.raises(AlgebraicDynamicsError, match="static coordinate shear"):
        build_algebraic_static_reduction(
            stiffness,
            sparse.diags((0.0, 1.0), format="csr"),
            sparse.csr_matrix(np.asarray(((1.0,), (0.0,)), dtype=float)),
            dense_size_limit=512,
        )


def test_static_reduction_rejects_excessive_composed_coordinate_shear() -> None:
    root_half = float(1.0 / np.sqrt(2.0))
    basis = sparse.csr_matrix(
        np.asarray(((root_half,), (root_half,)), dtype=float)
    )
    mass = sparse.csr_matrix(
        np.asarray(((1.0, -1.0), (-1.0, 1.0)), dtype=float)
    )
    stiffness = sparse.csr_matrix(
        np.asarray(
            (
                (185.57359313, -204.78679656),
                (-204.78679656, 226.0),
            ),
            dtype=float,
        )
    )

    # Partition and K-static map conditions are each below 256, while their
    # certified composition exceeds it.
    with pytest.raises(AlgebraicDynamicsError, match="composed coordinate shear"):
        build_algebraic_static_reduction(
            stiffness,
            mass,
            basis,
            dense_size_limit=512,
        )


def test_large_full_partition_shear_is_rejected_when_pivot_alone_is_bounded() -> None:
    nullity = 513
    physical_dimension = nullity + 1
    size = nullity + physical_dimension
    pivot_scale = 0.01
    shared_scale = float(np.sqrt(1.0 - pivot_scale**2))
    basis = sparse.csc_matrix(
        (
            np.concatenate(
                (
                    np.full(nullity, pivot_scale, dtype=float),
                    np.full(nullity, shared_scale, dtype=float),
                )
            ),
            (
                np.concatenate(
                    (
                        np.arange(nullity, dtype=np.intp),
                        np.full(nullity, nullity, dtype=np.intp),
                    )
                ),
                np.concatenate(
                    (
                        np.arange(nullity, dtype=np.intp),
                        np.arange(nullity, dtype=np.intp),
                    )
                ),
            ),
        ),
        shape=(size, nullity),
    )
    physical_block = basis[nullity:, :].tocsc()
    coupling = physical_block * (1.0 / pivot_scale)
    annihilator = sparse.hstack(
        (-coupling, sparse.eye(physical_dimension, format="csc")),
        format="csc",
    )
    mass = (annihilator.T @ annihilator).tocsr()

    with pytest.raises(AlgebraicDynamicsError, match="coordinate shear"):
        build_algebraic_static_reduction(
            sparse.eye(size, format="csr"),
            mass,
            basis,
            dense_size_limit=512,
        )


@pytest.mark.parametrize("malformed", ("stiffness", "mass"))
def test_static_reduction_rejects_nonsymmetric_descriptor_inputs(
    malformed: str,
) -> None:
    stiffness = sparse.diags((2.0, 3.0), format="lil")
    mass = sparse.diags((0.0, 1.0), format="lil")
    target = stiffness if malformed == "stiffness" else mass
    target[0, 1] = 1.0e-6

    with pytest.raises(
        AlgebraicDynamicsError,
        match=rf"descriptor {malformed} matrix is not symmetric",
    ):
        build_algebraic_static_reduction(
            stiffness.tocsr(),
            mass.tocsr(),
            sparse.csr_matrix(np.asarray(((1.0,), (0.0,)), dtype=float)),
            dense_size_limit=512,
        )


def test_static_reduction_wraps_construction_pivot_solve_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anyfesolver.algebraic_dynamics as algebraic_module

    original = algebraic_module.sparse_linalg.splu
    call_count = 0

    def adversarial_splu(matrix, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return _AdversarialSparseFactor(fail=True)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(algebraic_module.sparse_linalg, "splu", adversarial_splu)
    with pytest.raises(
        AlgebraicDynamicsError,
        match="algebraic pivot inverse solve failed",
    ) as caught:
        _small_static_reduction()

    assert "backend-private-detail" not in str(caught.value)


def test_sparse_pivot_matching_is_invariant_to_internal_index_order() -> None:
    import anyfesolver.algebraic_dynamics as algebraic_module

    row_count = 520
    dense = np.zeros((row_count, 3), dtype=float)
    dense[0, (0, 1)] = (2.0, 1.0)
    dense[1, (1, 2)] = (2.0, 1.0)
    dense[2, (0, 2)] = (1.0, 2.0)
    canonical = sparse.csc_matrix(dense)
    reversed_storage = canonical.copy()
    for column in range(reversed_storage.shape[1]):
        start = int(reversed_storage.indptr[column])
        stop = int(reversed_storage.indptr[column + 1])
        reversed_storage.indices[start:stop] = reversed_storage.indices[
            start:stop
        ][::-1]
        reversed_storage.data[start:stop] = reversed_storage.data[start:stop][::-1]
    reversed_storage.has_sorted_indices = False

    canonical_rows, canonical_method = (
        algebraic_module._algebraic_coordinate_pivot_rows(
            canonical,
            dense_size_limit=512,
        )
    )
    reversed_rows, reversed_method = (
        algebraic_module._algebraic_coordinate_pivot_rows(
            reversed_storage,
            dense_size_limit=512,
        )
    )

    np.testing.assert_array_equal(reversed_rows, canonical_rows)
    assert reversed_method == canonical_method == (
        "sparse_maximum_bipartite_matching"
    )


def test_large_coordinate_shear_certificate_is_sparse_and_fail_closed() -> None:
    nullity = 513
    size = 2 * nullity
    basis = sparse.vstack(
        (
            sparse.eye(nullity, format="csc"),
            sparse.csc_matrix((nullity, nullity), dtype=float),
        ),
        format="csc",
    )
    mass_diagonal = np.concatenate(
        (np.zeros(nullity, dtype=float), np.ones(nullity, dtype=float))
    )

    reduction = build_algebraic_static_reduction(
        sparse.eye(size, format="csr"),
        sparse.diags(mass_diagonal, format="csr"),
        basis,
        dense_size_limit=512,
    )

    assert reduction.diagnostics["partition_coordinate_shear_method"].startswith(
        "sparse_certified_full_partition_condition:"
    )
    assert reduction.diagnostics["coordinate_shear_upper_bound"] == pytest.approx(
        1.0,
        rel=0.0,
        abs=2.0e-12,
    )


def test_descriptor_effective_certificate_rejects_an_indefinite_operator() -> None:
    with pytest.raises(
        AlgebraicDynamicsError,
        match="effective operator is not positive definite",
    ):
        certify_descriptor_effective_operator(
            sparse.diags((1.0, -1.0), format="csc"),
            dense_size_limit=512,
        )


@pytest.mark.parametrize(
    ("rayleigh_alpha", "rayleigh_beta"),
    (
        (-1.0e-4, 0.0),
        (0.0, -1.0e-4),
        (float("nan"), 0.0),
        (0.0, float("nan")),
    ),
)
def test_descriptor_transient_rejects_negative_rayleigh_damping(
    rayleigh_alpha: float,
    rayleigh_beta: float,
) -> None:
    model = _single_s3_model(coupled=True)
    with pytest.raises(AlgebraicDynamicsError, match="non-negative Rayleigh damping"):
        solve_transient_newmark(
            model,
            TransientConfig(
                dt=1.0e-4,
                t_end=1.0e-4,
                rayleigh_alpha=rayleigh_alpha,
                rayleigh_beta=rayleigh_beta,
            ),
        )


def test_descriptor_transient_rejects_a_nonfinite_pressure_history() -> None:
    model = _single_s3_model(coupled=True)
    patch = PressurePatch(
        "nonfinite",
        pressure_time=lambda _time: float("nan"),
        element_ids=[1],
    )

    with pytest.raises(AlgebraicDynamicsError, match="load is non-finite"):
        solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-4, t_end=1.0e-4),
            pressure_patches=[patch],
        )


def test_descriptor_effective_factorization_failure_never_uses_general_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anyfesolver.dynamics as dynamics_module

    model = _single_s3_model(coupled=True)
    original = dynamics_module.factorize
    calls: list[MatrixClass] = []

    def failed_effective(matrix, matrix_class, *, signature=None, **kwargs):
        made_class = MatrixClass(matrix_class)
        calls.append(made_class)
        if str(signature).startswith("transient.effective.descriptor:"):
            return FactorizationHandle(
                matrix_shape=tuple(int(value) for value in matrix.shape),
                matrix_class=made_class,
                backend_name="adversarial_failed_backend",
                ordering="none",
                signature=signature,
                factorization_time=0.0,
                status="failed",
                failure_reason="backend-private-detail-must-not-leak",
            )
        return original(
            matrix,
            made_class,
            signature=signature,
            **kwargs,
        )

    monkeypatch.setattr(dynamics_module, "factorize", failed_effective)
    with pytest.raises(
        AlgebraicDynamicsError,
        match="failed its SPD factorization",
    ) as caught:
        solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-4, t_end=1.0e-4),
        )

    assert "backend-private-detail" not in str(caught.value)
    assert MatrixClass.GENERAL not in calls
    assert calls[-1] is MatrixClass.SPD


def test_descriptor_transient_rejects_zero_areal_mass() -> None:
    model = _single_s3_model(coupled=True)
    model.materials["steel"].density = 0.0

    with pytest.raises(
        AlgebraicDynamicsError,
        match="positive areal mass and rotary inertia",
    ):
        solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-4, t_end=1.0e-4),
        )


@pytest.mark.parametrize("rayleigh_beta", (0.0, 2.0e-5))
def test_sheared_s3_transient_enforces_algebraic_dae_and_initial_acceleration(
    rayleigh_beta: float,
) -> None:
    model = _single_s3_model(coupled=True)
    load = LoadCase("constant-physical-load")
    load.add_nodal_load(3, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assembled = _assembled(model, load)

    coupling = assembled.basis.reduced_basis.T @ (
        (assembled.transform.T @ assembled.stiffness @ assembled.transform)
    )
    assert float(np.linalg.norm(coupling.toarray())) > 1.0e6

    config = TransientConfig(
        dt=5.0e-5,
        t_end=1.5e-4,
        beta=0.30,
        gamma=0.55,
        rayleigh_beta=rayleigh_beta,
    )
    result = solve_transient_newmark(model, config, base_load_case=load)

    assert result.status == "completed"
    expected_endpoint_policy = (
        DESCRIPTOR_TRANSIENT_STATIC_POLICY_ID
        if rayleigh_beta == 0.0
        else DESCRIPTOR_TRANSIENT_FIRST_ORDER_POLICY_ID
    )
    assert result.diagnostics["descriptor_transient"] is True
    assert result.diagnostics["policy_id"] == DESCRIPTOR_TRANSIENT_POLICY_ID
    assert result.diagnostics["algebraic_endpoint_policy"] == (
        expected_endpoint_policy
    )
    assert result.diagnostics["effective_stiffness_factorization"][
        "matrix_class"
    ] == "spd"
    assert result.diagnostics["declared_algebraic_element_ids"] == [1]
    assert result.result_case["metadata"]["descriptor_transient_provenance"] == {
        "algebraic_endpoint_policy": expected_endpoint_policy,
        "declared_algebraic_element_ids": [1],
        "declared_algebraic_formulations": [
            {
                "algebraic_coordinate_policy": "S3_NODAL_DRILL_ZERO_INERTIA_V1",
                "element_id": 1,
                "formulation_id": "E4_PL_QUALIFIED_S3_COMPANION_V1",
            }
        ],
        "disposition": "ACTIVE_COMPATIBLE_ALGEBRAIC_REDUCTION",
        "policy_id": DESCRIPTOR_TRANSIENT_POLICY_ID,
    }
    assert float(np.linalg.norm(result.accelerations[0])) > 0.0
    _assert_constant_load_dae(
        assembled,
        result,
        rayleigh_beta=rayleigh_beta,
    )
    _assert_discrete_newmark_kinematics(result, config)


@pytest.mark.parametrize("rayleigh_beta", (0.0, 2.0e-5))
def test_hht_descriptor_step_preserves_weighted_dae_and_endpoint_equilibrium(
    rayleigh_beta: float,
) -> None:
    model = _single_s3_model(coupled=True)
    load = LoadCase("constant-hht-load")
    load.add_nodal_load(3, [1.0, -0.4, 0.2, 0.0, 0.0, 0.0])
    assembled = _assembled(model, load)
    hht_alpha = -0.12

    config = TransientConfig(
        dt=5.0e-5,
        t_end=2.0e-4,
        hht_alpha=hht_alpha,
        rayleigh_beta=rayleigh_beta,
    )
    result = solve_transient_newmark(
        model,
        config,
        base_load_case=load,
    )

    assert result.status == "completed"
    assert result.diagnostics["method"] == "hht_alpha"
    _assert_hht_constant_load_dae(
        assembled,
        result,
        hht_alpha=hht_alpha,
        rayleigh_beta=rayleigh_beta,
    )
    _assert_discrete_newmark_kinematics(result, config)


@pytest.mark.parametrize(
    ("hht_alpha", "beta", "gamma"),
    ((0.0, 0.30, 0.55), (-0.12, 0.25, 0.50)),
)
def test_first_order_direct_drill_damping_preserves_discrete_kinematics(
    hht_alpha: float,
    beta: float,
    gamma: float,
) -> None:
    model = _single_s3_model(coupled=False)
    load = LoadCase("first-order-direct-drill")
    load.add_nodal_load(3, moments=np.asarray((0.0, 0.0, 23.0)))
    assembled = _assembled(model, load)
    config = TransientConfig(
        dt=4.0e-5,
        t_end=1.6e-4,
        beta=beta,
        gamma=gamma,
        hht_alpha=hht_alpha,
        rayleigh_beta=1.0e-4,
    )

    result = solve_transient_newmark(model, config, base_load_case=load)

    assert result.status == "completed"
    _assert_discrete_newmark_kinematics(result, config)
    if hht_alpha == 0.0:
        _assert_constant_load_dae(
            assembled,
            result,
            rayleigh_beta=config.rayleigh_beta,
        )
    else:
        _assert_hht_constant_load_dae(
            assembled,
            result,
            hht_alpha=hht_alpha,
            rayleigh_beta=config.rayleigh_beta,
        )


def test_descriptor_hht_ramped_pressure_retains_impulse_and_single_factorization() -> None:
    model = _single_s3_model(coupled=True)
    patch = PressurePatch(
        "triangular-ramp",
        pressure_time=((0.0, 0.0), (1.0e-4, 1000.0), (2.0e-4, 0.0)),
        element_ids=[1],
    )
    config = TransientConfig(
        dt=5.0e-5,
        t_end=2.0e-4,
        hht_alpha=-0.1,
        rayleigh_alpha=1.0e-3,
        rayleigh_beta=2.0e-5,
    )

    result = solve_transient_newmark(
        model,
        config,
        pressure_patches=[patch],
    )

    assert result.status == "completed"
    assert result.diagnostics["factorization_count"] == 1
    assert result.diagnostics["max_dae_relative_residual"] < 3.0e-9
    assert result.diagnostics["max_algebraic_relative_residual"] < 3.0e-10
    assert result.force_impulse[2] == pytest.approx(0.045, rel=2.0e-13)
    assert result.diagnostics["pressure_patches"][0][
        "num_selected_elements"
    ] == 1
    _assert_discrete_newmark_kinematics(result, config)


def test_direct_drill_moment_is_retained_as_algebraic_work_and_impulse() -> None:
    moment = 123.0
    model = _single_s3_model(coupled=False)
    load = LoadCase("direct-drill-moment")
    load.add_nodal_load(3, moments=np.asarray((0.0, 0.0, moment)))
    assembled = _assembled(model, load)
    algebraic = assembled.basis.reduced_basis.toarray()
    reduced_stiffness = (
        assembled.transform.T @ assembled.stiffness @ assembled.transform
    ).toarray()
    reduced_load = np.asarray(assembled.transform.T @ assembled.load, dtype=float)
    algebraic_stiffness = algebraic.T @ reduced_stiffness @ algebraic
    algebraic_load = algebraic.T @ reduced_load
    expected_reduced = algebraic @ np.linalg.solve(
        algebraic_stiffness, algebraic_load
    )
    expected = np.asarray(
        assembled.transform @ expected_reduced + assembled.affine_offset,
        dtype=float,
    )

    config = TransientConfig(dt=1.0e-4, t_end=3.0e-4)
    result = solve_transient_newmark(model, config, base_load_case=load)

    assert result.status == "completed"
    assert result.diagnostics["factorization_count"] == 1
    np.testing.assert_allclose(
        result.displacements,
        np.broadcast_to(expected, result.displacements.shape),
        rtol=2.0e-12,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(result.velocities, 0.0, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(result.accelerations, 0.0, rtol=0.0, atol=2.0e-8)
    external_work = float(assembled.load @ expected)
    internal_work = float(expected @ (assembled.stiffness @ expected))
    assert external_work == pytest.approx(internal_work, rel=2.0e-12, abs=2.0e-14)
    assert result.diagnostics["strain_energy"][0] == pytest.approx(
        0.5 * internal_work, rel=2.0e-12, abs=2.0e-14
    )
    assert result.diagnostics["algebraic_strain_energy"][0] == pytest.approx(
        0.5 * internal_work, rel=2.0e-12, abs=2.0e-14
    )
    assert result.diagnostics[
        "algebraic_external_work_after_initial_consistency"
    ] == pytest.approx(0.0, abs=2.0e-14)
    assert result.diagnostics["max_algebraic_generalized_load_norm"] > 0.0
    assert result.load_impulse[model.mesh.get_node(3).dofs[5]] == pytest.approx(
        moment * config.t_end, rel=2.0e-14
    )
    assert result.moment_impulse[2] == pytest.approx(
        moment * config.t_end, rel=2.0e-14
    )
    _assert_constant_load_dae(assembled, result)


@pytest.mark.parametrize("rayleigh_beta", (0.0, 1.0e-4))
def test_initial_state_projection_preserves_mass_state_and_dae_consistency(
    rayleigh_beta: float,
) -> None:
    model = _single_s3_model(coupled=True)
    load = LoadCase("zero")
    assembled = _assembled(model, load)
    size = model.mesh.dof_manager.total_dofs
    initial_displacement = np.zeros(size, dtype=float)
    initial_velocity = np.zeros(size, dtype=float)
    node2 = model.mesh.get_node(2)
    node3 = model.mesh.get_node(3)
    initial_displacement[node3.dofs[0]] = 2.0e-7
    initial_displacement[node3.dofs[5]] = 3.0e-4
    initial_velocity[node3.dofs[0]] = -4.0e-3
    initial_velocity[node2.dofs[5]] = 5.0e-4

    result = solve_transient_newmark(
        model,
        TransientConfig(
            dt=1.0e-4,
            t_end=0.0,
            rayleigh_beta=rayleigh_beta,
            initial_displacement=initial_displacement,
            initial_velocity=initial_velocity,
        ),
        base_load_case=load,
    )

    assert result.status == "completed"
    displacement_correction = result.displacements[0] - initial_displacement
    velocity_correction = result.velocities[0] - initial_velocity
    assert _relative_residual(
        assembled.mass @ displacement_correction,
        assembled.mass @ initial_displacement,
    ) <= 3.0e-12
    assert _relative_residual(
        assembled.mass @ velocity_correction,
        assembled.mass @ initial_velocity,
    ) <= 3.0e-12
    if rayleigh_beta > 0.0:
        # Stiffness-proportional damping makes the algebraic row first order:
        # q is the supplied state and the massless velocity is projected.
        np.testing.assert_array_equal(result.displacements[0], initial_displacement)
    _assert_constant_load_dae(
        assembled,
        result,
        rayleigh_beta=rayleigh_beta,
    )


def test_constrained_direct_drill_moment_is_preserved_as_a_reaction() -> None:
    moment = 37.0
    model = _single_s3_model(coupled=False)
    model.add_boundary_condition(
        BoundaryCondition("fix-node-3-drill", [3], {"rz": 0.0})
    )
    load = LoadCase("constrained-direct-drill")
    load.add_nodal_load(3, moments=np.asarray((0.0, 0.0, moment)))
    assembled = _assembled(model, load)
    assert assembled.basis.reduced_basis.shape[1] == 0

    result = solve_transient_newmark(
        model,
        TransientConfig(dt=1.0e-4, t_end=2.0e-4),
        base_load_case=load,
    )

    drill_dof = int(model.mesh.get_node(3).dofs[5])
    assert result.diagnostics["descriptor_transient"] is False
    assert result.diagnostics["algebraic_endpoint_policy"] == (
        DESCRIPTOR_TRANSIENT_CONSTRAINED_POLICY_ID
    )
    assert result.diagnostics["declared_algebraic_mass_certificate"][
        "mass_basis"
    ]["compatible_global_nullity"] == 0
    assert result.diagnostics["initial_mass_factorization"]["matrix_class"] == "spd"
    assert result.diagnostics["effective_stiffness_factorization"][
        "matrix_class"
    ] == "spd"
    assert len(result.diagnostics["effective_operator_certificates"]) == 1
    assert result.result_case["metadata"]["descriptor_transient_provenance"] == {
        "algebraic_endpoint_policy": DESCRIPTOR_TRANSIENT_CONSTRAINED_POLICY_ID,
        "declared_algebraic_element_ids": [1],
        "declared_algebraic_formulations": [
            {
                "algebraic_coordinate_policy": "S3_NODAL_DRILL_ZERO_INERTIA_V1",
                "element_id": 1,
                "formulation_id": "E4_PL_QUALIFIED_S3_COMPANION_V1",
            }
        ],
        "disposition": (
            "COMPATIBLE_ALGEBRAIC_NULLITY_ZERO_ORDINARY_REDUCED_SYSTEM"
        ),
        "policy_id": DESCRIPTOR_TRANSIENT_POLICY_ID,
    }
    np.testing.assert_array_equal(result.displacements[:, drill_dof], 0.0)
    for displacement, velocity, acceleration in zip(
        result.displacements, result.velocities, result.accelerations
    ):
        residual = (
            assembled.mass @ acceleration
            + assembled.stiffness @ displacement
            - assembled.load
        )
        assert float(residual[drill_dof]) == pytest.approx(-moment, rel=2.0e-12)


@pytest.mark.parametrize(
    ("failed_signature", "message"),
    (
        (
            "transient.initial_descriptor_constrained_mass",
            "descriptor constrained physical mass factorization failed",
        ),
        (
            "transient.effective.descriptor_constrained:",
            "descriptor effective operator failed its SPD factorization",
        ),
    ),
)
def test_constrained_s3_factor_failures_are_canonical_and_never_fall_back(
    monkeypatch: pytest.MonkeyPatch,
    failed_signature: str,
    message: str,
) -> None:
    import anyfesolver.dynamics as dynamics_module

    model = _single_s3_model(coupled=False)
    model.add_boundary_condition(
        BoundaryCondition("fix-node-3-drill", [3], {"rz": 0.0})
    )
    original = dynamics_module.factorize
    calls: list[MatrixClass] = []

    def failed_guarded(matrix, matrix_class, *, signature=None, **kwargs):
        made_class = MatrixClass(matrix_class)
        calls.append(made_class)
        if str(signature).startswith(failed_signature):
            return FactorizationHandle(
                matrix_shape=tuple(int(value) for value in matrix.shape),
                matrix_class=made_class,
                backend_name="adversarial_failed_backend",
                ordering="none",
                signature=signature,
                factorization_time=0.0,
                status="failed",
                failure_reason="backend-private-detail-must-not-leak",
            )
        return original(
            matrix,
            made_class,
            signature=signature,
            **kwargs,
        )

    monkeypatch.setattr(dynamics_module, "factorize", failed_guarded)
    with pytest.raises(AlgebraicDynamicsError, match=message) as caught:
        solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-4, t_end=1.0e-4),
        )

    assert "backend-private-detail" not in str(caught.value)
    assert MatrixClass.GENERAL not in calls
    assert all(call is MatrixClass.SPD for call in calls)


def test_mixed_q4_s3_transient_eliminates_only_the_unshared_s3_drill() -> None:
    model = _mixed_q4_s3_model()
    load = LoadCase("mixed-physical-load")
    load.add_nodal_load(5, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assembled = _assembled(model, load)

    assert assembled.basis.diagnostics["candidate_node_ids"] == [5]
    assert assembled.basis.diagnostics["mass_removed_node_ids"] == [2, 3]
    assert assembled.basis.reduced_basis.shape[1] == 1

    config = TransientConfig(dt=5.0e-5, t_end=1.5e-4, beta=0.28, gamma=0.53)
    first = solve_transient_newmark(model, config, base_load_case=load)

    repeated_model = _mixed_q4_s3_model()
    repeated_load = LoadCase("mixed-physical-load")
    repeated_load.add_nodal_load(5, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    second = solve_transient_newmark(
        repeated_model,
        config,
        base_load_case=repeated_load,
    )

    assert first.status == second.status == "completed"
    _assert_constant_load_dae(assembled, first)
    np.testing.assert_array_equal(first.times, second.times)
    np.testing.assert_array_equal(first.displacements, second.displacements)
    np.testing.assert_array_equal(first.velocities, second.velocities)
    np.testing.assert_array_equal(first.accelerations, second.accelerations)


def test_descriptor_selected_history_and_session_reuse_match_full_history() -> None:
    model = _single_s3_model(coupled=True)
    load = LoadCase("selected-descriptor-load")
    load.add_nodal_load(3, [0.5, -0.2, 0.1, 0.0, 0.0, 0.0])
    common = {
        "dt": 5.0e-5,
        "t_end": 2.2e-4,
        "rayleigh_beta": 2.0e-5,
        "output_nodes": [3],
    }
    full = solve_transient_newmark(
        model,
        TransientConfig(**common),
        base_load_case=load,
    )
    selected_config = TransientConfig(
        **common,
        recovery=RecoveryConfig(
            node_ids=[3],
            include_stresses=False,
            history_mode="selected",
            store_full_histories=False,
        ),
    )
    with AnalysisSession(model) as session:
        selected = solve_transient_newmark(
            model,
            selected_config,
            base_load_case=load,
            session=session,
        )
        repeated = solve_transient_newmark(
            model,
            selected_config,
            base_load_case=load,
            session=session,
        )

    node_dofs = np.asarray(model.mesh.get_node(3).dofs, dtype=np.intp)
    np.testing.assert_array_equal(selected.times, full.times)
    np.testing.assert_allclose(
        selected.displacements,
        full.displacements[:, node_dofs],
        rtol=2.0e-12,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        selected.velocities,
        full.velocities[:, node_dofs],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        selected.accelerations,
        full.accelerations[:, node_dofs],
        rtol=2.0e-12,
        atol=2.0e-8,
    )
    np.testing.assert_array_equal(repeated.displacements, selected.displacements)
    np.testing.assert_array_equal(repeated.velocities, selected.velocities)
    np.testing.assert_array_equal(repeated.accelerations, selected.accelerations)
    assert selected.diagnostics["session_output_plans_active"] is True
    assert repeated.diagnostics["session_output_plans_active"] is True
    assert selected.diagnostics["descriptor_transient"] is True


def test_descriptor_near_multiple_final_step_uses_exact_effective_coefficients() -> None:
    model = _single_s3_model(coupled=True)
    load = LoadCase("near-multiple-final-step")
    load.add_nodal_load(3, [0.7, -0.1, 0.2, 0.0, 0.0, 0.0])
    assembled = _assembled(model, load)
    config = TransientConfig(
        dt=1.0e-4,
        t_end=1.99995e-4,
        rayleigh_beta=2.0e-5,
    )

    result = solve_transient_newmark(model, config, base_load_case=load)

    assert result.status == "completed"
    assert result.diagnostics["factorization_count"] == 2
    _assert_discrete_newmark_kinematics(result, config)
    _assert_constant_load_dae(
        assembled,
        result,
        rayleigh_beta=config.rayleigh_beta,
    )


def test_mpc_intersection_retains_one_tied_drill_and_its_direct_work() -> None:
    moment = 17.0
    model = _mpc_tied_s3_model()
    load = LoadCase("slave-direct-drill")
    load.add_nodal_load(6, moments=np.asarray((0.0, 0.0, moment)))
    assembled = _assembled(model, load)

    assert assembled.basis.reduced_basis.shape[1] == 1
    assert assembled.basis.diagnostics["compatible_global_nullity"] == 1
    result = solve_transient_newmark(
        model,
        TransientConfig(dt=1.0e-4, t_end=2.0e-4),
        base_load_case=load,
    )

    assert result.status == "completed"
    rz3 = int(model.mesh.get_node(3).dofs[5])
    rz6 = int(model.mesh.get_node(6).dofs[5])
    np.testing.assert_allclose(
        result.displacements[:, rz6],
        result.displacements[:, rz3],
        rtol=0.0,
        atol=2.0e-14,
    )
    assert abs(float(result.displacements[0, rz6])) > 0.0
    _assert_constant_load_dae(assembled, result)
