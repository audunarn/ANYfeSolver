"""Focused checks for the modal rigid-body classifier.

The rigid-body correlation must measure the fraction of modal mass captured
by the whole rigid subspace, so it is invariant under orthogonal re-mixing of
the rigid basis, under mass rescaling, and under arbitrary scaling of the
eigensolver-returned vectors, while elastic modes must stay excluded.
"""

from __future__ import annotations

import numpy as np
import pytest

import anyfesolver.modal as modal_module
from anyfesolver import (
    BoundaryCondition,
    FEModel,
    FixedSupport,
    solve_free_vibration,
)
from anyfesolver.elements import BeamElement


def _free_free_beam_model(density: float = 2.0) -> FEModel:
    model = FEModel("modal_rigid_classification_beam")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=density)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(
        1,
        BeamElement(
            1,
            [1, 2],
            "steel",
            {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
        ),
    )
    return model


def _orthogonal_mix(width: int) -> np.ndarray:
    rng = np.random.default_rng(20261007)
    mixed, _ = np.linalg.qr(rng.standard_normal((width, width)))
    if np.linalg.det(mixed) < 0.0:
        mixed[:, 0] *= -1.0
    return mixed


def _assert_matching_classifications(
    baseline: object, variant: object, *, correlation_rel: float = 1.0e-9
) -> None:
    assert variant.solver_status == baseline.solver_status
    assert (
        variant.diagnostics["num_rigid_body_modes"]
        == baseline.diagnostics["num_rigid_body_modes"]
    )
    for base_mode, other_mode in zip(baseline.modes, variant.modes):
        assert other_mode.is_rigid_body == base_mode.is_rigid_body
        assert other_mode.rigid_body_correlation == pytest.approx(
            base_mode.rigid_body_correlation, rel=correlation_rel, abs=1.0e-12
        )


def test_free_free_beam_retains_six_rigid_modes_with_unit_subspace_correlation() -> None:
    result = solve_free_vibration(_free_free_beam_model(), num_modes=6)

    assert result.solver_status == "ok"
    assert result.diagnostics["num_rigid_body_modes"] == 6
    assert result.nullspace_info["rank"] == 6
    assert np.max(result.frequencies_hz[:6]) < 1.0e-5
    for mode in result.modes:
        assert mode.is_rigid_body
        assert mode.rigid_body_correlation > 0.99


def test_rigid_classification_invariant_under_rigid_basis_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = solve_free_vibration(_free_free_beam_model(), num_modes=12)
    assert baseline.diagnostics["num_rigid_body_modes"] == 6

    original_builder = modal_module.build_reduced_rigid_body_modes
    observed_deviations: list[float] = []

    def rotated_builder(model, independent_dofs, total_dofs, transformation=None):
        Q, info = original_builder(
            model, independent_dofs, total_dofs, transformation=transformation
        )
        if Q.shape[1]:
            mixed = _orthogonal_mix(Q.shape[1])
            np.testing.assert_allclose(mixed.T @ mixed, np.eye(mixed.shape[0]), atol=1.0e-12)
            rotated = np.asarray(Q @ mixed, dtype=float)
            observed_deviations.append(float(np.max(np.abs(rotated - Q))))
            Q = rotated
        return Q, info

    monkeypatch.setattr(
        modal_module, "build_reduced_rigid_body_modes", rotated_builder
    )
    rotated = solve_free_vibration(_free_free_beam_model(), num_modes=12)

    assert observed_deviations and max(observed_deviations) > 0.1
    np.testing.assert_allclose(
        rotated.frequencies_hz, baseline.frequencies_hz, rtol=1.0e-9, atol=1.0e-12
    )
    _assert_matching_classifications(baseline, rotated)
    assert all(mode.rigid_body_correlation > 0.99 for mode in rotated.modes[:6])
    assert all(not mode.is_rigid_body for mode in rotated.modes[6:])


def test_rigid_classification_invariant_under_mass_rescaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Isolate the classifier's mass scaling from LAPACK's choice of basis in
    # repeated eigenspaces. Independent solves can give different roundoff in
    # nominally zero elastic/rigid correlations even for equivalent pencils.
    original_dense = modal_module._dense_eigensolve
    reference: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def consistently_scaled_eigensolve(K, M):
        if not reference:
            values, vectors = original_dense(K, M)
            reference.append((K.toarray(), M.toarray(), values.copy(), vectors.copy()))
            return values, vectors
        base_K, base_M, values, vectors = reference[0]
        np.testing.assert_array_equal(K.toarray(), base_K)
        np.testing.assert_allclose(M.toarray(), 100.0 * base_M, rtol=1.0e-12, atol=0.0)
        return values / 100.0, vectors / 10.0

    monkeypatch.setattr(modal_module, "_dense_eigensolve", consistently_scaled_eigensolve)
    baseline = solve_free_vibration(_free_free_beam_model(density=2.0), num_modes=12)
    rescaled = solve_free_vibration(
        _free_free_beam_model(density=200.0), num_modes=12
    )

    assert baseline.diagnostics["num_rigid_body_modes"] == 6
    _assert_matching_classifications(baseline, rescaled)
    # Elastic frequencies scale as 1/sqrt(density); rigid ones stay near zero.
    for base_mode, scaled_mode in zip(baseline.modes[6:], rescaled.modes[6:]):
        assert scaled_mode.frequency_hz == pytest.approx(
            base_mode.frequency_hz / 10.0, rel=1.0e-6
        )


def test_rigid_classification_invariant_under_eigensolver_vector_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = solve_free_vibration(_free_free_beam_model(), num_modes=12)

    original_dense = modal_module._dense_eigensolve
    observed_factors: list[float] = []

    def scaled_dense_eigensolve(K, M):
        values, vectors = original_dense(K, M)
        if vectors.ndim == 2 and vectors.shape[1]:
            factors = np.geomspace(1.0e-1, 1.0e2, vectors.shape[1])
            observed_factors.extend(float(factor) for factor in factors)
            vectors = vectors * factors
        return values, vectors

    monkeypatch.setattr(modal_module, "_dense_eigensolve", scaled_dense_eigensolve)
    rescaled = solve_free_vibration(_free_free_beam_model(), num_modes=12)

    assert observed_factors and min(observed_factors) < 0.5
    assert max(observed_factors) > 10.0
    np.testing.assert_allclose(
        rescaled.frequencies_hz, baseline.frequencies_hz, rtol=1.0e-9, atol=1.0e-12
    )
    _assert_matching_classifications(baseline, rescaled)


def test_elastic_beam_modes_are_excluded_from_rigid_classification() -> None:
    result = solve_free_vibration(_free_free_beam_model(), num_modes=12)

    assert result.solver_status == "ok"
    assert result.diagnostics["num_rigid_body_modes"] == 6
    for mode in result.modes[:6]:
        assert mode.is_rigid_body
    for mode in result.modes[6:]:
        assert not mode.is_rigid_body
        assert mode.rigid_body_correlation < 0.1
        assert mode.frequency_hz > 1.0e-6


def test_pinned_beam_supported_component_classifies_via_frequency_branch() -> None:
    model = _free_free_beam_model()
    model.add_boundary_condition(
        BoundaryCondition("pin-node-1", [1], {"ux": 0.0, "uy": 0.0, "uz": 0.0})
    )

    result = solve_free_vibration(model, num_modes=9)

    assert result.solver_status == "ok"
    # A component with any constrained DOF contributes no rigid candidates, so
    # the correlation branch is inactive (correlation exactly zero) and the
    # three near-zero rotational modes are classified through the frequency
    # branch alone; elastic modes must stay excluded.
    assert result.nullspace_info["rank"] == 0
    assert result.diagnostics["num_rigid_body_modes"] == 3
    for mode in result.modes[:3]:
        assert mode.is_rigid_body
        assert mode.rigid_body_correlation == 0.0
        assert mode.frequency_hz <= 1.0e-6
    for mode in result.modes[3:]:
        assert not mode.is_rigid_body
        assert mode.rigid_body_correlation == 0.0
        assert mode.frequency_hz > 1.0e-6


def test_clamped_cantilever_has_empty_rigid_basis() -> None:
    model = _free_free_beam_model()
    model.add_boundary_condition(FixedSupport("clamp-node-1", [1]))

    result = solve_free_vibration(model, num_modes=6)

    assert result.solver_status == "ok"
    assert result.nullspace_info["rank"] == 0
    assert result.diagnostics["num_rigid_body_modes"] == 0
    for mode in result.modes:
        assert mode.rigid_body_correlation == 0.0
        assert not mode.is_rigid_body
        assert mode.frequency_hz > 1.0e-6
