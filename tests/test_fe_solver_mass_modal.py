"""Batch 06 mass-property and modal-analysis checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import anyfesolver.modal as modal_module

from anyfesolver import (
    BoundaryCondition,
    FEModel,
    FactorizationCache,
    FixedSupport,
    calculate_mass_properties,
    generate_simple_panel_mesh,
    solve_free_vibration,
)
from anyfesolver.boundary import LoadCase, LoadCombination
from anyfesolver.elements import BeamElement


def _axial_bar_model() -> FEModel:
    model = FEModel("axial_bar")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    return model


def test_beam_mass_properties_match_integrated_line_mass_and_assembled_translations() -> None:
    model = _axial_bar_model()

    props = calculate_mass_properties(model, reference_point=(0.0, 0.0, 0.0))

    assert props.total_mass == pytest.approx(2.0)
    np.testing.assert_allclose(props.center_of_mass, [0.5, 0.0, 0.0], atol=1.0e-14)
    assert props.assembled_translation_masses["x"] == pytest.approx(props.total_mass)
    assert props.assembled_translation_masses["y"] == pytest.approx(props.total_mass)
    assert props.assembled_translation_masses["z"] == pytest.approx(props.total_mass)
    assert props.rigid_body_mass_matrix.shape == (6, 6)


def test_point_mass_contributes_to_scalar_mass_center_and_origin_inertia() -> None:
    model = _axial_bar_model()
    bare = calculate_mass_properties(model, reference_point=(0.0, 0.0, 0.0))

    model.add_point_mass(2, 3.0)
    loaded = calculate_mass_properties(model, reference_point=(0.0, 0.0, 0.0))

    assert loaded.total_mass == pytest.approx(5.0)
    np.testing.assert_allclose(loaded.center_of_mass, [0.8, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(
        loaded.inertia_tensor_origin - bare.inertia_tensor_origin,
        np.diag([0.0, 3.0, 3.0]),
        atol=1.0e-14,
    )
    for axis in ("x", "y", "z"):
        assert loaded.assembled_translation_masses[axis] - bare.assembled_translation_masses[axis] == pytest.approx(3.0)
    assert loaded.num_mass_points == bare.num_mass_points + 1


@pytest.mark.parametrize("mass", [-1.0, np.inf, -np.inf, np.nan])
def test_add_point_mass_rejects_invalid_mass_values(mass: float) -> None:
    model = _axial_bar_model()

    with pytest.raises(ValueError, match="finite and non-negative"):
        model.add_point_mass(1, mass)

    assert model.mesh.point_masses == {}


def test_add_point_mass_rejects_missing_node() -> None:
    model = _axial_bar_model()

    with pytest.raises(ValueError, match="missing node 99"):
        model.add_point_mass(99, 1.0)


def test_mass_properties_rejects_corrupt_direct_point_mass_entries() -> None:
    model = _axial_bar_model()
    model.mesh.point_masses[1] = np.nan

    with pytest.raises(ValueError, match="node 1 must be finite and non-negative"):
        calculate_mass_properties(model)


def test_load_combination_forwards_material_density_for_gravity() -> None:
    model = _axial_bar_model()
    model.materials["steel"].density = 7.0
    gravity = LoadCase("dead_load")
    gravity.set_gravity(0.0, 0.0, -3.0)
    combination = LoadCombination("uls", {"dead_load": 2.5})

    combined = combination.get_combined_load_vector(
        [gravity],
        model.mesh,
        model.mesh.dof_manager,
        material_getter=model.get_material,
    )

    z_dofs = [node.dofs[2] for node in model.mesh.nodes.values()]
    assert np.sum(combined[z_dofs]) == pytest.approx(2.5 * 7.0 * 1.0 * 1.0 * -3.0)
    np.testing.assert_allclose(np.delete(combined, z_dofs), 0.0, atol=1.0e-14)


def test_shell_mass_properties_match_density_area_thickness() -> None:
    model = generate_simple_panel_mesh(2.0, 1.0, 0.02, num_divisions_x=1, num_divisions_y=1)
    model.materials["steel"].density = 7850.0

    props = calculate_mass_properties(model)

    assert props.total_mass == pytest.approx(2.0 * 1.0 * 0.02 * 7850.0)
    np.testing.assert_allclose(props.center_of_mass, [1.0, 0.5, 0.0], atol=1.0e-12)
    assert props.num_mass_points == 4
    assert props.skipped_elements == []


def test_constrained_axial_bar_modal_frequency_matches_sdof_reference() -> None:
    model = _axial_bar_model()
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))

    result = solve_free_vibration(model, num_modes=1)

    expected = np.sqrt(100.0 / 1.0) / (2.0 * np.pi)
    assert result.solver_status == "ok"
    assert result.frequencies_hz[0] == pytest.approx(expected, rel=1.0e-12)
    assert result.modes[0].modal_mass == pytest.approx(1.0)
    assert result.diagnostics["max_residual_norm"] < 1.0e-10
    assert result.result_case["analysis_case"]["analysis_type"] == "modal"
    spectral = result.diagnostics["spectral_guard_diagnostics"]
    assert spectral["residual_batch_columns"] >= result.num_modes_returned
    assert result.diagnostics["phase_timings_seconds"]["residual_batch"] >= 0.0
    assert spectral["full_guard_count"] > 0
    assert spectral["trusted_guard_count"] >= 0


def test_sparse_modal_shift_invert_uses_factorization_cache() -> None:
    model = FEModel("axial_chain_shift")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    for i in range(5):
        model.add_node(i + 1, float(i), 0.0, 0.0)
    for i in range(4):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    model.add_boundary_condition(BoundaryCondition("axial_only", [1, 2, 3, 4, 5], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    model.add_boundary_condition(BoundaryCondition("fix_left", [1], {"ux": 0.0}))
    cache = FactorizationCache(name="modal_test", max_entries=2)

    result = solve_free_vibration(model, num_modes=2, shift=1.0, dense_size_limit=1, factorization_cache=cache)

    assert result.solver_status == "ok"
    assert result.diagnostics["solver"] == "sparse_scipy_eigsh"
    assert result.diagnostics["shift_invert"] is True
    assert result.diagnostics["factorization_cache"]["misses"] == 1
    assert cache.diagnostics()["entries"] == 1


def test_supported_unshifted_sparse_modal_uses_retained_zero_shift_inverse() -> None:
    model = FEModel("axial_chain_zero_shift")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    for i in range(8):
        model.add_node(i + 1, float(i), 0.0, 0.0)
    for i in range(7):
        model.add_element(
            i + 1,
            BeamElement(
                i + 1,
                [i + 1, i + 2],
                "steel",
                {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
            ),
        )
    model.add_boundary_condition(
        BoundaryCondition(
            "axial_only",
            list(model.mesh.nodes),
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(BoundaryCondition("fix_left", [1], {"ux": 0.0}))
    cache = FactorizationCache(name="modal_zero_shift_test", max_entries=2)

    first = solve_free_vibration(
        model,
        num_modes=3,
        dense_size_limit=1,
        factorization_cache=cache,
    )
    second = solve_free_vibration(
        model,
        num_modes=3,
        dense_size_limit=1,
        factorization_cache=cache,
    )
    dense_reference = solve_free_vibration(
        model,
        num_modes=3,
        dense_size_limit=100,
    )

    assert first.solver_status == second.solver_status == "ok"
    assert dense_reference.solver_status == "ok"
    np.testing.assert_allclose(second.frequencies_hz, first.frequencies_hz, rtol=1.0e-12)
    np.testing.assert_allclose(
        second.frequencies_hz,
        dense_reference.frequencies_hz,
        rtol=1.0e-11,
    )
    sparse_basis = np.column_stack(
        [mode.mode_shape for mode in second.modes]
    )
    dense_basis = np.column_stack(
        [mode.mode_shape for mode in dense_reference.modes]
    )
    sparse_q, _ = np.linalg.qr(sparse_basis)
    dense_q, _ = np.linalg.qr(dense_basis)
    principal_cosines = np.linalg.svd(
        dense_q.T @ sparse_q,
        compute_uv=False,
    )
    np.testing.assert_allclose(principal_cosines, 1.0, atol=1.0e-10)
    assert second.diagnostics["zero_shift_inverse"] is True
    assert second.diagnostics["factorization_cache"]["hits"] == 1
    assert cache.diagnostics()["entries"] == 1
    assert second.diagnostics["max_residual_norm"] < 1.0e-9


def test_zero_shift_inverse_falls_back_for_constrained_internal_mechanism() -> None:
    stiffness = sparse.diags(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        format="csr",
    )
    mass = sparse.identity(10, format="csr")

    values, vectors, diagnostics = modal_module._sparse_eigensolve(
        stiffness,
        mass,
        num_modes=2,
        shift=None,
        use_zero_shift_inverse=True,
    )

    assert diagnostics["zero_shift_inverse_fallback"] is True
    assert diagnostics["shift_invert"] is False
    assert np.all(np.isfinite(values))
    assert vectors.shape == (10, 6)
    assert np.min(np.abs(values)) < 1.0e-10


def test_free_free_beam_modal_solver_identifies_six_rigid_modes() -> None:
    model = _axial_bar_model()

    result = solve_free_vibration(model, num_modes=6)

    assert result.solver_status == "ok"
    assert result.diagnostics["num_rigid_body_modes"] == 6
    assert np.max(result.frequencies_hz[:6]) < 1.0e-5
    assert result.nullspace_info["rank"] == 6


def test_mass_modal_validity_report_cli() -> None:
    output_dir = Path(".pytest_tmp_mass_modal") / str(os.getpid())
    output = output_dir / "mass_modal.json"
    markdown = output_dir / "mass_modal.md"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_mass_modal_validity.py",
                "--output",
                str(output),
                "--markdown",
                str(markdown),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr + completed.stdout
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "passed"
        assert markdown.exists()
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)


def test_consistent_beam_mass_option_exact_rigid_inertia_and_better_frequencies() -> None:
    """cross_section['consistent_mass']=True: exact rigid rotary inertia, closer modal frequencies."""
    E, rho, L = 2.1e11, 7850.0, 1.0
    A, I = 1.0e-3, 1.0e-8

    def _cantilever(n: int, consistent: bool) -> FEModel:
        model = FEModel("consistent_mass_cantilever")
        model.add_material("steel", E, 0.3, density=rho)
        section = {"area": A, "Iy": I, "Iz": I, "J": 2.0e-8}
        if consistent:
            section["consistent_mass"] = True
        for i in range(n + 1):
            model.add_node(i + 1, L * i / n, 0.0, 0.0)
        for e in range(n):
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", dict(section)))
        model.add_boundary_condition(
            BoundaryCondition("clamp", [1], {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0})
        )
        return model

    # rigid-body rotational inertia about the support is exact for the
    # consistent option (the lumped default overshoots the bar term).
    props = calculate_mass_properties(_cantilever(1, consistent=True), reference_point=(0.0, 0.0, 0.0))
    exact_inertia = rho * A * L**3 / 3.0 + rho * I * L
    assert props.total_mass == pytest.approx(rho * A * L)
    assert props.rigid_body_mass_matrix[5, 5] == pytest.approx(exact_inertia, rel=1.0e-12)

    # first cantilever bending frequency: consistent mass is substantially
    # closer to the Euler-Bernoulli analytic value on the same mesh.
    analytic = (1.8751**2) * np.sqrt(E * I / (rho * A)) / (2.0 * np.pi * L**2)
    f_lumped = solve_free_vibration(_cantilever(4, consistent=False), num_modes=1).frequencies_hz[0]
    f_consistent = solve_free_vibration(_cantilever(4, consistent=True), num_modes=1).frequencies_hz[0]
    assert abs(f_consistent - analytic) < 0.25 * abs(f_lumped - analytic)
    assert f_consistent == pytest.approx(analytic, rel=0.01)


def test_point_mass_enters_mass_matrix_and_shifts_frequency() -> None:
    """model.add_point_mass augments M (modal/dynamic) and the acceleration load."""
    from anyfesolver.boundary import LoadCase
    from anyfesolver.elements import create_shell_element
    from anyfesolver.matrix_assembly import assemble_load_vector, assemble_mass_matrix

    def _plate(add_mass: float = 0.0):
        model = FEModel("pm")
        model.add_material("steel", 2.1e11, 0.3, density=7850.0)
        n = 4
        ids = {}
        nid = 1
        for j in range(n + 1):
            for i in range(n + 1):
                model.add_node(nid, i / n, j / n, 0.0)
                ids[(i, j)] = nid
                nid += 1
        eid = 1
        for j in range(n):
            for i in range(n):
                model.add_element(
                    eid,
                    create_shell_element(
                        eid,
                        [
                            ids[(i, j)],
                            ids[(i + 1, j)],
                            ids[(i + 1, j + 1)],
                            ids[(i, j + 1)],
                        ],
                        "steel",
                        thickness=0.01,
                    ),
                )
                eid += 1
        edge = (
            [ids[(i, 0)] for i in range(n + 1)] + [ids[(i, n)] for i in range(n + 1)]
            + [ids[(0, j)] for j in range(1, n)] + [ids[(n, j)] for j in range(1, n)]
        )
        model.add_boundary_condition(BoundaryCondition("e", edge, {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0}))
        if add_mass > 0.0:
            model.add_point_mass(ids[(2, 2)], add_mass)
        return model

    M0, _ = assemble_mass_matrix(_plate())
    M1, info = assemble_mass_matrix(_plate(50.0))
    assert info["diagnostics"]["point_mass_count"] == 1
    # 50 kg on 3 translational DOFs adds 150 to the diagonal
    assert M1.diagonal().sum() - M0.diagonal().sum() == pytest.approx(150.0, rel=1e-9)

    f_bare = solve_free_vibration(_plate(), num_modes=1).frequencies_hz[0]
    f_loaded = solve_free_vibration(_plate(50.0), num_modes=1).frequencies_hz[0]
    assert f_loaded < 0.7 * f_bare  # a heavy central mass drops the first frequency substantially

    load_case = LoadCase("g")
    load_case.set_acceleration(0.0, 0.0, -9.81)
    model = _plate(80.0)
    F, _ = assemble_load_vector(model, load_case)
    structural = 7850.0 * 0.01 * 1.0
    fz = sum(F[model.mesh.get_node(nid).dofs[2]] for nid in model.mesh.nodes)
    assert fz == pytest.approx((structural + 80.0) * -9.81, rel=1e-6)
