import numpy as np
import pytest
from scipy import linalg, sparse

import anyfesolver.buckling as buckling_module
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.buckling import solve_eigenvalue_buckling
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.linalg import FactorizationCache
from anyfesolver.matrix_assembly import (
    _pack_owned_geometric_states,
    assemble_geometric_stiffness_matrix,
)


def _beam_column_model(num_elements=8):
    length = 4.0
    model = FEModel("beam_column_buckling")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.current_material = "steel"

    for i in range(num_elements + 1):
        x = length * i / num_elements
        model.add_node(i + 1, x, 0.0, 0.0)

    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for i in range(num_elements):
        element_id = i + 1
        model.add_element(element_id, BeamElement(element_id, [i + 1, i + 2], "steel", section))

    all_nodes = list(range(1, num_elements + 2))
    end_nodes = [1, num_elements + 1]
    model.add_boundary_condition(
        BoundaryCondition("suppress_unrelated_dofs", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0})
    )
    model.add_boundary_condition(BoundaryCondition("pinned_lateral_ends", end_nodes, {"uy": 0.0}))
    model.apply_boundary_conditions()
    return model, length, section


def test_beam_geometric_stiffness_is_symmetric_and_scales_with_compression():
    model, _, _ = _beam_column_model(num_elements=1)

    KG_100, info = assemble_geometric_stiffness_matrix(model, {1: {"axial_compression": 100.0}})
    KG_250, _ = assemble_geometric_stiffness_matrix(model, {1: {"axial_compression": 250.0}})

    assert info["matrix_type"] == "geometric_stiffness"
    assert KG_100.nnz > 0
    np.testing.assert_allclose(KG_100.toarray(), KG_100.toarray().T, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(KG_250.toarray(), 2.5 * KG_100.toarray(), rtol=1.0e-13, atol=1.0e-12)


def test_owned_prestress_envelope_preserves_geometric_matrix() -> None:
    model, _, _ = _beam_column_model(num_elements=3)
    states = {
        element_id: {"axial_compression": float(element_id)}
        for element_id in model.mesh.elements
    }

    scalar, _ = assemble_geometric_stiffness_matrix(model, states)
    packed, info = assemble_geometric_stiffness_matrix(
        model,
        _pack_owned_geometric_states(states),
    )

    np.testing.assert_allclose(packed.toarray(), scalar.toarray(), rtol=0.0, atol=0.0)
    assert info["diagnostics"]["packed_prestress_state"] is True


def test_negative_axial_force_is_interpreted_as_compression_for_beam_kg():
    model, _, _ = _beam_column_model(num_elements=1)

    KG_compression, _ = assemble_geometric_stiffness_matrix(model, {1: {"axial_compression": 50.0}})
    KG_axial_force, _ = assemble_geometric_stiffness_matrix(model, {1: {"axial_force": -50.0}})

    np.testing.assert_allclose(KG_axial_force.toarray(), KG_compression.toarray(), rtol=0.0, atol=1.0e-12)


def test_eigenvalue_buckling_returns_euler_column_scale_for_pinned_beam():
    model, length, section = _beam_column_model(num_elements=10)
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}

    result = solve_eigenvalue_buckling(model, states, num_modes=3)

    # Only uy/rz are free, so buckling happens in the x-y plane and is governed
    # by bending about local z, i.e. Iz.
    expected_euler = np.pi**2 * model.get_material("steel").elastic_modulus * section["Iz"] / length**2
    assert result.solver_status == "ok"
    assert result.num_modes_returned == 3
    assert result.critical_load_factor == pytest.approx(expected_euler, rel=0.08)
    assert [mode.load_factor for mode in result.modes] == sorted(mode.load_factor for mode in result.modes)
    assert np.max(np.abs(result.modes[0].mode_shape)) == pytest.approx(1.0)
    assert result.diagnostics["max_residual_norm"] < 1.0e-8
    assert result.modes[0].validity_status == "ok"
    assert result.result_case["solver_backend"] in {"scipy_superlu", None}
    spectral = result.diagnostics["spectral_guard_diagnostics"]
    assert spectral["residual_batch_columns"] >= result.num_modes_returned
    assert result.diagnostics["phase_timings_seconds"]["residual_batch"] >= 0.0
    assert spectral["full_guard_count"] > 0
    assert spectral["trusted_guard_count"] >= 0
    assert result.assembly_info["geometric_stiffness"]["diagnostics"][
        "packed_prestress_state"
    ] is True
    assert result.assembly_info["external_load_tangent"]["diagnostics"][
        "dead_load_zero_tangent_fast_path"
    ] is True


def test_supported_unshifted_buckling_reuses_only_elastic_inverse() -> None:
    model, _, _ = _beam_column_model(num_elements=12)
    states = {
        element_id: {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }
    cache = FactorizationCache(name="buckling_elastic_inverse_test", max_entries=2)

    first = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=3,
        dense_size_limit=1,
        factorization_cache=cache,
    )
    second = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=3,
        dense_size_limit=1,
        factorization_cache=cache,
    )
    dense_reference = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=3,
        dense_size_limit=100,
    )

    assert first.solver_status == second.solver_status == "ok"
    assert dense_reference.solver_status == "ok"
    np.testing.assert_allclose(
        [mode.load_factor for mode in second.modes],
        [mode.load_factor for mode in first.modes],
        rtol=1.0e-11,
    )
    np.testing.assert_allclose(
        [mode.load_factor for mode in second.modes],
        [mode.load_factor for mode in dense_reference.modes],
        rtol=1.0e-10,
    )
    assert second.diagnostics["elastic_inverse"] is True
    assert (
        second.diagnostics["elastic_inverse_persistence"]
        == "caller_owned_elastic_only"
    )
    assert second.diagnostics["elastic_inverse_cache"]["hits"] == 1
    assert second.diagnostics["max_residual_norm"] < 1.0e-8


def test_buckling_preserves_external_load_tangent_replacement_seam(
    monkeypatch,
) -> None:
    model, _, _ = _beam_column_model(num_elements=6)
    states = {
        element_id: {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }
    total_dofs = model.mesh.dof_manager.total_dofs
    replacement = sparse.eye(total_dofs, format="csr", dtype=float) * 1.0e-3
    calls: list[int] = []

    def replacement_tangent(_model, _case, _displacements):
        calls.append(1)
        return replacement, {"matrix_type": "replacement_load_tangent"}

    monkeypatch.setattr(
        buckling_module,
        "assemble_external_load_tangent",
        replacement_tangent,
    )

    result = solve_eigenvalue_buckling(model, states, num_modes=2)

    assert result.solver_status == "ok"
    assert calls == [1]
    assert (
        result.assembly_info["external_load_tangent"]["matrix_type"]
        == "replacement_load_tangent"
    )
    assert result.diagnostics["follower_load_stiffness_included"] is True


def test_dead_load_fast_path_defers_to_load_case_descriptors(monkeypatch) -> None:
    model, _, _ = _beam_column_model(num_elements=4)
    states = {
        element_id: {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }
    load_case = LoadCase("descriptor_follower")
    monkeypatch.setattr(
        LoadCase,
        "follower_pressure",
        property(lambda _self: True),
    )

    result = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=1,
        reference_load_case=load_case,
    )

    diagnostics = result.assembly_info["external_load_tangent"]["diagnostics"]
    assert "dead_load_zero_tangent_fast_path" not in diagnostics
    assert result.assembly_info["external_load_tangent"][
        "pressure_configuration"
    ] == "current"


def test_dead_load_fast_path_defers_to_load_case_attribute_hooks(monkeypatch) -> None:
    model, _, _ = _beam_column_model(num_elements=4)
    states = {
        element_id: {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }
    load_case = LoadCase("hook_follower")

    def hooked_getattribute(self, name):
        if name == "follower_pressure":
            return True
        return object.__getattribute__(self, name)

    monkeypatch.setattr(LoadCase, "__getattribute__", hooked_getattribute)

    result = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=1,
        reference_load_case=load_case,
    )

    diagnostics = result.assembly_info["external_load_tangent"]["diagnostics"]
    assert "dead_load_zero_tangent_fast_path" not in diagnostics
    assert result.assembly_info["external_load_tangent"][
        "pressure_configuration"
    ] == "current"


def test_dead_load_fast_path_preserves_dynamic_pressure_surface_diagnostics(
    monkeypatch,
) -> None:
    model, _, _ = _beam_column_model(num_elements=4)
    states = {
        element_id: {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }
    load_case = LoadCase("dynamic_pressure_surface")
    load_case.pressure_loads[1] = 1.0
    monkeypatch.setattr(
        BeamElement,
        "pressure_surface_policy_id",
        property(lambda _self: "ELEMENT_NODAL_REFERENCE_SURFACE_V1"),
        raising=False,
    )

    result = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=1,
        reference_load_case=load_case,
    )

    tangent_info = result.assembly_info["external_load_tangent"]
    assert "dead_load_zero_tangent_fast_path" not in tangent_info["diagnostics"]
    assert tangent_info["qualified_s3_pressure_surfaces"] == [
        {
            "element_id": 1,
            "pressure_surface_id": "ELEMENT_NODAL_REFERENCE_SURFACE_V1",
            "reference_surface_offset": 0.0,
            "resultant_and_reaction_reference": (
                "GLOBAL_NODAL_REFERENCE_COORDINATES"
            ),
            "section_origin_offset_from_reference": -0.0,
            "virtual_work": "TRANSLATIONAL_NODAL_REFERENCE_SURFACE_ONLY",
        }
    ]


def test_dead_load_fast_path_preserves_static_pressure_diagnostics() -> None:
    model, _, _ = _beam_column_model(num_elements=4)
    states = {
        element_id: {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }
    load_case = LoadCase("static_pressure_surface")
    load_case.pressure_loads[1] = 1.0

    result = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=1,
        reference_load_case=load_case,
    )

    tangent_info = result.assembly_info["external_load_tangent"]
    assert "dead_load_zero_tangent_fast_path" not in tangent_info["diagnostics"]
    assert tangent_info["matrix_type"] == "external_load_tangent"


def test_dead_load_fast_path_preserves_missing_load_name_failure() -> None:
    model, _, _ = _beam_column_model(num_elements=4)
    states = {
        element_id: {"axial_compression": 1.0}
        for element_id in model.mesh.elements
    }
    load_case = LoadCase("missing_name")
    del load_case.name

    with pytest.raises(AttributeError):
        solve_eigenvalue_buckling(
            model,
            states,
            num_modes=1,
            reference_load_case=load_case,
        )


def test_buckling_applies_model_constraints_independent_of_prior_call_history():
    preapplied_model, _, _ = _beam_column_model(num_elements=8)
    fresh_model, _, _ = _beam_column_model(num_elements=8)
    fresh_model.mesh.dof_manager._constrained_dofs.clear()
    assert fresh_model.boundary_conditions
    assert fresh_model.mesh.dof_manager._constrained_dofs == set()
    states = {element_id: {"axial_compression": 1.0} for element_id in preapplied_model.mesh.elements}

    preapplied = solve_eigenvalue_buckling(preapplied_model, states, num_modes=2)
    fresh = solve_eigenvalue_buckling(fresh_model, states, num_modes=2)

    assert preapplied.solver_status == fresh.solver_status == "ok"
    assert fresh.critical_load_factor == pytest.approx(preapplied.critical_load_factor, rel=1.0e-12)
    assert fresh.num_modes_returned == preapplied.num_modes_returned
    constrained = sorted(fresh_model.mesh.dof_manager._constrained_dofs)
    assert constrained
    for mode in fresh.modes:
        np.testing.assert_allclose(mode.mode_shape[constrained], 0.0, atol=1.0e-12)


def test_buckling_mode_shapes_respect_fixed_constraint_dofs():
    model, _, _ = _beam_column_model(num_elements=6)
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}

    result = solve_eigenvalue_buckling(model, states, num_modes=2)

    constrained = sorted(model.mesh.dof_manager._constrained_dofs)
    assert constrained
    assert result.solver_status == "ok"
    for mode in result.modes:
        np.testing.assert_allclose(mode.mode_shape[constrained], 0.0, atol=1.0e-12)
        assert np.max(np.abs(mode.mode_shape)) == pytest.approx(1.0)
        assert mode.residual_norm < 1.0e-8


def _free_beam_model(length: float = 25.0) -> FEModel:
    model = FEModel("free_free_buckling_guard")
    model.add_material("steel", 210.0e9, 0.3)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, length, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 0.02, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    return model


def test_free_free_dead_prestress_fails_closed_when_geometric_operator_does_not_descend():
    model = _free_beam_model()

    result = solve_eigenvalue_buckling(
        model,
        {1: {"axial_compression": 1.0}},
        num_modes=2,
        allow_dense_fallback=True,
        allow_free_mechanisms=True,
    )

    assert result.solver_status == "invalid_rigid_quotient"
    assert result.modes == []
    assert result.diagnostics["solver"] == "dense_scipy_eigh_rigid_quotient"
    assert result.diagnostics["rigid_body_handling"] == "projected"
    assert result.diagnostics["free_mechanism_handling"] == "retained"
    assert "does not descend" in result.diagnostics["reason"]
    projection = result.diagnostics["rigid_projection"]
    dimension_tolerance = 4096.0 * projection["original_dofs"] * 2.220446049250313e-16
    assert projection["applied"] is True
    assert projection["metric_version"] == "dimensionless_full_dof_bbox_v1"
    assert projection["characteristic_length"] == pytest.approx(25.0)
    assert projection["rigid_rank"] == 6
    assert projection["elastic_null_residual"] <= dimension_tolerance
    assert projection["elastic_cross_residual"] <= dimension_tolerance
    assert projection["geometric_null_residual"] > dimension_tolerance
    assert projection["geometric_cross_residual"] > dimension_tolerance


def test_synthetic_descending_pencil_uses_basis_invariant_dimensionless_quotient(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _free_beam_model(length=100.0)
    total_dofs = model.mesh.dof_manager.total_dofs
    identity_transformation = sparse.eye(total_dofs, format="csr", dtype=float)

    original_builder = buckling_module.build_reduced_rigid_body_modes
    rigid_basis, _ = original_builder(
        model,
        np.arange(total_dofs, dtype=int),
        total_dofs,
        transformation=identity_transformation,
    )
    inverse_scale = np.ones(total_dofs, dtype=float)
    for node in model.mesh.nodes.values():
        inverse_scale[np.asarray(node.dofs[:3], dtype=np.intp)] = 1.0 / 100.0
    _metric_basis, metric_factor = linalg.qr(
        np.diag(inverse_scale),
        mode="economic",
    )
    metric_inverse = linalg.solve_triangular(
        metric_factor,
        np.eye(total_dofs),
        lower=False,
    )
    quotient_basis, _ = linalg.qr(metric_factor @ rigid_basis, mode="full")
    flexible_basis = quotient_basis[:, rigid_basis.shape[1] :]
    mapped_flexible = metric_inverse @ flexible_basis
    legacy_correlations = np.max(
        np.abs(
            rigid_basis.T
            @ (mapped_flexible / np.linalg.norm(mapped_flexible, axis=0, keepdims=True))
        ),
        axis=0,
    )
    descending_order = np.argsort(-legacy_correlations)
    flexible_basis = flexible_basis[:, descending_order]
    assert float(legacy_correlations[descending_order[0]]) > 0.90

    physical_factors = np.arange(1.0, flexible_basis.shape[1] + 1.0)
    stiffness_y = flexible_basis @ np.diag(physical_factors) @ flexible_basis.T
    geometric_y = flexible_basis @ flexible_basis.T
    stiffness = sparse.csr_matrix(metric_factor.T @ stiffness_y @ metric_factor)
    geometric = sparse.csr_matrix(metric_factor.T @ geometric_y @ metric_factor)
    zero = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)

    monkeypatch.setattr(
        buckling_module,
        "assemble_stiffness_matrix",
        lambda _model: (stiffness, {"matrix_type": "synthetic_stiffness"}),
    )
    monkeypatch.setattr(
        buckling_module,
        "assemble_geometric_stiffness_matrix",
        lambda _model, _states: (geometric, {"matrix_type": "synthetic_geometric_stiffness"}),
    )
    monkeypatch.setattr(
        buckling_module,
        "assemble_external_load_tangent",
        lambda _model, _case, _displacements: (zero, {"matrix_type": "synthetic_zero_load_tangent"}),
    )

    captured: dict[str, np.ndarray] = {}

    def _capture_basis(*args, **kwargs):
        basis, info = original_builder(*args, **kwargs)
        captured["basis"] = basis.copy()
        return basis, info

    monkeypatch.setattr(buckling_module, "build_reduced_rigid_body_modes", _capture_basis)
    result = solve_eigenvalue_buckling(
        model,
        {},
        num_modes=2,
        allow_dense_fallback=True,
    )

    assert result.solver_status == "ok", result.diagnostics
    assert result.num_modes_returned == 2
    assert result.diagnostics["nullspace_rank"] == 6
    assert result.diagnostics["free_mechanism_handling"] == "rigid_body_roots_filtered"
    assert result.diagnostics["rigid_body_handling"] == "projected"
    assert result.diagnostics["solver"] == "dense_scipy_eigh_rigid_quotient"
    assert result.assembly_info["nullspace"]["rank"] == 6
    projection = result.diagnostics["rigid_projection"]
    assert set(projection) == {
        "applied",
        "metric_version",
        "characteristic_length",
        "original_dofs",
        "projected_dofs",
        "rigid_rank",
        "orthonormality_residual",
        "elastic_null_residual",
        "geometric_null_residual",
        "elastic_cross_residual",
        "geometric_cross_residual",
        "elastic_min_eigenvalue",
        "elastic_max_eigenvalue",
        "spd_tolerance",
        "spd_sensitivity",
    }
    assert projection["applied"] is True
    assert projection["metric_version"] == "dimensionless_full_dof_bbox_v1"
    assert projection["characteristic_length"] == pytest.approx(100.0)
    assert projection["original_dofs"] == 12
    assert projection["projected_dofs"] == 6
    assert projection["rigid_rank"] == 6
    assert set(projection["spd_sensitivity"]) == {"0.25", "1", "4"}
    assert all(item["positive_definite"] for item in projection["spd_sensitivity"].values())
    assert all(mode.rigid_body_correlation < 1.0e-12 for mode in result.modes)

    # Orthogonality is defined in the registered dimensionless metric.  For
    # this long beam the mapped physical representative has high Euclidean-z
    # correlation with the legacy basis and must not be rejected on that basis.
    mapped_mode_correlations = [
        float(
            np.max(
                np.abs(
                    captured["basis"].T
                    @ mode.reduced_mode_shape
                    / np.linalg.norm(mode.reduced_mode_shape)
                )
            )
        )
        for mode in result.modes
    ]
    assert max(mapped_mode_correlations) > 0.90

    rigid_rank = captured["basis"].shape[1]
    orthogonal_change = np.eye(rigid_rank, dtype=float)
    for first, angle in ((0, 0.37), (2, -0.61), (4, 0.83)):
        cosine = np.cos(angle)
        sine = np.sin(angle)
        orthogonal_change[first : first + 2, first : first + 2] = [
            [cosine, -sine],
            [sine, cosine],
        ]
    np.testing.assert_allclose(
        orthogonal_change.T @ orthogonal_change,
        np.eye(rigid_rank),
        rtol=0.0,
        atol=1.0e-15,
    )

    def _rotated_basis(*args, **kwargs):
        basis, info = original_builder(*args, **kwargs)
        return basis @ orthogonal_change, info

    monkeypatch.setattr(buckling_module, "build_reduced_rigid_body_modes", _rotated_basis)
    rotated = solve_eigenvalue_buckling(
        model,
        {},
        num_modes=2,
        allow_dense_fallback=True,
    )
    assert rotated.solver_status == "ok"
    assert rotated.diagnostics["solver"] == "dense_scipy_eigh_rigid_quotient"
    assert rotated.diagnostics["rigid_projection"]["metric_version"] == "dimensionless_full_dof_bbox_v1"
    np.testing.assert_allclose(
        [mode.load_factor for mode in rotated.modes],
        [mode.load_factor for mode in result.modes],
        rtol=1.0e-11,
        atol=0.0,
    )

    requires_dense = solve_eigenvalue_buckling(
        model,
        {},
        num_modes=2,
        dense_size_limit=2,
        allow_dense_fallback=False,
    )
    assert requires_dense.solver_status == "rigid_projection_requires_dense"
    assert requires_dense.diagnostics["solver"] == "dense_scipy_eigh_rigid_quotient"
    assert requires_dense.diagnostics["rigid_projection"]["applied"] is False


def test_buckling_load_factor_scales_inverse_to_reference_compression():
    model, _, _ = _beam_column_model(num_elements=6)
    unit_states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    double_states = {element_id: {"axial_compression": 2.0} for element_id in model.mesh.elements}

    unit_result = solve_eigenvalue_buckling(model, unit_states, num_modes=1)
    double_result = solve_eigenvalue_buckling(model, double_states, num_modes=1)

    assert unit_result.solver_status == "ok"
    assert double_result.solver_status == "ok"
    assert double_result.critical_load_factor == pytest.approx(0.5 * unit_result.critical_load_factor, rel=1.0e-10)


def test_sparse_buckling_path_reports_residuals_and_rejected_roots():
    model, _, _ = _beam_column_model(num_elements=12)
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}

    result = solve_eigenvalue_buckling(model, states, num_modes=2, dense_size_limit=2)

    assert result.solver_status == "ok"
    assert result.diagnostics["solver"] == "sparse_scipy_eigsh_inverted_pencil"
    assert result.diagnostics["num_roots_considered"] >= result.num_modes_returned
    assert result.diagnostics["max_residual_norm"] < 1.0e-8
    assert "rejected_roots" in result.diagnostics
    assert all(mode.residual_norm < 1.0e-8 for mode in result.modes)


def test_sparse_buckling_shift_invert_reports_factorization_cache():
    model, _, _ = _beam_column_model(num_elements=12)
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    first = solve_eigenvalue_buckling(model, states, num_modes=1, dense_size_limit=2)
    cache = FactorizationCache(name="buckling_test", max_entries=2)

    shifted = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=1,
        dense_size_limit=2,
        shift_load_factor=first.critical_load_factor,
        factorization_cache=cache,
    )

    assert shifted.solver_status == "ok"
    assert shifted.diagnostics["shift_invert"] is True
    assert shifted.diagnostics["factorization_cache"]["misses"] == 1
    assert cache.diagnostics()["entries"] == 1


def test_buckling_factor_range_can_target_higher_mode_family():
    model, _, _ = _beam_column_model(num_elements=14)
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    first = solve_eigenvalue_buckling(model, states, num_modes=1)

    higher = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=1,
        load_factor_range=(3.0 * first.critical_load_factor, 5.0 * first.critical_load_factor),
        dense_size_limit=2,
    )

    assert higher.solver_status == "ok"
    assert higher.critical_load_factor == pytest.approx(4.0 * first.critical_load_factor, rel=0.05)
    assert higher.diagnostics["num_rejected_roots"] > 0
    assert all(item.get("reason") != "invalid_load_factor" for item in higher.diagnostics["rejected_roots"])


def test_repeated_buckling_mode_groups_are_reported_for_symmetric_column():
    length = 4.0
    num_elements = 12
    model = FEModel("symmetric_column")
    model.add_material("steel", 210.0e9, 0.3)
    for i in range(num_elements + 1):
        model.add_node(i + 1, length * i / num_elements, 0.0, 0.0)
    section = {"area": 0.02, "Iy": 4.0e-6, "Iz": 4.0e-6, "J": 2.0e-6}
    for element_id in range(1, num_elements + 1):
        model.add_element(element_id, BeamElement(element_id, [element_id, element_id + 1], "steel", section))
    all_nodes = list(range(1, num_elements + 2))
    end_nodes = [1, num_elements + 1]
    model.add_boundary_condition(BoundaryCondition("suppress_axial_torsion", all_nodes, {"ux": 0.0, "rx": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pinned_lateral_ends", end_nodes, {"uy": 0.0, "uz": 0.0}))
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}

    result = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=4,
        dense_size_limit=2,
        repeated_tolerance=1.0e-2,
        allow_free_mechanisms=True,
    )

    assert result.solver_status == "ok"
    # The declared supports remove every rigid mode; buckling applies them
    # itself rather than inheriting an empty constraint set from call history.
    assert result.assembly_info["nullspace"]["rank"] == 0
    assert result.diagnostics["num_repeated_mode_groups"] >= 1
    first_group = result.diagnostics["repeated_mode_groups"][0]
    assert first_group["mode_numbers"][:2] == [1, 2]
    assert result.modes[0].repeated_group == result.modes[1].repeated_group


def test_torsional_column_buckling_matches_wagner_analytic_load():
    """Axial compression + weak St. Venant torsion: P_cr = G*J*A/Ip (no warping)."""
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver.elements import BeamElement, QuadraticBeamElement
    from anyfesolver.fe_core import FEModel

    section = {"area": 1.0e-2, "Iy": 1.0e-5, "Iz": 1.0e-5, "J": 1.0e-9}

    def _column(quadratic: bool) -> FEModel:
        model = FEModel("torsion_column")
        model.add_material("steel", 2.1e11, 0.3, density=7850.0)
        n = 6
        grid = 2 * n if quadratic else n
        for i in range(grid + 1):
            model.add_node(i + 1, i / grid, 0.0, 0.0)
        for e in range(n):
            if quadratic:
                nodes = [2 * e + 1, 2 * e + 2, 2 * e + 3]
                model.add_element(e + 1, QuadraticBeamElement(e + 1, nodes, "steel", section))
            else:
                model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", section))
        end = grid + 1
        model.add_boundary_condition(BoundaryCondition("fork_a", [1], {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0}))
        model.add_boundary_condition(BoundaryCondition("fork_b", [end], {"uy": 0.0, "uz": 0.0, "rx": 0.0}))
        return model

    material_G = 2.1e11 / (2.0 * (1.0 + 0.3))
    polar = section["Iy"] + section["Iz"]
    expected_torsional = material_G * section["J"] * section["area"] / polar
    euler = np.pi**2 * 2.1e11 * section["Iy"] / 1.0**2
    assert expected_torsional < 0.01 * euler  # torsion must govern

    for quadratic in (False, True):
        model = _column(quadratic)
        states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
        result = solve_eigenvalue_buckling(model, states, num_modes=2)
        assert result.solver_status == "ok"
        # Same linear twist-gradient field in K and KG for the 2-node beam makes
        # the torsional eigenvalue mesh-exact; the quadratic beam is near-exact.
        assert result.critical_load_factor == pytest.approx(expected_torsional, rel=1.0e-3)
