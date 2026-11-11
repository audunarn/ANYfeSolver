from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import anyfesolver
import numpy as np
import pytest
from anyfesolver import runtime
from anyfesolver import anystructure_fem_mode
from anyfesolver import arc_length


def _flat_geometry() -> dict[str, object]:
    return {
        "domain": "flat",
        "is_cylinder": False,
        "length": 2.0,
        "width": 1.0,
        "plate_thickness": 0.012,
        "stiffener_spacing": 0.5,
        "girder_spacing": 1.0,
        "stiffener_section": {
            "area": 0.002,
            "Iy": 1.0e-6,
            "Iz": 2.0e-7,
            "J": 1.0e-8,
        },
        "girder_section": {
            "area": 0.004,
            "Iy": 4.0e-6,
            "Iz": 8.0e-7,
            "J": 4.0e-8,
        },
    }


def test_runtime_contract_is_headless_and_builds_geometry() -> None:
    config = runtime.LightweightFEMConfig(
        mesh_fidelity="coarse",
        include_stiffeners=True,
        include_girders=True,
    )

    generated = runtime.build_generated_geometry(_flat_geometry(), config)

    assert runtime.full_backend_available()
    assert generated["nodes"]
    assert generated["shells"]
    assert "beams" in generated


@pytest.mark.parametrize("include_end_lids", [False, True])
def test_custom_load_mode_keeps_cylinder_per_edge_dof_constraints(
    include_end_lids: bool,
) -> None:
    geometry = {
        "geometry": "cylinder",
        "radius_m": 2.0,
        "length_m": 5.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": False,
    }
    fixed_dofs = {dof: 0.0 for dof in ("ux", "uy", "uz", "rx", "ry", "rz")}
    config = runtime.LightweightFEMConfig(
        mesh_fidelity="coarse",
        include_end_lids=include_end_lids,
        custom_load_bc_enabled=True,
        custom_loads_add_to_imported=False,
        custom_use_nullspace_projection=False,
        boundary_constraint_json=json.dumps({"lower": fixed_dofs}),
        edge_load_components_json=json.dumps({"upper": {"fx": 100_000.0}}),
    )

    generated = runtime.build_generated_geometry(geometry, config)
    supports = [
        support
        for support in generated["supports"]
        if str(support["name"]).startswith("whole_boundary_dof_constraint")
    ]

    assert len(supports) == 1
    assert supports[0]["constraints"] == fixed_dofs
    coords = {
        int(node["id"]): tuple(float(value) for value in node["coords"])
        for node in generated["nodes"]
    }
    assert all(abs(coords[int(node_id)][2]) < 1.0e-12 for node_id in supports[0]["node_ids"])
    expected_count = 1 if include_end_lids else len(set(generated["plot_grid"][0]))
    assert len(supports[0]["node_ids"]) == expected_count


def test_custom_load_mode_keeps_flat_per_edge_dof_constraints() -> None:
    fixed_dofs = {dof: 0.0 for dof in ("ux", "uy", "uz", "rx", "ry", "rz")}
    config = runtime.LightweightFEMConfig(
        mesh_fidelity="coarse",
        custom_load_bc_enabled=True,
        custom_loads_add_to_imported=False,
        custom_use_nullspace_projection=False,
        boundary_constraint_json=json.dumps({"x0": fixed_dofs}),
        edge_load_components_json=json.dumps({"x1": {"fx": 100_000.0}}),
    )

    generated = runtime.build_generated_geometry(_flat_geometry(), config)
    supports = [
        support
        for support in generated["supports"]
        if str(support["name"]).startswith("whole_boundary_dof_constraint")
    ]

    assert len(supports) == 1
    assert supports[0]["constraints"] == fixed_dofs
    coords = {
        int(node["id"]): tuple(float(value) for value in node["coords"])
        for node in generated["nodes"]
    }
    assert all(abs(coords[int(node_id)][0]) < 1.0e-12 for node_id in supports[0]["node_ids"])


def test_buckling_mode_visualization_uses_displacement_amplitude_without_stress_recovery(
    monkeypatch,
) -> None:
    class Node:
        def __init__(self) -> None:
            self.dofs = [0, 1, 2, 3, 4, 5]
            self.x = 1.0
            self.y = 2.0
            self.z = 3.0

        def coords(self):
            return (self.x, self.y, self.z)

    node = Node()
    mesh = SimpleNamespace(
        nodes={1: node},
        elements={},
        get_node=lambda node_id: node if int(node_id) == 1 else None,
        get_element=lambda _element_id: None,
    )
    model = SimpleNamespace(mesh=mesh)
    mode = SimpleNamespace(
        mode_number=1,
        load_factor=2.5,
        mode_shape=np.asarray([3.0, 4.0, 0.0, 0.0, 0.0, 0.0]),
    )

    def forbidden_stress_recovery(*_args, **_kwargs):
        raise AssertionError("mode visualization must not recover stresses")

    monkeypatch.setattr(
        runtime,
        "_backend_compute_stresses",
        forbidden_stress_recovery,
    )

    visualizations = runtime._buckling_mode_visualizations(
        {"plot_type": "flat", "plot_grid": [[1]], "shells": [], "beams": []},
        model,
        SimpleNamespace(modes=(mode,)),
    )

    assert len(visualizations) == 1
    shape = visualizations[0]["shape"]
    assert shape["scalar_label"] == "mode amplitude"
    assert shape["stress_pa"] == ((5.0,),)
    assert shape["fields"]["custom_scalar"] == ((5.0,),)


def test_runtime_public_api_is_explicit_and_complete() -> None:
    expected = {
        "GeneratedGeometry",
        "LightweightFEMConfig",
        "LightweightFEMResult",
        "NormalizedGeometry",
        "RuntimeAnalysisSelection",
        "RuntimeAnalysisContext",
        "StatusCallback",
        "apply_mode_shape_imperfections",
        "build_generated_geometry",
        "dnv_c208_steel_properties",
        "full_backend_api",
        "full_backend_available",
        "run_lightweight_fem",
        "run_production_fem",
        "resolve_runtime_analysis",
        "runtime_imperfection_preview_offsets",
        "warm_fe_solver_kernels",
    }

    assert set(runtime.__all__) == expected
    assert all(hasattr(runtime, name) for name in runtime.__all__)


def test_runtime_forwards_optional_nonlinear_static_warmup(monkeypatch) -> None:
    observed = {}

    def warm(shell_orders, **options):
        observed["shell_orders"] = shell_orders
        observed.update(options)
        return {"status": "completed"}

    monkeypatch.setattr(runtime, "_backend_warm_fe_solver_kernels", warm)

    result = runtime.warm_fe_solver_kernels(
        ("S4",),
        include_nonlinear_static=True,
    )

    assert result == {"status": "completed"}
    assert observed == {
        "shell_orders": ("S4",),
        "include_nonlinear_impact": False,
        "include_nonlinear_static": True,
    }


def test_runtime_analysis_selection_is_public_and_normalized() -> None:
    selection = runtime.resolve_runtime_analysis(
        runtime.LightweightFEMConfig(
            analysis_type="geom. + material nonlinear static",
            nonlinear_solution_control="arc length",
            nonlinear_static_kinematics="Corotational",
        )
    )

    assert selection == runtime.RuntimeAnalysisSelection(
        static_nonlinear=True,
        material_nonlinear=True,
        solution_control="arc length",
        kinematics="corotational",
    )


def test_root_exports_arc_length_and_generic_generated_geometry_api() -> None:
    expected_root_names = {
        "ArcLengthControl",
        "ArcLengthResult",
        "ArcLengthStep",
        "solve_static_arc_length",
        "GeneratedGeometryFEMConfig",
        "GeneratedGeometryFEMResult",
        "run_generated_geometry_fem",
    }

    assert expected_root_names <= set(anyfesolver.__all__)
    assert anyfesolver.ArcLengthControl is arc_length.ArcLengthControl
    assert anyfesolver.ArcLengthResult is arc_length.ArcLengthResult
    assert anyfesolver.ArcLengthStep is arc_length.ArcLengthStep
    assert anyfesolver.solve_static_arc_length is arc_length.solve_static_arc_length


def test_generic_generated_geometry_aliases_match_historical_api() -> None:
    generic_names = {
        "GeneratedGeometryFEMConfig",
        "GeneratedGeometryFEMResult",
        "run_generated_geometry_fem",
    }

    assert generic_names <= set(anystructure_fem_mode.__all__)
    assert anyfesolver.GeneratedGeometryFEMConfig is anyfesolver.AnyStructureFEMConfig
    assert anyfesolver.GeneratedGeometryFEMResult is anyfesolver.AnyStructureFEMResult
    assert anyfesolver.run_generated_geometry_fem is anyfesolver.run_anystructure_fem_mode

    assert (
        anystructure_fem_mode.GeneratedGeometryFEMConfig
        is anystructure_fem_mode.AnyStructureFEMConfig
    )
    assert (
        anystructure_fem_mode.GeneratedGeometryFEMResult
        is anystructure_fem_mode.AnyStructureFEMResult
    )
    assert (
        anystructure_fem_mode.run_generated_geometry_fem
        is anystructure_fem_mode.run_anystructure_fem_mode
    )


def test_runtime_rejects_curved_b3_ring_girders() -> None:
    geometry = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 2.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": True,
        "girder_spacing_m": 1.0,
    }

    with pytest.raises(ValueError, match="B3 ring girders are not supported"):
        runtime.build_generated_geometry(
            geometry,
            runtime.LightweightFEMConfig(beam_element_order="B3"),
        )


def test_runtime_cylinder_adaptive_refinement_preserves_b3_midnodes() -> None:
    geometry = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 2.0,
        "thickness_m": 0.012,
        "has_stiffener": True,
        "stiffener_spacing_m": 1.0,
        "has_girder": False,
        "stiffener_section": {
            "area": 0.002,
            "Iy": 1.0e-6,
            "Iz": 2.0e-7,
            "J": 1.0e-8,
        },
    }
    config = runtime.LightweightFEMConfig(
        mesh_fidelity="coarse",
        beam_element_order="B3",
        include_stiffeners=True,
        include_girders=False,
        point_refinement_enabled=True,
        point_refinement_x_m=0.6,
        point_refinement_y_m=0.3,
        point_refinement_fine_factor=0.3,
        point_refinement_extent_m=0.35,
    )

    generated = runtime.build_generated_geometry(geometry, config)
    coordinates = {
        int(node["id"]): [float(value) for value in node["coords"]]
        for node in generated["nodes"]
    }
    b3_beams = [beam for beam in generated["beams"] if len(beam["node_ids"]) == 3]

    assert generated["adaptive_mesh"]["enabled"]
    assert b3_beams
    for beam in b3_beams:
        start, middle, end = (coordinates[int(node_id)] for node_id in beam["node_ids"])
        expected_middle = [0.5 * (a + b) for a, b in zip(start, end)]
        assert middle == pytest.approx(expected_middle, abs=1.0e-12)


def test_runtime_lightweight_smoke() -> None:
    result = runtime.run_lightweight_fem(
        _flat_geometry(),
        runtime.LightweightFEMConfig(mesh_fidelity="coarse", pressure_pa=10_000.0),
    )

    assert result.status == "ok"
    assert result.mesh_info["nodes"] > 0
    assert result.mesh_info["shells"] > 0


def test_runtime_cylinder_smoke() -> None:
    geometry = {
        "geometry": "cylinder",
        "length_m": 2.0,
        "radius_m": 1.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": False,
    }

    result = runtime.run_lightweight_fem(
        geometry,
        runtime.LightweightFEMConfig(mesh_fidelity="coarse", pressure_pa=10_000.0),
    )

    assert result.status == "ok"
    assert result.mesh_info["nodes"] > 0
    assert result.mesh_info["shells"] > 0


def test_requested_nonlinear_failure_is_not_reported_as_ok(monkeypatch) -> None:
    def fail_nonlinear(*_args, **_kwargs):
        raise RuntimeError("synthetic nonlinear failure")

    monkeypatch.setattr(runtime, "_backend_solve_static_nonlinear", fail_nonlinear)
    result = runtime.run_production_fem(
        _flat_geometry(),
        runtime.LightweightFEMConfig(
            mesh_fidelity="coarse",
            pressure_pa=10_000.0,
            analysis_type="geom. + material nonlinear static",
        ),
    )

    assert result.status == "nonlinear_static_failed"
    assert result.prestress_summary["nonlinear_static_status"] == "failed"
    assert any("synthetic nonlinear failure" in message for message in result.diagnostics)


def test_runtime_static_only_cylinder_never_enters_eigenvalue_buckling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = {
        "geometry": "cylinder",
        "length_m": 0.4,
        "radius_m": 0.2,
        "thickness_m": 0.01,
        "has_stiffener": False,
        "has_girder": False,
    }

    def forbidden_buckling(*_args, **_kwargs):
        raise AssertionError("static-only analysis must not call eigenvalue buckling")

    monkeypatch.setattr(runtime, "_backend_solve_buckling", forbidden_buckling)

    result = runtime.run_production_fem(
        geometry,
        runtime.LightweightFEMConfig(
            mesh_fidelity="coarse",
            runtime_solver="static only",
            analysis_type="linear static",
            pressure_pa=1_000.0,
            include_stiffeners=False,
            include_girders=False,
            num_buckling_modes=3,
        ),
    )

    assert result.status == "ok"
    assert result.buckling_factors == ()
    assert not any("buckling-only" in item.lower() for item in result.diagnostics)


def test_runtime_rejects_follower_pressure_outside_nonlinear_static() -> None:
    result = runtime.run_production_fem(
        _flat_geometry(),
        runtime.LightweightFEMConfig(
            mesh_fidelity="coarse",
            runtime_solver="static only",
            pressure_pa=1_000.0,
            follower_pressure=True,
        ),
    )

    assert result.status == "invalid_follower_pressure"
    assert any(
        "requires a nonlinear static or arc-length" in message
        for message in result.diagnostics
    )


def test_runtime_rejects_follower_pressure_with_stepwise_eigen_buckling() -> None:
    reason = runtime._invalid_follower_pressure_reason(
        runtime.LightweightFEMConfig(
            runtime_solver="stepwise",
            analysis_type="geometric nonlinear static",
            follower_pressure=True,
        )
    )

    assert "stability pencil" in reason
    assert "'static only' or 'nonlinear static'" in reason


def test_runtime_reuses_one_bounded_session_for_linear_prestress_and_buckling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = {
        "plot_type": "flat",
        "plot_grid": [],
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [1.0, 0.0, 0.0]},
            {"id": 3, "coords": [1.0, 1.0, 0.0]},
            {"id": 4, "coords": [0.0, 1.0, 0.0]},
        ],
        "shells": [
            {
                "id": 1,
                "node_ids": [1, 2, 3, 4],
                "thickness": 0.01,
                "role": "skin",
            }
        ],
        "beams": [],
    }
    model = anystructure_fem_mode.build_fe_model_from_generated_geometry(generated)
    captured: dict[str, object] = {}

    class RecordingSession:
        def __init__(self, owned_model):
            captured["owned_model"] = owned_model
            self.closed = False

        def diagnostics(self):
            return {"plan_reused": True, "stiffness_builds": 1}

        def close(self):
            self.closed = True
            captured["closed"] = True

    def fake_linear(model_arg, _load_case, **kwargs):
        captured["linear_session"] = kwargs.get("session")
        return np.zeros(model_arg.mesh.dof_manager.total_dofs), {
            "convergence_info": {
                "status": "converged",
                "backend": {"backend": "test_double"},
            },
            "constraint_method": "test_double",
            "constraint_mode": "test_double",
        }

    def fake_buckling(_model_arg, _states, **kwargs):
        captured["buckling_session"] = kwargs.get("session")
        return SimpleNamespace(
            modes=(),
            failed=False,
            status="completed",
            diagnostics={},
        )

    monkeypatch.setattr(runtime, "_backend_AnalysisSession", RecordingSession)
    monkeypatch.setattr(runtime, "_backend_solve_linear", fake_linear)
    monkeypatch.setattr(runtime, "_backend_solve_buckling", fake_buckling)

    result = runtime.run_production_fem(
        _flat_geometry(),
        runtime.LightweightFEMConfig(
            runtime_solver="stepwise",
            analysis_type="linear eigenvalue",
            num_buckling_modes=1,
        ),
        imported_fem_model=model,
        precomputed_generated_geometry=generated,
    )

    assert result.status == "ok"
    assert captured["owned_model"] is model
    assert captured["linear_session"] is captured["buckling_session"]
    assert captured["closed"] is True
    assert result.prestress_summary["analysis_session"]["plan_reused"] is True
    assert any("Reused one analysis session" in item for item in result.diagnostics)


def test_runtime_session_reuse_excludes_state_changing_paths() -> None:
    assert runtime._can_reuse_linear_buckling_session(
        runtime.LightweightFEMConfig(
            runtime_solver="stepwise",
            analysis_type="linear eigenvalue",
        )
    )
    assert not runtime._can_reuse_linear_buckling_session(
        runtime.LightweightFEMConfig(
            runtime_solver="nonlinear static",
            analysis_type="geometric nonlinear static",
        )
    )
    assert not runtime._can_reuse_linear_buckling_session(
        runtime.LightweightFEMConfig(collision_enabled=True)
    )


def test_runtime_analysis_context_admits_reference_elastic_spectral_only() -> None:
    spectral = runtime.LightweightFEMConfig(
        runtime_solver="stepwise",
        analysis_type="linear eigenvalue",
        include_end_lids=False,
    )

    assert runtime._runtime_context_eligible(
        spectral,
        {"geometry": "flat panel"},
    )
    assert not runtime._runtime_context_eligible(
        replace(spectral, added_mass_kg=10.0),
        {"geometry": "flat panel"},
    )
    assert not runtime._runtime_context_eligible(
        replace(spectral, include_end_lids=True),
        {"geometry": "cylinder"},
    )


def test_runtime_analysis_context_reuses_model_and_factor_for_load_only_change() -> None:
    geometry = {
        "geometry": "flat panel",
        "length_m": 0.4,
        "width_m": 0.2,
        "thickness_m": 0.01,
        "has_stiffener": False,
        "has_girder": False,
    }
    first_config = runtime.LightweightFEMConfig(
        mesh_fidelity="coarse",
        runtime_solver="static only",
        pressure_pa=1_000.0,
        include_stiffeners=False,
        include_girders=False,
    )
    generated = runtime.build_generated_geometry(geometry, first_config)
    context = runtime.RuntimeAnalysisContext()
    try:
        first = runtime.run_production_fem(
            geometry,
            first_config,
            precomputed_generated_geometry=generated,
            analysis_context=context,
        )
        second = runtime.run_production_fem(
            geometry,
            replace(first_config, pressure_pa=2_000.0),
            precomputed_generated_geometry=generated,
            analysis_context=context,
        )

        assert first.status == second.status == "ok"
        assert second.displacement_max_m == pytest.approx(
            2.0 * first.displacement_max_m,
            rel=1.0e-10,
        )
        diagnostics = context.diagnostics()
        assert diagnostics["hits"] == 1
        assert diagnostics["misses"] == 1
        assert diagnostics["analysis_session"]["factorization_cache"]["hits"] >= 1
    finally:
        context.close()

    assert context.diagnostics()["active"] is False


def test_runtime_analysis_context_invalidates_for_structural_change() -> None:
    geometry = {
        "geometry": "flat panel",
        "length_m": 0.4,
        "width_m": 0.2,
        "thickness_m": 0.01,
        "has_stiffener": False,
        "has_girder": False,
    }
    config = runtime.LightweightFEMConfig(
        mesh_fidelity="coarse",
        runtime_solver="static only",
        pressure_pa=1_000.0,
        include_stiffeners=False,
        include_girders=False,
    )
    generated = runtime.build_generated_geometry(geometry, config)
    context = runtime.RuntimeAnalysisContext()
    try:
        first = runtime.run_production_fem(
            geometry,
            config,
            precomputed_generated_geometry=generated,
            analysis_context=context,
        )
        changed = runtime.run_production_fem(
            geometry,
            replace(config, poisson_ratio=0.27),
            precomputed_generated_geometry=generated,
            analysis_context=context,
        )

        assert first.status == changed.status == "ok"
        diagnostics = context.diagnostics()
        assert diagnostics["hits"] == 0
        assert diagnostics["misses"] == 2
        assert diagnostics["invalidations"] >= 1
    finally:
        context.close()


def test_runtime_performs_one_shared_recovery_for_accepted_linear_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = runtime._runtime_display_stresses

    def recording_recovery(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_runtime_display_stresses", recording_recovery)
    result = runtime.run_production_fem(
        {
            "geometry": "flat panel",
            "length_m": 0.4,
            "width_m": 0.2,
            "thickness_m": 0.01,
            "has_stiffener": False,
            "has_girder": False,
        },
        runtime.LightweightFEMConfig(
            mesh_fidelity="coarse",
            runtime_solver="static only",
            pressure_pa=1_000.0,
            include_stiffeners=False,
            include_girders=False,
        ),
    )

    assert result.status == "ok"
    assert calls == 1
    assert result.prestress_summary["performance"]["shared_recovery_evaluations"] == 1


def test_geometric_nonlinear_elastic_runtime_does_not_claim_plastic_history() -> None:
    result = runtime.run_production_fem(
        {
            "geometry": "flat panel",
            "length_m": 0.6,
            "width_m": 0.3,
            "thickness_m": 0.01,
            "has_stiffener": False,
            "has_girder": False,
        },
        runtime.LightweightFEMConfig(
            mesh_fidelity="coarse",
            runtime_solver="nonlinear static",
            analysis_type="geometric nonlinear static",
            material_model="linear elastic",
            pressure_pa=1_000.0,
            nonlinear_steps=2,
            num_buckling_modes=1,
        ),
    )

    assert result.status == "ok"
    assert not any(
        "elastoplastic states" in message for message in result.diagnostics
    )
    assert (
        result.prestress_summary["stress_recovery"]["mode"]
        == "material_history"
    )
    assert result.prestress_summary["performance"]["shared_recovery_evaluations"] == 2
    assert "mixed_reconstruction_peak_von_mises_pa" not in result.prestress_summary


def test_runtime_bridges_generated_loads_and_corotational_arc_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = {
        "plot_type": "flat",
        "plot_grid": [],
        "nodes": [
            {"id": 1, "coords": [0.0, 0.0, 0.0]},
            {"id": 2, "coords": [2.0, 0.0, 0.0]},
            {"id": 3, "coords": [2.0, 1.0, 0.0]},
            {"id": 4, "coords": [0.0, 1.0, 0.0]},
        ],
        "shells": [
            {
                "id": 1,
                "node_ids": [1, 2, 3, 4],
                "thickness": 0.02,
                "role": "skin",
            }
        ],
        "beams": [],
    }
    model = anystructure_fem_mode.build_fe_model_from_generated_geometry(generated)
    captured: dict[str, object] = {}

    def fake_linear(model_arg, load_case, **_kwargs):
        captured["linear_load_case"] = load_case
        return np.zeros(model_arg.mesh.dof_manager.total_dofs), {
            "convergence_info": {
                "status": "converged",
                "backend": {"backend": "test_double"},
            },
            "constraint_method": "test_double",
            "constraint_mode": "test_double",
        }

    def fake_arc_length(model_arg, load_case, **kwargs):
        captured["arc_load_case"] = load_case
        captured["arc_kwargs"] = kwargs
        displacements = np.zeros(model_arg.mesh.dof_manager.total_dofs)
        return SimpleNamespace(
            status="completed",
            converged=True,
            capacity_estimate=0.5,
            displacements=displacements,
            element_states={},
            steps=(),
            info={"kinematics": kwargs["kinematics"], "num_layers": 5},
            peak_load_factor=0.5,
            peak_step_index=None,
        )

    monkeypatch.setattr(runtime, "_backend_solve_linear", fake_linear)
    monkeypatch.setattr(
        runtime,
        "_backend_solve_static_arc_length",
        fake_arc_length,
    )
    config = runtime.LightweightFEMConfig(
        runtime_solver="static only",
        analysis_type="geometric nonlinear static",
        nonlinear_solution_control="arc length",
        nonlinear_static_kinematics="corotational",
        pressure_pa=1_000.0,
        follower_pressure=True,
        shear_force_n=321.0,
        torsional_moment_nm=654.0,
    )

    result = runtime.run_production_fem(
        _flat_geometry(),
        config,
        imported_fem_model=model,
        precomputed_generated_geometry=generated,
    )

    assert result.status == "ok"
    load_case = captured["arc_load_case"]
    assert load_case is captured["linear_load_case"]
    assert load_case.follower_pressure is True
    nodal_loads = np.vstack(list(load_case.nodal_loads.values()))
    assert np.max(np.abs(nodal_loads[:, 1])) > 0.0
    assert np.max(np.abs(nodal_loads[:, 3])) > 0.0
    assert captured["arc_kwargs"]["kinematics"] == "corotational"
    assert (
        result.prestress_summary["nonlinear_static_kinematics"]
        == "corotational"
    )
