"""Closed-world constructor policy for the Q1M functional burn-in lane."""

from __future__ import annotations

import ast
import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _gate_module():
    path = ROOT / "scripts" / "run_e4_pl_burnin_gate.py"
    spec = importlib.util.spec_from_file_location("e4_pl_burnin_gate_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enclosing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return "<module>"


def _constructor_calls(
    paths: list[Path],
) -> dict[str, Counter[tuple[str, str]]]:
    calls = {
        "ShellElement": Counter(),
        "LegacyShellElement": Counter(),
        "QualifiedE4PLShellElement": Counter(),
    }
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        aliases = {name: {name} for name in calls}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for imported in node.names:
                if imported.name in aliases:
                    aliases[imported.name].add(imported.asname or imported.name)
        relative = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            for constructor, local_names in aliases.items():
                if called_name in local_names:
                    calls[constructor][
                        (relative, _enclosing_function(node, parents))
                    ] += 1
                    break
    return calls


def test_functional_lane_direct_shell_calls_are_non_q4_or_explicit_refusal() -> None:
    inventory = _gate_module().inventory()
    paths = [ROOT / relative for relative in inventory["functional"]]
    calls = _constructor_calls(paths)

    expected_non_q4 = Counter(
        {
            # Added recovery test exercises Q8, never legacy Q4.
            ("tests/test_recovery_shell_frames.py", "model8"): 1,
            ("tests/test_fe_solver_contact.py", "_q8_panel"): 1,
            ("tests/test_fe_solver_contact.py", "_tri_panel"): 1,
            (
                "tests/test_fe_solver_element_qualification.py",
                "test_q8r_hourglass_cache_is_invalidated_after_geometry_change",
            ): 1,
            (
                "tests/test_fe_solver_theory.py",
                "test_shell_shape_functions_partition_unity",
            ): 1,
            ("tests/test_fe_solver_triangular_shell.py", "_tri_model"): 1,
            (
                "tests/test_fe_solver_triangular_shell.py",
                "test_triangular_shell_shape_functions_interpolate_and_reproduce_fields",
            ): 1,
            (
                "tests/test_fe_solver_triangular_shell.py",
                "test_triangular_aliases_and_mixed_q4_t3_assembly",
            ): 3,
            ("tests/test_fe_solver_triangular_shell_backend.py", "_tri_model"): 1,
            (
                "tests/test_fe_solver_triangular_shell_backend.py",
                "test_triangular_shell_shape_functions_interpolate_and_reproduce_fields",
            ): 1,
            (
                "tests/test_fe_solver_triangular_shell_backend.py",
                "test_triangular_aliases_and_mixed_q4_t3_assembly",
            ): 3,
            ("tests/test_follower_pressure.py", "_single_shell"): 1,
            (
                "tests/test_ge_beam3_g3b_mq4.py",
                "test_rejection_precedes_both_element_operators",
            ): 1,
            (
                "tests/test_ge_beam3_g3b_ms3.py",
                "test_rejection_precedes_both_element_operators",
            ): 1,
            (
                "tests/test_ge_beam3_g3b_weighted.py",
                "test_rejection_precedes_both_element_operators",
            ): 1,
            ("tests/test_generalized_shell_sections.py", "_model_with_shell"): 1,
            ("tests/test_orthotropic_elements.py", "_shell_topology"): 1,
            (
                "tests/test_production_readiness.py",
                "test_production_validation_marks_q8r_as_experimental",
            ): 1,
            (
                "tests/test_production_validation.py",
                "test_material_construction_fails_fast_and_validation_rejects_invalid_thickness",
            ): 1,
            (
                "tests/test_production_validation.py",
                "test_validate_production_model_reports_q8_midside_and_warp_warnings",
            ): 1,
            (
                "tests/test_production_validation.py",
                "test_validate_production_model_accepts_triangular_shell_quality",
            ): 1,
            (
                "tests/test_recovery_qualification.py",
                "test_patch_rejects_reduced_q8_and_warped_q4_outside_qualified_scope",
            ): 1,
            # Submodelling S1: three-node shells the transfer must refuse.
            (
                "tests/test_submodel_transfer.py",
                "test_interpolation_refuses_convention_mismatch_and_unadmitted_elements",
            ): 1,
            (
                "tests/test_submodel_transfer.py",
                "test_unadmitted_target_shell_and_invalid_controls_refuse",
            ): 1,
        }
    )
    # This explicit negative constructs legacy Q4 only to verify rejection by
    # the qualified material-checkpoint route; it is not a Q4 solve admission.
    expected_refused_q4 = Counter({
        ("tests/test_general_contact_q4_material_checkpoint.py",
         "test_only_exact_q4_and_finite_reference_geometry_are_admitted"): 1})
    assert calls["ShellElement"] == expected_non_q4 + expected_refused_q4
    expected_qualified_q4 = Counter(
        {
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "test_b_coupled_section_requires_persistent_director_and_restart_identity",
            ): 3,
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "test_recovery_is_d4_and_proper_global_covariant_with_physical_material_direction",
            ): 3,
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "test_numerical_fields_and_nonfinite_values_cannot_enter_planar_recovery",
            ): 2,
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "test_authoritative_director_keeps_isotropic_top_and_bottom_physical",
            ): 2,
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "test_stationary_equilibrium_and_recovered_resultants_close_virtual_work",
            ): 1,
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "test_combined_patch_recovery_is_exact_at_gauss_and_arbitrary_common_points",
            ): 1,
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "test_generalized_section_keeps_resultants_only_schema_with_mixed_fields",
            ): 1,
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "test_reflected_abd_congruence_uses_exact_engineering_field_maps",
            ): 1,
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "make",
            ): 1,
            (
                "tests/test_e4_pl_planar_physical_recovery.py",
                "test_qualified_q4_fingerprint_cannot_downgrade_to_legacy_without_formulation_id",
            ): 1,
            (
                "tests/test_e4_pl_q4_stationary_conditioning.py",
                "_stationary_case",
            ): 1,
            (
                "tests/test_e4_pl_component_snapshot_integrity.py",
                "_q4_case",
            ): 1,
            (
                "tests/test_e4_pl_guarded_observations.py",
                "_qualified_case",
            ): 1,
            (
                "tests/test_e4_pl_guarded_observations.py",
                "test_exact_generalized_section_constructor_remains_supported",
            ): 1,
            (
                "tests/test_e4_pl_current_state_input_ownership.py",
                "_model_and_zero_state",
            ): 1,
            (
                "tests/test_e4_pl_mixed_current_state_route.py",
                "_mixed_model",
            ): 1,
            (
                "tests/test_e4_pl_mixed_current_state_route.py",
                "test_route_and_state_binding_mutations_fail_before_component_mechanics",
            ): 1,
            (
                "tests/test_e4_pl_orchestrator_operation_lease.py",
                "_two_q4_transient_model",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_q4_binding_rejects_every_configuration_and_state_mismatch",
            ): 2,
            ("tests/test_e4_pl_q4_current_tangent.py", "evaluate"): 2,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_layered_q4_components_are_independent_read_only_and_newton_exact",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_zero_state_has_exact_zero_stress_hessian_and_correction_is_material_only",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_q4_stress_hessian_matches_direct_gw_n_gw_and_reverses_with_stress",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_plastic_q4_origin_is_required_and_reproduces_the_accepted_core",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_q4_layered_state_boundaries_reject_unsupported_lobatto_counts",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_q4_admits_exact_supported_lobatto_layer_set",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_layered_plastic_q4_component_split_uses_algorithmic_material_tangent",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_isotropic_initial_field_and_orthotropic_hill_plastic_updates_are_refused",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_residual_only_plastic_candidate_cannot_masquerade_as_accepted_tangent",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_q4_seal_rejects_stored_generalized_kinematics_from_another_state",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_vectorized_q4_accepted_tangent_matches_sealed_scalar_replay_exactly",
            ): 1,
            (
                "tests/test_e4_pl_q4_current_tangent.py",
                "test_distorted_vectorized_q4_plastic_history_replays_after_multiple_increments",
            ): 1,
            ("tests/test_e4_pl_q4_state_lifecycle.py", "_loaded_q4_model"): 1,
            (
                "tests/test_e4_pl_q4_state_lifecycle.py",
                "test_fully_constrained_generalized_q4_is_materialized_and_sealed",
            ): 1,
            (
                "tests/test_e4_pl_q4_state_lifecycle.py",
                "test_fully_constrained_arc_q4_is_materialized_and_exactly_sealed",
            ): 1,
            (
                "tests/test_e4_pl_q4_state_lifecycle.py",
                "test_unknown_id_q4_descendant_cannot_downgrade_checkpoint_authority",
            ): 1,
            ("tests/test_e4_pl_q4_state_lifecycle.py", "model"): 1,
            (
                "tests/test_e4_pl_transient_authority.py",
                "_qualified_q4_model",
            ): 1,
            ("tests/test_ge_beam3_g3b_mq4.py", "parts"): 1,
            ("tests/test_ge_beam3_g3b_weighted.py", "parts"): 1,
            (
                "tests/test_mixed_shell_quadrature_grouping.py",
                "test_direct_qualified_q4_connectivity_change_invalidates_warm_sparsity",
            ): 1,
            (
                "tests/test_mixed_shell_quadrature_grouping.py",
                "_mixed_model",
            ): 1,
            (
                "tests/test_qualified_mutation_epoch.py",
                "_qualified_q4_model",
            ): 1,
            (
                "tests/test_qualified_mutation_epoch.py",
                "test_stress_recovery_uses_three_full_boundaries_not_two_per_element",
            ): 1,
            (
                "tests/test_qualified_mutation_epoch.py",
                "test_mixed_stress_recovery_bounds_qualified_checks_and_keeps_generic_boundary",
            ): 1,
            (
                "tests/test_qualified_mutation_epoch.py",
                "test_alternating_q4_beam_recovery_keeps_one_bracket_and_rejects_mutation",
            ): 1,
            (
                "tests/test_qualified_mutation_epoch.py",
                "test_large_alternating_q4_beam_recovery_has_constant_full_guard_count",
            ): 1,
            ("tests/test_qualified_q4_assembly_authority.py", "_model"): 1,
            ("tests/test_spectral_validation_preparation.py", "_q4"): 1,
        }
    )
    expected_qualified_q4.update({
        ("tests/test_arc_length_predictor_reuse.py", "panel"): 1,
        ("tests/test_q4_assumed_strain_plasticity.py", "test_hill48_plastic_q4_is_refused_with_typed_error"): 1,
        ("tests/test_wave_surface_regions.py", "test_qualified_formulations_match_their_native_dead_pressure_or_are_refused"): 1,
        ("tests/test_b2_f10_f11_point_attachment.py", "specimen"): 1,
        ("tests/test_stiffened_shell_b2_nodal_rigid_footprint.py", "_independent"): 1,
        ("tests/test_stiffened_shell_b2_physical_rigid_footprint.py", "test_distorted_skew_frozen_shell_tensor_frame_and_scalar_work"): 1,
        ("tests/test_stiffened_shell_b2_q4_panel_experimental.py", "_independent_dense"): 1,
        # Prepared static evaluator: staged qualified Q4 fixture.
        ("tests/test_prepared_static_q4.py", "_q4_model"): 1,
        ("tests/test_prepared_static_q4_compensated_policy.py", "test_original_planar_coordinate_policy_parity_and_source_deletion"): 1,
    })
    assert calls["QualifiedE4PLShellElement"] == expected_qualified_q4
    assert sum(calls["QualifiedE4PLShellElement"].values()) == sum(
        expected_qualified_q4.values()
    )


def test_functional_lane_legacy_q4_calls_are_explicit_and_closed_world() -> None:
    inventory = _gate_module().inventory()
    paths = [ROOT / relative for relative in inventory["functional"]]
    calls = _constructor_calls(paths)

    expected_legacy = Counter(
        {
            ("tests/test_corotational.py", "_single_legacy_shell_model"): 1,
            ("tests/test_follower_pressure.py", "_clamped_pressure_plate"): 1,
            ("tests/test_follower_pressure.py", "_legacy_q4_shell"): 1,
            ("tests/test_generalized_shell_sections.py", "_legacy_shell"): 2,
            (
                "tests/test_recovery_qualification.py",
                "test_patch_rejects_reduced_q8_and_warped_q4_outside_qualified_scope",
            ): 1,
            (
                "tests/test_s3_v2d_default_activation.py",
                "test_direct_and_explicit_legacy_routes_remain_available",
            ): 1,
        }
    )
    assert calls["LegacyShellElement"] == expected_legacy
    assert sum(calls["LegacyShellElement"].values()) == 7


def test_production_constructor_calls_are_confined_to_central_factory() -> None:
    paths = sorted((ROOT / "src" / "anysolver").rglob("*.py"))
    calls = _constructor_calls(paths)
    assert calls["ShellElement"] == Counter()
    assert calls["LegacyShellElement"] == Counter(
        {("src/anysolver/elements.py", "create_shell_element"): 1}
    )
    assert calls["QualifiedE4PLShellElement"] == Counter(
        {("src/anysolver/elements.py", "create_shell_element"): 1}
    )
