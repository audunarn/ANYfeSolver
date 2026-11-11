"""Qualification tests for incremental impact-damage matrix updates."""

from __future__ import annotations

import numpy as np

from anyfesolver.contact import (
    _assemble_damaged_linear_matrices,
    _assemble_damaged_linear_matrices_with_gate,
    _linear_element_matrix_terms,
)
from anyfesolver.damage_matrix_performance import (
    DAMAGE_MATRIX_PLAN_BREAK_EVEN_FUTURE_UPDATES,
    DamageMatrixPlan,
    DamageMatrixPlanGate,
    estimate_damage_matrix_cached_terms_retained_bytes,
    estimate_damage_matrix_combined_retained_bytes,
    estimate_damage_matrix_plan_retained_bytes,
)
from anyfesolver.elements import create_element
from anyfesolver.fe_core import FEModel


def _panel(nx: int = 2, ny: int = 2) -> FEModel:
    model = FEModel("damage_matrix_panel")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    node_ids: dict[tuple[int, int], int] = {}
    node_id = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            model.add_node(node_id, float(i), float(j), 0.0)
            node_ids[(i, j)] = node_id
            node_id += 1
    element_id = 1
    for j in range(ny):
        for i in range(nx):
            model.add_element(
                element_id,
                create_element(
                    "shell",
                    element_id,
                    [
                        node_ids[(i, j)],
                        node_ids[(i + 1, j)],
                        node_ids[(i + 1, j + 1)],
                        node_ids[(i, j + 1)],
                    ],
                    "steel",
                    thickness=0.02,
                ),
            )
            element_id += 1
    model.add_point_mass(node_ids[(1, 1)], 7.25)
    return model


def _assert_matrix_parity(candidate, reference) -> None:
    np.testing.assert_allclose(
        candidate.toarray(),
        reference.toarray(),
        rtol=5.0e-14,
        atol=1.0e-8,
    )


def test_incremental_damage_sequence_matches_scalar_and_keeps_csr_identity() -> None:
    model = _panel()
    terms = _linear_element_matrix_terms(model)
    plan = DamageMatrixPlan.build(model, terms)
    stiffness_identity = (id(plan.stiffness), id(plan.stiffness.data), id(plan.stiffness.indices))
    mass_identity = (id(plan.mass), id(plan.mass.data), id(plan.mass.indices))
    sequences = (
        {},
        {1: 0.75},
        {1: 0.5, 3: 0.0},
        {3: 0.25},  # Element 1 is omitted and must reset to an exact scale of one.
        {1: 1.4, 2: -0.2, 4: 0.6},  # Preserve the legacy [0, 1] scale clamp.
        {},
        {4: 0.6},
        {4: 0.6},  # No-change update must not touch matrix storage.
    )

    for scales in sequences:
        reference_k, reference_m = _assemble_damaged_linear_matrices(model, scales, cached_terms=terms)
        candidate_k, candidate_m = _assemble_damaged_linear_matrices(model, scales, cached_terms=plan)
        assert candidate_k is plan.stiffness
        assert candidate_m is plan.mass
        assert (id(candidate_k), id(candidate_k.data), id(candidate_k.indices)) == stiffness_identity
        assert (id(candidate_m), id(candidate_m.data), id(candidate_m.indices)) == mass_identity
        _assert_matrix_parity(candidate_k, reference_k)
        _assert_matrix_parity(candidate_m, reference_m)

    diagnostics = plan.diagnostics()
    assert diagnostics["fast_path_name"] == "incremental_damage_csr_updates"
    assert diagnostics["eligible_element_count"] == len(model.mesh.elements)
    assert diagnostics["update_count"] == len(sequences)
    assert diagnostics["no_change_count"] >= 2
    assert diagnostics["changed_element_count"] > 0
    assert diagnostics["fallback_count"] == 0
    assert diagnostics["retained_bytes"] > 0


def test_zero_element_scales_preserve_point_mass_and_reset_to_one() -> None:
    model = _panel()
    terms = _linear_element_matrix_terms(model)
    plan = DamageMatrixPlan.build(model, terms)
    all_zero = {element_id: 0.0 for element_id in model.mesh.elements}

    zero_k, zero_m = plan.update(model, all_zero)
    reference_zero_k, reference_zero_m = _assemble_damaged_linear_matrices(
        model,
        all_zero,
        cached_terms=terms,
    )
    _assert_matrix_parity(zero_k, reference_zero_k)
    _assert_matrix_parity(zero_m, reference_zero_m)
    center = model.mesh.get_node(5)
    assert center is not None
    np.testing.assert_allclose(
        zero_m.diagonal()[np.asarray(center.dofs[:3], dtype=int)],
        np.full(3, 7.25),
        rtol=0.0,
        atol=1.0e-12,
    )

    reset_k, reset_m = plan.update(model, {})
    reference_k, reference_m = _assemble_damaged_linear_matrices(model, {}, cached_terms=terms)
    _assert_matrix_parity(reset_k, reference_k)
    _assert_matrix_parity(reset_m, reference_m)
    assert plan.diagnostics()["active_scaled_element_count"] == 0


def test_model_revision_invalidates_plan_and_helper_falls_back_to_scalar() -> None:
    model = _panel()
    plan = DamageMatrixPlan.build(model, _linear_element_matrix_terms(model))
    plan.update(model, {1: 0.5})
    model.bump_revision("geometry")

    fallback_k, fallback_m = _assemble_damaged_linear_matrices(
        model,
        {1: 0.4, 2: 0.8},
        cached_terms=plan,
    )
    current_terms = _linear_element_matrix_terms(model)
    reference_k, reference_m = _assemble_damaged_linear_matrices(
        model,
        {1: 0.4, 2: 0.8},
        cached_terms=current_terms,
    )

    assert fallback_k is not plan.stiffness
    assert fallback_m is not plan.mass
    _assert_matrix_parity(fallback_k, reference_k)
    _assert_matrix_parity(fallback_m, reference_m)
    diagnostics = plan.diagnostics()
    assert diagnostics["invalidation_count"] == 1
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["fallback_reason"] == "model_revision_changed"

    replacement = DamageMatrixPlan.build(model, current_terms)
    replacement_k, replacement_m = replacement.update(model, {1: 0.4, 2: 0.8})
    _assert_matrix_parity(replacement_k, reference_k)
    _assert_matrix_parity(replacement_m, reference_m)


def test_plan_gate_requires_observed_future_break_even_and_bounds_memory() -> None:
    model = _panel()
    terms = _linear_element_matrix_terms(model)
    plan_estimate = estimate_damage_matrix_plan_retained_bytes(terms)
    cached_terms_bytes = estimate_damage_matrix_cached_terms_retained_bytes(terms)
    estimate = estimate_damage_matrix_combined_retained_bytes(terms)
    assert estimate == plan_estimate + cached_terms_bytes
    plan = DamageMatrixPlan.build(model, terms)
    assert plan_estimate >= plan.retained_bytes

    gate = DamageMatrixPlanGate(
        total_opportunities=40,
        preflight_memory_bytes=10_000,
    )
    gate.register_cached_terms(model)
    assert gate.consider(terms, opportunity_index=1) is False
    assert gate.diagnostics()["selection_reason"] == "insufficient_observed_update_events"
    assert gate.consider(terms, opportunity_index=2) is True
    diagnostics = gate.diagnostics()
    assert diagnostics["projected_future_update_events"] >= DAMAGE_MATRIX_PLAN_BREAK_EVEN_FUTURE_UPDATES
    assert diagnostics["retained_memory_allowance_bytes"] > estimate
    assert diagnostics["cached_terms_retained_bytes"] == cached_terms_bytes
    assert diagnostics["estimated_plan_retained_bytes"] == plan_estimate
    assert diagnostics["estimated_combined_retained_bytes"] == estimate
    assert diagnostics["estimated_retained_bytes"] == estimate

    late_gate = DamageMatrixPlanGate(total_opportunities=20, preflight_memory_bytes=0)
    late_gate.register_cached_terms(model)
    assert late_gate.consider(terms, opportunity_index=9) is False
    assert late_gate.consider(terms, opportunity_index=10) is False
    late_diagnostics = late_gate.diagnostics()
    assert late_diagnostics["projected_future_update_events"] < DAMAGE_MATRIX_PLAN_BREAK_EVEN_FUTURE_UPDATES
    assert late_diagnostics["selection_reason"] == "projected_future_updates_below_break_even"

    memory_gate = DamageMatrixPlanGate(
        total_opportunities=40,
        preflight_memory_bytes=1_000_000,
        configured_memory_limit_bytes=1_000_000 + estimate - 1,
    )
    memory_gate.register_cached_terms(model)
    assert memory_gate.consider(terms, opportunity_index=1) is False
    assert memory_gate.consider(terms, opportunity_index=2) is False
    memory_diagnostics = memory_gate.diagnostics()
    assert memory_diagnostics["retained_memory_allowance_bytes"] == estimate - 1
    assert memory_diagnostics["selection_reason"] == "estimated_retained_memory_exceeds_allowance"


def test_plan_gate_charges_cached_fallback_terms_to_tiny_panel_headroom() -> None:
    """Regression: plan-only headroom cannot cover two retained representations."""

    model = _panel()
    terms = _linear_element_matrix_terms(model)
    plan_only_estimate = estimate_damage_matrix_plan_retained_bytes(terms)
    cached_terms_bytes = estimate_damage_matrix_cached_terms_retained_bytes(terms)
    plan = DamageMatrixPlan.build(model, terms)

    # This is the audit reproduction: on this four-shell panel the legacy
    # arrays retain 73,728 bytes beside a 144,808-byte plan.  The former gate
    # saw only its conservative 192,512-byte plan estimate and selected it.
    assert (cached_terms_bytes, plan.retained_bytes, plan_only_estimate) == (
        73_728,
        144_808,
        192_512,
    )
    assert cached_terms_bytes == sum(
        int(np.asarray(array).nbytes)
        for term in terms[1]
        for array in term[1:]
    )
    assert cached_terms_bytes > 0
    assert plan.retained_bytes + cached_terms_bytes > plan_only_estimate

    preflight_bytes = 1_000_000
    gate = DamageMatrixPlanGate(
        total_opportunities=40,
        preflight_memory_bytes=preflight_bytes,
        configured_memory_limit_bytes=preflight_bytes + plan_only_estimate,
    )
    gate.register_cached_terms(model)
    assert gate.consider(terms, opportunity_index=1) is False
    assert gate.consider(terms, opportunity_index=2) is False

    diagnostics = gate.diagnostics()
    assert diagnostics["retained_memory_allowance_bytes"] == plan_only_estimate
    assert diagnostics["cached_terms_retained_bytes"] == cached_terms_bytes
    assert diagnostics["estimated_plan_retained_bytes"] == plan_only_estimate
    assert diagnostics["estimated_combined_retained_bytes"] == (
        plan_only_estimate + cached_terms_bytes
    )
    assert diagnostics["estimated_combined_retained_bytes"] > diagnostics[
        "retained_memory_allowance_bytes"
    ]
    assert diagnostics["accounted_combined_retained_bytes"] == diagnostics[
        "estimated_combined_retained_bytes"
    ]
    assert diagnostics["plan_build_count"] == 0
    assert diagnostics["selection_reason"] == "estimated_retained_memory_exceeds_allowance"

    default_cap_gate = DamageMatrixPlanGate(
        total_opportunities=40,
        preflight_memory_bytes=0,
        default_retained_bytes_cap=plan_only_estimate,
    )
    default_cap_gate.register_cached_terms(model)
    assert default_cap_gate.consider(terms, opportunity_index=1) is False
    assert default_cap_gate.consider(terms, opportunity_index=2) is False
    assert default_cap_gate.diagnostics()["selection_reason"] == (
        "estimated_retained_memory_exceeds_allowance"
    )


def test_plan_gate_disables_setup_after_cached_term_revision_refresh() -> None:
    model = _panel()
    terms = _linear_element_matrix_terms(model)
    gate = DamageMatrixPlanGate(total_opportunities=40, preflight_memory_bytes=0)
    gate.register_cached_terms(model)
    assert gate.consider(terms, opportunity_index=1) is False
    model.bump_revision("geometry")
    assert gate.cached_terms_match(model) is False
    refreshed = _linear_element_matrix_terms(model)
    gate.register_cached_terms(model, replacing_stale=True)
    assert gate.consider(refreshed, opportunity_index=2) is False
    diagnostics = gate.diagnostics()
    assert diagnostics["cached_terms_refresh_count"] == 1
    assert diagnostics["model_revision_fallback_count"] == 1
    assert diagnostics["selection_reason"] == "model_revision_changed_legacy_fallback"


def test_gated_matrix_helper_retains_legacy_terms_then_promotes_with_parity() -> None:
    model = _panel()
    gate = DamageMatrixPlanGate(total_opportunities=40, preflight_memory_bytes=0)
    terms = None
    plan = None

    first_scales = {1: 0.8}
    first_k, first_m, terms, plan = _assemble_damaged_linear_matrices_with_gate(
        model,
        first_scales,
        gate,
        opportunity_index=1,
        cached_terms=terms,
        plan=plan,
    )
    reference_k, reference_m = _assemble_damaged_linear_matrices(model, first_scales, cached_terms=terms)
    _assert_matrix_parity(first_k, reference_k)
    _assert_matrix_parity(first_m, reference_m)
    assert plan is None
    assert gate.diagnostics()["legacy_update_count"] == 1

    second_scales = {1: 0.6, 2: 0.9}
    second_k, second_m, retained_terms, plan = _assemble_damaged_linear_matrices_with_gate(
        model,
        second_scales,
        gate,
        opportunity_index=2,
        cached_terms=terms,
        plan=plan,
    )
    reference_k, reference_m = _assemble_damaged_linear_matrices(
        model,
        second_scales,
        cached_terms=retained_terms,
    )
    _assert_matrix_parity(second_k, reference_k)
    _assert_matrix_parity(second_m, reference_m)
    assert retained_terms is terms
    assert plan is not None
    diagnostics = gate.diagnostics()
    assert diagnostics["plan_selected"] is True
    assert diagnostics["plan_build_count"] == 1
    assert diagnostics["legacy_update_count"] == 1
    assert diagnostics["plan_update_count"] == 1
    assert diagnostics["actual_plan_retained_bytes"] == plan.retained_bytes
    assert diagnostics["actual_combined_retained_bytes"] == (
        diagnostics["cached_terms_retained_bytes"] + plan.retained_bytes
    )
    assert diagnostics["actual_retained_bytes"] == diagnostics[
        "actual_combined_retained_bytes"
    ]
    assert diagnostics["accounted_combined_retained_bytes"] >= diagnostics[
        "actual_combined_retained_bytes"
    ]
