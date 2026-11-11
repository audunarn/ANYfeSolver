from types import SimpleNamespace

import numpy as np
import pytest

import anyfesolver as fs
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.contact import _verification_contact_panel
from anyfesolver.elements import create_element
from anyfesolver.fe_core import FEModel
from anyfesolver.impact_performance import ImpactTangentReuseController
from anyfesolver.material_curves import DNVC208MaterialCurve


class _DummyHandle:
    pass


def _contact(element_id: int, classification: str = "face") -> SimpleNamespace:
    return SimpleNamespace(element_id=element_id, contact_classification=classification)


def test_reuse_controller_refreshes_on_all_observable_invalidation_signals():
    controller = ImpactTangentReuseController(2)
    controller.set_initial_contact(())
    controller.begin_substep(0.01, {}, ())

    refresh, reasons = controller.refresh_decision()
    assert refresh is True
    assert "first_iteration" in reasons
    controller.record_tangent_assembly(reasons)
    controller.record_factorization(_DummyHandle())

    refresh, reasons = controller.refresh_decision()
    assert refresh is False
    assert reasons == ()
    controller.record_reuse()
    controller.record_reuse()
    refresh, reasons = controller.refresh_decision()
    assert refresh is True
    assert "reuse_budget_exhausted" in reasons

    assert controller.observe_contact((_contact(4),)) == ("active_contact_set_change",)
    assert controller.observe_contact((_contact(4, "edge"),)) == ("contact_classification_change",)
    controller.observe_line_search(0.5)
    refresh, reasons = controller.refresh_decision()
    assert refresh is True
    assert {
        "active_contact_set_change",
        "contact_classification_change",
        "aggressive_line_search",
    }.issubset(reasons)

    controller.record_tangent_assembly(reasons)
    controller.record_factorization(_DummyHandle())
    assert controller.assess_trial_state(10.0, {1: {"alpha": np.zeros(2)}}) == ()
    dynamic = controller.assess_trial_state(9.5, {1: {"alpha": np.array([0.0, 0.01])}})
    assert "residual_stall" in dynamic
    assert "plastic_active_set_change" in dynamic

    controller.begin_substep(0.005, {1: 0.8}, (1,))
    refresh, reasons = controller.refresh_decision()
    assert refresh is True
    assert {"time_step_change", "damage_scale_change", "deletion_change"}.issubset(reasons)


def _run_rebound_case(reuse_iterations: int):
    return fs.solve_transient_sphere_impact(
        _verification_contact_panel(),
        fs.TransientConfig(dt=0.0025, t_end=0.04),
        fs.RigidSphereImpact(
            "tangent_reuse",
            radius=0.1,
            mass=1.0,
            start_point=(0.5, 0.5, 0.12),
            travel_direction=(0.0, 0.0, -1.0),
            speed=2.0,
        ),
        fs.SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        nonlinear_config=fs.NonlinearTransientConfig(
            enabled=True,
            max_iterations=15,
            max_cutbacks=4,
            tangent_reuse_iterations=reuse_iterations,
        ),
    )


def test_modified_newton_reuse_matches_legacy_history_and_reduces_factorizations():
    legacy = _run_rebound_case(0)
    reused = _run_rebound_case(2)

    assert legacy.status == reused.status == "completed"
    np.testing.assert_allclose(reused.times, legacy.times, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reused.displacements, legacy.displacements, rtol=1.0e-12, atol=1.0e-13)
    np.testing.assert_allclose(reused.velocities, legacy.velocities, rtol=1.0e-12, atol=1.0e-13)
    np.testing.assert_allclose(reused.contact_force_history, legacy.contact_force_history, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(reused.sphere_positions, legacy.sphere_positions, rtol=1.0e-12, atol=1.0e-13)
    np.testing.assert_allclose(reused.sphere_velocities, legacy.sphere_velocities, rtol=1.0e-12, atol=1.0e-13)
    assert reused.peak_contact_force == legacy.peak_contact_force
    assert reused.max_penetration == legacy.max_penetration
    assert reused.contact_duration == legacy.contact_duration
    assert reused.sphere_momentum_balance_error == legacy.sphere_momentum_balance_error

    legacy_diag = legacy.diagnostics
    reuse_diag = reused.diagnostics
    assert legacy_diag["tangent_reuse_count"] == 0
    assert legacy_diag["factorization_reuse_count"] == 0
    assert legacy_diag["tangent_assembly_count"] == legacy_diag["factorization_count"]
    assert reuse_diag["factorization_reuse_count"] > 0
    assert reuse_diag["tangent_reuse_count"] == reuse_diag["factorization_reuse_count"]
    assert reuse_diag["factorization_count"] < 0.6 * legacy_diag["factorization_count"]
    assert reuse_diag["tangent_assembly_count"] == reuse_diag["factorization_count"]
    assert reuse_diag["solve_count"] == legacy_diag["solve_count"]
    assert reuse_diag["iteration_counts"] == legacy_diag["iteration_counts"]
    assert reuse_diag["cutback_count"] == legacy_diag["cutback_count"] == 0
    assert reuse_diag["active_contact_set_changes"] >= 1
    assert "active_contact_set_change" in reuse_diag["refresh_reason_counts"]
    contact_work = reuse_diag["contact_work_buffer"]
    assert contact_work["path"] == "compact_internal_candidates_lazy_public_records"
    assert contact_work["lazy_public_materialization"] is True
    assert contact_work["direct_full_scatter_count"] == contact_work["assembly_calls"]
    assert contact_work["public_materialization_count"] < contact_work["assembly_calls"]
    assert reuse_diag["contact_public_materialization_count"] == contact_work["public_materialization_count"]
    assert reuse_diag["contact_direct_full_scatter_count"] == contact_work["direct_full_scatter_count"]


def test_zero_reuse_budget_reports_explicit_full_newton_oracle():
    controller = ImpactTangentReuseController(0)
    controller.begin_substep(0.01, {}, ())
    refresh, reasons = controller.refresh_decision()
    assert refresh is True
    assert reasons == ("legacy_full_newton",)
    assert controller.enabled is False


def _yielding_panel() -> FEModel:
    model = FEModel("impact_tangent_reuse_yielding_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    division = 4
    node_of = {}
    node_id = 1
    for j in range(division + 1):
        for i in range(division + 1):
            model.add_node(node_id, i / division, j / division, 0.0)
            node_of[(i, j)] = node_id
            node_id += 1
    element_id = 1
    for j in range(division):
        for i in range(division):
            model.add_element(
                element_id,
                create_element(
                    "shell",
                    element_id,
                    [
                        node_of[(i, j)],
                        node_of[(i + 1, j)],
                        node_of[(i + 1, j + 1)],
                        node_of[(i, j + 1)],
                    ],
                    "soft",
                    thickness=0.05,
                ),
            )
            element_id += 1
    edge_nodes = [
        node_of[(i, j)]
        for j in range(division + 1)
        for i in range(division + 1)
        if i in (0, division) or j in (0, division)
    ]
    model.add_boundary_condition(
        BoundaryCondition(
            "clamp",
            edge_nodes,
            {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.materials["soft"].hardening_curve = DNVC208MaterialCurve(
        sigma_prop=800.0,
        sigma_yield=1000.0,
        sigma_yield_2=1200.0,
        eps_p_y1=1.0e-5,
        eps_p_y2=1.0e-3,
        K=2000.0,
        n=0.1,
    )
    return model


def _run_yielding_case(reuse_iterations: int):
    return fs.solve_transient_sphere_impact(
        _yielding_panel(),
        fs.TransientConfig(dt=0.0025, t_end=0.03),
        fs.RigidSphereImpact(
            "yielding_tangent_reuse",
            radius=0.2,
            mass=20.0,
            start_point=(0.5, 0.5, 0.22),
            travel_direction=(0.0, 0.0, -1.0),
            speed=4.0,
        ),
        fs.SphereContactConfig(penalty_stiffness=2000.0, max_contact_iterations=20),
        nonlinear_config=fs.NonlinearTransientConfig(
            enabled=True,
            max_iterations=15,
            max_cutbacks=4,
            tangent_reuse_iterations=reuse_iterations,
        ),
    )


def test_reuse_refreshes_for_plastic_state_changes_and_preserves_yielding_response():
    legacy = _run_yielding_case(0)
    reused = _run_yielding_case(2)
    legacy_alpha = float(legacy.diagnostics["strain_summary"]["max_equivalent_plastic_strain"])
    reused_alpha = float(reused.diagnostics["strain_summary"]["max_equivalent_plastic_strain"])

    assert legacy.status == reused.status == "completed"
    assert legacy_alpha > 0.05
    assert reused_alpha == pytest.approx(legacy_alpha, rel=1.0e-8, abs=1.0e-11)
    np.testing.assert_allclose(reused.contact_force_history, legacy.contact_force_history, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(reused.displacements, legacy.displacements, rtol=1.0e-8, atol=2.0e-10)
    assert reused.diagnostics["iteration_counts"] == legacy.diagnostics["iteration_counts"]
    assert reused.diagnostics["cutback_count"] == legacy.diagnostics["cutback_count"] == 0
    assert reused.diagnostics["factorization_count"] < legacy.diagnostics["factorization_count"]
    reasons = reused.diagnostics["refresh_reason_counts"]
    assert reasons.get("plastic_active_set_change", 0) > 0
    assert reasons.get("plastic_state_change", 0) > 0
