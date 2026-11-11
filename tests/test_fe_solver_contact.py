"""Limited rigid-sphere contact dynamics tests."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    CoupledBeamShellElement,
    ImpactDamageConfig,
    ImpactFractureConfig,
    RigidSphereImpact,
    ShellElement,
    SphereContactConfig,
    TransientConfig,
    assemble_sphere_contact_load_vector,
    recommend_sphere_contact_penalty,
    solve_transient_sphere_impact,
    validate_contact_configuration,
    create_shell_element,
)
from anyfesolver.contact import _impact_contact_patch_area, _update_impact_damage_states
from anyfesolver.recovery import RecoveryConfig
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.validation import load_vector_resultant


def _contact_panel(stiffener: bool = False) -> FEModel:
    model = FEModel("contact_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "soft", thickness=0.05)
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "restrain_shell_nonimpact_modes",
            [1, 2, 3, 4],
            {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    if stiffener:
        section = {"area": 0.05, "Iy": 1.0e-3, "Iz": 1.0e-3, "J": 1.0e-3}
        model.add_node(10, 0.5, 0.0, 0.05)
        model.add_node(11, 0.5, 1.0, 0.05)
        model.add_element(10, BeamElement(10, [10, 11], "soft", section))
        model.add_element(20, CoupledBeamShellElement(20, beam_node_id=10, shell_node_id=1, material_name="soft"))
        model.add_element(21, CoupledBeamShellElement(21, beam_node_id=11, shell_node_id=4, material_name="soft"))
    return model


def _q8_panel() -> FEModel:
    model = FEModel("q8_contact_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (0.5, 0.0, 0.0),
        6: (1.0, 0.5, 0.0),
        7: (0.5, 1.0, 0.0),
        8: (0.0, 0.5, 0.0),
    }
    for node_id, xyz in coords.items():
        model.add_node(node_id, *xyz)
    model.add_element(1, ShellElement(1, list(range(1, 9)), "soft", thickness=0.05, reduced_integration=True))
    model.add_boundary_condition(BoundaryCondition("restrain", list(coords), {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model


def _tri_panel() -> FEModel:
    model = FEModel("tri_contact_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 0.0, 1.0, 0.0)
    model.add_element(1, ShellElement(1, [1, 2, 3], "soft", thickness=0.05))
    model.add_boundary_condition(BoundaryCondition("restrain", [1, 2, 3], {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model


def test_sphere_contact_load_resultant_and_nodal_distribution() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("unit", radius=0.2, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    config = SphereContactConfig(penalty_stiffness=1000.0)

    vector, sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        config,
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    resultant = load_vector_resultant(model, vector)

    assert len(records) == 1
    assert records[0].penetration == pytest.approx(0.1)
    assert records[0].normal_force == pytest.approx(100.0)
    np.testing.assert_allclose(sphere_force, [0.0, 0.0, 100.0], atol=1.0e-10)
    np.testing.assert_allclose(resultant.force, [0.0, 0.0, -100.0], atol=1.0e-10)
    assert sum(force[2] for force in records[0].nodal_forces.values()) == pytest.approx(-100.0)
    assert records[0].contact_classification == "face"


def test_contact_patch_load_distribution_limits_single_node_spike() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("unit", radius=0.2, mass=1.0, start_point=(1.0, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    config = SphereContactConfig(penalty_stiffness=1000.0, load_patch_radius_factor=1.25, min_load_patch_nodes=4)

    vector, _sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        config,
        sphere_position=np.array([1.0, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    resultant = load_vector_resultant(model, vector)

    assert len(records) == 1
    assert records[0].contact_classification == "edge"
    assert len(records[0].nodal_forces) >= 4
    np.testing.assert_allclose(resultant.force, [0.0, 0.0, -100.0], atol=1.0e-10)
    largest_share = max(abs(force[2]) for force in records[0].nodal_forces.values()) / 100.0
    assert largest_share < 0.55


def test_deleted_shell_is_not_a_contact_target() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("unit", radius=0.2, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    vector, sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
        deleted_element_ids=(1,),
    )

    assert not records
    assert np.linalg.norm(vector) == pytest.approx(0.0)
    np.testing.assert_allclose(sphere_force, 0.0, atol=1.0e-12)


def test_contact_projection_classifies_center_edge_corner_q8_and_triangle() -> None:
    cases = [
        (_contact_panel(), np.array([0.5, 0.5, 0.08]), "face"),
        (_contact_panel(), np.array([1.0, 0.5, 0.08]), "edge"),
        (_contact_panel(), np.array([1.0, 1.0, 0.08]), "corner"),
        (_q8_panel(), np.array([1.0, 0.5, 0.08]), "edge"),
        (_tri_panel(), np.array([0.0, 0.0, 0.08]), "corner"),
    ]
    for model, position, expected in cases:
        sphere = RigidSphereImpact("classify", radius=0.1, mass=1.0, start_point=position, travel_direction=(0.0, 0.0, -1.0), speed=0.0)
        _vector, _sphere_force, records = assemble_sphere_contact_load_vector(
            model,
            sphere,
            SphereContactConfig(penalty_stiffness=1000.0),
            sphere_position=position,
            sphere_velocity=np.zeros(3),
        )
        assert records
        assert records[0].contact_classification == expected


def test_contact_surface_thickness_offset_changes_penetration() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("surface", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)

    _mid_vector, _mid_force, mid_records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0, contact_surface="midsurface"),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    _top_vector, _top_force, top_records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0, contact_surface="top"),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )

    assert not mid_records
    assert top_records[0].penetration == pytest.approx(0.025)


def test_adjacent_coplanar_contact_is_reduced_to_single_deepest_record() -> None:
    model = FEModel("two_shell_contact")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    for node_id, xyz in {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (2.0, 0.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (1.0, 1.0, 0.0),
        6: (2.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, *xyz)
    model.add_element(
        1, create_shell_element(1, [1, 2, 5, 4], "soft", thickness=0.05)
    )
    model.add_element(
        2, create_shell_element(2, [2, 3, 6, 5], "soft", thickness=0.05)
    )
    sphere = RigidSphereImpact("shared_edge", radius=0.2, mass=1.0, start_point=(1.0, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)

    _vector, sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0, max_active_contacts=1),
        sphere_position=np.array([1.0, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )

    assert len(records) == 1
    np.testing.assert_allclose(sphere_force, [0.0, 0.0, 100.0], atol=1.0e-10)


def test_sphere_no_contact_follows_constant_velocity() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("miss", radius=0.1, mass=2.0, start_point=(0.5, 0.5, 2.0), travel_direction=(1.0, 0.0, 0.0), speed=3.0)
    config = SphereContactConfig(penalty_stiffness=1000.0)

    result = solve_transient_sphere_impact(model, TransientConfig(dt=0.01, t_end=0.03), sphere, config)

    assert result.status == "no_contact"
    np.testing.assert_allclose(result.sphere_positions[-1], [0.59, 0.5, 2.0], atol=1.0e-12)
    np.testing.assert_allclose(result.contact_force_history, 0.0, atol=1.0e-12)
    assert result.peak_contact_force == 0.0


def test_auto_penalty_limits_penetration_ratio_and_is_reported() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("auto", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    recommended = recommend_sphere_contact_penalty(model, sphere, target_penetration_fraction=0.05)

    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.0015, t_end=0.11),
        sphere,
        SphereContactConfig(target_penetration_fraction=0.05, auto_penalty_safety_factor=5.0, max_contact_iterations=40),
    )

    assert recommended > 0.0
    assert result.status == "completed"
    assert result.diagnostics["contact_config"]["penalty_stiffness"] >= recommended
    assert result.max_penetration_ratio < 0.08


def test_sphere_panel_impact_rebounds_and_conserves_impulse_direction() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("hit", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    config = SphereContactConfig(penalty_stiffness=4000.0, contact_damping=0.0, max_contact_iterations=40)

    result = solve_transient_sphere_impact(model, TransientConfig(dt=0.0025, t_end=0.12, output_nodes=[1]), sphere, config)

    assert result.status == "completed"
    assert result.peak_contact_force > 0.0
    assert result.max_penetration > 0.0
    assert result.sphere_velocities[-1, 2] > -2.0
    np.testing.assert_allclose(result.sphere_impulse, result.sphere_velocities[-1] - np.array([0.0, 0.0, -2.0]), rtol=5.0e-2, atol=5.0e-3)
    assert any(step for step in result.active_contact_history)


def test_sphere_impact_stops_cleanly_after_configured_rebound_hold() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("rebound_stop", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    config = SphereContactConfig(
        penalty_stiffness=4000.0,
        contact_damping=0.0,
        max_contact_iterations=40,
        post_separation_time=0.005,
    )

    result = solve_transient_sphere_impact(model, TransientConfig(dt=0.0025, t_end=0.5), sphere, config)

    assert result.status == "completed"
    assert result.diagnostics["stop_reason"] == "completed_after_contact_separation"
    assert result.times[-1] < 0.5
    assert result.diagnostics["post_contact_separation_time"] >= 0.005


def test_impact_fracture_deletes_contacted_shell_and_reports_record() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("fracture_hit", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.0025, t_end=0.12, output_nodes=[1]),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        fracture_config=ImpactFractureConfig(threshold=1.0, trigger="contact_force", max_deleted_fraction=1.0),
    )

    summary = result.diagnostics["impact_fracture_summary"]
    assert result.status in {"completed", "no_contact"}
    assert summary["deleted_count"] == 1
    assert summary["deleted_element_ids"] == [1]
    assert summary["records"][0]["trigger_name"] == "contact_force"
    assert summary["records"][0]["trigger_value"] >= 1.0
    assert result.result_case["analysis_case"]["settings"]["fracture"]["trigger"] == "contact_force"


def test_impact_fracture_max_deleted_fraction_stops_transient() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("fracture_stop", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.0025, t_end=0.12),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        fracture_config=ImpactFractureConfig(threshold=1.0, trigger="contact_force", max_deleted_fraction=0.5),
    )

    assert result.status == "max_deleted_fraction_reached"
    assert result.diagnostics["impact_fracture_summary"]["deleted_count"] == 1


def test_impact_damage_patch_area_is_bounded_and_monotonic() -> None:
    model = _contact_panel()
    element = model.mesh.get_element(1)
    sphere = RigidSphereImpact("damage_patch", radius=0.2, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    config = ImpactDamageConfig(capacity_basis="user", user_capacity=1000.0, min_contact_area=1.0e-4)
    _vector, _sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.array([0.5, 0.5, 0.15]),
        sphere_velocity=np.zeros(3),
    )
    _vector2, _sphere_force2, deeper_records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.array([0.5, 0.5, 0.05]),
        sphere_velocity=np.zeros(3),
    )

    shallow_area = _impact_contact_patch_area(records[0], element, config, sphere)
    deep_area = _impact_contact_patch_area(deeper_records[0], element, config, sphere)

    assert shallow_area >= config.min_contact_area
    assert deep_area > shallow_area


def test_impact_damage_accumulates_under_repeated_subthreshold_contact() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("damage_repeat", radius=0.2, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    _vector, _sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    config = ImpactDamageConfig(
        mode="accumulated_damage",
        capacity_basis="user",
        user_capacity=1000.0,
        damage_threshold=1.0,
        impulse_reference_time=0.01,
        delete_at=1.0,
        max_deleted_fraction=1.0,
    )
    states: dict[int, dict[str, object]] = {}
    deleted: set[int] = set()

    for step in range(4):
        new_deleted, _util, _diags, _changed = _update_impact_damage_states(
            model,
            records,
            config,
            sphere,
            states,
            tuple(deleted),
            step_index=step + 1,
            time_value=0.01 * (step + 1),
            dt=0.01,
        )
        deleted.update(record.element_id for record in new_deleted)
        if deleted:
            break

    assert states[1]["damage"] > 1.0
    assert deleted == {1}


def test_impact_damage_low_energy_or_high_capacity_does_not_delete() -> None:
    sphere = RigidSphereImpact("damage_low", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=0.5)

    result = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(capacity_basis="user", user_capacity=1.0e9, max_deleted_fraction=1.0),
    )

    summary = result.diagnostics["impact_damage_summary"]
    assert result.status in {"completed", "no_contact"}
    assert summary["deleted_count"] == 0
    assert summary["max_damage"] < 1.0
    assert result.result_case["analysis_case"]["settings"]["damage"]["capacity_basis"] == "user"


def test_impact_damage_high_energy_deletes_and_reports_governing_trigger() -> None:
    sphere = RigidSphereImpact("damage_high", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)

    result = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(
            mode="instant_threshold",
            capacity_basis="user",
            user_capacity=10.0,
            delete_at=1.0,
            max_deleted_fraction=1.0,
        ),
    )

    summary = result.diagnostics["impact_damage_summary"]
    assert result.status in {"completed", "no_contact"}
    assert summary["deleted_count"] == 1
    assert summary["deleted_element_ids"] == [1]
    assert summary["records"][0]["governing_component"] in {
        "contact_pressure",
        "impulse_per_area",
        "equivalent_plastic_strain_estimate",
    }
    assert summary["deletion_records"][0]["trigger_name"].startswith("impact_damage:")
    assert result.diagnostics["erosion_summary"]["damage_triggered_element_ids"] == [1]
    assert result.diagnostics["stop_reason"] in {"completed", "no_contact"}


def test_impact_damage_higher_capacity_delays_damage() -> None:
    sphere = RigidSphereImpact("damage_capacity", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    low = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(capacity_basis="user", user_capacity=20.0, max_deleted_fraction=1.0),
    )
    high = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(capacity_basis="user", user_capacity=1.0e6, max_deleted_fraction=1.0),
    )

    assert low.diagnostics["impact_damage_summary"]["max_damage"] > high.diagnostics["impact_damage_summary"]["max_damage"]
    assert low.diagnostics["impact_damage_summary"]["deleted_count"] >= high.diagnostics["impact_damage_summary"]["deleted_count"]


def test_impact_damage_below_softening_does_not_rebuild_matrices() -> None:
    sphere = RigidSphereImpact("damage_no_rebuild", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)

    result = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(
            capacity_basis="user",
            user_capacity=1.0e6,
            softening_start=0.9,
            delete_at=1.0,
            max_deleted_fraction=1.0,
        ),
    )

    assert result.diagnostics["impact_damage_summary"]["max_damage"] < 0.9
    assert result.diagnostics["damage_state_update_count"] > 0
    assert result.diagnostics["eroded_matrix_rebuild_count"] == 0
    assert result.diagnostics["linear_matrix_terms_cached"] is False
    assert result.diagnostics["damage_matrix_plan"] is None


def test_impact_damage_softening_rebuilds_matrices_with_cached_terms() -> None:
    sphere = RigidSphereImpact("damage_rebuild", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)

    result = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        damage_config=ImpactDamageConfig(
            mode="accumulated_damage",
            capacity_basis="user",
            user_capacity=20.0,
            softening_start=0.01,
            delete_at=1.0,
            max_deleted_fraction=1.0,
        ),
    )

    assert result.diagnostics["eroded_matrix_rebuild_count"] > 0
    assert result.diagnostics["linear_matrix_terms_cached"] is True
    selection = result.diagnostics["damage_matrix_plan_selection"]
    assert selection["observed_update_events"] == result.diagnostics["eroded_matrix_rebuild_count"]
    assert (
        selection["legacy_update_count"] + selection["plan_update_count"]
        == result.diagnostics["eroded_matrix_rebuild_count"]
    )
    assert selection["break_even_future_update_events"] == 11
    matrix_plan = result.diagnostics["damage_matrix_plan"]
    if matrix_plan is None:
        assert selection["plan_selected"] is False
        assert selection["legacy_update_count"] == result.diagnostics["eroded_matrix_rebuild_count"]
        assert selection["selection_reason"] in {
            "insufficient_observed_update_events",
            "projected_future_updates_below_break_even",
        }
    else:
        assert selection["plan_selected"] is True
        assert matrix_plan["fast_path_name"] == "incremental_damage_csr_updates"
        assert matrix_plan["update_count"] == selection["plan_update_count"]
        assert matrix_plan["fallback_count"] == 0
    assert result.diagnostics["impact_damage_summary"]["max_damage"] >= 0.01


def test_impact_damage_neighbor_smoothing_holds_one_step_deletion() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("damage_smooth", radius=0.2, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    _vector, _sphere_force, records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    config = ImpactDamageConfig(
        mode="instant_threshold",
        capacity_basis="user",
        user_capacity=1.0,
        neighbor_smoothing=True,
        max_deleted_fraction=1.0,
    )
    states: dict[int, dict[str, object]] = {}
    first_deleted, _util, first_diag, _changed = _update_impact_damage_states(
        model,
        records,
        config,
        sphere,
        states,
        (),
        step_index=1,
        time_value=0.01,
        dt=0.01,
    )
    second_deleted, _util2, _second_diag, _changed2 = _update_impact_damage_states(
        model,
        records,
        config,
        sphere,
        states,
        (),
        step_index=2,
        time_value=0.02,
        dt=0.01,
    )

    assert first_deleted == ()
    assert first_diag[0]["neighbor_smoothing_hold"] is True
    assert len(second_deleted) == 1


def test_impact_damage_and_simple_fracture_summaries_keep_trigger_ownership() -> None:
    sphere = RigidSphereImpact("combined_damage", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)

    result = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        fracture_config=ImpactFractureConfig(threshold=1.0, trigger="contact_force", max_deleted_fraction=1.0),
        damage_config=ImpactDamageConfig(capacity_basis="user", user_capacity=1.0e9, max_deleted_fraction=1.0),
    )

    fracture_summary = result.diagnostics["impact_fracture_summary"]
    damage_summary = result.diagnostics["impact_damage_summary"]
    erosion_summary = result.diagnostics["erosion_summary"]
    assert fracture_summary["deleted_element_ids"] == [1]
    assert damage_summary["deleted_element_ids"] == []
    assert erosion_summary["all_eroded_element_ids"] == [1]
    assert erosion_summary["fracture_triggered_element_ids"] == [1]
    assert erosion_summary["damage_triggered_element_ids"] == []
    assert any("IMPACT_DAMAGE010" in warning for warning in result.diagnostics["warnings"])


def test_impact_damage_reports_capacity_fallback_warning_and_stop_reason() -> None:
    sphere = RigidSphereImpact("fallback_capacity", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 2.0), travel_direction=(1.0, 0.0, 0.0), speed=1.0)

    result = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.01, t_end=0.02),
        sphere,
        SphereContactConfig(penalty_stiffness=1000.0),
        damage_config=ImpactDamageConfig(capacity_basis="yield", max_deleted_fraction=1.0),
    )

    assert result.status == "no_contact"
    assert result.diagnostics["stop_reason"] == "no_contact"
    assert result.result_case["metadata"]["solver_convergence"]["stop_reason"] == "no_contact"
    assert any("IMPACT_DAMAGE011" in warning for warning in result.diagnostics["warnings"])


def test_higher_penalty_reduces_penetration() -> None:
    low = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.11),
        RigidSphereImpact("low", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0),
        SphereContactConfig(penalty_stiffness=1000.0, max_contact_iterations=40),
    )
    high = solve_transient_sphere_impact(
        _contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.11),
        RigidSphereImpact("high", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0),
        SphereContactConfig(penalty_stiffness=5000.0, max_contact_iterations=40),
    )

    assert low.status == "completed"
    assert high.status == "completed"
    assert high.max_penetration < low.max_penetration


def test_event_substepping_catches_contact_that_single_large_step_would_miss() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("fast", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.45), travel_direction=(0.0, 0.0, -1.0), speed=20.0)
    config = SphereContactConfig(penalty_stiffness=4000.0, max_event_substeps=64, max_contact_iterations=40)

    result = solve_transient_sphere_impact(model, TransientConfig(dt=0.05, t_end=0.05), sphere, config)

    assert result.status == "completed"
    assert result.diagnostics["event_substep_count"] > 0
    assert result.peak_contact_force > 0.0


def test_stiffened_panel_impact_moves_coupled_beam_nodes() -> None:
    model = _contact_panel(stiffener=True)
    sphere = RigidSphereImpact("stiffened", radius=0.1, mass=1.0, start_point=(0.25, 0.25, 0.22), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    config = SphereContactConfig(penalty_stiffness=3000.0, max_contact_iterations=40)

    result = solve_transient_sphere_impact(model, TransientConfig(dt=0.0025, t_end=0.11, output_nodes=[10, 11]), sphere, config)

    assert result.status == "completed"
    assert result.peak_contact_force > 0.0
    assert np.max(np.abs(result.node_histories[10][:, 2])) > 0.0
    assert np.max(np.abs(result.node_histories[11][:, 2])) > 0.0


def test_contact_validation_reports_actionable_issue_codes() -> None:
    model = _contact_panel()
    model.materials["soft"].density = 0.0
    sphere = RigidSphereImpact("bad_config", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 1.0), travel_direction=(0.0, 0.0, -1.0), speed=10.0)

    report = validate_contact_configuration(model, sphere, SphereContactConfig(penalty_stiffness=1000.0, max_event_substeps=1), TransientConfig(dt=0.1, t_end=0.1))
    codes = {issue.code for issue in report.issues}

    assert report.status == "invalid"
    assert {"CONTACT002", "CONTACT005"} <= codes
    assert report.mesh_quality["shell_contact_targets"] == 1


def test_selective_recovery_stores_contact_history_without_full_displacement_history() -> None:
    model = _contact_panel()
    sphere = RigidSphereImpact("selected", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    recovery = RecoveryConfig(node_ids=[1], history_mode="selected", store_full_histories=False)

    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.0025, t_end=0.05, output_nodes=[1], recovery=recovery),
        sphere,
        SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
    )

    assert result.status in {"completed", "no_contact"}
    assert result.displacements.shape[1] == 6
    assert result.contact_force_history.shape[1] == 3
    assert result.result_case["recovery"]["history_storage_mode"] == "selected"


def test_invalid_sphere_and_contact_inputs_fail_early() -> None:
    with pytest.raises(ValueError, match="radius"):
        RigidSphereImpact("bad", radius=0.0, mass=1.0, start_point=(0.0, 0.0, 1.0), travel_direction=(0.0, 0.0, -1.0), speed=1.0)
    with pytest.raises(ValueError, match="mass"):
        RigidSphereImpact("bad", radius=0.1, mass=0.0, start_point=(0.0, 0.0, 1.0), travel_direction=(0.0, 0.0, -1.0), speed=1.0)
    with pytest.raises(ValueError, match="travel_direction"):
        RigidSphereImpact("bad", radius=0.1, mass=1.0, start_point=(0.0, 0.0, 1.0), travel_direction=(0.0, 0.0, 0.0), speed=1.0)
    with pytest.raises(ValueError, match="penalty_stiffness"):
        SphereContactConfig(penalty_stiffness=0.0)


def test_aitken_relaxation_converges_high_penalty_corner_contact() -> None:
    """Sticky selection + smooth patch kernel + Aitken rescue a chattering high-penalty hit."""

    def _stiff_panel(n: int = 10) -> FEModel:
        model = FEModel("stiff_panel")
        model.add_material("steel", 2.1e11, 0.3, density=7850.0)
        ids = {}
        node_id = 1
        for j in range(n + 1):
            for i in range(n + 1):
                model.add_node(node_id, i / n, j / n, 0.0)
                ids[(i, j)] = node_id
                node_id += 1
        element_id = 1
        for j in range(n):
            for i in range(n):
                model.add_element(
                    element_id,
                    create_shell_element(
                        element_id,
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
                element_id += 1
        edge = (
            [ids[(i, 0)] for i in range(n + 1)]
            + [ids[(i, n)] for i in range(n + 1)]
            + [ids[(0, j)] for j in range(1, n)]
            + [ids[(n, j)] for j in range(1, n)]
        )
        model.add_boundary_condition(
            BoundaryCondition("edges", edge, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0})
        )
        return model

    sphere = RigidSphereImpact("corner_hit", radius=0.12, mass=200.0, start_point=(0.5, 0.5, 0.15), travel_direction=(0.0, 0.0, -1.0), speed=3.0)
    transient = TransientConfig(dt=2.0e-4, t_end=0.012)

    plain = solve_transient_sphere_impact(
        _stiff_panel(),
        transient,
        sphere,
        SphereContactConfig(penalty_stiffness=2.0e8, max_contact_iterations=25, contact_relaxation="none"),
    )
    relaxed = solve_transient_sphere_impact(
        _stiff_panel(),
        transient,
        sphere,
        SphereContactConfig(penalty_stiffness=2.0e8, max_contact_iterations=25, contact_relaxation="aitken"),
    )

    assert plain.status == "contact_iteration_failed"
    assert relaxed.status == "completed"
    assert relaxed.peak_contact_force > 0.0
    assert max(relaxed.diagnostics["iteration_counts"]) <= 15
    assert relaxed.diagnostics["contact_config"]["contact_relaxation"] == "aitken"


def test_beam_contact_targets_detect_strike_and_balance() -> None:
    """Opt-in sphere-vs-beam-segment contact: exact penalty force, balance, rebound."""

    def _beam_bridge() -> FEModel:
        model = FEModel("beam_bridge")
        model.add_material("steel", 2.1e11, 0.3, density=7850.0)
        n = 6
        for i in range(n + 1):
            model.add_node(i + 1, i / n, 0.0, 0.0)
        for e in range(n):
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", {"area": 1e-2, "Iy": 1e-6, "Iz": 1e-6, "J": 1e-6}))
        model.add_boundary_condition(
            BoundaryCondition("ends", [1, n + 1], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0})
        )
        return model

    beam_radius = np.sqrt(1e-2 / np.pi)
    sphere = RigidSphereImpact("touch", radius=0.1, mass=1.0, start_point=(0.3, 0.0, 0.5), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    position = np.array([0.3, 0.0, 0.1 + beam_radius - 0.02])  # penetration = 0.02

    model = _beam_bridge()
    load, sphere_force, records = assemble_sphere_contact_load_vector(
        model, sphere, SphereContactConfig(penalty_stiffness=1000.0, beam_contact=True), position, np.zeros(3)
    )
    assert len(records) == 1
    assert records[0].contact_classification == "beam"
    assert records[0].normal_force == pytest.approx(1000.0 * 0.02, rel=1.0e-9)
    resultant = load_vector_resultant(model, load)
    assert np.linalg.norm(sphere_force + resultant.force) < 1.0e-9
    # x = 0.3 lies in bridge element 2 (nodes 2-3, length 1/6) at t = 0.8
    assert records[0].local_coordinates[0] == pytest.approx(0.6, abs=1.0e-9)
    assert records[0].nodal_forces[2][2] == pytest.approx(-20.0 * 0.2)
    assert records[0].nodal_forces[3][2] == pytest.approx(-20.0 * 0.8)

    # without the flag a beam-only model has no contact targets at all
    with pytest.raises(ValueError, match="CONTACT001"):
        solve_transient_sphere_impact(
            _beam_bridge(),
            TransientConfig(dt=2.0e-4, t_end=0.01),
            sphere,
            SphereContactConfig(penalty_stiffness=1000.0),
        )

    # transient drop: sphere strikes the girder and rebounds
    drop = RigidSphereImpact("drop", radius=0.05, mass=20.0, start_point=(0.5, 0.0, 0.3), travel_direction=(0.0, 0.0, -1.0), speed=3.0)
    result = solve_transient_sphere_impact(
        _beam_bridge(),
        TransientConfig(dt=2.0e-4, t_end=0.09),
        drop,
        SphereContactConfig(penalty_stiffness=2.0e7, beam_contact=True, max_contact_iterations=30),
    )
    assert result.status == "completed"
    assert result.peak_contact_force > 1.0e4
    assert result.sphere_velocities[-1][2] > 1.0  # rebound (started at -3)
    assert result.sphere_momentum_balance_error < 1.0e-6


def test_impact_fracture_erodes_struck_beam_target() -> None:
    """Impact fracture (contact-force trigger) erodes beam contact targets, not just shells."""
    from anyfesolver.fracture import ImpactFractureConfig

    def _beam_bridge() -> FEModel:
        model = FEModel("frac_bridge")
        model.add_material("steel", 2.1e11, 0.3, density=7850.0)
        n = 6
        for i in range(n + 1):
            model.add_node(i + 1, i / n, 0.0, 0.0)
        for e in range(n):
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", {"area": 1e-3, "Iy": 1e-7, "Iz": 1e-7, "J": 1e-7}))
        model.add_boundary_condition(
            BoundaryCondition("ends", [1, n + 1], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0})
        )
        return model

    sphere = RigidSphereImpact("drop", radius=0.05, mass=50.0, start_point=(0.5, 0.0, 0.2), travel_direction=(0.0, 0.0, -1.0), speed=5.0)
    result = solve_transient_sphere_impact(
        _beam_bridge(),
        TransientConfig(dt=1.0e-4, t_end=0.03),
        sphere,
        SphereContactConfig(penalty_stiffness=1.0e7, beam_contact=True, max_contact_iterations=30),
        fracture_config=ImpactFractureConfig(threshold=1.0e4, trigger="contact_force"),
    )

    summary = result.diagnostics["impact_fracture_summary"]
    assert summary["deleted_count"] >= 1
    assert summary["deleted_beam_length"] > 0.0
    assert summary["deleted_shell_area"] == 0.0
    assert summary["records"][0]["element_type"] == "beam"
    # the max_deleted_fraction denominator counts the 6 beam segments
    assert result.status in {"completed", "max_deleted_fraction_reached"}


def test_capacity_impact_damage_warns_it_skips_beam_contact_targets() -> None:
    from anyfesolver.fracture import ImpactDamageConfig

    model = FEModel("mixed_damage")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    for node_id, xyz in {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0)}.items():
        model.add_node(node_id, *xyz)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "steel", thickness=0.01)
    )
    model.add_node(5, 0.0, 0.5, 0.0)
    model.add_node(6, 1.0, 0.5, 0.0)
    model.add_element(2, BeamElement(2, [5, 6], "steel", {"area": 1e-3, "Iy": 1e-7, "Iz": 1e-7, "J": 1e-7}))
    model.add_boundary_condition(BoundaryCondition("edge", [1, 4], {"ux": 0, "uy": 0, "uz": 0, "rx": 0, "ry": 0, "rz": 0}))

    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.001, t_end=0.004),
        RigidSphereImpact("s", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.3), travel_direction=(0.0, 0.0, -1.0), speed=1.0),
        SphereContactConfig(penalty_stiffness=1000.0, beam_contact=True),
        damage_config=ImpactDamageConfig(),
    )
    assert any("IMPACT_DAMAGE014" in str(w) for w in result.diagnostics["warnings"])


def test_auto_contact_period_substepping_makes_coarse_dt_robust() -> None:
    """A coarse dt with stiff penalty auto-refines during contact to match fine dt."""

    def _panel(n: int = 16, size: float = 3.0) -> FEModel:
        model = FEModel("subpanel")
        model.add_material("steel", 2.1e11, 0.3, density=7850.0)
        ids = {}
        nid = 1
        for j in range(n + 1):
            for i in range(n + 1):
                model.add_node(nid, i * size / n, j * size / n, 0.0)
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
                        thickness=0.012,
                    ),
                )
                eid += 1
        edge = [ids[(i, 0)] for i in range(n + 1)] + [ids[(i, n)] for i in range(n + 1)] + [ids[(0, j)] for j in range(1, n)] + [ids[(n, j)] for j in range(1, n)]
        model.add_boundary_condition(BoundaryCondition("edges", edge, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
        return model

    sphere = RigidSphereImpact("p", radius=0.25, mass=500.0, start_point=(1.5, 1.5, 0.3), travel_direction=(0.0, 0.0, -1.0), speed=5.0)
    cfg = SphereContactConfig(penalty_stiffness=5.0e8, max_contact_iterations=25, max_event_substeps=32)

    fine = solve_transient_sphere_impact(_panel(), TransientConfig(dt=2.5e-4, t_end=0.02), sphere, cfg)
    coarse = solve_transient_sphere_impact(_panel(), TransientConfig(dt=1.0e-3, t_end=0.02), sphere, cfg)

    assert fine.status == "completed"
    assert coarse.status == "completed"
    # the coarse run auto-substepped during contact (event substeps recorded)
    assert coarse.diagnostics["event_substep_count"] > 0
    # and lands within a few percent of the fine-dt peak force despite 4x dt
    assert coarse.peak_contact_force == pytest.approx(fine.peak_contact_force, rel=0.08)

    # free flight (no contact) is not over-substepped by the contact-period rule
    miss = RigidSphereImpact("miss", radius=0.1, mass=2.0, start_point=(0.5, 0.5, 3.0), travel_direction=(1.0, 0.0, 0.0), speed=3.0)
    flyby = solve_transient_sphere_impact(_panel(), TransientConfig(dt=1.0e-3, t_end=0.02), miss, SphereContactConfig(penalty_stiffness=5.0e8, max_event_substeps=32))
    assert flyby.status == "no_contact"
    assert flyby.diagnostics["event_substep_count"] == 0
