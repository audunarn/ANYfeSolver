"""Ordinary force-control trials for the admitted augmented shell patch."""

from dataclasses import replace

import numpy as np

from anyfesolver import BoundaryCondition, LoadCase, solve_static_nonlinear
from test_general_contact_s3_patch_augmented import _augmented_case


def _loaded_patch():
    model, definition, settings = _augmented_case()
    for node in (1, 2, 3, 4):
        model.mesh.get_node(node).z -= 0.015
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=5e-3),
    ))
    settings = replace(settings, pressure_tolerance=2e-2,
                       maximum_augmentations=6)
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guided", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_clamp", (1, 4), {"uz": 0.0},
    ))
    load = LoadCase("press")
    for node in (2, 3):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.01))
    return model, definition, settings, load


def test_force_control_augmented_patch_restarts_after_contact():
    model, definition, settings, load = _loaded_patch()
    common = dict(contact=definition, contact_settings=settings,
                  load_case=load, control="force", kinematics="von_karman",
                  max_iterations=8, num_steps=2, min_step_fraction=0.1,
                  emit_restart_checkpoint=True)
    full = solve_static_nonlinear(model, **common)
    assert full.status == "completed", (full.failure_reason, full.info)
    assert full.info["general_contact"]["normal_law"] == "augmented"
    assert full.info["general_contact"]["gap_tolerance_met"]
    assert any(row["augmentation_iterations"] > 1 for row in
               full.info["force_displacement_history"])
    assert full.info["last_contact_augmentation_trace"]
    assert all("normal_multiplier_change_pa" in row for row in
               full.info["last_contact_augmentation_trace"])
    # Match the full run's first accepted load increment at 0.5. Two steps
    # to 0.5 would instead create a distinct 0.25, 0.5 pressure history.
    half = solve_static_nonlinear(
        model, max_load_factor=0.5, **{**common, "num_steps": 1},
    )
    assert half.status == "completed", half.failure_reason
    assert [row["load_factor"] for row in half.info["force_displacement_history"]] == [0.5]
    assert max(half.to_restart_checkpoint()["path_state"]["normal_multipliers"]) > 0
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    assert [row["load_factor"] for row in resumed.info["force_displacement_history"]] == [0.5, 1.0]
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        full.info["general_contact"]["normal_multiplier_estimate_pa"],
        rtol=1e-10, atol=1e-12,
    )


def test_force_control_exhausted_augmentation_restores_committed_state():
    model, definition, settings, load = _loaded_patch()
    settings = replace(settings, maximum_augmentations=1,
                       pressure_tolerance=1e-5)
    result = solve_static_nonlinear(
        model, load_case=load, contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=1,
        max_iterations=8, min_step_fraction=0.5,
    )
    assert result.status != "completed"
    assert result.failure_reason == "maximum_augmentations_reached"
    trace = result.info["last_contact_augmentation_trace"]
    assert len(trace) == 1
    assert "normal_multiplier_change_pa" in trace[0]
    np.testing.assert_array_equal(result.displacements, 0.0)
    np.testing.assert_array_equal(
        result.info["general_contact"]["normal_multiplier_estimate_pa"], 0.0,
    )
