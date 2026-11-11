"""Loaded ordinary-control trials for S3 pressure against a Q4 target."""

from dataclasses import replace

import numpy as np

from anyfesolver import (
    BoundaryCondition, DisplacementControl, LoadCase,
    solve_static_nonlinear, validate_general_contact,
)
from test_general_contact_s3_q4_patch_augmented import _augmented_case


def _mixed_case(*, force_control):
    model, definition, settings = _augmented_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=5e-3),
    ))
    settings = replace(settings, pressure_tolerance=2e-2,
                       maximum_augmentations=8)
    if force_control:
        for node in (1, 2, 3, 4):
            model.mesh.get_node(node).z -= 0.015
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    if force_control:
        model.add_boundary_condition(BoundaryCondition(
            "source_clamp", (1, 4), {"uz": 0.0},
        ))
    load = LoadCase("drive")
    if force_control:
        for node in (2, 3):
            load.add_nodal_load(node, forces=(0.0, 0.0, -0.01))
    else:
        load.add_nodal_load(1, forces=(0.0, 0.0, -1.0))
    return model, definition, settings, load


def _assert_contact(result):
    assert result.status == "completed", result.failure_reason
    contact = result.info["general_contact"]
    assert contact["normal_law"] == "augmented"
    assert contact["gap_tolerance_met"]
    assert contact["active_pressure_dofs"] > 0
    np.testing.assert_allclose(
        np.asarray(contact["integration_force"]) + contact["target_force"],
        0.0, atol=1e-6,
    )
    assert any(row["augmentation_iterations"] > 1 for row in
               result.info["force_displacement_history"])
    assert len(result.to_restart_checkpoint()["path_state"]["normal_multipliers"]) == 4


def test_mixed_augmented_ordinary_force_control():
    model, definition, settings, load = _mixed_case(force_control=True)
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=2,
        max_iterations=8, min_step_fraction=0.1,
        emit_restart_checkpoint=True,
    )
    _assert_contact(result)


def test_mixed_augmented_ordinary_displacement_control():
    model, definition, settings, load = _mixed_case(force_control=False)
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement", kinematics="von_karman", num_steps=2,
        max_iterations=15, emit_restart_checkpoint=True,
        displacement_control=DisplacementControl(
            node_id=1, dof="uz", target_displacement=-0.0155,
        ),
    )
    _assert_contact(result)
