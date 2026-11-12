"""Ordinary displacement-control trials for the admitted augmented patch."""

from dataclasses import replace

import numpy as np

from anyfesolver import (
    BoundaryCondition, DisplacementControl, LoadCase, solve_static_nonlinear,
)
from test_general_contact_s3_patch_augmented import _augmented_case


def _controlled_patch():
    model, definition, settings = _augmented_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=5e-3),
    ))
    settings = replace(settings, pressure_tolerance=2e-2,
                       maximum_augmentations=8)
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("drive")
    load.add_nodal_load(1, forces=(0.0, 0.0, -1.0))
    return model, definition, settings, load


def test_displacement_control_augmented_patch_restarts():
    model, definition, settings, load = _controlled_patch()
    common = dict(
        contact=definition, contact_settings=settings, control="displacement",
        kinematics="von_karman", max_iterations=15,
        displacement_control=DisplacementControl(
            node_id=1, dof="uz", target_displacement=-0.0155,
        ), emit_restart_checkpoint=True,
    )
    full = solve_static_nonlinear(model, load, num_steps=2, **common)
    assert full.status == "completed", full.failure_reason
    assert full.info["general_contact"]["normal_law"] == "augmented"
    assert full.info["general_contact"]["gap_tolerance_met"]
    assert any(row["augmentation_iterations"] > 1 for row in
               full.info["force_displacement_history"])
    half = solve_static_nonlinear(
        model, load, num_steps=1, max_load_factor=0.5, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, num_steps=1,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-2, atol=1e-6)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        full.info["general_contact"]["normal_multiplier_estimate_pa"],
        rtol=5e-2, atol=2e-2,
    )


def test_displacement_control_exhausted_augmentation_rolls_back():
    model, definition, settings, load = _controlled_patch()
    settings = replace(settings, pressure_tolerance=1e-5,
                       maximum_augmentations=1)
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement", kinematics="von_karman", num_steps=1,
        max_iterations=15,
        displacement_control=DisplacementControl(
            node_id=1, dof="uz", target_displacement=-0.0155,
        ),
    )
    assert result.status != "completed"
    assert result.failure_reason == "maximum_augmentations_reached"
    np.testing.assert_array_equal(result.displacements, 0.0)
    np.testing.assert_array_equal(
        result.info["general_contact"]["normal_multiplier_estimate_pa"], 0.0,
    )
