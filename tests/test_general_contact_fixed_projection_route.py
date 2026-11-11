"""Opt-in fixed-projection route for the refined Q4/S3 static patch."""

from dataclasses import replace
import os

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_static import (
    GeneralS3PatchManager, REFINED_Q4_S3_GRAPH_AUGMENTED_PROFILE,
)
from anyfesolver.general_contact.static_path import (
    GENERAL_Q4_S3_PATCH_REFINED_GRAPH_AUGMENTED_PROFILE,
)
from test_general_contact_skew_partial_work import _case


def _graph_case():
    model, old, _ = _case(2)
    definition = replace(old.definition, interactions=(replace(
        old.definition.interactions[0],
        normal_stiffness=10000., gap_tolerance=5.e-5,
    ),))
    settings = replace(
        old.settings, q4_source_gap_certificate="fixed_projection_graph",
        maximum_augmentations=16,
    )
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert report.profile == REFINED_Q4_S3_GRAPH_AUGMENTED_PROFILE
    return model, definition, settings, GeneralS3PatchManager(
        model, definition, settings,
    )


def _linear_approach(model, manager, amplitude):
    displacement = np.zeros(manager.n_dof)
    for node_id in range(1, 7):
        node = model.mesh.get_node(node_id)
        dof = model.mesh.dof_manager.get_node_dofs(node_id)[2]
        displacement[dof] = -amplitude*(node.x-.1)
    return displacement


def test_fixed_projection_graph_profile_certifies_skew_approach_and_cutback():
    model, definition, settings, manager = _graph_case()
    zero = np.zeros(manager.n_dof)
    safe = _linear_approach(model, manager, .018453)
    unsafe = _linear_approach(model, manager, .0185)
    assert manager._swept_admissible(zero, safe)
    assert manager._gap_lower(manager._positions(safe)) >= -5.e-5
    assert not manager._swept_admissible(zero, unsafe)
    assert manager.admissible_step_fraction(zero, unsafe) < 1.
    assert manager.restart_contract()["profile"] == (
        REFINED_Q4_S3_GRAPH_AUGMENTED_PROFILE
    )

    previous = GeneralS3PatchManager(model, definition, replace(
        settings, q4_source_gap_certificate="whole_face",
    ))
    assert not previous._swept_admissible(zero, safe)


def test_fixed_projection_graph_profile_rejects_lateral_sliding():
    model, _definition, _settings, manager = _graph_case()
    displacement = _linear_approach(model, manager, .01832)
    for node_id in range(1, 7):
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[0]] = 1.e-4
    with pytest.raises(ValueError, match="in-plane motion"):
        manager._swept_admissible(np.zeros(manager.n_dof), displacement)


def test_fixed_projection_graph_profile_rejects_other_solver_modes():
    model, definition, settings, _manager = _graph_case()
    invalid = replace(settings, pair_motion="restricted_sliding",
                      sliding_difference_step=2.e-5)
    report = validate_general_contact(model, definition, invalid)
    assert not report.valid
    assert "fixed-projection graph" in " ".join(report.issues)


@pytest.mark.skipif(os.environ.get("ANYSOLVER_LONG_CONTACT") != "1",
                    reason="opt-in two-step coupled fixed-projection path")
def test_fixed_projection_graph_runs_ordinary_static_path():
    model, definition, settings, _manager = _graph_case()
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (7, 8, 9, 10),
        {name: 0. for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4, 5, 6),
        {name: 0. for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_left", (1, 4), {"uz": 0.},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(3, "uz"),
                  ContactPathControl.node(6, "uz")),
        waypoints=((0., 0.), (-.018453, -.018453)),
        steps_per_segment=2,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    assert len(result.steps) == 2
    contact = result.info["general_contact"]
    assert contact["profile"] == REFINED_Q4_S3_GRAPH_AUGMENTED_PROFILE
    assert contact["minimum_gap"] >= -definition.interactions[0].gap_tolerance
    assert contact["gap_tolerance_met"]
    assert result.info["contact_path"]["profile"] == (
        GENERAL_Q4_S3_PATCH_REFINED_GRAPH_AUGMENTED_PROFILE
    )
    assert result.restart_checkpoint is not None
