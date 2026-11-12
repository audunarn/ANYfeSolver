"""Development opt-in through the ordinary S3 static contact manager."""

from dataclasses import replace

import numpy as np

from anyfesolver import (
    ContactPathControl, ContactPathProgram, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_s3_patch_augmented import _augmented_case, _path_case


def test_jet2_s3_development_mode_maps_full_static_residual_and_tangent():
    model, definition, settings = _augmented_case()
    selected = replace(settings, general_s3_differentiation="jet2_s3_dev")
    assert validate_general_contact(model, definition, selected).valid
    reference = GeneralS3PatchManager(model, definition, settings)
    candidate = GeneralS3PatchManager(model, definition, selected)
    assert candidate.law.geometry.differentiation == "jet2_s3"
    reference.set_normal_multipliers(np.full(4, 10.0))
    candidate.set_normal_multipliers(np.full(4, 10.0))
    displacement = np.zeros(candidate.n_dof)
    for node in (1, 2, 3, 4):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.016
        displacement[model.mesh.dof_manager.get_node_dofs(node)[0]] = 0.013
        displacement[model.mesh.dof_manager.get_node_dofs(node)[1]] = 0.007
    old_force, old_tangent, old_contact = reference.evaluate(displacement)
    force, tangent, contact = candidate.evaluate(displacement)
    np.testing.assert_allclose(force, old_force, rtol=2.0e-4, atol=2.0e-7)
    np.testing.assert_allclose(tangent.toarray(), old_tangent.toarray(),
                               rtol=2.0e-3, atol=2.0e-4)
    assert contact.active_pressure_dofs == old_contact.active_pressure_dofs
    assert contact.energy_evaluations < old_contact.energy_evaluations


def test_jet2_s3_mode_rejects_unqualified_patch_geometry():
    model, definition, settings = _augmented_case()
    selected = replace(settings, general_s3_differentiation="jet2_s3_dev")
    definition = replace(definition, surfaces=(
        replace(definition.surfaces[0], element_ids=(1,)),
        replace(definition.surfaces[1], element_ids=(3,)),
    ))
    report = validate_general_contact(model, definition, selected)
    assert not report.valid
    assert "Jet2 S3 development derivative" in " ".join(report.issues)


def test_jet2_s3_mode_runs_static_path_after_smooth_shift():
    model, definition, settings, _ = _path_case()
    settings = replace(settings, general_s3_differentiation="jet2_s3_dev")
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.013, 0.0),
                   (0.013, -0.0155)),
        steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    assert result.info["general_contact"]["active_pressure_dofs"] > 0
    half = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=0.0, atol=1.0e-9)
