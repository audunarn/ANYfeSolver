"""Coupling checks for the opt-in small-rotation warped Q4 plane profile."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, RigidContactBody,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.warped_q4_static import WarpedQ4PlaneManager


_CORNERS = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.08),
            (1.0, 1.0, -0.04), (0.0, 1.0, 0.03))


def _model(corners=_CORNERS):
    model = FEModel("warped_q4_static_contact")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, point in enumerate(corners, 1):
        model.add_node(node_id, *point)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.08,
        reference_normal=(0.0, 0.0, 1.0),
    ))
    model.add_boundary_condition(BoundaryCondition(
        "root", (1, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    return model


def _definition(*, plane_z=-0.1, side="midsurface", stiffness=1.0e5,
                tolerance=0.02, motion=None):
    return ContactDefinition(
        surfaces=(ContactSurface("skin", (1,), side=side),),
        rigid_bodies=(RigidContactBody(
            "floor", "plane", (0.0, 0.0, plane_z),
            (0.0, 0.0, 1.0), path_translation=motion,
        ),),
        interactions=(ContactInteraction(
            "support", "skin", "floor", stiffness, tolerance,
        ),),
    )


def _settings():
    return ContactSolverSettings(
        warped_q4_plane=True,
        maximum_shell_tilt=0.05,
        warped_q4_quadrature_order=6,
        warped_q4_control_order=12,
        warped_q4_integration_tolerance=1.0e-6,
        warped_q4_gap_accuracy=1.0e-4,
        warped_q4_maximum_gap_cells=10000,
        warped_q4_maximum_gap_depth=16,
    )


def test_warped_static_assembly_maps_local_work_and_rejects_unsupported_profile():
    model = _model()
    definition = _definition()
    settings = _settings()
    assert validate_general_contact(model, definition, settings).valid
    manager = WarpedQ4PlaneManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    u[manager.dofs[:, 2]] = -0.13
    force, stiffness, results = manager.evaluate(u)
    local = manager.law.evaluate(u[manager.dofs])
    np.testing.assert_allclose(force[manager.dofs], local.residual)
    np.testing.assert_allclose(stiffness[np.ix_(manager.dofs.reshape(24),
                                               manager.dofs.reshape(24))].toarray(),
                               local.tangent)
    assert results.active_pressure_dofs > 0
    np.testing.assert_allclose(results.shell_force, -np.sum(local.residual[:, :3], axis=0))
    np.testing.assert_allclose(results.rigid_force, -np.asarray(results.shell_force))
    assert manager.last_gap_certificate.minimum_sampled_gap >= results.minimum_gap
    assert not validate_general_contact(model, definition).valid
    with pytest.raises(ValueError, match="initial warped Q4 plane gap"):
        WarpedQ4PlaneManager(model, _definition(plane_z=-0.01), settings)
    with pytest.raises(MemoryError, match="gap .* budget"):
        WarpedQ4PlaneManager(model, _definition(side="top"), replace(
            settings, warped_q4_maximum_gap_cells=1,
            warped_q4_maximum_gap_depth=0,
            warped_q4_gap_accuracy=1.0e-12,
        ))
    large_rotation = np.zeros_like(u)
    large_rotation[manager.dofs[2, 3]] = 0.051
    with pytest.raises(ValueError, match="small-rotation limit"):
        manager.evaluate(large_rotation)
    assert 0.0 < manager.admissible_step_fraction(
        np.zeros_like(u), large_rotation,
    ) < 1.0
    moving = WarpedQ4PlaneManager(
        model, _definition(motion=(0.0, 0.0, 0.15)), settings,
    )
    assert not moving.swept_increment_admissible(
        np.zeros_like(u), np.zeros_like(u), 0.0, 1.0,
    )
    with pytest.raises(ValueError, match="von_karman"):
        solve_static_nonlinear(model, LoadCase("empty"), contact=definition,
                               contact_settings=settings, kinematics="corotational")
    with pytest.raises(ValueError, match="seeded material"):
        solve_static_nonlinear(model, LoadCase("empty"), contact=definition,
                               contact_settings=settings, initial_element_states={})
    plastic_model = _model()
    plastic_model.add_material("elastic", 1.0e5, 0.3, yield_stress=10.0)
    plastic_report = validate_general_contact(plastic_model, definition, settings)
    assert not plastic_report.valid
    assert any("homogeneous isotropic elastic" in issue
               for issue in plastic_report.issues)


def test_warped_static_reaction_and_restart_follow_coupled_solver():
    # This deliberately mild warpage admits a small physical-gap budget.
    # The strongly warped partial-contact case above remains a separate gate.
    model = _model(tuple((x, y, 0.01 * z) for x, y, z in _CORNERS))
    load = LoadCase("tip_push")
    for node in (2, 3):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.5))
    # The 3 mm development budget is deliberately above the programme's
    # prospective 0.08 mm thickness-scaled qualification budget.
    common = dict(contact=_definition(plane_z=-0.005, tolerance=0.003,
                                      stiffness=1.0e6),
                  contact_settings=_settings(), num_steps=2,
                  max_iterations=25, kinematics="von_karman")
    result = solve_static_nonlinear(model, load, emit_restart_checkpoint=True,
                                    **common)
    assert result.status == "completed", result.failure_reason
    assert result.info["general_contact"]["active_pressure_dofs"] > 0
    assert result.info["general_contact"]["gap_tolerance_met"]
    assert (result.info["general_contact"]["independent_gap_lower_bound"]
            <= result.info["general_contact"]["independent_gap_sample"])
    assert result.info["general_contact"]["contact_integration_relative_error"] < 1.0e-6
    vertical_contact = result.info["general_contact"]["shell_force"][2]
    vertical_root = result.steps[-1].support_reactions["root"][2]
    assert vertical_root + vertical_contact == pytest.approx(1.0, abs=2.0e-6)
    half = solve_static_nonlinear(model, load, max_load_factor=0.5,
                                  emit_restart_checkpoint=True, **common)
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1e-7, atol=1e-8)
    with pytest.raises(ValueError, match="checkpoint analysis contract is incompatible"):
        solve_static_nonlinear(
            model, load, restart_checkpoint=half.to_restart_checkpoint(),
            contact=_definition(plane_z=-0.005, tolerance=0.003,
                                stiffness=1.1e6),
            contact_settings=_settings(), num_steps=2,
            max_iterations=25, kinematics="von_karman",
        )
