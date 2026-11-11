"""Coupled S3 V2D plane-contact checks for the opt-in penalty profile."""

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    ContactDefinition,
    ContactInteraction,
    ContactSurface,
    LoadCase,
    RigidContactBody,
    create_shell_element,
    solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.core import _PlanarQ4PlaneManager


def _guided_s3_plane(director=(0.0, 0.0, 1.0)):
    director = np.asarray(director, dtype=float)
    director /= np.linalg.norm(director)
    model = FEModel("guided_s3_plane")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, (x, y) in enumerate(((0.0, 0.0), (1.0, 0.0),
                                       (0.0, 1.0)), 1):
        model.add_node(node_id, *(np.asarray((x, y, 0.0)) + 0.035 * director))
    model.add_element(1, create_shell_element(
        1, (1, 2, 3), "elastic", formulation="e4-pl-s3-v2d",
        thickness=0.05, reference_normal=tuple(director),
        reference_surface_offset=0.01,
    ))
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press_s3")
    for node_id in (1, 2, 3):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, -0.1))
    contact = ContactDefinition(
        surfaces=(ContactSurface("s3_bottom", (1,), side="bottom"),),
        rigid_bodies=(RigidContactBody(
            "floor", "plane", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        ),),
        interactions=(ContactInteraction(
            "s3_floor", "s3_bottom", "floor", 1.0e6, 1.0e-6,
        ),),
    )
    return model, load, contact, director


def test_s3_plane_static_reaction_and_gap_use_physical_reference_offset():
    model, load, contact, _director = _guided_s3_plane()
    report = validate_general_contact(model, contact)
    assert report.valid, report.issues
    assert report.profile == "PLANAR_Q4_S3_RIGID_PLANE_PENALTY_MORTAR_V0"
    result = solve_static_nonlinear(
        model, load, contact=contact, kinematics="von_karman",
        num_steps=1, max_iterations=12,
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["minimum_gap"] == pytest.approx(-6.0e-7, abs=1e-9)
    assert summary["shell_force"][2] == pytest.approx(0.3, abs=1e-8)
    assert summary["rigid_force"][2] == pytest.approx(-0.3, abs=1e-8)


def test_s3_plane_offset_couple_and_local_tangent():
    model, _load, contact, director = _guided_s3_plane((0.1, 0.0, 1.0))
    manager = _PlanarQ4PlaneManager(model, contact)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    direction = np.zeros_like(displacement)
    for node_id in (1, 2, 3):
        uz = model.mesh.dof_manager.get_node_dofs(node_id)[2]
        displacement[uz] = -1.0e-6
        direction[uz] = 1.0
    residual, tangent, summary = manager.evaluate(displacement)
    assert summary.minimum_gap == pytest.approx(-1.0e-6)
    assert summary.shell_force[2] == pytest.approx(0.5)
    couple_y = -sum(residual[model.mesh.dof_manager.get_node_dofs(node_id)[4]]
                    for node_id in (1, 2, 3))
    assert couple_y == pytest.approx(0.5 * 0.035 * director[0])
    h = 1.0e-8
    plus = manager.evaluate(displacement + h * direction, tangent=False)[0]
    minus = manager.evaluate(displacement - h * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2.0 * h), tangent @ direction,
                               rtol=1.0e-7, atol=1.0e-8)


def test_s3_plane_restart_matches_uninterrupted_continuation():
    model, load, contact, _director = _guided_s3_plane()
    common = dict(contact=contact, kinematics="von_karman",
                  num_steps=1, max_iterations=12)
    half = solve_static_nonlinear(
        model, load, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    assert half.restart_checkpoint["version"] == 2
    resumed = solve_static_nonlinear(
        model, load, max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    full = solve_static_nonlinear(
        model, load, contact=contact, kinematics="von_karman",
        num_steps=2, max_iterations=12,
    )
    assert resumed.status == full.status == "completed"
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-8, atol=1e-10)
