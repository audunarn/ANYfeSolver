"""Two-face Q4 pressure continuity and conservative physical-gap checks."""

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


_POINTS = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0004),
           (1.0, 0.0, 0.0008), (0.0, 1.0, 0.0003),
           (0.5, 1.0, -0.0001), (1.0, 1.0, -0.0004))
_FACES = ((1, 2, 5, 4), (2, 3, 6, 5))


def _model(faces=_FACES):
    model = FEModel("two_warped_q4_plane")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, xyz in enumerate(_POINTS, 1):
        model.add_node(node_id, *xyz)
    for element_id, nodes in enumerate(faces, 1):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.08,
            reference_normal=(0.0, 0.0, 1.0),
        ))
    model.add_boundary_condition(BoundaryCondition(
        "root", (1, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    return model


def _definition(tolerance=0.001):
    return ContactDefinition(
        surfaces=(ContactSurface("skin", (1, 2)),),
        rigid_bodies=(RigidContactBody(
            "floor", "plane", (0.0, 0.0, -0.005), (0.0, 0.0, 1.0),
        ),),
        interactions=(ContactInteraction(
            "support", "skin", "floor", 1.0e6, tolerance,
        ),),
    )


def _settings():
    return ContactSolverSettings(
        warped_q4_plane=True, maximum_shell_tilt=0.05,
        warped_q4_quadrature_order=6, warped_q4_control_order=12,
        warped_q4_integration_tolerance=1.0e-6,
        warped_q4_gap_accuracy=1.0e-4,
        warped_q4_maximum_gap_cells=10000,
        warped_q4_maximum_gap_depth=16,
    )


def _independent_patch_integrals():
    abscissae, weights = np.polynomial.legendre.leggauss(20)
    support = np.zeros(6)
    weighted = np.zeros(6)
    gradient = np.zeros((6, 6, 6))
    for face in _FACES:
        corners = np.asarray([_POINTS[node - 1] for node in face])
        rows = np.asarray(face) - 1
        for xi, wx in zip(abscissae, weights):
            for eta, wy in zip(abscissae, weights):
                shape = 0.25 * np.array(((1 - xi) * (1 - eta),
                                         (1 + xi) * (1 - eta),
                                         (1 + xi) * (1 + eta),
                                         (1 - xi) * (1 + eta)))
                dxi = 0.25 * np.array((-(1 - eta), 1 - eta,
                                       1 + eta, -(1 + eta)))
                deta = 0.25 * np.array((-(1 - xi), -(1 + xi),
                                        1 + xi, 1 - xi))
                area = np.linalg.norm(np.cross(dxi @ corners, deta @ corners))
                measure = wx * wy * area
                gap = float(shape @ corners[:, 2] + 0.005)
                for a, row in enumerate(rows):
                    support[row] += measure * shape[a]
                    weighted[row] += measure * shape[a] * gap
                    for b, column in enumerate(rows):
                        gradient[row, column, 2] += measure * shape[a] * shape[b]
    return support, weighted, gradient.reshape(6, 36)


def test_two_face_warped_q4_pressure_space_and_global_work():
    model = _model()
    definition = _definition()
    settings = _settings()
    assert validate_general_contact(model, definition, settings).valid
    manager = WarpedQ4PlaneManager(model, definition, settings)
    assert len(manager.node_ids) == 6
    with pytest.raises(MemoryError, match="patch gap cell budget"):
        WarpedQ4PlaneManager(
            model, definition, replace(settings, warped_q4_maximum_gap_cells=1),
        )
    support, weighted, gradient = _independent_patch_integrals()
    np.testing.assert_allclose(manager.law.support, support, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(manager.law.reference_weighted_gap,
                               weighted, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(manager.law.gap_gradient, gradient,
                               rtol=1e-8, atol=1e-11)
    assert np.all(manager.law.support > 0.0)
    assert np.linalg.matrix_rank(manager.law.gap_gradient[:, 2::6]) == 6
    reversed_definition = ContactDefinition(
        surfaces=(ContactSurface("skin", (2, 1)),),
        rigid_bodies=definition.rigid_bodies,
        interactions=definition.interactions,
    )
    reversed_manager = WarpedQ4PlaneManager(model, reversed_definition, settings)
    np.testing.assert_allclose(reversed_manager.law.support, manager.law.support,
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(reversed_manager.law.gap_gradient,
                               manager.law.gap_gradient, rtol=0, atol=1e-14)
    u = np.zeros(manager.n_dof)
    u[manager.dofs[:, 2]] = -0.006
    force, tangent, result = manager.evaluate(u)
    assert result.active_pressure_dofs == 6
    assert result.shell_force[2] > 0.0
    np.testing.assert_allclose(result.rigid_force, -np.asarray(result.shell_force))
    assert result.minimum_gap <= manager.last_gap_certificate.minimum_sampled_gap
    direction = np.zeros_like(u)
    direction[manager.dofs[2, 2]] = 1.0
    step = 1.0e-7
    numerical = (manager.evaluate(u + step * direction, tangent=False)[0]
                 - manager.evaluate(u - step * direction, tangent=False)[0]) / (2 * step)
    np.testing.assert_allclose(numerical, tangent @ direction,
                               rtol=1e-5, atol=1e-5)


def test_two_face_warped_q4_rejects_nonconforming_edge():
    model = _model(((1, 2, 5, 4), (1, 2, 5, 4)))
    report = validate_general_contact(model, _definition(), _settings())
    assert not report.valid
    assert any("share exactly one material edge" in issue for issue in report.issues)
    diagonal_model = _model(((1, 2, 5, 4), (1, 3, 6, 5)))
    diagonal_report = validate_general_contact(
        diagonal_model, _definition(), _settings(),
    )
    assert not diagonal_report.valid
    assert any("material edge" in issue for issue in diagonal_report.issues)
    regular_model = _model()
    offset_definition = ContactDefinition(
        surfaces=(ContactSurface("skin", (1, 2), side="top"),),
        rigid_bodies=_definition().rigid_bodies,
        interactions=_definition().interactions,
    )
    offset_report = validate_general_contact(
        regular_model, offset_definition, _settings(),
    )
    assert not offset_report.valid
    assert any("midsurface" in issue for issue in offset_report.issues)


def test_two_face_warped_q4_coupled_gap_budget_and_restart():
    model = _model()
    load = LoadCase("tip_push")
    for node in (3, 6):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.5))
    common = dict(contact_settings=_settings(), num_steps=2,
                  max_iterations=25, kinematics="von_karman")
    tight = solve_static_nonlinear(
        model, load, contact=_definition(tolerance=0.001), **common,
    )
    assert tight.status == "contact_gap_exceeded"
    assert tight.load_factor == pytest.approx(0.07958984375)
    assert tight.info["general_contact"]["active_pressure_dofs"] == 0
    loose_contact = _definition(tolerance=0.003)
    full = solve_static_nonlinear(
        model, load, contact=loose_contact,
        emit_restart_checkpoint=True, **common,
    )
    assert full.status == "completed", full.failure_reason
    report = full.info["general_contact"]
    assert report["active_pressure_dofs"] == 2
    assert report["independent_gap_lower_bound"] >= -0.003
    assert report["independent_gap_lower_bound"] <= report["independent_gap_sample"]
    assert report["shell_force"][2] > 0.0
    assert (full.steps[-1].support_reactions["root"][2]
            + report["shell_force"][2]) == pytest.approx(1.0, abs=2.0e-6)
    half = solve_static_nonlinear(
        model, load, contact=loose_contact, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, contact=loose_contact,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-8)
