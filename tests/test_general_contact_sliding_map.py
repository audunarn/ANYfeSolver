"""Local work-map checks for the restricted moving shell pair owner."""

import numpy as np
import pytest

from anyfesolver import (
    ContactDefinition,
    ContactInteraction,
    ContactSurface,
    create_shell_element,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.sliding import _RestrictedSlidingPairFEManager


def _two_q4_faces(director=None):
    model = FEModel("restricted_sliding_shell_pair")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    direction = (np.asarray(director, dtype=float)
                 if director is not None else np.array((0.0, 0.0, 1.0)))
    square = ((0, 0), (1, 0), (1, 1), (0, 1))
    for node_id, (x, y) in enumerate(square, 1):
        model.add_node(node_id, x, y, 0.0)
    for node_id, (x, y) in enumerate(square, 5):
        model.add_node(node_id, *(np.asarray((x, y, 0.0)) + 0.05 * direction))
    for element_id, nodes in ((1, (1, 2, 3, 4)),
                              (2, (5, 6, 7, 8))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            reference_normal=None if director is None else tuple(direction),
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("upper", (2,), side="bottom"),
                  ContactSurface("lower", (1,), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "upper", "lower", 1.0e6, 1.0e-3,
        ),),
    )
    return model, definition, direction


def test_restricted_sliding_maps_both_body_work_and_tangent():
    model, definition, _direction = _two_q4_faces()
    manager = _RestrictedSlidingPairFEManager(
        model, definition, difference_step=1.0e-7
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id, uz in zip((5, 6, 7, 8),
                           (-1.0e-4, -0.8e-4, -1.2e-4, -1.1e-4)):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        displacement[dofs[0]] = 0.2
        displacement[dofs[1]] = 0.1
        displacement[dofs[2]] = uz
    force, tangent, summary = manager.evaluate(displacement)
    assert summary.current_overlap_area == pytest.approx(0.72)
    assert summary.penalty_energy > 0.0
    assert summary.gap_tolerance_met
    assert summary.conservative_lower_gap <= summary.minimum_sampled_gap
    assert summary.to_dict()["minimum_gap_certified"]
    np.testing.assert_allclose(
        np.asarray(summary.integration_force) + np.asarray(summary.target_force),
        0.0, atol=1.0e-7,
    )
    direction = np.zeros_like(displacement)
    for node_id, uz_step in zip((5, 6, 7, 8), (0.2, -0.1, 0.15, -0.05)):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        direction[dofs[0]] = 0.3
        direction[dofs[1]] = -0.2
        direction[dofs[2]] = uz_step
    for node_id in (1, 2, 3, 4):
        direction[model.mesh.dof_manager.get_node_dofs(node_id)[0]] = 0.1
    h = 2.0e-8
    plus = manager.evaluate(displacement + h * direction, tangent=False)[0]
    minus = manager.evaluate(displacement - h * direction, tangent=False)[0]
    np.testing.assert_allclose(
        (plus - minus) / (2.0 * h), tangent @ direction,
        rtol=1.0e-5, atol=1.0e-4,
    )
    invalid = displacement.copy()
    invalid[model.mesh.dof_manager.get_node_dofs(5)[0]] += 1.0e-5
    with pytest.raises(ValueError, match="uniform face tangential"):
        manager.evaluate(invalid)


def test_restricted_sliding_reports_uncertified_physical_gap():
    model, definition, _direction = _two_q4_faces()
    manager = _RestrictedSlidingPairFEManager(
        model, definition, difference_step=1.0e-7
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (5, 6, 7, 8):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        displacement[dofs[0]] = 0.2
        displacement[dofs[1]] = 0.1
        displacement[dofs[2]] = -2.0e-3
    _force, _tangent, summary = manager.evaluate(displacement, tangent=False)
    assert not summary.gap_tolerance_met
    assert summary.minimum_sampled_gap < -manager.gap_tolerance
    assert not summary.to_dict()["minimum_gap_certified"]


def test_restricted_sliding_swept_bound_finds_hidden_overlap():
    model, definition, _direction = _two_q4_faces()
    manager = _RestrictedSlidingPairFEManager(
        model, definition, difference_step=1.0e-7
    )
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = np.zeros_like(start)
    for node_id in (5, 6, 7, 8):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        start[dofs[0]] = -2.0
        end[dofs[0]] = 2.0
        start[dofs[2]] = -2.0e-3
        end[dofs[2]] = -2.0e-3
    bound = manager.swept_trial_bound(start, end)
    assert bound.overlap_interval == pytest.approx((0.25, 0.75))
    assert not bound.tolerance_certified
    assert bound.conservative_lower_gap == pytest.approx(-2.0e-3)


def test_restricted_sliding_rejects_initial_overclosure():
    model, definition, _direction = _two_q4_faces()
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (5, 6, 7, 8):
        initial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -1.0e-5
    with pytest.raises(ValueError, match="initial penetration"):
        _RestrictedSlidingPairFEManager(
            model, definition, difference_step=1.0e-7,
            initial_displacement=initial,
        )


def test_restricted_sliding_preserves_physical_offset_virtual_couple():
    director = np.asarray((0.1, 0.0, 1.0), dtype=float)
    director /= np.linalg.norm(director)
    model, definition, _direction = _two_q4_faces(director)
    manager = _RestrictedSlidingPairFEManager(
        model, definition, difference_step=1.0e-7
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (5, 6, 7, 8):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        displacement[dofs[0]] = 0.1
        displacement[dofs[1]] = 0.1
        displacement[dofs[2]] = -1.0e-4
    force, _tangent, summary = manager.evaluate(displacement, tangent=False)
    assert summary.current_overlap_area == pytest.approx(0.81)
    normal_force = summary.integration_force[2]
    assert normal_force == pytest.approx(81.0)
    source_couple_y = -sum(
        force[model.mesh.dof_manager.get_node_dofs(node_id)[4]]
        for node_id in (5, 6, 7, 8)
    )
    assert source_couple_y == pytest.approx(normal_force * 0.025 * director[0])


def test_restricted_sliding_reports_shared_edge_derivative_transition():
    model, definition, _direction = _two_q4_faces()
    manager = _RestrictedSlidingPairFEManager(
        model, definition, difference_step=1.0e-7
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (5, 6, 7, 8):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        displacement[dofs[0]] = 0.2
        displacement[dofs[2]] = -1.0e-4
    with pytest.raises(ValueError, match="shared-edge overlap transition"):
        manager.evaluate(displacement, tangent=False)


def test_restricted_s3_q4_map_keeps_s3_reference_surface_offset():
    director = np.asarray((0.1, 0.0, 1.0), dtype=float)
    director /= np.linalg.norm(director)
    model = FEModel("restricted_s3_q4_slide")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, (x, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)), 1):
        model.add_node(node_id, x, y, 0.0)
    for node_id, (x, y) in enumerate(((0, 0), (1, 0), (0, 1)), 5):
        model.add_node(node_id, *(np.asarray((x, y, 0.0)) + 0.06 * director))
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.05,
        reference_normal=tuple(director),
    ))
    model.add_element(2, create_shell_element(
        2, (5, 6, 7), "elastic", formulation="e4-pl-s3-v2d",
        thickness=0.05, reference_normal=tuple(director),
        reference_surface_offset=0.01,
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("s3_bottom", (2,), side="bottom"),
                  ContactSurface("q4_top", (1,), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "s3_bottom", "q4_top", 1.0e6, 1.0e-3,
        ),),
    )
    manager = _RestrictedSlidingPairFEManager(
        model, definition, difference_step=1.0e-7
    )
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (5, 6, 7):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        displacement[dofs[0]] = 0.1
        displacement[dofs[1]] = 0.1
        displacement[dofs[2]] = -1.0e-4
    force, _tangent, summary = manager.evaluate(displacement, tangent=False)
    assert 0.0 < summary.current_overlap_area < 0.5
    normal_force = summary.integration_force[2]
    assert normal_force == pytest.approx(
        1.0e6 * summary.current_overlap_area * 1.0e-4, rel=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(summary.integration_force) + np.asarray(summary.target_force),
        0.0, atol=1e-8,
    )
    source_couple_y = -sum(
        force[model.mesh.dof_manager.get_node_dofs(node_id)[4]]
        for node_id in (5, 6, 7)
    )
    assert source_couple_y == pytest.approx(normal_force * 0.035 * director[0])
