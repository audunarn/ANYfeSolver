"""Nearest-cell ownership and symmetric work for two B2 chains."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl,
    LoadCase, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_chain_pair import _BeamChainPairManager


def _chains(*, kink=False):
    model = FEModel("two_round_b2_chains")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    points = [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
              (-1.0, 0.025, 0.0), (0.0, 0.030 if kink else 0.025, 0.0),
              (1.0, 0.025, 0.0)]
    for node, point in enumerate(points, 1):
        model.add_node(node, *point)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4}
    for element_id, node_ids in ((1, (1, 2)), (2, (2, 3)),
                                 (3, (4, 5)), (4, (5, 6))):
        model.add_element(element_id, BeamElement(
            element_id, node_ids, "elastic", section,
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("a", (1, 2), beam_radius=0.01),
                  ContactSurface("b", (3, 4), beam_radius=0.01)),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "a_b", "a", "b", None, 0.004, line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-6,
        maximum_line_subdivisions=12, sliding_difference_step=1.0e-7,
    )
    return model, definition, settings


def _closed(model):
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (4, 5, 6):
        trial[model.mesh.dof_manager.get_node_dofs(node)[1]] = -0.006
    return trial


def _distance_to_segment(point, first, last):
    vector = last - first
    fraction = np.clip(np.dot(point - first, vector) / np.dot(vector, vector), 0.0, 1.0)
    return np.linalg.norm(point - (first + fraction * vector))


def _independent_energy(a, b):
    def side_energy(source, target):
        total = 0.0
        for start, stop in zip(source[:-1], source[1:]):
            reference_length = np.linalg.norm(stop - start)

            def integrand(s):
                point = (1.0 - s) * start + s * stop
                distance = min(_distance_to_segment(point, first, last)
                               for first, last in zip(target[:-1], target[1:]))
                gap = min(0.0, distance - 0.02)
                return 0.5 * 1.0e4 * gap * gap * reference_length

            total += quad(integrand, 0.0, 1.0, epsabs=1.0e-12,
                          points=(0.25, 0.5, 0.75), limit=200)[0]
        return total

    return 0.5 * (side_energy(a, b) + side_energy(b, a))


def test_parallel_chains_have_one_nearest_owner_and_analytic_work():
    model, definition, settings = _chains()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = _BeamChainPairManager(model, definition, settings)
    trial = _closed(model)
    force, tangent, result = manager.evaluate(trial)
    assert result.minimum_gap == pytest.approx(-0.001)
    assert result.contact_energy == pytest.approx(0.01, rel=1.0e-8)
    assert result.beam_a_force[1] == pytest.approx(-20.0, rel=1.0e-8)
    np.testing.assert_allclose(np.asarray(result.beam_a_force) + result.beam_b_force,
                               np.zeros(3), atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(result.beam_a_moment) + result.beam_b_moment,
                               np.zeros(3), atol=1.0e-8)
    direction = np.zeros_like(trial)
    for node in (4, 5, 6):
        direction[model.mesh.dof_manager.get_node_dofs(node)[1]] = 1.0
    step = 3.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2.0 * step), tangent @ direction,
                               rtol=1.0e-5, atol=1.0e-5)
    assert np.any(force)
    exchanged = replace(definition,
                       surfaces=definition.surfaces[::-1],
                       interactions=(replace(definition.interactions[0],
                                             surface="b", target="a"),))
    exchanged_force, _, exchanged_result = _BeamChainPairManager(
        model, exchanged, settings,
    ).evaluate(trial, tangent=False)
    np.testing.assert_allclose(exchanged_force, force, atol=1.0e-8)
    assert exchanged_result.contact_energy == pytest.approx(result.contact_energy)
    through = np.zeros_like(trial)
    for node in (4, 5, 6):
        through[model.mesh.dof_manager.get_node_dofs(node)[1]] = -0.05
    assert manager._minimum_gap(through) == pytest.approx(0.005)
    assert not manager.swept_increment_admissible(
        np.zeros_like(trial), through, 0.0, 1.0,
    )


def test_kinked_chain_matches_independent_closest_polyline_integral():
    model, definition, settings = _chains(kink=True)
    manager = _BeamChainPairManager(model, definition, settings)
    trial = _closed(model)
    force, tangent, result = manager.evaluate(trial)
    a = np.array([model.mesh.get_node(node).coords() for node in (1, 2, 3)])
    b = np.array([model.mesh.get_node(node).coords() for node in (4, 5, 6)])
    b[:, 1] -= 0.006
    expected = _independent_energy(a, b)
    assert result.contact_energy == pytest.approx(expected, rel=1.0e-5)
    np.testing.assert_allclose(np.asarray(result.beam_a_force) + result.beam_b_force,
                               np.zeros(3), atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(result.beam_a_moment) + result.beam_b_moment,
                               np.zeros(3), atol=1.0e-7)
    direction = np.zeros_like(trial)
    direction[model.mesh.dof_manager.get_node_dofs(5)[1]] = 1.0
    step = 3.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2.0 * step), tangent @ direction,
                               rtol=1.0e-5, atol=1.0e-5)
    plus_energy = manager.evaluate(trial + step * direction, tangent=False)[2].contact_energy
    minus_energy = manager.evaluate(trial - step * direction, tangent=False)[2].contact_energy
    assert (plus_energy - minus_energy) / (2.0 * step) == pytest.approx(
        float(force @ direction), rel=1.0e-5, abs=1.0e-6,
    )
    assert np.any(force)


def test_chain_topology_and_resource_limits_fail_explicitly():
    model, definition, settings = _chains()
    disconnected = replace(definition, surfaces=(
        replace(definition.surfaces[0], element_ids=(1, 4)),
        definition.surfaces[1],
    ))
    report = validate_general_contact(model, disconnected, settings)
    assert not report.valid
    assert any("connected" in item or "material nodes" in item for item in report.issues)
    model.add_node(7, 0.0, -1.0, 0.0)
    model.add_element(5, BeamElement(
        5, (2, 7), "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    branched = replace(definition, surfaces=(
        replace(definition.surfaces[0], element_ids=(1, 2, 5)),
        definition.surfaces[1],
    ))
    rejected = validate_general_contact(model, branched, settings)
    assert not rejected.valid
    assert any("branch" in item for item in rejected.issues)
    limited = validate_general_contact(model, definition,
                                       replace(settings, maximum_candidate_pairs=3))
    assert not limited.valid
    assert any("resource limit" in item for item in limited.issues)
    missing_step = validate_general_contact(model, definition,
                                            replace(settings, sliding_difference_step=None))
    assert not missing_step.valid
    assert any("sliding_difference_step" in item for item in missing_step.issues)


def test_chain_coupled_displacement_route_and_restart():
    model, definition, settings = _chains()
    model.add_boundary_condition(BoundaryCondition(
        "a_fixed", (1, 2, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "b_root", (4,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "b_guides", (5, 6),
        {name: 0.0 for name in ("ux", "uz", "rx", "ry", "rz")},
    ))
    load = LoadCase("drive_b_chain")
    load.add_nodal_load(6, forces=(0.0, -1.0, 0.0))
    control = DisplacementControl(node_id=6, dof="uy", target_displacement=-0.008)
    common = dict(contact=definition, contact_settings=settings, control="displacement",
                  displacement_control=control, max_iterations=15, tolerance=1.0e-8)
    full = solve_static_nonlinear(model, load, num_steps=4, **common)
    assert full.status == "completed", (full.failure_reason, full.info.get("general_contact"))
    assert full.info["general_contact"]["gap_tolerance_met"]
    half = solve_static_nonlinear(
        model, load, num_steps=2, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, num_steps=2, restart_checkpoint=half.to_restart_checkpoint(),
        **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1.0e-6, atol=1.0e-8)
