"""Symmetric distributed B2/B2 line contact and crossing controls."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_pair import _BeamPairManager


def _pair(*, crossing=False):
    model = FEModel("b2_round_pair")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    if crossing:
        positions = ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0),
                     (0.0, -0.5, 0.03), (0.0, 0.5, 0.03))
    else:
        positions = ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0),
                     (-0.5, 0.025, 0.0), (0.5, 0.025, 0.0))
    for node_id, position in enumerate(positions, 1):
        model.add_node(node_id, *position)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4,
               "J": 1.0e-4}
    model.add_element(1, BeamElement(1, [1, 2], "elastic", section))
    model.add_element(2, BeamElement(2, [3, 4], "elastic", section))
    definition = ContactDefinition(
        surfaces=(ContactSurface("a", (1,), beam_radius=0.01),
                  ContactSurface("b", (2,), beam_radius=0.01)),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "a_b", "a", "b", normal_stiffness=None,
            gap_tolerance=0.004, line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-7,
    )
    return model, definition, settings


def test_parallel_pair_has_analytic_line_work_balanced_wrench_and_tangent():
    model, definition, settings = _pair()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = _BeamPairManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (3, 4):
        trial[model.mesh.dof_manager.get_node_dofs(node)[1]] = -0.006
    force, tangent, result = manager.evaluate(trial)
    assert result.minimum_gap == pytest.approx(-0.001)
    assert result.contact_energy == pytest.approx(0.005, rel=1.0e-9)
    np.testing.assert_allclose(result.beam_a_force, -np.asarray(result.beam_b_force),
                               atol=1.0e-9)
    np.testing.assert_allclose(result.beam_a_moment,
                               -np.asarray(result.beam_b_moment), atol=1.0e-9)
    assert result.beam_a_force[1] == pytest.approx(-10.0, rel=1.0e-8)
    assert result.maximum_integration_relative_error <= 1.0e-7
    np.testing.assert_allclose(tangent.toarray(), tangent.toarray().T, atol=1.0e-7)
    direction = np.zeros_like(trial)
    for node in (3, 4):
        direction[model.mesh.dof_manager.get_node_dofs(node)[1]] = 1.0
    step = 1.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2.0 * step),
                               tangent @ direction, rtol=1.0e-5, atol=1.0e-5)


def test_crossing_pair_uses_distributed_potential_and_detects_swept_crossing():
    model, definition, settings = _pair(crossing=True)
    manager = _BeamPairManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (3, 4):
        trial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.012
    force, tangent, result = manager.evaluate(trial)
    assert result.minimum_gap == pytest.approx(-0.002)
    half_width = (0.02**2 - 0.018**2)**0.5
    independent_energy = 0.5 * 1.0e4 * quad(
        lambda x: ((x * x + 0.018**2)**0.5 - 0.02)**2,
        -half_width, half_width, epsabs=1.0e-13,
    )[0]
    assert result.contact_energy == pytest.approx(independent_energy, rel=1.0e-6)
    np.testing.assert_allclose(result.beam_a_force, -np.asarray(result.beam_b_force),
                               atol=1.0e-8)
    np.testing.assert_allclose(result.beam_a_moment,
                               -np.asarray(result.beam_b_moment), atol=1.0e-8)
    direction = np.zeros_like(trial)
    direction[model.mesh.dof_manager.get_node_dofs(1)[0]] = 0.3
    direction[model.mesh.dof_manager.get_node_dofs(2)[2]] = -0.2
    direction[model.mesh.dof_manager.get_node_dofs(3)[1]] = 0.5
    direction[model.mesh.dof_manager.get_node_dofs(4)[2]] = 0.7
    step = 1.0e-7
    plus = manager.evaluate(trial + step * direction, tangent=False)[0]
    minus = manager.evaluate(trial - step * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2.0 * step),
                               tangent @ direction, rtol=1.0e-5, atol=1.0e-5)
    exchanged = replace(definition,
        surfaces=definition.surfaces[::-1],
        interactions=(replace(definition.interactions[0],
                              surface="b", target="a"),))
    exchanged_force, _K, exchanged_result = _BeamPairManager(
        model, exchanged, settings,
    ).evaluate(trial)
    np.testing.assert_allclose(exchanged_force, force, atol=1.0e-8)
    assert exchanged_result.contact_energy == pytest.approx(result.contact_energy)
    through = np.zeros_like(trial)
    for node in (3, 4):
        through[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.06
    assert manager._minimum_gap(through) == pytest.approx(0.01)
    assert not manager.swept_increment_admissible(
        np.zeros_like(trial), through, 0.0, 1.0,
    )
    assert manager.admissible_step_fraction(np.zeros_like(trial), through) < 1.0


def test_crossing_pair_runs_through_static_solver_and_checkpoint():
    model, definition, settings = _pair(crossing=True)
    model.add_boundary_condition(BoundaryCondition(
        "a_fixed", (1, 2),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "b_root", (3,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "b_tip_guide", (4,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("tip_drive")
    load.add_nodal_load(4, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=4, dof="uz", target_displacement=-0.025,
        ),
        num_steps=8, max_iterations=20, emit_restart_checkpoint=True,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("general_contact"),
    )
    summary = result.info["general_contact"]
    assert summary["minimum_gap"] < 0.0
    assert summary["gap_tolerance_met"]
    assert summary["active_source_intervals"] >= 2
    assert result.restart_checkpoint["version"] == 2
    changed_radius = replace(definition, surfaces=(
        replace(definition.surfaces[0], beam_radius=0.011),
        definition.surfaces[1],
    ))
    with pytest.raises(ValueError, match="restart|incompatible|contract"):
        solve_static_nonlinear(
            model, load, contact=changed_radius, contact_settings=settings,
            control="displacement",
            displacement_control=DisplacementControl(
                node_id=4, dof="uz", target_displacement=-0.025,
            ),
            num_steps=8, max_iterations=20,
            restart_checkpoint=result.to_restart_checkpoint(),
        )


def test_near_parallel_pair_remains_distributed_and_initial_crossing_is_rejected():
    model, definition, settings = _pair()
    model.mesh.nodes[4].y += 0.001
    manager = _BeamPairManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (3, 4):
        trial[model.mesh.dof_manager.get_node_dofs(node)[1]] = -0.006
    _force, _tangent, result = manager.evaluate(trial)
    assert result.active_source_intervals >= 1
    assert result.minimum_gap == pytest.approx(-0.001)
    assert result.maximum_integration_relative_error <= 1.0e-7

    crossed, crossed_definition, crossed_settings = _pair(crossing=True)
    initial = np.zeros(crossed.mesh.dof_manager.total_dofs)
    for node in (3, 4):
        initial[crossed.mesh.dof_manager.get_node_dofs(node)[2]] = -0.02
    with pytest.raises(ValueError, match="initial penetration"):
        _BeamPairManager(crossed, crossed_definition, crossed_settings, initial)


def test_pair_limits_fail_explicitly_without_dropping_contact():
    model, definition, settings = _pair(crossing=True)
    # Admission requires an explicit integration accuracy for this pair profile.
    missing_accuracy = validate_general_contact(
        model, definition, replace(settings, line_integration_relative_tolerance=None),
    )
    assert not missing_accuracy.valid
    assert "line_integration_relative_tolerance" in str(missing_accuracy.issues)
    manager = _BeamPairManager(model, definition, replace(
        settings, line_integration_relative_tolerance=1.0e-15,
        maximum_line_subdivisions=1,
    ))
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (3, 4):
        trial[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.012
    with pytest.raises(ValueError, match="integration subdivision budget exhausted"):
        manager.evaluate(trial)
    collapsed = np.zeros_like(trial)
    collapsed[model.mesh.dof_manager.get_node_dofs(2)[0]] = -1.0
    with pytest.raises(ValueError, match="zero or invalid length"):
        manager.evaluate(collapsed)


def test_pair_force_control_and_split_restart_match_full_path():
    model, definition, settings = _pair(crossing=True)
    model.add_boundary_condition(BoundaryCondition(
        "a_fixed", (1, 2),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "b_root", (3,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    load = LoadCase("tip_force")
    load.add_nodal_load(4, forces=(0.0, 0.0, -0.01))
    common = dict(contact=definition, contact_settings=settings,
                  max_iterations=20)
    full = solve_static_nonlinear(model, load, num_steps=8, **common)
    assert full.status == "completed", full.failure_reason
    assert full.info["general_contact"]["gap_tolerance_met"]
    assert full.info["general_contact"]["active_source_intervals"] >= 2
    half = solve_static_nonlinear(
        model, load, num_steps=4, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, num_steps=4, max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1.0e-6, atol=1.0e-8)
