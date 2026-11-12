"""Independent radial-line references and coupled static controls."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction, ContactSolverSettings,
    ContactSurface, DisplacementControl, LoadCase, RigidContactBody,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_radial import _BeamRadialManager


def _model(*, kind="sphere", quadratic=False):
    model = FEModel("round_beam_radial")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    nodes = ((-0.5, 0.0, 0.03), (0.0, 0.0, 0.03), (0.5, 0.0, 0.03))
    selected = nodes if quadratic else (nodes[0], nodes[2])
    for node_id, position in enumerate(selected, 1):
        model.add_node(node_id, *position)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4}
    element_type = QuadraticBeamElement if quadratic else BeamElement
    model.add_element(1, element_type(1, list(range(1, len(selected) + 1)),
                                      "elastic", section))
    body = RigidContactBody("obstacle", kind, (0.0, 0.0, 0.0),
                            radius=0.01,
                            normal=(1.0, 0.0, 0.0) if kind == "cylinder" else None)
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=0.01),),
        rigid_bodies=(body,),
        interactions=(ContactInteraction(
            "line_radial", "tube", "obstacle", normal_stiffness=None,
            line_stiffness=1.0e4, gap_tolerance=0.004,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-7,
    )
    return model, definition, settings


def _trial(model, node_ids, distance):
    result = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in node_ids:
        result[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = distance
    return result


def test_cylinder_b2_uniform_line_has_analytic_work_and_tangent():
    model, definition, settings = _model(kind="cylinder")
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = _BeamRadialManager(model, definition, settings)
    trial = _trial(model, (1, 2), -0.012)
    force, tangent, result = manager.evaluate(trial)
    assert result.minimum_gap == pytest.approx(-0.002)
    assert result.line_contact_energy == pytest.approx(0.02, rel=1.0e-10)
    assert result.beam_force[2] == pytest.approx(20.0, rel=1.0e-10)
    np.testing.assert_allclose(result.beam_force, -np.asarray(result.rigid_force))
    np.testing.assert_allclose(result.beam_moment, -np.asarray(result.rigid_moment))
    direction = _trial(model, (1, 2), 1.0)
    step = 1.0e-7
    finite = (manager.evaluate(trial + step * direction, tangent=False)[0]
              - manager.evaluate(trial - step * direction, tangent=False)[0]) / (2.0 * step)
    np.testing.assert_allclose(finite, tangent @ direction, rtol=1.0e-5, atol=1.0e-5)
    shifted_axis = replace(definition, rigid_bodies=(replace(
        definition.rigid_bodies[0], point=(0.2, 0.0, 0.0),
    ),))
    shifted_force, _shifted_tangent, shifted_result = _BeamRadialManager(
        model, shifted_axis, settings,
    ).evaluate(trial)
    np.testing.assert_allclose(shifted_force, force, atol=1.0e-10)
    assert shifted_result.minimum_gap == pytest.approx(result.minimum_gap)


def test_sphere_b2_partial_line_matches_independent_integral_and_sweep():
    model, definition, settings = _model(kind="sphere")
    manager = _BeamRadialManager(model, definition, settings)
    trial = _trial(model, (1, 2), -0.012)
    force, tangent, result = manager.evaluate(trial)
    assert result.minimum_gap == pytest.approx(-0.002)
    width = (0.02**2 - 0.018**2)**0.5
    reference = 0.5 * 1.0e4 * quad(
        lambda x: ((x*x + 0.018**2)**0.5 - 0.02)**2,
        -width, width, epsabs=1.0e-13,
    )[0]
    assert result.line_contact_energy == pytest.approx(reference, rel=1.0e-6)
    assert result.active_line_length == pytest.approx(2.0 * width)
    assert result.maximum_integration_relative_error <= 1.0e-7
    direction = _trial(model, (1,), 0.3) + _trial(model, (2,), 0.7)
    step = 1.0e-7
    finite = (manager.evaluate(trial + step * direction, tangent=False)[0]
              - manager.evaluate(trial - step * direction, tangent=False)[0]) / (2.0 * step)
    np.testing.assert_allclose(finite, tangent @ direction, rtol=1.0e-5, atol=1.0e-5)
    through = _trial(model, (1, 2), -0.06)
    assert manager._minimum_gap(through) == pytest.approx(0.01)
    assert not manager.swept_increment_admissible(np.zeros_like(trial), through, 0.0, 1.0)
    assert manager.admissible_step_fraction(np.zeros_like(trial), through) < 1.0
    assert force.shape == trial.shape


def test_sphere_b3_curved_interior_contact_and_explicit_limits():
    model, definition, settings = _model(kind="sphere", quadratic=True)
    manager = _BeamRadialManager(model, definition, settings)
    trial = _trial(model, (2,), -0.012)
    force, tangent, result = manager.evaluate(trial)
    assert result.minimum_gap == pytest.approx(-0.002)
    assert result.active_line_length > 0.0
    assert result.integration_points >= 24
    reference_energy = 0.5 * 1.0e4 * quad(
        lambda s: min(0.0, ((s - 0.5)**2
                             + (0.03 - 0.048 * s * (1.0 - s))**2)**0.5
                      - 0.02)**2,
        0.0, 1.0, epsabs=1.0e-12,
    )[0]
    assert result.line_contact_energy == pytest.approx(reference_energy, rel=2.0e-5)
    direction = _trial(model, (2,), 1.0)
    step = 1.0e-7
    finite = (manager.evaluate(trial + step * direction, tangent=False)[0]
              - manager.evaluate(trial - step * direction, tangent=False)[0]) / (2.0 * step)
    np.testing.assert_allclose(finite, tangent @ direction, rtol=1.0e-5, atol=1.0e-5)
    assert force.shape == trial.shape
    missing = validate_general_contact(model, definition, replace(
        settings, line_integration_relative_tolerance=None,
    ))
    assert not missing.valid
    with pytest.raises(ValueError, match="initial penetration"):
        _BeamRadialManager(model, definition, settings, trial)
    limited = _BeamRadialManager(model, definition, replace(
        settings, line_integration_relative_tolerance=1.0e-15,
        maximum_line_subdivisions=1,
    ))
    with pytest.raises(ValueError, match="integration subdivision budget exhausted"):
        limited.evaluate(trial)
    collapsed = _trial(model, (2,), 0.0)
    collapsed[model.mesh.dof_manager.get_node_dofs(3)[0]] = -1.0
    with pytest.raises(ValueError, match="zero or invalid centreline speed"):
        manager.evaluate(collapsed)


def test_sphere_two_b2_cells_preserve_integrated_line_work():
    single, one_definition, settings = _model(kind="sphere")
    one_trial = _trial(single, (1, 2), -0.012)
    one_result = _BeamRadialManager(single, one_definition, settings).evaluate(one_trial)[2]
    split = FEModel("split_round_beam")
    split.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node_id, position in enumerate((
        (-0.5, 0.0, 0.03), (0.0, 0.0, 0.03), (0.5, 0.0, 0.03),
    ), 1):
        split.add_node(node_id, *position)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4}
    split.add_element(1, BeamElement(1, [1, 2], "elastic", section))
    split.add_element(2, BeamElement(2, [2, 3], "elastic", section))
    definition = replace(one_definition, surfaces=(replace(
        one_definition.surfaces[0], element_ids=(1, 2),
    ),))
    trial = _trial(split, (1, 2, 3), -0.012)
    result = _BeamRadialManager(split, definition, settings).evaluate(trial)[2]
    assert result.candidate_cells == 2
    assert result.active_line_cells == 2
    assert result.minimum_gap == pytest.approx(one_result.minimum_gap)
    assert result.line_contact_energy == pytest.approx(one_result.line_contact_energy,
                                                       rel=1.0e-8)
    np.testing.assert_allclose(result.beam_force, one_result.beam_force,
                               rtol=1.0e-8, atol=1.0e-8)


def test_cylinder_b2_runs_through_static_solver_and_restart_binding():
    model, definition, settings = _model(kind="cylinder")
    model.add_boundary_condition(BoundaryCondition(
        "root", (1,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "tip_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("tip_drive")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2, dof="uz", target_displacement=-0.012,
        ),
        num_steps=8, max_iterations=20, emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["active_line_length"] > 0.0
    assert summary["gap_tolerance_met"]
    assert summary["beam_force"][2] == pytest.approx(-summary["rigid_force"][2])
    assert result.restart_checkpoint["version"] == 2
    changed = replace(definition, rigid_bodies=(replace(
        definition.rigid_bodies[0], radius=0.011,
    ),))
    with pytest.raises(ValueError, match="restart|incompatible|contract"):
        solve_static_nonlinear(
            model, load, contact=changed, contact_settings=settings,
            control="displacement",
            displacement_control=DisplacementControl(
                node_id=2, dof="uz", target_displacement=-0.012,
            ),
            num_steps=8, max_iterations=20,
            restart_checkpoint=result.to_restart_checkpoint(),
        )


def test_sphere_b3_interior_contact_runs_through_static_solver():
    model, definition, settings = _model(kind="sphere", quadratic=True)
    model.add_boundary_condition(BoundaryCondition(
        "ends_fixed", (1, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "middle_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("middle_drive")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2, dof="uz", target_displacement=-0.012,
        ),
        num_steps=8, max_iterations=20,
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["minimum_gap"] == pytest.approx(-0.002, abs=1.0e-7)
    assert summary["active_line_length"] > 0.0
    assert summary["gap_tolerance_met"]
