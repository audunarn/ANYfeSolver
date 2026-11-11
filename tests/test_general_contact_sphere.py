"""Analytic-sphere shell work and geometry controls."""

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    RigidContactBody, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.sphere import (
    _GapCertificateLimitError, _SphereShellManager,
)


def _q4_sphere(*, center_height=5.0, warp=0.0):
    model = FEModel("sphere_shell_local")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node, (x, y) in enumerate(((-0.1, -0.1), (0.1, -0.1),
                                    (0.1, 0.1), (-0.1, 0.1)), 1):
        model.add_node(node, x, y, -0.01 + (warp if node == 3 else 0.0))
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.02,
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("skin", (1,), side="top"),),
        rigid_bodies=(RigidContactBody("ball", "sphere", (0.0, 0.0, center_height),
                                       radius=5.0),),
        interactions=(ContactInteraction(
            "sphere_skin", "skin", "ball", 1.0e5, 4.0e-3,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general",
                                     maximum_shell_tilt=0.05)
    return model, definition, settings


def _s3_sphere():
    model = FEModel("s3_sphere_shell_local")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node, (x, y) in enumerate(((-0.1, -0.1), (0.1, -0.1), (0.0, 0.2)), 1):
        model.add_node(node, x, y, -0.005)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3), "elastic", thickness=0.02,
        formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
        reference_surface_offset=0.005,
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("skin", (1,), side="top"),),
        rigid_bodies=(RigidContactBody("ball", "sphere", (0.0, 0.0, 5.0),
                                       radius=5.0),),
        interactions=(ContactInteraction(
            "sphere_skin", "skin", "ball", 1.0e5, 4.0e-3,
        ),),
    )
    return model, definition, ContactSolverSettings(pair_motion="general",
                                                    maximum_shell_tilt=0.05)


def _square_sphere_patch(segments: int):
    model = FEModel(f"sphere_patch_{segments}")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    coordinates = np.linspace(-0.02, 0.02, segments + 1)
    for row, y in enumerate(coordinates):
        for column, x in enumerate(coordinates):
            model.add_node(row * (segments + 1) + column + 1,
                           float(x), float(y), -0.01)
    element_ids = []
    for row in range(segments):
        for column in range(segments):
            lower = row * (segments + 1) + column + 1
            element_id = row * segments + column + 1
            model.add_element(element_id, create_shell_element(
                element_id,
                (lower, lower + 1, lower + segments + 2,
                 lower + segments + 1),
                "elastic", thickness=0.02,
            ))
            element_ids.append(element_id)
    definition = ContactDefinition(
        surfaces=(ContactSurface("skin", tuple(element_ids), side="top"),),
        rigid_bodies=(RigidContactBody(
            "ball", "sphere", (0.0, 0.0, 5.0), radius=5.0,
        ),),
        interactions=(ContactInteraction(
            "sphere_skin", "skin", "ball", 1.0e7, 2.0e-5,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general",
                                     maximum_shell_tilt=0.05)
    return model, definition, settings


def test_sphere_q4_work_tangent_drill_and_wrench_balance():
    model, definition, settings = _q4_sphere()
    assert validate_general_contact(model, definition, settings).valid
    manager = _SphereShellManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in range(1, 5):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        u[dofs[2]] = 0.003
    force, tangent, result = manager.evaluate(u)
    assert result.active_pressure_dofs > 0
    assert result.gap_tolerance_met
    assert result.minimum_gap == pytest.approx(-0.003, abs=1.0e-5)
    assert result.minimum_gap <= result.minimum_sampled_gap
    assert result.minimum_weighted_gap < 0.0
    assert result.pressure_node_count == 4
    assert result.candidate_faces == 1
    assert result.gap_cells_checked > 0
    np.testing.assert_allclose(np.asarray(result.shell_force)
                               + np.asarray(result.rigid_force), 0.0, atol=1.0e-9)
    np.testing.assert_allclose(np.asarray(result.shell_moment)
                               + np.asarray(result.rigid_moment), 0.0, atol=1.0e-9)
    direction = np.linspace(-0.4, 0.6, len(u))
    h = 1.0e-7
    plus_force, _K, plus = manager.evaluate(u + h * direction, tangent=False)
    minus_force, _K, minus = manager.evaluate(u - h * direction, tangent=False)
    assert (plus.penalty_energy - minus.penalty_energy) / (2.0 * h) == pytest.approx(
        float(force @ direction), rel=1.0e-6, abs=1.0e-7,
    )
    np.testing.assert_allclose(
        (plus_force - minus_force) / (2.0 * h),
        tangent @ direction, rtol=1.0e-5, atol=1.0e-6,
    )
    drill = u.copy()
    for node_id in range(1, 5):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        drill[dofs[5]] += 0.25
    drill_force, _K, drill_result = manager.evaluate(drill, tangent=False)
    np.testing.assert_allclose(drill_force, force, atol=1.0e-9)
    assert drill_result.penalty_energy == pytest.approx(result.penalty_energy)
    excessive_tilt = u.copy()
    excessive_tilt[model.mesh.dof_manager.get_node_dofs(1)[3]] = 0.06
    with pytest.raises(ValueError, match="tilt exceeds"):
        manager.evaluate(excessive_tilt, tangent=False)
    inverted = u.copy()
    inverted[model.mesh.dof_manager.get_node_dofs(3)[0]] = -0.4
    with pytest.raises(ValueError, match="inverted or collapsed"):
        manager.evaluate(inverted, tangent=False)


def test_sphere_initial_interference_and_admission():
    model, definition, settings = _q4_sphere(center_height=4.999)
    with pytest.raises(ValueError, match="initial penetration"):
        _SphereShellManager(model, definition, settings)
    assert not validate_general_contact(model, definition,
                                        ContactSolverSettings()).valid
    assert not validate_general_contact(
        model, definition,
        ContactSolverSettings(pair_motion="general", maximum_shell_tilt=0.2),
    ).valid
    warped_model, warped_definition, warped_settings = _q4_sphere(warp=0.01)
    assert not validate_general_contact(
        warped_model, warped_definition, warped_settings,
    ).valid
    with pytest.raises(ValueError, match="von_karman kinematics only"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            kinematics="corotational",
        )


def test_sphere_s3_physical_reference_offset_and_side():
    model, definition, settings = _s3_sphere()
    assert validate_general_contact(model, definition, settings).valid
    manager = _SphereShellManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in range(1, 4):
        u[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 0.003
    _force, _tangent, result = manager.evaluate(u)
    assert result.minimum_gap == pytest.approx(-0.003, abs=1.0e-5)
    assert result.gap_tolerance_met
    assert result.active_pressure_dofs > 0
    assert result.shell_force[2] < 0.0
    from dataclasses import replace

    wrong_side = replace(definition, surfaces=(replace(
        definition.surfaces[0], side="bottom"),))
    assert not validate_general_contact(model, wrong_side, settings).valid


def test_sphere_swept_path_rejects_through_crossing_with_open_endpoints():
    model, definition, settings = _q4_sphere()
    manager = _SphereShellManager(model, definition, settings)
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = start.copy()
    for node_id in range(1, 5):
        end[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 10.0
    assert manager.evaluate(start, tangent=False)[2].minimum_gap >= 0.0
    with pytest.raises(ValueError, match="behind selected"):
        manager.evaluate(end, tangent=False)
    fraction = manager.admissible_step_fraction(start, end)
    assert 0.0 < fraction < 1.0e-3
    assert manager.swept_trial_cutbacks == 1
    assert manager._certify_gap(start, fraction * end,
                                manager.gap_tolerance)[0] >= -manager.gap_tolerance


def test_sphere_swept_path_rejects_interior_shell_collapse():
    model, definition, settings = _q4_sphere()
    manager = _SphereShellManager(model, definition, settings)
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = start.copy()
    for node_id in range(1, 5):
        node = model.mesh.get_node(node_id)
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        end[dofs[0]] = -2.0 * node.x
        end[dofs[1]] = -2.0 * node.y
    assert manager.evaluate(end, tangent=False)[2].gap_tolerance_met
    with pytest.raises(ValueError, match="collapses along the trial path"):
        manager.admissible_step_fraction(start, end)


def test_sphere_uncertified_trial_path_is_shortened(monkeypatch):
    model, definition, settings = _q4_sphere()
    manager = _SphereShellManager(model, definition, settings)
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = start.copy()
    for node_id in range(1, 5):
        end[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 0.001
    original_certificate = manager._certify_gap

    def bounded_certificate(first, candidate, tolerance):
        if np.max(candidate - first) > 0.0005:
            raise _GapCertificateLimitError("trial geometry budget exhausted")
        return original_certificate(first, candidate, tolerance)

    monkeypatch.setattr(manager, "_certify_gap", bounded_certificate)
    fraction = manager.admissible_step_fraction(start, end)
    assert 0.45 < fraction < 0.5
    assert manager.swept_trial_cutbacks == 1
    assert original_certificate(start, fraction * end,
                                manager.gap_tolerance)[0] >= -manager.gap_tolerance


def test_sphere_pressure_resolution_requires_shell_mesh_refinement():
    observed = []
    for segments in (1, 4):
        model, definition, settings = _square_sphere_patch(segments)
        manager = _SphereShellManager(model, definition, settings)
        displacement = np.zeros(model.mesh.dof_manager.total_dofs)
        for node_id in range(1, (segments + 1) ** 2 + 1):
            displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 1.0e-5
        residual, _tangent, result = manager.evaluate(displacement, tangent=False)
        assert result.minimum_gap < 0.0
        assert result.gap_tolerance_met
        observed.append((np.linalg.norm(residual), result.active_pressure_dofs))
    assert observed[0] == (0.0, 0)
    assert observed[1][0] > 0.0
    assert observed[1][1] > 0


def test_sphere_contact_spot_slides_across_two_q4_faces():
    from dataclasses import replace

    model = FEModel("two_face_sphere_slide")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, (x, y) in enumerate((
        (-0.2, -0.1), (0.0, -0.1), (0.2, -0.1),
        (-0.2, 0.1), (0.0, 0.1), (0.2, 0.1),
    ), 1):
        model.add_node(node_id, x, y, -0.01)
    for element_id, nodes in ((1, (1, 2, 5, 4)), (2, (2, 3, 6, 5))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.02,
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("skin", (1, 2), side="top"),),
        rigid_bodies=(RigidContactBody(
            "ball", "sphere", (-0.1, 0.0, 0.5), radius=0.5,
        ),),
        interactions=(ContactInteraction(
            "skin_ball", "skin", "ball", 1.0e5, 1.0e-2,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general",
                                     maximum_shell_tilt=0.05)
    manager = _SphereShellManager(model, definition, settings)
    first = np.zeros(model.mesh.dof_manager.total_dofs)
    last = first.copy()
    for node_id in range(1, 7):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        first[dofs[2]] = 0.008
        last[dofs[2]] = 0.008
        last[dofs[0]] = -0.2
    force_first, _K, result_first = manager.evaluate(first, tangent=False)
    force_last, _K, result_last = manager.evaluate(last, tangent=False)
    assert result_first.active_pressure_dofs > 0
    assert result_last.active_pressure_dofs > 0
    assert result_first.gap_tolerance_met and result_last.gap_tolerance_met

    def exclusive_force(force, nodes):
        return sum(abs(force[model.mesh.dof_manager.get_node_dofs(node)[2]])
                   for node in nodes)

    assert exclusive_force(force_first, (1, 4)) > exclusive_force(force_first, (3, 6))
    assert exclusive_force(force_last, (3, 6)) > exclusive_force(force_last, (1, 4))
    assert manager.admissible_step_fraction(first, last) == 1.0
    under_resolved = first.copy()
    for node_id in range(1, 7):
        under_resolved[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 0.003
    strict = replace(definition, interactions=(replace(
        definition.interactions[0], gap_tolerance=0.001,
    ),))
    _force, _K, strict_result = _SphereShellManager(
        model, strict, settings,
    ).evaluate(under_resolved, tangent=False)
    assert strict_result.active_pressure_dofs == 0
    assert not strict_result.gap_tolerance_met


def test_sphere_q4_static_displacement_control_and_restart():
    model, definition, settings = _q4_sphere()
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    control = DisplacementControl(
        weighted_dofs={(node_id, "uz"): 0.25 for node_id in range(1, 5)},
        target_displacement=0.003,
    )
    load = LoadCase("sphere_approach")
    for node_id in range(1, 5):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, 0.1))

    def solve(*, fraction=1.0, checkpoint=None):
        return solve_static_nonlinear(
            model, load, contact=definition, contact_settings=settings,
            control="displacement", displacement_control=control,
            max_load_factor=fraction, num_steps=2, max_iterations=18,
            restart_checkpoint=checkpoint, emit_restart_checkpoint=True,
        )

    full = solve()
    assert full.status == "completed", (
        full.failure_reason, full.info.get("general_contact"),
    )
    assert full.info["general_contact"]["gap_tolerance_met"]
    assert full.info["general_contact"]["active_pressure_dofs"] > 0
    assert (full.info["general_contact"]["conservative_lower_gap"]
            <= full.info["general_contact"]["minimum_sampled_gap"])
    assert full.info["general_contact"]["shell_force"][2] < 0.0
    assert full.restart_checkpoint["version"] == 2
    half = solve(fraction=0.5)
    assert half.status == "completed", half.failure_reason
    resumed = solve(checkpoint=half.to_restart_checkpoint())
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1.0e-7, atol=1.0e-9)


def test_sphere_q4_static_force_control():
    model, definition, settings = _q4_sphere()
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition("normal_anchor", (1,),
                                                    {"uz": 0.0}))
    load = LoadCase("press_to_sphere")
    for node_id in (2, 3, 4):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, 0.5))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=4, max_iterations=18,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("general_contact"),
    )
    assert result.info["general_contact"]["active_pressure_dofs"] > 0, (
        result.displacements, result.info["general_contact"],
    )
    assert result.info["general_contact"]["gap_tolerance_met"]
