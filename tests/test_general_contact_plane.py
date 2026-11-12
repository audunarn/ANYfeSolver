"""Physical checks for the first coupled general-contact profile."""

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    ContactDefinition,
    ContactInteraction,
    ContactSolverSettings,
    ContactSurface,
    DisplacementControl,
    LoadCase,
    RigidContactBody,
    create_shell_element,
    solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.core import (
    _PlanarQ4PlaneManager,
    active_contact_manager,
    contact_scope,
    suspend_contact,
)


def _panel():
    model = FEModel("contacted_cantilever")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, xyz in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
         (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)), 1
    ):
        model.add_node(node_id, *xyz)
    model.add_element(1, create_shell_element(1, [1, 2, 3, 4], "elastic", thickness=0.05))
    model.add_boundary_condition(BoundaryCondition("root", [1, 4],
                                                    {name: 0.0 for name in
                                                     ("ux", "uy", "uz", "rx", "ry", "rz")}))
    return model


def _guided_plate():
    model = FEModel("uniform_plane_reaction")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, xyz in enumerate(((0, 0, 0), (1, 0, 0),
                                   (1, 1, 0), (0, 1, 0)), 1):
        model.add_node(node_id, *xyz)
    model.add_element(1, create_shell_element(1, [1, 2, 3, 4],
                                               "elastic", thickness=0.05))
    model.add_boundary_condition(BoundaryCondition(
        "guide", [1, 2, 3, 4],
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")}
    ))
    load = LoadCase("uniform_pressure_equivalent")
    for node_id in (1, 2, 3, 4):
        load.add_nodal_load(node_id, forces=(0, 0, -0.1))
    return model, load


def _contact(plane_z=-0.01, tolerance=0.001, stiffness=1.0e5):
    return ContactDefinition(
        surfaces=(ContactSurface("skin", (1,)),),
        rigid_bodies=(RigidContactBody("floor", "plane", (0, 0, plane_z), (0, 0, 1)),),
        interactions=(ContactInteraction("support", "skin", "floor", stiffness,
                                         tolerance),),
    )


def test_plane_contact_uniform_pressure_work_and_tangent():
    model = _panel()
    manager = _PlanarQ4PlaneManager(model, _contact(plane_z=-0.02))
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -0.03
    force, tangent, result = manager.evaluate(u)
    assert result.minimum_gap == pytest.approx(-0.01, abs=1.0e-12)
    assert result.penalty_energy == pytest.approx(5.0)
    assert result.shell_force == pytest.approx((0.0, 0.0, 1000.0))
    assert result.rigid_force == pytest.approx((0.0, 0.0, -1000.0))
    direction = np.zeros_like(u)
    direction[model.mesh.dof_manager.get_node_dofs(2)[2]] = 1.0
    step = 1.0e-6
    force_plus = manager.evaluate(u + step * direction, tangent=False)[0]
    force_minus = manager.evaluate(u - step * direction, tangent=False)[0]
    np.testing.assert_allclose((force_plus - force_minus) / (2 * step),
                               tangent @ direction, rtol=1.0e-8, atol=1.0e-7)


def test_opening_and_recontact_do_not_retain_penalty_traction():
    model = _panel()
    manager = _PlanarQ4PlaneManager(model, _contact(plane_z=-0.02))
    closed = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (1, 2, 3, 4):
        closed[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -0.03
    first_force, _first_tangent, first = manager.evaluate(closed)
    opening = manager.evaluate(np.zeros_like(closed))[2]
    second_force, _second_tangent, second = manager.evaluate(closed)
    assert opening.active_pressure_dofs == 0
    assert opening.shell_force == pytest.approx((0.0, 0.0, 0.0))
    assert second.penalty_energy == pytest.approx(first.penalty_energy)
    np.testing.assert_allclose(second_force, first_force)


def test_plane_contact_changes_coupled_static_response():
    model = _panel()
    load = LoadCase("push")
    load.add_nodal_load(2, forces=(0.0, 0.0, -0.1))
    load.add_nodal_load(3, forces=(0.0, 0.0, -0.1))
    options = dict(num_steps=3, max_iterations=20, kinematics="corotational")
    free = solve_static_nonlinear(model, load, **options)
    contacted = solve_static_nonlinear(
        model, load, contact=_contact(tolerance=0.006, stiffness=1.0e7), **options
    )
    assert free.converged
    assert contacted.converged
    assert contacted.info["general_contact"]["active_pressure_dofs"] > 0
    assert contacted.info["general_contact"]["normal_stiffness"] == 1.0e7
    assert contacted.info["general_contact"]["gap_tolerance"] == 0.006
    free_tip = np.mean([free.displacements[model.mesh.dof_manager.get_node_dofs(n)[2]]
                        for n in (2, 3)])
    contact_tip = np.mean([contacted.displacements[model.mesh.dof_manager.get_node_dofs(n)[2]]
                           for n in (2, 3)])
    assert contact_tip > free_tip
    assert contacted.info["general_contact"]["gap_tolerance_met"]
    vertical_support = contacted.steps[-1].support_reactions["root"][2]
    vertical_contact = contacted.info["general_contact"]["shell_force"][2]
    assert vertical_support + vertical_contact - 0.2 == pytest.approx(
        0.0, abs=1e-8
    )


@pytest.mark.parametrize(
    ("side", "plane_z", "kinematics"),
    (("midsurface", 0.0, "von_karman"),
     ("midsurface", 0.0, "corotational"),
     ("top", 0.025, "von_karman"),
     ("bottom", -0.025, "von_karman")),
)
def test_uniform_plate_support_matches_an_independent_rigid_mode_solution(
    side, plane_z, kinematics
):
    model, load = _guided_plate()
    contact = ContactDefinition(
        surfaces=(ContactSurface("skin", (1,), side=side),),
        rigid_bodies=(RigidContactBody(
            "floor", "plane", (0, 0, plane_z), (0, 0, 1)
        ),),
        interactions=(ContactInteraction("support", "skin", "floor", 1e6, 1e-6),),
    )
    result = solve_static_nonlinear(model, load, contact=contact,
                                    num_steps=1, max_iterations=12,
                                    kinematics=kinematics)
    assert result.status == "completed", (
        result.failure_reason, result.info.get("first_failed_iteration_reason"),
        result.info.get("convergence_adaptation", [])[:3],
    )
    expected_gap = -0.4 / 1e6  # force / (pressure stiffness * 1 m²)
    assert result.info["general_contact"]["minimum_gap"] == pytest.approx(
        expected_gap, abs=1e-9
    )
    assert result.info["general_contact"]["shell_force"][2] == pytest.approx(
        0.4, rel=1e-5
    )
    for node_id in (1, 2, 3, 4):
        uz = result.displacements[model.mesh.dof_manager.get_node_dofs(node_id)[2]]
        assert uz == pytest.approx(expected_gap, abs=1e-9)
    if side == "midsurface" and kinematics == "von_karman":
        prescribed = solve_static_nonlinear(
            model, load, contact=contact, control="displacement",
            displacement_control=DisplacementControl(
                node_id=1, dof="uz", target_displacement=expected_gap
            ),
            num_steps=1, max_iterations=12, kinematics=kinematics,
        )
        assert prescribed.status == "completed"
        assert prescribed.load_factor == pytest.approx(1.0, rel=1e-6)
        assert prescribed.info["general_contact"]["shell_force"][2] == pytest.approx(
            0.4, rel=1e-5
        )


def test_gap_failure_keeps_last_accepted_static_state():
    model = _panel()
    load = LoadCase("push")
    for node_id in (2, 3):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, -0.1))
    result = solve_static_nonlinear(
        model, load, contact=_contact(stiffness=1.0e7, tolerance=0.001),
        num_steps=3, max_iterations=20, kinematics="corotational",
    )
    assert result.status == "contact_gap_exceeded"
    assert not result.converged
    assert result.failure_reason == "contact_gap_exceeded"
    assert 0.0 < result.load_factor < 1.0
    assert result.info["general_contact"]["gap_tolerance_met"]


def test_contact_restart_continues_and_rejects_a_changed_contact_law():
    model, load = _guided_plate()
    contact = _contact(plane_z=0.0, tolerance=1e-6, stiffness=1e6)
    common = dict(num_steps=1, max_iterations=12, kinematics="von_karman")
    half = solve_static_nonlinear(
        model, load, contact=contact, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed"
    assert half.restart_checkpoint["version"] == 2
    assert half.restart_checkpoint["contact_state"] == {}
    resumed = solve_static_nonlinear(
        model, load, contact=contact, max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    uninterrupted = solve_static_nonlinear(
        model, load, contact=contact, num_steps=2,
        max_iterations=12, kinematics="von_karman",
    )
    assert resumed.status == uninterrupted.status == "completed"
    np.testing.assert_allclose(resumed.displacements, uninterrupted.displacements,
                               rtol=1e-8, atol=1e-10)
    assert resumed.info["general_contact"]["shell_force"] == pytest.approx(
        uninterrupted.info["general_contact"]["shell_force"], abs=1e-8
    )
    with pytest.raises(ValueError, match="analysis contract is incompatible"):
        solve_static_nonlinear(
            model, load, contact=_contact(plane_z=0.0, tolerance=1e-6,
                                          stiffness=2e6),
            max_load_factor=1.0, restart_checkpoint=half.to_restart_checkpoint(),
            **common,
        )
    with pytest.raises(ValueError, match="corotational contact restart"):
        solve_static_nonlinear(
            model, load, contact=contact, kinematics="corotational",
            emit_restart_checkpoint=True,
        )


def test_nonqualified_shell_and_initial_overclosure_are_rejected():
    model = _panel()
    report = validate_general_contact(model, _contact())
    assert report.valid
    with pytest.raises(ValueError, match="initial contact overclosure"):
        _PlanarQ4PlaneManager(model, _contact(plane_z=0.02))
    with pytest.raises(ValueError, match="initial contact overclosure"):
        _PlanarQ4PlaneManager(model, _contact(plane_z=1e-7, tolerance=1e-6))
    with pytest.raises(ValueError, match="positive integer"):
        ContactSolverSettings(maximum_candidate_pairs=0)


def test_callback_failure_discards_contact_scope_and_allows_a_fresh_solve():
    model = _panel()
    load = LoadCase("push")
    load.add_nodal_load(2, forces=(0.0, 0.0, -0.1))
    load.add_nodal_load(3, forces=(0.0, 0.0, -0.1))

    def fail_callback(_message):
        raise RuntimeError("deliberate callback failure")

    with pytest.raises(RuntimeError, match="deliberate callback failure"):
        solve_static_nonlinear(
            model, load, contact=_contact(tolerance=0.006, stiffness=1.0e7),
            num_steps=3, max_iterations=20, kinematics="corotational",
            status_callback=fail_callback,
        )
    assert active_contact_manager() is None
    fresh = solve_static_nonlinear(
        model, load, contact=_contact(tolerance=0.006, stiffness=1.0e7),
        num_steps=3, max_iterations=20, kinematics="corotational",
    )
    assert fresh.status == "completed"


def test_two_patches_count_shared_nodes_once_and_respect_candidate_limit():
    model = FEModel("two_patches")
    model.add_material("elastic", 1.0e5, 0.3)
    for node_id, xyz in enumerate(((0, 0, 0), (1, 0, 0), (2, 0, 0),
                                   (0, 1, 0), (1, 1, 0), (2, 1, 0)), 1):
        model.add_node(node_id, *xyz)
    for element_id, nodes in ((1, [1, 2, 5, 4]), (2, [2, 3, 6, 5])):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("skin", (1, 2)),),
        rigid_bodies=(RigidContactBody("floor", "plane", (0, 0, -0.01),
                                        (0, 0, 1)),),
        interactions=(ContactInteraction("contact", "skin", "floor", 1000.0,
                                         0.02),),
    )
    with pytest.raises(MemoryError, match="candidate limit"):
        _PlanarQ4PlaneManager(model, definition,
                              ContactSolverSettings(maximum_candidate_pairs=1))
    manager = _PlanarQ4PlaneManager(model, definition)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in range(1, 7):
        u[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -0.02
    force, _tangent, result = manager.evaluate(u)
    assert result.shell_force[2] == pytest.approx(20.0)
    assert result.active_pressure_dofs == 6
    assert sum(force[model.mesh.dof_manager.get_node_dofs(node_id)[2]]
               for node_id in range(1, 7)) == pytest.approx(-20.0)


def test_nested_contact_scope_restores_outer_analysis():
    model = _panel()
    assert active_contact_manager() is None
    with contact_scope(model, _contact()) as manager:
        assert active_contact_manager() is manager
        with suspend_contact():
            assert active_contact_manager() is None
        assert active_contact_manager() is manager
    assert active_contact_manager() is None


def test_plane_swept_limit_cuts_a_large_trial_before_overclosure():
    model = _panel()
    manager = _PlanarQ4PlaneManager(model, _contact())
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = start.copy()
    end[model.mesh.dof_manager.get_node_dofs(2)[2]] = -0.1
    fraction = manager.admissible_step_fraction(start, end)
    assert fraction == pytest.approx(0.98 * 0.11)
    accepted = manager.evaluate(start + fraction * (end - start), tangent=False)[2]
    assert accepted.gap_tolerance_met
    assert manager.admissible_step_fraction(start, start) == 1.0


def test_contact_work_and_force_rotate_with_the_physical_model():
    base = _panel()
    rotation = np.array(((1.0, 0.0, 0.0),
                         (0.0, 0.0, -1.0),
                         (0.0, 1.0, 0.0)))
    rotated = FEModel("rotated_contact_panel")
    rotated.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id in (1, 2, 3, 4):
        position = rotation @ base.mesh.get_node(node_id).coords()
        rotated.add_node(node_id, *position)
    rotated.add_element(1, create_shell_element(
        1, [1, 2, 3, 4], "elastic", thickness=0.05
    ))
    plane = _contact(plane_z=-0.02)
    rotated_plane = ContactDefinition(
        surfaces=plane.surfaces,
        rigid_bodies=(RigidContactBody("floor", "plane",
                                        tuple(rotation @ np.array((0.0, 0.0, -0.02))),
                                        tuple(rotation @ np.array((0.0, 0.0, 1.0)))),),
        interactions=plane.interactions,
    )
    u = np.zeros(base.mesh.dof_manager.total_dofs)
    u_rotated = np.zeros(rotated.mesh.dof_manager.total_dofs)
    for node_id in (1, 2, 3, 4):
        u[base.mesh.dof_manager.get_node_dofs(node_id)[2]] = -0.03
        dofs_rotated = rotated.mesh.dof_manager.get_node_dofs(node_id)
        u_rotated[dofs_rotated[:3]] = rotation @ np.array((0.0, 0.0, -0.03))
    force, _K, result = _PlanarQ4PlaneManager(base, plane).evaluate(u)
    rotated_force, _Kr, rotated_result = _PlanarQ4PlaneManager(
        rotated, rotated_plane
    ).evaluate(u_rotated)
    assert rotated_result.penalty_energy == pytest.approx(result.penalty_energy)
    assert rotated_result.minimum_gap == pytest.approx(result.minimum_gap)
    np.testing.assert_allclose(rotated_result.shell_moment,
                               rotation @ np.asarray(result.shell_moment), atol=1e-9)
    for node_id in (1, 2, 3, 4):
        base_dofs = base.mesh.dof_manager.get_node_dofs(node_id)
        rotated_dofs = rotated.mesh.dof_manager.get_node_dofs(node_id)
        np.testing.assert_allclose(rotated_force[rotated_dofs[:3]],
                                   rotation @ force[base_dofs[:3]], atol=1e-9)
def test_offset_skin_generates_eccentric_moment_without_drill_work():
    model = _panel()
    definition = ContactDefinition(
        surfaces=(ContactSurface("top", (1,), side="top"),),
        rigid_bodies=(RigidContactBody("ramp", "plane", (0, 0, -0.1),
                                        (0.1, 0, 1)),),
        interactions=(ContactInteraction("oblique", "top", "ramp", 1.0e5,
                                         0.2),),
    )
    manager = _PlanarQ4PlaneManager(model, definition)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -0.2
    force, tangent, result = manager.evaluate(u)
    assert result.active_pressure_dofs > 0
    assert abs(result.shell_moment[1]) > 0
    np.testing.assert_allclose(np.asarray(result.shell_moment)
                               + np.asarray(result.rigid_moment), 0, atol=1e-10)
    rotation_y = model.mesh.dof_manager.get_node_dofs(2)[4]
    direction = np.zeros_like(u)
    direction[rotation_y] = 1.0
    h = 1e-6
    fd = (manager.evaluate(u + h * direction, tangent=False)[0]
          - manager.evaluate(u - h * direction, tangent=False)[0]) / (2 * h)
    np.testing.assert_allclose(fd, tangent @ direction, rtol=1e-8, atol=1e-6)
    for node_id in (1, 2, 3, 4):
        assert force[model.mesh.dof_manager.get_node_dofs(node_id)[5]] == 0.0
