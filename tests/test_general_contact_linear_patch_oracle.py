"""Independent Q1 mortar calculation for a bounded Q4/S3 patch.

The contact oracle integrates the Q1 pressure basis analytically. The
production kernel is queried only after its expected active set and resultant
have been calculated independently.
"""

import numpy as np
from dataclasses import replace

from anyfesolver import (
    BoundaryCondition,
    ContactDefinition,
    ContactInteraction,
    ContactSolverSettings,
    ContactSurface,
    LoadCase,
    create_shell_element,
    solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.core import contact_scope


def _case_and_local_check(*, target_left_x=-0.2):
    gap = 50.0e-6
    selected_displacement = 90.0e-6
    epsilon = 100_000.0
    model = FEModel("q4_s3_linear_clipped_overlap_oracle")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node, x, y, z in (
        (1, 0.0, 0.0, 0.04005),
        (2, 0.5, 0.0, 0.04005),
        (3, 1.0, 0.0, 0.04005),
        (4, 0.0, 1.0, 0.04005),
        (5, 0.5, 1.0, 0.04005),
        (6, 1.0, 1.0, 0.04005),
        (7, target_left_x, -0.1, 0.0),
        (8, 1.2, -0.1, 0.0),
        (9, 1.2, 1.1, 0.0),
        (10, target_left_x, 1.1, 0.0),
    ):
        model.add_node(node, x, y, z)
    for element_id, nodes in ((1, (1, 2, 5, 4)),
                              (2, (2, 3, 6, 5))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            reference_normal=(0.0, 0.0, 1.0),
        ))
    for element_id, nodes in ((3, (7, 8, 10)), (4, (8, 9, 10))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d",
            reference_normal=(0.0, 0.0, 1.0),
            reference_surface_offset=0.01,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "fixed_target", (7, 8, 9, 10),
        {axis: 0.0 for axis in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "guided_source", (1, 2, 3, 4, 5, 6),
        {axis: 0.0 for axis in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "supported_left_edge", (1, 4), {"uz": 0.0},
    ))

    # Each Q4 is a half-width rectangle. Independently assemble the exact
    # Q1 mass matrix and positive support over the two source faces.
    one_dimensional_mass = np.array([[1.0 / 3.0, 1.0 / 6.0],
                                      [1.0 / 6.0, 1.0 / 3.0]])
    q1_corners = ((0, 0), (1, 0), (1, 1), (0, 1))
    mortar_mass = np.zeros((6, 6))
    for face_index, nodes in enumerate(((0, 1, 4, 3), (1, 2, 5, 4))):
        # Exact integrals of (1-t)^2, t(1-t), and t^2 on the clipped
        # reference x interval. This oracle does not query contact geometry.
        lower = max(0.0, min(1.0, (target_left_x-0.5*face_index)/0.5))
        primitive = lambda t: np.array((
            t-t*t+t**3/3.0, t*t/2.0-t**3/3.0, t**3/3.0,
        ))
        a, b, c = primitive(1.0)-primitive(lower)
        x_mass = np.array(((a, b), (b, c)))
        local_mass = np.array([
            [x_mass[ix, jx] * one_dimensional_mass[iy, jy]
             for jx, jy in q1_corners]
            for ix, iy in q1_corners
        ])
        mortar_mass[np.ix_(nodes, nodes)] += 0.5 * local_mass
    support = np.sum(mortar_mass, axis=1)
    if target_left_x > 0.0:
        np.testing.assert_allclose(
            support, (0.045, 0.23, 0.125, 0.045, 0.23, 0.125),
            rtol=0.0, atol=1.0e-14,
        )
    prescribed_shape = np.array((0.0, -0.5, -1.0, 0.0, -0.5, -1.0))
    weighted_gap = gap + mortar_mass @ (
        prescribed_shape * selected_displacement
    ) / support
    pressure = np.maximum(0.0, -epsilon * weighted_gap)
    np.testing.assert_array_equal(
        pressure > 0.0, (False, False, True, False, False, True)
    )
    expected_contact_force = float(np.dot(support, pressure))
    expected_right_contact_force = float(np.sum(
        (mortar_mass @ pressure)[[2, 5]]
    ))

    definition = ContactDefinition(
        surfaces=(
            ContactSurface("q4_source", (1, 2), "bottom"),
            ContactSurface("s3_target", (3, 4), "top"),
        ),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "q4_source", "s3_target", epsilon, 50.0e-6,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        q4_source_gap_certificate="projected_matching",
        general_s3_difference_step=2.0e-5,
        maximum_general_s3_energy_evaluations=5000,
        q4_source_area_tolerance=1.0e-9,
        q4_source_weighted_gap_tolerance=1.0e-10,
        maximum_q4_source_integration_triangles=128,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
        pressure_tolerance=1.0e-5,
        maximum_augmentations=7,
    )
    capability = validate_general_contact(model, definition, settings)
    assert capability.valid, capability.issues
    with contact_scope(model, definition, settings,
                       allow_augmented=True) as manager:
        displacement = np.zeros(manager.n_dof)
        for node, shape in enumerate(prescribed_shape, start=1):
            displacement[model.mesh.get_node(node).dofs[2]] = (
                shape * selected_displacement
            )
        contact_force, contact_tangent, contact = manager.evaluate(
            displacement, tangent=True,
        )
    np.testing.assert_array_equal(contact.source_pressure_node_ids,
                                  (1, 2, 3, 4, 5, 6))
    np.testing.assert_allclose(contact.reference_support_m2, support,
                               rtol=1.0e-8,
                               atol=settings.q4_source_area_tolerance)
    np.testing.assert_allclose(contact.normalized_weighted_gap_m,
                               weighted_gap, rtol=1.0e-3, atol=1.0e-8)
    np.testing.assert_allclose(contact.numerical_pressure_pa,
                               pressure, rtol=1.0e-3, atol=1.0e-6)
    np.testing.assert_allclose(
        contact.integration_force[2], expected_contact_force,
        rtol=1.0e-3, atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(contact.integration_force)
        + np.asarray(contact.target_force),
        0.0, atol=1.0e-8,
    )
    np.testing.assert_allclose(
        -sum(contact_force[model.mesh.get_node(node).dofs[2]]
             for node in (3, 6)),
        expected_right_contact_force,
        rtol=1.0e-3, atol=1.0e-6,
    )
    active = (pressure > 0.0).astype(float)
    expected_contact_tangent = (
        epsilon * mortar_mass @ np.diag(active / support) @ mortar_mass
    )
    source_normal_dofs = [model.mesh.get_node(node).dofs[2]
                          for node in range(1, 7)]
    actual_normal_tangent = contact_tangent[np.ix_(
        source_normal_dofs, source_normal_dofs,
    )].toarray()
    assert (np.linalg.norm(actual_normal_tangent-expected_contact_tangent)
            / np.linalg.norm(expected_contact_tangent)) <= 1.0e-5
    right_dofs = [model.mesh.get_node(node).dofs[2] for node in (3, 6)]
    np.testing.assert_allclose(
        np.sum(contact_tangent[np.ix_(right_dofs, right_dofs)]),
        np.sum(expected_contact_tangent[np.ix_([2, 5], [2, 5])]),
        rtol=1.0e-3, atol=1.0e-3,
    )
    assert contact.gap_tolerance_met
    assert contact.active_pressure_dofs == 2
    return model, definition, settings, mortar_mass, support


def test_q4_against_nonmatching_s3_partial_pressure_oracle():
    _case_and_local_check()


def test_q4_against_s3_clipped_overlap_pressure_oracle():
    _case_and_local_check(target_left_x=0.2)


def _loaded_augmented_patch_oracle(*, target_left_x=-0.2):
    model, definition, settings, mortar_mass, support = _case_and_local_check(
        target_left_x=target_left_x,
    )
    structural = FEModel("structural_only_linear_oracle")
    structural.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node in range(1, 7):
        position = model.mesh.get_node(node).coords()
        structural.add_node(node, *map(float, position))
    stiffness = np.zeros((36, 36))
    for element_id, nodes in ((1, (1, 2, 5, 4)),
                              (2, (2, 3, 6, 5))):
        element = create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            reference_normal=(0.0, 0.0, 1.0),
        )
        structural.add_element(element_id, element)
        local = element.compute_stiffness_matrix(
            structural.mesh, structural.get_material("elastic")
        )
        indices = [6 * (node - 1) + axis for node in nodes for axis in range(6)]
        stiffness[np.ix_(indices, indices)] += local

    if target_left_x > 0.0:
        expected_pressure = np.array((0.0, 0.1, 0.2, 0.0, 0.1, 0.2))
        # Two independent weighted-gap constraints on the clipped mortar
        # domain determine the middle and right nodal displacements.
        nodal_shape = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                                (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        normal_map = mortar_mass @ nodal_shape / support[:, None]
        middle_right = np.linalg.solve(
            normal_map[[1, 2]], -50.0e-6*np.ones(2),
        )
        expected_normal = nodal_shape @ middle_right
        expected_minimum_gap = 50.0e-6 + float(np.min(middle_right))
    else:
        expected_pressure = np.array((0.0, 0.0, 0.1, 0.0, 0.0, 0.1))
        expected_normal = -60.0e-6*np.array((0.0, 0.5, 1.0,
                                               0.0, 0.5, 1.0))
        expected_minimum_gap = -1.0e-5
    q = np.zeros(36)
    for node in range(1, 7):
        q[6 * (node - 1) + 2] = expected_normal[node-1]
    normalized_gap = 50.0e-6 + mortar_mass @ q[2::6] / support
    active_nodes = np.flatnonzero(expected_pressure > 0.0)
    np.testing.assert_allclose(normalized_gap[active_nodes], 0.0,
                               atol=1.0e-15)
    assert np.min(normalized_gap[expected_pressure == 0.0]) > 0.0
    source_contact_force = mortar_mass @ expected_pressure
    external_force = stiffness @ q
    external_force[2::6] -= source_contact_force
    load = LoadCase("manufactured_augmented_right_edge")
    for node in (2, 3, 5, 6):
        load.add_nodal_load(node, forces=(
            0.0, 0.0, float(external_force[6 * (node - 1) + 2]),
        ))
    settings = replace(
        settings, augmentation_acceleration="supported_schur_dev",
        augmentation_schur_maximum_modes=6,
        augmentation_schur_maximum_pressure_pa=10.0,
        augmentation_schur_maximum_step_pa=5.0,
        augmentation_schur_difference_step_m=2.0e-6,
        augmentation_schur_maximum_attempts=2,
        augmentation_schur_maximum_relinearizations=0,
    )
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=1,
        max_iterations=12, min_step_fraction=0.25,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    np.testing.assert_allclose(
        [result.displacements[model.mesh.get_node(node).dofs[2]]
         for node in range(1, 7)],
        q[2::6], rtol=1.0e-3, atol=1.0e-8,
    )
    contact = result.info["general_contact"]
    np.testing.assert_allclose(
        contact["integration_force"][2],
        float(np.dot(support, expected_pressure)),
        rtol=1.0e-3, atol=1.0e-6,
    )
    assert contact["gap_tolerance_met"]
    assert result.restart_checkpoint is not None
    actual_uz = [float(result.displacements[
        model.mesh.get_node(node).dofs[2]
    ]) for node in range(1, 7)]
    reference_force = float(np.dot(support, expected_pressure))
    actual_linear_q = np.zeros_like(q)
    actual_linear_q[2::6] = actual_uz
    independent_linear_residual = stiffness @ actual_linear_q - external_force
    independent_linear_residual[2::6] -= source_contact_force
    free_normal = [6*(node-1)+2 for node in (2, 3, 5, 6)]
    expected_source_moment = np.zeros(3)
    for node, force in enumerate(source_contact_force, start=1):
        position = structural.mesh.get_node(node).coords()
        expected_source_moment += np.cross(
            position, (0.0, 0.0, float(force)),
        )
    final_trace = result.info["contact_augmentation_attempts"][-1]["trace"][-1]
    force_wrench_balance = float(np.linalg.norm(
        np.asarray(contact["integration_force"])
        + np.asarray(contact["target_force"])
    ))
    moment_wrench_balance = float(np.linalg.norm(
        np.asarray(contact["integration_moment"])
        + np.asarray(contact["target_moment"])
    ))
    return {
        "status": result.status,
        "accepted_steps": len(result.steps),
        "expected_source_uz_m": q[2::6].tolist(),
        "actual_source_uz_m": actual_uz,
        "maximum_absolute_uz_error_m": float(np.max(np.abs(
            np.asarray(actual_uz) - q[2::6]
        ))),
        "expected_integrated_normal_force_n": reference_force,
        "actual_integrated_normal_force_n": float(
            contact["integration_force"][2]
        ),
        "relative_integrated_force_error": float(abs(
            contact["integration_force"][2] - reference_force
        ) / reference_force),
        "actual_pressure_pa": contact["numerical_pressure_pa"],
        "expected_pressure_pa": expected_pressure.tolist(),
        "independent_source_support_m2": support.tolist(),
        "actual_source_support_m2": contact["reference_support_m2"],
        "source_pressure_node_ids": contact["source_pressure_node_ids"],
        "pressure_gram_supported_modes": contact.get("pressure_gram_supported_modes"),
        "pressure_gram_scaled_rank": contact.get("pressure_gram_scaled_rank"),
        "area_integration_error_m2": contact["area_integration_error_m2"],
        "weighted_gap_integration_error_m3": contact[
            "weighted_gap_integration_error_m3"
        ],
        "actual_minimum_gap_m": contact["minimum_gap"],
        "expected_minimum_gap_m": expected_minimum_gap,
        "gap_tolerance_met": bool(contact["gap_tolerance_met"]),
        "active_pressure_dofs": int(contact["active_pressure_dofs"]),
        "equilibrium_residual_norm_n": float(
            final_trace["equilibrium_residual_norm"]
        ),
        "normal_multiplier_change_pa": float(
            final_trace["normal_multiplier_change_pa"]
        ),
        "force_wrench_balance_n": force_wrench_balance,
        "moment_wrench_balance_nm": moment_wrench_balance,
        "independent_linear_free_residual_norm_n": float(np.linalg.norm(
            independent_linear_residual[free_normal]
        )),
        "expected_source_moment_nm": expected_source_moment.tolist(),
        "actual_source_moment_nm": contact["integration_moment"],
        "has_restart_checkpoint": result.restart_checkpoint is not None,
        "contact_augmentation_attempts": result.info.get(
            "contact_augmentation_attempts"
        ),
    }


def test_loaded_augmented_patch_matches_independent_linear_solution():
    metrics = _loaded_augmented_patch_oracle()
    np.testing.assert_allclose(
        metrics["actual_pressure_pa"], metrics["expected_pressure_pa"],
        rtol=0.0, atol=1.0e-4,
    )
    assert metrics["active_pressure_dofs"] == 2
    assert abs(metrics["actual_minimum_gap_m"]
               - metrics["expected_minimum_gap_m"]) <= 1.0e-9
    assert metrics["equilibrium_residual_norm_n"] <= 1.0e-6 * 0.025
    assert metrics["normal_multiplier_change_pa"] <= 1.0e-5
    assert metrics["force_wrench_balance_n"] <= 1.0e-6 * 0.025
    assert metrics["moment_wrench_balance_nm"] <= 1.0e-6 * 0.025


def test_loaded_clipped_patch_matches_independent_linear_solution():
    metrics = _loaded_augmented_patch_oracle(target_left_x=0.2)
    np.testing.assert_allclose(
        metrics["actual_pressure_pa"], metrics["expected_pressure_pa"],
        rtol=0.0, atol=1.0e-4,
    )
    assert metrics["active_pressure_dofs"] == 4
    assert abs(metrics["actual_minimum_gap_m"]
               - metrics["expected_minimum_gap_m"]) <= 1.0e-8
    assert metrics["relative_integrated_force_error"] <= 1.0e-3
    scale = metrics["expected_integrated_normal_force_n"]
    assert metrics["equilibrium_residual_norm_n"] <= 1.0e-6*scale
    assert metrics["force_wrench_balance_n"] <= 1.0e-6*scale
    assert metrics["moment_wrench_balance_nm"] <= 1.0e-6*scale
    assert metrics["independent_linear_free_residual_norm_n"] <= 1.0e-3*scale
    np.testing.assert_allclose(
        metrics["actual_source_moment_nm"],
        metrics["expected_source_moment_nm"],
        rtol=1.0e-3, atol=1.0e-3*scale,
    )
    assert metrics["normal_multiplier_change_pa"] <= 1.0e-5
    assert metrics["gap_tolerance_met"]
    assert metrics["has_restart_checkpoint"]
