"""Coupled nonmatching Q4/S3 partial-contact resolution controls."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.patch_sliding import _RestrictedSlidingPatchFEManager
from anyfesolver.nonlinear_static import _assemble_nonlinear_system_unchecked


@pytest.fixture(scope="module")
def accepted_patch_pose():
    divisions = 4
    model = FEModel("partial_q4_s3_patch_refinement")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node, (x, y) in enumerate(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), 1):
        model.add_node(node, x, y, 0.0)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3), "elastic", thickness=0.05,
        formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
        reference_surface_offset=0.01,
    ))

    def upper_node(i, j):
        return 4 + j * (divisions + 1) + i

    upper_nodes = []
    for j in range(divisions + 1):
        for i in range(divisions + 1):
            node = upper_node(i, j)
            model.add_node(node, 0.15 + i / divisions, 0.1 + j / divisions, 0.04)
            upper_nodes.append(node)
    source_elements = []
    load_weights = {node: 0.0 for node in upper_nodes}
    for j in range(divisions):
        for i in range(divisions):
            element = 2 + j * divisions + i
            nodes = (
                upper_node(i, j), upper_node(i + 1, j),
                upper_node(i + 1, j + 1), upper_node(i, j + 1),
            )
            model.add_element(element, create_shell_element(
                element, nodes, "elastic", thickness=0.05,
            ))
            source_elements.append(element)
            for node in nodes:
                load_weights[node] += 0.25 / divisions**2
    assert sum(load_weights.values()) == pytest.approx(1.0)
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (1, 2, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_normal_guide", tuple(upper_nodes),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("uniform_pressure_on_upper_square")
    for node, weight in load_weights.items():
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.4 * weight))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", tuple(source_elements), side="bottom"),
                  ContactSurface("target", (1,), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "source", "target", 4.0e6, 1.0e-5,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
        min_step_fraction=0.1,
    )
    return model, load, definition, settings, result


def test_refined_q4_patch_on_s3_meets_declared_pointwise_gap_budget(
    accepted_patch_pose,
):
    _model, _load, _definition, _settings, result = accepted_patch_pose
    assert result.status == "completed", (
        result.failure_reason, result.load_factor, result.info.get("general_contact"),
    )
    contact = result.info["general_contact"]
    assert contact["profile"] == "PLANAR_Q4_S3_PATCH_RESTRICTED_SLIDING_MORTAR_DEV_V0"
    assert contact["pressure_rank"] == 13
    assert contact["current_overlap_area"] == pytest.approx(0.28125)
    assert contact["gap_tolerance_met"]
    assert contact["conservative_lower_gap"] >= -1.0e-5
    assert contact["integration_force"][2] == pytest.approx(0.4, rel=1e-6)
    assert contact["target_force"][2] == pytest.approx(-0.4, rel=1e-6)
    assert result.steps[-1].support_reactions["target_fixed"][2] == pytest.approx(
        0.4, rel=1e-6,
    )


def _independent_triangle_overlap_integrals(patch, q, order):
    """Tensor Gauss over the known unit-triangle/square intersection.

    This oracle clips horizontal strips analytically. It does not use the
    production polygon clipping, common-refinement or triangle quadrature.
    """
    abscissae, weights = leggauss(order)
    source_count = len(patch.source_node_ids)
    target_count = len(patch.target_node_ids)
    support = np.zeros(source_count)
    weighted_gap = np.zeros(source_count)
    gradient = np.zeros((source_count, source_count + target_count))
    area = 0.0
    source_heights = q[6:6 + source_count]
    target_heights = q[6 + source_count:]
    target_face = patch.target_faces[0]
    target_columns = np.asarray([
        patch.target_index[node] for node in target_face.node_ids
    ])
    target_xy = target_face.geometry.corners[:, :2] + q[3:5]
    assert np.allclose(target_xy, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    target_z = target_face.geometry.corners[0, 2] + q[5]
    for face in patch.source_faces:
        corners = face.geometry.corners
        rows = np.asarray([patch.source_index[node] for node in face.node_ids])
        x0, x1 = float(corners[0, 0] + q[0]), float(corners[1, 0] + q[0])
        y0, y1 = float(corners[0, 1] + q[1]), float(corners[3, 1] + q[1])
        assert x0 < x1 and y0 < y1
        x_left, x_right = max(x0, 0.0), min(x1, 1.0 - max(y0, 0.0))
        if x_left >= x_right:
            continue
        for xa, xb in ((x_left, min(x_right, 1.0 - y1)),
                       (max(x_left, 1.0 - y1), x_right)):
            if xa >= xb:
                continue
            for gx, wx in zip(abscissae, weights):
                x = xa + 0.5 * (gx + 1.0) * (xb - xa)
                upper = min(y1, 1.0 - x)
                lower = max(y0, 0.0)
                if upper <= lower:
                    continue
                for gy, wy in zip(abscissae, weights):
                    y = lower + 0.5 * (gy + 1.0) * (upper - lower)
                    xi = (x - x0) / (x1 - x0)
                    eta = (y - y0) / (y1 - y0)
                    source_shape = np.array(((1 - xi) * (1 - eta),
                                             xi * (1 - eta), xi * eta,
                                             (1 - xi) * eta))
                    target_shape = np.array((1 - x - y, x, y))
                    assert np.min(source_shape) >= -1e-12
                    assert np.min(target_shape) >= -1e-12
                    weight = wx * wy * (xb - xa) * (upper - lower) / 4.0
                    gap = (float(patch.normal[2]) *
                           (float(corners[0, 2] + q[2]) - target_z)
                           + source_shape @ source_heights[rows]
                           - target_shape @ target_heights[target_columns])
                    area += weight
                    support[rows] += weight * source_shape
                    weighted_gap[rows] += weight * source_shape * gap
                    gradient[np.ix_(rows, rows)] += (
                        weight * np.outer(source_shape, source_shape)
                    )
                    gradient[np.ix_(rows, source_count + target_columns)] -= (
                        weight * np.outer(source_shape, target_shape)
                    )
    return area, support, weighted_gap, gradient


def test_accepted_q4_s3_pose_contact_integration_has_independent_error_budget(
    accepted_patch_pose,
):
    model, _load, definition, settings, result = accepted_patch_pose
    assert result.status == "completed"
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    q = manager._reduced_pose(result.displacements)
    source_count = len(manager.patch.source_node_ids)
    state = manager.patch.evaluate(
        q[:3], q[3:6], q[6:6 + source_count], q[6 + source_count:]
    )
    control = _independent_triangle_overlap_integrals(manager.patch, q, 12)
    refined = _independent_triangle_overlap_integrals(manager.patch, q, 16)
    for earlier, later in zip(control, refined):
        np.testing.assert_allclose(earlier, later, rtol=2e-12, atol=2e-13)
    area, support, weighted_gap, gradient = refined
    assert area == pytest.approx(0.28125, abs=1e-12)
    np.testing.assert_allclose(state.support, support, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(state.weighted_gap, weighted_gap,
                               rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(state.gap_gradient, gradient,
                               rtol=2e-12, atol=2e-13)
    active = support > 0
    gap = np.zeros_like(support)
    gap[active] = weighted_gap[active] / support[active]
    closed_gap = np.minimum(gap, 0.0)
    stiffness = manager.patch.stiffness
    energy = 0.5 * stiffness * float(support @ (closed_gap * closed_gap))
    residual = gradient.T @ (stiffness * closed_gap)
    force_scale = 0.4
    force_error = np.max(np.abs(state.normal_residual - residual)) / force_scale
    energy_error = abs(state.energy - energy) / max(energy, 1e-12)
    support_error = np.max(np.abs(state.support - support)) / area
    weighted_gap_error = np.max(np.abs(state.weighted_gap - weighted_gap)) / (
        area * 1.0e-5
    )
    gradient_error = np.max(np.abs(state.gap_gradient - gradient)) / area
    assert energy_error <= 0.002
    assert force_error <= 0.002
    moment_errors = []
    for faces, offset, count in ((manager.patch.source_faces, 0, source_count),
                                 (manager.patch.target_faces, source_count,
                                  len(manager.patch.target_node_ids))):
        points = {}
        for face in faces:
            points.update(zip(face.node_ids, face.geometry.corners))
        ids = (manager.patch.source_node_ids if offset == 0
               else manager.patch.target_node_ids)
        xy = np.asarray([points[node][:2] for node in ids])
        actual = state.normal_residual[offset:offset + count]
        expected = residual[offset:offset + count]
        moment_actual = np.array((xy[:, 1] @ actual, -xy[:, 0] @ actual))
        moment_expected = np.array((xy[:, 1] @ expected, -xy[:, 0] @ expected))
        moment_error = np.max(np.abs(moment_actual - moment_expected)) / force_scale
        moment_errors.append(moment_error)
        assert moment_error <= 0.002
    print(f"accepted-pose area={area:.12g} m^2; normalized support="
          f"{support_error:.3e}, weighted gap={weighted_gap_error:.3e}, "
          f"gradient={gradient_error:.3e}, force={force_error:.3e}, "
          f"moment={max(moment_errors):.3e}, energy={energy_error:.3e}")


def test_q4_s3_coupled_response_against_independent_contact_equilibrium(
    accepted_patch_pose,
):
    model, _load, definition, settings, result = accepted_patch_pose
    assert result.status == "completed"
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    patch = manager.patch
    normal_z = float(patch.normal[2])
    assert abs(normal_z) == pytest.approx(1.0)
    source_ids = patch.source_node_ids
    source_count = len(source_ids)
    free = np.asarray([
        model.mesh.dof_manager.get_node_dofs(node)[2] for node in source_ids
    ], dtype=np.intp)
    external = np.zeros(model.mesh.dof_manager.total_dofs)
    incidence = {node: 0 for node in source_ids}
    for face in patch.source_faces:
        for node in face.node_ids:
            incidence[node] += 1
    for index, node in enumerate(source_ids):
        external[free[index]] = -0.4 * incidence[node] / (
            4 * len(patch.source_faces)
        )
    assert np.sum(external[free]) == pytest.approx(-0.4)

    displacement = np.zeros_like(result.displacements)
    displacement[free] = 1.05 * result.displacements[free]
    residual_ratio = float("inf")
    for iteration in range(12):
        structural_force, structural_tangent, _trial = (
            _assemble_nonlinear_system_unchecked(
                model, displacement, {}, 5, tangent=True,
                kinematics="von_karman",
            )
        )
        q = np.r_[np.zeros(6), normal_z * displacement[free],
                  np.zeros(len(patch.target_node_ids))]
        _area, support, weighted_gap, gradient = (
            _independent_triangle_overlap_integrals(patch, q, 16)
        )
        active = support > 0.0
        normalized = np.zeros(source_count)
        normalized[active] = weighted_gap[active] / support[active]
        closed = active & (normalized < 0.0)
        compression = np.minimum(normalized, 0.0)
        contact_force = gradient.T @ (patch.stiffness * compression)
        contact_tangent = (patch.stiffness * gradient[closed].T
                           @ np.diag(1.0 / support[closed]) @ gradient[closed])
        residual = (structural_force[free] + normal_z * contact_force[:source_count]
                    - external[free])
        residual_ratio = float(np.max(np.abs(residual))) / 0.4
        if residual_ratio <= 1.0e-6:
            break
        tangent = structural_tangent[np.ix_(free, free)].toarray()
        tangent += contact_tangent[:source_count, :source_count]
        displacement[free] -= np.linalg.solve(tangent, residual)
    assert residual_ratio <= 1.0e-6
    displacement_error = (
        np.max(np.abs(displacement[free] - result.displacements[free]))
        / max(np.max(np.abs(result.displacements[free])), 1.0e-12)
    )
    assert displacement_error <= 0.001
    print(f"independent coupled equilibrium: {iteration + 1} iterations, "
          f"force residual={residual_ratio:.3e}, "
          f"displacement error={displacement_error:.3e}")


def test_q4_s3_fixed_mesh_penalty_enforcement_sensitivity(accepted_patch_pose):
    model, load, definition, settings, baseline = accepted_patch_pose
    summaries = [(4.0e6, baseline)]
    for stiffness in (8.0e6, 16.0e6):
        interaction = replace(definition.interactions[0],
                              normal_stiffness=stiffness)
        candidate = replace(definition, interactions=(interaction,))
        result = solve_static_nonlinear(
            model, load, contact=candidate, contact_settings=settings,
            kinematics="von_karman", num_steps=1, max_iterations=12,
            min_step_fraction=0.1,
        )
        assert result.status == "completed", (
            stiffness, result.failure_reason, result.info.get("general_contact"),
        )
        summaries.append((stiffness, result))
    lower_gaps = []
    source_ids = tuple(sorted({
        node for element_id in definition.surfaces[0].element_ids
        for node in model.mesh.elements[element_id].node_ids
    }))
    source_z_dofs = np.asarray([
        model.mesh.dof_manager.get_node_dofs(node)[2] for node in source_ids
    ], dtype=np.intp)
    for stiffness, result in summaries:
        contact = result.info["general_contact"]
        assert contact["gap_tolerance_met"]
        assert contact["integration_force"][2] == pytest.approx(0.4, rel=1e-6)
        assert contact["target_force"][2] == pytest.approx(-0.4, rel=1e-6)
        lower_gaps.append(contact["conservative_lower_gap"])
        mean_source_z = float(np.mean(result.displacements[source_z_dofs]))
        print(f"stiffness={stiffness:.0f} Pa/m; certified lower gap="
              f"{contact['conservative_lower_gap'] * 1e6:.6f} um; "
              f"mean source uz={mean_source_z * 1e6:.6f} um; "
              f"penalty storage={contact['penalty_energy']:.9g} J")
    assert lower_gaps[0] < lower_gaps[1] < lower_gaps[2] < 0.0
