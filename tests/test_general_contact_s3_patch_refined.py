"""Development comparison for pair-local S3 mortar derivatives."""

import numpy as np
from dataclasses import replace

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathProgram, ContactSolverSettings,
    ContactSurface, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel

from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty, S3PatchFace,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_s3_patch_augmented import _augmented_case


def test_pair_local_derivatives_match_whole_patch_smooth_branch():
    model, definition, settings = _augmented_case()
    manager = GeneralS3PatchManager(model, definition, settings)
    whole = manager.law
    geometry = whole.geometry
    local = MovingS3PatchProjectedNormal(MovingS3PatchPrimalMortarPenalty(
        geometry.source_faces, geometry.target_faces, geometry.stiffness,
        difference_step=geometry.difference_step,
        differentiation="pair_local", maximum_candidate_pairs=4,
        maximum_energy_evaluations=2600,
    ))
    u = np.zeros(manager.n_dof)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.016
    source, target = manager._positions(u)
    multipliers = np.full(4, 10.0)
    global_work = whole.work(source, target, multipliers)
    local_work = local.work(source, target, multipliers)
    np.testing.assert_allclose(local_work.state.reference_support,
                               global_work.state.reference_support, atol=1e-14)
    np.testing.assert_allclose(local_work.residual, global_work.residual,
                               rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(local_work.tangent, global_work.tangent,
                               rtol=1e-4, atol=1e-4)
    assert local_work.energy_evaluations <= 2600


def _refinement_case(corner_size):
    a = (0.0, 0.0)
    b = (1.0, 0.0)
    c = (1.0, 1.0)
    d = (0.0, 1.0)
    e = (corner_size, 0.0)
    f = (0.0, corner_size)
    g = (corner_size, corner_size)
    coarse_xy = np.asarray((a, b, c, d))
    refined_xy = np.asarray((a, e, g, f, b, c, d))
    target_xy = np.asarray(((-.2, -.3), (1.3, -.2),
                            (1.2, 1.3), (-.3, 1.2)))
    coarse_reference = np.column_stack((coarse_xy, np.full(4, .02)))
    refined_reference = np.column_stack((refined_xy, np.full(7, .02)))
    target_reference = np.column_stack((target_xy, np.zeros(4)))
    coarse_faces = (
        S3PatchFace((1, 2, 3), coarse_reference[[0, 1, 2]]),
        S3PatchFace((1, 3, 4), coarse_reference[[0, 2, 3]]),
    )
    refined_triangles = ((1, 2, 3), (1, 3, 4), (2, 5, 6),
                         (2, 6, 3), (4, 3, 6), (4, 6, 7))
    refined_faces = tuple(S3PatchFace(
        nodes, refined_reference[[node-1 for node in nodes]],
    ) for nodes in refined_triangles)
    target_faces = (
        S3PatchFace((101, 102, 103), target_reference[[0, 1, 2]]),
        S3PatchFace((101, 103, 104), target_reference[[0, 2, 3]]),
    )
    def deformed(reference):
        result = reference.copy()
        result[:, 2] = -.0001+.0011*(result[:, 0]+result[:, 1])
        return result
    return (coarse_faces, refined_faces, target_faces,
            deformed(coarse_reference), deformed(refined_reference),
            target_reference)


def test_refined_pressure_space_detects_local_penetration():
    coarse_faces, refined_faces, target_faces, coarse, refined, target = (
        _refinement_case(.1)
    )
    coarse_law = MovingS3PatchPrimalMortarPenalty(
        coarse_faces, target_faces, 1000.0, difference_step=2e-5,
    )
    refined_law = MovingS3PatchPrimalMortarPenalty(
        refined_faces, target_faces, 1000.0, difference_step=2e-5,
        differentiation="pair_local", maximum_candidate_pairs=12,
        maximum_energy_evaluations=7800,
    )
    coarse_state = coarse_law.evaluate(coarse, target)
    refined_state = refined_law.evaluate(refined, target)
    independent_support = np.zeros(7)
    independent_gram = np.zeros((7, 7))
    for face in refined_faces:
        corners = face.reference_positions
        area = .5*np.linalg.norm(np.cross(corners[1]-corners[0],
                                             corners[2]-corners[0]))
        nodes = np.asarray(face.node_ids)-1
        independent_support[nodes] += area/3
        independent_gram[np.ix_(nodes, nodes)] += (
            area/12*(np.ones((3, 3))+np.eye(3))
        )
    np.testing.assert_allclose(refined_state.reference_support,
                               independent_support, atol=1e-12)
    np.testing.assert_allclose(refined_state.pressure_gram,
                               independent_gram, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(refined_state.pressure_gram)) > 0.0
    np.testing.assert_allclose(coarse_state.reference_support.sum(), 1.0,
                               atol=1e-12)
    np.testing.assert_allclose(refined_state.reference_support.sum(), 1.0,
                               atol=1e-12)
    assert min(coarse_state.normalized_gap) > 0.0
    assert coarse_state.active_pressure_dofs == 0
    assert refined_state.normalized_gap[0] < 0.0
    assert refined_state.pressure[0] > 0.0
    assert refined_state.minimum_physical_gap < 0.0


def _refined_model():
    _coarse_faces, source_faces, target_faces, _coarse, current, target = (
        _refinement_case(.1)
    )
    model = FEModel("refined_s3_augmented_patch")
    model.add_material("elastic", 1e5, .3, density=1.0)
    source_nodes = {
        node: position for face in source_faces
        for node, position in zip(face.node_ids, face.reference_positions)
    }
    target_nodes = {
        node: position for face in target_faces
        for node, position in zip(face.node_ids, face.reference_positions)
    }
    for node, position in source_nodes.items():
        model.add_node(node, *position[:2], position[2]+.035)
    for node, position in target_nodes.items():
        model.add_node(node, *position[:2], position[2]-.015)
    for element_id, face in enumerate((*source_faces, *target_faces), 1):
        model.add_element(element_id, create_shell_element(
            element_id, face.node_ids, "elastic", thickness=.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=.01,
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", tuple(range(1, 7)), "bottom"),
                  ContactSurface("target", (7, 8), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "refined", "source", "target", 1000.0, 5e-5,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5,
        maximum_general_s3_energy_evaluations=7800,
        maximum_candidate_pairs=12, pressure_tolerance=1e-3,
    )
    return model, definition, settings, current, target


def test_refined_patch_public_owner_maps_pressure_and_checks_gap():
    model, definition, settings, current, _target = _refined_model()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.profile == report.profile
    displacement = np.zeros(manager.n_dof)
    for index, node in enumerate(manager.law.source_node_ids):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = (
            current[index, 2]-.02
        )
    force, tangent, result = manager.evaluate(displacement)
    assert result.active_pressure_dofs >= 1
    assert result.numerical_pressure_pa[0] > 0.0
    assert result.minimum_gap < -5e-5
    assert not result.gap_tolerance_met
    assert manager.projected_normal_multipliers(displacement)[0] > 0.0
    assert tangent.shape == (manager.n_dof, manager.n_dof)
    assert np.linalg.norm(force) > 0.0
    assert result.energy_evaluations <= settings.maximum_general_s3_energy_evaluations
    assert result.pair_geometry_evaluations == result.energy_evaluations+12
    np.testing.assert_allclose(np.asarray(result.integration_force)+result.target_force,
                               0.0, atol=1e-7)
    direction = np.zeros_like(displacement)
    direction[model.mesh.dof_manager.get_node_dofs(1)[2]] = 1.0
    step = 1e-6
    plus = manager.evaluate(displacement+step*direction, tangent=False)[0]
    minus = manager.evaluate(displacement-step*direction, tangent=False)[0]
    reference = (plus-minus)/(2*step)
    relative_error = np.linalg.norm(tangent@direction-reference)/np.linalg.norm(reference)
    assert relative_error <= 1e-5


def test_refined_patch_resource_budgets_fail_closed():
    model, definition, settings, _current, _target = _refined_model()
    too_few_pairs = validate_general_contact(
        model, definition, replace(settings, maximum_candidate_pairs=11),
    )
    assert not too_few_pairs.valid
    assert "candidate pair" in " ".join(too_few_pairs.issues)
    too_few_evaluations = validate_general_contact(
        model, definition,
        replace(settings, maximum_general_s3_energy_evaluations=7799),
    )
    assert not too_few_evaluations.valid
    assert "evaluation budget" in " ".join(too_few_evaluations.issues)


def _coupled_case():
    model, definition, settings, _current, _target = _refined_model()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=0.005),
    ))
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (101, 102, 103, 104),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4, 5, 6, 7),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.0205)), steps_per_segment=1,
    )
    return model, definition, settings, program


def test_refined_patch_couples_through_static_path():
    model, definition, settings, program = _coupled_case()
    # The controlled material node is driven 0.5 mm through the obstacle;
    # this development pressure-change budget is explicit and permissive.
    settings = replace(settings, pressure_tolerance=1.0)
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    assert result.info["contact_path"]["profile"] == (
        "S3_PATCH_REFINED_AUGMENTED_PATH_DEV_V1"
    )
    assert result.info["general_contact"]["gap_tolerance_met"]
    assert result.info["general_contact"]["active_pressure_dofs"] > 0
    assert result.restart_checkpoint["version"] == 2


def test_refined_patch_exhausted_augmentation_rolls_back():
    model, definition, settings, program = _coupled_case()
    settings = replace(settings, maximum_augmentations=1)
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
    )
    assert result.status == "diverged"
    assert result.failure_reason == "maximum_augmentations_reached"
    np.testing.assert_array_equal(result.displacements, 0.0)
    np.testing.assert_array_equal(
        result.info["general_contact"]["normal_multiplier_estimate_pa"], 0.0,
    )
