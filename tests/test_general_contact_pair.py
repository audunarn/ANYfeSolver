"""Integrated controls for the exploratory guided Q4/Q4 pair slice."""

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
    create_shell_element,
    solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.pair import _PlanarShellPairManager
from anyfesolver.general_contact.patch_sliding import _RestrictedSlidingPatchFEManager


def _guided_pair(*, split_target=False, target_shift=0.0,
                 target_y_shift=0.0,
                 director=None, lateral_free=False):
    model = FEModel("guided_shell_pair")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    if split_target:
        lower_positions = ((0, 0, 0), (0.5, 0, 0), (1, 0, 0),
                           (0, 1, 0), (0.5, 1, 0), (1, 1, 0))
        lower_elements = ((1, (1, 2, 5, 4)), (2, (2, 3, 6, 5)))
        upper_nodes = (7, 8, 9, 10)
        upper_element = 3
    else:
        lower_positions = ((0, 0, 0), (1, 0, 0),
                           (1, 1, 0), (0, 1, 0))
        lower_elements = ((1, (1, 2, 3, 4)),)
        upper_nodes = (5, 6, 7, 8)
        upper_element = 2
    offset = 0.05 * (np.asarray(director, dtype=float)
                     if director is not None else np.array((0.0, 0.0, 1.0)))
    upper_positions = tuple(tuple(np.asarray((x, y, 0.0)) + offset)
                            for x, y in ((0, 0), (1, 0), (1, 1), (0, 1)))
    lower_positions = tuple((x + target_shift, y + target_y_shift, z)
                            for x, y, z in lower_positions)
    for node_id, position in enumerate((*lower_positions, *upper_positions), 1):
        model.add_node(node_id, *position)
    for element_id, nodes in lower_elements:
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            reference_normal=director,
        ))
    model.add_element(upper_element, create_shell_element(
        upper_element, upper_nodes, "elastic", thickness=0.05,
        reference_normal=director,
    ))
    lower_nodes = tuple(range(1, len(lower_positions) + 1))
    model.add_boundary_condition(BoundaryCondition(
        "lower", lower_nodes,
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "guide", upper_nodes,
        {name: 0.0 for name in (("uy", "rx", "ry", "rz") if lateral_free
                                 else ("ux", "uy", "rx", "ry", "rz"))},
    ))
    load = LoadCase("push_upper_shell")
    for node_id in upper_nodes:
        load.add_nodal_load(node_id, forces=(0.0, 0.0, -0.1))
    definition = ContactDefinition(
        surfaces=(ContactSurface("upper", (upper_element,), side="bottom"),
                  ContactSurface("lower", tuple(item[0] for item in lower_elements),
                                 side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "upper_lower", "upper", "lower", 1.0e6, 1.0e-6,
        ),),
    )
    return model, load, definition, upper_nodes


def test_restricted_sliding_patch_runs_through_nonlinear_static_and_restart():
    from dataclasses import replace

    model, load, definition, _upper_nodes = _guided_pair(
        split_target=True, target_shift=0.2, target_y_shift=0.1,
    )
    definition = replace(
        definition,
        interactions=(replace(definition.interactions[0], gap_tolerance=6.0e-6),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert "PATCH" in report.profile
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    assert len(manager.patch.target_faces) == 2
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=1, max_iterations=12, kinematics="von_karman",
    )
    assert result.status == "completed", (
        result.failure_reason, result.load_factor, result.info.get("general_contact"),
    )
    summary = result.info["general_contact"]
    assert summary["overlapping_pairs"] == 2
    assert summary["current_overlap_area"] == pytest.approx(0.72)
    assert summary["integration_force"][2] == pytest.approx(0.4, rel=1e-5)
    assert summary["target_force"][2] == pytest.approx(-0.4, rel=1e-5)
    assert summary["gap_tolerance_met"]
    half = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=1, max_iterations=12, kinematics="von_karman",
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=1, max_iterations=12, kinematics="von_karman",
        max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(
        resumed.displacements, result.displacements, rtol=1e-7, atol=1e-9,
    )


def test_restricted_sliding_patch_follows_lateral_displacement_path():
    from dataclasses import replace

    model, press, definition, upper_nodes = _guided_pair(
        split_target=True, target_shift=0.2, target_y_shift=0.1,
        lateral_free=True,
    )
    definition = replace(
        definition,
        interactions=(replace(definition.interactions[0], gap_tolerance=6.0e-6),),
    )
    master = int(model.mesh.dof_manager.get_node_dofs(upper_nodes[0])[0])
    for node in upper_nodes[1:]:
        slave = int(model.mesh.dof_manager.get_node_dofs(node)[0])
        model.add_constraint_equation(
            terms=((slave, 1.0), (master, -1.0)),
            source_id=f"upper_uniform_x_{node}",
        )
    drive = LoadCase("lateral_drive")
    drive.add_nodal_load(upper_nodes[0], forces=(1.0, 0.0, 0.0))
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    result = solve_static_nonlinear(
        model, drive, constant_load_case=press,
        contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=upper_nodes[0], dof="ux", target_displacement=0.1,
        ),
        num_steps=2, max_iterations=12, kinematics="von_karman",
    )
    assert result.status == "completed", (
        result.failure_reason, result.load_factor, result.info.get("general_contact"),
    )
    summary = result.info["general_contact"]
    assert summary["overlapping_pairs"] == 2
    assert summary["current_overlap_area"] == pytest.approx(0.81)
    assert summary["integration_force"][2] == pytest.approx(0.4, rel=1e-5)
    assert summary["target_force"][2] == pytest.approx(-0.4, rel=1e-5)
    assert summary["gap_tolerance_met"]


@pytest.mark.parametrize("split_target", (False, True))
def test_guided_pair_solve_balances_both_shells_and_matches_scalar_reference(split_target):
    model, load, definition, _upper_nodes = _guided_pair(split_target=split_target)
    report = validate_general_contact(model, definition)
    assert report.valid
    assert report.profile == "PLANAR_Q4_PAIR_NORMAL_TRANSLATION_PENALTY_MORTAR_V0"
    result = solve_static_nonlinear(
        model, load, contact=definition, num_steps=1, max_iterations=12,
        kinematics="von_karman",
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["reference_overlap_area"] == pytest.approx(1.0)
    assert summary["minimum_gap"] == pytest.approx(-0.4 / 1.0e6, abs=1e-9)
    assert summary["integration_force"] == pytest.approx((0.0, 0.0, 0.4), abs=1e-8)
    assert summary["target_force"] == pytest.approx((0.0, 0.0, -0.4), abs=1e-8)
    np.testing.assert_allclose(
        np.asarray(summary["integration_moment"])
        + np.asarray(summary["target_moment"]), 0.0, atol=1e-8
    )
    assert result.steps[-1].support_reactions["lower"][2] == pytest.approx(
        0.4, abs=1e-8
    )


def test_guided_pair_potential_tangent_and_swept_limit():
    model, _load, definition, upper_nodes = _guided_pair()
    manager = _PlanarShellPairManager(model, definition)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in upper_nodes:
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -3.0e-7
    force, tangent, summary = manager.evaluate(displacement)
    assert summary.minimum_gap == pytest.approx(-3.0e-7)
    assert summary.penalty_energy == pytest.approx(0.5 * 1.0e6 * (3.0e-7) ** 2)
    direction = np.zeros_like(displacement)
    for node_id in upper_nodes:
        direction[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 1.0
    h = 1.0e-8
    plus = manager.evaluate(displacement + h * direction, tangent=False)[0]
    minus = manager.evaluate(displacement - h * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2 * h), tangent @ direction,
                               rtol=1e-7, atol=1e-8)
    target_direction = np.zeros_like(displacement)
    for node_id in range(1, 5):
        target_direction[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 1.0
    plus = manager.evaluate(displacement + h * target_direction, tangent=False)[0]
    minus = manager.evaluate(displacement - h * target_direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2 * h),
                               tangent @ target_direction, rtol=1e-7, atol=1e-8)
    excessive = np.zeros_like(displacement)
    for node_id in upper_nodes:
        excessive[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -1.0e-3
    fraction = manager.admissible_step_fraction(np.zeros_like(displacement), excessive)
    assert fraction == pytest.approx(0.98e-3)
    assert manager.evaluate(fraction * excessive, tangent=False)[2].gap_tolerance_met
    assert np.linalg.norm(force) > 0.0


def test_guided_pair_displacement_control_recovers_target_reaction():
    model, load, definition, upper_nodes = _guided_pair(split_target=True)
    result = solve_static_nonlinear(
        model, load, contact=definition, kinematics="von_karman",
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=upper_nodes[0], dof="uz", target_displacement=-4.0e-7,
        ),
        num_steps=1, max_iterations=12,
    )
    assert result.status == "completed", result.failure_reason
    assert result.load_factor == pytest.approx(1.0, rel=1e-6)
    summary = result.info["general_contact"]
    assert summary["integration_force"][2] == pytest.approx(0.4, rel=1e-6)
    assert summary["target_force"][2] == pytest.approx(-0.4, rel=1e-6)
    assert result.steps[-1].support_reactions["lower"][2] == pytest.approx(0.4)


def test_partial_projected_overlap_transfers_equal_opposite_resultants():
    model, _load, definition, upper_nodes = _guided_pair(target_shift=0.5)
    manager = _PlanarShellPairManager(model, definition)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in upper_nodes:
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -5.0e-7
    _force, _tangent, summary = manager.evaluate(displacement)
    assert summary.reference_overlap_area == pytest.approx(0.5)
    assert summary.integration_force[2] == pytest.approx(0.25)
    assert summary.target_force[2] == pytest.approx(-0.25)
    np.testing.assert_allclose(
        np.asarray(summary.integration_moment) + np.asarray(summary.target_moment),
        0.0, atol=1e-10,
    )


def test_tilted_physical_skins_transfer_offset_force_couples():
    director = np.asarray((0.1, 0.0, 1.0), dtype=float)
    director /= np.linalg.norm(director)
    model, _load, definition, upper_nodes = _guided_pair(director=director)
    manager = _PlanarShellPairManager(model, definition)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in upper_nodes:
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -1.0e-4
    residual, _tangent, summary = manager.evaluate(displacement)
    upper_couple_y = -sum(
        residual[model.mesh.dof_manager.get_node_dofs(node_id)[4]]
        for node_id in upper_nodes
    )
    assert summary.minimum_gap == pytest.approx(-1.0e-4)
    assert summary.integration_force[2] == pytest.approx(100.0)
    assert upper_couple_y == pytest.approx(100.0 * 0.025 * director[0])
    np.testing.assert_allclose(
        np.asarray(summary.integration_moment) + np.asarray(summary.target_moment),
        0.0, atol=1e-10,
    )


def test_common_lateral_translation_preserves_pair_contact_work():
    model, _load, definition, upper_nodes = _guided_pair()
    manager = _PlanarShellPairManager(model, definition)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in upper_nodes:
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -5.0e-7
    reference_force, reference_tangent, reference = manager.evaluate(displacement)
    moved = displacement.copy()
    for node_id in manager.node_dofs:
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        moved[dofs[0]] += 0.2
        moved[dofs[1]] -= 0.1
    moved_force, moved_tangent, translated = manager.evaluate(moved)
    np.testing.assert_allclose(moved_force, reference_force, atol=1e-10)
    np.testing.assert_allclose((moved_tangent - reference_tangent).toarray(),
                               0.0, atol=1e-10)
    assert translated.minimum_gap == pytest.approx(reference.minimum_gap)
    assert translated.penalty_energy == pytest.approx(reference.penalty_energy)
    np.testing.assert_allclose(
        np.asarray(translated.integration_moment) + np.asarray(translated.target_moment),
        0.0, atol=1e-10,
    )


def _guided_mixed_pair(source_kind, target_kind, *, director=None,
                       source_shift=(0.0, 0.0)):
    model = FEModel("guided_mixed_shell_pair")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    shape_xy = {"Q4": ((0, 0), (1, 0), (1, 1), (0, 1)),
                "S3": ((0, 0), (1, 0), (0, 1))}
    target_reference_top = 0.025 if target_kind == "Q4" else 0.015
    source_reference_bottom = -0.025 if source_kind == "Q4" else -0.035
    source_offset = ((target_reference_top - source_reference_bottom)
                     * (np.asarray(director, dtype=float)
                        if director is not None else np.array((0.0, 0.0, 1.0))))
    target_nodes = tuple(range(1, len(shape_xy[target_kind]) + 1))
    source_nodes = tuple(range(len(target_nodes) + 1,
                               len(target_nodes) + len(shape_xy[source_kind]) + 1))
    for node_id, (x, y) in zip(target_nodes, shape_xy[target_kind]):
        model.add_node(node_id, x, y, 0.0)
    for node_id, (x, y) in zip(source_nodes, shape_xy[source_kind]):
        model.add_node(node_id, *(np.asarray((
            x + source_shift[0], y + source_shift[1], 0.0,
        )) + source_offset))
    for element_id, nodes, kind in ((1, target_nodes, target_kind),
                                    (2, source_nodes, source_kind)):
        options = ({"formulation": "e4-pl-s3-v2d",
                    "reference_normal": (0, 0, 1) if director is None else director,
                    "reference_surface_offset": 0.01}
                   if kind == "S3" else
                   ({"reference_normal": director} if director is not None else {}))
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05, **options,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", target_nodes,
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guided", source_nodes,
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press_source")
    for node_id in source_nodes:
        load.add_nodal_load(node_id, forces=(0.0, 0.0, -0.1))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", (2,), side="bottom"),
                  ContactSurface("target", (1,), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction("pair", "source", "target", 1.0e6, 1.0e-6),),
    )
    return model, load, definition, source_nodes


@pytest.mark.parametrize("source_kind,target_kind", (
    ("S3", "Q4"), ("Q4", "S3"), ("S3", "S3"),
))
def test_guided_mixed_shell_pair_has_physical_offsets_and_balanced_forces(
        source_kind, target_kind):
    model, _load, definition, source_nodes = _guided_mixed_pair(
        source_kind, target_kind
    )
    report = validate_general_contact(model, definition)
    assert report.valid, report.issues
    assert report.profile == "PLANAR_Q4_S3_PAIR_NORMAL_TRANSLATION_PENALTY_MORTAR_V0"
    manager = _PlanarShellPairManager(model, definition)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in source_nodes:
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -1.0e-6
    _force, tangent, summary = manager.evaluate(displacement)
    assert summary.minimum_gap == pytest.approx(-1.0e-6)
    assert summary.reference_overlap_area == pytest.approx(
        0.5 if "S3" in (source_kind, target_kind) else 1.0
    )
    assert summary.integration_force[2] == pytest.approx(0.5)
    assert summary.target_force[2] == pytest.approx(-0.5)
    np.testing.assert_allclose(
        np.asarray(summary.integration_moment) + np.asarray(summary.target_moment),
        0.0, atol=1e-10,
    )
    direction = np.zeros_like(displacement)
    for node_id in source_nodes:
        direction[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = 1.0
    h = 1.0e-8
    plus = manager.evaluate(displacement + h * direction, tangent=False)[0]
    minus = manager.evaluate(displacement - h * direction, tangent=False)[0]
    np.testing.assert_allclose((plus - minus) / (2.0 * h), tangent @ direction,
                               rtol=1.0e-7, atol=1.0e-8)


def test_guided_s3_source_solves_through_static_route():
    model, load, definition, _source_nodes = _guided_mixed_pair("S3", "Q4")
    result = solve_static_nonlinear(
        model, load, contact=definition, kinematics="von_karman",
        num_steps=1, max_iterations=12,
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["integration_force"][2] == pytest.approx(0.3, abs=1e-8)
    assert summary["target_force"][2] == pytest.approx(-0.3, abs=1e-8)
    assert summary["minimum_gap"] == pytest.approx(-6.0e-7, abs=1e-9)


def test_restricted_sliding_s3_source_solves_with_physical_offset():
    model, load, original, source_nodes = _guided_mixed_pair(
        "S3", "Q4", source_shift=(0.15, 0.1),
    )
    contact = ContactDefinition(
        surfaces=original.surfaces, rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "source", "target", 1.0e6, 1.0e-5,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    report = validate_general_contact(model, contact, settings)
    assert report.valid, report.issues
    result = solve_static_nonlinear(
        model, load, contact=contact, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("general_contact"),
    )
    summary = result.info["general_contact"]
    assert 0.0 < summary["current_overlap_area"] < 0.5
    assert summary["gap_tolerance_met"]
    expected_force = 0.1 * len(source_nodes)
    assert summary["integration_force"][2] == pytest.approx(expected_force, rel=1e-5)
    assert summary["target_force"][2] == pytest.approx(-expected_force, rel=1e-5)


def test_restricted_swept_bound_uses_fixed_overlap_endpoint_certificate():
    from anyfesolver.general_contact.moving import bound_swept_moving_mortar_gap
    from anyfesolver.general_contact.sliding import _RestrictedSlidingPairFEManager

    model, _load, original, source_nodes = _guided_mixed_pair(
        "Q4", "S3", source_shift=(0.15, 0.1),
    )
    contact = ContactDefinition(
        surfaces=original.surfaces, rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "source", "target", 1.0e6, 1.0e-5,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    manager = _RestrictedSlidingPairFEManager(model, contact, settings)
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = start.copy()
    for node, height in zip(source_nodes, (-1.0e-6, -1.0e-6,
                                           -3.0e-5, -1.0e-6)):
        end[model.mesh.dof_manager.get_node_dofs(node)[2]] = height
    split = manager.height_offsets[1]
    first = manager._reduced_pose(start)
    last = manager._reduced_pose(end)
    coarse = bound_swept_moving_mortar_gap(
        *manager.faces, manager.normal,
        first[:3], last[:3], first[3:6], last[3:6],
        first[6:split], last[6:split], first[split:], last[split:],
        gap_tolerance=manager.gap_tolerance,
    )
    assert not coarse.tolerance_certified
    refined = manager.swept_trial_bound(start, end)
    assert refined.tolerance_certified
    assert refined.conservative_lower_gap >= -manager.gap_tolerance


def test_s3_reference_surface_offset_contributes_contact_couple():
    director = np.asarray((0.1, 0.0, 1.0), dtype=float)
    director /= np.linalg.norm(director)
    model, _load, definition, source_nodes = _guided_mixed_pair(
        "S3", "Q4", director=director
    )
    manager = _PlanarShellPairManager(model, definition)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in source_nodes:
        displacement[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -1.0e-6
    residual, _tangent, summary = manager.evaluate(displacement)
    source_couple_y = -sum(
        residual[model.mesh.dof_manager.get_node_dofs(node_id)[4]]
        for node_id in source_nodes
    )
    assert summary.integration_force[2] == pytest.approx(0.5)
    assert source_couple_y == pytest.approx(0.5 * 0.035 * director[0])


def test_guided_pair_rejects_nonuniform_or_relative_lateral_motion():
    model, load, definition, upper_nodes = _guided_pair()
    manager = _PlanarShellPairManager(model, definition)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    displacement[model.mesh.dof_manager.get_node_dofs(upper_nodes[0])[2]] = -1.0e-5
    with pytest.raises(ValueError, match="uniform normal translation"):
        manager.evaluate(displacement)
    displacement[:] = 0.0
    displacement[model.mesh.dof_manager.get_node_dofs(upper_nodes[0])[0]] = 1.0e-5
    with pytest.raises(ValueError, match="uniform normal translation"):
        manager.evaluate(displacement)
    displacement[:] = 0.0
    displacement[model.mesh.dof_manager.get_node_dofs(upper_nodes[0])[3]] = 1.0e-5
    with pytest.raises(ValueError, match="fixed shell rotations"):
        manager.evaluate(displacement)
    outward_definition = ContactDefinition(
        surfaces=(ContactSurface("source", (2,), side="bottom"),
                  ContactSurface("target", (1,), side="bottom")),
        rigid_bodies=(),
        interactions=(ContactInteraction("pair", "source", "target", 1.0e6, 1.0e-6),),
    )
    outward_report = validate_general_contact(model, outward_definition)
    assert not outward_report.valid
    assert any("faces away" in issue for issue in outward_report.issues)
    split_model, _split_load, split_definition, _split_upper = _guided_pair(
        split_target=True
    )
    with pytest.raises(MemoryError, match="candidate limit"):
        _PlanarShellPairManager(
            split_model, split_definition,
            ContactSolverSettings(maximum_candidate_pairs=1),
        )


@pytest.mark.parametrize("source_kind", ("Q4", "S3"))
def test_guided_pair_restart_matches_continuation_and_binds_contact_law(source_kind):
    if source_kind == "Q4":
        model, load, definition, _upper_nodes = _guided_pair(split_target=True)
    else:
        model, load, definition, _upper_nodes = _guided_mixed_pair("S3", "Q4")
    common = dict(contact=definition, kinematics="von_karman",
                  num_steps=1, max_iterations=12)
    half = solve_static_nonlinear(
        model, load, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    assert half.restart_checkpoint["version"] == 2
    assert half.restart_checkpoint["contact_state"] == {}
    resumed = solve_static_nonlinear(
        model, load, max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    full = solve_static_nonlinear(
        model, load, contact=definition, kinematics="von_karman",
        num_steps=2, max_iterations=12,
    )
    assert resumed.status == full.status == "completed"
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-8, atol=1e-10)
    assert resumed.info["general_contact"]["integration_force"] == pytest.approx(
        full.info["general_contact"]["integration_force"], abs=1e-8
    )
    interaction = definition.interactions[0]
    changed = ContactDefinition(
        surfaces=definition.surfaces, rigid_bodies=(),
        interactions=(ContactInteraction(
            interaction.name, interaction.surface, interaction.target,
            2.0e6, interaction.gap_tolerance,
        ),),
    )
    with pytest.raises(ValueError, match="analysis contract is incompatible"):
        solve_static_nonlinear(
            model, load, contact=changed, kinematics="von_karman",
            num_steps=1, max_iterations=12, max_load_factor=1.0,
            restart_checkpoint=half.to_restart_checkpoint(),
        )
