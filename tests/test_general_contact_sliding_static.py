"""End-to-end development checks for explicitly selected restricted sliding."""

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


def test_restricted_sliding_pair_runs_through_nonlinear_static():
    model = FEModel("restricted_sliding_static")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for node, (x, y) in enumerate(corners, 1):
        model.add_node(node, x, y, 0.0)
    for node, (x, y) in enumerate(corners, 5):
        model.add_node(node, x + 0.2, y + 0.1, 0.05)
    for element, nodes in ((1, (1, 2, 3, 4)), (2, (5, 6, 7, 8))):
        model.add_element(element, create_shell_element(
            element, nodes, "elastic", thickness=0.05,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_guided", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press")
    for node in (5, 6, 7, 8):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.1))
    contact = ContactDefinition(
        surfaces=(ContactSurface("upper", (2,), side="bottom"),
                  ContactSurface("lower", (1,), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "upper", "lower", 1.0e6, 6.0e-6,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    report = validate_general_contact(model, contact, settings)
    assert report.valid
    assert "RESTRICTED_SLIDING" in report.profile
    result = solve_static_nonlinear(
        model, load, contact=contact, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
    )
    assert result.status == "completed", (
        result.failure_reason, result.load_factor,
        result.info.get("general_contact"), len(result.steps),
    )
    summary = result.info["general_contact"]
    assert summary["gap_tolerance_met"]
    assert summary["current_overlap_area"] == pytest.approx(0.72)
    np.testing.assert_allclose(
        np.asarray(summary["integration_force"])
        + np.asarray(summary["target_force"]),
        0.0, atol=1.0e-6,
    )
    assert summary["integration_force"][2] == pytest.approx(0.4, rel=1e-5)
    half = solve_static_nonlinear(
        model, load, contact=contact, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    assert half.restart_checkpoint["version"] == 2
    resumed = solve_static_nonlinear(
        model, load, contact=contact, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
        max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(
        resumed.displacements, result.displacements, rtol=1.0e-7, atol=1.0e-9,
    )
    assert resumed.info["general_contact"]["integration_force"] == pytest.approx(
        summary["integration_force"], rel=1.0e-7, abs=1.0e-7,
    )
    with pytest.raises(ValueError, match="analysis contract is incompatible"):
        solve_static_nonlinear(
            model, load, contact=contact,
            contact_settings=ContactSolverSettings(
                pair_motion="restricted_sliding", sliding_difference_step=2.0e-8,
            ),
            kinematics="von_karman", num_steps=1, max_iterations=12,
            max_load_factor=1.0,
            restart_checkpoint=half.to_restart_checkpoint(),
        )


def test_overlap_aware_swept_gap_excludes_distant_deep_source_nodes():
    """Independent linear height field is safe throughout nonzero sliding."""
    from anyfesolver.general_contact.overlap import PlanarFace
    from anyfesolver.general_contact.moving import (
        bound_swept_moving_mortar_gap, certify_swept_moving_mortar_gap,
    )
    from anyfesolver.general_contact.sliding import _RestrictedSlidingPairFEManager

    source = PlanarFace((1, 1), np.array(
        ((0., 0., 0.), (2., 0., 0.), (2., 1., 0.), (0., 1., 0.))))
    target = PlanarFace((2, 1), np.array(
        ((0., 0., 0.), (.8, 0., 0.), (.8, 1., 0.), (0., 1., 0.))))
    heights = np.array((0., -1e-5, -1e-5, 0.))
    zero = np.zeros(3)
    arguments = (source, target, np.array((0., 0., 1.)), zero,
                 np.array((.1, 0., 0.)), zero, zero,
                 heights, heights, np.zeros(4), np.zeros(4))
    coarse = bound_swept_moving_mortar_gap(*arguments, gap_tolerance=6e-6)
    assert not coarse.tolerance_certified
    certificate = certify_swept_moving_mortar_gap(
        *arguments, gap_tolerance=6e-6, maximum_boxes=4096)
    # h=-5e-6*x_material; overlap x_material lies in [0,.8] at all times.
    assert certificate.tolerance_certified
    assert -6e-6 <= certificate.conservative_lower_gap <= -4e-6

    model = FEModel("overlap_aware_nonuniform_height")
    model.add_material("elastic", 1e5, .3, density=1.)
    for index, (face, z) in enumerate(((target, 0.), (source, .05))):
        for local, point in enumerate(face.corners):
            model.add_node(1+4*index+local, point[0], point[1], z)
    for eid, nodes in ((1, (1, 2, 3, 4)), (2, (5, 6, 7, 8))):
        model.add_element(eid, create_shell_element(
            eid, nodes, "elastic", thickness=.05))
    definition = ContactDefinition(
        (ContactSurface("source", (2,), "bottom"),
         ContactSurface("target", (1,), "top")), (),
        (ContactInteraction("pair", "source", "target", 1e6, 6e-6),))
    manager = _RestrictedSlidingPairFEManager(model, definition,
        ContactSolverSettings(pair_motion="restricted_sliding",
                              sliding_difference_step=1e-8))
    start = np.zeros(model.mesh.dof_manager.total_dofs)
    end = start.copy()
    for node, height in zip((5, 6, 7, 8), heights):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        start[dofs[2]] = end[dofs[2]] = height
        end[dofs[0]] = .1
    assert manager.swept_trial_bound(start, end).tolerance_certified
    assert manager.admissible_step_fraction(start, end) == 1.


def test_overlap_aware_swept_gap_refuses_hidden_temporal_penetration():
    from anyfesolver.general_contact.overlap import PlanarFace
    from anyfesolver.general_contact.moving import (
        certify_moving_mortar_gap, certify_swept_moving_mortar_gap,
    )
    source = PlanarFace((1, 1), np.array(
        ((-1., 0., 0.), (2., 0., 0.), (2., 1., 0.), (-1., 1., 0.))))
    target = PlanarFace((2, 1), np.array(
        ((.49, 0., 0.), (.51, 0., 0.), (.51, 1., 0.), (.49, 1., 0.))))
    zero = np.zeros(3)
    direction = np.array((0., 0., 1.))
    last = np.array((.5, 0., 0.))
    heights = np.array((1e-4, -2e-4, -2e-4, 1e-4))
    for translation, field in ((zero, np.zeros(4)), (last, heights)):
        assert certify_moving_mortar_gap(source, target, direction,
            translation, zero, field, np.zeros(4), gap_tolerance=6e-6,
            maximum_triangles=4096).tolerance_met
    # At x=.5, gap(t)=5e-5*t*(t-1), with -12.5um at t=.5.
    certificate = certify_swept_moving_mortar_gap(source, target, direction,
        zero, last, zero, zero, np.zeros(4), heights,
        np.zeros(4), np.zeros(4), gap_tolerance=6e-6, maximum_boxes=256)
    assert not certificate.tolerance_certified


def test_overlap_aware_swept_gap_keeps_nonaffine_q4_and_s3_interpolation():
    from anyfesolver.general_contact.overlap import PlanarFace
    from anyfesolver.general_contact.moving import certify_swept_moving_mortar_gap
    source = PlanarFace((1, 1), np.array(
        ((0., 0., 0.), (2., 0., 0.), (2.5, 1., 0.), (0., 1., 0.))))
    target = PlanarFace((2, 1), np.array(
        ((0., 0., 0.), (.8, 0., 0.), (0., 1., 0.))))
    zero = np.zeros(3)
    heights = np.array((0., -1e-5, -1e-5, 0.))
    certificate = certify_swept_moving_mortar_gap(source, target,
        np.array((0., 0., 1.)), zero, np.array((.1, 0., 0.)), zero, zero,
        heights, heights, np.zeros(3), np.zeros(3),
        gap_tolerance=6e-6, maximum_boxes=4096)
    # The original map is x=(2+.5*v)*u, y=v. On the entire overlap,
    # u<=.8/2=.4, hence h=-1e-5*u>=-4e-6 even while translating.
    assert certificate.tolerance_certified
    assert -6e-6 <= certificate.conservative_lower_gap <= -4e-6


def test_overlap_aware_swept_gap_budget_exhaustion_is_non_success():
    from anyfesolver.general_contact.overlap import PlanarFace
    from anyfesolver.general_contact.moving import certify_swept_moving_mortar_gap
    face = PlanarFace((1, 1), np.array(
        ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))))
    zero = np.zeros(3)
    heights = np.array((0., -1e-5, -1e-5, 0.))
    arguments = (face, face, np.array((0., 0., 1.)), zero,
                 np.array((.01, 0., 0.)), zero, zero,
                 heights, heights, heights, heights)
    # The actual gap is +1e-7*t, but one dependency-wide interval cannot
    # establish it. An exhausted certificate never becomes a physical pass.
    assert not certify_swept_moving_mortar_gap(
        *arguments, gap_tolerance=6e-6, maximum_boxes=1).tolerance_certified
    assert certify_swept_moving_mortar_gap(
        *arguments, gap_tolerance=6e-6, maximum_boxes=4096).tolerance_certified


def test_swept_gap_fast_path_refuses_catastrophic_normal_cancellation():
    from anyfesolver.general_contact.overlap import PlanarFace
    from anyfesolver.general_contact.moving import (
        bound_swept_moving_mortar_gap, certify_swept_moving_mortar_gap,
    )
    xy = np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    source = PlanarFace((1, 1), np.column_stack((xy, -np.ones(4))))
    target = PlanarFace((2, 1), np.column_stack((xy, np.zeros(4))))
    zero = np.zeros(3)
    huge = np.full(4, 1e16)
    arguments = (source, target, np.array((0., 0., 1.)), zero,
                 np.array((.01, 0., 0.)), zero, zero, huge, huge, huge, huge)
    # Exact gap=-1: (-1+1e16)-1e16 must not round to a certified zero.
    assert not bound_swept_moving_mortar_gap(
        *arguments, gap_tolerance=6e-6).tolerance_certified
    assert not certify_swept_moving_mortar_gap(
        *arguments, gap_tolerance=6e-6, maximum_boxes=64).tolerance_certified


def test_swept_gap_fast_path_refuses_tiny_closing_overlap():
    from anyfesolver.general_contact.overlap import PlanarFace
    from anyfesolver.general_contact.moving import (
        bound_swept_moving_mortar_gap, certify_swept_moving_mortar_gap,
    )
    right = 1e-4
    left = np.nextafter(right, np.inf)
    source = PlanarFace((1, 1), np.array(
        ((0., 0., 0.), (right, 0., 0.), (right, 1., 0.), (0., 1., 0.))))
    target = PlanarFace((2, 1), np.array(
        ((left, 0., 0.), (1.0001, 0., 0.), (1.0001, 1., 0.), (left, 1., 0.))))
    zero = np.zeros(3)
    arguments = (source, target, np.array((0., 0., 1.)), zero,
                 np.array((1e-16, 0., 0.)), zero, zero,
                 np.full(4, -1e-3), np.full(4, -1e-3), np.zeros(4), np.zeros(4))
    assert right + 1e-16 > left  # Positive projected overlap exists at t=1.
    assert not bound_swept_moving_mortar_gap(
        *arguments, gap_tolerance=6e-6).tolerance_certified
    assert not certify_swept_moving_mortar_gap(
        *arguments, gap_tolerance=6e-6, maximum_boxes=64).tolerance_certified


def test_restricted_sliding_pair_follows_prescribed_lateral_path():
    model = FEModel("restricted_sliding_displacement_path")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for node, (x, y) in enumerate(corners, 1):
        model.add_node(node, x, y, 0.0)
    for node, (x, y) in enumerate(corners, 5):
        model.add_node(node, x + 0.2, y + 0.1, 0.05)
    for element, nodes in ((1, (1, 2, 3, 4)), (2, (5, 6, 7, 8))):
        model.add_element(element, create_shell_element(
            element, nodes, "elastic", thickness=0.05,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_lateral_guide", (5, 6, 7, 8),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    master_x = int(model.mesh.dof_manager.get_node_dofs(5)[0])
    for node in (6, 7, 8):
        slave_x = int(model.mesh.dof_manager.get_node_dofs(node)[0])
        model.add_constraint_equation(
            terms=((slave_x, 1.0), (master_x, -1.0)),
            source_id=f"upper_uniform_x_{node}",
        )
    press = LoadCase("press")
    for node in (5, 6, 7, 8):
        press.add_nodal_load(node, forces=(0.0, 0.0, -0.1))
    drive = LoadCase("lateral_drive")
    drive.add_nodal_load(5, forces=(1.0, 0.0, 0.0))
    contact = ContactDefinition(
        surfaces=(ContactSurface("upper", (2,), side="bottom"),
                  ContactSurface("lower", (1,), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "upper", "lower", 1.0e6, 6.0e-6,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    result = solve_static_nonlinear(
        model, drive, constant_load_case=press,
        contact=contact, contact_settings=settings, kinematics="von_karman",
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=5, dof="ux", target_displacement=0.1,
        ),
        num_steps=2, max_iterations=12,
    )
    assert result.status == "completed", (
        result.failure_reason, result.load_factor, result.info.get("general_contact"),
    )
    summary = result.info["general_contact"]
    assert summary["current_overlap_area"] == pytest.approx(0.63, rel=1e-8)
    assert summary["gap_tolerance_met"]
    assert summary["integration_force"][2] == pytest.approx(0.4, rel=1e-5)
