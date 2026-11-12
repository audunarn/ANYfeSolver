"""Local gates for Q4/S3 float/Jet2 geometry comparison."""

import numpy as np
import pytest
from dataclasses import replace
import json
from decimal import Decimal
from pathlib import Path

from scripts.general_contact_d1_coupled_refinement import build_case
from anyfesolver import create_shell_element
from anyfesolver.general_contact.core import ContactSolverSettings
from anyfesolver.general_contact.overlap import PlanarFace, planar_projected_overlap
from anyfesolver.general_contact.q4_source_overlap import projected_q4_source_face
from anyfesolver.general_contact.q4_s3_overlap_derivatives import UnsupportedQ4S3Branch
from anyfesolver.general_contact.s3_patch_augmented import (
    MovingS3PatchProjectedNormal,
    _q4_s3_pair_value_rtol, _q4_s3_pair_values_agree,
    _q4_s3_gap_operand_roundoff,
    _touching_shared_target_edge,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def test_tiny_wedge_area_disagreement_has_small_bounded_value_allowance():
    local = 4.45786736357e-16
    derivative = 4.45786737393e-16
    tolerance = _q4_s3_pair_value_rtol(local, derivative, 6.35274248175e-20)
    assert 1.0e-9 < tolerance < 2.0e-8


def test_moving_physical_edge_wedge_value_agreement_stays_bounded():
    # Captured at the 50 mm slide of an eight-Q4/two-S3 force path. Both
    # clippers retained this physical-edge wedge; their positive areas agree
    # far inside the polygon-coordinate resolution. Q1 support varies across
    # the wedge and has a slightly larger relative float/Jet2 difference.
    local_area = 5.32271210325e-10
    derivative_area = 5.32271200712e-10
    area_resolution = 2.65519977967e-12
    support_local = 2.9259797572551105e-10
    support_derivative = 2.925979576395046e-10
    allowance = _q4_s3_pair_value_rtol(
        local_area, derivative_area, area_resolution,
    )
    assert abs(local_area-derivative_area) < area_resolution
    assert allowance == 1.0e-6
    assert abs(support_local-support_derivative) <= allowance*support_local
    second_area = 6.73575833514e-10
    second_derivative_area = 6.73575838977e-10
    second_support = 3.0309933156822006e-10
    second_derivative_support = 3.0309930687616105e-10
    second_allowance = _q4_s3_pair_value_rtol(
        second_area, second_derivative_area, 2.65519978062e-12,
    )
    assert second_allowance == 1.0e-6
    assert (abs(second_support-second_derivative_support)
            <= second_allowance*second_support)


def test_captured_tiny_q4_s3_overlap_agrees_with_decimal_area_oracle():
    # Physical coordinates from a rejected left-edge D1 press candidate.
    # Independent 80-digit Decimal projection and clipping gives the area
    # below. The former global-coordinate shoelace lost 1.31 ppm here.
    source = np.array((
        (0.09999669777838385, 0.6, 0.014984999744974806),
        (0.3499982279944439, 0.6, 0.014959007393097431),
        (0.34999954835060437, 0.8500029552441929, 0.014974181631346554),
        (0.09999844804776516, 0.8500000882463871, 0.014984999920000677),
    ))
    target = np.array((
        (1.2999985000000076, -0.1, 0.015064999925000003),
        (1.2999985000000076, 1.2, 0.015064999925000003),
        (-1.4999999925000002e-6, 1.2, 0.014934999925000001),
    ))
    source_face, normal = projected_q4_source_face(source)
    projected_target = (target-((target-source_face.corners[0]) @ normal)
                        [:, None]*normal)
    overlap = planar_projected_overlap(
        source_face, PlanarFace((2, 1), projected_target),
    )
    assert overlap is not None
    decimal_area = 8.0125108710039545e-12
    assert abs(overlap.area-decimal_area)/decimal_area < 1.0e-8


def test_subresolved_wedge_compares_full_q1_vector_at_shared_scale():
    # Captured from pair 12 during the eight-Q4 slide. One almost-zero Q1
    # component has large relative error while the whole support vector
    # differs by less than 1e-6 of its leading component.
    local_area = 3.19672221464e-19
    derivative_area = 3.19672249205e-19
    resolution = 1.93440972087e-21
    local = np.array([
        3.1967222082844646e-19, 6.824979926081431e-28,
        1.3649028736265628e-37, 8.520318350401215e-29,
    ])
    derivative = np.array([
        3.19672248366244e-19, 6.816181601628408e-28,
        1.3625379198364865e-37, 8.520226095437441e-29,
    ])
    allowance = _q4_s3_pair_value_rtol(
        local_area, derivative_area, resolution,
    )
    roundoff = 1.45370108096e-31
    assert allowance == 1.0e-6
    assert not np.allclose(local, derivative, rtol=allowance, atol=roundoff)
    options = dict(value_rtol=allowance, roundoff_atol=roundoff,
                   local_area=local_area, derivative_area=derivative_area,
                   area_resolution=resolution)
    assert _q4_s3_pair_values_agree(local, derivative, **options)
    gross = derivative.copy()
    gross[1] += 1.0e-22
    assert not _q4_s3_pair_values_agree(local, gross, **options)


def test_near_resolution_area_cannot_open_unbounded_value_tolerance():
    with pytest.raises(UnsupportedQ4S3Branch, match="area agreement is unresolved"):
        _q4_s3_pair_value_rtol(1.0e-16, 1.1e-16, 2.0e-17)
    with pytest.raises(UnsupportedQ4S3Branch, match="area agreement is unresolved"):
        _q4_s3_pair_value_rtol(1.0e-4, 1.00000001e-4, 1.0e-14)


def test_subresolved_area_agreement_uses_its_own_relative_cap():
    local = 4.29448513597e-19
    derivative = local + 1.15753273864e-25
    resolution = 2.24164536308e-21
    assert _q4_s3_pair_value_rtol(local, derivative, resolution) == 1.0e-6
    with pytest.raises(UnsupportedQ4S3Branch,
                       match="area agreement is unresolved"):
        _q4_s3_pair_value_rtol(local, local*(1.0+2.0e-6), resolution)


def test_resolved_q4_weighted_gap_cancellation_uses_vector_value_bound():
    # Captured from a rejected sixteen-Q4 recontact Schur trial at pair 27.
    # The positive overlap and resolved support are substantial, but two
    # weighted-gap components nearly cancel and cannot carry a useful
    # componentwise relative comparison.
    local = np.array([
        6.461447169920361e-9, -9.054382905387276e-17,
        -7.707115119329735e-17, 5.295586592146162e-9,
    ])
    derivative = np.array([
        6.461447169913053e-9, -9.053425443828246e-17,
        -7.707468951006633e-17, 5.295586592148127e-9,
    ])
    options = dict(value_rtol=1.00000328955e-9,
                   roundoff_atol=5.87665196683e-21,
                   local_area=0.0337500006879,
                   derivative_area=0.0337500006879,
                   area_resolution=2.94233915524e-12)
    assert not _q4_s3_pair_values_agree(local, derivative, **options)
    assert _q4_s3_pair_values_agree(local, derivative,
                                    normwise=True, **options)
    gross = derivative.copy()
    gross[1] += 1.0e-15
    assert not _q4_s3_pair_values_agree(local, gross,
                                        normwise=True, **options)


def test_resolved_pair_gap_operand_roundoff_is_bounded_and_specific():
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "general_contact_d1_distributed_pair28.json"
                          ).read_text(encoding="utf-8"))
    source = np.asarray(fixture["local_source"])
    target = np.asarray(fixture["local_target"])
    support_scale = 0.00390423719797
    ordinary = fixture["comparison"]["roundoff_atol"]
    operand = _q4_s3_gap_operand_roundoff(source, target, support_scale)
    local = np.asarray(fixture["local_values_m3"])
    derivative = np.asarray(fixture["derivative_values_m3"])
    comparison = dict(fixture["comparison"])
    assert operand > ordinary
    assert operand/support_scale < 1.0e-9*min(
        np.linalg.norm(np.roll(source, -1, axis=0)-source, axis=1)
    )
    assert not _q4_s3_pair_values_agree(local, derivative, **comparison)
    comparison["roundoff_atol"] = max(ordinary, operand)
    assert _q4_s3_pair_values_agree(local, derivative, **comparison)
    gross = derivative.copy()
    gross[1] += 1.0e-15
    assert not _q4_s3_pair_values_agree(local, gross, **comparison)
    oracle = json.loads((Path(__file__).parent / "fixtures" /
                         "general_contact_d1_distributed_pair28_decimal.json"
                         ).read_text(encoding="utf-8"))
    rows = oracle["rows"]
    assert all(row["clip_signature"] == rows[0]["clip_signature"]
               for row in rows)
    for index in range(4):
        reference = Decimal(rows[-1]["weighted_gap_m3"][index])
        assert max(abs(Decimal(row["weighted_gap_m3"][index])-reference)
                   for row in rows[:-1]) < Decimal("1e-25")
        assert (abs(Decimal.from_float(float(local[index]))-reference)
                < Decimal.from_float(operand))
        assert (abs(Decimal.from_float(float(derivative[index]))-reference)
                < Decimal.from_float(operand))
    assert abs(Decimal(rows[-1]["projected_area_m2"])
               - Decimal.from_float(comparison["local_area"])) < Decimal("1e-14")
    warped = source.copy()
    warped[2, 2] += 0.005
    assert _q4_s3_gap_operand_roundoff(warped, target, support_scale) > 0.0
    with pytest.raises(UnsupportedQ4S3Branch,
                       match="coordinate cancellation exceeds"):
        _q4_s3_gap_operand_roundoff(
            warped+[0.0, 0.0, 1.0e6],
            target+[0.0, 0.0, 1.0e6], support_scale,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(UnsupportedQ4S3Branch,
                           match="frame is degenerate|not finite"):
            _q4_s3_gap_operand_roundoff(
                source*1.0e308, target*1.0e308, support_scale,
            )


def test_shared_edge_tie_settings_require_explicit_bounded_jet2_profile():
    with pytest.raises(ValueError, match="needs step, direction and relative tolerance"):
        ContactSolverSettings(q4_s3_shared_edge_tie_step_m=3.0e-8)
    with pytest.raises(ValueError, match="needs Jet2 Q4/S3"):
        ContactSolverSettings(
            q4_s3_shared_edge_tie_step_m=3.0e-8,
            q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
            q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
        )
    with pytest.raises(ValueError, match="unit 3-vector"):
        ContactSolverSettings(
            general_s3_differentiation="jet2_q4_s3_dev",
            q4_s3_shared_edge_tie_step_m=3.0e-8,
            q4_s3_shared_edge_tie_direction=(2.0, 0.0, 0.0),
            q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
        )


def test_inactive_zero_area_branch_requires_touch_on_shared_material_edge():
    source = PlanarFace((1, 1), np.array((
        (0.0, 0.25, 0.0), (1.0, 0.25, 0.0),
        (1.0, 0.75, 0.0), (0.0, 0.75, 0.0),
    )))
    ids = np.array((5, 6, 7))
    shared = {5, 6}
    shared_edge_touch = np.array(((1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                                  (1.5, 0.5, 0.0)))
    exterior_vertex_touch = np.array(((1.5, 0.0, 0.0), (1.5, 1.0, 0.0),
                                      (1.0, 0.5, 0.0)))
    shared_endpoint_touch = np.array(((1.0, 0.5, 0.0), (1.5, 1.0, 0.0),
                                      (1.5, 0.0, 0.0)))
    assert _touching_shared_target_edge(source, shared_edge_touch, ids, shared)
    assert not _touching_shared_target_edge(source, exterior_vertex_touch,
                                            ids, shared)
    assert not _touching_shared_target_edge(source, shared_endpoint_touch,
                                            ids, shared)


def test_patch_owner_fails_at_exact_positive_tie_and_balances_both_smooth_sides():
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    assert "q4_s3_shared_edge_tie_step_m" not in owner.restart_contract()["settings"]
    geometry = owner.law.geometry
    source = geometry.reference_source.copy()
    target = geometry.reference_target
    source[:, 0] += 0.05
    source[:, 2] -= 0.01504
    multipliers = np.zeros(len(geometry.source_node_ids))
    with pytest.raises(UnsupportedQ4S3Branch, match="clipping boundary"):
        owner.law.work(source, target, multipliers, tangent=True)

    for slide_offset in (-3.0e-8, 3.0e-8):
        smooth = source.copy()
        smooth[:, 0] += slide_offset
        work = owner.law.work(smooth, target, multipliers, tangent=True)
        assert work.state.active_pressure_dofs > 0
        force = work.residual.reshape(-1, 3)
        points = np.vstack((smooth, target))
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-8)
        np.testing.assert_allclose(np.cross(points, force).sum(axis=0),
                                   0.0, atol=1.0e-8)
        direction = np.zeros_like(points)
        direction[:len(smooth), 2] = 1.0
        direction = direction.ravel()
        step = 1.0e-6
        plus = smooth.copy()
        minus = smooth.copy()
        plus[:, 2] += step
        minus[:, 2] -= step
        finite_difference = (
            owner.law.work(plus, target, multipliers, tangent=False).residual
            - owner.law.work(minus, target, multipliers, tangent=False).residual
        )/(2.0*step)
        tangent_action = work.tangent @ direction
        error = np.linalg.norm(finite_difference-tangent_action)
        assert error/max(1.0, np.linalg.norm(tangent_action)) < 1.0e-5


def test_opt_in_exact_shared_edge_limit_preserves_exact_state_and_wrench():
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    settings = replace(
        settings, q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    assert owner.restart_contract()["settings"]["q4_s3_shared_edge_tie_direction"] == [1.0, 0.0, 0.0]
    geometry = owner.law.geometry
    source = geometry.reference_source.copy()
    source[:, 0] += 0.05
    source[:, 2] -= 0.01504
    target = geometry.reference_target
    multipliers = np.zeros(len(geometry.source_node_ids))
    exact = owner.law.evaluate(source, target, multipliers)
    work = owner.law.work(source, target, multipliers, tangent=True)
    assert work.state is not exact
    np.testing.assert_array_equal(work.state.reference_support,
                                  exact.reference_support)
    np.testing.assert_array_equal(work.state.pressure, exact.pressure)
    assert work.tie_resolution_relative_error is not None
    assert work.tie_resolution_relative_error < 1.0e-6
    assert work.energy_evaluations <= settings.maximum_general_s3_energy_evaluations
    force = work.residual.reshape(-1, 3)
    points = np.vstack((source, target))
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-8)
    np.testing.assert_allclose(np.cross(points, force).sum(axis=0),
                               0.0, atol=1.0e-8)
    for sign in (-1, 1):
        neighbor = source.copy()
        neighbor[:, 0] += sign*1.5e-8
        smooth = owner.law.work(neighbor, target, multipliers, tangent=True)
        assert np.linalg.norm(work.residual-smooth.residual)/np.linalg.norm(work.residual) < 1.0e-6
        assert np.linalg.norm(work.tangent-smooth.tangent)/np.linalg.norm(work.tangent) < 1.0e-6
    direction = np.zeros_like(points)
    direction[:len(source), 2] = 1.0
    plus = source.copy()
    minus = source.copy()
    plus[:, 2] += 1.0e-6
    minus[:, 2] -= 1.0e-6
    finite_difference = (
        owner.law.work(plus, target, multipliers, tangent=False).residual
        - owner.law.work(minus, target, multipliers, tangent=False).residual
    )/(2.0e-6)
    tangent_action = work.tangent @ direction.ravel()
    assert np.linalg.norm(finite_difference-tangent_action)/max(
        1.0, np.linalg.norm(tangent_action)) < 1.0e-5

    normal_axes = np.arange(2, work.residual.size, 3)
    owner.law.set_tangent_active_coordinates(normal_axes)
    reduced = owner.law.work(source, target, multipliers, tangent=True)
    np.testing.assert_allclose(reduced.residual, work.residual, rtol=1.0e-10,
                               atol=1.0e-12)
    np.testing.assert_array_equal(
        reduced.tangent[:, np.setdiff1d(np.arange(work.residual.size), normal_axes)],
        0.0,
    )
    owner.law.set_tangent_active_coordinates(np.arange(work.residual.size))

    owner.law.shared_edge_tie = (3.0e-8, np.array((1.0, 0.0, 0.0)), 1.0e-12)
    with pytest.raises(UnsupportedQ4S3Branch, match="shared-edge tie"):
        owner.law.work(source, target, multipliers, tangent=True)
    owner.law.shared_edge_tie = (3.0e-8, np.array((1.0, 0.0, 0.0)), 1.0e-6)
    warped = target.copy()
    warped[2, 2] += 1.0e-4
    assert not owner.law._only_artificial_shared_edge_ties(source, warped)


def test_three_face_smooth_patch_owns_existing_internal_tie():
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    # Extend the target along its left rim. The added face is adjacent to
    # the original internal edge's owner, but does not touch the source.
    new_node = max(model.mesh.nodes) + 1
    new_face = max(model.mesh.elements) + 1
    model.add_node(new_node, -1.5, 0.5, 0.0)
    old_target = definition.surfaces[1].element_ids
    first_face = model.mesh.get_element(old_target[0])
    left_bottom, left_top = first_face.node_ids[0], first_face.node_ids[2]
    model.add_element(new_face, create_shell_element(
        new_face, (left_top, new_node, left_bottom), "elastic",
        thickness=0.05, formulation="e4-pl-s3-v2d",
        reference_normal=(0.0, 0.0, 1.0), reference_surface_offset=0.01,
    ))
    definition = replace(definition, surfaces=(
        definition.surfaces[0],
        replace(definition.surfaces[1], element_ids=(*old_target, new_face)),
    ))
    settings = replace(
        settings, q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    law = owner.law
    source = law.geometry.reference_source.copy()
    source[:, 0] += 0.05
    source[:, 2] -= 0.01504
    target = law.geometry.reference_target
    multipliers = np.zeros(len(law.geometry.source_node_ids))
    assert len(law.target_face_indices) == 3
    assert law._only_artificial_shared_edge_ties(source, target)
    work = law.work(source, target, multipliers, tangent=True)
    assert work.tie_resolution_relative_error < 1.0e-6
    for sign in (-1, 1):
        smooth_source = source.copy()
        smooth_source[:, 0] += sign*1.5e-8
        neighbor = law.work(smooth_source, target, multipliers, tangent=True)
        assert np.linalg.norm(work.residual-neighbor.residual)/np.linalg.norm(work.residual) < 1.0e-6
        assert np.linalg.norm(work.tangent-neighbor.tangent)/np.linalg.norm(work.tangent) < 1.0e-6
    plus = source.copy()
    minus = source.copy()
    plus[:, 2] += 1.0e-6
    minus[:, 2] -= 1.0e-6
    derivative = (law.work(plus, target, multipliers, tangent=False).residual
                  - law.work(minus, target, multipliers, tangent=False).residual)/2.0e-6
    direction = np.zeros_like(np.vstack((source, target)))
    direction[:len(source), 2] = 1.0
    action = work.tangent @ direction.ravel()
    assert np.linalg.norm(derivative-action)/max(1.0, np.linalg.norm(action)) < 1.0e-5
    forces = work.residual.reshape(-1, 3)
    points = np.vstack((source, target))
    assert np.linalg.norm(forces.sum(axis=0)) < 1.0e-6
    assert np.linalg.norm(np.cross(points, forces).sum(axis=0)) < 1.0e-6
    creased_reference = law.reference_target.copy()
    creased_reference[2, 2] += 1.0e-4
    law.reference_target = creased_reference
    assert not law._only_artificial_shared_edge_ties(source, target)
    with pytest.raises(UnsupportedQ4S3Branch):
        law.work(source, target, multipliers, tangent=True)


def _three_face_tie_geometry():
    law = object.__new__(MovingS3PatchProjectedNormal)
    law.reference_target = np.array((
        (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0),
        (0.5, 1.0, 0.0), (-0.5, 1.0, 0.0),
        (-1.5, 0.5, 0.0),
    ))
    law.target_face_indices = (
        np.array((0, 1, 3)), np.array((1, 2, 3)),
        np.array((3, 4, 0)),
    )
    law.source_face_indices = (np.arange(4),)
    law.shared_edge_tie = (3.0e-8, np.array((1.0, 0.0, 0.0)), 1.0e-6)
    return law


def test_three_face_tie_rejects_rim_crease_and_ambiguous_topology():
    law = _three_face_tie_geometry()
    smooth = law._reference_smooth_target_edges()
    assert (1, (1, 3)) in smooth[0]
    assert (2, (0, 3)) in smooth[0]
    rim_source = np.array(((0.5, 0.25, 0.1), (1.0, 0.25, 0.1),
                           (1.0, 0.75, 0.1), (0.5, 0.75, 0.1)))
    assert not law._only_artificial_shared_edge_ties(
        rim_source, law.reference_target)

    left_source = np.array(((-0.5, 0.25, 0.1), (-0.2, 0.25, 0.1),
                            (-0.2, 0.75, 0.1), (-0.5, 0.75, 0.1)))
    assert law._only_artificial_shared_edge_ties(left_source,
                                                  law.reference_target)
    # With three overlapped target faces, the full-patch derivative norm does
    # not certify the local edge; this branch must stay unavailable.
    busy_source = np.array(((-0.6, 0.25, 0.1), (0.5, 0.25, 0.1),
                            (0.5, 0.75, 0.1), (-0.6, 0.75, 0.1)))
    assert not law._only_artificial_shared_edge_ties(
        busy_source, law.reference_target)
    creased = law.reference_target.copy()
    creased[4, 2] += 1.0e-4
    assert not law._only_artificial_shared_edge_ties(left_source, creased)
    law.reference_target = creased
    assert (2, (0, 3)) not in law._reference_smooth_target_edges()[0]

    law = _three_face_tie_geometry()
    law.target_face_indices += (np.array((3, 1, 0)),)
    assert law._reference_smooth_target_edges() is None
    law = _three_face_tie_geometry()
    law.reference_target = np.vstack((law.reference_target, (-2.0, 0.0, 0.0)))
    law.target_face_indices += (np.array((0, 3, 5)),)
    assert law._reference_smooth_target_edges() is None
    law = _three_face_tie_geometry()
    law.reference_target = np.vstack((law.reference_target, (-0.5, 0.0, 0.0)))
    law.target_face_indices += (np.array((5, 2, 4)),)
    assert law._reference_smooth_target_edges() is None


@pytest.mark.parametrize("fixture_name, unresolved", [
    ("general_contact_q4_s3_recontact_shared_edge_pose.json",
     "degenerate integration triangle"),
    ("general_contact_q4_s3_augmented_shared_edge_pose.json",
     "paired overlap area agreement is unresolved"),
])
def test_recontact_tiny_shared_edge_wedge_uses_resolved_limit(
        fixture_name, unresolved):
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          fixture_name).read_text())
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    settings = replace(
        settings, q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    source = np.asarray(fixture["source"])
    target = np.asarray(fixture["target"])
    multipliers = np.asarray(fixture["multipliers"])
    assert owner.law._only_artificial_shared_edge_ties(
        source, target, failed_pair=2)
    with pytest.raises(UnsupportedQ4S3Branch, match=unresolved):
        owner.law._work_unresolved(source, target, multipliers, tangent=True)
    work = owner.law.work(source, target, multipliers, tangent=True)
    assert work.tie_resolution_relative_error < 1.0e-6
    assert work.energy_evaluations <= settings.maximum_general_s3_energy_evaluations
    force = work.residual.reshape(-1, 3)
    points = np.vstack((source, target))
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-8)
    np.testing.assert_allclose(np.cross(points, force).sum(axis=0),
                               0.0, atol=1.0e-8)


def test_tiny_wedge_requires_verified_reduced_derivatives():
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "general_contact_q4_s3_neighbor_value_gate_pose.json").read_text())
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    settings = replace(
        settings, q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    with pytest.raises(UnsupportedQ4S3Branch,
                       match="shared-edge tie smooth neighbor unresolved"):
        owner.law.work(np.asarray(fixture["source"]),
                       np.asarray(fixture["target"]),
                       np.asarray(fixture["multipliers"]), tangent=True)
    owner.law.set_tangent_active_coordinates([
        3*node+axis for node in range(len(owner.law.source_node_ids))
        for axis in (0, 2)
    ])
    work = owner.law.work(np.asarray(fixture["source"]),
                          np.asarray(fixture["target"]),
                          np.asarray(fixture["multipliers"]), tangent=True)
    assert work.tie_resolution_relative_error < 1.0e-6
    assert work.energy_evaluations <= settings.maximum_general_s3_energy_evaluations
    force = work.residual.reshape(-1, 3)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-8)


def test_exact_seam_uses_resolved_outer_directional_quotient():
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "general_contact_q4_s3_neighbor_value_gate_pose.json").read_text())
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    settings = replace(
        settings, q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    owner.law.set_tangent_active_coordinates([
        3*node+axis for node in range(len(owner.law.source_node_ids))
        for axis in (0, 2)
    ])
    # A saved rejected Schur proposal at this pose has concentrated pressure.
    multipliers = np.asarray((
        21.541356, 0.0, 0.0, 0.0, 0.0,
        13.067722, 1.0e-6, 0.0, 0.0, 0.0,
        43.148131, 0.0, 0.0, 0.0, 0.0,
    ))
    work = owner.law.work(
        np.asarray(fixture["source"]), np.asarray(fixture["target"]),
        multipliers, tangent=True,
    )
    assert work.tie_resolution_relative_error < 1.0e-6
    assert work.energy_evaluations <= settings.maximum_general_s3_energy_evaluations
