"""Independent geometric controls for the internal C1 common-refinement kernel."""

import numpy as np
import pytest

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.overlap import (
    PlanarFace,
    current_planar_shell_face,
    planar_projected_overlap,
    planar_overlap_translation_gradient,
    planar_overlap_translation_derivatives,
    swept_planar_overlap_interval,
    reference_shell_face,
)


def _face(feature, xy, z=0.0):
    return PlanarFace(feature, np.asarray([(x, y, z) for x, y in xy], dtype=float))


def test_opposed_q4_faces_preserve_their_original_interpolation():
    integration = _face((1, 1), ((0, 0), (1, 0), (1, 1), (0, 1)))
    target = _face((2, 1), ((0.5, 0), (0.5, 1), (1.5, 1), (1.5, 0)), z=0.1)
    overlap = planar_projected_overlap(integration, target)
    assert overlap is not None
    assert overlap.area == pytest.approx(0.5)
    assert len(overlap.polygon) == 4
    point = np.array((0.75, 0.5, 0.0))
    for face in (integration, target):
        shape = overlap.shape_values(face, point)
        assert min(shape) >= 0.0
        np.testing.assert_allclose((shape @ face.corners)[:2], point[:2], atol=1e-12)
    assert (overlap.shape_values(target, point) @ target.corners)[2] == pytest.approx(0.1)


def test_q4_s3_partial_overlap_has_the_analytical_area():
    square = _face((1, 1), ((0, 0), (1, 0), (1, 1), (0, 1)))
    triangle = _face((2, 1), ((0.5, 0), (1.5, 0), (0.5, 1)), z=0.2)
    overlap = planar_projected_overlap(square, triangle)
    assert overlap is not None
    assert overlap.area == pytest.approx(0.375)
    centroid = np.mean(overlap.polygon, axis=0)
    assert sum(overlap.shape_values(square, centroid)) == pytest.approx(1.0)
    assert sum(overlap.shape_values(triangle, centroid)) == pytest.approx(1.0)
    for face in (square, triangle):
        values, gradient = overlap.shape_values_and_gradient(face, centroid)
        np.testing.assert_allclose(values, overlap.shape_values(face, centroid),
                                   atol=1e-12)
        np.testing.assert_allclose(np.sum(gradient, axis=0), 0.0, atol=1e-12)
        direction = 0.3 * overlap.tangent - 0.2 * overlap.binormal
        h = 1.0e-6
        finite_difference = (
            overlap.shape_values(face, centroid + h * direction)
            - overlap.shape_values(face, centroid - h * direction)
        ) / (2.0 * h)
        np.testing.assert_allclose(finite_difference, gradient @ direction,
                                   rtol=1e-7, atol=1e-9)


def test_s3_s3_partial_overlap_has_the_analytical_area():
    first = _face((1, 1), ((0, 0), (1, 0), (0, 1)))
    second = _face((2, 1), ((0.5, 0), (1.5, 0), (0.5, 1)), z=-0.1)
    overlap = planar_projected_overlap(first, second)
    assert overlap is not None
    assert overlap.area == pytest.approx(0.125)


def test_crossed_faces_are_found_when_no_vertex_is_inside_the_other_face():
    horizontal = _face((1, 1), ((-1, -0.1), (1, -0.1),
                                (1, 0.1), (-1, 0.1)))
    vertical = _face((2, 1), ((-0.1, -1), (0.1, -1),
                              (0.1, 1), (-0.1, 1)), z=0.01)
    overlap = planar_projected_overlap(horizontal, vertical)
    assert overlap is not None
    assert overlap.area == pytest.approx(0.04)


def test_projected_area_and_shape_values_are_rigid_frame_invariant():
    square = _face((1, 1), ((0, 0), (1, 0), (1, 1), (0, 1)))
    triangle = _face((2, 1), ((0.5, 0), (1.5, 0), (0.5, 1)), z=0.2)
    reference = planar_projected_overlap(square, triangle)
    rotation = np.array(((0.0, -1.0, 0.0),
                         (0.0, 0.0, -1.0),
                         (1.0, 0.0, 0.0)))
    shift = np.array((1.0, -2.0, 3.0))
    rotated_square = PlanarFace((1, 1), square.corners @ rotation.T + shift)
    rotated_triangle = PlanarFace((2, 1), triangle.corners @ rotation.T + shift)
    rotated = planar_projected_overlap(rotated_square, rotated_triangle)
    assert rotated is not None
    assert rotated.area == pytest.approx(reference.area)
    point = np.mean(reference.polygon, axis=0)
    rotated_point = rotation @ point + shift
    np.testing.assert_allclose(rotated.shape_values(rotated_square, rotated_point),
                               reference.shape_values(square, point), atol=1e-12)
    np.testing.assert_allclose(rotated.shape_values(rotated_triangle, rotated_point),
                               reference.shape_values(triangle, point), atol=1e-12)


def test_overlap_uses_face_scale_for_micrometre_geometry():
    scale = 1.0e-6
    first = _face((1, 1), tuple((scale * x, scale * y)
                                 for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))))
    second = _face((2, 1), tuple((scale * x, scale * y)
                                  for x, y in ((0.5, 0), (1.5, 0),
                                               (1.5, 1), (0.5, 1))))
    overlap = planar_projected_overlap(first, second)
    assert overlap is not None
    assert overlap.area == pytest.approx(0.5 * scale * scale, rel=1e-9)
    point = np.array((0.75 * scale, 0.5 * scale, 0.0))
    np.testing.assert_allclose(overlap.shape_values(first, point),
                               (0.125, 0.375, 0.375, 0.125), atol=1e-10)


def test_warped_and_crossed_q4_are_rejected():
    with pytest.raises(ValueError, match="planar"):
        PlanarFace((1, 1), np.array(((0, 0, 0), (1, 0, 0),
                                    (1, 1, 0.1), (0, 1, 0))))
    with pytest.raises(ValueError, match="strictly convex"):
        _face((1, 1), ((0, 0), (1, 1), (0, 1), (1, 0)))


def test_reference_shell_adapter_uses_s3_material_offset_and_physical_director():
    model = FEModel("reference_shell_skins")
    model.add_material("elastic", 1.0e5, 0.3)
    for node_id, xyz in enumerate(((0, 0, 0), (1, 0, 0),
                                   (0, 1, 0), (1, 1, 0)), 1):
        model.add_node(node_id, *xyz)
    model.add_element(1, create_shell_element(
        1, [1, 2, 3], "elastic", formulation="e4-pl-s3-v2d",
        thickness=0.2, reference_normal=(0, 0, 1),
        reference_surface_offset=0.03,
    ))
    model.add_element(2, create_shell_element(
        2, [1, 2, 4, 3], "elastic", thickness=0.2,
    ))
    assert reference_shell_face(model, 7, 1, "top").corners[:, 2] == pytest.approx(
        (0.07,) * 3
    )
    assert reference_shell_face(model, 7, 1, "midsurface").corners[:, 2] == pytest.approx(
        (-0.03,) * 3
    )
    assert reference_shell_face(model, 7, 1, "bottom").corners[:, 2] == pytest.approx(
        (-0.13,) * 3
    )
    assert reference_shell_face(model, 8, 2, "top").corners[:, 2] == pytest.approx(
        (0.1,) * 4
    )


def test_current_physical_faces_track_relative_sliding_and_recontact():
    model = FEModel("trial_shell_overlap")
    model.add_material("elastic", 1.0e5, 0.3)
    square = ((0, 0), (1, 0), (1, 1), (0, 1))
    for node_id, (x, y, z) in enumerate(
        (*((x, y, 0.0) for x, y in square),
         *((x, y, 0.05) for x, y in square)), 1
    ):
        model.add_node(node_id, x, y, z)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.05,
    ))
    model.add_element(2, create_shell_element(
        2, (5, 6, 7, 8), "elastic", thickness=0.05,
    ))
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    target = current_planar_shell_face(model, 2, 1, "top", displacement)
    for shift, expected_area in ((0.0, 1.0), (0.5, 0.5), (1.1, None),
                                 (0.25, 0.75)):
        for node_id in (5, 6, 7, 8):
            displacement[model.mesh.dof_manager.get_node_dofs(node_id)[0]] = shift
        integration = current_planar_shell_face(
            model, 1, 2, "bottom", displacement
        )
        overlap = planar_projected_overlap(integration, target)
        if expected_area is None:
            assert overlap is None
        else:
            assert overlap is not None
            assert overlap.area == pytest.approx(expected_area)
    displacement[:] = 0.0
    displacement[model.mesh.dof_manager.get_node_dofs(8)[2]] = 0.01
    with pytest.raises(ValueError, match="planar"):
        current_planar_shell_face(model, 1, 2, "bottom", displacement)


def test_current_s3_skin_uses_rotated_material_offset():
    model = FEModel("trial_s3_skin")
    model.add_material("elastic", 1.0e5, 0.3)
    for node_id, (x, y) in enumerate(((0, 0), (1, 0), (0, 1)), 1):
        model.add_node(node_id, x, y, 0.0)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3), "elastic", formulation="e4-pl-s3-v2d",
        thickness=0.2, reference_normal=(0, 0, 1),
        reference_surface_offset=0.03,
    ))
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (1, 2, 3):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        displacement[dofs[2]] = 0.02
        displacement[dofs[4]] = 0.1
    face = current_planar_shell_face(model, 1, 1, "top", displacement)
    assert face.corners[:, 0] == pytest.approx((0.007, 1.007, 0.007))
    assert face.corners[:, 2] == pytest.approx((0.09, 0.09, 0.09))


def test_projected_overlap_boundary_derivative_matches_independent_area_difference():
    integration = _face((1, 1), ((0, 0), (1, 0), (1, 1), (0, 1)))
    target = _face((2, 1), ((0.5, 0.25), (1.5, 0.25),
                             (1.5, 1.25), (0.5, 1.25)))
    gradient = planar_overlap_translation_gradient(integration, target)
    np.testing.assert_allclose(gradient, (0.75, 0.5, 0.0), atol=1e-12)
    direction = np.array((0.3, -0.4, 0.0))
    h = 1.0e-6
    plus = PlanarFace((1, 1), integration.corners + h * direction)
    minus = PlanarFace((1, 1), integration.corners - h * direction)
    numerical = (planar_projected_overlap(plus, target).area
                 - planar_projected_overlap(minus, target).area) / (2.0 * h)
    assert numerical == pytest.approx(float(gradient @ direction), rel=1e-8)
    distant = _face((2, 2), ((2, 0), (3, 0), (3, 1), (2, 1)))
    np.testing.assert_array_equal(
        planar_overlap_translation_gradient(integration, distant),
        np.zeros(3),
    )


def test_q4_s3_overlap_boundary_derivative_is_rigid_frame_covariant():
    square = _face((1, 1), ((0, 0), (1, 0), (1, 1), (0, 1)))
    triangle = _face((2, 1), ((0.5, 0.1), (1.5, 0.1), (0.5, 1.1)))
    gradient = planar_overlap_translation_gradient(square, triangle)
    derivative_gradient, hessian = planar_overlap_translation_derivatives(
        square, triangle
    )
    np.testing.assert_allclose(derivative_gradient, gradient, atol=1e-12)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
    h = 1.0e-6
    direction = np.array((0.3, 0.2, 0.0))
    plus = PlanarFace((1, 1), square.corners + h * direction)
    minus = PlanarFace((1, 1), square.corners - h * direction)
    numerical = (planar_projected_overlap(plus, triangle).area
                 - planar_projected_overlap(minus, triangle).area) / (2.0 * h)
    assert numerical == pytest.approx(float(gradient @ direction), rel=1e-8)
    gradient_difference = (
        planar_overlap_translation_gradient(plus, triangle)
        - planar_overlap_translation_gradient(minus, triangle)
    ) / (2.0 * h)
    np.testing.assert_allclose(gradient_difference, hessian @ direction,
                               rtol=1e-7, atol=1e-9)
    rotation = np.array(((0.0, -1.0, 0.0),
                         (0.0, 0.0, -1.0),
                         (1.0, 0.0, 0.0)))
    shift = np.array((1.0, -2.0, 3.0))
    rotated_square = PlanarFace((1, 1), square.corners @ rotation.T + shift)
    rotated_triangle = PlanarFace((2, 1), triangle.corners @ rotation.T + shift)
    np.testing.assert_allclose(
        planar_overlap_translation_gradient(rotated_square, rotated_triangle),
        rotation @ gradient, atol=1e-12,
    )
    _rotated_gradient, rotated_hessian = planar_overlap_translation_derivatives(
        rotated_square, rotated_triangle
    )
    np.testing.assert_allclose(rotated_hessian, rotation @ hessian @ rotation.T,
                               atol=1e-12)


def test_swept_overlap_detects_a_crossing_missed_by_both_endpoints():
    source = _face((1, 1), ((0, 0), (1, 0), (1, 1), (0, 1)))
    target = _face((2, 1), ((2, 0), (3, 0), (3, 1), (2, 1)))
    zero = np.zeros(3)
    travel = np.array((4.0, 0.0, 0.0))
    assert planar_projected_overlap(source, target) is None
    assert planar_projected_overlap(
        PlanarFace(source.feature_id, source.corners + travel), target
    ) is None
    interval = swept_planar_overlap_interval(
        source, target, zero, travel, zero, zero,
    )
    assert interval == pytest.approx((0.25, 0.75))
    for fraction in (0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0):
        moved = PlanarFace(source.feature_id, source.corners + fraction * travel)
        overlap = planar_projected_overlap(moved, target)
        assert (overlap is not None) == (0.25 < fraction < 0.75)
    assert swept_planar_overlap_interval(
        source, target, zero, np.array((-1.0, 0.0, 0.0)), zero, zero,
    ) is None


def test_swept_overlap_interval_is_rigid_frame_invariant():
    source = _face((1, 1), ((0, 0), (1, 0), (0, 1)))
    target = _face((2, 1), ((2, 0), (3, 0), (2, 1)))
    zero = np.zeros(3)
    travel = np.array((4.0, 0.0, 0.0))
    reference = swept_planar_overlap_interval(
        source, target, zero, travel, zero, zero,
    )
    rotation = np.array(((0.0, -1.0, 0.0),
                         (0.0, 0.0, -1.0),
                         (1.0, 0.0, 0.0)))
    shift = np.array((1.0, -2.0, 3.0))
    moved_source = PlanarFace(source.feature_id, source.corners @ rotation.T + shift)
    moved_target = PlanarFace(target.feature_id, target.corners @ rotation.T + shift)
    observed = swept_planar_overlap_interval(
        moved_source, moved_target, zero, rotation @ travel, zero, zero,
    )
    assert observed == pytest.approx(reference)
