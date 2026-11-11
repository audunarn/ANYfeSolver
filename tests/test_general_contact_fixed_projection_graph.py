"""Independent controls for the restricted fixed-projection swept certificate."""

import numpy as np
import pytest

from anyfesolver.general_contact.fixed_projection_graph import (
    fixed_projection_graph_swept_certificate,
)


def _corner_path(amplitude):
    points = ((.91, .1), (.915, .1), (.915, .125), (.91, .125))
    start = np.asarray([(x, y, .03) for x, y in points])
    end = np.asarray([(x, y, .03-amplitude*(x-.1)) for x, y in points])
    target = np.asarray(((.92, 0., .015),
                         (.86, 1.15, .015),
                         (.15, .95, .015)))
    return start, end, target


def _rotation():
    axis = np.asarray((1., 2., -1.))
    axis /= np.linalg.norm(axis)
    angle = .73
    cross = np.asarray(((0., -axis[2], axis[1]),
                        (axis[2], 0., -axis[0]),
                        (-axis[1], axis[0], 0.)))
    return (np.eye(3)*np.cos(angle)
            + (1.-np.cos(angle))*np.outer(axis, axis)
            + np.sin(angle)*cross)


def test_skew_corner_swept_certificate_is_tighter_and_rigid_invariant():
    start, end, target = _corner_path(.018453)
    accepted = fixed_projection_graph_swept_certificate(
        start, end, target, target, 5.e-5,
    )
    assert accepted.admissible
    assert -5.e-5 < accepted.vertical_lower_m < 0.
    assert accepted.expanded_overlap_vertices >= 3
    rejected = fixed_projection_graph_swept_certificate(
        start, _corner_path(.0185)[1], target, target, 5.e-5,
    )
    assert not rejected.admissible
    rotation = _rotation()
    translation = np.asarray((3., -2., 1.))
    transformed = fixed_projection_graph_swept_certificate(
        start@rotation.T+translation, end@rotation.T+translation,
        target@rotation.T+translation, target@rotation.T+translation, 5.e-5,
    )
    assert transformed.admissible
    assert transformed.vertical_lower_m == pytest.approx(
        accepted.vertical_lower_m, abs=1.e-11,
    )


def test_fixed_projection_rejects_motion_and_keeps_empty_overlap_safe():
    start, end, target = _corner_path(.0185)
    shifted = target+np.asarray((2., 0., 0.))
    empty = fixed_projection_graph_swept_certificate(
        start, end, shifted, shifted, 5.e-5,
    )
    assert empty.admissible and empty.expanded_overlap_vertices == 0
    moved = target+np.asarray((.001, 0., 0.))
    with pytest.raises(ValueError, match="in-plane motion"):
        fixed_projection_graph_swept_certificate(
            start, end, target, moved, 5.e-5,
        )
    tilted = target.copy()
    tilted[1, 2] += .001
    with pytest.raises(ValueError, match="planar parallel target"):
        fixed_projection_graph_swept_certificate(
            start, end, target, tilted, 5.e-5,
        )


def test_target_normal_motion_and_small_projected_cut_are_checked():
    source = np.asarray(((0., 0., .02), (1., 0., .02),
                         (1., 1., .02), (0., 1., .02)))
    delta = 1.e-5
    target = np.asarray(((1.-delta, .5, 0.),
                         (1.+delta, .5-delta, 0.),
                         (1.+delta, .5+delta, 0.)))
    safe = fixed_projection_graph_swept_certificate(
        source, source, target, target+np.asarray((0., 0., .019)), .002,
    )
    assert safe.admissible and safe.expanded_overlap_vertices >= 3
    unsafe = fixed_projection_graph_swept_certificate(
        source, source, target, target+np.asarray((0., 0., .023)), .002,
    )
    assert not unsafe.admissible


def _inside_triangle(point, triangle):
    for first, second in zip(triangle, np.roll(triangle, -1, axis=0)):
        edge = second-first
        relative = point-first
        if edge[0]*relative[1]-edge[1]*relative[0] < -1.e-12:
            return False
    return True


def test_safe_random_graph_paths_have_no_sampled_threshold_crossing():
    rng = np.random.default_rng(1039)
    xy = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    target = np.asarray(((.2, .2, 0.), (.8, .1, 0.), (.4, .9, 0.)))
    tau = .005
    safe = unsafe = 0
    for _ in range(24):
        heights_start = rng.uniform(.015, .04, 4)
        heights_end = rng.uniform(-.012, .03, 4)
        start = np.column_stack((xy, heights_start))
        end = np.column_stack((xy, heights_end))
        certificate = fixed_projection_graph_swept_certificate(
            start, end, target, target, tau,
        )
        safe += int(certificate.admissible)
        unsafe += int(not certificate.admissible)
        if not certificate.admissible:
            continue
        for time in np.linspace(0., 1., 11):
            heights = (1.-time)*heights_start+time*heights_end
            for u in np.linspace(0., 1., 17):
                for v in np.linspace(0., 1., 17):
                    z = ((1.-u)*(1.-v)*heights[0]
                         + u*(1.-v)*heights[1]
                         + u*v*heights[2]
                         + (1.-u)*v*heights[3])
                    zx = ((1.-v)*(heights[1]-heights[0])
                          + v*(heights[2]-heights[3]))
                    zy = ((1.-u)*(heights[3]-heights[0])
                          + u*(heights[2]-heights[1]))
                    cosine = 1./np.sqrt(1.+zx*zx+zy*zy)
                    distance = z/cosine
                    target_xy = np.asarray((u+distance*zx*cosine,
                                            v+distance*zy*cosine))
                    if _inside_triangle(target_xy, target[:, :2]):
                        assert distance >= -tau-1.e-12
    assert safe > 0 and unsafe > 0
