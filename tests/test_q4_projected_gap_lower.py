"""Independent sampled-pair controls for the candidate Q4 gap lower bound."""

import numpy as np
from scipy.optimize import least_squares

from anyfesolver.general_contact.q4_source_overlap import (
    q4_source_gap_lower, q4_source_projected_gap_lower,
    q4_source_projected_swept_gap_lower, q4_source_swept_gap_lower,
)


def _q4_point(corners, u, v):
    return (corners[0]*(1-u)*(1-v) + corners[1]*u*(1-v)
            + corners[2]*u*v + corners[3]*(1-u)*v)


def _q4_normal(corners, u, v):
    first = (1-v)*(corners[1]-corners[0]) + v*(corners[2]-corners[3])
    second = (1-u)*(corners[3]-corners[0]) + u*(corners[2]-corners[1])
    normal = np.cross(first, second)
    return normal/np.linalg.norm(normal)


def _target_point(corners, u, v):
    if len(corners) == 4:
        return _q4_point(corners, u, v)
    return corners[0] + u*(corners[1]-corners[0]) + v*(corners[2]-corners[0])


def _sampled_matching_gaps(source, target):
    projection = np.cross(source[1]-source[0], source[3]-source[0])
    projection /= np.linalg.norm(projection)
    first = source[1]-source[0]
    first -= (first @ projection)*projection
    first /= np.linalg.norm(first)
    axes = np.asarray((first, np.cross(projection, first)))
    gaps = []
    for u in np.linspace(0.0, 1.0, 13):
        for v in np.linspace(0.0, 1.0, 13):
            point = _q4_point(source, u, v)
            projected = axes @ point
            fit = least_squares(
                lambda values: axes @ _target_point(target, *values)-projected,
                x0=(0.5, 0.5), bounds=((0.0, 0.0), (1.0, 1.0)),
                xtol=1e-13, ftol=1e-13, gtol=1e-13,
            )
            if (np.linalg.norm(fit.fun) > 1e-9
                    or (len(target) == 3 and fit.x.sum() > 1.0+1e-10)):
                continue
            target_point = _target_point(target, *fit.x)
            gaps.append(float(_q4_normal(source, u, v) @
                              (point-target_point)))
    assert len(gaps) >= 10
    return np.asarray(gaps)


def test_projected_q4_bound_controls_planar_and_warped_target_samples():
    source = np.array(((0.1, 0.1, 0.03), (1.1, 0.1, 0.03),
                       (1.1, 1.1, 0.03), (0.1, 1.1, 0.03)))
    target = np.array(((0.0, 0.0, 0.015), (1.0, 0.0, 0.015),
                       (1.0, 1.0, 0.015), (0.0, 1.0, 0.015)))
    warped_source = source.copy()
    warped_source[0, 2] -= 0.01505
    warped_target = target.copy()
    warped_target[2, 2] += 0.002
    for first, second in (
        (warped_source, target), (warped_source, warped_target),
        (warped_source+np.array((0.0, 0.0, 0.03)), target),
        (warped_source, target[[0, 1, 3]]),
    ):
        lower = q4_source_projected_gap_lower(first, second)
        samples = _sampled_matching_gaps(first, second)
        assert np.all(samples >= lower-1e-10)
        assert lower >= q4_source_gap_lower(first, second)
    normal = np.cross(warped_source[1]-warped_source[0],
                      warped_source[3]-warped_source[0])
    normal /= np.linalg.norm(normal)
    exact_projection_distance = (
        warped_source[:, 2].min()-target[0, 2]
    )/normal[2]
    np.testing.assert_allclose(
        q4_source_projected_gap_lower(warped_source, target),
        exact_projection_distance, rtol=0.0, atol=1e-10,
    )
    assert q4_source_gap_lower(warped_source, target) < -1e-2


def test_projected_q4_bound_is_invariant_under_common_rigid_motion():
    source = np.array(((0.1, 0.1, 0.01495), (1.1, 0.1, 0.03),
                       (1.1, 1.1, 0.03), (0.1, 1.1, 0.03)))
    target = np.array(((0.0, 0.0, 0.015), (1.0, 0.0, 0.015),
                       (1.0, 1.0, 0.017), (0.0, 1.0, 0.015)))
    angle = 0.7
    rotation = np.array(((np.cos(angle), -np.sin(angle), 0.0),
                         (np.sin(angle), np.cos(angle), 0.0),
                         (0.0, 0.0, 1.0)))
    first = q4_source_projected_gap_lower(source, target)
    moved = q4_source_projected_gap_lower(
        source @ rotation.T+np.array((2.0, -1.0, 0.4)),
        target @ rotation.T+np.array((2.0, -1.0, 0.4)),
    )
    np.testing.assert_allclose(moved, first, rtol=0.0, atol=1e-11)


def test_projected_q4_swept_bound_controls_independent_path_samples():
    source_start = np.array(((0.1, 0.1, 0.03), (1.1, 0.1, 0.03),
                             (1.1, 1.1, 0.03), (0.1, 1.1, 0.03)))
    source_end = source_start.copy()
    source_end[0, 2] -= 0.01505
    target_start = np.array(((0.0, 0.0, 0.015), (1.0, 0.0, 0.015),
                             (1.0, 1.0, 0.015), (0.0, 1.0, 0.015)))
    for target_end in (target_start.copy(),
                       target_start+np.array((0.0, 0.0, 0.01))):
        lower = q4_source_projected_swept_gap_lower(
            source_start, source_end, target_start, target_end,
            np.array((0.0, 0.0, 1.0)),
        )
        old = q4_source_swept_gap_lower(
            source_start, source_end, target_start, target_end,
            np.array((0.0, 0.0, 1.0)),
        )
        assert lower >= old
        for time in np.linspace(0.0, 1.0, 11):
            first = (1-time)*source_start+time*source_end
            second = (1-time)*target_start+time*target_end
            assert np.all(_sampled_matching_gaps(first, second) >= lower-1e-10)
    fixed_target_lower = q4_source_projected_swept_gap_lower(
        source_start, source_end, target_start, target_start,
        np.array((0.0, 0.0, 1.0)),
    )
    assert fixed_target_lower > -6e-5


def test_projected_q4_swept_bound_falls_back_outside_narrow_normal_cone():
    source_start = np.array(((0.1, 0.1, 0.03), (1.1, 0.1, 0.03),
                             (1.1, 1.1, 0.03), (0.1, 1.1, 0.03)))
    target = np.array(((0.0, 0.0, 0.015), (1.0, 0.0, 0.015),
                       (1.0, 1.0, 0.015), (0.0, 1.0, 0.015)))
    angle = 1.3
    rotation = np.array(((np.cos(angle), 0.0, np.sin(angle)),
                         (0.0, 1.0, 0.0),
                         (-np.sin(angle), 0.0, np.cos(angle))))
    source_end = source_start @ rotation.T
    improved = q4_source_projected_swept_gap_lower(
        source_start, source_end, target, target,
        np.array((0.0, 0.0, 1.0)),
    )
    old = q4_source_swept_gap_lower(
        source_start, source_end, target, target,
        np.array((0.0, 0.0, 1.0)),
    )
    assert improved == old
