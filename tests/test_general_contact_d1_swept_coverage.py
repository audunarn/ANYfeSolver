"""Independent geometry controls for the D1 physical-path union certificate."""

from fractions import Fraction
from math import comb
import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.general_contact.d1_interior_profile import (
    _bernstein_cross, _bernstein_dot, _interval_sub,
    swept_projected_coverage_admissible,
)


QUAD = np.array(((0, 1, 2, 3),))
TRIANGLES = np.array(((0, 1, 2), (0, 2, 3)))


def square(radius=1.0):
    return np.array(((-radius, -radius, 0.), (radius, -radius, 0.),
                     (radius, radius, 0.), (-radius, radius, 0.)))


def rotate(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array(((0., -axis[2], axis[1]), (axis[2], 0., -axis[0]),
                      (-axis[1], axis[0], 0.)))
    return (np.eye(3)*np.cos(angle) + (1-np.cos(angle))*np.outer(axis, axis)
            + np.sin(angle)*cross)


def boundary_margins(source, target):
    """Independent direct four-edge determinant; no production clipping."""
    normal = np.cross(source[1]-source[0], source[3]-source[0])
    return np.array([
        np.cross(target[(i+1) % 4]-target[i], point-target[i]) @ normal
        for i in range(4) for point in source
    ])


def certificate(a0, a1, b0, b1, triangles=TRIANGLES):
    return swept_projected_coverage_admissible(a0, a1, b0, b1, QUAD, triangles)


@pytest.mark.parametrize("moving_source", [False, True])
def test_endpoint_covered_rotating_target_has_positive_interior_cut(moving_source):
    source = square()
    target = square(1.4)
    common = rotate((1., .2, .1), .35) if moving_source else np.eye(3)
    source_end = source @ common.T
    target_end = target @ (common @ rotate((0., 0., 1.), np.pi/3)).T
    assert np.min(boundary_margins(source, target)) > 0.
    assert np.min(boundary_margins(source_end, target_end)) > 0.
    assert np.min(boundary_margins(.5*(source+source_end),
                                   .5*(target+target_end))) < -.01
    assert not certificate(source, source_end, target, target_end)
    # Coordinate pose changes cannot conceal the same physical cut.
    pose = rotate((1., 2., 3.), .61)
    translation = np.array((7., -3., 2.))
    transformed = [item @ pose.T + translation
                   for item in (source, source_end, target, target_end)]
    assert not certificate(*transformed)


def test_narrow_interior_cut_between_several_covered_temporal_samples():
    source = square(1.e-6)

    def target_at(time):
        s = time-.37
        a = np.array((s-2., 1.e-4-2.*s, 0.))
        edge = np.array((4., 4.*s, 0.))
        inward = np.array((-4.*s, 4., 0.))
        return np.array((a, a+edge, a+edge+inward, a+inward))

    first, last = target_at(0.), target_at(1.)
    for time in (0., .25, .5, .75, 1.):
        np.testing.assert_allclose((1-time)*first+time*last, target_at(time))
        assert np.min(boundary_margins(source, target_at(time))) > 0.
    # At the centre q=(0,0), the first-edge determinant divided by 4 is
    # exactly (t-.37)^2-1e-4: a positive-area cut exists near t=.37.
    assert np.min(boundary_margins(source, target_at(.37))) < 0.
    assert not certificate(source, source, first, last)


def test_covered_union_accepts_internal_seam_and_common_physical_rotation():
    source = square()
    target = square(2.)
    rotation = rotate((1., .4, .2), .3)
    translation = np.array((.02, -.03, .01))
    source_end = source @ rotation.T + translation
    target_end = target @ rotation.T + translation
    assert certificate(source, source_end, target, target_end)
    assert certificate(source_end, source, target_end, target)
    # Reversing every target face preserves the physical union.
    assert certificate(source, source_end, target, target_end,
                       TRIANGLES[:, ::-1])
    # Nonplanar deformable interior node: only the physical external boundary
    # enters containment, while all four projected triangle areas stay positive.
    centre_target = np.vstack((target, (0., 0., .05)))
    centre_end = np.vstack((target_end, (.02, -.03, .08)))
    fan = np.array(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    assert certificate(source, source_end, centre_target, centre_end, fan)


def test_no_overlap_is_certified_but_entry_through_positive_cut_is_rejected():
    source = square()
    target = square(.5) + np.array((4., 0., 0.))
    assert certificate(source, source, target, target+(.1, .2, .01))
    other_side = target - np.array((8., 0., 0.))
    assert not certificate(source, source, target, other_side)


@pytest.mark.parametrize("defect", ["nonfinite", "collapsed", "grazing",
                                   "mixed_winding", "duplicate", "pinched",
                                   "touching", "tiny_cut"])
def test_invalid_or_ill_conditioned_geometry_fails_closed(defect):
    source, target = square(), square(2.)
    faces = TRIANGLES
    if defect == "nonfinite":
        source[0, 0] = np.nan
    elif defect == "collapsed":
        source[:, 1] = 0.
    elif defect == "grazing":
        target = target @ rotate((1., 0., 0.), np.pi/2-1.e-10).T
    elif defect == "mixed_winding":
        faces = np.array(((0, 1, 2), (0, 3, 2)))
    elif defect == "duplicate":
        faces = np.vstack((TRIANGLES, TRIANGLES))
    elif defect == "pinched":
        target = np.vstack((target, (4., 2., 0.), (4., 4., 0.)))
        faces = np.vstack((TRIANGLES, (2, 4, 5)))
    elif defect == "touching":
        target = square()
    elif defect == "tiny_cut":
        target = square()
        target[1:3, 0] -= 1.e-12
    assert not certificate(source, source, target, target, faces)


@pytest.mark.parametrize("scale", [1.e-160, 1.e100, 1.e200])
def test_extreme_geometric_conditioning_fails_without_overflow_escape(scale):
    source, target = square(), square(2.)
    source[:, 0] *= scale
    target[:, 0] *= scale
    if scale == 1.e100:
        # Finite areas but an unresolvable aspect ratio; the scale screen must
        # reject rather than overflowing a Python float power operation.
        source[:, 1] /= scale
        target[:, 1] /= scale
    assert not certificate(source, source, target, target)


def test_malformed_connectivity_and_coordinate_shapes_fail_closed():
    source, target = square(), square(2.)
    assert not swept_projected_coverage_admissible(
        source, source, target, target, ((0, 1), (2, 3, 0)), TRIANGLES)
    assert not swept_projected_coverage_admissible(
        source, source, target, target, QUAD, ((0, 1, 8),))
    assert not swept_projected_coverage_admissible(
        source, source[:3], target, target, QUAD, TRIANGLES)


def test_bernstein_intervals_enclose_independently_exact_quartic_coefficients():
    # All coordinates are dyadic; Fraction preserves the supplied binary floats.
    a = np.array(((.25, -.5, .125), (.75, .25, -.25)))
    b = np.array(((1., .25, -.125), (1.25, -.25, .375)))
    q = np.array(((.5, 1., .5), (-.25, .75, .125)))
    c = np.array(((0., .25, -.5), (.5, -.25, .125)))
    d = np.array(((.25, 1., .125), (1., .5, -.25)))
    bounds = _bernstein_dot(
        _bernstein_cross(_interval_sub((b, b), (a, a)),
                         _interval_sub((q, q), (a, a))),
        _bernstein_cross((c, c), (d, d)))

    def linear(points):
        first = [Fraction(float(value)) for value in points[0]]
        last = [Fraction(float(value)) for value in points[1]]
        return [first, [y-x for x, y in zip(first, last)]]

    def difference(x, y):
        return [[a-b for a, b in zip(u, v)] for u, v in zip(x, y)]

    def cross(x, y):
        result = [[Fraction(0)]*3 for _ in range(len(x)+len(y)-1)]
        for i, u in enumerate(x):
            for j, v in enumerate(y):
                for axis, (h, k) in enumerate(((1, 2), (2, 0), (0, 1))):
                    result[i+j][axis] += u[h]*v[k]-u[k]*v[h]
        return result

    first = cross(difference(linear(b), linear(a)),
                  difference(linear(q), linear(a)))
    second = cross(linear(c), linear(d))
    power = [Fraction(0)]*5
    for i, u in enumerate(first):
        for j, v in enumerate(second):
            power[i+j] += sum(x*y for x, y in zip(u, v))
    exact = [sum(power[j]*Fraction(comb(k, j), comb(4, j))
                 for j in range(k+1)) for k in range(5)]
    for lower, truth, upper in zip(bounds[0], exact, bounds[1]):
        assert Fraction(float(lower)) <= truth <= Fraction(float(upper))


def test_actual_eight_s3_captured_physical_skin_trial_is_covered():
    # Reuse the actual preserved development geometry, without a coupled solve.
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    from scripts.general_contact_d1_interior_press_dev import build_interior_press

    path = Path(__file__).resolve().parents[1] / (
        "docs/d1_review_support/d1_target8s3_swept_capture_20260929.json")
    capture = json.loads(path.read_text(encoding="utf-8"))["swept_failure_discriminator"]
    model, definition, settings, _ = build_interior_press(
        force_n=.04, reference_slope=1.e-4,
        source_support="perimeter", source_columns=8, source_rows=8,
        target_columns=2, target_rows=2,
        target_reference_full_left_clamp=True,
        load_shape="physical_bump", slide_body="target")
    manager = GeneralS3PatchManager(model, definition, settings)
    first = manager._positions(np.asarray(capture["captured_start_displacement"]))
    last = manager._positions(np.asarray(capture["captured_proposed_displacement"]))
    assert swept_projected_coverage_admissible(
        first[0], last[0], first[1], last[1],
        manager.law.source_face_indices, manager.law.target_face_indices)
