"""Independent checks for the instantaneous planar overlap gap ingredient."""

import numpy as np
import pytest

from anyfesolver.general_contact.q4_source_overlap import (
    q4_source_planar_overlap_gap_lower,
)


def _skew_corner():
    amplitude = .018453
    xlo, xhi = .91, .915
    ylo, yhi = .1, .125
    source = np.asarray([
        (x, y, .03-amplitude*(x-.1))
        for x, y in ((xlo, ylo), (xhi, ylo),
                     (xhi, yhi), (xlo, yhi))
    ])
    target = np.asarray([
        (.92, 0., .015), (.86, 1.15, .015), (.15, .95, .015),
    ])
    return source, target, amplitude


def test_planar_skew_corner_uses_overlap_instead_of_unmatched_face_corner():
    source, target, amplitude = _skew_corner()
    lower = q4_source_planar_overlap_gap_lower(source, target)
    assert lower is not None
    # Independent clipping: the right target edge intersects the source's
    # y=0.1 boundary here; the affine vertical gap decreases with x.
    right_edge_x = .92+(.86-.92)*(.1/1.15)
    independent_vertical_gap = .015-amplitude*(right_edge_x-.1)
    assert lower == pytest.approx(independent_vertical_gap, abs=1.e-8)
    assert -5.e-5 < lower < 0.


def test_planar_overlap_gap_is_rigid_invariant_and_has_explicit_domain():
    source, target, _amplitude = _skew_corner()
    baseline = q4_source_planar_overlap_gap_lower(source, target)
    rotation = np.asarray(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
    translation = np.asarray((3., -2., 1.))
    transformed = q4_source_planar_overlap_gap_lower(
        source @ rotation.T+translation,
        target @ rotation.T+translation,
    )
    assert transformed == pytest.approx(baseline, abs=1.e-11)
    assert q4_source_planar_overlap_gap_lower(
        source, target+np.asarray((2., 0., 0.)),
    ) is None
    warped = source.copy()
    warped[2, 2] += .001
    with pytest.raises(ValueError, match="planar face"):
        q4_source_planar_overlap_gap_lower(warped, target)
    inverted = source[[0, 2, 1, 3]]
    with pytest.raises(ValueError, match="strictly convex"):
        q4_source_planar_overlap_gap_lower(inverted, target)


def test_planar_overlap_gap_keeps_a_small_positive_area_cut():
    source = np.asarray(((0., 0., 0.), (1., 0., 0.),
                         (1., 1., 0.), (0., 1., 0.)))
    delta = 1.e-5
    target = np.asarray(((1.-delta, .5, -.002),
                         (1.+delta, .5-delta, -.002),
                         (1.+delta, .5+delta, -.002)))
    assert q4_source_planar_overlap_gap_lower(source, target) == pytest.approx(
        .002, abs=1.e-10,
    )
