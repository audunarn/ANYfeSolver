"""Independent sampled controls for the curved-target source-chord bound."""

import numpy as np
import pytest

from anyfesolver.general_contact.beam_curve_pair import (
    _source_interval_target_gap_lower,
)


def _sample_curve(nodes, stations):
    stations = np.asarray(stations)
    return ((2*stations*stations - 3*stations + 1)[:, None]*nodes[0]
            + (4*stations - 4*stations*stations)[:, None]*nodes[1]
            + (2*stations*stations - stations)[:, None]*nodes[2])


def test_curved_target_chord_bound_resolves_captured_near_contact_interval():
    source = np.array(((0., 0., 0.), (0.5, 0., 0.), (1., 0., 0.)))
    target = np.array(((0., 0.09998668964006521, 0.),
                       (0.5, 0.10000665245358208, 0.),
                       (1., 0.09996, 0.)))
    left, right = 0.15625, 0.1562504768371582
    lower = _source_interval_target_gap_lower(source, left, right, target, 0.1)
    assert 0.0 < lower < 8.38e-8
    source_points = _sample_curve(source, np.linspace(left, right, 17))
    target_points = _sample_curve(target, np.linspace(0.0, 1.0, 10001))
    sampled = min(
        float(np.min(np.linalg.norm(target_points - point, axis=1))) - 0.1
        for point in source_points
    )
    assert lower <= sampled
    large_origin = np.array((1.0e6, -2.0e6, 3.0e6))
    shifted_lower = _source_interval_target_gap_lower(
        source + large_origin, left, right, target + large_origin, 0.1,
    )
    assert shifted_lower <= sampled


@pytest.mark.parametrize("seed", range(6))
def test_curved_target_chord_bound_is_below_independent_dense_minimum(seed):
    rng = np.random.default_rng(seed)
    source = np.array(((0., 0., 0.),
                       (0.5, 0.015*rng.normal(), 0.01*rng.normal()),
                       (1., 0., 0.)))
    target = np.array(((0., 0.1, 0.),
                       (0.5, 0.1 + 0.025*rng.normal(),
                        0.025*rng.normal()),
                       (1., 0.1 + 0.01*rng.normal(), 0.)))
    left = 0.1 + 0.5*rng.random()
    right = left + 0.05 + 0.2*rng.random()
    lower = _source_interval_target_gap_lower(source, left, right, target, 0.1)
    source_points = _sample_curve(source, np.linspace(left, right, 151))
    target_points = _sample_curve(target, np.linspace(0.0, 1.0, 2001))
    sampled = min(
        float(np.min(np.linalg.norm(target_points - point, axis=1))) - 0.1
        for point in source_points
    )
    assert lower <= sampled + 1.0e-12
