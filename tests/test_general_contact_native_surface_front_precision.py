"""Independent formula coverage for certified final-segment strip accounting.

The prefix replay, contact-side turning certificate and final-segment work
bound are checked against independently restated plane formulas and against
genuine replay on the saved second-load path. These are developmental
integral enclosures; no point history is admitted by these tests.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.general_contact.native_surface_front import (axial_contact_front,
    stationary_front_angular_cover, prepare_stationary_band, stationary_front_partition_prepared)
from anyfesolver.general_contact.native_surface_front_boundary import stationary_endpoint_angular_partition
from anyfesolver.general_contact.native_surface_friction_integral import _final_segment_depth
from anyfesolver.general_contact.native_surface_front_precision import (PrefixReplayField,
    contact_side_turning, final_segment_strip_work_bound)
from anyfesolver.general_contact.native_surface_front_work import _segment_motion_bounds
from anyfesolver.general_contact.native_surface_history import (NativeSurfaceHistoryField,
    NativeSurfaceMotionSegment, NativeSurfaceReplayBudget)
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
from anyfesolver.general_contact.native_surface_support import SupportIntervalBudget
from test_general_contact_native_surface_normal import _settings as normal_settings
from test_general_contact_native_surface_patch import _patch
from test_general_contact_native_surface_friction_integral import _settings as friction_settings

CELL = 1
BAND = (3.1423828187051774, 3.32016756626797)
SEGMENT = 1


def _saved_field():
    folder = Path(__file__).parents[1] / 'docs/d1_review_support/contact_c5_native_surface_point_20261002'
    saved = json.loads((folder / 'replay_owner_second_load_rejected_path.json').read_text())
    _, bridge = _patch(activation_policy='transverse_event_sequence', gap_check='straight_patch',
        normal_integration='straight_segmented', normal_integration_settings=normal_settings(),
        friction_integration='replay', friction_integration_settings=friction_settings())
    return NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k: np.asarray(v) for k, v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k: np.asarray(v) for k, v in s.items()})
            for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])


def test_contact_side_turning_requires_strict_matching_signs():
    # H=-A/B has H''=-N'/B^2 at a root; the contact-side extreme needs
    # B and N' strictly same-signed in the prepared bounds.
    assert contact_side_turning((1., 2., -1., 1., .5, 1.))
    assert contact_side_turning((-2., -1., -1., 1., -1., -.5))
    assert not contact_side_turning((-1., 2., -1., 1., -1., 1.))
    assert not contact_side_turning((1., 2., -1., 1., -1., .5))
    assert not contact_side_turning((1., 2., -1., 1., -1., -2.))
    assert not contact_side_turning((-2., -1., -1., 1., 1., 2.))


def test_saved_final_segment_turning_certificate_and_strip_bound():
    field = _saved_field()
    rule = field.patch.rule
    domain = (max(rule.definition['xi_range'][0], CELL - 1), min(rule.definition['xi_range'][1], CELL))
    budget = SupportIntervalBudget(128)
    partition = stationary_endpoint_angular_partition(field, CELL, rule.definition['angle_range'],
        budget=budget)
    cover = stationary_front_angular_cover(field, CELL, SEGMENT, BAND, budget=budget,
        xi_bounds=domain, endpoint_partition=partition, _variable_endpoint_branches=True,
        _allow_angular_subdivision=True)
    assert len(cover) == 1
    prepared = prepare_stationary_band(field, cover, budget=budget)
    assert len(prepared.branches) == 1
    bounds = prepared.branches[0][2]
    assert contact_side_turning(bounds)
    # Independent plane formula: sample B and N' over the certified bracket.
    left, right = prepared.branches[0][:2]
    for time in np.linspace(left, right, 7):
        for angle in np.linspace(*BAND, 5):
            value = axial_contact_front(field, CELL, SEGMENT, float(time), float(angle))
            assert bounds[0] <= value.b <= bounds[1] and value.b < 0
            assert bounds[4] <= value.stationary_rate <= bounds[5] and value.stationary_rate < 0
    middle = sum(BAND) / 2
    whole = stationary_front_partition_prepared(field, prepared, float(middle),
        maximum_axial_uncertainty=1e-7)
    roots, endpoints = whole[2], whole[3]
    assert len(roots) == 1
    strip = roots[0].axial_interval
    assert strip[0] <= roots[0].axial_coordinate <= strip[1]
    assert strip[1] - strip[0] <= 1e-7
    depth = _final_segment_depth(prepared, whole, strip[0], strip[1])
    # The certified per-angle depth is the strip width up to the endpoint
    # coordinate padding, orders below the angular-envelope whole-history depth.
    assert 0 < depth < 4e-7
    bound = final_segment_strip_work_bound(field, CELL, depth, SEGMENT)
    assert bound[1] < 1e-3
    # Genuine replay inside the strip lies inside the certified bound.
    value = field.query(float(sum(strip) / 2), float(middle), budget=NativeSurfaceReplayBudget(64))
    point = value.last_contribution
    stored = 0. if point is None else float(point.trial.stored_energy_density)
    dissipated = 0. if point is None else float(point.trial.dissipated_increment_density)
    released = 0. if point is None else float(point.trial.released_storage_density)
    cumulative = float(value.history.cumulative_dissipation_density)
    assert 0 <= stored <= bound[0]
    assert 0 <= dissipated <= bound[1]
    assert 0 <= released <= 2 * bound[2]
    # Prefix boundary enclosure plus the final-segment increment encloses it.
    prefix = PrefixReplayField(field)
    edges = [prefix.boundary_state(float(xi), float(middle), NativeSurfaceReplayBudget(64))
        for xi in strip]
    assert min(c for _, c in edges) <= cumulative <= max(c for _, c in edges) + bound[3]


def test_final_segment_bound_matches_independent_cap_formula():
    field = _saved_field()
    pair = field.patch.pairs[0]
    slip, chord, padding = _segment_motion_bounds(field, CELL, SEGMENT)
    width = 3.7e-9
    bound = final_segment_strip_work_bound(field, CELL, width, SEGMENT)
    cap = pair.friction_coefficient * pair.normal_stiffness_Pa_per_m * (chord * width + padding)
    kt = pair.tangential_stiffness_Pa_per_m
    assert bound[0] == pytest.approx(cap * cap / (2 * kt), rel=1e-12)
    assert bound[1] == pytest.approx(cap * (slip + 2 * pair.event_history_substeps * cap / kt), rel=1e-12)
    assert bound[2] == bound[0] and bound[3] == bound[1]
    with pytest.raises(ValueError, match='valid explicit segment index'):
        final_segment_strip_work_bound(field, CELL, width, -1)
    with pytest.raises(ValueError, match='valid explicit segment index'):
        final_segment_strip_work_bound(field, CELL, width, len(field.segments))
    with pytest.raises(ValueError, match='positive certified strip width'):
        final_segment_strip_work_bound(field, CELL, 0., SEGMENT)
    with pytest.raises(ValueError, match='positive certified strip width'):
        final_segment_strip_work_bound(field, CELL, np.inf, SEGMENT)


def test_prefix_boundary_state_matches_full_field_for_quiet_final_segment():
    field = _saved_field()
    zero = NativeSurfaceMotionSegment(np.zeros((3, 3)), np.zeros((2, 3)), np.zeros(18))
    full = NativeSurfaceHistoryField(field.patch, field.initial, (*field.segments, zero),
        maximum_intervals=field.maximum_intervals)
    prefix = PrefixReplayField(full)
    with pytest.raises(TypeError, match='exact material history field'):
        PrefixReplayField(object())
    with pytest.raises(TypeError, match='explicit shared material replay budget'):
        prefix.boundary_state(.1, 3.199, object())
    budget = NativeSurfaceReplayBudget(256)
    stored, cumulative = prefix.boundary_state(.75, 3.1990185900472157, budget)
    value = full.query(.75, 3.1990185900472157, budget=budget)
    point = value.last_contribution
    assert point is not None
    assert stored == pytest.approx(float(point.trial.stored_energy_density), rel=1e-12, abs=1e-15)
    assert cumulative == pytest.approx(float(value.history.cumulative_dissipation_density),
        rel=1e-12, abs=1e-15)
    # The point is in contact at the shared final pose (pure stick here, so the
    # cumulative dissipation is legitimately zero while stored is positive).
    assert stored > 0


def test_quiet_final_segment_preserves_nonzero_prefix_dissipation():
    from test_general_contact_native_surface_friction_integral import _field
    from test_general_contact_native_surface_history import _segment
    base = _field()
    zero = NativeSurfaceMotionSegment(np.zeros((3, 3)), np.zeros((2, 3)), np.zeros(18))
    full = NativeSurfaceHistoryField(base.patch, base.initial,
        (_segment(.01, -.00201), _segment(0., .004), zero),
        maximum_intervals=base.maximum_intervals)
    prefix = PrefixReplayField(full)
    budget = NativeSurfaceReplayBudget(256)
    stored, cumulative = prefix.boundary_state(.5, np.pi, budget)
    value = full.query(.5, np.pi, budget=budget)
    assert stored == 0.
    assert cumulative > 0.
    assert cumulative == pytest.approx(value.history.cumulative_dissipation_density,
                                       rel=1e-12, abs=1e-15)


def test_material_node_rounding_keeps_tiny_interval_and_refuses_no_interior():
    from anyfesolver.general_contact.native_surface_friction_integral import _interior_material_coordinate
    left,right=.7999999999999997,.8
    for node in np.polynomial.legendre.leggauss(5)[0]:
        actual=_interior_material_coordinate(left,right,node,allow_rounding=True)
        assert left<actual<right
        ideal=(left+right)/2+(right-left)*float(node)/2
        assert abs(actual-ideal)<=np.spacing(right)
    adjacent=float(np.nextafter(right,-np.inf))
    assert not adjacent<_interior_material_coordinate(adjacent,right,0.,allow_rounding=True)<right
    assert _interior_material_coordinate(.2,.8,.3,allow_rounding=True)==_interior_material_coordinate(.2,.8,.3)
