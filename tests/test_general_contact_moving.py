"""Independent scalar controls for translating planar mortar patches."""

import numpy as np
import pytest

from anyfesolver.general_contact.moving import (
    TranslatedPairMortar,
    moving_mortar_state,
    moving_mortar_translation_residual,
    moving_mortar_reduced_system,
    certify_moving_mortar_gap,
    bound_swept_moving_mortar_gap,
)
from anyfesolver.general_contact.overlap import PlanarFace


def _square(feature, x0, y0, z=0.0):
    return PlanarFace(feature, np.asarray((
        (x0, y0, z), (x0 + 1.0, y0, z),
        (x0 + 1.0, y0 + 1.0, z), (x0, y0 + 1.0, z),
    )))


def test_translating_pair_potential_force_tangent_and_common_motion():
    law = TranslatedPairMortar(
        (_square((1, 1), 0.0, 0.0),),
        (_square((2, 1), 0.5, 0.25),),
        np.array((0.0, 0.0, 1.0)), 1.0e6,
    )
    source = np.array((0.0, 0.0, -1.0e-4))
    target = np.zeros(3)
    result = law.evaluate(source, target)
    assert result.overlap_area == pytest.approx(0.375)
    assert result.gap == pytest.approx(-1.0e-4)
    assert result.energy == pytest.approx(0.001875)
    np.testing.assert_allclose(result.source_residual,
                               (0.00375, 0.0025, -37.5), atol=1e-10)
    np.testing.assert_allclose(result.source_residual + result.target_residual,
                               0.0, atol=1e-12)
    np.testing.assert_allclose(result.relative_tangent,
                               result.relative_tangent.T, atol=1e-10)
    perturbation = np.array((0.3, -0.2, 0.4))
    h = 1.0e-6
    plus = law.evaluate(source + h * perturbation, target)
    minus = law.evaluate(source - h * perturbation, target)
    assert ((plus.energy - minus.energy) / (2.0 * h)
            == pytest.approx(float(result.source_residual @ perturbation), rel=1e-8))
    np.testing.assert_allclose(
        (plus.source_residual - minus.source_residual) / (2.0 * h),
        result.relative_tangent @ perturbation,
        rtol=1e-7, atol=1e-7,
    )
    common = np.array((0.2, -0.1, 0.3))
    translated = law.evaluate(source + common, target + common)
    assert translated.energy == pytest.approx(result.energy)
    np.testing.assert_allclose(translated.source_residual,
                               result.source_residual, atol=1e-9)
    np.testing.assert_allclose(result.tangent @ np.r_[common, common],
                               0.0, atol=1e-9)


def test_translating_pair_opens_loses_overlap_and_recontacts():
    law = TranslatedPairMortar(
        (_square((1, 1), 0.0, 0.0),),
        (_square((2, 1), 0.5, 0.25),),
        np.array((0.0, 0.0, 1.0)), 1.0e6,
    )
    reference = law.evaluate(np.array((0.0, 0.0, -1.0e-4)), np.zeros(3))
    open_pose = law.evaluate(np.array((0.0, 0.0, 1.0e-4)), np.zeros(3))
    assert open_pose.overlap_area == pytest.approx(reference.overlap_area)
    assert open_pose.energy == 0.0
    np.testing.assert_array_equal(open_pose.source_residual, np.zeros(3))
    distant = law.evaluate(np.array((-1.1, 0.0, -1.0e-4)), np.zeros(3))
    assert distant.overlap_area == 0.0
    assert distant.energy == 0.0
    recontact = law.evaluate(np.array((0.2, 0.0, -1.0e-4)), np.zeros(3))
    assert recontact.overlap_area > 0.0
    assert recontact.energy > 0.0


def test_translated_nonmatching_patch_and_candidate_budget():
    source = (_square((1, 1), 0.0, 0.0),)
    targets = (_square((2, 1), 0.0, 0.0),
               _square((2, 2), 1.0, 0.0))
    law = TranslatedPairMortar(source, targets,
                               np.array((0.0, 0.0, 1.0)), 1.0e6)
    result = law.evaluate(np.array((0.5, 0.0, -1.0e-6)), np.zeros(3))
    assert result.overlap_area == pytest.approx(1.0)
    assert result.energy == pytest.approx(0.5e-6)
    with pytest.raises(MemoryError, match="candidate limit"):
        TranslatedPairMortar(source, targets,
                             np.array((0.0, 0.0, 1.0)), 1.0e6,
                             maximum_candidate_pairs=1)


def test_moving_q1_pressure_reproduces_constant_gap_and_both_body_work():
    source = _square((1, 1), 0.0, 0.0)
    target = _square((2, 1), 0.5, 0.25)
    zero = np.zeros(3)
    source_height = np.full(4, -1.0e-4)
    target_height = np.zeros(4)
    state = moving_mortar_state(
        source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
        zero, zero, source_height, target_height,
    )
    assert state.overlap_area == pytest.approx(0.375)
    assert np.sum(state.support) == pytest.approx(state.overlap_area)
    np.testing.assert_allclose(state.normalized_gap, -1.0e-4, atol=1e-12)
    assert state.energy == pytest.approx(0.001875)
    assert np.sum(state.normal_residual[:4]) == pytest.approx(-37.5)
    assert np.sum(state.normal_residual[4:]) == pytest.approx(37.5)
    assert state.minimum_vertex_gap == pytest.approx(-1.0e-4)
    direction = np.array((0.1, -0.2, 0.3, 0.05,
                          -0.1, 0.2, -0.05, 0.15))
    h = 1.0e-6
    def trial(fraction):
        return moving_mortar_state(
            source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
            zero, zero,
            source_height + fraction * direction[:4],
            target_height + fraction * direction[4:],
        )
    plus, minus = trial(h), trial(-h)
    assert ((plus.energy - minus.energy) / (2.0 * h)
            == pytest.approx(float(state.normal_residual @ direction), rel=1e-8))
    np.testing.assert_allclose(
        (plus.normal_residual - minus.normal_residual) / (2.0 * h),
        state.normal_tangent @ direction,
        rtol=1e-7, atol=1e-7,
    )


def test_moving_p1_q1_partial_gap_uses_independent_physical_vertex_check():
    source = PlanarFace((1, 1), np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ))
    target = _square((2, 1), 0.0, 0.0)
    zero = np.zeros(3)
    state = moving_mortar_state(
        source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
        np.array((0.2, 0.0, 0.0)), zero,
        np.array((-1.0e-4, 0.0, 0.0)), np.zeros(4),
    )
    assert state.overlap_area > 0.0
    assert state.minimum_vertex_gap < 0.0
    assert state.energy > 0.0
    assert np.all(state.support > 0.0)
    assert np.sum(state.normal_residual[:3]) == pytest.approx(
        -np.sum(state.normal_residual[3:])
    )


def test_moving_pressure_energy_matches_translation_boundary_control():
    source = _square((1, 1), 0.0, 0.0)
    target = _square((2, 1), 0.5, 0.25)
    control = TranslatedPairMortar((source,), (target,),
                                   np.array((0.0, 0.0, 1.0)), 1.0e6)
    direction = np.array((0.3, -0.2, 0.0))
    def energy(shift):
        return moving_mortar_state(
            source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
            shift * direction, np.zeros(3),
            np.full(4, -1.0e-4), np.zeros(4),
        ).energy
    h = 1.0e-6
    reference = control.evaluate(np.array((0.0, 0.0, -1.0e-4)), np.zeros(3))
    assert energy(0.0) == pytest.approx(reference.energy)
    assert ((energy(h) - energy(-h)) / (2.0 * h)
            == pytest.approx(float(reference.source_residual @ direction), rel=1e-8))
    source_residual, target_residual = moving_mortar_translation_residual(
        source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
        np.zeros(3), np.zeros(3),
        np.full(4, -1.0e-4), np.zeros(4),
    )
    np.testing.assert_allclose(source_residual, reference.source_residual,
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(target_residual, -source_residual, atol=1e-9)


def test_moving_nonuniform_gap_transport_matches_energy_change():
    source = _square((1, 1), 0.0, 0.0)
    target = _square((2, 1), 0.5, 0.25)
    source_heights = np.array((-2.0e-4, -1.0e-4, -1.5e-4, -2.5e-4))
    target_heights = np.array((0.0, 0.25e-4, -0.1e-4, 0.15e-4))
    zero = np.zeros(3)
    source_residual, target_residual = moving_mortar_translation_residual(
        source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
        zero, zero, source_heights, target_heights,
    )
    np.testing.assert_allclose(source_residual + target_residual,
                               0.0, atol=1e-7)
    direction = np.array((0.3, -0.2, 0.0))
    h = 1.0e-6
    def energy(shift_source, shift_target):
        return moving_mortar_state(
            source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
            shift_source, shift_target, source_heights, target_heights,
        ).energy
    source_difference = (energy(h * direction, zero)
                         - energy(-h * direction, zero)) / (2.0 * h)
    target_difference = (energy(zero, h * direction)
                         - energy(zero, -h * direction)) / (2.0 * h)
    assert source_difference == pytest.approx(float(source_residual @ direction),
                                              rel=1e-5)
    assert target_difference == pytest.approx(float(target_residual @ direction),
                                              rel=1e-5)


def test_moving_translation_normal_cross_derivatives_are_consistent():
    source = _square((1, 1), 0.0, 0.0)
    target = _square((2, 1), 0.5, 0.25)
    source_heights = np.array((-2.0e-4, -1.0e-4, -1.5e-4, -2.5e-4))
    target_heights = np.array((0.0, 0.25e-4, -0.1e-4, 0.15e-4))
    zero = np.zeros(3)
    unit_x = np.array((1.0, 0.0, 0.0))
    unit_height = np.array((1.0, 0.0, 0.0, 0.0))
    h = 1.0e-7
    force_plus = moving_mortar_translation_residual(
        source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
        zero, zero, source_heights + h * unit_height, target_heights,
    )[0][0]
    force_minus = moving_mortar_translation_residual(
        source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
        zero, zero, source_heights - h * unit_height, target_heights,
    )[0][0]
    normal_plus = moving_mortar_state(
        source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
        h * unit_x, zero, source_heights, target_heights,
    ).normal_residual[0]
    normal_minus = moving_mortar_state(
        source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
        -h * unit_x, zero, source_heights, target_heights,
    ).normal_residual[0]
    assert ((force_plus - force_minus) / (2.0 * h)
            == pytest.approx((normal_plus - normal_minus) / (2.0 * h), rel=1e-5))


def test_reduced_moving_tangent_couples_sliding_and_nodal_normal_fields():
    source = _square((1, 1), 0.0, 0.0)
    target = _square((2, 1), 0.5, 0.25)
    heights_a = np.array((-2.0e-4, -1.0e-4, -1.5e-4, -2.5e-4))
    heights_b = np.array((0.0, 0.25e-4, -0.1e-4, 0.15e-4))
    zero = np.zeros(3)
    normal = np.array((0.0, 0.0, 1.0))
    system = moving_mortar_reduced_system(
        source, target, normal, 1.0e6,
        zero, zero, heights_a, heights_b,
        difference_step=1.0e-7,
    )
    assert system.tangent.shape == (14, 14)
    assert system.relative_tangent_asymmetry < 1.0e-5
    finer = moving_mortar_reduced_system(
        source, target, normal, 1.0e6,
        zero, zero, heights_a, heights_b,
        difference_step=2.0e-8,
    )
    np.testing.assert_allclose(finer.tangent, system.tangent,
                               rtol=1.0e-5, atol=1.0e-4)
    direction = np.array((0.3, -0.2, 0.1, -0.15, 0.2, -0.05,
                          0.1, -0.1, 0.2, -0.2, -0.1, 0.05, -0.15, 0.1))
    h = 2.0e-8
    def residual(fraction):
        variables = np.r_[zero, zero, heights_a, heights_b] + fraction * direction
        ta, tb = variables[:3], variables[3:6]
        ha, hb = variables[6:10], variables[10:]
        state = moving_mortar_state(source, target, normal, 1.0e6,
                                    ta, tb, ha, hb)
        fa, fb = moving_mortar_translation_residual(
            source, target, normal, 1.0e6, ta, tb, ha, hb,
        )
        return np.r_[fa, fb, state.normal_residual]
    np.testing.assert_allclose(
        (residual(h) - residual(-h)) / (2.0 * h),
        system.tangent @ direction, rtol=1.0e-5, atol=1.0e-5,
    )


def test_moving_gap_certificate_distinguishes_pass_violation_and_budget():
    source = _square((1, 1), 0.0, 0.0)
    target = _square((2, 1), 0.5, 0.25)
    zero = np.zeros(3)
    normal = np.array((0.0, 0.0, 1.0))
    accepted = certify_moving_mortar_gap(
        source, target, normal, zero, zero,
        np.full(4, -4.0e-7), np.zeros(4),
        gap_tolerance=1.0e-6, maximum_triangles=16,
    )
    assert accepted.tolerance_met
    assert accepted.minimum_sampled_gap == pytest.approx(-4.0e-7)
    assert accepted.conservative_lower_gap >= -1.0e-6
    violated = certify_moving_mortar_gap(
        source, target, normal, zero, zero,
        np.full(4, -4.0e-6), np.zeros(4),
        gap_tolerance=1.0e-6, maximum_triangles=16,
    )
    assert not violated.tolerance_met
    assert violated.minimum_sampled_gap < -1.0e-6
    nonaffine_source = PlanarFace((1, 1), np.array((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.9, 1.0, 0.0), (0.0, 1.0, 0.0),
    )))
    exhausted = certify_moving_mortar_gap(
        nonaffine_source, target, normal, zero, zero,
        np.full(4, -4.0e-7), np.zeros(4),
        gap_tolerance=1.0e-6, maximum_triangles=1,
    )
    assert not exhausted.tolerance_met
    assert exhausted.conservative_lower_gap == float("-inf")
    assert exhausted.method == "subdivision"


def test_moving_gap_certificate_bounds_independent_bilinear_samples():
    source = _square((1, 1), 0.0, 0.0)
    target = _square((2, 1), 0.5, 0.25)
    source_heights = np.array((-0.4, -0.9, -0.2, -0.7)) * 1.0e-4
    target_heights = np.array((0.1, -0.2, 0.3, 0.0)) * 1.0e-4
    result = certify_moving_mortar_gap(
        source, target, np.array((0.0, 0.0, 1.0)),
        np.zeros(3), np.zeros(3), source_heights, target_heights,
        gap_tolerance=1.0e-3, maximum_triangles=64,
    )

    def bilinear(values, x, y):
        return ((1.0 - x) * (1.0 - y) * values[0]
                + x * (1.0 - y) * values[1]
                + x * y * values[2]
                + (1.0 - x) * y * values[3])

    dense_minimum = min(
        bilinear(source_heights, x, y)
        - bilinear(target_heights, x - 0.5, y - 0.25)
        for x in np.linspace(0.5, 1.0, 51)
        for y in np.linspace(0.25, 1.0, 51)
    )
    assert result.tolerance_met
    assert result.conservative_lower_gap <= dense_minimum + 1.0e-12
    assert result.minimum_sampled_gap >= dense_minimum - 1.0e-12


def test_quadratic_gap_certificate_finds_unsampled_edge_minimum_and_is_objective():
    source = _square((1, 1), 0.0, 0.0)
    target = PlanarFace((2, 1), np.array((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    )))
    normal = np.array((0.0, 0.0, 1.0))
    heights = np.array((0.0, 0.0, -1.0, 0.0))
    zero = np.zeros(3)
    result = certify_moving_mortar_gap(
        source, target, normal, zero, zero, heights, np.zeros(3),
        gap_tolerance=0.2, maximum_triangles=1,
    )
    assert result.method == "quadratic_extrema"
    assert result.triangles_checked == 0
    assert result.minimum_sampled_gap == pytest.approx(0.0)
    assert result.conservative_lower_gap == pytest.approx(-0.25, abs=1.0e-12)
    assert not result.tolerance_met

    angle = 0.37
    rotation = np.array((
        (np.cos(angle), 0.0, np.sin(angle)),
        (0.0, 1.0, 0.0),
        (-np.sin(angle), 0.0, np.cos(angle)),
    ))
    shift = np.array((0.3, -0.4, 0.2))
    transformed = certify_moving_mortar_gap(
        PlanarFace(source.feature_id, source.corners @ rotation.T + shift),
        PlanarFace(target.feature_id, target.corners @ rotation.T + shift),
        rotation @ normal, zero, zero, heights, np.zeros(3),
        gap_tolerance=0.2, maximum_triangles=1,
    )
    assert transformed.method == "quadratic_extrema"
    assert transformed.conservative_lower_gap == pytest.approx(
        result.conservative_lower_gap, abs=1.0e-12,
    )


def test_weighted_mortar_gap_does_not_certify_partial_contact_pointwise_gap():
    source = _square((1, 2), 0.15, 0.1)
    target = PlanarFace((2, 1), np.array((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    )))
    source_heights = np.array((
        3.647970327683101e-5, -3.216029592989825e-5,
        -6.336029579114779e-5, -3.216029592989776e-5,
    ))
    zero = np.zeros(3)
    state = moving_mortar_state(
        source, target, np.array((0.0, 0.0, 1.0)), 1.6e7,
        zero, zero, source_heights, np.zeros(3),
    )
    assert np.count_nonzero(state.normalized_gap < 0.0) == 1
    certificate = certify_moving_mortar_gap(
        source, target, np.array((0.0, 0.0, 1.0)),
        zero, zero, source_heights, np.zeros(3),
        gap_tolerance=1.0e-5, maximum_triangles=1,
    )
    assert certificate.method == "quadratic_extrema"
    assert certificate.conservative_lower_gap == pytest.approx(-1.5e-5, abs=1.0e-9)
    assert not certificate.tolerance_met


def test_quadratic_gap_certificate_includes_planar_geometric_height():
    source = PlanarFace((1, 1), np.array((
        (0.0, 0.0, 0.0), (1.0, 0.0, -1.0e-6),
        (1.0, 1.0, -1.0e-6), (0.0, 1.0, 0.0),
    )))
    target = _square((2, 1), 0.0, 0.0)
    result = certify_moving_mortar_gap(
        source, target, np.array((0.0, 0.0, 1.0)),
        np.zeros(3), np.zeros(3), np.zeros(4), np.zeros(4),
        gap_tolerance=5.0e-7, maximum_triangles=100,
    )
    assert result.conservative_lower_gap == pytest.approx(-1.0e-6, abs=1.0e-10)
    assert not result.tolerance_met


def test_swept_gap_bound_detects_clear_to_clear_crossing():
    source = _square((1, 1), 0.0, 0.0)
    target = _square((2, 1), 0.0, 0.0)
    source_start = np.array((-2.0, 0.0, -1.0e-4))
    source_end = np.array((2.0, 0.0, -1.0e-4))
    zero = np.zeros(3)
    heights = np.zeros(4)
    crossing = bound_swept_moving_mortar_gap(
        source, target, np.array((0.0, 0.0, 1.0)),
        source_start, source_end, zero, zero,
        heights, heights, heights, heights,
        gap_tolerance=1.0e-6,
    )
    assert crossing.overlap_interval == pytest.approx((0.25, 0.75))
    assert crossing.conservative_lower_gap == pytest.approx(-1.0e-4)
    assert not crossing.tolerance_certified

    separated = bound_swept_moving_mortar_gap(
        source, target, np.array((0.0, 0.0, 1.0)),
        source_start + np.array((0.0, 0.0, 2.0e-4)),
        source_end + np.array((0.0, 0.0, 2.0e-4)), zero, zero,
        heights, heights, heights, heights,
        gap_tolerance=1.0e-6,
    )
    assert separated.tolerance_certified
    assert separated.conservative_lower_gap == pytest.approx(1.0e-4)
