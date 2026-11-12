"""Global-pressure controls for nonmatching moving planar patch overlap."""

import numpy as np
import pytest

from anyfesolver.general_contact.moving import (
    moving_mortar_state, moving_mortar_translation_residual,
)
from anyfesolver.general_contact.overlap import PlanarFace
from anyfesolver.general_contact.patch_mortar import MovingPatchMortar, PatchFace


def _quad(feature, x0, x1, y0=0.0, y1=1.0):
    return PlanarFace(feature, np.array((
        (x0, y0, 0.0), (x1, y0, 0.0),
        (x1, y1, 0.0), (x0, y1, 0.0),
    )))


def _patches():
    source = PatchFace(_quad((1, 1), 0.0, 1.0), (1, 2, 3, 4))
    whole = PatchFace(_quad((2, 1), 0.0, 1.0), (10, 12, 15, 13))
    left = PatchFace(_quad((2, 2), 0.0, 0.5), (10, 11, 14, 13))
    right = PatchFace(_quad((2, 3), 0.5, 1.0), (11, 12, 15, 14))
    return source, whole, left, right


def test_split_target_shares_one_source_pressure_space():
    source, whole, left, right = _patches()
    normal = np.array((0.0, 0.0, 1.0))
    single = MovingPatchMortar((source,), (whole,), normal, 1.0e6)
    split = MovingPatchMortar((source,), (left, right), normal, 1.0e6)
    source_heights = np.array((-1.0e-4, -1.2e-4, -0.8e-4, -1.1e-4))
    whole_heights = np.array((0.0, 1.0e-5, -5.0e-6, 5.0e-6))
    split_heights = np.array((0.0, 5.0e-6, 1.0e-5,
                              -5.0e-6, 0.0, 5.0e-6))
    zero = np.zeros(3)
    whole_state = single.evaluate(zero, zero, source_heights, whole_heights)
    split_state = split.evaluate(zero, zero, source_heights, split_heights)
    reversed_state = MovingPatchMortar(
        (source,), (right, left), normal, 1.0e6,
    ).evaluate(zero, zero, source_heights, split_heights)
    assert whole_state.overlap_area == pytest.approx(1.0)
    assert split_state.overlap_area == pytest.approx(1.0)
    assert split_state.overlapping_pairs == 2
    assert split_state.source_node_ids == (1, 2, 3, 4)
    assert whole_state.target_node_ids == (10, 12, 13, 15)
    assert split_state.pressure_rank == 4
    np.testing.assert_allclose(split_state.support, whole_state.support,
                               rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(split_state.weighted_gap, whole_state.weighted_gap,
                               rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(split_state.normalized_gap, whole_state.normalized_gap,
                               rtol=1.0e-11, atol=1.0e-12)
    assert split_state.energy == pytest.approx(whole_state.energy, rel=1.0e-11)
    np.testing.assert_allclose(split_state.normal_residual[:4],
                               whole_state.normal_residual[:4],
                               rtol=1.0e-10, atol=1.0e-9)
    np.testing.assert_allclose(split_state.support, reversed_state.support)
    np.testing.assert_allclose(split_state.normal_residual,
                               reversed_state.normal_residual, atol=1.0e-9)
    assert sum(split_state.normal_residual) == pytest.approx(0.0, abs=1.0e-9)

    local = moving_mortar_state(
        source.geometry, whole.geometry, normal, 1.0e6,
        zero, zero, source_heights, whole_heights[[0, 1, 3, 2]],
    )
    global_order = np.array((0, 1, 2, 3, 4, 5, 7, 6))
    assert whole_state.energy == pytest.approx(local.energy)
    np.testing.assert_allclose(whole_state.normal_residual,
                               local.normal_residual[global_order])
    np.testing.assert_allclose(whole_state.normal_tangent,
                               local.normal_tangent[np.ix_(global_order, global_order)])


def test_patch_normal_work_tangent_gap_and_resource_limits():
    source, _whole, left, right = _patches()
    patch = MovingPatchMortar(
        (source,), (left, right), np.array((0.0, 0.0, 1.0)), 1.0e6,
    )
    source_heights = np.full(4, -1.0e-4)
    target_heights = np.zeros(6)
    zero = np.zeros(3)
    state = patch.evaluate(zero, zero, source_heights, target_heights)
    direction = np.array((0.2, -0.1, 0.3, -0.2,
                          0.1, -0.25, 0.15, -0.1, 0.2, -0.05))
    h = 1.0e-7

    def trial(fraction):
        variables = np.r_[source_heights, target_heights] + fraction * direction
        return patch.evaluate(zero, zero, variables[:4], variables[4:])

    plus, minus = trial(h), trial(-h)
    assert (plus.energy - minus.energy) / (2.0 * h) == pytest.approx(
        float(state.normal_residual @ direction), rel=1.0e-6,
    )
    np.testing.assert_allclose(
        (plus.normal_residual - minus.normal_residual) / (2.0 * h),
        state.normal_tangent @ direction, rtol=1.0e-6, atol=1.0e-7,
    )
    certificate = patch.certify_gap(
        zero, zero, source_heights, target_heights,
        gap_tolerance=2.0e-4, maximum_triangles_per_pair=1,
    )
    assert certificate.tolerance_met
    assert certificate.overlap_pairs_checked == 2
    assert certificate.conservative_lower_gap == pytest.approx(-1.0e-4)
    with pytest.raises(MemoryError, match="candidate limit"):
        MovingPatchMortar(
            (source,), (left, right), np.array((0.0, 0.0, 1.0)), 1.0e6,
            maximum_candidate_pairs=1,
        )
    with pytest.raises(MemoryError, match="normal-DOF limit"):
        MovingPatchMortar(
            (source,), (left, right), np.array((0.0, 0.0, 1.0)), 1.0e6,
            maximum_normal_dofs=9,
        )


def test_local_mortar_lateral_forces_preserve_translation_work():
    source = PatchFace(_quad((1, 1), 0.0, 1.0), (1, 2, 3, 4))
    target = PatchFace(_quad((2, 1), -0.2, 0.7, -0.1, 1.2), (5, 6, 7, 8))
    patch = MovingPatchMortar(
        (source,), (target,), np.array((0.0, 0.0, 1.0)), 1.0e6,
    )
    zero = np.zeros(3)
    source_height = np.array((-1.0e-4, -1.2e-4, -0.8e-4, -1.1e-4))
    target_height = np.array((0.0, 1.0e-5, -5.0e-6, 5.0e-6))
    for multiplier in (None, np.array((1.0, 2.0, 3.0, 4.0))):
        state = patch.evaluate(zero, zero, source_height, target_height,
                               multiplier_estimate=multiplier)
        source, target = patch.translation_residual(
            zero, zero, source_height, target_height,
            multiplier_estimate=multiplier,
        )
        source_local, target_local = patch.local_tangential_residual(
            zero, zero, source_height, target_height,
            multiplier_estimate=multiplier,
        )
        np.testing.assert_allclose(
            np.sum(source_local, axis=0),
            source - np.sum(state.normal_residual[:4]) * patch.normal,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            np.sum(target_local, axis=0),
            target - np.sum(state.normal_residual[4:]) * patch.normal,
            atol=1.0e-10,
        )


def test_fixed_augmented_multiplier_work_and_projected_update():
    source, _whole, left, right = _patches()
    patch = MovingPatchMortar(
        (source,), (left, right), np.array((0.0, 0.0, 1.0)), 1.0e6,
    )
    multiplier = np.array((30.0, 40.0, 50.0, 60.0))
    source_heights = np.array((-1.0e-4, -1.1e-4, -0.9e-4, -1.2e-4))
    target_heights = np.zeros(6)
    source_translation = np.array((0.17, 0.05, 0.0))
    zero = np.zeros(3)

    def trial(shift, heights):
        return patch.evaluate(
            source_translation + shift, zero, heights, target_heights,
            multiplier_estimate=multiplier,
        )

    state = trial(zero, source_heights)
    np.testing.assert_allclose(
        patch.projected_multiplier(state, multiplier),
        multiplier - 1.0e6 * state.normalized_gap,
    )
    height_direction = np.array((0.2, -0.1, 0.3, -0.2))
    h = 1.0e-7
    height_work = (
        trial(zero, source_heights + h * height_direction).energy
        - trial(zero, source_heights - h * height_direction).energy
    ) / (2.0 * h)
    assert height_work == pytest.approx(
        float(state.normal_residual[:4] @ height_direction), rel=1.0e-6,
    )
    plus_height = trial(zero, source_heights + h * height_direction)
    minus_height = trial(zero, source_heights - h * height_direction)
    np.testing.assert_allclose(
        (plus_height.normal_residual - minus_height.normal_residual) / (2.0 * h),
        state.normal_tangent[:, :4] @ height_direction,
        rtol=1.0e-6, atol=1.0e-7,
    )
    translation_direction = np.array((0.3, -0.2, 0.0))
    force, _ = patch.translation_residual(
        source_translation, zero, source_heights, target_heights,
        multiplier_estimate=multiplier,
    )
    translation_work = (
        trial(h * translation_direction, source_heights).energy
        - trial(-h * translation_direction, source_heights).energy
    ) / (2.0 * h)
    assert translation_work == pytest.approx(
        float(force @ translation_direction), rel=1.0e-5, abs=1.0e-8,
    )
    opened = trial(zero, np.full(4, 1.0e-3))
    np.testing.assert_array_equal(patch.projected_multiplier(opened, multiplier),
                                  np.zeros(4))


def test_moving_overlap_transfers_support_without_new_pressure_unknowns():
    source, _whole, left, right = _patches()
    patch = MovingPatchMortar(
        (source,), (left, right), np.array((0.0, 0.0, 1.0)), 1.0e6,
    )
    zero = np.zeros(3)
    source_heights = np.full(4, -1.0e-4)
    target_heights = np.zeros(6)
    for offset, expected_pairs in ((0.49, 2), (0.5, 1), (0.51, 1)):
        source_translation = np.array((offset, 0.0, 0.0))
        state = patch.evaluate(
            source_translation, zero, source_heights, target_heights,
        )
        assert state.source_node_ids == (1, 2, 3, 4)
        assert len(state.support) == 4
        assert state.overlapping_pairs == expected_pairs
        assert state.overlap_area == pytest.approx(1.0 - offset)
        np.testing.assert_allclose(state.normalized_gap,
                                   np.full(4, -1.0e-4), atol=1.0e-12)
        assert state.energy == pytest.approx(
            0.5 * 1.0e6 * (1.0 - offset) * 1.0e-8,
        )
        assert sum(state.normal_residual[:4]) == pytest.approx(
            -1.0e6 * (1.0 - offset) * 1.0e-4,
        )


def test_global_patch_transport_matches_energy_and_single_pair():
    source, whole, left, right = _patches()
    normal = np.array((0.0, 0.0, 1.0))
    patch = MovingPatchMortar((source,), (left, right), normal, 1.0e6)
    one_pair = MovingPatchMortar((source,), (whole,), normal, 1.0e6)
    source_heights = np.array((-1.2e-4, -1.0e-4, -0.9e-4, -1.1e-4))
    target_heights = np.array((0.0, 5.0e-6, 1.0e-5,
                               -5.0e-6, 0.0, 5.0e-6))
    source_translation = np.array((0.2, 0.1, 0.0))
    target_translation = np.zeros(3)
    source_force, target_force = patch.translation_residual(
        source_translation, target_translation,
        source_heights, target_heights,
    )
    np.testing.assert_allclose(source_force + target_force, 0.0, atol=1.0e-7)

    h = 1.0e-6
    direction_a = np.array((0.3, -0.2, 0.1))
    direction_b = np.array((-0.1, 0.15, -0.05))

    def energy(source_shift, target_shift):
        return patch.evaluate(
            source_shift, target_shift, source_heights, target_heights,
        ).energy

    source_difference = (
        energy(source_translation + h * direction_a, target_translation)
        - energy(source_translation - h * direction_a, target_translation)
    ) / (2.0 * h)
    target_difference = (
        energy(source_translation, target_translation + h * direction_b)
        - energy(source_translation, target_translation - h * direction_b)
    ) / (2.0 * h)
    assert source_difference == pytest.approx(float(source_force @ direction_a),
                                              rel=1.0e-5, abs=1.0e-8)
    assert target_difference == pytest.approx(float(target_force @ direction_b),
                                              rel=1.0e-5, abs=1.0e-8)

    whole_heights = target_heights[[0, 2, 3, 5]]
    one_source, one_target = one_pair.translation_residual(
        source_translation, target_translation,
        source_heights, whole_heights,
    )
    local_source, local_target = moving_mortar_translation_residual(
        source.geometry, whole.geometry, normal, 1.0e6,
        source_translation, target_translation,
        source_heights, whole_heights[[0, 1, 3, 2]],
    )
    np.testing.assert_allclose(one_source, local_source, atol=1.0e-9)
    np.testing.assert_allclose(one_target, local_target, atol=1.0e-9)
    np.testing.assert_allclose(source_force, one_source, atol=1.0e-7)
    np.testing.assert_allclose(target_force, one_target, atol=1.0e-7)


def test_global_patch_coupled_tangent_matches_independent_direction():
    source, _whole, left, right = _patches()
    patch = MovingPatchMortar(
        (source,), (left, right), np.array((0.0, 0.0, 1.0)), 1.0e6,
    )
    source_translation = np.array((0.2, 0.1, 0.0))
    target_translation = np.zeros(3)
    source_heights = np.array((-1.2e-4, -1.0e-4, -0.9e-4, -1.1e-4))
    target_heights = np.array((0.0, 5.0e-6, 1.0e-5,
                               -5.0e-6, 0.0, 5.0e-6))
    system = patch.reduced_system(
        source_translation, target_translation,
        source_heights, target_heights, difference_step=1.0e-7,
    )
    assert system.tangent.shape == (16, 16)
    assert system.relative_tangent_asymmetry < 1.0e-4
    direction = np.array((
        0.3, -0.2, 0.1, -0.1, 0.15, -0.05,
        0.2, -0.1, 0.3, -0.2,
        0.1, -0.25, 0.15, -0.1, 0.2, -0.05,
    ))
    baseline = np.r_[source_translation, target_translation,
                     source_heights, target_heights]

    def residual(fraction):
        values = baseline + fraction * direction
        state = patch.evaluate(values[:3], values[3:6],
                               values[6:10], values[10:])
        force_a, force_b = patch.translation_residual(
            values[:3], values[3:6], values[6:10], values[10:],
        )
        return np.r_[force_a, force_b, state.normal_residual]

    h = 2.0e-8
    np.testing.assert_allclose(
        (residual(h) - residual(-h)) / (2.0 * h),
        system.tangent @ direction, rtol=1.0e-5, atol=1.0e-4,
    )
