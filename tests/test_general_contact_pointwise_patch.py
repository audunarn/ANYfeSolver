"""Local work checks for the separate pointwise patch penalty control."""

import numpy as np
import pytest

from anyfesolver.general_contact.overlap import PlanarFace
from anyfesolver.general_contact.patch_mortar import MovingPatchMortar, PatchFace
from anyfesolver.general_contact.pointwise_patch import PointwisePatchPenalty


def _face(identity, xy, node_ids):
    return PatchFace(
        PlanarFace(identity, np.c_[np.asarray(xy, dtype=float), np.zeros(len(xy))]),
        node_ids,
    )


def _pair(subdivisions=1, maximum_triangles=1000):
    source = _face((1, 1), ((0, 0), (1, 0), (1, 1), (0, 1)), (1, 2, 3, 4))
    target = _face((2, 1), ((-0.2, -0.1), (0.7, -0.1),
                           (0.7, 1.2), (-0.2, 1.2)), (5, 6, 7, 8))
    return PointwisePatchPenalty(
        (source,), (target,), np.array((0.0, 0.0, 1.0)), 1.0e6,
        integration_subdivisions=subdivisions,
        maximum_integration_triangles=maximum_triangles,
    )


def test_uniform_closure_has_exact_area_force_and_boundary_work():
    patch = _pair()
    zero = np.zeros(3)
    source_height = np.full(4, -1.0e-4)
    target_height = np.zeros(4)
    state = patch.evaluate(zero, zero, source_height, target_height)
    assert state.overlap_area == pytest.approx(0.7)
    assert state.energy == pytest.approx(0.5 * 1.0e6 * 0.7 * 1.0e-8)
    assert np.sum(state.normal_residual[:4]) == pytest.approx(-70.0)
    assert np.sum(state.normal_residual[4:]) == pytest.approx(70.0)
    assert state.source_translation_residual[0] == pytest.approx(
        -0.5 * 1.0e6 * 1.0e-8, abs=1.0e-12,
    )
    np.testing.assert_allclose(
        np.sum(state.source_local_tangent_residual, axis=0),
        state.source_translation_residual
        - np.sum(state.normal_residual[:4]) * patch.normal,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        np.sum(state.target_local_tangent_residual, axis=0),
        state.target_translation_residual
        - np.sum(state.normal_residual[4:]) * patch.normal,
        atol=1.0e-11,
    )
    source_xy = patch.source_faces[0].geometry.corners
    target_xy = patch.target_faces[0].geometry.corners
    local_moment = sum(
        np.cross(point, force)[2]
        for point, force in zip(source_xy, state.source_local_tangent_residual)
    ) + sum(
        np.cross(point, force)[2]
        for point, force in zip(target_xy, state.target_local_tangent_residual)
    )
    assert local_moment == pytest.approx(0.0, abs=1.0e-11)
    np.testing.assert_allclose(
        state.source_translation_residual + state.target_translation_residual,
        np.zeros(3), atol=1.0e-11,
    )
    step = 1.0e-6
    shift = np.array((step, 0.0, 0.0))
    plus = patch.evaluate(shift, zero, source_height, target_height)
    minus = patch.evaluate(-shift, zero, source_height, target_height)
    assert (plus.energy - minus.energy) / (2 * step) == pytest.approx(
        state.source_translation_residual[0], rel=1.0e-8, abs=1.0e-10,
    )


def test_normal_work_and_tangent_on_smooth_active_branch():
    patch = _pair(subdivisions=2)
    zero = np.zeros(3)
    source_height = np.array((-1.0e-4, -1.2e-4, -0.8e-4, -1.1e-4))
    target_height = np.array((0.0, 1.0e-5, -5.0e-6, 5.0e-6))
    variables = np.r_[source_height, target_height]
    direction = np.array((0.2, -0.1, 0.3, -0.2, 0.1, -0.25, 0.15, -0.1))

    def trial(value):
        heights = variables + value * direction
        return patch.evaluate(zero, zero, heights[:4], heights[4:])

    state = trial(0.0)
    step = 1.0e-7
    plus, minus = trial(step), trial(-step)
    assert (plus.energy - minus.energy) / (2 * step) == pytest.approx(
        state.normal_residual @ direction, rel=1.0e-6,
    )
    np.testing.assert_allclose(
        (plus.normal_residual - minus.normal_residual) / (2 * step),
        state.normal_tangent @ direction, rtol=1.0e-6, atol=1.0e-7,
    )
    np.testing.assert_allclose(
        (plus.source_local_tangent_residual - minus.source_local_tangent_residual)
        / (2 * step),
        np.einsum("ijk,k->ij", state.source_local_tangent_height_derivative,
                  direction), rtol=1.0e-5, atol=1.0e-7,
    )
    np.testing.assert_allclose(
        (plus.target_local_tangent_residual - minus.target_local_tangent_residual)
        / (2 * step),
        np.einsum("ijk,k->ij", state.target_local_tangent_height_derivative,
                  direction), rtol=1.0e-5, atol=1.0e-7,
    )
    for axis in (0, 1):
        shift = np.zeros(3)
        shift[axis] = 1.0e-6
        source_plus = patch.evaluate(shift, zero, source_height, target_height)
        source_minus = patch.evaluate(-shift, zero, source_height, target_height)
        target_plus = patch.evaluate(zero, shift, source_height, target_height)
        target_minus = patch.evaluate(zero, -shift, source_height, target_height)
        assert (source_plus.energy - source_minus.energy) / (2 * shift[axis]) == pytest.approx(
            state.source_translation_residual[axis], rel=1.0e-5, abs=1.0e-9,
        )
        assert (target_plus.energy - target_minus.energy) / (2 * shift[axis]) == pytest.approx(
            state.target_translation_residual[axis], rel=1.0e-5, abs=1.0e-9,
        )
    np.testing.assert_allclose(
        state.source_translation_residual + state.target_translation_residual,
        np.zeros(3), atol=1.0e-9,
    )
    mortar = MovingPatchMortar(
        patch.source_faces, patch.target_faces, patch.normal, patch.stiffness,
    ).evaluate(zero, zero, source_height, target_height)
    assert state.energy > mortar.energy
    with pytest.raises(MemoryError, match="triangle limit"):
        _pair(maximum_triangles=1).evaluate(zero, zero, source_height, target_height)


def test_partial_contact_refinement_and_role_exchange():
    zero = np.zeros(3)
    source_height = np.array((-1.0e-4, 2.0e-4, 2.0e-4, -1.0e-4))
    target_height = np.zeros(4)
    states = [
        _pair(subdivisions=depth).evaluate(zero, zero, source_height, target_height)
        for depth in (1, 2, 3)
    ]
    assert 0.0 < states[-1].active_quadrature_area < states[-1].overlap_area
    assert abs(states[2].energy - states[1].energy) < abs(states[1].energy - states[0].energy)
    source, target = _pair().source_faces, _pair().target_faces
    reversed_patch = PointwisePatchPenalty(
        target, source, np.array((0.0, 0.0, -1.0)), 1.0e6,
        integration_subdivisions=3, maximum_integration_triangles=1000,
    )
    reversed_state = reversed_patch.evaluate(zero, zero, -target_height, -source_height)
    assert reversed_state.energy == pytest.approx(states[-1].energy, rel=1.0e-11)
    np.testing.assert_allclose(
        reversed_state.normal_residual,
        -np.r_[states[-1].normal_residual[4:], states[-1].normal_residual[:4]],
        atol=1.0e-10,
    )


def test_reduced_tangent_matches_mixed_directional_residual():
    patch = _pair(subdivisions=1)
    zero = np.zeros(3)
    source_height = np.array((-1.0e-4, -1.2e-4, -0.8e-4, -1.1e-4))
    target_height = np.zeros(4)
    reduced = patch.reduced_system(
        zero, zero, source_height, target_height,
        difference_step=1.0e-7,
    )
    direction = np.zeros(14)
    direction[0] = 0.3
    direction[4] = -0.4
    direction[6] = 0.2
    direction[8] = -0.5
    direction[11] = 0.15
    step = 1.0e-7

    def force(fraction):
        variables = np.r_[zero, zero, source_height, target_height] + fraction * direction
        state = patch.evaluate(variables[:3], variables[3:6],
                               variables[6:10], variables[10:])
        return np.r_[state.source_translation_residual,
                     state.target_translation_residual, state.normal_residual]

    numerical = (force(step) - force(-step)) / (2 * step)
    np.testing.assert_allclose(
        numerical, reduced.tangent @ direction,
        rtol=1.0e-5, atol=1.0e-5,
    )
    assert reduced.relative_tangent_asymmetry < 1.0e-5
