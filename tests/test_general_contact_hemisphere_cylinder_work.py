"""Supporting geometry/work checks, not a cylinder-contact qualification.

The reference searches the physical cap circle with SciPy bounded minimization.
It neither imports production stationary-point helpers nor differentiates their
equations. Finite differences below use independent reference gaps for work.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar

from anyfesolver.general_contact.beam_hemisphere_cylinder_work import (
    hemisphere_cylinder_work,
)
from anyfesolver.general_contact.beam_hemisphere_support import (
    HemispherePlaneGeometryError,
)


def _cap_direction(positions, endpoint, sign):
    # Actual B2/B3 endpoint derivative, independently from shape interpolation.
    if len(positions) == 2:
        tangent = positions[1] - positions[0]
    elif endpoint == 0:
        tangent = 4 * (positions[1] - positions[0]) - (positions[2] - positions[0])
    else:
        tangent = (positions[0] - positions[2]) - 4 * (positions[1] - positions[2])
    tangent = sign * tangent
    return tangent / np.linalg.norm(tangent)


def _reference(positions, endpoint, sign, radius, point, axis, obstacle_radius):
    """Dense bracketing of every rim local minimum plus a valid sphere point.

    Sampling only supplies brackets; the returned distance is continuously
    minimized. The objective is squared Euclidean distance to the infinite line.
    Smooth fixtures are deliberately separated from degenerate extrema.
    """
    center = positions[endpoint]
    direction = _cap_direction(positions, endpoint, sign)
    axis = axis / np.linalg.norm(axis)
    radial = center - point
    radial -= axis * np.dot(radial, axis)
    inward = -radial / np.linalg.norm(radial)
    candidates = []
    if np.dot(inward, direction) >= 0:
        skin = center + radius * inward
        candidates.append((np.linalg.norm(radial) - radius, skin, "sphere"))

    seed = np.eye(3)[np.argmin(np.abs(direction))]
    first = np.cross(direction, seed)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)

    def skin_at(angle):
        return center + radius * (np.cos(angle)[..., None] * first
                                  + np.sin(angle)[..., None] * second)

    def squared_distance(angle):
        relative = skin_at(np.asarray(angle)) - point
        residual = relative - np.sum(relative * axis, axis=-1)[..., None] * axis
        return np.sum(residual * residual, axis=-1)

    count = 720
    spacing = 2 * math.pi / count
    angles = np.arange(count) * spacing
    values = squared_distance(angles)
    brackets = np.flatnonzero((values <= np.roll(values, 1))
                             & (values <= np.roll(values, -1)))
    assert len(brackets), "reference failed to bracket the rim minimum"
    for index in brackets:
        angle = angles[index]
        result = minimize_scalar(squared_distance, bounds=(angle - spacing, angle + spacing),
                                 method="bounded", options={"xatol": 1e-14, "maxiter": 100})
        assert result.success, result.message
        candidates.append((math.sqrt(float(result.fun)), skin_at(np.asarray(result.x)), "rim"))
    distance, skin, branch = min(candidates, key=lambda item: item[0])
    return distance - obstacle_radius, skin, branch


def _fixture(nodes, endpoint, branch):
    sign = -1 if endpoint == 0 else 1
    center = np.array([1.05, 0.17, 0.24])
    direction = np.array([-0.8, -0.2, 0.5]) if branch == "sphere" else np.array([0.42, 0.23, 0.88])
    direction /= np.linalg.norm(direction)
    positions = np.empty((nodes, 3))
    positions[endpoint] = center
    if nodes == 2:
        positions[1 - endpoint] = center - 0.8 * direction
    else:
        # Curvature makes the actual endpoint tangent differ from the chord.
        middle = center - 0.4 * direction + np.array([0.07, -0.05, 0.03])
        positions[1] = middle
        positions[2 - endpoint] = 0.8 * direction + 4 * middle - 3 * center
    return positions, endpoint, sign, 0.4, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.87


SMOOTH_CASES = [(n, endpoint, branch) for n in (2, 3)
                for endpoint in (0, n - 1) for branch in ("sphere", "rim")]


def _unpack_work(work, original):
    positions, endpoint, sign, radius, _, axis, obstacle_radius = original
    return (work[:-3].reshape(positions.shape), endpoint, sign, radius,
            work[-3:], axis, obstacle_radius)


@pytest.mark.parametrize("nodes,endpoint,branch", SMOOTH_CASES)
def test_independent_physical_minimum_and_points(nodes, endpoint, branch):
    args = _fixture(nodes, endpoint, branch)
    state = hemisphere_cylinder_work(*args)
    reference_gap, _, reference_branch = _reference(*args)
    assert state.branch == reference_branch == branch
    assert state.gap < -0.05  # Stable active cases, away from penalty activation.
    assert_allclose(state.gap, reference_gap, atol=1e-10, rtol=0)
    positions, endpoint, sign, radius, point, axis, obstacle_radius = args
    direction = _cap_direction(positions, endpoint, sign)
    arm = state.skin_point - positions[endpoint]
    assert_allclose(np.linalg.norm(arm), radius, atol=2e-12, rtol=0)
    assert np.dot(arm, direction) >= -2e-12
    if branch == "rim":
        assert_allclose(np.dot(arm, direction), 0, atol=2e-12, rtol=0)
    else:
        assert np.dot(arm, direction) > 0.1
    relative = state.skin_point - point
    line_point = point + np.dot(relative, axis) * axis
    radial = state.skin_point - line_point
    expected_normal = radial / np.linalg.norm(radial)
    assert_allclose(state.normal, expected_normal, atol=2e-12, rtol=0)
    assert_allclose(state.rigid_point, line_point + obstacle_radius * expected_normal,
                    atol=2e-12, rtol=0)
    assert_allclose(state.skin_point - state.rigid_point, state.gap * state.normal,
                    atol=2e-12, rtol=0)


@pytest.mark.parametrize("nodes,endpoint,branch", SMOOTH_CASES)
def test_complete_virtual_work_and_hessian(nodes, endpoint, branch):
    args = _fixture(nodes, endpoint, branch)
    state = hemisphere_cylinder_work(*args)
    work = np.r_[args[0].ravel(), args[4]]
    size = len(work)
    stiffness = 2.7
    force = stiffness * state.gap * state.gradient
    tangent = stiffness * (np.outer(state.gradient, state.gradient)
                           + state.gap * state.hessian)

    def reference_gap(at):
        return _reference(*_unpack_work(at, args))[0]

    # All coordinates, including the cylinder point, contribute to virtual work.
    h = 1e-5
    reference_gradient = np.empty(size)
    for index in range(size):
        delta = h * np.eye(size)[index]
        reference_gradient[index] = (reference_gap(work + delta) - reference_gap(work - delta)) / (2 * h)
    assert_allclose(state.gradient, reference_gradient, atol=2e-8, rtol=2e-7)
    assert_allclose(state.hessian, state.hessian.T, atol=2e-12, rtol=0)
    assert_allclose(state.gradient.reshape(-1, 3).sum(axis=0), 0, atol=2e-12, rtol=0)

    rng = np.random.default_rng(291 + nodes + endpoint)
    directions = [rng.normal(size=size), np.zeros(size)]
    other_start = 3 * (1 if nodes == 3 else 1 - endpoint)
    directions[1][other_start:other_start + 3] = [0.3, -0.4, 0.2]
    directions[1][-3:] = [-0.2, 0.15, 0.25]
    for direction in directions:
        direction /= np.linalg.norm(direction)
        for h in (1e-5, 5e-6):
            plus = hemisphere_cylinder_work(*_unpack_work(work + h * direction, args))
            minus = hemisphere_cylinder_work(*_unpack_work(work - h * direction, args))
            assert plus.branch == minus.branch == branch
            assert_allclose((plus.gradient - minus.gradient) / (2 * h),
                            state.hessian @ direction, atol=3e-7, rtol=3e-6)
            gp, gm, g0 = (reference_gap(work + h * direction),
                          reference_gap(work - h * direction), reference_gap(work))
            assert_allclose((gp - gm) / (2 * h), np.dot(state.gradient, direction),
                            atol=2e-8, rtol=2e-7)
            assert_allclose((gp - 2 * g0 + gm) / h**2,
                            direction @ state.hessian @ direction, atol=8e-5, rtol=2e-4)
            ep, em, e0 = (0.5 * stiffness * g**2 for g in (gp, gm, g0))
            assert_allclose((ep - em) / (2 * h), np.dot(force, direction),
                            atol=2e-8, rtol=2e-7)
            assert_allclose((ep - 2 * e0 + em) / h**2,
                            direction @ tangent @ direction, atol=8e-5, rtol=2e-4)
    if branch == "rim":
        other = 1 if nodes == 3 else 1 - endpoint
        block = slice(3 * other, 3 * other + 3)
        assert np.linalg.norm(state.gradient[block]) > 1e-3
        assert np.linalg.norm(state.hessian[block, -3:]) > 1e-3


@pytest.mark.parametrize("nodes,endpoint,branch", SMOOTH_CASES)
def test_translation_rotation_and_axis_representation(nodes, endpoint, branch):
    args = _fixture(nodes, endpoint, branch)
    state = hemisphere_cylinder_work(*args)
    positions, endpoint, sign, radius, point, axis, obstacle_radius = args
    rotation_axis = np.array([0.2, -0.5, 0.7])
    rotation_axis /= np.linalg.norm(rotation_axis)
    a, b, c = rotation_axis
    cross = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    angle = 0.61
    rotation = np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross
    shift = np.array([0.21, -0.38, 0.17])
    transformed = hemisphere_cylinder_work(positions @ rotation.T + shift, endpoint, sign,
                                           radius, rotation @ point + shift,
                                           rotation @ axis, obstacle_radius)
    mapping = np.kron(np.eye(nodes + 1), rotation)
    assert transformed.branch == state.branch
    assert_allclose(transformed.gap, state.gap, atol=2e-12, rtol=0)
    assert_allclose(transformed.gradient, mapping @ state.gradient, atol=2e-10, rtol=0)
    assert_allclose(transformed.hessian, mapping @ state.hessian @ mapping.T, atol=2e-9, rtol=0)
    assert_allclose(transformed.skin_point, rotation @ state.skin_point + shift, atol=2e-10, rtol=0)
    assert_allclose(transformed.rigid_point, rotation @ state.rigid_point + shift, atol=2e-10, rtol=0)
    assert_allclose(transformed.normal, rotation @ state.normal, atol=2e-10, rtol=0)
    for multiplier in (3.0, -2.0):
        changed_axis = hemisphere_cylinder_work(positions, endpoint, sign, radius, point,
                                                multiplier * axis, obstacle_radius)
        assert_allclose(changed_axis.gap, state.gap, atol=2e-12, rtol=0)
        assert_allclose(changed_axis.gradient, state.gradient, atol=2e-10, rtol=0)
        assert_allclose(changed_axis.hessian, state.hessian, atol=2e-9, rtol=0)
    # Moving the chosen point along the infinite axis leaves the obstacle fixed.
    axial = np.zeros(3 * (nodes + 1))
    axial[-3:] = axis
    assert_allclose(state.gradient @ axial, 0, atol=2e-12, rtol=0)
    assert_allclose(state.hessian @ axial, 0, atol=2e-12, rtol=0)


def test_exterior_and_regular_geometry_rejects_unsupported_inputs():
    args = list(_fixture(3, 0, "rim"))
    for change in ("inside_cylinder", "inside_cap_radius", "zero_tangent", "zero_axis"):
        invalid = args.copy()
        invalid[0] = args[0].copy()
        if change == "inside_cylinder":
            invalid[-1] = 1.1  # Endpoint radial distance is below this radius.
        elif change == "inside_cap_radius":
            invalid[3] = 1.1
        elif change == "zero_tangent":
            invalid[0][:] = args[0][0]
        else:
            invalid[5] = np.zeros(3)
        with pytest.raises(HemispherePlaneGeometryError):
            hemisphere_cylinder_work(*invalid)


def test_active_branch_seam_selects_sphere_one_sided_hessian():
    # The inward radial point is exactly on the cap rim: gap is C1 here but its
    # Hessian changes with the sign of the tangent perturbation.
    positions = np.array([[1.0, 0.0, 0.0], [1.0, -0.8, 0.0]])
    args = (positions, 0, -1, 0.4, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.8)
    state = hemisphere_cylinder_work(*args)
    assert state.branch == "sphere"
    assert_allclose(state.gap, -0.2, atol=2e-12, rtol=0)
    work = np.r_[positions.ravel(), np.zeros(3)]
    direction = np.array([0.3, 0.1, 0.2, 0.7, -0.1, 0.05, 0.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)
    h = 1e-5
    first = hemisphere_cylinder_work(*_unpack_work(work + h * direction, args))
    second = hemisphere_cylinder_work(*_unpack_work(work + 2 * h * direction, args))
    assert first.branch == second.branch == "sphere"
    assert_allclose((-3 * state.gradient + 4 * first.gradient - second.gradient) / (2 * h),
                    state.hessian @ direction, atol=2e-8, rtol=2e-7)


def test_open_nonunique_rim_is_handled_conservatively():
    # Rim is a yz circle. Its two axial extrema have the same closest radial
    # distance, so selecting a skin point would create arbitrary work/history.
    positions = np.array([[1.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    try:
        state = hemisphere_cylinder_work(positions, 0, -1, 0.4, np.zeros(3),
                                        np.array([0.0, 0.0, 1.0]), 0.8)
    except HemispherePlaneGeometryError:
        return
    assert_allclose(state.gap, 0.2, atol=2e-12, rtol=0)
    assert state.skin_point is state.rigid_point is state.normal is None
    assert_allclose(state.gradient, 0, atol=0, rtol=0)
    assert_allclose(state.hessian, 0, atol=0, rtol=0)


def test_active_nonunique_rim_rejects_without_selecting_an_axial_extremum():
    # With an in-plane outward tangent the rim projects to a line segment.
    # Its interior closest point has two cap preimages with opposite z signs.
    center = np.array([1.001, 0.0, 0.0])
    direction = np.array([0.9989, math.sqrt(1 - 0.9989**2), 0.0])
    positions = np.array([center, center - 0.8 * direction])
    radius, obstacle_radius = 0.05, 1.0
    assert np.linalg.norm(center[:2]) > max(radius, obstacle_radius)
    assert 1.001 * math.sqrt(1 - direction[0]**2) < radius
    assert 1.001 * direction[0] - obstacle_radius < 0
    with pytest.raises(HemispherePlaneGeometryError):
        hemisphere_cylinder_work(positions, 0, -1, radius, np.zeros(3),
                                 np.array([0.0, 0.0, 1.0]), obstacle_radius)


def test_reference_circle_search_has_dimensional_accuracy():
    # Symmetry gives the unique closest rim direction (-0.8, 0, 0.6).
    center = np.array([1.0, 0.0, 0.2])
    direction = np.array([0.6, 0.0, 0.8])
    positions = np.array([center, center - 0.8 * direction])
    gap, _, branch = _reference(positions, 0, -1, 0.4, np.zeros(3),
                                np.array([0.0, 0.0, 1.0]), 0.8)
    assert branch == "rim"
    assert_allclose(gap, 1 - 0.4 * 0.8 - 0.8, atol=1e-11, rtol=0)


def test_large_exact_axial_shift_preserves_gap_and_complete_work():
    # Literal axis dot base = 0 exactly. Both shifted nodal differences remain
    # exactly representable, while a floating I-aaT projection leaks this shift.
    axis = np.array([1.0, 2.0, 3.0])
    base = np.array([1.0, -0.5, 0.0])
    positions = np.array([base, base - np.array([1.0, 0.0, 0.0])])
    original = hemisphere_cylinder_work(positions, 0, -1, 0.4, np.zeros(3), axis, 0.87)
    shifted_positions = positions + 2.0**40 * axis
    assert_allclose(shifted_positions[0] - shifted_positions[1],
                    positions[0] - positions[1], atol=0, rtol=0)
    shifted = hemisphere_cylinder_work(shifted_positions, 0, -1, 0.4,
                                      np.zeros(3), axis, 0.87)
    assert shifted.branch == original.branch == "rim"
    assert_allclose(shifted.gap, original.gap, atol=2e-12, rtol=0)
    assert_allclose(shifted.gradient, original.gradient, atol=2e-11, rtol=0)
    assert_allclose(shifted.hessian, original.hessian, atol=2e-10, rtol=0)
    # At this magnitude point coordinates are quantized much more coarsely
    # than the local distance/work; no tiny skin-point difference is asserted.


@pytest.mark.parametrize("endpoint_x", [np.nextafter(1.0, 0.0), 1.0,
                                        np.nextafter(1.0, 2.0)])
def test_sphere_activation_uses_conjugate_sign_without_fabricated_zero(endpoint_x):
    # Binary literals 0.4 and 0.6 sum to exactly one as rational values.
    # These are activation checks, separate from every smooth derivative test.
    positions = np.array([[endpoint_x, 0.0, 0.0], [endpoint_x + 0.5, 0.0, 0.0]])
    state = hemisphere_cylinder_work(positions, 0, -1, 0.4, np.zeros(3),
                                    np.array([0.0, 0.0, 1.0]), 0.6)
    assert state.branch == "sphere"
    expected = endpoint_x - 1.0
    if expected:
        assert state.gap != 0
        assert np.signbit(state.gap) == np.signbit(expected)
    assert_allclose(state.gap, expected, atol=0, rtol=1e-12)


@pytest.mark.parametrize("endpoint_x", [np.nextafter(1.0, 0.0), 1.0,
                                        np.nextafter(1.0, 2.0)])
def test_unique_rim_activation_sign_is_certified_or_explicitly_unresolved(endpoint_x):
    # Tangent (3/4,0,1) has unit direction (3/5,0,4/5). R=5/8 makes
    # the radial rim offset exactly 1/2; gap therefore equals endpoint_x-1.
    # A root enclosure may leave this tiny sign unresolved and must say so.
    center = np.array([endpoint_x, 0.0, 0.0])
    positions = np.array([center, center - np.array([0.75, 0.0, 1.0])])
    try:
        state = hemisphere_cylinder_work(positions, 0, -1, 0.625, np.zeros(3),
                                        np.array([0.0, 0.0, 1.0]), 0.5)
    except HemispherePlaneGeometryError as exc:
        message = str(exc).lower()
        assert any(word in message for word in ("sign", "activation", "unresolved", "certif")), message
        return
    assert state.branch == "rim"
    expected = endpoint_x - 1.0
    if expected:
        assert state.gap != 0
        assert np.signbit(state.gap) == np.signbit(expected)
    assert_allclose(state.gap, expected, atol=0, rtol=1e-10)
