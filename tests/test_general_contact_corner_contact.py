"""Focused corner-contact geometry, gradient, certificate caller and NCP-core checks."""
from types import SimpleNamespace

import numpy as np
import pytest

from anysolver.general_contact.corner_contact_certificate import (
    CornerContactRefusal, PENETRATION_LIMIT_M, directional_exclusion, whole_face_certificates,
)
from anysolver.general_contact.corner_contact_geometry import closest_panel_point, corner_gaps
from anysolver.general_contact.corner_contact_solver import (
    Cancelled, Evaluation, NewtonFailure, solve_ncp,
)


def make_panel(height, nx=2, ny=1, size=1.0):
    """Grid of original bilinear faces; shared corners are copied per face."""
    positions, nodes, slices = [], [], []
    for iy in range(ny):
        for ix in range(nx):
            ids = [(ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1)]
            slices.append((len(positions), len(positions) + 4))
            nodes.append(tuple(i + 100 * j for i, j in ids))
            positions += [(i * size, j * size, height(i * size, j * size)) for i, j in ids]
    positions = np.array(positions, dtype=float)
    pose = SimpleNamespace(positions=positions, director_gradient=np.zeros((len(positions), 3, 3)),
                           position_errors=np.zeros_like(positions))
    panel_map = SimpleNamespace(face_slices=tuple(slices),
                                faces=[SimpleNamespace(node_ids=n) for n in nodes])
    return panel_map, pose


def gap(x, panel_map, pose):
    return closest_panel_point(np.asarray(x, float), pose.positions, panel_map.face_slices,
                               [f.node_ids for f in panel_map.faces], pose.director_gradient)


def brute_distance(x, c, side):
    xi = np.linspace(-1, 1, 801)
    X, E = np.meshgrid(xi, xi)
    w = [0.25 * (1 - X) * (1 - E), 0.25 * (1 + X) * (1 - E), 0.25 * (1 + X) * (1 + E),
         0.25 * (1 - X) * (1 + E)]
    pts = sum(wi[..., None] * ci for wi, ci in zip(w, c))
    return float(np.min(np.linalg.norm(pts - x, axis=-1)))


def test_interior_projection_sign_and_weights():
    panel, pose = make_panel(lambda x, y: 0.0)
    above, below = gap((0.5, 0.5, 0.3), panel, pose), gap((0.5, 0.5, -0.2), panel, pose)
    assert above["gap"] == pytest.approx(0.3, abs=1e-14) and above["face"] == 0
    assert below["gap"] == pytest.approx(-0.2, abs=1e-14)
    np.testing.assert_allclose(above["weights"], 0.25)
    np.testing.assert_allclose(above["direction"], (0, 0, 1), atol=1e-14)


def test_tilted_plane_gap_is_normal_distance():
    panel, pose = make_panel(lambda x, y: 0.1 * x)
    x = np.array((0.5, 0.5, 0.4))
    expected = (0.4 - 0.05) / np.sqrt(1.01)
    assert gap(x, panel, pose)["gap"] == pytest.approx(expected, rel=1e-13)


def test_bilinear_warped_face_matches_dense_minimum_and_fd_gradients():
    height = lambda x, y: 0.2 * x * y
    panel, pose = make_panel(height, nx=1)
    x = np.array((0.45, 0.62, 0.35))
    info = gap(x, panel, pose)
    assert info["gap"] == pytest.approx(brute_distance(x, pose.positions[:4], 1), abs=2e-5)
    assert info["gap"] > 0
    h = 1e-6
    for axis in range(3):
        d = np.zeros(3)
        d[axis] = h
        fd = (gap(x + d, panel, pose)["gap"] - gap(x - d, panel, pose)["gap"]) / (2 * h)
        assert fd == pytest.approx(info["direction"][axis], abs=1e-8)
    for k in range(4):
        for axis in range(3):
            plus, minus = pose.positions.copy(), pose.positions.copy()
            plus[k, axis] += h
            minus[k, axis] -= h
            gp = closest_panel_point(x, plus, panel.face_slices, [f.node_ids for f in panel.faces],
                                     pose.director_gradient)["gap"]
            gm = closest_panel_point(x, minus, panel.face_slices, [f.node_ids for f in panel.faces],
                                     pose.director_gradient)["gap"]
            assert (gp - gm) / (2 * h) == pytest.approx(
                -info["weights"][k] * info["direction"][axis], abs=1e-8)


def test_shared_seam_has_one_owner_and_edge_weights():
    panel, pose = make_panel(lambda x, y: 0.0)
    info = gap((1.0, 0.5, 0.3), panel, pose)
    assert info["owners"] == [0, 1] and info["face"] == 0
    assert info["gap"] == pytest.approx(0.3)
    np.testing.assert_allclose(info["weights"], (0, 0.5, 0.5, 0), atol=1e-14)
    gaps = corner_gaps(np.array([[1.0, 0.5, 0.3]]), panel, pose, radius=0.5)
    assert list(gaps) == [0]


def test_convex_ridge_seam_uses_euclidean_distance_and_pair_wrench_balances():
    ridge = lambda x, y: 0.1 if x == 1.0 else 0.0
    panel, pose = make_panel(ridge)
    x = np.array((1.0, 0.5, 0.3))
    info = gap(x, panel, pose)
    assert info["owners"] == [0, 1] and info["gap"] == pytest.approx(0.2)
    lam = 7.0
    f_sphere = lam * info["direction"]
    a, _ = panel.face_slices[info["face"]]
    forces = {-1: (x, f_sphere)}
    for k, w in enumerate(info["weights"]):
        forces[a + k] = (pose.positions[a + k], -lam * w * info["direction"])
    total = sum(f for _, f in forces.values())
    moment = sum(np.cross(p, f) for p, f in forces.values())
    np.testing.assert_allclose(total, 0, atol=1e-13)
    np.testing.assert_allclose(moment, 0, atol=1e-13)


def test_refusals_nonunique_zero_distance_and_inconsistent_seam():
    valley = lambda x, y: -0.1 if x == 1.0 else 0.0
    panel, pose = make_panel(valley)
    with pytest.raises(CornerContactRefusal) as tie:
        gap((1.0, 0.5, 0.3), panel, pose)
    assert tie.value.code == "nonunique_closest_point"
    flat, fpose = make_panel(lambda x, y: 0.0)
    with pytest.raises(CornerContactRefusal) as zero:
        gap((1.0, 0.5, 0.0), flat, fpose)
    assert zero.value.code == "seam_zero_distance"
    broken = fpose.director_gradient.copy()
    broken[4] += 1e-3             # second face's copy of a shared corner has another chart
    bad = SimpleNamespace(positions=fpose.positions, director_gradient=broken)
    with pytest.raises(CornerContactRefusal):
        gap((1.0, 0.2, 0.3), flat, bad)


def test_warped_face_and_boundary_zero_distance_refuse():
    warped, wpose = make_panel(lambda x, y: 5.0 * x * y, nx=1)
    with pytest.raises(CornerContactRefusal) as inadmissible:
        gap((0.5, 0.5, 0.3), warped, wpose)
    assert inadmissible.value.code == "inadmissible_face"
    flat, fpose = make_panel(lambda x, y: 0.0)
    with pytest.raises(CornerContactRefusal) as boundary:
        gap((0.0, 0.5, 0.0), flat, fpose)      # outer edge of the only owning face
    assert boundary.value.code == "zero_distance_boundary"
    # interior zero distance keeps the ordinary signed-normal law
    assert gap((0.5, 0.5, 0.0), flat, fpose)["gap"] == pytest.approx(0.0, abs=1e-15)


def test_corner_search_prunes_far_corners_and_refuses_deep_below_or_nonfinite():
    panel, pose = make_panel(lambda x, y: 0.0)
    found = corner_gaps(np.array([[0.5, 0.5, 0.01], [0.5, 0.5, 1.0], [5.0, 5.0, 0.0]]),
                        panel, pose, radius=0.05)
    assert list(found) == [0]
    with pytest.raises(CornerContactRefusal) as deep:
        corner_gaps(np.array([[0.5, 0.5, -1.0]]), panel, pose, radius=0.05)
    assert deep.value.code == "corner_beyond_panel"
    with pytest.raises(CornerContactRefusal):
        corner_gaps(np.array([[0.5, np.nan, 0.0]]), panel, pose)


def _sphere(z, xs=(0.4, 0.6, 0.8), ys=(0.4, 0.8, 0.4)):
    # Clockwise from above: the outward (downward-facing) S3 normal opposes the panel
    # top normal, as the actual lower sphere facets do.
    positions = np.array([(x, y, z) for x, y in zip(xs, ys)], dtype=float)
    return (SimpleNamespace(face_slices=((0, 3),)),
            SimpleNamespace(positions=positions, position_errors=np.zeros_like(positions)))


def test_certificate_caller_is_directional_not_any_axis():
    panel, ppose = make_panel(lambda x, y: 0.0)
    below_map, below = _sphere(-0.1)
    cert = whole_face_certificates(below_map, below, panel, ppose)
    assert cert["pairs_certified"] >= 1 and cert["unavailable_pairs"] == []
    assert cert["minimum_bound_m"] == pytest.approx(-0.1, abs=1e-6)
    above_map, above = _sphere(0.2)
    cert = whole_face_certificates(above_map, above, panel, ppose)
    assert cert["pairs_certified"] == 0 and cert["minimum_bound_m"] == 0.0
    touch_map, touch = _sphere(0.0)
    cert = whole_face_certificates(touch_map, touch, panel, ppose)
    assert cert["unavailable_pairs"] == [] and cert["minimum_bound_m"] >= -1e-12
    out_map, out = _sphere(0.5, xs=(7, 7.5, 8), ys=(0, 1, 0))
    with pytest.raises(CornerContactRefusal) as outside:
        whole_face_certificates(out_map, out, panel, ppose)
    assert outside.value.code == "outside_panel"
    above_flag, horizontal = directional_exclusion(
        np.array((0, 0, -1.0)), np.array((1, 1, -0.5)),
        np.array([(0, 0, 0.0)]), np.array([(1, 1, 0.0)]))
    assert not above_flag[0] and not horizontal[0]


class Spring:
    """One DOF, spring k, load P, gap g0 + q: exact answer q = -g0, lambda = k q - P."""

    def __init__(self, load=-50.0):
        self.load = load

    def evaluate(self, q, mu, lam):
        value = lam.get(0, 0.0)
        return Evaluation(np.array([100.0 * q[0] - self.load - value]), np.array([[100.0]]),
                          [0], np.array([0.3 + q[0]]), np.array([[1.0]]), np.zeros(0),
                          np.zeros((0, 1)))


def test_ncp_core_loaded_unloaded_cancelled_and_singular():
    result = solve_ncp(Spring(), np.zeros(1), np.zeros(0), {}, f_ref=50.0, force_tolerance=5e-7)
    assert result["q"][0] == pytest.approx(-0.3, abs=1e-12)
    assert result["lam"][0] == pytest.approx(20.0, rel=1e-10)
    free = solve_ncp(Spring(-10.0), np.zeros(1), np.zeros(0), {0: 5.0}, f_ref=50.0,
                     force_tolerance=5e-7)
    assert free["lam"] == {} and free["q"][0] == pytest.approx(-0.1)
    with pytest.raises(Cancelled):
        solve_ncp(Spring(), np.zeros(1), np.zeros(0), {}, f_ref=50.0, force_tolerance=5e-7,
                  cancel=lambda: True)

    class Singular(Spring):
        def evaluate(self, q, mu, lam):
            ev = super().evaluate(q, mu, lam)
            ev.stiffness = np.zeros((1, 1))
            ev.gradients = np.zeros((1, 1))
            return ev
    with pytest.raises(NewtonFailure):
        solve_ncp(Singular(), np.zeros(1), np.zeros(0), {}, f_ref=50.0, force_tolerance=5e-7)
