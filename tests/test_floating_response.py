"""Pure numeric floating-body response API: analytic heave/roll/pitch, axes covariance, refusals. No native work."""
import numpy as np
import pytest

from anysolver import floating_response as fr

RHO, G = 1025.0, 9.81


def box_hull(length=20.0, beam=6.0, draft=2.0):
    x0, x1, y0, y1, z0 = -length / 2, length / 2, -beam / 2, beam / 2, -draft
    corners = {"bottom": ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (0, 0, -1)),
               "front": ((x1, y0, z0), (x1, y1, z0), (x1, y1, 0), (x1, y0, 0), (1, 0, 0)),
               "back": ((x0, y0, z0), (x0, y1, z0), (x0, y1, 0), (x0, y0, 0), (-1, 0, 0)),
               "port": ((x0, y1, z0), (x1, y1, z0), (x1, y1, 0), (x0, y1, 0), (0, 1, 0)),
               "starboard": ((x0, y0, z0), (x1, y0, z0), (x1, y0, 0), (x0, y0, 0), (0, -1, 0))}
    verts, faces = [], []
    for *quad, out in corners.values():
        base = len(verts)
        verts.extend(quad)
        for tri in ((0, 1, 2), (0, 2, 3)):
            a, b, c = (np.array(quad[k], float) for k in tri)
            flip = np.dot(np.cross(b - a, c - a), out) < 0
            faces.append([base + tri[0], base + tri[2], base + tri[1]] if flip else [base + k for k in tri])
    return np.array(verts, float), np.array(faces)


def rot(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def box_restoring(zg=-0.5, length=20.0, beam=6.0, draft=2.0):
    v, f = box_hull(length, beam, draft)
    mass = RHO * length * beam * draft
    return fr.hydrostatic_restoring_world(v, f, rho=RHO, gravity=G, mass=mass, cg_world=(0.0, 0.0, zg)), mass


def test_box_restoring_matches_analytic_and_is_statically_balanced():
    out, _ = box_restoring()
    C = out["C_world"]
    L, B, T, zg = 20.0, 6.0, 2.0, -0.5
    V = L * B * T
    assert out["force_residual_rel"] < 1e-12 and out["moment_arm_horizontal"] < 1e-12
    np.testing.assert_allclose(C[2, 2], RHO * G * L * B, rtol=1e-12)
    np.testing.assert_allclose(C[3, 3], RHO * G * (L * B ** 3 / 12 + V * (-T / 2 - zg)), rtol=1e-12)
    np.testing.assert_allclose(C[4, 4], RHO * G * (B * L ** 3 / 12 + V * (-T / 2 - zg)), rtol=1e-12)
    assert np.allclose(C[[0, 1, 5]][:, :], 0) and np.allclose(C[:, [0, 1, 5]], 0)
    fr.require_static_balance(out, force_tolerance=1e-9, moment_arm_tolerance=1e-9)


def test_direct_source_quadrature_restoring_equals_the_closed_form_at_exact_equilibrium_and_keeps_the_residual_coupling_otherwise():
    v, f = box_hull()
    out, mass = box_restoring()
    args = dict(Q0=np.eye(3), cg_world=(0.0, 0.0, -0.5), rho=RHO, gravity=G)
    C = fr.restoring_body_from_source(v, f, mass=mass, **args)
    np.testing.assert_allclose(C, out["C_world"], rtol=1e-9, atol=1e-9 * np.abs(C).max())      # same physics at exact balance
    # a (tolerated) 1 % weight imbalance enters ONLY through the rotating-axes coupling theta x (residual force): rows 0-1, cols 3-4
    heavy = fr.restoring_body_from_source(v, f, mass=1.01 * mass, **args)
    diff = heavy - C
    assert abs(diff[0, 4]) > 0 and abs(diff[1, 3]) > 0
    np.testing.assert_allclose(diff[0, 4], -diff[1, 3], rtol=1e-9)                              # the cross-product structure
    # the WHOLE difference is exactly the analytic residual operator (theta x F0, theta x T0); translation columns are untouched
    R = fr.residual_restoring_operator(np.eye(3), buoyancy=out["buoyancy_force"], weight=1.01 * mass * G,
                                       buoyancy_centre_relative_cg=out["buoyancy_centre_relative_cg"])
    np.testing.assert_allclose(diff, R, rtol=1e-9, atol=1e-9 * np.abs(R).max())
    assert np.abs(diff[:, :3]).max() <= 1e-9 * np.abs(diff).max()


def test_unbalanced_mass_is_refused_not_rescaled():
    v, f = box_hull()
    out = fr.hydrostatic_restoring_world(v, f, rho=RHO, gravity=G, mass=1.1 * RHO * 240, cg_world=(0, 0, 0))
    with pytest.raises(fr.FloatingResponseError):
        fr.require_static_balance(out, force_tolerance=1e-3, moment_arm_tolerance=1e-3)


def _operators(nw=3):
    out, mass = box_restoring()
    Mdry = fr.dry_mass_matrix_body(mass, np.diag([2e6, 8e6, 9e6]), np.eye(3))
    w = np.array([0.4, 0.8, 1.2])[:nw]
    A = np.stack([np.diag([1e5, 1e5, 4e5, 2e5, 3e6, 3e6]) for _ in w])
    B = np.stack([np.diag([1e3, 1e3, 5e4, 1e4, 1e5, 1e5]) for _ in w])
    return w, Mdry, A, B, out["C_world"]


def test_heave_matches_single_dof_analytic_and_free_directions_stay_free():
    w, M, A, B, C = _operators()
    F = np.zeros((3, 6), complex)
    F[:, 2] = 1e5 * (1 + 0.3j)
    resp = fr.solve_floating_response(w, M, A, B, C, F, asymmetry_tolerance=1e-12)
    expected = F[:, 2] / (C[2, 2] - w ** 2 * (M[2, 2] + A[:, 2, 2]) - 1j * w * B[:, 2, 2])
    np.testing.assert_allclose(resp.Hq[:, 2], expected, rtol=1e-12)
    assert resp.restoring_null_dimension == 3          # surge, sway, yaw free; no stiffness invented
    assert np.all(resp.residual <= 1e-10)
    # free surge responds to an x force without any stabilising stiffness/damping
    F2 = np.zeros((3, 6), complex)
    F2[:, 0] = 1.0
    r2 = fr.solve_floating_response(w, M, A, B, C, F2, asymmetry_tolerance=1e-12)
    np.testing.assert_allclose(r2.Hq[:, 0], 1.0 / (-w ** 2 * (M[0, 0] + A[:, 0, 0]) - 1j * w * B[:, 0, 0]), rtol=1e-12)
    assert not resp.diagnostics["damping_added"] and not resp.diagnostics["stiffness_added"]


def test_refuses_asymmetry_unstable_restoring_and_bad_frequencies():
    w, M, A, B, C = _operators()
    F = np.ones((3, 6), complex)
    A2 = A.copy(); A2[0, 0, 5] = 1e5
    with pytest.raises(fr.FloatingResponseError, match="asymmetry"):
        fr.solve_floating_response(w, M, A2, B, C, F, asymmetry_tolerance=1e-6)
    Cn = C.copy(); Cn[2, 2] = -1.0
    with pytest.raises(fr.FloatingResponseError, match="unstable"):
        fr.solve_floating_response(w, M, A, B, Cn, F, asymmetry_tolerance=1e-12)
    for bad in (np.array([0.0, 1.0, 2.0]), np.array([-1.0, 1.0, 2.0]), np.array([np.nan, 1.0, 2.0])):
        with pytest.raises(fr.FloatingResponseError):
            fr.solve_floating_response(bad, M, A, B, C, F, asymmetry_tolerance=1e-12)
    Bn = B.copy(); Bn[1, 2, 2] = -1e3
    with pytest.raises(fr.FloatingResponseError, match="damping"):
        fr.solve_floating_response(w, M, A, Bn, C, F, asymmetry_tolerance=1e-12)


def test_body_axes_covariance_with_heel_and_trim():
    """World-axes operators transformed to mean body axes give the same physical response (rotated back)."""
    w, M, A, B, C = _operators()
    Q0 = rot("x", np.deg2rad(4.0)) @ rot("y", np.deg2rad(-2.5))
    T = fr.basis_6(Q0)
    Mw, Aw, Bw = (T @ X @ T.T for X in (M, A, B))
    Cw = C
    F_world = np.zeros((3, 6), complex)
    F_world[:, 2] = 1e5
    F_world[:, 3] = 2e4j
    ref = fr.solve_floating_response(w, fr.world_to_body_operator(Mw, Q0), fr.world_to_body_operator(Aw, Q0),
                                     fr.world_to_body_operator(Bw, Q0), fr.world_to_body_operator(Cw, Q0),
                                     fr.world_to_body_generalized_force(F_world, Q0), asymmetry_tolerance=1e-12)
    world = fr.solve_floating_response(w, Mw, Aw, Bw, Cw, F_world, asymmetry_tolerance=1e-12)
    np.testing.assert_allclose(ref.Hq @ T.T, world.Hq, rtol=1e-9, atol=1e-18)
    # free world-horizontal / yaw directions are transformed, not assumed to be body rows
    Cb = fr.world_to_body_operator(Cw, Q0)
    assert np.linalg.matrix_rank(Cb, tol=1e-6 * np.abs(Cb).max()) == 3
    # body rows 0,1,5 are NOT the free ones at a heeled/trimmed pose: the null space is the transformed world one
    null = np.linalg.svd(0.5 * (Cb + Cb.T))[2][-3:]
    np.testing.assert_allclose(np.abs(null @ T.T[:, [2, 3, 4]]), 0, atol=1e-6)
    assert np.max(np.abs(Cb[[0, 1, 5], :])) > 0


def test_origin_covariance_of_rigid_motion_and_gravity_perturbation():
    Q0 = rot("x", np.deg2rad(3.0))
    cg = np.array([1.0, 2.0, -0.5])
    h = np.array([0.1, 0.0, 0.2, 0.01, -0.02, 0.005], complex) * (1 + 0.5j)
    pts = np.array([[3.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
    m = fr.rigid_motion_world(pts, cg, Q0, h)
    # point displacement equals CG displacement plus rotation cross offset (origin shift covariance)
    cg_disp = fr.rigid_motion_world(cg[None, :], cg, Q0, h)["displacement"][0]
    np.testing.assert_allclose(cg_disp, Q0 @ h[:3])
    shifted = cg_disp[None, :] + np.cross((Q0 @ h[3:])[None, :], pts - cg)
    np.testing.assert_allclose(m["displacement"], shifted)
    a = fr.rigid_acceleration_world(pts, cg, Q0, h, 0.7)
    np.testing.assert_allclose(a["translation"], -0.49 * m["displacement"])
    np.testing.assert_allclose(a["rotation"], -0.49 * m["rotation"])
    dp = fr.hydrostatic_perturbation_pressure(pts, cg, Q0, h, rho=RHO, gravity=G)
    np.testing.assert_allclose(dp, -RHO * G * m["displacement"][:, 2])
    g0 = fr.gravity_body(Q0, G)
    np.testing.assert_allclose(Q0 @ g0, [0, 0, -G], atol=1e-12)
    th = h[3:]
    np.testing.assert_allclose(fr.perturbed_gravity_body(Q0, G, th), -np.cross(th, g0))


# ------------------------------------------------------------- non-native rotated pose: audit, residual operator, sensitivity
def sheared_hull(shear=(0.4, -0.25)):
    """Closed submerged prism whose bottom is sheared (waterplane unchanged): non-zero horizontal CB offset, still closed/outward."""
    v, f = box_hull()
    v = v.copy()
    v[:, 0] += shear[0] * v[:, 2]
    v[:, 1] += shear[1] * v[:, 2]
    return v, f


Q_TILT = rot("x", np.deg2rad(3.0)) @ rot("y", np.deg2rad(2.0))          # 3 deg heel / 2 deg trim mean axes


def tilted_case(mass_factor=1.0, cg_shift=(0.0, 0.0)):
    """(vertices, faces, mass, cg) with the CG exactly under the CB (exact equilibrium) then optionally shifted (residual moment)."""
    v, f = sheared_hull()
    mass = RHO * 20.0 * 6.0 * 2.0
    probe = fr.hydrostatic_restoring_world(v, f, rho=RHO, gravity=G, mass=mass, cg_world=(0.0, 0.0, -0.5))
    xb, yb, _ = probe["buoyancy_centre_relative_cg"]
    cg = np.array([xb + cg_shift[0], yb + cg_shift[1], -0.5])
    return v, f, mass * mass_factor, cg


def _identity_residual(mass_factor, cg_shift):
    v, f, mass, cg = tilted_case(mass_factor, cg_shift)
    rest = fr.hydrostatic_restoring_world(v, f, rho=RHO, gravity=G, mass=mass, cg_world=cg)
    Cd = fr.restoring_body_from_source(v, f, Q0=Q_TILT, cg_world=cg, rho=RHO, gravity=G, mass=mass)
    Cg = fr.world_to_body_operator(rest["C_world"], Q_TILT)
    R = fr.residual_restoring_operator(Q_TILT, buoyancy=rest["buoyancy_force"], weight=rest["weight"],
                                       buoyancy_centre_relative_cg=rest["buoyancy_centre_relative_cg"])
    return v, f, mass, cg, rest, Cd, Cg, R


def test_rotated_exact_equilibrium_direct_equals_transformed_geometric_operator():
    *_, rest, Cd, Cg, R = _identity_residual(1.0, (0.0, 0.0))
    assert rest["force_residual_rel"] < 1e-12 and rest["moment_arm_horizontal"] < 1e-12
    assert np.abs(R).max() <= 1e-9 * np.abs(Cd).max()
    np.testing.assert_allclose(Cd, Cg, rtol=1e-9, atol=1e-9 * np.abs(Cd).max())


@pytest.mark.parametrize("mass_factor, cg_shift", [(1.0 + 1e-3, (0.0, 0.0)), (1.0, (0.05, -0.03)), (1.0 - 2e-3, (-0.04, 0.02))],
                         ids=["residual-force", "residual-moment", "force-and-moment"])
def test_rotated_residual_cases_follow_the_full_analytic_residual_operator(mass_factor, cg_shift):
    v, f, mass, cg, rest, Cd, Cg, R = _identity_residual(mass_factor, cg_shift)
    assert np.abs(R).max() > 1e-6 * np.abs(Cd).max()                          # the residual is genuinely exercised
    np.testing.assert_allclose(Cd - Cg, R, rtol=0.0, atol=1e-9 * np.abs(Cd).max())
    audit = fr.restoring_audit(Cd, Cg, R, length=20.0)
    assert audit["relative"] <= 1e-8 and audit["expected_residual_operator_scaled_max"] > 0
    assert np.abs(Cd - Cg).max() > 0                                          # the source C is the direct derivative, not the geometric one


@pytest.mark.parametrize("mass_factor, cg_shift", [(1.0, (0.0, 0.0)), (1.0 + 1e-3, (0.0, 0.0)), (1.0, (0.05, -0.03))])
def test_direct_operator_annihilates_world_horizontal_and_world_yaw_in_mean_body_axes(mass_factor, cg_shift):
    *_, Cd, Cg, R = _identity_residual(mass_factor, cg_shift)
    scale = np.abs(Cd).max()
    Q = Q_TILT
    for name, q in (("world x", np.r_[Q.T @ [1, 0, 0], 0, 0, 0]), ("world y", np.r_[Q.T @ [0, 1, 0], 0, 0, 0]),
                    ("world yaw", np.r_[0, 0, 0, Q.T @ [0, 0, 1]])):
        assert np.abs(Cd @ q).max() <= 1e-9 * scale, name
    assert np.abs(Cd @ np.r_[0, 0, 0, 0, 0, 1]).max() > 1e-6 * scale         # the MEAN-BODY yaw column is not the free one at 3/2 deg


def test_restoring_audit_refuses_sign_and_scale_corruption():
    *_, Cd, Cg, R = _identity_residual(1.0 + 1e-3, (0.05, -0.03))
    fr.restoring_audit(Cd, Cg, R, length=20.0)
    bad = Cd.copy(); bad[2, 3] *= -1                                          # one sign flip in the source operator
    with pytest.raises(fr.FloatingResponseError, match="restoring audit"):
        fr.restoring_audit(bad, Cg, R, length=20.0)
    with pytest.raises(fr.FloatingResponseError, match="restoring audit"):
        fr.restoring_audit(Cd, Cg, -R, length=20.0)                           # sign-corrupted residual operator
    with pytest.raises(fr.FloatingResponseError):
        fr.restoring_audit(Cd, Cg, R, length=0.0)
    small = Cd.copy(); small[0, 4] += 1e-6 * np.abs(Cd[:3, :3]).max()        # tiny coupling error, still above the 1e-8 closure
    with pytest.raises(fr.FloatingResponseError):
        fr.restoring_audit(small, Cg, R, length=20.0)


def _tilted_response(Cd, Cg, mass, asymmetry=0.05):
    M = fr.dry_mass_matrix_body(mass, Q_TILT @ np.diag([2e6, 8e6, 9e6]) @ Q_TILT.T, Q_TILT)
    w = np.array([0.5, 0.9, 1.3])
    A = np.stack([fr.world_to_body_operator(np.diag([1e5, 1e5, 4e5, 2e5, 3e6, 3e6]), Q_TILT) for _ in w])
    Bm = np.stack([fr.world_to_body_operator(np.diag([1e3, 1e3, 5e4, 1e4, 1e5, 1e5]), Q_TILT) for _ in w])
    F = np.zeros((3, 6), complex); F[:, 2] = 1e5 * (1 + 0.3j); F[:, 3] = 2e4j; F[:, 0] = 1e4
    F = fr.world_to_body_generalized_force(F, Q_TILT)
    direct = fr.solve_floating_response(w, M, A, Bm, Cd, F, asymmetry_tolerance=asymmetry)
    reference = fr.solve_floating_response(w, M, A, Bm, Cg, F, asymmetry_tolerance=asymmetry)
    return w, np.asarray(direct.Hq), np.asarray(reference.Hq)


def test_response_sensitivity_is_zero_at_exact_equilibrium_and_passes_a_small_rotated_residual():
    for mass_factor in (1.0, 1.0 + 1e-4):
        *_, Cd, Cg, R = _identity_residual(mass_factor, (0.0, 0.0))
        mass = RHO * 240.0 * mass_factor
        w, H, Hr = _tilted_response(Cd, Cg, mass)
        s = fr.response_sensitivity(w, H, Hr, length=20.0)
        assert s["max_error"] <= fr.RESPONSE_SENSITIVITY_LIMIT and s["bound"] == 0.01
        assert np.array(s["per_frequency_component_error"]).shape == (3, 6)
        if mass_factor == 1.0:
            assert s["max_error"] < 1e-8                                     # identical operators: identical responses


def test_response_sensitivity_refusal_floor_and_all_zero_behaviour():
    L = 20.0
    H = np.full((2, 6), 1e-3 + 0j)
    ref = H.copy(); ref[1, 4] *= 1.05                                        # 5 % in one component at one frequency
    with pytest.raises(fr.FloatingResponseError, match="sensitivity"):
        fr.response_sensitivity([0.5, 1.0], H, ref, length=L)
    ok = H.copy(); ok[0, 1] *= 1.005
    assert fr.response_sensitivity([0.5, 1.0], H, ok, length=L)["max_error"] < 0.01
    # near-zero channel: a component below 10 % of the dominant is measured against the floor, not divided by ~0
    d = np.array([[1.0, 1e-12, 0, 0, 0, 0]], complex); r = d.copy(); r[0, 1] = 3e-12
    assert fr.response_sensitivity([0.5], d, r, length=L)["max_error"] < 1e-9
    r2 = d.copy(); r2[0, 0] = 1.02                                           # the same kind of error in a DOMINANT component
    with pytest.raises(fr.FloatingResponseError):
        fr.response_sensitivity([0.5], d, r2, length=L)
    zero = np.zeros((2, 6), complex)
    assert fr.response_sensitivity([0.5, 1.0], zero, zero, length=L)["max_error"] == 0.0


def test_an_excessive_rotated_residual_is_refused_by_an_existing_gate_or_by_the_sensitivity_bound():
    """A 40 % mass imbalance is far beyond any admitted tolerance.  The existing stability/asymmetry gates may legitimately refuse it
    FIRST (they are not relaxed here); otherwise the sensitivity bound must.  Direct sensitivity refusal is covered by the manufactured test."""
    *_, Cd, Cg, R = _identity_residual(1.0 + 0.4, (0.0, 0.0))
    with pytest.raises(fr.FloatingResponseError):
        w, H, Hr = _tilted_response(Cd, Cg, RHO * 240.0 * 1.4)
        fr.response_sensitivity(w, H, Hr, length=20.0)


def test_helper_gates_cannot_be_relaxed_and_inputs_are_validated():
    *_, Cd, Cg, R = _identity_residual(1.0, (0.0, 0.0))
    for bad in (1e-7, 0.0, -1e-9, float("nan"), float("inf")):
        with pytest.raises(fr.FloatingResponseError):
            fr.restoring_audit(Cd, Cg, R, length=20.0, tolerance=bad)
    fr.restoring_audit(Cd, Cg, R, length=20.0, tolerance=1e-10)                 # stricter is permitted
    for bad_length in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(fr.FloatingResponseError):
            fr.restoring_audit(Cd, Cg, R, length=bad_length)
    nan6 = Cd.copy(); nan6[0, 0] = np.nan
    with pytest.raises(fr.FloatingResponseError):
        fr.restoring_audit(nan6, Cg, R, length=20.0)
    with pytest.raises(fr.FloatingResponseError):
        fr.restoring_audit(Cd[:5], Cg, R, length=20.0)
    H = np.ones((2, 6), complex)
    for kwargs in (dict(limit=0.05), dict(limit=0.0), dict(limit=float("nan")), dict(floor_fraction=0.5), dict(floor_fraction=float("nan"))):
        with pytest.raises(fr.FloatingResponseError):
            fr.response_sensitivity([0.5, 1.0], H, H, length=20.0, **kwargs)
    fr.response_sensitivity([0.5, 1.0], H, H, length=20.0, limit=0.005)         # stricter is permitted
    bad_h = H.copy(); bad_h[0, 0] = np.nan
    for omega, a, b in (([0.5], H, H), ([0.5, -1.0], H, H), ([0.5, np.nan], H, H), ([0.5, 1.0], bad_h, H), ([0.5, 1.0], H, H[:, :5])):
        with pytest.raises(fr.FloatingResponseError):
            fr.response_sensitivity(omega, a, b, length=20.0)


def test_rotation_envelope_and_waterline_envelope_refusals():
    H = np.zeros((2, 6), complex)
    H[:, 3] = np.deg2rad(2.0)
    assert fr.rotation_envelope(H, [1.0, 1.0]) == pytest.approx(np.deg2rad(4.0))
    with pytest.raises(fr.FloatingResponseError):
        fr.rotation_envelope(H, [1.0, 2.0])
    Hh = np.zeros((1, 6), complex)
    Hh[0, 2] = 1.0
    pts = np.array([[0.0, 0.0, 0.0]])
    env = fr.relative_waterline_envelope(np.array([[1.5 + 0j]]), pts, (0, 0, 0), np.eye(3), Hh, [2.0], clearance=2.0)
    assert env == pytest.approx(1.0)
    with pytest.raises(fr.FloatingResponseError):
        fr.relative_waterline_envelope(np.array([[1.5 + 0j]]), pts, (0, 0, 0), np.eye(3), Hh, [2.0], clearance=0.9)
