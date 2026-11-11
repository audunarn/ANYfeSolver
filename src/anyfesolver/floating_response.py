"""Pure numeric linear 6-DOF floating-body frequency response (zero forward speed).

All generalized coordinates are translations and right-hand incremental rotations in MEAN BODY axes through the
mean centre of gravity; ``Q0`` maps body vectors to world vectors.  Time dependence is ``exp(-i w t)``.  No ANYmarine,
ANYhydro or Capytaine import: callers pass numeric arrays.  Nothing here repairs, symmetrises or damps an operator;
unacceptable inputs are refused with ``FloatingResponseError``.  The strict-K / time-domain gates of the marine
stack are unrelated to this API and unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

SCHEMA = "anysolver.floating_response.v1"
ROTATION_ENVELOPE_LIMIT_RAD = float(np.deg2rad(5.0))


class FloatingResponseError(ValueError):
    """An input or result cannot be admitted; nothing was repaired."""


def _frozen(value: Any, dtype: Any, shape: Optional[tuple] = None, name: str = "array") -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    if shape is not None and out.shape != shape:
        raise FloatingResponseError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise FloatingResponseError(f"{name} must be finite")
    # bytes-backed: the read-only flag cannot be flipped back on an owned buffer
    return np.frombuffer(out.tobytes(), dtype=out.dtype).reshape(out.shape)


def rotation_matrix(Q0: Any) -> np.ndarray:
    Q = np.array(Q0, dtype=np.float64, copy=True)
    if Q.shape != (3, 3) or not np.all(np.isfinite(Q)):
        raise FloatingResponseError("Q0 must be a finite 3x3 matrix")
    if np.max(np.abs(Q.T @ Q - np.eye(3))) > 1e-10 or np.linalg.det(Q) < 0.0:
        raise FloatingResponseError("Q0 must be a proper rotation (body->world)")
    return Q


def basis_6(Q0: Any) -> np.ndarray:
    """T with u_world6 = T @ q_body6 (translations and rotations both rotate with Q0)."""
    Q = rotation_matrix(Q0)
    T = np.zeros((6, 6))
    T[:3, :3] = Q
    T[3:, 3:] = Q
    return T


def world_to_body_operator(matrix_world: Any, Q0: Any) -> np.ndarray:
    """Work-conjugate transform T^T M T of a world-axes 6x6 operator (full off-diagonal blocks kept)."""
    T = basis_6(Q0)
    M = np.asarray(matrix_world)
    if M.shape[-2:] != (6, 6):
        raise FloatingResponseError("operator must end in 6x6")
    return np.einsum("ji,...jk,kl->...il", T, M, T, optimize=True)


def world_to_body_generalized_force(force_world: Any, Q0: Any) -> np.ndarray:
    """Generalized force (force, moment about mean CG) from world axes to body axes."""
    T = basis_6(Q0)
    F = np.asarray(force_world)
    if F.shape[-1] != 6:
        raise FloatingResponseError("generalized force must end in 6")
    return F @ T


def dry_mass_matrix_body(mass: float, inertia_world_about_cg: Any, Q0: Any) -> np.ndarray:
    """Explicit dry rigid-body mass matrix about the mean CG in mean body axes (no added mass)."""
    m = float(mass)
    if not np.isfinite(m) or m <= 0.0:
        raise FloatingResponseError("mass must be positive and finite")
    Q = rotation_matrix(Q0)
    I = np.array(inertia_world_about_cg, dtype=np.float64)
    if I.shape != (3, 3) or not np.all(np.isfinite(I)):
        raise FloatingResponseError("inertia must be a finite 3x3 tensor")
    out = np.zeros((6, 6))
    out[:3, :3] = m * np.eye(3)
    out[3:, 3:] = Q.T @ I @ Q
    return out


def _triangle_quadrature(vertices: np.ndarray, faces: np.ndarray):
    """Edge-midpoint 3-point rule (exact for quadratics on planar triangles): points (n,3,3), area vectors (n,3)."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    area_vec = 0.5 * np.cross(b - a, c - a)  # outward when faces are ordered outward
    mids = np.stack([(a + b) / 2, (b + c) / 2, (c + a) / 2], axis=1)
    return mids, area_vec


def hydrostatic_restoring_world(vertices: Any, faces: Any, *, rho: float, gravity: float, mass: float,
                                cg_world: Any) -> dict:
    """Total (hydrostatic + gravity) restoring in WORLD axes about the mean CG from the SAME panel quadrature.

    ``vertices/faces`` are the closed-by-waterplane submerged hull triangles in world axes with outward (into the
    fluid) normals, clipped at z=0.  Uses the divergence identities for a polyhedral hull:
    int_wp f(x,y) dA = -int_hull f n_z dS, V = +int z n_z dS, V x_B = +int x z n_z dS, V z_B = +1/2 int z^2 n_z dS.
    The matrix is NOT symmetrised.  Returns the operator plus the static balance residuals of the same quadrature.
    """
    V = np.array(vertices, dtype=np.float64)
    F = np.array(faces, dtype=np.intp)
    cg = np.array(cg_world, dtype=np.float64).reshape(3)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3 or F.size == 0:
        raise FloatingResponseError("hull must be a non-empty triangle mesh")
    if not (np.all(np.isfinite(V)) and np.all(np.isfinite(cg))):
        raise FloatingResponseError("hull and CG must be finite")
    rho, g, m = float(rho), float(gravity), float(mass)
    if min(rho, g, m) <= 0.0:
        raise FloatingResponseError("rho, gravity and mass must be positive")
    pts, av = _triangle_quadrature(V, F)
    nz = av[:, 2]
    x = pts[:, :, 0] - cg[0]
    y = pts[:, :, 1] - cg[1]
    z = pts[:, :, 2]  # absolute waterplane datum z=0
    zr = pts[:, :, 2] - cg[2]
    w = nz[:, None] / 3.0

    def q(f):
        return float(-np.sum(f * w))

    awp = q(np.ones_like(x))
    sx, sy = q(x), q(y)
    ixx, iyy, ixy = q(x * x), q(y * y), q(x * y)
    # Volume moments over the closed surface: int f_z dV = +oint f n_z dS (waterplane z=0 adds nothing for z-weighted f).
    vol = -q(z)
    vxb = -q(x * z)
    vyb = -q(y * z)
    vzb_abs = -q(0.5 * z * z)
    # centroid of buoyancy relative to CG
    xb = vxb / vol if vol != 0 else float("nan")
    yb = vyb / vol if vol != 0 else float("nan")
    zb = vzb_abs / vol if vol != 0 else float("nan")
    if not np.isfinite(vol) or vol <= 0.0:
        raise FloatingResponseError("hull displaced volume must be positive (check outward normals / clipping)")
    zb_rel = zb - cg[2]
    rg = rho * g
    C = np.zeros((6, 6))
    C[2, 2] = rg * awp
    C[2, 3] = rg * sy
    C[2, 4] = -rg * sx
    C[3, 2] = rg * sy
    C[4, 2] = -rg * sx
    C[3, 3] = rg * (iyy + vol * zb_rel) - 0.0  # weight term m g z_G is zero about CG origin
    C[4, 4] = rg * (ixx + vol * zb_rel)
    C[3, 4] = -rg * ixy
    C[4, 3] = -rg * ixy
    C[3, 5] = -rg * vol * xb + 0.0
    C[4, 5] = -rg * vol * yb + 0.0
    # C[3,3] and [4,4] already include the gravity cancellation -m g z_G with z measured from the CG
    # (rho g V z_B - m g z_G with origin at CG gives rho g V (z_B - z_G)).  Mass enters only the static residual.
    buoy = rg * vol
    weight = m * g
    moment_arm = np.array([xb, yb])
    return {
        "schema": SCHEMA,
        "C_world": C,
        "volume": vol, "waterplane_area": awp, "waterplane_first_moments": (sx, sy),
        "buoyancy_centre_relative_cg": (xb, yb, zb_rel),
        "buoyancy_force": buoy, "weight": weight,
        "force_residual_rel": abs(buoy - weight) / weight,
        "moment_arm_horizontal": float(np.hypot(*moment_arm)),
    }


_BARY_3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])      # exact for quadratics on a triangle


def restoring_body_from_source(vertices: Any, faces: Any, *, Q0: Any, cg_world: Any, rho: float, gravity: float,
                               mass: float) -> np.ndarray:
    """6x6 restoring C directly from the SAME source-panel quadrature and dry gravity, in MEAN BODY axes about the mean CG.

    For each unit generalized displacement q_j (translation / right-hand rotation in mean body axes): the panel pressure changes by
    -rho g dz with dz = (Q0 (t + theta x r_b))_z, the panel body-frame normals are fixed (no extra normal-rotation term), and the dry
    weight changes by m (-theta x g0), g0 = Q0^T g_world.  C[:, j] = -(force, moment about the CG).  Unlike the closed-form world
    waterplane formulas this keeps the coupling of the rotating mean-axes with a (tolerated) small equilibrium residual, so it is
    exactly the load derivative the FE load composition uses.  Not symmetrised; faces are outward triangles closed by the waterplane."""
    Q = rotation_matrix(Q0)
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.intp)
    cg = np.asarray(cg_world, dtype=np.float64).reshape(3)
    tri = V[F]                                                               # (n,3,3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal_body = (cross / (2.0 * area[:, None])) @ Q                          # world outward normal expressed in body axes
    points = np.einsum('qa,eac->eqc', _BARY_3, tri)                            # (n,3,3) quadrature points (world)
    r_body = (points - cg) @ Q
    g0 = Q.T @ np.array([0.0, 0.0, -float(gravity)])
    C = np.zeros((6, 6))
    for j in range(6):
        q = np.eye(6)[j]
        displacement = q[:3] + np.cross(q[3:], r_body)                          # body-axes displacement of the quadrature points
        dz = displacement @ Q[2, :]
        pressure = -float(rho) * float(gravity) * dz
        force = -pressure[:, :, None] * normal_body[:, None, :] * area[:, None, None] / 3.0
        wrench = np.r_[force.sum(axis=(0, 1)), np.cross(r_body, force).sum(axis=(0, 1))]
        wrench[:3] += float(mass) * -np.cross(q[3:], g0)
        C[:, j] = -wrench
    return C


RESTORING_AUDIT_TOLERANCE = 1e-8          # existing numerical-closure accuracy
RESPONSE_SENSITIVITY_LIMIT = 0.01         # fixed per-component bound (no user override)
RESPONSE_NEAR_ZERO_FLOOR = 0.1            # near-zero channel floor: 10 % of the frequency's dominant response


def residual_restoring_operator(Q0: Any, *, buoyancy: float, weight: float, buoyancy_centre_relative_cg: Any) -> np.ndarray:
    """Analytic coordinate-rotation operator that separates the body-frame source derivative from the geometric (fixed-axes) operator.

    With the mean residual force F0 = (0, 0, B - m g) and moment T0 = (B yB, -B xB, 0) about the mean CG (world axes; (xB, yB) the
    horizontal CB-CG offset), a rotation column theta of the mean body axes contributes ``theta x F0`` and ``theta x T0`` (components in
    mean body axes, Q0 transforms both):  C_direct = Q0-transformed geometric C + R.  At an exact equilibrium R = 0."""
    Q = rotation_matrix(Q0)
    xb, yb = float(buoyancy_centre_relative_cg[0]), float(buoyancy_centre_relative_cg[1])
    B = float(buoyancy)
    F0 = Q.T @ np.array([0.0, 0.0, B - float(weight)])
    T0 = Q.T @ np.array([B * yb, -B * xb, 0.0])
    R = np.zeros((6, 6))
    for j in range(3):
        e = np.eye(3)[j]
        R[:3, 3 + j] = np.cross(e, F0)
        R[3:, 3 + j] = np.cross(e, T0)
    return R


def _stricter_gate(value: float, maximum: float, name: str) -> float:
    """Fixed gates may only be tightened: a positive finite value not above the fixed maximum."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0 or v > maximum:
        raise FloatingResponseError(f"{name} must be positive, finite and not above the fixed {maximum:g}")
    return v


def _audit_scales(length: float):
    L = float(length)
    if not np.isfinite(L) or L <= 0.0:
        raise FloatingResponseError("hull length scale must be positive and finite")
    return np.array([1, 1, 1, 1 / L, 1 / L, 1 / L]), np.array([L, L, L, 1.0, 1.0, 1.0])


def restoring_audit(C_direct: Any, C_geometric_body: Any, residual_operator: Any, *, length: float,
                    tolerance: float = RESTORING_AUDIT_TOLERANCE) -> dict:
    """Independently scaled audit: ``C_direct - C_geometric - R`` must vanish at the numerical closure accuracy.

    Rows (force / moment) are scaled by 1 and 1/L, columns (translation / rotation) by L and 1, so every scaled entry is a force in N.
    The expected residual operator is removed first; sign errors, missing couplings or any other numerical/geometric inconsistency
    remain and REFUSE.  The unscaled mixed-unit absolute maximum is only a labelled legacy diagnostic."""
    rows, cols = _audit_scales(length)
    tolerance = _stricter_gate(tolerance, RESTORING_AUDIT_TOLERANCE, "restoring audit tolerance")
    for name, X in (("C_direct", C_direct), ("C_geometric", C_geometric_body), ("residual operator", residual_operator)):
        X = np.asarray(X, dtype=float)
        if X.shape != (6, 6) or not np.all(np.isfinite(X)):
            raise FloatingResponseError(f"restoring audit: {name} must be a finite 6x6 matrix")
    scale = lambda X: np.asarray(X, dtype=float) * rows[:, None] * cols[None, :]
    D = scale(np.asarray(C_direct, dtype=float) - np.asarray(C_geometric_body, dtype=float) - np.asarray(residual_operator, dtype=float))
    reference = max(float(np.abs(scale(C_direct)).max()), float(np.abs(scale(C_geometric_body)).max()), 1e-300)
    relative = float(np.abs(D).max() / reference)
    out = dict(relative=relative, scaled_max_difference=float(np.abs(D).max()), reference=reference, tolerance=float(tolerance),
               expected_residual_operator_scaled_max=float(np.abs(scale(residual_operator)).max()),
               legacy_unscaled_absolute_max_difference=float(np.abs(np.asarray(C_direct) - np.asarray(C_geometric_body)).max()))
    if not np.isfinite(relative) or relative > float(tolerance):
        raise FloatingResponseError(f"restoring audit: direct source operator minus geometric operator minus the analytic residual operator "
                                    f"is {relative:.3e} (scaled), above {float(tolerance):.1e}")
    return out


def response_sensitivity(omega: Any, Hq_direct: Any, Hq_reference: Any, *, length: float,
                         limit: float = RESPONSE_SENSITIVITY_LIMIT, floor_fraction: float = RESPONSE_NEAR_ZERO_FLOOR) -> dict:
    """Per-frequency, per-component sensitivity of the response to the admitted mean residual.

    ``Hq_reference`` is the response with the transformed geometric restoring and the SAME M/A/B/F (diagnostic only; the actual response
    keeps the direct C).  Translations are divided by the hull length, rotations are unscaled; a component's error is
    |d - r| / max(|d|, |r|, floor) with floor = floor_fraction x the frequency's dominant scaled response.  Above ``limit`` it REFUSES."""
    limit = _stricter_gate(limit, RESPONSE_SENSITIVITY_LIMIT, "response sensitivity limit")
    floor_fraction = _stricter_gate(floor_fraction, 1.0, "near-zero floor fraction")
    if floor_fraction != RESPONSE_NEAR_ZERO_FLOOR and floor_fraction > RESPONSE_NEAR_ZERO_FLOOR:
        raise FloatingResponseError("near-zero floor fraction must not exceed the fixed 0.1")
    _audit_scales(length)
    H = np.atleast_2d(np.asarray(Hq_direct, dtype=np.complex128))
    Rr = np.atleast_2d(np.asarray(Hq_reference, dtype=np.complex128))
    w = np.asarray(omega, dtype=float).reshape(-1)
    if H.shape != Rr.shape or H.ndim != 2 or H.shape[1] != 6 or w.size != H.shape[0]:
        raise FloatingResponseError("direct and reference responses must have matching (nw, 6) shape, one row per frequency")
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise FloatingResponseError("response sensitivity needs positive finite frequencies")
    if not (np.all(np.isfinite(H)) and np.all(np.isfinite(Rr))):
        raise FloatingResponseError("response sensitivity needs finite responses")
    scale = np.array([1 / float(length)] * 3 + [1.0] * 3)
    d, r = H * scale, Rr * scale
    err = np.zeros(H.shape)
    for k in range(H.shape[0]):
        dominant = max(float(np.abs(d[k]).max()), float(np.abs(r[k]).max()))
        if dominant == 0.0:
            continue                                                    # nothing responds: zero error, safely
        floor = float(floor_fraction) * dominant
        err[k] = np.abs(d[k] - r[k]) / np.maximum(np.maximum(np.abs(d[k]), np.abs(r[k])), floor)
    worst = float(err.max()) if err.size else 0.0
    out = dict(max_error=worst, per_frequency_component_error=err.tolist(), bound=float(limit), floor_fraction=float(floor_fraction),
               omega=np.asarray(omega, dtype=float).tolist(), Hq_reference=[[[z.real, z.imag] for z in row] for row in Rr])
    if not np.isfinite(worst) or worst > float(limit):
        raise FloatingResponseError(f"response sensitivity to the admitted static residual is {worst:.3e}, above the fixed bound {float(limit):g}")
    return out


def require_static_balance(restoring: dict, *, force_tolerance: float, moment_arm_tolerance: float) -> None:
    """Refuse an unbalanced mean state using caller-supplied (owner) criteria; no defaults, no repair."""
    if restoring["force_residual_rel"] > float(force_tolerance):
        raise FloatingResponseError(f"static vertical balance residual {restoring['force_residual_rel']:.3e} "
                                    f"exceeds {force_tolerance:.3e}")
    if restoring["moment_arm_horizontal"] > float(moment_arm_tolerance):
        raise FloatingResponseError(f"CB-CG horizontal offset {restoring['moment_arm_horizontal']:.3e} "
                                    f"exceeds {moment_arm_tolerance:.3e}")


def _rel_asym(X: np.ndarray) -> float:
    n = np.linalg.norm(X)
    return 0.0 if n == 0.0 else float(np.linalg.norm(X - X.swapaxes(-1, -2)) / n)


@dataclass(frozen=True)
class FloatingResponse:
    schema: str
    omega: np.ndarray            # (nw,)
    Hq: np.ndarray               # (nw,6) complex, mean body axes, per unit excitation
    condition: np.ndarray        # (nw,)
    residual: np.ndarray         # (nw,) relative ||ZH-F||/||F|| (0 for zero excitation)
    asymmetry: dict
    restoring_min_eig_rel: float
    restoring_null_dimension: int
    diagnostics: dict


def solve_floating_response(omega: Any, mass: Any, added_mass: Any, damping: Any, restoring: Any, excitation: Any, *,
                            asymmetry_tolerance: float, condition_limit: float = 1e12,
                            residual_limit: float = 1e-10, null_tolerance: float = 1e-9) -> FloatingResponse:
    """Hq = solve(Z, Fexc), Z = C - w^2 (Mdry + A) - i w B, positive finite w only.

    ``mass`` (6,6) dry; ``added_mass``/``damping`` (nw,6,6); ``restoring`` (6,6) possibly only positive
    semidefinite (free world-horizontal / yaw); ``excitation`` (nw,6).  Operators stay unsymmetrised; asymmetry is
    reported and refused above ``asymmetry_tolerance``.  No artificial damping or stiffness is ever added.
    """
    w = _frozen(omega, np.float64, name="omega")
    if w.ndim != 1 or w.size == 0 or np.any(w <= 0.0):
        raise FloatingResponseError("omega must be a non-empty vector of positive finite frequencies")
    nw = w.size
    M = _frozen(mass, np.float64, (6, 6), "mass")
    A = _frozen(added_mass, np.float64, (nw, 6, 6), "added_mass")
    B = _frozen(damping, np.float64, (nw, 6, 6), "damping")
    C = _frozen(restoring, np.float64, (6, 6), "restoring")
    Fx = _frozen(excitation, np.complex128, (nw, 6), "excitation")
    tol = float(asymmetry_tolerance)
    if not tol >= 0.0:
        raise FloatingResponseError("asymmetry_tolerance is required")
    asym = {"mass": _rel_asym(M), "added_mass": float(max(_rel_asym(a) for a in A)),
            "damping": float(max(_rel_asym(b) for b in B)), "restoring": _rel_asym(C)}
    for name, value in asym.items():
        if value > tol:
            raise FloatingResponseError(f"{name} asymmetry {value:.3e} exceeds {tol:.3e}")
    sym = lambda X: 0.5 * (X + X.swapaxes(-1, -2))
    cs = np.linalg.eigvalsh(sym(C))
    cmax = max(abs(cs).max(), 1e-300)
    min_c = float(cs.min() / cmax)
    if min_c < -float(null_tolerance):
        raise FloatingResponseError(f"restoring has a negative direction ({min_c:.3e}); equilibrium is unstable")
    null_dim = int(np.sum(cs <= float(null_tolerance) * cmax))
    Hq = np.zeros((nw, 6), dtype=np.complex128)
    damping_floor = np.zeros(nw)
    cond = np.zeros(nw)
    res = np.zeros(nw)
    # Explicit dry-mass admission: positive added mass must never hide an invalid dry mass.
    dry = np.linalg.eigvalsh(sym(M))
    if dry.min() <= 1e-12 * max(dry.max(), 1e-300):
        raise FloatingResponseError("dry mass matrix is not positive definite")
    for k in range(nw):
        ma = sym(M + A[k])
        ev = np.linalg.eigvalsh(ma)
        if ev.min() <= 1e-12 * ev.max():
            raise FloatingResponseError(f"M+A is not positive definite at omega={w[k]:.6g}")
        bv = np.linalg.eigvalsh(sym(B[k]))
        # Passivity (radiation damping must dissipate) is independent of symmetry: its own strict gate; the eigenvalue is
        # reported as a diagnostic and nothing is symmetrised, trimmed or rescaled.
        damping_floor[k] = float(bv.min() / max(abs(bv).max(), 1e-300))
        if bv.min() < -float(null_tolerance) * max(abs(bv).max(), 1e-300):
            raise FloatingResponseError(f"radiation damping has a negative direction at omega={w[k]:.6g} "
                                        f"(min/max symmetric eigenvalue {bv.min():.3e}/{bv.max():.3e}, "
                                        f"relative {bv.min() / max(abs(bv).max(), 1e-300):.3e}; asymmetry {asym['damping']:.3e})")
        Z = C - w[k] ** 2 * (M + A[k]) - 1j * w[k] * B[k]
        cond[k] = np.linalg.cond(Z)
        if not np.isfinite(cond[k]) or cond[k] > condition_limit:
            raise FloatingResponseError(f"impedance condition {cond[k]:.3e} too large at omega={w[k]:.6g}")
        Hq[k] = np.linalg.solve(Z, Fx[k])
        fn = np.linalg.norm(Fx[k])
        res[k] = 0.0 if fn == 0.0 else np.linalg.norm(Z @ Hq[k] - Fx[k]) / fn
        if res[k] > residual_limit:
            raise FloatingResponseError(f"response residual {res[k]:.3e} exceeds {residual_limit:.3e}")
    Hq, cond, res = (_frozen(x, x.dtype) for x in (Hq, cond, res))
    return FloatingResponse(SCHEMA, w, Hq, cond, res, asym, min_c, null_dim,
                            {"convention": "exp(-i w t)", "axes": "mean_body_through_mean_cg",
                             "damping_min_eigenvalue_relative": damping_floor.tolist(),
                             "impedance": "C - w^2 (Mdry + A) - i w B", "damping_added": False,
                             "stiffness_added": False})


def rotation_envelope(Hq: Any, amplitudes: Any, *, limit: float = ROTATION_ENVELOPE_LIMIT_RAD,
                      refuse: bool = True) -> float:
    """sum_k amplitude_k * norm(complex rotation response_k) (rad); conservative, refuses above the 5 deg limit."""
    H = np.asarray(Hq, dtype=np.complex128)
    a = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if H.ndim != 2 or H.shape != (a.size, 6) or np.any(a < 0) or not np.all(np.isfinite(a)):
        raise FloatingResponseError("Hq must be (n,6) matching non-negative amplitudes")
    env = float(np.sum(a * np.linalg.norm(H[:, 3:], axis=1)))
    if refuse and env > float(limit):
        raise FloatingResponseError(f"rotation envelope {np.rad2deg(env):.3f} deg exceeds {np.rad2deg(limit):.3f} deg")
    return env


def rigid_motion_world(points_world: Any, cg_world: Any, Q0: Any, Hq: Any) -> dict:
    """Complex rigid motion of points from ONE body-axis response vector Hq (6,).

    translation_world = Q0 t, rotation_world = Q0 theta; u(x) = t_w + theta_w x (x - cg).
    Returns ``displacement`` (n,3) and ``rotation`` (n,3) (global axes, per node), plus body rotation for gravity.
    """
    P = np.asarray(points_world, dtype=np.float64)
    cg = np.asarray(cg_world, dtype=np.float64).reshape(3)
    Q = rotation_matrix(Q0)
    h = np.asarray(Hq, dtype=np.complex128).reshape(6)
    if P.ndim != 2 or P.shape[1] != 3:
        raise FloatingResponseError("points must be (n,3)")
    t = Q @ h[:3]
    th = Q @ h[3:]
    r = P - cg
    disp = t[None, :] + np.cross(th[None, :], r)
    rot = np.broadcast_to(th, P.shape).copy()
    return {"displacement": disp, "rotation": rot, "theta_body": h[3:].copy(), "translation_body": h[:3].copy()}


def rigid_acceleration_world(points_world: Any, cg_world: Any, Q0: Any, Hq: Any, omega: float) -> dict:
    """Primary nodal acceleration a = -w^2 R(r) Hq including angular nodal DOFs (exp(-i w t))."""
    m = rigid_motion_world(points_world, cg_world, Q0, Hq)
    s = -float(omega) ** 2
    return {"translation": s * m["displacement"], "rotation": s * m["rotation"]}


def hydrostatic_perturbation_pressure(points_world: Any, cg_world: Any, Q0: Any, Hq: Any, *, rho: float,
                                      gravity: float) -> np.ndarray:
    """-rho g ez . Q0 (t + theta x r_b), evaluated DIRECTLY at structural points (complex)."""
    m = rigid_motion_world(points_world, cg_world, Q0, Hq)
    return -float(rho) * float(gravity) * m["displacement"][:, 2]


def gravity_body(Q0: Any, gravity: float) -> np.ndarray:
    """g0 = Q0^T g_world, g_world = (0, 0, -g)."""
    return rotation_matrix(Q0).T @ np.array([0.0, 0.0, -float(gravity)])


def perturbed_gravity_body(Q0: Any, gravity: float, theta_body: Any) -> np.ndarray:
    """delta_g = -theta x g0 in body axes (no extra normal-rotation term)."""
    g0 = gravity_body(Q0, gravity)
    th = np.asarray(theta_body, dtype=np.complex128).reshape(3)
    return -np.cross(th, g0)


def relative_waterline_envelope(eta_total: Any, points_world: Any, cg_world: Any, Q0: Any, Hq: Any,
                                amplitudes: Any, *, clearance: float, refuse: bool = True) -> float:
    """Conservative sum_k amp_k |eta_total_k - dz_body_k| per point, max over points; must stay below clearance.

    ``eta_total`` (nk, npts) is the ACTUAL z_world=0 retained-field evaluation (incident+diffraction+radiation);
    never derive it from submerged top-panel pressure.  Gross admission check only.
    """
    E = np.asarray(eta_total, dtype=np.complex128)
    a = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    H = np.asarray(Hq, dtype=np.complex128)
    if E.ndim != 2 or E.shape[0] != a.size or H.shape != (a.size, 6):
        raise FloatingResponseError("eta_total/Hq/amplitudes shapes disagree")
    c = float(clearance)
    if not np.isfinite(c) or c <= 0.0:
        raise FloatingResponseError("clearance must be positive finite")
    env = np.zeros(E.shape[1])
    for k in range(a.size):
        dz = rigid_motion_world(points_world, cg_world, Q0, H[k])["displacement"][:, 2]
        env += a[k] * np.abs(E[k] - dz)
    out = float(env.max()) if env.size else 0.0
    if refuse and out >= c:
        raise FloatingResponseError(f"relative waterline envelope {out:.4g} m is not below clearance {c:.4g} m")
    return out
