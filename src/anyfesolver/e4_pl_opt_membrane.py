"""Felippa OPT membrane triangle with drilling freedoms (ANDES LST-3/9R template).

Port of C. A. Felippa, "A study of optimal membrane triangles with drilling
freedoms", Comput. Methods Appl. Mech. Engrg. 192 (2003) 2125-2168, Figures 10
and 12 (template (31), optimal signature (41)/(43)).  Pure kernel: no element
state, no solver imports.  Freedom order per node: (u, v, theta) in the element
plane, theta = counter-clockwise rotation about the plane normal.

Fixed optimal signature: alpha_b = 3/2, beta_1..beta_9 = 1,2,1,0,1,-1,-1,-1,-2;
beta_0 = max(2/E11C11avg - 3/2, 1/100) with the Figure 12 invariant (equals
(1 - 4 nu^2)/2 for isotropic material).
"""
from __future__ import annotations

import numpy as np

ALPHA_B = 1.5
BETAS = (1.0, 2.0, 1.0, 0.0, 1.0, -1.0, -1.0, -1.0, -2.0)
BETA0_FLOOR = 0.01


def isotropic_plane_stress(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    e, nu = float(youngs_modulus), float(poisson_ratio)
    if not (np.isfinite(e) and e > 0.0 and np.isfinite(nu) and -1.0 < nu < 0.5):
        raise ValueError("isotropic plane-stress constants out of range")
    return e / (1.0 - nu * nu) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])


def _checked_material(emat) -> np.ndarray:
    d = np.array(emat, dtype=float)
    if d.shape != (3, 3) or not np.all(np.isfinite(d)):
        raise ValueError("elasticity matrix must be a finite 3x3 array")
    if np.max(np.abs(d - d.T)) > 1e-12 * np.max(np.abs(d)):
        raise ValueError("elasticity matrix must be symmetric")
    d = 0.5 * (d + d.T)
    if np.min(np.linalg.eigvalsh(d)) <= 0.0:
        raise ValueError("elasticity matrix must be positive definite")
    return d


def optimal_beta0(emat) -> float:
    """Overall higher-order scaling beta_0 of Figure 12 (orientation-averaged optimum)."""
    d = _checked_material(emat)
    e11, e12, e13 = d[0]
    e22, e23, e33 = d[1, 1], d[1, 2], d[2, 2]
    edet = float(np.linalg.det(d))
    w = (-6 * e12**3 + 5 * e11**2 * e22 - 5 * e12**2 * e22 - e22 * (75 * e13**2 + 14 * e13 * e23 + 3 * e23**2)
         + 2 * e12 * (7 * e13**2 + 46 * e13 * e23 + 7 * e23**2)
         - e11 * (5 * e12**2 + 3 * e13**2 - 6 * e12 * e22 - 5 * e22**2 + 14 * e13 * e23 + 75 * e23**2)
         + (3 * e11**2 + 82 * e11 * e22 + 3 * e22**2 - 4 * (6 * e12**2 + 5 * e13**2 - 6 * e13 * e23 + 5 * e23**2)) * e33
         + 4 * (5 * e11 - 6 * e12 + 5 * e22) * e33**2)
    return max(2.0 / (w / (128.0 * edet)) - 1.5, BETA0_FLOOR)


def _template_matrices(xy, thickness: float):
    p = np.array(xy, dtype=float)
    if p.shape != (3, 2) or not np.all(np.isfinite(p)):
        raise ValueError("triangle coordinates must be a finite 3x2 array")
    h = float(thickness)
    if not (np.isfinite(h) and h > 0.0):
        raise ValueError("thickness must be positive and finite")
    (x1, y1), (x2, y2), (x3, y3) = p
    x12, x23, x31 = x1 - x2, x2 - x3, x3 - x1
    y12, y23, y31 = y1 - y2, y2 - y3, y3 - y1
    x21, x32, x13 = -x12, -x23, -x31
    y21, y32, y13 = -y12, -y23, -y31
    area = 0.5 * (y21 * x13 - x21 * y13)
    scale = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
    if not area > 1e-12 * scale * scale:
        raise ValueError("triangle must be counter-clockwise with non-degenerate area")
    a2, a4 = 2.0 * area, 4.0 * area
    b = ALPHA_B
    lm = np.array([
        [y23, 0, x32], [0, x32, y23],
        np.array([y23 * (y13 - y21), x32 * (x31 - x12), (x31 * y13 - x12 * y21) * 2]) * b / 6,
        [y31, 0, x13], [0, x13, y31],
        np.array([y31 * (y21 - y32), x13 * (x12 - x23), (x12 * y21 - x23 * y32) * 2]) * b / 6,
        [y12, 0, x21], [0, x21, y12],
        np.array([y12 * (y32 - y13), x21 * (x23 - x31), (x23 * y32 - x31 * y13) * 2]) * b / 6]) * h / 2
    tu = np.array([[x32, y32, a4, x13, y13, 0, x21, y21, 0],
                   [x32, y32, 0, x13, y13, a4, x21, y21, 0],
                   [x32, y32, 0, x13, y13, 0, x21, y21, a4]]) / a4
    l21, l32, l13 = x21**2 + y21**2, x32**2 + y32**2, x13**2 + y13**2
    te = np.array([[y23 * y13 * l21, y31 * y21 * l32, y12 * y32 * l13],
                   [x23 * x13 * l21, x31 * x21 * l32, x12 * x32 * l13],
                   [(y23 * x31 + x32 * y13) * l21, (y31 * x12 + x13 * y21) * l32, (y12 * x23 + x21 * y32) * l13]]) / (area * a4)
    k = BETAS
    q1 = np.array([np.array(k[0:3]) / l21, np.array(k[3:6]) / l32, np.array(k[6:9]) / l13]) * a2 / 3
    q2 = np.array([np.array([k[8], k[6], k[7]]) / l21, np.array([k[2], k[0], k[1]]) / l32,
                   np.array([k[5], k[3], k[4]]) / l13]) * a2 / 3
    q3 = np.array([np.array([k[4], k[5], k[3]]) / l21, np.array([k[7], k[8], k[6]]) / l32,
                   np.array([k[1], k[2], k[0]]) / l13]) * a2 / 3
    return h, area, lm, tu, te, (q1, q2, q3)


def opt_membrane_stiffness_parts(xy, emat, thickness: float, beta0: float | None = None, *, validated: bool = False):
    """Return (K_basic, K_higher), each 9x9, for a counter-clockwise triangle."""
    h, area, lm, tu, te, (q1, q2, q3) = _template_matrices(xy, thickness)
    em = np.asarray(emat, dtype=float) if validated else _checked_material(emat)
    b0 = optimal_beta0(em) if beta0 is None else float(beta0)
    kb = lm @ em @ lm.T / (h * area)
    q4, q5, q6 = (q1 + q2) / 2, (q2 + q3) / 2, (q3 + q1) / 2
    en = te.T @ em @ te
    kt = 0.75 * b0 * h * area * (q4.T @ en @ q4 + q5.T @ en @ q5 + q6.T @ en @ q6)
    return kb, tu.T @ kt @ tu


RECOVERY_E0 = 1.5


def opt_membrane_strain_operator(xy, barycentric, thickness: float = 1.0, e0: float = RECOVERY_E0) -> np.ndarray:
    """Strain recovery operators, paper eq. (48): shape (n, 3, 9) for each barycentric row."""
    h, area, lm, tu, te, (q1, q2, q3) = _template_matrices(xy, thickness)
    bary = np.asarray(barycentric, dtype=float).reshape(-1, 3)
    basic = lm.T / (h * area)
    out = np.empty((bary.shape[0], 3, 9))
    for index, (z1, z2, z3) in enumerate(bary):
        out[index] = basic + te @ (float(e0) * (z1 * q1 + z2 * q2 + z3 * q3)) @ tu
    return out


def opt_membrane_stiffness(
    xy, emat, thickness: float, beta0: float | None = None, *, validated: bool = False
) -> np.ndarray:
    kb, kh = opt_membrane_stiffness_parts(xy, emat, thickness, beta0, validated=validated)
    k = kb + kh
    return 0.5 * (k + k.T)
