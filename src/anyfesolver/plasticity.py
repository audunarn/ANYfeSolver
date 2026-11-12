"""Vectorized J2 plane-stress plasticity with isotropic hardening.

The return mapping follows the classical plane-stress projected algorithm
(Simo & Hughes).  In the eigenbasis of C*P the update decouples into two
scalar modes, so the plastic multiplier is found by a scalar Newton iteration
that runs simultaneously for every yielding integration point / thickness
layer (numpy arrays, no Python-level point loops).

When a tangent is requested for a nonlinear global solve, the return-map
residual is differentiated analytically with the implicit-function theorem.
The resulting algorithmic tangent includes the dependence of both the plastic
multiplier and the isotropic-hardening variable on total strain.  A central
finite-difference derivative of the same discrete update remains available as
an explicit qualification oracle and as an automatic fallback for
non-finite/ill-conditioned pathological states.

Conventions
-----------
Stress/strain vectors are [xx, yy, xy] with engineering shear strain.
The yield function is f = 1/2 sigma^T P sigma - 1/3 sigma_y(alpha)^2 with

    P = 1/3 [[ 2, -1, 0],
             [-1,  2, 0],
             [ 0,  0, 6]]

and alpha the equivalent plastic strain with rate
alpha_dot = lambda_dot * sqrt(2/3 sigma^T P sigma).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Dict, Iterator, Tuple

import numpy as np

from .jit_compiler import njit

if TYPE_CHECKING:
    from .material_curves import DNVC208MaterialCurve
    from .materials import Hill48Yield

_P_MATRIX = np.array(
    [[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 6.0]],
    dtype=float,
) / 3.0


@njit
def plane_stress_elastic_matrix(E: float, nu: float) -> np.ndarray:
    return E / (1.0 - nu**2) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
    )


_ANALYTICAL_TANGENT_CONDITION_LIMIT = 1.0e12
_ANALYTICAL_TANGENT_AMPLIFICATION_LIMIT = 1.0e8
_TANGENT_METHOD_OVERRIDE: ContextVar[str | None] = ContextVar(
    "anyfesolver_plane_stress_tangent_method",
    default=None,
)


class PlaneStressConvergenceError(RuntimeError):
    """Raised when the local plane-stress consistency solve does not converge."""


def _validate_plane_stress_inputs(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    E: float,
    nu: float,
    max_iterations: int,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, float]:
    """Normalize and validate the shared constitutive-array contract."""
    strain_array = np.asarray(strain, dtype=float)
    plastic_array = np.asarray(plastic_strain, dtype=float)
    alpha_array = np.asarray(alpha, dtype=float)
    if strain_array.ndim != 2 or strain_array.shape[1:] != (3,):
        raise ValueError("strain must have shape (n_points, 3)")
    if plastic_array.shape != strain_array.shape:
        raise ValueError("plastic_strain must have the same shape as strain")
    if alpha_array.shape != (strain_array.shape[0],):
        raise ValueError("alpha must have shape (n_points,)")
    if np.any(~np.isfinite(strain_array)) or np.any(~np.isfinite(plastic_array)):
        raise ValueError("strain and plastic_strain must contain only finite values")
    if np.any(~np.isfinite(alpha_array)):
        raise ValueError("alpha must contain only finite values")
    if np.any(alpha_array < 0.0):
        raise ValueError("alpha must be nonnegative")

    modulus = float(E)
    poisson = float(nu)
    if not np.isfinite(modulus) or modulus <= 0.0:
        raise ValueError("E must be finite and positive")
    if not np.isfinite(poisson) or not (-1.0 < poisson < 0.5):
        raise ValueError("nu must be finite and satisfy -1 < nu < 0.5")
    if isinstance(max_iterations, (bool, np.bool_)):
        raise ValueError("max_iterations must be a positive integer")
    iterations = int(max_iterations)
    if iterations <= 0 or float(iterations) != float(max_iterations):
        raise ValueError("max_iterations must be a positive integer")
    local_tolerance = float(tolerance)
    if not np.isfinite(local_tolerance) or local_tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return (
        strain_array,
        plastic_array,
        alpha_array,
        modulus,
        poisson,
        iterations,
        local_tolerance,
    )


def _normalize_tangent_method(tangent_method: str) -> str:
    method = str(tangent_method).strip().lower().replace("-", "_")
    if method in {"analytic", "automatic", "auto"}:
        method = "analytical"
    if method in {"finite_difference", "fd", "oracle"}:
        method = "numerical"
    if method not in {"analytical", "numerical"}:
        raise ValueError(
            "tangent_method must be 'analytical' or 'numerical' "
            f"(got {tangent_method!r})"
        )
    return method


@contextmanager
def plane_stress_tangent_method(tangent_method: str) -> Iterator[None]:
    """Temporarily override the tangent used by all plane-stress updates.

    This context-local diagnostic hook lets qualification run an unchanged
    global nonlinear model with the numerical oracle.  Normal production
    calls should use the default analytical tangent.  ``ContextVar`` keeps
    concurrent solver contexts isolated.
    """
    method = _normalize_tangent_method(tangent_method)
    token = _TANGENT_METHOD_OVERRIDE.set(method)
    try:
        yield
    finally:
        _TANGENT_METHOD_OVERRIDE.reset(token)


@njit
def _jit_flow_stress(
    eps_p: np.ndarray,
    sigma_prop: float,
    sigma_yield: float,
    sigma_yield_2: float,
    eps_p_y1: float,
    eps_p_y2: float,
    K: float,
    n: float,
    power_offset: float,
) -> np.ndarray:
    eps_p = np.maximum(eps_p, 0.0)
    slope_1 = (sigma_yield - sigma_prop) / eps_p_y1
    slope_2 = (sigma_yield_2 - sigma_yield) / (eps_p_y2 - eps_p_y1)
    part_1 = sigma_prop + slope_1 * eps_p
    part_2 = sigma_yield + slope_2 * (eps_p - eps_p_y1)

    res = np.zeros(eps_p.shape[0])
    for i in range(eps_p.shape[0]):
        val = eps_p[i]
        if val <= eps_p_y1:
            res[i] = part_1[i]
        elif val <= eps_p_y2:
            res[i] = part_2[i]
        else:
            res[i] = K * np.power(max(val + power_offset, 1.0e-12), n)
    return res


@njit
def _jit_flow_stress_scalar(
    eps_p: float,
    sigma_prop: float,
    sigma_yield: float,
    sigma_yield_2: float,
    eps_p_y1: float,
    eps_p_y2: float,
    K: float,
    n: float,
    power_offset: float,
) -> float:
    """Scalar flow stress used by the safeguarded consistency solve."""
    value = max(eps_p, 0.0)
    if value <= eps_p_y1:
        slope = (sigma_yield - sigma_prop) / eps_p_y1
        return sigma_prop + slope * value
    if value <= eps_p_y2:
        slope = (sigma_yield_2 - sigma_yield) / (eps_p_y2 - eps_p_y1)
        return sigma_yield + slope * (value - eps_p_y1)
    return K * np.power(max(value + power_offset, 1.0e-12), n)


@njit
def _jit_consistency_residual_scalar(
    plastic_multiplier: float,
    b1: float,
    b2: float,
    b3: float,
    alpha_n: float,
    c_a: float,
    shear_modulus: float,
    sigma_prop: float,
    sigma_yield: float,
    sigma_yield_2: float,
    eps_p_y1: float,
    eps_p_y2: float,
    K: float,
    n: float,
    power_offset: float,
) -> Tuple[float, float]:
    """Return the scalar consistency residual and current flow stress."""
    d_a = 1.0 + c_a * plastic_multiplier
    d_b = 1.0 + 2.0 * shear_modulus * plastic_multiplier
    # Match the returned-stress arithmetic, rather than squaring trial modes
    # before division. Equivalent expressions can round to opposite sides of
    # the local tolerance, prematurely ending the consistency solve.
    sig_a = b1 / d_a
    sig_b = b2 / d_b
    tau = b3 / d_b
    sx = 0.5 * (sig_a + sig_b)
    sy = 0.5 * (sig_a - sig_b)
    phi2 = (sx**2 - sx * sy + sy**2) / 3.0 + tau**2
    g = 2.0 * np.sqrt(max(phi2 / 3.0, 1.0e-30))
    alpha_value = alpha_n + plastic_multiplier * g
    flow = _jit_flow_stress_scalar(
        alpha_value,
        sigma_prop,
        sigma_yield,
        sigma_yield_2,
        eps_p_y1,
        eps_p_y2,
        K,
        n,
        power_offset,
    )
    return phi2 - flow**2 / 3.0, flow


@njit
def _jit_hardening_modulus(
    eps_p: np.ndarray,
    sigma_prop: float,
    sigma_yield: float,
    sigma_yield_2: float,
    eps_p_y1: float,
    eps_p_y2: float,
    K: float,
    n: float,
    power_offset: float,
) -> np.ndarray:
    eps_p = np.maximum(eps_p, 0.0)
    slope_1 = (sigma_yield - sigma_prop) / eps_p_y1
    slope_2 = (sigma_yield_2 - sigma_yield) / (eps_p_y2 - eps_p_y1)

    res = np.zeros(eps_p.shape[0])
    for i in range(eps_p.shape[0]):
        val = eps_p[i]
        if val <= eps_p_y1:
            res[i] = slope_1
        elif val <= eps_p_y2:
            res[i] = slope_2
        else:
            base = max(val + power_offset, 1.0e-12)
            res[i] = K * n * np.power(base, n - 1.0)
    return res


@njit
def _jit_plane_stress_return_map(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    E: float,
    nu: float,
    sigma_prop: float,
    sigma_yield: float,
    sigma_yield_2: float,
    eps_p_y1: float,
    eps_p_y2: float,
    K: float,
    n: float,
    power_offset: float,
    max_iterations: int = 30,
    tolerance: float = 1.0e-10,
    compute_tangent: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_points = strain.shape[0]
    C = plane_stress_elastic_matrix(E, nu)
    C_inv = np.linalg.inv(C)
    G = E / (2.0 * (1.0 + nu))
    c_a = E / (3.0 * (1.0 - nu))

    sigma = (strain - plastic_strain) @ C.T
    sy_n = _jit_flow_stress(
        alpha, sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K, n, power_offset
    )
    a1 = sigma[:, 0] + sigma[:, 1]
    a2 = sigma[:, 0] - sigma[:, 1]
    a3 = sigma[:, 2]
    phi2_trial = a1**2 / 12.0 + a2**2 / 4.0 + a3**2
    f_trial = phi2_trial - sy_n**2 / 3.0

    yielding = f_trial > tolerance * np.maximum(sy_n**2, 1.0)
    C_ep = np.zeros((n_points, 3, 3))
    if compute_tangent:
        for i in range(n_points):
            C_ep[i] = C
    new_plastic = plastic_strain.copy()
    new_alpha = alpha.copy()
    converged = np.ones(n_points, dtype=np.bool_)
    scaled_residual = np.zeros(n_points, dtype=float)

    if not np.any(yielding):
        return sigma, C_ep, new_plastic, new_alpha, converged, scaled_residual

    # Identify yielding indices
    yielding_indices = np.where(yielding)[0]
    n_yielding = yielding_indices.shape[0]

    b1 = a1[yielding]
    b2 = a2[yielding]
    b3 = a3[yielding]
    b23 = b2**2 / 4.0 + b3**2
    alpha_y = alpha[yielding]
    dl = np.zeros(n_yielding)

    for _ in range(max_iterations):
        dA = 1.0 + c_a * dl
        dB = 1.0 + 2.0 * G * dl
        sig_a = b1 / dA
        sig_b = b2 / dB
        tau = b3 / dB
        sx = 0.5 * (sig_a + sig_b)
        sy = 0.5 * (sig_a - sig_b)
        phi2 = (sx**2 - sx * sy + sy**2) / 3.0 + tau**2
        phi = np.sqrt(np.maximum(phi2, 1.0e-30))
        g = 2.0 * np.sqrt(np.maximum(phi2 / 3.0, 1.0e-30))
        alpha_new = alpha_y + dl * g
        sy = _jit_flow_stress(
            alpha_new, sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K, n, power_offset
        )
        H = _jit_hardening_modulus(
            alpha_new, sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K, n, power_offset
        )
        f = phi2 - sy**2 / 3.0

        all_scaled = True
        for i in range(n_yielding):
            if np.abs(f[i]) / max(sy[i]**2, 1.0) > tolerance:
                all_scaled = False
                break
        if all_scaled:
            break

        d_phi2 = -2.0 * (b1**2 * c_a / (12.0 * dA**3) + 2.0 * G * b23 / dB**3)
        d_g = d_phi2 / (3.0 * np.maximum(np.sqrt(phi2 / 3.0), 1.0e-30))
        d_alpha = g + dl * d_g
        d_f = d_phi2 - (2.0 / 3.0) * sy * H * d_alpha

        # Safe division step
        for i in range(n_yielding):
            df_val = d_f[i]
            if np.abs(df_val) <= 1.0e-30:
                df_val = -1.0e-30 if df_val < 0.0 else 1.0e-30
            step = f[i] / df_val
            dl[i] = max(dl[i] - step, 0.0)

    # Newton is fast for ordinary increments but can stall at a piecewise
    # hardening corner or during severe impact increments. The consistency
    # residual is monotone for the supported isotropic-hardening curves, so a
    # bracketed bisection safely completes any unconverged point.
    for i in range(n_yielding):
        residual_i, flow_i = _jit_consistency_residual_scalar(
            dl[i],
            b1[i],
            b2[i],
            b3[i],
            alpha_y[i],
            c_a,
            G,
            sigma_prop,
            sigma_yield,
            sigma_yield_2,
            eps_p_y1,
            eps_p_y2,
            K,
            n,
            power_offset,
        )
        if (
            np.isfinite(residual_i)
            and np.abs(residual_i) / max(flow_i**2, 1.0) <= tolerance
        ):
            continue

        lower = 0.0
        upper = max(dl[i], 1.0 / max(np.abs(E), 1.0))
        upper_residual, upper_flow = _jit_consistency_residual_scalar(
            upper,
            b1[i],
            b2[i],
            b3[i],
            alpha_y[i],
            c_a,
            G,
            sigma_prop,
            sigma_yield,
            sigma_yield_2,
            eps_p_y1,
            eps_p_y2,
            K,
            n,
            power_offset,
        )
        for _ in range(100):
            if np.isfinite(upper_residual) and upper_residual <= 0.0:
                break
            upper *= 2.0
            upper_residual, upper_flow = _jit_consistency_residual_scalar(
                upper,
                b1[i],
                b2[i],
                b3[i],
                alpha_y[i],
                c_a,
                G,
                sigma_prop,
                sigma_yield,
                sigma_yield_2,
                eps_p_y1,
                eps_p_y2,
                K,
                n,
                power_offset,
            )

        if not np.isfinite(upper_residual) or upper_residual > 0.0:
            dl[i] = upper
            continue

        midpoint = 0.5 * (lower + upper)
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            midpoint_residual, midpoint_flow = _jit_consistency_residual_scalar(
                midpoint,
                b1[i],
                b2[i],
                b3[i],
                alpha_y[i],
                c_a,
                G,
                sigma_prop,
                sigma_yield,
                sigma_yield_2,
                eps_p_y1,
                eps_p_y2,
                K,
                n,
                power_offset,
            )
            if (
                np.isfinite(midpoint_residual)
                and np.abs(midpoint_residual)
                / max(midpoint_flow**2, 1.0) <= tolerance
            ):
                break
            if not np.isfinite(midpoint_residual) or midpoint_residual > 0.0:
                lower = midpoint
            else:
                upper = midpoint
        dl[i] = midpoint

    dA = 1.0 + c_a * dl
    dB = 1.0 + 2.0 * G * dl
    sig_a = b1 / dA
    sig_b = b2 / dB
    tau = b3 / dB

    sigma_y_pts = np.zeros((n_yielding, 3))
    sigma_y_pts[:, 0] = (sig_a + sig_b) / 2.0
    sigma_y_pts[:, 1] = (sig_a - sig_b) / 2.0
    sigma_y_pts[:, 2] = tau

    for idx, i in enumerate(yielding_indices):
        sigma[i] = sigma_y_pts[idx]

    phi2 = (sigma_y_pts[:, 0]**2 - sigma_y_pts[:, 0] * sigma_y_pts[:, 1]
            + sigma_y_pts[:, 1]**2) / 3.0 + tau**2
    new_alpha[yielding] = alpha_y + dl * 2.0 * np.sqrt(np.maximum(phi2 / 3.0, 1.0e-30))
    sy_final = _jit_flow_stress(
        new_alpha[yielding],
        sigma_prop,
        sigma_yield,
        sigma_yield_2,
        eps_p_y1,
        eps_p_y2,
        K,
        n,
        power_offset,
    )
    final_residual = phi2 - sy_final**2 / 3.0
    for idx, i in enumerate(yielding_indices):
        denominator = max(sy_final[idx] ** 2, 1.0)
        scaled_residual[i] = np.abs(final_residual[idx]) / denominator
        if (
            not np.isfinite(final_residual[idx])
            or scaled_residual[i] > tolerance
        ):
            converged[i] = False

    p_strain_yielding = strain[yielding] - sigma_y_pts @ C_inv.T
    for idx, i in enumerate(yielding_indices):
        new_plastic[i] = p_strain_yielding[idx]

    if not compute_tangent:
        return sigma, C_ep, new_plastic, new_alpha, converged, scaled_residual

    # Exact derivative of the discrete projected return map.
    #
    # In the invariant trial-stress basis b = [sx+sy, sx-sy, txy],
    # returned modes are
    #
    #   q = [b1/A, b2/B, b3/B],
    #   A = 1 + c_a*dl,  B = 1 + 2G*dl.
    #
    # The scalar consistency equation f(b, dl) = 0 therefore gives
    # d(dl)/db = -f_b/f_dl.  The chain T * dq/db * L * C maps the mode
    # derivative back to [sxx, syy, txy] versus engineering strain.
    H_final = _jit_hardening_modulus(
        new_alpha[yielding], sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K, n, power_offset
    )
    T = np.array(
        [[0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.0, 1.0]]
    )
    L = np.array(
        [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    for idx, i in enumerate(yielding_indices):
        p = max(phi2[idx], 1.0e-30)
        sqrt_p = np.sqrt(p)
        g = 2.0 * np.sqrt(p / 3.0)
        dp_dl = -2.0 * (
            b1[idx] ** 2 * c_a / (12.0 * dA[idx] ** 3)
            + 2.0 * G * b23[idx] / dB[idx] ** 3
        )
        dg_dl = dp_dl / (np.sqrt(3.0) * sqrt_p)
        da_dl = g + dl[idx] * dg_dl
        df_dl = dp_dl - (2.0 / 3.0) * sy_final[idx] * H_final[idx] * da_dl

        dp_db = np.empty(3)
        dp_db[0] = b1[idx] / (6.0 * dA[idx] ** 2)
        dp_db[1] = b2[idx] / (2.0 * dB[idx] ** 2)
        dp_db[2] = 2.0 * b3[idx] / dB[idx] ** 2

        dl_db = np.empty(3)
        derivative_scale = (
            np.abs(dp_dl)
            + np.abs(
                (2.0 / 3.0)
                * sy_final[idx]
                * H_final[idx]
                * da_dl
            )
            + 1.0
        )
        local_derivative_valid = (
            np.isfinite(df_dl)
            and np.isfinite(g)
            and g > 1.0e-30
            and np.isfinite(dA[idx])
            and np.isfinite(dB[idx])
            and np.abs(dA[idx]) > 1.0e-14
            and np.abs(dB[idx]) > 1.0e-14
            and np.abs(df_dl) > 1.0e-14 * derivative_scale
        )
        for col in range(3):
            dg_db = dp_db[col] / (np.sqrt(3.0) * sqrt_p)
            da_db = dl[idx] * dg_db
            df_db = dp_db[col] - (2.0 / 3.0) * sy_final[idx] * H_final[idx] * da_db
            if local_derivative_valid:
                dl_db[col] = -df_db / df_dl
            else:
                dl_db[col] = np.nan

        dq_db = np.zeros((3, 3))
        dq_db[0, 0] = 1.0 / dA[idx]
        dq_db[1, 1] = 1.0 / dB[idx]
        dq_db[2, 2] = 1.0 / dB[idx]
        dq_dl = np.empty(3)
        dq_dl[0] = -c_a * b1[idx] / dA[idx] ** 2
        dq_dl[1] = -2.0 * G * b2[idx] / dB[idx] ** 2
        dq_dl[2] = -2.0 * G * b3[idx] / dB[idx] ** 2
        for row in range(3):
            for col in range(3):
                dq_db[row, col] += dq_dl[row] * dl_db[col]

        # T maps invariant returned modes to stress, while L maps trial
        # stress to the invariant basis.
        mode_tangent = T @ dq_db @ L
        material_tangent = mode_tangent @ C
        for r in range(3):
            for c in range(3):
                C_ep[i, r, c] = 0.5 * (
                    material_tangent[r, c] + material_tangent[c, r]
                )

    return sigma, C_ep, new_plastic, new_alpha, converged, scaled_residual


def _require_local_convergence(
    converged: np.ndarray,
    scaled_residual: np.ndarray,
    *,
    max_iterations: int,
) -> None:
    """Fail closed when any local consistency solve exhausted its iteration budget."""
    converged = np.asarray(converged, dtype=bool).reshape(-1)
    if np.all(converged):
        return
    residual = np.asarray(scaled_residual, dtype=float).reshape(-1)
    failed = np.flatnonzero(~converged)
    maximum = float(np.max(residual[failed])) if failed.size else float("nan")
    preview = ", ".join(str(int(index)) for index in failed[:8])
    suffix = "" if failed.size <= 8 else ", ..."
    raise PlaneStressConvergenceError(
        "Plane-stress return mapping did not satisfy the scaled yield "
        f"residual after {int(max_iterations)} iterations at point(s) "
        f"{preview}{suffix}; maximum scaled residual={maximum:.6g}."
    )


def plane_stress_return_map(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    E: float,
    nu: float,
    curve: "DNVC208MaterialCurve",
    max_iterations: int = 30,
    tolerance: float = 1.0e-10,
    compute_tangent: bool = True,
    tangent_method: str = "analytical",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map total strains to stresses, tangents and updated plastic state.

    ``tangent_method="analytical"`` (the default) implicitly differentiates
    the converged scalar consistency equation.  Invalid analytical rows
    automatically fall back to the numerical oracle.  Use
    ``tangent_method="numerical"`` to request that central-difference oracle
    explicitly for qualification and diagnosis.
    """
    (
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        max_iterations,
        tolerance,
    ) = _validate_plane_stress_inputs(
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        max_iterations,
        tolerance,
    )
    override = _TANGENT_METHOD_OVERRIDE.get()
    method = (
        _normalize_tangent_method(override)
        if override is not None
        else _normalize_tangent_method(tangent_method)
    )

    if curve is None:
        C = plane_stress_elastic_matrix(E, nu)
        n_points = strain.shape[0]
        C_ep = np.broadcast_to(C, (n_points, 3, 3)).copy() if compute_tangent else np.zeros((n_points, 3, 3))
        sigma = (strain - plastic_strain) @ C.T
        return sigma, C_ep, plastic_strain.copy(), alpha.copy()

    (
        sigma,
        analytical_tangent,
        new_plastic,
        new_alpha,
        converged,
        scaled_residual,
    ) = _jit_plane_stress_return_map(
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        float(curve.sigma_prop),
        float(curve.sigma_yield),
        float(curve.sigma_yield_2),
        float(curve.eps_p_y1),
        float(curve.eps_p_y2),
        float(curve.K),
        float(curve.n),
        float(curve._power_offset),
        max_iterations,
        tolerance,
        compute_tangent and method == "analytical",
    )
    _require_local_convergence(
        converged,
        scaled_residual,
        max_iterations=max_iterations,
    )
    if not compute_tangent:
        return sigma, analytical_tangent, new_plastic, new_alpha

    if method == "numerical":
        numerical_tangent = plane_stress_numerical_tangent(
            strain,
            plastic_strain,
            alpha,
            E,
            nu,
            curve,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        return sigma, numerical_tangent, new_plastic, new_alpha

    fallback_mask = _analytical_tangent_fallback_mask(
        analytical_tangent, E=E, nu=nu
    )
    if np.any(fallback_mask):
        analytical_tangent[fallback_mask] = plane_stress_numerical_tangent(
            strain[fallback_mask],
            plastic_strain[fallback_mask],
            alpha[fallback_mask],
            E,
            nu,
            curve,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    if np.any(~np.isfinite(analytical_tangent)):
        raise FloatingPointError(
            "Plane-stress tangent remained non-finite after the numerical fallback"
        )
    return sigma, analytical_tangent, new_plastic, new_alpha


def _analytical_tangent_fallback_mask(
    tangent: np.ndarray,
    *,
    E: float,
    nu: float,
    condition_limit: float = _ANALYTICAL_TANGENT_CONDITION_LIMIT,
    amplification_limit: float = _ANALYTICAL_TANGENT_AMPLIFICATION_LIMIT,
) -> np.ndarray:
    """Identify analytical tangent rows that require the numerical fallback."""
    tangent = np.asarray(tangent, dtype=float)
    if tangent.ndim != 3 or tangent.shape[1:] != (3, 3):
        raise ValueError("tangent must have shape (n_points, 3, 3)")
    fallback = ~np.all(np.isfinite(tangent), axis=(1, 2))
    try:
        elastic_eigenvalues = np.array(
            [
                float(E) / (1.0 - float(nu)),
                float(E) / (1.0 + float(nu)),
                float(E) / (2.0 * (1.0 + float(nu))),
            ],
            dtype=float,
        )
        elastic_magnitudes = np.abs(elastic_eigenvalues)
        elastic_norm = float(np.linalg.norm(elastic_magnitudes))
        elastic_condition = float(
            np.max(elastic_magnitudes)
            / max(np.min(elastic_magnitudes), np.finfo(float).tiny)
        )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.ones(tangent.shape[0], dtype=bool)
    if not np.isfinite(elastic_norm) or not np.isfinite(elastic_condition):
        return np.ones(tangent.shape[0], dtype=bool)
    # A large elastic condition number is diagnostic evidence, not by itself
    # proof that the analytical derivative is invalid. In particular, the
    # exact elastic tangent remains preferable to a differenced derivative
    # when nu approaches a stability boundary.
    _ = condition_limit
    tangent_norm = np.linalg.norm(tangent, axis=(1, 2))
    fallback |= tangent_norm > float(amplification_limit) * max(elastic_norm, 1.0)
    return fallback


def plane_stress_tangent_diagnostics(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    E: float,
    nu: float,
    curve: "DNVC208MaterialCurve",
    max_iterations: int = 30,
    tolerance: float = 1.0e-10,
    oracle_step: float = 1.0e-7,
) -> Dict[str, Any]:
    """Compare analytical rows with the numerical oracle and report fallbacks."""
    (
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        max_iterations,
        tolerance,
    ) = _validate_plane_stress_inputs(
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        max_iterations,
        tolerance,
    )
    (
        _stress,
        analytical,
        _plastic,
        _alpha,
        converged,
        scaled_residual,
    ) = _jit_plane_stress_return_map(
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        float(curve.sigma_prop),
        float(curve.sigma_yield),
        float(curve.sigma_yield_2),
        float(curve.eps_p_y1),
        float(curve.eps_p_y2),
        float(curve.K),
        float(curve.n),
        float(curve._power_offset),
        max_iterations,
        tolerance,
        True,
    )
    _require_local_convergence(
        converged,
        scaled_residual,
        max_iterations=max_iterations,
    )
    fallback_mask = _analytical_tangent_fallback_mask(analytical, E=E, nu=nu)
    oracle = plane_stress_numerical_tangent(
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        curve,
        max_iterations=max_iterations,
        tolerance=tolerance,
        step=oracle_step,
    )
    relative_errors = np.linalg.norm(analytical - oracle, axis=(1, 2)) / np.maximum(
        np.linalg.norm(oracle, axis=(1, 2)), 1.0
    )
    symmetry_errors = np.linalg.norm(
        analytical - np.swapaxes(analytical, 1, 2),
        axis=(1, 2),
    ) / np.maximum(np.linalg.norm(analytical, axis=(1, 2)), 1.0)
    return {
        "method": "analytical_implicit_consistent",
        "oracle": "central_finite_difference_discrete_return_map",
        "num_points": int(strain.shape[0]),
        "fallback_count": int(np.count_nonzero(fallback_mask)),
        "fallback_indices": np.flatnonzero(fallback_mask).astype(int).tolist(),
        "relative_errors": relative_errors.tolist(),
        "max_relative_error": float(np.max(relative_errors)) if relative_errors.size else 0.0,
        "symmetry_relative_errors": symmetry_errors.tolist(),
        "max_symmetry_relative_error": (
            float(np.max(symmetry_errors)) if symmetry_errors.size else 0.0
        ),
    }


def plane_stress_numerical_tangent(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    E: float,
    nu: float,
    curve: "DNVC208MaterialCurve",
    max_iterations: int = 30,
    tolerance: float = 1.0e-10,
    step: float = 1.0e-7,
) -> np.ndarray:
    """Central-difference oracle for the exact discrete stress update."""
    (
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        max_iterations,
        tolerance,
    ) = _validate_plane_stress_inputs(
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        max_iterations,
        tolerance,
    )
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")

    n_points = int(strain.shape[0])
    if curve is None:
        return np.broadcast_to(
            plane_stress_elastic_matrix(E, nu),
            (n_points, 3, 3),
        ).copy()
    tangent = np.zeros((n_points, 3, 3), dtype=float)
    elastic = plane_stress_elastic_matrix(E, nu)
    for col in range(3):
        # Keep the perturbation stress scale bounded when the elastic matrix
        # is highly conditioned (for example nu -> -1+). This preserves the
        # local derivative and avoids driving the perturbed return maps many
        # orders of magnitude away from the base state.
        column_scale = float(np.linalg.norm(elastic[:, col]))
        effective_step = step * min(
            1.0,
            max(abs(float(E)), 1.0) / max(column_scale, 1.0),
        )
        if not np.isfinite(effective_step) or effective_step <= 0.0:
            raise ValueError(
                "Could not construct a finite numerical-tangent perturbation"
            )
        strain_plus = strain.copy()
        strain_minus = strain.copy()
        strain_plus[:, col] = strain[:, col] + effective_step
        strain_minus[:, col] = strain[:, col] - effective_step
        unchanged_plus = strain_plus[:, col] == strain[:, col]
        unchanged_minus = strain_minus[:, col] == strain[:, col]
        strain_plus[unchanged_plus, col] = np.nextafter(
            strain[unchanged_plus, col],
            np.inf,
        )
        strain_minus[unchanged_minus, col] = np.nextafter(
            strain[unchanged_minus, col],
            -np.inf,
        )
        denominator = strain_plus[:, col] - strain_minus[:, col]
        if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
            raise ValueError(
                "Could not construct a representable numerical-tangent perturbation"
            )
        sigma_plus, _, _, _, plus_converged, plus_residual = _jit_plane_stress_return_map(
            strain_plus,
            plastic_strain,
            alpha,
            E,
            nu,
            float(curve.sigma_prop),
            float(curve.sigma_yield),
            float(curve.sigma_yield_2),
            float(curve.eps_p_y1),
            float(curve.eps_p_y2),
            float(curve.K),
            float(curve.n),
            float(curve._power_offset),
            max_iterations,
            tolerance,
            False,
        )
        _require_local_convergence(
            plus_converged,
            plus_residual,
            max_iterations=max_iterations,
        )
        sigma_minus, _, _, _, minus_converged, minus_residual = _jit_plane_stress_return_map(
            strain_minus,
            plastic_strain,
            alpha,
            E,
            nu,
            float(curve.sigma_prop),
            float(curve.sigma_yield),
            float(curve.sigma_yield_2),
            float(curve.eps_p_y1),
            float(curve.eps_p_y2),
            float(curve.K),
            float(curve.n),
            float(curve._power_offset),
            max_iterations,
            tolerance,
            False,
        )
        _require_local_convergence(
            minus_converged,
            minus_residual,
            max_iterations=max_iterations,
        )
        tangent[:, :, col] = (
            sigma_plus - sigma_minus
        ) / denominator[:, None]
    if np.any(~np.isfinite(tangent)):
        raise FloatingPointError("Numerical plane-stress tangent is non-finite")
    return tangent


def _finite_difference_algorithmic_tangent(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    E: float,
    nu: float,
    curve: "DNVC208MaterialCurve",
    max_iterations: int = 30,
    tolerance: float = 1.0e-10,
    step: float = 1.0e-7,
) -> np.ndarray:
    """Backward-compatible private alias for the numerical tangent oracle."""
    return plane_stress_numerical_tangent(
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        curve,
        max_iterations=max_iterations,
        tolerance=tolerance,
        step=step,
    )


def _hill48_strength_values(
    yield_model: "Hill48Yield | Any",
) -> Tuple[float, float, float, float, float, float]:
    """Return and validate the six symmetric Hill-48 strengths.

    ``yield_model`` is intentionally consumed by protocol rather than by a
    concrete runtime import.  This keeps the constitutive kernel independent
    of the model layer and lets an ANYmaterial object provide the same
    ``X``, ``Y``, ``Z``, ``S12``, ``S13`` and ``S23`` attributes.
    """
    names = ("X", "Y", "Z", "S12", "S13", "S23")
    values = []
    for name in names:
        try:
            value = float(getattr(yield_model, name))
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "yield_model must provide finite positive X, Y, Z, "
                "S12, S13 and S23 strengths"
            ) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Hill-48 strength {name} must be finite and positive"
            )
        values.append(value)

    X, Y, Z, S12, S13, S23 = values
    ratios = np.asarray(
        [X / Y, X / Z, X / S12, X / S13, X / S23],
        dtype=float,
    )
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise ValueError("Hill-48 strength ratios must be finite and positive")

    # Convexity is checked on the dimensionless normal-stress block.  It has
    # one intentional hydrostatic null mode; all other eigenvalues must be
    # nonnegative.  Checking the complete quadratic is less restrictive and
    # more accurate than requiring each of F, G and H separately to be
    # positive.
    xy2 = (X / Y) ** 2
    xz2 = (X / Z) ** 2
    F = 0.5 * (xy2 + xz2 - 1.0)
    G = 0.5 * (xz2 + 1.0 - xy2)
    H = 0.5 * (1.0 + xy2 - xz2)
    normal_metric = np.asarray(
        [
            [G + H, -H, -G],
            [-H, F + H, -F],
            [-G, -F, F + G],
        ],
        dtype=float,
    )
    eigenvalues = np.linalg.eigvalsh(normal_metric)
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1.0e-12 * eigenvalue_scale:
        raise ValueError(
            "Hill-48 strengths do not define a convex quadratic yield surface"
        )
    return X, Y, Z, S12, S13, S23


def hill48_coefficients(
    yield_model: "Hill48Yield | Any",
) -> Dict[str, float]:
    """Return the standard 3-D Hill-48 coefficients ``F`` through ``N``.

    The convention is

    ``F(s2-s3)^2 + G(s3-s1)^2 + H(s1-s2)^2
       + 2L*t23^2 + 2M*t13^2 + 2N*t12^2 = 1``.
    """
    X, Y, Z, S12, S13, S23 = _hill48_strength_values(yield_model)
    inv_x2 = 1.0 / X**2
    inv_y2 = 1.0 / Y**2
    inv_z2 = 1.0 / Z**2
    coefficients = {
        "F": 0.5 * (inv_y2 + inv_z2 - inv_x2),
        "G": 0.5 * (inv_z2 + inv_x2 - inv_y2),
        "H": 0.5 * (inv_x2 + inv_y2 - inv_z2),
        "L": 0.5 / S23**2,
        "M": 0.5 / S13**2,
        "N": 0.5 / S12**2,
    }
    if any(not np.isfinite(value) for value in coefficients.values()):
        raise ValueError("Hill-48 coefficients must be finite")
    return coefficients


def hill48_plane_stress_coefficients(
    yield_model: "Hill48Yield | Any",
) -> np.ndarray:
    """Return the material-axis plane-stress Hill quadratic matrix.

    For stress order ``[s11, s22, t12]``, the unscaled yield surface is
    ``stress.T @ A @ stress = 1``.  The matrix therefore has inverse-stress
    squared units.  :func:`hill48_plane_stress_equivalent_stress` multiplies
    its square root by ``X`` to return a stress-valued equivalent measure.
    """
    coefficients = hill48_coefficients(yield_model)
    X, Y, _Z, S12, _S13, _S23 = _hill48_strength_values(yield_model)
    H = coefficients["H"]
    return np.asarray(
        [
            [1.0 / X**2, -H, 0.0],
            [-H, 1.0 / Y**2, 0.0],
            [0.0, 0.0, 1.0 / S12**2],
        ],
        dtype=float,
    )


def hill48_equivalent_stress(
    stress: np.ndarray,
    yield_model: "Hill48Yield | Any",
) -> np.ndarray:
    """Return X-referenced 3-D Hill equivalent stress.

    Stress order is engineering Voigt ``[11, 22, 33, 23, 13, 12]``.  The
    function consumes the six-strength protocol directly so material records
    from ANYmaterial need not inherit ANYfeSolver classes.
    """

    values = np.asarray(stress, dtype=float)
    if values.ndim == 0 or values.shape[-1:] != (6,):
        raise ValueError("stress must have shape (..., 6)")
    if np.any(~np.isfinite(values)):
        raise ValueError("stress must contain only finite values")
    coefficients = hill48_coefficients(yield_model)
    s1, s2, s3, t23, t13, t12 = np.moveaxis(values, -1, 0)
    utilization_squared = (
        coefficients["F"] * (s2 - s3) ** 2
        + coefficients["G"] * (s3 - s1) ** 2
        + coefficients["H"] * (s1 - s2) ** 2
        + 2.0 * coefficients["L"] * t23**2
        + 2.0 * coefficients["M"] * t13**2
        + 2.0 * coefficients["N"] * t12**2
    )
    scale = np.maximum(np.sum(values * values, axis=-1), 1.0)
    if np.any(utilization_squared < -1.0e-12 * scale):
        raise FloatingPointError("Hill equivalent stress squared is negative")
    return float(yield_model.X) * np.sqrt(
        np.maximum(utilization_squared, 0.0)
    )


def _hill48_plane_stress_metric(
    yield_model: "Hill48Yield | Any",
) -> Tuple[np.ndarray, float]:
    """Return a well-scaled stress-valued equivalent-stress metric."""
    X, Y, Z, S12, _S13, _S23 = _hill48_strength_values(yield_model)
    h_scaled = 0.5 * (1.0 + (X / Y) ** 2 - (X / Z) ** 2)
    metric = np.asarray(
        [
            [1.0, -h_scaled, 0.0],
            [-h_scaled, (X / Y) ** 2, 0.0],
            [0.0, 0.0, (X / S12) ** 2],
        ],
        dtype=float,
    )
    metric = 0.5 * (metric + metric.T)
    eigenvalues = np.linalg.eigvalsh(metric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) <= 1.0e-14 * scale:
        raise ValueError(
            "Hill-48 strengths must define a positive-definite plane-stress "
            "quadratic"
        )
    return metric, X


def hill48_plane_stress_equivalent_stress(
    stress: np.ndarray,
    yield_model: "Hill48Yield | Any",
) -> np.ndarray:
    """Return the stress-valued Hill equivalent stress in material axes.

    The reference scale is the material-axis-1 strength ``X``.  Consequently
    the base yield condition is ``equivalent_stress == X``.  In the isotropic
    limit ``X=Y=Z=sy`` and ``S12=S13=S23=sy/sqrt(3)``, this is exactly the
    conventional plane-stress von Mises stress.
    """
    stress_array = np.asarray(stress, dtype=float)
    if stress_array.ndim == 0 or stress_array.shape[-1:] != (3,):
        raise ValueError("stress must have shape (..., 3)")
    if np.any(~np.isfinite(stress_array)):
        raise ValueError("stress must contain only finite values")
    metric, _reference_strength = _hill48_plane_stress_metric(yield_model)
    equivalent_squared = np.einsum(
        "...i,ij,...j->...",
        stress_array,
        metric,
        stress_array,
    )
    scale = np.maximum(
        np.einsum("...i,...i->...", stress_array, stress_array),
        1.0,
    )
    if np.any(equivalent_squared < -1.0e-12 * scale):
        raise FloatingPointError("Hill equivalent stress squared is negative")
    return np.sqrt(np.maximum(equivalent_squared, 0.0))


def _validate_hill48_plane_stress_inputs(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    elastic_matrix: np.ndarray,
    yield_model: "Hill48Yield | Any",
    max_iterations: int,
    tolerance: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
    float,
]:
    strain_array = np.asarray(strain, dtype=float)
    plastic_array = np.asarray(plastic_strain, dtype=float)
    alpha_array = np.asarray(alpha, dtype=float)
    if strain_array.ndim != 2 or strain_array.shape[1:] != (3,):
        raise ValueError("strain must have shape (n_points, 3)")
    if plastic_array.shape != strain_array.shape:
        raise ValueError("plastic_strain must have the same shape as strain")
    if alpha_array.shape != (strain_array.shape[0],):
        raise ValueError("alpha must have shape (n_points,)")
    if np.any(~np.isfinite(strain_array)) or np.any(~np.isfinite(plastic_array)):
        raise ValueError("strain and plastic_strain must contain only finite values")
    if np.any(~np.isfinite(alpha_array)):
        raise ValueError("alpha must contain only finite values")
    if np.any(alpha_array < 0.0):
        raise ValueError("alpha must be nonnegative")

    elastic = np.asarray(elastic_matrix, dtype=float)
    if elastic.shape != (3, 3):
        raise ValueError("elastic_matrix must have shape (3, 3)")
    if np.any(~np.isfinite(elastic)):
        raise ValueError("elastic_matrix must contain only finite values")
    elastic_scale = max(float(np.linalg.norm(elastic, ord=np.inf)), 1.0)
    if not np.allclose(
        elastic,
        elastic.T,
        rtol=1.0e-12,
        atol=1.0e-12 * elastic_scale,
    ):
        raise ValueError("elastic_matrix must be symmetric")
    elastic = 0.5 * (elastic + elastic.T)
    try:
        np.linalg.cholesky(elastic)
    except np.linalg.LinAlgError as exc:
        raise ValueError("elastic_matrix must be positive definite") from exc

    if isinstance(max_iterations, (bool, np.bool_)):
        raise ValueError("max_iterations must be a positive integer")
    iterations = int(max_iterations)
    if iterations <= 0 or float(iterations) != float(max_iterations):
        raise ValueError("max_iterations must be a positive integer")
    local_tolerance = float(tolerance)
    if not np.isfinite(local_tolerance) or local_tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    metric, reference_strength = _hill48_plane_stress_metric(yield_model)
    return (
        strain_array,
        plastic_array,
        alpha_array,
        elastic,
        metric,
        reference_strength,
        iterations,
        local_tolerance,
    )


def _curve_scalar_value(curve: Any, method_name: str, alpha: float) -> float:
    method = getattr(curve, method_name, None)
    if method is None or not callable(method):
        raise TypeError(f"hardening curve must provide {method_name}()")
    argument = np.asarray([float(alpha)], dtype=float)
    try:
        result = np.asarray(method(argument), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        result = np.asarray([method(float(alpha))], dtype=float).reshape(-1)
    if result.size != 1 or not np.isfinite(result[0]):
        raise ValueError(
            f"hardening curve {method_name}() must return one finite value"
        )
    return float(result[0])


def _hill48_reference_flow(curve: Any | None) -> float:
    if curve is None:
        return 1.0
    reference = _curve_scalar_value(curve, "flow_stress", 0.0)
    if reference <= 0.0:
        raise ValueError("hardening curve flow_stress(0) must be positive")
    return reference


def _hill48_flow_stress_and_modulus(
    alpha: float,
    reference_strength: float,
    curve: Any | None,
    reference_flow: float,
) -> Tuple[float, float]:
    if curve is None:
        return reference_strength, 0.0
    flow = _curve_scalar_value(curve, "flow_stress", alpha)
    if flow <= 0.0:
        raise ValueError("hardening curve flow stress must remain positive")
    hardening_method = getattr(curve, "hardening_modulus", None)
    if callable(hardening_method):
        hardening = _curve_scalar_value(curve, "hardening_modulus", alpha)
    else:
        derivative_step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(alpha))
        upper = _curve_scalar_value(
            curve,
            "flow_stress",
            alpha + derivative_step,
        )
        if alpha > derivative_step:
            lower = _curve_scalar_value(
                curve,
                "flow_stress",
                alpha - derivative_step,
            )
            hardening = (upper - lower) / (2.0 * derivative_step)
        else:
            hardening = (upper - flow) / derivative_step
    scaled_flow = reference_strength * flow / reference_flow
    scaled_hardening = reference_strength * hardening / reference_flow
    if not np.isfinite(scaled_flow) or not np.isfinite(scaled_hardening):
        raise ValueError("scaled Hill-48 flow stress and modulus must be finite")
    hardening_scale = max(abs(scaled_flow), reference_strength, 1.0)
    if scaled_hardening < -1.0e-12 * hardening_scale:
        raise ValueError("hardening curve must be nondecreasing")
    return scaled_flow, max(scaled_hardening, 0.0)


def _hill48_gamma_state(
    trial_stress: np.ndarray,
    elastic_matrix: np.ndarray,
    metric: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Evaluate stress, equivalent stress, normal and d(phi)/d(gamma)."""
    elastic_metric = elastic_matrix @ metric
    projection = np.eye(3, dtype=float) + float(gamma) * elastic_metric
    try:
        stress = np.linalg.solve(projection, trial_stress)
        stress_derivative = -np.linalg.solve(
            projection,
            elastic_metric @ stress,
        )
    except np.linalg.LinAlgError as exc:
        raise PlaneStressConvergenceError(
            "Hill-48 local projection matrix is singular"
        ) from exc
    metric_stress = metric @ stress
    equivalent_squared = float(stress @ metric_stress)
    if not np.isfinite(equivalent_squared) or equivalent_squared <= 0.0:
        raise PlaneStressConvergenceError(
            "Hill-48 local solve produced a nonpositive equivalent stress"
        )
    equivalent = float(np.sqrt(equivalent_squared))
    normal = metric_stress / equivalent
    equivalent_derivative = float(normal @ stress_derivative)
    return stress, equivalent, normal, equivalent_derivative


def _hill48_local_return(
    trial_stress: np.ndarray,
    alpha_n: float,
    elastic_matrix: np.ndarray,
    metric: np.ndarray,
    reference_strength: float,
    curve: Any | None,
    reference_flow: float,
    max_iterations: int,
    tolerance: float,
) -> Tuple[np.ndarray, float, float, np.ndarray, float]:
    """Safeguarded scalar backward-Euler return for one material point."""

    def evaluate(
        gamma: float,
    ) -> Tuple[
        Tuple[np.ndarray, float, np.ndarray, float, float, float, float],
        float,
    ]:
        stress, equivalent, normal, equivalent_derivative = (
            _hill48_gamma_state(
                trial_stress,
                elastic_matrix,
                metric,
                gamma,
            )
        )
        plastic_increment = gamma * equivalent
        alpha_value = alpha_n + plastic_increment
        flow, hardening = _hill48_flow_stress_and_modulus(
            alpha_value,
            reference_strength,
            curve,
            reference_flow,
        )
        residual = equivalent - flow
        plastic_increment_derivative = (
            equivalent + gamma * equivalent_derivative
        )
        residual_derivative = (
            equivalent_derivative
            - hardening * plastic_increment_derivative
        )
        return (
            stress,
            equivalent,
            normal,
            plastic_increment,
            flow,
            hardening,
            residual,
        ), residual_derivative

    lower = 0.0
    lower_state, _lower_derivative = evaluate(lower)
    lower_residual = float(lower_state[-1])
    residual_scale = max(
        abs(float(lower_state[1])),
        abs(float(lower_state[4])),
        1.0,
    )
    if lower_residual <= tolerance * residual_scale:
        stress, equivalent, normal, _increment, _flow, hardening, _residual = (
            lower_state
        )
        return stress, 0.0, alpha_n, normal, hardening

    upper = 1.0 / max(float(np.linalg.norm(elastic_matrix, ord=2)), 1.0)
    upper_state = lower_state
    for _ in range(128):
        upper_state, _upper_derivative = evaluate(upper)
        if float(upper_state[-1]) <= 0.0:
            break
        upper *= 2.0
    else:
        raise PlaneStressConvergenceError(
            "Could not bracket the Hill-48 plane-stress consistency root"
        )

    gamma = lower
    state = lower_state
    converged = False
    for _ in range(max_iterations):
        state, derivative = evaluate(gamma)
        residual = float(state[-1])
        residual_scale = max(
            abs(float(state[1])),
            abs(float(state[4])),
            1.0,
        )
        if abs(residual) <= tolerance * residual_scale:
            converged = True
            break
        if residual > 0.0:
            lower = gamma
        else:
            upper = gamma
        candidate = float("nan")
        if np.isfinite(derivative) and derivative < 0.0:
            candidate = gamma - residual / derivative
        if (
            not np.isfinite(candidate)
            or candidate <= lower
            or candidate >= upper
        ):
            candidate = 0.5 * (lower + upper)
        gamma = candidate

    if not converged:
        state, _derivative = evaluate(gamma)
        residual = abs(float(state[-1]))
        residual_scale = max(
            abs(float(state[1])),
            abs(float(state[4])),
            1.0,
        )
        if residual > tolerance * residual_scale:
            raise PlaneStressConvergenceError(
                "Hill-48 plane-stress return mapping did not satisfy the "
                "scaled yield residual after "
                f"{max_iterations} iterations; scaled residual="
                f"{residual / residual_scale:.6g}."
            )

    stress, _equivalent, normal, plastic_increment, _flow, hardening, _ = state
    return (
        stress,
        float(plastic_increment),
        float(alpha_n + plastic_increment),
        normal,
        float(hardening),
    )


def _hill48_return_map_core(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    elastic_matrix: np.ndarray,
    metric: np.ndarray,
    reference_strength: float,
    curve: Any | None,
    reference_flow: float,
    max_iterations: int,
    tolerance: float,
    compute_tangent: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_points = int(strain.shape[0])
    trial_stress = (strain - plastic_strain) @ elastic_matrix.T
    stress = trial_stress.copy()
    tangent = (
        np.broadcast_to(elastic_matrix, (n_points, 3, 3)).copy()
        if compute_tangent
        else np.zeros((n_points, 3, 3), dtype=float)
    )
    new_plastic = plastic_strain.copy()
    new_alpha = alpha.copy()
    tangent_valid = np.ones(n_points, dtype=bool)

    for point in range(n_points):
        flow_n, _hardening_n = _hill48_flow_stress_and_modulus(
            float(alpha[point]),
            reference_strength,
            curve,
            reference_flow,
        )
        equivalent_trial = float(
            np.sqrt(
                max(
                    float(trial_stress[point] @ metric @ trial_stress[point]),
                    0.0,
                )
            )
        )
        if (
            equivalent_trial - flow_n
            <= tolerance * max(flow_n, equivalent_trial, 1.0)
        ):
            continue

        (
            returned_stress,
            plastic_increment,
            alpha_value,
            normal,
            hardening,
        ) = _hill48_local_return(
            trial_stress[point],
            float(alpha[point]),
            elastic_matrix,
            metric,
            reference_strength,
            curve,
            reference_flow,
            max_iterations,
            tolerance,
        )
        stress[point] = returned_stress
        new_plastic[point] = plastic_strain[point] + plastic_increment * normal
        new_alpha[point] = alpha_value

        if not compute_tangent:
            continue
        equivalent = float(
            np.sqrt(returned_stress @ metric @ returned_stress)
        )
        hessian = (
            metric - np.outer(normal, normal)
        ) / equivalent
        jacobian = np.zeros((4, 4), dtype=float)
        jacobian[:3, :3] = (
            np.eye(3, dtype=float)
            + plastic_increment * elastic_matrix @ hessian
        )
        jacobian[:3, 3] = elastic_matrix @ normal
        jacobian[3, :3] = normal
        jacobian[3, 3] = -hardening
        right_hand_side = np.zeros((4, 3), dtype=float)
        right_hand_side[:3] = elastic_matrix
        try:
            local_derivative = np.linalg.solve(jacobian, right_hand_side)[:3]
        except np.linalg.LinAlgError:
            tangent_valid[point] = False
            tangent[point] = np.nan
            continue
        local_derivative = 0.5 * (
            local_derivative + local_derivative.T
        )
        elastic_norm = max(float(np.linalg.norm(elastic_matrix)), 1.0)
        tangent_valid[point] = bool(
            np.all(np.isfinite(local_derivative))
            and float(np.linalg.norm(local_derivative))
            <= _ANALYTICAL_TANGENT_AMPLIFICATION_LIMIT * elastic_norm
        )
        tangent[point] = local_derivative
    return stress, tangent, new_plastic, new_alpha, tangent_valid


def _hill48_return_map_dispatch(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    elastic_matrix: np.ndarray,
    metric: np.ndarray,
    reference_strength: float,
    curve: Any | None,
    reference_flow: float,
    max_iterations: int,
    tolerance: float,
    compute_tangent: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """Select the compiled canonical path or the exact scalar oracle.

    Public validation has already completed before this function is called.
    Unsupported custom curve protocols, unavailable JIT support, kernel-level
    failures, and failed individual rows all remain explicit scalar fallbacks.
    """

    from . import vectorized_hill48 as accelerated

    curve_pack, eligibility_reason = accelerated.pack_hill48_curve(curve)
    curve_name = (
        curve_pack.name
        if curve_pack is not None
        else type(curve).__name__ if curve is not None else "perfect_plasticity"
    )
    point_count = int(strain.shape[0])
    metadata: Dict[str, Any] = {
        "point_count": point_count,
        "curve_name": curve_name,
        "compiled": False,
        "scalar_fallback_points": 0,
        "fallback_reason_counts": {},
    }

    fallback_reason = eligibility_reason
    if fallback_reason is None and not accelerated.JIT_ENABLED:
        fallback_reason = "jit_unavailable"
    if fallback_reason is not None:
        metadata["scalar_fallback_points"] = point_count
        metadata["fallback_reason_counts"] = {fallback_reason: point_count}
        try:
            result = _hill48_return_map_core(
                strain,
                plastic_strain,
                alpha,
                elastic_matrix,
                metric,
                reference_strength,
                curve,
                reference_flow,
                max_iterations,
                tolerance,
                compute_tangent,
            )
        except Exception:
            accelerated.record_hill48_execution(**metadata)
            raise
        return (*result, metadata)

    assert curve_pack is not None
    metadata["compiled"] = True
    try:
        (
            stress,
            tangent,
            new_plastic,
            new_alpha,
            status,
            _scaled_residual,
            tangent_valid,
        ) = accelerated.compiled_hill48_return_map(
            strain,
            plastic_strain,
            alpha,
            elastic_matrix,
            metric,
            reference_strength,
            reference_flow,
            curve_pack,
            max_iterations,
            tolerance,
            _ANALYTICAL_TANGENT_AMPLIFICATION_LIMIT,
            compute_tangent,
        )
    except Exception:
        reason = "compiled_kernel_exception"
        metadata["scalar_fallback_points"] = point_count
        metadata["fallback_reason_counts"] = {reason: point_count}
        try:
            result = _hill48_return_map_core(
                strain,
                plastic_strain,
                alpha,
                elastic_matrix,
                metric,
                reference_strength,
                curve,
                reference_flow,
                max_iterations,
                tolerance,
                compute_tangent,
            )
        except Exception:
            accelerated.record_hill48_execution(**metadata)
            raise
        return (*result, metadata)

    failed = np.flatnonzero(np.asarray(status, dtype=np.int8) != 0)
    if failed.size:
        reasons: Dict[str, int] = {}
        for code in np.asarray(status, dtype=np.int8)[failed]:
            reason = accelerated.status_fallback_reason(int(code))
            reasons[reason] = reasons.get(reason, 0) + 1
        metadata["scalar_fallback_points"] = int(failed.size)
        metadata["fallback_reason_counts"] = reasons
        try:
            (
                fallback_stress,
                fallback_tangent,
                fallback_plastic,
                fallback_alpha,
                fallback_tangent_valid,
            ) = _hill48_return_map_core(
                strain[failed],
                plastic_strain[failed],
                alpha[failed],
                elastic_matrix,
                metric,
                reference_strength,
                curve,
                reference_flow,
                max_iterations,
                tolerance,
                compute_tangent,
            )
        except Exception:
            accelerated.record_hill48_execution(**metadata)
            raise
        stress[failed] = fallback_stress
        tangent[failed] = fallback_tangent
        new_plastic[failed] = fallback_plastic
        new_alpha[failed] = fallback_alpha
        tangent_valid[failed] = fallback_tangent_valid
    return stress, tangent, new_plastic, new_alpha, tangent_valid, metadata


def hill48_plane_stress_return_map(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    elastic_matrix: np.ndarray,
    yield_model: "Hill48Yield | Any",
    curve: Any | None = None,
    max_iterations: int = 50,
    tolerance: float = 1.0e-10,
    compute_tangent: bool = True,
    tangent_method: str = "analytical",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-Euler associated Hill-48 update in material axes.

    Inputs use ``[e11, e22, gamma12]`` and ``[s11, s22, t12]`` engineering
    order.  ``elastic_matrix`` may be any symmetric positive-definite 3-by-3
    plane-stress matrix, including an off-axis orthotropic matrix.  Plastic
    strain and Hill equivalent plastic strain ``alpha`` are state at the
    beginning of the increment.

    Directional strengths are scaled by
    ``curve.flow_stress(alpha) / curve.flow_stress(0)``.  With ``curve=None``
    the response is elastic-perfectly-plastic at the supplied strengths.
    The default tangent is the exact derivative of the converged local
    residual; invalid rows fall back to the central-difference oracle.
    """
    (
        strain,
        plastic_strain,
        alpha,
        elastic_matrix,
        metric,
        reference_strength,
        max_iterations,
        tolerance,
    ) = _validate_hill48_plane_stress_inputs(
        strain,
        plastic_strain,
        alpha,
        elastic_matrix,
        yield_model,
        max_iterations,
        tolerance,
    )
    override = _TANGENT_METHOD_OVERRIDE.get()
    method = (
        _normalize_tangent_method(override)
        if override is not None
        else _normalize_tangent_method(tangent_method)
    )
    reference_flow = _hill48_reference_flow(curve)
    (
        stress,
        tangent,
        new_plastic,
        new_alpha,
        tangent_valid,
        execution_metadata,
    ) = _hill48_return_map_dispatch(
        strain,
        plastic_strain,
        alpha,
        elastic_matrix,
        metric,
        reference_strength,
        curve,
        reference_flow,
        max_iterations,
        tolerance,
        compute_tangent and method == "analytical",
    )
    from .vectorized_hill48 import record_hill48_execution

    if not compute_tangent:
        record_hill48_execution(**execution_metadata)
        return stress, tangent, new_plastic, new_alpha
    if method == "numerical":
        tangent = hill48_plane_stress_numerical_tangent(
            strain,
            plastic_strain,
            alpha,
            elastic_matrix,
            yield_model,
            curve,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        record_hill48_execution(
            **execution_metadata,
            numerical_tangent_rows=int(strain.shape[0]),
        )
        return stress, tangent, new_plastic, new_alpha
    if np.any(~tangent_valid):
        fallback_count = int(np.count_nonzero(~tangent_valid))
        reasons = dict(execution_metadata["fallback_reason_counts"])
        reasons["analytical_tangent_invalid"] = (
            int(reasons.get("analytical_tangent_invalid", 0)) + fallback_count
        )
        execution_metadata["fallback_reason_counts"] = reasons
        tangent[~tangent_valid] = hill48_plane_stress_numerical_tangent(
            strain[~tangent_valid],
            plastic_strain[~tangent_valid],
            alpha[~tangent_valid],
            elastic_matrix,
            yield_model,
            curve,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    if np.any(~np.isfinite(tangent)):
        record_hill48_execution(**execution_metadata)
        raise FloatingPointError(
            "Hill-48 plane-stress tangent remained non-finite after the "
            "numerical fallback"
        )
    record_hill48_execution(**execution_metadata)
    return stress, tangent, new_plastic, new_alpha


def hill48_plane_stress_numerical_tangent(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    elastic_matrix: np.ndarray,
    yield_model: "Hill48Yield | Any",
    curve: Any | None = None,
    max_iterations: int = 50,
    tolerance: float = 1.0e-10,
    step: float = 1.0e-7,
) -> np.ndarray:
    """Central-difference oracle for the discrete Hill-48 stress update."""
    (
        strain,
        plastic_strain,
        alpha,
        elastic_matrix,
        _metric,
        _reference_strength,
        max_iterations,
        tolerance,
    ) = _validate_hill48_plane_stress_inputs(
        strain,
        plastic_strain,
        alpha,
        elastic_matrix,
        yield_model,
        max_iterations,
        tolerance,
    )
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    tangent = np.zeros((strain.shape[0], 3, 3), dtype=float)
    for column in range(3):
        strain_plus = strain.copy()
        strain_minus = strain.copy()
        strain_plus[:, column] += step
        strain_minus[:, column] -= step
        unchanged_plus = strain_plus[:, column] == strain[:, column]
        unchanged_minus = strain_minus[:, column] == strain[:, column]
        strain_plus[unchanged_plus, column] = np.nextafter(
            strain[unchanged_plus, column],
            np.inf,
        )
        strain_minus[unchanged_minus, column] = np.nextafter(
            strain[unchanged_minus, column],
            -np.inf,
        )
        denominator = (
            strain_plus[:, column] - strain_minus[:, column]
        )
        if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
            raise ValueError(
                "Could not construct a representable numerical-tangent "
                "perturbation"
            )
        stress_plus, _tangent, _plastic, _alpha = (
            hill48_plane_stress_return_map(
                strain_plus,
                plastic_strain,
                alpha,
                elastic_matrix,
                yield_model,
                curve,
                max_iterations=max_iterations,
                tolerance=tolerance,
                compute_tangent=False,
            )
        )
        stress_minus, _tangent, _plastic, _alpha = (
            hill48_plane_stress_return_map(
                strain_minus,
                plastic_strain,
                alpha,
                elastic_matrix,
                yield_model,
                curve,
                max_iterations=max_iterations,
                tolerance=tolerance,
                compute_tangent=False,
            )
        )
        tangent[:, :, column] = (
            stress_plus - stress_minus
        ) / denominator[:, None]
    if np.any(~np.isfinite(tangent)):
        raise FloatingPointError(
            "Numerical Hill-48 plane-stress tangent is non-finite"
        )
    return tangent


_LOBATTO_RULES = {
    3: (np.array([-1.0, 0.0, 1.0]), np.array([1.0, 4.0, 1.0]) / 3.0),
    5: (
        np.array([-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0]),
        np.array([1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0]),
    ),
    7: (
        np.array(
            [-1.0, -0.830223896278567, -0.468848793470714, 0.0,
             0.468848793470714, 0.830223896278567, 1.0]
        ),
        np.array(
            [0.047619047619048, 0.276826047361566, 0.431745381209863, 0.487619047619048,
             0.431745381209863, 0.276826047361566, 0.047619047619048]
        ),
    ),
    9: (
        np.array(
            [-1.0, -0.899757995411460, -0.677186279510738, -0.363117463826178, 0.0,
             0.363117463826178, 0.677186279510738, 0.899757995411460, 1.0]
        ),
        np.array(
            [0.027777777777778, 0.165495361560806, 0.274538712500162, 0.346428510973046,
             0.371519274376417, 0.346428510973046, 0.274538712500162, 0.165495361560806,
             0.027777777777778]
        ),
    ),
    11: (
        np.array(
            [-1.0, -0.934001430408059, -0.784483473663144, -0.565235326996205,
             -0.295758135586939, 0.0, 0.295758135586939, 0.565235326996205,
             0.784483473663144, 0.934001430408059, 1.0]
        ),
        np.array(
            [0.018181818181818, 0.109612273266995, 0.187169881780305, 0.248048104264028,
             0.286879124779008, 0.300217595455691, 0.286879124779008, 0.248048104264028,
             0.187169881780305, 0.109612273266995, 0.018181818181818]
        ),
    ),
}


def lobatto_layers(num_layers: int, thickness: float) -> Tuple[np.ndarray, np.ndarray]:
    """Through-thickness Gauss-Lobatto coordinates and weights.

    Lobatto rules include the surface points, where yielding starts first.
    Returns (z, w) with z in [-h/2, h/2] and sum(w) = h.
    """
    if num_layers not in _LOBATTO_RULES:
        raise ValueError(f"num_layers must be one of {sorted(_LOBATTO_RULES)}")
    points, weights = _LOBATTO_RULES[num_layers]
    return 0.5 * thickness * points, 0.5 * thickness * weights
