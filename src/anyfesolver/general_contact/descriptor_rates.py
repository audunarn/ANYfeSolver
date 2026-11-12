"""Directional rate solve for affine penalty contact and massless coordinates.

This is an internal opt-in development kernel. It gives the right directional
derivative of an algebraic force equation at contact touch; it does not locate
events or certify a reconstructed transient path.
"""

import numpy as np


def directional_descriptor_rate(structural_tangent, null_basis, seed,
                                gap_gradient, weights, fixed_active,
                                directional, *, offset=None, maximum_iterations=20):
    """Solve Z.T (Ks*r + B.T*w*Dmin(B*r) + offset)=0 at fixed P*r.

    ``fixed_active`` denotes smooth closed constraints. ``directional`` denotes
    touch constraints whose branch depends on this rate. The two masks must be
    disjoint. At directional zero choose the closed generalized Jacobian, while
    the force derivative itself remains zero. A finite active-set budget never
    changes the law. Physical projection is Euclidean onto the complement of
    the certified mass-null basis, and no mass is assigned to that basis.
    """
    K = np.asarray(structural_tangent, dtype=float)
    Z = np.asarray(null_basis, dtype=float)
    B = np.asarray(gap_gradient, dtype=float)
    w = np.asarray(weights, dtype=float)
    fixed = np.asarray(fixed_active, dtype=bool)
    switching = np.asarray(directional, dtype=bool)
    rate = np.asarray(seed, dtype=float)
    rhs_offset = np.zeros_like(rate) if offset is None else np.asarray(offset, dtype=float)
    n = rate.size
    if (rate.ndim != 1 or K.shape != (n, n) or Z.ndim != 2 or Z.shape[0] != n
            or B.ndim != 2 or B.shape[1] != n or w.shape != (B.shape[0],)
            or fixed.shape != w.shape or switching.shape != w.shape
            or rhs_offset.shape != (n,) or np.any(fixed & switching)
            or np.any(w <= 0.) or type(maximum_iterations) is not int
            or maximum_iterations < 1
            or any(not np.all(np.isfinite(v)) for v in (K, Z, B, w, rate, rhs_offset))):
        raise ValueError("invalid directional descriptor rate data")
    try:
        physical = rate-Z @ np.linalg.solve(Z.T @ Z, Z.T @ rate)
        active = fixed | (switching & (B @ rate <= 0.))
        seen = set()
        for _ in range(maximum_iterations):
            key = active.tobytes()
            if key in seen:
                raise ValueError("directional descriptor active set cycled")
            seen.add(key)
            tangent = K+B[active].T @ (w[active, None]*B[active])
            candidate = physical+Z @ np.linalg.solve(
                Z.T @ tangent @ Z, -Z.T @ (tangent @ physical+rhs_offset),
            )
            updated = fixed | (switching & (B @ candidate <= 0.))
            if np.array_equal(active, updated):
                if not np.all(np.isfinite(candidate)):
                    raise ValueError("nonfinite directional descriptor rate")
                return candidate
            rate, active = candidate, updated
    except np.linalg.LinAlgError as exc:
        raise ValueError("directional descriptor algebraic solve is singular") from exc
    raise ValueError("directional descriptor active-set limit exhausted")
