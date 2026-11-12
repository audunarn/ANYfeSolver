"""Initial-contact seed: one origin-linearized KKT pressure proposal.

This module owns the opt-in ``initial_contact_seed='linearized_kkt_dev'``
candidate. At a declared cold-start origin it builds ONE factorization of the
control-constrained structural map

    [[K_c, -C], [control_rows, 0]]

and reuses that single factorization for the contact-free constrained
response and for every supported pressure right-hand side. The equilibrated
gap response ``H = G K_c^{-1} G.T`` (pressure force z has N, normalized gap
has m, so H has m/N) and the free-response gap ``g_free`` then define the
nonnegative pressure QP

    min 0.5 z.T H z + g_free.T z   subject to   z >= 0,

whose KKT conditions are exactly the linearized complementarity of the same
residual convention the Newton path uses (R = R_struct - G.T z). The result
is ONLY an initial proposed (q, control reactions, z) for the unchanged
Newton iteration; it is never an accepted equilibrium, history or commit.

A singular, indefinite, ill-conditioned or asymmetric operator refuses the
seed explicitly. No artificial diagonal, stiffness or regularization is
added, and no mode with genuinely positive support is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse

from anyfesolver.linalg import MatrixClass, factorize

from .core import (_INITIAL_CONTACT_SEED_REVISION,
                   _INITIAL_CONTACT_SEED_GLOBALIZATION_REVISION)

#: Explicit cap on the supported pressure-mode count admitted by the seed.
MAXIMUM_SEED_PRESSURE_MODES = 512
#: Relative backward-error cap for the single constrained factorization.
SEED_BACKWARD_ERROR_TOLERANCE = 1.0e-8
#: Relative cap on the asymmetry of the equilibrated gap response.
SEED_GAP_RESPONSE_ASYMMETRY_TOLERANCE = 1.0e-9
#: Relative cap on negative eigenvalues of the symmetrized gap response.
SEED_GAP_RESPONSE_INDEFINITENESS_TOLERANCE = 1.0e-9
#: Relative KKT optimality tolerance of the bounded internal QP solver.
SEED_QP_OPTIMALITY_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class InitialContactSeedProposal:
    """Proposed initial iterates; carries no equilibrium or history authority."""

    q: np.ndarray
    reactions: np.ndarray
    nodal_forces: np.ndarray
    diagnostics: dict

    @property
    def refused(self) -> bool:
        return False


@dataclass(frozen=True)
class InitialContactSeedRefusal:
    """Explicit refusal with an untouched origin."""

    reason: str
    diagnostics: dict

    @property
    def refused(self) -> bool:
        return True


def _bounded_nonnegative_qp(h: np.ndarray, g: np.ndarray, *,
                            iteration_cap: int,
                            release_alternatives: bool = False):
    """Bounded primal active-set solve of min 0.5 z'Hz + g'z, z >= 0.

    The route keeps a working set of indices fixed at the zero bound, releases
    the most negative-gradient bound index, solves the stationary system on
    the free face exactly and line-searches to the first blocking clamp. It
    never regularizes: a singular free face or a degenerate zero step refuses
    explicitly. With ``release_alternatives`` the v2 rule changes the
    sequencing only: a bound index is released just when the current
    iterate is already stationary on the current free face (releasing
    while the free face is nonstationary is the cycling driver on the
    original operator), and a release candidate whose enlarged-face
    minimizer keeps it at zero is re-clamped and excluded for the
    current face so the next candidate is released instead. Optimality
    is certified ONLY from the full unmasked KKT residual — no bound
    gradient below -tolerance and no free gradient magnitude above
    tolerance — never from a release mask or a repeated-face fixed
    point; a fixed point without that residual support terminates
    honestly as ``kkt_fixed_point`` and the caller refuses on the
    reported residual. Returns ``(z, iterations, terminal)`` with
    terminal one of ``optimal``, ``iteration_cap``, ``singular_free_face``,
    ``degenerate_step``, ``kkt_fixed_point``.
    """
    modes = len(g)
    z = np.zeros(modes)
    working = np.ones(modes, dtype=bool)
    blocked = np.zeros(modes, dtype=bool)
    scale = max(1.0, float(np.max(np.abs(g), initial=0.0)),
                float(np.max(np.abs(h), initial=0.0)))
    tolerance = SEED_QP_OPTIMALITY_TOLERANCE*scale
    iterations = 0
    for iterations in range(1, iteration_cap + 1):
        gradient = h @ z + g
        free = ~working
        free_residual = (float(np.max(np.abs(gradient[free]), initial=0.0))
                         if np.any(free) else 0.0)
        bound_violation = working & (gradient < -tolerance)
        if not np.any(bound_violation) and free_residual <= tolerance:
            return z, iterations - 1, "optimal"
        released = False
        entering = -1
        if np.any(bound_violation):
            if not release_alternatives:
                entering = int(np.argmin(np.where(bound_violation, gradient,
                                                 np.inf)))
            elif (free_residual <= tolerance
                    and np.any(bound_violation & ~blocked)):
                entering = int(np.argmin(np.where(bound_violation & ~blocked,
                                                  gradient, np.inf)))
            if entering >= 0:
                working[entering] = False
                released = True
                free = ~working
        try:
            face = h[np.ix_(free, free)]
            if not np.all(np.isfinite(face)):
                return z, iterations, "singular_free_face"
            target = np.linalg.solve(face, -g[free])
        except np.linalg.LinAlgError:
            return z, iterations, "singular_free_face"
        if not np.all(np.isfinite(target)):
            return z, iterations, "singular_free_face"
        if np.all(target >= 0.0):
            candidate = np.zeros(modes)
            candidate[free] = target
            if release_alternatives and np.array_equal(candidate, z):
                if released:
                    # Degenerate release: the enlarged-face minimizer keeps
                    # the entering index at zero. Re-clamp it and exclude it
                    # for the current face; never certify optimal here.
                    working[entering] = True
                    blocked[entering] = True
                    continue
                # Repeated-face fixed point without full unmasked KKT
                # support (a negative bound gradient remains): honest
                # terminal, never a false optimal.
                return z, iterations, "kkt_fixed_point"
            z = candidate
            if release_alternatives:
                blocked[:] = False
            continue
        # First blocking clamp between the current z and the face target.
        direction = np.zeros(modes)
        direction[free] = target - z[free]
        blocking = free & (direction < 0.0) & (z > 0.0)
        if not np.any(blocking):
            # The entering variable itself blocks at zero: keep it clamped.
            entering = int(np.argmin(np.where(
                free & (direction < 0.0), direction, np.inf)))
            working[entering] = True
            if release_alternatives:
                blocked[entering] = True
            continue
        fractions = z[blocking]/(-direction[blocking])
        index = int(np.flatnonzero(blocking)[int(np.argmin(fractions))])
        alpha = min(1.0, float(np.min(fractions)))
        if alpha <= 0.0 or not np.isfinite(alpha):
            return z, iterations, "degenerate_step"
        z = z + alpha*direction
        z[index] = 0.0
        working[index] = True
        if release_alternatives:
            blocked[:] = False
    return z, iterations, "iteration_cap"


def propose_linearized_kkt_seed(*, kqq, kqz, gap_q, control_reactions,
                                control_rows, residual, constraint, q,
                                normalized_gap, support, signature,
                                selection="linearized_kkt_dev"):
    """Propose the initial (q, control reactions, z) at the committed origin.

    All blocks use the same free-coordinate convention as the semismooth
    Newton path: ``kqq`` is the constrained structural tangent,
    ``control_reactions``/``control_rows`` map control multipliers and rows,
    ``gap_q`` is the normalized-gap gradient (m x n) and ``kqz`` the pressure
    force block of the Newton system, so the pressure right-hand sides are
    exactly ``-kqz`` columns and the signs derive from the live residual
    ``R = R_struct - G.T z``. The ``linearized_kkt_dev`` selection keeps the
    exact v1 bounded QP release rule; the
    ``linearized_kkt_merit_globalization_dev`` selection additionally enables
    the bounded anti-cycling release alternatives of the internal QP so the
    complete linearized correction exists for the caller's merit
    globalization trials.
    """
    started = time.perf_counter()
    revision = (_INITIAL_CONTACT_SEED_GLOBALIZATION_REVISION
                if selection == "linearized_kkt_merit_globalization_dev"
                else _INITIAL_CONTACT_SEED_REVISION)
    release_alternatives = (
        selection == "linearized_kkt_merit_globalization_dev")
    q = np.asarray(q, dtype=float).reshape(-1)
    residual = np.asarray(residual, dtype=float).reshape(-1)
    constraint = np.asarray(constraint, dtype=float).reshape(-1)
    normalized_gap = np.asarray(normalized_gap, dtype=float).reshape(-1)
    support = np.asarray(support, dtype=float).reshape(-1)
    gap_q = np.asarray(gap_q.todense()
                       if sparse.issparse(gap_q) else gap_q, dtype=float)
    modes = len(normalized_gap)
    if (gap_q.shape != (modes, len(q)) or support.shape != (modes,)
            or residual.shape != kqq.shape[0:1] or modes == 0):
        return InitialContactSeedRefusal("inconsistent_seed_blocks", {
            "revision": revision,
            "selection": selection,
            "gap_gradient_shape": list(gap_q.shape),
            "pressure_modes": int(modes)})
    supported = support > 0.0
    supported_modes = int(np.count_nonzero(supported))
    diagnostics = {
        "revision": revision,
        "selection": selection,
        "qp_release_alternatives": bool(release_alternatives),
        "total_pressure_modes": int(modes),
        "supported_pressure_modes": supported_modes,
    }
    if supported_modes == 0:
        return InitialContactSeedRefusal("no_supported_pressure_modes",
                                        dict(diagnostics))
    if supported_modes > MAXIMUM_SEED_PRESSURE_MODES:
        return InitialContactSeedRefusal("pressure_mode_cap_exceeded",
                                        dict(diagnostics))
    if not (np.all(np.isfinite(kqq.data)) and np.all(np.isfinite(residual))
            and np.all(np.isfinite(constraint))
            and np.all(np.isfinite(normalized_gap))
            and np.all(np.isfinite(gap_q))):
        return InitialContactSeedRefusal("nonfinite_seed_blocks",
                                        dict(diagnostics))
    size = kqq.shape[0]
    controls = np.asarray(control_reactions, dtype=float)
    rows = np.asarray(control_rows, dtype=float)
    if (controls.shape != (size, len(constraint))
            or rows.shape != (len(constraint), size)):
        return InitialContactSeedRefusal("inconsistent_seed_blocks",
                                        dict(diagnostics))
    zero_controls = sparse.csr_matrix((len(constraint), len(constraint)))
    operator = sparse.bmat((
        (sparse.csr_matrix(kqq), sparse.csr_matrix(-controls)),
        (sparse.csr_matrix(rows), zero_controls),
    ), format="csr")
    diagnostics["constrained_operator_dofs"] = int(size + len(constraint))
    try:
        handle = factorize(operator, MatrixClass.GENERAL,
                           signature=f"{signature}:constrained")
    except Exception as exc:
        diagnostics["detail"] = f"{type(exc).__name__}: {exc}"[:200]
        return InitialContactSeedRefusal("constrained_factorization_failed",
                                        diagnostics)
    if handle.status != "ok":
        diagnostics["detail"] = str(handle.failure_reason)[:200]
        return InitialContactSeedRefusal("constrained_factorization_failed",
                                        diagnostics)
    diagnostics["constrained_factorizations"] = 1

    def solve_columns(rhs):
        solved = np.asarray(handle.solve(np.asarray(rhs, dtype=float)),
                            dtype=float).reshape(-1, len(rhs[0]))
        backward = np.asarray(operator @ solved - np.asarray(rhs, dtype=float))
        error = float(np.max(np.abs(backward), initial=0.0))
        reference = float(np.max(np.abs(np.asarray(rhs, dtype=float)),
                                initial=0.0))
        return solved, error/max(1.0, reference)

    free_rhs = np.zeros((size + len(constraint), 1))
    free_rhs[:size, 0] = residual
    free_rhs[size:, 0] = constraint
    try:
        free, free_error = solve_columns(free_rhs)
    except Exception as exc:
        diagnostics["detail"] = f"{type(exc).__name__}: {exc}"[:200]
        return InitialContactSeedRefusal("constrained_solve_failed",
                                        diagnostics)
    pressure_rhs = np.zeros((size + len(constraint), supported_modes))
    pressure_rhs[:size, :] = (-np.asarray(
        kqz.todense() if sparse.issparse(kqz) else kqz, dtype=float))[:, supported]
    try:
        pressure, pressure_error = solve_columns(pressure_rhs)
    except Exception as exc:
        diagnostics["detail"] = f"{type(exc).__name__}: {exc}"[:200]
        return InitialContactSeedRefusal("constrained_solve_failed",
                                        diagnostics)
    diagnostics["constrained_solves"] = 1 + supported_modes
    backward = max(free_error, pressure_error)
    diagnostics["constrained_backward_error"] = backward
    if not np.all(np.isfinite(free)) or not np.all(np.isfinite(pressure)):
        return InitialContactSeedRefusal("nonfinite_constrained_response",
                                        diagnostics)
    if backward > SEED_BACKWARD_ERROR_TOLERANCE:
        return InitialContactSeedRefusal("ill_conditioned_constrained_operator",
                                        diagnostics)
    dq_free = free[:size, 0]
    dmu_free = free[size:, 0]
    dq_pressure = pressure[:size, :]
    dmu_pressure = pressure[size:, :]
    g_free = normalized_gap[supported] + gap_q[supported] @ dq_free
    h = gap_q[supported] @ dq_pressure
    asymmetry = float(np.max(np.abs(h - h.T), initial=0.0))
    scale = max(1.0, float(np.max(np.abs(h), initial=0.0)))
    diagnostics["gap_response_asymmetry"] = asymmetry
    if asymmetry > SEED_GAP_RESPONSE_ASYMMETRY_TOLERANCE*scale:
        return InitialContactSeedRefusal("asymmetric_gap_response",
                                        diagnostics)
    h = 0.5*(h + h.T)
    eigenvalues = np.linalg.eigvalsh(h)
    diagnostics["gap_response_eigenvalue_min"] = float(eigenvalues[0])
    diagnostics["gap_response_eigenvalue_max"] = float(eigenvalues[-1])
    if eigenvalues[0] < -SEED_GAP_RESPONSE_INDEFINITENESS_TOLERANCE*scale:
        return InitialContactSeedRefusal("indefinite_gap_response",
                                        diagnostics)
    diagnostics["free_gap_min_m"] = float(np.min(g_free))
    z, qp_iterations, terminal = _bounded_nonnegative_qp(
        h, g_free, iteration_cap=4*supported_modes + 16,
        release_alternatives=release_alternatives)
    diagnostics["qp_iterations"] = int(qp_iterations)
    diagnostics["qp_terminal"] = terminal
    # Computed full KKT/gradient residual of the returned iterate, reported
    # for every terminal: an `optimal` terminal certifies exactly this
    # residual (no bound gradient below -tolerance, no free gradient
    # magnitude above tolerance); any other terminal reports the honest
    # residual the caller refuses on.
    qp_gradient = h @ z + g_free
    diagnostics["qp_kkt_gradient_max_abs"] = float(
        np.max(np.abs(qp_gradient), initial=0.0))
    diagnostics["qp_kkt_gradient_min"] = float(np.min(qp_gradient))
    if terminal != "optimal":
        return InitialContactSeedRefusal(f"qp_{terminal}", diagnostics)
    if not np.all(np.isfinite(z)) or np.any(z < 0.0):
        return InitialContactSeedRefusal("nonfinite_seed_pressure", diagnostics)
    nodal_forces = np.zeros(modes)
    nodal_forces[supported] = z
    seed_q = q + dq_free + dq_pressure @ z
    seed_reactions = dmu_free + dmu_pressure @ z
    if not (np.all(np.isfinite(seed_q))
            and np.all(np.isfinite(seed_reactions))):
        return InitialContactSeedRefusal("nonfinite_seed_candidate",
                                        diagnostics)
    diagnostics["seed_pressure_force_n"] = float(np.linalg.norm(z))
    diagnostics["seed_reaction_norm"] = float(np.linalg.norm(seed_reactions))
    diagnostics["seed_displacement_norm_m"] = float(
        np.linalg.norm(dq_free + dq_pressure @ z))
    diagnostics["cost_s"] = float(time.perf_counter() - started)
    return InitialContactSeedProposal(
        np.asarray(seed_q, dtype=float).copy(),
        np.asarray(seed_reactions, dtype=float).copy(),
        np.asarray(nodal_forces, dtype=float).copy(),
        diagnostics)


__all__ = ["InitialContactSeedProposal", "InitialContactSeedRefusal",
           "MAXIMUM_SEED_PRESSURE_MODES", "propose_linearized_kkt_seed"]
