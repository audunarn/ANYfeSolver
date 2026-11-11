"""Block elimination for one frozen monolithic contact Newton linearization.

The structural/control and contact Jacobian is unchanged. This helper only
reuses its structural/control factorization while the semismooth active set
changes. It is a bounded development backend for the same coupled equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from anyfesolver.linalg import MatrixClass, factorize


@dataclass(frozen=True)
class FrozenBlockResponse:
    """One factorized structural/control response and pressure Schur block."""

    base_direction: np.ndarray
    force_response: np.ndarray
    gap_response: np.ndarray
    gap_from_base: np.ndarray
    q_size: int
    mu_size: int
    structural_control: sparse.csr_matrix
    contact_columns: sparse.csr_matrix
    gap_q: np.ndarray
    right_hand_side: np.ndarray
    absolute_structural_control: sparse.csr_matrix
    absolute_contact_columns: sparse.csr_matrix
    absolute_gap_q: np.ndarray

    def direction(self, active: np.ndarray, nodal_forces: np.ndarray,
                  gap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active = np.asarray(active, dtype=bool)
        z = np.asarray(nodal_forces, dtype=float)
        g = np.asarray(gap, dtype=float)
        modes = self.force_response.shape[1]
        if any(item.shape != (modes,) for item in (active, z, g)):
            raise ValueError("block active-set vectors have incompatible sizes")
        dz = -z.copy()
        if np.any(active):
            inactive = ~active
            right = g[active] + self.gap_from_base[active]
            if np.any(inactive):
                right -= self.gap_response[np.ix_(active, inactive)] @ dz[inactive]
            dz[active] = np.linalg.solve(
                self.gap_response[np.ix_(active, active)], right,
            )
        state_direction = self.base_direction - self.force_response @ dz
        structural_error = (self.structural_control @ state_direction
                            + self.contact_columns @ dz - self.right_hand_side)
        structural_scale = (self.absolute_structural_control @ abs(state_direction)
                            + self.absolute_contact_columns @ abs(dz)
                            + abs(self.right_hand_side))
        # A pure relative floor at 1e-12 would demand a 1e-21 absolute
        # residual in almost-zero rows. The absolute term is far below the
        # admitted force and geometric tolerances while admitting roundoff.
        structural_backward_error = np.max(
            abs(structural_error) / (1e-14 + 1e-9*structural_scale),
            initial=0.0,
        )
        gap_error = g[active] + self.gap_q[active] @ state_direction[:self.q_size]
        gap_scale = (abs(g[active])
                     + self.absolute_gap_q[active] @ abs(state_direction[:self.q_size]))
        gap_backward_error = np.max(
            abs(gap_error) / (1e-14 + 1e-9*gap_scale), initial=0.0,
        )
        inactive_error = dz[~active] + z[~active]
        if (not np.all(np.isfinite(state_direction))
                or not np.all(np.isfinite(dz))
                or not np.isfinite(structural_backward_error)
                or not np.isfinite(gap_backward_error)
                or structural_backward_error > 1.0
                or gap_backward_error > 1.0
                or np.any(inactive_error != 0.0)):
            raise ArithmeticError(
                "monolithic block direction failed coupled backward-error check "
                f"(structural={structural_backward_error:.3e}, "
                f"active_gap={gap_backward_error:.3e}, "
                f"absolute_structural={np.max(abs(structural_error), initial=0.0):.3e}, "
                f"absolute_active_gap={np.max(abs(gap_error), initial=0.0):.3e})"
            )
        return (state_direction[:self.q_size],
                state_direction[self.q_size:self.q_size+self.mu_size], dz)


def prepare_frozen_block_response(*, kqq: sparse.spmatrix,
                                  control_reactions: np.ndarray,
                                  control_rows: np.ndarray,
                                  kqz: sparse.spmatrix,
                                  gap_q: np.ndarray,
                                  residual: np.ndarray,
                                  constraint: np.ndarray,
                                  signature: str,
                                  maximum_response_entries: int = 2_000_000,
                                  maximum_modes: int = 256,
                                  maximum_factor_bytes: int = 256_000_000) -> FrozenBlockResponse:
    """Factor once and form the exact finite-dimensional gap response."""
    q_size = kqq.shape[0]
    mu_size = len(constraint)
    modes = kqz.shape[1]
    system_size = q_size + mu_size
    # Bound a dense upper estimate for the sparse LU factors and its response.
    # The full process still has a separately enforced case-level memory cap.
    if (modes > maximum_modes
            or system_size*(modes+1) > maximum_response_entries
            or 16*system_size*system_size > maximum_factor_bytes):
        raise MemoryError("monolithic block response exceeds declared dense budget")
    if (kqq.shape != (q_size, q_size)
            or kqz.shape != (q_size, modes)
            or np.shape(control_reactions) != (q_size, mu_size)
            or np.shape(control_rows) != (mu_size, q_size)
            or np.shape(gap_q) != (modes, q_size)
            or np.shape(residual) != (q_size,)):
        raise ValueError("monolithic block response has incompatible dimensions")
    structural_control = sparse.bmat((
        (kqq, sparse.csr_matrix(-control_reactions)),
        (sparse.csr_matrix(control_rows),
         sparse.csr_matrix((mu_size, mu_size))),
    ), format="csr")
    rhs = np.empty((q_size+mu_size, modes+1))
    rhs[:q_size, 0] = residual
    rhs[q_size:, 0] = constraint
    rhs[:q_size, 1:] = kqz.toarray()
    rhs[q_size:, 1:] = 0.0
    response = np.asarray(factorize(
        structural_control, MatrixClass.GENERAL, signature=signature,
    ).solve_many(rhs))
    base = response[:, 0].copy()
    force_response = response[:, 1:].copy()
    gap_from_base = np.asarray(gap_q @ base[:q_size]).reshape(-1)
    gap_response = np.asarray(gap_q @ force_response[:q_size, :])
    if not all(np.all(np.isfinite(item)) for item in (
            base, force_response, gap_from_base, gap_response)):
        raise ValueError("monolithic block response is nonfinite")
    contact_columns = sparse.vstack((kqz, sparse.csr_matrix((mu_size, modes))),
                                     format="csr")
    return FrozenBlockResponse(base, force_response, gap_response,
                               gap_from_base, q_size, mu_size,
                               structural_control, contact_columns,
                               np.asarray(gap_q).copy(), rhs[:, 0].copy(),
                               abs(structural_control), abs(contact_columns),
                               abs(np.asarray(gap_q)).copy())
