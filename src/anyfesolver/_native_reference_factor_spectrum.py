"""Private reference-energy spectrum without normal-equation eigenvalues.

REFERENCE PSD FACTORS ONLY. Not an indefinite/prestressed tangent adapter.
Eliminate exactly massless trace coordinates globally by QR, whiten the
retained kinetic factor by QR and extract frequencies by SVD of strain rows.
No inertia floor, mode clipping, local inertial condensation or public route.
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from ._native_reference_modal import _owned
from .control import cancellation_safe_point


POLICY = 'PRIVATE_REFERENCE_ENERGY_KINETIC_QR_SVD_V1'


@dataclass(frozen=True)
class ReferenceFactorSpectrum:
    eigenvalues: np.ndarray
    full_modes: np.ndarray
    dynamic_map: np.ndarray
    normalized_residual: float
    policy: str = POLICY
    production_qualified: bool = False
    prestressed_tangent_authorized: bool = False


def solve_reference_factor_spectrum(elastic_factor, kinetic_factor, free_dofs, algebraic_dofs, *,
                                    num_modes=6, cancellation_token=None):
    """Small factor-only solve; supplying factors asserts a reference PSD form.

    The caller must establish the reference state and factor provenance. This
    numerical kernel cannot infer those from matrices and must never receive
    a square root manufactured by dropping negative tangent eigenvalues.
    """
    def checkpoint(stage):
        cancellation_safe_point(cancellation_token, 'reference_factor_spectrum.'+stage)
    checkpoint('start')
    elastic, kinetic = _owned(elastic_factor), _owned(kinetic_factor)
    if (elastic.ndim != 2 or kinetic.ndim != 2 or elastic.shape[1] != kinetic.shape[1]
            or not 1 <= elastic.shape[1] <= 256 or not 1 <= elastic.shape[0] <= 8192
            or not 1 <= kinetic.shape[0] <= 8192):
        raise ValueError('matching bounded reference factors required')
    total = elastic.shape[1]
    for slots in (free_dofs, algebraic_dofs):
        if (type(slots) is not tuple or len(set(slots)) != len(slots)
                or any(type(i) is not int or not 0 <= i < total for i in slots)):
            raise ValueError('exact distinct in-range reference DOFs required')
    if not set(algebraic_dofs) <= set(free_dofs):
        raise ValueError('algebraic coordinates must be free')
    if np.any(kinetic[:, algebraic_dofs] != 0.):
        raise ValueError('declared algebraic inertia must be exactly absent')
    physical = tuple(i for i in free_dofs if i not in algebraic_dofs)
    if type(num_modes) is not int or not 1 <= num_modes <= len(physical):
        raise ValueError('requested reference modes outside physical dimension')
    mapping = np.zeros((total, len(physical)))
    mapping[list(physical)] = np.eye(len(physical))
    if algebraic_dofs:
        trace = elastic[:, algebraic_dofs]
        if trace.shape[0] < trace.shape[1]:
            raise ValueError('unresolved algebraic reference rank')
        q, r = linalg.qr(trace, mode='economic')
        # Rank verification is a fail-closed numerical usability check, not a
        # proof of reference coercivity. Do not form r.T@r for this check.
        singular = linalg.svdvals(r)
        if singular[-1] <= np.finfo(float).eps*max(r.shape)*singular[0]:
            raise ValueError('unresolved algebraic reference rank')
        mapping[list(algebraic_dofs)] = -linalg.solve_triangular(r, q.T@elastic[:, physical])
    strain, speed = elastic@mapping, kinetic@mapping
    if min(strain.shape[0], speed.shape[0]) < len(physical):
        raise ValueError('insufficient reference factor rows')
    checkpoint('trace_equilibrium')
    _, mass_r = linalg.qr(speed, mode='economic')
    singular_mass = linalg.svdvals(mass_r)
    if singular_mass[-1] <= np.finfo(float).eps*max(mass_r.shape)*singular_mass[0]:
        raise ValueError('unresolved physical kinetic rank')
    normalized = linalg.solve_triangular(mass_r.T, strain.T, lower=True).T
    # Squaring singular values only AFTER extracting them avoids the squared
    # conditioning of (normalized.T@normalized)'s symmetric eigenproblem.
    _, singular, right = linalg.svd(normalized, full_matrices=False, lapack_driver='gesvd')
    chosen = np.arange(len(singular)-1, len(singular)-num_modes-1, -1)
    values = singular[chosen]**2
    modes = mapping@linalg.solve_triangular(mass_r, right.T[:, chosen])
    if not np.isfinite(values).all() or not np.isfinite(modes).all():
        raise ValueError('nonfinite reference spectrum')
    for j in range(num_modes):
        if modes[int(np.argmax(np.abs(modes[:, j]))), j] < 0.: modes[:, j] *= -1.
    ev, gv = elastic@modes, kinetic@modes
    residual = (elastic.T@ev-(kinetic.T@gv)*values)[list(free_dofs)]
    scale = max(1., np.linalg.norm(elastic)**2*np.linalg.norm(modes),
                np.linalg.norm(kinetic)**2*np.linalg.norm(modes*values))
    error = float(np.linalg.norm(residual)/scale)
    if not np.isfinite(error) or error > 1e-11:
        raise ValueError('reference factor eigenpair residual failed')
    if np.linalg.norm(gv.T@gv-np.eye(num_modes)) > 1e-11:
        raise ValueError('reference factor modal mass normalization failed')
    checkpoint('output')
    return ReferenceFactorSpectrum(_owned(values), _owned(modes), _owned(mapping), error)
