"""Private signed stationary pencil with explicitly declared massless DOFs.

Unlike reference energy-factor assembly, this algebra must retain negative
stiffness eigenvalues. It does not certify a material branch or equilibrium;
the caller must establish those before treating its output as vibration data.
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from ._native_reference_modal import _owned
from .control import cancellation_safe_point


@dataclass(frozen=True)
class StationarySpectrum:
    eigenvalues: np.ndarray
    full_modes: np.ndarray
    dynamic_map: np.ndarray
    normalized_residual: float
    production_qualified: bool = False


def solve_stationary_spectrum(stiffness, mass, free_dofs, algebraic_dofs, *,
                              num_modes=6, cancellation_token=None):
    """Eliminate only positive, exactly massless assembled trace equations.

    The trace block must be positive: a hidden negative algebraic-energy mode
    must fail rather than disappear behind a positive finite spectrum.
    All finite physical modes, including negative and zero values, are kept
    eligible for selection. No stiffness factorization requiring global PSD,
    eigenvalue clipping, mass regularization or inferred mass kernel is used.
    Supports have already selected ``free_dofs``. Dense development scope only.
    """
    cancellation_safe_point(cancellation_token, 'stationary_spectrum.start')
    k, m = _owned(stiffness), _owned(mass)
    if (k.ndim != 2 or k.shape[0] != k.shape[1] or m.shape != k.shape
            or not 1 <= len(k) <= 256):
        raise ValueError('matching bounded stationary matrices required')
    for matrix in (k, m):
        if np.linalg.norm(matrix-matrix.T) > 1e-11*max(1., np.linalg.norm(matrix)):
            raise ValueError('symmetric stationary pencil required')
    for slots in (free_dofs, algebraic_dofs):
        if (type(slots) is not tuple or len(set(slots)) != len(slots)
                or any(type(i) is not int or not 0 <= i < len(k) for i in slots)):
            raise ValueError('exact distinct in-range DOFs required')
    if not set(algebraic_dofs) <= set(free_dofs):
        raise ValueError('algebraic DOFs must be free')
    if np.any(m[:, algebraic_dofs] != 0.) or np.any(m[algebraic_dofs, :] != 0.):
        raise ValueError('declared algebraic mass must be exactly zero')
    physical = tuple(i for i in free_dofs if i not in algebraic_dofs)
    if type(num_modes) is not int or not 1 <= num_modes <= len(physical):
        raise ValueError('requested modes outside physical dimension')
    mapping = np.zeros((len(k), len(physical)))
    mapping[list(physical)] = np.eye(len(physical))
    if algebraic_dofs:
        a = k[np.ix_(algebraic_dofs, algebraic_dofs)]
        coupling = k[np.ix_(algebraic_dofs, physical)]
        np.linalg.cholesky(a)
        mapping[list(algebraic_dofs)] = -np.linalg.solve(a, coupling)
    reduced_k, reduced_m = mapping.T@k@mapping, mapping.T@m@mapping
    np.linalg.cholesky(reduced_m)
    cancellation_safe_point(cancellation_token, 'stationary_spectrum.reduced')
    values, vectors = linalg.eigh(reduced_k, reduced_m, subset_by_index=(0, num_modes-1))
    modes = mapping@vectors
    for index in range(num_modes):
        pivot = int(np.argmax(np.abs(modes[:, index])))
        if modes[pivot, index] < 0.: modes[:, index] *= -1.
    residual = (k@modes-(m@modes)*values)[list(free_dofs)]
    scale = max(1., np.linalg.norm(k)*np.linalg.norm(modes), np.linalg.norm(m)*np.linalg.norm(modes*values))
    error = float(np.linalg.norm(residual)/scale)
    if not np.isfinite(error) or error > 1e-11:
        raise ValueError('stationary full-pencil eigenpair residual failed')
    if np.linalg.norm(modes.T@m@modes-np.eye(num_modes)) > 1e-11:
        raise ValueError('stationary modal mass normalization failed')
    cancellation_safe_point(cancellation_token, 'stationary_spectrum.output')
    return StationarySpectrum(_owned(values), _owned(modes), _owned(mapping), error)
