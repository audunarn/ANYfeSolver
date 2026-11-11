"""Scale-aware diagnostics for a supported primal mortar pressure space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupportedPressureGram:
    supported_modes: int
    scaled_rank: int
    scaled_condition: float | None
    smallest_scaled_eigenvalue: float


def supported_pressure_gram(gram: np.ndarray, support: np.ndarray) -> SupportedPressureGram:
    """Diagnose linear independence without confusing area scale with rank.

    Modes with zero reference support have no represented pressure field and
    are excluded.  Each supported basis function is normalized by the square
    root of its Gram diagonal before an eigenvalue rank test.  The production
    contact law does not invert this diagnostic matrix.
    """

    matrix = np.asarray(gram, dtype=float)
    area = np.asarray(support, dtype=float).reshape(-1)
    if (matrix.shape != (len(area), len(area)) or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(area)) or np.any(area < 0.0)):
        raise ValueError("pressure Gram and support must be finite and dimensionally consistent")
    if not np.allclose(matrix, matrix.T, atol=0.0, rtol=64.0 * np.finfo(float).eps):
        raise ValueError("pressure Gram must be symmetric")
    selected = np.flatnonzero(area > 0.0)
    count = int(len(selected))
    if not count:
        return SupportedPressureGram(0, 0, None, 0.0)
    block = matrix[np.ix_(selected, selected)]
    diagonal = np.diag(block)
    if np.any(diagonal <= 0.0):
        raise ValueError("a supported pressure mode has nonpositive Gram diagonal")
    inverse_scale = 1.0 / np.sqrt(diagonal)
    normalized = (inverse_scale[:, None] * block) * inverse_scale[None, :]
    eigenvalues = np.linalg.eigvalsh(normalized)
    largest = float(eigenvalues[-1])
    threshold = count * np.finfo(float).eps * largest
    if eigenvalues[0] < -threshold:
        raise ValueError("pressure Gram has a materially negative scaled eigenvalue")
    rank = int(np.count_nonzero(eigenvalues > threshold))
    condition = (float(largest / eigenvalues[0]) if rank == count else None)
    return SupportedPressureGram(count, rank, condition, float(eigenvalues[0]))
