"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

HALVES = ((0, 1), (1, 2))


def skew(vector: Any) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))


def rotation(vector: Any) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    cross = skew(vector)
    if angle < 1e-8:
        return np.eye(3) + np.sinc(angle / np.pi) * cross + (
            0.5 - angle**2 / 24.0 + angle**4 / 720.0
        ) * cross @ cross
    return np.eye(3) + np.sin(angle) / angle * cross + (
        (1 - np.cos(angle)) / angle**2
    ) * cross @ cross


def log_rotation(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("finite proper rotation required")
    if np.linalg.norm(matrix.T @ matrix - np.eye(3)) > 1e-11 or abs(
        np.linalg.det(matrix) - 1.0
    ) > 1e-11:
        raise ValueError("finite proper rotation required")
    angle = np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))
    if angle >= 0.9 * np.pi:
        raise ValueError("relative rotation requires refinement or cutback")
    axial = np.array((matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0],
                      matrix[1, 0]-matrix[0, 1])) / 2
    factor = 1 + angle**2 / 6 + 7 * angle**4 / 360 if angle < 1e-5 else (
        angle / np.sin(angle)
    )
    return factor * axial


def validate_section(section: Any) -> np.ndarray:
    made = np.array(section, dtype=float, copy=True)
    if made.shape != (6, 6) or not np.isfinite(made).all():
        raise ValueError("section must be finite 6 by 6")
    if not np.array_equal(made, made.T):
        raise ValueError("probe section must be exactly symmetric")
    try:
        np.linalg.cholesky(made)
    except np.linalg.LinAlgError as exc:
        raise ValueError("positive definite section required") from exc
    return made


@dataclass(frozen=True)
class CellMetrics:
    force: np.ndarray
    compliance: np.ndarray
    coupling: np.ndarray


def metrics(reference: Any, section: Any, cell: int, order: int = 24) -> CellMetrics:
    section = validate_section(section)
    if cell not in (0, 1) or not 2 <= order <= 64:
        raise ValueError("probe cell/order outside registered diagnostic range")
    a, b, d = section[:3, :3], section[:3, 3:], section[3:, 3:]
    d_inverse = np.linalg.solve(d, np.eye(3))
    force = np.zeros((3, 3))
    compliance = np.zeros((6, 6))
    coupling = np.zeros((6, 3))
    stations, weights = np.polynomial.legendre.leggauss(order)
    for station, weight in zip(stations, weights):
        fraction = (station + 1) / 2
        xi = cell - 1 + fraction
        jacobian = reference.jacobian(xi)
        v = reference.frame(xi).T / jacobian
        l = np.hstack(((1-fraction)*np.eye(3), fraction*np.eye(3)))
        measure = weight * jacobian / 2
        force += measure * v.T @ (a - b @ d_inverse @ b.T) @ v
        compliance += measure * l.T @ d_inverse @ l
        coupling += measure * l.T @ d_inverse @ b.T @ v
    return CellMetrics(force, compliance, coupling)
