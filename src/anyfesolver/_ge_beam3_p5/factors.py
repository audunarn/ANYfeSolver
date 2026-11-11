"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

from dataclasses import dataclass
import numpy as np
from .algebra import HALVES, metrics, skew, validate_section

@dataclass(frozen=True)
class ReferenceFactors:
    full: np.ndarray
    condensed: np.ndarray
    local_map: np.ndarray

    def energy(self, displacement):
        displacement = np.asarray(displacement, dtype=float)
        if displacement.shape != (18,) or not np.isfinite(displacement).all():
            raise ValueError("finite 18-component displacement required")
        residual = self.condensed @ displacement
        return float(residual @ residual / 2)


def reference_factors(reference, section, order=24):
    """Return B, D, T with min_a ||B [q;a]||^2 = ||D q||^2, a=Tq.

    B has 18 rows (nine per half), 18 external and six local columns.
    For H=L L^T, the complementary energy factor is L^-1 De, not
    L^-T De. Complete QR of the six internal columns performs elimination
    without normal equations. Dense D^T D can still lose small-mode energy
    when subsequently contracted; consumers must retain D for that check.
    """
    section = validate_section(section)
    coordinates, frames = reference.coordinates, reference.nodal_triads
    rows = []
    for cell, (left, right) in enumerate(HALVES):
        data = metrics(reference, section, cell, order)
        dz = np.zeros((3, 24))
        dz[:, 6*left:6*left+3] = -np.eye(3)
        dz[:, 6*right:6*right+3] = np.eye(3)
        local = slice(18+3*cell, 21+3*cell)
        dz[:, local] = skew(coordinates[right]-coordinates[left])
        de = np.zeros((6, 24))
        de[:3, 6*left+3:6*left+6] = -frames[left].T
        de[:3, local] = frames[left].T
        de[3:, 6*right+3:6*right+6] = frames[right].T
        de[3:, local] = -frames[right].T
        de += data.coupling @ dz
        rows.extend((np.linalg.cholesky(data.force).T @ dz,
                     np.linalg.solve(np.linalg.cholesky(data.compliance), de)))
    full = np.vstack(rows)
    orthogonal, triangular = np.linalg.qr(full[:, 18:], mode="complete")
    # A singular internal block is an error, never a pseudoinverse fallback.
    local_map = -np.linalg.solve(triangular[:6], orthogonal[:, :6].T @ full[:, :18])
    condensed = orthogonal[:, 6:].T @ full[:, :18]
    for made in (full, condensed, local_map):
        if not np.isfinite(made).all():
            raise ValueError("nonfinite reference factor")
        made.setflags(write=False)
    return ReferenceFactors(full, condensed, local_map)
