"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

from dataclasses import dataclass
import numpy as np
from .algebra import HALVES, skew, validate_section
from .factors import reference_factors
from .arrays import _array, _readonly

@dataclass(frozen=True)
class ReferenceKineticFactors:
    full: np.ndarray
    reduced: np.ndarray
    static_map: np.ndarray
    stiffness_factor: np.ndarray
    uncondensed_stiffness_factor: np.ndarray

    def kinetic_energy(self, velocity):
        velocity = _array(velocity, (18,), "external velocity")
        field = self.reduced @ velocity
        return float(field @ field/2)

    @property
    def mass(self):
        return self.reduced.T @ self.reduced

    @property
    def stiffness(self):
        return self.stiffness_factor.T @ self.stiffness_factor


def reference_kinetic_factors(reference, section, section_mass, *, order=24):
    """Integrate reference kinetic energy, then use the static local map.

    v=I(v_node)+omega_cell cross (r0-I(X)); angular velocity=omega_cell.
    At each station express both spatial velocities in the physical R0 frame
    before applying the symmetric positive-definite 6x6 section inertia.
    The full vector has external18 + local6 coordinates. Its static map is
    [I;T], using exactly the existing elastic local elimination. This is a
    Guyan approximation to dynamics, NOT exact elimination of local inertia.
    """
    if not isinstance(order, int) or isinstance(order, bool) or not 2 <= order <= 64:
        raise ValueError("mass quadrature order must be an integer from 2 to 64")
    section_mass = validate_section(section_mass)
    inertia_factor = np.linalg.cholesky(section_mass).T
    elastic = reference_factors(reference, section, order)
    mapping = np.vstack((np.eye(18), elastic.local_map))
    nodes = reference.coordinates
    points, weights = np.polynomial.legendre.leggauss(order)
    rows = []
    for cell, (left, right) in enumerate(HALVES):
        for point, weight in zip(points, weights):
            t = (point+1)/2
            xi = cell-1+t
            frame = reference.frame(xi)
            offset = reference.position(xi)-((1-t)*nodes[left]+t*nodes[right])
            velocity = np.zeros((6, 24))
            velocity[:3, 6*left:6*left+3] = (1-t)*frame.T
            velocity[:3, 6*right:6*right+3] = t*frame.T
            local = slice(18+3*cell, 21+3*cell)
            velocity[:3, local] = -frame.T @ skew(offset)
            velocity[3:, local] = frame.T
            measure = weight*reference.jacobian(xi)/2
            rows.append(np.sqrt(measure)*inertia_factor @ velocity)
    full = np.vstack(rows)
    return ReferenceKineticFactors(_readonly(full), _readonly(full @ mapping),
                                    _readonly(mapping), elastic.condensed, elastic.full)
