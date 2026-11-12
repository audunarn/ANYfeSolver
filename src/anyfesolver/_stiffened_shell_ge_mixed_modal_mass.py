"""Geometry and explicit physical inertia only, in native inertial 24 slots.

This is the kinetic half of the existing centered reference factor evaluator.
No elastic section surrogate, stiffness factor, condensation or trace mass is
needed for physical-fibre sections.
"""
import numpy as np
from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from ._ge_beam3_p5.algebra import HALVES, skew


def native_physical_mass(reference, inertia):
    if type(reference) is not CenteredCurvedBeam3ReferenceGeometry:
        raise ValueError("centered native reference geometry required")
    factor = np.linalg.cholesky(inertia).T
    points, weights = np.polynomial.legendre.leggauss(24)
    rows = []
    for cell, (left, right) in enumerate(HALVES):
        for point, weight in zip(points, weights):
            t = float((point+1)/2); xi = cell-1+t
            frame = reference.frame(xi)
            velocity = np.zeros((6, 24))
            velocity[:3, 6*left:6*left+3] = (1-t)*frame.T
            velocity[:3, 6*right:6*right+3] = t*frame.T
            local = slice(18+3*cell, 21+3*cell)
            velocity[:3, local] = -frame.T@skew(reference.half_cell_lift(cell, t))
            velocity[3:, local] = frame.T
            rows.append(np.sqrt(weight*reference.jacobian(xi)/2)*factor@velocity)
    kinetic = np.vstack(rows)
    matrix = kinetic.T@kinetic
    if not np.isfinite(matrix).all():
        raise ValueError("finite explicit native reference inertia required")
    return matrix
