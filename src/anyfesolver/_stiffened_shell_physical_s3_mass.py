"""Consistent physical mass for the flat S3 nodal director field.

Global nodal coordinates are [v, omega]. The thickness velocity is
v + z*(omega cross n); normal spin is exactly massless. This operator does
not alter the element's historical translational/lumped mass.
"""
from fractions import Fraction

import numpy as np

from ._stiffened_shell_physical_q4_mass import _positive

POLICY = "STIFFENED_SHELL_PHYSICAL_FLAT_S3_LINEAR_NODAL_DIRECTOR_MASS_V1"


def physical_mass(vertices, *, density_kg_m3, thickness_m,
                  reference_normal=(0., 0., 1.)):
    rho = _positive(density_kg_m3, "density")
    h = _positive(thickness_m, "thickness")
    x = np.asarray(vertices, dtype=float)
    n = np.asarray(reference_normal, dtype=float)
    if (x.shape != (3, 3) or n.shape != (3,) or
            not np.isfinite(x).all() or not np.isfinite(n).all() or
            abs(np.linalg.norm(n)-1.) > 1e-10):
        raise ValueError("finite flat S3 coordinates and unit normal required")
    x = x-x[0]
    length = float(np.max(np.linalg.norm(x, axis=1)))
    if (not np.isfinite(length) or length == 0. or
            np.max(abs(x@n)) > 1e-10*length):
        raise ValueError("flat S3 mass geometry and parallel director required")
    n = n/np.linalg.norm(n)
    area = .5*abs(float(np.cross(x[1], x[2])@n))
    if not np.isfinite(area) or area <= 0.:
        raise ValueError("nondegenerate S3 mass geometry required")
    nx, ny, nz = n
    cross = np.array(((0., -nz, ny), (nz, 0., -nx), (-ny, nx, 0.)))
    projector = cross.T@cross
    try:
        translation = float(Fraction(rho)*Fraction(h))
        rotation = float(Fraction(rho)*Fraction(h)**3/12)
    except OverflowError as exc:
        raise ValueError("physical S3 inertia is not representable") from exc
    section = np.zeros((6, 6))
    section[:3, :3] = translation*np.eye(3)
    section[3:, 3:] = rotation*projector
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        matrix = np.kron(area/12.*(np.ones((3, 3))+np.eye(3)), section)
    required = np.tile(section != 0., (3, 3))
    if (not np.isfinite(matrix).all() or translation <= 0. or rotation <= 0. or
            np.any(section[3:, 3:][projector != 0.] == 0.) or
            np.any(matrix[required] == 0.)):
        raise ValueError("physical S3 inertia is not representable")
    return matrix
