"""Reference kinetic operator for a flat homogeneous Q4 director plate.

The mid-surface is the mass origin. At thickness coordinate z the physical
velocity is v + z (omega cross n), so thickness integration gives rho*h*I
and rho*h**3*(I-n*n.T)/12. Normal drilling spin has no kinetic inertia.
This is a distinct private operator; the legacy shell mass is unchanged.
"""

from fractions import Fraction
from numbers import Real

import numpy as np


POLICY = "STIFFENED_SHELL_PHYSICAL_FLAT_Q4_DIRECTOR_MASS_V1"


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"positive finite physical {name} required")
    value = float(value)
    if not np.isfinite(value) or value <= 0.:
        raise ValueError(f"positive finite physical {name} required")
    return value


def physical_mass(vertices, *, density_kg_m3, thickness_m,
                  reference_normal=(0., 0., 1.)):
    """Owned 24x24 mass in global nodal [v, omega] coordinates, SI units.

    Only a flat, nonfolded, homogeneous plate about its mid-surface is admitted.
    The normal is a unit director parallel to the actual plane. Reversal of
    the director alone does not change physical inertia. No density, mass
    center, section offset or rotational inertia is inferred from stiffness.
    """
    rho = _positive(density_kg_m3, "density")
    h = _positive(thickness_m, "thickness")
    x = np.asarray(vertices, dtype=float)
    normal = np.asarray(reference_normal, dtype=float)
    if (x.shape != (4, 3) or not np.isfinite(x).all() or
            normal.shape != (3,) or not np.isfinite(normal).all() or
            abs(np.linalg.norm(normal)-1.) > 1e-10):
        raise ValueError("finite flat Q4 coordinates and unit normal required")
    # Centering avoids origin-dependent geometry precision in the Jacobian.
    x = x-x[0]
    length = np.max(np.linalg.norm(x, axis=1))
    if length == 0. or not np.isfinite(length) or np.max(abs(x@normal)) > 1e-10*length:
        raise ValueError("flat Q4 mass geometry and parallel director required")
    normal = normal/np.linalg.norm(normal)
    nx, ny, nz = normal
    cross_normal = np.array(((0., -nz, ny), (nz, 0., -nx), (-ny, nx, 0.)))
    # The Gram form represents exactly the same director kinetic law while
    # avoiding spurious negative normal inertia from an almost-unit input.
    projector = cross_normal.T@cross_normal
    section = np.zeros((6, 6))
    try:
        area_mass = float(Fraction(rho)*Fraction(h))
        rotary_mass = float(Fraction(rho)*Fraction(h)**3/12)
    except OverflowError as exc:
        raise ValueError("physical plate inertia is not representable") from exc
    section[:3, :3] = area_mass*np.eye(3)
    section[3:, 3:] = rotary_mass*projector
    if (not np.isfinite(section).all() or section[0, 0] <= 0. or
            np.trace(section[3:, 3:]) <= 0. or
            np.any(section[3:, 3:][projector != 0.] == 0.)):
        raise ValueError("physical plate inertia is not representable")
    gram = np.zeros((4, 4))
    sign = None
    # The planar bilinear-map determinant is affine. Corner positivity,
    # unlike positivity only at Gauss points, excludes a folded corner.
    for xi, eta in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)):
        dx = .25*np.array((-(1.-eta), 1.-eta, 1.+eta, -(1.+eta)))
        dy = .25*np.array((-(1.-xi), -(1.+xi), 1.+xi, 1.-xi))
        jacobian = float(np.cross(dx@x, dy@x)@normal)
        if sign is None:
            sign = np.sign(jacobian)
        if not np.isfinite(jacobian) or sign*jacobian <= 0.:
            raise ValueError("nondegenerate nonfolded Q4 mass geometry required")
    for xi in (-1./np.sqrt(3.), 1./np.sqrt(3.)):
        for eta in (-1./np.sqrt(3.), 1./np.sqrt(3.)):
            shape = .25*np.array(((1.-xi)*(1.-eta), (1.+xi)*(1.-eta),
                                  (1.+xi)*(1.+eta), (1.-xi)*(1.+eta)))
            dx = .25*np.array((-(1.-eta), 1.-eta, 1.+eta, -(1.+eta)))
            dy = .25*np.array((-(1.-xi), -(1.+xi), 1.+xi, 1.-xi))
            jacobian = float(np.cross(dx@x, dy@x)@normal)
            if not np.isfinite(jacobian) or sign*jacobian <= 0.:
                raise ValueError("nondegenerate nonfolded Q4 mass geometry required")
            gram += abs(jacobian)*np.outer(shape, shape)
    # Every Gram coefficient is strictly positive on an admitted element.
    # Do not accept lost rotary inertia merely because translation survives.
    if not np.isfinite(gram).all() or np.any(gram <= 0.):
        raise ValueError("physical plate mass assembly is not representable")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        matrix = np.kron(gram, section)
    required = np.tile(section != 0., (4, 4))
    if not np.isfinite(matrix).all() or np.any(matrix[required] == 0.):
        raise ValueError("physical plate mass assembly is not representable")
    return matrix
