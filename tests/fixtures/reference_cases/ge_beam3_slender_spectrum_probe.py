"""Tiny straight reference modal conditioning diagnostic, not qualification.

The alternate factor path uses the SAME mechanical factors. It is a numerical
comparison, not an independent mechanics oracle. Do not use it at prestress.
"""

import numpy as np

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5_centered.mass import reference_kinetic_factors
from anyfesolver._native_stationary_spectrum import solve_stationary_spectrum
from anyfesolver._native_reference_factor_spectrum import solve_reference_factor_spectrum


def factors(slenderness):
    """One straight length-two macro, square section, EI=1, rho*A=1.

    h=2/slenderness, EA=12/h^2, nu=.3, shear coefficient5/6;
    rotary inertia h^2/12 about each bending axis. Torsion=2 is an explicit
    diagnostic choice, not a claimed square-section Saint-Venant constant.
    """
    h = 2./slenderness; ea = 12./h**2; shear = (5./6)*ea/2.6
    section = np.diag([ea, shear, shear, 2., 1., 1.])
    inertia = np.diag([1., 1., 1., h*h/6, h*h/12, h*h/12])
    geometry = CenteredCurvedBeam3ReferenceGeometry(
        np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), np.tile(np.eye(3), (3, 1, 1)))
    made = reference_kinetic_factors(geometry, section, inertia, order=8)
    return made.uncondensed_stiffness_factor, made.full


def comparison(slenderness):
    elastic, kinetic = factors(slenderness)
    free = tuple(range(6, 24)); algebraic = (9, 10, 11, 15, 16, 17)
    physical = tuple(i for i in free if i not in algebraic)
    q, r = np.linalg.qr(elastic[:, algebraic], mode='reduced')
    mapping = np.zeros((24, len(physical))); mapping[list(physical)] = np.eye(len(physical))
    mapping[list(algebraic)] = -np.linalg.solve(r, q.T@elastic[:, physical])
    strain, speed = elastic@mapping, kinetic@mapping
    _, mass_r = np.linalg.qr(speed, mode='reduced')
    normalized = np.linalg.solve(mass_r.T, strain.T).T
    singular = np.linalg.svd(normalized, compute_uv=False)
    factor_values = singular[::-1]**2
    made = solve_reference_factor_spectrum(elastic, kinetic, free, algebraic)
    result = dict(slenderness=slenderness, factor_squared_frequencies=factor_values[:6],
        successor_squared_frequencies=made.eigenvalues, successor_normalized_residual=made.normalized_residual,
        production_qualified=False, prestressed_tangent_authorized=False)
    try:
        dense = solve_stationary_spectrum(elastic.T@elastic, kinetic.T@kinetic, free, algebraic)
        result.update(dense_status='returned', dense_squared_frequencies=dense.eigenvalues,
            dense_normalized_residual=dense.normalized_residual)
    except ValueError as exc:
        # Preserve the failure; never turn it into a zero/positive modal result.
        result.update(dense_status='failed', dense_error=str(exc))
    return result
