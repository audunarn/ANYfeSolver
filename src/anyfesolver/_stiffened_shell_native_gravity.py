"""Native full-42 conservative gravity work per reference arclength.

The section inertia uses [v, omega] in reference-frame axes, about the
centerline. Only its physical mass and first moment enter constant gravity.
The retained cell rotations carry the curved lift and eccentric first moment;
there is no standalone condensation. Order 24 matches the native kinetic rule,
not a claim of exact continuum integration.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import fsum, isfinite

import numpy as np

from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_p5.arrays import _array, _frames
from ._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair


def _finite_array(value, shape, label):
    if np.iscomplexobj(value):
        raise ValueError(f'{label} must be real')
    return _array(value, shape, label)


def _owned(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('native gravity result is not representable')
    return np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)


def _sum(values):
    try:
        value = fsum(float(v) for v in values)
    except (OverflowError, ValueError) as error:
        raise ValueError('native gravity sum is not representable') from error
    if not isfinite(value):
        raise ValueError('native gravity sum is not representable')
    return value


def _product(a, b):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        result = np.asarray(a)*np.asarray(b)
    if (not np.isfinite(result).all()
            or np.any((np.asarray(a) != 0.) & (np.asarray(b) != 0.) & (result == 0.))):
        raise ValueError('native gravity product is not representable')
    return result


def _resolve_zero_underflow(value, terms):
    """Check a final zero, preserving exact cancellation of tiny products.

    Ordinary nonzero results and ordinary cancellation keep binary64 behavior.
    Only a zero involving underflow-range products needs the exact binary64
    rational fallback. No magnitude floor or per-term rejection is imposed.
    """
    if value != 0.:
        return value
    terms = tuple(tuple(float(v) for v in term) for term in terms)
    tiny = np.finfo(float).tiny
    affected = False
    for term in terms:
        product = 1.
        for factor in term:
            product *= factor
        if all(factor != 0. for factor in term) and abs(product) < tiny:
            affected = True
            break
    if not affected:
        return value
    exact = Fraction(0)
    for term in terms:
        product = Fraction(1)
        for factor in term:
            product *= Fraction(factor)
        exact += product
    if exact == 0:
        return 0.
    restored = float(exact)
    if restored == 0. or not isfinite(restored):
        raise ValueError('native gravity component is not representable')
    return restored


@dataclass(frozen=True)
class Work:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray
    spatial_jacobian: np.ndarray
    total_force: np.ndarray
    total_moment: np.ndarray

    def __post_init__(self):
        if not isfinite(self.value):
            raise ValueError('native gravity work is not representable')
        for name in ('gradient', 'hessian', 'spatial_jacobian', 'total_force', 'total_moment'):
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True)
class NativeGravity:
    coordinates: np.ndarray
    inertia: np.ndarray
    nodal_masses: np.ndarray
    cell_first_moments: np.ndarray
    order: int

    def __post_init__(self):
        for name in ('coordinates', 'inertia', 'nodal_masses', 'cell_first_moments'):
            object.__setattr__(self, name, _owned(getattr(self, name)))

    def evaluate(self, positions, position_low, rotations, acceleration):
        """Return load work and derivatives; moment is about the global origin.

        ``rotations`` are the two spatial relative cell rotations U. Nodal
        rotation slots 3:6, 9:12, 15:18 and stress slots 24:42 remain zero.
        The spatial Jacobian differentiates the returned torque after actual
        left updates Exp(delta) U; the Hessian differentiates work in one chart.
        """
        return evaluate_reference_vector_work(
            self.coordinates, self.nodal_masses, self.cell_first_moments,
            positions, position_low, rotations, acceleration)


def evaluate_reference_vector_work(coordinates, nodal_weights, cell_first_moments,
                                   positions, position_low, rotations, vector):
    """Analytical retained reference-vector work for preintegrated geometry.

    Neutral weights need not represent mass. Gravity retains its original
    validation labels and arithmetic through this shared private implementation.
    """
    x = _finite_array(positions, (3, 3), 'gravity positions')
    low = _finite_array(position_low, (3, 3), 'gravity low positions')
    rotation_values = _finite_array(rotations, (2, 3, 3), 'gravity rotations')
    rotations = _frames(rotation_values, 2, 'gravity rotations')
    g = _finite_array(vector, (3,), 'gravity acceleration')
    for high, tail in zip(x.flat, low.flat):
        validate_pair(float(high), float(tail))
    forces = _product(nodal_weights[:, None], g[None, :])
    gradient = np.zeros(42)
    hessian = np.zeros((42, 42))
    spatial = np.zeros((42, 42))
    terms = []
    moments = []
    moment_products = [[], [], []]
    for node in range(3):
        gradient[6*node:6*node+3] = forces[node]
        for component in range(3):
            high, tail = split_sum((float(x[node, component]), float(low[node, component]),
                                    -float(coordinates[node, component])))
            terms.extend((_product(forces[node, component], high),
                          _product(forces[node, component], tail)))
        moments.extend((np.cross(x[node], forces[node]), np.cross(low[node], forces[node])))
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for position in (x[node], low[node]):
                moment_products[i].extend(((position[j], forces[node, k]),
                                           (-position[k], forces[node, j])))
    for cell in (0, 1):
        b = cell_first_moments[cell]
        rotated = rotations[cell]@b
        local = slice(18+3*cell, 21+3*cell)
        torque = np.cross(rotated, g)
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            products = ((rotated[j], g[k]), (-rotated[k], g[j]))
            torque[i] = _resolve_zero_underflow(torque[i], products)
            moment_products[i].extend(products)
        gradient[local] = torque
        product = np.outer(rotated, g)
        dot = _sum(product.diagonal())
        spatial[local, local] = product-dot*np.eye(3)
        hessian[local, local] = .5*product+.5*product.T-dot*np.eye(3)
        for i in range(3):
            for j in range(3):
                diagonal = [(-rotated[k], g[k]) for k in range(3)] if i == j else []
                row, column = 18+3*cell+i, 18+3*cell+j
                spatial[row, column] = _resolve_zero_underflow(spatial[row, column],
                    [(rotated[i], g[j]), *diagonal])
                hessian[row, column] = _resolve_zero_underflow(hessian[row, column],
                    [(.5, rotated[i], g[j]), (.5, rotated[j], g[i]), *diagonal])
        # (U-I)b avoids subtracting two rounded first-moment vectors.
        change = (rotations[cell]-np.eye(3))@b
        terms.extend(_product(change, g))
        moments.append(torque)
    force = np.array([_sum(forces[:, i]) for i in range(3)])
    moment = np.array([_sum(row[i] for row in moments) for i in range(3)])
    for i in range(3):
        moment[i] = _resolve_zero_underflow(moment[i], moment_products[i])
    return Work(_sum(terms), gradient, hessian, spatial, force, moment)


def compile_native_gravity(reference, inertia, order=24):
    """Capture immutable geometry and physical sectional inertia once.

    m_i = integral mu N_i ds0; b_cell = integral (mu c0 + R0 h) ds0.
    The input tensor is captured unchanged; stiffness never supplies density.
    """
    from ._stiffened_shell_ge_reference_modal import _physical_spatial_inertia

    if type(reference) is not Reference:
        raise ValueError('native centered reference required for gravity')
    if type(order) is not int or order not in (4, 8, 24):
        raise ValueError('native gravity quadrature order must be 4, 8 or 24')
    tensor = _finite_array(inertia, (6, 6), 'gravity physical spatial inertia')
    mass, _, _ = _physical_spatial_inertia(tensor)
    try:
        np.linalg.cholesky(tensor)
    except np.linalg.LinAlgError as error:
        raise ValueError('positive definite physical spatial inertia required') from error
    h = np.array((tensor[1, 5], tensor[2, 3], tensor[0, 4]))
    nodes = _finite_array(reference.coordinates, (3, 3), 'gravity reference coordinates')
    nodal_terms = [[], [], []]
    moment_terms = [[], []]
    points, weights = np.polynomial.legendre.leggauss(order)
    for cell in (0, 1):
        for point, weight in zip(points, weights):
            t = float((point+1)/2)
            xi = cell-1+t
            measure = float(weight*reference.jacobian(xi)/2)
            if not isfinite(measure) or measure <= 0.:
                raise ValueError('positive representable gravity reference measure required')
            lift = _finite_array(reference.half_cell_lift(cell, t), (3,), 'gravity reference lift')
            frame = _frames([_finite_array(reference.frame(xi), (3, 3), 'gravity reference frame')],
                            1, 'gravity reference frame')[0]
            weighted_mass = float(_product(mass, measure))
            nodal_terms[cell].append(float(_product(weighted_mass, 1-t)))
            nodal_terms[cell+1].append(float(_product(weighted_mass, t)))
            moment_terms[cell].append(_product(_product(mass, lift)+frame@h, measure))
    masses = np.array([_sum(values) for values in nodal_terms])
    moments = np.array([[_sum(row[i] for row in rows) for i in range(3)] for rows in moment_terms])
    if np.any(masses <= 0.):
        raise ValueError('positive representable native gravity nodal masses required')
    return NativeGravity(nodes, tensor, masses, moments, order)
