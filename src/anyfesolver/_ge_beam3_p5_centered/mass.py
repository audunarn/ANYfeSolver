"""Full retained-cell inertia and reference factors on the centered geometry.

No static/Guyan reduction of cell inertia and no numerical trace mass. The
existing global native modal operator performs shared trace elimination.
"""

from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5.algebra import HALVES, metrics, skew, validate_section
from anyfesolver._ge_beam3_p5.arrays import _array, _readonly
from anyfesolver._ge_beam3_p5.compensated_coordinates import validate_pair, split_sum
from anyfesolver._ge_beam3_p5.inertia import LiftedInertia


def _reference(reference, order):
    if type(reference) is not CenteredCurvedBeam3ReferenceGeometry:
        raise ValueError('centered reference evaluator required; no legacy clone fallback')
    if type(order) is not int or order not in (4,8,24):
        raise ValueError('registered centered quadrature order required')


@dataclass(frozen=True)
class ReferenceKineticFactors:
    full: np.ndarray
    uncondensed_stiffness_factor: np.ndarray


def reference_kinetic_factors(reference, section, section_mass, *, order=24):
    _reference(reference,order)
    section = validate_section(section); section_mass = validate_section(section_mass)
    inertia_factor = np.linalg.cholesky(section_mass).T
    nodes, frames = reference.coordinates, reference.nodal_triads
    # Same complementary-energy Gram factor as the preserved reference
    # operator, before any elimination. No unused static cell-spin map is made.
    elastic_rows = []
    for cell,(left,right) in enumerate(HALVES):
        data = metrics(reference,section,cell,order)
        dz = np.zeros((3,24)); dz[:,6*left:6*left+3] = -np.eye(3); dz[:,6*right:6*right+3] = np.eye(3)
        local = slice(18+3*cell,21+3*cell)
        chord = np.array([split_sum((float(nodes[right,i]),-float(nodes[left,i])))[0] for i in range(3)])
        dz[:,local] = skew(chord)
        de = np.zeros((6,24)); de[:3,6*left+3:6*left+6] = -frames[left].T; de[:3,local] = frames[left].T
        de[3:,6*right+3:6*right+6] = frames[right].T; de[3:,local] = -frames[right].T
        de += data.coupling@dz
        elastic_rows.extend((np.linalg.cholesky(data.force).T@dz,
            np.linalg.solve(np.linalg.cholesky(data.compliance),de)))
    points, weights = np.polynomial.legendre.leggauss(order); kinetic_rows = []
    for cell,(left,right) in enumerate(HALVES):
        for point,weight in zip(points,weights):
            t = float((point+1)/2); xi = cell-1+t; frame = reference.frame(xi)
            offset = reference.half_cell_lift(cell,t)
            velocity = np.zeros((6,24)); velocity[:3,6*left:6*left+3] = (1-t)*frame.T
            velocity[:3,6*right:6*right+3] = t*frame.T
            local = slice(18+3*cell,21+3*cell)
            velocity[:3,local] = -frame.T@skew(offset); velocity[3:,local] = frame.T
            measure = weight*reference.jacobian(xi)/2
            kinetic_rows.append(np.sqrt(measure)*inertia_factor@velocity)
    elastic = np.vstack(elastic_rows); kinetic = np.vstack(kinetic_rows)
    if not np.isfinite(elastic).all() or not np.isfinite(kinetic).all():
        raise ValueError('nonfinite centered reference factors')
    return ReferenceKineticFactors(_readonly(kinetic),_readonly(elastic))


class _CenteredRestInertia(LiftedInertia):
    """Reuses the frozen kinetic operator with only its reference cache changed."""

    def __init__(self, reference, section_mass, *, order):
        _reference(reference,order)
        self.section_mass = _readonly(validate_section(section_mass))
        points, weights = np.polynomial.legendre.leggauss(order); stations = []
        for cell,(left,right) in enumerate(HALVES):
            for point,weight in zip(points,weights):
                t = float((point+1)/2); xi = cell-1+t
                frame = reference.frame(xi); offset = reference.half_cell_lift(cell,t)
                measure = float(weight*reference.jacobian(xi)/2)
                if not np.isfinite(measure) or measure <= 0.:
                    raise ValueError('positive finite kinetic measure required')
                stations.append((cell,left,right,t,_readonly(frame),_readonly(offset),measure))
        self._stations = tuple(stations)

    def evaluate(self, positions, local_rotations, velocity, acceleration):
        velocity = _array(velocity,(24,),'rest velocity'); acceleration = _array(acceleration,(24,),'rest acceleration')
        if np.any(velocity) or np.any(acceleration):
            raise ValueError('centered native inertia currently supports rest linearization only')
        return super().evaluate(positions,local_rotations,velocity,acceleration)


def current_rest_mass(reference, section_inertia, order, high, low, rotations):
    _reference(reference,order)
    high = _array(high,(3,3),'rest coordinates'); low = _array(low,(3,3),'rest low coordinates')
    for a,b in zip(high.flat,low.flat): validate_pair(float(a),float(b))
    result = _CenteredRestInertia(reference,section_inertia,order=order).evaluate(
        high,rotations,np.zeros(24),np.zeros(24))
    if result.kinetic_energy != 0. or np.any(result.generalized_momentum) or np.any(result.angular_momentum):
        raise ValueError('rest mass produced nonzero kinetic state')
    return result.mass
