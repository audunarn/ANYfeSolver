"""Captured full-42 reference-arclength force and fixed spatial couple work.

Forces act on the centered two-cell lifted field. Couples act only on the
retained cell rotations and have zero spatial derivative, with no SO(3)
potential. The element's order-4/8 rule is retained exactly.
"""
from dataclasses import dataclass
from math import isfinite

import numpy as np

from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_retained_fibre import RetainedFibreOperator
from ._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from ._stiffened_shell_native_gravity import (
    _finite_array, _owned, _product, _sum, evaluate_reference_vector_work,
)


@dataclass(frozen=True)
class DistributedWork:
    external_force: np.ndarray
    spatial_jacobian: np.ndarray
    total_force: np.ndarray
    total_moment: np.ndarray
    force_potential: float
    potential: float | None

    def __post_init__(self):
        for name, shape in (('external_force', (42,)), ('spatial_jacobian', (42, 42)),
                            ('total_force', (3,)), ('total_moment', (3,))):
            object.__setattr__(self, name, _owned(_finite_array(getattr(self, name), shape, name)))
        if not isfinite(self.force_potential) or (self.potential is not None and not isfinite(self.potential)):
            raise ValueError('distributed load potential is not representable')


@dataclass(frozen=True)
class NativeDistributed:
    coordinates: np.ndarray
    nodal_weights: np.ndarray
    cell_first_moments: np.ndarray
    cell_measures: np.ndarray
    force_density: np.ndarray
    couple_density: np.ndarray
    cell_couples: np.ndarray
    order: int

    def __post_init__(self):
        for name, shape in (('coordinates', (3, 3)), ('nodal_weights', (3,)),
                            ('cell_first_moments', (2, 3)), ('cell_measures', (2,)),
                            ('force_density', (3,)), ('couple_density', (3,)),
                            ('cell_couples', (2, 3))):
            object.__setattr__(self, name, _owned(_finite_array(getattr(self, name), shape, name)))
        if type(self.order) is not int or self.order not in (4, 8):
            raise ValueError('distributed load requires element quadrature order 4 or 8')
        if np.any(self.nodal_weights <= 0.) or np.any(self.cell_measures <= 0.):
            raise ValueError('positive representable distributed load measures required')

    def evaluate(self, positions, position_low, rotations):
        return self._evaluate(positions, position_low, rotations,
                              self.force_density, self.couple_density, self.cell_couples)

    def evaluate_density(self, positions, position_low, rotations, force_density, couple_density):
        """Evaluate a stage's densities using captured geometry and quadrature.

        Zero stages are valid. No density or result borrows caller storage, and
        the original captured load remains unchanged.
        """
        force = _finite_array(force_density, (3,), 'spatial reference force density')
        couple = _finite_array(couple_density, (3,), 'spatial reference couple density')
        couples = _product(self.cell_measures[:, None], couple[None, :])
        return self._evaluate(positions, position_low, rotations, force, couple, couples)

    def _evaluate(self, positions, position_low, rotations, force_density, couple_density, cell_couples):
        work = evaluate_reference_vector_work(
            self.coordinates, self.nodal_weights, self.cell_first_moments,
            positions, position_low, rotations, force_density)
        force = work.gradient.copy()
        for cell in (0, 1):
            sl = slice(18+3*cell, 21+3*cell)
            force[sl] = [_sum((force[18+3*cell+i], cell_couples[cell, i])) for i in range(3)]
        moment = [_sum((work.total_moment[i], *cell_couples[:, i])) for i in range(3)]
        # Never use the couple torque in a conservative chart-skew correction.
        potential = None if np.any(couple_density) else work.value
        return DistributedWork(force, work.spatial_jacobian, work.total_force,
                               moment, work.value, potential)


def compile_native_distributed(operator, force_density, couple_density):
    """Own geometry and densities; evaluation never reads the source operator."""
    if type(operator) not in (RetainedGeneralizedOperator, RetainedFibreOperator):
        raise ValueError('exact generalized or physical-fibre retained operator required')
    operator.guard()
    reference, order = operator.reference, operator.order
    if type(reference) is not Reference or type(order) is not int or order not in (4, 8):
        raise ValueError('native centered reference and element quadrature required')
    force = _finite_array(force_density, (3,), 'spatial reference force density')
    couple = _finite_array(couple_density, (3,), 'spatial reference couple density')
    nodes = _finite_array(reference.coordinates, (3, 3), 'distributed reference coordinates')
    nodal_terms = [[], [], []]
    lift_terms = [[], []]
    measure_terms = [[], []]
    points, weights = np.polynomial.legendre.leggauss(order)
    for cell in (0, 1):
        for point, weight in zip(points, weights):
            t = float((point+1)/2)
            measure = float(weight*reference.jacobian(cell-1+t)/2)
            if not isfinite(measure) or measure <= 0.:
                raise ValueError('positive representable distributed reference measure required')
            lift = _finite_array(reference.half_cell_lift(cell, t), (3,), 'distributed reference lift')
            nodal_terms[cell].append(float(_product(measure, 1-t)))
            nodal_terms[cell+1].append(float(_product(measure, t)))
            lift_terms[cell].append(_product(lift, measure))
            measure_terms[cell].append(measure)
    nodal_weights = np.array([_sum(row) for row in nodal_terms])
    moments = np.array([[_sum(row[i] for row in rows) for i in range(3)] for rows in lift_terms])
    measures = np.array([_sum(row) for row in measure_terms])
    cell_couples = _product(measures[:, None], couple[None, :])
    operator.guard()
    return NativeDistributed(nodes, nodal_weights, moments, measures, force, couple, cell_couples, order)
