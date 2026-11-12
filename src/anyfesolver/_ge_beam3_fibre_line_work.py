"""Private conservative spatial force per reference arclength on the lifted beam.

W = integral f . [sum N_i (x_i-X_i) + (U-I) c_0] ds_0.
The cell-rotation work is retained, not replaced with nodal lumping. Derivatives
are analytic Exp-chart variations of this load-only potential; material energy
is never subtracted to recover a small external work. No public routing changes.
"""
from dataclasses import dataclass
from math import isfinite
import numpy as np

from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_p5.arrays import _array, _frames
from ._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair
from ._ge_beam3_p5.compensated import sum_jets
from ._ge_beam3_mixed_ad import Jet2, so3_exp, matmul, constant_matrix
from ._native_reference_modal import _owned


POLICY = 'GE_BEAM3_RETAINED_FIBRE_SPATIAL_DEAD_REFERENCE_LINE_FORCE_V1'


@dataclass(frozen=True)
class ReferenceLineForces:
    rows: tuple  # Unique ascending (element_id, fx, fy, fz).

    def __post_init__(self):
        if type(self.rows) is not tuple or not 1 <= len(self.rows) <= 64:
            raise ValueError('one to 64 explicit reference line-force rows required')
        ids = []
        for row in self.rows:
            if (type(row) is not tuple or len(row) != 4 or type(row[0]) is not int or row[0] <= 0
                    or any(type(v) is not float or not isfinite(v) for v in row[1:])):
                raise ValueError('positive element and finite binary64 line-force components required')
            if not any(v != 0. for v in row[1:]):
                raise ValueError('each explicit line-force row must be nonzero')
            ids.append(row[0])
        if ids != sorted(set(ids)):
            raise ValueError('unique ascending line-force element IDs required')

    def descriptor(self):
        return dict(policy=POLICY, rows=self.rows, axes='FIXED_SPATIAL',
            measure='REFERENCE_ARCLENGTH', quadrature='ELEMENT_STIFFNESS_RULE',
            field='CENTERED_TWO_CELL_OBJECTIVE_LIFT', conservative_potential=True,
            internal_load_work_retained=True, production_qualified=False)


@dataclass(frozen=True)
class Work:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray


def evaluate(reference, positions, position_low, rotations, force, *, order):
    if type(reference) is not Reference or type(order) is not int or order not in (4, 8):
        raise ValueError('native centered reference and element quadrature required')
    x = _array(positions, (3, 3), 'line-load positions')
    low = _array(position_low, (3, 3), 'line-load low positions')
    rotations = _frames(rotations, 2, 'line-load cell rotations')
    force = _array(force, (3,), 'spatial reference line force')
    for a, b in zip(x.flat, low.flat): validate_pair(float(a), float(b))
    # Only 24 geometric coordinates carry work. The final 18 retained stress
    # coordinates are explicitly zero, not inferred by a material subtraction.
    delta = [Jet2.variable(0., i, 24) for i in range(24)]
    points, weights = np.polynomial.legendre.leggauss(order)
    terms = []
    for cell in (0, 1):
        u = matmul(so3_exp(delta[18+3*cell:21+3*cell]), constant_matrix(rotations[cell], 24))
        for point, weight in zip(points, weights):
            t = float((point+1)/2); xi = cell-1+t
            lift = reference.half_cell_lift(cell, t)
            measure = float(weight*reference.jacobian(xi)/2)
            for i in range(3):
                displacement = [(u[i][k]-(1 if i == k else 0))*lift[k] for k in range(3)]
                for node, shape in ((cell, 1-t), (cell+1, t)):
                    high, tail = split_sum((float(x[node, i]), float(low[node, i]),
                                           -float(reference.coordinates[node, i])))
                    displacement.extend((Jet2.constant(shape*high, 24),
                                         Jet2.constant(shape*tail, 24), shape*delta[6*node+i]))
                terms.append(measure*force[i]*sum_jets(displacement))
    work = sum_jets(terms)
    gradient = np.zeros(42); gradient[:24] = work.gradient
    hessian = np.zeros((42, 42)); hessian[:24, :24] = work.hessian
    if not isfinite(work.value) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise ValueError('nonfinite native line work')
    return Work(float(work.value), _owned(gradient), _owned(hessian))
