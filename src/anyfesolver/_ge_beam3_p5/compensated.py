"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

import math
import numpy as np

from anyfesolver._ge_beam3_mixed_ad import (
    Jet2, constant_matrix, matmul, matvec, transpose, so3_exp, so3_log,
)
from anyfesolver._ge_beam3_p5.mixed import (
    StationaryMixedBeam, MixedBeamEvaluation, BeamStationTrial,
    NonlinearLocalError, compose_density,
)
from anyfesolver._ge_beam3_p5.algebra import HALVES
from anyfesolver._ge_beam3_p5.arrays import _array, _frames, _readonly
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair


SCHEMA = 'GE_BEAM3_P5_COMPENSATED_MIXED_EVALUATION_V1'


def sum_jets(terms):
    """Analytic value/first/second variations of a sum, stable scalar value."""
    terms = tuple(terms)
    if not terms: raise ValueError('nonempty jet sum required')
    return Jet2(math.fsum(t.value for t in terms),
                sum((t.gradient for t in terms), np.zeros_like(terms[0].gradient)),
                sum((t.hessian for t in terms), np.zeros_like(terms[0].hessian)))


def compensated_strain(made_u, reference, position_high, position_low, variables, left, right):
    """All derivative terms use the two-part chord before scalar collapse.

    (U^T-I)d0 + U^T(d-d0), retaining both parts of d0 and d-d0. Coordinate
    differences are linear maps: their high jet carries the exact unit nodal
    derivatives, while their low jet is a constant at the evaluation point.
    Products with U retain the low part in rotational first/second variations.
    """
    size = variables[0].gradient.size
    base_parts, change_parts = [], []
    for i in range(3):
        baseline = (float(reference[right, i]), -float(reference[left, i]))
        base_parts.append(split_sum(baseline))
        terms = (float(position_high[right, i]), float(position_low[right, i]),
                 -float(position_high[left, i]), -float(position_low[left, i]),
                 variables[6*right+i].value, -variables[6*left+i].value,
                 -baseline[0], -baseline[1])
        high, low = split_sum(terms)
        derivative = variables[6*right+i]-variables[6*left+i]
        change_parts.append((Jet2(high, derivative.gradient, derivative.hessian), Jet2.constant(low, size)))
    u_t = transpose(made_u)
    result = []
    for i in range(3):
        terms = []
        for k in range(3):
            difference = u_t[i][k]-(1 if i == k else 0)
            terms.extend(difference*v for v in base_parts[k])
            terms.extend(u_t[i][k]*v for v in change_parts[k])
        result.append(sum_jets(terms))
    return result


class CompensatedStationaryBeam(StationaryMixedBeam):
    def __init__(self, reference, section, *, position_low, **kwargs):
        super().__init__(reference, section, **kwargs)
        self._position_low = _readonly(_array(position_low, (3, 3), 'coordinate low parts'))

    @property
    def position_low(self):
        return self._position_low.copy()

    def evaluate(self, positions, vertex_frames, cell_rotations, moments, *, increment=None):
        x = _array(positions, (3, 3), 'positions')
        q = _frames(vertex_frames, 3, 'vertex frames')
        u = _frames(cell_rotations, 2, 'cell rotations')
        m = _array(moments, (2, 2, 3), 'endpoint moments')
        for high, low in zip(x.flat, self._position_low.flat):
            validate_pair(float(high), float(low))
        delta = np.zeros(36) if increment is None else _array(increment, (36,), 'mixed increment')
        angular = [delta[6*n+3:6*n+6] for n in range(3)]+[delta[18+3*c:21+3*c] for c in (0, 1)]
        if max(np.linalg.norm(value) for value in angular) >= .9*np.pi:
            raise ValueError('rotation increment requires cutback before endpoint evaluation')
        variables = [Jet2.variable(value, i, 36) for i, value in enumerate(delta)]
        made_q = [matmul(so3_exp(variables[6*n+3:6*n+6]), constant_matrix(q[n], 36)) for n in range(3)]
        terms, station_trials = [], []
        for cell, (left, right) in enumerate(HALVES):
            made_u = matmul(so3_exp(variables[18+3*cell:21+3*cell]), constant_matrix(u[cell], 36))
            z = compensated_strain(made_u, self.reference.coordinates, x, self._position_low, variables, left, right)
            endpoints = [[Jet2.constant(m[cell, n, i], 36)+variables[24+6*cell+3*n+i]
                          for i in range(3)] for n in (0, 1)]
            for endpoint, node, sign in ((0, left, -1), (1, right, 1)):
                frame = matmul(made_u, constant_matrix(self.reference.nodal_triads[node], 36))
                ell = so3_log(matmul(transpose(frame), made_q[node]))
                terms.extend(sign*ell[i]*endpoints[endpoint][i] for i in range(3))
            for index, t, xi, measure, v, offset, reference_position in self._stations[cell]:
                gamma = matvec(constant_matrix(v, 36), z)
                moment = [(1-t)*endpoints[0][i]+t*endpoints[1][i] for i in range(3)]
                density = self.section.mixed_response([value.value for value in gamma],
                    [value.value for value in moment], self.origins[cell*self.order+index])
                terms.append(measure*compose_density(gamma+moment, density))
                if np.any(self.line_force):
                    lift = matvec(made_u, [Jet2.constant(value, 36) for value in offset])
                    for i in range(3):
                        # Keep additive translation components separate through
                        # load work too. No low part is silently dropped.
                        position_terms = [lift[i], Jet2.constant(-reference_position[i], 36)]
                        for node, weight in ((left, 1-t), (right, t)):
                            position_terms.extend((Jet2.constant(weight*x[node, i], 36),
                                Jet2.constant(weight*self._position_low[node, i], 36), weight*variables[6*node+i]))
                        terms.append(-measure*self.line_force[i]*sum_jets(position_terms))
                station_trials.append(BeamStationTrial(cell, index, xi, density.section))
        total = sum_jets(terms)
        if not (np.isfinite(total.value) and np.isfinite(total.gradient).all() and np.isfinite(total.hessian).all()):
            raise NonlinearLocalError('nonfinite mixed functional')
        return MixedBeamEvaluation(total.value, _readonly(total.gradient), _readonly(total.hessian), tuple(station_trials))
