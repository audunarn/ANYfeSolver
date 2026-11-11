"""Private stable-reference stationary operator successor.

The physical variational operator, local solve and section law are inherited
from the preserved P5 package. Only reference evaluation and algebraically
equivalent spatial-dead-load work summation change here. Native package V2
has not yet adopted this class; no public selector or qualification is implied.
"""

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import (
    Jet2, constant_matrix, matmul, matvec, transpose, so3_exp, so3_log,
)
from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5.compensated import CompensatedStationaryBeam, sum_jets, compensated_strain
from anyfesolver._ge_beam3_p5.mixed import (
    MixedBeamEvaluation, BeamStationTrial, NonlinearLocalError, compose_density,
)
from anyfesolver._ge_beam3_p5.section import DirectedHardeningSection, SectionHistory, _history
from anyfesolver._ge_beam3_p5.algebra import HALVES
from anyfesolver._ge_beam3_p5.arrays import _array, _frames, _readonly
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair


SCHEMA = 'GE_BEAM3_CENTERED_COMPENSATED_STATIONARY_DEVELOPMENT_V1'


class CenteredStationaryBeam(CompensatedStationaryBeam):
    def __init__(self, reference, section, *, position_low, order=24, origins=None, line_force=None):
        if type(reference) not in (CurvedBeam3ReferenceGeometry, CenteredCurvedBeam3ReferenceGeometry):
            raise ValueError('explicit admitted Q2 reference geometry required')
        if type(section) is not DirectedHardeningSection:
            raise ValueError('declared directed-hardening section required')
        if type(order) is not int or order not in (4, 8, 24):
            raise ValueError('registered quadrature order required')
        self.reference = CenteredCurvedBeam3ReferenceGeometry(reference.coordinates, reference.nodal_triads,
            regularity_relative_tolerance=reference._regularity_relative_tolerance,
            rotation_tolerance=reference._rotation_tolerance, frame_tolerance=reference._frame_tolerance)
        self.section, self.order = section, order
        origins = tuple(SectionHistory() for _ in range(2*order)) if origins is None else tuple(origins)
        if len(origins) != 2*order: raise ValueError('one fixed origin per station required')
        self.origins = tuple(_history(value) for value in origins)
        self.line_force = _readonly(np.zeros(3) if line_force is None else _array(line_force, (3,), 'line force'))
        self._position_low = _readonly(_array(position_low, (3, 3), 'coordinate low parts'))
        points, weights = np.polynomial.legendre.leggauss(order)
        stations = []
        for cell in (0, 1):
            values = []
            for index, (point, weight) in enumerate(zip(points, weights)):
                t = float((point+1)/2); xi = cell-1+t
                jacobian = self.reference.jacobian(xi)
                values.append((index, t, xi, weight*jacobian/2, self.reference.frame(xi).T/jacobian,
                               self.reference.half_cell_lift(cell, t), self.reference.position(xi)))
            stations.append(tuple(values))
        self._stations = tuple(stations)

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
                    # r_h-r0 = I(x-X) + (U-I)*reference_lift. The
                    # uncentered form multiplies world positions by weights
                    # before cancellation, losing load work at large offsets.
                    for i in range(3):
                        position_terms = [(made_u[i][k]-(1 if i == k else 0))*offset[k]
                                          for k in range(3)]
                        for node, weight in ((left, 1-t), (right, t)):
                            dh, dl = split_sum((float(x[node, i]), float(self._position_low[node, i]),
                                                -float(self.reference.coordinates[node, i])))
                            position_terms.extend((Jet2.constant(weight*dh, 36),
                                Jet2.constant(weight*dl, 36), weight*variables[6*node+i]))
                        terms.append(-measure*self.line_force[i]*sum_jets(position_terms))
                station_trials.append(BeamStationTrial(cell, index, xi, density.section))
        total = sum_jets(terms)
        if not (np.isfinite(total.value) and np.isfinite(total.gradient).all() and np.isfinite(total.hessian).all()):
            raise NonlinearLocalError('nonfinite mixed functional')
        return MixedBeamEvaluation(total.value, _readonly(total.gradient), _readonly(total.hessian), tuple(station_trials))
