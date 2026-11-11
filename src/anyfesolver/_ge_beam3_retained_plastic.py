"""Private retained-resultant directed-plastic beam potential; no public route.

Pi(q,p; origin)=p.k(q)-Psi*(p; origin). The complete station conjugate
retains paired gradients, compliance, fields and proposed plastic history.
No local Schur reduction, history publication, load tangent, or mass is hidden
inside this operator. Global equilibrium/state qualification remains required.
"""
from dataclasses import dataclass
from math import fsum
import numpy as np
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
from anyfesolver._ge_beam3_complementary_section import ComplementaryDirectedSection
from anyfesolver._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, transpose, so3_exp, so3_log
from anyfesolver._ge_beam3_p5.compensated import compensated_strain
from anyfesolver._ge_beam3_p5.compensated_coordinates import validate_pair
from anyfesolver._ge_beam3_p5.algebra import HALVES
from anyfesolver._ge_beam3_p5.arrays import _array, _frames
from anyfesolver._native_reference_modal import _owned
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_station_resultant_cell import StationResultantCell

POLICY = 'CANDIDATE_GE_BEAM3_RETAINED_DIRECTED_PLASTIC_V1'


@dataclass(frozen=True)
class Evaluation:
    potential: float
    residual: np.ndarray
    hessian: np.ndarray
    hessian_low: np.ndarray
    kinematics: np.ndarray
    history: np.ndarray
    material: bytes
    production_qualified: bool = False


class RetainedPlasticOperator:
    __slots__ = ('reference', 'cell', 'stations', 'identity', '_reference_identity', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False): raise AttributeError('retained plastic operator is captured')
        object.__setattr__(self, name, value)

    def __init__(self, element):
        core = element.core
        if np.any(core.line_force): raise ValueError('distributed work not implemented by retained plastic operator')
        section = ComplementaryDirectedSection(core.section)
        source = CenteredStationaryBeam(core.reference, core.section, position_low=np.zeros((3, 3)), order=core.order)
        self.reference = source.reference
        data = dict(elastic=section.elastic.tolist(), direction=section.direction.tolist(),
            yield_force=section.yield_force, hardening=section.hardening, stations=[])
        stations = []
        for cell in (0, 1):
            for index, t, xi, weight, v, _, _ in source._stations[cell]:
                data['stations'].append(dict(cell=cell, t=float(t), weight=float(weight), v=v.tolist(), origin=[0., 0.]))
                stations.append((cell, index, float(xi), float(weight), _owned(self.reference.frame(xi))))
        self.cell = StationResultantCell(data)
        self.stations = tuple(stations); self._reference_identity = self.reference.fingerprint()
        self.identity = sha(dict(formulation_id=POLICY, section=section.identity,
            reference=self._reference_identity, quadrature=core.order, cell_input=data))
        self._sealed = True

    def guard(self):
        if self.reference.fingerprint() != self._reference_identity: raise ValueError('captured retained reference changed')

    def evaluate(self, positions, position_low, vertices, rotations, resultants, *, origins=None, increment=None):
        self.guard()
        x = _array(positions, (3, 3), 'retained positions'); low = _array(position_low, (3, 3), 'retained low positions')
        for h, l in zip(x.flat, low.flat): validate_pair(float(h), float(l))
        q = _frames(vertices, 3, 'retained nodal frames'); u = _frames(rotations, 2, 'retained cell rotations')
        p = _array(resultants, (18,), 'retained forces/moments')
        delta = np.zeros(42) if increment is None else _array(increment, (42,), 'retained chart increment')
        angular = [delta[6*n+3:6*n+6] for n in range(3)]+[delta[18+3*c:21+3*c] for c in (0, 1)]
        if max(np.linalg.norm(v) for v in angular) >= .9*np.pi: raise ValueError('retained rotation chart requires cutback')
        variables = [Jet2.variable(value, i, 24) for i, value in enumerate(delta[:24])]
        made_q = [matmul(so3_exp(variables[6*n+3:6*n+6]), constant_matrix(q[n], 24)) for n in range(3)]
        z = []; ell = []
        for cell, (left, right) in enumerate(HALVES):
            made_u = matmul(so3_exp(variables[18+3*cell:21+3*cell]), constant_matrix(u[cell], 24))
            z.extend(compensated_strain(made_u, self.reference.coordinates, x, low, variables, left, right))
            for endpoint, node, sign in ((0, left, -1), (1, right, 1)):
                frame = matmul(made_u, constant_matrix(self.reference.nodal_triads[node], 24))
                ell.extend(sign*v for v in so3_log(matmul(transpose(frame), made_q[node])))
        values = z+ell; k = np.array([v.value for v in values]); j = np.array([v.gradient for v in values])
        p = p+delta[24:]
        response = self.cell.response(p.tolist(), origins=origins)
        gh, gl = response['gradient']
        compatibility = [fsum((float(v), -h, -l)) for v, h, l in zip(k, gh, gl)]
        residual = np.r_[j.T@p, compatibility]
        geometric = sum((v.hessian*force for v, force in zip(values, p)), np.zeros((24, 24)))
        ch = np.array([row[0] for row in response['hessian']]); cl = np.array([row[1] for row in response['hessian']])
        hessian = np.block([[geometric, j.T], [j, -ch]])
        hessian_low = np.zeros((42, 42)); hessian_low[24:, 24:] = -cl
        potential = fsum([*(float(a)*float(b) for a, b in zip(p, k)),
                          -response['potential'][0][0], -response['potential'][1][0]])
        if not np.isfinite(residual).all() or not np.isfinite(hessian).all() or not np.isfinite(potential):
            raise ValueError('finite retained plastic evaluation required')
        self.guard()
        return Evaluation(potential, _owned(residual), _owned(hessian), _owned(hessian_low),
            _owned(k), _owned(response['history']), canonical(response))

    def recover(self, rotations, resultants, *, origins=None):
        self.guard(); u = _frames(rotations, 2, 'retained recovery rotations')
        p = _array(resultants, (18,), 'retained recovery resultants')
        response = self.cell.response(p.tolist(), origins=origins)
        rows = []
        for (cell, index, xi, weight, ref), fields in zip(self.stations, response['stations']):
            rows.append(dict(cell=cell, station=index, xi=xi, measure=weight,
                strain=_owned(fields['strain'][0]), strain_low=_owned(fields['strain'][1]),
                elastic_strain=_owned(fields['elastic_strain'][0]), elastic_strain_low=_owned(fields['elastic_strain'][1]),
                resultants=_owned(fields['resultants'][0]), resultants_low=_owned(fields['resultants'][1]),
                history=_owned(response['history'][len(rows)]), reference_frame=ref,
                current_frame=_owned(u[cell]@ref), formulation_id=POLICY,
                recovery_policy='PAIRED_COMPLEMENTARY_RETAINED_FIELDS_V1'))
        self.guard()
        return tuple(rows)
