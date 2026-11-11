"""Private elastic retained-resultant operator, not publicly registered.

Port of the committed research Legendre operator; the geometric potential
and compliance construction remain unchanged. Native checkpoint ownership
and complementary recovery are separate from historical V5 state layouts.
Only virgin elastic-interior sections and zero distributed work are admitted.
"""
from dataclasses import dataclass
from math import fsum
import numpy as np
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
from anyfesolver._ge_beam3_mixed_ad import (
    Jet2, constant_matrix, matmul, transpose, so3_exp, so3_log,
)
from anyfesolver._ge_beam3_p5.compensated import compensated_strain
from anyfesolver._ge_beam3_p5.algebra import HALVES
from anyfesolver._ge_beam3_p5.arrays import _array, _frames
from anyfesolver._native_reference_modal import _owned as _readonly


POLICY = 'CANDIDATE_GE_BEAM3_RETAINED_ELASTIC_V1'


@dataclass(frozen=True)
class Evaluation:
    potential: float
    residual: np.ndarray
    hessian: np.ndarray
    kinematics: np.ndarray
    production_qualified: bool = False


class RetainedElasticOperator:
    """18 nodal + 6 cell rotations + 6 cell forces + 12 endpoint moments.

Phi(z,m)=z.A.z/2+z.B.m-m.S.m/2. Its additional Legendre
transform is Psi*(a,m)=(a-Bm).A^-1.(a-Bm)/2+m.S.m/2.
Pi=a.z+m.ell-Psi*. Eliminating a reproduces Phi+m.ell.
"""
    def __init__(self, element):
        core = element.core
        if np.any(core.line_force):
            raise ValueError('distributed load not implemented in elastic probe')
        self.reference = core.reference
        original = CenteredStationaryBeam(core.reference, core.section,
            position_low=np.zeros((3, 3)), order=core.order)
        a = np.zeros((6, 6)); b = np.zeros((6, 12)); s = np.zeros((12, 12))
        for cell in (0, 1):
            cell_slots = slice(3*cell, 3*cell+3)
            for index, t, xi, weight, v, _, _ in original._stations[cell]:
                density = core.section.mixed_response(np.zeros(3), np.zeros(3),
                    original.origins[cell*core.order+index])
                if density.section.plastic_active or np.any(density.gradient):
                    raise ValueError('virgin elastic origin required')
                h = density.hessian
                n = np.zeros((3, 12))
                n[:, 6*cell:6*cell+3] = (1-t)*np.eye(3)
                n[:, 6*cell+3:6*cell+6] = t*np.eye(3)
                a[cell_slots, cell_slots] += weight*v.T@h[:3, :3]@v
                b[cell_slots, :] += weight*v.T@h[:3, 3:]@n
                s -= weight*n.T@h[3:, 3:]@n
        np.linalg.cholesky(a); np.linalg.cholesky(s)
        ai = np.linalg.solve(a, np.eye(6)); aib = np.linalg.solve(a, b)
        compliance = np.block([[ai, -aib], [-aib.T, s+b.T@aib]])
        np.linalg.cholesky(compliance)
        self.a, self.b, self.s = map(_readonly, (a, b, s))
        self.compliance = _readonly(compliance)
        self._stationary = original

    def evaluate(self, positions, position_low, vertices, rotations, resultants, *, increment=None):
        x = _array(positions, (3, 3), 'probe positions')
        low = _array(position_low, (3, 3), 'probe low positions')
        q = _frames(vertices, 3, 'probe nodal frames')
        u = _frames(rotations, 2, 'probe cell rotations')
        p = _array(resultants, (18,), 'probe cell forces and moments')
        delta = np.zeros(42) if increment is None else _array(increment, (42,), 'probe increment')
        angular = [delta[6*n+3:6*n+6] for n in range(3)]+[delta[18+3*c:21+3*c] for c in (0, 1)]
        if max(np.linalg.norm(v) for v in angular) >= .9*np.pi:
            raise ValueError('probe rotation chart requires cutback')
        variables = [Jet2.variable(value, i, 24) for i, value in enumerate(delta[:24])]
        made_q = [matmul(so3_exp(variables[6*n+3:6*n+6]), constant_matrix(q[n], 24)) for n in range(3)]
        z = []; ell = []
        for cell, (left, right) in enumerate(HALVES):
            made_u = matmul(so3_exp(variables[18+3*cell:21+3*cell]), constant_matrix(u[cell], 24))
            z.extend(compensated_strain(made_u, self.reference.coordinates, x, low, variables, left, right))
            for endpoint, node, sign in ((0, left, -1), (1, right, 1)):
                frame = matmul(made_u, constant_matrix(self.reference.nodal_triads[node], 24))
                ell.extend(sign*v for v in so3_log(matmul(transpose(frame), made_q[node])))
        values = z+ell
        k = np.array([v.value for v in values]); j = np.array([v.gradient for v in values])
        p = p+delta[24:]
        cp = self.compliance@p
        for cell in (0, 1):
            for index, t, _, _, v, _, _ in self._stationary._stations[cell]:
                moment = (1-t)*p[6+6*cell:9+6*cell]+t*p[9+6*cell:12+6*cell]
                density = self._stationary.section.mixed_response(v@cp[3*cell:3*cell+3],
                    moment, self._stationary.origins[cell*self._stationary.order+index])
                if density.section.plastic_active:
                    raise ValueError('nonlinear section branch not implemented in elastic probe')
        residual = np.r_[j.T@p, k-cp]
        geometric = sum((v.hessian*force for v, force in zip(values, p)), np.zeros((24, 24)))
        hessian = np.block([[geometric, j.T], [j, -self.compliance]])
        potential = fsum([*(float(a)*float(b) for a, b in zip(p, k)),
                          *(-.5*float(a)*float(b) for a, b in zip(p, cp))])
        if not np.isfinite(residual).all() or not np.isfinite(hessian).all() or not np.isfinite(potential):
            raise ValueError('finite full-resultant probe required')
        return Evaluation(potential, _readonly(residual), _readonly(hessian), _readonly(k))

    def recover(self, rotations, resultants):
        """Complementary station fields; no inference from rounded positions.

        Resultant moments are the authoritative interpolated mixed variables.
        Report compatibility separately at the element/global state level.
        No fibre stress or plastic history is fabricated by this adapter.
        """
        u = _frames(rotations, 2, 'retained recovery cell rotations')
        p = _array(resultants, (18,), 'retained recovery resultants')
        dual = self.compliance@p
        rows = []
        for cell in (0, 1):
            for index, t, xi, measure, v, _, _ in self._stationary._stations[cell]:
                gamma = v@dual[3*cell:3*cell+3]
                moment = (1-t)*p[6+6*cell:9+6*cell]+t*p[9+6*cell:12+6*cell]
                density = self._stationary.section.mixed_response(gamma, moment,
                    self._stationary.origins[cell*self._stationary.order+index])
                if density.section.plastic_active:
                    raise ValueError('nonlinear retained recovery not implemented')
                frame = self.reference.frame(xi)
                rows.append(dict(cell=cell, station=index, xi=float(xi), measure=float(measure),
                    strain=_readonly(np.r_[gamma, -density.gradient[3:]]),
                    resultants=_readonly(np.r_[density.gradient[:3], moment]),
                    reference_frame=_readonly(frame), current_frame=_readonly(u[cell]@frame),
                    formulation_id=POLICY, recovery_policy='COMPLEMENTARY_RETAINED_FIELDS_V1'))
        return tuple(rows)
