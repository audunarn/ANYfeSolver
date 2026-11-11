"""Private finite-rotation retained generalized ellipsoid beam potential.

Kinematic map ported unchanged from the frozen retained physical-fibre operator.
Pi(q,p;origin)=p.k(q)-Psi*(p;origin). The geometry is the preserved centered
two-cell kinematic map; material comes from the coupled generalized ellipsoid cell.
No history commit, hidden static condensation or public routing occurs here.
"""
from dataclasses import dataclass
from math import fsum
import numpy as np

from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_generalized_cell import GeneralizedCellConjugate, GeneralizedCellHistory
from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from ._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, transpose, so3_exp, so3_log
from ._ge_beam3_p5.compensated import compensated_strain
from ._ge_beam3_centered_so3_strain import (POLICY as STABLE_SO3_POLICY,
    rotation_jets as stable_rotation_jets, strain as stable_strain)
from ._ge_beam3_p5.compensated_coordinates import validate_pair
from ._ge_beam3_p5.algebra import HALVES
from ._ge_beam3_p5.arrays import _array, _frames
from ._native_reference_modal import _owned
from ._ge_beam3_reference_identity import CapturedReferenceIdentity
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_precise_geometric_work import POLICY as PRECISE_WORK_POLICY, potential as precise_work
from ._stiffened_shell_f06_increment_arithmetic import POLICY as COMPENSATED_STATE_POLICY


POLICY = 'CANDIDATE_GE_BEAM3_RETAINED_GENERALIZED_ELLIPSOID_V1'


@dataclass(frozen=True)
class Evaluation:
    potential: float
    residual: np.ndarray
    hessian: np.ndarray
    hessian_low: np.ndarray
    kinematics: np.ndarray
    history: GeneralizedCellHistory
    material: bytes
    production_qualified: bool = False


class RetainedGeneralizedOperator:
    __slots__ = ('reference', 'section', 'order', 'cell', 'stations', 'identity', '_reference_identity', '_reference_binding', '_sealed', 'arithmetic_policy', '_arithmetic_capture')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False): raise AttributeError('retained generalized operator is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, reference, section, *, order, arithmetic_policy=None):
        if arithmetic_policy is not None and (type(arithmetic_policy) is not str or arithmetic_policy not in (PRECISE_WORK_POLICY, STABLE_SO3_POLICY, COMPENSATED_STATE_POLICY)):
            raise ValueError('explicit registered geometric arithmetic policy required')
        if type(reference) is not Reference or type(section) is not EllipsoidalGeneralizedSection:
            raise ValueError('explicit centered reference and native generalized ellipsoid section required')
        if type(order) is not int or order not in (4, 8):
            raise ValueError('private retained generalized entry supports explicit quadrature order 4 or 8')
        self.reference = Reference(reference.coordinates, reference.nodal_triads,
            regularity_relative_tolerance=reference._regularity_relative_tolerance,
            rotation_tolerance=reference._rotation_tolerance, frame_tolerance=reference._frame_tolerance)
        self.section = section; self.order = order
        points, weights = np.polynomial.legendre.leggauss(order)
        data = []; stations = []
        for cell in (0, 1):
            for index, (point, weight) in enumerate(zip(points, weights)):
                t = float((point+1)/2); xi = cell-1+t; jacobian = self.reference.jacobian(xi)
                measure = float(weight*jacobian/2); frame = _owned(self.reference.frame(xi))
                data.append(dict(cell=cell, t=t, weight=measure, v=(frame.T/jacobian).tolist()))
                stations.append((cell, index, xi, measure, frame))
        self.cell = GeneralizedCellConjugate(section, data); self.stations = tuple(stations)
        self._reference_binding = CapturedReferenceIdentity(self.reference)
        self._reference_identity = self._reference_binding.require(self.reference)
        self.identity = sha(dict(formulation_id=POLICY, section=section.identity,
            reference=self._reference_identity, quadrature=order, cell=self.cell.identity))
        self.arithmetic_policy = arithmetic_policy
        if arithmetic_policy is not None:
            self.identity = sha(dict(source_operator=self.identity, arithmetic_policy=arithmetic_policy))
        self._arithmetic_capture = (self.arithmetic_policy, self.identity)
        self._sealed = True

    def guard(self):
        if (self.arithmetic_policy, self.identity) != self._arithmetic_capture:
            raise ValueError('captured arithmetic policy changed')
        self.cell.guard()
        self.reference_identity()

    def reference_identity(self):
        if self._reference_binding.require(self.reference) != self._reference_identity:
            raise ValueError('captured retained generalized reference changed')
        return self._reference_identity

    def evaluate(self, positions, position_low, vertices, rotations, resultants, *, origin=None, increment=None, check=None, vertices_low=None, rotations_low=None, resultants_low=None):
        self.guard()
        if check is not None: check()
        x = _array(positions, (3, 3), 'generalized positions'); low = _array(position_low, (3, 3), 'generalized low positions')
        for h, l in zip(x.flat, low.flat): validate_pair(float(h), float(l))
        q = _frames(vertices, 3, 'generalized nodal frames'); u = _frames(rotations, 2, 'generalized cell rotations')
        p = _array(resultants, (18,), 'generalized retained resultants')
        if self.arithmetic_policy == COMPENSATED_STATE_POLICY:
            if increment is not None:raise ValueError('durable compensated state requires re-centered spatial evaluation')
            ql=np.zeros_like(q) if vertices_low is None else _array(vertices_low,q.shape,'native frame corrections')
            ul=np.zeros_like(u) if rotations_low is None else _array(rotations_low,u.shape,'cell frame corrections')
            pl=np.zeros_like(p) if resultants_low is None else _array(resultants_low,p.shape,'resultant corrections')
            from ._stiffened_shell_f06_increment_arithmetic import validate_frames,validate_pairs
            validate_frames(q,ql,'nodal frames');validate_frames(u,ul,'cell frames');validate_pairs(p,pl,'resultants')
            response=self.cell.response(p.tolist(),origin,check=check,resultants_low=pl.tolist())
            from ._stiffened_shell_f06_increment_arithmetic import retained_evaluation
            fields=retained_evaluation(self.reference,x,low,q,ql,u,ul,p,pl,response)
            return Evaluation(*(float(v) if i==0 else _owned(v) for i,v in enumerate(fields)),
                response['history'],canonical(response))
        if any(v is not None for v in (vertices_low,rotations_low,resultants_low)):
            raise ValueError('native corrections require selected durable compensated policy')
        delta = np.zeros(42) if increment is None else _array(increment, (42,), 'generalized chart increment')
        angular = [delta[6*n+3:6*n+6] for n in range(3)]+[delta[18+3*c:21+3*c] for c in (0, 1)]
        if max(np.linalg.norm(v) for v in angular) >= .9*np.pi:
            raise ValueError('retained generalized rotation chart requires cutback')
        variables = [Jet2.variable(value, i, 24) for i, value in enumerate(delta[:24])]
        stable = self.arithmetic_policy == STABLE_SO3_POLICY
        if stable:
            prepared_q = [stable_rotation_jets(q[n], variables[6*n+3:6*n+6]) for n in range(3)]
            made_q = [row[0] for row in prepared_q]
            base_q_delta = np.array([row[2] for row in prepared_q]); base_u_delta = []
        else:
            made_q = [matmul(so3_exp(variables[6*n+3:6*n+6]), constant_matrix(q[n], 24)) for n in range(3)]
        z = []; ell = []
        for cell, (left, right) in enumerate(HALVES):
            if stable:
                made_u, made_delta, base_delta = stable_rotation_jets(u[cell], variables[18+3*cell:21+3*cell])
                base_u_delta.append(base_delta)
                z.extend(stable_strain(made_delta, self.reference.coordinates, x, low, variables, left, right))
            else:
                made_u = matmul(so3_exp(variables[18+3*cell:21+3*cell]), constant_matrix(u[cell], 24))
                z.extend(compensated_strain(made_u, self.reference.coordinates, x, low, variables, left, right))
            for node, sign in ((left, -1), (right, 1)):
                frame = matmul(made_u, constant_matrix(self.reference.nodal_triads[node], 24))
                ell.extend(sign*v for v in so3_log(matmul(transpose(frame), made_q[node])))
        values = z+ell; k = np.array([v.value for v in values]); j = np.array([v.gradient for v in values])
        p = p+delta[24:]
        response = self.cell.response(p.tolist(), origin, check=check)
        gh, gl = response['gradient']
        compatibility = [fsum((float(v), -h, -l)) for v, h, l in zip(k, gh, gl)]
        residual = np.r_[j.T@p, compatibility]
        geometric = sum((v.hessian*force for v, force in zip(values, p)), np.zeros((24, 24)))
        ch = np.array([row[0] for row in response['hessian']]); cl = np.array([row[1] for row in response['hessian']])
        hessian = np.block([[geometric, j.T], [j, -ch]])
        hessian_low = np.zeros((42, 42)); hessian_low[24:, 24:] = -cl
        potential = fsum([*(float(a)*float(b) for a, b in zip(p, k)),
                         -response['potential'][0][0], -response['potential'][1][0]])
        if stable:
            potential = precise_work(self.reference.coordinates, self.reference.nodal_triads,
                x, low, q, u, delta, p, [response['potential'][0][0], response['potential'][1][0]],
                check=check if check is not None else lambda: None,
                effective_nodal_deltas=base_q_delta, effective_cell_deltas=np.array(base_u_delta))
        elif self.arithmetic_policy == PRECISE_WORK_POLICY:
            potential = precise_work(self.reference.coordinates, self.reference.nodal_triads,
                x, low, q, u, delta, p, [response['potential'][0][0], response['potential'][1][0]],
                check=check if check is not None else lambda: None)
        if not np.isfinite(potential): raise ValueError('finite retained generalized potential required')
        self.guard()
        if check is not None: check()
        return Evaluation(potential, _owned(residual), _owned(hessian), _owned(hessian_low), _owned(k),
                          response['history'], canonical(response))

    def recover(self, rotations, resultants, *, origin=None, check=None, rotations_low=None, resultants_low=None):
        self.guard(); u = _frames(rotations, 2, 'retained generalized recovery rotations')
        p = _array(resultants, (18,), 'retained generalized recovery resultants')
        low = None if resultants_low is None else _array(resultants_low, (18,), 'recovery resultant corrections')
        ulow = None if rotations_low is None else _array(rotations_low, (2,3,3), 'recovery frame corrections')
        if (low is not None or ulow is not None) and self.arithmetic_policy != COMPENSATED_STATE_POLICY:
            raise ValueError('compensated recovery requires durable state policy')
        if self.arithmetic_policy==COMPENSATED_STATE_POLICY:
            from ._stiffened_shell_f06_increment_arithmetic import validate_frames,validate_pairs
            validate_frames(u,np.zeros_like(u) if ulow is None else ulow,'recovery frames')
            validate_pairs(p,np.zeros_like(p) if low is None else low,'recovery resultants')
        response = self.cell.response(p.tolist(), origin, check=check, resultants_low=None if low is None else low.tolist())
        rows = []
        for (cell, index, xi, weight, ref), fields, history in zip(self.stations, response['stations'], response['history'].stations):
            rows.append(dict(cell=cell, station=index, xi=xi, measure=weight,
                strain=_owned(fields['strain'][0]), strain_low=_owned(fields['strain'][1]),
                resultants=_owned(fields['resultants'][0]), resultants_low=_owned(fields['resultants'][1]),
                history=history, reference_frame=ref,
                current_frame=_owned((u[cell] if ulow is None else u[cell]+ulow[cell])@ref), formulation_id=POLICY,
                recovery_policy='PAIRED_GENERALIZED_RESULTANT_RETAINED_FIELDS_V1'))
        self.guard()
        if check is not None: check()
        return tuple(rows)
