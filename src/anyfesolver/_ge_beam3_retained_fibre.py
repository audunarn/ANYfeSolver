"""Private finite-rotation retained physical-fibre beam potential.

Pi(q,p;origin)=p.k(q)-Psi*(p;origin). The geometry is the preserved centered
two-cell kinematic map; material comes from the coupled physical-fibre cell.
No history commit, hidden static condensation or public routing occurs here.
"""
from dataclasses import dataclass
from math import fsum
import numpy as np

from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_fibre_cell import FibreCellConjugate, CellHistory
from ._ge_beam3_fibre_section import PhysicalFibreSection
from ._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, transpose, so3_exp, so3_log
from ._ge_beam3_p5.compensated import compensated_strain
from ._ge_beam3_p5.compensated_coordinates import validate_pair
from ._ge_beam3_p5.algebra import HALVES
from ._ge_beam3_p5.arrays import _array, _frames
from ._native_reference_modal import _owned
from ._ge_beam3_p5_seeded.core import canonical, sha


POLICY = 'CANDIDATE_GE_BEAM3_RETAINED_PHYSICAL_FIBRE_V1'


@dataclass(frozen=True)
class Evaluation:
    potential: float
    residual: np.ndarray
    hessian: np.ndarray
    hessian_low: np.ndarray
    kinematics: np.ndarray
    history: CellHistory
    material: bytes
    production_qualified: bool = False


@dataclass(frozen=True)
class BoundaryEvaluation(Evaluation):
    """New-policy-only owned payload; historical Evaluation schema stays fixed."""
    trial_material: object = None


@dataclass(frozen=True)
class MaterialRecovery:
    """Native beam material quantities in J, not the mixed saddle potential."""
    stations: tuple
    history: CellHistory
    stored_energy: tuple[float, float]
    previous_hardening_energy: tuple[float, float]
    incremental_dissipation: tuple[float, float]
    material_potential: tuple[float, float]
    energy_recovery_station_updates: int


class RetainedFibreOperator:
    __slots__ = ('reference', 'section', 'order', 'cell', 'stations', 'identity', '_reference_identity', '_sealed', 'arithmetic_policy', '_arithmetic_capture')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False): raise AttributeError('retained fibre operator is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, reference, section, *, order, arithmetic_policy=None):
        from ._stiffened_shell_f06_increment_arithmetic import POLICY as compensated
        if arithmetic_policy is not None and (type(arithmetic_policy) is not str or arithmetic_policy != compensated):
            raise ValueError('explicit physical fibre arithmetic policy required')
        if type(reference) is not Reference or type(section) is not PhysicalFibreSection:
            raise ValueError('explicit centered reference and native physical fibre section required')
        if type(order) is not int or order not in (4, 8):
            raise ValueError('private retained fibre entry supports explicit quadrature order 4 or 8')
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
        self.cell = FibreCellConjugate(section, data); self.stations = tuple(stations)
        self._reference_identity = self.reference.fingerprint()
        self.identity = sha(dict(formulation_id=POLICY, section=section.identity,
            reference=self._reference_identity, quadrature=order, cell=self.cell.identity))
        self.arithmetic_policy = arithmetic_policy
        if arithmetic_policy is not None:
            self.identity = sha(dict(source_operator=self.identity, arithmetic_policy=arithmetic_policy))
        self._arithmetic_capture = (self.arithmetic_policy, self.identity)
        self._sealed = True

    def guard(self):
        if (self.arithmetic_policy, self.identity) != self._arithmetic_capture:
            raise ValueError('captured fibre arithmetic policy changed')
        if self.reference.fingerprint() != self._reference_identity:
            raise ValueError('captured retained fibre reference changed')

    def evaluate(self, positions, position_low, vertices, rotations, resultants, *, origin=None, increment=None, check=None,
                 capture_trial=False, vertices_low=None, rotations_low=None, resultants_low=None):
        if type(capture_trial) is not bool:
            raise ValueError('explicit fibre trial capture policy required')
        self.guard()
        if check is not None: check()
        x = _array(positions, (3, 3), 'fibre positions'); low = _array(position_low, (3, 3), 'fibre low positions')
        for h, l in zip(x.flat, low.flat): validate_pair(float(h), float(l))
        q = _frames(vertices, 3, 'fibre nodal frames'); u = _frames(rotations, 2, 'fibre cell rotations')
        p = _array(resultants, (18,), 'fibre retained resultants')
        if self.arithmetic_policy is not None:
            if increment is not None:
                raise ValueError('durable compensated state requires re-centered spatial evaluation')
            ql = np.zeros_like(q) if vertices_low is None else _array(vertices_low, q.shape, 'native frame corrections')
            ul = np.zeros_like(u) if rotations_low is None else _array(rotations_low, u.shape, 'cell frame corrections')
            pl = np.zeros_like(p) if resultants_low is None else _array(resultants_low, p.shape, 'resultant corrections')
            from ._stiffened_shell_f06_increment_arithmetic import validate_frames, validate_pairs, retained_evaluation
            validate_frames(q, ql, 'nodal frames'); validate_frames(u, ul, 'cell frames'); validate_pairs(p, pl, 'resultants')
            response = self.cell.response(p.tolist(), origin, check=check, resultants_low=pl.tolist(),
                                          **({'capture_trial': True} if capture_trial else {}))
            trial_material = response.pop('_trial_state', None)
            fields = retained_evaluation(self.reference, x, low, q, ql, u, ul, p, pl, response)
            fields = tuple(float(v) if i == 0 else _owned(v) for i, v in enumerate(fields)) + (response['history'], canonical(response))
            return BoundaryEvaluation(*fields, trial_material=trial_material) if capture_trial else Evaluation(*fields)
        if any(v is not None for v in (vertices_low, rotations_low, resultants_low)):
            raise ValueError('native corrections require selected durable compensated policy')
        delta = np.zeros(42) if increment is None else _array(increment, (42,), 'fibre chart increment')
        angular = [delta[6*n+3:6*n+6] for n in range(3)]+[delta[18+3*c:21+3*c] for c in (0, 1)]
        if max(np.linalg.norm(v) for v in angular) >= .9*np.pi:
            raise ValueError('retained fibre rotation chart requires cutback')
        variables = [Jet2.variable(value, i, 24) for i, value in enumerate(delta[:24])]
        made_q = [matmul(so3_exp(variables[6*n+3:6*n+6]), constant_matrix(q[n], 24)) for n in range(3)]
        z = []; ell = []
        for cell, (left, right) in enumerate(HALVES):
            made_u = matmul(so3_exp(variables[18+3*cell:21+3*cell]), constant_matrix(u[cell], 24))
            z.extend(compensated_strain(made_u, self.reference.coordinates, x, low, variables, left, right))
            for node, sign in ((left, -1), (right, 1)):
                frame = matmul(made_u, constant_matrix(self.reference.nodal_triads[node], 24))
                ell.extend(sign*v for v in so3_log(matmul(transpose(frame), made_q[node])))
        values = z+ell; k = np.array([v.value for v in values]); j = np.array([v.gradient for v in values])
        p = p+delta[24:]
        response = self.cell.response(p.tolist(), origin, check=check,
                                      **({'capture_trial': True} if capture_trial else {}))
        trial_material = response.pop('_trial_state', None)
        gh, gl = response['gradient']
        compatibility = [fsum((float(v), -h, -l)) for v, h, l in zip(k, gh, gl)]
        residual = np.r_[j.T@p, compatibility]
        geometric = sum((v.hessian*force for v, force in zip(values, p)), np.zeros((24, 24)))
        ch = np.array([row[0] for row in response['hessian']]); cl = np.array([row[1] for row in response['hessian']])
        hessian = np.block([[geometric, j.T], [j, -ch]])
        hessian_low = np.zeros((42, 42)); hessian_low[24:, 24:] = -cl
        potential = fsum([*(float(a)*float(b) for a, b in zip(p, k)),
                         -response['potential'][0][0], -response['potential'][1][0]])
        if not np.isfinite(potential): raise ValueError('finite retained fibre potential required')
        self.guard()
        if check is not None: check()
        fields = (potential, _owned(residual), _owned(hessian), _owned(hessian_low), _owned(k),
                  response['history'], canonical(response))
        return BoundaryEvaluation(*fields, trial_material=trial_material) if capture_trial else Evaluation(*fields)

    def _recovery_input(self, rotations, resultants, rotations_low, resultants_low):
        u = _frames(rotations, 2, 'retained fibre recovery rotations')
        p = _array(resultants, (18,), 'retained fibre recovery resultants')
        extra = {}
        if rotations_low is not None or resultants_low is not None:
            if self.arithmetic_policy is None:
                raise ValueError('native corrections require selected durable compensated policy')
            ul = np.zeros_like(u) if rotations_low is None else _array(rotations_low, u.shape, 'recovery frame corrections')
            pl = np.zeros_like(p) if resultants_low is None else _array(resultants_low, p.shape, 'recovery resultant corrections')
            from ._stiffened_shell_f06_increment_arithmetic import validate_pairs, validate_frames
            validate_frames(u, ul, 'recovery cell frames'); validate_pairs(p, pl, 'recovery resultants')
            u = u+ul
            extra['resultants_low'] = pl.tolist()
        return u, p, extra

    def recover(self, rotations, resultants, *, origin=None, check=None, rotations_low=None, resultants_low=None):
        self.guard()
        u, p, extra = self._recovery_input(rotations, resultants, rotations_low, resultants_low)
        response = self.cell.response(p.tolist(), origin, check=check, **extra)
        rows = self._recovery_rows(u, response)
        self.guard()
        if check is not None: check()
        return rows

    def _recovery_rows(self, u, response):
        rows = []
        for (cell, index, xi, weight, ref), fields, history in zip(self.stations, response['stations'], response['history'].stations):
            rows.append(dict(cell=cell, station=index, xi=xi, measure=weight,
                strain=_owned(fields['strain'][0]), strain_low=_owned(fields['strain'][1]),
                resultants=_owned(fields['resultants'][0]), resultants_low=_owned(fields['resultants'][1]),
                fibres=fields['fibres'], history=history, reference_frame=ref,
                current_frame=_owned(u[cell]@ref), formulation_id=POLICY,
                recovery_policy='PAIRED_PHYSICAL_FIBRE_RETAINED_FIELDS_V1'))
        return tuple(rows)

    def recover_material(self, rotations, resultants, *, origin=None, check=None, rotations_low=None, resultants_low=None):
        """One local solve plus final accepted return-map energy capture.

        Default evaluate/recover payloads and their identities remain unchanged.
        The supplied origin is the last accepted material predecessor.
        """
        self.guard()
        u, p, extra = self._recovery_input(rotations, resultants, rotations_low, resultants_low)
        response = self.cell.response(p.tolist(), origin, check=check, include_energy=True, **extra)
        rows = self._recovery_rows(u, response)
        self.guard()
        if check is not None: check()
        return MaterialRecovery(rows, response['history'], response['stored_energy'],
            response['previous_hardening_energy'], response['incremental_dissipation'],
            tuple(value[0] for value in response['material_potential']),
            response['energy_recovery_station_updates'])
