"""Private stress-controlled conjugate of the directed-hardening potential.

This is a section building block, not an assembled retained-beam material
adapter. Shared cell strains still require the constrained cell conjugate.
No historical section law or accepted history is modified.
"""
from dataclasses import dataclass
from math import fsum
import numpy as np
from anyfesolver._ge_beam3_p5.section import DirectedHardeningSection, SectionHistory, _history
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum
from anyfesolver._native_reference_modal import _owned
from anyfesolver._ge_beam3_p5_seeded.core import sha
from anyfesolver._dyadic_bilinear import _capture, _multiply, _add, _rounded


POLICY = 'GE_BEAM3_DIRECTED_HARDENING_FULL_SECTION_CONJUGATE_V1'


@dataclass(frozen=True)
class ComplementaryResponse:
    origin: SectionHistory
    history: SectionHistory
    resultants: np.ndarray
    elastic_strain: np.ndarray
    elastic_strain_low: np.ndarray
    total_strain: np.ndarray
    total_strain_low: np.ndarray
    compliance: np.ndarray
    compliance_factor: np.ndarray
    elastic_residual_inf: float
    dual_potential: float
    primal_incremental_potential: float
    dissipation: float
    plastic_increment: float
    branch: str
    derivative_kind: str
    section_identity: str
    production_qualified: bool = False


class ComplementaryDirectedSection:
    """Full Fenchel conjugate of the existing incremental section potential.

W*(s;z0,p0)=s.C^-1.s/2 + z0*(a.s)
             + max(|a.s|-Y-H*p0,0)^2/(2H).
Its gradient is C^-1*s+a*z_new. On a smooth plastic branch its
Hessian is C^-1+a*a^T/H. Elastic and plastic strain remain separate.
"""
    __slots__ = ('elastic', 'direction', 'yield_force', 'hardening', 'scale', 'root',
                 'elastic_compliance', 'elastic_factor', '_exact_elastic', 'identity', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False): raise AttributeError('complementary section is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, section):
        if type(section) is not DirectedHardeningSection:
            raise ValueError('explicit directed-hardening section required')
        self.elastic = _owned(section._elastic)
        self.direction = _owned(section._direction)
        self.yield_force = float(section._yield); self.hardening = float(section._hardening)
        self.scale = _owned(np.sqrt(np.diag(self.elastic)))
        self.root = _owned(np.linalg.cholesky(self.elastic/self.scale[:, None]/self.scale[None, :]))
        self.elastic_compliance = _owned(self._elastic_strain(np.eye(6)))
        self.elastic_factor = _owned(np.linalg.solve(self.root, np.diag(1/self.scale)))
        self._exact_elastic = _capture(self.elastic, lambda: None)
        self.identity = sha(dict(policy=POLICY, elastic=self.elastic, direction=self.direction,
            yield_force=self.yield_force, hardening=self.hardening))
        self._sealed = True

    def _elastic_strain(self, stress):
        if stress.ndim == 1:
            return np.linalg.solve(self.root.T, np.linalg.solve(self.root, stress/self.scale))/self.scale
        return np.linalg.solve(self.root.T, np.linalg.solve(self.root, stress/self.scale[:, None]))/self.scale[:, None]

    def response(self, resultants, origin=SectionHistory()):
        origin = _history(origin); stress = _owned(resultants)
        if stress.shape != (6,): raise ValueError('six finite section resultants required')
        drive = fsum(float(a)*float(s) for a, s in zip(self.direction, stress))
        radius = self.yield_force+self.hardening*origin.accumulated
        excess = abs(drive)-radius
        increment = max(0., excess)/self.hardening
        sign = 1. if drive >= 0 else -1.
        history = _history(SectionHistory(origin.plastic_coordinate+sign*increment,
                                         origin.accumulated+increment))
        elastic = self._elastic_strain(stress); elastic_low = np.zeros(6)
        for iteration in range(4):
            # Audit the supplied C and both strain parts without intermediate
            # cancellation rounding. The binary64 solve is a preconditioner.
            pair = _add(_capture(elastic[:, None], lambda: None),
                        _capture(elastic_low[:, None], lambda: None))
            residual = _rounded(_add(_multiply(self._exact_elastic, pair, lambda: None),
                                     _capture(-stress[:, None], lambda: None))).ravel()
            residual_inf = float(np.linalg.norm(residual, np.inf))
            if residual_inf <= 1e-13*max(1., np.linalg.norm(stress, np.inf)): break
            if iteration == 3: raise ValueError('complementary elastic refinement limit')
            correction = self._elastic_strain(-residual)
            for i in range(6):
                elastic[i], elastic_low[i] = split_sum((float(elastic[i]), float(elastic_low[i]), float(correction[i])))
        total = np.empty(6); low = np.empty(6)
        for i in range(6):
            total[i], low[i] = split_sum((float(elastic[i]), float(elastic_low[i]),
                                         float(self.direction[i]*history.plastic_coordinate)))
        compliance = self.elastic_compliance.copy()
        if increment > 0: compliance += np.outer(self.direction, self.direction)/self.hardening
        factor = np.vstack((self.elastic_factor, self.direction/np.sqrt(self.hardening))) if increment > 0 else self.elastic_factor
        elastic_energy = .5*fsum([*(float(s)*float(e) for s, e in zip(stress, elastic)),
                                  *(float(s)*float(e) for s, e in zip(stress, elastic_low))])
        dual = fsum((elastic_energy, origin.plastic_coordinate*drive, .5*max(0., excess)*increment))
        dissipation = self.yield_force*increment
        hardening_increment = .5*self.hardening*increment*(2*origin.accumulated+increment)
        primal = fsum((elastic_energy, hardening_increment, dissipation))
        if not np.isfinite([drive, radius, excess, increment, dual, primal, dissipation]).all():
            raise ValueError('finite complementary return mapping required')
        if excess == 0.:
            branch = 'YIELD_BOUNDARY'; derivative_kind = 'SEMISMOOTH_ELASTIC_SELECTION'
        else:
            branch = ('PLASTIC_POSITIVE' if drive > 0 else 'PLASTIC_NEGATIVE') if increment > 0 else 'ELASTIC'
            derivative_kind = 'CLASSICAL_SMOOTH_BRANCH'
        return ComplementaryResponse(origin, history, stress, _owned(elastic), _owned(elastic_low), _owned(total), _owned(low),
            _owned(compliance), _owned(factor), residual_inf, dual, primal, dissipation, increment, branch, derivative_kind, self.identity)
