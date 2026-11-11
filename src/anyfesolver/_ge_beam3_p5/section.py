"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

from dataclasses import dataclass
import numpy as np

def _array(value, shape):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"finite section data of shape {shape} required")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SectionHistory:
    plastic_coordinate: float = 0.
    accumulated: float = 0.


def _history(value):
    if not isinstance(value, SectionHistory):
        raise ValueError("explicit section history required")
    z, p = value.plastic_coordinate, value.accumulated
    if (isinstance(z, (bool, np.bool_)) or isinstance(p, (bool, np.bool_))
            or not np.isfinite(z) or not np.isfinite(p) or p < abs(z)):
        raise ValueError("finite history with accumulated >= absolute plastic coordinate required")
    return SectionHistory(float(z), float(p))


@dataclass(frozen=True)
class SectionResponse:
    origin: SectionHistory
    history: SectionHistory
    strain: np.ndarray
    resultants: np.ndarray
    tangent: np.ndarray
    incremental_potential: float
    stored_energy: float
    dissipation_increment: float
    plastic_active: bool


@dataclass(frozen=True)
class MixedSectionResponse:
    section: SectionResponse
    potential: float
    gradient: np.ndarray
    hessian: np.ndarray


class DirectedHardeningSection:
    """Incremental potential with plastic strain z*a and accumulated p.

    Minimize over dz: .5*(e-(z_old+dz)*a)^T C*(e-(z_old+dz)*a)
      + .5*H*((p_old+abs(dz))^2-p_old^2) + yield_force*abs(dz).
    Direction a is supplied in physical material coordinates, not normalized
    or inferred. Its units/scale are part of the declared constitutive law.
    H>0 ensures the branch tangents remain SPD for the partial transform.
    """

    def __init__(self, elastic, direction, yield_force, hardening):
        self._elastic = _array(elastic, (6, 6))
        if not np.array_equal(self._elastic, self._elastic.T):
            raise ValueError("exactly symmetric section elasticity required")
        np.linalg.cholesky(self._elastic)
        self._direction = _array(direction, (6,))
        if not np.any(self._direction):
            raise ValueError("nonzero plastic direction required")
        for value in (yield_force, hardening):
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError("finite positive yield force and hardening required")
        self._yield, self._hardening = float(yield_force), float(hardening)
        self._ca = self._elastic @ self._direction
        self._metric = float(self._direction @ self._ca)
        self._b = self._elastic[:3, 3:]
        self._d = self._elastic[3:, 3:]
        self._schur = self._elastic[:3, :3]-self._b @ np.linalg.solve(self._d, self._b.T)
        if not np.isfinite(self._metric) or self._metric <= 0 or not np.isfinite(self._schur).all():
            raise ValueError("finite positive constitutive reduction required")
        np.linalg.cholesky(self._schur)

    def _advance(self, drive, metric, history):
        excess = abs(drive)-(self._yield+self._hardening*history.accumulated)
        increment = max(0., excess/(metric+self._hardening))
        sign = 1. if drive >= 0 else -1.
        return SectionHistory(history.plastic_coordinate+sign*increment,
                              history.accumulated+increment), increment

    def _response(self, strain, origin, history, increment):
        elastic_strain = strain-history.plastic_coordinate*self._direction
        stress = self._elastic @ elastic_strain
        tangent = self._elastic.copy()
        if increment > 0:
            tangent -= np.outer(self._ca, self._ca)/(self._metric+self._hardening)
        # The declared hardening law is strictly convex. Roundoff that loses
        # this property is an error, not permission to regularize the tangent.
        np.linalg.cholesky(tangent)
        stored = float(elastic_strain @ stress/2+self._hardening*history.accumulated**2/2)
        dissipation = self._yield*increment
        potential = stored-self._hardening*origin.accumulated**2/2+dissipation
        if not np.isfinite([stored, dissipation, potential]).all():
            raise ValueError("nonfinite constitutive response")
        return SectionResponse(origin, _history(history), _array(strain, (6,)), _array(stress, (6,)),
                               _array(tangent, (6, 6)), potential, stored, dissipation, increment > 0)

    def strain_response(self, strain, history=SectionHistory()):
        strain, history = _array(strain, (6,)), _history(history)
        drive = float(self._direction @ (self._elastic @ (strain-history.plastic_coordinate*self._direction)))
        new, increment = self._advance(drive, self._metric, history)
        return self._response(strain, history, new, increment)

    def mixed_response(self, gamma, moment, history=SectionHistory()):
        """Partial Legendre density at prescribed gamma and physical moment.

        Eliminate curvature in the incremental potential, not by substituting
        a plastic tangent into the elastic P5 F/H/J formulas. The scalar return
        equation at fixed moment has metric a_gamma^T Schur(C) a_gamma.
        The full-strain return map instead has metric a^T C a.
        """
        gamma, moment, history = _array(gamma, (3,)), _array(moment, (3,)), _history(history)
        a_gamma, a_kappa = self._direction[:3], self._direction[3:]
        drive = float(a_gamma @ (self._schur @ (gamma-history.plastic_coordinate*a_gamma)
                                  +self._b @ np.linalg.solve(self._d, moment))+a_kappa @ moment)
        metric = float(a_gamma @ self._schur @ a_gamma)
        new, increment = self._advance(drive, metric, history)
        curvature = new.plastic_coordinate*a_kappa+np.linalg.solve(
            self._d, moment-self._b.T @ (gamma-new.plastic_coordinate*a_gamma))
        response = self._response(np.r_[gamma, curvature], history, new, increment)
        c = response.tangent
        compliance = np.linalg.solve(c[3:, 3:], np.eye(3))
        cross = c[:3, 3:] @ compliance
        hessian = np.block([[c[:3, :3]-cross @ c[3:, :3], cross], [cross.T, -compliance]])
        gradient = np.r_[response.resultants[:3], -curvature]
        return MixedSectionResponse(response, response.incremental_potential-float(moment @ curvature),
                                    _array(gradient, (6,)), _array(hessian, (6, 6)))
