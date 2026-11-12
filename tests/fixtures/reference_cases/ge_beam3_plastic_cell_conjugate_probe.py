"""Bounded research conjugate of the integrated directed-hardening cell.

Retains the existing shared cell strain, quadrature and moment interpolation.
Plastic station increments solve a convex piecewise-quadratic problem.
No native history commit, public routing, tolerance relaxation or retry.
"""
from dataclasses import dataclass
from time import monotonic
import numpy as np
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
from anyfesolver._ge_beam3_p5.section import SectionHistory, _history
from anyfesolver._native_reference_modal import _owned


@dataclass(frozen=True)
class CellResponse:
    potential: float
    gradient: np.ndarray
    hessian: np.ndarray
    histories: tuple
    increments: np.ndarray
    kkt_residual: float
    iterations: int
    production_qualified: bool = False


class PlasticCellConjugate:
    def __init__(self, element, origins=None):
        core = element.core
        if np.any(core.line_force): raise ValueError('cell conjugate contains no external work')
        self.source = CenteredStationaryBeam(core.reference, core.section,
            position_low=np.zeros((3, 3)), order=core.order)
        count = 2*core.order
        self.origins = tuple(SectionHistory() for _ in range(count)) if origins is None else tuple(_history(v) for v in origins)
        if len(self.origins) != count: raise ValueError('one fixed origin per station required')
        self.weights = np.zeros(count)
        a = np.zeros((6, 6)); b = np.zeros((6, 12)); s = np.zeros((12, 12))
        dz = np.zeros((6, count)); dm = np.zeros((12, count))
        strain_factor = np.zeros((3*count, 6)); plastic_factor = np.zeros((3*count, count))
        ag = core.section._direction[:3]; ak = core.section._direction[3:]
        for cell in (0, 1):
            sl = slice(3*cell, 3*cell+3)
            for index, t, _, w, v, _, _ in self.source._stations[cell]:
                j = cell*core.order+index; self.weights[j] = w
                density = core.section.mixed_response(np.zeros(3), np.zeros(3), SectionHistory())
                h = density.hessian
                n = np.zeros((3, 12)); n[:, 6*cell:6*cell+3] = (1-t)*np.eye(3)
                n[:, 6*cell+3:6*cell+6] = t*np.eye(3)
                root = np.sqrt(w)*np.linalg.cholesky(h[:3, :3]).T
                strain_factor[3*j:3*j+3, sl] = root@v
                plastic_factor[3*j:3*j+3, j] = root@ag
                a[sl, sl] += w*v.T@h[:3, :3]@v
                b[sl] += w*v.T@h[:3, 3:]@n
                s -= w*n.T@h[3:, 3:]@n
                dz[sl, j] = w*v.T@h[:3, :3]@ag
                dm[:, j] = w*n.T@(h[3:, :3]@ag+ak)
        ai = np.linalg.solve(a, np.eye(6)); aib = np.linalg.solve(a, b)
        self.c0 = _owned(np.block([[ai, -aib], [-aib.T, s+b.T@aib]]))
        self.g = _owned(np.column_stack((dz.T@ai, dm.T-dz.T@aib)))
        # Gram representation avoids subtracting two large PSD matrices.
        orthogonal, _ = np.linalg.qr(strain_factor, mode='complete')
        remainder = orthogonal[:, 6:].T@plastic_factor
        self.k = _owned(remainder.T@remainder)
        self.q = _owned(self.k+np.diag(self.weights*core.section._hardening))
        np.linalg.cholesky(self.q)
        self.z0 = _owned([h.plastic_coordinate for h in self.origins])
        self.radius = _owned(self.weights*(core.section._yield+core.section._hardening*np.array([h.accumulated for h in self.origins])))

    def response(self, resultants, *, progress=None):
        started = monotonic(); p = _owned(resultants)
        if p.shape != (18,): raise ValueError('18 retained resultants required')
        drive = self.g@p-self.k@self.z0
        delta = np.zeros(len(self.origins)); working = np.zeros(len(delta), dtype=bool)
        signs = np.zeros(len(delta))
        for iteration in range(33):
            if monotonic()-started > 30: raise ValueError('cell conjugate deadline')
            derivative = self.q@delta-drive
            active = delta != 0
            error = np.maximum(np.abs(derivative)-self.radius, 0.)
            error[active] = np.abs(derivative[active]+self.radius[active]*np.sign(delta[active]))
            residual = float(np.max(error))
            if progress is not None:
                progress(dict(iteration=iteration, increments=delta.copy(), derivative=derivative.copy(),
                              kkt_residual=residual))
            if residual <= 1e-11: break
            if iteration == 32: raise ValueError('cell conjugate active-set limit')
            # Solve within the present orthant before releasing one violated
            # zero coordinate. Never jump through a sign boundary: stop at
            # the first zero and remove that bound from the working set.
            if not np.any(working) or np.max(np.abs(derivative[working]+self.radius[working]*signs[working])) <= 1e-11:
                violations = np.maximum(np.abs(derivative)-self.radius, 0.)
                violations[working] = 0.
                release = int(np.argmax(violations))
                if violations[release] <= 1e-11: raise ValueError('cell conjugate unresolved optimality')
                working[release] = True; signs[release] = -np.sign(derivative[release])
            target = np.zeros_like(delta)
            block = self.q[np.ix_(working, working)]; scale = np.sqrt(np.diag(block))
            target[working] = np.linalg.solve(block/scale[:, None]/scale[None, :],
                (drive[working]-self.radius[working]*signs[working])/scale)/scale
            crossing = np.flatnonzero(working & (signs*target <= 0.))
            if len(crossing):
                ratios = delta[crossing]/(delta[crossing]-target[crossing])
                hit = int(crossing[np.argmin(ratios)]); alpha = float(np.min(ratios))
                if not 0 <= alpha <= 1 or not np.isfinite(alpha): raise ValueError('invalid cell orthant step')
                delta += alpha*(target-delta); delta[hit] = 0.
                working[hit] = False; signs[hit] = 0.
            else:
                delta = target
        hessian = self.c0.copy()
        if np.any(active):
            hessian += self.g[active].T@np.linalg.solve(self.q[np.ix_(active, active)], self.g[active])
        z = self.z0+delta
        potential = .5*p@self.c0@p+self.z0@self.g@p-.5*self.z0@self.k@self.z0
        potential += delta@drive-.5*delta@self.q@delta-self.radius@np.abs(delta)
        histories = tuple(_history(SectionHistory(float(v), old.accumulated+abs(float(d))))
            for old, v, d in zip(self.origins, z, delta))
        if not np.isfinite(potential): raise ValueError('nonfinite cell conjugate potential')
        return CellResponse(float(potential), _owned(self.c0@p+self.g.T@z), _owned(hessian),
            histories, _owned(delta), residual, iteration)
