"""Additive elastic 42-variable centered beam, analytic variations only."""
from time import monotonic
import numpy as np
from ._ge_beam3_g1_elastic import ElasticSection, ElasticCell, owned, sha, solve
from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_reference_identity import CapturedReferenceIdentity
from ._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, transpose, so3_exp, so3_log
from ._ge_beam3_p5.compensated import compensated_strain
from ._ge_beam3_p5.algebra import HALVES, rotation
from ._ge_beam3_p5.arrays import _frames
from ._ge_beam3_p5.compensated_coordinates import validate_pair
from ._ge_beam3_fibre_line_work import evaluate as line_work
from ._ge_beam3_generalized_static_boundary import spatial_jacobian

POLICY = "CANDIDATE_GE_BEAM3_G1_ELASTIC_RETAINED_V1"


class ElasticOperator:
    def __init__(self, reference, section, *, order=4):
        if type(reference) is not Reference or type(section) is not ElasticSection:
            raise ValueError("exact reference and elastic section required")
        if type(order) is not int or order not in (4, 8):
            raise ValueError("registered quadrature required")
        self.reference, self.section, self.order = reference, section, order
        self._ref = CapturedReferenceIdentity(reference)
        self.reference_identity = self._ref.require(reference)
        pts, weights = np.polynomial.legendre.leggauss(order)
        rows, stations = [], []
        for c in (0, 1):
            for j, (point, w) in enumerate(zip(pts, weights)):
                t = (point+1)/2; xi = c-1+t
                J = reference.jacobian(xi); R = reference.frame(xi)
                rows.append((c, t, w*J/2, R.T/J))
                stations.append((c, j, float(xi), float(w*J/2), owned(R)))
        self.cell = ElasticCell(section, rows)
        self.stations = tuple(stations)
        self.identity = self._identity()

    def _identity(self):
        return sha(dict(policy=POLICY, reference=self.reference_identity,
                        section=self.section.identity, order=self.order,
                        cell=self.cell._capture, stations=self.stations))

    def guard(self):
        self.cell.guard()
        if (self._ref.require(self.reference) != self.reference_identity
                or self.cell.section is not self.section or self._identity() != self.identity):
            raise ValueError("elastic operator authority changed")

    def evaluate(self, positions, low, nodal_frames, cell_rotations, resultants,
                 *, line=(0., 0., 0.), couple=(0., 0., 0.)):
        self.guard()
        x, low = owned(positions, (3, 3)), owned(low, (3, 3))
        for h, l in zip(x.flat, low.flat):
            validate_pair(float(h), float(l))
        q, u = _frames(nodal_frames, 3, "nodal frames"), _frames(cell_rotations, 2, "cell rotations")
        p = owned(resultants, (18,)); load = owned(line, (3,)); density = owned(couple, (3,))
        variables = [Jet2.variable(0., i, 24) for i in range(24)]
        made_q = [matmul(so3_exp(variables[6*n+3:6*n+6]), constant_matrix(q[n], 24)) for n in range(3)]
        z, ell = [], []
        for c, (left, right) in enumerate(HALVES):
            made_u = matmul(so3_exp(variables[18+3*c:21+3*c]), constant_matrix(u[c], 24))
            z.extend(compensated_strain(made_u, self.reference.coordinates, x, low, variables, left, right))
            for node, sign in ((left, -1), (right, 1)):
                frame = matmul(made_u, constant_matrix(self.reference.nodal_triads[node], 24))
                ell.extend(sign*v for v in so3_log(matmul(transpose(frame), made_q[node])))
        terms = z+ell
        k = np.array([v.value for v in terms]); J = np.array([v.gradient for v in terms])
        dual, gradient, compliance = self.cell.response(p)
        r = np.r_[J.T @ p, k-gradient]
        geometric = sum((v.hessian*s for v, s in zip(terms, p)), np.zeros((24, 24)))
        H = np.block([[geometric, J.T], [J, -compliance]])
        potential = float(p @ k-dual)
        if np.any(load):
            work = line_work(self.reference, x, low, u, load, order=self.order)
            r -= work.gradient; H -= work.hessian; potential -= work.value
        jac = spatial_jacobian(r, H)
        applied = np.zeros(42)
        for c in (0, 1):
            applied[18+3*c:21+3*c] = sum(row[3] for row in self.stations if row[0] == c)*density
        return dict(residual=owned(r-applied), hessian=owned(H), jacobian=owned(jac),
                    potential=potential, conservative=not bool(np.any(density)))

    def recover(self, rotations, resultants):
        u = _frames(rotations, 2, "recovery rotations")
        return tuple({**row, "cell": c, "station": j, "xi": xi, "measure": w,
                      "reference_frame": R, "current_frame": owned(u[c] @ R),
                      "formulation_id": POLICY}
                     for row, (c, j, xi, w, R) in zip(self.cell.recover(resultants), self.stations))


def schur(residual, jacobian):
    r, H = owned(residual, (42,)), owned(jacobian, (42, 42))
    solved = solve(H[18:, 18:], np.c_[r[18:], H[18:, :18]], assume_a="gen")
    return (owned(r[:18]-H[:18, 18:] @ solved[:, 0]),
            owned(H[:18, :18]-H[:18, 18:] @ solved[:, 1:]),
            owned(-solved[:, 1:]), owned(-solved[:, 0]))


def local_solve(operator, x, low, frames, seed_u, seed_p, *, line, couple, check=lambda: None):
    if type(operator) is not ElasticOperator:
        raise ValueError("exact elastic operator required")
    started = monotonic(); u = _frames(seed_u, 2, "seed rotations").copy(); p = owned(seed_p, (18,)).copy()
    length = float(np.linalg.norm(operator.reference.coordinates[-1]-operator.reference.coordinates[0]))
    if length <= 0:
        raise ValueError("positive characteristic length required")
    scale = np.r_[np.full(12, length), np.ones(12)]
    def evaluate(a, b):
        check()
        if monotonic()-started > 60:
            raise TimeoutError("elastic internal solve deadline")
        value = operator.evaluate(x, low, frames, a, b, line=line, couple=couple)
        error = float(np.linalg.norm(value["residual"][18:]/scale))
        return value, error
    for iteration in range(25):
        value, error = evaluate(u, p)
        if error <= 1e-11:
            r, K, lift, correction = schur(value["residual"], value["jacobian"])
            return dict(rotations=owned(u), resultants=owned(p), residual=r, tangent=K,
                        lift=lift, correction=correction, full=value, internal_error=error,
                        history=(), iterations=iteration)
        if iteration == 24:
            break
        step = solve(value["jacobian"][18:, 18:], -value["residual"][18:], assume_a="gen")
        angle = max(np.linalg.norm(step[:3]), np.linalg.norm(step[3:6]))
        fraction = min(1., .45*np.pi/max(angle, np.finfo(float).tiny))
        for cut in range(9):
            delta = step*fraction*.5**cut
            trial_u = np.array([rotation(delta[3*c:3*c+3]) @ u[c] for c in (0, 1)])
            trial_p = p+delta[6:]
            _, metric = evaluate(trial_u, trial_p)
            if metric < error or metric <= 1e-11:
                u, p = trial_u, trial_p
                break
        else:
            raise ValueError("elastic local line search failed")
    raise ValueError("elastic local equilibrium iteration limit")
