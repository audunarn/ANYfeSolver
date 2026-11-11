"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import (
    Jet2, constant_matrix, matmul, matvec, transpose, so3_exp, so3_log,
)
from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5.algebra import HALVES, rotation
from anyfesolver._ge_beam3_p5.arrays import _array, _frames, _readonly
from anyfesolver._ge_beam3_p5.section import (
    DirectedHardeningSection, SectionHistory, SectionResponse, _history,
)


def compose_density(inputs, density):
    """Analytic chain rule, including curvature of the geometric input map."""
    jacobian = np.array([item.gradient for item in inputs])
    curvature = sum((weight*item.hessian for weight, item in zip(density.gradient, inputs)),
                    np.zeros_like(inputs[0].hessian))
    return Jet2(density.potential, jacobian.T @ density.gradient,
                jacobian.T @ density.hessian @ jacobian+curvature)


@dataclass(frozen=True)
class BeamStationTrial:
    cell: int
    index: int
    reference_coordinate: float
    response: SectionResponse


@dataclass(frozen=True)
class MixedBeamEvaluation:
    potential: float
    residual: np.ndarray
    hessian: np.ndarray
    stations: tuple


@dataclass(frozen=True)
class NonlinearCondensedResponse:
    potential: float
    residual: np.ndarray
    tangent: np.ndarray
    local_rotations: np.ndarray
    moments: np.ndarray
    stations: tuple
    local_residual_norm: float
    iterations: int
    evaluations: int


class NonlinearLocalError(RuntimeError):
    """No stable, admissible stationary mixed state within the fixed budget."""


@dataclass(frozen=True)
class LocalForceAccuracy:
    """Opt-in research accuracy budget; not an assembly qualification bound.

Use the assembly's physical moment-to-force length and force normalization.
The limit bounds a first-order estimate, NOT a rigorous residual error.
No global residual or recovered resultant is replaced by that estimate.
"""

    limit: float
    rotation_length: float
    force_scale: float = 1.

    def __post_init__(self):
        values = (self.limit, self.rotation_length, self.force_scale)
        if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0
               for v in values):
            raise ValueError('finite positive local force accuracy values required')
        if self.limit > 1e-11 or self.force_scale < 1:
            raise ValueError('local force limit must not relax equilibrium normalization')


def _accuracy_metrics(evaluation, accuracy):
    """Internal Newton correction and estimated physical external-force error."""
    if type(accuracy) is not LocalForceAccuracy:
        raise ValueError('explicit LocalForceAccuracy required')
    try:
        step = np.linalg.solve(evaluation.hessian[18:, 18:], -evaluation.residual[18:])
    except np.linalg.LinAlgError as exc:
        raise NonlinearLocalError('singular local accuracy estimate') from exc
    effect = (evaluation.hessian[:18, 18:] @ step).reshape(3, 6)
    effect[:, 3:] /= accuracy.rotation_length
    error = float(np.linalg.norm(effect)/accuracy.force_scale)
    norm = float(np.linalg.norm(evaluation.residual[18:], np.inf))
    merit = max(norm/1e-11, error/accuracy.limit)
    if not np.isfinite(step).all() or not np.isfinite(error) or not np.isfinite(merit):
        raise NonlinearLocalError('nonfinite local accuracy estimate')
    return step, error, merit


class StationaryMixedBeam:
    """18 external + six cell rotations + twelve material endpoint moments.

    The 36-coordinate jet uses spatial multiplicative nodal/cell increments
    and additive moment increments. Condensation is performed only after
    local stationarity, with an explicit negative moment block and positive
    cell-rotation Schur block. This is not a high-contrast retained chart.
    """

    def __init__(self, reference, section, *, order=24, origins=None, line_force=None):
        if type(section) is not DirectedHardeningSection:
            raise ValueError("directed-hardening research section required")
        if not isinstance(order, int) or isinstance(order, bool) or order not in (4, 8, 24):
            raise ValueError("diagnostic quadrature order must be 4, 8 or 24")
        self.reference = CurvedBeam3ReferenceGeometry(reference.coordinates, reference.nodal_triads)
        self.section, self.order = section, order
        if origins is None:
            origins = [SectionHistory() for _ in range(2*order)]
        origins = tuple(origins)
        if len(origins) != 2*order:
            raise ValueError("one fixed origin per quadrature station required")
        self.origins = tuple(_history(value) for value in origins)
        self.line_force = _readonly(np.zeros(3) if line_force is None else _array(line_force, (3,), "line force"))
        points, weights = np.polynomial.legendre.leggauss(order)
        self._stations = []
        for cell, (left, right) in enumerate(HALVES):
            values = []
            for index, (point, weight) in enumerate(zip(points, weights)):
                t = (point+1)/2
                xi = cell-1+t
                jacobian = self.reference.jacobian(xi)
                reference_position = self.reference.position(xi)
                offset = reference_position-((1-t)*self.reference.coordinates[left]+t*self.reference.coordinates[right])
                values.append((index, t, xi, weight*jacobian/2,
                               self.reference.frame(xi).T/jacobian, offset, reference_position))
            self._stations.append(tuple(values))

    def _chord_strain(self, made_u, chord, left, right):
        """Historical direct evaluation; successors may rearrange this identity."""
        local = matvec(transpose(made_u), chord)
        return [local[i]-(self.reference.coordinates[right, i]-self.reference.coordinates[left, i]) for i in range(3)]

    def evaluate(self, positions, vertex_frames, cell_rotations, moments, *, increment=None):
        x = _array(positions, (3, 3), "positions")
        q = _frames(vertex_frames, 3, "vertex frames")
        u = _frames(cell_rotations, 2, "cell rotations")
        m = _array(moments, (2, 2, 3), "endpoint moments")
        delta = np.zeros(36) if increment is None else _array(increment, (36,), "mixed increment")
        angular = [delta[6*n+3:6*n+6] for n in range(3)]+[delta[18+3*c:21+3*c] for c in (0, 1)]
        if max(np.linalg.norm(value) for value in angular) >= .9*np.pi:
            raise ValueError("rotation increment requires cutback before endpoint evaluation")
        variables = [Jet2.variable(value, i, 36) for i, value in enumerate(delta)]
        made_x = [[Jet2.constant(x[n, i], 36)+variables[6*n+i] for i in range(3)] for n in range(3)]
        made_q = [matmul(so3_exp(variables[6*n+3:6*n+6]), constant_matrix(q[n], 36)) for n in range(3)]
        total = Jet2.constant(0., 36)
        station_trials = []
        for cell, (left, right) in enumerate(HALVES):
            made_u = matmul(so3_exp(variables[18+3*cell:21+3*cell]), constant_matrix(u[cell], 36))
            chord = [made_x[right][i]-made_x[left][i] for i in range(3)]
            z = self._chord_strain(made_u, chord, left, right)
            endpoints = [[Jet2.constant(m[cell, n, i], 36)+variables[24+6*cell+3*n+i]
                          for i in range(3)] for n in (0, 1)]
            for endpoint, node, sign in ((0, left, -1), (1, right, 1)):
                frame = matmul(made_u, constant_matrix(self.reference.nodal_triads[node], 36))
                ell = so3_log(matmul(transpose(frame), made_q[node]))
                total += sum((sign*ell[i]*endpoints[endpoint][i] for i in range(3)), Jet2.constant(0., 36))
            for index, t, xi, measure, v, offset, reference_position in self._stations[cell]:
                gamma = matvec(constant_matrix(v, 36), z)
                moment = [(1-t)*endpoints[0][i]+t*endpoints[1][i] for i in range(3)]
                density = self.section.mixed_response([value.value for value in gamma],
                    [value.value for value in moment], self.origins[cell*self.order+index])
                total += measure*compose_density(gamma+moment, density)
                if np.any(self.line_force):
                    lift = matvec(made_u, [Jet2.constant(value, 36) for value in offset])
                    current = [(1-t)*made_x[left][i]+t*made_x[right][i]+lift[i] for i in range(3)]
                    total -= measure*sum((self.line_force[i]*(current[i]-reference_position[i])
                                          for i in range(3)), Jet2.constant(0., 36))
                station_trials.append(BeamStationTrial(cell, index, xi, density.section))
        if not (np.isfinite(total.value) and np.isfinite(total.gradient).all() and np.isfinite(total.hessian).all()):
            raise NonlinearLocalError("nonfinite mixed functional")
        return MixedBeamEvaluation(total.value, _readonly(total.gradient), _readonly(total.hessian), tuple(station_trials))

    def solve(self, positions, vertex_frames, *, initial_rotations=None, initial_moments=None,
              max_iterations=25, max_evaluations=64, force_accuracy=None):
        if force_accuracy is not None and type(force_accuracy) is not LocalForceAccuracy:
            raise ValueError('explicit LocalForceAccuracy required')
        for value, limit in ((max_iterations, 25), (max_evaluations, 64)):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= limit:
                raise ValueError("local iteration/evaluation limit outside research bounds")
        x = _array(positions, (3, 3), "positions")
        q = _frames(vertex_frames, 3, "vertex frames")
        # Midpoint rotations are only initial guesses; no trial history update.
        from anyfesolver._ge_beam3_p5.algebra import log_rotation
        relatives = q @ self.reference.nodal_triads.transpose(0, 2, 1)
        default = np.array([relatives[left] @ rotation(log_rotation(relatives[left].T @ relatives[right])/2)
                            for left, right in HALVES])
        local = default if initial_rotations is None else _frames(initial_rotations, 2, "initial rotations")
        moments = np.zeros((2, 2, 3)) if initial_moments is None else _array(initial_moments, (2, 2, 3), "initial moments")
        evaluations = 0

        def evaluate(rotations, endpoint_moments):
            nonlocal evaluations
            if evaluations == max_evaluations:
                raise NonlinearLocalError("local evaluation budget exhausted")
            evaluations += 1
            return self.evaluate(x, q, rotations, endpoint_moments)

        current = evaluate(local, moments)
        for iteration in range(max_iterations+1):
            norm = float(np.linalg.norm(current.residual[18:], np.inf))
            accuracy_step, merit = None, norm
            if force_accuracy is not None:
                accuracy_step, _, merit = _accuracy_metrics(current, force_accuracy)
            if norm <= 1e-11 and (force_accuracy is None or merit <= 1):
                h = current.hessian
                try:
                    np.linalg.cholesky(-h[24:, 24:])
                    rotation_schur = h[18:24, 18:24]-h[18:24, 24:] @ np.linalg.solve(h[24:, 24:], h[24:, 18:24])
                    np.linalg.cholesky(rotation_schur)
                    tangent = h[:18, :18]-h[:18, 18:] @ np.linalg.solve(h[18:, 18:], h[18:, :18])
                except np.linalg.LinAlgError as exc:
                    raise NonlinearLocalError("unstable or singular local stationary blocks") from exc
                return NonlinearCondensedResponse(current.potential, _readonly(current.residual[:18]),
                    _readonly(tangent), _readonly(local), _readonly(moments), current.stations, norm, iteration, evaluations)
            if iteration == max_iterations:
                raise NonlinearLocalError("local iteration budget exhausted")
            try:
                step = (np.linalg.solve(current.hessian[18:, 18:], -current.residual[18:])
                        if force_accuracy is None else accuracy_step)
            except np.linalg.LinAlgError as exc:
                raise NonlinearLocalError("singular local Newton system") from exc
            if not np.isfinite(step).all():
                raise NonlinearLocalError("nonfinite local Newton step")
            accepted = False
            for backtrack in range(12):
                scaled = step*(.5**backtrack)
                if max(np.linalg.norm(scaled[3*c:3*c+3]) for c in (0, 1)) >= .9*np.pi:
                    continue
                made_u = np.array([rotation(scaled[3*c:3*c+3]) @ local[c] for c in (0, 1)])
                made_m = moments+scaled[6:].reshape(2, 2, 3)
                try:
                    trial = evaluate(made_u, made_m)
                    trial_merit = (np.linalg.norm(trial.residual[18:], np.inf)
                                   if force_accuracy is None else _accuracy_metrics(trial, force_accuracy)[2])
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if trial_merit < merit:
                    local, moments, current = made_u, made_m, trial
                    accepted = True
                    break
            if not accepted:
                raise NonlinearLocalError("local line search exhausted; explicit cutback required")
        raise AssertionError("unreachable local iteration")
