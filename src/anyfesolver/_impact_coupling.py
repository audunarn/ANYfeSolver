"""Small multisecant accelerator for the coupled impact iteration (not physics)."""
import numpy as np


class ImpactCoupledIteration:
    """Anderson type-II update of displacement fixed-point residuals.

    All returned coordinates must be re-evaluated by the physical contact law.
    History is local to one substep; no material state is stored here.
    """

    def __init__(self, depth=4):
        self.depth = depth
        self.x = []
        self.f = []
        self.signatures = []
        self.mode_origin = None
        self.mode_direction = None
        self.mode_bracket = {}
        self.globalization_count = 0

    def update(self, current, mapped, *, signature=None):
        x = np.asarray(current, dtype=float)
        f = np.asarray(mapped, dtype=float) - x
        # Contact is a piecewise map. Keep one bridge sample at a branch
        # crossing, but remove that bridge after two samples on the new branch
        # establish its own secant. A window mixing stale open/closed slopes
        # otherwise models neither branch and can crawl near first contact.
        if signature is not None and self.signatures and self._on_scalar_line(x, f):
            if signature != self.signatures[-1]:
                self.x = self.x[-1:]
                self.f = self.f[-1:]
                self.signatures = self.signatures[-1:]
            else:
                start = len(self.signatures) - 1
                while start > 0 and self.signatures[start - 1] == signature:
                    start -= 1
                self.x = self.x[start:]
                self.f = self.f[start:]
                self.signatures = self.signatures[start:]
        self.signatures.append(signature)
        self.signatures = self.signatures[-self.depth - 1:]
        self.x.append(x.copy())
        self.f.append(f.copy())
        self.x = self.x[-self.depth - 1:]
        self.f = self.f[-self.depth - 1:]
        if len(self.x) < 2:
            return self._globalize(x, f, np.asarray(mapped, dtype=float).copy())
        dx = np.column_stack([b-a for a,b in zip(self.x[:-1], self.x[1:])])
        df = np.column_stack([b-a for a,b in zip(self.f[:-1], self.f[1:])])
        try:
            weights = np.linalg.lstsq(df, f, rcond=1.e-10)[0]
        except np.linalg.LinAlgError:
            return self._globalize(x, f, np.asarray(mapped, dtype=float).copy())
        # Reject ill-conditioned extrapolations, never clip physical forces.
        if not np.all(np.isfinite(weights)) or np.linalg.norm(weights, 1) > 10.:
            self.x = self.x[-1:]
            self.f = self.f[-1:]
            self.signatures = self.signatures[-1:]
            return self._globalize(x, f, np.asarray(mapped, dtype=float).copy())
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = np.asarray(mapped, dtype=float) - (dx + df) @ weights
        if not np.all(np.isfinite(candidate)):
            self.x = self.x[-1:]
            self.f = self.f[-1:]
            self.signatures = self.signatures[-1:]
            return self._globalize(x, f, np.asarray(mapped, dtype=float).copy())
        return self._globalize(x, f, candidate)


    def _globalize(self, x, f, proposal):
        """Safeguard a detected scalar mode with its *raw* residual bracket.

        Opposite vector residuals alone do not bracket a vector root. Use this
        only while both coordinates and residuals lie on one numerical line;
        otherwise forget the bracket and retain the general Anderson proposal.
        The middle-half rule is a bisection safeguard, not force relaxation.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            norm = float(np.linalg.norm(f))
        if norm == 0.:
            return x.copy()
        if not np.isfinite(norm):
            self.mode_bracket = {}
            return proposal
        if self.mode_direction is None:
            self.mode_origin = x.copy()
            self.mode_direction = f / norm
        direction = self.mode_direction
        offset = x - self.mode_origin
        coordinate = float(np.dot(offset, direction))
        residual = float(np.dot(f, direction))
        if not self._on_scalar_line(x, f):
            self.mode_origin = x.copy()
            self.mode_direction = f / norm
            self.mode_bracket = {}
            return proposal
        sign = 1 if residual > 0. else -1
        self.mode_bracket[sign] = (coordinate, x.copy())
        if len(self.mode_bracket) != 2:
            return proposal
        endpoints = sorted(self.mode_bracket.values(), key=lambda item: item[0])
        low, high = endpoints[0][0], endpoints[1][0]
        width = high - low
        if not np.isfinite(width) or width <= 0.:
            self.mode_bracket = {}
            return proposal
        proposal_offset = proposal - self.mode_origin
        projected = float(np.dot(proposal_offset, direction))
        if (np.linalg.norm(proposal_offset - projected * direction)
                > self._coordinate_roundoff(proposal, proposal_offset)):
            self.mode_bracket = {}
            return proposal
        if not (low + .25 * width <= projected <= high - .25 * width):
            self.globalization_count += 1
            return .5 * endpoints[0][1] + .5 * endpoints[1][1]
        return proposal


    def _coordinate_roundoff(self, x, offset):
        eps = np.finfo(float).eps
        return (np.sqrt(eps) * float(np.linalg.norm(offset))
                + eps * (float(np.linalg.norm(x)) + float(np.linalg.norm(self.mode_origin))))

    def _on_scalar_line(self, x, f):
        if self.mode_direction is None:
            return False
        offset = x - self.mode_origin
        direction = self.mode_direction
        return (np.linalg.norm(f - np.dot(f, direction) * direction)
                <= np.sqrt(np.finfo(float).eps) * float(np.linalg.norm(f))
                and np.linalg.norm(offset - np.dot(offset, direction) * direction)
                <= self._coordinate_roundoff(x, offset))


def numerical_contact_jacobian(evaluate, coordinates, length_scale, safe_point=None):
    """Centered derivative of original contact loads wrt displacement coordinates.

    Velocity dependence is included by the caller's Newmark-consistent evaluate.
    Probes are force-only and must not mutate constitutive or contact history.
    The cube-root-epsilon step balances centered truncation and roundoff; it is
    not a contact-law smoothing, penalty or acceptance tolerance.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    columns = []
    for index in range(coordinates.size):
        if safe_point is not None:
            safe_point()
        # Truncation scale is physical and invariant to a world-origin shift.
        # Absolute coordinates enter only through their representable spacing.
        step = max(np.cbrt(np.finfo(float).eps) * float(length_scale),
                   4. * abs(float(np.spacing(coordinates[index]))))
        plus = coordinates.copy(); minus = coordinates.copy()
        plus[index] += step; minus[index] -= step
        denominator = float(plus[index] - minus[index])
        if not np.isfinite(denominator) or denominator <= 0.:
            raise ValueError("unrepresentable numerical contact perturbation")
        columns.append((evaluate(plus) - evaluate(minus)) / denominator)
    jacobian = np.column_stack(columns)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("nonfinite numerical contact Jacobian")
    return jacobian
