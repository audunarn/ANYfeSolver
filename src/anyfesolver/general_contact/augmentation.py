"""Optional acceleration of a projected augmented-contact fixed point.

The secant predictor never accepts an increment. The caller must re-equilibrate
with the proposed multiplier and check the original projected residual and
independent physical gap before committing any state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActiveSetPivot:
    """One bounded linearized complementarity proposal, never an acceptance."""

    multipliers: np.ndarray
    retained_local_indices: tuple[int, ...]
    schur_condition: float


def propose_active_set_pivot(
        multipliers: np.ndarray, normalized_gap: np.ndarray,
        active_indices: np.ndarray, gap_schur: np.ndarray, *,
        maximum_pressure: float, gap_consistency: float,
        ) -> ActiveSetPivot | None:
    """Enumerate nonnegative roots on subsets of one current active set.

    `gap_schur[i, j]` is the equilibrated gap derivative in m/Pa. A proposal
    must be re-equilibrated and checked against the actual projected law.
    """

    lam = np.asarray(multipliers, dtype=float)
    gap = np.asarray(normalized_gap, dtype=float)
    active = np.asarray(active_indices, dtype=np.intp)
    schur = np.asarray(gap_schur, dtype=float)
    count = len(active)
    if (lam.ndim != 1 or gap.shape != lam.shape or count < 2 or count > 4
            or active.ndim != 1 or len(np.unique(active)) != count
            or np.any(active < 0) or np.any(active >= len(lam))
            or schur.shape != (count, count)
            or not all(np.all(np.isfinite(array)) for array in (lam, gap, schur))
            or np.any(lam < 0.0)):
        raise ValueError("active-set pivot inputs are invalid")
    if (not np.isfinite(maximum_pressure) or maximum_pressure <= 0.0
            or not np.isfinite(gap_consistency) or gap_consistency <= 0.0):
        raise ValueError("active-set pivot bounds must be finite and positive")
    if np.any(lam[np.setdiff1d(np.arange(len(lam)), active)] > 0.0):
        return None
    try:
        condition = float(np.linalg.cond(schur))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(condition) or condition > 1.0e7:
        return None
    roots: list[tuple[np.ndarray, tuple[int, ...]]] = []
    for mask in range(1, (1 << count) - 1):
        keep = np.asarray([i for i in range(count) if mask & (1 << i)],
                          dtype=np.intp)
        drop = np.asarray([i for i in range(count) if not mask & (1 << i)],
                          dtype=np.intp)
        block = schur[np.ix_(keep, keep)]
        try:
            if np.linalg.cond(block) > 1.0e7:
                continue
            positive = (lam[active[keep]]
                        - np.linalg.solve(
                            block,
                            gap[active[keep]]
                            - schur[np.ix_(keep, drop)] @ lam[active[drop]],
                        ))
        except np.linalg.LinAlgError:
            continue
        candidate = np.zeros_like(lam)
        candidate[active[keep]] = positive
        if (not np.all(np.isfinite(candidate))
                or np.any(positive <= 0.0)
                or np.max(candidate) > maximum_pressure):
            continue
        predicted_gap = gap[active] + schur @ (candidate[active]-lam[active])
        if (np.any(predicted_gap[drop] < -gap_consistency)
                or np.max(np.abs(predicted_gap[keep])) > gap_consistency):
            continue
        roots.append((candidate, tuple(int(i) for i in keep)))
    if len(roots) != 1:
        return None
    candidate, retained = roots[0]
    return ActiveSetPivot(candidate, retained, condition)


class ProjectedAugmentationSecant:
    """Fit the active projected-map residual within one physical increment."""

    def __init__(self) -> None:
        self._active: np.ndarray | None = None
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.attempts = 0
        self.applied = 0

    def propose(self, current: np.ndarray, projected: np.ndarray) -> np.ndarray | None:
        current = np.asarray(current, dtype=float)
        projected = np.asarray(projected, dtype=float)
        if (current.ndim != 1 or current.shape != projected.shape
                or not np.all(np.isfinite(current))
                or not np.all(np.isfinite(projected))
                or np.any(current < 0.0) or np.any(projected < 0.0)):
            raise ValueError("projected augmentation samples must be finite nonnegative vectors")
        active = projected > 0.0
        count = int(np.count_nonzero(active))
        if (count < 1 or count > 4
                or np.any((current > 0.0) & ~active)):
            self._active = None
            self._samples.clear()
            return None
        if self._active is None or not np.array_equal(active, self._active):
            self._active = active.copy()
            self._samples.clear()
        self._samples.append((current[active].copy(),
                              (projected - current)[active].copy()))
        self._samples = self._samples[-(count + 2):]
        if len(self._samples) < count + 2:
            return None
        self.attempts += 1
        coordinates = np.vstack([sample[0] for sample in self._samples])
        residuals = np.vstack([sample[1] for sample in self._samples])
        origin = coordinates.mean(axis=0)
        centered = coordinates - origin
        values = residuals - residuals.mean(axis=0)
        try:
            slope, _, rank, singular = np.linalg.lstsq(centered, values, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if (rank != count or singular[-1] <= 0.0
                or singular[0] / singular[-1] > 1.0e6):
            return None
        fit_error = np.linalg.norm(values - centered @ slope)
        if fit_error > 1.0e-3 * np.linalg.norm(values):
            return None
        derivative = slope.T
        try:
            if np.linalg.cond(derivative) > 1.0e8:
                return None
            step = np.linalg.solve(derivative, -residuals[-1])
        except np.linalg.LinAlgError:
            return None
        candidate = current[active] + step
        amplitude = max(float(np.max(current[active])),
                        float(np.max(projected[active])))
        if (not np.all(np.isfinite(candidate)) or np.any(candidate <= 0.0)
                or np.max(np.abs(step)) > 2.0 * amplitude):
            return None
        proposal = np.zeros_like(current)
        proposal[active] = candidate
        self.applied += 1
        return proposal
