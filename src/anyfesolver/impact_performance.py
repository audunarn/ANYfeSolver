"""Conservative tangent-reuse policy for nonlinear impact iterations.

The controller deliberately owns no finite-element data.  It only decides
whether the effective tangent/factorization from the current time substep may
be reused and records why a refresh was required.  Keeping this policy out of
``contact.py`` makes the zero-budget full-Newton oracle explicit and lets the
invalidation rules be qualified independently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def _plastic_state_measure(states: Mapping[int, Any]) -> Tuple[float, float, int]:
    """Return inexpensive scalar measures of the current plastic state.

    Nonlinear shell and beam states expose accumulated plastic strain through
    ``alpha``.  The maximum, mean active value, and active-entry count detect
    both material activation/deactivation and meaningful hardening changes
    without hashing or copying all integration-point arrays.
    """

    maximum = 0.0
    active_sum = 0.0
    active_count = 0
    for state in states.values():
        if not isinstance(state, Mapping):
            continue
        alpha = np.asarray(state.get("alpha", ()), dtype=float).reshape(-1)
        if alpha.size == 0:
            continue
        finite = alpha[np.isfinite(alpha)]
        if finite.size == 0:
            continue
        finite = np.maximum(finite, 0.0)
        maximum = max(maximum, float(np.max(finite)))
        active = finite > 1.0e-14
        if np.any(active):
            active_values = finite[active]
            active_sum += float(np.sum(active_values))
            active_count += int(active_values.size)
    mean_active = active_sum / max(active_count, 1)
    return maximum, mean_active, active_count


def _damage_signature(scales: Mapping[int, float]) -> Tuple[Tuple[int, float], ...]:
    return tuple(sorted((int(element_id), float(scale)) for element_id, scale in scales.items()))


def _contact_signature(records: Sequence[Any]) -> Tuple[Tuple[int, str], ...]:
    return tuple(
        sorted(
            (
                int(record.element_id),
                str(getattr(record, "contact_classification", "unknown")),
            )
            for record in records
        )
    )


@dataclass
class ImpactTangentReuseController:
    """Bounded modified-Newton reuse with conservative invalidation.

    ``max_reuse_iterations`` is the number of solves allowed after a fresh
    factorization.  A value of zero is the legacy full-Newton oracle.
    """

    max_reuse_iterations: int
    residual_stall_ratio: float = 0.90
    plastic_relative_threshold: float = 5.0e-3
    cached_handle: Optional[Any] = field(default=None, init=False, repr=False)
    _reuse_since_refresh: int = field(default=0, init=False, repr=False)
    _pending_refresh_reasons: set[str] = field(default_factory=set, init=False, repr=False)
    _last_dt: Optional[float] = field(default=None, init=False, repr=False)
    _last_damage_signature: Optional[Tuple[Tuple[int, float], ...]] = field(default=None, init=False, repr=False)
    _last_deleted_signature: Optional[Tuple[int, ...]] = field(default=None, init=False, repr=False)
    _last_contact_signature: Optional[Tuple[Tuple[int, str], ...]] = field(default=None, init=False, repr=False)
    _last_residual_norm: Optional[float] = field(default=None, init=False, repr=False)
    _last_plastic_measure: Optional[Tuple[float, float, int]] = field(default=None, init=False, repr=False)
    tangent_assembly_count: int = field(default=0, init=False)
    tangent_reuse_count: int = field(default=0, init=False)
    factorization_count: int = field(default=0, init=False)
    factorization_reuse_count: int = field(default=0, init=False)
    active_contact_set_changes: int = field(default=0, init=False)
    contact_classification_changes: int = field(default=0, init=False)
    plastic_state_change_max: float = field(default=0.0, init=False)
    refresh_reason_counts: Counter[str] = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        if int(self.max_reuse_iterations) < 0:
            raise ValueError("max_reuse_iterations must be non-negative")
        self.max_reuse_iterations = int(self.max_reuse_iterations)

    @property
    def enabled(self) -> bool:
        return self.max_reuse_iterations > 0

    def set_initial_contact(self, records: Sequence[Any]) -> None:
        """Seed contact tracking without reporting an artificial change."""

        if not self.enabled:
            return
        self._last_contact_signature = _contact_signature(records)

    def begin_substep(
        self,
        dt: float,
        damage_scales: Mapping[int, float],
        deleted_element_ids: Iterable[int],
    ) -> None:
        """Start a new substep and invalidate any prior effective tangent."""

        if not self.enabled:
            self.cached_handle = None
            self._reuse_since_refresh = 0
            return
        current_dt = float(dt)
        current_damage = _damage_signature(damage_scales)
        current_deleted = tuple(sorted(int(element_id) for element_id in deleted_element_ids))
        reasons = {"first_iteration"}
        if self._last_dt is not None and not np.isclose(
            current_dt,
            self._last_dt,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            reasons.add("time_step_change")
        if self._last_damage_signature is not None and current_damage != self._last_damage_signature:
            reasons.add("damage_scale_change")
        if self._last_deleted_signature is not None and current_deleted != self._last_deleted_signature:
            reasons.add("deletion_change")
        self._last_dt = current_dt
        self._last_damage_signature = current_damage
        self._last_deleted_signature = current_deleted
        self.cached_handle = None
        self._reuse_since_refresh = 0
        self._pending_refresh_reasons.update(reasons)
        self._last_residual_norm = None
        self._last_plastic_measure = None

    def request_refresh(self, reason: str) -> None:
        if self.enabled:
            self._pending_refresh_reasons.add(str(reason))

    def refresh_decision(self) -> Tuple[bool, Tuple[str, ...]]:
        """Return whether this iteration must assemble/factorize a tangent."""

        if not self.enabled:
            return True, ("legacy_full_newton",)
        reasons = set(self._pending_refresh_reasons)
        if self.cached_handle is None:
            reasons.add("missing_cached_factorization")
        if self._reuse_since_refresh >= self.max_reuse_iterations:
            reasons.add("reuse_budget_exhausted")
        return bool(reasons), tuple(sorted(reasons))

    def assess_trial_state(self, residual_norm: float, states: Mapping[int, Any]) -> Tuple[str, ...]:
        """Detect convergence stalls and material-state changes at the trial."""

        if not self.enabled:
            return tuple()
        reasons: set[str] = set()
        residual_value = float(residual_norm)
        previous_residual = self._last_residual_norm
        if (
            self.enabled
            and previous_residual is not None
            and np.isfinite(previous_residual)
            and np.isfinite(residual_value)
            and residual_value >= self.residual_stall_ratio * max(previous_residual, 1.0e-30)
        ):
            reasons.add("residual_stall")
        self._last_residual_norm = residual_value

        measure = _plastic_state_measure(states)
        previous_measure = self._last_plastic_measure
        if self.enabled and previous_measure is not None:
            maximum, mean_active, active_count = measure
            previous_maximum, previous_mean, previous_count = previous_measure
            change = max(abs(maximum - previous_maximum), abs(mean_active - previous_mean))
            self.plastic_state_change_max = max(self.plastic_state_change_max, float(change))
            threshold = max(
                1.0e-10,
                self.plastic_relative_threshold
                * max(maximum, previous_maximum, mean_active, previous_mean, 1.0e-6),
            )
            if active_count != previous_count:
                reasons.add("plastic_active_set_change")
            elif change > threshold:
                reasons.add("plastic_state_change")
        self._last_plastic_measure = measure
        return tuple(sorted(reasons))

    def observe_contact(self, records: Sequence[Any]) -> Tuple[str, ...]:
        """Track active element and contact-classification changes."""

        if not self.enabled:
            return tuple()
        return self.observe_contact_signature(_contact_signature(records))

    def observe_contact_signature(
        self,
        signature: Sequence[Tuple[int, str]],
    ) -> Tuple[str, ...]:
        """Track contact changes without materializing public records."""

        if not self.enabled:
            return tuple()
        signature = tuple(
            sorted(
                (int(element_id), str(classification))
                for element_id, classification in signature
            )
        )
        previous = self._last_contact_signature
        self._last_contact_signature = signature
        if previous is None:
            return tuple()
        reasons: set[str] = set()
        previous_ids = tuple(item[0] for item in previous)
        current_ids = tuple(item[0] for item in signature)
        if current_ids != previous_ids:
            self.active_contact_set_changes += 1
            reasons.add("active_contact_set_change")
        elif signature != previous:
            self.contact_classification_changes += 1
            reasons.add("contact_classification_change")
        for reason in reasons:
            self.request_refresh(reason)
        return tuple(sorted(reasons))

    def observe_line_search(self, factor: float) -> None:
        if float(factor) <= 0.5:
            self.request_refresh("aggressive_line_search")

    def record_tangent_assembly(self, reasons: Sequence[str]) -> None:
        self.tangent_assembly_count += 1
        for reason in reasons:
            self.refresh_reason_counts[str(reason)] += 1

    def record_factorization(self, handle: Any) -> None:
        self.cached_handle = handle
        self.factorization_count += 1
        self._reuse_since_refresh = 0
        self._pending_refresh_reasons.clear()

    def record_reuse(self) -> None:
        self.tangent_reuse_count += 1
        self.factorization_reuse_count += 1
        self._reuse_since_refresh += 1

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "max_reuse_iterations": int(self.max_reuse_iterations),
            "tangent_assembly_count": int(self.tangent_assembly_count),
            "tangent_reuse_count": int(self.tangent_reuse_count),
            "factorization_count": int(self.factorization_count),
            "factorization_reuse_count": int(self.factorization_reuse_count),
            "refresh_reason_counts": dict(sorted(self.refresh_reason_counts.items())),
            "active_contact_set_changes": int(self.active_contact_set_changes),
            "contact_classification_changes": int(self.contact_classification_changes),
            "residual_stall_ratio": float(self.residual_stall_ratio),
            "plastic_relative_threshold": float(self.plastic_relative_threshold),
            "plastic_state_change_max": float(self.plastic_state_change_max),
        }
