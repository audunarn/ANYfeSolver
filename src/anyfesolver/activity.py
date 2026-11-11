"""Vectorized element softening, deletion, and restart state.

The manager in this module owns element *state*, not assembled matrices.  It
therefore works with cached assembly plans and fixed CSR patterns: callers scale
existing element blocks or contribution buffers and leave row/column storage
unchanged.  A zero scale is a numerical deletion; it is never an instruction to
remove a sparse entry.

Element identifiers are stable external identifiers.  Array positions are an
internal implementation detail and are recovered through a vectorized sorted
lookup.  Damage is irreversible by default, while an explicit policy can permit
healing for staged construction or qualification workflows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np


_RESTART_SCHEMA = "anysolver.element_activity"
_RESTART_VERSION = 1
_QUANTITIES = ("stiffness", "mass", "damping", "load", "contact")


class ElementActivityError(ValueError):
    """Base class for element-activity contract errors."""


class RestartStateError(ElementActivityError):
    """Raised when serialized activity state is malformed or incompatible."""


class CouplingResolutionError(ElementActivityError):
    """Raised when a strict coupling policy encounters a deleted owner."""


class ContributionPolicy(str, Enum):
    """How one element-owned contribution follows activity.

    ``ACTIVITY`` applies the floating activity value. ``DELETE_ONLY`` keeps the
    full contribution while the element is merely softened and sets it to zero
    after hard deletion. ``RETAIN`` deliberately ignores deletion.
    """

    ACTIVITY = "activity"
    SCALE_WITH_ACTIVITY = "activity"
    DELETE_ONLY = "delete_only"
    ACTIVE_ONLY = "delete_only"
    REMOVE_ON_DELETE = "delete_only"
    RETAIN = "retain"


class CouplingPolicy(str, Enum):
    """Policy for couplings whose current owner has been hard-deleted."""

    DEACTIVATE = "deactivate"
    REASSIGN = "reassign"
    REASSIGN_OR_DEACTIVATE = "reassign"
    KEEP = "keep"
    ERROR = "error"


def _enum_value(enum_type: type[Enum], value: Any, name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ElementActivityError(f"{name} must be one of {choices}") from exc


@dataclass(frozen=True, slots=True)
class ElementActivityPolicy:
    """Explicit contribution and lifecycle policies for element activity."""

    stiffness: ContributionPolicy = ContributionPolicy.ACTIVITY
    mass: ContributionPolicy = ContributionPolicy.DELETE_ONLY
    damping: ContributionPolicy = ContributionPolicy.ACTIVITY
    load: ContributionPolicy = ContributionPolicy.DELETE_ONLY
    contact: ContributionPolicy = ContributionPolicy.DELETE_ONLY
    coupling: CouplingPolicy = CouplingPolicy.DEACTIVATE
    hard_delete_threshold: float = 0.0
    minimum_stiffness_scale: float = 0.0
    allow_healing: bool = False
    conditioning_warning_ratio: float = 1.0e8

    def __post_init__(self) -> None:
        for name in _QUANTITIES:
            object.__setattr__(
                self,
                name,
                _enum_value(ContributionPolicy, getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "coupling",
            _enum_value(CouplingPolicy, self.coupling, "coupling"),
        )
        threshold = float(self.hard_delete_threshold)
        minimum_scale = float(self.minimum_stiffness_scale)
        warning_ratio = float(self.conditioning_warning_ratio)
        if not np.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ElementActivityError("hard_delete_threshold must be finite and in [0, 1)")
        if not np.isfinite(minimum_scale) or not 0.0 <= minimum_scale <= 1.0:
            raise ElementActivityError("minimum_stiffness_scale must be finite and in [0, 1]")
        if not np.isfinite(warning_ratio) or warning_ratio < 1.0:
            raise ElementActivityError(
                "conditioning_warning_ratio must be finite and at least one"
            )
        object.__setattr__(self, "hard_delete_threshold", threshold)
        object.__setattr__(self, "minimum_stiffness_scale", minimum_scale)
        object.__setattr__(self, "allow_healing", bool(self.allow_healing))
        object.__setattr__(self, "conditioning_warning_ratio", warning_ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stiffness": self.stiffness.value,
            "mass": self.mass.value,
            "damping": self.damping.value,
            "load": self.load.value,
            "contact": self.contact.value,
            "coupling": self.coupling.value,
            "hard_delete_threshold": float(self.hard_delete_threshold),
            "minimum_stiffness_scale": float(self.minimum_stiffness_scale),
            "allow_healing": bool(self.allow_healing),
            "conditioning_warning_ratio": float(self.conditioning_warning_ratio),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ElementActivityPolicy":
        if not isinstance(payload, Mapping):
            raise RestartStateError("activity policy must be a mapping")
        allowed = {
            "stiffness",
            "mass",
            "damping",
            "load",
            "contact",
            "coupling",
            "hard_delete_threshold",
            "minimum_stiffness_scale",
            "allow_healing",
            "conditioning_warning_ratio",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise RestartStateError(f"unknown activity policy fields: {names}")
        try:
            return cls(**dict(payload))
        except ElementActivityError as exc:
            raise RestartStateError(str(exc)) from exc


# Short aliases keep the public vocabulary convenient without creating a second
# policy implementation.
ActivityPolicy = ElementActivityPolicy
ElementActivityPolicies = ElementActivityPolicy


def _readonly_array(values: Any, dtype: Any) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _element_id_vector(
    values: Any,
    *,
    name: str = "element_ids",
    allow_scalar: bool = True,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 0 and allow_scalar:
        raw = raw.reshape(1)
    if raw.ndim != 1:
        raise ElementActivityError(f"{name} must be a one-dimensional integer array")
    if raw.dtype.kind not in "iu":
        raise ElementActivityError(f"{name} must contain integers")
    try:
        result = np.asarray(raw, dtype=np.int64)
    except (OverflowError, ValueError) as exc:
        raise ElementActivityError(f"{name} values must fit in signed 64-bit integers") from exc
    if np.any(result < 0):
        raise ElementActivityError(f"{name} must contain non-negative stable identifiers")
    return np.array(result, dtype=np.int64, copy=True, order="C")


def _activity_vector(values: Any, count: int, name: str = "activity") -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ElementActivityError(f"{name} must contain floating values") from exc
    if raw.ndim == 0:
        result = np.full(count, float(raw), dtype=float)
    elif raw.shape == (count,):
        result = np.array(raw, dtype=float, copy=True, order="C")
    else:
        raise ElementActivityError(f"{name} must be scalar or have shape ({count},)")
    if np.any(~np.isfinite(result)):
        raise ElementActivityError(f"{name} must be finite")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ElementActivityError(f"{name} must be in [0, 1]")
    return result


def _boolean_vector(values: Any, count: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 0:
        return np.full(count, bool(raw), dtype=bool)
    if raw.shape != (count,):
        raise ElementActivityError(f"{name} must be scalar or have shape ({count},)")
    if raw.dtype.kind != "b":
        raise ElementActivityError(f"{name} must contain booleans")
    return np.array(raw, dtype=bool, copy=True, order="C")


def _normalized_axis(axis: int, ndim: int) -> int:
    if ndim == 0:
        raise ElementActivityError("element data must have at least one dimension")
    result = int(axis)
    if result < 0:
        result += ndim
    if not 0 <= result < ndim:
        raise ElementActivityError(f"axis {axis} is out of range for an array with {ndim} dimensions")
    return result


@dataclass(frozen=True, slots=True)
class DamageHistoryEntry:
    """One owned, JSON-safe change in irreversible damage history."""

    sequence: int
    step: int | None
    time: float | None
    reason: str
    element_ids: tuple[int, ...]
    previous_activity: tuple[float, ...]
    activity: tuple[float, ...]
    newly_hard_deleted_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if int(self.sequence) <= 0:
            raise RestartStateError("damage history sequence must be positive")
        if self.step is not None and int(self.step) < 0:
            raise RestartStateError("damage history step must be non-negative")
        if self.time is not None and not np.isfinite(float(self.time)):
            raise RestartStateError("damage history time must be finite")
        if not str(self.reason):
            raise RestartStateError("damage history reason must not be empty")
        count = len(self.element_ids)
        if len(self.previous_activity) != count or len(self.activity) != count:
            raise RestartStateError("damage history activity arrays must have equal lengths")
        if len(set(int(value) for value in self.element_ids)) != count:
            raise RestartStateError("damage history element_ids must be unique per event")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.previous_activity):
            raise RestartStateError("damage history previous_activity must be in [0, 1]")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.activity):
            raise RestartStateError("damage history activity must be in [0, 1]")
        if not set(self.newly_hard_deleted_ids).issubset(self.element_ids):
            raise RestartStateError(
                "newly_hard_deleted_ids must be a subset of event element_ids"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "step": None if self.step is None else int(self.step),
            "time": None if self.time is None else float(self.time),
            "reason": str(self.reason),
            "element_ids": [int(value) for value in self.element_ids],
            "previous_activity": [float(value) for value in self.previous_activity],
            "activity": [float(value) for value in self.activity],
            "newly_hard_deleted_ids": [
                int(value) for value in self.newly_hard_deleted_ids
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DamageHistoryEntry":
        if not isinstance(payload, Mapping):
            raise RestartStateError("each damage history entry must be a mapping")
        try:
            return cls(
                sequence=int(payload["sequence"]),
                step=None if payload.get("step") is None else int(payload["step"]),
                time=None if payload.get("time") is None else float(payload["time"]),
                reason=str(payload["reason"]),
                element_ids=tuple(int(value) for value in payload["element_ids"]),
                previous_activity=tuple(
                    float(value) for value in payload["previous_activity"]
                ),
                activity=tuple(float(value) for value in payload["activity"]),
                newly_hard_deleted_ids=tuple(
                    int(value)
                    for value in payload.get("newly_hard_deleted_ids", ())
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RestartStateError("malformed damage history entry") from exc


@dataclass(frozen=True, slots=True)
class ActivityChange:
    """Owned result of one activity update."""

    sequence: int
    element_ids: np.ndarray
    previous_activity: np.ndarray
    activity: np.ndarray
    newly_hard_deleted_mask: np.ndarray

    def __post_init__(self) -> None:
        ids = _readonly_array(self.element_ids, np.int64)
        previous = _readonly_array(self.previous_activity, float)
        activity = _readonly_array(self.activity, float)
        hard = _readonly_array(self.newly_hard_deleted_mask, bool)
        if not (ids.shape == previous.shape == activity.shape == hard.shape):
            raise ElementActivityError("activity change arrays must have matching shapes")
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "element_ids", ids)
        object.__setattr__(self, "previous_activity", previous)
        object.__setattr__(self, "activity", activity)
        object.__setattr__(self, "newly_hard_deleted_mask", hard)

    @property
    def changed_count(self) -> int:
        return int(self.element_ids.size)

    @property
    def newly_hard_deleted_ids(self) -> np.ndarray:
        return _readonly_array(
            self.element_ids[self.newly_hard_deleted_mask],
            np.int64,
        )


@dataclass(frozen=True, slots=True)
class ActivityFilter:
    """Reusable selection and scaling plan for element-owned records."""

    kind: str
    owner_element_ids: np.ndarray
    mask: np.ndarray
    scales: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        owners = _readonly_array(self.owner_element_ids, np.int64)
        mask = _readonly_array(self.mask, bool)
        scales = _readonly_array(self.scales, float)
        indices = _readonly_array(self.indices, np.intp)
        if owners.ndim != 1 or mask.shape != owners.shape or scales.shape != owners.shape:
            raise ElementActivityError("activity filter arrays must be one-dimensional and aligned")
        if not np.array_equal(indices, np.flatnonzero(mask)):
            raise ElementActivityError("activity filter indices must match its mask")
        object.__setattr__(self, "owner_element_ids", owners)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "indices", indices)

    @property
    def active_count(self) -> int:
        return int(self.indices.size)

    def apply(self, values: Any, *, axis: int = 0, scale: bool = False) -> np.ndarray:
        """Filter an aligned array and optionally apply selected owner scales."""

        array = np.asarray(values)
        normalized_axis = _normalized_axis(axis, array.ndim)
        if array.shape[normalized_axis] != self.owner_element_ids.size:
            raise ElementActivityError(
                f"filter axis has length {array.shape[normalized_axis]}, expected "
                f"{self.owner_element_ids.size}"
            )
        selected = np.take(array, self.indices, axis=normalized_axis)
        if not scale:
            return selected
        shape = [1] * selected.ndim
        shape[normalized_axis] = self.active_count
        return np.multiply(selected, self.scales[self.mask].reshape(shape))

    def apply_many(
        self,
        *values: Any,
        axis: int = 0,
        scale: bool = False,
    ) -> tuple[np.ndarray, ...]:
        return tuple(self.apply(value, axis=axis, scale=scale) for value in values)


@dataclass(frozen=True, slots=True)
class OrphanDofReport:
    """Support counts and orphan DOFs after activity filtering."""

    total_dofs: int
    original_support_counts: np.ndarray
    active_support_counts: np.ndarray
    orphan_mask: np.ndarray
    orphan_dofs: np.ndarray
    batch_size: int

    def __post_init__(self) -> None:
        original = _readonly_array(self.original_support_counts, np.int64)
        active = _readonly_array(self.active_support_counts, np.int64)
        mask = _readonly_array(self.orphan_mask, bool)
        dofs = _readonly_array(self.orphan_dofs, np.int64)
        expected = (int(self.total_dofs),)
        if original.shape != expected or active.shape != expected or mask.shape != expected:
            raise ElementActivityError("orphan-DOF report arrays have inconsistent shapes")
        if not np.array_equal(dofs, np.flatnonzero(mask)):
            raise ElementActivityError("orphan_dofs must match orphan_mask")
        object.__setattr__(self, "total_dofs", int(self.total_dofs))
        object.__setattr__(self, "original_support_counts", original)
        object.__setattr__(self, "active_support_counts", active)
        object.__setattr__(self, "orphan_mask", mask)
        object.__setattr__(self, "orphan_dofs", dofs)
        object.__setattr__(self, "batch_size", int(self.batch_size))

    @property
    def support_counts(self) -> np.ndarray:
        return self.active_support_counts

    @property
    def orphan_count(self) -> int:
        return int(self.orphan_dofs.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_dofs": int(self.total_dofs),
            "orphan_dof_count": self.orphan_count,
            "orphan_dofs": self.orphan_dofs.tolist(),
            "batch_size": int(self.batch_size),
        }


@dataclass(frozen=True, slots=True)
class CouplingResolution:
    """Deterministic assignments produced by a coupling lifecycle policy."""

    policy: CouplingPolicy
    owner_element_ids: np.ndarray
    assigned_element_ids: np.ndarray
    active_mask: np.ndarray
    reassigned_mask: np.ndarray
    deactivated_mask: np.ndarray

    def __post_init__(self) -> None:
        policy = _enum_value(CouplingPolicy, self.policy, "coupling")
        owners = _readonly_array(self.owner_element_ids, np.int64)
        assigned = _readonly_array(self.assigned_element_ids, np.int64)
        active = _readonly_array(self.active_mask, bool)
        reassigned = _readonly_array(self.reassigned_mask, bool)
        deactivated = _readonly_array(self.deactivated_mask, bool)
        if not (
            owners.shape
            == assigned.shape
            == active.shape
            == reassigned.shape
            == deactivated.shape
        ):
            raise ElementActivityError("coupling resolution arrays must have matching shapes")
        if np.any(active & deactivated) or not np.array_equal(deactivated, ~active):
            raise ElementActivityError("coupling active and deactivated masks are inconsistent")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "owner_element_ids", owners)
        object.__setattr__(self, "assigned_element_ids", assigned)
        object.__setattr__(self, "active_mask", active)
        object.__setattr__(self, "reassigned_mask", reassigned)
        object.__setattr__(self, "deactivated_mask", deactivated)

    @property
    def deactivated_count(self) -> int:
        return int(np.count_nonzero(self.deactivated_mask))

    @property
    def reassigned_count(self) -> int:
        return int(np.count_nonzero(self.reassigned_mask))

    def apply(self, values: Any, *, axis: int = 0) -> np.ndarray:
        array = np.asarray(values)
        normalized_axis = _normalized_axis(axis, array.ndim)
        if array.shape[normalized_axis] != self.active_mask.size:
            raise ElementActivityError("coupling payload is not aligned with the resolution")
        return np.compress(self.active_mask, array, axis=normalized_axis)


class ElementActivity:
    """Own vectorized activity and hard-deletion state for stable element IDs."""

    def __init__(
        self,
        element_ids: Sequence[int] | np.ndarray,
        activity: Sequence[float] | np.ndarray | float | None = None,
        *,
        policy: ElementActivityPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        ids = _element_id_vector(element_ids, allow_scalar=False)
        if np.unique(ids).size != ids.size:
            raise ElementActivityError("element_ids must contain unique stable identifiers")
        if policy is None:
            resolved_policy = ElementActivityPolicy()
        elif isinstance(policy, ElementActivityPolicy):
            resolved_policy = policy
        elif isinstance(policy, Mapping):
            resolved_policy = ElementActivityPolicy.from_dict(policy)
        else:
            raise ElementActivityError("policy must be ElementActivityPolicy or a mapping")

        initial = (
            np.ones(ids.size, dtype=float)
            if activity is None
            else _activity_vector(activity, ids.size)
        )
        hard_deleted = initial <= resolved_policy.hard_delete_threshold
        initial[hard_deleted] = 0.0

        stable_ids = np.array(ids, dtype=np.int64, copy=True, order="C")
        stable_ids.setflags(write=False)
        order = np.argsort(stable_ids, kind="stable")
        sorted_ids = np.array(stable_ids[order], copy=True)
        sorted_positions = np.array(order, dtype=np.intp, copy=True)
        sorted_ids.setflags(write=False)
        sorted_positions.setflags(write=False)

        self._element_ids = stable_ids
        self._sorted_ids = sorted_ids
        self._sorted_positions = sorted_positions
        self._id_to_index = MappingProxyType(
            {int(element_id): index for index, element_id in enumerate(stable_ids)}
        )
        self._activity = initial
        self._minimum_activity = initial.copy()
        self._hard_deleted = hard_deleted
        self._policy = resolved_policy
        self._history: list[DamageHistoryEntry] = []
        self._sequence = 0

    @property
    def policy(self) -> ElementActivityPolicy:
        return self._policy

    @property
    def policies(self) -> ElementActivityPolicy:
        return self._policy

    @property
    def element_ids(self) -> np.ndarray:
        result = self._element_ids.view()
        result.setflags(write=False)
        return result

    @property
    def activity(self) -> np.ndarray:
        result = self._activity.view()
        result.setflags(write=False)
        return result

    @property
    def minimum_activity(self) -> np.ndarray:
        result = self._minimum_activity.view()
        result.setflags(write=False)
        return result

    @property
    def damage(self) -> np.ndarray:
        return _readonly_array(1.0 - self._activity, float)

    @property
    def hard_deleted_mask(self) -> np.ndarray:
        result = self._hard_deleted.view()
        result.setflags(write=False)
        return result

    @property
    def deleted_mask(self) -> np.ndarray:
        return self.hard_deleted_mask

    @property
    def active_mask(self) -> np.ndarray:
        return _readonly_array(~self._hard_deleted, bool)

    @property
    def softened_mask(self) -> np.ndarray:
        return _readonly_array(
            (~self._hard_deleted) & (self._activity < 1.0),
            bool,
        )

    @property
    def history(self) -> tuple[DamageHistoryEntry, ...]:
        return tuple(self._history)

    @property
    def element_count(self) -> int:
        return int(self._element_ids.size)

    @property
    def n_elements(self) -> int:
        return self.element_count

    @property
    def sequence(self) -> int:
        """Monotonic state revision for cache and restart invalidation."""

        return int(self._sequence)

    def index(self, element_id: int) -> int:
        try:
            return int(self._id_to_index[int(element_id)])
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"Unknown element ID {element_id!r}") from exc

    def _indices_for_ids(self, element_ids: Any, *, name: str = "element_ids") -> tuple[np.ndarray, np.ndarray]:
        ids = _element_id_vector(element_ids, name=name)
        positions = np.searchsorted(self._sorted_ids, ids)
        matched = positions < self._sorted_ids.size
        if np.any(matched):
            valid = np.flatnonzero(matched)
            matched[valid] = self._sorted_ids[positions[valid]] == ids[valid]
        if not np.all(matched):
            unknown = ids[~matched]
            preview = ", ".join(str(int(value)) for value in unknown[:5])
            if unknown.size > 5:
                preview += ", ..."
            raise KeyError(f"Unknown element IDs: {preview}")
        return ids, self._sorted_positions[positions]

    def activity_for(self, element_ids: Any) -> np.ndarray:
        _ids, indices = self._indices_for_ids(element_ids)
        return _readonly_array(self._activity[indices], float)

    def set_activity(
        self,
        element_ids: Any,
        activity: Any,
        *,
        hard_delete: bool | Sequence[bool] | np.ndarray = False,
        allow_healing: bool | None = None,
        step: int | None = None,
        time: float | None = None,
        reason: str = "activity_update",
    ) -> ActivityChange:
        """Set activity for selected IDs and append one owned history event.

        Hard deletion is irreversible and forces an exact zero activity.  Unless
        healing is explicitly enabled, a larger requested value is clamped to
        the current value rather than reviving damage.
        """

        ids, indices = self._indices_for_ids(element_ids)
        if np.unique(ids).size != ids.size:
            raise ElementActivityError("element_ids must be unique within an activity update")
        values = _activity_vector(activity, ids.size)
        forced_hard = _boolean_vector(hard_delete, ids.size, "hard_delete")
        heal = self._policy.allow_healing if allow_healing is None else bool(allow_healing)
        normalized_step = None if step is None else int(step)
        normalized_time = None if time is None else float(time)
        normalized_reason = str(reason)
        if normalized_step is not None and normalized_step < 0:
            raise ElementActivityError("step must be non-negative")
        if normalized_time is not None and not np.isfinite(normalized_time):
            raise ElementActivityError("time must be finite")
        if not normalized_reason:
            raise ElementActivityError("reason must not be empty")

        previous = self._activity[indices].copy()
        previous_hard = self._hard_deleted[indices].copy()
        proposed = values if heal else np.minimum(values, previous)
        proposed[previous_hard] = 0.0
        next_hard = (
            previous_hard
            | forced_hard
            | (proposed <= self._policy.hard_delete_threshold)
        )
        proposed[next_hard] = 0.0
        newly_hard = next_hard & ~previous_hard
        changed = (proposed != previous) | newly_hard

        if not np.any(changed):
            empty_ids = np.empty(0, dtype=np.int64)
            empty_float = np.empty(0, dtype=float)
            empty_bool = np.empty(0, dtype=bool)
            return ActivityChange(
                sequence=self._sequence,
                element_ids=empty_ids,
                previous_activity=empty_float,
                activity=empty_float,
                newly_hard_deleted_mask=empty_bool,
            )

        changed_indices = indices[changed]
        changed_ids = ids[changed]
        self._activity[changed_indices] = proposed[changed]
        self._hard_deleted[changed_indices] = next_hard[changed]
        self._minimum_activity[changed_indices] = np.minimum(
            self._minimum_activity[changed_indices],
            proposed[changed],
        )
        self._sequence += 1
        event = DamageHistoryEntry(
            sequence=self._sequence,
            step=normalized_step,
            time=normalized_time,
            reason=normalized_reason,
            element_ids=tuple(int(value) for value in changed_ids),
            previous_activity=tuple(float(value) for value in previous[changed]),
            activity=tuple(float(value) for value in proposed[changed]),
            newly_hard_deleted_ids=tuple(
                int(value) for value in ids[newly_hard & changed]
            ),
        )
        self._history.append(event)
        return ActivityChange(
            sequence=self._sequence,
            element_ids=changed_ids,
            previous_activity=previous[changed],
            activity=proposed[changed],
            newly_hard_deleted_mask=newly_hard[changed],
        )

    def soften(
        self,
        element_ids: Any,
        factors: Any,
        *,
        relative: bool = True,
        step: int | None = None,
        time: float | None = None,
        reason: str = "softening",
    ) -> ActivityChange:
        """Apply relative softening factors or absolute activity values."""

        ids, indices = self._indices_for_ids(element_ids)
        factor_values = _activity_vector(factors, ids.size, "factors")
        target = self._activity[indices] * factor_values if relative else factor_values
        return self.set_activity(
            ids,
            target,
            step=step,
            time=time,
            reason=reason,
        )

    def apply_damage(
        self,
        element_ids: Any,
        damage: Any,
        *,
        incremental: bool = False,
        step: int | None = None,
        time: float | None = None,
        reason: str = "damage",
    ) -> ActivityChange:
        """Apply absolute damage ``1 - activity`` or an incremental fraction."""

        ids, indices = self._indices_for_ids(element_ids)
        damage_values = _activity_vector(damage, ids.size, "damage")
        target = (
            self._activity[indices] * (1.0 - damage_values)
            if incremental
            else 1.0 - damage_values
        )
        return self.set_activity(
            ids,
            target,
            step=step,
            time=time,
            reason=reason,
        )

    def hard_delete(
        self,
        element_ids: Any,
        *,
        step: int | None = None,
        time: float | None = None,
        reason: str = "hard_deletion",
    ) -> ActivityChange:
        ids, _indices = self._indices_for_ids(element_ids)
        return self.set_activity(
            ids,
            np.zeros(ids.size, dtype=float),
            hard_delete=True,
            step=step,
            time=time,
            reason=reason,
        )

    delete = hard_delete

    def _all_scales(self, quantity: str) -> np.ndarray:
        normalized = str(quantity).lower()
        if normalized not in _QUANTITIES:
            choices = ", ".join(_QUANTITIES)
            raise ElementActivityError(f"quantity must be one of {choices}")
        contribution_policy = getattr(self._policy, normalized)
        if contribution_policy is ContributionPolicy.ACTIVITY:
            scales = self._activity.copy()
            scales[self._hard_deleted] = 0.0
        elif contribution_policy is ContributionPolicy.DELETE_ONLY:
            scales = (~self._hard_deleted).astype(float)
        else:
            scales = np.ones(self.element_count, dtype=float)
        if normalized == "stiffness" and self._policy.minimum_stiffness_scale > 0.0:
            active = ~self._hard_deleted
            scales[active] = np.maximum(
                scales[active],
                self._policy.minimum_stiffness_scale,
            )
        return scales

    def scales(self, quantity: str, element_ids: Any | None = None) -> np.ndarray:
        values = self._all_scales(quantity)
        if element_ids is None:
            return _readonly_array(values, float)
        _ids, indices = self._indices_for_ids(element_ids)
        return _readonly_array(values[indices], float)

    def stiffness_scales(self, element_ids: Any | None = None) -> np.ndarray:
        return self.scales("stiffness", element_ids)

    def mass_scales(self, element_ids: Any | None = None) -> np.ndarray:
        return self.scales("mass", element_ids)

    def damping_scales(self, element_ids: Any | None = None) -> np.ndarray:
        return self.scales("damping", element_ids)

    def load_scales(self, element_ids: Any | None = None) -> np.ndarray:
        return self.scales("load", element_ids)

    def contact_scales(self, element_ids: Any | None = None) -> np.ndarray:
        return self.scales("contact", element_ids)

    @property
    def stiffness_scale(self) -> np.ndarray:
        return self.stiffness_scales()

    @property
    def mass_scale(self) -> np.ndarray:
        return self.mass_scales()

    @property
    def damping_scale(self) -> np.ndarray:
        return self.damping_scales()

    @property
    def load_scale(self) -> np.ndarray:
        return self.load_scales()

    @property
    def contact_scale(self) -> np.ndarray:
        return self.contact_scales()

    def scale(
        self,
        values: Any,
        quantity: str,
        *,
        element_ids: Any | None = None,
        axis: int = 0,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Scale aligned data without changing its shape or storage topology."""

        array = np.asarray(values)
        normalized_axis = _normalized_axis(axis, array.ndim)
        factors = self.scales(quantity, element_ids)
        if array.shape[normalized_axis] != factors.size:
            raise ElementActivityError(
                f"element axis has length {array.shape[normalized_axis]}, expected {factors.size}"
            )
        shape = [1] * array.ndim
        shape[normalized_axis] = factors.size
        broadcast = factors.reshape(shape)
        if out is None:
            return np.multiply(array, broadcast)
        target = np.asarray(out)
        if target.shape != array.shape:
            raise ElementActivityError("out must have the same shape as values")
        try:
            np.multiply(array, broadcast, out=target, casting="same_kind")
        except (TypeError, ValueError) as exc:
            raise ElementActivityError(
                "out must have a floating dtype compatible with scaled values"
            ) from exc
        return target

    def scale_stiffness(self, values: Any, **kwargs: Any) -> np.ndarray:
        return self.scale(values, "stiffness", **kwargs)

    def scale_mass(self, values: Any, **kwargs: Any) -> np.ndarray:
        return self.scale(values, "mass", **kwargs)

    def scale_damping(self, values: Any, **kwargs: Any) -> np.ndarray:
        return self.scale(values, "damping", **kwargs)

    def scale_load(self, values: Any, **kwargs: Any) -> np.ndarray:
        return self.scale(values, "load", **kwargs)

    def scale_contact(self, values: Any, **kwargs: Any) -> np.ndarray:
        return self.scale(values, "contact", **kwargs)

    def scale_contributions(
        self,
        values: Any,
        owner_element_ids: Any,
        quantity: str,
        *,
        axis: int = 0,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Scale repeated per-entry owners, suitable for fixed CSR data maps."""

        return self.scale(
            values,
            quantity,
            element_ids=owner_element_ids,
            axis=axis,
            out=out,
        )

    def _build_filter(self, owner_element_ids: Any, kind: str) -> ActivityFilter:
        ids, _indices = self._indices_for_ids(owner_element_ids, name="owner_element_ids")
        scales = np.asarray(self.scales(kind, ids))
        mask = scales > 0.0
        return ActivityFilter(
            kind=kind,
            owner_element_ids=ids,
            mask=mask,
            scales=scales,
            indices=np.flatnonzero(mask),
        )

    def load_filter(self, owner_element_ids: Any) -> ActivityFilter:
        return self._build_filter(owner_element_ids, "load")

    def contact_filter(self, owner_element_ids: Any) -> ActivityFilter:
        return self._build_filter(owner_element_ids, "contact")

    def active_load_mask(self, owner_element_ids: Any) -> np.ndarray:
        return self.load_filter(owner_element_ids).mask

    def active_contact_mask(self, owner_element_ids: Any) -> np.ndarray:
        return self.contact_filter(owner_element_ids).mask

    def filter_loads(
        self,
        owner_element_ids: Any,
        values: Any | None = None,
        *,
        axis: int = 0,
        scale: bool = True,
    ) -> ActivityFilter | np.ndarray:
        selection = self.load_filter(owner_element_ids)
        if values is None:
            return selection
        return selection.apply(values, axis=axis, scale=scale)

    def filter_contacts(
        self,
        owner_element_ids: Any,
        values: Any | None = None,
        *,
        axis: int = 0,
        scale: bool = True,
    ) -> ActivityFilter | np.ndarray:
        selection = self.contact_filter(owner_element_ids)
        if values is None:
            return selection
        return selection.apply(values, axis=axis, scale=scale)

    def detect_orphan_dofs(
        self,
        connectivity: Any,
        *,
        element_ids: Any | None = None,
        connectivity_element_ids: Any | None = None,
        total_dofs: int | None = None,
        batch_size: int = 4096,
        activity_threshold: float = 0.0,
        include_never_connected: bool = False,
    ) -> OrphanDofReport:
        """Find DOFs that lose every supporting element, in bounded batches.

        ``connectivity`` is a rectangular element-to-global-DOF array.  ``-1``
        may pad shorter rows.  By default only DOFs supported by the original
        connectivity can become orphaned; globally allocated but never connected
        DOFs are omitted unless ``include_never_connected`` is true.
        """

        raw = np.asarray(connectivity)
        if raw.ndim != 2 or raw.dtype.kind not in "iu":
            raise ElementActivityError("connectivity must be a two-dimensional integer array")
        try:
            dofs = np.asarray(raw, dtype=np.int64)
        except (OverflowError, ValueError) as exc:
            raise ElementActivityError("connectivity values must fit signed 64-bit integers") from exc
        if np.any(dofs < -1):
            raise ElementActivityError("connectivity may use only -1 as a padding sentinel")
        if element_ids is not None and connectivity_element_ids is not None:
            raise ElementActivityError(
                "provide element_ids or connectivity_element_ids, not both"
            )
        row_id_input = (
            connectivity_element_ids
            if connectivity_element_ids is not None
            else element_ids
        )
        if row_id_input is None:
            if dofs.shape[0] != self.element_count:
                raise ElementActivityError(
                    "connectivity row count must equal element count when row IDs are omitted"
                )
            row_indices = np.arange(self.element_count, dtype=np.intp)
        else:
            row_ids, row_indices = self._indices_for_ids(
                row_id_input,
                name="connectivity_element_ids",
            )
            if row_ids.size != dofs.shape[0]:
                raise ElementActivityError(
                    "connectivity_element_ids must have one ID per connectivity row"
                )

        normalized_batch_size = int(batch_size)
        if normalized_batch_size <= 0:
            raise ElementActivityError("batch_size must be positive")
        threshold = float(activity_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ElementActivityError("activity_threshold must be finite and in [0, 1)")

        connected = dofs[dofs >= 0]
        inferred_total = int(connected.max()) + 1 if connected.size else 0
        if total_dofs is None:
            normalized_total_dofs = inferred_total
        else:
            normalized_total_dofs = int(total_dofs)
            if normalized_total_dofs < 0:
                raise ElementActivityError("total_dofs must be non-negative")
            if inferred_total > normalized_total_dofs:
                raise ElementActivityError(
                    "connectivity contains a DOF outside the requested total_dofs"
                )

        original_counts = np.zeros(normalized_total_dofs, dtype=np.int64)
        active_counts = np.zeros(normalized_total_dofs, dtype=np.int64)
        row_active = (
            (~self._hard_deleted[row_indices])
            & (self._activity[row_indices] > threshold)
        )
        for start in range(0, dofs.shape[0], normalized_batch_size):
            stop = min(start + normalized_batch_size, dofs.shape[0])
            block = dofs[start:stop]
            valid = block >= 0
            block_dofs = block[valid]
            if block_dofs.size:
                original_counts += np.bincount(
                    block_dofs,
                    minlength=normalized_total_dofs,
                ).astype(np.int64, copy=False)
            active_rows = row_active[start:stop]
            if np.any(active_rows):
                active_valid = valid & active_rows[:, None]
                active_dofs = block[active_valid]
                if active_dofs.size:
                    active_counts += np.bincount(
                        active_dofs,
                        minlength=normalized_total_dofs,
                    ).astype(np.int64, copy=False)

        orphan_mask = active_counts == 0
        if not include_never_connected:
            orphan_mask &= original_counts > 0
        orphan_dofs = np.flatnonzero(orphan_mask).astype(np.int64, copy=False)
        return OrphanDofReport(
            total_dofs=normalized_total_dofs,
            original_support_counts=original_counts,
            active_support_counts=active_counts,
            orphan_mask=orphan_mask,
            orphan_dofs=orphan_dofs,
            batch_size=normalized_batch_size,
        )

    def orphan_dofs(self, connectivity: Any, **kwargs: Any) -> np.ndarray:
        return self.detect_orphan_dofs(connectivity, **kwargs).orphan_dofs

    find_orphan_dofs = orphan_dofs

    def resolve_couplings(
        self,
        owner_element_ids: Any,
        candidate_element_ids: Any | None = None,
        *,
        policy: CouplingPolicy | str | None = None,
    ) -> CouplingResolution:
        """Deactivate or deterministically reassign deleted coupling owners.

        Reassignment chooses the active candidate with greatest activity; ties
        retain candidate-column order.  Unknown candidates and ``-1`` padding
        are ignored.  A coupling with no eligible candidate is deactivated.
        """

        owners, owner_indices = self._indices_for_ids(
            owner_element_ids,
            name="owner_element_ids",
        )
        resolved_policy = (
            self._policy.coupling
            if policy is None
            else _enum_value(CouplingPolicy, policy, "coupling")
        )
        owner_active = ~self._hard_deleted[owner_indices]
        assigned = owners.copy()
        reassigned = np.zeros(owners.size, dtype=bool)

        if resolved_policy is CouplingPolicy.KEEP:
            active = np.ones(owners.size, dtype=bool)
        elif resolved_policy is CouplingPolicy.ERROR:
            if np.any(~owner_active):
                deleted = ", ".join(str(int(value)) for value in owners[~owner_active])
                raise CouplingResolutionError(
                    f"deleted coupling owners require resolution: {deleted}"
                )
            active = owner_active.copy()
        elif resolved_policy is CouplingPolicy.DEACTIVATE:
            active = owner_active.copy()
            assigned[~active] = -1
        else:
            active = owner_active.copy()
            assigned[~owner_active] = -1
            if candidate_element_ids is None:
                candidates = np.empty((owners.size, 0), dtype=np.int64)
            else:
                raw_candidates = np.asarray(candidate_element_ids)
                if raw_candidates.ndim == 1:
                    if raw_candidates.size != owners.size:
                        raise ElementActivityError(
                            "one-dimensional candidate_element_ids must have one candidate per coupling"
                        )
                    raw_candidates = raw_candidates.reshape(owners.size, 1)
                if raw_candidates.ndim != 2 or raw_candidates.shape[0] != owners.size:
                    raise ElementActivityError(
                        "candidate_element_ids must be a two-dimensional array with one row per coupling"
                    )
                if raw_candidates.dtype.kind not in "iu":
                    raise ElementActivityError("candidate_element_ids must contain integers")
                candidates = np.asarray(raw_candidates, dtype=np.int64)
                if np.any(candidates < -1):
                    raise ElementActivityError(
                        "candidate_element_ids may use only -1 as a padding sentinel"
                    )

            if candidates.shape[1] > 0 and np.any(~owner_active):
                flat = candidates.reshape(-1)
                candidate_indices = np.full(flat.shape, -1, dtype=np.int64)
                nonpadding = flat >= 0
                if np.any(nonpadding):
                    values = flat[nonpadding]
                    positions = np.searchsorted(self._sorted_ids, values)
                    known = positions < self._sorted_ids.size
                    if np.any(known):
                        known_indices = np.flatnonzero(known)
                        known[known_indices] = (
                            self._sorted_ids[positions[known_indices]]
                            == values[known_indices]
                        )
                    mapped = np.full(values.shape, -1, dtype=np.int64)
                    mapped[known] = self._sorted_positions[positions[known]]
                    candidate_indices[nonpadding] = mapped
                candidate_indices = candidate_indices.reshape(candidates.shape)
                eligible = candidate_indices >= 0
                scores = np.full(candidates.shape, -np.inf, dtype=float)
                if np.any(eligible):
                    rows, columns = np.nonzero(eligible)
                    local_indices = candidate_indices[rows, columns]
                    candidate_active = ~self._hard_deleted[local_indices]
                    active_rows = rows[candidate_active]
                    active_columns = columns[candidate_active]
                    active_indices = local_indices[candidate_active]
                    scores[active_rows, active_columns] = self._activity[active_indices]
                has_replacement = np.any(np.isfinite(scores), axis=1)
                best_column = np.argmax(scores, axis=1)
                selected = candidates[np.arange(owners.size), best_column]
                reassigned = (~owner_active) & has_replacement
                assigned[reassigned] = selected[reassigned]
                active[reassigned] = True

        deactivated = ~active
        return CouplingResolution(
            policy=resolved_policy,
            owner_element_ids=owners,
            assigned_element_ids=assigned,
            active_mask=active,
            reassigned_mask=reassigned,
            deactivated_mask=deactivated,
        )

    def to_restart(self, *, include_history: bool = True) -> dict[str, Any]:
        """Return an owned, JSON-safe restart payload."""

        return {
            "schema": _RESTART_SCHEMA,
            "version": _RESTART_VERSION,
            "element_ids": self._element_ids.tolist(),
            "activity": self._activity.tolist(),
            "minimum_activity": self._minimum_activity.tolist(),
            "hard_deleted": self._hard_deleted.tolist(),
            "policy": self._policy.to_dict(),
            "sequence": int(self._sequence),
            "history": (
                [entry.to_dict() for entry in self._history]
                if include_history
                else []
            ),
        }

    serialize = to_restart
    restart_state = to_restart

    @classmethod
    def from_restart(
        cls,
        payload: Mapping[str, Any],
        *,
        policy: ElementActivityPolicy | Mapping[str, Any] | None = None,
    ) -> "ElementActivity":
        """Construct a manager from a validated restart payload."""

        if not isinstance(payload, Mapping):
            raise RestartStateError("element activity restart must be a mapping")
        if payload.get("schema") != _RESTART_SCHEMA:
            raise RestartStateError("unsupported element activity restart schema")
        try:
            version = int(payload["version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RestartStateError("element activity restart version is missing") from exc
        if version != _RESTART_VERSION:
            raise RestartStateError(f"unsupported element activity restart version {version}")
        try:
            ids = _element_id_vector(payload["element_ids"], allow_scalar=False)
            activity = _activity_vector(payload["activity"], ids.size)
        except (KeyError, ElementActivityError) as exc:
            raise RestartStateError(str(exc)) from exc

        if policy is None:
            try:
                resolved_policy = ElementActivityPolicy.from_dict(payload["policy"])
            except KeyError as exc:
                raise RestartStateError("element activity restart policy is missing") from exc
        elif isinstance(policy, ElementActivityPolicy):
            resolved_policy = policy
        elif isinstance(policy, Mapping):
            resolved_policy = ElementActivityPolicy.from_dict(policy)
        else:
            raise RestartStateError("policy override must be ElementActivityPolicy or a mapping")

        try:
            hard_raw = np.asarray(payload["hard_deleted"])
        except KeyError as exc:
            raise RestartStateError("hard_deleted restart state is missing") from exc
        if hard_raw.shape != (ids.size,) or hard_raw.dtype.kind != "b":
            raise RestartStateError("hard_deleted must be a boolean array aligned with element_ids")
        hard_deleted = np.array(hard_raw, dtype=bool, copy=True)

        manager = cls(ids, activity, policy=resolved_policy)
        if not np.array_equal(manager._hard_deleted, hard_deleted):
            raise RestartStateError(
                "hard_deleted is inconsistent with activity and hard_delete_threshold"
            )
        try:
            minimum = _activity_vector(
                payload.get("minimum_activity", activity),
                ids.size,
                "minimum_activity",
            )
        except ElementActivityError as exc:
            raise RestartStateError(str(exc)) from exc
        if np.any(minimum > activity):
            raise RestartStateError("minimum_activity cannot exceed current activity")

        raw_history = payload.get("history", ())
        if not isinstance(raw_history, Sequence) or isinstance(raw_history, (str, bytes)):
            raise RestartStateError("damage history must be a sequence")
        history = [DamageHistoryEntry.from_dict(entry) for entry in raw_history]
        sequences = [entry.sequence for entry in history]
        if any(later <= earlier for earlier, later in zip(sequences, sequences[1:])):
            raise RestartStateError("damage history sequences must be strictly increasing")
        for entry in history:
            try:
                manager._indices_for_ids(entry.element_ids, name="history element_ids")
            except KeyError as exc:
                raise RestartStateError(str(exc)) from exc
        try:
            sequence = int(payload.get("sequence", sequences[-1] if sequences else 0))
        except (TypeError, ValueError) as exc:
            raise RestartStateError("restart sequence must be an integer") from exc
        if sequence < 0 or (sequences and sequence < sequences[-1]):
            raise RestartStateError("restart sequence precedes damage history")

        manager._minimum_activity = minimum
        manager._history = history
        manager._sequence = sequence
        return manager

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        policy: ElementActivityPolicy | Mapping[str, Any] | None = None,
    ) -> "ElementActivity":
        return cls.from_restart(payload, policy=policy)

    @classmethod
    def deserialize(
        cls,
        payload: Mapping[str, Any],
        *,
        policy: ElementActivityPolicy | Mapping[str, Any] | None = None,
    ) -> "ElementActivity":
        return cls.from_restart(payload, policy=policy)

    def restore_restart(
        self,
        payload: Mapping[str, Any],
        *,
        require_same_policy: bool = True,
    ) -> None:
        """Restore into an existing manager while preserving stable ID layout."""

        restored = type(self).from_restart(payload)
        if not np.array_equal(restored._element_ids, self._element_ids):
            raise RestartStateError("restart element_ids do not match this manager")
        if require_same_policy and restored.policy != self.policy:
            raise RestartStateError("restart policy does not match this manager")
        self._activity[:] = restored._activity
        self._minimum_activity[:] = restored._minimum_activity
        self._hard_deleted[:] = restored._hard_deleted
        self._history = list(restored._history)
        self._sequence = restored._sequence

    load_restart = restore_restart

    def _element_values(self, values: Any, name: str) -> np.ndarray:
        try:
            result = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ElementActivityError(f"{name} must contain floating values") from exc
        if result.shape != (self.element_count,):
            raise ElementActivityError(
                f"{name} must have shape ({self.element_count},) in stable element order"
            )
        if np.any(~np.isfinite(result)) or np.any(result < 0.0):
            raise ElementActivityError(f"{name} must be finite and non-negative")
        return result

    def conditioning_diagnostics(
        self,
        *,
        reference_condition_number: float | None = None,
    ) -> dict[str, Any]:
        """Estimate conditioning amplification introduced by stiffness scales."""

        scales = np.asarray(self.stiffness_scales())
        positive = scales[scales > 0.0]
        if scales.size == 0:
            minimum_positive = 1.0
            maximum = 1.0
            ratio = 1.0
        elif positive.size == 0:
            minimum_positive = 0.0
            maximum = 0.0
            ratio = float("inf")
        else:
            minimum_positive = float(np.min(positive))
            maximum = float(np.max(positive))
            ratio = maximum / minimum_positive
        if reference_condition_number is None:
            estimate = None
            warning_value = ratio
        else:
            reference = float(reference_condition_number)
            if not np.isfinite(reference) or reference <= 0.0:
                raise ElementActivityError(
                    "reference_condition_number must be finite and positive"
                )
            estimate = reference * ratio
            warning_value = estimate
        return {
            "minimum_positive_stiffness_scale": minimum_positive,
            "maximum_stiffness_scale": maximum,
            "stiffness_scale_ratio": float(ratio),
            "zero_stiffness_scale_count": int(np.count_nonzero(scales == 0.0)),
            "reference_condition_number": (
                None
                if reference_condition_number is None
                else float(reference_condition_number)
            ),
            "estimated_condition_number": None if estimate is None else float(estimate),
            "conditioning_warning_ratio": float(
                self._policy.conditioning_warning_ratio
            ),
            "conditioning_warning": bool(
                not np.isfinite(warning_value)
                or warning_value >= self._policy.conditioning_warning_ratio
            ),
        }

    def removed_mass_energy_diagnostics(
        self,
        *,
        element_mass: Any | None = None,
        element_energy: Any | None = None,
    ) -> dict[str, Any]:
        """Report contribution-weighted removed mass and structural energy."""

        result: dict[str, Any] = {
            "mass_diagnostics_available": element_mass is not None,
            "input_mass": None,
            "retained_mass": None,
            "removed_mass": None,
            "removed_mass_fraction": None,
            "energy_diagnostics_available": element_energy is not None,
            "input_energy": None,
            "retained_energy": None,
            "removed_energy": None,
            "removed_energy_fraction": None,
            "energy_scale_basis": "stiffness",
        }
        if element_mass is not None:
            mass = self._element_values(element_mass, "element_mass")
            mass_scales = np.asarray(self.mass_scales())
            input_mass = float(np.sum(mass, dtype=float))
            retained_mass = float(np.dot(mass, mass_scales))
            removed_mass = float(np.dot(mass, 1.0 - mass_scales))
            result.update(
                {
                    "input_mass": input_mass,
                    "retained_mass": retained_mass,
                    "removed_mass": removed_mass,
                    "removed_mass_fraction": (
                        0.0 if input_mass == 0.0 else removed_mass / input_mass
                    ),
                }
            )
        if element_energy is not None:
            energy = self._element_values(element_energy, "element_energy")
            energy_scales = np.asarray(self.stiffness_scales())
            input_energy = float(np.sum(energy, dtype=float))
            retained_energy = float(np.dot(energy, energy_scales))
            removed_energy = float(np.dot(energy, 1.0 - energy_scales))
            result.update(
                {
                    "input_energy": input_energy,
                    "retained_energy": retained_energy,
                    "removed_energy": removed_energy,
                    "removed_energy_fraction": (
                        0.0 if input_energy == 0.0 else removed_energy / input_energy
                    ),
                }
            )
        return result

    def diagnostics(
        self,
        *,
        element_mass: Any | None = None,
        element_energy: Any | None = None,
        reference_condition_number: float | None = None,
        orphan_report: OrphanDofReport | None = None,
    ) -> dict[str, Any]:
        """Return lifecycle, conditioning, and removed quantity diagnostics."""

        if orphan_report is not None and not isinstance(orphan_report, OrphanDofReport):
            raise ElementActivityError("orphan_report must be OrphanDofReport")
        if self.element_count:
            activity_min = float(np.min(self._activity))
            activity_max = float(np.max(self._activity))
            activity_mean = float(np.mean(self._activity))
        else:
            activity_min = activity_max = activity_mean = 1.0
        result: dict[str, Any] = {
            "element_count": self.element_count,
            "active_element_count": int(np.count_nonzero(~self._hard_deleted)),
            "softened_element_count": int(np.count_nonzero(self.softened_mask)),
            "hard_deleted_element_count": int(np.count_nonzero(self._hard_deleted)),
            "activity_min": activity_min,
            "activity_max": activity_max,
            "activity_mean": activity_mean,
            "history_event_count": len(self._history),
            "history_sequence": int(self._sequence),
            "orphan_dof_count": (
                None if orphan_report is None else orphan_report.orphan_count
            ),
            "policies": self._policy.to_dict(),
            "fixed_matrix_sparsity": True,
        }
        result.update(
            self.conditioning_diagnostics(
                reference_condition_number=reference_condition_number
            )
        )
        result.update(
            self.removed_mass_energy_diagnostics(
                element_mass=element_mass,
                element_energy=element_energy,
            )
        )
        return result


ElementActivityManager = ElementActivity


__all__ = [
    "ActivityChange",
    "ActivityFilter",
    "ActivityPolicy",
    "ContributionPolicy",
    "CouplingPolicy",
    "CouplingResolution",
    "CouplingResolutionError",
    "DamageHistoryEntry",
    "ElementActivity",
    "ElementActivityError",
    "ElementActivityManager",
    "ElementActivityPolicies",
    "ElementActivityPolicy",
    "OrphanDofReport",
    "RestartStateError",
]
