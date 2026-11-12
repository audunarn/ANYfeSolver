"""Cooperative solve cancellation and structured progress contracts.

The numerical kernels in :mod:`anyfesolver` are intentionally synchronous.  A
caller that runs them in a worker thread can nevertheless request cancellation
at deterministic safe points by sharing a :class:`CancellationToken`.  Sparse
factorizations performed by third-party libraries cannot be interrupted; the
next safe point raises :class:`SolveCancelled` as soon as control returns.

``ProgressEvent`` implements ``Mapping`` so callbacks written for the legacy
dictionary payloads continue to use ``event["type"]`` and ``event.get(...)``.
New consumers can use the typed attributes and :meth:`to_dict` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock
from time import time
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, TypeAlias


class SolveCancelled(RuntimeError):
    """Raised at a solver safe point after cooperative cancellation."""

    def __init__(self, reason: str = "cancelled by caller", *, stage: str = "") -> None:
        self.reason = str(reason or "cancelled by caller")
        self.stage = str(stage or "")
        suffix = f" at {self.stage}" if self.stage else ""
        super().__init__(f"Solve cancelled{suffix}: {self.reason}")


class CancellationToken:
    """Thread-safe, one-way cooperative cancellation token.

    ``cancel`` is idempotent.  The first non-empty reason wins, which keeps job
    diagnostics stable when several UI actions race to cancel the same solve.
    """

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._reason = "cancelled by caller"

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""

        return self._event.is_set()

    @property
    def reason(self) -> str:
        """Stable human-readable cancellation reason."""

        with self._lock:
            return self._reason

    def cancel(self, reason: str = "cancelled by caller") -> bool:
        """Request cancellation and return ``True`` only for the first call."""

        with self._lock:
            if self._event.is_set():
                return False
            self._reason = str(reason or "cancelled by caller")
            self._event.set()
            return True

    def raise_if_cancelled(self, stage: str = "") -> None:
        """Raise :class:`SolveCancelled` when cancellation was requested."""

        if self._event.is_set():
            raise SolveCancelled(self.reason, stage=stage)

    def checkpoint(self, stage: str = "") -> None:
        """Alias for :meth:`raise_if_cancelled` suited to iterative code."""

        self.raise_if_cancelled(stage)


def cancellation_safe_point(
    token: Optional[CancellationToken],
    stage: str = "",
) -> None:
    """Check an optional token at a documented solver safe point."""

    if token is not None:
        token.raise_if_cancelled(stage)


@dataclass(frozen=True)
class ProgressEvent(Mapping[str, Any]):
    """One structured, mapping-compatible solver progress notification.

    ``completed`` and ``total`` describe work units; ``fraction`` is derived
    when omitted.  Solver-specific values live in ``metadata`` and are exposed
    as top-level mapping keys for compatibility with historical callbacks.
    """

    event_type: str
    stage: str
    message: str = ""
    completed: Optional[float] = None
    total: Optional[float] = None
    fraction: Optional[float] = None
    iteration: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)

    def __post_init__(self) -> None:
        if not str(self.event_type).strip():
            raise ValueError("event_type must not be empty")
        if not str(self.stage).strip():
            raise ValueError("stage must not be empty")
        completed = None if self.completed is None else float(self.completed)
        total = None if self.total is None else float(self.total)
        fraction = None if self.fraction is None else float(self.fraction)
        if completed is not None and completed < 0.0:
            raise ValueError("completed must be non-negative")
        if total is not None and total < 0.0:
            raise ValueError("total must be non-negative")
        if fraction is None and completed is not None and total not in (None, 0.0):
            fraction = completed / total
        if fraction is not None and not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        object.__setattr__(self, "completed", completed)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def type(self) -> str:
        """Legacy-friendly alias for :attr:`event_type`."""

        return self.event_type

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self.metadata)
        payload.update(
            {
                "type": self.event_type,
                "stage": self.stage,
                "message": self.message,
                "completed": self.completed,
                "total": self.total,
                "fraction": self.fraction,
                "iteration": self.iteration,
                "timestamp": float(self.timestamp),
            }
        )
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def copy(self) -> Dict[str, Any]:
        """Return a mutable dictionary, matching legacy progress payloads."""

        return self.to_dict()


ProgressCallback: TypeAlias = Callable[[ProgressEvent], None]


def emit_progress(
    callback: Optional[Callable[[ProgressEvent], None]],
    event_type: str,
    stage: str,
    *,
    message: str = "",
    completed: Optional[float] = None,
    total: Optional[float] = None,
    fraction: Optional[float] = None,
    iteration: Optional[int] = None,
    suppress_callback_errors: bool = True,
    **metadata: Any,
) -> Optional[ProgressEvent]:
    """Create and deliver one :class:`ProgressEvent` when a callback exists.

    Progress observers are non-authoritative by default: an exception in UI
    rendering must not corrupt a numerical solve.  Diagnostic applications can
    opt out of suppression when calling this helper directly.
    """

    if callback is None:
        return None
    event = ProgressEvent(
        event_type=event_type,
        stage=stage,
        message=message,
        completed=completed,
        total=total,
        fraction=fraction,
        iteration=iteration,
        metadata=metadata,
    )
    try:
        callback(event)
    except Exception:
        if not suppress_callback_errors:
            raise
    return event
