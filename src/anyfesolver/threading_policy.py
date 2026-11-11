"""Scoped native/solver thread controls with observable diagnostics."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import threading
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, TypeVar

try:
    from threadpoolctl import threadpool_info, threadpool_limits

    _HAS_THREADPOOLCTL = True
    _THREADPOOLCTL_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - exercised in isolated tests
    threadpool_info = None  # type: ignore[assignment]
    threadpool_limits = None  # type: ignore[assignment]
    _HAS_THREADPOOLCTL = False
    _THREADPOOLCTL_ERROR = str(exc)


F = TypeVar("F", bound=Callable[..., Any])
_SOLVER_THREADS: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "anyfesolver_solver_threads", default=None
)
_ACTIVE_NATIVE_LIMIT: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "anyfesolver_active_native_limit", default=None
)
_ACTIVE_NATIVE_LIMIT_OWNER: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "anyfesolver_active_native_limit_owner", default=None
)
_ACTIVE_NATIVE_DEFAULT_OWNER: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "anyfesolver_active_native_default_owner", default=None
)


class _NativeScopeCoordinator:
    """Coordinate process-global native-pool mutations across Python threads.

    Unlimited/default scopes are concurrent readers. Explicit limits are
    exclusive writers because ``threadpoolctl`` changes process-global native
    pools. A reader may upgrade in-place: its read depth is temporarily
    removed while it owns the writer and restored when the explicit scope
    exits. Same-thread nesting remains reentrant.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._reader_depth: Dict[int, int] = {}
        self._reader_count = 0
        self._writer_owner: Optional[int] = None
        self._writer_depth = 0
        self._waiting_writers = 0

    def acquire_reader(self) -> tuple[str, int]:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_owner == owner:
                return ("writer_inherited", 0)
            own_depth = self._reader_depth.get(owner, 0)
            while self._writer_owner is not None or (
                self._waiting_writers > 0 and own_depth == 0
            ):
                self._condition.wait()
                own_depth = self._reader_depth.get(owner, 0)
            self._reader_depth[owner] = own_depth + 1
            self._reader_count += 1
            return ("reader", 1)

    def release_reader(self, token: tuple[str, int]) -> None:
        if token[0] == "writer_inherited":
            return
        owner = threading.get_ident()
        with self._condition:
            depth = self._reader_depth.get(owner, 0)
            if depth <= 0:  # pragma: no cover - internal misuse guard
                raise RuntimeError("native thread reader scope released by non-owner")
            if depth == 1:
                self._reader_depth.pop(owner, None)
            else:
                self._reader_depth[owner] = depth - 1
            self._reader_count -= 1
            if self._reader_count == 0:
                self._condition.notify_all()

    def acquire_writer(self) -> tuple[str, int]:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_owner == owner:
                self._writer_depth += 1
                return ("writer_reentrant", 0)

            restore_depth = self._reader_depth.pop(owner, 0)
            if restore_depth:
                self._reader_count -= restore_depth
                self._condition.notify_all()

            self._waiting_writers += 1
            acquired = False
            try:
                while self._writer_owner is not None or self._reader_count > 0:
                    self._condition.wait()
                self._writer_owner = owner
                self._writer_depth = 1
                acquired = True
            finally:
                self._waiting_writers -= 1
                if not acquired and restore_depth:
                    self._reader_depth[owner] = (
                        self._reader_depth.get(owner, 0) + restore_depth
                    )
                    self._reader_count += restore_depth
                if not acquired:
                    self._condition.notify_all()
            return ("writer", restore_depth)

    def release_writer(self, token: tuple[str, int]) -> None:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_owner != owner:  # pragma: no cover - internal misuse guard
                raise RuntimeError("native thread writer scope released by non-owner")
            self._writer_depth -= 1
            if self._writer_depth > 0:
                return
            self._writer_owner = None
            restore_depth = int(token[1])
            if restore_depth:
                self._reader_depth[owner] = (
                    self._reader_depth.get(owner, 0) + restore_depth
                )
                self._reader_count += restore_depth
            self._condition.notify_all()


_NATIVE_SCOPE_COORDINATOR = _NativeScopeCoordinator()


def _pool_snapshot() -> list[Dict[str, Any]]:
    if not _HAS_THREADPOOLCTL or threadpool_info is None:
        return []
    pools = []
    for pool in threadpool_info():
        pools.append(
            {
                "user_api": pool.get("user_api"),
                "internal_api": pool.get("internal_api"),
                "prefix": pool.get("prefix"),
                "num_threads": pool.get("num_threads"),
                "version": pool.get("version"),
            }
        )
    return pools


def current_solver_threads() -> Optional[int]:
    """Return the solver-thread limit active in the current context."""

    return _SOLVER_THREADS.get()


@functools.lru_cache(maxsize=32)
def _cached_inherited_policy(
    phase: str,
    requested: Optional[int],
    coordination: str,
    status: str,
    limiter_available: bool,
    fallback_reason: Optional[str],
) -> Mapping[str, Any]:
    """Return a read-only-by-convention report template for a no-op scope."""

    return {
        "phase": phase,
        "requested_threads": requested,
        "limiter_available": limiter_available,
        "fallback_reason": fallback_reason,
        "pools_before": (),
        "pools_active": (),
        "pools_after": (),
        "restored": True,
        "coordination": coordination,
        "status": status,
    }


def _inherited_native_thread_policy(
    requested_threads: Optional[int], *, phase: str
) -> Optional[Mapping[str, Any]]:
    """Return an inherited policy when an outer same-thread scope is enough.

    Low-level factorization and solve calls use this before constructing a
    nested context manager.  The owner checks are identical to
    :func:`native_thread_scope`, so copied contexts on another thread cannot
    inherit the originating thread's reader or writer.
    """

    requested = None if requested_threads is None else int(requested_threads)
    if requested is not None and requested <= 0:
        raise ValueError("requested_threads must be positive when provided")
    owner = threading.get_ident()
    if (
        requested is not None
        and _ACTIVE_NATIVE_LIMIT.get() == requested
        and _ACTIVE_NATIVE_LIMIT_OWNER.get() == owner
    ):
        return _cached_inherited_policy(
            str(phase),
            requested,
            "writer_inherited",
            "inherited_limit",
            _HAS_THREADPOOLCTL,
            _THREADPOOLCTL_ERROR,
        )
    if requested is None and _ACTIVE_NATIVE_DEFAULT_OWNER.get() == owner:
        inherited_explicit = _ACTIVE_NATIVE_LIMIT_OWNER.get() == owner
        return _cached_inherited_policy(
            str(phase),
            requested,
            "writer_inherited" if inherited_explicit else "reader_inherited",
            (
                "inherited_explicit_limit"
                if inherited_explicit
                else "inherited_default"
            ),
            _HAS_THREADPOOLCTL,
            _THREADPOOLCTL_ERROR,
        )
    return None


@contextlib.contextmanager
def native_thread_scope(
    requested_threads: Optional[int], *, phase: str
) -> Iterator[Dict[str, Any]]:
    """Temporarily limit BLAS/OpenMP pools and expose the effective policy."""

    requested = None if requested_threads is None else int(requested_threads)
    if requested is not None and requested <= 0:
        raise ValueError("requested_threads must be positive when provided")
    owner = threading.get_ident()
    if (
        requested is not None
        and _ACTIVE_NATIVE_LIMIT.get() == requested
        and _ACTIVE_NATIVE_LIMIT_OWNER.get() == owner
    ):
        # The outer scope already owns the process-global limiter.  This is the
        # hot path for repeated solves through one factorization handle, so do
        # not repeat threadpool discovery for every right-hand side.
        report: Dict[str, Any] = {
            "phase": str(phase),
            "requested_threads": requested,
            "limiter_available": _HAS_THREADPOOLCTL,
            "fallback_reason": _THREADPOOLCTL_ERROR,
            "pools_before": [],
            "pools_active": [],
            "pools_after": [],
            "restored": False,
            "coordination": "writer_inherited",
            "status": "inherited_limit",
        }
        try:
            yield report
        finally:
            report["restored"] = True
        return
    if requested is None and _ACTIVE_NATIVE_DEFAULT_OWNER.get() == owner:
        # A same-thread outer default scope already owns a coordinator reader.
        # Keep that reader as the lifetime guard instead of taking the
        # condition lock for every nested solve.  Context copies retain the
        # originating thread id, so a child thread cannot inherit this path.
        inherited_explicit = _ACTIVE_NATIVE_LIMIT_OWNER.get() == owner
        report = {
            "phase": str(phase),
            "requested_threads": requested,
            "limiter_available": _HAS_THREADPOOLCTL,
            "fallback_reason": _THREADPOOLCTL_ERROR,
            "pools_before": [],
            "pools_active": [],
            "pools_after": [],
            "restored": False,
            "coordination": (
                "writer_inherited" if inherited_explicit else "reader_inherited"
            ),
            "status": (
                "inherited_explicit_limit"
                if inherited_explicit
                else "inherited_default"
            ),
        }
        try:
            yield report
        finally:
            report["restored"] = True
        return
    if requested is None:
        coordinator_token = _NATIVE_SCOPE_COORDINATOR.acquire_reader()
        report: Dict[str, Any] = {
            "phase": str(phase),
            "requested_threads": requested,
            "limiter_available": _HAS_THREADPOOLCTL,
            "fallback_reason": _THREADPOOLCTL_ERROR,
            "pools_before": [],
            "pools_active": [],
            "pools_after": [],
            "restored": False,
            "coordination": coordinator_token[0],
            "status": (
                "inherited_explicit_limit"
                if coordinator_token[0] == "writer_inherited"
                else "unlimited_default"
            ),
        }
        default_owner_token = None
        if coordinator_token[0] == "reader":
            default_owner_token = _ACTIVE_NATIVE_DEFAULT_OWNER.set(owner)
        try:
            yield report
        finally:
            try:
                report["restored"] = True
                if default_owner_token is not None:
                    _ACTIVE_NATIVE_DEFAULT_OWNER.reset(default_owner_token)
            finally:
                _NATIVE_SCOPE_COORDINATOR.release_reader(coordinator_token)
        return

    coordinator_token = _NATIVE_SCOPE_COORDINATOR.acquire_writer()
    report = {
        "phase": str(phase),
        "requested_threads": requested,
        "limiter_available": _HAS_THREADPOOLCTL,
        "fallback_reason": _THREADPOOLCTL_ERROR,
        "pools_before": [],
        "pools_active": [],
        "pools_after": [],
        "restored": False,
        "coordination": coordinator_token[0],
    }
    limit_token = None
    owner_token = None
    try:
        # Snapshot only after exclusive ownership: threadpoolctl reports and
        # mutations must describe one coherent process-global interval.
        report["pools_before"] = _pool_snapshot()
        if (
            _ACTIVE_NATIVE_LIMIT.get() == requested
            and _ACTIVE_NATIVE_LIMIT_OWNER.get() == owner
        ):
            report["status"] = "inherited_limit"
            yield report
        elif not _HAS_THREADPOOLCTL or threadpool_limits is None:
            report["status"] = "backend_unavailable"
            report["fallback_reason"] = (
                _THREADPOOLCTL_ERROR or "threadpoolctl unavailable"
            )
            yield report
        else:
            limit_token = _ACTIVE_NATIVE_LIMIT.set(requested)
            owner_token = _ACTIVE_NATIVE_LIMIT_OWNER.set(owner)
            with threadpool_limits(limits=requested):
                report["status"] = "limited"
                report["pools_active"] = _pool_snapshot()
                yield report
    finally:
        try:
            report["pools_after"] = _pool_snapshot()
        finally:
            report["restored"] = True
            try:
                if owner_token is not None:
                    _ACTIVE_NATIVE_LIMIT_OWNER.reset(owner_token)
                if limit_token is not None:
                    _ACTIVE_NATIVE_LIMIT.reset(limit_token)
            finally:
                _NATIVE_SCOPE_COORDINATOR.release_writer(coordinator_token)


@contextlib.contextmanager
def solver_thread_scope(resource_config: Any) -> Iterator[Dict[str, Any]]:
    """Apply ``ResourceConfig.solver_threads`` for a complete solver call."""

    requested = current_solver_threads() if resource_config is None else resource_config.solver_threads
    token = _SOLVER_THREADS.set(None if requested is None else int(requested))
    try:
        with native_thread_scope(requested, phase="solver") as report:
            yield report
    finally:
        _SOLVER_THREADS.reset(token)


def _resource_config_from_call(func: F, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    bound = inspect.signature(func).bind_partial(*args, **kwargs)
    config = bound.arguments.get("resource_config")
    if config is not None:
        return config
    for name in ("config", "transient_config"):
        container = bound.arguments.get(name)
        if container is not None and getattr(container, "resource_config", None) is not None:
            return container.resource_config
    return None


def resource_threaded(func: F) -> F:
    """Decorate a public solver while preserving its callable signature."""

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any):
        resource_config = _resource_config_from_call(func, args, kwargs)
        with solver_thread_scope(resource_config):
            return func(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


def thread_policy_diagnostics(resource_config: Any = None) -> Dict[str, Any]:
    """Return requested settings and currently detected native pools."""

    return {
        "requested_solver_threads": (
            None if resource_config is None else resource_config.solver_threads
        ),
        "requested_assembly_threads": (
            None if resource_config is None else resource_config.assembly_threads
        ),
        "active_solver_threads": current_solver_threads(),
        "limiter_available": _HAS_THREADPOOLCTL,
        "fallback_reason": _THREADPOOLCTL_ERROR,
        "runtime_pools": _pool_snapshot(),
    }
