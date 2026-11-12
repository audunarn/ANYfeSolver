"""Private portable mutation epochs for qualified shell authority guards.

The observer deliberately covers ordinary Python attribute mutation only.
It has no model, thread, task, or execution-scope state. Qualified callers
retain their exact slow validators and use an epoch merely to prove that the
already-validated attribute surface has not changed between calls.

Direct writes through a module ``__dict__``, direct ``type.__setattr__``
calls, and code/closure surgery are internal interpreter manipulation rather
than supported production mutation surfaces. The exact slow validators stay
authoritative whenever an observed mutation dirties an epoch.
"""

from __future__ import annotations

from abc import ABCMeta
import sys
from threading import RLock
from types import ModuleType
from typing import Any, Callable, Iterable, TypeVar


_MutationCallback = Callable[[bool], None]
_SlowValidator = Callable[[], None]
_Result = TypeVar("_Result")


def _make_attribute_registry() -> tuple[
    Callable[
        [ModuleType, Iterable[str], _MutationCallback],
        tuple[bool, type[ModuleType]],
    ],
    Callable[[type[Any], Iterable[str] | None, _MutationCallback], bool],
    Callable[[type[Any], Iterable[str]], bool],
    Callable[[Callable[[], _Result]], _Result],
    type[ABCMeta],
]:
    """Create closure-owned subscribers and one mutation/validation lock."""

    lock = RLock()
    module_subscribers: dict[
        ModuleType, dict[_MutationCallback, frozenset[str]]
    ] = {}
    type_subscribers: dict[
        type[Any], dict[_MutationCallback, frozenset[str] | None]
    ] = {}
    tracked_module_types: dict[type[ModuleType], type[ModuleType]] = {}
    protected_type_entries: dict[
        type[Any], dict[str, tuple[Any, Callable[..., Any]]]
    ] = {}
    missing = object()

    def synchronized(action: Callable[[], _Result]) -> _Result:
        """Run one epoch transaction under the registry mutation lock."""

        with lock:
            return action()

    def module_callbacks_locked(
        module: ModuleType,
        name: str,
    ) -> tuple[_MutationCallback, ...]:
        return tuple(
            callback
            for callback, names in tuple(
            module_subscribers.get(module, {}).items()
            )
            if name in names
        )

    def type_callbacks_locked(
        owner: type[Any],
        name: str,
    ) -> tuple[_MutationCallback, ...]:
        return tuple(
            callback
            for callback, names in tuple(
                type_subscribers.get(owner, {}).items()
            )
            if names is None or name in names
        )

    class AuthorityEpochMeta(ABCMeta):
        """ABC-compatible metaclass observing ordinary class mutations."""

        def __setattr__(cls, name: str, value: Any) -> None:
            with lock:
                exact_name = str(name)
                callbacks = type_callbacks_locked(cls, exact_name)
                for callback in callbacks:
                    callback(True)
                try:
                    protected = protected_type_entries.get(cls, {}).get(
                        exact_name
                    )
                    if protected is not None:
                        expected, blocked = protected
                        value = (
                            value
                            if expected is not missing and value is expected
                            else blocked
                        )
                    super().__setattr__(name, value)
                finally:
                    for callback in callbacks:
                        callback(False)

        def __delattr__(cls, name: str) -> None:
            with lock:
                exact_name = str(name)
                callbacks = type_callbacks_locked(cls, exact_name)
                for callback in callbacks:
                    callback(True)
                try:
                    protected = protected_type_entries.get(cls, {}).get(
                        exact_name
                    )
                    if protected is None:
                        super().__delattr__(name)
                    else:
                        expected, blocked = protected
                        if expected is missing:
                            super().__delattr__(name)
                        else:
                            super().__setattr__(name, blocked)
                finally:
                    for callback in callbacks:
                        callback(False)

    def tracked_module_type(
        original: type[ModuleType],
    ) -> type[ModuleType]:
        made = tracked_module_types.get(original)
        if made is not None:
            return made

        class AuthorityTrackedModule(original):  # type: ignore[misc, valid-type]
            def __setattr__(self, name: str, value: Any) -> None:
                with lock:
                    callbacks = module_callbacks_locked(self, str(name))
                    for callback in callbacks:
                        callback(True)
                    try:
                        original.__setattr__(self, name, value)
                    finally:
                        for callback in callbacks:
                            callback(False)

            def __delattr__(self, name: str) -> None:
                with lock:
                    callbacks = module_callbacks_locked(self, str(name))
                    for callback in callbacks:
                        callback(True)
                    try:
                        original.__delattr__(self, name)
                    finally:
                        for callback in callbacks:
                            callback(False)

        AuthorityTrackedModule.__name__ = (
            f"_AuthorityTracked{original.__name__}"
        )
        AuthorityTrackedModule.__qualname__ = AuthorityTrackedModule.__name__
        AuthorityTrackedModule.__module__ = __name__
        tracked_module_types[original] = AuthorityTrackedModule
        return AuthorityTrackedModule

    def watch_module(
        module: ModuleType,
        names: Iterable[str],
        callback: _MutationCallback,
    ) -> tuple[bool, type[ModuleType]]:
        exact_names = frozenset(str(name) for name in names) | {"__class__"}
        with lock:
            current_type = type(module)
            if current_type in tracked_module_types.values():
                expected_type = current_type
            else:
                expected_type = tracked_module_type(current_type)
                current_type.__setattr__(module, "__class__", expected_type)
            subscribers = module_subscribers.setdefault(module, {})
            previous = subscribers.get(callback, frozenset())
            merged = previous | exact_names
            changed = merged != previous
            subscribers[callback] = merged
            return changed, expected_type

    def watch_type(
        owner: type[Any],
        names: Iterable[str] | None,
        callback: _MutationCallback,
    ) -> bool:
        if not isinstance(owner, AuthorityEpochMeta):
            raise TypeError(
                f"{owner.__module__}.{owner.__qualname__} is not "
                "authority-epoch tracked"
            )
        exact_names = (
            None if names is None else frozenset(str(name) for name in names)
        )
        with lock:
            subscribers = type_subscribers.setdefault(owner, {})
            if callback not in subscribers:
                subscribers[callback] = exact_names
                return True
            previous = subscribers[callback]
            if previous is None or exact_names is None:
                merged = None
            else:
                merged = previous | exact_names
            changed = merged != previous
            subscribers[callback] = merged
            return changed

    def protect_type_entries(
        owner: type[Any],
        names: Iterable[str],
    ) -> bool:
        if not isinstance(owner, AuthorityEpochMeta):
            raise TypeError(
                f"{owner.__module__}.{owner.__qualname__} is not "
                "authority-epoch tracked"
            )
        with lock:
            protected = protected_type_entries.setdefault(owner, {})
            namespace = type.__getattribute__(owner, "__dict__")
            changed = False
            for raw_name in names:
                name = str(raw_name)
                if name in protected:
                    continue
                expected = namespace.get(name, missing)
                owner_name = type.__getattribute__(owner, "__name__")

                def blocked(
                    *_args: Any,
                    _owner_name: str = owner_name,
                    _name: str = name,
                    **_kwargs: Any,
                ) -> Any:
                    raise ValueError(
                        f"qualified {_owner_name}.{_name} class authority "
                        "was replaced"
                    )

                blocked.__name__ = name
                blocked.__qualname__ = f"{owner_name}.{name}"
                replacement = (
                    property(blocked)
                    if type(expected) is property
                    else blocked
                )
                protected[name] = (expected, replacement)
                changed = True
            return changed

    return (
        watch_module,
        watch_type,
        protect_type_entries,
        synchronized,
        AuthorityEpochMeta,
    )


(
    _WATCH_MODULE,
    _WATCH_TYPE,
    _PROTECT_TYPE_ENTRIES,
    _SYNCHRONIZED,
    AuthorityEpochMeta,
) = _make_attribute_registry()


def _supports_lock_free_epoch_reads() -> bool:
    if sys.implementation.name != "cpython":
        return False
    gil_reader = getattr(sys, "_is_gil_enabled", None)
    return gil_reader is None or bool(gil_reader())


_LOCK_FREE_EPOCH_READS = _supports_lock_free_epoch_reads()


class _AuthorityEpochManager:
    """One closure-backed monotonic dirty generation and exact fallback."""

    __slots__ = (
        "_active_mutations",
        "_generation",
        "_label",
        "_mutation_callback",
        "_watched_modules",
    )

    def __init__(self, label: str) -> None:
        self._label = str(label)
        self._generation = 0
        self._active_mutations = 0
        self._watched_modules: dict[
            ModuleType, tuple[type[ModuleType], str]
        ] = {}

        def changed(starting: bool) -> None:
            # Every caller holds the closure-owned registry lock.
            self._generation += 1
            if starting:
                self._active_mutations += 1
            else:
                self._active_mutations -= 1
                if self._active_mutations < 0:
                    raise RuntimeError(
                        f"{self._label} mutation epoch underflow"
                    )

        self._mutation_callback = changed

    def watch_module(
        self,
        module: ModuleType,
        names: Iterable[str],
    ) -> None:
        def install() -> None:
            changed, expected_type = _WATCH_MODULE(
                module, names, self._mutation_callback
            )
            prior = self._watched_modules.get(module)
            if prior is not None and prior[0] is not expected_type:
                raise RuntimeError(
                    f"{self._label} module tracking identity changed"
                )
            self._watched_modules[module] = (
                expected_type,
                str(module.__name__),
            )
            if changed:
                self._generation += 1

        _SYNCHRONIZED(install)

    def watch_type(
        self,
        owner: type[Any],
        names: Iterable[str] | None = None,
    ) -> None:
        def install() -> None:
            if _WATCH_TYPE(owner, names, self._mutation_callback):
                self._generation += 1

        _SYNCHRONIZED(install)

    def protect_type_entries(
        self,
        owner: type[Any],
        names: Iterable[str],
    ) -> None:
        def install() -> None:
            if _PROTECT_TYPE_ENTRIES(owner, names):
                self._generation += 1

        _SYNCHRONIZED(install)

    def _require_tracked_module_types(self) -> None:
        changed = [
            expected_name
            for module, (expected_type, expected_name) in (
                self._watched_modules.items()
            )
            if type(module) is not expected_type
        ]
        if changed:
            raise ValueError(
                f"{self._label} authority module tracking changed: "
                + ", ".join(sorted(changed))
            )

    def _steady(self, accepted_generation: int) -> bool:
        if not _LOCK_FREE_EPOCH_READS:
            return False
        if (
            self._active_mutations != 0
            or self._generation != accepted_generation
        ):
            return False
        return True

    def capture_generation(self) -> int:
        """Capture a call-local generation after an exact preflight."""

        if _LOCK_FREE_EPOCH_READS:
            if self._active_mutations:
                raise ValueError(
                    f"{self._label} authority mutation is incomplete"
                )
            return int(self._generation)

        def transaction() -> int:
            if self._active_mutations:
                raise ValueError(
                    f"{self._label} authority mutation is incomplete"
                )
            return int(self._generation)

        return _SYNCHRONIZED(transaction)

    def require_generation(self, expected_generation: int) -> None:
        """Reject any mutation during a guarded call, even after undo."""

        if _LOCK_FREE_EPOCH_READS:
            if (
                self._active_mutations
                or self._generation != expected_generation
            ):
                raise ValueError(
                    f"{self._label} authority changed during guarded call"
                )
            return

        def transaction() -> None:
            if (
                self._active_mutations
                or self._generation != expected_generation
            ):
                raise ValueError(
                    f"{self._label} authority changed during guarded call"
                )

        _SYNCHRONIZED(transaction)

    def bind(self, slow_validator: _SlowValidator) -> _SlowValidator:
        """Capture a nonreplaceable reader and exact slow fallback."""

        if not callable(slow_validator):
            raise TypeError("slow authority validator must be callable")
        manager = self
        accepted_generation = -1

        def require() -> None:
            nonlocal accepted_generation
            if manager._steady(accepted_generation):
                return

            def transaction() -> None:
                nonlocal accepted_generation
                if manager._active_mutations:
                    raise ValueError(
                        f"{manager._label} authority mutation is incomplete"
                    )
                manager._require_tracked_module_types()
                before = manager._generation
                if before == accepted_generation:
                    return
                slow_validator()
                manager._require_tracked_module_types()
                after = manager._generation
                if after != before:
                    raise ValueError(
                        f"{manager._label} authority changed during exact "
                        "validation"
                    )
                accepted_generation = after

            _SYNCHRONIZED(transaction)

        return require

    def bind_context(
        self,
        slow_validator: Callable[..., None],
    ) -> Callable[..., None]:
        """Bind an exact validator with one required ``context`` keyword."""

        if not callable(slow_validator):
            raise TypeError("slow authority validator must be callable")
        manager = self
        accepted_generation = -1

        def require(*, context: str) -> None:
            nonlocal accepted_generation
            if manager._steady(accepted_generation):
                return

            def transaction() -> None:
                nonlocal accepted_generation
                if manager._active_mutations:
                    raise ValueError(
                        f"{manager._label} authority mutation is incomplete"
                    )
                manager._require_tracked_module_types()
                before = manager._generation
                if before == accepted_generation:
                    return
                slow_validator(context=context)
                manager._require_tracked_module_types()
                after = manager._generation
                if after != before:
                    raise ValueError(
                        f"{manager._label} authority changed during exact "
                        "validation"
                    )
                accepted_generation = after

            _SYNCHRONIZED(transaction)

        return require

    def bind_argument(
        self,
        slow_validator: Callable[[Any], _Result],
    ) -> Callable[[Any], _Result | None]:
        """Bind a dirty exact fallback plus per-call steady validation.

        The slow validator receives the current call argument on a dirty
        generation. A steady call returns ``None`` so the caller can perform
        its mandatory bounded instance checks without repeating global work.
        """

        if not callable(slow_validator):
            raise TypeError("slow authority validator must be callable")
        manager = self
        accepted_generation = -1

        def require(argument: Any) -> _Result | None:
            nonlocal accepted_generation
            if manager._steady(accepted_generation):
                return None
            result: _Result | None = None

            def transaction() -> None:
                nonlocal accepted_generation, result
                if manager._active_mutations:
                    raise ValueError(
                        f"{manager._label} authority mutation is incomplete"
                    )
                manager._require_tracked_module_types()
                before = manager._generation
                if before == accepted_generation:
                    return
                result = slow_validator(argument)
                manager._require_tracked_module_types()
                after = manager._generation
                if after != before:
                    raise ValueError(
                        f"{manager._label} authority changed during exact "
                        "validation"
                    )
                accepted_generation = after

            _SYNCHRONIZED(transaction)
            return result

        return require


def make_authority_epoch_manager(label: str) -> _AuthorityEpochManager:
    """Return one private manager for an independently validated domain."""

    return _AuthorityEpochManager(label)


__all__: tuple[str, ...] = ()
