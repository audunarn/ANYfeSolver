"""Persistent, solver-owned storage for nonlinear constitutive state.

The public nonlinear solver contract uses ``{element_id: state}`` mappings.  That
is a useful interchange format for restart, recovery, and result snapshots, but
packing and allocating those dictionaries in every Newton iteration is costly.
This module provides a deliberately small internal alternative:

* :class:`ShellStateLayout` is immutable geometry for one compatible shell batch;
* :class:`ShellStateBatch` owns contiguous committed and trial buffers;
* :class:`NonlinearStateStore` coordinates one or more batches for one solve.

The storage objects are solver-owned.  They must not be attached to cached
assembly plans: a plan can be shared by sequential analyses, while constitutive
history belongs to exactly one analysis.  Public mappings are always materialized
as owned copies so restart/recovery consumers cannot mutate the active solve.
Unsupported state dictionaries remain available through an explicit, diagnosed
legacy fallback.
"""

from __future__ import annotations

import copy
import threading
import time
from contextvars import ContextVar
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Optional

import numpy as np

from ._native_rotation_state import (
    NativeElementRotationView,
    NativeRotationStateStore,
    NativeRotationTrialToken,
    NativeRotationValidationError,
    create_native_rotation_state_store,
    validate_proper_rotation_matrices,
)
from ._native_material_protocol import (
    PROTOCOL as _NATIVE_MATERIAL_PROTOCOL,
    NativeMaterialContext,
    NativeMaterialValidator,
)


_CORE_FIELDS = ("plastic_strain", "alpha", "layer_strain")
_FIELD_INDEX = {name: index for index, name in enumerate(_CORE_FIELDS)}
_Q4_ALGORITHMIC_ORIGIN_KEY = "qualified_q4_algorithmic_origin"
_Q4_ALGORITHMIC_ORIGIN_SCHEMA_ID = (
    "E4_PL_Q4_ACCEPTED_DISCRETE_RETURN_MAP_ORIGIN_V1"
)
_Q4_ALGORITHMIC_ORIGIN_KIND = "LAYERED_DISCRETE_RETURN_MAP_PARENT_STATE"
_INITIAL_FIELD_KEYS = (
    "initial_membrane_stress",
    "initial_bending_stress",
    "initial_membrane_prestrain",
    "initial_curvature_prestrain",
)
_KNOWN_PROVENANCE_KEYS = (
    "provenance",
    "initial_field_provenance",
    "state_provenance",
    "material_provenance",
)


class NonlinearStateError(RuntimeError):
    """Base class for persistent nonlinear-state contract errors."""


class StateTransactionError(NonlinearStateError):
    """Raised when trial-state transaction ordering is invalid."""


class StaleStateTokenError(StateTransactionError):
    """Raised when a trial token no longer names the active generation."""


class ImmutableStateSidecarError(NonlinearStateError):
    """Raised when an update attempts to alter an immutable initial field."""


class PersistentStateEligibilityError(NonlinearStateError):
    """Raised when a caller requests an array path for a fallback state."""


class StateMaterializationPolicy(str, Enum):
    """Reason an owned public dictionary is being requested."""

    EXPLICIT = "explicit"
    FINAL_RESULT = "final_result"
    SAVED_STATE = "saved_state"
    RESTART = "restart"
    RECOVERY = "recovery"


@dataclass(frozen=True, slots=True)
class ShellStateLayout:
    """Immutable array layout for a homogeneous shell batch.

    Parameters match the persistent shell assembly plan vocabulary.  State is
    stored at ``n_gp * num_layers`` material points per element, with three
    engineering in-plane components for plastic and total layer strain.
    """

    element_ids: tuple[int, ...]
    n_gp: int
    num_layers: int
    _element_index: Mapping[int, int] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        element_ids = tuple(int(value) for value in self.element_ids)
        if not element_ids:
            raise ValueError("ShellStateLayout requires at least one element")
        if len(set(element_ids)) != len(element_ids):
            raise ValueError("ShellStateLayout element_ids must be unique")
        if int(self.n_gp) <= 0:
            raise ValueError("ShellStateLayout n_gp must be positive")
        if int(self.num_layers) <= 0:
            raise ValueError("ShellStateLayout num_layers must be positive")
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "n_gp", int(self.n_gp))
        object.__setattr__(self, "num_layers", int(self.num_layers))
        object.__setattr__(
            self,
            "_element_index",
            MappingProxyType(
                {element_id: index for index, element_id in enumerate(element_ids)}
            ),
        )

    @classmethod
    def from_dimensions(
        cls,
        element_ids: Sequence[int],
        n_gp: int,
        num_layers: int,
    ) -> "ShellStateLayout":
        return cls(tuple(int(value) for value in element_ids), int(n_gp), int(num_layers))

    @property
    def n_elements(self) -> int:
        return len(self.element_ids)

    @property
    def points_per_element(self) -> int:
        return int(self.n_gp * self.num_layers)

    @property
    def state_point_count(self) -> int:
        return int(self.n_elements * self.points_per_element)

    @property
    def buffer_value_count(self) -> int:
        # plastic strain (3), alpha (1), and total layer strain (3).
        return int(self.state_point_count * 7)

    def index(self, element_id: int) -> int:
        try:
            return int(self._element_index[int(element_id)])
        except KeyError as exc:
            raise KeyError(f"Element {int(element_id)} is not in this shell layout") from exc

    def compatible_with(self, other: "ShellStateLayout") -> bool:
        return bool(
            isinstance(other, ShellStateLayout)
            and self.element_ids == other.element_ids
            and self.n_gp == other.n_gp
            and self.num_layers == other.num_layers
        )


@dataclass(frozen=True, slots=True)
class StateTrialToken:
    """Opaque generation-checked capability for one active trial update."""

    generation: int
    serial: int
    _owner: object = field(repr=False, compare=False, hash=False)


@dataclass(frozen=True, slots=True)
class _NativeElementStateBinding:
    """Frozen model ownership needed to cross-check redundant S3 state."""

    node_ids: tuple[int, ...]
    dof_mapping: np.ndarray = field(repr=False, compare=False, hash=False)
    reference_directors: np.ndarray = field(repr=False, compare=False, hash=False)
    state_consistency_required: bool = False
    material_validator: Optional[NativeMaterialValidator] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.material_validator is not None and type(self.material_validator) is not NativeMaterialValidator:
            raise NativeRotationValidationError("Exact native material validator required")
        nodes = tuple(int(value) for value in self.node_ids)
        if not nodes or len(set(nodes)) != len(nodes):
            raise NativeRotationValidationError(
                "native element binding requires unique nonempty node_ids"
            )
        dofs = np.asarray(self.dof_mapping, dtype=np.intp)
        if dofs.shape != (6 * len(nodes),) or np.any(dofs < 0):
            raise NativeRotationValidationError(
                "native element binding requires six valid DOFs per node"
            )
        directors = np.asarray(self.reference_directors, dtype=np.float64)
        if directors.shape != (len(nodes), 3) or not np.all(np.isfinite(directors)):
            raise NativeRotationValidationError(
                "native element binding has incompatible reference directors"
            )
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(
            self,
            "state_consistency_required",
            bool(self.state_consistency_required),
        )
        object.__setattr__(
            self,
            "dof_mapping",
            np.frombuffer(
                np.ascontiguousarray(dofs).tobytes(order="C"),
                dtype=np.intp,
            ).reshape(dofs.shape),
        )
        object.__setattr__(
            self,
            "reference_directors",
            np.frombuffer(
                np.ascontiguousarray(directors).tobytes(order="C"),
                dtype=np.float64,
            ).reshape(directors.shape),
        )


_NATIVE_ELEMENT_BINDINGS_ATTRIBUTE = "_anyfesolver_native_element_state_bindings_v1"
_NATIVE_DIRECTOR_CONSISTENCY_TOLERANCE = 1.0e-12

# A synchronous solve owns only stores explicitly activated in its context.
# Nested/concurrent solves do not share a mutable global cleanup registry.
_SOLVE_STATE_CLEANUP: ContextVar[Optional[list[Any]]] = ContextVar(
    "anyfesolver_nonlinear_state_cleanup", default=None
)


def _register_nonlinear_state_cleanup(store: Any) -> Any:
    if not isinstance(store, NonlinearStateStore):
        raise TypeError("Only a solver-owned nonlinear state store can be registered")
    owned = _SOLVE_STATE_CLEANUP.get()
    if owned is not None and not any(item is store for item in owned):
        owned.append(store)
    return store


def _run_with_nonlinear_state_cleanup(operation: Any, *args: Any, **kwargs: Any) -> Any:
    """Discard this solve's remaining trials on return, failure or cancellation.

    Committed material/kinematic generations are never advanced or rolled back.
    Cleanup failures are reported after attempting every owned store; they must
    not silently turn an interrupted solve into a clean checkpoint.
    """
    owned: list[Any] = []
    token = _SOLVE_STATE_CLEANUP.set(owned)
    try:
        return operation(*args, **kwargs)
    finally:
        _SOLVE_STATE_CLEANUP.reset(token)
        failures = []
        for store in reversed(owned):
            try:
                if store.has_active_trial:
                    store.discard_trial(store.active_trial_token())
                rotations = store.native_rotation_store
                if store.has_active_trial or (rotations is not None and rotations.has_active_trial):
                    raise StateTransactionError("Nonlinear solve left an inconsistent active trial")
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise StateTransactionError(
                f"Nonlinear solve cleanup failed for {len(failures)} owned state store(s)"
            ) from failures[0]


@dataclass(frozen=True, slots=True)
class ShellStateArrays:
    """Read-only array views of one committed shell-state buffer."""

    plastic_strain: np.ndarray
    alpha: np.ndarray
    layer_strain: np.ndarray


@dataclass(frozen=True, slots=True)
class _FrozenList:
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _FrozenTuple:
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _FrozenSet:
    values: frozenset[Any]


@dataclass(frozen=True, slots=True)
class _FrozenFrozenSet:
    values: frozenset[Any]


def _owned_copy(value: Any) -> Any:
    """Copy an arbitrary legacy state without exposing caller-owned arrays."""

    try:
        return copy.deepcopy(value)
    except Exception as exc:  # pragma: no cover - exotic third-party states
        raise TypeError("Nonlinear state must support owned deep-copy semantics") from exc


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True, order="C")
        result.setflags(write=False)
        return result
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(tuple(_freeze_value(item) for item in value))
    if isinstance(value, tuple):
        return _FrozenTuple(tuple(_freeze_value(item) for item in value))
    if isinstance(value, set):
        return _FrozenSet(frozenset(_freeze_value(item) for item in value))
    if isinstance(value, frozenset):
        return _FrozenFrozenSet(frozenset(_freeze_value(item) for item in value))
    return _owned_copy(value)


def _thaw_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_thaw_value(item) for item in value.values]
    if isinstance(value, _FrozenTuple):
        return tuple(_thaw_value(item) for item in value.values)
    if isinstance(value, _FrozenSet):
        return {_thaw_value(item) for item in value.values}
    if isinstance(value, _FrozenFrozenSet):
        return frozenset(_thaw_value(item) for item in value.values)
    return _owned_copy(value)


def _state_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return bool(
            set(left) == set(right)
            and all(_state_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return bool(
            len(left) == len(right)
            and all(_state_values_equal(a, b) for a, b in zip(left, right))
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _is_sidecar_key(key: str) -> bool:
    return bool(
        key in _INITIAL_FIELD_KEYS
        or key in _KNOWN_PROVENANCE_KEYS
        or key.endswith("_provenance")
    )


def _fallback_reason(state: Any, layout: ShellStateLayout) -> Optional[str]:
    if state is None:
        return None
    if not isinstance(state, Mapping):
        return "state_not_mapping"
    unsupported = sorted(
        str(key)
        for key in state
        if str(key) not in _CORE_FIELDS
        and str(key) != _Q4_ALGORITHMIC_ORIGIN_KEY
        and not _is_sidecar_key(str(key))
    )
    if unsupported:
        return "unsupported_state_keys:" + ",".join(unsupported)
    missing = [key for key in ("plastic_strain", "alpha") if key not in state]
    if missing:
        return "missing_required_core_fields:" + ",".join(missing)
    points = layout.points_per_element
    expected = {
        "plastic_strain": (points, 3),
        "alpha": (points,),
        "layer_strain": (points, 3),
    }
    for key, shape in expected.items():
        if key not in state:
            continue
        try:
            actual = np.asarray(state[key], dtype=float).shape
        except (TypeError, ValueError):
            return f"invalid_{key}_values"
        if actual != shape:
            return f"invalid_{key}_shape:{actual}"
    if _Q4_ALGORITHMIC_ORIGIN_KEY in state:
        origin = state[_Q4_ALGORITHMIC_ORIGIN_KEY]
        if not isinstance(origin, Mapping):
            return "invalid_q4_algorithmic_origin_mapping"
        if set(origin) != {
            "schema_id",
            "kind",
            "num_layers",
            "parent_plastic_strain",
            "parent_alpha",
        }:
            return "invalid_q4_algorithmic_origin_schema"
        if (
            origin.get("schema_id") != _Q4_ALGORITHMIC_ORIGIN_SCHEMA_ID
            or origin.get("kind") != _Q4_ALGORITHMIC_ORIGIN_KIND
            or origin.get("num_layers") != layout.num_layers
        ):
            return "incompatible_q4_algorithmic_origin_identity"
        try:
            parent_plastic = np.asarray(
                origin["parent_plastic_strain"], dtype=float
            )
            parent_alpha = np.asarray(origin["parent_alpha"], dtype=float)
        except (TypeError, ValueError):
            return "invalid_q4_algorithmic_origin_values"
        if (
            parent_plastic.shape != (points, 3)
            or parent_alpha.shape != (points,)
            or not np.all(np.isfinite(parent_plastic))
            or not np.all(np.isfinite(parent_alpha))
        ):
            return "invalid_q4_algorithmic_origin_shape_or_values"
    return None


def _extract_sidecar(state: Any) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        return MappingProxyType({})
    return MappingProxyType(
        {
            str(key): _freeze_value(value)
            for key, value in state.items()
            if _is_sidecar_key(str(key))
        }
    )


def _readonly_view(array: np.ndarray) -> np.ndarray:
    result = array.view()
    result.setflags(write=False)
    return result


class ShellStateBatch(Mapping[int, Any]):
    """Contiguous committed/trial history for one compatible shell batch.

    ``begin_trial`` does not copy committed values.  Full-batch kernel updates
    overwrite the inactive buffer and ``commit`` swaps two pointers.  Partial
    updates copy only missing fields/elements at commit.  ``discard_trial`` is
    always O(1) with respect to state size.
    """

    def __init__(
        self,
        layout: ShellStateLayout,
        committed_states: Optional[Mapping[int, Any]] = None,
    ) -> None:
        if not isinstance(layout, ShellStateLayout):
            raise TypeError("layout must be a ShellStateLayout")
        self.layout = layout
        self._owner = object()
        self._lock = threading.RLock()
        self._committed_buffer = np.zeros(layout.buffer_value_count, dtype=np.float64)
        self._trial_buffer = np.empty_like(self._committed_buffer)
        self._packed_mask = np.ones(layout.n_elements, dtype=bool)
        self._deleted_mask = np.zeros(layout.n_elements, dtype=bool)
        self._present_committed = np.zeros(layout.n_elements, dtype=bool)
        self._present_trial_stamps = np.zeros(layout.n_elements, dtype=np.int64)
        self._sidecars: list[Mapping[str, Any]] = [
            MappingProxyType({}) for _ in range(layout.n_elements)
        ]
        self._fallback_committed: Dict[int, Any] = {}
        self._fallback_reasons: Dict[int, str] = {}
        self._trial_legacy: Dict[int, tuple[Any, str]] = {}
        # Allocated lazily only for path-dependent qualified Q4 batches.  The
        # two parent arrays are an exact replay descriptor for the accepted
        # discrete return-map tangent; they are transactionally committed with
        # the ordinary plastic/alpha/layer core without forcing dictionary
        # fallback or inflating unrelated shell-state buffers.
        self._q4_origin_committed_plastic: Optional[np.ndarray] = None
        self._q4_origin_trial_plastic: Optional[np.ndarray] = None
        self._q4_origin_committed_alpha: Optional[np.ndarray] = None
        self._q4_origin_trial_alpha: Optional[np.ndarray] = None
        self._q4_origin_present_committed = np.zeros(
            layout.n_elements, dtype=bool
        )
        self._q4_origin_trial_stamps = np.zeros(
            layout.n_elements, dtype=np.int64
        )
        # Generation stamps avoid clearing an O(elements) boolean mask at begin.
        self._write_stamps = np.zeros((len(_CORE_FIELDS), layout.n_elements), dtype=np.int64)
        self._generation = 0
        self._serial = 0
        self._active_token: Optional[StateTrialToken] = None
        self._metrics: Dict[str, Any] = {
            "state_pack_seconds": 0.0,
            "state_trial_update_seconds": 0.0,
            "state_commit_seconds": 0.0,
            "state_discard_seconds": 0.0,
            "state_materialization_seconds": 0.0,
            "state_materialization_count": 0,
            "state_trial_begin_count": 0,
            "state_trial_update_count": 0,
            "state_commit_count": 0,
            "state_discard_count": 0,
            "state_swap_commit_count": 0,
            "state_bounded_copy_commit_count": 0,
            "state_commit_copied_element_fields": 0,
            "state_trial_updated_point_count": 0,
            "stale_token_error_count": 0,
            "materialization_reasons": {},
        }
        self._pack(committed_states or {})

    @classmethod
    def from_mapping(
        cls,
        layout: ShellStateLayout,
        committed_states: Optional[Mapping[int, Any]] = None,
    ) -> "ShellStateBatch":
        return cls(layout, committed_states)

    @classmethod
    def pack(
        cls,
        layout: ShellStateLayout,
        committed_states: Optional[Mapping[int, Any]] = None,
    ) -> "ShellStateBatch":
        """Named constructor used by restart/import boundaries."""

        return cls(layout, committed_states)

    @property
    def generation(self) -> int:
        return int(self._generation)

    @property
    def has_active_trial(self) -> bool:
        return self._active_token is not None

    @property
    def all_packed(self) -> bool:
        return bool(np.all(self._packed_mask))

    @property
    def has_initial_fields(self) -> bool:
        return any(
            any(key in sidecar for key in _INITIAL_FIELD_KEYS)
            for sidecar in self._sidecars
        )

    @property
    def fallback_element_ids(self) -> tuple[int, ...]:
        return tuple(
            element_id
            for index, element_id in enumerate(self.layout.element_ids)
            if not bool(self._packed_mask[index])
        )

    @property
    def deleted_element_ids(self) -> tuple[int, ...]:
        return tuple(
            element_id
            for index, element_id in enumerate(self.layout.element_ids)
            if bool(self._deleted_mask[index])
        )

    @property
    def committed_buffer(self) -> np.ndarray:
        """Read-only view used for contiguity/memory diagnostics."""

        return _readonly_view(self._committed_buffer)

    @property
    def trial_buffer(self) -> np.ndarray:
        """Read-only view of the inactive/trial allocation."""

        return _readonly_view(self._trial_buffer)

    def _views(self, buffer: np.ndarray) -> ShellStateArrays:
        points = self.layout.state_point_count
        n_elements = self.layout.n_elements
        points_per_element = self.layout.points_per_element
        plastic_stop = points * 3
        alpha_stop = plastic_stop + points
        plastic = buffer[:plastic_stop].reshape(n_elements, points_per_element, 3)
        alpha = buffer[plastic_stop:alpha_stop].reshape(n_elements, points_per_element)
        layer = buffer[alpha_stop:].reshape(n_elements, points_per_element, 3)
        return ShellStateArrays(plastic, alpha, layer)

    def _ensure_q4_algorithmic_origin_buffers(self) -> None:
        if self._q4_origin_committed_plastic is not None:
            return
        shape_plastic = (
            self.layout.n_elements,
            self.layout.points_per_element,
            3,
        )
        shape_alpha = (
            self.layout.n_elements,
            self.layout.points_per_element,
        )
        self._q4_origin_committed_plastic = np.zeros(
            shape_plastic, dtype=np.float64
        )
        self._q4_origin_trial_plastic = np.empty(
            shape_plastic, dtype=np.float64
        )
        self._q4_origin_committed_alpha = np.zeros(
            shape_alpha, dtype=np.float64
        )
        self._q4_origin_trial_alpha = np.empty(shape_alpha, dtype=np.float64)

    def committed_arrays(self) -> ShellStateArrays:
        """Return read-only, zero-copy views for constitutive kernels."""

        views = self._views(self._committed_buffer)
        return ShellStateArrays(
            _readonly_view(views.plastic_strain),
            _readonly_view(views.alpha),
            _readonly_view(views.layer_strain),
        )

    def _pack(self, committed_states: Mapping[int, Any]) -> None:
        start = time.perf_counter()
        views = self._views(self._committed_buffer)
        points = self.layout.points_per_element
        for index, element_id in enumerate(self.layout.element_ids):
            present = element_id in committed_states
            self._present_committed[index] = present
            state = committed_states.get(element_id)
            self._sidecars[index] = _extract_sidecar(state)
            reason = (
                "state_is_explicit_none"
                if present and state is None
                else _fallback_reason(state, self.layout)
            )
            if reason is not None:
                self._packed_mask[index] = False
                self._fallback_committed[element_id] = _owned_copy(state)
                self._fallback_reasons[element_id] = reason
                continue
            if state is None:
                continue
            if "plastic_strain" in state:
                views.plastic_strain[index] = np.asarray(
                    state["plastic_strain"], dtype=float
                ).reshape(points, 3)
            if "alpha" in state:
                views.alpha[index] = np.asarray(state["alpha"], dtype=float).reshape(points)
            if "layer_strain" in state:
                views.layer_strain[index] = np.asarray(
                    state["layer_strain"], dtype=float
                ).reshape(points, 3)
            if _Q4_ALGORITHMIC_ORIGIN_KEY in state:
                origin = state[_Q4_ALGORITHMIC_ORIGIN_KEY]
                assert isinstance(origin, Mapping)  # validated above
                self._ensure_q4_algorithmic_origin_buffers()
                assert self._q4_origin_committed_plastic is not None
                assert self._q4_origin_committed_alpha is not None
                self._q4_origin_committed_plastic[index] = np.asarray(
                    origin["parent_plastic_strain"], dtype=float
                ).reshape(points, 3)
                self._q4_origin_committed_alpha[index] = np.asarray(
                    origin["parent_alpha"], dtype=float
                ).reshape(points)
                self._q4_origin_present_committed[index] = True
        self._metrics["state_pack_seconds"] += time.perf_counter() - start

    def compatible_with(self, layout: ShellStateLayout) -> bool:
        return self.layout.compatible_with(layout)

    def persistent_eligibility(
        self,
        layout: Optional[ShellStateLayout] = None,
    ) -> tuple[bool, Optional[str]]:
        if layout is not None and not self.compatible_with(layout):
            return False, "shell_state_layout_mismatch"
        if not self.all_packed:
            return False, "legacy_dictionary_fallback"
        if self.has_initial_fields:
            return False, "immutable_initial_field_requires_exact_override"
        return True, None

    def begin_trial(self) -> StateTrialToken:
        with self._lock:
            if self._active_token is not None:
                raise StateTransactionError("A shell-state trial is already active")
            self._serial += 1
            token = StateTrialToken(self._generation, self._serial, self._owner)
            self._active_token = token
            self._trial_legacy = {}
            self._metrics["state_trial_begin_count"] += 1
            return token

    def begin(self) -> StateTrialToken:
        """Short transaction alias for :meth:`begin_trial`."""

        return self.begin_trial()

    def _require_active(self, token: StateTrialToken) -> None:
        valid = bool(
            isinstance(token, StateTrialToken)
            and token._owner is self._owner
            and token.generation == self._generation
            and self._active_token is token
        )
        if not valid:
            self._metrics["stale_token_error_count"] += 1
            raise StaleStateTokenError(
                "Trial token is stale, belongs to another store, or is not active"
            )

    def validate_trial_token(self, token: StateTrialToken) -> None:
        """Validate a token before a caller starts an expensive kernel."""

        with self._lock:
            self._require_active(token)

    def _indices(self, element_ids: Optional[Sequence[int]]) -> np.ndarray:
        if element_ids is None:
            return np.arange(self.layout.n_elements, dtype=np.intp)
        return np.asarray(
            [self.layout.index(int(element_id)) for element_id in element_ids],
            dtype=np.intp,
        )

    def _coerce_update(
        self,
        field_name: str,
        values: Any,
        count: int,
    ) -> np.ndarray:
        points = self.layout.points_per_element
        tail = (points,) if field_name == "alpha" else (points, 3)
        expected = (count, *tail)
        array = np.asarray(values, dtype=float)
        if count == 1 and array.shape == tail:
            array = array.reshape(expected)
        elif array.size == int(np.prod(expected, dtype=np.int64)):
            array = array.reshape(expected)
        if array.shape != expected:
            raise ValueError(
                f"{field_name} trial update must have shape {expected}; got {array.shape}"
            )
        return array

    def update_trial(
        self,
        token: StateTrialToken,
        *,
        plastic_strain: Any = None,
        alpha: Any = None,
        layer_strain: Any = None,
        element_ids: Optional[Sequence[int]] = None,
    ) -> int:
        """Write packed trial fields and return the number of active elements.

        Deleted elements are deliberately skipped.  A later commit copies their
        qualified committed rows into the trial allocation before swapping.
        """

        with self._lock:
            self._require_active(token)
            supplied = {
                "plastic_strain": plastic_strain,
                "alpha": alpha,
                "layer_strain": layer_strain,
            }
            supplied = {
                name: value for name, value in supplied.items() if value is not None
            }
            if not supplied:
                raise ValueError("At least one shell trial field must be supplied")
            indices = self._indices(element_ids)
            if np.any(~self._packed_mask[indices]):
                raise PersistentStateEligibilityError(
                    "Array trial updates cannot include legacy dictionary fallback elements"
                )
            active = indices[~self._deleted_mask[indices]]
            start = time.perf_counter()
            trial_views = self._views(self._trial_buffer)
            for field_name, values in supplied.items():
                array = self._coerce_update(field_name, values, int(indices.size))
                if active.size:
                    active_in_input = ~self._deleted_mask[indices]
                    getattr(trial_views, field_name)[active] = array[active_in_input]
                    self._write_stamps[_FIELD_INDEX[field_name], active] = self._serial
            self._present_trial_stamps[active] = self._serial
            elapsed = time.perf_counter() - start
            self._metrics["state_trial_update_seconds"] += elapsed
            self._metrics["state_trial_update_count"] += 1
            self._metrics["state_trial_updated_point_count"] += int(
                active.size * self.layout.points_per_element
            )
            return int(active.size)

    def update_q4_algorithmic_origin_trial(
        self,
        token: StateTrialToken,
        *,
        parent_plastic_strain: Any,
        parent_alpha: Any,
        element_ids: Sequence[int],
    ) -> int:
        """Atomically retain the parent history of accepted Q4 trial updates."""

        with self._lock:
            self._require_active(token)
            indices = self._indices(element_ids)
            if np.any(~self._packed_mask[indices]):
                raise PersistentStateEligibilityError(
                    "Q4 algorithmic origins require packed shell-state rows"
                )
            plastic = self._coerce_update(
                "plastic_strain",
                parent_plastic_strain,
                int(indices.size),
            )
            alpha = self._coerce_update(
                "alpha",
                parent_alpha,
                int(indices.size),
            )
            self._ensure_q4_algorithmic_origin_buffers()
            assert self._q4_origin_trial_plastic is not None
            assert self._q4_origin_trial_alpha is not None
            active_in_input = ~self._deleted_mask[indices]
            active = indices[active_in_input]
            if active.size:
                self._q4_origin_trial_plastic[active] = plastic[active_in_input]
                self._q4_origin_trial_alpha[active] = alpha[active_in_input]
                self._q4_origin_trial_stamps[active] = self._serial
            return int(active.size)

    def _validate_or_merge_sidecar(self, index: int, state: Any) -> Any:
        if not isinstance(state, Mapping):
            if self._sidecars[index]:
                raise ImmutableStateSidecarError(
                    "A state carrying immutable initial fields must remain a mapping"
                )
            return _owned_copy(state)
        result = _owned_copy(dict(state))
        frozen = self._sidecars[index]
        for key in result:
            if _is_sidecar_key(str(key)) and str(key) not in frozen:
                raise ImmutableStateSidecarError(
                    f"Trial update cannot introduce immutable sidecar {key!r}"
                )
        for key, frozen_value in frozen.items():
            expected = _thaw_value(frozen_value)
            if key in result and not _state_values_equal(result[key], expected):
                raise ImmutableStateSidecarError(
                    f"Trial update cannot modify immutable sidecar {key!r}"
                )
            result[key] = expected
        return result

    def set_trial_state(
        self,
        token: StateTrialToken,
        element_id: int,
        state: Any,
    ) -> None:
        """Route one scalar/reference state through packed or dictionary storage."""

        with self._lock:
            self._require_active(token)
            index = self.layout.index(int(element_id))
            if self._deleted_mask[index]:
                return
            owned_state = self._validate_or_merge_sidecar(index, state)
            if not self._packed_mask[index]:
                self._trial_legacy[int(element_id)] = (
                    owned_state,
                    self._fallback_reasons.get(
                        int(element_id), "legacy_dictionary_fallback"
                    ),
                )
                self._present_trial_stamps[index] = self._serial
                return
            reason = _fallback_reason(owned_state, self.layout)
            if reason is not None:
                self._trial_legacy[int(element_id)] = (owned_state, reason)
                self._present_trial_stamps[index] = self._serial
                return
            updates = {
                key: owned_state[key]
                for key in _CORE_FIELDS
                if isinstance(owned_state, Mapping) and key in owned_state
            }
            if updates:
                self.update_trial(token, element_ids=(int(element_id),), **updates)
            if (
                isinstance(owned_state, Mapping)
                and _Q4_ALGORITHMIC_ORIGIN_KEY in owned_state
            ):
                origin = owned_state[_Q4_ALGORITHMIC_ORIGIN_KEY]
                # _fallback_reason already established the exact closed shape
                # before this packed branch was selected.
                assert isinstance(origin, Mapping)
                self.update_q4_algorithmic_origin_trial(
                    token,
                    parent_plastic_strain=origin[
                        "parent_plastic_strain"
                    ],
                    parent_alpha=origin["parent_alpha"],
                    element_ids=(int(element_id),),
                )

    def freeze_deleted(self, element_ids: Sequence[int]) -> None:
        """Permanently freeze the qualified committed state of deleted elements."""

        with self._lock:
            for element_id in element_ids:
                self._deleted_mask[self.layout.index(int(element_id))] = True

    def _materialized_packed_state(
        self,
        index: int,
        token: Optional[StateTrialToken],
    ) -> Dict[str, Any]:
        committed = self._views(self._committed_buffer)
        trial = self._views(self._trial_buffer)
        result: Dict[str, Any] = {}
        use_trial = token is not None and not bool(self._deleted_mask[index])
        for field_name in _CORE_FIELDS:
            field_index = _FIELD_INDEX[field_name]
            source = (
                getattr(trial, field_name)
                if use_trial and self._write_stamps[field_index, index] == self._serial
                else getattr(committed, field_name)
            )
            result[field_name] = np.array(source[index], copy=True, order="C")
        for key, value in self._sidecars[index].items():
            result[key] = _thaw_value(value)
        origin_is_trial = bool(
            use_trial
            and self._q4_origin_trial_stamps[index] == self._serial
        )
        origin_is_committed = bool(self._q4_origin_present_committed[index])
        if origin_is_trial or origin_is_committed:
            assert self._q4_origin_committed_plastic is not None
            assert self._q4_origin_trial_plastic is not None
            assert self._q4_origin_committed_alpha is not None
            assert self._q4_origin_trial_alpha is not None
            plastic_source = (
                self._q4_origin_trial_plastic
                if origin_is_trial
                else self._q4_origin_committed_plastic
            )
            alpha_source = (
                self._q4_origin_trial_alpha
                if origin_is_trial
                else self._q4_origin_committed_alpha
            )
            result[_Q4_ALGORITHMIC_ORIGIN_KEY] = {
                "schema_id": _Q4_ALGORITHMIC_ORIGIN_SCHEMA_ID,
                "kind": _Q4_ALGORITHMIC_ORIGIN_KIND,
                "num_layers": self.layout.num_layers,
                "parent_plastic_strain": np.asarray(
                    plastic_source[index], dtype=np.float64
                ).tolist(),
                "parent_alpha": np.asarray(
                    alpha_source[index], dtype=np.float64
                ).tolist(),
            }
        return result

    def _is_present(
        self,
        index: int,
        token: Optional[StateTrialToken],
    ) -> bool:
        return bool(
            self._present_committed[index]
            or (
                token is not None
                and self._present_trial_stamps[index] == self._serial
            )
        )

    def materialize(
        self,
        *,
        element_ids: Optional[Sequence[int]] = None,
        trial_token: Optional[StateTrialToken] = None,
        policy: StateMaterializationPolicy | str = StateMaterializationPolicy.EXPLICIT,
    ) -> Dict[int, Any]:
        """Materialize owned public dictionaries for restart/recovery/results."""

        with self._lock:
            if trial_token is not None:
                self._require_active(trial_token)
            reason = StateMaterializationPolicy(policy).value
            start = time.perf_counter()
            indices = self._indices(element_ids)
            result: Dict[int, Any] = {}
            for index_value in indices:
                index = int(index_value)
                element_id = self.layout.element_ids[index]
                if not self._is_present(index, trial_token):
                    continue
                if trial_token is not None and element_id in self._trial_legacy:
                    result[element_id] = _owned_copy(self._trial_legacy[element_id][0])
                elif not self._packed_mask[index]:
                    result[element_id] = _owned_copy(self._fallback_committed[element_id])
                else:
                    result[element_id] = self._materialized_packed_state(index, trial_token)
            elapsed = time.perf_counter() - start
            self._metrics["state_materialization_seconds"] += elapsed
            self._metrics["state_materialization_count"] += 1
            reasons = self._metrics["materialization_reasons"]
            reasons[reason] = int(reasons.get(reason, 0)) + 1
            return result

    def materialize_owned(
        self,
        *,
        element_ids: Optional[Sequence[int]] = None,
        trial_token: Optional[StateTrialToken] = None,
        policy: StateMaterializationPolicy | str = StateMaterializationPolicy.EXPLICIT,
    ) -> Dict[int, Any]:
        """Explicitly named alias emphasizing the ownership guarantee."""

        return self.materialize(
            element_ids=element_ids,
            trial_token=trial_token,
            policy=policy,
        )

    def commit(self, token: StateTrialToken) -> int:
        """Commit the active trial, using a pointer swap for full updates."""

        with self._lock:
            self._require_active(token)
            start = time.perf_counter()
            converted_indices = {
                self.layout.index(element_id) for element_id in self._trial_legacy
            }
            committed = self._views(self._committed_buffer)
            trial = self._views(self._trial_buffer)
            copied_element_fields = 0
            for field_name in _CORE_FIELDS:
                field_index = _FIELD_INDEX[field_name]
                missing = self._write_stamps[field_index] != self._serial
                missing = np.logical_or(missing, self._deleted_mask)
                if converted_indices:
                    missing[list(converted_indices)] = True
                if np.any(missing):
                    getattr(trial, field_name)[missing] = getattr(
                        committed, field_name
                    )[missing]
                    copied_element_fields += int(np.count_nonzero(missing))

            next_fallback = dict(self._fallback_committed)
            for element_id, (state, reason) in self._trial_legacy.items():
                index = self.layout.index(element_id)
                if self._deleted_mask[index]:
                    continue
                self._packed_mask[index] = False
                next_fallback[element_id] = _owned_copy(state)
                self._fallback_reasons[element_id] = reason
            self._fallback_committed = next_fallback
            self._present_committed = np.logical_or(
                self._present_committed,
                self._present_trial_stamps == self._serial,
            )

            if self._q4_origin_committed_plastic is not None:
                assert self._q4_origin_trial_plastic is not None
                assert self._q4_origin_committed_alpha is not None
                assert self._q4_origin_trial_alpha is not None
                missing_origin = self._q4_origin_trial_stamps != self._serial
                missing_origin = np.logical_or(
                    missing_origin, self._deleted_mask
                )
                if converted_indices:
                    missing_origin[list(converted_indices)] = True
                if np.any(missing_origin):
                    self._q4_origin_trial_plastic[missing_origin] = (
                        self._q4_origin_committed_plastic[missing_origin]
                    )
                    self._q4_origin_trial_alpha[missing_origin] = (
                        self._q4_origin_committed_alpha[missing_origin]
                    )
                self._q4_origin_present_committed = np.logical_or(
                    self._q4_origin_present_committed,
                    self._q4_origin_trial_stamps == self._serial,
                )
                (
                    self._q4_origin_committed_plastic,
                    self._q4_origin_trial_plastic,
                ) = (
                    self._q4_origin_trial_plastic,
                    self._q4_origin_committed_plastic,
                )
                (
                    self._q4_origin_committed_alpha,
                    self._q4_origin_trial_alpha,
                ) = (
                    self._q4_origin_trial_alpha,
                    self._q4_origin_committed_alpha,
                )

            self._committed_buffer, self._trial_buffer = (
                self._trial_buffer,
                self._committed_buffer,
            )
            self._generation += 1
            self._active_token = None
            self._trial_legacy = {}
            self._metrics["state_commit_count"] += 1
            if copied_element_fields:
                self._metrics["state_bounded_copy_commit_count"] += 1
                self._metrics["state_commit_copied_element_fields"] += copied_element_fields
            else:
                self._metrics["state_swap_commit_count"] += 1
            self._metrics["state_commit_seconds"] += time.perf_counter() - start
            return int(self._generation)

    def discard_trial(self, token: StateTrialToken) -> None:
        """Reject the active trial without copying any constitutive arrays."""

        with self._lock:
            self._require_active(token)
            start = time.perf_counter()
            self._active_token = None
            self._trial_legacy = {}
            self._metrics["state_discard_count"] += 1
            self._metrics["state_discard_seconds"] += time.perf_counter() - start

    def discard(self, token: StateTrialToken) -> None:
        """Short transaction alias for :meth:`discard_trial`."""

        self.discard_trial(token)

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            fallback_reasons: Dict[str, list[int]] = {}
            for element_id, reason in self._fallback_reasons.items():
                if not self._packed_mask[self.layout.index(element_id)]:
                    fallback_reasons.setdefault(reason, []).append(int(element_id))
            for element_ids in fallback_reasons.values():
                element_ids.sort()
            q4_origin_bytes = 0
            if self._q4_origin_committed_plastic is not None:
                assert self._q4_origin_trial_plastic is not None
                assert self._q4_origin_committed_alpha is not None
                assert self._q4_origin_trial_alpha is not None
                q4_origin_bytes = int(
                    self._q4_origin_committed_plastic.nbytes
                    + self._q4_origin_trial_plastic.nbytes
                    + self._q4_origin_committed_alpha.nbytes
                    + self._q4_origin_trial_alpha.nbytes
                )
            return {
                "state_batch_count": 1,
                "state_point_count": int(self.layout.state_point_count),
                "state_buffer_bytes": int(
                    self._committed_buffer.nbytes + self._trial_buffer.nbytes
                    + q4_origin_bytes
                ),
                "q4_algorithmic_origin_buffer_bytes": q4_origin_bytes,
                "q4_algorithmic_origin_element_count": int(
                    np.count_nonzero(self._q4_origin_present_committed)
                ),
                "state_pack_seconds": float(self._metrics["state_pack_seconds"]),
                "state_trial_update_seconds": float(
                    self._metrics["state_trial_update_seconds"]
                ),
                "state_commit_seconds": float(self._metrics["state_commit_seconds"]),
                "state_discard_seconds": float(self._metrics["state_discard_seconds"]),
                "state_materialization_seconds": float(
                    self._metrics["state_materialization_seconds"]
                ),
                "dictionary_fallback_element_count": len(self.fallback_element_ids),
                "dictionary_fallback_reasons": fallback_reasons,
                "deleted_element_count": int(np.count_nonzero(self._deleted_mask)),
                "committed_present_element_count": int(
                    np.count_nonzero(self._present_committed)
                ),
                "generation": int(self._generation),
                **{
                    key: (_owned_copy(value) if isinstance(value, dict) else int(value))
                    for key, value in self._metrics.items()
                    if key
                    not in {
                        "state_pack_seconds",
                        "state_trial_update_seconds",
                        "state_commit_seconds",
                        "state_discard_seconds",
                        "state_materialization_seconds",
                    }
                },
            }

    def __getitem__(self, element_id: int) -> Any:
        key = int(element_id)
        return self.materialize(element_ids=(key,))[key]

    def __iter__(self) -> Iterator[int]:
        return (
            element_id
            for index, element_id in enumerate(self.layout.element_ids)
            if bool(self._present_committed[index])
        )

    def __len__(self) -> int:
        return int(np.count_nonzero(self._present_committed))


class NonlinearStateStore(Mapping[int, Any]):
    """Solver-owned transaction coordinator for multiple state batches."""

    def __init__(self) -> None:
        self._owner = object()
        self._lock = threading.RLock()
        self._batches: Dict[Hashable, ShellStateBatch] = {}
        self._element_to_batch: Dict[int, Hashable] = {}
        self._fallback_committed: Dict[int, Any] = {}
        self._fallback_sidecars: Dict[int, Mapping[str, Any]] = {}
        self._deleted_fallback: set[int] = set()
        self._generation = 0
        self._serial = 0
        self._active_token: Optional[StateTrialToken] = None
        self._batch_trial_tokens: Dict[Hashable, StateTrialToken] = {}
        self._native_rotation_store: Optional[NativeRotationStateStore] = None
        self._native_rotation_trial_token: Optional[NativeRotationTrialToken] = None
        self._native_element_bindings: Optional[
            Dict[int, _NativeElementStateBinding]
        ] = None
        self._native_trial_full_displacement: Optional[np.ndarray] = None
        self._fallback_trial: Dict[int, Any] = {}
        self._pack_seconds = 0.0
        self._trial_update_seconds = 0.0
        self._commit_seconds = 0.0
        self._discard_seconds = 0.0
        self._materialization_seconds = 0.0
        self._materialization_count = 0
        self._stale_token_errors = 0

    @classmethod
    def from_shell_layouts(
        cls,
        layouts: Sequence[ShellStateLayout],
        committed_states: Optional[Mapping[int, Any]] = None,
    ) -> "NonlinearStateStore":
        store = cls()
        states = committed_states or {}
        consumed: set[int] = set()
        for index, layout in enumerate(layouts):
            store.add_shell_batch(layout, states, batch_key=index)
            consumed.update(layout.element_ids)
        remaining = {
            int(element_id): state
            for element_id, state in states.items()
            if int(element_id) not in consumed
        }
        store.add_dictionary_fallback(remaining)
        return store

    @classmethod
    def pack(
        cls,
        layouts: Sequence[ShellStateLayout],
        committed_states: Optional[Mapping[int, Any]] = None,
    ) -> "NonlinearStateStore":
        """Pack shell layouts and preserve every unbatched state as fallback."""

        return cls.from_shell_layouts(layouts, committed_states)

    @property
    def generation(self) -> int:
        return int(self._generation)

    @property
    def has_active_trial(self) -> bool:
        return self._active_token is not None

    @property
    def has_native_rotations(self) -> bool:
        return self._native_rotation_store is not None

    @property
    def native_rotation_store(self) -> Optional[NativeRotationStateStore]:
        return self._native_rotation_store

    def attach_native_rotation_store(
        self,
        rotation_store: NativeRotationStateStore,
    ) -> None:
        """Attach the one solver-owned native kinematic history before use."""

        if not isinstance(rotation_store, NativeRotationStateStore):
            raise TypeError("rotation_store must be a NativeRotationStateStore")
        with self._lock:
            if self._active_token is not None:
                raise StateTransactionError(
                    "Cannot attach native rotations during an active trial"
                )
            if self._native_rotation_store is not None:
                raise StateTransactionError(
                    "Native rotation state is already attached"
                )
            if self._generation != 0 or rotation_store.generation != 0:
                raise StateTransactionError(
                    "Native rotations must be attached before the first nonlinear trial"
                )
            raw_bindings = getattr(
                rotation_store,
                _NATIVE_ELEMENT_BINDINGS_ATTRIBUTE,
                MappingProxyType({}),
            )
            if not isinstance(raw_bindings, Mapping):
                raise NativeRotationValidationError(
                    "Native rotation store carries malformed element ownership"
                )
            bindings: Dict[int, _NativeElementStateBinding] = {}
            for raw_element_id, binding in raw_bindings.items():
                element_id = int(raw_element_id)
                if not isinstance(binding, _NativeElementStateBinding):
                    raise NativeRotationValidationError(
                        f"Native element {element_id} has malformed state ownership"
                    )
                if (
                    binding.dof_mapping.size
                    and int(np.max(binding.dof_mapping))
                    >= rotation_store.committed_full_displacement.size
                ):
                    raise NativeRotationValidationError(
                        f"Native element {element_id} DOF ownership is out of bounds"
                    )
                bindings[element_id] = binding
            self._native_rotation_store = rotation_store
            self._native_element_bindings = bindings

    def add_shell_batch(
        self,
        layout: ShellStateLayout,
        committed_states: Optional[Mapping[int, Any]] = None,
        *,
        batch_key: Optional[Hashable] = None,
    ) -> ShellStateBatch:
        with self._lock:
            if self._active_token is not None:
                raise StateTransactionError("Cannot add a shell batch during an active trial")
            key: Hashable = len(self._batches) if batch_key is None else batch_key
            if key in self._batches:
                raise KeyError(f"Duplicate shell-state batch key {key!r}")
            duplicates = set(layout.element_ids).intersection(self._element_to_batch)
            duplicates.update(set(layout.element_ids).intersection(self._fallback_committed))
            if duplicates:
                raise ValueError(f"Element states are already registered: {sorted(duplicates)}")
            batch = ShellStateBatch(layout, committed_states)
            self._batches[key] = batch
            for element_id in layout.element_ids:
                self._element_to_batch[element_id] = key
            return batch

    def add_dictionary_fallback(self, states: Mapping[int, Any]) -> None:
        with self._lock:
            if self._active_token is not None:
                raise StateTransactionError(
                    "Cannot add dictionary fallback states during an active trial"
                )
            start = time.perf_counter()
            for raw_element_id, state in states.items():
                element_id = int(raw_element_id)
                if element_id in self._element_to_batch or element_id in self._fallback_committed:
                    raise ValueError(f"Element state {element_id} is already registered")
                self._fallback_committed[element_id] = _owned_copy(state)
                self._fallback_sidecars[element_id] = _extract_sidecar(state)
            self._pack_seconds += time.perf_counter() - start

    def shell_batch(self, batch_key: Hashable) -> ShellStateBatch:
        return self._batches[batch_key]

    def shell_batch_for_layout(self, layout: ShellStateLayout) -> Optional[ShellStateBatch]:
        if not layout.element_ids:
            return None
        key = self._element_to_batch.get(layout.element_ids[0])
        if key is None:
            return None
        batch = self._batches[key]
        return batch if batch.compatible_with(layout) else None

    def begin_trial(
        self,
        *,
        full_displacement: Any = None,
        full_coordinates: Any = None,
    ) -> StateTrialToken:
        with self._lock:
            if self._active_token is not None:
                raise StateTransactionError("A nonlinear-state trial is already active")
            if self._native_rotation_store is not None and (
                full_displacement is None or full_coordinates is None
            ):
                raise StateTransactionError(
                    "Native nonlinear-state trials require the complete global "
                    "displacement and nodal-coordinate arrays"
                )
            self._serial += 1
            token = StateTrialToken(self._generation, self._serial, self._owner)
            child_tokens: Dict[Hashable, StateTrialToken] = {}
            native_token: Optional[NativeRotationTrialToken] = None
            try:
                for key, batch in self._batches.items():
                    child_tokens[key] = batch.begin_trial()
                if self._native_rotation_store is not None:
                    native_token = self._native_rotation_store.begin_trial(
                        full_displacement,
                        full_coordinates,
                    )
            except Exception:
                for key, child_token in child_tokens.items():
                    self._batches[key].discard_trial(child_token)
                if (
                    self._native_rotation_store is not None
                    and native_token is not None
                ):
                    self._native_rotation_store.discard_trial(native_token)
                raise
            self._batch_trial_tokens = child_tokens
            self._native_rotation_trial_token = native_token
            self._native_trial_full_displacement = (
                None
                if native_token is None
                else np.frombuffer(
                    np.ascontiguousarray(
                        np.asarray(full_displacement, dtype=np.float64)
                    ).tobytes(order="C"),
                    dtype=np.float64,
                ).reshape(np.asarray(full_displacement).shape)
            )
            self._fallback_trial = {}
            self._active_token = token
            return token

    def begin(
        self,
        *,
        full_displacement: Any = None,
        full_coordinates: Any = None,
    ) -> StateTrialToken:
        """Short transaction alias for :meth:`begin_trial`."""

        return self.begin_trial(
            full_displacement=full_displacement,
            full_coordinates=full_coordinates,
        )

    def active_trial_token(self) -> StateTrialToken:
        """Return the active token for an assembly evaluator."""

        with self._lock:
            if self._active_token is None:
                raise StateTransactionError("No nonlinear-state trial is active")
            return self._active_token

    def replace_trial(
        self,
        *,
        full_displacement: Any = None,
        full_coordinates: Any = None,
    ) -> StateTrialToken:
        """Reject any prior candidate and begin another from committed state."""

        with self._lock:
            if self._active_token is not None:
                self.discard_trial(self._active_token)
            return self.begin_trial(
                full_displacement=full_displacement,
                full_coordinates=full_coordinates,
            )

    def _require_active(self, token: StateTrialToken) -> None:
        valid = bool(
            isinstance(token, StateTrialToken)
            and token._owner is self._owner
            and token.generation == self._generation
            and self._active_token is token
        )
        if not valid:
            self._stale_token_errors += 1
            raise StaleStateTokenError(
                "Trial token is stale, belongs to another store, or is not active"
            )

    def update_shell_trial(
        self,
        token: StateTrialToken,
        batch_key: Hashable,
        **fields: Any,
    ) -> int:
        with self._lock:
            self._require_active(token)
            result = self._batches[batch_key].update_trial(
                self._batch_trial_tokens[batch_key],
                **fields,
            )
            return result

    def shell_trial_for_layout(
        self,
        token: StateTrialToken,
        layout: ShellStateLayout,
    ) -> Optional[tuple[ShellStateBatch, StateTrialToken]]:
        """Resolve a registered batch and its child transaction token."""

        with self._lock:
            self._require_active(token)
            batch = self.shell_batch_for_layout(layout)
            if batch is None:
                return None
            key = self._element_to_batch[layout.element_ids[0]]
            return batch, self._batch_trial_tokens[key]

    def native_element_rotation_view(
        self,
        token: StateTrialToken,
        element_id: Hashable,
        node_ids: Sequence[int],
        reference_directors: Any,
    ) -> NativeElementRotationView:
        """Return one immutable node-shared native trial in element order."""

        with self._lock:
            self._require_active(token)
            if (
                self._native_rotation_store is None
                or self._native_rotation_trial_token is None
            ):
                raise StateTransactionError(
                    "No native rotation transaction is active"
                )
            key = int(element_id)
            bindings = self._native_element_bindings
            binding = None if bindings is None else bindings.get(key)
            if binding is None:
                # Low-level synthetic/native-extension stores can be attached
                # without model metadata.  They may consume a node-shared view,
                # but cannot persist an S3 redundant kinematic state.
                return self._native_rotation_store.element_view(
                    element_id,
                    node_ids,
                    reference_directors,
                    trial_token=self._native_rotation_trial_token,
                )
            normalized_nodes = tuple(int(value) for value in node_ids)
            made_directors = np.asarray(reference_directors, dtype=np.float64)
            if normalized_nodes != binding.node_ids:
                raise NativeRotationValidationError(
                    f"Native element {key} connectivity disagrees with its model binding"
                )
            if (
                made_directors.shape != binding.reference_directors.shape
                or not np.all(np.isfinite(made_directors))
                or not np.array_equal(made_directors, binding.reference_directors)
            ):
                raise NativeRotationValidationError(
                    f"Native element {key} reference directors disagree with its model binding"
                )
            return self._native_rotation_store.element_view(
                key,
                binding.node_ids,
                binding.reference_directors,
                trial_token=self._native_rotation_trial_token,
            )

    def _native_view_for_binding(
        self,
        element_id: int,
        *,
        trial: bool,
    ) -> NativeElementRotationView:
        rotation_store = self._native_rotation_store
        bindings = self._native_element_bindings
        if rotation_store is None or bindings is None or element_id not in bindings:
            raise NativeRotationValidationError(
                f"Native element {element_id} has no solver-owned state binding"
            )
        binding = bindings[element_id]
        trial_token = None
        if trial:
            trial_token = self._native_rotation_trial_token
            if trial_token is None:
                raise StateTransactionError(
                    "Native state consistency requires an active rotation candidate"
                )
        return rotation_store.element_view(
            element_id,
            binding.node_ids,
            binding.reference_directors,
            trial_token=trial_token,
        )

    def native_material_context(self, token: StateTrialToken, element_id: int) -> NativeMaterialContext:
        """Issue a live context only for an explicitly model-bound protocol."""
        with self._lock:
            self._require_active(token)
            binding = (self._native_element_bindings or {}).get(int(element_id))
            if binding is None or binding.material_validator is None:
                raise StateTransactionError("Element has no native material protocol binding")
            return NativeMaterialContext(self, token, int(element_id), binding.node_ids,
                                         binding.reference_directors)

    def _validate_native_material_candidate(self, element_id: int, state: Any, full: Any) -> None:
        binding = (self._native_element_bindings or {}).get(element_id)
        if binding is None or binding.material_validator is None:
            return
        token = self.active_trial_token()
        if element_id in self._deleted_fallback:
            raise StateTransactionError("Native material protocol does not authorize deletion")
        if element_id in self._element_to_batch or element_id not in self._fallback_committed:
            raise StateTransactionError("Native material protocol requires explicit unbatched committed state")
        values = np.asarray(full, dtype=np.float64)
        if values.ndim != 1 or int(np.max(binding.dof_mapping)) >= values.size:
            raise StateTransactionError("Native material validation requires complete global displacement")
        binding.material_validator.validate(
            _owned_copy(state), previous_state=_owned_copy(self._fallback_committed[element_id]),
            native_view=self._native_view_for_binding(element_id, trial=True),
            displacement=np.array(values[binding.dof_mapping], copy=True), phase="trial",
        )
        self._require_active(token)

    def _validate_native_element_state(
        self,
        element_id: int,
        state: Any,
        full_displacement: Any,
    ) -> None:
        bindings = self._native_element_bindings
        if (
            bindings is None
            or element_id not in bindings
            or not bindings[element_id].state_consistency_required
        ):
            return
        if not isinstance(state, Mapping):
            raise NativeRotationValidationError(
                f"Native element {element_id} trial state must be a mapping"
            )
        required = (
            "committed_total_u",
            "reference_corner_directors",
            "committed_nodal_rotation_matrices",
            "committed_director_triads",
        )
        missing = [name for name in required if name not in state]
        if missing:
            raise NativeRotationValidationError(
                f"Native element {element_id} trial state is missing "
                + ", ".join(missing)
            )
        binding = bindings[element_id]
        full = np.asarray(full_displacement, dtype=np.float64)
        if (
            full.ndim != 1
            or not np.all(np.isfinite(full))
            or int(np.max(binding.dof_mapping)) >= full.size
        ):
            raise NativeRotationValidationError(
                "Native state consistency requires a complete finite displacement vector"
            )
        expected_total = full[binding.dof_mapping]
        stored_total = np.asarray(state["committed_total_u"], dtype=np.float64)
        if (
            stored_total.shape != expected_total.shape
            or not np.all(np.isfinite(stored_total))
            or not np.array_equal(stored_total, expected_total)
        ):
            raise NativeRotationValidationError(
                f"Native element {element_id} committed_total_u disagrees with "
                "the solver-owned trial"
            )
        view = self._native_view_for_binding(element_id, trial=True)
        stored_reference = np.asarray(
            state["reference_corner_directors"], dtype=np.float64
        )
        if (
            stored_reference.shape != view.reference_directors.shape
            or not np.all(np.isfinite(stored_reference))
            or not np.array_equal(stored_reference, view.reference_directors)
        ):
            raise NativeRotationValidationError(
                f"Native element {element_id} reference_corner_directors "
                "disagree with the solver-owned trial"
            )
        stored_rotations = np.asarray(
            state["committed_nodal_rotation_matrices"], dtype=np.float64
        )
        if (
            stored_rotations.shape != view.trial_rotation_matrices.shape
            or not np.all(np.isfinite(stored_rotations))
            or not np.array_equal(stored_rotations, view.trial_rotation_matrices)
        ):
            raise NativeRotationValidationError(
                f"Native element {element_id} committed_nodal_rotation_matrices "
                "disagree with the solver-owned trial"
            )
        triads = np.asarray(state["committed_director_triads"], dtype=np.float64)
        corner_count = len(binding.node_ids)
        if (
            triads.ndim != 3
            or triads.shape[0] < corner_count
            or triads.shape[1:] != (3, 3)
            or not np.all(np.isfinite(triads))
            or not np.allclose(
                triads[:corner_count, :, 2],
                view.trial_directors,
                rtol=0.0,
                atol=_NATIVE_DIRECTOR_CONSISTENCY_TOLERANCE,
            )
        ):
            raise NativeRotationValidationError(
                f"Native element {element_id} corner director normals disagree "
                "with the solver-owned trial"
            )

    def _materialize_native_element_state(
        self,
        element_id: int,
        state: Any,
        full_displacement: Any,
        *,
        trial: bool,
    ) -> Any:
        bindings = self._native_element_bindings
        if (
            bindings is None
            or element_id not in bindings
            or not bindings[element_id].state_consistency_required
        ):
            return _owned_copy(state)
        if not isinstance(state, Mapping):
            raise NativeRotationValidationError(
                f"Native element {element_id} committed state must be a mapping"
            )
        binding = bindings[element_id]
        full = np.asarray(full_displacement, dtype=np.float64)
        if full.ndim != 1 or int(np.max(binding.dof_mapping)) >= full.size:
            raise NativeRotationValidationError(
                "Native state materialization requires a complete displacement vector"
            )
        view = self._native_view_for_binding(element_id, trial=trial)
        result = _owned_copy(dict(state))
        triads = np.asarray(
            result.get("committed_director_triads", ()), dtype=np.float64
        ).copy()
        corner_count = len(binding.node_ids)
        if triads.ndim != 3 or triads.shape[0] < corner_count or triads.shape[1:] != (3, 3):
            raise NativeRotationValidationError(
                f"Native element {element_id} has no materializable director triads"
            )
        from .e4_pl_s3_state import reconstruct_director_triad

        directors = view.trial_directors if trial else view.committed_directors
        for local_node in range(corner_count):
            triads[local_node] = reconstruct_director_triad(directors[local_node])
        result["committed_total_u"] = np.asarray(
            full[binding.dof_mapping], dtype=np.float64
        ).copy()
        result["reference_corner_directors"] = np.asarray(
            view.reference_directors, dtype=np.float64
        ).copy()
        result["committed_nodal_rotation_matrices"] = np.asarray(
            view.trial_rotation_matrices
            if trial
            else view.committed_rotation_matrices,
            dtype=np.float64,
        ).copy()
        result["committed_director_triads"] = triads
        if "state_integrity_sha256" in result:
            from .e4_pl_s3_state import seal_committed_s3_state

            result = seal_committed_s3_state(result)
        return result

    def _validate_fallback_sidecar(self, element_id: int, state: Any) -> Any:
        sidecar = self._fallback_sidecars.get(element_id, MappingProxyType({}))
        if not isinstance(state, Mapping):
            if sidecar:
                raise ImmutableStateSidecarError(
                    "A state carrying immutable initial fields must remain a mapping"
                )
            return _owned_copy(state)
        result = _owned_copy(dict(state))
        for key in result:
            if _is_sidecar_key(str(key)) and str(key) not in sidecar:
                raise ImmutableStateSidecarError(
                    f"Trial update cannot introduce immutable sidecar {key!r}"
                )
        for key, frozen_value in sidecar.items():
            expected = _thaw_value(frozen_value)
            if key in result and not _state_values_equal(result[key], expected):
                raise ImmutableStateSidecarError(
                    f"Trial update cannot modify immutable sidecar {key!r}"
                )
            result[key] = expected
        return result

    def set_trial_state(
        self,
        token: StateTrialToken,
        element_id: int,
        state: Any,
    ) -> None:
        with self._lock:
            self._require_active(token)
            key = int(element_id)
            start = time.perf_counter()
            self._validate_native_material_candidate(key, state, self._native_trial_full_displacement)
            if (
                self._native_element_bindings is not None
                and key in self._native_element_bindings
                and self._native_element_bindings[key].state_consistency_required
            ):
                if self._native_trial_full_displacement is None:
                    raise StateTransactionError(
                        "Native element trial state has no solver-owned displacement"
                    )
                self._validate_native_element_state(
                    key,
                    state,
                    self._native_trial_full_displacement,
                )
            batch_key = self._element_to_batch.get(key)
            if batch_key is not None:
                self._batches[batch_key].set_trial_state(
                    self._batch_trial_tokens[batch_key], key, state
                )
            elif key in self._fallback_committed:
                if key not in self._deleted_fallback:
                    self._fallback_trial[key] = self._validate_fallback_sidecar(key, state)
                    self._trial_update_seconds += time.perf_counter() - start
            else:
                # Scalar/non-shell elements can acquire their first material
                # history during this candidate.  It becomes visible only if
                # the enclosing increment is committed.
                self._fallback_trial[key] = _owned_copy(state)
                self._trial_update_seconds += time.perf_counter() - start

    def set_trial_states(
        self,
        token: StateTrialToken,
        states: Mapping[int, Any],
    ) -> None:
        for element_id, state in states.items():
            self.set_trial_state(token, int(element_id), state)

    def freeze_deleted(self, element_ids: Sequence[int]) -> None:
        with self._lock:
            by_batch: Dict[Hashable, list[int]] = {}
            for raw_element_id in element_ids:
                element_id = int(raw_element_id)
                batch_key = self._element_to_batch.get(element_id)
                if batch_key is not None:
                    by_batch.setdefault(batch_key, []).append(element_id)
                elif element_id in self._fallback_committed:
                    self._deleted_fallback.add(element_id)
                else:
                    raise KeyError(f"Element state {element_id} is not registered")
            for batch_key, ids in by_batch.items():
                self._batches[batch_key].freeze_deleted(ids)

    def commit(
        self,
        token: StateTrialToken,
        *,
        accepted_full_displacement: Any = None,
        accepted_full_coordinates: Any = None,
    ) -> int:
        with self._lock:
            self._require_active(token)
            # Validate every child before the first pointer swap for atomicity.
            for key, batch in self._batches.items():
                batch._require_active(self._batch_trial_tokens[key])
            if self._native_rotation_store is not None:
                if self._native_rotation_trial_token is None:
                    raise StateTransactionError(
                        "Native nonlinear state has no active rotation candidate"
                    )
                self._native_rotation_store.validate_commit_configuration(
                    self._native_rotation_trial_token,
                    accepted_full_displacement,
                    accepted_full_coordinates,
                )
            next_fallback: Optional[Dict[int, Any]] = None
            next_sidecars: Optional[Dict[int, Mapping[str, Any]]] = None
            if self._fallback_trial or self._native_element_bindings is not None:
                next_fallback = dict(self._fallback_committed)
                next_fallback.update(
                    {
                        element_id: _owned_copy(state)
                        for element_id, state in self._fallback_trial.items()
                        if element_id not in self._deleted_fallback
                    }
                )
                next_sidecars = dict(self._fallback_sidecars)
                for element_id, state in self._fallback_trial.items():
                    if element_id not in next_sidecars:
                        next_sidecars[element_id] = _extract_sidecar(state)
            if self._native_element_bindings is not None:
                assert next_fallback is not None
                for element_id in self._native_element_bindings:
                    if self._native_element_bindings[element_id].material_validator is not None:
                        if element_id not in self._fallback_trial:
                            raise StateTransactionError("Native material element produced no candidate state")
                        self._validate_native_material_candidate(
                            element_id, self._fallback_trial[element_id], accepted_full_displacement
                        )
                    if not self._native_element_bindings[
                        element_id
                    ].state_consistency_required:
                        continue
                    if element_id in self._deleted_fallback:
                        # A deleted formulation-native element retains its
                        # exact deletion-time material and kinematic state.
                        # The model-wide node rotation history may advance for
                        # surviving neighbours, but rebinding this frozen
                        # payload to the accepted global u would falsely make
                        # it an ACTIVE current-state record.
                        continue
                    if element_id in self._element_to_batch:
                        raise NativeRotationValidationError(
                            f"Native element {element_id} cannot use a generic packed state batch"
                        )
                    if element_id not in next_fallback:
                        raise NativeRotationValidationError(
                            f"Native element {element_id} has no committed material state"
                        )
                    if element_id not in self._deleted_fallback:
                        if element_id not in self._fallback_trial:
                            raise NativeRotationValidationError(
                                f"Native element {element_id} produced no trial material state"
                            )
                        self._validate_native_element_state(
                            element_id,
                            self._fallback_trial[element_id],
                            accepted_full_displacement,
                        )
                    next_fallback[element_id] = self._materialize_native_element_state(
                        element_id,
                        next_fallback[element_id],
                        accepted_full_displacement,
                        trial=True,
                    )
            start = time.perf_counter()
            for key, batch in self._batches.items():
                batch.commit(self._batch_trial_tokens[key])
            if self._native_rotation_store is not None:
                assert self._native_rotation_trial_token is not None
                self._native_rotation_store.commit_trial(
                    self._native_rotation_trial_token,
                    accepted_full_displacement,
                    accepted_full_coordinates,
                )
            if self._fallback_trial or self._native_element_bindings is not None:
                assert next_fallback is not None
                assert next_sidecars is not None
                self._fallback_committed = next_fallback
                self._fallback_sidecars = next_sidecars
            self._generation += 1
            self._active_token = None
            self._batch_trial_tokens = {}
            self._native_rotation_trial_token = None
            self._native_trial_full_displacement = None
            self._fallback_trial = {}
            self._commit_seconds += time.perf_counter() - start
            return int(self._generation)

    def discard_trial(self, token: StateTrialToken) -> None:
        with self._lock:
            self._require_active(token)
            start = time.perf_counter()
            for key, batch in self._batches.items():
                batch.discard_trial(self._batch_trial_tokens[key])
            if self._native_rotation_store is not None:
                if self._native_rotation_trial_token is None:
                    raise StateTransactionError(
                        "Native nonlinear state has no active rotation candidate"
                    )
                self._native_rotation_store.discard_trial(
                    self._native_rotation_trial_token
                )
            self._active_token = None
            self._batch_trial_tokens = {}
            self._native_rotation_trial_token = None
            self._native_trial_full_displacement = None
            self._fallback_trial = {}
            self._discard_seconds += time.perf_counter() - start

    def discard(self, token: StateTrialToken) -> None:
        """Short transaction alias for :meth:`discard_trial`."""

        self.discard_trial(token)

    def materialize(
        self,
        *,
        trial_token: Optional[StateTrialToken] = None,
        policy: StateMaterializationPolicy | str = StateMaterializationPolicy.EXPLICIT,
    ) -> Dict[int, Any]:
        with self._lock:
            if trial_token is not None:
                self._require_active(trial_token)
            start = time.perf_counter()
            result: Dict[int, Any] = {}
            for key, batch in self._batches.items():
                result.update(
                    batch.materialize(
                        trial_token=(
                            self._batch_trial_tokens[key]
                            if trial_token is not None
                            else None
                        ),
                        policy=policy,
                    )
                )
            for element_id, state in self._fallback_committed.items():
                source = (
                    self._fallback_trial[element_id]
                    if trial_token is not None and element_id in self._fallback_trial
                    else state
                )
                result[element_id] = _owned_copy(source)
            if trial_token is not None:
                for element_id, state in self._fallback_trial.items():
                    if element_id not in result:
                        result[element_id] = _owned_copy(state)
            self._materialization_seconds += time.perf_counter() - start
            self._materialization_count += 1
            return result

    def trial_view(self, token: StateTrialToken) -> "NonlinearStateTrialView":
        with self._lock:
            self._require_active(token)
            return NonlinearStateTrialView(self, token)

    def _trial_keys(self, token: StateTrialToken) -> tuple[int, ...]:
        self._require_active(token)
        keys: list[int] = []
        for batch_key, batch in self._batches.items():
            child_token = self._batch_trial_tokens[batch_key]
            for index, element_id in enumerate(batch.layout.element_ids):
                if batch._is_present(index, child_token):
                    keys.append(element_id)
        keys.extend(
            element_id
            for element_id in self._fallback_committed
            if element_id not in keys
        )
        keys.extend(
            element_id
            for element_id in self._fallback_trial
            if element_id not in keys
        )
        return tuple(keys)

    def _materialize_trial_element(
        self,
        token: StateTrialToken,
        element_id: int,
    ) -> Any:
        with self._lock:
            self._require_active(token)
            key = int(element_id)
            batch_key = self._element_to_batch.get(key)
            if batch_key is not None:
                child_token = self._batch_trial_tokens[batch_key]
                result = self._batches[batch_key].materialize(
                    element_ids=(key,),
                    trial_token=child_token,
                )
                try:
                    return result[key]
                except KeyError as exc:
                    raise KeyError(key) from exc
            if key in self._fallback_trial:
                return _owned_copy(self._fallback_trial[key])
            try:
                return _owned_copy(self._fallback_committed[key])
            except KeyError as exc:
                raise KeyError(key) from exc

    def materialize_owned(
        self,
        *,
        trial_token: Optional[StateTrialToken] = None,
        policy: StateMaterializationPolicy | str = StateMaterializationPolicy.EXPLICIT,
    ) -> Dict[int, Any]:
        """Explicitly named alias emphasizing the ownership guarantee."""

        return self.materialize(trial_token=trial_token, policy=policy)

    def max_equivalent_plastic_strain(self) -> float:
        """Return the committed alpha envelope without public materialization.

        Accepted-increment progress reporting needs this scalar frequently.  A
        direct reduction keeps the normal lifecycle entirely in solver-owned
        arrays; dictionary fallbacks are inspected in place and remain exact.
        """

        with self._lock:
            maximum = 0.0
            for batch in self._batches.values():
                packed_present = np.logical_and(
                    batch._packed_mask,
                    batch._present_committed,
                )
                if np.any(packed_present):
                    alpha = batch._views(batch._committed_buffer).alpha[packed_present]
                    if alpha.size:
                        maximum = max(maximum, float(np.max(alpha)))
                for element_id, state in batch._fallback_committed.items():
                    index = batch.layout.index(element_id)
                    if not batch._present_committed[index]:
                        continue
                    if isinstance(state, Mapping):
                        alpha = np.asarray(state.get("alpha", ()), dtype=float)
                        if alpha.size:
                            maximum = max(maximum, float(np.max(alpha)))
            for state in self._fallback_committed.values():
                if isinstance(state, Mapping):
                    alpha = np.asarray(state.get("alpha", ()), dtype=float)
                    if alpha.size:
                        maximum = max(maximum, float(np.max(alpha)))
            return float(maximum)

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            batch_diagnostics = [
                batch.diagnostics() for batch in self._batches.values()
            ]
            fallback_reasons: Dict[str, list[int]] = {
                "unbatched_element_state": sorted(self._fallback_committed)
            }
            for diagnostic in batch_diagnostics:
                for reason, element_ids in diagnostic[
                    "dictionary_fallback_reasons"
                ].items():
                    fallback_reasons.setdefault(reason, []).extend(element_ids)
            fallback_reasons = {
                reason: sorted(set(ids))
                for reason, ids in fallback_reasons.items()
                if ids
            }
            return {
                "state_batch_count": len(self._batches),
                "state_point_count": int(
                    sum(item["state_point_count"] for item in batch_diagnostics)
                ),
                "state_buffer_bytes": int(
                    sum(item["state_buffer_bytes"] for item in batch_diagnostics)
                ),
                "state_pack_seconds": float(
                    self._pack_seconds
                    + sum(item["state_pack_seconds"] for item in batch_diagnostics)
                ),
                "state_trial_update_seconds": float(
                    self._trial_update_seconds
                    + sum(
                        item["state_trial_update_seconds"]
                        for item in batch_diagnostics
                    )
                ),
                "state_commit_seconds": float(
                    self._commit_seconds
                ),
                "state_discard_seconds": float(
                    self._discard_seconds
                ),
                "state_materialization_seconds": float(
                    self._materialization_seconds
                ),
                "state_materialization_count": int(
                    self._materialization_count
                ),
                "dictionary_fallback_element_count": int(
                    len(self._fallback_committed)
                    + sum(
                        item["dictionary_fallback_element_count"]
                        for item in batch_diagnostics
                    )
                ),
                "dictionary_fallback_reasons": fallback_reasons,
                "deleted_element_count": int(
                    len(self._deleted_fallback)
                    + sum(item["deleted_element_count"] for item in batch_diagnostics)
                ),
                "generation": int(self._generation),
                "native_rotation_activated": self._native_rotation_store is not None,
                "native_rotation_node_count": (
                    0
                    if self._native_rotation_store is None
                    else len(self._native_rotation_store.node_ids)
                ),
                "native_rotation_generation": (
                    0
                    if self._native_rotation_store is None
                    else self._native_rotation_store.generation
                ),
                "stale_token_error_count": int(
                    self._stale_token_errors
                    + sum(item["stale_token_error_count"] for item in batch_diagnostics)
                ),
                "shell_batches": batch_diagnostics,
            }

    def __getitem__(self, element_id: int) -> Any:
        key = int(element_id)
        batch_key = self._element_to_batch.get(key)
        if batch_key is not None:
            return self._batches[batch_key][key]
        try:
            return _owned_copy(self._fallback_committed[key])
        except KeyError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[int]:
        for batch in self._batches.values():
            yield from batch
        yield from self._fallback_committed

    def __len__(self) -> int:
        return sum(len(batch) for batch in self._batches.values()) + len(
            self._fallback_committed
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> Dict[int, Any]:
        del memo
        return self.materialize_owned()


class NonlinearStateTrialView(Mapping[int, Any]):
    """Owned-on-access mapping for one uncommitted candidate generation."""

    def __init__(self, store: NonlinearStateStore, token: StateTrialToken) -> None:
        self._store = store
        self._token = token

    @property
    def store(self) -> NonlinearStateStore:
        return self._store

    @property
    def token(self) -> StateTrialToken:
        return self._token

    def commit(
        self,
        *,
        accepted_full_displacement: Any = None,
        accepted_full_coordinates: Any = None,
    ) -> NonlinearStateStore:
        self._store.commit(
            self._token,
            accepted_full_displacement=accepted_full_displacement,
            accepted_full_coordinates=accepted_full_coordinates,
        )
        return self._store

    def discard(self) -> None:
        self._store.discard_trial(self._token)

    def materialize_owned(
        self,
        *,
        policy: StateMaterializationPolicy | str = StateMaterializationPolicy.EXPLICIT,
    ) -> Dict[int, Any]:
        return self._store.materialize(trial_token=self._token, policy=policy)

    def __getitem__(self, element_id: int) -> Any:
        return self._store._materialize_trial_element(
            self._token,
            int(element_id),
        )

    def __iter__(self) -> Iterator[int]:
        return iter(self._store._trial_keys(self._token))

    def __len__(self) -> int:
        return len(self._store._trial_keys(self._token))

    def __deepcopy__(self, memo: Dict[int, Any]) -> Dict[int, Any]:
        del memo
        return self.materialize_owned()


def _model_full_coordinates(
    model: Any,
    displacements: Any,
    coordinate_node_ids: Sequence[int],
) -> np.ndarray:
    """Build physical nodal coordinates in an explicit frozen row order."""

    mesh = getattr(model, "mesh", None)
    nodes = getattr(mesh, "nodes", None)
    if not isinstance(nodes, Mapping):
        raise StateTransactionError(
            "Native state evaluation requires a model with an explicit node mapping"
        )
    full = np.asarray(displacements, dtype=np.float64)
    total_dofs = int(getattr(getattr(mesh, "dof_manager", None), "total_dofs", -1))
    if (
        full.ndim != 1
        or full.size != total_dofs
        or not np.all(np.isfinite(full))
    ):
        raise StateTransactionError(
            "Native state evaluation requires the complete finite global displacement vector"
        )
    model_node_ids = tuple(int(value) for value in nodes)
    normalized_node_ids = tuple(int(value) for value in coordinate_node_ids)
    if set(model_node_ids) != set(normalized_node_ids):
        raise StateTransactionError(
            "Native rotation topology differs from the current model node set"
        )
    coordinates = np.empty((len(normalized_node_ids), 3), dtype=np.float64)
    for row, node_id in enumerate(normalized_node_ids):
        try:
            node = nodes[node_id]
        except KeyError as exc:
            raise StateTransactionError(
                f"Native rotation topology references missing node {node_id}"
            ) from exc
        node_dofs = np.asarray(getattr(node, "dofs", ()), dtype=np.intp)
        if node_dofs.size < 3 or np.any(node_dofs[:3] < 0) or np.any(node_dofs[:3] >= full.size):
            raise StateTransactionError(
                f"Native node {node_id} has an invalid translational DOF mapping"
            )
        reference = np.asarray(node.coords(), dtype=np.float64)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise StateTransactionError(
                f"Native node {node_id} has invalid reference coordinates"
            )
        coordinates[row] = reference + full[node_dofs[:3]]
    return coordinates


def native_trial_full_coordinates(
    committed_states: NonlinearStateStore,
    model: Any,
    displacements: Any,
) -> np.ndarray:
    """Build physical nodal coordinates in the store's frozen row order."""

    rotation_store = committed_states.native_rotation_store
    if rotation_store is None:
        raise StateTransactionError("Nonlinear state has no native rotations")
    return _model_full_coordinates(
        model,
        displacements,
        rotation_store.coordinate_node_ids,
    )


def require_no_native_total_lagrangian_elements(
    model: Any,
    *,
    context: str,
) -> None:
    """Reject entry points that cannot own the native rotation transaction.

    A formulation-native element cannot safely fall through a legacy solver
    merely because it exposes familiar nodal DOFs.  Such a path would bypass
    the node-shared multiplicative history and could commit element-local
    additive rotations.  Entry points without the complete transaction
    lifecycle call this guard before evaluating any mechanics.
    """

    elements = getattr(getattr(model, "mesh", None), "elements", None)
    if not isinstance(elements, Mapping):
        raise StateTransactionError(
            f"{context} requires an explicit model element mapping"
        )
    native_ids = sorted(
        int(element_id)
        for element_id, element in elements.items()
        if bool(getattr(element, "formulation_native_total_lagrangian", False))
    )
    if native_ids:
        labels = ", ".join(str(value) for value in native_ids)
        raise StateTransactionError(
            f"{context} does not implement solver-owned native total-"
            "Lagrangian rotation transactions; use solve_static_nonlinear "
            f"for native element IDs [{labels}]"
        )


def create_model_native_rotation_store(
    model: Any,
    committed_states: Mapping[int, Any],
    committed_full_displacement: Any,
    *,
    noncurrent_element_ids: Sequence[int] = (),
) -> Optional[NativeRotationStateStore]:
    """Reconstruct one node-shared native history from model-bound states.

    Redundant per-element operator copies must agree exactly at a shared node.
    In the absence of such history, only a zero rotational coordinate is
    unambiguous; a nonzero additive coordinate cannot reconstruct a finite
    multiplicative path and therefore fails closed.
    """

    mesh = getattr(model, "mesh", None)
    elements = getattr(mesh, "elements", None)
    nodes = getattr(mesh, "nodes", None)
    if not isinstance(elements, Mapping) or not isinstance(nodes, Mapping):
        raise NativeRotationValidationError(
            "Native rotation setup requires explicit model element and node mappings"
        )
    noncurrent = {int(value) for value in noncurrent_element_ids}
    unknown_noncurrent = noncurrent.difference(int(value) for value in elements)
    if unknown_noncurrent:
        raise NativeRotationValidationError(
            "Native noncurrent element IDs are not present in the model"
        )
    native_elements = tuple(
        (int(element_id), element)
        for element_id, element in elements.items()
        if bool(getattr(element, "formulation_native_total_lagrangian", False))
        and int(element_id) not in noncurrent
    )
    if not native_elements:
        return None
    full = np.asarray(committed_full_displacement, dtype=np.float64)
    total_dofs = int(getattr(getattr(mesh, "dof_manager", None), "total_dofs", -1))
    if full.ndim != 1 or full.size != total_dofs or not np.all(np.isfinite(full)):
        raise NativeRotationValidationError(
            "Native rotation setup requires the complete finite committed displacement"
        )

    native_node_ids: set[int] = set()
    rotation_by_node: Dict[int, np.ndarray] = {}
    element_bindings: Dict[int, _NativeElementStateBinding] = {}
    for element_id, element in native_elements:
        element_node_ids = tuple(int(value) for value in getattr(element, "node_ids", ()))
        if not element_node_ids or len(set(element_node_ids)) != len(element_node_ids):
            raise NativeRotationValidationError(
                f"Native element {element_id} has invalid connectivity"
            )
        native_node_ids.update(element_node_ids)
        try:
            dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        except Exception as exc:
            raise NativeRotationValidationError(
                f"Native element {element_id} has no valid global DOF mapping"
            ) from exc
        expected_size = 6 * len(element_node_ids)
        if dof_mapping.shape != (expected_size,):
            raise NativeRotationValidationError(
                f"Native element {element_id} must expose six nodal DOFs per node"
            )
        reference_provider = getattr(element, "native_reference_directors", None)
        if not callable(reference_provider):
            raise NativeRotationValidationError(
                f"Native element {element_id} does not expose reference directors"
            )
        try:
            reference_directors = np.asarray(
                reference_provider(mesh), dtype=np.float64
            )
        except Exception as exc:
            raise NativeRotationValidationError(
                f"Native element {element_id} reference directors are unavailable"
            ) from exc
        if (
            reference_directors.shape != (len(element_node_ids), 3)
            or not np.all(np.isfinite(reference_directors))
        ):
            raise NativeRotationValidationError(
                f"Native element {element_id} reference directors are incompatible"
            )
        state = committed_states.get(element_id)
        native_state_keys = {
            "committed_total_u",
            "reference_corner_directors",
            "committed_nodal_rotation_matrices",
            "committed_director_triads",
        }
        state_consistency_required = bool(
            str(getattr(element, "formulation_id", ""))
            == "E4_PL_QUALIFIED_S3_COMPANION_V1"
            or getattr(element, "native_state_consistency_required", False)
        )
        if state_consistency_required:
            if not isinstance(state, Mapping):
                raise NativeRotationValidationError(
                    f"Native element {element_id} requires a committed state mapping"
                )
            missing = sorted(native_state_keys.difference(state))
            if missing:
                raise NativeRotationValidationError(
                    f"Native element {element_id} committed state is missing "
                    + ", ".join(missing)
                )
            stored_reference = np.asarray(
                state["reference_corner_directors"], dtype=np.float64
            )
            if (
                stored_reference.shape != reference_directors.shape
                or not np.all(np.isfinite(stored_reference))
                or not np.array_equal(stored_reference, reference_directors)
            ):
                raise NativeRotationValidationError(
                    f"Native element {element_id} reference_corner_directors "
                    "disagree with the model binding"
                )
        if isinstance(state, Mapping) and "committed_total_u" in state:
            stored_total = np.asarray(state["committed_total_u"], dtype=np.float64)
            if (
                stored_total.shape != (expected_size,)
                or not np.all(np.isfinite(stored_total))
                or not np.array_equal(stored_total, full[dof_mapping])
            ):
                raise NativeRotationValidationError(
                    f"Native element {element_id} committed_total_u disagrees with "
                    "the solver's committed global displacement"
                )
        stored_rotations = None
        if isinstance(state, Mapping) and "committed_nodal_rotation_matrices" in state:
            stored_rotations = validate_proper_rotation_matrices(
                state["committed_nodal_rotation_matrices"],
                name=(
                    f"element[{element_id}].committed_nodal_rotation_matrices"
                ),
            )
            if stored_rotations.shape != (len(element_node_ids), 3, 3):
                raise NativeRotationValidationError(
                    f"Native element {element_id} stored rotations do not match connectivity"
                )
        if state_consistency_required:
            assert isinstance(state, Mapping)
            assert stored_rotations is not None
            triads = np.asarray(
                state["committed_director_triads"], dtype=np.float64
            )
            expected_directors = np.einsum(
                "nij,nj->ni", stored_rotations, reference_directors
            )
            if (
                triads.ndim != 3
                or triads.shape[0] < len(element_node_ids)
                or triads.shape[1:] != (3, 3)
                or not np.all(np.isfinite(triads))
                or not np.allclose(
                    triads[: len(element_node_ids), :, 2],
                    expected_directors,
                    rtol=0.0,
                    atol=_NATIVE_DIRECTOR_CONSISTENCY_TOLERANCE,
                )
            ):
                raise NativeRotationValidationError(
                    f"Native element {element_id} corner director normals disagree "
                    "with its committed rotation history"
                )
        material_validator = None
        material_protocol = getattr(element, "native_material_state_protocol", None)
        if material_protocol is not None:
            provider = getattr(element, "create_native_material_validator", None)
            if material_protocol != _NATIVE_MATERIAL_PROTOCOL or not callable(provider):
                raise NativeRotationValidationError("Unsupported native material validation protocol")
            if state_consistency_required:
                raise NativeRotationValidationError("Native material protocol cannot replace the S3 state protocol")
            material_validator = provider(mesh)
            if type(material_validator) is not NativeMaterialValidator:
                raise NativeRotationValidationError("Exact native material validator required")
        element_bindings[element_id] = _NativeElementStateBinding(
            node_ids=element_node_ids,
            dof_mapping=dof_mapping,
            reference_directors=reference_directors,
            state_consistency_required=state_consistency_required,
            material_validator=material_validator,
        )
        for local_node, node_id in enumerate(element_node_ids):
            if node_id not in nodes:
                raise NativeRotationValidationError(
                    f"Native element {element_id} references missing node {node_id}"
                )
            node_dofs = np.asarray(getattr(nodes[node_id], "dofs", ()), dtype=np.intp)
            if node_dofs.size < 6:
                raise NativeRotationValidationError(
                    f"Native node {node_id} does not expose six shell DOFs"
                )
            if stored_rotations is None:
                if np.any(full[node_dofs[3:6]] != 0.0):
                    raise NativeRotationValidationError(
                        f"Native node {node_id} has nonzero committed rotation "
                        "coordinates but no multiplicative rotation history"
                    )
                candidate = np.eye(3, dtype=np.float64)
            else:
                candidate = np.asarray(stored_rotations[local_node], dtype=np.float64)
            previous = rotation_by_node.get(node_id)
            if previous is not None and not np.array_equal(previous, candidate):
                raise NativeRotationValidationError(
                    f"Native shared node {node_id} has conflicting committed rotation copies"
                )
            rotation_by_node[node_id] = candidate.copy()

    coordinate_node_ids = tuple(sorted(int(value) for value in nodes))
    coordinate_rows = {
        node_id: row for row, node_id in enumerate(coordinate_node_ids)
    }
    full_coordinates = _model_full_coordinates(
        model,
        full,
        coordinate_node_ids,
    )
    rotational_dofs = {
        node_id: tuple(int(value) for value in nodes[node_id].dofs[3:6])
        for node_id in sorted(native_node_ids)
    }
    result = create_native_rotation_state_store(
        tuple(sorted(native_node_ids)),
        rotational_dofs=rotational_dofs,
        coordinate_rows=coordinate_rows,
        coordinate_node_ids=coordinate_node_ids,
        committed_full_displacement=full,
        committed_full_coordinates=full_coordinates,
        committed_rotation_matrices=rotation_by_node,
    )
    assert result is not None
    for element_id, binding in element_bindings.items():
        if binding.material_validator is not None:
            binding.material_validator.validate(
                _owned_copy(committed_states.get(element_id)), previous_state=None,
                native_view=result.element_view(element_id, binding.node_ids, binding.reference_directors),
                displacement=np.array(full[binding.dof_mapping], copy=True), phase="committed",
            )
    setattr(
        result,
        _NATIVE_ELEMENT_BINDINGS_ATTRIBUTE,
        MappingProxyType(dict(element_bindings)),
    )
    return result


def begin_state_evaluation(
    committed_states: Mapping[int, Any],
    *,
    model: Any = None,
    displacements: Any = None,
) -> Optional[StateTrialToken]:
    """Begin a fresh candidate when ``committed_states`` is persistent."""

    if isinstance(committed_states, NonlinearStateStore):
        full_coordinates = None
        if committed_states.has_native_rotations:
            if model is None or displacements is None:
                raise StateTransactionError(
                    "Native state evaluation requires model and full displacements"
                )
            full_coordinates = native_trial_full_coordinates(
                committed_states,
                model,
                displacements,
            )
        return committed_states.replace_trial(
            full_displacement=displacements,
            full_coordinates=full_coordinates,
        )
    return None


def finish_state_evaluation(
    committed_states: Mapping[int, Any],
    token: Optional[StateTrialToken],
    legacy_trial_states: Mapping[int, Any],
) -> Mapping[int, Any]:
    """Route scalar fallbacks into a store and return a lazy trial mapping."""

    if not isinstance(committed_states, NonlinearStateStore):
        # Every assembler already owns a fresh trial dictionary.  Preserve
        # the legacy zero-copy return path when persistent storage is not in
        # use; copying O(elements) here is measurable on mature elastic
        # assembly and provides no additional ownership guarantee.
        return legacy_trial_states
    if token is None:
        raise StateTransactionError("Persistent state evaluation has no trial token")
    committed_states.set_trial_states(token, legacy_trial_states)
    return committed_states.trial_view(token)


def commit_state_candidate(
    committed_states: Mapping[int, Any],
    candidate_states: Mapping[int, Any],
    *,
    accepted_full_displacement: Any = None,
    model: Any = None,
) -> Mapping[int, Any]:
    """Commit a lazy candidate or preserve the legacy mapping assignment."""

    if isinstance(committed_states, NonlinearStateStore):
        if not isinstance(candidate_states, NonlinearStateTrialView):
            raise StateTransactionError(
                "Persistent committed state requires a persistent trial candidate"
            )
        if candidate_states.store is not committed_states:
            raise StateTransactionError("Candidate belongs to another state store")
        accepted_coordinates = None
        if committed_states.has_native_rotations:
            if model is None or accepted_full_displacement is None:
                raise StateTransactionError(
                    "Native state commit requires model and accepted full displacements"
                )
            accepted_coordinates = native_trial_full_coordinates(
                committed_states,
                model,
                accepted_full_displacement,
            )
        return candidate_states.commit(
            accepted_full_displacement=accepted_full_displacement,
            accepted_full_coordinates=accepted_coordinates,
        )
    return candidate_states


def discard_active_state_candidate(committed_states: Mapping[int, Any]) -> None:
    """Discard a pending persistent candidate, if any."""

    if isinstance(committed_states, NonlinearStateStore) and committed_states.has_active_trial:
        committed_states.discard_trial(committed_states.active_trial_token())


def materialize_state_mapping(
    states: Mapping[int, Any],
    *,
    policy: StateMaterializationPolicy | str = StateMaterializationPolicy.EXPLICIT,
) -> Dict[int, Any]:
    """Return an owned public mapping while preserving ordinary dict behavior."""

    if isinstance(states, NonlinearStateStore):
        discard_active_state_candidate(states)
        return states.materialize_owned(policy=policy)
    if isinstance(states, NonlinearStateTrialView):
        return states.materialize_owned(policy=policy)
    return _owned_copy(dict(states))


__all__ = [
    "ImmutableStateSidecarError",
    "begin_state_evaluation",
    "commit_state_candidate",
    "create_model_native_rotation_store",
    "discard_active_state_candidate",
    "finish_state_evaluation",
    "materialize_state_mapping",
    "native_trial_full_coordinates",
    "NativeElementRotationView",
    "NativeRotationStateStore",
    "NonlinearStateError",
    "NonlinearStateStore",
    "NonlinearStateTrialView",
    "PersistentStateEligibilityError",
    "ShellStateArrays",
    "ShellStateBatch",
    "ShellStateLayout",
    "StaleStateTokenError",
    "StateMaterializationPolicy",
    "StateTransactionError",
    "StateTrialToken",
    "require_no_native_total_lagrangian_elements",
]
