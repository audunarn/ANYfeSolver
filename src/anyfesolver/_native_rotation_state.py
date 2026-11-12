"""Solver-owned multiplicative rotations for formulation-native elements.

The ordinary ANYfeSolver displacement vector retains additive rotational
coordinates for compatibility with existing elements.  A formulation-native
large-rotation element needs a different history variable: one proper rotation
operator shared by every native element incident on a node.  This module owns
that history without changing the public displacement or element APIs.

There is deliberately no empty-store object.  The factory returns ``None``
when a model has no native nodes, so established Q4 and legacy models pay no
allocation or transaction cost.  A non-empty store permits exactly one trial
at a time.  Every candidate is reconstructed from the committed base as

``Q_trial = Exp(theta_trial - theta_committed) @ Q_committed``.

Rejected candidates never modify committed state.  Commit additionally binds
the candidate to the exact full displacement and coordinate arrays accepted by
the solver; committing element-local or stale kinematics is therefore not
possible.
"""

from __future__ import annotations

import threading
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


_SO3_TOLERANCE = 1.0e-10


class NativeRotationStateError(RuntimeError):
    """Base class for solver-owned native-rotation failures."""


class NativeRotationValidationError(NativeRotationStateError, ValueError):
    """Raised when coordinates, DOFs, or rotations violate the contract."""


class NativeRotationTransactionError(NativeRotationStateError):
    """Raised when native-rotation transaction ordering is invalid."""


class StaleNativeRotationTokenError(NativeRotationTransactionError):
    """Raised when a trial capability is stale or belongs to another store."""


class AcceptedConfigurationMismatchError(NativeRotationTransactionError):
    """Raised when commit does not name the exact evaluated configuration."""


def _immutable_float_array(values: Any, *, name: str) -> np.ndarray:
    """Return a finite float64 array backed by immutable bytes."""

    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise NativeRotationValidationError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(array)
    # A view backed by ``bytes`` cannot be made writable again with setflags().
    payload = contiguous.tobytes(order="C")
    return np.frombuffer(payload, dtype=np.float64).reshape(contiguous.shape)


def _coerce_full_displacement(values: Any, *, name: str) -> np.ndarray:
    result = _immutable_float_array(values, name=name)
    if result.ndim != 1:
        raise NativeRotationValidationError(f"{name} must be one-dimensional")
    return result


def _coerce_full_coordinates(values: Any, *, name: str) -> np.ndarray:
    result = _immutable_float_array(values, name=name)
    if result.ndim != 2 or result.shape[1] != 3:
        raise NativeRotationValidationError(f"{name} must have shape (n, 3)")
    return result


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = (float(value) for value in vector)
    return np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=np.float64,
    )


def rotation_exponential(rotation_vector: Any) -> np.ndarray:
    """Return the proper rotation ``Exp(skew(rotation_vector))``.

    The small-angle coefficients avoid normalizing by a vanishing angle and
    retain the quadratic term required by nonlinear trial construction.
    """

    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise NativeRotationValidationError(
            "rotation_vector must be a finite three-component vector"
        )
    angle_squared = float(vector @ vector)
    cross = _skew(vector)
    if angle_squared < 1.0e-16:
        angle_fourth = angle_squared * angle_squared
        sine_coefficient = 1.0 - angle_squared / 6.0 + angle_fourth / 120.0
        cosine_coefficient = 0.5 - angle_squared / 24.0 + angle_fourth / 720.0
    else:
        angle = float(np.sqrt(angle_squared))
        sine_coefficient = float(np.sin(angle) / angle)
        cosine_coefficient = float((1.0 - np.cos(angle)) / angle_squared)
    return (
        np.eye(3, dtype=np.float64)
        + sine_coefficient * cross
        + cosine_coefficient * (cross @ cross)
    )


def validate_proper_rotation_matrices(
    matrices: Any,
    *,
    name: str = "rotation_matrices",
    tolerance: float = _SO3_TOLERANCE,
) -> np.ndarray:
    """Validate finite matrices in SO(3) and return an immutable copy.

    A single ``(3, 3)`` matrix and a stack ``(n, 3, 3)`` are accepted.  The
    values are not projected or silently repaired: an improper or materially
    non-orthogonal input is rejected.
    """

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    result = _immutable_float_array(matrices, name=name)
    single = result.ndim == 2
    stack = result.reshape(1, 3, 3) if single and result.shape == (3, 3) else result
    if stack.ndim != 3 or stack.shape[1:] != (3, 3):
        raise NativeRotationValidationError(
            f"{name} must have shape (3, 3) or (n, 3, 3)"
        )
    identity = np.eye(3, dtype=np.float64)
    for index, matrix in enumerate(stack):
        orthogonality_error = float(np.max(np.abs(matrix.T @ matrix - identity)))
        determinant = float(np.linalg.det(matrix))
        if (
            determinant <= 0.0
            or orthogonality_error > tolerance
            or abs(determinant - 1.0) > tolerance
        ):
            raise NativeRotationValidationError(
                f"{name}[{index}] is not a proper SO(3) matrix "
                f"(orthogonality_error={orthogonality_error:.3e}, "
                f"determinant={determinant:.17g})"
            )
    return result


@dataclass(frozen=True, slots=True)
class NativeRotationTrialToken:
    """Opaque generation-checked capability for one native-rotation trial."""

    generation: int
    serial: int
    _owner: object = field(repr=False, compare=False, hash=False)


@dataclass(frozen=True, slots=True)
class NativeElementRotationView:
    """Immutable native-node data in one element's connectivity order.

    Rotation matrices are node-shared.  Reference directors are supplied by
    the element and copied into this view, so two elements meeting at a crease
    retain distinct reference directors while observing the same nodal
    rotation operator.
    """

    element_id: Hashable
    node_ids: tuple[int, ...]
    committed_coordinates: np.ndarray
    trial_coordinates: np.ndarray
    coordinate_increment: np.ndarray
    committed_rotation_coordinates: np.ndarray
    trial_rotation_coordinates: np.ndarray
    rotation_coordinate_increment: np.ndarray
    committed_rotation_matrices: np.ndarray
    trial_rotation_matrices: np.ndarray
    reference_directors: np.ndarray
    committed_directors: np.ndarray
    trial_directors: np.ndarray
    generation: int
    trial_serial: Optional[int]

    @property
    def coordinates(self) -> np.ndarray:
        """Compatibility alias for the selected (committed or trial) coordinates."""

        return self.trial_coordinates

    @property
    def rotation_matrices(self) -> np.ndarray:
        """Compatibility alias for the selected (committed or trial) operators."""

        return self.trial_rotation_matrices


@dataclass(frozen=True, slots=True)
class _ActiveNativeRotationTrial:
    token: NativeRotationTrialToken
    full_displacement: np.ndarray
    full_coordinates: np.ndarray
    native_coordinates: np.ndarray
    rotation_matrices: np.ndarray


class NativeRotationCandidate:
    """Exception-safe facade over one active store transaction."""

    __slots__ = ("_store", "_token", "_closed")

    def __init__(
        self,
        store: "NativeRotationStateStore",
        token: NativeRotationTrialToken,
    ) -> None:
        self._store = store
        self._token = token
        self._closed = False

    @property
    def token(self) -> NativeRotationTrialToken:
        return self._token

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def element_view(
        self,
        element_id: Hashable,
        node_ids: Sequence[int],
        reference_directors: Any,
    ) -> NativeElementRotationView:
        return self._store.element_view(
            element_id,
            node_ids,
            reference_directors,
            trial_token=self._token,
        )

    def commit(
        self,
        accepted_full_displacement: Any,
        accepted_full_coordinates: Any,
    ) -> int:
        generation = self._store.commit_trial(
            self._token,
            accepted_full_displacement,
            accepted_full_coordinates,
        )
        self._closed = True
        return generation

    def discard(self) -> None:
        self._store.discard_trial(self._token)
        self._closed = True

    def __enter__(self) -> "NativeRotationCandidate":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if not self._closed:
            self._store._discard_if_active(self._token)
            self._closed = True
        return False


class NativeRotationStateStore:
    """Committed and trial SO(3) state for a non-empty set of native nodes."""

    def __init__(
        self,
        native_node_ids: Sequence[int],
        *,
        rotational_dofs: Mapping[int, Sequence[int]],
        coordinate_rows: Mapping[int, int],
        committed_full_displacement: Any,
        committed_full_coordinates: Any,
        committed_rotation_matrices: Mapping[int, Any] | Any | None = None,
        coordinate_node_ids: Sequence[int] | None = None,
    ) -> None:
        normalized_ids = tuple(sorted(int(value) for value in native_node_ids))
        if not normalized_ids:
            raise NativeRotationValidationError(
                "NativeRotationStateStore requires at least one native node; "
                "use create_native_rotation_state_store for optional creation"
            )
        if len(set(normalized_ids)) != len(normalized_ids):
            raise NativeRotationValidationError("native_node_ids must be unique")
        self._node_ids = normalized_ids
        self._node_index = {
            node_id: index for index, node_id in enumerate(self._node_ids)
        }
        self._committed_full_displacement = _coerce_full_displacement(
            committed_full_displacement,
            name="committed_full_displacement",
        )
        self._committed_full_coordinates = _coerce_full_coordinates(
            committed_full_coordinates,
            name="committed_full_coordinates",
        )
        if coordinate_node_ids is None:
            normalized_coordinate_node_ids = tuple(
                range(self._committed_full_coordinates.shape[0])
            )
        else:
            normalized_coordinate_node_ids = tuple(
                int(value) for value in coordinate_node_ids
            )
        if (
            len(normalized_coordinate_node_ids)
            != self._committed_full_coordinates.shape[0]
            or len(set(normalized_coordinate_node_ids))
            != len(normalized_coordinate_node_ids)
        ):
            raise NativeRotationValidationError(
                "coordinate_node_ids must uniquely identify every full-coordinate row"
            )
        self._coordinate_node_ids = normalized_coordinate_node_ids

        dof_rows: list[tuple[int, int, int]] = []
        coordinate_indices: list[int] = []
        for node_id in self._node_ids:
            try:
                node_dofs = tuple(int(value) for value in rotational_dofs[node_id])
            except KeyError as exc:
                raise NativeRotationValidationError(
                    f"rotational_dofs is missing native node {node_id}"
                ) from exc
            if len(node_dofs) != 3 or len(set(node_dofs)) != 3:
                raise NativeRotationValidationError(
                    f"native node {node_id} must have three distinct rotational DOFs"
                )
            if min(node_dofs) < 0 or max(node_dofs) >= self._committed_full_displacement.size:
                raise NativeRotationValidationError(
                    f"rotational DOFs for native node {node_id} are out of bounds"
                )
            dof_rows.append(node_dofs)
            try:
                coordinate_row = int(coordinate_rows[node_id])
            except KeyError as exc:
                raise NativeRotationValidationError(
                    f"coordinate_rows is missing native node {node_id}"
                ) from exc
            if not 0 <= coordinate_row < self._committed_full_coordinates.shape[0]:
                raise NativeRotationValidationError(
                    f"coordinate row for native node {node_id} is out of bounds"
                )
            coordinate_indices.append(coordinate_row)
        if len({value for row in dof_rows for value in row}) != 3 * len(dof_rows):
            raise NativeRotationValidationError(
                "native nodes must not share rotational displacement DOFs"
            )
        if len(set(coordinate_indices)) != len(coordinate_indices):
            raise NativeRotationValidationError(
                "native nodes must not share full-coordinate rows"
            )
        self._rotational_dofs = np.asarray(dof_rows, dtype=np.intp)
        self._coordinate_rows = np.asarray(coordinate_indices, dtype=np.intp)

        if committed_rotation_matrices is None:
            initial = np.repeat(
                np.eye(3, dtype=np.float64)[None, :, :],
                len(self._node_ids),
                axis=0,
            )
        elif isinstance(committed_rotation_matrices, Mapping):
            try:
                initial = np.asarray(
                    [committed_rotation_matrices[node_id] for node_id in self._node_ids],
                    dtype=np.float64,
                )
            except KeyError as exc:
                raise NativeRotationValidationError(
                    f"committed_rotation_matrices is missing native node {int(exc.args[0])}"
                ) from exc
        else:
            initial = np.asarray(committed_rotation_matrices, dtype=np.float64)
        if initial.shape != (len(self._node_ids), 3, 3):
            raise NativeRotationValidationError(
                "committed_rotation_matrices must have shape (native_node_count, 3, 3)"
            )
        self._committed_rotation_matrices = validate_proper_rotation_matrices(
            initial,
            name="committed_rotation_matrices",
        )
        self._committed_native_coordinates = _immutable_float_array(
            self._committed_full_coordinates[self._coordinate_rows],
            name="committed_native_coordinates",
        )

        self._owner = object()
        self._generation = 0
        self._serial = 0
        self._active: _ActiveNativeRotationTrial | None = None
        self._lock = threading.RLock()

    @property
    def node_ids(self) -> tuple[int, ...]:
        return self._node_ids

    @property
    def generation(self) -> int:
        with self._lock:
            return int(self._generation)

    @property
    def committed_coordinates(self) -> np.ndarray:
        with self._lock:
            return self._committed_native_coordinates

    @property
    def committed_rotation_matrices(self) -> np.ndarray:
        with self._lock:
            return self._committed_rotation_matrices

    @property
    def committed_full_displacement(self) -> np.ndarray:
        with self._lock:
            return self._committed_full_displacement

    @property
    def committed_full_coordinates(self) -> np.ndarray:
        with self._lock:
            return self._committed_full_coordinates

    @property
    def coordinate_node_ids(self) -> tuple[int, ...]:
        return self._coordinate_node_ids

    @property
    def has_active_trial(self) -> bool:
        with self._lock:
            return self._active is not None

    def _require_active(
        self,
        token: NativeRotationTrialToken,
    ) -> _ActiveNativeRotationTrial:
        active = self._active
        valid = bool(
            isinstance(token, NativeRotationTrialToken)
            and token._owner is self._owner
            and token.generation == self._generation
            and active is not None
            and active.token is token
        )
        if not valid:
            raise StaleNativeRotationTokenError(
                "Native-rotation token is stale, belongs to another store, "
                "or is not active"
            )
        return active

    def begin_trial(
        self,
        trial_full_displacement: Any,
        trial_full_coordinates: Any,
    ) -> NativeRotationTrialToken:
        """Create the sole trial, always relative to the committed base."""

        with self._lock:
            if self._active is not None:
                raise NativeRotationTransactionError(
                    "A native-rotation trial is already active"
                )
            displacement = _coerce_full_displacement(
                trial_full_displacement,
                name="trial_full_displacement",
            )
            coordinates = _coerce_full_coordinates(
                trial_full_coordinates,
                name="trial_full_coordinates",
            )
            if displacement.shape != self._committed_full_displacement.shape:
                raise NativeRotationValidationError(
                    "trial_full_displacement shape differs from the committed full vector"
                )
            if coordinates.shape != self._committed_full_coordinates.shape:
                raise NativeRotationValidationError(
                    "trial_full_coordinates shape differs from the committed coordinate array"
                )
            committed_theta = self._committed_full_displacement[
                self._rotational_dofs
            ]
            trial_theta = displacement[self._rotational_dofs]
            rotations = np.empty_like(self._committed_rotation_matrices)
            for index, delta in enumerate(trial_theta - committed_theta):
                rotations[index] = (
                    rotation_exponential(delta)
                    @ self._committed_rotation_matrices[index]
                )
            rotations = validate_proper_rotation_matrices(
                rotations,
                name="trial_rotation_matrices",
            )
            native_coordinates = _immutable_float_array(
                coordinates[self._coordinate_rows],
                name="trial_native_coordinates",
            )
            self._serial += 1
            token = NativeRotationTrialToken(
                self._generation,
                self._serial,
                self._owner,
            )
            self._active = _ActiveNativeRotationTrial(
                token=token,
                full_displacement=displacement,
                full_coordinates=coordinates,
                native_coordinates=native_coordinates,
                rotation_matrices=rotations,
            )
            return token

    def candidate(
        self,
        trial_full_displacement: Any,
        trial_full_coordinates: Any,
    ) -> NativeRotationCandidate:
        """Return an exception-safe context facade for a new trial."""

        return NativeRotationCandidate(
            self,
            self.begin_trial(trial_full_displacement, trial_full_coordinates),
        )

    def validate_trial_token(self, token: NativeRotationTrialToken) -> None:
        with self._lock:
            self._require_active(token)

    def element_view(
        self,
        element_id: Hashable,
        node_ids: Sequence[int],
        reference_directors: Any,
        *,
        trial_token: NativeRotationTrialToken | None = None,
    ) -> NativeElementRotationView:
        """Return committed or candidate state in one element's node order."""

        with self._lock:
            ordered_node_ids = tuple(int(value) for value in node_ids)
            if not ordered_node_ids or len(set(ordered_node_ids)) != len(ordered_node_ids):
                raise NativeRotationValidationError(
                    "element native node_ids must be non-empty and unique"
                )
            try:
                indices = np.asarray(
                    [self._node_index[node_id] for node_id in ordered_node_ids],
                    dtype=np.intp,
                )
            except KeyError as exc:
                raise NativeRotationValidationError(
                    f"element references non-native node {int(exc.args[0])}"
                ) from exc
            directors = _immutable_float_array(
                reference_directors,
                name="reference_directors",
            )
            if directors.shape != (len(ordered_node_ids), 3):
                raise NativeRotationValidationError(
                    "reference_directors must have shape (element_native_node_count, 3)"
                )
            if trial_token is None:
                trial_coordinates = self._committed_native_coordinates
                trial_rotations = self._committed_rotation_matrices
                trial_full_displacement = self._committed_full_displacement
                serial: Optional[int] = None
            else:
                active = self._require_active(trial_token)
                trial_coordinates = active.native_coordinates
                trial_rotations = active.rotation_matrices
                trial_full_displacement = active.full_displacement
                serial = int(trial_token.serial)
            committed_coordinates = self._committed_native_coordinates[indices]
            selected_coordinates = trial_coordinates[indices]
            committed_rotation_coordinates = self._committed_full_displacement[
                self._rotational_dofs[indices]
            ]
            trial_rotation_coordinates = trial_full_displacement[
                self._rotational_dofs[indices]
            ]
            committed_rotations = self._committed_rotation_matrices[indices]
            selected_rotations = trial_rotations[indices]
            committed_directors = np.einsum(
                "nij,nj->ni", committed_rotations, directors
            )
            trial_directors = np.einsum(
                "nij,nj->ni", selected_rotations, directors
            )
            return NativeElementRotationView(
                element_id=element_id,
                node_ids=ordered_node_ids,
                committed_coordinates=_immutable_float_array(
                    committed_coordinates,
                    name="element_committed_coordinates",
                ),
                trial_coordinates=_immutable_float_array(
                    selected_coordinates,
                    name="element_trial_coordinates",
                ),
                coordinate_increment=_immutable_float_array(
                    selected_coordinates - committed_coordinates,
                    name="element_coordinate_increment",
                ),
                committed_rotation_coordinates=_immutable_float_array(
                    committed_rotation_coordinates,
                    name="element_committed_rotation_coordinates",
                ),
                trial_rotation_coordinates=_immutable_float_array(
                    trial_rotation_coordinates,
                    name="element_trial_rotation_coordinates",
                ),
                rotation_coordinate_increment=_immutable_float_array(
                    trial_rotation_coordinates - committed_rotation_coordinates,
                    name="element_rotation_coordinate_increment",
                ),
                committed_rotation_matrices=validate_proper_rotation_matrices(
                    committed_rotations,
                    name="element_committed_rotation_matrices",
                ),
                trial_rotation_matrices=validate_proper_rotation_matrices(
                    selected_rotations,
                    name="element_trial_rotation_matrices",
                ),
                reference_directors=directors,
                committed_directors=_immutable_float_array(
                    committed_directors,
                    name="element_committed_directors",
                ),
                trial_directors=_immutable_float_array(
                    trial_directors,
                    name="element_trial_directors",
                ),
                generation=int(self._generation),
                trial_serial=serial,
            )

    @staticmethod
    def _configuration_matches(expected: np.ndarray, accepted: Any) -> bool:
        try:
            actual = np.asarray(accepted, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return bool(
            actual.shape == expected.shape
            and np.all(np.isfinite(actual))
            and np.array_equal(actual, expected)
        )

    def commit_trial(
        self,
        token: NativeRotationTrialToken,
        accepted_full_displacement: Any,
        accepted_full_coordinates: Any,
    ) -> int:
        """Commit only when the solver accepted this exact full candidate."""

        with self._lock:
            active = self._validate_commit_configuration(
                token,
                accepted_full_displacement,
                accepted_full_coordinates,
            )
            self._committed_full_displacement = active.full_displacement
            self._committed_full_coordinates = active.full_coordinates
            self._committed_native_coordinates = active.native_coordinates
            self._committed_rotation_matrices = active.rotation_matrices
            self._generation += 1
            self._active = None
            return int(self._generation)

    def _validate_commit_configuration(
        self,
        token: NativeRotationTrialToken,
        accepted_full_displacement: Any,
        accepted_full_coordinates: Any,
    ) -> _ActiveNativeRotationTrial:
        active = self._require_active(token)
        if not self._configuration_matches(
            active.full_displacement,
            accepted_full_displacement,
        ):
            raise AcceptedConfigurationMismatchError(
                "accepted full displacement does not exactly match the evaluated trial"
            )
        if not self._configuration_matches(
            active.full_coordinates,
            accepted_full_coordinates,
        ):
            raise AcceptedConfigurationMismatchError(
                "accepted full coordinates do not exactly match the evaluated trial"
            )
        return active

    def validate_commit_configuration(
        self,
        token: NativeRotationTrialToken,
        accepted_full_displacement: Any,
        accepted_full_coordinates: Any,
    ) -> None:
        """Validate an atomic parent transaction without changing history."""

        with self._lock:
            self._validate_commit_configuration(
                token,
                accepted_full_displacement,
                accepted_full_coordinates,
            )

    def commit(
        self,
        token: NativeRotationTrialToken,
        accepted_full_displacement: Any,
        accepted_full_coordinates: Any,
    ) -> int:
        return self.commit_trial(
            token,
            accepted_full_displacement,
            accepted_full_coordinates,
        )

    def discard_trial(self, token: NativeRotationTrialToken) -> None:
        """Reject the active candidate without changing committed state."""

        with self._lock:
            self._require_active(token)
            self._active = None

    def discard(self, token: NativeRotationTrialToken) -> None:
        self.discard_trial(token)

    def _discard_if_active(self, token: NativeRotationTrialToken) -> None:
        """Best-effort context cleanup that never masks an outer exception."""

        with self._lock:
            if self._active is not None and self._active.token is token:
                self._active = None


def create_native_rotation_state_store(
    native_node_ids: Sequence[int],
    *,
    rotational_dofs: Mapping[int, Sequence[int]] | None = None,
    coordinate_rows: Mapping[int, int] | None = None,
    committed_full_displacement: Any = None,
    committed_full_coordinates: Any = None,
    committed_rotation_matrices: Mapping[int, Any] | Any | None = None,
    coordinate_node_ids: Sequence[int] | None = None,
) -> NativeRotationStateStore | None:
    """Create native history, or return ``None`` without touching other inputs.

    The early empty return is the Q4/legacy fast path.  Required construction
    inputs are validated only when at least one native node exists.
    """

    if len(native_node_ids) == 0:
        return None
    if rotational_dofs is None or coordinate_rows is None:
        raise NativeRotationValidationError(
            "rotational_dofs and coordinate_rows are required for native nodes"
        )
    return NativeRotationStateStore(
        native_node_ids,
        rotational_dofs=rotational_dofs,
        coordinate_rows=coordinate_rows,
        committed_full_displacement=committed_full_displacement,
        committed_full_coordinates=committed_full_coordinates,
        committed_rotation_matrices=committed_rotation_matrices,
        coordinate_node_ids=coordinate_node_ids,
    )


__all__ = [
    "AcceptedConfigurationMismatchError",
    "NativeElementRotationView",
    "NativeRotationCandidate",
    "NativeRotationStateError",
    "NativeRotationStateStore",
    "NativeRotationTransactionError",
    "NativeRotationTrialToken",
    "NativeRotationValidationError",
    "StaleNativeRotationTokenError",
    "create_native_rotation_state_store",
    "rotation_exponential",
    "validate_proper_rotation_matrices",
]
