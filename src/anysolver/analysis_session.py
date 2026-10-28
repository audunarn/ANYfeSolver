"""Bounded, revision-aware reuse for repeated analyses.

The session owns immutable structural matrices, the affine constraint plan and
factorizations for one :class:`~anysolver.fe_core.FEModel`.  Load values are
deliberately excluded from structural cache keys.  Callers may therefore reuse
one session across static load cases, modal analysis and linear buckling while
ordinary one-off solves retain their existing behaviour.

The public model objects remain mutable for backwards compatibility.  In
addition to mesh revisions, constraint plans use canonical equation
fingerprints so direct edits to prescribed values or MPC coefficients cannot
silently reuse a stale transformation.
"""

from __future__ import annotations

import threading
import time
import weakref
import uuid
import hashlib
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse

from .constraint_audit import require_valid_constraints
from .linalg import FactorizationCache
from .activity import ElementActivity, ElementActivityPolicy


def _activity_callable_metadata(function: Any) -> Tuple[Any, ...]:
    return (function, function.__code__, function.__defaults__,
            tuple((function.__kwdefaults__ or {}).items()))


_ACTIVITY_EPOCH_METHODS = {
    name: (vars(ElementActivity)[name], _activity_callable_metadata(vars(ElementActivity)[name]))
    for name in ("scales", "_all_scales", "_indices_for_ids", "set_activity", "hard_delete")
}
_ACTIVITY_EPOCH_PROPERTIES = {
    name: (vars(ElementActivity)[name], _activity_callable_metadata(vars(ElementActivity)[name].fget))
    for name in ("policy", "sequence")
}


def _has_builtin_activity_epoch(provider: Any) -> bool:
    if type(provider) is not ElementActivity:
        return False
    namespace = vars(provider)
    for name, (function, metadata) in _ACTIVITY_EPOCH_METHODS.items():
        if (name in namespace or vars(ElementActivity).get(name) is not function
                or _activity_callable_metadata(function) != metadata):
            return False
    for name, (descriptor, metadata) in _ACTIVITY_EPOCH_PROPERTIES.items():
        if (vars(ElementActivity).get(name) is not descriptor
                or _activity_callable_metadata(descriptor.fget) != metadata):
            return False
    return type(provider.policy) is ElementActivityPolicy

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


def _revision_key(model: "FEModel", *categories: str) -> Tuple[int, ...]:
    revisions = model.revision_signature()
    return tuple(int(revisions.get(category, 0)) for category in categories)


def _immutable_array(values: np.ndarray) -> np.ndarray:
    made = np.asarray(values)
    return np.frombuffer(made.tobytes(), dtype=made.dtype).reshape(made.shape)


def _immutable_csr(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    made = matrix.tocsr(copy=True)
    made.data = _immutable_array(made.data)
    made.indices = _immutable_array(made.indices)
    made.indptr = _immutable_array(made.indptr)
    return made


def _constraint_fingerprints(
    model: "FEModel",
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Return structure/value fingerprints from the qualified audit path."""

    audit = require_valid_constraints(model)
    structure = tuple(
        (
            int(equation.dependent_dof),
            str(equation.kind),
            tuple(
                (int(dof), float(coefficient))
                for dof, coefficient in equation.coefficients
            ),
        )
        for equation in audit.equations
    )
    values = tuple(float(equation.value) for equation in audit.equations)
    return structure, values


def _load_case_linear_pattern(load_case: Optional["LoadCase"]) -> tuple[Any, float] | None:
    """Return an exact proportional-load pattern and its scalar amplitude.

    Only the exact built-in dead-load container is eligible.  Gravity and
    follower pressure are deliberately excluded because their effective load
    can depend on mass or current configuration rather than one scalar.
    """

    if load_case is None:
        return (("none",), 0.0)
    from .boundary import LoadCase

    if type(load_case) is not LoadCase or load_case.gravity is not None or load_case.follower_pressure:
        return None

    numeric_values: list[float] = []

    def array_record(value: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("load case contains non-finite values")
        values = tuple(float(item) for item in array.reshape(-1))
        numeric_values.extend(values)
        return tuple(int(size) for size in array.shape), values

    nodal = tuple(
        (int(key), *array_record(value))
        for key, value in sorted(load_case.nodal_loads.items())
    )
    element = tuple(
        (int(key), *array_record(value))
        for key, value in sorted(load_case.element_loads.items())
    )
    pressure_records = []
    for key, value in sorted(load_case.pressure_loads.items()):
        shape, values = array_record(value)
        pressure_records.append((int(key), shape, values))
    masses = tuple(
        (int(key), float(value))
        for key, value in sorted(load_case.added_node_masses.items())
    )
    if any(not np.isfinite(value) for _key, value in masses):
        raise ValueError("load case contains non-finite added masses")
    amplitude = max((abs(value) for value in numeric_values), default=0.0)
    if amplitude == 0.0:
        normalized = lambda values: values
    else:
        normalized = lambda values: tuple(float(value / amplitude) for value in values)
    pattern = (
        "linear_dead_load_v1",
        tuple((key, shape, normalized(values)) for key, shape, values in nodal),
        tuple((key, shape, normalized(values)) for key, shape, values in element),
        tuple(
            (key, shape, normalized(values))
            for key, shape, values in pressure_records
        ),
        masses,
    )
    return pattern, float(amplitude)


def _sparse_nbytes(matrix: Optional[sparse.spmatrix]) -> int:
    if matrix is None:
        return 0
    csr = matrix.tocsr(copy=False)
    return int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)


@dataclass(frozen=True)
class ConstraintPlan:
    """One affine ``u = T q + u0`` plan and its reduced stiffness."""

    transformation: sparse.csr_matrix
    prescribed_offset: np.ndarray
    independent_dofs: np.ndarray
    reduced_stiffness: sparse.csr_matrix
    info: Mapping[str, Any]
    structure_key: Tuple[Any, ...]
    value_key: Tuple[Any, ...]
    stiffness_key: Tuple[Any, ...]

    @property
    def T(self) -> sparse.csr_matrix:
        return self.transformation

    @property
    def u0(self) -> np.ndarray:
        return self.prescribed_offset

    @property
    def K_red(self) -> sparse.csr_matrix:
        return self.reduced_stiffness

    def reduce_load(
        self,
        stiffness: sparse.spmatrix,
        load: np.ndarray,
    ) -> np.ndarray:
        residual = np.asarray(load, dtype=float).reshape(-1) - np.asarray(
            stiffness @ self.prescribed_offset,
            dtype=float,
        ).reshape(-1)
        return np.asarray(self.transformation.T @ residual, dtype=float).reshape(-1)


@dataclass(frozen=True)
class StructuralMatrixPlan:
    """Cached full structural matrix with its assembly provenance."""

    matrix: sparse.csr_matrix
    info: Mapping[str, Any]
    revision_key: Tuple[Any, ...]
    matrix_type: str


@dataclass(frozen=True)
class StaticOperatorPlan:
    """Prepared scaled or rigid-mode augmented static operator."""

    constraint: ConstraintPlan
    matrix: sparse.csr_matrix
    scaling: np.ndarray
    rigid_modes: np.ndarray
    handle: Any
    augmented: bool
    key: Tuple[Any, ...]
    info: Mapping[str, Any]


@dataclass(frozen=True)
class OutputSelectionPlan:
    """Selected rows of the affine transform for reduced-coordinate output."""

    full_dofs: np.ndarray
    transformation_rows: sparse.csr_matrix
    prescribed_rows: np.ndarray
    structure_key: Tuple[Any, ...]
    value_key: Tuple[Any, ...] = ()

    def reconstruct(self, reduced: np.ndarray, *, affine: bool = True) -> np.ndarray:
        values = np.asarray(self.transformation_rows @ reduced, dtype=float)
        if affine:
            if values.ndim == 1:
                values = values + self.prescribed_rows
            else:
                values = values + self.prescribed_rows[:, None]
        return values


class AnalysisSession:
    """Explicit bounded cache for repeated analyses of one live model.

    A session never owns the model. Closing it releases all retained matrices
    and its owned factorization handles; an explicitly supplied cache remains
    caller-owned. Using it after close or with a different model
    raises immediately instead of risking stale numerical data.
    """

    def __init__(
        self,
        model: "FEModel",
        *,
        max_factorizations: int = 4,
        max_output_plans: int = 8,
        factorization_cache: Optional[FactorizationCache] = None,
    ) -> None:
        if max_factorizations <= 0:
            raise ValueError("max_factorizations must be positive")
        if max_output_plans <= 0:
            raise ValueError("max_output_plans must be positive")
        self._model_ref = weakref.ref(model)
        self._model_id = id(model)
        self._lock = threading.RLock()
        self._closed = False
        self._stiffness: Optional[StructuralMatrixPlan] = None
        self._mass: Optional[StructuralMatrixPlan] = None
        self._constraint: Optional[ConstraintPlan] = None
        self._static_operator: Optional[StaticOperatorPlan] = None
        self._inertia_relief: Optional[Tuple[Tuple[Any, ...], Any]] = None
        self._reduced_mass: Optional[Tuple[Tuple[Any, ...], sparse.csr_matrix]] = None
        self._rigid_modes: Optional[Tuple[Tuple[Any, ...], np.ndarray, Dict[str, Any]]] = None
        self._load_vectors: "OrderedDict[Tuple[Any, ...], Tuple[np.ndarray, Dict[str, Any]]]" = OrderedDict()
        self._output_plans: "OrderedDict[Tuple[Any, ...], OutputSelectionPlan]" = OrderedDict()
        self._activity_snapshots: Dict[str, Tuple[Any, Tuple[Any, ...], Tuple[Any, ...], Any]] = {}
        self._activity_identity = uuid.uuid4().hex
        self._activity_generation = 0
        self._max_output_plans = int(max_output_plans)
        self._owns_factorization_cache = factorization_cache is None
        self.factorization_cache = FactorizationCache(
            name="analysis_session",
            max_entries=int(max_factorizations),
        ) if factorization_cache is None else factorization_cache
        self._counters: Counter[str] = Counter()
        self._invalidation_reasons: Counter[str] = Counter()
        self._setup_seconds = 0.0

    @property
    def model(self) -> "FEModel":
        model = self._model_ref()
        if model is None:
            raise RuntimeError("AnalysisSession model no longer exists")
        return model

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def _require_model(self, model: Optional["FEModel"] = None) -> "FEModel":
        if self._closed:
            raise RuntimeError("AnalysisSession is closed")
        owned = self.model
        if model is not None and (id(model) != self._model_id or model is not owned):
            raise ValueError("AnalysisSession belongs to a different FEModel")
        return owned

    def _require_current_structural_plan(
        self,
        plan: Optional[StructuralMatrixPlan],
        *,
        matrix_type: str,
        model: "FEModel",
    ) -> StructuralMatrixPlan:
        """Return the current session-owned structural plan or reject ``plan``.

        Revision keys are deliberately insufficient as an ownership token: two
        sessions for the same model can legitimately produce plans with equal
        keys but independent retained matrices and invalidation lifecycles.
        """

        if matrix_type == "stiffness":
            current = self.stiffness_plan(model)
        elif matrix_type == "mass":
            current = self.mass_plan(model)
        else:  # pragma: no cover - private API guard
            raise ValueError(f"unsupported structural matrix type: {matrix_type!r}")
        if plan is None:
            return current
        if (
            plan is not current
            or plan.matrix_type != matrix_type
            or plan.revision_key != current.revision_key
        ):
            self._counters[f"{matrix_type}_plan_rejections"] += 1
            raise ValueError(
                f"stale or foreign StructuralMatrixPlan for {matrix_type}; "
                f"call session.{matrix_type}_plan() again after model changes"
            )
        return current

    def _require_current_constraint_plan(
        self,
        plan: Optional[ConstraintPlan],
        *,
        model: "FEModel",
    ) -> ConstraintPlan:
        """Return the current session-owned affine plan or reject ``plan``."""

        current = self.constraint_plan(model=model)
        if plan is None:
            return current
        if (
            plan is not current
            or plan.structure_key != current.structure_key
            or plan.value_key != current.value_key
            or plan.stiffness_key != current.stiffness_key
        ):
            self._counters["constraint_plan_rejections"] += 1
            raise ValueError(
                "stale or foreign ConstraintPlan; call "
                "session.constraint_plan() again after model or prescribed-value changes"
            )
        return current

    def __enter__(self) -> "AnalysisSession":
        self._require_model()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._owns_factorization_cache:
                self.factorization_cache.clear()
            self._stiffness = None
            self._mass = None
            self._constraint = None
            self._static_operator = None
            self._inertia_relief = None
            self._reduced_mass = None
            self._rigid_modes = None
            self._load_vectors.clear()
            self._output_plans.clear()
            self._activity_snapshots.clear()
            self._closed = True
            self._counters["close_count"] += 1

    release = close

    def _activity_contribution_key(self, model: "FEModel", quantity: str) -> Any:
        """Track effective builtin contribution weights, retaining provider identity.

        Sequence alone misses attachment/replacement and needlessly invalidates
        DELETE_ONLY/RETAIN contributions on softening.  Provider-owned callbacks
        have no cheap epoch contract, so their effective weights are observed
        afresh using the assembly provider validation on every lookup.
        """
        from .matrix_assembly import _activity_scales

        provider = model.mesh.element_activity
        if provider is None:
            self._activity_snapshots.pop(quantity, None)
            return None
        epoch = None
        if _has_builtin_activity_epoch(provider):
            epoch = (provider.sequence, provider.policy, _revision_key(model, "topology"))
        previous = self._activity_snapshots.get(quantity)
        if (epoch is not None and previous is not None and previous[3] is not None
                and previous[0] is provider
                and previous[3][0] == epoch[0] and previous[3][1] is epoch[1]
                and previous[3][2] == epoch[2]):
            return previous[2]
        observed, scales, _info = _activity_scales(model, quantity)
        values = tuple(scales.items())
        if previous is not None and previous[0] is observed and previous[1] == values:
            self._activity_snapshots[quantity] = (observed, values, previous[2], epoch)
            return previous[2]
        self._activity_generation += 1
        token = ("activity", self._activity_identity, self._activity_generation)
        self._activity_snapshots[quantity] = (observed, values, token, epoch)
        return token

    def stiffness_plan(self, model: Optional["FEModel"] = None) -> StructuralMatrixPlan:
        from .matrix_assembly import assemble_stiffness_matrix

        with self._lock:
            owned = self._require_model(model)
            key = (*_revision_key(owned, "topology", "geometry", "material"),
                   self._activity_contribution_key(owned, "stiffness"))
            if self._stiffness is not None and self._stiffness.revision_key == key:
                self._counters["stiffness_hits"] += 1
                return self._stiffness
            if self._stiffness is not None:
                self._inertia_relief = None
                self._invalidation_reasons["stiffness_revision"] += 1
                self.factorization_cache.clear()
                self._constraint = None
                self._static_operator = None
                self._reduced_mass = None
                self._rigid_modes = None
                self._load_vectors.clear()
                self._output_plans.clear()
            start = time.perf_counter()
            matrix, info = assemble_stiffness_matrix(owned)
            plan = StructuralMatrixPlan(_immutable_csr(matrix), dict(info), key, "stiffness")
            self._stiffness = plan
            self._setup_seconds += time.perf_counter() - start
            self._counters["stiffness_builds"] += 1
            return plan

    def static_operator_plan(
        self, *, constraint_mode: str = "auto", model: Optional["FEModel"] = None,
    ) -> StaticOperatorPlan:
        """Reuse preparation and factorization for unchanged static operators."""
        from .assembly import _equilibrate_supported_stiffness, _SYMMETRIC_DIAGONAL_EQUILIBRATION_ID
        from .linalg import MatrixClass, factorize_cached

        with self._lock:
            owned = self._require_model(model)
            constraint = self.constraint_plan(model=owned)
            modes, mode_info = self.rigid_body_modes(constraint, owned)
            mode = str(constraint_mode).strip().lower()
            if mode not in {"auto", "transformation", "nullspace"}:
                raise ValueError("constraint_mode must be auto, transformation or nullspace")
            augmented = mode == "nullspace" or (
                mode == "auto" and int(constraint.info["num_fixed_dofs"]) == 0 and modes.shape[1] > 0
            )
            augmented = bool(augmented and modes.shape[1])
            key = (self._activity_identity, self._model_id, constraint.stiffness_key,
                   constraint.structure_key, augmented, modes.shape,
                   hashlib.sha256(modes.tobytes()).hexdigest())
            old = self._static_operator
            if old is not None and old.key == key and (old.handle is None or old.handle.status == "ok"):
                self._counters["static_operator_hits"] += 1
                handle = None if old.handle is None else factorize_cached(old.matrix,
                    MatrixClass.SYMMETRIC_INDEFINITE, cache=self.factorization_cache,
                    signature=old.handle.signature)
                if old.constraint is constraint and handle is old.handle:
                    return old
                refreshed = StaticOperatorPlan(constraint, old.matrix, old.scaling,
                    old.rigid_modes, handle, old.augmented, key, old.info)
                self._static_operator = refreshed
                return refreshed
            if augmented:
                q_sparse = sparse.csr_matrix(modes)
                matrix = sparse.bmat([[constraint.K_red, q_sparse],
                    [q_sparse.T, sparse.csr_matrix((modes.shape[1], modes.shape[1]))]], format="csr")
                scaling = np.ones(matrix.shape[0])
                equilibration = {"applied": False, "disposition": "NULLSPACE_AUGMENTED"}
                signature = repr(key) + ":rigid_body_nullspace_augmented"
            else:
                matrix, scaling, equilibration = _equilibrate_supported_stiffness(constraint.K_red)
                signature = self.factorization_signature("linear_static", constraint) + ":" + _SYMMETRIC_DIAGONAL_EQUILIBRATION_ID
            handle = None if matrix.shape[0] == 0 else factorize_cached(
                matrix, MatrixClass.SYMMETRIC_INDEFINITE,
                cache=self.factorization_cache, signature=signature)
            plan = StaticOperatorPlan(constraint, matrix, scaling, modes, handle,
                augmented, key, {"equilibration": equilibration, "nullspace_info": mode_info})
            plan = StaticOperatorPlan(constraint, _immutable_csr(matrix), _immutable_array(scaling),
                _immutable_array(modes), handle, augmented, key, plan.info)
            self._static_operator = plan
            self._counters["static_operator_builds"] += 1
            return plan

    def inertia_relief_plan(self, *, model: Optional["FEModel"] = None) -> Any:
        """Reuse physical rigid inertia and elastic certification, never load data.

        The reference point is deliberately absent: accelerations are calculated
        at the centre of mass and transformed to each requested reporting point.
        """
        from .inertia_relief import _admit, _prepare_inertia_relief

        with self._lock:
            owned = self._require_model(model)
            _admit(owned, None)
            # Mass invalidation can clear the static operator/factorization cache.
            mass = self.mass_plan(owned)
            stiffness = self.stiffness_plan(owned)
            constraint = self.constraint_plan(stiffness, owned)
            key = (mass.revision_key, stiffness.revision_key,
                   constraint.structure_key, constraint.value_key)
            if self._inertia_relief is not None and self._inertia_relief[0] == key:
                self._counters["inertia_relief_hits"] += 1
                return self._inertia_relief[1]
            modes, info = self.rigid_body_modes(constraint, owned)
            start = time.perf_counter()
            plan = _prepare_inertia_relief(owned, stiffness.matrix, mass.matrix,
                                          constraint, modes, info)
            self._inertia_relief = (key, plan)
            self._setup_seconds += time.perf_counter()-start
            self._counters["inertia_relief_builds"] += 1
            return plan

    def mass_plan(self, model: Optional["FEModel"] = None) -> StructuralMatrixPlan:
        from .matrix_assembly import assemble_mass_matrix
        from .concentrated_mass import identity as concentrated_mass_identity

        with self._lock:
            owned = self._require_model(model)
            key = (*_revision_key(
                owned,
                "topology",
                "geometry",
                "material",
                "mass",
            ), self._activity_contribution_key(owned, "mass"),
                concentrated_mass_identity(owned.mesh), tuple(
                (int(node_id), float(value))
                for node_id, value in sorted(owned.mesh.point_masses.items())
            ))
            if self._mass is not None and self._mass.revision_key == key:
                self._counters["mass_hits"] += 1
                return self._mass
            if self._mass is not None:
                self._inertia_relief = None
                self._invalidation_reasons["mass_revision"] += 1
                self._reduced_mass = None
                self.factorization_cache.clear()
                self._static_operator = None
            start = time.perf_counter()
            matrix, info = assemble_mass_matrix(owned)
            plan = StructuralMatrixPlan(_immutable_csr(matrix), dict(info), key, "mass")
            self._mass = plan
            self._setup_seconds += time.perf_counter() - start
            self._counters["mass_builds"] += 1
            return plan

    def constraint_plan(
        self,
        stiffness: Optional[StructuralMatrixPlan] = None,
        model: Optional["FEModel"] = None,
    ) -> ConstraintPlan:
        from .assembly import build_constraint_transformation

        with self._lock:
            owned = self._require_model(model)
            stiffness = self._require_current_structural_plan(
                stiffness,
                matrix_type="stiffness",
                model=owned,
            )
            structure, values = _constraint_fingerprints(owned)
            structure_key = (
                *_revision_key(owned, "topology", "geometry", "boundary", "mpc"),
                int(stiffness.matrix.shape[0]),
                structure,
            )
            existing = self._constraint
            if (
                existing is not None
                and existing.structure_key == structure_key
                and existing.value_key == values
                and existing.stiffness_key == stiffness.revision_key
            ):
                self._counters["constraint_hits"] += 1
                return existing
            value_only_refresh = bool(
                existing is not None
                and existing.structure_key == structure_key
                and existing.stiffness_key == stiffness.revision_key
                and existing.value_key != values
            )
            if existing is not None:
                self._inertia_relief = None
                if existing.structure_key != structure_key:
                    self._invalidation_reasons["constraint_structure"] += 1
                    self._output_plans.clear()
                    self._rigid_modes = None
                elif existing.value_key != values:
                    self._invalidation_reasons["constraint_values"] += 1
                    self._counters["constraint_value_refreshes"] += 1
                    self._output_plans.clear()
                else:
                    self._invalidation_reasons["reduced_stiffness"] += 1
                if not value_only_refresh:
                    self.factorization_cache.clear()
                    self._reduced_mass = None
            start = time.perf_counter()
            zero = np.zeros(stiffness.matrix.shape[0], dtype=float)
            K_red, _F_red, T, u0, independent, info = build_constraint_transformation(
                stiffness.matrix,
                zero,
                owned,
            )
            plan = ConstraintPlan(
                (existing.T if value_only_refresh and existing is not None else _immutable_csr(T)),
                _immutable_array(np.asarray(u0, dtype=float)),
                (
                    existing.independent_dofs
                    if value_only_refresh and existing is not None
                    else _immutable_array(np.asarray(independent, dtype=np.intp))
                ),
                (
                    existing.K_red
                    if value_only_refresh and existing is not None
                    else _immutable_csr(K_red)
                ),
                dict(info),
                structure_key,
                values,
                stiffness.revision_key,
            )
            self._constraint = plan
            self._setup_seconds += time.perf_counter() - start
            self._counters["constraint_builds"] += 1
            return plan

    def reduced_mass(
        self,
        constraint: Optional[ConstraintPlan] = None,
        model: Optional["FEModel"] = None,
    ) -> Tuple[sparse.csr_matrix, Mapping[str, Any]]:
        with self._lock:
            owned = self._require_model(model)
            constraint = self._require_current_constraint_plan(
                constraint,
                model=owned,
            )
            mass = self.mass_plan(owned)
            key = (mass.revision_key, constraint.structure_key)
            if self._reduced_mass is not None and self._reduced_mass[0] == key:
                self._counters["reduced_mass_hits"] += 1
                return self._reduced_mass[1], mass.info
            start = time.perf_counter()
            reduced = (constraint.T.T @ mass.matrix @ constraint.T).tocsr()
            self._reduced_mass = (key, reduced)
            self._setup_seconds += time.perf_counter() - start
            self._counters["reduced_mass_builds"] += 1
            return reduced, mass.info

    def rigid_body_modes(
        self,
        constraint: Optional[ConstraintPlan] = None,
        model: Optional["FEModel"] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        from .assembly import build_reduced_rigid_body_modes

        with self._lock:
            owned = self._require_model(model)
            constraint = self._require_current_constraint_plan(
                constraint,
                model=owned,
            )
            key = constraint.structure_key
            if self._rigid_modes is not None and self._rigid_modes[0] == key:
                self._counters["rigid_mode_hits"] += 1
                return self._rigid_modes[1], dict(self._rigid_modes[2])
            start = time.perf_counter()
            modes, info = build_reduced_rigid_body_modes(
                owned,
                constraint.independent_dofs,
                int(constraint.T.shape[0]),
                transformation=constraint.T,
            )
            modes = _immutable_array(modes)
            self._rigid_modes = (key, modes, dict(info))
            self._setup_seconds += time.perf_counter() - start
            self._counters["rigid_mode_builds"] += 1
            return modes, dict(info)

    def output_selection_plan(
        self,
        full_dofs: np.ndarray,
        constraint: Optional[ConstraintPlan] = None,
        model: Optional["FEModel"] = None,
    ) -> OutputSelectionPlan:
        with self._lock:
            owned = self._require_model(model)
            constraint = self._require_current_constraint_plan(
                constraint,
                model=owned,
            )
            dofs = np.asarray(full_dofs, dtype=np.intp).reshape(-1)
            if np.any(dofs < 0) or np.any(dofs >= constraint.T.shape[0]):
                raise ValueError("output selection contains an out-of-range DOF")
            key = (
                constraint.structure_key,
                constraint.value_key,
                tuple(int(value) for value in dofs),
            )
            existing = self._output_plans.get(key)
            if existing is not None:
                self._output_plans.move_to_end(key)
                self._counters["output_plan_hits"] += 1
                return existing
            plan = OutputSelectionPlan(
                _immutable_array(dofs),
                _immutable_csr(constraint.T[dofs]),
                _immutable_array(constraint.u0[dofs]),
                constraint.structure_key,
                constraint.value_key,
            )
            self._output_plans[key] = plan
            self._output_plans.move_to_end(key)
            while len(self._output_plans) > self._max_output_plans:
                self._output_plans.popitem(last=False)
                self._counters["output_plan_evictions"] += 1
            self._counters["output_plan_builds"] += 1
            return plan

    def linear_system(
        self,
        load_case: Optional["LoadCase"] = None,
        model: Optional["FEModel"] = None,
    ) -> Tuple[
        sparse.csr_matrix,
        np.ndarray,
        Dict[str, Any],
        ConstraintPlan,
        np.ndarray,
    ]:
        from .matrix_assembly import assemble_load_vector
        from .concentrated_mass import identity as concentrated_mass_identity

        with self._lock:
            owned = self._require_model(model)
            stiffness = self.stiffness_plan(owned)
            proportional = _load_case_linear_pattern(load_case)
            load_key = None
            amplitude = 1.0
            cached_load = None
            if proportional is not None:
                pattern, amplitude = proportional
                load_key = (
                    _revision_key(
                        owned,
                        "topology",
                        "geometry",
                        "material",
                        "mass",
                    ),
                    self._activity_contribution_key(owned, "load"),
                    concentrated_mass_identity(owned.mesh),
                    pattern,
                )
                cached_load = self._load_vectors.get(load_key)
            if cached_load is None:
                load, load_info = assemble_load_vector(owned, load_case)
                if load_key is not None:
                    unit_load = (
                        np.asarray(load, dtype=float).copy()
                        if amplitude == 0.0
                        else np.asarray(load, dtype=float) / amplitude
                    )
                    self._load_vectors[load_key] = (unit_load, dict(load_info))
                    self._load_vectors.move_to_end(load_key)
                    while len(self._load_vectors) > 8:
                        self._load_vectors.popitem(last=False)
                        self._counters["load_vector_evictions"] += 1
                    self._counters["load_vector_builds"] += 1
            else:
                unit_load, cached_info = cached_load
                load = np.asarray(unit_load, dtype=float) * amplitude
                load_info = {
                    **cached_info,
                    "assembly_time": 0.0,
                    "prepared_load_vector_reused": True,
                }
                self._load_vectors.move_to_end(load_key)
                self._counters["load_vector_hits"] += 1
            constraint = self.constraint_plan(stiffness, owned)
            reduced_load = constraint.reduce_load(stiffness.matrix, load)
            info: Dict[str, Any] = {
                "num_elements": int(stiffness.info.get("num_elements", owned.mesh.num_elements)),
                "num_nodes": int(owned.mesh.num_nodes),
                "total_dofs": int(owned.mesh.dof_manager.total_dofs),
                "includes_mass_matrix": False,
                "assembly_time": float(load_info.get("assembly_time", 0.0)),
                "stiffness": dict(stiffness.info),
                "load": dict(load_info),
                "element_times": dict(stiffness.info.get("element_times", {})),
                "analysis_session": self.diagnostics(),
            }
            return stiffness.matrix, load, info, constraint, reduced_load

    def factorization_signature(
        self,
        purpose: str,
        constraint: Optional[ConstraintPlan] = None,
    ) -> str:
        with self._lock:
            owned = self._require_model()
            constraint = self._require_current_constraint_plan(
                constraint,
                model=owned,
            )
            # The factorization is a property of the matrix, not the caller.
            # The purpose argument is retained for readable call sites while
            # static one-RHS and many-RHS solves share the same handle.
            _ = purpose
            return repr(
                (
                    self._activity_identity,
                    self._model_id,
                    constraint.stiffness_key,
                    constraint.structure_key,
                    tuple(int(value) for value in constraint.K_red.shape),
                    int(constraint.K_red.nnz),
                )
            )

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            retained = 0
            if self._inertia_relief is not None:
                retained += self._inertia_relief[1].retained_bytes
            if self._static_operator is not None:
                retained += _sparse_nbytes(self._static_operator.matrix)
                retained += int(self._static_operator.scaling.nbytes)
            if self._stiffness is not None:
                retained += _sparse_nbytes(self._stiffness.matrix)
            if self._mass is not None:
                retained += _sparse_nbytes(self._mass.matrix)
            if self._constraint is not None:
                retained += _sparse_nbytes(self._constraint.T)
                retained += _sparse_nbytes(self._constraint.K_red)
                retained += int(self._constraint.u0.nbytes + self._constraint.independent_dofs.nbytes)
            if self._reduced_mass is not None:
                retained += _sparse_nbytes(self._reduced_mass[1])
            for load, _info in self._load_vectors.values():
                retained += int(load.nbytes)
            for plan in self._output_plans.values():
                retained += _sparse_nbytes(plan.transformation_rows)
                retained += int(plan.full_dofs.nbytes + plan.prescribed_rows.nbytes)
            hits = sum(value for key, value in self._counters.items() if key.endswith("_hits"))
            builds = sum(value for key, value in self._counters.items() if key.endswith("_builds"))
            return {
                "closed": bool(self._closed),
                "plan_setup_seconds": float(self._setup_seconds),
                "plan_reused": bool(hits > 0),
                "plan_hits": int(hits),
                "plan_builds": int(builds),
                "counters": dict(sorted(self._counters.items())),
                "invalidation_reasons": dict(sorted(self._invalidation_reasons.items())),
                "output_plan_count": int(len(self._output_plans)),
                "load_vector_plan_count": int(len(self._load_vectors)),
                "estimated_retained_bytes": int(retained),
                "factorization_cache": self.factorization_cache.diagnostics(),
            }
