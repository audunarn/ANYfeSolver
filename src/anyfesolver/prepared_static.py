"""Opt-in prepared static evaluator for repeated linear-static evaluation.

``prepare_static_evaluator`` captures one immutable numerical snapshot of a
fixed structure, factorizes its static operator through a private
factorization cache and then evaluates arbitrary ordered static cases without
observing the live model again.  The route is opt-in: every default API,
``StaticCase`` signature and the existing live stream keep their behaviour.

Guarantees:

- The snapshot owns every container; backing buffers are immutable.  Source
  mutation, source deletion or closing the session used during preparation
  cannot change evaluator state.
- The factorization is private to the evaluator; a caller's session lifetime
  or an external same-pattern factorization cannot invalidate it, and no
  caller factor is ever stolen or transferred.
- Loads are evaluated from canonical compiled maps (nodal, element,
  reference-configuration pressure including the linear-static follower
  treatment and its refusals, and consistent acceleration with per-case
  added masses).  Every family is verified against the live assembly oracle
  during preparation; unsupported capabilities are refused explicitly.
- The full actual load vector is used: ``b = T.T @ (F - K @ u0)`` with the
  existing supported scaling or free-free nullspace augmentation, per-case
  backward error certification at the existing limit, and the complete
  compatibility/gauge/multiplier policy.
- Reactions preserve fixed/slave/master separation with complete residual
  diagnostics; stress recovery is compiled only for source-reviewed affine
  families and verified against the recovery oracle.
"""

from __future__ import annotations

import copy
import sys
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from scipy import sparse

from .analysis_session import _immutable_array, _immutable_csr
from .linalg import AutoSparseSolverBackend, FactorizationCache, MatrixClass, factorize_cached
from .prepared_static_loads import (
    PreparedStaticCapabilityError,
    PreparedStaticError,
    SUPPORTED_LOAD_FAMILIES,
    _normalized_families,
    evaluate_load_vector,
    ingest_static_case,
    load_packet_storage_bytes,
    require_preparation_model_authority,
)
from .prepared_static_recovery import (
    apply_compiled_recovery,
    constraint_force_diagnostics,
    owned_recovery_scope,
    trusted_capture_scope,
)
from .prepared_static_snapshot import (
    PreparedStaticSnapshot,
    capture_prepared_static_snapshot,
)
from .recovery import RecoveryConfig, ResourceConfig, ResourcePolicyError
from .prepared_static_envelopes import EnvelopeAccumulator
from .static_batch import StaticCase
from . import prepared_static_shared as _shared_recovery
from .boundary import _callable_authority, _callable_authority_matches


_PLAIN_ATOMIC = frozenset((type(None), bool, int, float, complex, str, bytes))
_MISSING = object()


def _plain_deepcopy(value, memo=None):
    """copy.deepcopy with the same result for the evaluator's plain per-case
    records (M31): dicts, lists and tuples are copied as deepcopy copies them
    (aliasing kept through the memo, a tuple of unchanged items is returned
    itself), atomic values are returned, and a list of atomic values is copied at C
    speed. Anything else goes to copy.deepcopy with the same memo."""
    memo = {} if memo is None else memo
    kind = type(value)
    if kind in _PLAIN_ATOMIC:
        return value
    found = memo.get(id(value), _MISSING)
    if found is not _MISSING:
        return found
    if kind is list:
        if all(type(item) in _PLAIN_ATOMIC for item in value):
            made = value.copy()
            memo[id(value)] = made
            return made
        made = []
        memo[id(value)] = made
        for item in value:
            made.append(_plain_deepcopy(item, memo))
        return made
    if kind is dict:
        made = {}
        memo[id(value)] = made
        for key, item in value.items():
            made[_plain_deepcopy(key, memo)] = _plain_deepcopy(item, memo)
        return made
    if kind is tuple:
        items = [_plain_deepcopy(item, memo) for item in value]
        found = memo.get(id(value), _MISSING)
        if found is not _MISSING:
            return found
        if all(a is b for a, b in zip(items, value)):
            return value
        made = tuple(items)
        memo[id(value)] = made
        return made
    return copy.deepcopy(value, memo)


def _immutable_buffer(array):
    """M31c: an ndarray whose memory is an immutable bytes object (as made by
    analysis_session._immutable_array), so no view of it can be written."""
    if type(array) is not np.ndarray:
        return False
    base = array
    while type(base) is np.ndarray:
        if base.flags.writeable:
            return False
        base = base.base
    return type(base) is bytes


def _array_state(array):
    """M31d: how an array's values are read: the object, its buffer owner, the data
    address, dtype, shape and strides (all of which an assignment can change)."""
    if type(array) is not np.ndarray:
        return None
    return (array, array.base, array.__array_interface__['data'][0], array.dtype, array.shape, array.strides)


def _matrix_parts(matrix):
    """M31c: what a CSR matrix's products and row sums are computed from."""
    return (type(matrix), _array_state(getattr(matrix, 'data', None)), _array_state(getattr(matrix, 'indices', None)),
            _array_state(getattr(matrix, 'indptr', None)), getattr(matrix, 'shape', None))


def _same_parts(matrix, parts):
    now = _matrix_parts(matrix)
    if now[0] is not parts[0] or now[4] != parts[4]:
        return False
    for a, b in zip(now[1:4], parts[1:4]):
        if a is None or b is None or a[0] is not b[0] or a[1] is not b[1]:
            return False
        if a[2] != b[2] or a[3] != b[3] or a[4] != b[4] or a[5] != b[5]:
            return False
    return True


_FLOAT64, _INDEX_TYPES = np.dtype(np.float64), (np.dtype(np.int32), np.dtype(np.int64))
from scipy.sparse import _sparsetools  # noqa: E402  (SciPy's own canonical-format check)


def _immutable_matrix(matrix):
    """M31c: a CSR matrix whose data, indices and index pointers are immutable. M31f:
    with float64 data and int32/int64 indices, so that converting it (as the row sums
    do) cannot warn."""
    if type(matrix) is not sparse.csr_matrix or not all(
            _immutable_buffer(part) for part in (matrix.data, matrix.indices, matrix.indptr)):
        return False
    if not (matrix.data.dtype == _FLOAT64 and matrix.indices.dtype in _INDEX_TYPES
            and matrix.indptr.dtype == matrix.indices.dtype):
        return False
    # M31i: consistent index pointers and canonical format (sorted, no duplicates),
    # computed afresh (SciPy's cached flag can be stale after a part is replaced).
    # The row sums then neither rewrite the parts (sum_duplicates on read-only
    # buffers raises) nor index outside them, also for data the original would
    # never sum (non-finite data returns early there).
    data, indices, indptr = matrix.data, matrix.indices, matrix.indptr
    rows = matrix.shape[0]
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1 or len(indptr) != rows + 1:
        return False
    if len(indices) != len(data) or int(indptr[0]) != 0 or int(indptr[-1]) != len(indices):
        return False
    canonical = getattr(_sparsetools, 'csr_has_canonical_format', None)
    return canonical is not None and bool(canonical(rows, indptr, indices))


def _operator_norm_data(matrix, immutable=None):
    """M30: the row-sum data assembly._backward_error derives from a matrix, by the
    same expressions, computed once; kept together with the matrix it describes
    (M31c: and with the parts it was computed from)."""
    parts = _matrix_parts(matrix)
    made_matrix = sparse.csr_matrix(matrix, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        row_sums = np.asarray(np.abs(made_matrix).sum(axis=1), dtype=float).reshape(-1)
    finite = not bool(np.any(~np.isfinite(row_sums)))
    # As in assembly._backward_error, the maximum is only taken of finite sums.
    matrix_norm = (float(np.max(row_sums)) if row_sums.size else 0.0) if finite else None
    # M31i: `immutable` is passed when the caller has just established it.
    return matrix, finite, matrix_norm, parts, _immutable_matrix(matrix) if immutable is None else immutable


from .assembly import _backward_error as _ASSEMBLY_BACKWARD_ERROR  # the original, captured at import
from .prepared_static_loads import _capture_originals, _originals_unchanged, _standard_float_environment

# M31d: the library bindings whose calls the shortcut skips or repeats differently
# (the row sums of assembly._backward_error), as loaded.
_NORM_ORIGINALS = _capture_originals(
    (np, ('asarray', 'abs', 'max', 'isfinite', 'any', 'errstate', 'geterr')), (sparse, ('csr_matrix',)),
    (sparse.csr_matrix, ('__init__', '__abs__', 'sum')))


def _norm_shortcut_admitted():
    """M31d: the library bindings as loaded and the standard floating-point
    environment (the row sums were and would be computed alike)."""
    return _originals_unchanged(_NORM_ORIGINALS) and _standard_float_environment()


def _block_norm_data(matrix, implementation):
    """M31b: the operator norm data for one block, computed after the block's solve;
    None for a replaced function. M31c: also None unless the matrix is a CSR matrix
    whose data, indices and index pointers are immutable buffers, so its values
    cannot change while the parts _certified_backward_error checks stay the same."""
    if implementation is not _ASSEMBLY_BACKWARD_ERROR or not _norm_shortcut_admitted():
        return None
    if not _immutable_matrix(matrix):
        return None
    return _operator_norm_data(matrix, True)


def _certified_backward_error(matrix, solution, rhs, norm_data, implementation):
    """assembly._backward_error for one solve (bitwise the same result); the matrix's
    row-sum data comes from norm_data when it was computed for this very matrix, whose
    buffers are immutable and whose parts are still read as when it was computed (the
    same arrays, buffer owners, addresses, dtypes, shapes and strides), with the
    library bindings as loaded and the standard floating-point environment then and now.
    `implementation` is the evaluator's backward-error function; anything other than
    the original assembly._backward_error captured at import (for example a test's
    fault injection, or a rebound module attribute) is called as is."""
    if implementation is not _ASSEMBLY_BACKWARD_ERROR:
        return implementation(matrix, solution, rhs)
    if (norm_data is None or norm_data[0] is not matrix or not norm_data[4] or not _norm_shortcut_admitted()
            or not _same_parts(matrix, norm_data[3]) or type(solution) is not np.ndarray
            or solution.ndim != 1):
        return _ASSEMBLY_BACKWARD_ERROR(matrix, solution, rhs)
    made_matrix = sparse.csr_matrix(matrix, dtype=float)
    made_solution = np.asarray(solution, dtype=float)
    made_rhs = np.asarray(rhs, dtype=float)
    if (
        np.any(~np.isfinite(made_matrix.data))
        or np.any(~np.isfinite(made_solution))
        or np.any(~np.isfinite(made_rhs))
    ):
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = np.asarray(made_matrix @ made_solution - made_rhs, dtype=float)
    if np.any(~np.isfinite(residual)) or not norm_data[1]:
        return float("inf")
    matrix_norm = norm_data[2]
    if not np.isfinite(matrix_norm):
        return float("inf")
    if made_solution.ndim != 1:  # pragma: no cover - excluded above
        return _ASSEMBLY_BACKWARD_ERROR(matrix, solution, rhs)
    solution_norm = float(np.linalg.norm(made_solution, ord=np.inf))
    rhs_norm = float(np.linalg.norm(made_rhs, ord=np.inf))
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))
    denominator = matrix_norm * solution_norm + rhs_norm
    if not all(
        np.isfinite(value)
        for value in (solution_norm, rhs_norm, residual_norm, denominator)
    ):
        return float("inf")
    if denominator <= 0.0:
        return 0.0 if residual_norm == 0.0 else float("inf")
    return residual_norm / denominator
_RESPONSE_EXECUTION_GUARD = None
_ENVELOPE_CONSTRUCTOR = EnvelopeAccumulator
_SHARED_BINDINGS = tuple((name, getattr(_shared_recovery, name), _callable_authority(getattr(_shared_recovery, name)))
                        for name in ('recover_shared_block', 'require_shared_recovery_authority', 'shared_workspace_bytes'))

def _require_shared_execution(owner=None):
    if owner is not None and _RESPONSE_EXECUTION_GUARD is not None:
        guard, authority = _RESPONSE_EXECUTION_GUARD
        if not _callable_authority_matches(guard, authority):
            raise PreparedStaticCapabilityError("response authority guard changed")
        guard(owner)
    if EnvelopeAccumulator is not _ENVELOPE_CONSTRUCTOR:
        raise PreparedStaticCapabilityError('shared recovery envelope constructor authority changed')
    if any(getattr(_shared_recovery, name, None) is not function or not _callable_authority_matches(function, authority)
           for name, function, authority in _SHARED_BINDINGS):
        raise PreparedStaticCapabilityError('shared recovery entry authority changed')
    _SHARED_BINDINGS[1][1]()


__all__ = [
    "PreparedStaticBatch",
    "PreparedStaticCapabilityError",
    "PreparedStaticError",
    "PreparedStaticEvaluator",
    "prepare_static_evaluator",
]


def _release_owned_resources(cache: FactorizationCache, backend: AutoSparseSolverBackend) -> None:
    """Release resources without capturing their owning evaluator."""
    try:
        cache.clear()
    finally:
        backend.release_pattern_slots()


@dataclass(frozen=True)
class PreparedStaticBatch:
    """One certified block of independent prepared static responses."""

    input_indices: Tuple[int, ...]
    case_ids: Tuple[str, ...]
    times: Tuple[Optional[float], ...]
    output_dofs: np.ndarray
    displacements: np.ndarray
    case_results: Tuple[Dict[str, Any], ...]
    diagnostics: Dict[str, Any]


class PreparedStaticEvaluator:
    """Context-managed prepared static evaluator for one fixed structure."""

    def __init__(
        self,
        model: Any,
        *,
        output_dofs: Optional[Any] = None,
        session: Any = None,
        cancellation_token: Any = None,
        progress_callback: Any = None,
        recovery_config: Optional[RecoveryConfig] = None,
        envelope_recovery_config: Optional[RecoveryConfig] = None,
        resource_config: Optional[ResourceConfig] = None,
        load_families: Optional[Iterable[str]] = None,
        constraint_mode: str = "auto",
        allow_unbalanced_free_free: bool = False,
        batch_size: int = 32,
        trusted_capture: bool = False,
    ) -> None:
        # Constructor admission cannot dispatch through an overridable instance
        # hook before the native preparation scopes have authenticated the owner.
        if type(self) is not PreparedStaticEvaluator:
            if _RESPONSE_EXECUTION_GUARD is None:
                raise PreparedStaticCapabilityError("prepared evaluator subclasses are not admitted")
            guard, authority = _RESPONSE_EXECUTION_GUARD
            if not _callable_authority_matches(guard, authority):
                raise PreparedStaticCapabilityError("response authority guard changed")
            guard(self)
        from .assembly import (
            _LINEAR_STATIC_BACKWARD_ERROR_LIMIT,
            _backward_error,
            _equilibrate_supported_stiffness,
        )
        from .threading_policy import solver_thread_scope
        from .control import SolveCancelled, cancellation_safe_point, emit_progress

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(trusted_capture) is not bool:
            raise ValueError("trusted_capture must be True or False")
        mode = str(constraint_mode).strip().lower()
        if mode not in {"auto", "transformation", "nullspace"}:
            raise ValueError("constraint_mode must be auto, transformation or nullspace")
        families = _normalized_families(None if load_families is None else tuple(load_families))
        recovery = (
            RecoveryConfig(include_displacements=False, include_stresses=False,
                           include_reactions=False)
            if recovery_config is None else recovery_config
        )
        if session is not None:
            from .analysis_session import AnalysisSession
            if type(session) is not AnalysisSession:
                raise TypeError("session must be an AnalysisSession")
            session._require_model(model)
        self._output_dofs = None if output_dofs is None else np.asarray(output_dofs, dtype=np.intp).reshape(-1).copy()
        recovery = copy.deepcopy(recovery)
        resource_config = copy.deepcopy(resource_config)
        self._recovery = recovery
        self._resource_config = resource_config
        self._batch_size = int(batch_size)
        self._trusted_capture = trusted_capture
        self._constraint_mode = mode
        self._allow_unbalanced_free_free = bool(allow_unbalanced_free_free)
        self._backward_error_limit = float(_LINEAR_STATIC_BACKWARD_ERROR_LIMIT)
        self._backward_error = _backward_error
        self._lock = threading.RLock()
        self._closed = False
        self._envelope_active = False
        self._envelope = None
        self._snapshot = None
        self._identity = uuid.uuid4().hex
        self._counters: Dict[str, int] = {}
        self._failures: Dict[str, int] = {}
        self._blocks_yielded = 0
        self._cases_certified = 0
        self._backend = AutoSparseSolverBackend()
        self._factorization_cache = FactorizationCache(
            name="prepared_static_evaluator", max_entries=2, backend=self._backend
        )
        self._finalizer = weakref.finalize(self, _release_owned_resources, self._factorization_cache, self._backend)
        try:
            cancellation_safe_point(cancellation_token, "prepared_static.before_preparation")
            require_preparation_model_authority(model)
            # M25f: the trusted capture runs only when the caller opted in.
            with trusted_capture_scope(trusted_capture):
                self._snapshot = capture_prepared_static_snapshot(
                    model,
                    recovery_config=recovery,
                    envelope_recovery_config=copy.deepcopy(envelope_recovery_config),
                    load_families=families,
                    cancellation_token=cancellation_token,
                    progress_callback=progress_callback,
                    _resource_config=self._resource_config,
                )
            _require_shared_execution(self)
            self._envelope = None if self._snapshot.envelope_plan is None else self._new_envelope(self._snapshot.envelope_plan)
            self._envelope_provenance = {'snapshot_content_digest': self._snapshot.content_digest,
                'source_revision': dict(self._snapshot.source_revision)}
        except (PreparedStaticError, ResourcePolicyError, SolveCancelled):
            self._finalizer()
            self._count("preparation_failures")
            raise
        except Exception as error:
            self._finalizer()
            self._count("preparation_failures")
            raise PreparedStaticError(
                "prepared static snapshot capture failed",
                diagnostics={"stage": "snapshot capture", "error": str(error),
                            "original_exception_type": type(error).__name__},
            ) from error
        snapshot = self._snapshot
        augmented = mode == "nullspace" or (
            mode == "auto"
            and int(snapshot.constraint_info.get("num_fixed_dofs", 0)) == 0
            and snapshot.rigid_modes.shape[1] > 0
        )
        augmented = bool(augmented and snapshot.rigid_modes.shape[1])
        try:
            if augmented:
                modes = sparse.csr_matrix(snapshot.rigid_modes)
                rank = int(snapshot.rigid_modes.shape[1])
                matrix = sparse.bmat(
                    [[snapshot.reduced_stiffness, modes],
                     [modes.T, sparse.csr_matrix((rank, rank))]],
                    format="csr",
                )
                scaling = np.ones(matrix.shape[0], dtype=float)
                equilibration = {"applied": False, "disposition": "NULLSPACE_AUGMENTED"}
            else:
                matrix, scaling, equilibration = _equilibrate_supported_stiffness(
                    snapshot.reduced_stiffness
                )
            signature = (
                f"prepared_static:{self._identity}:"
                f"{snapshot.reduced_stiffness.shape[0]}:{snapshot.reduced_stiffness.nnz}:"
                f"{augmented}:{int(snapshot.rigid_modes.shape[1])}"
            )
            if resource_config is not None and resource_config.memory_limit_bytes is not None:
                from .recovery import MemoryEstimate
                matrix_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
                required = snapshot.nbytes() + matrix_bytes * 5 + snapshot.total_dofs * 80
                if self._envelope is not None:
                    required += 3 * self._envelope.nbytes()
                _require_shared_execution(self)
                required += _SHARED_BINDINGS[2][1](snapshot, 1, recovery.include_reactions)
                if required > resource_config.memory_limit_bytes:
                    estimate_fields = dict(snapshot.memory_estimate)
                    estimate_fields["total_bytes_estimate"] = int(required)
                    raise ResourcePolicyError("prepared factor and one case exceed memory policy", context="prepare_static_evaluator", memory_estimate=MemoryEstimate(**estimate_fields), resource_config=resource_config)
            cancellation_safe_point(cancellation_token, "prepared_static.before_factorization")
            emit_progress(progress_callback, "prepared_static_preparation", "prepared_static.factorization", suppress_callback_errors=False)
            cancellation_safe_point(cancellation_token, "prepared_static.before_factorization")
            with solver_thread_scope(resource_config):
                handle = None if matrix.shape[0] == 0 else factorize_cached(
                    matrix,
                    MatrixClass.SYMMETRIC_INDEFINITE,
                    cache=self._factorization_cache,
                    signature=signature,
                )
            cancellation_safe_point(cancellation_token, "prepared_static.after_factorization")
            if handle is not None and handle.status != "ok":
                raise PreparedStaticError(
                    "prepared static factorization failed",
                    diagnostics=dict(handle.diagnostics()),
                )
        except (PreparedStaticError, ResourcePolicyError, SolveCancelled):
            self._finalizer()
            self._count("preparation_failures")
            raise
        except Exception as error:
            self._finalizer()
            self._count("preparation_failures")
            raise PreparedStaticError(
                "prepared static operator preparation failed",
                diagnostics={"stage": "operator preparation", "error": str(error),
                            "original_exception_type": type(error).__name__},
            ) from error
        self._operator_matrix = _immutable_csr(matrix)
        self._operator_scaling = _immutable_array(np.asarray(scaling, dtype=float))
        self._operator_augmented = augmented
        self._operator_handle = handle
        self._operator_info = {
            "equilibration": dict(equilibration),
            "nullspace_info": dict(snapshot.nullspace_info),
        }
        self._count("preparations")

    # -- context management -------------------------------------------------

    def __enter__(self) -> "PreparedStaticEvaluator":
        if self._closed:
            raise RuntimeError("PreparedStaticEvaluator is closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._finalizer()
            self._operator_handle = None
            if self._envelope is not None:
                self._envelope.plan = None
            self._snapshot = None
            self._operator_matrix = sparse.csr_matrix((0, 0))
            self._operator_scaling = np.zeros(0)
            self._closed = True
            self._count("close_count")

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    # -- diagnostics ---------------------------------------------------------

    def _count(self, name: str) -> None:
        self._counters[name] = self._counters.get(name, 0) + 1

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = self._snapshot
            return {
                "closed": bool(self._closed),
                "identity": self._identity,
                "constraint_mode": self._constraint_mode,
                "augmented": bool(self._operator_augmented),
                "batch_size": int(self._batch_size),
                "allow_unbalanced_free_free": bool(self._allow_unbalanced_free_free),
                "load_families": list(self._snapshot_families()),
                "trusted_capture": bool(self._trusted_capture),
                "blocks_yielded": int(self._blocks_yielded),
                "cases_certified": int(self._cases_certified),
                "counters": dict(sorted(self._counters.items())),
                "failures": dict(sorted(self._failures.items())),
                "snapshot": None if snapshot is None else {
                    "total_dofs": int(snapshot.total_dofs),
                    "num_nodes": int(snapshot.num_nodes),
                    "num_elements": int(snapshot.num_elements),
                    "retained_bytes": int(snapshot.nbytes()),
                    "setup_seconds": float(snapshot.setup_seconds),
                    "phase_timings_seconds": dict(snapshot.phase_timings),
                    "load_map_verification": {
                        family: residual
                        for family, residual in snapshot.load_maps.verification
                    },
                    "acceleration_available": bool(snapshot.load_maps.acceleration_available),
                    "memory_estimate": copy.deepcopy(dict(snapshot.memory_estimate)),
                    "recovery_compiled_elements": (
                        0 if snapshot.recovery_compilation is None
                        else len(snapshot.recovery_compilation.elements)
                    ),
                    "recovery_probe_count": (
                        0 if snapshot.recovery_compilation is None
                        else int(snapshot.recovery_compilation.probe_count)
                    ),
                },
                "factorization_cache": self._factorization_cache.diagnostics(),
                "operator_bytes": int(
                    self._operator_matrix.data.nbytes + self._operator_matrix.indices.nbytes
                    + self._operator_matrix.indptr.nbytes + self._operator_scaling.nbytes
                ),
            }

    def _snapshot_families(self) -> Tuple[str, ...]:
        snapshot = self._snapshot
        return () if snapshot is None else tuple(snapshot.load_maps.families)

    def _require_open(self) -> PreparedStaticSnapshot:
        if _RESPONSE_EXECUTION_GUARD is not None and type(self) is not PreparedStaticEvaluator:
            guard, authority = _RESPONSE_EXECUTION_GUARD
            if not _callable_authority_matches(guard, authority):
                raise PreparedStaticCapabilityError("response authority guard changed")
            guard(self)
        if self._closed or self._snapshot is None:
            raise RuntimeError("PreparedStaticEvaluator is closed")
        return self._snapshot

    # -- evaluation ---------------------------------------------------------

    def iter_batches(
        self,
        cases: Iterable[StaticCase],
        *,
        output_dofs: Optional[Any] = None,
        batch_size: Optional[int] = None,
        cancellation_token: Any = None,
        progress_callback: Any = None,
    ) -> Iterator[PreparedStaticBatch]:
        """Yield certified prepared static blocks of independent cases.

        The closed check is eager: a closed evaluator refuses before any
        input is consumed.
        """

        _require_shared_execution(self)
        self._require_open()
        return self._iter_batches_open(
            cases,
            output_dofs=self._output_dofs if output_dofs is None else output_dofs,
            batch_size=batch_size,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )

    def envelope_result(self):
        """Return a detached summary of the current/last envelope run."""
        if self._envelope is None:
            return None
        _require_shared_execution(self)
        result = self._envelope.result(self._identity)
        result.diagnostics.update(copy.deepcopy(self._envelope_provenance))
        return result

    def _iter_batches_open(self, cases, **kwargs):
        if self._envelope is None:
            yield from self._iter_batches_impl(cases, **kwargs)
            return
        if self._envelope_active:
            raise RuntimeError("only one prepared envelope iterator may be active")
        _require_shared_execution(self)
        self._envelope_active = True
        self._envelope = self._new_envelope(self._require_open().envelope_plan)
        try:
            yield from self._iter_batches_impl(cases, **kwargs)
            # Input exhaustion and cleanup are caller observations, too.
            from .control import cancellation_safe_point
            self._require_open()
            cancellation_safe_point(kwargs.get('cancellation_token'), 'prepared_static.exhausted')
            self._require_open()
            with owned_recovery_scope(()):
                pass
            _require_shared_execution(self)
            self._envelope.complete = True
        finally:
            self._envelope_active = False

    def _iter_batches_impl(
        self,
        cases: Iterable[StaticCase],
        *,
        output_dofs: Optional[Any] = None,
        batch_size: Optional[int] = None,
        cancellation_token: Any = None,
        progress_callback: Any = None,
        _numeric_recovery: bool = False,
    ) -> Iterator[PreparedStaticBatch]:
        from .control import cancellation_safe_point, emit_progress
        from .threading_policy import solver_thread_scope

        snapshot = self._require_open()
        recovery = self._recovery
        resource_config = self._resource_config
        cancellation_safe_point(cancellation_token, "prepared_static.start")
        self._require_open()
        total_dofs = int(snapshot.total_dofs)
        dofs = (
            np.arange(total_dofs, dtype=np.intp)
            if output_dofs is None
            else np.asarray(output_dofs, dtype=np.intp).reshape(-1).copy()
        )
        if np.any(dofs < 0) or np.any(dofs >= total_dofs):
            raise ValueError("output_dofs contains an out-of-range DOF")
        rows = snapshot.transformation[dofs].tocsr()
        prescribed = snapshot.prescribed_offset[dofs]
        requested_batch_size = self._batch_size if batch_size is None else batch_size
        if type(requested_batch_size) is not int or requested_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        selected_elements = len(snapshot.selected_element_ids)
        operator_bytes = self.diagnostics()["operator_bytes"]
        factor_bytes = (0 if self._operator_handle is None else int(self._operator_handle.metadata.get("factor_storage_bytes", operator_bytes * 4)))
        base_bytes = snapshot.nbytes() + operator_bytes + factor_bytes + self._response_storage_bytes()
        _require_shared_execution(self)
        tile_width = min(8, requested_batch_size)
        if resource_config is not None and resource_config.memory_limit_bytes is not None:
            tile_width = 1
        base_bytes += _SHARED_BINDINGS[2][1](snapshot, tile_width, recovery.include_reactions)
        fixed_base_bytes = base_bytes
        if self._envelope is not None:
            base_bytes += 3 * self._envelope.nbytes()
        recovery_bytes = 0
        if snapshot.recovery_compilation is not None:
            for element in snapshot.recovery_compilation.elements.values():
                recovery_bytes += sum(base.nbytes for _name, base, _matrix, _shape in element.array_components)
                recovery_bytes += sum(value.nbytes for _name, value in element.constant_components if isinstance(value, np.ndarray))
                recovery_bytes += len(element.derived_components) * max((base.nbytes for _name, base, _matrix, _shape in element.array_components), default=8)
        per_case_bytes = int((total_dofs * 10 + dofs.size) * 8 + recovery_bytes
            + (total_dofs * 160 + snapshot.reaction_maps.retained_bytes * 2 if recovery.include_reactions else 0))
        if _numeric_recovery and getattr(self, '_numeric_export_plan', None) is not None:
            plan = self._numeric_export_plan
            width = max((entry[4].shape[0] for entry in self._element_library.values()),default=0)
            patterns = len(self._pattern_names)
            # Full source oracle, selected F planes, stack, derived tile and
            # component rows coexist. Reserve conservatively before recovery.
            per_case_bytes += plan[2] + 64*width*(patterns+1)*32
        effective = requested_batch_size
        limit = None if resource_config is None else resource_config.memory_limit_bytes
        if limit is not None:
            available = int(limit) - base_bytes
            effective = min(requested_batch_size, available // max(per_case_bytes, 1))
            if effective < 1:
                from .recovery import MemoryEstimate

                estimate_fields = dict(snapshot.memory_estimate)
                estimate_fields["total_bytes_estimate"] = base_bytes + per_case_bytes
                estimate_fields["notes"] = tuple(
                    estimate_fields.get("notes", ())
                ) + ("prepared block and factor storage estimated",)
                raise ResourcePolicyError(
                    "one prepared static case exceeds configured memory estimate limit",
                    context="prepared_static_evaluator.iter_batches",
                    memory_estimate=MemoryEstimate(**estimate_fields),
                    resource_config=resource_config,
                )
        # A variable input packet is accounted before copying. With a memory
        # limit, one-case blocks avoid retaining an unbounded pending packet.
        if limit is not None and not getattr(self, "_fixed_response_packets", False):
            effective = 1
        iterator = iter(cases)
        next_index = 0
        try:
            self._require_open()
            while True:
                self._require_open()
                _require_shared_execution(self)
                if self._envelope is not None:
                    base_bytes = fixed_base_bytes + 3 * self._envelope.nbytes()
                block_started = time.perf_counter()
                cancellation_safe_point(cancellation_token, "prepared_static.before_input")
                self._require_open()
                block: List[Tuple[str, Optional[float], Any]] = []
                indices: List[int] = []
                packet_bytes = 0
                for _ in range(effective):
                    try:
                        case = next(iterator)
                    except StopIteration:
                        break
                    except Exception as error:
                        from .control import SolveCancelled
                        if isinstance(error, SolveCancelled):
                            raise
                        raise PreparedStaticError("static case provider failed", diagnostics={"stage": "case input", "input_index": next_index, "error": str(error)}) from error
                    self._require_open()
                    try:
                        incoming_bytes = self._packet_storage_bytes(case)
                        if limit is not None and getattr(self, "_fixed_response_packets", False):
                            capacity = (int(limit) - base_bytes - packet_bytes - incoming_bytes) // max(per_case_bytes, 1)
                            # Reduce before copying an unusually large native packet.
                            # No unaccounted pending packet is retained across a yield.
                            effective = min(effective, max(len(block) + 1, capacity))
                        if limit is not None and base_bytes + per_case_bytes * effective + packet_bytes + incoming_bytes > limit:
                            from .recovery import MemoryEstimate
                            fields = dict(snapshot.memory_estimate)
                            fields["total_bytes_estimate"] = base_bytes + per_case_bytes * effective + packet_bytes + incoming_bytes
                            raise ResourcePolicyError("prepared load packet exceeds remaining block memory policy before copy", context="prepared_static.load_ingestion", memory_estimate=MemoryEstimate(**fields), resource_config=resource_config)
                        case_id, case_time, owned = self._ingest_case(case)
                        packet_bytes += incoming_bytes
                        self._require_open()
                    except PreparedStaticError as error:
                        raise type(error)(str(error), diagnostics={**error.diagnostics, "input_index": next_index, "case_id": getattr(case, "case_id", None)}) from error
                    block.append((case_id, case_time, owned))
                    indices.append(next_index)
                    next_index += 1
                    cancellation_safe_point(cancellation_token, "prepared_static.after_input")
                    self._require_open()
                    if len(block) >= effective:
                        break
                if not block:
                    return
                phases: Dict[str, float] = {}
                phase_start = time.perf_counter()
                vectors = []
                load_infos = []
                _require_shared_execution(self)
                with owned_recovery_scope(self._load_scope_recipes(snapshot, block)):
                    for case_index, (case_id, _case_time, owned) in enumerate(block):
                        try:
                            vector, load_info = self._load_vector(snapshot, owned)
                        except PreparedStaticError as error:
                            raise type(error)(str(error), diagnostics={**error.diagnostics, "case_id": case_id, "input_index": indices[case_index]}) from error
                        vectors.append(vector)
                        load_infos.append(load_info)
                F = np.column_stack(vectors)
                del vectors
                phases["load_evaluation"] = time.perf_counter() - phase_start
                phase_start = time.perf_counter()
                rhs = np.asarray(
                    snapshot.transformation.T
                    @ (F - (snapshot.stiffness @ snapshot.prescribed_offset)[:, None]),
                    dtype=float,
                )
                rank = int(snapshot.rigid_modes.shape[1])
                compatibility = []
                for j, (case_id, _case_time, _owned) in enumerate(block):
                    components = np.asarray(
                        snapshot.rigid_modes.T @ rhs[:, j], dtype=float
                    ).reshape(-1)
                    imbalance = float(np.linalg.norm(components))
                    relative = imbalance / max(float(np.linalg.norm(rhs[:, j])), 1.0)
                    warnings = []
                    if self._operator_augmented and relative > 1e-7:
                        message = (
                            "The external load vector has a non-zero rigid-body component. "
                            "For a physical free-free static solution, use self-equilibrated "
                            "loads."
                        )
                        if not self._allow_unbalanced_free_free:
                            self._count_failure("incompatible_free_free_load")
                            raise PreparedStaticError(
                                message,
                                diagnostics={
                                    "status": "incompatible_free_free_load",
                                    "case_id": case_id,
                                    "input_index": indices[j],
                                    "relative_rigid_body_load_imbalance": relative,
                                },
                            )
                        warnings.append(
                            message
                            + " The nullspace solve returned a gauged displacement field "
                            "and balancing generalized reactions."
                        )
                    compatibility.append({
                        "rigid_body_load_components": components.tolist(),
                        "rigid_body_load_imbalance_norm": imbalance,
                        "relative_rigid_body_load_imbalance": relative,
                        "warnings": warnings,
                    })
                solve_rhs = (
                    np.vstack((rhs, np.zeros((rank, len(block)))))
                    if self._operator_augmented
                    else rhs
                )
                phases["load_reduction_and_compatibility"] = time.perf_counter() - phase_start
                phase_start = time.perf_counter()
                cancellation_safe_point(cancellation_token, "prepared_static.before_solve")
                self._require_open()
                if self._operator_handle is None and self._operator_matrix.shape[0] != 0:
                    raise RuntimeError("prepared static operator handle is unavailable")
                try:
                    with solver_thread_scope(resource_config):
                        solved = self._solve_block(solve_rhs, block)
                except (PreparedStaticError, ResourcePolicyError):
                    raise
                except Exception as error:
                    self._count_failure("backend_solve_failures")
                    raise PreparedStaticError(
                        "prepared static backend solve failed",
                        diagnostics={
                            "stage": "solve",
                            "case_ids": tuple(case_id for case_id, _t, _o in block),
                            "error": str(error),
                            "original_exception_type": type(error).__name__,
                        },
                    ) from error
                cancellation_safe_point(cancellation_token, "prepared_static.after_solve")
                self._require_open()
                phases["solve"] = time.perf_counter() - phase_start
                phase_start = time.perf_counter()
                q = solved[: rhs.shape[0]]
                values = np.asarray(rows @ q, dtype=float) + prescribed[:, None]
                need_full = (
                    self._envelope is not None
                    or recovery.include_stresses
                    or recovery.include_reactions
                    or recovery.include_displacements
                )
                full = (
                    np.asarray(
                        snapshot.transformation @ q + snapshot.prescribed_offset[:, None]
                    )
                    if need_full
                    else None
                )
                phases["output_reconstruction"] = time.perf_counter() - phase_start
                phase_start = time.perf_counter()
                certified = []
                certification_norm = _block_norm_data(
                    self._operator_matrix if self._operator_augmented else snapshot.reduced_stiffness,
                    self._backward_error)
                for j, (case_id, _case_time, _owned) in enumerate(block):
                    error = (
                        _certified_backward_error(self._operator_matrix, solved[:, j], solve_rhs[:, j],
                                                  certification_norm, self._backward_error)
                        if self._operator_augmented
                        else _certified_backward_error(
                            snapshot.reduced_stiffness, q[:, j], rhs[:, j], certification_norm,
                            self._backward_error
                        )
                    )
                    if not np.isfinite(error) or error > self._backward_error_limit:
                        self._count_failure("backward_error_limit_exceeded")
                        raise PreparedStaticError(
                            "static case exceeded certified backward-error limit",
                            diagnostics={
                                "case_id": case_id,
                                "input_index": indices[j],
                                "relative_backward_error": error if np.isfinite(error) else None,
                                "relative_backward_error_disposition": (
                                    "LIMIT_EXCEEDED" if np.isfinite(error)
                                    else "NONFINITE_OR_OVERFLOW"
                                ),
                            },
                        )
                    info = {
                        "status": "converged",
                        "case_id": case_id,
                        "input_index": indices[j],
                        "relative_backward_error": error,
                        "relative_backward_error_limit": self._backward_error_limit,
                        "relative_backward_error_disposition": "CERTIFIED",
                        "load": load_infos[j],
                        "backend": {} if self._operator_handle is None
                        else self._operator_handle.diagnostics(),
                        **_plain_deepcopy(self._operator_info),
                        **compatibility[j],
                    }
                    if self._operator_augmented:
                        info.update(
                            method="rigid_body_nullspace_augmented",
                            nullspace_rank=rank,
                            rigid_body_lagrange_multipliers=solved[
                                rhs.shape[0]:, j
                            ].tolist(),
                            gauge_residual_norm=float(
                                np.linalg.norm(snapshot.rigid_modes.T @ q[:, j])
                            ),
                            augmented_residual_norm=float(
                                np.linalg.norm(
                                    snapshot.reduced_stiffness @ q[:, j]
                                    + snapshot.rigid_modes @ solved[rhs.shape[0]:, j]
                                    - rhs[:, j]
                                )
                            ),
                        )
                    certified.append(info)
                phases["certification"] = time.perf_counter() - phase_start
                phase_start = time.perf_counter()
                _require_shared_execution(self)
                try:
                    recover = self._recover_numeric_block if _numeric_recovery else self._recover_block
                    numeric_options = {'cancellation_token':cancellation_token} if _numeric_recovery else {}
                    results, envelope_transaction, shared_statistics = recover(
                        snapshot, full, F, recovery,
                        tuple((case_id, case_time, indices[j]) for j, (case_id, case_time, _owned) in enumerate(block)),
                        certified, self._envelope, block, **numeric_options)
                except PreparedStaticError:
                    self._count_failure('case_recovery_failures')
                    raise
                _require_shared_execution(self)
                if envelope_transaction is not None and limit is not None:
                    required = base_bytes + per_case_bytes * len(block) + packet_bytes + max(0, envelope_transaction.nbytes() - self._envelope.nbytes())
                    if required > limit:
                        from .recovery import MemoryEstimate
                        estimate_fields = dict(snapshot.memory_estimate)
                        estimate_fields['total_bytes_estimate'] = required
                        raise ResourcePolicyError('prepared envelope transaction exceeds memory policy', context='prepared_static.envelope', memory_estimate=MemoryEstimate(**estimate_fields), resource_config=resource_config)
                phases["recovery"] = time.perf_counter() - phase_start
                diagnostics = {
                    "status": "converged",
                    "batch_size": len(block),
                    "requested_batch_size": requested_batch_size,
                    "effective_batch_size": effective,
                    "estimated_block_bytes": base_bytes + per_case_bytes * len(block) + packet_bytes,
                    "retained_load_packet_bytes": packet_bytes,
                    "prepared_static": {
                        "identity": self._identity,
                        "augmented": bool(self._operator_augmented),
                        "load_families": list(snapshot.load_maps.families),
                        "snapshot_bytes": int(snapshot.nbytes()),
                        "factorization_cache": self._factorization_cache.diagnostics(),
                    },
                    "elapsed_seconds": time.perf_counter() - block_started,
                    "phase_timings_seconds": phases,
                    "shared_recovery": shared_statistics,
                    "full_reconstruction": need_full,
                    "prepared_reactions": bool(recovery.include_reactions),
                }
                emit_progress(
                    progress_callback,
                    "prepared_static_batch_complete",
                    "prepared_static.complete",
                    completed=next_index,
                    num_cases=len(block),
                    suppress_callback_errors=False,
                )
                cancellation_safe_point(cancellation_token, "prepared_static.before_yield")
                self._require_open()
                _require_shared_execution(self)
                if envelope_transaction is not None:
                    from .prepared_static_loads import require_native_source_authority
                    require_native_source_authority()
                    from .boundary import _numpy_error_routes_are_builtin
                    if not _numpy_error_routes_are_builtin():
                        raise PreparedStaticCapabilityError("native NumPy error routes changed")
                    self._envelope = envelope_transaction
                self._blocks_yielded += 1
                self._cases_certified += len(block)
                self._count("blocks")
                values.setflags(write=False)
                dofs.setflags(write=False)
                yield PreparedStaticBatch(
                    tuple(indices),
                    tuple(case_id for case_id, _t, _o in block),
                    tuple(case_time for _c, case_time, _o in block),
                    _immutable_array(dofs).view(),
                    values,
                    tuple(results),
                    diagnostics,
                )
                del F, rhs, solve_rhs, solved, q, values, full, results, certified, block
        finally:
            # No authority lease, model reference or live object is held across
            # a yield, a callback or a close; nothing needs revocation here.
            primary = sys.exc_info()[1]
            close_input = getattr(iterator, "close", None)
            if close_input is not None:
                try:
                    close_input()
                except Exception as cleanup_error:
                    if primary is None or isinstance(primary, GeneratorExit):
                        raise
                    primary.add_note("input iterator cleanup also failed: " + type(cleanup_error).__name__ + ": " + str(cleanup_error))

    def _new_envelope(self, plan):
        return EnvelopeAccumulator(plan)

    def _load_scope_recipes(self, snapshot, block):
        return tuple(snapshot.load_maps.pressure_recipes.values())

    def _response_storage_bytes(self):
        return 0

    def _packet_storage_bytes(self, case):
        return load_packet_storage_bytes(case)

    def _ingest_case(self, case):
        return ingest_static_case(case)

    def _load_vector(self, snapshot, owned):
        return evaluate_load_vector(snapshot.load_maps, owned)

    def _solve_block(self, solve_rhs, block):
        return (np.zeros_like(solve_rhs) if self._operator_handle is None else
                self._operator_scaling[:, None] * self._operator_handle.solve_many(
                    self._operator_scaling[:, None] * solve_rhs))

    def _recover_block(self, snapshot, full, loads, recovery, labels, certified, envelope, block):
        return _SHARED_BINDINGS[0][1](snapshot, full, loads, recovery, labels, certified, envelope)

    def _count_failure(self, name: str) -> None:
        self._failures[name] = self._failures.get(name, 0) + 1


def prepare_static_evaluator(
    model: Any,
    *,
    output_dofs: Optional[Any] = None,
    session: Any = None,
    cancellation_token: Any = None,
    progress_callback: Any = None,
    recovery_config: Optional[RecoveryConfig] = None,
    envelope_recovery_config: Optional[RecoveryConfig] = None,
    resource_config: Optional[ResourceConfig] = None,
    load_families: Optional[Iterable[str]] = None,
    constraint_mode: str = "auto",
    allow_unbalanced_free_free: bool = False,
    batch_size: int = 32,
    trusted_capture: bool = False,
) -> PreparedStaticEvaluator:
    """Prepare one opt-in static evaluator for a fixed structure.

    The returned evaluator is a context manager exposing ``iter_batches``,
    ``close`` and ``diagnostics``.  Geometry, materials, constraints,
    prescribed offsets, element activity and structural mass are frozen at
    preparation; arbitrary static time labels and changing case loads remain
    accepted for every supported load family.

    ``trusted_capture=True`` (opt-in, default off) captures the per-element
    recovery recipes inside the native trusted operation scope when its
    admission checks hold, which shortens preparation; results are identical.
    It relies on no foreign code running during preparation (no tracing,
    profiling or monitoring hooks, other Python threads or similar), the same
    assumption as the trusted recovery scopes. The default captures every
    recipe with full per-call guards.
    """

    return PreparedStaticEvaluator(
        model,
        output_dofs=output_dofs,
        session=session,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
        recovery_config=recovery_config,
        envelope_recovery_config=envelope_recovery_config,
        resource_config=resource_config,
        load_families=load_families,
        constraint_mode=constraint_mode,
        allow_unbalanced_free_free=allow_unbalanced_free_free,
        batch_size=batch_size,
        trusted_capture=trusted_capture,
    )
