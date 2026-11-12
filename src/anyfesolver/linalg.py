"""Sparse linear algebra backends for FE analyses.

SciPy SuperLU is the universal fallback. ``AutoSparseSolverBackend`` can select
PyPardiso for sufficiently large systems, with matrix-class-aware mtypes,
pattern-slot symbolic reuse, bounded cache size, and general-matrix fallback.
All paths report backend, ordering, timings, reuse, and failure diagnostics
through the same factorization-handle API.
"""

from __future__ import annotations

import ctypes
from functools import wraps
import importlib.util
import site
import sys
import time
import warnings
import weakref
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import LinearOperator, splu

from .threading_policy import (
    current_solver_threads,
    _inherited_native_thread_policy,
    native_thread_scope,
)

_NUMERIC_OPERATION_LOCK = threading.Lock()
_NUMERIC_OPERATIONS = {
    "scipy_factorization_attempts": 0,
    "pardiso_factorization_attempts": 0,
    "pardiso_stale_rebuilds": 0,
}


def backend_operation_counts() -> Dict[str, int]:
    """Process-wide monotonic counts of actual numerical backend operations.

    Differences across an isolated run include failed attempts, PARDISO phase
    22 refreshes and hidden stale-handle refactorizations. These are operations,
    not cache misses; concurrent analyses are intentionally not attributed to
    an individual session.
    """
    with _NUMERIC_OPERATION_LOCK:
        return dict(_NUMERIC_OPERATIONS)


def _count_numeric_operation(name: str) -> None:
    with _NUMERIC_OPERATION_LOCK:
        _NUMERIC_OPERATIONS[name] += 1

try:
    _HAS_PYPARDISO = importlib.util.find_spec("pypardiso") is not None
except (ImportError, ModuleNotFoundError, ValueError):
    _HAS_PYPARDISO = False

_PYPARDISO_SOLVER_CLASS: Any = None
_PYPARDISO_PROCESS_LOCK = threading.RLock()


def _serialized_pypardiso_call(function):
    """Serialize access to process-global PyPardiso/MKL solver state."""

    @wraps(function)
    def serialized(*args, **kwargs):
        with _PYPARDISO_PROCESS_LOCK:
            return function(*args, **kwargs)

    return serialized


@_serialized_pypardiso_call
def _new_pypardiso_solver(*, mtype: int) -> Any:
    """Construct PyPardiso only after the backend has actually been selected."""

    global _PYPARDISO_SOLVER_CLASS
    if not _HAS_PYPARDISO:
        raise ImportError("pypardiso is not installed")
    if _PYPARDISO_SOLVER_CLASS is None:
        from pypardiso import PyPardisoSolver as solver_class

        _PYPARDISO_SOLVER_CLASS = solver_class
    return _PYPARDISO_SOLVER_CLASS(mtype=int(mtype))

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper
    prange = range
    _HAS_NUMBA = False


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return parsed if parsed > 0 else int(default)


class MatrixClass(str, Enum):
    """Declared numerical class of a sparse matrix."""

    SPD = "spd"
    SYMMETRIC_SEMIDEFINITE = "symmetric_semidefinite"
    SYMMETRIC_INDEFINITE = "symmetric_indefinite"
    GENERAL = "general"


@dataclass
class FactorizationHandle:
    """Reusable sparse factorization and solve diagnostics."""

    matrix_shape: tuple[int, int]
    matrix_class: MatrixClass
    backend_name: str
    ordering: str
    signature: Optional[str]
    factorization_time: float
    factorization_count: int = 1
    solve_count: int = 0
    solve_time: float = 0.0
    status: str = "ok"
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    solver_threads: Optional[int] = field(default=None, repr=False)
    _solver: Any = field(default=None, repr=False)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return solve(self, rhs)

    def solve_many(self, rhs_matrix: np.ndarray) -> np.ndarray:
        return solve_many(self, rhs_matrix)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "matrix_class": self.matrix_class.value,
            "ordering": self.ordering,
            "signature": self.signature,
            "shape": list(self.matrix_shape),
            "status": self.status,
            "failure_reason": self.failure_reason,
            "factorization_time": self.factorization_time,
            "factorization_count": self.factorization_count + int(
                getattr(self._solver, "stale_rebuild_count", 0)
            ),
            "solve_count": self.solve_count,
            "solve_time": self.solve_time,
            **self.metadata,
        }


class SparseSolverBackend:
    """SciPy/SuperLU sparse backend with a stable FE-facing interface."""

    name = "scipy_superlu"

    def factorize(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> FactorizationHandle:
        options = dict(options or {})
        ordering = str(options.get("ordering", "COLAMD"))
        start = time.time()
        try:
            csc = sparse.csc_matrix(matrix)
            _count_numeric_operation("scipy_factorization_attempts")
            solver = splu(csc, permc_spec=ordering)
        except Exception as exc:
            return FactorizationHandle(
                matrix_shape=tuple(int(v) for v in matrix.shape),
                matrix_class=matrix_class,
                backend_name=self.name,
                ordering=ordering,
                signature=signature,
                factorization_time=time.time() - start,
                status="failed",
                failure_reason=str(exc),
            )
        return FactorizationHandle(
            matrix_shape=tuple(int(v) for v in matrix.shape),
            matrix_class=matrix_class,
            backend_name=self.name,
            ordering=ordering,
            signature=signature,
            factorization_time=time.time() - start,
            metadata={
                "matrix_nnz": int(csc.nnz),
                "factor_nnz": int(solver.L.nnz + solver.U.nnz),
                "factor_fill_ratio": float(
                    (solver.L.nnz + solver.U.nnz) / max(int(csc.nnz), 1)
                ),
                "factor_storage_bytes": int(
                    solver.L.data.nbytes
                    + solver.L.indices.nbytes
                    + solver.L.indptr.nbytes
                    + solver.U.data.nbytes
                    + solver.U.indices.nbytes
                    + solver.U.indptr.nbytes
                ),
                "matrix_storage_bytes": int(
                    csc.data.nbytes + csc.indices.nbytes + csc.indptr.nbytes
                ),
            },
            _solver=solver,
        )


_MKL_RT_ENV_READY = False


def _ensure_mkl_rt_env() -> None:
    """Locate mkl_rt once before importing PyPardiso.

    Windows MKL wheels normally place the runtime directly in
    ``<python>/Library/bin``.  Probe a small set of standard locations
    non-recursively; pypardiso's recursive fallback can otherwise dominate
    first-use time.
    """

    global _MKL_RT_ENV_READY
    if _MKL_RT_ENV_READY or not _HAS_PYPARDISO:
        return
    if os.environ.get("PYPARDISO_MKL_RT"):
        _MKL_RT_ENV_READY = True
        return
    from ctypes.util import find_library

    path = find_library("mkl_rt") or find_library("mkl_rt.1")
    if path is None:
        prefixes = [
            os.environ.get("CONDA_PREFIX"), sys.prefix,
            sys.base_prefix, sys.base_exec_prefix, site.USER_BASE,
        ]
        directories: List[Path] = []
        seen_prefixes = set()
        for prefix in prefixes:
            if not prefix:
                continue
            normalized = os.path.normcase(os.path.abspath(str(prefix)))
            if normalized in seen_prefixes:
                continue
            seen_prefixes.add(normalized)
            base = Path(prefix)
            directories.extend((base / "Library" / "bin", base / "DLLs", base / "lib"))
        candidates: List[Path] = []
        for directory in directories:
            try:
                candidates.extend(directory.glob("mkl_rt*"))
            except OSError:
                continue
        unique_candidates = {
            os.path.normcase(os.path.abspath(str(candidate)))
            for candidate in candidates
        }
        for candidate_text in sorted(unique_candidates, key=len):
            try:
                ctypes.CDLL(candidate_text)
            except OSError:
                continue
            path = candidate_text
            break
    if path:
        os.environ["PYPARDISO_MKL_RT"] = path
    _MKL_RT_ENV_READY = True


def _pardiso_mtype_candidates(matrix_class: MatrixClass) -> Tuple[int, ...]:
    """PARDISO mtypes to try for a declared matrix class, best first.

    Symmetric classes factorize the upper triangle (mtype 2 Cholesky for SPD,
    mtype -2 Bunch-Kaufman otherwise) with the general real path (mtype 11) as
    the final fallback.
    """

    if matrix_class == MatrixClass.SPD:
        return (2, -2, 11)
    if matrix_class in (MatrixClass.SYMMETRIC_SEMIDEFINITE, MatrixClass.SYMMETRIC_INDEFINITE):
        return (-2, 11)
    return (11,)


def _pardiso_prepared_matrix(csr: sparse.csr_matrix, mtype: int) -> sparse.csr_matrix:
    """Return the CSR matrix PARDISO should factorize for the given mtype.

    Symmetric mtypes take the upper triangle only; MKL additionally requires
    every diagonal entry to be structurally present, so missing diagonals are
    inserted as explicit zeros.
    """

    if mtype == 11:
        return csr
    upper = sparse.triu(csr, format="csr")
    diagonal = upper.diagonal()
    if np.any(diagonal == 0.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            upper.setdiag(diagonal)
        upper = sparse.csr_matrix(upper)
    upper.sort_indices()
    return upper


@_serialized_pypardiso_call
def _release_mkl_solver(solver: Any) -> None:
    """Release MKL's internal factorization memory (PARDISO phase -1)."""
    try:
        solver.free_memory(everything=True)
    except Exception:
        pass


@_serialized_pypardiso_call
def _pardiso_full_factorize(solver: Any, prepared: sparse.csr_matrix) -> None:
    solver._check_A(prepared)
    solver.set_phase(12)
    _count_numeric_operation("pardiso_factorization_attempts")
    solver._call_pardiso(prepared, np.zeros((prepared.shape[0], 1)))


class _PardisoPatternSlot:
    """One retained PARDISO instance per sparsity pattern for analysis reuse."""

    __slots__ = ("solver", "mtype", "shape", "indptr", "indices", "generation")

    def __init__(self, solver: Any, mtype: int, prepared: sparse.csr_matrix):
        self.solver = solver
        self.mtype = int(mtype)
        self.shape = tuple(int(v) for v in prepared.shape)
        self.indptr = prepared.indptr.copy()
        self.indices = prepared.indices.copy()
        self.generation = 0

    @_serialized_pypardiso_call
    def matches(self, prepared: sparse.csr_matrix, mtype: int) -> bool:
        return (
            self.solver is not None
            and self.mtype == int(mtype)
            and self.shape == tuple(int(v) for v in prepared.shape)
            and self.indptr.size == prepared.indptr.size
            and self.indices.size == prepared.indices.size
            and np.array_equal(self.indptr, prepared.indptr)
            and np.array_equal(self.indices, prepared.indices)
        )

    @_serialized_pypardiso_call
    def release(self) -> None:
        self.generation += 1
        solver, self.solver = self.solver, None
        if solver is not None:
            _release_mkl_solver(solver)


class _PardisoFactorization:
    """Solve interface bound to a pattern slot.

    The slot's PARDISO instance holds the factorization of the *most recent*
    matrix with that sparsity pattern.  A handle created earlier therefore
    checks a generation token before solving; if the slot has moved on (or was
    evicted), the handle transparently refactorizes its own matrix into a
    private solver whose MKL memory is released by a finalizer when the handle
    is garbage collected.
    """

    def __init__(self, slot: _PardisoPatternSlot, prepared: sparse.csr_matrix, mtype: int):
        self._slot: Optional[_PardisoPatternSlot] = slot
        self._generation = slot.generation
        self._matrix = prepared
        self._mtype = int(mtype)
        self._private_solver: Any = None
        self.stale_rebuild_count = 0

    @_serialized_pypardiso_call
    def _active_solver(self) -> Any:
        if self._private_solver is not None:
            return self._private_solver
        slot = self._slot
        if slot is not None and slot.solver is not None and slot.generation == self._generation:
            return slot.solver
        solver = _new_pypardiso_solver(mtype=self._mtype)
        _pardiso_full_factorize(solver, self._matrix)
        self._private_solver = solver
        weakref.finalize(self, _release_mkl_solver, solver)
        self._slot = None
        self.stale_rebuild_count += 1
        _count_numeric_operation("pardiso_stale_rebuilds")
        return solver

    @_serialized_pypardiso_call
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        solver = self._active_solver()
        b = solver._check_b(self._matrix, np.asarray(rhs, dtype=np.float64))
        solver.set_phase(33)
        return solver._call_pardiso(self._matrix, b)


class PyPardisoSolverBackend:
    """Intel MKL PARDISO backend using pypardiso.

    Optimizations over naive pypardiso usage:

    - mkl_rt is resolved once per process (``PYPARDISO_MKL_RT``) instead of
      re-searching the filesystem on every ``PyPardisoSolver`` construction;
    - symmetric matrix classes factorize the upper triangle with symmetric
      mtypes (2 / -2), falling back to the general path on numerical failure;
    - refactorizations with an unchanged sparsity pattern reuse the symbolic
      analysis (PARDISO phase 22) through a small LRU of pattern slots;
    - MKL internal memory is bounded and released: evicted slots and privately
      rebuilt factorizations free their memory (phase -1) via finalizers.

    PyPardiso/MKL solver state is serialized process-wide.  This includes
    independent backend instances and handles owned by separate analysis
    caches; SciPy/SuperLU operations do not acquire this lock.
    """

    name = "pypardiso"

    def __init__(self, *, max_pattern_slots: int = 4):
        self.max_pattern_slots = _env_int("FE_SOLVER_PYPARDISO_MAX_PATTERN_SLOTS", int(max_pattern_slots))
        self._slots: List[_PardisoPatternSlot] = []

    @_serialized_pypardiso_call
    def release_pattern_slots(self) -> None:
        """Release all retained MKL factorization memory."""
        while self._slots:
            self._slots.pop().release()

    @property
    @_serialized_pypardiso_call
    def initialized(self) -> bool:
        """Whether this backend has completed a retained factorization."""

        return bool(self._slots)

    @property
    @_serialized_pypardiso_call
    def retained_pattern_slots(self) -> int:
        """Number of live symbolic-analysis slots retained by the backend."""

        return len(self._slots)

    def _has_compatible_pattern_unlocked(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass,
    ) -> bool:
        if not self._slots:
            return False
        try:
            csr = sparse.csr_matrix(matrix, copy=True)
            csr.sort_indices()
            for mtype in _pardiso_mtype_candidates(matrix_class):
                prepared = _pardiso_prepared_matrix(csr, mtype)
                if any(slot.matches(prepared, mtype) for slot in self._slots):
                    return True
        except Exception:
            return False
        return False

    @_serialized_pypardiso_call
    def has_compatible_pattern(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass,
    ) -> bool:
        """Return whether a retained slot can reuse this matrix's analysis.

        A prior PARDISO solve alone is not enough to justify the lower warm
        thresholds: reuse requires the same prepared sparsity pattern and a
        PARDISO mtype compatible with the incoming matrix class.
        """

        return self._has_compatible_pattern_unlocked(matrix, matrix_class)

    @_serialized_pypardiso_call
    def selection_state(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass,
    ) -> Tuple[bool, int, bool]:
        """Return one coherent auto-selection snapshot under the process lock."""

        return (
            bool(self._slots),
            len(self._slots),
            self._has_compatible_pattern_unlocked(matrix, matrix_class),
        )

    @_serialized_pypardiso_call
    def _factorize_prepared(self, prepared: sparse.csr_matrix, mtype: int) -> Tuple[_PardisoFactorization, bool]:
        for slot in self._slots:
            if slot.matches(prepared, mtype):
                slot.generation += 1
                try:
                    slot.solver.set_phase(22)
                    _count_numeric_operation("pardiso_factorization_attempts")
                    slot.solver._call_pardiso(prepared, np.zeros((prepared.shape[0], 1)))
                except Exception:
                    _pardiso_full_factorize(slot.solver, prepared)
                self._slots.remove(slot)
                self._slots.insert(0, slot)
                return _PardisoFactorization(slot, prepared, mtype), True
        solver = _new_pypardiso_solver(mtype=int(mtype))
        _pardiso_full_factorize(solver, prepared)
        slot = _PardisoPatternSlot(solver, mtype, prepared)
        self._slots.insert(0, slot)
        while len(self._slots) > max(int(self.max_pattern_slots), 1):
            self._slots.pop().release()
        return _PardisoFactorization(slot, prepared, mtype), False

    @_serialized_pypardiso_call
    def factorize(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> FactorizationHandle:
        start = time.time()
        _ensure_mkl_rt_env()
        failure_reasons: List[str] = []
        try:
            csr = sparse.csr_matrix(matrix)
            if csr is matrix:
                csr = csr.copy()
            csr.sort_indices()
        except Exception as exc:
            failure_reasons.append(str(exc))
            csr = None
        wrapper: Optional[_PardisoFactorization] = None
        used_mtype: Optional[int] = None
        symbolic_reused = False
        if csr is not None:
            for mtype in _pardiso_mtype_candidates(matrix_class):
                try:
                    prepared = _pardiso_prepared_matrix(csr, mtype)
                    wrapper, symbolic_reused = self._factorize_prepared(prepared, mtype)
                    used_mtype = int(mtype)
                    break
                except Exception as exc:
                    failure_reasons.append(f"mtype={mtype}: {exc}")
        if wrapper is None:
            return FactorizationHandle(
                matrix_shape=tuple(int(v) for v in matrix.shape),
                matrix_class=matrix_class,
                backend_name=self.name,
                ordering="MKL PARDISO",
                signature=signature,
                factorization_time=time.time() - start,
                status="failed",
                failure_reason="; ".join(failure_reasons) or "unknown pypardiso failure",
            )
        handle = FactorizationHandle(
            matrix_shape=tuple(int(v) for v in matrix.shape),
            matrix_class=matrix_class,
            backend_name=self.name,
            ordering="MKL PARDISO",
            signature=signature,
            factorization_time=time.time() - start,
            _solver=wrapper,
        )
        handle.metadata["pardiso_mtype"] = used_mtype
        handle.metadata["pardiso_symbolic_reused"] = bool(symbolic_reused)
        handle.metadata["pardiso_pattern_slots"] = len(self._slots)
        if failure_reasons:
            handle.metadata["pardiso_fallback_attempts"] = list(failure_reasons)
        return handle


class AutoSparseSolverBackend:
    """Size-aware backend selector with SciPy fallback.

    PyPardiso has substantial setup overhead on tiny systems.  The auto backend
    keeps SuperLU as the fast small-matrix path and uses PyPardiso only once the
    matrix is large enough to amortize that overhead.  Callers may force a
    backend with ``options={"backend": "pypardiso"}`` or
    ``options={"backend": "scipy_superlu"}``.
    """

    name = "auto"

    def __init__(
        self,
        *,
        scipy_backend: Optional[SparseSolverBackend] = None,
        pardiso_backend: Optional[PyPardisoSolverBackend] = None,
        pypardiso_min_dimension: int = 10_000,
        pypardiso_min_nnz: int = 250_000,
        pypardiso_warm_min_dimension: int = 1_000,
        pypardiso_warm_min_nnz: int = 25_000,
    ):
        self.scipy_backend = scipy_backend or SparseSolverBackend()
        self.pardiso_backend = (pardiso_backend or PyPardisoSolverBackend()) if _HAS_PYPARDISO else None
        self.pypardiso_min_dimension = _env_int("FE_SOLVER_PYPARDISO_MIN_DIMENSION", int(pypardiso_min_dimension))
        self.pypardiso_min_nnz = _env_int("FE_SOLVER_PYPARDISO_MIN_NNZ", int(pypardiso_min_nnz))
        self.pypardiso_warm_min_dimension = _env_int(
            "FE_SOLVER_PYPARDISO_WARM_MIN_DIMENSION",
            int(pypardiso_warm_min_dimension),
        )
        self.pypardiso_warm_min_nnz = _env_int(
            "FE_SOLVER_PYPARDISO_WARM_MIN_NNZ",
            int(pypardiso_warm_min_nnz),
        )

    def release_pattern_slots(self) -> None:
        """Release MKL factorization memory retained for pattern reuse."""
        if self.pardiso_backend is not None:
            self.pardiso_backend.release_pattern_slots()

    def _selection_thresholds(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass,
        options: Mapping[str, Any],
        *,
        compatible_pattern: Optional[bool] = None,
    ) -> Tuple[int, int, bool]:
        if compatible_pattern is None:
            compatible_pattern = bool(
                self.pardiso_backend
                and self.pardiso_backend.has_compatible_pattern(
                    matrix, matrix_class
                )
            )
        default_dimension = (
            self.pypardiso_warm_min_dimension
            if compatible_pattern
            else self.pypardiso_min_dimension
        )
        default_nnz = self.pypardiso_warm_min_nnz if compatible_pattern else self.pypardiso_min_nnz
        min_dimension = int(options.get("pypardiso_min_dimension", default_dimension))
        min_nnz = int(options.get("pypardiso_min_nnz", default_nnz))
        return min_dimension, min_nnz, compatible_pattern

    def _use_pypardiso(
        self,
        matrix: sparse.spmatrix,
        options: Mapping[str, Any],
        min_dimension: int,
        min_nnz: int,
    ) -> bool:
        requested = str(options.get("backend", options.get("solver", "auto"))).lower()
        if requested in {"scipy", "scipy_superlu", "superlu"}:
            return False
        if requested in {"pypardiso", "pardiso", "mkl_pardiso"}:
            return self.pardiso_backend is not None
        if self.pardiso_backend is None:
            return False
        return max(int(matrix.shape[0]), int(matrix.shape[1])) >= min_dimension and int(matrix.nnz) >= min_nnz

    def factorize(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> FactorizationHandle:
        options_dict = dict(options or {})
        if self.pardiso_backend is None:
            was_initialized = False
            retained_pattern_slots = 0
            compatible_pattern = False
        else:
            (
                was_initialized,
                retained_pattern_slots,
                compatible_pattern,
            ) = self.pardiso_backend.selection_state(matrix, matrix_class)
        active_dimension, active_nnz, compatible_pattern = self._selection_thresholds(
            matrix,
            matrix_class,
            options_dict,
            compatible_pattern=compatible_pattern,
        )
        selection_metadata = {
            "pypardiso_active_min_dimension": active_dimension,
            "pypardiso_active_min_nnz": active_nnz,
            "pypardiso_initialized_before_selection": was_initialized,
            "pypardiso_compatible_pattern_before_selection": compatible_pattern,
            "pypardiso_warm_thresholds_active": compatible_pattern,
            "pypardiso_retained_pattern_slots_before_selection": retained_pattern_slots,
        }
        if not self._use_pypardiso(matrix, options_dict, active_dimension, active_nnz):
            handle = self.scipy_backend.factorize(matrix, matrix_class, signature=signature, options=options_dict)
            handle.metadata.setdefault("auto_backend_policy", "scipy_small_matrix")
            handle.metadata.setdefault("pypardiso_min_dimension", self.pypardiso_min_dimension)
            handle.metadata.setdefault("pypardiso_min_nnz", self.pypardiso_min_nnz)
            for key, value in selection_metadata.items():
                handle.metadata.setdefault(key, value)
            handle.metadata.setdefault(
                "pypardiso_initialized",
                was_initialized,
            )
            return handle

        assert self.pardiso_backend is not None
        handle = self.pardiso_backend.factorize(matrix, matrix_class, signature=signature, options=options_dict)
        requested = str(options_dict.get("backend", options_dict.get("solver", "auto"))).lower()
        if requested in {"pypardiso", "pardiso", "mkl_pardiso"}:
            selection_policy = "pypardiso_forced"
        elif compatible_pattern:
            selection_policy = "pypardiso_compatible_pattern"
        else:
            selection_policy = "pypardiso_large_matrix"
        handle.metadata.setdefault("auto_backend_policy", selection_policy)
        handle.metadata.setdefault("pypardiso_min_dimension", self.pypardiso_min_dimension)
        handle.metadata.setdefault("pypardiso_min_nnz", self.pypardiso_min_nnz)
        for key, value in selection_metadata.items():
            handle.metadata.setdefault(key, value)
        handle.metadata.setdefault("pypardiso_initialized", bool(self.pardiso_backend.initialized))
        if handle.status == "ok":
            return handle

        fallback = self.scipy_backend.factorize(matrix, matrix_class, signature=signature, options=options_dict)
        fallback.metadata["auto_backend_policy"] = "scipy_after_pypardiso_failure"
        fallback.metadata["pypardiso_min_dimension"] = self.pypardiso_min_dimension
        fallback.metadata["pypardiso_min_nnz"] = self.pypardiso_min_nnz
        fallback.metadata.update(selection_metadata)
        fallback.metadata["fallback_from_backend"] = handle.backend_name
        fallback.metadata["fallback_failure_reason"] = handle.failure_reason
        return fallback


# One size-aware policy in every environment (it falls back to SuperLU without
# PyPardiso). Choosing it here, not later on import of the nonlinear bootstrap,
# keeps the default fixed for the whole process, which the prepared static
# evaluator's binding checks rely on.
DEFAULT_BACKEND = AutoSparseSolverBackend()


def _options_signature(options: Optional[Mapping[str, Any]]) -> str:
    return json.dumps(dict(options or {}), sort_keys=True, default=str, separators=(",", ":"))


def sparse_matrix_signature(matrix: sparse.spmatrix) -> str:
    """Content signature for a sparse matrix used in local factorization caches."""

    csr = sparse.csr_matrix(matrix)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(v) for v in csr.shape)).encode("ascii"))
    digest.update(str(int(csr.nnz)).encode("ascii"))
    digest.update(np.asarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass
class FactorizationCache:
    """Explicit local cache for sparse factorizations.

    The cache is intentionally not global.  Analyses that can safely reuse a
    matrix factorization own the cache and therefore own its lifetime.
    Cache lookup, insertion, eviction and counters are protected for concurrent
    callers.  Returned handles retain the concurrency semantics of their
    numerical backend; the cache does not serialize subsequent ``solve`` calls.
    """

    name: str = "factorization_cache"
    max_entries: int = 8
    backend: Optional[SparseSolverBackend] = None
    _handles: Dict[Tuple[str, str, str, str], FactorizationHandle] = field(default_factory=dict, init=False, repr=False)
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)
    hits: int = 0
    misses: int = 0
    factorization_failures: int = 0

    def key(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass | str,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[str, str, str, str]:
        matrix_key = str(signature) if signature is not None else sparse_matrix_signature(matrix)
        return (
            matrix_key,
            _coerce_matrix_class(matrix_class).value,
            _options_signature(options),
            tuple(int(v) for v in matrix.shape).__repr__(),
        )

    def factorize(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass | str,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> FactorizationHandle:
        cache_key = self.key(matrix, matrix_class, signature=signature, options=options)
        with self._lock:
            if cache_key in self._handles:
                self.hits += 1
                handle = self._handles[cache_key]
                options_dict = dict(options or {})
                handle.solver_threads = options_dict.get("solver_threads", current_solver_threads())
                handle.metadata["cache_name"] = self.name
                handle.metadata["cache_hit"] = True
                return handle
            self.misses += 1
            handle = factorize(
                matrix,
                matrix_class,
                signature=signature or cache_key[0],
                options=options,
                backend=self.backend,
            )
            handle.metadata["cache_name"] = self.name
            handle.metadata["cache_hit"] = False
            if handle.status == "ok":
                if self.max_entries > 0 and len(self._handles) >= self.max_entries:
                    oldest = next(iter(self._handles))
                    self._handles.pop(oldest, None)
                if self.max_entries != 0:
                    self._handles[cache_key] = handle
            else:
                self.factorization_failures += 1
            return handle

    def set_backend_if_absent(self, backend: SparseSolverBackend) -> None:
        """Select a backend once without racing concurrent cache users."""

        with self._lock:
            if self.backend is None:
                self.backend = backend

    def linear_operator(
        self,
        matrix: sparse.spmatrix,
        matrix_class: MatrixClass | str,
        *,
        signature: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> tuple[LinearOperator, FactorizationHandle]:
        """Return a LinearOperator that applies the cached inverse."""

        handle = self.factorize(matrix, matrix_class, signature=signature, options=options)
        if handle.status != "ok":
            raise RuntimeError(f"Cannot create inverse operator from failed factorization: {handle.failure_reason}")

        def matvec(rhs: np.ndarray) -> np.ndarray:
            return handle.solve(rhs)

        operator = LinearOperator(matrix.shape, matvec=matvec, dtype=float)
        return operator, handle

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "max_entries": int(self.max_entries),
                "entries": int(len(self._handles)),
                "hits": int(self.hits),
                "misses": int(self.misses),
                "factorization_failures": int(self.factorization_failures),
                "backend": (self.backend or DEFAULT_BACKEND).name,
            }

    def clear(self) -> None:
        with self._lock:
            self._handles.clear()


def _coerce_matrix_class(matrix_class: MatrixClass | str) -> MatrixClass:
    if isinstance(matrix_class, MatrixClass):
        return matrix_class
    return MatrixClass(str(matrix_class))


def factorize(
    matrix: sparse.spmatrix,
    matrix_class: MatrixClass | str,
    *,
    signature: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
    backend: Optional[SparseSolverBackend] = None,
) -> FactorizationHandle:
    """Factorize a sparse matrix and return a reusable handle."""
    backend = backend or DEFAULT_BACKEND
    options_dict = dict(options or {})
    requested_threads = options_dict.pop("solver_threads", current_solver_threads())
    policy = _inherited_native_thread_policy(
        requested_threads, phase="factorization"
    )
    if policy is None:
        with native_thread_scope(
            requested_threads, phase="factorization"
        ) as scoped_policy:
            handle = backend.factorize(
                matrix,
                _coerce_matrix_class(matrix_class),
                signature=signature,
                options=options_dict,
            )
        policy = scoped_policy
    else:
        handle = backend.factorize(
            matrix,
            _coerce_matrix_class(matrix_class),
            signature=signature,
            options=options_dict,
        )
    handle.solver_threads = requested_threads
    handle.metadata["thread_policy"] = dict(policy)
    return handle


def factorize_cached(
    matrix: sparse.spmatrix,
    matrix_class: MatrixClass | str,
    *,
    cache: Optional[FactorizationCache] = None,
    signature: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
    backend: Optional[SparseSolverBackend] = None,
) -> FactorizationHandle:
    """Factorize through a local cache when supplied."""

    if cache is None:
        return factorize(matrix, matrix_class, signature=signature, options=options, backend=backend)
    if backend is not None:
        cache.set_backend_if_absent(backend)
    return cache.factorize(matrix, matrix_class, signature=signature, options=options)


def cached_inverse_operator(
    matrix: sparse.spmatrix,
    matrix_class: MatrixClass | str,
    *,
    cache: Optional[FactorizationCache] = None,
    signature: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> tuple[LinearOperator, FactorizationHandle]:
    """Build a sparse inverse operator, using a local cache if supplied."""

    local_cache = cache or FactorizationCache(name="single_inverse_operator", max_entries=1)
    return local_cache.linear_operator(matrix, matrix_class, signature=signature, options=options)


def solve(handle: FactorizationHandle, rhs: np.ndarray) -> np.ndarray:
    """Solve one right-hand side using an existing factorization."""
    if handle.status != "ok" or handle._solver is None:
        raise RuntimeError(f"Cannot solve with failed factorization: {handle.failure_reason}")
    rhs = np.asarray(rhs, dtype=float)
    start = time.time()
    policy = _inherited_native_thread_policy(
        handle.solver_threads, phase="linear_solve"
    )
    if policy is None:
        with native_thread_scope(
            handle.solver_threads, phase="linear_solve"
        ) as scoped_policy:
            result = np.asarray(handle._solver.solve(rhs), dtype=float)
        policy = scoped_policy
    else:
        result = np.asarray(handle._solver.solve(rhs), dtype=float)
    handle.metadata["last_solve_thread_policy"] = dict(policy)
    handle.solve_time += time.time() - start
    handle.solve_count += 1 if rhs.ndim == 1 else int(rhs.shape[1])
    if np.any(~np.isfinite(result)):
        raise RuntimeError("Sparse solve produced NaN/Inf values")
    return result


def solve_many(handle: FactorizationHandle, rhs_matrix: np.ndarray) -> np.ndarray:
    """Solve one or more right-hand sides with one numerical factorization."""
    rhs = np.asarray(rhs_matrix, dtype=float)
    if rhs.ndim == 1:
        return solve(handle, rhs)
    if rhs.ndim != 2:
        raise ValueError("rhs_matrix must be one- or two-dimensional")
    return solve(handle, rhs)
