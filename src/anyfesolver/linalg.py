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
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

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
    "pardiso_lease_conflicts": 0,
    "pardiso_lease_detachments": 0,
}


def backend_operation_counts() -> Dict[str, int]:
    """Process-wide monotonic counts of actual numerical backend operations.

    Differences across an isolated run include failed attempts and PARDISO
    phase 22 refreshes. These are operations, not cache misses; concurrent
    analyses are intentionally not attributed to an individual session.
    ``pardiso_stale_rebuilds`` is retained for comparability with earlier
    releases and stays 0: a stale PARDISO handle is never rebuilt (it raises
    :class:`PardisoStaleHandleError`).

    ``pardiso_lease_conflicts`` counts factorizations whose pattern slot was
    pinned by a :class:`FactorizationLease`: the newcomer took a fresh
    analysis (phase 12) in a new slot where an unleased run would have reused
    the pinned slot's analysis (phase 22).  While it stays 0, every
    factorization takes the same phase 12/22 path as without leases.
    ``pardiso_lease_detachments`` counts pinned slots that were evicted,
    released or refactorized while leased; they stay valid for their leases
    and are freed when the last lease ends.
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
            "factorization_count": self.factorization_count,
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


def _free_mkl_solver(solver: Any) -> Optional[BaseException]:
    """Run PARDISO phase -1 once; return the raised exception, or ``None`` if freed."""

    with _PYPARDISO_PROCESS_LOCK:
        try:
            solver.free_memory(everything=True)
        except BaseException as exc:  # reported to the caller, never swallowed silently
            return exc
    return None


@_serialized_pypardiso_call
def _release_mkl_solver(solver: Any) -> None:
    """Release MKL's internal factorization memory (PARDISO phase -1)."""
    _free_mkl_solver(solver)


class _MklSolverOwner:
    """Sole owner of one MKL PARDISO instance's native memory.

    Created right after the instance, before any numerical phase.  Its
    finalizer runs phase -1 exactly once: on the first explicit
    :meth:`release` or, for an owner dropped without release, at garbage
    collection, so native memory can be delayed but never leaked.  The
    release is reported as done only if ``free_memory`` returned normally.
    """

    __slots__ = ("solver", "_finalizer", "__weakref__")

    def __init__(self, solver: Any) -> None:
        self.solver = solver
        finalizer = weakref.finalize(self, _free_mkl_solver, solver)
        finalizer.atexit = False
        self._finalizer = finalizer

    @property
    def alive(self) -> bool:
        return self._finalizer.alive

    def release(self) -> Tuple[bool, Optional[BaseException]]:
        """Free the instance once; return ``(ran_now, error_from_free_memory)``."""

        with _PYPARDISO_PROCESS_LOCK:
            if not self._finalizer.alive:
                return False, None
            return True, self._finalizer()


def _new_mkl_owner(mtype: int) -> _MklSolverOwner:
    solver = _new_pypardiso_solver(mtype=int(mtype))
    try:
        return _MklSolverOwner(solver)
    except BaseException:
        _free_mkl_solver(solver)
        raise


def _release_note(error: Optional[BaseException], ran: bool = True) -> str:
    if error is None:
        return "its native memory was released (phase -1)" if ran else "its native memory was already released"
    return f"releasing its native memory failed ({type(error).__name__}: {error})"


def _add_note(exc: BaseException, note: str) -> None:
    try:
        exc.add_note(note)
    except Exception:
        pass


def _raise_interrupt(error: Optional[BaseException], context: str) -> None:
    """Re-raise a non-``Exception`` cleanup error, or attach it to the active exception.

    ``Exception`` failures of ``free_memory`` are tolerated as before (the
    instance is never used again); an interrupt is not swallowed, but never
    replaces an exception that is already propagating.
    """

    if error is None or isinstance(error, Exception):
        return
    active = sys.exc_info()[1]
    if active is not None and active is not error:
        _add_note(active, f"{context} raised {type(error).__name__}: {error}")
        return
    raise error


PARDISO_PERTURBED_PIVOTS = "perturbed_pivots"
# Metadata value for mtypes whose iparm[13] is not a documented pivot count.
# Intel oneMKL documents output iparm(14) (perturbed pivots) for the symmetric
# indefinite (-2) and general (11) paths only; for mtype 2 (Cholesky) a read
# value is never reported as a measured count.  A positive read value is still
# refused (fail closed).
PARDISO_PIVOT_COUNT_UNSUPPORTED = "unsupported"
_PARDISO_PIVOT_COUNT_DOCUMENTED_MTYPES = frozenset({-2, 11})


def _reported_pivot_count(mtype: int, count: Optional[int]) -> Any:
    if int(mtype) in _PARDISO_PIVOT_COUNT_DOCUMENTED_MTYPES:
        return count
    return PARDISO_PIVOT_COUNT_UNSUPPORTED


class PardisoPerturbedPivotError(RuntimeError):
    """A PARDISO numerical factorization perturbed (near-)zero pivots.

    MKL PARDISO does not fail on a (near-)singular pivot for the symmetric
    indefinite and general mtypes: it replaces the pivot by a small value and
    reports the count in output ``iparm(14)`` (0-based ``iparm[13]``).  The
    resulting factorization belongs to a different matrix, so it is refused.
    ``count`` is ``None`` when the count could not be read (also refused).
    ``mtype`` (when known) selects the wording: for an mtype whose
    ``iparm[13]`` is not a documented pivot count (mtype 2) a positive raw
    value is reported as such, never as a number of perturbed pivots.
    """

    def __init__(
        self,
        count: Optional[int],
        *,
        phase: int,
        detail: str = "",
        mtype: Optional[int] = None,
    ) -> None:
        self.count = count
        self.phase = int(phase)
        self.detail = str(detail)
        self.mtype = None if mtype is None else int(mtype)
        if count is None:
            body = f"perturbed-pivot count iparm[13] is unreadable ({detail})"
        elif self.mtype is not None and self.mtype not in _PARDISO_PIVOT_COUNT_DOCUMENTED_MTYPES:
            body = (
                f"(mtype {self.mtype}) returned raw iparm[13]={int(count)}, which is not "
                f"a documented perturbed-pivot count for mtype {self.mtype}; refused "
                "fail closed"
            )
        else:
            body = f"perturbed {int(count)} pivot(s) (iparm[13])"
        super().__init__(f"{PARDISO_PERTURBED_PIVOTS}: PARDISO phase {int(phase)} {body}")

    def for_mtype(self, mtype: int) -> "PardisoPerturbedPivotError":
        """The same rejection, worded for ``mtype`` (chained to this one)."""

        error = PardisoPerturbedPivotError(
            self.count, phase=self.phase, detail=self.detail, mtype=mtype
        )
        error.__cause__ = self
        return error


PARDISO_STALE_HANDLE = "pardiso_stale_handle"


class PardisoStaleHandleError(RuntimeError):
    """A PARDISO handle whose factorization no longer exists was solved.

    PARDISO handles are bound to a retained pattern slot of their backend.
    The slot holds the factorization of the *most recent* matrix with that
    sparsity pattern: once it is refactorized by another matrix, evicted from
    the backend's LRU, released (``release_pattern_slots``) or retired after
    an interrupted factorization, every handle created earlier is stale.  A
    stale handle is never rebuilt: each solve (sequential or concurrent)
    raises this error with the same message, without any native call and
    without returning a solution.  Factorize again to continue;
    :class:`FactorizationCache` treats a stale cached handle as a miss.
    Holders that solve one factorization across other work pin it with a
    :class:`FactorizationLease` (:func:`leased_factorization`,
    :func:`lease_handle`), which keeps it from ever becoming stale.
    """

    MESSAGE = (
        f"{PARDISO_STALE_HANDLE}: the PARDISO factorization behind this handle no "
        "longer exists (its pattern slot was refactorized by another matrix, evicted, "
        "released or retired after an interrupted factorization); stale handles are "
        "never rebuilt - factorize the matrix again (FactorizationCache treats a "
        "stale handle as a miss)"
    )

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)


def _pardiso_perturbed_pivot_count(solver: Any) -> Tuple[Optional[int], str]:
    try:
        iparm = np.asarray(solver.iparm)
        if iparm.ndim != 1 or iparm.size < 14:
            return None, f"iparm has shape {iparm.shape}"
        count = int(iparm[13])
    except Exception as exc:  # fail closed: no readable count is no accepted factor
        return None, f"{type(exc).__name__}: {exc}"
    if count < 0:
        return None, f"negative count {count}"
    return count, ""


@_serialized_pypardiso_call
def _pardiso_numeric_factorize(
    solver: Any,
    prepared: sparse.csr_matrix,
    phase: int,
    *,
    mtype: Optional[int] = None,
    retire: Optional[Callable[[], None]] = None,
    owner: Optional["_MklSolverOwner"] = None,
) -> int:
    """Run one PARDISO numerical factorization phase; the only such native call.

    Every PARDISO numerical factorization (fresh phase 12, reused-pattern
    phase 22 and the phase 12 re-analysis retry) goes through this function,
    so ``iparm[13]`` is read for the factorization just performed and a
    perturbed or unreadable report raises :class:`PardisoPerturbedPivotError`
    worded for ``mtype``.  Callers must never retry that error into an
    accepted factorization.

    Interrupt safety: on a ``BaseException`` that is not an ``Exception``
    (``KeyboardInterrupt``, ``SystemExit``, ...) raised by the native call,
    ``retire`` (the owning slot's retirement, if any) runs first so that the
    slot can never be reused, then the pivot report is read, the solver's
    native memory is released (phase -1) and the original exception is
    re-raised unchanged (with a note recording the report).  A fresh
    factorization passes no ``retire``: its solver is released here, so no
    native memory leaks.  ``owner`` (the instance's :class:`_MklSolverOwner`)
    makes that release the single phase -1 of the instance.  Each cleanup
    step runs even if an earlier one raises; nested cleanup errors and a
    failed ``free_memory`` are attached as notes and never replace the
    original exception (the note says "released" only when phase -1
    returned normally).  The same cleanup covers an interrupt raised while
    the report is read after a completed native call.
    """

    solver.set_phase(int(phase))
    _count_numeric_operation("pardiso_factorization_attempts")
    try:
        try:
            solver._call_pardiso(prepared, np.zeros((prepared.shape[0], 1)))
        except Exception as exc:
            # A native error does not hide a perturbation report: a positive or
            # unreadable count after a raised call is a perturbation failure (no
            # retry, no mtype advance).  Only a verified zero count lets the
            # original error propagate, so that a caller may retry or advance.
            count, detail = _pardiso_perturbed_pivot_count(solver)
            if count is None or count > 0:
                raise PardisoPerturbedPivotError(
                    count,
                    phase=int(phase),
                    detail=f"{detail}; after native error {type(exc).__name__}: {exc}",
                    mtype=mtype,
                ) from exc
            raise
        count, detail = _pardiso_perturbed_pivot_count(solver)
    except Exception:
        raise
    except BaseException as exc:
        notes: List[str] = []
        if retire is not None:
            try:
                retire()
            except BaseException as nested:
                notes.append(f"retiring the slot raised {type(nested).__name__}: {nested}")
        try:
            count, detail = _pardiso_perturbed_pivot_count(solver)
        except BaseException as nested:
            count, detail = None, f"reading it raised {type(nested).__name__}: {nested}"
        if owner is not None:
            ran, error = owner.release()
        else:
            ran, error = True, _free_mkl_solver(solver)
        report = f"iparm[13]={count}" if count is not None else f"iparm[13] unreadable ({detail})"
        _add_note(
            exc,
            f"PARDISO phase {int(phase)} interrupted by {type(exc).__name__}: "
            f"{report}; {_release_note(error, ran)} and its pattern slot retired",
        )
        for note in notes:
            _add_note(exc, note)
        raise
    if count is None or count > 0:
        raise PardisoPerturbedPivotError(count, phase=int(phase), detail=detail, mtype=mtype)
    return count


@_serialized_pypardiso_call
def _pardiso_full_factorize(
    solver: Any,
    prepared: sparse.csr_matrix,
    *,
    mtype: Optional[int] = None,
    retire: Optional[Callable[[], None]] = None,
    owner: Optional["_MklSolverOwner"] = None,
) -> int:
    solver._check_A(prepared)
    return _pardiso_numeric_factorize(solver, prepared, 12, mtype=mtype, retire=retire, owner=owner)


class _PardisoPatternSlot:
    """One retained PARDISO instance per sparsity pattern for analysis reuse.

    ``generation`` changes whenever the slot's factorization is replaced or
    dropped; handles bound to an older generation are stale.  ``bound``
    holds the live handle wrappers of the current generation so that an
    invalidation reports their stale status immediately.

    ``leases`` counts the live :class:`FactorizationLease` objects pinning the
    slot's factorization.  A pinned slot is never refactorized; where an
    unpinned slot would be refactorized, evicted or released, a pinned slot
    is ``detached`` instead (removed from its backend's LRU, never matched
    again) and released when its last lease ends.  The native instance is
    owned by ``owner`` (:class:`_MklSolverOwner`).
    """

    __slots__ = (
        "owner", "mtype", "shape", "indptr", "indices", "generation", "bound",
        "leases", "detached", "detach_cause", "__weakref__",
    )

    def __init__(self, solver: Any, mtype: int, prepared: sparse.csr_matrix):
        self.owner: Optional[_MklSolverOwner] = (
            solver if isinstance(solver, _MklSolverOwner) else _MklSolverOwner(solver)
        )
        self.mtype = int(mtype)
        self.shape = tuple(int(v) for v in prepared.shape)
        self.indptr = prepared.indptr.copy()
        self.indices = prepared.indices.copy()
        self.generation = 0
        self.bound: "weakref.WeakSet[_PardisoFactorization]" = weakref.WeakSet()
        self.leases = 0
        self.detached = False
        self.detach_cause: Optional[str] = None

    @property
    def solver(self) -> Any:
        owner = self.owner
        return None if owner is None else owner.solver

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
    def invalidate(self, cause: str) -> None:
        """Make every handle bound to the current factorization stale."""

        self.generation += 1
        bound, self.bound = list(self.bound), weakref.WeakSet()
        for wrapper in bound:
            wrapper._mark_stale(cause)

    @_serialized_pypardiso_call
    def retire(self, cause: str) -> None:
        """Invalidate and drop the solver without a native call (its owner releases it)."""

        self.owner = None
        self.invalidate(cause)

    @_serialized_pypardiso_call
    def release(self, cause: str = "released") -> Optional[BaseException]:
        """Invalidate and free the instance once; return a ``free_memory`` error."""

        owner, self.owner = self.owner, None
        self.invalidate(cause)
        if owner is None:
            return None
        return owner.release()[1]


def _end_slot_lease(slot: _PardisoPatternSlot) -> Optional[BaseException]:
    """One lease of ``slot`` ended (explicit close or finalizer); never raises."""

    with _PYPARDISO_PROCESS_LOCK:
        slot.leases = max(int(slot.leases) - 1, 0)
        if slot.leases == 0 and slot.detached and slot.owner is not None:
            return slot.release(slot.detach_cause or "released")
    return None


class FactorizationLease:
    """Explicit, program-ordered pin of one factorization (context manager).

    While a lease is open, the PARDISO pattern slot behind ``handle`` is
    never refactorized, evicted or released: its handle stays valid and
    solves exactly as before.  A newcomer with the same pattern gets a new
    slot (counted in ``pardiso_lease_conflicts``); eviction and
    ``release_pattern_slots`` detach the pinned slot, which is freed when its
    last lease ends.  Closing is explicit (``close`` / ``with``); a lease
    dropped without closing is ended by its finalizer.  SuperLU handles and
    failed or refused handles are never pinned (``pinned`` is False): the
    lease then only carries the handle, whose status the caller checks as
    before.  ``operator`` is set by :func:`leased_inverse_operator`.
    """

    __slots__ = ("handle", "operator", "_finalizer", "__weakref__")

    def __init__(self, handle: Any, slot: Optional[_PardisoPatternSlot] = None) -> None:
        self.handle = handle
        self.operator: Optional[LinearOperator] = None
        self._finalizer: Optional[weakref.finalize] = None
        if slot is not None:
            finalizer = weakref.finalize(self, _end_slot_lease, slot)
            finalizer.atexit = False
            self._finalizer = finalizer

    @property
    def pinned(self) -> bool:
        return self._finalizer is not None and self._finalizer.alive

    def close(self) -> None:
        """End the lease (idempotent).  A detached slot is released here."""

        self._close(None)

    def _close(self, active: Optional[BaseException]) -> None:
        finalizer = self._finalizer
        if finalizer is None:
            return
        with _PYPARDISO_PROCESS_LOCK:
            if not finalizer.alive:
                return
            error = finalizer()
        if active is not None and error is not None and not isinstance(error, Exception):
            _add_note(active, "releasing a detached PARDISO slot at lease end raised "
                      f"{type(error).__name__}: {error}")
            return
        _raise_interrupt(error, "releasing a detached PARDISO slot at lease end")

    def __enter__(self) -> "FactorizationLease":
        return self

    def __exit__(self, _exc_type, exc, _traceback) -> None:
        self._close(exc)


class _PardisoFactorization:
    """Solve interface bound to the current factorization of a pattern slot.

    The slot's PARDISO instance holds the factorization of the *most recent*
    matrix with that sparsity pattern.  A handle created earlier checks the
    slot's generation token inside the process lock before solving; if the
    slot has moved on (refactorized, evicted, released or retired), the
    handle is stale and every solve raises :class:`PardisoStaleHandleError`
    - one type and message for sequential and concurrent callers, with no
    native call, no rebuild and no solution.  The owning handle reports
    ``status "failed"`` with ``pardiso_failure_kind "pardiso_stale_handle"``
    as soon as the slot is invalidated.
    """

    def __init__(self, slot: _PardisoPatternSlot, prepared: sparse.csr_matrix, mtype: int):
        self._slot = slot
        self._generation = slot.generation
        self._matrix = prepared
        self._mtype = int(mtype)
        self._handle_ref: Optional["weakref.ReferenceType[FactorizationHandle]"] = None
        # True when this factorization took a new slot because a leased slot
        # with the same pattern could not be refactorized.
        self.lease_conflict = False
        slot.bound.add(self)

    def is_stale(self) -> bool:
        slot = self._slot
        return slot.solver is None or slot.generation != self._generation

    def _mark_stale(self, cause: str) -> None:
        """Report the stale status on the owning handle (process lock held)."""

        handle = self._handle_ref() if self._handle_ref is not None else None
        if handle is None or handle.metadata.get("pardiso_failure_kind") == PARDISO_STALE_HANDLE:
            return
        handle.status = "failed"
        handle.failure_reason = PardisoStaleHandleError.MESSAGE
        handle.metadata["pardiso_failure_kind"] = PARDISO_STALE_HANDLE
        handle.metadata["pardiso_failure_stage"] = "stale_handle"
        handle.metadata["pardiso_stale_cause"] = str(cause)

    @_serialized_pypardiso_call
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        # Inside the process lock: a solve admitted before a concurrent
        # invalidation sees it here and raises the same typed error.
        if self.is_stale():
            self._mark_stale("stale")
            raise PardisoStaleHandleError()
        solver = self._slot.solver
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
    - MKL internal memory is bounded and released: evicted, released, failed
      and interrupted factorizations free their memory (phase -1).

    A handle is valid only while its slot holds its factorization.  After the
    slot is refactorized by another matrix, evicted, released or retired, the
    handle is stale and raises :class:`PardisoStaleHandleError`; it is never
    rebuilt.  A slot pinned by a :class:`FactorizationLease` is never
    refactorized, evicted or released while leased (see
    :meth:`_factorize_prepared`); ``pardiso_lease_conflicts`` counts the
    factorizations whose slot choice this changed.

    PyPardiso/MKL solver state is serialized process-wide.  This includes
    independent backend instances and handles owned by separate analysis
    caches; SciPy/SuperLU operations do not acquire this lock.
    """

    name = "pypardiso"

    def __init__(self, *, max_pattern_slots: int = 4):
        self.max_pattern_slots = _env_int("FE_SOLVER_PYPARDISO_MAX_PATTERN_SLOTS", int(max_pattern_slots))
        self._slots: List[_PardisoPatternSlot] = []
        self._detached: "weakref.WeakSet[_PardisoPatternSlot]" = weakref.WeakSet()

    def _detach_slot(self, slot: _PardisoPatternSlot, cause: str) -> None:
        """Take a leased slot out of the LRU; it is released at its last lease end."""

        if slot in self._slots:
            self._slots.remove(slot)
        slot.detached = True
        slot.detach_cause = str(cause)
        self._detached.add(slot)
        _count_numeric_operation("pardiso_lease_detachments")

    def _discard_slot(self, slot: _PardisoPatternSlot, cause: str) -> Optional[BaseException]:
        """Evict or release an LRU slot: leased slots are detached, others freed."""

        if slot.leases > 0:
            self._detach_slot(slot, cause)
            return None
        if slot in self._slots:
            self._slots.remove(slot)
        return slot.release(cause)

    @_serialized_pypardiso_call
    def release_pattern_slots(self) -> None:
        """Release all retained MKL factorization memory.

        Every handle bound to a released slot becomes stale
        (:class:`PardisoStaleHandleError` on solve).  A slot pinned by a
        :class:`FactorizationLease` is detached instead: its handle keeps
        solving and the slot is released when its last lease ends.
        """
        interrupt: Optional[BaseException] = None
        while self._slots:
            error = self._discard_slot(self._slots[-1], "released")
            if interrupt is None and error is not None and not isinstance(error, Exception):
                interrupt = error
        _raise_interrupt(interrupt, "releasing PARDISO pattern slots")

    @property
    @_serialized_pypardiso_call
    def leased_pattern_slots(self) -> int:
        """Number of live slots (in the LRU or detached) pinned by leases."""

        return sum(1 for slot in self._slots if slot.leases > 0) + self.detached_pattern_slots

    @property
    @_serialized_pypardiso_call
    def detached_pattern_slots(self) -> int:
        """Number of detached slots still alive for their leases."""

        return sum(1 for slot in list(self._detached) if slot.owner is not None)

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
    def _factorize_prepared(
        self, prepared: sparse.csr_matrix, mtype: int
    ) -> Tuple[_PardisoFactorization, bool, int]:
        """Factorize into a retained slot; return (wrapper, symbolic_reused, count).

        A perturbed-pivot rejection is final: it is never retried (neither by
        a phase 12 re-analysis nor by another mtype), and a slot whose
        numerical factorization failed is released and dropped.  An
        interrupted numerical phase (``BaseException``) retires the slot (it
        can never be reused) and releases its native memory; an interrupted
        fresh factorization releases its new solver.  The original exception
        is re-raised.

        Leases: a matching slot pinned by a lease is never refactorized.  It
        is detached (an unleased run would have refactorized it in place) and
        the newcomer takes a fresh analysis in a new slot at the front of the
        LRU, counted in ``pardiso_lease_conflicts``.  Eviction detaches a
        pinned victim instead of freeing it.  The LRU list therefore holds
        the same patterns in the same order as without leases.
        """

        lease_conflict = False
        for slot in self._slots:
            if slot.matches(prepared, mtype):
                if slot.leases > 0:
                    self._detach_slot(slot, "refactorized")
                    _count_numeric_operation("pardiso_lease_conflicts")
                    lease_conflict = True
                    break
                # Invalidate handles bound to the slot's previous values before
                # any numerical phase overwrites the slot's factorization.
                slot.invalidate("refactorized")

                def retire(slot: _PardisoPatternSlot = slot) -> None:
                    if slot in self._slots:
                        self._slots.remove(slot)
                    slot.retire("retired after an interrupted factorization")

                symbolic_reused = True
                owner = slot.owner
                solver = slot.solver
                try:
                    try:
                        count = _pardiso_numeric_factorize(
                            solver, prepared, 22, mtype=mtype, retire=retire, owner=owner
                        )
                    except PardisoPerturbedPivotError:
                        raise
                    except Exception:
                        # A PARDISO error other than a perturbation: re-analyse
                        # the current values (phase 12, checked again).
                        count = _pardiso_full_factorize(
                            solver, prepared, mtype=mtype, retire=retire, owner=owner
                        )
                        symbolic_reused = False
                except Exception:
                    if slot in self._slots:
                        self._slots.remove(slot)
                    _raise_interrupt(
                        slot.release("released after a failed factorization"),
                        "releasing a failed PARDISO slot",
                    )
                    raise
                except BaseException as exc:
                    # Interrupted outside the native call (e.g. the analysis
                    # check of the retry): retire and release exactly once.
                    if slot in self._slots:
                        self._slots.remove(slot)
                    ran, error = (False, None) if owner is None else owner.release()
                    slot.retire("retired after an interrupted factorization")
                    if ran:
                        _add_note(exc, f"PARDISO slot retired after {type(exc).__name__}; "
                                  f"{_release_note(error, ran)}")
                    raise
                self._slots.remove(slot)
                self._slots.insert(0, slot)
                return _PardisoFactorization(slot, prepared, mtype), symbolic_reused, count
        owner = _new_mkl_owner(int(mtype))
        try:
            count = _pardiso_full_factorize(owner.solver, prepared, mtype=mtype, owner=owner)
            slot = _PardisoPatternSlot(owner, mtype, prepared)
        except Exception:
            _raise_interrupt(owner.release()[1], "releasing a failed PARDISO instance")
            raise
        except BaseException as exc:
            ran, error = owner.release()
            if ran:
                _add_note(exc, f"PARDISO instance released after {type(exc).__name__}: "
                          f"{_release_note(error, ran)}")
            raise
        self._slots.insert(0, slot)
        wrapper = _PardisoFactorization(slot, prepared, mtype)
        wrapper.lease_conflict = lease_conflict
        interrupt: Optional[BaseException] = None
        while len(self._slots) > max(int(self.max_pattern_slots), 1):
            error = self._discard_slot(self._slots[-1], "evicted")
            if interrupt is None and error is not None and not isinstance(error, Exception):
                interrupt = error
        _raise_interrupt(interrupt, "evicting a PARDISO pattern slot")
        return wrapper, False, count

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
            # Snapshot the caller's arrays: the handle may refactorize its own
            # matrix later (stale-handle rebuild), and ``csr_matrix(csr)`` shares
            # ``data``/``indices``/``indptr`` with the input, so an in-place edit
            # by the caller would otherwise be factorized silently.
            csr = sparse.csr_matrix(matrix, copy=True)
            csr.sort_indices()
        except Exception as exc:
            failure_reasons.append(str(exc))
            csr = None
        wrapper: Optional[_PardisoFactorization] = None
        used_mtype: Optional[int] = None
        symbolic_reused = False
        # Perturbed-pivot count (iparm[13]) per attempted mtype; None =
        # unreadable, PARDISO_PIVOT_COUNT_UNSUPPORTED = not documented (mtype 2).
        perturbed_pivots: Dict[str, Any] = {}
        raw_iparm13: Dict[str, int] = {}
        if csr is not None:
            for mtype in _pardiso_mtype_candidates(matrix_class):
                try:
                    prepared = _pardiso_prepared_matrix(csr, mtype)
                    wrapper, symbolic_reused, count = self._factorize_prepared(prepared, mtype)
                    used_mtype = int(mtype)
                    perturbed_pivots[str(int(mtype))] = _reported_pivot_count(mtype, count)
                    break
                except PardisoPerturbedPivotError as exc:
                    # Perturbed pivots are evidence of a (near-)singular
                    # matrix, not of an unsuitable mtype: a later mtype can
                    # factorize the same matrix through a rounding-level pivot
                    # without reporting it.  Stop and fail this backend.
                    perturbed_pivots[str(int(mtype))] = _reported_pivot_count(mtype, exc.count)
                    if (
                        int(mtype) not in _PARDISO_PIVOT_COUNT_DOCUMENTED_MTYPES
                        and exc.count is not None
                    ):
                        # Refused fail closed, but not a documented count.
                        raw_iparm13[str(int(mtype))] = exc.count
                    # ``exc`` is already worded for ``mtype`` (no chained
                    # cause makes a count claim for mtype 2).
                    failure_reasons.append(f"mtype={mtype}: {exc}")
                    break
                except Exception as exc:
                    failure_reasons.append(f"mtype={mtype}: {exc}")
        if wrapper is None:
            # Only a perturbation rejection records a count on the failure path.
            rejected_perturbed = bool(perturbed_pivots)
            reason = "; ".join(failure_reasons) or "unknown pypardiso failure"
            failed = FactorizationHandle(
                matrix_shape=tuple(int(v) for v in matrix.shape),
                matrix_class=matrix_class,
                backend_name=self.name,
                ordering="MKL PARDISO",
                signature=signature,
                factorization_time=time.time() - start,
                status="failed",
                failure_reason=(
                    f"{PARDISO_PERTURBED_PIVOTS}: {reason}" if rejected_perturbed else reason
                ),
            )
            failed.metadata["pardiso_failure_kind"] = (
                PARDISO_PERTURBED_PIVOTS if rejected_perturbed else "pardiso_error"
            )
            failed.metadata["pardiso_perturbed_pivots"] = dict(perturbed_pivots)
            if raw_iparm13:
                failed.metadata["pardiso_iparm13_raw"] = dict(raw_iparm13)
            return failed
        handle = FactorizationHandle(
            matrix_shape=tuple(int(v) for v in matrix.shape),
            matrix_class=matrix_class,
            backend_name=self.name,
            ordering="MKL PARDISO",
            signature=signature,
            factorization_time=time.time() - start,
            _solver=wrapper,
        )
        wrapper._handle_ref = weakref.ref(handle)
        handle.metadata["pardiso_mtype"] = used_mtype
        handle.metadata["pardiso_symbolic_reused"] = bool(symbolic_reused)
        handle.metadata["pardiso_pattern_slots"] = len(self._slots)
        handle.metadata["pardiso_perturbed_pivots"] = dict(perturbed_pivots)
        if wrapper.lease_conflict:
            handle.metadata["pardiso_lease_conflict"] = True
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
            # A handle that later becomes stale raises PardisoStaleHandleError
            # (no rebuild, no solve-time fallback; see _PardisoFactorization).
            return handle
        failure_kind = handle.metadata.get("pardiso_failure_kind")
        if selection_policy == "pypardiso_forced" and failure_kind == PARDISO_PERTURBED_PIVOTS:
            # An explicitly requested PARDISO factorization whose pivots were
            # perturbed fails closed with the typed PARDISO failure; it is not
            # silently replaced by another backend's answer.
            return handle

        fallback = self.scipy_backend.factorize(matrix, matrix_class, signature=signature, options=options_dict)
        fallback.metadata["auto_backend_policy"] = "scipy_after_pypardiso_failure"
        fallback.metadata["pypardiso_min_dimension"] = self.pypardiso_min_dimension
        fallback.metadata["pypardiso_min_nnz"] = self.pypardiso_min_nnz
        fallback.metadata.update(selection_metadata)
        fallback.metadata["fallback_from_backend"] = handle.backend_name
        fallback.metadata["fallback_failure_reason"] = handle.failure_reason
        if failure_kind is not None:
            fallback.metadata["fallback_failure_kind"] = failure_kind
        if "pardiso_perturbed_pivots" in handle.metadata:
            fallback.metadata["fallback_pardiso_perturbed_pivots"] = dict(
                handle.metadata["pardiso_perturbed_pivots"]
            )
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
    Without an explicit ``signature`` the key is a content digest of the
    matrix, and every backend factorizes a private snapshot of its values, so
    a caller that edits a matrix in place after factorizing gets a fresh
    factorization for the edited values and the original one for the
    original values.  An explicit ``signature`` is a caller-owned revision
    stamp and is trusted: the caller must change it whenever the values change.
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
            cached = self._handles.get(cache_key)
            state = None if cached is None else _handle_state(cached)
            if state == "stale":
                # A stale PARDISO handle (its slot was refactorized, evicted
                # or released) is never rebuilt: evict it and factorize afresh
                # (a miss, not a failure).  Stale and failed are decided in
                # one process-lock section, so a concurrent invalidation is
                # never counted as a failure.
                self._handles.pop(cache_key, None)
            elif state == "failed":
                # Failed after insertion: evict it, count the failure and
                # factorize afresh (a miss).
                self._handles.pop(cache_key, None)
                self.factorization_failures += 1
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


def factorization_is_stale(handle: Any) -> bool:
    """Whether ``handle`` is a PARDISO handle whose factorization no longer exists.

    True after its pattern slot was refactorized by another matrix, evicted,
    released or retired; solving it raises :class:`PardisoStaleHandleError`.
    SuperLU and failed handles are never stale.
    """

    solver = getattr(handle, "_solver", None)
    return isinstance(solver, _PardisoFactorization) and solver.is_stale()


def _handle_state(handle: Any) -> str:
    """``"ok"``, ``"stale"`` or ``"failed"``, decided once under the process lock.

    The PARDISO wrapper's slot generation is the only stale authority; a
    PARDISO handle with a wrapper is failed only because it is stale.
    """

    solver = getattr(handle, "_solver", None)
    if isinstance(solver, _PardisoFactorization):
        with _PYPARDISO_PROCESS_LOCK:
            if solver.is_stale() or handle.metadata.get("pardiso_failure_kind") == PARDISO_STALE_HANDLE:
                solver._mark_stale("stale")
                return "stale"
            return "ok" if handle.status == "ok" else "failed"
    return "ok" if handle.status == "ok" and solver is not None else "failed"


def lease_handle(handle: Any) -> FactorizationLease:
    """Pin the factorization behind an existing handle (see :class:`FactorizationLease`).

    A current PARDISO handle is pinned in one process-lock section; a stale
    one raises :class:`PardisoStaleHandleError`.  SuperLU, failed and refused
    handles return an unpinned lease that only carries the handle.
    """

    solver = getattr(handle, "_solver", None)
    if not isinstance(solver, _PardisoFactorization):
        return FactorizationLease(handle)
    with _PYPARDISO_PROCESS_LOCK:
        state = _handle_state(handle)
        if state == "stale":
            raise PardisoStaleHandleError()
        if state != "ok":
            return FactorizationLease(handle)
        slot = solver._slot
        slot.leases += 1
        try:
            return FactorizationLease(handle, slot)
        except BaseException:
            slot.leases -= 1
            raise


def leased_factorization(
    matrix: sparse.spmatrix,
    matrix_class: MatrixClass | str,
    *,
    cache: Optional[FactorizationCache] = None,
    signature: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
    backend: Optional[SparseSolverBackend] = None,
) -> FactorizationLease:
    """Factorize like :func:`factorize_cached` and pin the result for a holder.

    For holders that solve one factorization across other work (callbacks,
    iterations, other factorizations): the returned lease keeps ``handle``
    valid until ``close`` (or the end of a ``with`` block).  If another
    thread invalidates the new handle before it is pinned, it is factorized
    once more through the same route (a stale cache entry is a miss); a
    second stale result raises :class:`PardisoStaleHandleError`.  A failed
    factorization is returned unpinned with its failed handle.
    """

    return lease_from(
        factorize_cached,
        matrix, matrix_class, cache=cache, signature=signature, options=options, backend=backend,
    )


def lease_from(factorize_function: Callable[..., Any], *args: Any, **kwargs: Any) -> FactorizationLease:
    """Call ``factorize_function(*args, **kwargs)`` and pin its handle.

    The holder's own factorization route (and any seam around it) is kept;
    a handle that went stale before it could be pinned is factorized once
    more through the same call, then a second stale result raises
    :class:`PardisoStaleHandleError`.
    """

    try:
        return lease_handle(factorize_function(*args, **kwargs))
    except PardisoStaleHandleError:
        pass
    return lease_handle(factorize_function(*args, **kwargs))


def leased_inverse_operator(
    matrix: sparse.spmatrix,
    matrix_class: MatrixClass | str,
    *,
    cache: Optional[FactorizationCache] = None,
    signature: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> FactorizationLease:
    """:func:`cached_inverse_operator` under a lease (``lease.operator``, ``lease.handle``)."""

    local_cache = cache or FactorizationCache(name="single_inverse_operator", max_entries=1)
    lease = leased_factorization(
        matrix, matrix_class, cache=local_cache, signature=signature, options=options
    )
    handle = lease.handle
    if handle.status != "ok":
        lease.close()
        raise RuntimeError(f"Cannot create inverse operator from failed factorization: {handle.failure_reason}")

    def matvec(rhs: np.ndarray) -> np.ndarray:
        return handle.solve(rhs)

    lease.operator = LinearOperator(matrix.shape, matvec=matvec, dtype=float)
    return lease


def refactorize_if_stale(
    handle: Any,
    matrix: sparse.spmatrix,
    matrix_class: MatrixClass | str,
    *,
    cache: Optional[FactorizationCache] = None,
    signature: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
    backend: Optional[SparseSolverBackend] = None,
) -> Any:
    """Return ``handle``, or a fresh factorization of ``matrix`` if it is stale.

    For holders that keep a handle across other factorizations: the fresh
    factorization goes through ``cache`` (where a stale entry is a miss) or
    the backend, exactly like the original ``factorize_cached`` call.
    """

    if handle is None or not factorization_is_stale(handle):
        return handle
    return factorize_cached(
        matrix, matrix_class, cache=cache, signature=signature, options=options, backend=backend
    )


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


def _require_solvable(handle: FactorizationHandle) -> None:
    """Raise the typed stale error first, then the failed-factorization error.

    Both are decided in one process-lock section (:func:`_handle_state`); a
    later concurrent invalidation is caught by the wrapper's own check inside
    the lock, with the same stale type.
    """

    state = _handle_state(handle)
    if state == "stale":
        raise PardisoStaleHandleError()
    if state != "ok":
        raise RuntimeError(f"Cannot solve with failed factorization: {handle.failure_reason}")


def solve(handle: FactorizationHandle, rhs: np.ndarray) -> np.ndarray:
    """Solve one right-hand side using an existing factorization."""
    _require_solvable(handle)
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
    _require_solvable(handle)
    rhs = np.asarray(rhs_matrix, dtype=float)
    if rhs.ndim == 1:
        return solve(handle, rhs)
    if rhs.ndim != 2:
        raise ValueError("rhs_matrix must be one- or two-dimensional")
    return solve(handle, rhs)
