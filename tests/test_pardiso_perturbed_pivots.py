"""PARDISO factorizations with perturbed pivots are failures, never solutions.

MKL PARDISO replaces (near-)zero pivots by a small perturbation instead of
failing (output ``iparm(14)``, 0-based ``iparm[13]``).  A perturbed
factorization of a singular matrix returns a huge solution whose normwise
backward error can be tiny, so the static solve used to report it as
converged/CERTIFIED.  These tests pin the fail-closed behaviour:

- the auto backend treats perturbed pivots as a PARDISO failure and falls
  back to SuperLU, which solves a nonsingular matrix or refuses a singular one;
- an explicitly forced PARDISO backend returns a typed failure instead;
- reused pattern slots (phase 22) are checked too;
- a stale handle (its slot was refactorized, evicted, released or retired) is
  never rebuilt: it raises ``PardisoStaleHandleError`` without a native call;
- every native numerical PARDISO call is interrupt-safe (release, retire);
- nonsingular, well-conditioned PARDISO-sized matrices are unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import anyfesolver.assembly as assembly
import anyfesolver.linalg as linalg
from anyfesolver.linalg import AutoSparseSolverBackend, MatrixClass, PyPardisoSolverBackend

_N = 10_000


def _singular_diagonal(n: int = _N) -> sparse.csr_matrix:
    diagonal = np.ones(n)
    diagonal[-1] = 0.0
    return sparse.diags(diagonal, 0, format="csr")


def _free_chain(n: int = _N) -> sparse.csr_matrix:
    """Free-floating spring chain: singular with a rigid-body mode."""

    return sparse.diags(
        [-np.ones(n - 1), np.r_[1.0, 2.0 * np.ones(n - 2), 1.0], -np.ones(n - 1)],
        [-1, 0, 1],
        format="csr",
    )


def _band_matrix(n: int = _N, half_bandwidth: int = 13, shift: float = 0.0) -> sparse.csr_matrix:
    """Band graph Laplacian (singular for shift 0) with >= 250k nonzeros."""

    offsets = [k for k in range(-half_bandwidth, half_bandwidth + 1) if k]
    adjacency = sparse.diags([np.ones(n - abs(k)) for k in offsets], offsets, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(degree + shift) - adjacency).tocsr()


# --------------------------------------------------------------------------
# Fake PARDISO: deterministic perturbed-pivot reports without MKL.
# --------------------------------------------------------------------------


class _PivotReportingFakeSolver:
    """Minimal PyPardisoSolver stand-in reporting ``iparm[13]`` per factorization."""

    def __init__(self, perturbed_for) -> None:
        self.iparm = np.zeros(64, dtype=np.int32)
        self.phase = 0
        self.phases: list[int] = []
        self.freed = False
        self._perturbed_for = perturbed_for

    def _check_A(self, matrix):
        return matrix

    def _check_b(self, matrix, rhs):
        del matrix
        return np.asarray(rhs, dtype=float)

    def set_phase(self, phase: int) -> None:
        self.phase = int(phase)

    def _call_pardiso(self, matrix, rhs):
        self.phases.append(self.phase)
        if self.phase in (12, 22):
            if getattr(self._perturbed_for, "wants_phase", False):
                count = self._perturbed_for(matrix, self.phase)
            else:
                count = self._perturbed_for(matrix)
            # Fault injection: ``(count, error)`` writes the report, then raises.
            error = None
            if isinstance(count, tuple):
                count, error = count
            if self.iparm is not None:
                self.iparm[13] = int(count)
            if error is not None:
                raise error
            return np.zeros_like(rhs)
        # A perturbed factor "solves" with an enormous answer.
        return np.asarray(rhs, dtype=float) * 1.0e16

    def free_memory(self, *, everything: bool) -> None:
        assert everything is True
        self.freed = True


def _install_fake(monkeypatch, perturbed_for):
    solvers: list[_PivotReportingFakeSolver] = []

    def factory(*, mtype: int):
        del mtype
        solver = _PivotReportingFakeSolver(perturbed_for)
        solvers.append(solver)
        return solver

    monkeypatch.setattr(linalg, "_HAS_PYPARDISO", True)
    monkeypatch.setattr(linalg, "_new_pypardiso_solver", factory)
    monkeypatch.setattr(linalg, "_ensure_mkl_rt_env", lambda: None)
    return solvers


def _by_phase(function):
    """Mark a fake pivot report as depending on ``(matrix, phase)``."""

    function.wants_phase = True
    return function


def _zero_diagonal_count(matrix) -> int:
    return int(np.count_nonzero(sparse.csr_matrix(matrix).diagonal() == 0.0))


def test_fake_auto_backend_falls_back_to_superlu_on_perturbed_pivots(monkeypatch) -> None:
    _install_fake(monkeypatch, _zero_diagonal_count)
    backend = AutoSparseSolverBackend(pypardiso_min_dimension=0, pypardiso_min_nnz=0)

    singular = sparse.csr_matrix(np.diag([1.0, 2.0, 0.0]))
    handle = backend.factorize(singular, MatrixClass.SYMMETRIC_INDEFINITE)
    assert handle.backend_name == "scipy_superlu"
    assert handle.status == "failed"
    assert "singular" in str(handle.failure_reason).lower()
    assert handle.metadata["auto_backend_policy"] == "scipy_after_pypardiso_failure"
    assert handle.metadata["fallback_from_backend"] == "pypardiso"
    assert "perturbed_pivots" in str(handle.metadata["fallback_failure_reason"])
    assert handle.metadata["fallback_failure_kind"] == "perturbed_pivots"

    # A nonsingular matrix on which PARDISO (fake) perturbs is solved exactly by SuperLU.
    def always_perturbed(matrix) -> int:
        del matrix
        return 1

    _install_fake(monkeypatch, always_perturbed)
    backend = AutoSparseSolverBackend(pypardiso_min_dimension=0, pypardiso_min_nnz=0)
    regular = sparse.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    handle = backend.factorize(regular, MatrixClass.SPD)
    assert handle.backend_name == "scipy_superlu"
    assert handle.status == "ok"
    assert handle.metadata["fallback_failure_kind"] == "perturbed_pivots"
    np.testing.assert_allclose(regular @ handle.solve(np.array([1.0, 2.0])), [1.0, 2.0])


def test_fake_forced_pardiso_fails_closed_with_typed_status(monkeypatch) -> None:
    _install_fake(monkeypatch, _zero_diagonal_count)
    singular = sparse.csr_matrix(np.diag([1.0, 2.0, 0.0]))

    forced = AutoSparseSolverBackend().factorize(
        singular, MatrixClass.SYMMETRIC_INDEFINITE, options={"backend": "pypardiso"}
    )
    assert forced.backend_name == "pypardiso"
    assert forced.status == "failed"
    assert str(forced.failure_reason).startswith("perturbed_pivots")
    assert forced.metadata["pardiso_failure_kind"] == "perturbed_pivots"
    assert forced.metadata["pardiso_perturbed_pivots"] == {"-2": 1}
    assert forced.metadata["auto_backend_policy"] == "pypardiso_forced"
    with pytest.raises(RuntimeError, match="perturbed_pivots"):
        forced.solve(np.ones(3))

    direct = PyPardisoSolverBackend().factorize(singular, MatrixClass.SYMMETRIC_INDEFINITE)
    assert direct.status == "failed"
    assert direct.metadata["pardiso_failure_kind"] == "perturbed_pivots"


def test_fake_reused_slot_refactorization_is_checked(monkeypatch) -> None:
    solvers = _install_fake(monkeypatch, _zero_diagonal_count)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    regular = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    # Same stored pattern as ``regular`` with an explicit zero pivot.
    singular = sparse.csr_matrix(
        (np.array([1.0, 2.0, 0.0]), regular.indices.copy(), regular.indptr.copy()), shape=(3, 3)
    )

    first = backend.factorize(regular, MatrixClass.SPD)
    assert first.status == "ok"
    # iparm[13] is not a documented output for mtype 2: never a measured 0.
    assert first.metadata["pardiso_perturbed_pivots"] == {"2": "unsupported"}

    reused = backend.factorize(singular, MatrixClass.SPD)
    assert reused.status == "failed"
    assert reused.metadata["pardiso_failure_kind"] == "perturbed_pivots"
    # The retained slot was tried with phase 22 only (no re-analysis retry)
    # and released after the rejection.
    assert solvers[0].phases == [12, 22]
    assert solvers[0].freed is True
    assert backend.retained_pattern_slots == 0

    again = backend.factorize(regular * 2.0, MatrixClass.SPD)
    assert again.status == "ok"
    assert again.metadata["pardiso_symbolic_reused"] is False
    assert again.metadata["pardiso_perturbed_pivots"] == {"2": "unsupported"}


def _warm_auto_backend(max_pattern_slots: int = 2) -> AutoSparseSolverBackend:
    return AutoSparseSolverBackend(
        pardiso_backend=PyPardisoSolverBackend(max_pattern_slots=max_pattern_slots),
        pypardiso_min_dimension=0,
        pypardiso_min_nnz=0,
        pypardiso_warm_min_dimension=0,
        pypardiso_warm_min_nnz=0,
    )


@pytest.mark.parametrize("matrix_class", [MatrixClass.GENERAL, MatrixClass.SYMMETRIC_INDEFINITE])
def test_fake_perturbed_phase22_is_not_retried_into_acceptance(monkeypatch, matrix_class) -> None:
    """Review R2 P1: a perturbed phase 22 is a failure; no phase 12 retry may hide it."""

    state = {"armed": False}

    @_by_phase
    def phase22_only(matrix, phase) -> int:
        del matrix
        return 1 if (state["armed"] and phase == 22) else 0

    solvers = _install_fake(monkeypatch, phase22_only)
    mtype = "11" if matrix_class == MatrixClass.GENERAL else "-2"
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))

    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    first = backend.factorize(A, matrix_class)
    assert first.status == "ok"
    state["armed"] = True
    second = backend.factorize((A * 2.0).tocsr(), matrix_class)
    assert second.status == "failed"
    assert second.metadata["pardiso_failure_kind"] == "perturbed_pivots"
    assert second.metadata["pardiso_perturbed_pivots"] == {mtype: 1}
    assert "pardiso_symbolic_reused" not in second.metadata
    assert solvers[0].phases == [12, 22]
    # P3: the failed warm solver's native factors are released.
    assert solvers[0].freed is True
    assert backend.retained_pattern_slots == 0
    # Option B: the handle bound to the released slot is stale and is never
    # rebuilt (previously it rebuilt its own matrix privately).
    state["armed"] = False
    with pytest.raises(linalg.PardisoStaleHandleError):
        first.solve(np.ones(3))
    assert first.status == "failed"
    assert solvers[0].phases == [12, 22] and len(solvers) == 1

    auto = _warm_auto_backend()
    assert auto.factorize(A, matrix_class).backend_name == "pypardiso"
    state["armed"] = True
    fallback = auto.factorize((A * 2.0).tocsr(), matrix_class)
    assert fallback.backend_name == "scipy_superlu"
    assert fallback.status == "ok"
    assert fallback.metadata["auto_backend_policy"] == "scipy_after_pypardiso_failure"
    assert fallback.metadata["fallback_failure_kind"] == "perturbed_pivots"
    np.testing.assert_allclose((A * 2.0) @ fallback.solve(np.ones(3)), np.ones(3))

    state["armed"] = False
    forced_backend = _warm_auto_backend()
    options = {"backend": "pypardiso"}
    assert forced_backend.factorize(A, matrix_class, options=options).status == "ok"
    state["armed"] = True
    forced = forced_backend.factorize((A * 2.0).tocsr(), matrix_class, options=options)
    assert forced.backend_name == "pypardiso"
    assert forced.status == "failed"
    assert forced.metadata["pardiso_failure_kind"] == "perturbed_pivots"


def test_fake_phase22_error_retry_is_reported_and_failed_retry_is_released(monkeypatch) -> None:
    """A non-perturbation phase 22 error may be re-analysed (phase 12), truthfully."""

    state = {"fail22": False, "fail12": False}

    @_by_phase
    def failing(matrix, phase) -> int:
        del matrix
        if (phase == 22 and state["fail22"]) or (phase == 12 and state["fail12"]):
            raise RuntimeError("simulated PARDISO error -3")
        return 0

    solvers = _install_fake(monkeypatch, failing)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    assert backend.factorize(A, MatrixClass.GENERAL).status == "ok"

    state["fail22"] = True
    retried = backend.factorize((A * 2.0).tocsr(), MatrixClass.GENERAL)
    assert retried.status == "ok"
    assert solvers[0].phases == [12, 22, 12]
    # Phase 12 redid the analysis: the symbolic analysis was not reused.
    assert retried.metadata["pardiso_symbolic_reused"] is False
    assert retried.metadata["pardiso_perturbed_pivots"] == {"11": 0}

    state["fail12"] = True
    failed = backend.factorize((A * 3.0).tocsr(), MatrixClass.GENERAL)
    assert failed.status == "failed"
    assert solvers[0].freed is True
    assert backend.retained_pattern_slots == 0


# --------------------------------------------------------------------------
# Codex review round 2 findings, fixed in round 3 (fault injection).
# --------------------------------------------------------------------------


def _stale_forced_handle(monkeypatch, state, *, matrix_class=MatrixClass.SYMMETRIC_INDEFINITE):
    """A direct-PARDISO handle made stale by a second factorization of its pattern."""

    def controlled(matrix):
        del matrix
        return state["report"]

    solvers = _install_fake(monkeypatch, controlled)
    backend = PyPardisoSolverBackend(max_pattern_slots=1)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    first = backend.factorize(A, matrix_class)
    assert first.status == "ok"
    assert backend.factorize((A * 2.0).tocsr(), matrix_class).status == "ok"
    return backend, first, solvers


@pytest.mark.parametrize("phase", [12, 22])
def test_r2_native_error_does_not_hide_a_positive_pivot_report(monkeypatch, phase) -> None:
    """Codex r2 P1 #2: a call that reports a perturbation and then raises is refused."""

    state = {"armed": False}

    @_by_phase
    def report_then_raise(matrix, call_phase):
        del matrix
        if state["armed"] and call_phase == phase:
            return 1, RuntimeError("simulated native PARDISO error")
        return 0

    solvers = _install_fake(monkeypatch, report_then_raise)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    if phase == 22:
        assert backend.factorize(A, MatrixClass.GENERAL).status == "ok"
    state["armed"] = True
    handle = backend.factorize((A * 2.0).tocsr(), MatrixClass.GENERAL)
    assert handle.status == "failed"
    assert handle.metadata["pardiso_failure_kind"] == "perturbed_pivots"
    assert handle.metadata["pardiso_perturbed_pivots"] == {"11": 1}
    # No phase 12 retry after the perturbed (raised) phase 22.
    assert solvers[0].phases == ([12, 22] if phase == 22 else [12])


def test_r2_native_error_with_perturbation_stops_the_mtype_chain(monkeypatch) -> None:
    @_by_phase
    def report_then_raise(matrix, phase):
        del matrix, phase
        return 1, RuntimeError("simulated native PARDISO error")

    solvers = _install_fake(monkeypatch, report_then_raise)
    handle = PyPardisoSolverBackend().factorize(
        sparse.csr_matrix(np.diag([1.0, 2.0, 3.0])), MatrixClass.SYMMETRIC_INDEFINITE
    )
    assert handle.status == "failed"
    assert handle.metadata["pardiso_failure_kind"] == "perturbed_pivots"
    assert handle.metadata["pardiso_perturbed_pivots"] == {"-2": 1}
    assert len(solvers) == 1  # mtype 11 was never tried


def test_r2_non_perturbation_native_error_retry_is_checked(monkeypatch) -> None:
    """Retry only after a verified zero count; the retry's own report is checked."""

    state = {"armed": False}

    @_by_phase
    def error22_then_perturbed12(matrix, phase):
        del matrix
        if not state["armed"]:
            return 0
        if phase == 22:
            return 0, RuntimeError("simulated PARDISO error -3")
        return 1

    solvers = _install_fake(monkeypatch, error22_then_perturbed12)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    assert backend.factorize(A, MatrixClass.GENERAL).status == "ok"
    state["armed"] = True
    handle = backend.factorize((A * 2.0).tocsr(), MatrixClass.GENERAL)
    assert solvers[0].phases == [12, 22, 12]
    assert handle.status == "failed"
    assert handle.metadata["pardiso_failure_kind"] == "perturbed_pivots"


def test_r2_mtype2_rejection_does_not_publish_a_measured_count(monkeypatch) -> None:
    """Codex r2 P3: for mtype 2 a positive raw iparm[13] is refused but is not a count."""

    _install_fake(monkeypatch, lambda matrix: 1)
    handle = PyPardisoSolverBackend().factorize(
        sparse.csr_matrix(np.diag([1.0, 2.0, 3.0])), MatrixClass.SPD
    )
    assert handle.status == "failed"
    assert handle.metadata["pardiso_failure_kind"] == "perturbed_pivots"
    assert handle.metadata["pardiso_perturbed_pivots"] == {"2": "unsupported"}
    assert handle.metadata["pardiso_iparm13_raw"] == {"2": 1}


def test_r4_mtype2_failure_text_does_not_claim_perturbed_pivots(monkeypatch) -> None:
    """Codex r3 P3: a raw mtype 2 iparm[13] is not described as a pivot count."""

    _install_fake(monkeypatch, lambda matrix: 1)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    handle = PyPardisoSolverBackend().factorize(A, MatrixClass.SPD)
    assert handle.status == "failed"
    reason = str(handle.failure_reason)
    assert "pivot(s)" not in reason
    assert "raw iparm[13]=1" in reason
    assert "not a documented perturbed-pivot count for mtype 2" in reason
    assert "fail closed" in reason
    # Documented mtypes keep the count wording.
    indefinite = PyPardisoSolverBackend().factorize(A, MatrixClass.SYMMETRIC_INDEFINITE)
    assert "perturbed 1 pivot(s)" in str(indefinite.failure_reason)


def test_r4_open_certification_xfails_match_only_the_intended_failure() -> None:
    """Codex r3 P3: an unrelated exception must FAIL the strict xfails, not XFAIL them."""

    for name in (
        "test_open_static_certification_of_rounding_singular_band",
        "test_open_spd_cholesky_accepts_rounding_singular_chain",
    ):
        function = globals()[name]
        marks = [m for m in getattr(function, "pytestmark", []) if m.name == "xfail"]
        assert len(marks) == 1, name
        assert marks[0].kwargs.get("strict") is True, name
        assert marks[0].kwargs.get("raises") is AssertionError, name


# --------------------------------------------------------------------------
# Option B (owner decision 2026-11-25, Codex round 4): a stale PARDISO handle
# is never rebuilt; every native numerical PARDISO call is interrupt-safe.
# Each test below fails on d89a7a82b.
# --------------------------------------------------------------------------


def _stale_by(monkeypatch, cause, state):
    """A direct-PARDISO handle made stale by ``cause`` (slot refactorized/evicted/released)."""

    solvers = _install_fake(monkeypatch, lambda matrix: state["report"])
    backend = PyPardisoSolverBackend(max_pattern_slots=1)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    first = backend.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE)
    assert first.status == "ok" and not linalg.factorization_is_stale(first)
    if cause == "refactorized":
        assert backend.factorize((A * 2.0).tocsr(), MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
    elif cause == "evicted":
        assert backend.factorize(sparse.eye(4, format="csr"), MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
    else:
        backend.release_pattern_slots()
    return backend, first, solvers


@pytest.mark.parametrize("report", [0, 1])
@pytest.mark.parametrize("cause", ["refactorized", "evicted", "released"])
def test_b_stale_handle_raises_one_typed_error_without_a_native_call(monkeypatch, cause, report) -> None:
    state = {"report": 0}
    _backend, first, solvers = _stale_by(monkeypatch, cause, state)
    state["report"] = report  # any rebuild would report this; none may run
    built, phases = len(solvers), [list(s.phases) for s in solvers]
    # Truthful status as soon as the slot moved on, before any solve.
    assert linalg.factorization_is_stale(first)
    assert first.status == "failed"
    assert first.failure_reason == linalg.PardisoStaleHandleError.MESSAGE
    assert first.metadata["pardiso_failure_kind"] == linalg.PARDISO_STALE_HANDLE
    assert first.metadata["pardiso_failure_stage"] == "stale_handle"
    assert first.metadata["pardiso_stale_cause"] == cause
    rhs = np.ones(3)
    for call in (first.solve, first.solve_many, lambda b: linalg.solve(first, b),
                 lambda b: linalg.solve_many(first, b[:, None]), first._solver.solve):
        with pytest.raises(linalg.PardisoStaleHandleError) as raised:
            call(rhs)
        assert type(raised.value) is linalg.PardisoStaleHandleError
        assert str(raised.value) == linalg.PardisoStaleHandleError.MESSAGE
    assert isinstance(raised.value, RuntimeError)
    assert len(solvers) == built and [s.phases for s in solvers] == phases  # no native call
    assert linalg.backend_operation_counts()["pardiso_stale_rebuilds"] == 0


def test_b_concurrent_stale_solves_raise_the_same_type(monkeypatch) -> None:
    """Codex r4 P2: a solve admitted before the invalidation raises the same type."""

    import threading
    import time

    state = {"report": 0}
    solvers = _install_fake(monkeypatch, lambda matrix: state["report"])
    backend = PyPardisoSolverBackend(max_pattern_slots=1)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    first = backend.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE)
    outcomes: list[object] = []

    def solve(call) -> None:
        try:
            call(np.ones(3))
            outcomes.append("result")
        except BaseException as exc:  # noqa: BLE001 - the exact type is the assertion
            outcomes.append(type(exc))

    # The admitted caller passes the handle checks while the handle is still
    # current, then waits for the process lock held by the invalidating caller.
    with linalg._PYPARDISO_PROCESS_LOCK:
        admitted = threading.Thread(target=solve, args=(first.solve,))
        admitted.start()
        time.sleep(0.2)
        assert backend.factorize((A * 2.0).tocsr(), MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
        state["report"] = 1  # a perturbed rebuild would now raise a different type
    admitted.join(timeout=5.0)
    barrier = threading.Barrier(6)
    threads = [threading.Thread(target=lambda c=c: (barrier.wait(), solve(c)))
               for c in (first.solve, first._solver.solve, first.solve_many) * 2]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert outcomes == [linalg.PardisoStaleHandleError] * 7
    assert len(solvers) == 1 and 33 not in solvers[0].phases[1:]


def test_b_interrupted_cleanup_cannot_revive_a_stale_handle(monkeypatch) -> None:
    """Codex r4 P1: a phase -1 interrupt used to bypass the permanent rejection."""

    state = {"report": 0}
    _backend, first, solvers = _stale_by(monkeypatch, "refactorized", state)
    real_factory = linalg._new_pypardiso_solver

    def factory(*, mtype: int):
        solver = real_factory(mtype=mtype)

        def interrupted_free(*, everything: bool) -> None:
            raise KeyboardInterrupt

        solver.free_memory = interrupted_free
        return solver

    monkeypatch.setattr(linalg, "_new_pypardiso_solver", factory)
    built = len(solvers)
    for report in (1, 0):  # perturbed, then a clean report a rebuild would accept
        state["report"] = report
        with pytest.raises(linalg.PardisoStaleHandleError):
            first.solve(np.ones(3))
    assert len(solvers) == built
    assert all(33 not in solver.phases for solver in solvers)


@pytest.mark.parametrize("report", [0, 1])
def test_b_interrupted_warm_factorization_retires_its_slot(monkeypatch, report) -> None:
    """Codex r4 P2: an interrupted phase 22 left the slot eligible for reuse."""

    interrupt = KeyboardInterrupt()
    state = {"armed": False}

    @_by_phase
    def interrupted22(matrix, phase):
        del matrix
        return (report, interrupt) if (state["armed"] and phase == 22) else 0

    solvers = _install_fake(monkeypatch, interrupted22)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    first = backend.factorize(A, MatrixClass.GENERAL)
    state["armed"] = True
    with pytest.raises(KeyboardInterrupt) as raised:
        backend.factorize((A * 2.0).tocsr(), MatrixClass.GENERAL)
    assert raised.value is interrupt  # the original exception, re-raised
    assert any(f"iparm[13]={report}" in note for note in getattr(raised.value, "__notes__", []))
    assert solvers[0].phases == [12, 22]
    assert solvers[0].freed is True  # phase -1 ran
    assert backend.retained_pattern_slots == 0  # the slot can never be reused
    with pytest.raises(linalg.PardisoStaleHandleError):
        first.solve(np.ones(3))
    state["armed"] = False
    third = backend.factorize((A * 3.0).tocsr(), MatrixClass.GENERAL)
    assert third.status == "ok"
    assert third.metadata["pardiso_symbolic_reused"] is False
    assert len(solvers) == 2 and solvers[1].phases == [12] and solvers[0].phases == [12, 22]


def test_b_interrupted_phase12_retry_retires_its_slot(monkeypatch) -> None:
    state = {"armed": False}

    @_by_phase
    def error22_then_interrupt12(matrix, phase):
        del matrix
        if not state["armed"]:
            return 0
        return (0, RuntimeError("simulated PARDISO error -3")) if phase == 22 else (0, KeyboardInterrupt())

    solvers = _install_fake(monkeypatch, error22_then_interrupt12)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    assert backend.factorize(A, MatrixClass.GENERAL).status == "ok"
    state["armed"] = True
    with pytest.raises(KeyboardInterrupt):
        backend.factorize((A * 2.0).tocsr(), MatrixClass.GENERAL)
    assert solvers[0].phases == [12, 22, 12]
    assert solvers[0].freed is True
    assert backend.retained_pattern_slots == 0


@pytest.mark.parametrize("report", [0, 1])
def test_b_interrupted_fresh_factorization_releases_its_solver(monkeypatch, report) -> None:
    """Codex r4 P2: an interrupted fresh phase 12 leaked its native memory."""

    @_by_phase
    def interrupted12(matrix, phase):
        del matrix, phase
        return report, SystemExit(3)

    solvers = _install_fake(monkeypatch, interrupted12)
    backend = PyPardisoSolverBackend()
    with pytest.raises(SystemExit):
        backend.factorize(sparse.csr_matrix(np.diag([1.0, 2.0, 3.0])), MatrixClass.SYMMETRIC_INDEFINITE)
    assert len(solvers) == 1 and solvers[0].phases == [12]
    assert solvers[0].freed is True
    assert backend.retained_pattern_slots == 0


def test_b_cache_treats_a_stale_handle_as_a_miss(monkeypatch) -> None:
    state = {"report": 0}
    solvers = _install_fake(monkeypatch, lambda matrix: state["report"])
    backend = PyPardisoSolverBackend(max_pattern_slots=1)
    cache = linalg.FactorizationCache(name="b_cache", backend=backend)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    first = cache.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE)
    assert cache.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE) is first  # ordinary hit
    assert backend.factorize((A * 2.0).tocsr(), MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
    again = cache.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE)
    assert again is not first and again.status == "ok"
    assert again.metadata["cache_hit"] is False
    np.testing.assert_array_equal(again.solve(np.ones(3)), np.ones(3) * 1.0e16)
    diagnostics = cache.diagnostics()
    assert (diagnostics["hits"], diagnostics["misses"], diagnostics["factorization_failures"]) == (1, 2, 0)
    with pytest.raises(linalg.PardisoStaleHandleError):
        first.solve(np.ones(3))
    # refactorize_if_stale (holders of a handle) uses the same route.
    assert linalg.refactorize_if_stale(again, A, MatrixClass.SYMMETRIC_INDEFINITE, cache=cache) is again
    assert backend.factorize((A * 3.0).tocsr(), MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
    fresh = linalg.refactorize_if_stale(again, A, MatrixClass.SYMMETRIC_INDEFINITE, cache=cache)
    assert fresh is not again and fresh.status == "ok" and not linalg.factorization_is_stale(fresh)
    assert len(solvers) == 1  # every refactorization reused the one slot (phase 22)


def test_b_mtype2_chained_cause_makes_no_count_claim(monkeypatch) -> None:
    """Codex r4 P3: the chained cause of an mtype 2 rejection said 'perturbed 1 pivot(s)'."""

    _install_fake(monkeypatch, lambda matrix: 1)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    prepared = linalg._pardiso_prepared_matrix(A, 2)
    with pytest.raises(linalg.PardisoPerturbedPivotError) as raised:
        PyPardisoSolverBackend()._factorize_prepared(prepared, 2)
    error: BaseException | None = raised.value
    assert "not a documented perturbed-pivot count for mtype 2" in str(error)
    while error is not None:
        assert "pivot(s)" not in str(error)
        error = error.__cause__ or error.__context__


def test_b_strict_xfails_fail_on_an_unrelated_assertion_inside_the_call(monkeypatch) -> None:
    """Codex r4 P3: raises=AssertionError let an unrelated AssertionError XFAIL."""

    def unrelated(*args, **kwargs):
        raise AssertionError("unrelated internal assertion")

    monkeypatch.setattr(assembly, "_solve_reduced_system", unrelated)
    monkeypatch.setattr(PyPardisoSolverBackend, "factorize", unrelated)
    for call in (
        test_minres_inconsistent_singular_system_is_not_converged,
        lambda: test_open_static_certification_of_rounding_singular_band(200),
        test_open_spd_cholesky_accepts_rounding_singular_chain,
    ):
        with pytest.raises(pytest.fail.Exception, match="unrelated internal assertion"):
            call()


def test_fake_pivot_count_metadata_per_mtype(monkeypatch) -> None:
    """Review R2 P3: iparm[13] is documented for mtypes -2 and 11 only."""

    _install_fake(monkeypatch, lambda matrix: 0)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    backend = PyPardisoSolverBackend()
    assert backend.factorize(A, MatrixClass.SPD).metadata["pardiso_perturbed_pivots"] == {"2": "unsupported"}
    assert backend.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE).metadata["pardiso_perturbed_pivots"] == {"-2": 0}
    assert backend.factorize(A, MatrixClass.GENERAL).metadata["pardiso_perturbed_pivots"] == {"11": 0}


def test_unreadable_pivot_report_fails_closed(monkeypatch) -> None:
    solvers = _install_fake(monkeypatch, _zero_diagonal_count)

    real_factory = linalg._new_pypardiso_solver

    def factory(*, mtype: int):
        solver = real_factory(mtype=mtype)
        solver.iparm = None
        return solver

    monkeypatch.setattr(linalg, "_new_pypardiso_solver", factory)
    handle = PyPardisoSolverBackend().factorize(
        sparse.csr_matrix(np.diag([1.0, 2.0, 3.0])), MatrixClass.SPD
    )
    assert handle.status == "failed"
    assert "perturbed_pivots" in str(handle.failure_reason)
    assert solvers


# --------------------------------------------------------------------------
# Real MKL PARDISO.
# --------------------------------------------------------------------------

requires_pypardiso = pytest.mark.skipif(not linalg._HAS_PYPARDISO, reason="pypardiso not installed")


@requires_pypardiso
@pytest.mark.parametrize(
    ("name", "matrix_class"),
    [
        ("diagonal", MatrixClass.SYMMETRIC_INDEFINITE),
        ("diagonal", MatrixClass.SPD),
        ("chain", MatrixClass.SYMMETRIC_INDEFINITE),
        ("chain", MatrixClass.SYMMETRIC_SEMIDEFINITE),
        ("chain", MatrixClass.GENERAL),
    ],
)
def test_real_auto_backend_refuses_singular_matrices_via_superlu(name, matrix_class) -> None:
    matrix = _singular_diagonal() if name == "diagonal" else _free_chain()
    backend = AutoSparseSolverBackend(pypardiso_min_nnz=0)
    handle = backend.factorize(matrix, matrix_class)
    assert handle.backend_name == "scipy_superlu"
    assert handle.status == "failed"
    assert "singular" in str(handle.failure_reason).lower()
    assert handle.metadata["auto_backend_policy"] == "scipy_after_pypardiso_failure"
    assert handle.metadata["fallback_failure_kind"] == "perturbed_pivots"


@requires_pypardiso
def test_real_forced_pardiso_refuses_perturbed_singular_factorization() -> None:
    handle = AutoSparseSolverBackend().factorize(
        _singular_diagonal(), MatrixClass.SYMMETRIC_INDEFINITE, options={"backend": "pypardiso"}
    )
    assert handle.backend_name == "pypardiso"
    assert handle.status == "failed"
    assert handle.metadata["pardiso_failure_kind"] == "perturbed_pivots"
    assert handle.metadata["pardiso_perturbed_pivots"]["-2"] >= 1


@requires_pypardiso
def test_real_static_solve_refuses_singular_diagonal(monkeypatch) -> None:
    """End-to-end: the reduced direct static solve must not certify a singular system."""

    monkeypatch.setattr(linalg, "DEFAULT_BACKEND", AutoSparseSolverBackend(pypardiso_min_nnz=0))
    matrix = _singular_diagonal()
    q, info = assembly._solve_reduced_system(matrix, np.ones(_N), "direct")
    assert info["status"] == "failed"
    assert info.get("relative_backward_error_disposition") != "CERTIFIED"
    assert "singular" in str(info["error"]).lower()
    assert info["backend"]["fallback_failure_kind"] == "perturbed_pivots"
    assert np.all(np.isfinite(q))


@requires_pypardiso
def test_real_large_singular_band_rejects_pardiso_and_does_not_try_weaker_mtypes() -> None:
    """A >= 250k-nnz singular matrix reaches PARDISO under the default thresholds.

    mtype -2 perturbs one pivot.  mtype 11 would factorize the same matrix
    through a rounding-level pivot *without* reporting a perturbation, so the
    chain must stop at the first perturbed attempt.
    """

    matrix = _band_matrix()
    assert matrix.nnz >= 250_000
    direct = PyPardisoSolverBackend().factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE)
    assert direct.status == "failed"
    assert direct.metadata["pardiso_perturbed_pivots"] == {"-2": 1}
    handle = AutoSparseSolverBackend().factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE)
    assert handle.backend_name == "scipy_superlu"
    assert handle.metadata["fallback_from_backend"] == "pypardiso"
    assert handle.metadata["fallback_failure_kind"] == "perturbed_pivots"


# Open owner decisions (pre-existing certification holes, not perturbed pivots).
# Strict xfails: they turn into failures as soon as the behaviour changes.  All
# of them belong to the separate static-solve certification slice.

def _call_under_test(function, *args):
    """Run the call whose outcome a strict xfail asserts on.

    The xfails use ``raises=AssertionError`` for their final outcome
    assertion only: an AssertionError raised *inside* the call is unrelated
    and fails the test instead of being counted as the expected failure.
    """

    try:
        return function(*args)
    except AssertionError as exc:
        pytest.fail(f"unrelated AssertionError inside the call under test: {exc}")


_CERTIFICATION_SLICE = (
    " Deferred to the static linear-solve certification slice (owner decision; "
    "outside the perturbed-pivot fix)."
)


@requires_pypardiso
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "OPEN: a rounding-singular matrix (band graph Laplacian, constant "
        "nullspace, inconsistent load) is factorized by SuperLU through an "
        "O(1e-13) pivot; the normwise backward error (~4e-16) then certifies "
        "a ~1e16 solution. Same on main below the PARDISO threshold (n=2000)."
        + _CERTIFICATION_SLICE
    ),
)
@pytest.mark.parametrize("n", [2_000, _N])
def test_open_static_certification_of_rounding_singular_band(n) -> None:
    matrix = _band_matrix(n)
    _q, info = _call_under_test(assembly._solve_reduced_system, matrix, np.ones(n), "direct")
    assert info["status"] == "failed"


@requires_pypardiso
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "OPEN: PARDISO mtype 2 (Cholesky, MatrixClass.SPD) factorizes the "
        "singular free spring chain through a rounding-level pivot and returns "
        "|x| ~ 1e16; iparm[13] is not a documented pivot count for mtype 2, so "
        "no perturbation can be detected there (metadata says 'unsupported')."
        + _CERTIFICATION_SLICE
    ),
)
def test_open_spd_cholesky_accepts_rounding_singular_chain() -> None:
    # Only the certification outcome: a correct refusal (by any route) XPASSes.
    handle = _call_under_test(PyPardisoSolverBackend().factorize, _free_chain(), MatrixClass.SPD)
    assert handle.status == "failed"


def test_minres_inconsistent_singular_system_is_not_converged() -> None:
    # Formerly a strict xfail (MINRES reported 'converged' for A=[[1,-1],[-1,1]],
    # b=[1,1] with x=[0,0], relative residual 1). Closed by the iterative
    # true-residual verification (I1, PR #87): the status is now decided by
    # the verified residual, not SciPy's info flag.
    matrix = sparse.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    rhs = np.ones(2)
    _q, info = _call_under_test(assembly._solve_reduced_system, matrix, rhs, "minres")
    assert info["status"] != "converged"


@requires_pypardiso
def test_real_nonsingular_large_matrix_still_uses_pardiso_unchanged() -> None:
    matrix = _band_matrix(shift=1.0)
    assert matrix.shape[0] >= 10_000 and matrix.nnz >= 250_000
    backend = AutoSparseSolverBackend()
    for matrix_class in (MatrixClass.SPD, MatrixClass.SYMMETRIC_INDEFINITE, MatrixClass.GENERAL):
        handle = backend.factorize(matrix, matrix_class)
        assert handle.backend_name == "pypardiso"
        assert handle.status == "ok"
        assert handle.metadata["auto_backend_policy"] in ("pypardiso_large_matrix", "pypardiso_compatible_pattern")
        mtype = str(handle.metadata["pardiso_mtype"])
        expected = "unsupported" if mtype == "2" else 0
        assert handle.metadata["pardiso_perturbed_pivots"][mtype] == expected
        rhs = np.linspace(1.0, 2.0, matrix.shape[0])
        x = handle.solve(rhs)
        assert np.max(np.abs(matrix @ x - rhs)) <= 1.0e-9 * np.max(np.abs(rhs))
    backend.release_pattern_slots()


@requires_pypardiso
def test_real_warm_reused_slot_rejects_singular_values_and_recovers() -> None:
    backend = AutoSparseSolverBackend()
    regular = _band_matrix(shift=1.0)
    singular = _band_matrix(shift=0.0)
    assert np.array_equal(regular.indptr, singular.indptr)
    assert np.array_equal(regular.indices, singular.indices)

    first = backend.factorize(regular, MatrixClass.SYMMETRIC_INDEFINITE)
    assert first.backend_name == "pypardiso" and first.status == "ok"

    reused = backend.factorize(singular, MatrixClass.SYMMETRIC_INDEFINITE)
    assert reused.metadata["pypardiso_compatible_pattern_before_selection"] is True
    assert reused.backend_name == "scipy_superlu"
    assert reused.metadata["auto_backend_policy"] == "scipy_after_pypardiso_failure"
    assert reused.metadata["fallback_failure_kind"] == "perturbed_pivots"

    # The failed warm slot was released, so the recovery is a fresh analysis.
    assert backend.pardiso_backend.retained_pattern_slots == 0
    recovered = backend.factorize((regular * 3.0).tocsr(), MatrixClass.SYMMETRIC_INDEFINITE)
    assert recovered.backend_name == "pypardiso" and recovered.status == "ok"
    assert recovered.metadata["pardiso_symbolic_reused"] is False
    rhs = np.ones(regular.shape[0])
    np.testing.assert_allclose((regular * 3.0) @ recovered.solve(rhs), rhs, atol=1.0e-9)
    # Option B: the handle created before the failed refactorization is stale
    # (its slot was refactorized and released) and is never rebuilt.
    with pytest.raises(linalg.PardisoStaleHandleError):
        first.solve(rhs)
    backend.release_pattern_slots()


@requires_pypardiso
def test_real_general_warm_slot_perturbed_phase22_is_rejected() -> None:
    """Review R2 P1 (real MKL): GENERAL band shift 1 then shift 0 on one pattern.

    Phase 22 perturbs a pivot; the old phase 12 re-analysis of the singular
    values reported none and returned max|x| ~ 1e15, residual ~29, as ok.
    """

    regular = _band_matrix(shift=1.0)
    singular = _band_matrix(shift=0.0)
    rhs = np.ones(regular.shape[0])

    backend = AutoSparseSolverBackend()
    first = backend.factorize(regular, MatrixClass.GENERAL)
    assert first.backend_name == "pypardiso" and first.status == "ok"
    reused = backend.factorize(singular, MatrixClass.GENERAL)
    assert reused.metadata["pypardiso_compatible_pattern_before_selection"] is True
    assert reused.backend_name == "scipy_superlu"
    assert reused.metadata["fallback_failure_kind"] == "perturbed_pivots"
    assert reused.metadata["fallback_pardiso_perturbed_pivots"]["11"] >= 1
    assert backend.pardiso_backend.retained_pattern_slots == 0
    with pytest.raises(linalg.PardisoStaleHandleError):  # option B: never rebuilt
        first.solve(rhs)
    backend.release_pattern_slots()

    forced_backend = AutoSparseSolverBackend()
    options = {"backend": "pypardiso"}
    assert forced_backend.factorize(regular, MatrixClass.GENERAL, options=options).status == "ok"
    forced = forced_backend.factorize(singular, MatrixClass.GENERAL, options=options)
    assert forced.backend_name == "pypardiso"
    assert forced.status == "failed"
    assert forced.metadata["pardiso_failure_kind"] == "perturbed_pivots"
    forced_backend.release_pattern_slots()


def _beam_chain(n_elements: int = 40):
    from anyfesolver import FixedSupport
    from anyfesolver.elements import BeamElement
    from anyfesolver.fe_core import FEModel

    model = FEModel("stale_stream")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node in range(n_elements + 1):
        model.add_node(node + 1, 0.25 * node, 0.01 * node * node, 0.0)
    for element in range(n_elements):
        model.add_element(element + 1, BeamElement(element + 1, [element + 1, element + 2], "steel",
            {"area": 1.0e-2, "Iy": 2.0e-5, "Iz": 3.0e-5, "J": 4.0e-5}))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    return model


def _stream_cases(n_nodes: int, count: int = 5):
    from anyfesolver import LoadCase, StaticCase

    cases = []
    for index in range(count):
        load = LoadCase(str(index))
        for node in range(2, n_nodes + 1, 7):
            load.add_nodal_load(node, [1.0e3 * (index + 1), -2.0e2 * node, 3.0e1 * index, 0.0, 5.0, -7.0 * index])
        cases.append(StaticCase(str(index), load, float(index)))
    return cases


@requires_pypardiso
@pytest.mark.parametrize("interference", ["none", "shared_slot", "eviction"])
def test_b_static_stream_keeps_its_leased_handle_bitwise_identically(interference) -> None:
    """iter_linear_batches pins its operator with a lease (design D2).

    A consumer factorization between blocks either targets the stream's
    mtype -2 slot (a SEMIDEFINITE matrix with the same pattern) or evicts it
    (five other patterns past the 4-slot LRU).  The stream must continue and
    stay bitwise identical to the undisturbed stream.  Contract change (owner
    decision 2026-11-25, D2): on a5beb3f34 the stream re-fetched its stale
    handle through the session cache (2 extra misses); the leased handle is
    never stale, so there is no re-fetch, and the same-pattern newcomer takes
    a new slot (a lease conflict, fresh analysis) instead of refactorizing
    the stream's slot.
    """

    from anyfesolver import AnalysisSession, iter_linear_batches

    def run(mode):
        model = _beam_chain()
        cases = _stream_cases(41)
        backend = PyPardisoSolverBackend()
        cache = linalg.FactorizationCache(backend=backend)
        blocks = []
        try:
            with AnalysisSession(model, factorization_cache=cache) as session:
                for k, block in enumerate(iter_linear_batches(model, cases, batch_size=2, session=session)):
                    blocks.append(np.array(block.displacements, dtype=float, copy=True))
                    matrix = session.static_operator_plan().matrix
                    if mode == "shared_slot":
                        other = backend.factorize(matrix * (2.0 + k), MatrixClass.SYMMETRIC_SEMIDEFINITE)
                        # Block 0: conflict with the pinned slot (fresh phase 12);
                        # later blocks reuse the newcomer's own slot (phase 22).
                        assert other.status == "ok" and other.metadata["pardiso_symbolic_reused"] is (k > 0)
                        assert other.metadata.get("pardiso_lease_conflict", False) is (k == 0)
                    elif mode == "eviction":
                        for j in range(5):
                            other = sparse.eye(7 + j + 5 * k, format="csr") * (1.0 + j)
                            assert backend.factorize(other, MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
            return np.concatenate([b.reshape(-1) for b in blocks]), cache.diagnostics()
        finally:
            backend.release_pattern_slots()

    expected, _ = run("none")
    observed, diagnostics = run(interference)
    assert observed.shape == expected.shape and observed.size > 0
    assert observed.tobytes() == expected.tobytes()
    assert diagnostics["factorization_failures"] == 0
    assert diagnostics["misses"] == 1  # 3 blocks, never re-fetched


def test_b_handle_holder_refactorizes_instead_of_solving_a_stale_handle(monkeypatch) -> None:
    """Decision 3: holders that keep a handle re-factorize it once it is stale."""

    from anyfesolver._b2_attachment_elimination_dev import EliminatedHandle

    solvers = _install_fake(monkeypatch, lambda matrix: 0)
    backend = PyPardisoSolverBackend(max_pattern_slots=1)
    monkeypatch.setattr(linalg, "DEFAULT_BACKEND", backend)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))

    class Maps:
        reduced, info, primal_count, constraint_count = A, {}, 3, 0

        def solve(self, handle, rhs):
            return handle.solve(rhs)

    held = EliminatedHandle(Maps(), linalg.factorize_cached(A, MatrixClass.GENERAL))
    original = held.handle
    assert backend.factorize((A * 2.0).tocsr(), MatrixClass.GENERAL).status == "ok"
    np.testing.assert_array_equal(held.solve(np.ones(3)), np.ones(3) * 1.0e16)
    assert held.handle is not original and not linalg.factorization_is_stale(held.handle)
    assert solvers[0].phases == [12, 22, 22, 33]  # refactorized into the slot, then solved
