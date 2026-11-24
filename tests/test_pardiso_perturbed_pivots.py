"""PARDISO factorizations with perturbed pivots are failures, never solutions.

MKL PARDISO replaces (near-)zero pivots by a small perturbation instead of
failing (output ``iparm(14)``, 0-based ``iparm[13]``).  A perturbed
factorization of a singular matrix returns a huge solution whose normwise
backward error can be tiny, so the static solve used to report it as
converged/CERTIFIED.  These tests pin the fail-closed behaviour:

- the auto backend treats perturbed pivots as a PARDISO failure and falls
  back to SuperLU, which solves a nonsingular matrix or refuses a singular one;
- an explicitly forced PARDISO backend returns a typed failure instead;
- reused pattern slots (phase 22) and stale-handle rebuilds are checked too;
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
            if self.iparm is not None:
                self.iparm[13] = int(self._perturbed_for(matrix))
            return np.zeros_like(rhs)
        # A perturbed factor "solves" with an enormous answer.
        return np.asarray(rhs, dtype=float) * 1.0e16

    def free_memory(self, *, everything: bool) -> None:
        assert everything is True


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
    assert first.metadata["pardiso_perturbed_pivots"] == {"2": 0}

    reused = backend.factorize(singular, MatrixClass.SPD)
    assert reused.status == "failed"
    assert reused.metadata["pardiso_failure_kind"] == "perturbed_pivots"
    # The retained slot was tried with phase 22 and re-analysed (phase 12)
    # before the attempt was rejected.
    assert 22 in solvers[0].phases

    again = backend.factorize(regular * 2.0, MatrixClass.SPD)
    assert again.status == "ok"
    assert again.metadata["pardiso_symbolic_reused"] is True
    assert again.metadata["pardiso_perturbed_pivots"] == {"2": 0}


def test_fake_stale_handle_rebuild_is_checked(monkeypatch) -> None:
    state = {"perturb": 0}

    def controlled(matrix) -> int:
        del matrix
        return state["perturb"]

    _install_fake(monkeypatch, controlled)
    backend = PyPardisoSolverBackend(max_pattern_slots=1)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    first = backend.factorize(A, MatrixClass.SPD)
    assert first.status == "ok"
    second = backend.factorize(A * 2.0, MatrixClass.SPD)
    assert second.status == "ok"
    state["perturb"] = 1  # the private rebuild of ``first`` now perturbs
    with pytest.raises(RuntimeError, match="perturbed_pivots"):
        first.solve(np.ones(3))


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
# Strict xfails: they turn into failures as soon as the behaviour changes.


@requires_pypardiso
@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN: a rounding-singular matrix (band graph Laplacian, constant "
        "nullspace, inconsistent load) is factorized by SuperLU through an "
        "O(1e-13) pivot; the normwise backward error (~4e-16) then certifies "
        "a ~1e16 solution. Same on main below the PARDISO threshold (n=2000)."
    ),
)
@pytest.mark.parametrize("n", [2_000, _N])
def test_open_static_certification_of_rounding_singular_band(n) -> None:
    matrix = _band_matrix(n)
    _q, info = assembly._solve_reduced_system(matrix, np.ones(n), "direct")
    assert info["status"] == "failed"


@requires_pypardiso
@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN: PARDISO mtype 2 (Cholesky, MatrixClass.SPD) factorizes the "
        "singular free spring chain through a rounding-level pivot with "
        "iparm[13] == 0 (no perturbation is reported) and returns |x| ~ 1e16."
    ),
)
def test_open_spd_cholesky_accepts_rounding_singular_chain() -> None:
    handle = PyPardisoSolverBackend().factorize(_free_chain(), MatrixClass.SPD)
    assert handle.status == "failed"


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
        assert handle.metadata["pardiso_perturbed_pivots"][mtype] == 0
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

    recovered = backend.factorize((regular * 3.0).tocsr(), MatrixClass.SYMMETRIC_INDEFINITE)
    assert recovered.backend_name == "pypardiso" and recovered.status == "ok"
    assert recovered.metadata["pardiso_symbolic_reused"] is True
    rhs = np.ones(regular.shape[0])
    np.testing.assert_allclose((regular * 3.0) @ recovered.solve(rhs), rhs, atol=1.0e-9)
    # The handle created before the failed refactorization still solves its own matrix.
    np.testing.assert_allclose(regular @ first.solve(rhs), rhs, atol=1.0e-9)
    backend.release_pattern_slots()
