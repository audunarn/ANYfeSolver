"""Infrastructure checks for the SESTRA-inspired FE solver batch."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import anyfesolver
import anyfesolver.assembly as assembly_module
import anyfesolver.linalg as linalg
from anyfesolver import (
    AssemblyError,
    FactorizationCache,
    LoadCase,
    MatrixClass,
    TransientConfig,
    assemble_load_matrix,
    assemble_stiffness_matrix,
    create_fe_result,
    factorize,
    factorize_cached,
    generate_beam_mesh,
    run_infrastructure_benchmarks,
    solve_linear,
    solve_linear_many,
    solve_transient_newmark,
)
from anyfesolver.assembly import build_constraint_transformation, build_reduced_rigid_body_modes
from anyfesolver.baselines import compare_baseline_documents, generate_baseline_document
from anyfesolver.linalg import _HAS_PYPARDISO, AutoSparseSolverBackend
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.buckling import solve_eigenvalue_buckling
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_static import solve_static_nonlinear


def test_public_all_symbols_importable() -> None:
    missing = [name for name in anyfesolver.__all__ if not hasattr(anyfesolver, name)]
    assert missing == []


def test_supported_stiffness_uses_symmetric_congruence_and_reports_backward_error() -> None:
    matrix = sparse.csr_matrix(
        np.asarray(
            (
                (1.0e-12, 1.0e-8),
                (1.0e-8, 1.0),
            ),
            dtype=float,
        )
    )
    expected = np.asarray((1.25, -0.75), dtype=float)
    rhs = np.asarray(matrix @ expected, dtype=float)

    equilibrated, scaling, diagnostics = (
        assembly_module._equilibrate_supported_stiffness(matrix)
    )
    congruence = scaling[:, None] * matrix.toarray() * scaling[None, :]
    assert equilibrated.toarray() == pytest.approx(congruence, rel=0.0, abs=0.0)
    assert np.diag(equilibrated.toarray()) == pytest.approx(np.ones(2))
    assert diagnostics == {
        "id": "SYMMETRIC_DIAGONAL_CONGRUENCE_Q1C_V1",
        "applied": True,
        "diagonal_floor": pytest.approx(1.0e-30),
        "diagonal_ratio": pytest.approx(1.0e12),
        "scaled_diagonal_ratio": pytest.approx(1.0),
        "disposition": "EQUILIBRATED",
    }

    solution, info = assembly_module._solve_reduced_system(
        matrix,
        rhs,
        "direct",
    )
    assert solution == pytest.approx(expected, rel=4.0e-12, abs=4.0e-12)
    assert info["status"] == "converged"
    assert info["equilibration"]["id"] == diagnostics["id"]
    assert info["relative_backward_error"] <= 8.0 * np.finfo(float).eps


def test_supported_stiffness_diagnostics_are_strict_json_and_backward_error_fails_closed() -> None:
    zero = sparse.csr_matrix((2, 2), dtype=float)
    _matrix, _scaling, diagnostics = assembly_module._equilibrate_supported_stiffness(
        zero
    )
    assert diagnostics["disposition"] == "NO_POSITIVE_DIAGONAL"
    assert diagnostics["diagonal_ratio"] is None
    assert diagnostics["scaled_diagonal_ratio"] is None
    json.dumps(diagnostics, allow_nan=False)

    overflow_norm = sparse.csr_matrix(
        np.asarray(
            ((1.0e308, 9.0e307), (9.0e307, 1.0e308)),
            dtype=float,
        )
    )
    expected = np.asarray((1.0e-308, -1.0e-308), dtype=float)
    rhs = np.asarray(overflow_norm @ expected, dtype=float)
    assert np.all(np.isfinite(rhs))
    assert math.isinf(assembly_module._backward_error(overflow_norm, expected, rhs))

    _solution, info = assembly_module._solve_reduced_system(
        overflow_norm,
        rhs,
        "direct",
    )
    assert info["status"] == "failed"
    assert info["relative_backward_error"] is None
    assert info["relative_backward_error_disposition"] == "NONFINITE_OR_OVERFLOW"
    json.dumps(info, allow_nan=False)


def test_root_import_does_not_eagerly_load_optional_accelerators() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (str(repo_root / "src"), env.get("PYTHONPATH", ""))
        if path
    )
    code = (
        "import sys; import anyfesolver; "
        "assert 'pypardiso' not in sys.modules; "
        "assert 'anyfesolver.nonlinear_performance' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_sparse_backend_solves_spd_indefinite_many_and_reports_failure() -> None:
    A = sparse.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    handle = factorize(A, MatrixClass.SPD, signature="spd_test")
    x = handle.solve(np.array([1.0, 2.0]))
    np.testing.assert_allclose(A @ x, [1.0, 2.0])

    rhs_many = np.eye(2)
    X = handle.solve_many(rhs_many)
    np.testing.assert_allclose(A @ X, rhs_many)
    diagnostics = handle.diagnostics()
    assert diagnostics["backend"] in ("scipy_superlu", "pypardiso")
    assert diagnostics["matrix_class"] == MatrixClass.SPD.value
    assert diagnostics["factorization_count"] == 1
    assert diagnostics["solve_count"] == 3

    augmented = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    indefinite = factorize(augmented, MatrixClass.SYMMETRIC_INDEFINITE)
    np.testing.assert_allclose(augmented @ indefinite.solve(np.array([2.0, 5.0])), [2.0, 5.0])

    singular = factorize(sparse.csr_matrix((2, 2)), MatrixClass.SYMMETRIC_SEMIDEFINITE)
    assert singular.status == "failed"
    assert singular.failure_reason


def test_default_sparse_backend_uses_superlu_for_small_matrices_and_can_force_pypardiso() -> None:
    A = sparse.csr_matrix([[4.0, 1.0], [1.0, 3.0]])

    default_handle = factorize(A, MatrixClass.SPD)
    assert default_handle.backend_name == "scipy_superlu"
    assert default_handle.metadata["auto_backend_policy"] == "scipy_small_matrix"

    forced = factorize(A, MatrixClass.SPD, options={"backend": "pypardiso"})
    assert forced.backend_name in ("pypardiso", "scipy_superlu")
    if _HAS_PYPARDISO and forced.backend_name == "pypardiso":
        assert forced.metadata["auto_backend_policy"] == "pypardiso_forced"
    elif _HAS_PYPARDISO:
        assert forced.metadata["auto_backend_policy"] == "scipy_after_pypardiso_failure"
    else:
        assert forced.metadata["auto_backend_policy"] == "scipy_small_matrix"


def test_pypardiso_backend_symmetric_mtypes_pattern_reuse_and_stale_handles() -> None:
    if not _HAS_PYPARDISO:
        pytest.skip("pypardiso not installed")
    from anyfesolver.linalg import PyPardisoSolverBackend

    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    rng = np.random.default_rng(0)
    n = 60
    lower = sparse.random(n, n, density=0.08, random_state=rng, format="csr")
    A = (lower + lower.T + sparse.eye(n) * float(n)).tocsr()
    b = rng.random(n)

    first = backend.factorize(A, MatrixClass.SPD, signature="pattern:v1")
    assert first.status == "ok"
    assert first.metadata["pardiso_mtype"] in (2, -2)
    assert first.metadata["pardiso_symbolic_reused"] is False
    x_first = first.solve(b)
    np.testing.assert_allclose(A @ x_first, b, atol=1.0e-8)

    A_scaled = (A * 1.5).tocsr()
    second = backend.factorize(A_scaled, MatrixClass.SPD, signature="pattern:v2")
    assert second.status == "ok"
    assert second.metadata["pardiso_symbolic_reused"] is True
    np.testing.assert_allclose(A_scaled @ second.solve(b), b, atol=1.0e-8)

    # The first handle's slot was refactorized for A_scaled; solving through
    # the stale handle must transparently rebuild and stay correct for A.
    np.testing.assert_allclose(A @ first.solve(b), b, atol=1.0e-8)
    assert first._solver.stale_rebuild_count == 1

    indefinite = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    handle_indef = backend.factorize(indefinite, MatrixClass.SYMMETRIC_INDEFINITE)
    assert handle_indef.status == "ok"
    np.testing.assert_allclose(indefinite @ handle_indef.solve(np.array([2.0, 5.0])), [2.0, 5.0], atol=1.0e-10)

    backend.release_pattern_slots()
    assert backend._slots == []
    # Handles remain usable after slot release via private rebuild.
    np.testing.assert_allclose(A_scaled @ second.solve(b), b, atol=1.0e-8)


class _BlockingPardisoCalls:
    """Deterministic overlap probe used by the fake PyPardiso tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.block_label: str | None = None
        self.block_entered = threading.Event()
        self.release_block = threading.Event()
        self.overlap = threading.Event()

    def block(self, label: str) -> None:
        self.block_label = label
        self.block_entered.clear()
        self.release_block.clear()
        self.overlap.clear()
        self.max_active = 0

    def call(self, label: str, value=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active > 1:
                self.overlap.set()
        try:
            if label == self.block_label:
                self.block_entered.set()
                if not self.release_block.wait(timeout=3.0):
                    raise RuntimeError(f"timed out waiting to release {label}")
            return value
        finally:
            with self._lock:
                self.active -= 1


class _FakePardisoSolver:
    def __init__(self, name: str, calls: _BlockingPardisoCalls) -> None:
        self.name = name
        self.calls = calls
        self.phase = 0

    def _check_A(self, matrix):
        return matrix

    def _check_b(self, matrix, rhs):
        del matrix
        return np.asarray(rhs, dtype=float)

    def set_phase(self, phase: int) -> None:
        self.phase = int(phase)

    def _call_pardiso(self, matrix, rhs):
        del matrix
        return self.calls.call(f"{self.name}:phase-{self.phase}", np.asarray(rhs, dtype=float).copy())

    def free_memory(self, *, everything: bool) -> None:
        assert everything is True
        self.calls.call(f"{self.name}:free")


def _install_fake_pardiso(monkeypatch):
    calls = _BlockingPardisoCalls()
    solvers = []

    def factory(*, mtype: int):
        del mtype
        solver = _FakePardisoSolver(f"solver-{len(solvers)}", calls)
        solvers.append(solver)
        return solver

    monkeypatch.setattr(linalg, "_new_pypardiso_solver", factory)
    monkeypatch.setattr(linalg, "_ensure_mkl_rt_env", lambda: None)
    return calls, solvers


def test_pypardiso_solves_are_process_serialized_across_independent_caches(monkeypatch) -> None:
    calls, _solvers = _install_fake_pardiso(monkeypatch)
    matrix = sparse.eye(3, format="csr")
    cache_a = FactorizationCache(
        name="pardiso_a",
        backend=linalg.PyPardisoSolverBackend(max_pattern_slots=1),
    )
    cache_b = FactorizationCache(
        name="pardiso_b",
        backend=linalg.PyPardisoSolverBackend(max_pattern_slots=1),
    )
    handle_a = cache_a.factorize(matrix, MatrixClass.SPD, signature="cache-a")
    handle_b = cache_b.factorize(matrix, MatrixClass.SPD, signature="cache-b")
    assert handle_a.status == handle_b.status == "ok"

    calls.block("solver-0:phase-33")
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(handle_a.solve, np.array([1.0, 2.0, 3.0]))
        assert calls.block_entered.wait(timeout=2.0)
        second = executor.submit(handle_b.solve, np.array([3.0, 2.0, 1.0]))
        assert not calls.overlap.wait(timeout=0.1)
        calls.release_block.set()
        np.testing.assert_allclose(first.result(timeout=2.0), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(second.result(timeout=2.0), [3.0, 2.0, 1.0])

    assert calls.max_active == 1


def test_pypardiso_cleanup_and_factorization_share_process_lock(monkeypatch) -> None:
    calls, solvers = _install_fake_pardiso(monkeypatch)
    matrix = sparse.eye(3, format="csr")
    releasing_backend = linalg.PyPardisoSolverBackend(max_pattern_slots=1)
    factoring_backend = linalg.PyPardisoSolverBackend(max_pattern_slots=1)
    assert releasing_backend.factorize(matrix, MatrixClass.SPD).status == "ok"

    calls.block("solver-0:free")
    with ThreadPoolExecutor(max_workers=2) as executor:
        cleanup = executor.submit(releasing_backend.release_pattern_slots)
        assert calls.block_entered.wait(timeout=2.0)
        factorization = executor.submit(factoring_backend.factorize, matrix, MatrixClass.SPD)
        assert not calls.overlap.wait(timeout=0.1)
        assert len(solvers) == 1
        assert not factorization.done()
        calls.release_block.set()
        cleanup.result(timeout=2.0)
        handle = factorization.result(timeout=2.0)

    assert handle.status == "ok"
    assert calls.max_active == 1
    assert releasing_backend.retained_pattern_slots == 0


def test_pypardiso_process_lock_does_not_serialize_scipy() -> None:
    matrix = sparse.csr_matrix([[4.0, 1.0], [1.0, 3.0]])

    def scipy_factorize_and_solve():
        handle = factorize(
            matrix,
            MatrixClass.SPD,
            backend=linalg.SparseSolverBackend(),
        )
        return handle.solve(np.array([1.0, 2.0]))

    with linalg._PYPARDISO_PROCESS_LOCK:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(scipy_factorize_and_solve).result(timeout=2.0)

    np.testing.assert_allclose(matrix @ result, [1.0, 2.0])


def test_auto_sparse_backend_thresholds_are_environment_tunable(monkeypatch) -> None:
    monkeypatch.setenv("FE_SOLVER_PYPARDISO_MIN_DIMENSION", "123")
    monkeypatch.setenv("FE_SOLVER_PYPARDISO_MIN_NNZ", "456")
    backend = AutoSparseSolverBackend()
    assert backend.pypardiso_min_dimension == 123
    assert backend.pypardiso_min_nnz == 456

    A = sparse.eye(4, format="csr")
    handle = factorize(A, MatrixClass.SPD, backend=backend)
    assert handle.metadata["pypardiso_min_dimension"] == 123
    assert handle.metadata["pypardiso_min_nnz"] == 456
    assert handle.metadata["pypardiso_active_min_dimension"] == 123
    assert handle.metadata["pypardiso_active_min_nnz"] == 456
    assert handle.metadata["pypardiso_initialized_before_selection"] is False
    assert handle.metadata["pypardiso_compatible_pattern_before_selection"] is False
    assert handle.metadata["pypardiso_warm_thresholds_active"] is False
    assert handle.metadata["pypardiso_retained_pattern_slots_before_selection"] == 0


def test_auto_sparse_backend_uses_one_coherent_selection_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(linalg, "_HAS_PYPARDISO", True)

    class SelectionSpy:
        def __init__(self) -> None:
            self.calls = 0

        def selection_state(self, matrix, matrix_class):
            del matrix, matrix_class
            self.calls += 1
            return True, 2, True

        @property
        def initialized(self):
            raise AssertionError("legacy initialized accessor was used")

        @property
        def retained_pattern_slots(self):
            raise AssertionError("legacy retained-slot accessor was used")

        def has_compatible_pattern(self, matrix, matrix_class):
            del matrix, matrix_class
            raise AssertionError("legacy pattern accessor was used")

    spy = SelectionSpy()
    backend = AutoSparseSolverBackend(
        pardiso_backend=spy,
        pypardiso_min_dimension=200,
        pypardiso_min_nnz=200,
        pypardiso_warm_min_dimension=100,
        pypardiso_warm_min_nnz=100,
    )
    handle = backend.factorize(sparse.eye(2, format="csr"), MatrixClass.SPD)

    assert handle.backend_name == "scipy_superlu"
    assert spy.calls == 1
    assert handle.metadata["pypardiso_initialized_before_selection"] is True
    assert handle.metadata["pypardiso_retained_pattern_slots_before_selection"] == 2
    assert handle.metadata["pypardiso_compatible_pattern_before_selection"] is True
    assert handle.metadata["pypardiso_warm_thresholds_active"] is True
    assert handle.metadata["pypardiso_active_min_dimension"] == 100
    assert handle.metadata["pypardiso_active_min_nnz"] == 100


def test_auto_sparse_backend_uses_warm_thresholds_only_for_a_compatible_pattern(
    monkeypatch,
) -> None:
    for name in (
        "FE_SOLVER_PYPARDISO_MIN_DIMENSION",
        "FE_SOLVER_PYPARDISO_MIN_NNZ",
        "FE_SOLVER_PYPARDISO_WARM_MIN_DIMENSION",
        "FE_SOLVER_PYPARDISO_WARM_MIN_NNZ",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(linalg, "_HAS_PYPARDISO", True)

    A = sparse.csr_matrix(
        [
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
        ]
    )
    pardiso = linalg.PyPardisoSolverBackend(max_pattern_slots=2)
    prepared = linalg._pardiso_prepared_matrix(A, 2)
    pardiso._slots.append(linalg._PardisoPatternSlot(object(), 2, prepared))

    def fake_pardiso_factorize(matrix, matrix_class, *, signature=None, options=None):
        return linalg.FactorizationHandle(
            matrix_shape=matrix.shape,
            matrix_class=matrix_class,
            backend_name="pypardiso",
            ordering="MKL PARDISO",
            signature=signature,
            factorization_time=0.0,
            _solver=object(),
        )

    monkeypatch.setattr(pardiso, "factorize", fake_pardiso_factorize)
    backend = AutoSparseSolverBackend(
        pardiso_backend=pardiso,
        pypardiso_min_dimension=100,
        pypardiso_min_nnz=100,
        pypardiso_warm_min_dimension=3,
        pypardiso_warm_min_nnz=7,
    )

    reused = backend.factorize(A * 1.5, MatrixClass.SPD)
    assert reused.backend_name == "pypardiso"
    assert reused.metadata["auto_backend_policy"] == "pypardiso_compatible_pattern"
    assert reused.metadata["pypardiso_initialized_before_selection"] is True
    assert reused.metadata["pypardiso_compatible_pattern_before_selection"] is True
    assert reused.metadata["pypardiso_warm_thresholds_active"] is True
    assert reused.metadata["pypardiso_active_min_dimension"] == 3
    assert reused.metadata["pypardiso_active_min_nnz"] == 7

    changed_pattern = sparse.eye(3, format="csr")
    cold = backend.factorize(changed_pattern, MatrixClass.SPD)
    assert cold.backend_name == "scipy_superlu"
    assert cold.metadata["pypardiso_initialized_before_selection"] is True
    assert cold.metadata["pypardiso_compatible_pattern_before_selection"] is False
    assert cold.metadata["pypardiso_warm_thresholds_active"] is False
    assert cold.metadata["pypardiso_active_min_dimension"] == 100
    assert cold.metadata["pypardiso_active_min_nnz"] == 100

    incompatible_class = backend.factorize(A, MatrixClass.GENERAL)
    assert incompatible_class.backend_name == "scipy_superlu"
    assert incompatible_class.metadata["pypardiso_initialized_before_selection"] is True
    assert incompatible_class.metadata["pypardiso_compatible_pattern_before_selection"] is False


def test_factorization_cache_reuses_same_matrix_and_separates_changed_values() -> None:
    cache = FactorizationCache(name="unit_cache", max_entries=4)
    A = sparse.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    B = sparse.csr_matrix([[5.0, 1.0], [1.0, 3.0]])

    first = factorize_cached(A, MatrixClass.SPD, cache=cache)
    second = factorize_cached(A.copy(), MatrixClass.SPD, cache=cache)
    third = factorize_cached(B, MatrixClass.SPD, cache=cache)

    assert first is second
    assert third is not first
    diagnostics = cache.diagnostics()
    assert diagnostics["hits"] == 1
    assert diagnostics["misses"] == 2
    assert diagnostics["entries"] == 2
    np.testing.assert_allclose(A @ first.solve(np.array([1.0, 0.0])), [1.0, 0.0])


def test_factorization_cache_serializes_concurrent_mutation(monkeypatch) -> None:
    cache = FactorizationCache(name="threaded_cache", max_entries=2)
    matrix = sparse.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    original_factorize = linalg.factorize
    call_count = 0
    call_count_lock = threading.Lock()
    start = threading.Barrier(8)

    def slow_factorize(*args, **kwargs):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        time.sleep(0.02)
        return original_factorize(*args, **kwargs)

    def cached_call():
        start.wait()
        return cache.factorize(matrix, MatrixClass.SPD, signature="shared")

    monkeypatch.setattr(linalg, "factorize", slow_factorize)
    with ThreadPoolExecutor(max_workers=8) as executor:
        handles = list(executor.map(lambda _index: cached_call(), range(8)))

    assert all(handle is handles[0] for handle in handles)
    assert call_count == 1
    assert cache.diagnostics() == {
        "name": "threaded_cache",
        "max_entries": 2,
        "entries": 1,
        "hits": 7,
        "misses": 1,
        "factorization_failures": 0,
        "backend": cache.diagnostics()["backend"],
    }


def test_multiple_rhs_static_solve_matches_individual_solves() -> None:
    model = generate_beam_mesh(1.0, num_divisions=2, cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6})
    load_x = LoadCase("x")
    load_x.add_nodal_load(3, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_z = LoadCase("z")
    load_z.add_nodal_load(3, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0])

    F_many, load_info = assemble_load_matrix(model, [load_x, load_z])
    assert F_many.shape[1] == 2
    assert load_info["num_load_cases"] == 2

    many, info = solve_linear_many(model, [load_x, load_z])
    one_x, _ = solve_linear(model, load_x)
    one_z, _ = solve_linear(model, load_z)
    assert info["status"] == "converged"
    assert info["backend"]["factorization_count"] == 1
    assert info["factorization_cache"]["misses"] == 1
    assert info["equilibration"]["id"] == "SYMMETRIC_DIAGONAL_CONGRUENCE_Q1C_V1"
    assert info["relative_backward_error"] <= info["relative_backward_error_limit"]
    assert info["relative_backward_error_disposition"] == "CERTIFIED"
    assert info["result_case"]["analysis_case"]["analysis_type"] == "linear_static_many"
    assert [item["name"] for item in info["result_case"]["analysis_case"]["load_cases"]] == ["x", "z"]
    np.testing.assert_allclose(many[:, 0], one_x, rtol=1.0e-9, atol=1.0e-12)
    np.testing.assert_allclose(many[:, 1], one_z, rtol=1.0e-9, atol=1.0e-12)


def test_multiple_rhs_backward_error_overflow_fails_closed(monkeypatch) -> None:
    model = generate_beam_mesh(
        1.0,
        num_divisions=2,
        cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
    )
    first = LoadCase("first")
    first.add_nodal_load(3, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    second = LoadCase("second")
    second.add_nodal_load(3, [0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        assembly_module,
        "_backward_error",
        lambda _matrix, _solution, _rhs: float("inf"),
    )

    displacement, info = solve_linear_many(model, [first, second])
    assert info["status"] == "failed"
    assert info["relative_backward_error"] is None
    assert info["relative_backward_error_disposition"] == "NONFINITE_OR_OVERFLOW"
    np.testing.assert_array_equal(displacement, np.zeros_like(displacement))
    json.dumps(info, allow_nan=False)


def test_static_result_provenance_is_attached_to_solver_info_and_fe_result() -> None:
    model = generate_beam_mesh(1.0, num_divisions=1)
    load_case = LoadCase("tip_z")
    load_case.add_nodal_load(2, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0])

    displacements, solver_info = solve_linear(model, load_case)
    result_case = solver_info["result_case"]

    assert result_case["analysis_case"]["analysis_type"] == "linear_static"
    assert result_case["analysis_case"]["load_cases"][0]["name"] == "tip_z"
    assert result_case["matrix_signature"]
    assert result_case["load_signature"]
    assert result_case["solver_backend"] in ("scipy_superlu", "pypardiso")

    result = create_fe_result(model, displacements, solver_info)
    assert result.result_case == result_case


def test_buckling_transient_and_nonlinear_results_include_result_cases() -> None:
    column = FEModel("column")
    column.add_material("steel", 210.0e9, 0.3)
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for node_id, x in [(1, 0.0), (2, 1.0), (3, 2.0)]:
        column.add_node(node_id, x, 0.0, 0.0)
    column.add_element(1, BeamElement(1, [1, 2], "steel", section))
    column.add_element(2, BeamElement(2, [2, 3], "steel", section))
    column.add_boundary_condition(BoundaryCondition("suppress", [1, 2, 3], {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0}))
    column.add_boundary_condition(BoundaryCondition("pins", [1, 3], {"uy": 0.0}))
    buckling = solve_eigenvalue_buckling(column, {1: {"axial_compression": 1.0}, 2: {"axial_compression": 1.0}}, num_modes=1)
    assert buckling.result_case["analysis_case"]["analysis_type"] == "linear_buckling"
    assert buckling.result_case["matrix_signature"]

    sdof = FEModel("sdof")
    sdof.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    sdof.add_node(1, 0.0, 0.0, 0.0)
    sdof.add_node(2, 1.0, 0.0, 0.0)
    sdof.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    sdof.add_boundary_condition(FixedSupport("fixed", [1]))
    sdof.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    transient_load = LoadCase("step")
    transient_load.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    transient = solve_transient_newmark(sdof, TransientConfig(dt=0.001, t_end=0.002), base_load_case=transient_load)
    assert transient.result_case["analysis_case"]["analysis_type"] == "linear_transient"
    assert transient.result_case == transient.diagnostics["result_case"]

    nonlinear_model = generate_beam_mesh(1.0, num_divisions=1)
    nonlinear_load = LoadCase("small_axial")
    nonlinear_load.add_nodal_load(2, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nonlinear = solve_static_nonlinear(nonlinear_model, nonlinear_load, num_steps=1, max_iterations=5)
    assert nonlinear.info["result_case"]["analysis_case"]["analysis_type"] == "nonlinear_static"


def test_revision_signatures_and_sparsity_cache_are_safe_for_geometry_and_topology() -> None:
    model = generate_beam_mesh(1.0, num_divisions=1)
    K1, info1 = assemble_stiffness_matrix(model)
    sig1 = info1["sparsity_signature"]
    rev1 = model.revision_signature()

    model.set_node_coordinates(2, 1.0, 0.0, 0.001)
    K2, info2 = assemble_stiffness_matrix(model)
    rev2 = model.revision_signature()
    assert info2["sparsity_signature"] == sig1
    assert rev2["geometry"] > rev1["geometry"]
    assert K2.shape == K1.shape

    model.add_node(3, 2.0, 0.0, 0.0)
    model.add_element(2, BeamElement(2, [2, 3], "steel", {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    _K3, info3 = assemble_stiffness_matrix(model)
    assert info3["sparsity_signature"] != sig1


def test_adding_an_element_preserves_unrelated_element_local_caches() -> None:
    model = generate_beam_mesh(1.0, num_divisions=1)
    model.add_node(3, 2.0, 0.0, 0.0)
    first = model.mesh.get_element(1)
    assert first is not None
    material = model.get_material(first.material_name)
    cached = first.compute_stiffness_matrix(model.mesh, material)
    assert first._stiffness_matrix is cached

    model.add_element(
        2,
        BeamElement(
            2,
            [2, 3],
            first.material_name,
            {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
        ),
    )
    assert first._stiffness_matrix is cached


def test_adding_precomputed_element_clears_only_its_cross_mesh_caches() -> None:
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 2.0e-6, "J": 1.0e-6}
    source = FEModel("source")
    source.add_material("steel", 210.0e9, 0.3, density=7850.0)
    source.add_node(1, 0.0, 0.0, 0.0)
    source.add_node(2, 1.0, 0.0, 0.0)
    incoming = BeamElement(1, [1, 2], "steel", section)
    source_stiffness = incoming.compute_stiffness_matrix(
        source.mesh,
        source.get_material("steel"),
    ).copy()
    incoming._mass_matrix = np.ones((12, 12))
    incoming._internal_forces = np.ones(12)
    incoming._nl_cache = {"mesh": "source"}
    incoming._hourglass_stiffness_matrix = np.eye(12)

    target = FEModel("target")
    target.add_material("steel", 210.0e9, 0.3, density=7850.0)
    target.add_node(1, 0.0, 0.0, 0.0)
    target.add_node(2, 2.0, 0.0, 0.0)
    target.add_element(1, incoming)

    for name in (
        "_stiffness_matrix",
        "_mass_matrix",
        "_internal_forces",
        "_nl_cache",
        "_hourglass_stiffness_matrix",
    ):
        assert getattr(incoming, name) is None
    target_stiffness = incoming.compute_stiffness_matrix(
        target.mesh,
        target.get_material("steel"),
    )
    assert not np.allclose(target_stiffness, source_stiffness)


def test_adding_new_nodes_preserves_element_caches_and_rejects_duplicate_ids() -> None:
    model = generate_beam_mesh(1.0, num_divisions=1)
    first = model.mesh.get_element(1)
    assert first is not None
    cached = first.compute_stiffness_matrix(
        model.mesh,
        model.get_material(first.material_name),
    )
    revisions_before = model.revision_signature()

    model.add_node(3, 2.0, 0.0, 0.0)
    assert first._stiffness_matrix is cached
    assert model.revision_signature()["geometry"] == revisions_before["geometry"] + 1

    total_dofs = model.mesh.dof_manager.total_dofs
    revisions_before_duplicate = model.revision_signature()
    node_before_duplicate = model.mesh.get_node(3)
    with pytest.raises(ValueError, match="already exists"):
        model.add_node(3, 99.0, 0.0, 0.0)
    assert model.mesh.get_node(3) is node_before_duplicate
    assert model.mesh.dof_manager.total_dofs == total_dofs
    assert model.revision_signature() == revisions_before_duplicate


class _BadElement:
    material_name = "steel"

    def __init__(self, values: np.ndarray):
        self.node_ids = [1, 2]
        self.values = np.asarray(values, dtype=float)

    def get_dof_mapping(self, mesh):
        return np.asarray(mesh.get_node(1).dofs + mesh.get_node(2).dofs, dtype=int)

    def compute_stiffness_matrix(self, mesh, material):
        return self.values


def _bad_model(values: np.ndarray) -> FEModel:
    model = FEModel("bad")
    model.add_material("steel", 210.0e9, 0.3)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, _BadElement(values))
    return model


def test_assembly_diagnostics_reject_nonfinite_and_nonsymmetric_element_matrices() -> None:
    with pytest.raises(AssemblyError, match="non-finite"):
        assemble_stiffness_matrix(_bad_model(np.full((12, 12), np.nan)))

    nonsymmetric = np.eye(12)
    nonsymmetric[0, 1] = 0.25
    with pytest.raises(AssemblyError, match="nonsymmetric"):
        assemble_stiffness_matrix(_bad_model(nonsymmetric))


def _free_two_body_model() -> FEModel:
    model = FEModel("two_free_beams")
    model.add_material("steel", 210.0e9, 0.3)
    section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    for node_id, x in [(1, 0.0), (2, 1.0), (3, 3.0), (4, 4.0)]:
        model.add_node(node_id, x, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_element(2, BeamElement(2, [3, 4], "steel", section))
    return model


def test_connected_component_nullspace_counts_and_free_free_rejection() -> None:
    supported = generate_beam_mesh(1.0, num_divisions=1)
    supported.apply_boundary_conditions()
    K, _ = assemble_stiffness_matrix(supported)
    _, F_red, _T, _u0, independent, _constraint = build_constraint_transformation(K, np.zeros(K.shape[0]), supported)
    Q, supported_info = build_reduced_rigid_body_modes(supported, independent, K.shape[0])
    assert F_red.shape[0] == K.shape[0] - 6
    assert Q.shape[1] == 0
    assert supported_info["rank"] == 0

    one_free = generate_beam_mesh(1.0, num_divisions=1)
    one_free.boundary_conditions.clear()
    load = LoadCase("unbalanced")
    load.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _u, info = solve_linear(one_free, load)
    assert info["convergence_info"]["status"] == "incompatible_free_free_load"
    assert info["nullspace_info"]["rank"] == 6

    two_free = _free_two_body_model()
    _u2, info2 = solve_linear(two_free, None)
    assert info2["convergence_info"]["status"] == "converged"
    assert info2["nullspace_info"]["component_count"] == 2
    assert info2["nullspace_info"]["rank"] == 12


def test_baseline_generation_is_deterministic_for_values_and_comparison() -> None:
    first = generate_baseline_document(include_timing=False)
    second = generate_baseline_document(include_timing=False)
    first["generated_at"] = second["generated_at"] = "ignored"
    assert first == second
    assert compare_baseline_documents(first, second)["status"] == "passed"


def test_infrastructure_benchmarks_report_timing_and_memory() -> None:
    report = run_infrastructure_benchmarks()
    assert report["cases"]
    assert {
        "static_beam",
        "multi_rhs_static",
        "shell_stiffness_S4_cold_warm",
        "shell_stiffness_Q8_cold_warm",
        "shell_stiffness_Q8R_cold_warm",
        "shell_assembly",
        "beam_column_buckling",
        "transient_newmark",
        "factorization_reuse",
        "nonlinear_assembly",
        "fracture_damage",
    } <= set(report["cases"])
    for case in report["cases"].values():
        assert case["timing"]["wall_seconds"] >= 0.0
        assert case["memory"]["peak_bytes"] >= 0
        assert case["status"] in {"completed", "converged", "ok", "skipped"}
    nonlinear = report["cases"]["nonlinear_assembly"]
    if nonlinear["status"] == "completed":
        assert nonlinear["results"]["relative_force_error"] < 1.0e-10
        assert nonlinear["results"]["relative_tangent_error"] < 1.0e-9
    for name in ("shell_stiffness_S4_cold_warm", "shell_stiffness_Q8_cold_warm", "shell_stiffness_Q8R_cold_warm"):
        case = report["cases"][name]
        assert case["jit_enabled"] in {True, False}
        assert "parallel_threads" in case
        assert case["timing"]["cold_assembly_seconds"] >= 0.0
        assert case["timing"]["warm_assembly_seconds"] >= 0.0
        assert case["timing"]["warm_speedup"] >= 0.0
        assert case["results"]["matrix_difference_norm"] < 1.0e-12
    fracture = report["cases"]["fracture_damage"]
    assert fracture["timing"]["static_deletion_scan_seconds"] >= 0.0
    assert fracture["timing"]["impact_damage_state_update_count"] >= 0
    assert fracture["timing"]["impact_eroded_matrix_rebuild_count"] == 0
    assert fracture["results"]["sub_softening_rebuilds_skipped"] is True


def test_solve_linear_many_uses_revision_signature_for_factorization_cache(monkeypatch) -> None:
    model = generate_beam_mesh(1.0, num_divisions=2)
    load = LoadCase("tip")
    load.add_nodal_load(3, [0.0, 0.0, -1.0, 0.0, 0.0, 0.0])

    def fail_content_hash(_matrix):
        raise AssertionError("solve_linear_many should use a revision signature instead of content hashing K")

    monkeypatch.setattr(linalg, "sparse_matrix_signature", fail_content_hash)
    _U, info = solve_linear_many(model, [load])
    assert info["status"] == "converged"
    assert info["factorization_cache"]["misses"] == 1


def test_verification_report_quick_mode_writes_json_and_markdown() -> None:
    output_dir = Path(".pytest_tmp_fe_verification") / str(os.getpid())
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [sys.executable, "scripts/run_fe_verification.py", "--quick", "--output-dir", str(output_dir)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        report_path = output_dir / "fe_verification_report.json"
        markdown_path = output_dir / "fe_verification_report.md"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert markdown_path.exists()
        assert report["status"] == "passed"
        assert report["environment"]["dependencies"]["scipy"] is not None
        assert {item["name"] for item in report["commands"]} == {"public_imports", "baseline_generation_smoke"}
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


def test_verification_report_family_mode_runs_independent_transient_family() -> None:
    output_dir = Path(".pytest_tmp_fe_verification_family") / str(os.getpid())
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [sys.executable, "scripts/run_fe_verification.py", "--transient", "--output-dir", str(output_dir)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        report = json.loads((output_dir / "fe_verification_report.json").read_text(encoding="utf-8"))
        assert report["selected_families"] == ["transient"]
        assert [item["name"] for item in report["commands"]] == ["public_imports", "transient_tests"]
        assert report["status"] == "passed"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)



@pytest.mark.parametrize("base_location", ["base_prefix", "base_exec_prefix"])
def test_mkl_runtime_discovery_checks_virtualenv_base_prefixes(monkeypatch, tmp_path, base_location):
    import ctypes.util
    import site
    monkeypatch.setattr(linalg, "_HAS_PYPARDISO", True)
    monkeypatch.setattr(linalg, "_MKL_RT_ENV_READY", False)
    monkeypatch.setenv("PYPARDISO_MKL_RT", "")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.setattr(sys, "base_exec_prefix", str(tmp_path / "exec-python"))
    monkeypatch.setattr(site, "USER_BASE", str(tmp_path / "user"))
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    runtime = Path(getattr(sys, base_location)) / "Library" / "bin" / "mkl_rt.3.dll"
    directories = []
    def glob(directory, pattern):
        assert pattern == "mkl_rt*"
        directories.append(directory)
        return [runtime] if directory == runtime.parent else []
    monkeypatch.setattr(Path, "glob", glob)
    loaded = []
    monkeypatch.setattr(linalg.ctypes, "CDLL", lambda path: loaded.append(path))
    linalg._ensure_mkl_rt_env()
    expected = os.path.normcase(os.path.abspath(str(runtime)))
    assert os.environ["PYPARDISO_MKL_RT"] == expected
    assert loaded == [expected]
    assert runtime.parent in directories
    assert len(directories) == len(set(directories))


def test_mkl_runtime_discovery_deduplicates_prefixes(monkeypatch, tmp_path):
    import ctypes.util
    import site
    monkeypatch.setattr(linalg, "_HAS_PYPARDISO", True)
    monkeypatch.setattr(linalg, "_MKL_RT_ENV_READY", False)
    monkeypatch.setenv("PYPARDISO_MKL_RT", "")
    prefix = str(tmp_path / "python")
    monkeypatch.setenv("CONDA_PREFIX", prefix)
    for name in ("prefix", "base_prefix", "base_exec_prefix"):
        monkeypatch.setattr(sys, name, prefix)
    monkeypatch.setattr(site, "USER_BASE", prefix)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    directories = []
    monkeypatch.setattr(Path, "glob", lambda directory, pattern: directories.append(directory) or [])
    linalg._ensure_mkl_rt_env()
    assert directories == [Path(prefix) / "Library" / "bin", Path(prefix) / "DLLs", Path(prefix) / "lib"]


def test_mkl_runtime_discovery_preserves_explicit_environment(monkeypatch):
    monkeypatch.setattr(linalg, "_HAS_PYPARDISO", True)
    monkeypatch.setattr(linalg, "_MKL_RT_ENV_READY", False)
    monkeypatch.setenv("PYPARDISO_MKL_RT", "explicit-runtime.dll")
    def unexpected(*args):
        pytest.fail("Explicit MKL runtime must not trigger discovery or loading")
    monkeypatch.setattr(Path, "glob", unexpected)
    monkeypatch.setattr(linalg.ctypes, "CDLL", unexpected)
    linalg._ensure_mkl_rt_env()
    assert os.environ["PYPARDISO_MKL_RT"] == "explicit-runtime.dll"
    assert linalg._MKL_RT_ENV_READY is True
