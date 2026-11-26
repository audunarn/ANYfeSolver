"""PARDISO slot ownership: explicit leases (design D2, owner decision 2026-11-25).

A holder that solves one factorization across other work (progress
callbacks, eigensolver iterations, stream consumers, other factorizations)
pins it with a :class:`~anyfesolver.linalg.FactorizationLease`.  A pinned
slot is never refactorized, evicted or released: a same-pattern newcomer
takes a new slot (``pardiso_lease_conflicts``), eviction and
``release_pattern_slots`` detach it, and it is freed when the last lease
ends.  Unleased handles keep the fail-closed contract (a stale handle raises
:class:`~anyfesolver.linalg.PardisoStaleHandleError`).  Every MKL instance
has one native owner from birth and is freed exactly once, with the original
exception preserved.  Numbers in the docstrings refer to the acceptance
tests of the design note (section 8).
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
from scipy import sparse

import anyfesolver.linalg as linalg
from anyfesolver.linalg import AutoSparseSolverBackend, MatrixClass, PyPardisoSolverBackend

from test_pardiso_perturbed_pivots import (
    _beam_chain,
    _by_phase,
    _install_fake,
    _stream_cases,
    requires_pypardiso,
)

# The fake solver solves exactly when its factor is ok (rule A's known-solution
# check needs a true solve): diag(1, 2, 3) x = ones.
_DIAG123_SOLUTION = np.ones(3) / np.array([1.0, 2.0, 3.0])


def _counts() -> dict:
    return linalg.backend_operation_counts()


def _delta(before: dict, key: str) -> int:
    return _counts().get(key, 0) - before.get(key, 0)


def _leased(backend) -> int:
    """Pinned slots still alive (0 where leases do not exist, for old-code checks)."""

    return getattr(backend, "leased_pattern_slots", 0)


def _forced_one_slot_default(monkeypatch) -> AutoSparseSolverBackend:
    """Route every default factorization to a one-slot PARDISO backend."""

    backend = AutoSparseSolverBackend(
        pardiso_backend=PyPardisoSolverBackend(max_pattern_slots=1),
        pypardiso_min_dimension=1,
        pypardiso_min_nnz=1,
    )
    monkeypatch.setattr(linalg, "DEFAULT_BACKEND", backend)
    return backend


def _disturb_default() -> None:
    """An unrelated factorization on the shared default backend (evicts a 1-slot LRU)."""

    handle = linalg.factorize(sparse.csr_matrix([[3.0]]), MatrixClass.SPD)
    assert handle.status == "ok" and handle.backend_name == "pypardiso"


# --------------------------------------------------------------------------
# 1-2. Long-lived holders survive unrelated factorizations, bitwise.
# --------------------------------------------------------------------------


def _deterministic_eigsh(monkeypatch, state: dict) -> None:
    """ARPACK with a fixed start vector (run-to-run bitwise), disturbed on demand.

    ``state["disturb"]`` (a callable or None) runs right before the eigensolve,
    i.e. after the inverse operator was built and before it is applied.
    """

    import scipy.sparse.linalg as sparse_linalg

    real_eigsh = sparse_linalg.eigsh

    def eigsh(A, *args, **kwargs):
        kwargs.setdefault("v0", np.linspace(1.0, 2.0, A.shape[0]))
        if state.get("disturb") is not None:
            state["disturb"]()
        return real_eigsh(A, *args, **kwargs)

    monkeypatch.setattr(sparse_linalg, "eigsh", eigsh)


def _newmark(disturb: bool):
    from anyfesolver import LoadCase
    from anyfesolver.dynamics import TransientConfig, solve_transient_newmark

    model = _beam_chain(2)
    load = LoadCase("tip")
    load.add_nodal_load(3, [1.0e3, -2.0e3, 5.0e2, 0.0, 10.0, -20.0])
    calls = {"n": 0}

    def progress(*_args, **_kwargs) -> None:
        calls["n"] += 1
        if disturb and calls["n"] >= 2:  # every step callback (the first precedes K_eff)
            _disturb_default()

    result = solve_transient_newmark(
        model, TransientConfig(dt=0.001, t_end=0.002), base_load_case=load,
        progress_callback=progress,
    )
    return result, calls["n"]


@requires_pypardiso
def test_newmark_effective_operator_survives_a_callback_factorization(monkeypatch) -> None:
    """1 (Codex r5): on a5beb3f34 step two raised PardisoStaleHandleError."""

    backend = _forced_one_slot_default(monkeypatch)
    try:
        expected, _ = _newmark(False)
        before = _counts()
        observed, calls = _newmark(True)
        assert calls >= 2
        assert observed.status == expected.status
        assert observed.displacements.tobytes() == expected.displacements.tobytes()
        assert observed.velocities.tobytes() == expected.velocities.tobytes()
        assert _delta(before, "pardiso_lease_conflicts") == 0
        assert _leased(backend.pardiso_backend) == 0  # lease closed on exit
    finally:
        backend.release_pattern_slots()


@requires_pypardiso
def test_linear_impact_effective_operator_survives_a_callback_factorization(monkeypatch) -> None:
    """2: the linear sphere-impact K_eff holder (same refresh-on-dt-only pattern)."""

    from anyfesolver.contact import (
        RigidSphereImpact, SphereContactConfig, _verification_contact_panel,
        solve_transient_sphere_impact,
    )
    from anyfesolver.dynamics import TransientConfig

    backend = _forced_one_slot_default(monkeypatch)

    def run(disturb: bool):
        calls = {"n": 0}

        def progress(*_args, **_kwargs) -> None:
            calls["n"] += 1
            if disturb:
                _disturb_default()

        result = solve_transient_sphere_impact(
            _verification_contact_panel(),
            TransientConfig(dt=0.0025, t_end=0.02, output_nodes=[1]),
            RigidSphereImpact("lease_hit", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25),
                              travel_direction=(0.0, 0.0, -1.0), speed=2.0),
            SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
            progress_callback=progress,
        )
        return result, calls["n"]

    try:
        expected, _ = run(False)
        observed, calls = run(True)
        assert calls >= 3
        assert observed.status == expected.status
        assert observed.displacements.tobytes() == expected.displacements.tobytes()
        assert _leased(backend.pardiso_backend) == 0
    finally:
        backend.release_pattern_slots()


@requires_pypardiso
def test_modal_shift_invert_operator_survives_a_guard_factorization(monkeypatch) -> None:
    """2: the modal shift-invert inverse operator, disturbed after factorization."""

    from anyfesolver.modal import _sparse_eigensolve

    rng = np.random.default_rng(7)
    n = 120
    lower = sparse.random(n, n, density=0.05, random_state=3, format="csr")
    K = (lower + lower.T + sparse.eye(n) * float(n)).tocsr()
    M = (sparse.eye(n) * 2.0 + sparse.diags(rng.random(n) * 0.1)).tocsr()

    state: dict = {}
    _deterministic_eigsh(monkeypatch, state)

    def run(disturb: bool):
        backend = PyPardisoSolverBackend(max_pattern_slots=1)
        cache = linalg.FactorizationCache(name="lease_modal", backend=backend)

        def guard() -> None:
            if disturb:
                assert backend.factorize(sparse.csr_matrix([[3.0]]), MatrixClass.SPD).status == "ok"

        try:
            values, vectors, info = _sparse_eigensolve(K, M, 3, 0.5, cache, _post_factorization_guard=guard)
            assert info["shift_factorization"]["backend"] == "pypardiso"
            assert _leased(backend) == 0
            return values, vectors
        finally:
            backend.release_pattern_slots()

    expected = run(False)
    observed = run(True)
    assert observed[0].tobytes() == expected[0].tobytes()
    assert observed[1].tobytes() == expected[1].tobytes()


@requires_pypardiso
@pytest.mark.parametrize("shift_load_factor", [None, 50.0])
def test_buckling_inverse_operators_survive_an_iteration_factorization(monkeypatch, shift_load_factor) -> None:
    """2: buckling elastic inverse (SPD) and shift-invert (indefinite) operators."""

    from anyfesolver import solve_eigenvalue_buckling
    from test_analysis_session import _buckling_column

    state: dict = {}
    _deterministic_eigsh(monkeypatch, state)

    def run(disturb: bool):
        backend = PyPardisoSolverBackend(max_pattern_slots=1)

        def other() -> None:
            assert backend.factorize(sparse.csr_matrix([[3.0]]), MatrixClass.SPD).status == "ok"

        state["disturb"] = other if disturb else None
        model, states = _buckling_column(20)
        try:
            result = solve_eigenvalue_buckling(
                model, states, num_modes=2, dense_size_limit=10,
                shift_load_factor=shift_load_factor,
                factorization_cache=linalg.FactorizationCache(name="lease_buckling", backend=backend),
            )
            assert result.solver_status == "ok"
            assert _leased(backend) == 0
            return np.array([mode.load_factor for mode in result.modes])
        finally:
            backend.release_pattern_slots()

    expected = run(False)
    observed = run(True)
    assert observed.size == 2
    assert observed.tobytes() == expected.tobytes()


# --------------------------------------------------------------------------
# 3. The static stream keeps its operator (no mid-stream backend switch).
# --------------------------------------------------------------------------


@requires_pypardiso
@pytest.mark.parametrize("interference", ["none", "release", "shared_slot", "eviction"])
def test_static_stream_keeps_its_pardiso_operator_bitwise(interference) -> None:
    """3 (Codex r5): on a5beb3f34 releasing the warmed slot after block 1 made the
    stream re-fetch through auto selection, switch to SuperLU and change 480
    values (max 4.35e-10)."""

    from anyfesolver import AnalysisSession, iter_linear_batches

    def run(mode):
        model = _beam_chain()
        cases = _stream_cases(41, 10)
        auto = AutoSparseSolverBackend(pypardiso_warm_min_dimension=1, pypardiso_warm_min_nnz=1)
        matrix = AnalysisSession(model).static_operator_plan().matrix
        warm = auto.pardiso_backend.factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE)
        assert warm.status == "ok"
        del warm
        cache = linalg.FactorizationCache(backend=auto)
        blocks, backends = [], []
        before = _counts()
        try:
            with AnalysisSession(model, factorization_cache=cache) as session:
                for k, block in enumerate(iter_linear_batches(model, cases, batch_size=2, session=session)):
                    blocks.append(np.array(block.displacements, dtype=float, copy=True))
                    backends.append(session.static_operator_plan().handle.backend_name)
                    if mode == "release" and k == 0:
                        auto.release_pattern_slots()
                    elif mode == "shared_slot":
                        other = auto.pardiso_backend.factorize(matrix * (2.0 + k), MatrixClass.SYMMETRIC_SEMIDEFINITE)
                        assert other.status == "ok"
                    elif mode == "eviction":
                        for j in range(5):
                            other = sparse.eye(7 + j + 5 * k, format="csr") * (1.0 + j)
                            assert auto.pardiso_backend.factorize(other, MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
            conflicts = _delta(before, "pardiso_lease_conflicts")
            assert _leased(auto.pardiso_backend) == 0
            return np.concatenate([b.reshape(-1) for b in blocks]), backends, cache.diagnostics(), conflicts
        finally:
            auto.release_pattern_slots()

    expected, expected_backends, _, base_conflicts = run("none")
    observed, backends, diagnostics, conflicts = run(interference)
    assert set(expected_backends) == {"pypardiso"} and base_conflicts == 0
    assert set(backends) == {"pypardiso"}
    assert observed.size == 10 * 41 * 6  # 10 cases x 246 DOFs, in 5 blocks
    assert observed.tobytes() == expected.tobytes()
    assert diagnostics["factorization_failures"] == 0
    assert diagnostics["misses"] == 1  # never re-fetched
    # Only a same-pattern newcomer against the pinned slot is a conflict.
    assert conflicts == (1 if interference == "shared_slot" else 0)


# --------------------------------------------------------------------------
# 4. Same-pattern conflict (GENERAL, mtype 11) with real MKL.
# --------------------------------------------------------------------------


@requires_pypardiso
def test_general_same_pattern_conflict_takes_a_new_slot_and_keeps_the_holder_bitwise() -> None:
    """4: a5beb3f34 had no lease (the holder was refactorized in place, stale)."""

    rng = np.random.default_rng(11)
    n = 300
    A = (sparse.random(n, n, density=0.03, random_state=5) + sparse.eye(n) * 4.0).tocsr()
    B = A.copy()
    B.data = B.data * 1.5 + 0.25
    b = rng.random(n)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    try:
        before = _counts()
        with linalg.leased_factorization(A, MatrixClass.GENERAL, backend=backend) as lease:
            assert lease.pinned
            x_before = lease.handle.solve(b)
            newcomer = backend.factorize(B, MatrixClass.GENERAL)
            assert newcomer.status == "ok"
            assert newcomer.metadata["pardiso_symbolic_reused"] is False  # fresh phase 12
            assert newcomer.metadata["pardiso_lease_conflict"] is True
            assert _delta(before, "pardiso_lease_conflicts") == 1
            assert lease.handle.status == "ok" and not linalg.factorization_is_stale(lease.handle)
            assert lease.handle.solve(b).tobytes() == x_before.tobytes()
            np.testing.assert_allclose(B @ newcomer.solve(b), b, atol=1.0e-9)
            assert backend.retained_pattern_slots == 1 and backend.detached_pattern_slots == 1
        assert _leased(backend) == 0
        assert backend.retained_pattern_slots <= backend.max_pattern_slots
        assert backend.detached_pattern_slots == 0
        # The holder's slot was freed at lease end; its handle is now stale.
        with pytest.raises(linalg.PardisoStaleHandleError):
            lease.handle.solve(b)
        assert lease.handle.metadata["pardiso_stale_cause"] == "refactorized"
        # Ordinary reuse continues on the newcomer's slot (phase 22, no conflict).
        again = backend.factorize(A, MatrixClass.GENERAL)
        assert again.metadata["pardiso_symbolic_reused"] is True
        assert "pardiso_lease_conflict" not in again.metadata
        assert _delta(before, "pardiso_lease_conflicts") == 1
    finally:
        backend.release_pattern_slots()


def test_unconflicted_lease_takes_the_unleased_path(monkeypatch) -> None:
    """Without conflicts the phase sequence equals the unleased one (bit-identity claim)."""

    def phases(leased: bool):
        solvers = _install_fake(monkeypatch, lambda matrix: 0)
        backend = PyPardisoSolverBackend(max_pattern_slots=2)
        A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
        first = backend.factorize(A, MatrixClass.GENERAL)
        lease = linalg.lease_handle(first) if leased else None
        backend.factorize(sparse.eye(5, format="csr"), MatrixClass.GENERAL).solve(np.ones(5))
        if lease is not None:
            lease.handle.solve(np.ones(3))
            lease.close()
        later = backend.factorize((A * 2.0).tocsr(), MatrixClass.GENERAL)
        return [s.phases for s in solvers], later.metadata["pardiso_symbolic_reused"]

    before = _counts()
    unleased = phases(False)
    leased = phases(True)
    assert leased[1] is unleased[1] is True
    assert leased[0][1] == unleased[0][1]
    # Each factorization is followed by its rule-A probe solve (33); the lease
    # adds only its own solve.
    assert unleased[0][0] == [12, 33, 22, 33]
    assert leased[0][0] == [12, 33, 33, 22, 33]
    assert _delta(before, "pardiso_lease_conflicts") == 0


# --------------------------------------------------------------------------
# 5. Unleased stale handles: one typed error, a cache miss, never a failure.
# --------------------------------------------------------------------------


def test_unleased_stale_handle_sequential_interleaved_and_rank3(monkeypatch) -> None:
    """5 (Codex r5): the interleaving raised RuntimeError and counted a cache failure."""

    _install_fake(monkeypatch, lambda matrix: 0)
    backend = PyPardisoSolverBackend(max_pattern_slots=1)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))

    # Sequential.
    first = backend.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE)
    assert backend.factorize((A * 2.0).tocsr(), MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
    with pytest.raises(linalg.PardisoStaleHandleError):
        first.solve(np.ones(3))
    # Rank-3 right-hand side on a stale handle: the stale error comes first.
    with pytest.raises(linalg.PardisoStaleHandleError):
        linalg.solve_many(first, np.ones((3, 1, 1)))
    with pytest.raises(linalg.PardisoStaleHandleError):
        first.solve_many(np.ones((3, 1, 1)))

    # Interleaving: the slot is refactorized right after the old pre-check.
    real_check = linalg.factorization_is_stale

    def pausing_check(handle):
        answer = real_check(handle)
        if handle is target["handle"] and not target["done"]:
            target["done"] = True
            assert backend.factorize((A * 3.0).tocsr(), MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
        return answer

    real_policy = linalg._inherited_native_thread_policy

    def pausing_policy(*args, **kwargs):
        # Later point of the same solve (after its handle checks): reached
        # first when the stale/failed decision is atomic (no outer re-check).
        if not target["done"] and target["solving"]:
            target["done"] = True
            assert backend.factorize((A * 3.0).tocsr(), MatrixClass.SYMMETRIC_INDEFINITE).status == "ok"
        return real_policy(*args, **kwargs)

    target = {"handle": backend.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE), "done": False, "solving": True}
    monkeypatch.setattr(linalg, "factorization_is_stale", pausing_check)
    monkeypatch.setattr(linalg, "_inherited_native_thread_policy", pausing_policy)
    with pytest.raises(linalg.PardisoStaleHandleError):
        target["handle"].solve(np.ones(3))
    assert target["done"]
    target["solving"] = False

    cache = linalg.FactorizationCache(name="lease_interleave", backend=backend)
    cached = cache.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE)
    target.update(handle=cached, done=False)
    again = cache.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE)
    assert target["done"] or again is cached
    third = cache.factorize(A, MatrixClass.SYMMETRIC_INDEFINITE)
    assert third.status == "ok" and not real_check(third)
    diagnostics = cache.diagnostics()
    assert diagnostics["factorization_failures"] == 0
    assert diagnostics["hits"] + diagnostics["misses"] == 3


# --------------------------------------------------------------------------
# 6. release_pattern_slots during a lease.
# --------------------------------------------------------------------------


def _counting_frees(monkeypatch, report=lambda matrix: 0):
    solvers = _install_fake(monkeypatch, report)
    frees: dict[int, int] = {}
    factory = linalg._new_pypardiso_solver

    def counting_factory(*, mtype: int):
        solver = factory(mtype=mtype)
        original = solver.free_memory
        frees[id(solver)] = 0

        def free_memory(*, everything: bool, _solver=solver, _original=original):
            frees[id(_solver)] += 1
            return _original(everything=everything)

        solver.free_memory = free_memory
        return solver

    monkeypatch.setattr(linalg, "_new_pypardiso_solver", counting_factory)
    return solvers, frees


def test_release_during_a_lease_detaches_and_frees_once_at_lease_end(monkeypatch) -> None:
    """6."""

    solvers, frees = _counting_frees(monkeypatch)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    lease = linalg.leased_factorization(A, MatrixClass.SPD, backend=backend)
    other = backend.factorize(sparse.eye(4, format="csr"), MatrixClass.SPD)
    backend.release_pattern_slots()
    assert backend.retained_pattern_slots == 0 and backend.detached_pattern_slots == 1
    assert frees[id(solvers[1])] == 1 and frees[id(solvers[0])] == 0
    with pytest.raises(linalg.PardisoStaleHandleError):
        other.solve(np.ones(4))
    np.testing.assert_array_equal(lease.handle.solve(np.ones(3)), _DIAG123_SOLUTION)
    # A same-pattern factorization never matches the detached slot.
    fresh = backend.factorize((A * 2.0).tocsr(), MatrixClass.SPD)
    assert fresh.metadata["pardiso_symbolic_reused"] is False
    assert solvers[0].phases == [12, 33, 33]  # rule-A probe solve, then the held solve
    lease.close()
    lease.close()  # idempotent
    assert frees[id(solvers[0])] == 1
    assert backend.detached_pattern_slots == 0
    with pytest.raises(linalg.PardisoStaleHandleError):
        lease.handle.solve(np.ones(3))
    assert lease.handle.metadata["pardiso_stale_cause"] == "released"
    backend.release_pattern_slots()
    assert all(count == 1 for count in frees.values())


def test_dropped_lease_is_ended_by_its_finalizer(monkeypatch) -> None:
    solvers, frees = _counting_frees(monkeypatch)
    backend = PyPardisoSolverBackend(max_pattern_slots=1)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    lease = linalg.leased_factorization(A, MatrixClass.SPD, backend=backend)
    backend.release_pattern_slots()
    assert frees[id(solvers[0])] == 0
    del lease
    gc.collect()
    assert frees[id(solvers[0])] == 1 and backend.leased_pattern_slots == 0


def test_superlu_and_failed_handles_are_never_pinned(monkeypatch) -> None:
    lease = linalg.leased_factorization(sparse.eye(3, format="csr"), MatrixClass.SPD,
                                        backend=linalg.SparseSolverBackend())
    assert lease.handle.backend_name == "scipy_superlu" and not lease.pinned
    lease.close()
    _install_fake(monkeypatch, lambda matrix: 1)  # every factorization perturbed
    refused = linalg.leased_factorization(sparse.csr_matrix(np.diag([1.0, 2.0, 3.0])),
                                          MatrixClass.SYMMETRIC_INDEFINITE,
                                          backend=PyPardisoSolverBackend())
    assert refused.handle.status == "failed" and not refused.pinned
    with pytest.raises(RuntimeError, match="Cannot create inverse operator"):
        linalg.leased_inverse_operator(sparse.csr_matrix(np.diag([1.0, 2.0, 3.0])),
                                       MatrixClass.SYMMETRIC_INDEFINITE,
                                       cache=linalg.FactorizationCache(backend=PyPardisoSolverBackend()))


# --------------------------------------------------------------------------
# 7. Native ownership under fault injection.
# --------------------------------------------------------------------------


def test_keyboard_interrupt_in_phase12_frees_once_and_propagates(monkeypatch) -> None:
    """7."""

    interrupt = KeyboardInterrupt()

    @_by_phase
    def interrupted(matrix, phase):
        del matrix, phase
        return 0, interrupt

    solvers, frees = _counting_frees(monkeypatch, interrupted)
    backend = PyPardisoSolverBackend()
    with pytest.raises(KeyboardInterrupt) as raised:
        backend.factorize(sparse.csr_matrix(np.diag([1.0, 2.0, 3.0])), MatrixClass.GENERAL)
    assert raised.value is interrupt
    assert frees[id(solvers[0])] == 1 and backend.retained_pattern_slots == 0
    gc.collect()
    assert frees[id(solvers[0])] == 1


@pytest.mark.parametrize("warm", [False, True])
def test_system_exit_on_the_iparm_read_frees_once_and_propagates(monkeypatch, warm) -> None:
    """7 (Codex r5): phase -1 was skipped and SystemExit escaped without release."""

    solvers, frees = _counting_frees(monkeypatch)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    if warm:
        assert backend.factorize(A, MatrixClass.GENERAL).status == "ok"
    exit_ = SystemExit(7)
    real_count = linalg._pardiso_perturbed_pivot_count
    state = {"armed": True}

    def exploding_count(solver):
        if state["armed"]:
            state["armed"] = False
            raise exit_
        return real_count(solver)

    monkeypatch.setattr(linalg, "_pardiso_perturbed_pivot_count", exploding_count)
    with pytest.raises(SystemExit) as raised:
        backend.factorize((A * 2.0).tocsr(), MatrixClass.GENERAL)
    assert raised.value is exit_
    assert backend.retained_pattern_slots == 0
    assert [frees[id(s)] for s in solvers] == [1] * len(solvers)
    gc.collect()
    assert [frees[id(s)] for s in solvers] == [1] * len(solvers)


def test_raising_free_memory_is_not_reported_as_released(monkeypatch) -> None:
    """7 (Codex r5): the interrupt note claimed 'released' after a raising free_memory."""

    interrupt = KeyboardInterrupt()

    @_by_phase
    def interrupted(matrix, phase):
        del matrix, phase
        return 0, interrupt

    solvers, frees = _counting_frees(monkeypatch, interrupted)
    factory = linalg._new_pypardiso_solver

    def raising_factory(*, mtype: int):
        solver = factory(mtype=mtype)
        counted = solver.free_memory

        def free_memory(*, everything: bool):
            counted(everything=everything)
            raise RuntimeError("simulated phase -1 failure")

        solver.free_memory = free_memory
        return solver

    monkeypatch.setattr(linalg, "_new_pypardiso_solver", raising_factory)
    with pytest.raises(KeyboardInterrupt) as raised:
        PyPardisoSolverBackend().factorize(sparse.csr_matrix(np.diag([1.0, 2.0])), MatrixClass.GENERAL)
    assert raised.value is interrupt
    notes = " ".join(getattr(raised.value, "__notes__", []))
    assert "releasing its native memory failed" in notes and "was released" not in notes
    gc.collect()
    assert frees[id(solvers[0])] == 1


def test_memory_error_in_slot_construction_frees_once(monkeypatch) -> None:
    """7 (Codex r5): the solver leaked when the slot constructor raised MemoryError."""

    solvers, frees = _counting_frees(monkeypatch)
    error = MemoryError()

    class ExplodingSlot(linalg._PardisoPatternSlot):
        __slots__ = ()

        def __init__(self, *args, **kwargs):
            raise error

    monkeypatch.setattr(linalg, "_PardisoPatternSlot", ExplodingSlot)
    backend = PyPardisoSolverBackend()
    handle = backend.factorize(sparse.csr_matrix(np.diag([1.0, 2.0])), MatrixClass.SPD)
    # MemoryError is an Exception: the backend reports a failed factorization
    # (each mtype attempt failed) and frees every instance exactly once.
    assert handle.status == "failed" and "mtype=2" in str(handle.failure_reason)
    assert backend.retained_pattern_slots == 0
    assert len(solvers) >= 1 and [frees[id(s)] for s in solvers] == [1] * len(solvers)
    gc.collect()
    assert [frees[id(s)] for s in solvers] == [1] * len(solvers)


# --------------------------------------------------------------------------
# 8. Refused factorizations are never leasable; a pinned slot is unaffected.
# --------------------------------------------------------------------------


def test_refused_newcomer_is_not_leasable_and_leaves_the_leased_slot_intact(monkeypatch) -> None:
    """8 (perturbed pivots; the cert-A probe joins after the A+B rebase)."""

    state = {"report": 0}
    solvers, frees = _counting_frees(monkeypatch, lambda matrix: state["report"])
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    cache = linalg.FactorizationCache(name="lease_refused", backend=backend)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    with linalg.leased_factorization(A, MatrixClass.GENERAL, cache=cache) as held:
        state["report"] = 2
        refused = linalg.leased_factorization((A * 2.0).tocsr(), MatrixClass.GENERAL, cache=cache)
        assert refused.handle.status == "failed" and not refused.pinned
        assert refused.handle.metadata["pardiso_failure_kind"] == linalg.PARDISO_PERTURBED_PIVOTS
        assert cache.diagnostics()["entries"] == 1  # not cached
        assert frees[id(solvers[1])] == 1  # the refused instance is freed
        assert held.handle.status == "ok"
        np.testing.assert_array_equal(held.handle.solve(np.ones(3)), _DIAG123_SOLUTION)
        assert solvers[0].phases == [12, 33, 33]  # rule-A probe solve, then the held solve
    assert frees[id(solvers[0])] == 1  # detached by the conflict, freed at lease end
    assert _leased(backend) == 0


def test_probe_refused_newcomer_is_not_leasable_and_leaves_the_leased_slot_intact(monkeypatch) -> None:
    """8 with rule A: a newcomer refused by the known-solution check is never
    leased, cached or retained, and the pinned slot keeps solving correctly."""

    state = {"garbage": False}

    def report(matrix) -> int:
        del matrix
        return 0

    report.garbage_solve = lambda: bool(state["garbage"])
    solvers, frees = _counting_frees(monkeypatch, report)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    cache = linalg.FactorizationCache(name="lease_probe_refused", backend=backend)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    with linalg.leased_factorization(A, MatrixClass.GENERAL, cache=cache) as held:
        state["garbage"] = True  # every solve is now wrong: the newcomer's probe fails
        refused = linalg.leased_factorization((A * 2.0).tocsr(), MatrixClass.GENERAL, cache=cache)
        assert refused.handle.status == "failed" and not refused.pinned
        assert refused.handle.metadata["pardiso_failure_kind"] == linalg.FORWARD_ERROR_PROBE_EXCEEDED
        assert refused.handle._solver is None
        assert cache.diagnostics()["entries"] == 1  # not cached
        assert frees[id(solvers[1])] == 1  # the refused instance is freed
        state["garbage"] = False
        assert held.handle.status == "ok"
        np.testing.assert_array_equal(held.handle.solve(np.ones(3)), _DIAG123_SOLUTION)
        assert solvers[0].phases == [12, 33, 33]  # never refactorized by the newcomer
    assert frees[id(solvers[0])] == 1
    assert _leased(backend) == 0


# --------------------------------------------------------------------------
# 9. Codex round-1 repairs (2026-11-25).
# --------------------------------------------------------------------------


def _unsorted_general(n: int = 160, seed: int = 7) -> sparse.csr_matrix:
    """Diagonally dominant GENERAL matrix whose rows list their columns in reverse."""

    A = (sparse.random(n, n, density=0.04, random_state=seed) + sparse.eye(n) * 4.0).tocsr()
    A.sort_indices()
    indices, data = A.indices.copy(), A.data.copy()
    for row in range(n):
        start, stop = A.indptr[row], A.indptr[row + 1]
        indices[start:stop] = indices[start:stop][::-1]
        data[start:stop] = data[start:stop][::-1]
    unsorted = sparse.csr_matrix((data, indices, A.indptr.copy()), shape=A.shape)
    unsorted.has_sorted_indices = False
    return unsorted


@requires_pypardiso
def test_leased_general_handle_keeps_its_values_after_in_place_caller_edits() -> None:
    """Codex r1 #1: on 024e8a4c4 the handle aliased (and re-sorted) the caller's CSR arrays."""

    rng = np.random.default_rng(3)
    A = _unsorted_general()
    original = A.copy()
    b = rng.random(A.shape[0])
    expected = np.linalg.solve(original.toarray(), b)
    backend = PyPardisoSolverBackend(max_pattern_slots=2)
    try:
        with linalg.leased_factorization(A, MatrixClass.GENERAL, backend=backend) as lease:
            assert lease.pinned
            # The caller's arrays are never reordered by the factorization.
            assert np.array_equal(A.indices, original.indices)
            assert np.array_equal(A.data, original.data)
            x_before = lease.handle.solve(b)
            np.testing.assert_allclose(x_before, expected, rtol=1.0e-10, atol=1.0e-12)
            # In-place value edit, then a same-pattern factorization (a conflict).
            A.data *= -2.5
            A.data[::7] += 1.0
            newcomer = backend.factorize(A, MatrixClass.GENERAL)
            assert newcomer.status == "ok" and newcomer.metadata["pardiso_lease_conflict"] is True
            np.testing.assert_allclose(A @ newcomer.solve(b), b, atol=1.0e-9)
            assert lease.handle.solve(b).tobytes() == x_before.tobytes()
            # In-place index edit (same nnz, another pattern), then a factorization.
            row0 = slice(A.indptr[0], A.indptr[1])
            A.indices[row0] = np.roll(A.indices[row0], 1)
            A.has_sorted_indices = False
            assert backend.factorize(A, MatrixClass.GENERAL).status == "ok"
            assert lease.handle.status == "ok" and not linalg.factorization_is_stale(lease.handle)
            assert lease.handle.solve(b).tobytes() == x_before.tobytes()
        assert _leased(backend) == 0
    finally:
        backend.release_pattern_slots()


def _rebound_with_status(disturb: bool):
    import anyfesolver as fs
    from anyfesolver.contact import _verification_contact_panel

    calls = {"n": 0}

    def status(_message: str) -> None:
        calls["n"] += 1
        if disturb:
            # Alternate two patterns: a warm 1x1 pattern would take the warm
            # auto thresholds (SuperLU) and evict nothing.
            size = 1 + calls["n"] % 2
            handle = linalg.factorize(sparse.diags(np.arange(1.0, size + 1.0), format="csr"), MatrixClass.SPD)
            assert handle.status == "ok" and handle.backend_name == "pypardiso"

    result = fs.solve_transient_sphere_impact(
        _verification_contact_panel(),
        fs.TransientConfig(dt=0.0025, t_end=0.04),
        fs.RigidSphereImpact("lease_reuse", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.12),
                             travel_direction=(0.0, 0.0, -1.0), speed=2.0),
        fs.SphereContactConfig(penalty_stiffness=4000.0, max_contact_iterations=40),
        nonlinear_config=fs.NonlinearTransientConfig(
            enabled=True, max_iterations=15, max_cutbacks=4, tangent_reuse_iterations=2,
        ),
        status_callback=status,
    )
    return result, calls["n"]


@requires_pypardiso
def test_nonlinear_impact_reused_tangent_survives_a_status_callback_factorization(monkeypatch) -> None:
    """Audit (Codex r1 #2): on 024e8a4c4 the reused tangent went stale (nonlinear_tangent_failed)."""

    backend = _forced_one_slot_default(monkeypatch)
    try:
        backend.release_pattern_slots()
        expected, _ = _rebound_with_status(False)
        backend.release_pattern_slots()
        before = _counts()
        observed, calls = _rebound_with_status(True)
        assert calls >= 3
        assert expected.status == "completed"
        assert expected.diagnostics["factorization_reuse_count"] > 0
        assert observed.status == expected.status
        for key in ("factorization_reuse_count", "factorization_count", "iteration_counts", "solve_count"):
            assert observed.diagnostics[key] == expected.diagnostics[key], key
        assert _delta(before, "pardiso_lease_conflicts") == 0
        assert _delta(before, "pardiso_lease_detachments") >= 1  # the pin was exercised
        # Not bitwise: every disturbance evicts the tangent's warm slot, so the
        # next refresh is a fresh phase 12 where the undisturbed run reuses the
        # analysis (phase 22); the held tangent itself is never refactorized.
        np.testing.assert_allclose(observed.displacements, expected.displacements, rtol=1.0e-12, atol=1.0e-13)
        np.testing.assert_allclose(observed.velocities, expected.velocities, rtol=1.0e-12, atol=1.0e-13)
        assert _leased(backend.pardiso_backend) == 0  # released on exit
    finally:
        backend.release_pattern_slots()


def _racing_auto(monkeypatch, *, policy: str):
    """Auto backend whose PARDISO handle is invalidated right after it is returned."""

    _install_fake(monkeypatch, lambda matrix: 0)
    pardiso = PyPardisoSolverBackend(max_pattern_slots=1)
    auto = AutoSparseSolverBackend(pardiso_backend=pardiso, pypardiso_min_dimension=1, pypardiso_min_nnz=1,
                                   pypardiso_warm_min_dimension=1, pypardiso_warm_min_nnz=1)
    real = pardiso.factorize
    state = {"armed": 0}

    def racing(matrix, matrix_class, **kwargs):
        handle = real(matrix, matrix_class, **kwargs)
        if state["armed"] > 0:
            state["armed"] -= 1
            # Another thread refactorizes the same pattern before auto reads the status.
            assert real((matrix * 2.0).tocsr(), matrix_class).status == "ok"
        return handle

    monkeypatch.setattr(pardiso, "factorize", racing)
    options = {"backend": "pypardiso"} if policy == "forced" else {}
    return auto, state, options


@pytest.mark.parametrize("policy", ["forced", "auto"])
def test_concurrent_invalidation_never_switches_a_pardiso_factorization_to_superlu(monkeypatch, policy) -> None:
    """Codex r1 #3: on 024e8a4c4 a stale-at-return handle fell back to SuperLU silently."""

    auto, state, options = _racing_auto(monkeypatch, policy=policy)
    A = sparse.csr_matrix(np.diag([1.0, 2.0, 3.0]))

    state["armed"] = 1
    handle = auto.factorize(A, MatrixClass.GENERAL, options=options)
    assert handle.backend_name == "pypardiso"
    assert handle.metadata["pardiso_failure_kind"] == linalg.PARDISO_STALE_HANDLE
    assert "fallback_from_backend" not in handle.metadata
    with pytest.raises(linalg.PardisoStaleHandleError):
        handle.solve(np.ones(3))

    # A lease refactorizes once on the same backend and pins the result.
    state["armed"] = 1
    with linalg.leased_factorization(A, MatrixClass.GENERAL, backend=auto, options=options) as lease:
        assert lease.pinned and lease.handle.backend_name == "pypardiso"
        np.testing.assert_array_equal(lease.handle.solve(np.ones(3)), _DIAG123_SOLUTION)

    # A cache treats the stale result as a miss, never as a failure or an entry.
    cache = linalg.FactorizationCache(name="lease_race", backend=auto)
    state["armed"] = 1
    stale = cache.factorize(A, MatrixClass.GENERAL, options=options)
    assert stale.backend_name == "pypardiso" and linalg.factorization_is_stale(stale)
    diagnostics = cache.diagnostics()
    assert diagnostics["factorization_failures"] == 0 and diagnostics["entries"] == 0
    fresh = cache.factorize(A, MatrixClass.GENERAL, options=options)
    assert fresh.status == "ok" and fresh.backend_name == "pypardiso"


def _raise_on_free(solver, interrupt) -> None:
    """Make ``solver``'s phase -1 free its memory, then raise ``interrupt``."""

    original = solver.free_memory

    def free_memory(*, everything: bool):
        original(everything=everything)
        raise interrupt

    solver.free_memory = free_memory


def _interrupting_factory(monkeypatch, interrupt) -> None:
    factory = linalg._new_pypardiso_solver

    def raising_factory(*, mtype: int):
        solver = factory(mtype=mtype)
        _raise_on_free(solver, interrupt)
        return solver

    monkeypatch.setattr(linalg, "_new_pypardiso_solver", raising_factory)


def test_release_interrupt_inside_an_unrelated_except_block_propagates(monkeypatch) -> None:
    """Codex r1 #4: on 024e8a4c4 the interrupt became a note on the handled ValueError."""

    solvers = _install_fake(monkeypatch, lambda matrix: 0)
    interrupt = KeyboardInterrupt()
    backend = PyPardisoSolverBackend()
    assert backend.factorize(sparse.csr_matrix(np.diag([1.0, 2.0])), MatrixClass.SPD).status == "ok"
    _raise_on_free(solvers[0], interrupt)
    with pytest.raises(KeyboardInterrupt) as raised:
        try:
            raise ValueError("recoverable")
        except ValueError:
            backend.release_pattern_slots()
    assert raised.value is interrupt
    assert solvers[0].freed and backend.retained_pattern_slots == 0


def test_failed_factorization_cleanup_interrupt_is_not_swallowed(monkeypatch) -> None:
    """Codex r1 #4 (same class): on 024e8a4c4 the interrupt was a note on an error the mtype loop swallowed."""

    interrupt = KeyboardInterrupt()

    @_by_phase
    def failing(matrix, phase):
        del matrix, phase
        return 0, RuntimeError("simulated PARDISO error")

    solvers = _install_fake(monkeypatch, failing)
    _interrupting_factory(monkeypatch, interrupt)
    with pytest.raises(KeyboardInterrupt) as raised:
        PyPardisoSolverBackend().factorize(sparse.csr_matrix(np.diag([1.0, 2.0])), MatrixClass.GENERAL)
    assert raised.value is interrupt
    assert isinstance(raised.value.__context__, RuntimeError)
    assert solvers[0].freed


def test_lease_end_interrupt_is_never_left_as_a_note_on_an_ordinary_exception(monkeypatch) -> None:
    """Codex r1 #4 (same class): an interrupt at lease end replaces an ordinary exception.

    An interrupt that is already propagating is kept (the cleanup interrupt is a note).
    """

    solvers = _install_fake(monkeypatch, lambda matrix: 0)
    backend = PyPardisoSolverBackend()
    lease = linalg.leased_factorization(sparse.csr_matrix(np.diag([1.0, 2.0])), MatrixClass.SPD,
                                        backend=backend)
    other = linalg.leased_factorization(sparse.csr_matrix(np.diag([1.0, 2.0, 3.0])), MatrixClass.SPD,
                                        backend=backend)
    backend.release_pattern_slots()  # both detached: freed at lease end
    assert backend.detached_pattern_slots == 2
    interrupt = KeyboardInterrupt()
    _raise_on_free(solvers[0], interrupt)
    with pytest.raises(KeyboardInterrupt) as raised:
        with lease:
            raise ValueError("recoverable")
    assert raised.value is interrupt
    assert isinstance(raised.value.__context__, ValueError)

    _raise_on_free(solvers[1], KeyboardInterrupt())
    primary = SystemExit(3)
    with pytest.raises(SystemExit) as kept:
        with other:
            raise primary
    assert kept.value is primary
    assert "KeyboardInterrupt" in " ".join(getattr(primary, "__notes__", []))
    assert solvers[0].freed and solvers[1].freed and backend.detached_pattern_slots == 0
